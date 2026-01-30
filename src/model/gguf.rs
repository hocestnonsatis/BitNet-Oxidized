//! GGUF format reader/writer for BitNet models.
//!
//! Follows llama.cpp GGUF v3 layout: header, metadata KV, tensor infos, aligned tensor data.

use crate::errors::BitNetError;
use crate::kernels::TernaryTensor;
use crate::model::{BitNetConfig, BitNetLayer, BitNetModel};
use crate::quantization::absmax_quantize;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// GGUF magic: "GGUF" = 0x46554747
const GGUF_MAGIC: u32 = 0x4655_4747;
/// GGUF version 3
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: u64 = 32;

/// Value types in metadata
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// Tensor storage type. F32 = 0, I2_S (2-bit BitNet) = 40 (custom).
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
enum GGUFTensorType {
    F32 = 0,
    F16 = 1,
    /// BitNet 2-bit: 4 weights per byte
    I2_S = 40,
}

/// Header: magic, version, tensor_count, metadata_kv_count
struct GGUFHeader {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

impl GGUFHeader {
    fn read<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            magic: r.read_u32::<LittleEndian>()?,
            version: r.read_u32::<LittleEndian>()?,
            tensor_count: r.read_u64::<LittleEndian>()?,
            metadata_kv_count: r.read_u64::<LittleEndian>()?,
        })
    }
    fn write<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.magic)?;
        w.write_u32::<LittleEndian>(self.version)?;
        w.write_u64::<LittleEndian>(self.tensor_count)?;
        w.write_u64::<LittleEndian>(self.metadata_kv_count)?;
        Ok(())
    }
}

/// Tensor info: name, dimensions, type, file offset
struct GGUFTensorInfo {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

fn read_string<R: Read>(r: &mut R) -> std::io::Result<String> {
    let len = r.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid utf8"))
}

fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    let b = s.as_bytes();
    w.write_u64::<LittleEndian>(b.len() as u64)?;
    w.write_all(b)?;
    Ok(())
}

/// Read metadata section into key-value map (values as raw bytes or parsed for known keys).
fn read_metadata<R: Read>(r: &mut R, kv_count: u64) -> std::io::Result<HashMap<String, GGUFValue>> {
    let mut meta = HashMap::new();
    for _ in 0..kv_count {
        let key = read_string(r)?;
        let value_type = r.read_u32::<LittleEndian>()?;
        let val = read_gguf_value(r, value_type)?;
        meta.insert(key, val);
    }
    Ok(meta)
}

#[derive(Clone)]
#[allow(dead_code)]
enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

fn read_gguf_value<R: Read>(r: &mut R, value_type: u32) -> std::io::Result<GGUFValue> {
    use GGUFValueType as T;
    match value_type {
        x if x == T::UInt8 as u32 => Ok(GGUFValue::UInt8(r.read_u8()?)),
        x if x == T::Int8 as u32 => Ok(GGUFValue::Int8(r.read_i8()?)),
        x if x == T::UInt16 as u32 => Ok(GGUFValue::UInt16(r.read_u16::<LittleEndian>()?)),
        x if x == T::Int16 as u32 => Ok(GGUFValue::Int16(r.read_i16::<LittleEndian>()?)),
        x if x == T::UInt32 as u32 => Ok(GGUFValue::UInt32(r.read_u32::<LittleEndian>()?)),
        x if x == T::Int32 as u32 => Ok(GGUFValue::Int32(r.read_i32::<LittleEndian>()?)),
        x if x == T::Float32 as u32 => Ok(GGUFValue::Float32(r.read_f32::<LittleEndian>()?)),
        x if x == T::Bool as u32 => Ok(GGUFValue::Bool(r.read_u8()? != 0)),
        x if x == T::String as u32 => Ok(GGUFValue::String(read_string(r)?)),
        x if x == T::UInt64 as u32 => Ok(GGUFValue::UInt64(r.read_u64::<LittleEndian>()?)),
        x if x == T::Int64 as u32 => Ok(GGUFValue::Int64(r.read_i64::<LittleEndian>()?)),
        x if x == T::Float64 as u32 => Ok(GGUFValue::Float64(r.read_f64::<LittleEndian>()?)),
        x if x == T::Array as u32 => {
            let elem_type = r.read_u32::<LittleEndian>()?;
            let len = r.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_gguf_value(r, elem_type)?);
            }
            Ok(GGUFValue::Array(arr))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unknown value type",
        )),
    }
}

fn extract_u64(v: &GGUFValue) -> Option<u64> {
    match v {
        GGUFValue::UInt32(x) => Some(*x as u64),
        GGUFValue::UInt64(x) => Some(*x),
        GGUFValue::Int32(x) => Some(*x as u64),
        _ => None,
    }
}

fn extract_f32(v: &GGUFValue) -> Option<f32> {
    match v {
        GGUFValue::Float32(x) => Some(*x),
        _ => None,
    }
}

/// Extract BitNetConfig from GGUF metadata (bitnet.* and general.architecture).
fn config_from_metadata(meta: &HashMap<String, GGUFValue>) -> Result<BitNetConfig, BitNetError> {
    let arch = meta.get("general.architecture").and_then(|v| {
        if let GGUFValue::String(s) = v {
            Some(s.as_str())
        } else {
            None
        }
    });
    if arch != Some("bitnet") {
        return Err(BitNetError::InvalidFormat(format!(
            "expected general.architecture bitnet, got {:?}",
            arch
        )));
    }

    let vocab_size = meta
        .get("bitnet.vocab_size")
        .and_then(extract_u64)
        .ok_or_else(|| BitNetError::InvalidFormat("bitnet.vocab_size missing".into()))?
        as usize;
    let context_length = meta
        .get("bitnet.context_length")
        .and_then(extract_u64)
        .unwrap_or(2048) as usize;
    let embedding_length = meta
        .get("bitnet.embedding_length")
        .and_then(extract_u64)
        .ok_or_else(|| BitNetError::InvalidFormat("bitnet.embedding_length missing".into()))?
        as usize;
    let block_count = meta
        .get("bitnet.block_count")
        .and_then(extract_u64)
        .ok_or_else(|| BitNetError::InvalidFormat("bitnet.block_count missing".into()))?
        as usize;
    let feed_forward_length = meta
        .get("bitnet.feed_forward_length")
        .and_then(extract_u64)
        .ok_or_else(|| BitNetError::InvalidFormat("bitnet.feed_forward_length missing".into()))?
        as usize;
    let head_count = meta
        .get("bitnet.attention.head_count")
        .and_then(extract_u64)
        .unwrap_or((embedding_length / 64) as u64) as usize;
    let key_value_head_count = meta
        .get("bitnet.attention.key_value_head_count")
        .and_then(extract_u64)
        .unwrap_or(head_count as u64) as usize;
    let rms_eps = meta
        .get("bitnet.attention.layer_norm_rms_epsilon")
        .and_then(extract_f32)
        .unwrap_or(1e-6);

    Ok(BitNetConfig {
        vocab_size,
        hidden_size: embedding_length,
        num_attention_heads: head_count,
        num_key_value_heads: key_value_head_count,
        num_hidden_layers: block_count,
        intermediate_size: feed_forward_length,
        max_position_embeddings: context_length,
        rms_norm_eps: rms_eps,
    })
}

/// Read tensor info list
fn read_tensor_infos<R: Read>(
    r: &mut R,
    tensor_count: u64,
) -> std::io::Result<Vec<GGUFTensorInfo>> {
    let mut infos = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(r)?;
        let n_dims = r.read_u32::<LittleEndian>()? as usize;
        let mut dimensions = vec![0u64; n_dims];
        for d in &mut dimensions {
            *d = r.read_u64::<LittleEndian>()?;
        }
        let tensor_type = r.read_u32::<LittleEndian>()?;
        let offset = r.read_u64::<LittleEndian>()?;
        infos.push(GGUFTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }
    Ok(infos)
}

/// Load tensor data: for F32 read floats, for I2_S read raw bytes and build TernaryTensor
fn load_tensor_data(data: &[u8], info: &GGUFTensorInfo) -> Result<Vec<u8>, BitNetError> {
    let start = info.offset as usize;
    let n_elems: usize = info.dimensions.iter().product::<u64>() as usize;
    let (n_bytes, _) = tensor_type_size(info.tensor_type, n_elems);
    let end = start + n_bytes;
    if end > data.len() {
        return Err(BitNetError::InvalidFormat(format!(
            "tensor {} out of bounds",
            info.name
        )));
    }
    Ok(data[start..end].to_vec())
}

fn tensor_type_size(tensor_type: u32, n_elems: usize) -> (usize, bool) {
    match tensor_type {
        x if x == GGUFTensorType::F32 as u32 => (n_elems * 4, true),
        x if x == GGUFTensorType::I2_S as u32 => (n_elems.div_ceil(4), false),
        _ => (0, false),
    }
}

/// Load a BitNet model from a GGUF file.
pub fn load_gguf(path: impl AsRef<Path>) -> Result<BitNetModel, BitNetError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(BitNetError::Io)?;
    let header = GGUFHeader::read(&mut file).map_err(BitNetError::Io)?;

    if header.magic != GGUF_MAGIC {
        return Err(BitNetError::InvalidFormat("invalid GGUF magic".into()));
    }
    if header.version != GGUF_VERSION {
        return Err(BitNetError::UnsupportedGGUFVersion(header.version));
    }

    let meta = read_metadata(&mut file, header.metadata_kv_count).map_err(BitNetError::Io)?;
    let config = config_from_metadata(&meta)?;

    let tensor_infos =
        read_tensor_infos(&mut file, header.tensor_count).map_err(BitNetError::Io)?;

    let pos = file.stream_position().map_err(BitNetError::Io)?;
    let padding = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
    file.seek(SeekFrom::Current(padding as i64))
        .map_err(BitNetError::Io)?;
    let _data_base = file.stream_position().map_err(BitNetError::Io)?;

    let mut all_data = Vec::new();
    file.read_to_end(&mut all_data).map_err(BitNetError::Io)?;

    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let num_layers = config.num_hidden_layers;

    let mut name_to_data: HashMap<String, Vec<u8>> = HashMap::new();
    for info in &tensor_infos {
        let bytes = load_tensor_data(&all_data, info)?;
        name_to_data.insert(info.name.clone(), bytes);
    }

    let get_f32 = |name: &str| -> Result<Vec<f32>, BitNetError> {
        let data = name_to_data
            .get(name)
            .ok_or_else(|| BitNetError::InvalidFormat(format!("missing tensor {}", name)))?;
        let n = data.len() / 4;
        let mut out = vec![0f32; n];
        for (i, chunk) in data.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out)
    };

    // Load a weight tensor as TernaryTensor: I2_S from raw bytes, F32 by quantizing on load.
    let get_ternary = |name: &str| -> Result<TernaryTensor, BitNetError> {
        let data = name_to_data
            .get(name)
            .ok_or_else(|| BitNetError::InvalidFormat(format!("missing tensor {}", name)))?;
        let info = tensor_infos
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| BitNetError::InvalidFormat(format!("missing tensor info {}", name)))?;
        let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        let len: usize = dims.iter().product();
        if info.tensor_type == GGUFTensorType::I2_S as u32 {
            TernaryTensor::from_raw(data.clone(), len)
                .map_err(|e| BitNetError::InvalidFormat(format!("tensor {}: {:?}", name, e)))
        } else if info.tensor_type == GGUFTensorType::F32 as u32 {
            let mut weights = vec![0f32; len];
            for (i, chunk) in data.chunks_exact(4).take(len).enumerate() {
                weights[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            let (ternary, _scale) = absmax_quantize(&weights, dims);
            Ok(ternary)
        } else {
            Err(BitNetError::InvalidFormat(format!(
                "tensor {}: unsupported type {} (use F32 or I2_S)",
                name, info.tensor_type
            )))
        }
    };

    let embeddings_raw = get_f32("token_embd")?;
    let embeddings: Vec<Vec<f32>> = embeddings_raw
        .chunks(hidden)
        .take(vocab)
        .map(|c| c.to_vec())
        .collect();
    if embeddings.len() < vocab {
        return Err(BitNetError::DimensionMismatch {
            expected: vocab,
            actual: embeddings.len(),
        });
    }

    let norm = get_f32("output_norm")?;
    if norm.len() != hidden {
        return Err(BitNetError::DimensionMismatch {
            expected: hidden,
            actual: norm.len(),
        });
    }

    let lm_head = get_ternary("output")?;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("blk.{}.", i);
        let input_layernorm = get_f32(&format!("{}attn_norm", prefix))?;
        let post_attention_layernorm = get_f32(&format!("{}ffn_norm", prefix))?;
        if input_layernorm.len() != hidden || post_attention_layernorm.len() != hidden {
            return Err(BitNetError::DimensionMismatch {
                expected: hidden,
                actual: input_layernorm.len().max(post_attention_layernorm.len()),
            });
        }
        layers.push(BitNetLayer {
            q_proj: get_ternary(&format!("{}attn_q", prefix))?,
            k_proj: get_ternary(&format!("{}attn_k", prefix))?,
            v_proj: get_ternary(&format!("{}attn_v", prefix))?,
            o_proj: get_ternary(&format!("{}attn_output", prefix))?,
            gate_proj: get_ternary(&format!("{}ffn_gate", prefix))?,
            up_proj: get_ternary(&format!("{}ffn_up", prefix))?,
            down_proj: get_ternary(&format!("{}ffn_down", prefix))?,
            input_layernorm,
            post_attention_layernorm,
        });
    }

    Ok(BitNetModel {
        config,
        embeddings,
        layers,
        norm,
        lm_head,
    })
}

/// Write a single metadata KV
fn write_metadata_kv<W: Write>(w: &mut W, key: &str, value: &GGUFValue) -> std::io::Result<()> {
    write_string(w, key)?;
    match value {
        GGUFValue::UInt32(v) => {
            w.write_u32::<LittleEndian>(GGUFValueType::UInt32 as u32)?;
            w.write_u32::<LittleEndian>(*v)?;
        }
        GGUFValue::UInt64(v) => {
            w.write_u32::<LittleEndian>(GGUFValueType::UInt64 as u32)?;
            w.write_u64::<LittleEndian>(*v)?;
        }
        GGUFValue::Float32(v) => {
            w.write_u32::<LittleEndian>(GGUFValueType::Float32 as u32)?;
            w.write_f32::<LittleEndian>(*v)?;
        }
        GGUFValue::String(s) => {
            w.write_u32::<LittleEndian>(GGUFValueType::String as u32)?;
            write_string(w, s)?;
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "unsupported value type",
            ))
        }
    }
    Ok(())
}

/// Save a BitNet model to GGUF.
pub fn save_gguf(model: &BitNetModel, path: impl AsRef<Path>) -> Result<(), BitNetError> {
    let path = path.as_ref();
    let mut file = File::create(path).map_err(BitNetError::Io)?;

    let config = &model.config;
    let num_layers = model.layers.len();

    let meta_kv: Vec<(String, GGUFValue)> = vec![
        (
            "general.architecture".into(),
            GGUFValue::String("bitnet".into()),
        ),
        (
            "bitnet.vocab_size".into(),
            GGUFValue::UInt32(config.vocab_size as u32),
        ),
        (
            "bitnet.context_length".into(),
            GGUFValue::UInt32(config.max_position_embeddings as u32),
        ),
        (
            "bitnet.embedding_length".into(),
            GGUFValue::UInt32(config.hidden_size as u32),
        ),
        (
            "bitnet.block_count".into(),
            GGUFValue::UInt32(num_layers as u32),
        ),
        (
            "bitnet.feed_forward_length".into(),
            GGUFValue::UInt32(config.intermediate_size as u32),
        ),
        (
            "bitnet.attention.head_count".into(),
            GGUFValue::UInt32(config.num_attention_heads as u32),
        ),
        (
            "bitnet.attention.key_value_head_count".into(),
            GGUFValue::UInt32(config.num_key_value_heads as u32),
        ),
        (
            "bitnet.attention.layer_norm_rms_epsilon".into(),
            GGUFValue::Float32(config.rms_norm_eps),
        ),
    ];

    let mut tensor_infos: Vec<(String, Vec<u64>, u32)> = Vec::new();

    // token_embd [vocab, hidden]
    tensor_infos.push((
        "token_embd".into(),
        vec![config.vocab_size as u64, config.hidden_size as u64],
        GGUFTensorType::F32 as u32,
    ));
    // output_norm [hidden]
    tensor_infos.push((
        "output_norm".into(),
        vec![config.hidden_size as u64],
        GGUFTensorType::F32 as u32,
    ));
    // output (lm_head) [vocab, hidden]
    tensor_infos.push((
        "output".into(),
        vec![config.vocab_size as u64, config.hidden_size as u64],
        GGUFTensorType::I2_S as u32,
    ));

    for i in 0..num_layers {
        let prefix = format!("blk.{}.", i);
        tensor_infos.push((
            format!("{}attn_norm", prefix),
            vec![config.hidden_size as u64],
            GGUFTensorType::F32 as u32,
        ));
        tensor_infos.push((
            format!("{}attn_q", prefix),
            vec![config.hidden_size as u64, config.hidden_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
        tensor_infos.push((
            format!("{}attn_k", prefix),
            vec![config.hidden_size as u64, config.hidden_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
        tensor_infos.push((
            format!("{}attn_v", prefix),
            vec![config.hidden_size as u64, config.hidden_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
        tensor_infos.push((
            format!("{}attn_output", prefix),
            vec![config.hidden_size as u64, config.hidden_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
        tensor_infos.push((
            format!("{}ffn_norm", prefix),
            vec![config.hidden_size as u64],
            GGUFTensorType::F32 as u32,
        ));
        tensor_infos.push((
            format!("{}ffn_gate", prefix),
            vec![config.intermediate_size as u64, config.hidden_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
        tensor_infos.push((
            format!("{}ffn_up", prefix),
            vec![config.intermediate_size as u64, config.hidden_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
        tensor_infos.push((
            format!("{}ffn_down", prefix),
            vec![config.hidden_size as u64, config.intermediate_size as u64],
            GGUFTensorType::I2_S as u32,
        ));
    }

    let tensor_count = tensor_infos.len() as u64;
    let metadata_kv_count = meta_kv.len() as u64;

    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION,
        tensor_count,
        metadata_kv_count,
    };
    header.write(&mut file).map_err(BitNetError::Io)?;

    for (k, v) in &meta_kv {
        write_metadata_kv(&mut file, k, v).map_err(BitNetError::Io)?;
    }

    let mut data_offset: u64 = 0;
    for (name, dims, dtype) in &tensor_infos {
        write_string(&mut file, name).map_err(BitNetError::Io)?;
        file.write_u32::<LittleEndian>(dims.len() as u32)
            .map_err(BitNetError::Io)?;
        for &d in dims {
            file.write_u64::<LittleEndian>(d).map_err(BitNetError::Io)?;
        }
        file.write_u32::<LittleEndian>(*dtype)
            .map_err(BitNetError::Io)?;
        file.write_u64::<LittleEndian>(data_offset)
            .map_err(BitNetError::Io)?;
        let n_elems: usize = dims.iter().product::<u64>() as usize;
        let (n_bytes, _) = tensor_type_size(*dtype, n_elems);
        data_offset += n_bytes as u64;
    }

    let pos = file.stream_position().map_err(BitNetError::Io)?;
    let padding = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
    for _ in 0..padding {
        file.write_all(&[0u8]).map_err(BitNetError::Io)?;
    }

    // Write tensor data in same order

    for row in &model.embeddings {
        for &f in row {
            file.write_all(&f.to_le_bytes()).map_err(BitNetError::Io)?;
        }
    }
    for &f in &model.norm {
        file.write_all(&f.to_le_bytes()).map_err(BitNetError::Io)?;
    }
    file.write_all(model.lm_head.raw_data())
        .map_err(BitNetError::Io)?;

    for layer in &model.layers {
        for &f in &layer.input_layernorm {
            file.write_all(&f.to_le_bytes()).map_err(BitNetError::Io)?;
        }
        file.write_all(layer.q_proj.raw_data())
            .map_err(BitNetError::Io)?;
        file.write_all(layer.k_proj.raw_data())
            .map_err(BitNetError::Io)?;
        file.write_all(layer.v_proj.raw_data())
            .map_err(BitNetError::Io)?;
        file.write_all(layer.o_proj.raw_data())
            .map_err(BitNetError::Io)?;
        for &f in &layer.post_attention_layernorm {
            file.write_all(&f.to_le_bytes()).map_err(BitNetError::Io)?;
        }
        file.write_all(layer.gate_proj.raw_data())
            .map_err(BitNetError::Io)?;
        file.write_all(layer.up_proj.raw_data())
            .map_err(BitNetError::Io)?;
        file.write_all(layer.down_proj.raw_data())
            .map_err(BitNetError::Io)?;
    }

    Ok(())
}
