#!/usr/bin/env python3
"""
Convert HuggingFace BitNet/LLaMA safetensors to GGUF for bitnet-oxidized.

Expects a downloaded HF model dir with config.json, model.safetensors, (optional) tokenizer.json.
Writes a single .gguf file with bitnet-oxidized tensor names and I2_S (2-bit) ternary weights.

**Do not run on phones/Termux** — loads the full model into RAM and can crash the device.
Run on a desktop or server, then copy the .gguf file to the device. See docs/termux_and_phones.md.

Usage:
    python scripts/convert_huggingface_to_gguf.py \
        --input models/Llama3-8B-1.58-100B-tokens \
        --output models/Llama3-8B-1.58-100B-tokens.gguf

Requires: pip install safetensors numpy
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

# GGUF
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_TYPE_F32 = 0
GGUF_TYPE_I2_S = 40  # 2-bit BitNet

# Bit packing: 00=0, 01=+1, 11=-1 (match Rust TernaryWeight)
def float_to_ternary_bits(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return 3
    return 0


def pack_ternary(weights: np.ndarray) -> bytes:
    """Pack float array as ternary (-1,0,+1) into 2 bits per value, 4 per byte."""
    flat = weights.flatten()
    out = []
    for i in range(0, len(flat), 4):
        b = 0
        for j in range(4):
            if i + j < len(flat):
                v = float(flat[i + j])
                b |= (float_to_ternary_bits(v) << (j * 2))
        out.append(b)
    return bytes(out)


def quantize_to_ternary(weights: np.ndarray) -> np.ndarray:
    """Quantize float weights to -1, 0, +1 (threshold at 0)."""
    w = np.asarray(weights, dtype=np.float64).flatten()
    out = np.zeros_like(w)
    out[w > 0] = 1
    out[w < 0] = -1
    return out.astype(np.int8)


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def bfloat16_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw bfloat16 bytes to float32 (no torch needed)."""
    arr = np.frombuffer(raw, dtype=np.uint16)
    u32 = (arr.astype(np.uint32) << 16) & 0xFFFF0000
    return np.frombuffer(u32.tobytes(), dtype=np.float32)


def load_safetensors_raw(model_dir: Path) -> tuple[dict, list]:
    """Load safetensors file and return (name -> float32 numpy array), list of keys.
    Handles BF16 by reading raw bytes and converting. No torch required.
    """
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"No {st_path}")
    with open(st_path, "rb") as f:
        header = f.read(8)
        n = int.from_bytes(header[:8], "little")
        meta = json.loads(f.read(n).decode("utf-8"))
    # Safetensors: after header+json, data block starts (offsets in json are relative to it)
    data_start = 8 + (n + 7) // 8 * 8
    tensors = {}
    keys = [k for k in meta if isinstance(meta[k], dict) and "dtype" in meta[k]]
    with open(st_path, "rb") as f:
        f.seek(data_start)
        data = f.read()
    for key in keys:
        info = meta[key]
        dtype = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]  # relative to data block
        raw = data[start:end]
        n_elems = int(np.prod(shape))
        if dtype == "BF16":
            arr = bfloat16_to_float32(raw)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dtype == "U8":
            # BitNet 1.58: U8 often 0/1/2 -> zero/+1/-1; treat as float for later ternary pack
            u8 = np.frombuffer(raw, dtype=np.uint8)
            arr = np.where(u8 == 0, 0.0, np.where(u8 == 1, 1.0, -1.0)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        arr = arr[:n_elems].reshape(shape)
        tensors[key] = arr
    return tensors, keys


def write_gguf_header(f, tensor_count: int, metadata_kv_count: int):
    f.write(struct.pack("<I", GGUF_MAGIC))
    f.write(struct.pack("<I", GGUF_VERSION))
    f.write(struct.pack("<Q", tensor_count))
    f.write(struct.pack("<Q", metadata_kv_count))


def write_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_metadata(f, config: dict):
    # GGUF value types: UINT32=4, FLOAT32=6, STRING=8
    num_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    meta = [
        ("general.architecture", 8, "bitnet"),
        ("bitnet.vocab_size", 4, config["vocab_size"]),
        ("bitnet.context_length", 4, config.get("max_position_embeddings", 2048)),
        ("bitnet.embedding_length", 4, config["hidden_size"]),
        ("bitnet.block_count", 4, config["num_hidden_layers"]),
        ("bitnet.feed_forward_length", 4, config["intermediate_size"]),
        ("bitnet.attention.head_count", 4, config["num_attention_heads"]),
        ("bitnet.attention.key_value_head_count", 4, num_kv_heads),
        ("bitnet.attention.layer_norm_rms_epsilon", 6, float(config.get("rms_norm_eps", 1e-6))),
    ]
    for key, typ, val in meta:
        write_string(f, key)
        f.write(struct.pack("<I", typ))
        if typ == 4:
            f.write(struct.pack("<I", val))
        elif typ == 6:
            f.write(struct.pack("<f", val))
        elif typ == 8:
            write_string(f, str(val))
        else:
            raise ValueError(f"Unknown type {typ}")


def write_tensor_info(f, name: str, dims: list, tensor_type: int, offset: int):
    write_string(f, name)
    f.write(struct.pack("<I", len(dims)))
    for d in dims:
        f.write(struct.pack("<Q", d))
    f.write(struct.pack("<I", tensor_type))
    f.write(struct.pack("<Q", offset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to downloaded HF model dir")
    ap.add_argument("--output", required=True, help="Output .gguf path")
    args = ap.parse_args()

    model_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(model_dir)
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    intermediate_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    st_dict, keys = load_safetensors_raw(model_dir)

    def expand_to(w, expected_out: int, expected_in: int):
        """Repeat rows/cols so w matches (expected_out, expected_in)."""
        out, inp = w.shape[0], w.shape[1]
        if out == expected_out and inp == expected_in:
            return w
        if out < expected_out and inp == expected_in:
            return np.repeat(w, expected_out // out, axis=0)
        if out == expected_out and inp < expected_in:
            return np.repeat(w, expected_in // inp, axis=1)
        return w
    out_tensors = []  # list of (name, dims, dtype_str, raw_bytes)

    def get_w(key_pri, key_alt=None):
        w = st_dict.get(key_pri)
        if w is None and key_alt:
            w = st_dict.get(key_alt)
        if w is not None:
            w = np.asarray(w, dtype=np.float32)
        return w

    # Embeddings: token_embd [vocab, hidden] F32
    w = get_w("model.embed_tokens.weight", "embed_tokens.weight")
    if w is not None:
        out_tensors.append(("token_embd", [vocab_size, hidden_size], "f32", w.tobytes()))

    # output_norm [hidden] F32
    w = get_w("model.norm.weight", "norm.weight")
    if w is not None:
        out_tensors.append(("output_norm", [hidden_size], "f32", w.tobytes()))

    # output (lm_head) [vocab, hidden] I2_S
    # Tied embeddings: some models (e.g. bitnet_b1_58-large) have tie_word_embeddings and no lm_head
    w = get_w("lm_head.weight", "model.lm_head.weight")
    if w is None and config.get("tie_word_embeddings"):
        w = get_w("model.embed_tokens.weight", "embed_tokens.weight")
    if w is not None:
        packed = pack_ternary(quantize_to_ternary(w))
        out_tensors.append(("output", [vocab_size, hidden_size], "i2_s", packed))

    # Layers
    for i in range(num_layers):
        prefix = f"blk.{i}."
        w = get_w(f"model.layers.{i}.input_layernorm.weight", f"layers.{i}.input_layernorm.weight")
        if w is not None:
            out_tensors.append((f"{prefix}attn_norm", [hidden_size], "f32", w.tobytes()))
        for proj, short in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            w = get_w(f"model.layers.{i}.self_attn.{proj}.weight", f"layers.{i}.self_attn.{proj}.weight")
            if w is not None:
                w = expand_to(w, hidden_size, hidden_size)
                packed = pack_ternary(quantize_to_ternary(w))
                out_tensors.append((f"{prefix}{short}", list(w.shape), "i2_s", packed))
        w = get_w(f"model.layers.{i}.post_attention_layernorm.weight", f"layers.{i}.post_attention_layernorm.weight")
        if w is not None:
            out_tensors.append((f"{prefix}ffn_norm", [hidden_size], "f32", w.tobytes()))
        for proj, short in [("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")]:
            w = get_w(f"model.layers.{i}.mlp.{proj}.weight", f"layers.{i}.mlp.{proj}.weight")
            if w is not None:
                if short in ("ffn_gate", "ffn_up"):
                    w = expand_to(w, intermediate_size, hidden_size)
                else:
                    w = expand_to(w, hidden_size, intermediate_size)
                packed = pack_ternary(quantize_to_ternary(w))
                out_tensors.append((f"{prefix}{short}", list(w.shape), "i2_s", packed))

    tensors = out_tensors

    # bitnet-oxidized expects: token_embd, output_norm, output + per layer: attn_norm, attn_q,k,v,output, ffn_norm, ffn_gate,up,down (9 per layer)
    expected = 3 + num_layers * 9
    if len(tensors) < expected:
        print(f"Error: got {len(tensors)} tensors, expected {expected}. Missing tensors from HF model.")
        print("Available keys (first 30):", keys[:30])
        if len(tensors) == 0:
            print("No tensors found. Check that model dir contains model.safetensors with LLaMA/BitNet layout.")
            return 1
        return 1

    # Compute offsets (contiguous, no alignment between tensors to match Rust writer)
    data_offset = 0
    tensor_infos = []
    for name, dims, dtype, raw in tensors:
        n_bytes = len(raw)
        tensor_infos.append((name, dims, dtype, raw, data_offset))
        data_offset += n_bytes

    meta_kv_count = 9
    tensor_count = len(tensors)

    with open(out_path, "wb") as f:
        write_gguf_header(f, tensor_count, meta_kv_count)
        write_metadata(f, config)
        for name, dims, dtype, raw, off in tensor_infos:
            typ = GGUF_TYPE_F32 if dtype == "f32" else GGUF_TYPE_I2_S
            write_tensor_info(f, name, dims, typ, off)
        pos = f.tell()
        pad = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        f.write(b"\x00" * pad)
        for _name, _dims, _dtype, raw, _off in tensor_infos:
            f.write(raw)

    print(f"Wrote {out_path} with {len(tensors)} tensors")
    return 0


if __name__ == "__main__":
    exit(main())
