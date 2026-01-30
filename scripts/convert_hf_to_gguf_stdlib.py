#!/usr/bin/env python3
"""
Convert HuggingFace BitNet safetensors to GGUF using only Python stdlib (no numpy/safetensors).

**Do not run on phones/Termux** — loads the full model into RAM and can crash all apps.
Run on a desktop or server, then copy the .gguf file to the device. See docs/termux_and_phones.md.

Usage (on a desktop/server with enough RAM):
    python3 scripts/convert_hf_to_gguf_stdlib.py \
        --input models/bitnet-b1.58-2B-4T \
        --output models/bitnet-b1.58-2B-4T.gguf
"""

import argparse
import json
import struct
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_TYPE_F32 = 0
GGUF_TYPE_I2_S = 40


def bfloat16_to_float32_raw(raw: bytes) -> list:
    """Convert raw bfloat16 bytes to list of float32."""
    n = len(raw) // 2
    out = []
    for i in range(n):
        u16 = struct.unpack_from("<H", raw, i * 2)[0]
        u32 = (u16 << 16) & 0xFFFF0000
        f = struct.unpack("<f", struct.pack("<I", u32))[0]
        out.append(f)
    return out


def load_f32_raw(raw: bytes) -> list:
    n = len(raw) // 4
    return list(struct.unpack_from(f"<{n}f", raw))


def u8_to_ternary_floats(raw: bytes) -> list:
    """U8 0/1/2 -> 0.0, 1.0, -1.0."""
    out = []
    for b in raw:
        if b == 0:
            out.append(0.0)
        elif b == 1:
            out.append(1.0)
        else:
            out.append(-1.0)
    return out


def float_to_ternary_bits(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return 3
    return 0


def quantize_to_ternary(weights: list) -> list:
    """Quantize floats to -1, 0, +1 (threshold at 0)."""
    return [1.0 if w > 0 else (-1.0 if w < 0 else 0.0) for w in weights]


def pack_ternary(weights: list) -> bytes:
    """Pack ternary floats into 2 bits each, 4 per byte."""
    out = []
    for i in range(0, len(weights), 4):
        b = 0
        for j in range(4):
            if i + j < len(weights):
                b |= float_to_ternary_bits(weights[i + j]) << (j * 2)
        out.append(b)
    return bytes(out)


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def load_safetensors_raw(model_dir: Path) -> tuple[dict, list]:
    """Load safetensors; return (name -> (shape, list of floats)), keys."""
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(st_path)
    with open(st_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n).decode("utf-8"))
    data_start = 8 + (n + 7) // 8 * 8
    keys = [k for k in meta if isinstance(meta.get(k), dict) and "dtype" in meta[k]]
    with open(st_path, "rb") as f:
        f.seek(data_start)
        data = f.read()
    tensors = {}
    for key in keys:
        info = meta[key]
        dtype = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]
        raw = data[start:end]
        n_elems = 1
        for s in shape:
            n_elems *= s
        if dtype == "BF16":
            floats = bfloat16_to_float32_raw(raw)[:n_elems]
        elif dtype == "F32":
            floats = load_f32_raw(raw)[:n_elems]
        elif dtype == "U8":
            floats = u8_to_ternary_floats(raw[:n_elems])
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        tensors[key] = (list(shape), floats)
    return tensors, keys


def expand_to(shape: list, floats: list, expected_out: int, expected_in: int) -> list:
    """Repeat rows/cols so matrix (row-major) matches (expected_out, expected_in)."""
    out, inp = shape[0], shape[1]
    if out == expected_out and inp == expected_in:
        return floats
    result = []
    if out < expected_out and inp == expected_in:
        repeat = expected_out // out
        for row in range(out):
            row_data = floats[row * inp : (row + 1) * inp]
            for _ in range(repeat):
                result.extend(row_data)
        return result
    if out == expected_out and inp < expected_in:
        repeat = expected_in // inp
        for row in range(out):
            row_data = floats[row * inp : (row + 1) * inp]
            for _ in range(repeat):
                result.extend(row_data)
        return result
    return floats


def write_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_metadata(f, config: dict):
    meta = [
        ("general.architecture", 8, "bitnet"),
        ("bitnet.vocab_size", 4, config["vocab_size"]),
        ("bitnet.context_length", 4, config.get("max_position_embeddings", 2048)),
        ("bitnet.embedding_length", 4, config["hidden_size"]),
        ("bitnet.block_count", 4, config["num_hidden_layers"]),
        ("bitnet.feed_forward_length", 4, config["intermediate_size"]),
        ("bitnet.attention.head_count", 4, config["num_attention_heads"]),
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
    st_dict, keys = load_safetensors_raw(model_dir)

    def get_w(key_pri, key_alt=None):
        v = st_dict.get(key_pri)
        if v is None and key_alt:
            v = st_dict.get(key_alt)
        return v

    out_tensors = []

    # Embeddings
    w = get_w("model.embed_tokens.weight", "embed_tokens.weight")
    if w:
        shape, floats = w
        n = vocab_size * hidden_size
        out_tensors.append(("token_embd", [vocab_size, hidden_size], "f32", struct.pack(f"<{n}f", *floats[:n])))

    # output_norm
    w = get_w("model.norm.weight", "norm.weight")
    if w:
        _, floats = w
        out_tensors.append(("output_norm", [hidden_size], "f32", struct.pack(f"<{hidden_size}f", *floats[:hidden_size])))

    # output (lm_head) - tied or separate
    w = get_w("lm_head.weight", "model.lm_head.weight")
    if w is None and config.get("tie_word_embeddings"):
        w = get_w("model.embed_tokens.weight", "embed_tokens.weight")
    if w:
        shape, floats = w
        ternary = quantize_to_ternary(floats[: vocab_size * hidden_size])
        packed = pack_ternary(ternary)
        out_tensors.append(("output", [vocab_size, hidden_size], "i2_s", packed))

    # Layers
    for i in range(num_layers):
        prefix = f"blk.{i}."
        w = get_w(f"model.layers.{i}.input_layernorm.weight", f"layers.{i}.input_layernorm.weight")
        if w:
            _, floats = w
            out_tensors.append((f"{prefix}attn_norm", [hidden_size], "f32", struct.pack(f"<{hidden_size}f", *floats[:hidden_size])))
        for proj, short in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            w = get_w(f"model.layers.{i}.self_attn.{proj}.weight", f"layers.{i}.self_attn.{proj}.weight")
            if w:
                shape, floats = w
                expanded = expand_to(shape, floats, hidden_size, hidden_size)
                ternary = quantize_to_ternary(expanded)
                packed = pack_ternary(ternary)
                out_tensors.append((f"{prefix}{short}", [hidden_size, hidden_size], "i2_s", packed))
        w = get_w(f"model.layers.{i}.post_attention_layernorm.weight", f"layers.{i}.post_attention_layernorm.weight")
        if w:
            _, floats = w
            out_tensors.append((f"{prefix}ffn_norm", [hidden_size], "f32", struct.pack(f"<{hidden_size}f", *floats[:hidden_size])))
        for proj, short in [("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")]:
            w = get_w(f"model.layers.{i}.mlp.{proj}.weight", f"layers.{i}.mlp.{proj}.weight")
            if w:
                shape, floats = w
                if short in ("ffn_gate", "ffn_up"):
                    expanded = expand_to(shape, floats, intermediate_size, hidden_size)
                    out_dims = [intermediate_size, hidden_size]
                else:
                    expanded = expand_to(shape, floats, hidden_size, intermediate_size)
                    out_dims = [hidden_size, intermediate_size]
                ternary = quantize_to_ternary(expanded)
                packed = pack_ternary(ternary)
                out_tensors.append((f"{prefix}{short}", out_dims, "i2_s", packed))

    if len(out_tensors) < 3:
        print("Error: too few tensors. Available keys:", keys[:30])
        return 1

    data_offset = 0
    tensor_infos = []
    for name, dims, dtype, raw in out_tensors:
        tensor_infos.append((name, dims, dtype, raw, data_offset))
        data_offset += len(raw)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(out_tensors)))
        f.write(struct.pack("<Q", 8))
        write_metadata(f, config)
        for name, dims, dtype, raw, off in tensor_infos:
            typ = GGUF_TYPE_F32 if dtype == "f32" else GGUF_TYPE_I2_S
            write_tensor_info(f, name, dims, typ, off)
        pos = f.tell()
        pad = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        f.write(b"\x00" * pad)
        for _n, _d, _t, raw, _o in tensor_infos:
            f.write(raw)

    print(f"Wrote {out_path} with {len(out_tensors)} tensors")
    return 0


if __name__ == "__main__":
    exit(main())
