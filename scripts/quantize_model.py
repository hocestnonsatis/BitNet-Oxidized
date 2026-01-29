#!/usr/bin/env python3
"""
Quantize a FP32 GGUF model to BitNet (ternary) format.

Usage:
    python scripts/quantize_model.py \
        --input models/llama-7b-fp32.gguf \
        --output models/llama-7b-bitnet.gguf \
        --method absmax

Requires: bitnet-oxidized GGUF reader/writer (use Rust API or implement reader here).
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input FP32 .gguf path")
    ap.add_argument("--output", required=True, help="Output BitNet .gguf path")
    ap.add_argument("--method", default="absmax", choices=["absmax"], help="Quantization method")
    args = ap.parse_args()

    print("Quantize: use bitnet-oxidized from Rust:")
    print("  1. Load FP32 GGUF (if supported) or use create_demo_model() + save_gguf()")
    print("  2. For in-memory: use bitnet_oxidized::quantization::absmax_quantize()")
    print("  3. Save with model::gguf::save_gguf()")
    print(f"  Input: {args.input}, Output: {args.output}, Method: {args.method}")
    return 0


if __name__ == "__main__":
    exit(main())
