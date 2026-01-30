#!/usr/bin/env python3
"""
Quantize a FP32 GGUF model to BitNet (ternary) format.

For file-to-file quantization, use the Rust CLI:

    cargo run -- quantize --input <path> --output <path>

This script is a stub that invokes the same; run from the repo root.
"""

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Quantize GGUF to BitNet (ternary). Delegates to cargo run -- quantize."
    )
    ap.add_argument("--input", required=True, help="Input .gguf path (F32 or I2_S weights)")
    ap.add_argument("--output", required=True, help="Output BitNet .gguf path")
    ap.add_argument("--method", default="absmax", choices=["absmax"], help="Ignored; AbsMax is used")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = ["cargo", "run", "--", "quantize", "--input", args.input, "--output", args.output]
    rc = subprocess.call(cmd, cwd=repo)
    if rc != 0:
        sys.exit(rc)
    print(f"Quantized: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
