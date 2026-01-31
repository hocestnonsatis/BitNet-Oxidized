#!/usr/bin/env python3
"""Test script: verify Python gguf package is installed (for testing only).

Note: BitNet-Oxidized GGUF files use custom tensor type I2_S (40), which the
standard gguf package does not support. Use bitnet-oxidized CLI to load/infer.
"""
import sys

def main():
    import gguf
    print("gguf package installed successfully.")
    if hasattr(gguf, "__version__"):
        print(f"Version: {gguf.__version__}")
    # BitNet GGUF uses custom type 40 (I2_S); standard reader would fail on it.
    print("(BitNet .gguf files use custom type I2_S; use: cargo run -- chat --model <path>)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
