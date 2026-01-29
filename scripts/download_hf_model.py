#!/usr/bin/env python3
"""
Download a HuggingFace model (e.g. Llama3-8B-1.58) to a local directory.

Usage:
    pip install huggingface_hub
    python scripts/download_hf_model.py
    # or:
    python scripts/download_hf_model.py --repo HF1BitLLM/Llama3-8B-1.58-100B-tokens --out models/Llama3-8B-1.58
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Download HuggingFace model for bitnet-oxidized")
    ap.add_argument(
        "--repo",
        default="HF1BitLLM/Llama3-8B-1.58-100B-tokens",
        help="HuggingFace repo id",
    )
    ap.add_argument(
        "--out",
        default="models/Llama3-8B-1.58-100B-tokens",
        help="Local output directory",
    )
    ap.add_argument("--token", default=None, help="HF token if repo is gated")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} -> {out.absolute()}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        token=args.token,
    )
    print("Done. Files:", list(out.iterdir()))
    return 0


if __name__ == "__main__":
    exit(main())
