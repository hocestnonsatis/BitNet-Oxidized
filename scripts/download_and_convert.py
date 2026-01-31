#!/usr/bin/env python3
"""
Download a Hugging Face model and convert to GGUF in one step.

Optionally validates the output with the Rust binary (inspect).

Usage:
    pip install huggingface_hub
    python scripts/download_and_convert.py --repo 1bitLLM/bitnet_b1_58-large --out models/bitnet_b1_58-large
    python scripts/download_and_convert.py --repo 1bitLLM/bitnet_b1_58-large --out models/bitnet_b1_58-large --validate

For batch conversion of multiple repos, run this script in a loop or use a config file.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Download Hugging Face model and convert to GGUF for bitnet-oxidized"
    )
    ap.add_argument("--repo", required=True, help="Hugging Face repo id (e.g. 1bitLLM/bitnet_b1_58-large)")
    ap.add_argument(
        "--out",
        required=True,
        help="Local output directory (model files + <dir>.gguf)",
    )
    ap.add_argument("--token", default=None, help="HF token if repo is gated")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Run cargo run -- inspect on the output GGUF",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; only convert (input dir must exist)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    gguf_path = out_dir.with_suffix(out_dir.suffix + ".gguf")

    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("Install: pip install huggingface_hub", file=sys.stderr)
            return 1

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.repo} -> {out_dir.absolute()}")
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            token=args.token,
        )
        print("Download done.")
    else:
        if not out_dir.exists():
            print(f"Error: output dir does not exist: {out_dir}", file=sys.stderr)
            return 1

    script_dir = Path(__file__).resolve().parent
    convert_script = script_dir / "convert_huggingface_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: convert script not found: {convert_script}", file=sys.stderr)
        return 1

    print(f"Converting {out_dir} -> {gguf_path}")
    r = subprocess.run(
        [sys.executable, str(convert_script), "--input", str(out_dir), "--output", str(gguf_path)],
        cwd=script_dir.parent,
    )
    if r.returncode != 0:
        print("Conversion failed.", file=sys.stderr)
        return r.returncode
    print(f"Wrote {gguf_path}")

    if args.validate and gguf_path.exists():
        print("Validating with bitnet-oxidized inspect...")
        r2 = subprocess.run(
            ["cargo", "run", "--", "inspect", "--model", str(gguf_path), "--verbose"],
            cwd=script_dir.parent,
        )
        if r2.returncode != 0:
            print("Inspect failed.", file=sys.stderr)
            return r2.returncode
        print("Inspect OK.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
