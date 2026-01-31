# bitnet-oxidized

[![CI](https://github.com/hocestnonsatis/BitNet-Oxidized/actions/workflows/ci.yml/badge.svg)](https://github.com/hocestnonsatis/BitNet-Oxidized/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hocestnonsatis/BitNet-Oxidized/graph/badge.svg)](https://codecov.io/gh/hocestnonsatis/BitNet-Oxidized)

Experimental **1-bit (ternary) LLM inference** in pure Rust, inspired by [Microsoft's BitNet](https://arxiv.org/abs/2310.11453). The name plays on Rust's "oxidized" theme.

This is a research-oriented implementation. It is not production-hardened and does not ship with pretrained BitNet models—you need to convert or create them yourself (see [models/README.md](models/README.md)).

---

## What's implemented

- **Ternary weights**: 2-bit packing (4 weights per byte) with values `{-1, 0, +1}` (~16× smaller than FP32).
- **CPU mat-vec kernels**: basic (parallel dot product), blocked (4-wide unrolling), and **LUT-based** (typically 2–5× faster than basic). The inference engine uses the **SIMD** path by default (AVX2 on x86_64, NEON on aarch64), with LUT fallback when SIMD is unavailable.
- **BitNet transformer**: Pre-norm attention and FFN with ternary projections and full-precision RMS LayerNorm.
- **Inference**: Forward pass and text generation (greedy, top-k, top-p). Generation uses an integrated **KV cache**: one forward step per token (no full-sequence recompute).
- **GGUF**: Load and save BitNet models in GGUF v3 format (F32 + custom 2-bit tensor type).
- **Tokenizer**: Optional tokenizer from file (via the `tokenizers` crate) for `infer` and `chat`; otherwise a naive whitespace/hash tokenizer is used.
- **Quantization**: AbsMax-style FP32 → ternary conversion and error metrics. CLI `quantize --input <path> --output <path>` loads a GGUF (F32 or I2_S) and saves a BitNet GGUF (ternary). In-memory workflow in `examples/quantize_model.rs`.
- **CLI**: `demo`, `infer`, `info`, `bench`, `chat`, `serve`, `profile`, `quantize`, `test-tokenizer`. `chat` and `profile` can fall back to the built-in demo model if no GGUF path is provided or the file is missing.
- **HTTP server**: `serve` runs an Axum-based API for inference (prompt completion and chat-style endpoints). See [docs/deployment_guide.md](docs/deployment_guide.md) for deployment options.
- **Errors and profiling**: `BitNetError` and `Profiler` / `DetailedMetrics` for diagnostics.

---

## Limitations

- **No pretrained BitNet GGUF in this repo.** Use the scripts under `scripts/` (e.g. `convert_huggingface_to_gguf.py`) to convert Hugging Face BitNet models, or create a demo model in code. See [models/README.md](models/README.md).
- **CLI quantize**: Input GGUF must use F32 or I2_S for weight tensors; output is always ternary (I2_S).
- **KV cache**: Used by the generator; `forward_step` + cache gives one-token decoding. Full `forward()` remains for batch/prefill when needed.
- **SIMD**: The engine uses the SIMD mat-vec by default (AVX2/NEON). You can still call `mat_vec_mul_lut`, `mat_vec_mul_basic`, or `mat_vec_mul_blocked` directly if needed.
- **Phones / Termux**: Do **not** run HF→GGUF conversion on the device—it loads the full model into RAM and can crash. Run conversion on a desktop or server and copy the `.gguf` file. See [docs/termux_and_phones.md](docs/termux_and_phones.md).

---

## Project structure

```
bitnet-oxidized/
├── src/
│   ├── kernels/      # Ternary tensor, basic/blocked/LUT mat-vec, SIMD
│   ├── model/        # BitNet config, layers, demo, GGUF load/save
│   ├── inference/    # Engine, generator, KV cache, streaming
│   ├── quantization/# AbsMax quantize/dequantize
│   ├── server/       # HTTP API
│   ├── tokenizer/   # BitNetTokenizer wrapper
│   └── utils/       # Metrics (perplexity, argmax), profiler
├── examples/        # basic_inference, benchmark, quantize_model, debug_chat
├── benches/         # Criterion kernel + e2e benchmarks
├── scripts/         # convert_huggingface_to_gguf.py, download_hf_model.py, etc.
├── tests/           # Integration tests
├── docs/            # deployment_guide, model_card, termux_and_phones
└── models/          # Model configs and instructions (see models/README.md)
```

---

## Quick start

```bash
# Demo (small random model, no file needed)
cargo run -- demo

# Basic inference example
cargo run --example basic_inference

# Compare kernel performance
cargo run --example benchmark

# In-memory quantize (FP32 → ternary)
cargo run --example quantize_model

# Run tests
cargo test

# Run benchmarks
cargo bench
```

---

## CLI reference

| Command | Description |
|--------|-------------|
| `cargo run -- demo` | Run with built-in demo model (no file). |
| `cargo run -- infer --model <path> --prompt <text>` | Run inference. Optional: `--tokenizer <path>`, `--max-tokens <n>` (default: 50). |
| `cargo run -- info --model <path>` | Print model info (vocab_size, hidden_size, num_layers). |
| `cargo run -- bench --model <path>` | Benchmark forward pass. |
| `cargo run -- chat --model <path>` | Interactive chat. Omit or use missing path for demo model. Optional: `--tokenizer <path>`, `--temperature <f>` (default 0.7), `--top-p <f>` (0.9), `--top-k <n>` (50), `--repetition-penalty <f>` (1.2), `--frequency-penalty`, `--presence-penalty`, `--system-prompt <s>`, `--debug`, `--greedy`. |
| `cargo run -- serve --model <path>` | HTTP API. Optional: `--port <port>` (default 8080), `--batch-size <n>` (default 4). |
| `cargo run -- profile --model <path>` | Profile forward pass. Optional: `--iterations <n>` (default 100). |
| `cargo run -- quantize --input <path> --output <path>` | Load GGUF (F32 or ternary) and save BitNet GGUF (ternary). |
| `cargo run -- test-tokenizer --tokenizer <path> --text <s>` | Test tokenizer: encode/decode and vocab check. |

**Examples with tokenizer (decoded text):**

```bash
cargo run --release -- infer --model models/bitnet_b1_58-large.gguf \
  --prompt "Hello, my name is" --max-tokens 40 \
  --tokenizer models/bitnet_b1_58-large/tokenizer.json
```

---

## Architecture (brief)

- **Ternary tensor**: Weights packed 4 per byte (00=Zero, 01=+1, 11=−1). Row-major, shape `[out_features, in_features]`.
- **Forward**: Embed → for each layer: RMS norm → attention (Q,K,V,O) → residual → RMS norm → FFN (gate/up → SiLU → down) → residual → final RMS norm → LM head → logits.
- **Generation**: Single-token steps with KV cache; each step runs one forward for the new token and samples the next.

---

## Performance

- The engine uses the **SIMD kernel** (AVX2/NEON) by default; LUT is used when SIMD is unavailable. Set `RAYON_NUM_THREADS` to control parallelism for non-SIMD paths.
- Ternary weights give ~16× smaller weight memory than FP32; embeddings and LayerNorm remain FP32 here.

---

## Documentation

- [models/README.md](models/README.md) — Download and convert Hugging Face BitNet models.
- [docs/deployment_guide.md](docs/deployment_guide.md) — Deploying the HTTP server and usage.
- [docs/model_card.md](docs/model_card.md) — Model card and capabilities.
- [docs/termux_and_phones.md](docs/termux_and_phones.md) — Running on Termux and phones.

---

## Development

Optional pre-commit hook to run `cargo fmt` and re-stage changed `.rs` files:

```bash
./scripts/setup-git-hooks.sh
```

**CI**: On push/PR to `main` or `master`, GitHub Actions runs tests (Ubuntu + macOS), formatting, Clippy, and coverage (Tarpaulin → Codecov). See [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## License

MIT. See [LICENSE](LICENSE).
