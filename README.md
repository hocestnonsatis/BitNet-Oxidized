# bitnet-oxidized

[![CI](https://github.com/hocestnonsatis/BitNet-Oxidized/actions/workflows/ci.yml/badge.svg)](https://github.com/hocestnonsatis/BitNet-Oxidized/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hocestnonsatis/BitNet-Oxidized/graph/badge.svg)](https://codecov.io/gh/hocestnonsatis/BitNet-Oxidized)

Experimental **1-bit (ternary) LLM inference** in pure Rust, inspired by [Microsoft's BitNet](https://arxiv.org/abs/2310.11453). The name plays on Rust's "oxidized" theme.

This is a research-oriented implementation. It is not production-hardened and does not ship with pretrained BitNet models—you need to convert or create them yourself (see [models/README.md](models/README.md)).

---

## What's implemented

- **Ternary weights**: 2-bit packing (4 weights per byte) with values `{-1, 0, +1}` (~16× smaller than FP32).
- **CPU mat-vec kernels**: basic, blocked, **LUT**, and **SIMD** (AVX2/NEON). **Kernel autotuning**: `best_kernel_for_size()` with cache; override via `BITNET_KERNEL_OVERRIDE=basic|blocked|lut|simd`.
- **BitNet transformer**: Pre-norm attention and FFN with ternary projections and full-precision RMS LayerNorm.
- **Inference**: Forward pass and text generation (greedy, top-k, top-p). **Advanced sampling**: Mirostat, locally-typical, contrastive, min-p; **logit processors** (temperature, repetition/top-k/top-p, bad words, frequency penalty); **constraints** (lexical, allowed tokens); **GenerationPipeline** with configurable strategy.
- **KV cache**: One forward step per token; prefix cache and speculative decoding support.
- **GGUF**: Load/save in GGUF v2/v3; **inspect** (metadata/tensors), **repair** (re-save v3, alignment). Stub **convert** (safetensors→GGUF; use Python scripts for Hugging Face).
- **Model zoo**: **Registry** and **from_pretrained(name_or_path)**: `"demo"` or file path or registered name. Env: `BITNET_REGISTRY` (JSON), `BITNET_MODEL_<NAME>=/path`. CLI **models** lists registered models; **infer**, **chat**, **serve**, **profile**, **info**, **bench** accept `--model <name or path>`.
- **Validation**: Full suite (GGUF load, forward pass, attention); CLI **validate** and global **--validate**; layer tracer and Chrome trace export for debugging.
- **Profiling**: Per-layer timing, FLOPS estimate, Chrome trace export; **profile_inference** example; **memory pool** for reusable f32 buffers.
- **Tokenizer**: Optional tokenizer from file for `infer`/`chat`; otherwise naive tokenizer.
- **Quantization**: AbsMax FP32→ternary; CLI **quantize**.
- **CLI**: `demo`, `infer`, `info`, `bench`, `chat`, `serve`, `profile`, `quantize`, `test-tokenizer`, **validate**, **convert**, **repair**, **inspect**, **models**.
- **HTTP server**: Axum API (completions, chat, health, metrics). See [docs/deployment_guide.md](docs/deployment_guide.md).
- **Errors and profiling**: `BitNetError`, `Profiler`, `DetailedMetrics`, `InferenceProfiler`.

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
│   ├── model/        # BitNet config, layers, demo, GGUF load/save, registry
│   ├── inference/    # Engine, generator, KV cache, sampling, pipeline, streaming
│   ├── optimization/ # Kernel autotuning (best_kernel_for_size)
│   ├── profiling/    # InferenceProfiler, Chrome trace
│   ├── validation/   # GGUF/forward/attention validation
│   ├── debugging/    # Layer tracer
│   ├── quantization/# AbsMax quantize/dequantize
│   ├── server/       # HTTP API
│   ├── tokenizer/   # BitNetTokenizer wrapper
│   └── utils/       # Metrics, profiler, memory_pool
├── examples/        # basic_inference, benchmark, quantize_model, validate_model, profile_inference, advanced_generation
├── benches/         # Criterion kernel + e2e + kernel_comparison
├── scripts/         # convert_huggingface_to_gguf.py, download_and_convert.py, etc.
├── tests/           # Integration + reference_comparison
├── docs/            # deployment_guide, model_card, termux_and_phones, TODO
└── models/          # Model configs and instructions (see models/README.md)
```

---

## Quick start

```bash
# Demo (small random model, no file needed)
cargo run -- demo

# Inference by name or path (demo = built-in model)
cargo run -- infer --model demo --prompt "hello" --max-tokens 10

# List registered models
cargo run -- models

# Basic inference example
cargo run --example basic_inference

# Compare kernel performance
cargo run --example benchmark

# Profile inference (per-layer timing, optional Chrome trace)
cargo run --example profile_inference -- --trace trace.json

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
| `cargo run -- models` | List registered models (demo + env `BITNET_REGISTRY` / `BITNET_MODEL_*`). |
| `cargo run -- infer --model <name or path> --prompt <text>` | Run inference. Use `demo` or a GGUF path. Optional: `--tokenizer <path>`, `--max-tokens <n>` (default: 50). |
| `cargo run -- info --model <name or path>` | Print model info (vocab_size, hidden_size, num_layers). |
| `cargo run -- bench --model <name or path>` | Benchmark forward pass. |
| `cargo run -- chat --model <name or path>` | Interactive chat. Optional: `--tokenizer <path>`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--frequency-penalty`, `--presence-penalty`, `--system-prompt`, `--debug`, `--greedy`. |
| `cargo run -- serve --model <name or path>` | HTTP API. Optional: `--port` (default 8080), `--batch-size` (default 4). |
| `cargo run -- profile --model <name or path>` | Profile forward pass. Optional: `--iterations` (default 100). |
| `cargo run -- validate --model <path> [--output <path>]` | Run full validation suite; report to stdout or JSON file. |
| `cargo run -- convert --input <path> --output <path>` | Stub: convert to GGUF (use Python scripts for HF). |
| `cargo run -- repair --model <path> --output <path>` | Repair GGUF (re-save v3, alignment). |
| `cargo run -- inspect --model <path> [--verbose]` | Inspect GGUF metadata and tensors. |
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
- [docs/TODO.md](docs/TODO.md) — Remaining tasks (Gradio/UI, K8s/health, tests, vision, sharding).

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
