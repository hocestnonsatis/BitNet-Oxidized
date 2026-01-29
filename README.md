# bitnet-oxidized

High-performance **1-bit LLM inference** framework in pure Rust, inspired by [Microsoft's BitNet](https://arxiv.org/abs/2310.11453). The name emphasizes Rust's oxidation theme.

## Features

- **Ternary weight system**: 2-bit packing (4 weights per byte) with values `{-1, 0, +1}` (~16× smaller than FP32).
- **Three CPU mat-vec kernels**: basic (parallel dot product), blocked (4-wide unrolling), and **LUT-based** (fastest; 2–5× speedup over basic).
- **BitNet transformer**: Pre-norm attention + FFN with ternary projections and full-precision RMS LayerNorm.
- **Inference engine**: Forward pass and text generation (greedy, top-k, top-p).
- **Quantization**: AbsMax-style FP32 → ternary conversion and error metrics.

## Project structure

```
bitnet-oxidized/
├── src/
│   ├── kernels/     # Ternary tensor, basic/blocked/LUT mat-vec
│   ├── model/       # BitNet config, layers, demo, GGUF stub
│   ├── inference/   # Engine + text generator
│   ├── quantization/# AbsMax quantize/dequantize
│   └── utils/       # Metrics (perplexity, argmax)
├── examples/        # basic_inference, benchmark, quantize_model
├── benches/         # Criterion kernel benchmarks
└── tests/           # Integration tests
```

## Quick start

```bash
# Demo (small random model)
cargo run -- demo

# Basic inference example
cargo run --example basic_inference

# Compare kernel performance
cargo run --example benchmark

# Quantize FP32 → ternary
cargo run --example quantize_model

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## CLI

| Command | Description |
|--------|-------------|
| `cargo run -- demo` | Run with a demo model (no file needed). |
| `cargo run -- infer --model <path> --prompt <text>` | Run inference (requires GGUF model). |
| `cargo run -- info --model <path>` | Show model info. |
| `cargo run -- bench --model <path>` | Benchmark forward pass. |
| `cargo run -- quantize --input <fp32> --output <ternary>` | Quantize (stub; use example for in-memory). |

GGUF load/save are stubs; use `create_demo_model()` or the quantize example for in-memory workflows.

## Architecture overview

1. **Ternary tensor**: Weights packed 4 per byte (00=Zero, 01=+1, 11=−1). `TernaryTensor` provides `get`/`set`, `memory_usage()`, and `to_f32_vec()`.
2. **Mat-vec**: Weight shape `[out_features, in_features]`, row-major. All three kernels (basic, blocked, LUT) produce identical results; LUT uses a 256-entry lookup table per byte for speed.
3. **Forward pass**: Embed → for each layer: input RMS norm → attention (Q,K,V,O) → residual → post-attention RMS norm → FFN (gate/up → SiLU → down) → residual → final RMS norm → LM head → logits.
4. **Generation**: Single-token steps; each step runs full forward on current sequence and samples next token (greedy, top-k, or top-p).

## Performance tuning

- Use **LUT kernel** for inference (default in the engine).
- Set `RAYON_NUM_THREADS` to control parallelism.
- For tiny models, reduce layers/hidden size in the demo config to measure latency.

## Memory

- Ternary: **2 bits per weight** → ~**16×** smaller than FP32.
- Embeddings and LayerNorm remain FP32 in this implementation.

## Phase 2 (Production)

- **GGUF**: Full reader/writer in `src/model/gguf.rs` (header, metadata, tensor infos, F32 + I2_S).
- **Tokenizer**: `BitNetTokenizer` from `tokenizers` crate (`from_file`, `encode`, `decode`).
- **KV cache**: `KVCache` in `src/inference/cache.rs` for fast autoregressive generation.
- **SIMD**: `mat_vec_mul_simd` with AVX2 (x86_64) and NEON (aarch64), LUT fallback.
- **Errors**: `BitNetError` in `src/errors.rs` (InvalidFormat, DimensionMismatch, UnsupportedGGUFVersion, etc.).
- **Profiler**: `Profiler` and `DetailedMetrics` in `src/utils/metrics.rs`.
- **CLI**: `chat`, `serve`, `profile` in addition to demo, infer, info, bench, quantize.
- **Scripts**: `scripts/convert_huggingface_to_gguf.py`, `scripts/quantize_model.py`.

### GGUF usage

```bash
# Save demo model to GGUF
cargo run --example basic_inference  # then use save_gguf in code

# Load GGUF (requires bitnet architecture and our tensor names)
cargo run -- infer --model model.gguf --prompt "Hello"
cargo run -- info --model model.gguf
```

### Chat and profile

```bash
cargo run -- chat --model /path/to/model.gguf   # or omit for demo model
cargo run -- profile --model /path/to/model.gguf --iterations 100
```

## License

MIT. See [LICENSE](LICENSE).
