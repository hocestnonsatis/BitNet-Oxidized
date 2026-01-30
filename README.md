# bitnet-oxidized

Experimental **1-bit (ternary) LLM inference** in pure Rust, inspired by [Microsoft's BitNet](https://arxiv.org/abs/2310.11453). The name plays on Rust's "oxidized" theme.

This is a research-oriented implementation. It is not production-hardened and does not ship with pretrained BitNet models—you need to convert or create them yourself.

## What's implemented

- **Ternary weights**: 2-bit packing (4 weights per byte) with values `{-1, 0, +1}` (~16× smaller than FP32).
- **CPU mat-vec kernels**: basic (parallel dot product), blocked (4-wide unrolling), and **LUT-based** (default in the engine; typically 2–5× faster than basic). A **SIMD** path (AVX2 on x86_64, NEON on aarch64) exists but is not wired into the inference engine.
- **BitNet transformer**: Pre-norm attention and FFN with ternary projections and full-precision RMS LayerNorm.
- **Inference**: Forward pass and text generation (greedy, top-k, top-p). Each generation step runs a full forward pass; a `KVCache` type exists but is not yet integrated.
- **GGUF**: Load and save BitNet models in GGUF v3 format (F32 + custom 2-bit tensor type).
- **Tokenizer**: Optional tokenizer from file (via the `tokenizers` crate) for the `infer` command; otherwise a naive whitespace/hash tokenizer is used.
- **Quantization**: AbsMax-style FP32 → ternary conversion and error metrics. In-memory workflow is in `examples/quantize_model.rs`. The CLI `quantize` subcommand (file-to-file) is **not** implemented.
- **CLI**: `demo`, `infer`, `info`, `bench`, `chat`, `serve`, `profile`. `chat` and `profile` can fall back to the built-in demo model if no GGUF path is provided or the file is missing.
- **HTTP server**: `serve` runs an Axum-based API for inference.
- **Errors and profiling**: `BitNetError` and `Profiler` / `DetailedMetrics` for diagnostics.

## Limitations

- **No pretrained BitNet GGUF in this repo.** Use the scripts under `scripts/` (e.g. `convert_huggingface_to_gguf.py`) to convert Hugging Face BitNet models, or create a demo model in code.
- **CLI quantize**: The `quantize` subcommand bails with "not yet implemented." Use the `quantize_model` example for in-memory quantization.
- **KV cache**: The cache type is implemented and exported but not used in the inference engine; each step does a full forward over the current sequence.
- **SIMD**: The SIMD mat-vec is available in code but the engine uses the LUT kernel. You can swap to SIMD in your own code if desired.

## Project structure

```
bitnet-oxidized/
├── src/
│   ├── kernels/     # Ternary tensor, basic/blocked/LUT mat-vec, SIMD
│   ├── model/       # BitNet config, layers, demo, GGUF load/save
│   ├── inference/   # Engine, generator, KV cache (type only), streaming
│   ├── quantization/# AbsMax quantize/dequantize
│   ├── server/      # HTTP API
│   ├── tokenizer/   # BitNetTokenizer wrapper
│   └── utils/       # Metrics (perplexity, argmax), profiler
├── examples/        # basic_inference, benchmark, quantize_model
├── benches/         # Criterion kernel benchmarks
├── scripts/         # convert_huggingface_to_gguf.py, quantize_model.py, setup-git-hooks.sh
└── tests/           # Integration tests
```

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

## CLI

| Command | Description |
|--------|-------------|
| `cargo run -- demo` | Run with built-in demo model (no file). |
| `cargo run -- infer --model <path> --prompt <text>` | Run inference. Use `--tokenizer <path>` for decoded text. |
| `cargo run -- info --model <path>` | Print model info (vocab_size, hidden_size, num_layers). |
| `cargo run -- bench --model <path>` | Benchmark forward pass. |
| `cargo run -- chat --model <path>` | Interactive chat (omit or use missing path for demo model). |
| `cargo run -- serve --model <path> --port <port>` | HTTP API for inference. |
| `cargo run -- profile --model <path> --iterations <n>` | Profile forward pass latency. |
| `cargo run -- quantize --input <path> --output <path>` | **Not implemented**; use `examples/quantize_model.rs` for in-memory quantize. |

## Architecture (brief)

- **Ternary tensor**: Weights packed 4 per byte (00=Zero, 01=+1, 11=−1). Row-major, shape `[out_features, in_features]`.
- **Forward**: Embed → for each layer: RMS norm → attention (Q,K,V,O) → residual → RMS norm → FFN (gate/up → SiLU → down) → residual → final RMS norm → LM head → logits.
- **Generation**: Single-token steps; each step runs full forward on the current sequence and samples the next token.

## Performance

- Use the **LUT kernel** (default in the engine). Set `RAYON_NUM_THREADS` to control parallelism.
- Ternary weights give ~16× smaller weight memory than FP32; embeddings and LayerNorm remain FP32 here.

## Development

Optional pre-commit hook to run `cargo fmt` and re-stage changed `.rs` files:

```bash
./scripts/setup-git-hooks.sh
```

## License

MIT. See [LICENSE](LICENSE).
