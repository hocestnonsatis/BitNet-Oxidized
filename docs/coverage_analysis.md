# Code Coverage Analysis (~42%)

This document explains why the GitHub Actions Coverage job reports **~42%** and how coverage can be improved.

## How Coverage Is Measured

- **CI**: `cargo tarpaulin --out Xml` → uploads `cobertura.xml` to Codecov.
- Tarpaulin by default **runs only tests**: lib + integration tests via `cargo test`. **The binary (`src/main.rs`) and examples/benches are not included in coverage.**

## Why Is It Low?

### 1. Binary (CLI) Is Never Measured

- `src/main.rs` (~660 lines) contains the entire CLI (infer, chat, serve, bench, profile, quantize, validate, etc.).
- Tarpaulin only exercises the **lib** target; the binary is never run, so that code counts as **0%** covered.

### 2. Large Modules Are Barely or Never Used in Tests

| Module | Lines | Status |
|--------|-------|--------|
| `model/gguf.rs` | ~843 | Only save/load, inspect, repair roundtrip are tested; v3, from_pretrained, various tensor types and branches are not exercised. |
| `server/mod.rs` | ~400 | HTTP API is never tested (serve command lives only in the binary). |
| `validation/mod.rs` | ~371 | `validate_model` / `validate_model_from_path` are not called from integration tests. |
| `debugging/` | ~190 | Layer tracer etc. are not used in tests. |
| `profiling/performance.rs` | ~165 | Profiler is not run in tests. |
| `optimization/kernel_autotuning.rs` | ~123 | Autotuning is not triggered in tests. |
| `inference/` (streaming, pipeline, …) | many | Only forward + generator (greedy/top_k/top_p) and speculative/prefix_cache are tested; streaming, pipeline, logit processor chains are partially or not exercised. |
| `tokenizer/mod.rs` | ~37 | Integration tests do not use the tokenizer (only token id lists). |

### 3. What Tests Do Cover (well-covered areas)

- **Kernels**: `TernaryTensor`, mat_vec (basic/blocked/lut), MoE forward.
- **Model**: `create_demo_model` / `create_demo_model_seeded`, forward, GGUF save/load/inspect/repair.
- **Inference**: `InferenceEngine::forward`, `TextGenerator` (greedy, top_k, top_p), `SpeculativeDecoder`, `PrefixCache`, telemetry export.

**Summary**: A large share of total lines is in CLI + server + validation + GGUF/from_pretrained branches + profiling/debugging. With only “lib + current integration tests”, **~42%** is a reasonable outcome.

## Recommendations to Improve Coverage

1. **Add more lib tests**
   - Test `from_pretrained` with a small GGUF (or fixture).
   - Add at least one integration test for `validate_model` / `validate_model_from_path`.
   - **Server**: use `axum::test` (or similar) to send requests to the router.
   - **Streaming**: test `StreamGenerator` with a short generate and assert on chunks.
   - **Profiler**: at least one test that runs a forward pass and calls `write_report`.

2. **Make Tarpaulin scope explicit**
   - To measure only the lib: `cargo tarpaulin --lib --out Xml` (behavior is already close; binary still not measured).
   - Optionally use `--exclude-files` to exclude benchmarks/examples explicitly.

3. **Including the binary in coverage**
   - To measure the binary with Tarpaulin: a separate run or extra args (e.g. `cargo tarpaulin --bin bitnet-oxidized --out Xml`) is needed; often lib + tests remain the main metric and the binary is tracked separately.

4. **Codecov configuration**
   - Use `codecov.yml` to include only `src/` (and optionally `tests/`) and set thresholds accordingly, so that the ~42% figure is clearly “lib + current tests” and documented.

## Checking Coverage in CI

To inspect the Coverage job and the uploaded percentage:

```bash
gh run list --workflow=ci.yml --limit 5
gh run view <run_id> --log   # Look for "Generate coverage" / "Upload coverage" steps
```

For the percentage and graphs: the repo’s Codecov page (link from the README badge).
