# TODO — Remaining Work

Remaining tasks (in suggested order). Completed: validation, GGUF v2/converter, advanced sampling, profiling/optimization, model zoo/registry, Task 8 (API, Web UI), Task 9 (K8s, Docker, Health, Middleware, Monitoring), **Task 10 (Tests, Coverage, CI)**.

---

## Task 8: Gradio API, Web UI, Static Server — Done

- Expose a Gradio-like or simple web API for chat/completion. **Done**: `/v1/completions`, `/v1/chat/completions` already present.
- Add a minimal web UI (HTML/JS) for prompt + response. **Done**: `GET /ui` serves embedded `static/index.html` (prompt textarea, Generate button, response + usage).
- Optional: static file server for serving the UI. **Skipped**: UI is embedded via `include_str!` so no runtime static dir is required.

---

## Task 9: K8s, Docker, Health, Middleware, Monitoring — Done

- **Kubernetes**: `k8s/deployment.yaml` uses `/health` for liveness/readiness with failureThreshold and timeoutSeconds; `k8s/hpa.yaml` added for optional HPA.
- **Docker**: Dockerfile uses Rust 1.83; deployment guide has Verify step (curl /health, /ui).
- **Health**: `/health` documented as preferred for probes; K8s probes point to `/health`.
- **Middleware**: TraceLayer request logging documented; rate limiting and auth recommended at reverse proxy.
- **Monitoring**: Prometheus scrape config and pod annotations documented in deployment_guide.

---

## Task 10: Tests, Coverage, CI Workflows — Done

- **Tests**: Integration tests added: `engine_forward_empty_input_errors`, `validate_model_runs`, `validate_model_from_path_runs`, `streaming_generator_emits_tokens`. Edge cases and validation/streaming coverage expanded.
- **Coverage**: Tarpaulin + Codecov unchanged; new tests improve coverage.
- **CI**: Release build job added (`cargo build --release` on ubuntu). Tests, fmt, clippy, coverage unchanged.

---

## Task 5: Vision Encoder, Multimodal Foundation

- Add a vision encoder path (e.g. for image + text).
- Lay groundwork for multimodal inputs (image + prompt → text).

---

## Task 6: Distributed Sharding, Communication

- Model/shard distribution across multiple processes or nodes.
- Communication layer for shard coordination and result aggregation.
