# TODO — Remaining Work

Remaining tasks (in suggested order). Completed: validation, GGUF v2/converter, advanced sampling, profiling/optimization, model zoo/registry.

---

## Task 8: Gradio API, Web UI, Static Server

- Expose a Gradio-like or simple web API for chat/completion.
- Add a minimal web UI (HTML/JS) for prompt + response.
- Optional: static file server for serving the UI.

---

## Task 9: K8s, Docker, Health, Middleware, Monitoring

- Kubernetes: refine `k8s/deployment.yaml` (resources, probes, scaling).
- Docker: ensure image builds and runs; document usage.
- Health: `/health` and readiness checks; optional liveness.
- Middleware: request logging, rate limiting, auth if needed.
- Monitoring: align Prometheus metrics with deployment (scrape config, dashboards).

---

## Task 10: Tests, Coverage, CI Workflows

- Expand unit and integration tests (edge cases, error paths).
- Increase coverage (e.g. Tarpaulin) and keep CI green.
- CI: run tests, `cargo fmt`, `cargo clippy`, coverage upload; optional release builds.

---

## Task 5: Vision Encoder, Multimodal Foundation

- Add a vision encoder path (e.g. for image + text).
- Lay groundwork for multimodal inputs (image + prompt → text).

---

## Task 6: Distributed Sharding, Communication

- Model/shard distribution across multiple processes or nodes.
- Communication layer for shard coordination and result aggregation.
