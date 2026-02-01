# Deployment Guide

## Docker Deployment

### Build

```bash
docker build -t bitnet-oxidized:latest .
```

### Run (with model mounted)

```bash
docker run -p 8080:8080 \
  -v /path/to/models:/models:ro \
  bitnet-oxidized:latest
```

Default command: `bitnet-oxidized serve --model /models/model.gguf --port 8080`

Override to pass a different model path or tokenizer:

```bash
docker run -p 8080:8080 -v /path/to/models:/models:ro \
  bitnet-oxidized:latest \
  bitnet-oxidized serve --model /models/my_model.gguf --port 8080
```

### Verify

After starting the container, check health and Web UI:

```bash
curl -s http://localhost:8080/health
# => OK

# Open in browser: http://localhost:8080/ui
```

### Dockerfile reference

```dockerfile
FROM rust:1.83-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/bitnet-oxidized /usr/local/bin/
RUN mkdir -p /models
WORKDIR /app
EXPOSE 8080
CMD ["bitnet-oxidized", "serve", "--model", "/models/model.gguf", "--port", "8080"]
```

## Kubernetes Deployment

Example manifests are in `k8s/deployment.yaml`: Deployment (3 replicas) and Service (ClusterIP on port 8080).

1. **Image**: Build and push to your registry, then set `image` in the Deployment (e.g. `myreg/bitnet-oxidized:latest`).

2. **Model volume**: The sample uses `emptyDir` for `/models`. For production:
   - Use a PersistentVolumeClaim (PVC) and populate it with your GGUF model, or
   - Use a CSI driver or init container to fetch the model from object storage.

3. **Apply**:

   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

4. **Expose**: For ingress, add an Ingress or LoadBalancer Service pointing to `bitnet-oxidized:8080`.

### Resource tuning

- Adjust `resources.limits` / `resources.requests` (memory/CPU) to match your model size and traffic.
- Set `replicas` and optional HPA based on CPU or custom metrics.

## Performance Tuning

- **CPU**: Set `RAYON_NUM_THREADS` to control parallelism (e.g. number of cores).
- **Native CPU**: Build with `RUSTFLAGS="-C target-cpu=native"` for best throughput (same CPU family as production).
- **Release + LTO**: Use `cargo build --release`; in `Cargo.toml` you can add:
  ```toml
  [profile.release]
  lto = true
  codegen-units = 1
  ```
- **Batch size**: The `serve` command accepts `--batch-size`; increase for higher throughput if the server supports batching.

## HTTP API and Web UI

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/`, `/health` | GET | Health check (returns `OK`). |
| `/ui` | GET | Minimal web UI: prompt textarea and Generate button; calls `/v1/completions`. |
| `/v1/completions` | POST | Text completion (JSON: `prompt`, `max_tokens`, `temperature`, etc.). |
| `/v1/chat/completions` | POST | Chat completion (JSON: `messages`, `max_tokens`, `temperature`). |
| `/v1/models` | GET | List models. |
| `/metrics` | GET | Prometheus metrics (if telemetry enabled). |

## Health and readiness

The HTTP server exposes `/` and `/health`; both return `OK`. Prefer **`/health`** for liveness and readiness probes.

In `k8s/deployment.yaml`:

- **Liveness**: `GET /health` on 8080; failure restarts the pod (initialDelaySeconds 15, period 20s, failureThreshold 3).
- **Readiness**: `GET /health` on 8080; failure removes the pod from Service endpoints (initialDelaySeconds 10, period 10s, failureThreshold 3).

Optional scaling: apply `k8s/hpa.yaml` for a HorizontalPodAutoscaler (CPU/memory targets). Requires metrics-server or custom metrics.

## Middleware and production hardening

- **Request logging**: The server uses `TraceLayer`; HTTP requests and responses are logged via `tracing` (set `RUST_LOG=info` or `debug`).
- **Rate limiting**: Not built in. For production, put a reverse proxy (nginx, Envoy, or an API gateway) in front and configure rate limits and auth there.
- **Auth**: Optional; add at the proxy or implement a custom Axum layer (e.g. API key header).

## Monitoring (Prometheus)

When the server is started with telemetry (default for `serve`), `GET /metrics` returns Prometheus text format (counters for requests and tokens).

Example **Prometheus scrape config**:

```yaml
scrape_configs:
  - job_name: bitnet-oxidized
    static_configs:
      - targets: ['bitnet-oxidized:8080']
    metrics_path: /metrics
    scrape_interval: 15s
```

For Kubernetes, use a ServiceMonitor (Prometheus Operator) or annotate the Service so your Prometheus scrapes the pods. Example pod annotation:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
```
