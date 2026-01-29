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

### Dockerfile reference

```dockerfile
FROM rust:1.75-bookworm AS builder
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

## Health and readiness

The HTTP server exposes `/` for health. The Kubernetes manifests use:

- **Liveness**: `GET /` on 8080; failure restarts the pod.
- **Readiness**: `GET /` on 8080; failure removes the pod from Service endpoints.

Use `/metrics` (if implemented) for monitoring and autoscaling.
