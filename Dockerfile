# BitNet-Oxidized production image
# Multi-stage: build then minimal runtime

FROM rust:1.83-bookworm AS builder
WORKDIR /app

COPY . .
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/bitnet-oxidized /usr/local/bin/
RUN mkdir -p /models

WORKDIR /app
EXPOSE 8080

# Default: serve on 8080; override CMD to pass --model /models/model.gguf etc.
CMD ["bitnet-oxidized", "serve", "--model", "/models/model.gguf", "--port", "8080"]
