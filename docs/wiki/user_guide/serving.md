# Serving and Deployment

AGInfer runs as a single binary HTTP server exposing an OpenAI-compatible API. This page covers how to serve, deploy, and tune it for different environments. AGInfer currently supports online serving only — offline batch inference is on the roadmap.

## Basic Deployment

Before running the server, you need to build the binary for your target device. See [Getting Started](setup.md#build) for the full build instructions and feature flag reference.

### GPU

```bash
# Build
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm

# Run
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --port 8000 \
  --host 0.0.0.0
```

### CPU

```bash
# Build
cargo build -p prelude-server --release --features onednn

# Run
PRELUDE_DEVICE=cpu ./target/release/prelude-server \
  --model Qwen/Qwen3-0.6B \
  --port 8000
```

### Verify the server is ready

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

The `/health` endpoint is always open — no authentication required.

For container and orchestration deployment, see the [Docker](#docker), [Nginx](#nginx-load-balancing), and [Kubernetes](#kubernetes) sections below.


## Authentication

Authentication is opt-in and configured at startup — there is no runtime toggle. By default (no `--api-key` set), the server accepts all requests without any credentials.

### Step 1: Start the server with an API key

Pass one or more `--api-key` flags when starting the server. All API endpoints will then require a valid Bearer token; `/health` and `/metrics` remain public.

```bash
./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --api-key my-secret-key \
  --api-key another-key
```

Alternatively, set the `PRELUDE_API_KEY` environment variable (merged with any `--api-key` flags):

```bash
PRELUDE_API_KEY=my-secret-key ./target/release/prelude-server \
  --model Qwen/Qwen3-4B
```

### Step 2: Include the key in every request

Once the server is started with a key, clients must pass it as a Bearer token in every request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

Requests without a valid key will receive a `401 Unauthorized` response.

## OpenAI-Compatible API

AGInfer implements the OpenAI HTTP API, making it a drop-in replacement for vLLM and SGLang clients. For request examples across all endpoints (chat, completions, embeddings, classification) and the full parameter reference, see the [API Endpoints](../api/endpoints.md) page.

For quick curl and Python SDK examples to get your first request working, see [Getting Started → First Request](setup.md#first-request).

## Parallelism and scaling

AGInfer currently runs on a single GPU per process. Native multi-GPU tensor parallelism (splitting one model across multiple GPUs) is on the roadmap. The options below achieve multi-GPU throughput by running independent single-GPU instances behind a load balancer.

### Running multiple instances for higher throughput

For higher throughput, run one process per GPU and load balance across them:

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server --model Qwen/Qwen3-4B --port 8000 &
CUDA_VISIBLE_DEVICES=1 ./target/release/prelude-server --model Qwen/Qwen3-4B --port 8001 &
```

Then use a load balancer (Nginx, HAProxy, or a cloud LB) to distribute requests across ports. See the [Nginx Load Balancing](#nginx-load-balancing) section below for a ready-to-use config.

<!-- ### Context parallel
### Data parallel
### Expert parallel -->

## Performance Tuning

See the [Features](features.md) page for advanced options that can improve inference performance.

<!-- ## Troubleshooting

**CUDA out of memory at startup.** Lower `--gpu-memory-utilization`. The server pre-allocates the KV cache pool on startup.

**High latency at low concurrency.** Disable `--max-batch-wait-ms` (set to `0`) to dispatch requests immediately without waiting to form a batch.

**FA4 AOT compilation is slow on first build.** Expected — FA4 compiles ~120 CuTeDSL kernel variants per SM architecture (10–20 minutes). Fully cached after the first build.

**401 Unauthorized errors.** Ensure the client is sending `Authorization: Bearer <key>` and the key matches one passed to `--api-key` or `PRELUDE_API_KEY`. -->

## Deployment

### Docker

!!! note "No pre-built image yet"
    A published image at `ghcr.io/opensage-agent/aginfer` is on the roadmap. For now, build the image locally from source.

#### Building from source

```bash
docker build -t aginfer:local .
```

#### GPU

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  aginfer:local \
  --model Qwen/Qwen3-4B --port 8000
```

`-v ~/.cache/huggingface:/root/.cache/huggingface` mounts your local model cache so models are not re-downloaded on every container start.

#### CPU-only

```bash
docker run \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PRELUDE_DEVICE=cpu \
  aginfer:local \
  --model Qwen/Qwen3-0.6B --port 8000
```

### Nginx Load Balancing

Nginx load balancing support is on the roadmap.

<!-- For higher throughput, run one AGInfer process per GPU and load balance across them with Nginx.

#### Start one instance per GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server --model Qwen/Qwen3-4B --port 8000 &
CUDA_VISIBLE_DEVICES=1 ./target/release/prelude-server --model Qwen/Qwen3-4B --port 8001 &
```

!!! note
    Start instances sequentially to avoid concurrent HuggingFace Hub downloads racing over the same model cache.

#### Nginx config

```nginx
upstream aginfer {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;

    location /health {
        proxy_pass http://aginfer;
    }

    location / {
        proxy_pass http://aginfer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

`least_conn` routes each new request to the instance with the fewest active connections, which naturally balances load across GPUs. -->

### Kubernetes

Kubernetes deployment manifests are on the roadmap.

## Security


### TLS termination

AGInfer does not handle TLS directly. Terminate TLS at the reverse proxy layer (Nginx, Caddy, or a cloud load balancer) and forward plain HTTP to the server.

Example Nginx TLS config:

<!-- ```nginx
server {
    listen 443 ssl;
    ssl_certificate     /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
``` -->
<!-- 
### Network isolation

In Kubernetes, restrict access to the AGInfer service using a `NetworkPolicy` so only authorized pods can reach the inference endpoint. -->

## Monitoring and Observability

### Log levels

Control verbosity via `RUST_LOG`:

```bash
RUST_LOG=info      # default — request/response summaries
RUST_LOG=debug     # verbose — scheduler decisions, KV cache stats
RUST_LOG=prelude_core=debug,prelude_server=info  # per-crate control
```

### Health and model endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `200 OK` when the server is ready |
| `GET /v1/models` | Lists the loaded model and its metadata |

These are suitable as Kubernetes liveness/readiness probes and uptime monitor targets.

## Next Steps

- [Configuration](configuration.md) — full CLI flags and environment variable reference
- [Supported Models](supported-models.md) — model compatibility and GGUF support
- [Features](features.md) - key feature usage 