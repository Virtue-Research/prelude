# Getting Started

Prelude is a from-scratch LLM inference server in Rust, serving an OpenAI-compatible API.
This guide gets you from zero to a running server with your first request.

## Prerequisites

| Dependency       | Version    | Required for         |
|------------------|------------|----------------------|
| Rust (stable)    | 1.85+      | All builds           |
| CUDA Toolkit     | 12.x       | GPU builds           |
| CMake            | >= 3.18    | oneDNN (CPU BF16)    |
| Python + CUDA GPU| -          | FA4 AOT kernel compilation (first build only) |
| Python + PyTorch | 2.x        | Precision tests (`cargo test`, optional — tests skip if missing) |

oneDNN is auto-downloaded and statically linked on first build -- no manual setup.

## Build

```bash
# GPU: pulls in FlashInfer, FA4, DeepGEMM, CUTLASS, quant-gemm, cuLA, tvm-ffi
cargo build -p prelude-server --release --features cuda

# CPU only: oneDNN BF16 GEMM + AVX-512 kernels
cargo build -p prelude-server --release             # (default = cpu)

# Both
cargo build -p prelude-server --release --features full
```

| Feature | What it enables | GPU requirement |
|---|---|---|
| `cuda` | FlashInfer + FA4 + DeepGEMM + CUTLASS + quant-gemm + cuLA paged KV + CUDA graph decode | SM80+ (FA2 paths), SM90+ (FA4/FA3/DeepGEMM), SM103 tested (B300) |
| `cpu` | oneDNN BF16 / AVX-512 kernels (default) | — |
| `full` | `cpu` + `cuda` | — |

Individual GPU-backend crates (FA4, FlashInfer, DeepGEMM, …) live under
`crates/prelude-cuda/` and are all pulled in together by `--features cuda`.
The GPU attention dispatch priority at runtime is FA4 → FlashInfer →
composed F32 SDPA.

### Docker (alternative)

One Dockerfile, works with or without a GPU. `--gpus all` is ignored on machines
without the NVIDIA runtime, so the same image works everywhere.

```bash
# Build image (once per machine, or after Dockerfile changes)
docker build -t prelude-dev .

# Interactive development
docker run --gpus all -it -v $(pwd):/workspace prelude-dev bash

# CPU benchmarks (no GPU needed)
docker run -v $(pwd):/workspace prelude-dev \
  cargo bench -p prelude-core --bench cpu_ops_bench -- quant

# Tests
docker run -v $(pwd):/workspace prelude-dev \
  cargo test -p prelude-core --lib -- --test-threads=1
```

The image includes: Rust stable, CUDA 12.8 toolkit, cmake, llama.cpp (for
benchmark comparison), and Claude Code.

## Run

### GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B \
  --port 8000 \
  --gpu-memory-utilization 0.4
```

### CPU

```bash
PRELUDE_DEVICE=cpu ./target/release/prelude-server \
  --model Qwen/Qwen3-0.6B \
  --port 8000
```

### Server flags

| Flag                     | Default         | Description                              |
|--------------------------|-----------------|------------------------------------------|
| `--model <MODEL>`        | (required)      | HuggingFace repo ID or local path        |
| `--port <PORT>`          | `8000`          | Server port                              |
| `--host <HOST>`          | `0.0.0.0`       | Bind address                             |
| `--max-batch-size <N>`   | `32`            | Dynamic batch size limit                 |
| `--max-batch-wait-ms <MS>`| `5`            | Max wait before dispatching a batch      |
| `--dtype <DTYPE>`        | auto            | `f32` or `bf16` (auto: GPU=BF16, CPU=F32)|
| `--api-key <KEY>`        | (none)          | Bearer token auth (repeatable)           |

### Environment variables

| Variable                        | Default | Description                                      |
|---------------------------------|---------|--------------------------------------------------|
| `PRELUDE_DEVICE`                | `auto`  | `auto`, `cpu`, `cuda`, `cuda:N`                  |
| `CUDA_VISIBLE_DEVICES`          | (all)   | Standard CUDA device visibility                  |
| `PRELUDE_PAGED_BLOCK_SIZE`      | `128`*  | Paged KV block size (128 w/ FA3, 16 otherwise)   |
| `PRELUDE_PREFIX_CACHE_BLOCKS`   | `4096`  | Prefix cache capacity in paged KV blocks (`0` disables) |
| `PRELUDE_PREFIX_BLOCK_SIZE`     | `128`*  | Tokens per prefix cache block                    |
| `PRELUDE_API_KEY`               | (none)  | API key (merged with `--api-key` CLI args)       |
| `PRELUDE_MOCK`                  | `0`     | `1` to use mock engine (no model needed)         |
| `RUST_LOG`                      | `info`  | Log level (`prelude_core=debug` for verbose)     |

## First request

### Chat completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "What is Rust?"}],
    "max_tokens": 64
  }'
```

With streaming:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "What is Rust?"}],
    "max_tokens": 64,
    "stream": true
  }'
```

### Text completion

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "prompt": "The capital of France is",
    "max_tokens": 32
  }'
```

### Embeddings

Requires an embedding model (e.g., `Qwen/Qwen3-Embedding`):

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding",
    "input": ["Search query: what is deep learning?", "Search query: how to train a model"]
  }'
```

### Classification

Requires a classifier model (e.g., `Qwen/Qwen3-Reranker`):

```bash
curl -s http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker",
    "input": ["This is a positive review", "This is a negative review"]
  }'
```

## Verify

Health check (always open, no auth required):

```bash
curl -s http://localhost:8000/health
```

List loaded models:

```bash
curl -s http://localhost:8000/v1/models
```

## Troubleshooting

**FA4 AOT compilation is slow on first build.** Flash Attention v4 compiles ~120 CuTeDSL kernel variants per SM architecture. This takes 10-20 minutes on first build but is cached afterwards. Skip with `flash-attn-v3` only if you don't need FA4.

**CMake version too old for oneDNN.** oneDNN requires CMake >= 3.18. On Ubuntu 20.04, install via `pip install cmake` or use the Kitware PPA.

**CUDA out of memory.** Lower `--gpu-memory-utilization` (default 0.4). This controls the fraction of free GPU memory used for paged KV cache.

## Development

### Test

```bash
# All library tests (single-threaded to avoid GemmPool conflicts)
cargo test -p prelude-core --lib -- --test-threads=1

# Specific module
cargo test -p prelude-core --lib -- quant
cargo test -p prelude-core --lib -- linear
```

### Benchmark

```bash
# CPU kernel benchmarks (quantized, GEMM, attention, etc.)
cargo bench -p prelude-core --bench cpu_ops_bench

# Filter specific benchmark
cargo bench -p prelude-core --bench cpu_ops_bench -- quant
cargo bench -p prelude-core --bench cpu_ops_bench -- gemm
```

See [benchmarking.md](benchmarking.md) for the end-to-end server benchmark suite.

## Next steps

- [Server Configuration](server.md) -- all CLI flags and environment variables
- [API Reference](api.md) -- endpoint details, curl + Python examples
- [Supported Models](models.md) -- what models work and how to load them
