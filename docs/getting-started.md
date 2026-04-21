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

oneDNN is auto-downloaded and statically linked on first build -- no manual setup.

The workspace uses `[patch.crates-io]` to override `candle-core` with a local copy at `crates/candle-core/`.
This directory must exist (symlink or copy of candle-core source).

## Build

Three main configurations:

```bash
# GPU — full stack (recommended): FlashInfer + FA4 + DeepGEMM + oneDNN
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm

# GPU — FlashInfer only (no FA4)
cargo build -p prelude-server --release --features flashinfer,onednn

# CPU only — oneDNN BF16 GEMM
cargo build -p prelude-server --release --features onednn
```

Feature flags cascade: `flashinfer`, `flash-attn-v4`, and `flash-attn` each imply `cuda`.

| Feature | What it does | GPU requirement |
|---|---|---|
| `flashinfer-v4` | FlashInfer + FA4 combined (recommended) | SM80+ |
| `flashinfer` | FlashInfer AOT attention (FA2 SM80+ / FA3 SM90+) | SM80+ |
| `flash-attn-v4` | FA4 CuTeDSL AOT attention | SM80+ |
| `deepgemm` | DeepGEMM BF16 GEMM, replaces cuBLAS. 17-2x faster decode | SM90+ |
| `onednn` | CPU BF16 GEMM via oneDNN | None (CPU) |
| `cuda` | GPU fused ops + paged KV (implied by above) | Any CUDA |

Attention dispatch priority: FA4 -> FlashInfer -> FA3 -> FA2 -> CPU.

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
  cargo run -p prelude-core --bin cpu_ops_bench --release -- quant

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
| `PRELUDE_PREFIX_CACHE_BLOCKS`   | `0`     | Prefix cache capacity (0 = disabled)             |
| `PRELUDE_PREFIX_BLOCK_SIZE`     | `64`    | Tokens per prefix cache block                    |
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

**`candle-core` not found.** The workspace uses `[patch.crates-io]` to override `candle-core` with a local copy at `crates/candle-core/`. This directory must exist (symlink or copy of candle-core source).

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
cargo run -p prelude-core --bin cpu_ops_bench --release

# Filter specific benchmark
cargo run -p prelude-core --bin cpu_ops_bench --release -- quant
cargo run -p prelude-core --bin cpu_ops_bench --release -- gemm
```

See [benchmarking.md](benchmarking.md) for the end-to-end server benchmark suite.

## Next steps

- [Server Configuration](server.md) -- all CLI flags and environment variables
- [API Reference](api.md) -- endpoint details, curl + Python examples
- [Supported Models](models.md) -- what models work and how to load them
