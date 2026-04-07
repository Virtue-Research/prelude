# Getting Started

This guide gets you from zero to a running AGInfer server with your first request.

## Prerequisites

| Dependency | Version | Required for |
|---|---|---|
| Rust (stable) | 1.85+ | All builds |
| CUDA Toolkit | 12.x | GPU builds |
| CMake | >= 3.18 | oneDNN CPU backend |
| Python + CUDA GPU | — | FA4 AOT kernel compilation (first build only) |
| Python + PyTorch | 2.x | Precision tests (`cargo test`, optional) |

oneDNN is auto-downloaded and statically linked on first build — no manual setup required.

## Build

!!! tip "Initialize submodules first"
    After cloning, initialize the required git submodules (FlashInfer, FA4, DeepGEMM, CUTLASS):

    ```bash
    git submodule update --init --recursive
    ```

    Without this step, `cargo build` will fail with a submodule-not-found error.

Three main configurations:

```bash
# GPU — full stack (recommended): FlashInfer + FA4 + DeepGEMM + oneDNN
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm

# GPU — FlashInfer only (no FA4)
cargo build -p prelude-server --release --features flashinfer,onednn

# CPU only — oneDNN BF16 GEMM
cargo build -p prelude-server --release --features onednn

# GGUF models (auto-detected, no extra flags needed)
cargo build -p prelude-server --release --features onednn
```

Feature flags cascade: `flashinfer`, `flash-attn-v4`, and `flash-attn` each imply `cuda`.

| Feature | What it does | GPU requirement |
|---|---|---|
| `flashinfer-v4` | FlashInfer + FA4 combined (recommended) | SM80+ |
| `flashinfer` | FlashInfer AOT attention (FA2 SM80+ / FA3 SM90+) | SM80+ |
| `flash-attn-v4` | FA4 CuTeDSL AOT attention | SM80+ |
| `deepgemm` | DeepGEMM BF16 GEMM — up to 2× faster than cuBLAS | SM90+ |
| `onednn` | CPU BF16 GEMM via oneDNN | None (CPU) |
| `cuda` | GPU fused ops + paged KV cache (implied by above) | Any CUDA |

Attention dispatch priority: FA4 → FlashInfer → FA3 → FA2 → CPU fallback.

GGUF models are auto-detected from HuggingFace Hub or local `.gguf` files — no extra build flags required beyond `onednn`.

!!! note "FA4 first build"
    FA4 AOT compilation compiles ~120 CuTeDSL kernel variants per SM architecture. This takes 10–20 minutes on first build but is fully cached afterwards. Use `flashinfer` only if you want to skip FA4.

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

### Key server flags

| Flag | Default | Description |
|---|---|---|
| `--model <MODEL>` | (required) | HuggingFace repo ID or local path |
| `--port <PORT>` | `8000` | Server port |
| `--host <HOST>` | `0.0.0.0` | Bind address |
| `--max-batch-size <N>` | `32` | Dynamic batch size limit |
| `--max-batch-wait-ms <MS>` | `5` | Max wait before dispatching a batch |
| `--gpu-memory-utilization <F>` | `0.4` | Fraction of free GPU memory for KV cache |
| `--dtype <DTYPE>` | auto | `f32` or `bf16` (auto: GPU=BF16, CPU=F32) |
| `--api-key <KEY>` | (none) | Bearer token auth (repeatable) |

See [Configuration](configuration.md) for the full flag and environment variable reference.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PRELUDE_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |
| `CUDA_VISIBLE_DEVICES` | (all) | Standard CUDA device visibility |
| `PRELUDE_PAGED_BLOCK_SIZE` | `128`* | Paged KV block size (128 w/ FA3, 16 otherwise) |
| `PRELUDE_PREFIX_CACHE_BLOCKS` | `0` | Prefix cache capacity (0 = disabled) |
| `PRELUDE_PREFIX_BLOCK_SIZE` | `64` | Tokens per prefix cache block |
| `PRELUDE_API_KEY` | (none) | API key (merged with `--api-key` CLI args) |
| `PRELUDE_MOCK` | `0` | `1` to use mock engine (no model needed) |
| `RUST_LOG` | `info` | Log level (`prelude_core=debug` for verbose) |

## First Request

=== "Python"

    Install the OpenAI Python SDK if you haven't already:

    ```bash
    pip install openai
    ```

    ### Chat completion

    ```python
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "What is Rust?"}],
        max_tokens=64,
    )
    print(response.choices[0].message.content)
    ```

    With streaming:

    ```python
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "What is Rust?"}],
        max_tokens=64,
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    ```

    ### Text completion

    ```python
    response = client.completions.create(
        model="Qwen/Qwen3-4B",
        prompt="The capital of France is",
        max_tokens=32,
    )
    print(response.choices[0].text)
    ```

    ### Embeddings

    Requires an embedding model (e.g., `Qwen/Qwen3-Embedding`):

    ```python
    response = client.embeddings.create(
        model="Qwen/Qwen3-Embedding",
        input=["Search query: what is deep learning?", "Search query: how to train a model"],
    )
    print(response.data[0].embedding)
    ```

=== "curl"

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

Health check (no auth required):

```bash
curl -s http://localhost:8000/health
```

List loaded models:

```bash
curl -s http://localhost:8000/v1/models
```

## Troubleshooting

**CMake version too old for oneDNN.** oneDNN requires CMake >= 3.18. On Ubuntu 20.04, install via `pip install cmake` or use the Kitware PPA.

**CUDA out of memory.** Lower `--gpu-memory-utilization` (default 0.4). This controls the fraction of free GPU memory reserved for the paged KV cache.

**FA4 compilation hangs.** This is expected on first build — FA4 compiles ~120 kernel variants per SM architecture and can take 10–20 minutes. It is fully cached after the first build.

## Next Steps

- [Serving and Deployment](serving.md) — deployment options and production setup
- [Configuration](configuration.md) — full CLI flags and environment variable reference
- [Supported Models](supported-models.md) — model compatibility matrix
- [Features](features.md) - key feature usage 