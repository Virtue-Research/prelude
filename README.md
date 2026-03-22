<p align="center">
  <h1 align="center">Prelude</h1>
  <p align="center">Fast LLM inference engine in Rust. Optimized for prefill throughput.</p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#supported-models">Models</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#architecture">Architecture</a>
</p>

---

Prelude is a from-scratch LLM inference server written in Rust, built on [candle](https://github.com/huggingface/candle)
tensors. It serves an OpenAI-compatible API for text generation, classification, and embeddings with focus on prefill
throughput and low per-token latency.

## Performance

**GPU (H200, Qwen3-0.6B, BF16)**

| Task           | Concurrency |  Throughput   | Latency P95 |
|----------------|:-----------:|:-------------:|:-----------:|
| Generation     |      8      |  1,768 tok/s  |    138ms    |
| Generation     |     16      |  1,822 tok/s  |    285ms    |
| Classification |     16      | 2,060 items/s |    166ms    |
| Embedding      |     16      | 1,983 items/s |    168ms    |

TPOT (time per output token): **3.2ms** at c=1, **3.8ms** at c=16.

**GPU (H200, Qwen3-4B, prefill throughput vs vLLM / SGLang)**

| Engine  | Peak Throughput | Tokens/s        | vs vLLM   | vs SGLang |
|---------|:--------------:|:---------------:|:---------:|:---------:|
| Prelude | 186.7 req/s    | **95,590 t/s**  | **1.39x** | **1.23x** |
| vLLM    | 134.5 req/s    | 68,864 t/s      | —         | —         |
| SGLang  | 151.5 req/s    | 77,568 t/s      | —         | —         |

Peak at concurrency=96 (512-token inputs, max_tokens=1). Tokens/s = req/s × 512.

**Latency at c=1 — single request (H200, Qwen3-4B)**

| Engine  | P50        | P95        |
|---------|:----------:|:----------:|
| Prelude | **15.4ms** | **21.1ms** |
| vLLM    | 18.1ms     | 27.9ms     |
| SGLang  | 20.8ms     | 26.2ms     |

**CPU (Xeon 8480+, Qwen3-0.6B, BF16 via oneDNN)**

| Benchmark              |   Prelude   |   SGLang   |  Speedup  |
|------------------------|:-----------:|:----------:|:---------:|
| Prefill (128 tok, c=1) |  3,629 t/s  | 2,298 t/s  | **1.58x** |
| Prefill (128 tok, c=4) |  5,710 t/s  | 3,970 t/s  | **1.44x** |
| Decode (32 in, 32 out) | 109.6 out/s | 57.1 out/s | **1.92x** |

## Quick Start

### Prerequisites

- **Rust** (stable, 1.85+)
- **CUDA Toolkit** (for GPU)
- **CMake** >= 3.18 (for oneDNN CPU backend)

### Build

```bash
# GPU — full stack (recommended): FlashInfer + FA4 + DeepGEMM + oneDNN
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm

# GPU — FlashInfer only (no FA4)
cargo build -p prelude-server --release --features flashinfer,onednn

# CPU only with oneDNN BF16 GEMM
cargo build -p prelude-server --release --features onednn

# Distribution build (smallest binary, slower compilation)
cargo build -p prelude-server --profile dist --features flashinfer-v4,onednn,deepgemm
```

### Run

```bash
# GPU
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server \
  --model Qwen/Qwen3-4B --port 8000 --gpu-memory-utilization 0.4

# CPU
PRELUDE_DEVICE=cpu ./target/release/prelude-server \
  --model Qwen/Qwen3-0.6B --port 8000
```

### Query

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

## Supported Models

| Architecture   | Models                           | GPU Backend         | Continuous Batching  |
|----------------|----------------------------------|---------------------|----------------------|
| Dense          | Qwen3 (0.6B-32B)                 | FlashInfer / FA4    | Yes                  |
| MoE            | Qwen3-MoE (30B-A3B)              | FlashInfer / FA4    | Yes                  |
| Hybrid         | Qwen3.5 (0.8B-27B, dense + MoE)  | FlashInfer / FA4    | Yes (decode batch=1) |
| Hybrid         | Qwen3-Next (80B-A3B)             | FlashInfer / FA4    | Yes (decode batch=1) |
| Classification | Qwen3-Reranker, Gemma3           | FlashInfer / FA4    | Yes                  |
| Embedding      | Qwen3-Embedding                  | FlashInfer / FA4    | Yes                  |
| Quantized      | GGUF (Qwen3, LLaMA, Gemma, Phi3) | CUDA / CPU          | No                   |

Hybrid models use DeltaNet (linear recurrent attention) for most layers with standard attention every 4th layer.

## API

OpenAI-compatible endpoints:

| Endpoint                    | Description                         |
|-----------------------------|-------------------------------------|
| `GET /health`               | Health check                        |
| `GET /v1/models`            | List models                         |
| `POST /v1/completions`      | Text completion (streaming + batch) |
| `POST /v1/chat/completions` | Chat completion (streaming + batch) |
| `POST /v1/embeddings`       | Text embeddings                     |
| `POST /classify`            | Sequence classification             |

Supports `logprobs`, `top_logprobs`, `stop` sequences, and `stream` mode. Compatible with OpenAI SDK, vLLM clients, and
SGLang clients.

## Architecture

```
Request -> Dynamic Batcher -> GPU Queue (FIFO) -> GPU Worker -> Response
                                    |
                            Model Forward Pass
                       (pluggable attention backend)
```

**Attention backends** are modular -- each in its own file under `attn/`:

| Backend            | GPU                       | Features                                                 |
|--------------------|---------------------------|----------------------------------------------------------|
| FlashInfer         | SM80+ (FA2) / SM90+ (FA3) | AOT, all attention paths, CUDA graph (32 graphs, no seqlen bucketing), plan caching |
| Flash Attention v4 | SM80+ (CuTeDSL AOT)      | Prefill + paged KV, multi-arch, statically linked        |
| CPU                | Any                       | Tiled BF16 (AVX-512) + F32 matmul SDPA                   |

Dispatch priority: FA4 → FlashInfer → CPU. Adding a new backend = one file + one dispatch line.

**GEMM backends**:

| Backend       | GPU            | Features                                                                         |
|---------------|----------------|----------------------------------------------------------------------------------|
| candle/cuBLAS | Any CUDA       | Default, requires CUDA Toolkit                                                   |
| DeepGEMM      | SM90+ (Hopper) | BF16 GEMM, statically linked, no cuBLAS needed. Decode 17%~2x faster than cuBLAS |
| oneDNN        | CPU            | BF16 brgemm via oneDNN micro-kernel                                              |

**Key components:**

- **Continuous batching scheduler** (SGLang-inspired) with preemption and budget constraints
- **Paged KV cache** with block manager and automatic prefix caching
- **Per-backend cache layout**: flash layout (FlashInfer/FA4) or v1 layout (FA2 legacy), selected at compile time
- **Fused CUDA kernels**: QKNorm+RoPE, SiLU*Mul, Add+RMSNorm, fused KV cache write
- **Pure Rust CPU kernels**: AVX-512 BF16 attention, RMSNorm, RoPE, SiLU (zero external dependencies)
- **oneDNN**: BF16 GEMM for CPU inference (auto-built, statically linked)

## Configuration

### Server flags

```
--model <MODEL>          HuggingFace repo ID or local path (auto-detects GGUF)
--port <PORT>            Server port (default: 8000)
--max-batch-size <N>     Dynamic batch size limit (default: 128)
--max-batch-wait-ms <MS> Max wait before dispatching batch (default: 5)
--api-key <KEY>          Bearer token auth (repeatable)
```

### Environment variables

| Variable                      | Default | Description                             |
|-------------------------------|---------|-----------------------------------------|
| `PRELUDE_DEVICE`              | `auto`  | Device: `auto`, `cpu`, `cuda`, `cuda:N` |
| `PRELUDE_PAGED_BLOCK_SIZE`    | `128`*  | Paged KV cache block size (128 with FA3, 16 otherwise) |
| `PRELUDE_PREFIX_CACHE_BLOCKS` | `0`     | Prefix cache capacity (0 = disabled)    |
| `PRELUDE_PREFIX_BLOCK_SIZE`   | `64`    | Tokens per prefix cache block           |
| `RUST_LOG`                    | `info`  | Log level                               |

## License

MIT
