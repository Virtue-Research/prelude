# AGInfer Documentation

Welcome to the AGInfer documentation!

AGInfer is a high-performance LLM inference engine written in Rust, optimized for prefill throughput with an OpenAI-compatible API.

## 30-Second Start

```bash
# Build
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm

# Run
CUDA_VISIBLE_DEVICES=0 ./target/release/prelude-server --model Qwen/Qwen3-4B --port 8000

# Query
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'
```

## Documentation Index

### User Guide
- [Getting Started](user_guide/Getting-Started.md) — Build, install, and run AGInfer for the first time
- [Configuration](user_guide/Configuration.md) — CLI flags and environment variable reference
- [Supported Models](user_guide/Supported-Models.md) — Model compatibility matrix

### Developer Guide
- [Project Structure](developer_guide/Project-Structure.md) — Crate layout and module overview
- [Architecture](developer_guide/Architecture.md) — Engine design, scheduler, attention, and GEMM backends
- [Adding a Model](developer_guide/Adding-a-Model.md) — Step-by-step guide to integrating a new model architecture

### API Reference
- [Overview](api/index.md) — API design and authentication
- [Endpoints](api/Endpoints.md) — Full endpoint reference (chat, completions, embeddings, classify)

### Benchmarks
- [Results](benchmarks/Results.md) — Latest throughput and latency benchmarks vs vLLM and SGLang
- [Running Benchmarks](benchmarks/Running.md) — How to reproduce benchmark results

### Community
- [Contributing](community/Contributing.md) — How to contribute code, report bugs, and request features

---

## Performance

**GPU (H200, Qwen3-4B, 512-token inputs)**

| Engine | Throughput (tok/s) | P50 Latency | P95 Latency |
|--------|-------------------|-------------|-------------|
| **AGInfer** | **95,590** | **15.4ms** | **21.1ms** |
| vLLM | 68,780 | 18.1ms | 27.9ms |
| SGLang | 77,700 | 20.8ms | 26.2ms |

- **1.39× faster** than vLLM · **1.23× faster** than SGLang at peak throughput (c=96)

## Key Features

- **Pure Rust** — built on Candle, no Python/PyTorch dependency
- **OpenAI-compatible API** — drop-in replacement for vLLM/SGLang clients
- **Continuous batching** with paged KV cache, prefix caching, and CUDA graph decode
- **FlashInfer + FA4** — AOT attention with plan caching and 32-graph CUDA decode
- **GPU + CPU inference** — BF16 via FlashInfer/FA4 (GPU) or oneDNN + AVX-512 (CPU)
- **DeepGEMM** — SM90+ BF16 GEMM, up to 2× faster than cuBLAS
- **GGUF support** — auto-detected from HuggingFace Hub, llama.cpp FFI backend
- **Hybrid model support** — Qwen3.5 and Qwen3-Next (DeltaNet + attention + MoE)
- **Single binary** — 118MB, links only `libcuda.so.1`, zero CUDA Toolkit runtime dependency

---

*Check the [GitHub repository](https://github.com/Virtue-Research/prelude) for the latest code.*
