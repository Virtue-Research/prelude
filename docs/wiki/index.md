# AGInfer Documentation

Welcome to the AGInfer documentation!

AGInfer is a high-performance LLM inference engine written in Rust, optimized for scheduling throughput, built on a modular design for extensibility to new architectures, and tailored for agentic serving workloads.

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
- [Getting Started](user_guide/setup.md) — Build, install, and run AGInfer for the first time
- [Serving and Deployment](user_guide/serving.md) — How to serve and deploy AGInfer
- [Configuration](user_guide/configuration.md) — CLI flags and environment variable reference
- [Supported Models](user_guide/supported-models.md) — Model compatibility matrix
- [Features](user_guide/features.md) — Overview of supported features

### Developer Guide
- [General](developer_guide/general.md) — Developer overview and contribution guidelines
- [Adding a Model](developer_guide/adding-a-model.md) — Step-by-step guide to integrating a new model architecture
- [Integration](developer_guide/integration.md) — Dynamo backend, Mooncake transport, and RL training integration

#### Design
- [Overview](developer_guide/design/overview.md) — High-level design overview
- [Scheduler](developer_guide/design/schedular.md) — Scheduler design
- [Models](developer_guide/design/models.md) — Model architecture internals
- [Modules and Operators](developer_guide/design/ops.md) — Module and operator design
- [Devices](developer_guide/design/devices.md) — Device abstraction and backends

### API Reference
- [Overview](api/index.md) — API design and authentication
- [Endpoints](api/endpoints.md) — Full endpoint reference (chat, completions, embeddings, classify)

### CLI Reference
- [CLI Reference](cli/index.md) — Command-line interface reference

### Benchmarking
- [Results](benchmarks/results.md) — Latest throughput and latency benchmarks vs vLLM and SGLang
- [Running Benchmarks](benchmarks/running.md) — How to reproduce benchmark results

### Community
- [Contributing](community/contributing.md) — How to contribute code, report bugs, and request features

---

## Performance Highlight

### Prefill throughout 

| Engine | Throughput (tok/s) | P50 Latency | P95 Latency |
|--------|-------------------|-------------|-------------|
| **AGInfer** | **95,590** | **15.4ms** | **21.1ms** |
| vLLM | 68,780 | 18.1ms | 27.9ms |
| SGLang | 77,700 | 20.8ms | 26.2ms |

GPU (H200, Qwen3-4B, 512-token inputs)

## Key Features

- **Rust-native** — modular, memory-safe, and designed for clean abstractions and extensibility
- **OpenAI-compatible API** — drop-in replacement for vLLM/SGLang clients
- **Agent-aware scheduling** — batching and KV cache management optimized for agentic workloads, including KV reuse and multi-turn session scheduling
- **Kernel optimizations** — purpose-built kernels for GEMM (DeepGEMM, cuBLAS, oneDNN), attention (FlashInfer, FA4), and fused ops
- **Multi-device support** — BF16/FP32 inference on CUDA GPUs (single and multi-GPU) and CPUs (AVX-512 via oneDNN), with automatic device detection
- **Extensive model support** — autoregressive (Qwen3, Llama, Mistral, Gemma), diffusion (FLUX), and hybrid architectures (Qwen3-Next with DeltaNet linear attention + MoE, Mamba, RWKV); GGUF models via llama.cpp FFI
- **Latest techniques** — Keep up with the latest techniques: CUDA graph decode, prefix caching, paged KV cache, speculative decoding, disaggregated prefill, and continuous batching
---

*Check the [GitHub repository](https://github.com/opensage-agent/aginfer) for the latest code.*
