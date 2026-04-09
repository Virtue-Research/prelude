# Prelude

Fast LLM inference engine in Rust. OpenAI-compatible API for generation, classification, and embeddings.

## 30-Second Start

```bash
cargo build -p prelude-server --release --features full
./target/release/prelude-server --model Qwen/Qwen3-0.6B
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

## I want to...

| Goal | Page |
|------|------|
| Build and run Prelude for the first time | [Getting Started](getting-started.md) |
| See all CLI flags and environment variables | [Server Configuration](server.md) |
| Call the API (completions, chat, embeddings, classify) | [API Reference](api.md) |
| Check which models are supported | [Supported Models](models.md) |
| Understand how the engine works internally | [Architecture](architecture.md) |
| Run performance benchmarks | [Benchmarking](benchmark.md) |
| See latest benchmark results | [Results](results.md) |
| Add a new model architecture | [Skill: Adding a Model](skills/adding-a-model.md) |

## Key Features

- **Pure Rust** -- built on Candle, no Python/PyTorch dependency
- **OpenAI-compatible API** -- drop-in replacement for vLLM/SGLang clients
- **Continuous batching** with paged KV cache, prefix caching, and CUDA graph decode
- **FlashInfer + FA4** -- AOT attention with plan caching and 32-graph CUDA decode
- **GPU + CPU inference** -- BF16 via FlashInfer/FA4 (GPU) or oneDNN + AVX-512 (CPU)
- **DeepGEMM** -- SM90+ BF16 GEMM, 17%-2x faster decode than cuBLAS
- **GGUF support** -- auto-detected from HuggingFace Hub, llama.cpp FFI backend
- **Hybrid model support** -- Qwen3.5 and Qwen3-Next (DeltaNet + attention + MoE)

## Project Layout

```
crates/
  prelude-server/        HTTP server (axum), OpenAI-compatible routes
  prelude-core/          Engine, scheduler, models, attention backends, KV cache
  prelude-cpu/           CPU kernels (AVX-512, oneDNN BF16 GEMM)
  prelude-cuda/          GPU backend — contains all GPU kernel sub-crates:
    flashinfer/          FlashInfer AOT kernels (attention, norm, activation, sampling)
    fa4/                 Flash Attention v4 AOT kernels (SM80+)
    deepgemm/            DeepGEMM BF16/FP8 GEMM (SM90+)
    cutlass-gemm/        CUTLASS BF16 GEMM (SM80+)
    quant-gemm/          Quantized GEMM kernels
    cula/                cuLA (custom CUDA linear algebra)
    tvm-ffi/             TVM FFI for FlashInfer kernel dispatch
benchmark/               Benchmark + profiling (bench.sh, profile.sh)
tests/                   Accuracy tests (PPL, logprobs)
docs/                    This documentation
```
