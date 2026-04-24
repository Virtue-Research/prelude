# Prelude

Fast LLM inference engine in Rust. OpenAI-compatible API for generation, classification, and embeddings.

## 30-Second Start

```bash
cargo build -p prelude-server --release --features cuda      # GPU (or omit --features for CPU)
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
| Understand FlashInfer integration | [FlashInfer](flashinfer-integration.md) |
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
  prelude-server/           HTTP server (axum), OpenAI-compatible routes
  prelude-core/             Engine, scheduler, models, KV cache, tokenizer
  prelude-cpu/              CPU backend (oneDNN + AVX-512 BF16 GEMM, cpu attention)
  prelude-cuda/             CUDA backend + nested kernel crates:
    fa4/                      Flash Attention v4 AOT kernels (SM90+)
    flashinfer/               FlashInfer AOT kernels (BF16/FP16, SM80+)
    deepgemm/                 DeepGEMM BF16/FP8 GEMM (SM90+/SM100+/SM103)
    cutlass-gemm/             CUTLASS 3.x BF16/FP16/F32 GEMM (SM80+)
    quant-gemm/               llama.cpp-style quantized MMQ (GGUF)
    cula/                     cuLA linear attention (KDA, SM90/SM100)
    tvm-ffi/                  TVM-FFI shim for AOT kernels
    prelude-kernelbuild/      Shared nvcc/cutedsl build helpers
benchmark/                  Benchmark suite (bench.sh + bench_utils.py)
tests/                      Accuracy and integration tests
docs/                       This documentation
third_party/                Submodules: cutlass, DeepGEMM, flashinfer,
                              flash-attention, llama.cpp, oneDNN, cuLA, tvm-ffi
```
