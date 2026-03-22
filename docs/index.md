# Prelude

Fast LLM inference engine in Rust. OpenAI-compatible API for generation, classification, and embeddings.

## 30-Second Start

```bash
cargo build -p prelude-server --release --features flash-attn-v3
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
| Run performance benchmarks | [Benchmarking](benchmarking.md) |
| Add a new model architecture | [Skill: Adding a Model](skills/adding-a-model.md) |
| Add an attention backend | [Skill: Adding an Attention Backend](skills/adding-an-attention-backend.md) |
| Write a custom CUDA kernel | [Skill: Writing CUDA Ops](skills/writing-cuda-ops.md) |

## Key Features

- **Pure Rust** -- built on Candle, no Python/PyTorch dependency
- **OpenAI-compatible API** -- drop-in replacement for vLLM/SGLang clients
- **Continuous batching** with paged KV cache and automatic prefix caching
- **Multiple attention backends** -- Flash Attention v4/v3/v2 + CPU, selected at compile time
- **GPU + CPU inference** -- BF16 via Flash Attention (GPU) or oneDNN + AVX-512 (CPU)
- **First-class classify & embed** -- batched pooling with dedicated endpoints, not generation afterthoughts
- **Hybrid model support** -- Qwen3.5 and Qwen3-Next (DeltaNet + attention + MoE)

## Project Layout

```
crates/
  prelude-server/        HTTP server (axum), OpenAI-compatible routes
  prelude-core/          Engine, scheduler, models, attention backends, KV cache
  candle-flash-attn-v3/  Flash Attention v3 bindings (Hopper)
  prelude-flash-attn-v4/ Flash Attention v4 AOT kernels (SM80+)
  candle-paged-attn/     Paged attention v1/v2 kernels
  prelude-deepgemm/      DeepGEMM BF16 GEMM (SM90+)
  onednn-ffi/            oneDNN FFI for CPU BF16 GEMM
benchmark/               Benchmark tool + presets (see benchmarking.md)
docs-new/                This documentation
  skills/                Step-by-step contributor guides
```
