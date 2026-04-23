# Prelude Inference Engine -- Architecture Overview

Target audience: contributors and power users who want to understand the internals.

## 1. Engine Hierarchy

All inference goes through the `InferenceEngine` trait (`engine/mod.rs`).
Three implementations exist: `PseudoEngine` for testing (no model), `Engine` for
direct single-request inference, and `ScheduledEngine` which
wraps `Engine` with dynamic batching, pipelined tokenization, and streaming support.
`ScheduledEngine` is the production path.

```
InferenceEngine (trait)
  |
  +-- PseudoEngine          mock, no model
  +-- Engine                 real forward pass (own tensors)
  +-- ScheduledEngine        Engine + dynamic batching + GPU queue
```

## 2. Request Flow

HTTP requests enter the server (`prelude-server`), which is backend-agnostic --
it only knows about `InferenceEngine`. `ScheduledEngine` tokenizes on CPU (rayon),
packs work into `GpuPacket`s, and pushes them into a single FIFO GPU queue. One
dedicated OS thread drains the queue and calls into `Engine` for the forward pass.

```
HTTP Request
    |
    v
prelude-server (axum router, auth, SSE)
    |
    v
ScheduledEngine
    |  - dynamic batching (max_batch_size + wait timeout)
    |  - CPU tokenization (rayon, overlapped with GPU)
    |  - scheduler decides prefill vs decode
    v
GpuPacket  ------>  GPU Queue (single FIFO)
                        |
                        v  (one dedicated OS thread)
                    GPU Worker
                        |  - Engine::forward (prefill / decode / classify / embed)
                        |  - attention backend dispatch
                        |  - KV cache management
                        v
                    Response (token / embedding / class)
                        |
                        v
                    HTTP Response / SSE stream
```

## 3. Scheduler

A minimal continuous-batching scheduler (`scheduler.rs`), inspired by SGLang.
Operates independently from `ScheduledEngine`.

**Sequence state machine:**
```
Waiting --> Prefilling --> Decoding --> Finished
                ^             |
                +--(preempt)--+
```

**Budget constraints** (all configurable via CLI):
- `max_running_requests` -- concurrent sequences in prefill + decode
- `max_prefill_tokens` -- tokens schedulable in a single prefill step
- `max_total_tokens` -- total KV slots across all running sequences

When memory is tight, the scheduler preempts the lowest-priority decode sequence
to free KV blocks. Waiting queue ordering is pluggable via `SchedulePolicy`
(FCFS by default).

## 4. GPU Queue

All GPU-bound work is serialized through a single FIFO queue (`runtime/gpu_queue.rs`)
consumed by one dedicated OS thread (not a tokio task -- avoids `spawn_blocking`
overhead, critical when per-token decode is only a few milliseconds).

**GpuPacket variants:**

| Variant          | Description                                  | Producer           |
|------------------|----------------------------------------------|--------------------|
| `GenerateBatch`  | Prefill-only generation (max_new=1)          | Batch runtime      |
| `PrefillPaged`   | Varlen prefill + paged KV write              | Continuous runtime  |
| `DecodePaged`    | Batch decode with paged KV (Q=1 per seq)     | Continuous runtime  |
| `ClassifyBatch`  | Classification forward pass                  | Batch runtime      |
| `EmbedBatch`     | Embedding forward pass                       | Batch runtime      |

## 5. Attention Backends

Dispatch lives in `prelude-cuda::cuda_ops::paged_attention` /
`varlen_attention`. Adding a backend is one file in `crates/prelude-cuda/src/attn/`
and one arm in the dispatch.

| Backend | GPU requirement | Key capabilities |
|---|---|---|
| FA4 | SM90+ (Hopper/Blackwell) | Prefill + paged decode, AOT CuTeDSL |
| FlashInfer | SM80+ (FA2) / SM90+ (FA3) | All attention paths, CUDA graph (32 graphs), plan caching |
| Composed SDPA | Any | F32 matmul fallback in `ops/traits/attention.rs` |

**Dispatch priority (GPU):** FA4 → FlashInfer → composed SDPA.
Recommended GPU build: `--features cuda` (FA4 + FlashInfer + DeepGEMM + CUTLASS
are all enabled by default in that feature set).

## 6. GEMM Backends

GPU GEMM goes through a 3-tier dispatch in `prelude-cuda::ops::gemm`:

| Tier | Target | Notes |
|------|--------|-------|
| DeepGEMM | SM90+/SM100/SM103 BF16 | Fastest path, non-batched BF16. Skipped per-shape after first failure (thread-local cache) |
| CUTLASS 3.x | SM80+ BF16/FP16/F32 | Batched + all transposes. Runtime-gated to SM90 only when `compute_major == 9` |
| cuBLAS | All | Universal fallback. Preallocates 64 MB workspace so CUDA graph capture doesn't hit `cudaMalloc` |

CPU GEMM: oneDNN (BF16/F32, packed weights, feature `onednn`) with a pure-Rust
F32 fallback when oneDNN is absent.

## 7. KV Cache

**Paged KV cache** with `BlockManager` (vLLM-style). Block size defaults to
128 on CUDA (matches FlashInfer / FA4 TMA tile) and is overridable via
`PRELUDE_PAGED_BLOCK_SIZE`.

**KV cache write**: custom vectorized PTX kernel (`scatter_kv_cache_flash` in
`crates/prelude-cuda/src/ops/kv_cache.rs`), 128-bit float4 loads/stores.

**Prefix cache** (`prefix_cache.rs`): hash-trie structure that matches incoming
prompts against cached token blocks using hash chains. LRU eviction of leaf
blocks. Ref-counted integration with `BlockManager` so cached blocks survive
across requests.

**CUDA graph decode** (`cuda_graph.rs`): Optional graph capture for decode steps.
FlashInfer mode: 32 graphs (no seqlen bucketing), ~750ms warmup. Pre-allocated
metadata buffers for address stability across capture/replay.

## 8. CPU Optimization

Two layers, both BF16-focused:

**Pure Rust kernels (`cpu_ops/`)** -- zero external dependencies:
- RMSNorm, fused Add+RMSNorm, SiLU*Mul, RoPE
- FlashAttention-style tiled prefill + decode (online softmax)
- Runtime AVX-512 detection with scalar fallback
- NUMA-aware rayon pool with physical core binding

**oneDNN (feature `onednn`)** -- BF16/F32 GEMM only:
- Pre-packed weights (`PackedWeight::pack`) for amortized matmul
- Primitive cache keyed by (M,K,N) avoids JIT recompilation
- Static linking adds ~33MB to binary, auto-built via CMake
