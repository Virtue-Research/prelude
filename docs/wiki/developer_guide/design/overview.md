# Design Overview

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

Modular dispatch lives exclusively in `models/common/attn/mod.rs`. Model code
has zero `#[cfg]` gates for attention -- adding a backend means one file in
`attn/` and one dispatch branch in `mod.rs`.

| Backend | Feature flag | GPU requirement | Key capabilities |
|---|---|---|---|
| FA4 | `flash-attn-v4` | SM80+ | Prefill + paged decode, AOT CuTeDSL |
| FlashInfer | `flashinfer` | SM80+ (FA2) / SM90+ (FA3) | All attention paths, CUDA graph (32 graphs), plan caching |
| FA3 | `flash-attn-v3` | SM90 (Hopper) | Legacy, replaced by FlashInfer |
| FA2 | `flash-attn` | SM80+ | Legacy, replaced by FlashInfer |
| CPU | (always available) | None | Tiled BF16 (AVX-512) + F32 matmul SDPA |

**Dispatch priority:** FA4 -> FlashInfer -> FA3 -> FA2 -> CPU.
Recommended GPU build: `flashinfer-v4,onednn,deepgemm` (~98MB binary).

## 6. GEMM Backends

| Backend   | Feature flag | Target   | Notes                                         |
|-----------|-------------|----------|-----------------------------------------------|
| cuBLAS    | `cuda`      | GPU      | Default GPU GEMM                              |
| DeepGEMM  | `deepgemm`  | SM90+    | BF16, replaces cuBLAS. Decode 17%-2x faster   |
| oneDNN    | `onednn`    | CPU      | BF16 + F32 GEMM, packed weights, static link  |
| Built-in  | (default)   | CPU      | Fallback F32 GEMM when oneDNN is absent       |

## 7. KV Cache

**Paged KV cache** with `BlockManager` (vLLM-style). Block size is auto-tuned
for the active attention backend (128 with FlashInfer/FA3, 16 otherwise) and
overridable via `PRELUDE_PAGED_BLOCK_SIZE`.

**KV cache write**: custom vectorized PTX kernel (`scatter_kv_cache_flash` in
`ops/gpu/kv_cache.rs`), 128-bit float4 loads/stores.

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
