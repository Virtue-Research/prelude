# Prelude Inference Engine -- Architecture Overview

Target audience: contributors and power users who want to understand the internals.

## 1. Engine Hierarchy

All inference goes through the `InferenceEngine` trait (`engine/mod.rs`).
Three implementations exist: `PseudoEngine` for testing (no model), `Engine` for
direct single-request inference via Candle tensors, and `ScheduledEngine` which
wraps `Engine` with dynamic batching, pipelined tokenization, and streaming support.
`ScheduledEngine` is the production path.

```
InferenceEngine (trait)
  |
  +-- PseudoEngine          mock, no model
  +-- Engine                 real forward pass (Candle tensors)
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

Modular dispatch lives exclusively in `models/layers/attn/mod.rs`. Model code
has zero `#[cfg]` gates for attention -- adding a backend means one file in
`attn/` and one dispatch branch in `mod.rs`.

| Backend | Feature flag       | GPU requirement | Key capabilities                        |
|---------|--------------------|-----------------|----------------------------------------|
| FA4     | `flash-attn-v4`    | SM80+ (Ampere)  | Prefill only, AOT CuTeDSL, no paged KV |
| FA3     | `flash-attn-v3`    | SM90 (Hopper)   | Prefill + paged decode, prefix cache    |
| FA2     | `flash-attn`       | SM80+ (Ampere)  | Prefill + paged decode (v1 layout)      |
| CPU     | (always available) | None            | Matmul SDPA, tiled BF16 attention       |

**Dispatch priority:** FA4 (prefill) > FA3 > FA2 > CPU.
Recommended GPU build: `flash-attn-v4,flash-attn-v3` (FA4 prefill + FA3 decode).

## 6. GEMM Backends

| Backend   | Feature flag | Target   | Notes                                         |
|-----------|-------------|----------|-----------------------------------------------|
| cuBLAS    | `cuda`      | GPU      | Default GPU GEMM                              |
| DeepGEMM  | `deepgemm`  | SM90+    | BF16, replaces cuBLAS. Decode 17%-2x faster   |
| oneDNN    | `onednn`    | CPU      | BF16 + F32 GEMM, packed weights, static link  |
| Candle    | (default)   | CPU      | Fallback F32 GEMM when oneDNN is absent       |

## 7. KV Cache

**Paged KV cache** with `BlockManager` (vLLM-style). Block size is auto-tuned
for the active attention backend (128 with FA3, 16 otherwise) and overridable
via `PRELUDE_PAGED_BLOCK_SIZE`.

**Prefix cache** (`prefix_cache.rs`): hash-trie structure that matches incoming
prompts against cached token blocks using hash chains. LRU eviction of leaf
blocks. Ref-counted integration with `BlockManager` so cached blocks survive
across requests. `AssembledKvCache` avoids repeated `Tensor::cat` for the same
prefix chain. Prefix cache requires FA3 (needs `flash_attn_varlen_paged`).

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
