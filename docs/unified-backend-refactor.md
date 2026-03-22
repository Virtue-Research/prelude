# Unified Backend Refactor

> Design doc for consolidating ScheduledEngine into a cleaner two-scheduler + single GPU queue architecture.
> Supersedes the old multi-path forward dispatch.

## Goals

1. **Two scheduling paths only**: `max_new_tokens = 1` (Batch) and `max_new_tokens > 1` (Continuous Batch, paged required)
2. **Remove the fallback**: no more "max_new > 1 without paged-attn" path — paged is mandatory for multi-token generation
3. **Unified GPU queue**: both schedulers emit `GpuPacket`s into a single FIFO queue; one GPU worker consumes them
4. **Model is a dumb executor**: the packet fully describes the work (prefill vs prefill+paged vs decode+paged). Model just runs what it's told — no decision logic inside

## Architecture Overview

See `docs/architecture.html` for the visual diagram.

```
HTTP Request
    │
    ▼
GenerateRequest / EmbedRequest / ClassifyRequest
    │
    ▼
┌─────────────────────────────┐
│     max_new_tokens ?        │
├──────────┬──────────────────┤
│  = 1     │  > 1             │
│          │  (paged required) │
▼          ▼
Batch      Continuous Batch
Scheduler  Scheduler
│          │
▼          ▼
└──► GpuPacket::Prefill      ◄── (from Batch Scheduler, no paged KV)
     GpuPacket::PrefillWithKV ◄── (from Continuous Scheduler, allocate paged KV)
     GpuPacket::Decode       ◄── (from Continuous Scheduler, paged KV decode)
          │
          ▼
    ┌─────────────┐
    │  GPU Queue   │  (single FIFO)
    │  (FIFO)      │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  GPU Worker  │  (single-threaded, pop → execute → next)
    └──────┬──────┘
           ▼
    Model Forward (dumb executor — packet tells it what to do):
      1. Prefix Cache Lookup        [always]
      2. Varlen Forward             [always]
      3. Paged KV Write             [if packet is PrefillWithKV or Decode]
      4. Post-processing            [always]
    No branching on request params — the packet variant encodes the intent.
           │
           ▼
    ┌──────┴──────┐
    │             │
    ▼             ▼
   Done      Continue Decode
   (return)   (→ Continuous Scheduler → new GpuPacket::Decode → re-enqueue)
```

## What Changes

### Removed


| Component                                                               | Reason                                                          |
| ----------------------------------------------------------------------- | --------------------------------------------------------------- |
| Standard concat KV forward path                                         | Replaced by unified varlen path                                 |
| `max_new > 1` without paged fallback in `select_generation_scheduler()` | Paged is now mandatory for multi-token                          |
| CUDA Graph decode path (`forward_decode_graph`)                         | Can be re-added later as an internal optimization within varlen |
| Multiple forward entry points on models                                 | Single `forward()` via `BatchAttnContext`                       |
| Direct GPU dispatch from schedulers                                     | Schedulers now only produce `GpuPacket`s                        |


### Added


| Component                                                 | Purpose                                                                             |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `GpuPacket` enum (`Prefill` / `PrefillWithKV` / `Decode`) | Unified work unit for GPU queue — packet variant fully describes the execution mode |
| GPU Queue (FIFO channel)                                  | Decouples scheduling from execution                                                 |
| GPU Worker loop                                           | Single consumer that pops and executes packets                                      |
| Mandatory prefix cache in forward pipeline                | Always checked, no-op if miss                                                       |


### Modified


| Component           | Change                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `ScheduledEngine`   | Remove `select_generation_scheduler()` fallback; route `max_new > 1` exclusively to Continuous Scheduler     |
| `BatchRuntime`      | Output `GpuPacket::Prefill` to GPU queue instead of directly spawning GPU work                               |
| `ContinuousRuntime` | Output `GpuPacket::PrefillWithKV` and `GpuPacket::Decode` to GPU queue; receive results back for decode loop |
| Model `forward()`   | Dumb executor — packet variant determines whether to write paged KV. Model has zero decision logic.          |


## Key Design Decisions

### Why a single GPU queue?

- GPU is the bottleneck — serializing all work through one FIFO ensures no contention and predictable ordering
- Schedulers focus on *what* to compute; the GPU worker focuses on *how* to execute
- Makes it trivial to add priority scheduling later (swap FIFO for priority queue)

### Why remove the non-paged fallback?

- Simplifies the codebase significantly — one path to test and maintain
- Paged attention is required for efficient multi-token decode anyway
- CPU-only users with `max_new > 1` should still work via a minimal paged implementation (block size = sequence length)

### Why always varlen?

- Varlen handles both single-sequence and multi-sequence batches
- When flash-attn-v3 is available, it uses the fused kernel; otherwise, standard varlen attention
- Eliminates the need for separate `forward()` vs `forward_varlen()` vs `forward_decode_paged()` methods

## File-Level Changes

### `crates/prelude-core/src/runtime/`


| File                    | Action                                                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `engine.rs`             | Remove `GenerationSchedulerRoute` fallback. Add `GpuPacket` enum. Create GPU queue channel (`mpsc`). Spawn GPU worker task.                         |
| `batch_runtime.rs`      | Change GPU dispatch: instead of `spawn_blocking(engine.generate_prepared_batch())`, send `GpuPacket::Prefill` to queue. Receive result via oneshot. |
| `continuous_runtime.rs` | Remove `#[cfg(all(feature = "paged-attn", feature = "flash-attn-v3"))]` gate — always compiled. Send `GpuPacket::Prefill`/`Decode` to queue.        |
| `mod.rs`                | Export `GpuPacket`, GPU queue types.                                                                                                                |


### `crates/prelude-core/src/engine/`


| File                | Action                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `core.rs`           | Remove `ExecutionKind` branching. Unify into single `execute_packet()` method that runs the serial pipeline. |
| `inference_impl.rs` | Consolidate prefill-only and multi-token paths. Always call prefix cache → varlen → conditional paged write. |


### `crates/prelude-core/src/task/`


| File          | Action                                                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `generate.rs` | Simplify `generate_prepared_batch()` — remove separate CPU/GPU/prefill-only/multi-token dispatch. Single path through serial pipeline. |


### `crates/prelude-core/src/models/`


| File                             | Action                                                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `model_forward.rs`               | Keep `ModelForward::forward()` as the single entry point. Remove any separate decode-specific trait methods if they exist.      |
| Architecture impls (qwen3, etc.) | Consolidate internal forward dispatch. The model receives `BatchAttnContext` and handles varlen + optional paged KV internally. |


## GpuPacket Definition (Draft)

```rust
enum GpuPacket {
    /// Prefill-only (max_new=1): varlen forward, no paged KV, discard KV after use
    Prefill {
        sequences: Vec<PreparedSequence>,
        result_tx: oneshot::Sender<Vec<PrefillResult>>,
    },
    /// Prefill with paged KV (max_new>1): varlen forward + write KV to paged cache
    /// The scheduler has already allocated blocks — model just writes to them
    PrefillWithKV {
        sequences: Vec<PreparedSequence>,
        block_tables: Vec<Vec<u32>>,      // pre-allocated by scheduler
        result_tx: oneshot::Sender<Vec<PrefillWithKVResult>>,
    },
    /// Decode step: single-token forward with paged KV read/write
    Decode {
        sequences: Vec<DecodeSequence>,   // each carries its block_table
        result_tx: oneshot::Sender<Vec<DecodeStepResult>>,
    },
}
```

The key insight: **all decisions are made by the scheduler, not the model.** The scheduler decides the packet variant,
allocates paged blocks if needed, and the model is a pure executor that just runs the forward pass described by the packet.

## GPU Worker Loop (Draft)

```rust
async fn gpu_worker(
    mut rx: mpsc::UnboundedReceiver<GpuPacket>,
    engine: Arc<Engine>,
) {
    while let Some(packet) = rx.recv().await {
        let result = tokio::task::spawn_blocking({
            let engine = engine.clone();
            move || engine.execute_packet(packet)
        }).await;
        // result_tx inside each packet sends the result back
    }
}
```

## Migration Strategy

1. **Phase 1**: Add `GpuPacket` enum and GPU queue. GPU worker initially just calls existing engine methods. ✅ Done (2026-03-08)
2. **Phase 2**: Migrate `BatchRuntime` to produce packets instead of direct GPU dispatch. ✅ Done (2026-03-08)
3. **Phase 3**: Migrate `ContinuousRuntime` to produce packets. Remove feature gate. ✅ Done (2026-03-08)
4. **Phase 4**: Unified model forward pipeline + classify/embed GPU queue integration. ✅ Done (2026-03-09)
5. **Phase 5**: Clean up — migrate streaming batch to GPU queue, remove dead code, update CLAUDE.md. ✅ Done (2026-03-09)
6. **Phase 6**: Dead code cleanup — remove StreamingBatch, execute_multi_token_batch, MultiTokenDecode; error at routing for unsupported max_new>1. ✅ Done (2026-03-10)

### Phase 4 Notes

Phase 4 unified the model forward path across all task types (generate, classify, embed):

1. **GPU queue for all tasks**: Added `ClassifyBatch` and `EmbedBatch` packet variants to `GpuPacket`.
   Classify/embed requests in `batch_runtime.rs` now route through the GPU queue (previously used
   `spawn_blocking` which bypassed the queue). This eliminates GPU contention between task types.

2. **Unified prefill pipeline** (`Engine::prefill_pipeline()` in `tokenize.rs`):
   All task types now flow through the same serial pipeline:
   - Step 1: Prefix cache lookup (`find_common_prefix_from_groups` → `try_prefix_match_for_prefill`)
   - Step 2: Pack tokens with offset (`pack_varlen_token_groups_with_offset`) — skips cached prefix
   - Step 3: Varlen forward (`model.forward()` via `BatchAttnContext`)
   - Step 4: Convert to F32 output rows
   This enables prefix caching for classify/embed, not just generation.

3. **Dead code removal**: Removed `batch_varlen_forward()` and `pack_varlen_token_groups()` — these were
   the old standalone helpers that classify/embed called directly. Both now use `prefill_pipeline()`.

4. **Extracted `execute_gpu_packet()`** from the GPU worker loop as the unified execution entry point.
   The GPU worker loop is now trivial: `while packet = recv() { execute_gpu_packet(&engine, packet) }`

**Architecture after Phase 4:**
```
All Requests → GPU Queue (FIFO) → GPU Worker → execute_gpu_packet()
                                                  │
                                    ┌─────────────┼─────────────┐
                                    ▼             ▼             ▼
                              GenerateBatch  ClassifyBatch  EmbedBatch
                                    │             │             │
                                    └─────────────┴─────────────┘
                                                  │
                                          prefill_pipeline()
                                    1. Prefix Cache Lookup [always]
                                    2. Pack Tokens (skip cached prefix)
                                    3. Varlen Forward [always]
                                    4. Post-processing [task-specific]
```

**TODO for paged prefix cache**: `prefill_pipeline()` finds cached block IDs but does not yet
build `PagedKvBatchContext` from them for classify/embed. The hook is in place (`_cached_block_ids`),
just needs the context construction when paged prefix is enabled for non-generation tasks.

### Phase 5 Notes

1. **Streaming batch migrated to GPU queue**: Added `StreamingBatch` packet variant. The entire streaming
   lifecycle (tokenize → prefill → decode loop) now runs on the GPU worker thread instead of a random
   `spawn_blocking` thread. Removed `block_in_place` wrappers from `process_streaming_batch` and
   `process_streaming_sequential` since they now execute on a sync OS thread.

2. **Dead code cleanup**:
   - Removed `GenerationRequestState::prepare_generate_request()` (unused wrapper)
   - Changed `tokenize_gen_batch_sync` to take `&Engine` instead of `Arc<Engine>`
   - Fixed `unused_mut` warning in `continuous_runtime.rs`

3. **CLAUDE.md updated**: Added "GPU Queue" and "Unified Prefill Pipeline" architecture sections,
   updated "Model Forward Modes" to reflect single `ModelForward::forward()` entry point.

### Phase 6: Dead Code Cleanup (2026-03-10)

1. **Removed `StreamingBatch` GPU packet variant and all streaming compat machinery**:
   - `StreamingBatch` in `GpuPacket`, `submit_streaming_batch()`, `process_streaming_batch()`,
     `process_streaming_sequential()` — all removed.
   - `StreamingCompatBatch`, `ReadyGenerationWork` enum, `enqueue_streaming_compat()`,
     `normalize_streaming_batch()` — all removed from `batch_runtime.rs`.
   - `pending_streaming_gpu` variable and all its references in the main scheduler loop — removed.
   - `generate_stream_sync_pub()` wrapper in `cache/paged.rs` — removed (only caller was streaming sequential).

2. **Removed `execute_multi_token_batch`**: This function was unreachable dead code. If paged-attn
   is available, `max_new > 1` routes to the continuous runtime (not the batch runtime). If paged-attn
   is not available, `ensure_multi_token_decode_ready()` returns `Err` before reaching it.

3. **Removed `MultiTokenDecode` execution kind and `PreparedGenerateBatchPlan` enum**:
   - `GenerationBatchExecutionKind` now only has `CudaPrefillOnly` and `CpuPrefillOnly`.
   - `PreparedGenerateBatch.plan` is now `PrefillPlan` directly (not `PreparedGenerateBatchPlan`).
   - `plan_generate_batch()` simplified — no longer checks `all_prefill_only`.
   - `generate_prepared_batch()` matches directly on `plan.execution_kind`.

4. **`max_new > 1` without paged-attn now returns error at routing time**:
   - Previously: routed to batch runtime → `StreamingBatch` → fallback sequential decode (unreliable).
   - Now: `enqueue_request()` / `enqueue_stream_request()` return `EngineError::Unavailable`
     immediately with "multi-token generation requires paged attention support".
   - Removed `GenerationDispatchKind`, `GenerationSchedulerRoute`, `select_generation_scheduler()`,
     `supports_continuous_generation()` — replaced with direct if/else routing.

5. **Simplified `GenerationRequestState`**:
   - Removed `prepared_max_new` field, `dispatch_kind()`, `fresh_logits_processor()`, `stream_sender()`.
   - These were only used by the streaming compat path.

**Benchmark (2026-03-10, GPU 2, Qwen3-0.6B, flash-attn-v3 + paged-attn):**
No regression. Results within normal variance of Phase 4 baseline.

| Concurrency | Req/s | Tok/s | TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 9.28 | 296.91 | 3.30 |
| 4 | 30.90 | 988.83 | 3.67 |
| 8 | 54.15 | 1732.88 | 3.84 |
| 16 | 58.50 | 1871.93 | 3.76 |

**Remaining bypass (by design):**
- `CandleEngine` direct path (`generate_sync`, `generate_stream_sync`) bypasses the GPU queue —
  this is intentional for `PRELUDE_NO_SCHEDULER=1` mode (accuracy tests, debugging).

