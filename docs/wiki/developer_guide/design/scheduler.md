# Scheduler

The scheduler runs entirely on CPU. Its job is to decide — each step — which sequences get KV cache blocks and GPU time. It produces a `ScheduledBatch` that the run loop hands to the `Executor` for the forward pass.

**Scheduler decides *what* to run. Executor decides *how* to run it on the device.** These are cleanly separated: the scheduler has no GPU calls, no device types, no kernel knowledge.

There is intentionally no shared `Scheduler` trait. AR, diffusion, TTS, and DLLM have fundamentally different state machines — a shared abstraction would either be too thin to be useful or would leak workload-specific concerns into every caller. Each scheduler is self-contained and only needs to be understood on its own terms.

## Relationship with `ScheduledEngine`

The scheduler folder and `engine/scheduled.rs` have a strict division of responsibility:

| | `scheduler/` | `engine/scheduled.rs` + `engine/run/ar.rs` |
|---|---|---|
| **Knows about** | Sequences, KV blocks, budgets, queues | Tokio, async channels, GPU executor, HTTP responses |
| **Does** | Decides *what* to run each step | Wires that decision to I/O and device |
| **Has GPU calls** | No | Yes (via `Executor`) |
| **Has async** | No | Yes |

`ScheduledEngine` (`engine/scheduled.rs`) is the async entry point. It owns the channel that accepts incoming requests and spawns `ar_loop` as a long-running tokio task:

```rust
// engine/scheduled.rs
pub struct ScheduledEngine {
    ar_tx: mpsc::UnboundedSender<ArMessage>,  // send requests in
    executor: Arc<dyn Executor>,
    engine: Arc<Engine>,
    _ar_loop_handle: tokio::task::JoinHandle<()>,
}
```

When `generate()` is called, `ScheduledEngine` tokenizes the request synchronously and puts it on the channel — that is its entire job:

```rust
fn prepare_and_enqueue(&self, request, response) {
    let prepared = self.engine.prepare_generate_request(&request)?;  // tokenize
    self.ar_tx.send(ArMessage::NewRequest { prepared, response })?;  // hand off
}
```

`ar_loop` (`engine/run/ar.rs`) is what actually drives the scheduler. It owns a `Scheduler` instance and calls it every iteration:

```rust
// engine/run/ar.rs
async fn ar_loop(engine, executor, config, mut rx) {
    let mut scheduler = ArScheduler::new(config);   // ← lives here, not in ScheduledEngine

    loop {
        // drain incoming ArMessages → scheduler.add_request()
        // scheduler.schedule_step() → SchedulerStep { prefill_ids, decode_ids }
        // build ForwardBatch, executor.submit() → handle.recv().await
        // process output → stream tokens → scheduler.finish_sequence()
    }
}
```

The flow in full:

```
HTTP / API caller
      │  async generate()
      ▼
ScheduledEngine          (engine/scheduled.rs)
      │  ar_tx.send(ArMessage::NewRequest)
      ▼
ar_loop                  (engine/run/ar.rs)
      │  scheduler.add_request(seq)
      │  scheduler.schedule_step() ──► SchedulerStep
      │  build ForwardBatch
      │  executor.submit(batch) ──────► device (prelude-cuda / prelude-cpu)
      │  handle.recv().await
      │  sample + check stop conditions
      │  scheduler.finish_sequence()
      │  stream token ──────────────────► caller via ResponseChannel
      ▼
  (loop)
```

`scheduler/` has no knowledge of any of this. It only sees `add_request`, `schedule_step`, and `finish_sequence` calls. Everything async, device-related, and I/O-related is in `ar_loop` and `ScheduledEngine`.

## File Structure

```
prelude-core/src/scheduler/
├── mod.rs                             # re-exports
├── ar.rs                              # ArScheduler — continuous batching for AR LLMs
├── dllm.rs                            # DllmScheduler — block-level demasking for diffusion LLMs
├── diffusion.rs                       # DiffusionScheduler — denoising loop for image/video
├── oneshot.rs                         # OneShotScheduler — embed, classify, prefill-only
├── tts.rs                             # TtsPipelineScheduler — multi-stage TTS streaming
├── types.rs                           # ScheduledBatch, ScheduledRequest, StepResult, etc.
└── components/                        # Reusable components, used by schedulers as needed
    ├── cache/
    │   ├── block_manager.rs           # BlockManager — block alloc/free + ref counting
    │   ├── prefix_cache.rs            # PrefixKvCache — block-level hash-trie + LRU eviction
    │   ├── prefix_index.rs            # PrefixMatchIndex — tensor-free prefix matching algorithm
    │   ├── kv_buf.rs                  # KvBuf — per-sequence KV buffer (non-paged path)
    │   └── deltanet_pool.rs           # DeltaNetPool — pre-alloc recurrent state for hybrid models
    └── request_queue.rs               # RequestQueue — FCFS / priority / cache-aware ordering
```

## Scheduler Types

| Scheduler | Workload | File | KV cache |
|-----------|----------|------|----------|
| `ArScheduler` | AR LLMs (Qwen3, LLaMA, DeepSeek, ...) | `ar.rs` | Paged KV + prefix cache |
| `DllmScheduler` | Diffusion LLMs | `dllm.rs` | Paged KV + prefix cache |
| `DiffusionScheduler` | Image / video diffusion | `diffusion.rs` | None |
| `TtsPipelineScheduler` | TTS | `tts.rs` | Per AR stage (wraps ArScheduler) |
| `OneShotScheduler` | Embed / classify / rerank | `oneshot.rs` | None |

## AR Scheduler

`ArScheduler` is the main scheduler. It implements continuous batching for autoregressive LLMs: prefill and decode sequences share the same step, KV cache is paged, and prefix caching is integrated at admission time.

### Sequence State Machine

```
          ┌─────────────────────────────────┐
          │         (preempt: memory full)  │
          ▼                                 │
       Waiting ──► Prefilling ──► Decoding ──► Finished
```

- **Waiting** — admitted to queue, not yet scheduled
- **Prefilling** — prompt tokens being processed (may span multiple steps via chunked prefill)
- **Decoding** — generating tokens one step at a time
- **Finished** — stop token, length limit, abort, or KV transferred (P/D disaggregation)
- **Preemption** — a decode sequence is evicted back to Waiting when memory is tight; its KV blocks are freed and it will be re-prefilled

The key field tracking progress is `num_computed_tokens` on each request. A sequence is in prefill while `num_computed_tokens < prompt_len`, and in decode afterwards. This unified token-centric view makes chunked prefill and speculative rejection handling straightforward.

### Step Loop

Each call to `scheduler.step()`:

1. **Prefix cache lookup** — for all waiting requests, run LPM (longest prefix match) against `PrefixKvCache`. Hit blocks are ref-counted and reserved before scheduling decisions are made.
2. **Schedule running requests** — check each running sequence against token budgets. Preempt the lowest-priority decode sequences if `max_total_tokens` would be exceeded.
3. **Admit waiting requests** — pull from `RequestQueue` and assign fresh KV blocks via `BlockManager` until budgets are exhausted.
4. **Build `ScheduledBatch`** — pack all scheduled sequences into a flat batch with per-sequence `token_ids`, `block_table`, and `num_cached_tokens`.
5. **Return to run loop** — run loop calls `executor.submit(batch)`, overlapping GPU execution with CPU scheduling of the next step.
6. **Update on completion** — advance `num_computed_tokens`, insert finished sequences into `PrefixKvCache`, free blocks of terminated sequences.

### Token Budget Constraints

Three knobs, all configurable via CLI:

| Flag | What it limits | Typical binding condition |
|------|---------------|--------------------------|
| `--max-running-requests` | Concurrent sequences in prefill + decode | Large batch sizes |
| `--max-prefill-tokens` | Tokens scheduled in a single prefill step | Long prompts |
| `--max-total-tokens` | Total KV slots across all running sequences | GPU memory |

`max_prefill_tokens` is the key lever for preventing head-of-line blocking: it ensures a single long prefill cannot monopolize a step and starve decode sequences from making progress.

### Chunked Prefill

When a prompt is longer than `max_prefill_tokens`, it is split across multiple steps. Each step processes a chunk of the prompt, with the remaining decode sequences continuing to generate tokens in the same batch. This keeps time-to-first-token (TTFT) bounded and avoids decode latency spikes during large prefills.

### Prefix Cache Integration

At admission, `PrefixKvCache` performs a block-level longest prefix match on the incoming token sequence. Matching blocks are ref-counted and reused — the sequence starts with `num_computed_tokens` already set to the matched prefix length, skipping those tokens entirely during prefill.

`PrefixKvCache` uses a hash-trie structure. LRU eviction removes leaf blocks when capacity is full. The integration with `BlockManager` is ref-counted: cached blocks are never freed while a running sequence holds a reference, even if they are evicted from the cache index.

### Preemption

When `max_total_tokens` is tight and a new high-priority sequence needs blocks, the scheduler preempts the decode sequence with the fewest computed tokens (least work lost). Its blocks are freed immediately and it re-enters `Waiting`. It will be re-prefilled from scratch — prefix cache may cover part or all of the prompt again, reducing the re-prefill cost.

There is no KV rollback mechanism. Speculative decoding rejection is handled naturally: rejected draft tokens simply do not advance `num_computed_tokens`, so the next step re-schedules those slots without any special recovery logic.

### Request Queue Ordering

`RequestQueue` ordering is pluggable via `SchedulePolicy`. Default is FCFS. The cache-aware variant reorders the waiting queue to maximize prefix hit rate — requests sharing a long common prefix are batched together so their shared blocks stay hot in `PrefixKvCache`.

## Shared Components

### BlockManager

Paged KV cache block allocator (vLLM-style). Manages a fixed pool of blocks, each holding `block_size` tokens of KV state. Allocates and frees blocks with ref counting so cached blocks shared across sequences are not freed prematurely.

Block size is auto-tuned per attention backend: 128 tokens with FlashInfer/FA3, 16 otherwise. Override with `PRELUDE_PAGED_BLOCK_SIZE`.

### PrefixKvCache

Hash-trie that maps token hash chains to KV block sequences. Supports block-level LPM: an incoming sequence matches as many leading blocks as possible. LRU eviction removes leaf blocks when capacity is full. Ref-counted integration with `BlockManager` ensures live blocks are never evicted while in use.

### RequestQueue

Holds waiting requests with pluggable ordering. Used by `ArScheduler` and `DllmScheduler`. Three policies: FCFS (default), priority-based, and cache-aware (LPM-sorted to maximize prefix hit rate).

## Advanced Features

### Speculative Decoding

`SpecDecodeRunner` sits between `ArScheduler` and the model runner. It uses a pluggable `DraftProposer` to generate draft tokens:

| Proposer | Method |
|----------|--------|
| `DraftModel` | Smaller model generates draft tokens |
| `EAGLE` | Draft head predicts from hidden states |
| `Ngram` | Prompt n-gram matching |
| `Medusa` | Multiple decoding heads in parallel |

Draft tokens are appended to the batch as extra entries. The verify step runs the target model over all draft tokens in one forward pass using tree attention masks (for EAGLE/Medusa multi-path speculation). Rejected tokens use `PADDING_SLOT_ID = -1` — no KV rollback needed.

### Constrained Decoding

Grammar-based token filtering runs between `model.forward()` and sampling. The `GrammarBackend` trait (default: llguidance) compiles a `ConstraintSpec` (JSON schema, regex, EBNF grammar, or choice list) into a per-step bitmask that masks invalid tokens before sampling.

Compilation is async on a thread pool, overlapping with GPU work. The full sampling pipeline per step:

```
repetition / presence / frequency penalties
    → grammar bitmask
    → temperature scaling
    → sampling (top-p, top-k, min-p)
    → grammar state advance
```

Grammar state integrates cleanly with speculative decoding: rejected tokens trigger `GrammarMatcher::rollback()` to rewind grammar state.

### Disaggregated Prefill/Decode

Prefill and decode run on separate worker pools. The coordinator routes requests based on prefix cache hits on decode workers. The `ArScheduler` core loop is unchanged; the only additions are:

- `FinishReason::Transferred` — prefill worker marks a sequence done after shipping its KV blocks
- `ArRequest::preloaded_blocks` — decode worker receives pre-imported blocks and starts decode immediately
- `BlockAllocator::import_blocks()` — registers incoming KV blocks from the transfer layer

KV transfer is transport-agnostic (`KvTransfer` trait). Mooncake handles the actual transfer (NVLink, RDMA, or TCP relay depending on topology). See [Integration](../integration.md) for details.

## Adding a New Scheduler

Only three files need to change:

1. **New scheduler file** — `prelude-core/src/scheduler/your_scheduler.rs`. Use `oneshot.rs` as the simplest reference. Implement a `step()` method that returns `ScheduledBatch`.
2. **New run loop** — `prelude-core/src/engine/run/your_loop.rs`. Call `scheduler.step()`, submit to executor, call `scheduler.update()` on completion.
3. **Config branch** — `prelude-core/src/engine/config.rs`. Add a variant to `EngineMode` and wire it to your run loop.

No changes needed to model code, ops, or device crates.

## Design Choices

| Feature | Source | Kept | Dropped |
|---------|--------|------|---------|
| Token-centric scheduling | vLLM V1 | Unified prefill/decode via `num_computed_tokens` | Separate status enum states for prefilling/decoding |
| Chunked prefill | vLLM V1 | Interleaved prefill + decode in same batch | — |
| Preemption | vLLM V1 | Full recompute (simple, no swap complexity) | KV swap to CPU |
| Paged KV blocks | vLLM V1 | `BlockManager` with ref counting | KV connector coupling to specific transfer backends |
| Radix tree prefix cache | SGLang | Block-level hash-trie with LRU eviction | Configurable eviction policies |
| Cache-aware scheduling | SGLang | LPM queue ordering to maximize prefix hits | — |
| Compute-schedule overlap | SGLang | CPU schedules next batch while GPU runs current | — |
