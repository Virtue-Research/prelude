# Scheduler

The scheduler runs entirely on CPU. Its job is to decide — each step — which sequences get KV cache blocks and GPU time. It produces a `SchedulerStep` that the run loop hands to the `Executor` for the forward pass.

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
    let mut scheduler = Scheduler::new(config);   // ← lives here, not in ScheduledEngine

    loop {
        // drain incoming ArMessages → scheduler.add_request()
        // scheduler.schedule_step() → SchedulerStep { prefill_request_ids, decode_request_ids }
        // build ForwardBatch, executor.submit() → handle.recv().await
        // process output → stream tokens → scheduler.finish_request()
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
      │  scheduler.finish_request()
      │  stream token ──────────────────► caller via ResponseChannel
      ▼
  (loop)
```

`scheduler/` has no knowledge of any of this. It only sees `add_request`, `schedule_step`, and `finish_request` calls. Everything async, device-related, and I/O-related is in `ar_loop` and `ScheduledEngine`.

## File Structure

```
prelude-core/src/scheduler/
├── mod.rs                             # re-exports
├── state.rs                           # Scheduler, Sequence, SchedulerStep, SchedulerConfig, SequenceStatus, etc.
├── admission.rs                       # admit_waiting_sequences, get_new_prefill_batch, get_decode_batch, get_mixed_batch
├── preemption.rs                      # preempt_one_from_running, drain_finished, sort_waiting_queue
├── adaptive.rs                        # AdaptiveSchedulerState — EWMA batch size and wait time tuning
├── dllm.rs                            # DllmScheduler — block-level demasking for diffusion LLMs
├── diffusion.rs                       # DiffusionScheduler — denoising loop for image/video
├── oneshot.rs                         # OneShotScheduler — embed, classify, prefill-only
├── tts.rs                             # TtsPipelineScheduler — multi-stage TTS streaming
└── components/                        # Reusable components, used by schedulers as needed
    ├── cache/
    │   ├── block_manager.rs           # BlockManager — block alloc/free + ref counting
    │   ├── prefix_cache.rs            # PrefixKvCache — block-level hash-trie + LRU eviction
    │   ├── prefix_index.rs            # PrefixMatchIndex — tensor-free prefix matching algorithm
    │   ├── kv_buf.rs                  # KvBuf — per-sequence KV buffer (non-paged path)
    │   └── deltanet_pool.rs           # DeltaNetPool — pre-alloc recurrent state for hybrid models
    └── request_queue.rs               # RequestQueue — stub, not yet extracted (logic inline in state.rs / preemption.rs)
```

## Workflow

```
engine/run/ar.rs
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  HTTP request ──► scheduler.add_request(seq)                                          │
│                                                                                       │
│  ┌── Batch Scheduler  scheduler/state.rs + admission.rs ───────────────────────────┐ │
│  │                                                                                  │ │
│  │  waiting_queue            running                finished                        │ │
│  │  ┌─────────────┐  admit  ┌─────────────┐  done  ┌─────────────┐                  │ │
│  │  │  Sequence   │────────►│  Sequence   │───────►│  Sequence   │                  │ │
│  │  │  Sequence   │         │  Sequence   │        │  ...        │                  │ │
│  │  │  ...        │◄────────│  ...        │        └─────────────┘                  │ │
│  │  └──────┬──────┘ preempt └──────┬──────┘                                         │ │
│  │         │                       │                                                │ │
│  │         │   schedule_step():    │                                                │ │
│  │         │   ① check budgets ◄──┘  ← internal: tokens_in_use vs                  │ │
│  │         │      max_total_tokens, max_prefill_tokens, max_running_requests        │ │
│  │         │   ② admit / preempt                                                   │ │
│  │         │   ③ build SchedulerStep (prefill_request_ids, decode_request_ids)     │ │
│  └─────────┼───────────────────────────────────────────────────────────────────────-┘ │
│            │                                                                          │
│  SchedulerStep ──► executor.submit(batch)  ──► GPU forward pass                       │
│                                                        │                              │
│  scheduler.finish_request() ◄── output tokens ─────────┘                              │
│    → stream token to caller                                                           │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Scheduler Types

| Scheduler | Workload | File | KV cache |
|-----------|----------|------|----------|
| `Scheduler` | AR LLMs (Qwen3, LLaMA, DeepSeek, ...) | `state.rs` + `admission.rs` | Paged KV + prefix cache (planned) |
| `DllmScheduler` | Diffusion LLMs | `dllm.rs` | Paged KV + prefix cache |
| `DiffusionScheduler` | Image / video diffusion | `diffusion.rs` | None |
| `TtsPipelineScheduler` | TTS | `tts.rs` | Per AR stage (wraps Scheduler) |
| `OneShotScheduler` | Embed / classify / rerank | `oneshot.rs` | None |

## AR Scheduler

`Scheduler` is the main scheduler. It implements continuous batching for autoregressive LLMs: prefill and decode sequences share the same step. KV cache block allocation is handled by the Engine layer (`planner.rs`, `paged_prefill.rs`) during execution, not inside `schedule_step`.

### Sequence State Machine

```
          ┌─────────────────────────────────┐
          │         (preempt: memory full)  │
          ▼                                 │
       Waiting ──► Prefilling ──► Decoding ──► Finished
```

- **Waiting** — admitted to queue, not yet scheduled
- **Prefilling** — prompt tokens being processed
- **Decoding** — generating tokens one step at a time
- **Finished** — stop token, length limit, abort, or KV transferred (P/D disaggregation)
- **Preemption** — a decode sequence is evicted back to Waiting when memory is tight; it will be re-prefilled

The key field tracking progress is `kv_computed_len` on each request. A sequence is in prefill while `kv_computed_len < prompt_len`, and in decode afterwards.

### Scheduling Algorithm

```
Algorithm: Scheduler.schedule_step()
─────────────────────────────────────────────────────────────────────────────
Input:  waiting_queue, running, config, tokens_in_use, effective_new_token_ratio
Output: SchedulerStep { prefill_request_ids, decode_request_ids, forward_mode }

1. DRAIN FINISHED
   for seq in running where seq.status == Finished:
       tokens_in_use -= seq.total_len()
       move seq → finished

2. SELECT MODE
   if chunked_prefill:  goto MIXED
   elif waiting_queue non-empty:  goto PREFILL
   else:  goto DECODE

── PREFILL ──────────────────────────────────────────────────────────────────
3. COMPUTE BUDGETS
   available_slots    ← max_running_requests − |running|
   reserved_decode    ← Σ min(remaining_tokens(seq), decode_reservation_cap)
                        for seq in running
                        * effective_new_token_ratio
   total_token_budget ← max_total_tokens − tokens_in_use − reserved_decode
   prefill_budget     ← max_prefill_tokens

4. SORT waiting_queue by policy (FCFS | priority)

5. ADMIT (greedy, FCFS within budget)
   to_prefill ← []
   for seq in waiting_queue (front to back):
       if |to_prefill| == available_slots: break
       if seq.prompt_len > prefill_budget:  break          ← head-of-line guard
       if seq.estimated_total > total_token_budget:
           if can preempt one from running:
               victim ← running sequence with latest arrival_time (LIFO)
               total_token_budget += victim.total_len()
               push victim → front of waiting_queue
           else: break

       prefill_budget     -= seq.prompt_len
       total_token_budget -= seq.estimated_total
       tokens_in_use      += seq.prompt_len
       push seq → to_prefill

6. return SchedulerStep { prefill: to_prefill, decode: [], mode: Prefill }

── DECODE ───────────────────────────────────────────────────────────────────
7. ENSURE CAPACITY
   while tokens_in_use + |running| > max_total_tokens:
       victim ← running sequence with latest arrival_time (LIFO)
       tokens_in_use -= victim.total_len()
       push victim → front of waiting_queue

8. effective_new_token_ratio ← max(ratio − decay, min_ratio)

9. return SchedulerStep { prefill: [], decode: running, mode: Decode }

── MIXED (chunked_prefill=true) ─────────────────────────────────────────────
10. Run steps 3–5 (admit new prefills) + step 7 (ensure decode capacity)
    return SchedulerStep { prefill: to_prefill, decode: running, mode: Mixed }

── POST STEP (called by run loop after GPU completes) ───────────────────────
11. for each completed seq:
        if seq finished (stop token / length limit):
            scheduler.finish_request(id, reason)  ← picked up by step 1 next iteration
        else:
            scheduler.on_token_generated(id, token)
─────────────────────────────────────────────────────────────────────────────
```

### Planned: Prefix Cache Admission Integration

The following KV cache integration is **not yet wired into `schedule_step()`**. It currently lives in `Engine` (`planner.rs` / `paged_prefill.rs`) and is planned to move into the admission path:

```
ADMIT (with prefix cache — planned):
   for seq in waiting_queue:
       ...budget checks...

       ① cached_len, hit_block_ids ← PKV.match_prefix(seq.tokens)
                                    ← longest prefix match in hash-trie;
                                      touches LRU so matched blocks stay warm
       ② suffix_blocks ← BM.allocate_for_tokens(seq.prompt_len − cached_len)
                                    ← fresh blocks for uncached suffix only
       seq.block_table ← hit_block_ids + suffix_blocks
       seq.kv_computed_len ← cached_len   ← executor skips these during prefill

POST STEP (with prefix cache — planned):
   after prefill:
       ③ stored ← PKV.insert_blocks_with_paged(seq.tokens, layer_kvs, seq.block_table)
          BM.increment_refs(stored)   ← prefix cache now co-owns these blocks
   on preemption / finish:
       BM.free(victim.block_table)
       PKV.take_evicted_paged_blocks() |> BM.decrement_refs()
```

### Step Loop

Each call to `scheduler.schedule_step()`:

1. **Drain finished** — remove completed sequences from `running`, subtract their tokens from `tokens_in_use`.
2. **Select mode** — mixed if `mixed_chunked`, else prefill if waiting queue is non-empty, else decode.
3. **Compute budgets** — `available_slots`, `reserved_decode`, `total_token_budget`, `prefill_budget`.
4. **Sort waiting queue** — by FCFS, or by priority field if any sequence has one set.
5. **Admit or ensure capacity** — greedy admission from waiting queue (prefill path), or preempt to free slots (decode path).
6. **Return `SchedulerStep`** — contains `prefill_request_ids` and `decode_request_ids`; run loop builds the `ForwardBatch` and calls `executor.submit()`.
7. **Update on completion** — run loop calls `on_token_generated` per token, `finish_request` when done.

### Token Budget Constraints

Four knobs, all configurable via CLI:

| Flag | What it limits | Typical binding condition |
|------|---------------|--------------------------|
| `--max-running-requests` | Concurrent sequences in prefill + decode | Large batch sizes |
| `--max-prefill-tokens` | Tokens scheduled in a single prefill step | Long prompts |
| `--max-total-tokens` | Total KV slots across all running sequences | GPU memory |
| `--decode-reservation-cap` | Per-request decode reservation cap | Prevents one oversized sequence from starving admission |

`max_prefill_tokens` is the key lever for preventing head-of-line blocking: it ensures a single long prefill cannot monopolize a step and starve decode sequences from making progress.

### Adaptive Batching

`AdaptiveSchedulerState` (`adaptive.rs`) dynamically tunes batch size and wait time using two EWMA signals:

- **Arrival rate** (`lambda_hat`) — updated on each batch of incoming requests; clamped to prevent burst explosion.
- **Per-request GPU time** (`s_hat[b]`) — EWMA of observed `total_gpu_ms / batch_size` at each batch size `b`. Unobserved sizes are interpolated via a `C/b + α` cost model that captures fixed kernel launch overhead amortized over the batch.

**Optimal batch selection** uses the marginal rule: keep increasing batch size as long as the per-request GPU time savings from adding one more sequence exceed the expected inter-arrival wait, subject to `max_batch_wait_ms` as a hard cap.

`max_batch_size` and `max_batch_wait_ms` from `SchedulerConfig` act as hard ceilings; the adaptive logic operates within them.

### Chunked Prefill

When `--chunked-prefill` is enabled (on by default), the scheduler runs `get_mixed_batch()` each step, admitting new prefills alongside running decode sequences. This interleaves prefill and decode to keep TTFT bounded under load.

Note: individual prompts longer than `max_prefill_tokens` are not yet split across steps — the head-of-line guard (`if prompt_len > prefill_budget: break`) still applies. Splitting oversized prompts across multiple steps is a planned future improvement.

### Prefix Cache Integration

`PrefixKvCache` performs a block-level longest prefix match on the incoming token sequence at admission. Matching blocks are ref-counted and reused — the sequence starts with `kv_computed_len` already set to the matched prefix length, skipping those tokens entirely during prefill.

`PrefixKvCache` uses a hash-trie structure. LRU eviction removes leaf blocks when capacity is full. The integration with `BlockManager` is ref-counted: cached blocks are never freed while a running sequence holds a reference, even if they are evicted from the cache index.

This integration currently lives in the Engine layer. See [Planned: Prefix Cache Admission Integration](#planned-prefix-cache-admission-integration) for the intended future call sites inside `schedule_step`.

### Preemption

When `max_total_tokens` is tight and a new sequence needs blocks, the scheduler preempts the **most recently arrived** decode sequence (LIFO by arrival time). Its blocks are freed immediately and it re-enters `Waiting`. It will be re-prefilled from scratch — prefix cache may cover part or all of the prompt again, reducing the re-prefill cost.

There is no KV rollback mechanism. Speculative decoding rejection is handled naturally: rejected draft tokens simply do not advance `kv_computed_len`, so the next step re-schedules those slots without any special recovery logic.

### rollback_prefill

`rollback_prefill(request_ids)` returns a set of in-progress prefill sequences back to `Waiting`. Used when the executor rejects a batch (e.g. OOM). The sequences have their `kv_computed_len` reset to 0 and `status` reset to `Waiting`, and are pushed to the front of the waiting queue.

### Request Queue Ordering

`RequestQueue` ordering is pluggable via `SchedulePolicy`. Current implementation:

- **FCFS** (default) — ordered by arrival time. If any sequence has a `priority` field set, the queue is sorted by priority first, then arrival time.

**Planned:**
- **Cache-aware** — reorders by longest prefix match length, so sequences sharing a long common prefix are batched together while their shared blocks are still hot in `PrefixKvCache`.

The queue logic currently lives inline in `state.rs` and `preemption.rs:sort_waiting_queue`. `request_queue.rs` is a placeholder for a future extraction.

## KV Cache

The KV cache subsystem lives entirely in `scheduler/components/cache/`. It has no GPU calls — it only tracks which blocks exist, who owns them, and which token prefixes are cached. The executor does the actual tensor movement.

**These components are called from the Engine layer (`planner.rs`, `paged_prefill.rs`) during prefill execution, not from inside `schedule_step()`.** See [Planned: Prefix Cache Admission Integration](#planned-prefix-cache-admission-integration) for the intended future integration into the scheduler.

### Request Lifecycle and Algorithm Order

The four algorithms are not independent — they are called in a fixed order driven by each request's lifecycle. Algorithm 3 is always internal to Algorithm 2; Algorithm 4 is a utility called from all stages.

```
Request arrives
      │
      ▼
① match_prefix()              [prefix_index.rs:109]   ← ADMISSION
  "what prefix is already cached?"
  → cached_len, hit_block_ids
      │
      │  cached_len tells how many tokens to skip;
      │  hit_block_ids go directly into seq.block_table
      ▼
④ BM.allocate_block_tables_from_plan()  [planner.rs:118]  ← ADMISSION
  for each sequence being admitted:
      if prefix hit:
          bt = hit_block_ids + new suffix blocks
          BM.increment_refs(hit_block_ids)   ← this sequence now co-owns the cached blocks
      else:
          bt = BM.allocate_for_tokens(prompt_len)
      seq.block_table ← bt
      │
      ▼
  ┌─────────────────────────────────────────────────┐
  │  GPU: prefill forward pass                      │
  │  reads KV from hit_block_ids (cached prefix)    │
  │  writes KV into suffix blocks (new tokens)      │
  └─────────────────────────────────────────────────┘
      │
      ▼
② insert_blocks_with_paged()  [prefix_cache.rs:118]   ← AFTER PREFILL (in executor)
  "store newly computed KV so future requests can reuse it"
  [called by device executor, not by ar.rs]
      │
      ├──► ③ evict_if_needed()  [prefix_index.rs:321] ← INSIDE ②
      │      "trie over capacity — remove coldest leaf"
      │      → collects evicted_paged_block_ids (deferred to caller)
      │
      ▼
④ BM.increment_refs(stored)   [block_manager.rs]      ← AFTER ② (in executor)
  "prefix cache now co-owns the newly stored blocks"
      │
      ▼
  ┌─────────────────────────────────────────────────┐
  │  GPU: decode steps (N iterations)               │
  │  reads KV from full block_table each step       │
  │  writes one new KV slot per step                │
  │  [no KV cache algorithm involved during decode] │
  └─────────────────────────────────────────────────┘
      │
      ▼
④ BM.free(seq.block_table)    [block_manager.rs]      ← SEQUENCE FINISHES
  "sequence releases its ownership of all blocks it held"
  [called from release_resources() in ar.rs]
      │
      │  shared prefix blocks stay alive if prefix cache still holds a reference
      │  until LRU eviction fires inside the next insert_blocks()
      │
      ▼
④ BM.decrement_refs(evicted)  [block_manager.rs]      ← AFTER LRU EVICTION
  "prefix cache releases its ownership"
  → block returns to free pool only when all owners have released
```

### Ref Counting

`increment_refs` is called in **two places** — once when a block enters the prefix cache (prefix cache takes ownership), and once when any new sequence reuses a cached block at admission (that sequence takes ownership). So ref_count = 1 (prefix cache) + N (sequences currently using the block).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Ref Count Rules                                                            │
│                                                                             │
│  BM.allocate()              sets   ref_count = 1   (new sequence owns it)  │
│  BM.increment_refs()        adds   ref_count += 1  called in two places:   │
│    ① planner.rs:136   when a new sequence reuses a cached prefix block     │
│    ② after insert_blocks()  when prefix cache stores a newly computed block│
│  BM.free()                  subs   ref_count -= 1  (sequence releases)     │
│  BM.decrement_refs()        subs   ref_count -= 1  (LRU eviction releases) │
│                                                                             │
│  ref_count == 0  →  block returned to free pool                            │
│  ref_count can exceed 2 if multiple sequences reuse the same cached block  │
└─────────────────────────────────────────────────────────────────────────────┘

Example: seq A completes prefill, seq B later hits the same prefix
─────────────────────────────────────────────────────────────────────────────
Event                                    blk_0   blk_1   blk_2   free pool
─────────────────────────────────────────────────────────────────────────────
initial state                              -       -       -     [0,1,2,...]

seq A admitted (no prefix hit):
  BM.allocate(32)                          1       1       -     [2,...]
  → blk_0, blk_1 assigned to seq A

seq A prefill done (executor):
  PKV.insert_blocks_with_paged(...)
  BM.increment_refs([blk_0, blk_1])        2       2       -     [2,...]
  → prefix cache co-owns blk_0, blk_1

seq B admitted (prefix hit on blk_0, blk_1):
  BM.increment_refs([blk_0, blk_1])        3       3       -     [2,...]  ← seq B takes ownership
  BM.allocate(suffix)                      3       3       1     [3,...]
  → blk_0, blk_1 reused; blk_2 is seq B's fresh suffix block

seq A finishes:
  BM.free([blk_0, blk_1])                 2       2       1     [3,...]
  → ref drops but NOT to 0 — seq B and prefix cache still hold references

seq B finishes:
  BM.free([blk_0, blk_1, blk_2])          1       1       0     [2,3,...]
  → blk_2 returned (rc 1→0)
  → blk_0, blk_1 at rc=1 — prefix cache still holds them

LRU eviction fires (trie over capacity):
  PKV.take_evicted_paged_blocks() → [blk_0, blk_1]
  BM.decrement_refs([blk_0, blk_1])        0       0       -     [0,1,2,3,...]
  → both reach 0 — returned to free pool
─────────────────────────────────────────────────────────────────────────────
```

### Algorithms

```
───────────────────────────────────────────────────────────────────────────
Algorithm 1: PKV.match_prefix(tokens)   [prefix_index.rs:109]
Goal: find the longest prefix of tokens already cached; skip recomputing it
───────────────────────────────────────────────────────────────────────────
Input:  tokens[]
Output: cached_len, hit_block_ids[]

1. max_matchable ← (len(tokens) − 1) / block_size
   [always keep ≥1 token for the suffix forward pass]

2. parent_hash ← 0
   matched ← []
   for i in 0..max_matchable:
       block ← tokens[i*B .. (i+1)*B]
       hash  ← H(parent_hash ‖ block)      [parent-chained: same tokens at
                                             different positions → different hash]
       if hash not in entries: break        [trie miss — stop here]
       touch(hash)                          [refresh LRU so hit blocks stay warm]
       matched.append(hash)
       parent_hash ← hash

3. cached_len  ← len(matched) * block_size
   hit_block_ids ← collect paged_block_ids from each matched entry

4. return cached_len, hit_block_ids
   [executor sets seq.kv_computed_len = cached_len → skips those tokens]

───────────────────────────────────────────────────────────────────────────
Algorithm 2: PKV.insert_blocks(tokens, layer_kvs, block_table)
             [prefix_cache.rs:118, prefix_index.rs:188]
Goal: store newly computed KV blocks in the trie after prefill completes
───────────────────────────────────────────────────────────────────────────
Input:  tokens[], layer_kvs (GPU tensors per layer), block_table[]
Output: stored_paged_block_ids[] (caller must BM.increment_refs on these)

1. full_blocks ← len(tokens) / block_size
   paged_map   ← map each prefix block → overlapping paged block_ids

2. parent_hash ← 0
   for i in 0..full_blocks:
       block ← tokens[i*B .. (i+1)*B]
       hash  ← H(parent_hash ‖ block)

       if hash already in entries:
           touch(hash)                      [already cached — just refresh LRU]
           if entry has no paged_block_ids: attach paged_map[i]
       else:
           if parent had children == 0:
               remove parent from leaf_set  [parent is no longer a leaf]
           parent.children += 1

           entries[hash] ← PrefixEntry {
               parent:          parent_hash,
               paged_block_ids: paged_map[i],
               children:        0,           [starts as a leaf]
               access_id:       next_id(),
           }
           leaf_set.insert(hash)
           leaf_lru.push_back((hash, access_id))

           kv_store[hash] ← layer_kvs[i*B .. (i+1)*B]  [slice KV tensors per layer]

       parent_hash ← hash

3. evict_if_needed()                        [see Algorithm 3 below]

4. return stored_paged_block_ids
   [caller: BM.increment_refs(stored) so prefix cache co-owns these blocks]

───────────────────────────────────────────────────────────────────────────
Algorithm 3: evict_if_needed()   [prefix_index.rs:321]
Goal: keep trie within max_blocks capacity using LRU on leaves only
───────────────────────────────────────────────────────────────────────────
while len(entries) > max_blocks:
    (hash, access_id) ← leaf_lru.pop_front()

    if hash not in leaf_set:   continue     [stale queue entry — skip]
    if entry.access_id != access_id: continue  [entry was re-touched — skip]
    if entry.children > 0:     continue     [promoted to internal — skip]

    entry ← entries.remove(hash)
    leaf_set.remove(hash)
    evicted_hashes.push(hash)               [caller removes kv_store[hash]]
    evicted_paged_blocks += entry.paged_block_ids

    if entry.parent exists:
        parent.children -= 1
        if parent.children == 0:
            leaf_set.insert(parent)         [parent becomes new eviction candidate]
            leaf_lru.push_back((parent, parent.access_id))

[Why leaves only: internal nodes are shared by multiple sequences.
 Evicting an internal node would invalidate all its children's hashes.
 Only childless (leaf) nodes are safe to remove independently.]

───────────────────────────────────────────────────────────────────────────
Algorithm 4: BM.allocate / free / ref counting   [block_manager.rs]
Goal: manage physical GPU KV block ownership with ref counting
───────────────────────────────────────────────────────────────────────────
State:
  free_blocks: VecDeque<block_id>   FIFO pool of unowned blocks
  ref_counts:  Vec<u32>             one entry per block

allocate_for_tokens(n):
  needed ← ceil(n / block_size)
  if needed > len(free_blocks): return None    [OOM → triggers preemption]
  pop `needed` block_ids from free_blocks
  set ref_counts[id] = 1 for each
  return block_ids

free(block_table):                             [called when sequence finishes]
  decrement_refs(block_table)

increment_refs(ids):                           [called after PKV.insert_blocks]
  ref_counts[id] += 1 for each id             [prefix cache now co-owns block]

decrement_refs(ids):                           [called after LRU eviction]
  for each id:
      ref_counts[id] -= 1
      if ref_counts[id] == 0:
          free_blocks.push_back(id)            [back to pool only when all owners release]

slot(block_table, pos):                        [called by executor each forward pass]
  return block_table[pos / block_size] * block_size + (pos % block_size)
  [maps token position → physical index in GPU KV tensor]
```

### BlockManager (`block_manager.rs`)

Fixed GPU KV block pool with ref counting. Each block holds `block_size` tokens of key/value state.

```rust
pub struct BlockManager {
    block_size: usize,
    free_blocks: VecDeque<u32>,   // FIFO free pool
    ref_counts: Vec<u32>,         // per-block ref count
}
```

Key methods:

```rust
// Allocate ceil(num_tokens / block_size) blocks. Returns None if pool exhausted.
pub fn allocate_for_tokens(&mut self, num_tokens: usize) -> Option<Vec<u32>>

// Decrement ref counts; blocks reaching 0 return to free_blocks.
pub fn free(&mut self, block_table: &[u32])

// Called by PrefixKvCache when it stores a reference to a live block.
pub fn increment_refs(&mut self, block_ids: &[u32])

// Called when eviction releases a block from the prefix cache.
pub fn decrement_refs(&mut self, block_ids: &[u32])

// Compute the physical KV tensor slot index for a token position.
// slot = block_table[pos / block_size] * block_size + (pos % block_size)
pub fn slot(block_table: &[u32], pos: usize, block_size: usize) -> i64
```

Block size is auto-tuned per attention backend: 128 tokens with FlashInfer/FA3, 16 otherwise. Override with `PRELUDE_PAGED_BLOCK_SIZE`.

Ref counting is what allows prefix cache sharing: a block can be simultaneously referenced by a running sequence and the prefix cache index. It is only returned to the free pool when both release it.

### PrefixKvCache (`prefix_cache.rs`, `prefix_index.rs`)

Two-layer design: `PrefixMatchIndex` handles the trie logic with no tensors; `PrefixKvCache` wraps it with actual KV tensor storage.

**`PrefixMatchIndex` — tensor-free trie core:**

```rust
struct PrefixEntry {
    parent: Option<u64>,              // hash of parent block (trie linkage)
    paged_block_ids: Option<Vec<u32>>,// physical blocks in BlockManager
    children: usize,                  // number of child blocks
    access_id: u64,                   // LRU timestamp
}

pub struct PrefixMatchIndex {
    entries: HashMap<u64, PrefixEntry>,
    leaf_set: HashSet<u64>,           // blocks with children==0 (eviction candidates)
    leaf_lru: VecDeque<(u64, u64)>,  // (hash, access_id) — LRU order
    max_blocks: usize,
}
```

**Hashing:** each block of `block_size` tokens is hashed with its parent chain, so the same token block at different positions in different sequences gets a different hash:

```rust
fn hash_block(parent_hash: u64, tokens: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}
```

**Matching** (`match_prefix`, `prefix_index.rs:109`): walk the trie block by block until a miss. Always reserves at least 1 suffix token for the forward pass — `max_matchable = (tokens.len() - 1) / block_size`.

**LRU eviction** (`evict_if_needed`, `prefix_index.rs:321`): only leaf nodes (blocks with no children) are eviction candidates. When a leaf is evicted its parent's child count decrements; if the parent reaches 0 children it becomes the new leaf. This preserves shared prefixes: a shared internal block cannot be evicted while any child refers to it.

**`PrefixKvCache` — tensor layer:**

```rust
pub struct PrefixKvCache {
    index: PrefixMatchIndex,
    kv_store: HashMap<u64, Vec<(Tensor, Tensor)>>, // per-block KV per layer
    assembled_cache: HashMap<u64, AssembledKvCache>,// pre-concatenated prefixes
}
```

On a cache hit, `match_and_assemble_paged` returns both the paged block IDs (so the executor can reuse them without recompute) and the assembled KV tensors (so the attention kernel can attend over the cached prefix). On a cache miss, both are empty and the executor runs full prefill.

### RequestQueue (`request_queue.rs`)

> **Planned — not yet extracted.** The queue logic currently lives inline in `state.rs` (`waiting_queue: VecDeque<Sequence>`) and `preemption.rs` (`sort_waiting_queue`). `request_queue.rs` is a placeholder for a future extraction.

Intended to hold waiting sequences with pluggable ordering. Used by `Scheduler` and `DllmScheduler`.

Planned policies:
- **FCFS** (default) — ordered by arrival time
- **Priority** — ordered by explicit `priority` field, then arrival time
- **Cache-aware** — reorders by longest prefix match length, so sequences sharing a long common prefix are batched together while their shared blocks are still hot in `PrefixKvCache`

<!-- ## Advanced Features

### Speculative Decoding

`SpecDecodeRunner` sits between `Scheduler` and the model runner. It uses a pluggable `DraftProposer` to generate draft tokens:

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

Prefill and decode run on separate worker pools. The coordinator routes requests based on prefix cache hits on decode workers. The `Scheduler` core loop is unchanged; the only additions are:

- `FinishReason::Transferred` — prefill worker marks a sequence done after shipping its KV blocks
- `ArRequest::preloaded_blocks` — decode worker receives pre-imported blocks and starts decode immediately
- `BlockAllocator::import_blocks()` — registers incoming KV blocks from the transfer layer

KV transfer is transport-agnostic (`KvTransfer` trait). Mooncake handles the actual transfer (NVLink, RDMA, or TCP relay depending on topology). See [Integration](../integration.md) for details. -->

## Adding a New Scheduler

Only three files need to change:

1. **New scheduler file** — `prelude-core/src/scheduler/your_scheduler.rs`. Use `oneshot.rs` as the simplest reference. Implement a `step()` method that returns `SchedulerStep`.
2. **New run loop** — `prelude-core/src/engine/run/your_loop.rs`. Call `scheduler.step()`, submit to executor, call `scheduler.update()` on completion.
3. **Config branch** — `prelude-core/src/engine/config.rs`. Add a variant to `EngineMode` and wire it to your run loop.

No changes needed to model code, ops, or device crates.

## Design Choices

| Feature | Source | Kept | Dropped |
|---------|--------|------|---------|
| Token-centric scheduling | vLLM V1 | `kv_computed_len` tracks prefill progress; explicit `Prefilling`/`Decoding` states retained | 3-state unified approach |
| Chunked prefill | vLLM V1 | `mixed_chunked` flag for interleaved prefill + decode in same batch | Individual prompt splitting across steps (planned) |
| Preemption | vLLM V1 | Full recompute (simple, no swap complexity) | KV swap to CPU |
| Paged KV blocks | vLLM V1 | `BlockManager` with ref counting | KV connector coupling to specific transfer backends |
| Radix tree prefix cache | SGLang | Block-level hash-trie with LRU eviction | Configurable eviction policies |
| Cache-aware scheduling | SGLang | LPM queue ordering to maximize prefix hits (planned) | — |
| Compute-schedule overlap | SGLang | CPU schedules next batch while GPU runs current | — |
| Adaptive batching | Original | EWMA arrival rate + GPU time → optimal batch size and wait | — |
