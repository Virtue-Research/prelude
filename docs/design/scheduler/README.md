# Scheduler Architecture

## File Layout

```
crates/prelude-core/src/
├── lib.rs                                 # pub use engine::Engine;
│
├── engine/                                # Engine — the public API, minimal external surface
│   ├── mod.rs                             # pub struct Engine, trait InferenceEngine
│   ├── config.rs                          # EngineConfig
│   ├── weight_loader.rs                   # WeightLoader
│   ├── executor.rs                        # trait Executor { submit(batch) -> Handle, collect(Handle) -> Output }
│   ├── run/                               # Scheduling-paradigm loops
│   │   ├── ar.rs
│   │   ├── dllm.rs
│   │   ├── diffusion.rs
│   │   └── tts.rs
│   ├── speculative/                       # Speculative decoding — see speculative.md
│   │   ├── mod.rs                         # SpecDecodeRunner: draft → verify → accept loop
│   │   ├── proposer.rs                    # trait DraftProposer (EAGLE/DraftModel/Ngram/Medusa)
│   │   ├── rejection.rs                   # Rejection sampling (strict, probabilistic)
│   │   └── tree.rs                        # Tree attention mask construction
│   └── sampling/                          # Sampling orchestration — see constrained_decoding.md
│       ├── mod.rs                         # Sampler: penalties → grammar → ops.sampling → token IDs
│       ├── grammar.rs                     # GrammarManager: async compile + bitmask fill
│       └── logits_processor.rs            # LogitsProcessor trait (penalties, grammar are impls)
│
├── scheduler/                             # Scheduling decisions (pure CPU, no GPU)
│   ├── mod.rs                             # re-exports
│   ├── ar.rs                              # ArScheduler — continuous batching for AR LLMs
│   ├── dllm.rs                            # DllmScheduler — block-level demasking for diffusion LLMs
│   ├── diffusion.rs                       # DiffusionScheduler — denoising loop for image/video
│   ├── oneshot.rs                         # OneShotScheduler — embed, classify, prefill-only
│   ├── tts.rs                             # TtsPipelineScheduler — multi-stage TTS streaming
│   ├── types.rs                           # ScheduledBatch, ScheduledRequest, StepResult, etc.
│   └── components/                        # Optional components, used by schedulers as needed
│       ├── cache/
│       │   ├── mod.rs
│       │   ├── block_manager.rs           # BlockManager (was BlockAllocator)
│       │   ├── prefix_cache.rs
│       │   ├── prefix_index.rs
│       │   ├── kv_buf.rs
│       │   └── deltanet_pool.rs
│       └── request_queue.rs               # RequestQueue — FCFS / priority / cache-aware ordering
│
├── disaggregated/                         # Multi-instance deployment (skip for single-machine)
│   ├── pd/                                # Prefill/Decode separation
│   │   ├── coordinator.rs                 # Cross-worker request routing, prefix-aware placement
│   │   └── kv_transfer.rs                 # KV cache cross-worker transfer protocol
│   └── afd/                               # Attention-FFN separation
│       └── ffn_follower.rs                # Passive FFN process (Engine::run_as_ffn_follower impl)
```

**Reading guide:**
- **`engine/`** — the public API. External users (Rust callers, pyo3 bindings) only interact with
  `Engine`. Everything else is `pub(crate)`.
- **`scheduler/`** — pure scheduling logic, no GPU dependency. Each scheduler file (`ar.rs`,
  `dllm.rs`, etc.) is self-contained and independent. Read only the one for your workload.
- **`scheduler/components/`** — optional reusable components. `BlockAllocator` and `PrefixCache`
  are used by AR and DLLM schedulers. `RequestQueue` is used by all. Schedulers that don't
  need KV cache (diffusion, oneshot) ignore this directory entirely.
- **`engine/run/`** — the main loop for each scheduling paradigm (`run::ar()`, `run::dllm()`, etc.).
  Wires scheduler to executor: `scheduler.step() → executor.submit(batch) → executor.collect() → scheduler.update()`.
- **`engine/executor.rs`** — trait `Executor`: `submit(batch) -> Handle`, `collect(Handle) -> Output`.
  Translates `ScheduledBatch` into tensors, calls `model.forward()`, samples tokens. Shared by all modes.
- **`disaggregated/`** — multi-instance deployment. Single-machine users skip this entirely.
  `pd/` for prefill/decode separation, `afd/` for attention-FFN separation.

## Principles

1. **Scheduler is not a trait.** Different workloads (AR serving, diffusion, DLLM, TTS, embedding)
   have fundamentally different scheduling loops. Abstracting them behind a shared interface
   creates a leaky abstraction that's harder to reason about than the concrete implementations.
   Each workload gets its own scheduler.

2. **No forced abstractions between schedulers.** Different schedulers are built by different
   people (AR serving engineers, diffusion researchers, DLLM researchers). Each person should
   be able to read and modify their scheduler without understanding any other scheduler's code.
   Forced abstractions (shared traits, generic type parameters) require everyone to first learn
   the abstraction layer — cognitive overhead that doesn't exist in the domain itself.

3. **Standalone components, not shared abstractions.** `BlockAllocator`, `PrefixCache`, and
   `RequestQueue` are independent, self-contained components with clear semantics. Schedulers
   that need KV cache management pick up `BlockAllocator` and `PrefixCache`. Schedulers that
   don't (diffusion, embedding) simply ignore them. No coupling, no mandatory dependency.

4. **Model code never sees the scheduler.** The scheduler prepares its output, the model runner
   translates it into `model.forward()` calls using the `Ops` bundle. The model only knows `Ops`.

5. **Scheduler-to-Ops boundary is `OpsSession`.** The scheduler calls `begin_forward()` /
   `end_forward()` and optionally `precompute_paged_plan()`. Nothing else. The scheduler does
   not call attention kernels, GEMM, or norm. It manages metadata (block tables, sequence lengths),
   not computation.

## Goals

1. A person working on one scheduler can do their job without reading any other scheduler's code.
2. Adding a new scheduler does not require modifying existing schedulers.
3. Standalone components (`BlockAllocator`, `PrefixCache`) are optional — use if you need, ignore if you don't.
4. The scheduler makes all scheduling decisions on the CPU. GPU is only used for model execution.

## Architecture

```
API Layer           receives     Requests from clients
                        ↓
Request Queue       orders       by policy (FCFS, priority, cache-aware)
                        ↓
Scheduler           produces     ScheduledBatch
  ├── Block Allocator             manages KV cache block allocation / eviction
  └── Prefix Cache                radix tree for KV cache prefix sharing
                        ↓
Model Runner        translates   ScheduledBatch → model inputs (tensors, metadata)
  │                  calls        ops.session.begin_forward()
  │                  calls        ops.session.precompute_paged_plan(block_tables, cu_seqlens_k)
  │                  calls        model.forward(input_ids, ops, paged_ctx)
  │                  calls        ops.session.end_forward()
  │                  returns      StepResult (sampled token IDs per request)
  ├── Model                       the neural network (Qwen3, Flux, ...)
  └── Ops                         device dispatch layer (CudaOps, RocmOps, ...)
                        ↓
Scheduler           consumes     StepResult → updates request state, emits outputs
                        ↓
Output Processor    streams      tokens / images / audio back to clients
```

**Data flow in one scheduling step:**

```
Scheduler.step()
    │
    ▼
ScheduledBatch                      ← scheduler's decision (CPU-side metadata)
  entries: Vec<ScheduledRequest>       per-request: token_ids, block_table, num_cached_tokens
  total_tokens: usize
    │
    ▼
Executor.execute(&scheduled_batch)
    │  1. Build input tensors from ScheduledBatch:
    │       input_ids:    [total_tokens]           ← concat all entries' token_ids
    │       cu_seqlens_q: [num_requests + 1]       ← cumulative token offsets
    │       cu_seqlens_k: [num_requests + 1]       ← cumulative (cached + new) lengths
    │       block_tables: [num_requests * max_blocks]  ← from entries' block_table
    │       slot_mapping: [total_tokens]           ← block_table → per-token slot indices
    │
    │  2. Call ops:
    │       ops.session.begin_forward()
    │       ops.session.precompute_paged_plan(&block_tables, &cu_seqlens_k, block_size)
    │       logits = model.forward(&input_ids, &ops, &paged_ctx)
    │       ops.session.end_forward()
    │
    │  3. Sample:
    │       sampled_tokens = sample(&logits, &sampling_params)
    │
    ▼
StepResult
  sampled: Vec<(RequestId, Vec<u32>)>   ← per-request sampled token IDs
    │
    ▼
Scheduler.update(&step_result)
    │  advance num_computed_tokens, check stop conditions,
    │  finish/preempt requests, insert into prefix cache
    ▼
  (next step)
```

**The scheduler is the only component that mutates scheduling state** (block tables, request status,
token budgets). The model runner is stateless — it receives a `ScheduledBatch`, translates it into
tensors, calls `model.forward()`, and returns `StepResult`. This separation means the scheduler
can be tested without a GPU.

## Components (`scheduler/components/`)

Optional reusable components that schedulers can pick up if they need them. Each is self-contained
with clear semantics. No scheduler is required to use any of them.

| Component | Used by | Not used by |
|-----------|---------|-------------|
| **BlockAllocator** | ArScheduler, DllmScheduler, TTS (AR stages) | DiffusionScheduler, OneShotScheduler |
| **PrefixCache** | ArScheduler, DllmScheduler | DiffusionScheduler, OneShotScheduler |
| **RequestQueue** | All schedulers | — |

### Block Allocator (`scheduler/components/block_allocator.rs`)

Manages physical KV cache blocks on the device. Used by schedulers that need paged KV cache
(AR, DLLM, TTS AR stages). Schedulers without KV cache (diffusion, embedding) ignore it.

```rust
// scheduler/components/block_allocator.rs

struct BlockAllocator {
    blocks: Vec<Block>,             // all physical blocks, indexed by block_id
    free_list: FreeList,            // doubly-linked list, LRU order
    num_free: usize,
    block_size: usize,             // tokens per block (e.g., 16)
}

struct Block {
    block_id: u32,
    ref_count: u16,                // number of requests referencing this block
    hash: Option<BlockHash>,       // set when block is full and cached (prefix caching)
}

impl BlockAllocator {
    /// Allocate `n` blocks. Returns None if not enough free blocks
    /// (after considering evictable cached blocks).
    fn allocate(&mut self, n: usize) -> Option<Vec<u32>>;

    /// Free blocks by decrementing ref_count. Blocks with ref_count=0
    /// return to free list tail (most recently used).
    fn free(&mut self, block_ids: &[u32]);

    /// Increment ref_count for shared blocks (prefix cache hit).
    fn share(&mut self, block_ids: &[u32]);

    /// Number of blocks available (free + evictable cached blocks).
    fn available(&self) -> usize;

    /// Import externally-loaded blocks (disaggregated serving).
    /// The decode worker receives KV cache data from a prefill worker and writes
    /// it into these block IDs. The blocks are marked as in-use (ref_count = 1)
    /// but were not allocated through the normal allocate() path.
    fn import_blocks(&mut self, block_ids: &[u32]);
}
```

**Design decisions:**

- **Reference counting, not ownership.** Multiple requests can share a cached block (prefix cache).
  A block is only evictable when `ref_count == 0`.

- **LRU eviction order.** The free list is ordered by last access time. When allocating and no
  free blocks exist, evict from head (least recently used). This is simpler than SGLang's
  configurable eviction policies (LRU/LFU/FIFO/MRU) — LRU is sufficient for production workloads.
  If a better policy is needed, replace the free list ordering. Not worth the config surface now.

- **Block size is fixed per allocator.** The engine creates one `BlockAllocator` per cache spec
  (e.g., one for KV cache, one for encoder cache). Different block sizes = different allocators.

### Prefix Cache (`scheduler/components/prefix_cache.rs`)

Radix tree for matching and sharing KV cache block prefixes across requests.
Inspired by SGLang's RadixAttention — the key innovation that makes prefix caching efficient.

```rust
// scheduler/components/prefix_cache.rs

struct PrefixCache {
    root: Node,
    allocator: *mut BlockAllocator,  // borrows allocator for ref_count ops
}

struct Node {
    children: HashMap<u32, Box<Node>>,  // token_id → child
    block_ids: Vec<u32>,                // KV cache blocks for this node's token span
    token_len: usize,                   // number of tokens in this node's span
    lock_count: u32,                    // >0 = pinned by running request, not evictable
    last_access: u64,                   // monotonic timestamp for LRU
}

impl PrefixCache {
    /// Find longest matching prefix for the given token sequence.
    /// Returns (matched_block_ids, num_matched_tokens).
    /// Increments lock_count on matched nodes (caller must unlock when done).
    fn match_prefix(&mut self, tokens: &[u32]) -> (Vec<u32>, usize);

    /// Insert a completed sequence's blocks into the cache.
    /// Splits existing nodes if necessary (radix tree property).
    fn insert(&mut self, tokens: &[u32], block_ids: &[u32]);

    /// Unlock nodes after request finishes. Decrements lock_count.
    fn unlock(&mut self, tokens: &[u32], num_tokens: usize);

    /// Evict unlocked leaf nodes until `n` blocks are freed.
    /// Returns number of blocks actually freed.
    fn evict(&mut self, n: usize) -> usize;
}
```

**Design decisions:**

- **Radix tree, not hash table.** vLLM uses per-block hashing (`SHA256(tokens_in_block)`) for
  prefix matching. This works but has limitations: two requests with a 1-token difference in
  the middle produce completely different hashes for all subsequent blocks.
  A radix tree matches the longest common prefix naturally — shared prefixes share tree paths,
  and divergence points are explicit. SGLang proved this is superior for real workloads
  (system prompts, multi-turn conversations, few-shot examples).

- **Node splitting on insert.** When inserting `[A, B, C, D]` into a tree that has `[A, B, C, E, F]`,
  the node `[A, B, C, E, F]` is split into `[A, B, C]` → `[E, F]` and a new leaf `[D]` is added.
  This is the standard radix tree operation that enables precise prefix sharing.

- **Lock counting prevents eviction of in-use prefixes.** When a request matches a prefix,
  the matched nodes are locked. The eviction loop only evicts unlocked leaf nodes.
  This avoids the race condition where prefix blocks are evicted while a request is using them.

- **No namespace key (for now).** SGLang's `extra_key` separates LoRA adapters and cache salts.
  We skip this for initial implementation — LoRA adapter isolation can be added by including
  adapter_id in the token sequence hash. Don't design for hypothetical requirements.

### Request Queue (`scheduler/components/request_queue.rs`)

Ordered queue of waiting requests. Policy determines scheduling order.

```rust
// scheduler/components/request_queue.rs

struct RequestQueue {
    queue: VecDeque<RequestId>,
    policy: QueuePolicy,
}

enum QueuePolicy {
    /// First come, first served. Simple, fair, predictable.
    Fcfs,
    /// Priority-based. Higher priority requests scheduled first.
    /// Ties broken by arrival time (FCFS within same priority).
    Priority,
    /// Cache-aware: prefer requests with longer prefix cache hits.
    /// Maximizes prefix cache utilization. Inspired by SGLang's LPM policy.
    LongestPrefixMatch,
}
```

**Design decisions:**

- **Cache-aware scheduling (LPM) is a queue policy, not a scheduler mode.** SGLang's insight:
  scheduling requests that share cached prefixes first improves both throughput (less compute)
  and memory efficiency (shared blocks). This belongs in the queue ordering, not the scheduler loop.

- **FCFS is the default.** LPM is only useful when prefix cache hit rates are high (multi-turn chat,
  shared system prompts). For single-turn or diverse workloads, FCFS is better (avoids the overhead
  of radix tree lookups during queue ordering).

## Schedulers

Each scheduler is independent. Read only the one for your workload.

| Scheduler | File | Doc | Uses KV cache |
|-----------|------|-----|---------------|
| **ArScheduler** | `scheduler/ar.rs` | [ar.md](ar.md) | yes (BlockAllocator + PrefixCache) |
| **DllmScheduler** | `scheduler/dllm.rs` | [dllm.md](dllm.md) | yes (BlockAllocator + PrefixCache) |
| **DiffusionScheduler** | `scheduler/diffusion.rs` | [diffusion.md](diffusion.md) | no |
| **TtsPipelineScheduler** | `scheduler/tts.rs` | [tts.md](tts.md) | yes (AR stages reuse ArScheduler) |
| **OneShotScheduler** | `scheduler/oneshot.rs` | [oneshot.md](oneshot.md) | no |

## Component Reuse Matrix

| Component | AR | DLLM | Disaggregated AR | Diffusion | TTS | OneShot |
|-----------|-----|------|-----------------|-----------|-----|--------|
| **BlockAllocator** | yes | yes | yes (per-worker) | — | yes (AR stages) | — |
| **PrefixCache** | yes | yes | yes (per-worker) | — | optional | — |
| **RequestQueue** | yes | yes | yes (per-worker) | yes | yes (per-stage) | yes |
| **StreamBuffer** | — | — | — | — | yes (inter-stage) | — |
| **Coordinator** | — | — | yes | — | — | — |
| **FFN Follower** | — | — | AFD only | — | — | — |

## Scheduler ↔ Ops Interface (`engine/executor.rs`)

The scheduler interacts with the ops layer through the Executor trait:

```
Scheduler → ScheduledBatch → Executor::submit(batch) → model.forward(ops) → Executor::collect() → StepResult
```

The scheduler **never** calls `AttentionOps`, `GemmOps`, `NormOps`, etc.
It only manages metadata: block tables, sequence lengths, token IDs.

```rust
// engine/executor.rs — Executor translates ScheduledBatch into model calls

let handle = executor.submit(scheduled_batch);
// ... scheduler can prepare next batch concurrently ...
let step_result = executor.collect(handle);
```

Inside `Executor::submit`, the implementation calls:
```rust
ops.session.begin_forward();
ops.session.precompute_paged_plan(&block_tables, &cu_seqlens_k, block_size);
model.forward(&input_ids, &ops, &paged_ctx);
ops.session.end_forward();
```

For CUDA graphs, the executor (not scheduler) downcasts to `CudaOps`:
```rust
let cuda_ops: &CudaOps = ops.downcast();
cuda_ops.precompute_paged_plan_graphed(&block_tables, &cu_seqlens_k, block_size, &graph_bufs);
```

The scheduler doesn't know about CUDA graphs. The executor decides whether to use
graph capture/replay based on batch shape stability.

## Design Comparisons

See [design-comparisons.md](design-comparisons.md) for detailed
tables of what we take from vLLM V1 and SGLang, and what we discard from both.

## Disaggregated Serving (`disaggregated/`)

See [disaggregated.md](disaggregated.md) for full design
including P/D separation (coordinator, KV transfer, prefill/decode worker flows) and
attention-FFN disaggregation (FFN follower, AFD).

Summary: Prefill and decode run on separate worker pools. Prefill workers are optimized for
compute-heavy prefill (high FLOPS utilization). Decode workers are optimized for
memory-bandwidth-bound decode (large batch sizes).

## Speculative Decoding

See [speculative.md](speculative.md) for the draft-then-verify framework:
`DraftProposer` trait (EAGLE/DraftModel/Ngram/Medusa pluggable), modular rejection sampling
(strict/probabilistic), tree attention via `MaskType::Custom`, no KV cache rollback.

## Constrained Decoding (Structured Output)

See [constrained_decoding.md](constrained_decoding.md) for grammar-based token filtering:
`GrammarBackend` trait (xgrammar/outlines pluggable), async compilation overlapping with GPU,
per-request bitmask, integration with sampling pipeline and speculative decoding.

## Examples

See [examples.md](examples.md) for 10 concrete scenarios:
AR prefix caching, chunked prefill, speculative decode, preemption, diffusion batch,
TTS streaming, embedding, DLLM demasking, P/D disaggregation, attention-FFN disaggregation.

## Workflows

See [workflows.md](workflows.md) for end-to-end call flows (AR serving, DLLM demasking,
diffusion, P/D disaggregation, adding a new scheduler) with file references.

## Summary

| Concern | Solution |
|---------|----------|
| AR continuous batching | `ArScheduler`: token-centric, unified prefill/decode |
| Prefix caching | `PrefixCache`: radix tree, shared blocks, lock counting |
| KV cache allocation | `BlockAllocator`: reference-counted blocks, LRU eviction |
| Chunked prefill | `ArScheduler`: cap `num_new_tokens` at threshold, resume next step |
| Preemption | Free blocks, reset state, re-queue. Prefix cache may recover partial state |
| Speculative decoding | Draft tokens appended to request, standard scheduling + rejection rewind |
| Compute-schedule overlap | Async execute on GPU, schedule next batch on CPU concurrently |
| Diffusion | `DiffusionScheduler`: job queue, fixed-step denoising loop |
| DLLM demasking | `DllmScheduler`: job queue, iterative demasking until all masks replaced |
| TTS streaming | `TtsPipelineScheduler`: chain `ArScheduler` stages with `StreamBuffer` |
| Embedding / prefill-only | `OneShotScheduler`: one forward per request, no KV cache, no decode loop |
| P/D disaggregation | Coordinator above per-worker ArSchedulers. `FinishReason::Transferred` + `preloaded_blocks` + `import_blocks()`. ArScheduler core loop unchanged |
| A/F disaggregation (AFD) | FFN follower loop (separate from ArScheduler). ArScheduler unchanged — AFD hidden inside `modules::moe_layer` |
| Scheduler ↔ Ops | `OpsSession::begin/end_forward` + `precompute_paged_plan`. Nothing else |
| Model code | Unchanged. Models see `Ops`, not schedulers |
