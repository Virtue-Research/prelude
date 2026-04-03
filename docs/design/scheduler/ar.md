# AR Serving Scheduler (`scheduler/ar.rs`)

Back to [main design doc](README.md).


The main scheduler for autoregressive LLM inference. Handles continuous batching, chunked prefill,
prefix caching, preemption, and speculative decoding.

### Core State

```rust
// scheduler/ar.rs

struct ArScheduler {
    /// Requests currently being computed (prefill or decode).
    running: Vec<RequestId>,
    /// Requests waiting to be scheduled.
    waiting: RequestQueue,
    /// All active requests (running + waiting), keyed by ID.
    requests: HashMap<RequestId, ArRequest>,
    /// KV cache block allocator.
    block_allocator: BlockAllocator,
    /// Prefix cache (radix tree).
    prefix_cache: PrefixCache,
    /// Configuration.
    config: ArSchedulerConfig,
}

struct ArRequest {
    id: RequestId,
    status: RequestStatus,

    // ── Token state ────────────────────────────────────
    prompt_tokens: Vec<u32>,         // original prompt
    output_tokens: Vec<u32>,         // generated tokens so far
    num_computed_tokens: usize,      // tokens with KV cache computed

    // ── KV cache state ─────────────────────────────────
    block_ids: Vec<u32>,             // allocated block IDs (ordered)
    num_prefix_tokens: usize,        // tokens from prefix cache hit (no compute needed)

    // ── Generation config ──────────────────────────────
    max_output_tokens: usize,
    stop_token_ids: Vec<u32>,
    sampling: SamplingParams,

    // ── Speculative decoding (optional) ────────────────
    draft_token_ids: Vec<u32>,       // proposed tokens from draft model

    // ── Disaggregated serving (optional) ───────────────
    /// Pre-loaded KV cache blocks received from a prefill worker.
    /// When set, the request arrives with KV already computed — skip prefill,
    /// start decode immediately. Block IDs refer to blocks imported via
    /// BlockAllocator::import_blocks().
    preloaded_blocks: Option<Vec<u32>>,
}

enum RequestStatus {
    Waiting,
    Running,
    Preempted,                       // freed KV cache, back to waiting
    Finished(FinishReason),
}

enum FinishReason {
    Stop,           // EOS or stop string
    Length,         // max_output_tokens reached
    Abort,          // client cancelled
    Transferred,    // KV cache transferred to decode worker (disaggregated serving)
}

struct ArSchedulerConfig {
    max_running_requests: usize,
    max_tokens_per_step: usize,      // total token budget per forward pass
    chunked_prefill_threshold: usize, // split prefills longer than this (e.g., 8192)
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
}
```

### Token-Centric Scheduling

Inspired by vLLM V1: no artificial prefill/decode boundary. Each request has `num_computed_tokens`,
and the scheduler computes `num_new_tokens = total_tokens - num_computed_tokens` to decide how
many tokens to schedule. This naturally handles:

- **Initial prefill:** `num_computed_tokens = 0` → schedule all prompt tokens (or a chunk)
- **Continued prefill (chunked):** `num_computed_tokens < prompt_len` → schedule next chunk
- **Decode:** `num_computed_tokens == total_len - 1` → schedule 1 token
- **Prefix cache hit:** `num_computed_tokens = num_prefix_tokens` → skip cached prefix
- **Speculative decode:** schedule `1 + num_draft_tokens` verification tokens

### Scheduling Loop

```rust
// scheduler/ar.rs

impl ArScheduler {
    /// Called once per forward pass. Produces a ScheduledBatch for the model runner.
    fn step(&mut self) -> ScheduledBatch {
        let mut plan = ScheduledBatch::new();
        let mut token_budget = self.config.max_tokens_per_step;

        // ── Phase 1: Schedule running requests ──────────────────
        // Running requests always get scheduled first (avoid starvation).
        let mut to_preempt = Vec::new();
        for &req_id in &self.running {
            let req = &self.requests[req_id];
            let mut num_new = req.num_new_tokens();

            // Chunked prefill: cap if over threshold
            if self.config.enable_chunked_prefill && num_new > self.config.chunked_prefill_threshold {
                num_new = self.config.chunked_prefill_threshold;
            }

            // Allocate KV cache blocks for new tokens
            let blocks_needed = self.blocks_needed(req, num_new);
            while self.block_allocator.available() < blocks_needed {
                // Not enough blocks — preempt lowest-priority running request
                match self.select_preempt_victim() {
                    Some(victim) => to_preempt.push(victim),
                    None => break, // can't preempt further — this request also stalls
                }
            }

            if let Some(new_blocks) = self.block_allocator.allocate(blocks_needed) {
                plan.add(req_id, num_new, &new_blocks);
                token_budget -= num_new;
            }
        }

        // Execute preemptions
        for victim in to_preempt {
            self.preempt(victim);
        }

        // ── Phase 2: Schedule waiting requests ──────────────────
        while token_budget > 0 && self.running.len() < self.config.max_running_requests {
            let req_id = match self.waiting.pop() {
                Some(id) => id,
                None => break,
            };
            let req = &mut self.requests[req_id];

            // Prefix cache lookup (first schedule only)
            if self.config.enable_prefix_caching && req.num_computed_tokens == 0 {
                let (cached_blocks, num_cached) =
                    self.prefix_cache.match_prefix(&req.prompt_tokens);
                req.num_prefix_tokens = num_cached;
                req.num_computed_tokens = num_cached;
                self.block_allocator.share(&cached_blocks);
                req.block_ids.extend_from_slice(&cached_blocks);
            }

            let mut num_new = req.num_new_tokens();
            if self.config.enable_chunked_prefill && num_new > self.config.chunked_prefill_threshold {
                num_new = self.config.chunked_prefill_threshold;
            }
            num_new = num_new.min(token_budget);

            let blocks_needed = self.blocks_needed(req, num_new);
            match self.block_allocator.allocate(blocks_needed) {
                Some(new_blocks) => {
                    req.block_ids.extend_from_slice(&new_blocks);
                    plan.add(req_id, num_new, &new_blocks);
                    token_budget -= num_new;
                    req.status = RequestStatus::Running;
                    self.running.push(req_id);
                }
                None => {
                    // Not enough blocks even after considering eviction.
                    // Put back in queue and stop scheduling new requests.
                    self.waiting.push_front(req_id);
                    break;
                }
            }
        }

        plan
    }
}
```

### ScheduledBatch (`scheduler/types.rs`)

The scheduler's output. Contains everything the model runner needs to execute.

```rust
// scheduler/types.rs

struct ScheduledBatch {
    /// Per-request scheduling: which tokens to compute.
    entries: Vec<ScheduledRequest>,
    /// Total tokens across all requests.
    total_tokens: usize,
}

struct ScheduledRequest {
    req_id: RequestId,
    /// Token IDs to process in this step.
    /// Prefill: prompt[computed..computed+num_new]. Decode: [last_output_token].
    /// Spec decode verify: [accepted_token, draft_0, draft_1, ...].
    token_ids: Vec<u32>,
    /// Number of new tokens to compute (= token_ids.len()).
    num_new_tokens: usize,
    /// Block table for paged attention: all blocks for this request's KV cache.
    block_table: Vec<u32>,
    /// Number of tokens already in KV cache (for cu_seqlens_k computation).
    num_cached_tokens: usize,
    /// Is this a new request (first time scheduled)?
    is_new: bool,
}
```

**Design decisions:**

- **Flat structure, not new/cached split.** vLLM splits `SchedulerOutput` into `scheduled_new_reqs`
  and `scheduled_cached_reqs` to minimize data transfer (send full data for new, deltas for cached).
  This optimization adds complexity. We use a single `entries` list — the model runner can
  diff locally. Simpler to reason about, and in Rust the data is already in-process (no pickling
  or IPC overhead like Python).

- **`block_table` is the full table, not a delta.** The model runner needs the complete block table
  for `precompute_paged_plan`. Sending deltas forces the runner to maintain state, making it harder
  to test independently. The block table is small (kilobytes) — full copy is fine.

- **`token_ids` is explicit.** The scheduler resolves what tokens to compute. The model runner
  doesn't need to know about prompt vs output vs draft — it just processes `token_ids`.

### Executor (`engine/executor.rs`)

Device-specific execution strategy. Receives `ScheduledBatch`, runs model forward on
the device, returns output. Implemented by each device crate (CudaExecutor, CpuExecutor, etc.).

```rust
// prelude-core/src/engine/executor.rs

trait Executor: Send + Sync {
    /// Submit a batch for execution. Non-blocking on GPU (queues work),
    /// blocking on CPU (runs inline). Returns a handle for collection.
    fn submit(&self, batch: &ScheduledBatch) -> ExecutorHandle;

    /// Collect results from a submitted batch. Blocks until complete.
    fn collect(&self, handle: ExecutorHandle) -> StepResult;
}
```

The run loop uses submit/collect to naturally get double-buffering on GPU
(prepare batch N+1 while device runs batch N) and sequential execution on CPU.

struct StepResult {
    /// Per-request: sampled token IDs.
    sampled: Vec<(RequestId, Vec<u32>)>,
}
```

### State Update After Execution

```rust
// scheduler/ar.rs

impl ArScheduler {
    fn update(&mut self, result: &StepResult, plan: &ScheduledBatch) {
        // Advance num_computed_tokens for all scheduled requests
        for entry in &plan.entries {
            let req = self.requests.get_mut(&entry.req_id).unwrap();
            req.num_computed_tokens += entry.num_new_tokens;
        }

        // Process sampled tokens
        for (req_id, sampled_ids) in &result.sampled {
            let req = self.requests.get_mut(req_id).unwrap();

            // Handle speculative decode rejection
            if !req.draft_token_ids.is_empty() {
                let num_accepted = sampled_ids.len() - 1;
                let num_rejected = req.draft_token_ids.len() - num_accepted;
                req.num_computed_tokens -= num_rejected;
                req.draft_token_ids.clear();
            }

            req.output_tokens.extend_from_slice(sampled_ids);

            // Check stop conditions
            if req.should_stop() {
                self.finish_request(*req_id);
            }
        }
    }

    fn finish_request(&mut self, req_id: RequestId) {
        let req = self.requests.get_mut(&req_id).unwrap();
        req.status = RequestStatus::Finished(req.finish_reason());

        // Insert into prefix cache before freeing
        if self.config.enable_prefix_caching {
            let all_tokens: Vec<u32> = req.prompt_tokens.iter()
                .chain(req.output_tokens.iter())
                .copied().collect();
            self.prefix_cache.insert(&all_tokens, &req.block_ids);
            self.prefix_cache.unlock(&req.prompt_tokens, req.num_prefix_tokens);
        }

        // Free blocks (ref_count decrement — shared prefix blocks stay if ref_count > 0)
        self.block_allocator.free(&req.block_ids);
        self.running.retain(|&id| id != req_id);
    }
}
```

### Preemption

When KV cache is exhausted, the scheduler preempts a running request to free blocks.

```rust
// scheduler/ar.rs

impl ArScheduler {
    fn preempt(&mut self, req_id: RequestId) {
        let req = self.requests.get_mut(&req_id).unwrap();

        // Unlock prefix cache nodes
        if self.config.enable_prefix_caching {
            self.prefix_cache.unlock(&req.prompt_tokens, req.num_prefix_tokens);
        }

        // Free all blocks
        self.block_allocator.free(&req.block_ids);
        req.block_ids.clear();

        // Reset computation state — must recompute from scratch
        // (prefix cache may still have blocks, re-matched on next schedule)
        req.num_computed_tokens = 0;
        req.num_prefix_tokens = 0;
        req.draft_token_ids.clear();
        req.status = RequestStatus::Preempted;

        self.running.retain(|&id| id != req_id);
        self.waiting.push_front(req_id);
    }

    fn select_preempt_victim(&self) -> Option<RequestId> {
        // Preempt the request with fewest computed tokens (least work wasted).
        // Alternative: preempt longest-running (most blocks freed).
        self.running.iter()
            .filter(|&&id| id != self.running[0]) // don't preempt the request we're trying to schedule
            .min_by_key(|&&id| self.requests[&id].num_computed_tokens)
            .copied()
    }
}
```

**Design decision: full recompute on preemption, no partial save.**

vLLM also does full recompute. The alternative (saving partial KV state to host memory)
adds significant complexity for marginal benefit — preemption is rare in well-tuned deployments.
When the preempted request is re-scheduled, prefix cache hit may recover some of its KV cache
(if other requests haven't evicted those blocks). This is "free" partial recovery without
any explicit save/restore mechanism.

### Speculative Decoding Integration

Speculative decoding is scheduler-level coordination, not a scheduler variant. The AR scheduler
handles it as a special case within the same loop.

```rust
// scheduler/ar.rs

impl ArScheduler {
    fn step_with_speculation(&mut self, draft_model: &Executor) -> ScheduledBatch {
        // 1. Draft phase: generate N draft tokens for each decode request
        for &req_id in &self.running {
            let req = &mut self.requests[req_id];
            if req.is_decode_phase() {
                let draft_plan = self.build_draft_plan(req_id);
                let draft_result = draft_model.execute(&draft_plan);
                req.draft_token_ids = draft_result.sampled[0].1.clone();
            }
        }

        // 2. Verify phase: schedule all running requests (including draft tokens)
        //    num_new_tokens for decode requests = 1 (accepted) + N (drafts) = N+1
        self.step()  // standard scheduling — draft tokens are just extra tokens to verify
    }
}

impl ArRequest {
    fn num_new_tokens(&self) -> usize {
        let total = self.prompt_tokens.len() + self.output_tokens.len()
            + self.draft_token_ids.len();
        total - self.num_computed_tokens
    }
}
```

The draft tokens are appended to the request's token sequence. `num_new_tokens()` naturally
includes them. After verification, `update()` handles rejection by rewinding `num_computed_tokens`.

### Compute-Schedule Overlap (`engine/run.rs`)

Overlap GPU execution with CPU scheduling to hide scheduling latency.

```
Step N:
  GPU: execute(plan_N)           ──────────────────→
  CPU:                    update(result_{N-1})  →  plan_{N+1} = step()
```

```rust
// engine/run.rs

fn run_loop(scheduler: &mut ArScheduler, runner: &Executor) {
    let mut plan = scheduler.step();
    loop {
        // Launch GPU execution (non-blocking with async ops)
        let result_future = runner.execute_async(&plan);

        // While GPU works on plan_N, prepare plan_{N+1}
        if let Some(prev_result) = prev_result.take() {
            scheduler.update(&prev_result, &prev_plan);
            // Emit outputs to clients for finished requests
            emit_outputs(scheduler);
        }

        // Wait for GPU result
        let result = result_future.wait();
        prev_result = Some(result);
        prev_plan = plan;

        // Schedule next step
        plan = scheduler.step();
    }
}
```

