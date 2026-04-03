# Disaggregated Serving (`disaggregated/`)

Multi-instance deployment strategies. Single-machine users can skip this entirely.
See [main design doc](README.md) for single-instance architecture.

## Prefill/Decode Separation (`disaggregated/pd/`)

Prefill and decode run on separate worker pools. Prefill workers are optimized for
compute-heavy prefill (high FLOPS utilization). Decode workers are optimized for
memory-bandwidth-bound decode (large batch sizes).

### Architecture

```
Client → Coordinator (routes requests)
              │
              ├→ Prefill Worker 0 ──KV transfer──→ Decode Worker 0
              ├→ Prefill Worker 1 ──KV transfer──→ Decode Worker 1
              └→ Prefill Worker 2 ──KV transfer──→ Decode Worker 0  (load-balanced)
```

**The coordinator is a new component above the per-worker schedulers.**
Each worker runs a standard `ArScheduler` internally. The coordinator
decides routing; the workers don't know about each other.

### What changes in ArScheduler

Almost nothing. Three small additions to support disaggregated mode:

**1. `FinishReason::Transferred`** — prefill worker marks a request as finished
when KV cache transfer completes, not when generation is done.

**2. `ArRequest::preloaded_blocks`** — decode worker receives a request with
pre-loaded KV cache. The scheduler skips prefill and starts decode immediately.

**3. `BlockAllocator::import_blocks`** — decode worker registers externally-received
blocks without going through the normal allocate path.

### Prefill Worker Flow

```rust
// scheduler/ar.rs (P/D disaggregation: prefill worker variant)

impl ArScheduler {
    /// Prefill-only mode: finish after prefill, don't decode.
    fn update_prefill_worker(&mut self, result: &StepResult, plan: &ScheduledBatch,
                             transfer: &KvTransfer) {
        for entry in &plan.entries {
            let req = self.requests.get_mut(&entry.req_id).unwrap();
            req.num_computed_tokens += entry.num_new_tokens;

            // Prefill complete → initiate KV cache transfer
            if req.num_computed_tokens >= req.prompt_tokens.len() {
                transfer.send(KvTransferRequest {
                    req_id: req.id,
                    block_ids: req.block_ids.clone(),
                    prompt_tokens: req.prompt_tokens.clone(),
                    num_computed_tokens: req.num_computed_tokens,
                });
                // Request is done on this worker
                req.status = RequestStatus::Finished(FinishReason::Transferred);
                // Don't free blocks yet — decode worker needs the data.
                // Blocks are freed after transfer confirmation.
            }
        }
    }
}
```

### Decode Worker Flow

```rust
// scheduler/ar.rs (P/D disaggregation: decode worker variant)

impl ArScheduler {
    /// Receive a pre-filled request from a prefill worker.
    fn receive_transferred_request(&mut self, transfer: KvTransferResult) {
        // Import KV cache blocks (already loaded into GPU memory by transfer layer)
        self.block_allocator.import_blocks(&transfer.block_ids);

        let req = ArRequest {
            id: transfer.req_id,
            status: RequestStatus::Waiting,
            prompt_tokens: transfer.prompt_tokens,
            output_tokens: Vec::new(),
            num_computed_tokens: transfer.num_computed_tokens,
            block_ids: transfer.block_ids,
            num_prefix_tokens: transfer.num_computed_tokens,
            preloaded_blocks: Some(transfer.block_ids.clone()),
            ..Default::default()
        };

        self.requests.insert(req.id, req);
        self.waiting.push_back(req.id);
        // Next step(): Phase 2 picks up this request.
        // num_new_tokens() = 1 (first decode token). Standard decode path.
    }
}
```

### Coordinator (`disaggregated/pd/coordinator.rs`)

The coordinator is **not part of ArScheduler** — it sits above, at the orchestration layer.
Its responsibilities:

1. **Routing**: decide which prefill worker handles a new request.
2. **Placement**: decide which decode worker receives the KV cache.
3. **Cross-worker prefix awareness**: know which decode worker has which prefixes cached,
   to avoid redundant KV transfer (e.g., system prompt already on decode worker 0).

```rust
// disaggregated/pd/coordinator.rs

struct Coordinator {
    prefill_workers: Vec<WorkerHandle>,
    decode_workers: Vec<WorkerHandle>,
    /// Which prefixes each decode worker has cached (for KV-aware routing).
    decode_prefix_index: HashMap<WorkerId, PrefixCache>,
}

impl Coordinator {
    fn route_request(&mut self, req: IncomingRequest) -> RoutingDecision {
        // 1. Find decode worker with best prefix cache hit
        let (best_decode, prefix_hit_len) = self.decode_workers.iter()
            .map(|w| {
                let (_, hit) = self.decode_prefix_index[&w.id]
                    .match_prefix(&req.prompt_tokens);
                (w.id, hit)
            })
            .max_by_key(|(_, hit)| *hit)
            .unwrap();

        // 2. Pick least-loaded prefill worker
        let prefill = self.prefill_workers.iter()
            .min_by_key(|w| w.queue_len())
            .unwrap();

        RoutingDecision {
            prefill_worker: prefill.id,
            decode_worker: best_decode,
            prefix_hit_len,
        }
    }
}
```

### KV Transfer Mechanism (`disaggregated/pd/kv_transfer.rs`)

KV cache transfer is a **system-level concern**, not an ops-level or scheduler-level concern.
The transfer layer moves block data between workers using the best available transport:

| Transport | When | Latency |
|-----------|------|---------|
| NVLink P2P | same node, NVLink connected | ~us |
| GPU RDMA (GDR) | same cluster, RDMA fabric | ~100us |
| CPU relay (GPU→CPU→network→CPU→GPU) | cross-node, no RDMA | ~ms |

The scheduler doesn't know which transport is used. It calls `transfer.send()` and
`transfer.receive()`. The transfer layer handles the rest.

### What stays unchanged

- **ArScheduler core loop** (`step()`, `update()`): identical on both prefill and decode workers.
  Prefill worker just never enters decode phase. Decode worker just never enters prefill phase.
- **BlockAllocator**: same logic. `import_blocks()` is a small addition for decode workers.
- **PrefixCache**: same per-worker radix tree. Cross-worker prefix awareness is coordinator's job.
- **Model code**: zero changes. Models don't know about disaggregation.
- **Ops layer**: zero changes. The model runner calls the same `Ops` on both worker types.

## Attention-FFN Disaggregation (`disaggregated/afd/`)

A second form of disaggregation — separating attention and FFN/expert layers onto different
GPU pools. Primarily useful for large MoE models (DeepSeek-V3, Qwen3-MoE) where expert weights
dominate GPU memory. Moving experts to dedicated FFN GPUs frees attention GPUs' memory for
KV cache → larger batch sizes → higher throughput.

See ops dispatch doc for the building block changes (`MoeMode::Disaggregated`,
`CommOps::send/recv`). Here we cover the scheduler impact.

**The attention side's ArScheduler is completely unchanged.** It runs `step()` → `ScheduledBatch` →
`execute()` → `update()` as normal. The attention-FFN communication happens inside
`blocks::moe_layer` during model forward — invisible to the scheduler.

**The FFN side needs a passive follower loop**, because the FFN process doesn't make scheduling
decisions — it just executes FFN layers when the attention side sends hidden states.

```rust
// disaggregated/afd/ffn_follower.rs

/// Runs on FFN workers in attention-FFN disaggregated mode.
/// No scheduler, no request management, no block allocation.
fn run_ffn_follower(model_runner: &ModelRunner, sync: &AfdSync) {
    loop {
        // Wait for attention side to signal "run one forward pass"
        match sync.recv() {
            AfdSignal::Forward => {
                // Run FFN-only forward: iterate MoE layers, recv/compute/send for each.
                // Hidden states arrive via CommOps::recv inside blocks::moe_layer.
                model_runner.execute_ffn_only();
            }
            AfdSignal::Shutdown => break,
        }
    }
}

/// Attention side sends a sync signal before each model.forward().
/// This is called inside the model runner, not the scheduler.
fn execute_with_afd(model_runner: &ModelRunner, plan: &ScheduledBatch, sync: &AfdSync) -> StepResult {
    // Signal FFN workers to start their forward pass
    sync.send(AfdSignal::Forward);
    // Normal model forward — blocks::moe_layer handles send/recv internally
    model_runner.execute(plan)
}
```

### Key design differences from SGLang

| Concern | SGLang | Prelude |
|---------|--------|---------|
| MoE layer | Replace class (`AFDATTNMoE` / `AFDFFNMoE`) | `blocks::moe_layer` absorbs AFD internally |
| Model code | Must swap MoE class per model | Zero changes |
| FFN scheduler | Modified `event_loop_afd_ffn_normal` in main scheduler | Separate `run_ffn_follower` loop, not part of ArScheduler |
| Sync mechanism | ZMQ IPC socket, custom `AFSyncReq` | `AfdSync` abstraction, transport-agnostic |
| Communication | StepMesh (push-pull, tensor caching) | `CommOps::send/recv` (device impl chooses transport) |
| Model support | Qwen3MoE only, hardcoded check | Any MoE model using `blocks::moe_layer` |

**The FFN follower is not an ArScheduler mode.** It's a separate, simpler loop that doesn't
manage requests, blocks, or queues. Keeping it separate from ArScheduler avoids polluting the
core scheduler with AFD concerns. The ArScheduler has no knowledge of AFD — it just calls
`model.forward()`, and the building block handles the rest.
