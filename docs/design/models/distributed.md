## Distributed Execution

### Tensor Parallelism (TP)

TP shards model weights across N GPUs. **Attention ops don't know about TP** — all parallelism is handled by the linear layers that surround them.

```
Input (full hidden_state, identical on all ranks)
    ↓
[Linear tp:Column] QKV projection (each rank: num_heads/TP heads)
    ↓
[Local attention kernel] — each rank computes on its shard, no communication
    ↓
[Linear tp:Row] O projection → all_reduce_sum across ranks
    ↓
Output (full hidden_state, identical on all ranks)
```

Model code with TP — uses the unified `Linear` struct with `TpMode`:

```rust
// prelude-core/src/models/

struct Attention {
    qkv_proj: Linear,  // tp: Column { gather_output: false }
    o_proj: Linear,     // tp: Row
}

impl Attention {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &OpsBundle, kv: &PagedKvCtx) -> Result<Tensor> {
        // Column parallel: local GEMM, output is sharded (heads/TP). No communication.
        let qkv = self.qkv_proj.forward(x, ctx, ops)?;
        let (q, k, v) = models::commons::split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

        // Attention is LOCAL — each rank has its own Q/K/V shard
        ops.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;

        // Row parallel: local GEMM + all_reduce (inside Linear::forward)
        self.o_proj.forward(&o, ctx, ops)
    }
}
```

TP behavior is configured at load time via `TpMode`, not expressed in model code.
Inside `Linear::forward`, TP is three sequential steps — GEMM is always local,
communication happens only in step 3:

```rust
// prelude-core/src/models/commons/linear.rs — Linear::forward internals for TP

impl Linear {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &OpsBundle) -> Result<Tensor> {
        // Step 1: GEMM — always local, on this rank's weight shard.
        //
        // Column parallel (QKV projection):
        //   Full weight: [3*num_heads*head_dim, hidden_dim]
        //   This rank's shard: [3*(num_heads/TP)*head_dim, hidden_dim]
        //   Input x: [total_tokens, hidden_dim] (full, identical on all ranks)
        //   Output:  [total_tokens, 3*(num_heads/TP)*head_dim] (sharded)
        //
        // Row parallel (O projection):
        //   Full weight: [hidden_dim, num_heads*head_dim]
        //   This rank's shard: [hidden_dim, (num_heads/TP)*head_dim]
        //   Input x: [total_tokens, (num_heads/TP)*head_dim] (sharded, from local attention)
        //   Output:  [total_tokens, hidden_dim] (partial sum, needs all_reduce)
        //
        let out = ops.matmul(x, &self.weight)?;  // self.weight is already the shard

        // Step 2: LoRA (if configured) — also local, no communication

        // Step 3: TP communication
        match self.tp {
            TpMode::Row => ops.all_reduce_sum(&out),
            //   Each rank has a partial sum → all_reduce produces full output on all ranks.
            TpMode::Column { gather_output: true } => ops.all_gather(&out, -1),
            //   Each rank has a shard → all_gather reconstructs the full output.
            //   Used when downstream needs full tensor (e.g., MLP gate+up projection).
            _ => Ok(out),
            //   Column { gather: false }: keep sharded. Used for QKV — attention
            //   operates on the local shard (num_heads/TP heads), no communication needed.
        }
    }
}
```

**KV cache with TP:** Each rank holds `num_kv_heads / TP` heads in its cache. No communication during KV cache access — each rank reads its own shard.

**GQA + TP edge case:** When `num_kv_heads < TP` (e.g., 8 KV heads with TP=16), KV heads are replicated: each rank gets 1 KV head replicated from the global set. The attention kernel handles K→Q head broadcasting locally.

### Pipeline Parallelism (PP)

PP splits layers across stages. **Engine-level orchestration, not ops-level.**

```
Stage 0 (GPU 0): layers [0:16]  → send activations → Stage 1 (GPU 1): layers [16:32]
```

Each stage has its own `OpsBundle` bundle (same device type). The engine manages send/recv of activations between stages. No impact on op traits.

### Sequence Parallelism (SP)

For long-sequence diffusion/video models. Two patterns:

**Ulysses (all-to-all):** Shard sequence dim across ranks, all-gather before attention, reduce-scatter after.

```rust
// prelude-core/src/models/ — model-level SP loop

fn sp_attention(x: &Tensor, ops: &OpsBundle) -> Result<Tensor> {
    let x_local = ops.reduce_scatter(x, /*dim=*/0)?;  // shard sequence
    let o_local = ops.varlen_attention(&q, &k, &v, &params)?;  // local attention
    ops.all_gather(&o_local, /*dim=*/0)                 // reconstruct full sequence
}
```

**Ring attention:** Rotate K/V between neighbors. Each rank computes partial attention and accumulates. Requires custom attention loop — not expressible through `AttentionOps` alone. Model owns the ring loop and calls `CommOps` for send/recv between steps.

### Expert Parallelism (EP)

EP distributes **complete experts** across ranks for MoE models (vs TP which shards each expert's weights). A model with 256 experts on EP=8 gives each rank 32 experts.

**Three-phase pattern: dispatch → compute → combine**

```
Phase 1 — DISPATCH: Route tokens to expert-owning ranks (all-to-all)
    Router selects top-K experts per token
    All-to-all sends each token to the rank that owns its expert

Phase 2 — COMPUTE: Local grouped GEMM on owned experts
    Each rank runs grouped_gemm on its local experts
    Only processes tokens routed to its experts

Phase 3 — COMBINE: Send results back to original ranks (all-to-all)
    Reverse all-to-all returns expert outputs to token-owning ranks
    Results weighted by router scores and summed
```

In our design, EP is a **model-level module** using `CommOps` + `GemmOps`:

```rust
// prelude-core/src/models/commons/moe.rs

struct MoELayer {
    gate: MoeGate,                     // router
    expert_weights: Tensor,            // [num_local_experts, N, K]
    ep_size: usize,                    // expert parallel world size
    ep_rank: usize,                    // this rank's EP index
}

impl MoELayer {
    fn forward(&self, x: &Tensor, ops: &OpsBundle) -> Result<Tensor> {
        // 1. Route: select top-K experts per token
        let (topk_ids, topk_weights) = self.gate.route(x)?;

        if self.ep_size > 1 {
            // 2. Dispatch: all-to-all sends tokens to expert owners
            let (recv_tokens, recv_meta) = ep_dispatch(x, &topk_ids, ops)?;

            // 3. Compute: local grouped GEMM on owned experts
            let expert_out = ops.grouped_gemm(
                &recv_tokens, &self.expert_weights,
                &recv_meta.sorted_ids, &recv_meta.expert_ids, &recv_meta.num_tokens,
            )?;

            // 4. Combine: all-to-all sends results back
            ep_combine(&expert_out, &recv_meta, &topk_weights, ops)
        } else {
            // EP=1: standard local MoE (same as Qwen3 example)
            ops.grouped_gemm(x, &self.expert_weights, ..)
        }
    }
}

/// EP dispatch: all-to-all to send tokens to expert-owning ranks.
fn ep_dispatch(x: &Tensor, topk_ids: &Tensor, ops: &OpsBundle) -> Result<(Tensor, DispatchMeta)> {
    let (send_counts, recv_counts) = compute_dispatch_layout(topk_ids, ops.world_size())?;
    let recv_tokens = ops.all_to_all(x, &send_counts, &recv_counts)?;
    Ok((recv_tokens, DispatchMeta { .. }))
}
```

**EP + TP combined:** When EP=8 and TP=2 on 16 GPUs, experts are distributed across 8 EP ranks, and each expert's weight is sharded across 2 TP ranks. After expert compute, results are all-reduced across the TP group.

**Multiple dispatch backends:** The `all_to_all` in `CommOps` is the base primitive. Production systems use specialized backends (DeepEP for NVLink+RDMA, FlashInfer, Mooncake for elastic EP) that fuse quantization + communication for higher throughput. These can be exposed as device-specific optimizations on `CudaOps`, similar to how `CudaOps::precompute_paged_plan_graphed` is CUDA-specific.

### Attention-FFN Disaggregation (AFD)

AFD physically separates attention and FFN/expert layers onto different GPU pools.
For large MoE models (DeepSeek-V3, Qwen3-MoE), this is significant:
- **Attention GPUs**: hold KV cache (memory-bound during decode), no expert weights
- **FFN GPUs**: hold expert weights (compute-bound), no KV cache
- Removing expert weights from attention GPUs frees massive memory for KV cache → larger batch sizes

AFD is handled by `models::commons::moe_layer` as a third distribution mode alongside local and EP:

```rust
// prelude-core/src/models/commons/moe.rs

/// MoE distribution mode. Configured at model load time.
enum MoeMode {
    /// All experts on this GPU. Single-device or TP-only.
    Local,
    /// Expert parallelism: experts distributed across EP ranks via all-to-all.
    ExpertParallel(EpConfig),
    /// Attention-FFN disaggregation: attention and FFN on separate GPU pools.
    Disaggregated(AfdConfig),
}

struct AfdConfig {
    role: AfdRole,
    ffn_target: RemoteTarget,  // address of FFN worker pool
}

enum AfdRole {
    Attention,  // this process runs attention, sends hidden states to FFN
    Ffn,        // this process runs experts, receives hidden states from attention
}
```

```rust
// prelude-core/src/models/commons/moe.rs

pub fn moe_layer(x: &Tensor, gate: &MoeGate, weights: &Tensor,
                 config: &MoeConfig, ops: &OpsBundle) -> Result<Tensor> {
    match &config.mode {
        MoeMode::Local => {
            let (topk_ids, topk_weights) = gate.route(x)?;
            ops.grouped_gemm(x, weights, ..)
        }
        MoeMode::ExpertParallel(ep) => {
            let (topk_ids, topk_weights) = gate.route(x)?;
            let (recv_tokens, meta) = ep_dispatch(x, &topk_ids, ep, ops)?;
            let out = ops.grouped_gemm(&recv_tokens, weights, ..)?;
            ep_combine(&out, &meta, &topk_weights, ops)
        }
        MoeMode::Disaggregated(afd) => match afd.role {
            AfdRole::Attention => {
                // Route locally, then send hidden states + routing info to FFN pool
                let (topk_ids, topk_weights) = gate.route(x)?;
                ops.send(&x, afd.ffn_target)?;
                ops.send(&topk_ids, afd.ffn_target)?;
                ops.send(&topk_weights, afd.ffn_target)?;
                // Wait for FFN result
                ops.recv(afd.ffn_target)
            }
            AfdRole::Ffn => {
                // Receive hidden states + routing info from attention pool
                let hidden = ops.recv(afd.ffn_target)?;
                let topk_ids = ops.recv(afd.ffn_target)?;
                let topk_weights = ops.recv(afd.ffn_target)?;
                // Compute experts locally
                let out = ops.grouped_gemm(&hidden, weights, ..)?;
                // Send result back to attention pool
                ops.send(&out, afd.ffn_target)?;
                Ok(out)
            }
        }
    }
}
```

**Model code is unchanged.** The model still calls `models::commons::moe_layer(&h, &gate, &weights, &config, ops)`.
The `MoeMode` is set at model load time based on deployment configuration. The module
absorbs the disaggregation logic, just like it absorbs EP logic.

**Comparison with SGLang's approach:** SGLang replaces the MoE class with `AFDATTNMoE` / `AFDFFNMoE`
(model code changes). Our approach keeps AFD inside the module — the model never knows.

**Scheduler impact:** The FFN side needs a passive event loop (see scheduler doc: FFN follower mode).
The attention side's scheduler is unchanged — it runs the same `step()` / `update()` loop.
