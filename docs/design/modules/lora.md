## LoRA (Low-Rank Adaptation)

Multi-LoRA serving: a single batch contains tokens from different LoRA adapters.
Each token maps to a different adapter (or no adapter).

**Core computation:** `y = W @ x + scale * (lora_b @ lora_a @ x)`

### Where LoRA Sits

LoRA is absorbed by the unified `Linear` struct. When `Linear.lora` is `Some`,
the forward method applies LoRA after the base GEMM. Model code is identical
whether LoRA is active or not.

```rust
// prelude-core/src/modules/linear.rs

struct Linear {
    weight: Tensor,                    // bf16 or quantized
    quant: Option<QuantInfo>,          // scale, scheme
    lora: Option<LoRAInfo>,            // ← LoRA weights live here
    tp: TpMode,
}

struct LoRAInfo {
    lora_a: Tensor,                    // [num_adapters, rank, in]
    lora_b: Tensor,                    // [num_adapters, out, rank]
    scale: f32,                        // alpha / rank
}

impl Linear {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
        // 1. Base GEMM (plain or quantized)
        let out = match &self.quant {
            Some(q) => ops.gemm.quantized_matmul(x, &self.weight, ...)?,
            None => ops.gemm.matmul(x, &self.weight)?,
        };
        // 2. LoRA (fused or fallback) — only if lora is configured AND adapter_ids present
        let out = if let (Some(lora), Some(ids)) = (&self.lora, ctx.adapter_ids) {
            modules::apply_lora(&out, x, lora, ids, ops)?
        } else { out };
        // 3. TP
        match self.tp { /* all_reduce / all_gather / passthrough */ }
    }
}

/// Fused BGMV/Punica on CUDA, per-adapter fallback elsewhere.
fn apply_lora(base_out: &Tensor, x: &Tensor, lora: &LoRAInfo,
              adapter_ids: &Tensor, ops: &Ops) -> Result<Tensor> {
    match ops.fused.fused_lora_matmul(
        x, &lora.lora_a, &lora.lora_b, adapter_ids, lora.scale,
    ) {
        Some(r) => Ok((base_out + &r?)?),
        None => {
            // Fallback: split batch by adapter, N separate matmuls
            let delta = lora_fallback(x, &lora.lora_a, &lora.lora_b, adapter_ids, lora.scale, ops)?;
            Ok((base_out + &delta)?)
        }
    }
}
```

- **CUDA:** `fused_lora_matmul` returns `Some` — Punica/BGMV kernel. O(1) kernel launch for all adapters.
- **Other devices:** returns `None` — fallback splits batch by adapter_id, runs N matmuls. Correct but slower.
- **No LoRA:** `Linear.lora` is `None`, step 2 is skipped entirely. Zero overhead.

### LoRA + Quantization

Handled naturally by `Linear::forward` — step 1 does quantized GEMM, step 2 does LoRA
in full precision. No special code:

```rust
// prelude-core/src/modules/linear.rs — same forward, no special LoRA+quant path

// At load time, Linear is configured with both:
let proj = Linear {
    weight: weight_q4,                         // quantized base
    quant: Some(QuantInfo { scheme: W4A16 { group_size: 128 }, scale: w_scale }),
    lora: Some(LoRAInfo { lora_a, lora_b, scale: 0.5 }),
    tp: TpMode::Column { gather_output: false },
};
// forward() calls quantized_matmul for base, then apply_lora for LoRA. Automatic.
```

### LoRA + TP

With tensor parallelism, `lora_a` is replicated across ranks, `lora_b` is sharded
like the base weight. `apply_lora` handles the all-gather between A and B phases internally:

```rust
// prelude-core/src/modules/linear.rs — inside apply_lora, TP-aware path

let shrunk = matmul(x, &lora.lora_a)?;               // local: x @ lora_a
let gathered = ops.comm.all_gather(&shrunk, -1)?;     // synchronize
let expanded = matmul(&gathered, &lora.lora_b_shard)?; // local: shard of lora_b
```

### Model code: zero changes

```rust
// prelude-core/src/models/llama.rs — same code with or without LoRA

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;    // Linear handles LoRA if configured
    // ... attention ...
    let h = self.o_proj.forward(&o, ctx, ops)?;          // Linear handles LoRA if configured
    // ... MLP ...
}
// Whether LoRA is active depends on how qkv_proj/o_proj were constructed at load time.
// The forward code never knows.
```
