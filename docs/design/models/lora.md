## LoRA (Low-Rank Adaptation)

Multi-LoRA serving: a single batch contains tokens from different LoRA adapters.
Each token maps to a different adapter (or no adapter).

**Core computation:** `y = W @ x + scale * (lora_b @ lora_a @ x)`

### Where LoRA Sits

LoRA is absorbed by the unified `Linear` struct. When `Linear.lora` is `Some`,
the forward method applies LoRA after the base GEMM. Model code is identical
whether LoRA is active or not.

```rust
// prelude-core/src/models/commons/linear.rs

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
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &OpsBundle) -> Result<Tensor> {
        // 1. Base GEMM (plain or quantized)
        let out = match &self.quant {
            Some(q) => ops.quantized_matmul(x, &self.weight, ...)?,
            None => ops.matmul(x, &self.weight)?,
        };
        // 2. LoRA (fused or fallback) — only if lora is configured AND adapter_ids present
        let out = if let (Some(lora), Some(ids)) = (&self.lora, ctx.adapter_ids) {
            ops.apply_lora(&out, x, lora, ids)?
        } else { out };
        // 3. TP
        match self.tp { /* all_reduce / all_gather / passthrough */ }
    }
}
```

`ops.apply_lora()` in OpsBundle: tries fused BGMV/Punica kernel → fallback to per-adapter matmul.

- **CUDA:** fused kernel. O(1) kernel launch for all adapters.
- **Other devices:** fallback splits batch by adapter_id, runs N matmuls. Correct but slower.
- **No LoRA:** `Linear.lora` is `None`, step 2 is skipped entirely. Zero overhead.

### LoRA + Quantization

Handled naturally by `Linear::forward` — step 1 does quantized GEMM, step 2 does LoRA
in full precision. No special code:

```rust
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
like the base weight. OpsBundle handles the all-gather between A and B phases internally.

### Model code: zero changes

```rust
// models/qwen3.rs — same code with or without LoRA

fn forward(&self, x: &Tensor, ctx: &LayerAttnContext) -> Result<Tensor> {
    let (q, k, v) = self.fused_qkv_projection(x, total_q, &bs, ops)?;
    let (q, k) = ops.qknorm_rope_and_cache(&q, &k, &v, ..., paged_kv)?;
    let attn = ops.varlen_attention(&q, &k, &v, &params)?;
    self.o_proj.forward(&attn, &bs, ops)?    // Linear handles LoRA if configured
}
// Whether LoRA is active depends on how projections were constructed at load time.
// The forward code never knows.
```
