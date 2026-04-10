## Model Code Pattern

Models are self-contained (like vLLM): each model file defines its own structs and forward
logic, 1:1 mapping to HuggingFace transformers. Inference details (paged attention, fused
kernels, CUDA graph) are hidden inside shared layer abstractions.

### Decoder Layer (vLLM-style residual threading)

```rust
// prelude-core/src/models/qwen3.rs — DecoderLayer

fn forward(&self, hidden: &Tensor, residual: Option<&Tensor>, ctx: &LayerAttnContext)
    -> Result<(Tensor, Tensor)>
{
    let ops = ctx.ops;
    // RmsNorm.forward_residual: fused add+norm when residual is Some
    let (residual, hidden) = self.input_layernorm.forward_residual(hidden, residual, ops)?;
    let hidden = self.self_attn.forward(&hidden, ctx)?;
    let (residual, hidden) = self.post_attention_layernorm.forward_residual(&hidden, Some(&residual), ops)?;
    let hidden = self.mlp.forward(&hidden, ops)?;
    Ok((hidden, residual))
}
```

### Model Backbone (residual chain)

```rust
// prelude-core/src/models/qwen3.rs — Model

fn forward(&mut self, packed_input: &Tensor, ctx: &mut BatchAttnContext) -> Result<Tensor> {
    let mut hidden = self.embed_tokens.forward(packed_input)?;
    let mut residual: Option<Tensor> = None;
    for (i, layer) in self.layers.iter_mut().enumerate() {
        let layer_ctx = ctx.layer(i);
        let (h, r) = layer.forward(&hidden, residual.as_ref(), &layer_ctx)?;
        hidden = h;
        residual = Some(r);
    }
    let (_, normed) = self.norm.forward_residual(&hidden, residual.as_ref(), ctx.ops)?;
    Ok(normed)
}
```

### Where kernels are hidden

- **`RmsNorm.forward_residual()`** — calls `ops.rms_norm()` / `ops.add_rmsnorm()` → fused CUDA PTX
- **`self_attn.forward()`** — internally does QKV proj → qknorm+rope → KV cache write → paged attention (FA4/FlashInfer)
- **`mlp.forward()`** — gate/up proj → `ops.silu_mul()` (fused PTX) → down proj
- **`Linear.forward()`** — delegates to the active `LinearBackend` impl (`DenseLinear` →
  `Tensor::matmul()` → registered GEMM dispatch → DeepGEMM/CUTLASS; `OnednnLinear` → oneDNN
  packed GEMM; `Q4_0Linear` / `Q4KLinear` / `GpuQuantLinear` → quantized matmul). Backend
  is picked at load time from the checkpoint format

Model code never calls fused kernels directly — all dispatch is inside the layer abstractions.

### KV Sharing (Gemma4 YOCO-style)

Shared layers reuse source layer's K/V. No new kernel — pure model-level routing.

```rust
fn forward(&self, x: &Tensor, ops: &dyn Ops, kv: &PagedKvCtx, shared_kv: Option<(&Tensor, &Tensor)>)
    -> Result<Tensor>
{
    let q = self.q_proj.forward(&h, &bs, ops)?;
    let (k, v) = if let Some((sk, sv)) = shared_kv {
        (sk.clone(), sv.clone())   // Shared: reuse source layer's K/V
    } else {
        // Non-shared: compute K/V, write cache
        let k = self.k_proj.forward(&h, &bs, ops)?;
        let v = self.v_proj.forward(&h, &bs, ops)?;
        ops.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        (k, v)
    };
    ops.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?
}
```
