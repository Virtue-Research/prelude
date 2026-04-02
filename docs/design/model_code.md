## Model Code Pattern

Models compose modules for common patterns and call raw ops for model-specific logic.
Linear projections use the unified `Linear` struct — TP, quantization, and LoRA are
configured at load time and invisible to forward code.

```rust
// prelude-core/src/models/ — LLM attention layer

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

    // Linear handles TP + quant + LoRA internally
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
    let (q, k, v) = modules::split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

    // Fused QK-norm + RoPE (module handles fusion internally)
    let (q, k) = modules::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

    // KV cache + attention (raw ops)
    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
    let h = self.o_proj.forward(&o, ctx, ops)?;

    let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
    let h = modules::gated_mlp(&h, &self.gate, &self.up, &self.down, ops)?;
    Ok((&residual + &h)?)
}
```

```rust
// prelude-core/src/models/flux.rs — Diffusion self-attention, same AttentionOps

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
    let qkv = self.qkv_proj.forward(x, ctx, ops)?;
    let (q, k, v) = modules::split_qkv(&qkv, num_heads, num_heads);
    let params = VarlenParams { mask: MaskType::Bidirectional, .. };
    ops.attn.varlen_attention(&q, &k, &v, &params)
}
```

```rust
// prelude-core/src/models/ — Diffusion cross-attention, Q from decoder, K/V from encoder

fn forward(&self, x: &Tensor, context: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
    let q = self.q_proj.forward(x, ctx, ops)?;
    let k = self.k_proj.forward(context, ctx, ops)?;
    let v = self.v_proj.forward(context, ctx, ops)?;
    let params = VarlenParams {
        cu_seqlens_q: /* decoder seqlens */,
        cu_seqlens_k: /* encoder seqlens */,
        mask: MaskType::Bidirectional,
        ..
    };
    ops.attn.varlen_attention(&q, &k, &v, &params)
}
```
