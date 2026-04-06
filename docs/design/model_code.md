## Model Code Pattern

Models compose modules for common patterns and call raw ops for model-specific logic.
Linear projections use the unified `Linear` struct — TP, quantization, and LoRA are
configured at load time and invisible to forward code.

```rust
// prelude-core/src/models/ — LLM attention layer

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &OpsBundle, kv: &PagedKvCtx) -> Result<Tensor> {
    let (residual, h) = models::commons::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

    // Linear handles TP + quant + LoRA internally
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
    let (q, k, v) = models::commons::split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

    // Fused QK-norm + RoPE (module handles fusion internally)
    let (q, k) = models::commons::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

    // KV cache + attention (raw ops)
    ops.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
    let o = ops.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
    let h = self.o_proj.forward(&o, ctx, ops)?;

    let (residual, h) = models::commons::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
    let h = models::commons::gated_mlp(&h, &self.gate, &self.up, &self.down, ops)?;
    Ok((&residual + &h)?)
}
```

```rust
// prelude-core/src/models/flux.rs — Diffusion self-attention, same AttentionOps

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &OpsBundle) -> Result<Tensor> {
    let qkv = self.qkv_proj.forward(x, ctx, ops)?;
    let (q, k, v) = models::commons::split_qkv(&qkv, num_heads, num_heads);
    let params = VarlenParams { mask: MaskType::Bidirectional, .. };
    ops.varlen_attention(&q, &k, &v, &params)
}
```

```rust
// prelude-core/src/models/ — KV sharing (Gemma4 YOCO-style)
// Shared layers reuse source layer's K/V. No new kernel — pure model-level routing.

fn forward(&self, x: &Tensor, ops: &OpsBundle, kv: &PagedKvCtx, shared_kv: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
    let q = self.q_proj.forward(&h, ctx, ops)?;
    let q = ops.rms_norm(&q, &self.q_norm_weight, eps)?;
    let q = rope(&q, cos, sin)?;

    let (k, v) = if let Some((sk, sv)) = shared_kv {
        (sk.clone(), sv.clone())   // Shared: reuse source layer's K/V
    } else {
        let k = self.k_proj.forward(&h, ctx, ops)?;  // Non-shared: compute K/V
        let k = ops.rms_norm(&k, &self.k_norm_weight, eps)?;
        let k = rope(&k, cos, sin)?;
        let v = self.v_norm.forward(&self.v_proj.forward(&h, ctx, ops)?)?;
        (k, v)
    };

    if !is_shared {
        ops.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;  // Write cache
    }
    // Shared layers skip reshape_and_cache — cache is aliased to source layer's cache.
    ops.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?
}

// Engine side: CacheManager aliases cache tensors for shared layers.
// model.kv_cache_sharing() returns [None, None, ..., Some(13), Some(13), ..., Some(14), ...]
// cache[15] = cache[13].clone()  (Arc pointer copy, same underlying memory)
```

```rust
// prelude-core/src/models/ — Diffusion cross-attention, Q from decoder, K/V from encoder

fn forward(&self, x: &Tensor, context: &Tensor, ctx: &BatchState, ops: &OpsBundle) -> Result<Tensor> {
    let q = self.q_proj.forward(x, ctx, ops)?;
    let k = self.k_proj.forward(context, ctx, ops)?;
    let v = self.v_proj.forward(context, ctx, ops)?;
    let params = VarlenParams {
        cu_seqlens_q: /* decoder seqlens */,
        cu_seqlens_k: /* encoder seqlens */,
        mask: MaskType::Bidirectional,
        ..
    };
    ops.varlen_attention(&q, &k, &v, &params)
}
```
