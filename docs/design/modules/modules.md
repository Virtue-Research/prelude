## Shared Building Blocks

Modules are **shared layer implementations** that contain fusion/fallback logic.
Models compose them instead of calling raw ops. This is how one kernel optimization
reaches all models automatically.

### Why Building Blocks

**Problem:** Without modules, every model must write its own fusion fallback:

```rust
// Qwen3 model code — manual fusion logic
let (residual, h) = match ops.fused.fused_add_rmsnorm(x, &res, &w, eps) {
    Some(r) => r?,
    None => { let r = (x + &res)?; let h = ops.norm.rms_norm(&r, &w, eps)?; (r, h) }
};
```

If 20 models have this pattern, adding `fused_add_rmsnorm` requires updating all 20.

**Solution:** The module contains the logic once:

```rust
// prelude-core/src/modules/norm.rs — written once, used by all models
pub fn residual_norm(
    residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32, ops: &Ops,
) -> Result<(Tensor, Tensor)> {
    match ops.fused.fused_add_rmsnorm(residual, x, weight, eps) {
        Some(r) => r,
        None => {
            let r = (residual + x)?;
            let h = ops.norm.rms_norm(&r, weight, eps)?;
            Ok((r, h))
        }
    }
}
```

Now the model just writes `modules::residual_norm(&residual, &h, &w, eps, ops)?`.
When the kernel dev adds `fused_add_rmsnorm` to CudaOps, all 20 models benefit with zero changes.

### Building Block Catalog

```rust
// prelude-core/src/modules/

// ── norm.rs ─────────────────────────────────────────────────────

/// Residual add + RMSNorm. Fuses to 1 kernel on CUDA, 2 ops elsewhere.
pub fn residual_norm(residual, x, weight, eps, ops) -> (Tensor, Tensor);

/// Residual add + LayerNorm. Same pattern for diffusion models.
pub fn residual_layer_norm(residual, x, weight, bias, eps, ops) -> (Tensor, Tensor);

/// AdaLN-Zero: layer_norm + scale + shift + gate. Fuses to 1 kernel on CUDA.
pub fn adaln_zero(x, weight, bias, scale, shift, gate, eps, ops) -> (Tensor, Tensor);

/// AdaLN continuous: layer_norm + scale + shift (no gate).
pub fn adaln_continuous(x, weight, bias, scale, shift, eps, ops) -> Tensor;

// ── attn_utils.rs ──────────────────────────────────────────────

/// QK-norm + RoPE. Fuses to 1 kernel on CUDA (FlashInfer fused_qknorm_rope).
pub fn qk_norm_rope(q, k, qw, kw, cos, sin, pos, eps, ops) -> (Tensor, Tensor);

/// K-norm + RoPE + KV cache write. Fuses to 1 kernel on CUDA.
pub fn knorm_rope_cache_write(k, v, kw, cos, sin, pos, cache_k, cache_v, slots, eps, ops) -> ();

/// Apply rotary position embedding to Q/K. Pure math, no device dispatch.
pub fn apply_rope(q, k, cos, sin) -> (Tensor, Tensor);

/// Split fused QKV projection output into separate Q, K, V tensors.
pub fn split_qkv(qkv, num_heads, num_kv_heads) -> (Tensor, Tensor, Tensor);

// ── mlp.rs ──────────────────────────────────────────────────────

/// SiLU-gated MLP: silu(gate) * up → down. Fuses gate*up to 1 kernel.
/// gate/up are Column parallel (no comm), down is Row parallel (all_reduce inside).
pub fn gated_mlp(x: &Tensor, gate: &Linear, up: &Linear, down: &Linear,
                 ctx: &BatchState, ops: &Ops) -> Result<Tensor>;

/// GELU MLP (diffusion): gelu(fc1) → fc2. Uses gelu_approximate on CUDA.
pub fn gelu_mlp(x: &Tensor, fc1: &Linear, fc2: &Linear,
                ctx: &BatchState, ops: &Ops) -> Result<Tensor>;

// ── linear.rs ───────────────────────────────────────────────────

/// Unified linear layer. Handles TP, quantization, and LoRA internally.
/// Configured at load time — model forward code is identical regardless of
/// which combination of TP/quant/LoRA is active.
struct Linear {
    weight: Tensor,                    // bf16 or quantized
    quant: Option<QuantInfo>,          // scale, scheme (FP8, W4A16, ...)
    lora: Option<LoRAInfo>,            // lora_a, lora_b, scale
    tp: TpMode,                        // None, Column { gather }, Row
}

enum TpMode { None, Column { gather_output: bool }, Row }

impl Linear {
    /// forward: GEMM (plain/quantized) → LoRA (fused/fallback) → TP (reduce/gather).
    /// ~15 lines, three sequential steps. Complexity lives in device impls, not here.
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor>;
}

/// Apply LoRA residual: scale * (lora_b @ lora_a @ x). Called by Linear internally.
/// Fuses to BGMV/Punica on CUDA, per-adapter fallback elsewhere.
fn apply_lora(base_out, x, lora, adapter_ids, ops) -> Tensor;

// ── moe.rs ──────────────────────────────────────────────────────

/// MoE layer: route → dispatch → grouped GEMM → combine.
/// Handles three modes internally: local, expert parallel (EP), and
/// attention-FFN disaggregation (AFD). Model code is the same for all modes.
pub fn moe_layer(x, gate, expert_weights, moe_config, ops) -> Tensor;
```

### How Models Use Modules

```rust
// prelude-core/src/models/qwen3.rs — using modules, no fusion logic visible
fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    // 1. Pre-attention norm (module handles fusion internally)
    let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

    // 2. QKV projection — Linear handles TP + quantization + LoRA internally
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
    let (q, k, v) = modules::split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

    // 3. QK-norm + RoPE (module handles fusion internally)
    let (q, k) = modules::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

    // 4. Attention (raw op — no fusion opportunity here)
    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
    let h = self.o_proj.forward(&o, ctx, ops)?;

    // 5. Post-attention norm + MoE (module handles EP + fusion)
    let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
    let h = modules::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
    Ok((&residual + &h)?)
}
```

```rust
// prelude-core/src/models/flux.rs — same modules, different composition
fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<..> {
    let (scale1, shift1, gate1, ..) = self.img_mod.forward(temb)?;

    // AdaLN-Zero (module handles fusion internally)
    let (img_normed, img_gate) = modules::adaln_zero(
        img, &self.img_ln, None, &scale1, &shift1, &gate1, eps, ops,
    )?;

    // QK-norm (same module as Qwen3, different context)
    let img_q = ops.norm.rms_norm(&img_q, &self.img_q_norm, eps)?;
    // ...

    // Joint attention (raw op)
    let attn_out = ops.attn.varlen_attention(&q, &k, &v, &params)?;

    // GELU MLP (module)
    let mlp_out = modules::gelu_mlp(&img_mlp_in, &self.img_fc1, &self.img_fc2, ctx, ops)?;
    // ...
}
```

### Kernel Optimization Reach

When a kernel dev adds a new optimization, how many models benefit?

| Optimization | Changed in | Models that benefit |
|-------------|-----------|-------------------|
| Faster FlashInfer FA3 | `CudaOps::varlen_attention` | **All models** (every model calls attention) |
| `fused_add_rmsnorm` kernel | `CudaOps` + `modules::residual_norm` | **All transformer models** (Qwen3, Llama, Gemma, ...) |
| `fused_adaln_zero` kernel | `CudaOps` + `modules::adaln_zero` | **All diffusion models** (Flux, HunyuanVideo, Sana, ...) |
| `fused_qknorm_rope` kernel | `CudaOps` + `modules::qk_norm_rope` | **All QK-norm models** (Qwen3, Gemma3, ...) |
| `fused_silu_mul` kernel | `CudaOps` + `modules::gated_mlp` | **All SiLU-gated MLP models** (most LLMs) |
| DeepGEMM FP8 improvement | `CudaOps::matmul` | **All models** (every model does matmul) |
| BGMV LoRA kernel | `CudaOps` + `Linear::forward` | **All LoRA-served models** |
| Better NCCL all-reduce | `CudaOps::all_reduce_sum` | **All TP-distributed models** |

**Rule of thumb:** Ops-level improvements (attention, matmul) reach ALL models. Module-level improvements (fusion) reach all models that use that module. Both are O(1) changes for O(N) model benefit.
