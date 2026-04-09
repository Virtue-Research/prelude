//! CudaOps — single `impl Ops` for CUDA devices.
//!
//! Overrides tensor primitives with CUDA kernels, plus norm, fused, attention, etc.
//! Methods not overridden inherit defaults from `trait Ops`.

use prelude_core::tensor::{bail, DType, Result, Tensor};
use prelude_core::ops::traits::*;

pub struct CudaOps;

/// Static CudaOps instance.
pub fn cuda_ops() -> &'static dyn Ops {
    use std::sync::LazyLock;
    static OPS: LazyLock<CudaOps> = LazyLock::new(|| {
        // Register GPU GEMM dispatch on first access.
        crate::ops::gemm::register_gpu_gemm();
        CudaOps
    });
    &*OPS
}

// ── Helpers ────────────────────────────────────────────────────────


fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}

// ── The single impl ───────────────────────────────────────────────
// Basic tensor ops (matmul, unary, binary, etc.) are handled by candle-core natively.
// CudaOps only overrides fused/inference-specific ops.

impl Ops for CudaOps {
    fn default_impl(&self) -> &dyn Ops { self }

    // ── Normalization (CUDA kernels) ──────────────────────────────

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if x.dtype() == DType::BF16 {
            return crate::ops::rmsnorm::fast_rmsnorm(x, weight, eps as f64);
        }
        // Non-BF16: use composed fallback
        prelude_core::ops::traits::norm::rms_norm(x, weight, eps)
    }

    // layer_norm: uses trait default (composed via tensor methods)

    // ── GEMM ──────────────────────────────────────────────────────

    fn grouped_gemm(&self, input: &Tensor, weights: &Tensor, sorted_token_ids: &Tensor, sorted_expert_ids: &Tensor, _num_tokens_per_expert: &Tensor) -> Result<Tensor> {
        let num_assignments = sorted_token_ids.elem_count();
        let num_tokens = input.dims()[0];
        let topk = if num_tokens > 0 { num_assignments / num_tokens } else { 1 };
        crate::ops::moe::moe_gemm_wmma(input, weights, &None, sorted_token_ids, sorted_expert_ids, topk, num_tokens > 1)
    }

    // ── KV cache ──────────────────────────────────────────────────

    fn reshape_and_cache(&self, key: &Tensor, value: &Tensor, key_cache: &Tensor, value_cache: &Tensor, slot_mapping: &Tensor) -> Result<()> {
        crate::ops::kv_cache::scatter_kv_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    }

    // ── Attention ─────────────────────────────────────────────────
    // Runtime dispatch: try FA4 first (has runtime kernel registry with SM detection),
    // fall back to FlashInfer, then composed CPU fallback.

    fn attn_name(&self) -> &str { "cuda" }

    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        if let Some(r) = try_fa4_varlen(q, k, v, params) { return r; }
        if let Some(r) = try_flashinfer_varlen(q, k, v, params) { return r; }
        // Ultimate fallback: composed matmul SDPA
        prelude_core::ops::traits::attention::varlen_attention(q, k, v, params)
    }

    fn paged_attention(&self, q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, params: &PagedParams) -> Result<Tensor> {
        // FA4 paged: handles both prefill (Q>1) and decode (Q=1).
        let seqused_k = cu_seqlens_to_lens(params.cu_seqlens_k)?;
        if let Some(r) = crate::attn::flash_v4::try_varlen_paged(
            q, key_cache, value_cache, params.block_tables,
            params.cu_seqlens_q, &seqused_k, params.max_seqlen_q, params.max_seqlen_k, params.scale,
        ) { return r; }
        // FlashInfer fallback (SM80 without FA4, or non-BF16).
        if let Some(r) = try_flashinfer_paged(q, key_cache, value_cache, params) { return r; }
        bail!("paged_attention: no kernel available for this configuration")
    }

    fn paged_block_size_hint(&self, head_dim: usize) -> usize {
        // FlashInfer is the paged fallback, use its alignment requirements
        if head_dim == 256 { 64 } else { 128 }
    }

    // ── Fused ops (CUDA kernels) ─────────────────────────────────

    fn fused_add_rmsnorm(&self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> Option<Result<(Tensor, Tensor)>> {
        if x.dtype() != DType::BF16 { return None; }
        Some(crate::ops::rmsnorm::fused_add_rmsnorm(x, residual, weight, eps as f64))
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
        if gate.dtype() != DType::BF16 { return None; }
        Some(crate::ops::elementwise::fused_silu_mul(gate, up))
    }

    fn silu_mul_concat(&self, gate_up: &Tensor) -> Option<Result<Tensor>> {
        if gate_up.dtype() != DType::BF16 { return None; }
        Some(crate::attn::flashinfer::silu_and_mul(gate_up))
    }

    fn fused_add(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        if a.dtype() == DType::BF16 && b.dtype() == DType::BF16 {
            Some(crate::ops::elementwise::vectorized_add(a, b))
        } else { None }
    }

    fn fused_qknorm_rope(&self, q: &Tensor, k: &Tensor, q_weight: &Tensor, k_weight: &Tensor, cos: &Tensor, sin: &Tensor, position_ids: &Tensor, eps: f32) -> Option<Result<(Tensor, Tensor)>> {
        if q.dtype() != DType::BF16 { return None; }
        // CUDA kernel supports head_dim <= 256 (8 elements per lane, vals[8])
        let head_dim = q.dims().last().copied().unwrap_or(0);
        if head_dim > 256 { return None; }
        let q_out = match crate::ops::rope::fused_qknorm_rope_varlen(q, q_weight, cos, sin, position_ids, eps as f64) {
            Ok(t) => t, Err(e) => return Some(Err(e)),
        };
        let k_out = match crate::ops::rope::fused_qknorm_rope_varlen(k, k_weight, cos, sin, position_ids, eps as f64) {
            Ok(t) => t, Err(e) => return Some(Err(e)),
        };
        Some(Ok((q_out, k_out)))
    }

    fn fused_knorm_rope_cache_write(&self, k: &Tensor, v: &Tensor, k_weight: &Tensor, cos: &Tensor, sin: &Tensor, position_ids: &Tensor, key_cache: &Tensor, value_cache: &Tensor, slot_mapping: &Tensor, eps: f32) -> Option<Result<()>> {
        if k.dtype() != DType::BF16 { return None; }
        if !crate::ops::kv_cache::fused_kv_cache_write_enabled() { return None; }
        let k_dims = k.dims();
        let kc_dims = key_cache.dims();
        let num_kv_heads = if k_dims.len() == 3 { k_dims[1] } else { return None };
        let head_dim = if k_dims.len() == 3 { k_dims[2] } else { return None };
        let block_size = if kc_dims.len() == 4 { kc_dims[1] } else { return None };
        Some(crate::ops::kv_cache::fused_knorm_rope_kv_cache_write_varlen(
            k, v, k_weight, cos, sin, position_ids, key_cache, value_cache, slot_mapping,
            num_kv_heads, head_dim, block_size, eps as f64,
        ))
    }

    fn fused_moe_routing(&self, gate_logits: &Tensor, top_k: usize) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> {
        if gate_logits.dtype() != DType::BF16 { return None; }
        Some(crate::ops::moe::fused_moe_routing(gate_logits, top_k, true))
    }

    fn fused_moe_gemm(&self, input: &Tensor, weights: &Tensor, topk_weights: &Tensor, sorted_token_ids: &Tensor, sorted_expert_ids: &Tensor, topk: usize, is_prefill: bool) -> Option<Result<Tensor>> {
        if !matches!(input.dtype(), DType::BF16 | DType::F16) { return None; }
        Some(crate::ops::moe::moe_gemm_wmma(input, weights, &Some(topk_weights.clone()), sorted_token_ids, sorted_expert_ids, topk, is_prefill))
    }

    // ── Session ───────────────────────────────────────────────────

    fn begin_forward(&self) {
        crate::attn::flashinfer::begin_forward();
    }
    fn end_forward(&self) {
        crate::attn::flashinfer::end_forward();
    }
    fn precompute_paged_plan(&self, q_shape: (usize, usize, usize), key_cache: &Tensor, cu_seqlens_q: &Tensor, block_tables: &Tensor, cu_seqlens_k: &Tensor, softmax_scale: f32) -> Result<()> {
        crate::attn::flashinfer::precompute_paged_plan(q_shape, key_cache, cu_seqlens_q, block_tables, cu_seqlens_k, softmax_scale)?;
        Ok(())
    }
    fn gpu_free_memory(&self) -> Option<usize> {
        unsafe extern "C" { fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32; }
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cudaMemGetInfo(&mut free, &mut total) } == 0 { Some(free) } else { None }
    }
}

// ── Attention dispatch helpers ─────────────────────────────────────

/// Try FA4 for varlen attention. Returns None if FA4 can't handle this request.
fn try_fa4_varlen(q: &Tensor, k: &Tensor, v: &Tensor, p: &VarlenParams) -> Option<Result<Tensor>> {
    use crate::attn::flash_v4;
    match &p.mask {
        MaskType::Causal => flash_v4::try_varlen_causal(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::Bidirectional => flash_v4::try_varlen_bidirectional(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::SlidingWindow { left, right } => flash_v4::try_varlen_windowed(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale, Some(*left), Some(*right)),
        MaskType::Custom(_) => None, // FA4 doesn't support custom masks
    }
}

/// Try FlashInfer for varlen attention. Returns None if FlashInfer can't handle this.
fn try_flashinfer_varlen(q: &Tensor, k: &Tensor, v: &Tensor, p: &VarlenParams) -> Option<Result<Tensor>> {
    use crate::attn::flashinfer;
    // FlashInfer supports BF16/FP16 on SM80+; head_dim up to 256 (FA2/FA3 limit)
    if !matches!(q.dtype(), DType::BF16 | DType::F16) { return None; }
    let head_dim = q.dims().last().copied().unwrap_or(0);
    if head_dim > 256 { return None; }
    let result = match &p.mask {
        MaskType::Causal => flashinfer::varlen_causal(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::Bidirectional => flashinfer::varlen_bidirectional(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale),
        MaskType::SlidingWindow { left, right } => flashinfer::varlen_windowed(q, k, v, p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale, Some(*left), Some(*right)),
        MaskType::Custom(_) => return None,
    };
    // If FlashInfer fails due to missing kernel variant, return None to fall through to SDPA
    match &result {
        Err(e) if e.to_string().contains("no FA3") || e.to_string().contains("no variant") => {
            tracing::debug!("FlashInfer fallback: {e}");
            None
        }
        _ => Some(result),
    }
}

/// Try FlashInfer for paged attention.
fn try_flashinfer_paged(q: &Tensor, key_cache: &Tensor, value_cache: &Tensor, p: &PagedParams) -> Option<Result<Tensor>> {
    if !matches!(q.dtype(), DType::BF16 | DType::F16) { return None; }
    Some(crate::attn::flashinfer::varlen_paged(
        q, key_cache, value_cache, p.block_tables,
        p.cu_seqlens_q, p.cu_seqlens_k, p.max_seqlen_q, p.max_seqlen_k, p.scale,
    ))
}
