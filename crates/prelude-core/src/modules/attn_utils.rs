//! Shared attention utilities for model architectures.
//!
//! Contains: QKV projection, rotary embeddings (RoPE), QK-norm + RoPE dispatch,
//! and attention dispatch wrappers (varlen, paged, windowed, bidirectional).

use crate::tensor::{DType, Device, Module, Result, Tensor};
use crate::nn_ops::Qwen3Config;
use crate::ops::{MaskType, VarlenParams, PagedParams};

use super::linear::{Linear, RmsNorm};
use super::norm::{debug_disable_fused_qknorm_rope, fast_rms_norm};
use super::PagedKvContext;

// ── Rotary Position Embedding (RoPE) ───────────────────────────────

/// Precomputed sin/cos tables for rotary position embedding.
///
/// Supports standard [B, H, L, D], THD [B, L, H, D], and varlen [total, H, D] layouts.
#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    pub(crate) sin: Tensor,
    pub(crate) cos: Tensor,
    /// Packed [cos || sin] cache for cpu_ops / sgl-kernel RoPE: [max_seq_len, head_dim]
    pub(crate) cos_sin_cache: Option<Tensor>,
}

impl RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.broadcast_mul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        let cos_sin_cache = if dev.is_cpu() && (dtype == DType::BF16 || dtype == DType::F32) {
            Some(Tensor::cat(&[&cos, &sin], 1)?)
        } else {
            None
        };

        Ok(Self {
            sin,
            cos,
            cos_sin_cache,
        })
    }

    /// Apply RoPE to packed Q/K in [total_tokens, H, D] format with explicit position_ids.
    /// Used by varlen attention where sequences have different lengths.
    pub(crate) fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Fast path: cpu_ops RoPE (in-place, no Tensor allocs)
        // candle rope_thd
        // q: (total_tokens, num_heads, head_dim), position_ids: (total_tokens,)
        let cos = self.cos.index_select(position_ids, 0)?;
        let sin = self.sin.index_select(position_ids, 0)?;
        // Wrap in batch dim=1 for rope_thd: (1, total_tokens, H, D)
        let (total, h_q, d) = q.dims3()?;
        let h_k = k.dim(1)?;
        let q4 = q.reshape((1, total, h_q, d))?;
        let k4 = k.reshape((1, total, h_k, d))?;
        let q_embed = crate::nn_ops::rotary_emb::rope_thd(&q4, &cos, &sin)?;
        let k_embed = crate::nn_ops::rotary_emb::rope_thd(&k4, &cos, &sin)?;
        Ok((
            q_embed.reshape((total, h_q, d))?,
            k_embed.reshape((total, h_k, d))?,
        ))
    }
}

// ── Fused QKV Projection ───────────────────────────────────────────

/// Project Q, K, V and reshape to `[total_tokens, H, D]` (varlen layout).
///
/// Uses fused QKV GEMM when available (1 GEMM instead of 3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_qkv_projection(
    x: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    qkv_proj: Option<&Linear>,
    total: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    if let Some(qkv) = qkv_proj {
        let qkv_out = qkv.forward(x)?;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let q = qkv_out
            .narrow(1, 0, q_size)?
            .reshape((total, num_heads, head_dim))?;
        let k = qkv_out
            .narrow(1, q_size, kv_size)?
            .reshape((total, num_kv_heads, head_dim))?;
        let v = qkv_out
            .narrow(1, q_size + kv_size, kv_size)?
            .reshape((total, num_kv_heads, head_dim))?;
        return Ok((q, k, v));
    }
    let q = q_proj.forward(x)?;
    let k = k_proj.forward(x)?;
    let v = v_proj.forward(x)?;
    Ok((
        q.reshape((total, num_heads, head_dim))?,
        k.reshape((total, num_kv_heads, head_dim))?,
        v.reshape((total, num_kv_heads, head_dim))?,
    ))
}

// ── QK-Norm + RoPE dispatch ────────────────────────────────────────

/// QK-Norm + RoPE for varlen layout `[total, H, D]`.
///
/// Ops fused: fused qknorm_rope kernel.
/// Fallback: fast_rms_norm + rotary_emb.apply_varlen.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qknorm_rope_varlen(
    ops: &crate::ops::Ops,
    q: &Tensor,
    k: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    q_norm: &RmsNorm,
    k_norm: &RmsNorm,
    rotary_emb: &RotaryEmbedding,
    position_ids: &Tensor,
    total: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
) -> Result<(Tensor, Tensor)> {
    // ── Fused QK-norm + RoPE kernel via Ops trait ──
    if let Some(result) = ops.fused.fused_qknorm_rope(
        q, k, q_norm_weight, k_norm_weight,
        &rotary_emb.cos, &rotary_emb.sin,
        position_ids, rms_norm_eps as f32,
    ) {
        if !debug_disable_fused_qknorm_rope() {
            return result;
        }
    }

    // ── Generic fallback (CpuOps handles CPU-specific kernels via Ops trait) ──
    let q = fast_rms_norm(ops, &q.flatten(0, 1)?, q_norm, q_norm_weight, rms_norm_eps)?
        .reshape((total, num_heads, head_dim))?;
    let k = fast_rms_norm(ops, &k.flatten(0, 1)?, k_norm, k_norm_weight, rms_norm_eps)?
        .reshape((total, num_kv_heads, head_dim))?;
    rotary_emb.apply_varlen(&q, &k, position_ids)
}

// ── Varlen Attention Dispatch ──────────────────────────────────────

/// Unified varlen attention with optional paged KV cache.
///
/// When `paged_kv` is `Some`:
/// - Writes K/V to paged cache via `ops.kv_cache.reshape_and_cache()`
/// - Dispatches paged attention via `ops.attn.paged_attention()`
///
/// When `paged_kv` is `None`: standard varlen attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention(
    ops: &crate::ops::Ops,
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    paged_kv: Option<&PagedKvContext>,
) -> Result<Tensor> {
    if let Some(kv) = paged_kv {
        // Write K/V to paged cache
        ops.kv_cache.reshape_and_cache(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;

        // FA2 can't do paged prefill — fall back to non-paged varlen (local K/V only)
        if max_seqlen_q > 1 && !ops.attn.supports_paged_prefill() {
            let params = VarlenParams {
                cu_seqlens_q, cu_seqlens_k: cu_seqlens_q,
                max_seqlen_q, max_seqlen_k: max_seqlen_q,
                scale: softmax_scale, mask: MaskType::Causal, softcap: None,
            };
            return ops.attn.varlen_attention(q, k, v, &params);
        }

        let params = PagedParams {
            block_tables: kv.block_tables,
            cu_seqlens_q, cu_seqlens_k: kv.cu_seqlens_k,
            max_seqlen_q, max_seqlen_k: kv.max_seqlen_k,
            scale: softmax_scale, mask: MaskType::Causal, softcap: None,
        };
        return ops.attn.paged_attention(q, kv.key_cache, kv.value_cache, &params);
    }

    let params = VarlenParams {
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale: softmax_scale, mask: MaskType::Causal, softcap: None,
    };
    ops.attn.varlen_attention(q, k, v, &params)
}

/// Varlen attention with sliding window (Gemma3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_windowed(
    ops: &crate::ops::Ops,
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, window_right: Option<usize>,
) -> Result<Tensor> {
    let mask = match (window_left, window_right) {
        (Some(left), right) => MaskType::SlidingWindow {
            left,
            right: right.unwrap_or(0),
        },
        _ => MaskType::Causal,
    };
    let params = VarlenParams {
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale: softmax_scale, mask, softcap: None,
    };
    ops.attn.varlen_attention(q, k, v, &params)
}

/// Non-causal (bidirectional) varlen attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_bidirectional(
    ops: &crate::ops::Ops,
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let params = VarlenParams {
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale: softmax_scale, mask: MaskType::Bidirectional, softcap: None,
    };
    ops.attn.varlen_attention(q, k, v, &params)
}

/// Write K/V to paged cache (backend-agnostic dispatch).
pub(crate) fn reshape_and_cache(
    ops: &crate::ops::Ops,
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    ops.kv_cache.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
}

/// Paged varlen attention (read from paged KV cache, no KV write).
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_paged(
    ops: &crate::ops::Ops,
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let params = PagedParams {
        block_tables,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale: softmax_scale, mask: MaskType::Causal, softcap: None,
    };
    ops.attn.paged_attention(q, key_cache, value_cache, &params)
}

// ── Helpers ────────────────────────────────────────────────────────

/// FA4 tile_n for paged attention (causal, non-windowed).
pub(crate) fn fa4_tile_n(head_dim: usize, head_dim_v: usize) -> usize {
    match head_dim {
        d if d <= 64 => 128,
        d if d <= 96 => 128,
        d if d <= 128 => 128,
        d if d <= 192 => if head_dim_v <= 128 { 128 } else { 112 },
        _ => 80,
    }
}

/// Convert cumulative sequence lengths to per-sequence lengths.
pub(crate) fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}
