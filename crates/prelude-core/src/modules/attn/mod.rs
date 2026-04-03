//! Attention dispatch — routes through the Ops trait system.
//!
//! Models call the functions here with an `&crate::ops::Ops` handle;
//! the Ops bundle delegates to the appropriate `AttentionOps` implementation
//! (CudaOps or CpuOps), selected at engine startup.

mod backend;

use crate::tensor::{Result, Tensor};
use crate::ops::{MaskType, VarlenParams, PagedParams};
use super::PagedKvContext;

// ── Unified varlen attention ─────────────────────────────────────────

/// Unified varlen attention with optional paged KV cache.
///
/// When `paged_kv` is `Some`:
/// - Writes K/V to paged cache via `ops.kv_cache.reshape_and_cache()`
/// - Dispatches paged attention via `ops.attn.paged_attention()`
/// - FA2 prefill fallback: attends local K/V only (FA2 lacks paged prefill)
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

// ── Windowed varlen attention ────────────────────────────────────────

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

// ── Bidirectional varlen attention ───────────────────────────────────

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

// ── Paged-only attention ─────────────────────────────────────────────

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

// ── FA4 tile size ────────────────────────────────────────────────────

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

// ── Helpers ──────────────────────────────────────────────────────────

/// Convert cumulative sequence lengths to per-sequence lengths.
pub(crate) fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}
