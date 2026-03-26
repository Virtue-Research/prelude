//! Attention backend dispatch.
//!
//! This module is the **only** place where attention backend feature flags appear.
//! Models call the functions here; they never import backend crates directly.
//!
//! Backend priority: FA4 (CuTeDSL) → FlashInfer → FA3 (Hopper) → FA2 (Ampere+) → CPU.
//!
//! There are two ways to use this module:
//!
//! 1. **Free functions** (backward-compatible): `varlen_attention()`, `reshape_and_cache()`, etc.
//!    These contain `#[cfg]` dispatch internally.
//!
//! 2. **Trait interface**: call `select_backend()` to get a `&'static dyn AttentionBackend`,
//!    then call methods on it. The backend is resolved once and cached for the process lifetime.

mod backend;
pub(crate) use backend::{AttentionBackend, select_backend};

#[cfg(feature = "flash-attn-v4")]
mod flash_v4;
#[cfg(feature = "flashinfer")]
mod flashinfer;
#[cfg(feature = "flashinfer")]
pub(crate) use flashinfer::{
    allocate_fi_graph_meta,
    begin_forward as fi_begin_forward, end_forward as fi_end_forward,
    precompute_paged_plan as fi_precompute_paged_plan,
    precompute_paged_plan_graphed as fi_precompute_paged_plan_graphed,
};
#[cfg(feature = "flash-attn-v3")]
mod flash_v3;
#[cfg(feature = "flash-attn")]
mod flash_v2;
#[cfg(feature = "cuda")]
pub(crate) mod paged;
pub(crate) mod cpu;

use candle_core::{Result, Tensor};
use super::PagedKvContext;

// ── Unified varlen attention ─────────────────────────────────────────

/// Unified varlen attention with optional paged KV cache.
///
/// When `paged_kv` is `Some`:
/// - Writes K/V to paged cache via `backend.reshape_and_cache()`
/// - Dispatches paged attention via `backend.varlen_attention_paged()`
/// - FA2 prefill fallback: attends local K/V only (FA2 lacks paged prefill)
///
/// When `paged_kv` is `None`: standard varlen attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    paged_kv: Option<&PagedKvContext>,
) -> Result<Tensor> {
    let backend = select_backend(q.device().is_cuda());

    if let Some(kv) = paged_kv {
        // Write K/V to paged cache (flash layout or v1 layout, backend handles it)
        backend.reshape_and_cache(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;

        // FA2 can't do paged prefill — fall back to non-paged varlen (local K/V only)
        if max_seqlen_q > 1 && !backend.supports_paged_prefill() {
            return backend.varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_q,
                max_seqlen_q, max_seqlen_q, softmax_scale,
            );
        }

        return backend.varlen_attention_paged(
            q, kv.key_cache, kv.value_cache, kv.block_tables,
            cu_seqlens_q, kv.cu_seqlens_k, max_seqlen_q, kv.max_seqlen_k,
            softmax_scale,
        );
    }

    backend.varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale,
    )
}

// ── Windowed varlen attention ────────────────────────────────────────

/// Varlen attention with sliding window (Gemma3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_windowed(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, window_right: Option<usize>,
) -> Result<Tensor> {
    select_backend(q.device().is_cuda()).varlen_attention_windowed(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale,
        window_left, window_right,
    )
}

// ── Bidirectional varlen attention ───────────────────────────────────

/// Non-causal (bidirectional) varlen attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    select_backend(q.device().is_cuda()).varlen_attention_bidirectional(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale,
    )
}

// ── Paged-only attention ─────────────────────────────────────────────

/// Write K/V to paged cache (backend-agnostic dispatch).
pub(crate) fn reshape_and_cache(
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    select_backend(key.device().is_cuda()).reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping,
    )
}

/// Paged varlen attention (read from paged KV cache, no KV write).
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_paged(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    select_backend(q.device().is_cuda()).varlen_attention_paged(
        q, key_cache, value_cache, block_tables,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale,
    )
}

// ── FA4 tile size ────────────────────────────────────────────────────

/// FA4 tile_n for paged attention (causal, non-windowed).
/// Mirrors `_tile_size_fwd_sm90()` from upstream `flash_attn/cute/interface.py`.
/// Used by both the attention dispatch and `CacheManager` to ensure
/// `page_size == tile_n` for the TMA fast path.
#[cfg(feature = "flash-attn-v4")]
pub(crate) fn fa4_tile_n(head_dim: usize, head_dim_v: usize) -> usize {
    match head_dim {
        d if d <= 64 => 128,
        d if d <= 96 => 128,
        d if d <= 128 => 128,
        d if d <= 192 => if head_dim_v <= 128 { 128 } else { 112 },
        _ => 80, // hdim 256: non-local (paged) → 80
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Convert cumulative sequence lengths `[0, 5, 12, 18]` to per-sequence
/// context lengths `[5, 7, 6]` (as a Tensor). Needed for `paged_attention`
/// which expects per-seq lengths, not cumulative.
fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    // GPU-side: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    // No GPU→CPU sync — stays entirely on device.
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}
