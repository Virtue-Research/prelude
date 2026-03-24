//! Attention backend dispatch.
//!
//! This module is the **only** place where attention backend feature flags appear.
//! Models call the functions here; they never import backend crates directly.
//!
//! Backend priority: FA4 (CuTeDSL) → FlashInfer → FA3 (Hopper) → FA2 (Ampere+) → CPU.

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
/// - Writes K/V to paged cache
/// - FA3: uses fused paged attention (`flash_attn_varlen_paged`)
/// - FA2: decode (Q=1) uses `paged_attention`, prefill uses non-paged FA2
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
    // ── Paged path ──
    if let Some(kv) = paged_kv {
        // FA4 paged: handles both prefill (Q>1) and decode (Q=1).
        #[cfg(feature = "flash-attn-v4")]
        if q.device().is_cuda() {
            paged::reshape_and_cache_flash(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            let seqused_k = cu_seqlens_to_lens(kv.cu_seqlens_k)?;
            return flash_v4::varlen_paged(
                q, kv.key_cache, kv.value_cache, kv.block_tables,
                cu_seqlens_q, &seqused_k, max_seqlen_q, kv.max_seqlen_k,
                softmax_scale,
            );
        }

        // FlashInfer paged: prefill + decode, SM80+ (FA2) or SM90+ (FA3)
        #[cfg(feature = "flashinfer")]
        if q.device().is_cuda() {
            paged::reshape_and_cache_flash(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            return flashinfer::varlen_paged(
                q, kv.key_cache, kv.value_cache, kv.block_tables,
                cu_seqlens_q, kv.cu_seqlens_k, max_seqlen_q, kv.max_seqlen_k,
                softmax_scale,
            );
        }

        // FA3: flash layout cache + fused paged attention (prefill + decode)
        #[cfg(feature = "flash-attn-v3")]
        if q.device().is_cuda() {
            paged::reshape_and_cache_flash(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            return flash_v3::varlen_paged(
                q, kv.key_cache, kv.value_cache, kv.block_tables,
                cu_seqlens_q, kv.cu_seqlens_k, max_seqlen_q, kv.max_seqlen_k,
                softmax_scale,
            );
        }

        // FA2: v1 layout cache + paged_attention for decode, FA2 varlen for prefill
        #[cfg(feature = "flash-attn")]
        if q.device().is_cuda() {
            paged::reshape_and_cache_v1(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;
            if max_seqlen_q == 1 {
                // Decode: Q=1, use vLLM paged_attention kernel.
                // Convert cu_seqlens_k (cumulative) → context_lens (per-sequence).
                let context_lens = cu_seqlens_to_lens(kv.cu_seqlens_k)?;
                return paged::decode_attention(
                    q, kv.key_cache, kv.value_cache, kv.block_tables,
                    &context_lens, kv.max_seqlen_k, softmax_scale,
                );
            }
            // Prefill with paged write: attend local K/V only (no prefix cache read)
            return flash_v2::varlen_causal(
                q, k, v, cu_seqlens_q, cu_seqlens_q,
                max_seqlen_q, max_seqlen_q, softmax_scale,
            );
        }
    }

    // ── Non-paged path ──
    let _ = paged_kv;

    // FA4: non-paged varlen (prefill). Preferred over FA3 when available.
    #[cfg(feature = "flash-attn-v4")]
    if q.device().is_cuda() {
        return flash_v4::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(feature = "flashinfer")]
    if q.device().is_cuda() {
        return flashinfer::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(feature = "flash-attn-v3")]
    if q.device().is_cuda() {
        return flash_v3::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() {
        return flash_v2::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    let _ = (max_seqlen_q, max_seqlen_k);
    cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
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
    #[cfg(feature = "flash-attn-v4")]
    if q.device().is_cuda() {
        return flash_v4::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        );
    }

    #[cfg(feature = "flashinfer")]
    if q.device().is_cuda() {
        return flashinfer::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        );
    }

    #[cfg(feature = "flash-attn-v3")]
    if q.device().is_cuda() {
        return flash_v3::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        );
    }

    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() {
        return flash_v2::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        );
    }

    // CPU fallback: ignore window, use standard causal attention.
    let _ = (max_seqlen_q, max_seqlen_k, window_left, window_right);
    cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
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
    #[cfg(feature = "flash-attn-v4")]
    if q.device().is_cuda() {
        return flash_v4::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(feature = "flashinfer")]
    if q.device().is_cuda() {
        return flashinfer::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(feature = "flash-attn-v3")]
    if q.device().is_cuda() {
        return flash_v3::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() {
        return flash_v2::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        );
    }

    let _ = (max_seqlen_q, max_seqlen_k);
    cpu::varlen_bidirectional(q, k, v, cu_seqlens_q, softmax_scale)
}

// ── Paged-only attention ─────────────────────────────────────────────

/// Write K/V to paged cache (backend-agnostic dispatch).
///
/// Selects flash layout (FA4/FA3) or v1 layout (FA2/others) based on compiled features.
#[allow(unused_variables)]
pub(crate) fn reshape_and_cache(
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    // FA4, FlashInfer, FA3 all use flash layout: [num_blocks, block_size, num_heads, head_dim]
    #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
    return paged::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping);

    #[cfg(all(feature = "cuda", not(feature = "flash-attn-v3"), not(feature = "flash-attn-v4"), not(feature = "flashinfer")))]
    return paged::reshape_and_cache_v1(key, value, key_cache, value_cache, slot_mapping);

    #[cfg(not(feature = "cuda"))]
    unreachable!("reshape_and_cache requires cuda feature")
}

/// Paged varlen attention (read from paged KV cache, no KV write).
///
/// FA4: uses `fa4_varlen_paged_fwd`.
/// FA3: uses `flash_attn_varlen_paged`.
/// FA2: uses `paged_attention` (decode-only, Q=1).
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn varlen_attention_paged(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    // FA4 paged: handles both prefill (Q>1) and decode (Q=1).
    #[cfg(feature = "flash-attn-v4")]
    if q.device().is_cuda() {
        let seqused_k = cu_seqlens_to_lens(cu_seqlens_k)?;
        return flash_v4::varlen_paged(
            q, key_cache, value_cache, block_tables,
            cu_seqlens_q, &seqused_k, max_seqlen_q, max_seqlen_k,
            softmax_scale,
        );
    }

    // FlashInfer paged (prefill + decode)
    #[cfg(feature = "flashinfer")]
    return flashinfer::varlen_paged(
        q, key_cache, value_cache, block_tables,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale,
    );

    #[cfg(feature = "flash-attn-v3")]
    return flash_v3::varlen_paged(
        q, key_cache, value_cache, block_tables,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale,
    );

    #[cfg(all(feature = "cuda", not(feature = "flash-attn-v3"), not(feature = "flash-attn-v4"), not(feature = "flashinfer")))]
    {
        let context_lens = cu_seqlens_to_lens(cu_seqlens_k)?;
        return paged::decode_attention(
            q, key_cache, value_cache, block_tables,
            &context_lens, max_seqlen_k, softmax_scale,
        );
    }

    #[cfg(not(feature = "cuda"))]
    unreachable!("varlen_attention_paged requires cuda feature")
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
