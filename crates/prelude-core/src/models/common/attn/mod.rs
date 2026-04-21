//! Attention dispatch.
//!
//! Models call the `pub(crate)` free functions here; they never import backend
//! crates directly. Feature-gated `#[cfg]` is concentrated in this one module.
//!
//! Backend priority (compile-time): FA4 → FlashInfer → FA3 → FA2 → CPU.
//! Only one GPU backend is active per build; dispatch is a single branch.

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

// Only FA2 lacks paged prefill; all higher-priority GPU backends support Q>1.
const GPU_SUPPORTS_PAGED_PREFILL: bool = cfg!(any(
    feature = "flash-attn-v4",
    feature = "flashinfer",
    feature = "flash-attn-v3",
));

// ── GPU dispatch (one active branch per build, chosen by priority) ────

#[allow(clippy::too_many_arguments)]
fn gpu_varlen_causal(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn-v4")]
    return flash_v4::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flashinfer", not(feature = "flash-attn-v4")))]
    return flashinfer::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flash-attn-v3",
        not(any(feature = "flash-attn-v4", feature = "flashinfer"))))]
    return flash_v3::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flash-attn",
        not(any(feature = "flash-attn-v4", feature = "flashinfer", feature = "flash-attn-v3"))))]
    return flash_v2::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer",
        feature = "flash-attn-v3", feature = "flash-attn")))]
    {
        let _ = (max_seqlen_q, max_seqlen_k);
        cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    }
}

#[allow(clippy::too_many_arguments)]
fn gpu_varlen_windowed(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, window_right: Option<usize>,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn-v4")]
    return flash_v4::varlen_windowed(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, window_left, window_right);
    #[cfg(all(feature = "flashinfer", not(feature = "flash-attn-v4")))]
    return flashinfer::varlen_windowed(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, window_left, window_right);
    #[cfg(all(feature = "flash-attn-v3",
        not(any(feature = "flash-attn-v4", feature = "flashinfer"))))]
    return flash_v3::varlen_windowed(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, window_left, window_right);
    #[cfg(all(feature = "flash-attn",
        not(any(feature = "flash-attn-v4", feature = "flashinfer", feature = "flash-attn-v3"))))]
    return flash_v2::varlen_windowed(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, window_left, window_right);
    #[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer",
        feature = "flash-attn-v3", feature = "flash-attn")))]
    {
        let _ = (max_seqlen_q, max_seqlen_k, window_left, window_right);
        cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    }
}

#[allow(clippy::too_many_arguments)]
fn gpu_varlen_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn-v4")]
    return flash_v4::varlen_bidirectional(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flashinfer", not(feature = "flash-attn-v4")))]
    return flashinfer::varlen_bidirectional(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flash-attn-v3",
        not(any(feature = "flash-attn-v4", feature = "flashinfer"))))]
    return flash_v3::varlen_bidirectional(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flash-attn",
        not(any(feature = "flash-attn-v4", feature = "flashinfer", feature = "flash-attn-v3"))))]
    return flash_v2::varlen_bidirectional(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer",
        feature = "flash-attn-v3", feature = "flash-attn")))]
    {
        let _ = (cu_seqlens_k, max_seqlen_q, max_seqlen_k);
        cpu::varlen_bidirectional(q, k, v, cu_seqlens_q, softmax_scale)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_varlen_paged(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn-v4")]
    {
        let seqused_k = cu_seqlens_to_lens(cu_seqlens_k)?;
        return flash_v4::varlen_paged(q, key_cache, value_cache, block_tables,
            cu_seqlens_q, &seqused_k, max_seqlen_q, max_seqlen_k, softmax_scale);
    }
    #[cfg(all(feature = "flashinfer", not(feature = "flash-attn-v4")))]
    return flashinfer::varlen_paged(q, key_cache, value_cache, block_tables,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flash-attn-v3",
        not(any(feature = "flash-attn-v4", feature = "flashinfer"))))]
    return flash_v3::varlen_paged(q, key_cache, value_cache, block_tables,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale);
    #[cfg(all(feature = "flash-attn",
        not(any(feature = "flash-attn-v4", feature = "flashinfer", feature = "flash-attn-v3"))))]
    {
        // FA2 paged: decode-only (Q=1) via vLLM paged_attention kernel.
        let _ = (cu_seqlens_q, max_seqlen_q);
        let context_lens = cu_seqlens_to_lens(cu_seqlens_k)?;
        return paged::decode_attention(q, key_cache, value_cache, block_tables,
            &context_lens, max_seqlen_k, softmax_scale);
    }
    #[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer",
        feature = "flash-attn-v3", feature = "flash-attn")))]
    {
        let _ = (q, key_cache, value_cache, block_tables, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale);
        candle_core::bail!("no GPU attention backend compiled");
    }
}

#[cfg(feature = "cuda")]
fn gpu_reshape_and_cache(
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    // FA2 uses v1 cache layout; all higher-priority backends use flash layout.
    #[cfg(any(feature = "flash-attn-v4", feature = "flashinfer", feature = "flash-attn-v3"))]
    return paged::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping);
    #[cfg(all(feature = "flash-attn",
        not(any(feature = "flash-attn-v4", feature = "flashinfer", feature = "flash-attn-v3"))))]
    return paged::reshape_and_cache_v1(key, value, key_cache, value_cache, slot_mapping);
    #[cfg(not(any(feature = "flash-attn-v4", feature = "flashinfer",
        feature = "flash-attn-v3", feature = "flash-attn")))]
    {
        let _ = (key, value, key_cache, value_cache, slot_mapping);
        candle_core::bail!("no GPU attention backend compiled");
    }
}

// ── Public entry points ──────────────────────────────────────────────

/// Unified varlen attention with optional paged KV cache.
///
/// When `paged_kv` is `Some`:
/// - Writes K/V to paged cache (`reshape_and_cache`)
/// - Dispatches paged attention
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
    if let Some(kv) = paged_kv {
        if !q.device().is_cuda() {
            candle_core::bail!("paged attention is not supported on CPU");
        }
        #[cfg(feature = "cuda")]
        {
            gpu_reshape_and_cache(k, v, kv.key_cache, kv.value_cache, kv.slot_mapping)?;

            // FA2 can't do paged prefill — fall back to non-paged varlen (local K/V only).
            if max_seqlen_q > 1 && !GPU_SUPPORTS_PAGED_PREFILL {
                return gpu_varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_q,
                    max_seqlen_q, max_seqlen_q, softmax_scale);
            }
            return gpu_varlen_paged(q, kv.key_cache, kv.value_cache, kv.block_tables,
                cu_seqlens_q, kv.cu_seqlens_k, max_seqlen_q, kv.max_seqlen_k, softmax_scale);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_k);
            candle_core::bail!("paged attention requires the `cuda` feature");
        }
    }

    if q.device().is_cuda() {
        gpu_varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale)
    } else {
        let _ = (max_seqlen_q, max_seqlen_k);
        cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    }
}

/// Varlen attention with sliding window (Gemma3). CPU ignores the window.
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_windowed(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, window_right: Option<usize>,
) -> Result<Tensor> {
    if q.device().is_cuda() {
        gpu_varlen_windowed(q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale, window_left, window_right)
    } else {
        let _ = (max_seqlen_q, max_seqlen_k, window_left, window_right);
        cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    }
}

/// Non-causal (bidirectional) varlen attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn varlen_attention_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    if q.device().is_cuda() {
        gpu_varlen_bidirectional(q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale)
    } else {
        let _ = (cu_seqlens_k, max_seqlen_q, max_seqlen_k);
        cpu::varlen_bidirectional(q, k, v, cu_seqlens_q, softmax_scale)
    }
}

/// Write K/V to paged cache.
pub(crate) fn reshape_and_cache(
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if !key.device().is_cuda() {
        candle_core::bail!("reshape_and_cache is not supported on CPU");
    }
    #[cfg(feature = "cuda")]
    {
        gpu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (key, value, key_cache, value_cache, slot_mapping);
        candle_core::bail!("reshape_and_cache requires the `cuda` feature");
    }
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
    if !q.device().is_cuda() {
        candle_core::bail!("varlen_attention_paged is not supported on CPU");
    }
    #[cfg(feature = "cuda")]
    {
        gpu_varlen_paged(q, key_cache, value_cache, block_tables,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, key_cache, value_cache, block_tables, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale);
        candle_core::bail!("varlen_attention_paged requires the `cuda` feature");
    }
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
/// which expects per-seq lengths, not cumulative. GPU-side, no CPU sync.
#[cfg(feature = "cuda")]
fn cu_seqlens_to_lens(cu_seqlens: &Tensor) -> Result<Tensor> {
    let n = cu_seqlens.dim(0)? - 1;
    let hi = cu_seqlens.narrow(0, 1, n)?;
    let lo = cu_seqlens.narrow(0, 0, n)?;
    hi.sub(&lo)
}
