//! Paged KV cache operations (backend-agnostic).
//!
//! Provides cache write and decode-only attention from paged KV.
//! These are independent of the prefill attention backend (FA2/FA3/FlashInfer).

use candle_core::{Result, Tensor};

/// Write K/V to paged cache using flash-friendly layout.
/// Layout: `[num_blocks, block_size, num_kv_heads, head_dim]`.
/// Used by FA4/FA3 backends (prefill + decode both read flash layout).
///
/// Uses our own vectorized PTX kernel (scatter_kv_cache_flash) instead of
/// candle-paged-attn, eliminating that external dependency for FA4/FA3 paths.
pub fn reshape_and_cache_flash(
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    crate::ops::gpu::scatter_kv_cache_flash(key, value, key_cache, value_cache, slot_mapping)
}

/// Write K/V to paged cache using v1 layout.
/// Layout: `[num_blocks, num_kv_heads, head_dim/x, block_size, x]`.
/// Used by FA2 backend (decode reads v1 layout via paged_attention).
#[cfg(feature = "paged-attn")]
pub fn reshape_and_cache_v1(
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    candle_paged_attn::reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
}

/// Decode attention from paged KV cache (Q=1 per sequence).
///
/// Uses vLLM-style paged_attention kernel. Works with v1 cache layout.
/// This is the decode path for non-FA3 backends (FA2, FlashInfer).
#[cfg(feature = "paged-attn")]
pub fn decode_attention(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    max_context_len: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    candle_paged_attn::paged_attention(
        q, key_cache, value_cache, block_tables,
        context_lens, max_context_len, softmax_scale,
    )
}
