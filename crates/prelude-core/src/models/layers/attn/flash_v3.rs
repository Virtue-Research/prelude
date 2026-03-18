//! Flash Attention v3 backend (Hopper SM90+).
//!
//! Thin wrappers around `candle_flash_attn_v3` that adapt its API to the
//! unified attention interface used by `attn/mod.rs`.

use candle_core::{Result, Tensor};

/// Auto-detect GQA packing from Q/K head counts (Hopper optimization).
#[inline]
fn gqa_packing(q: &Tensor, k: &Tensor) -> bool {
    let nq = q.dim(1).unwrap_or(1);
    let nk = k.dim(1).unwrap_or(1);
    nk > 0 && nq / nk >= 2
}

/// GQA packing for paged path where K comes from cache tensor.
/// key_cache shape: [num_blocks, block_size, num_kv_heads, head_dim].
#[inline]
fn gqa_packing_paged(q: &Tensor, key_cache: &Tensor) -> bool {
    let nq = q.dim(1).unwrap_or(1);
    let nk = key_cache.dim(2).unwrap_or(1);
    nk > 0 && nq / nk >= 2
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_causal(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let gqa = gqa_packing(q, k);
    candle_flash_attn_v3::flash_attn_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, true, gqa,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let gqa = gqa_packing(q, k);
    candle_flash_attn_v3::flash_attn_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, false, gqa,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_windowed(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, window_right: Option<usize>,
) -> Result<Tensor> {
    let gqa = gqa_packing(q, k);
    candle_flash_attn_v3::flash_attn_varlen_windowed(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale,
        window_left, window_right, gqa,
    )
}

/// Paged varlen attention: read KV from paged cache (no KV write).
#[allow(clippy::too_many_arguments)]
pub fn varlen_paged(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let gqa = gqa_packing_paged(q, key_cache);
    candle_flash_attn_v3::flash_attn_varlen_paged(
        q, key_cache, value_cache, block_tables,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale, true, gqa,
    )
}
