//! Flash Attention v2 backend (Ampere+).
//!
//! Thin wrappers around `candle_flash_attn` (v2). FA2 handles GQA natively
//! (infers from Q/K head counts). No `flash_attn_varlen_paged` — paged
//! decode uses `paged::decode_attention()` (vLLM kernel) instead.

use candle_core::{Result, Tensor};

#[allow(clippy::too_many_arguments)]
pub fn varlen_causal(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, false,
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
    candle_flash_attn::flash_attn_varlen_windowed(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale,
        window_left, window_right,
    )
}
