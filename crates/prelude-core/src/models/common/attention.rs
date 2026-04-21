//! Shared attention helpers for model architectures.
//!
//! These are composable helper functions, NOT a monolithic attention trait.
//! Each architecture calls the helpers it needs in its own forward().

use candle_core::{Module, Result, Tensor};

use super::linear::Linear;

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
