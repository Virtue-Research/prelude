// Generic pre-norm decoder layer block that encapsulates the standard pattern:
//
//   pre_attn_norm → attention → fused_residual_add_norm → MLP+residual
//
// The attention AND MLP are both injected via closures. TransformerBlock
// owns only the two norms — the universal part of every decoder layer.
//
// This works for: Qwen3 (dense MLP), Qwen3-MoE (sparse MLP), Qwen3.5/Next
// (DeltaNet attention), and any future model with the standard 2-norm pattern.
// Gemma3 (4-norm) should continue using its own DecoderLayer.

use candle_core::{Result, Tensor};

use super::linear::RmsNorm;
use super::ops::{fast_rms_norm, fused_add_rmsnorm};
use crate::profiling::{nvtx_push, nvtx_pop};

/// A generic pre-norm transformer decoder block.
///
/// Owns only the two RMSNorm layers. Both attention and MLP are provided
/// by the caller as closures, keeping this block model-agnostic.
///
/// # Usage
///
/// ```ignore
/// // In DecoderLayer::forward:
/// self.block.forward(ctx.ops, x,
///     |h| self.self_attn.forward(h, ctx),
///     |x_res, h2| fast_add(ctx.ops, x_res, &self.mlp.forward(&h2)?),
/// )
/// ```
#[derive(Debug, Clone)]
pub(crate) struct TransformerBlock {
    /// Pre-attention RMSNorm (input_layernorm).
    pub(crate) ln1: RmsNorm,
    pub(crate) ln1_weight: Tensor,
    /// Post-attention RMSNorm (post_attention_layernorm).
    pub(crate) ln2: RmsNorm,
    pub(crate) ln2_weight: Tensor,
    /// RMS norm epsilon (shared by both norms; models always use the same eps).
    pub(crate) rms_norm_eps: f64,
    /// Layer index.
    pub(crate) layer_idx: usize,
}

impl TransformerBlock {
    pub(crate) fn new(
        ln1: RmsNorm,
        ln1_weight: Tensor,
        ln2: RmsNorm,
        ln2_weight: Tensor,
        rms_norm_eps: f64,
        layer_idx: usize,
    ) -> Self {
        Self {
            ln1,
            ln1_weight,
            ln2,
            ln2_weight,
            rms_norm_eps,
            layer_idx,
        }
    }

    /// Standard pre-norm decoder forward with fused-add-rmsnorm optimization.
    ///
    /// Flow:
    ///   1. h = rms_norm(x, ln1)
    ///   2. h = attn_fn(h)
    ///   3. (x_res, h2) = fused_add_rmsnorm(x, h, ln2)
    ///   4. result = residual_mlp_fn(x_res, h2)
    ///
    /// `residual_mlp_fn` receives `(x_res, h2)` and must return the final
    /// output (typically `fast_add(x_res, &mlp.forward(&h2)?)`). This lets
    /// models with raw CPU paths do in-place residual add.
    pub(crate) fn forward<A, M>(
        &self,
        ops: &crate::ops::Ops,
        x: &Tensor,
        attn_fn: A,
        residual_mlp_fn: M,
    ) -> Result<Tensor>
    where
        A: FnOnce(&Tensor) -> Result<Tensor>,
        M: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let h = fast_rms_norm(ops, x, &self.ln1, &self.ln1_weight, self.rms_norm_eps)?;
        nvtx_push!("attention");
        let h = attn_fn(&h)?;
        nvtx_pop!();
        let (x_res, h2) = fused_add_rmsnorm(
            ops, x, &h, &self.ln2, &self.ln2_weight, self.rms_norm_eps,
        )?;
        nvtx_push!("mlp");
        let result = residual_mlp_fn(&x_res, &h2);
        nvtx_pop!();
        result
    }
}
