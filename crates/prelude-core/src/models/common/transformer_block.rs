// Generic pre-norm decoder layer block that encapsulates the standard pattern:
//
//   pre_attn_norm → attention → residual → post_attn_norm → MLP → residual
//
// This is OPTIONAL: models with custom layer patterns (e.g. Gemma3 with its
// 4-norm architecture) should continue using their own DecoderLayer.
//
// The attention computation is injected via a closure, so each model can
// plug in its own attention implementation (Qwen3Attention, MoE routing, etc.)
// without the block needing to know about model-specific details.

use candle_core::{Result, Tensor};

use super::linear::RmsNorm;
use super::mlp::GatedMlp;
use super::ops::{fast_add, fast_rms_norm, fused_add_rmsnorm};
use super::LayerAttnContext;

/// A generic pre-norm transformer decoder block.
///
/// Owns the two RMSNorm layers and the GatedMlp. The attention computation
/// is provided by the caller via a closure at forward time, keeping this block
/// model-agnostic.
///
/// # Usage
///
/// ```ignore
/// let block = TransformerBlock::new(ln1, ln1_weight, ln2, ln2_weight, mlp, eps, layer_idx);
///
/// // In DecoderLayer::forward:
/// block.forward(x, ctx, |normed, ctx| self.self_attn.forward(normed, ctx))
/// ```
#[derive(Debug, Clone)]
pub(crate) struct TransformerBlock {
    /// Pre-attention RMSNorm (input_layernorm).
    pub(crate) ln1: RmsNorm,
    pub(crate) ln1_weight: Tensor,
    /// Post-attention RMSNorm (post_attention_layernorm).
    pub(crate) ln2: RmsNorm,
    pub(crate) ln2_weight: Tensor,
    /// SiLU-gated MLP (gate_proj, up_proj, down_proj).
    pub(crate) mlp: GatedMlp,
    /// RMS norm epsilon (shared by both norms; models always use the same eps).
    pub(crate) rms_norm_eps: f64,
    /// Layer index (used for profiling output).
    pub(crate) layer_idx: usize,
}

impl TransformerBlock {
    pub(crate) fn new(
        ln1: RmsNorm,
        ln1_weight: Tensor,
        ln2: RmsNorm,
        ln2_weight: Tensor,
        mlp: GatedMlp,
        rms_norm_eps: f64,
        layer_idx: usize,
    ) -> Self {
        Self {
            ln1,
            ln1_weight,
            ln2,
            ln2_weight,
            mlp,
            rms_norm_eps,
            layer_idx,
        }
    }

    // ── Core forward ─────────────────────────────────────────────────────

    /// Standard pre-norm decoder forward with fused-add-rmsnorm optimization.
    ///
    /// `attn_fn` receives the pre-attention-normed hidden state and the
    /// layer attention context, and must return the attention output tensor.
    ///
    /// Flow:
    ///   1. h = rms_norm(x, ln1)
    ///   2. h = attn_fn(h, ctx)
    ///   3. (x_res, h2) = fused_add_rmsnorm(x, h, ln2)   // residual + post-attn norm
    ///   4. mlp_out = mlp(h2)
    ///   5. return x_res + mlp_out
    pub(crate) fn forward<F>(
        &self,
        x: &Tensor,
        ctx: &LayerAttnContext,
        attn_fn: F,
    ) -> Result<Tensor>
    where
        F: FnOnce(&Tensor, &LayerAttnContext) -> Result<Tensor>,
    {
        let profile = crate::config::global_runtime()
            .map(|r| r.profile)
            .unwrap_or(false);

        if !profile {
            let h = fast_rms_norm(x, &self.ln1, &self.ln1_weight, self.rms_norm_eps)?;
            let h = attn_fn(&h, ctx)?;
            return self.residual_mlp(x, &h);
        }

        // Profiled path: time each component separately.
        let t = std::time::Instant::now();
        let h = fast_rms_norm(x, &self.ln1, &self.ln1_weight, self.rms_norm_eps)?;
        let norm1_ms = t.elapsed().as_secs_f32() * 1000.0;

        let t = std::time::Instant::now();
        let h = attn_fn(&h, ctx)?;
        let attn_ms = t.elapsed().as_secs_f32() * 1000.0;

        // Inline residual_mlp to split norm2 and mlp timing.
        let t = std::time::Instant::now();
        let (x_res, h2) = fused_add_rmsnorm(
            x,
            &h,
            &self.ln2,
            &self.ln2_weight,
            self.rms_norm_eps,
        )?;
        let norm2_ms = t.elapsed().as_secs_f32() * 1000.0;

        let t = std::time::Instant::now();
        let result = self.residual_mlp_after_norm(&x_res, &h2)?;
        let mlp_ms = t.elapsed().as_secs_f32() * 1000.0;

        let total = norm1_ms + attn_ms + norm2_ms + mlp_ms;
        tracing::info!(
            layer = self.layer_idx,
            norm1 = format!("{norm1_ms:.2}"),
            attn = format!("{attn_ms:.2}"),
            norm2 = format!("{norm2_ms:.2}"),
            mlp = format!("{mlp_ms:.2}"),
            total = format!("{total:.2}"),
            "layer_profile"
        );

        Ok(result)
    }

    // ── Residual + MLP helpers ───────────────────────────────────────────

    /// Fused residual-add + post-attn-norm + MLP + residual-add.
    ///
    /// This is the second half of the decoder block: after attention produces `h`,
    /// compute `x + mlp(post_attn_norm(x + h))` with the fused kernel path.
    #[inline]
    pub(crate) fn residual_mlp(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let (x_res, h2) =
            fused_add_rmsnorm(x, h, &self.ln2, &self.ln2_weight, self.rms_norm_eps)?;
        self.residual_mlp_after_norm(&x_res, &h2)
    }

    /// Given already-normed post-attention residual and hidden state, run MLP
    /// and add back to the residual. Dispatches to raw CPU paths when available.
    #[inline]
    fn residual_mlp_after_norm(&self, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        // Raw BF16 MLP path: eliminate Tensor allocations in MLP forward.
        #[cfg(feature = "onednn")]
        if h2.device().is_cpu() && h2.dtype() == candle_core::DType::BF16 {
            if self.mlp.gate_up_brgemm_weight().is_some() {
                return self.residual_mlp_raw_bf16(x_res, h2);
            }
        }

        // Raw F32 MLP path.
        #[cfg(feature = "onednn")]
        if h2.device().is_cpu() && h2.dtype() == candle_core::DType::F32 {
            if self.mlp.gate_up_f32_packed_weight().is_some() {
                return self.residual_mlp_raw_f32(x_res, h2);
            }
        }

        fast_add(x_res, &self.mlp.forward(h2)?)
    }

    /// Raw BF16 MLP: forward_raw + in-place residual add. Zero Tensor allocations.
    #[cfg(feature = "onednn")]
    fn residual_mlp_raw_bf16(&self, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        use super::raw_cpu;
        use crate::ops::cpu::buf_tensor::CpuTensor;

        let h2_buf = CpuTensor::from_candle(h2)?;
        let needed = h2_buf.len();

        raw_cpu::with_scratch(|scratch| {
            raw_cpu::ensure_len(&mut scratch.mlp_out, needed);
            unsafe {
                self.mlp.forward_raw(&h2_buf, scratch.mlp_out.as_mut_ptr());
            }

            // In-place add: x_res += mlp_out
            crate::ops::cpu::inplace_add_bf16(x_res, &scratch.mlp_out[..needed]).unwrap();
        });

        Ok(x_res.clone())
    }

    /// Raw F32 MLP: forward_raw_f32 + in-place residual add. Zero Tensor allocations.
    #[cfg(feature = "onednn")]
    fn residual_mlp_raw_f32(&self, x_res: &Tensor, h2: &Tensor) -> Result<Tensor> {
        use super::raw_cpu;

        let h2_slice = crate::ops::cpu::tensor_as_f32_slice(h2)?;
        let total = h2.dim(0)?;
        let hidden_size = h2.dim(1)?;
        let needed = total * hidden_size;

        raw_cpu::with_scratch_f32(|scratch| {
            raw_cpu::ensure_len_f32(&mut scratch.mlp_out, needed);
            unsafe {
                self.mlp.forward_raw_f32(
                    h2_slice.as_ptr(),
                    total,
                    hidden_size,
                    scratch.mlp_out.as_mut_ptr(),
                );
            }

            crate::ops::cpu::inplace_add_f32(x_res, &scratch.mlp_out[..needed]).unwrap();
        });

        Ok(x_res.clone())
    }
}
