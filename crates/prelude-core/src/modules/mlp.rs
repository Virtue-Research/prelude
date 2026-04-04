// Shared SiLU gated MLP (gate_proj, up_proj, down_proj).
//
// Used by Qwen3, Qwen3-MoE (dense layers), and other architectures.
// Raw CPU forward paths (brgemm, raw_f32) are in prelude-cpu.

use crate::tensor::{DType, Result, Tensor};
use crate::loading::var_builder::VarBuilder;
use crate::models::config::Qwen3Config;

use super::BatchState;
use super::linear::Linear;

#[derive(Debug, Clone)]
pub(crate) struct GatedMlp {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
    /// Merged [gate; up] weights for fused gate_up GEMM.
    pub(crate) gate_up_proj: Option<Linear>,
}

impl GatedMlp {
    pub(crate) fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = Linear::load(vb.pp("gate_proj"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let up_proj = Linear::load(vb.pp("up_proj"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let down_proj = Linear::load(vb.pp("down_proj"), cfg.intermediate_size, cfg.hidden_size, false)?;

        // Merge gate + up weights for fused GEMM on CPU BF16
        let gate_up_proj = {
            let gw = gate_proj.weight();
            if gw.device().is_cpu() && gw.dtype() == DType::BF16 {
                let merged_w = Tensor::cat(&[gw, up_proj.weight()], 0)?;
                match Linear::from_weight(merged_w, None) {
                    Ok(l) => Some(l),
                    Err(e) => {
                        tracing::warn!("Failed to create merged gate_up_proj: {e}");
                        None
                    }
                }
            } else {
                None
            }
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj,
        })
    }

    pub(crate) fn forward(&self, ctx: &BatchState, ops: &crate::ops::Ops, x: &Tensor) -> Result<Tensor> {
        // Fused gate_up GEMM path (CPU BF16) — uses merged weight if available
        if let Some(ref gup) = self.gate_up_proj {
            let gate_up = gup.forward(x, ctx, ops)?;
            // SiLU×Mul via Ops fused kernel or fallback
            let dims = gate_up.dims();
            let dim = dims[dims.len() - 1] / 2;
            let gate = gate_up.narrow(dims.len() - 1, 0, dim)?;
            let up = gate_up.narrow(dims.len() - 1, dim, dim)?;
            return self.down_proj.forward(&super::norm::fast_silu_mul(ops, &gate, &up)?, ctx, ops);
        }

        let gate = self.gate_proj.forward(x, ctx, ops)?;
        let up = self.up_proj.forward(x, ctx, ops)?;
        self.down_proj.forward(&super::norm::fast_silu_mul(ops, &gate, &up)?, ctx, ops)
    }
}
