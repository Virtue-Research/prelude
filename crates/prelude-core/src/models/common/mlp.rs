// Shared SiLU gated MLP (gate_proj, up_proj, down_proj).
//
// Used by Qwen3, Qwen3-MoE (dense layers), and other architectures.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config as Qwen3Config;

use super::linear::Linear;
use super::ops::debug_disable_fused_silu_mul;

#[derive(Debug, Clone)]
pub(crate) struct GatedMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// Merged [gate; up] weights for fused gate_up GEMM (single GEMM instead of two).
    /// Available on CPU BF16 path for both sgl-cpu and onednn backends.
    gate_up_proj: Option<Linear>,
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

    /// Check if gate_up brgemm weight is available (for raw path dispatch).
    #[cfg(feature = "onednn")]
    pub(crate) fn gate_up_brgemm_weight(&self) -> Option<&crate::ops::onednn::BrgemmPackedWeight> {
        self.gate_up_proj.as_ref()?.brgemm_weight()
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Fused gate_up GEMM path (CPU BF16)
        if let Some(ref gup) = self.gate_up_proj {
            {
                let profile = crate::config::global_runtime()
                    .map(|r| r.profile)
                    .unwrap_or(false);

                // Fused GEMM+SiLU path: gate_up GEMM + SiLU×Mul in one pass,
                // keeping F32 accumulators hot in L2 cache.
                // Only for small M: the C++ fused kernel uses scalar expf which
                // is ~1000x slower than AVX-512 cpu_ops silu_mul. At M>128 the
                // scalar expf cost overwhelms the L2 cache savings from fusion.
                // The unfused path still benefits from 2D M×N tiling at large M.
                #[cfg(feature = "onednn")]
                if let Some(brg) = gup.brgemm_weight() {
                    let n = brg.n;
                    let dims = x.dims();
                    let m: usize = dims.iter().product::<usize>() / dims[dims.len() - 1];
                    if n % 2 == 0 && m <= 128 {
                        let dim = n / 2;
                        let k = dims[dims.len() - 1];

                        let t0 = std::time::Instant::now();
                        let activated = crate::ops::onednn::brgemm_fused_silu_mul(
                            x, brg, m, k, dim,
                        )?;
                        let fused_ms = t0.elapsed().as_secs_f32() * 1000.0;

                        let t0 = std::time::Instant::now();
                        let result = activated.apply(&self.down_proj);
                        let down_ms = t0.elapsed().as_secs_f32() * 1000.0;

                        if profile {
                            tracing::info!(
                                gate_up = format!("{fused_ms:.3}"),
                                silu_mul = format!("0.000"),
                                down = format!("{down_ms:.3}"),
                                "mlp_profile"
                            );
                        }
                        return result;
                    }
                }

                let t0 = std::time::Instant::now();
                let gate_up = gup.forward(x)?;
                let gate_up_ms = t0.elapsed().as_secs_f32() * 1000.0;

                let dims = gate_up.dims();
                let is_3d = dims.len() == 3;
                let flat = if is_3d {
                    let (b, s, d) = gate_up.dims3()?;
                    gate_up.reshape((b * s, d))?
                } else {
                    gate_up
                };

                let t0 = std::time::Instant::now();
                let activated = crate::ops::cpu::cpu_silu_and_mul(&flat)?;
                let silu_ms = t0.elapsed().as_secs_f32() * 1000.0;

                let activated = if is_3d {
                    let (b, s, _) = x.dims3()?;
                    activated.reshape((b, s, activated.dim(1)?))?
                } else {
                    activated
                };

                let t0 = std::time::Instant::now();
                let result = activated.apply(&self.down_proj);
                let down_ms = t0.elapsed().as_secs_f32() * 1000.0;

                if profile {
                    tracing::info!(
                        gate_up = format!("{gate_up_ms:.3}"),
                        silu_mul = format!("{silu_ms:.3}"),
                        down = format!("{down_ms:.3}"),
                        "mlp_profile"
                    );
                }

                return result;
            }
        }

        // gate_up_proj is always Some on CPU BF16 — this path is for CUDA only
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        super::ops::fast_silu_mul(&gate, &up)?.apply(&self.down_proj)
    }

    /// Raw MLP forward: operates on CpuTensor with thread-local scratch buffers.
    /// Eliminates all intermediate `Tensor::from_vec` allocations (~0.37ms/layer saved).
    ///
    /// - `input`: `[total, hidden_size]` BF16 via CpuTensor
    /// - `output`: pre-allocated `[total * hidden_size]` u16 buffer
    ///
    /// # Safety
    /// - `output` must point to `[total * hidden_size]` pre-allocated u16 elements.
    #[cfg(feature = "onednn")]
    pub(crate) unsafe fn forward_raw(
        &self,
        input: &crate::ops::cpu::buf_tensor::CpuTensor,
        output: *mut u16,
    ) {
        let gup = match self.gate_up_proj {
            Some(ref gup) => gup,
            None => return,
        };
        let gate_up_brg = match gup.brgemm_weight() {
            Some(b) => b,
            None => return,
        };
        let down_brg = match self.down_proj.brgemm_weight() {
            Some(b) => b,
            None => return,
        };

        super::raw_cpu::with_scratch(|scratch| {
            unsafe {
                super::raw_cpu::raw_mlp_forward(
                    scratch, input, gate_up_brg, down_brg, output,
                );
            }
        });
    }

    #[cfg(feature = "onednn")]
    pub(crate) fn gate_up_f32_packed_weight(&self) -> Option<&crate::ops::onednn::OnednnF32PackedWeight> {
        self.gate_up_proj.as_ref()?.f32_packed_weight()
    }

    /// Raw F32 MLP forward: gate_up GEMM → SiLU×Mul → down GEMM on raw f32 buffers.
    #[cfg(feature = "onednn")]
    pub(crate) unsafe fn forward_raw_f32(
        &self,
        input: *const f32,
        total: usize,
        hidden_size: usize,
        output: *mut f32,
    ) {
        let gup = match self.gate_up_proj {
            Some(ref gup) => gup,
            None => return,
        };
        let gate_up_pw = match gup.f32_packed_weight() {
            Some(b) => b,
            None => return,
        };
        let down_pw = match self.down_proj.f32_packed_weight() {
            Some(b) => b,
            None => return,
        };

        super::raw_cpu::with_scratch_f32(|scratch| {
            unsafe {
                super::raw_cpu::raw_mlp_forward_f32(
                    scratch, input, total, hidden_size,
                    gate_up_pw, down_pw, output,
                );
            }
        });
    }
}
