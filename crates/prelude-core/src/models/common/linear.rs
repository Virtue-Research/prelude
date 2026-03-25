// Shared Linear & RmsNorm — model code uses these, never touches candle_nn types directly.
//
// Linear: auto-dispatches to oneDNN BRGeMM (BF16 CPU), oneDNN F32, or candle matmul.
// RmsNorm: AVX-512 on CPU, candle on GPU.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

// ── Linear ──────────────────────────────────────────────────────────────

/// Unified linear layer. Loads weights from VarBuilder and automatically
/// selects the best GEMM backend at construction time.
///
/// Models should always use this instead of `candle_nn::Linear`.
#[derive(Debug, Clone)]
pub struct Linear {
    inner: LinearInner,
}

#[derive(Debug, Clone)]
enum LinearInner {
    Candle(candle_nn::Linear),
    #[cfg(feature = "onednn")]
    Onednn(crate::ops::onednn::OnednnLinear),
}

impl Linear {
    /// Load from VarBuilder (reads "weight" and optionally "bias" tensors).
    pub fn load(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        let raw = if bias {
            candle_nn::linear(in_dim, out_dim, vb)?
        } else {
            candle_nn::linear_no_bias(in_dim, out_dim, vb)?
        };
        Self::from_candle(raw)
    }

    /// Construct from a raw weight tensor (e.g., for tied embeddings / lm_head).
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_candle(candle_nn::Linear::new(weight, bias))
    }

    /// Wrap an existing `candle_nn::Linear`, packing weights for acceleration.
    pub fn from_candle(linear: candle_nn::Linear) -> Result<Self> {
        #[cfg(feature = "onednn")]
        {
            let w = linear.weight();
            if w.device().is_cpu() {
                return Ok(Self {
                    inner: LinearInner::Onednn(crate::ops::onednn::OnednnLinear::new(linear)?),
                });
            }
        }
        Ok(Self {
            inner: LinearInner::Candle(linear),
        })
    }

    /// Access the underlying weight tensor.
    pub fn weight(&self) -> &Tensor {
        match &self.inner {
            LinearInner::Candle(l) => l.weight(),
            #[cfg(feature = "onednn")]
            LinearInner::Onednn(l) => l.weight(),
        }
    }

    /// Access brgemm packed weight (BF16 CPU acceleration), if available.
    #[cfg(feature = "onednn")]
    pub fn brgemm_weight(&self) -> Option<&crate::ops::onednn::BrgemmPackedWeight> {
        match &self.inner {
            LinearInner::Onednn(l) => l.brgemm_weight(),
            _ => None,
        }
    }

    /// Access F32 packed weight (F32 CPU acceleration), if available.
    #[cfg(feature = "onednn")]
    pub fn f32_packed_weight(&self) -> Option<&crate::ops::onednn::OnednnF32PackedWeight> {
        match &self.inner {
            LinearInner::Onednn(l) => l.f32_packed_weight(),
            _ => None,
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.inner {
            LinearInner::Candle(l) => l.forward(x),
            #[cfg(feature = "onednn")]
            LinearInner::Onednn(l) => l.forward(x),
        }
    }
}

// ── RmsNorm ─────────────────────────────────────────────────────────────

/// Unified RMS normalization layer. AVX-512 on CPU, CUDA kernel on GPU,
/// candle fallback elsewhere.
///
/// Models should always use this instead of `candle_nn::RmsNorm`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Load from VarBuilder (reads "weight" tensor).
    pub fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self::from_weight(weight, eps))
    }

    /// Construct from an existing weight tensor.
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Access the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Access epsilon.
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() {
            crate::ops::cpu::cpu_rmsnorm(x, &self.weight, self.eps)
        } else {
            // GPU: use candle's built-in RmsNorm which dispatches to CUDA kernel
            let norm = candle_nn::RmsNorm::new(self.weight.clone(), self.eps);
            norm.forward(x)
        }
    }
}

