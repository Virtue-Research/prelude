// Shared Linear & RmsNorm — model code uses these, never touches candle_nn types directly.
//
// Linear: auto-dispatches to oneDNN BRGeMM (BF16 CPU), oneDNN F32, or candle matmul.
// RmsNorm: AVX-512 on CPU, candle on GPU.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_core::quantized::{GgmlDType, QMatMul};
use crate::loading::var_builder::VarBuilder;
use crate::nn_ops::{CandleLinear, CandleRmsNorm};

// ── Linear ──────────────────────────────────────────────────────────────

/// Unified linear layer. Loads weights from VarBuilder and automatically
/// selects the best GEMM backend at construction time.
///
/// Supports both standard (FP16/BF16/F32) and quantized (GGUF Q4/Q8) weights.
/// Models call `linear.forward(x)` without knowing which backend is active.
#[derive(Debug, Clone)]
pub struct Linear {
    inner: LinearInner,
}

#[derive(Debug, Clone)]
enum LinearInner {
    Candle(CandleLinear),
    #[cfg(feature = "onednn")]
    Onednn(crate::ops::onednn::OnednnLinear),
    /// GGUF quantized weights (Q4_0, Q4_K_M, Q8_0, etc.)
    Quantized(QMatMul),
}

impl Linear {
    /// Load from VarBuilder (reads "weight" and optionally "bias" tensors).
    pub fn load(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        Self::from_candle(CandleLinear::new(weight, bias))
    }

    /// Construct from a raw weight tensor (e.g., for tied embeddings / lm_head).
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_candle(CandleLinear::new(weight, bias))
    }

    /// Wrap an existing `CandleLinear`, packing weights for acceleration.
    pub fn from_candle(linear: CandleLinear) -> Result<Self> {
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

    /// Construct from a quantized QMatMul (GGUF).
    pub fn from_qmatmul(qmm: QMatMul) -> Self {
        Self { inner: LinearInner::Quantized(qmm) }
    }

    /// Construct from a quantized QTensor (GGUF).
    pub fn from_qtensor(qtensor: std::sync::Arc<candle_core::quantized::QTensor>) -> Result<Self> {
        Ok(Self::from_qmatmul(QMatMul::from_arc(qtensor)?))
    }

    /// Whether this linear uses quantized weights.
    pub fn is_quantized(&self) -> bool {
        matches!(&self.inner, LinearInner::Quantized(_))
    }

    /// Access the underlying weight tensor.
    /// Panics on quantized variant — use `is_quantized()` to check first.
    pub fn weight(&self) -> &Tensor {
        match &self.inner {
            LinearInner::Candle(l) => l.weight(),
            #[cfg(feature = "onednn")]
            LinearInner::Onednn(l) => l.weight(),
            LinearInner::Quantized(_) => panic!("weight() not available on quantized Linear"),
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
            LinearInner::Quantized(q) => forward_quantized(q, x),
        }
    }
}

/// Quantized forward: use our native SIMD kernel for Q4_0 on CPU,
/// fall back to candle's QMatMul for other formats or GPU.
fn forward_quantized(q: &QMatMul, x: &Tensor) -> Result<Tensor> {
    if let QMatMul::QTensor(qt) = q {
        if qt.device().is_cpu() && qt.dtype() == GgmlDType::Q4_0 {
            return forward_q4_0_cpu(qt, x);
        }
    }
    q.forward(x)
}

/// Native Q4_0 matmul on CPU: quantize activations to Q8_0,
/// compute dot products with SIMD, return F32 Tensor.
fn forward_q4_0_cpu(
    qt: &candle_core::quantized::QTensor,
    x: &Tensor,
) -> Result<Tensor> {
    use crate::ops::cpu::quant::{BlockQ4_0, quantized_matmul_f32};

    let x = x.to_dtype(DType::F32)?;
    let x_shape = x.shape();
    let w_shape = qt.shape();

    // W is [N, K], x is [..., M, K], output is [..., M, N]
    let k = *w_shape.dims().last().unwrap();
    let n = w_shape.dims()[0];

    let x_dims = x_shape.dims();
    let m: usize = x_dims[..x_dims.len() - 1].iter().product();
    let x_k = *x_dims.last().unwrap();
    if x_k != k {
        candle_core::bail!("Q4_0 matmul: x inner dim {x_k} != weight dim {k}");
    }

    // Get raw weight bytes → &[BlockQ4_0]
    let w_data = qt.data()?;
    let w_blocks: &[BlockQ4_0] = bytemuck::cast_slice(&w_data);

    // Get x as contiguous F32 slice
    let x_cont = x.flatten_all()?;
    let x_storage = x_cont.storage_and_layout().0;
    let x_slice = match &*x_storage {
        candle_core::Storage::Cpu(cpu) => cpu.as_slice::<f32>()?,
        _ => candle_core::bail!("Q4_0 matmul: expected CPU tensor"),
    };

    // Compute
    let mut out = vec![0.0f32; m * n];
    quantized_matmul_f32(x_slice, w_blocks, &mut out, m, n, k);

    // Wrap as Tensor with correct shape
    let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
    out_dims.push(n);
    Tensor::from_vec(out, out_dims.as_slice(), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{QTensor, GgmlDType};

    #[test]
    fn q4_0_linear_forward_matches_candle() {
        // Create F32 weight [N=4, K=64], quantize to Q4_0
        let k = 64;
        let n = 4;
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w_tensor = Tensor::from_vec(w_data, (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&w_tensor, GgmlDType::Q4_0, &Device::Cpu).unwrap();
        let qt = std::sync::Arc::new(qt);

        // Create input x [M=2, K=64]
        let m = 2;
        let x_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.03).cos()).collect();
        let x = Tensor::from_vec(x_data, (m, k), &Device::Cpu).unwrap();

        // Our path: Linear with native Q4_0 kernel
        let our_linear = Linear::from_qtensor(qt.clone()).unwrap();
        let our_out = our_linear.forward(&x).unwrap();

        // Reference path: candle's QMatMul dequant→F32→matmul
        let w_deq = qt.dequantize(&Device::Cpu).unwrap();
        let ref_out = x.matmul(&w_deq.t().unwrap()).unwrap();

        // Compare: both start from same Q4_0 weights, but our path also
        // quantizes activations to Q8_0. Normalized error should be small.
        let our_flat = our_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ref_flat = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let n_elems = our_flat.len();
        let mean_abs_err: f32 = our_flat.iter().zip(ref_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / n_elems as f32;

        // Activation quantization adds some noise, but should be small
        assert!(
            mean_abs_err < 0.05,
            "Q4_0 Linear forward: mean abs error {mean_abs_err} too high"
        );
    }

    #[test]
    fn q4_0_linear_output_shape() {
        // Verify output shape is correct for batched input
        let k = 32;
        let n = 8;
        let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.01).collect();
        let w_tensor = Tensor::from_vec(w_data, (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&w_tensor, GgmlDType::Q4_0, &Device::Cpu).unwrap();
        let linear = Linear::from_qtensor(std::sync::Arc::new(qt)).unwrap();

        // [3, 32] → [3, 8]
        let x = Tensor::zeros((3, k), DType::F32, &Device::Cpu).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dims(), &[3, 8]);

        // [2, 5, 32] → [2, 5, 8]
        let x = Tensor::zeros((2, 5, k), DType::F32, &Device::Cpu).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 5, 8]);
    }
}

// ── RmsNorm ─────────────────────────────────────────────────────────────

/// Unified RMS normalization layer. AVX-512 on CPU, CUDA kernel on GPU,
/// candle fallback elsewhere.
///
/// Models should always use this instead of `CandleRmsNorm`.
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
            let norm = CandleRmsNorm::new(self.weight.clone(), self.eps);
            norm.forward(x)
        }
    }
}

