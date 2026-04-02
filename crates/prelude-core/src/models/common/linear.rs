// Shared Linear & RmsNorm — model code uses these, never touches backend types directly.
//
// Linear: auto-dispatches to GpuLinear (CUDA), OnednnLinear (CPU), or registered
//         quantization backends (Q4_0, future Q4_K_M / FP8 / INT4).
// RmsNorm: AVX-512 on CPU, candle on GPU.

use crate::loading::var_builder::VarBuilder;
use crate::nn_ops::{CandleLinear, CandleRmsNorm};
use candle_core::{Module, Result, Tensor};
use std::any::Any;
use std::sync::Arc;

// ── LinearBackend trait ───────────────────────────────────────────────────

/// Trait for all Linear layer backends.
///
/// Models call `Linear::forward(x)` which delegates to the active backend.
/// Each backend handles its own weight storage and GEMM dispatch.
///
/// Implemented by: `GpuLinear`, `OnednnLinear`, `QuantizedWeight`, and
/// future backends (FP8, INT4 GEMM, ...).
pub trait LinearBackend: Module + Send + Sync + std::fmt::Debug {
    /// Backend name for logging (e.g., "gpu/cutlass", "cpu/onednn", "quant/q4_0").
    fn name(&self) -> &str;

    /// Access the underlying weight tensor.
    /// Returns `None` for quantized backends where raw weights aren't available.
    fn weight(&self) -> Option<&Tensor> {
        None
    }

    /// Whether this backend uses quantized weights.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn LinearBackend>;

    /// Downcast for backend-specific operations (e.g., brgemm_weight on OnednnLinear).
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn LinearBackend> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ── QuantFormat registry ──────────────────────────────────────────────────

/// Registry entry for a quantization format.
///
/// Each format (Q4_0, Q4_K_M, FP8, ...) implements this trait and registers
/// via `inventory::submit!`. `Linear::from_qtensor()` iterates the registry
/// to find a handler for the given GGML dtype.
pub trait QuantFormat: Send + Sync {
    fn name(&self) -> &str;
    fn can_handle(&self, dtype: candle_core::quantized::GgmlDType) -> bool;
    fn load(&self, qtensor: Arc<candle_core::quantized::QTensor>) -> Result<Box<dyn LinearBackend>>;
}

/// Wrapper for `inventory` auto-registration.
pub struct QuantFormatEntry {
    pub format: &'static dyn QuantFormat,
}

impl QuantFormatEntry {
    pub const fn new(format: &'static dyn QuantFormat) -> Self {
        Self { format }
    }
}

inventory::collect!(QuantFormatEntry);

/// Find a registered QuantFormat that can handle the given dtype.
fn find_quant_format(
    dtype: candle_core::quantized::GgmlDType,
) -> Option<&'static dyn QuantFormat> {
    inventory::iter::<QuantFormatEntry>()
        .find(|entry| entry.format.can_handle(dtype))
        .map(|entry| entry.format)
}

// ── Linear ──────────────────────────────────────────────────────────────

/// Unified linear layer. Loads weights from VarBuilder and automatically
/// selects the best backend at construction time.
///
/// Supports both standard (FP16/BF16/F32) and quantized (GGUF) weights.
/// Models call `linear.forward(x)` without knowing which backend is active.
#[derive(Debug, Clone)]
pub struct Linear {
    inner: Box<dyn LinearBackend>,
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

    /// Wrap an existing `CandleLinear`, selecting the best backend by device.
    ///
    /// CUDA: uses `CandleLinear` — `Tensor::matmul()` routes through the registered
    /// CUTLASS/DeepGEMM dispatch (set up by `CudaOps::create()`).
    /// CPU: uses `OnednnLinear` for BF16/F32 brgemm acceleration.
    pub fn from_candle(linear: CandleLinear) -> Result<Self> {
        if linear.weight().device().is_cpu() {
            Ok(Self {
                inner: Box::new(crate::ops::onednn::OnednnLinear::new(linear)?),
            })
        } else {
            Ok(Self {
                inner: Box::new(CandleLinearBackend(linear)),
            })
        }
    }

    /// Construct from a quantized QTensor (GGUF).
    ///
    /// Looks up registered `QuantFormat` backends to find one that handles
    /// the given quantization type. Returns error if no backend supports it.
    pub fn from_qtensor(qtensor: Arc<candle_core::quantized::QTensor>) -> Result<Self> {
        let dtype = qtensor.dtype();
        match find_quant_format(dtype) {
            Some(fmt) => Ok(Self {
                inner: fmt.load(qtensor)?,
            }),
            None => candle_core::bail!(
                "Linear::from_qtensor: no registered backend for {:?}",
                dtype
            ),
        }
    }

    /// Whether this linear uses quantized weights.
    pub fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }

    /// Access the underlying weight tensor.
    /// Panics on quantized variant — use `is_quantized()` to check first.
    pub fn weight(&self) -> &Tensor {
        self.inner
            .weight()
            .expect("weight() not available on quantized Linear")
    }

    /// Access brgemm packed weight (BF16 CPU acceleration), if available.
    pub fn brgemm_weight(&self) -> Option<&crate::ops::onednn::BrgemmPackedWeight> {
        self.inner
            .as_any()
            .downcast_ref::<crate::ops::onednn::OnednnLinear>()
            .and_then(|l| l.brgemm_weight())
    }

    /// Access F32 packed weight (F32 CPU acceleration), if available.
    pub fn f32_packed_weight(&self) -> Option<&crate::ops::onednn::OnednnF32PackedWeight> {
        self.inner
            .as_any()
            .downcast_ref::<crate::ops::onednn::OnednnLinear>()
            .and_then(|l| l.f32_packed_weight())
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// ── CandleLinear backend (CUDA via registered dispatch) ─────────────────

/// Simple wrapper around `CandleLinear` for CUDA devices.
/// `Tensor::matmul()` is intercepted by the registered GEMM dispatch
/// (CUTLASS/DeepGEMM), so no direct FFI needed here.
#[derive(Debug, Clone)]
struct CandleLinearBackend(CandleLinear);

impl Module for CandleLinearBackend {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.0.forward(x)
    }
}

impl LinearBackend for CandleLinearBackend {
    fn name(&self) -> &str { "gpu/candle" }
    fn weight(&self) -> Option<&Tensor> { Some(self.0.weight()) }
    fn clone_box(&self) -> Box<dyn LinearBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

// ── Q4_0 quantized backend ───────────────────────────────────────────────

/// Owned quantized weight storage (Q4_0). Raw block data extracted from QTensor
/// at construction time — forward uses our native SIMD kernels directly.
#[derive(Debug, Clone)]
struct QuantizedWeight {
    blocks: Vec<crate::ops::cpu::quant::BlockQ4_0>,
    n: usize,
    k: usize,
}

impl Module for QuantizedWeight {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use crate::ops::cpu::quant::quantized_matmul_f32;
        use candle_core::{DType, Device};

        let x = x.to_dtype(DType::F32)?;
        let x_dims = x.shape().dims();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_k = *x_dims.last().unwrap();
        if x_k != self.k {
            candle_core::bail!(
                "quantized matmul: x inner dim {x_k} != weight dim {}",
                self.k
            );
        }

        let x_cont = x.flatten_all()?;
        let x_storage = x_cont.storage_and_layout().0;
        let x_slice = match &*x_storage {
            candle_core::Storage::Cpu(cpu) => cpu.as_slice::<f32>()?,
            _ => candle_core::bail!("quantized matmul: expected CPU tensor"),
        };

        let mut out = vec![0.0f32; m * self.n];
        quantized_matmul_f32(x_slice, &self.blocks, &mut out, m, self.n, self.k);

        let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(self.n);
        Tensor::from_vec(out, out_dims.as_slice(), &Device::Cpu)
    }
}

impl LinearBackend for QuantizedWeight {
    fn name(&self) -> &str {
        "quant/q4_0"
    }
    fn is_quantized(&self) -> bool {
        true
    }
    fn clone_box(&self) -> Box<dyn LinearBackend> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Q4_0 format handler — registered via `inventory`.
struct Q4_0Format;

impl QuantFormat for Q4_0Format {
    fn name(&self) -> &str {
        "Q4_0"
    }
    fn can_handle(&self, dtype: candle_core::quantized::GgmlDType) -> bool {
        dtype == candle_core::quantized::GgmlDType::Q4_0
    }
    fn load(&self, qtensor: Arc<candle_core::quantized::QTensor>) -> Result<Box<dyn LinearBackend>> {
        let shape = qtensor.shape();
        let dims = shape.dims();
        let n = dims[0];
        let k = dims[1];
        let raw = qtensor.data()?;
        let blocks: Vec<crate::ops::cpu::quant::BlockQ4_0> =
            bytemuck::cast_slice(&raw).to_vec();
        Ok(Box::new(QuantizedWeight { blocks, n, k }))
    }
}

inventory::submit!(QuantFormatEntry::new(&Q4_0Format));

// ── Q4_K quantized backend ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QuantizedWeightQ4K {
    blocks: Vec<crate::ops::cpu::quant::BlockQ4K>,
    n: usize,
    k: usize,
}

impl Module for QuantizedWeightQ4K {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use crate::ops::cpu::quant::q4_k::quantized_matmul_q4k;
        use candle_core::{DType, Device};

        let x = x.to_dtype(DType::F32)?;
        let x_dims = x.shape().dims();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_k = *x_dims.last().unwrap();
        if x_k != self.k {
            candle_core::bail!(
                "Q4_K matmul: x inner dim {x_k} != weight dim {}",
                self.k
            );
        }

        let x_cont = x.flatten_all()?;
        let x_storage = x_cont.storage_and_layout().0;
        let x_slice = match &*x_storage {
            candle_core::Storage::Cpu(cpu) => cpu.as_slice::<f32>()?,
            _ => candle_core::bail!("Q4_K matmul: expected CPU tensor"),
        };

        let mut out = vec![0.0f32; m * self.n];
        quantized_matmul_q4k(x_slice, &self.blocks, &mut out, m, self.n, self.k);

        let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(self.n);
        Tensor::from_vec(out, out_dims.as_slice(), &Device::Cpu)
    }
}

impl LinearBackend for QuantizedWeightQ4K {
    fn name(&self) -> &str { "quant/q4_k" }
    fn is_quantized(&self) -> bool { true }
    fn clone_box(&self) -> Box<dyn LinearBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

struct Q4KFormat;

impl QuantFormat for Q4KFormat {
    fn name(&self) -> &str { "Q4_K" }
    fn can_handle(&self, dtype: candle_core::quantized::GgmlDType) -> bool {
        dtype == candle_core::quantized::GgmlDType::Q4K
    }
    fn load(&self, qtensor: Arc<candle_core::quantized::QTensor>) -> Result<Box<dyn LinearBackend>> {
        let shape = qtensor.shape();
        let dims = shape.dims();
        let n = dims[0];
        let k = dims[1];
        let raw = qtensor.data()?;
        let blocks: Vec<crate::ops::cpu::quant::BlockQ4K> =
            bytemuck::cast_slice(&raw).to_vec();
        Ok(Box::new(QuantizedWeightQ4K { blocks, n, k }))
    }
}

inventory::submit!(QuantFormatEntry::new(&Q4KFormat));

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device};

    #[test]
    fn q4_0_linear_forward_matches_candle() {
        let k = 64;
        let n = 4;
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w_tensor = Tensor::from_vec(w_data, (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&w_tensor, GgmlDType::Q4_0, &Device::Cpu).unwrap();
        let qt = std::sync::Arc::new(qt);

        let m = 2;
        let x_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.03).cos()).collect();
        let x = Tensor::from_vec(x_data, (m, k), &Device::Cpu).unwrap();

        let our_linear = Linear::from_qtensor(qt.clone()).unwrap();
        let our_out = our_linear.forward(&x).unwrap();

        let w_deq = qt.dequantize(&Device::Cpu).unwrap();
        let ref_out = x.matmul(&w_deq.t().unwrap()).unwrap();

        let our_flat = our_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ref_flat = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let n_elems = our_flat.len();
        let mean_abs_err: f32 = our_flat
            .iter()
            .zip(ref_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n_elems as f32;

        assert!(
            mean_abs_err < 0.05,
            "Q4_0 Linear forward: mean abs error {mean_abs_err} too high"
        );
    }

    #[test]
    fn q4_0_linear_output_shape() {
        let k = 32;
        let n = 8;
        let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.01).collect();
        let w_tensor = Tensor::from_vec(w_data, (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&w_tensor, GgmlDType::Q4_0, &Device::Cpu).unwrap();
        let linear = Linear::from_qtensor(std::sync::Arc::new(qt)).unwrap();

        let x = Tensor::zeros((3, k), DType::F32, &Device::Cpu).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dims(), &[3, 8]);

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
            let norm = CandleRmsNorm::new(self.weight.clone(), self.eps);
            norm.forward(x)
        }
    }
}
