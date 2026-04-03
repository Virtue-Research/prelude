// Shared Linear & RmsNorm — model code uses these, never touches backend types directly.
//
// Linear: auto-dispatches to GpuLinear (CUDA), OnednnLinear (CPU), or registered
//         quantization backends (Q4_0, future Q4_K_M / FP8 / INT4).
// RmsNorm: AVX-512 on CPU, candle on GPU.

use crate::loading::var_builder::VarBuilder;
use crate::nn_ops::{CandleLinear, CandleRmsNorm};
use crate::tensor::{Module, Result, Tensor};
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
    fn can_handle(&self, dtype: crate::tensor::quantized::GgmlDType) -> bool;
    fn load(&self, qtensor: Arc<crate::tensor::quantized::QTensor>) -> Result<Box<dyn LinearBackend>>;
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
    dtype: crate::tensor::quantized::GgmlDType,
) -> Option<&'static dyn QuantFormat> {
    inventory::iter::<QuantFormatEntry>()
        .find(|entry| entry.format.can_handle(dtype))
        .map(|entry| entry.format)
}

// ── CPU Linear factory registry ─────────────────────────────────────────
// Device crates (prelude-cpu) register optimized CPU linear backends via inventory.

/// Factory for creating optimized CPU linear backends.
pub trait CpuLinearFactory: Send + Sync {
    fn create(&self, linear: CandleLinear) -> Result<Box<dyn LinearBackend>>;
}

pub struct CpuLinearFactoryEntry {
    pub factory: &'static dyn CpuLinearFactory,
}

impl CpuLinearFactoryEntry {
    pub const fn new(factory: &'static dyn CpuLinearFactory) -> Self {
        Self { factory }
    }
}

inventory::collect!(CpuLinearFactoryEntry);

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
    /// CPU: uses registered CPU linear factory (OnednnLinear from prelude-cpu) if available,
    ///      otherwise falls back to CandleLinear.
    pub fn from_candle(linear: CandleLinear) -> Result<Self> {
        if linear.weight().device().is_cpu() {
            // Use registered CPU linear factory if available (e.g., OnednnLinear)
            if let Some(entry) = inventory::iter::<CpuLinearFactoryEntry>().next() {
                return Ok(Self {
                    inner: entry.factory.create(linear)?,
                });
            }
        }
        Ok(Self {
            inner: Box::new(CandleLinearBackend(linear)),
        })
    }

    /// Construct from a quantized QTensor (GGUF).
    ///
    /// Looks up registered `QuantFormat` backends to find one that handles
    /// the given quantization type. Returns error if no backend supports it.
    pub fn from_qtensor(qtensor: Arc<crate::tensor::quantized::QTensor>) -> Result<Self> {
        let dtype = qtensor.dtype();
        match find_quant_format(dtype) {
            Some(fmt) => Ok(Self {
                inner: fmt.load(qtensor)?,
            }),
            None => crate::tensor::bail!(
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

    /// Downcast to a specific backend type (for device-specific optimized paths).
    pub fn backend_as<T: 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
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

// Q4_0, Q4_K, OnednnLinear backends moved to prelude-cpu crate.
// They register via inventory when prelude-cpu is linked.

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
        let norm = CandleRmsNorm::new(self.weight.clone(), self.eps);
        norm.forward(x)
    }
}
