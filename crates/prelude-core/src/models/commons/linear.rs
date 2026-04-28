// Shared Linear & RmsNorm — model code uses these, never touches backend types directly.
//
// Linear: front-end wrapper over `Box<dyn LinearBackend>`. Backends cover the
//         weight-format axis (DenseLinear, Q4_0Linear, Q4KLinear, GpuQuantLinear,
//         OnednnLinear). The backend is picked at construction time — model code
//         only sees `Linear` and doesn't know which format is loaded.
//
// Attention and MLP deliberately don't have an analogous `AttentionBackend` /
// `MlpBackend` trait: weight format is the only dimension that genuinely varies
// at runtime for a given model, and it only touches the matmul step. Attention
// kernel selection (FA4 vs FlashInfer vs SDPA) lives inside `Ops.varlen_attention`
// as a fallback chain instead.
//
// RmsNorm: AVX-512 on CPU, fused CUDA kernel on GPU via `Ops`.

use crate::loading::var_builder::VarBuilder;
use crate::tensor::{DType, Module, Result, Tensor, D};
use std::any::Any;
use std::sync::Arc;

// ── DenseLinear (fp16/bf16/f32 dense backend) ──────────────────────────

/// Dense floating-point linear layer: `y = x @ weight.T + bias`.
///
/// The default `LinearBackend` for non-quantized weights. On CUDA, the inner
/// `matmul()` is intercepted by the registered GEMM dispatch (DeepGEMM → CUTLASS).
/// On CPU, it falls through to candle's native matmul (or is replaced by
/// `OnednnLinear` via the `CpuLinearFactory` registry).
#[derive(Clone, Debug)]
pub struct DenseLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl DenseLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn bias(&self) -> Option<&Tensor> { self.bias.as_ref() }
}

impl Module for DenseLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?.matmul(&w)?.reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?.matmul(&w)?.reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

impl LinearBackend for DenseLinear {
    fn name(&self) -> &str { "dense" }
    fn weight(&self) -> Option<&Tensor> { Some(&self.weight) }
    fn clone_box(&self) -> Box<dyn LinearBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

// ── LinearBackend trait ───────────────────────────────────────────────────

/// Backend trait for `Linear`. Exists only because Linear has multiple real
/// runtime implementations (dense fp, Q4_0, Q4_K, GPU quant, oneDNN), one per
/// weight format. Attention and MLP don't have an analogous trait because
/// their implementation is fixed per model.
///
/// All impls must also be `Module` (that's where the actual `forward` lives).
/// The extra methods here are for identity queries and boxed cloning.
///
/// Current impls:
/// - `DenseLinear` (prelude-core, fp16/bf16/f32)
/// - `OnednnLinear` (prelude-cpu, BF16/F32 with oneDNN)
/// - `Q4_0Linear`, `Q4KLinear` (prelude-cpu, GGUF quantization)
/// - `GpuQuantLinear` (prelude-cuda, GGUF quantization via quant-gemm)
pub trait LinearBackend: Module + Send + Sync + std::fmt::Debug {
    /// Backend name for logging (e.g., "dense", "cpu/onednn", "cpu/q4_0").
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
    fn create(&self, linear: DenseLinear) -> Result<Box<dyn LinearBackend>>;
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
        Self::from_dense(DenseLinear::new(weight, bias))
    }

    /// Construct from a raw weight tensor (e.g., for tied embeddings / lm_head).
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_dense(DenseLinear::new(weight, bias))
    }

    /// Wrap an existing `DenseLinear`, selecting the best backend by device.
    ///
    /// CUDA: uses `DenseLinear` directly — its `forward` calls `Tensor::matmul()`
    /// which routes through the registered CUTLASS/DeepGEMM dispatch.
    /// CPU: uses registered CPU linear factory (OnednnLinear from prelude-cpu) if available,
    ///      otherwise falls back to `DenseLinear` + candle native matmul.
    pub fn from_dense(linear: DenseLinear) -> Result<Self> {
        if linear.weight().device().is_cpu() {
            // Use registered CPU linear factory if available (e.g., OnednnLinear)
            if let Some(entry) = inventory::iter::<CpuLinearFactoryEntry>().next() {
                return Ok(Self {
                    inner: entry.factory.create(linear)?,
                });
            }
        }
        Ok(Self {
            inner: Box::new(linear),
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

    /// Forward pass: GEMM (plain/quantized) → LoRA (fused/fallback) → TP (reduce/gather).
    ///
    /// `ctx` carries per-batch LoRA adapter routing. `ops` provides device kernels.
    /// Currently LoRA and TP are not yet implemented — `ctx` and `ops` are accepted
    /// for API stability and will be used when those features land.
    pub fn forward(&self, x: &Tensor, _ctx: &super::BatchState, _ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// Q4_0Linear, Q4KLinear, OnednnLinear backends live in prelude-cpu.
// GpuQuantLinear lives in prelude-cuda. They register themselves via inventory
// when their crate is linked.

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

    /// Ops-accelerated RMS normalization (fused CUDA kernel when available).
    pub fn forward_ops(&self, x: &Tensor, ops: &dyn crate::ops::Ops) -> Result<Tensor> {
        ops.rms_norm(x, &self.weight, self.eps as f32)
    }

    /// Fused residual-add + RMS normalization (vLLM-style).
    ///
    /// - `residual = None` (first layer): returns `(hidden, rms_norm(hidden))`
    /// - `residual = Some(r)`: returns `fused_add_rmsnorm(r, hidden, weight, eps)`
    pub fn forward_residual(
        &self,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        ops: &dyn crate::ops::Ops,
    ) -> Result<(Tensor, Tensor)> {
        match residual {
            Some(res) => {
                let (new_res, normed) = ops.add_rmsnorm(res, hidden, &self.weight, self.eps as f32)?;
                Ok((new_res, normed))
            }
            None => {
                let normed = ops.rms_norm(hidden, &self.weight, self.eps as f32)?;
                Ok((hidden.clone(), normed))
            }
        }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xs_f32 = x.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_t = Tensor::new(&[self.eps as f32], x.device())?.broadcast_as(variance.shape())?;
        let inv_rms = (variance + eps_t)?.sqrt()?.recip()?;
        let normed = xs_f32.broadcast_mul(&inv_rms)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        normed.broadcast_mul(&weight)?.to_dtype(x.dtype())
    }
}
