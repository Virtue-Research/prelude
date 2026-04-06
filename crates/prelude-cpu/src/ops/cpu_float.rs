//! `CpuFloat` trait — generic dtype abstraction for CPU kernels.
//!
//! All CPU kernels compute in F32 internally. This trait provides the
//! load (to_f32) and store (from_f32) conversions for each supported dtype.
//! Adding a new dtype (e.g., FP16) requires only one `impl CpuFloat`.

use prelude_core::tensor::{DType, Device, Result, Tensor};

/// A floating-point type that can be used in CPU kernels.
///
/// Implementations must be layout-compatible with their `DType`
/// so that zero-copy slice extraction from `Tensor` storage works.
pub trait CpuFloat: Copy + Send + Sync + 'static {
    /// The corresponding DType.
    const DTYPE: DType;

    /// Convert a single value to f32 for computation.
    fn to_f32(self) -> f32;

    /// Convert from f32 back to this type for storage.
    fn from_f32(v: f32) -> Self;

    /// Zero value.
    fn zero() -> Self;

    /// Extract a copy of the data from a contiguous CPU Tensor.
    /// (Copy is required because the storage guard is temporary.)
    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<Self>>;

    /// Create a Tensor from a Vec<Self>.
    fn vec_to_tensor(buf: Vec<Self>, shape: &[usize], device: &Device) -> Result<Tensor>;
}

// ── BF16 ────────────────────────────────────────────────────────────────────

impl CpuFloat for half::bf16 {
    const DTYPE: DType = DType::BF16;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        half::bf16::from_f32(v)
    }

    #[inline(always)]
    fn zero() -> Self {
        half::bf16::ZERO
    }

    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<Self>> {
        Ok(tensor.as_slice::<Self>()?.to_vec())
    }

    fn vec_to_tensor(buf: Vec<Self>, shape: &[usize], device: &Device) -> Result<Tensor> {
        Tensor::from_vec(buf, shape, device)
    }
}

// ── F16 ─────────────────────────────────────────────────────────────────────

impl CpuFloat for half::f16 {
    const DTYPE: DType = DType::F16;

    #[inline(always)]
    fn to_f32(self) -> f32 { half::f16::to_f32(self) }

    #[inline(always)]
    fn from_f32(v: f32) -> Self { half::f16::from_f32(v) }

    #[inline(always)]
    fn zero() -> Self { half::f16::ZERO }

    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<Self>> {
        Ok(tensor.as_slice::<Self>()?.to_vec())
    }

    fn vec_to_tensor(buf: Vec<Self>, shape: &[usize], device: &Device) -> Result<Tensor> {
        Tensor::from_vec(buf, shape, device)
    }
}

// ── F32 ─────────────────────────────────────────────────────────────────────

impl CpuFloat for f32 {
    const DTYPE: DType = DType::F32;

    #[inline(always)]
    fn to_f32(self) -> f32 { self }

    #[inline(always)]
    fn from_f32(v: f32) -> Self { v }

    #[inline(always)]
    fn zero() -> Self { 0.0 }

    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<Self>> {
        Ok(tensor.as_slice::<Self>()?.to_vec())
    }

    fn vec_to_tensor(buf: Vec<Self>, shape: &[usize], device: &Device) -> Result<Tensor> {
        Tensor::from_vec(buf, shape, device)
    }
}
