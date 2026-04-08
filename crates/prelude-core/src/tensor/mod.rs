//! Prelude Tensor — backed by candle-core.
//!
//! Re-exports candle-core types as the tensor API. All basic tensor ops
//! (matmul, reshape, contiguous, etc.) go through candle's zero-overhead
//! dispatch. Custom inference ops (fused kernels, paged attention) use
//! the Ops trait in `crate::ops`.

// ── Re-exports from candle-core ─────────────────────────────────

pub use candle_core::{
    bail, DType, Device, Error, IndexOp, Layout, Module, Result, Shape, Tensor,
    WithDType, FloatDType, IntDType,
    CpuStorage, Storage,
    D,
};

pub use candle_core::error::Context;

#[cfg(feature = "cuda")]
pub use candle_core::{CudaDevice, CudaStorage};

#[cfg(feature = "cuda")]
pub use candle_core::cuda_backend as cuda;

// ── Our modules (kept) ──────────────────────────────────────────

pub mod quantized;

// ── Compatibility: Dim trait (candle uses D enum, we also need Dim) ─

/// Trait to convert dimension specifiers to indices.
/// candle_core::D implements this natively, but we also allow usize.
pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        if *self >= rank {
            bail!("{op}: index {self} out of range for rank {rank}")
        }
        Ok(*self)
    }
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        if *self > rank {
            bail!("{op}: index {self} out of range for rank+1 {}", rank + 1)
        }
        Ok(*self)
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            D::Minus1 => {
                if rank == 0 { bail!("{op}: cannot use Minus1 on rank-0 tensor") }
                Ok(rank - 1)
            }
            D::Minus2 => {
                if rank < 2 { bail!("{op}: cannot use Minus2 on rank-{rank} tensor") }
                Ok(rank - 2)
            }
        }
    }
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        // For unsqueeze, Minus1 means last position = rank
        let rank = shape.rank();
        match self {
            D::Minus1 => Ok(rank),
            D::Minus2 => {
                if rank == 0 { bail!("{op}: cannot use Minus2 on rank-0 tensor") }
                Ok(rank - 1)
            }
        }
    }
}

// ── Compatibility: ShapeWithOneHole ─────────────────────────────

/// Allows reshape with one unknown dimension (specified as ()).
pub trait ShapeWithOneHole {
    fn into_shape(self, elem_count: usize) -> Result<Shape>;
}

impl ShapeWithOneHole for Shape {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(self) }
}

impl ShapeWithOneHole for &[usize] {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(self.to_vec())) }
}

impl ShapeWithOneHole for Vec<usize> {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(self)) }
}

impl ShapeWithOneHole for usize {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(vec![self])) }
}

impl ShapeWithOneHole for (usize,) {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(vec![self.0])) }
}

impl ShapeWithOneHole for (usize, usize) {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(vec![self.0, self.1])) }
}

impl ShapeWithOneHole for (usize, usize, usize) {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(vec![self.0, self.1, self.2])) }
}

impl ShapeWithOneHole for (usize, usize, usize, usize) {
    fn into_shape(self, _: usize) -> Result<Shape> { Ok(Shape::from(vec![self.0, self.1, self.2, self.3])) }
}

// ── Compatibility: extension methods on candle Tensor ───────────

/// Extension trait adding prelude-specific methods to candle's Tensor.
pub trait TensorExt {
    /// Alias for `layout()` — backward compat with our old API.
    fn our_layout(&self) -> &Layout;

    /// Get raw storage ref — wraps candle's `storage_and_layout()`.
    /// Note: this holds a read lock on storage.
    fn storage_ref(&self) -> std::sync::RwLockReadGuard<'_, Storage>;
}

impl TensorExt for Tensor {
    fn our_layout(&self) -> &Layout {
        self.layout()
    }

    fn storage_ref(&self) -> std::sync::RwLockReadGuard<'_, Storage> {
        // storage_and_layout() returns (guard, &layout), we just want the guard
        self.storage_and_layout().0
    }
}

// ── Compatibility: DeviceExt ────────────────────────────────────

/// Extension trait for Device compatibility.
pub trait DeviceExt {
    fn ordinal(&self) -> usize;
}

impl DeviceExt for Device {
    fn ordinal(&self) -> usize {
        match self {
            Device::Cpu => 0,
            #[cfg(feature = "cuda")]
            Device::Cuda(d) => d.ordinal(),
            #[allow(unreachable_patterns)]
            _ => 0,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// CPU storage helpers — kept for quantized module and CPU ops.
pub use candle_core::cpu_backend::CpuStorage as CpuStorageInner;

pub fn cpu_extract_vec<T: WithDType>(
    storage: &CpuStorage,
    layout: &Layout,
) -> Result<Vec<T>> {
    storage.as_slice::<T>(layout)
        .map(|s| s.to_vec())
}
