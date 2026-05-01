//! AmdOps — Ops implementation for AMD GPUs.
//!
//! Routes operations based on detected GPU architecture:
//! - GFX1100 (RDNA3): uses t0-gpu's compiled kernels via GpuRuntime
//! - GFX942 (CDNA3): planned (via HSA runtime)

use candle_core::{CustomDevice, DType};
use prelude_core::ops::Ops;
use prelude_core::ops::traits::{UnaryOp, BinaryOp};
use prelude_core::tensor::{Tensor, Result};
use crate::device::{GfxArch, detect_arch};

#[cfg(feature = "rocm")]
use t0_gpu::ignis::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use t0_gpu::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use crate::storage;
#[cfg(feature = "rocm")]
use std::sync::Arc;

pub struct AmdOps {
    pub(crate) arch: GfxArch,
    pub(crate) custom_device: CustomDevice,
    #[cfg(feature = "rocm")]
    pub(crate) runtime: Arc<GpuRuntime>,
}

pub fn amd_ops() -> &'static dyn Ops {
    use std::sync::LazyLock;
    static OPS: LazyLock<AmdOps> = LazyLock::new(|| {
        let arch = detect_arch();
        tracing::info!("AmdOps: arch={:?}", arch);

        #[cfg(feature = "rocm")]
        let runtime = GpuRuntime::new().expect("failed to create t0-gpu GpuRuntime");

        let custom_device = CustomDevice::new(arch, 0, format!("amd:{:?}", arch));

        AmdOps {
            arch,
            custom_device,
            #[cfg(feature = "rocm")]
            runtime,
        }
    });
    &*OPS
}

// ── Helpers ─────────────────────────────────────────────────────

#[cfg(feature = "rocm")]
impl AmdOps {
    /// Extract GpuBuffer from a candle Tensor backed by CustomStorage.
    fn buf(&self, t: &Tensor) -> Option<Arc<GpuBuffer>> {
        let (st, _) = t.storage_and_layout();
        match &*st {
            candle_core::Storage::Custom(cs) => storage::extract_buffer(cs).map(|ab| ab.buf.clone()),
            _ => None,
        }
    }

    /// Wrap a GpuBuffer result into a candle Tensor.
    fn wrap(&self, buf: GpuBuffer, shape: &[usize], dtype: DType) -> Result<Tensor> {
        storage::tensor_from_buffer(Arc::new(buf), &self.runtime, shape.to_vec(), dtype, &self.custom_device)
    }

    /// Map a t0-gpu Result<GpuBuffer, String> to candle Result<Tensor>.
    fn wrap_result(&self, r: std::result::Result<GpuBuffer, String>, shape: &[usize], dtype: DType) -> Result<Tensor> {
        self.wrap(r.map_err(|e| candle_core::Error::Msg(e))?, shape, dtype)
    }
}

// ── Ops implementation ──────────────────────────────────────────

impl Ops for AmdOps {
    fn default_impl(&self) -> &dyn Ops { self }

    // ── CPU → AMD GPU transfer ──────────────────────────────────

    #[cfg(feature = "rocm")]
    fn to_device(&self, x: &Tensor, device: &candle_core::Device) -> Option<Result<Tensor>> {
        if !x.device().is_cpu() || !device.is_custom() { return None; }
        Some((|| -> Result<Tensor> {
            let dtype = x.dtype();
            let shape: Vec<usize> = x.dims().to_vec();
            let flat = x.flatten_all()?;
            let data: Vec<f32> = flat.to_dtype(DType::F32)?.to_vec1()?;
            let buf = crate::kernels::upload_f32(&self.runtime, &data)
                .map_err(|e| candle_core::Error::Msg(e))?;
            storage::tensor_from_buffer(Arc::new(buf), &self.runtime, shape, dtype, &self.custom_device)
        })())
    }

    // ── GEMM ────────────────────────────────────────────────────

    #[cfg(feature = "rocm")]
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        if !a.device().is_custom() { return None; }
        if self.arch != GfxArch::Gfx1100 {
            return Some(Err(candle_core::Error::Msg(format!("AMD {:?}: matmul not supported", self.arch))));
        }
        Some((|| -> Result<Tensor> {
            let ab = self.buf(a).ok_or_else(|| candle_core::Error::Msg("matmul: not AMD tensor".into()))?;
            let bb = self.buf(b).ok_or_else(|| candle_core::Error::Msg("matmul: not AMD tensor".into()))?;
            let dims = a.dims();
            let dim = dims.len();
            let (m, k, n) = (dims[dim-2], dims[dim-1], b.dims()[dim-1]);
            let out = crate::kernels::gemm_bf16(&self.runtime, &ab, &bb, m, k, n)
                .map_err(|e| candle_core::Error::Msg(e))?;
            let mut shape = dims[..dim-2].to_vec();
            shape.extend_from_slice(&[m, n]);
            self.wrap(out, &shape, DType::F32)
        })())
    }

    // ── Unary elementwise ───────────────────────────────────────

    #[cfg(feature = "rocm")]
    fn unary(&self, x: &Tensor, op: UnaryOp) -> Option<Result<Tensor>> {
        if !x.device().is_custom() || self.arch != GfxArch::Gfx1100 { return None; }
        let xb = self.buf(x)?;
        let n = x.elem_count();
        let shape: Vec<usize> = x.dims().to_vec();
        let rt = &self.runtime;
        let r = match op {
            UnaryOp::Exp  => crate::kernels::exp_f32(rt, &xb, n),
            UnaryOp::Neg  => crate::kernels::neg_f32(rt, &xb, n),
            UnaryOp::Sqrt => crate::kernels::sqrt_f32(rt, &xb, n),
            UnaryOp::Recip => crate::kernels::recip_f32(rt, &xb, n),
            UnaryOp::Abs  => crate::kernels::abs_f32(rt, &xb, n),
            _ => return None,
        };
        Some(self.wrap_result(r, &shape, x.dtype()))
    }

    // ── Binary elementwise ──────────────────────────────────────

    #[cfg(feature = "rocm")]
    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Option<Result<Tensor>> {
        if !a.device().is_custom() || self.arch != GfxArch::Gfx1100 { return None; }
        let ab = self.buf(a)?;
        let bb = self.buf(b)?;
        let n = a.elem_count();
        let shape: Vec<usize> = a.dims().to_vec();
        let rt = &self.runtime;
        let r = match op {
            BinaryOp::Add => crate::kernels::add_f32(rt, &ab, &bb, n),
            BinaryOp::Sub => crate::kernels::sub_f32(rt, &ab, &bb, n),
            BinaryOp::Mul => crate::kernels::mul_f32(rt, &ab, &bb, n),
            BinaryOp::Div => crate::kernels::div_f32(rt, &ab, &bb, n),
            _ => return None,
        };
        Some(self.wrap_result(r, &shape, a.dtype()))
    }

    // ── Stubs ───────────────────────────────────────────────────

    fn contiguous(&self, x: &Tensor) -> Option<Result<Tensor>> {
        if !x.device().is_custom() { return None; }
        None // TODO: strided → contiguous copy kernel
    }

    fn to_dtype(&self, x: &Tensor, _dtype: DType) -> Option<Result<Tensor>> {
        if !x.device().is_custom() { return None; }
        None // TODO: type conversion kernel
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        // TODO: t0-gpu rmsnorm kernel for GFX1100
        prelude_core::ops::traits::norm::rms_norm(x, weight, eps)
    }
}
