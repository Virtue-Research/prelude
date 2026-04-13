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

        let custom_device = CustomDevice::new(
            arch,
            0,
            format!("amd:{:?}", arch),
        );

        AmdOps {
            arch,
            custom_device,
            #[cfg(feature = "rocm")]
            runtime,
        }
    });
    &*OPS
}

impl Ops for AmdOps {
    fn default_impl(&self) -> &dyn Ops { self }

    // ════════════════════════════════════════════════════════════════
    // TensorHook overrides
    // ════════════════════════════════════════════════════════════════

    fn hook_matmul(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
        if !a.device().is_custom() { return None; }
        match self.arch {
            GfxArch::Gfx1100 => {
                // TODO: t0-gpu GEMM dispatch
                None
            }
            _ => Some(Err(candle_core::Error::Msg(
                format!("AMD {:?}: matmul not supported", self.arch),
            ).into())),
        }
    }

    fn hook_unary(&self, x: &Tensor, _op: UnaryOp) -> Option<Result<Tensor>> {
        if !x.device().is_custom() { return None; }
        match self.arch {
            GfxArch::Gfx1100 => None, // TODO: t0-gpu elementwise
            _ => Some(Err(candle_core::Error::Msg(
                format!("AMD {:?}: unary op not supported", self.arch),
            ).into())),
        }
    }

    fn hook_binary(&self, a: &Tensor, _b: &Tensor, _op: BinaryOp) -> Option<Result<Tensor>> {
        if !a.device().is_custom() { return None; }
        match self.arch {
            GfxArch::Gfx1100 => None, // TODO
            _ => Some(Err(candle_core::Error::Msg(
                format!("AMD {:?}: binary op not supported", self.arch),
            ).into())),
        }
    }

    fn hook_contiguous(&self, x: &Tensor) -> Option<Result<Tensor>> {
        if !x.device().is_custom() { return None; }
        None // TODO
    }

    fn hook_to_dtype(&self, x: &Tensor, _dtype: DType) -> Option<Result<Tensor>> {
        if !x.device().is_custom() { return None; }
        None // TODO
    }

    fn hook_to_device(&self, x: &Tensor, device: &candle_core::Device) -> Option<Result<Tensor>> {
        if !x.device().is_cpu() || !device.is_custom() {
            return None;
        }
        // CPU → AMD transfer
        // TODO: implement via KFD GTT memory upload
        Some(Err(candle_core::Error::Msg(
            "AMD to_device: not yet implemented".to_string(),
        ).into()))
    }

    // ════════════════════════════════════════════════════════════════
    // Normalization
    // ════════════════════════════════════════════════════════════════

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if x.device().is_custom() && self.arch == GfxArch::Gfx1100 {
            // TODO: t0-gpu rmsnorm kernel
        }
        prelude_core::ops::traits::norm::rms_norm(x, weight, eps)
    }
}
