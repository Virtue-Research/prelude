//! Bridge between candle's CustomStorage and t0-gpu's GpuBuffer.

#[cfg(feature = "rocm")]
use t0_gpu::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use t0_gpu::ignis::gpu_context::GpuRuntime;

use candle_core::{CustomDevice, CustomStorage, DType, Storage};
use std::sync::Arc;

/// Wrapper around t0-gpu's GpuBuffer, stored inside CustomStorage.
#[cfg(feature = "rocm")]
pub struct AmdBuffer {
    pub buf: Arc<GpuBuffer>,
    pub runtime: Arc<GpuRuntime>,
}

/// Extract the AmdBuffer from a candle CustomStorage.
#[cfg(feature = "rocm")]
pub fn extract_buffer(cs: &CustomStorage) -> Option<&AmdBuffer> {
    cs.downcast_ref::<AmdBuffer>()
}

/// Create a candle CustomStorage wrapping an AmdBuffer.
#[cfg(feature = "rocm")]
pub fn wrap_buffer(buf: Arc<GpuBuffer>, runtime: &Arc<GpuRuntime>, dtype: DType, device: &CustomDevice) -> CustomStorage {
    CustomStorage::new(
        AmdBuffer { buf, runtime: runtime.clone() },
        dtype,
        device.clone(),
    )
}

/// Create a candle Tensor from an AmdBuffer with contiguous layout.
///
/// Uses `Tensor::zeros` as a base then replaces storage, since
/// `Tensor::from_storage` is pub(crate) in candle-core.
#[cfg(feature = "rocm")]
pub fn tensor_from_buffer(
    buf: Arc<GpuBuffer>,
    runtime: &Arc<GpuRuntime>,
    shape: impl Into<candle_core::Shape>,
    dtype: DType,
    device: &CustomDevice,
) -> candle_core::Result<candle_core::Tensor> {
    let custom_storage = wrap_buffer(buf, runtime, dtype, device);
    let storage = Storage::Custom(custom_storage);
    let shape = shape.into();
    // Use the public from_storage API
    Ok(candle_core::Tensor::from_storage(
        storage,
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}
