//! CUDA device helpers — cached contexts, PTX loading, GpuDType, tensor conversion.
//!
//! `CudaStorage` and `CudaStorageSlice` are defined in `prelude_core::tensor` (behind
//! the `cuda` feature). This module re-exports them and provides:
//! - `GpuDType` trait: type-safe access to typed slices
//! - `CudaStorageExt` trait: `as_slice::<T>()` (depends on GpuDType)
//! - Helper functions for Tensor ↔ CudaStorage conversion
//! - `CuResultExt` trait: convert cudarc errors to our Result
//! - `cuda_device()`/`cuda_stream()`: cached device/stream registry
//! - PTX module loading and caching

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use cudarc::driver::{
    CudaFunction, CudaModule, ValidAsZeroBits,
};
use half::{bf16, f16};
use prelude_core::tensor::{
    bail, CpuStorage, DType, Device, Layout, Result, Shape, Storage, Tensor, WithDType,
};

// Re-export CUDA storage types so downstream (cuda_ops, tensor_ops_kernels, ops/*)
// can keep `use crate::device::{CudaStorage, CudaStorageSlice, ...}`
pub use prelude_core::tensor::{CudaStorage, CudaStorageSlice, CudaSlice, CudaStream, CudaContext};

// ── Error conversion ────────��─────────────────────────────────────

/// Convert cudarc DriverError to our Error.
pub trait CuResultExt<T> {
    fn ce(self) -> Result<T>;
}

impl<T> CuResultExt<T> for std::result::Result<T, cudarc::driver::DriverError> {
    fn ce(self) -> Result<T> {
        self.map_err(|e| prelude_core::tensor::Error::Msg(format!("CUDA: {e}")))
    }
}

// ── Device/Stream registry ───────��───────────────────────────────

/// Get or create a cached CudaContext for the given GPU ordinal.
///
/// Event tracking is disabled because we use a single stream per device.
/// cudarc's event tracking records inter-operation dependencies that break
/// CUDA graph capture isolation — disabling it from the start avoids this.
pub fn cuda_device(ordinal: usize) -> Result<Arc<CudaContext>> {
    static CACHE: Mutex<Vec<Option<Arc<CudaContext>>>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap();
    if cache.len() <= ordinal {
        cache.resize(ordinal + 1, None);
    }
    if cache[ordinal].is_none() {
        let ctx = CudaContext::new(ordinal).ce()?;
        // Safety: single stream per device — no cross-stream ordering needed.
        // Disabling event tracking is required for CUDA graph capture: cudarc's
        // automatic CuEventRecord/CuStreamWaitEvent calls create cross-stream
        // dependencies that violate graph capture isolation.
        unsafe { ctx.disable_event_tracking(); }
        cache[ordinal] = Some(ctx);
    }
    Ok(cache[ordinal].clone().unwrap())
}

/// Get or create the CudaStream for a GPU ordinal.
///
/// Uses `new_stream()` (CU_STREAM_NON_BLOCKING) — the NULL stream from
/// `default_stream()` does not support CUDA graph capture.
pub fn cuda_stream(ordinal: usize) -> Result<Arc<CudaStream>> {
    static CACHE: Mutex<Vec<Option<Arc<CudaStream>>>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap();
    if cache.len() <= ordinal {
        cache.resize(ordinal + 1, None);
    }
    if cache[ordinal].is_none() {
        let dev = cuda_device(ordinal)?;
        cache[ordinal] = Some(dev.new_stream().ce()?);
    }
    Ok(cache[ordinal].clone().unwrap())
}

// ── PTX module loading ───────────────────────────────────────────

/// Load a PTX module into a CudaContext, with caching by module name.
/// Uses cudarc's `load_module(Ptx::Src(...))` which calls `cuModuleLoadData`
/// (driver API — no nvrtc runtime needed despite the feature flag).
pub fn load_ptx_module(
    ctx: &Arc<CudaContext>,
    module_name: &str,
    ptx_source: &str,
) -> Result<Arc<CudaModule>> {
    static CACHE: Mutex<Option<HashMap<(usize, String), Arc<CudaModule>>>> = Mutex::new(None);
    let key = (ctx.ordinal(), module_name.to_string());
    let mut cache = CACHE.lock().unwrap();
    let map = cache.get_or_insert_with(HashMap::new);
    if let Some(module) = map.get(&key) {
        return Ok(module.clone());
    }
    let ptx = cudarc::nvrtc::Ptx::from_src(ptx_source.to_string());
    let module = ctx.load_module(ptx).ce()?;
    map.insert(key, module.clone());
    Ok(module)
}

/// Load a CUDA function from a cached PTX module.
pub fn get_or_load_func(
    ctx: &Arc<CudaContext>,
    fn_name: &str,
    module_name: &str,
    ptx_source: &str,
) -> Result<CudaFunction> {
    let module = load_ptx_module(ctx, module_name, ptx_source)?;
    module.load_function(fn_name).ce()
}


// ── CudaStorageExt — as_slice depends on GpuDType which lives here ──

/// Extension trait for `CudaStorage` methods that depend on `GpuDType`.
pub trait CudaStorageExt {
    fn as_slice<T: GpuDType>(&self) -> Result<&CudaSlice<T>>;
}

impl CudaStorageExt for CudaStorage {
    fn as_slice<T: GpuDType>(&self) -> Result<&CudaSlice<T>> {
        T::as_cuda_slice(&self.slice)
    }
}


/// Type-safe CUDA dtype mapping. Enables generic kernel dispatch.
pub trait GpuDType: DeviceRepr + Sized + Copy + 'static {
    const DTYPE: DType;
    /// Extract typed slice from CudaStorageSlice.
    fn as_cuda_slice(s: &CudaStorageSlice) -> Result<&CudaSlice<Self>>;
    /// Extract mutable typed slice from CudaStorageSlice.
    fn as_cuda_slice_mut(s: &mut CudaStorageSlice) -> Result<&mut CudaSlice<Self>>;
    /// Wrap typed slice into CudaStorageSlice.
    fn wrap_cuda_slice(s: CudaSlice<Self>) -> CudaStorageSlice;
    /// Kernel name suffix: "f32", "bf16", etc.
    fn kernel_suffix() -> &'static str;
}

macro_rules! impl_gpu_dtype {
    ($ty:ty, $variant:ident, $dtype:expr, $suffix:expr) => {
        impl GpuDType for $ty {
            const DTYPE: DType = $dtype;
            fn as_cuda_slice(s: &CudaStorageSlice) -> Result<&CudaSlice<Self>> {
                match s {
                    CudaStorageSlice::$variant(s) => Ok(s),
                    other => bail!(concat!("expected ", $suffix, ", got {:?}"), other.dtype()),
                }
            }
            fn as_cuda_slice_mut(s: &mut CudaStorageSlice) -> Result<&mut CudaSlice<Self>> {
                match s {
                    CudaStorageSlice::$variant(s) => Ok(s),
                    other => bail!(concat!("expected mut ", $suffix, ", got {:?}"), other.dtype()),
                }
            }
            fn wrap_cuda_slice(s: CudaSlice<Self>) -> CudaStorageSlice {
                CudaStorageSlice::$variant(s)
            }
            fn kernel_suffix() -> &'static str {
                $suffix
            }
        }
    };
}

impl_gpu_dtype!(u8, U8, DType::U8, "u8");
impl_gpu_dtype!(u32, U32, DType::U32, "u32");
impl_gpu_dtype!(i32, I32, DType::I32, "i32");
impl_gpu_dtype!(i64, I64, DType::I64, "i64");
impl_gpu_dtype!(bf16, BF16, DType::BF16, "bf16");
impl_gpu_dtype!(f16, F16, DType::F16, "f16");
impl_gpu_dtype!(f32, F32, DType::F32, "f32");
impl_gpu_dtype!(f64, F64, DType::F64, "f64");

// ── Tensor ↔ CudaStorage helpers ─────────────────────────────────

/// Extract `&CudaStorage` from a Storage reference.
pub fn as_cuda<'a>(storage: &'a Storage, _ctx: &str) -> Result<&'a CudaStorage> {
    storage.as_cuda()
}

/// Extract `&mut CudaStorage` from a mutable Storage reference.
pub fn as_cuda_mut<'a>(storage: &'a mut Storage, _ctx: &str) -> Result<&'a mut CudaStorage> {
    storage.as_cuda_mut()
}

// ── Tensor ↔ CudaStorage extraction helpers ────────────────────

/// Extract storage + layout from a Tensor.
pub fn storage_and_layout(t: &Tensor) -> (&Storage, &Layout) {
    (t.storage(), t.our_layout())
}

/// Re-export cudarc types used by ops kernels.
pub use cudarc::driver::{
    DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};

/// Create a Tensor from a CudaStorage (contiguous, no grad).
pub fn tensor_from_device(storage: CudaStorage, shape: Shape) -> Tensor {
    let dtype = storage.dtype();
    let device = Device::Cuda(storage.stream.context().ordinal());
    let layout = Layout::contiguous(shape);
    Tensor::from_storage_layout(Arc::new(Storage::Cuda(storage)), layout, dtype, device)
}

/// Create a Tensor from a typed CudaSlice (contiguous, no grad).
pub fn tensor_from_cuda<T: GpuDType>(
    slice: CudaSlice<T>,
    stream: Arc<CudaStream>,
    shape: impl Into<Shape>,
) -> Tensor {
    let storage = CudaStorage {
        slice: T::wrap_cuda_slice(slice),
        stream,
    };
    tensor_from_device(storage, shape.into())
}

/// Get the CudaStream from a Tensor.
pub fn tensor_stream(t: &Tensor) -> Result<Arc<CudaStream>> {
    let cuda = as_cuda(t.storage(), "tensor_stream")?;
    Ok(cuda.stream.clone())
}

/// Get the CudaContext from a Tensor.
pub fn tensor_cuda_device(t: &Tensor) -> Result<Arc<CudaContext>> {
    let cuda = as_cuda(t.storage(), "tensor_cuda_device")?;
    Ok(cuda.device().clone())
}

// ── Synchronize ──────────────────────────────────────────────────

/// Synchronize the default CUDA stream for the given device.
pub fn synchronize(device: &prelude_core::tensor::Device) -> Result<()> {
    let stream = cuda_stream(device.ordinal())?;
    stream.synchronize().ce()
}

