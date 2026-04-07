//! Own CUDA storage — own CUDA storage wrapping cudarc types.
//!
//! This module provides:
//! - `CudaStorageSlice` enum: typed GPU memory slices
//! - `CudaStorage` struct: GPU storage with device/stream handle
//! - `GpuDType` trait: type-safe access to typed slices
//! - Helper functions for Tensor ↔ CudaStorage conversion
//! - `CuResultExt` trait: convert cudarc errors to our Result
//! - `cuda_device()`/`cuda_stream()`: cached device/stream registry
//! - PTX module loading and caching

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, ValidAsZeroBits,
};
use half::{bf16, f16};
use prelude_core::tensor::{
    bail, CpuStorage, DType, Device, DeviceStorage, DeviceStorageTrait, Layout, Result, Shape,
    Storage, Tensor, WithDType,
};

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
pub fn cuda_device(ordinal: usize) -> Result<Arc<CudaContext>> {
    static CACHE: Mutex<Vec<Option<Arc<CudaContext>>>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap();
    if cache.len() <= ordinal {
        cache.resize(ordinal + 1, None);
    }
    if cache[ordinal].is_none() {
        cache[ordinal] = Some(CudaContext::new(ordinal).ce()?);
    }
    Ok(cache[ordinal].clone().unwrap())
}

/// Get or create the default CudaStream for a GPU ordinal.
pub fn cuda_stream(ordinal: usize) -> Result<Arc<CudaStream>> {
    static CACHE: Mutex<Vec<Option<Arc<CudaStream>>>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap();
    if cache.len() <= ordinal {
        cache.resize(ordinal + 1, None);
    }
    if cache[ordinal].is_none() {
        let dev = cuda_device(ordinal)?;
        cache[ordinal] = Some(dev.default_stream());
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

// ── CudaStorageSlice ─────��───────────────────────────────────────

/// Typed CUDA memory slice. Each variant holds a `CudaSlice<T>`.
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(s) => s.len(),
            Self::U32(s) => s.len(),
            Self::I32(s) => s.len(),
            Self::I64(s) => s.len(),
            Self::BF16(s) => s.len(),
            Self::F16(s) => s.len(),
            Self::F32(s) => s.len(),
            Self::F64(s) => s.len(),
        }
    }
}

impl std::fmt::Debug for CudaStorageSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaStorageSlice({:?}, len={})", self.dtype(), self.len())
    }
}

// ── CudaStorage ──────────────────────────────────────────────────

/// GPU tensor storage wrapping cudarc types directly.
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub stream: Arc<CudaStream>,
}

impl CudaStorage {
    /// The CudaContext (GPU device) this storage lives on.
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.stream.context()
    }

    pub fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Get a typed CudaSlice reference.
    pub fn as_slice<T: GpuDType>(&self) -> Result<&CudaSlice<T>> {
        T::as_cuda_slice(&self.slice)
    }

    /// Allocate zeroed GPU memory of given dtype and element count.
    pub fn zeros(stream: &Arc<CudaStream>, dtype: DType, n: usize) -> Result<Self> {
        let slice = match dtype {
            DType::U8 => CudaStorageSlice::U8(stream.alloc_zeros::<u8>(n).ce()?),
            DType::U32 => CudaStorageSlice::U32(stream.alloc_zeros::<u32>(n).ce()?),
            DType::I32 => CudaStorageSlice::I32(stream.alloc_zeros::<i32>(n).ce()?),
            DType::I64 => CudaStorageSlice::I64(stream.alloc_zeros::<i64>(n).ce()?),
            DType::BF16 => CudaStorageSlice::BF16(stream.alloc_zeros::<bf16>(n).ce()?),
            DType::F16 => CudaStorageSlice::F16(stream.alloc_zeros::<f16>(n).ce()?),
            DType::F32 => CudaStorageSlice::F32(stream.alloc_zeros::<f32>(n).ce()?),
            DType::F64 => CudaStorageSlice::F64(stream.alloc_zeros::<f64>(n).ce()?),
            dt => bail!("CudaStorage::zeros: unsupported dtype {dt:?}"),
        };
        Ok(Self { slice, stream: stream.clone() })
    }

    /// Upload CPU data to GPU.
    pub fn from_cpu(stream: &Arc<CudaStream>, cpu: &CpuStorage, layout: &Layout) -> Result<Self> {
        macro_rules! upload {
            ($data:expr, $variant:ident) => {{
                let data = if layout.is_contiguous() {
                    let start = layout.start_offset();
                    let len = layout.shape().elem_count();
                    &$data[start..start + len]
                } else {
                    let extracted: Vec<_> = layout.strided_index().map(|i| $data[i]).collect();
                    return Ok(Self {
                        slice: CudaStorageSlice::$variant(stream.clone_htod(&extracted).ce()?),
                        stream: stream.clone(),
                    });
                };
                CudaStorageSlice::$variant(stream.clone_htod(data).ce()?)
            }};
        }
        let slice = match cpu {
            CpuStorage::U8(v) => upload!(v, U8),
            CpuStorage::U32(v) => upload!(v, U32),
            CpuStorage::I32(v) => upload!(v, I32),
            CpuStorage::I64(v) => upload!(v, I64),
            CpuStorage::BF16(v) => upload!(v, BF16),
            CpuStorage::F16(v) => upload!(v, F16),
            CpuStorage::F32(v) => upload!(v, F32),
            CpuStorage::F64(v) => upload!(v, F64),
        };
        Ok(Self { slice, stream: stream.clone() })
    }

    /// Download GPU data to CPU storage.
    pub fn to_cpu(&self, layout: &Layout) -> Result<CpuStorage> {
        macro_rules! download {
            ($slice:expr, $variant:ident) => {{
                let data = self.stream.clone_dtoh($slice).ce()?;
                if layout.is_contiguous() {
                    let start = layout.start_offset();
                    let len = layout.shape().elem_count();
                    CpuStorage::$variant(data[start..start + len].to_vec())
                } else {
                    let extracted: Vec<_> = layout.strided_index().map(|i| data[i]).collect();
                    CpuStorage::$variant(extracted)
                }
            }};
        }
        Ok(match &self.slice {
            CudaStorageSlice::U8(s) => download!(s, U8),
            CudaStorageSlice::U32(s) => download!(s, U32),
            CudaStorageSlice::I32(s) => download!(s, I32),
            CudaStorageSlice::I64(s) => download!(s, I64),
            CudaStorageSlice::BF16(s) => download!(s, BF16),
            CudaStorageSlice::F16(s) => download!(s, F16),
            CudaStorageSlice::F32(s) => download!(s, F32),
            CudaStorageSlice::F64(s) => download!(s, F64),
        })
    }
}

impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudaStorage({:?}, gpu:{})",
            self.slice,
            self.stream.context().ordinal()
        )
    }
}

// ── DeviceStorageTrait impl ──────────────────────────────────────

impl DeviceStorageTrait for CudaStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn DeviceStorageTrait> {
        macro_rules! clone_slice {
            ($s:expr, $variant:ident) => {
                CudaStorageSlice::$variant(self.stream.clone_dtod($s).unwrap())
            };
        }
        let new_slice = match &self.slice {
            CudaStorageSlice::U8(s) => clone_slice!(s, U8),
            CudaStorageSlice::U32(s) => clone_slice!(s, U32),
            CudaStorageSlice::I32(s) => clone_slice!(s, I32),
            CudaStorageSlice::I64(s) => clone_slice!(s, I64),
            CudaStorageSlice::BF16(s) => clone_slice!(s, BF16),
            CudaStorageSlice::F16(s) => clone_slice!(s, F16),
            CudaStorageSlice::F32(s) => clone_slice!(s, F32),
            CudaStorageSlice::F64(s) => clone_slice!(s, F64),
        };
        Box::new(CudaStorage {
            slice: new_slice,
            stream: self.stream.clone(),
        })
    }
    fn dtype(&self) -> DType {
        self.slice.dtype()
    }
    fn len(&self) -> usize {
        self.slice.len()
    }
}

// ── GpuDType trait ─────────���─────────────────────────────────────

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

/// Extract `&CudaStorage` from a Tensor's storage guard.
/// Caller must hold the RwLockReadGuard for the duration of GPU access.
pub fn as_cuda<'a>(
    guard: &'a std::sync::RwLockReadGuard<Storage>,
    ctx: &str,
) -> Result<&'a CudaStorage> {
    match &**guard {
        Storage::Device(ds) => ds
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| prelude_core::tensor::Error::Msg(format!("{ctx}: not CudaStorage"))),
        _ => bail!("{ctx}: requires CUDA tensor"),
    }
}

/// Extract `&mut CudaStorage` from a Tensor's write guard.
pub fn as_cuda_mut<'a>(
    guard: &'a mut std::sync::RwLockWriteGuard<Storage>,
    ctx: &str,
) -> Result<&'a mut CudaStorage> {
    match &mut **guard {
        Storage::Device(ds) => ds
            .downcast_mut::<CudaStorage>()
            .ok_or_else(|| prelude_core::tensor::Error::Msg(format!("{ctx}: not CudaStorage"))),
        _ => bail!("{ctx}: requires CUDA tensor"),
    }
}

// ── Tensor ↔ CudaStorage extraction helpers ────────────────────

/// Extract storage guard + layout from a Tensor.
pub fn storage_and_layout(t: &Tensor) -> (std::sync::RwLockReadGuard<'_, Storage>, &Layout) {
    let guard = t.storage_rw().read().expect("lock poisoned");
    let layout = t.our_layout();
    (guard, layout)
}

/// Re-export cudarc types used by ops kernels.
pub use cudarc::driver::{
    CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};

/// Create a Tensor from a CudaStorage (contiguous, no grad).
pub fn tensor_from_device(storage: CudaStorage, shape: Shape) -> Tensor {
    let dtype = storage.dtype();
    let device = Device::Cuda(storage.stream.context().ordinal());
    let layout = Layout::contiguous(shape);
    let ds = DeviceStorage::new(Box::new(storage));
    Tensor::from_storage_layout(Arc::new(RwLock::new(Storage::Device(ds))), layout, dtype, device)
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
    let guard = t.storage_rw().read().map_err(|_| {
        prelude_core::tensor::Error::Msg("lock poisoned".into())
    })?;
    let cuda = as_cuda(&guard, "tensor_stream")?;
    Ok(cuda.stream.clone())
}

/// Get the CudaContext from a Tensor.
pub fn tensor_cuda_device(t: &Tensor) -> Result<Arc<CudaContext>> {
    let guard = t.storage_rw().read().map_err(|_| {
        prelude_core::tensor::Error::Msg("lock poisoned".into())
    })?;
    let cuda = as_cuda(&guard, "tensor_cuda_device")?;
    Ok(cuda.device().clone())
}

// ── Synchronize ──────────────────────────────────────────────────

/// Synchronize the default CUDA stream for the given device.
pub fn synchronize(device: &prelude_core::tensor::Device) -> Result<()> {
    let stream = cuda_stream(device.ordinal())?;
    stream.synchronize().ce()
}

