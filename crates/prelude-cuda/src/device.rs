//! CUDA device helpers and own storage types.
//!
//! - `CudaStorageSlice` / `CudaStorage`: typed GPU memory (used by tensor_ops_kernels)
//! - `GpuDType` trait: type-safe access to typed slices
//! - `tensor_from_cuda()`: wrap a CudaSlice into a candle Tensor
//! - `tensor_stream()`: get CudaStream from a Tensor's device
//! - PTX module loading and caching

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice,
};
use half::{bf16, f16};
use prelude_core::tensor::{
    bail, CpuStorage, DType, Device, Layout, Result, Shape,
    Storage, Tensor,
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

// ── Device/Stream helpers ────────────────────────────────────────

/// Probe whether a CUDA device exists at the given ordinal.
/// Only used for startup probing — creates a temporary context.
pub fn cuda_probe(ordinal: usize) -> bool {
    CudaContext::new(ordinal).is_ok()
}

/// Get the CudaStream from a candle Device.
pub fn device_stream(device: &Device) -> Result<Arc<CudaStream>> {
    match device {
        Device::Cuda(d) => Ok(d.cuda_stream()),
        _ => bail!("expected CUDA device"),
    }
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

    /// Upload contiguous CPU data to GPU.
    pub fn from_cpu(stream: &Arc<CudaStream>, cpu: &CpuStorage, layout: &Layout) -> Result<Self> {
        macro_rules! upload {
            ($data:expr, $variant:ident) => {{
                let start = layout.start_offset();
                let len = layout.shape().elem_count();
                CudaStorageSlice::$variant(stream.clone_htod(&$data[start..start + len]).ce()?)
            }};
        }
        assert!(layout.is_contiguous(), "from_cpu: non-contiguous layout; use tensor.contiguous() first");
        let slice = match cpu {
            CpuStorage::U8(v) => upload!(v, U8),
            CpuStorage::U32(v) => upload!(v, U32),
            CpuStorage::I32(v) => upload!(v, I32),
            CpuStorage::I64(v) => upload!(v, I64),
            CpuStorage::BF16(v) => upload!(v, BF16),
            CpuStorage::F16(v) => upload!(v, F16),
            CpuStorage::F32(v) => upload!(v, F32),
            CpuStorage::F64(v) => upload!(v, F64),
            _ => bail!("from_cpu: unsupported dtype"),
        };
        Ok(Self { slice, stream: stream.clone() })
    }

    /// Download contiguous GPU data to CPU storage.
    pub fn to_cpu(&self, _layout: &Layout) -> Result<CpuStorage> {
        macro_rules! download {
            ($slice:expr, $variant:ident) => {{
                CpuStorage::$variant(self.stream.clone_dtoh($slice).ce()?)
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

/// Re-export cudarc types used by ops kernels.
pub use cudarc::driver::{
    CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};

/// Create a candle Tensor from a typed CudaSlice (contiguous, no grad).
pub fn tensor_from_cuda<T: GpuDType + candle_core::cuda_backend::CudaDType>(
    slice: CudaSlice<T>,
    dev: &candle_core::CudaDevice,
    shape: impl Into<Shape>,
) -> Tensor {
    let candle_storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
    Tensor::from_storage(
        Storage::Cuda(candle_storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    )
}

/// Get the CudaStream from a Tensor.
pub fn tensor_stream(t: &Tensor) -> Result<Arc<CudaStream>> {
    device_stream(t.device())
}

// ── Synchronize ──────────────────────────────────────────────────

/// Synchronize the CUDA stream for the given device.
pub fn synchronize(device: &Device) -> Result<()> {
    device_stream(device)?.synchronize().ce()
}

