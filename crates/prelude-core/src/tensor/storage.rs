//! Tensor storage — direct enum dispatch, zero overhead.
//!
//! `Storage::Cpu(CpuStorage)` — CPU tensors (Vec<T>).
//! `Storage::Cuda(CudaStorage)` — GPU tensors (cudarc CudaSlice<T>), behind `#[cfg(feature = "cuda")]`.

use std::sync::Arc;
use super::{DType, Error, Layout, Result};

// ── CPU Storage ─────────────────────────────────────────────────

/// Typed CPU tensor data. Each variant holds a flat contiguous array.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    BF16(Vec<half::bf16>),
    F16(Vec<half::f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8, Self::U32(_) => DType::U32, Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64, Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16, Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(), Self::U32(v) => v.len(), Self::I32(v) => v.len(), Self::I64(v) => v.len(),
            Self::BF16(v) => v.len(), Self::F16(v) => v.len(),
            Self::F32(v) => v.len(), Self::F64(v) => v.len(),
        }
    }

    pub fn zeros(dtype: DType, len: usize) -> Self {
        match dtype {
            DType::U8 => Self::U8(vec![0u8; len]),
            DType::U32 => Self::U32(vec![0u32; len]),
            DType::I32 => Self::I32(vec![0i32; len]),
            DType::I64 => Self::I64(vec![0i64; len]),
            DType::BF16 => Self::BF16(vec![half::bf16::ZERO; len]),
            DType::F16 => Self::F16(vec![half::f16::ZERO; len]),
            DType::F32 => Self::F32(vec![0f32; len]),
            DType::F64 => Self::F64(vec![0f64; len]),
            _ => panic!("unsupported dtype for zeros: {dtype:?}"),
        }
    }

    pub fn from_typed_vec<T: super::WithDType>(data: Vec<T>) -> Self {
        use super::DType;
        match <T as super::WithDType>::DTYPE {
            DType::U8 => Self::U8(unsafe { transmute_vec(data) }),
            DType::U32 => Self::U32(unsafe { transmute_vec(data) }),
            DType::I32 => Self::I32(unsafe { transmute_vec(data) }),
            DType::I64 => Self::I64(unsafe { transmute_vec(data) }),
            DType::BF16 => Self::BF16(unsafe { transmute_vec(data) }),
            DType::F16 => Self::F16(unsafe { transmute_vec(data) }),
            DType::F32 => Self::F32(unsafe { transmute_vec(data) }),
            DType::F64 => Self::F64(unsafe { transmute_vec(data) }),
            _ => panic!("unsupported dtype for from_typed_vec"),
        }
    }

    pub fn from_raw_bytes(bytes: &[u8], dtype: DType, elem_count: usize) -> Result<Self> {
        let expected = elem_count * dtype.size_in_bytes();
        if bytes.len() < expected {
            return Err(Error::Msg(format!(
                "from_raw_bytes: expected {expected} bytes, got {}", bytes.len()
            ).into()).bt());
        }
        Ok(match dtype {
            DType::U8 => Self::U8(bytes[..expected].to_vec()),
            DType::U32 => Self::U32(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            DType::I32 => Self::I32(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            DType::I64 => Self::I64(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            DType::BF16 => Self::BF16(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            DType::F16 => Self::F16(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            DType::F32 => Self::F32(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            DType::F64 => Self::F64(bytemuck::cast_slice(&bytes[..expected]).to_vec()),
            _ => return Err(Error::Msg(format!("from_raw_bytes: unsupported dtype {dtype:?}").into()).bt()),
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::U8(v) => v.as_slice(),
            Self::U32(v) => bytemuck::cast_slice(v),
            Self::I32(v) => bytemuck::cast_slice(v),
            Self::I64(v) => bytemuck::cast_slice(v),
            Self::BF16(v) => bytemuck::cast_slice(v),
            Self::F16(v) => bytemuck::cast_slice(v),
            Self::F32(v) => bytemuck::cast_slice(v),
            Self::F64(v) => bytemuck::cast_slice(v),
        }
    }
}

unsafe fn transmute_vec<T, U>(mut v: Vec<T>) -> Vec<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    let ptr = v.as_mut_ptr() as *mut U;
    let len = v.len();
    let cap = v.capacity();
    std::mem::forget(v);
    Vec::from_raw_parts(ptr, len, cap)
}

pub fn cpu_extract_vec<T: super::WithDType>(
    cpu: &CpuStorage,
    layout: &super::Layout,
) -> Result<Vec<T>> {
    macro_rules! extract {
        ($data:expr) => {{
            if layout.is_contiguous() {
                let start = layout.start_offset();
                let len = layout.shape().elem_count();
                let slice = &$data[start..start + len];
                Ok(slice.iter().map(|&v| {
                    <T as super::WithDType>::from_f64(<_ as super::WithDType>::to_f64(v))
                }).collect())
            } else {
                Ok(layout.strided_index().map(|i| {
                    <T as super::WithDType>::from_f64(<_ as super::WithDType>::to_f64($data[i]))
                }).collect())
            }
        }};
    }
    match cpu {
        CpuStorage::U8(v) => extract!(v), CpuStorage::U32(v) => extract!(v),
        CpuStorage::I32(v) => extract!(v), CpuStorage::I64(v) => extract!(v),
        CpuStorage::BF16(v) => extract!(v), CpuStorage::F16(v) => extract!(v),
        CpuStorage::F32(v) => extract!(v), CpuStorage::F64(v) => extract!(v),
    }
}

pub fn cpu_slice_set(
    dst: &mut CpuStorage, dst_layout: &super::Layout,
    src: &CpuStorage, src_layout: &super::Layout,
    dim: usize, start: usize,
) -> Result<()> {
    macro_rules! slice_set_typed {
        ($dst_v:expr, $src_v:expr) => {{
            let dst_dims = dst_layout.dims();
            let src_dims = src_layout.dims();
            let dst_stride = dst_layout.stride();
            for (src_i, src_phys) in src_layout.strided_index().enumerate() {
                let mut multi = vec![0usize; src_dims.len()];
                let mut rem = src_i;
                for d in (0..src_dims.len()).rev() {
                    multi[d] = rem % src_dims[d];
                    rem /= src_dims[d];
                }
                multi[dim] += start;
                let dst_phys = dst_layout.start_offset()
                    + multi.iter().zip(dst_stride).map(|(m, s)| m * s).sum::<usize>();
                $dst_v[dst_phys] = $src_v[src_phys];
            }
        }};
    }
    match (dst, src) {
        (CpuStorage::F32(d), CpuStorage::F32(s)) => slice_set_typed!(d, s),
        (CpuStorage::BF16(d), CpuStorage::BF16(s)) => slice_set_typed!(d, s),
        (CpuStorage::F16(d), CpuStorage::F16(s)) => slice_set_typed!(d, s),
        (CpuStorage::F64(d), CpuStorage::F64(s)) => slice_set_typed!(d, s),
        (CpuStorage::U8(d), CpuStorage::U8(s)) => slice_set_typed!(d, s),
        (CpuStorage::U32(d), CpuStorage::U32(s)) => slice_set_typed!(d, s),
        (CpuStorage::I32(d), CpuStorage::I32(s)) => slice_set_typed!(d, s),
        (CpuStorage::I64(d), CpuStorage::I64(s)) => slice_set_typed!(d, s),
        _ => return Err(Error::Msg("slice_set: dtype mismatch".into()).bt()),
    }
    Ok(())
}

macro_rules! cpu_storage_data {
    ($storage:expr, $variant:ident, $ty:ty) => {
        match $storage {
            CpuStorage::$variant(v) => Ok(v.as_slice()),
            other => Err(Error::UnexpectedDType {
                msg: concat!("expected ", stringify!($variant)),
                expected: <$ty as crate::tensor::WithDType>::DTYPE,
                got: other.dtype(),
            }.bt()),
        }
    };
}
pub(crate) use cpu_storage_data;

// ── CUDA Storage (behind feature gate) ──────────────────────────

#[cfg(feature = "cuda")]
pub use cudarc::driver::{CudaSlice, CudaStream, CudaContext};

#[cfg(feature = "cuda")]
use half::{bf16, f16};

/// cudarc DriverError → our Error
#[cfg(feature = "cuda")]
trait CuResultExt<T> {
    fn ce(self) -> Result<T>;
}
#[cfg(feature = "cuda")]
impl<T> CuResultExt<T> for std::result::Result<T, cudarc::driver::DriverError> {
    fn ce(self) -> Result<T> {
        self.map_err(|e| Error::Msg(format!("CUDA driver: {e}").into()).bt())
    }
}

/// Typed CUDA memory slice. Each variant holds a `CudaSlice<T>`.
#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
impl CudaStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8, Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32, Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16, Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32, Self::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(s) => s.len(), Self::U32(s) => s.len(),
            Self::I32(s) => s.len(), Self::I64(s) => s.len(),
            Self::BF16(s) => s.len(), Self::F16(s) => s.len(),
            Self::F32(s) => s.len(), Self::F64(s) => s.len(),
        }
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaStorageSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaStorageSlice({:?}, len={})", self.dtype(), self.len())
    }
}

/// GPU tensor storage wrapping cudarc types directly.
#[cfg(feature = "cuda")]
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl CudaStorage {
    pub fn device(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    pub fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

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
            dt => return Err(Error::Msg(format!("CudaStorage::zeros: unsupported dtype {dt:?}").into()).bt()),
        };
        Ok(Self { slice, stream: stream.clone() })
    }

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

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaStorage({:?}, gpu:{})", self.slice, self.stream.context().ordinal())
    }
}

// ── Top-level Storage ───────────────────────────────────────────

/// Where tensor data lives — direct enum, zero-overhead dispatch.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(c) => c.dtype(),
            #[cfg(feature = "cuda")]
            Self::Cuda(c) => c.dtype(),
        }
    }

    pub fn as_cpu(&self) -> Result<&CpuStorage> {
        match self {
            Self::Cpu(c) => Ok(c),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::Msg("expected CPU storage, got CUDA".into()).bt()),
        }
    }

    pub fn as_cpu_mut(&mut self) -> Result<&mut CpuStorage> {
        match self {
            Self::Cpu(c) => Ok(c),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::Msg("expected CPU storage, got CUDA".into()).bt()),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda(&self) -> Result<&CudaStorage> {
        match self {
            Self::Cuda(c) => Ok(c),
            _ => Err(Error::Msg("expected CUDA storage, got CPU".into()).bt()),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda_mut(&mut self) -> Result<&mut CudaStorage> {
        match self {
            Self::Cuda(c) => Ok(c),
            _ => Err(Error::Msg("expected CUDA storage, got CPU".into()).bt()),
        }
    }
}
