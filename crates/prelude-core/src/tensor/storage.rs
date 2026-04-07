//! Tensor storage — device-based storage with opaque trait objects.
//!
//! - `Storage::Device`: Opaque trait object — device crates provide concrete impls.
//!   prelude-cpu wraps CpuStorage (Vec<T>), prelude-cuda wraps CudaStorage (cudarc).

use super::{DType, Error, Result};

// ── CPU Storage (used by prelude-cpu legacy path via DeviceStorage) ─

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

// ── CpuStorage as DeviceStorageTrait (legacy CPU path) ────────────

impl DeviceStorageTrait for CpuStorage {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn DeviceStorageTrait> { Box::new(self.clone()) }
    fn dtype(&self) -> DType { self.dtype() }
    fn len(&self) -> usize { self.len() }
}

// ── Device Storage (opaque trait object) ──────────────────────────

/// Trait for device-specific storage (CPU legacy, CUDA legacy, etc.).
/// prelude-core sees this as opaque; device crates downcast via Any.
pub trait DeviceStorageTrait: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn clone_box(&self) -> Box<dyn DeviceStorageTrait>;
    fn dtype(&self) -> DType;
    fn len(&self) -> usize;
}

#[derive(Debug)]
pub struct DeviceStorage(pub Box<dyn DeviceStorageTrait>);

impl Clone for DeviceStorage {
    fn clone(&self) -> Self { Self(self.0.clone_box()) }
}

impl DeviceStorage {
    pub fn new(inner: Box<dyn DeviceStorageTrait>) -> Self { Self(inner) }
    pub fn dtype(&self) -> DType { self.0.dtype() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> { self.0.as_any().downcast_ref() }
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> { self.0.as_any_mut().downcast_mut() }

    /// Convenience: create DeviceStorage wrapping CpuStorage (legacy CPU path).
    pub fn from_cpu(cpu: CpuStorage) -> Self { Self(Box::new(cpu)) }
}

// ── Top-level Storage ──────────────────────────────────────────────

/// Where tensor data lives — device-based storage.
#[derive(Debug, Clone)]
pub enum Storage {
    Device(DeviceStorage),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Device(s) => s.dtype(),
        }
    }

    pub fn as_device(&self) -> Result<&DeviceStorage> {
        match self {
            Self::Device(s) => Ok(s),
        }
    }

    pub fn as_device_mut(&mut self) -> Result<&mut DeviceStorage> {
        match self {
            Self::Device(s) => Ok(s),
        }
    }

    /// Convenience: access CpuStorage inside DeviceStorage.
    pub fn as_cpu(&self) -> Result<&CpuStorage> {
        match self {
            Self::Device(s) => s.downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::Msg("expected CPU storage inside Device".into()).bt()),
        }
    }

    pub fn as_cpu_mut(&mut self) -> Result<&mut CpuStorage> {
        match self {
            Self::Device(s) => s.downcast_mut::<CpuStorage>()
                .ok_or_else(|| Error::Msg("expected CPU storage inside Device".into()).bt()),
        }
    }
}
