//! Prelude Tensor — own storage, layout, and compute.
//! The Tensor struct itself holds our own Storage + Layout.

// ── Own types ──────────────────────────────────────────────────────

pub mod error;
pub mod layout;
pub mod shape;
pub mod storage;
pub mod with_dtype;

pub use error::{DeviceLocation, Error, Result};
pub use shape::{D, Dim, Dims, Shape, ShapeWithOneHole};
pub use layout::Layout;
pub use storage::{CpuStorage, CubeCLStorage, DeviceStorage, DeviceStorageTrait, Storage, cpu_extract_vec};
pub use with_dtype::{WithDType, IntDType, FloatDType};

// ── DType ──────────────────────────────────────────────────────────

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 | Self::F8E4M3 => 1,
            Self::I16 | Self::BF16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
    pub fn is_float(&self) -> bool {
        matches!(self, Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::F8E4M3)
    }
    pub fn is_int(&self) -> bool { !self.is_float() }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8", Self::U32 => "u32", Self::I16 => "i16",
            Self::I32 => "i32", Self::I64 => "i64", Self::BF16 => "bf16",
            Self::F16 => "f16", Self::F32 => "f32", Self::F64 => "f64",
            Self::F8E4M3 => "f8e4m3",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str(self.as_str()) }
}

// ── Device ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device { Cpu, Cuda(usize) }

impl Device {
    pub fn is_cuda(&self) -> bool { matches!(self, Self::Cuda(_)) }
    pub fn is_cpu(&self) -> bool { matches!(self, Self::Cpu) }
    pub fn ordinal(&self) -> usize { match self { Self::Cpu => 0, Self::Cuda(n) => *n } }

}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self { Self::Cpu => write!(f, "cpu"), Self::Cuda(n) => write!(f, "cuda:{n}") }
    }
}

pub use crate::bail;

pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

// ── TensorId ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

// ── Tensor ─────────────────────────────────────────────────────────

use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    dtype: DType,
    device: Device,
    id: TensorId,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor[{:?}, {:?}, {:?}]", self.layout.shape(), self.dtype, self.device)
    }
}

/// Device-aware ops: thread-local first, then device-based fallback.
fn ops_for(device: &Device) -> &'static dyn crate::ops::Ops { crate::ops::ops_for(device) }

/// Whether we're using the Device (pure CpuStorage) backend path.
fn use_device_backend() -> bool {
    use std::sync::LazyLock;
    // Default: Device backend (CpuStorage / CudaStorage).
    // Set PRELUDE_TENSOR_BACKEND=cubecl to use CubeCL runtime instead.
    static CHOICE: LazyLock<bool> = LazyLock::new(|| {
        !std::env::var("PRELUDE_TENSOR_BACKEND")
            .map(|v| v == "cubecl")
            .unwrap_or(false)
    });
    *CHOICE
}

impl Tensor {
    /// Device-aware ops dispatch for this tensor.
    fn ops(&self) -> &'static dyn crate::ops::Ops { ops_for(&self.device) }

    /// Create a Tensor from existing storage + layout.
    pub fn from_storage_layout(
        storage: Arc<RwLock<Storage>>,
        layout: Layout,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self { storage, layout, dtype, device, id: TensorId::new() }
    }

    /// Create a legacy CPU tensor (wraps CpuStorage in DeviceStorage).
    /// Used only by the legacy path. CubeCL path uses ops().zeros/from_bytes.
    fn new_cpu_legacy(cpu_storage: CpuStorage, shape: Shape) -> Self {
        let dtype = cpu_storage.dtype();
        let layout = Layout::contiguous(shape);
        Self {
            storage: Arc::new(RwLock::new(Storage::Device(DeviceStorage::from_cpu(cpu_storage)))),
            layout, dtype, device: Device::Cpu, id: TensorId::new(),
        }
    }

    fn view(&self, layout: Layout) -> Self {
        Self {
            storage: self.storage.clone(),
            layout,
            dtype: self.dtype,
            device: self.device,
            id: TensorId::new(),
        }
    }

    // ── Construction ────────────────────────────────────────────────

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        ops_for(device).zeros(&shape, dtype, device)
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        if device.is_cpu() {
            let n = shape.elem_count();
            match dtype {
                DType::F32 => Self::from_vec(vec![1.0f32; n], shape, device),
                DType::BF16 => Self::from_vec(vec![half::bf16::ONE; n], shape, device),
                DType::F16 => Self::from_vec(vec![half::f16::ONE; n], shape, device),
                DType::F64 => Self::from_vec(vec![1.0f64; n], shape, device),
                DType::U8 => Self::from_vec(vec![1u8; n], shape, device),
                DType::U32 => Self::from_vec(vec![1u32; n], shape, device),
                DType::I64 => Self::from_vec(vec![1i64; n], shape, device),
                _ => { let t = Self::zeros(&shape, dtype, device)?; t.affine(0.0, 1.0) }
            }
        } else {
            let t = Self::zeros(&shape, dtype, device)?;
            t.affine(0.0, 1.0)
        }
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape(), self.dtype, self.device())
    }
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape(), self.dtype, self.device())
    }

    pub fn full<S: Into<Shape>>(val: f64, shape: S, device: &Device) -> Result<Self> {
        let shape = shape.into();
        let t = Self::zeros(&shape, DType::F64, device)?;
        let t = t.affine(0.0, val)?; // 0 * x + val = val
        Ok(t)
    }

    pub fn from_vec<T: WithDType>(data: Vec<T>, shape: impl ShapeWithOneHole, device: &Device) -> Result<Self> {
        let shape = shape.into_shape(data.len())?;
        let dtype = T::DTYPE;

        if device.is_cuda() {
            // GPU: create on CPU first, then upload via to_device
            let cpu_tensor = Self::from_vec_cpu(data, shape)?;
            return cpu_tensor.to_device(device);
        }

        let cpu_storage = CpuStorage::from_typed_vec(data);
        let bytes = cpu_storage.as_bytes();

        // Choose storage based on active backend — no bridge between paths.
        let storage = if use_device_backend() {
            Storage::Device(DeviceStorage::from_cpu(cpu_storage))
        } else {
            let cubecl_device: cubecl::cpu::CpuDevice = Default::default();
            let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&cubecl_device);
            let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(bytes.to_vec()));
            Storage::CubeCL(CubeCLStorage::new(handle, dtype, shape.elem_count()))
        };
        let t = Self::from_storage_layout(
            Arc::new(RwLock::new(storage)),
            Layout::contiguous(shape),
            dtype,
            *device,
        );
        Ok(t)
    }

    /// Create a CPU tensor from vec (always Device storage for upload path).
    fn from_vec_cpu<T: WithDType>(data: Vec<T>, shape: Shape) -> Result<Self> {
        let dtype = T::DTYPE;
        let cpu_storage = CpuStorage::from_typed_vec(data);
        let storage = Storage::Device(DeviceStorage::from_cpu(cpu_storage));
        Ok(Self::from_storage_layout(
            Arc::new(RwLock::new(storage)),
            Layout::contiguous(shape),
            dtype,
            Device::Cpu,
        ))
    }

    pub fn from_slice<T: WithDType>(data: &[T], shape: impl Into<Shape>, device: &Device) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape.into(), device)
    }

    pub fn new<T: WithDType, S: AsRef<[T]>>(data: S, device: &Device) -> Result<Self> {
        let data = data.as_ref();
        Self::from_slice(data, data.len(), device)
    }

    pub fn rand<S: Into<Shape>>(_lo: f64, _hi: f64, shape: S, device: &Device) -> Result<Self> {
        // Minimal: use from_vec with random data
        let shape = shape.into();
        let n = shape.elem_count();
        let data: Vec<f32> = (0..n).map(|i| {
            // Simple LCG for reproducible "random" — not crypto, just testing
            let x = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) as f32;
            _lo as f32 + (x.abs() % 1.0) * (_hi - _lo) as f32
        }).collect();
        let t = Self::from_vec(data, shape, &Device::Cpu)?;
        if device.is_cuda() { t.to_device(device) } else { Ok(t) }
    }

    pub fn randn<S: Into<Shape>>(mean: f64, std: f64, shape: S, device: &Device) -> Result<Self> {
        // Minimal: approximate with uniform for testing
        Self::rand(mean - std * 1.73, mean + std * 1.73, shape, device)
    }

    pub fn arange<T: WithDType>(start: T, end: T, device: &Device) -> Result<Self> {
        let start_f64 = WithDType::to_f64(start);
        let end_f64 = WithDType::to_f64(end);
        let n = (end_f64 - start_f64).ceil() as usize;
        let data: Vec<T> = (0..n).map(|i| <T as WithDType>::from_f64(start_f64 + i as f64)).collect();
        let len = data.len();
        Self::from_vec(data, len, device)
    }

    pub fn cat<DD: Dim, T: std::borrow::Borrow<Self>>(tensors: &[T], dim: DD) -> Result<Self> {
        if tensors.is_empty() { bail!("cat: empty tensor list") }
        let first = tensors[0].borrow();
        let d = dim.to_index(first.shape(), "cat")?;
        let refs: Vec<&Self> = tensors.iter().map(|t| t.borrow()).collect();
        ops_for(first.device()).cat(&refs, d)
    }

    pub fn stack<DD: Dim, T: std::borrow::Borrow<Self>>(tensors: &[T], dim: DD) -> Result<Self> {
        if tensors.is_empty() { bail!("stack: empty tensor list") }
        let first = tensors[0].borrow();
        let d = dim.to_index_plus_one(first.shape(), "stack")?;
        let unsqueezed: Vec<Self> = tensors.iter().map(|t| t.borrow().unsqueeze(d)).collect::<Result<_>>()?;
        let refs: Vec<&Self> = unsqueezed.iter().collect();
        ops_for(first.device()).cat(&refs, d)
    }

    // ── Metadata ────────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape { self.layout.shape() }
    pub fn dims(&self) -> &[usize] { self.layout.dims() }
    pub fn dim<DD: Dim>(&self, d: DD) -> Result<usize> { d.to_index(self.shape(), "dim").map(|i| self.dims()[i]) }
    pub fn dims2(&self) -> Result<(usize, usize)> { self.shape().dims2() }
    pub fn dims3(&self) -> Result<(usize, usize, usize)> { self.shape().dims3() }
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> { self.shape().dims4() }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn device(&self) -> &Device { &self.device }
    pub fn rank(&self) -> usize { self.shape().rank() }
    pub fn elem_count(&self) -> usize { self.shape().elem_count() }
    pub fn is_contiguous(&self) -> bool { self.layout.is_contiguous() }
    pub fn our_layout(&self) -> &Layout { &self.layout }
    pub fn stride(&self) -> &[usize] { self.layout.stride() }
    pub fn tensor_id(&self) -> TensorId { self.id }

    // ── View ops (pure metadata — shares storage Arc) ───────────────

    pub fn reshape(&self, s: impl ShapeWithOneHole) -> Result<Self> {
        let target = s.into_shape(self.elem_count())?;
        if !self.is_contiguous() {
            return self.contiguous()?.reshape(target);
        }
        let layout = Layout::contiguous_with_offset(target, self.layout.start_offset());
        Ok(self.view(layout))
    }

    pub fn narrow<DD: Dim>(&self, dim: DD, start: usize, len: usize) -> Result<Self> {
        let d = dim.to_index(self.shape(), "narrow")?;
        Ok(self.view(self.layout.narrow(d, start, len)?))
    }

    pub fn unsqueeze<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        Ok(self.view(self.layout.unsqueeze(d)?))
    }

    pub fn squeeze<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(self.shape(), "squeeze")?;
        Ok(self.view(self.layout.squeeze(d)?))
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        let i1 = d1.to_index(self.shape(), "transpose")?;
        let i2 = d2.to_index(self.shape(), "transpose")?;
        Ok(self.view(self.layout.transpose(i1, i2)?))
    }

    pub fn t(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 { bail!("t() requires rank >= 2, got {rank}") }
        self.transpose(rank - 2, rank - 1)
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() { return Ok(self.clone()) }
        self.ops().contiguous(self)
    }

    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(self.elem_count())
    }

    pub fn flatten<D1: Dim, D2: Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        let i1 = d1.to_index(self.shape(), "flatten")?;
        let i2 = d2.to_index(self.shape(), "flatten")?;
        if i1 > i2 { bail!("flatten: d1 ({i1}) > d2 ({i2})") }
        let dims = self.dims();
        let merged: usize = dims[i1..=i2].iter().product();
        let mut new_dims: Vec<usize> = dims[..i1].to_vec();
        new_dims.push(merged);
        new_dims.extend_from_slice(&dims[i2 + 1..]);
        self.reshape(new_dims.as_slice())
    }

    pub fn chunk<DD: Dim>(&self, n: usize, dim: DD) -> Result<Vec<Self>> {
        let d = dim.to_index(self.shape(), "chunk")?;
        let size = self.dims()[d];
        let chunk_size = (size + n - 1) / n;
        let mut chunks = Vec::with_capacity(n);
        let mut start = 0;
        while start < size {
            let len = (chunk_size).min(size - start);
            chunks.push(self.narrow(d, start, len)?);
            start += len;
        }
        Ok(chunks)
    }

    pub fn get(&self, index: usize) -> Result<Self> {
        self.narrow(0, index, 1)?.squeeze(0)
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        Ok(self.view(self.layout.broadcast_as(shape)?))
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    pub fn broadcast_left<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let target = shape.into();
        let rank_diff = target.rank().saturating_sub(self.rank());
        let mut new_dims = target.dims()[..rank_diff].to_vec();
        new_dims.extend_from_slice(self.dims());
        self.broadcast_as(new_dims.as_slice())
    }

    pub fn pad_with_zeros<DD: Dim>(&self, dim: DD, left: usize, right: usize) -> Result<Self> {
        let d = dim.to_index(self.shape(), "pad_with_zeros")?;
        if left == 0 && right == 0 { return Ok(self.clone()) }
        // Use cat to avoid slice_set (works on all devices including GPU).
        let mut parts: Vec<Self> = Vec::new();
        if left > 0 {
            let mut zdims = self.dims().to_vec();
            zdims[d] = left;
            parts.push(Self::zeros(zdims.as_slice(), self.dtype, self.device())?);
        }
        parts.push(self.clone());
        if right > 0 {
            let mut zdims = self.dims().to_vec();
            zdims[d] = right;
            parts.push(Self::zeros(zdims.as_slice(), self.dtype, self.device())?);
        }
        let refs: Vec<&Self> = parts.iter().collect();
        Self::cat(&refs, d)
    }

    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> {
        if repeats.len() != self.rank() { bail!("repeat: rank mismatch") }
        let mut result = self.clone();
        for (d, &r) in repeats.iter().enumerate() {
            if r == 1 { continue }
            let copies: Vec<Self> = (0..r).map(|_| result.clone()).collect();
            let refs: Vec<&Self> = copies.iter().collect();
            result = result.ops().cat(&refs, d)?;
        }
        Ok(result)
    }

    // ── Conversion + data extraction ────────────────────────────────

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> { self.ops().cast(self, dtype) }
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device == *device { return Ok(self.clone()) }
        // Route through GPU ops if either side is CUDA (GPU ops handle both directions)
        let d = if self.device.is_cuda() { &self.device } else { device };
        ops_for(d).to_device(self, device)
    }

    pub fn to_vec1<T: WithDType>(&self) -> Result<Vec<T>> {
        // GPU tensor: transfer to CPU first, then extract
        if self.device.is_cuda() {
            return self.to_device(&Device::Cpu)?.to_vec1();
        }
        let t = self.contiguous()?.flatten_all()?;
        let guard = t.storage.read().map_err(|_| Error::Msg("lock poisoned".into()))?;
        match &*guard {
            Storage::CubeCL(s) => {
                let device: cubecl::cpu::CpuDevice = Default::default();
                let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);
                let bytes = client.read_one(s.handle.clone())
                    .map_err(|e| Error::Msg(format!("CubeCL readback: {e:?}").into()))?;
                let cpu = CpuStorage::from_raw_bytes(&bytes, s.dtype, s.len)?;
                storage::cpu_extract_vec::<T>(&cpu, &t.layout)
            }
            Storage::Device(dev) => {
                if let Some(cpu) = dev.downcast_ref::<CpuStorage>() {
                    storage::cpu_extract_vec::<T>(cpu, &t.layout)
                } else {
                    Err(Error::Msg("to_vec1: device storage is not CPU — transfer to CPU first".into()).bt())
                }
            }
        }
    }

    pub fn to_vec2<T: WithDType>(&self) -> Result<Vec<Vec<T>>> {
        let (d0, d1) = self.dims2()?;
        let flat: Vec<T> = self.to_vec1()?;
        Ok((0..d0).map(|i| flat[i * d1..(i + 1) * d1].to_vec()).collect())
    }

    pub fn to_vec3<T: WithDType>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        let (d0, d1, d2) = self.dims3()?;
        let flat: Vec<T> = self.to_vec1()?;
        Ok((0..d0).map(|i| {
            (0..d1).map(|j| flat[(i * d1 + j) * d2..(i * d1 + j + 1) * d2].to_vec()).collect()
        }).collect())
    }

    pub fn to_vec0<T: WithDType>(&self) -> Result<T> { self.to_scalar() }

    pub fn to_scalar<T: WithDType>(&self) -> Result<T> {
        if self.elem_count() != 1 { bail!("to_scalar: expected 1 element, got {}", self.elem_count()) }
        Ok(self.to_vec1::<T>()?[0])
    }

    // ── Compute ops (routed through TensorOps — unchanged) ──────────

    pub fn matmul(&self, rhs: &Self) -> Result<Self> { self.ops().matmul(self, rhs) }
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Add) }
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Sub) }
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Mul) }
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Div) }
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> { self.ops().where_cond(self, on_true, on_false) }
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> { self.ops().affine(self, mul, add) }
    pub fn clamp<T: Into<f64>>(&self, min: T, max: T) -> Result<Self> {
        let (min, max) = (min.into(), max.into());
        let z = self.ops().zeros(self.shape(), self.dtype(), self.device())?;
        let min_t = z.affine(0.0, min)?;
        let max_t = z.affine(0.0, max)?;
        self.minimum(&max_t)?.maximum(&min_t)
    }

    pub fn sum<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "sum")?; self.ops().reduce(self, d, false, crate::ops::ReduceOp::Sum) }
    pub fn sum_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "sum_keepdim")?; self.ops().reduce(self, d, true, crate::ops::ReduceOp::Sum) }
    pub fn sum_all(&self) -> Result<Self> { let mut t = self.clone(); for d in (0..self.rank()).rev() { t = t.sum(d)?; } Ok(t) }
    pub fn mean<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "mean")?; self.sum(d)?.affine(1.0 / self.dims()[d] as f64, 0.0) }
    pub fn mean_all(&self) -> Result<Self> { self.sum_all()?.affine(1.0 / self.elem_count() as f64, 0.0) }
    pub fn max<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "max")?; self.ops().reduce(self, d, false, crate::ops::ReduceOp::Max) }
    pub fn max_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "max_keepdim")?; self.ops().reduce(self, d, true, crate::ops::ReduceOp::Max) }
    pub fn min<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "min")?; self.ops().reduce(self, d, false, crate::ops::ReduceOp::Min) }
    pub fn min_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "min_keepdim")?; self.ops().reduce(self, d, true, crate::ops::ReduceOp::Min) }
    pub fn argmax<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "argmax")?; self.ops().reduce(self, d, false, crate::ops::ReduceOp::ArgMax) }
    pub fn argmin<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "argmin")?; self.ops().reduce(self, d, false, crate::ops::ReduceOp::ArgMin) }
    pub fn mean_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> { let d = dim.to_index(self.shape(), "mean_keepdim")?; self.sum_keepdim(d)?.affine(1.0 / self.dims()[d] as f64, 0.0) }

    pub fn exp(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Exp) }
    pub fn log(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Log) }
    pub fn abs(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Abs) }
    pub fn sqrt(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Sqrt) }
    pub fn sqr(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Sqr) }
    pub fn recip(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Recip) }
    pub fn sin(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Sin) }
    pub fn cos(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Cos) }
    pub fn tanh(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Tanh) }
    pub fn relu(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Relu) }
    pub fn gelu(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Gelu) }
    pub fn gelu_erf(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::GeluErf) }
    pub fn silu(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Silu) }
    pub fn neg(&self) -> Result<Self> { self.ops().unary(self, crate::ops::UnaryOp::Neg) }
    pub fn minimum(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Min) }
    pub fn maximum(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Max) }
    pub fn sub(&self, rhs: &Self) -> Result<Self> { self.ops().binary(self, rhs, crate::ops::BinaryOp::Sub) }
    pub fn powf(&self, e: f64) -> Result<Self> { self.log()?.affine(e, 0.0)?.exp() }
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        // ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
        // Equivalent: relu(x) + alpha * (exp(min(x, 0)) - 1)
        // This avoids where_cond which has GPU issues with broadcast layouts.
        let z = Tensor::zeros(self.shape(), self.dtype(), self.device())?;
        let neg_part = self.minimum(&z)?;                // min(x, 0)
        let pos_part = self.maximum(&z)?;                // max(x, 0) = relu(x)
        (&pos_part + &neg_part.exp()?.affine(alpha, -alpha)?)
    }

    // ── Comparison ops ──────────────────────────────────────────────

    pub fn eq_t(&self, rhs: &Self) -> Result<Self> { self.ops().compare(self, rhs, crate::ops::CompareOp::Eq) }
    pub fn ne_t(&self, rhs: &Self) -> Result<Self> { self.ops().compare(self, rhs, crate::ops::CompareOp::Ne) }
    pub fn lt_t(&self, rhs: &Self) -> Result<Self> { self.ops().compare(self, rhs, crate::ops::CompareOp::Lt) }
    pub fn gt_t(&self, rhs: &Self) -> Result<Self> { self.ops().compare(self, rhs, crate::ops::CompareOp::Gt) }
    pub fn ge_t(&self, rhs: &Self) -> Result<Self> { self.ops().compare(self, rhs, crate::ops::CompareOp::Ge) }
    pub fn le_t(&self, rhs: &Self) -> Result<Self> { self.ops().compare(self, rhs, crate::ops::CompareOp::Le) }

    pub fn ge<T: WithDType>(&self, rhs: T) -> Result<Self> { let r = self.affine(0.0, WithDType::to_f64(rhs))?; self.ge_t(&r) }
    pub fn gt<T: WithDType>(&self, rhs: T) -> Result<Self> { let r = self.affine(0.0, WithDType::to_f64(rhs))?; self.gt_t(&r) }
    pub fn le<T: WithDType>(&self, rhs: T) -> Result<Self> { let r = self.affine(0.0, WithDType::to_f64(rhs))?; self.le_t(&r) }
    pub fn lt<T: WithDType>(&self, rhs: T) -> Result<Self> { let r = self.affine(0.0, WithDType::to_f64(rhs))?; self.lt_t(&r) }
    pub fn eq_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> { let r = self.affine(0.0, WithDType::to_f64(rhs))?; self.eq_t(&r) }
    pub fn ne_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> { let r = self.affine(0.0, WithDType::to_f64(rhs))?; self.ne_t(&r) }

    pub fn softmax<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        let max = self.max_keepdim(dim)?;
        let diff = self.broadcast_sub(&max)?;
        let num = diff.exp()?;
        let den = num.sum_keepdim(dim)?;
        num.broadcast_div(&den)
    }

    // ── Conv / misc ops ─────────────────────────────────────────────

    pub fn conv1d(&self, kernel: &Self, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.ops().conv1d(self, kernel, None, stride, padding)
    }
    pub fn conv2d(&self, kernel: &Self, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.ops().conv2d(self, kernel, None, [stride, stride], [padding, padding])
    }
    pub fn conv_transpose1d(&self, kernel: &Self, padding: usize, output_padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.ops().conv_transpose1d(self, kernel, None, stride, padding, output_padding)
    }
    /// Nearest-neighbor 1D interpolation (upsample/downsample).
    /// Input: `[batch, channels, length]`, output: `[batch, channels, target_len]`.
    pub fn interpolate1d(&self, target_len: usize) -> Result<Self> {
        let (b, c, src_len) = self.dims3()?;
        let scale = src_len as f64 / target_len as f64;
        let x = self.contiguous()?;
        let src: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut dst = vec![0f32; b * c * target_len];
        for bi in 0..b {
            for ci in 0..c {
                let src_off = (bi * c + ci) * src_len;
                let dst_off = (bi * c + ci) * target_len;
                for i in 0..target_len {
                    let src_idx = ((i as f64 * scale) as usize).min(src_len - 1);
                    dst[dst_off + i] = src[src_off + src_idx];
                }
            }
        }
        Tensor::from_vec(dst, (b, c, target_len), self.device())?.to_dtype(self.dtype())
    }

    pub fn embedding(&self, ids: &Self) -> Result<Self> { self.ops().index_select(self, ids, 0) }
    pub fn index_select<DD: Dim>(&self, indices: &Self, dim: DD) -> Result<Self> {
        let d = dim.to_index(self.shape(), "index_select")?;
        self.ops().index_select(self, indices, d)
    }
    pub fn gather<DD: Dim>(&self, indices: &Self, dim: DD) -> Result<Self> {
        let d = dim.to_index(self.shape(), "gather")?;
        self.ops().gather(self, indices, d)
    }
    pub fn scatter_add<DD: Dim>(&self, indices: &Self, source: &Self, dim: DD) -> Result<Self> {
        let d = dim.to_index(self.shape(), "scatter_add")?;
        self.ops().scatter_add(self, indices, source, d)
    }
    pub fn index_add<DD: Dim>(&self, ids: &Self, src: &Self, dim: DD) -> Result<Self> {
        let d = dim.to_index(self.shape(), "index_add")?;
        self.ops().index_add(self, ids, src, d)
    }
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Self, Self)> {
        self.ops().sort_last_dim(self, asc)
    }
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Self> {
        let (_, indices) = self.ops().sort_last_dim(self, asc)?;
        Ok(indices)
    }

    pub fn slice_set<DD: Dim>(&self, src: &Self, dim: DD, start: usize) -> Result<()> {
        let d = dim.to_index(self.shape(), "slice_set")?;
        let src = src.contiguous()?;

        let guard = self.storage.read().map_err(|_| Error::Msg("lock poisoned".into()))?;
        match &*guard {
            Storage::CubeCL(_) => {
                drop(guard);
                // CubeCL path: use kernel_slice_assign
                let ops = crate::ops::ops_for(&self.device);
                let src_handle = {
                    let sg = src.storage.read().map_err(|_| Error::Msg("lock poisoned".into()))?;
                    match &*sg {
                        Storage::CubeCL(s) => s.handle.clone(),
                        _ => return Err(Error::Msg("slice_set: mixed storage".into()).bt()),
                    }
                };
                let dst_handle = {
                    let dg = self.storage.read().map_err(|_| Error::Msg("lock poisoned".into()))?;
                    match &*dg {
                        Storage::CubeCL(s) => s.handle.clone(),
                        _ => return Err(Error::Msg("slice_set: mixed storage".into()).bt()),
                    }
                };

                let dtype = self.dtype;
                let n = src.shape().elem_count();
                let cubecl_device: cubecl::cpu::CpuDevice = Default::default();
                let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&cubecl_device);
                let (cc, cd) = {
                    let cube_dim = cubecl::prelude::CubeDim::new(&client, n);
                    (cubecl::calculate_cube_count_elemwise(&client, n, cube_dim), cube_dim)
                };

                let mut offsets = vec![0u32; self.rank()];
                offsets[d] = start as u32;
                let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(offsets));
                let offsets_th = cubecl::std::tensor::TensorHandle::new_contiguous(
                    vec![self.rank()], offsets_h, crate::ops::cubecl_backend::elementwise::dtype_to_storage(DType::U32),
                );

                let dst_th = cubecl::std::tensor::TensorHandle::new_contiguous(
                    self.shape().dims().to_vec(), dst_handle, crate::ops::cubecl_backend::elementwise::dtype_to_storage(dtype),
                );
                let src_th = cubecl::std::tensor::TensorHandle::new_contiguous(
                    src.shape().dims().to_vec(), src_handle, crate::ops::cubecl_backend::elementwise::dtype_to_storage(dtype),
                );

                crate::ops::cubecl_backend::elementwise::kernel_slice_assign::launch::<f32, cubecl::cpu::CpuRuntime>(
                    &client, cc, cd, dst_th.into_arg(), src_th.into_arg(), offsets_th.into_arg(),
                );
                Ok(())
            }
            Storage::Device(_) => {
                drop(guard);
                let src = if src.device.is_cuda() { src.to_device(&Device::Cpu)? } else { src };
                let self_c = self.contiguous()?;
                let mut guard = self_c.storage.write().map_err(|_| Error::Msg("lock poisoned".into()))?;
                let src_guard = src.storage.read().map_err(|_| Error::Msg("lock poisoned".into()))?;
                let dst_cpu = guard.as_cpu_mut()?;
                let src_cpu = src_guard.as_cpu()?;
                storage::cpu_slice_set(dst_cpu, &self_c.layout, src_cpu, &src.layout, d, start)?;
                Ok(())
            }
        }
    }

    pub fn slice_assign<D: std::ops::RangeBounds<usize>>(&self, _ranges: &[D], _src: &Self) -> Result<Self> {
        bail!("slice_assign: not yet implemented")
    }

    pub fn rope_thd(&self, cos: &Self, sin: &Self) -> Result<Self> {
        let (_b_sz, seq_len, _n_head, _n_embd) = self.dims4()?;
        let last_dim = self.dim(D::Minus1)?;
        let xs1 = self.narrow(D::Minus1, 0, last_dim / 2)?;
        let xs2 = self.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
        let neg_xs2 = xs2.neg()?;
        let cat_cos = Tensor::cat(&[cos, cos], D::Minus1)?;
        let cat_sin = Tensor::cat(&[sin, sin], D::Minus1)?;
        let cat_cos = cat_cos.narrow(0, 0, seq_len)?;
        let cat_sin = cat_sin.narrow(0, 0, seq_len)?;
        let cat_cos = cat_cos.unsqueeze(0)?.unsqueeze(2)?;
        let cat_sin = cat_sin.unsqueeze(0)?.unsqueeze(2)?;
        let rotated = Tensor::cat(&[&neg_xs2, &xs1], D::Minus1)?;
        self.broadcast_mul(&cat_cos)?.broadcast_add(&rotated.broadcast_mul(&cat_sin)?)
    }

    pub fn apply<M: Module + ?Sized>(&self, m: &M) -> Result<Self> { m.forward(self) }

    // ── Storage access (our own types) ──────────────────────────────

    pub fn storage_rw(&self) -> &Arc<RwLock<Storage>> { &self.storage }

    /// Zero-copy typed slice view with read lock held.
    ///
    /// Checks dtype, contiguity, and device. Returns a guard that derefs to `&[T]`.
    /// The read lock is held for the guard's lifetime — concurrent reads OK,
    /// concurrent writes blocked.
    pub fn as_slice<T: WithDType>(&self) -> Result<SliceGuard<'_, T>> {
        if T::DTYPE != self.dtype {
            bail!("as_slice: dtype mismatch: expected {:?}, tensor is {:?}", T::DTYPE, self.dtype);
        }
        if !self.is_contiguous() {
            bail!("as_slice: tensor is not contiguous");
        }
        if !self.device.is_cpu() {
            bail!("as_slice: only CPU tensors can be viewed as host slices");
        }
        let n = self.elem_count();
        // SAFETY: dtype, contiguity, and device checked. Pointer valid while
        // guard (and thus the tensor's storage) is alive.
        unsafe {
            let ptr = self.ops().data_ptr(self)? as *const T;
            let slice = std::slice::from_raw_parts(ptr, n);
            let guard = self.storage.read()
                .map_err(|_| Error::Msg("lock poisoned".into()))?;
            Ok(SliceGuard { _guard: guard, slice })
        }
    }

    /// Zero-copy mutable typed slice view with write lock held.
    ///
    /// Checks dtype, contiguity, and device. Returns a guard that derefs to `&mut [T]`.
    /// The write lock is held for the guard's lifetime — all other access blocked.
    pub fn as_mut_slice<T: WithDType>(&self) -> Result<MutSliceGuard<'_, T>> {
        if T::DTYPE != self.dtype {
            bail!("as_mut_slice: dtype mismatch: expected {:?}, tensor is {:?}", T::DTYPE, self.dtype);
        }
        if !self.is_contiguous() {
            bail!("as_mut_slice: tensor is not contiguous");
        }
        if !self.device.is_cpu() {
            bail!("as_mut_slice: only CPU tensors can be mutated as host slices");
        }
        let n = self.elem_count();
        // SAFETY: dtype, contiguity, and device checked. Write lock ensures exclusive access.
        unsafe {
            let ptr = self.ops().data_ptr_mut(self)? as *mut T;
            let slice = std::slice::from_raw_parts_mut(ptr, n);
            let guard = self.storage.write()
                .map_err(|_| Error::Msg("lock poisoned".into()))?;
            Ok(MutSliceGuard { _guard: guard, slice })
        }
    }

    /// Raw data pointer, like PyTorch's `tensor.data_ptr()`.
    ///
    /// Works for any device. CPU: host pointer. CUDA: device pointer.
    /// Prefer `as_slice<T>()` for safe typed access on CPU.
    ///
    /// # Safety
    /// Pointer is valid as long as the tensor is alive.
    pub unsafe fn data_ptr(&self) -> Result<*const u8> {
        self.ops().data_ptr(self)
    }

    /// Mutable raw data pointer. Same contract as `data_ptr`.
    pub unsafe fn data_ptr_mut(&self) -> Result<*mut u8> {
        self.ops().data_ptr_mut(self)
    }
}

// ── Slice guards (hold RwLock for safe zero-copy access) ──────────

/// Read guard: holds a read lock, derefs to `&[T]`.
pub struct SliceGuard<'a, T> {
    _guard: std::sync::RwLockReadGuard<'a, Storage>,
    slice: &'a [T],
}

impl<T> std::ops::Deref for SliceGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.slice }
}

/// Write guard: holds a write lock, derefs to `&mut [T]`.
pub struct MutSliceGuard<'a, T> {
    _guard: std::sync::RwLockWriteGuard<'a, Storage>,
    slice: &'a mut [T],
}

impl<T> std::ops::Deref for MutSliceGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.slice }
}

impl<T> std::ops::DerefMut for MutSliceGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] { self.slice }
}

// ── Operator overloads ─────────────────────────────────────────────

impl std::ops::Add for Tensor {
    type Output = Result<Self>;
    fn add(self, rhs: Self) -> Result<Self> { &self + &rhs }
}
impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> {
        ops_for(&self.device).add_or_fused(self, rhs)
    }
}
impl std::ops::Add<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> { &self + rhs }
}
impl std::ops::Add<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: Tensor) -> Result<Tensor> { self + &rhs }
}
impl std::ops::Add<f64> for Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(&self, 1.0, rhs) }
}
impl std::ops::Add<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(self, 1.0, rhs) }
}
impl std::ops::Add<Tensor> for f64 {
    type Output = Result<Tensor>;
    fn add(self, rhs: Tensor) -> Result<Tensor> { ops_for(&rhs.device).affine(&rhs, 1.0, self) }
}

impl std::ops::Sub for Tensor {
    type Output = Result<Self>;
    fn sub(self, rhs: Self) -> Result<Self> { ops_for(&self.device).binary(&self, &rhs, crate::ops::BinaryOp::Sub) }
}
impl std::ops::Sub for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { ops_for(&self.device).binary(self, rhs, crate::ops::BinaryOp::Sub) }
}
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { ops_for(&self.device).binary(&self, rhs, crate::ops::BinaryOp::Sub) }
}
impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: Tensor) -> Result<Tensor> { ops_for(&self.device).binary(self, &rhs, crate::ops::BinaryOp::Sub) }
}
impl std::ops::Sub<f64> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(&self, 1.0, -rhs) }
}
impl std::ops::Sub<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(self, 1.0, -rhs) }
}

impl std::ops::Mul for Tensor {
    type Output = Result<Self>;
    fn mul(self, rhs: Self) -> Result<Self> { ops_for(&self.device).binary(&self, &rhs, crate::ops::BinaryOp::Mul) }
}
impl std::ops::Mul for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { ops_for(&self.device).binary(self, rhs, crate::ops::BinaryOp::Mul) }
}
impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { ops_for(&self.device).binary(&self, rhs, crate::ops::BinaryOp::Mul) }
}
impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: Tensor) -> Result<Tensor> { ops_for(&self.device).binary(self, &rhs, crate::ops::BinaryOp::Mul) }
}
impl std::ops::Mul<f64> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(&self, rhs, 0.0) }
}
impl std::ops::Mul<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(self, rhs, 0.0) }
}

impl std::ops::Div for Tensor {
    type Output = Result<Self>;
    fn div(self, rhs: Self) -> Result<Self> { ops_for(&self.device).binary(&self, &rhs, crate::ops::BinaryOp::Div) }
}
impl std::ops::Div for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: &Tensor) -> Result<Tensor> { ops_for(&self.device).binary(self, rhs, crate::ops::BinaryOp::Div) }
}
impl std::ops::Div<f64> for Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(&self, 1.0 / rhs, 0.0) }
}
impl std::ops::Div<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { ops_for(&self.device).affine(self, 1.0 / rhs, 0.0) }
}

impl std::ops::Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { ops_for(&self.device).unary(&self, crate::ops::UnaryOp::Neg) }
}
impl std::ops::Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { ops_for(&self.device).unary(self, crate::ops::UnaryOp::Neg) }
}

// ── Display ────────────────────────────────────────────────────────

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor[{:?}, {}, {}]", self.shape(), self.dtype, self.device)
    }
}

pub mod safetensors;
pub mod quantized;
