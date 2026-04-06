//! CubeCL TensorOps backend — operates on `Storage::CubeCL`.
//!
//! Generic over `cubecl::Runtime`. Core uses `CpuRuntime` as default.
//! Device crates instantiate with their runtime (e.g. CUDA).

pub(crate) mod elementwise;
mod reduce;
mod matmul;

use std::sync::{Arc, LazyLock, RwLock};

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::layout::linear::{
    LinearViewLayoutLaunch, LinearViewLaunch, linear_view,
};

use elementwise as K;
use elementwise::dtype_to_storage;
use crate::ops::traits::{
    Ops,
    BinaryOp as BinaryOpEnum, CompareOp as CompareOpEnum, ReduceOp, UnaryOp,
};
use crate::tensor::{CubeCLStorage, DType, Device, Layout, Shape, Storage, Tensor, Result};

pub struct CubeCLTensorOps<R: Runtime> {
    pub client: ComputeClient<R>,
    pub device: R::Device,
}

impl CubeCLTensorOps<cubecl::cpu::CpuRuntime> {
    pub fn new_default_cpu() -> Self {
        let device = Default::default();
        let client = cubecl::cpu::CpuRuntime::client(&device);
        Self { client, device }
    }
}

impl<R: Runtime> CubeCLTensorOps<R> {
    pub fn new(device: &R::Device) -> Self {
        let client = R::client(device);
        Self { client, device: device.clone() }
    }

    fn alloc(&self, size_bytes: usize) -> Handle {
        self.client.empty(size_bytes)
    }

    /// Get CubeCL handle + strides + start_offset from a tensor.
    fn get_handle_strided(&self, tensor: &Tensor) -> Result<(Handle, Vec<usize>, usize)> {
        let guard = tensor.storage_rw().read()
            .map_err(|_| crate::tensor::Error::Msg("lock poisoned".into()))?;
        let strides = tensor.our_layout().stride().to_vec();
        let start_offset = tensor.our_layout().start_offset();
        match &*guard {
            Storage::CubeCL(s) => Ok((s.handle.clone(), strides, start_offset)),
            _ => crate::bail!("CubeCLTensorOps: expected CubeCL storage (got Device — wrong backend path)"),
        }
    }

    fn get_handle(&self, tensor: &Tensor) -> Result<Handle> {
        let (handle, _, start_offset) = self.get_handle_strided(tensor)?;
        // Apply element offset as byte offset
        let byte_offset = start_offset * tensor.dtype().size_in_bytes();
        if byte_offset > 0 {
            Ok(handle.offset_start(byte_offset as u64))
        } else {
            Ok(handle)
        }
    }

    fn make_tensor(&self, handle: Handle, shape: Shape, dtype: DType, device: &Device) -> Tensor {
        let len = shape.elem_count();
        let storage = Storage::CubeCL(CubeCLStorage::new(handle, dtype, len));
        Tensor::from_storage_layout(
            Arc::new(RwLock::new(storage)),
            Layout::contiguous(shape),
            dtype,
            *device,
        )
    }

    /// Create a TensorHandle with explicit strides (for inputs that may be non-contiguous).
    fn tensor_handle(&self, handle: Handle, shape: &Shape, strides: &[usize], dtype: DType)
        -> cubecl::std::tensor::TensorHandle<R>
    {
        cubecl::std::tensor::TensorHandle::new(
            handle, shape.dims().to_vec(), strides.to_vec(), dtype_to_storage(dtype),
        )
    }

    /// Create a contiguous TensorHandle.
    fn tensor_handle_contiguous(&self, handle: Handle, shape: &Shape, dtype: DType)
        -> cubecl::std::tensor::TensorHandle<R>
    {
        cubecl::std::tensor::TensorHandle::new_contiguous(
            shape.dims().to_vec(), handle, dtype_to_storage(dtype),
        )
    }

    /// Apply start_offset to a handle as byte offset.
    fn apply_offset(&self, handle: Handle, start_offset: usize, dtype: DType) -> Handle {
        let byte_offset = start_offset * dtype.size_in_bytes();
        if byte_offset > 0 {
            handle.offset_start(byte_offset as u64)
        } else {
            handle
        }
    }

    /// Create a LinearView from a tensor (respects strides + start_offset).
    fn to_linear_view(&self, tensor: &Tensor) -> Result<LinearViewLaunch<R>> {
        let (handle, strides, start_offset) = self.get_handle_strided(tensor)?;
        let handle = self.apply_offset(handle, start_offset, tensor.dtype());
        let th = self.tensor_handle(handle, tensor.shape(), &strides, tensor.dtype());
        Ok(linear_view(th.binding()))
    }

    /// Create a LinearView broadcast to a reference shape.
    /// Handles rank expansion (prepends 1-dims) and stride zeroing for broadcast dims.
    fn to_linear_view_like(&self, tensor: &Tensor, ref_shape: &Shape) -> Result<LinearViewLaunch<R>> {
        let (handle, mut strides, start_offset) = self.get_handle_strided(tensor)?;
        let handle = self.apply_offset(handle, start_offset, tensor.dtype());
        let mut dims = tensor.shape().dims().to_vec();
        let ref_dims = ref_shape.dims();

        // Rank expansion: prepend 1-dims with stride=0
        while dims.len() < ref_dims.len() {
            dims.insert(0, 1);
            strides.insert(0, 0);
        }
        // Zero strides for broadcast dimensions
        for i in 0..dims.len() {
            if dims[i] == 1 && ref_dims[i] > 1 {
                strides[i] = 0;
            }
        }

        let expanded_shape = Shape::from(dims);
        let th = self.tensor_handle(handle, &expanded_shape, &strides, tensor.dtype());
        let ref_cubecl_shape: cubecl::zspace::Shape = ref_dims.to_vec().into();
        let layout = LinearViewLayoutLaunch::from_reference_shape(ref_cubecl_shape);
        Ok(LinearViewLaunch::new_tensor::<cubecl::std::tensor::layout::linear::LinearViewLayout>(
            th.binding().into_tensor_arg(), layout,
        ))
    }

    /// Create a LinearView for a freshly allocated contiguous output.
    fn output_linear_view(&self, handle: Handle, shape: &Shape, dtype: DType) -> LinearViewLaunch<R> {
        let th = self.tensor_handle_contiguous(handle, shape, dtype);
        linear_view(th.binding())
    }

    fn launch_config(&self, num_elems: usize) -> (CubeCount, CubeDim) {
        let cube_dim = CubeDim::new(&self.client, num_elems);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);
        (cube_count, cube_dim)
    }
}

// ── Default CPU fallback ──────────────────────────────────────────

pub fn cubecl_ops() -> &'static dyn Ops {
    static OPS: LazyLock<CubeCLTensorOps<cubecl::cpu::CpuRuntime>> = LazyLock::new(
        CubeCLTensorOps::<cubecl::cpu::CpuRuntime>::new_default_cpu
    );
    &*OPS
}

// ── DType dispatch macros ─────────────────────────────────────────
//
// For ops that still need type-level dispatch (cast, where, index ops).
// Unary/binary/compare use op family traits + #[define] instead.

macro_rules! dispatch_float {
    ($dtype:expr, $body:expr) => {
        match $dtype {
            DType::F32 => { type F = f32; $body },
            DType::F16 => { type F = half::f16; $body },
            DType::BF16 => { type F = half::bf16; $body },
            DType::F64 => { type F = f32; $body },
            _ => crate::bail!("unsupported dtype for CubeCL float kernel: {:?}", $dtype),
        }
    };
}

macro_rules! dispatch_numeric {
    ($dtype:expr, $body:expr) => {
        match $dtype {
            DType::F32 => { type T = f32; $body },
            DType::F16 => { type T = half::f16; $body },
            DType::BF16 => { type T = half::bf16; $body },
            DType::F64 => { type T = f32; $body },
            DType::U32 => { type T = u32; $body },
            DType::U8 => { type T = u8; $body },
            DType::I32 => { type T = i32; $body },
            DType::I64 => { type T = i64; $body },
            _ => crate::bail!("unsupported dtype for CubeCL numeric kernel: {:?}", $dtype),
        }
    };
}

// ── Broadcast helper ──────────────────────────────────────────────

impl<R: Runtime> CubeCLTensorOps<R> {
    fn broadcast_shapes(a: &Shape, b: &Shape) -> Result<Shape> {
        let a_dims = a.dims();
        let b_dims = b.dims();
        let rank = a_dims.len().max(b_dims.len());
        let mut out = vec![0; rank];
        for i in 0..rank {
            let ad = if i < rank - a_dims.len() { 1 } else { a_dims[i - (rank - a_dims.len())] };
            let bd = if i < rank - b_dims.len() { 1 } else { b_dims[i - (rank - b_dims.len())] };
            if ad != bd && ad != 1 && bd != 1 {
                crate::bail!("broadcast: incompatible shapes {:?} and {:?}", a_dims, b_dims);
            }
            out[i] = ad.max(bd);
        }
        Ok(Shape::from(out))
    }
}

// ── TensorOps ─────────────────────────────────────────────────────

impl<R: Runtime> Ops for CubeCLTensorOps<R> {
    fn default_impl(&self) -> &dyn Ops { self }

    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> {
        let shape = x.shape().clone();
        let dtype = x.dtype();
        let n = shape.elem_count();
        let storage_type = dtype_to_storage(dtype);

        let input = self.to_linear_view(x)?;
        let out_h = self.alloc(n * dtype.size_in_bytes());
        let output = self.output_linear_view(out_h.clone(), &shape, dtype);

        use K::BasicFloatUnaryKind::*;
        let kind = match op {
            UnaryOp::Exp =>     Exp,
            UnaryOp::Log =>     Log,
            UnaryOp::Sqrt =>    Sqrt,
            UnaryOp::Abs =>     Abs,
            UnaryOp::Neg =>     Neg,
            UnaryOp::Recip =>   Recip,
            UnaryOp::Sin =>     Sin,
            UnaryOp::Cos =>     Cos,
            UnaryOp::Tanh =>    Tanh,
            UnaryOp::Ceil =>    Ceil,
            UnaryOp::Floor =>   Floor,
            UnaryOp::Round =>   Round,
            UnaryOp::Sqr =>     Sqr,
            UnaryOp::Relu =>    Relu,
            UnaryOp::Silu =>    Silu,
            UnaryOp::Gelu =>    Gelu,
            UnaryOp::GeluErf => GeluErf,
            UnaryOp::Sign =>    Sign,
        };

        K::launch_unary_float(&self.client, input, output, kind, n, storage_type);

        Ok(self.make_tensor(out_h, shape, dtype, x.device()))
    }

    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOpEnum) -> Result<Tensor> {
        let out_shape = Self::broadcast_shapes(a.shape(), b.shape())?;
        let dtype = a.dtype();
        let n = out_shape.elem_count();
        let storage_type = dtype_to_storage(dtype);

        // LinearView handles broadcasting via reference shape — no contiguous expansion needed
        let lhs = if a.shape() != &out_shape {
            self.to_linear_view_like(a, &out_shape)?
        } else {
            self.to_linear_view(a)?
        };
        let rhs = if b.shape() != &out_shape {
            self.to_linear_view_like(b, &out_shape)?
        } else {
            self.to_linear_view(b)?
        };

        let out_h = self.alloc(n * dtype.size_in_bytes());
        let output = self.output_linear_view(out_h.clone(), &out_shape, dtype);

        match op {
            BinaryOpEnum::Add => K::launch_binop::<R, K::AddOp>(&self.client, lhs, rhs, output, n, storage_type),
            BinaryOpEnum::Sub => K::launch_binop::<R, K::SubOp>(&self.client, lhs, rhs, output, n, storage_type),
            BinaryOpEnum::Mul => K::launch_binop::<R, K::MulOp>(&self.client, lhs, rhs, output, n, storage_type),
            BinaryOpEnum::Div => K::launch_binop::<R, K::DivOp>(&self.client, lhs, rhs, output, n, storage_type),
            BinaryOpEnum::Min => K::launch_binop::<R, K::MinOp>(&self.client, lhs, rhs, output, n, storage_type),
            BinaryOpEnum::Max => K::launch_binop::<R, K::MaxOp>(&self.client, lhs, rhs, output, n, storage_type),
        }

        Ok(self.make_tensor(out_h, out_shape, dtype, a.device()))
    }

    fn compare(&self, a: &Tensor, b: &Tensor, op: CompareOpEnum) -> Result<Tensor> {
        let out_shape = Self::broadcast_shapes(a.shape(), b.shape())?;
        let dtype = a.dtype();
        let n = out_shape.elem_count();
        let in_storage = dtype_to_storage(dtype);
        let out_storage = dtype_to_storage(DType::U32);

        let lhs = if a.shape() != &out_shape {
            self.to_linear_view_like(a, &out_shape)?
        } else {
            self.to_linear_view(a)?
        };
        let rhs = if b.shape() != &out_shape {
            self.to_linear_view_like(b, &out_shape)?
        } else {
            self.to_linear_view(b)?
        };

        let out_h = self.alloc(n * DType::U32.size_in_bytes());
        let output = self.output_linear_view(out_h.clone(), &out_shape, DType::U32);

        match op {
            CompareOpEnum::Eq => K::launch_cmp::<R, K::EqOp>(&self.client, lhs, rhs, output, n, in_storage, out_storage),
            CompareOpEnum::Ne => K::launch_cmp::<R, K::NeOp>(&self.client, lhs, rhs, output, n, in_storage, out_storage),
            CompareOpEnum::Lt => K::launch_cmp::<R, K::LtOp>(&self.client, lhs, rhs, output, n, in_storage, out_storage),
            CompareOpEnum::Gt => K::launch_cmp::<R, K::GtOp>(&self.client, lhs, rhs, output, n, in_storage, out_storage),
            CompareOpEnum::Ge => K::launch_cmp::<R, K::GeOp>(&self.client, lhs, rhs, output, n, in_storage, out_storage),
            CompareOpEnum::Le => K::launch_cmp::<R, K::LeOp>(&self.client, lhs, rhs, output, n, in_storage, out_storage),
        }

        Ok(self.make_tensor(out_h, out_shape, DType::U32, a.device()))
    }

    fn reduce(&self, x: &Tensor, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Tensor> {
        let x = &x.contiguous()?;
        let h = self.get_handle(x)?;
        let (out_h, out_shape, out_dtype) = reduce::launch_reduce(
            &self.client, h, x.shape(), x.dtype(), dim, op,
        ).map_err(|e| crate::tensor::Error::Msg(format!("cubek reduce: {e:?}").into()))?;

        let mut t = self.make_tensor(out_h, out_shape.clone(), out_dtype, x.device());
        if !keepdim {
            let mut dims = out_shape.dims().to_vec();
            dims.remove(dim);
            if dims.is_empty() { dims.push(1); }
            t = t.reshape(dims)?;
        }
        Ok(t)
    }

    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor> {
        if x.dtype() == dtype { return Ok(x.clone()); }
        let h = self.get_handle(x)?;
        let shape = x.shape().clone();
        let n = shape.elem_count();
        let (cc, cd) = self.launch_config(n);

        let inp = self.tensor_handle_contiguous(h, &shape, x.dtype()).into_arg();
        let out_h = self.alloc(n * dtype.size_in_bytes());
        let out = self.tensor_handle_contiguous(out_h.clone(), &shape, dtype).into_arg();

        macro_rules! cast_dispatch {
            ($src:ty, $dst:ty) => {
                K::kernel_cast::launch::<$src, $dst, R>(&self.client, cc, cd, inp, out)
            };
        }

        match (x.dtype(), dtype) {
            (DType::F32, DType::BF16) => cast_dispatch!(f32, half::bf16),
            (DType::F32, DType::F16) => cast_dispatch!(f32, half::f16),
            (DType::F32, DType::F64) => cast_dispatch!(f32, f64),
            (DType::BF16, DType::F32) => cast_dispatch!(half::bf16, f32),
            (DType::BF16, DType::F16) => cast_dispatch!(half::bf16, half::f16),
            (DType::F16, DType::F32) => cast_dispatch!(half::f16, f32),
            (DType::F16, DType::BF16) => cast_dispatch!(half::f16, half::bf16),
            (DType::F64, DType::F32) => cast_dispatch!(f64, f32),
            (DType::U8, DType::U32) => cast_dispatch!(u8, u32),
            (DType::U32, DType::U8) => cast_dispatch!(u32, u8),
            (DType::U8, DType::F32) => cast_dispatch!(u8, f32),
            (DType::U32, DType::F32) => cast_dispatch!(u32, f32),
            (DType::F32, DType::U32) => cast_dispatch!(f32, u32),
            (DType::F32, DType::U8) => cast_dispatch!(f32, u8),
            (DType::I32, DType::F32) => cast_dispatch!(i32, f32),
            (DType::F32, DType::I32) => cast_dispatch!(f32, i32),
            (DType::I64, DType::F32) => cast_dispatch!(i64, f32),
            (DType::F32, DType::I64) => cast_dispatch!(f32, i64),
            (src, dst) => crate::bail!("cast: unsupported dtype pair {src:?} → {dst:?}"),
        }

        Ok(self.make_tensor(out_h, shape, dtype, x.device()))
    }

    fn contiguous(&self, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && x.our_layout().start_offset() == 0 {
            return Ok(x.clone());
        }
        let (h, strides, start_offset) = self.get_handle_strided(x)?;
        let h = self.apply_offset(h, start_offset, x.dtype());
        let shape = x.shape().clone();
        let dtype = x.dtype();
        let storage_type = dtype_to_storage(dtype);

        let input_th = self.tensor_handle(h, &shape, &strides, dtype);
        let output_th = cubecl::std::tensor::into_contiguous(
            &self.client, input_th.binding(), storage_type,
        );

        Ok(self.make_tensor(output_th.handle, shape, dtype, x.device()))
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = &a.contiguous()?;
        let b = &b.contiguous()?;
        let ah = self.get_handle(a)?;
        let bh = self.get_handle(b)?;
        let (out_h, out_shape) = matmul::launch_matmul(
            &self.client, ah, a.shape(), bh, b.shape(), a.dtype(),
        ).map_err(|e| crate::tensor::Error::Msg(format!("cubek matmul: {e:?}").into()))?;

        Ok(self.make_tensor(out_h, out_shape, a.dtype(), a.device()))
    }

    fn to_device(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        if x.device() == device { return Ok(x.clone()); }
        let x = self.contiguous(x)?;
        let h = self.get_handle(&x)?;
        let bytes = self.client.read_one(h)
            .map_err(|e| crate::tensor::Error::Msg(format!("to_device read: {e:?}").into()))?;
        let shape = x.shape().clone();
        let dtype = x.dtype();
        let new_handle = self.client.create(bytes);
        Ok(self.make_tensor(new_handle, shape, dtype, device))
    }

    fn index_select(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let xh = self.get_handle(x)?;
        let ih = self.get_handle(indices)?;
        let dtype = x.dtype();

        let mut out_dims = x.shape().dims().to_vec();
        out_dims[dim] = indices.shape().dims()[0];
        let out_shape = Shape::from(out_dims);
        let n = out_shape.elem_count();
        let (cc, cd) = self.launch_config(n);

        let xa = self.tensor_handle_contiguous(xh, x.shape(), dtype).into_arg();
        let ia = self.tensor_handle_contiguous(ih, indices.shape(), DType::U32).into_arg();
        let out_h = self.alloc(n * dtype.size_in_bytes());
        let oa = self.tensor_handle_contiguous(out_h.clone(), &out_shape, dtype).into_arg();

        dispatch_float!(dtype, {
            K::kernel_index_select::launch::<F, R>(&self.client, cc, cd, xa, ia, oa, dim as u32);
        });

        Ok(self.make_tensor(out_h, out_shape, dtype, x.device()))
    }

    fn gather(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let xh = self.get_handle(x)?;
        let ih = self.get_handle(indices)?;
        let dtype = x.dtype();
        let out_shape = indices.shape().clone();
        let n = out_shape.elem_count();
        let (cc, cd) = self.launch_config(n);

        let xa = self.tensor_handle_contiguous(xh, x.shape(), dtype).into_arg();
        let ia = self.tensor_handle_contiguous(ih, &out_shape, DType::U32).into_arg();
        let out_h = self.alloc(n * dtype.size_in_bytes());
        let oa = self.tensor_handle_contiguous(out_h.clone(), &out_shape, dtype).into_arg();

        dispatch_float!(dtype, K::kernel_gather::launch::<F, R>(&self.client, cc, cd, xa, ia, oa, dim as u32));

        Ok(self.make_tensor(out_h, out_shape, dtype, x.device()))
    }

    fn scatter_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        let xh = self.get_handle(x)?;
        let ih = self.get_handle(indices)?;
        let sh = self.get_handle(src)?;
        let dtype = x.dtype();
        let shape = x.shape().clone();
        let n = src.shape().elem_count();
        let (cc, cd) = self.launch_config(n);

        // Copy x to output first (scatter_add mutates in-place)
        let out_h = {
            let size = shape.elem_count() * dtype.size_in_bytes();
            let out = self.alloc(size);
            let xa = self.tensor_handle_contiguous(xh, &shape, dtype);
            let oa = self.tensor_handle_contiguous(out.clone(), &shape, dtype);
            cubecl::std::tensor::copy_into(
                &self.client, xa.binding(), oa.binding(), dtype_to_storage(dtype),
            );
            out
        };

        let out_a = self.tensor_handle_contiguous(out_h.clone(), &shape, dtype).into_arg();
        let ia = self.tensor_handle_contiguous(ih, indices.shape(), DType::U32).into_arg();
        let sa = self.tensor_handle_contiguous(sh, src.shape(), dtype).into_arg();

        dispatch_float!(dtype, K::kernel_scatter_add::launch::<F, R>(&self.client, cc, cd, out_a, ia, sa, dim as u32));

        Ok(self.make_tensor(out_h, shape, dtype, x.device()))
    }

    fn index_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        let xh = self.get_handle(x)?;
        let ih = self.get_handle(indices)?;
        let sh = self.get_handle(src)?;
        let dtype = x.dtype();
        let shape = x.shape().clone();
        let n = src.shape().elem_count();
        let (cc, cd) = self.launch_config(n);

        // Copy x to output (index_add mutates in-place)
        let out_h = {
            let size = shape.elem_count() * dtype.size_in_bytes();
            let out = self.alloc(size);
            let xa = self.tensor_handle_contiguous(xh, &shape, dtype);
            let oa = self.tensor_handle_contiguous(out.clone(), &shape, dtype);
            cubecl::std::tensor::copy_into(
                &self.client, xa.binding(), oa.binding(), dtype_to_storage(dtype),
            );
            out
        };

        let out_a = self.tensor_handle_contiguous(out_h.clone(), &shape, dtype).into_arg();
        let ia = self.tensor_handle_contiguous(ih, indices.shape(), DType::U32).into_arg();
        let sa = self.tensor_handle_contiguous(sh, src.shape(), dtype).into_arg();

        dispatch_float!(dtype, K::kernel_index_add::launch::<F, R>(&self.client, cc, cd, out_a, ia, sa, dim as u32));

        Ok(self.make_tensor(out_h, shape, dtype, x.device()))
    }

    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        let cond = if cond.dtype() != DType::U32 {
            &cond.to_dtype(DType::U32)?
        } else {
            cond
        };
        let ch = self.get_handle(cond)?;
        let th = self.get_handle(on_true)?;
        let fh = self.get_handle(on_false)?;
        let shape = on_true.shape().clone();
        let dtype = on_true.dtype();
        let n = shape.elem_count();
        let (cc, cd) = self.launch_config(n);

        let ca = self.tensor_handle_contiguous(ch, cond.shape(), DType::U32).into_arg();
        let ta = self.tensor_handle_contiguous(th, &shape, dtype).into_arg();
        let fa = self.tensor_handle_contiguous(fh, &shape, dtype).into_arg();
        let out_h = self.alloc(n * dtype.size_in_bytes());
        let oa = self.tensor_handle_contiguous(out_h.clone(), &shape, dtype).into_arg();

        dispatch_float!(dtype, K::kernel_where::launch::<F, R>(&self.client, cc, cd, ca, ta, fa, oa));

        Ok(self.make_tensor(out_h, shape, dtype, on_true.device()))
    }

    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            crate::bail!("cat: empty tensor list");
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let dtype = tensors[0].dtype();

        // dim=0 with contiguous tensors: copy bytes sequentially
        if dim == 0 && tensors.iter().all(|t| t.is_contiguous()) {
            let mut out_dims = tensors[0].shape().dims().to_vec();
            for t in &tensors[1..] {
                out_dims[0] += t.shape().dims()[0];
            }
            let out_shape = Shape::from(out_dims);
            let total_bytes = out_shape.elem_count() * dtype.size_in_bytes();
            let out_h = self.alloc(total_bytes);

            let mut byte_offset = 0;
            for t in tensors {
                let th = self.get_handle(t)?;
                let t_bytes = t.shape().elem_count() * dtype.size_in_bytes();
                let src_th = self.tensor_handle_contiguous(th, t.shape(), dtype);
                let out_th = cubecl::std::tensor::TensorHandle::new_contiguous(
                    t.shape().dims().to_vec(),
                    out_h.clone().offset_start(byte_offset as u64),
                    dtype_to_storage(dtype),
                );
                cubecl::std::tensor::copy_into(
                    &self.client, src_th.binding(), out_th.binding(), dtype_to_storage(dtype),
                );
                byte_offset += t_bytes;
            }

            return Ok(self.make_tensor(out_h, out_shape, dtype, tensors[0].device()));
        }

        // General case: make inputs contiguous, then slice_assign into output
        let mut out_dims = tensors[0].shape().dims().to_vec();
        for t in &tensors[1..] {
            out_dims[dim] += t.shape().dims()[dim];
        }
        let out_shape = Shape::from(out_dims.clone());
        let out_h = self.alloc(out_shape.elem_count() * dtype.size_in_bytes());

        let mut dim_offset = 0u32;
        for t in tensors {
            // kernel_slice_assign reads value[ABSOLUTE_POS] — requires contiguous input
            let t = &t.contiguous()?;
            let th = self.get_handle(t)?;
            let t_n = t.shape().elem_count();
            let (cc, cd) = self.launch_config(t_n);

            let mut offsets = vec![0u32; out_dims.len()];
            offsets[dim] = dim_offset;
            let offsets_h = self.client.create(cubecl::bytes::Bytes::from_elems(offsets));
            let offsets_a = self.tensor_handle_contiguous(offsets_h, &Shape::from(vec![out_dims.len()]), DType::U32).into_arg();

            let out_a = self.tensor_handle_contiguous(out_h.clone(), &out_shape, dtype).into_arg();
            let src_a = self.tensor_handle_contiguous(th, t.shape(), dtype).into_arg();

            dispatch_float!(dtype, {
                K::kernel_slice_assign::launch::<F, R>(&self.client, cc, cd, out_a, src_a, offsets_a)
            });

            dim_offset += t.shape().dims()[dim] as u32;
        }

        Ok(self.make_tensor(out_h, out_shape, dtype, tensors[0].device()))
    }

    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Result<Tensor> {
        let shape = x.shape().clone();
        let dtype = x.dtype();
        let n = shape.elem_count();
        let storage_type = dtype_to_storage(dtype);

        let input = self.to_linear_view(x)?;
        let out_h = self.alloc(n * dtype.size_in_bytes());
        let output = self.output_linear_view(out_h.clone(), &shape, dtype);

        let mul_h = self.client.create(cubecl::bytes::Bytes::from_elems(vec![mul as f32]));
        let mul_a = self.tensor_handle_contiguous(mul_h, &Shape::from(vec![1]), DType::F32).into_arg();
        let add_h = self.client.create(cubecl::bytes::Bytes::from_elems(vec![add as f32]));
        let add_a = self.tensor_handle_contiguous(add_h, &Shape::from(vec![1]), DType::F32).into_arg();

        let vector_size = 1;
        let working_units = n / vector_size as usize;
        let cube_dim = CubeDim::new(&self.client, working_units);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, working_units, cube_dim);

        unsafe {
            K::kernel_affine::launch_unchecked::<R>(
                &self.client, cube_count, cube_dim,
                cubecl::ir::AddressType::default(),
                vector_size,
                input, output, mul_a, add_a, storage_type,
            );
        }

        Ok(self.make_tensor(out_h, shape, dtype, x.device()))
    }

    fn zeros(&self, shape: &Shape, dtype: DType, device: &Device) -> Result<Tensor> {
        let size_bytes = shape.elem_count() * dtype.size_in_bytes();
        // Must actually zero the buffer — client.empty() returns uninitialized memory
        let handle = self.client.create(cubecl::bytes::Bytes::from_bytes_vec(vec![0u8; size_bytes]));
        Ok(self.make_tensor(handle, shape.clone(), dtype, device))
    }

    fn sort_last_dim(&self, x: &Tensor, asc: bool) -> Result<(Tensor, Tensor)> {
        // CPU fallback sort — same approach as burn-cubecl.
        // Read data to host, sort with Rust's sort_unstable_by, write back.
        let x = &x.contiguous()?;
        let shape = x.shape().clone();
        let dims = shape.dims();
        let last_dim = dims[dims.len() - 1];
        let num_rows: usize = dims[..dims.len() - 1].iter().product();

        // Read all data as f32
        let data: Vec<f32> = x.flatten_all()?.to_vec1()?;

        let mut sorted_vals = Vec::with_capacity(data.len());
        let mut sorted_idxs: Vec<u32> = Vec::with_capacity(data.len());

        for row in 0..num_rows {
            let start = row * last_dim;
            let row_data = &data[start..start + last_dim];

            // Create (index, value) pairs and sort by value
            let mut pairs: Vec<(u32, f32)> = row_data.iter().enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();

            if asc {
                pairs.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            for (idx, val) in &pairs {
                sorted_idxs.push(*idx);
                sorted_vals.push(*val);
            }
        }

        let vals = Tensor::from_vec(sorted_vals, shape.clone(), x.device())?;
        let idxs = Tensor::from_vec(sorted_idxs, shape, x.device())?
            .to_dtype(DType::U32)?;

        Ok((vals, idxs))
    }

    unsafe fn data_ptr(&self, x: &Tensor) -> Result<*const u8> {
        let guard = x.storage_rw().read()
            .map_err(|_| crate::tensor::Error::Msg("lock poisoned".into()))?;
        let byte_offset = x.our_layout().start_offset() * x.dtype().size_in_bytes();
        match &*guard {
            Storage::CubeCL(s) => {
                let bytes = self.client.read_one(s.handle.clone())
                    .map_err(|e| crate::tensor::Error::Msg(
                        format!("data_ptr: read_one failed: {e:?}").into()))?;
                let ptr = bytes.as_ptr();
                Ok(ptr.add(byte_offset))
            }
            _ => crate::bail!("CubeCLTensorOps::data_ptr: expected CubeCL storage"),
        }
    }

    unsafe fn data_ptr_mut(&self, x: &Tensor) -> Result<*mut u8> {
        Ok(self.data_ptr(x)? as *mut u8)
    }
}
