//! CUDA kernel launch wrappers for TensorOps.
//!
//! Each function launches a PTX kernel for a specific operation category.
//! These provide direct PTX kernel launches.

use std::sync::Arc;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use half::{bf16, f16};
use prelude_core::tensor::{bail, CpuStorage, DType, Layout, Result, Shape};

use crate::device::{
    CuResultExt, CudaStorage, CudaStorageSlice, CudaStorageExt, GpuDType, get_or_load_func, tensor_from_device,
};
use crate::{
    MOD_UNARY, MOD_BINARY, MOD_CAST, MOD_REDUCE, MOD_INDEXING,
    MOD_TERNARY, MOD_AFFINE, MOD_FILL, MOD_SORT,
    PTX_UNARY, PTX_BINARY, PTX_CAST, PTX_REDUCE, PTX_INDEXING,
    PTX_TERNARY, PTX_AFFINE, PTX_FILL, PTX_SORT,
};

// ── Layout info helpers ──────────────────────────────────────────

/// Upload dims + strides to GPU for strided kernel access.
/// Returns None if the layout is contiguous (kernel can use fast path).
fn layout_info(stream: &Arc<CudaStream>, layout: &Layout) -> Result<Option<CudaSlice<usize>>> {
    if layout.is_contiguous() {
        return Ok(None);
    }
    let dims = layout.shape().dims();
    let strides = layout.stride();
    let mut info: Vec<usize> = Vec::with_capacity(dims.len() + strides.len());
    info.extend_from_slice(dims);
    info.extend_from_slice(strides);
    Ok(Some(stream.clone_htod(&info).ce()?))
}

/// Upload dims + lhs_strides + rhs_strides for binary ops.
fn binary_layout_info(
    stream: &Arc<CudaStream>,
    shape: &Shape,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<Option<CudaSlice<usize>>> {
    let lhs_cont = lhs_layout.is_contiguous();
    let rhs_cont = rhs_layout.is_contiguous();
    if lhs_cont && rhs_cont {
        return Ok(None);
    }
    let dims = shape.dims();
    let mut info: Vec<usize> = Vec::with_capacity(dims.len() * 3);
    info.extend_from_slice(dims);
    info.extend_from_slice(lhs_layout.stride());
    info.extend_from_slice(rhs_layout.stride());
    Ok(Some(stream.clone_htod(&info).ce()?))
}

/// Upload dims + 3 strides for ternary ops (cond, true, false).
fn ternary_layout_info(
    stream: &Arc<CudaStream>,
    shape: &Shape,
    cond_layout: &Layout,
    t_layout: &Layout,
    f_layout: &Layout,
) -> Result<Option<CudaSlice<usize>>> {
    let all_cont = cond_layout.is_contiguous()
        && t_layout.is_contiguous()
        && f_layout.is_contiguous();
    if all_cont {
        return Ok(None);
    }
    let dims = shape.dims();
    let mut info: Vec<usize> = Vec::with_capacity(dims.len() * 4);
    info.extend_from_slice(dims);
    info.extend_from_slice(cond_layout.stride());
    info.extend_from_slice(t_layout.stride());
    info.extend_from_slice(f_layout.stride());
    Ok(Some(stream.clone_htod(&info).ce()?))
}

// ── Unary ops ────────────────────────────────────────────────────

/// Launch a unary CUDA kernel: out[i] = op(inp[i]).
pub fn launch_unary(
    cuda: &CudaStorage,
    layout: &Layout,
    kernel_prefix: &str,
) -> Result<CudaStorage> {
    let el = layout.shape().elem_count();
    if el == 0 {
        return CudaStorage::zeros(&cuda.stream, cuda.dtype(), 0);
    }
    let ctx = cuda.device();
    let stream = &cuda.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = layout_info(stream, layout)?;
    let num_dims = layout.shape().dims().len();

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            let src = cuda.as_slice::<$ty>()?;
            let src = src.slice(layout.start_offset()..);
            let kernel_name = format!("{}_{}", kernel_prefix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_UNARY, PTX_UNARY)?;
            let out = unsafe { stream.alloc::<$ty>(el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&src);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    let result = match cuda.dtype() {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        DType::U8 => dispatch!(u8, U8),
        DType::U32 => dispatch!(u32, U32),
        DType::I64 => dispatch!(i64, I64),
        dt => bail!("unary {kernel_prefix}: unsupported dtype {dt:?}"),
    };
    Ok(result)
}

/// Unary op with one scalar parameter (e.g., powf, elu).
pub fn launch_unary1<P: cudarc::driver::DeviceRepr>(
    cuda: &CudaStorage,
    layout: &Layout,
    kernel_prefix: &str,
    param: P,
) -> Result<CudaStorage> {
    let el = layout.shape().elem_count();
    if el == 0 {
        return CudaStorage::zeros(&cuda.stream, cuda.dtype(), 0);
    }
    let ctx = cuda.device();
    let stream = &cuda.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = layout_info(stream, layout)?;
    let num_dims = layout.shape().dims().len();

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            let src = cuda.as_slice::<$ty>()?;
            let src = src.slice(layout.start_offset()..);
            let kernel_name = format!("{}_{}", kernel_prefix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_UNARY, PTX_UNARY)?;
            let out = unsafe { stream.alloc::<$ty>(el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&param);
            builder.arg(&src);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    let result = match cuda.dtype() {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        dt => bail!("unary1 {kernel_prefix}: unsupported dtype {dt:?}"),
    };
    Ok(result)
}

// ── Binary ops ───────────────────────────────────────────────────

/// Launch a binary CUDA kernel: out[i] = op(lhs[i], rhs[i]).
/// Handles broadcasting via layout strides.
pub fn launch_binary(
    lhs: &CudaStorage,
    lhs_layout: &Layout,
    rhs: &CudaStorage,
    rhs_layout: &Layout,
    out_shape: &Shape,
    kernel_prefix: &str,
) -> Result<CudaStorage> {
    let el = out_shape.elem_count();
    if el == 0 {
        return CudaStorage::zeros(&lhs.stream, lhs.dtype(), 0);
    }
    let ctx = lhs.device();
    let stream = &lhs.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = binary_layout_info(stream, out_shape, lhs_layout, rhs_layout)?;
    let num_dims = out_shape.dims().len();

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            let l = lhs.as_slice::<$ty>()?.slice(lhs_layout.start_offset()..);
            let r = rhs.as_slice::<$ty>()?.slice(rhs_layout.start_offset()..);
            let kernel_name = format!("{}_{}", kernel_prefix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_BINARY, PTX_BINARY)?;
            let out = unsafe { stream.alloc::<$ty>(el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&l);
            builder.arg(&r);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    let result = match lhs.dtype() {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        DType::U8 => dispatch!(u8, U8),
        DType::U32 => dispatch!(u32, U32),
        DType::I64 => dispatch!(i64, I64),
        dt => bail!("binary {kernel_prefix}: unsupported dtype {dt:?}"),
    };
    Ok(result)
}

/// Launch a binary comparison kernel (output is u8).
pub fn launch_compare(
    lhs: &CudaStorage,
    lhs_layout: &Layout,
    rhs: &CudaStorage,
    rhs_layout: &Layout,
    out_shape: &Shape,
    kernel_prefix: &str,
) -> Result<CudaStorage> {
    let el = out_shape.elem_count();
    if el == 0 {
        return CudaStorage::zeros(&lhs.stream, DType::U8, 0);
    }
    let ctx = lhs.device();
    let stream = &lhs.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = binary_layout_info(stream, out_shape, lhs_layout, rhs_layout)?;
    let num_dims = out_shape.dims().len();

    macro_rules! dispatch {
        ($ty:ty) => {{
            let l = lhs.as_slice::<$ty>()?.slice(lhs_layout.start_offset()..);
            let r = rhs.as_slice::<$ty>()?.slice(rhs_layout.start_offset()..);
            let kernel_name = format!("{}_{}", kernel_prefix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_BINARY, PTX_BINARY)?;
            let out = unsafe { stream.alloc::<u8>(el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&l);
            builder.arg(&r);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::U8(out), stream: stream.clone() }
        }};
    }

    let result = match lhs.dtype() {
        DType::BF16 => dispatch!(bf16),
        DType::F16 => dispatch!(f16),
        DType::F32 => dispatch!(f32),
        DType::F64 => dispatch!(f64),
        DType::U8 => dispatch!(u8),
        DType::U32 => dispatch!(u32),
        DType::I64 => dispatch!(i64),
        dt => bail!("compare {kernel_prefix}: unsupported dtype {dt:?}"),
    };
    Ok(result)
}

// ── Reduce ops ───────────────────────────────────────────────────

/// Launch a fast reduce kernel (sum/max/min/argmax/argmin) over the last dim.
pub fn launch_reduce(
    cuda: &CudaStorage,
    layout: &Layout,
    kernel_name_prefix: &str,
    reduce_dim: usize,
) -> Result<(CudaStorage, Shape)> {
    let shape = layout.shape();
    let dims = shape.dims();
    let stride = layout.stride();

    // Rearrange dims so that the reduce dim is last
    let mut src_dims: Vec<usize> = Vec::new();
    let mut src_stride: Vec<usize> = Vec::new();
    let mut dst_el = 1usize;
    for (d, (&dim, &s)) in dims.iter().zip(stride.iter()).enumerate() {
        if d != reduce_dim {
            src_dims.push(dim);
            src_stride.push(s);
            dst_el *= dim;
        }
    }
    src_dims.push(dims[reduce_dim]);
    src_stride.push(stride[reduce_dim]);

    let src_el = shape.elem_count();
    let el_to_sum_per_block = src_el / dst_el;

    let ctx = cuda.device();
    let stream = &cuda.stream;

    let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
    let cfg = LaunchConfig {
        grid_dim: (dst_el as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut info: Vec<usize> = Vec::with_capacity(src_dims.len() * 2);
    info.extend_from_slice(&src_dims);
    info.extend_from_slice(&src_stride);
    let ds = stream.clone_htod(&info).ce()?;

    // Output shape: remove the reduced dimension
    let mut out_dims: Vec<usize> = dims.to_vec();
    out_dims.remove(reduce_dim);
    if out_dims.is_empty() {
        out_dims.push(1);
    }
    let out_shape = Shape::from(out_dims);

    let is_argop = kernel_name_prefix.starts_with("fast_arg");

    let n_dims = src_dims.len();

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            let src = cuda.as_slice::<$ty>()?.slice(layout.start_offset()..);
            let kernel_name = format!("{}_{}", kernel_name_prefix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_REDUCE, PTX_REDUCE)?;
            if is_argop {
                let out = unsafe { stream.alloc::<u32>(dst_el) }.ce()?;
                let mut builder = stream.launch_builder(&func);
                builder.arg(&src_el);
                builder.arg(&el_to_sum_per_block);
                builder.arg(&n_dims);
                builder.arg(&ds);
                builder.arg(&src);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.ce()?;
                CudaStorage { slice: CudaStorageSlice::U32(out), stream: stream.clone() }
            } else {
                let out = unsafe { stream.alloc::<$ty>(dst_el) }.ce()?;
                let mut builder = stream.launch_builder(&func);
                builder.arg(&src_el);
                builder.arg(&el_to_sum_per_block);
                builder.arg(&n_dims);
                builder.arg(&ds);
                builder.arg(&src);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.ce()?;
                CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
            }
        }};
    }

    let result = match cuda.dtype() {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        DType::U32 => dispatch!(u32, U32),
        DType::I64 => dispatch!(i64, I64),
        dt => bail!("reduce {kernel_name_prefix}: unsupported dtype {dt:?}"),
    };
    Ok((result, out_shape))
}

// ── Cast/dtype ops ───────────────────────────────────────────────

/// Launch a cast kernel: convert from one dtype to another.
pub fn launch_cast(
    cuda: &CudaStorage,
    layout: &Layout,
    target_dtype: DType,
) -> Result<CudaStorage> {
    let el = layout.shape().elem_count();
    if el == 0 {
        return CudaStorage::zeros(&cuda.stream, target_dtype, 0);
    }
    if cuda.dtype() == target_dtype {
        return launch_unary(cuda, layout, "ucopy");
    }

    let ctx = cuda.device();
    let stream = &cuda.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = layout_info(stream, layout)?;
    let num_dims = layout.shape().dims().len();

    // Kernel name: cast_{src}_{dst}
    let src_suffix = match cuda.dtype() {
        DType::U8 => "u8", DType::U32 => "u32", DType::I64 => "i64",
        DType::BF16 => "bf16", DType::F16 => "f16", DType::F32 => "f32", DType::F64 => "f64",
        dt => bail!("cast: unsupported src dtype {dt:?}"),
    };
    let dst_suffix = match target_dtype {
        DType::U8 => "u8", DType::U32 => "u32", DType::I64 => "i64",
        DType::BF16 => "bf16", DType::F16 => "f16", DType::F32 => "f32", DType::F64 => "f64",
        dt => bail!("cast: unsupported dst dtype {dt:?}"),
    };
    let kernel_name = format!("cast_{src_suffix}_{dst_suffix}");
    let func = get_or_load_func(ctx, &kernel_name, MOD_CAST, PTX_CAST)?;

    macro_rules! do_cast {
        ($src_ty:ty, $dst_ty:ty, $dst_variant:ident) => {{
            let src = cuda.as_slice::<$src_ty>()?.slice(layout.start_offset()..);
            let out = unsafe { stream.alloc::<$dst_ty>(el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&src);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$dst_variant(out), stream: stream.clone() }
        }};
    }

    // Dispatch on (src_dtype, dst_dtype) pair
    macro_rules! dispatch_src {
        ($src_ty:ty) => {
            match target_dtype {
                DType::U8 => do_cast!($src_ty, u8, U8),
                DType::U32 => do_cast!($src_ty, u32, U32),
                DType::I64 => do_cast!($src_ty, i64, I64),
                DType::BF16 => do_cast!($src_ty, bf16, BF16),
                DType::F16 => do_cast!($src_ty, f16, F16),
                DType::F32 => do_cast!($src_ty, f32, F32),
                DType::F64 => do_cast!($src_ty, f64, F64),
                dt => bail!("cast: unsupported dst dtype {dt:?}"),
            }
        };
    }

    let result = match cuda.dtype() {
        DType::U8 => dispatch_src!(u8),
        DType::U32 => dispatch_src!(u32),
        DType::I64 => dispatch_src!(i64),
        DType::BF16 => dispatch_src!(bf16),
        DType::F16 => dispatch_src!(f16),
        DType::F32 => dispatch_src!(f32),
        DType::F64 => dispatch_src!(f64),
        dt => bail!("cast: unsupported src dtype {dt:?}"),
    };
    Ok(result)
}

// ── Contiguous copy ──────────────────────────────────────────────

/// Make a non-contiguous tensor contiguous by copying data.
pub fn launch_contiguous(
    cuda: &CudaStorage,
    layout: &Layout,
) -> Result<CudaStorage> {
    if layout.is_contiguous() && layout.start_offset() == 0 {
        // Already contiguous at offset 0: device-to-device copy
        macro_rules! clone {
            ($ty:ty, $variant:ident) => {{
                let src = cuda.as_slice::<$ty>()?;
                let out = cuda.stream.clone_dtod(src).ce()?;
                CudaStorage { slice: CudaStorageSlice::$variant(out), stream: cuda.stream.clone() }
            }};
        }
        return Ok(match cuda.dtype() {
            DType::U8 => clone!(u8, U8),
            DType::U32 => clone!(u32, U32),
            DType::I64 => clone!(i64, I64),
            DType::BF16 => clone!(bf16, BF16),
            DType::F16 => clone!(f16, F16),
            DType::F32 => clone!(f32, F32),
            DType::F64 => clone!(f64, F64),
            dt => bail!("contiguous: unsupported dtype {dt:?}"),
        });
    }
    // Non-contiguous: use unary copy kernel with stride handling
    launch_unary(cuda, layout, "ucopy")
}

// ── Affine ───────────────────────────────────────────────────────

/// Launch affine kernel: out[i] = inp[i] * mul + add.
pub fn launch_affine(
    cuda: &CudaStorage,
    layout: &Layout,
    mul: f64,
    add: f64,
) -> Result<CudaStorage> {
    let el = layout.shape().elem_count();
    if el == 0 {
        return CudaStorage::zeros(&cuda.stream, cuda.dtype(), 0);
    }
    let ctx = cuda.device();
    let stream = &cuda.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = layout_info(stream, layout)?;
    let num_dims = layout.shape().dims().len();

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            let src = cuda.as_slice::<$ty>()?.slice(layout.start_offset()..);
            let kernel_name = format!("affine_{}", <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_AFFINE, PTX_AFFINE)?;
            let out = unsafe { stream.alloc::<$ty>(el) }.ce()?;
            let mul_val = <$ty as prelude_core::tensor::WithDType>::from_f64(mul);
            let add_val = <$ty as prelude_core::tensor::WithDType>::from_f64(add);
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&src);
            builder.arg(&out);
            builder.arg(&mul_val);
            builder.arg(&add_val);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    let result = match cuda.dtype() {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        dt => bail!("affine: unsupported dtype {dt:?}"),
    };
    Ok(result)
}

// ── Where/Ternary ────────────────────────────────────────────────

/// Launch where_cond: out[i] = cond[i] ? on_true[i] : on_false[i].
pub fn launch_where_cond(
    cond: &CudaStorage, cond_layout: &Layout,
    on_true: &CudaStorage, t_layout: &Layout,
    on_false: &CudaStorage, f_layout: &Layout,
    out_shape: &Shape,
) -> Result<CudaStorage> {
    let el = out_shape.elem_count();
    if el == 0 {
        return CudaStorage::zeros(&on_true.stream, on_true.dtype(), 0);
    }
    let ctx = on_true.device();
    let stream = &on_true.stream;
    let cfg = LaunchConfig::for_num_elems(el as u32);
    let info = ternary_layout_info(stream, out_shape, cond_layout, t_layout, f_layout)?;
    let num_dims = out_shape.dims().len();

    let cond_suffix = match cond.dtype() {
        DType::U8 => "u8", DType::U32 => "u32", DType::I64 => "i64",
        dt => bail!("where_cond: unsupported cond dtype {dt:?}"),
    };

    macro_rules! dispatch {
        ($ty:ty, $variant:ident, $cond_ty:ty) => {{
            let ids = cond.as_slice::<$cond_ty>()?.slice(cond_layout.start_offset()..);
            let t = on_true.as_slice::<$ty>()?.slice(t_layout.start_offset()..);
            let f = on_false.as_slice::<$ty>()?.slice(f_layout.start_offset()..);
            let kernel_name = format!("where_{}_{}", cond_suffix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_TERNARY, PTX_TERNARY)?;
            let out = unsafe { stream.alloc::<$ty>(el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&el);
            builder.arg(&num_dims);
            match &info {
                Some(ds) => builder.arg(ds),
                None => builder.arg(&0u64),
            };
            builder.arg(&ids);
            builder.arg(&t);
            builder.arg(&f);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    macro_rules! dispatch_val {
        ($ty:ty, $variant:ident) => {
            match cond.dtype() {
                DType::U8 => dispatch!($ty, $variant, u8),
                DType::U32 => dispatch!($ty, $variant, u32),
                DType::I64 => dispatch!($ty, $variant, i64),
                _ => bail!("where_cond: unsupported cond dtype"),
            }
        };
    }

    let result = match on_true.dtype() {
        DType::BF16 => dispatch_val!(bf16, BF16),
        DType::F16 => dispatch_val!(f16, F16),
        DType::F32 => dispatch_val!(f32, F32),
        DType::F64 => dispatch_val!(f64, F64),
        DType::U8 => dispatch_val!(u8, U8),
        DType::U32 => dispatch_val!(u32, U32),
        DType::I64 => dispatch_val!(i64, I64),
        dt => bail!("where_cond: unsupported value dtype {dt:?}"),
    };
    Ok(result)
}

// ── Fill (zeros) ─────────────────────────────────────────────────

/// Allocate zeroed GPU memory.
pub fn launch_zeros(
    stream: &Arc<CudaStream>,
    dtype: DType,
    n: usize,
) -> Result<CudaStorage> {
    CudaStorage::zeros(stream, dtype, n)
}

// ── Index select ─────────────────────────────────────────────────

/// out[left, ids[id], right] = inp[left, src_dim, right] for each id
pub fn launch_index_select(
    src: &CudaStorage, src_layout: &Layout,
    ids: &CudaStorage, ids_layout: &Layout,
    dim: usize,
) -> Result<(CudaStorage, Shape)> {
    let src_dims = src_layout.shape().dims();
    let ids_dim_size = ids_layout.shape().elem_count();

    let left_size: usize = src_dims[..dim].iter().product();
    let src_dim_size = src_dims[dim];
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let dst_el = ids_dim_size * left_size * right_size;

    let mut out_dims = src_dims.to_vec();
    out_dims[dim] = ids_dim_size;
    let out_shape = Shape::from(out_dims);

    if dst_el == 0 {
        return Ok((CudaStorage::zeros(&src.stream, src.dtype(), 0)?, out_shape));
    }

    let ctx = src.device();
    let stream = &src.stream;
    let cfg = LaunchConfig::for_num_elems(dst_el as u32);

    // Upload ids layout info (dims + strides for strided ids)
    let ids_dims = ids_layout.shape().dims();
    let ids_stride = ids_layout.stride();
    let mut ids_info: Vec<usize> = Vec::with_capacity(ids_dims.len() * 2);
    ids_info.extend_from_slice(ids_dims);
    ids_info.extend_from_slice(ids_stride);
    let ds = stream.clone_htod(&ids_info).ce()?;
    let ids_num_dims = ids_dims.len();

    // Determine index type suffix
    let ids_suffix = match ids.dtype() {
        DType::U32 => "u32", DType::U8 => "u8", DType::I64 => "i64",
        dt => bail!("index_select: unsupported index dtype {dt:?}"),
    };

    macro_rules! dispatch {
        ($ty:ty, $variant:ident, $ids_ty:ty) => {{
            let src_slice = src.as_slice::<$ty>()?;
            let src_slice = match src_layout.contiguous_offsets() {
                Some((o1, o2)) => src_slice.slice(o1..o2),
                None => bail!("index_select: requires contiguous source"),
            };
            let ids_slice = ids.as_slice::<$ids_ty>()?.slice(ids_layout.start_offset()..);
            let kernel_name = format!("is_{}_{}", ids_suffix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_INDEXING, PTX_INDEXING)?;
            let out = unsafe { stream.alloc::<$ty>(dst_el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&dst_el);
            builder.arg(&ids_num_dims);
            builder.arg(&ds);
            builder.arg(&ids_slice);
            builder.arg(&src_slice);
            builder.arg(&out);
            builder.arg(&left_size);
            builder.arg(&src_dim_size);
            builder.arg(&ids_dim_size);
            builder.arg(&right_size);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    macro_rules! dispatch_ids {
        ($ids_ty:ty) => {
            match src.dtype() {
                DType::BF16 => dispatch!(bf16, BF16, $ids_ty),
                DType::F16 => dispatch!(f16, F16, $ids_ty),
                DType::F32 => dispatch!(f32, F32, $ids_ty),
                DType::F64 => dispatch!(f64, F64, $ids_ty),
                DType::U8 => dispatch!(u8, U8, $ids_ty),
                DType::U32 => dispatch!(u32, U32, $ids_ty),
                DType::I64 => dispatch!(i64, I64, $ids_ty),
                dt => bail!("index_select: unsupported src dtype {dt:?}"),
            }
        };
    }

    let result = match ids.dtype() {
        DType::U32 => dispatch_ids!(u32),
        DType::U8 => dispatch_ids!(u8),
        DType::I64 => dispatch_ids!(i64),
        dt => bail!("index_select: unsupported ids dtype {dt:?}"),
    };
    Ok((result, out_shape))
}

// ── Gather ───────────────────────────────────────────────────────

pub fn launch_gather(
    src: &CudaStorage, src_layout: &Layout,
    ids: &CudaStorage, ids_layout: &Layout,
    dim: usize,
) -> Result<(CudaStorage, Shape)> {
    let src_dims = src_layout.shape().dims();
    let ids_dims = ids_layout.shape().dims();
    let ids_el = ids_layout.shape().elem_count();

    let left_size: usize = ids_dims[..dim].iter().product();
    let src_dim_size = src_dims[dim];
    let ids_dim_size = ids_dims[dim];
    let right_size: usize = ids_dims[dim + 1..].iter().product();

    let out_shape = ids_layout.shape().clone();
    if ids_el == 0 {
        return Ok((CudaStorage::zeros(&src.stream, src.dtype(), 0)?, out_shape));
    }

    let ctx = src.device();
    let stream = &src.stream;
    let cfg = LaunchConfig::for_num_elems(ids_el as u32);

    let ids_suffix = match ids.dtype() {
        DType::U32 => "u32", DType::U8 => "u8", DType::I64 => "i64",
        dt => bail!("gather: unsupported index dtype {dt:?}"),
    };

    macro_rules! dispatch {
        ($ty:ty, $variant:ident, $ids_ty:ty) => {{
            let src_slice = src.as_slice::<$ty>()?.slice(src_layout.start_offset()..);
            let ids_slice = ids.as_slice::<$ids_ty>()?.slice(ids_layout.start_offset()..);
            let kernel_name = format!("gather_{}_{}", ids_suffix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_INDEXING, PTX_INDEXING)?;
            let out = unsafe { stream.alloc::<$ty>(ids_el) }.ce()?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&ids_el);
            builder.arg(&ids_slice);
            builder.arg(&src_slice);
            builder.arg(&out);
            builder.arg(&left_size);
            builder.arg(&src_dim_size);
            builder.arg(&ids_dim_size);
            builder.arg(&right_size);
            unsafe { builder.launch(cfg) }.ce()?;
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    macro_rules! dispatch_ids {
        ($ids_ty:ty) => {
            match src.dtype() {
                DType::BF16 => dispatch!(bf16, BF16, $ids_ty),
                DType::F16 => dispatch!(f16, F16, $ids_ty),
                DType::F32 => dispatch!(f32, F32, $ids_ty),
                DType::F64 => dispatch!(f64, F64, $ids_ty),
                DType::U8 => dispatch!(u8, U8, $ids_ty),
                DType::U32 => dispatch!(u32, U32, $ids_ty),
                DType::I64 => dispatch!(i64, I64, $ids_ty),
                dt => bail!("gather: unsupported src dtype {dt:?}"),
            }
        };
    }

    let result = match ids.dtype() {
        DType::U32 => dispatch_ids!(u32),
        DType::U8 => dispatch_ids!(u8),
        DType::I64 => dispatch_ids!(i64),
        dt => bail!("gather: unsupported ids dtype {dt:?}"),
    };
    Ok((result, out_shape))
}

// ── Scatter add ──────────────────────────────────────────────────

pub fn launch_scatter_add(
    dst: &CudaStorage, dst_layout: &Layout,
    ids: &CudaStorage, ids_layout: &Layout,
    src: &CudaStorage, src_layout: &Layout,
    dim: usize,
) -> Result<CudaStorage> {
    let dst_dims = dst_layout.shape().dims();
    let src_dims = src_layout.shape().dims();

    let left_size: usize = src_dims[..dim].iter().product();
    let src_dim_size = src_dims[dim];
    let dst_dim_size = dst_dims[dim];
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let src_el = src_layout.shape().elem_count();

    let stream = &dst.stream;
    let ctx = dst.device();

    let ids_suffix = match ids.dtype() {
        DType::U32 => "u32", DType::U8 => "u8", DType::I64 => "i64",
        dt => bail!("scatter_add: unsupported index dtype {dt:?}"),
    };

    // First, copy dst to output (scatter_add accumulates into output)
    let out_storage = launch_contiguous(dst, dst_layout)?;

    let cfg = LaunchConfig::for_num_elems(src_el as u32);

    macro_rules! dispatch {
        ($ty:ty, $variant:ident, $ids_ty:ty) => {{
            let ids_slice = ids.as_slice::<$ids_ty>()?.slice(ids_layout.start_offset()..);
            let src_slice = src.as_slice::<$ty>()?.slice(src_layout.start_offset()..);
            // out_storage is contiguous, get mutable reference
            let out_slice = <$ty as GpuDType>::as_cuda_slice(&out_storage.slice)?;
            let kernel_name = format!("sa_{}_{}", ids_suffix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_INDEXING, PTX_INDEXING)?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&ids_slice);
            builder.arg(&src_slice);
            builder.arg(out_slice);
            builder.arg(&left_size);
            builder.arg(&src_dim_size);
            builder.arg(&dst_dim_size);
            builder.arg(&right_size);
            unsafe { builder.launch(cfg) }.ce()?;
        }};
    }

    macro_rules! dispatch_ids {
        ($ids_ty:ty) => {
            match dst.dtype() {
                DType::BF16 => dispatch!(bf16, BF16, $ids_ty),
                DType::F16 => dispatch!(f16, F16, $ids_ty),
                DType::F32 => dispatch!(f32, F32, $ids_ty),
                DType::F64 => dispatch!(f64, F64, $ids_ty),
                dt => bail!("scatter_add: unsupported dtype {dt:?}"),
            }
        };
    }

    match ids.dtype() {
        DType::U32 => dispatch_ids!(u32),
        DType::U8 => dispatch_ids!(u8),
        DType::I64 => dispatch_ids!(i64),
        dt => bail!("scatter_add: unsupported ids dtype {dt:?}"),
    }

    Ok(out_storage)
}

// ── Index add ────────────────────────────────────────────────────

pub fn launch_index_add(
    dst: &CudaStorage, dst_layout: &Layout,
    ids: &CudaStorage, ids_layout: &Layout,
    src: &CudaStorage, src_layout: &Layout,
    dim: usize,
) -> Result<CudaStorage> {
    let dst_dims = dst_layout.shape().dims();
    let src_dims = src_layout.shape().dims();

    let left_size: usize = src_dims[..dim].iter().product();
    let src_dim_size = src_dims[dim];
    let dst_dim_size = dst_dims[dim];
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let ids_dim_size = ids_layout.shape().elem_count();

    let stream = &dst.stream;
    let ctx = dst.device();

    let ids_suffix = match ids.dtype() {
        DType::U32 => "u32", DType::U8 => "u8", DType::I64 => "i64",
        dt => bail!("index_add: unsupported index dtype {dt:?}"),
    };

    // Copy dst to output first
    let out_storage = launch_contiguous(dst, dst_layout)?;

    let cfg = LaunchConfig::for_num_elems(ids_dim_size as u32);

    macro_rules! dispatch {
        ($ty:ty, $variant:ident, $ids_ty:ty) => {{
            let ids_slice = ids.as_slice::<$ids_ty>()?.slice(ids_layout.start_offset()..);
            let src_slice = src.as_slice::<$ty>()?.slice(src_layout.start_offset()..);
            let out_slice = <$ty as GpuDType>::as_cuda_slice(&out_storage.slice)?;
            let kernel_name = format!("ia_{}_{}", ids_suffix, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_INDEXING, PTX_INDEXING)?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&ids_slice);
            builder.arg(&ids_dim_size);
            builder.arg(&src_slice);
            builder.arg(out_slice);
            builder.arg(&left_size);
            builder.arg(&src_dim_size);
            builder.arg(&dst_dim_size);
            builder.arg(&right_size);
            unsafe { builder.launch(cfg) }.ce()?;
        }};
    }

    macro_rules! dispatch_ids {
        ($ids_ty:ty) => {
            match dst.dtype() {
                DType::BF16 => dispatch!(bf16, BF16, $ids_ty),
                DType::F16 => dispatch!(f16, F16, $ids_ty),
                DType::F32 => dispatch!(f32, F32, $ids_ty),
                DType::F64 => dispatch!(f64, F64, $ids_ty),
                dt => bail!("index_add: unsupported dtype {dt:?}"),
            }
        };
    }

    match ids.dtype() {
        DType::U32 => dispatch_ids!(u32),
        DType::U8 => dispatch_ids!(u8),
        DType::I64 => dispatch_ids!(i64),
        dt => bail!("index_add: unsupported ids dtype {dt:?}"),
    }

    Ok(out_storage)
}

// ── Sort ─────────────────────────────────────────────────────────

/// Sort last dimension, returning (sorted_values, indices).
pub fn launch_sort_last_dim(
    cuda: &CudaStorage,
    layout: &Layout,
    asc: bool,
) -> Result<(CudaStorage, CudaStorage)> {
    let shape = layout.shape();
    let dims = shape.dims();
    let ncols = *dims.last().unwrap();
    let nrows = shape.elem_count() / ncols;
    let ncols_pad = ncols.next_power_of_two();

    let ctx = cuda.device();
    let stream = &cuda.stream;

    // Sort kernel: one block per row, ncols_pad threads (min 32)
    let block_size = ncols_pad.max(32) as u32;
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let direction = if asc { "asc" } else { "desc" };

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            let src = cuda.as_slice::<$ty>()?.slice(layout.start_offset()..);
            let kernel_name = format!("asort_{}_{}", direction, <$ty as GpuDType>::kernel_suffix());
            let func = get_or_load_func(ctx, &kernel_name, MOD_SORT, PTX_SORT)?;
            let idx_out = unsafe { stream.alloc::<u32>(nrows * ncols) }.ce()?;
            let ncols_i = ncols as i32;
            let ncols_pad_i = ncols_pad as i32;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&src);
            builder.arg(&idx_out);
            builder.arg(&ncols_i);
            builder.arg(&ncols_pad_i);
            unsafe { builder.launch(cfg) }.ce()?;

            // The sort kernel only outputs indices — reconstruct sorted values via gather
            // Actually, the sort kernel outputs indices. We need to gather to get values.
            // For now, gather on CPU (TODO: GPU gather)
            let idx_storage = CudaStorage {
                slice: CudaStorageSlice::U32(idx_out),
                stream: stream.clone(),
            };
            // Gather values using our index_select on last dim
            // Actually this is more complex. Let me just return indices and
            // let the caller do the gather. But TensorOps::sort_last_dim
            // returns (values, indices).
            // Simple approach: download indices, gather on CPU, upload.
            // Better approach: use a simple gather kernel.
            // For efficiency, just download all and sort on CPU for now.
            // Actually the sort kernel is bitonic sort in shared memory —
            // it takes input values and outputs sorted indices.
            // We need to also produce sorted values.
            // Let's just do a copy kernel using the indices.
            let val_out = unsafe { stream.alloc::<$ty>(nrows * ncols) }.ce()?;
            // Use a simple gather: val_out[row, j] = src[row, idx_out[row, j]]
            // This is just index_select on last dim per row.
            // For now, do it via downloading indices and values.
            let idx_vec: Vec<u32> = stream.clone_dtoh(idx_storage.as_slice::<u32>()?).ce()?;
            let src_vec: Vec<$ty> = stream.clone_dtoh(cuda.as_slice::<$ty>()?).ce()?;
            let start = layout.start_offset();
            let mut sorted: Vec<$ty> = Vec::with_capacity(nrows * ncols);
            for row in 0..nrows {
                for j in 0..ncols {
                    let idx = idx_vec[row * ncols + j] as usize;
                    sorted.push(src_vec[start + row * ncols + idx]);
                }
            }
            let val_slice = stream.clone_htod(&sorted).ce()?;
            let val_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(val_slice),
                stream: stream.clone(),
            };
            (val_storage, idx_storage)
        }};
    }

    let (vals, idxs) = match cuda.dtype() {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        dt => bail!("sort: unsupported dtype {dt:?}"),
    };
    Ok((vals, idxs))
}

// ── Cat (concatenation) ──────────────────────────────────────────

/// Concatenate tensors along a dimension by copying into a single output buffer.
pub fn launch_cat(
    storages: &[(&CudaStorage, &Layout)],
    dim: usize,
    out_shape: &Shape,
) -> Result<CudaStorage> {
    let dtype = storages[0].0.dtype();
    let stream = &storages[0].0.stream;
    let el = out_shape.elem_count();

    if el == 0 {
        return CudaStorage::zeros(stream, dtype, 0);
    }

    let out_dims = out_shape.dims();
    // For cat on dim d: output is contiguous, each input contributes a slice
    // left_size = product of dims before d
    // right_size = product of dims after d
    let left_size: usize = out_dims[..dim].iter().product();
    let right_size: usize = out_dims[dim + 1..].iter().product();

    // Allocate output
    let ctx = storages[0].0.device();

    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            // Simple approach: download all inputs, concatenate on CPU, upload result.
            // This is correct and handles all stride/offset cases.
            // TODO: optimize with GPU copy kernels for contiguous inputs.
            let mut all_data: Vec<$ty> = Vec::with_capacity(el);
            for &(storage, layout) in storages {
                let src_cpu = storage.to_cpu(layout)?;
                match src_cpu {
                    CpuStorage::$variant(v) => all_data.extend_from_slice(&v),
                    _ => bail!("cat: dtype mismatch"),
                }
            }
            // Rearrange for non-0 dim concatenation
            if dim > 0 {
                let mut result: Vec<$ty> = vec![<$ty as prelude_core::tensor::WithDType>::from_f64(0.0); el];
                let mut src_offset = 0;
                let mut dst_col_offset = 0;
                for &(_, layout) in storages {
                    let src_dims = layout.shape().dims();
                    let src_dim_size = src_dims[dim];
                    let src_right = src_dim_size * right_size;
                    let dst_right = out_dims[dim] * right_size;
                    let src_data = &all_data[src_offset..src_offset + layout.shape().elem_count()];
                    for row in 0..left_size {
                        let src_start = row * src_right;
                        let dst_start = row * dst_right + dst_col_offset;
                        result[dst_start..dst_start + src_right]
                            .copy_from_slice(&src_data[src_start..src_start + src_right]);
                    }
                    dst_col_offset += src_right;
                    src_offset += layout.shape().elem_count();
                }
                let out = stream.clone_htod(&result).ce()?;
                CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
            } else {
                // dim == 0: simple concatenation, data is already in order
                let out = stream.clone_htod(&all_data).ce()?;
                CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
            }
        }};
    }

    let result = match dtype {
        DType::BF16 => dispatch!(bf16, BF16),
        DType::F16 => dispatch!(f16, F16),
        DType::F32 => dispatch!(f32, F32),
        DType::F64 => dispatch!(f64, F64),
        DType::U8 => dispatch!(u8, U8),
        DType::U32 => dispatch!(u32, U32),
        DType::I64 => dispatch!(i64, I64),
        dt => bail!("cat: unsupported dtype {dt:?}"),
    };
    Ok(result)
}

// ── Matmul ───────────────────────────────────────────────────────

/// Matrix multiply via CUTLASS/DeepGEMM dispatch.
/// Both inputs must be contiguous CUDA tensors.
/// Matmul dispatch via CUTLASS/DeepGEMM.
pub fn launch_matmul(
    a: &CudaStorage, a_layout: &Layout,
    b: &CudaStorage, b_layout: &Layout,
) -> Result<(CudaStorage, Shape)> {
    use std::ffi::c_void;

    let a_dims = a_layout.shape().dims();
    let b_dims = b_layout.shape().dims();

    // Standard matmul: [..., M, K] x [..., K, N] → [..., M, N]
    let k = *a_dims.last().unwrap();
    let m: usize = a_layout.shape().elem_count() / k;
    let n = *b_dims.last().unwrap();
    let k2 = b_dims[b_dims.len().saturating_sub(2).max(0)];
    if k != k2 {
        bail!("matmul: inner dimensions mismatch {k} vs {k2}");
    }
    let dtype_code = match a.dtype() {
        DType::BF16 => 0u32,
        DType::F16 => 1u32,
        DType::F32 => 2u32,
        dt => bail!("matmul: unsupported dtype {dt:?}"),
    };

    let stream = &a.stream;

    macro_rules! do_matmul {
        ($ty:ty, $variant:ident) => {{
            let a_slice = a.as_slice::<$ty>()?;
            let b_slice = b.as_slice::<$ty>()?;
            let out = unsafe { stream.alloc::<$ty>(m * n) }.ce()?;

            let raw_stream = unsafe { stream.cu_stream() } as *const c_void;
            // Extract raw pointers (borrows released in block scope)
            let a_raw = { let (p, _g) = a_slice.device_ptr(stream); p };
            let b_raw = { let (p, _g) = b_slice.device_ptr(stream); p };
            let out_raw = { let (p, _g) = out.device_ptr(stream); p };

            let ret = unsafe {
                crate::ops::gemm::gemm_dispatch_impl(
                    b_raw as *const c_void,   // A = B[N,K] row-major
                    a_raw as *const c_void,   // B = A[M,K] row-major
                    out_raw as *mut c_void,   // D = output [M,N]
                    n as i32,    // m = N
                    m as i32,    // n = M
                    k as i32,
                    1,
                    k as i32,    // lda = K (row stride of B[N,K])
                    k as i32,    // ldb = K (row stride of A[M,K])
                    n as i32,    // ldd = N
                    0, 0, 0,
                    true, false, // TN
                    dtype_code,
                    raw_stream,
                )
            };
            if ret != 0 {
                bail!("GPU matmul failed (code {ret}) M={m} N={n} K={k}");
            }
            CudaStorage { slice: CudaStorageSlice::$variant(out), stream: stream.clone() }
        }};
    }

    let out_shape = Shape::from(vec![m, n]);
    let result = match a.dtype() {
        DType::BF16 => do_matmul!(bf16, BF16),
        DType::F16 => do_matmul!(f16, F16),
        DType::F32 => do_matmul!(f32, F32),
        dt => bail!("matmul: unsupported dtype {dt:?}"),
    };
    Ok((result, out_shape))
}
