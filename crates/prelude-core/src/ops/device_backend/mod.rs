//! Pure CPU TensorOps implementation over `Storage::Device(CpuStorage)`.
//!
//! `DeviceTensorOps` is a correctness reference — no SIMD, no parallelism.
//! Performance-critical paths are handled by prelude-cpu overrides.
//!
//! All outputs are `Storage::Device(DeviceStorage::from_cpu(...))`.

use std::sync::Arc;

use half::{bf16, f16};

use crate::ops::traits::{BinaryOp, CompareOp, ReduceOp, Ops, UnaryOp};
use crate::tensor::{
    CpuStorage, DType, Device, DeviceStorage, Layout, Result, Shape, Storage, Tensor,
    WithDType,
    error::Error,
};

// ── Helper: build an output tensor from CpuStorage ────────────────

fn make_tensor(cpu: CpuStorage, shape: Shape, device: &Device) -> Tensor {
    let dtype = cpu.dtype();
    Tensor::from_storage_layout(
        Arc::new(Storage::Device(DeviceStorage::from_cpu(cpu))),
        Layout::contiguous(shape),
        dtype,
        *device,
    )
}

// ── Inline strided binary-map with broadcasting ───────────────────

/// Map over two layouts that may differ in shape (broadcast).
/// Both `lhs_layout` and `rhs_layout` must have been broadcast-expanded to `out_shape`
/// before calling this.
fn binary_map_broadcast<T, U, F>(
    lhs: &[T],
    lhs_l: &Layout,
    rhs: &[T],
    rhs_l: &Layout,
    mut f: F,
) -> Vec<U>
where
    T: Copy,
    U: Copy,
    F: FnMut(T, T) -> U,
{
    match (lhs_l.contiguous_offsets(), rhs_l.contiguous_offsets()) {
        (Some((l1, l2)), Some((r1, r2))) => lhs[l1..l2]
            .iter()
            .zip(rhs[r1..r2].iter())
            .map(|(&l, &r)| f(l, r))
            .collect(),
        _ => lhs_l
            .strided_index()
            .zip(rhs_l.strided_index())
            .map(|(li, ri)| f(lhs[li], rhs[ri]))
            .collect(),
    }
}

// ── Helper: broadcast two layouts to a common shape ───────────────

fn broadcast_layouts(
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<(Shape, Layout, Layout)> {
    let out_shape = lhs_layout
        .shape()
        .broadcast_shape_binary_op(rhs_layout.shape(), "binary")?;
    let lhs_bc = lhs_layout.broadcast_as(&out_shape)?;
    let rhs_bc = rhs_layout.broadcast_as(&out_shape)?;
    Ok((out_shape, lhs_bc, rhs_bc))
}

// ── unary_map: apply element-wise function to a CpuStorage ────────

/// Apply a unary op via f64 round-trip (works for all WithDType types).
fn unary_via_f64<T: WithDType, F: Fn(f64) -> f64>(
    data: &[T],
    layout: &Layout,
    f: F,
) -> Vec<T> {
    match layout.contiguous_offsets() {
        Some((s, e)) => data[s..e]
            .iter()
            .map(|&v| T::from_f64(f(v.to_f64())))
            .collect(),
        None => layout
            .strided_index()
            .map(|i| T::from_f64(f(data[i].to_f64())))
            .collect(),
    }
}

fn erf_f64(x: f64) -> f64 {
    libm::erf(x)
}

// ── gelu helpers ────────────────────────────────────────────────────

fn gelu_tanh(x: f64) -> f64 {
    // tanh-approximate gelu (PyTorch's default gelu)
    let c = 0.044715f64;
    let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt2pi * (x + c * x.powi(3))).tanh())
}

fn gelu_erf(x: f64) -> f64 {
    // exact gelu via erf
    0.5 * x * (1.0 + erf_f64(x / std::f64::consts::SQRT_2))
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

// ── apply a UnaryOp to a CpuStorage ─────────────────────────────────

fn apply_unary_op(cpu: &CpuStorage, layout: &Layout, op: UnaryOp) -> Result<CpuStorage> {
    macro_rules! via_f64 {
        ($f:expr) => {{
            Ok(match cpu {
                CpuStorage::U8(v)   => CpuStorage::U8(unary_via_f64(v, layout, $f)),
                CpuStorage::U32(v)  => CpuStorage::U32(unary_via_f64(v, layout, $f)),
                CpuStorage::I32(v)  => CpuStorage::I32(unary_via_f64(v, layout, $f)),
                CpuStorage::I64(v)  => CpuStorage::I64(unary_via_f64(v, layout, $f)),
                CpuStorage::BF16(v) => CpuStorage::BF16(unary_via_f64(v, layout, $f)),
                CpuStorage::F16(v)  => CpuStorage::F16(unary_via_f64(v, layout, $f)),
                CpuStorage::F32(v)  => CpuStorage::F32(unary_via_f64(v, layout, $f)),
                CpuStorage::F64(v)  => CpuStorage::F64(unary_via_f64(v, layout, $f)),
            })
        }};
    }

    match op {
        UnaryOp::Exp    => via_f64!(|x| x.exp()),
        UnaryOp::Log    => via_f64!(|x| x.ln()),
        UnaryOp::Sin    => via_f64!(|x| x.sin()),
        UnaryOp::Cos    => via_f64!(|x| x.cos()),
        UnaryOp::Abs    => via_f64!(|x| x.abs()),
        UnaryOp::Neg    => via_f64!(|x| -x),
        UnaryOp::Sqr    => via_f64!(|x| x * x),
        UnaryOp::Sqrt   => via_f64!(|x| x.sqrt()),
        UnaryOp::Recip  => via_f64!(|x| 1.0 / x),
        UnaryOp::Tanh   => via_f64!(|x| x.tanh()),
        UnaryOp::Relu   => via_f64!(relu),
        UnaryOp::Ceil   => via_f64!(|x| x.ceil()),
        UnaryOp::Floor  => via_f64!(|x| x.floor()),
        UnaryOp::Round  => via_f64!(|x| x.round()),
        UnaryOp::Sign   => via_f64!(sign),
        UnaryOp::Gelu   => via_f64!(gelu_tanh),
        UnaryOp::GeluErf => via_f64!(gelu_erf),
        UnaryOp::Silu   => via_f64!(silu),
    }
}

// ── apply a BinaryOp element-wise ────────────────────────────────────

fn apply_binary_op(
    lhs_cpu: &CpuStorage,
    lhs_layout: &Layout,
    rhs_cpu: &CpuStorage,
    rhs_layout: &Layout,
    op: BinaryOp,
) -> Result<CpuStorage> {
    let (out_shape, lhs_bc, rhs_bc) = broadcast_layouts(lhs_layout, rhs_layout)?;

    macro_rules! bin_op {
        ($lhsv:expr, $rhsv:expr, $ty:ident, $f:expr) => {{
            let v = binary_map_broadcast($lhsv, &lhs_bc, $rhsv, &rhs_bc, $f);
            (CpuStorage::$ty(v), out_shape)
        }};
    }

    // BF16/F16: upcast to f32, compute, downcast — matches PyTorch behavior.
    macro_rules! bin_op_via_f32 {
        ($lhsv:expr, $rhsv:expr, $half_ty:ident, $f:expr) => {{
            let v = binary_map_broadcast($lhsv, &lhs_bc, $rhsv, &rhs_bc,
                |a: $half_ty, b: $half_ty| $half_ty::from_f32($f(a.to_f32(), b.to_f32())));
            (CpuStorage::$half_ty(v), out_shape)
        }};
    }

    let (result_cpu, _out_shape) = match op {
        BinaryOp::Add => match (lhs_cpu, rhs_cpu) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => bin_op!(l, r, F32, |a, b| a + b),
            (CpuStorage::F64(l), CpuStorage::F64(r)) => bin_op!(l, r, F64, |a, b| a + b),
            (CpuStorage::BF16(l), CpuStorage::BF16(r)) => bin_op!(l, r, BF16, |a, b| a + b),
            (CpuStorage::F16(l), CpuStorage::F16(r)) => bin_op!(l, r, F16, |a, b| a + b),
            (CpuStorage::U8(l), CpuStorage::U8(r)) => bin_op!(l, r, U8, |a: u8, b: u8| a.wrapping_add(b)),
            (CpuStorage::U32(l), CpuStorage::U32(r)) => bin_op!(l, r, U32, |a: u32, b: u32| a.wrapping_add(b)),
            (CpuStorage::I64(l), CpuStorage::I64(r)) => bin_op!(l, r, I64, |a: i64, b: i64| a.wrapping_add(b)),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "add" }.bt()),
        },
        BinaryOp::Sub => match (lhs_cpu, rhs_cpu) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => bin_op!(l, r, F32, |a, b| a - b),
            (CpuStorage::F64(l), CpuStorage::F64(r)) => bin_op!(l, r, F64, |a, b| a - b),
            (CpuStorage::BF16(l), CpuStorage::BF16(r)) => bin_op!(l, r, BF16, |a, b| a - b),
            (CpuStorage::F16(l), CpuStorage::F16(r)) => bin_op!(l, r, F16, |a, b| a - b),
            (CpuStorage::U8(l), CpuStorage::U8(r)) => bin_op!(l, r, U8, |a: u8, b: u8| a.wrapping_sub(b)),
            (CpuStorage::U32(l), CpuStorage::U32(r)) => bin_op!(l, r, U32, |a: u32, b: u32| a.wrapping_sub(b)),
            (CpuStorage::I64(l), CpuStorage::I64(r)) => bin_op!(l, r, I64, |a: i64, b: i64| a.wrapping_sub(b)),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "sub" }.bt()),
        },
        BinaryOp::Mul => match (lhs_cpu, rhs_cpu) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => bin_op!(l, r, F32, |a, b| a * b),
            (CpuStorage::F64(l), CpuStorage::F64(r)) => bin_op!(l, r, F64, |a, b| a * b),
            (CpuStorage::BF16(l), CpuStorage::BF16(r)) => bin_op!(l, r, BF16, |a, b| a * b),
            (CpuStorage::F16(l), CpuStorage::F16(r)) => bin_op!(l, r, F16, |a, b| a * b),
            (CpuStorage::U8(l), CpuStorage::U8(r)) => bin_op!(l, r, U8, |a: u8, b: u8| a.wrapping_mul(b)),
            (CpuStorage::U32(l), CpuStorage::U32(r)) => bin_op!(l, r, U32, |a: u32, b: u32| a.wrapping_mul(b)),
            (CpuStorage::I64(l), CpuStorage::I64(r)) => bin_op!(l, r, I64, |a: i64, b: i64| a.wrapping_mul(b)),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "mul" }.bt()),
        },
        BinaryOp::Div => match (lhs_cpu, rhs_cpu) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => bin_op!(l, r, F32, |a, b| a / b),
            (CpuStorage::F64(l), CpuStorage::F64(r)) => bin_op!(l, r, F64, |a, b| a / b),
            (CpuStorage::BF16(l), CpuStorage::BF16(r)) => bin_op!(l, r, BF16, |a, b| a / b),
            (CpuStorage::F16(l), CpuStorage::F16(r)) => bin_op!(l, r, F16, |a, b| a / b),
            (CpuStorage::U8(l), CpuStorage::U8(r)) => bin_op!(l, r, U8, |a: u8, b: u8| a / b),
            (CpuStorage::U32(l), CpuStorage::U32(r)) => bin_op!(l, r, U32, |a: u32, b: u32| a / b),
            (CpuStorage::I64(l), CpuStorage::I64(r)) => bin_op!(l, r, I64, |a: i64, b: i64| a / b),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "div" }.bt()),
        },
        BinaryOp::Min => match (lhs_cpu, rhs_cpu) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => bin_op!(l, r, F32, |a: f32, b: f32| a.min(b)),
            (CpuStorage::F64(l), CpuStorage::F64(r)) => bin_op!(l, r, F64, |a: f64, b: f64| a.min(b)),
            (CpuStorage::BF16(l), CpuStorage::BF16(r)) => bin_op!(l, r, BF16, |a: bf16, b: bf16| if a < b { a } else { b }),
            (CpuStorage::F16(l), CpuStorage::F16(r)) => bin_op!(l, r, F16, |a: f16, b: f16| if a < b { a } else { b }),
            (CpuStorage::U8(l), CpuStorage::U8(r)) => bin_op!(l, r, U8, |a: u8, b: u8| a.min(b)),
            (CpuStorage::U32(l), CpuStorage::U32(r)) => bin_op!(l, r, U32, |a: u32, b: u32| a.min(b)),
            (CpuStorage::I64(l), CpuStorage::I64(r)) => bin_op!(l, r, I64, |a: i64, b: i64| a.min(b)),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "min" }.bt()),
        },
        BinaryOp::Max => match (lhs_cpu, rhs_cpu) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => bin_op!(l, r, F32, |a: f32, b: f32| a.max(b)),
            (CpuStorage::F64(l), CpuStorage::F64(r)) => bin_op!(l, r, F64, |a: f64, b: f64| a.max(b)),
            (CpuStorage::BF16(l), CpuStorage::BF16(r)) => bin_op!(l, r, BF16, |a: bf16, b: bf16| if a > b { a } else { b }),
            (CpuStorage::F16(l), CpuStorage::F16(r)) => bin_op!(l, r, F16, |a: f16, b: f16| if a > b { a } else { b }),
            (CpuStorage::U8(l), CpuStorage::U8(r)) => bin_op!(l, r, U8, |a: u8, b: u8| a.max(b)),
            (CpuStorage::U32(l), CpuStorage::U32(r)) => bin_op!(l, r, U32, |a: u32, b: u32| a.max(b)),
            (CpuStorage::I64(l), CpuStorage::I64(r)) => bin_op!(l, r, I64, |a: i64, b: i64| a.max(b)),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "max" }.bt()),
        },
    };
    Ok(result_cpu)
}

// ── Compare ops → U8 output ─────────────────────────────────────────

fn apply_compare_op(
    lhs_cpu: &CpuStorage,
    lhs_layout: &Layout,
    rhs_cpu: &CpuStorage,
    rhs_layout: &Layout,
    op: CompareOp,
) -> Result<CpuStorage> {
    let (_out_shape, lhs_bc, rhs_bc) = broadcast_layouts(lhs_layout, rhs_layout)?;

    macro_rules! cmp_typed {
        ($l:expr, $r:expr, $f:expr) => {
            binary_map_broadcast($l, &lhs_bc, $r, &rhs_bc, $f)
        };
    }

    macro_rules! dispatch_cmp {
        ($lv:expr, $rv:expr) => {
            match op {
                CompareOp::Eq => cmp_typed!($lv, $rv, |a, b| u8::from(a == b)),
                CompareOp::Ne => cmp_typed!($lv, $rv, |a, b| u8::from(a != b)),
                CompareOp::Lt => cmp_typed!($lv, $rv, |a, b| u8::from(a < b)),
                CompareOp::Gt => cmp_typed!($lv, $rv, |a, b| u8::from(a > b)),
                CompareOp::Ge => cmp_typed!($lv, $rv, |a, b| u8::from(a >= b)),
                CompareOp::Le => cmp_typed!($lv, $rv, |a, b| u8::from(a <= b)),
            }
        };
    }

    let result: Vec<u8> = match (lhs_cpu, rhs_cpu) {
        (CpuStorage::F32(l), CpuStorage::F32(r)) => dispatch_cmp!(l, r),
        (CpuStorage::F64(l), CpuStorage::F64(r)) => dispatch_cmp!(l, r),
        (CpuStorage::BF16(l), CpuStorage::BF16(r)) => dispatch_cmp!(l, r),
        (CpuStorage::F16(l), CpuStorage::F16(r)) => dispatch_cmp!(l, r),
        (CpuStorage::U8(l), CpuStorage::U8(r)) => dispatch_cmp!(l, r),
        (CpuStorage::U32(l), CpuStorage::U32(r)) => dispatch_cmp!(l, r),
        (CpuStorage::I64(l), CpuStorage::I64(r)) => dispatch_cmp!(l, r),
        _ => return Err(Error::DTypeMismatchBinaryOp {
            lhs: lhs_cpu.dtype(), rhs: rhs_cpu.dtype(), op: "compare"
        }.bt()),
    };
    Ok(CpuStorage::U8(result))
}

// ── Reduce ops ───────────────────────────────────────────────────────

/// Reduce a single dimension. Returns `(output_data, output_shape)`.
fn apply_reduce_op(
    cpu: &CpuStorage,
    layout: &Layout,
    dim: usize,
    op: ReduceOp,
) -> Result<CpuStorage> {
    macro_rules! reduce_typed {
        ($v:expr, $ty:ident) => {
            reduce_dim_typed($v, layout, dim, op).map(|(r, _)| CpuStorage::$ty(r))
        };
    }
    macro_rules! reduce_arg_typed {
        ($v:expr) => {
            reduce_dim_arg($v, layout, dim, op).map(CpuStorage::U32)
        };
    }

    let is_arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);

    if is_arg {
        match cpu {
            CpuStorage::F32(v) => reduce_arg_typed!(v),
            CpuStorage::F64(v) => reduce_arg_typed!(v),
            CpuStorage::BF16(v) => reduce_arg_typed!(v),
            CpuStorage::F16(v) => reduce_arg_typed!(v),
            CpuStorage::U8(v) => reduce_arg_typed!(v),
            CpuStorage::U32(v) => reduce_arg_typed!(v),
            CpuStorage::I32(v) => reduce_arg_typed!(v),
            CpuStorage::I64(v) => reduce_arg_typed!(v),
        }
    } else {
        match cpu {
            CpuStorage::F32(v) => reduce_typed!(v, F32),
            CpuStorage::F64(v) => reduce_typed!(v, F64),
            CpuStorage::BF16(v) => reduce_typed!(v, BF16),
            CpuStorage::F16(v) => reduce_typed!(v, F16),
            CpuStorage::U8(v) => reduce_typed!(v, U8),
            CpuStorage::U32(v) => reduce_typed!(v, U32),
            CpuStorage::I32(v) => reduce_typed!(v, I32),
            CpuStorage::I64(v) => reduce_typed!(v, I64),
        }
    }
}

/// Reduce a single dimension of a typed slice; returns `(result_vec, dst_shape)`.
/// Works for Sum, Max, Min.
fn reduce_dim_typed<T: WithDType>(
    src: &[T],
    layout: &Layout,
    dim: usize,
    op: ReduceOp,
) -> Result<(Vec<T>, Shape)> {
    let dims = layout.dims();
    let _reduce_size = dims[dim];
    let dst_elems: usize = dims.iter().enumerate()
        .filter(|&(d, _)| d != dim)
        .map(|(_, &s)| s)
        .product::<usize>()
        .max(1);

    // Init: Sum=0, Max/Min deferred to per-slice first element.
    let mut dst = vec![T::from_f64(0.0); dst_elems];

    // Fast path: contiguous reduce over last dim (most common case, e.g. mean/sum over hidden).
    if dim == dims.len() - 1 {
        if let Some((start, _end)) = layout.contiguous_offsets() {
            let reduce_size = dims[dim];
            let src = &src[start..];
            for (dst_i, dst_v) in dst.iter_mut().enumerate() {
                let base = dst_i * reduce_size;
                // Init from first element of this slice.
                *dst_v = src[base];
                match op {
                    ReduceOp::Sum => {
                        let mut acc = src[base].to_f64();
                        for i in 1..reduce_size {
                            acc += src[base + i].to_f64();
                        }
                        *dst_v = T::from_f64(acc);
                    }
                    ReduceOp::Max => {
                        for i in 1..reduce_size {
                            if src[base + i].to_f64() > dst_v.to_f64() {
                                *dst_v = src[base + i];
                            }
                        }
                    }
                    ReduceOp::Min => {
                        for i in 1..reduce_size {
                            if src[base + i].to_f64() < dst_v.to_f64() {
                                *dst_v = src[base + i];
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            let out_dims: Vec<usize> = dims.iter().enumerate()
                .filter(|&(d, _)| d != dim)
                .map(|(_, &s)| s)
                .collect();
            let out_shape = if out_dims.is_empty() { Shape::from(1usize) } else { Shape::from(out_dims) };
            return Ok((dst, out_shape));
        }
    }

    // General path: strided/non-contiguous.
    let mut initialized = vec![false; dst_elems];
    let mut src_multi = vec![0usize; dims.len()];
    for src_phys in layout.strided_index() {
        let mut dst_lin = 0usize;
        let mut factor = 1usize;
        for d in (0..dims.len()).rev() {
            if d == dim { continue; }
            dst_lin += src_multi[d] * factor;
            factor *= dims[d];
        }
        if !initialized[dst_lin] {
            // First element for this output position — use as init.
            dst[dst_lin] = src[src_phys];
            initialized[dst_lin] = true;
        } else {
            match op {
                ReduceOp::Sum => {
                    dst[dst_lin] = T::from_f64(dst[dst_lin].to_f64() + src[src_phys].to_f64());
                }
                ReduceOp::Max => {
                    if src[src_phys].to_f64() > dst[dst_lin].to_f64() {
                        dst[dst_lin] = src[src_phys];
                    }
                }
                ReduceOp::Min => {
                    if src[src_phys].to_f64() < dst[dst_lin].to_f64() {
                        dst[dst_lin] = src[src_phys];
                    }
                }
                ReduceOp::ArgMax | ReduceOp::ArgMin => unreachable!(),
            }
        }
        advance_multi_index(&mut src_multi, dims);
    }

    // Build output shape (remove the reduced dim).
    let out_dims: Vec<usize> = dims.iter().enumerate()
        .filter(|&(d, _)| d != dim)
        .map(|(_, &s)| s)
        .collect();
    let out_shape = if out_dims.is_empty() { Shape::from(1usize) } else { Shape::from(out_dims) };
    Ok((dst, out_shape))
}

/// ArgMax / ArgMin over a single dimension → U32 output.
fn reduce_dim_arg<T: WithDType>(
    src: &[T],
    layout: &Layout,
    dim: usize,
    op: ReduceOp,
) -> Result<Vec<u32>> {
    let dims = layout.dims();
    let _reduce_size = dims[dim];
    let dst_elems: usize = dims.iter().enumerate()
        .filter(|&(d, _)| d != dim)
        .map(|(_, &s)| s)
        .product::<usize>()
        .max(1);

    // For each output position track (best_val, best_idx).
    let mut best_val: Vec<f64> = Vec::with_capacity(dst_elems);
    let mut best_idx: Vec<u32> = vec![0u32; dst_elems];
    // Initialize with first encounter per output slot.
    let mut initialized: Vec<bool> = vec![false; dst_elems];

    let mut src_multi = vec![0usize; dims.len()];
    for src_phys in layout.strided_index() {
        let mut dst_lin = 0usize;
        let mut factor = 1usize;
        for d in (0..dims.len()).rev() {
            if d == dim { continue; }
            dst_lin += src_multi[d] * factor;
            factor *= dims[d];
        }
        let val = src[src_phys].to_f64();
        let idx_in_dim = src_multi[dim];
        if !initialized[dst_lin] {
            best_val.push(val);
            best_idx[dst_lin] = idx_in_dim as u32;
            initialized[dst_lin] = true;
        } else {
            let better = match op {
                ReduceOp::ArgMax => val > best_val[dst_lin],
                ReduceOp::ArgMin => val < best_val[dst_lin],
                _ => unreachable!(),
            };
            if better {
                best_val[dst_lin] = val;
                best_idx[dst_lin] = idx_in_dim as u32;
            }
        }
        advance_multi_index(&mut src_multi, dims);
    }
    Ok(best_idx)
}

/// Advance a multi-index in row-major order.
fn advance_multi_index(multi: &mut [usize], dims: &[usize]) {
    for d in (0..dims.len()).rev() {
        multi[d] += 1;
        if multi[d] < dims[d] { break; }
        multi[d] = 0;
    }
}

// ── cast (dtype conversion) ─────────────────────────────────────────

fn apply_cast(cpu: &CpuStorage, layout: &Layout, dst_dtype: DType) -> Result<CpuStorage> {
    // Direct type-to-type cast via `as`, matching candle's behavior.
    // Avoids f64 intermediate which loses precision for i64 > 2^53.
    macro_rules! collect {
        ($v:expr, $map:expr) => {{
            match layout.contiguous_offsets() {
                Some((s, e)) => $v[s..e].iter().map($map).collect::<Vec<_>>(),
                None => layout.strided_index().map(|i| ($map)(&$v[i])).collect::<Vec<_>>(),
            }
        }};
    }

    macro_rules! cast_from {
        ($v:expr, $src_ty:ty) => {
            match dst_dtype {
                DType::U8   => Ok(CpuStorage::U8(collect!($v, |&x: &$src_ty| x as u8))),
                DType::U32  => Ok(CpuStorage::U32(collect!($v, |&x: &$src_ty| x as u32))),
                DType::I32  => Ok(CpuStorage::I32(collect!($v, |&x: &$src_ty| x as i32))),
                DType::I64  => Ok(CpuStorage::I64(collect!($v, |&x: &$src_ty| x as i64))),
                DType::F32  => Ok(CpuStorage::F32(collect!($v, |&x: &$src_ty| x as f32))),
                DType::F64  => Ok(CpuStorage::F64(collect!($v, |&x: &$src_ty| x as f64))),
                DType::BF16 => Ok(CpuStorage::BF16(collect!($v, |&x: &$src_ty| bf16::from_f32(x as f32)))),
                DType::F16  => Ok(CpuStorage::F16(collect!($v, |&x: &$src_ty| f16::from_f32(x as f32)))),
                other => Err(Error::UnsupportedDTypeForOp(other, "cast").bt()),
            }
        };
    }

    match cpu {
        CpuStorage::U8(v)   => cast_from!(v, u8),
        CpuStorage::U32(v)  => cast_from!(v, u32),
        CpuStorage::I32(v)  => cast_from!(v, i32),
        CpuStorage::I64(v)  => cast_from!(v, i64),
        CpuStorage::F32(v)  => cast_from!(v, f32),
        CpuStorage::F64(v)  => cast_from!(v, f64),
        CpuStorage::BF16(v) => {
            // BF16 → f32 first for better precision in subsequent cast
            match dst_dtype {
                DType::BF16 => Ok(CpuStorage::BF16(collect!(v, |&x: &bf16| x))),
                DType::F16  => Ok(CpuStorage::F16(collect!(v, |&x: &bf16| f16::from_f32(x.to_f32())))),
                DType::F32  => Ok(CpuStorage::F32(collect!(v, |&x: &bf16| x.to_f32()))),
                DType::F64  => Ok(CpuStorage::F64(collect!(v, |&x: &bf16| x.to_f32() as f64))),
                DType::U8   => Ok(CpuStorage::U8(collect!(v, |&x: &bf16| x.to_f32() as u8))),
                DType::U32  => Ok(CpuStorage::U32(collect!(v, |&x: &bf16| x.to_f32() as u32))),
                DType::I32  => Ok(CpuStorage::I32(collect!(v, |&x: &bf16| x.to_f32() as i32))),
                DType::I64  => Ok(CpuStorage::I64(collect!(v, |&x: &bf16| x.to_f32() as i64))),
                other => Err(Error::UnsupportedDTypeForOp(other, "cast").bt()),
            }
        }
        CpuStorage::F16(v) => {
            match dst_dtype {
                DType::F16  => Ok(CpuStorage::F16(collect!(v, |&x: &f16| x))),
                DType::BF16 => Ok(CpuStorage::BF16(collect!(v, |&x: &f16| bf16::from_f32(x.to_f32())))),
                DType::F32  => Ok(CpuStorage::F32(collect!(v, |&x: &f16| x.to_f32()))),
                DType::F64  => Ok(CpuStorage::F64(collect!(v, |&x: &f16| x.to_f32() as f64))),
                DType::U8   => Ok(CpuStorage::U8(collect!(v, |&x: &f16| x.to_f32() as u8))),
                DType::U32  => Ok(CpuStorage::U32(collect!(v, |&x: &f16| x.to_f32() as u32))),
                DType::I32  => Ok(CpuStorage::I32(collect!(v, |&x: &f16| x.to_f32() as i32))),
                DType::I64  => Ok(CpuStorage::I64(collect!(v, |&x: &f16| x.to_f32() as i64))),
                other => Err(Error::UnsupportedDTypeForOp(other, "cast").bt()),
            }
        }
    }
}

// ── affine transform ─────────────────────────────────────────────────

fn apply_affine(cpu: &CpuStorage, layout: &Layout, mul: f64, add: f64) -> CpuStorage {
    match cpu {
        CpuStorage::F32(v)  => CpuStorage::F32(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::F64(v)  => CpuStorage::F64(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::BF16(v) => CpuStorage::BF16(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::F16(v)  => CpuStorage::F16(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::U8(v)   => CpuStorage::U8(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::U32(v)  => CpuStorage::U32(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::I32(v)  => CpuStorage::I32(unary_via_f64(v, layout, |x| x * mul + add)),
        CpuStorage::I64(v)  => CpuStorage::I64(unary_via_f64(v, layout, |x| x * mul + add)),
    }
}

// ── matmul ────────────────────────────────────────────────────────────
//
// Naive O(n³) batch matmul. Correctness over speed.
// The layout must be contiguous or we make it so via flatten to f64 first.

fn matmul_f64(
    a: &[f64], a_layout: &Layout,
    b: &[f64], b_layout: &Layout,
) -> Result<(Vec<f64>, Shape)> {
    let a_dims = a_layout.dims();
    let b_dims = b_layout.dims();
    if a_dims.len() < 2 || b_dims.len() < 2 {
        crate::bail!("matmul: tensors must be at least 2D");
    }
    let (m, k) = (a_dims[a_dims.len() - 2], a_dims[a_dims.len() - 1]);
    let (k2, n) = (b_dims[b_dims.len() - 2], b_dims[b_dims.len() - 1]);
    if k != k2 {
        crate::bail!("matmul: inner dimension mismatch {k} != {k2}");
    }

    // Compute broadcast batch shape.
    let a_batch: Shape = Shape::from(&a_dims[..a_dims.len() - 2]);
    let b_batch: Shape = Shape::from(&b_dims[..b_dims.len() - 2]);
    let bc_batch = a_batch.broadcast_shape_binary_op(&b_batch, "matmul")?;
    let batch_elems = bc_batch.elem_count().max(1);

    let a_bc_layout = a_layout.broadcast_as(
        Shape::from([bc_batch.dims(), &[m, k]].concat())
    )?;
    let b_bc_layout = b_layout.broadcast_as(
        Shape::from([bc_batch.dims(), &[k, n]].concat())
    )?;

    let out_len = batch_elems * m * n;
    let mut out = vec![0f64; out_len];

    let mn = m * n;
    let mk = m * k;
    let kn = k * n;

    // Collect a and b into contiguous f64 slices honouring strides.
    let a_flat: Vec<f64> = a_bc_layout.strided_index().map(|i| a[i]).collect();
    let b_flat: Vec<f64> = b_bc_layout.strided_index().map(|i| b[i]).collect();

    for batch in 0..batch_elems {
        let a_off = batch * mk;
        let b_off = batch * kn;
        let o_off = batch * mn;
        for i in 0..m {
            for kk in 0..k {
                let av = a_flat[a_off + i * k + kk];
                for j in 0..n {
                    out[o_off + i * n + j] += av * b_flat[b_off + kk * n + j];
                }
            }
        }
    }

    let out_shape = if bc_batch.dims().is_empty() {
        Shape::from(vec![m, n])
    } else {
        Shape::from([bc_batch.dims(), &[m, n]].concat())
    };
    Ok((out, out_shape))
}

fn apply_matmul_typed<T: WithDType>(
    a: &[T],
    a_layout: &Layout,
    b: &[T],
    b_layout: &Layout,
) -> Result<(Vec<T>, Shape)> {
    let a_f64: Vec<f64> = a_layout.strided_index().map(|i| a[i].to_f64()).collect();
    let b_f64: Vec<f64> = b_layout.strided_index().map(|i| b[i].to_f64()).collect();

    // Rebuild contiguous layouts of the same shape for matmul_f64.
    let a_cont_layout = Layout::contiguous(a_layout.shape().clone());
    let b_cont_layout = Layout::contiguous(b_layout.shape().clone());

    let (out_f64, out_shape) = matmul_f64(&a_f64, &a_cont_layout, &b_f64, &b_cont_layout)?;
    let out: Vec<T> = out_f64.iter().map(|&v| T::from_f64(v)).collect();
    Ok((out, out_shape))
}

// ── index_select ──────────────────────────────────────────────────────

fn apply_index_select(
    cpu: &CpuStorage,
    layout: &Layout,
    idx_cpu: &CpuStorage,
    idx_layout: &Layout,
    dim: usize,
) -> Result<CpuStorage> {
    // Collect indices as usize
    let indices: Vec<usize> = match idx_cpu {
        CpuStorage::U32(v) => idx_layout.strided_index().map(|i| v[i] as usize).collect(),
        CpuStorage::I64(v) => idx_layout.strided_index().map(|i| v[i] as usize).collect(),
        CpuStorage::U8(v)  => idx_layout.strided_index().map(|i| v[i] as usize).collect(),
        other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), "index_select indices").bt()),
    };

    let dims = layout.dims();
    let dim_size = dims[dim];
    let num_idx = indices.len();

    // Output shape: replace dims[dim] with num_idx.
    let mut out_dims = dims.to_vec();
    out_dims[dim] = num_idx;

    macro_rules! index_select_typed {
        ($v:expr, $ty:ident) => {{
            let out_len: usize = out_dims.iter().product();
            let mut out = Vec::with_capacity(out_len);

            // Iterate over output elements using a multi-index.
            let mut out_multi = vec![0usize; out_dims.len()];
            for _ in 0..out_len {
                // Map output multi-index to source multi-index.
                let mut src_multi = out_multi.clone();
                let mapped_idx = indices[out_multi[dim]];
                if mapped_idx >= dim_size {
                    return Err(Error::InvalidIndex { op: "index_select", index: mapped_idx, size: dim_size }.bt());
                }
                src_multi[dim] = mapped_idx;
                // Compute physical source index.
                let src_phys = layout.start_offset()
                    + src_multi.iter().zip(layout.stride()).map(|(m, s)| m * s).sum::<usize>();
                out.push($v[src_phys]);
                advance_multi_index(&mut out_multi, &out_dims);
            }
            CpuStorage::$ty(out)
        }};
    }

    Ok(match cpu {
        CpuStorage::F32(v)  => index_select_typed!(v, F32),
        CpuStorage::F64(v)  => index_select_typed!(v, F64),
        CpuStorage::BF16(v) => index_select_typed!(v, BF16),
        CpuStorage::F16(v)  => index_select_typed!(v, F16),
        CpuStorage::U8(v)   => index_select_typed!(v, U8),
        CpuStorage::U32(v)  => index_select_typed!(v, U32),
        CpuStorage::I32(v)  => index_select_typed!(v, I32),
        CpuStorage::I64(v)  => index_select_typed!(v, I64),
    })
}

// ── gather ────────────────────────────────────────────────────────────
// out[...i...] = x[...indices[...i...]...] for the gather dim.

fn apply_gather(
    cpu: &CpuStorage,
    layout: &Layout,
    idx_cpu: &CpuStorage,
    idx_layout: &Layout,
    dim: usize,
) -> Result<CpuStorage> {
    let idx_dims = idx_layout.dims().to_vec();
    let out_len: usize = idx_dims.iter().product();

    macro_rules! gather_typed {
        ($v:expr, $ty:ident) => {{
            let mut out = Vec::with_capacity(out_len);
            let mut idx_multi = vec![0usize; idx_dims.len()];
            for idx_phys in idx_layout.strided_index() {
                let gather_idx = match idx_cpu {
                    CpuStorage::U32(iv) => iv[idx_phys] as usize,
                    CpuStorage::I64(iv) => iv[idx_phys] as usize,
                    CpuStorage::U8(iv)  => iv[idx_phys] as usize,
                    other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), "gather indices").bt()),
                };
                let mut src_multi = idx_multi.clone();
                src_multi[dim] = gather_idx;
                let src_phys = layout.start_offset()
                    + src_multi.iter().zip(layout.stride()).map(|(m, s)| m * s).sum::<usize>();
                out.push($v[src_phys]);
                advance_multi_index(&mut idx_multi, &idx_dims);
            }
            CpuStorage::$ty(out)
        }};
    }

    Ok(match cpu {
        CpuStorage::F32(v)  => gather_typed!(v, F32),
        CpuStorage::F64(v)  => gather_typed!(v, F64),
        CpuStorage::BF16(v) => gather_typed!(v, BF16),
        CpuStorage::F16(v)  => gather_typed!(v, F16),
        CpuStorage::U8(v)   => gather_typed!(v, U8),
        CpuStorage::U32(v)  => gather_typed!(v, U32),
        CpuStorage::I32(v)  => gather_typed!(v, I32),
        CpuStorage::I64(v)  => gather_typed!(v, I64),
    })
}

// ── scatter_add ───────────────────────────────────────────────────────
// out = x.clone(); out[..idx..] += src[..idx..]

fn apply_scatter_add(
    x_cpu: &CpuStorage,
    x_layout: &Layout,
    idx_cpu: &CpuStorage,
    idx_layout: &Layout,
    src_cpu: &CpuStorage,
    src_layout: &Layout,
    dim: usize,
) -> Result<CpuStorage> {
    // Start from a contiguous copy of x.
    let mut out_cpu = apply_contiguous(x_cpu, x_layout);
    let out_dims = x_layout.dims().to_vec();
    let out_layout = Layout::contiguous(Shape::from(out_dims.clone()));

    macro_rules! scatter_add_typed {
        ($out:expr, $src:expr, $ty:ident) => {{
            let idx_dims = idx_layout.dims().to_vec();
            let mut idx_multi = vec![0usize; idx_dims.len()];
            let mut src_iter = src_layout.strided_index();
            for idx_phys in idx_layout.strided_index() {
                let scatter_idx = match idx_cpu {
                    CpuStorage::U32(iv) => iv[idx_phys] as usize,
                    CpuStorage::I64(iv) => iv[idx_phys] as usize,
                    CpuStorage::U8(iv)  => iv[idx_phys] as usize,
                    other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), "scatter_add indices").bt()),
                };
                let src_phys = src_iter.next().unwrap();
                // Output multi-index = idx_multi but with [dim] replaced by scatter_idx.
                let mut out_multi = idx_multi.clone();
                out_multi[dim] = scatter_idx;
                let out_phys: usize = out_multi.iter().zip(out_layout.stride()).map(|(m, s)| m * s).sum();
                let v = ($out[out_phys].to_f64() + $src[src_phys].to_f64());
                $out[out_phys] = <_ as WithDType>::from_f64(v);
                advance_multi_index(&mut idx_multi, &idx_dims);
            }
        }};
    }

    match (&mut out_cpu, src_cpu) {
        (CpuStorage::F32(o), CpuStorage::F32(s)) => scatter_add_typed!(o, s, F32),
        (CpuStorage::F64(o), CpuStorage::F64(s)) => scatter_add_typed!(o, s, F64),
        (CpuStorage::BF16(o), CpuStorage::BF16(s)) => scatter_add_typed!(o, s, BF16),
        (CpuStorage::F16(o), CpuStorage::F16(s)) => scatter_add_typed!(o, s, F16),
        (CpuStorage::U8(o), CpuStorage::U8(s)) => scatter_add_typed!(o, s, U8),
        (CpuStorage::U32(o), CpuStorage::U32(s)) => scatter_add_typed!(o, s, U32),
        (CpuStorage::I64(o), CpuStorage::I64(s)) => scatter_add_typed!(o, s, I64),
        _ => return Err(Error::DTypeMismatchBinaryOp { lhs: x_cpu.dtype(), rhs: src_cpu.dtype(), op: "scatter_add" }.bt()),
    }
    Ok(out_cpu)
}

// ── index_add ─────────────────────────────────────────────────────────
// out = x.clone(); for each i, out[indices[i]] += src[i] along dim.

fn apply_index_add(
    x_cpu: &CpuStorage,
    x_layout: &Layout,
    idx_cpu: &CpuStorage,
    idx_layout: &Layout,
    src_cpu: &CpuStorage,
    src_layout: &Layout,
    dim: usize,
) -> Result<CpuStorage> {
    let mut out_cpu = apply_contiguous(x_cpu, x_layout);
    let out_dims = x_layout.dims().to_vec();
    let out_layout = Layout::contiguous(Shape::from(out_dims.clone()));

    // Collect indices.
    let indices: Vec<usize> = match idx_cpu {
        CpuStorage::U32(v) => idx_layout.strided_index().map(|i| v[i] as usize).collect(),
        CpuStorage::I64(v) => idx_layout.strided_index().map(|i| v[i] as usize).collect(),
        CpuStorage::U8(v)  => idx_layout.strided_index().map(|i| v[i] as usize).collect(),
        other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), "index_add indices").bt()),
    };

    let src_dims = src_layout.dims().to_vec();
    let _num_idx = indices.len();
    // src has same rank as x, src.dims()[dim] == _num_idx.

    macro_rules! index_add_typed {
        ($out:expr, $src:expr) => {{
            let mut src_multi = vec![0usize; src_dims.len()];
            for src_phys in src_layout.strided_index() {
                let mapped = indices[src_multi[dim]];
                let mut out_multi = src_multi.clone();
                out_multi[dim] = mapped;
                let out_phys: usize = out_multi.iter().zip(out_layout.stride()).map(|(m, s)| m * s).sum();
                let v = $out[out_phys].to_f64() + $src[src_phys].to_f64();
                $out[out_phys] = <_ as WithDType>::from_f64(v);
                advance_multi_index(&mut src_multi, &src_dims);
            }
        }};
    }

    match (&mut out_cpu, src_cpu) {
        (CpuStorage::F32(o), CpuStorage::F32(s)) => index_add_typed!(o, s),
        (CpuStorage::F64(o), CpuStorage::F64(s)) => index_add_typed!(o, s),
        (CpuStorage::BF16(o), CpuStorage::BF16(s)) => index_add_typed!(o, s),
        (CpuStorage::F16(o), CpuStorage::F16(s)) => index_add_typed!(o, s),
        (CpuStorage::U8(o), CpuStorage::U8(s)) => index_add_typed!(o, s),
        (CpuStorage::U32(o), CpuStorage::U32(s)) => index_add_typed!(o, s),
        (CpuStorage::I64(o), CpuStorage::I64(s)) => index_add_typed!(o, s),
        _ => return Err(Error::DTypeMismatchBinaryOp { lhs: x_cpu.dtype(), rhs: src_cpu.dtype(), op: "index_add" }.bt()),
    }
    Ok(out_cpu)
}

// ── where_cond ────────────────────────────────────────────────────────

fn apply_where_cond(
    cond_cpu: &CpuStorage,
    cond_layout: &Layout,
    t_cpu: &CpuStorage,
    t_layout: &Layout,
    f_cpu: &CpuStorage,
    f_layout: &Layout,
) -> Result<CpuStorage> {
    // Condition can be any integer type — nonzero = true.
    macro_rules! is_true {
        ($v:expr, $i:expr) => { $v[$i] != 0 }
    }
    macro_rules! is_true_i64 {
        ($v:expr, $i:expr) => { $v[$i] != 0 }
    }

    macro_rules! where_typed {
        ($tv:expr, $fv:expr, $ty:ident) => {{
            let out: Vec<_> = cond_layout
                .strided_index()
                .zip(t_layout.strided_index().zip(f_layout.strided_index()))
                .map(|(ci, (ti, fi))| {
                    let cond_true = match cond_cpu {
                        CpuStorage::U8(v) => v[ci] != 0,
                        CpuStorage::U32(v) => v[ci] != 0,
                        CpuStorage::I64(v) => v[ci] != 0,
                        _ => false,
                    };
                    if cond_true { $tv[ti] } else { $fv[fi] }
                })
                .collect();
            CpuStorage::$ty(out)
        }};
    }

    Ok(match (t_cpu, f_cpu) {
        (CpuStorage::F32(t), CpuStorage::F32(f)) => where_typed!(t, f, F32),
        (CpuStorage::F64(t), CpuStorage::F64(f)) => where_typed!(t, f, F64),
        (CpuStorage::BF16(t), CpuStorage::BF16(f)) => where_typed!(t, f, BF16),
        (CpuStorage::F16(t), CpuStorage::F16(f)) => where_typed!(t, f, F16),
        (CpuStorage::U8(t), CpuStorage::U8(f)) => where_typed!(t, f, U8),
        (CpuStorage::U32(t), CpuStorage::U32(f)) => where_typed!(t, f, U32),
        (CpuStorage::I64(t), CpuStorage::I64(f)) => where_typed!(t, f, I64),
        _ => return Err(Error::DTypeMismatchBinaryOp { lhs: t_cpu.dtype(), rhs: f_cpu.dtype(), op: "where_cond" }.bt()),
    })
}

// ── contiguous (copy to contiguous layout) ────────────────────────────

fn apply_contiguous(cpu: &CpuStorage, layout: &Layout) -> CpuStorage {
    macro_rules! cont_typed {
        ($v:expr, $ty:ident) => {{
            match layout.contiguous_offsets() {
                Some((s, e)) => CpuStorage::$ty($v[s..e].to_vec()),
                None => CpuStorage::$ty(layout.strided_index().map(|i| $v[i]).collect()),
            }
        }};
    }
    match cpu {
        CpuStorage::F32(v)  => cont_typed!(v, F32),
        CpuStorage::F64(v)  => cont_typed!(v, F64),
        CpuStorage::BF16(v) => cont_typed!(v, BF16),
        CpuStorage::F16(v)  => cont_typed!(v, F16),
        CpuStorage::U8(v)   => cont_typed!(v, U8),
        CpuStorage::U32(v)  => cont_typed!(v, U32),
        CpuStorage::I32(v)  => cont_typed!(v, I32),
        CpuStorage::I64(v)  => cont_typed!(v, I64),
    }
}

// ── sort_last_dim ─────────────────────────────────────────────────────

fn apply_sort_last_dim(
    cpu: &CpuStorage,
    layout: &Layout,
    asc: bool,
) -> Result<(CpuStorage, CpuStorage)> {
    let dims = layout.dims();
    if dims.is_empty() {
        crate::bail!("sort_last_dim: scalar tensor has no last dim");
    }
    let last = dims[dims.len() - 1];
    let batch: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

    // Collect all elements in logical order.
    let all_f64: Vec<f64> = layout.strided_index()
        .map(|i| match cpu {
            CpuStorage::F32(v) => v[i] as f64,
            CpuStorage::F64(v) => v[i],
            CpuStorage::BF16(v) => v[i].to_f64(),
            CpuStorage::F16(v) => v[i].to_f64(),
            CpuStorage::U8(v) => v[i] as f64,
            CpuStorage::U32(v) => v[i] as f64,
            CpuStorage::I32(v) => v[i] as f64,
            CpuStorage::I64(v) => v[i] as f64,
        })
        .collect();

    let mut sorted_f64 = all_f64.clone();
    let mut indices_u32 = vec![0u32; batch * last];

    for b in 0..batch {
        let slice = &all_f64[b * last..(b + 1) * last];
        let mut idx: Vec<u32> = (0..last as u32).collect();
        if asc {
            idx.sort_unstable_by(|&i, &j| slice[i as usize].partial_cmp(&slice[j as usize]).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            idx.sort_unstable_by(|&i, &j| slice[j as usize].partial_cmp(&slice[i as usize]).unwrap_or(std::cmp::Ordering::Equal));
        }
        let base = b * last;
        for (k, &orig) in idx.iter().enumerate() {
            sorted_f64[base + k] = slice[orig as usize];
            indices_u32[base + k] = orig;
        }
    }

    // Convert back to original dtype.
    macro_rules! back_to_dtype {
        ($ty:ident, $T:ty) => {
            CpuStorage::$ty(sorted_f64.iter().map(|&v| <$T as WithDType>::from_f64(v)).collect())
        };
    }
    let sorted_cpu = match cpu {
        CpuStorage::F32(_)  => back_to_dtype!(F32, f32),
        CpuStorage::F64(_)  => back_to_dtype!(F64, f64),
        CpuStorage::BF16(_) => back_to_dtype!(BF16, bf16),
        CpuStorage::F16(_)  => back_to_dtype!(F16, f16),
        CpuStorage::U8(_)   => back_to_dtype!(U8, u8),
        CpuStorage::U32(_)  => back_to_dtype!(U32, u32),
        CpuStorage::I32(_)  => back_to_dtype!(I32, i32),
        CpuStorage::I64(_)  => back_to_dtype!(I64, i64),
    };

    Ok((sorted_cpu, CpuStorage::U32(indices_u32)))
}

// ── cat ───────────────────────────────────────────────────────────────

fn apply_cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        crate::bail!("cat: empty tensor list");
    }
    let first = tensors[0];
    let first_dims = first.our_layout().dims().to_vec();
    let rank = first_dims.len();
    if dim >= rank {
        crate::bail!("cat: dim {dim} out of range for rank {rank}");
    }

    // Validate all tensors have matching shape except on cat dim.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        let d = t.our_layout().dims();
        if d.len() != rank {
            crate::bail!("cat: rank mismatch at index {i}");
        }
        for ax in 0..rank {
            if ax != dim && d[ax] != first_dims[ax] {
                crate::bail!("cat: shape mismatch at axis {ax} for tensor {i}");
            }
        }
        if t.dtype() != first.dtype() {
            crate::bail!("cat: dtype mismatch at index {i}");
        }
    }

    // Output shape.
    let cat_size: usize = tensors.iter().map(|t| t.our_layout().dims()[dim]).sum();
    let mut out_dims = first_dims.clone();
    out_dims[dim] = cat_size;
    let device = *first.device();

    // Allocate output.
    let out_elems: usize = out_dims.iter().product();
    let dtype = first.dtype();

    // We'll accumulate by writing each tensor's data into slices of the output.
    // Use a mutable CpuStorage output then copy via index.
    let mut out_cpu = CpuStorage::zeros(dtype, out_elems);
    let out_layout = Layout::contiguous(Shape::from(out_dims.clone()));

    let mut start = 0usize;
    for t in tensors {
        let src_cpu = t.storage().as_cpu()?;
        let src_layout = t.our_layout();
        let src_dim_size = src_layout.dims()[dim];

        // Copy each element from src to its position in out.
        let src_dims = src_layout.dims().to_vec();
        let mut src_multi = vec![0usize; rank];
        for src_phys in src_layout.strided_index() {
            let mut out_multi = src_multi.clone();
            out_multi[dim] += start;
            let out_phys: usize = out_multi.iter().zip(out_layout.stride()).map(|(m, s)| m * s).sum();

            match (src_cpu, &mut out_cpu) {
                (CpuStorage::F32(s), CpuStorage::F32(o)) => o[out_phys] = s[src_phys],
                (CpuStorage::F64(s), CpuStorage::F64(o)) => o[out_phys] = s[src_phys],
                (CpuStorage::BF16(s), CpuStorage::BF16(o)) => o[out_phys] = s[src_phys],
                (CpuStorage::F16(s), CpuStorage::F16(o)) => o[out_phys] = s[src_phys],
                (CpuStorage::U8(s), CpuStorage::U8(o)) => o[out_phys] = s[src_phys],
                (CpuStorage::U32(s), CpuStorage::U32(o)) => o[out_phys] = s[src_phys],
                (CpuStorage::I64(s), CpuStorage::I64(o)) => o[out_phys] = s[src_phys],
                _ => return Err(Error::DTypeMismatchBinaryOp { lhs: src_cpu.dtype(), rhs: out_cpu.dtype(), op: "cat" }.bt()),
            }
            advance_multi_index(&mut src_multi, &src_dims);
        }
        start += src_dim_size;
    }

    Ok(make_tensor(out_cpu, Shape::from(out_dims), &device))
}

// ══════════════════════════════════════════════════════════════════
// DeviceTensorOps struct and TensorOps impl
// ══════════════════════════════════════════════════════════════════

/// TensorOps for the `Storage::Device(CpuStorage)` path.
///
/// Correctness implementation — no SIMD, no parallelism.
/// prelude-cpu overrides this with optimised kernels.
pub struct DeviceTensorOps;

impl Ops for DeviceTensorOps {
    fn default_impl(&self) -> &dyn Ops { self }

    // ── unary ──────────────────────────────────────────────────────

    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> {
        let cpu = x.storage().as_cpu()?;
        let layout = x.our_layout();
        let out_cpu = apply_unary_op(cpu, layout, op)?;
        let shape = layout.shape().clone();
        let device = *x.device();
        Ok(make_tensor(out_cpu, shape, &device))
    }

    // ── binary ─────────────────────────────────────────────────────

    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
        let a_cpu = a.storage().as_cpu()?;
        let b_cpu = b.storage().as_cpu()?;
        let a_layout = a.our_layout();
        let b_layout = b.our_layout();

        let out_shape = a_layout.shape().broadcast_shape_binary_op(b_layout.shape(), "binary")?;
        let out_cpu = apply_binary_op(a_cpu, a_layout, b_cpu, b_layout, op)?;
        let device = *a.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── compare ────────────────────────────────────────────────────

    fn compare(&self, a: &Tensor, b: &Tensor, op: CompareOp) -> Result<Tensor> {
        let a_cpu = a.storage().as_cpu()?;
        let b_cpu = b.storage().as_cpu()?;
        let a_layout = a.our_layout();
        let b_layout = b.our_layout();

        let out_shape = a_layout.shape().broadcast_shape_binary_op(b_layout.shape(), "compare")?;
        let out_cpu = apply_compare_op(a_cpu, a_layout, b_cpu, b_layout, op)?;
        let device = *a.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── reduce ─────────────────────────────────────────────────────

    fn reduce(&self, x: &Tensor, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Tensor> {
        let cpu = x.storage().as_cpu()?;
        let layout = x.our_layout();
        let dims = layout.dims();

        if dim >= dims.len() {
            crate::bail!("reduce: dim {dim} out of range for rank {}", dims.len());
        }

        let out_cpu = apply_reduce_op(cpu, layout, dim, op)?;

        // Build output shape.
        let mut out_dims: Vec<usize> = dims.iter().enumerate()
            .filter(|&(d, _)| d != dim)
            .map(|(_, &s)| s)
            .collect();
        if out_dims.is_empty() { out_dims.push(1); }
        let mut out_shape = Shape::from(out_dims);

        if keepdim {
            let mut kd_dims = dims.to_vec();
            kd_dims[dim] = 1;
            out_shape = Shape::from(kd_dims);
        }

        let device = *x.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── cast ───────────────────────────────────────────────────────

    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor> {
        if x.dtype() == dtype {
            // Still materialise a contiguous copy in case layout is non-contiguous.
            return self.contiguous(x);
        }
        let cpu = x.storage().as_cpu()?;
        let layout = x.our_layout();
        let out_cpu = apply_cast(cpu, layout, dtype)?;
        let shape = layout.shape().clone();
        let device = *x.device();
        Ok(make_tensor(out_cpu, shape, &device))
    }

    // ── contiguous ─────────────────────────────────────────────────

    fn contiguous(&self, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() { return Ok(x.clone()); }
        let cpu = x.storage().as_cpu()?;
        let layout = x.our_layout();
        let out_cpu = apply_contiguous(cpu, layout);
        let shape = layout.shape().clone();
        let device = *x.device();
        Ok(make_tensor(out_cpu, shape, &device))
    }

    // ── matmul ─────────────────────────────────────────────────────

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_cpu = a.storage().as_cpu()?;
        let b_cpu = b.storage().as_cpu()?;
        let a_layout = a.our_layout();
        let b_layout = b.our_layout();

        macro_rules! mm_typed {
            ($av:expr, $bv:expr, $ty:ident) => {{
                let (out, out_shape) = apply_matmul_typed($av, a_layout, $bv, b_layout)?;
                (CpuStorage::$ty(out), out_shape)
            }};
        }

        let (out_cpu, out_shape) = match (a_cpu, b_cpu) {
            (CpuStorage::F32(av), CpuStorage::F32(bv)) => mm_typed!(av, bv, F32),
            (CpuStorage::F64(av), CpuStorage::F64(bv)) => mm_typed!(av, bv, F64),
            (CpuStorage::BF16(av), CpuStorage::BF16(bv)) => mm_typed!(av, bv, BF16),
            (CpuStorage::F16(av), CpuStorage::F16(bv)) => mm_typed!(av, bv, F16),
            _ => return Err(Error::DTypeMismatchBinaryOp { lhs: a_cpu.dtype(), rhs: b_cpu.dtype(), op: "matmul" }.bt()),
        };
        let device = *a.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── to_device ──────────────────────────────────────────────────

    fn to_device(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        if x.device() == device { return Ok(x.clone()); }
        // CPU→CPU is trivial (different device enum values but same storage type).
        if device.is_cpu() {
            let cpu = x.storage().as_cpu()?;
            let layout = x.our_layout();
            let out_cpu = apply_contiguous(cpu, layout);
            let shape = layout.shape().clone();
            return Ok(make_tensor(out_cpu, shape, device));
        }
        crate::bail!("DeviceTensorOps::to_device: CPU→GPU transfer not implemented here")
    }

    // ── index_select ───────────────────────────────────────────────

    fn index_select(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let x_cpu = x.storage().as_cpu()?;
        let idx_cpu = indices.storage().as_cpu()?;
        let x_layout = x.our_layout();
        let idx_layout = indices.our_layout();

        let num_idx = idx_layout.shape().elem_count();
        let mut out_dims = x_layout.dims().to_vec();
        out_dims[dim] = num_idx;
        let out_shape = Shape::from(out_dims);

        let out_cpu = apply_index_select(x_cpu, x_layout, idx_cpu, idx_layout, dim)?;
        let device = *x.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── gather ─────────────────────────────────────────────────────

    fn gather(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let x_cpu = x.storage().as_cpu()?;
        let idx_cpu = indices.storage().as_cpu()?;
        let x_layout = x.our_layout();
        let idx_layout = indices.our_layout();

        let out_shape = idx_layout.shape().clone();
        let out_cpu = apply_gather(x_cpu, x_layout, idx_cpu, idx_layout, dim)?;
        let device = *x.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── scatter_add ────────────────────────────────────────────────

    fn scatter_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        let x_cpu = x.storage().as_cpu()?;
        let idx_cpu = indices.storage().as_cpu()?;
        let src_cpu = src.storage().as_cpu()?;
        let x_layout = x.our_layout();
        let idx_layout = indices.our_layout();
        let src_layout = src.our_layout();

        let out_shape = x_layout.shape().clone();
        let out_cpu = apply_scatter_add(x_cpu, x_layout, idx_cpu, idx_layout, src_cpu, src_layout, dim)?;
        let device = *x.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── index_add ──────────────────────────────────────────────────

    fn index_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        let x_cpu = x.storage().as_cpu()?;
        let idx_cpu = indices.storage().as_cpu()?;
        let src_cpu = src.storage().as_cpu()?;
        let x_layout = x.our_layout();
        let idx_layout = indices.our_layout();
        let src_layout = src.our_layout();

        let out_shape = x_layout.shape().clone();
        let out_cpu = apply_index_add(x_cpu, x_layout, idx_cpu, idx_layout, src_cpu, src_layout, dim)?;
        let device = *x.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── where_cond ─────────────────────────────────────────────────

    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        let cond_cpu = cond.storage().as_cpu()?;
        let t_cpu = on_true.storage().as_cpu()?;
        let f_cpu = on_false.storage().as_cpu()?;
        let cond_layout = cond.our_layout();
        let t_layout = on_true.our_layout();
        let f_layout = on_false.our_layout();

        // Output shape is the broadcast of all three.
        let tf_shape = t_layout.shape().broadcast_shape_binary_op(f_layout.shape(), "where_cond")?;
        let out_shape = cond_layout.shape().broadcast_shape_binary_op(&tf_shape, "where_cond")?;

        // Broadcast all layouts to out_shape.
        let cond_bc = cond_layout.broadcast_as(&out_shape)?;
        let t_bc = t_layout.broadcast_as(&out_shape)?;
        let f_bc = f_layout.broadcast_as(&out_shape)?;

        let out_cpu = apply_where_cond(cond_cpu, &cond_bc, t_cpu, &t_bc, f_cpu, &f_bc)?;
        let device = *cond.device();
        Ok(make_tensor(out_cpu, out_shape, &device))
    }

    // ── cat ────────────────────────────────────────────────────────

    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        apply_cat(tensors, dim)
    }

    // ── affine ─────────────────────────────────────────────────────

    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Result<Tensor> {
        let cpu = x.storage().as_cpu()?;
        let layout = x.our_layout();
        let out_cpu = apply_affine(cpu, layout, mul, add);
        let shape = layout.shape().clone();
        let device = *x.device();
        Ok(make_tensor(out_cpu, shape, &device))
    }

    // ── zeros ──────────────────────────────────────────────────────

    fn zeros(&self, shape: &Shape, dtype: DType, device: &Device) -> Result<Tensor> {
        let n = shape.elem_count();
        let cpu = CpuStorage::zeros(dtype, n);
        Ok(make_tensor(cpu, shape.clone(), device))
    }

    // ── sort_last_dim ──────────────────────────────────────────────

    fn sort_last_dim(&self, x: &Tensor, asc: bool) -> Result<(Tensor, Tensor)> {
        let cpu = x.storage().as_cpu()?;
        let layout = x.our_layout();
        let shape = layout.shape().clone();
        let device = *x.device();

        let (sorted_cpu, idx_cpu) = apply_sort_last_dim(cpu, layout, asc)?;
        let sorted = make_tensor(sorted_cpu, shape.clone(), &device);
        let indices = make_tensor(idx_cpu, shape, &device);
        Ok((sorted, indices))
    }

    // ── data_ptr / data_ptr_mut ────────────────────────────────────

    unsafe fn data_ptr(&self, x: &Tensor) -> Result<*const u8> {
        let cpu = x.storage().as_cpu()?;
        let byte_offset = x.our_layout().start_offset() * x.dtype().size_in_bytes();
        let ptr = (cpu.as_bytes().as_ptr() as usize + byte_offset) as *const u8;
        // No need to forget anything — Arc keeps storage alive as long as Tensor exists
        Ok(ptr)
    }

    unsafe fn data_ptr_mut(&self, x: &Tensor) -> Result<*mut u8> {
        // Safety: caller guarantees exclusive access for mut operations
        let storage_ptr = Arc::as_ptr(x.storage_arc()) as *mut Storage;
        let storage_mut = unsafe { &mut *storage_ptr };
        let cpu = storage_mut.as_cpu_mut()?;
        let byte_offset = x.our_layout().start_offset() * x.dtype().size_in_bytes();
        let ptr = match cpu {
            CpuStorage::U8(v)   => v.as_mut_ptr() as *mut u8,
            CpuStorage::U32(v)  => v.as_mut_ptr() as *mut u8,
            CpuStorage::I32(v)  => v.as_mut_ptr() as *mut u8,
            CpuStorage::I64(v)  => v.as_mut_ptr() as *mut u8,
            CpuStorage::BF16(v) => v.as_mut_ptr() as *mut u8,
            CpuStorage::F16(v)  => v.as_mut_ptr() as *mut u8,
            CpuStorage::F32(v)  => v.as_mut_ptr() as *mut u8,
            CpuStorage::F64(v)  => v.as_mut_ptr() as *mut u8,
        };
        let ptr = (ptr as usize + byte_offset) as *mut u8;
        Ok(ptr)
    }
}

/// Static instance for the Device backend.
pub fn device_ops() -> &'static dyn Ops {
    static OPS: DeviceTensorOps = DeviceTensorOps;
    &OPS
}
