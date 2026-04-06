//! CubeCL kernel implementations for TensorOps primitives.
//!
//! Follows burn-cubecl patterns:
//! - Op family traits: ONE kernel per category, compile-time dispatch via `comptime![]`
//! - LinearView: handles strided/broadcast indexing in the layout layer
//! - launch_unchecked + address_type="dynamic": efficient launch
//! - Proper vectorization via `tensor_vector_size_parallel`

use cubecl::prelude::*;
use cubecl::num_traits::{One, Zero};
use cubecl::std::tensor::layout::linear::{
    LinearView, LinearViewLayoutLaunch, LinearViewLaunch,
};

use crate::tensor::DType;

// ── DType → StorageType mapping ───────────────────────────────────

pub fn dtype_to_storage(dtype: DType) -> cubecl::ir::StorageType {
    use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind, StorageType};
    StorageType::Scalar(match dtype {
        DType::F32 => ElemType::Float(FloatKind::F32),
        DType::F64 => ElemType::Float(FloatKind::F64),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        DType::F8E4M3 => ElemType::Float(FloatKind::Flex32),
        DType::U8 => ElemType::UInt(UIntKind::U8),
        DType::U32 => ElemType::UInt(UIntKind::U32),
        DType::I16 => ElemType::Int(IntKind::I16),
        DType::I32 => ElemType::Int(IntKind::I32),
        DType::I64 => ElemType::Int(IntKind::I64),
    })
}

// ══════════════════════════════════════════════════════════════════
// Unary op family — one kernel for all float unary ops
// ══════════════════════════════════════════════════════════════════

pub(crate) trait FloatUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<F: Float, N: Size>: FloatUnaryOp<F, N, Options = Self::Options>;
}

#[cube]
pub(crate) trait FloatUnaryOp<F: Float, N: Size>: 'static + Send + Sync {
    type Options: LaunchArg;
    fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N>;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_unary_float<F: Float, N: Size, O: FloatUnaryOpFamily>(
    input: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
    options: &O::Options,
    #[define(F)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    output[ABSOLUTE_POS] = O::Unary::<F, N>::execute(input[ABSOLUTE_POS], options);
}

// ── Concrete: basic float unary ops (no extra args) ──────────────

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BasicFloatUnaryKind {
    Exp, Log, Sqrt, Abs, Neg, Recip, Sin, Cos, Tanh,
    Ceil, Floor, Round, Erf, Sqr, Relu, Silu, Gelu, GeluErf, Sign,
}

#[derive(CubeLaunch, CubeType)]
struct BasicFloatUnaryOptions {
    #[cube(comptime)]
    kind: BasicFloatUnaryKind,
}

struct BasicFloatUnary;

#[cube]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for BasicFloatUnary {
    type Options = BasicFloatUnaryOptions;

    fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N> {
        match comptime![options.kind] {
            BasicFloatUnaryKind::Exp => Vector::exp(input),
            BasicFloatUnaryKind::Log => Vector::ln(input),
            BasicFloatUnaryKind::Sqrt => Vector::sqrt(input),
            BasicFloatUnaryKind::Abs => Vector::abs(input),
            BasicFloatUnaryKind::Neg => {
                let zero = Vector::zero();
                zero - input
            }
            BasicFloatUnaryKind::Recip => Vector::recip(input),
            BasicFloatUnaryKind::Sin => Vector::sin(input),
            BasicFloatUnaryKind::Cos => Vector::cos(input),
            BasicFloatUnaryKind::Tanh => Vector::tanh(input),
            BasicFloatUnaryKind::Ceil => Vector::ceil(input),
            BasicFloatUnaryKind::Floor => Vector::floor(input),
            BasicFloatUnaryKind::Round => Vector::round(input),
            BasicFloatUnaryKind::Erf => Vector::erf(input),
            BasicFloatUnaryKind::Sqr => {
                input * input
            }
            BasicFloatUnaryKind::Relu => {
                let zero = Vector::zero();
                select_many(input.greater_than(zero), input, zero)
            }
            BasicFloatUnaryKind::Silu => {
                let one = Vector::one();
                let zero = Vector::zero();
                input / (one + Vector::exp(zero - input))
            }
            BasicFloatUnaryKind::Gelu => {
                let half = Vector::new(F::new(0.5));
                let one = Vector::one();
                let coeff = Vector::new(F::new(0.044715));
                let sqrt_2_pi = Vector::new(F::new(0.7978845608));
                let inner = sqrt_2_pi * (input + coeff * input * input * input);
                half * input * (one + Vector::tanh(inner))
            }
            BasicFloatUnaryKind::GeluErf => {
                let half = Vector::new(F::new(0.5));
                let one = Vector::one();
                let inv_sqrt2 = Vector::new(F::new(0.7071067811865476));
                half * input * (one + Vector::erf(input * inv_sqrt2))
            }
            BasicFloatUnaryKind::Sign => {
                let zero = Vector::zero();
                let one = Vector::one();
                let neg_one = Vector::new(F::new(-1.0));
                let pos = select_many(input.greater_than(zero), one, zero);
                select_many(input.less_than(zero), neg_one, pos)
            }
        }
    }
}

impl FloatUnaryOpFamily for BasicFloatUnary {
    type Options = BasicFloatUnaryOptions;
    type Unary<F: Float, N: Size> = Self;
}

/// Public launch helper for basic float unary ops.
pub(crate) fn launch_unary_float<R: Runtime>(
    client: &cubecl::client::ComputeClient<R>,
    input: LinearViewLaunch<R>,
    output: LinearViewLaunch<R>,
    kind: BasicFloatUnaryKind,
    num_elems: usize,
    dtype: cubecl::ir::StorageType,
) {
    let vector_size = 1; // TODO: proper vectorization
    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, working_units, cube_dim);

    unsafe {
        kernel_unary_float::launch_unchecked::<BasicFloatUnary, R>(
            client,
            cube_count,
            cube_dim,
            cubecl::ir::AddressType::default(),
            vector_size,
            input,
            output,
            BasicFloatUnaryOptionsLaunch::new(kind),
            dtype,
        );
    }
}

// ══════════════════════════════════════════════════════════════════
// Binary op family — one kernel for all numeric binary ops
// ══════════════════════════════════════════════════════════════════

pub(crate) trait BinaryOpFamily: Send + Sync + 'static {
    type BinaryOp<C: Numeric, N: Size>: BinaryOp<C, N>;
}

#[cube]
pub(crate) trait BinaryOp<C: Numeric, N: Size>: 'static + Send + Sync {
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> Vector<C, N>;
}

pub(crate) struct AddOp;
pub(crate) struct SubOp;
pub(crate) struct MulOp;
pub(crate) struct DivOp;
pub(crate) struct MinOp;
pub(crate) struct MaxOp;

impl BinaryOpFamily for AddOp { type BinaryOp<C: Numeric, N: Size> = Self; }
impl BinaryOpFamily for SubOp { type BinaryOp<C: Numeric, N: Size> = Self; }
impl BinaryOpFamily for MulOp { type BinaryOp<C: Numeric, N: Size> = Self; }
impl BinaryOpFamily for DivOp { type BinaryOp<C: Numeric, N: Size> = Self; }
impl BinaryOpFamily for MinOp { type BinaryOp<C: Numeric, N: Size> = Self; }
impl BinaryOpFamily for MaxOp { type BinaryOp<C: Numeric, N: Size> = Self; }

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for AddOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> { lhs + rhs }
}
#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for SubOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> { lhs - rhs }
}
#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for MulOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> { lhs * rhs }
}
#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for DivOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> { lhs / rhs }
}
#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for MinOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        select_many(lhs.less_than(rhs), lhs, rhs)
    }
}
#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for MaxOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        select_many(lhs.greater_than(rhs), lhs, rhs)
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_binop<C: Numeric, N: Size, O: BinaryOpFamily>(
    lhs: &LinearView<Vector<C, N>>,
    rhs: &LinearView<Vector<C, N>>,
    out: &mut LinearView<Vector<C, N>, ReadWrite>,
    #[define(C)] _dtype: StorageType,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    out[ABSOLUTE_POS] = O::BinaryOp::<C, N>::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
}

/// Launch a binary op. Inputs may have different shapes (broadcasting handled by LinearView).
pub(crate) fn launch_binop<R: Runtime, O: BinaryOpFamily>(
    client: &cubecl::client::ComputeClient<R>,
    lhs: LinearViewLaunch<R>,
    rhs: LinearViewLaunch<R>,
    output: LinearViewLaunch<R>,
    num_elems: usize,
    dtype: cubecl::ir::StorageType,
) {
    let vector_size = 1; // TODO: proper vectorization
    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, working_units, cube_dim);

    unsafe {
        kernel_binop::launch_unchecked::<O, R>(
            client, cube_count, cube_dim,
            cubecl::ir::AddressType::default(),
            vector_size,
            lhs, rhs, output, dtype,
        );
    }
}

// ══════════════════════════════════════════════════════════════════
// Comparison op family — one kernel for all comparison ops
// ══════════════════════════════════════════════════════════════════

pub(crate) trait ComparisonOpFamily: 'static + Send + Sync {
    type Operation<T: Numeric, N: Size>: ComparisonOp<T, N>;
}

#[cube]
pub(crate) trait ComparisonOp<C: Numeric, N: Size>: 'static + Send + Sync {
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> bool;
}

pub(crate) struct EqOp;
pub(crate) struct NeOp;
pub(crate) struct LtOp;
pub(crate) struct GtOp;
pub(crate) struct GeOp;
pub(crate) struct LeOp;

impl ComparisonOpFamily for EqOp { type Operation<T: Numeric, N: Size> = Self; }
impl ComparisonOpFamily for NeOp { type Operation<T: Numeric, N: Size> = Self; }
impl ComparisonOpFamily for LtOp { type Operation<T: Numeric, N: Size> = Self; }
impl ComparisonOpFamily for GtOp { type Operation<T: Numeric, N: Size> = Self; }
impl ComparisonOpFamily for GeOp { type Operation<T: Numeric, N: Size> = Self; }
impl ComparisonOpFamily for LeOp { type Operation<T: Numeric, N: Size> = Self; }

#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for EqOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool { lhs == rhs }
}
#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for NeOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool { lhs != rhs }
}
#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for LtOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool { lhs < rhs }
}
#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for GtOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool { lhs > rhs }
}
#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for GeOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool { lhs >= rhs }
}
#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for LeOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool { lhs <= rhs }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_cmp<T: Numeric, Bool: Numeric, N: Size, O: ComparisonOpFamily>(
    lhs: &LinearView<Vector<T, N>>,
    rhs: &LinearView<Vector<T, N>>,
    out: &mut LinearView<Vector<Bool, N>, ReadWrite>,
    #[define(T, Bool)] _dtypes: [StorageType; 2],
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    out[ABSOLUTE_POS] = Vector::cast_from(O::Operation::<T, N>::execute(
        lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS],
    ));
}

/// Launch a comparison op. Returns U32 tensor.
pub(crate) fn launch_cmp<R: Runtime, O: ComparisonOpFamily>(
    client: &cubecl::client::ComputeClient<R>,
    lhs: LinearViewLaunch<R>,
    rhs: LinearViewLaunch<R>,
    output: LinearViewLaunch<R>,
    num_elems: usize,
    input_dtype: cubecl::ir::StorageType,
    output_dtype: cubecl::ir::StorageType,
) {
    let vector_size = 1; // TODO: proper vectorization
    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, working_units, cube_dim);

    unsafe {
        kernel_cmp::launch_unchecked::<O, R>(
            client, cube_count, cube_dim,
            cubecl::ir::AddressType::default(),
            vector_size,
            lhs, rhs, output,
            [input_dtype, output_dtype],
        );
    }
}

// ══════════════════════════════════════════════════════════════════
// Affine: output = input * mul + add (scalar parameters)
// ══════════════════════════════════════════════════════════════════

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_affine<F: Float, N: Size>(
    input: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
    mul: &Tensor<F>,
    add: &Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let mul_v = Vector::new(mul[0]);
    let add_v = Vector::new(add[0]);
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * mul_v + add_v;
}

// ══════════════════════════════════════════════════════════════════
// Cast: dtype conversion
// ══════════════════════════════════════════════════════════════════

#[cube(launch)]
pub fn kernel_cast<I: Numeric, O: Numeric>(
    input: &Tensor<I>,
    output: &mut Tensor<O>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = O::cast_from(input[ABSOLUTE_POS]);
}

// ══════════════════════════════════════════════════════════════════
// Where/conditional select
// ══════════════════════════════════════════════════════════════════

#[cube(launch)]
pub fn kernel_where<F: Float>(
    cond: &Tensor<u32>,
    on_true: &Tensor<F>,
    on_false: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let c = cond[ABSOLUTE_POS];
    output[ABSOLUTE_POS] = select(c != 0u32, on_true[ABSOLUTE_POS], on_false[ABSOLUTE_POS]);
}

// ══════════════════════════════════════════════════════════════════
// Index/gather/scatter ops — scalar, handle arbitrary rank
// ══════════════════════════════════════════════════════════════════

#[cube(launch)]
pub fn kernel_index_select<T: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<u32>,
    output: &mut Tensor<T>,
    #[comptime] dim: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut remaining = ABSOLUTE_POS;
    let mut input_offset = 0;
    let rank = output.rank();

    for d in 0..rank {
        let stride = output.stride(d);
        let coord = remaining / stride;
        remaining = remaining % stride;

        if d == comptime![dim as usize] {
            let idx_coord = coord % indices.shape(0);
            let src_coord = usize::cast_from(indices[idx_coord]);
            input_offset += src_coord * input.stride(d);
        } else {
            input_offset += coord * input.stride(d);
        }
    }

    output[ABSOLUTE_POS] = input[input_offset];
}

#[cube(launch)]
pub fn kernel_gather<T: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<u32>,
    output: &mut Tensor<T>,
    #[comptime] dim: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut remaining = ABSOLUTE_POS;
    let mut input_offset = 0;
    let rank = output.rank();

    for d in 0..rank {
        let stride = output.stride(d);
        let coord = remaining / stride;
        remaining = remaining % stride;

        if d == comptime![dim as usize] {
            let src_coord = usize::cast_from(indices[ABSOLUTE_POS]);
            input_offset += src_coord * input.stride(d);
        } else {
            input_offset += coord * input.stride(d);
        }
    }

    output[ABSOLUTE_POS] = input[input_offset];
}

#[cube(launch)]
pub fn kernel_scatter_add<T: Numeric>(
    input: &mut Tensor<T>,
    indices: &Tensor<u32>,
    src: &Tensor<T>,
    #[comptime] dim: u32,
) {
    if ABSOLUTE_POS >= src.len() {
        terminate!();
    }

    let mut remaining = ABSOLUTE_POS;
    let mut input_offset = 0;
    let rank = src.rank();

    for d in 0..rank {
        let stride = src.stride(d);
        let coord = remaining / stride;
        remaining = remaining % stride;

        if d == comptime![dim as usize] {
            // scatter_add: indices has same shape as src
            let dst_coord = usize::cast_from(indices[ABSOLUTE_POS]);
            input_offset += dst_coord * input.stride(d);
        } else {
            input_offset += coord * input.stride(d);
        }
    }

    input[input_offset] = input[input_offset] + src[ABSOLUTE_POS];
}

/// index_add: indices is 1D (length == src.shape[dim]).
/// For each element in src, maps its dim-coordinate via indices to the dst dim-coordinate.
#[cube(launch)]
pub fn kernel_index_add<T: Numeric>(
    dst: &mut Tensor<T>,
    indices: &Tensor<u32>,
    src: &Tensor<T>,
    #[comptime] dim: u32,
) {
    if ABSOLUTE_POS >= src.len() {
        terminate!();
    }

    let rank = src.rank();
    let mut remaining = ABSOLUTE_POS;
    let mut dst_offset = 0;

    for d in 0..rank {
        let stride = src.stride(d);
        let coord = remaining / stride;
        remaining = remaining % stride;

        if d == comptime![dim as usize] {
            // index_add: indices is 1D, index by coordinate in scatter dim
            let dst_coord = usize::cast_from(indices[coord]);
            dst_offset += dst_coord * dst.stride(d);
        } else {
            dst_offset += coord * dst.stride(d);
        }
    }

    dst[dst_offset] = dst[dst_offset] + src[ABSOLUTE_POS];
}

#[cube(launch)]
pub fn kernel_slice_assign<T: Numeric>(
    input: &mut Tensor<T>,
    value: &Tensor<T>,
    offsets: &Tensor<u32>,
) {
    if ABSOLUTE_POS >= value.len() {
        terminate!();
    }

    let rank = value.rank();
    let mut remaining = ABSOLUTE_POS;
    let mut input_offset = 0;

    for d in 0..rank {
        let stride = value.stride(d);
        let coord = remaining / stride;
        remaining = remaining % stride;
        let input_coord = coord + usize::cast_from(offsets[d]);
        input_offset += input_coord * input.stride(d);
    }

    input[input_offset] = value[ABSOLUTE_POS];
}

// Note: flat copy is now handled by cubecl::std::tensor::copy_into()
