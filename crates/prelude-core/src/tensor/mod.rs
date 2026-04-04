//! Prelude Tensor abstraction layer.
//!
//! All model code imports from here — never from candle_core directly.
//! candle_core is used internally for compute (transitional).

// ── Own types (will fully replace candle re-exports) ────────────────

pub mod error;
pub mod layout;
pub mod shape;

// Re-export our own types.
pub use error::{DeviceLocation, Error, Result};
pub use shape::{D, Dim, Dims, Shape, ShapeWithOneHole};
pub use layout::Layout;

// ── Core types ──────────────────────────────────────────────────────

/// Element data type.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    I16,
    I32,
    I64,
    BF16,
    F16,
    F32,
    F64,
    F8E4M3,
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

    pub fn is_int(&self) -> bool {
        !self.is_float()
    }

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
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<DType> for candle_core::DType {
    fn from(d: DType) -> Self {
        match d {
            DType::U8 => Self::U8, DType::U32 => Self::U32,
            DType::I16 => Self::I16, DType::I32 => Self::I32, DType::I64 => Self::I64,
            DType::BF16 => Self::BF16, DType::F16 => Self::F16,
            DType::F32 => Self::F32, DType::F64 => Self::F64,
            DType::F8E4M3 => Self::F8E4M3,
        }
    }
}

impl From<candle_core::DType> for DType {
    fn from(d: candle_core::DType) -> Self {
        match d {
            candle_core::DType::U8 => Self::U8, candle_core::DType::U32 => Self::U32,
            candle_core::DType::I16 => Self::I16, candle_core::DType::I32 => Self::I32,
            candle_core::DType::I64 => Self::I64,
            candle_core::DType::BF16 => Self::BF16, candle_core::DType::F16 => Self::F16,
            candle_core::DType::F32 => Self::F32, candle_core::DType::F64 => Self::F64,
            candle_core::DType::F8E4M3 => Self::F8E4M3,
            _ => panic!("unsupported candle DType: {:?}", d),
        }
    }
}

pub use candle_core::Device;

// ── Shape bridge (our Shape ↔ candle Shape) ─────────────────────────

impl From<Shape> for candle_core::Shape {
    fn from(s: Shape) -> Self { candle_core::Shape::from(s.into_dims()) }
}
impl From<&candle_core::Shape> for Shape {
    fn from(s: &candle_core::Shape) -> Self { Shape::from(s.dims()) }
}
impl From<candle_core::Shape> for Shape {
    fn from(s: candle_core::Shape) -> Self { Shape::from(s.into_dims()) }
}

// Our Dim/Dims impl candle's Dim/Dims so they can be passed to candle methods.
impl candle_core::shape::Dim for D {
    fn to_index(&self, shape: &candle_core::Shape, op: &'static str) -> candle_core::Result<usize> {
        let our_shape = Shape::from(shape);
        Dim::to_index(self, &our_shape, op).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
    fn to_index_plus_one(&self, shape: &candle_core::Shape, op: &'static str) -> candle_core::Result<usize> {
        let our_shape = Shape::from(shape);
        Dim::to_index_plus_one(self, &our_shape, op).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

// Re-export bail! macro so `crate::tensor::bail!` works.
pub use crate::bail;

// ── Module trait ────────────────────────────────────────────────────
//
// Our own Module trait, using our Tensor type.
// Replaces candle_core::Module so all forward() returns our Tensor.

pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

// ── Tensor ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Tensor {
    inner: candle_core::Tensor,
    /// Cached shape — our own type so `shape()` returns `&Shape`.
    shape_cache: Shape,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

/// Shorthand for current_ops() — used by Tensor methods.
fn ops() -> &'static crate::ops::Ops {
    crate::ops::current_ops()
}

/// Wrap a candle tensor, caching our Shape.
fn wrap(t: candle_core::Tensor) -> Tensor {
    let shape_cache = Shape::from(t.shape());
    Tensor { inner: t, shape_cache }
}

fn wrap_r(r: candle_core::Result<candle_core::Tensor>) -> Result<Tensor> {
    Ok(wrap(r?))
}

// ── Construction ────────────────────────────────────────────────────

impl Tensor {
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(candle_core::Tensor::zeros(cs, candle_core::DType::from(dtype), device))
    }
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(candle_core::Tensor::ones(cs, candle_core::DType::from(dtype), device))
    }
    pub fn zeros_like(&self) -> Result<Self> {
        wrap_r(self.inner.zeros_like())
    }
    pub fn ones_like(&self) -> Result<Self> {
        wrap_r(self.inner.ones_like())
    }
    pub fn full<S: Into<Shape>>(val: f64, shape: S, device: &Device) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(candle_core::Tensor::full(val, cs, device))
    }
    pub fn from_vec<S: candle_core::shape::ShapeWithOneHole, T: candle_core::WithDType>(data: Vec<T>, shape: S, device: &Device) -> Result<Self> {
        wrap_r(candle_core::Tensor::from_vec(data, shape, device))
    }
    pub fn from_slice<S: candle_core::shape::ShapeWithOneHole, T: candle_core::WithDType>(data: &[T], shape: impl Into<candle_core::Shape>, device: &Device) -> Result<Self> {
        wrap_r(candle_core::Tensor::from_slice(data, shape, device))
    }
    pub fn new<A: candle_core::NdArray>(data: A, device: &Device) -> Result<Self> {
        wrap_r(candle_core::Tensor::new(data, device))
    }
    pub fn rand<S: Into<Shape>>(lo: f64, hi: f64, shape: S, device: &Device) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(candle_core::Tensor::rand(lo, hi, cs, device))
    }
    pub fn randn<S: Into<Shape>>(mean: f64, std: f64, shape: S, device: &Device) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(candle_core::Tensor::randn(mean, std, cs, device))
    }
    pub fn arange<T: candle_core::WithDType>(start: T, end: T, device: &Device) -> Result<Self> {
        wrap_r(candle_core::Tensor::arange(start, end, device))
    }
    pub fn cat<DD: Dim, T: std::borrow::Borrow<Self>>(tensors: &[T], dim: DD) -> Result<Self> {
        let first = tensors.first().map(|t| t.borrow());
        let shape = first.map(|t| t.shape_cache.clone()).unwrap_or_else(|| Shape::from(()));
        let dim = dim.to_index(&shape, "cat")?;
        let inner: Vec<&candle_core::Tensor> = tensors.iter().map(|t| &t.borrow().inner).collect();
        wrap_r(candle_core::Tensor::cat(&inner, dim))
    }
    pub fn stack<DD: Dim, T: std::borrow::Borrow<Self>>(tensors: &[T], dim: DD) -> Result<Self> {
        let first = tensors.first().map(|t| t.borrow());
        let shape = first.map(|t| t.shape_cache.clone()).unwrap_or_else(|| Shape::from(()));
        let dim_idx = dim.to_index_plus_one(&shape, "stack")?;
        let inner: Vec<&candle_core::Tensor> = tensors.iter().map(|t| &t.borrow().inner).collect();
        wrap_r(candle_core::Tensor::stack(&inner, dim_idx))
    }

    // ── Metadata ────────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape { &self.shape_cache }
    pub fn dims(&self) -> &[usize] { self.shape_cache.dims() }
    pub fn dim<DD: Dim>(&self, d: DD) -> Result<usize> { self.shape_cache.dim(d) }
    pub fn dims2(&self) -> Result<(usize, usize)> { self.shape_cache.dims2() }
    pub fn dims3(&self) -> Result<(usize, usize, usize)> { self.shape_cache.dims3() }
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> { self.shape_cache.dims4() }
    pub fn dtype(&self) -> DType { DType::from(self.inner.dtype()) }
    pub fn device(&self) -> &Device { self.inner.device() }
    pub fn rank(&self) -> usize { self.shape_cache.rank() }
    pub fn elem_count(&self) -> usize { self.shape_cache.elem_count() }
    pub fn is_contiguous(&self) -> bool { self.inner.is_contiguous() }
    pub fn layout(&self) -> &candle_core::Layout { self.inner.layout() }
    pub fn stride(&self) -> &[usize] { self.inner.stride() }
    pub fn id(&self) -> candle_core::TensorId { self.inner.id() }

    // ── View ops (no data copy) ─────────────────────────────────────

    pub fn reshape<S: candle_core::shape::ShapeWithOneHole>(&self, s: S) -> Result<Self> {
        self.inner.reshape(s)
        .map(wrap).map_err(Error::from)
    }
    pub fn narrow<D: candle_core::shape::Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        self.inner.narrow(dim, start, len)
        .map(wrap).map_err(Error::from)
    }
    pub fn unsqueeze<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { wrap_r(self.inner.unsqueeze(dim)) }
    pub fn squeeze<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { wrap_r(self.inner.squeeze(dim)) }
    pub fn transpose<D1: candle_core::shape::Dim, D2: candle_core::shape::Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        self.inner.transpose(d1, d2)
        .map(wrap).map_err(Error::from)
    }
    pub fn t(&self) -> Result<Self> { wrap_r(self.inner.t()) }
    pub fn contiguous(&self) -> Result<Self> { ops().tensor.contiguous(self) }
    pub fn flatten_all(&self) -> Result<Self> { wrap_r(self.inner.flatten_all()) }
    pub fn flatten<D1: candle_core::shape::Dim, D2: candle_core::shape::Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        self.inner.flatten(d1, d2)
        .map(wrap).map_err(Error::from)
    }
    pub fn chunk<DD: candle_core::shape::Dim>(&self, n: usize, dim: DD) -> Result<Vec<Self>> {
        Ok(self.inner.chunk(n, dim)?.into_iter().map(wrap).collect())
    }
    pub fn get(&self, index: usize) -> Result<Self> { wrap_r(self.inner.get(index)) }
    pub fn index_select<D: candle_core::shape::Dim>(&self, indices: &Self, dim: D) -> Result<Self> {
        self.inner.index_select(&indices.inner, dim)
        .map(wrap).map_err(Error::from)
    }
    pub fn gather<D: candle_core::shape::Dim>(&self, indices: &Self, dim: D) -> Result<Self> {
        self.inner.gather(&indices.inner, dim)
        .map(wrap).map_err(Error::from)
    }
    pub fn scatter_add<D: candle_core::shape::Dim>(&self, indices: &Self, source: &Self, dim: D) -> Result<Self> {
        self.inner.scatter_add(&indices.inner, &source.inner, dim)
        .map(wrap).map_err(Error::from)
    }
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(self.inner.broadcast_as(cs))
    }
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(self.inner.expand(cs))
    }
    pub fn pad_with_zeros<D: candle_core::shape::Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        self.inner.pad_with_zeros(dim, left, right)
        .map(wrap).map_err(Error::from)
    }
    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> { wrap_r(self.inner.repeat(repeats)) }
    pub fn broadcast_left<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let cs: candle_core::Shape = shape.into().into();
        wrap_r(self.inner.broadcast_left(cs))
    }

    // ── Conversion ──────────────────────────────────────────────────

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> { ops().tensor.cast(self, dtype) }
    pub fn to_device(&self, device: &Device) -> Result<Self> { ops().tensor.to_device(self, device) }
    pub fn to_vec0<T: candle_core::WithDType>(&self) -> Result<T> { Ok(self.inner.to_vec0()?) }
    pub fn to_vec1<T: candle_core::WithDType>(&self) -> Result<Vec<T>> { Ok(self.inner.to_vec1()?) }
    pub fn to_vec2<T: candle_core::WithDType>(&self) -> Result<Vec<Vec<T>>> { Ok(self.inner.to_vec2()?) }
    pub fn to_vec3<T: candle_core::WithDType>(&self) -> Result<Vec<Vec<Vec<T>>>> { Ok(self.inner.to_vec3()?) }
    pub fn to_scalar<T: candle_core::WithDType>(&self) -> Result<T> { Ok(self.inner.to_scalar()?) }

    // ── Compute ops (routed through TensorOps) ──────────────────────

    pub fn matmul(&self, rhs: &Self) -> Result<Self> { wrap_r(self.inner.matmul(&rhs.inner)) }
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> { ops().tensor.add(self, rhs) }
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> { ops().tensor.sub(self, rhs) }
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> { ops().tensor.mul(self, rhs) }
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> { ops().tensor.div(self, rhs) }
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> { ops().tensor.where_cond(self, on_true, on_false) }
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> { ops().tensor.affine(self, mul, add) }
    pub fn clamp<T: Into<f64>>(&self, min: T, max: T) -> Result<Self> { ops().tensor.clamp(self, min.into(), max.into()) }
    pub fn sum<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "sum")?;
        ops().tensor.sum(self, d, false)
    }
    pub fn sum_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "sum_keepdim")?;
        ops().tensor.sum(self, d, true)
    }
    pub fn sum_all(&self) -> Result<Self> { wrap_r(self.inner.sum_all()) }
    pub fn mean<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "mean")?;
        wrap_r(self.inner.mean(d))
    }
    pub fn mean_all(&self) -> Result<Self> { wrap_r(self.inner.mean_all()) }
    pub fn max<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "max")?;
        ops().tensor.max_reduce(self, d, false)
    }
    pub fn max_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "max_keepdim")?;
        ops().tensor.max_reduce(self, d, true)
    }
    pub fn min<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "min")?;
        ops().tensor.min_reduce(self, d, false)
    }
    pub fn min_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "min_keepdim")?;
        wrap_r(self.inner.min_keepdim(d))
    }
    pub fn argmax<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "argmax")?;
        ops().tensor.argmax(self, d)
    }
    pub fn argmin<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "argmin")?;
        ops().tensor.argmin(self, d)
    }
    pub fn mean_keepdim<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let d = dim.to_index(&self.shape_cache, "mean_keepdim")?;
        wrap_r(self.inner.mean_keepdim(d))
    }
    pub fn exp(&self) -> Result<Self> { ops().tensor.exp(self) }
    pub fn log(&self) -> Result<Self> { ops().tensor.log(self) }
    pub fn abs(&self) -> Result<Self> { ops().tensor.abs(self) }
    pub fn sqrt(&self) -> Result<Self> { ops().tensor.sqrt(self) }
    pub fn sqr(&self) -> Result<Self> { ops().tensor.sqr(self) }
    pub fn recip(&self) -> Result<Self> { ops().tensor.recip(self) }
    pub fn sin(&self) -> Result<Self> { ops().tensor.sin(self) }
    pub fn cos(&self) -> Result<Self> { ops().tensor.cos(self) }
    pub fn tanh(&self) -> Result<Self> { ops().tensor.tanh(self) }
    pub fn powf(&self, e: f64) -> Result<Self> { ops().tensor.powf(self, e) }
    pub fn relu(&self) -> Result<Self> { ops().tensor.unary(self, crate::ops::traits::UnaryOp::Relu) }
    pub fn gelu(&self) -> Result<Self> { wrap_r(self.inner.gelu()) }
    pub fn gelu_erf(&self) -> Result<Self> { wrap_r(self.inner.gelu_erf()) }
    pub fn silu(&self) -> Result<Self> { wrap_r(self.inner.silu()) }
    pub fn neg(&self) -> Result<Self> { ops().tensor.neg(self) }
    pub fn elu(&self, alpha: f64) -> Result<Self> { ops().tensor.elu(self, alpha) }
    pub fn minimum(&self, rhs: &Self) -> Result<Self> { ops().tensor.minimum(self, rhs) }
    pub fn maximum(&self, rhs: &Self) -> Result<Self> { ops().tensor.maximum(self, rhs) }
    pub fn sub(&self, rhs: &Self) -> Result<Self> { ops().tensor.sub(self, rhs) }

    // ── Comparison ops (routed through TensorOps) ───────────────────

    pub fn eq_t(&self, rhs: &Self) -> Result<Self> { ops().tensor.eq(self, rhs) }
    pub fn ne_t(&self, rhs: &Self) -> Result<Self> { ops().tensor.ne(self, rhs) }
    pub fn lt_t(&self, rhs: &Self) -> Result<Self> { ops().tensor.lt(self, rhs) }
    pub fn gt_t(&self, rhs: &Self) -> Result<Self> { ops().tensor.gt(self, rhs) }
    pub fn ge_t(&self, rhs: &Self) -> Result<Self> { ops().tensor.ge(self, rhs) }
    pub fn le_t(&self, rhs: &Self) -> Result<Self> { ops().tensor.le(self, rhs) }

    // Scalar comparison (still via candle — needs scalar broadcast support in TensorOps)
    pub fn ge<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { wrap_r(self.inner.ge(rhs)) }
    pub fn gt<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { wrap_r(self.inner.gt(rhs)) }
    pub fn le<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { wrap_r(self.inner.le(rhs)) }
    pub fn lt<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { wrap_r(self.inner.lt(rhs)) }
    pub fn eq_scalar<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { wrap_r(self.inner.eq(rhs)) }
    pub fn ne_scalar<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { wrap_r(self.inner.ne(rhs)) }

    pub fn softmax<DD: Dim>(&self, dim: DD) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        let max = self.max_keepdim(dim)?;
        let diff = self.broadcast_sub(&max)?;
        let num = diff.exp()?;
        let den = num.sum_keepdim(dim)?;
        num.broadcast_div(&den)
    }

    pub fn conv1d(&self, kernel: &Self, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.inner.conv1d(&kernel.inner, padding, stride, dilation, groups)
        .map(wrap).map_err(Error::from)
    }
    pub fn conv2d(&self, kernel: &Self, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.inner.conv2d(&kernel.inner, padding, stride, dilation, groups)
        .map(wrap).map_err(Error::from)
    }
    pub fn conv_transpose1d(&self, kernel: &Self, padding: usize, output_padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.inner.conv_transpose1d(&kernel.inner, padding, output_padding, stride, dilation, groups)
        .map(wrap).map_err(Error::from)
    }
    pub fn interpolate1d(&self, target_len: usize) -> Result<Self> { self.inner.interpolate1d(target_len).map(wrap).map_err(Error::from) }

    pub fn embedding(&self, ids: &Self) -> Result<Self> { wrap_r(self.inner.embedding(&ids.inner)) }

    pub fn index_add<D: candle_core::shape::Dim>(&self, ids: &Self, src: &Self, dim: D) -> Result<Self> {
        self.inner.index_add(&ids.inner, &src.inner, dim)
        .map(wrap).map_err(Error::from)
    }
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Self, Self)> {
        let (a, b) = self.inner.sort_last_dim(asc)?;
        Ok((wrap(a), wrap(b)))
    }
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Self> {
        self.inner.arg_sort_last_dim(asc)
        .map(wrap).map_err(Error::from)
    }

    /// In-place slice set. Copies `src` into `self[dim, offset..offset+src.size(dim)]`.
    pub fn slice_set<DD: candle_core::shape::Dim>(&self, src: &Self, dim: DD, start: usize) -> Result<()> {
        Ok(self.inner.slice_set(&src.inner, dim, start)?)
    }

    /// Slice-assign: returns a new tensor with `src` placed at `ranges`.
    pub fn slice_assign<D: std::ops::RangeBounds<usize>>(&self, ranges: &[D], src: &Self) -> Result<Self> {
        self.inner.slice_assign(ranges, &src.inner)
        .map(wrap).map_err(Error::from)
    }

    /// RoPE T-H-D variant (uses candle's custom op if available, else portable math).
    pub fn rope_thd(&self, cos: &Self, sin: &Self) -> Result<Self> {
        // Portable RoPE: x * cos + rotate_half(x) * sin
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
        let result = self.broadcast_mul(&cat_cos)?.broadcast_add(&rotated.broadcast_mul(&cat_sin)?)?;
        Ok(result)
    }

    /// Apply a Module, bridging our Tensor through forward().
    pub fn apply<M: Module + ?Sized>(&self, m: &M) -> Result<Self> {
        m.forward(self)
    }

    // ── Storage access (for device crates / FFI) ────────────────────

    pub fn storage_and_layout(&self) -> (std::sync::RwLockReadGuard<'_, candle_core::Storage>, &candle_core::Layout) {
        self.inner.storage_and_layout()
    }

    /// Mutable storage access (unsafe — caller must ensure exclusive access).
    pub unsafe fn storage_mut_and_layout(&self) -> (std::sync::RwLockWriteGuard<'_, candle_core::Storage>, &candle_core::Layout) {
        self.inner.storage_mut_and_layout()
    }

    /// Construct from raw candle storage (used by CUDA kernels).
    pub fn from_storage(
        storage: candle_core::Storage,
        shape: impl Into<candle_core::Shape>,
        op: candle_core::op::BackpropOp,
        is_variable: bool,
    ) -> Self {
        wrap(candle_core::Tensor::from_storage(storage, shape, op, is_variable))
    }

    // ── candle interop (temporary) ──────────────────────────────────

    /// Access the inner candle_core::Tensor.
    pub fn inner(&self) -> &candle_core::Tensor { &self.inner }

    /// Wrap a candle_core::Tensor.
    pub fn from_candle(t: candle_core::Tensor) -> Self { wrap(t) }

    /// Unwrap to candle_core::Tensor.
    pub fn into_candle(self) -> candle_core::Tensor { self.inner }
}

// ── From/Into candle_core::Tensor ────────────────────────────────────

impl From<candle_core::Tensor> for Tensor {
    fn from(t: candle_core::Tensor) -> Self { wrap(t) }
}

impl From<Tensor> for candle_core::Tensor {
    fn from(t: Tensor) -> Self { t.inner }
}

impl AsRef<candle_core::Tensor> for Tensor {
    fn as_ref(&self) -> &candle_core::Tensor { &self.inner }
}

impl std::borrow::Borrow<candle_core::Tensor> for Tensor {
    fn borrow(&self) -> &candle_core::Tensor { &self.inner }
}

// NOTE: Deref removed — all access to candle_core::Tensor goes through
// explicit methods or .inner(). This prevents candle types from leaking.

// ── Operator overloads ──────────────────────────────────────────────

// Tensor + Tensor — tries fused_add from current Ops context, falls back to candle.
impl std::ops::Add for Tensor {
    type Output = Result<Self>;
    fn add(self, rhs: Self) -> Result<Self> { &self + &rhs }
}
impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> {
        if let Some(result) = crate::ops::current_ops().fused.fused_add(self, rhs) {
            return result;
        }
        (&self.inner + &rhs.inner)
        .map(wrap).map_err(Error::from)
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
// Tensor + f64
impl std::ops::Add<f64> for Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { (self.inner + rhs).map(wrap).map_err(Error::from) }
}
impl std::ops::Add<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { (&self.inner + rhs).map(wrap).map_err(Error::from) }
}
// f64 + Tensor
impl std::ops::Add<Tensor> for f64 {
    type Output = Result<Tensor>;
    fn add(self, rhs: Tensor) -> Result<Tensor> { (self + rhs.inner).map(wrap).map_err(Error::from) }
}

// Tensor - Tensor
impl std::ops::Sub for Tensor {
    type Output = Result<Self>;
    fn sub(self, rhs: Self) -> Result<Self> { (self.inner - rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Sub for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { (&self.inner - &rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { (self.inner - &rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: Tensor) -> Result<Tensor> { (&self.inner - rhs.inner).map(wrap).map_err(Error::from) }
}
// Tensor - f64
impl std::ops::Sub<f64> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { (self.inner - rhs).map(wrap).map_err(Error::from) }
}
impl std::ops::Sub<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { (&self.inner - rhs).map(wrap).map_err(Error::from) }
}

// Tensor * Tensor
impl std::ops::Mul for Tensor {
    type Output = Result<Self>;
    fn mul(self, rhs: Self) -> Result<Self> { (self.inner * rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Mul for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { (&self.inner * &rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { (self.inner * &rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: Tensor) -> Result<Tensor> { (&self.inner * rhs.inner).map(wrap).map_err(Error::from) }
}
// Tensor * f64
impl std::ops::Mul<f64> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { (self.inner * rhs).map(wrap).map_err(Error::from) }
}
impl std::ops::Mul<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { (&self.inner * rhs).map(wrap).map_err(Error::from) }
}

// Tensor / Tensor
impl std::ops::Div for Tensor {
    type Output = Result<Self>;
    fn div(self, rhs: Self) -> Result<Self> { (self.inner / rhs.inner).map(wrap).map_err(Error::from) }
}
impl std::ops::Div for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: &Tensor) -> Result<Tensor> { (&self.inner / &rhs.inner).map(wrap).map_err(Error::from) }
}
// Tensor / f64
impl std::ops::Div<f64> for Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { (self.inner / rhs).map(wrap).map_err(Error::from) }
}
impl std::ops::Div<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { (&self.inner / rhs).map(wrap).map_err(Error::from) }
}

// -Tensor
impl std::ops::Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { wrap_r(self.inner.neg()) }
}
impl std::ops::Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { wrap_r(self.inner.neg()) }
}

// ── Display ─────────────────────────────────────────────────────────

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

// ── Backend internals (for device crates only, not model code) ──────

pub mod backend {
    pub use candle_core::backend::BackendStorage;
    pub use candle_core::Storage;
    pub use candle_core::CpuStorage;
    pub use candle_core::op::BackpropOp;
    pub use candle_core::Layout;
}

// Note: shape::{Dim, Dims, ShapeWithOneHole} are re-exported from tensor::shape at top of file.

pub mod safetensors {
    pub use candle_core::safetensors::*;
}

pub mod quantized {
    pub use candle_core::quantized::*;
}

// Re-export WithDType for use in generic code.
pub use candle_core::WithDType;
