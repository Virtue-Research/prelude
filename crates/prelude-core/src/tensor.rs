//! Prelude Tensor abstraction layer.
//!
//! All model code imports from here — never from candle_core directly.
//! Each type wraps candle_core internally (temporary — will be replaced by DeviceBuffer).

// ── Core types (re-export candle's for now, replace incrementally) ──

pub use candle_core::DType;
pub use candle_core::Device;
pub use candle_core::Shape;
pub use candle_core::D;

// ── Error handling ──────────────────────────────────────────────────

pub use candle_core::Error;
pub use candle_core::Result;
pub use candle_core::bail;

// ── Module trait ────────────────────────────────────────────────────
//
// Our own Module trait, using our Tensor type.
// Replaces candle_core::Module so all forward() returns our Tensor.

pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

// ── Tensor ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Tensor(candle_core::Tensor);

// ── Construction ────────────────────────────────────────────────────

impl Tensor {
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        candle_core::Tensor::zeros(shape, dtype, device).map(Self)
    }
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        candle_core::Tensor::ones(shape, dtype, device).map(Self)
    }
    pub fn zeros_like(&self) -> Result<Self> {
        self.0.zeros_like().map(Self)
    }
    pub fn ones_like(&self) -> Result<Self> {
        self.0.ones_like().map(Self)
    }
    pub fn full<S: Into<Shape>>(val: f64, shape: S, device: &Device) -> Result<Self> {
        candle_core::Tensor::full(val, shape, device).map(Self)
    }
    pub fn from_vec<S: candle_core::shape::ShapeWithOneHole, D: candle_core::WithDType>(data: Vec<D>, shape: S, device: &Device) -> Result<Self> {
        candle_core::Tensor::from_vec(data, shape, device).map(Self)
    }
    pub fn from_slice<S: candle_core::shape::ShapeWithOneHole, D: candle_core::WithDType>(data: &[D], shape: impl Into<Shape>, device: &Device) -> Result<Self> {
        candle_core::Tensor::from_slice(data, shape, device).map(Self)
    }
    pub fn new<A: candle_core::NdArray>(data: A, device: &Device) -> Result<Self> {
        candle_core::Tensor::new(data, device).map(Self)
    }
    pub fn rand<S: Into<Shape>>(lo: f64, hi: f64, shape: S, device: &Device) -> Result<Self> {
        candle_core::Tensor::rand(lo, hi, shape, device).map(Self)
    }
    pub fn randn<S: Into<Shape>>(mean: f64, std: f64, shape: S, device: &Device) -> Result<Self> {
        candle_core::Tensor::randn(mean, std, shape, device).map(Self)
    }
    pub fn arange<D: candle_core::WithDType>(start: D, end: D, device: &Device) -> Result<Self> {
        candle_core::Tensor::arange(start, end, device).map(Self)
    }
    pub fn cat<D: candle_core::shape::Dim, T: std::borrow::Borrow<Self>>(tensors: &[T], dim: D) -> Result<Self> {
        let first = tensors.first().map(|t| t.borrow());
        let shape = first.map(|t| t.0.shape().clone()).unwrap_or_else(|| Shape::from(()));
        let dim = dim.to_index(&shape, "cat")?;
        let inner: Vec<&candle_core::Tensor> = tensors.iter().map(|t| &t.borrow().0).collect();
        candle_core::Tensor::cat(&inner, dim).map(Self)
    }
    pub fn stack<D: candle_core::shape::Dim, T: std::borrow::Borrow<Self>>(tensors: &[T], dim: D) -> Result<Self> {
        let first = tensors.first().map(|t| t.borrow());
        let shape = first.map(|t| t.0.shape().clone()).unwrap_or_else(|| Shape::from(()));
        let dim_idx = dim.to_index_plus_one(&shape, "stack")?;
        let inner: Vec<&candle_core::Tensor> = tensors.iter().map(|t| &t.borrow().0).collect();
        candle_core::Tensor::stack(&inner, dim_idx).map(Self)
    }

    // ── Metadata ────────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape { self.0.shape() }
    pub fn dims(&self) -> &[usize] { self.0.dims() }
    pub fn dim<D: candle_core::shape::Dim>(&self, d: D) -> Result<usize> { self.0.dim(d) }
    pub fn dims2(&self) -> Result<(usize, usize)> { self.0.dims2() }
    pub fn dims3(&self) -> Result<(usize, usize, usize)> { self.0.dims3() }
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> { self.0.dims4() }
    pub fn dtype(&self) -> DType { self.0.dtype() }
    pub fn device(&self) -> &Device { self.0.device() }
    pub fn rank(&self) -> usize { self.0.rank() }
    pub fn elem_count(&self) -> usize { self.0.elem_count() }
    pub fn is_contiguous(&self) -> bool { self.0.is_contiguous() }
    pub fn layout(&self) -> &candle_core::Layout { self.0.layout() }
    pub fn stride(&self) -> &[usize] { self.0.stride() }
    pub fn id(&self) -> candle_core::TensorId { self.0.id() }

    // ── View ops (no data copy) ─────────────────────────────────────

    pub fn reshape<S: candle_core::shape::ShapeWithOneHole>(&self, s: S) -> Result<Self> {
        self.0.reshape(s).map(Self)
    }
    pub fn narrow<D: candle_core::shape::Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        self.0.narrow(dim, start, len).map(Self)
    }
    pub fn unsqueeze<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.unsqueeze(dim).map(Self) }
    pub fn squeeze<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.squeeze(dim).map(Self) }
    pub fn transpose<D1: candle_core::shape::Dim, D2: candle_core::shape::Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        self.0.transpose(d1, d2).map(Self)
    }
    pub fn t(&self) -> Result<Self> { self.0.t().map(Self) }
    pub fn contiguous(&self) -> Result<Self> { self.0.contiguous().map(Self) }
    pub fn flatten_all(&self) -> Result<Self> { self.0.flatten_all().map(Self) }
    pub fn flatten<D1: candle_core::shape::Dim, D2: candle_core::shape::Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        self.0.flatten(d1, d2).map(Self)
    }
    pub fn chunk<D: candle_core::shape::Dim>(&self, n: usize, dim: D) -> Result<Vec<Self>> {
        self.0.chunk(n, dim).map(|v| v.into_iter().map(Self).collect())
    }
    pub fn get(&self, index: usize) -> Result<Self> { self.0.get(index).map(Self) }
    pub fn index_select<D: candle_core::shape::Dim>(&self, indices: &Self, dim: D) -> Result<Self> {
        self.0.index_select(&indices.0, dim).map(Self)
    }
    pub fn gather<D: candle_core::shape::Dim>(&self, indices: &Self, dim: D) -> Result<Self> {
        self.0.gather(&indices.0, dim).map(Self)
    }
    pub fn scatter_add<D: candle_core::shape::Dim>(&self, indices: &Self, source: &Self, dim: D) -> Result<Self> {
        self.0.scatter_add(&indices.0, &source.0, dim).map(Self)
    }
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> { self.0.broadcast_as(shape).map(Self) }
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> { self.0.expand(shape).map(Self) }
    pub fn pad_with_zeros<D: candle_core::shape::Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        self.0.pad_with_zeros(dim, left, right).map(Self)
    }
    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> { self.0.repeat(repeats).map(Self) }
    pub fn broadcast_left<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.0.broadcast_left(shape).map(Self)
    }

    // ── Conversion ──────────────────────────────────────────────────

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> { self.0.to_dtype(dtype).map(Self) }
    pub fn to_device(&self, device: &Device) -> Result<Self> { self.0.to_device(device).map(Self) }
    pub fn to_vec0<T: candle_core::WithDType>(&self) -> Result<T> { self.0.to_vec0() }
    pub fn to_vec1<T: candle_core::WithDType>(&self) -> Result<Vec<T>> { self.0.to_vec1() }
    pub fn to_vec2<T: candle_core::WithDType>(&self) -> Result<Vec<Vec<T>>> { self.0.to_vec2() }
    pub fn to_vec3<T: candle_core::WithDType>(&self) -> Result<Vec<Vec<Vec<T>>>> { self.0.to_vec3() }
    pub fn to_scalar<T: candle_core::WithDType>(&self) -> Result<T> { self.0.to_scalar() }

    // ── Compute ops ─────────────────────────────────────────────────

    pub fn matmul(&self, rhs: &Self) -> Result<Self> { self.0.matmul(&rhs.0).map(Self) }
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> { self.0.broadcast_add(&rhs.0).map(Self) }
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> { self.0.broadcast_sub(&rhs.0).map(Self) }
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> { self.0.broadcast_mul(&rhs.0).map(Self) }
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> { self.0.broadcast_div(&rhs.0).map(Self) }
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        self.0.where_cond(&on_true.0, &on_false.0).map(Self)
    }
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> { self.0.affine(mul, add).map(Self) }
    pub fn clamp<T: Into<f64>>(&self, min: T, max: T) -> Result<Self> { self.0.clamp(min.into(), max.into()).map(Self) }
    pub fn sum<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.sum(dim).map(Self) }
    pub fn sum_keepdim<D: candle_core::shape::Dims>(&self, dim: D) -> Result<Self> { self.0.sum_keepdim(dim).map(Self) }
    pub fn sum_all(&self) -> Result<Self> { self.0.sum_all().map(Self) }
    pub fn mean<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.mean(dim).map(Self) }
    pub fn mean_all(&self) -> Result<Self> { self.0.mean_all().map(Self) }
    pub fn max<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.max(dim).map(Self) }
    pub fn max_keepdim<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.max_keepdim(dim).map(Self) }
    pub fn min<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.min(dim).map(Self) }
    pub fn min_keepdim<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.min_keepdim(dim).map(Self) }
    pub fn argmax<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.argmax(dim).map(Self) }
    pub fn argmin<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.argmin(dim).map(Self) }
    pub fn mean_keepdim<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> { self.0.mean_keepdim(dim).map(Self) }
    pub fn exp(&self) -> Result<Self> { self.0.exp().map(Self) }
    pub fn log(&self) -> Result<Self> { self.0.log().map(Self) }
    pub fn abs(&self) -> Result<Self> { self.0.abs().map(Self) }
    pub fn sqrt(&self) -> Result<Self> { self.0.sqrt().map(Self) }
    pub fn sqr(&self) -> Result<Self> { self.0.sqr().map(Self) }
    pub fn recip(&self) -> Result<Self> { self.0.recip().map(Self) }
    pub fn sin(&self) -> Result<Self> { self.0.sin().map(Self) }
    pub fn cos(&self) -> Result<Self> { self.0.cos().map(Self) }
    pub fn tanh(&self) -> Result<Self> { self.0.tanh().map(Self) }
    pub fn powf(&self, e: f64) -> Result<Self> { self.0.powf(e).map(Self) }
    pub fn relu(&self) -> Result<Self> { self.0.relu().map(Self) }
    pub fn gelu(&self) -> Result<Self> { self.0.gelu().map(Self) }
    pub fn gelu_erf(&self) -> Result<Self> { self.0.gelu_erf().map(Self) }
    pub fn silu(&self) -> Result<Self> { self.0.silu().map(Self) }
    pub fn neg(&self) -> Result<Self> { self.0.neg().map(Self) }
    pub fn elu(&self, alpha: f64) -> Result<Self> { self.0.elu(alpha).map(Self) }
    pub fn minimum(&self, rhs: &Self) -> Result<Self> { self.0.minimum(&rhs.0).map(Self) }
    pub fn maximum(&self, rhs: &Self) -> Result<Self> { self.0.maximum(&rhs.0).map(Self) }
    pub fn sub(&self, rhs: &Self) -> Result<Self> { self.0.sub(&rhs.0).map(Self) }

    // ── Comparison ops ──────────────────────────────────────────────

    pub fn eq_t(&self, rhs: &Self) -> Result<Self> { self.0.eq(&rhs.0).map(Self) }
    pub fn ne_t(&self, rhs: &Self) -> Result<Self> { self.0.ne(&rhs.0).map(Self) }
    pub fn lt_t(&self, rhs: &Self) -> Result<Self> { self.0.lt(&rhs.0).map(Self) }
    pub fn gt_t(&self, rhs: &Self) -> Result<Self> { self.0.gt(&rhs.0).map(Self) }
    pub fn ge_t(&self, rhs: &Self) -> Result<Self> { self.0.ge(&rhs.0).map(Self) }
    pub fn le_t(&self, rhs: &Self) -> Result<Self> { self.0.le(&rhs.0).map(Self) }

    pub fn ge<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { self.0.ge(rhs).map(Self) }
    pub fn gt<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { self.0.gt(rhs).map(Self) }
    pub fn le<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { self.0.le(rhs).map(Self) }
    pub fn lt<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { self.0.lt(rhs).map(Self) }
    pub fn eq_scalar<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { self.0.eq(rhs).map(Self) }
    pub fn ne_scalar<T: candle_core::WithDType>(&self, rhs: T) -> Result<Self> { self.0.ne(rhs).map(Self) }

    pub fn softmax<D: candle_core::shape::Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        let max = self.max_keepdim(dim)?;
        let diff = self.broadcast_sub(&max)?;
        let num = diff.exp()?;
        let den = num.sum_keepdim(dim)?;
        num.broadcast_div(&den)
    }

    pub fn conv1d(&self, kernel: &Self, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.0.conv1d(&kernel.0, padding, stride, dilation, groups).map(Self)
    }
    pub fn conv2d(&self, kernel: &Self, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.0.conv2d(&kernel.0, padding, stride, dilation, groups).map(Self)
    }
    pub fn conv_transpose1d(&self, kernel: &Self, padding: usize, output_padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        self.0.conv_transpose1d(&kernel.0, padding, output_padding, stride, dilation, groups).map(Self)
    }
    pub fn interpolate1d(&self, target_len: usize) -> Result<Self> { self.0.interpolate1d(target_len).map(Self) }

    pub fn embedding(&self, ids: &Self) -> Result<Self> { self.0.embedding(&ids.0).map(Self) }

    pub fn index_add<D: candle_core::shape::Dim>(&self, ids: &Self, src: &Self, dim: D) -> Result<Self> {
        self.0.index_add(&ids.0, &src.0, dim).map(Self)
    }
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Self, Self)> {
        self.0.sort_last_dim(asc).map(|(a, b)| (Self(a), Self(b)))
    }
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Self> {
        self.0.arg_sort_last_dim(asc).map(Self)
    }

    /// In-place slice set. Copies `src` into `self[dim, offset..offset+src.size(dim)]`.
    pub fn slice_set<D: candle_core::shape::Dim>(&self, src: &Self, dim: D, start: usize) -> Result<()> {
        self.0.slice_set(&src.0, dim, start)
    }

    /// Slice-assign: returns a new tensor with `src` placed at `ranges`.
    pub fn slice_assign<D: std::ops::RangeBounds<usize>>(&self, ranges: &[D], src: &Self) -> Result<Self> {
        self.0.slice_assign(ranges, &src.0).map(Self)
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
        self.0.storage_and_layout()
    }

    /// Mutable storage access (unsafe — caller must ensure exclusive access).
    pub unsafe fn storage_mut_and_layout(&self) -> (std::sync::RwLockWriteGuard<'_, candle_core::Storage>, &candle_core::Layout) {
        self.0.storage_mut_and_layout()
    }

    /// Construct from raw candle storage (used by CUDA kernels).
    pub fn from_storage(
        storage: candle_core::Storage,
        shape: impl Into<Shape>,
        op: candle_core::op::BackpropOp,
        is_variable: bool,
    ) -> Self {
        Self(candle_core::Tensor::from_storage(storage, shape, op, is_variable))
    }

    // ── candle interop (temporary) ──────────────────────────────────

    /// Access the inner candle_core::Tensor.
    pub fn inner(&self) -> &candle_core::Tensor { &self.0 }

    /// Wrap a candle_core::Tensor.
    pub fn from_candle(t: candle_core::Tensor) -> Self { Self(t) }

    /// Unwrap to candle_core::Tensor.
    pub fn into_candle(self) -> candle_core::Tensor { self.0 }
}

// ── From/Into candle_core::Tensor ────────────────────────────────────

impl From<candle_core::Tensor> for Tensor {
    fn from(t: candle_core::Tensor) -> Self { Self(t) }
}

impl From<Tensor> for candle_core::Tensor {
    fn from(t: Tensor) -> Self { t.0 }
}

impl AsRef<candle_core::Tensor> for Tensor {
    fn as_ref(&self) -> &candle_core::Tensor { &self.0 }
}

impl std::borrow::Borrow<candle_core::Tensor> for Tensor {
    fn borrow(&self) -> &candle_core::Tensor { &self.0 }
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
        (&self.0 + &rhs.0).map(Tensor)
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
    fn add(self, rhs: f64) -> Result<Tensor> { (self.0 + rhs).map(Tensor) }
}
impl std::ops::Add<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f64) -> Result<Tensor> { (&self.0 + rhs).map(Tensor) }
}
// f64 + Tensor
impl std::ops::Add<Tensor> for f64 {
    type Output = Result<Tensor>;
    fn add(self, rhs: Tensor) -> Result<Tensor> { (self + rhs.0).map(Tensor) }
}

// Tensor - Tensor
impl std::ops::Sub for Tensor {
    type Output = Result<Self>;
    fn sub(self, rhs: Self) -> Result<Self> { (self.0 - rhs.0).map(Self) }
}
impl std::ops::Sub for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { (&self.0 - &rhs.0).map(Tensor) }
}
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> { (self.0 - &rhs.0).map(Tensor) }
}
impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: Tensor) -> Result<Tensor> { (&self.0 - rhs.0).map(Tensor) }
}
// Tensor - f64
impl std::ops::Sub<f64> for Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { (self.0 - rhs).map(Tensor) }
}
impl std::ops::Sub<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f64) -> Result<Tensor> { (&self.0 - rhs).map(Tensor) }
}

// Tensor * Tensor
impl std::ops::Mul for Tensor {
    type Output = Result<Self>;
    fn mul(self, rhs: Self) -> Result<Self> { (self.0 * rhs.0).map(Self) }
}
impl std::ops::Mul for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { (&self.0 * &rhs.0).map(Tensor) }
}
impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> { (self.0 * &rhs.0).map(Tensor) }
}
impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: Tensor) -> Result<Tensor> { (&self.0 * rhs.0).map(Tensor) }
}
// Tensor * f64
impl std::ops::Mul<f64> for Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { (self.0 * rhs).map(Tensor) }
}
impl std::ops::Mul<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f64) -> Result<Tensor> { (&self.0 * rhs).map(Tensor) }
}

// Tensor / Tensor
impl std::ops::Div for Tensor {
    type Output = Result<Self>;
    fn div(self, rhs: Self) -> Result<Self> { (self.0 / rhs.0).map(Self) }
}
impl std::ops::Div for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: &Tensor) -> Result<Tensor> { (&self.0 / &rhs.0).map(Tensor) }
}
// Tensor / f64
impl std::ops::Div<f64> for Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { (self.0 / rhs).map(Tensor) }
}
impl std::ops::Div<f64> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: f64) -> Result<Tensor> { (&self.0 / rhs).map(Tensor) }
}

// -Tensor
impl std::ops::Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { self.0.neg().map(Tensor) }
}
impl std::ops::Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> { self.0.neg().map(Tensor) }
}

// ── Display ─────────────────────────────────────────────────────────

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(f)
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

pub mod shape {
    pub use candle_core::shape::Dim;
    pub use candle_core::shape::Dims;
    pub use candle_core::shape::ShapeWithOneHole;
}

pub mod safetensors {
    pub use candle_core::safetensors::*;
}

pub mod quantized {
    pub use candle_core::quantized::*;
}

// Re-export WithDType for use in generic code.
pub use candle_core::WithDType;
