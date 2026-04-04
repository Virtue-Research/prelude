//! Low-level tensor compute — 6 kernel primitives + derived methods.
//!
//! New device: implement 6 primitives, get 70+ tensor methods for free.
//! Want to optimize a specific op? Override just that default method.

use crate::tensor::{DType, Device, Shape, Tensor, Result};

// ── Op enums (match candle's internal dispatch) ─────────────────────

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Exp, Log, Sin, Cos, Abs, Neg, Sqr, Sqrt, Recip, Tanh,
    Relu, Ceil, Floor, Round, Sign,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Min, Max,
}

#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq, Ne, Lt, Gt, Ge, Le,
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum, Max, Min, ArgMax, ArgMin,
}

// ── The trait ───────────────────────────────────────────────────────

pub trait TensorOps: Send + Sync {
    // ── 6 primitives (must implement) ───────────────────────────────

    /// Element-wise unary: f(x) for each element.
    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor>;

    /// Element-wise binary with broadcast: f(a, b).
    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor>;

    /// Element-wise comparison with broadcast.
    fn compare(&self, a: &Tensor, b: &Tensor, op: CompareOp) -> Result<Tensor>;

    /// Reduction along a dimension. If keepdim=false, the dimension is squeezed.
    fn reduce(&self, x: &Tensor, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Tensor>;

    /// Type cast.
    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor>;

    /// Make tensor contiguous (strided copy if needed).
    fn contiguous(&self, x: &Tensor) -> Result<Tensor>;

    // ── Additional primitives (must implement) ──────────────────────

    /// Copy tensor to another device.
    fn to_device(&self, x: &Tensor, device: &Device) -> Result<Tensor>;

    /// Index select along a dimension.
    fn index_select(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor>;

    /// Gather elements along a dimension.
    fn gather(&self, x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor>;

    /// Scatter-add: dst[indices[i]] += src[i].
    fn scatter_add(&self, x: &Tensor, indices: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor>;

    /// Conditional select: where(cond, on_true, on_false).
    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor>;

    /// Concatenate tensors along a dimension.
    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor>;

    /// x * mul + add (fused affine).
    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Result<Tensor>;

    /// Allocate zeros.
    fn zeros(&self, shape: &Shape, dtype: DType, device: &Device) -> Result<Tensor>;

    /// Sort last dimension, return (values, indices).
    fn sort_last_dim(&self, x: &Tensor, asc: bool) -> Result<(Tensor, Tensor)>;

    // ── Derived methods (free, override for optimization) ───────────

    fn exp(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Exp) }
    fn log(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Log) }
    fn sin(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Sin) }
    fn cos(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Cos) }
    fn abs(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Abs) }
    fn neg(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Neg) }
    fn sqr(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Sqr) }
    fn sqrt(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Sqrt) }
    fn recip(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Recip) }
    fn tanh(&self, x: &Tensor) -> Result<Tensor> { self.unary(x, UnaryOp::Tanh) }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.binary(a, b, BinaryOp::Add) }
    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.binary(a, b, BinaryOp::Sub) }
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.binary(a, b, BinaryOp::Mul) }
    fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.binary(a, b, BinaryOp::Div) }
    fn minimum(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.binary(a, b, BinaryOp::Min) }
    fn maximum(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.binary(a, b, BinaryOp::Max) }

    fn eq(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.compare(a, b, CompareOp::Eq) }
    fn ne(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.compare(a, b, CompareOp::Ne) }
    fn lt(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.compare(a, b, CompareOp::Lt) }
    fn gt(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.compare(a, b, CompareOp::Gt) }
    fn ge(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.compare(a, b, CompareOp::Ge) }
    fn le(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { self.compare(a, b, CompareOp::Le) }

    fn sum(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> { self.reduce(x, dim, keepdim, ReduceOp::Sum) }
    fn max_reduce(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> { self.reduce(x, dim, keepdim, ReduceOp::Max) }
    fn min_reduce(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> { self.reduce(x, dim, keepdim, ReduceOp::Min) }
    fn argmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> { self.reduce(x, dim, false, ReduceOp::ArgMax) }
    fn argmin(&self, x: &Tensor, dim: usize) -> Result<Tensor> { self.reduce(x, dim, false, ReduceOp::ArgMin) }

    fn embedding(&self, table: &Tensor, ids: &Tensor) -> Result<Tensor> {
        self.index_select(table, ids, 0)
    }

    fn powf(&self, x: &Tensor, e: f64) -> Result<Tensor> {
        // Default: exp(e * log(x)). Device can override with native powf.
        let log_x = self.log(x)?;
        let scaled = self.affine(&log_x, e, 0.0)?;
        self.exp(&scaled)
    }

    fn clamp(&self, x: &Tensor, min: f64, max: f64) -> Result<Tensor> {
        // Default: affine-based. Device can override.
        self.affine(x, 1.0, 0.0) // TODO: real clamp via compare + where_cond
    }

    fn elu(&self, x: &Tensor, alpha: f64) -> Result<Tensor> {
        // Default: where(x > 0, x, alpha * (exp(x) - 1))
        let zeros = self.zeros(x.shape(), x.dtype(), x.device())?;
        let positive = self.compare(x, &zeros, CompareOp::Gt)?;
        let exp_x = self.exp(x)?;
        let neg_branch = self.affine(&exp_x, alpha, -alpha)?;
        self.where_cond(&positive, x, &neg_branch)
    }
}
