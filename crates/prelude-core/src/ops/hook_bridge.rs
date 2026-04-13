//! Bridge between candle's TensorHook and prelude's Ops trait.
//!
//! `OpsHookBridge` implements `candle_core::TensorHook` and forwards
//! each call to the current thread-local Ops. Device crates decide
//! which hooks to handle:
//!
//! - CUDA: all hook methods return None → candle's own kernels run
//! - AMD/Custom: hook methods dispatch to device-specific kernels

use candle_core::hook::TensorHook;
use crate::tensor::{DType, Tensor, Result};
use crate::ops::traits::{UnaryOp, BinaryOp};

/// Singleton bridge — registered as candle's TensorHook.
pub(crate) static BRIDGE: OpsHookBridge = OpsHookBridge;

pub(crate) struct OpsHookBridge;

/// Get current ops from thread-local, or fall back to device-based selection.
fn current_ops(device: &candle_core::Device) -> &'static dyn crate::ops::Ops {
    crate::ops::ops_for(device)
}

impl TensorHook for OpsHookBridge {
    // ── Unary ops ────────────────────────────────────────────────
    fn exp(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Exp) }
    fn log(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Log) }
    fn sin(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Sin) }
    fn cos(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Cos) }
    fn tanh(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Tanh) }
    fn abs(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Abs) }
    fn neg(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Neg) }
    fn recip(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Recip) }
    fn sqr(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Sqr) }
    fn sqrt(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Sqrt) }
    fn gelu(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Gelu) }
    fn gelu_erf(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::GeluErf) }
    fn erf(&self, x: &Tensor) -> Option<Result<Tensor>> { None } // no UnaryOp::Erf
    fn relu(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Relu) }
    fn silu(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Silu) }
    fn ceil(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Ceil) }
    fn floor(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Floor) }
    fn round(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Round) }
    fn sign(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_unary(x, UnaryOp::Sign) }

    // ── Binary ops ───────────────────────────────────────────────
    fn add(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_binary(a, b, BinaryOp::Add) }
    fn mul(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_binary(a, b, BinaryOp::Mul) }
    fn sub(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_binary(a, b, BinaryOp::Sub) }
    fn div(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_binary(a, b, BinaryOp::Div) }
    fn maximum(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_binary(a, b, BinaryOp::Max) }
    fn minimum(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_binary(a, b, BinaryOp::Min) }

    // ── Core compute ops ─────────────────────────────────────────
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> { current_ops(a.device()).hook_matmul(a, b) }
    fn contiguous(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_contiguous(x) }
    fn to_dtype(&self, x: &Tensor, dtype: DType) -> Option<Result<Tensor>> { current_ops(x.device()).hook_to_dtype(x, dtype) }
    fn copy_strided(&self, x: &Tensor) -> Option<Result<Tensor>> { current_ops(x.device()).hook_contiguous(x) }

    // ── Indexing ─────────────────────────────────────────────────
    fn index_select(&self, x: &Tensor, ids: &Tensor, dim: usize) -> Option<Result<Tensor>> { current_ops(x.device()).hook_index_select(x, ids, dim) }
    fn gather(&self, x: &Tensor, ids: &Tensor, dim: usize) -> Option<Result<Tensor>> { current_ops(x.device()).hook_gather(x, ids, dim) }

    // ── Reduction ────────────────────────────────────────────────
    fn sum(&self, x: &Tensor, dims: &[usize]) -> Option<Result<Tensor>> { current_ops(x.device()).hook_reduce_sum(x, dims) }
    fn max(&self, x: &Tensor, dims: &[usize]) -> Option<Result<Tensor>> { current_ops(x.device()).hook_reduce_max(x, dims) }
    fn min(&self, x: &Tensor, dims: &[usize]) -> Option<Result<Tensor>> { current_ops(x.device()).hook_reduce_min(x, dims) }
    fn argmax(&self, _x: &Tensor, _dim: usize) -> Option<Result<Tensor>> { None }
    fn argmin(&self, _x: &Tensor, _dim: usize) -> Option<Result<Tensor>> { None }

    // ── Compose ──────────────────────────────────────────────────
    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Option<Result<Tensor>> {
        if tensors.is_empty() { return None; }
        current_ops(tensors[0].device()).hook_cat(tensors, dim)
    }
    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Option<Result<Tensor>> {
        current_ops(cond.device()).hook_where_cond(cond, on_true, on_false)
    }

    // ── Misc ─────────────────────────────────────────────────────
    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Option<Result<Tensor>> { current_ops(x.device()).hook_affine(x, mul, add) }
    fn powf(&self, _x: &Tensor, _e: f64) -> Option<Result<Tensor>> { None }
    fn elu(&self, _x: &Tensor, _alpha: f64) -> Option<Result<Tensor>> { None }
    fn to_device(&self, x: &Tensor, device: &candle_core::Device) -> Option<Result<Tensor>> { current_ops(x.device()).hook_to_device(x, device) }
}
