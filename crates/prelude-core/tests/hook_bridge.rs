//! Tests for the TensorHook → Ops bridge mechanism.
//!
//! Verifies that:
//! 1. Registering a hook intercepts candle's native tensor ops
//! 2. Hook is cleared during dispatch (recursion guard)
//! 3. Returning None falls through to candle's default
//! 4. Multiple ops can be overridden independently

mod common;

use prelude_core::tensor::{DType, Device, Result, Tensor};
use prelude_core::ops::{self, traits::{Ops, UnaryOp, BinaryOp}};
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Test Ops that counts hook invocations ────────────────────────

static UNARY_COUNT: AtomicUsize = AtomicUsize::new(0);
static BINARY_COUNT: AtomicUsize = AtomicUsize::new(0);
static MATMUL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Tests share `UNARY_COUNT` etc. as process-wide atomics. cargo test
/// runs them in parallel by default — without serialization, one test's
/// `reset_counts()` clobbers another's mid-flight `assert_eq!(.., 1)`.
/// Hold this mutex for the full duration of every hook-test body.
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());


struct CountingOps;

impl Ops for CountingOps {
    fn default_impl(&self) -> &dyn Ops { self }

    fn unary(&self, x: &Tensor, op: UnaryOp) -> Option<Result<Tensor>> {
        UNARY_COUNT.fetch_add(1, Ordering::Relaxed);
        match op {
            // Override exp: return 42.0 filled tensor (proves hook ran)
            UnaryOp::Exp => {
                Some(Tensor::ones(x.shape(), x.dtype(), x.device()).and_then(|t| t.affine(42.0, 0.0)))
            }
            // Override sqrt: use candle's own sqrt (tests recursion guard)
            UnaryOp::Sqrt => {
                // Inside hook, the hook is cleared, so x.sqrt() goes through candle default
                Some(x.sqrt())
            }
            // Everything else: fall through to candle
            _ => None,
        }
    }

    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Option<Result<Tensor>> {
        BINARY_COUNT.fetch_add(1, Ordering::Relaxed);
        match op {
            // Override add: return a * 2 + b * 3 (proves hook ran)
            BinaryOp::Add => {
                let result = (|| -> Result<Tensor> {
                    let aa = a.affine(2.0, 0.0)?;
                    let bb = b.affine(3.0, 0.0)?;
                    Ok((&aa + &bb)?)
                })();
                Some(result)
            }
            _ => None,
        }
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Option<Result<Tensor>> {
        MATMUL_COUNT.fetch_add(1, Ordering::Relaxed);
        None // fall through to candle's matmul
    }
}

static COUNTING_OPS: CountingOps = CountingOps;

fn reset_counts() {
    UNARY_COUNT.store(0, Ordering::Relaxed);
    BINARY_COUNT.store(0, Ordering::Relaxed);
    MATMUL_COUNT.store(0, Ordering::Relaxed);
}

/// Set up hook bridge and ops for a test closure.
fn with_hook<F: FnOnce() -> Result<()>>(f: F) -> Result<()> {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    reset_counts();
    candle_core::hook::set(Some(ops::hook_bridge_ref()));
    let result = ops::with_ops(&COUNTING_OPS, f);
    candle_core::hook::set(None);
    result
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn hook_intercepts_exp() -> Result<()> {
    with_hook(|| {
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let out = t.exp()?;
        let vals: Vec<f32> = out.to_vec1()?;
        // Should be 42.0, not e^1, e^2, e^3
        assert!(vals.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "expected 42.0 from hooked exp, got {:?}", vals);
        assert_eq!(UNARY_COUNT.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn hook_falls_through_for_unoverridden_ops() -> Result<()> {
    with_hook(|| {
        let t = Tensor::new(&[1.0f32, 0.0, -1.0], &Device::Cpu)?;
        let out = t.sin()?;
        let vals: Vec<f32> = out.to_vec1()?;
        // Should be real sin values (hook returned None for Sin)
        assert!((vals[0] - 0.8415).abs() < 1e-3, "sin(1.0)={}", vals[0]);
        assert!((vals[1]).abs() < 1e-5, "sin(0.0)={}", vals[1]);
        // Hook was called (count incremented) but returned None
        assert_eq!(UNARY_COUNT.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn hook_recursion_guard() -> Result<()> {
    // When hook_unary(Sqrt) calls x.sqrt() internally,
    // the hook is cleared so it doesn't recurse infinitely.
    with_hook(|| {
        let t = Tensor::new(&[4.0f32, 9.0, 16.0], &Device::Cpu)?;
        let out = t.sqrt()?;
        let vals: Vec<f32> = out.to_vec1()?;
        // Should be [2, 3, 4] — the hook calls candle's sqrt internally
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
        assert!((vals[2] - 4.0).abs() < 1e-5);
        // Hook called once (not infinite recursion)
        assert_eq!(UNARY_COUNT.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn hook_intercepts_binary_add() -> Result<()> {
    with_hook(|| {
        let a = Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?;
        let b = Tensor::new(&[10.0f32, 20.0], &Device::Cpu)?;
        let out = (&a + &b)?;
        let vals: Vec<f32> = out.to_vec1()?;
        // Hooked add: a*2 + b*3 = [2+30, 4+60] = [32, 64]
        assert!((vals[0] - 32.0).abs() < 1e-5, "got {}", vals[0]);
        assert!((vals[1] - 64.0).abs() < 1e-5, "got {}", vals[1]);
        assert_eq!(BINARY_COUNT.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn hook_matmul_passthrough() -> Result<()> {
    with_hook(|| {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
        let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &Device::Cpu)?;
        let out = a.matmul(&b)?;
        // Hook returns None → candle's matmul runs. Count should be 1.
        let vals: Vec<Vec<f32>> = out.to_vec2()?;
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert!((vals[0][0] - 19.0).abs() < 1e-3);
        assert!((vals[1][1] - 50.0).abs() < 1e-3);
        assert_eq!(MATMUL_COUNT.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn no_hook_uses_candle_default() -> Result<()> {
    // Without any hook set, tensor ops should work normally
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    candle_core::hook::set(None);
    let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    let out = t.exp()?;
    let vals: Vec<f32> = out.to_vec1()?;
    assert!((vals[0] - 1.0f32.exp()).abs() < 1e-5);
    assert!((vals[1] - 2.0f32.exp()).abs() < 1e-5);
    Ok(())
}

#[test]
fn broadcast_ops_go_through_hook() -> Result<()> {
    // broadcast_add calls add internally, so it should trigger the hook
    with_hook(|| {
        let a = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?; // [1, 2]
        let b = Tensor::new(&[10.0f32, 20.0], &Device::Cpu)?; // [2]
        let out = a.broadcast_add(&b)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        // broadcast_add calls add internally → hook intercepts
        // Hooked add: a*2 + b*3 = [2+30, 4+60] = [32, 64]
        assert!((vals[0] - 32.0).abs() < 1e-5, "got {}", vals[0]);
        assert!((vals[1] - 64.0).abs() < 1e-5, "got {}", vals[1]);
        Ok(())
    })
}
