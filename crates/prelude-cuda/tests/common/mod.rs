//! Shared helpers for the prelude-cuda GPU kernel test suites.
//!
//! Cargo compiles `tests/common/mod.rs` into each test binary that declares
//! `mod common;` (it is NOT run as its own test). Because each binary only uses
//! a subset of these helpers, per-binary dead code is expected — hence the
//! crate-level allow.
#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};

/// Standard-normal bf16 tensor with a per-call offset. Generated on the CPU
/// (the candle fork removed CUDA `rand_normal`) and uploaded to `dev`.
pub fn randn_bf16(shape: &[usize], dev: &Device, off: f64) -> Tensor {
    let t = Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap();
    let t = (t + off).unwrap();
    t.to_device(dev).unwrap().to_dtype(DType::BF16).unwrap()
}

/// `(max_abs_diff, cosine_similarity)` between two tensors, flattened to f32.
/// The shared core of the per-suite comparison helpers.
pub fn cosine_stats(a: &Tensor, b: &Tensor) -> (f32, f64) {
    let to_vec = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let x = to_vec(a);
    let y = to_vec(b);
    assert_eq!(x.len(), y.len(), "cosine_stats: length mismatch");
    let mut max_abs = 0f32;
    let (mut dot, mut n1, mut n2) = (0f64, 0f64, 0f64);
    for (p, q) in x.iter().zip(y.iter()) {
        max_abs = max_abs.max((p - q).abs());
        dot += (*p as f64) * (*q as f64);
        n1 += (*p as f64) * (*p as f64);
        n2 += (*q as f64) * (*q as f64);
    }
    (max_abs, dot / (n1.sqrt() * n2.sqrt()).max(1e-30))
}
