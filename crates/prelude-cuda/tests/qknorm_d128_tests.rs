//! Bit-exactness + perf check for the D=128 specialized fused qknorm+rope
//! kernel vs the generic kernel. Requires a CUDA GPU.
//!
//! The d128 variant must be BIT-IDENTICAL: same fp32 math in the same order,
//! only the memory access pattern differs (vectorized uint2, full unroll).

use candle_core::{DType, Device, Tensor};
use prelude_cuda::qknorm_rope_qk_for_tests;

mod common;
use common::randn_bf16;

const D: usize = 128;
const HQ: usize = 32;
const HKV: usize = 4;

fn dev() -> Device {
    Device::new_cuda(0).expect("needs CUDA device 0")
}

/// Build the serving-shaped inputs: q/k are non-contiguous views into a fused
/// QKV projection output (token stride = (HQ+2*HKV)*D), plus rope tables and
/// ragged position ids.
fn build_inputs(
    total_tokens: usize,
    dev: &Device,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let fused = randn_bf16(&[total_tokens, HQ + 2 * HKV, D], dev, 0.0);
    let q = fused.narrow(1, 0, HQ).unwrap();
    let k = fused.narrow(1, HQ, HKV).unwrap();
    let qw = randn_bf16(&[D], dev, 1.0);
    let kw = randn_bf16(&[D], dev, 1.0);
    let max_pos = 4096usize;
    // real rope tables: cos/sin of pos/theta^(2i/d)
    let mut cos = vec![0f32; max_pos * D / 2];
    let mut sin = vec![0f32; max_pos * D / 2];
    for p in 0..max_pos {
        for i in 0..D / 2 {
            let inv_freq = 1f64 / 10000f64.powf(2.0 * i as f64 / D as f64);
            let ang = (p as f64) * inv_freq;
            cos[p * D / 2 + i] = ang.cos() as f32;
            sin[p * D / 2 + i] = ang.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos, (max_pos, D / 2), &Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let sin = Tensor::from_vec(sin, (max_pos, D / 2), &Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    // ragged positions (mix of prefill runs and decode singles)
    let pos: Vec<u32> = (0..total_tokens)
        .map(|i| ((i * 37 + 11) % 4000) as u32)
        .collect();
    let pos = Tensor::from_vec(pos, (total_tokens,), dev).unwrap();
    (q, k, qw, kw, cos, sin, pos)
}

fn bits(t: &Tensor) -> Vec<u16> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<half::bf16>()
        .unwrap()
        .into_iter()
        .map(half::bf16::to_bits)
        .collect()
}

#[test]
fn d128_bit_exact_vs_generic() {
    let dev = dev();
    for total_tokens in [3usize, 17, 1037, 8192] {
        let (q, k, qw, kw, cos, sin, pos) = build_inputs(total_tokens, &dev);
        let (qg, kg) =
            qknorm_rope_qk_for_tests(&q, &k, &qw, &kw, &cos, &sin, &pos, 1e-6, Some(false))
                .unwrap();
        let (qd, kd) =
            qknorm_rope_qk_for_tests(&q, &k, &qw, &kw, &cos, &sin, &pos, 1e-6, Some(true)).unwrap();
        assert_eq!(bits(&qg), bits(&qd), "Q mismatch at T={total_tokens}");
        assert_eq!(bits(&kg), bits(&kd), "K mismatch at T={total_tokens}");
        println!(
            "T={total_tokens}: bit-exact OK ({} Q rows)",
            total_tokens * HQ
        );
    }
}

#[test]
fn d128_perf_vs_generic() {
    let dev = dev();
    let total_tokens = 8192usize;
    let (q, k, qw, kw, cos, sin, pos) = build_inputs(total_tokens, &dev);
    let iters = 200;
    for (label, force) in [("generic", Some(false)), ("d128", Some(true))] {
        // warmup
        for _ in 0..20 {
            let _ =
                qknorm_rope_qk_for_tests(&q, &k, &qw, &kw, &cos, &sin, &pos, 1e-6, force).unwrap();
        }
        dev.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ =
                qknorm_rope_qk_for_tests(&q, &k, &qw, &kw, &cos, &sin, &pos, 1e-6, force).unwrap();
        }
        dev.synchronize().unwrap();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        println!("{label}: {us:.1} us/call (T={total_tokens}, {HQ}+{HKV} heads, d={D})");
    }
}
