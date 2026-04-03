#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

// Quick correctness and performance test for fused CUDA kernels.
// Usage: cargo run --bin fused_ops_test -p prelude-cuda --release

use candle_core::{DType, Device, Module, Tensor};

fn main() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let ops = prelude_cuda::create_cuda_ops();
    let n = 1024 * 3072; // typical intermediate_size * seq

    println!(
        "Testing fused ops on {} elements ({:.1} MB BF16)",
        n,
        n as f64 * 2.0 / 1e6
    );

    // ── Test fused_silu_mul ─────────────────────────────────────
    let gate = Tensor::randn(0f32, 1.0, n, &dev)?.to_dtype(DType::BF16)?;
    let up = Tensor::randn(0f32, 1.0, n, &dev)?.to_dtype(DType::BF16)?;

    // Reference: silu(gate) * up
    let silu_gate = prelude_core::nn_ops::Activation::Silu.forward(&gate)?;
    let ref_result = (&silu_gate * &up)?;

    // Fused
    let fused_result = ops.fused.fused_silu_mul(&gate, &up)
        .expect("CudaOps must support fused_silu_mul")?;

    // Compare
    let ref_f32 = ref_result.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let fused_f32 = fused_result.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let max_diff = ref_f32
        .iter()
        .zip(fused_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = ref_f32.iter().map(|x| x.abs()).sum::<f32>() / ref_f32.len() as f32;
    println!(
        "fused_silu_mul: max_diff={:.6}, mean_abs={:.4}, relative={:.6}",
        max_diff,
        mean_abs,
        max_diff / mean_abs.max(1e-8)
    );

    // ── Test vectorized_add ─────────────────────────────────────
    let a = Tensor::randn(0f32, 1.0, n, &dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::randn(0f32, 1.0, n, &dev)?.to_dtype(DType::BF16)?;

    let ref_add = (&a + &b)?;
    let fused_add = ops.fused.fused_add(&a, &b)
        .expect("CudaOps must support fused_add")?;

    let ref_f32 = ref_add.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let fused_f32 = fused_add.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let max_diff = ref_f32
        .iter()
        .zip(fused_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = ref_f32.iter().map(|x| x.abs()).sum::<f32>() / ref_f32.len() as f32;
    println!(
        "vectorized_add: max_diff={:.6}, mean_abs={:.4}, relative={:.6}",
        max_diff,
        mean_abs,
        max_diff / mean_abs.max(1e-8)
    );

    // ── Performance test ─────────────────────────────────────
    // Warmup
    for _ in 0..5 {
        let _ = ops.fused.fused_silu_mul(&gate, &up);
        let _ = ops.fused.fused_add(&a, &b);
    }
    dev.synchronize()?;

    let iters = 100;

    // fused_silu_mul perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = ops.fused.fused_silu_mul(&gate, &up);
    }
    dev.synchronize()?;
    let fused_silu_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Reference silu*mul perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let s = prelude_core::nn_ops::Activation::Silu.forward(&gate)?;
        let _ = (&s * &up)?;
    }
    dev.synchronize()?;
    let ref_silu_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // vectorized_add perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = ops.fused.fused_add(&a, &b);
    }
    dev.synchronize()?;
    let fused_add_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Reference add perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = (&a + &b)?;
    }
    dev.synchronize()?;
    let ref_add_us = t0.elapsed().as_micros() as f64 / iters as f64;

    println!("\nPerformance (n={}):", n);
    println!(
        "  fused_silu_mul: {:.0} us  (ref: {:.0} us, speedup: {:.1}x)",
        fused_silu_us,
        ref_silu_us,
        ref_silu_us / fused_silu_us
    );
    println!(
        "  vectorized_add: {:.0} us  (ref: {:.0} us, speedup: {:.1}x)",
        fused_add_us,
        ref_add_us,
        ref_add_us / fused_add_us
    );

    Ok(())
}
