#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

// Quick correctness and performance test for fused CUDA kernels.
// Usage: cargo run --bin fused_ops_test -p prelude-cuda --release

use prelude_core::ops::traits::Ops;
use prelude_core::tensor::{bail, DType, Device, Module, Result, Tensor};

/// Helper: call a fused op that returns Option<Result<T>>, bail if None.
fn must_fuse<T>(opt: Option<Result<T>>, name: &str) -> Result<T> {
    match opt {
        Some(r) => r,
        None => bail!("{name}: fused kernel not available"),
    }
}

/// Generate random BF16 tensor on GPU (via CPU, since curand is removed).
fn rand_gpu(shape: impl Into<prelude_core::tensor::Shape>, dev: &Device) -> Result<Tensor> {
    Tensor::randn(0f64, 1.0, shape, &Device::Cpu)?.to_dtype(DType::BF16)?.to_device(dev)
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let ops = prelude_cuda::cuda_ops();
    let n = 1024 * 3072; // typical intermediate_size * seq

    println!(
        "Testing fused ops on {} elements ({:.1} MB BF16)",
        n,
        n as f64 * 2.0 / 1e6
    );

    // ── Test fused_silu_mul ─────────────────────────────────────
    let gate = rand_gpu(n, &dev)?;
    let up = rand_gpu(n, &dev)?;

    // Reference: silu(gate) * up
    let silu_gate = gate.apply(&prelude_core::models::commons::activation::Activation::Silu)?;
    let ref_result = (&silu_gate * &up)?;

    // Fused
    let fused_result = must_fuse(ops.fused_silu_mul(&gate, &up), "fused_silu_mul")?;

    // Compare
    let ref_f32: Vec<f32> = ref_result.to_dtype(DType::F32)?.to_vec1()?;
    let fused_f32: Vec<f32> = fused_result.to_dtype(DType::F32)?.to_vec1()?;

    let max_diff: f32 = ref_f32
        .iter()
        .zip(fused_f32.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs: f32 = ref_f32.iter().map(|x: &f32| x.abs()).sum::<f32>() / ref_f32.len() as f32;
    println!(
        "fused_silu_mul: max_diff={:.6}, mean_abs={:.4}, relative={:.6}",
        max_diff,
        mean_abs,
        max_diff / mean_abs.max(1e-8)
    );

    // ── Test vectorized_add ─────────────────────────────────────
    let a = rand_gpu(n, &dev)?;
    let b = rand_gpu(n, &dev)?;

    let ref_add = (&a + &b)?;
    let fused_add = must_fuse(ops.fused_add(&a, &b), "fused_add")?;

    let ref_f32: Vec<f32> = ref_add.to_dtype(DType::F32)?.to_vec1()?;
    let fused_f32: Vec<f32> = fused_add.to_dtype(DType::F32)?.to_vec1()?;

    let max_diff: f32 = ref_f32
        .iter()
        .zip(fused_f32.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs: f32 = ref_f32.iter().map(|x: &f32| x.abs()).sum::<f32>() / ref_f32.len() as f32;
    println!(
        "vectorized_add: max_diff={:.6}, mean_abs={:.4}, relative={:.6}",
        max_diff,
        mean_abs,
        max_diff / mean_abs.max(1e-8)
    );

    // ── Performance test ─────────────────────────────────────
    // Warmup
    for _ in 0..5 {
        let _ = ops.fused_silu_mul(&gate, &up);
        let _ = ops.fused_add(&a, &b);
    }
    prelude_cuda::device::synchronize(&dev)?;

    let iters = 100;

    // fused_silu_mul perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = ops.fused_silu_mul(&gate, &up);
    }
    prelude_cuda::device::synchronize(&dev)?;
    let fused_silu_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Reference silu*mul perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let s = gate.apply(&prelude_core::models::commons::activation::Activation::Silu)?;
        let _ = (&s * &up)?;
    }
    prelude_cuda::device::synchronize(&dev)?;
    let ref_silu_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // vectorized_add perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = ops.fused_add(&a, &b);
    }
    prelude_cuda::device::synchronize(&dev)?;
    let fused_add_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Reference add perf
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = (&a + &b)?;
    }
    prelude_cuda::device::synchronize(&dev)?;
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

    // ── silu_mul_concat vs fused_silu_mul (model shapes) ────
    //
    // silu_mul_concat:  takes [tokens, 2*dim], splits internally (FlashInfer kernel)
    // fused_silu_mul:   takes separate gate, up tensors (our custom kernel)
    //
    // In the fused gate_up path, the "separate" approach needs narrow+contiguous
    // copy first, so we measure that full pipeline too.

    println!("\n= silu_mul_concat vs fused_silu_mul (model decode shapes) =");
    println!("{:<12} {:<8} {:>12} {:>12} {:>12} {:>8}",
        "model", "tokens", "concat(us)", "split(us)", "split+cp(us)", "concat/split");

    // (label, tokens, intermediate_size)
    let shapes = [
        ("Qwen3-0.6B", 1,  3072),
        ("Qwen3-0.6B", 4,  3072),
        ("Qwen3-8B",   1,  12288),
        ("Qwen3-8B",   4,  12288),
        ("Qwen3-8B",   128, 12288),
    ];

    let warmup = 10;
    let iters = 200;

    for (label, tokens, intermediate) in shapes {
        let dim2 = intermediate * 2;
        let gate_up = rand_gpu((tokens, dim2), &dev)?;

        // Pre-split (contiguous) gate and up for fused_silu_mul
        let gate_c = rand_gpu((tokens, intermediate), &dev)?;
        let up_c = rand_gpu((tokens, intermediate), &dev)?;

        // Warmup
        for _ in 0..warmup {
            let _ = ops.silu_mul_concat(&gate_up);
            let _ = ops.fused_silu_mul(&gate_c, &up_c);
        }
        prelude_cuda::device::synchronize(&dev)?;

        // Bench silu_mul_concat (FlashInfer)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = must_fuse(ops.silu_mul_concat(&gate_up), "silu_mul_concat")?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let concat_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        // Bench fused_silu_mul on pre-split contiguous tensors (best case, no copy)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = must_fuse(ops.fused_silu_mul(&gate_c, &up_c), "fused_silu_mul")?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let split_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        // Bench narrow+contiguous+fused_silu_mul (actual path without silu_mul_concat)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let g = gate_up.narrow(1, 0, intermediate)?.reshape((tokens, intermediate))?;
            let u = gate_up.narrow(1, intermediate, intermediate)?.reshape((tokens, intermediate))?;
            let _ = must_fuse(ops.fused_silu_mul(&g, &u), "fused_silu_mul")?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let split_copy_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        let ratio = concat_us / split_us;
        println!("{:<12} {:<8} {:>11.1} {:>11.1} {:>11.1} {:>7.2}x",
            label, tokens, concat_us, split_us, split_copy_us, ratio);
    }

    // ── fused_add_rmsnorm: ours vs FlashInfer ─────────────────
    //
    // Our kernel: allocates new output buffers (out_sum, out_norm)
    // FlashInfer: in-place (modifies input + residual)

    println!("\n= fused_add_rmsnorm: ours vs FlashInfer (decode shapes) =");
    println!("{:<12} {:<8} {:>12} {:>12} {:>8}",
        "model", "tokens", "ours(us)", "flashinfer(us)", "ratio");

    // (label, tokens, hidden_size)
    let norm_shapes = [
        ("Qwen3-0.6B", 1,  1536),
        ("Qwen3-0.6B", 4,  1536),
        ("Qwen3-8B",   1,  4096),
        ("Qwen3-8B",   4,  4096),
        ("Qwen3-8B",   128, 4096),
    ];

    let warmup = 10;
    let iters = 200;
    let eps = 1e-6f32;

    for (label, tokens, hidden) in norm_shapes {
        let x = rand_gpu((tokens, hidden), &dev)?;
        let residual = rand_gpu((tokens, hidden), &dev)?;
        let weight = rand_gpu(hidden, &dev)?;

        // Warmup both
        for _ in 0..warmup {
            let _ = ops.fused_add_rmsnorm(&residual, &x, &weight, eps);
            // FlashInfer is in-place, needs fresh tensors each call for warmup
            let x2 = rand_gpu((tokens, hidden), &dev)?;
            let r2 = rand_gpu((tokens, hidden), &dev)?;
            let _ = prelude_cuda::fi_fused_add_rmsnorm(&x2, &r2, &weight, eps as f64);
        }
        prelude_cuda::device::synchronize(&dev)?;

        // Bench ours
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = must_fuse(ops.fused_add_rmsnorm(&residual, &x, &weight, eps), "fused_add_rmsnorm")?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let ours_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        // Bench FlashInfer (in-place, reuse same buffers — result is wrong but timing is valid)
        let fi_x = rand_gpu((tokens, hidden), &dev)?;
        let fi_r = rand_gpu((tokens, hidden), &dev)?;
        for _ in 0..warmup {
            prelude_cuda::fi_fused_add_rmsnorm(&fi_x, &fi_r, &weight, eps as f64)?;
        }
        prelude_cuda::device::synchronize(&dev)?;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            prelude_cuda::fi_fused_add_rmsnorm(&fi_x, &fi_r, &weight, eps as f64)?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let fi_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        let ratio = ours_us / fi_us;
        println!("{:<12} {:<8} {:>11.1} {:>13.1} {:>7.2}x",
            label, tokens, ours_us, fi_us, ratio);
    }

    Ok(())
}
