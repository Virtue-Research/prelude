// Benchmark fused vs separate gate_up MLP path.
//
// Investigates whether `fused gate_up GEMM + silu_and_mul` (fused path)
// is actually faster than `2× GEMM + silu*mul` (separate path).
//
// Run: cargo run --bin bench_kernel -p prelude-cuda --release
//
// Tests Qwen3-0.6B (H=1024, I=3072) and Qwen3-8B (H=4096, I=12288)
// across multiple token counts to find any crossover.

#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

use prelude_core::ops::traits::Ops;
use prelude_core::tensor::{DType, Device, Module, Result, Tensor};

/// One model's MLP shape.
struct ModelShape {
    name: &'static str,
    hidden: usize,
    intermediate: usize,
}

const MODELS: &[ModelShape] = &[
    ModelShape {
        name: "Qwen3-0.6B",
        hidden: 1024,
        intermediate: 3072,
    },
    ModelShape {
        name: "Qwen3-8B",
        hidden: 4096,
        intermediate: 12288,
    },
    ModelShape {
        name: "Qwen3-32B",
        hidden: 5120,
        intermediate: 25600,
    },
];

const T_VALUES: &[usize] = &[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

fn rand_bf16(shape: impl Into<prelude_core::tensor::Shape>, dev: &Device) -> Result<Tensor> {
    Tensor::randn(0f64, 1.0, shape, &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(dev)
}

/// Time a closure: warmup + N iterations + sync. Returns avg microseconds.
fn time_us<F: FnMut() -> Result<()>>(dev: &Device, mut f: F, iters: usize) -> Result<f64> {
    // Warmup
    for _ in 0..3 {
        f()?;
    }
    prelude_cuda::device::synchronize(dev)?;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    prelude_cuda::device::synchronize(dev)?;
    Ok(t0.elapsed().as_micros() as f64 / iters as f64)
}

/// Fused path: 1× GEMM [T, H] × [H, 2I] + silu_and_mul → [T, I]
/// `gate_up_w` is `[2I, H]` (row-major weight); we use `.t()` lazy view for TN GEMM.
fn run_fused(x: &Tensor, gate_up_w: &Tensor, ops: &dyn Ops) -> Result<Tensor> {
    let gate_up = x.matmul(&gate_up_w.t()?)?; // [T, 2I]
    match ops.silu_mul_concat(&gate_up) {
        Some(r) => r,
        None => {
            let dim = gate_up.dim(1)? / 2;
            let g = gate_up.narrow(1, 0, dim)?;
            let u = gate_up.narrow(1, dim, dim)?;
            let s = g.apply(&prelude_core::models::commons::activation::Activation::Silu)?;
            &s * &u
        }
    }
}

/// Separate path: 2× GEMM + silu * mul → [T, I]
fn run_separate(
    x: &Tensor,
    gate_w: &Tensor, // [I, H]
    up_w: &Tensor,   // [I, H]
) -> Result<Tensor> {
    let gate = x.matmul(&gate_w.t()?)?; // [T, I]
    let up = x.matmul(&up_w.t()?)?; // [T, I]
    let s = gate.apply(&prelude_core::models::commons::activation::Activation::Silu)?;
    &s * &up
}

fn bench_model(dev: &Device, ops: &dyn Ops, m: &ModelShape) -> Result<()> {
    println!(
        "\n═══ {} (H={}, I={}) ═══",
        m.name, m.hidden, m.intermediate
    );
    println!(
        "{:>6} {:>14} {:>14} {:>10} {:>14} {:>14}",
        "T", "fused (us)", "separate (us)", "speedup", "fused GEMM", "fused silu"
    );

    // Build weights once.
    let gate_w = rand_bf16((m.intermediate, m.hidden), dev)?;
    let up_w = rand_bf16((m.intermediate, m.hidden), dev)?;
    let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?; // [2I, H]

    for &t in T_VALUES {
        // Input: [T, H]
        let x = rand_bf16((t, m.hidden), dev)?;

        // Iterations: more for small T to reduce noise.
        let iters = if t <= 32 {
            200
        } else if t <= 256 {
            100
        } else if t <= 1024 {
            50
        } else {
            20
        };

        let fused_us = time_us(
            dev,
            || {
                let _ = run_fused(&x, &gate_up_w, ops)?;
                Ok(())
            },
            iters,
        )?;

        let sep_us = time_us(
            dev,
            || {
                let _ = run_separate(&x, &gate_w, &up_w)?;
                Ok(())
            },
            iters,
        )?;

        // Component breakdown for fused path
        let fused_gemm_us = time_us(
            dev,
            || {
                let _ = x.matmul(&gate_up_w.t()?)?;
                Ok(())
            },
            iters,
        )?;

        let gate_up = x.matmul(&gate_up_w.t()?)?;
        let fused_silu_us = time_us(
            dev,
            || {
                let _ = ops.silu_mul_concat(&gate_up).ok_or_else(|| {
                    candle_core::Error::Msg("silu_mul_concat not supported".into())
                })??;
                Ok(())
            },
            iters,
        )?;

        let speedup = sep_us / fused_us;
        println!(
            "{:>6} {:>14.2} {:>14.2} {:>9.2}x {:>14.2} {:>14.2}",
            t, fused_us, sep_us, speedup, fused_gemm_us, fused_silu_us
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    prelude_cuda::register();
    let ops = prelude_cuda::cuda_ops();

    // Probe DeepGEMM directly across multiple M values for the problematic shape.
    println!("\n═══ DeepGEMM N=24576 K=4096 support across M ═══");
    for m in [1i32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let n = 24576i32;
        let k = 4096i32;
        let cfg = deepgemm::query_config(m, n, k);
        // Try a real launch via Tensor::matmul + measure
        let x = rand_bf16((m as usize, k as usize), &dev)?;
        let w = rand_bf16((n as usize, k as usize), &dev)?;
        let us = time_us(
            &dev,
            || {
                let _ = x.matmul(&w.t()?)?;
                Ok(())
            },
            50,
        )?;
        // Bandwidth lower bound
        let bw_us = (n * k * 2) as f64 / (4.8 * 1e6);
        let eff = bw_us / us * 100.0;
        let mark = if eff < 50.0 {
            "← SLOW (likely CUTLASS)"
        } else {
            ""
        };
        println!(
            "  M={:>4}  cfg={:?}  matmul={:>7.2}us  ({:>5.1}% bw)  {}",
            m, cfg, us, eff, mark
        );
    }

    println!("\n═══ Same probe for N=12288 (single gate/up) ═══");
    for m in [1i32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let n = 12288i32;
        let k = 4096i32;
        let cfg = deepgemm::query_config(m, n, k);
        let x = rand_bf16((m as usize, k as usize), &dev)?;
        let w = rand_bf16((n as usize, k as usize), &dev)?;
        let us = time_us(
            &dev,
            || {
                let _ = x.matmul(&w.t()?)?;
                Ok(())
            },
            50,
        )?;
        let bw_us = (n * k * 2) as f64 / (4.8 * 1e6);
        let eff = bw_us / us * 100.0;
        println!(
            "  M={:>4}  cfg={:?}  matmul={:>7.2}us  ({:>5.1}% bw)",
            m, cfg, us, eff
        );
    }

    // Time all GEMMs in Qwen3 models via Tensor::matmul (which exercises dispatch).
    // Slow outliers indicate CUTLASS fallback (DeepGEMM couldn't handle the shape).
    println!("\n═══ Per-layer GEMM timing via Tensor::matmul (M=4, BF16) ═══");
    println!("Slow rows = DeepGEMM doesn't have kernel, falls back to CUTLASS\n");

    for mdl in MODELS {
        let h = mdl.hidden;
        let i = mdl.intermediate;
        let kv = h / 4; // Qwen3 4× GQA
        let vocab = 151936;
        let layers: Vec<(&str, usize, usize)> = vec![
            ("q_proj", h, h),
            ("k_proj", kv, h),
            ("v_proj", kv, h),
            ("o_proj", h, h),
            ("qkv_fused", h + 2 * kv, h),
            ("gate_proj", i, h),
            ("up_proj", i, h),
            ("gate_up_fused", 2 * i, h),
            ("down_proj", h, i),
            ("lm_head", vocab, h),
        ];
        println!("─── {} ───", mdl.name);
        let m = 4usize;
        for (name, n, k) in &layers {
            let x = rand_bf16((m, *k), &dev)?;
            let w = rand_bf16((*n, *k), &dev)?;
            let us = time_us(
                &dev,
                || {
                    let _ = x.matmul(&w.t()?)?;
                    Ok(())
                },
                100,
            )?;
            // Compute pure-bandwidth lower bound (BF16, 2 bytes/elem; weight dominates)
            let bw_us = (n * k * 2) as f64 / (4.8 * 1e6); // 4.8 TB/s H200
            let efficiency = bw_us / us * 100.0;
            println!(
                "  {:<14} N={:>6} K={:>5}  {:>7.2}us  ({:>5.1}% of bandwidth bound {:>5.1}us)",
                name, n, k, us, efficiency, bw_us
            );
        }
        println!();
    }

    println!("\nBench: gate_up MLP path (BF16, CUDA)");
    println!("Comparing fused (1× GEMM + silu_and_mul) vs separate (2× GEMM + silu*mul)");

    for m in MODELS {
        bench_model(&dev, ops, m)?;
    }

    println!("\nNote: speedup > 1 means fused is faster, < 1 means separate is faster.");
    Ok(())
}
