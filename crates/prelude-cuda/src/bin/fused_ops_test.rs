#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

// Quick correctness and performance test for fused CUDA kernels.
// Usage: cargo run --bin fused_ops_test -p prelude-cuda --release

use prelude_core::ops::traits::Ops;
use prelude_core::tensor::{DType, Device, Module, Result, Tensor, bail};

/// Helper: call a fused op that returns Option<Result<T>>, bail if None.
fn must_fuse<T>(opt: Option<Result<T>>, name: &str) -> Result<T> {
    match opt {
        Some(r) => r,
        None => bail!("{name}: fused kernel not available"),
    }
}

/// Generate random BF16 tensor on GPU (via CPU, since curand is removed).
fn rand_gpu(shape: impl Into<prelude_core::tensor::Shape>, dev: &Device) -> Result<Tensor> {
    Tensor::randn(0f64, 1.0, shape, &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(dev)
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
    println!(
        "{:<12} {:<8} {:>12} {:>12} {:>12} {:>8}",
        "model", "tokens", "concat(us)", "split(us)", "split+cp(us)", "concat/split"
    );

    // (label, tokens, intermediate_size)
    let shapes = [
        ("Qwen3-0.6B", 1, 3072),
        ("Qwen3-0.6B", 4, 3072),
        ("Qwen3-8B", 1, 12288),
        ("Qwen3-8B", 4, 12288),
        ("Qwen3-8B", 128, 12288),
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
            let g = gate_up
                .narrow(1, 0, intermediate)?
                .reshape((tokens, intermediate))?;
            let u = gate_up
                .narrow(1, intermediate, intermediate)?
                .reshape((tokens, intermediate))?;
            let _ = must_fuse(ops.fused_silu_mul(&g, &u), "fused_silu_mul")?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let split_copy_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        let ratio = concat_us / split_us;
        println!(
            "{:<12} {:<8} {:>11.1} {:>11.1} {:>11.1} {:>7.2}x",
            label, tokens, concat_us, split_us, split_copy_us, ratio
        );
    }

    // ── fused_add_rmsnorm (FlashInfer in-place) ────────────────

    println!("\n= fused_add_rmsnorm via FlashInfer (decode shapes) =");
    println!("{:<12} {:<8} {:>12}", "model", "tokens", "time(us)");

    let norm_shapes = [
        ("Qwen3-0.6B", 1, 1536),
        ("Qwen3-0.6B", 4, 1536),
        ("Qwen3-8B", 1, 4096),
        ("Qwen3-8B", 4, 4096),
        ("Qwen3-8B", 128, 4096),
        ("Qwen3-32B", 1, 5120),
        ("Qwen3-32B", 4, 5120),
    ];

    let warmup = 10;
    let iters = 200;
    let eps = 1e-6f32;

    for (label, tokens, hidden) in norm_shapes {
        let x = rand_gpu((tokens, hidden), &dev)?;
        let residual = rand_gpu((tokens, hidden), &dev)?;
        let weight = rand_gpu(hidden, &dev)?;

        for _ in 0..warmup {
            let _ = ops.fused_add_rmsnorm(&residual, &x, &weight, eps);
        }
        prelude_cuda::device::synchronize(&dev)?;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = must_fuse(
                ops.fused_add_rmsnorm(&residual, &x, &weight, eps),
                "fused_add_rmsnorm",
            )?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        println!("{:<12} {:<8} {:>11.1}", label, tokens, us);
    }

    // ── gdn_post_conv (Qwen3.5 DeltaNet fused post-conv1d prep) ─────
    //
    // Correctness: compute QKV split + L2 norm Q/K + scalar gate
    // (softplus + A_log + dt_bias + exp) + sigmoid beta via plain
    // candle ops, compare against the fused kernel. Shapes here match
    // Qwen3.5-35B-A3B's DeltaNet layer (HK=16, HV=32, D=128).
    println!("\n= gdn_post_conv (Qwen3.5 DeltaNet shape) =");
    {
        let l = 1024usize;
        let hk = 16usize;
        let hv = 32usize;
        let d = 128usize;
        let conv_dim = 2 * hk * d + hv * d;

        let mixed_qkv = rand_gpu((l, conv_dim), &dev)?;
        let a_raw_in = rand_gpu((l, hv), &dev)?;
        let b_raw_in = rand_gpu((l, hv), &dev)?;
        // Match the loader: A_log is F32 native precision; dt_bias we
        // pre-cast to F32 for the kernel contract.
        let a_log = Tensor::randn(0f64, 1.0, hv, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_device(&dev)?;
        let dt_bias = Tensor::randn(0f64, 0.5, hv, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_device(&dev)?;

        // ── Reference via candle ops (mirrors delta_rule_prefill_gdn) ─
        let qkv_ref = mixed_qkv
            .narrow(prelude_core::tensor::D::Minus1, 0, hk * d)?
            .reshape((l, hk, d))?;
        let k_slice_ref = mixed_qkv
            .narrow(prelude_core::tensor::D::Minus1, hk * d, hk * d)?
            .reshape((l, hk, d))?;
        let v_slice_ref = mixed_qkv
            .narrow(prelude_core::tensor::D::Minus1, 2 * hk * d, hv * d)?
            .reshape((l, hv, d))?;

        // L2 norm helper duplicated here rather than importing from
        // prelude-core (keeps this bin self-contained).
        let l2 = |t: &Tensor| -> Result<Tensor> {
            let f32 = t.to_dtype(DType::F32)?;
            let norm = f32
                .sqr()?
                .sum_keepdim(prelude_core::tensor::D::Minus1)?
                .sqrt()?;
            let norm = (norm + 1e-12)?;
            f32.broadcast_div(&norm)?.to_dtype(DType::BF16)
        };
        let q_ref = l2(&qkv_ref)?;
        let k_ref = l2(&k_slice_ref)?;
        let v_ref = v_slice_ref.contiguous()?;

        let neg_exp_a = a_log.exp()?.neg()?;
        let a_plus_dt = a_raw_in.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?;
        // softplus = log(1 + exp(x)) with PyTorch default beta=1.0,
        // threshold=20 (fall back to linear for large x)
        let exp_x = a_plus_dt.exp()?;
        let softplus_unchecked = ((&exp_x + 1f64)?).log()?;
        let mask_large = a_plus_dt.gt(20.0f64)?;
        let softplus_ref = mask_large.where_cond(&a_plus_dt, &softplus_unchecked)?;
        let g_scalar = softplus_ref.broadcast_mul(&neg_exp_a)?;
        let alpha_ref = g_scalar.exp()?.contiguous()?;
        let beta_ref = ops.sigmoid(&b_raw_in.to_dtype(DType::F32)?)?.contiguous()?;

        // ── Fused kernel ───────────────────────────────────────────
        let (q_fused, k_fused, v_fused, alpha_fused, beta_fused) = must_fuse(
            ops.gdn_post_conv(
                &mixed_qkv, &a_raw_in, &b_raw_in, &a_log, &dt_bias, hk, hv, d,
            ),
            "gdn_post_conv",
        )?;

        let max_diff = |lhs: &Tensor, rhs: &Tensor| -> Result<(f32, f32, f32)> {
            let a: Vec<f32> = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let b: Vec<f32> = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let max = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let mean_abs: f32 = a.iter().map(|v| v.abs()).sum::<f32>() / a.len() as f32;
            let rel = max / mean_abs.max(1e-8);
            Ok((max, mean_abs, rel))
        };

        let (dq, mq, rq) = max_diff(&q_ref, &q_fused)?;
        let (dk, _, rk) = max_diff(&k_ref, &k_fused)?;
        let (dv, _, rv) = max_diff(&v_ref, &v_fused)?;
        let (da, ma, ra) = max_diff(&alpha_ref, &alpha_fused)?;
        let (db, _, rb) = max_diff(&beta_ref, &beta_fused)?;
        println!(
            "  q: max_diff={dq:.6e} mean_abs={mq:.4} rel={rq:.6e}\n  \
             k: max_diff={dk:.6e} rel={rk:.6e}\n  \
             v: max_diff={dv:.6e} rel={rv:.6e}  (should be 0 — pure copy)\n  \
             alpha: max_diff={da:.6e} mean_abs={ma:.4} rel={ra:.6e}\n  \
             beta: max_diff={db:.6e} rel={rb:.6e}",
        );
        // BF16 → F32 → BF16 round-trip bounds the absolute error at
        // roughly 1/256 of the magnitude. Be generous for L2-normed Q/K
        // (which have tiny magnitudes after normalisation).
        assert!(dq < 5e-3, "q max_diff={dq}");
        assert!(dk < 5e-3, "k max_diff={dk}");
        assert!(dv < 1e-6, "v max_diff={dv} (should be exact copy)");
        // alpha = exp(...) has higher absolute error because the output
        // is in [0, 1] and each BF16 step is ~1/128; allow 1e-2.
        assert!(da < 1e-2, "alpha max_diff={da}");
        assert!(db < 1e-3, "beta max_diff={db}");

        // ── Perf ───────────────────────────────────────────────────
        for _ in 0..10 {
            let _ = must_fuse(
                ops.gdn_post_conv(
                    &mixed_qkv, &a_raw_in, &b_raw_in, &a_log, &dt_bias, hk, hv, d,
                ),
                "gdn_post_conv",
            )?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let iters_pc = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..iters_pc {
            let _ = must_fuse(
                ops.gdn_post_conv(
                    &mixed_qkv, &a_raw_in, &b_raw_in, &a_log, &dt_bias, hk, hv, d,
                ),
                "gdn_post_conv",
            )?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let fused_us = t0.elapsed().as_nanos() as f64 / iters_pc as f64 / 1000.0;

        // Reference path (same 20-ish candle ops the kernel replaces).
        let bench_ref = || -> Result<()> {
            let qkv_r = mixed_qkv
                .narrow(prelude_core::tensor::D::Minus1, 0, hk * d)?
                .reshape((l, hk, d))?;
            let k_r = mixed_qkv
                .narrow(prelude_core::tensor::D::Minus1, hk * d, hk * d)?
                .reshape((l, hk, d))?;
            let _v_r = mixed_qkv
                .narrow(prelude_core::tensor::D::Minus1, 2 * hk * d, hv * d)?
                .reshape((l, hv, d))?
                .contiguous()?;
            let _q = l2(&qkv_r)?;
            let _k = l2(&k_r)?;
            let neg_exp_a_r = a_log.exp()?.neg()?;
            let plus = a_raw_in.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?;
            let ex = plus.exp()?;
            let sp_u = ((&ex + 1f64)?).log()?;
            let mask = plus.gt(20.0f64)?;
            let sp = mask.where_cond(&plus, &sp_u)?;
            let _alpha = sp.broadcast_mul(&neg_exp_a_r)?.exp()?.contiguous()?;
            let _beta = ops.sigmoid(&b_raw_in.to_dtype(DType::F32)?)?.contiguous()?;
            Ok(())
        };
        for _ in 0..10 {
            bench_ref()?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters_pc {
            bench_ref()?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let ref_us = t0.elapsed().as_nanos() as f64 / iters_pc as f64 / 1000.0;

        println!(
            "  perf L={l}: fused={fused_us:.1} us, ref={ref_us:.1} us, speedup={:.1}x",
            ref_us / fused_us
        );
    }

    // ── gather_log_softmax (vLLM-aligned fused prompt_logprobs) ─────
    //
    // Correctness: compare against candle's `log_softmax` + `gather`
    // reference path. Qwen3.5-35B-A3B shape: [1024 tokens, 151_936
    // vocab]. We test both BF16 and F32 logits.
    println!("\n= gather_log_softmax (Qwen3.5-35B-A3B shape) =");
    {
        use prelude_core::tensor::D;

        let num_tokens = 1024usize;
        let vocab_size = 151_936usize;

        // Random logits. We intentionally keep the dynamic range small
        // so the reference softmax path doesn't NaN — the fused kernel
        // handles larger ranges fine since it does max subtraction
        // correctly, but the reference path will overflow in BF16 with
        // spikier inputs.
        let logits_f32 = Tensor::randn(0f64, 1.0, (num_tokens, vocab_size), &Device::Cpu)?
            .to_device(&dev)?
            .to_dtype(DType::F32)?;
        let logits_bf16 = logits_f32.to_dtype(DType::BF16)?;

        // Random target ids in [0, vocab_size).
        let target_host: Vec<u32> = (0..num_tokens)
            .map(|i| ((i * 7919) % vocab_size) as u32)
            .collect();
        let target = Tensor::from_vec(target_host.clone(), (num_tokens,), &dev)?;

        // Reference: per-row log_softmax via `x - logsumexp(x, keepdim)`,
        // then gather. Inlined because prelude-cuda doesn't depend on
        // candle_nn and we don't want to pull it in just for the test
        // scaffold.
        let ref_log_softmax = |x: &Tensor| -> Result<Tensor> {
            let x_f32 = x.to_dtype(DType::F32)?;
            let max = x_f32.max_keepdim(D::Minus1)?;
            let shifted = x_f32.broadcast_sub(&max)?;
            let exp = shifted.exp()?;
            let sum = exp.sum_keepdim(D::Minus1)?;
            let lse = sum.log()?.broadcast_add(&max)?;
            x_f32.broadcast_sub(&lse)
        };
        let reference_logprobs = |logits: &Tensor| -> Result<Vec<f32>> {
            let log_probs = ref_log_softmax(logits)?;
            let idx = target.reshape((num_tokens, 1))?;
            let gathered = log_probs.gather(&idx, 1)?;
            Ok(gathered.flatten_all()?.to_vec1::<f32>()?)
        };

        for (label, logits, tol) in [
            ("F32", &logits_f32, 1e-4f32),
            ("BF16", &logits_bf16, 5e-2f32), // BF16 input, F32 reduction
        ] {
            let ref_lp = reference_logprobs(logits)?;
            let fused_tensor = must_fuse(
                ops.gather_log_softmax(logits, &target),
                "gather_log_softmax",
            )?;
            let fused_lp: Vec<f32> = fused_tensor.to_vec1()?;

            let max_diff: f32 = ref_lp
                .iter()
                .zip(fused_lp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let mean_abs: f32 = ref_lp.iter().map(|x| x.abs()).sum::<f32>() / ref_lp.len() as f32;
            println!(
                "  {label}: max_diff={max_diff:.6e} mean_abs={mean_abs:.4} relative={:.6e}",
                max_diff / mean_abs.max(1e-8)
            );
            if max_diff > tol {
                bail!(
                    "gather_log_softmax {label} correctness failed: max_diff={max_diff:.6e} > tol={tol:.6e}"
                );
            }
        }

        // ── Perf: fused kernel vs candle reference path ────────────
        let bench_fused = || -> Result<()> {
            let _ = ops.gather_log_softmax(&logits_bf16, &target);
            Ok(())
        };
        let bench_ref = || -> Result<()> {
            let lp = ref_log_softmax(&logits_bf16)?;
            let _ = lp.gather(&target.reshape((num_tokens, 1))?, 1)?;
            Ok(())
        };

        // Warmup
        for _ in 0..10 {
            bench_fused()?;
            bench_ref()?;
        }
        prelude_cuda::device::synchronize(&dev)?;

        let iters = 50;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            bench_fused()?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let fused_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            bench_ref()?;
        }
        prelude_cuda::device::synchronize(&dev)?;
        let ref_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        println!(
            "  perf [T={num_tokens} V={vocab_size}]: fused={fused_us:.1} us, ref={ref_us:.1} us, speedup={:.1}x",
            ref_us / fused_us
        );
    }

    Ok(())
}
