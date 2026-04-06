//! Quantized kernel benchmarks: precision + throughput.
//!
//! Precision follows llama.cpp's test-quantize-fns.cpp:
//!   - Test vectors: `0.1 + 2.0 * cos(i + offset)` with GGML_TEST_SIZE=4096
//!   - Error metric: `|quant_dot - f32_dot| / length` (normalized per element)
//!   - Thresholds: MAX_DOT_PRODUCT_ERROR=0.02, per-format reference errors
//!
//! Throughput: dot product and matmul at typical LLM dimensions.

use prelude_core::tensor::Result;
use std::time::Instant;

use prelude_cpu::ops::quant::*;

/// llama.cpp constants
const GGML_TEST_SIZE: usize = 32 * 128; // 4096
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

/// Reference errors (empirically measured, our quantizers + our kernels).
/// These may differ slightly from llama.cpp due to quantizer implementation.
fn reference_error(name: &str) -> f32 {
    match name {
        "Q4_0" => 0.001143,
        "Q2_K" => 0.004086,
        "Q3_K" => 0.018000, // our Q3_K quantizer has higher error than llama.cpp
        "Q4_K" => 0.002425,
        "Q5_K" => 0.001401,
        "Q6_K" => 0.000520,
        "Q8_0" => 0.000092,
        _ => 0.01,
    }
}

/// llama.cpp test vector: `0.1 + 2.0 * cos(i + offset)`
fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// F32 reference dot product.
fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Helper: quantize K-quant test vector into blocks and compute dot product error.
fn check_dot_error(
    name: &str,
    a: &[f32],
    b: &[f32],
    f32_dot: f32,
    err_multiplier: f32,
) -> Result<()> {
    let k = a.len();
    assert_eq!(k % 256, 0);

    let b_q8k = quantize_row_q8k(b);

    let quant_dot = match name {
        "Q2_K" => {
            let blocks = quantize_f32_q2k(a);
            q2_k::vec_dot_q2k_q8k(&blocks, &b_q8k)
        }
        "Q3_K" => {
            let blocks = quantize_f32_q3k(a);
            q3_k::vec_dot_q3k_q8k(&blocks, &b_q8k)
        }
        "Q4_K" => {
            let blocks = quantize_f32_q4k(a);
            vec_dot_q4k_q8k(&blocks, &b_q8k)
        }
        "Q5_K" => {
            let blocks = quantize_f32_q5k(a);
            q5_k::vec_dot_q5k_q8k(&blocks, &b_q8k)
        }
        "Q6_K" => {
            let blocks = quantize_f32_q6k(a);
            q6_k::vec_dot_q6k_q8k(&blocks, &b_q8k)
        }
        _ => unreachable!("unsupported K-quant: {name}"),
    };

    let error = (quant_dot - f32_dot).abs() / k as f32;
    let ref_err = reference_error(name) * err_multiplier;
    let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR && error <= ref_err * 1.1;
    println!(
        "  {name:>5}:  error={error:.6}  ref={ref_err:.6}  f32={f32_dot:.2}  quant={quant_dot:.2}  [{}]",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(())
}

// ── Precision verification (llama.cpp style) ─────────────────────────────

/// Verify quantized dot product precision against F32 reference.
/// Follows llama.cpp test-quantize-fns.cpp exactly.
pub fn verify_dot_precision() -> Result<()> {
    println!("\n=== Quantized dot product precision (llama.cpp style) ===");
    println!("  Test size: {GGML_TEST_SIZE}, threshold: {GGML_MAX_DOT_PRODUCT_ERROR}");
    println!("  Error = |quant_dot - f32_dot| / length\n");

    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    let f32_dot = vec_dot_reference(&a, &b);

    // Q4_0 dot product
    {
        let a_blocks = quantize_f32_q4_0(&a);
        let b_q8 = quantize_row_q8_0(&b);

        let quant_dot = vec_dot_q4_0_q8_0(&a_blocks, &b_q8);
        let error = (quant_dot - f32_dot).abs() / GGML_TEST_SIZE as f32;
        let ref_err = reference_error("Q4_0");
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR && error <= ref_err * 1.1;
        println!(
            "  Q4_0:  error={error:.6}  ref={ref_err:.6}  f32={f32_dot:.2}  quant={quant_dot:.2}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // K-quant dot products
    for name in &["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"] {
        check_dot_error(name, &a, &b, f32_dot, 1.0)?;
    }

    // Second test vector (overflow-prone: linear ramp 0→1)
    println!();
    let a2: Vec<f32> = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect();
    let b2: Vec<f32> = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect();
    let f32_dot2 = vec_dot_reference(&a2, &b2);

    {
        let a_blocks = quantize_f32_q4_0(&a2);
        let b_q8 = quantize_row_q8_0(&b2);
        let quant_dot = vec_dot_q4_0_q8_0(&a_blocks, &b_q8);
        let error = (quant_dot - f32_dot2).abs() / GGML_TEST_SIZE as f32;
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR;
        println!(
            "  Q4_0 (overflow):  error={error:.6}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    for name in &["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"] {
        check_dot_error(&format!("{name} (overflow)"), &a2, &b2, f32_dot2, 2.0)?;
    }

    Ok(())
}

// ── Performance benchmarks ───────────────────────────────────────────────

/// Benchmark dot product throughput for all formats.
pub fn bench_dot(k: usize, warmup: usize, repeats: usize) -> Result<()> {
    let a_data: Vec<f32> = (0..k).map(|i| 0.1 + 2.0 * (i as f32).cos()).collect();
    let b_data: Vec<f32> = (0..k).map(|i| 0.1 + 2.0 * (i as f32 + 1.0).cos()).collect();

    // Q4_0
    let q4_0_us = {
        let w_blocks = quantize_f32_q4_0(&a_data);
        let x_q8 = quantize_row_q8_0(&b_data);

        for _ in 0..warmup {
            std::hint::black_box(vec_dot_q4_0_q8_0(&w_blocks, &x_q8));
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            std::hint::black_box(vec_dot_q4_0_q8_0(&w_blocks, &x_q8));
        }
        t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // K-quant benchmarks (all use Q8_K activation)
    let b_q8k = quantize_row_q8k(&b_data);

    let mut results: Vec<(&str, f64)> = vec![("Q4_0", q4_0_us)];

    if k % 256 == 0 {
        // Q2_K
        {
            let blocks = quantize_f32_q2k(&a_data);
            let us = bench_kernel(warmup, repeats, || q2_k::vec_dot_q2k_q8k(&blocks, &b_q8k));
            results.push(("Q2_K", us));
        }
        // Q3_K
        {
            let blocks = quantize_f32_q3k(&a_data);
            let us = bench_kernel(warmup, repeats, || q3_k::vec_dot_q3k_q8k(&blocks, &b_q8k));
            results.push(("Q3_K", us));
        }
        // Q4_K
        {
            let blocks = quantize_f32_q4k(&a_data);
            let us = bench_kernel(warmup, repeats, || vec_dot_q4k_q8k(&blocks, &b_q8k));
            results.push(("Q4_K", us));
        }
        // Q5_K
        {
            let blocks = quantize_f32_q5k(&a_data);
            let us = bench_kernel(warmup, repeats, || q5_k::vec_dot_q5k_q8k(&blocks, &b_q8k));
            results.push(("Q5_K", us));
        }
        // Q6_K
        {
            let blocks = quantize_f32_q6k(&a_data);
            let us = bench_kernel(warmup, repeats, || q6_k::vec_dot_q6k_q8k(&blocks, &b_q8k));
            results.push(("Q6_K", us));
        }
    }

    print!("  K={k:>5}");
    for (name, us) in &results {
        print!("  {name}={us:>8.2}us");
    }
    println!();
    Ok(())
}

fn bench_kernel<F: Fn() -> f32>(warmup: usize, repeats: usize, f: F) -> f64 {
    for _ in 0..warmup {
        std::hint::black_box(f());
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(f());
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}

/// Naive F32 matmul: y[m,n] = x[m,k] @ w[n,k]^T
fn naive_matmul_f32(x: &[f32], w: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += x[i * k + l] * w[j * k + l];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Benchmark matmul throughput: y[M,N] = x[M,K] @ W[N,K]^T
pub fn bench_matmul(m: usize, k: usize, n: usize, warmup: usize, repeats: usize) -> Result<()> {
    let x_data: Vec<f32> = (0..m * k)
        .map(|i| 0.1 + 2.0 * (i as f32 * 0.007).cos())
        .collect();
    let w_data: Vec<f32> = (0..n * k)
        .map(|i| 0.1 + 2.0 * (i as f32 * 0.013).cos())
        .collect();

    // F32 reference (naive)
    let f32_us = {
        let mut out = vec![0.0f32; m * n];
        for _ in 0..warmup {
            naive_matmul_f32(&x_data, &w_data, &mut out, m, n, k);
            std::hint::black_box(&out);
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            naive_matmul_f32(&x_data, &w_data, &mut out, m, n, k);
            std::hint::black_box(&out);
        }
        t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // Q4_0
    let q4_0_us = {
        let mut all_blocks = Vec::new();
        for j in 0..n {
            let row = &w_data[j * k..(j + 1) * k];
            all_blocks.extend(quantize_f32_q4_0(row));
        }
        let mut out = vec![0.0f32; m * n];
        for _ in 0..warmup {
            quantized_matmul_f32(&x_data, &all_blocks, &mut out, m, n, k);
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            quantized_matmul_f32(&x_data, &all_blocks, &mut out, m, n, k);
        }
        t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // K-quant matmuls
    let mut kquant_results: Vec<(&str, f64)> = Vec::new();

    if k % 256 == 0 {
        // Q2_K
        {
            let us = bench_matmul_kquant_q2k(&x_data, &w_data, m, n, k, warmup, repeats);
            kquant_results.push(("Q2_K", us));
        }
        // Q3_K
        {
            let us = bench_matmul_kquant_q3k(&x_data, &w_data, m, n, k, warmup, repeats);
            kquant_results.push(("Q3_K", us));
        }
        // Q4_K
        {
            let us = bench_matmul_kquant_q4k(&x_data, &w_data, m, n, k, warmup, repeats);
            kquant_results.push(("Q4_K", us));
        }
        // Q5_K
        {
            let us = bench_matmul_kquant_q5k(&x_data, &w_data, m, n, k, warmup, repeats);
            kquant_results.push(("Q5_K", us));
        }
        // Q6_K
        {
            let us = bench_matmul_kquant_q6k(&x_data, &w_data, m, n, k, warmup, repeats);
            kquant_results.push(("Q6_K", us));
        }
    }

    let w_mb = (n * k * 4) as f64 / 1e6;
    print!(
        "  M={m:>4} K={k:>5} N={n:>5}  F32={f32_us:>8.0}us({w_mb:.1}MB)  Q4_0={q4_0_us:>8.0}us"
    );
    for (name, us) in &kquant_results {
        print!("  {name}={us:>8.0}us");
    }
    let sp0 = f32_us / q4_0_us;
    print!("  Q4_0/F32={sp0:.2}x");
    if let Some((_, us)) = kquant_results.last() {
        print!("  Q6_K/F32={:.2}x", f32_us / us);
    }
    println!();
    Ok(())
}

/// K-quant matmul benchmark helpers — one per dtype to avoid generics + GgmlDType.

fn bench_matmul_kquant_q2k(
    x_data: &[f32], w_data: &[f32], m: usize, n: usize, k: usize,
    warmup: usize, repeats: usize,
) -> f64 {
    let mut w_blocks = Vec::new();
    for j in 0..n {
        let row = &w_data[j * k..(j + 1) * k];
        w_blocks.extend(quantize_f32_q2k(row));
    }
    let mut out = vec![0.0f32; m * n];
    for _ in 0..warmup {
        q2_k::quantized_matmul_q2k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        q2_k::quantized_matmul_q2k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}

fn bench_matmul_kquant_q3k(
    x_data: &[f32], w_data: &[f32], m: usize, n: usize, k: usize,
    warmup: usize, repeats: usize,
) -> f64 {
    let mut w_blocks = Vec::new();
    for j in 0..n {
        let row = &w_data[j * k..(j + 1) * k];
        w_blocks.extend(quantize_f32_q3k(row));
    }
    let mut out = vec![0.0f32; m * n];
    for _ in 0..warmup {
        q3_k::quantized_matmul_q3k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        q3_k::quantized_matmul_q3k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}

fn bench_matmul_kquant_q4k(
    x_data: &[f32], w_data: &[f32], m: usize, n: usize, k: usize,
    warmup: usize, repeats: usize,
) -> f64 {
    let mut w_blocks = Vec::new();
    for j in 0..n {
        let row = &w_data[j * k..(j + 1) * k];
        w_blocks.extend(quantize_f32_q4k(row));
    }
    let mut out = vec![0.0f32; m * n];
    for _ in 0..warmup {
        q4_k::quantized_matmul_q4k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        q4_k::quantized_matmul_q4k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}

fn bench_matmul_kquant_q5k(
    x_data: &[f32], w_data: &[f32], m: usize, n: usize, k: usize,
    warmup: usize, repeats: usize,
) -> f64 {
    let mut w_blocks = Vec::new();
    for j in 0..n {
        let row = &w_data[j * k..(j + 1) * k];
        w_blocks.extend(quantize_f32_q5k(row));
    }
    let mut out = vec![0.0f32; m * n];
    for _ in 0..warmup {
        q5_k::quantized_matmul_q5k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        q5_k::quantized_matmul_q5k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}

fn bench_matmul_kquant_q6k(
    x_data: &[f32], w_data: &[f32], m: usize, n: usize, k: usize,
    warmup: usize, repeats: usize,
) -> f64 {
    let mut w_blocks = Vec::new();
    for j in 0..n {
        let row = &w_data[j * k..(j + 1) * k];
        w_blocks.extend(quantize_f32_q6k(row));
    }
    let mut out = vec![0.0f32; m * n];
    for _ in 0..warmup {
        q6_k::quantized_matmul_q6k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        q6_k::quantized_matmul_q6k(x_data, &w_blocks, &mut out, m, n, k);
        std::hint::black_box(&out);
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}
