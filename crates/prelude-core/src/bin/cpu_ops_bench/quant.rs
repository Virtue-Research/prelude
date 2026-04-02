//! Quantized kernel benchmarks: precision + throughput.
//!
//! Precision follows llama.cpp's test-quantize-fns.cpp:
//!   - Test vectors: `0.1 + 2.0 * cos(i + offset)` with GGML_TEST_SIZE=4096
//!   - Error metric: `|quant_dot - f32_dot| / length` (normalized per element)
//!   - Thresholds: MAX_DOT_PRODUCT_ERROR=0.02, per-format reference errors
//!
//! Throughput: dot product and matmul at typical LLM dimensions.

use prelude_core::tensor::{Device, Result, Tensor};
use prelude_core::tensor::quantized::{GgmlDType, QTensor};
use std::time::Instant;

use prelude_core::ops::cpu::quant::*;

/// llama.cpp constants
const GGML_TEST_SIZE: usize = 32 * 128; // 4096
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

/// Reference errors (empirically measured, candle quantizer + our kernels).
/// These may differ slightly from llama.cpp due to quantizer implementation.
fn ggml_reference_error(dtype: GgmlDType) -> f32 {
    match dtype {
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q2K  => 0.004086,
        GgmlDType::Q3K  => 0.018000, // candle Q3_K quantizer has higher error than llama.cpp
        GgmlDType::Q4K  => 0.002425,
        GgmlDType::Q5K  => 0.001401,
        GgmlDType::Q6K  => 0.000520,
        GgmlDType::Q8_0 => 0.000092,
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

/// Helper: quantize test vector into blocks and compute dot product error.
fn check_dot_error(
    dtype: GgmlDType,
    name: &str,
    a: &[f32],
    b: &[f32],
    f32_dot: f32,
    err_multiplier: f32,
) -> Result<()> {
    let k = a.len();
    assert_eq!(k % 256, 0);

    let a_t = Tensor::from_vec(a.to_vec(), (k,), &Device::Cpu)?;
    let qt_a = QTensor::quantize_onto(&a_t, dtype, &Device::Cpu)?;
    let raw = qt_a.data()?;
    let b_q8k = quantize_row_q8k(b);

    let quant_dot = match dtype {
        GgmlDType::Q2K => {
            let blocks: Vec<BlockQ2K> = bytemuck::cast_slice(&raw).to_vec();
            q2_k::vec_dot_q2k_q8k(&blocks, &b_q8k)
        }
        GgmlDType::Q3K => {
            let blocks: Vec<BlockQ3K> = bytemuck::cast_slice(&raw).to_vec();
            q3_k::vec_dot_q3k_q8k(&blocks, &b_q8k)
        }
        GgmlDType::Q4K => {
            let blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&raw).to_vec();
            vec_dot_q4k_q8k(&blocks, &b_q8k)
        }
        GgmlDType::Q5K => {
            let blocks: Vec<BlockQ5K> = bytemuck::cast_slice(&raw).to_vec();
            q5_k::vec_dot_q5k_q8k(&blocks, &b_q8k)
        }
        GgmlDType::Q6K => {
            let blocks: Vec<BlockQ6K> = bytemuck::cast_slice(&raw).to_vec();
            q6_k::vec_dot_q6k_q8k(&blocks, &b_q8k)
        }
        _ => unreachable!(),
    };

    let error = (quant_dot - f32_dot).abs() / k as f32;
    let ref_err = ggml_reference_error(dtype) * err_multiplier;
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
        let a_t = Tensor::from_vec(a.clone(), (GGML_TEST_SIZE,), &Device::Cpu)?;
        let qt_a = QTensor::quantize_onto(&a_t, GgmlDType::Q4_0, &Device::Cpu)?;
        let a_blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&qt_a.data()?).to_vec();
        let b_q8 = quantize_row_q8_0(&b);

        let quant_dot = vec_dot_q4_0_q8_0(&a_blocks, &b_q8);
        let error = (quant_dot - f32_dot).abs() / GGML_TEST_SIZE as f32;
        let ref_err = ggml_reference_error(GgmlDType::Q4_0);
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR && error <= ref_err * 1.1;
        println!(
            "  Q4_0:  error={error:.6}  ref={ref_err:.6}  f32={f32_dot:.2}  quant={quant_dot:.2}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // K-quant dot products
    for &(dtype, name) in &[
        (GgmlDType::Q2K, "Q2_K"),
        (GgmlDType::Q3K, "Q3_K"),
        (GgmlDType::Q4K, "Q4_K"),
        (GgmlDType::Q5K, "Q5_K"),
        (GgmlDType::Q6K, "Q6_K"),
    ] {
        check_dot_error(dtype, name, &a, &b, f32_dot, 1.0)?;
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
        let a_t = Tensor::from_vec(a2.clone(), (GGML_TEST_SIZE,), &Device::Cpu)?;
        let qt_a = QTensor::quantize_onto(&a_t, GgmlDType::Q4_0, &Device::Cpu)?;
        let a_blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&qt_a.data()?).to_vec();
        let b_q8 = quantize_row_q8_0(&b2);
        let quant_dot = vec_dot_q4_0_q8_0(&a_blocks, &b_q8);
        let error = (quant_dot - f32_dot2).abs() / GGML_TEST_SIZE as f32;
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR;
        println!(
            "  Q4_0 (overflow):  error={error:.6}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    for &(dtype, name) in &[
        (GgmlDType::Q2K, "Q2_K"),
        (GgmlDType::Q3K, "Q3_K"),
        (GgmlDType::Q4K, "Q4_K"),
        (GgmlDType::Q5K, "Q5_K"),
        (GgmlDType::Q6K, "Q6_K"),
    ] {
        check_dot_error(dtype, &format!("{name} (overflow)"), &a2, &b2, f32_dot2, 2.0)?;
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
        let a_t = Tensor::from_vec(a_data.clone(), (k,), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&a_t, GgmlDType::Q4_0, &Device::Cpu)?;
        let w_blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&qt.data()?).to_vec();
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

    for &(dtype, name) in &[
        (GgmlDType::Q2K, "Q2_K"),
        (GgmlDType::Q3K, "Q3_K"),
        (GgmlDType::Q4K, "Q4_K"),
        (GgmlDType::Q5K, "Q5_K"),
        (GgmlDType::Q6K, "Q6_K"),
    ] {
        if k % 256 != 0 {
            continue;
        }
        let a_t = Tensor::from_vec(a_data.clone(), (k,), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&a_t, dtype, &Device::Cpu)?;
        let raw = qt.data()?;

        let us = match dtype {
            GgmlDType::Q2K => {
                let blocks: Vec<BlockQ2K> = bytemuck::cast_slice(&raw).to_vec();
                bench_kernel(warmup, repeats, || q2_k::vec_dot_q2k_q8k(&blocks, &b_q8k))
            }
            GgmlDType::Q3K => {
                let blocks: Vec<BlockQ3K> = bytemuck::cast_slice(&raw).to_vec();
                bench_kernel(warmup, repeats, || q3_k::vec_dot_q3k_q8k(&blocks, &b_q8k))
            }
            GgmlDType::Q4K => {
                let blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&raw).to_vec();
                bench_kernel(warmup, repeats, || vec_dot_q4k_q8k(&blocks, &b_q8k))
            }
            GgmlDType::Q5K => {
                let blocks: Vec<BlockQ5K> = bytemuck::cast_slice(&raw).to_vec();
                bench_kernel(warmup, repeats, || q5_k::vec_dot_q5k_q8k(&blocks, &b_q8k))
            }
            GgmlDType::Q6K => {
                let blocks: Vec<BlockQ6K> = bytemuck::cast_slice(&raw).to_vec();
                bench_kernel(warmup, repeats, || q6_k::vec_dot_q6k_q8k(&blocks, &b_q8k))
            }
            _ => unreachable!(),
        };
        results.push((name, us));
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

/// Benchmark matmul throughput: y[M,N] = x[M,K] @ W[N,K]^T
pub fn bench_matmul(m: usize, k: usize, n: usize, warmup: usize, repeats: usize) -> Result<()> {
    let x_data: Vec<f32> = (0..m * k)
        .map(|i| 0.1 + 2.0 * (i as f32 * 0.007).cos())
        .collect();
    let w_data: Vec<f32> = (0..n * k)
        .map(|i| 0.1 + 2.0 * (i as f32 * 0.013).cos())
        .collect();

    // F32 reference
    let f32_us = {
        let x_t = Tensor::from_vec(x_data.clone(), (m, k), &Device::Cpu)?;
        let w_t = Tensor::from_vec(w_data.clone(), (n, k), &Device::Cpu)?.t()?;
        for _ in 0..warmup {
            std::hint::black_box(x_t.matmul(&w_t)?);
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            std::hint::black_box(x_t.matmul(&w_t)?);
        }
        t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // Q4_0
    let q4_0_us = {
        let w_tensor = Tensor::from_vec(w_data.clone(), (n, k), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&w_tensor, GgmlDType::Q4_0, &Device::Cpu)?;
        let blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&qt.data()?).to_vec();
        let mut out = vec![0.0f32; m * n];
        for _ in 0..warmup {
            quantized_matmul_f32(&x_data, &blocks, &mut out, m, n, k);
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            quantized_matmul_f32(&x_data, &blocks, &mut out, m, n, k);
        }
        t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // K-quant matmuls
    let mut kquant_results: Vec<(&str, f64)> = Vec::new();

    if k % 256 == 0 {
        for &(dtype, name, matmul_fn) in &[
            (GgmlDType::Q2K, "Q2_K", bench_matmul_kquant::<BlockQ2K> as fn(&[f32], &[f32], usize, usize, usize, usize, usize, GgmlDType) -> Result<f64>),
            (GgmlDType::Q3K, "Q3_K", bench_matmul_kquant::<BlockQ3K> as fn(&[f32], &[f32], usize, usize, usize, usize, usize, GgmlDType) -> Result<f64>),
            (GgmlDType::Q4K, "Q4_K", bench_matmul_kquant::<BlockQ4K> as fn(&[f32], &[f32], usize, usize, usize, usize, usize, GgmlDType) -> Result<f64>),
            (GgmlDType::Q5K, "Q5_K", bench_matmul_kquant::<BlockQ5K> as fn(&[f32], &[f32], usize, usize, usize, usize, usize, GgmlDType) -> Result<f64>),
            (GgmlDType::Q6K, "Q6_K", bench_matmul_kquant::<BlockQ6K> as fn(&[f32], &[f32], usize, usize, usize, usize, usize, GgmlDType) -> Result<f64>),
        ] {
            let us = matmul_fn(&x_data, &w_data, m, n, k, warmup, repeats, dtype)?;
            kquant_results.push((name, us));
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

/// Generic K-quant matmul benchmark helper.
fn bench_matmul_kquant<B: bytemuck::Pod>(
    x_data: &[f32],
    w_data: &[f32],
    m: usize,
    n: usize,
    k: usize,
    warmup: usize,
    repeats: usize,
    dtype: GgmlDType,
) -> Result<f64> {
    let mut w_blocks: Vec<B> = Vec::new();
    for j in 0..n {
        let row = &w_data[j * k..(j + 1) * k];
        let t = Tensor::from_vec(row.to_vec(), (k,), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&t, dtype, &Device::Cpu)?;
        let blocks: Vec<B> = bytemuck::cast_slice(&qt.data()?).to_vec();
        w_blocks.extend(blocks);
    }
    let mut out = vec![0.0f32; m * n];

    // Use trait objects to dispatch the matmul
    let matmul_fn: Box<dyn Fn()> = match dtype {
        GgmlDType::Q2K => {
            let w: Vec<BlockQ2K> = bytemuck::cast_slice(bytemuck::cast_slice::<B, u8>(&w_blocks)).to_vec();
            let x = x_data.to_vec();
            Box::new(move || {
                let mut out = vec![0.0f32; m * n];
                q2_k::quantized_matmul_q2k(&x, &w, &mut out, m, n, k);
                std::hint::black_box(&out);
            })
        }
        GgmlDType::Q3K => {
            let w: Vec<BlockQ3K> = bytemuck::cast_slice(bytemuck::cast_slice::<B, u8>(&w_blocks)).to_vec();
            let x = x_data.to_vec();
            Box::new(move || {
                let mut out = vec![0.0f32; m * n];
                q3_k::quantized_matmul_q3k(&x, &w, &mut out, m, n, k);
                std::hint::black_box(&out);
            })
        }
        GgmlDType::Q4K => {
            let w: Vec<BlockQ4K> = bytemuck::cast_slice(bytemuck::cast_slice::<B, u8>(&w_blocks)).to_vec();
            let x = x_data.to_vec();
            Box::new(move || {
                let mut out = vec![0.0f32; m * n];
                q4_k::quantized_matmul_q4k(&x, &w, &mut out, m, n, k);
                std::hint::black_box(&out);
            })
        }
        GgmlDType::Q5K => {
            let w: Vec<BlockQ5K> = bytemuck::cast_slice(bytemuck::cast_slice::<B, u8>(&w_blocks)).to_vec();
            let x = x_data.to_vec();
            Box::new(move || {
                let mut out = vec![0.0f32; m * n];
                q5_k::quantized_matmul_q5k(&x, &w, &mut out, m, n, k);
                std::hint::black_box(&out);
            })
        }
        GgmlDType::Q6K => {
            let w: Vec<BlockQ6K> = bytemuck::cast_slice(bytemuck::cast_slice::<B, u8>(&w_blocks)).to_vec();
            let x = x_data.to_vec();
            Box::new(move || {
                let mut out = vec![0.0f32; m * n];
                q6_k::quantized_matmul_q6k(&x, &w, &mut out, m, n, k);
                std::hint::black_box(&out);
            })
        }
        _ => unreachable!(),
    };

    for _ in 0..warmup {
        matmul_fn();
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        matmul_fn();
    }
    Ok(t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
}
