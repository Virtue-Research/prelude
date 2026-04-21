//! Quantized kernel benchmarks: precision + throughput.
//!
//! Precision follows llama.cpp's test-quantize-fns.cpp:
//!   - Test vectors: `0.1 + 2.0 * cos(i + offset)` with GGML_TEST_SIZE=4096
//!   - Error metric: `|quant_dot - f32_dot| / length` (normalized per element)
//!   - Thresholds: MAX_DOT_PRODUCT_ERROR=0.02, per-format reference errors
//!
//! Throughput: dot product and matmul at typical LLM dimensions.

use candle_core::{Device, Result, Tensor};
use candle_core::quantized::{GgmlDType, QTensor};
use std::time::Instant;

use prelude_core::ops::cpu::quant::*;

/// llama.cpp constants
const GGML_TEST_SIZE: usize = 32 * 128; // 4096
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

/// Reference errors from llama.cpp (per-format expected normalized error).
fn ggml_reference_error(dtype: GgmlDType) -> f32 {
    match dtype {
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4K => 0.002425,
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

    // Q4_K dot product
    if GGML_TEST_SIZE % 256 == 0 {
        let a_t = Tensor::from_vec(a.clone(), (GGML_TEST_SIZE,), &Device::Cpu)?;
        let qt_a = QTensor::quantize_onto(&a_t, GgmlDType::Q4K, &Device::Cpu)?;
        let a_blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&qt_a.data()?).to_vec();
        let b_q8k = quantize_row_q8k(&b);

        let quant_dot = vec_dot_q4k_q8k(&a_blocks, &b_q8k);
        let error = (quant_dot - f32_dot).abs() / GGML_TEST_SIZE as f32;
        let ref_err = ggml_reference_error(GgmlDType::Q4K);
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR && error <= ref_err * 1.1;
        println!(
            "  Q4_K:  error={error:.6}  ref={ref_err:.6}  f32={f32_dot:.2}  quant={quant_dot:.2}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // Second test vector (from candle: more likely to trigger overflow)
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
        let ref_err = ggml_reference_error(GgmlDType::Q4_0) * 2.0; // err_m=2.0 for this test
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR;
        println!(
            "  Q4_0 (overflow test): error={error:.6}  ref={ref_err:.6}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    if GGML_TEST_SIZE % 256 == 0 {
        let a_t = Tensor::from_vec(a2.clone(), (GGML_TEST_SIZE,), &Device::Cpu)?;
        let qt_a = QTensor::quantize_onto(&a_t, GgmlDType::Q4K, &Device::Cpu)?;
        let a_blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&qt_a.data()?).to_vec();
        let b_q8k = quantize_row_q8k(&b2);

        let quant_dot = vec_dot_q4k_q8k(&a_blocks, &b_q8k);
        let error = (quant_dot - f32_dot2).abs() / GGML_TEST_SIZE as f32;
        let ref_err = ggml_reference_error(GgmlDType::Q4K) * 2.0;
        let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR;
        println!(
            "  Q4_K (overflow test): error={error:.6}  ref={ref_err:.6}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    Ok(())
}

// ── Performance benchmarks ───────────────────────────────────────────────

/// Benchmark dot product throughput.
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

    // Q4_K
    let q4k_us = if k % 256 == 0 {
        let a_t = Tensor::from_vec(a_data.clone(), (k,), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&a_t, GgmlDType::Q4K, &Device::Cpu)?;
        let w_blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&qt.data()?).to_vec();
        let x_q8k = quantize_row_q8k(&b_data);

        for _ in 0..warmup {
            std::hint::black_box(vec_dot_q4k_q8k(&w_blocks, &x_q8k));
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            std::hint::black_box(vec_dot_q4k_q8k(&w_blocks, &x_q8k));
        }
        Some(t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
    } else {
        None
    };

    print!("  K={k:>5}  Q4_0={q4_0_us:>8.2}us");
    if let Some(us) = q4k_us {
        print!("  Q4_K={us:>8.2}us");
    }
    println!();
    Ok(())
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

    // Q4_K
    let q4k_us = if k % 256 == 0 {
        let mut w_blocks = Vec::new();
        for j in 0..n {
            let row = &w_data[j * k..(j + 1) * k];
            let t = Tensor::from_vec(row.to_vec(), (k,), &Device::Cpu)?;
            let qt = QTensor::quantize_onto(&t, GgmlDType::Q4K, &Device::Cpu)?;
            let blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&qt.data()?).to_vec();
            w_blocks.extend(blocks);
        }
        let mut out = vec![0.0f32; m * n];
        for _ in 0..warmup {
            prelude_core::ops::cpu::quant::q4_k::quantized_matmul_q4k(
                &x_data, &w_blocks, &mut out, m, n, k,
            );
        }
        let t0 = Instant::now();
        for _ in 0..repeats {
            prelude_core::ops::cpu::quant::q4_k::quantized_matmul_q4k(
                &x_data, &w_blocks, &mut out, m, n, k,
            );
        }
        Some(t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
    } else {
        None
    };

    let w_mb = (n * k * 4) as f64 / 1e6;
    print!(
        "  M={m:>4} K={k:>5} N={n:>5}  F32={f32_us:>8.0}us({w_mb:.1}MB)  Q4_0={q4_0_us:>8.0}us"
    );
    if let Some(us) = q4k_us {
        print!("  Q4_K={us:>8.0}us");
    }
    let sp0 = f32_us / q4_0_us;
    print!("  Q4_0/F32={sp0:.2}x");
    if let Some(us) = q4k_us {
        print!("  Q4_K/F32={:.2}x", f32_us / us);
    }
    println!();
    Ok(())
}
