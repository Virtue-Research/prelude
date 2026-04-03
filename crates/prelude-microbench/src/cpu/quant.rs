//! Quantized kernel benchmarks: precision + throughput.
//!
//! Precision follows llama.cpp's test-quantize-fns.cpp:
//!   - Test vectors: `0.1 + 2.0 * cos(i + offset)` with GGML_TEST_SIZE=4096
//!   - Error metric: `|quant_dot - f32_dot| / length` (normalized per element)
//!   - Thresholds: MAX_DOT_PRODUCT_ERROR=0.02, per-format reference errors
//!
//! Throughput: dot product and matmul at typical LLM dimensions.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};
use std::time::Instant;

use prelude_core::ops::cpu::quant::*;

use crate::report::{self, BenchEntry, BenchReport};

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

fn verify_single(
    dtype: GgmlDType,
    label: &str,
    a_data: &[f32],
    b_data: &[f32],
    f32_dot: f32,
    err_multiplier: f32,
) -> Result<()> {
    let k = a_data.len();
    let ref_err = ggml_reference_error(dtype) * err_multiplier;

    match dtype {
        GgmlDType::Q4_0 => {
            let a_t = Tensor::from_vec(a_data.to_vec(), (k,), &Device::Cpu)?;
            let qt_a = QTensor::quantize_onto(&a_t, GgmlDType::Q4_0, &Device::Cpu)?;
            let a_blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&qt_a.data()?).to_vec();
            let b_q8 = quantize_row_q8_0(b_data);

            let quant_dot = vec_dot_q4_0_q8_0(&a_blocks, &b_q8);
            let error = (quant_dot - f32_dot).abs() / k as f32;
            let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR && error <= ref_err * 1.1;
            println!(
                "  {label}:  error={error:.6}  ref={ref_err:.6}  [{}]",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        GgmlDType::Q4K => {
            if k % 256 != 0 {
                return Ok(());
            }
            let a_t = Tensor::from_vec(a_data.to_vec(), (k,), &Device::Cpu)?;
            let qt_a = QTensor::quantize_onto(&a_t, GgmlDType::Q4K, &Device::Cpu)?;
            let a_blocks: Vec<BlockQ4K> = bytemuck::cast_slice(&qt_a.data()?).to_vec();
            let b_q8k = quantize_row_q8k(b_data);

            let quant_dot = vec_dot_q4k_q8k(&a_blocks, &b_q8k);
            let error = (quant_dot - f32_dot).abs() / k as f32;
            let pass = error <= GGML_MAX_DOT_PRODUCT_ERROR && error <= ref_err * 1.1;
            println!(
                "  {label}:  error={error:.6}  ref={ref_err:.6}  [{}]",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        _ => {}
    }
    Ok(())
}

pub fn verify_dot_precision() -> Result<()> {
    println!("\n=== Quantized dot product precision (llama.cpp style) ===");
    println!("  Test size: {GGML_TEST_SIZE}, threshold: {GGML_MAX_DOT_PRODUCT_ERROR}");
    println!("  Error = |quant_dot - f32_dot| / length\n");

    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    let f32_dot = vec_dot_reference(&a, &b);

    verify_single(GgmlDType::Q4_0, "Q4_0", &a, &b, f32_dot, 1.0)?;
    verify_single(GgmlDType::Q4K, "Q4_K", &a, &b, f32_dot, 1.0)?;

    // Second test vector (overflow-prone)
    let a2: Vec<f32> = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect();
    let b2: Vec<f32> = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect();
    let f32_dot2 = vec_dot_reference(&a2, &b2);

    verify_single(GgmlDType::Q4_0, "Q4_0 (overflow)", &a2, &b2, f32_dot2, 2.0)?;
    verify_single(GgmlDType::Q4K, "Q4_K (overflow)", &a2, &b2, f32_dot2, 2.0)?;

    Ok(())
}

// ── Performance benchmarks ───────────────────────────────────────────────

pub fn bench_dot(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    println!("\n=== Quantized dot product throughput ===");
    #[cfg(ggml_baseline)]
    println!("  (vs ggml baseline)");

    for &k in &[256, 512, 1024, 2048, 4096] {
        let a_data: Vec<f32> = (0..k).map(|i| 0.1 + 2.0 * (i as f32).cos()).collect();
        let b_data: Vec<f32> = (0..k).map(|i| 0.1 + 2.0 * (i as f32 + 1.0).cos()).collect();

        // Q4_0: quantize weights and activations
        let a_t = Tensor::from_vec(a_data.clone(), (k,), &Device::Cpu)?;
        let qt_q4_0 = QTensor::quantize_onto(&a_t, GgmlDType::Q4_0, &Device::Cpu)?;
        let q4_0_raw = qt_q4_0.data()?;
        let w_blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&q4_0_raw).to_vec();
        let x_q8 = quantize_row_q8_0(&b_data);

        // Our Q4_0 dot
        let q4_0_us = bench_fn(warmup, repeats, || {
            std::hint::black_box(vec_dot_q4_0_q8_0(&w_blocks, &x_q8));
        });

        // ggml Q4_0 dot (same quantized data, ggml's own Q8_0 quantization)
        #[cfg(ggml_baseline)]
        let ggml_q4_0_us = {
            let ggml_q8 = crate::baselines::ggml::quantize_q8_0(&b_data);
            bench_fn(warmup, repeats, || {
                std::hint::black_box(crate::baselines::ggml::dot_q4_0_q8_0(&q4_0_raw, &ggml_q8, k));
            })
        };

        report.add(BenchEntry {
            category: "cpu/quant/dot".into(),
            name: format!("Q4_0 K={k}"),
            ours_us: q4_0_us,
            #[cfg(ggml_baseline)]
            baseline_name: Some("ggml".into()),
            #[cfg(ggml_baseline)]
            baseline_us: Some(ggml_q4_0_us),
            #[cfg(not(ggml_baseline))]
            baseline_name: None,
            #[cfg(not(ggml_baseline))]
            baseline_us: None,
            note: None,
        });

        // Q4_K
        if k % 256 == 0 {
            let qt_q4k = QTensor::quantize_onto(&a_t, GgmlDType::Q4K, &Device::Cpu)?;
            let q4k_raw = qt_q4k.data()?;
            let w_blocks_k: Vec<BlockQ4K> = bytemuck::cast_slice(&q4k_raw).to_vec();
            let x_q8k = quantize_row_q8k(&b_data);

            // Our Q4_K dot
            let q4k_us = bench_fn(warmup, repeats, || {
                std::hint::black_box(vec_dot_q4k_q8k(&w_blocks_k, &x_q8k));
            });

            // ggml Q4_K dot
            #[cfg(ggml_baseline)]
            let ggml_q4k_us = {
                let ggml_q8k = crate::baselines::ggml::quantize_q8_k(&b_data);
                bench_fn(warmup, repeats, || {
                    std::hint::black_box(crate::baselines::ggml::dot_q4_k_q8_k(&q4k_raw, &ggml_q8k, k));
                })
            };

            report.add(BenchEntry {
                category: "cpu/quant/dot".into(),
                name: format!("Q4_K K={k}"),
                ours_us: q4k_us,
                #[cfg(ggml_baseline)]
                baseline_name: Some("ggml".into()),
                #[cfg(ggml_baseline)]
                baseline_us: Some(ggml_q4k_us),
                #[cfg(not(ggml_baseline))]
                baseline_name: None,
                #[cfg(not(ggml_baseline))]
                baseline_us: None,
                note: None,
            });

            // Print comparison
            #[cfg(ggml_baseline)]
            {
                let ratio_0 = ggml_q4_0_us / q4_0_us;
                let ratio_k = ggml_q4k_us / q4k_us;
                println!(
                    "  K={k:>5}  Q4_0: ours={q4_0_us:.2}us ggml={ggml_q4_0_us:.2}us ({ratio_0:.2}x)  \
                     Q4_K: ours={q4k_us:.2}us ggml={ggml_q4k_us:.2}us ({ratio_k:.2}x)"
                );
            }
            #[cfg(not(ggml_baseline))]
            report::print_result("cpu/quant/dot", &format!("K={k}"), q4k_us, Some(("Q4_0", q4_0_us)));
        } else {
            println!("  K={k:>5}  Q4_0={q4_0_us:.2}us");
        }
    }
    Ok(())
}

fn bench_fn(warmup: usize, repeats: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..repeats {
        f();
    }
    t0.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
}

pub fn bench_matmul(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    println!("\n=== Quantized matmul ===");
    #[cfg(ggml_baseline)]
    println!("  (vs ggml — same rayon scheduling, different dot kernel)");

    let configs: &[(usize, usize, usize)] = &[
        (1, 1024, 1024),
        (1, 2048, 2048),
        (1, 4096, 4096),
        (4, 2048, 2048),
        (4, 4096, 4096),
        (16, 4096, 4096),
        (1, 1024, 4096),
        (1, 4096, 1024),
    ];

    for &(m, k, n) in configs {
        let x_data: Vec<f32> = (0..m * k)
            .map(|i| 0.1 + 2.0 * (i as f32 * 0.007).cos())
            .collect();
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| 0.1 + 2.0 * (i as f32 * 0.013).cos())
            .collect();

        let label = format!("M={m} K={k} N={n}");

        // Q4_0
        {
            let w_tensor = Tensor::from_vec(w_data.clone(), (n, k), &Device::Cpu)?;
            let qt = QTensor::quantize_onto(&w_tensor, GgmlDType::Q4_0, &Device::Cpu)?;
            let raw = qt.data()?;
            let blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(&raw).to_vec();

            let ours = {
                let mut out = vec![0.0f32; m * n];
                bench_fn(warmup, repeats, || {
                    quantized_matmul_f32(&x_data, &blocks, &mut out, m, n, k);
                })
            };

            #[cfg(ggml_baseline)]
            let ggml = {
                let mut out = vec![0.0f32; m * n];
                bench_fn(warmup, repeats, || {
                    crate::baselines::ggml::matmul_q4_0(&x_data, &raw, &mut out, m, n, k);
                })
            };

            #[cfg(ggml_baseline)]
            {
                let ratio = ggml / ours;
                println!("  Q4_0 {label}  ours={ours:.0}us  ggml={ggml:.0}us  ({ratio:.2}x)");
                report.add(BenchEntry {
                    category: "cpu/quant/matmul".into(),
                    name: format!("Q4_0 {label}"),
                    ours_us: ours,
                    baseline_name: Some("ggml".into()),
                    baseline_us: Some(ggml),
                    note: None,
                });
            }
            #[cfg(not(ggml_baseline))]
            {
                println!("  Q4_0 {label}  {ours:.0}us");
                report.add(BenchEntry {
                    category: "cpu/quant/matmul".into(),
                    name: format!("Q4_0 {label}"),
                    ours_us: ours,
                    baseline_name: None, baseline_us: None, note: None,
                });
            }
        }

        // Q4_K
        if k % 256 == 0 {
            let mut w_blocks = Vec::new();
            let mut q4k_raw = Vec::new();
            for j in 0..n {
                let row = &w_data[j * k..(j + 1) * k];
                let t = Tensor::from_vec(row.to_vec(), (k,), &Device::Cpu)?;
                let qt = QTensor::quantize_onto(&t, GgmlDType::Q4K, &Device::Cpu)?;
                let data = qt.data()?;
                let b: Vec<BlockQ4K> = bytemuck::cast_slice(&data).to_vec();
                w_blocks.extend(b);
                q4k_raw.extend_from_slice(&data);
            }

            let ours = {
                let mut out = vec![0.0f32; m * n];
                bench_fn(warmup, repeats, || {
                    prelude_core::ops::cpu::quant::q4_k::quantized_matmul_q4k(
                        &x_data, &w_blocks, &mut out, m, n, k,
                    );
                })
            };

            #[cfg(ggml_baseline)]
            let ggml = {
                let mut out = vec![0.0f32; m * n];
                bench_fn(warmup, repeats, || {
                    crate::baselines::ggml::matmul_q4_k(&x_data, &q4k_raw, &mut out, m, n, k);
                })
            };

            #[cfg(ggml_baseline)]
            {
                let ratio = ggml / ours;
                println!("  Q4_K {label}  ours={ours:.0}us  ggml={ggml:.0}us  ({ratio:.2}x)");
                report.add(BenchEntry {
                    category: "cpu/quant/matmul".into(),
                    name: format!("Q4_K {label}"),
                    ours_us: ours,
                    baseline_name: Some("ggml".into()),
                    baseline_us: Some(ggml),
                    note: None,
                });
            }
            #[cfg(not(ggml_baseline))]
            {
                println!("  Q4_K {label}  {ours:.0}us");
                report.add(BenchEntry {
                    category: "cpu/quant/matmul".into(),
                    name: format!("Q4_K {label}"),
                    ours_us: ours,
                    baseline_name: None, baseline_us: None, note: None,
                });
            }
        }
    }
    Ok(())
}
