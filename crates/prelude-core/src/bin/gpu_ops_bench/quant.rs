//! GPU dequantization correctness + throughput benchmarks.
//!
//! Correctness: quantize on CPU → upload raw bytes to GPU → dequantize with CUDA kernel
//!              → compare against candle's CPU dequantize (ground truth).
//!
//! Throughput: measure dequantize kernel time at various sizes.

use candle_core::{DType, Device, Result, Tensor};
use candle_core::quantized::{GgmlDType, QTensor};
use std::time::Instant;

use prelude_core::ops::gpu::quant;
use prelude_core::ops::gpu::mmvq;

// ── Correctness verification ─────────────────────────────────────────────

/// Verify a single format: quantize → upload → GPU dequantize → compare vs CPU dequantize.
fn verify_format(
    dtype: GgmlDType,
    kernel_name: &str,
    label: &str,
    n: usize,
    device: &Device,
) -> Result<()> {
    // Generate test data
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
    let data_t = Tensor::from_vec(data, (n,), &Device::Cpu)?;

    // Quantize on CPU
    let qt = QTensor::quantize_onto(&data_t, dtype, &Device::Cpu)?;

    // CPU dequantize (ground truth)
    let ref_bf16 = qt.dequantize(&Device::Cpu)?;
    let ref_f32: Vec<f32> = ref_bf16.to_dtype(DType::F32)?.to_vec1()?;

    // Upload raw quantized bytes to GPU
    let raw_bytes = qt.data()?.to_vec();
    let gpu_bytes = Tensor::from_vec(raw_bytes.clone(), (raw_bytes.len(),), device)?;

    // GPU dequantize
    let gpu_bf16 = quant::dequantize_to_bf16(&gpu_bytes, n, kernel_name)?;
    let gpu_f32: Vec<f32> = gpu_bf16.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1()?;

    // Compare
    assert_eq!(ref_f32.len(), gpu_f32.len());
    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut fail_count = 0usize;
    for (r, g) in ref_f32.iter().zip(gpu_f32.iter()) {
        let err = (r - g).abs();
        max_abs = max_abs.max(err);
        sum_abs += err as f64;
        // BF16 has ~7 bits mantissa, so tolerance is ~2^-7 * max(|r|,|g|) ≈ 0.01 * val
        let tol = 0.01 + 0.01 * r.abs().max(g.abs());
        if err > tol {
            fail_count += 1;
        }
    }
    let mean_abs = (sum_abs / ref_f32.len() as f64) as f32;
    let status = if fail_count == 0 { "PASS" } else { "FAIL" };
    println!(
        "  {label:>5} [{n:>6}]  {status}  max_abs={max_abs:.6}  mean_abs={mean_abs:.6}  fail={fail_count}/{}",
        ref_f32.len()
    );
    Ok(())
}

/// Verify all supported formats.
pub fn verify(device: &Device) -> Result<()> {
    println!("=== GPU Dequantize Correctness ===");
    println!("  Compare: GPU CUDA kernel vs candle CPU dequantize\n");

    let sizes = [256, 1024, 4096];

    for &n in &sizes {
        // 32-element block formats
        verify_format(GgmlDType::Q4_0, "dequantize_q4_0_bf16", "Q4_0", n, device)?;
        verify_format(GgmlDType::Q4_1, "dequantize_q4_1_bf16", "Q4_1", n, device)?;
        verify_format(GgmlDType::Q5_0, "dequantize_q5_0_bf16", "Q5_0", n, device)?;
        verify_format(GgmlDType::Q5_1, "dequantize_q5_1_bf16", "Q5_1", n, device)?;
        verify_format(GgmlDType::Q8_0, "dequantize_q8_0_bf16", "Q8_0", n, device)?;

        // K-quants use 256-element blocks
        if n % 256 == 0 {
            verify_format(GgmlDType::Q2K, "dequantize_q2_K_bf16", "Q2_K", n, device)?;
            verify_format(GgmlDType::Q3K, "dequantize_q3_K_bf16", "Q3_K", n, device)?;
            verify_format(GgmlDType::Q4K, "dequantize_q4_K_bf16", "Q4_K", n, device)?;
            verify_format(GgmlDType::Q5K, "dequantize_q5_K_bf16", "Q5_K", n, device)?;
            verify_format(GgmlDType::Q6K, "dequantize_q6_K_bf16", "Q6_K", n, device)?;
        }
        println!();
    }
    Ok(())
}

// ── Performance benchmarks ───────────────────────────────────────────────

/// Benchmark dequantize throughput for a single format.
fn bench_format(
    dtype: GgmlDType,
    kernel_name: &str,
    label: &str,
    n: usize,
    warmup: usize,
    repeats: usize,
    device: &Device,
) -> Result<f64> {
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
    let data_t = Tensor::from_vec(data, (n,), &Device::Cpu)?;
    let qt = QTensor::quantize_onto(&data_t, dtype, &Device::Cpu)?;

    let raw_bytes = qt.data()?.to_vec();
    let block_bytes = raw_bytes.len();
    let gpu_bytes = Tensor::from_vec(raw_bytes, (block_bytes,), device)?;

    // Warmup
    for _ in 0..warmup {
        let _ = quant::dequantize_to_bf16(&gpu_bytes, n, kernel_name)?;
    }
    device.synchronize()?;

    // Timed runs
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = quant::dequantize_to_bf16(&gpu_bytes, n, kernel_name)?;
    }
    device.synchronize()?;
    let us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    Ok(us)
}

/// Benchmark all formats at LLM-relevant sizes.
pub fn bench(device: &Device) -> Result<()> {
    println!("=== GPU Dequantize Throughput ===");

    let warmup = 20;
    let repeats = 200;

    // Test at typical LLM weight sizes: hidden_size × intermediate_size
    let sizes: &[(usize, &str)] = &[
        (1024 * 4096,  "1K×4K"),
        (4096 * 11008, "4K×11K"),
        (4096 * 4096,  "4K×4K"),
        (5120 * 25600, "5K×26K"),
    ];

    for &(n, label) in sizes {
        print!("  {label:>7} ({:>5.1}M elems)", n as f64 / 1e6);

        for &(dtype, kernel, name) in &[
            (GgmlDType::Q4_0, "dequantize_q4_0_bf16", "Q4_0"),
            (GgmlDType::Q2K,  "dequantize_q2_K_bf16", "Q2_K"),
            (GgmlDType::Q3K,  "dequantize_q3_K_bf16", "Q3_K"),
            (GgmlDType::Q4K,  "dequantize_q4_K_bf16", "Q4_K"),
            (GgmlDType::Q5K,  "dequantize_q5_K_bf16", "Q5_K"),
            (GgmlDType::Q6K,  "dequantize_q6_K_bf16", "Q6_K"),
        ] {
            let us = bench_format(dtype, kernel, name, n, warmup, repeats, device)?;
            let gb_per_sec = (n as f64 * 2.0) / us / 1e3; // BF16 output bytes / time
            print!("  {name}={us:>7.1}us({gb_per_sec:.0}GB/s)");
        }
        println!();
    }

    Ok(())
}

// ── MMVQ correctness + throughput ───────────────────────────────────────

/// Verify MMVQ: quantize weights on CPU → upload to GPU → MMVQ → compare vs dequant+dot.
fn verify_mmvq_format(
    dtype: GgmlDType,
    kernel_name: &str,
    label: &str,
    n: usize,
    k: usize,
    qk: usize,
    device: &Device,
) -> Result<()> {
    // Generate test data
    let w_data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).sin() * 2.0)
        .collect();
    let x_data: Vec<f32> = (0..k)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();

    // Quantize each row on CPU and compute reference dot product
    let mut ref_output = vec![0.0f32; n];
    let mut all_raw_bytes: Vec<u8> = Vec::new();

    for i in 0..n {
        let w_row = &w_data[i * k..(i + 1) * k];
        let w_t = Tensor::from_vec(w_row.to_vec(), (k,), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&w_t, dtype, &Device::Cpu)?;

        // Reference: dequantize → f32 dot product
        let w_deq: Vec<f32> = qt.dequantize(&Device::Cpu)?.to_vec1()?;
        ref_output[i] = w_deq.iter().zip(x_data.iter()).map(|(w, x)| w * x).sum();

        // Collect raw quantized bytes
        all_raw_bytes.extend_from_slice(&qt.data()?.to_vec());
    }

    // Upload to GPU
    let gpu_w = Tensor::from_vec(all_raw_bytes, (n * k / qk * block_bytes(dtype),), device)?;
    let gpu_x = Tensor::from_vec(x_data.clone(), (k,), &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(device)?;

    // MMVQ
    let gpu_y = mmvq::mmvq(&gpu_w, &gpu_x, n, k, kernel_name, qk)?;
    let gpu_output: Vec<f32> = gpu_y.to_device(&Device::Cpu)?.to_vec1()?;

    // Compare
    assert_eq!(ref_output.len(), gpu_output.len());
    let mut max_rel: f32 = 0.0;
    let mut fail_count = 0usize;
    for (r, g) in ref_output.iter().zip(gpu_output.iter()) {
        let err = (r - g).abs();
        let denom = r.abs().max(1e-6);
        let rel = err / denom;
        max_rel = max_rel.max(rel);
        // MMVQ has two quantization steps (weight + activation Q8_1),
        // so tolerance is higher. Near-zero values can have large rel error.
        if rel > 0.15 && err > 0.5 {
            fail_count += 1;
        }
    }
    let status = if fail_count == 0 { "PASS" } else { "FAIL" };
    println!(
        "  {label:>5} [{n:>4}×{k:>5}]  {status}  max_rel={max_rel:.4}  fail={fail_count}/{n}"
    );
    Ok(())
}

/// Return byte size of one quantized block for the given GGML type.
fn block_bytes(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 => 18,
        GgmlDType::Q4_1 => 20,
        GgmlDType::Q5_0 => 22,
        GgmlDType::Q5_1 => 24,
        GgmlDType::Q8_0 => 34,
        GgmlDType::Q2K  => 84,
        GgmlDType::Q3K  => 110,
        GgmlDType::Q4K  => 144,
        GgmlDType::Q5K  => 176,
        GgmlDType::Q6K  => 210,
        _ => panic!("unsupported dtype for block_bytes: {dtype:?}"),
    }
}

/// Verify all MMVQ formats.
pub fn verify_mmvq(device: &Device) -> Result<()> {
    println!("=== GPU MMVQ Correctness ===");
    println!("  Compare: GPU MMVQ (fused quant+dot) vs CPU dequant+f32 dot\n");

    // Test with typical decode-like dimensions
    let configs: &[(usize, usize)] = &[
        (64, 1024),
        (128, 4096),
    ];

    for &(n, k) in configs {
        // Simple formats (QK=32)
        verify_mmvq_format(GgmlDType::Q4_0, "mmvq_q4_0", "Q4_0", n, k, 32, device)?;
        verify_mmvq_format(GgmlDType::Q4_1, "mmvq_q4_1", "Q4_1", n, k, 32, device)?;
        verify_mmvq_format(GgmlDType::Q5_0, "mmvq_q5_0", "Q5_0", n, k, 32, device)?;
        verify_mmvq_format(GgmlDType::Q5_1, "mmvq_q5_1", "Q5_1", n, k, 32, device)?;
        verify_mmvq_format(GgmlDType::Q8_0, "mmvq_q8_0", "Q8_0", n, k, 32, device)?;

        // K-quant formats (QK=256, K must be multiple of 256)
        if k % 256 == 0 {
            verify_mmvq_format(GgmlDType::Q2K, "mmvq_q2_K", "Q2_K", n, k, 256, device)?;
            verify_mmvq_format(GgmlDType::Q3K, "mmvq_q3_K", "Q3_K", n, k, 256, device)?;
            verify_mmvq_format(GgmlDType::Q4K, "mmvq_q4_K", "Q4_K", n, k, 256, device)?;
            verify_mmvq_format(GgmlDType::Q5K, "mmvq_q5_K", "Q5_K", n, k, 256, device)?;
            verify_mmvq_format(GgmlDType::Q6K, "mmvq_q6_K", "Q6_K", n, k, 256, device)?;
        }
        println!();
    }
    Ok(())
}

/// Helper: prepare quantized weights and BF16 activations for benchmarking.
fn prepare_bench_data(
    dtype: GgmlDType,
    n: usize,
    k: usize,
    qk: usize,
    device: &Device,
) -> Result<(Tensor, Vec<u8>)> {
    let w_data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).sin() * 2.0)
        .collect();
    let mut all_raw_bytes: Vec<u8> = Vec::new();
    for i in 0..n {
        let w_row = &w_data[i * k..(i + 1) * k];
        let w_t = Tensor::from_vec(w_row.to_vec(), (k,), &Device::Cpu)?;
        let qt = QTensor::quantize_onto(&w_t, dtype, &Device::Cpu)?;
        all_raw_bytes.extend_from_slice(&qt.data()?.to_vec());
    }
    let gpu_w = Tensor::from_vec(
        all_raw_bytes.clone(),
        (n * k / qk * block_bytes(dtype),),
        device,
    )?;
    Ok((gpu_w, all_raw_bytes))
}

fn dequant_kernel_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4_0 => "dequantize_q4_0_bf16",
        GgmlDType::Q4_1 => "dequantize_q4_1_bf16",
        GgmlDType::Q5_0 => "dequantize_q5_0_bf16",
        GgmlDType::Q5_1 => "dequantize_q5_1_bf16",
        GgmlDType::Q8_0 => "dequantize_q8_0_bf16",
        GgmlDType::Q2K  => "dequantize_q2_K_bf16",
        GgmlDType::Q3K  => "dequantize_q3_K_bf16",
        GgmlDType::Q4K  => "dequantize_q4_K_bf16",
        GgmlDType::Q5K  => "dequantize_q5_K_bf16",
        GgmlDType::Q6K  => "dequantize_q6_K_bf16",
        _ => panic!("unsupported"),
    }
}

/// Baseline: dequantize to BF16 → Tensor::matmul (goes through dispatch GEMM).
/// Includes dequantize time in measurement — fair comparison since weights
/// are stored quantized in real inference.
fn bench_deq_matmul(
    gpu_bytes: &Tensor,
    dtype: GgmlDType,
    gpu_x: &Tensor,
    n: usize,
    k: usize,
    warmup: usize,
    repeats: usize,
    device: &Device,
) -> Result<f64> {
    let dk = dequant_kernel_name(dtype);

    // Warmup (dequantize + matmul each iteration)
    for _ in 0..warmup {
        let w_bf16 = quant::dequantize_to_bf16(gpu_bytes, n * k, dk)?.reshape((n, k))?;
        let _ = gpu_x.matmul(&w_bf16.t()?)?;
    }
    device.synchronize()?;

    let start = Instant::now();
    for _ in 0..repeats {
        let w_bf16 = quant::dequantize_to_bf16(gpu_bytes, n * k, dk)?.reshape((n, k))?;
        let _ = gpu_x.matmul(&w_bf16.t()?)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
}

/// Benchmark MMVQ at LLM decode-relevant sizes (M=1).
/// Compares fused MMVQ vs dequantize→matmul baseline.
pub fn bench_mmvq(device: &Device) -> Result<()> {
    println!("=== GPU MMVQ Throughput (decode M=1) ===");
    println!("  W[N,K] @ x[K] → y[N]");
    println!("  mmvq = fused quantized dot,  deq+mm = dequantize→matmul\n");

    let warmup = 20;
    let repeats = 200;

    let sizes: &[(usize, usize, &str)] = &[
        (4096,  4096,  "4K×4K"),
        (11008, 4096,  "11K×4K"),
        (4096,  11008, "4K×11K"),
    ];

    for &(n, k, label) in sizes {
        print!("  {label:>7}");

        for &(dtype, kernel, name, qk) in &[
            (GgmlDType::Q4_0, "mmvq_q4_0", "Q4_0", 32usize),
            (GgmlDType::Q4K,  "mmvq_q4_K", "Q4_K", 256),
            (GgmlDType::Q6K,  "mmvq_q6_K", "Q6_K", 256),
        ] {
            let (gpu_w, _) = prepare_bench_data(dtype, n, k, qk, device)?;

            let x_data: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
            let gpu_x_1d = Tensor::from_vec(x_data.clone(), (k,), &Device::Cpu)?
                .to_dtype(DType::BF16)?.to_device(device)?;
            let gpu_x_2d = Tensor::from_vec(x_data, (1, k), &Device::Cpu)?
                .to_dtype(DType::BF16)?.to_device(device)?;

            // MMVQ
            for _ in 0..warmup {
                let _ = mmvq::mmvq(&gpu_w, &gpu_x_1d, n, k, kernel, qk)?;
            }
            device.synchronize()?;
            let start = Instant::now();
            for _ in 0..repeats {
                let _ = mmvq::mmvq(&gpu_w, &gpu_x_1d, n, k, kernel, qk)?;
            }
            device.synchronize()?;
            let us_mmvq = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

            // Baseline: dequantize → matmul (includes dequant time)
            let us_deq = bench_deq_matmul(&gpu_w, dtype, &gpu_x_2d, n, k, warmup, repeats, device)?;

            let speedup = us_deq / us_mmvq;
            print!("  {name}: mmvq={us_mmvq:.0}us deq+mm={us_deq:.0}us ({speedup:.1}x)");
        }
        println!();
    }

    Ok(())
}

// ── Tiled MMQ benchmark (vendored llama.cpp) ────────────────────────────

/// Benchmark tiled MMQ (llama.cpp) vs dequant+CUTLASS.
#[cfg(feature = "quant-gemm")]
pub fn bench_tiled_mmq(device: &Device) -> Result<()> {
    use prelude_core::ops::gpu::tiled_mmq;

    println!("=== GPU Tiled MMQ Throughput (llama.cpp vendor) ===");
    println!("  Y[M,N] = X[M,K] @ W[N,K]^T");
    println!("  tiled = llama.cpp MMQ (DP4A+MMA),  deq+mm = dequantize→CUTLASS\n");

    let warmup = 10;
    let repeats = 50;

    let sizes: &[(usize, usize, usize, &str)] = &[
        (32,  4096,  4096,  "32×4K×4K"),
        (128, 4096,  4096,  "128×4K×4K"),
        (512, 4096,  4096,  "512×4K×4K"),
        (128, 11008, 4096,  "128×11K×4K"),
    ];

    for &(m, n, k, label) in sizes {
        print!("  {label:>12}");

        for &(dtype, name, qk) in &[
            (GgmlDType::Q4_0, "Q4_0", 32usize),
            (GgmlDType::Q4K,  "Q4_K", 256),
        ] {
            let (gpu_w, _) = prepare_bench_data(dtype, n, k, qk, device)?;

            let x_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
            let gpu_x_2d = Tensor::from_vec(x_data, (m, k), &Device::Cpu)?
                .to_dtype(DType::BF16)?.to_device(device)?;
            let gpu_x_flat = gpu_x_2d.flatten_all()?;

            // Tiled MMQ (llama.cpp)
            for _ in 0..warmup {
                let _ = tiled_mmq::tiled_mmq(&gpu_w, &gpu_x_flat, m, n, k, dtype)?;
            }
            device.synchronize()?;
            let start = Instant::now();
            for _ in 0..repeats {
                let _ = tiled_mmq::tiled_mmq(&gpu_w, &gpu_x_flat, m, n, k, dtype)?;
            }
            device.synchronize()?;
            let us_tiled = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

            // Baseline: dequantize → matmul
            let us_deq = bench_deq_matmul(&gpu_w, dtype, &gpu_x_2d, n, k, warmup, repeats, device)?;

            let speedup = us_deq / us_tiled;
            print!("  {name}: tiled={us_tiled:.0}us deq+mm={us_deq:.0}us ({speedup:.1}x)");
        }
        println!();
    }

    Ok(())
}
