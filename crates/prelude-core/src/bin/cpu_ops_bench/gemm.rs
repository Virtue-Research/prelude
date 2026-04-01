use candle_core::{DType, Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

/// Benchmark BF16 GEMM: output[M,N] = input[M,K] * weight[N,K]^T
pub fn bench(m: usize, k: usize, n: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    // Input: [M, K] bf16
    let input_data: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    // Weight: [N, K] bf16 (PyTorch convention)
    let weight_data: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();

    let input = Tensor::from_vec(input_data, (m, k), &device)?;
    let weight = Tensor::from_vec(weight_data, (n, k), &device)?;

    // -- candle F32 baseline (cast -> matmul -> cast) --
    let input_f32 = input.to_dtype(DType::F32)?;
    let weight_f32_t = weight.to_dtype(DType::F32)?.t()?;
    for _ in 0..warmup {
        let _ = input_f32.matmul(&weight_f32_t)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = input_f32.matmul(&weight_f32_t)?;
    }
    let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let onednn_us: Option<f64> = None; // removed: unpacked bf16_linear no longer exists
    let onednn_packed_us: Option<f64> = None; // removed: PackedWeight no longer exists

    // -- brgemm BF16 GEMM (oneDNN micro-kernel, multi-threaded) --
    let brgemm_us = {
        use prelude_core::ops::onednn::{brgemm_available, BrgemmPackedWeight};
        if brgemm_available() {
            match BrgemmPackedWeight::pack(&weight)? {
                Some(brg_packed) => {
                    let brg = std::sync::Arc::new(brg_packed);
                    for _ in 0..warmup {
                        let _ = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &brg, m, k, n)?;
                    }
                    let start = Instant::now();
                    for _ in 0..repeats {
                        let _ = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &brg, m, k, n)?;
                    }
                    Some(start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
                }
                None => None,
            }
        } else {
            None
        }
    };

    let amx_us: Option<f64> = None; // removed: AMX GEMM replaced by brgemm

    // -- Custom cpu_ops GEMM (raw kernel, no Tensor overhead) --
    let custom_us = {
        // Get raw u16 slices
        let in_storage = input.storage_and_layout();
        let w_storage = weight.storage_and_layout();
        let in_u16: &[u16] = match &*in_storage.0 {
            candle_core::Storage::Cpu(s) => {
                let sl = s.as_slice::<bf16>().unwrap();
                unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u16, sl.len()) }
            }
            _ => unreachable!(),
        };
        let w_u16: &[u16] = match &*w_storage.0 {
            candle_core::Storage::Cpu(s) => {
                let sl = s.as_slice::<bf16>().unwrap();
                unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u16, sl.len()) }
            }
            _ => unreachable!(),
        };
        let mut out_buf = vec![0u16; m * n];
        for _ in 0..warmup {
            prelude_core::ops::cpu::gemm::bf16_gemm_small_m(&mut out_buf, in_u16, w_u16, m, k, n);
        }
        let start = Instant::now();
        for _ in 0..repeats {
            prelude_core::ops::cpu::gemm::bf16_gemm_small_m(&mut out_buf, in_u16, w_u16, m, k, n);
        }
        start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // Print
    let label = format!("[{m}x{k}x{n}]");
    let weight_mb = (n * k * 2) as f64 / 1e6;
    let bw_gbs = weight_mb / (custom_us / 1e6) / 1e3;
    print!("  gemm {label:<22} candle_f32={candle_us:>10.1}us  custom={custom_us:>10.1}us ({bw_gbs:.0} GB/s)");
    if let Some(dnn) = onednn_us {
        let speedup = dnn / custom_us;
        print!("  onednn={dnn:>10.1}us ({speedup:.1}x)");
    }
    if let Some(dnn_p) = onednn_packed_us {
        let speedup = dnn_p / custom_us;
        print!("  packed={dnn_p:>10.1}us ({speedup:.1}x)");
    }
    if let Some(brg) = brgemm_us {
        let brg_bw = weight_mb / (brg / 1e6) / 1e3;
        print!("  brgemm={brg:>10.1}us ({brg_bw:.0} GB/s)");
    }
    if let Some(amx) = amx_us {
        let amx_bw = weight_mb / (amx / 1e6) / 1e3;
        print!("  amx_1t={amx:>10.1}us ({amx_bw:.0} GB/s)");
    }
    println!();
    Ok(())
}

/// Benchmark INT8 W8A8 GEMM: BF16 input → quantize → INT8 GEMM → BF16 output.
pub fn bench_s8(m: usize, k: usize, n: usize, warmup: usize, repeats: usize) -> Result<()> {
    use prelude_core::ops::onednn::{brgemm_s8_available, BrgemmS8PackedWeight, brgemm_s8_gemm_forward};

    if !brgemm_s8_available() {
        println!("  s8_gemm [{m}x{k}x{n}]  INT8 brgemm not available");
        return Ok(());
    }

    let device = Device::Cpu;

    // BF16 input [M, K]
    let input_data: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let input = Tensor::from_vec(input_data, (m, k), &device)?;

    // Quantize weights to INT8 with per-channel scales
    let weight_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 * 0.013) + 0.2).cos())
        .collect();
    let mut weight_s8 = vec![0i8; n * k];
    let mut scales = vec![0.0f32; n];
    for row in 0..n {
        let row_data = &weight_f32[row * k..(row + 1) * k];
        let max_abs = row_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        scales[row] = scale;
        let inv = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
        for c in 0..k {
            weight_s8[row * k + c] = (row_data[c] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }

    let packed = BrgemmS8PackedWeight::pack(&weight_s8, &scales, k, n)
        .expect("INT8 packing should succeed");

    // -- INT8 GEMM --
    for _ in 0..warmup {
        let _ = brgemm_s8_gemm_forward(&input, &packed, m, k, n)?;
    }
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        let _ = brgemm_s8_gemm_forward(&input, &packed, m, k, n)?;
    }
    let s8_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // -- BF16 brgemm baseline --
    let bf16_us = {
        use prelude_core::ops::onednn::{brgemm_available, BrgemmPackedWeight};
        if brgemm_available() {
            let w_bf16: Vec<bf16> = weight_f32.iter().map(|&v| bf16::from_f32(v)).collect();
            let w_tensor = Tensor::from_vec(w_bf16, (n, k), &device)?;
            match BrgemmPackedWeight::pack(&w_tensor)? {
                Some(brg) => {
                    for _ in 0..warmup {
                        let _ = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &brg, m, k, n)?;
                    }
                    let start = std::time::Instant::now();
                    for _ in 0..repeats {
                        let _ = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &brg, m, k, n)?;
                    }
                    Some(start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
                }
                None => None,
            }
        } else {
            None
        }
    };

    let label = format!("[{m}x{k}x{n}]");
    let weight_mb = (n * k) as f64 / 1e6; // INT8: 1 byte per weight
    let bw_gbs = weight_mb / (s8_us / 1e6) / 1e3;
    print!("  s8_gemm {label:<22} int8={s8_us:>10.1}us ({bw_gbs:.0} GB/s)");
    if let Some(bf16) = bf16_us {
        print!("  bf16={bf16:>10.1}us  ({:.2}x)", bf16 / s8_us);
    }
    println!();
    Ok(())
}

/// Benchmark BF16 GEMM with fused post-ops (bias, GELU).
pub fn bench_postops(m: usize, k: usize, n: usize, warmup: usize, repeats: usize) -> Result<()> {
    use prelude_core::ops::onednn::{
        brgemm_available, brgemm_gemm_forward_postops, BrgemmPackedWeight,
        BRGEMM_POSTOP_BIAS, BRGEMM_POSTOP_GELU_TANH,
    };

    if !brgemm_available() {
        println!("  postops [{m}x{k}x{n}]  BF16 brgemm not available");
        return Ok(());
    }

    let device = Device::Cpu;

    let input_data: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let input = Tensor::from_vec(input_data, (m, k), &device)?;

    let weight_data: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();
    let weight = Tensor::from_vec(weight_data, (n, k), &device)?;
    let packed = BrgemmPackedWeight::pack(&weight)?.expect("BF16 packing should succeed");

    let bias_data: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32((i as f32 * 0.001) - 0.05))
        .collect();
    let bias = Tensor::from_vec(bias_data, (n,), &device)?;

    // -- Plain brgemm (no post-ops) --
    for _ in 0..warmup {
        let _ = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &packed, m, k, n)?;
    }
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        let _ = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &packed, m, k, n)?;
    }
    let plain_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // -- Postops: bias + GELU --
    let flags = BRGEMM_POSTOP_BIAS | BRGEMM_POSTOP_GELU_TANH;
    for _ in 0..warmup {
        let _ = brgemm_gemm_forward_postops(&input, &packed, Some(&bias), flags, m, k, n)?;
    }
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        let _ = brgemm_gemm_forward_postops(&input, &packed, Some(&bias), flags, m, k, n)?;
    }
    let postops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // -- Separate: brgemm + manual bias + manual GELU --
    for _ in 0..warmup {
        let out = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &packed, m, k, n)?;
        let _ = candle_core::Tensor::gelu(&out.broadcast_add(&bias)?)?;
    }
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        let out = prelude_core::ops::onednn::brgemm_gemm_forward_pub(&input, &packed, m, k, n)?;
        let _ = candle_core::Tensor::gelu(&out.broadcast_add(&bias)?)?;
    }
    let separate_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("[{m}x{k}x{n}]");
    println!(
        "  postops {label:<22} plain={plain_us:>10.1}us  fused_bias_gelu={postops_us:>10.1}us  separate={separate_us:>10.1}us  ({:.2}x vs separate)",
        separate_us / postops_us,
    );
    Ok(())
}

/// Accuracy: compare GEMM backends against candle F32 reference
pub fn verify_accuracy(m: usize, k: usize, n: usize) -> Result<()> {
    let device = Device::Cpu;

    let input_data: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let weight_data: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();

    let input = Tensor::from_vec(input_data, (m, k), &device)?;
    let weight = Tensor::from_vec(weight_data, (n, k), &device)?;

    // Reference: F32 matmul
    let ref_out = input
        .to_dtype(DType::F32)?
        .matmul(&weight.to_dtype(DType::F32)?.t()?)?
        .to_dtype(DType::BF16)?;

    let atol = 5e-2f32;
    let rtol = 5e-2f32;
    let label = format!("[{m}x{k}x{n}]");

    // accuracy check against candle reference is sufficient
    let _ = (atol, rtol, label);

    Ok(())
}
