use prelude_core::tensor::{DType, Device, Module, Result, Tensor};
use half::bf16;
use std::time::Instant;

use super::{print_result, BenchResult};

pub fn bench(hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    let input_data: Vec<bf16> = (0..batch * hidden)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let weight_data: Vec<bf16> = (0..hidden)
        .map(|i| bf16::from_f32(0.8 + i as f32 * 0.001))
        .collect();
    let input = Tensor::from_vec(input_data, (batch, hidden), &device)?;
    let weight = Tensor::from_vec(weight_data, (hidden,), &device)?;

    // cpu_ops
    for _ in 0..warmup {
        let _ = prelude_cpu::ops::cpu_rmsnorm(&input, &weight, 1e-6)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_cpu::ops::cpu_rmsnorm(&input, &weight, 1e-6)?;
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // candle F32
    let input_f32 = input.to_dtype(DType::F32)?;
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let candle_norm = prelude_core::nn_ops::CandleRmsNorm::new(weight_f32, 1e-6);
    for _ in 0..warmup {
        let _ = candle_norm.forward(&input_f32)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = candle_norm.forward(&input_f32)?;
    }
    let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let sgl_us: Option<f64> = None;

    print_result("rmsnorm  ", hidden, batch, &BenchResult { cpu_ops_us, candle_us, sgl_us });
    Ok(())
}

/// Standalone RMSNorm benchmark for F16 and F32 dtypes.
pub fn bench_dtypes(hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    let input_f32: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i as f32 * 0.007) - 0.5).sin())
        .collect();
    let weight_f32: Vec<f32> = (0..hidden)
        .map(|i| 0.8 + i as f32 * 0.001)
        .collect();
    let weight_f32_t = Tensor::from_vec(weight_f32, (hidden,), &device)?;
    let candle_norm = prelude_core::nn_ops::CandleRmsNorm::new(weight_f32_t.clone(), 1e-6);

    for &(dtype, label) in &[(DType::F16, "rms_f16 "), (DType::F32, "rms_f32 ")] {
        let input = Tensor::from_vec(input_f32.clone(), (batch, hidden), &device)?.to_dtype(dtype)?;
        let weight = weight_f32_t.to_dtype(dtype)?;

        for _ in 0..warmup {
            let _ = prelude_cpu::ops::cpu_rmsnorm(&input, &weight, 1e-6)?;
        }
        let start = Instant::now();
        for _ in 0..repeats {
            let _ = prelude_cpu::ops::cpu_rmsnorm(&input, &weight, 1e-6)?;
        }
        let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

        let input_c = input.to_dtype(DType::F32)?;
        for _ in 0..warmup {
            let _ = candle_norm.forward(&input_c)?;
        }
        let start = Instant::now();
        for _ in 0..repeats {
            let _ = candle_norm.forward(&input_c)?;
        }
        let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

        print_result(label, hidden, batch, &BenchResult { cpu_ops_us, candle_us, sgl_us: None });
    }
    Ok(())
}

pub fn bench_fused(hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;
    let n = batch * hidden;

    let h_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i as f32 * 0.013) - 0.3).cos())).collect();
    let res_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i as f32 * 0.007) + 0.1).sin())).collect();
    let weight_data: Vec<bf16> = (0..hidden).map(|i| bf16::from_f32(0.9 + i as f32 * 0.002)).collect();
    let weight = Tensor::from_vec(weight_data, (hidden,), &device)?;

    // cpu_ops — inputs are immutable (&Tensor), no need to clone per iteration
    let h = Tensor::from_vec(h_data.clone(), (batch, hidden), &device)?;
    let r = Tensor::from_vec(res_data.clone(), (batch, hidden), &device)?;
    for _ in 0..warmup {
        let _ = prelude_cpu::ops::cpu_fused_add_rmsnorm(&h, &r, &weight, 1e-6)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_cpu::ops::cpu_fused_add_rmsnorm(&h, &r, &weight, 1e-6)?;
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // candle F32 (separate add + norm) — same: reuse input tensors
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let candle_norm = prelude_core::nn_ops::CandleRmsNorm::new(weight_f32, 1e-6);
    let h_f32 = h.to_dtype(DType::F32)?;
    let r_f32 = r.to_dtype(DType::F32)?;
    for _ in 0..warmup {
        let _ = candle_norm.forward(&(&h_f32 + &r_f32)?)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = candle_norm.forward(&(&h_f32 + &r_f32)?)?;
    }
    let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let sgl_us: Option<f64> = None;

    print_result("fused_rms", hidden, batch, &BenchResult { cpu_ops_us, candle_us, sgl_us });
    Ok(())
}

/// Fused Add+RMSNorm benchmark for F16 and F32 dtypes.
pub fn bench_fused_dtypes(hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;
    let n = batch * hidden;

    let h_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013) - 0.3).cos()).collect();
    let res_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1).sin()).collect();
    let w_f32: Vec<f32> = (0..hidden).map(|i| 0.9 + i as f32 * 0.002).collect();
    let weight_f32 = Tensor::from_vec(w_f32, (hidden,), &device)?;

    for &(dtype, label) in &[(DType::F16, "fused_f16"), (DType::F32, "fused_f32")] {
        let h = Tensor::from_vec(h_f32.clone(), (batch, hidden), &device)?.to_dtype(dtype)?;
        let r = Tensor::from_vec(res_f32.clone(), (batch, hidden), &device)?.to_dtype(dtype)?;
        let w = weight_f32.to_dtype(dtype)?;

        for _ in 0..warmup {
            let _ = prelude_cpu::ops::cpu_fused_add_rmsnorm(&h, &r, &w, 1e-6)?;
        }
        let start = Instant::now();
        for _ in 0..repeats {
            let _ = prelude_cpu::ops::cpu_fused_add_rmsnorm(&h, &r, &w, 1e-6)?;
        }
        let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

        // candle baseline: separate add + norm in the same dtype converted to F32
        let h_c = h.to_dtype(DType::F32)?;
        let r_c = r.to_dtype(DType::F32)?;
        let w_c = w.to_dtype(DType::F32)?;
        let candle_norm = prelude_core::nn_ops::CandleRmsNorm::new(w_c, 1e-6);
        for _ in 0..warmup {
            let _ = candle_norm.forward(&(&h_c + &r_c)?)?;
        }
        let start = Instant::now();
        for _ in 0..repeats {
            let _ = candle_norm.forward(&(&h_c + &r_c)?)?;
        }
        let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

        print_result(label, hidden, batch, &BenchResult { cpu_ops_us, candle_us, sgl_us: None });
    }
    Ok(())
}
