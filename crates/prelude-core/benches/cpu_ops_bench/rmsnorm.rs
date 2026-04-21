use candle_core::{DType, Device, Module, Result, Tensor};
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
        let _ = prelude_core::ops::cpu::cpu_rmsnorm(&input, &weight, 1e-6)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_core::ops::cpu::cpu_rmsnorm(&input, &weight, 1e-6)?;
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

pub fn bench_fused(hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;
    let n = batch * hidden;

    let h_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i as f32 * 0.013) - 0.3).cos())).collect();
    let res_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i as f32 * 0.007) + 0.1).sin())).collect();
    let weight_data: Vec<bf16> = (0..hidden).map(|i| bf16::from_f32(0.9 + i as f32 * 0.002)).collect();
    let weight = Tensor::from_vec(weight_data, (hidden,), &device)?;

    // cpu_ops
    for _ in 0..warmup {
        let h = Tensor::from_vec(h_data.clone(), (batch, hidden), &device)?;
        let r = Tensor::from_vec(res_data.clone(), (batch, hidden), &device)?;
        let _ = prelude_core::ops::cpu::cpu_fused_add_rmsnorm(&h, &r, &weight, 1e-6)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let h = Tensor::from_vec(h_data.clone(), (batch, hidden), &device)?;
        let r = Tensor::from_vec(res_data.clone(), (batch, hidden), &device)?;
        let _ = prelude_core::ops::cpu::cpu_fused_add_rmsnorm(&h, &r, &weight, 1e-6)?;
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // candle F32 (separate add + norm)
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let candle_norm = prelude_core::nn_ops::CandleRmsNorm::new(weight_f32, 1e-6);
    let h_f32: Vec<f32> = h_data.iter().map(|v| v.to_f32()).collect();
    let r_f32: Vec<f32> = res_data.iter().map(|v| v.to_f32()).collect();
    for _ in 0..warmup {
        let h = Tensor::from_vec(h_f32.clone(), (batch, hidden), &device)?;
        let r = Tensor::from_vec(r_f32.clone(), (batch, hidden), &device)?;
        let _ = candle_norm.forward(&(&h + &r)?)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let h = Tensor::from_vec(h_f32.clone(), (batch, hidden), &device)?;
        let r = Tensor::from_vec(r_f32.clone(), (batch, hidden), &device)?;
        let _ = candle_norm.forward(&(&h + &r)?)?;
    }
    let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let sgl_us: Option<f64> = None;

    print_result("fused_rms", hidden, batch, &BenchResult { cpu_ops_us, candle_us, sgl_us });
    Ok(())
}
