use prelude_core::tensor::{DType, Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

use super::{print_result, BenchResult};

pub fn bench(dim: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    // Input: [batch, 2*dim] with gate in [-3, 3] and up in [0, 2]
    let input_data: Vec<bf16> = (0..batch * 2 * dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.0031) - 0.7).sin() * 3.0))
        .collect();
    let input = Tensor::from_vec(input_data.clone(), (batch, 2 * dim), &device)?;

    // cpu_ops
    for _ in 0..warmup {
        let _ = prelude_core::ops::cpu::cpu_silu_and_mul(&input)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_core::ops::cpu::cpu_silu_and_mul(&input)?;
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // candle F32 (separate silu + mul)
    let input_f32 = input.to_dtype(DType::F32)?;
    let gate = input_f32.narrow(1, 0, dim)?;
    let up = input_f32.narrow(1, dim, dim)?;
    for _ in 0..warmup {
        let _ = (prelude_core::nn_ops::ops::silu(&gate)? * &up)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = (prelude_core::nn_ops::ops::silu(&gate)? * &up)?;
    }
    let candle_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let sgl_us: Option<f64> = None;

    print_result("silu_mul ", dim, batch, &BenchResult { cpu_ops_us, candle_us, sgl_us });
    Ok(())
}
