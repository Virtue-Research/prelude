use half::bf16;
use prelude_core::tensor::{DType, Device, Result, Tensor};
use std::time::Instant;

use super::{BenchResult, print_result};

pub fn bench(dim: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    // Input: [batch, 2*dim] with gate in [-3, 3] and up in [0, 2]
    let input_data: Vec<bf16> = (0..batch * 2 * dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.0031) - 0.7).sin() * 3.0))
        .collect();
    let input = Tensor::from_vec(input_data.clone(), (batch, 2 * dim), &device)?;

    // cpu_ops
    for _ in 0..warmup {
        let _ = prelude_cpu::ops::cpu_silu_and_mul(&input)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_cpu::ops::cpu_silu_and_mul(&input)?;
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // naive F32 (separate silu + mul)
    let input_f32 = input.to_dtype(DType::F32)?;
    let gate = input_f32.narrow(1, 0, dim)?;
    let up = input_f32.narrow(1, dim, dim)?;
    for _ in 0..warmup {
        let _ = (gate.silu()? * &up)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = (gate.silu()? * &up)?;
    }
    let naive_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let sgl_us: Option<f64> = None;

    print_result(
        "silu_mul ",
        dim,
        batch,
        &BenchResult {
            cpu_ops_us,
            naive_us,
            sgl_us,
        },
    );
    Ok(())
}
