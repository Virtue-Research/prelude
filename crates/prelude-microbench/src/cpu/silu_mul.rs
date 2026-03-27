use candle_core::{Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

pub fn bench_all(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    let mlp_dims = &[4864, 8960, 38656];
    let batches = &[1, 4, 16, 64, 256];

    println!("\n=== SiLU*Mul ===");
    for &dim in mlp_dims {
        for &batch in batches {
            bench(report, dim, batch, warmup, repeats)?;
        }
        println!();
    }
    Ok(())
}

fn bench(report: &mut BenchReport, dim: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    let input_data: Vec<bf16> = (0..batch * 2 * dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.0031) - 0.7).sin() * 3.0))
        .collect();
    let input = Tensor::from_vec(input_data, (batch, 2 * dim), &device)?;

    for _ in 0..warmup {
        let _ = prelude_core::ops::cpu::cpu_silu_and_mul(&input)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_core::ops::cpu::cpu_silu_and_mul(&input)?;
    }
    let us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("{batch}x{dim}");
    println!("  silu_mul  [{label:>10}]  {us:>8.1}us");

    report.add(BenchEntry {
        category: "cpu/silu_mul".into(),
        name: label,
        ours_us: us,
        baseline_name: None,
        baseline_us: None,
        note: None,
    });
    Ok(())
}
