use prelude_core::tensor::{Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

pub fn bench_all(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    let batches = &[1, 4, 16, 64, 256];
    let hiddens = &[896, 1536, 4864, 7168];

    println!("\n=== RMSNorm ===");
    for &hidden in hiddens {
        for &batch in batches {
            bench(report, hidden, batch, warmup, repeats)?;
        }
        println!();
    }

    println!("=== Fused Add+RMSNorm ===");
    for &hidden in hiddens {
        for &batch in batches {
            bench_fused(report, hidden, batch, warmup, repeats)?;
        }
        println!();
    }
    Ok(())
}

fn bench(report: &mut BenchReport, hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    let input_data: Vec<bf16> = (0..batch * hidden)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let weight_data: Vec<bf16> = (0..hidden)
        .map(|i| bf16::from_f32(0.8 + i as f32 * 0.001))
        .collect();
    let input = Tensor::from_vec(input_data, (batch, hidden), &device)?;
    let weight = Tensor::from_vec(weight_data, (hidden,), &device)?;

    for _ in 0..warmup {
        let _ = prelude_cpu::ops::cpu_rmsnorm(&input, &weight, 1e-6)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = prelude_cpu::ops::cpu_rmsnorm(&input, &weight, 1e-6)?;
    }
    let us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("{batch}x{hidden}");
    println!("  rmsnorm   [{label:>10}]  {us:>8.1}us");

    report.add(BenchEntry {
        category: "cpu/rmsnorm".into(),
        name: label,
        ours_us: us,
        baseline_name: None,
        baseline_us: None,
        note: None,
    });
    Ok(())
}

fn bench_fused(report: &mut BenchReport, hidden: usize, batch: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;
    let n = batch * hidden;

    let h_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i as f32 * 0.013) - 0.3).cos())).collect();
    let res_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i as f32 * 0.007) + 0.1).sin())).collect();
    let weight_data: Vec<bf16> = (0..hidden).map(|i| bf16::from_f32(0.9 + i as f32 * 0.002)).collect();
    let weight = Tensor::from_vec(weight_data, (hidden,), &device)?;

    for _ in 0..warmup {
        let h = Tensor::from_vec(h_data.clone(), (batch, hidden), &device)?;
        let r = Tensor::from_vec(res_data.clone(), (batch, hidden), &device)?;
        let _ = prelude_cpu::ops::cpu_fused_add_rmsnorm(&h, &r, &weight, 1e-6)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let h = Tensor::from_vec(h_data.clone(), (batch, hidden), &device)?;
        let r = Tensor::from_vec(res_data.clone(), (batch, hidden), &device)?;
        let _ = prelude_cpu::ops::cpu_fused_add_rmsnorm(&h, &r, &weight, 1e-6)?;
    }
    let us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("{batch}x{hidden}");
    println!("  fused_rms [{label:>10}]  {us:>8.1}us");

    report.add(BenchEntry {
        category: "cpu/fused_rmsnorm".into(),
        name: label,
        ours_us: us,
        baseline_name: None,
        baseline_us: None,
        note: None,
    });
    Ok(())
}
