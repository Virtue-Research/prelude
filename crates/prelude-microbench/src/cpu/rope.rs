use prelude_core::tensor::{Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

fn build_cos_sin_cache(max_pos: usize, rotary_dim: usize) -> Vec<bf16> {
    let embed_dim = rotary_dim / 2;
    (0..max_pos * rotary_dim)
        .map(|idx| {
            let pos = idx / rotary_dim;
            let j = idx % rotary_dim;
            let i = if j < embed_dim { j } else { j - embed_dim };
            let freq = 1.0 / (1_000_000.0f64).powf(2.0 * i as f64 / rotary_dim as f64);
            let theta = pos as f64 * freq;
            let val = if j < embed_dim { theta.cos() } else { theta.sin() };
            bf16::from_f64(val)
        })
        .collect()
}

pub fn bench_all(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    let rope_configs: &[(usize, usize, usize)] = &[(128, 16, 8), (128, 32, 8), (128, 56, 14)];
    let batches = &[1, 4, 16, 64, 256];

    println!("\n=== RoPE ===");
    for &(head_dim, num_heads, num_kv_heads) in rope_configs {
        for &batch in batches {
            bench(report, head_dim, num_heads, num_kv_heads, batch, warmup, repeats)?;
        }
        println!();
    }
    Ok(())
}

fn bench(
    report: &mut BenchReport,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    batch: usize,
    warmup: usize,
    repeats: usize,
) -> Result<()> {
    let device = Device::Cpu;
    let seq_len = 1;
    let rotary_dim = head_dim;
    let max_pos = 2048;

    let cache_data = build_cos_sin_cache(max_pos, rotary_dim);
    let cos_sin_cache = Tensor::from_vec(cache_data, (max_pos, rotary_dim), &device)?;

    let q_data: Vec<bf16> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let k_data: Vec<bf16> = (0..batch * seq_len * num_kv_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();

    let offset = 42usize;

    for _ in 0..warmup {
        let q = Tensor::from_vec(q_data.clone(), (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::from_vec(k_data.clone(), (batch, seq_len, num_kv_heads, head_dim), &device)?;
        let _ = prelude_cpu::ops::cpu_rotary_embedding(&q, &k, &cos_sin_cache, offset, num_heads, num_kv_heads)?;
    }
    let start = Instant::now();
    for _ in 0..repeats {
        let q = Tensor::from_vec(q_data.clone(), (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::from_vec(k_data.clone(), (batch, seq_len, num_kv_heads, head_dim), &device)?;
        let _ = prelude_cpu::ops::cpu_rotary_embedding(&q, &k, &cos_sin_cache, offset, num_heads, num_kv_heads)?;
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("{batch}x h={num_heads}k={num_kv_heads}d={head_dim}");
    println!("  rope [{label:>30}]  cpu_ops={cpu_ops_us:>8.1}us");

    report.add(BenchEntry {
        category: "cpu/rope".into(),
        name: label,
        ours_us: cpu_ops_us,
        baseline_name: None,
        baseline_us: None,
        note: None,
    });
    Ok(())
}
