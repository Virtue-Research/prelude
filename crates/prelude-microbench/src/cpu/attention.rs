use candle_core::Result;
use half::bf16;
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

pub fn bench_all(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    let extend_configs: &[(usize, usize, usize, usize, usize)] = &[
        (16, 8, 128, 1, 1),
        (16, 8, 128, 4, 1),
        (16, 8, 128, 16, 1),
        (16, 8, 128, 64, 1),
        (16, 8, 128, 128, 1),
        (16, 8, 128, 512, 1),
        (16, 8, 128, 1024, 1),
        (16, 8, 128, 2048, 1),
        (16, 8, 128, 4096, 1),
        (16, 8, 128, 8192, 1),
        (16, 8, 128, 32, 4),
        (16, 8, 128, 8, 8),
        (16, 4, 128, 128, 1),
        (16, 4, 128, 2048, 1),
        (64, 8, 128, 128, 1),
        (64, 8, 128, 1024, 1),
        (64, 8, 128, 4096, 1),
    ];
    let decode_configs: &[(usize, usize, usize, usize, usize)] = &[
        (16, 8, 128, 128, 1),
        (16, 8, 128, 512, 1),
        (16, 8, 128, 1024, 1),
        (16, 8, 128, 2048, 1),
        (16, 8, 128, 4096, 1),
        (16, 8, 128, 8192, 1),
        (16, 8, 128, 128, 4),
        (64, 8, 128, 128, 1),
        (64, 8, 128, 2048, 1),
    ];

    println!("\n=== Attention (extend/prefill) ===");
    for &(nh, nkv, hd, slen, nseq) in extend_configs {
        let r = if slen >= 4096 { 10 } else if slen >= 1024 { 50 } else { repeats.min(200) };
        bench_extend(report, nh, nkv, hd, slen, nseq, warmup.min(5), r)?;
    }

    println!("\n=== Attention (decode) ===");
    for &(nh, nkv, hd, ctx_len, nseq) in decode_configs {
        let r = if ctx_len >= 4096 { 20 } else if ctx_len >= 1024 { 50 } else { repeats.min(200) };
        bench_decode(report, nh, nkv, hd, ctx_len, nseq, warmup.min(5), r)?;
    }
    Ok(())
}

fn bench_extend(
    report: &mut BenchReport,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    seq_len: usize, num_seqs: usize,
    warmup: usize, repeats: usize,
) -> Result<()> {
    let total_tokens = seq_len * num_seqs;
    let sm_scale = 1.0 / (head_dim as f64).sqrt();
    let seq_lens: Vec<usize> = vec![seq_len; num_seqs];

    let q_data: Vec<bf16> = (0..total_tokens * num_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let k_data: Vec<bf16> = (0..total_tokens * num_kv_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();
    let v_data: Vec<bf16> = (0..total_tokens * num_kv_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.017) - 0.3).sin()))
        .collect();

    let q_u16: Vec<u16> = q_data.iter().map(|b| b.to_bits()).collect();
    let k_u16: Vec<u16> = k_data.iter().map(|b| b.to_bits()).collect();
    let v_u16: Vec<u16> = v_data.iter().map(|b| b.to_bits()).collect();
    let mut out = vec![0u16; total_tokens * num_heads * head_dim];

    for _ in 0..warmup {
        prelude_cpu::ops::attention::prefill_attention_bf16(
            &mut out, &q_u16, &k_u16, &v_u16, &seq_lens,
            num_heads, num_kv_heads, head_dim, sm_scale as f32,
        );
    }
    let start = Instant::now();
    for _ in 0..repeats {
        prelude_cpu::ops::attention::prefill_attention_bf16(
            &mut out, &q_u16, &k_u16, &v_u16, &seq_lens,
            num_heads, num_kv_heads, head_dim, sm_scale as f32,
        );
    }
    let us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("{num_seqs}x{seq_len} H={num_heads}/{num_kv_heads} D={head_dim}");
    println!("  extend [{label:<35}]  {us:>10.1}us");

    report.add(BenchEntry {
        category: "cpu/attention/extend".into(),
        name: label,
        ours_us: us,
        baseline_name: None,
        baseline_us: None,
        note: None,
    });
    Ok(())
}

fn bench_decode(
    report: &mut BenchReport,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    cache_len: usize, num_seqs: usize,
    warmup: usize, repeats: usize,
) -> Result<()> {
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    let max_context_len = cache_len + 64;
    let max_total_tokens = num_seqs * max_context_len;

    let q_data: Vec<u16> = (0..num_seqs * num_heads * head_dim)
        .map(|i| f32_to_bf16(((i as f32 * 0.007) - 0.5).sin()))
        .collect();

    let mut k_cache = vec![0u16; max_total_tokens * num_kv_heads * head_dim];
    let mut v_cache = vec![0u16; max_total_tokens * num_kv_heads * head_dim];
    for i in 0..k_cache.len() {
        k_cache[i] = f32_to_bf16(((i as f32 * 0.013) + 0.2).cos());
        v_cache[i] = f32_to_bf16(((i as f32 * 0.017) - 0.3).sin());
    }

    let mut req_to_token = vec![0i32; num_seqs * max_context_len];
    for r in 0..num_seqs {
        let base = r * max_context_len;
        for j in 0..cache_len {
            req_to_token[r * max_context_len + j] = (base + j) as i32;
        }
    }

    let seq_lens: Vec<i64> = vec![cache_len as i64; num_seqs];
    let mut output = vec![0u16; num_seqs * num_heads * head_dim];

    for _ in 0..warmup {
        prelude_cpu::ops::attention::decode_attention_bf16(
            &mut output, &q_data, &k_cache, &v_cache,
            &req_to_token, &seq_lens, num_seqs, max_context_len,
            num_heads, num_kv_heads, head_dim, sm_scale,
        );
    }
    let start = Instant::now();
    for _ in 0..repeats {
        prelude_cpu::ops::attention::decode_attention_bf16(
            &mut output, &q_data, &k_cache, &v_cache,
            &req_to_token, &seq_lens, num_seqs, max_context_len,
            num_heads, num_kv_heads, head_dim, sm_scale,
        );
    }
    let us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!("{num_seqs}req ctx={cache_len} H={num_heads}/{num_kv_heads} D={head_dim}");
    println!("  decode [{label:<40}]  {us:>10.1}us");

    report.add(BenchEntry {
        category: "cpu/attention/decode".into(),
        name: label,
        ours_us: us,
        baseline_name: None,
        baseline_us: None,
        note: None,
    });
    Ok(())
}

#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}
