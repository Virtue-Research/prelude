use candle_core::{DType, Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

/// Benchmark prefill (extend) attention.
/// Measures: cpu_ops pure Rust, candle F32 baseline.
pub fn bench_extend(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    num_seqs: usize,
    warmup: usize,
    repeats: usize,
) -> Result<()> {
    let total_tokens = seq_len * num_seqs;
    let sm_scale = 1.0 / (head_dim as f64).sqrt();
    let seq_lens: Vec<usize> = vec![seq_len; num_seqs];

    // Generate test data
    let q_data: Vec<bf16> = (0..total_tokens * num_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let k_data: Vec<bf16> = (0..total_tokens * num_kv_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();
    let v_data: Vec<bf16> = (0..total_tokens * num_kv_heads * head_dim)
        .map(|i| bf16::from_f32(((i as f32 * 0.017) - 0.3).sin()))
        .collect();

    let device = Device::Cpu;

    // -- cpu_ops (raw kernel, no Tensor wrapper overhead) --
    let q_u16_cpu: Vec<u16> = q_data.iter().map(|b| b.to_bits()).collect();
    let k_u16_cpu: Vec<u16> = k_data.iter().map(|b| b.to_bits()).collect();
    let v_u16_cpu: Vec<u16> = v_data.iter().map(|b| b.to_bits()).collect();
    let mut out_cpu = vec![0u16; total_tokens * num_heads * head_dim];
    for _ in 0..warmup {
        prelude_core::ops::cpu::attention::prefill_attention_bf16(
            &mut out_cpu, &q_u16_cpu, &k_u16_cpu, &v_u16_cpu, &seq_lens,
            num_heads, num_kv_heads, head_dim, sm_scale as f32,
        );
    }
    let start = Instant::now();
    for _ in 0..repeats {
        prelude_core::ops::cpu::attention::prefill_attention_bf16(
            &mut out_cpu, &q_u16_cpu, &k_u16_cpu, &v_u16_cpu, &seq_lens,
            num_heads, num_kv_heads, head_dim, sm_scale as f32,
        );
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    // -- candle F32 baseline (matmul-based) -- skip for large sequences (too slow) --
    let candle_us: Option<f64> = if seq_len <= 1024 {
        let q_f32 =
            Tensor::from_vec(q_data.clone(), (total_tokens, num_heads, head_dim), &device)?
                .to_dtype(DType::F32)?;
        let k_f32 =
            Tensor::from_vec(k_data.clone(), (total_tokens, num_kv_heads, head_dim), &device)?
                .to_dtype(DType::F32)?;
        let v_f32 =
            Tensor::from_vec(v_data.clone(), (total_tokens, num_kv_heads, head_dim), &device)?
                .to_dtype(DType::F32)?;

        let slen = seq_lens[0];
        let gqa_ratio = num_heads / num_kv_heads;

        for _ in 0..warmup {
            for h in 0..num_heads {
                let kv_h = h / gqa_ratio;
                let q_h = q_f32
                    .narrow(0, 0, slen)?
                    .narrow(1, h, 1)?
                    .squeeze(1)?;
                let k_h = k_f32
                    .narrow(0, 0, slen)?
                    .narrow(1, kv_h, 1)?
                    .squeeze(1)?;
                let v_h = v_f32
                    .narrow(0, 0, slen)?
                    .narrow(1, kv_h, 1)?
                    .squeeze(1)?;
                let scores = q_h.matmul(&k_h.t()?)?;
                let _ = candle_nn::ops::softmax(&(scores * sm_scale)?, 1)?
                    .matmul(&v_h)?;
            }
        }
        let start = Instant::now();
        for _ in 0..repeats {
            for h in 0..num_heads {
                let kv_h = h / gqa_ratio;
                let q_h = q_f32
                    .narrow(0, 0, slen)?
                    .narrow(1, h, 1)?
                    .squeeze(1)?;
                let k_h = k_f32
                    .narrow(0, 0, slen)?
                    .narrow(1, kv_h, 1)?
                    .squeeze(1)?;
                let v_h = v_f32
                    .narrow(0, 0, slen)?
                    .narrow(1, kv_h, 1)?
                    .squeeze(1)?;
                let scores = q_h.matmul(&k_h.t()?)?;
                let _ = candle_nn::ops::softmax(&(scores * sm_scale)?, 1)?
                    .matmul(&v_h)?;
            }
        }
        Some(start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
    } else {
        None
    };

    // Print
    let label = format!(
        "[{num_seqs}x{seq_len} H={num_heads}/{num_kv_heads} D={head_dim}]"
    );
    print!("  extend {label:<40} cpu_ops={cpu_ops_us:>10.1}us");
    if let Some(candle) = candle_us {
        print!("  candle_f32={candle:>10.1}us  ({:.2}x)", candle / cpu_ops_us);
    }
    println!();
    Ok(())
}

/// Benchmark decode attention (single Q token against cached KV).
pub fn bench_decode(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache_len: usize,
    num_seqs: usize,
    warmup: usize,
    repeats: usize,
) -> Result<()> {
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    let max_context_len = cache_len + 64; // some headroom
    let max_total_tokens = num_seqs * max_context_len;

    // Q: [num_seqs, num_heads, head_dim]
    let q_data: Vec<u16> = (0..num_seqs * num_heads * head_dim)
        .map(|i| f32_to_bf16(((i as f32 * 0.007) - 0.5).sin()))
        .collect();

    // KV cache: [max_total_tokens, num_kv_heads, head_dim]
    let mut k_cache: Vec<u16> = vec![0u16; max_total_tokens * num_kv_heads * head_dim];
    let mut v_cache: Vec<u16> = vec![0u16; max_total_tokens * num_kv_heads * head_dim];
    for i in 0..k_cache.len() {
        k_cache[i] = f32_to_bf16(((i as f32 * 0.013) + 0.2).cos());
        v_cache[i] = f32_to_bf16(((i as f32 * 0.017) - 0.3).sin());
    }

    // req_to_token: identity mapping per request
    let mut req_to_token = vec![0i32; num_seqs * max_context_len];
    for r in 0..num_seqs {
        let base = r * max_context_len;
        for j in 0..cache_len {
            req_to_token[r * max_context_len + j] = (base + j) as i32;
        }
    }

    let seq_lens: Vec<i64> = vec![cache_len as i64; num_seqs];
    let mut output = vec![0u16; num_seqs * num_heads * head_dim];

    // -- cpu_ops --
    for _ in 0..warmup {
        prelude_core::ops::cpu::attention::decode_attention_bf16(
            &mut output,
            &q_data,
            &k_cache,
            &v_cache,
            &req_to_token,
            &seq_lens,
            num_seqs,
            max_context_len,
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
        );
    }
    let start = Instant::now();
    for _ in 0..repeats {
        prelude_core::ops::cpu::attention::decode_attention_bf16(
            &mut output,
            &q_data,
            &k_cache,
            &v_cache,
            &req_to_token,
            &seq_lens,
            num_seqs,
            max_context_len,
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
        );
    }
    let cpu_ops_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

    let label = format!(
        "[{num_seqs}req ctx={cache_len} H={num_heads}/{num_kv_heads} D={head_dim}]"
    );
    print!("  decode {label:<40} cpu_ops={cpu_ops_us:>10.1}us");
    println!();
    Ok(())
}

// -- Helpers --

#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}
