use super::*;
use super::super::max_sglang_violation;

mod correctness;
mod precision;
mod determinism;

fn to_bf16(v: f32) -> u16 {
    f32_to_bf16(v)
}

fn from_bf16(v: u16) -> f32 {
    bf16_to_f32(v)
}


/// Naive single-head attention for reference.
fn naive_attention(
    q: &[f32],
    ks: &[Vec<f32>],
    vs: &[Vec<f32>],
    sm_scale: f32,
) -> Vec<f32> {
    let head_dim = q.len();
    let seq_len = ks.len();

    let scores: Vec<f32> = ks
        .iter()
        .map(|k| q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum::<f32>() * sm_scale)
        .collect();

    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

    let mut out = vec![0.0f32; head_dim];
    for (w, v) in weights.iter().zip(vs.iter()) {
        for (o, vi) in out.iter_mut().zip(v.iter()) {
            *o += w * vi;
        }
    }
    out
}

/// Naive causal attention for a single token position.
fn naive_causal_attention(
    q_row: &[f32],
    ks: &[Vec<f32>],
    vs: &[Vec<f32>],
    max_key_pos: usize, // inclusive: attend to keys 0..=max_key_pos
    sm_scale: f32,
) -> Vec<f32> {
    let head_dim = q_row.len();
    let scores: Vec<f32> = ks[..=max_key_pos]
        .iter()
        .map(|k| q_row.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum::<f32>() * sm_scale)
        .collect();

    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

    let mut out = vec![0.0f32; head_dim];
    for (w, v) in weights.iter().zip(vs[..=max_key_pos].iter()) {
        for (o, vi) in out.iter_mut().zip(v.iter()) {
            *o += w * vi;
        }
    }
    out
}

/// Run extend attention sequentially (no GemmPool), using auto-detected CAPS.
fn prefill_sequential(
    output: &mut [u16], q: &[u16], k: &[u16], v: &[u16],
    seq_lens: &[usize], num_heads: usize, num_kv_heads: usize,
    head_dim: usize, sm_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;

    // Build per-request offsets
    let mut offsets = Vec::with_capacity(seq_lens.len() + 1);
    offsets.push(0usize);
    for &slen in seq_lens {
        offsets.push(offsets.last().unwrap() + slen);
    }

    let max_slen = seq_lens.iter().max().copied().unwrap_or(0);
    let (block_m, _) = common::select_blocks(max_slen);

    for (req_idx, &slen) in seq_lens.iter().enumerate() {
        let num_mb = (slen + block_m - 1) / block_m;
        for head in 0..num_heads {
            for mb in 0..num_mb {
                prefill_attention_one_head(
                    output, q, k, v, &offsets, seq_lens, req_idx, head,
                    num_heads, num_kv_heads, head_dim, gqa_ratio, sm_scale, mb,
                );
            }
        }
    }
}

/// Run decode attention sequentially (no GemmPool).
fn decode_sequential(
    output: &mut [u16], q: &[u16], k_cache: &[u16], v_cache: &[u16],
    req_to_token: &[i32], seq_lens: &[i64], num_seqs: usize,
    max_context_len: usize, num_heads: usize, num_kv_heads: usize,
    head_dim: usize, sm_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;
    for seq_idx in 0..num_seqs {
        for head in 0..num_heads {
            decode_attention_one_head(
                output, q, k_cache, v_cache, req_to_token, seq_lens,
                seq_idx, head, max_context_len,
                num_heads, num_kv_heads, head_dim, gqa_ratio, sm_scale,
            );
        }
    }
}

/// Generate deterministic test data for a given config.
fn gen_test_data(
    seq_len: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut q = vec![0u16; seq_len * num_heads * head_dim];
    let mut k = vec![0u16; seq_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; seq_len * num_kv_heads * head_dim];
    for i in 0..q.len() { q[i] = to_bf16(((i as f32 * 0.007) - 0.5).sin()); }
    for i in 0..k.len() { k[i] = to_bf16(((i as f32 * 0.013) + 0.2).cos()); }
    for i in 0..v.len() { v[i] = to_bf16(((i as f32 * 0.017) - 0.3).sin()); }
    (q, k, v)
}

/// Assert two u16 slices are bit-exact identical.
fn assert_bit_exact(a: &[u16], b: &[u16], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x, y,
            "{label}: mismatch at index {i}: {x:#06x} vs {y:#06x} ({} vs {})",
            from_bf16(x), from_bf16(y),
        );
    }
}
