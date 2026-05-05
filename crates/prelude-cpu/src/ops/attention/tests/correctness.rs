use super::*;

#[test]
fn test_prefill_attention_single_seq() {
    let head_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let seq_len = 4;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    // Generate deterministic test data
    let mut q_data = vec![0u16; seq_len * num_heads * head_dim];
    let mut k_data = vec![0u16; seq_len * num_kv_heads * head_dim];
    let mut v_data = vec![0u16; seq_len * num_kv_heads * head_dim];

    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.1) - 0.5).sin());
    }
    for i in 0..k_data.len() {
        k_data[i] = to_bf16(((i as f32 * 0.13) + 0.2).cos());
    }
    for i in 0..v_data.len() {
        v_data[i] = to_bf16(((i as f32 * 0.07) - 0.3).sin());
    }

    let mut output = vec![0u16; seq_len * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Verify against naive implementation for head 0, last token (has full causal context)
    let head = 0;
    let t = seq_len - 1;
    let q_f32: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(q_data[t * num_heads * head_dim + head * head_dim + d]))
        .collect();
    let ks: Vec<Vec<f32>> = (0..seq_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(k_data[j * num_kv_heads * head_dim + head * head_dim + d]))
                .collect()
        })
        .collect();
    let vs: Vec<Vec<f32>> = (0..seq_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(v_data[j * num_kv_heads * head_dim + head * head_dim + d]))
                .collect()
        })
        .collect();

    let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
    let o_off = t * num_heads * head_dim + head * head_dim;
    let actual: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(output[o_off + d]))
        .collect();

    let violation = max_sglang_violation(&actual, &expected);
    assert!(
        violation <= 0.0,
        "prefill_attention worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
    );
}

#[test]
fn test_prefill_attention_gqa() {
    // GQA: 4 Q heads, 2 KV heads (ratio 2)
    let head_dim = 8;
    let num_heads = 4;
    let num_kv_heads = 2;
    let seq_len = 3;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let mut q_data = vec![0u16; seq_len * num_heads * head_dim];
    let mut k_data = vec![0u16; seq_len * num_kv_heads * head_dim];
    let mut v_data = vec![0u16; seq_len * num_kv_heads * head_dim];

    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.1) - 0.5).sin());
    }
    for i in 0..k_data.len() {
        k_data[i] = to_bf16(((i as f32 * 0.13) + 0.2).cos());
    }
    for i in 0..v_data.len() {
        v_data[i] = to_bf16(((i as f32 * 0.07) - 0.3).sin());
    }

    let mut output = vec![0u16; seq_len * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Verify ALL heads at last token against naive (checks GQA indexing correctness)
    let gqa_ratio = num_heads / num_kv_heads;
    let t = seq_len - 1;
    for head in 0..num_heads {
        let kv_head = head / gqa_ratio;
        let q_f32: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(q_data[t * num_heads * head_dim + head * head_dim + d]))
            .collect();
        let ks: Vec<Vec<f32>> = (0..seq_len)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(k_data[j * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();
        let vs: Vec<Vec<f32>> = (0..seq_len)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(v_data[j * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();

        let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
        let o_off = t * num_heads * head_dim + head * head_dim;
        let actual: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(output[o_off + d]))
            .collect();

        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "GQA head={head} kv_head={kv_head} violation={violation:.6}"
        );
    }
}

#[test]
fn test_prefill_attention_multi_seq() {
    let head_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let seq_lens = [3, 5, 2];
    let total_tokens: usize = seq_lens.iter().sum();
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let mut q_data = vec![0u16; total_tokens * num_heads * head_dim];
    let mut k_data = vec![0u16; total_tokens * num_kv_heads * head_dim];
    let mut v_data = vec![0u16; total_tokens * num_kv_heads * head_dim];

    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.1) - 0.5).sin());
    }
    for i in 0..k_data.len() {
        k_data[i] = to_bf16(((i as f32 * 0.13) + 0.2).cos());
    }
    for i in 0..v_data.len() {
        v_data[i] = to_bf16(((i as f32 * 0.07) - 0.3).sin());
    }

    let mut output = vec![0u16; total_tokens * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output,
        &q_data,
        &k_data,
        &v_data,
        &seq_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Verify last token of EACH sequence, head 0, against naive
    // This checks that multi-seq offset calculation is correct
    let head = 0;
    let mut offset = 0usize;
    for (req_idx, &slen) in seq_lens.iter().enumerate() {
        let t = slen - 1; // last token in this sequence
        let q_f32: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(q_data[(offset + t) * num_heads * head_dim + head * head_dim + d]))
            .collect();
        let ks: Vec<Vec<f32>> = (0..slen)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(
                            k_data[(offset + j) * num_kv_heads * head_dim + head * head_dim + d],
                        )
                    })
                    .collect()
            })
            .collect();
        let vs: Vec<Vec<f32>> = (0..slen)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(
                            v_data[(offset + j) * num_kv_heads * head_dim + head * head_dim + d],
                        )
                    })
                    .collect()
            })
            .collect();

        let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
        let o_off = (offset + t) * num_heads * head_dim + head * head_dim;
        let actual: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(output[o_off + d]))
            .collect();

        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "multi-seq req={req_idx} slen={slen} violation={violation:.6}"
        );
        offset += slen;
    }
}

#[test]
fn test_decode_attention() {
    let head_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let num_seqs = 1;
    let cache_len = 5; // 5 cached tokens
    let max_context_len = 16;
    let max_total_tokens = 32;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    // Q: single token
    let mut q_data = vec![0u16; num_seqs * num_heads * head_dim];
    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.1) - 0.5).sin());
    }

    // KV cache: filled at slots 0..5
    let mut k_cache = vec![0u16; max_total_tokens * num_kv_heads * head_dim];
    let mut v_cache = vec![0u16; max_total_tokens * num_kv_heads * head_dim];
    for slot in 0..cache_len {
        for kh in 0..num_kv_heads {
            for d in 0..head_dim {
                let off = slot * num_kv_heads * head_dim + kh * head_dim + d;
                k_cache[off] = to_bf16(((off as f32 * 0.13) + 0.2).cos());
                v_cache[off] = to_bf16(((off as f32 * 0.07) - 0.3).sin());
            }
        }
    }

    // req_to_token: identity mapping
    let mut req_to_token = vec![0i32; num_seqs * max_context_len];
    for j in 0..cache_len {
        req_to_token[j] = j as i32;
    }

    let seq_lens = vec![cache_len as i64];
    let mut output = vec![0u16; num_seqs * num_heads * head_dim];

    decode_attention_bf16(
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

    // Verify against naive for head 0
    let head = 0;
    let kv_head = 0;
    let q_f32: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(q_data[head * head_dim + d]))
        .collect();
    let ks: Vec<Vec<f32>> = (0..cache_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(k_cache[j * num_kv_heads * head_dim + kv_head * head_dim + d]))
                .collect()
        })
        .collect();
    let vs: Vec<Vec<f32>> = (0..cache_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(v_cache[j * num_kv_heads * head_dim + kv_head * head_dim + d]))
                .collect()
        })
        .collect();

    let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
    let actual: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(output[head * head_dim + d]))
        .collect();

    let violation = max_sglang_violation(&actual, &expected);
    assert!(
        violation <= 0.0,
        "decode_attention worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
    );
}

#[test]
fn test_dot_bf16_f32() {
    let a: Vec<u16> = (0..128).map(|i| to_bf16((i as f32 * 0.01).sin())).collect();
    let b: Vec<u16> = (0..128).map(|i| to_bf16((i as f32 * 0.02).cos())).collect();

    let result = dot_bf16_f32(&a, &b, 128);
    let expected: f32 = (0..128).map(|i| from_bf16(a[i]) * from_bf16(b[i])).sum();

    let diff = (result - expected).abs();
    assert!(
        diff < 1e-3,
        "dot_bf16_f32 diff {diff} too large (result={result}, expected={expected})"
    );
}

#[test]
fn test_prefill_causal_early_tokens() {
    // Verify causal masking for early tokens (not just the last token)
    let head_dim = 128;
    let num_heads = 2;
    let num_kv_heads = 2;
    let seq_len = 8;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let mut q_data = vec![0u16; seq_len * num_heads * head_dim];
    let mut k_data = vec![0u16; seq_len * num_kv_heads * head_dim];
    let mut v_data = vec![0u16; seq_len * num_kv_heads * head_dim];

    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.007) - 0.5).sin());
    }
    for i in 0..k_data.len() {
        k_data[i] = to_bf16(((i as f32 * 0.013) + 0.2).cos());
    }
    for i in 0..v_data.len() {
        v_data[i] = to_bf16(((i as f32 * 0.017) - 0.3).sin());
    }

    let mut output = vec![0u16; seq_len * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Build reference K/V arrays for head 0
    let head = 0;
    let kv_head = 0;
    let ks: Vec<Vec<f32>> = (0..seq_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(k_data[j * num_kv_heads * head_dim + kv_head * head_dim + d]))
                .collect()
        })
        .collect();
    let vs: Vec<Vec<f32>> = (0..seq_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(v_data[j * num_kv_heads * head_dim + kv_head * head_dim + d]))
                .collect()
        })
        .collect();

    // Check ALL token positions (not just last)
    for t in 0..seq_len {
        let q_f32: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(q_data[t * num_heads * head_dim + head * head_dim + d]))
            .collect();
        let expected = naive_causal_attention(&q_f32, &ks, &vs, t, sm_scale);
        let o_off = t * num_heads * head_dim + head * head_dim;
        let actual: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(output[o_off + d]))
            .collect();

        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "token {t}: causal attention worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }
}

#[test]
fn test_prefill_attention_large_dim() {
    // Reproduce accuracy test config: head_dim=128, num_heads=16, num_kv_heads=8, slen=32
    let head_dim = 128;
    let num_heads = 16;
    let num_kv_heads = 8;
    let seq_len = 32;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let mut q_data = vec![0u16; seq_len * num_heads * head_dim];
    let mut k_data = vec![0u16; seq_len * num_kv_heads * head_dim];
    let mut v_data = vec![0u16; seq_len * num_kv_heads * head_dim];

    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.007) - 0.5).sin());
    }
    for i in 0..k_data.len() {
        k_data[i] = to_bf16(((i as f32 * 0.013) + 0.2).cos());
    }
    for i in 0..v_data.len() {
        v_data[i] = to_bf16(((i as f32 * 0.017) - 0.3).sin());
    }

    let mut output = vec![0u16; seq_len * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Verify last token, head 0 against naive
    let head = 0;
    let t = seq_len - 1;
    let kv_head = 0;
    let q_f32: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(q_data[t * num_heads * head_dim + head * head_dim + d]))
        .collect();
    let ks: Vec<Vec<f32>> = (0..seq_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(k_data[j * num_kv_heads * head_dim + kv_head * head_dim + d]))
                .collect()
        })
        .collect();
    let vs: Vec<Vec<f32>> = (0..seq_len)
        .map(|j| {
            (0..head_dim)
                .map(|d| from_bf16(v_data[j * num_kv_heads * head_dim + kv_head * head_dim + d]))
                .collect()
        })
        .collect();

    let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
    let o_off = t * num_heads * head_dim + head * head_dim;
    let actual: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(output[o_off + d]))
        .collect();

    // Check for NaN/Inf first
    for (d, &val) in actual.iter().enumerate() {
        assert!(
            val.is_finite(),
            "output[{d}] is not finite: {val} (expected {})",
            expected[d]
        );
    }

    let violation = max_sglang_violation(&actual, &expected);
    assert!(
        violation <= 0.0,
        "prefill_attention (large) worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
    );
}

/// Helper: verify prefill attention at given config against naive F32 reference.
/// Checks ALL heads at the last token (full causal context).
fn verify_prefill_config(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    label: &str,
) {
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = num_heads / num_kv_heads;

    let mut q_data = vec![0u16; seq_len * num_heads * head_dim];
    let mut k_data = vec![0u16; seq_len * num_kv_heads * head_dim];
    let mut v_data = vec![0u16; seq_len * num_kv_heads * head_dim];

    for i in 0..q_data.len() {
        q_data[i] = to_bf16(((i as f32 * 0.007) - 0.5).sin());
    }
    for i in 0..k_data.len() {
        k_data[i] = to_bf16(((i as f32 * 0.013) + 0.2).cos());
    }
    for i in 0..v_data.len() {
        v_data[i] = to_bf16(((i as f32 * 0.017) - 0.3).sin());
    }

    let mut output = vec![0u16; seq_len * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Verify last token across multiple heads
    let t = seq_len - 1;
    for head in [0, num_heads / 2, num_heads - 1] {
        let kv_head = head / gqa_ratio;
        let q_f32: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(q_data[t * num_heads * head_dim + head * head_dim + d]))
            .collect();
        let ks: Vec<Vec<f32>> = (0..seq_len)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(k_data[j * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();
        let vs: Vec<Vec<f32>> = (0..seq_len)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(v_data[j * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();

        let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
        let o_off = t * num_heads * head_dim + head * head_dim;
        let actual: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(output[o_off + d]))
            .collect();

        for (d, &val) in actual.iter().enumerate() {
            assert!(
                val.is_finite(),
                "{label} head={head} output[{d}] not finite: {val}"
            );
        }
        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "{label} head={head} worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }
}

/// Helper: verify decode attention at given config against naive F32 reference.
fn verify_decode_config(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache_len: usize,
    label: &str,
) {
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = num_heads / num_kv_heads;
    let num_seqs = 1;
    let max_context_len = cache_len + 64;

    let q_data: Vec<u16> = (0..num_seqs * num_heads * head_dim)
        .map(|i| to_bf16(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let mut k_cache = vec![0u16; num_seqs * max_context_len * num_kv_heads * head_dim];
    let mut v_cache = vec![0u16; num_seqs * max_context_len * num_kv_heads * head_dim];
    for i in 0..k_cache.len() {
        k_cache[i] = to_bf16(((i as f32 * 0.013) + 0.2).cos());
        v_cache[i] = to_bf16(((i as f32 * 0.017) - 0.3).sin());
    }

    let req_to_token: Vec<i32> = (0..max_context_len as i32).collect();
    let seq_lens = vec![cache_len as i64];
    let mut output = vec![0u16; num_seqs * num_heads * head_dim];

    decode_attention_bf16(
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

    for head in [0, num_heads / 2, num_heads - 1] {
        let kv_head = head / gqa_ratio;
        let q_f32: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(q_data[head * head_dim + d]))
            .collect();
        let ks: Vec<Vec<f32>> = (0..cache_len)
            .map(|j| {
                let slot = j;
                (0..head_dim)
                    .map(|d| {
                        from_bf16(k_cache[slot * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();
        let vs: Vec<Vec<f32>> = (0..cache_len)
            .map(|j| {
                let slot = j;
                (0..head_dim)
                    .map(|d| {
                        from_bf16(v_cache[slot * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();

        let expected = naive_attention(&q_f32, &ks, &vs, sm_scale);
        let o_off = head * head_dim;
        let actual: Vec<f32> = (0..head_dim)
            .map(|d| from_bf16(output[o_off + d]))
            .collect();

        for (d, &val) in actual.iter().enumerate() {
            assert!(
                val.is_finite(),
                "{label} head={head} output[{d}] not finite: {val}"
            );
        }
        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "{label} head={head} worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }
}

#[test]
fn test_prefill_realistic_configs() {
    // Qwen3-0.6B: H=16, KV=8, D=128
    verify_prefill_config(16, 8, 128, 64, "0.6B slen=64");
    verify_prefill_config(16, 8, 128, 256, "0.6B slen=256");
    verify_prefill_config(16, 8, 128, 512, "0.6B slen=512");
    // Qwen3-1.7B: H=16, KV=4, D=128
    verify_prefill_config(16, 4, 128, 128, "1.7B slen=128");
    verify_prefill_config(16, 4, 128, 512, "1.7B slen=512");
    // Qwen3-32B: H=64, KV=8, D=128
    verify_prefill_config(64, 8, 128, 64, "32B slen=64");
    verify_prefill_config(64, 8, 128, 256, "32B slen=256");
}

#[test]
fn test_decode_realistic_configs() {
    // Qwen3-0.6B
    verify_decode_config(16, 8, 128, 128, "0.6B ctx=128");
    verify_decode_config(16, 8, 128, 512, "0.6B ctx=512");
    verify_decode_config(16, 8, 128, 2048, "0.6B ctx=2048");
    // Qwen3-1.7B
    verify_decode_config(16, 4, 128, 256, "1.7B ctx=256");
    verify_decode_config(16, 4, 128, 1024, "1.7B ctx=1024");
    // Qwen3-32B
    verify_decode_config(64, 8, 128, 128, "32B ctx=128");
    verify_decode_config(64, 8, 128, 512, "32B ctx=512");
}
