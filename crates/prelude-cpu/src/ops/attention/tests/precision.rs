use super::*;

/// Verify prefill attention precision against naive F32 reference (strict SGLang tolerance).
/// Uses sequential path to isolate precision from GemmPool scheduling.
fn verify_prefill_precision(
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
    prefill_sequential(
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

/// Verify decode attention against naive F32 reference.
fn verify_decode_config_strict(
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

    decode_sequential(
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
                (0..head_dim)
                    .map(|d| {
                        from_bf16(k_cache[j * num_kv_heads * head_dim + kv_head * head_dim + d])
                    })
                    .collect()
            })
            .collect();
        let vs: Vec<Vec<f32>> = (0..cache_len)
            .map(|j| {
                (0..head_dim)
                    .map(|d| {
                        from_bf16(v_cache[j * num_kv_heads * head_dim + kv_head * head_dim + d])
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
fn test_prefill_precision() {
    verify_prefill_precision(16, 8, 128, 32, "D=128 slen=32");
    verify_prefill_precision(16, 8, 128, 64, "D=128 slen=64");
    verify_prefill_precision(2, 2, 8, 4, "D=8 slen=4");
}

#[test]
fn test_decode_precision() {
    // Decode has no Backend parameter — internally dispatches avx512_bf16 vs F32.
    verify_decode_config_strict(16, 8, 128, 128, "D=128 ctx=128");
    verify_decode_config_strict(16, 8, 128, 512, "D=128 ctx=512");
    verify_decode_config_strict(2, 2, 8, 5, "D=8 ctx=5");
}

/// Compare GemmPool vs sequential vs naive violations.
#[test]
fn test_prefill_gemmpool_vs_sequential() {
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

    // GemmPool path (public API)
    let mut output_pool = vec![0u16; seq_len * num_heads * head_dim];
    prefill_attention_bf16(
        &mut output_pool,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // Sequential path (same auto-detected caps)
    let mut output_seq = vec![0u16; seq_len * num_heads * head_dim];
    prefill_sequential(
        &mut output_seq,
        &q_data,
        &k_data,
        &v_data,
        &[seq_len],
        num_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    );

    // These MUST be identical (same backend, same data, same algorithm)
    let pool_f32: Vec<f32> = output_pool.iter().map(|&v| from_bf16(v)).collect();
    let seq_f32: Vec<f32> = output_seq.iter().map(|&v| from_bf16(v)).collect();
    let pool_vs_seq = max_sglang_violation(&pool_f32, &seq_f32);

    // Also compare both against naive for head 0, last token
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

    let pool_head0: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(output_pool[o_off + d]))
        .collect();
    let seq_head0: Vec<f32> = (0..head_dim)
        .map(|d| from_bf16(output_seq[o_off + d]))
        .collect();
    let pool_vs_naive = max_sglang_violation(&pool_head0, &expected);
    let seq_vs_naive = max_sglang_violation(&seq_head0, &expected);

    eprintln!(
        "[debug] amx={} pool_vs_seq={pool_vs_seq:.6} pool_vs_naive={pool_vs_naive:.6} seq_vs_naive={seq_vs_naive:.6}",
        CAPS.amx
    );

    assert!(
        pool_vs_seq <= 0.0,
        "GemmPool vs sequential: {pool_vs_seq:.6}"
    );
    assert!(
        pool_vs_naive <= 0.0,
        "GemmPool vs naive: {pool_vs_naive:.6}"
    );
    assert!(
        seq_vs_naive <= 0.0,
        "Sequential vs naive: {seq_vs_naive:.6}"
    );
}

/// AMX precision: only runs when oneDNN AMX is available.
/// Since prefill_attention_one_head auto-selects AMX, this just verifies the
/// auto-detected path on AMX-capable machines.
#[test]
fn test_prefill_amx_precision() {
    if !CAPS.amx {
        return;
    }
    verify_prefill_precision(16, 8, 128, 32, "amx D=128 slen=32");
}
