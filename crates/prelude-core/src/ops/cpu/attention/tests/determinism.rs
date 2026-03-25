use super::*;

/// Extend attention (GemmPool multi-threaded): run twice, assert bit-exact.
#[test]
fn test_prefill_determinism() {
    let (head_dim, num_heads, num_kv_heads, seq_len) = (128, 16, 8, 32);
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    let (q, k, v) = gen_test_data(seq_len, num_heads, num_kv_heads, head_dim);

    let mut out1 = vec![0u16; seq_len * num_heads * head_dim];
    let mut out2 = vec![0u16; seq_len * num_heads * head_dim];

    prefill_attention_bf16(&mut out1, &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, sm_scale);
    prefill_attention_bf16(&mut out2, &q, &k, &v, &[seq_len], num_heads, num_kv_heads, head_dim, sm_scale);

    assert_bit_exact(&out1, &out2, "prefill_attention determinism");
}

/// Decode attention (GemmPool multi-threaded): run twice, assert bit-exact.
#[test]
fn test_decode_determinism() {
    let (head_dim, num_heads, num_kv_heads) = (128, 16, 8);
    let num_seqs = 1;
    let cache_len = 128;
    let max_context_len = cache_len + 64;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<u16> = (0..num_seqs * num_heads * head_dim)
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

    let mut out1 = vec![0u16; num_seqs * num_heads * head_dim];
    let mut out2 = vec![0u16; num_seqs * num_heads * head_dim];

    decode_attention_bf16(&mut out1, &q, &k_cache, &v_cache, &req_to_token, &seq_lens, num_seqs, max_context_len, num_heads, num_kv_heads, head_dim, sm_scale);
    decode_attention_bf16(&mut out2, &q, &k_cache, &v_cache, &req_to_token, &seq_lens, num_seqs, max_context_len, num_heads, num_kv_heads, head_dim, sm_scale);

    assert_bit_exact(&out1, &out2, "decode_attention determinism");
}

/// RMSNorm (rayon multi-threaded): run twice, assert bit-exact.
#[test]
fn test_rmsnorm_determinism() {
    use crate::ops::cpu::cpu_rmsnorm;
    use candle_core::{DType, Device, Tensor};

    let hidden = 1024;
    let batch = 16;
    let bf16_vals: Vec<half::bf16> = (0..batch * hidden)
        .map(|i| half::bf16::from_f32(((i as f32 * 0.003) - 0.5).sin()))
        .collect();
    let input = Tensor::from_vec(bf16_vals, &[batch, hidden], &Device::Cpu).unwrap();
    let w_vals: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(0.5 + (i as f32 * 0.001)))
        .collect();
    let weight = Tensor::from_vec(w_vals, &[hidden], &Device::Cpu).unwrap();

    let out1 = cpu_rmsnorm(&input, &weight, 1e-6).unwrap();
    let out2 = cpu_rmsnorm(&input, &weight, 1e-6).unwrap();

    let v1: Vec<f32> = out1.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = out2.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1.len(), v2.len());
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "rmsnorm determinism: mismatch at {i}: {a} vs {b}"
        );
    }
}

/// Fused add+rmsnorm (rayon multi-threaded, in-place mutation): run twice, assert bit-exact.
#[test]
fn test_fused_add_rmsnorm_determinism() {
    use crate::ops::cpu::cpu_fused_add_rmsnorm;
    use candle_core::{DType, Device, Tensor};

    let hidden = 1024;
    let batch = 16;
    let make_tensor = |seed: f32| -> Tensor {
        let vals: Vec<half::bf16> = (0..batch * hidden)
            .map(|i| half::bf16::from_f32(((i as f32 * seed) - 0.5).sin()))
            .collect();
        Tensor::from_vec(vals, &[batch, hidden], &Device::Cpu).unwrap()
    };
    let w_vals: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(0.5 + (i as f32 * 0.001)))
        .collect();
    let weight = Tensor::from_vec(w_vals, &[hidden], &Device::Cpu).unwrap();

    // Run 1
    let input1 = make_tensor(0.003);
    let residual1 = make_tensor(0.007);
    let (res_out1, norm_out1) = cpu_fused_add_rmsnorm(&input1, &residual1, &weight, 1e-6).unwrap();
    let r1: Vec<f32> = res_out1.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let n1: Vec<f32> = norm_out1.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    // Run 2 (fresh tensors, same data)
    let input2 = make_tensor(0.003);
    let residual2 = make_tensor(0.007);
    let (res_out2, norm_out2) = cpu_fused_add_rmsnorm(&input2, &residual2, &weight, 1e-6).unwrap();
    let r2: Vec<f32> = res_out2.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let n2: Vec<f32> = norm_out2.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    for (i, (&a, &b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "fused_add_rmsnorm residual mismatch at {i}: {a} vs {b}");
    }
    for (i, (&a, &b)) in n1.iter().zip(n2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "fused_add_rmsnorm norm mismatch at {i}: {a} vs {b}");
    }
}

/// Linear forward (candle Linear → brgemm dispatch): run twice, assert bit-exact.
/// Tests the full Linear layer path including Tensor allocation/dispatch.
#[cfg(feature = "onednn")]
#[test]
fn test_qwen_linear_determinism() {
    use crate::models::common::linear::Linear;
    use candle_core::{DType, Device, Tensor};

    if !crate::ops::onednn::brgemm_available() { return; }

    let in_dim = 1024;
    let out_dim = 2048;
    let batch = 8;

    // Create a BF16 Linear layer (will auto-pack brgemm weights)
    let w_vals: Vec<half::bf16> = (0..out_dim * in_dim)
        .map(|i| half::bf16::from_f32(((i as f32 * 0.0003) - 0.15).sin()))
        .collect();
    let w = Tensor::from_vec(w_vals, &[out_dim, in_dim], &Device::Cpu).unwrap();
    let linear = Linear::from_weight(w, None).unwrap();

    let make_input = || -> Tensor {
        let vals: Vec<half::bf16> = (0..batch * in_dim)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.002) + 0.1).cos()))
            .collect();
        Tensor::from_vec(vals, &[batch, in_dim], &Device::Cpu).unwrap()
    };

    let out1 = candle_core::Module::forward(&linear, &make_input()).unwrap();
    let out2 = candle_core::Module::forward(&linear, &make_input()).unwrap();

    let v1: Vec<f32> = out1.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = out2.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "Linear determinism mismatch at {i}: {a} vs {b}");
    }
}

/// Chain: rmsnorm → brgemm GEMM → fused_add_rmsnorm → brgemm GEMM.
/// Simulates DecoderLayer forward data flow. Run twice, assert bit-exact.
#[cfg(feature = "onednn")]
#[test]
fn test_layer_chain_determinism() {
    use crate::ops::cpu::{cpu_rmsnorm, cpu_fused_add_rmsnorm};
    use crate::ops::onednn;
    use candle_core::{DType, Device, Tensor};

    if !onednn::brgemm_available() { return; }

    let hidden = 1024;
    let inter = 2048;
    let seq_len = 9; // typical short prompt

    // Synthetic weights
    let make_bf16 = |n: usize, seed: f32| -> Vec<half::bf16> {
        (0..n).map(|i| half::bf16::from_f32(((i as f32 * seed) - 0.3).sin() * 0.1))
            .collect()
    };
    let norm_w = Tensor::from_vec(make_bf16(hidden, 0.001), &[hidden], &Device::Cpu).unwrap();
    let proj_w = Tensor::from_vec(make_bf16(inter * hidden, 0.0003), &[inter, hidden], &Device::Cpu).unwrap();
    let proj_packed = onednn::BrgemmPackedWeight::pack(&proj_w).unwrap().unwrap();
    // down: [hidden, hidden] simulates o_proj or down_proj after narrow
    // Input is [seq_len, hidden] from fused_add_rmsnorm, so K=hidden.
    let down_w = Tensor::from_vec(make_bf16(hidden * hidden, 0.0004), &[hidden, hidden], &Device::Cpu).unwrap();
    let down_packed = onednn::BrgemmPackedWeight::pack(&down_w).unwrap().unwrap();

    // Polluter: run prefill_attention via GemmPool first
    let (head_dim2, nh, nkvh, sl) = (128, 16, 8, 32);
    let sms = 1.0 / (head_dim2 as f32).sqrt();
    let (qq, kk2, vv) = gen_test_data(sl, nh, nkvh, head_dim2);
    let mut ao = vec![0u16; sl * nh * head_dim2];
    prefill_attention_bf16(&mut ao, &qq, &kk2, &vv, &[sl], nh, nkvh, head_dim2, sms);

    let make_input = || -> Tensor {
        let x_vals: Vec<half::bf16> = (0..seq_len * hidden)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.002) + 0.1).cos() * 0.5))
            .collect();
        Tensor::from_vec(x_vals, &[seq_len, hidden], &Device::Cpu).unwrap()
    };

    // Step-by-step determinism check: find first diverging operation
    let x1 = make_input();
    let x2 = make_input();

    // Step 1: RMSNorm
    let h1 = cpu_rmsnorm(&x1, &norm_w, 1e-6).unwrap();
    let h2 = cpu_rmsnorm(&x2, &norm_w, 1e-6).unwrap();
    let s1: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&h1).unwrap().to_vec();
    let s2: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&h2).unwrap().to_vec();
    assert_bit_exact(&s1, &s2, "step1 rmsnorm");

    // Step 2: brgemm GEMM
    let mut p1 = vec![0u16; seq_len * inter];
    let mut p2 = vec![0u16; seq_len * inter];
    unsafe {
        onednn::brgemm_gemm_raw(s1.as_ptr(), &proj_packed, p1.as_mut_ptr(), seq_len, inter);
        onednn::brgemm_gemm_raw(s2.as_ptr(), &proj_packed, p2.as_mut_ptr(), seq_len, inter);
    }
    assert_bit_exact(&p1, &p2, "step2 brgemm");

    // Step 3: fused_add_rmsnorm with narrow (non-contiguous input)
    let pt1 = Tensor::from_vec(
        unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(p1) },
        &[seq_len, inter], &Device::Cpu).unwrap();
    let pt2 = Tensor::from_vec(
        unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(p2) },
        &[seq_len, inter], &Device::Cpu).unwrap();
    let n1 = pt1.narrow(1, 0, hidden).unwrap();
    let n2 = pt2.narrow(1, 0, hidden).unwrap();
    let (_, no1) = cpu_fused_add_rmsnorm(&x1, &n1, &norm_w, 1e-6).unwrap();
    let (_, no2) = cpu_fused_add_rmsnorm(&x2, &n2, &norm_w, 1e-6).unwrap();
    let ns1: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&no1).unwrap().to_vec();
    let ns2: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&no2).unwrap().to_vec();
    assert_bit_exact(&ns1, &ns2, "step3 fused_add_rmsnorm");

    // Step 4: Final GEMM — run 3 times to see the pattern
    let mut f1 = vec![0u16; seq_len * hidden];
    let mut f2 = vec![0u16; seq_len * hidden];
    let mut f3 = vec![0u16; seq_len * hidden];
    unsafe {
        onednn::brgemm_gemm_raw(ns1.as_ptr(), &down_packed, f1.as_mut_ptr(), seq_len, hidden);
        onednn::brgemm_gemm_raw(ns1.as_ptr(), &down_packed, f2.as_mut_ptr(), seq_len, hidden);
        onednn::brgemm_gemm_raw(ns1.as_ptr(), &down_packed, f3.as_mut_ptr(), seq_len, hidden);
    }
    // f2 and f3 should be identical (both after f1 has "warmed up" the cache)
    let diffs_12: usize = f1.iter().zip(f2.iter()).filter(|(a,b)| a != b).count();
    let diffs_23: usize = f2.iter().zip(f3.iter()).filter(|(a,b)| a != b).count();
    let diffs_13: usize = f1.iter().zip(f3.iter()).filter(|(a,b)| a != b).count();
    eprintln!("[det] step4: f1≠f2={diffs_12} f2≠f3={diffs_23} f1≠f3={diffs_13} (total={})", seq_len * hidden);
    assert_eq!(diffs_12, 0, "step4: f1 vs f2 differ by {diffs_12} elements");
    assert_eq!(diffs_23, 0, "step4: f2 vs f3 differ by {diffs_23} elements");
}

/// Model-like chain: rmsnorm → Linear → fused_add_rmsnorm → Linear.
/// Run twice, assert bit-exact. Passes alone but FAILS after test_prefill_amx_precision:
///   cargo test -p prelude-core --lib -- "test_extend_amx" "test_linear_chain" --test-threads=1
/// Root cause: 1 ULP non-determinism in brgemm_bf16_linear (C++ oneDNN FFI) when
/// the test harness runs verify_prefill_precision (brgemm attention + naive comparison)
/// in a prior #[test]. Cannot be reproduced within a single test function.
/// The non-determinism is in Linear::forward's second call vs first call.
#[cfg(feature = "onednn")]
#[test]
fn test_linear_chain_determinism() {
    use crate::ops::cpu::{cpu_rmsnorm, cpu_fused_add_rmsnorm};
    use crate::models::common::linear::Linear;
    use candle_core::{DType, Device, Module, Tensor};

    if !crate::ops::onednn::brgemm_available() { return; }

    // Trigger CAPS LazyLock initialization (includes brgemm_available() probe)
    let _ = CAPS.amx;

    let hidden = 1024;
    let inter = 2048;
    let seq_len = 9;

    let make_bf16_tensor = |rows: usize, cols: usize, seed: f32| -> Tensor {
        let vals: Vec<half::bf16> = (0..rows * cols)
            .map(|i| half::bf16::from_f32(((i as f32 * seed) - 0.3).sin() * 0.1))
            .collect();
        Tensor::from_vec(vals, &[rows, cols], &Device::Cpu).unwrap()
    };

    let norm_w = make_bf16_tensor(1, hidden, 0.001).reshape(&[hidden]).unwrap();
    let proj = Linear::from_weight(make_bf16_tensor(inter, hidden, 0.0003), None).unwrap();
    // down: [hidden, hidden] — input from fused_add_rmsnorm is [seq_len, hidden]
    // (Previously [hidden, inter] caused K=inter mismatch: brgemm read past buffer)
    let down = Linear::from_weight(make_bf16_tensor(hidden, hidden, 0.0004), None).unwrap();

    // Step-by-step: find first diverging operation
    let x1 = make_bf16_tensor(seq_len, hidden, 0.002);
    let x2 = make_bf16_tensor(seq_len, hidden, 0.002);

    let h1 = cpu_rmsnorm(&x1, &norm_w, 1e-6).unwrap();
    let h2 = cpu_rmsnorm(&x2, &norm_w, 1e-6).unwrap();
    let s1: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&h1).unwrap().to_vec();
    let s2: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&h2).unwrap().to_vec();
    assert_bit_exact(&s1, &s2, "step1 rmsnorm");

    let p1 = proj.forward(&h1).unwrap();
    let p2 = proj.forward(&h2).unwrap();
    let ps1: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&p1).unwrap().to_vec();
    let ps2: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&p2).unwrap().to_vec();
    assert_bit_exact(&ps1, &ps2, "step2 proj.forward (OnednnLinear)");

    // Step 3: fused_add_rmsnorm with narrow (must make contiguous first)
    let n1 = p1.narrow(1, 0, hidden).unwrap().contiguous().unwrap();
    let n2 = p2.narrow(1, 0, hidden).unwrap().contiguous().unwrap();
    let (_, no1) = cpu_fused_add_rmsnorm(&x1, &n1, &norm_w, 1e-6).unwrap();
    let (_, no2) = cpu_fused_add_rmsnorm(&x2, &n2, &norm_w, 1e-6).unwrap();
    let ns1: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&no1).unwrap().to_vec();
    let ns2: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&no2).unwrap().to_vec();
    assert_bit_exact(&ns1, &ns2, "step3 fused_add_rmsnorm");

    // Step 4: down.forward (OnednnLinear)
    assert_bit_exact(&ns1, &ns2, "step4 input check");
    let f1 = down.forward(&no1).unwrap();
    let f2 = down.forward(&no2).unwrap();
    let fs1: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&f1).unwrap().to_vec();
    let fs2: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&f2).unwrap().to_vec();
    assert_bit_exact(&fs1, &fs2, "step4 down.forward (OnednnLinear)");
}


/// brgemm GEMM with different shapes interleaved: tests if JIT cache / scratchpad
/// residuals cause non-determinism. Simulates what happens when different layers
/// (with different M) use the same GemmPool thread.
#[cfg(feature = "onednn")]
#[test]
fn test_brgemm_interleaved_shapes_determinism() {
    use crate::ops::onednn;
    use candle_core::{Device, Tensor};

    if !onednn::brgemm_available() { return; }

    let k = 1024;
    let n = 2048;

    let w_bf16: Vec<half::bf16> = (0..n * k)
        .map(|i| half::bf16::from_f32(((i as f32 * 0.0003) - 0.15).sin()))
        .collect();
    let w = Tensor::from_vec(w_bf16, &[n, k], &Device::Cpu).unwrap();
    let packed = onednn::BrgemmPackedWeight::pack(&w).unwrap().unwrap();

    let make_input = |m: usize| -> Vec<u16> {
        (0..m * k).map(|i| to_bf16(((i as f32 * 0.002) + 0.1).cos())).collect()
    };

    // Run with M=9 (our target shape)
    let input9 = make_input(9);
    let mut out_before = vec![0u16; 9 * n];
    unsafe { onednn::brgemm_gemm_raw(input9.as_ptr(), &packed, out_before.as_mut_ptr(), 9, n); }

    // Pollute: run a different shape (M=32) to change thread-local brgemm state
    let input32 = make_input(32);
    let mut _discard = vec![0u16; 32 * n];
    unsafe { onednn::brgemm_gemm_raw(input32.as_ptr(), &packed, _discard.as_mut_ptr(), 32, n); }

    // Run M=9 again — must match the first run
    let mut out_after = vec![0u16; 9 * n];
    unsafe { onednn::brgemm_gemm_raw(input9.as_ptr(), &packed, out_after.as_mut_ptr(), 9, n); }

    assert_bit_exact(&out_before, &out_after, "brgemm interleaved shapes");
}

/// brgemm GEMM (GemmPool 2D M×N dispatch): run twice, assert bit-exact.
/// This is the Linear layer's hot path — the most likely source of non-determinism.
#[cfg(feature = "onednn")]
#[test]
fn test_brgemm_gemm_determinism() {
    use crate::ops::onednn;
    use candle_core::{Device, Tensor};

    if !onednn::brgemm_available() { return; }

    let m = 16; // batch tokens
    let k = 1024; // hidden_size
    let n = 2048; // intermediate_size

    // Create packed weight via Tensor API
    let w_bf16: Vec<half::bf16> = (0..k * n)
        .map(|i| half::bf16::from_f32(((i as f32 * 0.001) - 0.3).sin()))
        .collect();
    let w_tensor = Tensor::from_vec(w_bf16, &[n, k], &Device::Cpu).unwrap();
    let packed = onednn::BrgemmPackedWeight::pack(&w_tensor).unwrap();
    let packed = match packed {
        Some(p) => p,
        None => { eprintln!("brgemm pack not available, skipping"); return; }
    };

    // Input
    let input: Vec<u16> = (0..m * k)
        .map(|i| to_bf16(((i as f32 * 0.002) + 0.1).cos()))
        .collect();

    let mut out1 = vec![0u16; m * n];
    let mut out2 = vec![0u16; m * n];

    unsafe {
        onednn::brgemm_gemm_raw(input.as_ptr(), &packed, out1.as_mut_ptr(), m, n);
        onednn::brgemm_gemm_raw(input.as_ptr(), &packed, out2.as_mut_ptr(), m, n);
    }

    assert_bit_exact(&out1, &out2, "brgemm_gemm determinism");
}

/// Multi-layer decoder chain: simulates N transformer layers.
/// Each layer: rmsnorm → attn(prefill) → fused_add_rmsnorm → gate_up GEMM → SiLU → down GEMM → residual_add.
/// Uses Qwen3-0.6B dimensions: hidden=1024, inter=3072, heads=16, kv_heads=8, head_dim=64.
/// Run full chain twice with same input, assert bit-exact output.
#[cfg(feature = "onednn")]
#[test]
fn test_multi_layer_determinism() {
    use crate::ops::cpu::{cpu_rmsnorm, cpu_fused_add_rmsnorm, cpu_silu_and_mul};
    use crate::ops::onednn;
    use candle_core::{DType, Device, Tensor};

    if !onednn::brgemm_available() { return; }

    // Qwen3-0.6B-like dimensions
    let hidden = 1024;
    let inter = 3072; // intermediate_size (Qwen3-0.6B = 3072)
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = hidden / num_heads; // 64
    let num_layers = 4; // keep fast; 28 layers also passes
    let seq_len = 9;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let make_bf16 = |n: usize, seed: f32| -> Vec<half::bf16> {
        (0..n).map(|i| half::bf16::from_f32(((i as f32 * seed) - 0.3).sin() * 0.1))
            .collect()
    };
    let make_tensor = |rows: usize, cols: usize, seed: f32| -> Tensor {
        Tensor::from_vec(make_bf16(rows * cols, seed), &[rows, cols], &Device::Cpu).unwrap()
    };

    // Create per-layer weights (shared across both runs)
    struct LayerWeights {
        norm1_w: Tensor,
        norm2_w: Tensor,
        // Attention: QKV → attention → O_proj
        qkv_packed: onednn::BrgemmPackedWeight,
        oproj_packed: onednn::BrgemmPackedWeight,
        // MLP: gate_up → SiLU×Mul → down
        gate_up_packed: onednn::BrgemmPackedWeight,
        down_packed: onednn::BrgemmPackedWeight,
    }

    let q_size = num_heads * head_dim;
    let kv_size = num_kv_heads * head_dim;
    let qkv_n = q_size + 2 * kv_size; // 1024 + 2*512 = 2048

    let mut layers = Vec::new();
    for l in 0..num_layers {
        let seed_base = (l + 1) as f32 * 0.0001;
        let norm1_w = make_tensor(1, hidden, seed_base + 0.001).reshape(&[hidden]).unwrap();
        let norm2_w = make_tensor(1, hidden, seed_base + 0.002).reshape(&[hidden]).unwrap();
        let qkv_packed = onednn::BrgemmPackedWeight::pack(
            &make_tensor(qkv_n, hidden, seed_base + 0.003)).unwrap().unwrap();
        let oproj_packed = onednn::BrgemmPackedWeight::pack(
            &make_tensor(hidden, q_size, seed_base + 0.004)).unwrap().unwrap();
        let gate_up_packed = onednn::BrgemmPackedWeight::pack(
            &make_tensor(2 * inter, hidden, seed_base + 0.005)).unwrap().unwrap();
        let down_packed = onednn::BrgemmPackedWeight::pack(
            &make_tensor(hidden, inter, seed_base + 0.006)).unwrap().unwrap();
        layers.push(LayerWeights {
            norm1_w, norm2_w, qkv_packed, oproj_packed, gate_up_packed, down_packed,
        });
    }

    let final_norm_w = make_tensor(1, hidden, 0.009).reshape(&[hidden]).unwrap();

    /// Run the full chain and return hidden states as Vec<u16>.
    fn run_chain(
        input: &Tensor,
        layers: &[LayerWeights],
        final_norm_w: &Tensor,
        seq_len: usize,
        hidden: usize,
        inter: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
        eps: f64,
    ) -> Vec<u16> {
        let mut h = input.clone();

        for (i, lw) in layers.iter().enumerate() {
            // Step 1: RMSNorm
            let normed = cpu_rmsnorm(&h, &lw.norm1_w, eps).unwrap();
            let normed_raw: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&normed).unwrap().to_vec();

            // Step 2: QKV GEMM → [seq_len, qkv_n]
            let q_size = num_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let qkv_n = q_size + 2 * kv_size;
            let mut qkv = vec![0u16; seq_len * qkv_n];
            unsafe {
                onednn::brgemm_gemm_raw(
                    normed_raw.as_ptr(), &lw.qkv_packed, qkv.as_mut_ptr(), seq_len, qkv_n);
            }

            // Step 3: Split Q/K/V (simplified: no norm/RoPE, just slice)
            let mut q_data = vec![0u16; seq_len * q_size];
            let mut k_data = vec![0u16; seq_len * kv_size];
            let mut v_data = vec![0u16; seq_len * kv_size];
            for t in 0..seq_len {
                let base = t * qkv_n;
                q_data[t * q_size..(t + 1) * q_size].copy_from_slice(&qkv[base..base + q_size]);
                k_data[t * kv_size..(t + 1) * kv_size].copy_from_slice(&qkv[base + q_size..base + q_size + kv_size]);
                v_data[t * kv_size..(t + 1) * kv_size].copy_from_slice(&qkv[base + q_size + kv_size..base + qkv_n]);
            }

            // Step 4: Attention
            let mut attn_out = vec![0u16; seq_len * q_size];
            prefill_attention_bf16(
                &mut attn_out, &q_data, &k_data, &v_data,
                &[seq_len], num_heads, num_kv_heads, head_dim, sm_scale,
            );

            // Step 5: O_proj GEMM → [seq_len, hidden]
            let mut proj_out = vec![0u16; seq_len * hidden];
            unsafe {
                onednn::brgemm_gemm_raw(
                    attn_out.as_ptr(), &lw.oproj_packed, proj_out.as_mut_ptr(), seq_len, hidden);
            }

            // Step 6: fused_add_rmsnorm(residual=h, attn_out=proj_out)
            let proj_tensor = {
                let vals: Vec<half::bf16> = unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(proj_out) };
                Tensor::from_vec(vals, &[seq_len, hidden], &Device::Cpu).unwrap()
            };
            let (x_res, h2) = cpu_fused_add_rmsnorm(&h, &proj_tensor, &lw.norm2_w, eps).unwrap();

            // Step 7: gate_up GEMM → [seq_len, 2*inter]
            let h2_raw: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&h2).unwrap().to_vec();
            let mut gate_up = vec![0u16; seq_len * 2 * inter];
            unsafe {
                onednn::brgemm_gemm_raw(
                    h2_raw.as_ptr(), &lw.gate_up_packed, gate_up.as_mut_ptr(), seq_len, 2 * inter);
            }

            // Step 8: SiLU×Mul → [seq_len, inter]
            let gu_tensor = {
                let vals: Vec<half::bf16> = unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(gate_up) };
                Tensor::from_vec(vals, &[seq_len, 2 * inter], &Device::Cpu).unwrap()
            };
            let silu_out = cpu_silu_and_mul(&gu_tensor).unwrap();

            // Step 9: down GEMM → [seq_len, hidden]
            let silu_raw: Vec<u16> = crate::ops::cpu::tensor_as_u16_slice_pub(&silu_out).unwrap().to_vec();
            let mut mlp_out = vec![0u16; seq_len * hidden];
            unsafe {
                onednn::brgemm_gemm_raw(
                    silu_raw.as_ptr(), &lw.down_packed, mlp_out.as_mut_ptr(), seq_len, hidden);
            }

            // Step 10: residual add: h = x_res + mlp_out
            let mlp_tensor = {
                let vals: Vec<half::bf16> = unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(mlp_out) };
                Tensor::from_vec(vals, &[seq_len, hidden], &Device::Cpu).unwrap()
            };
            h = (x_res + mlp_tensor).unwrap();
        }

        // Final norm
        let out = cpu_rmsnorm(&h, final_norm_w, eps).unwrap();
        crate::ops::cpu::tensor_as_u16_slice_pub(&out).unwrap().to_vec()
    }

    let input = make_tensor(seq_len, hidden, 0.002);
    let eps = 1e-6;

    let result1 = run_chain(
        &input, &layers, &final_norm_w,
        seq_len, hidden, inter, num_heads, num_kv_heads, head_dim, sm_scale, eps);
    let result2 = run_chain(
        &input, &layers, &final_norm_w,
        seq_len, hidden, inter, num_heads, num_kv_heads, head_dim, sm_scale, eps);

    let diffs: Vec<usize> = result1.iter().zip(result2.iter()).enumerate()
        .filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
    if !diffs.is_empty() {
        let i = diffs[0];
        eprintln!("[multi-layer] {} diffs, first at {i}: {:#06x} vs {:#06x} ({} vs {})",
            diffs.len(), result1[i], result2[i],
            from_bf16(result1[i]), from_bf16(result2[i]));
    }
    assert!(diffs.is_empty(), "multi-layer chain: {} diffs out of {}", diffs.len(), result1.len());
}
