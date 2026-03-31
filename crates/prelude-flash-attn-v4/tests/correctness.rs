//! FA4 kernel correctness tests.
//!
//! Compares FA4 GPU output against a naive CPU attention reference implementation.
//! Uses deterministic sin/cos input patterns for reproducibility.
//!
//! Run: cargo test -p prelude-flash-attn-v4 --test correctness -- --ignored --nocapture

use half::bf16;
use prelude_flash_attn_v4::{KernelDtype, KernelKey, KernelRegistry};
use std::ffi::c_void;

// ── CUDA FFI ────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
}

const H2D: i32 = 1;
const D2H: i32 = 2;

fn cuda_check(code: i32, msg: &str) {
    if code != 0 {
        let err = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(code)) };
        panic!("{msg}: CUDA error {code}: {}", err.to_string_lossy());
    }
}

fn test_stream() -> *mut c_void {
    use std::sync::OnceLock;
    static STREAM: OnceLock<usize> = OnceLock::new();
    let ptr = *STREAM.get_or_init(|| {
        let mut stream: *mut c_void = std::ptr::null_mut();
        cuda_check(unsafe { cudaStreamCreate(&mut stream) }, "cudaStreamCreate");
        stream as usize
    });
    ptr as *mut c_void
}

// ── GPU memory helpers ──────────────────────────────────────────────

fn gpu_alloc(bytes: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "cudaMalloc");
    unsafe { cudaMemset(ptr, 0, bytes); }
    ptr
}

fn gpu_upload<T: Copy>(data: &[T]) -> *mut c_void {
    let bytes = data.len() * std::mem::size_of::<T>();
    let ptr = gpu_alloc(bytes);
    cuda_check(
        unsafe { cudaMemcpy(ptr, data.as_ptr() as _, bytes, H2D) },
        "cudaMemcpy H2D",
    );
    ptr
}

fn gpu_download<T: Copy + Default>(ptr: *mut c_void, count: usize) -> Vec<T> {
    let bytes = count * std::mem::size_of::<T>();
    let mut out = vec![T::default(); count];
    cuda_check(
        unsafe { cudaMemcpy(out.as_mut_ptr() as _, ptr, bytes, D2H) },
        "cudaMemcpy D2H",
    );
    out
}

// ── Naive CPU attention reference ───────────────────────────────────

/// Compute naive attention: softmax(Q * K^T * scale) * V
/// All computation in F32 for reference accuracy.
/// Q: [total_q, num_heads_q, head_dim]
/// K: [total_k, num_heads_k, head_dim]
/// V: [total_k, num_heads_k, head_dim]
fn naive_attention(
    q: &[f32], k: &[f32], v: &[f32],
    num_heads_q: usize, num_heads_k: usize, head_dim: usize,
    cu_seqlens_q: &[i32], cu_seqlens_k: &[i32],
    softmax_scale: f32,
    causal: bool,
    softcap: Option<f32>,
    window_left: Option<i32>,
    _window_right: Option<i32>,
) -> Vec<f32> {
    let total_q = q.len() / (num_heads_q * head_dim);
    let gqa_ratio = num_heads_q / num_heads_k;
    let batch_size = cu_seqlens_q.len() - 1;

    let mut out = vec![0.0f32; total_q * num_heads_q * head_dim];

    for b in 0..batch_size {
        let q_start = cu_seqlens_q[b] as usize;
        let q_end = cu_seqlens_q[b + 1] as usize;
        let k_start = cu_seqlens_k[b] as usize;
        let k_end = cu_seqlens_k[b + 1] as usize;
        let seq_len_q = q_end - q_start;
        let seq_len_k = k_end - k_start;

        // Flash attention convention: Q tokens are at the END of the K sequence.
        let causal_offset = seq_len_k.saturating_sub(seq_len_q);

        for hq in 0..num_heads_q {
            let hk = hq / gqa_ratio;

            for qi in 0..seq_len_q {
                let mut scores = vec![f32::NEG_INFINITY; seq_len_k];

                for ki in 0..seq_len_k {
                    if causal && ki > causal_offset + qi {
                        continue;
                    }
                    if let Some(wl) = window_left {
                        if ((causal_offset + qi) as i32 - ki as i32) > wl {
                            continue;
                        }
                    }

                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = (q_start + qi) * num_heads_q * head_dim + hq * head_dim + d;
                        let k_idx = (k_start + ki) * num_heads_k * head_dim + hk * head_dim + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    let mut score = dot * softmax_scale;

                    // Softcap: score = softcap * tanh(score / softcap)
                    if let Some(cap) = softcap {
                        score = cap * (score / cap).tanh();
                    }

                    scores[ki] = score;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut exp_scores = vec![0.0f32; seq_len_k];
                for ki in 0..seq_len_k {
                    if scores[ki] > f32::NEG_INFINITY {
                        exp_scores[ki] = (scores[ki] - max_score).exp();
                        sum_exp += exp_scores[ki];
                    }
                }
                if sum_exp > 0.0 {
                    for ki in 0..seq_len_k {
                        exp_scores[ki] /= sum_exp;
                    }
                }

                // Weighted sum
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for ki in 0..seq_len_k {
                        let v_idx = (k_start + ki) * num_heads_k * head_dim + hk * head_dim + d;
                        val += exp_scores[ki] * v[v_idx];
                    }
                    let o_idx = (q_start + qi) * num_heads_q * head_dim + hq * head_dim + d;
                    out[o_idx] = val;
                }
            }
        }
    }

    out
}

// ── Test helpers ────────────────────────────────────────────────────

fn generate_deterministic_bf16(count: usize, seed: u32) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            let x = ((i as f32 + seed as f32 * 0.1) * 0.01).sin() * 0.5;
            bf16::from_f32(x)
        })
        .collect()
}

fn bf16_to_f32(data: &[bf16]) -> Vec<f32> {
    data.iter().map(|x| x.to_f32()).collect()
}

fn compare_outputs(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> (f32, f32, usize) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut mismatches = 0;

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        if diff > tol {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "  mismatch at [{i}]: actual={a:.6} expected={e:.6} diff={diff:.6} tol={tol:.6}"
                );
            }
        }
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }

    let mean_diff = if actual.is_empty() { 0.0 } else { sum_diff / actual.len() as f32 };
    (max_diff, mean_diff, mismatches)
}

// ── Non-paged varlen test runner ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_test(
    head_dim: usize,
    num_heads_q: usize,
    num_heads_k: usize,
    cu_seqlens: &[i32],
    causal: bool,
    window_left: Option<i32>,
    window_right: Option<i32>,
    softcap: Option<f32>,
    dtype: KernelDtype,
    atol: f32,
    rtol: f32,
) {
    let registry = KernelRegistry::new();
    let gqa_ratio = num_heads_q / num_heads_k;
    let has_window = window_left.is_some() || window_right.is_some();
    let key = KernelKey::new(head_dim as u32, gqa_ratio as u32, causal, has_window)
        .with_softcap(softcap)
        .with_dtype(dtype);

    let func = match registry.get(&key) {
        Some(f) => f,
        None => {
            eprintln!("SKIP: FA4 kernel not compiled for {:?}", key);
            return;
        }
    };

    let batch_size = cu_seqlens.len() - 1;
    let total_tokens = *cu_seqlens.last().unwrap() as usize;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    let q_elems = total_tokens * num_heads_q * head_dim;
    let k_elems = total_tokens * num_heads_k * head_dim;
    let q_bf16 = generate_deterministic_bf16(q_elems, 1);
    let k_bf16 = generate_deterministic_bf16(k_elems, 2);
    let v_bf16 = generate_deterministic_bf16(k_elems, 3);

    let q_f32 = bf16_to_f32(&q_bf16);
    let k_f32 = bf16_to_f32(&k_bf16);
    let v_f32 = bf16_to_f32(&v_bf16);
    let expected = naive_attention(
        &q_f32, &k_f32, &v_f32,
        num_heads_q, num_heads_k, head_dim,
        cu_seqlens, cu_seqlens,
        softmax_scale, causal, softcap, window_left, window_right,
    );

    let q_gpu = gpu_upload(&q_bf16);
    let k_gpu = gpu_upload(&k_bf16);
    let v_gpu = gpu_upload(&v_bf16);
    let o_gpu = gpu_alloc(q_elems * 2);
    let cu_gpu = gpu_upload(cu_seqlens);

    let q_shape: [i64; 3] = [total_tokens as _, num_heads_q as _, head_dim as _];
    let k_shape: [i64; 3] = [total_tokens as _, num_heads_k as _, head_dim as _];
    let o_shape = q_shape;
    let lse_shape: [i64; 2] = [num_heads_q as _, total_tokens as _];
    let cu_shape: [i64; 1] = [(batch_size + 1) as _];

    let result = unsafe {
        prelude_flash_attn_v4::fa4_varlen_fwd(
            &registry, func,
            q_gpu, k_gpu, v_gpu, o_gpu,
            std::ptr::null_mut(),
            softmax_scale,
            test_stream(),
            cu_gpu, cu_gpu,
            &q_shape, &k_shape, &o_shape, &lse_shape, &cu_shape,
            0, window_left, window_right,
            None, None,
        )
    };
    result.expect("FA4 kernel call failed");

    cuda_check(unsafe { cudaDeviceSynchronize() }, "sync after kernel");
    assert_eq!(unsafe { cudaGetLastError() }, 0, "CUDA error after kernel");

    let out_bf16: Vec<bf16> = gpu_download(o_gpu, q_elems);
    let actual = bf16_to_f32(&out_bf16);

    let (max_diff, mean_diff, mismatches) = compare_outputs(&actual, &expected, atol, rtol);
    eprintln!(
        "  max_diff={max_diff:.6} mean_diff={mean_diff:.6} mismatches={mismatches}/{q_elems}"
    );
    assert_eq!(
        mismatches, 0,
        "FA4 output mismatch: {mismatches}/{q_elems} elements exceed tolerance (atol={atol}, rtol={rtol})"
    );

    unsafe { cudaFree(q_gpu); cudaFree(k_gpu); cudaFree(v_gpu); cudaFree(o_gpu); cudaFree(cu_gpu); }
}

// ── Paged varlen test runner ────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_test_paged(
    head_dim: usize,
    num_heads_q: usize,
    num_heads_k: usize,
    seq_lens_q: &[usize],
    seq_lens_k: &[usize],
    atol: f32,
    rtol: f32,
) {
    let registry = KernelRegistry::new();
    let gqa_ratio = num_heads_q / num_heads_k;

    let block_size = match head_dim {
        d if d <= 128 => 128,
        d if d <= 192 => 112,
        _ => 80,
    };

    let key = KernelKey::new(head_dim as u32, gqa_ratio as u32, true, false)
        .with_paged(true);

    let func = match registry.get(&key) {
        Some(f) => f,
        None => {
            eprintln!("SKIP: FA4 paged kernel not compiled for {:?}", key);
            return;
        }
    };

    let batch_size = seq_lens_q.len();
    assert_eq!(seq_lens_k.len(), batch_size);

    let total_q: usize = seq_lens_q.iter().sum();
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    let mut cu_seqlens_q = vec![0i32; batch_size + 1];
    for (i, &len) in seq_lens_q.iter().enumerate() {
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + len as i32;
    }

    let seqused_k: Vec<i32> = seq_lens_k.iter().map(|&l| l as i32).collect();

    let blocks_per_seq: Vec<usize> = seq_lens_k.iter()
        .map(|&l| (l + block_size - 1) / block_size)
        .collect();
    let max_blocks_per_seq = *blocks_per_seq.iter().max().unwrap();
    let total_blocks: usize = blocks_per_seq.iter().sum();

    let q_elems = total_q * num_heads_q * head_dim;
    let q_bf16 = generate_deterministic_bf16(q_elems, 1);

    let total_k: usize = seq_lens_k.iter().sum();
    let k_flat_bf16 = generate_deterministic_bf16(total_k * num_heads_k * head_dim, 2);
    let v_flat_bf16 = generate_deterministic_bf16(total_k * num_heads_k * head_dim, 3);

    let mut page_table = vec![0i32; batch_size * max_blocks_per_seq];
    let mut next_block = 0usize;
    for seq in 0..batch_size {
        for blk in 0..blocks_per_seq[seq] {
            page_table[seq * max_blocks_per_seq + blk] = next_block as i32;
            next_block += 1;
        }
    }

    let cache_elems = total_blocks * block_size * num_heads_k * head_dim;
    let mut k_cache = vec![bf16::ZERO; cache_elems];
    let mut v_cache = vec![bf16::ZERO; cache_elems];

    let mut k_offset = 0usize;
    for seq in 0..batch_size {
        for tok in 0..seq_lens_k[seq] {
            let blk_idx = tok / block_size;
            let blk_offset = tok % block_size;
            let phys_block = page_table[seq * max_blocks_per_seq + blk_idx] as usize;

            for h in 0..num_heads_k {
                for d in 0..head_dim {
                    let src = (k_offset + tok) * num_heads_k * head_dim + h * head_dim + d;
                    let dst = phys_block * block_size * num_heads_k * head_dim
                        + blk_offset * num_heads_k * head_dim
                        + h * head_dim + d;
                    k_cache[dst] = k_flat_bf16[src];
                    v_cache[dst] = v_flat_bf16[src];
                }
            }
        }
        k_offset += seq_lens_k[seq];
    }

    // CPU reference: flat K/V
    let q_f32 = bf16_to_f32(&q_bf16);
    let k_f32 = bf16_to_f32(&k_flat_bf16);
    let v_f32 = bf16_to_f32(&v_flat_bf16);

    let mut cu_seqlens_k_flat = vec![0i32; batch_size + 1];
    for (i, &len) in seq_lens_k.iter().enumerate() {
        cu_seqlens_k_flat[i + 1] = cu_seqlens_k_flat[i] + len as i32;
    }

    let expected = naive_attention(
        &q_f32, &k_f32, &v_f32,
        num_heads_q, num_heads_k, head_dim,
        &cu_seqlens_q, &cu_seqlens_k_flat,
        softmax_scale, true, None, None, None,
    );

    // GPU
    let q_gpu = gpu_upload(&q_bf16);
    let k_cache_gpu = gpu_upload(&k_cache);
    let v_cache_gpu = gpu_upload(&v_cache);
    let o_gpu = gpu_alloc(q_elems * 2);
    let cu_q_gpu = gpu_upload(&cu_seqlens_q);
    let seqused_k_gpu = gpu_upload(&seqused_k);
    let pt_gpu = gpu_upload(&page_table);

    let q_shape: [i64; 3] = [total_q as _, num_heads_q as _, head_dim as _];
    let k_shape: [i64; 4] = [total_blocks as _, block_size as _, num_heads_k as _, head_dim as _];
    let o_shape: [i64; 3] = q_shape;
    let lse_shape: [i64; 2] = [num_heads_q as _, total_q as _];
    let cu_q_shape: [i64; 1] = [(batch_size + 1) as _];
    let seqused_k_shape: [i64; 1] = [batch_size as _];
    let pt_shape: [i64; 2] = [batch_size as _, max_blocks_per_seq as _];

    let registry2 = KernelRegistry::new();
    registry2.set_stream(0, test_stream());

    let result = unsafe {
        prelude_flash_attn_v4::fa4_varlen_paged_fwd(
            &registry2, func,
            q_gpu, k_cache_gpu, v_cache_gpu, o_gpu,
            std::ptr::null_mut(),
            softmax_scale,
            test_stream(),
            cu_q_gpu,
            seqused_k_gpu,
            pt_gpu,
            &q_shape, &k_shape, &o_shape, &lse_shape,
            &cu_q_shape, &seqused_k_shape, &pt_shape,
            0, None, None,
        )
    };
    result.expect("FA4 paged kernel call failed");

    cuda_check(unsafe { cudaDeviceSynchronize() }, "sync after paged kernel");
    assert_eq!(unsafe { cudaGetLastError() }, 0, "CUDA error after paged kernel");

    let out_bf16: Vec<bf16> = gpu_download(o_gpu, q_elems);
    let actual = bf16_to_f32(&out_bf16);

    let (max_diff, mean_diff, mismatches) = compare_outputs(&actual, &expected, atol, rtol);
    eprintln!(
        "  max_diff={max_diff:.6} mean_diff={mean_diff:.6} mismatches={mismatches}/{q_elems}"
    );
    assert_eq!(
        mismatches, 0,
        "FA4 paged output mismatch: {mismatches}/{q_elems} elements exceed tolerance"
    );

    unsafe {
        cudaFree(q_gpu); cudaFree(k_cache_gpu); cudaFree(v_cache_gpu);
        cudaFree(o_gpu); cudaFree(cu_q_gpu); cudaFree(seqused_k_gpu); cudaFree(pt_gpu);
    }
}

// ── Non-paged tests ────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_single_seq_causal() {
    eprintln!("test_fa4_single_seq_causal: hdim=128, gqa=1, seq=64");
    run_test(128, 8, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_single_seq_noncausal() {
    eprintln!("test_fa4_single_seq_noncausal: hdim=128, gqa=1, seq=64");
    run_test(128, 8, 8, &[0, 64], false, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_multi_seq() {
    eprintln!("test_fa4_multi_seq: hdim=128, gqa=1, seqs=[32,48,16]");
    run_test(128, 8, 8, &[0, 32, 80, 96], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

// ── GQA variants ───────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_gqa2() {
    eprintln!("test_fa4_gqa2: hdim=128, gqa=2, seq=64");
    run_test(128, 16, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_gqa4() {
    eprintln!("test_fa4_gqa4: hdim=128, gqa=4, seq=64");
    run_test(128, 32, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_gqa8() {
    eprintln!("test_fa4_gqa8: hdim=128, gqa=8, seq=64");
    run_test(128, 64, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_gqa16() {
    eprintln!("test_fa4_gqa16: hdim=128, gqa=16, seq=64");
    run_test(128, 128, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_gqa32() {
    eprintln!("test_fa4_gqa32: hdim=128, gqa=32, seq=64");
    run_test(128, 256, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

// ── Head dimension variants ────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_hdim64() {
    eprintln!("test_fa4_hdim64: hdim=64, gqa=1, seq=64");
    run_test(64, 8, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_hdim96() {
    eprintln!("test_fa4_hdim96: hdim=96, gqa=1, seq=64");
    run_test(96, 8, 8, &[0, 64], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_hdim192() {
    eprintln!("test_fa4_hdim192: hdim=192, gqa=1, seq=32");
    run_test(192, 8, 8, &[0, 32], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_hdim256() {
    eprintln!("test_fa4_hdim256: hdim=256, gqa=1, seq=32");
    run_test(256, 8, 8, &[0, 32], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

// ── Sliding window ─────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_window() {
    eprintln!("test_fa4_window: hdim=128, gqa=1, seq=128, window_left=32");
    run_test(128, 8, 8, &[0, 128], true, Some(32), Some(0), None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_window_gqa4() {
    eprintln!("test_fa4_window_gqa4: hdim=128, gqa=4, seq=128, window_left=64");
    run_test(128, 32, 8, &[0, 128], true, Some(64), Some(0), None, KernelDtype::BF16, 1e-2, 1e-2);
}

// ── Softcap ────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_softcap30() {
    eprintln!("test_fa4_softcap30: hdim=128, gqa=1, seq=64, softcap=30.0 (Gemma2)");
    run_test(128, 8, 8, &[0, 64], true, None, None, Some(30.0), KernelDtype::BF16, 2e-2, 2e-2);
}

#[test]
#[ignore]
fn test_fa4_softcap50() {
    eprintln!("test_fa4_softcap50: hdim=128, gqa=1, seq=64, softcap=50.0 (Gemma3)");
    run_test(128, 8, 8, &[0, 64], true, None, None, Some(50.0), KernelDtype::BF16, 2e-2, 2e-2);
}

#[test]
#[ignore]
fn test_fa4_softcap_window() {
    eprintln!("test_fa4_softcap_window: hdim=128, softcap=30.0, window_left=32");
    run_test(128, 8, 8, &[0, 128], true, Some(32), Some(0), Some(30.0), KernelDtype::BF16, 2e-2, 2e-2);
}

// ── FP16 dtype ─────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_fp16() {
    eprintln!("test_fa4_fp16: hdim=128, gqa=1, seq=64, fp16");
    run_test(128, 8, 8, &[0, 64], true, None, None, None, KernelDtype::FP16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_fp16_gqa4() {
    eprintln!("test_fa4_fp16_gqa4: hdim=128, gqa=4, seq=64, fp16");
    run_test(128, 32, 8, &[0, 64], true, None, None, None, KernelDtype::FP16, 1e-2, 1e-2);
}

// ── Longer sequences ───────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_long_seq() {
    eprintln!("test_fa4_long_seq: hdim=128, gqa=4, seq=512");
    run_test(128, 32, 8, &[0, 512], true, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_long_seq_noncausal() {
    eprintln!("test_fa4_long_seq_noncausal: hdim=128, gqa=4, seq=256");
    run_test(128, 32, 8, &[0, 256], false, None, None, None, KernelDtype::BF16, 1e-2, 1e-2);
}

// ── Determinism ────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_determinism() {
    eprintln!("test_fa4_determinism: run twice, assert bitwise equal");

    let registry = KernelRegistry::new();
    let key = KernelKey::new(128, 2, true, false);
    let func = match registry.get(&key) {
        Some(f) => f,
        None => {
            eprintln!("SKIP: kernel not compiled");
            return;
        }
    };

    let total_tokens = 64usize;
    let num_heads_q = 16usize;
    let num_heads_k = 8usize;
    let head_dim = 128usize;
    let q_elems = total_tokens * num_heads_q * head_dim;
    let k_elems = total_tokens * num_heads_k * head_dim;

    let q_bf16 = generate_deterministic_bf16(q_elems, 10);
    let k_bf16 = generate_deterministic_bf16(k_elems, 20);
    let v_bf16 = generate_deterministic_bf16(k_elems, 30);

    let q_gpu = gpu_upload(&q_bf16);
    let k_gpu = gpu_upload(&k_bf16);
    let v_gpu = gpu_upload(&v_bf16);
    let o_gpu = gpu_alloc(q_elems * 2);
    let cu_seqlens: [i32; 2] = [0, total_tokens as i32];
    let cu_gpu = gpu_upload(&cu_seqlens);

    let q_shape: [i64; 3] = [total_tokens as _, num_heads_q as _, head_dim as _];
    let k_shape: [i64; 3] = [total_tokens as _, num_heads_k as _, head_dim as _];
    let o_shape = q_shape;
    let lse_shape: [i64; 2] = [num_heads_q as _, total_tokens as _];
    let cu_shape: [i64; 1] = [2];

    let mut outputs = Vec::new();
    for run in 0..2 {
        // Zero output between runs
        unsafe { cudaMemset(o_gpu, 0, q_elems * 2); }

        unsafe {
            prelude_flash_attn_v4::fa4_varlen_fwd(
                &registry, func,
                q_gpu, k_gpu, v_gpu, o_gpu,
                std::ptr::null_mut(),
                1.0 / (head_dim as f32).sqrt(),
                std::ptr::null_mut(),
                cu_gpu, cu_gpu,
                &q_shape, &k_shape, &o_shape, &lse_shape, &cu_shape,
                0, None, None, None, None,
            )
            .expect("kernel failed");
        }
        cuda_check(unsafe { cudaDeviceSynchronize() }, "sync");

        let out: Vec<bf16> = gpu_download(o_gpu, q_elems);
        outputs.push(out);
        eprintln!("  run {run}: first 4 values = {:?}",
                  &outputs.last().unwrap()[..4].iter().map(|x: &bf16| x.to_f32()).collect::<Vec<_>>());
    }

    let a: Vec<u16> = outputs[0].iter().map(|x: &bf16| x.to_bits()).collect();
    let b: Vec<u16> = outputs[1].iter().map(|x: &bf16| x.to_bits()).collect();
    assert_eq!(a, b, "FA4 kernel is not deterministic (bitwise mismatch)");
    eprintln!("  determinism: PASS (bitwise equal)");

    unsafe { cudaFree(q_gpu); cudaFree(k_gpu); cudaFree(v_gpu); cudaFree(o_gpu); cudaFree(cu_gpu); }
}

// ── Paged KV tests ─────────────────────────────────────────────────

#[test]
#[ignore]
fn test_fa4_paged_hdim128_gqa1() {
    eprintln!("test_fa4_paged_hdim128_gqa1: single seq, 64 Q tokens, 256 KV tokens");
    run_test_paged(128, 8, 8, &[64], &[256], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_hdim128_gqa4() {
    eprintln!("test_fa4_paged_hdim128_gqa4: single seq, 32 Q, 128 KV");
    run_test_paged(128, 32, 8, &[32], &[128], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_hdim128_multi_seq() {
    eprintln!("test_fa4_paged_hdim128_multi_seq: 3 seqs with different lengths");
    run_test_paged(128, 16, 8, &[32, 16, 48], &[256, 128, 384], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_hdim64() {
    eprintln!("test_fa4_paged_hdim64: gqa=2, single seq");
    run_test_paged(64, 16, 8, &[64], &[256], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_hdim96() {
    eprintln!("test_fa4_paged_hdim96: gqa=4, single seq");
    run_test_paged(96, 32, 8, &[64], &[256], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_gqa8() {
    eprintln!("test_fa4_paged_gqa8: hdim=128, gqa=8, multi seq");
    run_test_paged(128, 64, 8, &[32, 64], &[256, 512], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_gqa16() {
    eprintln!("test_fa4_paged_gqa16: hdim=128, gqa=16, single seq");
    run_test_paged(128, 128, 8, &[32], &[256], 1e-2, 1e-2);
}

#[test]
#[ignore]
fn test_fa4_paged_long_kv() {
    eprintln!("test_fa4_paged_long_kv: hdim=128, gqa=4, 16 Q tokens, 2048 KV tokens");
    run_test_paged(128, 32, 8, &[16], &[2048], 1e-2, 1e-2);
}

// NOTE: FA4 paged with Q=1 (decode) produces incorrect results on SM90.
// Our dispatch routes Q=1 decode to FA3, so this is not a production issue.
