//! Pure Rust CPU attention kernels for BF16 tensors.
//!
//! Replaces sgl-kernel's `extend_attention_cpu` (`prefill_attention_bf16`) and `decode_attention_cpu`
//! without any external C/C++ or libtorch dependency.
//!
//! Two entry points:
//!   - `prefill_attention_bf16` — prefill (multi-token) with causal mask (was `extend_attention_bf16`)
//!   - `decode_attention_bf16`  — decode (single-token Q) against cached KV
//!
//! All data is BF16 (u16 bit patterns), accumulation is F32.

mod buffers;
mod common;
#[cfg(target_arch = "x86_64")]
mod dpbf16;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod small;

#[cfg(target_arch = "x86_64")]
use dpbf16::*;
#[cfg(target_arch = "x86_64")]
use avx512::*;

use buffers::{ensure_len_f32, ensure_len_u16, get_attn_bufs};
use common::*;

use std::sync::LazyLock;

// ── CPU capabilities ────────────────────────────────────────────────────

/// CPU SIMD capabilities + runtime library availability, detected once at startup.
struct Caps {
    avx512: bool,
    avx512_bf16: bool,
    /// Intel AMX available via oneDNN (prefill attention only, M>1).
    amx: bool,
}

static CAPS: LazyLock<Caps> = LazyLock::new(|| {
    let mut caps = Caps { avx512: false, avx512_bf16: false, amx: false };
    #[cfg(target_arch = "x86_64")]
    {
        caps.avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw");
        caps.avx512_bf16 = is_x86_feature_detected!("avx512bf16");
    }
    {
        caps.amx = crate::ops::onednn::brgemm_available();
    }
    caps
});

// ── Prefill attention ──────────────────────────────────────────────────

/// Prefill attention with causal mask for batched variable-length sequences.
///
/// Layout (all `u16` = BF16 bit patterns, contiguous):
///   q: `[total_tokens, num_heads, head_dim]`
///   k: `[total_tokens, num_kv_heads, head_dim]`
///   v: `[total_tokens, num_kv_heads, head_dim]`
///   output: `[total_tokens, num_heads, head_dim]` (pre-allocated)
///   seq_lens: per-request sequence lengths (sum = total_tokens)
///
/// Implements: O[t,h] = softmax(Q[t,h] @ K[0..t+1, h_kv]^T * sm_scale) @ V[0..t+1, h_kv]
/// with GQA (grouped query attention): h_kv = h / (num_heads / num_kv_heads).
pub fn prefill_attention_bf16(
    output: &mut [u16],
    q: &[u16],
    k: &[u16],
    v: &[u16],
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) {
    let total_tokens: usize = seq_lens.iter().sum();
    debug_assert_eq!(q.len(), total_tokens * num_heads * head_dim);
    debug_assert_eq!(k.len(), total_tokens * num_kv_heads * head_dim);
    debug_assert_eq!(v.len(), total_tokens * num_kv_heads * head_dim);
    debug_assert_eq!(output.len(), total_tokens * num_heads * head_dim);

    let max_slen = seq_lens.iter().max().copied().unwrap_or(0);

    // Fast path for small sequences (seq_len ≤ 16):
    // Simple per-head loop with no Vec allocations, dispatched via spinning GemmPool.
    // Above 16, the tiled kernel's dpbf16ps vectorization outweighs its Vec alloc overhead.
    if max_slen <= 16 {
        small::prefill_attention_bf16_small(
            output, q, k, v, seq_lens,
            num_heads, num_kv_heads, head_dim, sm_scale,
        );
        return;
    }

    prefill_attention_bf16_tiled(
        output, q, k, v, seq_lens,
        num_heads, num_kv_heads, head_dim, sm_scale,
    );
}

/// AVX-512 tiled attention (FlashAttention-style online softmax).
/// Dispatched via spinning GemmPool (same as decode/small paths) to avoid
/// rayon futex parking overhead between GEMM and attention calls.
fn prefill_attention_bf16_tiled(
    output: &mut [u16],
    q: &[u16],
    k: &[u16],
    v: &[u16],
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;

    // Build per-request offsets
    let mut offsets = Vec::with_capacity(seq_lens.len() + 1);
    offsets.push(0usize);
    for &slen in seq_lens {
        offsets.push(offsets.last().unwrap() + slen);
    }

    // Parallelize over (req, head, m_block) — same as SGLang.
    // This gives N× more work items than per-head parallelism.
    let max_slen = seq_lens.iter().max().copied().unwrap_or(0);
    let (block_m, _block_n) = select_blocks(max_slen);
    let max_mb = (max_slen + block_m - 1) / block_m;
    let total_work = seq_lens.len() * num_heads * max_mb;

    #[repr(C)]
    struct AttnTiledCtx {
        out_ptr: usize,
        q_ptr: usize,
        k_ptr: usize,
        v_ptr: usize,
        offsets_ptr: usize,
        seq_lens_ptr: usize,
        num_reqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        gqa_ratio: usize,
        sm_scale: f32,
        max_mb: usize,
    }

    let ctx = AttnTiledCtx {
        out_ptr: output.as_mut_ptr() as usize,
        q_ptr: q.as_ptr() as usize,
        k_ptr: k.as_ptr() as usize,
        v_ptr: v.as_ptr() as usize,
        offsets_ptr: offsets.as_ptr() as usize,
        seq_lens_ptr: seq_lens.as_ptr() as usize,
        num_reqs: seq_lens.len(),
        num_heads,
        num_kv_heads,
        head_dim,
        gqa_ratio,
        sm_scale,
        max_mb,
    };

    unsafe fn attn_tiled_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const AttnTiledCtx);
            let total_work = ctx.num_reqs * ctx.num_heads * ctx.max_mb;
            let items_per_thread = (total_work + n_threads - 1) / n_threads;
            let start = tid * items_per_thread;
            let end = (start + items_per_thread).min(total_work);

            let offsets = std::slice::from_raw_parts(ctx.offsets_ptr as *const usize, ctx.num_reqs + 1);
            let seq_lens = std::slice::from_raw_parts(ctx.seq_lens_ptr as *const usize, ctx.num_reqs);
            let out_total = {
                let last = *offsets.last().unwrap_or(&0);
                last * ctx.num_heads * ctx.head_dim
            };
            let output = std::slice::from_raw_parts_mut(ctx.out_ptr as *mut u16, out_total);
            let q_total = out_total;
            let q = std::slice::from_raw_parts(ctx.q_ptr as *const u16, q_total);
            let k_total = offsets.last().unwrap_or(&0) * ctx.num_kv_heads * ctx.head_dim;
            let k = std::slice::from_raw_parts(ctx.k_ptr as *const u16, k_total);
            let v = std::slice::from_raw_parts(ctx.v_ptr as *const u16, k_total);

            for work_id in start..end {
                // Decompose: work_id = req_idx * (num_heads * max_mb) + head_idx * max_mb + mb
                let req_idx = work_id / (ctx.num_heads * ctx.max_mb);
                let rem = work_id % (ctx.num_heads * ctx.max_mb);
                let head_idx = rem / ctx.max_mb;
                let mb = rem % ctx.max_mb;
                prefill_attention_one_head(
                    output, q, k, v, offsets, seq_lens, req_idx, head_idx,
                    ctx.num_heads, ctx.num_kv_heads, ctx.head_dim, ctx.gqa_ratio, ctx.sm_scale,
                    mb,
                );
            }
        }
    }

    let pool = super::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(total_work).max(1);
    unsafe {
        pool.dispatch(
            attn_tiled_work,
            &ctx as *const AttnTiledCtx as *const u8,
            n_threads,
        );
    }
}



/// Tiled attention for one (request, head) pair using FlashAttention-style online softmax.
///
/// Brgemm vs default path selected via `CAPS.brgemm`. Within the default path,
/// dpbf16ps QK optimization selected via `CAPS.avx512_bf16`.
#[allow(clippy::too_many_arguments)]
fn prefill_attention_one_head(
    output: &mut [u16],
    q: &[u16],
    k: &[u16],
    v: &[u16],
    offsets: &[usize],
    seq_lens: &[usize],
    req_idx: usize,
    head_idx: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    sm_scale: f32,
    m_block_idx: usize,
) {
    let use_amx = CAPS.amx;
    let use_avx512_bf16 = CAPS.avx512_bf16;
    let use_avx512 = CAPS.avx512 || use_amx; // AMX implies AVX-512

    let req_start = offsets[req_idx];
    let slen = seq_lens[req_idx];
    let kv_head = head_idx / gqa_ratio;

    // SAFETY: GemmPool threads are single-threaded; no re-entrancy.
    let bufs = unsafe { get_attn_bufs() };

    let (block_m, block_n) = select_blocks(slen);

    // Single M-block: compute start and size
    let m = m_block_idx * block_m;
    if m >= slen {
        return;
    }
    let m_size = (slen - m).min(block_m);

    let num_keys_for_block = (m + m_size).min(slen);
    let kv_stride = num_kv_heads * head_dim;
    let k_base = req_start * kv_stride + kv_head * head_dim;
    let v_base = req_start * kv_stride + kv_head * head_dim;

    // ── KV gather (backend-dependent) ───────────────────────────────────
    // Brgemm: no gather (strided access). Others: gather BF16 KV.
    // F32: also pre-convert K to F32 for dot product (V stays BF16).
    if !use_amx {
        let kv_len = num_keys_for_block * head_dim;
        ensure_len_u16(&mut bufs.k_buf, kv_len);
        ensure_len_u16(&mut bufs.v_buf, kv_len);
        for j in 0..num_keys_for_block {
            let kv_off = (req_start + j) * kv_stride + kv_head * head_dim;
            bufs.k_buf[j * head_dim..(j + 1) * head_dim].copy_from_slice(&k[kv_off..kv_off + head_dim]);
            bufs.v_buf[j * head_dim..(j + 1) * head_dim].copy_from_slice(&v[kv_off..kv_off + head_dim]);
        }
        if !use_avx512_bf16 {
            // F32 dot path needs K pre-converted to F32. dpbf16ps reads BF16 directly.
            ensure_len_f32(&mut bufs.k_f32, kv_len);
            convert_bf16_to_f32(&mut bufs.k_f32[..kv_len], &bufs.k_buf[..kv_len], use_avx512);
        }
    }

    // Ensure scratch buffers
    ensure_len_f32(&mut bufs.s_i, block_m * block_n);
    ensure_len_f32(&mut bufs.v_prime, block_m * head_dim);
    ensure_len_f32(&mut bufs.s_prime, block_m);
    ensure_len_f32(&mut bufs.m_prime, block_m);

    let q_stride = num_heads * head_dim;
    let q_base = req_start * q_stride + head_idx * head_dim;

    // ── Q F32 conversion (only when dpbf16 unavailable) ────────────────
    if !use_avx512_bf16 && !use_amx {
        ensure_len_f32(&mut bufs.q_f32_block, block_m * head_dim);
    }

    // ── Process single M-block ──────────────────────────────────────────
    bufs.v_prime[..m_size * head_dim].fill(0.0);
    bufs.s_prime[..m_size].fill(0.0);
    bufs.m_prime[..m_size].fill(f32::NEG_INFINITY);

    // Q gather (convert to F32 only when dpbf16 unavailable)
    if !use_avx512_bf16 && !use_amx {
        for i in 0..m_size {
            let q_off = q_base + (m + i) * q_stride;
            convert_bf16_to_f32(
                &mut bufs.q_f32_block[i * head_dim..(i + 1) * head_dim],
                &q[q_off..q_off + head_dim],
                use_avx512,
            );
        }
    }

    let num_keys = (m + m_size).min(slen);

    let mut n = 0;
    while n < num_keys {
        let n_size = (num_keys - n).min(block_n);

        // ── QK GEMM (backend-specific) ──────────────────────────────────
        if use_amx {
            // AMX: strided Q/K access, oneDNN brgemm GEMM
            {
                unsafe {
                    crate::ops::onednn::ffi::brgemm_qk_gemm(
                        q.as_ptr().add(q_base + m * q_stride),
                        k.as_ptr().add(k_base + n * kv_stride),
                        bufs.s_i.as_mut_ptr(),
                        m_size as i64, n_size as i64, head_dim as i64,
                        q_stride as i64, kv_stride as i64, block_n as i64, sm_scale,
                    );
                }
                // brgemm outputs unscaled Q@K^T — apply sm_scale uniformly.
                for i in 0..m_size {
                    for j in 0..n_size {
                        bufs.s_i[i * block_n + j] *= sm_scale;
                    }
                }
            }
        } else if use_avx512_bf16 {
            // AVX-512 BF16: native dpbf16ps dot (2x throughput, no F32 conversion)
            #[cfg(target_arch = "x86_64")]
            unsafe {
                micro_gemm_qk_bf16(
                    q.as_ptr().add(q_base + m * q_stride),
                    bufs.k_buf[(n * head_dim)..].as_ptr(),
                    bufs.s_i.as_mut_ptr(),
                    m_size, n_size, head_dim,
                    q_stride, head_dim, block_n,
                    sm_scale,
                );
            }
        } else {
            // F32 dot product (AVX-512 SIMD or scalar)
            for i in 0..m_size {
                let q_row = &bufs.q_f32_block[i * head_dim..(i + 1) * head_dim];
                for j in 0..n_size {
                    let k_row = &bufs.k_f32[(n + j) * head_dim..(n + j + 1) * head_dim];
                    bufs.s_i[i * block_n + j] =
                        dot_f32_f32(q_row, k_row, head_dim, use_avx512) * sm_scale;
                }
            }
        }

        // ── Causal mask (shared) ────────────────────────────────────────
        if n + n_size > m {
            for i in 0..m_size {
                let last_valid_col = (m + i).saturating_sub(n);
                if last_valid_col + 1 < n_size {
                    for j in (last_valid_col + 1)..n_size {
                        bufs.s_i[i * block_n + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // ── Online softmax (shared: avx512 vs scalar dispatch) ──────────
        // All backends output scaled scores (QK * sm_scale), so softmax uses 1.0.
        for i in 0..m_size {
            let (m_i, block_sum, rescale) = if use_avx512 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    online_softmax_avx512(
                        bufs.s_i.as_mut_ptr().add(i * block_n),
                        n_size,
                        bufs.m_prime[i],
                        1.0,
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            } else {
                softmax_scalar(
                    &mut bufs.s_i[i * block_n..i * block_n + n_size],
                    n_size, bufs.m_prime[i], 1.0,
                )
            };

            bufs.s_prime[i] *= rescale;
            scale_f32(&mut bufs.v_prime[i * head_dim..(i + 1) * head_dim], rescale, use_avx512);
            bufs.s_prime[i] += block_sum;
            bufs.m_prime[i] = m_i;
        }

        // ── V accumulation ───────────────────────────────────────────────
        if use_amx {
            unsafe {
                crate::ops::onednn::ffi::brgemm_score_v_accum(
                    bufs.s_i.as_ptr(),
                    v.as_ptr().add(v_base + n * kv_stride),
                    bufs.v_prime.as_mut_ptr(),
                    m_size as i64, n_size as i64, head_dim as i64,
                    block_n as i64, kv_stride as i64,
                );
            }
        } else if use_avx512 {
            #[cfg(target_arch = "x86_64")]
            for i in 0..m_size {
                unsafe {
                    weight_v_accum_bf16_avx512(
                        bufs.v_prime[(i * head_dim)..].as_mut_ptr(),
                        bufs.v_buf[(n * head_dim)..].as_ptr(),
                        bufs.s_i[(i * block_n)..].as_ptr(),
                        n_size, head_dim, head_dim,
                    );
                }
            }
        } else {
            for i in 0..m_size {
                for j in 0..n_size {
                    let w = bufs.s_i[i * block_n + j];
                    if w > 0.0 {
                        let v_row = &bufs.v_buf[(n + j) * head_dim..(n + j + 1) * head_dim];
                        let acc = &mut bufs.v_prime[i * head_dim..(i + 1) * head_dim];
                        for d in 0..head_dim {
                            acc[d] += w * bf16_to_f32(v_row[d]);
                        }
                    }
                }
            }
        }

        n += n_size;
    }

    // ── Post-loop cleanup (brgemm only) ─────────────────────────────────
    if use_amx {
        unsafe { crate::ops::onednn::ffi::brgemm_attn_release(); }
    }

    // ── Normalize and write output (shared: avx512 vs scalar) ───────────
    for i in 0..m_size {
        let inv_sum = if bufs.s_prime[i] > 0.0 { 1.0 / bufs.s_prime[i] } else { 0.0 };
        let o_off = (req_start + m + i) * num_heads * head_dim + head_idx * head_dim;
        if use_avx512 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                normalize_output_avx512(
                    output[o_off..].as_mut_ptr(),
                    bufs.v_prime[i * head_dim..].as_ptr(),
                    inv_sum, head_dim,
                );
            }
        } else {
            let v_row = &bufs.v_prime[i * head_dim..(i + 1) * head_dim];
            for d in 0..head_dim {
                output[o_off + d] = f32_to_bf16(v_row[d] * inv_sum);
            }
        }
    }

}

// ── Decode (single-token) attention ─────────────────────────────────────

/// Decode attention: single Q token per request against full KV cache.
///
/// Layout:
///   q: `[num_seqs, num_heads, head_dim]` (one token per request)
///   k_cache: `[max_total_tokens, num_kv_heads, head_dim]` (flat pool buffer)
///   v_cache: `[max_total_tokens, num_kv_heads, head_dim]` (flat pool buffer)
///   output: `[num_seqs, num_heads, head_dim]` (pre-allocated)
///   req_to_token: `[max_num_reqs, max_context_len]` — maps (req, pos) -> slot in cache
///   seq_lens: `[num_seqs]` — number of KV tokens per request (including new token)
///
/// The new K/V token should already be written to the cache at the appropriate slot
/// before calling this function.
pub fn decode_attention_bf16(
    output: &mut [u16],
    q: &[u16],
    k_cache: &[u16],
    v_cache: &[u16],
    req_to_token: &[i32],
    seq_lens: &[i64],
    num_seqs: usize,
    max_context_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;
    let total_work = num_seqs * num_heads;

    #[repr(C)]
    struct DecodeCtx {
        out_ptr: usize,
        out_len: usize,
        q_ptr: usize,
        q_len: usize,
        k_cache_ptr: usize,
        k_cache_len: usize,
        v_cache_ptr: usize,
        v_cache_len: usize,
        req_to_token_ptr: usize,
        req_to_token_len: usize,
        seq_lens_ptr: usize,
        seq_lens_len: usize,
        num_seqs: usize,
        max_context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        gqa_ratio: usize,
        sm_scale: f32,
    }

    let ctx = DecodeCtx {
        out_ptr: output.as_mut_ptr() as usize,
        out_len: output.len(),
        q_ptr: q.as_ptr() as usize,
        q_len: q.len(),
        k_cache_ptr: k_cache.as_ptr() as usize,
        k_cache_len: k_cache.len(),
        v_cache_ptr: v_cache.as_ptr() as usize,
        v_cache_len: v_cache.len(),
        req_to_token_ptr: req_to_token.as_ptr() as usize,
        req_to_token_len: req_to_token.len(),
        seq_lens_ptr: seq_lens.as_ptr() as usize,
        seq_lens_len: seq_lens.len(),
        num_seqs,
        max_context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        gqa_ratio,
        sm_scale,
    };

    unsafe fn decode_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const DecodeCtx);
            let total_work = ctx.num_seqs * ctx.num_heads;
            let items_per_thread = (total_work + n_threads - 1) / n_threads;
            let start = tid * items_per_thread;
            let end = (start + items_per_thread).min(total_work);

            let output = std::slice::from_raw_parts_mut(ctx.out_ptr as *mut u16, ctx.out_len);
            let q = std::slice::from_raw_parts(ctx.q_ptr as *const u16, ctx.q_len);
            let k_cache = std::slice::from_raw_parts(ctx.k_cache_ptr as *const u16, ctx.k_cache_len);
            let v_cache = std::slice::from_raw_parts(ctx.v_cache_ptr as *const u16, ctx.v_cache_len);
            let req_to_token = std::slice::from_raw_parts(ctx.req_to_token_ptr as *const i32, ctx.req_to_token_len);
            let seq_lens = std::slice::from_raw_parts(ctx.seq_lens_ptr as *const i64, ctx.seq_lens_len);

            for work_id in start..end {
                let req_idx = work_id / ctx.num_heads;
                let head_idx = work_id % ctx.num_heads;
                decode_attention_one_head(
                    output, q, k_cache, v_cache, req_to_token, seq_lens,
                    req_idx, head_idx, ctx.max_context_len,
                    ctx.num_heads, ctx.num_kv_heads, ctx.head_dim, ctx.gqa_ratio, ctx.sm_scale,
                );
            }
        }
    }

    let pool = super::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(total_work).max(1);
    unsafe {
        pool.dispatch(
            decode_work,
            &ctx as *const DecodeCtx as *const u8,
            n_threads,
        );
    }
}

/// Decode attention for one (request, head) pair.
/// No backend parameter — decode always uses Default (avx512_bf16 dispatch internal).
#[allow(clippy::too_many_arguments)]
fn decode_attention_one_head(
    output: &mut [u16],
    q: &[u16],
    k_cache: &[u16],
    v_cache: &[u16],
    req_to_token: &[i32],
    seq_lens: &[i64],
    req_idx: usize,
    head_idx: usize,
    max_context_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    sm_scale: f32,
) {
    let use_avx512 = CAPS.avx512;
    let use_avx512_bf16 = CAPS.avx512_bf16;

    let kv_head = head_idx / gqa_ratio;
    let slen = seq_lens[req_idx] as usize;
    if slen == 0 {
        return;
    }

    let q_off = req_idx * num_heads * head_dim + head_idx * head_dim;
    let q_row = &q[q_off..q_off + head_dim];

    let block_n: usize = if slen <= 1024 { slen } else { 512 };

    let mut k_block = vec![0u16; block_n * head_dim];
    let mut v_block = vec![0u16; block_n * head_dim];
    let mut scores = vec![0.0f32; block_n];

    let mut out_f32 = vec![0.0f32; head_dim];
    let mut m_prime = f32::NEG_INFINITY;
    let mut s_prime = 0.0f32;

    // Pre-convert Q to F32 when avx512_bf16 is unavailable (F32 dot path)
    let q_f32 = if !use_avx512_bf16 {
        let mut qf = vec![0.0f32; head_dim];
        convert_bf16_to_f32(&mut qf, q_row, use_avx512);
        qf
    } else {
        vec![]
    };

    let req_token_base = req_idx * max_context_len;

    let mut n = 0;
    while n < slen {
        let n_size = (slen - n).min(block_n);

        // Lazy gather KV block
        for j in 0..n_size {
            let slot = req_to_token[req_token_base + n + j] as usize;
            let kv_off = slot * num_kv_heads * head_dim + kv_head * head_dim;
            k_block[j * head_dim..(j + 1) * head_dim]
                .copy_from_slice(&k_cache[kv_off..kv_off + head_dim]);
            v_block[j * head_dim..(j + 1) * head_dim]
                .copy_from_slice(&v_cache[kv_off..kv_off + head_dim]);
        }

        // ── QK scores ────────────────────────────────────────────────────
        #[cfg(target_arch = "x86_64")]
        if use_avx512_bf16 {
            for j in 0..n_size {
                scores[j] = dot_bf16_bf16_native(
                    q_row, &k_block[j * head_dim..], head_dim,
                ) * sm_scale;
            }
        } else {
            for j in 0..n_size {
                let k_row = &k_block[j * head_dim..(j + 1) * head_dim];
                let mut kf = vec![0.0f32; head_dim];
                convert_bf16_to_f32(&mut kf, k_row, use_avx512);
                scores[j] = dot_f32_f32(&q_f32, &kf, head_dim, use_avx512) * sm_scale;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for j in 0..n_size {
                let k_row = &k_block[j * head_dim..(j + 1) * head_dim];
                let mut kf = vec![0.0f32; head_dim];
                convert_bf16_to_f32(&mut kf, k_row, use_avx512);
                scores[j] = dot_f32_f32(&q_f32, &kf, head_dim, use_avx512) * sm_scale;
            }
        }

        // ── Online softmax (shared) ─────────────────────────────────────
        let (m_i, block_sum, rescale) = if use_avx512 {
            #[cfg(target_arch = "x86_64")]
            unsafe { online_softmax_avx512(scores.as_mut_ptr(), n_size, m_prime, 1.0) }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!()
        } else {
            softmax_scalar(&mut scores, n_size, m_prime, 1.0)
        };

        if rescale != 1.0 {
            s_prime *= rescale;
            scale_f32(&mut out_f32, rescale, use_avx512);
        }
        s_prime += block_sum;
        m_prime = m_i;

        // ── V accumulation ──────────────────────────────────────────────
        if use_avx512 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                weight_v_accum_bf16_avx512(
                    out_f32.as_mut_ptr(),
                    v_block.as_ptr(),
                    scores.as_ptr(),
                    n_size, head_dim, head_dim,
                );
            }
        } else {
            for j in 0..n_size {
                let w = scores[j];
                if w > 0.0 {
                    for d in 0..head_dim {
                        out_f32[d] += w * bf16_to_f32(v_block[j * head_dim + d]);
                    }
                }
            }
        }

        n += block_n;
    }

    // Normalize and write
    let inv_sum = if s_prime > 0.0 { 1.0 / s_prime } else { 0.0 };
    let o_off = req_idx * num_heads * head_dim + head_idx * head_dim;
    for d in 0..head_dim {
        output[o_off + d] = f32_to_bf16(out_f32[d] * inv_sum);
    }
}

// ── F32 Decode Attention ─────────────────────────────────────────────────

/// Decode attention for F32 tensors: single Q token against KV cache.
///
/// Layout: q `[num_seqs, num_heads, head_dim]`, k/v_cache `[total_slots, num_kv_heads, head_dim]`.
/// `req_to_token[req * max_context_len + i]` maps to the cache slot for position i.
/// `seq_lens[req]` is the KV length for request `req`.
/// Output: `[num_seqs, num_heads, head_dim]` F32.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32(
    output: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    req_to_token: &[i32],
    seq_lens: &[i64],
    num_seqs: usize,
    max_context_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) {
    use rayon::prelude::*;

    let gqa_ratio = num_heads / num_kv_heads;
    let total_work = num_seqs * num_heads;

    let out_ptr = output.as_mut_ptr() as usize;
    let out_len = output.len();

    (0..total_work).into_par_iter().for_each(|work_id| {
        let req_idx = work_id / num_heads;
        let head_idx = work_id % num_heads;
        let output = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        decode_attention_one_head_f32(
            output, q, k_cache, v_cache, req_to_token, seq_lens,
            req_idx, head_idx, max_context_len,
            num_heads, num_kv_heads, head_dim, gqa_ratio, sm_scale,
        );
    });
}

/// Decode attention for one (request, head) pair — F32 variant.
/// Uses online softmax with block-based processing and AVX-512 vectorized dot/FMA.
#[allow(clippy::too_many_arguments)]
fn decode_attention_one_head_f32(
    output: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    req_to_token: &[i32],
    seq_lens: &[i64],
    req_idx: usize,
    head_idx: usize,
    max_context_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    sm_scale: f32,
) {
    let use_avx512 = CAPS.avx512;

    let kv_head = head_idx / gqa_ratio;
    let slen = seq_lens[req_idx] as usize;
    if slen == 0 {
        return;
    }

    let q_off = req_idx * num_heads * head_dim + head_idx * head_dim;
    let q_row = &q[q_off..q_off + head_dim];

    let block_n: usize = if slen <= 1024 { slen } else { 512 };

    let mut k_block = vec![0.0f32; block_n * head_dim];
    let mut v_block = vec![0.0f32; block_n * head_dim];
    let mut scores = vec![0.0f32; block_n];

    let mut out_f32 = vec![0.0f32; head_dim];
    let mut m_prime = f32::NEG_INFINITY;
    let mut s_prime = 0.0f32;

    let req_token_base = req_idx * max_context_len;

    let mut n = 0;
    while n < slen {
        let n_size = (slen - n).min(block_n);

        for j in 0..n_size {
            let slot = req_to_token[req_token_base + n + j] as usize;
            let kv_off = slot * num_kv_heads * head_dim + kv_head * head_dim;
            k_block[j * head_dim..(j + 1) * head_dim]
                .copy_from_slice(&k_cache[kv_off..kv_off + head_dim]);
            v_block[j * head_dim..(j + 1) * head_dim]
                .copy_from_slice(&v_cache[kv_off..kv_off + head_dim]);
        }

        // QK scores — already F32, use dot_f32_f32 directly
        for j in 0..n_size {
            let k_row = &k_block[j * head_dim..(j + 1) * head_dim];
            scores[j] = dot_f32_f32(q_row, k_row, head_dim, use_avx512) * sm_scale;
        }

        // Online softmax
        let (m_i, block_sum, rescale) = if use_avx512 {
            #[cfg(target_arch = "x86_64")]
            unsafe { online_softmax_avx512(scores.as_mut_ptr(), n_size, m_prime, 1.0) }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!()
        } else {
            softmax_scalar(&mut scores, n_size, m_prime, 1.0)
        };

        if rescale != 1.0 {
            s_prime *= rescale;
            scale_f32(&mut out_f32, rescale, use_avx512);
        }
        s_prime += block_sum;
        m_prime = m_i;

        // V accumulation — F32 FMA
        for j in 0..n_size {
            let w = scores[j];
            if w > 0.0 {
                let v_row = &v_block[j * head_dim..(j + 1) * head_dim];
                fma_f32_f32(&mut out_f32, v_row, w, use_avx512);
            }
        }

        n += block_n;
    }

    let inv_sum = if s_prime > 0.0 { 1.0 / s_prime } else { 0.0 };
    let o_off = req_idx * num_heads * head_dim + head_idx * head_dim;
    for d in 0..head_dim {
        output[o_off + d] = out_f32[d] * inv_sum;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
