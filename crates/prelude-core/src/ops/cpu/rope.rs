//! Pure Rust AVX-512 Rotary Position Embedding (RoPE) for BF16 tensors.
//!
//! GPT-NeoX split-half rotation (used by Qwen3, LLaMA, etc.):
//!   q'[i]           = q[i]          * cos[i] - q[embed_dim+i] * sin[i]
//!   q'[embed_dim+i] = q[embed_dim+i]* cos[i] + q[i]          * sin[i]
//!
//! cos_sin_cache layout: `[max_seq_len, rotary_dim]` where first half is cos,
//! second half is sin: `[cos[0..embed_dim] | sin[0..embed_dim]]`.

use rayon::prelude::*;

/// Minimum elements per thread to justify parallelization overhead.
/// RoPE per-element work is very light (~0.26ns/elem AVX-512). On 64+ core
/// systems, pool-wide overhead is ~50-100µs, so we need large total
/// work before parallelizing. 32K elems/thread with pool-wide floor.
const MIN_ELEMS_PER_THREAD: usize = 32768;

/// Apply NeoX split-half RoPE in-place to Q and K tensors (4D THD layout).
///
/// - `q`: `[batch_size * seq_len * num_heads * head_dim]` as `u16` (BF16)
/// - `k`: `[batch_size * seq_len * num_kv_heads * head_dim]` as `u16` (BF16)
/// - `cos_sin_cache`: `[max_seq_len * rotary_dim]` as `u16` (BF16), packed `[cos|sin]`
/// - `positions`: `[batch_size * seq_len]` as `i64`
pub fn rope_neox_bf16(
    q: &mut [u16],
    k: &mut [u16],
    cos_sin_cache: &[u16],
    positions: &[i64],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) {
    let embed_dim = rotary_dim / 2;
    let total_tokens = batch_size * seq_len;
    let q_token_stride = num_heads * head_dim;
    let k_token_stride = num_kv_heads * head_dim;
    let total_elems = total_tokens * (num_heads + num_kv_heads) * head_dim;

    if !super::should_parallelize(total_elems, total_tokens, MIN_ELEMS_PER_THREAD) {
        for t in 0..total_tokens {
            let pos = positions[t];
            if pos < 0 { continue; }
            let cache_off = pos as usize * rotary_dim;
            let q_base = t * q_token_stride;
            let k_base = t * k_token_stride;
            for h in 0..num_heads {
                let off = q_base + h * head_dim;
                rope_neox_row(&mut q[off..off + head_dim], cos_sin_cache, cache_off, embed_dim);
            }
            for h in 0..num_kv_heads {
                let off = k_base + h * head_dim;
                rope_neox_row(&mut k[off..off + head_dim], cos_sin_cache, cache_off, embed_dim);
            }
        }
        return;
    }

    // Use GemmPool (spinning threads) instead of rayon to avoid contention
    #[repr(C)]
    struct RopeCtx {
        q_ptr: usize, k_ptr: usize, cache_ptr: usize, pos_ptr: usize,
        total_tokens: usize, q_stride: usize, k_stride: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        embed_dim: usize, rotary_dim: usize,
    }
    unsafe fn work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const RopeCtx);
            let rows_per = (c.total_tokens + n_threads - 1) / n_threads;
            let start = tid * rows_per;
            let end = (start + rows_per).min(c.total_tokens);
            if start >= end { return; }
            let positions = std::slice::from_raw_parts(c.pos_ptr as *const i64, c.total_tokens);
            let q = c.q_ptr as *mut u16;
            let k = c.k_ptr as *mut u16;
            for t in start..end {
                let pos = positions[t];
                if pos < 0 { continue; }
                let cache_off = pos as usize * c.rotary_dim;
                let cache = std::slice::from_raw_parts(
                    c.cache_ptr as *const u16, (pos as usize + 1) * c.rotary_dim);
                let q_base = t * c.q_stride;
                let k_base = t * c.k_stride;
                for h in 0..c.num_heads {
                    let off = q_base + h * c.head_dim;
                    let row = std::slice::from_raw_parts_mut(q.add(off), c.head_dim);
                    rope_neox_row(row, cache, cache_off, c.embed_dim);
                }
                for h in 0..c.num_kv_heads {
                    let off = k_base + h * c.head_dim;
                    let row = std::slice::from_raw_parts_mut(k.add(off), c.head_dim);
                    rope_neox_row(row, cache, cache_off, c.embed_dim);
                }
            }
        }
    }
    let ctx = RopeCtx {
        q_ptr: q.as_mut_ptr() as usize, k_ptr: k.as_mut_ptr() as usize,
        cache_ptr: cos_sin_cache.as_ptr() as usize, pos_ptr: positions.as_ptr() as usize,
        total_tokens, q_stride: q_token_stride, k_stride: k_token_stride,
        num_heads, num_kv_heads, head_dim, embed_dim, rotary_dim,
    };
    let pool = super::gemm_pool::gemm_pool();
    let n = pool.num_threads().min(total_tokens);
    unsafe { pool.dispatch(work, &ctx as *const RopeCtx as *const u8, n); }
}

// ── Per-row rotation ────────────────────────────────────────────────────

/// Apply NeoX split-half RoPE to a single head row `[head_dim]` in-place.
/// `cache_off` points to the start of this position's cos/sin in the cache.
/// `embed_dim` = rotary_dim / 2 = number of rotation pairs.
///
/// Split-half: pairs (row[i], row[embed_dim+i]) for i in 0..embed_dim.
pub(crate) fn rope_neox_row(row: &mut [u16], cos_sin_cache: &[u16], cache_off: usize, embed_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                return rope_neox_row_avx512(row, cos_sin_cache, cache_off, embed_dim);
            }
        }
    }
    rope_neox_row_scalar(row, cos_sin_cache, cache_off, embed_dim);
}

fn rope_neox_row_scalar(row: &mut [u16], cache: &[u16], cache_off: usize, embed_dim: usize) {
    // cos at cache[cache_off..cache_off+embed_dim]
    // sin at cache[cache_off+embed_dim..cache_off+2*embed_dim]
    // Split-half: x = row[i], y = row[embed_dim+i]
    for i in 0..embed_dim {
        let cos = bf16_to_f32(cache[cache_off + i]);
        let sin = bf16_to_f32(cache[cache_off + embed_dim + i]);
        let x = bf16_to_f32(row[i]);
        let y = bf16_to_f32(row[embed_dim + i]);
        row[i] = f32_to_bf16(x * cos - y * sin);
        row[embed_dim + i] = f32_to_bf16(y * cos + x * sin);
    }
}

/// AVX-512 NeoX split-half RoPE. Processes 16 pairs per iteration.
///
/// For each chunk of 16 indices i:
///   x  = row[i..i+16]           (first half)
///   y  = row[embed_dim+i..+16]  (second half)
///   cos = cache[cache_off+i..+16]
///   sin = cache[cache_off+embed_dim+i..+16]
///   row[i..]          = x * cos - y * sin
///   row[embed_dim+i..] = y * cos + x * sin
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn rope_neox_row_avx512(row: &mut [u16], cache: &[u16], cache_off: usize, embed_dim: usize) {
    use core::arch::x86_64::*;

    let cos_ptr = unsafe { cache.as_ptr().add(cache_off) };
    let sin_ptr = unsafe { cache.as_ptr().add(cache_off + embed_dim) };
    let row_ptr = row.as_mut_ptr();

    let chunks = embed_dim / 16;

    for i in 0..chunks {
        let off = i * 16;
        // Load first-half and second-half elements
        let x = bf16x16_load_as_f32(unsafe { row_ptr.add(off) as *const u16 });
        let y = bf16x16_load_as_f32(unsafe { row_ptr.add(embed_dim + off) as *const u16 });

        // Load cos and sin
        let cos = bf16x16_load_as_f32(unsafe { cos_ptr.add(off) });
        let sin = bf16x16_load_as_f32(unsafe { sin_ptr.add(off) });

        // out_x = x * cos - y * sin
        let out_x = _mm512_sub_ps(_mm512_mul_ps(x, cos), _mm512_mul_ps(y, sin));
        // out_y = y * cos + x * sin
        let out_y = _mm512_fmadd_ps(x, sin, _mm512_mul_ps(y, cos));

        f32x16_store_as_bf16(unsafe { row_ptr.add(off) }, out_x);
        f32x16_store_as_bf16(unsafe { row_ptr.add(embed_dim + off) }, out_y);
    }

    // Scalar remainder
    for i in (chunks * 16)..embed_dim {
        let cos = bf16_to_f32(cache[cache_off + i]);
        let sin = bf16_to_f32(cache[cache_off + embed_dim + i]);
        let x = bf16_to_f32(row[i]);
        let y = bf16_to_f32(row[embed_dim + i]);
        row[i] = f32_to_bf16(x * cos - y * sin);
        row[embed_dim + i] = f32_to_bf16(y * cos + x * sin);
    }
}

// ── BF16 <-> F32 helpers ────────────────────────────────────────────────

#[inline(always)]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
fn bf16x16_load_as_f32(ptr: *const u16) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    unsafe {
        let bf16_vals = _mm256_loadu_si256(ptr as *const __m256i);
        let extended = _mm512_cvtepu16_epi32(bf16_vals);
        let shifted = _mm512_slli_epi32(extended, 16);
        _mm512_castsi512_ps(shifted)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
fn f32x16_store_as_bf16(ptr: *mut u16, vals: core::arch::x86_64::__m512) {
    use core::arch::x86_64::*;
    unsafe {
        let bits = _mm512_castps_si512(vals);
        let lsb = _mm512_srli_epi32::<16>(bits);
        let lsb_masked = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
        let rounding_bias = _mm512_add_epi32(lsb_masked, _mm512_set1_epi32(0x7FFF));
        let rounded = _mm512_add_epi32(bits, rounding_bias);
        let shifted = _mm512_srli_epi32::<16>(rounded);
        let packed = _mm512_cvtepi32_epi16(shifted);
        _mm256_storeu_si256(ptr as *mut __m256i, packed);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::max_sglang_violation;

    fn make_bf16(v: f32) -> u16 {
        f32_to_bf16(v)
    }

    fn make_bf16_vec(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| f32_to_bf16(v)).collect()
    }

    /// Build a cos_sin_cache: cache[pos, 0..embed_dim] = cos, cache[pos, embed_dim..rotary_dim] = sin
    fn build_test_cache(max_pos: usize, rotary_dim: usize, base: f64) -> Vec<u16> {
        let embed_dim = rotary_dim / 2;
        let mut cache = vec![0u16; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..embed_dim {
                let freq = 1.0 / base.powf(2.0 * i as f64 / rotary_dim as f64);
                let theta = pos as f64 * freq;
                cache[pos * rotary_dim + i] = make_bf16(theta.cos() as f32);
                cache[pos * rotary_dim + embed_dim + i] = make_bf16(theta.sin() as f32);
            }
        }
        cache
    }

    /// Reference NeoX split-half RoPE in f32 for accuracy comparison.
    fn rope_ref_f32(x: &mut [f32], cos: &[f32], sin: &[f32], embed_dim: usize) {
        for i in 0..embed_dim {
            let x0 = x[i];
            let x1 = x[embed_dim + i];
            x[i] = x0 * cos[i] - x1 * sin[i];
            x[embed_dim + i] = x1 * cos[i] + x0 * sin[i];
        }
    }

    #[test]
    fn test_rope_scalar_basic() {
        let head_dim = 64;
        let rotary_dim = head_dim;
        let embed_dim = rotary_dim / 2;
        let cache = build_test_cache(16, rotary_dim, 10000.0);

        let q_f32: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1) - 1.6).collect();
        let k_f32: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.05) + 0.3).collect();

        let mut q = make_bf16_vec(&q_f32);
        let mut k = make_bf16_vec(&k_f32);
        let positions = vec![5i64];

        rope_neox_bf16(&mut q, &mut k, &cache, &positions, 1, 1, 1, 1, head_dim, rotary_dim);

        // Compare with scalar reference
        let mut q_ref: Vec<f32> = q_f32.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
        let cos: Vec<f32> = (0..embed_dim).map(|i| bf16_to_f32(cache[5 * rotary_dim + i])).collect();
        let sin: Vec<f32> = (0..embed_dim).map(|i| bf16_to_f32(cache[5 * rotary_dim + embed_dim + i])).collect();
        rope_ref_f32(&mut q_ref, &cos, &sin, embed_dim);

        let actual: Vec<f32> = q.iter().map(|&v| bf16_to_f32(v)).collect();
        let violation = max_sglang_violation(&actual, &q_ref);
        assert!(violation <= 0.0, "rope scalar worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)");
    }

    #[test]
    fn test_rope_dispatch_matches_scalar() {
        let head_dim = 128;
        let rotary_dim = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let batch_size = 2;
        let seq_len = 3;
        let n_q = batch_size * seq_len * num_heads * head_dim;
        let n_k = batch_size * seq_len * num_kv_heads * head_dim;

        let cache = build_test_cache(32, rotary_dim, 1000000.0);
        let q_f32: Vec<f32> = (0..n_q).map(|i| ((i as f32 * 0.007) - 0.5).sin()).collect();
        let k_f32: Vec<f32> = (0..n_k).map(|i| ((i as f32 * 0.013) + 0.2).cos()).collect();
        let positions: Vec<i64> = (0..(batch_size * seq_len) as i64).collect();

        // Scalar
        let mut q_scalar = make_bf16_vec(&q_f32);
        let mut k_scalar = make_bf16_vec(&k_f32);
        for b in 0..batch_size {
            for s in 0..seq_len {
                let pos = positions[b * seq_len + s] as usize;
                let cache_off = pos * rotary_dim;
                let embed_dim = rotary_dim / 2;
                for h in 0..num_heads {
                    let off = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim;
                    rope_neox_row_scalar(&mut q_scalar[off..off + head_dim], &cache, cache_off, embed_dim);
                }
                for h in 0..num_kv_heads {
                    let off = b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h * head_dim;
                    rope_neox_row_scalar(&mut k_scalar[off..off + head_dim], &cache, cache_off, embed_dim);
                }
            }
        }

        // Dispatched
        let mut q_dispatch = make_bf16_vec(&q_f32);
        let mut k_dispatch = make_bf16_vec(&k_f32);
        rope_neox_bf16(
            &mut q_dispatch, &mut k_dispatch, &cache, &positions,
            batch_size, seq_len, num_heads, num_kv_heads, head_dim, rotary_dim,
        );

        assert_eq!(q_scalar, q_dispatch, "Q dispatch should match scalar");
        assert_eq!(k_scalar, k_dispatch, "K dispatch should match scalar");
    }

    #[test]
    fn test_rope_partial_rotary() {
        // Test where rotary_dim < head_dim (only first rotary_dim elements rotated)
        let head_dim = 128;
        let rotary_dim = 64;
        let embed_dim = rotary_dim / 2;
        let cache = build_test_cache(16, rotary_dim, 10000.0);

        let q_f32: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.01).collect();
        let mut q = make_bf16_vec(&q_f32);
        let mut k = make_bf16_vec(&q_f32);
        let q_orig = q.clone();
        let positions = vec![3i64];

        rope_neox_bf16(&mut q, &mut k, &cache, &positions, 1, 1, 1, 1, head_dim, rotary_dim);

        // Split-half: first half [0..embed_dim] and second half [embed_dim..rotary_dim] should be modified
        assert_ne!(&q[..embed_dim], &q_orig[..embed_dim], "first half should change");
        assert_ne!(&q[embed_dim..rotary_dim], &q_orig[embed_dim..rotary_dim], "second half should change");
        // Elements beyond rotary_dim should be unchanged
        assert_eq!(&q[rotary_dim..], &q_orig[rotary_dim..], "non-rotary part should be unchanged");
    }

    /// Verify RoPE at realistic model dimensions against F32 scalar reference.
    fn verify_rope_config(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        label: &str,
    ) {
        let rotary_dim = head_dim;
        let embed_dim = rotary_dim / 2;
        let cache = build_test_cache(seq_len + 16, rotary_dim, 1000000.0);

        let n_q = seq_len * num_heads * head_dim;
        let n_k = seq_len * num_kv_heads * head_dim;
        let q_f32: Vec<f32> = (0..n_q).map(|i| ((i as f32 * 0.007) - 0.5).sin()).collect();
        let k_f32: Vec<f32> = (0..n_k).map(|i| ((i as f32 * 0.013) + 0.2).cos()).collect();
        let positions: Vec<i64> = (0..seq_len as i64).collect();

        // Scalar reference
        let mut q_ref = make_bf16_vec(&q_f32);
        let mut k_ref = make_bf16_vec(&k_f32);
        for s in 0..seq_len {
            let pos = s;
            let cache_off = pos * rotary_dim;
            for h in 0..num_heads {
                let off = s * num_heads * head_dim + h * head_dim;
                rope_neox_row_scalar(&mut q_ref[off..off + head_dim], &cache, cache_off, embed_dim);
            }
            for h in 0..num_kv_heads {
                let off = s * num_kv_heads * head_dim + h * head_dim;
                rope_neox_row_scalar(&mut k_ref[off..off + head_dim], &cache, cache_off, embed_dim);
            }
        }

        // Dispatched (AVX-512)
        let mut q_disp = make_bf16_vec(&q_f32);
        let mut k_disp = make_bf16_vec(&k_f32);
        rope_neox_bf16(
            &mut q_disp, &mut k_disp, &cache, &positions,
            1, seq_len, num_heads, num_kv_heads, head_dim, rotary_dim,
        );

        // Compare element-by-element (SGLang tolerance)
        let q_actual: Vec<f32> = q_disp.iter().map(|&v| bf16_to_f32(v)).collect();
        let q_expected: Vec<f32> = q_ref.iter().map(|&v| bf16_to_f32(v)).collect();
        let q_violation = max_sglang_violation(&q_actual, &q_expected);
        assert!(
            q_violation <= 0.0,
            "{label} Q worst violation={q_violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );

        let k_actual: Vec<f32> = k_disp.iter().map(|&v| bf16_to_f32(v)).collect();
        let k_expected: Vec<f32> = k_ref.iter().map(|&v| bf16_to_f32(v)).collect();
        let k_violation = max_sglang_violation(&k_actual, &k_expected);
        assert!(
            k_violation <= 0.0,
            "{label} K worst violation={k_violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }

    #[test]
    fn test_rope_realistic_configs() {
        // Qwen3-0.6B: H=16, KV=8, D=128
        verify_rope_config(16, 8, 128, 64, "0.6B slen=64");
        verify_rope_config(16, 8, 128, 256, "0.6B slen=256");
        // Qwen3-1.7B: H=16, KV=4, D=128
        verify_rope_config(16, 4, 128, 128, "1.7B slen=128");
        // Qwen3-32B: H=64, KV=8, D=128 (subset to keep test fast)
        verify_rope_config(64, 8, 128, 32, "32B slen=32");
        verify_rope_config(64, 8, 128, 128, "32B slen=128");
    }
}
