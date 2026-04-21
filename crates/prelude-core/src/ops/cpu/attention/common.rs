//! Shared primitives for attention kernels: BF16<->F32 conversion, softmax, SIMD helpers.

// ── Scalar BF16 <-> F32 helpers ─────────────────────────────────────────

#[inline(always)]
pub(super) fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[inline(always)]
pub(super) fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// AVX-512 fused normalize + F32->BF16 output: output[d] = bf16(v_prime[d] * inv_sum)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn normalize_output_avx512(
    output: *mut u16,
    v_prime: *const f32,
    inv_sum: f32,
    head_dim: usize,
) {
    use std::arch::x86_64::*;
    let scale = _mm512_set1_ps(inv_sum);
    let bias = _mm512_set1_epi32(0x7FFF_i32);
    let one = _mm512_set1_epi32(1);

    let mut d = 0usize;
    while d + 16 <= head_dim {
        let v = unsafe { _mm512_loadu_ps(v_prime.add(d)) };
        let scaled = _mm512_mul_ps(v, scale);
        // Round-to-nearest-even BF16 conversion (same as scalar f32_to_bf16)
        let bits = _mm512_castps_si512(scaled);
        let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(bits), one);
        let rounded = _mm512_add_epi32(_mm512_add_epi32(bits, bias), lsb);
        let bf16_32 = _mm512_srli_epi32::<16>(rounded);
        // Pack 16 × i32 → 16 × i16 (truncation, values already in [0,65535])
        let bf16_16 = _mm512_cvtepi32_epi16(bf16_32);
        unsafe { _mm256_storeu_si256(output.add(d) as *mut __m256i, bf16_16) };
        d += 16;
    }
    // Scalar remainder
    while d < head_dim {
        unsafe { *output.add(d) = f32_to_bf16(*v_prime.add(d) * inv_sum) };
        d += 1;
    }
}

// ── BF16→F32 bulk conversion ──────────────────────────────────────────

/// Convert BF16 slice to F32 slice with AVX-512 dispatch.
#[inline]
pub(super) fn convert_bf16_to_f32(dst: &mut [f32], src: &[u16], use_avx512: bool) {
    let len = dst.len();
    debug_assert!(src.len() >= len);
    #[cfg(target_arch = "x86_64")]
    if use_avx512 {
        unsafe {
            convert_bf16_to_f32_avx512(dst.as_mut_ptr(), src.as_ptr(), len);
        }
        return;
    }
    let _ = use_avx512;
    for i in 0..len {
        dst[i] = bf16_to_f32(src[i]);
    }
}

/// AVX-512: bulk BF16→F32 conversion (16 elements at a time).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn convert_bf16_to_f32_avx512(dst: *mut f32, src: *const u16, len: usize) {
    let chunks = len / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = bf16x16_load_as_f32(unsafe { src.add(off) });
        unsafe { core::arch::x86_64::_mm512_storeu_ps(dst.add(off), v) };
    }
    for i in (chunks * 16)..len {
        unsafe { *dst.add(i) = bf16_to_f32(*src.add(i)) };
    }
}

// ── F32-F32 vectorized helpers (used after BF16→F32 pre-conversion) ───

/// Dot product of two F32 slices with AVX-512 dispatch.
#[inline]
pub(super) fn dot_f32_f32(a: &[f32], b: &[f32], len: usize, use_avx512: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if use_avx512 {
        return unsafe { dot_f32_f32_avx512(a.as_ptr(), b.as_ptr(), len) };
    }
    let _ = use_avx512;
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// AVX-512: dot product of two F32 vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn dot_f32_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use core::arch::x86_64::*;
    let chunks = len / 16;
    let mut acc = _mm512_setzero_ps();
    for i in 0..chunks {
        let off = i * 16;
        let va = unsafe { _mm512_loadu_ps(a.add(off)) };
        let vb = unsafe { _mm512_loadu_ps(b.add(off)) };
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for j in (chunks * 16)..len {
        sum += unsafe { *a.add(j) * *b.add(j) };
    }
    sum
}

/// acc[d] += w * src[d] for d in 0..acc.len() (both F32)
#[inline]
pub(super) fn fma_f32_f32(acc: &mut [f32], src: &[f32], w: f32, use_avx512: bool) {
    let len = acc.len();
    #[cfg(target_arch = "x86_64")]
    if use_avx512 {
        unsafe {
            fma_f32_f32_avx512(acc.as_mut_ptr(), src.as_ptr(), w, len);
        }
        return;
    }
    let _ = use_avx512;
    for d in 0..len {
        acc[d] += w * src[d];
    }
}

/// AVX-512: acc[d] += w * src[d] (both F32)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn fma_f32_f32_avx512(acc: *mut f32, src: *const f32, w: f32, len: usize) {
    use core::arch::x86_64::*;
    let vw = _mm512_set1_ps(w);
    let chunks = len / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v_src = unsafe { _mm512_loadu_ps(src.add(off)) };
        let v_acc = unsafe { _mm512_loadu_ps(acc.add(off)) };
        let v_res = _mm512_fmadd_ps(vw, v_src, v_acc);
        unsafe { _mm512_storeu_ps(acc.add(off), v_res) };
    }
    for d in (chunks * 16)..len {
        unsafe { *acc.add(d) += w * *src.add(d) };
    }
}

/// acc[d] *= scale for d in 0..acc.len()
#[inline]
pub(super) fn scale_f32(acc: &mut [f32], scale: f32, use_avx512: bool) {
    let len = acc.len();
    #[cfg(target_arch = "x86_64")]
    if use_avx512 {
        unsafe {
            scale_f32_avx512(acc.as_mut_ptr(), scale, len);
        }
        return;
    }
    let _ = use_avx512;
    for d in 0..len {
        acc[d] *= scale;
    }
}

/// AVX-512: acc[d] *= scale
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn scale_f32_avx512(acc: *mut f32, scale: f32, len: usize) {
    use core::arch::x86_64::*;
    let vs = _mm512_set1_ps(scale);
    let chunks = len / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = unsafe { _mm512_loadu_ps(acc.add(off)) };
        unsafe { _mm512_storeu_ps(acc.add(off), _mm512_mul_ps(v, vs)) };
    }
    for d in (chunks * 16)..len {
        unsafe { *acc.add(d) *= scale };
    }
}

// ── AVX-512 vectorized exp (Cody-Waite + Horner 5th order) ──────────

/// Vectorized exp(x) for 16 F32 values using AVX-512.
/// Same implementation as silu_mul.rs — Cody-Waite range reduction + 5th order Horner.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn exp_ps_avx512(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm512_set1_ps(0.693145752);
    let ln2_lo = _mm512_set1_ps(1.42860677e-6);
    let half = _mm512_set1_ps(0.5);

    let x = _mm512_max_ps(x, _mm512_set1_ps(-87.33654));
    let x = _mm512_min_ps(x, _mm512_set1_ps(88.72284));

    let fx = _mm512_fmadd_ps(x, log2e, half);
    let n = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    let r = _mm512_sub_ps(x, _mm512_mul_ps(n, ln2_hi));
    let r = _mm512_sub_ps(r, _mm512_mul_ps(n, ln2_lo));

    let c1 = _mm512_set1_ps(1.0);
    let c2 = _mm512_set1_ps(0.5);
    let c3 = _mm512_set1_ps(0.16666666);
    let c4 = _mm512_set1_ps(0.04166666);
    let c5 = _mm512_set1_ps(0.00833333);

    let mut p = _mm512_fmadd_ps(c5, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c1);

    let n_i = _mm512_cvtps_epi32(n);
    let bias = _mm512_set1_epi32(127);
    let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(n_i, bias), 23));
    _mm512_mul_ps(p, pow2n)
}

/// Vectorized online softmax for one Q row's score block:
///   1. Find max of scores[0..n_size]
///   2. Rescale previous v_prime and s_prime with exp(old_max - new_max)
///   3. Compute exp(scores[j] - new_max) and accumulate sum
///
/// Returns (new_max, block_sum). scores[] is modified in-place to contain weights.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) fn online_softmax_avx512(
    scores: *mut f32,     // [block_n], raw Q@K^T (unscaled), modified in-place to exp weights
    n_size: usize,
    m_prime: f32,         // previous max (already scaled)
    sm_scale: f32,        // softmax scale (1/sqrt(head_dim)), fused to avoid separate pass
) -> (f32, f32, f32) {   // (new_max, block_sum, rescale_factor)
    use core::arch::x86_64::*;

    let chunks16 = n_size / 16;
    let scale_vec = _mm512_set1_ps(sm_scale);

    // Step 1: Find max(score * sm_scale)
    let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
    for c in 0..chunks16 {
        let v = _mm512_mul_ps(unsafe { _mm512_loadu_ps(scores.add(c * 16)) }, scale_vec);
        max_vec = _mm512_max_ps(max_vec, v);
    }
    let mut m_i = _mm512_reduce_max_ps(max_vec);
    for j in (chunks16 * 16)..n_size {
        let v = unsafe { *scores.add(j) } * sm_scale;
        if v > m_i { m_i = v; }
    }
    m_i = m_i.max(m_prime);

    let rescale = (m_prime - m_i).exp();

    // Step 2: Vectorized exp(score * sm_scale - max) + sum
    let m_i_vec = _mm512_set1_ps(m_i);
    let mut sum_vec = _mm512_setzero_ps();
    for c in 0..chunks16 {
        let off = c * 16;
        let v = _mm512_mul_ps(unsafe { _mm512_loadu_ps(scores.add(off)) }, scale_vec);
        let e = exp_ps_avx512(_mm512_sub_ps(v, m_i_vec));
        unsafe { _mm512_storeu_ps(scores.add(off), e) };
        sum_vec = _mm512_add_ps(sum_vec, e);
    }
    let mut block_sum = _mm512_reduce_add_ps(sum_vec);
    for j in (chunks16 * 16)..n_size {
        let w = (unsafe { *scores.add(j) } * sm_scale - m_i).exp();
        unsafe { *scores.add(j) = w };
        block_sum += w;
    }

    (m_i, block_sum, rescale)
}

/// Scalar fallback for online softmax with fused sm_scale.
pub(super) fn softmax_scalar(scores: &mut [f32], n_size: usize, m_prime: f32, sm_scale: f32) -> (f32, f32, f32) {
    let mut m_i = f32::NEG_INFINITY;
    for j in 0..n_size {
        let v = scores[j] * sm_scale;
        if v > m_i { m_i = v; }
    }
    m_i = m_i.max(m_prime);
    let rescale = (m_prime - m_i).exp();
    let mut block_sum = 0.0f32;
    for j in 0..n_size {
        let w = (scores[j] * sm_scale - m_i).exp();
        scores[j] = w;
        block_sum += w;
    }
    (m_i, block_sum, rescale)
}

/// Select adaptive block sizes based on sequence length (matches SGLang).
/// Both QK^T and V accumulation now use brgemm (AMX), so larger blocks are efficient.
pub(super) fn select_blocks(slen: usize) -> (usize, usize) {
    if crate::ops::onednn::brgemm_available() {
        return match slen {
            0..=256 => (32, 64),
            0..=512 => (128, 256),
            0..=1024 => (128, 512),
            0..=4096 => (256, 512),
            _ => (512, 768),
        };
    }
    // Fallback for non-brgemm: smaller blocks for hand-written micro-kernels
    match slen {
        0..=256 => (32, 64),
        0..=1024 => (64, 128),
        0..=4096 => (128, 256),
        _ => (256, 512),
    }
}

// ── BF16 dot product ────────────────────────────────────────────────────

/// Dot product of two BF16 vectors, accumulated in F32.
#[inline]
pub(super) fn dot_bf16_f32(a: &[u16], b: &[u16], len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // Safety: feature detection checked above
            return unsafe { dot_bf16_f32_avx512(a.as_ptr(), b.as_ptr(), len) };
        }
    }
    dot_bf16_f32_scalar(a, b, len)
}

/// Scalar fallback dot product.
#[inline]
pub(super) fn dot_bf16_f32_scalar(a: &[u16], b: &[u16], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
    }
    sum
}

/// AVX-512 BF16 dot product: loads 16 BF16 pairs at a time, FMA in F32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn dot_bf16_f32_avx512(a: *const u16, b: *const u16, len: usize) -> f32 {
    use core::arch::x86_64::*;
    let chunks = len / 16;
    let mut acc = _mm512_setzero_ps();

    for i in 0..chunks {
        let av = bf16x16_load_as_f32(unsafe { a.add(i * 16) });
        let bv = bf16x16_load_as_f32(unsafe { b.add(i * 16) });
        acc = _mm512_fmadd_ps(av, bv, acc);
    }

    let mut sum = _mm512_reduce_add_ps(acc);

    // Handle remainder
    for j in (chunks * 16)..len {
        sum += unsafe { bf16_to_f32(*a.add(j)) * bf16_to_f32(*b.add(j)) };
    }

    sum
}

// ── AVX-512 BF16 load helper ────────────────────────────────────────────

/// Load 16 BF16 values and convert to F32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn bf16x16_load_as_f32(ptr: *const u16) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    let bf16_vals = unsafe { _mm256_loadu_si256(ptr as *const __m256i) };
    let extended = _mm512_cvtepu16_epi32(bf16_vals);
    let shifted = _mm512_slli_epi32(extended, 16);
    _mm512_castsi512_ps(shifted)
}
