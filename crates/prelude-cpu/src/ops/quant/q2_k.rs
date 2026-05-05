//! Q2_K × Q8_K dot product kernel.
//!
//! Q2_K is an extreme-compression K-quant: 256 values per block, 2.625 bpw.
//! 16 sub-blocks of 16 elements, each with 4-bit scale (low nibble) and
//! 4-bit minimum (high nibble) packed in `scales[16]`.
//! Values are 2-bit, four per byte in `qs[64]`.
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Scalar Q2_K × Q8_K dot product (reference implementation).
///
/// Direct port of llama.cpp `ggml_vec_dot_q2_K_q8_K_generic`.
pub fn vec_dot_q2k_q8k_scalar(x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(x.len(), y.len());

    let mut sumf: f32 = 0.0;

    for (xb, yb) in x.iter().zip(y.iter()) {
        // Min contribution via bsums: sum(bsums[j] * (scales[j] >> 4))
        let mut summs: i32 = 0;
        for j in 0..16 {
            summs += yb.bsums[j] as i32 * (xb.scales[j] >> 4) as i32;
        }

        let dall = yb.d * fp16_to_f32(xb.d);
        let dmin = yb.d * fp16_to_f32(xb.dmin);

        let mut isum: i32 = 0;
        let mut is = 0usize;
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;

        for _k in 0..(QK_K / 128) {
            let mut shift = 0u32;
            for _j in 0..4 {
                let d = (xb.scales[is] & 0xF) as i32;
                is += 1;
                let mut isuml: i32 = 0;
                for l in 0..16 {
                    isuml += yb.qs[q8_off + l] as i32 * ((xb.qs[q2_off + l] as i32 >> shift) & 3);
                }
                isum += d * isuml;

                let d = (xb.scales[is] & 0xF) as i32;
                is += 1;
                isuml = 0;
                for l in 16..32 {
                    isuml += yb.qs[q8_off + l] as i32 * ((xb.qs[q2_off + l] as i32 >> shift) & 3);
                }
                isum += d * isuml;

                shift += 2;
                q8_off += 32;
            }
            q2_off += 32;
        }

        sumf += dall * isum as f32 - dmin * summs as f32;
    }

    sumf
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    // Same scale shuffle as Q3_K: broadcasts scale pairs.
    static SCALE_SHUFFLE_Q3K: [[u8; 32]; 4] = [
        [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
            3, 2, 3,
        ],
        [
            4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6,
            7, 6, 7,
        ],
        [
            8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
            10, 11, 10, 11, 10, 11,
        ],
        [
            12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15,
            14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        ],
    ];

    #[inline(always)]
    unsafe fn get_scale_shuffle_q3k(i: usize) -> __m256i {
        unsafe { _mm256_loadu_si256(SCALE_SHUFFLE_Q3K.get_unchecked(i).as_ptr() as *const __m256i) }
    }

    #[inline(always)]
    unsafe fn hsum_float_8(x: __m256) -> f32 {
        unsafe {
            let hi128 = _mm256_extractf128_ps(x, 1);
            let lo128 = _mm256_castps256_ps128(x);
            let sum128 = _mm_add_ps(hi128, lo128);
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_movehdup_ps(sum64);
            _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
        }
    }

    /// AVX2 Q2_K × Q8_K dot product.
    ///
    /// Port of llama.cpp x86 AVX2 implementation.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q2k_q8k_avx2(x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        unsafe {
            let m3 = _mm256_set1_epi8(3);
            let m4 = _mm_set1_epi8(0x0F);

            let mut acc = _mm256_setzero_ps();

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = yb.d * fp16_to_f32(xb.d);
                let dmin = -yb.d * fp16_to_f32(xb.dmin);

                let mut q2 = xb.qs.as_ptr();
                let mut q8 = yb.qs.as_ptr();

                // Load and split scales (low nibble) and mins (high nibble)
                let mins_and_scales = _mm_loadu_si128(xb.scales.as_ptr() as *const __m128i);
                let scales8 = _mm_and_si128(mins_and_scales, m4);
                let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
                let mins = _mm256_cvtepi8_epi16(mins8);
                let prod = _mm256_madd_epi16(
                    mins,
                    _mm256_loadu_si256(yb.bsums.as_ptr() as *const __m256i),
                );

                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

                let all_scales = _mm256_cvtepi8_epi16(scales8);
                let l_scales = _mm256_extracti128_si256(all_scales, 0);
                let h_scales = _mm256_extracti128_si256(all_scales, 1);
                let scales = [
                    _mm256_set_m128i(l_scales, l_scales),
                    _mm256_set_m128i(h_scales, h_scales),
                ];

                let mut sumi = _mm256_setzero_si256();

                for j in 0..(QK_K / 128) {
                    let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
                    q2 = q2.add(32);

                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);

                    let q2_0 = _mm256_and_si256(q2bits, m3);
                    let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                    let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                    let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

                    let mut p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                    let mut p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                    let mut p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                    let mut p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                    p0 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)),
                        p0,
                    );
                    p1 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)),
                        p1,
                    );
                    p2 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)),
                        p2,
                    );
                    p3 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)),
                        p3,
                    );

                    p0 = _mm256_add_epi32(p0, p1);
                    p2 = _mm256_add_epi32(p2, p3);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
                }

                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
            }

            hsum_float_8(acc)
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Q2_K × Q8_K dot product, auto-dispatching to the best available kernel.
#[inline]
pub fn vec_dot_q2k_q8k(x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { super::neon::q2_k::vec_dot_q2k_q8k_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q2k_q8k_avx2(x, y) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { /* fall through to scalar below */ }

    #[allow(unreachable_code)]
    vec_dot_q2k_q8k_scalar(x, y)
}

/// Q2_K matmul: y[M,N] = x[M,K] @ W[N,K]^T.
pub fn quantized_matmul_q2k(
    x: &[f32],
    w: &[BlockQ2K],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(k % QK_K, 0, "k must be multiple of {QK_K}");
    let nb = k / QK_K;
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * nb);
    assert_eq!(out.len(), m * n);

    use super::quantize::quantize_row_q8k;
    use rayon::prelude::*;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let x_row = &x[i * k..(i + 1) * k];
        let x_q8 = quantize_row_q8k(x_row);

        for j in 0..n {
            let w_row = &w[j * nb..(j + 1) * nb];
            out_row[j] = vec_dot_q2k_q8k(w_row, &x_q8);
        }
    });
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_q2k_blocks(values: &[f32]) -> Vec<BlockQ2K> {
        crate::ops::quant::quantize_f32_q2k(values)
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() + 0.5).collect();
        let q2 = make_test_q2k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q2k_q8k_scalar(&q2, &q8);
        assert!(
            result > 0.0,
            "self dot product should be positive, got {result}"
        );
    }

    #[test]
    fn scalar_self_dot_consistency() {
        let k = QK_K;
        let values: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q2 = make_test_q2k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);
        let our_dot = vec_dot_q2k_q8k_scalar(&q2, &q8);

        // Result should be finite and non-zero for non-trivial inputs
        assert!(
            our_dot.is_finite(),
            "dot product should be finite, got {our_dot}"
        );
        assert!(
            our_dot.abs() > 1e-6,
            "dot product should be non-zero for non-trivial inputs"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let values: Vec<f32> = (0..QK_K * 4)
            .map(|i| ((i as f32) * 0.007).sin() * 2.0)
            .collect();
        let x_vals: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q2 = make_test_q2k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);

        let scalar = vec_dot_q2k_q8k_scalar(&q2, &q8);
        let avx2 = unsafe { avx2::vec_dot_q2k_q8k_avx2(&q2, &q8) };

        let rel_err = (scalar - avx2).abs() / scalar.abs().max(1e-6);
        assert!(
            rel_err < 1e-5,
            "AVX2 vs scalar: scalar={scalar}, avx2={avx2}, rel_err={rel_err}"
        );
    }

    #[test]
    fn matmul_basic() {
        let k = QK_K;
        let n = 4;
        let m = 2;
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let x_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.011).cos()).collect();

        let mut w_blocks = Vec::new();
        for j in 0..n {
            w_blocks.extend(make_test_q2k_blocks(&w_data[j * k..(j + 1) * k]));
        }

        let mut out = vec![0.0f32; m * n];
        quantized_matmul_q2k(&x_data, &w_blocks, &mut out, m, n, k);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite output: {out:?}"
        );
    }
}
