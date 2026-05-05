//! Q3_K × Q8_K dot product kernel.
//!
//! Q3_K is a compact K-quant: 256 values per block, 3.4375 bpw.
//! Values are 3-bit: 2 low bits in `qs`, 1 high bit in `hmask`.
//! 16 sub-blocks of 16 elements, scales are 6-bit signed (stored as unsigned, subtract 32).
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Scalar Q3_K × Q8_K dot product (reference implementation).
///
/// Direct port of llama.cpp `ggml_vec_dot_q3_K_q8_K_generic`.
pub fn vec_dot_q3k_q8k_scalar(x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(x.len(), y.len());

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    let mut auxs = [0u32; 4];

    for (xb, yb) in x.iter().zip(y.iter()) {
        aux32.fill(0);

        // Unpack 3-bit values: 2 low bits from qs, 1 high bit from hmask
        let mut a_idx = 0;
        let mut q3_off = 0;
        let mut m: u8 = 1;
        for _j in 0..(QK_K / 128) {
            for l in 0..32 {
                aux8[a_idx + l] = (xb.qs[q3_off + l] & 3) as i8;
                aux8[a_idx + l] -= if xb.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a_idx += 32;
            m <<= 1;
            for l in 0..32 {
                aux8[a_idx + l] = ((xb.qs[q3_off + l] >> 2) & 3) as i8;
                aux8[a_idx + l] -= if xb.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a_idx += 32;
            m <<= 1;
            for l in 0..32 {
                aux8[a_idx + l] = ((xb.qs[q3_off + l] >> 4) & 3) as i8;
                aux8[a_idx + l] -= if xb.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a_idx += 32;
            m <<= 1;
            for l in 0..32 {
                aux8[a_idx + l] = ((xb.qs[q3_off + l] >> 6) & 3) as i8;
                aux8[a_idx + l] -= if xb.hmask[l] & m != 0 { 0 } else { 4 };
            }
            a_idx += 32;
            m <<= 1;
            q3_off += 32;
        }

        // Unpack 6-bit signed scales (subtract 32)
        auxs[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
        auxs[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
        auxs[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

        let tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        auxs[3] = ((auxs[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        auxs[0] = (auxs[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        auxs[1] = (auxs[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let scales: [i8; 16] = bytemuck::cast(auxs);

        // Dot product with signed scales
        let mut q8_off = 0;
        let mut a_off = 0;
        for j in 0..(QK_K / 16) {
            let s = (scales[j] as i32) - 32;
            for l in 0..8 {
                aux16[l] = yb.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for l in 0..8 {
                aux32[l] += s * aux16[l] as i32;
            }
            q8_off += 8;
            a_off += 8;
            for l in 0..8 {
                aux16[l] = yb.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for l in 0..8 {
                aux32[l] += s * aux16[l] as i32;
            }
            q8_off += 8;
            a_off += 8;
        }

        let d = fp16_to_f32(xb.d) * yb.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }

    sums.iter().sum()
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    // Q3_K scale shuffle: broadcasts scale pairs across 32-byte vector.
    // Same layout as Q2_K: 16 bytes repeated in low/high 128-bit lanes.
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

    /// AVX2 Q3_K × Q8_K dot product.
    ///
    /// Port of llama.cpp x86 AVX2 implementation.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q3k_q8k_avx2(x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        unsafe {
            let m3 = _mm256_set1_epi8(3);
            let mone = _mm256_set1_epi8(1);
            let m32 = _mm_set1_epi8(32);

            let mut acc = _mm256_setzero_ps();
            let mut aux = [0u32; 3];

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = yb.d * fp16_to_f32(xb.d);

                let mut q3 = xb.qs.as_ptr();
                let mut q8 = yb.qs.as_ptr();

                // Set up scales: unpack 6-bit signed scales
                aux[0] =
                    u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
                aux[1] =
                    u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
                aux[2] =
                    u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

                let mut scales128 = _mm_set_epi32(
                    (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
                    (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
                    ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
                    ((aux[0] & KMASK2) | (((aux[2] >> 0) & KMASK1) << 4)) as i32,
                );
                scales128 = _mm_sub_epi8(scales128, m32);
                let all_scales = _mm256_cvtepi8_epi16(scales128);
                let l_scales = _mm256_extracti128_si256(all_scales, 0);
                let h_scales = _mm256_extracti128_si256(all_scales, 1);
                let scales = [
                    _mm256_set_m128i(l_scales, l_scales),
                    _mm256_set_m128i(h_scales, h_scales),
                ];

                let hbits = _mm256_loadu_si256(xb.hmask.as_ptr() as *const __m256i);

                let mut sumi = _mm256_setzero_si256();
                let mut bit: i64 = 0;

                for j in 0..(QK_K / 128) {
                    let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
                    q3 = q3.add(32);

                    // Use _mm256_sll_epi16 (variable shift) since bit is runtime
                    let bv = _mm_cvtsi64_si128(bit);

                    // Unpack low 2 bits and compute high bit contribution
                    let q3l_0 = _mm256_and_si256(q3bits, m3);
                    let q3h_0 = _mm256_slli_epi16::<2>(_mm256_srl_epi16(
                        _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, bv)),
                        bv,
                    ));
                    bit += 1;
                    let bv = _mm_cvtsi64_si128(bit);

                    let q3l_1 = _mm256_and_si256(_mm256_srli_epi16::<2>(q3bits), m3);
                    let q3h_1 = _mm256_slli_epi16::<2>(_mm256_srl_epi16(
                        _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, bv)),
                        bv,
                    ));
                    bit += 1;
                    let bv = _mm_cvtsi64_si128(bit);

                    let q3l_2 = _mm256_and_si256(_mm256_srli_epi16::<4>(q3bits), m3);
                    let q3h_2 = _mm256_slli_epi16::<2>(_mm256_srl_epi16(
                        _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, bv)),
                        bv,
                    ));
                    bit += 1;
                    let bv = _mm_cvtsi64_si128(bit);

                    let q3l_3 = _mm256_and_si256(_mm256_srli_epi16::<6>(q3bits), m3);
                    let q3h_3 = _mm256_slli_epi16::<2>(_mm256_srl_epi16(
                        _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, bv)),
                        bv,
                    ));
                    bit += 1;

                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);

                    // q3h contribution: if hmask bit NOT set, q3h = 2 (which gets subtracted)
                    // if hmask bit IS set, q3h = 0 (no subtraction)
                    let q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                    let q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                    let q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                    let q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

                    let mut p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
                    let mut p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
                    let mut p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
                    let mut p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

                    p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                    p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                    p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                    p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                    // Multiply by scales (indices 0-3 select 4 scale pairs within scales[j])
                    p16_0 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)),
                        p16_0,
                    );
                    p16_1 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)),
                        p16_1,
                    );
                    p16_2 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)),
                        p16_2,
                    );
                    p16_3 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)),
                        p16_3,
                    );

                    p16_0 = _mm256_add_epi32(p16_0, p16_1);
                    p16_2 = _mm256_add_epi32(p16_2, p16_3);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
                }

                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
            }

            hsum_float_8(acc)
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Q3_K × Q8_K dot product, auto-dispatching to the best available kernel.
#[inline]
pub fn vec_dot_q3k_q8k(x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { super::neon::q3_k::vec_dot_q3k_q8k_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q3k_q8k_avx2(x, y) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { /* fall through to scalar below */ }

    #[allow(unreachable_code)]
    vec_dot_q3k_q8k_scalar(x, y)
}

/// Q3_K matmul: y[M,N] = x[M,K] @ W[N,K]^T.
pub fn quantized_matmul_q3k(
    x: &[f32],
    w: &[BlockQ3K],
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
            out_row[j] = vec_dot_q3k_q8k(w_row, &x_q8);
        }
    });
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_q3k_blocks(values: &[f32]) -> Vec<BlockQ3K> {
        crate::ops::quant::quantize_f32_q3k(values)
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() + 0.5).collect();
        let q3 = make_test_q3k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q3k_q8k_scalar(&q3, &q8);
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

        let q3 = make_test_q3k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);
        let our_dot = vec_dot_q3k_q8k_scalar(&q3, &q8);

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

        let q3 = make_test_q3k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);

        let scalar = vec_dot_q3k_q8k_scalar(&q3, &q8);
        let avx2 = unsafe { avx2::vec_dot_q3k_q8k_avx2(&q3, &q8) };

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
            w_blocks.extend(make_test_q3k_blocks(&w_data[j * k..(j + 1) * k]));
        }

        let mut out = vec![0.0f32; m * n];
        quantized_matmul_q3k(&x_data, &w_blocks, &mut out, m, n, k);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite output: {out:?}"
        );
    }
}
