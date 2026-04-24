//! Q5_K × Q8_K dot product kernel.
//!
//! Q5_K is a high-quality K-quant: 256 values per block, 5.5 bpw.
//! Same scale/min packing as Q4_K (8 sub-blocks, 6-bit scales).
//! Values are 5-bit: 4 low bits in `qs` + 1 high bit in `qh`.
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Unpack 6-bit scales and mins from the 12-byte packed format.
/// Same packing as Q4_K.
fn unpack_scales_mins(raw_scales: &[u8; K_SCALE_SIZE]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes([raw_scales[0], raw_scales[1], raw_scales[2], raw_scales[3]]);
    utmp[1] = u32::from_le_bytes([raw_scales[4], raw_scales[5], raw_scales[6], raw_scales[7]]);
    utmp[2] = u32::from_le_bytes([raw_scales[8], raw_scales[9], raw_scales[10], raw_scales[11]]);

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let scales: [u8; 8] = bytemuck::cast([utmp[0], utmp[1]]);
    let mins: [u8; 8] = bytemuck::cast([utmp[2], utmp[3]]);
    (scales, mins)
}

/// Scalar Q5_K × Q8_K dot product (reference implementation).
///
/// Direct port of llama.cpp `ggml_vec_dot_q5_K_q8_K_generic`.
pub fn vec_dot_q5k_q8k_scalar(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(x.len(), y.len());

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    let mut sumf: f32 = 0.0;
    for (xb, yb) in x.iter().zip(y.iter()) {
        aux32.fill(0);

        // Unpack 5-bit values: 4 low bits from qs, 1 high bit from qh
        let mut a_idx = 0;
        let mut q4_off = 0;
        let mut m: u8 = 1;
        for _j in 0..(QK_K / 64) {
            for l in 0..32 {
                aux8[a_idx + l] = (xb.qs[q4_off + l] & 0xF) as i8;
                aux8[a_idx + l] += if xb.qh[l] & m != 0 { 16 } else { 0 };
            }
            a_idx += 32;
            m <<= 1;
            for l in 0..32 {
                aux8[a_idx + l] = (xb.qs[q4_off + l] >> 4) as i8;
                aux8[a_idx + l] += if xb.qh[l] & m != 0 { 16 } else { 0 };
            }
            a_idx += 32;
            m <<= 1;
            q4_off += 32;
        }

        let (scales, mins) = unpack_scales_mins(&xb.scales);

        // Min contribution via bsums
        let mut sumi: i32 = 0;
        for j in 0..(QK_K / 16) {
            sumi += yb.bsums[j] as i32 * mins[j / 2] as i32;
        }

        // Main dot product
        let mut a_off = 0;
        let mut q8_off = 0;
        let mut is = 0;
        for _j in 0..(QK_K / 32) {
            let scale = scales[is] as i32;
            is += 1;
            for l in 0..8 {
                aux16[l] = yb.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as i32;
            }
            q8_off += 8;
            a_off += 8;
            for l in 0..8 {
                aux16[l] = yb.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as i32;
            }
            q8_off += 8;
            a_off += 8;
            for l in 0..8 {
                aux16[l] = yb.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as i32;
            }
            q8_off += 8;
            a_off += 8;
            for l in 0..8 {
                aux16[l] = yb.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as i32;
            }
            q8_off += 8;
            a_off += 8;
        }

        let d = fp16_to_f32(xb.d) * yb.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        let dmin = fp16_to_f32(xb.dmin) * yb.d;
        sumf -= dmin * sumi as f32;
    }

    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    // Reuse the same scale shuffle table as Q4_K
    static SCALE_SHUFFLE_K4: [[u8; 32]; 8] = [
        [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [ 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        [ 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
        [ 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
        [ 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9],
        [10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11],
        [12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13],
        [14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15],
    ];

    #[inline(always)]
    unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
        unsafe { _mm256_loadu_si256(SCALE_SHUFFLE_K4.get_unchecked(i).as_ptr() as *const __m256i) }
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

    /// AVX2 Q5_K × Q8_K dot product.
    ///
    /// Port of llama.cpp x86 AVX2 implementation.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q5k_q8k_avx2(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        unsafe {
            let m4 = _mm256_set1_epi8(0x0F);
            let mone = _mm256_set1_epi8(1);
            let mzero = _mm_setzero_si128();

            let mut acc = _mm256_setzero_ps();
            let mut summs = 0.0f32;

            let mut utmp = [0u32; 4];

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = yb.d * fp16_to_f32(xb.d);
                let dmin = -yb.d * fp16_to_f32(xb.dmin);

                // Unpack scales/mins
                utmp[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
                utmp[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
                utmp[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

                utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
                let uaux = utmp[1] & KMASK1;
                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[2] = uaux;
                utmp[0] &= KMASK1;

                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32,
                ));

                // Min contribution via bsums
                let q8sums = _mm256_loadu_si256(yb.bsums.as_ptr() as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                summs += dmin * _mm_extract_epi32(hsum, 0) as f32;

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_set_m128i(sc128, sc128);

                let hbits = _mm256_loadu_si256(xb.qh.as_ptr() as *const __m256i);
                let mut hmask = mone;

                let mut sumi = _mm256_setzero_si256();
                let mut q5 = xb.qs.as_ptr();
                let mut q8 = yb.qs.as_ptr();

                let mut bit: i64 = 0;

                for j in 0..(QK_K / 64) {
                    let scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q5bits = _mm256_loadu_si256(q5 as *const __m256i);
                    q5 = q5.add(32);

                    // Low 4 bits + high bit for sub-block 0
                    let bv = _mm_cvtsi64_si128(bit);
                    let q5l_0 = _mm256_and_si256(q5bits, m4);
                    let q5h_0 = _mm256_slli_epi16::<4>(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), bv),
                    );
                    let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                    hmask = _mm256_slli_epi16::<1>(hmask);
                    bit += 1;

                    // High 4 bits + high bit for sub-block 1
                    let bv = _mm_cvtsi64_si128(bit);
                    let q5l_1 = _mm256_and_si256(_mm256_srli_epi16::<4>(q5bits), m4);
                    let q5h_1 = _mm256_slli_epi16::<4>(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), bv),
                    );
                    let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                    hmask = _mm256_slli_epi16::<1>(hmask);
                    bit += 1;

                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);

                    let mut p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                    let mut p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

                    p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                    p16_1 = _mm256_madd_epi16(scale_1, p16_1);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                }

                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }

            hsum_float_8(acc) + summs
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Q5_K × Q8_K dot product, auto-dispatching to the best available kernel.
#[inline]
pub fn vec_dot_q5k_q8k(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { super::neon::q5_k::vec_dot_q5k_q8k_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q5k_q8k_avx2(x, y) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { /* fall through to scalar below */ }

    #[allow(unreachable_code)]
    vec_dot_q5k_q8k_scalar(x, y)
}

/// Q5_K matmul: y[M,N] = x[M,K] @ W[N,K]^T.
pub fn quantized_matmul_q5k(
    x: &[f32],
    w: &[BlockQ5K],
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

    use rayon::prelude::*;
    use super::quantize::quantize_row_q8k;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let x_row = &x[i * k..(i + 1) * k];
        let x_q8 = quantize_row_q8k(x_row);

        for j in 0..n {
            let w_row = &w[j * nb..(j + 1) * nb];
            out_row[j] = vec_dot_q5k_q8k(w_row, &x_q8);
        }
    });
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_q5k_blocks(values: &[f32]) -> Vec<BlockQ5K> {
        crate::ops::quant::quantize_f32_q5k(values)
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() + 0.5).collect();
        let q5 = make_test_q5k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q5k_q8k_scalar(&q5, &q8);
        assert!(result > 0.0, "self dot product should be positive, got {result}");
    }

    #[test]
    fn scalar_zeros() {
        let values = vec![0.0f32; QK_K];
        let q5 = make_test_q5k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q5k_q8k_scalar(&q5, &q8);
        assert!(result.abs() < 1e-6, "zero dot should be ~0, got {result}");
    }

    #[test]
    fn scalar_self_dot_consistency() {
        let k = QK_K;
        let values: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q5 = make_test_q5k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);
        let our_dot = vec_dot_q5k_q8k_scalar(&q5, &q8);

        assert!(our_dot.is_finite(), "dot product should be finite, got {our_dot}");
        assert!(our_dot.abs() > 1e-6, "dot product should be non-zero for non-trivial inputs");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let values: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q5 = make_test_q5k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);

        let scalar = vec_dot_q5k_q8k_scalar(&q5, &q8);
        let avx2 = unsafe { avx2::vec_dot_q5k_q8k_avx2(&q5, &q8) };

        let rel_err = (scalar - avx2).abs() / scalar.abs().max(1e-6);
        assert!(rel_err < 1e-5, "AVX2 vs scalar: scalar={scalar}, avx2={avx2}, rel_err={rel_err}");
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
            w_blocks.extend(make_test_q5k_blocks(&w_data[j * k..(j + 1) * k]));
        }

        let mut out = vec![0.0f32; m * n];
        quantized_matmul_q5k(&x_data, &w_blocks, &mut out, m, n, k);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    }
}
