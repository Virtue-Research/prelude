//! Q5_0 × Q8_0 dot product kernel.
//!
//! Q5_0 is symmetric 5-bit: 4 low bits in `qs`, 1 high bit in `qh[4]`.
//! Value = d * (combined_5bit - 16), where combined ∈ [0..31].
//! Paired with Q8_0 activation.
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Scalar Q5_0 × Q8_0 dot product.
pub fn vec_dot_q5_0_q8_0_scalar(x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len(), y.len());
    let mut sumf: f32 = 0.0;

    for (xb, yb) in x.iter().zip(y.iter()) {
        let qh = u32::from_le_bytes(xb.qh);

        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..16 {
            let xh_0 = (((qh & (1u32 << j)) >> j) << 4) as u8;
            let xh_1 = ((qh & (1u32 << (j + 16))) >> (j + 12)) as u8;

            let x0 = ((xb.qs[j] & 0x0F) | xh_0) as i8 - 16;
            let x1 = ((xb.qs[j] >> 4) | xh_1) as i8 - 16;

            sumi0 += x0 as i32 * yb.qs[j] as i32;
            sumi1 += x1 as i32 * yb.qs[j + 16] as i32;
        }

        sumf += (fp16_to_f32(xb.d) * fp16_to_f32(yb.d)) * (sumi0 + sumi1) as f32;
    }
    sumf
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

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

    /// Spread 32 bits to 32 bytes of {0x00, 0xFF}.
    #[inline(always)]
    unsafe fn bytes_from_bits_32(x: &[u8; 4]) -> __m256i {
        unsafe {
            let x32 = u32::from_le_bytes(*x);
            let shuf_mask = _mm256_set_epi64x(
                0x0303030303030303u64 as i64,
                0x0202020202020202u64 as i64,
                0x0101010101010101u64 as i64,
                0x0000000000000000u64 as i64,
            );
            let mut bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32 as i32), shuf_mask);
            let bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeu64 as i64);
            bytes = _mm256_or_si256(bytes, bit_mask);
            _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q5_0_q8_0_avx2(x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());

        unsafe {
            let m4 = _mm256_set1_epi8(0x0F);
            let mut acc = _mm256_setzero_ps();

            for (xb, yb) in x.iter().zip(y.iter()) {
                // Unpack nibbles
                let q4bits = _mm_loadu_si128(xb.qs.as_ptr() as *const __m128i);
                let q4 = _mm256_set_m128i(_mm_srli_epi16(q4bits, 4), q4bits);
                let q4 = _mm256_and_si256(q4, m4);

                // Expand high bits to bytes and OR with low 4 bits
                let qh = bytes_from_bits_32(&xb.qh);
                let qh = _mm256_and_si256(qh, _mm256_set1_epi8(0x10));
                let q5 = _mm256_or_si256(q4, qh);

                // Subtract 16 bias: q5 values are [0..31], need [-16..+15]
                // Use sign trick: maddubs(q5, q8) - maddubs(16, q8)
                let q8 = _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i);
                let s16 = _mm256_set1_epi8(16);

                let p = _mm256_maddubs_epi16(q5, q8);
                let s = _mm256_maddubs_epi16(s16, q8);
                let p = _mm256_sub_epi16(p, s);

                let ones = _mm256_set1_epi16(1);
                let p32 = _mm256_madd_epi16(p, ones);

                let d = fp16_to_f32(xb.d) * fp16_to_f32(yb.d);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
            }

            hsum_float_8(acc)
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

#[inline]
pub fn vec_dot_q5_0_q8_0(x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { super::neon::q5_0::vec_dot_q5_0_q8_0_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q5_0_q8_0_avx2(x, y) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { /* fall through to scalar below */ }

    #[allow(unreachable_code)]
    vec_dot_q5_0_q8_0_scalar(x, y)
}

/// Q5_0 matmul: y[M,N] = x[M,K] @ W[N,K]^T.
pub fn quantized_matmul_q5_0(
    x: &[f32],
    w: &[BlockQ5_0],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(k % QK8_0, 0);
    let nb = k / QK8_0;
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * nb);
    assert_eq!(out.len(), m * n);

    use super::quantize::quantize_row_q8_0;
    use rayon::prelude::*;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let x_row = &x[i * k..(i + 1) * k];
        let x_q8 = quantize_row_q8_0(x_row);
        for j in 0..n {
            out_row[j] = vec_dot_q5_0_q8_0(&w[j * nb..(j + 1) * nb], &x_q8);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_blocks(values: &[f32]) -> Vec<BlockQ5_0> {
        crate::ops::quant::quantize_f32_q5_0(values)
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.1).sin() + 0.5).collect();
        let q5 = make_test_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8_0_scalar(&values);
        let result = vec_dot_q5_0_q8_0_scalar(&q5, &q8);
        assert!(result > 0.0, "self dot should be positive, got {result}");
    }

    #[test]
    fn scalar_vs_dequant() {
        let k = 128;
        let values: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let q5 = make_test_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8_0_scalar(&x_vals);
        let our_dot = vec_dot_q5_0_q8_0_scalar(&q5, &q8);

        // Dequantize our blocks and compute reference dot product
        let mut w_deq = vec![0.0f32; k];
        for (bi, b) in q5.iter().enumerate() {
            let d = fp16_to_f32(b.d);
            let qh = u32::from_le_bytes(b.qh);
            for j in 0..16 {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
                let x0 = ((b.qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((b.qs[j] >> 4) | xh_1) as i32 - 16;
                w_deq[bi * 32 + j] = x0 as f32 * d;
                w_deq[bi * 32 + 16 + j] = x1 as f32 * d;
            }
        }
        let ref_dot: f32 = w_deq.iter().zip(x_vals.iter()).map(|(w, x)| w * x).sum();

        let rel_err = (our_dot - ref_dot).abs() / ref_dot.abs().max(1e-6);
        assert!(
            rel_err < 0.05,
            "our={our_dot}, ref={ref_dot}, rel_err={rel_err}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let values: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.013).cos()).collect();
        let q5 = make_test_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8_0_scalar(&x_vals);
        let scalar = vec_dot_q5_0_q8_0_scalar(&q5, &q8);
        let avx2 = unsafe { avx2::vec_dot_q5_0_q8_0_avx2(&q5, &q8) };
        let rel_err = (scalar - avx2).abs() / scalar.abs().max(1e-6);
        assert!(
            rel_err < 1e-5,
            "scalar={scalar}, avx2={avx2}, rel_err={rel_err}"
        );
    }
}
