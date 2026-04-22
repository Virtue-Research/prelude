//! Q4_1 × Q8_1 dot product kernel.
//!
//! Q4_1 is asymmetric 4-bit: each 32-element block has scale `d` and minimum `m`.
//! Value = d * nibble + m, where nibble ∈ [0..15].
//! Paired with Q8_1 activation (which precomputes `s = d * sum(qs)`).
//!
//! The dot product formula:
//!   dot = d_w * d_a * sum(nibble[i] * q8[i]) + m_w * s_a
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Scalar Q4_1 × Q8_1 dot product (reference implementation).
pub fn vec_dot_q4_1_q8_1_scalar(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    assert_eq!(x.len(), y.len());
    let mut sumf: f32 = 0.0;

    for (xb, yb) in x.iter().zip(y.iter()) {
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..16 {
            let v0 = (xb.qs[j] & 0x0F) as i32;
            let v1 = (xb.qs[j] >> 4) as i32;
            sumi0 += v0 * yb.qs[j] as i32;
            sumi1 += v1 * yb.qs[j + 16] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += (fp16_to_f32(xb.d) * fp16_to_f32(yb.d)) * sumi as f32
              + fp16_to_f32(xb.m) * fp16_to_f32(yb.s);
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

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q4_1_q8_1_avx2(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
        assert_eq!(x.len(), y.len());

        unsafe {
            let m4 = _mm256_set1_epi8(0x0F);
            let mut acc = _mm256_setzero_ps();
            let mut summs: f32 = 0.0;

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = fp16_to_f32(xb.d) * fp16_to_f32(yb.d);
                summs += fp16_to_f32(xb.m) * fp16_to_f32(yb.s);

                let q4bits = _mm_loadu_si128(xb.qs.as_ptr() as *const __m128i);
                let q4 = _mm256_set_m128i(
                    _mm_srli_epi16(q4bits, 4),
                    q4bits,
                );
                let q4 = _mm256_and_si256(q4, m4);

                let q8 = _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i);

                let p = _mm256_maddubs_epi16(q4, q8);
                let ones = _mm256_set1_epi16(1);
                let p32 = _mm256_madd_epi16(p, ones);

                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(d),
                    _mm256_cvtepi32_ps(p32),
                    acc,
                );
            }

            hsum_float_8(acc) + summs
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

#[inline]
pub fn vec_dot_q4_1_q8_1(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q4_1_q8_1_avx2(x, y) };
        }
    }
    vec_dot_q4_1_q8_1_scalar(x, y)
}

/// Q4_1 matmul: y[M,N] = x[M,K] @ W[N,K]^T.
pub fn quantized_matmul_q4_1(
    x: &[f32], w: &[BlockQ4_1], out: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    assert_eq!(k % QK8_0, 0);
    let nb = k / QK8_0;
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * nb);
    assert_eq!(out.len(), m * n);

    use rayon::prelude::*;
    use super::quantize::quantize_row_q8_1;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let x_row = &x[i * k..(i + 1) * k];
        let x_q8 = quantize_row_q8_1(x_row);
        for j in 0..n {
            out_row[j] = vec_dot_q4_1_q8_1(&w[j * nb..(j + 1) * nb], &x_q8);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    fn make_test_blocks(values: &[f32]) -> Vec<BlockQ4_1> {
        assert!(values.len() % 32 == 0);
        let t = Tensor::from_vec(values.to_vec(), (values.len(),), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&t, GgmlDType::Q4_1, &Device::Cpu).unwrap();
        bytemuck::cast_slice(&qt.data().unwrap()).to_vec()
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.1).sin() + 0.5).collect();
        let q4 = make_test_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8_1_scalar(&values);
        let result = vec_dot_q4_1_q8_1_scalar(&q4, &q8);
        assert!(result > 0.0, "self dot should be positive, got {result}");
    }

    #[test]
    fn scalar_vs_candle_dequant() {
        let k = 128;
        let values: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q4 = make_test_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8_1_scalar(&x_vals);
        let our_dot = vec_dot_q4_1_q8_1_scalar(&q4, &q8);

        let t = Tensor::from_vec(values, (k,), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&t, GgmlDType::Q4_1, &Device::Cpu).unwrap();
        let w_deq = qt.dequantize(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        let ref_dot: f32 = w_deq.iter().zip(x_vals.iter()).map(|(w, x)| w * x).sum();

        let rel_err = (our_dot - ref_dot).abs() / ref_dot.abs().max(1e-6);
        assert!(rel_err < 0.05, "our={our_dot}, ref={ref_dot}, rel_err={rel_err}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }
        let values: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.013).cos()).collect();
        let q4 = make_test_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8_1_scalar(&x_vals);
        let scalar = vec_dot_q4_1_q8_1_scalar(&q4, &q8);
        let avx2 = unsafe { avx2::vec_dot_q4_1_q8_1_avx2(&q4, &q8) };
        let rel_err = (scalar - avx2).abs() / scalar.abs().max(1e-6);
        assert!(rel_err < 1e-5, "scalar={scalar}, avx2={avx2}, rel_err={rel_err}");
    }
}
