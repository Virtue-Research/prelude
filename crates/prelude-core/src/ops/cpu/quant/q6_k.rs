//! Q6_K × Q8_K dot product kernel.
//!
//! Q6_K is a high-quality K-quant: 256 values per block, 6.5625 bpw.
//! 16 sub-blocks of 16 elements each, with signed 8-bit scales.
//! Values are stored as 4 low bits (ql) + 2 high bits (qh), range [0..63] → subtract 32.
//!
//! ql layout: 128 bytes = 256 nibbles (2 per byte, low/high halves interleaved)
//! qh layout: 64 bytes = 256 × 2 bits (4 per byte)
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Scalar Q6_K × Q8_K dot product (reference implementation).
///
/// Direct port of llama.cpp `ggml_vec_dot_q6_K_q8_K_generic`.
pub fn vec_dot_q6k_q8k_scalar(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(x.len(), y.len());

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    for (xb, yb) in x.iter().zip(y.iter()) {
        aux32.fill(0);

        // Unpack 6-bit values: 4 low bits from ql, 2 high bits from qh
        let mut a_idx = 0;
        let mut ql_idx = 0;
        let mut qh_idx = 0;
        for _j in 0..(QK_K / 128) {
            for l in 0..32 {
                aux8[a_idx + l] =
                    ((xb.ql[ql_idx + l] & 0xF) | (((xb.qh[qh_idx + l] >> 0) & 3) << 4)) as i8 - 32;
                aux8[a_idx + l + 32] =
                    ((xb.ql[ql_idx + l + 32] & 0xF) | (((xb.qh[qh_idx + l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[a_idx + l + 64] =
                    ((xb.ql[ql_idx + l] >> 4) | (((xb.qh[qh_idx + l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[a_idx + l + 96] =
                    ((xb.ql[ql_idx + l + 32] >> 4) | (((xb.qh[qh_idx + l] >> 6) & 3) << 4)) as i8 - 32;
            }
            a_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
        }

        // Dot product with per-sub-block 8-bit scales
        let mut q8_off = 0;
        let mut a_off = 0;
        for is in 0..(QK_K / 16) {
            let scale = xb.scales[is] as i32;
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
    }

    sums.iter().sum()
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// Scale shuffle table for Q6_K: entry i broadcasts scale 2*i to first 8 bytes
    /// and scale 2*i+1 to last 8 bytes of a 128-bit register.
    /// 8 entries × 16 bytes = 128 bytes, matching llama.cpp's `get_scale_shuffle`.
    static SCALE_SHUFFLE_Q6K: [[u8; 16]; 8] = [
        [ 0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2,  3, 3, 3, 3, 3, 3, 3, 3],
        [ 4, 4, 4, 4, 4, 4, 4, 4,  5, 5, 5, 5, 5, 5, 5, 5],
        [ 6, 6, 6, 6, 6, 6, 6, 6,  7, 7, 7, 7, 7, 7, 7, 7],
        [ 8, 8, 8, 8, 8, 8, 8, 8,  9, 9, 9, 9, 9, 9, 9, 9],
        [10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11],
        [12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13],
        [14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15],
    ];

    #[inline(always)]
    unsafe fn get_scale_shuffle(i: usize) -> __m128i {
        unsafe { _mm_loadu_si128(SCALE_SHUFFLE_Q6K.get_unchecked(i).as_ptr() as *const __m128i) }
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

    /// AVX2 Q6_K × Q8_K dot product.
    ///
    /// Port of llama.cpp x86 AVX2 implementation.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q6k_q8k_avx2(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        unsafe {
            let m4 = _mm256_set1_epi8(0x0F);
            let m2 = _mm256_set1_epi8(3);
            let m32s = _mm256_set1_epi8(32);

            let mut acc = _mm256_setzero_ps();

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = yb.d * fp16_to_f32(xb.d);

                let mut q4 = xb.ql.as_ptr();
                let mut qh = xb.qh.as_ptr();
                let mut q8 = yb.qs.as_ptr();

                let scales = _mm_loadu_si128(xb.scales.as_ptr() as *const __m128i);

                let mut sumi = _mm256_setzero_si256();
                let mut is = 0usize;

                for _j in 0..(QK_K / 128) {
                    let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                    let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                    let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                    let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
                    is += 4;

                    let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
                    q4 = q4.add(32);
                    let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
                    q4 = q4.add(32);
                    let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
                    qh = qh.add(32);

                    // Combine low 4 bits with high 2 bits
                    let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                    let q4h_1 = _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2),
                        4,
                    );
                    let q4h_2 = _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2),
                        4,
                    );
                    let q4h_3 = _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2),
                        4,
                    );

                    let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    let q4_2 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4),
                        q4h_2,
                    );
                    let q4_3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4),
                        q4h_3,
                    );

                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);

                    // Subtract 32 bias: maddubs(32, q8) gives 32*sum(q8 pairs)
                    let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                    let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                    let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                    let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                    // Unsigned × signed dot product
                    let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                    let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                    let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                    let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                    // Subtract bias: (q6 - 32) * q8 = q6*q8 - 32*q8
                    p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                    p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                    p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                    p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                    // Multiply by scales (signed 8-bit → 16-bit) and accumulate to 32-bit
                    p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                    p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                    p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                    p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                }

                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
            }

            hsum_float_8(acc)
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Q6_K × Q8_K dot product, auto-dispatching to the best available kernel.
#[inline]
pub fn vec_dot_q6k_q8k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q6k_q8k_avx2(x, y) };
        }
    }
    vec_dot_q6k_q8k_scalar(x, y)
}

/// Q6_K matmul: y[M,N] = x[M,K] @ W[N,K]^T.
pub fn quantized_matmul_q6k(
    x: &[f32],
    w: &[BlockQ6K],
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
            out_row[j] = vec_dot_q6k_q8k(w_row, &x_q8);
        }
    });
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    fn make_test_q6k_blocks(values: &[f32]) -> Vec<BlockQ6K> {
        assert!(values.len() % QK_K == 0);
        let t = Tensor::from_vec(values.to_vec(), (values.len(),), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&t, GgmlDType::Q6K, &Device::Cpu).unwrap();
        let raw = qt.data().unwrap();
        bytemuck::cast_slice(&raw).to_vec()
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() + 0.5).collect();
        let q6 = make_test_q6k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q6k_q8k_scalar(&q6, &q8);
        assert!(result > 0.0, "self dot product should be positive, got {result}");
    }

    #[test]
    fn scalar_zeros() {
        let values = vec![0.0f32; QK_K];
        let q6 = make_test_q6k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q6k_q8k_scalar(&q6, &q8);
        assert!(result.abs() < 1e-6, "zero dot should be ~0, got {result}");
    }

    #[test]
    fn scalar_vs_candle_dequant() {
        let k = QK_K;
        let values: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q6 = make_test_q6k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);
        let our_dot = vec_dot_q6k_q8k_scalar(&q6, &q8);

        let t = Tensor::from_vec(values, (k,), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&t, GgmlDType::Q6K, &Device::Cpu).unwrap();
        let w_deq = qt.dequantize(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        let ref_dot: f32 = w_deq.iter().zip(x_vals.iter()).map(|(w, x)| w * x).sum();

        let rel_err = (our_dot - ref_dot).abs() / ref_dot.abs().max(1e-6);
        assert!(rel_err < 0.05, "scalar vs dequant: our={our_dot}, ref={ref_dot}, rel_err={rel_err}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let values: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q6 = make_test_q6k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);

        let scalar = vec_dot_q6k_q8k_scalar(&q6, &q8);
        let avx2 = unsafe { avx2::vec_dot_q6k_q8k_avx2(&q6, &q8) };

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
            w_blocks.extend(make_test_q6k_blocks(&w_data[j * k..(j + 1) * k]));
        }

        let mut out = vec![0.0f32; m * n];
        quantized_matmul_q6k(&x_data, &w_blocks, &mut out, m, n, k);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    }
}
