//! Q4_K × Q8_K dot product kernel.
//!
//! Q4_K is a K-quant format: 256 values per block, split into 8 sub-blocks of 32,
//! each with its own 6-bit scale and minimum. Higher quality than Q4_0 at the same
//! 4.5 bpw compression ratio.
//!
//! qs layout (from llama.cpp `from_float`):
//!   For each group of 64 consecutive values (j..j+64):
//!     qs[(j/64)*32 + k] = l[j+k] | (l[j+k+32] << 4)   for k in 0..32
//!   i.e. low nibble = sub-block 2i, high nibble = sub-block 2i+1
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Unpack 6-bit scales and mins from the 12-byte packed format.
///
/// Returns (scales[8], mins[8]) matching llama.cpp's bit manipulation.
fn unpack_scales_mins(raw_scales: &[u8; K_SCALE_SIZE]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp = [0u32; 4];
    // Read first 12 bytes as 3 little-endian u32
    utmp[0] = u32::from_le_bytes([raw_scales[0], raw_scales[1], raw_scales[2], raw_scales[3]]);
    utmp[1] = u32::from_le_bytes([raw_scales[4], raw_scales[5], raw_scales[6], raw_scales[7]]);
    utmp[2] = u32::from_le_bytes([raw_scales[8], raw_scales[9], raw_scales[10], raw_scales[11]]);

    // Unpack — exactly matches candle-core/k_quants.rs vec_dot_unopt
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // utmp[0..2] → scales[0..8], utmp[2..4] → mins[0..8]
    let scales: [u8; 8] = bytemuck::cast([utmp[0], utmp[1]]);
    let mins: [u8; 8] = bytemuck::cast([utmp[2], utmp[3]]);
    (scales, mins)
}

/// Scalar Q4_K × Q8_K dot product (reference implementation).
///
/// Follows candle-core's `vec_dot_unopt` which is a direct port of llama.cpp.
pub fn vec_dot_q4k_q8k_scalar(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(x.len(), y.len());
    let mut sumf: f32 = 0.0;

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = fp16_to_f32(xb.d) * yb.d;
        let dmin = fp16_to_f32(xb.dmin) * yb.d;

        let (scales, mins) = unpack_scales_mins(&xb.scales);

        // Min contribution via bsums: sum(mins[j/2] * bsums[j]) for j in 0..16
        let mut sum_mins: i32 = 0;
        for j in 0..QK_K / 16 {
            sum_mins += yb.bsums[j] as i32 * mins[j / 2] as i32;
        }

        // Main dot product: 4 groups of 64 elements, each group has 2 sub-blocks
        // qs layout: low nibble = sub-block 2i (32 values), high nibble = sub-block 2i+1 (32 values)
        let mut sumi: i32 = 0;
        for i in 0..(QK_K / 64) {
            let sc0 = scales[2 * i] as i32;
            let sc1 = scales[2 * i + 1] as i32;

            let qs_off = i * 32;
            let q8_off = i * 64;

            let mut sum0: i32 = 0;
            let mut sum1: i32 = 0;
            for k in 0..32 {
                let q4_lo = (xb.qs[qs_off + k] & 0xF) as i32;
                let q4_hi = (xb.qs[qs_off + k] >> 4) as i32;
                sum0 += q4_lo * yb.qs[q8_off + k] as i32;
                sum1 += q4_hi * yb.qs[q8_off + 32 + k] as i32;
            }
            sumi += sc0 * sum0 + sc1 * sum1;
        }

        sumf += d * sumi as f32 - dmin * sum_mins as f32;
    }

    sumf
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    // Scale shuffle lookup table: broadcasts scale pairs across 16-byte lanes.
    // Entry i broadcasts bytes (2*i, 2*i+1) to all positions.
    pub(in super::super) static SCALE_SHUFFLE_K4: [[u8; 32]; 8] = [
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

    /// AVX2 Q4_K × Q8_K dot product — optimized to match ggml.
    ///
    /// Key optimizations vs previous version:
    /// - SIMD mins computation (hadd + madd instead of scalar loop)
    /// - Cross-block float accumulation (one hsum at the end)
    /// - Shuffle-based scale broadcasting (lookup table instead of set1)
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q4k_q8k_avx2(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        unsafe {
            let m4 = _mm256_set1_epi8(0x0F);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            let mut utmp = [0u32; 4];

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = yb.d * fp16_to_f32(xb.d);
                let dmin = -yb.d * fp16_to_f32(xb.dmin);

                // Unpack scales/mins from 12-byte packed format
                utmp[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
                utmp[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
                utmp[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

                utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
                let uaux = utmp[1] & KMASK1;
                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[2] = uaux;
                utmp[0] &= KMASK1;

                // Load scales+mins as 16-bit: lower 128 = scales, upper 128 = mins
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32,
                ));

                // SIMD mins contribution: hadd bsums pairs, madd with mins, FMA accumulate
                let q8sums = _mm256_loadu_si256(yb.bsums.as_ptr() as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                // Extract scales for shuffle-based broadcasting
                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_set_m128i(sc128, sc128);

                let mut sumi = _mm256_setzero_si256();
                let q4 = xb.qs.as_ptr();
                let q8 = yb.qs.as_ptr();

                for j in 0..(QK_K / 64) {
                    // Broadcast scales via shuffle table
                    let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q4bits = _mm256_loadu_si256(q4.add(j * 32) as *const __m256i);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                    let q8l = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
                    let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
                    p16l = _mm256_madd_epi16(scale_l, p16l);

                    let q8h = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);
                    let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
                    p16h = _mm256_madd_epi16(scale_h, p16h);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }

                // Accumulate across blocks in float (no per-block hsum)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }

            // Final horizontal sum
            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

            hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
        }
    }
}

// ── AVX-512BW implementation ─────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum_float_16(x: __m512) -> f32 {
        unsafe {
            // Reduce 512 → 256
            let hi256 = _mm512_extractf32x8_ps(x, 1);
            let lo256 = _mm512_castps512_ps256(x);
            let sum256 = _mm256_add_ps(hi256, lo256);
            // Reduce 256 → scalar
            let hi128 = _mm256_extractf128_ps(sum256, 1);
            let lo128 = _mm256_castps256_ps128(sum256);
            let sum128 = _mm_add_ps(hi128, lo128);
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_movehdup_ps(sum64);
            _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
        }
    }

    /// AVX-512BW Q4_K × Q8_K dot product.
    ///
    /// Processes 128 elements per inner iteration (vs 64 in AVX2).
    /// Two 64-element groups are merged into a single 512-bit pass.
    #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
    pub(super) unsafe fn vec_dot_q4k_q8k_avx512(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        unsafe {
            let m4 = _mm512_set1_epi8(0x0F);
            let mut acc = _mm512_setzero_ps();
            let mut acc_m = _mm_setzero_ps();
            let mut utmp = [0u32; 4];

            for (xb, yb) in x.iter().zip(y.iter()) {
                let d = yb.d * fp16_to_f32(xb.d);
                let dmin = -yb.d * fp16_to_f32(xb.dmin);

                // Unpack scales/mins (same as AVX2)
                utmp[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
                utmp[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
                utmp[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

                utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
                let uaux = utmp[1] & KMASK1;
                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[2] = uaux;
                utmp[0] &= KMASK1;

                // Min contribution (same as AVX2 — uses 256-bit)
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32,
                ));
                let q8sums = _mm256_loadu_si256(yb.bsums.as_ptr() as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                // Scales: expand 8 × u8 scales to 8 × i16, then broadcast for 512-bit ops
                // We need 4 scale pairs for 4 inner groups. With 512-bit, we process 2 groups at once,
                // so we need 2 pairs per 512-bit iteration.
                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                // scales128 has 8 × i16 scales: [s0, s1, s2, s3, s4, s5, s6, s7]

                let mut sumi = _mm512_setzero_si512();
                let q4 = xb.qs.as_ptr();
                let q8 = yb.qs.as_ptr();

                // Same 4-iteration structure as AVX2, but concatenate [q4l|q4h] and [q8l|q8h]
                // into 512-bit registers. Each iteration processes 64 elements (32 low + 32 high).
                // Scale vector: [scale_l×16 | scale_h×16] as i16.
                let scales256 = _mm256_set_m128i(sc128, sc128);

                for j in 0..(QK_K / 64) {
                    let q4bits = _mm256_loadu_si256(q4.add(j * 32) as *const __m256i);
                    let q4l = _mm256_and_si256(q4bits, _mm256_set1_epi8(0x0F));
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), _mm256_set1_epi8(0x0F));

                    // Concat [q4l | q4h] into 512-bit
                    let q4_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4l), q4h, 1);

                    let q8l = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
                    let q8h = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);
                    let q8_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q8l), q8h, 1);

                    // Build scale: [scale_l broadcast × 16 | scale_h broadcast × 16]
                    let shuf_l = super::avx2::SCALE_SHUFFLE_K4[2 * j];
                    let shuf_h = super::avx2::SCALE_SHUFFLE_K4[2 * j + 1];
                    let scale_l = _mm256_shuffle_epi8(
                        scales256,
                        _mm256_loadu_si256(shuf_l.as_ptr() as *const __m256i),
                    );
                    let scale_h = _mm256_shuffle_epi8(
                        scales256,
                        _mm256_loadu_si256(shuf_h.as_ptr() as *const __m256i),
                    );
                    let scale_512 = _mm512_inserti64x4(_mm512_castsi256_si512(scale_l), scale_h, 1);

                    let mut p16 = _mm512_maddubs_epi16(q4_512, q8_512);
                    p16 = _mm512_madd_epi16(scale_512, p16);
                    sumi = _mm512_add_epi32(sumi, p16);
                }

                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }

            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

            hsum_float_16(acc) + _mm_cvtss_f32(acc_m)
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Q4_K × Q8_K dot product, auto-dispatching to the best available kernel.
#[inline]
pub fn vec_dot_q4k_q8k(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { avx512::vec_dot_q4k_q8k_avx512(x, y) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q4k_q8k_avx2(x, y) };
        }
    }
    vec_dot_q4k_q8k_scalar(x, y)
}

/// Q4_K matmul: y[M,N] = x[M,K] @ W[N,K]^T.
///
/// Same structure as `quantized_matmul_f32` but for Q4_K weights and Q8_K activations.
pub fn quantized_matmul_q4k(
    x: &[f32],
    w: &[BlockQ4K],
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
            out_row[j] = vec_dot_q4k_q8k(w_row, &x_q8);
        }
    });
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QTensor};
    use crate::tensor::{Device, Tensor};

    /// Use candle-core's own Q4_K quantization as ground truth for test data.
    /// This ensures our qs/scales layout matches exactly.
    fn make_test_q4k_blocks(values: &[f32]) -> Vec<BlockQ4K> {
        assert!(values.len() % QK_K == 0);
        let t = Tensor::from_vec(values.to_vec(), (values.len(),), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&t, GgmlDType::Q4K, &Device::Cpu).unwrap();
        let raw = qt.data().unwrap();
        bytemuck::cast_slice(&raw).to_vec()
    }

    #[test]
    fn scalar_self_dot_positive() {
        let values: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() + 0.5).collect();
        let q4 = make_test_q4k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q4k_q8k_scalar(&q4, &q8);
        assert!(result > 0.0, "self dot product should be positive, got {result}");
    }

    #[test]
    fn scalar_zeros() {
        let values = vec![0.0f32; QK_K];
        let q4 = make_test_q4k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&values);
        let result = vec_dot_q4k_q8k_scalar(&q4, &q8);
        assert!(result.abs() < 1e-6, "zero dot should be ~0, got {result}");
    }

    #[test]
    fn scalar_vs_candle_dequant() {
        // Compare our Q4_K kernel against dequant→F32→matmul reference
        let k = QK_K;
        let values: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q4 = make_test_q4k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);
        let our_dot = vec_dot_q4k_q8k_scalar(&q4, &q8);

        // Dequant reference
        let t = Tensor::from_vec(values.clone(), (k,), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&t, GgmlDType::Q4K, &Device::Cpu).unwrap();
        let w_deq = qt.dequantize(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        let ref_dot: f32 = w_deq.iter().zip(x_vals.iter()).map(|(w, x)| w * x).sum();

        let rel_err = (our_dot - ref_dot).abs() / ref_dot.abs().max(1e-6);
        assert!(
            rel_err < 0.05,
            "scalar vs dequant: our={our_dot}, ref={ref_dot}, rel_err={rel_err}"
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
        let x_vals: Vec<f32> = (0..QK_K * 4)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();

        let q4 = make_test_q4k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);

        let scalar = vec_dot_q4k_q8k_scalar(&q4, &q8);
        let avx2 = unsafe { avx2::vec_dot_q4k_q8k_avx2(&q4, &q8) };

        let rel_err = (scalar - avx2).abs() / scalar.abs().max(1e-6);
        assert!(
            rel_err < 1e-5,
            "AVX2 vs scalar: scalar={scalar}, avx2={avx2}, rel_err={rel_err}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let values: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..QK_K * 4).map(|i| ((i as f32) * 0.013).cos()).collect();

        let q4 = make_test_q4k_blocks(&values);
        let q8 = super::super::quantize::quantize_row_q8k_scalar(&x_vals);

        let scalar = vec_dot_q4k_q8k_scalar(&q4, &q8);
        let avx512 = unsafe { avx512::vec_dot_q4k_q8k_avx512(&q4, &q8) };

        let rel_err = (scalar - avx512).abs() / scalar.abs().max(1e-6);
        assert!(
            rel_err < 1e-5,
            "AVX512 vs scalar: scalar={scalar}, avx512={avx512}, rel_err={rel_err}"
        );
    }

    #[test]
    fn matmul_basic() {
        let k = QK_K;
        let n = 4;
        let m = 2;

        let w_data: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let x_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.011).cos()).collect();

        // Quantize weights
        let mut w_blocks = Vec::new();
        for j in 0..n {
            w_blocks.extend(make_test_q4k_blocks(&w_data[j * k..(j + 1) * k]));
        }

        let mut out = vec![0.0f32; m * n];
        quantized_matmul_q4k(&x_data, &w_blocks, &mut out, m, n, k);

        // All outputs should be finite
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    }
}
