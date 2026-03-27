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
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// AVX2 Q4_K × Q8_K dot product.
    ///
    /// Follows llama.cpp / candle-core avx.rs `vec_dot_q4k_q8k` logic:
    /// - Unpack scales with KMASK bit manipulation
    /// - Process 4 groups of 64 elements (2 sub-blocks each)
    /// - maddubs for u8×i8 multiply, madd for scale application
    #[target_feature(enable = "avx2")]
    pub(super) fn vec_dot_q4k_q8k_avx2(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let m4 = _mm256_set1_epi8(0x0F);
        let mut sumf = 0.0f32;

        for (xb, yb) in x.iter().zip(y.iter()) {
            let d = fp16_to_f32(xb.d) * yb.d;
            let dmin = fp16_to_f32(xb.dmin) * yb.d;

            // Unpack scales/mins from 12-byte packed format
            let mut utmp = [0u32; 4];
            utmp[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
            utmp[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
            utmp[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            // scales in utmp[0..2] (8 bytes), mins in utmp[2..4] (8 bytes)
            let scales = bytemuck::cast::<[u32; 2], [u8; 8]>([utmp[0], utmp[1]]);
            let mins = bytemuck::cast::<[u32; 2], [u8; 8]>([utmp[2], utmp[3]]);

            // Min contribution via bsums
            let mut sum_mins: i32 = 0;
            for j in 0..QK_K / 16 {
                sum_mins += yb.bsums[j] as i32 * mins[j / 2] as i32;
            }

            // Main dot product: 4 iterations, each processes 64 elements (2 sub-blocks)
            let mut sumi = _mm256_setzero_si256();

            for i in 0..(QK_K / 64) {
                // Load 32 bytes of Q4 nibble pairs
                let q4bits = unsafe {
                    _mm256_loadu_si256(xb.qs.as_ptr().add(i * 32) as *const __m256i)
                };
                // Low nibbles → sub-block 2i, high nibbles → sub-block 2i+1
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                // Load 64 bytes of Q8 values (32 for each sub-block)
                let q8l = unsafe {
                    _mm256_loadu_si256(yb.qs.as_ptr().add(i * 64) as *const __m256i)
                };
                let q8h = unsafe {
                    _mm256_loadu_si256(yb.qs.as_ptr().add(i * 64 + 32) as *const __m256i)
                };

                // u8 × i8 → i16 pairs
                let p16l = _mm256_maddubs_epi16(q4l, q8l);
                let p16h = _mm256_maddubs_epi16(q4h, q8h);

                // Scale and accumulate: madd with scale broadcast as i16
                let sc0 = scales[2 * i] as i16;
                let sc1 = scales[2 * i + 1] as i16;
                let p32l = _mm256_madd_epi16(p16l, _mm256_set1_epi16(sc0));
                let p32h = _mm256_madd_epi16(p16h, _mm256_set1_epi16(sc1));

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p32l, p32h));
            }

            // Horizontal sum of sumi (i32 × 8)
            let hi128 = _mm256_extracti128_si256(sumi, 1);
            let lo128 = _mm256_castsi256_si128(sumi);
            let sum128 = _mm_add_epi32(hi128, lo128);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(sum128, hi64);
            let hi32 = _mm_shuffle_epi32(sum64, 0b_01_01_01_01);
            let sum32 = _mm_add_epi32(sum64, hi32);
            let sumi_scalar = _mm_cvtsi128_si32(sum32);

            sumf += d * sumi_scalar as f32 - dmin * sum_mins as f32;
        }

        sumf
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Q4_K × Q8_K dot product, auto-dispatching to the best available kernel.
pub fn vec_dot_q4k_q8k(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
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
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

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
