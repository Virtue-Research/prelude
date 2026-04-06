//! Activation quantization: FP32 → Q8_0 / Q8_K.
//!
//! During inference, weights are stored quantized; activations arrive as FP32/BF16.
//! We quantize activations on-the-fly:
//! - Q8_0 (32-element blocks) paired with Q4_0 weights
//! - Q8_K (256-element blocks with bsums) paired with Q4_K weights
//!
//! Scalar reference + AVX2 implementations.

use super::types::*;

// ── Scalar reference ─────────────────────────────────────────────────────

/// Quantize a row of FP32 values into Q8_0 blocks (scalar reference).
///
/// `x.len()` must be a multiple of 32. Output length = `x.len() / 32`.
pub fn quantize_row_q8_0_scalar(x: &[f32]) -> Vec<BlockQ8_0> {
    assert!(x.len() % QK8_0 == 0, "input length must be multiple of {QK8_0}");
    let nb = x.len() / QK8_0;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let block = &x[i * QK8_0..(i + 1) * QK8_0];

        // Find max absolute value
        let mut amax: f32 = 0.0;
        for &v in block {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        let d = amax / 127.0;
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };

        let mut qs = [0i8; QK8_0];
        for (j, &v) in block.iter().enumerate() {
            qs[j] = (v * id).round() as i8;
        }

        output.push(BlockQ8_0 {
            d: f32_to_fp16(d),
            qs,
        });
    }

    output
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// Quantize a row of FP32 values into Q8_0 blocks (AVX2).
    ///
    /// Processes 32 floats at a time using 4 × __m256 registers.
    #[target_feature(enable = "avx2")]
    pub(super) fn quantize_row_q8_0_avx2(x: &[f32]) -> Vec<BlockQ8_0> {
        let nb = x.len() / QK8_0;
        let mut output = Vec::with_capacity(nb);
        let mut ptr = x.as_ptr();

        let sign_bit = _mm256_set1_ps(-0.0f32);

        for _ in 0..nb {
            // SAFETY: ptr advances by 32 per iteration, total = nb * 32 = x.len()
            let (v0, v1, v2, v3) = unsafe {
                let v0 = _mm256_loadu_ps(ptr);
                let v1 = _mm256_loadu_ps(ptr.add(8));
                let v2 = _mm256_loadu_ps(ptr.add(16));
                let v3 = _mm256_loadu_ps(ptr.add(24));
                ptr = ptr.add(32);
                (v0, v1, v2, v3)
            };

            // max(abs(e)) for the block — all safe SIMD ops
            let mut max_abs = _mm256_andnot_ps(sign_bit, v0);
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));

            // Horizontal max across 8 lanes
            let hi128 = _mm256_extractf128_ps(max_abs, 1);
            let lo128 = _mm256_castps256_ps128(max_abs);
            let max4 = _mm_max_ps(hi128, lo128);
            let max2 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            let max1 = _mm_max_ss(max2, _mm_movehdup_ps(max2));
            let max_scalar = _mm_cvtss_f32(max1);

            let d = max_scalar / 127.0f32;
            let id = if max_scalar != 0.0 { 127.0f32 / max_scalar } else { 0.0f32 };
            let mul = _mm256_set1_ps(id);

            // Scale + round
            let v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // f32 → i32 → i16 → i8
            let i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            let i2 = _mm256_cvtps_epi32(v2);
            let i3 = _mm256_cvtps_epi32(v3);

            let packed16_01 = _mm256_packs_epi32(i0, i1);
            let packed16_23 = _mm256_packs_epi32(i2, i3);
            let packed8 = _mm256_packs_epi16(packed16_01, packed16_23);

            // Fix lane order after packs (AVX2 interleaves 128-bit lanes)
            let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let packed8 = _mm256_permutevar8x32_epi32(packed8, perm);

            let mut block = BlockQ8_0 {
                d: f32_to_fp16(d),
                qs: [0i8; 32],
            };
            // SAFETY: block.qs is a [i8; 32], 32 bytes writable
            unsafe { _mm256_storeu_si256(block.qs.as_mut_ptr() as *mut __m256i, packed8) };
            output.push(block);
        }

        output
    }

    /// Quantize a row of FP32 values into Q8_K blocks (AVX2).
    ///
    /// Processes 256 floats per block. Each block uses f32 scale and
    /// precomputes 16-element sub-block sums (bsums).
    #[target_feature(enable = "avx2")]
    pub(super) fn quantize_row_q8k_avx2(x: &[f32]) -> Vec<BlockQ8K> {
        let nb = x.len() / QK_K;
        let mut output = Vec::with_capacity(nb);

        let sign_bit = _mm256_set1_ps(-0.0f32);

        for i in 0..nb {
            let block_start = i * QK_K;
            let ptr = unsafe { x.as_ptr().add(block_start) };

            // Find max absolute value across all 256 elements (8 groups of 32)
            let mut max_abs = _mm256_setzero_ps();
            for g in 0..8 {
                // SAFETY: block_start + g*32 + 24 < block_start + 256 <= x.len()
                let v0 = unsafe { _mm256_loadu_ps(ptr.add(g * 32)) };
                let v1 = unsafe { _mm256_loadu_ps(ptr.add(g * 32 + 8)) };
                let v2 = unsafe { _mm256_loadu_ps(ptr.add(g * 32 + 16)) };
                let v3 = unsafe { _mm256_loadu_ps(ptr.add(g * 32 + 24)) };
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v0));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));
            }

            // Horizontal max across 8 lanes
            let hi128 = _mm256_extractf128_ps(max_abs, 1);
            let lo128 = _mm256_castps256_ps128(max_abs);
            let max4 = _mm_max_ps(hi128, lo128);
            let max2 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            let max1 = _mm_max_ss(max2, _mm_movehdup_ps(max2));
            let amax = _mm_cvtss_f32(max1);

            let d = amax / 127.0f32;
            let id = if amax != 0.0 { 127.0f32 / amax } else { 0.0f32 };
            let mul = _mm256_set1_ps(id);

            let mut qs = [0i8; QK_K];
            let mut bsums = [0i16; QK_K / 16];

            // Process 256 elements in groups of 32 (matching Q8_0 pattern),
            // quantize to i8 and compute bsums
            let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

            for g in 0..8 {
                let base = g * 32;
                // SAFETY: ptr.add(base + 24) < ptr + 256
                let v0 = unsafe { _mm256_loadu_ps(ptr.add(base)) };
                let v1 = unsafe { _mm256_loadu_ps(ptr.add(base + 8)) };
                let v2 = unsafe { _mm256_loadu_ps(ptr.add(base + 16)) };
                let v3 = unsafe { _mm256_loadu_ps(ptr.add(base + 24)) };

                // Scale + round
                let r0 = _mm256_round_ps(
                    _mm256_mul_ps(v0, mul),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                );
                let r1 = _mm256_round_ps(
                    _mm256_mul_ps(v1, mul),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                );
                let r2 = _mm256_round_ps(
                    _mm256_mul_ps(v2, mul),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                );
                let r3 = _mm256_round_ps(
                    _mm256_mul_ps(v3, mul),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                );

                // f32 → i32
                let i0 = _mm256_cvtps_epi32(r0);
                let i1 = _mm256_cvtps_epi32(r1);
                let i2 = _mm256_cvtps_epi32(r2);
                let i3 = _mm256_cvtps_epi32(r3);

                // i32 → i16 → i8 (same pack+permute as Q8_0)
                let packed16_01 = _mm256_packs_epi32(i0, i1);
                let packed16_23 = _mm256_packs_epi32(i2, i3);
                let packed8 = _mm256_packs_epi16(packed16_01, packed16_23);
                let packed8 = _mm256_permutevar8x32_epi32(packed8, perm);

                // Store 32 quantized bytes
                // SAFETY: qs[base..base+32] is within [0..256]
                unsafe {
                    _mm256_storeu_si256(qs.as_mut_ptr().add(base) as *mut __m256i, packed8);
                }

                // Compute bsums for the two 16-element subblocks in this group.
                // First subblock: elements [base..base+16] from i0+i1
                // Second subblock: elements [base+16..base+32] from i2+i3
                //
                // For each pair, add the two __m256i vectors, then reduce the
                // resulting 8 × i32 to a single sum.
                let s01 = _mm256_add_epi32(i0, i1); // 8 partial sums
                let s01_hi = _mm256_extracti128_si256(s01, 1);
                let s01_lo = _mm256_castsi256_si128(s01);
                let s01_4 = _mm_add_epi32(s01_lo, s01_hi); // 4 sums
                let s01_2 = _mm_add_epi32(s01_4, _mm_srli_si128(s01_4, 8)); // 2 sums
                let s01_1 = _mm_add_epi32(s01_2, _mm_srli_si128(s01_2, 4)); // 1 sum
                bsums[g * 2] = _mm_cvtsi128_si32(s01_1) as i16;

                let s23 = _mm256_add_epi32(i2, i3);
                let s23_hi = _mm256_extracti128_si256(s23, 1);
                let s23_lo = _mm256_castsi256_si128(s23);
                let s23_4 = _mm_add_epi32(s23_lo, s23_hi);
                let s23_2 = _mm_add_epi32(s23_4, _mm_srli_si128(s23_4, 8));
                let s23_1 = _mm_add_epi32(s23_2, _mm_srli_si128(s23_2, 4));
                bsums[g * 2 + 1] = _mm_cvtsi128_si32(s23_1) as i16;
            }

            output.push(BlockQ8K { d, qs, bsums });
        }

        output
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Quantize FP32 activations to Q8_0, selecting the best kernel at runtime.
///
/// `x.len()` must be a multiple of 32.
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<BlockQ8_0> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guarantees AVX2 is available
            return unsafe { avx2::quantize_row_q8_0_avx2(x) };
        }
    }
    quantize_row_q8_0_scalar(x)
}

// ══════════════════════════════════════════════════════════════════════════
// Q8_K quantization (256-element blocks, paired with Q4_K weights)
// ══════════════════════════════════════════════════════════════════════════

/// Quantize a row of FP32 values into Q8_K blocks (scalar).
///
/// `x.len()` must be a multiple of 256. Output length = `x.len() / 256`.
/// Q8_K uses f32 scale and pre-computes 16-element sub-block sums (`bsums`).
pub fn quantize_row_q8k_scalar(x: &[f32]) -> Vec<BlockQ8K> {
    assert!(x.len() % QK_K == 0, "input length must be multiple of {QK_K}");
    let nb = x.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let block = &x[i * QK_K..(i + 1) * QK_K];

        // Find max absolute value
        let mut amax: f32 = 0.0;
        for &v in block {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        let d = amax / 127.0;
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };

        let mut qs = [0i8; QK_K];
        for (j, &v) in block.iter().enumerate() {
            qs[j] = (v * id).round().clamp(-128.0, 127.0) as i8;
        }

        // Compute sub-block sums (16 sums, each covering 16 elements)
        let mut bsums = [0i16; QK_K / 16];
        for (s, sum) in bsums.iter_mut().enumerate() {
            let mut acc: i16 = 0;
            for j in 0..16 {
                acc += qs[s * 16 + j] as i16;
            }
            *sum = acc;
        }

        output.push(BlockQ8K { d, qs, bsums });
    }

    output
}

/// Quantize FP32 activations to Q8_K, selecting the best kernel at runtime.
pub fn quantize_row_q8k(x: &[f32]) -> Vec<BlockQ8K> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guarantees AVX2 is available
            return unsafe { avx2::quantize_row_q8k_avx2(x) };
        }
    }
    quantize_row_q8k_scalar(x)
}

// ══════════════════════════════════════════════════════════════════════════
// Q8_1 quantization (32-element blocks, paired with Q4_1/Q5_1 weights)
// ══════════════════════════════════════════════════════════════════════════

/// Quantize a row of FP32 values into Q8_1 blocks (scalar).
///
/// Like Q8_0 but also precomputes `s = d * sum(qs)` for asymmetric dot products.
pub fn quantize_row_q8_1_scalar(x: &[f32]) -> Vec<BlockQ8_1> {
    assert!(x.len() % QK8_0 == 0, "input length must be multiple of {QK8_0}");
    let nb = x.len() / QK8_0;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let block = &x[i * QK8_0..(i + 1) * QK8_0];

        let mut amax: f32 = 0.0;
        for &v in block {
            let av = v.abs();
            if av > amax { amax = av; }
        }

        let d = amax / 127.0;
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };

        let mut qs = [0i8; QK8_0];
        let mut sum: f32 = 0.0;
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round();
            qs[j] = q as i8;
            sum += q;
        }

        output.push(BlockQ8_1 {
            d: f32_to_fp16(d),
            s: f32_to_fp16(d * sum),
            qs,
        });
    }
    output
}

/// Quantize FP32 activations to Q8_1, selecting the best kernel at runtime.
pub fn quantize_row_q8_1(x: &[f32]) -> Vec<BlockQ8_1> {
    quantize_row_q8_1_scalar(x)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_basic() {
        // 32 values = 1 block
        let input: Vec<f32> = (0..32).map(|i| i as f32 - 15.5).collect();
        let blocks = quantize_row_q8_0_scalar(&input);
        assert_eq!(blocks.len(), 1);

        // Verify round-trip: dequantize and compare
        let d = fp16_to_f32(blocks[0].d);
        for (j, &orig) in input.iter().enumerate() {
            let reconstructed = blocks[0].qs[j] as f32 * d;
            let err = (reconstructed - orig).abs();
            // Q8_0 with 8-bit precision: max error ≈ d/2 ≈ amax/(2*127)
            assert!(
                err < d + 0.01,
                "j={j}: orig={orig}, recon={reconstructed}, err={err}, d={d}"
            );
        }
    }

    #[test]
    fn scalar_all_zeros() {
        let input = vec![0.0f32; 64]; // 2 blocks
        let blocks = quantize_row_q8_0_scalar(&input);
        assert_eq!(blocks.len(), 2);
        for b in &blocks {
            assert_eq!(fp16_to_f32(b.d), 0.0);
            assert!(b.qs.iter().all(|&q| q == 0));
        }
    }

    #[test]
    fn scalar_symmetric() {
        // Symmetric values: [-1, 1, -1, 1, ...]
        let input: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let blocks = quantize_row_q8_0_scalar(&input);
        let d = fp16_to_f32(blocks[0].d);
        // Scale should be 1.0/127 ≈ 0.00787
        assert!((d - 1.0 / 127.0).abs() < 0.001, "d={d}");
        // Values should be ±127
        for (j, &q) in blocks[0].qs.iter().enumerate() {
            let expected = if j % 2 == 0 { -127 } else { 127 };
            assert_eq!(q, expected, "j={j}");
        }
    }

    #[test]
    fn roundtrip_precision() {
        // Generate varied data and check round-trip error
        let mut input = vec![0.0f32; 128]; // 4 blocks
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i as f32) * 0.1).sin() * 10.0;
        }
        let blocks = quantize_row_q8_0_scalar(&input);
        assert_eq!(blocks.len(), 4);

        let mut max_abs_err: f32 = 0.0;
        for (bi, block) in blocks.iter().enumerate() {
            let d = fp16_to_f32(block.d);
            for j in 0..32 {
                let orig = input[bi * 32 + j];
                let recon = block.qs[j] as f32 * d;
                let abs_err = (recon - orig).abs();
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                }
            }
        }
        // Q8_0: max abs error per block ≤ d (scale), typically d ≈ max/127.
        // For sin(x)*10, max ≈ 10, so d ≈ 0.079, max abs error ≈ 0.04.
        assert!(
            max_abs_err < 0.1,
            "max absolute error too high: {max_abs_err}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    mod simd_tests {
        use super::*;

        #[test]
        fn avx2_matches_scalar() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            // Generate test data with varied magnitudes
            let mut input = vec![0.0f32; 256]; // 8 blocks
            for (i, v) in input.iter_mut().enumerate() {
                *v = ((i as f32) * 0.37).sin() * ((i as f32) * 0.13).cos() * 50.0;
            }

            let scalar_blocks = quantize_row_q8_0_scalar(&input);
            // SAFETY: AVX2 checked above
            let avx2_blocks = unsafe { avx2::quantize_row_q8_0_avx2(&input) };

            assert_eq!(scalar_blocks.len(), avx2_blocks.len());

            for (bi, (sb, ab)) in scalar_blocks.iter().zip(avx2_blocks.iter()).enumerate() {
                // Scales should match exactly (both compute the same max + FP16 conversion)
                assert_eq!(
                    sb.d, ab.d,
                    "block {bi}: scale mismatch scalar={} avx2={}",
                    fp16_to_f32(sb.d),
                    fp16_to_f32(ab.d)
                );

                // Quantized values should match exactly
                for j in 0..32 {
                    assert_eq!(
                        sb.qs[j], ab.qs[j],
                        "block {bi} elem {j}: scalar={}, avx2={}",
                        sb.qs[j], ab.qs[j]
                    );
                }
            }
        }

        #[test]
        fn avx2_all_zeros() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
            let input = vec![0.0f32; 64];
            // SAFETY: AVX2 checked above
            let blocks = unsafe { avx2::quantize_row_q8_0_avx2(&input) };
            assert_eq!(blocks.len(), 2);
            for b in &blocks {
                assert_eq!(fp16_to_f32(b.d), 0.0);
                assert!(b.qs.iter().all(|&q| q == 0));
            }
        }

        #[test]
        fn dispatch_matches_scalar() {
            let mut input = vec![0.0f32; 128];
            for (i, v) in input.iter_mut().enumerate() {
                *v = (i as f32 - 64.0) * 0.5;
            }
            let scalar = quantize_row_q8_0_scalar(&input);
            let dispatched = quantize_row_q8_0(&input);

            for (bi, (sb, db)) in scalar.iter().zip(dispatched.iter()).enumerate() {
                assert_eq!(sb.d, db.d, "block {bi} scale mismatch");
                assert_eq!(sb.qs, db.qs, "block {bi} values mismatch");
            }
        }

        #[test]
        fn avx2_q8k_matches_scalar() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            // Generate test data: 2 blocks = 512 floats with varied magnitudes
            let mut input = vec![0.0f32; 512];
            for (i, v) in input.iter_mut().enumerate() {
                *v = ((i as f32) * 0.37).sin() * ((i as f32) * 0.13).cos() * 50.0;
            }

            let scalar_blocks = quantize_row_q8k_scalar(&input);
            // SAFETY: AVX2 checked above
            let avx2_blocks = unsafe { avx2::quantize_row_q8k_avx2(&input) };

            assert_eq!(scalar_blocks.len(), avx2_blocks.len());

            for (bi, (sb, ab)) in scalar_blocks.iter().zip(avx2_blocks.iter()).enumerate() {
                assert!(
                    (sb.d - ab.d).abs() < 1e-6,
                    "block {bi}: scale mismatch scalar={} avx2={}",
                    sb.d, ab.d,
                );

                for j in 0..QK_K {
                    assert_eq!(
                        sb.qs[j], ab.qs[j],
                        "block {bi} elem {j}: scalar={}, avx2={}",
                        sb.qs[j], ab.qs[j]
                    );
                }

                for j in 0..(QK_K / 16) {
                    assert_eq!(
                        sb.bsums[j], ab.bsums[j],
                        "block {bi} bsum {j}: scalar={}, avx2={}",
                        sb.bsums[j], ab.bsums[j]
                    );
                }
            }
        }

        #[test]
        fn avx2_q8k_all_zeros() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
            let input = vec![0.0f32; 256];
            // SAFETY: AVX2 checked above
            let blocks = unsafe { avx2::quantize_row_q8k_avx2(&input) };
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].d, 0.0);
            assert!(blocks[0].qs.iter().all(|&q| q == 0));
            assert!(blocks[0].bsums.iter().all(|&s| s == 0));
        }

        #[test]
        fn q8k_dispatch_matches_scalar() {
            let mut input = vec![0.0f32; 512];
            for (i, v) in input.iter_mut().enumerate() {
                *v = (i as f32 - 256.0) * 0.3;
            }
            let scalar = quantize_row_q8k_scalar(&input);
            let dispatched = quantize_row_q8k(&input);

            for (bi, (sb, db)) in scalar.iter().zip(dispatched.iter()).enumerate() {
                assert!((sb.d - db.d).abs() < 1e-6, "block {bi} scale mismatch");
                assert_eq!(sb.qs, db.qs, "block {bi} values mismatch");
                assert_eq!(sb.bsums, db.bsums, "block {bi} bsums mismatch");
            }
        }
    }
}
