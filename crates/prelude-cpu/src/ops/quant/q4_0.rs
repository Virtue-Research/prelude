//! Q4_0 × Q8_0 dot-product kernel.
//!
//! Scalar reference + AVX2 + auto-dispatch.
//! All SIMD variants are tested against the scalar reference for precision.

use super::types::*;

// ── Scalar reference (always compiled, used as ground truth) ─────────────

/// Q4_0 · Q8_0 dot product — scalar reference implementation.
///
/// Each Q4_0 block has 32 values packed as 16 nibble pairs in `[0..15]`,
/// which represent signed values `[-8..+7]` after subtracting 8.
pub fn vec_dot_q4_0_q8_0_scalar(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len(), y.len());

    let mut sumf: f32 = 0.0;

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = fp16_to_f32(xb.d) * fp16_to_f32(yb.d);

        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..(QK8_0 / 2) {
            // Low nibble: bits [3:0]
            let v0 = (xb.qs[j] & 0x0F) as i32 - 8;
            // High nibble: bits [7:4]
            let v1 = (xb.qs[j] >> 4) as i32 - 8;

            sumi0 += v0 * (yb.qs[j] as i32);
            sumi1 += v1 * (yb.qs[j + QK8_0 / 2] as i32);
        }

        sumf += (sumi0 + sumi1) as f32 * d;
    }

    sumf
}

// ── AVX2 implementation ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    // Helpers: NO #[target_feature] → CAN be #[inline(always)].
    // Only called from #[target_feature] functions, LLVM inlines with correct ISA.

    /// Hardware FP16→F32 using F16C (vcvtph2ps). All AVX2 CPUs have F16C.
    #[inline(always)]
    unsafe fn fp16_to_f32_hw(h: u16) -> f32 {
        unsafe {
            let v = _mm_cvtsi32_si128(h as i32);
            _mm_cvtss_f32(_mm_cvtph_ps(v))
        }
    }

    #[inline(always)]
    unsafe fn bytes_from_nibbles_32(ptr: *const u8) -> __m256i {
        unsafe {
            let raw = _mm_loadu_si128(ptr as *const __m128i);
            let hi = _mm_srli_epi16(raw, 4);
            let combined = _mm256_set_m128i(hi, raw);
            _mm256_and_si256(combined, _mm256_set1_epi8(0x0F))
        }
    }

    #[inline(always)]
    unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
        unsafe {
            let ax = _mm256_sign_epi8(x, x);
            let sy = _mm256_sign_epi8(y, x);
            let dot = _mm256_maddubs_epi16(ax, sy);
            let sum32 = _mm256_madd_epi16(dot, _mm256_set1_epi16(1));
            _mm256_cvtepi32_ps(sum32)
        }
    }

    #[inline(always)]
    unsafe fn mul_sum_us8_pairs_float_vnni(ax: __m256i, sy: __m256i) -> __m256 {
        unsafe {
            let sum32 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy);
            _mm256_cvtepi32_ps(sum32)
        }
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

    /// AVX2 Q4_0 · Q8_0 dot product.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn vec_dot_q4_0_q8_0_avx2(
        x: &[BlockQ4_0],
        y: &[BlockQ8_0],
    ) -> f32 {
        unsafe {
            let nb = x.len();
            let mut acc = _mm256_setzero_ps();
            let off = _mm256_set1_epi8(8);

            for ib in 0..nb {
                let xb = x.get_unchecked(ib);
                let yb = y.get_unchecked(ib);

                let d = _mm256_set1_ps(fp16_to_f32_hw(xb.d) * fp16_to_f32_hw(yb.d));
                let qx = _mm256_sub_epi8(bytes_from_nibbles_32(xb.qs.as_ptr()), off);
                let qy = _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i);
                acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc);
            }

            hsum_float_8(acc)
        }
    }

    /// AVX-VNNI Q4_0 · Q8_0 dot product — dpbusd replaces maddubs+madd.
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub(super) unsafe fn vec_dot_q4_0_q8_0_vnni(
        x: &[BlockQ4_0],
        y: &[BlockQ8_0],
    ) -> f32 {
        unsafe {
            let nb = x.len();
            let mut acc = _mm256_setzero_ps();
            let off = _mm256_set1_epi8(8);

            for ib in 0..nb {
                let xb = x.get_unchecked(ib);
                let yb = y.get_unchecked(ib);

                let d = _mm256_set1_ps(fp16_to_f32_hw(xb.d) * fp16_to_f32_hw(yb.d));
                let qx = _mm256_sub_epi8(bytes_from_nibbles_32(xb.qs.as_ptr()), off);
                let qy = _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i);

                let ax = _mm256_sign_epi8(qx, qx);
                let sy = _mm256_sign_epi8(qy, qx);
                acc = _mm256_fmadd_ps(d, mul_sum_us8_pairs_float_vnni(ax, sy), acc);
            }

            hsum_float_8(acc)
        }
    }
}

// ── Auto-dispatch ────────────────────────────────────────────────────────

/// Compute Q4_0 · Q8_0 dot product, automatically selecting the best kernel.
#[inline]
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if is_x86_feature_detected!("avxvnni") {
                return unsafe { avx2::vec_dot_q4_0_q8_0_vnni(x, y) };
            }
            return unsafe { avx2::vec_dot_q4_0_q8_0_avx2(x, y) };
        }
    }
    vec_dot_q4_0_q8_0_scalar(x, y)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a Q4_0 block with a given scale and nibble values.
    fn make_q4_0(scale: f32, values: &[i8; 32]) -> BlockQ4_0 {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            // Low nibble: values[j], High nibble: values[j+16]
            // Both shifted from [-8..+7] to [0..15]
            let lo = (values[j] + 8) as u8;
            let hi = (values[j + 16] + 8) as u8;
            qs[j] = lo | (hi << 4);
        }
        BlockQ4_0 {
            d: f32_to_fp16(scale),
            qs,
        }
    }

    /// Create a Q8_0 block with a given scale and int8 values.
    fn make_q8_0(scale: f32, values: &[i8; 32]) -> BlockQ8_0 {
        BlockQ8_0 {
            d: f32_to_fp16(scale),
            qs: *values,
        }
    }

    #[test]
    fn scalar_single_block() {
        let vals_w = [1i8; 32];
        let vals_a = [1i8; 32];
        let x = [make_q4_0(1.0, &vals_w)];
        let y = [make_q8_0(1.0, &vals_a)];

        let result = vec_dot_q4_0_q8_0_scalar(&x, &y);
        // dot product: 32 × (1 × 1) × (1.0 × 1.0) = 32.0
        assert!((result - 32.0).abs() < 0.01, "expected ~32.0, got {result}");
    }

    #[test]
    fn scalar_zero_scale() {
        let vals = [7i8; 32];
        let x = [make_q4_0(0.0, &vals)];
        let y = [make_q8_0(1.0, &vals)];
        let result = vec_dot_q4_0_q8_0_scalar(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn scalar_negative_values() {
        let vals_w = [-4i8; 32];
        let vals_a = [2i8; 32];
        let x = [make_q4_0(1.0, &vals_w)];
        let y = [make_q8_0(1.0, &vals_a)];
        let result = vec_dot_q4_0_q8_0_scalar(&x, &y);
        // dot: 32 × (-4 × 2) × 1.0 = -256.0
        assert!(
            (result - (-256.0)).abs() < 0.1,
            "expected ~-256.0, got {result}"
        );
    }

    #[test]
    fn scalar_multi_block() {
        let x = [
            make_q4_0(2.0, &[1i8; 32]),
            make_q4_0(0.5, &[-2i8; 32]),
        ];
        let y = [
            make_q8_0(1.0, &[3i8; 32]),
            make_q8_0(1.0, &[4i8; 32]),
        ];
        let result = vec_dot_q4_0_q8_0_scalar(&x, &y);
        // Block 0: 32 × 1 × 3 × (2.0 × 1.0) = 192.0
        // Block 1: 32 × (-2) × 4 × (0.5 × 1.0) = -128.0
        // Total: 64.0
        assert!((result - 64.0).abs() < 0.5, "expected ~64.0, got {result}");
    }

    #[cfg(target_arch = "x86_64")]
    mod simd_tests {
        use super::*;

        fn random_blocks(n: usize, seed: u64) -> (Vec<BlockQ4_0>, Vec<BlockQ8_0>) {
            // Simple LCG for reproducible pseudo-random data
            let mut state = seed;
            let mut next = || -> u64 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                state
            };

            let mut xblocks = Vec::with_capacity(n);
            let mut yblocks = Vec::with_capacity(n);

            for _ in 0..n {
                let scale_x = (next() % 1000) as f32 / 100.0 - 5.0;
                let scale_y = (next() % 1000) as f32 / 100.0 - 5.0;
                let mut vals_x = [0i8; 32];
                let mut vals_y = [0i8; 32];
                for v in vals_x.iter_mut() {
                    *v = ((next() % 15) as i8) - 7; // [-7..+7] (Q4_0 range)
                }
                for v in vals_y.iter_mut() {
                    *v = ((next() % 254) as i32 - 127) as i8; // [-127..+126]
                }
                xblocks.push(make_q4_0(scale_x, &vals_x));
                yblocks.push(make_q8_0(scale_y, &vals_y));
            }
            (xblocks, yblocks)
        }

        #[test]
        fn avx2_matches_scalar_single() {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
                return;
            }
            let x = [make_q4_0(2.0, &[3i8; 32])];
            let y = [make_q8_0(1.5, &[5i8; 32])];

            let scalar = vec_dot_q4_0_q8_0_scalar(&x, &y);
            // SAFETY: AVX2+FMA checked above
            let simd = unsafe { avx2::vec_dot_q4_0_q8_0_avx2(&x, &y) };

            let rel_err = if scalar.abs() > 1e-10 {
                ((simd - scalar) / scalar).abs()
            } else {
                (simd - scalar).abs()
            };
            assert!(
                rel_err < 1e-5,
                "AVX2 vs scalar mismatch: scalar={scalar}, avx2={simd}, rel_err={rel_err}"
            );
        }

        #[test]
        fn avx2_matches_scalar_random() {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
                return;
            }

            for seed in 0..100 {
                let n = (seed % 20) + 1; // 1..20 blocks
                let (x, y) = random_blocks(n as usize, seed);

                let scalar = vec_dot_q4_0_q8_0_scalar(&x, &y);
                // SAFETY: AVX2+FMA checked above
                let simd = unsafe { avx2::vec_dot_q4_0_q8_0_avx2(&x, &y) };

                let rel_err = if scalar.abs() > 1.0 {
                    ((simd - scalar) / scalar).abs()
                } else {
                    (simd - scalar).abs()
                };
                assert!(
                    rel_err < 1e-4,
                    "seed={seed} n={n}: scalar={scalar}, avx2={simd}, rel_err={rel_err}"
                );
            }
        }

        #[test]
        fn avx2_all_zeros() {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
                return;
            }
            let x = [make_q4_0(0.0, &[0i8; 32])];
            let y = [make_q8_0(0.0, &[0i8; 32])];
            // SAFETY: AVX2+FMA checked above
            let result = unsafe { avx2::vec_dot_q4_0_q8_0_avx2(&x, &y) };
            assert_eq!(result, 0.0);
        }

        #[test]
        fn avx2_large_k() {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
                return;
            }
            // K=4096 → 128 blocks (typical hidden_size)
            let (x, y) = random_blocks(128, 42);
            let scalar = vec_dot_q4_0_q8_0_scalar(&x, &y);
            // SAFETY: AVX2+FMA checked above
            let simd = unsafe { avx2::vec_dot_q4_0_q8_0_avx2(&x, &y) };

            let rel_err = if scalar.abs() > 1.0 {
                ((simd - scalar) / scalar).abs()
            } else {
                (simd - scalar).abs()
            };
            assert!(
                rel_err < 1e-4,
                "large K: scalar={scalar}, avx2={simd}, rel_err={rel_err}"
            );
        }

        #[test]
        fn dispatch_works() {
            let (x, y) = random_blocks(16, 12345);
            let scalar = vec_dot_q4_0_q8_0_scalar(&x, &y);
            let dispatched = vec_dot_q4_0_q8_0(&x, &y);

            let rel_err = if scalar.abs() > 1.0 {
                ((dispatched - scalar) / scalar).abs()
            } else {
                (dispatched - scalar).abs()
            };
            assert!(
                rel_err < 1e-4,
                "dispatch vs scalar: scalar={scalar}, dispatched={dispatched}"
            );
        }
    }
}
