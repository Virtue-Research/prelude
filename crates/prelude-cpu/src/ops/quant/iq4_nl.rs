//! IQ4_NL × Q8_0 dot-product kernel.
//!
//! Non-linear 4-bit quantization: nibbles index into KVALUES_IQ4NL lookup table
//! instead of the uniform [-8..+7] mapping used by Q4_0.
//!
//! Scalar reference + AVX2 + auto-dispatch.

use super::types::*;

// ── Scalar reference ────────────────────────────────────────────────────

/// IQ4_NL · Q8_0 dot product — scalar reference implementation.
///
/// Each IQ4_NL block has 32 values packed as 16 nibble pairs. Each nibble
/// is an index into [`KVALUES_IQ4NL`], giving a non-uniform signed value.
pub fn vec_dot_iq4_nl_q8_0_scalar(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len(), y.len());

    let mut sumf: f32 = 0.0;

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = fp16_to_f32(xb.d) * fp16_to_f32(yb.d);

        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..(QK4_NL / 2) {
            let v0 = KVALUES_IQ4NL[(xb.qs[j] & 0x0F) as usize] as i32;
            let v1 = KVALUES_IQ4NL[(xb.qs[j] >> 4) as usize] as i32;

            sumi0 += v0 * (yb.qs[j] as i32);
            sumi1 += v1 * (yb.qs[j + QK4_NL / 2] as i32);
        }

        sumf += (sumi0 + sumi1) as f32 * d;
    }

    sumf
}

// ── AVX2 implementation ─────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn fp16_to_f32_hw(h: u16) -> f32 {
        unsafe {
            let v = _mm_cvtsi32_si128(h as i32);
            _mm_cvtss_f32(_mm_cvtph_ps(v))
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

    /// Load the IQ4_NL lookup table into a 128-bit register.
    #[inline(always)]
    unsafe fn load_iq4nl_table() -> __m128i {
        unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i) }
    }

    /// Look up 32 nibble indices from qs[16] using the IQ4_NL table.
    /// Returns a 256-bit vector of signed 8-bit looked-up values.
    #[inline(always)]
    unsafe fn lookup_iq4nl_32(qs_ptr: *const u8, table: __m128i) -> __m256i {
        unsafe {
            let raw = _mm_loadu_si128(qs_ptr as *const __m128i);
            let mask = _mm_set1_epi8(0x0F);

            // Low nibbles (elements 0..15)
            let lo_idx = _mm_and_si128(raw, mask);
            let lo_vals = _mm_shuffle_epi8(table, lo_idx);

            // High nibbles (elements 16..31)
            let hi_idx = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
            let hi_vals = _mm_shuffle_epi8(table, hi_idx);

            _mm256_set_m128i(hi_vals, lo_vals)
        }
    }

    /// AVX2 IQ4_NL · Q8_0 dot product.
    #[target_feature(enable = "avx2,fma,f16c,ssse3")]
    pub(super) unsafe fn vec_dot_iq4_nl_q8_0_avx2(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
        unsafe {
            let nb = x.len();
            let mut acc = _mm256_setzero_ps();
            let table = load_iq4nl_table();

            for ib in 0..nb {
                let xb = x.get_unchecked(ib);
                let yb = y.get_unchecked(ib);

                let d = _mm256_set1_ps(fp16_to_f32_hw(xb.d) * fp16_to_f32_hw(yb.d));
                let qx = lookup_iq4nl_32(xb.qs.as_ptr(), table);
                let qy = _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i);

                // Signed × signed dot product via sign trick + maddubs
                let ax = _mm256_sign_epi8(qx, qx);
                let sy = _mm256_sign_epi8(qy, qx);
                let dot = _mm256_maddubs_epi16(ax, sy);
                let sum32 = _mm256_madd_epi16(dot, _mm256_set1_epi16(1));
                let sumf = _mm256_cvtepi32_ps(sum32);

                acc = _mm256_fmadd_ps(d, sumf, acc);
            }

            hsum_float_8(acc)
        }
    }
}

// ── Auto-dispatch ───────────────────────────────────────────────────────

/// Compute IQ4_NL · Q8_0 dot product, automatically selecting the best kernel.
#[inline]
pub fn vec_dot_iq4_nl_q8_0(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("ssse3")
        {
            return unsafe { avx2::vec_dot_iq4_nl_q8_0_avx2(x, y) };
        }
    }
    vec_dot_iq4_nl_q8_0_scalar(x, y)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_iq4_nl(scale: f32, indices: &[u8; 32]) -> BlockIQ4NL {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = (indices[j] & 0x0F) | ((indices[j + 16] & 0x0F) << 4);
        }
        BlockIQ4NL {
            d: f32_to_fp16(scale),
            qs,
        }
    }

    fn make_q8_0(scale: f32, values: &[i8; 32]) -> BlockQ8_0 {
        BlockQ8_0 {
            d: f32_to_fp16(scale),
            qs: *values,
        }
    }

    #[test]
    fn scalar_single_block() {
        // All indices = 8 → KVALUES_IQ4NL[8] = 1
        let indices = [8u8; 32];
        let x = [make_iq4_nl(1.0, &indices)];
        let y = [make_q8_0(1.0, &[1i8; 32])];

        let result = vec_dot_iq4_nl_q8_0_scalar(&x, &y);
        // dot: 32 × (1 × 1) × 1.0 = 32.0
        assert!((result - 32.0).abs() < 0.1, "expected ~32.0, got {result}");
    }

    #[test]
    fn scalar_table_values() {
        // Verify that lookup table values are used correctly
        // Index 0 → -127, index 15 → 113
        let mut indices = [0u8; 32];
        indices[0] = 0; // KVALUES_IQ4NL[0] = -127
        indices[16] = 15; // KVALUES_IQ4NL[15] = 113

        let x = [make_iq4_nl(1.0, &indices)];
        let mut q8_vals = [0i8; 32];
        q8_vals[0] = 1;
        q8_vals[16] = 1;
        let y = [make_q8_0(1.0, &q8_vals)];

        let result = vec_dot_iq4_nl_q8_0_scalar(&x, &y);
        // -127 * 1 + 113 * 1 = -14
        assert!(
            (result - (-14.0)).abs() < 0.5,
            "expected ~-14.0, got {result}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    mod simd_tests {
        use super::*;

        fn random_blocks(n: usize, seed: u64) -> (Vec<BlockIQ4NL>, Vec<BlockQ8_0>) {
            let mut state = seed;
            let mut next = || -> u64 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                state
            };

            let mut xblocks = Vec::with_capacity(n);
            let mut yblocks = Vec::with_capacity(n);

            for _ in 0..n {
                let scale_x = (next() % 1000) as f32 / 100.0 - 5.0;
                let scale_y = (next() % 1000) as f32 / 100.0 - 5.0;
                let mut indices = [0u8; 32];
                let mut vals_y = [0i8; 32];
                for v in indices.iter_mut() {
                    *v = (next() % 16) as u8;
                }
                for v in vals_y.iter_mut() {
                    *v = ((next() % 254) as i32 - 127) as i8;
                }
                xblocks.push(make_iq4_nl(scale_x, &indices));
                yblocks.push(make_q8_0(scale_y, &vals_y));
            }
            (xblocks, yblocks)
        }

        #[test]
        fn avx2_matches_scalar_random() {
            if !is_x86_feature_detected!("avx2")
                || !is_x86_feature_detected!("fma")
                || !is_x86_feature_detected!("ssse3")
            {
                return;
            }

            for seed in 0..100 {
                let n = (seed % 20) + 1;
                let (x, y) = random_blocks(n as usize, seed);

                let scalar = vec_dot_iq4_nl_q8_0_scalar(&x, &y);
                let simd = unsafe { avx2::vec_dot_iq4_nl_q8_0_avx2(&x, &y) };

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
    }
}
