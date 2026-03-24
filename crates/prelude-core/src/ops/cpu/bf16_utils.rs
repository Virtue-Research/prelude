//! Shared BF16 ↔ F32 conversion utilities — scalar and AVX-512 vectorized.
//!
//! These are used across rmsnorm, silu_mul, rope, and attention kernels.
//! Centralised here to eliminate per-file duplication.

/// Reinterpret a raw `u16` BF16 bit pattern as `f32`.
#[inline(always)]
pub(crate) fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Round-to-nearest-even conversion from `f32` to BF16 (returned as `u16`).
#[inline(always)]
pub(crate) fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Load 16 BF16 values (256 bits) and widen to 16×F32 (512 bits).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) fn bf16x16_load_as_f32(ptr: *const u16) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    let bf16_vals = unsafe { _mm256_loadu_si256(ptr as *const __m256i) };
    let extended = _mm512_cvtepu16_epi32(bf16_vals);
    let shifted = _mm512_slli_epi32(extended, 16);
    _mm512_castsi512_ps(shifted)
}

/// Narrow 16×F32 to 16 BF16 values with round-to-nearest-even, and store.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) fn f32x16_store_as_bf16(ptr: *mut u16, vals: core::arch::x86_64::__m512) {
    use core::arch::x86_64::*;
    let bits = _mm512_castps_si512(vals);
    let lsb = _mm512_srli_epi32::<16>(bits);
    let lsb_masked = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
    let rounding_bias = _mm512_add_epi32(lsb_masked, _mm512_set1_epi32(0x7FFF));
    let rounded = _mm512_add_epi32(bits, rounding_bias);
    let shifted = _mm512_srli_epi32::<16>(rounded);
    let packed = _mm512_cvtepi32_epi16(shifted);
    unsafe { _mm256_storeu_si256(ptr as *mut __m256i, packed) };
}

#[cfg(test)]
pub(crate) fn make_bf16_vec(vals: &[f32]) -> Vec<u16> {
    vals.iter().map(|&v| f32_to_bf16(v)).collect()
}

#[cfg(test)]
pub(crate) fn to_f32_vec(vals: &[u16]) -> Vec<f32> {
    vals.iter().map(|&v| bf16_to_f32(v)).collect()
}
