//! NEON Q5_0 x Q8_0 dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q5_0 x Q8_0 dot product.
///
/// Q5_0 is symmetric 5-bit: 4 low bits in qs, 1 high bit in qh[4].
/// Value = d * (combined_5bit - 16).
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q5_0_q8_0_neon(x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();

    let mut sumv0 = vdupq_n_f32(0.0f32);
    let m4b = vdupq_n_u8(0x0F);
    let s16b = vdupq_n_s8(16);

    for i in 0..nb {
        let xb = x.get_unchecked(i);
        let yb = y.get_unchecked(i);

        let qh = u32::from_le_bytes(xb.qh);

        // Load 16 bytes of packed nibbles (low 4 bits)
        let v0 = vld1q_u8(xb.qs.as_ptr());
        let v0l = vandq_u8(v0, m4b); // low nibble: elements 0..15
        let v0h = vshrq_n_u8(v0, 4); // high nibble: elements 16..31

        // Extract high bits and combine with low 4 bits
        // For elements 0..15, high bit is bit j of qh
        // For elements 16..31, high bit is bit (j+16) of qh
        let mut qh_lo = [0u8; 16];
        let mut qh_hi = [0u8; 16];
        for j in 0..16 {
            qh_lo[j] = (((qh >> j) & 1) << 4) as u8;
            qh_hi[j] = (((qh >> (j + 16)) & 1) << 4) as u8;
        }

        let qh_lo_v = vld1q_u8(qh_lo.as_ptr());
        let qh_hi_v = vld1q_u8(qh_hi.as_ptr());

        // Combine: 5-bit value = (low 4 bits) | (high bit << 4)
        let q5l = vreinterpretq_s8_u8(vorrq_u8(v0l, qh_lo_v));
        let q5h = vreinterpretq_s8_u8(vorrq_u8(v0h, qh_hi_v));

        // Subtract 16 to get signed range [-16..+15]
        let q5ls = vsubq_s8(q5l, s16b);
        let q5hs = vsubq_s8(q5h, s16b);

        // Load Q8_0 values
        let v1l = vld1q_s8(yb.qs.as_ptr());
        let v1h = vld1q_s8(yb.qs.as_ptr().add(16));

        let pl0 = vdotq_s32(q5ls, v1l);
        let ph0 = vdotq_s32(q5hs, v1h);

        sumv0 = vmlaq_n_f32(
            sumv0,
            vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
            fp16_to_f32(xb.d) * fp16_to_f32(yb.d),
        );
    }

    vaddvq_f32(sumv0)
}
