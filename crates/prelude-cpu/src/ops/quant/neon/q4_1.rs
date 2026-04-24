//! NEON Q4_1 x Q8_1 dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

/// Unsigned x signed dot product for Q4_1 (unsigned nibbles) x Q8_1 (signed i8).
///
/// Computes sum(u8[i] * i8[i]) as i32x4 using widening multiplies.
#[inline(always)]
unsafe fn vdotq_u8_s8(a: uint8x16_t, b: int8x16_t) -> int32x4_t {
    // Treat both as signed for vmull_s8: since a is in [0..15], no sign issues.
    let a_s8 = vreinterpretq_s8_u8(a);
    let p0 = vmull_s8(vget_low_s8(a_s8), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a_s8), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q4_1 x Q8_1 dot product.
///
/// Q4_1 is asymmetric: dot = d_w * d_a * sum(nibble[i] * q8[i]) + m_w * s_a
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q4_1_q8_1_neon(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();

    let mut sumv0 = vdupq_n_f32(0.0f32);
    let mut summs: f32 = 0.0;
    let m4b = vdupq_n_u8(0x0F);

    for i in 0..nb {
        let xb = x.get_unchecked(i);
        let yb = y.get_unchecked(i);

        // Accumulate minimum contribution
        summs += fp16_to_f32(xb.m) * fp16_to_f32(yb.s);

        // Load 16 bytes of packed nibbles
        let v0 = vld1q_u8(xb.qs.as_ptr());

        // Unpack low and high nibbles (unsigned [0..15])
        let v0l = vandq_u8(v0, m4b);
        let v0h = vshrq_n_u8(v0, 4);

        // Load Q8_1 values
        let v1l = vld1q_s8(yb.qs.as_ptr());
        let v1h = vld1q_s8(yb.qs.as_ptr().add(16));

        // Dot products (unsigned x signed)
        let pl0 = vdotq_u8_s8(v0l, v1l);
        let ph0 = vdotq_u8_s8(v0h, v1h);

        sumv0 = vmlaq_n_f32(
            sumv0,
            vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
            fp16_to_f32(xb.d) * fp16_to_f32(yb.d),
        );
    }

    vaddvq_f32(sumv0) + summs
}
