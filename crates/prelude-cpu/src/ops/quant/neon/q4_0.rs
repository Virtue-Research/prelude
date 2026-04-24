//! NEON Q4_0 x Q8_0 dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

/// Signed 8-bit dot product: sum of element-wise i8*i8, accumulated into i32x4.
///
/// Equivalent to the `vdotq_s32` instruction (ARMv8.2 dotprod), but implemented
/// using baseline NEON (vmull + vpaddl) for universal aarch64 compatibility.
#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q4_0 x Q8_0 dot product.
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q4_0_q8_0_neon(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();

    let mut sumv0 = vdupq_n_f32(0.0f32);
    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x8);

    for i in 0..nb {
        let xb = x.get_unchecked(i);
        let yb = y.get_unchecked(i);

        // Load 16 bytes of packed nibbles
        let v0 = vld1q_u8(xb.qs.as_ptr());

        // Unpack low and high nibbles to i8
        let v0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
        let v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));

        // Subtract 8 to get signed values [-8..+7]
        let v0ls = vsubq_s8(v0l, s8b);
        let v0hs = vsubq_s8(v0h, s8b);

        // Load Q8_0 values (32 x i8 = two 16-byte loads)
        let v1l = vld1q_s8(yb.qs.as_ptr());
        let v1h = vld1q_s8(yb.qs.as_ptr().add(16));

        // Dot products
        let pl0 = vdotq_s32(v0ls, v1l);
        let ph0 = vdotq_s32(v0hs, v1h);

        // Accumulate: scale * sum
        sumv0 = vmlaq_n_f32(
            sumv0,
            vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
            fp16_to_f32(xb.d) * fp16_to_f32(yb.d),
        );
    }

    vaddvq_f32(sumv0)
}
