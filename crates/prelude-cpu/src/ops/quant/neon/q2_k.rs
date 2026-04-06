//! NEON Q2_K x Q8_K dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// Helper: dot product of two 16-byte signed vectors, multiplied by two scale bytes.
#[inline(always)]
unsafe fn multiply_accum_with_scale(
    aux: &[u8; 16],
    is: usize,
    index: usize,
    q2bytes: (int8x16_t, int8x16_t),
    q8bytes: (int8x16_t, int8x16_t),
) -> i32 {
    let p1 = vdotq_s32(q2bytes.0, q8bytes.0);
    let p2 = vdotq_s32(q2bytes.1, q8bytes.1);
    vaddvq_s32(p1) * aux[is + index] as i32
        + vaddvq_s32(p2) * aux[is + 1 + index] as i32
}

/// NEON Q2_K x Q8_K dot product.
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q2k_q8k_neon(x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let mut sumf = 0f32;
    let mut aux = [0u8; 16];

    let m3 = vdupq_n_u8(0x3);
    let m4 = vdupq_n_u8(0xF);

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = yb.d * fp16_to_f32(xb.d);
        let dmin = -yb.d * fp16_to_f32(xb.dmin);

        let mut q2 = xb.qs.as_ptr();
        let mut q8 = yb.qs.as_ptr();

        // Load and split scales (low nibble) and mins (high nibble)
        let mins_and_scales = vld1q_u8(xb.scales.as_ptr());
        let scales = vandq_u8(mins_and_scales, m4);
        vst1q_u8(aux.as_mut_ptr(), scales);

        let mins = vshrq_n_u8(mins_and_scales, 4);

        // Min contribution via bsums
        let q8sums_0 = vld1q_s16(yb.bsums.as_ptr());
        let q8sums_1 = vld1q_s16(yb.bsums.as_ptr().add(8));
        let mins16_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
        let mins16_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));

        let s0 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16_0), vget_low_s16(q8sums_0)),
            vmull_s16(vget_high_s16(mins16_0), vget_high_s16(q8sums_0)),
        );
        let s1 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16_1), vget_low_s16(q8sums_1)),
            vmull_s16(vget_high_s16(mins16_1), vget_high_s16(q8sums_1)),
        );
        sumf += dmin * vaddvq_s32(vaddq_s32(s0, s1)) as f32;

        let mut isum = 0i32;
        let mut is = 0usize;

        for _j in 0..(QK_K / 128) {
            let q2bits_0 = vld1q_u8(q2);
            let q2bits_1 = vld1q_u8(q2.add(16));
            q2 = q2.add(32);

            // Shift 0
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            q8 = q8.add(32);
            let q2b = (
                vreinterpretq_s8_u8(vandq_u8(q2bits_0, m3)),
                vreinterpretq_s8_u8(vandq_u8(q2bits_1, m3)),
            );
            isum += multiply_accum_with_scale(&aux, is, 0, q2b, (q8b_0, q8b_1));

            // Shift 2
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            q8 = q8.add(32);
            let q2b = (
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0, 2), m3)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1, 2), m3)),
            );
            isum += multiply_accum_with_scale(&aux, is, 2, q2b, (q8b_0, q8b_1));

            // Shift 4
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            q8 = q8.add(32);
            let q2b = (
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0, 4), m3)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1, 4), m3)),
            );
            isum += multiply_accum_with_scale(&aux, is, 4, q2b, (q8b_0, q8b_1));

            // Shift 6
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            q8 = q8.add(32);
            let q2b = (
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0, 6), m3)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1, 6), m3)),
            );
            isum += multiply_accum_with_scale(&aux, is, 6, q2b, (q8b_0, q8b_1));

            is += 8;
        }
        sumf += d * isum as f32;
    }
    sumf
}
