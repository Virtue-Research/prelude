//! NEON Q6_K x Q8_K dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q6_K x Q8_K dot product.
///
/// Q6_K: 6-bit values stored as 4 low bits (ql) + 2 high bits (qh).
/// Symmetric: value = combined_6bit - 32.
/// 16 sub-blocks of 16 elements each with signed 8-bit scales.
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q6k_q8k_neon(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let mut sum = 0f32;

    let m4b = vdupq_n_u8(0xF);
    let mone = vdupq_n_u8(3);

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d_all = fp16_to_f32(xb.d);

        let mut q6 = xb.ql.as_ptr();
        let mut qh = xb.qh.as_ptr();
        let mut q8 = yb.qs.as_ptr();
        let mut scale = xb.scales.as_ptr();

        // Compute the bsums * scales contribution for the -32 bias
        let q8sums_0 = vld1q_s16(yb.bsums.as_ptr());
        let q8sums_1 = vld1q_s16(yb.bsums.as_ptr().add(8));
        let scales_v = vld1q_s8(xb.scales.as_ptr());
        let q6scales_0 = vmovl_s8(vget_low_s8(scales_v));
        let q6scales_1 = vmovl_s8(vget_high_s8(scales_v));

        let prod = vaddq_s32(
            vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_0), vget_low_s16(q6scales_0)),
                vmull_s16(vget_high_s16(q8sums_0), vget_high_s16(q6scales_0)),
            ),
            vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_1), vget_low_s16(q6scales_1)),
                vmull_s16(vget_high_s16(q8sums_1), vget_high_s16(q6scales_1)),
            ),
        );
        let isum_mins = vaddvq_s32(prod);

        let mut isum = 0i32;

        for _j in 0..(QK_K / 128) {
            // Load high bits: 32 bytes of qh
            let qhbits_0 = vld1q_u8(qh);
            let qhbits_1 = vld1q_u8(qh.add(16));
            qh = qh.add(32);

            // Load low bits: 64 bytes of ql (four 16-byte loads)
            let q6bits_0 = vld1q_u8(q6);
            let q6bits_1 = vld1q_u8(q6.add(16));
            let q6bits_2 = vld1q_u8(q6.add(32));
            let q6bits_3 = vld1q_u8(q6.add(48));
            q6 = q6.add(64);

            // Load Q8 values: first 64
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            let q8b_2 = vld1q_s8(q8.add(32));
            let q8b_3 = vld1q_s8(q8.add(48));
            q8 = q8.add(64);

            // Combine low nibbles with high 2 bits (sub-blocks 0-3)
            let q6h_0 = vshlq_n_u8(vandq_u8(mone, qhbits_0), 4);
            let q6h_1 = vshlq_n_u8(vandq_u8(mone, qhbits_1), 4);
            let shifted_0 = vshrq_n_u8(qhbits_0, 2);
            let q6h_2 = vshlq_n_u8(vandq_u8(mone, shifted_0), 4);
            let shifted_1 = vshrq_n_u8(qhbits_1, 2);
            let q6h_3 = vshlq_n_u8(vandq_u8(mone, shifted_1), 4);

            let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_0, m4b), q6h_0));
            let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_1, m4b), q6h_1));
            let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_2, m4b), q6h_2));
            let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_3, m4b), q6h_3));

            let p0 = vdotq_s32(q6bytes_0, q8b_0);
            let p1 = vdotq_s32(q6bytes_1, q8b_1);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
            scale = scale.add(2);

            let p2 = vdotq_s32(q6bytes_2, q8b_2);
            let p3 = vdotq_s32(q6bytes_3, q8b_3);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
            scale = scale.add(2);

            // Load Q8 values: second 64
            let q8b_4 = vld1q_s8(q8);
            let q8b_5 = vld1q_s8(q8.add(16));
            let q8b_6 = vld1q_s8(q8.add(32));
            let q8b_7 = vld1q_s8(q8.add(48));
            q8 = q8.add(64);

            // High nibbles with high 2 bits (sub-blocks 4-7)
            let shifted_2 = vshrq_n_u8(qhbits_0, 4);
            let q6h_4 = vshlq_n_u8(vandq_u8(mone, shifted_2), 4);
            let shifted_3 = vshrq_n_u8(qhbits_1, 4);
            let q6h_5 = vshlq_n_u8(vandq_u8(mone, shifted_3), 4);
            let shifted_4 = vshrq_n_u8(qhbits_0, 6);
            let q6h_6 = vshlq_n_u8(vandq_u8(mone, shifted_4), 4);
            let shifted_5 = vshrq_n_u8(qhbits_1, 6);
            let q6h_7 = vshlq_n_u8(vandq_u8(mone, shifted_5), 4);

            let q6bytes_4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_0, 4), q6h_4));
            let q6bytes_5 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_1, 4), q6h_5));
            let q6bytes_6 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_2, 4), q6h_6));
            let q6bytes_7 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_3, 4), q6h_7));

            let p4 = vdotq_s32(q6bytes_4, q8b_4);
            let p5 = vdotq_s32(q6bytes_5, q8b_5);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p4) * scale0 + vaddvq_s32(p5) * scale1;
            scale = scale.add(2);

            let p6 = vdotq_s32(q6bytes_6, q8b_6);
            let p7 = vdotq_s32(q6bytes_7, q8b_7);
            let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
            isum += vaddvq_s32(p6) * scale0 + vaddvq_s32(p7) * scale1;
            scale = scale.add(2);
        }
        sum += d_all * yb.d * ((isum - 32 * isum_mins) as f32);
    }
    sum
}
