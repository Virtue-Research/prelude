//! NEON Q3_K x Q8_K dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q3_K x Q8_K dot product.
///
/// Q3_K: 3-bit values with 2 low bits in qs and 1 high bit in hmask.
/// Scales are 6-bit signed (stored as unsigned - 32).
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q3k_q8k_neon(x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut aux = [0u32; 3];

    let m3b = vdupq_n_u8(0x3);
    let m0 = vdupq_n_u8(1);
    let m1 = vshlq_n_u8(m0, 1);
    let m2 = vshlq_n_u8(m0, 2);
    let m3 = vshlq_n_u8(m0, 3);

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = yb.d * fp16_to_f32(xb.d);
        let mut q3 = xb.qs.as_ptr();
        let qh = xb.hmask.as_ptr();
        let mut q8 = yb.qs.as_ptr();

        let qhbits_0 = vld1q_u8(qh);
        let qhbits_1 = vld1q_u8(qh.add(16));
        let mut qhbits = (qhbits_0, qhbits_1);

        let mut isum = 0i32;

        // Unpack 6-bit signed scales (subtract 32)
        aux[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
        aux[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
        aux[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

        utmp[3] = ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4);
        utmp[2] = ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4);
        utmp[1] = (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4);
        utmp[0] = (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4);

        let scale_bytes = utmp.as_mut_ptr() as *mut i8;
        for j in 0..16 {
            *scale_bytes.add(j) -= 32i8;
        }
        let mut scale = scale_bytes as *const i8;

        for j in 0..(QK_K / 128) {
            let q3bits_0 = vld1q_u8(q3);
            let q3bits_1 = vld1q_u8(q3.add(16));
            q3 = q3.add(32);

            // First 64 Q8 values
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            let q8b_2 = vld1q_s8(q8.add(32));
            let q8b_3 = vld1q_s8(q8.add(48));
            q8 = q8.add(64);

            // Second 64 Q8 values
            let q8b_4 = vld1q_s8(q8);
            let q8b_5 = vld1q_s8(q8.add(16));
            let q8b_6 = vld1q_s8(q8.add(32));
            let q8b_7 = vld1q_s8(q8.add(48));
            q8 = q8.add(64);

            // Sub-block 0: low 2 bits, hmask bit 0
            let q3h_0 = vshlq_n_u8(vbicq_u8(m0, qhbits.0), 2);
            let q3h_1 = vshlq_n_u8(vbicq_u8(m0, qhbits.1), 2);
            let q3bytes_0 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(q3bits_0, m3b)),
                vreinterpretq_s8_u8(q3h_0),
            );
            let q3bytes_1 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(q3bits_1, m3b)),
                vreinterpretq_s8_u8(q3h_1),
            );

            // Sub-block 1: bits 2..3, hmask bit 1
            let q3h_2 = vshlq_n_u8(vbicq_u8(m1, qhbits.0), 1);
            let q3h_3 = vshlq_n_u8(vbicq_u8(m1, qhbits.1), 1);
            let q3bytes_2 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0, 2), m3b)),
                vreinterpretq_s8_u8(q3h_2),
            );
            let q3bytes_3 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1, 2), m3b)),
                vreinterpretq_s8_u8(q3h_3),
            );

            let p0 = vdotq_s32(q3bytes_0, q8b_0);
            let p1 = vdotq_s32(q3bytes_1, q8b_1);
            let p2 = vdotq_s32(q3bytes_2, q8b_2);
            let p3 = vdotq_s32(q3bytes_3, q8b_3);
            isum += vaddvq_s32(p0) * *scale as i32
                + vaddvq_s32(p1) * *scale.add(1) as i32
                + vaddvq_s32(p2) * *scale.add(2) as i32
                + vaddvq_s32(p3) * *scale.add(3) as i32;
            scale = scale.add(4);

            // Sub-block 2: bits 4..5, hmask bit 2
            let q3h_4 = vbicq_u8(m2, qhbits.0);
            let q3h_5 = vbicq_u8(m2, qhbits.1);
            let q3bytes_4 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0, 4), m3b)),
                vreinterpretq_s8_u8(q3h_4),
            );
            let q3bytes_5 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1, 4), m3b)),
                vreinterpretq_s8_u8(q3h_5),
            );

            // Sub-block 3: bits 6..7, hmask bit 3
            let q3h_6 = vshrq_n_u8(vbicq_u8(m3, qhbits.0), 1);
            let q3h_7 = vshrq_n_u8(vbicq_u8(m3, qhbits.1), 1);
            let q3bytes_6 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0, 6), m3b)),
                vreinterpretq_s8_u8(q3h_6),
            );
            let q3bytes_7 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1, 6), m3b)),
                vreinterpretq_s8_u8(q3h_7),
            );

            let p4 = vdotq_s32(q3bytes_4, q8b_4);
            let p5 = vdotq_s32(q3bytes_5, q8b_5);
            let p6 = vdotq_s32(q3bytes_6, q8b_6);
            let p7 = vdotq_s32(q3bytes_7, q8b_7);
            isum += vaddvq_s32(p4) * *scale as i32
                + vaddvq_s32(p5) * *scale.add(1) as i32
                + vaddvq_s32(p6) * *scale.add(2) as i32
                + vaddvq_s32(p7) * *scale.add(3) as i32;
            scale = scale.add(4);

            // Shift hmask right by 4 for the next iteration
            if j == 0 {
                qhbits.0 = vshrq_n_u8(qhbits.0, 4);
                qhbits.1 = vshrq_n_u8(qhbits.1, 4);
            }
        }
        sumf += d * isum as f32;
    }
    sumf
}
