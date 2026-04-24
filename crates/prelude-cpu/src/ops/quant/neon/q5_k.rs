//! NEON Q5_K x Q8_K dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q5_K x Q8_K dot product.
///
/// Q5_K: 5-bit values with 4 low bits in qs and 1 high bit in qh.
/// Same scale/min packing as Q4_K.
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q5k_q8k_neon(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0xF);
    let mone = vdupq_n_u8(1);
    let mtwo = vdupq_n_u8(2);

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = yb.d * fp16_to_f32(xb.d);
        let dmin = yb.d * fp16_to_f32(xb.dmin);

        // Min contribution via bsums
        let q8sums = vpaddq_s16(
            vld1q_s16(yb.bsums.as_ptr()),
            vld1q_s16(yb.bsums.as_ptr().add(8)),
        );

        // Unpack scales/mins from 12-byte packed format
        utmp[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
        utmp[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
        utmp[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        // mins are in utmp[2..4], scales in utmp[0..2]
        let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
        let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        let sumi_mins = vaddvq_s32(prod);

        let mut scales = utmp.as_ptr() as *const u8;
        let mut q5 = xb.qs.as_ptr();
        let mut q8 = yb.qs.as_ptr();

        let qhbits_0 = vld1q_u8(xb.qh.as_ptr());
        let qhbits_1 = vld1q_u8(xb.qh.as_ptr().add(16));
        let mut qhbits = (qhbits_0, qhbits_1);

        let mut sumi = 0i32;

        for _j in 0..(QK_K / 64) {
            let q5bits_0 = vld1q_u8(q5);
            let q5bits_1 = vld1q_u8(q5.add(16));
            q5 = q5.add(32);

            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            let q8b_2 = vld1q_s8(q8.add(32));
            let q8b_3 = vld1q_s8(q8.add(48));
            q8 = q8.add(64);

            // Extract high bits for sub-block 0 (low nibbles)
            let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
            let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
            // Extract high bits for sub-block 1 (high nibbles)
            let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
            let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);
            // Shift qhbits right by 2 for next iteration
            qhbits.0 = vshrq_n_u8(qhbits.0, 2);
            qhbits.1 = vshrq_n_u8(qhbits.1, 2);

            // Combine low 4 bits + high bit for sub-block 0
            let q5bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits_0, m4b), q5h_0));
            let q5bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits_1, m4b), q5h_1));
            // Combine high 4 bits + high bit for sub-block 1
            let q5bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits_0, 4), q5h_2));
            let q5bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits_1, 4), q5h_3));

            let p0 = vdotq_s32(q5bytes_0, q8b_0);
            let p1 = vdotq_s32(q5bytes_1, q8b_1);
            sumi += vaddvq_s32(vaddq_s32(p0, p1)) * *scales as i32;
            scales = scales.add(1);

            let p2 = vdotq_s32(q5bytes_2, q8b_2);
            let p3 = vdotq_s32(q5bytes_3, q8b_3);
            sumi += vaddvq_s32(vaddq_s32(p2, p3)) * *scales as i32;
            scales = scales.add(1);
        }
        sumf += d * sumi as f32 - dmin * sumi_mins as f32;
    }
    sumf
}
