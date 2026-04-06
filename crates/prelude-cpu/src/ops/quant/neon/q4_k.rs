//! NEON Q4_K x Q8_K dot product kernel.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

/// NEON Q4_K x Q8_K dot product.
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn vec_dot_q4k_q8k_neon(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0xF);

    for (xb, yb) in x.iter().zip(y.iter()) {
        let d = yb.d * fp16_to_f32(xb.d);
        let dmin = yb.d * fp16_to_f32(xb.dmin);

        // Unpack scales/mins from 12-byte packed format
        utmp[0] = u32::from_le_bytes([xb.scales[0], xb.scales[1], xb.scales[2], xb.scales[3]]);
        utmp[1] = u32::from_le_bytes([xb.scales[4], xb.scales[5], xb.scales[6], xb.scales[7]]);
        utmp[2] = u32::from_le_bytes([xb.scales[8], xb.scales[9], xb.scales[10], xb.scales[11]]);

        // Compute mins from the packing (same bit manipulation as candle/llama.cpp)
        let mins8 = vld1_u32(
            [
                utmp[1] & KMASK1,
                ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
            ]
            .as_ptr(),
        );

        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[0] &= KMASK1;

        // Min contribution via bsums
        let q8sums = vpaddq_s16(
            vld1q_s16(yb.bsums.as_ptr()),
            vld1q_s16(yb.bsums.as_ptr().add(8)),
        );
        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        sumf -= dmin * vaddvq_s32(prod) as f32;

        // Write unpacked scales into byte array
        let utmp_bytes: [u8; 16] = bytemuck::cast(utmp);
        scales.copy_from_slice(&utmp_bytes);

        let mut q4 = xb.qs.as_ptr();
        let mut q8 = yb.qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for j in 0..(QK_K / 64) {
            // Load 32 bytes of Q4_K nibbles (two 16-byte loads)
            let q4bits_0 = vld1q_u8(q4);
            let q4bits_1 = vld1q_u8(q4.add(16));
            q4 = q4.add(32);

            // Low nibbles paired with first 32 Q8 values
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            q8 = q8.add(32);
            let q4bytes_0 = vreinterpretq_s8_u8(vandq_u8(q4bits_0, m4b));
            let q4bytes_1 = vreinterpretq_s8_u8(vandq_u8(q4bits_1, m4b));
            let p0 = vdotq_s32(q4bytes_0, q8b_0);
            let p1 = vdotq_s32(q4bytes_1, q8b_1);
            sumi1 += vaddvq_s32(vaddq_s32(p0, p1)) * scales[2 * j] as i32;

            // High nibbles paired with next 32 Q8 values
            let q8b_2 = vld1q_s8(q8);
            let q8b_3 = vld1q_s8(q8.add(16));
            q8 = q8.add(32);
            let q4bytes_2 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_0, 4));
            let q4bytes_3 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_1, 4));
            let p2 = vdotq_s32(q4bytes_2, q8b_2);
            let p3 = vdotq_s32(q4bytes_3, q8b_3);
            sumi2 += vaddvq_s32(vaddq_s32(p2, p3)) * scales[2 * j + 1] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}
