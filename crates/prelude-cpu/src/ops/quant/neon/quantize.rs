//! NEON activation quantization kernels: FP32 -> Q8_0 / Q8_K.

use crate::ops::quant::types::*;
use core::arch::aarch64::*;

/// Quantize a row of FP32 values into Q8_0 blocks using NEON.
///
/// Processes 32 floats at a time using 4 x float32x4_t registers per block.
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn quantize_row_q8_0_neon(x: &[f32]) -> Vec<BlockQ8_0> {
    debug_assert!(x.len() % QK8_0 == 0, "input length must be multiple of {QK8_0}");
    let nb = x.len() / QK8_0;
    let mut output = Vec::with_capacity(nb);

    let mut ptr = x.as_ptr();

    for _ in 0..nb {
        // Load 32 floats in 8 x float32x4_t (4 floats each)
        let v0 = vld1q_f32(ptr);
        let v1 = vld1q_f32(ptr.add(4));
        let v2 = vld1q_f32(ptr.add(8));
        let v3 = vld1q_f32(ptr.add(12));
        let v4 = vld1q_f32(ptr.add(16));
        let v5 = vld1q_f32(ptr.add(20));
        let v6 = vld1q_f32(ptr.add(24));
        let v7 = vld1q_f32(ptr.add(28));
        ptr = ptr.add(32);

        // Compute max absolute value across all 32 elements
        let a0 = vabsq_f32(v0);
        let a1 = vabsq_f32(v1);
        let a2 = vabsq_f32(v2);
        let a3 = vabsq_f32(v3);
        let a4 = vabsq_f32(v4);
        let a5 = vabsq_f32(v5);
        let a6 = vabsq_f32(v6);
        let a7 = vabsq_f32(v7);

        let m01 = vmaxq_f32(a0, a1);
        let m23 = vmaxq_f32(a2, a3);
        let m45 = vmaxq_f32(a4, a5);
        let m67 = vmaxq_f32(a6, a7);
        let m0123 = vmaxq_f32(m01, m23);
        let m4567 = vmaxq_f32(m45, m67);
        let mall = vmaxq_f32(m0123, m4567);
        let amax = vmaxvq_f32(mall);

        let d = amax / 127.0f32;
        let id = if amax != 0.0 { 127.0f32 / amax } else { 0.0f32 };
        let mul = vdupq_n_f32(id);

        // Scale, round, and convert to i32
        let r0 = vcvtnq_s32_f32(vmulq_f32(v0, mul));
        let r1 = vcvtnq_s32_f32(vmulq_f32(v1, mul));
        let r2 = vcvtnq_s32_f32(vmulq_f32(v2, mul));
        let r3 = vcvtnq_s32_f32(vmulq_f32(v3, mul));
        let r4 = vcvtnq_s32_f32(vmulq_f32(v4, mul));
        let r5 = vcvtnq_s32_f32(vmulq_f32(v5, mul));
        let r6 = vcvtnq_s32_f32(vmulq_f32(v6, mul));
        let r7 = vcvtnq_s32_f32(vmulq_f32(v7, mul));

        // Narrow i32 -> i16 -> i8
        let n01 = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
        let n23 = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));
        let n45 = vcombine_s16(vqmovn_s32(r4), vqmovn_s32(r5));
        let n67 = vcombine_s16(vqmovn_s32(r6), vqmovn_s32(r7));

        let b0 = vcombine_s8(vqmovn_s16(n01), vqmovn_s16(n23));
        let b1 = vcombine_s8(vqmovn_s16(n45), vqmovn_s16(n67));

        let mut block = BlockQ8_0 {
            d: f32_to_fp16(d),
            qs: [0i8; 32],
        };
        vst1q_s8(block.qs.as_mut_ptr(), b0);
        vst1q_s8(block.qs.as_mut_ptr().add(16), b1);
        output.push(block);
    }

    output
}

/// Quantize a row of FP32 values into Q8_K blocks using NEON.
///
/// Q8_K uses f32 scale and pre-computes 16-element sub-block sums (bsums).
///
/// # Safety
/// Requires aarch64 target (NEON is baseline).
#[inline]
pub unsafe fn quantize_row_q8k_neon(x: &[f32]) -> Vec<BlockQ8K> {
    debug_assert!(x.len() % QK_K == 0, "input length must be multiple of {QK_K}");
    let nb = x.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let block = &x[i * QK_K..(i + 1) * QK_K];

        // Find max absolute value using NEON
        let mut max_v = vdupq_n_f32(0.0);
        let mut ptr = block.as_ptr();
        for _ in 0..(QK_K / 4) {
            let v = vld1q_f32(ptr);
            max_v = vmaxq_f32(max_v, vabsq_f32(v));
            ptr = ptr.add(4);
        }
        let amax = vmaxvq_f32(max_v);

        let d = amax / 127.0f32;
        let id = if amax != 0.0 { 127.0f32 / amax } else { 0.0f32 };
        let mul = vdupq_n_f32(id);

        let mut qs = [0i8; QK_K];
        let mut bsums = [0i16; QK_K / 16];

        // Quantize and compute sub-block sums
        let mut src = block.as_ptr();
        let mut dst = qs.as_mut_ptr();

        for s in 0..(QK_K / 16) {
            let mut bsum = vdupq_n_s32(0);

            // Process 16 elements (4 x float32x4_t)
            for _ in 0..4 {
                let v = vld1q_f32(src);
                src = src.add(4);
                let r = vcvtnq_s32_f32(vmulq_f32(v, mul));
                bsum = vaddq_s32(bsum, r);

                // Narrow and store
                let n16 = vqmovn_s32(r);
                let n8 = vqmovn_s16(vcombine_s16(n16, n16));
                // Store only the low 4 bytes
                vst1_lane_s32(dst as *mut i32, vreinterpret_s32_s8(n8), 0);
                dst = dst.add(4);
            }

            bsums[s] = vaddvq_s32(bsum) as i16;
        }

        output.push(BlockQ8K { d, qs, bsums });
    }

    output
}
