//! AVX-512 F32 attention kernels (weight×V accumulation).

use super::common::{bf16_to_f32, bf16x16_load_as_f32};

/// Weight @ V accumulation: acc[d] += sum_j(weights[j] * V_bf16[j][d]) for d in 0..dim
///
/// j-outer loop order with all head_dim accumulators held in zmm registers.
/// Key advantages over the old d-chunk-outer approach:
///   - Sequential V access per j (row-by-row, prefetcher-friendly)
///   - Weight broadcast once per j, not once per (j, d_chunk) → 7x fewer for head_dim=128
///   - No extra F32 buffer allocation (operates on BF16 V directly)
///   - BF16 V = half the cache footprint of F32, better for L1/L2 residency
///
/// `v_stride`: distance in u16 elements between consecutive V rows.
/// Use `dim` for contiguous gathered buffers, or `num_kv_heads * dim` for strided access.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn weight_v_accum_bf16_avx512(
    acc: *mut f32,         // v_prime row: [dim] F32, read-modify-write
    v_bf16: *const u16,    // V buffer: [n_size rows × v_stride], BF16
    weights: *const f32,   // softmax weights: [n_size] F32
    n_size: usize,
    dim: usize,
    v_stride: usize,       // row stride in u16 elements
) {
    use core::arch::x86_64::*;
    let chunks = dim / 16;
    debug_assert!(chunks <= 16, "head_dim > 256 not supported by fixed-size accumulator array");

    // Load all accumulators into zmm registers (up to 16 × __m512 = head_dim ≤ 256)
    let mut accs = [_mm512_setzero_ps(); 16];
    for c in 0..chunks {
        accs[c] = unsafe { _mm512_loadu_ps(acc.add(c * 16)) };
    }

    for j in 0..n_size {
        let w = unsafe { *weights.add(j) };
        if w > 0.0 {
            let vw = _mm512_set1_ps(w);
            let base = unsafe { v_bf16.add(j * v_stride) };
            for c in 0..chunks {
                let v_src = bf16x16_load_as_f32(unsafe { base.add(c * 16) });
                accs[c] = _mm512_fmadd_ps(vw, v_src, accs[c]);
            }
        }
    }

    // Store back
    for c in 0..chunks {
        unsafe { _mm512_storeu_ps(acc.add(c * 16), accs[c]) };
    }

    // Handle remainder (head_dim not multiple of 16 — rare for transformers)
    for d in (chunks * 16)..dim {
        let mut sum = unsafe { *acc.add(d) };
        for j in 0..n_size {
            let w = unsafe { *weights.add(j) };
            if w > 0.0 {
                sum += w * bf16_to_f32(unsafe { *v_bf16.add(j * v_stride + d) });
            }
        }
        unsafe { *acc.add(d) = sum };
    }
}
