//! AVX-512 BF16 native kernels using dpbf16ps instruction.

use super::common::bf16_to_f32;

/// Dot product of two BF16 vectors using native dpbf16ps instruction.
/// Returns f32 result. Requires avx512_bf16 feature.
#[inline]
pub(super) fn dot_bf16_bf16_native(a: &[u16], b: &[u16], len: usize) -> f32 {
    if is_x86_feature_detected!("avx512bf16") {
        return unsafe { dot_bf16_bf16_dpbf16ps(a.as_ptr(), b.as_ptr(), len) };
    }
    // Fallback to existing bf16→f32 conversion path
    super::common::dot_bf16_f32_scalar(a, b, len)
}

/// AVX-512 BF16: dot product using _mm512_dpbf16_ps (32 BF16 pairs per iteration).
#[target_feature(enable = "avx512f,avx512bf16")]
fn dot_bf16_bf16_dpbf16ps(a: *const u16, b: *const u16, len: usize) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let chunks = len / 32;
        let mut acc = _mm512_setzero_ps();
        for i in 0..chunks {
            let off = i * 32;
            let va = _mm512_loadu_si512(a.add(off) as *const _);
            let vb = _mm512_loadu_si512(b.add(off) as *const _);
            acc = _mm512_dpbf16_ps(acc, va.as_bf16(), vb.as_bf16());
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        // Handle remainder (< 32 elements) with scalar
        for j in (chunks * 32)..len {
            sum += bf16_to_f32(*a.add(j)) * bf16_to_f32(*b.add(j));
        }
        sum
    }
}

/// Micro-GEMM: compute score_ij = (Q[i] · K[j]) * sm_scale for a tile of (m_size × n_size) pairs.
/// Inspired by fastllm's `mul_mat_bf16_bf16_direct_avx512` template.
///
/// Writes to scores[i * score_stride + j] = (Q_row_i · K_row_j) * sm_scale.
/// Uses dpbf16ps to process 32 BF16 pairs per instruction, with inner-loop
/// register tiling: loads each K block once, reuses across multiple Q rows.
/// sm_scale is fused into the output to avoid a separate multiplication pass.
///
/// `q_stride` / `k_stride`: distance in u16 elements between consecutive rows.
/// Set both to `dim` for contiguous buffers, or to `num_heads * dim` for strided access
/// into the original tensor layout.
#[target_feature(enable = "avx512f,avx512bf16")]
pub(super) fn micro_gemm_qk_bf16(
    q: *const u16,       // [m_size rows, each `q_stride` elements apart]
    k: *const u16,       // [n_size rows, each `k_stride` elements apart]
    scores: *mut f32,    // output: [m_size * score_stride]
    m_size: usize,
    n_size: usize,
    dim: usize,
    q_stride: usize,     // row stride of Q in u16 elements
    k_stride: usize,     // row stride of K in u16 elements
    score_stride: usize, // leading dimension of scores (= block_n)
    sm_scale: f32,       // fused into output: score *= sm_scale
) {
    use core::arch::x86_64::*;
    let chunks = dim / 32;

    // Process Q rows in groups of 4 (register tile: 4 Q × n_size K × chunks)
    let mut i = 0;
    unsafe {
        while i + 4 <= m_size {
            let mut j = 0;
            while j + 4 <= n_size {
                // 4×4 tile: acc[qi][kj] for qi=0..4, kj=0..4
                let mut acc = [[_mm512_setzero_ps(); 4]; 4];
                for c in 0..chunks {
                    let off = c * 32;
                    let q0 = _mm512_loadu_si512(q.add((i + 0) * q_stride + off) as *const _);
                    let q1 = _mm512_loadu_si512(q.add((i + 1) * q_stride + off) as *const _);
                    let q2 = _mm512_loadu_si512(q.add((i + 2) * q_stride + off) as *const _);
                    let q3 = _mm512_loadu_si512(q.add((i + 3) * q_stride + off) as *const _);
                    for kk in 0..4 {
                        let kv = _mm512_loadu_si512(k.add((j + kk) * k_stride + off) as *const _);
                        let kb = kv.as_bf16();
                        acc[0][kk] = _mm512_dpbf16_ps(acc[0][kk], q0.as_bf16(), kb);
                        acc[1][kk] = _mm512_dpbf16_ps(acc[1][kk], q1.as_bf16(), kb);
                        acc[2][kk] = _mm512_dpbf16_ps(acc[2][kk], q2.as_bf16(), kb);
                        acc[3][kk] = _mm512_dpbf16_ps(acc[3][kk], q3.as_bf16(), kb);
                    }
                }
                // Reduce + store with fused sm_scale
                for qi in 0..4 {
                    for kk in 0..4 {
                        let mut sum = _mm512_reduce_add_ps(acc[qi][kk]);
                        for r in (chunks * 32)..dim {
                            sum += bf16_to_f32(*q.add((i + qi) * q_stride + r))
                                * bf16_to_f32(*k.add((j + kk) * k_stride + r));
                        }
                        *scores.add((i + qi) * score_stride + j + kk) = sum * sm_scale;
                    }
                }
                j += 4;
            }
            // Remaining K columns (< 4)
            while j < n_size {
                for qi in 0..4 {
                    let mut acc = _mm512_setzero_ps();
                    for c in 0..chunks {
                        let off = c * 32;
                        let qv = _mm512_loadu_si512(q.add((i + qi) * q_stride + off) as *const _);
                        let kv = _mm512_loadu_si512(k.add(j * k_stride + off) as *const _);
                        acc = _mm512_dpbf16_ps(acc, qv.as_bf16(), kv.as_bf16());
                    }
                    let mut sum = _mm512_reduce_add_ps(acc);
                    for r in (chunks * 32)..dim {
                        sum += bf16_to_f32(*q.add((i + qi) * q_stride + r))
                            * bf16_to_f32(*k.add(j * k_stride + r));
                    }
                    *scores.add((i + qi) * score_stride + j) = sum * sm_scale;
                }
                j += 1;
            }
            i += 4;
        }
        // Remaining Q rows (< 4)
        while i < m_size {
            for j in 0..n_size {
                let mut acc = _mm512_setzero_ps();
                for c in 0..chunks {
                    let off = c * 32;
                    let qv = _mm512_loadu_si512(q.add(i * q_stride + off) as *const _);
                    let kv = _mm512_loadu_si512(k.add(j * k_stride + off) as *const _);
                    acc = _mm512_dpbf16_ps(acc, qv.as_bf16(), kv.as_bf16());
                }
                let mut sum = _mm512_reduce_add_ps(acc);
                for r in (chunks * 32)..dim {
                    sum += bf16_to_f32(*q.add(i * q_stride + r))
                        * bf16_to_f32(*k.add(j * k_stride + r));
                }
                *scores.add(i * score_stride + j) = sum * sm_scale;
            }
            i += 1;
        }
    }
}

/// Trait to cast __m512i to __m512bh for dpbf16ps intrinsic.
pub(super) trait AsBf16 {
    fn as_bf16(self) -> core::arch::x86_64::__m512bh;
}

impl AsBf16 for core::arch::x86_64::__m512i {
    #[inline(always)]
    fn as_bf16(self) -> core::arch::x86_64::__m512bh {
        unsafe { std::mem::transmute(self) }
    }
}

/// acc[d] += w * src_bf16[d] — weighted accumulation from BF16 source into F32 accumulator.
/// Uses dpbf16ps when available: broadcasts w as BF16 pair, multiplies with src.
#[inline]
pub(super) fn fma_bf16_f32_native(acc: &mut [f32], src: &[u16], w: f32, len: usize) {
    if is_x86_feature_detected!("avx512bf16") {
        unsafe {
            fma_bf16_f32_dpbf16ps(acc.as_mut_ptr(), src.as_ptr(), w, len);
        }
        return;
    }
    // Fallback: convert + fma
    for d in 0..len {
        acc[d] += w * bf16_to_f32(src[d]);
    }
}

/// AVX-512 BF16: acc[d] += w * src_bf16[d] using dpbf16ps.
/// We can't directly use dpbf16ps for scalar×vector, so we convert src to f32 and use fmadd.
/// But we still benefit from the faster bf16 load path (32 elements at a time via bf16x16_load_as_f32).
#[target_feature(enable = "avx512f,avx512bw")]
fn fma_bf16_f32_dpbf16ps(acc: *mut f32, src: *const u16, w: f32, len: usize) {
    use core::arch::x86_64::*;
    unsafe {
        let vw = _mm512_set1_ps(w);
        let chunks = len / 16;
        for i in 0..chunks {
            let off = i * 16;
            let v_src = super::common::bf16x16_load_as_f32(src.add(off));
            let v_acc = _mm512_loadu_ps(acc.add(off));
            let v_res = _mm512_fmadd_ps(vw, v_src, v_acc);
            _mm512_storeu_ps(acc.add(off), v_res);
        }
        for d in (chunks * 16)..len {
            *acc.add(d) += w * bf16_to_f32(*src.add(d));
        }
    }
}
