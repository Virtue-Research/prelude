/******************************************************************************
 * Q RMSNorm + RoPE prologue for FA3 SMEM Q tiles (Plan A).
 * Applies per-row norm+rope in shared memory after TMA Q load, before Q@K.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockMN, int kHeadDim, int NumThreads, int kBlockH, typename Element>
struct QNormRotaryPrologue {
    static constexpr int kBlockM_div_H = kBlockMN / kBlockH;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Head dim must divide vector load width");
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must divide evenly per row");

    using LayoutAtom = Layout<Shape<Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;
    using TiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    int const thread_idx;
    TiledCopyQK tiled_copy_q;

    CUTLASS_DEVICE
    QNormRotaryPrologue(int const thread_idx_) : thread_idx(thread_idx_), tiled_copy_q{} {}

    template <bool GqaPacked>
    CUTLASS_DEVICE int max_rows(int const seqlen_q, int const m_block) const {
        if constexpr (GqaPacked) {
            return std::min(seqlen_q * kBlockH - m_block * kBlockMN, kBlockMN);
        } else {
            return std::min(seqlen_q - m_block * kBlockMN, kBlockMN);
        }
    }

    template <bool GqaPacked>
    CUTLASS_DEVICE int token_index(int const row, int const cu_seq_start, int const m_block) const {
        if constexpr (GqaPacked) {
            return cu_seq_start + m_block * kBlockM_div_H + row / kBlockH;
        } else {
            return cu_seq_start + m_block * kBlockMN + row;
        }
    }

    template <bool GqaPacked, typename TensorsQ>
    CUTLASS_DEVICE void apply_fused_norm_rope(
        TensorsQ &sQ,
        Element const *q_weight,
        float eps,
        Element const *ptr_cos,
        Element const *ptr_sin,
        int const *position_ids,
        int cu_seq_start,
        int m_block,
        int seqlen_q) {
        int const half_d = kHeadDim / 2;
        int const max_rows_val = max_rows<GqaPacked>(seqlen_q, m_block);

        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor sQ_copy = cute::tiled_divide(sQ, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));

        #pragma unroll
        for (int m = 0; m < size<1>(tQcQ); ++m) {
            int const row = get<0>(tQcQ(_0{}, m, _0{}));
            if (row >= max_rows_val) { continue; }

            float ss = 0.f;
            #pragma unroll
            for (int k = 0; k < size<2>(tQcQ); ++k) {
                int const col = get<1>(tQcQ(_0{}, _0{}, k));
                if (col < kHeadDim) {
                    int const col_idx = col / kGmemElemsPerLoad;
                    Tensor rQ = make_fragment_like(sQ_copy(_, row, col_idx));
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx), rQ);
                    #pragma unroll
                    for (int i = 0; i < size(rQ); ++i) {
                        float v = float(rQ(i));
                        ss += v * v;
                    }
                }
            }

            #pragma unroll
            for (int offset = kGmemThreadsPerRow / 2; offset > 0; offset /= 2) {
                ss += __shfl_down_sync(0xffffffff, ss, offset);
            }
            if ((thread_idx % kGmemThreadsPerRow) == 0) {
                ss = __shfl_sync(0xffffffff, ss, thread_idx, kGmemThreadsPerRow);
            } else {
                ss = __shfl_sync(0xffffffff, ss, thread_idx - (thread_idx % kGmemThreadsPerRow), kGmemThreadsPerRow);
            }
            float scale = rsqrtf(ss / float(kHeadDim) + eps);
            int const pos = position_ids[token_index<GqaPacked>(row, cu_seq_start, m_block)];

            #pragma unroll
            for (int k = 0; k < size<2>(tQcQ); ++k) {
                int const col = get<1>(tQcQ(_0{}, _0{}, k));
                if (col < half_d) {
                    int const col_idx_left = col / kGmemElemsPerLoad;
                    int const col_idx_right = col / kGmemElemsPerLoad + half_d / kGmemElemsPerLoad;
                    Tensor rQ_left = make_fragment_like(sQ_copy(_, row, col_idx_left));
                    Tensor rQ_right = make_fragment_like(rQ_left);
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_left), rQ_left);
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_right), rQ_right);

                    #pragma unroll
                    for (int i = 0; i < size(rQ_left); ++i) {
                        float left = float(rQ_left(i)) * scale * float(q_weight[col + i]);
                        float right = float(rQ_right(i)) * scale * float(q_weight[col + half_d + i]);
                        float c = float(ptr_cos[pos * half_d + col + i]);
                        float sn = float(ptr_sin[pos * half_d + col + i]);
                        float real = left * c - right * sn;
                        float imag = left * sn + right * c;
                        rQ_left(i) = Element(real);
                        rQ_right(i) = Element(imag);
                    }
                    cute::copy(tiled_copy_q, rQ_left, sQ_copy(_, row, col_idx_left));
                    cute::copy(tiled_copy_q, rQ_right, sQ_copy(_, row, col_idx_right));
                }
            }
        }
    }

    template <bool GqaPacked, typename TensorsQ>
    CUTLASS_DEVICE void apply_rmsnorm(TensorsQ &sQ,
                                      Element const *q_weight,
                                      float eps,
                                      int const seqlen_q,
                                      int const m_block) {
        int const max_rows_val = max_rows<GqaPacked>(seqlen_q, m_block);
        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor sQ_copy = cute::tiled_divide(sQ, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));

        #pragma unroll
        for (int m = 0; m < size<1>(tQcQ); ++m) {
            int const row = get<0>(tQcQ(_0{}, m, _0{}));
            if (row >= max_rows_val) { continue; }

            float ss = 0.f;
            #pragma unroll
            for (int k = 0; k < size<2>(tQcQ); ++k) {
                int const col = get<1>(tQcQ(_0{}, _0{}, k));
                if (col < kHeadDim) {
                    int const col_idx = col / kGmemElemsPerLoad;
                    Tensor rQ = make_fragment_like(sQ_copy(_, row, col_idx));
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx), rQ);
                    #pragma unroll
                    for (int i = 0; i < size(rQ); ++i) {
                        float v = float(rQ(i));
                        ss += v * v;
                    }
                }
            }

            #pragma unroll
            for (int offset = kGmemThreadsPerRow / 2; offset > 0; offset /= 2) {
                ss += __shfl_down_sync(0xffffffff, ss, offset);
            }
            if ((thread_idx % kGmemThreadsPerRow) == 0) {
                ss = __shfl_sync(0xffffffff, ss, thread_idx, kGmemThreadsPerRow);
            } else {
                ss = __shfl_sync(0xffffffff, ss, thread_idx - (thread_idx % kGmemThreadsPerRow), kGmemThreadsPerRow);
            }
            float scale = rsqrtf(ss / float(kHeadDim) + eps);

            #pragma unroll
            for (int k = 0; k < size<2>(tQcQ); ++k) {
                int const col = get<1>(tQcQ(_0{}, _0{}, k));
                if (col < kHeadDim) {
                    int const col_idx = col / kGmemElemsPerLoad;
                    Tensor rQ = make_fragment_like(sQ_copy(_, row, col_idx));
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx), rQ);
                    #pragma unroll
                    for (int i = 0; i < size(rQ); ++i) {
                        float v = float(rQ(i)) * scale * float(q_weight[col + i]);
                        rQ(i) = Element(v);
                    }
                    cute::copy(tiled_copy_q, rQ, sQ_copy(_, row, col_idx));
                }
            }
        }
    }

    template <bool GqaPacked, typename TensorsQ>
    CUTLASS_DEVICE void apply_rope_with_position_ids(
        TensorsQ &sQ,
        Element const *ptr_cos,
        Element const *ptr_sin,
        int const *position_ids,
        int cu_seq_start,
        int m_block,
        int seqlen_q) {
        int const half_d = kHeadDim / 2;
        int const max_rows_val = max_rows<GqaPacked>(seqlen_q, m_block);

        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor sQ_copy = cute::tiled_divide(sQ, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));

        #pragma unroll
        for (int m = 0; m < size<1>(tQcQ); ++m) {
            int const row = get<0>(tQcQ(_0{}, m, _0{}));
            if (row >= max_rows_val) { continue; }

            int const pos = position_ids[token_index<GqaPacked>(row, cu_seq_start, m_block)];

            #pragma unroll
            for (int k = 0; k < size<2>(tQcQ); ++k) {
                int const col = get<1>(tQcQ(_0{}, _0{}, k));
                if (col < half_d) {
                    int const col_idx_left = col / kGmemElemsPerLoad;
                    int const col_idx_right = col / kGmemElemsPerLoad + half_d / kGmemElemsPerLoad;
                    Tensor rQ_left = make_fragment_like(sQ_copy(_, row, col_idx_left));
                    Tensor rQ_right = make_fragment_like(rQ_left);
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_left), rQ_left);
                    cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_right), rQ_right);

                    #pragma unroll
                    for (int i = 0; i < size(rQ_left); ++i) {
                        float c = float(ptr_cos[pos * half_d + col + i]);
                        float sn = float(ptr_sin[pos * half_d + col + i]);
                        float real = float(rQ_left(i)) * c - float(rQ_right(i)) * sn;
                        float imag = float(rQ_left(i)) * sn + float(rQ_right(i)) * c;
                        rQ_left(i) = Element(real);
                        rQ_right(i) = Element(imag);
                    }
                    cute::copy(tiled_copy_q, rQ_left, sQ_copy(_, row, col_idx_left));
                    cute::copy(tiled_copy_q, rQ_right, sQ_copy(_, row, col_idx_right));
                }
            }
        }
    }

    template <bool GqaPacked, typename TensorsQ>
    CUTLASS_DEVICE void apply(TensorsQ &sQ_pi,
                              Element const *q_weight,
                              float eps,
                              Element const *ptr_cos,
                              Element const *ptr_sin,
                              int const *position_ids,
                              int cu_seq_start,
                              int m_block,
                              int seqlen_q) {
        apply_rmsnorm<GqaPacked>(sQ_pi, q_weight, eps, seqlen_q, m_block);
        cutlass::arch::fence_view_async_shared();
        apply_rope_with_position_ids<GqaPacked>(
            sQ_pi, ptr_cos, ptr_sin, position_ids, cu_seq_start, m_block, seqlen_q);
        cutlass::arch::fence_view_async_shared();
    }
};

}  // namespace flash
