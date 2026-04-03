// DeepGEMM BF16/FP8 GEMM wrapper for Rust FFI.
// AOT-compiled kernel variants + runtime heuristic + TMA descriptor creation.
//
// Based on deepseek-ai/DeepGEMM (MIT license).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include <deep_gemm/impls/sm90_bf16_gemm.cuh>
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

// ── Include split modules ──────────────────────────────────────────

#include "common.cuh"
#include "layout.cuh"
#include "sm90_bf16.cuh"
#include "sm90_fp8.cuh"
#include "sm100_bf16.cuh"
#include "sm100_fp8.cuh"
#include "einsum.cuh"
#include "attention.cuh"

// ── C FFI ───────────────────────────────────────────────────────────

extern "C" {

/// BF16 GEMM: D = A[M,K] @ B[K,N]
/// Auto-dispatches to SM100 (Blackwell) when detected.
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_bf16_gemm(
    void* A, void* B, void* D,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_bf16_gemm(A, B, D, M, N, K, stream);
    return sm90_bf16_gemm(A, B, D, M, N, K, stream);
}

/// BF16 GEMM with FP32 accumulation: D(FP32) += A(BF16) @ B(BF16)
/// C: optional FP32 bias. If non-null and != D, copied to D before launch.
/// D: [M,N] FP32 output (accumulated in-place).
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_bf16_gemm_acc(
    void* A, void* B, void* C, void* D,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    return sm90_bf16_gemm_acc(A, B, C, D, M, N, K, stream);
}

/// Query BF16 kernel config for a given shape.
/// Auto-dispatches to SM100 when detected.
void deepgemm_query_config(int M, int N, int K,
                            int* out_block_m, int* out_block_n,
                            int* out_stages, int* out_smem) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100) {
        auto cfg = select_sm100_config(M, N, K, g_num_sms);
        *out_block_m = cfg.block_m; *out_block_n = cfg.block_n;
        *out_stages = cfg.num_stages; *out_smem = cfg.smem_size;
        return;
    }
    auto cfg = select_config(M, N, K, g_num_sms);
    *out_block_m = cfg.block_m;
    *out_block_n = cfg.block_n;
    *out_stages = cfg.num_stages;
    *out_smem = cfg.smem_size;
}

/// FP8 E4M3 GEMM (1D2D): D[M,N] = (scale_a ⊗ A_fp8[M,K]) @ (scale_b ⊗ B_fp8[K,N])
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_fp8_gemm(A, B, D, scale_a, scale_b, M, N, K, stream);
    return sm90_fp8_gemm(A, B, D, scale_a, scale_b, M, N, K, stream);
}

/// Query FP8 kernel config for a given shape.
void deepgemm_query_fp8_config(int M, int N, int K,
                                int* out_block_m, int* out_block_n,
                                int* out_stages, int* out_smem) {
    ensure_num_sms();
    auto cfg = select_fp8_config(M, N, K, g_num_sms);
    *out_block_m = cfg.block_m;
    *out_block_n = cfg.block_n;
    *out_stages = cfg.num_stages;
    *out_smem = cfg.smem_size;
}

/// M-Grouped Contiguous BF16 GEMM (MoE):
///   D[total_M, N] = grouped(A[total_M, K], B[G, N, K], grouped_layout[total_M])
/// Auto-dispatches to SM100 when detected.
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_m_grouped_bf16_gemm(
    void* A, void* B, void* D,
    void* grouped_layout,
    int M, int N, int K,
    int num_groups,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_m_grouped_bf16_gemm(A, B, D, grouped_layout, M, N, K, num_groups, stream);
    return sm90_m_grouped_bf16_gemm(A, B, D, grouped_layout, M, N, K, num_groups, stream);
}

/// M-Grouped Contiguous FP8 GEMM (1D2D, MoE):
///   D[total_M, N] = grouped(scale_a ⊗ A_fp8[total_M, K], scale_b ⊗ B_fp8[G, N, K])
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_m_grouped_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    void* grouped_layout,
    int M, int N, int K,
    int num_groups,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_m_grouped_fp8_gemm(A, B, D, scale_a, scale_b, grouped_layout,
                                         M, N, K, num_groups, stream);
    return sm90_m_grouped_fp8_gemm(A, B, D, scale_a, scale_b, grouped_layout,
                                    M, N, K, num_groups, stream);
}

/// FP8 E4M3 1D1D GEMM: D(FP32) = A(FP8) @ B(FP8) with per-block scaling on both.
/// scale_a, scale_b: both via TMA (per 128-channel block).
/// Output D is FP32 (not BF16).
int deepgemm_fp8_gemm_1d1d(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    return sm90_fp8_gemm_1d1d(A, B, D, scale_a, scale_b, M, N, K, stream);
}

/// Query grouped GEMM config for a given shape.
void deepgemm_query_grouped_config(int M, int N, int K,
                                    int* out_block_m, int* out_block_n,
                                    int* out_stages, int* out_smem) {
    ensure_num_sms();
    auto cfg = select_grouped_config(M, N, K, g_num_sms);
    *out_block_m = cfg.block_m;
    *out_block_n = cfg.block_n;
    *out_stages = cfg.num_stages;
    *out_smem = cfg.smem_size;
}

/// M-Grouped Masked BF16 GEMM:
///   D[G,M,N] = masked(A[G,M,K], B[G,N,K], masked_m[G])
/// A: [G,M,K] row-major BF16 (per-group, K-major for TMA)
/// B: [G,N,K] row-major BF16 (per-group, K-major for TMA)
/// D: [G,M,N] row-major BF16 output
/// masked_m: [G] int32, actual number of valid rows per group
/// M: per-group padded M, N: shared N, K: shared K
/// expected_m: expected per-group M for heuristic (e.g., average actual M)
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_m_grouped_masked_bf16_gemm(
    void* A, void* B, void* D,
    void* masked_m,
    int M, int N, int K,
    int num_groups,
    int expected_m,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_m_grouped_masked_bf16_gemm(A, B, D, masked_m,
                                                 M, N, K, num_groups, expected_m, stream);
    return sm90_m_grouped_masked_bf16_gemm(A, B, D, masked_m,
                                            M, N, K, num_groups, expected_m, stream);
}

/// M-Grouped Masked FP8 GEMM (1D2D):
///   D[G,M,N] = masked(scale_a ⊗ A_fp8[G,M,K], scale_b ⊗ B_fp8[G,N,K], masked_m[G])
/// A: [G,M,K] FP8 E4M3
/// B: [G,N,K] FP8 E4M3
/// D: [G,M,N] BF16 output
/// scale_a: [G, ceil(K/128), align(M,4)] FP32 (per-token, via TMA)
/// scale_b: [G, ceil(K/128), align(N,4)] FP32 (per-channel, global memory)
/// masked_m: [G] int32
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_m_grouped_masked_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    void* masked_m,
    int M, int N, int K,
    int num_groups,
    int expected_m,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_m_grouped_masked_fp8_gemm(A, B, D, scale_a, scale_b, masked_m,
                                                M, N, K, num_groups, expected_m, stream);
    return sm90_m_grouped_masked_fp8_gemm(A, B, D, scale_a, scale_b, masked_m,
                                           M, N, K, num_groups, expected_m, stream);
}

/// FP8 MQA Logits (prefill):
///   logits[seq_len, stride_logits] = weighted_relu_sum(
///     Q[seq_len*num_heads, head_dim] @ KV[seq_len_kv, head_dim]^T, weights)
/// Auto-dispatches SM90/SM100.
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_fp8_mqa_logits(
    void* q, void* kv, void* kv_scales, void* weights,
    void* cu_seq_len_k_start, void* cu_seq_len_k_end,
    void* logits,
    int seq_len, int seq_len_kv, int max_seqlen_k,
    int num_heads, int head_dim, int stride_logits,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_fp8_mqa_logits_launch(q, kv, kv_scales, weights,
            cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len, seq_len_kv, max_seqlen_k,
            num_heads, head_dim, stride_logits, stream);
    return sm90_fp8_mqa_logits_launch(q, kv, kv_scales, weights,
        cu_seq_len_k_start, cu_seq_len_k_end, logits,
        seq_len, seq_len_kv, max_seqlen_k,
        num_heads, head_dim, stride_logits, stream);
}

/// FP8 Paged MQA Logits (decode):
///   logits[batch_size, stride_logits] = paged_attention(Q, KV_cache, block_table)
/// Auto-dispatches SM90/SM100.
int deepgemm_fp8_paged_mqa_logits(
    void* q, void* kv_cache, void* kv_scales, void* weights,
    void* context_lens, void* logits, void* block_table, void* schedule_meta,
    int batch_size, int num_heads, int head_dim,
    int num_kv_blocks, int block_kv,
    int is_context_lens_2d,
    int kv_cache_stride_bytes, int logits_stride, int block_table_stride,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();
    bool cl2d = (is_context_lens_2d != 0);
    if (g_gpu_arch >= 100)
        return sm100_fp8_paged_mqa_logits_launch(q, kv_cache, kv_scales, weights,
            context_lens, logits, block_table, schedule_meta,
            batch_size, num_heads, head_dim, num_kv_blocks, block_kv,
            cl2d, kv_cache_stride_bytes, logits_stride, block_table_stride, stream);
    return sm90_fp8_paged_mqa_logits_launch(q, kv_cache, kv_scales, weights,
        context_lens, logits, block_table, schedule_meta,
        batch_size, num_heads, head_dim, num_kv_blocks, block_kv,
        cl2d, kv_cache_stride_bytes, logits_stride, block_table_stride, stream);
}

/// Compute scheduling metadata for paged MQA logits.
/// schedule_metadata: GPU buffer [(num_sms+1) * 2] uint32.
int deepgemm_paged_mqa_metadata(
    void* context_lens, void* schedule_metadata,
    int batch_size, int next_n, int is_context_lens_2d,
    int split_kv, int num_sms,
    void* stream
) {
    return paged_mqa_metadata_launch(context_lens, schedule_metadata,
        batch_size, next_n, (is_context_lens_2d != 0), split_kv, num_sms, stream);
}

/// Clean logits: fill -inf for out-of-range KV positions.
int deepgemm_clean_logits(
    void* cu_seq_len_k_start, void* cu_seq_len_k_end,
    void* logits,
    int seq_len, int seq_len_kv, int stride_logits,
    int next_n,
    void* stream
) {
    ensure_num_sms();
    return clean_logits_launch(cu_seq_len_k_start, cu_seq_len_k_end,
        logits, seq_len, seq_len_kv, stride_logits, next_n, stream);
}

/// Transform scaling factors: transpose from [G,MN,K/128] to MN-major TMA-aligned.
int deepgemm_transform_sf_transpose(
    void* sf_in, void* sf_out,
    int mn, int sf_k, int num_groups,
    void* stream
) {
    return transform_sf_transpose(sf_in, sf_out, mn, sf_k, num_groups, stream);
}

/// Transform + pack scaling factors to UE8M0 format (for SM100).
int deepgemm_transform_sf_pack_ue8m0(
    void* sf_in, void* sf_out,
    int mn, int sf_k, int num_groups,
    void* stream
) {
    return transform_sf_pack_ue8m0(sf_in, sf_out, mn, sf_k, num_groups, stream);
}

/// Get TMA-aligned size for a given element count and element size.
int deepgemm_get_tma_aligned_size(int size, int elem_size) {
    return get_tma_aligned_size_local(size, elem_size);
}

/// Get M/K alignment for contiguous grouped layout. Always 128.
int deepgemm_get_mk_alignment() {
    return 128;
}

/// BF16 Einsum: D[M,N] = sum_s A[s,M,K] @ B[s,N,K]^T
/// A: [shape_s * shape_m, shape_k] BF16, B: [shape_s * shape_n, shape_k] BF16
/// D: [shape_m, shape_n] FP32 (must be zero-initialized, accumulated via atomicAdd)
/// Auto-dispatches SM90/SM100. shape_m/n/k must match a pre-compiled configuration.
/// Returns 0 on success, -1 if no kernel, -2 on launch error.
int deepgemm_einsum(
    void* A, void* B, void* D,
    int shape_m, int shape_n, int shape_k, int shape_s,
    void* stream
) {
    ensure_gpu_arch();
    if (g_gpu_arch >= 100)
        return sm100_einsum_launch(A, B, D, shape_m, shape_n, shape_k, shape_s, stream);
    return sm90_einsum_launch(A, B, D, shape_m, shape_n, shape_k, shape_s, stream);
}

/// Query number of SMs and GPU arch. Useful for paged MQA metadata.
void deepgemm_query_device(int* out_num_sms, int* out_gpu_arch) {
    ensure_num_sms();
    ensure_gpu_arch();
    *out_num_sms = g_num_sms;
    *out_gpu_arch = g_gpu_arch;
}

} // extern "C"
