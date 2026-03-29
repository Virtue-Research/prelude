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
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

// ── Include split modules ──────────────────────────────────────────

#include "common.cuh"
#include "sm90_bf16.cuh"
#include "sm90_fp8.cuh"
#include "sm100_bf16.cuh"
#include "sm100_fp8.cuh"
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
    return sm90_m_grouped_fp8_gemm(A, B, D, scale_a, scale_b, grouped_layout,
                                    M, N, K, num_groups, stream);
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
    return sm90_m_grouped_masked_fp8_gemm(A, B, D, scale_a, scale_b, masked_m,
                                           M, N, K, num_groups, expected_m, stream);
}

} // extern "C"
