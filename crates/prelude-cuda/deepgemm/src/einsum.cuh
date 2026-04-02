// BF16 Einsum: D[M,N] = sum_s A[s,M,K] @ B[s,N,K]^T  (batched matmul with accumulation)
//
// Based on deepseek-ai/DeepGEMM sm90_bmk_bnk_mn.cuh / sm100_bmk_bnk_mn.cuh.
//
// Template params SHAPE_M, SHAPE_N, SHAPE_K must match the actual per-batch dims.
// The batch dimension `shape_s` is runtime.
// Output D is FP32, accumulated via atomicAdd — caller must zero-init D.

#pragma once

#include <deep_gemm/impls/sm90_bmk_bnk_mn.cuh>
#include <deep_gemm/impls/sm100_bmk_bnk_mn.cuh>

using namespace deep_gemm;

// ── Einsum kernel parameters ──────────────────────────────────────

struct EinsumKernelInfo {
    int shape_m, shape_n, shape_k;
    int block_n;
    int split_factor;
    int num_stages;
    int num_threads;  // tma + math
    const void* kernel;
};

// SM90: BLOCK_M=128, BLOCK_K=64, TMA=128, MATH=256 (fixed by kernel asserts)
// BLOCK_N = SHAPE_N for SHAPE_N <= 128, else 128
// kSplitFactor: set to 16 (good default for split-K accumulation)

static int compute_einsum_stages_sm90(int block_n) {
    // smem per stage = BLOCK_M * BLOCK_K * 2 + BLOCK_N * BLOCK_K * 2
    int smem_a = 128 * 64 * 2; // BF16
    int smem_b = block_n * 64 * 2;
    int barrier = 16; // 8 * 2
    int cap = 232448;
    for (int s = 32; s > 0; s--) {
        if (s * (smem_a + smem_b) + s * barrier <= cap)
            return s;
    }
    return 1;
}

// ── SM90 Einsum kernel instantiation ──────────────────────────────
// Pre-compiled for common (M, N, K) configurations used in inference.

#define SM90_EINSUM(M, N, K, BN, SF, ST) \
    __attribute__((used)) static auto* _sm90_ein_##M##_##N##_##K = \
        &sm90_bmn_bnk_mn_gemm_impl<M, N, K, 128, BN, 64, SF, ST, 128, 256>;

// (128, 128, 64): attention scoring with 128 heads, head_dim=64
SM90_EINSUM(128, 128, 64, 128, 16, 7)
// (128, 128, 128): attention with head_dim=128
SM90_EINSUM(128, 128, 128, 128, 16, 7)
// (128, 64, 64): smaller N
SM90_EINSUM(128, 64, 64, 64, 16, 9)
// (128, 64, 128): smaller N, larger K
SM90_EINSUM(128, 64, 128, 64, 16, 9)
// (256, 128, 64): larger M
SM90_EINSUM(256, 128, 64, 128, 16, 7)
// (256, 128, 128): larger M, larger K
SM90_EINSUM(256, 128, 128, 128, 16, 7)

#undef SM90_EINSUM

static const void* get_sm90_einsum_kernel(int m, int n, int k, int& block_n,
                                           int& split_factor, int& stages, int& threads) {
    #define M(SM, SN, SK, BN, SF, ST) \
        if (m == SM && n == SN && k == SK) { \
            block_n = BN; split_factor = SF; stages = ST; threads = 384; \
            return (const void*)&sm90_bmn_bnk_mn_gemm_impl<SM, SN, SK, 128, BN, 64, SF, ST, 128, 256>; \
        }
    M(128, 128, 64,  128, 16, 7)
    M(128, 128, 128, 128, 16, 7)
    M(128, 64,  64,  64,  16, 9)
    M(128, 64,  128, 64,  16, 9)
    M(256, 128, 64,  128, 16, 7)
    M(256, 128, 128, 128, 16, 7)
    #undef M
    return nullptr;
}

// ── SM100 Einsum kernel instantiation ─────────────────────────────
// SM100: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 (all fixed), swizzle=128

static int compute_einsum_stages_sm100() {
    // smem: CD (2 stages) + A/B per stage
    int smem_cd = 128 * 128 * 2; // 2 TMA store stages
    int smem_a = 128 * 64 * 2;
    int smem_b = 128 * 64 * 2;
    int barrier = 24 + 8; // per-stage barriers + tmem
    int cap = 232448;
    for (int s = 32; s > 0; s--) {
        if (smem_cd + s * (smem_a + smem_b) + s * barrier <= cap)
            return s;
    }
    return 1;
}

// SM100 only supports SHAPE_N divisible by BLOCK_N=128
#define SM100_EINSUM(M, N, K, SF, ST) \
    __attribute__((used)) static auto* _sm100_ein_##M##_##N##_##K = \
        &sm100_bmn_bnk_mn_gemm_impl<M, N, K, 128, 128, 64, SF, 128, 128, ST, 128>;

SM100_EINSUM(128, 128, 64,  16, 6)
SM100_EINSUM(128, 128, 128, 16, 6)
SM100_EINSUM(256, 128, 64,  16, 6)
SM100_EINSUM(256, 128, 128, 16, 6)

#undef SM100_EINSUM

static const void* get_sm100_einsum_kernel(int m, int n, int k, int& split_factor,
                                            int& stages, int& threads) {
    #define M(SM, SN, SK, SF, ST) \
        if (m == SM && n == SN && k == SK) { \
            split_factor = SF; stages = ST; threads = 128; \
            return (const void*)&sm100_bmn_bnk_mn_gemm_impl<SM, SN, SK, 128, 128, 64, SF, 128, 128, ST, 128>; \
        }
    M(128, 128, 64,  16, 6)
    M(128, 128, 128, 16, 6)
    M(256, 128, 64,  16, 6)
    M(256, 128, 128, 16, 6)
    #undef M
    return nullptr;
}

// ── TMA descriptor creation for einsum ────────────────────────────

struct EinsumTMA {
    CUtensorMap tma_a;  // A: [shape_s * SHAPE_M, SHAPE_K] BF16
    CUtensorMap tma_b;  // B: [shape_s * SHAPE_N, SHAPE_K] BF16
};

static EinsumTMA make_einsum_tma_sm90(void* a, void* b,
                                       int shape_m, int shape_n, int shape_k,
                                       int shape_s, int block_n) {
    EinsumTMA tma;
    int swizzle = 64 * 2; // BLOCK_K * sizeof(bf16) = 128 bytes
    // A: [SHAPE_K, shape_s * SHAPE_M], tile [BLOCK_K=64, BLOCK_M=128]
    tma.tma_a = make_2d_tma(a, shape_k, shape_s * shape_m,
                             64, 128, shape_k, swizzle);
    // B: [SHAPE_K, shape_s * SHAPE_N], tile [BLOCK_K=64, BLOCK_N]
    tma.tma_b = make_2d_tma(b, shape_k, shape_s * shape_n,
                             64, block_n, shape_k, swizzle);
    return tma;
}

static EinsumTMA make_einsum_tma_sm100(void* a, void* b,
                                        int shape_m, int shape_n, int shape_k,
                                        int shape_s) {
    EinsumTMA tma;
    int swizzle = 128; // fixed for SM100
    tma.tma_a = make_2d_tma(a, shape_k, shape_s * shape_m,
                             64, 128, shape_k, swizzle);
    tma.tma_b = make_2d_tma(b, shape_k, shape_s * shape_n,
                             64, 128, shape_k, swizzle);
    return tma;
}

// ── Einsum launch ─────────────────────────────────────────────────

static int sm90_einsum_launch(
    void* a, void* b, void* d,
    int shape_m, int shape_n, int shape_k, int shape_s,
    void* stream
) {
    int block_n, split_factor, stages, threads;
    auto kp = get_sm90_einsum_kernel(shape_m, shape_n, shape_k,
                                      block_n, split_factor, stages, threads);
    if (!kp) return -1;

    auto tma = make_einsum_tma_sm90(a, b, shape_m, shape_n, shape_k, shape_s, block_n);

    int k_blocks = shape_k / 64;
    int total_sk = shape_s * k_blocks;
    int num_sk_blocks = ceil_div_static(total_sk, split_factor);
    int num_mn_blocks = ceil_div_static(shape_m, 128) * ceil_div_static(shape_n, block_n);
    int total_blocks = num_mn_blocks * num_sk_blocks;

    int smem = stages * (128 * 64 * 2 + block_n * 64 * 2) + stages * 16;

    uint32_t ss = shape_s;
    void* args[] = {&ss, &tma.tma_a, &tma.tma_b, &d};

    // Launch with specific grid size (not num_sms)
    dim3 grid(total_blocks); dim3 block(threads);
    cudaFuncSetAttribute(kp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (cudaLaunchKernel(kp, grid, block, args, smem, (cudaStream_t)stream) != cudaSuccess) {
        cudaGetLastError(); return -2;
    }
    return 0;
}

static int sm100_einsum_launch(
    void* a, void* b, void* d,
    int shape_m, int shape_n, int shape_k, int shape_s,
    void* stream
) {
    int split_factor, stages, threads;
    auto kp = get_sm100_einsum_kernel(shape_m, shape_n, shape_k,
                                       split_factor, stages, threads);
    if (!kp) return -1;

    auto tma = make_einsum_tma_sm100(a, b, shape_m, shape_n, shape_k, shape_s);

    // SM100 also needs a TMA descriptor for D output (TMA reduce-add store)
    CUtensorMap tma_d;
    {
        ensure_driver_api();
        int swizzle = 128;
        int smem_inner = swizzle / 4; // FP32 element size
        CUtensorMap tmap{};
        cuuint64_t gmem_dims[2] = {(cuuint64_t)shape_n, (cuuint64_t)shape_m};
        cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)128};
        cuuint64_t gmem_strides[1] = {(cuuint64_t)(shape_n * 4)};
        cuuint32_t elem_strides[2] = {1, 1};
        p_cuTensorMapEncodeTiled(
            &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2, d, gmem_dims, gmem_strides, smem_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode_to_enum(swizzle),
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        tma_d = tmap;
    }

    int k_blocks = shape_k / 64;
    int total_sk = shape_s * k_blocks;
    int num_sk_blocks = ceil_div_static(total_sk, split_factor);
    int num_mn_blocks = ceil_div_static(shape_m, 128) * ceil_div_static(shape_n, 128);
    int total_blocks = num_mn_blocks * num_sk_blocks;

    // SM100 smem: CD store stages + AB load stages + barriers + tmem ptr
    int smem_cd = 128 * 128 * 2; // 2 store stages
    int smem = smem_cd + stages * (128 * 64 * 2 + 128 * 64 * 2)
              + stages * 24 + 8 + 4;

    uint32_t ss = shape_s;
    void* args[] = {&ss, &tma.tma_a, &tma.tma_b, &tma_d};

    dim3 grid(total_blocks); dim3 block(threads);
    cudaFuncSetAttribute(kp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (cudaLaunchKernel(kp, grid, block, args, smem, (cudaStream_t)stream) != cudaSuccess) {
        cudaGetLastError(); return -2;
    }
    return 0;
}
