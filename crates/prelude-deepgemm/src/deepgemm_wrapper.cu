// DeepGEMM BF16 GEMM wrapper for Rust FFI.
// AOT-compiled kernel variants + runtime heuristic + TMA descriptor creation.
//
// Based on deepseek-ai/DeepGEMM (MIT license).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

using namespace deep_gemm;

// ── Kernel config ───────────────────────────────────────────────────

struct KernelConfig {
    int block_m, block_n, block_k;
    int num_stages;
    int num_tma_threads, num_math_threads;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

// ── TMA descriptor helpers (no torch dependency) ────────────────────

static CUresult (*p_cuTensorMapEncodeTiled)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill) = nullptr;

static void ensure_driver_api() {
    if (p_cuTensorMapEncodeTiled) return;
    CUresult r = cuGetProcAddress("cuTensorMapEncodeTiled",
                                   (void**)&p_cuTensorMapEncodeTiled, 12000,
                                   CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
    if (r != CUDA_SUCCESS || !p_cuTensorMapEncodeTiled) {
        fprintf(stderr, "DeepGEMM: cuTensorMapEncodeTiled not available\n");
    }
}

static CUtensorMapSwizzle swizzle_mode_to_enum(int mode) {
    switch (mode) {
        case 0:   return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default:  return CU_TENSOR_MAP_SWIZZLE_128B;
    }
}

static CUtensorMap make_2d_tma(void* data, int inner_dim, int outer_dim,
                                int smem_inner, int smem_outer,
                                int outer_stride, int swizzle_mode) {
    ensure_driver_api();
    const int elem_size = 2; // bf16
    if (swizzle_mode != 0)
        smem_inner = swizzle_mode / elem_size;

    CUtensorMap tmap{};
    cuuint64_t gmem_dims[2] = {(cuuint64_t)inner_dim, (cuuint64_t)outer_dim};
    cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gmem_strides[1] = {(cuuint64_t)(outer_stride * elem_size)};
    cuuint32_t elem_strides[2] = {1, 1};

    p_cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, data, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode_to_enum(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tmap;
}

// ── Heuristic (translated from DeepGEMM sm90.hpp) ──────────────────

static int get_swizzle(int block_size) {
    for (int mode : {128, 64, 32, 16})
        if ((block_size * 2) % mode == 0) return mode;
    return 16;
}

static KernelConfig select_config(int m, int n, int k, int num_sms) {
    // block_k is fixed at 64 for BF16
    const int block_k = 64;

    // block_m candidates: {64, 128, 256} + optional {16, 32} for small M
    // (matches DeepGEMM sm90.hpp get_block_m_candidates exactly)
    int block_ms[5] = {64, 128, 256, 0, 0};
    int n_block_ms = 3;
    if (m <= 16) { block_ms[n_block_ms++] = 16; }
    if (m <= 32) { block_ms[n_block_ms++] = 32; }

    // block_n candidates: 16, 32, 48, ..., 256
    int block_ns[16];
    int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16) block_ns[n_block_ns++] = i;

    // Select by wave utilization (same logic as DeepGEMM heuristic)
    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

            // SM90 BF16: at least one of block_m, block_n must be <= 128
            if (bm > 128 && bn > 128)
                continue;

            bool better = false;
            if (best_bm == 0 || waves < best_waves) {
                better = true;
            } else if (waves == best_waves) {
                better = last_util > best_last;
                if (last_util == best_last) {
                    better |= (bm == best_bm && bn < best_bn);
                    better |= (bn == best_bn && bm < best_bm);
                    better |= (bm != best_bm && bn > best_bn && bn <= n && bm <= m);
                }
            }
            if (better) {
                best_bm = bm; best_bn = bn;
                best_waves = waves; best_last = last_util;
            }
        }
    }

    // Thread config: 128 TMA + 128 or 256 math
    int num_tma = 128;
    int num_math = (best_bm <= 64) ? 128 : 256;

    // Multicast: only for M >= 512, check divisibility
    // (matches DeepGEMM heuristic/common.hpp get_best_config multicast logic)
    int multicast = 1;
    bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, best_bm) % 2 == 0);
        // Order: prefer multicast on the smaller block dimension
        bool order[2] = {false, true}; // {on_b, on_a}
        if (best_bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_b, legal_on_a};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) {
                multicast = 2;
                mc_on_a = order[i];
                break;
            }
        }
    }

    // Swizzle modes
    int sw_a = get_swizzle(block_k); // A: K-major, inner=K
    int sw_b = get_swizzle(block_k); // B: K-major, inner=K
    int sw_d = get_swizzle(best_bn); // D: row-major, inner=N

    // Select max stages that fit in shared memory (232448 bytes for SM90)
    // (matches DeepGEMM heuristic/common.hpp get_smem_config exactly)
    const int smem_capacity = 232448;
    int smem_d = ((best_bm * best_bn * 2 + 1023) / 1024) * 1024; // align(block_m * block_n * sizeof(bf16), 1024)
    int smem_a_per_stage = best_bm * block_k * 2; // bf16
    int smem_b_per_stage = best_bn * block_k * 2;
    // BF16 kernel: no SF smem, no tmem_ptr, no tensormap (Normal GEMM)

    int best_stages = 0;
    int best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int smem_barrier = s * 8 * 2; // num_stages * sizeof(ClusterTransactionBarrier) * 2
        int smem = smem_d + s * (smem_a_per_stage + smem_b_per_stage) + smem_barrier;
        if (smem <= smem_capacity) {
            best_stages = s;
            best_smem = smem;
            break;
        }
    }

    return KernelConfig{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── Kernel instantiations ───────────────────────────────────────────
// Pre-compile common configs. Each is a specific template instantiation.

// Helper: get SM count at init time
static int g_num_sms = 0;
static void ensure_num_sms() {
    if (g_num_sms > 0) return;
    int dev; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
}

// Kernel type alias for a specific config (SM90, NT layout, normal BF16 GEMM, H200=132 SMs)
#define KERNEL_TYPE(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, SWIZZLE_D, NUM_MC) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, 1,                                                        \
        BLOCK_M, BLOCK_N, 64,                                              \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, NUM_MATH,                                             \
        NUM_MC, false, 132,                                                \
        GemmType::Normal, false, cutlass::bfloat16_t>

// ── Kernel variants (from heuristic output on H200, 132 SMs) ────────
// Collected by running select_config() for representative LLM shapes.
// Format: KERNEL_TYPE(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, SWIZZLE_D, NUM_MULTICAST)

// Prefill configs (heuristic-selected, mc=1 for M<512, mc=2 for M>=512)
// M=128, N=4096
__attribute__((used)) static auto* _k1 = &KERNEL_TYPE(64, 64, 13, 128, 128, 1);
// M=256, N=4096
__attribute__((used)) static auto* _k2 = &KERNEL_TYPE(64, 128, 8, 128, 128, 1);
// M=512, N=4096
__attribute__((used)) static auto* _k3 = &KERNEL_TYPE(64, 256, 4, 128, 128, 2);
// M=512, N=11008
__attribute__((used)) static auto* _k4 = &KERNEL_TYPE(256, 176, 2, 256, 32, 2);
// M=1024, N=4096
__attribute__((used)) static auto* _k5 = &KERNEL_TYPE(128, 256, 3, 256, 128, 2);
// M=2048, N=4096 (estimate)
__attribute__((used)) static auto* _k6 = &KERNEL_TYPE(256, 256, 1, 256, 128, 2);

// Decode configs (small M, mc=1)
__attribute__((used)) static auto* _k7 = &KERNEL_TYPE(16, 32, 32, 128, 64, 1);
__attribute__((used)) static auto* _k8 = &KERNEL_TYPE(32, 32, 28, 128, 64, 1);

// N=11008 configs: block_n=176 (swizzle_d=32) and block_n=96 (swizzle_d=64)
__attribute__((used)) static auto* _k9 = &KERNEL_TYPE(128, 176, 4, 256, 32, 2);
__attribute__((used)) static auto* _k10 = &KERNEL_TYPE(64, 176, 5, 128, 32, 1);
__attribute__((used)) static auto* _k11 = &KERNEL_TYPE(64, 176, 6, 128, 32, 1);
__attribute__((used)) static auto* _k12 = &KERNEL_TYPE(16, 176, 5, 128, 32, 1);
__attribute__((used)) static auto* _k13 = &KERNEL_TYPE(32, 176, 5, 128, 32, 1);
// block_n=96: 96*2=192, 192%128=64≠0, 192%64=0 → swizzle_d=64
__attribute__((used)) static auto* _k14 = &KERNEL_TYPE(16, 96, 15, 128, 64, 1);
__attribute__((used)) static auto* _k15 = &KERNEL_TYPE(32, 96, 13, 128, 64, 1);

// ── Kernel dispatch ─────────────────────────────────────────────────

static const void* get_kernel(const KernelConfig& cfg) {
    #define MATCH(BM, BN, ST, NM, SD, MC) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_math_threads == NM && cfg.swizzle_d == SD && cfg.num_multicast == MC) \
            return (const void*)&KERNEL_TYPE(BM, BN, ST, NM, SD, MC);

    // Prefill
    MATCH(64, 64, 13, 128, 128, 1)
    MATCH(64, 128, 8, 128, 128, 1)
    MATCH(64, 256, 4, 128, 128, 2)
    MATCH(256, 176, 2, 256, 32, 2)
    MATCH(128, 256, 3, 256, 128, 2)
    MATCH(256, 256, 1, 256, 128, 2)
    MATCH(64, 64, 13, 128, 128, 2)

    // Decode (small M, block_n=32, swizzle_d=64)
    MATCH(16, 32, 32, 128, 64, 1)
    MATCH(32, 32, 28, 128, 64, 1)

    // N=11008 variants
    MATCH(128, 176, 4, 256, 32, 2)
    MATCH(64, 176, 5, 128, 32, 1)
    MATCH(64, 176, 6, 128, 32, 1)
    MATCH(16, 176, 5, 128, 32, 1)
    MATCH(32, 176, 5, 128, 32, 1)
    MATCH(16, 96, 15, 128, 64, 1)
    MATCH(32, 96, 13, 128, 64, 1)

    #undef MATCH
    return nullptr;
}

// ── C FFI ───────────────────────────────────────────────────────────

extern "C" {

/// BF16 GEMM: D = A[M,K] @ B[K,N]  (A row-major, B col-major = weight[N,K] transposed)
/// Returns 0 on success.
int deepgemm_bf16_gemm(
    void* A, void* B, void* D,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    auto cfg = select_config(M, N, K, g_num_sms);

    auto kernel_ptr = get_kernel(cfg);
    if (!kernel_ptr) {
        fprintf(stderr, "DeepGEMM: no kernel for block_m=%d block_n=%d stages=%d math=%d mc=%d\n",
                cfg.block_m, cfg.block_n, cfg.num_stages, cfg.num_math_threads, cfg.num_multicast);
        return -1;
    }

    // Create TMA descriptors
    // A: K-major [M, K], inner=K, outer=M
    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    // B: K-major [N, K], inner=K, outer=N
    auto tma_b = make_2d_tma(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // D: row-major [M, N], inner=N, outer=M
    auto tma_d = make_2d_tma(D, N, M,
                             cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n,
                             cfg.block_m, N, cfg.swizzle_d);

    // Launch
    int num_threads = cfg.num_tma_threads + cfg.num_math_threads;
    dim3 grid(g_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    // Set shared memory
    cudaFuncSetAttribute(kernel_ptr,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          cfg.smem_size);

    // Kernel args
    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = {
        &grouped_layout,
        &um, &un, &uk,
        &tma_a, &tma_b, &tma_d
    };

    // Cluster launch for multicast
    if (cfg.num_multicast > 1) {
        cudaLaunchConfig_t launch_cfg = {};
        launch_cfg.gridDim = grid;
        launch_cfg.blockDim = block;
        launch_cfg.dynamicSmemBytes = cfg.smem_size;
        launch_cfg.stream = static_cast<cudaStream_t>(stream);

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {(unsigned)cfg.num_multicast, 1, 1};
        launch_cfg.attrs = attrs;
        launch_cfg.numAttrs = 1;

        cudaLaunchKernelExC(&launch_cfg, kernel_ptr, args);
    } else {
        cudaLaunchKernel(kernel_ptr, grid, block, args, cfg.smem_size,
                          static_cast<cudaStream_t>(stream));
    }

    return 0;
}

/// Query which config would be selected for a given shape (for debugging).
void deepgemm_query_config(int M, int N, int K,
                            int* out_block_m, int* out_block_n,
                            int* out_stages, int* out_smem) {
    ensure_num_sms();
    auto cfg = select_config(M, N, K, g_num_sms);
    *out_block_m = cfg.block_m;
    *out_block_n = cfg.block_n;
    *out_stages = cfg.num_stages;
    *out_smem = cfg.smem_size;
}

} // extern "C"
