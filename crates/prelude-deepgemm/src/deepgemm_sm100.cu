// DeepGEMM SM100 (Blackwell) BF16 GEMM wrapper.
// Compiled separately with -gencode=arch=compute_100a,code=sm_100a.
// Based on deepseek-ai/DeepGEMM (MIT license).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include <deep_gemm/impls/sm100_bf16_gemm.cuh>

using namespace deep_gemm;

// ── Reuse TMA / swizzle helpers from deepgemm_wrapper.cu ───────────
// These are duplicated here since this is a separate compilation unit.

static CUresult (*p_cuTensorMapEncodeTiled_sm100)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill) = nullptr;

static void ensure_driver_api_sm100() {
    if (p_cuTensorMapEncodeTiled_sm100) return;
    cuGetProcAddress("cuTensorMapEncodeTiled",
                     (void**)&p_cuTensorMapEncodeTiled_sm100, 12000,
                     CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
}

static CUtensorMapSwizzle swizzle_mode_to_enum_sm100(int mode) {
    switch (mode) {
        case 0:
        case 16:  return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default:  return CU_TENSOR_MAP_SWIZZLE_128B;
    }
}

static CUtensorMap make_2d_tma_sm100(void* data, int inner_dim, int outer_dim,
                                      int smem_inner, int smem_outer,
                                      int outer_stride, int swizzle_mode) {
    ensure_driver_api_sm100();
    const int elem_size = 2;
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;

    CUtensorMap tmap{};
    cuuint64_t gmem_dims[2] = {(cuuint64_t)inner_dim, (cuuint64_t)outer_dim};
    cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gmem_strides[1] = {(cuuint64_t)(outer_stride * elem_size)};
    cuuint32_t elem_strides[2] = {1, 1};

    p_cuTensorMapEncodeTiled_sm100(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, data, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode_to_enum_sm100(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tmap;
}

// ── SM100 kernel config ────────────────────────────────────────────

struct SM100Config {
    int block_m, block_n, block_k;
    int num_stages;
    int num_multicast;
    bool multicast_on_a; // always false for SM100
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

// ── SM100 Heuristic (matches upstream sm100.hpp) ───────────────────

static int get_swizzle_sm100(int block_size) {
    for (int mode : {128, 64, 32, 16})
        if ((block_size * 2) % mode == 0) return mode;
    return 16;
}

static SM100Config select_sm100_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;

    // SM100 block_m candidates: {128, 256} + {32, 64} for small M (K-major BF16)
    int block_ms[4] = {128, 256, 0, 0};
    int n_block_ms = 2;
    if (m <= 32) { block_ms[n_block_ms++] = 32; }
    if (m <= 64) { block_ms[n_block_ms++] = 64; }

    // SM100 block_n: {16, 32, 64, 96, ..., 256} step 32 after 16
    int block_ns[9] = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    int n_block_ns = 9;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm == 0) continue;
            // SM100 legality: block_n % 16 == 0 (always true for candidates)
            // Small K penalty: k <= 256 → both block_m and block_n <= 128
            if (k <= 256 && (bn > 128 || bm > 128)) continue;

            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

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

    // SM100 thread config: always 128 non-epilogue + 128 epilogue = 256
    // (no variation like SM90's 128+128 vs 128+256)

    // SM100 multicast: only on B (multicast on A is FORBIDDEN)
    int multicast = 1;
    if (m >= 512 && num_sms % 2 == 0) {
        // is_multicast_legal(m, block_m, 2, num_sms, require_divisible=true)
        bool legal_on_b = (ceil_div(m, best_bm) % 2 == 0) && (num_sms % 2 == 0);
        if (legal_on_b) multicast = 2;
    }

    // Swizzle: A/B K-major inner=block_k, D inner=block_n
    int sw_a = get_swizzle_sm100(block_k);
    int sw_b = get_swizzle_sm100(block_k);
    int sw_d = get_swizzle_sm100(best_bn);

    // SM100 SMEM calculation (differs from SM90):
    // smem_cd = min(block_m, 128) * swizzle_cd_mode * 2
    // smem_barrier = num_stages * 8 * 3 + 2 * 8 * 2 + 8
    // smem_tmem_ptr = 4
    const int smem_capacity = 232448;
    int smem_cd = std::min(best_bm, 128) * sw_d * 2;
    int smem_a_per_stage = best_bm * block_k * 2;
    int smem_b_per_stage = best_bn * block_k * 2;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int smem_barrier = s * 24 + 40; // s * 8 * 3 + 2 * 8 * 2 + 8
        int smem_tmem_ptr = 4;
        int total = smem_cd + s * (smem_a_per_stage + smem_b_per_stage)
                    + smem_barrier + smem_tmem_ptr;
        if (total <= smem_capacity) {
            best_stages = s;
            best_smem = total;
            break;
        }
    }

    return SM100Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast, .multicast_on_a = false,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── SM count ───────────────────────────────────────────────────────

static int g_sm100_num_sms = 0;
static void ensure_sm100_num_sms() {
    if (g_sm100_num_sms > 0) return;
    int dev; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_sm100_num_sms, cudaDevAttrMultiProcessorCount, dev);
}

// ── SM100 kernel instantiations ────────────────────────────────────
// Template: sm100_bf16_gemm_impl<
//   kMajorA, kMajorB,
//   SHAPE_M, SHAPE_N, SHAPE_K,
//   BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups,
//   kSwizzleA, kSwizzleB, kSwizzleCD,
//   kNumStages,
//   kNumNonEpilogueThreads, kNumEpilogueThreads,
//   kNumMulticast, kIsMulticastOnA, kNumSMs,
//   kGemmType, kWithAccumulation, cd_dtype_t, kTensorCoreUtilControl>

#define KERNEL_TYPE_SM100(BLOCK_M, BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm100_bf16_gemm_impl<                                            \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                            \
        0, 0, 0,                                                                \
        BLOCK_M, BLOCK_N, 64, 1,                                                \
        128, 128, SWIZZLE_D,                                                    \
        STAGES,                                                                 \
        128, 128,                                                               \
        NUM_MC, MC_ON_A, 132,                                                   \
        GemmType::Normal, false, cutlass::bfloat16_t, 100>

// block_m=32 (small M only)
__attribute__((used)) static auto* _s100_00 = &KERNEL_TYPE_SM100(32, 16, 32, 32, 1, false);
__attribute__((used)) static auto* _s100_01 = &KERNEL_TYPE_SM100(32, 32, 27, 64, 1, false);
__attribute__((used)) static auto* _s100_02 = &KERNEL_TYPE_SM100(32, 64, 22, 128, 1, false);
__attribute__((used)) static auto* _s100_03 = &KERNEL_TYPE_SM100(32, 96, 18, 64, 1, false);
__attribute__((used)) static auto* _s100_04 = &KERNEL_TYPE_SM100(32, 128, 15, 128, 1, false);

// block_m=64 (small M only)
__attribute__((used)) static auto* _s100_10 = &KERNEL_TYPE_SM100(64, 16, 32, 32, 1, false);
__attribute__((used)) static auto* _s100_11 = &KERNEL_TYPE_SM100(64, 32, 22, 64, 1, false);
__attribute__((used)) static auto* _s100_12 = &KERNEL_TYPE_SM100(64, 64, 15, 128, 1, false);
__attribute__((used)) static auto* _s100_13 = &KERNEL_TYPE_SM100(64, 96, 12, 64, 1, false);
__attribute__((used)) static auto* _s100_14 = &KERNEL_TYPE_SM100(64, 128, 10, 128, 1, false);

// block_m=128
__attribute__((used)) static auto* _s100_20 = &KERNEL_TYPE_SM100(128, 16, 12, 32, 1, false);
__attribute__((used)) static auto* _s100_21 = &KERNEL_TYPE_SM100(128, 32, 10, 64, 1, false);
__attribute__((used)) static auto* _s100_22 = &KERNEL_TYPE_SM100(128, 64, 8, 128, 1, false);
__attribute__((used)) static auto* _s100_23 = &KERNEL_TYPE_SM100(128, 96, 7, 64, 1, false);
__attribute__((used)) static auto* _s100_24 = &KERNEL_TYPE_SM100(128, 128, 6, 128, 1, false);
__attribute__((used)) static auto* _s100_25 = &KERNEL_TYPE_SM100(128, 160, 5, 64, 1, false);
__attribute__((used)) static auto* _s100_26 = &KERNEL_TYPE_SM100(128, 192, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_27 = &KERNEL_TYPE_SM100(128, 224, 4, 64, 1, false);
__attribute__((used)) static auto* _s100_28 = &KERNEL_TYPE_SM100(128, 256, 4, 128, 1, false);
// multicast on B (mc_on_a=false, multicast=2)
__attribute__((used)) static auto* _s100_29 = &KERNEL_TYPE_SM100(128, 16, 12, 32, 2, false);
__attribute__((used)) static auto* _s100_2a = &KERNEL_TYPE_SM100(128, 32, 10, 64, 2, false);
__attribute__((used)) static auto* _s100_2b = &KERNEL_TYPE_SM100(128, 64, 8, 128, 2, false);
__attribute__((used)) static auto* _s100_2c = &KERNEL_TYPE_SM100(128, 96, 7, 64, 2, false);
__attribute__((used)) static auto* _s100_2d = &KERNEL_TYPE_SM100(128, 128, 6, 128, 2, false);
__attribute__((used)) static auto* _s100_2e = &KERNEL_TYPE_SM100(128, 160, 5, 64, 2, false);
__attribute__((used)) static auto* _s100_2f = &KERNEL_TYPE_SM100(128, 192, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_30 = &KERNEL_TYPE_SM100(128, 224, 4, 64, 2, false);
__attribute__((used)) static auto* _s100_31 = &KERNEL_TYPE_SM100(128, 256, 4, 128, 2, false);

// block_m=256
__attribute__((used)) static auto* _s100_40 = &KERNEL_TYPE_SM100(256, 16, 6, 32, 1, false);
__attribute__((used)) static auto* _s100_41 = &KERNEL_TYPE_SM100(256, 32, 5, 64, 1, false);
__attribute__((used)) static auto* _s100_42 = &KERNEL_TYPE_SM100(256, 64, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_43 = &KERNEL_TYPE_SM100(256, 96, 4, 64, 1, false);
__attribute__((used)) static auto* _s100_44 = &KERNEL_TYPE_SM100(256, 128, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_45 = &KERNEL_TYPE_SM100(256, 160, 3, 64, 1, false);
__attribute__((used)) static auto* _s100_46 = &KERNEL_TYPE_SM100(256, 192, 3, 128, 1, false);
__attribute__((used)) static auto* _s100_47 = &KERNEL_TYPE_SM100(256, 224, 3, 64, 1, false);
__attribute__((used)) static auto* _s100_48 = &KERNEL_TYPE_SM100(256, 256, 3, 128, 1, false);
// multicast on B
__attribute__((used)) static auto* _s100_49 = &KERNEL_TYPE_SM100(256, 16, 6, 32, 2, false);
__attribute__((used)) static auto* _s100_4a = &KERNEL_TYPE_SM100(256, 32, 5, 64, 2, false);
__attribute__((used)) static auto* _s100_4b = &KERNEL_TYPE_SM100(256, 64, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_4c = &KERNEL_TYPE_SM100(256, 96, 4, 64, 2, false);
__attribute__((used)) static auto* _s100_4d = &KERNEL_TYPE_SM100(256, 128, 4, 128, 2, false);

// ── SM100 kernel dispatch ──────────────────────────────────────────

static const void* get_sm100_kernel(const SM100Config& cfg) {
    #define MATCH_SM100(BM, BN, ST, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_SM100(BM, BN, ST, SD, MC, MCA);

    // block_m=32
    MATCH_SM100(32, 16, 32, 32, 1, false)
    MATCH_SM100(32, 32, 27, 64, 1, false)
    MATCH_SM100(32, 64, 22, 128, 1, false)
    MATCH_SM100(32, 96, 18, 64, 1, false)
    MATCH_SM100(32, 128, 15, 128, 1, false)
    // block_m=64
    MATCH_SM100(64, 16, 32, 32, 1, false)
    MATCH_SM100(64, 32, 22, 64, 1, false)
    MATCH_SM100(64, 64, 15, 128, 1, false)
    MATCH_SM100(64, 96, 12, 64, 1, false)
    MATCH_SM100(64, 128, 10, 128, 1, false)
    // block_m=128
    MATCH_SM100(128, 16, 12, 32, 1, false)
    MATCH_SM100(128, 32, 10, 64, 1, false)
    MATCH_SM100(128, 64, 8, 128, 1, false)
    MATCH_SM100(128, 96, 7, 64, 1, false)
    MATCH_SM100(128, 128, 6, 128, 1, false)
    MATCH_SM100(128, 160, 5, 64, 1, false)
    MATCH_SM100(128, 192, 4, 128, 1, false)
    MATCH_SM100(128, 224, 4, 64, 1, false)
    MATCH_SM100(128, 256, 4, 128, 1, false)
    MATCH_SM100(128, 16, 12, 32, 2, false)
    MATCH_SM100(128, 32, 10, 64, 2, false)
    MATCH_SM100(128, 64, 8, 128, 2, false)
    MATCH_SM100(128, 96, 7, 64, 2, false)
    MATCH_SM100(128, 128, 6, 128, 2, false)
    MATCH_SM100(128, 160, 5, 64, 2, false)
    MATCH_SM100(128, 192, 4, 128, 2, false)
    MATCH_SM100(128, 224, 4, 64, 2, false)
    MATCH_SM100(128, 256, 4, 128, 2, false)
    // block_m=256
    MATCH_SM100(256, 16, 6, 32, 1, false)
    MATCH_SM100(256, 32, 5, 64, 1, false)
    MATCH_SM100(256, 64, 4, 128, 1, false)
    MATCH_SM100(256, 96, 4, 64, 1, false)
    MATCH_SM100(256, 128, 4, 128, 1, false)
    MATCH_SM100(256, 160, 3, 64, 1, false)
    MATCH_SM100(256, 192, 3, 128, 1, false)
    MATCH_SM100(256, 224, 3, 64, 1, false)
    MATCH_SM100(256, 256, 3, 128, 1, false)
    MATCH_SM100(256, 16, 6, 32, 2, false)
    MATCH_SM100(256, 32, 5, 64, 2, false)
    MATCH_SM100(256, 64, 4, 128, 2, false)
    MATCH_SM100(256, 96, 4, 64, 2, false)
    MATCH_SM100(256, 128, 4, 128, 2, false)

    #undef MATCH_SM100
    return nullptr;
}

// ── C FFI ──────────────────────────────────────────────────────────

extern "C" {

int deepgemm_sm100_bf16_gemm(
    void* A, void* B, void* D,
    int M, int N, int K,
    void* stream
) {
    ensure_sm100_num_sms();
    cudaGetLastError();

    auto cfg = select_sm100_config(M, N, K, g_sm100_num_sms);
    auto kernel_ptr = get_sm100_kernel(cfg);
    if (!kernel_ptr) return -1;

    // TMA descriptors (same layout as SM90: A K-major, B K-major, D row-major)
    auto tma_a = make_2d_tma_sm100(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_sm100(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);

    // SM100 CD store uses min(block_m, 128) for the store block_m
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma_sm100(D, N, M, d_smem_inner, cd_store_bm, N, cfg.swizzle_d);

    // SM100: always 128+128 = 256 threads
    int num_threads = 256;
    dim3 grid(g_sm100_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size);

    // Same kernel args as SM90: (grouped_layout, shape_m, shape_n, shape_k, tma_a, tma_b, tma_cd)
    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = {
        &grouped_layout,
        &um, &un, &uk,
        &tma_a, &tma_b, &tma_d
    };

    auto s = static_cast<cudaStream_t>(stream);

    if (cfg.num_multicast > 1) {
        cudaLaunchConfig_t launch_cfg = {};
        launch_cfg.gridDim = grid;
        launch_cfg.blockDim = block;
        launch_cfg.dynamicSmemBytes = cfg.smem_size;
        launch_cfg.stream = s;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {(unsigned)cfg.num_multicast, 1, 1};
        launch_cfg.attrs = attrs;
        launch_cfg.numAttrs = 1;

        if (cudaLaunchKernelExC(&launch_cfg, kernel_ptr, args) != cudaSuccess) {
            cudaGetLastError();
            return -2;
        }
    } else {
        if (cudaLaunchKernel(kernel_ptr, grid, block, args, cfg.smem_size, s) != cudaSuccess) {
            cudaGetLastError();
            return -2;
        }
    }
    return 0;
}

void deepgemm_sm100_query_config(int M, int N, int K,
                                  int* out_block_m, int* out_block_n,
                                  int* out_stages, int* out_smem) {
    ensure_sm100_num_sms();
    auto cfg = select_sm100_config(M, N, K, g_sm100_num_sms);
    *out_block_m = cfg.block_m;
    *out_block_n = cfg.block_n;
    *out_stages = cfg.num_stages;
    *out_smem = cfg.smem_size;
}

} // extern "C"
