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
        case 0:
        case 16:  return CU_TENSOR_MAP_SWIZZLE_NONE;
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

// FP8 data (1-byte elements)
static CUtensorMap make_2d_tma_u8(void* data, int inner_dim, int outer_dim,
                                   int smem_inner, int smem_outer,
                                   int outer_stride, int swizzle_mode) {
    ensure_driver_api();
    const int elem_size = 1;
    if (swizzle_mode != 0)
        smem_inner = swizzle_mode / elem_size;

    CUtensorMap tmap{};
    cuuint64_t gmem_dims[2] = {(cuuint64_t)inner_dim, (cuuint64_t)outer_dim};
    cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gmem_strides[1] = {(cuuint64_t)(outer_stride * elem_size)};
    cuuint32_t elem_strides[2] = {1, 1};

    p_cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, data, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode_to_enum(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tmap;
}

// FP32 data (for scaling factors and FP32 output D)
static CUtensorMap make_2d_tma_f32(void* data, int inner_dim, int outer_dim,
                                    int smem_inner, int smem_outer,
                                    int outer_stride, int swizzle_mode) {
    ensure_driver_api();
    const int elem_size = 4;
    if (swizzle_mode != 0)
        smem_inner = swizzle_mode / elem_size;

    CUtensorMap tmap{};
    cuuint64_t gmem_dims[2] = {(cuuint64_t)inner_dim, (cuuint64_t)outer_dim};
    cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gmem_strides[1] = {(cuuint64_t)(outer_stride * elem_size)};
    cuuint32_t elem_strides[2] = {1, 1};

    p_cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
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

    // Multicast: only for M >= 512, check divisibility.
    // Safe because Rust-side alignment guard ensures all dims are multiples of 64.
    // (matches DeepGEMM heuristic/common.hpp get_best_config multicast logic)
    int multicast = 1;
    bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, best_bm) % 2 == 0);
        // Order: prefer multicast on the smaller block dimension
        bool order[2] = {false, true}; // {on_b, on_a}
        if (best_bm > best_bn) { order[0] = true; order[1] = false; }
        // legal[0] = mc_on_a=false (B multicast) needs M-dim even
        // legal[1] = mc_on_a=true (A multicast) needs N-dim even
        bool legal[2] = {legal_on_a, legal_on_b};
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

// ── FP8 1D2D heuristic ─────────────────────────────────────────────
// SM90 FP8 E4M3 with per-token scaling, BF16 output (1D2D kernel).

struct FP8Config {
    int block_m, block_n, block_k;
    int num_stages, num_last_stages;
    int num_tma_threads, num_math_threads;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

static bool fp8_valid_block_n(int bn) {
    // 1D2D constraint: ceil_div(bn,128)==1 or gcd(bn,128)==bn-128
    if (bn <= 128) return true;
    auto gcd = [](int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; };
    return gcd(bn, 128) == bn - 128;
}

static FP8Config select_fp8_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;

    // 1D2D: BLOCK_M % WGMMA::M(64)==0 OR BLOCK_M < 64
    int block_ms[4] = {64, 128, 0, 0};
    int n_block_ms = 2;
    if (m <= 16) { block_ms[n_block_ms++] = 16; }
    if (m <= 32) { block_ms[n_block_ms++] = 32; }

    // Valid block_n for 1D2D
    int block_ns[12]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16)
        if (fp8_valid_block_n(i)) block_ns[n_block_ns++] = i;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm == 0) continue;
            if (bm > 128 && bn > 128) continue;
            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

            bool better = false;
            if (best_bm == 0 || waves < best_waves) better = true;
            else if (waves == best_waves) {
                better = last_util > best_last;
                if (last_util == best_last) {
                    better |= (bm == best_bm && bn < best_bn);
                    better |= (bn == best_bn && bm < best_bm);
                    better |= (bm != best_bm && bn > best_bn && bn <= n && bm <= m);
                }
            }
            if (better) { best_bm = bm; best_bn = bn; best_waves = waves; best_last = last_util; }
        }
    }

    int num_tma = 128, num_math = (best_bm <= 64) ? 128 : 256;

    // Multicast
    int multicast = 1; bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, best_bm) % 2 == 0);
        bool order[2] = {false, true};
        if (best_bm > best_bn) { order[0] = true; order[1] = false; }
        // legal[0] = mc_on_a=false (B multicast) needs M-dim even
        // legal[1] = mc_on_a=true (A multicast) needs N-dim even
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) { multicast = 2; mc_on_a = order[i]; break; }
        }
    }

    // Swizzle: A/B use FP8 (1 byte), D uses BF16 (2 bytes)
    int sw_a = 128, sw_b = 128; // block_k * 1 = 128 bytes → swizzle 128
    int sw_d = get_swizzle(best_bn); // BF16: block_n * 2 bytes

    // kNumLastStages for 1D2D: depends on whether 128 % block_n == 0
    // When block_n > block_k, unsigned arithmetic in the kernel gives ceil_div(bn, bn-bk_wrap) = 1
    int nls;
    if (block_k % best_bn == 0) nls = 0;
    else if (best_bn <= block_k) nls = ceil_div(best_bn, block_k - best_bn);
    else nls = 1;

    // SMEM: BF16 output, SFA per stage, SFB from global memory (one-time buffer)
    const int smem_capacity = 232448;
    int smem_d = ((best_bm * best_bn * 2 + 1023) / 1024) * 1024;
    int smem_a_per = best_bm * block_k;
    int smem_b_per = best_bn * block_k;
    int smem_sfa_per = ((best_bm * 4 + 127) / 128) * 128;
    int smem_per_stage = smem_a_per + smem_b_per + smem_sfa_per;

    int k_scales = ceil_div(k, block_k);
    bool must_uniform = (block_k % best_bn == 0);
    int sfb_buf = ((k_scales * (must_uniform ? 1 : 2) * 4 + 15) / 16) * 16;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int barrier = s * 16;
        int total = smem_d + s * smem_per_stage + barrier + sfb_buf;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return FP8Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages, .num_last_stages = nls,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── Grouped GEMM heuristic ─────────────────────────────────────────
// M-Grouped Contiguous BF16 GEMM (MoE). block_m fixed to 128
// (get_mk_alignment_for_contiguous_layout). Uses same kernel template
// as normal BF16 GEMM but with GemmType::MGroupedContiguous.

static KernelConfig select_grouped_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;
    // For MGroupedContiguous, block_m is always 128
    const int bm = 128;

    // block_n candidates: 16, 32, 48, ..., 256
    int block_ns[16]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16) block_ns[n_block_ns++] = i;

    // Select by wave utilization (upstream: num_groups=1 for contiguous since M is total_M)
    int best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int j = 0; j < n_block_ns; j++) {
        int bn = block_ns[j];
        // block_m=128 <= 128, so all block_n values are legal
        int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
        int waves = ceil_div(num_blocks, num_sms);
        int last_util = num_blocks % num_sms;
        if (last_util == 0) last_util = num_sms;

        bool better = false;
        if (best_bn == 0 || waves < best_waves) better = true;
        else if (waves == best_waves) {
            better = last_util > best_last;
            if (last_util == best_last)
                better |= (bn < best_bn);
        }
        if (better) { best_bn = bn; best_waves = waves; best_last = last_util; }
    }

    // block_m=128 > 64 → 256 math threads
    int num_tma = 128, num_math = 256;

    // Multicast (same logic as normal BF16; upstream passes num_groups=1 for contiguous)
    int multicast = 1; bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, bm) % 2 == 0);
        bool order[2] = {false, true};
        if (bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) { multicast = 2; mc_on_a = order[i]; break; }
        }
    }

    int sw_a = get_swizzle(block_k);
    int sw_b = get_swizzle(block_k);
    int sw_d = get_swizzle(best_bn);

    const int smem_capacity = 232448;
    int smem_d = ((bm * best_bn * 2 + 1023) / 1024) * 1024;
    int smem_a_per_stage = bm * block_k * 2;
    int smem_b_per_stage = best_bn * block_k * 2;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int smem_barrier = s * 8 * 2;
        int smem = smem_d + s * (smem_a_per_stage + smem_b_per_stage) + smem_barrier;
        if (smem <= smem_capacity) { best_stages = s; best_smem = smem; break; }
    }

    return KernelConfig{
        .block_m = bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── Grouped FP8 1D2D heuristic ─────────────────────────────────────
// M-Grouped Contiguous FP8 GEMM. block_m fixed to 128.

static FP8Config select_fp8_grouped_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;
    const int bm = 128; // MGroupedContiguous: fixed

    // Valid block_n for 1D2D (same as normal FP8)
    int block_ns[12]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16)
        if (fp8_valid_block_n(i)) block_ns[n_block_ns++] = i;

    int best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int j = 0; j < n_block_ns; j++) {
        int bn = block_ns[j];
        int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
        int waves = ceil_div(num_blocks, num_sms);
        int last_util = num_blocks % num_sms;
        if (last_util == 0) last_util = num_sms;

        bool better = false;
        if (best_bn == 0 || waves < best_waves) better = true;
        else if (waves == best_waves) {
            better = last_util > best_last;
            if (last_util == best_last) better |= (bn < best_bn);
        }
        if (better) { best_bn = bn; best_waves = waves; best_last = last_util; }
    }

    int num_tma = 128, num_math = 256;

    // Multicast
    int multicast = 1; bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, bm) % 2 == 0);
        bool order[2] = {false, true};
        if (bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) { multicast = 2; mc_on_a = order[i]; break; }
        }
    }

    int sw_a = 128, sw_b = 128;
    int sw_d = get_swizzle(best_bn);

    int nls;
    if (block_k % best_bn == 0) nls = 0;
    else if (best_bn <= block_k) nls = ceil_div(best_bn, block_k - best_bn);
    else nls = 1;

    const int smem_capacity = 232448;
    int smem_d = ((bm * best_bn * 2 + 1023) / 1024) * 1024;
    int smem_a_per = bm * block_k;
    int smem_b_per = best_bn * block_k;
    int smem_sfa_per = ((bm * 4 + 127) / 128) * 128;
    int smem_per_stage = smem_a_per + smem_b_per + smem_sfa_per;

    int k_scales = ceil_div(k, block_k);
    bool must_uniform = (block_k % best_bn == 0);
    int sfb_buf = ((k_scales * (must_uniform ? 1 : 2) * 4 + 15) / 16) * 16;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int barrier = s * 16;
        int total = smem_d + s * smem_per_stage + barrier + sfb_buf;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return FP8Config{
        .block_m = bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages, .num_last_stages = nls,
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
#define KERNEL_TYPE(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, 1,                                                        \
        BLOCK_M, BLOCK_N, 64,                                              \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, NUM_MATH,                                             \
        NUM_MC, MC_ON_A, 132,                                              \
        GemmType::Normal, false, cutlass::bfloat16_t>

// ── Kernel variants (auto-generated, 99 configs) ──────────────────
// Covers M=1..4096 × common LLM N/K dims on H200 (132 SMs).
// Generated by enumerating select_config() over all relevant shapes.

__attribute__((used)) static auto* _k00 = &KERNEL_TYPE(16, 16, 32, 128, 32, 1, false);
__attribute__((used)) static auto* _k01 = &KERNEL_TYPE(16, 32, 32, 128, 64, 1, false);
__attribute__((used)) static auto* _k02 = &KERNEL_TYPE(16, 48, 28, 128, 32, 1, false);
__attribute__((used)) static auto* _k03 = &KERNEL_TYPE(16, 64, 22, 128, 128, 1, false);
__attribute__((used)) static auto* _k04 = &KERNEL_TYPE(16, 96, 15, 128, 64, 1, false);
__attribute__((used)) static auto* _k05 = &KERNEL_TYPE(16, 112, 13, 128, 32, 1, false);
__attribute__((used)) static auto* _k06 = &KERNEL_TYPE(16, 224, 7, 128, 64, 1, false);
__attribute__((used)) static auto* _k07 = &KERNEL_TYPE(16, 240, 6, 128, 32, 1, false);
__attribute__((used)) static auto* _k08 = &KERNEL_TYPE(32, 16, 32, 128, 32, 1, false);
__attribute__((used)) static auto* _k09 = &KERNEL_TYPE(32, 32, 28, 128, 64, 1, false);
__attribute__((used)) static auto* _k10 = &KERNEL_TYPE(32, 48, 22, 128, 32, 1, false);
__attribute__((used)) static auto* _k11 = &KERNEL_TYPE(32, 64, 18, 128, 128, 1, false);
__attribute__((used)) static auto* _k12 = &KERNEL_TYPE(32, 96, 13, 128, 64, 1, false);
__attribute__((used)) static auto* _k13 = &KERNEL_TYPE(32, 112, 12, 128, 32, 1, false);
__attribute__((used)) static auto* _k14 = &KERNEL_TYPE(32, 224, 6, 128, 64, 1, false);
__attribute__((used)) static auto* _k15 = &KERNEL_TYPE(32, 240, 6, 128, 32, 1, false);
__attribute__((used)) static auto* _k16 = &KERNEL_TYPE(64, 16, 22, 128, 32, 1, false);
__attribute__((used)) static auto* _k17 = &KERNEL_TYPE(64, 32, 18, 128, 64, 1, false);
__attribute__((used)) static auto* _k18 = &KERNEL_TYPE(64, 48, 15, 128, 32, 1, false);
__attribute__((used)) static auto* _k19 = &KERNEL_TYPE(64, 64, 13, 128, 128, 1, false);
__attribute__((used)) static auto* _k20 = &KERNEL_TYPE(64, 64, 13, 128, 128, 2, false);
__attribute__((used)) static auto* _k21 = &KERNEL_TYPE(64, 80, 12, 128, 32, 1, false);
__attribute__((used)) static auto* _k22 = &KERNEL_TYPE(64, 80, 12, 128, 32, 2, true);
__attribute__((used)) static auto* _k23 = &KERNEL_TYPE(64, 96, 10, 128, 64, 1, false);
__attribute__((used)) static auto* _k24 = &KERNEL_TYPE(64, 96, 10, 128, 64, 2, true);
__attribute__((used)) static auto* _k25 = &KERNEL_TYPE(64, 112, 9, 128, 32, 1, false);
__attribute__((used)) static auto* _k26 = &KERNEL_TYPE(64, 112, 9, 128, 32, 2, false);
__attribute__((used)) static auto* _k27 = &KERNEL_TYPE(64, 128, 8, 128, 128, 1, false);
__attribute__((used)) static auto* _k28 = &KERNEL_TYPE(64, 128, 8, 128, 128, 2, false);
__attribute__((used)) static auto* _k29 = &KERNEL_TYPE(64, 144, 8, 128, 32, 1, false);
__attribute__((used)) static auto* _k30 = &KERNEL_TYPE(64, 160, 7, 128, 64, 1, false);
__attribute__((used)) static auto* _k31 = &KERNEL_TYPE(64, 160, 7, 128, 64, 2, true);
__attribute__((used)) static auto* _k32 = &KERNEL_TYPE(64, 176, 6, 128, 32, 1, false);
__attribute__((used)) static auto* _k33 = &KERNEL_TYPE(64, 176, 6, 128, 32, 2, false);
__attribute__((used)) static auto* _k34 = &KERNEL_TYPE(64, 192, 6, 128, 128, 1, false);
__attribute__((used)) static auto* _k35 = &KERNEL_TYPE(64, 192, 6, 128, 128, 2, false);
__attribute__((used)) static auto* _k36 = &KERNEL_TYPE(64, 192, 6, 128, 128, 2, true);
__attribute__((used)) static auto* _k37 = &KERNEL_TYPE(64, 208, 5, 128, 32, 2, false);
__attribute__((used)) static auto* _k38 = &KERNEL_TYPE(64, 208, 5, 128, 32, 2, true);
__attribute__((used)) static auto* _k39 = &KERNEL_TYPE(64, 224, 5, 128, 64, 1, false);
__attribute__((used)) static auto* _k40 = &KERNEL_TYPE(64, 224, 5, 128, 64, 2, false);
__attribute__((used)) static auto* _k41 = &KERNEL_TYPE(64, 240, 5, 128, 32, 1, false);
__attribute__((used)) static auto* _k42 = &KERNEL_TYPE(64, 240, 5, 128, 32, 2, false);
__attribute__((used)) static auto* _k43 = &KERNEL_TYPE(64, 240, 5, 128, 32, 2, true);
__attribute__((used)) static auto* _k44 = &KERNEL_TYPE(64, 256, 4, 128, 128, 1, false);
__attribute__((used)) static auto* _k45 = &KERNEL_TYPE(64, 256, 4, 128, 128, 2, false);
__attribute__((used)) static auto* _k46 = &KERNEL_TYPE(64, 256, 4, 128, 128, 2, true);
__attribute__((used)) static auto* _k47 = &KERNEL_TYPE(128, 16, 12, 256, 32, 1, false);
__attribute__((used)) static auto* _k48 = &KERNEL_TYPE(128, 32, 10, 256, 64, 1, false);
__attribute__((used)) static auto* _k49 = &KERNEL_TYPE(128, 48, 9, 256, 32, 1, false);
__attribute__((used)) static auto* _k50 = &KERNEL_TYPE(128, 48, 9, 256, 32, 2, true);
__attribute__((used)) static auto* _k51 = &KERNEL_TYPE(128, 64, 8, 256, 128, 1, false);
__attribute__((used)) static auto* _k52 = &KERNEL_TYPE(128, 64, 8, 256, 128, 2, true);
__attribute__((used)) static auto* _k53 = &KERNEL_TYPE(128, 80, 7, 256, 32, 1, false);
__attribute__((used)) static auto* _k54 = &KERNEL_TYPE(128, 80, 7, 256, 32, 2, false);
__attribute__((used)) static auto* _k55 = &KERNEL_TYPE(128, 80, 7, 256, 32, 2, true);
__attribute__((used)) static auto* _k56 = &KERNEL_TYPE(128, 96, 7, 256, 64, 1, false);
__attribute__((used)) static auto* _k57 = &KERNEL_TYPE(128, 96, 7, 256, 64, 2, true);
__attribute__((used)) static auto* _k58 = &KERNEL_TYPE(128, 112, 6, 256, 32, 2, false);
__attribute__((used)) static auto* _k59 = &KERNEL_TYPE(128, 128, 6, 256, 128, 1, false);
__attribute__((used)) static auto* _k60 = &KERNEL_TYPE(128, 128, 6, 256, 128, 2, false);
__attribute__((used)) static auto* _k61 = &KERNEL_TYPE(128, 144, 5, 256, 32, 1, false);
__attribute__((used)) static auto* _k62 = &KERNEL_TYPE(128, 144, 5, 256, 32, 2, false);
__attribute__((used)) static auto* _k63 = &KERNEL_TYPE(128, 160, 5, 256, 64, 1, false);
__attribute__((used)) static auto* _k64 = &KERNEL_TYPE(128, 160, 5, 256, 64, 2, false);
__attribute__((used)) static auto* _k65 = &KERNEL_TYPE(128, 160, 5, 256, 64, 2, true);
__attribute__((used)) static auto* _k66 = &KERNEL_TYPE(128, 176, 4, 256, 32, 1, false);
__attribute__((used)) static auto* _k67 = &KERNEL_TYPE(128, 176, 4, 256, 32, 2, false);
__attribute__((used)) static auto* _k68 = &KERNEL_TYPE(128, 176, 4, 256, 32, 2, true);
__attribute__((used)) static auto* _k69 = &KERNEL_TYPE(128, 192, 4, 256, 128, 1, false);
__attribute__((used)) static auto* _k70 = &KERNEL_TYPE(128, 192, 4, 256, 128, 2, false);
__attribute__((used)) static auto* _k71 = &KERNEL_TYPE(128, 192, 4, 256, 128, 2, true);
__attribute__((used)) static auto* _k72 = &KERNEL_TYPE(128, 208, 4, 256, 32, 1, false);
__attribute__((used)) static auto* _k73 = &KERNEL_TYPE(128, 208, 4, 256, 32, 2, false);
__attribute__((used)) static auto* _k74 = &KERNEL_TYPE(128, 208, 4, 256, 32, 2, true);
__attribute__((used)) static auto* _k75 = &KERNEL_TYPE(128, 224, 3, 256, 64, 1, false);
__attribute__((used)) static auto* _k76 = &KERNEL_TYPE(128, 224, 3, 256, 64, 2, false);
__attribute__((used)) static auto* _k77 = &KERNEL_TYPE(128, 224, 3, 256, 64, 2, true);
__attribute__((used)) static auto* _k78 = &KERNEL_TYPE(128, 240, 3, 256, 32, 1, false);
__attribute__((used)) static auto* _k79 = &KERNEL_TYPE(128, 240, 3, 256, 32, 2, false);
__attribute__((used)) static auto* _k80 = &KERNEL_TYPE(128, 240, 3, 256, 32, 2, true);
__attribute__((used)) static auto* _k81 = &KERNEL_TYPE(128, 256, 3, 256, 128, 1, false);
__attribute__((used)) static auto* _k82 = &KERNEL_TYPE(128, 256, 3, 256, 128, 2, false);
__attribute__((used)) static auto* _k83 = &KERNEL_TYPE(128, 256, 3, 256, 128, 2, true);
__attribute__((used)) static auto* _k84 = &KERNEL_TYPE(256, 16, 6, 256, 32, 1, false);
__attribute__((used)) static auto* _k85 = &KERNEL_TYPE(256, 32, 5, 256, 64, 2, true);
__attribute__((used)) static auto* _k86 = &KERNEL_TYPE(256, 48, 5, 256, 32, 1, false);
__attribute__((used)) static auto* _k87 = &KERNEL_TYPE(256, 48, 5, 256, 32, 2, true);
__attribute__((used)) static auto* _k88 = &KERNEL_TYPE(256, 64, 4, 256, 128, 2, false);
__attribute__((used)) static auto* _k89 = &KERNEL_TYPE(256, 64, 4, 256, 128, 2, true);
__attribute__((used)) static auto* _k90 = &KERNEL_TYPE(256, 80, 4, 256, 32, 1, false);
__attribute__((used)) static auto* _k91 = &KERNEL_TYPE(256, 80, 4, 256, 32, 2, false);
__attribute__((used)) static auto* _k92 = &KERNEL_TYPE(256, 80, 4, 256, 32, 2, true);
__attribute__((used)) static auto* _k93 = &KERNEL_TYPE(256, 96, 4, 256, 64, 2, true);
__attribute__((used)) static auto* _k94 = &KERNEL_TYPE(256, 112, 3, 256, 32, 1, false);
__attribute__((used)) static auto* _k95 = &KERNEL_TYPE(256, 112, 3, 256, 32, 2, false);
__attribute__((used)) static auto* _k96 = &KERNEL_TYPE(256, 112, 3, 256, 32, 2, true);
__attribute__((used)) static auto* _k97 = &KERNEL_TYPE(256, 128, 3, 256, 128, 2, false);
__attribute__((used)) static auto* _k98 = &KERNEL_TYPE(256, 128, 3, 256, 128, 2, true);
// Additional variants for Gemma vocab=262144, Qwen3-32B intermediate=25600, etc.
__attribute__((used)) static auto* _k99 = &KERNEL_TYPE(16, 80, 18, 128, 32, 1, false);
__attribute__((used)) static auto* _kA0 = &KERNEL_TYPE(16, 208, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _kA1 = &KERNEL_TYPE(16, 256, 6, 128, 128, 1, false);
__attribute__((used)) static auto* _kA2 = &KERNEL_TYPE(64, 144, 8, 128, 32, 2, false);
__attribute__((used)) static auto* _kA3 = &KERNEL_TYPE(64, 160, 7, 128, 64, 2, false);
__attribute__((used)) static auto* _kA4 = &KERNEL_TYPE(64, 208, 5, 128, 32, 1, false);

// ── FP8 1D2D kernel instantiations ──────────────────────────────────
// SM90 FP8 E4M3 1D2D (per-token scaling on A, per-channel on B), BF16 output.

#define KERNEL_TYPE_FP8(BLOCK_M, BLOCK_N, STAGES, LAST_STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_fp8_gemm_1d2d_impl<                                       \
        cute::UMMA::Major::MN,                                                 \
        0, 0, 0, 1,                                                            \
        BLOCK_M, BLOCK_N, 128,                                                 \
        128, 128, SWIZZLE_D,                                                   \
        STAGES, LAST_STAGES,                                                   \
        128, NUM_MATH,                                                         \
        NUM_MC, MC_ON_A, 132,                                                  \
        GemmType::Normal,                                                      \
        deep_gemm::EpilogueIdentity>

__attribute__((used)) static auto* _fp8_00 = &KERNEL_TYPE_FP8(16, 16, 32, 0, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_01 = &KERNEL_TYPE_FP8(16, 32, 32, 0, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_02 = &KERNEL_TYPE_FP8(16, 48, 27, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_03 = &KERNEL_TYPE_FP8(16, 64, 22, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_04 = &KERNEL_TYPE_FP8(16, 80, 18, 2, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_05 = &KERNEL_TYPE_FP8(16, 96, 15, 3, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_06 = &KERNEL_TYPE_FP8(16, 112, 13, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_07 = &KERNEL_TYPE_FP8(16, 256, 6, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_08 = &KERNEL_TYPE_FP8(32, 16, 32, 0, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_09 = &KERNEL_TYPE_FP8(32, 32, 27, 0, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_0a = &KERNEL_TYPE_FP8(32, 48, 21, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_0b = &KERNEL_TYPE_FP8(32, 48, 22, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_0c = &KERNEL_TYPE_FP8(32, 64, 18, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_0d = &KERNEL_TYPE_FP8(32, 80, 15, 2, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_0e = &KERNEL_TYPE_FP8(32, 96, 13, 3, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_0f = &KERNEL_TYPE_FP8(32, 112, 12, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_10 = &KERNEL_TYPE_FP8(32, 256, 5, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_11 = &KERNEL_TYPE_FP8(64, 16, 21, 0, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_12 = &KERNEL_TYPE_FP8(64, 32, 18, 0, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_13 = &KERNEL_TYPE_FP8(64, 48, 15, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_14 = &KERNEL_TYPE_FP8(64, 64, 13, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_15 = &KERNEL_TYPE_FP8(64, 64, 13, 0, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_16 = &KERNEL_TYPE_FP8(64, 80, 11, 2, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_17 = &KERNEL_TYPE_FP8(64, 96, 10, 3, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_18 = &KERNEL_TYPE_FP8(64, 112, 9, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_19 = &KERNEL_TYPE_FP8(64, 128, 8, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_1a = &KERNEL_TYPE_FP8(64, 128, 8, 0, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_1b = &KERNEL_TYPE_FP8(64, 144, 7, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_1c = &KERNEL_TYPE_FP8(64, 144, 7, 1, 128, 32, 2, false);
__attribute__((used)) static auto* _fp8_1d = &KERNEL_TYPE_FP8(64, 160, 7, 1, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_1e = &KERNEL_TYPE_FP8(64, 160, 7, 1, 128, 64, 2, false);
__attribute__((used)) static auto* _fp8_1f = &KERNEL_TYPE_FP8(64, 192, 6, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_20 = &KERNEL_TYPE_FP8(64, 192, 6, 1, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_21 = &KERNEL_TYPE_FP8(64, 256, 4, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_22 = &KERNEL_TYPE_FP8(64, 256, 4, 1, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_23 = &KERNEL_TYPE_FP8(64, 256, 4, 1, 128, 128, 2, true);
__attribute__((used)) static auto* _fp8_24 = &KERNEL_TYPE_FP8(128, 112, 6, 7, 256, 32, 1, false);
__attribute__((used)) static auto* _fp8_25 = &KERNEL_TYPE_FP8(128, 144, 5, 1, 256, 32, 1, false);
__attribute__((used)) static auto* _fp8_26 = &KERNEL_TYPE_FP8(128, 144, 5, 1, 256, 32, 2, false);
__attribute__((used)) static auto* _fp8_27 = &KERNEL_TYPE_FP8(128, 160, 5, 1, 256, 64, 2, false);
__attribute__((used)) static auto* _fp8_28 = &KERNEL_TYPE_FP8(128, 192, 4, 1, 256, 128, 1, false);
__attribute__((used)) static auto* _fp8_29 = &KERNEL_TYPE_FP8(128, 192, 4, 1, 256, 128, 2, false);
__attribute__((used)) static auto* _fp8_2a = &KERNEL_TYPE_FP8(128, 256, 3, 1, 256, 128, 1, false);
__attribute__((used)) static auto* _fp8_2b = &KERNEL_TYPE_FP8(128, 256, 3, 1, 256, 128, 2, false);
__attribute__((used)) static auto* _fp8_2c = &KERNEL_TYPE_FP8(128, 256, 3, 1, 256, 128, 2, true);

// ── M-Grouped Contiguous BF16 kernel instantiations ────────────────
// Same kernel template as normal BF16 but with GemmType::MGroupedContiguous.
// block_m is always 128 (get_mk_alignment_for_contiguous_layout = 128).

#define KERNEL_TYPE_GROUPED(BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, 1,                                                        \
        128, BLOCK_N, 64,                                                   \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, 256,                                                   \
        NUM_MC, MC_ON_A, 132,                                              \
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t>

__attribute__((used)) static auto* _grp_00 = &KERNEL_TYPE_GROUPED(16, 12, 32, 1, false);
__attribute__((used)) static auto* _grp_00a = &KERNEL_TYPE_GROUPED(16, 12, 32, 2, false);
__attribute__((used)) static auto* _grp_00b = &KERNEL_TYPE_GROUPED(16, 12, 32, 2, true);
__attribute__((used)) static auto* _grp_01 = &KERNEL_TYPE_GROUPED(32, 10, 64, 1, false);
__attribute__((used)) static auto* _grp_01a = &KERNEL_TYPE_GROUPED(32, 10, 64, 2, false);
__attribute__((used)) static auto* _grp_01b = &KERNEL_TYPE_GROUPED(32, 10, 64, 2, true);
__attribute__((used)) static auto* _grp_02 = &KERNEL_TYPE_GROUPED(48, 9, 32, 1, false);
__attribute__((used)) static auto* _grp_03 = &KERNEL_TYPE_GROUPED(48, 9, 32, 2, true);
__attribute__((used)) static auto* _grp_04 = &KERNEL_TYPE_GROUPED(64, 8, 128, 1, false);
__attribute__((used)) static auto* _grp_05 = &KERNEL_TYPE_GROUPED(64, 8, 128, 2, true);
__attribute__((used)) static auto* _grp_06 = &KERNEL_TYPE_GROUPED(80, 7, 32, 1, false);
__attribute__((used)) static auto* _grp_07 = &KERNEL_TYPE_GROUPED(80, 7, 32, 2, false);
__attribute__((used)) static auto* _grp_08 = &KERNEL_TYPE_GROUPED(80, 7, 32, 2, true);
__attribute__((used)) static auto* _grp_09 = &KERNEL_TYPE_GROUPED(96, 7, 64, 1, false);
__attribute__((used)) static auto* _grp_0a = &KERNEL_TYPE_GROUPED(96, 7, 64, 2, true);
__attribute__((used)) static auto* _grp_0b = &KERNEL_TYPE_GROUPED(112, 6, 32, 1, false);
__attribute__((used)) static auto* _grp_0b1 = &KERNEL_TYPE_GROUPED(112, 6, 32, 2, false);
__attribute__((used)) static auto* _grp_0c = &KERNEL_TYPE_GROUPED(128, 6, 128, 1, false);
__attribute__((used)) static auto* _grp_0d = &KERNEL_TYPE_GROUPED(128, 6, 128, 2, false);
__attribute__((used)) static auto* _grp_0e = &KERNEL_TYPE_GROUPED(144, 5, 32, 1, false);
__attribute__((used)) static auto* _grp_0f = &KERNEL_TYPE_GROUPED(144, 5, 32, 2, false);
__attribute__((used)) static auto* _grp_10 = &KERNEL_TYPE_GROUPED(160, 5, 64, 1, false);
__attribute__((used)) static auto* _grp_11 = &KERNEL_TYPE_GROUPED(160, 5, 64, 2, false);
__attribute__((used)) static auto* _grp_12 = &KERNEL_TYPE_GROUPED(160, 5, 64, 2, true);
__attribute__((used)) static auto* _grp_13 = &KERNEL_TYPE_GROUPED(176, 4, 32, 1, false);
__attribute__((used)) static auto* _grp_14 = &KERNEL_TYPE_GROUPED(176, 4, 32, 2, false);
__attribute__((used)) static auto* _grp_15 = &KERNEL_TYPE_GROUPED(176, 4, 32, 2, true);
__attribute__((used)) static auto* _grp_16 = &KERNEL_TYPE_GROUPED(192, 4, 128, 1, false);
__attribute__((used)) static auto* _grp_17 = &KERNEL_TYPE_GROUPED(192, 4, 128, 2, false);
__attribute__((used)) static auto* _grp_18 = &KERNEL_TYPE_GROUPED(192, 4, 128, 2, true);
__attribute__((used)) static auto* _grp_19 = &KERNEL_TYPE_GROUPED(208, 4, 32, 1, false);
__attribute__((used)) static auto* _grp_1a = &KERNEL_TYPE_GROUPED(208, 4, 32, 2, false);
__attribute__((used)) static auto* _grp_1b = &KERNEL_TYPE_GROUPED(208, 4, 32, 2, true);
__attribute__((used)) static auto* _grp_1c = &KERNEL_TYPE_GROUPED(224, 3, 64, 1, false);
__attribute__((used)) static auto* _grp_1d = &KERNEL_TYPE_GROUPED(224, 3, 64, 2, false);
__attribute__((used)) static auto* _grp_1e = &KERNEL_TYPE_GROUPED(224, 3, 64, 2, true);
__attribute__((used)) static auto* _grp_1f = &KERNEL_TYPE_GROUPED(240, 3, 32, 1, false);
__attribute__((used)) static auto* _grp_20 = &KERNEL_TYPE_GROUPED(240, 3, 32, 2, false);
__attribute__((used)) static auto* _grp_21 = &KERNEL_TYPE_GROUPED(240, 3, 32, 2, true);
__attribute__((used)) static auto* _grp_22 = &KERNEL_TYPE_GROUPED(256, 3, 128, 1, false);
__attribute__((used)) static auto* _grp_23 = &KERNEL_TYPE_GROUPED(256, 3, 128, 2, false);
__attribute__((used)) static auto* _grp_24 = &KERNEL_TYPE_GROUPED(256, 3, 128, 2, true);

// ── M-Grouped Contiguous FP8 1D2D kernel instantiations ────────────
// Same kernel as normal FP8 1D2D but with GemmType::MGroupedContiguous.
// block_m is always 128.

#define KERNEL_TYPE_FP8_GROUPED(BLOCK_N, STAGES, LAST_STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_fp8_gemm_1d2d_impl<                                       \
        cute::UMMA::Major::MN,                                                 \
        0, 0, 0, 1,                                                            \
        128, BLOCK_N, 128,                                                     \
        128, 128, SWIZZLE_D,                                                   \
        STAGES, LAST_STAGES,                                                   \
        128, 256,                                                              \
        NUM_MC, MC_ON_A, 132,                                                  \
        GemmType::MGroupedContiguous,                                          \
        deep_gemm::EpilogueIdentity>

// block_m=128, no multicast (stages computed for bm=128 FP8: smem_a=128*128=16384)
__attribute__((used)) static auto* _fp8g_00 = &KERNEL_TYPE_FP8_GROUPED(16, 12, 0, 32, 1, false);
__attribute__((used)) static auto* _fp8g_01 = &KERNEL_TYPE_FP8_GROUPED(32, 10, 0, 64, 1, false);
__attribute__((used)) static auto* _fp8g_02 = &KERNEL_TYPE_FP8_GROUPED(48, 9, 1, 32, 1, false);
__attribute__((used)) static auto* _fp8g_03 = &KERNEL_TYPE_FP8_GROUPED(64, 8, 0, 128, 1, false);
__attribute__((used)) static auto* _fp8g_04 = &KERNEL_TYPE_FP8_GROUPED(80, 7, 2, 32, 1, false);
__attribute__((used)) static auto* _fp8g_05 = &KERNEL_TYPE_FP8_GROUPED(96, 7, 3, 64, 1, false);
__attribute__((used)) static auto* _fp8g_06 = &KERNEL_TYPE_FP8_GROUPED(112, 6, 7, 32, 1, false);
__attribute__((used)) static auto* _fp8g_07 = &KERNEL_TYPE_FP8_GROUPED(128, 5, 0, 128, 1, false);
__attribute__((used)) static auto* _fp8g_08 = &KERNEL_TYPE_FP8_GROUPED(144, 5, 1, 32, 1, false);
__attribute__((used)) static auto* _fp8g_09 = &KERNEL_TYPE_FP8_GROUPED(160, 5, 1, 64, 1, false);
__attribute__((used)) static auto* _fp8g_0a = &KERNEL_TYPE_FP8_GROUPED(192, 4, 1, 128, 1, false);
__attribute__((used)) static auto* _fp8g_0b = &KERNEL_TYPE_FP8_GROUPED(256, 3, 1, 128, 1, false);
// block_m=128, multicast
__attribute__((used)) static auto* _fp8g_10 = &KERNEL_TYPE_FP8_GROUPED(16, 12, 0, 32, 2, true);
__attribute__((used)) static auto* _fp8g_11 = &KERNEL_TYPE_FP8_GROUPED(32, 10, 0, 64, 2, true);
__attribute__((used)) static auto* _fp8g_12 = &KERNEL_TYPE_FP8_GROUPED(48, 9, 1, 32, 2, true);
__attribute__((used)) static auto* _fp8g_13 = &KERNEL_TYPE_FP8_GROUPED(64, 8, 0, 128, 2, true);
__attribute__((used)) static auto* _fp8g_14 = &KERNEL_TYPE_FP8_GROUPED(80, 7, 2, 32, 2, true);
__attribute__((used)) static auto* _fp8g_15 = &KERNEL_TYPE_FP8_GROUPED(96, 7, 3, 64, 2, true);
__attribute__((used)) static auto* _fp8g_16 = &KERNEL_TYPE_FP8_GROUPED(112, 6, 7, 32, 2, false);
__attribute__((used)) static auto* _fp8g_17 = &KERNEL_TYPE_FP8_GROUPED(128, 5, 0, 128, 2, false);
__attribute__((used)) static auto* _fp8g_18 = &KERNEL_TYPE_FP8_GROUPED(144, 5, 1, 32, 2, false);
__attribute__((used)) static auto* _fp8g_19 = &KERNEL_TYPE_FP8_GROUPED(160, 5, 1, 64, 2, false);
__attribute__((used)) static auto* _fp8g_1a = &KERNEL_TYPE_FP8_GROUPED(192, 4, 1, 128, 2, false);
__attribute__((used)) static auto* _fp8g_1b = &KERNEL_TYPE_FP8_GROUPED(256, 3, 1, 128, 2, false);
__attribute__((used)) static auto* _fp8g_1c = &KERNEL_TYPE_FP8_GROUPED(256, 3, 1, 128, 2, true);

// ── Kernel dispatch ─────────────────────────────────────────────────

static const void* get_kernel(const KernelConfig& cfg) {
    #define MATCH(BM, BN, ST, NM, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_math_threads == NM && cfg.swizzle_d == SD && \
            cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE(BM, BN, ST, NM, SD, MC, MCA);

    MATCH(16, 16, 32, 128, 32, 1, false)
    MATCH(16, 32, 32, 128, 64, 1, false)
    MATCH(16, 48, 28, 128, 32, 1, false)
    MATCH(16, 64, 22, 128, 128, 1, false)
    MATCH(16, 96, 15, 128, 64, 1, false)
    MATCH(16, 112, 13, 128, 32, 1, false)
    MATCH(16, 224, 7, 128, 64, 1, false)
    MATCH(16, 240, 6, 128, 32, 1, false)
    MATCH(32, 16, 32, 128, 32, 1, false)
    MATCH(32, 32, 28, 128, 64, 1, false)
    MATCH(32, 48, 22, 128, 32, 1, false)
    MATCH(32, 64, 18, 128, 128, 1, false)
    MATCH(32, 96, 13, 128, 64, 1, false)
    MATCH(32, 112, 12, 128, 32, 1, false)
    MATCH(32, 224, 6, 128, 64, 1, false)
    MATCH(32, 240, 6, 128, 32, 1, false)
    MATCH(64, 16, 22, 128, 32, 1, false)
    MATCH(64, 32, 18, 128, 64, 1, false)
    MATCH(64, 48, 15, 128, 32, 1, false)
    MATCH(64, 64, 13, 128, 128, 1, false)
    MATCH(64, 64, 13, 128, 128, 2, false)
    MATCH(64, 80, 12, 128, 32, 1, false)
    MATCH(64, 80, 12, 128, 32, 2, true)
    MATCH(64, 96, 10, 128, 64, 1, false)
    MATCH(64, 96, 10, 128, 64, 2, true)
    MATCH(64, 112, 9, 128, 32, 1, false)
    MATCH(64, 112, 9, 128, 32, 2, false)
    MATCH(64, 128, 8, 128, 128, 1, false)
    MATCH(64, 128, 8, 128, 128, 2, false)
    MATCH(64, 144, 8, 128, 32, 1, false)
    MATCH(64, 160, 7, 128, 64, 1, false)
    MATCH(64, 160, 7, 128, 64, 2, true)
    MATCH(64, 176, 6, 128, 32, 1, false)
    MATCH(64, 176, 6, 128, 32, 2, false)
    MATCH(64, 192, 6, 128, 128, 1, false)
    MATCH(64, 192, 6, 128, 128, 2, false)
    MATCH(64, 192, 6, 128, 128, 2, true)
    MATCH(64, 208, 5, 128, 32, 2, false)
    MATCH(64, 208, 5, 128, 32, 2, true)
    MATCH(64, 224, 5, 128, 64, 1, false)
    MATCH(64, 224, 5, 128, 64, 2, false)
    MATCH(64, 240, 5, 128, 32, 1, false)
    MATCH(64, 240, 5, 128, 32, 2, false)
    MATCH(64, 240, 5, 128, 32, 2, true)
    MATCH(64, 256, 4, 128, 128, 1, false)
    MATCH(64, 256, 4, 128, 128, 2, false)
    MATCH(64, 256, 4, 128, 128, 2, true)
    MATCH(128, 16, 12, 256, 32, 1, false)
    MATCH(128, 32, 10, 256, 64, 1, false)
    MATCH(128, 48, 9, 256, 32, 1, false)
    MATCH(128, 48, 9, 256, 32, 2, true)
    MATCH(128, 64, 8, 256, 128, 1, false)
    MATCH(128, 64, 8, 256, 128, 2, true)
    MATCH(128, 80, 7, 256, 32, 1, false)
    MATCH(128, 80, 7, 256, 32, 2, false)
    MATCH(128, 80, 7, 256, 32, 2, true)
    MATCH(128, 96, 7, 256, 64, 1, false)
    MATCH(128, 96, 7, 256, 64, 2, true)
    MATCH(128, 112, 6, 256, 32, 2, false)
    MATCH(128, 128, 6, 256, 128, 1, false)
    MATCH(128, 128, 6, 256, 128, 2, false)
    MATCH(128, 144, 5, 256, 32, 1, false)
    MATCH(128, 144, 5, 256, 32, 2, false)
    MATCH(128, 160, 5, 256, 64, 1, false)
    MATCH(128, 160, 5, 256, 64, 2, false)
    MATCH(128, 160, 5, 256, 64, 2, true)
    MATCH(128, 176, 4, 256, 32, 1, false)
    MATCH(128, 176, 4, 256, 32, 2, false)
    MATCH(128, 176, 4, 256, 32, 2, true)
    MATCH(128, 192, 4, 256, 128, 1, false)
    MATCH(128, 192, 4, 256, 128, 2, false)
    MATCH(128, 192, 4, 256, 128, 2, true)
    MATCH(128, 208, 4, 256, 32, 1, false)
    MATCH(128, 208, 4, 256, 32, 2, false)
    MATCH(128, 208, 4, 256, 32, 2, true)
    MATCH(128, 224, 3, 256, 64, 1, false)
    MATCH(128, 224, 3, 256, 64, 2, false)
    MATCH(128, 224, 3, 256, 64, 2, true)
    MATCH(128, 240, 3, 256, 32, 1, false)
    MATCH(128, 240, 3, 256, 32, 2, false)
    MATCH(128, 240, 3, 256, 32, 2, true)
    MATCH(128, 256, 3, 256, 128, 1, false)
    MATCH(128, 256, 3, 256, 128, 2, false)
    MATCH(128, 256, 3, 256, 128, 2, true)
    MATCH(256, 16, 6, 256, 32, 1, false)
    MATCH(256, 32, 5, 256, 64, 2, true)
    MATCH(256, 48, 5, 256, 32, 1, false)
    MATCH(256, 48, 5, 256, 32, 2, true)
    MATCH(256, 64, 4, 256, 128, 2, false)
    MATCH(256, 64, 4, 256, 128, 2, true)
    MATCH(256, 80, 4, 256, 32, 1, false)
    MATCH(256, 80, 4, 256, 32, 2, false)
    MATCH(256, 80, 4, 256, 32, 2, true)
    MATCH(256, 96, 4, 256, 64, 2, true)
    MATCH(256, 112, 3, 256, 32, 1, false)
    MATCH(256, 112, 3, 256, 32, 2, false)
    MATCH(256, 112, 3, 256, 32, 2, true)
    MATCH(256, 128, 3, 256, 128, 2, false)
    MATCH(256, 128, 3, 256, 128, 2, true)
    MATCH(16, 80, 18, 128, 32, 1, false)
    MATCH(16, 208, 7, 128, 32, 1, false)
    MATCH(16, 256, 6, 128, 128, 1, false)
    MATCH(64, 144, 8, 128, 32, 2, false)
    MATCH(64, 160, 7, 128, 64, 2, false)
    MATCH(64, 208, 5, 128, 32, 1, false)

    #undef MATCH
    return nullptr;
}

// ── FP8 kernel dispatch ─────────────────────────────────────────────

static const void* get_fp8_kernel(const FP8Config& cfg) {
    #define MATCH_FP8(BM, BN, ST, NLS, NM, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_last_stages == NLS && cfg.num_math_threads == NM && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_FP8(BM, BN, ST, NLS, NM, SD, MC, MCA);

    MATCH_FP8(16, 16, 32, 0, 128, 32, 1, false)
    MATCH_FP8(16, 32, 32, 0, 128, 64, 1, false)
    MATCH_FP8(16, 48, 27, 1, 128, 32, 1, false)
    MATCH_FP8(16, 64, 22, 0, 128, 128, 1, false)
    MATCH_FP8(16, 80, 18, 2, 128, 32, 1, false)
    MATCH_FP8(16, 96, 15, 3, 128, 64, 1, false)
    MATCH_FP8(16, 112, 13, 7, 128, 32, 1, false)
    MATCH_FP8(16, 256, 6, 1, 128, 128, 1, false)
    MATCH_FP8(32, 16, 32, 0, 128, 32, 1, false)
    MATCH_FP8(32, 32, 27, 0, 128, 64, 1, false)
    MATCH_FP8(32, 48, 21, 1, 128, 32, 1, false)
    MATCH_FP8(32, 48, 22, 1, 128, 32, 1, false)
    MATCH_FP8(32, 64, 18, 0, 128, 128, 1, false)
    MATCH_FP8(32, 80, 15, 2, 128, 32, 1, false)
    MATCH_FP8(32, 96, 13, 3, 128, 64, 1, false)
    MATCH_FP8(32, 112, 12, 7, 128, 32, 1, false)
    MATCH_FP8(32, 256, 5, 1, 128, 128, 1, false)
    MATCH_FP8(64, 16, 21, 0, 128, 32, 1, false)
    MATCH_FP8(64, 32, 18, 0, 128, 64, 1, false)
    MATCH_FP8(64, 48, 15, 1, 128, 32, 1, false)
    MATCH_FP8(64, 64, 13, 0, 128, 128, 1, false)
    MATCH_FP8(64, 64, 13, 0, 128, 128, 2, false)
    MATCH_FP8(64, 80, 11, 2, 128, 32, 1, false)
    MATCH_FP8(64, 96, 10, 3, 128, 64, 1, false)
    MATCH_FP8(64, 112, 9, 7, 128, 32, 1, false)
    MATCH_FP8(64, 128, 8, 0, 128, 128, 1, false)
    MATCH_FP8(64, 128, 8, 0, 128, 128, 2, false)
    MATCH_FP8(64, 144, 7, 1, 128, 32, 1, false)
    MATCH_FP8(64, 144, 7, 1, 128, 32, 2, false)
    MATCH_FP8(64, 160, 7, 1, 128, 64, 1, false)
    MATCH_FP8(64, 160, 7, 1, 128, 64, 2, false)
    MATCH_FP8(64, 192, 6, 1, 128, 128, 1, false)
    MATCH_FP8(64, 192, 6, 1, 128, 128, 2, false)
    MATCH_FP8(64, 256, 4, 1, 128, 128, 1, false)
    MATCH_FP8(64, 256, 4, 1, 128, 128, 2, false)
    MATCH_FP8(64, 256, 4, 1, 128, 128, 2, true)
    MATCH_FP8(128, 112, 6, 7, 256, 32, 1, false)
    MATCH_FP8(128, 144, 5, 1, 256, 32, 1, false)
    MATCH_FP8(128, 144, 5, 1, 256, 32, 2, false)
    MATCH_FP8(128, 160, 5, 1, 256, 64, 2, false)
    MATCH_FP8(128, 192, 4, 1, 256, 128, 1, false)
    MATCH_FP8(128, 192, 4, 1, 256, 128, 2, false)
    MATCH_FP8(128, 256, 3, 1, 256, 128, 1, false)
    MATCH_FP8(128, 256, 3, 1, 256, 128, 2, false)
    MATCH_FP8(128, 256, 3, 1, 256, 128, 2, true)

    #undef MATCH_FP8
    return nullptr;
}

// ── Grouped kernel dispatch ─────────────────────────────────────────

static const void* get_grouped_kernel(const KernelConfig& cfg) {
    // block_m is always 128, num_math is always 256 for grouped GEMM
    #define MATCH_GRP(BN, ST, SD, MC, MCA) \
        if (cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_GROUPED(BN, ST, SD, MC, MCA);

    MATCH_GRP(16, 12, 32, 1, false)
    MATCH_GRP(16, 12, 32, 2, false)
    MATCH_GRP(16, 12, 32, 2, true)
    MATCH_GRP(32, 10, 64, 1, false)
    MATCH_GRP(32, 10, 64, 2, false)
    MATCH_GRP(32, 10, 64, 2, true)
    MATCH_GRP(48, 9, 32, 1, false)
    MATCH_GRP(48, 9, 32, 2, true)
    MATCH_GRP(64, 8, 128, 1, false)
    MATCH_GRP(64, 8, 128, 2, true)
    MATCH_GRP(80, 7, 32, 1, false)
    MATCH_GRP(80, 7, 32, 2, false)
    MATCH_GRP(80, 7, 32, 2, true)
    MATCH_GRP(96, 7, 64, 1, false)
    MATCH_GRP(96, 7, 64, 2, true)
    MATCH_GRP(112, 6, 32, 1, false)
    MATCH_GRP(112, 6, 32, 2, false)
    MATCH_GRP(128, 6, 128, 1, false)
    MATCH_GRP(128, 6, 128, 2, false)
    MATCH_GRP(144, 5, 32, 1, false)
    MATCH_GRP(144, 5, 32, 2, false)
    MATCH_GRP(160, 5, 64, 1, false)
    MATCH_GRP(160, 5, 64, 2, false)
    MATCH_GRP(160, 5, 64, 2, true)
    MATCH_GRP(176, 4, 32, 1, false)
    MATCH_GRP(176, 4, 32, 2, false)
    MATCH_GRP(176, 4, 32, 2, true)
    MATCH_GRP(192, 4, 128, 1, false)
    MATCH_GRP(192, 4, 128, 2, false)
    MATCH_GRP(192, 4, 128, 2, true)
    MATCH_GRP(208, 4, 32, 1, false)
    MATCH_GRP(208, 4, 32, 2, false)
    MATCH_GRP(208, 4, 32, 2, true)
    MATCH_GRP(224, 3, 64, 1, false)
    MATCH_GRP(224, 3, 64, 2, false)
    MATCH_GRP(224, 3, 64, 2, true)
    MATCH_GRP(240, 3, 32, 1, false)
    MATCH_GRP(240, 3, 32, 2, false)
    MATCH_GRP(240, 3, 32, 2, true)
    MATCH_GRP(256, 3, 128, 1, false)
    MATCH_GRP(256, 3, 128, 2, false)
    MATCH_GRP(256, 3, 128, 2, true)

    #undef MATCH_GRP
    return nullptr;
}

// ── FP8 grouped kernel dispatch ─────────────────────────────────────

static const void* get_fp8_grouped_kernel(const FP8Config& cfg) {
    #define MATCH_FP8G(BN, ST, NLS, SD, MC, MCA) \
        if (cfg.block_n == BN && cfg.num_stages == ST && cfg.num_last_stages == NLS && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_FP8_GROUPED(BN, ST, NLS, SD, MC, MCA);

    // no multicast
    MATCH_FP8G(16, 12, 0, 32, 1, false)
    MATCH_FP8G(32, 10, 0, 64, 1, false)
    MATCH_FP8G(48, 9, 1, 32, 1, false)
    MATCH_FP8G(64, 8, 0, 128, 1, false)
    MATCH_FP8G(80, 7, 2, 32, 1, false)
    MATCH_FP8G(96, 7, 3, 64, 1, false)
    MATCH_FP8G(112, 6, 7, 32, 1, false)
    MATCH_FP8G(128, 5, 0, 128, 1, false)
    MATCH_FP8G(144, 5, 1, 32, 1, false)
    MATCH_FP8G(160, 5, 1, 64, 1, false)
    MATCH_FP8G(192, 4, 1, 128, 1, false)
    MATCH_FP8G(256, 3, 1, 128, 1, false)
    // multicast
    MATCH_FP8G(16, 12, 0, 32, 2, true)
    MATCH_FP8G(32, 10, 0, 64, 2, true)
    MATCH_FP8G(48, 9, 1, 32, 2, true)
    MATCH_FP8G(64, 8, 0, 128, 2, true)
    MATCH_FP8G(80, 7, 2, 32, 2, true)
    MATCH_FP8G(96, 7, 3, 64, 2, true)
    MATCH_FP8G(112, 6, 7, 32, 2, false)
    MATCH_FP8G(128, 5, 0, 128, 2, false)
    MATCH_FP8G(144, 5, 1, 32, 2, false)
    MATCH_FP8G(160, 5, 1, 64, 2, false)
    MATCH_FP8G(192, 4, 1, 128, 2, false)
    MATCH_FP8G(256, 3, 1, 128, 2, false)
    MATCH_FP8G(256, 3, 1, 128, 2, true)

    #undef MATCH_FP8G
    return nullptr;
}

// ── GPU architecture detection ──────────────────────────────────────

static int g_gpu_arch = 0;
static void ensure_gpu_arch() {
    if (g_gpu_arch > 0) return;
    int dev; cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    g_gpu_arch = major * 10 + minor;
}

// SM100 functions (defined later in this file)
static int sm100_bf16_gemm_impl_wrapper(void*, void*, void*, int, int, int, void*);
static void sm100_query_config_impl(int, int, int, int*, int*, int*, int*);

// ── C FFI ───────────────────────────────────────────────────────────

extern "C" {

/// BF16 GEMM: D = A[M,K] @ B[K,N]  (A row-major, B col-major = weight[N,K] transposed)
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
/// Auto-dispatches to SM100 (Blackwell) when detected.
int deepgemm_bf16_gemm(
    void* A, void* B, void* D,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    ensure_gpu_arch();

    // Route to SM100 on Blackwell GPUs
    if (g_gpu_arch >= 100)
        return sm100_bf16_gemm_impl_wrapper(A, B, D, M, N, K, stream);

    // Clear any stale async errors from prior operations
    cudaGetLastError();

    auto cfg = select_config(M, N, K, g_num_sms);

    auto kernel_ptr = get_kernel(cfg);
    if (!kernel_ptr) {
        return -1;
    }

    // Create TMA descriptors
    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    auto tma_d = make_2d_tma(D, N, M,
                             cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n,
                             cfg.block_m, N, cfg.swizzle_d);

    int num_threads = cfg.num_tma_threads + cfg.num_math_threads;
    dim3 grid(g_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaFuncSetAttribute(kernel_ptr,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          cfg.smem_size);

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

/// Query which config would be selected for a given shape (for debugging).
/// Auto-dispatches to SM100 when detected.
void deepgemm_query_config(int M, int N, int K,
                            int* out_block_m, int* out_block_n,
                            int* out_stages, int* out_smem) {
    ensure_num_sms();
    ensure_gpu_arch();
    if (g_gpu_arch >= 100) {
        sm100_query_config_impl(M, N, K, out_block_m, out_block_n, out_stages, out_smem);
        return;
    }
    auto cfg = select_config(M, N, K, g_num_sms);
    *out_block_m = cfg.block_m;
    *out_block_n = cfg.block_n;
    *out_stages = cfg.num_stages;
    *out_smem = cfg.smem_size;
}

/// FP8 E4M3 GEMM (1D2D): D[M,N] = (scale_a ⊗ A_fp8[M,K]) @ (scale_b ⊗ B_fp8[K,N])
/// A: [M,K] row-major FP8 E4M3
/// B: [K,N] col-major FP8 E4M3 (= weight [N,K] row-major, transposed)
/// D: [M,N] row-major BF16 output
/// scale_a: [ceil(K/128), align(M,4)] FP32, M values contiguous (MN-major)
/// scale_b: [ceil(K/128), align(N,4)] FP32, N values contiguous (MN-major), accessed from global memory
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    int M, int N, int K,
    void* stream
) {
    ensure_num_sms();
    cudaGetLastError();

    auto cfg = select_fp8_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_fp8_kernel(cfg);
    if (!kernel_ptr) return -1;

    // TMA for A/B (FP8, 1-byte elements, K-major)
    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_u8(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);

    // TMA for D (BF16 output, with swizzle)
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);

    // TMA for SFA only (FP32, MN-major, no swizzle)
    // SFB is accessed from global memory directly (1D2D kernel design)
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };
    int sfa_inner = align4(M);
    int k_scales = (K + 127) / 128;
    auto tma_sfa = make_2d_tma_f32(scale_a, sfa_inner, k_scales, cfg.block_m, 1, sfa_inner, 0);

    int num_threads = cfg.num_tma_threads + cfg.num_math_threads;
    dim3 grid(g_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size);

    // 1D2D kernel signature: (float* sfb, int* grouped_layout, shape_m/n/k, tma_a, tma_b, tma_d, tma_sfa)
    float* sfb_ptr = (float*)scale_b;
    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;

    void* args[] = {
        &sfb_ptr,
        &grouped_layout,
        &um, &un, &uk,
        &tma_a, &tma_b, &tma_d, &tma_sfa
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

/// M-Grouped Contiguous FP8 GEMM (1D2D, for MoE):
///   D[total_M, N] = grouped(scale_a ⊗ A_fp8[total_M, K], scale_b ⊗ B_fp8[G, N, K])
/// A: [total_M, K] FP8 E4M3 (shared, K-major)
/// B: [G, N, K] FP8 E4M3 (per-group, K-major)
/// D: [total_M, N] BF16 output
/// scale_a: [ceil(K/128), align(total_M, 4)] FP32 (per-token, via TMA)
/// scale_b: [ceil(K/128), align(N, 4)] FP32 (per-channel, global memory)
/// grouped_layout: [total_M] int32, each group aligned to 128
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
    cudaGetLastError();

    auto cfg = select_fp8_grouped_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_fp8_grouped_kernel(cfg);
    if (!kernel_ptr) return -1;

    // TMA for A [total_M, K] FP8, num_groups=1
    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    // TMA for B [G, N, K] FP8, outer_dim = N * num_groups
    auto tma_b = make_2d_tma_u8(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // TMA for D [total_M, N] BF16, num_groups=1
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);
    // TMA for SFA (per-token scaling, num_groups=1)
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };
    int sfa_inner = align4(M);
    int k_scales = (K + 127) / 128;
    auto tma_sfa = make_2d_tma_f32(scale_a, sfa_inner, k_scales, cfg.block_m, 1, sfa_inner, 0);

    int num_threads = cfg.num_tma_threads + cfg.num_math_threads;
    dim3 grid(g_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size);

    // 1D2D kernel signature: (float* sfb, int* grouped_layout, shape_m/n/k, tma_a, tma_b, tma_d, tma_sfa)
    float* sfb_ptr = (float*)scale_b;
    int* layout_ptr = (int*)grouped_layout;
    uint32_t um = M, un = N, uk = K;
    void* args[] = {
        &sfb_ptr, &layout_ptr,
        &um, &un, &uk,
        &tma_a, &tma_b, &tma_d, &tma_sfa
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
            cudaGetLastError(); return -2;
        }
    } else {
        if (cudaLaunchKernel(kernel_ptr, grid, block, args, cfg.smem_size, s) != cudaSuccess) {
            cudaGetLastError(); return -2;
        }
    }
    return 0;
}

/// M-Grouped Contiguous BF16 GEMM (for MoE):
///   D[total_M, N] = grouped(A[total_M, K], B[G, N, K], grouped_layout[total_M])
/// A: [total_M, K] row-major BF16 (shared across groups, K-major for TMA)
/// B: [G, N, K] row-major BF16 (per-group weights, K-major for TMA)
/// D: [total_M, N] row-major BF16 output
/// grouped_layout: [total_M] int32, grouped_layout[r] = group index for row r
///   Each group's rows must be contiguous and aligned to 128.
/// Returns 0 on success, -1 if no kernel variant, -2 on launch error.
int deepgemm_m_grouped_bf16_gemm(
    void* A, void* B, void* D,
    void* grouped_layout,
    int M, int N, int K,
    int num_groups,
    void* stream
) {
    ensure_num_sms();
    cudaGetLastError();

    auto cfg = select_grouped_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_grouped_kernel(cfg);
    if (!kernel_ptr) return -1;

    // TMA for A [total_M, K] K-major (same as normal GEMM, num_groups=1)
    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);

    // TMA for B [G, N, K] K-major: outer_dim = N * num_groups
    // (upstream: make_tma_b_desc with num_groups applied to outer dimension)
    auto tma_b = make_2d_tma(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);

    // TMA for D [total_M, N] (same as normal GEMM, num_groups=1)
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);

    int num_threads = cfg.num_tma_threads + cfg.num_math_threads;
    dim3 grid(g_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size);

    // Same kernel signature as normal BF16: (int* grouped_layout, shape_m, shape_n, shape_k, tma_a, tma_b, tma_cd)
    int* layout_ptr = (int*)grouped_layout;
    uint32_t um = M, un = N, uk = K;
    void* args[] = {
        &layout_ptr,
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

} // extern "C"

// ════════════════════════════════════════════════════════════════════
// SM100 (Blackwell) BF16 GEMM — same file, compiled for both SM90a + SM100a.
// SM90 kernel bodies are guarded by __CUDA_ARCH__ >= 900 && < 1000,
// SM100 kernel bodies by __CUDA_ARCH__ >= 1000, so fat binary works.
// ════════════════════════════════════════════════════════════════════

struct SM100Config {
    int block_m, block_n, block_k;
    int num_stages;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

static SM100Config select_sm100_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;

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

    // SM100 multicast: only on B (A multicast forbidden)
    int multicast = 1;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal = (ceil_div(m, best_bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    int sw_a = get_swizzle(block_k);
    int sw_b = get_swizzle(block_k);
    int sw_d = get_swizzle(best_bn);

    // SM100 SMEM: smem_cd = min(bm,128)*sw_d*2, barrier = s*24+40, tmem_ptr = 4
    const int smem_capacity = 232448;
    int smem_cd = std::min(best_bm, 128) * sw_d * 2;
    int smem_a_per = best_bm * block_k * 2;
    int smem_b_per = best_bn * block_k * 2;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int total = smem_cd + s * (smem_a_per + smem_b_per) + s * 24 + 44;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return SM100Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast, .multicast_on_a = false,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── SM100 kernel instantiations ────────────────────────────────────

#define KERNEL_TYPE_SM100(BLOCK_M, BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm100_bf16_gemm_impl<                                            \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                            \
        0, 0, 0,                                                                \
        BLOCK_M, BLOCK_N, 64, 1,                                                \
        128, 128, SWIZZLE_D,                                                    \
        STAGES, 128, 128,                                                       \
        NUM_MC, MC_ON_A, 132,                                                   \
        GemmType::Normal, false, cutlass::bfloat16_t, 100>

__attribute__((used)) static auto* _s100_00 = &KERNEL_TYPE_SM100(32, 16, 32, 32, 1, false);
__attribute__((used)) static auto* _s100_01 = &KERNEL_TYPE_SM100(32, 32, 27, 64, 1, false);
__attribute__((used)) static auto* _s100_02 = &KERNEL_TYPE_SM100(32, 64, 22, 128, 1, false);
__attribute__((used)) static auto* _s100_03 = &KERNEL_TYPE_SM100(32, 96, 18, 64, 1, false);
__attribute__((used)) static auto* _s100_04 = &KERNEL_TYPE_SM100(32, 128, 15, 128, 1, false);
__attribute__((used)) static auto* _s100_10 = &KERNEL_TYPE_SM100(64, 16, 32, 32, 1, false);
__attribute__((used)) static auto* _s100_11 = &KERNEL_TYPE_SM100(64, 32, 22, 64, 1, false);
__attribute__((used)) static auto* _s100_12 = &KERNEL_TYPE_SM100(64, 64, 15, 128, 1, false);
__attribute__((used)) static auto* _s100_13 = &KERNEL_TYPE_SM100(64, 96, 12, 64, 1, false);
__attribute__((used)) static auto* _s100_14 = &KERNEL_TYPE_SM100(64, 128, 10, 128, 1, false);
__attribute__((used)) static auto* _s100_20 = &KERNEL_TYPE_SM100(128, 16, 12, 32, 1, false);
__attribute__((used)) static auto* _s100_21 = &KERNEL_TYPE_SM100(128, 32, 10, 64, 1, false);
__attribute__((used)) static auto* _s100_22 = &KERNEL_TYPE_SM100(128, 64, 8, 128, 1, false);
__attribute__((used)) static auto* _s100_23 = &KERNEL_TYPE_SM100(128, 96, 7, 64, 1, false);
__attribute__((used)) static auto* _s100_24 = &KERNEL_TYPE_SM100(128, 128, 6, 128, 1, false);
__attribute__((used)) static auto* _s100_25 = &KERNEL_TYPE_SM100(128, 160, 5, 64, 1, false);
__attribute__((used)) static auto* _s100_26 = &KERNEL_TYPE_SM100(128, 192, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_27 = &KERNEL_TYPE_SM100(128, 224, 4, 64, 1, false);
__attribute__((used)) static auto* _s100_28 = &KERNEL_TYPE_SM100(128, 256, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_29 = &KERNEL_TYPE_SM100(128, 16, 12, 32, 2, false);
__attribute__((used)) static auto* _s100_2a = &KERNEL_TYPE_SM100(128, 32, 10, 64, 2, false);
__attribute__((used)) static auto* _s100_2b = &KERNEL_TYPE_SM100(128, 64, 8, 128, 2, false);
__attribute__((used)) static auto* _s100_2c = &KERNEL_TYPE_SM100(128, 96, 7, 64, 2, false);
__attribute__((used)) static auto* _s100_2d = &KERNEL_TYPE_SM100(128, 128, 6, 128, 2, false);
__attribute__((used)) static auto* _s100_2e = &KERNEL_TYPE_SM100(128, 160, 5, 64, 2, false);
__attribute__((used)) static auto* _s100_2f = &KERNEL_TYPE_SM100(128, 192, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_30 = &KERNEL_TYPE_SM100(128, 224, 4, 64, 2, false);
__attribute__((used)) static auto* _s100_31 = &KERNEL_TYPE_SM100(128, 256, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_40 = &KERNEL_TYPE_SM100(256, 16, 6, 32, 1, false);
__attribute__((used)) static auto* _s100_41 = &KERNEL_TYPE_SM100(256, 32, 5, 64, 1, false);
__attribute__((used)) static auto* _s100_42 = &KERNEL_TYPE_SM100(256, 64, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_43 = &KERNEL_TYPE_SM100(256, 96, 4, 64, 1, false);
__attribute__((used)) static auto* _s100_44 = &KERNEL_TYPE_SM100(256, 128, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_45 = &KERNEL_TYPE_SM100(256, 160, 3, 64, 1, false);
__attribute__((used)) static auto* _s100_46 = &KERNEL_TYPE_SM100(256, 192, 3, 128, 1, false);
__attribute__((used)) static auto* _s100_47 = &KERNEL_TYPE_SM100(256, 224, 3, 64, 1, false);
__attribute__((used)) static auto* _s100_48 = &KERNEL_TYPE_SM100(256, 256, 3, 128, 1, false);
__attribute__((used)) static auto* _s100_49 = &KERNEL_TYPE_SM100(256, 16, 6, 32, 2, false);
__attribute__((used)) static auto* _s100_4a = &KERNEL_TYPE_SM100(256, 32, 5, 64, 2, false);
__attribute__((used)) static auto* _s100_4b = &KERNEL_TYPE_SM100(256, 64, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_4c = &KERNEL_TYPE_SM100(256, 96, 4, 64, 2, false);
__attribute__((used)) static auto* _s100_4d = &KERNEL_TYPE_SM100(256, 128, 4, 128, 2, false);

// ── SM100 dispatch ─────────────────────────────────────────────────

static const void* get_sm100_kernel(const SM100Config& cfg) {
    #define MATCH_SM100(BM, BN, ST, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_SM100(BM, BN, ST, SD, MC, MCA);

    MATCH_SM100(32, 16, 32, 32, 1, false) MATCH_SM100(32, 32, 27, 64, 1, false)
    MATCH_SM100(32, 64, 22, 128, 1, false) MATCH_SM100(32, 96, 18, 64, 1, false)
    MATCH_SM100(32, 128, 15, 128, 1, false)
    MATCH_SM100(64, 16, 32, 32, 1, false) MATCH_SM100(64, 32, 22, 64, 1, false)
    MATCH_SM100(64, 64, 15, 128, 1, false) MATCH_SM100(64, 96, 12, 64, 1, false)
    MATCH_SM100(64, 128, 10, 128, 1, false)
    MATCH_SM100(128, 16, 12, 32, 1, false) MATCH_SM100(128, 32, 10, 64, 1, false)
    MATCH_SM100(128, 64, 8, 128, 1, false) MATCH_SM100(128, 96, 7, 64, 1, false)
    MATCH_SM100(128, 128, 6, 128, 1, false) MATCH_SM100(128, 160, 5, 64, 1, false)
    MATCH_SM100(128, 192, 4, 128, 1, false) MATCH_SM100(128, 224, 4, 64, 1, false)
    MATCH_SM100(128, 256, 4, 128, 1, false)
    MATCH_SM100(128, 16, 12, 32, 2, false) MATCH_SM100(128, 32, 10, 64, 2, false)
    MATCH_SM100(128, 64, 8, 128, 2, false) MATCH_SM100(128, 96, 7, 64, 2, false)
    MATCH_SM100(128, 128, 6, 128, 2, false) MATCH_SM100(128, 160, 5, 64, 2, false)
    MATCH_SM100(128, 192, 4, 128, 2, false) MATCH_SM100(128, 224, 4, 64, 2, false)
    MATCH_SM100(128, 256, 4, 128, 2, false)
    MATCH_SM100(256, 16, 6, 32, 1, false) MATCH_SM100(256, 32, 5, 64, 1, false)
    MATCH_SM100(256, 64, 4, 128, 1, false) MATCH_SM100(256, 96, 4, 64, 1, false)
    MATCH_SM100(256, 128, 4, 128, 1, false) MATCH_SM100(256, 160, 3, 64, 1, false)
    MATCH_SM100(256, 192, 3, 128, 1, false) MATCH_SM100(256, 224, 3, 64, 1, false)
    MATCH_SM100(256, 256, 3, 128, 1, false)
    MATCH_SM100(256, 16, 6, 32, 2, false) MATCH_SM100(256, 32, 5, 64, 2, false)
    MATCH_SM100(256, 64, 4, 128, 2, false) MATCH_SM100(256, 96, 4, 64, 2, false)
    MATCH_SM100(256, 128, 4, 128, 2, false)

    #undef MATCH_SM100
    return nullptr;
}

// ── SM100 entry points (called from FFI above) ─────────────────────

static int sm100_bf16_gemm_impl_wrapper(
    void* A, void* B, void* D, int M, int N, int K, void* stream
) {
    ensure_num_sms();
    cudaGetLastError();

    auto cfg = select_sm100_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_sm100_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cd_store_bm, N, cfg.swizzle_d);

    dim3 grid(g_num_sms, 1, 1);
    dim3 block(256, 1, 1);

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size);

    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &grouped_layout, &um, &un, &uk, &tma_a, &tma_b, &tma_d };
    auto s = static_cast<cudaStream_t>(stream);

    if (cfg.num_multicast > 1) {
        cudaLaunchConfig_t lc = {};
        lc.gridDim = grid; lc.blockDim = block;
        lc.dynamicSmemBytes = cfg.smem_size; lc.stream = s;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {(unsigned)cfg.num_multicast, 1, 1};
        lc.attrs = attrs; lc.numAttrs = 1;
        if (cudaLaunchKernelExC(&lc, kernel_ptr, args) != cudaSuccess) {
            cudaGetLastError(); return -2;
        }
    } else {
        if (cudaLaunchKernel(kernel_ptr, grid, block, args, cfg.smem_size, s) != cudaSuccess) {
            cudaGetLastError(); return -2;
        }
    }
    return 0;
}

static void sm100_query_config_impl(
    int M, int N, int K, int* bm, int* bn, int* st, int* sm
) {
    ensure_num_sms();
    auto cfg = select_sm100_config(M, N, K, g_num_sms);
    *bm = cfg.block_m; *bn = cfg.block_n; *st = cfg.num_stages; *sm = cfg.smem_size;
}
