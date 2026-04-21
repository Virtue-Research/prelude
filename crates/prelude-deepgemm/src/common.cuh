// Common types, TMA helpers, GPU state, launch helper, and smem config.
// Shared by all sm90/sm100 .cuh files.
// Heuristic logic matches upstream DeepGEMM exactly.

#pragma once

// ── Utility ────────────────────────────────────────────────────────

static int ceil_div_static(int a, int b) { return (a + b - 1) / b; }
static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }
static int gcd_static(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }

// ── Enums for kernel/mma types (mirrors upstream types.hpp) ────────

enum class KernelKind { NoSF, Kernel1D1D, Kernel1D2D };
enum class MmaKindLocal { BF16, MXFP8FP4 };

// ── Swizzle mode (exact upstream get_swizzle_mode) ─────────────────

static int get_swizzle_mode(int block_size, int elem_size) {
    for (int mode : {128, 64, 32, 16})
        if ((block_size * elem_size) % mode == 0) return mode;
    return 16; // unreachable for valid inputs
}

// ── SM90 ArchSpec (exact upstream sm90.hpp) ────────────────────────

struct SM90Arch {
    static constexpr int smem_capacity = 232448;

    static int get_ab_load_block_m(int /*mc*/, int block_m) { return block_m; }
    static int get_ab_load_block_n(int /*mc*/, int block_n) { return block_n; }

    static bool enable_cd_swizzle(int cd_elem_size) {
        return cd_elem_size != 4; // disabled for FP32
    }

    static int get_smem_cd_size(KernelKind /*kt*/, int block_m, int block_n,
                                 int /*sw_cd*/, int cd_elem_size) {
        return align_up(block_m * block_n * cd_elem_size, 1024);
    }

    static void get_sf_smem_per_stage(KernelKind kt, int block_m, int block_n,
                                       MmaKindLocal mma, int& sfa, int& sfb) {
        if (mma == MmaKindLocal::BF16) { sfa = 0; sfb = 0; return; }
        sfa = align_up(block_m * 4, 128);
        sfb = (kt == KernelKind::Kernel1D1D) ? align_up(block_n * 4, 128) : 0;
    }

    static int get_extra_sfb(int k, int block_n, int block_k) {
        int use_uniform = (block_k % block_n == 0) ? 1 : 2;
        return align_up(ceil_div_static(k, block_k) * 4 * use_uniform, 8);
    }

    static int get_barrier_smem_size(int num_stages) { return num_stages * 8 * 2; }
    static int get_tmem_ptr_smem_size() { return 0; }
    static int get_tensormap_smem_size(bool k_grouped) {
        return k_grouped ? 4 * (int)sizeof(CUtensorMap) : 0;
    }

    static bool is_num_stages_legal(MmaKindLocal mma, int block_n, int block_k, int num_stages) {
        if (mma == MmaKindLocal::MXFP8FP4 && block_k % block_n != 0 &&
            block_k / gcd_static(block_n, block_k) <= 4)
            return num_stages <= 4;
        return true;
    }

    static int get_cd_store_block_m(int block_m, bool single_wg_sync = false) {
        return single_wg_sync ? 64 : block_m;
    }
};

// ── SM100 ArchSpec (exact upstream sm100.hpp) ──────────────────────

struct SM100Arch {
    static constexpr int smem_capacity = 232448;

    static int get_ab_load_block_m(int mc, bool mc_on_a, int block_m) {
        return block_m / (mc_on_a ? mc : 1);
    }
    static int get_ab_load_block_n(int mc, bool mc_on_a, int block_n) {
        return block_n / (mc_on_a ? 1 : mc);
    }

    static bool enable_cd_swizzle(int /*cd_elem_size*/) { return true; }

    static int get_smem_cd_size(KernelKind /*kt*/, int block_m, int block_n,
                                 int sw_cd, int /*cd_elem_size*/) {
        return std::min(block_m, 128) * sw_cd * 2;
    }

    static void get_sf_smem_per_stage(KernelKind kt, int block_m, int block_n,
                                       MmaKindLocal mma, int& sfa, int& sfb) {
        if (mma == MmaKindLocal::BF16) { sfa = 0; sfb = 0; return; }
        if (kt == KernelKind::Kernel1D1D) {
            sfa = align_up(block_m, 128) * 4;
            sfb = align_up(block_n, 128) * 4;
        } else {
            sfa = block_m * 4;
            sfb = 0;
        }
    }

    static int get_extra_sfb(int, int, int) { return 0; }

    static int get_barrier_smem_size(int num_stages) {
        return num_stages * 8 * 3 + 2 * 8 * 2 + 8;
    }
    static int get_tmem_ptr_smem_size() { return 4; }
    static int get_tensormap_smem_size(bool) { return 0; }

    static bool is_num_stages_legal(MmaKindLocal, int, int, int) { return true; }

    static int get_cd_store_block_m(int block_m) {
        return std::min(block_m, 128);
    }
};

// ── get_smem_config (exact upstream common.hpp) ────────────────────
// Computes total shared memory size and swizzle modes for a given config.
// Template on ArchSpec (SM90Arch or SM100Arch).

struct SmemConfig {
    int smem_size;
    int swizzle_a, swizzle_b, swizzle_cd;
};

template <typename Arch>
static SmemConfig compute_smem_config(
    KernelKind kernel_kind, MmaKindLocal mma_kind,
    int block_m, int block_n, int block_k,
    int num_stages, int num_multicast, bool mc_on_a,
    int ab_elem_size, int cd_elem_size,
    int m, int n, int k, bool k_grouped = false
) {
    int load_bm = block_m, load_bn = block_n;
    // SM100 adjusts load blocks for multicast
    if constexpr (std::is_same_v<Arch, SM100Arch>) {
        load_bm = Arch::get_ab_load_block_m(num_multicast, mc_on_a, block_m);
        load_bn = Arch::get_ab_load_block_n(num_multicast, mc_on_a, block_n);
    }

    // Swizzle modes (A/B are K-major for our NT layout)
    int sw_a = get_swizzle_mode(block_k, ab_elem_size);
    int sw_b = get_swizzle_mode(block_k, ab_elem_size);
    int sw_cd = Arch::enable_cd_swizzle(cd_elem_size) ? get_swizzle_mode(block_n, cd_elem_size) : 0;

    // Shared memory components
    int smem_cd = Arch::get_smem_cd_size(kernel_kind, block_m, block_n, sw_cd, cd_elem_size);
    int smem_a_per = load_bm * block_k * ab_elem_size;
    int smem_b_per = load_bn * block_k * ab_elem_size;
    int smem_sfa_per = 0, smem_sfb_per = 0;
    Arch::get_sf_smem_per_stage(kernel_kind, block_m, block_n, mma_kind, smem_sfa_per, smem_sfb_per);

    int smem_extra_sfb = 0;
    // SM90 1D2D has extra SFB buffer
    if constexpr (std::is_same_v<Arch, SM90Arch>) {
        if (kernel_kind == KernelKind::Kernel1D2D)
            smem_extra_sfb = Arch::get_extra_sfb(k, block_n, block_k);
    }

    int smem_barrier = Arch::get_barrier_smem_size(num_stages);
    int smem_tmem = Arch::get_tmem_ptr_smem_size();
    int smem_tmap = Arch::get_tensormap_smem_size(k_grouped);

    int total = smem_tmap + smem_cd
              + num_stages * (smem_a_per + smem_b_per + smem_sfa_per + smem_sfb_per)
              + smem_extra_sfb + smem_barrier + smem_tmem;

    return SmemConfig{ .smem_size = total, .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_cd = sw_cd };
}

// ── select_num_stages (exact upstream logic) ───────────────────────
// Finds the largest num_stages that fits in smem_capacity.

template <typename Arch>
static int select_num_stages(
    KernelKind kernel_kind, MmaKindLocal mma_kind,
    int block_m, int block_n, int block_k,
    int num_multicast, bool mc_on_a,
    int ab_elem_size, int cd_elem_size,
    int m, int n, int k, SmemConfig& out_cfg, bool k_grouped = false
) {
    for (int s = 32; s > 0; s--) {
        if (!Arch::is_num_stages_legal(mma_kind, block_n, block_k, s))
            continue;
        auto cfg = compute_smem_config<Arch>(kernel_kind, mma_kind,
            block_m, block_n, block_k, s, num_multicast, mc_on_a,
            ab_elem_size, cd_elem_size, m, n, k, k_grouped);
        if (cfg.smem_size <= Arch::smem_capacity) {
            out_cfg = cfg;
            return s;
        }
    }
    return 0;
}

// ── Config structs ─────────────────────────────────────────────────

struct KernelConfig {
    int block_m, block_n, block_k;
    int num_stages;
    int num_tma_threads, num_math_threads;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

struct FP8Config {
    int block_m, block_n, block_k;
    int num_stages, num_last_stages;
    int num_tma_threads, num_math_threads;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

struct SM100Config {
    int block_m, block_n, block_k;
    int num_stages;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b, swizzle_d;
    int smem_size;
};

// ── Legacy get_swizzle (for BF16 output, 2-byte elements) ──────────

static int get_swizzle(int block_size) {
    return get_swizzle_mode(block_size, 2);
}

// ── FP8 1D2D block_n validity ──────────────────────────────────────

static bool fp8_valid_block_n(int bn) {
    if (bn <= 128) return true;
    return gcd_static(bn, 128) == bn - 128;
}

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
    const int elem_size = 2;
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
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

static CUtensorMap make_2d_tma_u8(void* data, int inner_dim, int outer_dim,
                                   int smem_inner, int smem_outer,
                                   int outer_stride, int swizzle_mode) {
    ensure_driver_api();
    const int elem_size = 1;
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
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

static CUtensorMap make_2d_tma_f32(void* data, int inner_dim, int outer_dim,
                                    int smem_inner, int smem_outer,
                                    int outer_stride, int swizzle_mode) {
    ensure_driver_api();
    const int elem_size = 4;
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
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

// ── GPU state ───────────────────────────────────────────────────────

static int g_num_sms = 0;
static void ensure_num_sms() {
    if (g_num_sms > 0) return;
    int dev; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
}

static int g_gpu_arch = 0;
static void ensure_gpu_arch() {
    if (g_gpu_arch > 0) return;
    int dev; cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    g_gpu_arch = major * 10 + minor;
}

// ── Common launch helper ────────────────────────────────────────────

static int launch_kernel(const void* kernel_ptr, int num_threads, int smem_size,
                         int num_multicast, void** args, cudaStream_t stream) {
    dim3 grid(g_num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);
    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    if (num_multicast > 1) {
        cudaLaunchConfig_t lc = {};
        lc.gridDim = grid; lc.blockDim = block;
        lc.dynamicSmemBytes = smem_size; lc.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {(unsigned)num_multicast, 1, 1};
        lc.attrs = attrs; lc.numAttrs = 1;
        cudaError_t e = cudaLaunchKernelExC(&lc, kernel_ptr, args);
        if (e != cudaSuccess) {
            fprintf(stderr, "DeepGEMM cluster launch failed: %s (cluster=%d, threads=%d, smem=%d)\n",
                    cudaGetErrorString(e), num_multicast, num_threads, smem_size);
            cudaGetLastError(); return -2;
        }
    } else {
        cudaError_t e = cudaLaunchKernel(kernel_ptr, grid, block, args, smem_size, stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "DeepGEMM launch failed: %s (threads=%d, smem=%d)\n",
                    cudaGetErrorString(e), num_threads, smem_size);
            cudaGetLastError(); return -2;
        }
    }
    return 0;
}
