// Common types, TMA helpers, GPU state, and launch helper.
// Shared by all sm90/sm100 .cuh files.

#pragma once

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

// ── Utility functions ───────────────────────────────────────────────

static int get_swizzle(int block_size) {
    for (int mode : {128, 64, 32, 16})
        if ((block_size * 2) % mode == 0) return mode;
    return 16;
}

static bool fp8_valid_block_n(int bn) {
    // 1D2D constraint: ceil_div(bn,128)==1 or gcd(bn,128)==bn-128
    if (bn <= 128) return true;
    auto gcd = [](int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; };
    return gcd(bn, 128) == bn - 128;
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
        lc.gridDim = grid;
        lc.blockDim = block;
        lc.dynamicSmemBytes = smem_size;
        lc.stream = stream;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {(unsigned)num_multicast, 1, 1};
        lc.attrs = attrs;
        lc.numAttrs = 1;

        if (cudaLaunchKernelExC(&lc, kernel_ptr, args) != cudaSuccess) {
            cudaGetLastError();
            return -2;
        }
    } else {
        if (cudaLaunchKernel(kernel_ptr, grid, block, args, smem_size, stream) != cudaSuccess) {
            cudaGetLastError();
            return -2;
        }
    }
    return 0;
}
