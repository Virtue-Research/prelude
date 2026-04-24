// Layout transformation utilities for scaling factors.
// Matches upstream DeepGEMM smxx_layout.cuh functionality.
//
// Provides:
//   - transform_sf_transpose: transpose FP32 SF from [G,MN,K/128] to MN-major TMA-aligned
//   - get_tma_aligned_size: alignment for TMA descriptors
//   - get_mk_alignment_for_contiguous_layout: returns 128

#pragma once

// NOTE: We do NOT include <deep_gemm/impls/smxx_layout.cuh> because its
// templated kernels have conflicting `extern __shared__` variable types
// (float vs uint32_t) which fail in a single compilation unit (AOT).
// Instead, we provide runtime-parameterized equivalents below.

// ── Alignment utilities ───────────────────────────────────────────

/// TMA requires 16-byte aligned addresses. Returns the smallest aligned
/// element count >= `size` for elements of `elem_size` bytes.
static int get_tma_aligned_size_local(int size, int elem_size) {
    int align_elems = 16 / elem_size;
    return ((size + align_elems - 1) / align_elems) * align_elems;
}

/// Contiguous grouped layout requires M/K aligned to 128.
static int get_mk_alignment_local() { return 128; }

// ── Runtime-parameterized SF transpose ────────────────────────────
// The upstream uses JIT with SF_K as a template param. For AOT, we
// write a simple runtime version that works for any SF_K. Since layout
// transformation is a one-time cost at model load, the minor perf
// difference is acceptable.

__global__ void transpose_sf_fp32_rt(
    const float* __restrict__ sf_in,   // [num_groups, mn, sf_k] row-major
    float* __restrict__ sf_out,        // [num_groups, sf_k, tma_aligned_mn] MN-major output
    int mn, int sf_k, int tma_aligned_mn
) {
    int group = blockIdx.y;
    int mn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn_idx >= mn) return;

    const float* in_ptr = sf_in + (int64_t)group * mn * sf_k + (int64_t)mn_idx * sf_k;
    float* out_ptr = sf_out + (int64_t)group * tma_aligned_mn * sf_k;

    for (int k = 0; k < sf_k; k++) {
        out_ptr[(int64_t)k * tma_aligned_mn + mn_idx] = in_ptr[k];
    }
}

/// Transpose FP32 scaling factors from [num_groups, mn, sf_k] (K-major)
/// to [num_groups, sf_k, tma_aligned_mn] (MN-major, TMA-aligned).
/// sf_out must be pre-allocated with the correct size.
/// Returns 0 on success, -2 on launch error.
static int transform_sf_transpose(
    void* sf_in, void* sf_out,
    int mn, int sf_k, int num_groups,
    void* stream
) {
    int tma_aligned_mn = get_tma_aligned_size_local(mn, sizeof(float));
    constexpr int BLOCK = 256;
    dim3 grid(ceil_div_static(mn, BLOCK), num_groups);
    dim3 block(BLOCK);

    transpose_sf_fp32_rt<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)sf_in, (float*)sf_out,
        mn, sf_k, tma_aligned_mn
    );
    if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); return -2; }
    return 0;
}

// ── UE8M0 packing (for SM100) ────────────────────────────────────
// Runtime-parameterized version: packs FP32 SFs into UE8M0 format
// and transposes to MN-major layout.

__global__ void transpose_pack_sf_ue8m0_rt(
    const float* __restrict__ sf_in,  // [num_groups, mn, sf_k] row-major FP32
    uint32_t* __restrict__ sf_out,    // [num_groups, packed_sf_k, tma_aligned_mn] MN-major UE8M0
    int mn, int sf_k, int packed_sf_k, int tma_aligned_mn
) {
    int group = blockIdx.y;
    int mn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn_idx >= mn) return;

    const float* in_ptr = sf_in + (int64_t)group * mn * sf_k + (int64_t)mn_idx * sf_k;
    uint32_t* out_ptr = sf_out + (int64_t)group * packed_sf_k * tma_aligned_mn;

    for (int pk = 0; pk < packed_sf_k; pk++) {
        uint32_t values[4];
        for (int j = 0; j < 4; j++) {
            int k_idx = pk * 4 + j;
            if (k_idx < sf_k) {
                uint32_t bits;
                memcpy(&bits, &in_ptr[k_idx], sizeof(uint32_t));
                values[j] = bits;
            } else {
                values[j] = 0;
            }
        }
        uint32_t packed = (values[0] >> 23u) | (values[1] >> 15u) |
                          (values[2] >> 7u)  | (values[3] << 1u);
        out_ptr[(int64_t)pk * tma_aligned_mn + mn_idx] = packed;
    }
}

/// Transpose and pack FP32 scaling factors into UE8M0 format.
/// Input: [num_groups, mn, sf_k] FP32 (K-major)
/// Output: [num_groups, ceil(sf_k/4), tma_aligned_mn] uint32 (MN-major)
/// sf_out must be pre-allocated. Returns 0 on success, -2 on launch error.
static int transform_sf_pack_ue8m0(
    void* sf_in, void* sf_out,
    int mn, int sf_k, int num_groups,
    void* stream
) {
    int packed_sf_k = ceil_div_static(sf_k, 4);
    int tma_aligned_mn = get_tma_aligned_size_local(mn, sizeof(uint32_t));
    constexpr int BLOCK = 256;
    dim3 grid(ceil_div_static(mn, BLOCK), num_groups);
    dim3 block(BLOCK);

    transpose_pack_sf_ue8m0_rt<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)sf_in, (uint32_t*)sf_out,
        mn, sf_k, packed_sf_k, tma_aligned_mn
    );
    if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); return -2; }
    return 0;
}
