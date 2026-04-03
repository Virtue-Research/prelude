// Common definitions for Prelude CUDA kernels
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

// ─── Warp-level reduction utilities ─────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ─── Block-level reduction via shared memory ────────────────────────────

// Requires: extern __shared__ float smem[];
// smem must have at least (blockDim.x / 32) floats
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t num_warps = (blockDim.x + 31) / 32;

    // Warp-level reduction first
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t num_warps = (blockDim.x + 31) / 32;

    val = warp_reduce_max(val);

    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
    }

    return val;
}

// ─── Activation functions ───────────────────────────────────────────────

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_approx(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608f;  // sqrt(2/pi)
    const float a = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + a * x3)));
}
