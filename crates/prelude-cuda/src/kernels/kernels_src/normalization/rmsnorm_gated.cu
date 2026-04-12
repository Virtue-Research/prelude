// Fused RMSNorm + SiLU-gated kernel:
//   output = RMSNorm(x, weight, eps) * SiLU(gate)
//
// Replaces 7 decomposed kernels (cast + sqr + sum + sqrt + div + mul + silu_mul)
// with a single fused kernel. Used by Qwen3.5's GatedDeltaNet (per-head norm).
//
// Supports F32 weight (Qwen3.5 stores norm weight as F32 for precision).
// Input x and gate are BF16, weight is F32, output is BF16.
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void rmsnorm_gated_bf16(
    const __nv_bfloat16* __restrict__ x,       // [N, D]
    const __nv_bfloat16* __restrict__ gate,    // [N, D]
    const float* __restrict__ weight,           // [D] (F32)
    __nv_bfloat16* __restrict__ output,        // [N, D]
    uint32_t n_rows,
    uint32_t d,
    float eps
) {
    // ── Small D path: 1 warp per row (D <= 256) ──
    if (d <= 256) {
        const uint32_t warp_id = threadIdx.x / 32;
        const uint32_t lane_id = threadIdx.x % 32;
        const uint32_t rows_per_block = blockDim.x / 32;
        const uint32_t row = blockIdx.x * rows_per_block + warp_id;
        if (row >= n_rows) return;

        const __nv_bfloat16* x_row = x + (uint64_t)row * d;
        const __nv_bfloat16* g_row = gate + (uint64_t)row * d;
        __nv_bfloat16* out_row = output + (uint64_t)row * d;

        const uint32_t elems_per_lane = d / 32;
        float x_vals[8];
        float ss = 0.0f;

        #pragma unroll
        for (uint32_t e = 0; e < elems_per_lane; e++) {
            float v = __bfloat162float(x_row[lane_id * elems_per_lane + e]);
            x_vals[e] = v;
            ss += v * v;
        }

        ss = warp_reduce_sum(ss);
        float scale = rsqrtf(ss / (float)d + eps);

        #pragma unroll
        for (uint32_t e = 0; e < elems_per_lane; e++) {
            uint32_t idx = lane_id * elems_per_lane + e;
            float w = weight[idx];
            float g = __bfloat162float(g_row[idx]);
            float silu_g = g / (1.0f + expf(-g));  // SiLU(g) = g * sigmoid(g)
            out_row[idx] = __float2bfloat16(x_vals[e] * scale * w * silu_g);
        }
        return;
    }

    // ── Large D path: 1 block per row ──
    extern __shared__ float smem[];

    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    const __nv_bfloat16* x_row = x + (uint64_t)row * d;
    const __nv_bfloat16* g_row = gate + (uint64_t)row * d;
    __nv_bfloat16* out_row = output + (uint64_t)row * d;

    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    // Pass 1: compute sum of squares
    float ss = 0.0f;
    for (uint32_t i = tid; i < d; i += block_size) {
        float v = __bfloat162float(x_row[i]);
        ss += v * v;
    }

    ss = block_reduce_sum(ss, smem);

    __shared__ float rms_scale;
    if (tid == 0) {
        rms_scale = rsqrtf(ss / (float)d + eps);
    }
    __syncthreads();
    float scale = rms_scale;

    // Pass 2: normalize × weight × silu(gate)
    for (uint32_t i = tid; i < d; i += block_size) {
        float v = __bfloat162float(x_row[i]);
        float w = weight[i];
        float g = __bfloat162float(g_row[i]);
        float silu_g = g / (1.0f + expf(-g));
        out_row[i] = __float2bfloat16(v * scale * w * silu_g);
    }
}
