// Fused residual add + RMSNorm
// Computes: sum = x + residual; normed = rmsnorm(sum, weight, eps)
// Returns both sum (for next residual) and normed (for next sublayer)
// Saves one full read+write cycle by combining add and norm
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void fused_add_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x,         // [N, D]
    const __nv_bfloat16* __restrict__ residual,  // [N, D]
    const __nv_bfloat16* __restrict__ weight,    // [D]
    __nv_bfloat16* __restrict__ out_sum,         // [N, D] = x + residual
    __nv_bfloat16* __restrict__ out_norm,        // [N, D] = rmsnorm(out_sum)
    uint32_t n_rows,
    uint32_t d,
    float eps
) {
    extern __shared__ float smem[];

    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    const __nv_bfloat16* x_row = x + row * d;
    const __nv_bfloat16* r_row = residual + row * d;
    __nv_bfloat16* sum_row = out_sum + row * d;
    __nv_bfloat16* norm_row = out_norm + row * d;

    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    // Pass 1: compute sum = x + residual, accumulate sum-of-squares
    float ss = 0.0f;
    for (uint32_t i = tid; i < d; i += block_size) {
        float xf = __bfloat162float(x_row[i]);
        float rf = __bfloat162float(r_row[i]);
        float s = xf + rf;
        sum_row[i] = __float2bfloat16(s);
        ss += s * s;
    }

    // Block-level reduction
    ss = block_reduce_sum(ss, smem);

    __shared__ float rms_scale;
    if (tid == 0) {
        rms_scale = rsqrtf(ss / (float)d + eps);
    }
    __syncthreads();

    float scale = rms_scale;

    // Pass 2: normalize and write output
    for (uint32_t i = tid; i < d; i += block_size) {
        float s = __bfloat162float(sum_row[i]);
        float w = __bfloat162float(weight[i]);
        norm_row[i] = __float2bfloat16(s * scale * w);
    }
}
