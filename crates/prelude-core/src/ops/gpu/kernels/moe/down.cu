// Fused MoE Decode: Down projection with weighted reduction
// For single-token decode (M=1), replaces down moe_gemm + reshape + sum
// with a single kernel. Each warp computes one output position by iterating
// over all active experts and accumulating the weighted dot products.
//
// Grid:  (ceil(hidden_size/4), 1)
// Block: 128 threads (4 warps), each warp handles 1 output position
//
// No atomicAdd needed - each warp handles the full expert reduction.
// Relies on L2 cache for the intermediate tensor (12KB, stays hot).
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void moe_decode_down_reduce_bf16(
    const __nv_bfloat16* __restrict__ intermediate,  // [num_active, inter_size]
    const __nv_bfloat16* __restrict__ down_w,        // [num_experts, hidden_size, inter_size]
    const uint32_t* __restrict__ expert_ids,          // [num_active]
    const float* __restrict__ topk_weights,           // [num_active]
    __nv_bfloat16* __restrict__ output,               // [hidden_size]
    uint32_t hidden_size,
    uint32_t inter_size,
    uint32_t num_active
) {
    const uint32_t n_base = blockIdx.x * 4;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t n = n_base + warp_id;

    if (n >= hidden_size) return;

    float accum = 0.0f;

    // Iterate over all active experts, accumulate weighted dot products
    for (uint32_t e = 0; e < num_active; e++) {
        const uint32_t expert_id = expert_ids[e];
        const float weight = topk_weights[e];

        const __nv_bfloat16* inter_row = intermediate + (uint64_t)e * inter_size;
        const __nv_bfloat16* w_row = down_w + (uint64_t)expert_id * hidden_size * inter_size
                                     + (uint64_t)n * inter_size;

        float dot = 0.0f;

        // Coalesced dot product over inter_size dimension
        for (uint32_t k = lane_id * 8; k < inter_size; k += 256) {
            Vec8BF16 i_v, w_v;
            i_v.load(&inter_row[k]);
            w_v.load(&w_row[k]);

            const __nv_bfloat162* ip = i_v.as_bf162();
            const __nv_bfloat162* wp = w_v.as_bf162();

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 iv = __bfloat1622float2(ip[j]);
                float2 wv = __bfloat1622float2(wp[j]);
                dot += iv.x * wv.x + iv.y * wv.y;
            }
        }

        // Warp reduction
        dot = warp_reduce_sum(dot);

        accum += weight * dot;
    }

    // Write final output
    if (lane_id == 0) {
        output[n] = __float2bfloat16(accum);
    }
}
