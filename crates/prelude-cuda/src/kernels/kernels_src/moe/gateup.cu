// Fused MoE Decode: Gate+Up projection with SiLU
// For single-token decode (M=1), replaces 3 kernel calls (gate moe_gemm +
// up moe_gemm + fused_silu_mul) with a single kernel.
//
// Each warp computes one output position for one expert:
//   result[n] = SiLU(dot(x, gate_w[expert][n])) * dot(x, up_w[expert][n])
//
// Grid:  (ceil(inter_size/4), num_active_experts)
// Block: 128 threads (4 warps), each warp handles 1 output position
// Shared: hidden_size * 2 bytes for input vector
//
// Takes separate gate_w and up_w pointers (no pre-concatenation needed).
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void moe_decode_gateup_silu_bf16(
    const __nv_bfloat16* __restrict__ input,       // [hidden_size]
    const __nv_bfloat16* __restrict__ gate_w,      // [num_experts, inter_size, hidden_size]
    const __nv_bfloat16* __restrict__ up_w,        // [num_experts, inter_size, hidden_size]
    const uint32_t* __restrict__ expert_ids,        // [num_active] expert indices
    __nv_bfloat16* __restrict__ output,             // [num_active, inter_size]
    uint32_t hidden_size,
    uint32_t inter_size,
    uint32_t num_active
) {
    const uint32_t expert_local = blockIdx.y;
    if (expert_local >= num_active) return;

    const uint32_t n_base = blockIdx.x * 4;  // 4 output positions per block
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t n = n_base + warp_id;

    if (n >= inter_size) return;

    const uint32_t expert_id = expert_ids[expert_local];

    // Cooperatively load input vector to shared memory
    extern __shared__ __nv_bfloat16 x_shared[];
    for (uint32_t i = threadIdx.x * 8; i < hidden_size; i += blockDim.x * 8) {
        *reinterpret_cast<float4*>(&x_shared[i]) =
            *reinterpret_cast<const float4*>(&input[i]);
    }
    __syncthreads();

    // Weight row pointers for this expert and output position
    const uint64_t expert_stride = (uint64_t)inter_size * hidden_size;
    const __nv_bfloat16* gate_row = gate_w + (uint64_t)expert_id * expert_stride
                                    + (uint64_t)n * hidden_size;
    const __nv_bfloat16* up_row   = up_w + (uint64_t)expert_id * expert_stride
                                    + (uint64_t)n * hidden_size;

    // Dot product with coalesced access
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (uint32_t k = lane_id * 8; k < hidden_size; k += 256) {
        Vec8BF16 x_v, g_v, u_v;
        x_v.load(&x_shared[k]);
        g_v.load(&gate_row[k]);
        u_v.load(&up_row[k]);

        const __nv_bfloat162* xp = x_v.as_bf162();
        const __nv_bfloat162* gp = g_v.as_bf162();
        const __nv_bfloat162* up_p = u_v.as_bf162();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 xf = __bfloat1622float2(xp[i]);
            float2 gf = __bfloat1622float2(gp[i]);
            float2 uf = __bfloat1622float2(up_p[i]);
            gate_sum += xf.x * gf.x + xf.y * gf.y;
            up_sum   += xf.x * uf.x + xf.y * uf.y;
        }
    }

    // Warp-level reduction
    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);

    // SiLU(gate) * up, write result
    if (lane_id == 0) {
        float silu_gate = silu(gate_sum);
        output[(uint64_t)expert_local * inter_size + n] =
            __float2bfloat16(silu_gate * up_sum);
    }
}
