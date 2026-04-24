// Fused SiLU(gate) * up kernel
// Replaces: silu_activation(gate) then element-wise multiply with up
// Saves one full memory round-trip over the intermediate_size tensor
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void fused_silu_mul_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    uint32_t n
) {
    const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (idx + 8 <= n) {
        Vec8BF16 gate_v, up_v;
        gate_v.load(gate + idx);
        up_v.load(up + idx);
        const __nv_bfloat162* gp = gate_v.as_bf162();
        const __nv_bfloat162* up_p = up_v.as_bf162();

        Vec8BF16 result;
        __nv_bfloat162* op = result.as_bf162_mut();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 g = __bfloat1622float2(gp[i]);
            float2 u = __bfloat1622float2(up_p[i]);
            float2 r;
            r.x = silu(g.x) * u.x;
            r.y = silu(g.y) * u.y;
            op[i] = __float22bfloat162_rn(r);
        }

        result.store(out + idx);
    } else if (idx < n) {
        for (uint32_t i = idx; i < n; i++) {
            float g = __bfloat162float(gate[i]);
            float u = __bfloat162float(up[i]);
            out[i] = __float2bfloat16(silu(g) * u);
        }
    }
}
