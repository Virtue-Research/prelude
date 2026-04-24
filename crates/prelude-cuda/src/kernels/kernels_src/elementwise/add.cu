// Vectorized BF16 element-wise addition
// Processes 8 BF16 elements per thread via 128-bit loads
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void vectorized_add_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    uint32_t n
) {
    const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    vec8_binary(a, b, out, idx, n, [](float x, float y) { return x + y; });
}
