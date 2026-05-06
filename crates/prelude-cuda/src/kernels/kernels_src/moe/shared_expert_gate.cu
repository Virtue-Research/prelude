#include "../common/common.cuh"

extern "C" __global__ void shared_expert_gate_bf16(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ gate_weight,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int hidden
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const __nv_bfloat16* row_input = input + row * hidden;
    const __nv_bfloat16* row_shared = shared_out + row * hidden;
    __nv_bfloat16* row_output = output + row * hidden;

    float sum = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
        sum += __bfloat162float(row_input[col]) * __bfloat162float(gate_weight[col]);
    }

    extern __shared__ float smem[];
    const float dot = block_reduce_sum(sum, smem);
    __shared__ float scale_smem;
    if (tid == 0) {
        scale_smem = 1.0f / (1.0f + expf(-dot));
    }
    __syncthreads();
    const float scale = scale_smem;

    for (int col = tid; col < hidden; col += blockDim.x) {
        const float v = __bfloat162float(row_shared[col]) * scale;
        row_output[col] = __float2bfloat16(v);
    }
}
