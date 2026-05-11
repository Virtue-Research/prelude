#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ float clamp_fp8_e4m3(float x) {
    return fminf(fmaxf(x, -448.0f), 448.0f);
}

extern "C" __global__ void static_scaled_bf16_to_fp8_e4m3(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float inv_scale,
    uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < n; i += stride) {
        float v = __bfloat162float(input[i]) * inv_scale;
        output[i] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void static_scaled_f16_to_fp8_e4m3(
    const __half* input,
    __nv_fp8_e4m3* output,
    float inv_scale,
    uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < n; i += stride) {
        float v = __half2float(input[i]) * inv_scale;
        output[i] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void static_scaled_f32_to_fp8_e4m3(
    const float* input,
    __nv_fp8_e4m3* output,
    float inv_scale,
    uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < n; i += stride) {
        float v = input[i] * inv_scale;
        output[i] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void static_scaled_bf16_to_fp8_e4m3_padded(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float inv_scale,
    uint32_t m,
    uint32_t k,
    uint32_t m_pad) {
    uint64_t total = (uint64_t)m_pad * (uint64_t)k;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = idx; i < total; i += stride) {
        uint32_t row = (uint32_t)(i / k);
        uint32_t col = (uint32_t)(i - (uint64_t)row * k);
        float v = row < m ? __bfloat162float(input[(uint64_t)row * k + col]) * inv_scale : 0.0f;
        output[i] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void static_scaled_f16_to_fp8_e4m3_padded(
    const __half* input,
    __nv_fp8_e4m3* output,
    float inv_scale,
    uint32_t m,
    uint32_t k,
    uint32_t m_pad) {
    uint64_t total = (uint64_t)m_pad * (uint64_t)k;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = idx; i < total; i += stride) {
        uint32_t row = (uint32_t)(i / k);
        uint32_t col = (uint32_t)(i - (uint64_t)row * k);
        float v = row < m ? __half2float(input[(uint64_t)row * k + col]) * inv_scale : 0.0f;
        output[i] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void static_scaled_f32_to_fp8_e4m3_padded(
    const float* input,
    __nv_fp8_e4m3* output,
    float inv_scale,
    uint32_t m,
    uint32_t k,
    uint32_t m_pad) {
    uint64_t total = (uint64_t)m_pad * (uint64_t)k;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = idx; i < total; i += stride) {
        uint32_t row = (uint32_t)(i / k);
        uint32_t col = (uint32_t)(i - (uint64_t)row * k);
        float v = row < m ? input[(uint64_t)row * k + col] * inv_scale : 0.0f;
        output[i] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void moe_fp8_compute_padded_offsets(
    const int32_t* __restrict__ real_offsets,
    int32_t* __restrict__ padded_offsets,
    int num_experts,
    int align) {
    if (threadIdx.x != 0) return;
    int padded = 0;
    padded_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        int count = real_offsets[e + 1] - real_offsets[e];
        int aligned = ((count + align - 1) / align) * align;
        padded += aligned;
        padded_offsets[e + 1] = padded;
    }
}

extern "C" __global__ void moe_fp8_compute_padded_layout(
    const int32_t* __restrict__ real_offsets,
    int32_t* __restrict__ padded_offsets,
    int32_t* __restrict__ grouped_layout,
    int num_experts,
    int align) {
    if (threadIdx.x != 0) return;
    int padded = 0;
    padded_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        int count = real_offsets[e + 1] - real_offsets[e];
        int aligned = ((count + align - 1) / align) * align;
        int begin = padded;
        padded += aligned;
        padded_offsets[e + 1] = padded;
        for (int row = begin; row < padded; row++) {
            grouped_layout[row] = e;
        }
    }
}

extern "C" __global__ void moe_fp8_fill_padded_scales(
    const float* __restrict__ input_scales,
    const int32_t* __restrict__ padded_offsets,
    float* __restrict__ scale_a,
    int num_experts,
    int k_groups,
    int padded_total) {
    int e = blockIdx.x;
    int kg = blockIdx.y;
    if (e >= num_experts || kg >= k_groups) return;
    float scale = input_scales[e];
    int begin = padded_offsets[e];
    int end = padded_offsets[e + 1];
    for (int row = begin + threadIdx.x; row < end; row += blockDim.x) {
        scale_a[(size_t)row * k_groups + kg] = scale;
    }
}

extern "C" __global__ void moe_fp8_gather_quantize_bf16_padded(
    const __nv_bfloat16* __restrict__ input,
    const uint32_t* __restrict__ sorted_token_ids,
    const uint32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ real_offsets,
    const int32_t* __restrict__ padded_offsets,
    const float* __restrict__ input_scales,
    __nv_fp8_e4m3* __restrict__ gathered,
    int n_real,
    int k,
    int token_divisor) {
    int i = blockIdx.x;
    if (i >= n_real) return;
    int e = (int)sorted_expert_ids[i];
    int intra = i - real_offsets[e];
    int row = padded_offsets[e] + intra;
    int input_row = (int)sorted_token_ids[i] / token_divisor;
    float inv_scale = 1.0f / input_scales[e];

    const __nv_bfloat16* src = input + (size_t)input_row * k;
    __nv_fp8_e4m3* dst = gathered + (size_t)row * k;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        float v = __bfloat162float(src[col]) * inv_scale;
        dst[col] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void moe_fp8_build_assignment_rows(
    const uint32_t* __restrict__ sorted_token_ids,
    const uint32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ real_offsets,
    const int32_t* __restrict__ padded_offsets,
    int32_t* __restrict__ assignment_rows,
    int n_real) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_real) return;
    int e = (int)sorted_expert_ids[i];
    int intra = i - real_offsets[e];
    int padded_row = padded_offsets[e] + intra;
    assignment_rows[sorted_token_ids[i]] = padded_row;
}

extern "C" __global__ void moe_fp8_silu_mul_quantize_bf16_padded(
    const __nv_bfloat16* __restrict__ gate_up,
    const int32_t* __restrict__ grouped_layout,
    const float* __restrict__ input_scales,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale_a,
    int inter,
    int k_groups,
    int padded_total) {
    int row = blockIdx.x;
    if (row >= padded_total) return;
    int e = grouped_layout[row];
    if (e < 0) return;

    float scale = input_scales[e];
    float inv_scale = 1.0f / scale;
    if (threadIdx.x < (unsigned)k_groups) {
        scale_a[(size_t)threadIdx.x * padded_total + row] = scale;
    }

    const __nv_bfloat16* src = gate_up + (size_t)row * inter * 2;
    __nv_fp8_e4m3* dst = output + (size_t)row * inter;
    for (int col = threadIdx.x; col < inter; col += blockDim.x) {
        float up = __bfloat162float(src[col]);
        float gate = __bfloat162float(src[inter + col]);
        float silu = gate / (1.0f + expf(-gate));
        float v = silu * up * inv_scale;
        dst[col] = __nv_fp8_e4m3(clamp_fp8_e4m3(v));
    }
}

extern "C" __global__ void moe_fp8_weighted_sum_padded_bf16(
    const __nv_bfloat16* __restrict__ down_out,
    const int32_t* __restrict__ assignment_rows,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output,
    int n_tokens,
    int topk,
    int hidden) {
    int token = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= n_tokens || col >= hidden) return;

    float acc = 0.0f;
    int base = token * topk;
    for (int slot = 0; slot < topk; slot++) {
        int flat = base + slot;
        int row = assignment_rows[flat];
        float w = topk_weights[flat];
        float v = __bfloat162float(down_out[(size_t)row * hidden + col]);
        acc += v * w;
    }
    output[(size_t)token * hidden + col] = __float2bfloat16(acc);
}
