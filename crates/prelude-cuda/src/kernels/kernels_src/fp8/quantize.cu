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
