// Naive GEMM kernels — isolated from CUTLASS templates.
//
// One thread per output element, F32 accumulation.
// Used as fallback for small matrices (M<64 or N<64) where CUTLASS
// tile-based kernels fail due to TMA/alignment constraints.
//
// Layout: A=RowMajor, B=ColumnMajor, D=ColumnMajor (TN convention).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// ── BF16 ──────────────────────────────────────────────────────────────

__global__ void naive_bf16_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ D,
    int M, int N, int K)
{
    // D[i,j] = sum_k A[i,k] * B[k,j]
    // A is RM [M,K]: A[i,k] = A[i*K+k]
    // B is CM [K,N]: B[k,j] = B[j*K+k]
    // D is CM [M,N]: D[i,j] = D[j*M+i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx % M;
    int j = idx / M;
    float sum = 0.0f;
    for (int t = 0; t < K; t++) {
        sum += __bfloat162float(A[i * K + t]) * __bfloat162float(B[j * K + t]);
    }
    D[j * M + i] = __float2bfloat16(sum);
}

extern "C" int naive_bf16_gemm(const void* A, const void* B, void* D,
                                int m, int n, int k, cudaStream_t s) {
    cudaGetLastError();  // clear any prior error
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    naive_bf16_gemm_kernel<<<grid, block, 0, s>>>(
        (const __nv_bfloat16*)A, (const __nv_bfloat16*)B, (__nv_bfloat16*)D, m, n, k);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "naive_bf16_gemm FAILED: %s\n", cudaGetErrorString(err));
        return -31;
    }
    return 0;
}

// ── FP16 ──────────────────────────────────────────────────────────────

__global__ void naive_fp16_gemm_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ D,
    int M, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx % M;
    int j = idx / M;
    float sum = 0.0f;
    for (int t = 0; t < K; t++) {
        sum += __half2float(A[i * K + t]) * __half2float(B[j * K + t]);
    }
    D[j * M + i] = __float2half(sum);
}

extern "C" int naive_fp16_gemm(const void* A, const void* B, void* D,
                                int m, int n, int k, cudaStream_t s) {
    cudaGetLastError();
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    naive_fp16_gemm_kernel<<<grid, block, 0, s>>>(
        (const __half*)A, (const __half*)B, (__half*)D, m, n, k);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "naive_fp16_gemm FAILED: %s\n", cudaGetErrorString(err));
        return -30;
    }
    return 0;
}

// ── F32 ───────────────────────────────────────────────────────────────

__global__ void naive_f32_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ D,
    int M, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx % M;
    int j = idx / M;
    float sum = 0.0f;
    for (int t = 0; t < K; t++) {
        sum += A[i * K + t] * B[j * K + t];
    }
    D[j * M + i] = sum;
}

extern "C" int naive_f32_gemm(const void* A, const void* B, void* D,
                               int m, int n, int k, cudaStream_t s) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    naive_f32_gemm_kernel<<<grid, block, 0, s>>>(
        (const float*)A, (const float*)B, (float*)D, m, n, k);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -30;
}
