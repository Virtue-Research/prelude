/**
 *  @brief  Warp-reduction GEMV kernel for MoE decode (small-M path).
 *
 *  Tensor-core kernels (WMMA/WGMMA) are throughput-optimal for M >= 32
 *  but waste compute for decode (M == 1 or a handful per expert). This
 *  kernel dedicates one CUDA block to ONE output element of ONE token,
 *  uses float4 vectorized loads, and reduces via __shfl_xor_sync. Ported
 *  from mistral.rs / llama.cpp's batched GEMV approach.
 *
 *  Dispatch sits alongside moe_gemm_wmma: the Rust side picks this path
 *  when `!is_prefill && size_m <= 8`.
 */

#include "moe_utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace vllm_rs {

template <int WARP_SIZE = 32>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return x;
}

/**
 *  @brief  MoE GEMV kernel: computes `output[token][row] = Σ_k input[token][k] * W[expert][row][k]`.
 *
 *  Grid:  (N, size_m)   — one block per (output-row, token) pair.
 *  Block: BLOCK_SIZE threads, all cooperating on a single dot product.
 *
 *  For BF16 we multiply element-wise in float (fast enough at decode-M;
 *  avoids __hmul2 issues on some older CUDA BF16 paths). For FP16 we use
 *  native half2 fused multiply.
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_kernel(
    const T *__restrict__ input,                   // [size_m_eff, size_k]
    const T *__restrict__ weights,                 // [num_experts, size_n, size_k]
    const int32_t *__restrict__ sorted_token_ids,  // [size_m]
    const int32_t *__restrict__ expert_ids,        // [size_m]
    const float *__restrict__ topk_weights,        // [size_m] (optional, nullptr ok)
    T *__restrict__ output,                        // [size_m, size_n]
    const int num_experts, const int topk, const int size_m, const int size_n,
    const int size_k)
{
  const int row = blockIdx.x;        // output column (0..size_n-1)
  const int token_idx = blockIdx.y;  // sorted-token slot (0..size_m-1)

  if (token_idx >= size_m || row >= size_n) return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts) return;

  // If topk_weights is provided, tokens are NOT replicated (one entry per
  // token); otherwise the WMMA callers replicate tokens topk times, so
  // divide to recover the original token's input row.
  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row  = input   + (size_t)input_idx * size_k;
  const T *weight_row = weights + (size_t)expert * size_n * size_k
                                + (size_t)row * size_k;

  const int tid = threadIdx.x;

  // float4 = 128-bit = 8 half/bf16 elements per load.
  constexpr int LOAD_VEC_SIZE = 8;
  const int k_vec = size_k / LOAD_VEC_SIZE;

  const float4 *in_vec = reinterpret_cast<const float4 *>(input_row);
  const float4 *w_vec  = reinterpret_cast<const float4 *>(weight_row);

  using Vec2T = typename std::conditional<std::is_same<T, half>::value,
                                          half2, nv_bfloat162>::type;

  float sum = 0.0f;
  for (int k = tid; k < k_vec; k += BLOCK_SIZE) {
    float4 in_val = in_vec[k];
    float4 w_val  = w_vec[k];
    const Vec2T *in_v2 = reinterpret_cast<const Vec2T *>(&in_val);
    const Vec2T *w_v2  = reinterpret_cast<const Vec2T *>(&w_val);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (std::is_same<T, half>::value) {
        half2 prod = __hmul2(in_v2[i], w_v2[i]);
        sum += __low2float(prod) + __high2float(prod);
      } else {
        sum += vllm_rs::to_float(in_v2[i].x) * vllm_rs::to_float(w_v2[i].x);
        sum += vllm_rs::to_float(in_v2[i].y) * vllm_rs::to_float(w_v2[i].y);
      }
    }
  }

  // Scalar remainder for non-8-multiple K.
  const int remainder_start = k_vec * LOAD_VEC_SIZE;
  for (int k = remainder_start + tid; k < size_k; k += BLOCK_SIZE) {
    sum = __fmaf_rn(vllm_rs::to_float(input_row[k]),
                    vllm_rs::to_float(weight_row[k]), sum);
  }

  // Warp reduction, then inter-warp via shared memory.
  sum = vllm_rs::warp_reduce_sum(sum);

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) smem[warp_id] = sum;
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (lane_id == 0) {
      if (topk_weights) {
        sum *= topk_weights[token_id];
      }
      T out_val;
      vllm_rs::from_float(out_val, sum);
      // token_id addresses output rows directly: different sorted slots
      // with the same token_id (topk > 1) with topk_weights=nullptr would
      // collide, but the Rust dispatcher either passes topk_weights (one
      // output row per (token,expert) at sorted_slot granularity and
      // caller does weighted sum) or, when null, the output row is the
      // topk-replicated layout with one slot per pair — see WMMA callers.
      output[(size_t)token_id * size_n + row] = out_val;
    }
  }
}

} // namespace vllm_rs

extern "C" void moe_gemv(
    const void *input,
    const void *weights,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,   // device ptr or nullptr
    void *output,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int dtype,                   // 0 = fp16, 1 = bf16
    cudaStream_t stream)
{
  constexpr int BLOCK_SIZE = 256;
  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) {
    vllm_rs::moe_gemv_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights),
        sorted_token_ids, expert_ids, topk_weights,
        reinterpret_cast<half *>(output),
        num_experts, topk, size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) {
    vllm_rs::moe_gemv_kernel<nv_bfloat16, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        reinterpret_cast<const nv_bfloat16 *>(weights),
        sorted_token_ids, expert_ids, topk_weights,
        reinterpret_cast<nv_bfloat16 *>(output),
        num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv: unsupported dtype %d\n", dtype);
  }
}
