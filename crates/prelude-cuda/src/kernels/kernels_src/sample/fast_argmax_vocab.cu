// Multi-block argmax over a contiguous [B, V] logits tensor.
//
// Candle's `fast_argmax_*` (in `candle/reduce.cu`) launches **one block
// per output row**: for greedy LM-head sampling at B≤32, V≈151,936 this
// uses 4-32 of the GPU's ~114 SMs and spends 25% of GPU time on what
// should be a memory-bound kernel that finishes in microseconds. The
// inner loop also pays for `get_strided_index` (a per-element int
// mod+div chain) even though the logits are always contiguous.
//
// This file is a two-pass argmax that
//   (1) fans out across `blocks_per_row` blocks per row so all SMs see
//       work even when B is tiny,
//   (2) reads the logits with plain linear addressing (callers must
//       supply contiguous [B, V]), and
//   (3) breaks ties by picking the smaller column index, matching
//       NumPy / PyTorch / candle semantics.
//
// Pass 1 — chunked local argmax:
//   grid = (B, blocks_per_row, 1), block = (BLOCK_SIZE, 1, 1)
//   Each block scans `chunk_size` contiguous logits and writes one
//   (max_val, max_idx) entry into `partials_*`.
//
// Pass 2 — partial reduce:
//   grid = (B, 1, 1), block = (32, 1, 1)
//   A single warp reduces the per-row partials to one global argmax
//   index. Requires `blocks_per_row <= 32` so it fits in one warp; the
//   Rust launcher enforces that.

#include "../common/common.cuh"
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

namespace {

// Warp-level "max + argmax" reduction.
// Picks larger value, ties broken by smaller index.
__device__ __forceinline__ void warp_argmax_reduce(float& val, uint32_t& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float    other_v = __shfl_xor_sync(0xffffffff, val, offset);
        uint32_t other_i = __shfl_xor_sync(0xffffffff, idx, offset);
        bool take = (other_v > val) || (other_v == val && other_i < idx);
        val = take ? other_v : val;
        idx = take ? other_i : idx;
    }
}

// Block-wide argmax: every thread arrives with its (val, idx). On exit,
// thread 0 of warp 0 holds the block-global argmax.
__device__ __forceinline__ void block_argmax_reduce(
    float& val, uint32_t& idx, float* smem_val, uint32_t* smem_idx
) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t num_warps = (blockDim.x + 31u) >> 5;

    warp_argmax_reduce(val, idx);
    if (lane == 0) {
        smem_val[warp] = val;
        smem_idx[warp] = idx;
    }
    __syncthreads();

    if (warp == 0) {
        val = (lane < num_warps) ? smem_val[lane] : -FLT_MAX;
        idx = (lane < num_warps) ? smem_idx[lane] : 0u;
        warp_argmax_reduce(val, idx);
    }
}

}  // namespace

// ─── Pass 1 — bf16 ───────────────────────────────────────────────────────

extern "C" __global__ void fast_argmax_vocab_pass1_bf16(
    const __nv_bfloat16* __restrict__ logits,
    float*    __restrict__ partials_val,
    uint32_t* __restrict__ partials_idx,
    uint32_t B,
    uint32_t V,
    uint32_t blocks_per_row,
    uint32_t chunk_size
) {
    const uint32_t row   = blockIdx.x;
    const uint32_t chunk = blockIdx.y;
    if (row >= B || chunk >= blocks_per_row) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t bs  = blockDim.x;

    const uint32_t start = chunk * chunk_size;
    const uint32_t stop  = min(start + chunk_size, V);
    const __nv_bfloat16* row_ptr = logits + (size_t)row * V;

    float    my_max = -FLT_MAX;
    uint32_t my_idx = 0;

    for (uint32_t i = start + tid; i < stop; i += bs) {
        float v = __bfloat162float(row_ptr[i]);
        // Strict `>` => on tie, earlier (smaller) idx wins because each
        // thread scans its lane in increasing order.
        if (v > my_max) {
            my_max = v;
            my_idx = i;
        }
    }

    __shared__ float    smem_val[32];
    __shared__ uint32_t smem_idx[32];
    block_argmax_reduce(my_max, my_idx, smem_val, smem_idx);

    if (tid == 0) {
        partials_val[(size_t)row * blocks_per_row + chunk] = my_max;
        partials_idx[(size_t)row * blocks_per_row + chunk] = my_idx;
    }
}

// ─── Pass 1 — f32 ────────────────────────────────────────────────────────

extern "C" __global__ void fast_argmax_vocab_pass1_f32(
    const float*    __restrict__ logits,
    float*    __restrict__ partials_val,
    uint32_t* __restrict__ partials_idx,
    uint32_t B,
    uint32_t V,
    uint32_t blocks_per_row,
    uint32_t chunk_size
) {
    const uint32_t row   = blockIdx.x;
    const uint32_t chunk = blockIdx.y;
    if (row >= B || chunk >= blocks_per_row) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t bs  = blockDim.x;

    const uint32_t start = chunk * chunk_size;
    const uint32_t stop  = min(start + chunk_size, V);
    const float* row_ptr = logits + (size_t)row * V;

    float    my_max = -FLT_MAX;
    uint32_t my_idx = 0;

    for (uint32_t i = start + tid; i < stop; i += bs) {
        float v = row_ptr[i];
        if (v > my_max) {
            my_max = v;
            my_idx = i;
        }
    }

    __shared__ float    smem_val[32];
    __shared__ uint32_t smem_idx[32];
    block_argmax_reduce(my_max, my_idx, smem_val, smem_idx);

    if (tid == 0) {
        partials_val[(size_t)row * blocks_per_row + chunk] = my_max;
        partials_idx[(size_t)row * blocks_per_row + chunk] = my_idx;
    }
}

// ─── Pass 2 — collapse partials → final index ────────────────────────────
//
// blockDim.x must be 32 (single warp) and blocks_per_row must be <= 32.

extern "C" __global__ void fast_argmax_vocab_pass2(
    const float*    __restrict__ partials_val,
    const uint32_t* __restrict__ partials_idx,
    uint32_t*       __restrict__ out_idx,
    uint32_t B,
    uint32_t blocks_per_row
) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= B) return;

    float    val = -FLT_MAX;
    uint32_t idx = 0;
    if (tid < blocks_per_row) {
        val = partials_val[(size_t)row * blocks_per_row + tid];
        idx = partials_idx[(size_t)row * blocks_per_row + tid];
    }
    warp_argmax_reduce(val, idx);

    if (tid == 0) {
        out_idx[row] = idx;
    }
}
