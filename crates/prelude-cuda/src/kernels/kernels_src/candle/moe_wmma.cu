/**
 *  @brief  WMMA-based grouped MoE GEMM kernel.
 *
 *  Each block computes a tile of the output corresponding to:
 *    - One expert segment (group of tokens routed to the same expert)
 *    - One N-dimension tile (a sub-block of the expert's output features)
 *
 *  The kernel loads input activations and expert weights in tiles using shared memory,
 *  performs matrix multiplication using Tensor Cores (WMMA), and accumulates results
 *  into a shared C tile. The final results are written atomically into the global
 *  output buffer to support multi-expert (top-k > 1) routing where tokens appear in
 *  multiple experts’ outputs.
 *
 *  Adapted from https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm_wmma.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>
#include <cub/device/device_radix_sort.cuh>
#include "moe_utils.cuh"
using namespace nvcuda::wmma;

namespace vllm_rs {

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))

constexpr int WMMA_K = 16;
using VecT = float4;

// Vectorized load size (float4 = 128 bits = 8 half/bfloat16 values)
constexpr int VEC_SIZE = 8;
constexpr int NUM_VECS = 32;

// We use 4 Warps (128 threads) per block
constexpr int WARPS_PER_BLOCK = 4; // 4 warps
constexpr int BLOCK_THREADS = 128; // 128 threads

constexpr int M_BLK = 32;
constexpr int N_BLK = 32;
constexpr int K_BLK = WMMA_K;           // 16


/**
 *  @brief  WMMA-based grouped MoE GEMM kernel.
 *
 *  @tparam T               Data type: half or nv_bfloat16
 *
 *  @param input            [size_m or size_m/topk, size_k]
 *  @param weights          [num_experts, size_n, size_k] compacted expert weights
 *  @param sorted_token_ids [size_m] mapping of per-token row indices (sorted by expert)
 *  @param expert_offsets   [num_experts] array of {start, len} tokens indices for each expert
 *  @param topk_weights     [size_m] optional per-token scaling weights (nullptr if unused)
 *  @param output           [size_m, size_n] global output buffer (must be zero-initialized)
 *  @param num_experts      Total number of experts
 *  @param topk             Number of experts each token is routed to
 *  @param size_m           Number of tokens
 *  @param size_n           Output hidden dimension (per expert)
 *  @param size_k           Input hidden dimension
*/
template<typename T, int WMMA_M, int WMMA_N, int WARPS_N>
__global__ void moe_gemm_grouped_kernel(
    const T* __restrict__ input,           // [size_m, size_k]
    const T* __restrict__ weights,         // [num_experts, size_n, size_k]
    const int32_t* __restrict__ sorted_token_ids, // [size_m]
    const int32_t* __restrict__ expert_offsets,   // [num_experts]
    const float* __restrict__ topk_weights, // [size_m]
    T* __restrict__ output,                 // [size_m, size_n] (Zero-initialized)
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k
) {
    // Get Segment and N-Tile for this Block
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;
    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;

    if (num_rows_in_segment == 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const T* expert_w = weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

    extern __shared__ uint8_t smem_bytes[];
    
    // A tile: [M_BLK, K_BLK] (row-major)
    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    // B tile: [N_BLK, K_BLK] (row-major)
    T* B_sh = reinterpret_cast<T*>(A_sh + M_BLK * K_BLK);
    uint8_t* C_ptr = reinterpret_cast<uint8_t*>(B_sh + N_BLK * K_BLK);

    // align next pointer to float alignment
    size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
    if (offset != 0) {
        C_ptr += (alignof(float) - offset);
    }
    float* C_sh = reinterpret_cast<float*>(C_ptr); // shared scratch for final per-block tile writes

    const int threadId = threadIdx.x;
    const int warpId = threadId / 32;
    const int laneId = threadId % 32;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;

    const int B_ELEMS_PER_BLOCK = N_BLK * K_BLK;
    const int VEC_ELEMS_B = B_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
    const int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
    const int VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;
    
    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        // We'll accumulate full-K results in per-warp fragments (initialized here)
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        // For every k_block we will load B_sh and A_sh for this m_base subsequently
        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            // Load B Tile (Weights) into B_sh
            for (int i = threadId; i < VEC_ELEMS_B; i += BLOCK_THREADS) {
                int idx = i * VEC_SIZE; // element index (0..511)
                int n_local = idx / K_BLK;
                int k_local = idx % K_BLK;

                int n_global = n_base + n_local;
                int k_global = k_base + k_local;

                // this should be always satisfied since k dim aligned to 8
                if (n_global < size_n && k_global < size_k) {
                    *reinterpret_cast<VecT*>(&B_sh[n_local * K_BLK + k_local]) = *reinterpret_cast<const VecT*>(
                        &expert_w[(size_t)n_global * size_k + k_global]
                    );
                } else {
                    *reinterpret_cast<VecT*>(&B_sh[n_local * K_BLK + k_local]) = zero_vec;
                }
            }

            // Load A Tile (Inputs) into A_sh for this m_base and this k_base
            for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
                int idx = i * VEC_SIZE; // element index
                int m_local = idx / K_BLK;
                int k_local = idx % K_BLK;

                int m_seg = m_base + m_local; // row index within segment
                int k_global = k_base + k_local;

                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg; 
                    int token_index = sorted_token_ids[token_pair_index];
                    int input_index = token_index / (topk_weights? 1: topk);
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) = *reinterpret_cast<const VecT*>(
                        &input[(size_t)input_index * size_k + k_global]
                    );
                } else {
                    // in case m dim in this segment not aligned to 8
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) = zero_vec;
                }
            }

            __syncthreads();

            // Compute (Warp-level) : update c_frag for this k_block
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

            // Point this warp to its tile in shared memory
            const T* A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
            const T* B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);

            load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
            load_matrix_sync(b_frag, B_sh_ptr, K_BLK);

            // Accumulate into c_frag (which persists across k_base iterations)
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads(); // Fix shared memory mismatch on V100
        } // end k_base loop (we have a fully-accumulated c_frag for this m_base tile)

        // Store the accumulated c_frag to C_sh (shared) once per warp
        // Point this warp to its 16x16 tile *within* the 32x32 C_sh
        float* C_sh_ptr = C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
        // store the full accumulated 16x16 tile (note ld = N_BLK, result in row-major in C_sh)
        store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);

        __syncthreads();

        // Cooperative Store from C_sh to Global
        // 128 threads write [M_BLK, N_BLK] = [32, 32] = 1024 elements
        const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
            int m_local_c = i / N_BLK; // row in C_sh (0..31)
            int n_local_c = i % N_BLK; // col in C_sh (0..31)

            int m_seg = m_base + m_local_c;    // row index within segment
            int n_global = n_base + n_local_c; // col index in output

            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK + n_local_c]; 
                    if (topk_weights) {
                        val *= topk_weights[token_index];
                    }
                    from_float(output[(size_t)token_index * size_n + n_global], val);
                }
            }
        }
    } // end m_base loop
}

}

#define LAUNCH_MOE_WMMA(DTYPE, WMMA_M, WMMA_N, WARPS_N)\
    vllm_rs::moe_gemm_grouped_kernel<DTYPE, WMMA_M, WMMA_N, WARPS_N><<<grid, block, smem_bytes, stream>>>(\
        reinterpret_cast<const DTYPE*>(input),\
        reinterpret_cast<const DTYPE*>(weights),\
        sorted_token_ids,\
        expert_offsets,\
        topk_weights,\
        reinterpret_cast<DTYPE*>(output),\
        num_experts, topk,\
        size_m, size_n, size_k \
    );\

extern "C" void moe_gemm_wmma(
    const void* input,                // [size_m, size_k]
    const void* weights,              // [num_experts, size_n, size_k]
    const int32_t* sorted_token_ids,  // [size_m] (Device)
    const int32_t* expert_ids,   // [size_m * topk]
    const float* topk_weights,        // [size_m] (Device, can be nullptr)
    void* output,                     // [size_m, size_n]
    int32_t* expert_counts, // prealloc [num_experts]
    int32_t* expert_offsets, // prealloc [num_experts + 1]
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int data_type,                    // 0 = half, 1 = bfloat16
    bool is_prefill,
    cudaStream_t stream
) {
    if (is_prefill) {
        calculate_expert_offsets(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    } else {
        calculate_expert_offsets_light(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    }

    int grid_n = CEILDIV(size_n, vllm_rs::N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(vllm_rs::BLOCK_THREADS, 1, 1);

    // Shared memory: A_sh[M_BLK, K_BLK] + B_sh[N_BLK, K_BLK]
    size_t A_sh_bytes = vllm_rs::M_BLK * vllm_rs::K_BLK * 2; // (32*16 * 2) = 1024
    size_t B_sh_bytes = vllm_rs::N_BLK * vllm_rs::K_BLK * 2; // (32*16 * 2) = 1024
    size_t C_sh_bytes = vllm_rs::M_BLK * vllm_rs::N_BLK * sizeof(float);
    size_t AB_bytes = A_sh_bytes + B_sh_bytes;
    size_t pad = (16 - (AB_bytes % 16)) % 16; 
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes; // ~6KB total needed

    if (data_type == 0) { // half
        if (is_prefill) {
            LAUNCH_MOE_WMMA(half, 16, 16, 2)
        } else {
            // we use smaller M_tile and larger N_tile for decoding
            LAUNCH_MOE_WMMA(half, 8, 32, 1)
        }
    }
#ifndef NO_BF16_KERNEL
    else if (data_type == 1) { // bfloat16
        if (is_prefill) {
            LAUNCH_MOE_WMMA(nv_bfloat16, 16, 16, 2)
        } else {
            LAUNCH_MOE_WMMA(nv_bfloat16, 8, 32, 1)
        }
    }
#endif
}

/// In-place swap of gate/up halves in experts_gate_up weight tensor.
/// Swaps [E, 0:inter, H] ↔ [E, inter:2*inter, H] using a 2MB temp buffer.
/// Called once at model load time. No extra memory retained.
extern "C" void moe_swap_gate_up_inplace(
    void* data,          // [num_experts, 2*inter, hidden] BF16
    int num_experts,
    int inter,           // moe_intermediate_size (half of dim1)
    int hidden,
    cudaStream_t stream
) {
    size_t half_bytes = (size_t)inter * hidden * sizeof(__nv_bfloat16);
    void* temp = nullptr;
    cudaMalloc(&temp, half_bytes);

    for (int e = 0; e < num_experts; e++) {
        char* base = (char*)data + (size_t)e * 2 * inter * hidden * sizeof(__nv_bfloat16);
        char* first_half = base;                          // gate
        char* second_half = base + half_bytes;             // up
        // swap: first ↔ second via temp
        cudaMemcpyAsync(temp, first_half, half_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(first_half, second_half, half_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(second_half, temp, half_bytes, cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    cudaFree(temp);
}

/// Compute per-expert prefix-sum offsets from sorted expert ids.
///
/// Wraps the `calculate_expert_offsets_light` helper so other kernels
/// (e.g. the SM100 grouped GEMM in cutlass-gemm) can reuse it without
/// pulling in the moe_utils.cuh include path. CUDA-graph capturable
/// (no thrust calls in the light variant).
///
/// @param sorted_expert_ids [size_m] Device — globally sorted expert ids
/// @param size_m            Total assignments (num_tokens * topk)
/// @param num_experts       Total expert count
/// @param expert_counts_tmp [num_experts] Device scratch — overwritten
/// @param expert_offsets    [num_experts + 1] Device output — prefix sum
/// @param stream            CUDA stream
extern "C" void moe_compute_expert_offsets_light(
    const int32_t* sorted_expert_ids,
    int size_m,
    int num_experts,
    int32_t* expert_counts_tmp,
    int32_t* expert_offsets,
    cudaStream_t stream
) {
    calculate_expert_offsets_light(sorted_expert_ids, size_m,
                                   expert_counts_tmp, expert_offsets,
                                   num_experts, stream);
}

/// Tiny init kernel: out[i] = i. Replaces `thrust::sequence`, which had
/// ~590µs host launch overhead per call vs ~5µs for this direct kernel.
static __global__ void init_iota_u32(uint32_t* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (uint32_t)i;
}

/// Sort expert assignments by expert ID on GPU using `cub::DeviceRadixSort`.
/// Produces globally sorted (expert_ids, token_ids) arrays suitable for
/// the grouped GEMM kernel.
///
/// Replaces `thrust::sort_by_key`, which had ~496µs host overhead per call
/// vs ~7µs for the underlying CUB radix sort. nsys NVTX measured ~80ms
/// per forward (48 layers × ~1.6ms thrust dispatch) wasted on host-side
/// thrust validation/launch glue. Going direct cuts that to ~1ms/forward.
///
/// `end_bit=16` covers up to 65536 experts; cub does fewer radix passes
/// than the default 32-bit sort.
///
/// @param expert_ids_in   [n] Device — flat expert IDs (not necessarily sorted)
/// @param n               Total element count (num_tokens * topk)
/// @param sorted_experts  [n] Device output — expert IDs sorted ascending
/// @param sorted_tokens   [n] Device output — corresponding token indices
/// @param stream          CUDA stream
extern "C" void moe_sort_expert_assignments(
    const uint32_t* expert_ids_in,
    int n,
    uint32_t* sorted_experts,
    uint32_t* sorted_tokens,
    cudaStream_t stream
) {
    if (n <= 0) return;

    // Query CUB temp-storage size for SortPairs<u32, u32>.
    size_t cub_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        /*d_temp_storage=*/nullptr, cub_temp_bytes,
        /*keys_in=*/expert_ids_in, /*keys_out=*/sorted_experts,
        /*values_in=*/(const uint32_t*)nullptr, /*values_out=*/sorted_tokens,
        n, 0, 16, stream);

    // Layout: [values_in_buf | cub_temp]. CUB requires distinct
    // values_in / values_out buffers (no aliasing); we produce values_in
    // via init_iota_u32 then SortPairs writes the permutation into
    // sorted_tokens.
    size_t values_bytes = (size_t)n * sizeof(uint32_t);
    size_t total_bytes = values_bytes + cub_temp_bytes;
    void* scratch = nullptr;
    cudaMallocAsync(&scratch, total_bytes, stream);

    uint32_t* values_in = reinterpret_cast<uint32_t*>(scratch);
    void* cub_temp = static_cast<uint8_t*>(scratch) + values_bytes;

    constexpr int THREADS = 256;
    int blocks = (n + THREADS - 1) / THREADS;
    init_iota_u32<<<blocks, THREADS, 0, stream>>>(values_in, n);

    cub::DeviceRadixSort::SortPairs(
        cub_temp, cub_temp_bytes,
        expert_ids_in, sorted_experts,
        values_in, sorted_tokens,
        n, 0, 16, stream);

    cudaFreeAsync(scratch, stream);
}

// ════════════════════════════════════════════════════════════════════
// DeepGEMM grouped GEMM padding helpers
//
// DeepGEMM's `m_grouped_bf16_gemm` requires each expert's slice of the
// gathered input/output to start on a 128-row boundary. Our sort gives
// us per-expert counts that aren't 128-aligned, so we have to round
// each up and gather into a padded layout.
//
// Layout produced for `n_real` real assignments routed to `num_experts`:
//
//   gathered_padded :: [padded_total, K]   (zero-init; per-expert
//                                           padding rows hold zeros so
//                                           the GEMM is a no-op there)
//   grouped_layout  :: [padded_total]      (one entry per padded row,
//                                           stating which expert owns it)
//   padded_offsets  :: [num_experts + 1]   (cumulative padded starts;
//                                           padded_offsets[E] = padded_total)
//
// After the GEMM produces gemm_out_padded, scatter writes only the
// real rows back to the caller's flat `output[sorted_token_ids[i]]`.
// ════════════════════════════════════════════════════════════════════

constexpr int MOE_DG_ALIGN = 128;

/// Compute padded prefix-sum offsets from per-expert real counts.
/// Single block; serial sweep is fine because num_experts is small (<=256).
static __global__ void compute_padded_offsets_kernel(
    const int32_t* __restrict__ real_offsets,   // [num_experts + 1]
    int32_t* __restrict__ padded_offsets,       // [num_experts + 1]
    int num_experts
) {
    if (threadIdx.x != 0) return;
    int padded = 0;
    padded_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        int count = real_offsets[e + 1] - real_offsets[e];
        int aligned = ((count + MOE_DG_ALIGN - 1) / MOE_DG_ALIGN) * MOE_DG_ALIGN;
        padded += aligned;
        padded_offsets[e + 1] = padded;
    }
}

/// Fill grouped_layout[i] = expert that owns padded row i.
///   For padded rows i in [padded_offsets[e], padded_offsets[e+1]) → e.
/// One block per expert.
static __global__ void fill_grouped_layout_kernel(
    const int32_t* __restrict__ padded_offsets, // [num_experts + 1]
    int32_t* __restrict__ grouped_layout,       // [padded_total]
    int num_experts
) {
    int e = blockIdx.x;
    if (e >= num_experts) return;
    int beg = padded_offsets[e];
    int end = padded_offsets[e + 1];
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
        grouped_layout[i] = e;
    }
}

/// Gather A into padded layout. One block per real assignment.
///
/// padded_row(i) = padded_offsets[expert] + (i - real_offsets[expert])
template <class Element>
static __global__ void moe_gather_padded_kernel(
    const Element* __restrict__ input,           // [num_tokens, K]
    const uint32_t* __restrict__ sorted_token_ids, // [n_real]
    const uint32_t* __restrict__ sorted_expert_ids, // [n_real]
    const int32_t* __restrict__ real_offsets,    // [num_experts + 1]
    const int32_t* __restrict__ padded_offsets,  // [num_experts + 1]
    Element* __restrict__ gathered_padded,       // [padded_total, K]
    int n_real, int K, int topk
) {
    int i = blockIdx.x;
    if (i >= n_real) return;
    int e = (int)sorted_expert_ids[i];
    int intra = i - real_offsets[e];
    int padded_row = padded_offsets[e] + intra;
    int token = (int)sorted_token_ids[i] / topk;

    const Element* src = input + (size_t)token * K;
    Element* dst = gathered_padded + (size_t)padded_row * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        dst[k] = src[k];
    }
}

/// Scatter D from padded layout back to assignment-flat output.
template <class Element>
static __global__ void moe_scatter_padded_kernel(
    const Element* __restrict__ gemm_out_padded, // [padded_total, N]
    const uint32_t* __restrict__ sorted_token_ids, // [n_real]
    const uint32_t* __restrict__ sorted_expert_ids, // [n_real]
    const int32_t* __restrict__ real_offsets,    // [num_experts + 1]
    const int32_t* __restrict__ padded_offsets,  // [num_experts + 1]
    Element* __restrict__ output,                // [n_real, N]
    int n_real, int N
) {
    int i = blockIdx.x;
    if (i >= n_real) return;
    int e = (int)sorted_expert_ids[i];
    int intra = i - real_offsets[e];
    int padded_row = padded_offsets[e] + intra;
    int dst_row = (int)sorted_token_ids[i];

    const Element* src = gemm_out_padded + (size_t)padded_row * N;
    Element* dst = output + (size_t)dst_row * N;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        dst[n] = src[n];
    }
}

/// Plan a padded layout: from sorted_expert_ids, write padded_offsets
/// and grouped_layout, and return padded_total via a 1-element output.
extern "C" void moe_dg_compute_padded_layout(
    const int32_t* real_offsets,        // [num_experts + 1] device, prefix-sum already computed
    int32_t* padded_offsets,            // [num_experts + 1] device output
    int32_t* grouped_layout,            // [padded_total] device output (caller sized for upper bound)
    int num_experts,
    cudaStream_t stream
) {
    compute_padded_offsets_kernel<<<1, 1, 0, stream>>>(
        real_offsets, padded_offsets, num_experts);
    int threads = 256;
    fill_grouped_layout_kernel<<<num_experts, threads, 0, stream>>>(
        padded_offsets, grouped_layout, num_experts);
}

/// Padded gather wrapper. Output `gathered_padded` must be pre-zeroed
/// (cudaMemsetAsync) by the caller — padding rows depend on it.
/// dtype: 0 = fp16, 1 = bf16.
extern "C" void moe_dg_gather_padded(
    const void* input,
    const uint32_t* sorted_token_ids,
    const uint32_t* sorted_expert_ids,
    const int32_t* real_offsets,
    const int32_t* padded_offsets,
    void* gathered_padded,
    int n_real, int K, int topk, int dtype,
    cudaStream_t stream
) {
    int threads = 256;
    if (dtype == 1) {
        moe_gather_padded_kernel<__nv_bfloat16><<<n_real, threads, 0, stream>>>(
            (const __nv_bfloat16*)input,
            sorted_token_ids, sorted_expert_ids,
            real_offsets, padded_offsets,
            (__nv_bfloat16*)gathered_padded,
            n_real, K, topk);
    } else {
        moe_gather_padded_kernel<__half><<<n_real, threads, 0, stream>>>(
            (const __half*)input,
            sorted_token_ids, sorted_expert_ids,
            real_offsets, padded_offsets,
            (__half*)gathered_padded,
            n_real, K, topk);
    }
}

/// Padded scatter wrapper. Caller's `output` is overwritten only at
/// rows that appear in sorted_token_ids — caller must zero-init for
/// rows that may not be touched (e.g. when topk-reduction expects
/// per-(token,expert) contributions written exactly once).
extern "C" void moe_dg_scatter_padded(
    const void* gemm_out_padded,
    const uint32_t* sorted_token_ids,
    const uint32_t* sorted_expert_ids,
    const int32_t* real_offsets,
    const int32_t* padded_offsets,
    void* output,
    int n_real, int N, int dtype,
    cudaStream_t stream
) {
    int threads = 256;
    if (dtype == 1) {
        moe_scatter_padded_kernel<__nv_bfloat16><<<n_real, threads, 0, stream>>>(
            (const __nv_bfloat16*)gemm_out_padded,
            sorted_token_ids, sorted_expert_ids,
            real_offsets, padded_offsets,
            (__nv_bfloat16*)output,
            n_real, N);
    } else {
        moe_scatter_padded_kernel<__half><<<n_real, threads, 0, stream>>>(
            (const __half*)gemm_out_padded,
            sorted_token_ids, sorted_expert_ids,
            real_offsets, padded_offsets,
            (__half*)output,
            n_real, N);
    }
}
