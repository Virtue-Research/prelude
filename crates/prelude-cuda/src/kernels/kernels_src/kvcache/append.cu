// Graph-safe KV cache append kernel
// Copies a single token of K and V data into the pre-allocated KV cache buffer
// at the position read from a device pointer. Also updates cu_seqlens_k[1].
//
// This is CUDA-graph-safe: the write position is read from device memory
// (not a kernel argument), so the graph can be replayed with different positions
// by updating the device pointer between replays.
//
// k_buf, v_buf: (max_seq_len, num_kv_heads, head_dim) - pre-allocated
// k_new, v_new: (1, num_kv_heads, head_dim) - single token to append
// write_pos_ptr: device pointer to uint32 containing current write position
// cu_seqlens_k: device pointer to [2] uint32 array; [1] gets set to write_pos + 1
// stride: num_kv_heads * head_dim (elements per sequence position)
#include "../common/common.cuh"

extern "C" __global__ void kv_cache_append_bf16(
    __nv_bfloat16* __restrict__ k_buf,
    __nv_bfloat16* __restrict__ v_buf,
    const __nv_bfloat16* __restrict__ k_new,
    const __nv_bfloat16* __restrict__ v_new,
    const uint32_t* __restrict__ write_pos_ptr,
    uint32_t* __restrict__ cu_seqlens_k,
    uint32_t stride  // num_kv_heads * head_dim
) {
    const uint32_t pos = *write_pos_ptr;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread copies one element (vectorized copy of K and V)
    if (tid < stride) {
        const uint64_t dst_offset = (uint64_t)pos * stride + tid;
        k_buf[dst_offset] = k_new[tid];
        v_buf[dst_offset] = v_new[tid];
    }

    // Thread 0 updates cu_seqlens_k[1]
    if (tid == 0) {
        cu_seqlens_k[1] = pos + 1;
    }
}
