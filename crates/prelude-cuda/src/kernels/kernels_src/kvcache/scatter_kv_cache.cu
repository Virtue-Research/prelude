// Scatter-write K/V tokens into a flash-layout paged KV cache.
//
// Replaces candle-paged-attn's reshape_and_cache_flash kernel with a
// vectorized PTX kernel loaded via cudarc (no external .so dependency).
//
// key/value:       [num_tokens, num_heads, head_size]  (contiguous BF16)
// key_cache/value_cache: [num_blocks, block_size, num_heads, head_size]  (flash layout)
// slot_mapping:    [num_tokens] I64 — target slot per token
//
// Launch: grid=(num_tokens), block=(min(n/8, 128)) where n = num_heads * head_size
// Each thread copies 8 BF16 elements (128-bit) per iteration.

#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void scatter_kv_cache_flash_bf16(
    const __nv_bfloat16* __restrict__ key,          // [num_tokens, num_heads, head_size]
    const __nv_bfloat16* __restrict__ value,        // [num_tokens, num_heads, head_size]
    __nv_bfloat16* __restrict__ key_cache,           // [num_blocks, block_size, num_heads, head_size]
    __nv_bfloat16* __restrict__ value_cache,         // [num_blocks, block_size, num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,        // [num_tokens]
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t block_size,
    uint32_t key_stride,       // stride between tokens in key (typically num_heads * head_size)
    uint32_t value_stride      // stride between tokens in value
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;  // padding token

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const uint32_t n = num_heads * head_size;

    // Source: key[token_idx, :] and value[token_idx, :]
    const __nv_bfloat16* k_src = key + token_idx * key_stride;
    const __nv_bfloat16* v_src = value + token_idx * value_stride;

    // Target: cache[block_idx, block_offset, :] — contiguous in flash layout
    const int64_t tgt_offset = (block_idx * block_size + block_offset) * n;
    __nv_bfloat16* k_dst = key_cache + tgt_offset;
    __nv_bfloat16* v_dst = value_cache + tgt_offset;

    // Vectorized copy: 8 BF16 (128 bits = float4) per thread per iteration
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    const uint32_t n8 = n / 8;  // number of float4 chunks

    for (uint32_t i = tid; i < n8; i += stride) {
        const uint32_t elem_offset = i * 8;
        // 128-bit load + store for key
        float4 kv = *reinterpret_cast<const float4*>(k_src + elem_offset);
        *reinterpret_cast<float4*>(k_dst + elem_offset) = kv;
        // 128-bit load + store for value
        float4 vv = *reinterpret_cast<const float4*>(v_src + elem_offset);
        *reinterpret_cast<float4*>(v_dst + elem_offset) = vv;
    }

    // Handle tail elements (when n is not divisible by 8)
    const uint32_t tail_start = n8 * 8;
    for (uint32_t i = tail_start + tid; i < n; i += stride) {
        k_dst[i] = k_src[i];
        v_dst[i] = v_src[i];
    }
}
