// Fused K-Norm + RoPE + KV paged cache write kernel.
// Inspired by FastLLM's FastllmQKVRMSNormRopeSplitAppendPagedCacheKernel:
// https://github.com/ztxz16/fastllm/blob/845fcaa/src/devices/cuda/fastllm-cuda.cu#L3594
// FastLLM is licensed under the Apache License, Version 2.0.
//
// Fuses three operations that are otherwise separate kernel launches:
//   1. K RMSNorm (per-head)
//   2. K RoPE (rotary position embeddings)
//   3. reshape_and_cache (scatter-write K+V to paged KV cache)
//
// Q normalization stays as a separate kernel (Q is passed to attention, not to cache).
// Two variants match the two paged cache layouts used in Prelude:
//   - THD: decode path (interleaved key_cache layout)
//   - Varlen: prefill path (flash-friendly layout)
#include "../common/common.cuh"

// ─── THD decode variant ──────────────────────────────────────────────
// Key cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
// Value cache layout: [num_blocks, num_kv_heads, head_size, block_size]
// Position derived from row index: pos = (row / num_kv_heads) % seq_len + offset
//
// One warp (32 threads) handles one (token, kv_head) row of head_dim elements.
// 256 threads/block = 8 warps = 8 rows per block.
extern "C" __global__ void fused_knorm_rope_kv_cache_write_thd_bf16(
    const __nv_bfloat16* __restrict__ k_input,        // [B * num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_input,        // [B * num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_table,      // [max_seq_len, head_dim/2]
    const __nv_bfloat16* __restrict__ sin_table,      // [max_seq_len, head_dim/2]
    __nv_bfloat16* __restrict__ key_cache,            // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    __nv_bfloat16* __restrict__ value_cache,          // [num_blocks, num_kv_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,         // [B]
    uint32_t total_kv_rows,     // B * num_kv_heads
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t x,                 // interleave factor (e.g. 8)
    uint32_t seq_len,           // L (1 for decode)
    uint32_t offset,            // position offset
    float eps
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= total_kv_rows) return;

    const uint32_t half_d = head_dim / 2;
    const uint32_t epl = head_dim / 32;   // elements per lane
    const uint32_t token_idx = row / num_kv_heads;
    const uint32_t kv_head_idx = row % num_kv_heads;
    const uint32_t pos = (row / num_kv_heads) % seq_len + offset;

    const __nv_bfloat16* k_row = k_input + (uint64_t)row * head_dim;
    const uint32_t dim_start = lane_id * epl;

    // ── K: RMSNorm ──
    float vals[8];  // max epl = 256/32 = 8
    float ss = 0.0f;
    for (uint32_t e = 0; e < epl; e++) {
        float v = __bfloat162float(k_row[dim_start + e]);
        vals[e] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    float scale = rsqrtf(ss / (float)head_dim + eps);

    for (uint32_t e = 0; e < epl; e++) {
        vals[e] *= scale * __bfloat162float(k_norm_weight[dim_start + e]);
    }

    // ── K: RoPE via warp shuffle ──
    float partner[8];
    for (uint32_t e = 0; e < epl; e++) {
        partner[e] = __shfl_xor_sync(0xffffffff, vals[e], 16);
    }

    const bool first_half = (lane_id < 16);
    const uint32_t rope_d = first_half ? dim_start : (dim_start - half_d);
    const uint64_t cs_base = (uint64_t)pos * half_d + rope_d;

    for (uint32_t e = 0; e < epl; e++) {
        float c = __bfloat162float(cos_table[cs_base + e]);
        float sn = __bfloat162float(sin_table[cs_base + e]);
        if (first_half) {
            vals[e] = vals[e] * c - partner[e] * sn;
        } else {
            vals[e] = partner[e] * sn + vals[e] * c;
        }
    }

    // ── Scatter-write K to paged cache (THD interleaved layout) ──
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;  // padding token

    const int64_t block_idx = slot_idx / (int64_t)block_size;
    const int64_t block_offset = slot_idx % (int64_t)block_size;

    // key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    for (uint32_t e = 0; e < epl; e++) {
        uint32_t d_idx = dim_start + e;
        uint32_t x_idx = d_idx / x;
        uint32_t x_off = d_idx % x;
        int64_t tgt = block_idx * num_kv_heads * (head_dim / x) * block_size * x
                    + kv_head_idx * (head_dim / x) * block_size * x
                    + x_idx * block_size * x
                    + block_offset * x
                    + x_off;
        key_cache[tgt] = __float2bfloat16(vals[e]);
    }

    // ── Scatter-write V to paged cache ──
    // value_cache: [num_blocks, num_kv_heads, head_size, block_size]
    const __nv_bfloat16* v_row = v_input + (uint64_t)row * head_dim;
    for (uint32_t e = 0; e < epl; e++) {
        uint32_t d_idx = dim_start + e;
        int64_t tgt = block_idx * num_kv_heads * head_dim * block_size
                    + kv_head_idx * head_dim * block_size
                    + d_idx * block_size
                    + block_offset;
        value_cache[tgt] = v_row[d_idx];
    }
}

// ─── Varlen prefill variant ──────────────────────────────────────────
// Flash cache layout (no interleaving):
// Key cache: [num_blocks, block_size, num_kv_heads, head_dim]
// Value cache: [num_blocks, block_size, num_kv_heads, head_dim]
// Position from explicit pos_ids tensor.
extern "C" __global__ void fused_knorm_rope_kv_cache_write_varlen_bf16(
    const __nv_bfloat16* __restrict__ k_input,        // [total_tokens * num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_input,        // [total_tokens * num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_table,      // [max_seq_len, head_dim/2]
    const __nv_bfloat16* __restrict__ sin_table,      // [max_seq_len, head_dim/2]
    const uint32_t* __restrict__ pos_ids,             // [total_tokens]
    __nv_bfloat16* __restrict__ key_cache,            // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ value_cache,          // [num_blocks, block_size, num_kv_heads, head_dim]
    const int64_t* __restrict__ slot_mapping,         // [total_tokens]
    uint32_t total_kv_rows,     // total_tokens * num_kv_heads
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    float eps
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= total_kv_rows) return;

    const uint32_t half_d = head_dim / 2;
    const uint32_t epl = head_dim / 32;
    const uint32_t token_idx = row / num_kv_heads;
    const uint32_t kv_head_idx = row % num_kv_heads;
    const uint32_t pos = pos_ids[token_idx];

    const __nv_bfloat16* k_row = k_input + (uint64_t)row * head_dim;
    const uint32_t dim_start = lane_id * epl;

    // ── K: RMSNorm ──
    float vals[8];
    float ss = 0.0f;
    for (uint32_t e = 0; e < epl; e++) {
        float v = __bfloat162float(k_row[dim_start + e]);
        vals[e] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    float scale = rsqrtf(ss / (float)head_dim + eps);

    for (uint32_t e = 0; e < epl; e++) {
        vals[e] *= scale * __bfloat162float(k_norm_weight[dim_start + e]);
    }

    // ── K: RoPE via warp shuffle ──
    float partner[8];
    for (uint32_t e = 0; e < epl; e++) {
        partner[e] = __shfl_xor_sync(0xffffffff, vals[e], 16);
    }

    const bool first_half = (lane_id < 16);
    const uint32_t rope_d = first_half ? dim_start : (dim_start - half_d);
    const uint64_t cs_base = (uint64_t)pos * half_d + rope_d;

    for (uint32_t e = 0; e < epl; e++) {
        float c = __bfloat162float(cos_table[cs_base + e]);
        float sn = __bfloat162float(sin_table[cs_base + e]);
        if (first_half) {
            vals[e] = vals[e] * c - partner[e] * sn;
        } else {
            vals[e] = partner[e] * sn + vals[e] * c;
        }
    }

    // ── Scatter-write K to flash paged cache ──
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;

    const int64_t block_idx = slot_idx / (int64_t)block_size;
    const int64_t block_offset = slot_idx % (int64_t)block_size;

    // key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    const int64_t flash_base = block_idx * block_size * num_kv_heads * head_dim
                             + block_offset * num_kv_heads * head_dim
                             + kv_head_idx * head_dim;
    for (uint32_t e = 0; e < epl; e++) {
        key_cache[flash_base + dim_start + e] = __float2bfloat16(vals[e]);
    }

    // ── Scatter-write V to flash paged cache ──
    const __nv_bfloat16* v_row = v_input + (uint64_t)row * head_dim;
    for (uint32_t e = 0; e < epl; e++) {
        value_cache[flash_base + dim_start + e] = v_row[dim_start + e];
    }
}
