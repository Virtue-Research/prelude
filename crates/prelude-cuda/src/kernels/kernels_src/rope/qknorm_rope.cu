// Fused per-head QK-Norm + RoPE kernels
// Combines RMSNorm and Rotary Position Embeddings into a single kernel.
// Eliminates: separate norm kernel, index_select for cos/sin gather,
// separate rope kernel, and intermediate tensor allocations.
//
// Note: These kernels are specific to models with per-head QK-Norm (e.g., Qwen3).
// For standard models without QK-Norm, use separate RMSNorm + RoPE kernels.
#include "../common/common.cuh"

// ─── Fused QK-Norm + RoPE (with position_ids tensor) ────────────────────
// One warp (32 threads) handles one row = one (token, head) of D elements.
// With 256 threads per block (8 warps), processes 8 rows per block.
//
// Uses __shfl_xor_sync for:
//   1. Sum-of-squares reduction across the warp (for RMSNorm)
//   2. Exchange first/second half normalized values (for RoPE rotation)
//
// Lane 0-15 handle dims [0, D/2), lane 16-31 handle dims [D/2, D).
// Each lane holds D/32 consecutive elements. XOR with 16 swaps partners.
extern "C" __global__ void fused_qknorm_rope_bf16(
    const __nv_bfloat16* __restrict__ input,     // [n_rows, D]
    const __nv_bfloat16* __restrict__ weight,    // [D] norm weight
    const __nv_bfloat16* __restrict__ cos_table, // [max_seq_len, D/2]
    const __nv_bfloat16* __restrict__ sin_table, // [max_seq_len, D/2]
    const uint32_t* __restrict__ pos_ids,        // [total_tokens]
    __nv_bfloat16* __restrict__ output,          // [n_rows, D]
    uint32_t n_rows,        // total_tokens * num_heads
    uint32_t num_heads,
    uint32_t d,             // head_dim (must be multiple of 64, <= 256)
    float eps
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= n_rows) return;

    const uint32_t half_d = d / 2;
    const uint32_t epl = d / 32;   // elements per lane (4 for D=128)
    const uint32_t token = row / num_heads;
    const uint32_t pos = pos_ids[token];

    const __nv_bfloat16* in_row = input + (uint64_t)row * d;
    __nv_bfloat16* out_row = output + (uint64_t)row * d;
    const uint32_t dim_start = lane_id * epl;

    // ── Pass 1: Load input, compute sum-of-squares ──
    float vals[8];  // max 256/32 = 8
    float ss = 0.0f;
    for (uint32_t e = 0; e < epl; e++) {
        float v = __bfloat162float(in_row[dim_start + e]);
        vals[e] = v;
        ss += v * v;
    }

    // Warp-level reduction (no shared memory needed)
    ss = warp_reduce_sum(ss);
    float scale = rsqrtf(ss / (float)d + eps);

    // ── Normalize with weight ──
    for (uint32_t e = 0; e < epl; e++) {
        vals[e] *= scale * __bfloat162float(weight[dim_start + e]);
    }

    // ── Exchange first/second half via warp shuffle for RoPE ──
    float partner[8];
    for (uint32_t e = 0; e < epl; e++) {
        partner[e] = __shfl_xor_sync(0xffffffff, vals[e], 16);
    }

    // ── Apply rotary embeddings ──
    // First half:  out[d]      = normed[d] * cos[d] - normed[d+D/2] * sin[d]
    // Second half: out[d+D/2]  = normed[d] * sin[d] + normed[d+D/2] * cos[d]
    const bool first_half = (lane_id < 16);
    const uint32_t rope_d = first_half ? dim_start : (dim_start - half_d);
    const uint64_t cs_base = (uint64_t)pos * half_d + rope_d;

    for (uint32_t e = 0; e < epl; e++) {
        float c = __bfloat162float(cos_table[cs_base + e]);
        float sn = __bfloat162float(sin_table[cs_base + e]);
        float r;
        if (first_half) {
            r = vals[e] * c - partner[e] * sn;
        } else {
            r = partner[e] * sn + vals[e] * c;
        }
        out_row[dim_start + e] = __float2bfloat16(r);
    }
}

// ─── Fused QK-Norm + RoPE (THD layout) ──────────────────────────────────
// Same as fused_qknorm_rope_bf16 but for the THD [B,L,H,D] layout.
// Position is derived from the row index: pos = (row / num_heads) % seq_len + offset.
// No position_ids tensor needed.
extern "C" __global__ void fused_qknorm_rope_thd_bf16(
    const __nv_bfloat16* __restrict__ input,     // [B*L*H, D]
    const __nv_bfloat16* __restrict__ weight,    // [D] norm weight
    const __nv_bfloat16* __restrict__ cos_table, // [max_seq_len, D/2]
    const __nv_bfloat16* __restrict__ sin_table, // [max_seq_len, D/2]
    __nv_bfloat16* __restrict__ output,          // [B*L*H, D]
    uint32_t n_rows,        // B * L * num_heads
    uint32_t num_heads,
    uint32_t seq_len,       // L
    uint32_t d,             // head_dim
    uint32_t offset,        // position offset for KV cache
    float eps
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= n_rows) return;

    const uint32_t half_d = d / 2;
    const uint32_t epl = d / 32;
    const uint32_t pos = (row / num_heads) % seq_len + offset;

    const __nv_bfloat16* in_row = input + (uint64_t)row * d;
    __nv_bfloat16* out_row = output + (uint64_t)row * d;
    const uint32_t dim_start = lane_id * epl;

    float vals[8];
    float ss = 0.0f;
    for (uint32_t e = 0; e < epl; e++) {
        float v = __bfloat162float(in_row[dim_start + e]);
        vals[e] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    float scale = rsqrtf(ss / (float)d + eps);

    for (uint32_t e = 0; e < epl; e++) {
        vals[e] *= scale * __bfloat162float(weight[dim_start + e]);
    }

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
        float r;
        if (first_half) {
            r = vals[e] * c - partner[e] * sn;
        } else {
            r = partner[e] * sn + vals[e] * c;
        }
        out_row[dim_start + e] = __float2bfloat16(r);
    }
}

// ─── Fused QK-Norm + RoPE (THD, CUDA graph safe) ────────────────────────
// Same as fused_qknorm_rope_thd_bf16 but reads `offset` from a device pointer
// so the kernel can be captured in a CUDA graph and replayed with different offsets.
extern "C" __global__ void fused_qknorm_rope_thd_graphsafe_bf16(
    const __nv_bfloat16* __restrict__ input,     // [B*L*H, D]
    const __nv_bfloat16* __restrict__ weight,    // [D] norm weight
    const __nv_bfloat16* __restrict__ cos_table, // [max_seq_len, D/2]
    const __nv_bfloat16* __restrict__ sin_table, // [max_seq_len, D/2]
    __nv_bfloat16* __restrict__ output,          // [B*L*H, D]
    const uint32_t* __restrict__ offset_ptr,     // [1] device pointer to position offset
    uint32_t n_rows,        // B * L * num_heads
    uint32_t num_heads,
    uint32_t seq_len,       // L
    uint32_t d,             // head_dim
    float eps
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= n_rows) return;

    const uint32_t half_d = d / 2;
    const uint32_t epl = d / 32;
    const uint32_t offset = *offset_ptr;
    const uint32_t pos = (row / num_heads) % seq_len + offset;

    const __nv_bfloat16* in_row = input + (uint64_t)row * d;
    __nv_bfloat16* out_row = output + (uint64_t)row * d;
    const uint32_t dim_start = lane_id * epl;

    float vals[8];
    float ss = 0.0f;
    for (uint32_t e = 0; e < epl; e++) {
        float v = __bfloat162float(in_row[dim_start + e]);
        vals[e] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    float scale = rsqrtf(ss / (float)d + eps);

    for (uint32_t e = 0; e < epl; e++) {
        vals[e] *= scale * __bfloat162float(weight[dim_start + e]);
    }

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
        float r;
        if (first_half) {
            r = vals[e] * c - partner[e] * sn;
        } else {
            r = partner[e] * sn + vals[e] * c;
        }
        out_row[dim_start + e] = __float2bfloat16(r);
    }
}
