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
    const __nv_bfloat16* __restrict__ input,     // [total_tokens, num_heads, D] (token dim may be non-contig)
    const __nv_bfloat16* __restrict__ weight,    // [D] norm weight
    const __nv_bfloat16* __restrict__ cos_table, // [max_seq_len, D/2]
    const __nv_bfloat16* __restrict__ sin_table, // [max_seq_len, D/2]
    const uint32_t* __restrict__ pos_ids,        // [total_tokens]
    __nv_bfloat16* __restrict__ output,          // [n_rows, D] (always contiguous)
    uint32_t n_rows,        // total_tokens * num_heads
    uint32_t num_heads,
    uint32_t d,             // head_dim (must be multiple of 64, <= 256)
    float eps,
    uint32_t token_stride   // stride between tokens in input (num_heads*d when contiguous)
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= n_rows) return;

    const uint32_t half_d = d / 2;
    const uint32_t epl = d / 32;   // elements per lane (4 for D=128)
    const uint32_t token = row / num_heads;
    const uint32_t head = row % num_heads;
    const uint32_t pos = pos_ids[token];

    const __nv_bfloat16* in_row = input + (uint64_t)token * token_stride + (uint64_t)head * d;
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

// ─── Per-row body of fused_qknorm_rope_bf16, factored for reuse ─────────
// Processes one row = one (token, head): RMSNorm(weight) then full RoPE.
// Identical math to fused_qknorm_rope_bf16; the merged Q+K kernel below
// dispatches per-row to this with the correct Q/K pointers and params.
__device__ __forceinline__ void qknorm_rope_row_bf16(
    const __nv_bfloat16* __restrict__ input,     // base ptr for this tensor
    const __nv_bfloat16* __restrict__ weight,    // [d] norm weight
    const __nv_bfloat16* __restrict__ cos_table, // [max_seq_len, d/2]
    const __nv_bfloat16* __restrict__ sin_table, // [max_seq_len, d/2]
    const uint32_t* __restrict__ pos_ids,        // [total_tokens]
    __nv_bfloat16* __restrict__ output,          // contiguous [n_rows, d]
    uint32_t local_row,     // row index within this tensor
    uint32_t lane_id,
    uint32_t num_heads,
    uint32_t d,
    float eps,
    uint32_t token_stride
) {
    const uint32_t half_d = d / 2;
    const uint32_t epl = d / 32;   // elements per lane (4 for D=128)
    const uint32_t token = local_row / num_heads;
    const uint32_t head = local_row % num_heads;
    const uint32_t pos = pos_ids[token];

    const __nv_bfloat16* in_row = input + (uint64_t)token * token_stride + (uint64_t)head * d;
    __nv_bfloat16* out_row = output + (uint64_t)local_row * d;
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

// ─── Merged Q+K Fused QK-Norm + RoPE (single launch) ────────────────────
// One launch handles both Q and K to halve kernel launches per attn layer.
// Rows [0, q_rows) are Q (q_input/q_weight/q_output, q_num_heads); rows
// [q_rows, q_rows+k_rows) are K (k_*; local row = global_row - q_rows).
// cos/sin/pos_ids/d/eps are shared. Bit-exact vs two separate launches.
extern "C" __global__ void fused_qknorm_rope_qk_bf16(
    const __nv_bfloat16* __restrict__ q_input,
    const __nv_bfloat16* __restrict__ k_input,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    const uint32_t* __restrict__ pos_ids,
    __nv_bfloat16* __restrict__ q_output,
    __nv_bfloat16* __restrict__ k_output,
    uint32_t q_rows,            // total_tokens * q_num_heads
    uint32_t k_rows,            // total_tokens * k_num_heads
    uint32_t q_num_heads,
    uint32_t k_num_heads,
    uint32_t d,                 // head_dim (shared)
    float eps,
    uint32_t q_token_stride,
    uint32_t k_token_stride
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= q_rows + k_rows) return;

    if (row < q_rows) {
        qknorm_rope_row_bf16(q_input, q_weight, cos_table, sin_table, pos_ids,
                             q_output, row, lane_id, q_num_heads, d, eps, q_token_stride);
    } else {
        qknorm_rope_row_bf16(k_input, k_weight, cos_table, sin_table, pos_ids,
                             k_output, row - q_rows, lane_id, k_num_heads, d, eps, k_token_stride);
    }
}

extern "C" __global__ void fused_qknorm_partial_rope_bf16_f32_weight(
    const __nv_bfloat16* __restrict__ input,     // [total_tokens, num_heads, D]
    const float* __restrict__ weight,            // [D] norm weight
    const __nv_bfloat16* __restrict__ cos_table, // [max_seq_len, rotary_dim/2]
    const __nv_bfloat16* __restrict__ sin_table, // [max_seq_len, rotary_dim/2]
    const uint32_t* __restrict__ pos_ids,        // [total_tokens]
    __nv_bfloat16* __restrict__ output,          // [n_rows, D]
    uint32_t n_rows,
    uint32_t num_heads,
    uint32_t d,
    uint32_t rotary_dim,
    float eps,
    uint32_t token_stride
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= n_rows) return;

    const uint32_t epl = d / 32;
    const uint32_t half_rot = rotary_dim / 2;
    const uint32_t lane_xor = half_rot / epl;
    const uint32_t token = row / num_heads;
    const uint32_t head = row % num_heads;
    const uint32_t pos = pos_ids[token];

    const __nv_bfloat16* in_row = input + (uint64_t)token * token_stride + (uint64_t)head * d;
    __nv_bfloat16* out_row = output + (uint64_t)row * d;
    const uint32_t dim_start = lane_id * epl;

    float vals[8];
    float ss = 0.0f;
    for (uint32_t e = 0; e < epl; e++) {
        uint32_t dim = dim_start + e;
        float v = __bfloat162float(in_row[dim]);
        vals[e] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    float scale = rsqrtf(ss / (float)d + eps);

    for (uint32_t e = 0; e < epl; e++) {
        vals[e] *= scale * weight[dim_start + e];
    }

    float partner[8];
    for (uint32_t e = 0; e < epl; e++) {
        partner[e] = __shfl_xor_sync(0xffffffff, vals[e], lane_xor);
    }

    for (uint32_t e = 0; e < epl; e++) {
        uint32_t dim = dim_start + e;
        float r = vals[e];
        if (dim < rotary_dim) {
            const bool first_half = dim < half_rot;
            const uint32_t rope_d = first_half ? dim : (dim - half_rot);
            const uint64_t cs_base = (uint64_t)pos * half_rot + rope_d;
            float c = __bfloat162float(cos_table[cs_base]);
            float sn = __bfloat162float(sin_table[cs_base]);
            if (first_half) {
                r = vals[e] * c - partner[e] * sn;
            } else {
                r = partner[e] * sn + vals[e] * c;
            }
        }
        out_row[dim] = __float2bfloat16(r);
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

// ─── D=128 specialization of the merged Q+K kernel ──────────────────────
// The generic kernel keeps per-lane element arrays (vals[8]/partner[8])
// indexed by a RUNTIME loop bound (epl = d/32), so they live in local
// memory (64-byte stack frame per thread) and every global access is a
// scalar 2-byte ld/st — measured ~3x off the memory roof (173us/call at
// 8k tokens, roof ~57us). This d==128 variant fixes both:
//   * compile-time EPL=4 → everything fully unrolled into registers
//   * one uint2 (8 bytes = 4 bf16) vectorized load/store per lane
// Math is identical, in the same order (fp32 accumulate, same reduction,
// same fma sequence) → bit-exact vs the generic kernel.
extern "C" __global__ void fused_qknorm_rope_qk_d128_bf16(
    const __nv_bfloat16* __restrict__ q_input,
    const __nv_bfloat16* __restrict__ k_input,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos_table, // [max_pos, 64]
    const __nv_bfloat16* __restrict__ sin_table, // [max_pos, 64]
    const uint32_t* __restrict__ pos_ids,        // [total_tokens]
    __nv_bfloat16* __restrict__ q_output,
    __nv_bfloat16* __restrict__ k_output,
    uint32_t q_rows,
    uint32_t k_rows,
    uint32_t q_num_heads,
    uint32_t k_num_heads,
    float eps,
    uint32_t q_token_stride,
    uint32_t k_token_stride
) {
    constexpr uint32_t D = 128;
    constexpr uint32_t HALF_D = 64;
    constexpr uint32_t EPL = 4; // elements per lane

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t rows_per_block = blockDim.x / 32;
    const uint32_t row = blockIdx.x * rows_per_block + warp_id;
    if (row >= q_rows + k_rows) return;

    const bool is_q = row < q_rows;
    const uint32_t local_row = is_q ? row : row - q_rows;
    const uint32_t num_heads = is_q ? q_num_heads : k_num_heads;
    const uint32_t token_stride = is_q ? q_token_stride : k_token_stride;
    const __nv_bfloat16* input = is_q ? q_input : k_input;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;
    __nv_bfloat16* output = is_q ? q_output : k_output;

    const uint32_t token = local_row / num_heads;
    const uint32_t head = local_row % num_heads;
    const uint32_t pos = pos_ids[token];

    const __nv_bfloat16* in_row =
        input + (uint64_t)token * token_stride + (uint64_t)head * D;
    __nv_bfloat16* out_row = output + (uint64_t)local_row * D;
    const uint32_t dim_start = lane_id * EPL;

    // ── Vectorized load: 4 bf16 = one uint2 per lane ──
    // All row bases are 8B-aligned (head_dim*2B = 256B strides; the Rust
    // wrapper falls back to the generic kernel otherwise).
    uint2 in_v = *reinterpret_cast<const uint2*>(in_row + dim_start);
    __nv_bfloat162 p01 = *reinterpret_cast<const __nv_bfloat162*>(&in_v.x);
    __nv_bfloat162 p23 = *reinterpret_cast<const __nv_bfloat162*>(&in_v.y);
    float v0 = __bfloat162float(p01.x);
    float v1 = __bfloat162float(p01.y);
    float v2 = __bfloat162float(p23.x);
    float v3 = __bfloat162float(p23.y);

    // Same accumulation order as the generic kernel (e ascending).
    float ss = v0 * v0;
    ss += v1 * v1;
    ss += v2 * v2;
    ss += v3 * v3;
    ss = warp_reduce_sum(ss);
    const float scale = rsqrtf(ss / (float)D + eps);

    uint2 w_v = *reinterpret_cast<const uint2*>(weight + dim_start);
    __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&w_v.x);
    __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&w_v.y);
    v0 *= scale * __bfloat162float(w01.x);
    v1 *= scale * __bfloat162float(w01.y);
    v2 *= scale * __bfloat162float(w23.x);
    v3 *= scale * __bfloat162float(w23.y);

    // ── Exchange halves (lane i <-> lane i^16) for RoPE ──
    const float r0 = __shfl_xor_sync(0xffffffff, v0, 16);
    const float r1 = __shfl_xor_sync(0xffffffff, v1, 16);
    const float r2 = __shfl_xor_sync(0xffffffff, v2, 16);
    const float r3 = __shfl_xor_sync(0xffffffff, v3, 16);

    const bool first_half = (lane_id < 16);
    const uint32_t rope_d = first_half ? dim_start : (dim_start - HALF_D);
    const uint64_t cs_base = (uint64_t)pos * HALF_D + rope_d;

    uint2 c_v = *reinterpret_cast<const uint2*>(cos_table + cs_base);
    uint2 s_v = *reinterpret_cast<const uint2*>(sin_table + cs_base);
    __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(&c_v.x);
    __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(&c_v.y);
    __nv_bfloat162 s01 = *reinterpret_cast<const __nv_bfloat162*>(&s_v.x);
    __nv_bfloat162 s23 = *reinterpret_cast<const __nv_bfloat162*>(&s_v.y);

    float o0, o1, o2, o3;
    if (first_half) {
        o0 = v0 * __bfloat162float(c01.x) - r0 * __bfloat162float(s01.x);
        o1 = v1 * __bfloat162float(c01.y) - r1 * __bfloat162float(s01.y);
        o2 = v2 * __bfloat162float(c23.x) - r2 * __bfloat162float(s23.x);
        o3 = v3 * __bfloat162float(c23.y) - r3 * __bfloat162float(s23.y);
    } else {
        o0 = r0 * __bfloat162float(s01.x) + v0 * __bfloat162float(c01.x);
        o1 = r1 * __bfloat162float(s01.y) + v1 * __bfloat162float(c01.y);
        o2 = r2 * __bfloat162float(s23.x) + v2 * __bfloat162float(c23.x);
        o3 = r3 * __bfloat162float(s23.y) + v3 * __bfloat162float(c23.y);
    }

    __nv_bfloat162 q01 = __floats2bfloat162_rn(o0, o1);
    __nv_bfloat162 q23 = __floats2bfloat162_rn(o2, o3);
    uint2 out_v;
    out_v.x = *reinterpret_cast<const uint32_t*>(&q01);
    out_v.y = *reinterpret_cast<const uint32_t*>(&q23);
    *reinterpret_cast<uint2*>(out_row + dim_start) = out_v;
}
