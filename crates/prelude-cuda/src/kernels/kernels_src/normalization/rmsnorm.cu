// Fast RMSNorm kernel with two specialized paths:
//
// Small D (D <= 256): Multi-row warp-parallel. Each warp handles one row.
//   With 256 threads (8 warps), processes 8 rows per block. No shared memory
//   needed for reduction - pure warp shuffles.
//
// Large D (D > 256): 1 block per row, vectorized float4 loads (8 BF16/thread).
//   Register caching avoids re-reading input in pass 2 when d == block_size * 8.
#include "../common/common.cuh"
#include "../common/vec_utils.cuh"

extern "C" __global__ void fast_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ input,   // [N, D]
    const __nv_bfloat16* __restrict__ weight,  // [D]
    __nv_bfloat16* __restrict__ output,        // [N, D]
    uint32_t n_rows,
    uint32_t d,
    float eps
) {
    // ── Small D path: 1 warp per row, multiple rows per block ──
    if (d <= 256) {
        const uint32_t warp_id = threadIdx.x / 32;
        const uint32_t lane_id = threadIdx.x % 32;
        const uint32_t rows_per_block = blockDim.x / 32;
        const uint32_t row = blockIdx.x * rows_per_block + warp_id;
        if (row >= n_rows) return;

        const __nv_bfloat16* in_row = input + (uint64_t)row * d;
        __nv_bfloat16* out_row = output + (uint64_t)row * d;

        const uint32_t elems_per_lane = d / 32;
        float vals[8];  // max 256/32 = 8
        float ss = 0.0f;

        #pragma unroll
        for (uint32_t e = 0; e < elems_per_lane; e++) {
            float v = __bfloat162float(in_row[lane_id * elems_per_lane + e]);
            vals[e] = v;
            ss += v * v;
        }

        // Warp reduction (no shared memory needed)
        ss = warp_reduce_sum(ss);
        float scale = rsqrtf(ss / (float)d + eps);

        #pragma unroll
        for (uint32_t e = 0; e < elems_per_lane; e++) {
            uint32_t idx = lane_id * elems_per_lane + e;
            float w = __bfloat162float(weight[idx]);
            out_row[idx] = __float2bfloat16(vals[e] * scale * w);
        }
        return;
    }

    // ── Large D path: 1 block per row, vectorized float4 loads ──
    extern __shared__ float smem[];

    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    const __nv_bfloat16* in_row = input + (uint64_t)row * d;
    __nv_bfloat16* out_row = output + (uint64_t)row * d;

    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    float vals[8];
    float ss = 0.0f;

    // Vectorized path: each thread handles exactly 8 elements via float4
    const bool use_cached_vec = (d == block_size * 8);
    if (use_cached_vec) {
        Vec8BF16 v;
        v.load(in_row + tid * 8);
        const __nv_bfloat162* vp = v.as_bf162();
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(vp[j]);
            vals[j * 2]     = f.x;
            vals[j * 2 + 1] = f.y;
            ss += f.x * f.x + f.y * f.y;
        }
    } else {
        // Scalar fallback for arbitrary D (two-pass scheme)
        for (uint32_t i = tid; i < d; i += block_size) {
            float v = __bfloat162float(in_row[i]);
            ss += v * v;
        }
    }

    // Block-level reduction
    ss = block_reduce_sum(ss, smem);

    __shared__ float rms_scale;
    if (tid == 0) {
        rms_scale = rsqrtf(ss / (float)d + eps);
    }
    __syncthreads();
    float scale = rms_scale;

    // Pass 2: normalize
    if (use_cached_vec) {
        Vec8BF16 w;
        w.load(weight + tid * 8);
        const __nv_bfloat162* wp = w.as_bf162();

        Vec8BF16 result;
        __nv_bfloat162* op = result.as_bf162_mut();
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 wf = __bfloat1622float2(wp[j]);
            float2 r;
            r.x = vals[j * 2]     * scale * wf.x;
            r.y = vals[j * 2 + 1] * scale * wf.y;
            op[j] = __float22bfloat162_rn(r);
        }
        result.store(out_row + tid * 8);
    } else {
        for (uint32_t i = tid; i < d; i += block_size) {
            float v = __bfloat162float(in_row[i]);
            float w = __bfloat162float(weight[i]);
            out_row[i] = __float2bfloat16(v * scale * w);
        }
    }
}
