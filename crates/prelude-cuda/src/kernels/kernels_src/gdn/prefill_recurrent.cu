// BF16 Gated DeltaNet recurrent prefill fallback for Blackwell.
//
// This intentionally avoids Hopper WGMMA instructions. One CUDA block owns one
// `(value_head, value_row)` state row and keeps that row in registers while it
// scans the prompt tokens. That preserves the sequential recurrence while
// exposing `HV * D` independent blocks to the GPU.

#include "../common/common.cuh"

extern "C" __global__ void gdn_prefill_recurrent_bf16(
    const __nv_bfloat16* __restrict__ q,       // [T, HQ, D]
    const __nv_bfloat16* __restrict__ k,       // [T, HQ, D]
    const __nv_bfloat16* __restrict__ v,       // [T, HV, D]
    const float* __restrict__ alpha,           // [T, HV]
    const float* __restrict__ beta,            // [T, HV]
    const long long* __restrict__ cu_seqlens,  // [num_seqs + 1]
    const float* __restrict__ initial_state,   // [num_seqs, HV, D, D] or null
    __nv_bfloat16* __restrict__ output,        // [T, HV, D]
    float* __restrict__ output_state,          // [num_seqs, HV, D, D]
    int num_seqs,
    int HQ,
    int HV,
    int D,
    float q_scale,
    int has_initial_state
) {
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int seq = blockIdx.z;
    const int col = threadIdx.x;

    if (seq >= num_seqs || h >= HV || row >= D || col >= D || D != 128) {
        return;
    }

    // Qwen3.5 uses GVA: q/k heads are repeated across value-head groups.
    if (HV < HQ || (HV % HQ) != 0) {
        return;
    }
    const int qh = h / (HV / HQ);

    const long long seq_start = cu_seqlens[seq];
    const long long seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);

    const long long state_off = (((long long)seq * HV + h) * D + row) * D + col;
    float state = has_initial_state ? initial_state[state_off] : 0.0f;

    extern __shared__ float smem[];

    for (int local_t = 0; local_t < seq_len; ++local_t) {
        const long long t = seq_start + local_t;
        const long long qk_off = (t * HQ + qh) * D + col;
        const long long vh_off = (t * HV + h) * D;
        const long long ab_off = t * HV + h;

        const float k_val = __bfloat162float(k[qk_off]);
        const float q_val = __bfloat162float(q[qk_off]);
        const float v_val = __bfloat162float(v[vh_off + row]);
        const float a = alpha[ab_off];
        const float b = beta[ab_off];

        state *= a;

        float partial = state * k_val;
        float reduced = block_reduce_sum(partial, smem);
        if (col == 0) {
            smem[0] = reduced;
        }
        __syncthreads();
        const float state_k = smem[0];
        __syncthreads();

        const float v_prime = b * (v_val - state_k);
        state += v_prime * k_val;

        partial = state * q_val;
        reduced = block_reduce_sum(partial, smem);
        if (col == 0) {
            output[vh_off + row] = __float2bfloat16(reduced * q_scale);
        }
        __syncthreads();
    }

    output_state[state_off] = state;
}
