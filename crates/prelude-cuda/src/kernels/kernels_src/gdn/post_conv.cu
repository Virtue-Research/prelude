// Fused Gated DeltaNet post-conv1d prep kernel.
//
// Consumes the post-conv1d BF16 `mixed_qkv` tensor (layout
// `[Q (HK*K) | K (HK*K) | V (HV*V)]` along the channel axis) plus the
// per-token scalar gate inputs `a_raw`, `b_raw` and per-head parameters
// `A_log`, `dt_bias`. In one launch produces:
//
//   * q_out  — L2-normalised Q, shape `[L, HK, K]`
//   * k_out  — L2-normalised K, shape `[L, HK, K]`
//   * v_out  — raw V, shape `[L, HV, V]`
//   * alpha  — linear-space per-step decay, shape `[L, HV]`
//              alpha = exp(-exp(A_log) * softplus(a_raw + dt_bias))
//   * beta   — sigmoid(b_raw), shape `[L, HV]`
//
// Replaces a chain of ~20 candle ops per DeltaNet layer (cast, sqr,
// sum_keepdim, sqrt, div, broadcast_add, softplus as 3 ops, exp, neg,
// broadcast_mul, exp, sigmoid, ...) with a single kernel, saving ~600
// kernel launches per 30-layer prefill.
//
// Grid: (ceil(L / BLOCK_T), HK + HV)
//   * Blocks with blockIdx.y < HK handle one (Q, K) head pair
//   * Blocks with blockIdx.y >= HK handle one V head + its scalar gate
// Block: BLOCK_T * head_dim threads. Each thread owns one (t, d) cell
// inside its assigned (token-block, head). head_dim must be ≤ 128 and
// a multiple of 32 — we launch with `block_dim = BLOCK_T * head_dim`,
// which for BLOCK_T=4 and head_dim=128 is 512 threads per block.
//
// The kernel shape is parameterised at launch time by plain `int`
// arguments, so one compiled PTX serves any `(L, HK, HV, head_dim)`
// config. Template specialisation is handled by the thread-block size
// at launch, not via separate kernel instantiations.

#include "../common/common.cuh"

extern "C" __global__ void gdn_post_conv_bf16(
    const __nv_bfloat16* __restrict__ mixed_qkv,   // [L, HK*K*2 + HV*V]
    const __nv_bfloat16* __restrict__ a_raw,       // [L, HV]
    const __nv_bfloat16* __restrict__ b_raw,       // [L, HV]
    const float*         __restrict__ A_log,       // [HV]
    const float*         __restrict__ dt_bias,     // [HV]
    __nv_bfloat16* __restrict__ q_out,             // [L, HK, K]
    __nv_bfloat16* __restrict__ k_out,             // [L, HK, K]
    __nv_bfloat16* __restrict__ v_out,             // [L, HV, V]
    float*         __restrict__ alpha,             // [L, HV]
    float*         __restrict__ beta,              // [L, HV]
    int L,
    int HK,
    int HV,
    int head_dim,      // assumed K == V (Qwen3.5 pins head_k_dim == head_v_dim)
    int block_t,       // BLOCK_T: tokens per block
    float l2_eps
) {
    const int tb = blockIdx.x;          // token-block id
    const int head_id = blockIdx.y;     // head id (QK below HK, V/gate at or above HK)
    const int tid = threadIdx.x;        // 0..block_t*head_dim
    const int t_in_block = tid / head_dim;    // which token within this block
    const int d = tid - t_in_block * head_dim; // dim within this token's head
    const int t = tb * block_t + t_in_block;

    const int hk_k = HK * head_dim;
    const int mixed_stride = 2 * hk_k + HV * head_dim;  // conv_dim per token

    // Lane / warp masks for token-local warp reduction.
    // Each token occupies `head_dim` consecutive threads inside the block
    // (starting at `t_in_block * head_dim`). Since head_dim is typically
    // 128 = 4 warps, reductions across a token's dim axis span multiple
    // warps and must go through shared memory.
    //
    // Shared memory layout (per block):
    //   float ss[block_t * 2]  — per-token (ss_q, ss_k) partial sums
    //                            (QK blocks only)
    //   float scale[block_t * 2] — per-token (scale_q, scale_k) final
    //                              inverse sqrt values, broadcast back to
    //                              all head_dim threads in the token.
    // Total: 4 * block_t floats. At block_t=4 that's 64 bytes — trivial.
    extern __shared__ float smem[];

    const bool valid_t = (t < L);

    if (head_id < HK) {
        // ── Q / K head pair: load, L2-normalise, store ──────────────
        const int qk_base = t * mixed_stride + head_id * head_dim;
        const int qk_offset = qk_base + d;

        float q_val = 0.0f;
        float k_val = 0.0f;
        if (valid_t) {
            q_val = __bfloat162float(mixed_qkv[qk_offset]);
            k_val = __bfloat162float(mixed_qkv[qk_offset + hk_k]);
        }

        // Per-token sum of squares: reduce over the `head_dim` threads
        // that belong to the same token.
        //
        // With head_dim=128 and BLOCK_T=4 we have 4 warps per token but
        // the warps are at fixed offsets within the block (token i uses
        // warps 4i..4i+3). We use a per-token slice of shared memory to
        // gather the 4 warp sums, then the first lane of the token's
        // first warp does the cross-warp reduce.
        const int warps_per_token = head_dim / 32;
        const int warp_in_block = tid / 32;
        const int lane = tid & 31;
        const int warp_in_token = warp_in_block - t_in_block * warps_per_token;

        float q_sq = q_val * q_val;
        float k_sq = k_val * k_val;
        q_sq = warp_reduce_sum(q_sq);
        k_sq = warp_reduce_sum(k_sq);

        // Warp 0 of each token writes its partial sum to `smem`.
        float* token_ss = smem + t_in_block * (warps_per_token * 2);
        if (lane == 0) {
            token_ss[warp_in_token * 2 + 0] = q_sq;
            token_ss[warp_in_token * 2 + 1] = k_sq;
        }
        __syncthreads();

        // Each token's warp 0 lane 0 finishes the cross-warp reduce
        // and publishes scale_q / scale_k via a scratch slot in smem
        // for broadcast back to all head_dim threads of the token.
        float* token_scale = smem
            + block_t * (warps_per_token * 2)         // past the ss slots
            + t_in_block * 2;
        if (warp_in_token == 0 && lane == 0) {
            float total_q = 0.0f;
            float total_k = 0.0f;
            #pragma unroll
            for (int w = 0; w < warps_per_token; ++w) {
                total_q += token_ss[w * 2 + 0];
                total_k += token_ss[w * 2 + 1];
            }
            // L2 norm matches `l2_normalize_last_dim`'s Rust reference:
            //   x / (sqrt(sum(x^2)) + eps)
            float nq = sqrtf(total_q) + l2_eps;
            float nk = sqrtf(total_k) + l2_eps;
            token_scale[0] = 1.0f / nq;
            token_scale[1] = 1.0f / nk;
        }
        __syncthreads();

        if (valid_t) {
            float scale_q = token_scale[0];
            float scale_k = token_scale[1];
            const int out_off = t * hk_k + head_id * head_dim + d;
            q_out[out_off] = __float2bfloat16(q_val * scale_q);
            k_out[out_off] = __float2bfloat16(k_val * scale_k);
        }
    } else {
        // ── V head + scalar gate + beta ─────────────────────────────
        const int hv = head_id - HK;
        if (valid_t) {
            const int v_in_off =
                t * mixed_stride + 2 * hk_k + hv * head_dim + d;
            const int v_out_off = t * (HV * head_dim) + hv * head_dim + d;
            v_out[v_out_off] = mixed_qkv[v_in_off];
        }

        // Scalar per (t, hv) — only one thread per (token, hv) block
        // does this. `d == 0` + `warp_in_block == t_in_block * warps_per_token`
        // picks the first thread of the token's first warp.
        const int warps_per_token = head_dim / 32;
        const bool gate_writer =
            valid_t
            && d == 0
            && (tid / 32) == t_in_block * warps_per_token;
        if (gate_writer) {
            const int ab_off = t * HV + hv;
            float a = __bfloat162float(a_raw[ab_off]);
            float b = __bfloat162float(b_raw[ab_off]);
            float a_log_val = A_log[hv];
            float dt_bias_val = dt_bias[hv];

            // softplus(x) = log(1 + exp(x)), linearised for x > threshold
            // to avoid exp overflow. Matches PyTorch F.softplus default
            // `beta=1, threshold=20`, which is what HF Qwen3.5 uses.
            float x = a + dt_bias_val;
            float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));

            // alpha = exp(-exp(A_log) * softplus(a + dt_bias))
            //       = linear-space per-step decay for gdn_prefill
            float g = -expf(a_log_val) * softplus_x;
            alpha[ab_off] = expf(g);

            // beta = sigmoid(b_raw)
            beta[ab_off] = 1.0f / (1.0f + expf(-b));
        }
    }
}
