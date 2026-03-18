// Fused MoE Routing: softmax + topk + gather + normalize
// For decode: single token, num_experts logits -> top-k selection.
// Replaces 8+ separate kernel launches with a single kernel.
//
// Input:  router_logits [num_tokens, num_experts] (BF16)
// Output: topk_weights  [num_tokens, topk] (F32, normalized)
//         topk_ids      [num_tokens, topk] (U32, expert IDs)
//         sorted_expert_ids [num_tokens * topk] (U32, sorted by expert ID)
//         sorted_token_ids  [num_tokens * topk] (U32, indices into flat topk)
//
// One block per token. Each block handles the full routing for that token.
// For num_experts <= 256, fits in registers + shared memory.
#include "../common/common.cuh"

extern "C" __global__ void fused_moe_routing_bf16(
    const __nv_bfloat16* __restrict__ router_logits, // [num_tokens, num_experts]
    float* __restrict__ topk_weights,                 // [num_tokens, topk]
    uint32_t* __restrict__ topk_ids,                  // [num_tokens, topk]
    uint32_t* __restrict__ sorted_expert_ids,         // [num_tokens * topk]
    uint32_t* __restrict__ sorted_token_ids,          // [num_tokens * topk]
    uint32_t num_tokens,
    uint32_t num_experts,
    uint32_t topk,
    bool norm_topk_prob
) {
    const uint32_t token = blockIdx.x;
    if (token >= num_tokens) return;

    const uint32_t tid = threadIdx.x;
    // Shared memory layout: [num_experts] floats for softmax values
    //                      + [num_experts] uint32 for expert indices
    extern __shared__ uint8_t smem_raw[];
    float* smem_vals = reinterpret_cast<float*>(smem_raw);
    uint32_t* smem_idx = reinterpret_cast<uint32_t*>(smem_vals + num_experts);
    float* warp_reduce_smem = reinterpret_cast<float*>(smem_idx + num_experts);

    const __nv_bfloat16* logits_row = router_logits + (uint64_t)token * num_experts;

    // ── Step 1: Load logits and find max (for numerically stable softmax) ──
    float max_val = -FLT_MAX;
    for (uint32_t i = tid; i < num_experts; i += blockDim.x) {
        float v = __bfloat162float(logits_row[i]);
        smem_vals[i] = v;
        if (v > max_val) max_val = v;
    }
    __syncthreads();

    // Block-level max reduction
    max_val = block_reduce_max(max_val, warp_reduce_smem);

    __shared__ float global_max;
    if (tid == 0) global_max = max_val;
    __syncthreads();
    max_val = global_max;

    // ── Step 2: Compute exp(x - max) and sum ──
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < num_experts; i += blockDim.x) {
        float v = expf(smem_vals[i] - max_val);
        smem_vals[i] = v;
        local_sum += v;
    }
    __syncthreads();

    // Block-level sum reduction
    local_sum = block_reduce_sum(local_sum, warp_reduce_smem);

    __shared__ float global_sum;
    if (tid == 0) global_sum = local_sum;
    __syncthreads();

    // Normalize to get softmax probabilities
    float inv_sum = 1.0f / global_sum;
    for (uint32_t i = tid; i < num_experts; i += blockDim.x) {
        smem_vals[i] *= inv_sum;
        smem_idx[i] = i;
    }
    __syncthreads();

    // ── Step 3: Top-k selection using partial sort ──
    // Only thread 0 does the top-k selection (sequential for small k like 8)
    if (tid == 0) {
        // Simple selection sort for top-k (k=8, n=128 -> 1024 comparisons, fast on GPU)
        for (uint32_t k = 0; k < topk; k++) {
            uint32_t best_idx = k;
            float best_val = smem_vals[k];
            for (uint32_t j = k + 1; j < num_experts; j++) {
                if (smem_vals[j] > best_val) {
                    best_val = smem_vals[j];
                    best_idx = j;
                }
            }
            // Swap
            if (best_idx != k) {
                float tmp_v = smem_vals[k];
                smem_vals[k] = smem_vals[best_idx];
                smem_vals[best_idx] = tmp_v;
                uint32_t tmp_i = smem_idx[k];
                smem_idx[k] = smem_idx[best_idx];
                smem_idx[best_idx] = tmp_i;
            }
        }

        // ── Step 4: Normalize top-k weights if needed ──
        float topk_sum = 0.0f;
        if (norm_topk_prob) {
            for (uint32_t k = 0; k < topk; k++) {
                topk_sum += smem_vals[k];
            }
            float inv_topk_sum = 1.0f / topk_sum;
            for (uint32_t k = 0; k < topk; k++) {
                smem_vals[k] *= inv_topk_sum;
            }
        }

        // ── Step 5: Write topk_weights and topk_ids ──
        float* out_weights = topk_weights + (uint64_t)token * topk;
        uint32_t* out_ids = topk_ids + (uint64_t)token * topk;
        for (uint32_t k = 0; k < topk; k++) {
            out_weights[k] = smem_vals[k];
            out_ids[k] = smem_idx[k];
        }

        // ── Step 6: Sort by expert ID for moe_gemm ──
        uint32_t sort_ids[32]; // max topk=32
        uint32_t sort_experts[32];
        for (uint32_t k = 0; k < topk; k++) {
            sort_experts[k] = smem_idx[k];
            sort_ids[k] = token * topk + k;
        }
        // Insertion sort by expert ID
        for (uint32_t i = 1; i < topk; i++) {
            uint32_t key_e = sort_experts[i];
            uint32_t key_id = sort_ids[i];
            int j = (int)i - 1;
            while (j >= 0 && sort_experts[j] > key_e) {
                sort_experts[j + 1] = sort_experts[j];
                sort_ids[j + 1] = sort_ids[j];
                j--;
            }
            sort_experts[j + 1] = key_e;
            sort_ids[j + 1] = key_id;
        }

        uint32_t* out_sorted_experts = sorted_expert_ids + (uint64_t)token * topk;
        uint32_t* out_sorted_tokens = sorted_token_ids + (uint64_t)token * topk;
        for (uint32_t k = 0; k < topk; k++) {
            out_sorted_experts[k] = sort_experts[k];
            out_sorted_tokens[k] = sort_ids[k];
        }
    }
}
