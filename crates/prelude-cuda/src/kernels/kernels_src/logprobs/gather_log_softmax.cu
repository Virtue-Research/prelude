// Fused gather + log_softmax kernel for prompt_logprobs extraction.
//
// Equivalent Python (and what vLLM's _topk_log_softmax_kernel does):
//
//     # logits:       [T, V]  (F32 or BF16)
//     # target_ids:   [T]     (U32)
//     # out:          [T]     (F32)
//     max_v = logits.max(dim=-1)              # [T]
//     lse = max_v + (logits - max_v).exp().sum(dim=-1).log()  # [T]
//     out = logits[arange(T), target_ids] - lse
//
// Why fuse: the naive "log_softmax then gather" path materialises a
// `[T, V] F32` temporary on the GPU even when we only need `T` values
// back. At T=1024, V=151,936 (Qwen3.5-35B-A3B) that temporary is ~622
// MB. Fusing the reductions with the gather eliminates the temporary
// entirely and drops GPU memory traffic from ~3× vocab to ~2× vocab.
//
// ## Design
//
// One block per token (row). We assume V is large enough that V >>
// block_size, so the inner loop over vocab is strided. Two passes:
//
//   Pass 1: max reduction
//     * Each thread walks elements `tid, tid+bs, tid+2*bs, ...` and
//       tracks a local max.
//     * Block-reduce across threads → global max for this row.
//
//   Pass 2: logsumexp reduction
//     * Each thread re-walks the same elements, computes `exp(x - max)`
//       and accumulates in a local sum.
//     * If the stride happens to hit our target token's index, remember
//       its raw logit value in a thread-local register so we don't have
//       to re-read it later.
//     * Block-reduce the sums → `lse = max + log(sum)`.
//
// Finalisation: we still need the target logit. Rather than having every
// thread check on every iteration, we just do a single coalesced load
// `logits[row, target_ids[row]]` from thread 0 after the reduction and
// compute the output. The load is essentially free compared to the 151K
// element scan.
//
// Numerical stability: F32 throughout. BF16 would lose precision in the
// 152K-wide logsumexp reduction (exp of anything moderately large
// overflows BF16). Caller is expected to pass F32 logits.
//
// Block size: 512 threads (16 warps). Block-wise reductions use 16
// floats of shared memory via the shared `block_reduce_max`/`_sum`
// helpers in common.cuh.

#include "../common/common.cuh"

extern "C" __global__ void gather_log_softmax_f32(
    const float* __restrict__ logits,       // [T, V]
    const uint32_t* __restrict__ target_ids, // [T]
    float* __restrict__ out,                 // [T]
    uint32_t num_tokens,
    uint32_t vocab_size
) {
    extern __shared__ float smem[];

    const uint32_t row = blockIdx.x;
    if (row >= num_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    const float* row_ptr = logits + (uint64_t)row * vocab_size;

    // ── Pass 1: max reduction ───────────────────────────────────────
    float local_max = -FLT_MAX;
    for (uint32_t i = tid; i < vocab_size; i += block_size) {
        float v = row_ptr[i];
        if (v > local_max) local_max = v;
    }
    // block_reduce_max leaves the final value only in warp 0 / lane 0.
    local_max = block_reduce_max(local_max, smem);

    // Broadcast to all threads via shared memory (warp 0 lane 0 writes,
    // everyone else reads after sync).
    __shared__ float row_max_s;
    if (tid == 0) {
        row_max_s = local_max;
    }
    __syncthreads();
    const float row_max = row_max_s;

    // ── Pass 2: logsumexp reduction ─────────────────────────────────
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < vocab_size; i += block_size) {
        float v = row_ptr[i];
        local_sum += __expf(v - row_max);
    }
    local_sum = block_reduce_sum(local_sum, smem);

    __shared__ float row_lse_s;
    if (tid == 0) {
        // lse = max + log(sum(exp(x - max)))
        row_lse_s = row_max + __logf(local_sum);
    }
    __syncthreads();

    // ── Gather + output ─────────────────────────────────────────────
    // Thread 0 does the single gather load and writes the scalar
    // output. The load is sequential but tiny compared to the two
    // full-vocab reductions above.
    if (tid == 0) {
        const uint32_t tgt = target_ids[row];
        // Clamp so a malformed `target_ids` value can't OOB read.
        // Caller should ensure ids are in range; this is a safety net
        // that just returns -inf for bad inputs rather than crashing.
        if (tgt >= vocab_size) {
            out[row] = -INFINITY;
        } else {
            float tgt_logit = row_ptr[tgt];
            out[row] = tgt_logit - row_lse_s;
        }
    }
}

// BF16 variant — same algorithm, the logits come in as BF16 and we
// convert per-element to F32 before accumulating. The reduction path
// is still F32 for numerical stability (152K-wide logsumexp in BF16
// would overflow).
extern "C" __global__ void gather_log_softmax_bf16(
    const __nv_bfloat16* __restrict__ logits, // [T, V]
    const uint32_t* __restrict__ target_ids,  // [T]
    float* __restrict__ out,                  // [T]
    uint32_t num_tokens,
    uint32_t vocab_size
) {
    extern __shared__ float smem[];

    const uint32_t row = blockIdx.x;
    if (row >= num_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    const __nv_bfloat16* row_ptr = logits + (uint64_t)row * vocab_size;

    // Pass 1: max
    float local_max = -FLT_MAX;
    for (uint32_t i = tid; i < vocab_size; i += block_size) {
        float v = __bfloat162float(row_ptr[i]);
        if (v > local_max) local_max = v;
    }
    local_max = block_reduce_max(local_max, smem);

    __shared__ float row_max_s;
    if (tid == 0) row_max_s = local_max;
    __syncthreads();
    const float row_max = row_max_s;

    // Pass 2: logsumexp
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < vocab_size; i += block_size) {
        float v = __bfloat162float(row_ptr[i]);
        local_sum += __expf(v - row_max);
    }
    local_sum = block_reduce_sum(local_sum, smem);

    __shared__ float row_lse_s;
    if (tid == 0) row_lse_s = row_max + __logf(local_sum);
    __syncthreads();

    // Gather + output
    if (tid == 0) {
        const uint32_t tgt = target_ids[row];
        if (tgt >= vocab_size) {
            out[row] = -INFINITY;
        } else {
            float tgt_logit = __bfloat162float(row_ptr[tgt]);
            out[row] = tgt_logit - row_lse_s;
        }
    }
}
