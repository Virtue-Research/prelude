//! Fast attention path for small sequences (slen ≤ 16). Zero heap allocations per head.

use super::{bf16_to_f32, f32_to_bf16};

/// Fast attention for small seq_len (≤16). Zero heap allocations per head.
/// Uses stack-allocated scratch buffers and spinning GemmPool for dispatch.
pub(super) fn prefill_attention_bf16_small(
    output: &mut [u16],
    q: &[u16],
    k: &[u16],
    v: &[u16],
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;

    // Build per-request offsets
    let mut offsets = Vec::with_capacity(seq_lens.len() + 1);
    offsets.push(0usize);
    for &slen in seq_lens {
        offsets.push(offsets.last().unwrap() + slen);
    }

    // Work items: (request_idx, head_idx)
    let total_work = seq_lens.len() * num_heads;

    #[repr(C)]
    struct AttnSmallCtx {
        out_ptr: usize,
        q_ptr: usize,
        k_ptr: usize,
        v_ptr: usize,
        offsets_ptr: usize,
        seq_lens_ptr: usize,
        num_reqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        gqa_ratio: usize,
        sm_scale: f32,
    }

    let ctx = AttnSmallCtx {
        out_ptr: output.as_mut_ptr() as usize,
        q_ptr: q.as_ptr() as usize,
        k_ptr: k.as_ptr() as usize,
        v_ptr: v.as_ptr() as usize,
        offsets_ptr: offsets.as_ptr() as usize,
        seq_lens_ptr: seq_lens.as_ptr() as usize,
        num_reqs: seq_lens.len(),
        num_heads,
        num_kv_heads,
        head_dim,
        gqa_ratio,
        sm_scale,
    };

    unsafe fn attn_small_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const AttnSmallCtx);
            let total_work = ctx.num_reqs * ctx.num_heads;
            let items_per_thread = (total_work + n_threads - 1) / n_threads;
            let start = tid * items_per_thread;
            let end = (start + items_per_thread).min(total_work);

            let offsets =
                std::slice::from_raw_parts(ctx.offsets_ptr as *const usize, ctx.num_reqs + 1);
            let seq_lens =
                std::slice::from_raw_parts(ctx.seq_lens_ptr as *const usize, ctx.num_reqs);
            let q = ctx.q_ptr as *const u16;
            let k = ctx.k_ptr as *const u16;
            let v = ctx.v_ptr as *const u16;
            let output = ctx.out_ptr as *mut u16;

            for work_id in start..end {
                let req_idx = work_id / ctx.num_heads;
                let head_idx = work_id % ctx.num_heads;
                let req_start = offsets[req_idx];
                let slen = seq_lens[req_idx];
                let kv_head = head_idx / ctx.gqa_ratio;

                prefill_one_head_small(
                    output,
                    q,
                    k,
                    v,
                    req_start,
                    slen,
                    head_idx,
                    kv_head,
                    ctx.num_heads,
                    ctx.num_kv_heads,
                    ctx.head_dim,
                    ctx.sm_scale,
                );
            }
        }
    }

    let pool = super::super::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(total_work).max(1);
    unsafe {
        pool.dispatch(
            attn_small_work,
            &ctx as *const AttnSmallCtx as *const u8,
            n_threads,
        );
    }
}

/// Process one head of attention for small seq_len. Zero heap allocations.
/// Uses stack-allocated scratch buffers (max 16 KV positions × 256 head_dim).
#[inline(never)]
unsafe fn prefill_one_head_small(
    output: *mut u16,
    q: *const u16,
    k: *const u16,
    v: *const u16,
    req_start: usize,
    slen: usize,
    head_idx: usize,
    kv_head: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) {
    unsafe {
        // Stack scratch: scores[16], out_accum[256]
        // (caller guarantees slen ≤ 16, head_dim ≤ 256 for models we support)
        let mut scores = [0.0f32; 16];
        let mut out_accum = [0.0f32; 256];

        for qi in 0..slen {
            let kv_len = qi + 1; // causal mask
            let q_off = (req_start + qi) * num_heads * head_dim + head_idx * head_dim;

            // QK^T: dot product of Q[qi] with each K[0..kv_len]
            let mut max_score = f32::NEG_INFINITY;
            for ki in 0..kv_len {
                let k_off = (req_start + ki) * num_kv_heads * head_dim + kv_head * head_dim;
                let dot = dot_bf16_raw(q.add(q_off), k.add(k_off), head_dim);
                scores[ki] = dot * sm_scale;
                if scores[ki] > max_score {
                    max_score = scores[ki];
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for ki in 0..kv_len {
                scores[ki] = (scores[ki] - max_score).exp();
                sum_exp += scores[ki];
            }
            let inv_sum = 1.0 / sum_exp;
            for ki in 0..kv_len {
                scores[ki] *= inv_sum;
            }

            // Weighted sum of V
            out_accum[..head_dim].fill(0.0);
            for ki in 0..kv_len {
                let v_off = (req_start + ki) * num_kv_heads * head_dim + kv_head * head_dim;
                let w = scores[ki];
                for d in 0..head_dim {
                    out_accum[d] += w * bf16_to_f32(*v.add(v_off + d));
                }
            }

            // Write output (F32 → BF16)
            let out_off = (req_start + qi) * num_heads * head_dim + head_idx * head_dim;
            for d in 0..head_dim {
                *output.add(out_off + d) = f32_to_bf16(out_accum[d]);
            }
        }
    }
}

/// Raw BF16 dot product (scalar). Operates on raw pointers, no slice overhead.
#[inline]
unsafe fn dot_bf16_raw(a: *const u16, b: *const u16, len: usize) -> f32 {
    unsafe {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += bf16_to_f32(*a.add(i)) * bf16_to_f32(*b.add(i));
        }
        sum
    }
}
