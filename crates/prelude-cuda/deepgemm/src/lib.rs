//! DeepGEMM BF16 GEMM — SM90 warp-specialized kernel, statically linked.
//!
//! Based on deepseek-ai/DeepGEMM. Kernel + heuristic + TMA descriptor creation
//! all happen in the C++ wrapper. Rust just passes raw pointers.
//!
//! No cuBLAS dependency. No Python. No JIT.

use std::ffi::c_void;

unsafe extern "C" {
    fn deepgemm_bf16_gemm(
        A: *mut c_void,
        B: *mut c_void,
        D: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_query_config(
        M: i32, N: i32, K: i32,
        out_block_m: *mut i32,
        out_block_n: *mut i32,
        out_stages: *mut i32,
        out_smem: *mut i32,
    );

    fn deepgemm_fp8_gemm(
        A: *mut c_void,
        B: *mut c_void,
        D: *mut c_void,
        scale_a: *mut c_void,
        scale_b: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_query_fp8_config(
        M: i32, N: i32, K: i32,
        out_block_m: *mut i32,
        out_block_n: *mut i32,
        out_stages: *mut i32,
        out_smem: *mut i32,
    );

    fn deepgemm_m_grouped_fp8_gemm(
        A: *mut c_void,
        B: *mut c_void,
        D: *mut c_void,
        scale_a: *mut c_void,
        scale_b: *mut c_void,
        grouped_layout: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        num_groups: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_m_grouped_bf16_gemm(
        A: *mut c_void,
        B: *mut c_void,
        D: *mut c_void,
        grouped_layout: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        num_groups: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_query_grouped_config(
        M: i32, N: i32, K: i32,
        out_block_m: *mut i32,
        out_block_n: *mut i32,
        out_stages: *mut i32,
        out_smem: *mut i32,
    );

    fn deepgemm_bf16_gemm_acc(
        A: *mut c_void,
        B: *mut c_void,
        C: *mut c_void,
        D: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_fp8_gemm_1d1d(
        A: *mut c_void, B: *mut c_void, D: *mut c_void,
        scale_a: *mut c_void, scale_b: *mut c_void,
        M: i32, N: i32, K: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_m_grouped_masked_bf16_gemm(
        A: *mut c_void,
        B: *mut c_void,
        D: *mut c_void,
        masked_m: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        num_groups: i32,
        expected_m: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_m_grouped_masked_fp8_gemm(
        A: *mut c_void,
        B: *mut c_void,
        D: *mut c_void,
        scale_a: *mut c_void,
        scale_b: *mut c_void,
        masked_m: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        num_groups: i32,
        expected_m: i32,
        stream: *mut c_void,
    ) -> i32;

    // ── Attention kernels ───────────────────────────────────────────

    fn deepgemm_fp8_mqa_logits(
        q: *mut c_void, kv: *mut c_void,
        kv_scales: *mut c_void, weights: *mut c_void,
        cu_seq_len_k_start: *mut c_void, cu_seq_len_k_end: *mut c_void,
        logits: *mut c_void,
        seq_len: i32, seq_len_kv: i32, max_seqlen_k: i32,
        num_heads: i32, head_dim: i32, stride_logits: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_fp8_paged_mqa_logits(
        q: *mut c_void, kv_cache: *mut c_void,
        kv_scales: *mut c_void, weights: *mut c_void,
        context_lens: *mut c_void, logits: *mut c_void,
        block_table: *mut c_void, schedule_meta: *mut c_void,
        batch_size: i32, num_heads: i32, head_dim: i32,
        num_kv_blocks: i32, block_kv: i32,
        is_context_lens_2d: i32,
        kv_cache_stride_bytes: i32, logits_stride: i32, block_table_stride: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_paged_mqa_metadata(
        context_lens: *mut c_void, schedule_metadata: *mut c_void,
        batch_size: i32, next_n: i32, is_context_lens_2d: i32,
        split_kv: i32, num_sms: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_clean_logits(
        cu_seq_len_k_start: *mut c_void, cu_seq_len_k_end: *mut c_void,
        logits: *mut c_void,
        seq_len: i32, seq_len_kv: i32, stride_logits: i32,
        next_n: i32,
        stream: *mut c_void,
    ) -> i32;

    // ── Layout utilities ────────────────────────────────────────────

    fn deepgemm_transform_sf_transpose(
        sf_in: *mut c_void, sf_out: *mut c_void,
        mn: i32, sf_k: i32, num_groups: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_transform_sf_pack_ue8m0(
        sf_in: *mut c_void, sf_out: *mut c_void,
        mn: i32, sf_k: i32, num_groups: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_einsum(
        A: *mut c_void, B: *mut c_void, D: *mut c_void,
        shape_m: i32, shape_n: i32, shape_k: i32, shape_s: i32,
        stream: *mut c_void,
    ) -> i32;

    fn deepgemm_get_tma_aligned_size(size: i32, elem_size: i32) -> i32;
    fn deepgemm_get_mk_alignment() -> i32;
    fn deepgemm_query_device(out_num_sms: *mut i32, out_gpu_arch: *mut i32);
}

/// BF16 GEMM: D\[M,N\] = A\[M,K\] @ B\[K,N\]
///
/// - A: \[M, K\] row-major BF16
/// - B: \[K, N\] col-major BF16 (= weight \[N, K\] stored row-major, used transposed)
/// - D: \[M, N\] row-major BF16 output
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn bf16_gemm(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe { deepgemm_bf16_gemm(a, b, d, m, n, k, stream) };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// FP8 E4M3 GEMM (1D2D): D\[M,N\] = (scale_a ⊗ A_fp8) @ (scale_b ⊗ B_fp8)
///
/// - A: \[M, K\] row-major FP8 E4M3
/// - B: \[K, N\] col-major FP8 E4M3 (= weight \[N, K\] row-major, transposed)
/// - D: \[M, N\] row-major BF16 output
/// - scale_a: \[ceil(K/128), align(M,4)\] FP32, M values contiguous (MN-major)
/// - scale_b: \[ceil(K/128), align(N,4)\] FP32, N values contiguous (MN-major)
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn fp8_gemm(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    scale_a: *mut c_void,
    scale_b: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe { deepgemm_fp8_gemm(a, b, d, scale_a, scale_b, m, n, k, stream) };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM FP8: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM FP8: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// Query which FP8 kernel config would be selected for a given shape.
pub fn query_fp8_config(m: i32, n: i32, k: i32) -> (i32, i32, i32, i32) {
    let mut block_m = 0i32;
    let mut block_n = 0i32;
    let mut stages = 0i32;
    let mut smem = 0i32;
    unsafe {
        deepgemm_query_fp8_config(m, n, k, &mut block_m, &mut block_n, &mut stages, &mut smem);
    }
    (block_m, block_n, stages, smem)
}

/// M-Grouped Contiguous FP8 GEMM (1D2D, for MoE):
///   D\[total_M, N\] = grouped(scale_a ⊗ A_fp8, scale_b ⊗ B_fp8)
///
/// - A: \[total_M, K\] FP8 E4M3 (shared input)
/// - B: \[G, N, K\] FP8 E4M3 (per-group weights)
/// - D: \[total_M, N\] BF16 output
/// - scale_a: \[ceil(K/128), align(total_M, 4)\] FP32 (per-token, via TMA)
/// - scale_b: \[ceil(K/128), align(N, 4)\] FP32 (per-channel, global memory)
/// - grouped_layout: \[total_M\] int32, each group aligned to 128
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn m_grouped_fp8_gemm(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    scale_a: *mut c_void,
    scale_b: *mut c_void,
    grouped_layout: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    num_groups: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_m_grouped_fp8_gemm(a, b, d, scale_a, scale_b, grouped_layout, m, n, k, num_groups, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM grouped FP8: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM grouped FP8: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// M-Grouped Contiguous BF16 GEMM (for MoE):
///   D\[total_M, N\] = grouped(A\[total_M, K\], B\[G, N, K\], grouped_layout\[total_M\])
///
/// - A: \[total_M, K\] row-major BF16 (shared input, K-major)
/// - B: \[G, N, K\] row-major BF16 (per-group weights, K-major)
/// - D: \[total_M, N\] row-major BF16 output
/// - grouped_layout: \[total_M\] int32, grouped_layout\[r\] = group index for row r
///   Each group's rows must be contiguous and aligned to 128.
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn m_grouped_bf16_gemm(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    grouped_layout: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    num_groups: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_m_grouped_bf16_gemm(a, b, d, grouped_layout, m, n, k, num_groups, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM grouped: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM grouped: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// Query which grouped GEMM kernel config would be selected for a given shape.
pub fn query_grouped_config(m: i32, n: i32, k: i32) -> (i32, i32, i32, i32) {
    let mut block_m = 0i32;
    let mut block_n = 0i32;
    let mut stages = 0i32;
    let mut smem = 0i32;
    unsafe {
        deepgemm_query_grouped_config(m, n, k, &mut block_m, &mut block_n, &mut stages, &mut smem);
    }
    (block_m, block_n, stages, smem)
}

/// Query which BF16 kernel config would be selected for a given shape.
pub fn query_config(m: i32, n: i32, k: i32) -> (i32, i32, i32, i32) {
    let mut block_m = 0i32;
    let mut block_n = 0i32;
    let mut stages = 0i32;
    let mut smem = 0i32;
    unsafe {
        deepgemm_query_config(m, n, k, &mut block_m, &mut block_n, &mut stages, &mut smem);
    }
    (block_m, block_n, stages, smem)
}

/// FP8 E4M3 1D1D GEMM: D(FP32) = A(FP8) @ B(FP8) with per-block scaling.
///
/// Unlike `fp8_gemm()` (1D2D, BF16 output), this uses per-block scaling
/// on BOTH A and B dimensions via TMA, and outputs FP32.
///
/// - A: \[M, K\] row-major FP8 E4M3
/// - B: \[K, N\] col-major FP8 E4M3
/// - D: \[M, N\] row-major FP32 output
/// - scale_a: \[ceil(K/128), align(M,4)\] FP32 (per-block, loaded via TMA)
/// - scale_b: \[ceil(K/128), align(N,4)\] FP32 (per-block, loaded via TMA)
///
/// # Safety
/// All pointers must be valid CUDA device pointers. D must be FP32.
pub unsafe fn fp8_gemm_1d1d(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    scale_a: *mut c_void,
    scale_b: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe { deepgemm_fp8_gemm_1d1d(a, b, d, scale_a, scale_b, m, n, k, stream) };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM FP8 1D1D: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM FP8 1D1D: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// BF16 GEMM with FP32 accumulation: D(FP32) += A(BF16) @ B(BF16)
///
/// - A: \[M, K\] row-major BF16
/// - B: \[K, N\] col-major BF16
/// - C: optional \[M, N\] FP32 bias (null for no bias, same as D for in-place accumulation)
/// - D: \[M, N\] row-major FP32 output (accumulation target)
///
/// If C is non-null and != D, C is copied to D before the GEMM launch.
/// The kernel then atomically adds the GEMM result to D.
///
/// # Safety
/// All pointers must be valid CUDA device pointers. D must be FP32.
pub unsafe fn bf16_gemm_acc(
    a: *mut c_void,
    b: *mut c_void,
    c: *mut c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe { deepgemm_bf16_gemm_acc(a, b, c, d, m, n, k, stream) };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM acc: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM acc: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// M-Grouped Masked BF16 GEMM (for MoE with CUDA graphs):
///   D\[G,M,N\] = masked(A\[G,M,K\], B\[G,N,K\], masked_m\[G\])
///
/// - A: \[G, M, K\] row-major BF16 (per-group, padded to M rows)
/// - B: \[G, N, K\] row-major BF16 (per-group weights)
/// - D: \[G, M, N\] row-major BF16 output
/// - masked_m: \[G\] int32, actual number of valid rows per group
/// - M: per-group padded M dimension
/// - expected_m: expected per-group M for heuristic selection
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn m_grouped_masked_bf16_gemm(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    masked_m: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    num_groups: i32,
    expected_m: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_m_grouped_masked_bf16_gemm(a, b, d, masked_m, m, n, k, num_groups, expected_m, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM masked: no kernel variant for M={m} N={n} K={k} G={num_groups}")),
        code => Err(format!("DeepGEMM masked: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

/// M-Grouped Masked FP8 GEMM (1D2D, for MoE with CUDA graphs):
///   D\[G,M,N\] = masked(scale_a ⊗ A_fp8\[G,M,K\], scale_b ⊗ B_fp8\[G,N,K\], masked_m\[G\])
///
/// - A: \[G, M, K\] FP8 E4M3 (per-group)
/// - B: \[G, N, K\] FP8 E4M3 (per-group weights)
/// - D: \[G, M, N\] BF16 output
/// - scale_a: \[G, ceil(K/128), align(M,4)\] FP32 (per-token, via TMA)
/// - scale_b: \[G, ceil(K/128), align(N,4)\] FP32 (per-channel, global memory)
/// - masked_m: \[G\] int32, actual number of valid rows per group
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn m_grouped_masked_fp8_gemm(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    scale_a: *mut c_void,
    scale_b: *mut c_void,
    masked_m: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    num_groups: i32,
    expected_m: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_m_grouped_masked_fp8_gemm(a, b, d, scale_a, scale_b, masked_m,
                                            m, n, k, num_groups, expected_m, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM masked FP8: no kernel variant for M={m} N={n} K={k} G={num_groups}")),
        code => Err(format!("DeepGEMM masked FP8: launch failed (code {code}) for M={m} N={n} K={k}")),
    }
}

// ── Attention kernels ──────────────────────────────────────────────

/// FP8 MQA Logits (prefill phase):
///   logits = weighted_relu_sum(Q @ KV^T, weights)
///
/// - q: \[seq_len * num_heads, head_dim\] FP8 E4M3
/// - kv: \[seq_len_kv, head_dim\] FP8 E4M3
/// - kv_scales: \[tma_aligned(seq_len_kv)\] FP32 per-KV-token scaling
/// - weights: \[seq_len, num_heads\] FP32 MQA head weights
/// - cu_seq_len_k_start/end: \[seq_len\] uint32 cumulative KV range per query
/// - logits: \[seq_len, stride_logits\] FP32 output
/// - max_seqlen_k: >0 enables compressed logits format
///
/// Supported: num_heads ∈ {8,16,32,64}, head_dim ∈ {64,128}, num_heads * head_dim = 128 * head_dim
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn fp8_mqa_logits(
    q: *mut c_void,
    kv: *mut c_void,
    kv_scales: *mut c_void,
    weights: *mut c_void,
    cu_seq_len_k_start: *mut c_void,
    cu_seq_len_k_end: *mut c_void,
    logits: *mut c_void,
    seq_len: i32,
    seq_len_kv: i32,
    max_seqlen_k: i32,
    num_heads: i32,
    head_dim: i32,
    stride_logits: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_fp8_mqa_logits(q, kv, kv_scales, weights,
            cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len, seq_len_kv, max_seqlen_k,
            num_heads, head_dim, stride_logits, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM MQA: no kernel for heads={num_heads} dim={head_dim}")),
        code => Err(format!("DeepGEMM MQA: launch failed (code {code})")),
    }
}

/// FP8 Paged MQA Logits (decode phase):
///   logits = paged_attention(Q, KV_cache, block_table, weights)
///
/// - q: \[batch_size * num_heads, head_dim\] FP8
/// - kv_cache: paged KV cache, \[num_kv_blocks, block_kv, head_dim\] FP8
/// - kv_scales: \[num_kv_blocks, block_kv\] FP32
/// - weights: \[batch_size, num_heads\] FP32
/// - context_lens: \[batch_size\] uint32
/// - logits: \[batch_size, logits_stride\] FP32
/// - block_table: \[batch_size, block_table_stride\] uint32
/// - schedule_meta: \[(num_sms+1)*2\] uint32 (from paged_mqa_metadata)
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fp8_paged_mqa_logits(
    q: *mut c_void,
    kv_cache: *mut c_void,
    kv_scales: *mut c_void,
    weights: *mut c_void,
    context_lens: *mut c_void,
    logits: *mut c_void,
    block_table: *mut c_void,
    schedule_meta: *mut c_void,
    batch_size: i32,
    num_heads: i32,
    head_dim: i32,
    num_kv_blocks: i32,
    block_kv: i32,
    is_context_lens_2d: bool,
    kv_cache_stride_bytes: i32,
    logits_stride: i32,
    block_table_stride: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_fp8_paged_mqa_logits(q, kv_cache, kv_scales, weights,
            context_lens, logits, block_table, schedule_meta,
            batch_size, num_heads, head_dim, num_kv_blocks, block_kv,
            is_context_lens_2d as i32,
            kv_cache_stride_bytes, logits_stride, block_table_stride, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM paged MQA: no kernel for heads={num_heads} dim={head_dim}")),
        code => Err(format!("DeepGEMM paged MQA: launch failed (code {code})")),
    }
}

/// Compute scheduling metadata for paged MQA logits.
///
/// - context_lens: \[batch_size\] uint32 on GPU
/// - schedule_metadata: \[(num_sms+1)*2\] uint32 on GPU (pre-allocated)
/// - split_kv: SM90=256, SM100=512
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn paged_mqa_metadata(
    context_lens: *mut c_void,
    schedule_metadata: *mut c_void,
    batch_size: i32,
    next_n: i32,
    is_context_lens_2d: bool,
    split_kv: i32,
    num_sms: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_paged_mqa_metadata(context_lens, schedule_metadata,
            batch_size, next_n, is_context_lens_2d as i32,
            split_kv, num_sms, stream)
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("DeepGEMM metadata: launch failed (code {code})")),
    }
}

/// Clean logits: fill -inf for out-of-range KV positions after MQA.
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn clean_logits(
    cu_seq_len_k_start: *mut c_void,
    cu_seq_len_k_end: *mut c_void,
    logits: *mut c_void,
    seq_len: i32,
    seq_len_kv: i32,
    stride_logits: i32,
    next_n: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_clean_logits(cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len, seq_len_kv, stride_logits, next_n, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM clean_logits: unsupported next_n={next_n}")),
        code => Err(format!("DeepGEMM clean_logits: launch failed (code {code})")),
    }
}

// ── Einsum ──────────────────────────────────────────────────────────

/// BF16 Einsum: D\[M,N\] = sum_s A\[s,M,K\] @ B\[s,N,K\]^T
///
/// Batched matrix multiply with FP32 accumulation via atomicAdd.
/// D must be zero-initialized before calling.
///
/// - A: \[shape_s * shape_m, shape_k\] BF16 (row-major, K-major for TMA)
/// - B: \[shape_s * shape_n, shape_k\] BF16 (row-major, K-major for TMA)
/// - D: \[shape_m, shape_n\] FP32 output (zero-init, accumulated)
///
/// shape_m/n/k must match a pre-compiled configuration:
///   (128,128,64), (128,128,128), (128,64,64), (128,64,128),
///   (256,128,64), (256,128,128)
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn einsum(
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    shape_m: i32,
    shape_n: i32,
    shape_k: i32,
    shape_s: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_einsum(a, b, d, shape_m, shape_n, shape_k, shape_s, stream)
    };
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM einsum: no kernel for M={shape_m} N={shape_n} K={shape_k}")),
        code => Err(format!("DeepGEMM einsum: launch failed (code {code})")),
    }
}

// ── Layout utilities ────────────────────────────────────────────────

/// Transpose FP32 scaling factors from \[G, MN, K/128\] (K-major) to
/// \[G, K/128, tma_aligned(MN)\] (MN-major, TMA-aligned).
///
/// # Safety
/// sf_in and sf_out must be valid CUDA device pointers.
pub unsafe fn transform_sf_transpose(
    sf_in: *mut c_void,
    sf_out: *mut c_void,
    mn: i32,
    sf_k: i32,
    num_groups: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_transform_sf_transpose(sf_in, sf_out, mn, sf_k, num_groups, stream)
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("DeepGEMM SF transpose: launch failed (code {code})")),
    }
}

/// Transform + pack FP32 scaling factors to UE8M0 format (for SM100).
///
/// # Safety
/// sf_in and sf_out must be valid CUDA device pointers.
pub unsafe fn transform_sf_pack_ue8m0(
    sf_in: *mut c_void,
    sf_out: *mut c_void,
    mn: i32,
    sf_k: i32,
    num_groups: i32,
    stream: *mut c_void,
) -> Result<(), String> {
    let ret = unsafe {
        deepgemm_transform_sf_pack_ue8m0(sf_in, sf_out, mn, sf_k, num_groups, stream)
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("DeepGEMM SF UE8M0 pack: launch failed (code {code})")),
    }
}

/// Get TMA-aligned element count for a given size and element size.
pub fn get_tma_aligned_size(size: i32, elem_size: i32) -> i32 {
    unsafe { deepgemm_get_tma_aligned_size(size, elem_size) }
}

/// Get M/K alignment for contiguous grouped layout. Always 128.
pub fn get_mk_alignment() -> i32 {
    unsafe { deepgemm_get_mk_alignment() }
}

/// Query device properties: number of SMs and GPU architecture.
pub fn query_device() -> (i32, i32) {
    let mut num_sms = 0i32;
    let mut gpu_arch = 0i32;
    unsafe { deepgemm_query_device(&mut num_sms, &mut gpu_arch) };
    (num_sms, gpu_arch)
}
