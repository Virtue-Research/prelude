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
