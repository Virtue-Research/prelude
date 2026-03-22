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
    let ret = deepgemm_bf16_gemm(a, b, d, m, n, k, stream);
    match ret {
        0 => Ok(()),
        -1 => Err(format!("DeepGEMM: no kernel variant for M={m} N={n} K={k}")),
        code => Err(format!("DeepGEMM GEMM failed (code {code})")),
    }
}

/// Query which kernel config would be selected for a given shape.
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
