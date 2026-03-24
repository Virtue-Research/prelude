//! Pluggable GPU GEMM dispatch — replaces cuBLAS with custom backends (CUTLASS/DeepGEMM).
//!
//! Users register a GEMM function at startup via `register_gemm_dispatch()`.
//! All `Tensor::matmul()` on CUDA then routes through the registered function.

use std::ffi::c_void;
use std::sync::OnceLock;

/// Signature for a strided batched GEMM function.
///
/// Computes: D[b,m,n] = A[b,m,k] @ B[b,k,n] for each batch element.
///
/// - `a`, `b_ptr`, `d`: device pointers (already offset to start)
/// - `m`, `n`, `k`: matrix dimensions
/// - `batch`: number of batch elements (1 for non-batched)
/// - `lda`, `ldb`, `ldd`: leading dimensions
/// - `stride_a`, `stride_b`, `stride_d`: batch strides (0 if batch==1)
/// - `transa`, `transb`: 0 = no transpose, 1 = transpose
/// - `dtype`: 0 = BF16, 1 = F16, 2 = F32, 3 = F64
/// - `stream`: CUDA stream pointer
///
/// Returns 0 on success, nonzero on error.
pub type GemmDispatchFn = unsafe fn(
    a: *const c_void,
    b_ptr: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    stride_a: i64,
    stride_b: i64,
    stride_d: i64,
    transa: bool,
    transb: bool,
    dtype: u32,
    stream: *const c_void,
) -> i32;

static GEMM_DISPATCH: OnceLock<GemmDispatchFn> = OnceLock::new();

/// Register a custom GEMM dispatch function. Must be called before any GPU matmul.
pub fn register_gemm_dispatch(f: GemmDispatchFn) {
    GEMM_DISPATCH.set(f).ok();
}

/// Call the registered GEMM dispatch. Returns Err if no dispatch is registered.
pub(crate) fn call_gemm(
    a: *const c_void,
    b_ptr: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    stride_a: i64,
    stride_b: i64,
    stride_d: i64,
    transa: bool,
    transb: bool,
    dtype: u32,
    stream: *const c_void,
) -> std::result::Result<(), String> {
    let f = GEMM_DISPATCH.get().ok_or_else(|| {
        "No GPU GEMM backend registered. Call candle_core::cuda_backend::register_gemm_dispatch() \
         with a CUTLASS/DeepGEMM implementation before using GPU matmul."
            .to_string()
    })?;
    let ret = unsafe {
        f(
            a, b_ptr, d, m, n, k, batch, lda, ldb, ldd, stride_a, stride_b, stride_d, transa,
            transb, dtype, stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(format!(
            "GPU GEMM dispatch failed (code {ret}) for m={m} n={n} k={k} batch={batch} dtype={dtype}"
        ))
    }
}
