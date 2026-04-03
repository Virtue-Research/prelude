//! CUTLASS 3.x GEMM — cuBLAS replacement for SM80+. Statically linked, no JIT.
//!
//! Provides a cuBLAS-compatible dispatch interface used by candle-core's matmul.

use std::ffi::c_void;

unsafe extern "C" {
    fn cutlass_gemm_dispatch(
        a: *const c_void,
        b: *const c_void,
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
        transa: i32,
        transb: i32,
        dtype: u32,
        stream: *const c_void,
    ) -> i32;

    fn cutlass_gemm_sm80(
        a: *const c_void,
        b: *const c_void,
        d: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        dtype: u32,
        config: i32,
        stream: *const c_void,
    ) -> i32;
}

/// cuBLAS-compatible GEMM dispatch via CUTLASS.
///
/// Parameters follow cuBLAS column-major convention:
/// - D[m,n] = A[m,k] @ B[k,n] (column-major)
/// - transa: 0=no transpose, 1=transpose
/// - transb: 0=no transpose, 1=transpose
/// - dtype: 0=BF16, 1=F16, 2=F32, 3=F8E4M3
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn gemm_dispatch(
    a: *const c_void,
    b: *const c_void,
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
) -> Result<(), String> {
    let ret = unsafe {
        cutlass_gemm_dispatch(
            a, b, d, m, n, k, batch, lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            transa as i32, transb as i32,
            dtype, stream,
        )
    };
    match ret {
        0 => Ok(()),
        -10 => Err(format!(
            "CUTLASS: unsupported transpose combo transa={transa} transb={transb} \
             (only TN supported). m={m} n={n} k={k}"
        )),
        -20 => Err(format!("CUTLASS: unsupported dtype {dtype}")),
        -30 => Err(format!(
            "CUTLASS: batched GEMM failed on SM80 fallback (batch={batch}). m={m} n={n} k={k}"
        )),
        code => Err(format!(
            "CUTLASS GEMM failed (code {code}) for m={m} n={n} k={k} batch={batch} dtype={dtype}"
        )),
    }
}

/// Force SM80 CUTLASS path — for benchmarking the universal fallback on any GPU.
/// config: 0=128x128x32/s4 (default), 1=128x128x64/s3, 2=128x128x64/s4
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn gemm_sm80(
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    dtype: u32,
    config: i32,
    stream: *const c_void,
) -> Result<(), String> {
    let ret = unsafe { cutlass_gemm_sm80(a, b, d, m, n, k, dtype, config, stream) };
    match ret {
        0 => Ok(()),
        code => Err(format!(
            "CUTLASS SM80 GEMM failed (code {code}) for m={m} n={n} k={k} dtype={dtype} config={config}"
        )),
    }
}
