//! GPU GEMM dispatch registration.
//!
//! `register_gpu_gemm()` hooks `gemm_dispatch_impl` into candle-core's GEMM
//! dispatch so that every `Tensor::matmul()` on CUDA routes through DeepGEMM
//! (fast path for SM90+ BF16) with a CUTLASS fallback. `DenseLinear::forward`
//! calls `x.matmul(&weight.t())` and transparently gets this routing.

use std::ffi::c_void;

/// Register our GEMM dispatch. Must be called before any GPU matmul.
///
/// This replaces cuBLAS: all `Tensor::matmul()` on CUDA will route through
/// CUTLASS (SM80+) with optional DeepGEMM fast path (SM90+ BF16).
pub fn register_gpu_gemm() {
    candle_core::cuda_backend::gemm_dispatch::register_gemm_dispatch(gemm_dispatch_impl);
    tracing::info!("GPU GEMM backend registered (CUTLASS{})",
        if cfg!(feature = "deepgemm") { " + DeepGEMM" } else { "" });
}

/// GEMM dispatch implementation.
pub(crate) unsafe fn gemm_dispatch_impl(
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
) -> i32 {
    // Try DeepGEMM first for non-batched BF16 on SM90+ (fastest path).
    //
    // cuBLAS convention: m=N_features, n=M_tokens, a=weight, b=input.
    // DeepGEMM expects row-major: D[M,N] = A[M,K] @ B[K,N](col-major),
    // where M=tokens, N=features — the natural layout for LLM inference.
    // So we swap (a↔b, m↔n) to give DeepGEMM the correct orientation.
    //
    // After swap: DeepGEMM M=tokens(n), N=features(m), K=K.
    // Features (m) and K are always model-dimension-aligned.
    // Tokens (n) can be any value — DeepGEMM handles partial tiles for M.
    //
    if dtype == 0 && batch == 1 && transa && !transb {
        // Swap: DeepGEMM A=input(b), B=weight(a), M=tokens(n), N=features(m)
        let ret = deepgemm::bf16_gemm(
            b as *mut c_void, a as *mut c_void, d,
            n, m, k, stream as *mut c_void,
        );
        match &ret {
            Ok(()) => return 0,
            Err(e) => {
                tracing::debug!("DeepGEMM → CUTLASS fallback: {e}");
            }
        }
    }

    // CUTLASS fallback (SM80+, handles BF16/FP16/F32)
    {
        let ret = cutlass_gemm::gemm_dispatch(
            a, b, d, m, n, k, batch, lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            transa, transb, dtype, stream,
        );
        return match ret {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("CUTLASS GEMM error: {e}");
                -1
            }
        };
    }

    #[allow(unreachable_code)]
    -99 // No GEMM backend available
}
