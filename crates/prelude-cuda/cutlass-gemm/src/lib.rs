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

    fn moe_grouped_gemm_sm100(
        input: *const c_void,
        weights: *const c_void,
        sorted_token_ids: *const u32,
        expert_offsets: *const i32,
        output: *mut c_void,
        m_total: i32,
        n: i32,
        k: i32,
        num_experts: i32,
        topk: i32,
        data_type: i32,
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
            a,
            b,
            d,
            m,
            n,
            k,
            batch,
            lda,
            ldb,
            ldd,
            stride_a,
            stride_b,
            stride_d,
            transa as i32,
            transb as i32,
            dtype,
            stream,
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

/// CUTLASS Blackwell SM100 grouped MoE GEMM. Replaces the legacy
/// SM75-era `nvcuda::wmma` `moe_gemm_wmma` kernel on B300+ for BF16
/// MoE forward.
///
/// Inputs:
/// - `input`: `[num_tokens, K]` BF16
/// - `weights`: `[num_experts, N, K]` BF16
/// - `sorted_token_ids`: `[m_total]` U32, sorted by expert id; flat
///   index into `[num_tokens × topk]` (caller divides by `topk` to
///   recover the token id during gather).
/// - `expert_offsets`: `[num_experts + 1]` I32, prefix-sum of per-expert
///   assignment counts.
/// - `output`: `[m_total, N]` BF16, written in flat (sorted_token_id)
///   layout to match the existing `moe_gemm_wmma` semantics.
///
/// Returns `Err` with `Ok(())` mapped to 0; only BF16 supported.
///
/// # Safety
/// All device pointers must be valid; layouts must match the contract
/// above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn grouped_moe_sm100(
    input: *const c_void,
    weights: *const c_void,
    sorted_token_ids: *const u32,
    expert_offsets: *const i32,
    output: *mut c_void,
    m_total: i32,
    n: i32,
    k: i32,
    num_experts: i32,
    topk: i32,
    data_type: i32,
    stream: *const c_void,
) -> Result<(), String> {
    let ret = unsafe {
        moe_grouped_gemm_sm100(
            input,
            weights,
            sorted_token_ids,
            expert_offsets,
            output,
            m_total,
            n,
            k,
            num_experts,
            topk,
            data_type,
            stream,
        )
    };
    match ret {
        0 => Ok(()),
        -100 => Err("grouped_moe_sm100: built without SM100 support".into()),
        -20 => Err(format!("grouped_moe_sm100: unsupported dtype {data_type} (BF16 only)")),
        -21 => Err(format!("grouped_moe_sm100: invalid shape m={m_total} n={n} k={k} experts={num_experts}")),
        -22 => Err("grouped_moe_sm100: workspace cudaMalloc failed".into()),
        -10 => Err("grouped_moe_sm100: per-group metadata cudaMalloc failed".into()),
        -11 => Err("grouped_moe_sm100: CUTLASS can_implement returned not-implemented (problem-shape misalignment?)".into()),
        -12 => Err("grouped_moe_sm100: CUTLASS initialize() failed".into()),
        -13 => Err("grouped_moe_sm100: CUTLASS run() returned non-success".into()),
        code => Err(format!("grouped_moe_sm100 failed (code {code})")),
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
