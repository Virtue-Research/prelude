//! GPU quantized matrix multiply for GGUF formats.
//!
//! Provides tiled MMQ (matrix-matrix with quantized weights) for prefill,
//! vendored from llama.cpp's battle-tested DP4A + tensor core kernels.
//!
//! Supports: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

use std::ffi::c_void;

// ── FFI declarations ────────────────────────────────────────────────────

unsafe extern "C" {
    fn llama_mmq_quantize_q8_1(
        x_bf16: *const c_void,
        x_q8: *mut c_void,
        m: i64,
        k: i64,
        ggml_type_id: i32,
        stream: *const c_void,
    );

    fn llama_mmq_mul_mat(
        w: *const c_void,
        x_q8: *const c_void,
        y: *mut f32,
        m: i64,
        n: i64,
        k: i64,
        ggml_type_id: i32,
        compute_cap: i32,
        stream: *const c_void,
    );
}

// ── GGML type IDs (must match llama.cpp's ggml_type enum) ───────────────

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2K  = 10,
    Q3K  = 11,
    Q4K  = 12,
    Q5K  = 13,
    Q6K  = 14,
}

// ── Public API ──────────────────────────────────────────────────────────

/// Quantize BF16 activations to Q8_1_MMQ format on GPU.
///
/// # Safety
/// All pointers must be valid CUDA device pointers on the same device.
/// `stream` must be a valid `cudaStream_t`.
pub unsafe fn quantize_q8_1(
    x_bf16: *const c_void,
    x_q8: *mut c_void,
    m: i64,
    k: i64,
    weight_type: GgmlType,
    stream: *const c_void,
) {
    unsafe {
        llama_mmq_quantize_q8_1(x_bf16, x_q8, m, k, weight_type as i32, stream);
    }
}

/// Perform quantized matrix multiplication: Y[M,N] = X[M,K] @ W[N,K]^T.
///
/// `w` contains raw GGUF quantized blocks for all N rows.
/// `x_q8` must be pre-quantized via [`quantize_q8_1`].
/// `y` receives F32 output of shape [M, N].
///
/// # Safety
/// All pointers must be valid CUDA device pointers. `stream` must be a valid `cudaStream_t`.
pub unsafe fn mul_mat_q(
    w: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
    m: i64,
    n: i64,
    k: i64,
    weight_type: GgmlType,
    compute_cap: i32,
    stream: *const c_void,
) {
    unsafe {
        llama_mmq_mul_mat(w, x_q8, y, m, n, k, weight_type as i32, compute_cap, stream);
    }
}
