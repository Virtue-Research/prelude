//! GPU quantized matrix multiply for GGUF formats.
//!
//! Provides tiled MMQ (matrix-matrix with quantized weights) for prefill,
//! vendored from llama.cpp's battle-tested DP4A + tensor core kernels.
//!
//! Supports: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

use std::ffi::c_void;

// ── FFI declarations ────────────────────────────────────────────────────

unsafe extern "C" {
    fn llama_dequantize(
        data: *const c_void,
        output: *mut f32,
        num_elements: i64,
        ggml_type_id: i32,
    );

    fn llama_gpu_dequantize(
        input: *const c_void,
        output: *mut c_void,
        num_elements: i64,
        ggml_type_id: i32,
        stream: *const c_void,
    );

    fn llama_mmvq_mul_mat_vec(
        w: *const c_void,
        x_q8: *const c_void,
        y: *mut f32,
        n: i64,
        k: i64,
        ggml_type_id: i32,
        stream: *const c_void,
    );

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
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q2K     = 10,
    Q3K     = 11,
    Q4K     = 12,
    Q5K     = 13,
    Q6K     = 14,
    IQ2XXS  = 16,
    IQ2XS   = 17,
    IQ3XXS  = 18,
    IQ1S    = 19,
    IQ4NL   = 20,
    IQ3S    = 21,
    IQ2S    = 22,
    IQ4XS   = 23,
    IQ1M    = 29,
    MXFP4   = 39,
    NVFP4   = 40,
}

// ── Public API ──────────────────────────────────────────────────────────

/// CPU dequantize reference: converts quantized blocks to f32 using llama.cpp's scalar code.
///
/// Used for correctness testing — the CPU output serves as ground truth for GPU kernel verification.
///
/// # Safety
/// `data` must point to valid quantized block data of the given type.
/// `output` must have space for `num_elements` floats.
pub unsafe fn dequantize_ref(
    data: *const c_void,
    output: *mut f32,
    num_elements: i64,
    weight_type: GgmlType,
) {
    unsafe {
        llama_dequantize(data, output, num_elements, weight_type as i32);
    }
}

/// GPU dequantize: converts quantized blocks to BF16 on GPU.
///
/// # Safety
/// `input` must point to valid quantized block data on device.
/// `output` must be a BF16 device buffer with space for `num_elements` values.
/// `stream` must be a valid `cudaStream_t`.
pub unsafe fn gpu_dequantize(
    input: *const c_void,
    output: *mut c_void,
    num_elements: i64,
    weight_type: GgmlType,
    stream: *const c_void,
) {
    unsafe {
        llama_gpu_dequantize(input, output, num_elements, weight_type as i32, stream);
    }
}

/// Fused matrix-vector multiply with quantized weights (MMVQ).
///
/// Computes `y[N] = W[N,K] @ x[K]` where W is GGUF-quantized and x is Q8_1.
/// Uses llama.cpp's vec_dot functions for correctness across all formats.
///
/// # Safety
/// All pointers must be valid CUDA device pointers. `stream` must be a valid `cudaStream_t`.
pub unsafe fn mul_mat_vec_q(
    w: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
    n: i64,
    k: i64,
    weight_type: GgmlType,
    stream: *const c_void,
) {
    unsafe {
        llama_mmvq_mul_mat_vec(w, x_q8, y, n, k, weight_type as i32, stream);
    }
}

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
