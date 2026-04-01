// C FFI interface for vendored llama.cpp MMQ kernels.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantize BF16 activations to block_q8_1_mmq format on GPU.
void llama_mmq_quantize_q8_1(
    const void* x_bf16,   // device BF16 [M, K]
    void* x_q8,           // device output Q8_1_MMQ buffer
    int64_t M, int64_t K,
    int ggml_type_id,     // weight type (determines scale layout)
    void* stream          // cudaStream_t
);

// Y[M,N] = X_q8[M,K] @ W[N,K]^T (quantized tiled matmul)
void llama_mmq_mul_mat(
    const void* W,        // device quantized weights
    const void* x_q8,     // device Q8_1_MMQ activations
    float* y,             // device F32 output [M, N]
    int64_t M, int64_t N, int64_t K,
    int ggml_type_id,     // GGML_TYPE_Q4_0 = 2, etc.
    int compute_cap,      // e.g., 80 for Ampere
    void* stream          // cudaStream_t
);

// CPU dequantize reference (for correctness testing).
// Dispatches to llama.cpp's scalar dequantize_row_* by type ID.
void llama_dequantize(
    const void* data,       // quantized block data (host memory)
    float* output,          // f32 output buffer
    int64_t num_elements,   // total number of values (not bytes)
    int ggml_type_id        // GGML_TYPE_IQ4_NL = 20, etc.
);

#ifdef __cplusplus
}
#endif
