// Thin FFI wrapper exposing GGML quantized matmul operations.
// Compiled by build.rs alongside ggml-quants.c and arch/x86/quants.c.

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-cpu.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

// FP16 → F32 lookup table (required by simd-mappings.h).
// Normally defined in ggml-cpu.c; we provide it here to avoid pulling in the entire file.
float ggml_table_f32_f16[1 << 16];

static int _ggml_table_initialized = 0;

// Convert IEEE 754 FP16 bit pattern to FP32 (no dependency on ggml macros).
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Denormalized: convert to normalized FP32
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)mant << 13); // Inf/NaN
    } else {
        f = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

static void ensure_f16_table(void) {
    if (_ggml_table_initialized) return;
    for (int i = 0; i < (1 << 16); i++) {
        ggml_table_f32_f16[i] = fp16_to_fp32((uint16_t)i);
    }
    _ggml_table_initialized = 1;
}

// ── Quantized matrix-vector multiply ────────────────────────────────────
//
// Computes: output[m] = weight[m, k] @ input[k]
// where weight is in quantized format (Q8_0, Q4_K_M, etc.)
// and input/output are F32.
//
// This is the hot inner loop for GGUF inference.

void ggml_quants_matvec_q8_0(
    const void *weight,    // [m * k / 32 * sizeof(block_q8_0)] quantized weight
    const float *input,    // [k] F32 input vector
    float *output,         // [m] F32 output vector
    int m,                 // output dimension (rows)
    int k                  // inner dimension (cols)
) {
    ensure_f16_table();
    // Quantize input to Q8_0 for dot product
    const int nb_input = k / QK8_0;
    block_q8_0 *input_q8 = (block_q8_0 *)malloc(nb_input * sizeof(block_q8_0));
    quantize_row_q8_0(input, input_q8, k);

    const block_q8_0 *w = (const block_q8_0 *)weight;
    const int nb_row = k / QK8_0;  // blocks per row

    for (int i = 0; i < m; i++) {
        ggml_vec_dot_q8_0_q8_0(
            k, &output[i],
            sizeof(block_q8_0),
            w + i * nb_row, sizeof(block_q8_0),
            input_q8, sizeof(block_q8_0),
            1  // nrc
        );
    }

    free(input_q8);
}

void ggml_quants_matvec_q4_K(
    const void *weight,    // quantized Q4_K weight
    const float *input,    // [k] F32 input
    float *output,         // [m] F32 output
    int m, int k
) {
    ensure_f16_table();
    const int nb_input = k / QK_K;
    block_q8_K *input_q8k = (block_q8_K *)malloc(nb_input * sizeof(block_q8_K));
    quantize_row_q8_K(input, input_q8k, k);

    const block_q4_K *w = (const block_q4_K *)weight;
    const int nb_row = k / QK_K;

    for (int i = 0; i < m; i++) {
        ggml_vec_dot_q4_K_q8_K(
            k, &output[i],
            sizeof(block_q4_K),
            w + i * nb_row, sizeof(block_q4_K),
            input_q8k, sizeof(block_q8_K),
            1
        );
    }

    free(input_q8k);
}

// ── Batch matmul: weight[m, k] @ input[n, k]^T → output[n, m] ──────────
// For when we have multiple input vectors (batch/sequence tokens).

void ggml_quants_matmul_q8_0(
    const void *weight,    // [m, k] quantized Q8_0
    const float *input,    // [n, k] F32 input (row-major)
    float *output,         // [n, m] F32 output (row-major)
    int m, int k, int n
) {
    ensure_f16_table();
    const int nb_input = k / QK8_0;
    block_q8_0 *input_q8 = (block_q8_0 *)malloc(n * nb_input * sizeof(block_q8_0));

    // Quantize all input rows
    for (int j = 0; j < n; j++) {
        quantize_row_q8_0(input + j * k, input_q8 + j * nb_input, k);
    }

    const block_q8_0 *w = (const block_q8_0 *)weight;
    const int nb_row = k / QK8_0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            ggml_vec_dot_q8_0_q8_0(
                k, &output[j * m + i],
                sizeof(block_q8_0),
                w + i * nb_row, sizeof(block_q8_0),
                input_q8 + j * nb_input, sizeof(block_q8_0),
                1
            );
        }
    }

    free(input_q8);
}

void ggml_quants_matmul_q4_K(
    const void *weight,
    const float *input,
    float *output,
    int m, int k, int n
) {
    ensure_f16_table();
    const int nb_input = k / QK_K;
    block_q8_K *input_q8k = (block_q8_K *)malloc(n * nb_input * sizeof(block_q8_K));

    for (int j = 0; j < n; j++) {
        quantize_row_q8_K(input + j * k, input_q8k + j * nb_input, k);
    }

    const block_q4_K *w = (const block_q4_K *)weight;
    const int nb_row = k / QK_K;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            ggml_vec_dot_q4_K_q8_K(
                k, &output[j * m + i],
                sizeof(block_q4_K),
                w + i * nb_row, sizeof(block_q4_K),
                input_q8k + j * nb_input, sizeof(block_q8_K),
                1
            );
        }
    }

    free(input_q8k);
}

// ── Dequantize to F32 ───────────────────────────────────────────────────

void ggml_quants_dequantize_q8_0(
    const void *src, float *dst, int k
) {
    dequantize_row_q8_0((const block_q8_0 *)src, dst, k);
}

void ggml_quants_dequantize_q4_K(
    const void *src, float *dst, int k
) {
    dequantize_row_q4_K((const block_q4_K *)src, dst, k);
}
