// Minimal ggml.h stub for vendored llama.cpp MMQ kernels.
// Only provides types/enums actually used by the CUDA kernel code.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── ggml_type enum (quantization format IDs) ────────────────────────────

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_IQ1_M   = 31,
    GGML_TYPE_MXFP4   = 33,
    GGML_TYPE_NVFP4   = 34,
    GGML_TYPE_COUNT,
};

// ── ggml_op enum (minimal, only what common.cuh references) ─────────────

enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_MUL_MAT = 18,
    GGML_OP_MUL_MAT_ID = 19,
    GGML_OP_COUNT,
};

// ── ggml_glu_op enum ────────────────────────────────────────────────────

enum ggml_glu_op {
    GGML_GLU_OP_NONE = 0,
    GGML_GLU_OP_SWIGLU,
    GGML_GLU_OP_GEGLU,
    GGML_GLU_OP_REGLU,
    GGML_GLU_OP_COUNT,
};

// ── Tensor struct (minimal stub, only fields referenced by kernel code) ─

#define GGML_MAX_DIMS    4
#define GGML_MAX_SRC     10
#define GGML_MAX_OP_PARAMS 64

struct ggml_tensor {
    enum ggml_type type;
    int64_t  ne[GGML_MAX_DIMS];
    size_t   nb[GGML_MAX_DIMS];
    enum ggml_op op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    struct ggml_tensor * src[GGML_MAX_SRC];
    struct ggml_tensor * view_src;
    void * data;
    void * extra;
    char name[64];
    char _pad[128];
};

static inline size_t ggml_nbytes(const struct ggml_tensor * t) {
    // Simplified: just ne[0]*nb[0] for the common case
    size_t nbytes = t->nb[0];
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] <= 0) break;
        nbytes = t->ne[i] * t->nb[i];
    }
    return nbytes;
}

static inline size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    (void)type; (void)ne;
    return 0; // stub
}

static inline int64_t ggml_nrows(const struct ggml_tensor * t) {
    int64_t r = 1;
    for (int i = 1; i < GGML_MAX_DIMS; i++) r *= t->ne[i];
    return r;
}

static inline bool ggml_is_contiguous(const struct ggml_tensor * t) {
    (void)t;
    return true; // stub
}

static inline size_t ggml_type_size(enum ggml_type type) {
    (void)type;
    return 0; // stub
}

static inline int64_t ggml_blck_size(enum ggml_type type) {
    (void)type;
    return 1; // stub
}

#ifdef __cplusplus
}
#endif
