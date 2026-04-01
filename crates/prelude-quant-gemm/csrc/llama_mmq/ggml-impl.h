// Minimal ggml-impl.h stub for vendored llama.cpp MMQ kernels.
#pragma once

#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define GGML_UNUSED(x)      (void)(x)
#define GGML_UNUSED_VARS(...) (void)(__VA_ARGS__)
#define GGML_ASSERT(x)      assert(x)
#define GGML_ABORT(msg)      do { fprintf(stderr, "GGML_ABORT: %s\n", msg); abort(); } while(0)
#define GGML_PAD(x, n)       (((x) + (n) - 1) & ~((n) - 1))
#define GGML_LOG_DEBUG(...)  ((void)0)

// Tensor binary op locals macro (expands dimension variables from tensor pointers).
// Used by host-side launch code in common.cuh.
#define GGML_TENSOR_BINARY_OP_LOCALS \
    const int64_t ne00 = src0->ne[0]; const int64_t ne01 = src0->ne[1]; \
    const int64_t ne02 = src0->ne[2]; const int64_t ne03 = src0->ne[3]; \
    const size_t  nb00 = src0->nb[0]; const size_t  nb01 = src0->nb[1]; \
    const size_t  nb02 = src0->nb[2]; const size_t  nb03 = src0->nb[3]; \
    const int64_t ne10 = src1->ne[0]; const int64_t ne11 = src1->ne[1]; \
    const int64_t ne12 = src1->ne[2]; const int64_t ne13 = src1->ne[3]; \
    const size_t  nb10 = src1->nb[0]; const size_t  nb11 = src1->nb[1]; \
    const size_t  nb12 = src1->nb[2]; const size_t  nb13 = src1->nb[3]; \
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03); \
    GGML_UNUSED(nb00); GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03); \
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13); \
    GGML_UNUSED(nb10); GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13)
