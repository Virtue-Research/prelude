// Stub when GGML source is not available — all functions are no-ops.
// This allows the crate to compile without ggml source (will panic at runtime).

#include <stdio.h>
#include <stdlib.h>

static void panic_no_ggml(const char *fn) {
    fprintf(stderr, "FATAL: %s called but ggml kernels not compiled. Set GGML_SRC.\n", fn);
    abort();
}

void ggml_quants_matvec_q8_0(const void *w, const float *i, float *o, int m, int k)
    { panic_no_ggml(__func__); }
void ggml_quants_matvec_q4_K(const void *w, const float *i, float *o, int m, int k)
    { panic_no_ggml(__func__); }
void ggml_quants_matmul_q8_0(const void *w, const float *i, float *o, int m, int k, int n)
    { panic_no_ggml(__func__); }
void ggml_quants_matmul_q4_K(const void *w, const float *i, float *o, int m, int k, int n)
    { panic_no_ggml(__func__); }
void ggml_quants_dequantize_q8_0(const void *s, float *d, int k)
    { panic_no_ggml(__func__); }
void ggml_quants_dequantize_q4_K(const void *s, float *d, int k)
    { panic_no_ggml(__func__); }
