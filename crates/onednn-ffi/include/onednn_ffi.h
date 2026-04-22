#ifndef ONEDNN_FFI_H
#define ONEDNN_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Initialize oneDNN engine and stream (with rayon-backed threadpool).
/// Must be called once before any other function.
void onednn_init(void);

/// Cleanup oneDNN resources (optional, called at program exit).
void onednn_cleanup(void);

/// Set the number of threads (no-op with THREADPOOL runtime, kept for API compat).
void onednn_set_num_threads(int num_threads);

/// Bind threads to CPU cores (no-op with THREADPOOL runtime, kept for API compat).
void onednn_bind_threads(const int* cpu_ids, int num_threads);

/// Opaque handle for pre-packed weights.
typedef struct onednn_packed_weights* onednn_packed_weights_t;

/// Destroy packed weights and free associated memory.
void onednn_packed_weights_destroy(onednn_packed_weights_t pw);

// ── BF16 API ────────────────────────────────────────────────────────────

/// BF16 linear: output[M,N] = input[M,K] * weight[N,K]^T
void onednn_bf16_linear(
    const void* input, const void* weight, void* output,
    int64_t m, int64_t k, int64_t n);

/// BF16 matmul (no transpose): C[M,N] = A[M,K] * B[K,N]
void onednn_bf16_matmul(
    const void* a, const void* b, void* c,
    int64_t m, int64_t k, int64_t n);

/// Pack BF16 weight [N,K] into oneDNN's optimal blocked format.
onednn_packed_weights_t onednn_bf16_pack_weights(
    const void* weight, int64_t k, int64_t n, int64_t ref_m);

/// BF16 linear with pre-packed weights.
void onednn_bf16_linear_packed(
    const void* input, onednn_packed_weights_t packed_weight,
    void* output, int64_t m);

// ── F32 API ─────────────────────────────────────────────────────────────

/// F32 linear: output[M,N] = input[M,K] * weight[N,K]^T
void onednn_f32_linear(
    const void* input, const void* weight, void* output,
    int64_t m, int64_t k, int64_t n);

/// F32 matmul (no transpose): C[M,N] = A[M,K] * B[K,N]
void onednn_f32_matmul(
    const void* a, const void* b, void* c,
    int64_t m, int64_t k, int64_t n);

/// Pack F32 weight [N,K] into oneDNN's optimal blocked format.
onednn_packed_weights_t onednn_f32_pack_weights(
    const void* weight, int64_t k, int64_t n, int64_t ref_m);

/// F32 linear with pre-packed weights.
void onednn_f32_linear_packed(
    const void* input, onednn_packed_weights_t packed_weight,
    void* output, int64_t m);

// ── AMX BF16 GEMM API ───────────────────────────────────────────────────

/// Check if AMX-BF16 is available at runtime (returns 1 if yes, 0 if no).
int amx_bf16_available(void);

/// Pack BF16 weight [N,K] row-major into AMX VNNI tile format.
/// Returns packed buffer (caller must free with amx_bf16_free_packed).
/// Sets *packed_size to the buffer size in bytes. Returns NULL if AMX unavailable.
void* amx_bf16_pack_weights(
    const void* weight, int64_t k, int64_t n, int64_t* packed_size);

/// Free AMX packed weight buffer.
void amx_bf16_free_packed(void* packed);

/// BF16 GEMM with AMX-packed weights: output[M,N] = input[M,K] × packed^T
/// Computes columns [n_start, n_end) only. Caller parallelizes over N ranges.
void amx_bf16_gemm_packed(
    const void* input, const void* packed_weight, void* output,
    int64_t m, int64_t k, int64_t n,
    int64_t n_start, int64_t n_end);

// ── BRGeMM micro-kernel API ─────────────────────────────────────────────
// Uses oneDNN's brgemm ukernel for near-zero-overhead BF16 GEMM.
// Same approach as SGLang: VNNI-packed weights + JIT'd micro-kernels.

/// Opaque handle for VNNI-packed weights (for brgemm).
typedef struct brgemm_packed_b* brgemm_packed_b_t;

/// Returns 1 if brgemm ukernel API is available, 0 otherwise.
int brgemm_available(void);

/// Pack BF16 weight [N,K] row-major into VNNI block format for brgemm.
/// Returns NULL if brgemm not available.
brgemm_packed_b_t brgemm_bf16_pack(const void* weight, int64_t k, int64_t n);

/// Free brgemm packed weight.
void brgemm_bf16_pack_destroy(brgemm_packed_b_t pw);

/// BF16 linear using brgemm micro-kernels.
/// Computes output[M, n_start:n_end] = input[M,K] × packed_weight^T.
/// output has full N stride (n_total columns per row).
/// Each thread should call this for its N-range; uses thread-local JIT cache.
void brgemm_bf16_linear(
    const void* input,
    brgemm_packed_b_t pw,
    void* output,
    int64_t m,
    int64_t n_total,
    int64_t n_start, int64_t n_end);

/// Fused gate_up GEMM + SiLU×Mul using brgemm.
///
/// Weight is [2*dim, K] packed for gate||up concatenation.
/// For column range [n_start, n_end) (where n_start < dim):
///   1. Compute gate[M, n_start:n_end] = input × packed[n_start:n_end]^T  (F32)
///   2. Compute up[M, n_start:n_end]   = input × packed[n_start+dim:n_end+dim]^T  (F32)
///   3. output[r, n_start+c] = SiLU(gate[r,c]) * up[r,c]  (BF16)
///
/// Output is [M, dim] (half the size of unfused gate_up output).
/// n_start/n_end refer to columns in the FIRST half (gate), must be < dim.
void brgemm_bf16_linear_fused_silu_mul(
    const void* input,
    brgemm_packed_b_t pw,
    void* output,
    int64_t m,
    int64_t dim,
    int64_t n_start, int64_t n_end);

/// Score @ V accumulation for attention: C_f32 += scores_bf16 @ V_vnni
///
/// Converts F32 scores to BF16, packs V to VNNI format on-the-fly,
/// and calls brgemm with add_C=1 (beta=1 accumulation).
///
/// scores_f32: [m × lda] F32 softmax weights
/// V_bf16:     [k × n] BF16 V matrix (row-major)
/// C_f32:      [m × n] F32 accumulator (add in-place)
void brgemm_score_v_accum(
    const float* scores_f32,
    const uint16_t* V_bf16,
    float* C_f32,
    int64_t m, int64_t k, int64_t n, int64_t lda, int64_t v_stride);

/// Release AMX/brgemm HW context after attention N-block loop.
/// Call once per M-block (after all brgemm_qk_gemm/brgemm_score_v_accum calls).
void brgemm_attn_release(void);

/// QK^T GEMM for attention: scores = Q @ K^T * sm_scale
///
/// Q_bf16:      [m rows × head_dim] BF16, row stride = q_stride
/// K_bf16:      [n × head_dim] BF16, contiguous
/// scores_f32:  [m × ldc] F32 output (overwrite)
void brgemm_qk_gemm(
    const uint16_t* Q_bf16,
    const uint16_t* K_bf16,
    float* scores_f32,
    int64_t m, int64_t n, int64_t head_dim,
    int64_t q_stride, int64_t k_stride, int64_t ldc, float sm_scale);

// ── INT8 BRGeMM API (W8A8 quantization) ────────────────────────────────
// INT8 weights (per-channel scales) × INT8 activations (per-tensor dynamic
// quantization). Uses AVX-512 VNNI for INT8 GEMM.

/// Opaque handle for INT8 VNNI-packed weights with per-channel scales.
typedef struct brgemm_s8_packed_b* brgemm_s8_packed_b_t;

/// Returns 1 if INT8 brgemm is available on this CPU, 0 otherwise.
int brgemm_s8_available(void);

/// Pack INT8 weight [N,K] row-major into VNNI format for INT8 brgemm.
/// scales: float[N] per-channel weight quantization scales.
brgemm_s8_packed_b_t brgemm_s8_pack(
    const int8_t* weight, const float* scales, int64_t k, int64_t n);

/// Free INT8 packed weight.
void brgemm_s8_pack_destroy(brgemm_s8_packed_b_t pw);

/// Dynamic per-tensor quantization: BF16 [M,K] → INT8 [M,K].
/// Returns the activation scale factor. Writes quantized values to out_s8.
float brgemm_quantize_bf16_s8(
    const void* input_bf16, int8_t* out_s8,
    int64_t m, int64_t k);

/// INT8 W8A8 linear with pre-quantized INT8 input.
/// output[M, n_start:n_end] = dequant(input_s8 × packed_weight) → BF16.
/// a_scale: per-tensor activation scale from brgemm_quantize_bf16_s8.
void brgemm_s8_linear(
    const int8_t* input_s8, float a_scale,
    brgemm_s8_packed_b_t pw,
    void* output_bf16,
    int64_t m, int64_t n_total,
    int64_t n_start, int64_t n_end);

// ── FP8 BRGeMM API ────────────────────────────────────────────────────
// FP8 weights (per-channel scales) × FP8 activations (per-tensor).
// Requires AVX10.2 with AMX-2 (Intel Granite Rapids+). Code compiles on
// all platforms but returns 0/NULL at runtime if HW is unavailable.

/// Opaque handle for FP8 VNNI-packed weights with per-channel scales.
typedef struct brgemm_f8_packed_b* brgemm_f8_packed_b_t;

/// Returns 1 if FP8 (f8_e4m3) brgemm is available, 0 otherwise.
int brgemm_f8_available(void);

/// Pack FP8-E4M3 weight [N,K] into VNNI format with per-channel scales.
brgemm_f8_packed_b_t brgemm_f8e4m3_pack(
    const void* weight_f8, const float* scales, int64_t k, int64_t n);

/// Free FP8 packed weight.
void brgemm_f8_pack_destroy(brgemm_f8_packed_b_t pw);

/// Dynamic quantization: BF16 [M,K] → FP8-E4M3 [M,K].
/// Returns the activation scale factor.
float brgemm_quantize_bf16_f8e4m3(
    const void* input_bf16, void* out_f8,
    int64_t m, int64_t k);

/// FP8 linear: output[M, n_start:n_end] = dequant(input_f8 × packed) → BF16.
void brgemm_f8e4m3_linear(
    const void* input_f8, float a_scale,
    brgemm_f8_packed_b_t pw,
    void* output_bf16,
    int64_t m, int64_t n_total,
    int64_t n_start, int64_t n_end);

// ── BRGeMM post-ops flags ──────────────────────────────────────────────
// Fuse bias add, GELU, ReLU etc. into brgemm output path.
// Eliminates separate memory passes for these operations.

#define BRGEMM_POSTOP_BIAS      1
#define BRGEMM_POSTOP_GELU_TANH 2
#define BRGEMM_POSTOP_GELU_ERF  4
#define BRGEMM_POSTOP_RELU      8

/// BF16 linear with fused post-ops (F32→BF16 + optional bias/GELU/ReLU).
/// bias_bf16: pointer to BF16 bias[N], or NULL if BRGEMM_POSTOP_BIAS not set.
void brgemm_bf16_linear_postops(
    const void* input, brgemm_packed_b_t pw,
    void* output,
    const void* bias_bf16,
    int postop_flags,
    int64_t m, int64_t n_total,
    int64_t n_start, int64_t n_end);

#ifdef __cplusplus
}
#endif

#endif // ONEDNN_FFI_H
