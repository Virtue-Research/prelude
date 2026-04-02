// Dequantization CUDA kernels for GGUF quantized formats.
//
// Each kernel converts a block of quantized weights to BF16 or FP16 for use
// with standard GEMM. Ported from llama.cpp/ggml-cuda/dequantize.cuh.
//
// Supported formats: Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ── Block structures (matching GGUF on-disk format) ──────────────────────

#define QK4_0  32
#define QK8_0  32
#define QK_K   256
#define K_SCALE_SIZE 12

struct block_q4_0 {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // 32 × 4-bit values
};

struct block_q4_1 {
    uint16_t d;        // FP16 scale
    uint16_t m;        // FP16 minimum
    uint8_t qs[16];    // 32 × 4-bit values
};

struct block_q5_0 {
    uint16_t d;        // FP16 scale
    uint8_t qh[4];     // high bits (1 per element)
    uint8_t qs[16];    // 32 × 4-bit low values
};

struct block_q5_1 {
    uint16_t d;        // FP16 scale
    uint16_t m;        // FP16 minimum
    uint8_t qh[4];     // high bits (1 per element)
    uint8_t qs[16];    // 32 × 4-bit low values
};

struct block_q8_0 {
    uint16_t d;        // FP16 scale
    int8_t qs[32];     // 32 × 8-bit values
};

struct block_q2_K {
    uint8_t scales[16]; // 4-bit scales (low) + mins (high)
    uint8_t qs[64];     // 256 × 2-bit values
    uint16_t d;         // FP16 super-block scale
    uint16_t dmin;      // FP16 super-block minimum
};

struct block_q3_K {
    uint8_t hmask[32];  // high bits
    uint8_t qs[64];     // low 2 bits
    uint8_t scales[12]; // 6-bit signed scales
    uint16_t d;         // FP16 super-block scale
};

struct block_q4_K {
    uint16_t d;         // FP16 super-block scale
    uint16_t dmin;      // FP16 super-block minimum
    uint8_t scales[12]; // 6-bit packed scales/mins
    uint8_t qs[128];    // 256 × 4-bit values
};

struct block_q5_K {
    uint16_t d;         // FP16 super-block scale
    uint16_t dmin;      // FP16 super-block minimum
    uint8_t scales[12]; // 6-bit packed scales/mins
    uint8_t qh[32];     // high bits
    uint8_t qs[128];    // low 4 bits
};

struct block_q6_K {
    uint8_t ql[128];    // lower 4 bits
    uint8_t qh[64];     // upper 2 bits
    int8_t scales[16];  // signed 8-bit scales
    uint16_t d;         // FP16 super-block scale
};

// ── Helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// 6-bit scale/min unpacking for Q4_K / Q5_K
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// ── Q4_0 dequantize (32 values/block → BF16) ────────────────────────────

static __global__ void dequantize_q4_0_bf16(
    const block_q4_0* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK4_0;
    const uint32_t within_block = i % QK4_0;

    const block_q4_0& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    // qs[j] stores: low nibble = element j (0..15), high nibble = element j+16 (16..31)
    const uint32_t nibble = (within_block < 16)
        ? (blk.qs[within_block] & 0xF)
        : (blk.qs[within_block - 16] >> 4);

    y[i] = __float2bfloat16(d * ((float)nibble - 8.0f));
}

// ── Q8_0 dequantize (32 values/block → BF16) ────────────────────────────

static __global__ void dequantize_q8_0_bf16(
    const block_q8_0* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK8_0;
    const uint32_t within_block = i % QK8_0;

    const block_q8_0& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    y[i] = __float2bfloat16(d * (float)blk.qs[within_block]);
}

// ── Q4_1 dequantize (32 values/block → BF16) ────────────────────────────

static __global__ void dequantize_q4_1_bf16(
    const block_q4_1* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK4_0;
    const uint32_t within_block = i % QK4_0;

    const block_q4_1& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);
    const float m = fp16_to_f32(blk.m);

    const uint32_t nibble = (within_block < 16)
        ? (blk.qs[within_block] & 0xF)
        : (blk.qs[within_block - 16] >> 4);

    y[i] = __float2bfloat16(d * (float)nibble + m);
}

// ── Q5_0 dequantize (32 values/block → BF16) ────────────────────────────

static __global__ void dequantize_q5_0_bf16(
    const block_q5_0* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / 32;
    const uint32_t within_block = i % 32;

    const block_q5_0& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    uint32_t qh;
    memcpy(&qh, blk.qh, sizeof(qh));

    uint32_t nibble;
    uint32_t high_bit;
    if (within_block < 16) {
        nibble = blk.qs[within_block] & 0xF;
        high_bit = ((qh >> within_block) & 1) << 4;
    } else {
        nibble = blk.qs[within_block - 16] >> 4;
        high_bit = ((qh >> within_block) & 1) << 4;
    }

    y[i] = __float2bfloat16(d * ((float)(nibble | high_bit) - 16.0f));
}

// ── Q5_1 dequantize (32 values/block → BF16) ────────────────────────────

static __global__ void dequantize_q5_1_bf16(
    const block_q5_1* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / 32;
    const uint32_t within_block = i % 32;

    const block_q5_1& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);
    const float m = fp16_to_f32(blk.m);

    uint32_t qh;
    memcpy(&qh, blk.qh, sizeof(qh));

    uint32_t nibble;
    uint32_t high_bit;
    if (within_block < 16) {
        nibble = blk.qs[within_block] & 0xF;
        high_bit = ((qh >> within_block) & 1) << 4;
    } else {
        nibble = blk.qs[within_block - 16] >> 4;
        high_bit = ((qh >> within_block) & 1) << 4;
    }

    y[i] = __float2bfloat16(d * (float)(nibble | high_bit) + m);
}

// ── Q2_K dequantize (256 values/block → BF16) ───────────────────────────

static __global__ void dequantize_q2_K_bf16(
    const block_q2_K* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;

    const block_q2_K& blk = x[block_idx];
    const float dall = fp16_to_f32(blk.d);
    const float dmin = fp16_to_f32(blk.dmin);

    // Q2_K layout: 256 values stored as 2 bits each in qs[64].
    // qs is organized as 2 groups of 32 bytes (128 values each).
    // Within each 32-byte group, each byte stores 4 values at shifts 0,2,4,6.
    // The 4 shift positions correspond to 4 sub-groups of 32 elements each (128 total).
    //
    // For element within_block:
    //   group128 = within_block / 128  (which 32-byte group)
    //   sub32    = (within_block % 128) / 32  (shift = sub32 * 2)
    //   within32 = within_block % 32  (byte index within 32-byte group, but only 0..15 or 16..31)
    // But the scale uses 16-element sub-blocks:
    //   sub16 = within_block / 16  (0..15)

    const uint32_t sub16 = within_block / 16;
    const float d = dall * (float)(blk.scales[sub16] & 0xF);
    const float m = dmin * (float)(blk.scales[sub16] >> 4);

    // Extract 2-bit value
    const uint32_t group128 = within_block / 128;
    const uint32_t within128 = within_block % 128;
    const uint32_t sub32 = within128 / 32;
    const uint32_t within32 = within128 % 32;
    // qs byte layout: within each 32-byte group, byte [within32] packs 4 values at shifts 0,2,4,6
    // However within32 can be 0..31, and each byte stores 2-bit values for positions
    // at offsets 0, 32, 64, 96 within the 128-element group.
    // Actually: the scalar code iterates q8 in chunks of 32, and q2 in chunks of 32 bytes.
    // For each 32-byte q2 chunk, shift goes 0,2,4,6 and q8 advances by 32 each.
    // So byte_in_group = within32 (when within32 < 16) or within32 (when within32 >= 16... wait)
    //
    // Looking at scalar: q2[l] is indexed l=0..31 combined with shift, where
    // the 128-element group uses q2[0..31] with shifts 0,2,4,6 (each shift handles 32 q8 elements).
    // within_block maps to: group128 * 128 + sub32 * 32 + within32
    // q2 byte = group128 * 32 + within32
    // shift = sub32 * 2
    const uint32_t q2_byte = group128 * 32 + within32;
    const uint32_t shift = sub32 * 2;
    const uint32_t q2_val = (blk.qs[q2_byte] >> shift) & 3;

    y[i] = __float2bfloat16(d * (float)q2_val - m);
}

// ── Q3_K dequantize (256 values/block → BF16) ───────────────────────────

static __global__ void dequantize_q3_K_bf16(
    const block_q3_K* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;

    const block_q3_K& blk = x[block_idx];
    const float d_all = fp16_to_f32(blk.d);

    // Unpack 6-bit signed scale for this sub-block
    const uint32_t sub_block = within_block / 16;

    // Scale unpacking (same as CPU — 6-bit from 12-byte packed format)
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t auxs[4];
    memcpy(auxs, blk.scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);
    const float dl = d_all * (float)(scales[sub_block] - 32);

    // Extract 3-bit value: 2 low bits from qs, 1 high bit from hmask
    const uint32_t group128 = within_block / 128;
    const uint32_t within128 = within_block % 128;
    const uint32_t sub32 = within128 / 32;
    const uint32_t within32 = within128 % 32;

    const uint32_t qs_byte = group128 * 32 + within32;
    const uint32_t q3_low = (blk.qs[qs_byte] >> (sub32 * 2)) & 3;

    const uint32_t hmask_bit = group128 * 4 + sub32;
    const uint32_t q3_high = (blk.hmask[within32] >> hmask_bit) & 1;

    const int32_t q3_val = (int32_t)q3_low - (q3_high ? 0 : 4);

    y[i] = __float2bfloat16(dl * (float)q3_val);
}

// ── Q4_K dequantize (256 values/block → BF16) ───────────────────────────

static __global__ void dequantize_q4_K_bf16(
    const block_q4_K* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;

    const block_q4_K& blk = x[block_idx];
    const float dall = fp16_to_f32(blk.d);
    const float dmin = fp16_to_f32(blk.dmin);

    // Which 32-element sub-block?
    const uint32_t sub_block = within_block / 32;
    uint8_t sc, m;
    get_scale_min_k4(sub_block, blk.scales, sc, m);

    const float d = dall * (float)sc;
    const float min_val = dmin * (float)m;

    // Extract 4-bit value
    const uint32_t group64 = within_block / 64;
    const uint32_t within64 = within_block % 64;
    const uint32_t is_high = within64 >= 32 ? 1 : 0;
    const uint32_t qs_idx = group64 * 32 + (within64 % 32);
    const uint32_t q4_val = is_high ? (blk.qs[qs_idx] >> 4) : (blk.qs[qs_idx] & 0xF);

    y[i] = __float2bfloat16(d * (float)q4_val - min_val);
}

// ── Q5_K dequantize (256 values/block → BF16) ───────────────────────────

static __global__ void dequantize_q5_K_bf16(
    const block_q5_K* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;

    const block_q5_K& blk = x[block_idx];
    const float dall = fp16_to_f32(blk.d);
    const float dmin = fp16_to_f32(blk.dmin);

    // Which 32-element sub-block?
    const uint32_t sub_block = within_block / 32;
    uint8_t sc, m;
    get_scale_min_k4(sub_block, blk.scales, sc, m);

    const float d = dall * (float)sc;
    const float min_val = dmin * (float)m;

    // Extract 5-bit value: 4 low bits from qs + 1 high bit from qh
    const uint32_t group64 = within_block / 64;
    const uint32_t within64 = within_block % 64;
    const uint32_t is_high = within64 >= 32 ? 1 : 0;
    const uint32_t within32 = within64 % 32;
    const uint32_t qs_idx = group64 * 32 + within32;
    const uint32_t q5_low = is_high ? (blk.qs[qs_idx] >> 4) : (blk.qs[qs_idx] & 0xF);

    // High bit: bit (group64*2 + is_high) of qh[within32]
    const uint32_t qh_bit = (group64 * 2 + is_high);
    const uint32_t q5_high = (blk.qh[within32] >> qh_bit) & 1;

    y[i] = __float2bfloat16(d * (float)(q5_low + q5_high * 16) - min_val);
}

// ── IQ4_NL block structure and lookup table ─────────────────────────────

#define QK4_NL 32

struct block_iq4_nl {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // 32 × 4-bit non-linear indices
};

struct block_iq4_xs {
    uint16_t d;        // FP16 super-block scale
    uint16_t scales_h; // upper 2 bits of 8 sub-block scales
    uint8_t scales_l[4]; // lower 4 bits of 8 sub-block scales
    uint8_t qs[128];   // 256 × 4-bit non-linear indices
};

__device__ static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ── IQ4_NL dequantize (32 values/block → BF16) ─────────────────────────

static __global__ void dequantize_iq4_nl_bf16(
    const block_iq4_nl* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK4_NL;
    const uint32_t within_block = i % QK4_NL;

    const block_iq4_nl& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    const uint32_t nibble = (within_block < 16)
        ? (blk.qs[within_block] & 0xF)
        : (blk.qs[within_block - 16] >> 4);

    y[i] = __float2bfloat16(d * (float)kvalues_iq4nl[nibble]);
}

// ── IQ4_XS dequantize (256 values/block → BF16) ────────────────────────

static __global__ void dequantize_iq4_xs_bf16(
    const block_iq4_xs* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;

    const block_iq4_xs& blk = x[block_idx];
    const float d_all = fp16_to_f32(blk.d);

    // Which 32-element sub-block?
    const uint32_t sub = within_block / 32;
    const uint32_t within_sub = within_block % 32;

    // Extract 6-bit scale for this sub-block
    int ls = (blk.scales_l[sub / 2] >> ((sub % 2) * 4)) & 0x0F;
    ls |= ((blk.scales_h >> (sub * 2)) & 0x03) << 4;
    const float dl = d_all * (float)(ls - 32);

    // Extract 4-bit index → lookup in table
    const uint32_t nibble = (within_sub < 16)
        ? (blk.qs[sub * 16 + within_sub] & 0xF)
        : (blk.qs[sub * 16 + within_sub - 16] >> 4);

    y[i] = __float2bfloat16(dl * (float)kvalues_iq4nl[nibble]);
}

// ── Q6_K dequantize (256 values/block → BF16) ───────────────────────────

static __global__ void dequantize_q6_K_bf16(
    const block_q6_K* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;

    const block_q6_K& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    // Which 16-element sub-block?
    const uint32_t sub_block = within_block / 16;
    const float dl = d * (float)blk.scales[sub_block];

    // Extract 6-bit value: 4 low bits from ql + 2 high bits from qh
    // Layout (from scalar reference, per 128-element group):
    //   sub32=0: ql[base + l]      low nibble,  qh[base + l] >> 0
    //   sub32=1: ql[base + l + 32] low nibble,  qh[base + l] >> 2
    //   sub32=2: ql[base + l]      high nibble, qh[base + l] >> 4
    //   sub32=3: ql[base + l + 32] high nibble, qh[base + l] >> 6
    const uint32_t group128 = within_block / 128;
    const uint32_t within128 = within_block % 128;
    const uint32_t sub32 = within128 / 32;
    const uint32_t within32 = within128 % 32;

    const uint32_t ql_idx = group128 * 64 + (sub32 & 1) * 32 + within32;
    const uint32_t ql_val = (sub32 < 2)
        ? (blk.ql[ql_idx] & 0xF)
        : (blk.ql[ql_idx] >> 4);

    const uint32_t qh_idx = group128 * 32 + within32;
    const uint32_t qh_shift = sub32 * 2;
    const uint32_t qh_val = (blk.qh[qh_idx] >> qh_shift) & 3;

    const int32_t q6_val = (int32_t)(ql_val | (qh_val << 4)) - 32;

    y[i] = __float2bfloat16(dl * (float)q6_val);
}

// ── Remaining IQ formats: codebook-based dequantize ─────────────────────

#include "iq_tables.cuh"

// IQ block structures (matching GGUF on-disk layout)

struct block_iq3_xxs { uint16_t d; uint8_t qs[96]; };
struct block_iq3_s   { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; };
struct block_iq2_xxs { uint16_t d; uint16_t qs[32]; };
struct block_iq2_xs  { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; };
struct block_iq2_s   { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; };
struct block_iq1_s   { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
struct block_iq1_m   { uint8_t qs[32]; uint8_t qh[16]; uint8_t scales[8]; };

// ── IQ3_XXS dequantize (256 values/block → BF16) ───────────────────────
// Ref: llama.cpp ggml-quants.c dequantize_row_iq3_xxs
// Layout: qs[0..64] = 8 groups × 8 index bytes, qs[64..96] = 8 groups × 4 aux bytes

static __global__ void dequantize_iq3_xxs_bf16(
    const block_iq3_xxs* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq3_xxs& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    const uint32_t ib32 = within_block / 32;      // group of 32
    const uint32_t within32 = within_block % 32;
    const uint32_t l = within32 / 8;               // sub-group of 8 (0..3)
    const uint32_t j = within32 % 8;               // element within sub-group

    // Index bytes: qs[ib32*8 + 2*l + j/4]
    const uint8_t* qs = blk.qs + ib32 * 8;
    const uint8_t* scales_and_signs = blk.qs + QK_K / 4 + ib32 * 4;

    uint32_t aux32;
    memcpy(&aux32, scales_and_signs, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
    const uint8_t* grid = (const uint8_t*)&iq3xxs_grid[qs[2 * l + j / 4]];
    float val = (float)grid[j % 4];
    if (signs & kmask_iq2xs[j]) val = -val;

    y[i] = __float2bfloat16(db * val);
}

// ── IQ3_S dequantize (256 values/block → BF16) ─────────────────────────
// Ref: llama.cpp dequantize_row_iq3_s

static __global__ void dequantize_iq3_s_bf16(
    const block_iq3_s* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq3_s& blk = x[block_idx];
    const float d_all = fp16_to_f32(blk.d);

    // ib32 = group of 32, processed in pairs (ib32 and ib32+1 share one qh byte pair)
    const uint32_t ib32 = within_block / 32;
    const uint32_t within32 = within_block % 32;
    const uint32_t l = within32 / 8;    // 0..3 sub-group
    const uint32_t j = within32 % 8;    // element in sub-group

    // Scale: pairs of ib32 share scales[ib32/2]
    const int ls = (ib32 % 2 == 0)
        ? (1 + 2 * (blk.scales[ib32 / 2] & 0xf))
        : (1 + 2 * (blk.scales[ib32 / 2] >> 4));
    const float db = d_all * (float)ls;

    // qs pointer advances by 8 per ib32
    const uint8_t* qs = blk.qs + ib32 * 8;
    const uint8_t* signs = blk.signs + ib32 * 4;
    // qh: each ib32-pair uses 2 bytes; within the pair, ib32%2 selects which byte
    const uint8_t qh_byte = blk.qh[(ib32 / 2) * 2 + (ib32 % 2)];

    // 9-bit codebook index: for j<4 use grid1 pattern, j>=4 use grid2 pattern
    uint32_t idx;
    if (j < 4) {
        idx = qs[2 * l + 0] | ((qh_byte << (8 - 2 * l)) & 256);
    } else {
        idx = qs[2 * l + 1] | ((qh_byte << (7 - 2 * l)) & 256);
    }
    const uint8_t* grid = (const uint8_t*)&iq3s_grid[idx];
    float val = (float)grid[j % 4];

    // Sign from signs array
    if (signs[l] & kmask_iq2xs[j]) val = -val;

    y[i] = __float2bfloat16(db * val);
}

// ── IQ2_XXS dequantize (256 values/block → BF16) ───────────────────────
// Ref: llama.cpp dequantize_row_iq2_xxs
// Layout: qs = uint16_t[32] = 64 bytes; each group of 32 uses 8 bytes (2 uint32_t)

static __global__ void dequantize_iq2_xxs_bf16(
    const block_iq2_xxs* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq2_xxs& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    const uint32_t ib32 = within_block / 32;
    const uint32_t within32 = within_block % 32;
    const uint32_t l = within32 / 8;    // 0..3
    const uint32_t j = within32 % 8;

    // Read 8 bytes for this group: aux32[0] = indices, aux32[1] = signs+scale
    uint32_t aux32[2];
    memcpy(aux32, (const uint8_t*)blk.qs + ib32 * 8, 8);
    const uint8_t* aux8 = (const uint8_t*)aux32;

    const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;

    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
    const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
    float val = (float)grid[j];
    if (signs & kmask_iq2xs[j]) val = -val;

    y[i] = __float2bfloat16(db * val);
}

// ── IQ2_XS dequantize (256 values/block → BF16) ────────────────────────
// Ref: llama.cpp dequantize_row_iq2_xs

static __global__ void dequantize_iq2_xs_bf16(
    const block_iq2_xs* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq2_xs& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    const uint32_t ib32 = within_block / 32;
    const uint32_t within32 = within_block % 32;
    const uint32_t l = within32 / 8;    // 0..3
    const uint32_t j = within32 % 8;

    // Scale: two 4-bit scales per ib32 (low nibble for l<2, high for l>=2)
    float db = (l < 2)
        ? d * (0.5f + (float)(blk.scales[ib32] & 0xf)) * 0.25f
        : d * (0.5f + (float)(blk.scales[ib32] >> 4)) * 0.25f;

    uint16_t qval = blk.qs[4 * ib32 + l];
    const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (qval & 511));
    const uint8_t signs = ksigns_iq2xs[qval >> 9];
    float val = (float)grid[j];
    if (signs & kmask_iq2xs[j]) val = -val;

    y[i] = __float2bfloat16(db * val);
}

// ── IQ2_S dequantize (256 values/block → BF16) ─────────────────────────
// Ref: llama.cpp dequantize_row_iq2_s

static __global__ void dequantize_iq2_s_bf16(
    const block_iq2_s* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq2_s& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    const uint32_t ib32 = within_block / 32;
    const uint32_t within32 = within_block % 32;
    const uint32_t l = within32 / 8;    // 0..3
    const uint32_t j = within32 % 8;

    const uint8_t* qs = blk.qs + ib32 * 4;
    const uint8_t* signs_arr = blk.qs + QK_K / 8 + ib32 * 4;

    float db = (l < 2)
        ? d * (0.5f + (float)(blk.scales[ib32] & 0xf)) * 0.25f
        : d * (0.5f + (float)(blk.scales[ib32] >> 4)) * 0.25f;

    // 10-bit grid index
    const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | ((blk.qh[ib32] << (8 - 2 * l)) & 0x300)));
    float val = (float)grid[j];
    if (signs_arr[l] & kmask_iq2xs[j]) val = -val;

    y[i] = __float2bfloat16(db * val);
}

// ── IQ1_S dequantize (256 values/block → BF16) ─────────────────────────
// Ref: llama.cpp dequantize_row_iq1_s (uses iq1s_grid with int8_t values)

static __global__ void dequantize_iq1_s_bf16(
    const block_iq1_s* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq1_s& blk = x[block_idx];
    const float d = fp16_to_f32(blk.d);

    const uint32_t ib = within_block / 32;     // group of 32
    const uint32_t l = (within_block % 32) / 8; // sub-group (0..3)
    const uint32_t j = within_block % 8;

    uint16_t qh_val = blk.qh[ib];
    const float dl = d * (float)(2 * ((qh_val >> 12) & 7) + 1);
    const float delta = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

    int grid_idx = blk.qs[ib * 4 + l] | (((qh_val >> (3 * l)) & 7) << 8);
    const int8_t* grid = (const int8_t*)(iq1s_grid + grid_idx);
    y[i] = __float2bfloat16(dl * ((float)grid[j] + delta));
}

// ── IQ1_M dequantize (256 values/block → BF16) ─────────────────────────
// Ref: llama.cpp dequantize_row_iq1_m

static __global__ void dequantize_iq1_m_bf16(
    const block_iq1_m* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_K;
    const uint32_t within_block = i % QK_K;
    const block_iq1_m& blk = x[block_idx];

    const uint16_t* sc = (const uint16_t*)blk.scales;
    uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                       | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    float d = __half2float(*reinterpret_cast<const __half*>(&scale_u16));

    const uint32_t ib = within_block / 32;
    const uint32_t sub4 = (within_block % 32) / 8;   // 0..3
    const uint32_t j = within_block % 8;

    // Per-16-element scale (sub4 0,1 share dl1; sub4 2,3 share dl2)
    float dl = (sub4 < 2)
        ? d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1)
        : d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);

    // Grid index from qs + qh
    const uint8_t* qs = blk.qs + ib * 4;
    const uint8_t* qh = blk.qh + ib * 2;

    uint16_t idx;
    float delta;
    if (sub4 == 0) {
        idx = qs[0] | ((qh[0] << 8) & 0x700);
        delta = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
    } else if (sub4 == 1) {
        idx = qs[1] | ((qh[0] << 4) & 0x700);
        delta = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
    } else if (sub4 == 2) {
        idx = qs[2] | ((qh[1] << 8) & 0x700);
        delta = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
    } else {
        idx = qs[3] | ((qh[1] << 4) & 0x700);
        delta = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
    }

    const int8_t* grid = (const int8_t*)(iq1s_grid + idx);
    y[i] = __float2bfloat16(dl * ((float)grid[j] + delta));
}

// ── FP4 formats ─────────────────────────────────────────────────────────

#define QK_MXFP4 32
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16

__device__ static const int8_t kvalues_mxfp4_deq[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

struct block_mxfp4_deq { uint8_t e; uint8_t qs[16]; };
struct block_nvfp4_deq { uint8_t d[4]; uint8_t qs[32]; };

__device__ __forceinline__ float e8m0_to_f32_deq(uint8_t x) {
    if (x == 0) { uint32_t bits = 0x00400000u; float r; memcpy(&r, &bits, 4); return r; }
    uint32_t bits = (uint32_t)x << 23;
    float r; memcpy(&r, &bits, 4); return r;
}

__device__ __forceinline__ float ue4m3_to_f32_deq(uint8_t x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    int e = (x >> 3) & 0xF;
    int m = x & 0x7;
    float raw = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return raw * 0.5f;
}

// ── MXFP4 dequantize (32 values/block → BF16) ──────────────────────────

static __global__ void dequantize_mxfp4_bf16(
    const block_mxfp4_deq* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_MXFP4;
    const uint32_t within_block = i % QK_MXFP4;

    const block_mxfp4_deq& blk = x[block_idx];
    const float d = e8m0_to_f32_deq(blk.e) * 0.5f;

    const uint32_t nibble = (within_block < 16)
        ? (blk.qs[within_block] & 0xF)
        : (blk.qs[within_block - 16] >> 4);

    y[i] = __float2bfloat16(d * (float)kvalues_mxfp4_deq[nibble]);
}

// ── NVFP4 dequantize (64 values/block → BF16) ──────────────────────────

static __global__ void dequantize_nvfp4_bf16(
    const block_nvfp4_deq* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint64_t num_elements
) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const uint64_t block_idx = i / QK_NVFP4;
    const uint32_t within_block = i % QK_NVFP4;

    const block_nvfp4_deq& blk = x[block_idx];

    // Which 16-element sub-block?
    const uint32_t sub = within_block / QK_NVFP4_SUB;
    const uint32_t within_sub = within_block % QK_NVFP4_SUB;
    const float d = ue4m3_to_f32_deq(blk.d[sub]);

    const uint32_t nibble = (within_sub < 8)
        ? (blk.qs[sub * 8 + within_sub] & 0xF)
        : (blk.qs[sub * 8 + within_sub - 8] >> 4);

    y[i] = __float2bfloat16(d * (float)kvalues_mxfp4_deq[nibble]);
}

// ── FFI host dispatch: launch dequantize kernel by type ─────────────────

static void launch_dequant(
    const char* kernel_name,
    const void* input, void* output, uint64_t num_elements, cudaStream_t stream,
    void (*kernel_fn)(const void*, void*, uint64_t, cudaStream_t)
) {
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((num_elements + threads - 1) / threads);
    kernel_fn(input, output, num_elements, stream);
}

// Type-specific launchers (template would be cleaner but extern "C" needs concrete fns)
#define DEFINE_DEQUANT_LAUNCHER(name, block_type) \
static void launch_##name(const void* in, void* out, uint64_t n, cudaStream_t s) { \
    uint32_t threads = 256; \
    uint32_t blocks = (uint32_t)((n + threads - 1) / threads); \
    name<<<blocks, threads, 0, s>>>((const block_type*)in, (__nv_bfloat16*)out, n); \
}

DEFINE_DEQUANT_LAUNCHER(dequantize_q4_0_bf16, block_q4_0)
DEFINE_DEQUANT_LAUNCHER(dequantize_q4_1_bf16, block_q4_1)
DEFINE_DEQUANT_LAUNCHER(dequantize_q5_0_bf16, block_q5_0)
DEFINE_DEQUANT_LAUNCHER(dequantize_q5_1_bf16, block_q5_1)
DEFINE_DEQUANT_LAUNCHER(dequantize_q8_0_bf16, block_q8_0)
DEFINE_DEQUANT_LAUNCHER(dequantize_q2_K_bf16, block_q2_K)
DEFINE_DEQUANT_LAUNCHER(dequantize_q3_K_bf16, block_q3_K)
DEFINE_DEQUANT_LAUNCHER(dequantize_q4_K_bf16, block_q4_K)
DEFINE_DEQUANT_LAUNCHER(dequantize_q5_K_bf16, block_q5_K)
DEFINE_DEQUANT_LAUNCHER(dequantize_q6_K_bf16, block_q6_K)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq4_nl_bf16, block_iq4_nl)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq4_xs_bf16, block_iq4_xs)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq3_xxs_bf16, block_iq3_xxs)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq3_s_bf16, block_iq3_s)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq2_xxs_bf16, block_iq2_xxs)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq2_xs_bf16, block_iq2_xs)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq2_s_bf16, block_iq2_s)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq1_s_bf16, block_iq1_s)
DEFINE_DEQUANT_LAUNCHER(dequantize_iq1_m_bf16, block_iq1_m)
DEFINE_DEQUANT_LAUNCHER(dequantize_mxfp4_bf16, block_mxfp4_deq)
DEFINE_DEQUANT_LAUNCHER(dequantize_nvfp4_bf16, block_nvfp4_deq)

extern "C" void llama_gpu_dequantize(
    const void* input,
    void* output,          // BF16 device buffer
    int64_t num_elements,
    int ggml_type_id,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint64_t n = (uint64_t)num_elements;

    switch (ggml_type_id) {
        case  2: launch_dequantize_q4_0_bf16(input, output, n, stream); break;
        case  3: launch_dequantize_q4_1_bf16(input, output, n, stream); break;
        case  6: launch_dequantize_q5_0_bf16(input, output, n, stream); break;
        case  7: launch_dequantize_q5_1_bf16(input, output, n, stream); break;
        case  8: launch_dequantize_q8_0_bf16(input, output, n, stream); break;
        case 10: launch_dequantize_q2_K_bf16(input, output, n, stream); break;
        case 11: launch_dequantize_q3_K_bf16(input, output, n, stream); break;
        case 12: launch_dequantize_q4_K_bf16(input, output, n, stream); break;
        case 13: launch_dequantize_q5_K_bf16(input, output, n, stream); break;
        case 14: launch_dequantize_q6_K_bf16(input, output, n, stream); break;
        case 16: launch_dequantize_iq2_xxs_bf16(input, output, n, stream); break;
        case 17: launch_dequantize_iq2_xs_bf16(input, output, n, stream); break;
        case 18: launch_dequantize_iq3_xxs_bf16(input, output, n, stream); break;
        case 20: launch_dequantize_iq4_nl_bf16(input, output, n, stream); break;
        case 21: launch_dequantize_iq3_s_bf16(input, output, n, stream); break;
        case 22: launch_dequantize_iq2_s_bf16(input, output, n, stream); break;
        case 23: launch_dequantize_iq4_xs_bf16(input, output, n, stream); break;
        case 19: launch_dequantize_iq1_s_bf16(input, output, n, stream); break;
        case 29: launch_dequantize_iq1_m_bf16(input, output, n, stream); break;
        case 39: launch_dequantize_mxfp4_bf16(input, output, n, stream); break;
        case 40: launch_dequantize_nvfp4_bf16(input, output, n, stream); break;
        default: break;
    }
}
