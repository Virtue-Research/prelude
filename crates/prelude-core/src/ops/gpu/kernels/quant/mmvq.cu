// MMVQ: Fused matrix-vector multiply with quantized weights.
//
// Computes y[N] = W[N,K] @ x[K] where W is GGUF-quantized and x is BF16.
// Critical for decode (M=1): avoids materializing the full dequantized weight matrix,
// saving ~3.6x memory bandwidth vs dequantize + GEMV.
//
// Architecture: one warp per output row, 4 warps per CUDA block.
// Each thread processes complete quantized blocks, reduced via warp shuffle.
// Activations are quantized to Q8_1 on GPU before the fused dot product.
//
// Ported from llama.cpp ggml-cuda/mmvq.cu + vecdotq.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ── Block structures (matching GGUF on-disk layout) ─────────────────────

#define QK4_0  32
#define QK8_0  32
#define QK_K   256
#define WARP_SIZE 32
#define MMVQ_NWARPS 4

// NOTE: Many of these structs have sizes not divisible by 4 (e.g., Q4_0=18,
// Q5_0=22, Q8_0=34, Q3_K=110, Q6_K=210), which means consecutive blocks in
// arrays may not be 4-byte aligned. All multi-byte loads from struct fields
// MUST use memcpy to avoid misaligned access errors.

struct __align__(2) block_q4_0 {  // 18 bytes
    uint16_t d;
    uint8_t qs[16];
};

struct __align__(2) block_q4_1 {  // 20 bytes
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

struct __align__(2) block_q5_0 {  // 22 bytes
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct __align__(2) block_q5_1 {  // 24 bytes
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct __align__(2) block_q8_0 {  // 34 bytes
    uint16_t d;
    int8_t qs[32];
};

struct __align__(4) block_q8_1 {  // 36 bytes
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
};

struct __align__(2) block_q2_K {  // 84 bytes
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

struct __align__(2) block_q3_K {  // 110 bytes
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};

struct __align__(2) block_q4_K {  // 144 bytes
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct __align__(2) block_q5_K {  // 176 bytes
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct __align__(2) block_q6_K {  // 210 bytes
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

// ── Helpers ─────────────────────────────────────────────────────────────

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

/// Alignment-safe 4-byte load. Required because GGUF block structs may not
/// be 4-byte aligned in arrays (e.g., block_q4_0 is 18 bytes).
__device__ __forceinline__ int load_int(const void* ptr) {
    int val;
    memcpy(&val, ptr, sizeof(int));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_mmvq(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& sc, uint8_t& m
) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// ── BF16 → Q8_1 quantization kernel ────────────────────────────────────
//
// Each warp quantizes one block of 32 BF16 values → one block_q8_1.
// block_q8_1 is 36 bytes (4-byte aligned), so array indexing is safe.

extern "C" __global__ void quantize_bf16_q8_1(
    const __nv_bfloat16* __restrict__ x,
    void* __restrict__ y_raw,
    uint32_t K
) {
    block_q8_1* y = (block_q8_1*)y_raw;
    const uint32_t block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t num_blocks = K / 32;
    if (block_idx >= num_blocks) return;

    const uint32_t lane = threadIdx.x;
    float val = __bfloat162float(x[block_idx * 32 + lane]);

    // Warp reduction: max absolute value
    float amax = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));

    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;

    int8_t q = (int8_t)roundf(val * id);

    // Warp reduction: sum of quantized values
    int sum_val = (int)q;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, mask);

    y[block_idx].qs[lane] = q;
    if (lane == 0) {
        __half hd = __float2half_rn(d);
        __half hs = __float2half_rn(d * (float)sum_val);
        memcpy(&y[block_idx].d, &hd, 2);
        memcpy(&y[block_idx].s, &hs, 2);
    }
}

// ── Vec-dot device functions ────────────────────────────────────────────
//
// Each function computes the dot product of one quantized weight block
// with the corresponding Q8_1 activation block(s).
// Uses __dp4a (int8 dot product) for 4-element-at-a-time processing.
// All int32 loads use load_int() for alignment safety.
//
// Q8_1 blocks are 36 bytes (4-byte aligned), so direct int* casts are safe
// for their qs array (offset 4). Weight block arrays may be misaligned.

// ── Q4_0 × Q8_1: symmetric 4-bit, offset -8 ───────────────────────────

__device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const block_q4_0& w, const block_q8_1& a
) {
    float d4 = fp16_to_f32(w.d);
    float d8 = fp16_to_f32(a.d);
    float s8 = fp16_to_f32(a.s);

    const int* q8 = (const int*)a.qs; // Q8_1 is 4-byte aligned

    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int v = load_int(w.qs + j * 4);
        sumi = __dp4a(v & 0x0F0F0F0F, q8[j], sumi);
        sumi = __dp4a((v >> 4) & 0x0F0F0F0F, q8[j + 4], sumi);
    }

    return d4 * (d8 * (float)sumi - 8.0f * s8);
}

// ── Q4_1 × Q8_1: asymmetric 4-bit with minimum ────────────────────────

__device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const block_q4_1& w, const block_q8_1& a
) {
    float d4 = fp16_to_f32(w.d);
    float m4 = fp16_to_f32(w.m);
    float d8 = fp16_to_f32(a.d);
    float s8 = fp16_to_f32(a.s);

    const int* q8 = (const int*)a.qs;

    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int v = load_int(w.qs + j * 4);
        sumi = __dp4a(v & 0x0F0F0F0F, q8[j], sumi);
        sumi = __dp4a((v >> 4) & 0x0F0F0F0F, q8[j + 4], sumi);
    }

    return d4 * d8 * (float)sumi + m4 * s8;
}

// ── Q5_0 × Q8_1: symmetric 5-bit, offset -16 ──────────────────────────

__device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const block_q5_0& w, const block_q8_1& a
) {
    float d5 = fp16_to_f32(w.d);
    float d8 = fp16_to_f32(a.d);
    float s8 = fp16_to_f32(a.s);

    uint32_t qh;
    memcpy(&qh, w.qh, sizeof(qh));

    const int* q8 = (const int*)a.qs;

    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int v = load_int(w.qs + j * 4);
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        uint32_t hbits_lo = (qh >> (j * 4));
        v0 |= ((int)(hbits_lo & 1) << 4);
        v0 |= ((int)(hbits_lo & 2) << 11);
        v0 |= ((int)(hbits_lo & 4) << 18);
        v0 |= ((int)(hbits_lo & 8) << 25);

        uint32_t hbits_hi = (qh >> (16 + j * 4));
        v1 |= ((int)(hbits_hi & 1) << 4);
        v1 |= ((int)(hbits_hi & 2) << 11);
        v1 |= ((int)(hbits_hi & 4) << 18);
        v1 |= ((int)(hbits_hi & 8) << 25);

        sumi = __dp4a(v0, q8[j], sumi);
        sumi = __dp4a(v1, q8[j + 4], sumi);
    }

    return d5 * (d8 * (float)sumi - 16.0f * s8);
}

// ── Q5_1 × Q8_1: asymmetric 5-bit with minimum ────────────────────────

__device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const block_q5_1& w, const block_q8_1& a
) {
    float d5 = fp16_to_f32(w.d);
    float m5 = fp16_to_f32(w.m);
    float d8 = fp16_to_f32(a.d);
    float s8 = fp16_to_f32(a.s);

    uint32_t qh;
    memcpy(&qh, w.qh, sizeof(qh));

    const int* q8 = (const int*)a.qs;

    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int v = load_int(w.qs + j * 4);
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        uint32_t hbits_lo = (qh >> (j * 4));
        v0 |= ((int)(hbits_lo & 1) << 4);
        v0 |= ((int)(hbits_lo & 2) << 11);
        v0 |= ((int)(hbits_lo & 4) << 18);
        v0 |= ((int)(hbits_lo & 8) << 25);

        uint32_t hbits_hi = (qh >> (16 + j * 4));
        v1 |= ((int)(hbits_hi & 1) << 4);
        v1 |= ((int)(hbits_hi & 2) << 11);
        v1 |= ((int)(hbits_hi & 4) << 18);
        v1 |= ((int)(hbits_hi & 8) << 25);

        sumi = __dp4a(v0, q8[j], sumi);
        sumi = __dp4a(v1, q8[j + 4], sumi);
    }

    return d5 * d8 * (float)sumi + m5 * s8;
}

// ── Q8_0 × Q8_1: simple 8-bit ──────────────────────────────────────────

__device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const block_q8_0& w, const block_q8_1& a
) {
    float d0 = fp16_to_f32(w.d);
    float d1 = fp16_to_f32(a.d);

    const int* q1 = (const int*)a.qs;

    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int v = load_int(w.qs + j * 4);
        sumi = __dp4a(v, q1[j], sumi);
    }

    return d0 * d1 * (float)sumi;
}

// ── Q2_K × Q8_1: 2-bit K-quant ─────────────────────────────────────────

__device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const block_q2_K& w, const block_q8_1* a
) {
    float dall = fp16_to_f32(w.d);
    float dmin = fp16_to_f32(w.dmin);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int q8_idx = 0; q8_idx < 8; q8_idx++) {
        const int g = q8_idx / 4;
        const int s = q8_idx % 4;

        float d8 = fp16_to_f32(a[q8_idx].d);
        const int* q8_int = (const int*)a[q8_idx].qs;

        const int sub_first  = q8_idx * 2;
        const int sub_second = q8_idx * 2 + 1;
        const int sc_first  = w.scales[sub_first]  & 0xF;
        const int sc_second = w.scales[sub_second] & 0xF;
        const int m_first   = w.scales[sub_first]  >> 4;
        const int m_second  = w.scales[sub_second] >> 4;

        int sumi_first = 0, sumi_second = 0;
        int sum_ones_first = 0, sum_ones_second = 0;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int v = load_int(w.qs + g * 32 + j * 4);
            int q2 = (v >> (s * 2)) & 0x03030303;
            sumi_first = __dp4a(q2, q8_int[j], sumi_first);
            sum_ones_first = __dp4a(0x01010101, q8_int[j], sum_ones_first);
        }

        #pragma unroll
        for (int j = 4; j < 8; j++) {
            int v = load_int(w.qs + g * 32 + j * 4);
            int q2 = (v >> (s * 2)) & 0x03030303;
            sumi_second = __dp4a(q2, q8_int[j], sumi_second);
            sum_ones_second = __dp4a(0x01010101, q8_int[j], sum_ones_second);
        }

        sumf_d += d8 * ((float)(sumi_first * sc_first) +
                         (float)(sumi_second * sc_second));
        sumf_m += d8 * ((float)(sum_ones_first * m_first) +
                         (float)(sum_ones_second * m_second));
    }

    return dall * sumf_d - dmin * sumf_m;
}

// ── Q3_K × Q8_1: 3-bit K-quant ─────────────────────────────────────────

__device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const block_q3_K& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t auxs[4];
    memcpy(auxs, w.scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);

    float sumf = 0.0f;

    for (int q8_idx = 0; q8_idx < 8; q8_idx++) {
        const int g = q8_idx / 4;
        const int s = q8_idx % 4;
        const int hmask_bit = g * 4 + s;

        float d8 = fp16_to_f32(a[q8_idx].d);
        const int* q8_int = (const int*)a[q8_idx].qs;

        const int sub_first  = q8_idx * 2;
        const int sub_second = q8_idx * 2 + 1;
        float sc_first  = (float)(scales[sub_first]  - 32);
        float sc_second = (float)(scales[sub_second] - 32);

        int sumi_first = 0, sumi_second = 0;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int v = load_int(w.qs + g * 32 + j * 4);
            int low2 = (v >> (s * 2)) & 0x03030303;

            int h = load_int(w.hmask + j * 4);
            int h_bits = (h >> hmask_bit) & 0x01010101;
            int vih = ((0x01010101 - h_bits) << 2) & 0x04040404;
            int q3 = __vsubss4(low2, vih);

            sumi_first = __dp4a(q3, q8_int[j], sumi_first);
        }

        #pragma unroll
        for (int j = 4; j < 8; j++) {
            int v = load_int(w.qs + g * 32 + j * 4);
            int low2 = (v >> (s * 2)) & 0x03030303;

            int h = load_int(w.hmask + j * 4);
            int h_bits = (h >> hmask_bit) & 0x01010101;
            int vih = ((0x01010101 - h_bits) << 2) & 0x04040404;
            int q3 = __vsubss4(low2, vih);

            sumi_second = __dp4a(q3, q8_int[j], sumi_second);
        }

        sumf += d8 * ((float)sumi_first * sc_first +
                       (float)sumi_second * sc_second);
    }

    return d_all * sumf;
}

// ── Q4_K × Q8_1: 4-bit K-quant with scale/min ─────────────────────────

__device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const block_q4_K& w, const block_q8_1* a
) {
    float dall = fp16_to_f32(w.d);
    float dmin = fp16_to_f32(w.dmin);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int sub = 0; sub < 8; sub++) {
        uint8_t sc, m_val;
        get_scale_min_k4(sub, w.scales, sc, m_val);

        float d8 = fp16_to_f32(a[sub].d);
        float s8 = fp16_to_f32(a[sub].s);

        int qs_base = (sub / 2) * 32;
        const int* q8 = (const int*)a[sub].qs;

        int sumi = 0;
        if (sub % 2 == 0) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int v = load_int(w.qs + qs_base + j * 4);
                sumi = __dp4a(v & 0x0F0F0F0F, q8[j], sumi);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int v = load_int(w.qs + qs_base + j * 4);
                sumi = __dp4a((v >> 4) & 0x0F0F0F0F, q8[j], sumi);
            }
        }

        sumf_d += d8 * (float)sumi * (float)sc;
        sumf_m += (float)m_val * s8;
    }

    return dall * sumf_d - dmin * sumf_m;
}

// ── Q5_K × Q8_1: 5-bit K-quant with scale/min ─────────────────────────

__device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const block_q5_K& w, const block_q8_1* a
) {
    float dall = fp16_to_f32(w.d);
    float dmin = fp16_to_f32(w.dmin);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int sub = 0; sub < 8; sub++) {
        uint8_t sc, m_val;
        get_scale_min_k4(sub, w.scales, sc, m_val);

        float d8 = fp16_to_f32(a[sub].d);
        float s8 = fp16_to_f32(a[sub].s);

        int qs_base = (sub / 2) * 32;
        const int* q8 = (const int*)a[sub].qs;
        int qh_bit = sub;

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int v_raw = load_int(w.qs + qs_base + j * 4);
            int v;
            if (sub % 2 == 0) {
                v = v_raw & 0x0F0F0F0F;
            } else {
                v = (v_raw >> 4) & 0x0F0F0F0F;
            }

            int qh4 = load_int(w.qh + j * 4);
            int h = (qh4 >> qh_bit) & 0x01010101;
            v |= (h << 4);

            sumi = __dp4a(v, q8[j], sumi);
        }

        sumf_d += d8 * (float)sumi * (float)sc;
        sumf_m += (float)m_val * s8;
    }

    return dall * sumf_d - dmin * sumf_m;
}

// ── Q6_K × Q8_1: 6-bit K-quant ─────────────────────────────────────────

__device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const block_q6_K& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int q8_idx = 0; q8_idx < 8; q8_idx++) {
        const int g = q8_idx / 4;
        const int s = q8_idx % 4;

        float d8 = fp16_to_f32(a[q8_idx].d);
        const int* q8_int = (const int*)a[q8_idx].qs;

        int ql_base = g * 64 + (s & 1) * 32;
        int qh_base = g * 32;
        int qh_shift = s * 2;

        const int sub_first  = q8_idx * 2;
        const int sub_second = q8_idx * 2 + 1;
        float sc_first  = (float)w.scales[sub_first];
        float sc_second = (float)w.scales[sub_second];

        int sumi_first = 0, sumi_second = 0;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int ql4 = load_int(w.ql + ql_base + j * 4);
            int ql_val = (s < 2) ? (ql4 & 0x0F0F0F0F) : ((ql4 >> 4) & 0x0F0F0F0F);

            int qh4 = load_int(w.qh + qh_base + j * 4);
            int qh_val = ((qh4 >> qh_shift) & 0x03030303) << 4;

            int q6 = ql_val | qh_val;
            int q6_signed = __vsubss4(q6, 0x20202020);

            sumi_first = __dp4a(q6_signed, q8_int[j], sumi_first);
        }

        #pragma unroll
        for (int j = 4; j < 8; j++) {
            int ql4 = load_int(w.ql + ql_base + j * 4);
            int ql_val = (s < 2) ? (ql4 & 0x0F0F0F0F) : ((ql4 >> 4) & 0x0F0F0F0F);

            int qh4 = load_int(w.qh + qh_base + j * 4);
            int qh_val = ((qh4 >> qh_shift) & 0x03030303) << 4;

            int q6 = ql_val | qh_val;
            int q6_signed = __vsubss4(q6, 0x20202020);

            sumi_second = __dp4a(q6_signed, q8_int[j], sumi_second);
        }

        sumf += d8 * ((float)sumi_first * sc_first +
                       (float)sumi_second * sc_second);
    }

    return d_all * sumf;
}

// ── MMVQ kernel macros ──────────────────────────────────────────────────
//
// Grid:  ((N + MMVQ_NWARPS - 1) / MMVQ_NWARPS, 1, 1)
// Block: (WARP_SIZE, MMVQ_NWARPS, 1)
// Each warp computes one output row, reduced via warp shuffle.

// Simple formats (QK=32, 1 Q8_1 per block)
#define MMVQ_KERNEL_SIMPLE(name, block_type, vec_dot_fn)                    \
extern "C" __global__ void name(                                            \
    const void* __restrict__ W,                                             \
    const void* __restrict__ x_q8,                                          \
    float* __restrict__ y,                                                  \
    uint32_t N,                                                             \
    uint32_t blocks_per_row                                                 \
) {                                                                         \
    const uint32_t row = blockIdx.x * MMVQ_NWARPS + threadIdx.y;            \
    if (row >= N) return;                                                   \
                                                                            \
    const uint32_t lane = threadIdx.x;                                      \
    const uint8_t* W_base = (const uint8_t*)W;                              \
    const block_q8_1* x = (const block_q8_1*)x_q8;                         \
    const uint64_t row_bytes = (uint64_t)blocks_per_row * sizeof(block_type);\
                                                                            \
    float sum = 0.0f;                                                       \
    for (uint32_t blk = lane; blk < blocks_per_row; blk += WARP_SIZE) {     \
        const block_type* wp =                                              \
            (const block_type*)(W_base + row * row_bytes                    \
                                + (uint64_t)blk * sizeof(block_type));      \
        sum += vec_dot_fn(*wp, x[blk]);                                     \
    }                                                                       \
                                                                            \
    sum = warp_reduce_sum_mmvq(sum);                                        \
    if (lane == 0) y[row] = sum;                                            \
}

// K-quant formats (QK=256, 8 Q8_1 per block)
#define MMVQ_KERNEL_KQUANT(name, block_type, vec_dot_fn)                    \
extern "C" __global__ void name(                                            \
    const void* __restrict__ W,                                             \
    const void* __restrict__ x_q8,                                          \
    float* __restrict__ y,                                                  \
    uint32_t N,                                                             \
    uint32_t blocks_per_row                                                 \
) {                                                                         \
    const uint32_t row = blockIdx.x * MMVQ_NWARPS + threadIdx.y;            \
    if (row >= N) return;                                                   \
                                                                            \
    const uint32_t lane = threadIdx.x;                                      \
    const uint8_t* W_base = (const uint8_t*)W;                              \
    const block_q8_1* x = (const block_q8_1*)x_q8;                         \
    const uint64_t row_bytes = (uint64_t)blocks_per_row * sizeof(block_type);\
                                                                            \
    float sum = 0.0f;                                                       \
    for (uint32_t blk = lane; blk < blocks_per_row; blk += WARP_SIZE) {     \
        const block_type* wp =                                              \
            (const block_type*)(W_base + row * row_bytes                    \
                                + (uint64_t)blk * sizeof(block_type));      \
        sum += vec_dot_fn(*wp, x + blk * 8);                               \
    }                                                                       \
                                                                            \
    sum = warp_reduce_sum_mmvq(sum);                                        \
    if (lane == 0) y[row] = sum;                                            \
}

// ── IQ block structures ────────────────────────────────────────────────

#define QK4_NL 32

struct __align__(2) block_iq4_nl {  // 18 bytes (same layout as Q4_0)
    uint16_t d;
    uint8_t qs[16];
};

struct __align__(2) block_iq4_xs {  // 136 bytes
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};

// ── IQ4_NL lookup table ────────────────────────────────────────────────

__device__ static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ── get_int_from_table_16: 4-bit nibble → 8-bit table lookup ───────────
//
// Takes an int32 containing 8 packed 4-bit indices and returns int2 where
// .x has looked-up values for even nibbles, .y for odd nibbles.
// Uses __byte_perm for efficient byte selection on CUDA.

__device__ __forceinline__ int2 get_int_from_table_16(int q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    const uint32_t low_high_selection = (0x32103210 | ((q4 & 0x88888888) >> 1));
    uint32_t tmp[2];
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection >> shift);
    }
    return make_int2(
        __byte_perm(tmp[0], tmp[1], 0x6420),
        __byte_perm(tmp[0], tmp[1], 0x7531)
    );
}

// ── IQ4_NL × Q8_1: non-linear 4-bit ───────────────────────────────────

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const block_iq4_nl& w, const block_q8_1& a
) {
    float d4 = fp16_to_f32(w.d);
    float d8 = fp16_to_f32(a.d);

    const int* q8 = (const int*)a.qs;  // Q8_1 is 4-byte aligned

    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int aux_q4 = load_int(w.qs + j * 4);
        int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = __dp4a(v.x, q8[j], sumi);
        sumi = __dp4a(v.y, q8[j + 4], sumi);
    }

    return d4 * d8 * (float)sumi;
}

// ── IQ4_XS × Q8_1: non-linear 4-bit with per-sub-block scales ─────────

__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const block_iq4_xs& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int sub = 0; sub < 8; sub++) {
        float d8 = fp16_to_f32(a[sub].d);
        const int* q8 = (const int*)a[sub].qs;

        // Extract 6-bit scale: 4 low bits from scales_l + 2 high bits from scales_h
        int ls = (w.scales_l[sub / 2] >> ((sub % 2) * 4)) & 0x0F;
        ls |= ((w.scales_h >> (sub * 2)) & 0x03) << 4;
        float scale = (float)(ls - 32);

        // Process 32 elements (16 bytes of qs → 32 nibbles)
        int qs_base = sub * 16;
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int aux_q4 = load_int(w.qs + qs_base + j * 4);
            int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
            sumi = __dp4a(v.x, q8[j], sumi);
            sumi = __dp4a(v.y, q8[j + 4], sumi);
        }

        sumf += d8 * (float)sumi * scale;
    }

    return d_all * sumf;
}

// ── MMVQ kernel instantiations ─────────────────────────────────────────

MMVQ_KERNEL_SIMPLE(mmvq_q4_0, block_q4_0, vec_dot_q4_0_q8_1)
MMVQ_KERNEL_SIMPLE(mmvq_q4_1, block_q4_1, vec_dot_q4_1_q8_1)
MMVQ_KERNEL_SIMPLE(mmvq_q5_0, block_q5_0, vec_dot_q5_0_q8_1)
MMVQ_KERNEL_SIMPLE(mmvq_q5_1, block_q5_1, vec_dot_q5_1_q8_1)
MMVQ_KERNEL_SIMPLE(mmvq_q8_0, block_q8_0, vec_dot_q8_0_q8_1)

MMVQ_KERNEL_KQUANT(mmvq_q2_K, block_q2_K, vec_dot_q2_K_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_q3_K, block_q3_K, vec_dot_q3_K_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_q4_K, block_q4_K, vec_dot_q4_K_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_q5_K, block_q5_K, vec_dot_q5_K_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_q6_K, block_q6_K, vec_dot_q6_K_q8_1)

MMVQ_KERNEL_SIMPLE(mmvq_iq4_nl, block_iq4_nl, vec_dot_iq4_nl_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq4_xs, block_iq4_xs, vec_dot_iq4_xs_q8_1)

// ── Remaining IQ formats (codebook-based) ──────────────────────────────
//
// These formats use large lookup tables (codebooks) to decode quantized
// indices into signed int8 values, then dp4a with Q8_1 activations.

#include "iq_tables.cuh"

// ── Additional IQ block structures ─────────────────────────────────────

struct __align__(2) block_iq3_xxs {  // 98 bytes
    uint16_t d;
    uint8_t qs[96];   // 3*QK_K/8: packed indices + signs + scales
};

struct __align__(2) block_iq3_s {    // 110 bytes
    uint16_t d;
    uint8_t qs[64];    // QK_K/4: low 8 bits of indices
    uint8_t qh[8];     // QK_K/32: high bit of indices
    uint8_t signs[32]; // QK_K/8: sign bits
    uint8_t scales[4]; // QK_K/64: sub-block scales
};

struct __align__(2) block_iq2_xxs {  // 66 bytes
    uint16_t d;
    uint16_t qs[32];   // QK_K/8: packed codebook indices + signs + scales
};

struct __align__(2) block_iq2_xs {   // 74 bytes
    uint16_t d;
    uint16_t qs[32];   // QK_K/8: 9-bit index + 7-bit signs
    uint8_t scales[8]; // QK_K/32: sub-block scales
};

struct __align__(2) block_iq2_s {    // 82 bytes
    uint16_t d;
    uint8_t qs[64];    // QK_K/4: low 8 bits of indices
    uint8_t qh[8];     // QK_K/32: high bits
    uint8_t scales[8]; // QK_K/32: sub-block scales
};

struct __align__(2) block_iq1_s {    // 50 bytes
    uint16_t d;
    uint8_t qs[32];    // QK_K/8: grid index low bits
    uint16_t qh[8];    // QK_K/32: grid index high bits + scale/delta
};

struct __align__(2) block_iq1_m {    // 56 bytes (no d field!)
    uint8_t qs[32];    // QK_K/8: grid index low bits
    uint8_t qh[16];    // QK_K/16: grid index high + shift
    uint8_t scales[8]; // QK_K/32: 3-bit block scales
};

// ── IQ helper: unpack sign bits ────────────────────────────────────────

__device__ __forceinline__ uint32_t unpack_ksigns(uint8_t v) {
    const uint32_t p = __popc((uint32_t)v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

// ── IQ3_XXS × Q8_1 ────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const block_iq3_xxs& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int iqs = 0; iqs < 16; iqs += 2) {
        float d8 = fp16_to_f32(a[iqs / 2].d);
        const int* q8 = (const int*)a[iqs / 2].qs;

        int q3_lo = load_int(w.qs + iqs * 4);
        int q3_hi = load_int(w.qs + iqs * 4 + 4);
        const uint8_t* q3 = (const uint8_t*)&q3_lo;
        const uint8_t* q3b = (const uint8_t*)&q3_hi;
        uint32_t aux32 = load_int(w.qs + QK_K / 4 + (iqs / 2) * 4);

        int sumi = 0;
        #pragma unroll
        for (int l0 = 0; l0 < 4; l0++) {
            int2 grid_pos = make_int2(iq3xxs_grid[q3[l0]], iq3xxs_grid[q3b[l0]]);
            uint32_t signs = unpack_ksigns(aux32 >> (7 * l0));

            int signs0 = __vcmpne4(signs & 0x08040201, 0);
            int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = __dp4a(grid_l, q8[l0 * 2], sumi);

            int signs1 = __vcmpne4(signs & 0x80402010, 0);
            int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = __dp4a(grid_h, q8[l0 * 2 + 1], sumi);
        }

        int ls = aux32 >> 28;
        sumi = (ls * sumi + sumi / 2) / 2;
        sumf += d8 * (float)sumi;
    }

    return d_all * sumf;
}

// ── IQ3_S × Q8_1 ──────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const block_iq3_s& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int iqs = 0; iqs < 16; iqs += 2) {
        float d8 = fp16_to_f32(a[iqs / 2].d);
        const int* q8 = (const int*)a[iqs / 2].qs;

        int qs_lo = load_int(w.qs + iqs * 4);
        int qs_hi = load_int(w.qs + iqs * 4 + 4);
        const uint8_t* qs = (const uint8_t*)&qs_lo;
        const uint8_t* qsb = (const uint8_t*)&qs_hi;

        int qh;
        memcpy(&qh, w.qh + (iqs / 2) * 4, 4);

        int signs_packed;
        memcpy(&signs_packed, w.signs + (iqs / 2) * 4, 4);
        const uint8_t* sp = (const uint8_t*)&signs_packed;

        int sumi = 0;
        #pragma unroll
        for (int l0 = 0; l0 < 4; l0++) {
            int2 grid_pos = make_int2(
                iq3s_grid[qs[l0]  | ((qh << (8 - l0 * 2)) & 0x100)],
                iq3s_grid[qsb[l0] | ((qh << (7 - l0 * 2)) & 0x100)]);

            int signs0 = __vcmpne4(((sp[l0] & 0x03) << 7) | ((sp[l0] & 0x0C) << 21), 0);
            int signs1 = __vcmpne4(((sp[l0] & 0x30) << 3) | ((sp[l0] & 0xC0) << 17), 0);

            int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            sumi = __dp4a(grid_l, q8[l0 * 2], sumi);
            sumi = __dp4a(grid_h, q8[l0 * 2 + 1], sumi);
        }

        sumi *= 1 + 2 * ((w.scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        sumf += d8 * (float)sumi;
    }

    return d_all * sumf;
}

// ── IQ2_XXS × Q8_1 ────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const block_iq2_xxs& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int iqs = 0; iqs < 16; iqs += 2) {
        float d8 = fp16_to_f32(a[iqs / 2].d);
        const int* q8 = (const int*)a[iqs / 2].qs;

        int q2 = load_int((const uint8_t*)w.qs + iqs * 4);
        const uint8_t* aux8 = (const uint8_t*)&q2;
        uint32_t aux32;
        memcpy(&aux32, (const uint8_t*)w.qs + iqs * 4 + 4, 4);

        int sumi = 0;
        #pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            int signs0 = __vcmpne4(signs & 0x08040201, 0);
            int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = __dp4a(grid0, q8[k0], sumi);

            int signs1 = __vcmpne4(signs & 0x80402010, 0);
            int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = __dp4a(grid1, q8[k0 + 1], sumi);
        }

        int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        sumf += d8 * (float)sumi;
    }

    return d_all * sumf;
}

// ── IQ2_XS × Q8_1 ─────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const block_iq2_xs& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int iqs = 0; iqs < 16; iqs += 2) {
        float d8 = fp16_to_f32(a[iqs / 2].d);
        const int* q8 = (const int*)a[iqs / 2].qs;

        int q2_lo = load_int((const uint8_t*)w.qs + iqs * 4);
        int q2_hi = load_int((const uint8_t*)w.qs + iqs * 4 + 4);
        const uint16_t* q2 = (const uint16_t*)&q2_lo;
        const uint16_t* q2b = (const uint16_t*)&q2_hi;
        int ls0 = w.scales[iqs / 2] & 0x0F;
        int ls1 = w.scales[iqs / 2] >> 4;

        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int l0 = 0; l0 < 4; l0++) {
            const uint16_t qval = (l0 < 2) ? q2[l0] : q2b[l0 - 2];
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[qval & 0x1FF];
            const uint32_t signs = unpack_ksigns(qval >> 9);

            int signs0 = __vcmpne4(signs & 0x08040201, 0);
            int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            int signs1 = __vcmpne4(signs & 0x80402010, 0);
            int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            if (l0 < 2) {
                sumi0 = __dp4a(grid_l, q8[l0 * 2], sumi0);
                sumi0 = __dp4a(grid_h, q8[l0 * 2 + 1], sumi0);
            } else {
                sumi1 = __dp4a(grid_l, q8[l0 * 2], sumi1);
                sumi1 = __dp4a(grid_h, q8[l0 * 2 + 1], sumi1);
            }
        }
        int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
        sumf += d8 * (float)sumi;
    }

    return d_all * sumf;
}

// ── IQ2_S × Q8_1 ──────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const block_iq2_s& w, const block_q8_1* a
) {
    float d_all = fp16_to_f32(w.d);
    float sumf = 0.0f;

    for (int iqs = 0; iqs < 16; iqs += 2) {
        float d8 = fp16_to_f32(a[iqs / 2].d);
        const int* q8 = (const int*)a[iqs / 2].qs;

        int qs_packed = load_int(w.qs + (iqs / 2) * 4);
        const uint8_t* qs = (const uint8_t*)&qs_packed;
        int qh;
        memcpy(&qh, w.qh + (iqs / 2) * 4, 4);

        int signs_packed = load_int(w.qs + QK_K / 4 + (iqs / 2) * 4);
        const uint8_t* sp = (const uint8_t*)&signs_packed;

        int ls0 = w.scales[iqs / 2] & 0x0F;
        int ls1 = w.scales[iqs / 2] >> 4;

        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int l0 = 0; l0 < 4; l0++) {
            const int* grid_pos = (const int*)(iq2s_grid + (qs[l0] | ((qh << (8 - l0 * 2)) & 0x300)));

            int signs0 = __vcmpne4(((sp[l0] & 0x03) << 7) | ((sp[l0] & 0x0C) << 21), 0);
            int signs1 = __vcmpne4(((sp[l0] & 0x30) << 3) | ((sp[l0] & 0xC0) << 17), 0);

            int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

            if (l0 < 2) {
                sumi0 = __dp4a(grid_l, q8[l0 * 2], sumi0);
                sumi0 = __dp4a(grid_h, q8[l0 * 2 + 1], sumi0);
            } else {
                sumi1 = __dp4a(grid_l, q8[l0 * 2], sumi1);
                sumi1 = __dp4a(grid_h, q8[l0 * 2 + 1], sumi1);
            }
        }
        int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
        sumf += d8 * (float)sumi;
    }

    return d_all * sumf;
}

// ── IQ1_S × Q8_1 ──────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const block_iq1_s& w, const block_q8_1* a
) {
    float sumf = 0.0f;

    for (int iqs = 0; iqs < 8; iqs++) {
        const float2 ds = __half22float2(*(const __half2*)&a[iqs].d);
        const int* q8 = (const int*)a[iqs].qs;

        int qs_packed = load_int(w.qs + iqs * 4);
        const uint8_t* qs = (const uint8_t*)&qs_packed;

        uint16_t qh_val;
        memcpy(&qh_val, &w.qh[iqs], 2);
        int qh = (int)qh_val;

        int sumi = 0;
        #pragma unroll
        for (int l0 = 0; l0 < 4; l0++) {
            int grid = iq1s_grid_gpu[qs[l0] | (((qh >> (3 * l0)) & 0x07) << 8)];
            int grid0 = (grid >> 0) & 0x0F0F0F0F;
            int grid1 = (grid >> 4) & 0x0F0F0F0F;
            sumi = __dp4a(grid0, q8[l0 * 2], sumi);
            sumi = __dp4a(grid1, q8[l0 * 2 + 1], sumi);
        }

        float d1q = fp16_to_f32(w.d) * (float)(((qh >> 11) & 0x0E) + 1);
        float delta = -1.0f + IQ1S_DELTA - (float)(qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
        sumf += d1q * (ds.x * (float)sumi + ds.y * delta);
    }

    return sumf;
}

// ── IQ1_M × Q8_1 ──────────────────────────────────────────────────────

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const block_iq1_m& w, const block_q8_1* a
) {
    float sumf = 0.0f;
    int sumi_arr[2];
    float sumf_arr[2];

    for (int iqs = 0; iqs < 8; iqs++) {
        const float2 ds = __half22float2(*(const __half2*)&a[iqs].d);
        const int* q8 = (const int*)a[iqs].qs;

        int qs_packed = load_int(w.qs + iqs * 4);
        const uint8_t* qs = (const uint8_t*)&qs_packed;

        sumi_arr[0] = 0; sumi_arr[1] = 0;
        sumf_arr[0] = 0.0f; sumf_arr[1] = 0.0f;

        #pragma unroll
        for (int l0 = 0; l0 < 4; l0++) {
            int qhl = w.qh[2 * iqs + l0 / 2] >> (4 * (l0 % 2));
            int grid = iq1s_grid_gpu[qs[l0] | ((qhl & 0x07) << 8)];

            int grid0 = (grid >> 0) & 0x0F0F0F0F;
            int grid1 = (grid >> 4) & 0x0F0F0F0F;

            sumi_arr[l0 / 2] = __dp4a(grid0, q8[l0 * 2], sumi_arr[l0 / 2]);
            sumi_arr[l0 / 2] = __dp4a(grid1, q8[l0 * 2 + 1], sumi_arr[l0 / 2]);

            float delta = -1.0f + IQ1M_DELTA - (float)(qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
            int sumy = 0;
            sumy = __dp4a(q8[l0 * 2], 0x01010101, sumy);
            sumy = __dp4a(q8[l0 * 2 + 1], 0x01010101, sumy);
            sumf_arr[l0 / 2] += delta * (float)sumy;
        }

        // Extract scale from packed scales array
        const uint16_t* sc = (const uint16_t*)w.scales;
        uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                           | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        float d = __half2float(*reinterpret_cast<const __half*>(&scale_u16))
                * ds.x;

        int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;

        sumf += d * (((float)sumi_arr[0] + sumf_arr[0]) * (float)sc0
                   + ((float)sumi_arr[1] + sumf_arr[1]) * (float)sc1);
    }

    return sumf;
}

// ── MMVQ kernel instantiations for remaining IQ formats ────────────────

MMVQ_KERNEL_KQUANT(mmvq_iq3_xxs, block_iq3_xxs, vec_dot_iq3_xxs_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq3_s,   block_iq3_s,   vec_dot_iq3_s_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq2_xxs, block_iq2_xxs, vec_dot_iq2_xxs_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq2_xs,  block_iq2_xs,  vec_dot_iq2_xs_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq2_s,   block_iq2_s,   vec_dot_iq2_s_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq1_s,   block_iq1_s,   vec_dot_iq1_s_q8_1)
MMVQ_KERNEL_KQUANT(mmvq_iq1_m,   block_iq1_m,   vec_dot_iq1_m_q8_1)
