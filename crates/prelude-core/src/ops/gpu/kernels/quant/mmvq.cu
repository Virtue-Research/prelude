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
