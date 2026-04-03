// AMX BF16 GEMM micro-kernel for Intel Xeon (Sapphire Rapids+).
//
// Uses AMX tile instructions (tdpbf16ps) for small-M BF16 GEMM.
// Weight must be pre-packed into VNNI tile format via amx_bf16_pack_weights().
//
// AMX tile operation: C[M,16] += A[M,32_bf16] × B[16_k_pairs, 32_bf16]
// Each tile multiply processes 32 K elements × 16 N outputs.
//
// Requires: Intel Xeon with AMX-BF16 (Sapphire Rapids / Emerald Rapids).
// Falls back gracefully on non-AMX CPUs (functions return 0/NULL).

#include "onednn_ffi.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(__x86_64__) && defined(__AMX_TILE__) && defined(__AMX_BF16__)

#include <immintrin.h>
#include <cpuid.h>

// ── Runtime AMX detection ──────────────────────────────────────────────

static int check_amx_cpuid(void) {
    unsigned int eax, ebx, ecx, edx;
    // CPUID leaf 7, subleaf 0: EDX bit 22 = AMX-BF16, EDX bit 24 = AMX-TILE
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return 0;
    return ((edx >> 22) & 1) && ((edx >> 24) & 1);
}

int amx_bf16_available(void) {
    static int cached = -1;
    if (cached < 0) cached = check_amx_cpuid();
    return cached;
}

// ── Helpers ────────────────────────────────────────────────────────────

static inline uint16_t f32_to_bf16_rne(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t rounding = bits + 0x7FFFu + ((bits >> 16) & 1);
    return (uint16_t)(rounding >> 16);
}

// AMX palette 1 configuration — must be 64-byte aligned, exactly 64 bytes.
//
// Layout:
//   byte  0:    palette_id (1)
//   byte  1:    start_row (0)
//   bytes 2-15: reserved
//   bytes 16-47: colsb[0..15] as uint16_t (tiles 0-7 used)
//   bytes 48-63: rows[0..15] as uint8_t  (tiles 0-7 used)
typedef struct __attribute__((aligned(64))) {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
} TileCfg;

// ── Weight packing (VNNI format for AMX tdpbf16ps) ─────────────────────
//
// Input:  weight[N, K] row-major BF16
// Output: packed blocks of [16_k_pairs, 16_n_cols, 2_bf16] = 1024 bytes each
//
// Block (nt, kt) covers output columns [nt*16, nt*16+16) and K range [kt*32, kt*32+32).
// Within each block, row k_pair (0..15) contains:
//   {w[n0][2k], w[n0][2k+1], w[n1][2k], w[n1][2k+1], ..., w[n15][2k], w[n15][2k+1]}
// This matches tdpbf16ps B-operand VNNI layout.

void* amx_bf16_pack_weights(const void* weight_ptr, int64_t k, int64_t n,
                             int64_t* packed_size_out) {
    if (!amx_bf16_available()) {
        *packed_size_out = 0;
        return NULL;
    }

    const uint16_t* w = (const uint16_t*)weight_ptr;
    int64_t k_tiles = (k + 31) / 32;
    int64_t n_tiles = (n + 15) / 16;
    int64_t total_bytes = n_tiles * k_tiles * 1024;  // 512 uint16 × 2 bytes per block

    uint16_t* packed = (uint16_t*)aligned_alloc(64, (size_t)total_bytes);
    if (!packed) {
        *packed_size_out = 0;
        return NULL;
    }
    memset(packed, 0, (size_t)total_bytes);

    for (int64_t nt = 0; nt < n_tiles; nt++) {
        for (int64_t kt = 0; kt < k_tiles; kt++) {
            uint16_t* blk = packed + (nt * k_tiles + kt) * 512;
            for (int kp = 0; kp < 16; kp++) {
                for (int nc = 0; nc < 16; nc++) {
                    int64_t ni  = nt * 16 + nc;
                    int64_t ki0 = kt * 32 + kp * 2;
                    int64_t ki1 = ki0 + 1;
                    int off = kp * 32 + nc * 2;
                    if (ni < n && ki0 < k) blk[off]     = w[ni * k + ki0];
                    if (ni < n && ki1 < k) blk[off + 1] = w[ni * k + ki1];
                }
            }
        }
    }

    *packed_size_out = total_bytes;
    return packed;
}

void amx_bf16_free_packed(void* packed) {
    free(packed);
}

// ── AMX GEMM kernel ────────────────────────────────────────────────────
//
// Computes output[M, n_start:n_end] = input[M, K] × packed_weight^T
// using AMX tdpbf16ps tiles. Caller parallelizes over N ranges.
//
// Tiles used:
//   tmm0: A (input),       mt rows × 64 bytes (32 BF16)
//   tmm1: B (weight VNNI), 16 rows × 64 bytes
//   tmm2: C (accumulator), mt rows × 64 bytes (16 F32)

void amx_bf16_gemm_packed(
    const void* input_ptr,
    const void* packed_ptr,
    void* output_ptr,
    int64_t m, int64_t k, int64_t n,
    int64_t n_start, int64_t n_end)
{
    const uint16_t* input  = (const uint16_t*)input_ptr;
    const uint16_t* packed = (const uint16_t*)packed_ptr;
    uint16_t*       output = (uint16_t*)output_ptr;

    if (m <= 0 || n_start >= n_end) return;

    int64_t k_tiles = (k + 31) / 32;

    // Aligned buffers for input padding and accumulator readback
    float    acc_buf[16 * 16] __attribute__((aligned(64)));
    uint16_t in_buf[16 * 32]  __attribute__((aligned(64)));

    int64_t nt_start = n_start / 16;
    int64_t nt_end   = (n_end + 15) / 16;

    // M-tiling: process M in chunks of 16 rows
    for (int64_t mr = 0; mr < m; mr += 16) {
        int mt = (int)((mr + 16 <= m) ? 16 : (m - mr));
        const uint16_t* in_row = input + mr * k;
        uint16_t*       out_row = output + mr * n;

        // Configure AMX tiles for this M-tile height
        TileCfg cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.rows[0] = (uint8_t)mt;  cfg.colsb[0] = 64;  // tmm0: A
        cfg.rows[1] = 16;           cfg.colsb[1] = 64;  // tmm1: B
        cfg.rows[2] = (uint8_t)mt;  cfg.colsb[2] = 64;  // tmm2: C
        _tile_loadconfig(&cfg);

        for (int64_t nt = nt_start; nt < nt_end; nt++) {
            int64_t nc = nt * 16;
            int na = (int)((nc + 16 <= n) ? 16 : (n - nc));

            _tile_zero(2);

            for (int64_t kt = 0; kt < k_tiles; kt++) {
                int64_t ks = kt * 32;
                int     ka = (int)((ks + 32 <= k) ? 32 : (k - ks));

                // Prefetch next weight block into L1
                if (kt + 1 < k_tiles) {
                    const char* nxt = (const char*)(packed + (nt * k_tiles + kt + 1) * 512);
                    _mm_prefetch(nxt,       _MM_HINT_T0);
                    _mm_prefetch(nxt + 64,  _MM_HINT_T0);
                    _mm_prefetch(nxt + 128, _MM_HINT_T0);
                    _mm_prefetch(nxt + 192, _MM_HINT_T0);
                }

                // Load input tile A
                if (ka < 32) {
                    // Tail K chunk: copy with zero-padding
                    memset(in_buf, 0, (size_t)(mt * 64));
                    for (int r = 0; r < mt; r++)
                        memcpy(&in_buf[r * 32], &in_row[r * k + ks], (size_t)(ka * 2));
                    _tile_loadd(0, in_buf, 64);
                } else {
                    // Full K chunk: load directly with stride = k * 2 bytes
                    _tile_loadd(0, &in_row[ks], (int)(k * 2));
                }

                // Load packed weight tile B (VNNI format, 64-byte aligned)
                _tile_loadd(1, packed + (nt * k_tiles + kt) * 512, 64);

                // AMX BF16 tile multiply-accumulate: C += A × B
                _tile_dpbf16ps(2, 0, 1);
            }

            // Store accumulator (F32) and convert to BF16 output
            _tile_stored(2, acc_buf, 64);

            for (int r = 0; r < mt; r++) {
                for (int c = 0; c < na; c++)
                    out_row[r * n + nc + c] = f32_to_bf16_rne(acc_buf[r * 16 + c]);
            }
        }
    }

    _tile_release();
}

#else
// ── Non-AMX stubs ──────────────────────────────────────────────────────

int amx_bf16_available(void) { return 0; }

void* amx_bf16_pack_weights(const void* w, int64_t k, int64_t n, int64_t* s) {
    (void)w; (void)k; (void)n;
    *s = 0;
    return NULL;
}

void amx_bf16_free_packed(void* p) { (void)p; }

void amx_bf16_gemm_packed(
    const void* i, const void* w, void* o,
    int64_t m, int64_t k, int64_t n, int64_t ns, int64_t ne)
{
    (void)i;(void)w;(void)o;(void)m;(void)k;(void)n;(void)ns;(void)ne;
}

#endif
