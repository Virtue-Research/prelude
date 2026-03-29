// SM90 FP8 GEMM: Normal 1D2D + M-Grouped Contiguous 1D2D
// Heuristics, kernel instantiations, dispatch, and implementation.

#pragma once

// ── SM90 FP8 1D2D heuristic ───────────────────────────────────────

static FP8Config select_fp8_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;

    // 1D2D: BLOCK_M % WGMMA::M(64)==0 OR BLOCK_M < 64
    int block_ms[4] = {64, 128, 0, 0};
    int n_block_ms = 2;
    if (m <= 16) { block_ms[n_block_ms++] = 16; }
    if (m <= 32) { block_ms[n_block_ms++] = 32; }

    // Valid block_n for 1D2D
    int block_ns[12]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16)
        if (fp8_valid_block_n(i)) block_ns[n_block_ns++] = i;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm == 0) continue;
            if (bm > 128 && bn > 128) continue;
            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

            bool better = false;
            if (best_bm == 0 || waves < best_waves) better = true;
            else if (waves == best_waves) {
                better = last_util > best_last;
                if (last_util == best_last) {
                    better |= (bm == best_bm && bn < best_bn);
                    better |= (bn == best_bn && bm < best_bm);
                    better |= (bm != best_bm && bn > best_bn && bn <= n && bm <= m);
                }
            }
            if (better) { best_bm = bm; best_bn = bn; best_waves = waves; best_last = last_util; }
        }
    }

    int num_tma = 128, num_math = (best_bm <= 64) ? 128 : 256;

    // Multicast
    int multicast = 1; bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, best_bm) % 2 == 0);
        bool order[2] = {false, true};
        if (best_bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) { multicast = 2; mc_on_a = order[i]; break; }
        }
    }

    // Swizzle: A/B use FP8 (1 byte), D uses BF16 (2 bytes)
    int sw_a = 128, sw_b = 128;
    int sw_d = get_swizzle(best_bn);

    // kNumLastStages for 1D2D
    int nls;
    if (block_k % best_bn == 0) nls = 0;
    else if (best_bn <= block_k) nls = ceil_div(best_bn, block_k - best_bn);
    else nls = 1;

    // SMEM: BF16 output, SFA per stage, SFB from global memory (one-time buffer)
    const int smem_capacity = 232448;
    int smem_d = ((best_bm * best_bn * 2 + 1023) / 1024) * 1024;
    int smem_a_per = best_bm * block_k;
    int smem_b_per = best_bn * block_k;
    int smem_sfa_per = ((best_bm * 4 + 127) / 128) * 128;
    int smem_per_stage = smem_a_per + smem_b_per + smem_sfa_per;

    int k_scales = ceil_div(k, block_k);
    bool must_uniform = (block_k % best_bn == 0);
    int sfb_buf = ((k_scales * (must_uniform ? 1 : 2) * 4 + 15) / 16) * 16;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int barrier = s * 16;
        int total = smem_d + s * smem_per_stage + barrier + sfb_buf;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return FP8Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages, .num_last_stages = nls,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── Grouped FP8 1D2D heuristic ─────────────────────────────────────

static FP8Config select_fp8_grouped_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;
    const int bm = 128; // MGroupedContiguous: fixed

    int block_ns[12]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16)
        if (fp8_valid_block_n(i)) block_ns[n_block_ns++] = i;

    int best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int j = 0; j < n_block_ns; j++) {
        int bn = block_ns[j];
        int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
        int waves = ceil_div(num_blocks, num_sms);
        int last_util = num_blocks % num_sms;
        if (last_util == 0) last_util = num_sms;

        bool better = false;
        if (best_bn == 0 || waves < best_waves) better = true;
        else if (waves == best_waves) {
            better = last_util > best_last;
            if (last_util == best_last) better |= (bn < best_bn);
        }
        if (better) { best_bn = bn; best_waves = waves; best_last = last_util; }
    }

    int num_tma = 128, num_math = 256;

    int multicast = 1; bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, bm) % 2 == 0);
        bool order[2] = {false, true};
        if (bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) { multicast = 2; mc_on_a = order[i]; break; }
        }
    }

    int sw_a = 128, sw_b = 128;
    int sw_d = get_swizzle(best_bn);

    int nls;
    if (block_k % best_bn == 0) nls = 0;
    else if (best_bn <= block_k) nls = ceil_div(best_bn, block_k - best_bn);
    else nls = 1;

    const int smem_capacity = 232448;
    int smem_d = ((bm * best_bn * 2 + 1023) / 1024) * 1024;
    int smem_a_per = bm * block_k;
    int smem_b_per = best_bn * block_k;
    int smem_sfa_per = ((bm * 4 + 127) / 128) * 128;
    int smem_per_stage = smem_a_per + smem_b_per + smem_sfa_per;

    int k_scales = ceil_div(k, block_k);
    bool must_uniform = (block_k % best_bn == 0);
    int sfb_buf = ((k_scales * (must_uniform ? 1 : 2) * 4 + 15) / 16) * 16;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int barrier = s * 16;
        int total = smem_d + s * smem_per_stage + barrier + sfb_buf;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return FP8Config{
        .block_m = bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages, .num_last_stages = nls,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── FP8 1D2D kernel instantiations ─────────────────────────────────

#define KERNEL_TYPE_FP8(BLOCK_M, BLOCK_N, STAGES, LAST_STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_fp8_gemm_1d2d_impl<                                       \
        cute::UMMA::Major::MN,                                                 \
        0, 0, 0, 1,                                                            \
        BLOCK_M, BLOCK_N, 128,                                                 \
        128, 128, SWIZZLE_D,                                                   \
        STAGES, LAST_STAGES,                                                   \
        128, NUM_MATH,                                                         \
        NUM_MC, MC_ON_A, 132,                                                  \
        GemmType::Normal,                                                      \
        deep_gemm::EpilogueIdentity>

__attribute__((used)) static auto* _fp8_00 = &KERNEL_TYPE_FP8(16, 16, 32, 0, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_01 = &KERNEL_TYPE_FP8(16, 32, 32, 0, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_02 = &KERNEL_TYPE_FP8(16, 48, 27, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_03 = &KERNEL_TYPE_FP8(16, 64, 22, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_04 = &KERNEL_TYPE_FP8(16, 80, 18, 2, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_05 = &KERNEL_TYPE_FP8(16, 96, 15, 3, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_06 = &KERNEL_TYPE_FP8(16, 112, 13, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_07 = &KERNEL_TYPE_FP8(16, 256, 6, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_08 = &KERNEL_TYPE_FP8(32, 16, 32, 0, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_09 = &KERNEL_TYPE_FP8(32, 32, 27, 0, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_0a = &KERNEL_TYPE_FP8(32, 48, 21, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_0b = &KERNEL_TYPE_FP8(32, 48, 22, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_0c = &KERNEL_TYPE_FP8(32, 64, 18, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_0d = &KERNEL_TYPE_FP8(32, 80, 15, 2, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_0e = &KERNEL_TYPE_FP8(32, 96, 13, 3, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_0f = &KERNEL_TYPE_FP8(32, 112, 12, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_10 = &KERNEL_TYPE_FP8(32, 256, 5, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_11 = &KERNEL_TYPE_FP8(64, 16, 21, 0, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_12 = &KERNEL_TYPE_FP8(64, 32, 18, 0, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_13 = &KERNEL_TYPE_FP8(64, 48, 15, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_14 = &KERNEL_TYPE_FP8(64, 64, 13, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_15 = &KERNEL_TYPE_FP8(64, 64, 13, 0, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_16 = &KERNEL_TYPE_FP8(64, 80, 11, 2, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_17 = &KERNEL_TYPE_FP8(64, 96, 10, 3, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_18 = &KERNEL_TYPE_FP8(64, 112, 9, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_19 = &KERNEL_TYPE_FP8(64, 128, 8, 0, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_1a = &KERNEL_TYPE_FP8(64, 128, 8, 0, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_1b = &KERNEL_TYPE_FP8(64, 144, 7, 1, 128, 32, 1, false);
__attribute__((used)) static auto* _fp8_1c = &KERNEL_TYPE_FP8(64, 144, 7, 1, 128, 32, 2, false);
__attribute__((used)) static auto* _fp8_1d = &KERNEL_TYPE_FP8(64, 160, 7, 1, 128, 64, 1, false);
__attribute__((used)) static auto* _fp8_1e = &KERNEL_TYPE_FP8(64, 160, 7, 1, 128, 64, 2, false);
__attribute__((used)) static auto* _fp8_1f = &KERNEL_TYPE_FP8(64, 192, 6, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_20 = &KERNEL_TYPE_FP8(64, 192, 6, 1, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_21 = &KERNEL_TYPE_FP8(64, 256, 4, 1, 128, 128, 1, false);
__attribute__((used)) static auto* _fp8_22 = &KERNEL_TYPE_FP8(64, 256, 4, 1, 128, 128, 2, false);
__attribute__((used)) static auto* _fp8_23 = &KERNEL_TYPE_FP8(64, 256, 4, 1, 128, 128, 2, true);
__attribute__((used)) static auto* _fp8_24 = &KERNEL_TYPE_FP8(128, 112, 6, 7, 256, 32, 1, false);
__attribute__((used)) static auto* _fp8_25 = &KERNEL_TYPE_FP8(128, 144, 5, 1, 256, 32, 1, false);
__attribute__((used)) static auto* _fp8_26 = &KERNEL_TYPE_FP8(128, 144, 5, 1, 256, 32, 2, false);
__attribute__((used)) static auto* _fp8_27 = &KERNEL_TYPE_FP8(128, 160, 5, 1, 256, 64, 2, false);
__attribute__((used)) static auto* _fp8_28 = &KERNEL_TYPE_FP8(128, 192, 4, 1, 256, 128, 1, false);
__attribute__((used)) static auto* _fp8_29 = &KERNEL_TYPE_FP8(128, 192, 4, 1, 256, 128, 2, false);
__attribute__((used)) static auto* _fp8_2a = &KERNEL_TYPE_FP8(128, 256, 3, 1, 256, 128, 1, false);
__attribute__((used)) static auto* _fp8_2b = &KERNEL_TYPE_FP8(128, 256, 3, 1, 256, 128, 2, false);
__attribute__((used)) static auto* _fp8_2c = &KERNEL_TYPE_FP8(128, 256, 3, 1, 256, 128, 2, true);

// ── M-Grouped Contiguous FP8 1D2D kernel instantiations ────────────

#define KERNEL_TYPE_FP8_GROUPED(BLOCK_N, STAGES, LAST_STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_fp8_gemm_1d2d_impl<                                       \
        cute::UMMA::Major::MN,                                                 \
        0, 0, 0, 1,                                                            \
        128, BLOCK_N, 128,                                                     \
        128, 128, SWIZZLE_D,                                                   \
        STAGES, LAST_STAGES,                                                   \
        128, 256,                                                              \
        NUM_MC, MC_ON_A, 132,                                                  \
        GemmType::MGroupedContiguous,                                          \
        deep_gemm::EpilogueIdentity>

__attribute__((used)) static auto* _fp8g_00 = &KERNEL_TYPE_FP8_GROUPED(16, 12, 0, 32, 1, false);
__attribute__((used)) static auto* _fp8g_01 = &KERNEL_TYPE_FP8_GROUPED(32, 10, 0, 64, 1, false);
__attribute__((used)) static auto* _fp8g_02 = &KERNEL_TYPE_FP8_GROUPED(48, 9, 1, 32, 1, false);
__attribute__((used)) static auto* _fp8g_03 = &KERNEL_TYPE_FP8_GROUPED(64, 8, 0, 128, 1, false);
__attribute__((used)) static auto* _fp8g_04 = &KERNEL_TYPE_FP8_GROUPED(80, 7, 2, 32, 1, false);
__attribute__((used)) static auto* _fp8g_05 = &KERNEL_TYPE_FP8_GROUPED(96, 7, 3, 64, 1, false);
__attribute__((used)) static auto* _fp8g_06 = &KERNEL_TYPE_FP8_GROUPED(112, 6, 7, 32, 1, false);
__attribute__((used)) static auto* _fp8g_07 = &KERNEL_TYPE_FP8_GROUPED(128, 5, 0, 128, 1, false);
__attribute__((used)) static auto* _fp8g_08 = &KERNEL_TYPE_FP8_GROUPED(144, 5, 1, 32, 1, false);
__attribute__((used)) static auto* _fp8g_09 = &KERNEL_TYPE_FP8_GROUPED(160, 5, 1, 64, 1, false);
__attribute__((used)) static auto* _fp8g_0a = &KERNEL_TYPE_FP8_GROUPED(192, 4, 1, 128, 1, false);
__attribute__((used)) static auto* _fp8g_0b = &KERNEL_TYPE_FP8_GROUPED(256, 3, 1, 128, 1, false);
__attribute__((used)) static auto* _fp8g_10 = &KERNEL_TYPE_FP8_GROUPED(16, 12, 0, 32, 2, true);
__attribute__((used)) static auto* _fp8g_11 = &KERNEL_TYPE_FP8_GROUPED(32, 10, 0, 64, 2, true);
__attribute__((used)) static auto* _fp8g_12 = &KERNEL_TYPE_FP8_GROUPED(48, 9, 1, 32, 2, true);
__attribute__((used)) static auto* _fp8g_13 = &KERNEL_TYPE_FP8_GROUPED(64, 8, 0, 128, 2, true);
__attribute__((used)) static auto* _fp8g_14 = &KERNEL_TYPE_FP8_GROUPED(80, 7, 2, 32, 2, true);
__attribute__((used)) static auto* _fp8g_15 = &KERNEL_TYPE_FP8_GROUPED(96, 7, 3, 64, 2, true);
__attribute__((used)) static auto* _fp8g_16 = &KERNEL_TYPE_FP8_GROUPED(112, 6, 7, 32, 2, false);
__attribute__((used)) static auto* _fp8g_17 = &KERNEL_TYPE_FP8_GROUPED(128, 5, 0, 128, 2, false);
__attribute__((used)) static auto* _fp8g_18 = &KERNEL_TYPE_FP8_GROUPED(144, 5, 1, 32, 2, false);
__attribute__((used)) static auto* _fp8g_19 = &KERNEL_TYPE_FP8_GROUPED(160, 5, 1, 64, 2, false);
__attribute__((used)) static auto* _fp8g_1a = &KERNEL_TYPE_FP8_GROUPED(192, 4, 1, 128, 2, false);
__attribute__((used)) static auto* _fp8g_1b = &KERNEL_TYPE_FP8_GROUPED(256, 3, 1, 128, 2, false);
__attribute__((used)) static auto* _fp8g_1c = &KERNEL_TYPE_FP8_GROUPED(256, 3, 1, 128, 2, true);

// ── FP8 kernel dispatch ─────────────────────────────────────────────

static const void* get_fp8_kernel(const FP8Config& cfg) {
    #define MATCH_FP8(BM, BN, ST, NLS, NM, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_last_stages == NLS && cfg.num_math_threads == NM && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_FP8(BM, BN, ST, NLS, NM, SD, MC, MCA);

    MATCH_FP8(16, 16, 32, 0, 128, 32, 1, false)
    MATCH_FP8(16, 32, 32, 0, 128, 64, 1, false)
    MATCH_FP8(16, 48, 27, 1, 128, 32, 1, false)
    MATCH_FP8(16, 64, 22, 0, 128, 128, 1, false)
    MATCH_FP8(16, 80, 18, 2, 128, 32, 1, false)
    MATCH_FP8(16, 96, 15, 3, 128, 64, 1, false)
    MATCH_FP8(16, 112, 13, 7, 128, 32, 1, false)
    MATCH_FP8(16, 256, 6, 1, 128, 128, 1, false)
    MATCH_FP8(32, 16, 32, 0, 128, 32, 1, false)
    MATCH_FP8(32, 32, 27, 0, 128, 64, 1, false)
    MATCH_FP8(32, 48, 21, 1, 128, 32, 1, false)
    MATCH_FP8(32, 48, 22, 1, 128, 32, 1, false)
    MATCH_FP8(32, 64, 18, 0, 128, 128, 1, false)
    MATCH_FP8(32, 80, 15, 2, 128, 32, 1, false)
    MATCH_FP8(32, 96, 13, 3, 128, 64, 1, false)
    MATCH_FP8(32, 112, 12, 7, 128, 32, 1, false)
    MATCH_FP8(32, 256, 5, 1, 128, 128, 1, false)
    MATCH_FP8(64, 16, 21, 0, 128, 32, 1, false)
    MATCH_FP8(64, 32, 18, 0, 128, 64, 1, false)
    MATCH_FP8(64, 48, 15, 1, 128, 32, 1, false)
    MATCH_FP8(64, 64, 13, 0, 128, 128, 1, false)
    MATCH_FP8(64, 64, 13, 0, 128, 128, 2, false)
    MATCH_FP8(64, 80, 11, 2, 128, 32, 1, false)
    MATCH_FP8(64, 96, 10, 3, 128, 64, 1, false)
    MATCH_FP8(64, 112, 9, 7, 128, 32, 1, false)
    MATCH_FP8(64, 128, 8, 0, 128, 128, 1, false)
    MATCH_FP8(64, 128, 8, 0, 128, 128, 2, false)
    MATCH_FP8(64, 144, 7, 1, 128, 32, 1, false)
    MATCH_FP8(64, 144, 7, 1, 128, 32, 2, false)
    MATCH_FP8(64, 160, 7, 1, 128, 64, 1, false)
    MATCH_FP8(64, 160, 7, 1, 128, 64, 2, false)
    MATCH_FP8(64, 192, 6, 1, 128, 128, 1, false)
    MATCH_FP8(64, 192, 6, 1, 128, 128, 2, false)
    MATCH_FP8(64, 256, 4, 1, 128, 128, 1, false)
    MATCH_FP8(64, 256, 4, 1, 128, 128, 2, false)
    MATCH_FP8(64, 256, 4, 1, 128, 128, 2, true)
    MATCH_FP8(128, 112, 6, 7, 256, 32, 1, false)
    MATCH_FP8(128, 144, 5, 1, 256, 32, 1, false)
    MATCH_FP8(128, 144, 5, 1, 256, 32, 2, false)
    MATCH_FP8(128, 160, 5, 1, 256, 64, 2, false)
    MATCH_FP8(128, 192, 4, 1, 256, 128, 1, false)
    MATCH_FP8(128, 192, 4, 1, 256, 128, 2, false)
    MATCH_FP8(128, 256, 3, 1, 256, 128, 1, false)
    MATCH_FP8(128, 256, 3, 1, 256, 128, 2, false)
    MATCH_FP8(128, 256, 3, 1, 256, 128, 2, true)

    #undef MATCH_FP8
    return nullptr;
}

// ── FP8 grouped kernel dispatch ─────────────────────────────────────

static const void* get_fp8_grouped_kernel(const FP8Config& cfg) {
    #define MATCH_FP8G(BN, ST, NLS, SD, MC, MCA) \
        if (cfg.block_n == BN && cfg.num_stages == ST && cfg.num_last_stages == NLS && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_FP8_GROUPED(BN, ST, NLS, SD, MC, MCA);

    MATCH_FP8G(16, 12, 0, 32, 1, false)
    MATCH_FP8G(32, 10, 0, 64, 1, false)
    MATCH_FP8G(48, 9, 1, 32, 1, false)
    MATCH_FP8G(64, 8, 0, 128, 1, false)
    MATCH_FP8G(80, 7, 2, 32, 1, false)
    MATCH_FP8G(96, 7, 3, 64, 1, false)
    MATCH_FP8G(112, 6, 7, 32, 1, false)
    MATCH_FP8G(128, 5, 0, 128, 1, false)
    MATCH_FP8G(144, 5, 1, 32, 1, false)
    MATCH_FP8G(160, 5, 1, 64, 1, false)
    MATCH_FP8G(192, 4, 1, 128, 1, false)
    MATCH_FP8G(256, 3, 1, 128, 1, false)
    MATCH_FP8G(16, 12, 0, 32, 2, true)
    MATCH_FP8G(32, 10, 0, 64, 2, true)
    MATCH_FP8G(48, 9, 1, 32, 2, true)
    MATCH_FP8G(64, 8, 0, 128, 2, true)
    MATCH_FP8G(80, 7, 2, 32, 2, true)
    MATCH_FP8G(96, 7, 3, 64, 2, true)
    MATCH_FP8G(112, 6, 7, 32, 2, false)
    MATCH_FP8G(128, 5, 0, 128, 2, false)
    MATCH_FP8G(144, 5, 1, 32, 2, false)
    MATCH_FP8G(160, 5, 1, 64, 2, false)
    MATCH_FP8G(192, 4, 1, 128, 2, false)
    MATCH_FP8G(256, 3, 1, 128, 2, false)
    MATCH_FP8G(256, 3, 1, 128, 2, true)

    #undef MATCH_FP8G
    return nullptr;
}

// ── SM90 FP8 implementation functions ──────────────────────────────

static int sm90_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    int M, int N, int K, void* stream
) {
    cudaGetLastError();
    auto cfg = select_fp8_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_fp8_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_u8(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);

    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };
    int sfa_inner = align4(M);
    int k_scales = (K + 127) / 128;
    auto tma_sfa = make_2d_tma_f32(scale_a, sfa_inner, k_scales, cfg.block_m, 1, sfa_inner, 0);

    float* sfb_ptr = (float*)scale_b;
    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &sfb_ptr, &grouped_layout, &um, &un, &uk, &tma_a, &tma_b, &tma_d, &tma_sfa };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

static int sm90_m_grouped_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    void* grouped_layout,
    int M, int N, int K, int num_groups, void* stream
) {
    cudaGetLastError();
    auto cfg = select_fp8_grouped_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_fp8_grouped_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_u8(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);

    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };
    int sfa_inner = align4(M);
    int k_scales = (K + 127) / 128;
    auto tma_sfa = make_2d_tma_f32(scale_a, sfa_inner, k_scales, cfg.block_m, 1, sfa_inner, 0);

    float* sfb_ptr = (float*)scale_b;
    int* layout_ptr = (int*)grouped_layout;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &sfb_ptr, &layout_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_d, &tma_sfa };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

// ════════════════════════════════════════════════════════════════════
// M-Grouped Masked FP8 GEMM (1D2D)
// A[G,M,K] FP8 @ B[G,N,K] FP8 → D[G,M,N] BF16, with masked_m[G].
// Same 1D2D kernel with GemmType::MGroupedMasked.
// ════════════════════════════════════════════════════════════════════

// ── Masked FP8 heuristic ───────────────────────────────────────────

static FP8Config select_fp8_masked_config(int expected_m, int n, int k, int num_groups, int num_sms) {
    const int block_k = 128;
    // Masked: block_m = {64, 128}
    int block_ms[2] = {64, 128};
    int n_block_ms = 2;

    int block_ns[12]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16)
        if (fp8_valid_block_n(i)) block_ns[n_block_ns++] = i;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm > 128 && bn > 128) continue;
            int num_blocks = ceil_div(expected_m, bm) * ceil_div(n, bn) * num_groups;
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

            bool better = false;
            if (best_bm == 0 || waves < best_waves) better = true;
            else if (waves == best_waves) {
                better = last_util > best_last;
                if (last_util == best_last) {
                    better |= (bm == best_bm && bn < best_bn);
                    better |= (bn == best_bn && bm < best_bm);
                    better |= (bm != best_bm && bn > best_bn && bn <= n && bm <= expected_m);
                }
            }
            if (better) { best_bm = bm; best_bn = bn; best_waves = waves; best_last = last_util; }
        }
    }

    int num_tma = 128, num_math = (best_bm <= 64) ? 128 : 256;

    // Masked multicast: stricter
    int multicast = 1; bool mc_on_a = false;
    if (expected_m >= 512 && num_sms % 2 == 0) {
        bool n_div = (n % best_bn == 0);
        bool n_even = (ceil_div(n, best_bn) % 2 == 0);
        bool m_even = (ceil_div(expected_m, best_bm) % 2 == 0);
        bool legal_mc_b = n_even && n_div;
        bool legal_mc_a = m_even && n_even && n_div;
        if (best_bm > best_bn) {
            if (legal_mc_a) { multicast = 2; mc_on_a = true; }
            else if (legal_mc_b) { multicast = 2; mc_on_a = false; }
        } else {
            if (legal_mc_b) { multicast = 2; mc_on_a = false; }
            else if (legal_mc_a) { multicast = 2; mc_on_a = true; }
        }
    }

    int sw_a = 128, sw_b = 128;
    int sw_d = get_swizzle(best_bn);

    int nls;
    if (block_k % best_bn == 0) nls = 0;
    else if (best_bn <= block_k) nls = ceil_div(best_bn, block_k - best_bn);
    else nls = 1;

    const int smem_capacity = 232448;
    int smem_d = ((best_bm * best_bn * 2 + 1023) / 1024) * 1024;
    int smem_a_per = best_bm * block_k;
    int smem_b_per = best_bn * block_k;
    int smem_sfa_per = ((best_bm * 4 + 127) / 128) * 128;
    int smem_per_stage = smem_a_per + smem_b_per + smem_sfa_per;

    int k_scales = ceil_div(k, block_k);
    bool must_uniform = (block_k % best_bn == 0);
    int sfb_buf = ((k_scales * (must_uniform ? 1 : 2) * 4 + 15) / 16) * 16;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int barrier = s * 16;
        int total = smem_d + s * smem_per_stage + barrier + sfb_buf;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return FP8Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages, .num_last_stages = nls,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b, .swizzle_d = sw_d,
        .smem_size = best_smem,
    };
}

// ── Masked FP8 kernel instantiations ───────────────────────────────

// FP8 masked kernel: kNumGroups must match actual num_groups
#define KERNEL_TYPE_FP8_MASKED(BLOCK_M, BLOCK_N, STAGES, LAST_STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A, NGROUPS) \
    deep_gemm::sm90_fp8_gemm_1d2d_impl<                                       \
        cute::UMMA::Major::MN,                                                 \
        0, 0, 0, NGROUPS,                                                      \
        BLOCK_M, BLOCK_N, 128,                                                 \
        128, 128, SWIZZLE_D,                                                   \
        STAGES, LAST_STAGES,                                                   \
        128, NUM_MATH,                                                         \
        NUM_MC, MC_ON_A, 132,                                                  \
        GemmType::MGroupedMasked,                                              \
        deep_gemm::EpilogueIdentity>

// Core FP8 masked configs for each num_groups (no multicast).
// bm=64 (math=128): bn ∈ {16,32,64,128,256}
// bm=128 (math=256): bn ∈ {16,32,64,128,256}
#define FP8_MASKED_CONFIGS(G, TAG) \
    __attribute__((used)) static auto* _fp8m_##TAG##_00 = &KERNEL_TYPE_FP8_MASKED(64, 16, 21, 0, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_01 = &KERNEL_TYPE_FP8_MASKED(64, 32, 18, 0, 128, 64, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_02 = &KERNEL_TYPE_FP8_MASKED(64, 48, 15, 1, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_03 = &KERNEL_TYPE_FP8_MASKED(64, 64, 13, 0, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_04 = &KERNEL_TYPE_FP8_MASKED(64, 96, 10, 3, 128, 64, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_05 = &KERNEL_TYPE_FP8_MASKED(64, 128, 8, 0, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_06 = &KERNEL_TYPE_FP8_MASKED(64, 256, 4, 1, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_10 = &KERNEL_TYPE_FP8_MASKED(128, 16, 12, 0, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_11 = &KERNEL_TYPE_FP8_MASKED(128, 32, 10, 0, 256, 64, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_12 = &KERNEL_TYPE_FP8_MASKED(128, 64, 8, 0, 256, 128, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_13 = &KERNEL_TYPE_FP8_MASKED(128, 128, 5, 0, 256, 128, 1, false, G); \
    __attribute__((used)) static auto* _fp8m_##TAG##_14 = &KERNEL_TYPE_FP8_MASKED(128, 256, 3, 1, 256, 128, 1, false, G);

FP8_MASKED_CONFIGS(2,  g02)
FP8_MASKED_CONFIGS(4,  g04)
FP8_MASKED_CONFIGS(8,  g08)
FP8_MASKED_CONFIGS(16, g16)
FP8_MASKED_CONFIGS(32, g32)
FP8_MASKED_CONFIGS(64, g64)

// ── Masked FP8 kernel dispatch ─────────────────────────────────────

static const void* get_fp8_masked_kernel(const FP8Config& cfg, int num_groups) {
    #define MATCH_FP8M(BM, BN, ST, NLS, NM, SD, G) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_last_stages == NLS && cfg.num_math_threads == NM && \
            cfg.swizzle_d == SD && cfg.num_multicast == 1 && num_groups == G) \
            return (const void*)&KERNEL_TYPE_FP8_MASKED(BM, BN, ST, NLS, NM, SD, 1, false, G);

    #define MATCH_FP8M_ALL_G(BM, BN, ST, NLS, NM, SD) \
        MATCH_FP8M(BM, BN, ST, NLS, NM, SD, 2)  \
        MATCH_FP8M(BM, BN, ST, NLS, NM, SD, 4)  \
        MATCH_FP8M(BM, BN, ST, NLS, NM, SD, 8)  \
        MATCH_FP8M(BM, BN, ST, NLS, NM, SD, 16) \
        MATCH_FP8M(BM, BN, ST, NLS, NM, SD, 32) \
        MATCH_FP8M(BM, BN, ST, NLS, NM, SD, 64)

    // bm=64 (math=128)
    MATCH_FP8M_ALL_G(64, 16, 21, 0, 128, 32)
    MATCH_FP8M_ALL_G(64, 32, 18, 0, 128, 64)
    MATCH_FP8M_ALL_G(64, 48, 15, 1, 128, 32)
    MATCH_FP8M_ALL_G(64, 64, 13, 0, 128, 128)
    MATCH_FP8M_ALL_G(64, 96, 10, 3, 128, 64)
    MATCH_FP8M_ALL_G(64, 128, 8, 0, 128, 128)
    MATCH_FP8M_ALL_G(64, 256, 4, 1, 128, 128)
    // bm=128 (math=256)
    MATCH_FP8M_ALL_G(128, 16, 12, 0, 256, 32)
    MATCH_FP8M_ALL_G(128, 32, 10, 0, 256, 64)
    MATCH_FP8M_ALL_G(128, 64, 8, 0, 256, 128)
    MATCH_FP8M_ALL_G(128, 128, 5, 0, 256, 128)
    MATCH_FP8M_ALL_G(128, 256, 3, 1, 256, 128)

    #undef MATCH_FP8M_ALL_G
    #undef MATCH_FP8M
    return nullptr;
}

// ── Masked FP8 implementation ──────────────────────────────────────

static int sm90_m_grouped_masked_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    void* masked_m,
    int M, int N, int K, int num_groups, int expected_m, void* stream
) {
    cudaGetLastError();
    auto cfg = select_fp8_masked_config(expected_m, N, K, num_groups, g_num_sms);
    auto kernel_ptr = get_fp8_masked_kernel(cfg, num_groups);
    if (!kernel_ptr) return -1;

    // Masked: A[G,M,K] FP8, outer = M * num_groups
    auto tma_a = make_2d_tma_u8(A, K, M * num_groups, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    // B[G,N,K] FP8, outer = N * num_groups
    auto tma_b = make_2d_tma_u8(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // D[G,M,N] BF16, outer = M * num_groups
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M * num_groups, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);
    // SFA: [ceil(K/128), align(M,4)] per group → outer = ceil(K/128) * num_groups
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };
    int sfa_inner = align4(M);
    int k_scales = (K + 127) / 128;
    auto tma_sfa = make_2d_tma_f32(scale_a, sfa_inner, k_scales * num_groups,
                                    cfg.block_m, 1, sfa_inner, 0);

    float* sfb_ptr = (float*)scale_b;
    int* mask_ptr = (int*)masked_m;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &sfb_ptr, &mask_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_d, &tma_sfa };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

// ════════════════════════════════════════════════════════════════════
// SM90 FP8 1D1D GEMM — per-block scaling on both A and B via TMA.
// FP32 output. Different from 1D2D (per-token A via TMA, per-channel B global).
// ════════════════════════════════════════════════════════════════════

struct FP8_1D1D_Config {
    int block_m, block_n, block_k;
    int num_stages;
    int num_tma_threads, num_math_threads;
    int num_multicast;
    bool multicast_on_a;
    int swizzle_a, swizzle_b;
    int smem_size;
};

static FP8_1D1D_Config select_fp8_1d1d_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;

    // 1D1D: block_m = {64, 128, 256} (no small M candidates)
    // block_m=256 is illegal for FP32 output (upstream constraint)
    int block_ms[2] = {64, 128};
    int n_block_ms = 2;

    // block_n: standard multiples of 16, up to 128
    // (1D1D FP32 has max bn ~152, but we stay conservative at 128)
    int block_ns[8]; int n_block_ns = 0;
    for (int i = 16; i <= 128; i += 16) block_ns[n_block_ns++] = i;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm > 128 && bn > 128) continue;
            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

            bool better = false;
            if (best_bm == 0 || waves < best_waves) better = true;
            else if (waves == best_waves) {
                better = last_util > best_last;
                if (last_util == best_last) {
                    better |= (bm == best_bm && bn < best_bn);
                    better |= (bn == best_bn && bm < best_bm);
                    better |= (bm != best_bm && bn > best_bn && bn <= n && bm <= m);
                }
            }
            if (better) { best_bm = bm; best_bn = bn; best_waves = waves; best_last = last_util; }
        }
    }

    int num_tma = 128, num_math = (best_bm <= 64) ? 128 : 256;

    // Multicast (same rules as normal BF16)
    int multicast = 1; bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, best_bm) % 2 == 0);
        bool order[2] = {false, true};
        if (best_bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) { multicast = 2; mc_on_a = order[i]; break; }
        }
    }

    int sw_a = 128, sw_b = 128; // block_k=128, FP8 1-byte → 128B swizzle

    // SMEM: FP32 output (no swizzle), both SFA and SFB per stage
    const int smem_capacity = 232448;
    int smem_cd = ((best_bm * best_bn * 4 + 1023) / 1024) * 1024; // FP32 output
    int smem_a_per = best_bm * block_k; // FP8: 1 byte
    int smem_b_per = best_bn * block_k;
    int smem_sfa_per = ((best_bm * 4 + 127) / 128) * 128;
    int smem_sfb_per = ((best_bn * 4 + 127) / 128) * 128;

    int best_stages = 0, best_smem = 0;
    for (int s = 32; s > 0; s--) {
        int barrier = s * 16;
        int total = smem_cd + s * (smem_a_per + smem_b_per + smem_sfa_per + smem_sfb_per) + barrier;
        if (total <= smem_capacity) { best_stages = s; best_smem = total; break; }
    }

    return FP8_1D1D_Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = sw_a, .swizzle_b = sw_b,
        .smem_size = best_smem,
    };
}

// 1D1D kernel: 17 template params, cd_dtype=float, no swizzle_cd
#define KERNEL_TYPE_FP8_1D1D(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_fp8_gemm_1d1d_impl<                                     \
        0, 0, 0, 1,                                                         \
        BLOCK_M, BLOCK_N, 128,                                              \
        128, 128,                                                           \
        STAGES, 128, NUM_MATH,                                              \
        NUM_MC, MC_ON_A, 132,                                               \
        GemmType::Normal, float>

// Stages computed: smem_cd=align(bm*bn*4,1024), per_stage=bm*128+bn*128+align(bm*4,128)+align(bn*4,128)+16
__attribute__((used)) static auto* _1d1d_00 = &KERNEL_TYPE_FP8_1D1D(64, 16, 21, 128, 1, false);
__attribute__((used)) static auto* _1d1d_01 = &KERNEL_TYPE_FP8_1D1D(64, 32, 17, 128, 1, false);
__attribute__((used)) static auto* _1d1d_02 = &KERNEL_TYPE_FP8_1D1D(64, 48, 14, 128, 1, false);
__attribute__((used)) static auto* _1d1d_03 = &KERNEL_TYPE_FP8_1D1D(64, 64, 12, 128, 1, false);
__attribute__((used)) static auto* _1d1d_04 = &KERNEL_TYPE_FP8_1D1D(64, 80, 11, 128, 1, false);
__attribute__((used)) static auto* _1d1d_05 = &KERNEL_TYPE_FP8_1D1D(64, 96, 9, 128, 1, false);
__attribute__((used)) static auto* _1d1d_06 = &KERNEL_TYPE_FP8_1D1D(64, 112, 8, 128, 1, false);
__attribute__((used)) static auto* _1d1d_07 = &KERNEL_TYPE_FP8_1D1D(64, 128, 7, 128, 1, false);
__attribute__((used)) static auto* _1d1d_10 = &KERNEL_TYPE_FP8_1D1D(128, 16, 11, 256, 1, false);
__attribute__((used)) static auto* _1d1d_11 = &KERNEL_TYPE_FP8_1D1D(128, 32, 10, 256, 1, false);
__attribute__((used)) static auto* _1d1d_12 = &KERNEL_TYPE_FP8_1D1D(128, 48, 8, 256, 1, false);
__attribute__((used)) static auto* _1d1d_13 = &KERNEL_TYPE_FP8_1D1D(128, 64, 7, 256, 1, false);
__attribute__((used)) static auto* _1d1d_14 = &KERNEL_TYPE_FP8_1D1D(128, 80, 6, 256, 1, false);
__attribute__((used)) static auto* _1d1d_15 = &KERNEL_TYPE_FP8_1D1D(128, 96, 6, 256, 1, false);
__attribute__((used)) static auto* _1d1d_16 = &KERNEL_TYPE_FP8_1D1D(128, 112, 5, 256, 1, false);
__attribute__((used)) static auto* _1d1d_17 = &KERNEL_TYPE_FP8_1D1D(128, 128, 4, 256, 1, false);

static const void* get_fp8_1d1d_kernel(const FP8_1D1D_Config& cfg) {
    #define MATCH_1D1D(BM, BN, ST, NM, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_math_threads == NM && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_FP8_1D1D(BM, BN, ST, NM, MC, MCA);

    MATCH_1D1D(64, 16, 21, 128, 1, false) MATCH_1D1D(64, 32, 17, 128, 1, false)
    MATCH_1D1D(64, 48, 14, 128, 1, false) MATCH_1D1D(64, 64, 12, 128, 1, false)
    MATCH_1D1D(64, 80, 11, 128, 1, false) MATCH_1D1D(64, 96, 9, 128, 1, false)
    MATCH_1D1D(64, 112, 8, 128, 1, false) MATCH_1D1D(64, 128, 7, 128, 1, false)
    MATCH_1D1D(128, 16, 11, 256, 1, false) MATCH_1D1D(128, 32, 10, 256, 1, false)
    MATCH_1D1D(128, 48, 8, 256, 1, false) MATCH_1D1D(128, 64, 7, 256, 1, false)
    MATCH_1D1D(128, 80, 6, 256, 1, false) MATCH_1D1D(128, 96, 6, 256, 1, false)
    MATCH_1D1D(128, 112, 5, 256, 1, false) MATCH_1D1D(128, 128, 4, 256, 1, false)

    #undef MATCH_1D1D
    return nullptr;
}

/// SM90 FP8 1D1D GEMM: D(FP32) = A(FP8) @ B(FP8) with per-block scaling on both.
/// scale_a: [ceil(K/128), align(M, 4)] FP32, MN-major (per-block, loaded via TMA)
/// scale_b: [ceil(K/128), align(N, 4)] FP32, MN-major (per-block, loaded via TMA)
/// D: [M, N] FP32 output
static int sm90_fp8_gemm_1d1d(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    int M, int N, int K, void* stream
) {
    cudaGetLastError();
    auto cfg = select_fp8_1d1d_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_fp8_1d1d_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };

    // TMA for A/B (FP8, K-major)
    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_u8(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // TMA for D (FP32, no swizzle on SM90)
    auto tma_d = make_2d_tma_f32(D, N, M, cfg.block_n, 64, N, 0); // cd_store_block_m=64 for 1D1D
    // TMA for SFA: [ceil(K/128), align(M,4)] FP32, MN-major
    int sfa_inner = align4(M);
    int sfb_inner = align4(N);
    int k_scales = ceil_div(K, 128);
    auto tma_sfa = make_2d_tma_f32(scale_a, sfa_inner, k_scales, cfg.block_m, 1, sfa_inner, 0);
    auto tma_sfb = make_2d_tma_f32(scale_b, sfb_inner, k_scales, cfg.block_n, 1, sfb_inner, 0);

    // 1D1D kernel args: (gmem_a, gmem_b, grouped_layout, tensor_map_buffer, m, n, k, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd)
    // For Normal GEMM: gmem_a/b and tensor_map_buffer are nullptr
    __nv_fp8_e4m3* null_fp8 = nullptr;
    int* null_layout = nullptr;
    CUtensorMap* null_tmap = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = {
        &null_fp8, &null_fp8, &null_layout, &null_tmap,
        &um, &un, &uk,
        &tma_a, &tma_b, &tma_sfa, &tma_sfb, &tma_d
    };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}
