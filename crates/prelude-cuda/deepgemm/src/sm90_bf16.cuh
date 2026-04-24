// SM90 BF16 GEMM: Normal + M-Grouped Contiguous
// Heuristics, kernel instantiations, dispatch, and implementation.

#pragma once

// ── SM90 BF16 heuristic ────────────────────────────────────────────

static KernelConfig select_config(int m, int n, int k, int num_sms) {
    // block_k is fixed at 64 for BF16
    const int block_k = 64;

    // block_m candidates: {64, 128, 256} + optional {16, 32} for small M
    // (matches DeepGEMM sm90.hpp get_block_m_candidates exactly)
    int block_ms[5] = {64, 128, 256, 0, 0};
    int n_block_ms = 3;
    if (m <= 16) { block_ms[n_block_ms++] = 16; }
    if (m <= 32) { block_ms[n_block_ms++] = 32; }

    // block_n candidates: 16, 32, 48, ..., 256
    int block_ns[16];
    int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16) block_ns[n_block_ns++] = i;

    // Select by wave utilization (same logic as DeepGEMM heuristic)
    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

            // SM90 BF16: at least one of block_m, block_n must be <= 128
            if (bm > 128 && bn > 128)
                continue;

            bool better = false;
            if (best_bm == 0 || waves < best_waves) {
                better = true;
            } else if (waves == best_waves) {
                better = last_util > best_last;
                if (last_util == best_last) {
                    better |= (bm == best_bm && bn < best_bn);
                    better |= (bn == best_bn && bm < best_bm);
                    better |= (bm != best_bm && bn > best_bn && bn <= n && bm <= m);
                }
            }
            if (better) {
                best_bm = bm; best_bn = bn;
                best_waves = waves; best_last = last_util;
            }
        }
    }

    // Thread config: 128 TMA + 128 or 256 math
    int num_tma = 128;
    int num_math = (best_bm <= 64) ? 128 : 256;

    // Multicast: only for M >= 512, check divisibility.
    int multicast = 1;
    bool mc_on_a = false;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal_on_b = (ceil_div(n, best_bn) % 2 == 0);
        bool legal_on_a = (ceil_div(m, best_bm) % 2 == 0);
        bool order[2] = {false, true}; // {on_b, on_a}
        if (best_bm > best_bn) { order[0] = true; order[1] = false; }
        bool legal[2] = {legal_on_a, legal_on_b};
        for (int i = 0; i < 2; i++) {
            if (legal[order[i] ? 1 : 0]) {
                multicast = 2;
                mc_on_a = order[i];
                break;
            }
        }
    }

    // Stages and swizzle computed by upstream-equivalent smem formula
    SmemConfig scfg;
    int best_stages = select_num_stages<SM90Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        best_bm, best_bn, block_k, multicast, mc_on_a,
        2, 2, // ab_elem=BF16(2), cd_elem=BF16(2)
        m, n, 0, scfg);

    return KernelConfig{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// ── Grouped BF16 heuristic ─────────────────────────────────────────
// M-Grouped Contiguous BF16 GEMM (MoE). block_m fixed to 128.

static KernelConfig select_grouped_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;
    const int bm = 128;

    int block_ns[16]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16) block_ns[n_block_ns++] = i;

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
            if (last_util == best_last)
                better |= (bn < best_bn);
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

    SmemConfig scfg;
    int best_stages = select_num_stages<SM90Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        bm, best_bn, block_k, multicast, mc_on_a,
        2, 2, m, n, 0, scfg);

    return KernelConfig{
        .block_m = bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// ── Normal BF16 kernel instantiations ──────────────────────────────

#define KERNEL_TYPE(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, 1,                                                        \
        BLOCK_M, BLOCK_N, 64,                                              \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, NUM_MATH,                                             \
        NUM_MC, MC_ON_A, 132,                                              \
        GemmType::Normal, false, cutlass::bfloat16_t>

__attribute__((used)) static auto* _k00 = &KERNEL_TYPE(16, 16, 32, 128, 32, 1, false);
__attribute__((used)) static auto* _k01 = &KERNEL_TYPE(16, 32, 32, 128, 64, 1, false);
__attribute__((used)) static auto* _k02 = &KERNEL_TYPE(16, 48, 28, 128, 32, 1, false);
__attribute__((used)) static auto* _k03 = &KERNEL_TYPE(16, 64, 22, 128, 128, 1, false);
__attribute__((used)) static auto* _k04 = &KERNEL_TYPE(16, 96, 15, 128, 64, 1, false);
__attribute__((used)) static auto* _k05 = &KERNEL_TYPE(16, 112, 13, 128, 32, 1, false);
__attribute__((used)) static auto* _k06 = &KERNEL_TYPE(16, 224, 7, 128, 64, 1, false);
__attribute__((used)) static auto* _k07 = &KERNEL_TYPE(16, 240, 6, 128, 32, 1, false);
__attribute__((used)) static auto* _k08 = &KERNEL_TYPE(32, 16, 32, 128, 32, 1, false);
__attribute__((used)) static auto* _k09 = &KERNEL_TYPE(32, 32, 28, 128, 64, 1, false);
__attribute__((used)) static auto* _k10 = &KERNEL_TYPE(32, 48, 22, 128, 32, 1, false);
__attribute__((used)) static auto* _k11 = &KERNEL_TYPE(32, 64, 18, 128, 128, 1, false);
__attribute__((used)) static auto* _k12 = &KERNEL_TYPE(32, 96, 13, 128, 64, 1, false);
__attribute__((used)) static auto* _k13 = &KERNEL_TYPE(32, 112, 12, 128, 32, 1, false);
__attribute__((used)) static auto* _k14 = &KERNEL_TYPE(32, 224, 6, 128, 64, 1, false);
__attribute__((used)) static auto* _k15 = &KERNEL_TYPE(32, 240, 6, 128, 32, 1, false);
__attribute__((used)) static auto* _k16 = &KERNEL_TYPE(64, 16, 22, 128, 32, 1, false);
__attribute__((used)) static auto* _k17 = &KERNEL_TYPE(64, 32, 18, 128, 64, 1, false);
__attribute__((used)) static auto* _k18 = &KERNEL_TYPE(64, 48, 15, 128, 32, 1, false);
__attribute__((used)) static auto* _k19 = &KERNEL_TYPE(64, 64, 13, 128, 128, 1, false);
__attribute__((used)) static auto* _k20 = &KERNEL_TYPE(64, 64, 13, 128, 128, 2, false);
__attribute__((used)) static auto* _k21 = &KERNEL_TYPE(64, 80, 12, 128, 32, 1, false);
__attribute__((used)) static auto* _k22 = &KERNEL_TYPE(64, 80, 12, 128, 32, 2, true);
__attribute__((used)) static auto* _k23 = &KERNEL_TYPE(64, 96, 10, 128, 64, 1, false);
__attribute__((used)) static auto* _k24 = &KERNEL_TYPE(64, 96, 10, 128, 64, 2, true);
__attribute__((used)) static auto* _k25 = &KERNEL_TYPE(64, 112, 9, 128, 32, 1, false);
__attribute__((used)) static auto* _k26 = &KERNEL_TYPE(64, 112, 9, 128, 32, 2, false);
__attribute__((used)) static auto* _k27 = &KERNEL_TYPE(64, 128, 8, 128, 128, 1, false);
__attribute__((used)) static auto* _k28 = &KERNEL_TYPE(64, 128, 8, 128, 128, 2, false);
__attribute__((used)) static auto* _k29 = &KERNEL_TYPE(64, 144, 8, 128, 32, 1, false);
__attribute__((used)) static auto* _k30 = &KERNEL_TYPE(64, 160, 7, 128, 64, 1, false);
__attribute__((used)) static auto* _k31 = &KERNEL_TYPE(64, 160, 7, 128, 64, 2, true);
__attribute__((used)) static auto* _k32 = &KERNEL_TYPE(64, 176, 6, 128, 32, 1, false);
__attribute__((used)) static auto* _k33 = &KERNEL_TYPE(64, 176, 6, 128, 32, 2, false);
__attribute__((used)) static auto* _k34 = &KERNEL_TYPE(64, 192, 6, 128, 128, 1, false);
__attribute__((used)) static auto* _k35 = &KERNEL_TYPE(64, 192, 6, 128, 128, 2, false);
__attribute__((used)) static auto* _k36 = &KERNEL_TYPE(64, 192, 6, 128, 128, 2, true);
__attribute__((used)) static auto* _k37 = &KERNEL_TYPE(64, 208, 5, 128, 32, 2, false);
__attribute__((used)) static auto* _k38 = &KERNEL_TYPE(64, 208, 5, 128, 32, 2, true);
__attribute__((used)) static auto* _k39 = &KERNEL_TYPE(64, 224, 5, 128, 64, 1, false);
__attribute__((used)) static auto* _k40 = &KERNEL_TYPE(64, 224, 5, 128, 64, 2, false);
__attribute__((used)) static auto* _k41 = &KERNEL_TYPE(64, 240, 5, 128, 32, 1, false);
__attribute__((used)) static auto* _k42 = &KERNEL_TYPE(64, 240, 5, 128, 32, 2, false);
__attribute__((used)) static auto* _k43 = &KERNEL_TYPE(64, 240, 5, 128, 32, 2, true);
__attribute__((used)) static auto* _k44 = &KERNEL_TYPE(64, 256, 4, 128, 128, 1, false);
__attribute__((used)) static auto* _k45 = &KERNEL_TYPE(64, 256, 4, 128, 128, 2, false);
__attribute__((used)) static auto* _k46 = &KERNEL_TYPE(64, 256, 4, 128, 128, 2, true);
__attribute__((used)) static auto* _k47 = &KERNEL_TYPE(128, 16, 12, 256, 32, 1, false);
__attribute__((used)) static auto* _k48 = &KERNEL_TYPE(128, 32, 10, 256, 64, 1, false);
__attribute__((used)) static auto* _k49 = &KERNEL_TYPE(128, 48, 9, 256, 32, 1, false);
__attribute__((used)) static auto* _k50 = &KERNEL_TYPE(128, 48, 9, 256, 32, 2, true);
__attribute__((used)) static auto* _k51 = &KERNEL_TYPE(128, 64, 8, 256, 128, 1, false);
__attribute__((used)) static auto* _k52 = &KERNEL_TYPE(128, 64, 8, 256, 128, 2, true);
__attribute__((used)) static auto* _k53 = &KERNEL_TYPE(128, 80, 7, 256, 32, 1, false);
__attribute__((used)) static auto* _k54 = &KERNEL_TYPE(128, 80, 7, 256, 32, 2, false);
__attribute__((used)) static auto* _k55 = &KERNEL_TYPE(128, 80, 7, 256, 32, 2, true);
__attribute__((used)) static auto* _k56 = &KERNEL_TYPE(128, 96, 7, 256, 64, 1, false);
__attribute__((used)) static auto* _k57 = &KERNEL_TYPE(128, 96, 7, 256, 64, 2, true);
__attribute__((used)) static auto* _k58 = &KERNEL_TYPE(128, 112, 6, 256, 32, 2, false);
__attribute__((used)) static auto* _k59 = &KERNEL_TYPE(128, 128, 6, 256, 128, 1, false);
__attribute__((used)) static auto* _k60 = &KERNEL_TYPE(128, 128, 6, 256, 128, 2, false);
__attribute__((used)) static auto* _k61 = &KERNEL_TYPE(128, 144, 5, 256, 32, 1, false);
__attribute__((used)) static auto* _k62 = &KERNEL_TYPE(128, 144, 5, 256, 32, 2, false);
__attribute__((used)) static auto* _k63 = &KERNEL_TYPE(128, 160, 5, 256, 64, 1, false);
__attribute__((used)) static auto* _k64 = &KERNEL_TYPE(128, 160, 5, 256, 64, 2, false);
__attribute__((used)) static auto* _k65 = &KERNEL_TYPE(128, 160, 5, 256, 64, 2, true);
__attribute__((used)) static auto* _k66 = &KERNEL_TYPE(128, 176, 4, 256, 32, 1, false);
__attribute__((used)) static auto* _k67 = &KERNEL_TYPE(128, 176, 4, 256, 32, 2, false);
__attribute__((used)) static auto* _k68 = &KERNEL_TYPE(128, 176, 4, 256, 32, 2, true);
__attribute__((used)) static auto* _k69 = &KERNEL_TYPE(128, 192, 4, 256, 128, 1, false);
__attribute__((used)) static auto* _k70 = &KERNEL_TYPE(128, 192, 4, 256, 128, 2, false);
__attribute__((used)) static auto* _k71 = &KERNEL_TYPE(128, 192, 4, 256, 128, 2, true);
__attribute__((used)) static auto* _k72 = &KERNEL_TYPE(128, 208, 4, 256, 32, 1, false);
__attribute__((used)) static auto* _k73 = &KERNEL_TYPE(128, 208, 4, 256, 32, 2, false);
__attribute__((used)) static auto* _k74 = &KERNEL_TYPE(128, 208, 4, 256, 32, 2, true);
__attribute__((used)) static auto* _k75 = &KERNEL_TYPE(128, 224, 3, 256, 64, 1, false);
__attribute__((used)) static auto* _k76 = &KERNEL_TYPE(128, 224, 3, 256, 64, 2, false);
__attribute__((used)) static auto* _k77 = &KERNEL_TYPE(128, 224, 3, 256, 64, 2, true);
__attribute__((used)) static auto* _k78 = &KERNEL_TYPE(128, 240, 3, 256, 32, 1, false);
__attribute__((used)) static auto* _k79 = &KERNEL_TYPE(128, 240, 3, 256, 32, 2, false);
__attribute__((used)) static auto* _k80 = &KERNEL_TYPE(128, 240, 3, 256, 32, 2, true);
__attribute__((used)) static auto* _k81 = &KERNEL_TYPE(128, 256, 3, 256, 128, 1, false);
__attribute__((used)) static auto* _k82 = &KERNEL_TYPE(128, 256, 3, 256, 128, 2, false);
__attribute__((used)) static auto* _k83 = &KERNEL_TYPE(128, 256, 3, 256, 128, 2, true);
__attribute__((used)) static auto* _k84 = &KERNEL_TYPE(256, 16, 6, 256, 32, 1, false);
__attribute__((used)) static auto* _k85 = &KERNEL_TYPE(256, 32, 5, 256, 64, 2, true);
__attribute__((used)) static auto* _k86 = &KERNEL_TYPE(256, 48, 5, 256, 32, 1, false);
__attribute__((used)) static auto* _k87 = &KERNEL_TYPE(256, 48, 5, 256, 32, 2, true);
__attribute__((used)) static auto* _k88 = &KERNEL_TYPE(256, 64, 4, 256, 128, 2, false);
__attribute__((used)) static auto* _k89 = &KERNEL_TYPE(256, 64, 4, 256, 128, 2, true);
__attribute__((used)) static auto* _k90 = &KERNEL_TYPE(256, 80, 4, 256, 32, 1, false);
__attribute__((used)) static auto* _k91 = &KERNEL_TYPE(256, 80, 4, 256, 32, 2, false);
__attribute__((used)) static auto* _k92 = &KERNEL_TYPE(256, 80, 4, 256, 32, 2, true);
__attribute__((used)) static auto* _k93 = &KERNEL_TYPE(256, 96, 4, 256, 64, 2, true);
__attribute__((used)) static auto* _k94 = &KERNEL_TYPE(256, 112, 3, 256, 32, 1, false);
__attribute__((used)) static auto* _k95 = &KERNEL_TYPE(256, 112, 3, 256, 32, 2, false);
__attribute__((used)) static auto* _k96 = &KERNEL_TYPE(256, 112, 3, 256, 32, 2, true);
__attribute__((used)) static auto* _k97 = &KERNEL_TYPE(256, 128, 3, 256, 128, 2, false);
__attribute__((used)) static auto* _k98 = &KERNEL_TYPE(256, 128, 3, 256, 128, 2, true);
// Additional variants for Gemma vocab=262144, Qwen3-32B intermediate=25600, etc.
__attribute__((used)) static auto* _k99 = &KERNEL_TYPE(16, 80, 18, 128, 32, 1, false);
__attribute__((used)) static auto* _kA0 = &KERNEL_TYPE(16, 208, 7, 128, 32, 1, false);
__attribute__((used)) static auto* _kA1 = &KERNEL_TYPE(16, 256, 6, 128, 128, 1, false);
__attribute__((used)) static auto* _kA2 = &KERNEL_TYPE(64, 144, 8, 128, 32, 2, false);
__attribute__((used)) static auto* _kA3 = &KERNEL_TYPE(64, 160, 7, 128, 64, 2, false);
__attribute__((used)) static auto* _kA4 = &KERNEL_TYPE(64, 208, 5, 128, 32, 1, false);

// ── M-Grouped Contiguous BF16 kernel instantiations ────────────────

#define KERNEL_TYPE_GROUPED(BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, 1,                                                        \
        128, BLOCK_N, 64,                                                   \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, 256,                                                   \
        NUM_MC, MC_ON_A, 132,                                              \
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t>

__attribute__((used)) static auto* _grp_00 = &KERNEL_TYPE_GROUPED(16, 12, 32, 1, false);
__attribute__((used)) static auto* _grp_00a = &KERNEL_TYPE_GROUPED(16, 12, 32, 2, false);
__attribute__((used)) static auto* _grp_00b = &KERNEL_TYPE_GROUPED(16, 12, 32, 2, true);
__attribute__((used)) static auto* _grp_01 = &KERNEL_TYPE_GROUPED(32, 10, 64, 1, false);
__attribute__((used)) static auto* _grp_01a = &KERNEL_TYPE_GROUPED(32, 10, 64, 2, false);
__attribute__((used)) static auto* _grp_01b = &KERNEL_TYPE_GROUPED(32, 10, 64, 2, true);
__attribute__((used)) static auto* _grp_02 = &KERNEL_TYPE_GROUPED(48, 9, 32, 1, false);
__attribute__((used)) static auto* _grp_03 = &KERNEL_TYPE_GROUPED(48, 9, 32, 2, true);
__attribute__((used)) static auto* _grp_04 = &KERNEL_TYPE_GROUPED(64, 8, 128, 1, false);
__attribute__((used)) static auto* _grp_05 = &KERNEL_TYPE_GROUPED(64, 8, 128, 2, true);
__attribute__((used)) static auto* _grp_06 = &KERNEL_TYPE_GROUPED(80, 7, 32, 1, false);
__attribute__((used)) static auto* _grp_07 = &KERNEL_TYPE_GROUPED(80, 7, 32, 2, false);
__attribute__((used)) static auto* _grp_08 = &KERNEL_TYPE_GROUPED(80, 7, 32, 2, true);
__attribute__((used)) static auto* _grp_09 = &KERNEL_TYPE_GROUPED(96, 7, 64, 1, false);
__attribute__((used)) static auto* _grp_0a = &KERNEL_TYPE_GROUPED(96, 7, 64, 2, true);
__attribute__((used)) static auto* _grp_0b = &KERNEL_TYPE_GROUPED(112, 6, 32, 1, false);
__attribute__((used)) static auto* _grp_0b1 = &KERNEL_TYPE_GROUPED(112, 6, 32, 2, false);
__attribute__((used)) static auto* _grp_0c = &KERNEL_TYPE_GROUPED(128, 6, 128, 1, false);
__attribute__((used)) static auto* _grp_0d = &KERNEL_TYPE_GROUPED(128, 6, 128, 2, false);
__attribute__((used)) static auto* _grp_0e = &KERNEL_TYPE_GROUPED(144, 5, 32, 1, false);
__attribute__((used)) static auto* _grp_0f = &KERNEL_TYPE_GROUPED(144, 5, 32, 2, false);
__attribute__((used)) static auto* _grp_10 = &KERNEL_TYPE_GROUPED(160, 5, 64, 1, false);
__attribute__((used)) static auto* _grp_11 = &KERNEL_TYPE_GROUPED(160, 5, 64, 2, false);
__attribute__((used)) static auto* _grp_12 = &KERNEL_TYPE_GROUPED(160, 5, 64, 2, true);
__attribute__((used)) static auto* _grp_13 = &KERNEL_TYPE_GROUPED(176, 4, 32, 1, false);
__attribute__((used)) static auto* _grp_14 = &KERNEL_TYPE_GROUPED(176, 4, 32, 2, false);
__attribute__((used)) static auto* _grp_15 = &KERNEL_TYPE_GROUPED(176, 4, 32, 2, true);
__attribute__((used)) static auto* _grp_16 = &KERNEL_TYPE_GROUPED(192, 4, 128, 1, false);
__attribute__((used)) static auto* _grp_17 = &KERNEL_TYPE_GROUPED(192, 4, 128, 2, false);
__attribute__((used)) static auto* _grp_18 = &KERNEL_TYPE_GROUPED(192, 4, 128, 2, true);
__attribute__((used)) static auto* _grp_19 = &KERNEL_TYPE_GROUPED(208, 4, 32, 1, false);
__attribute__((used)) static auto* _grp_1a = &KERNEL_TYPE_GROUPED(208, 4, 32, 2, false);
__attribute__((used)) static auto* _grp_1b = &KERNEL_TYPE_GROUPED(208, 4, 32, 2, true);
__attribute__((used)) static auto* _grp_1c = &KERNEL_TYPE_GROUPED(224, 3, 64, 1, false);
__attribute__((used)) static auto* _grp_1d = &KERNEL_TYPE_GROUPED(224, 3, 64, 2, false);
__attribute__((used)) static auto* _grp_1e = &KERNEL_TYPE_GROUPED(224, 3, 64, 2, true);
__attribute__((used)) static auto* _grp_1f = &KERNEL_TYPE_GROUPED(240, 3, 32, 1, false);
__attribute__((used)) static auto* _grp_20 = &KERNEL_TYPE_GROUPED(240, 3, 32, 2, false);
__attribute__((used)) static auto* _grp_21 = &KERNEL_TYPE_GROUPED(240, 3, 32, 2, true);
__attribute__((used)) static auto* _grp_22 = &KERNEL_TYPE_GROUPED(256, 3, 128, 1, false);
__attribute__((used)) static auto* _grp_23 = &KERNEL_TYPE_GROUPED(256, 3, 128, 2, false);
__attribute__((used)) static auto* _grp_24 = &KERNEL_TYPE_GROUPED(256, 3, 128, 2, true);

// ── Normal BF16 kernel dispatch ────────────────────────────────────

static const void* get_kernel(const KernelConfig& cfg) {
    #define MATCH(BM, BN, ST, NM, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_math_threads == NM && cfg.swizzle_d == SD && \
            cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE(BM, BN, ST, NM, SD, MC, MCA);

    MATCH(16, 16, 32, 128, 32, 1, false)
    MATCH(16, 32, 32, 128, 64, 1, false)
    MATCH(16, 48, 28, 128, 32, 1, false)
    MATCH(16, 64, 22, 128, 128, 1, false)
    MATCH(16, 96, 15, 128, 64, 1, false)
    MATCH(16, 112, 13, 128, 32, 1, false)
    MATCH(16, 224, 7, 128, 64, 1, false)
    MATCH(16, 240, 6, 128, 32, 1, false)
    MATCH(32, 16, 32, 128, 32, 1, false)
    MATCH(32, 32, 28, 128, 64, 1, false)
    MATCH(32, 48, 22, 128, 32, 1, false)
    MATCH(32, 64, 18, 128, 128, 1, false)
    MATCH(32, 96, 13, 128, 64, 1, false)
    MATCH(32, 112, 12, 128, 32, 1, false)
    MATCH(32, 224, 6, 128, 64, 1, false)
    MATCH(32, 240, 6, 128, 32, 1, false)
    MATCH(64, 16, 22, 128, 32, 1, false)
    MATCH(64, 32, 18, 128, 64, 1, false)
    MATCH(64, 48, 15, 128, 32, 1, false)
    MATCH(64, 64, 13, 128, 128, 1, false)
    MATCH(64, 64, 13, 128, 128, 2, false)
    MATCH(64, 80, 12, 128, 32, 1, false)
    MATCH(64, 80, 12, 128, 32, 2, true)
    MATCH(64, 96, 10, 128, 64, 1, false)
    MATCH(64, 96, 10, 128, 64, 2, true)
    MATCH(64, 112, 9, 128, 32, 1, false)
    MATCH(64, 112, 9, 128, 32, 2, false)
    MATCH(64, 128, 8, 128, 128, 1, false)
    MATCH(64, 128, 8, 128, 128, 2, false)
    MATCH(64, 144, 8, 128, 32, 1, false)
    MATCH(64, 160, 7, 128, 64, 1, false)
    MATCH(64, 160, 7, 128, 64, 2, true)
    MATCH(64, 176, 6, 128, 32, 1, false)
    MATCH(64, 176, 6, 128, 32, 2, false)
    MATCH(64, 192, 6, 128, 128, 1, false)
    MATCH(64, 192, 6, 128, 128, 2, false)
    MATCH(64, 192, 6, 128, 128, 2, true)
    MATCH(64, 208, 5, 128, 32, 2, false)
    MATCH(64, 208, 5, 128, 32, 2, true)
    MATCH(64, 224, 5, 128, 64, 1, false)
    MATCH(64, 224, 5, 128, 64, 2, false)
    MATCH(64, 240, 5, 128, 32, 1, false)
    MATCH(64, 240, 5, 128, 32, 2, false)
    MATCH(64, 240, 5, 128, 32, 2, true)
    MATCH(64, 256, 4, 128, 128, 1, false)
    MATCH(64, 256, 4, 128, 128, 2, false)
    MATCH(64, 256, 4, 128, 128, 2, true)
    MATCH(128, 16, 12, 256, 32, 1, false)
    MATCH(128, 32, 10, 256, 64, 1, false)
    MATCH(128, 48, 9, 256, 32, 1, false)
    MATCH(128, 48, 9, 256, 32, 2, true)
    MATCH(128, 64, 8, 256, 128, 1, false)
    MATCH(128, 64, 8, 256, 128, 2, true)
    MATCH(128, 80, 7, 256, 32, 1, false)
    MATCH(128, 80, 7, 256, 32, 2, false)
    MATCH(128, 80, 7, 256, 32, 2, true)
    MATCH(128, 96, 7, 256, 64, 1, false)
    MATCH(128, 96, 7, 256, 64, 2, true)
    MATCH(128, 112, 6, 256, 32, 2, false)
    MATCH(128, 128, 6, 256, 128, 1, false)
    MATCH(128, 128, 6, 256, 128, 2, false)
    MATCH(128, 144, 5, 256, 32, 1, false)
    MATCH(128, 144, 5, 256, 32, 2, false)
    MATCH(128, 160, 5, 256, 64, 1, false)
    MATCH(128, 160, 5, 256, 64, 2, false)
    MATCH(128, 160, 5, 256, 64, 2, true)
    MATCH(128, 176, 4, 256, 32, 1, false)
    MATCH(128, 176, 4, 256, 32, 2, false)
    MATCH(128, 176, 4, 256, 32, 2, true)
    MATCH(128, 192, 4, 256, 128, 1, false)
    MATCH(128, 192, 4, 256, 128, 2, false)
    MATCH(128, 192, 4, 256, 128, 2, true)
    MATCH(128, 208, 4, 256, 32, 1, false)
    MATCH(128, 208, 4, 256, 32, 2, false)
    MATCH(128, 208, 4, 256, 32, 2, true)
    MATCH(128, 224, 3, 256, 64, 1, false)
    MATCH(128, 224, 3, 256, 64, 2, false)
    MATCH(128, 224, 3, 256, 64, 2, true)
    MATCH(128, 240, 3, 256, 32, 1, false)
    MATCH(128, 240, 3, 256, 32, 2, false)
    MATCH(128, 240, 3, 256, 32, 2, true)
    MATCH(128, 256, 3, 256, 128, 1, false)
    MATCH(128, 256, 3, 256, 128, 2, false)
    MATCH(128, 256, 3, 256, 128, 2, true)
    MATCH(256, 16, 6, 256, 32, 1, false)
    MATCH(256, 32, 5, 256, 64, 2, true)
    MATCH(256, 48, 5, 256, 32, 1, false)
    MATCH(256, 48, 5, 256, 32, 2, true)
    MATCH(256, 64, 4, 256, 128, 2, false)
    MATCH(256, 64, 4, 256, 128, 2, true)
    MATCH(256, 80, 4, 256, 32, 1, false)
    MATCH(256, 80, 4, 256, 32, 2, false)
    MATCH(256, 80, 4, 256, 32, 2, true)
    MATCH(256, 96, 4, 256, 64, 2, true)
    MATCH(256, 112, 3, 256, 32, 1, false)
    MATCH(256, 112, 3, 256, 32, 2, false)
    MATCH(256, 112, 3, 256, 32, 2, true)
    MATCH(256, 128, 3, 256, 128, 2, false)
    MATCH(256, 128, 3, 256, 128, 2, true)
    MATCH(16, 80, 18, 128, 32, 1, false)
    MATCH(16, 208, 7, 128, 32, 1, false)
    MATCH(16, 256, 6, 128, 128, 1, false)
    MATCH(64, 144, 8, 128, 32, 2, false)
    MATCH(64, 160, 7, 128, 64, 2, false)
    MATCH(64, 208, 5, 128, 32, 1, false)

    #undef MATCH
    return nullptr;
}

// ── Grouped BF16 kernel dispatch ───────────────────────────────────

static const void* get_grouped_kernel(const KernelConfig& cfg) {
    #define MATCH_GRP(BN, ST, SD, MC, MCA) \
        if (cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_GROUPED(BN, ST, SD, MC, MCA);

    MATCH_GRP(16, 12, 32, 1, false)
    MATCH_GRP(16, 12, 32, 2, false)
    MATCH_GRP(16, 12, 32, 2, true)
    MATCH_GRP(32, 10, 64, 1, false)
    MATCH_GRP(32, 10, 64, 2, false)
    MATCH_GRP(32, 10, 64, 2, true)
    MATCH_GRP(48, 9, 32, 1, false)
    MATCH_GRP(48, 9, 32, 2, true)
    MATCH_GRP(64, 8, 128, 1, false)
    MATCH_GRP(64, 8, 128, 2, true)
    MATCH_GRP(80, 7, 32, 1, false)
    MATCH_GRP(80, 7, 32, 2, false)
    MATCH_GRP(80, 7, 32, 2, true)
    MATCH_GRP(96, 7, 64, 1, false)
    MATCH_GRP(96, 7, 64, 2, true)
    MATCH_GRP(112, 6, 32, 1, false)
    MATCH_GRP(112, 6, 32, 2, false)
    MATCH_GRP(128, 6, 128, 1, false)
    MATCH_GRP(128, 6, 128, 2, false)
    MATCH_GRP(144, 5, 32, 1, false)
    MATCH_GRP(144, 5, 32, 2, false)
    MATCH_GRP(160, 5, 64, 1, false)
    MATCH_GRP(160, 5, 64, 2, false)
    MATCH_GRP(160, 5, 64, 2, true)
    MATCH_GRP(176, 4, 32, 1, false)
    MATCH_GRP(176, 4, 32, 2, false)
    MATCH_GRP(176, 4, 32, 2, true)
    MATCH_GRP(192, 4, 128, 1, false)
    MATCH_GRP(192, 4, 128, 2, false)
    MATCH_GRP(192, 4, 128, 2, true)
    MATCH_GRP(208, 4, 32, 1, false)
    MATCH_GRP(208, 4, 32, 2, false)
    MATCH_GRP(208, 4, 32, 2, true)
    MATCH_GRP(224, 3, 64, 1, false)
    MATCH_GRP(224, 3, 64, 2, false)
    MATCH_GRP(224, 3, 64, 2, true)
    MATCH_GRP(240, 3, 32, 1, false)
    MATCH_GRP(240, 3, 32, 2, false)
    MATCH_GRP(240, 3, 32, 2, true)
    MATCH_GRP(256, 3, 128, 1, false)
    MATCH_GRP(256, 3, 128, 2, false)
    MATCH_GRP(256, 3, 128, 2, true)

    #undef MATCH_GRP
    return nullptr;
}

// ════════════════════════════════════════════════════════════════════
// BF16 GEMM with accumulation: D(FP32) += A @ B
// kWithAccumulation=true, cd_dtype_t=float, uses TMA_REDUCE_ADD.
// ════════════════════════════════════════════════════════════════════

static KernelConfig select_acc_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;

    int block_ms[5] = {64, 128, 256, 0, 0};
    int n_block_ms = 3;
    if (m <= 16) { block_ms[n_block_ms++] = 16; }
    if (m <= 32) { block_ms[n_block_ms++] = 32; }

    int block_ns[16]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16) block_ns[n_block_ns++] = i;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm > 128 && bn > 128) continue;
            // With FP32 output, block_m must be <= 128 (upstream constraint)
            if (bm > 128) continue;
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

    int num_tma = 128;
    int num_math = (best_bm <= 64) ? 128 : 256;

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

    SmemConfig scfg;
    int best_stages = select_num_stages<SM90Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        best_bm, best_bn, block_k, multicast, mc_on_a,
        2, 4, // ab_elem=BF16(2), cd_elem=FP32(4)
        m, n, 0, scfg);

    return KernelConfig{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

#define KERNEL_TYPE_ACC(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, 1,                                                        \
        BLOCK_M, BLOCK_N, 64,                                              \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, NUM_MATH,                                             \
        NUM_MC, MC_ON_A, 132,                                              \
        GemmType::Normal, true, float>

// FP32 output: block_m <= 128 (upstream constraint). sw_d=0 (no swizzle for FP32 on SM90).
// Stages: smem_d=align(bm*bn*4,1024), per_stage=bm*64*2+bn*64*2+16
__attribute__((used)) static auto* _acc_00 = &KERNEL_TYPE_ACC(16, 16, 32, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_01 = &KERNEL_TYPE_ACC(16, 32, 32, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_02 = &KERNEL_TYPE_ACC(16, 64, 22, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_03 = &KERNEL_TYPE_ACC(16, 128, 12, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_04 = &KERNEL_TYPE_ACC(32, 16, 32, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_05 = &KERNEL_TYPE_ACC(32, 32, 27, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_06 = &KERNEL_TYPE_ACC(32, 64, 18, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_07 = &KERNEL_TYPE_ACC(32, 128, 10, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_10 = &KERNEL_TYPE_ACC(64, 16, 22, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_11 = &KERNEL_TYPE_ACC(64, 32, 18, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_12 = &KERNEL_TYPE_ACC(64, 64, 13, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_13 = &KERNEL_TYPE_ACC(64, 128, 8, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_14 = &KERNEL_TYPE_ACC(64, 256, 4, 128, 0, 1, false);
__attribute__((used)) static auto* _acc_20 = &KERNEL_TYPE_ACC(128, 16, 12, 256, 0, 1, false);
__attribute__((used)) static auto* _acc_21 = &KERNEL_TYPE_ACC(128, 32, 10, 256, 0, 1, false);
__attribute__((used)) static auto* _acc_22 = &KERNEL_TYPE_ACC(128, 64, 8, 256, 0, 1, false);
__attribute__((used)) static auto* _acc_23 = &KERNEL_TYPE_ACC(128, 128, 5, 256, 0, 1, false);

static const void* get_acc_kernel(const KernelConfig& cfg) {
    // sw_d is always 0 for FP32 accumulation
    #define MATCH_ACC(BM, BN, ST, NM) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_math_threads == NM && cfg.num_multicast == 1) \
            return (const void*)&KERNEL_TYPE_ACC(BM, BN, ST, NM, 0, 1, false);

    MATCH_ACC(16, 16, 32, 128) MATCH_ACC(16, 32, 32, 128)
    MATCH_ACC(16, 64, 22, 128) MATCH_ACC(16, 128, 12, 128)
    MATCH_ACC(32, 16, 32, 128) MATCH_ACC(32, 32, 27, 128)
    MATCH_ACC(32, 64, 18, 128) MATCH_ACC(32, 128, 10, 128)
    MATCH_ACC(64, 16, 22, 128) MATCH_ACC(64, 32, 18, 128)
    MATCH_ACC(64, 64, 13, 128) MATCH_ACC(64, 128, 8, 128)
    MATCH_ACC(64, 256, 4, 128)
    MATCH_ACC(128, 16, 12, 256) MATCH_ACC(128, 32, 10, 256)
    MATCH_ACC(128, 64, 8, 256) MATCH_ACC(128, 128, 5, 256)

    #undef MATCH_ACC
    return nullptr;
}

// ── SM90 BF16 implementation functions ─────────────────────────────

static int sm90_bf16_gemm(void* A, void* B, void* D, int M, int N, int K, void* stream) {
    cudaGetLastError();
    auto cfg = select_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    auto tma_d = make_2d_tma(D, N, M,
                             cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n,
                             cfg.block_m, N, cfg.swizzle_d);

    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &grouped_layout, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

/// BF16 GEMM with FP32 accumulation: D(FP32) += A(BF16) @ B(BF16)
/// If C != D and C != nullptr, copies C to D first (upstream early_return pattern).
/// D must be pre-allocated as FP32[M,N].
static int sm90_bf16_gemm_acc(void* A, void* B, void* C, void* D,
                               int M, int N, int K, void* stream) {
    cudaGetLastError();

    // If C provided and different from D, copy C to D first
    if (C && C != D) {
        auto s = static_cast<cudaStream_t>(stream);
        cudaMemcpyAsync(D, C, (size_t)M * N * 4, cudaMemcpyDeviceToDevice, s);
    }

    auto cfg = select_acc_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_acc_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // D is FP32, no swizzle on SM90 for FP32 output
    auto tma_d = make_2d_tma_f32(D, N, M, cfg.block_n, cfg.block_m, N, 0);

    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &grouped_layout, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

static int sm90_m_grouped_bf16_gemm(
    void* A, void* B, void* D, void* grouped_layout,
    int M, int N, int K, int num_groups, void* stream
) {
    cudaGetLastError();
    auto cfg = select_grouped_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_grouped_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);

    int* layout_ptr = (int*)grouped_layout;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &layout_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

// ════════════════════════════════════════════════════════════════════
// M-Grouped Masked BF16 GEMM
// A[G,M,K] @ B[G,N,K]^T → D[G,M,N] with masked_m[G] actual rows.
// Same kernel template with GemmType::MGroupedMasked.
// ════════════════════════════════════════════════════════════════════

// ── Masked BF16 heuristic ──────────────────────────────────────────

static KernelConfig select_masked_config(int expected_m, int n, int k, int num_groups, int num_sms) {
    const int block_k = 64;
    // Masked: block_m = {64, 128} only (upstream excludes 256)
    int block_ms[2] = {64, 128};
    int n_block_ms = 2;

    int block_ns[16]; int n_block_ns = 0;
    for (int i = 16; i <= 256; i += 16) block_ns[n_block_ns++] = i;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm > 128 && bn > 128) continue;
            // num_groups multiplies block count for wave calculation
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

    int num_tma = 128;
    int num_math = (best_bm <= 64) ? 128 : 256;

    // Masked multicast: stricter than normal
    int multicast = 1; bool mc_on_a = false;
    if (expected_m >= 512 && num_sms % 2 == 0) {
        bool n_div = (n % best_bn == 0);
        bool n_even = (ceil_div(n, best_bn) % 2 == 0);
        bool m_even = (ceil_div(expected_m, best_bm) % 2 == 0);
        // B multicast (mc_on_a=false): n_even AND n_div (require_divisible=true)
        bool legal_mc_b = n_even && n_div;
        // A multicast (mc_on_a=true): m_even AND n_even AND n_div
        bool legal_mc_a = m_even && n_even && n_div;

        if (best_bm > best_bn) {
            if (legal_mc_a) { multicast = 2; mc_on_a = true; }
            else if (legal_mc_b) { multicast = 2; mc_on_a = false; }
        } else {
            if (legal_mc_b) { multicast = 2; mc_on_a = false; }
            else if (legal_mc_a) { multicast = 2; mc_on_a = true; }
        }
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM90Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        best_bm, best_bn, block_k, multicast, mc_on_a,
        2, 2, expected_m, n, 0, scfg);

    return KernelConfig{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_tma_threads = num_tma, .num_math_threads = num_math,
        .num_multicast = multicast, .multicast_on_a = mc_on_a,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// ── Masked BF16 kernel instantiations ──────────────────────────────
// Same template as normal BF16 but GemmType::MGroupedMasked.
// block_m restricted to {64, 128}.

// Masked kernel macro: kNumGroups must match actual num_groups at runtime
// (the Scheduler uses kNumGroups to bound total work tiles).
#define KERNEL_TYPE_MASKED(BLOCK_M, BLOCK_N, STAGES, NUM_MATH, SWIZZLE_D, NUM_MC, MC_ON_A, NGROUPS) \
    deep_gemm::sm90_bf16_gemm_impl<                                        \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                       \
        0, 0, 0, NGROUPS,                                                  \
        BLOCK_M, BLOCK_N, 64,                                              \
        128, 128, SWIZZLE_D,                                               \
        STAGES, 128, NUM_MATH,                                             \
        NUM_MC, MC_ON_A, 132,                                              \
        GemmType::MGroupedMasked, false, cutlass::bfloat16_t>

// Core configs for each num_groups. No multicast (multicast rarely triggers for masked).
// bm=64 (math=128): bn ∈ {16,32,48,64,96,128,256}
// bm=128 (math=256): bn ∈ {16,32,48,64,96,128,192,256}
// Full bn coverage: all multiples of 16 from 16..256 for both bm=64 and bm=128.
#define MASKED_CONFIGS(G, TAG) \
    __attribute__((used)) static auto* _msk_##TAG##_00 = &KERNEL_TYPE_MASKED(64, 16, 22, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_01 = &KERNEL_TYPE_MASKED(64, 32, 18, 128, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_02 = &KERNEL_TYPE_MASKED(64, 48, 15, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_03 = &KERNEL_TYPE_MASKED(64, 64, 13, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_04 = &KERNEL_TYPE_MASKED(64, 80, 12, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_05 = &KERNEL_TYPE_MASKED(64, 96, 10, 128, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_06 = &KERNEL_TYPE_MASKED(64, 112, 9, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_07 = &KERNEL_TYPE_MASKED(64, 128, 8, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_08 = &KERNEL_TYPE_MASKED(64, 144, 8, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_09 = &KERNEL_TYPE_MASKED(64, 160, 7, 128, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_0a = &KERNEL_TYPE_MASKED(64, 176, 6, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_0b = &KERNEL_TYPE_MASKED(64, 192, 6, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_0c = &KERNEL_TYPE_MASKED(64, 208, 5, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_0d = &KERNEL_TYPE_MASKED(64, 224, 5, 128, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_0e = &KERNEL_TYPE_MASKED(64, 240, 5, 128, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_0f = &KERNEL_TYPE_MASKED(64, 256, 4, 128, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_10 = &KERNEL_TYPE_MASKED(128, 16, 12, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_11 = &KERNEL_TYPE_MASKED(128, 32, 10, 256, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_12 = &KERNEL_TYPE_MASKED(128, 48, 9, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_13 = &KERNEL_TYPE_MASKED(128, 64, 8, 256, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_14 = &KERNEL_TYPE_MASKED(128, 80, 7, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_15 = &KERNEL_TYPE_MASKED(128, 96, 7, 256, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_16 = &KERNEL_TYPE_MASKED(128, 112, 6, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_17 = &KERNEL_TYPE_MASKED(128, 128, 6, 256, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_18 = &KERNEL_TYPE_MASKED(128, 144, 5, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_19 = &KERNEL_TYPE_MASKED(128, 160, 5, 256, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_1a = &KERNEL_TYPE_MASKED(128, 176, 4, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_1b = &KERNEL_TYPE_MASKED(128, 192, 4, 256, 128, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_1c = &KERNEL_TYPE_MASKED(128, 208, 4, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_1d = &KERNEL_TYPE_MASKED(128, 224, 3, 256, 64, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_1e = &KERNEL_TYPE_MASKED(128, 240, 3, 256, 32, 1, false, G); \
    __attribute__((used)) static auto* _msk_##TAG##_1f = &KERNEL_TYPE_MASKED(128, 256, 3, 256, 128, 1, false, G);

// Instantiate for common MoE expert counts
MASKED_CONFIGS(2,  g02)
MASKED_CONFIGS(4,  g04)
MASKED_CONFIGS(8,  g08)
MASKED_CONFIGS(16, g16)
MASKED_CONFIGS(32, g32)
MASKED_CONFIGS(64, g64)

// ── Masked BF16 kernel dispatch ────────────────────────────────────

static const void* get_masked_kernel(const KernelConfig& cfg, int num_groups) {
    // Match config + num_groups. num_groups must match a compiled variant.
    #define MATCH_MSK(BM, BN, ST, NM, SD, G) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.num_math_threads == NM && cfg.swizzle_d == SD && \
            cfg.num_multicast == 1 && num_groups == G) \
            return (const void*)&KERNEL_TYPE_MASKED(BM, BN, ST, NM, SD, 1, false, G);

    // Expand for each supported num_groups
    #define MATCH_MSK_ALL_G(BM, BN, ST, NM, SD) \
        MATCH_MSK(BM, BN, ST, NM, SD, 2)  \
        MATCH_MSK(BM, BN, ST, NM, SD, 4)  \
        MATCH_MSK(BM, BN, ST, NM, SD, 8)  \
        MATCH_MSK(BM, BN, ST, NM, SD, 16) \
        MATCH_MSK(BM, BN, ST, NM, SD, 32) \
        MATCH_MSK(BM, BN, ST, NM, SD, 64)

    // bm=64 (math=128) — all bn from 16..256 step 16
    MATCH_MSK_ALL_G(64, 16, 22, 128, 32) MATCH_MSK_ALL_G(64, 32, 18, 128, 64)
    MATCH_MSK_ALL_G(64, 48, 15, 128, 32) MATCH_MSK_ALL_G(64, 64, 13, 128, 128)
    MATCH_MSK_ALL_G(64, 80, 12, 128, 32) MATCH_MSK_ALL_G(64, 96, 10, 128, 64)
    MATCH_MSK_ALL_G(64, 112, 9, 128, 32) MATCH_MSK_ALL_G(64, 128, 8, 128, 128)
    MATCH_MSK_ALL_G(64, 144, 8, 128, 32) MATCH_MSK_ALL_G(64, 160, 7, 128, 64)
    MATCH_MSK_ALL_G(64, 176, 6, 128, 32) MATCH_MSK_ALL_G(64, 192, 6, 128, 128)
    MATCH_MSK_ALL_G(64, 208, 5, 128, 32) MATCH_MSK_ALL_G(64, 224, 5, 128, 64)
    MATCH_MSK_ALL_G(64, 240, 5, 128, 32) MATCH_MSK_ALL_G(64, 256, 4, 128, 128)
    // bm=128 (math=256) — all bn from 16..256 step 16
    MATCH_MSK_ALL_G(128, 16, 12, 256, 32) MATCH_MSK_ALL_G(128, 32, 10, 256, 64)
    MATCH_MSK_ALL_G(128, 48, 9, 256, 32) MATCH_MSK_ALL_G(128, 64, 8, 256, 128)
    MATCH_MSK_ALL_G(128, 80, 7, 256, 32) MATCH_MSK_ALL_G(128, 96, 7, 256, 64)
    MATCH_MSK_ALL_G(128, 112, 6, 256, 32) MATCH_MSK_ALL_G(128, 128, 6, 256, 128)
    MATCH_MSK_ALL_G(128, 144, 5, 256, 32) MATCH_MSK_ALL_G(128, 160, 5, 256, 64)
    MATCH_MSK_ALL_G(128, 176, 4, 256, 32) MATCH_MSK_ALL_G(128, 192, 4, 256, 128)
    MATCH_MSK_ALL_G(128, 208, 4, 256, 32) MATCH_MSK_ALL_G(128, 224, 3, 256, 64)
    MATCH_MSK_ALL_G(128, 240, 3, 256, 32) MATCH_MSK_ALL_G(128, 256, 3, 256, 128)

    #undef MATCH_MSK_ALL_G
    #undef MATCH_MSK
    return nullptr;
}

// ── Masked BF16 implementation ─────────────────────────────────────

static int sm90_m_grouped_masked_bf16_gemm(
    void* A, void* B, void* D, void* masked_m,
    int M, int N, int K, int num_groups, int expected_m, void* stream
) {
    cudaGetLastError();
    auto cfg = select_masked_config(expected_m, N, K, num_groups, g_num_sms);
    auto kernel_ptr = get_masked_kernel(cfg, num_groups);
    if (!kernel_ptr) return -1;

    // Masked: A[G,M,K], outer = M * num_groups
    auto tma_a = make_2d_tma(A, K, M * num_groups, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    // B[G,N,K], outer = N * num_groups
    auto tma_b = make_2d_tma(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // D[G,M,N], outer = M * num_groups
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M * num_groups, d_smem_inner, cfg.block_m, N, cfg.swizzle_d);

    // masked_m is passed as grouped_layout (kernel interprets based on GemmType)
    int* mask_ptr = (int*)masked_m;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &mask_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, cfg.num_tma_threads + cfg.num_math_threads,
                         cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}
