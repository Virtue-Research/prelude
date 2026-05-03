// SM100 (Blackwell) BF16 GEMM: Normal
// Heuristic, kernel instantiations, dispatch, and implementation.
// SM90 kernel bodies are guarded by __CUDA_ARCH__ >= 900 && < 1000,
// SM100 kernel bodies by __CUDA_ARCH__ >= 1000, so fat binary works.

#pragma once

// ── SM100 BF16 heuristic ───────────────────────────────────────────

static SM100Config select_sm100_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;

    int block_ms[4] = {128, 256, 0, 0};
    int n_block_ms = 2;
    if (m <= 32) { block_ms[n_block_ms++] = 32; }
    if (m <= 64) { block_ms[n_block_ms++] = 64; }

    // SM100 block_n: {16, 32, 64, 96, ..., 256} step 32 after 16
    int block_ns[9] = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    int n_block_ns = 9;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm == 0) continue;
            if (k <= 256 && (bn > 128 || bm > 128)) continue;

            int num_blocks = ceil_div(m, bm) * ceil_div(n, bn);
            int waves = ceil_div(num_blocks, num_sms);
            int last_util = num_blocks % num_sms;
            if (last_util == 0) last_util = num_sms;

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

    // SM100 multicast: only on B (A multicast forbidden)
    int multicast = 1;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal = (ceil_div(m, best_bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    // Blackwell-Ultra (SM103, B300): the `(BM=256, BN=128, stages=4,
    // swizzle_d=128, multicast=2)` kernel variant OOBs on shared/local memory
    // (caught by compute-sanitizer; repro at e.g. M=1408 N=3072 K=1024).
    // Shrink the tile to the safe `(BM=128, BN=128)` and keep multicast
    // enabled when still legal under the smaller block.
    if (g_gpu_arch == 103 && best_bm == 256 && best_bn == 128) {
        best_bm = 128;
        if (multicast == 2 && ceil_div(m, best_bm) % 2 != 0) {
            multicast = 1;
        }
    }
    // Blackwell-Ultra: tiny-N kernels with multicast=2 also OOB on shared
    // memory — caught by compute-sanitizer at M=512 N=4096 K=2048 (Qwen3.5
    // in_proj_q at 512-token prefill), which selects (BM=128, BN=16,
    // stages=12, multicast=2). Multicast on B with BN<32 doesn't pay back
    // its sync overhead anyway; just disable it for these tile widths.
    if (g_gpu_arch == 103 && best_bn < 32 && multicast == 2) {
        multicast = 1;
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM100Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        best_bm, best_bn, block_k, multicast, false,
        2, 2, m, n, 0, scfg);

    return SM100Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast, .multicast_on_a = false,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// ── SM100 BF16 kernel instantiations ───────────────────────────────

#define KERNEL_TYPE_SM100(BLOCK_M, BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm100_bf16_gemm_impl<                                            \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                            \
        0, 0, 0,                                                                \
        BLOCK_M, BLOCK_N, 64, 1,                                                \
        128, 128, SWIZZLE_D,                                                    \
        STAGES, 128, 128,                                                       \
        NUM_MC, MC_ON_A, 132,                                                   \
        GemmType::Normal, false, cutlass::bfloat16_t, 100>

__attribute__((used)) static auto* _s100_00 = &KERNEL_TYPE_SM100(32, 16, 32, 32, 1, false);
__attribute__((used)) static auto* _s100_01 = &KERNEL_TYPE_SM100(32, 32, 27, 64, 1, false);
__attribute__((used)) static auto* _s100_02 = &KERNEL_TYPE_SM100(32, 64, 22, 128, 1, false);
__attribute__((used)) static auto* _s100_03 = &KERNEL_TYPE_SM100(32, 96, 18, 64, 1, false);
__attribute__((used)) static auto* _s100_04 = &KERNEL_TYPE_SM100(32, 128, 15, 128, 1, false);
__attribute__((used)) static auto* _s100_10 = &KERNEL_TYPE_SM100(64, 16, 32, 32, 1, false);
__attribute__((used)) static auto* _s100_11 = &KERNEL_TYPE_SM100(64, 32, 22, 64, 1, false);
__attribute__((used)) static auto* _s100_12 = &KERNEL_TYPE_SM100(64, 64, 15, 128, 1, false);
__attribute__((used)) static auto* _s100_13 = &KERNEL_TYPE_SM100(64, 96, 12, 64, 1, false);
__attribute__((used)) static auto* _s100_14 = &KERNEL_TYPE_SM100(64, 128, 10, 128, 1, false);
__attribute__((used)) static auto* _s100_20 = &KERNEL_TYPE_SM100(128, 16, 12, 32, 1, false);
__attribute__((used)) static auto* _s100_21 = &KERNEL_TYPE_SM100(128, 32, 10, 64, 1, false);
__attribute__((used)) static auto* _s100_22 = &KERNEL_TYPE_SM100(128, 64, 8, 128, 1, false);
__attribute__((used)) static auto* _s100_23 = &KERNEL_TYPE_SM100(128, 96, 7, 64, 1, false);
__attribute__((used)) static auto* _s100_24 = &KERNEL_TYPE_SM100(128, 128, 6, 128, 1, false);
__attribute__((used)) static auto* _s100_25 = &KERNEL_TYPE_SM100(128, 160, 5, 64, 1, false);
__attribute__((used)) static auto* _s100_26 = &KERNEL_TYPE_SM100(128, 192, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_27 = &KERNEL_TYPE_SM100(128, 224, 4, 64, 1, false);
__attribute__((used)) static auto* _s100_28 = &KERNEL_TYPE_SM100(128, 256, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_29 = &KERNEL_TYPE_SM100(128, 16, 12, 32, 2, false);
__attribute__((used)) static auto* _s100_2a = &KERNEL_TYPE_SM100(128, 32, 10, 64, 2, false);
__attribute__((used)) static auto* _s100_2b = &KERNEL_TYPE_SM100(128, 64, 8, 128, 2, false);
__attribute__((used)) static auto* _s100_2c = &KERNEL_TYPE_SM100(128, 96, 7, 64, 2, false);
__attribute__((used)) static auto* _s100_2d = &KERNEL_TYPE_SM100(128, 128, 6, 128, 2, false);
__attribute__((used)) static auto* _s100_2e = &KERNEL_TYPE_SM100(128, 160, 5, 64, 2, false);
__attribute__((used)) static auto* _s100_2f = &KERNEL_TYPE_SM100(128, 192, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_30 = &KERNEL_TYPE_SM100(128, 224, 4, 64, 2, false);
__attribute__((used)) static auto* _s100_31 = &KERNEL_TYPE_SM100(128, 256, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_40 = &KERNEL_TYPE_SM100(256, 16, 6, 32, 1, false);
__attribute__((used)) static auto* _s100_41 = &KERNEL_TYPE_SM100(256, 32, 5, 64, 1, false);
__attribute__((used)) static auto* _s100_42 = &KERNEL_TYPE_SM100(256, 64, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_43 = &KERNEL_TYPE_SM100(256, 96, 4, 64, 1, false);
__attribute__((used)) static auto* _s100_44 = &KERNEL_TYPE_SM100(256, 128, 4, 128, 1, false);
__attribute__((used)) static auto* _s100_45 = &KERNEL_TYPE_SM100(256, 160, 3, 64, 1, false);
__attribute__((used)) static auto* _s100_46 = &KERNEL_TYPE_SM100(256, 192, 3, 128, 1, false);
__attribute__((used)) static auto* _s100_47 = &KERNEL_TYPE_SM100(256, 224, 3, 64, 1, false);
__attribute__((used)) static auto* _s100_48 = &KERNEL_TYPE_SM100(256, 256, 3, 128, 1, false);
__attribute__((used)) static auto* _s100_49 = &KERNEL_TYPE_SM100(256, 16, 6, 32, 2, false);
__attribute__((used)) static auto* _s100_4a = &KERNEL_TYPE_SM100(256, 32, 5, 64, 2, false);
__attribute__((used)) static auto* _s100_4b = &KERNEL_TYPE_SM100(256, 64, 4, 128, 2, false);
__attribute__((used)) static auto* _s100_4c = &KERNEL_TYPE_SM100(256, 96, 4, 64, 2, false);
__attribute__((used)) static auto* _s100_4d = &KERNEL_TYPE_SM100(256, 128, 4, 128, 2, false);

// ── SM100 dispatch ─────────────────────────────────────────────────

static const void* get_sm100_kernel(const SM100Config& cfg) {
    #define MATCH_SM100(BM, BN, ST, SD, MC, MCA) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_SM100(BM, BN, ST, SD, MC, MCA);

    MATCH_SM100(32, 16, 32, 32, 1, false) MATCH_SM100(32, 32, 27, 64, 1, false)
    MATCH_SM100(32, 64, 22, 128, 1, false) MATCH_SM100(32, 96, 18, 64, 1, false)
    MATCH_SM100(32, 128, 15, 128, 1, false)
    MATCH_SM100(64, 16, 32, 32, 1, false) MATCH_SM100(64, 32, 22, 64, 1, false)
    MATCH_SM100(64, 64, 15, 128, 1, false) MATCH_SM100(64, 96, 12, 64, 1, false)
    MATCH_SM100(64, 128, 10, 128, 1, false)
    MATCH_SM100(128, 16, 12, 32, 1, false) MATCH_SM100(128, 32, 10, 64, 1, false)
    MATCH_SM100(128, 64, 8, 128, 1, false) MATCH_SM100(128, 96, 7, 64, 1, false)
    MATCH_SM100(128, 128, 6, 128, 1, false) MATCH_SM100(128, 160, 5, 64, 1, false)
    MATCH_SM100(128, 192, 4, 128, 1, false) MATCH_SM100(128, 224, 4, 64, 1, false)
    MATCH_SM100(128, 256, 4, 128, 1, false)
    MATCH_SM100(128, 16, 12, 32, 2, false) MATCH_SM100(128, 32, 10, 64, 2, false)
    MATCH_SM100(128, 64, 8, 128, 2, false) MATCH_SM100(128, 96, 7, 64, 2, false)
    MATCH_SM100(128, 128, 6, 128, 2, false) MATCH_SM100(128, 160, 5, 64, 2, false)
    MATCH_SM100(128, 192, 4, 128, 2, false) MATCH_SM100(128, 224, 4, 64, 2, false)
    MATCH_SM100(128, 256, 4, 128, 2, false)
    MATCH_SM100(256, 16, 6, 32, 1, false) MATCH_SM100(256, 32, 5, 64, 1, false)
    MATCH_SM100(256, 64, 4, 128, 1, false) MATCH_SM100(256, 96, 4, 64, 1, false)
    MATCH_SM100(256, 128, 4, 128, 1, false) MATCH_SM100(256, 160, 3, 64, 1, false)
    MATCH_SM100(256, 192, 3, 128, 1, false) MATCH_SM100(256, 224, 3, 64, 1, false)
    MATCH_SM100(256, 256, 3, 128, 1, false)
    MATCH_SM100(256, 16, 6, 32, 2, false) MATCH_SM100(256, 32, 5, 64, 2, false)
    MATCH_SM100(256, 64, 4, 128, 2, false) MATCH_SM100(256, 96, 4, 64, 2, false)
    MATCH_SM100(256, 128, 4, 128, 2, false)

    #undef MATCH_SM100
    return nullptr;
}

// ── SM100 BF16 implementation ──────────────────────────────────────

static int sm100_bf16_gemm(void* A, void* B, void* D, int M, int N, int K, void* stream) {
    cudaGetLastError();
    auto cfg = select_sm100_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_sm100_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cd_store_bm, N, cfg.swizzle_d);

    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &grouped_layout, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, 256, cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

// ════════════════════════════════════════════════════════════════════
// SM100 M-Grouped Contiguous BF16 GEMM
// Same template, GemmType::MGroupedContiguous, block_m fixed to 128.
// ════════════════════════════════════════════════════════════════════

static SM100Config select_sm100_grouped_config(int m, int n, int k, int num_sms) {
    const int block_k = 64;
    const int bm = 128; // fixed for MGroupedContiguous

    int block_ns[9] = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    int n_block_ns = 9;

    int best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int j = 0; j < n_block_ns; j++) {
        int bn = block_ns[j];
        if (k <= 256 && bn > 128) continue;
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

    // SM100 multicast: only on B (mc_on_a=false)
    int multicast = 1;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal = (ceil_div(m, bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    // Demote multicast=2 → 1 for `bn` values that don't have a mc=2
    // kernel registered below (160 and 224). Without this, lookup hits
    // nullptr → -1 even though the heuristic chose a sensible bn.
    if (multicast == 2 && (best_bn == 160 || best_bn == 224)) {
        multicast = 1;
    }
    // Blackwell-Ultra (SM103) MGroupedContiguous mc=2 kernels OOB on
    // shared memory across multiple bn values (compute-sanitizer
    // catches it inside `sm100_bf16_gemm_impl<…, mc=2, MGrouped, …>`,
    // first repro: M=1024 N=768 K=2048 G=8 → bn=64 mc=2). Same bug
    // family as the (256, 128) Normal-path workaround and the small-N
    // mc=2 workaround above; the grouped path can't downgrade BM
    // (fixed at 128 by MGroupedContiguous), so the only knob left is
    // mc. Force mc=1 for the entire SM103 grouped path until upstream
    // fixes the kernel.
    if (g_gpu_arch == 103 && multicast == 2) {
        multicast = 1;
    }

    // Clamp `(stages, swizzle_d)` to whatever variant is actually
    // compiled below for this `best_bn`. The upstream `select_num_stages`
    // returns a smem-fit-driven optimum that doesn't know about our
    // limited static kernel set, so on shapes like (M=1024, N=768,
    // K=2048) it lands on stages=9 while only stages=8 is registered
    // for bn=64 — `get_sm100_grouped_kernel` then returns nullptr → -1.
    static const int reg_bns[9]   = {16, 32, 64,  96, 128, 160, 192, 224, 256};
    static const int reg_stages[9]= {12, 10,  8,   7,   6,   5,   4,   4,   4};
    static const int reg_swdy[9]  = {32, 64,128,  64, 128,  64, 128,  64, 128};
    int forced_stages = -1, forced_swd = -1;
    for (int i = 0; i < 9; i++) {
        if (best_bn == reg_bns[i]) {
            forced_stages = reg_stages[i];
            forced_swd = reg_swdy[i];
            break;
        }
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM100Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        bm, best_bn, block_k, multicast, false,
        2, 2, m, n, 0, scfg);

    int swizzle_d = scfg.swizzle_cd;
    int smem_size = scfg.smem_size;

    if (forced_stages > 0) {
        // After forcing stages, the smem layout the registered kernel
        // expects no longer matches the one `select_num_stages` chose
        // for its larger stage count — passing the wrong smem_size to
        // the launch hangs the SM. Recompute the smem layout for the
        // exact (stages, swizzle_d) that the registered kernel uses.
        best_stages = forced_stages;
        swizzle_d = forced_swd;
        SmemConfig recomp = compute_smem_config<SM100Arch>(
            KernelKind::NoSF, MmaKindLocal::BF16,
            bm, best_bn, block_k, best_stages, multicast, false,
            2, 2, m, n, 0);
        smem_size = recomp.smem_size;
    }

    return SM100Config{
        .block_m = bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast, .multicast_on_a = false,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = swizzle_d,
        .smem_size = smem_size,
    };
}

#define KERNEL_TYPE_SM100_GROUPED(BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, MC_ON_A) \
    deep_gemm::sm100_bf16_gemm_impl<                                            \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                            \
        0, 0, 0,                                                                \
        128, BLOCK_N, 64, 1,                                                    \
        128, 128, SWIZZLE_D,                                                    \
        STAGES, 128, 128,                                                       \
        NUM_MC, MC_ON_A, 132,                                                   \
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t, 100>

// bm=128 only (grouped contiguous constraint)
__attribute__((used)) static auto* _s100g_00 = &KERNEL_TYPE_SM100_GROUPED(16, 12, 32, 1, false);
__attribute__((used)) static auto* _s100g_01 = &KERNEL_TYPE_SM100_GROUPED(32, 10, 64, 1, false);
__attribute__((used)) static auto* _s100g_02 = &KERNEL_TYPE_SM100_GROUPED(64, 8, 128, 1, false);
__attribute__((used)) static auto* _s100g_03 = &KERNEL_TYPE_SM100_GROUPED(96, 7, 64, 1, false);
__attribute__((used)) static auto* _s100g_04 = &KERNEL_TYPE_SM100_GROUPED(128, 6, 128, 1, false);
__attribute__((used)) static auto* _s100g_05 = &KERNEL_TYPE_SM100_GROUPED(160, 5, 64, 1, false);
__attribute__((used)) static auto* _s100g_06 = &KERNEL_TYPE_SM100_GROUPED(192, 4, 128, 1, false);
__attribute__((used)) static auto* _s100g_07 = &KERNEL_TYPE_SM100_GROUPED(224, 4, 64, 1, false);
__attribute__((used)) static auto* _s100g_08 = &KERNEL_TYPE_SM100_GROUPED(256, 4, 128, 1, false);
// multicast
__attribute__((used)) static auto* _s100g_10 = &KERNEL_TYPE_SM100_GROUPED(16, 12, 32, 2, false);
__attribute__((used)) static auto* _s100g_11 = &KERNEL_TYPE_SM100_GROUPED(32, 10, 64, 2, false);
__attribute__((used)) static auto* _s100g_12 = &KERNEL_TYPE_SM100_GROUPED(64, 8, 128, 2, false);
__attribute__((used)) static auto* _s100g_13 = &KERNEL_TYPE_SM100_GROUPED(96, 7, 64, 2, false);
__attribute__((used)) static auto* _s100g_14 = &KERNEL_TYPE_SM100_GROUPED(128, 6, 128, 2, false);
__attribute__((used)) static auto* _s100g_15 = &KERNEL_TYPE_SM100_GROUPED(192, 4, 128, 2, false);
__attribute__((used)) static auto* _s100g_16 = &KERNEL_TYPE_SM100_GROUPED(256, 4, 128, 2, false);

static const void* get_sm100_grouped_kernel(const SM100Config& cfg) {
    #define MATCH_SM100G(BN, ST, SD, MC, MCA) \
        if (cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == MC && cfg.multicast_on_a == MCA) \
            return (const void*)&KERNEL_TYPE_SM100_GROUPED(BN, ST, SD, MC, MCA);

    MATCH_SM100G(16, 12, 32, 1, false) MATCH_SM100G(32, 10, 64, 1, false)
    MATCH_SM100G(64, 8, 128, 1, false) MATCH_SM100G(96, 7, 64, 1, false)
    MATCH_SM100G(128, 6, 128, 1, false) MATCH_SM100G(160, 5, 64, 1, false)
    MATCH_SM100G(192, 4, 128, 1, false) MATCH_SM100G(224, 4, 64, 1, false)
    MATCH_SM100G(256, 4, 128, 1, false)
    MATCH_SM100G(16, 12, 32, 2, false) MATCH_SM100G(32, 10, 64, 2, false)
    MATCH_SM100G(64, 8, 128, 2, false) MATCH_SM100G(96, 7, 64, 2, false)
    MATCH_SM100G(128, 6, 128, 2, false) MATCH_SM100G(192, 4, 128, 2, false)
    MATCH_SM100G(256, 4, 128, 2, false)

    #undef MATCH_SM100G
    return nullptr;
}

static int sm100_m_grouped_bf16_gemm(
    void* A, void* B, void* D, void* grouped_layout,
    int M, int N, int K, int num_groups, void* stream
) {
    cudaGetLastError();
    auto cfg = select_sm100_grouped_config(M, N, K, g_num_sms);
    if (getenv("DEEPGEMM_DEBUG")) {
        fprintf(stderr,
            "[deepgemm sm100 grouped] M=%d N=%d K=%d G=%d "
            "→ bm=%d bn=%d bk=%d stages=%d sw_a=%d sw_b=%d sw_d=%d mc=%d mc_on_a=%d smem=%d\n",
            M, N, K, num_groups,
            cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_stages,
            cfg.swizzle_a, cfg.swizzle_b, cfg.swizzle_d,
            cfg.num_multicast, cfg.multicast_on_a ? 1 : 0,
            cfg.smem_size);
    }
    auto kernel_ptr = get_sm100_grouped_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cd_store_bm, N, cfg.swizzle_d);

    int* layout_ptr = (int*)grouped_layout;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &layout_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, 256, cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}

// ════════════════════════════════════════════════════════════════════
// SM100 M-Grouped Masked BF16 GEMM
// ════════════════════════════════════════════════════════════════════

static SM100Config select_sm100_masked_config(int expected_m, int n, int k, int num_groups, int num_sms) {
    const int block_k = 64;
    // Masked: block_m = {64, 128}
    int block_ms[2] = {64, 128};
    int n_block_ms = 2;

    // NOTE: restricted to the block_n values that actually have compiled
    // kernel instantiations below (see SM100_MASKED_CONFIGS). The full
    // {96, 160, 192, 224} set is covered for *non-masked* SM100 GEMMs but
    // not for masked yet — the heuristic previously included them and
    // then returned a nullptr at dispatch for shapes like
    // (M=256, N=4096, K=1024, G=4). Expand the instantiation table below
    // if those tiles turn out to be optimal for real workloads.
    int block_ns[5] = {16, 32, 64, 128, 256};
    int n_block_ns = 5;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (k <= 256 && (bn > 128 || bm > 128)) continue;
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

    // SM100 masked multicast: only on B (mc_on_a=false)
    int multicast = 1;
    if (expected_m >= 512 && num_sms % 2 == 0) {
        bool legal = (ceil_div(expected_m, best_bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM100Arch>(
        KernelKind::NoSF, MmaKindLocal::BF16,
        best_bm, best_bn, block_k, multicast, false,
        2, 2, expected_m, n, 0, scfg);

    return SM100Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast, .multicast_on_a = false,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_d = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// SM100 masked kernels also need kNumGroups as compile-time constant
#define KERNEL_TYPE_SM100_MASKED(BLOCK_M, BLOCK_N, STAGES, SWIZZLE_D, NUM_MC, NGROUPS) \
    deep_gemm::sm100_bf16_gemm_impl<                                            \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                            \
        0, 0, 0,                                                                \
        BLOCK_M, BLOCK_N, 64, NGROUPS,                                          \
        128, 128, SWIZZLE_D,                                                    \
        STAGES, 128, 128,                                                       \
        NUM_MC, false, 132,                                                     \
        GemmType::MGroupedMasked, false, cutlass::bfloat16_t, 100>

#define SM100_MASKED_CONFIGS(G, TAG) \
    __attribute__((used)) static auto* _s100m_##TAG##_00 = &KERNEL_TYPE_SM100_MASKED(64, 16, 32, 32, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_01 = &KERNEL_TYPE_SM100_MASKED(64, 32, 22, 64, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_02 = &KERNEL_TYPE_SM100_MASKED(64, 64, 15, 128, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_03 = &KERNEL_TYPE_SM100_MASKED(64, 128, 10, 128, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_04 = &KERNEL_TYPE_SM100_MASKED(128, 16, 12, 32, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_05 = &KERNEL_TYPE_SM100_MASKED(128, 32, 10, 64, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_06 = &KERNEL_TYPE_SM100_MASKED(128, 64, 8, 128, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_07 = &KERNEL_TYPE_SM100_MASKED(128, 128, 6, 128, 1, G); \
    __attribute__((used)) static auto* _s100m_##TAG##_08 = &KERNEL_TYPE_SM100_MASKED(128, 256, 4, 128, 1, G);

SM100_MASKED_CONFIGS(2,  g02)
SM100_MASKED_CONFIGS(4,  g04)
SM100_MASKED_CONFIGS(8,  g08)
SM100_MASKED_CONFIGS(16, g16)
SM100_MASKED_CONFIGS(32, g32)
SM100_MASKED_CONFIGS(64, g64)

static const void* get_sm100_masked_kernel(const SM100Config& cfg, int num_groups) {
    #define MATCH_SM100M(BM, BN, ST, SD, G) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_d == SD && cfg.num_multicast == 1 && num_groups == G) \
            return (const void*)&KERNEL_TYPE_SM100_MASKED(BM, BN, ST, SD, 1, G);

    #define MATCH_SM100M_ALL_G(BM, BN, ST, SD) \
        MATCH_SM100M(BM, BN, ST, SD, 2) MATCH_SM100M(BM, BN, ST, SD, 4) \
        MATCH_SM100M(BM, BN, ST, SD, 8) MATCH_SM100M(BM, BN, ST, SD, 16) \
        MATCH_SM100M(BM, BN, ST, SD, 32) MATCH_SM100M(BM, BN, ST, SD, 64)

    MATCH_SM100M_ALL_G(64, 16, 32, 32)
    MATCH_SM100M_ALL_G(64, 32, 22, 64)
    MATCH_SM100M_ALL_G(64, 64, 15, 128)
    MATCH_SM100M_ALL_G(64, 128, 10, 128)
    MATCH_SM100M_ALL_G(128, 16, 12, 32)
    MATCH_SM100M_ALL_G(128, 32, 10, 64)
    MATCH_SM100M_ALL_G(128, 64, 8, 128)
    MATCH_SM100M_ALL_G(128, 128, 6, 128)
    MATCH_SM100M_ALL_G(128, 256, 4, 128)

    #undef MATCH_SM100M_ALL_G
    #undef MATCH_SM100M
    return nullptr;
}

static int sm100_m_grouped_masked_bf16_gemm(
    void* A, void* B, void* D, void* masked_m,
    int M, int N, int K, int num_groups, int expected_m, void* stream
) {
    cudaGetLastError();
    auto cfg = select_sm100_masked_config(expected_m, N, K, num_groups, g_num_sms);
    auto kernel_ptr = get_sm100_masked_kernel(cfg, num_groups);
    if (!kernel_ptr) return -1;

    auto tma_a = make_2d_tma(A, K, M * num_groups, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_d > 0 ? cfg.swizzle_d / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M * num_groups, d_smem_inner, cd_store_bm, N, cfg.swizzle_d);

    int* mask_ptr = (int*)masked_m;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &mask_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_d };

    return launch_kernel(kernel_ptr, 256, cfg.smem_size, cfg.num_multicast, args,
                         static_cast<cudaStream_t>(stream));
}
