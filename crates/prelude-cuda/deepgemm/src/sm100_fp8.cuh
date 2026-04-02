// SM100 (Blackwell) FP8 GEMM: Normal (1D1D with UE8M0 packed scaling factors)
// Includes FP32→UE8M0 conversion kernel, heuristic, kernel instantiations,
// dispatch, and implementation.

#pragma once

// ── UE8M0 packing kernel ──────────────────────────────────────────
// Converts FP32 scaling factors to packed UE8M0 format.
// Input:  [sf_k, mn] FP32 (row-major, K-scale groups × MN)
// Output: [ceil(sf_k/4), mn] int32 (4 FP32→UE8M0 packed per int32)
// The pack operation: (val0 >> 23) | (val1 >> 15) | (val2 >> 7) | (val3 << 1)

__global__ void pack_fp32_to_ue8m0(const float* __restrict__ input,
                                    int* __restrict__ output,
                                    int mn, int sf_k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= mn) return;

    int packed_sf_k = (sf_k + 3) / 4;
    for (int pk = blockIdx.y; pk < packed_sf_k; pk += gridDim.y) {
        uint32_t vals[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++) {
            int k_idx = pk * 4 + i;
            if (k_idx < sf_k) {
                uint32_t bits;
                float v = input[k_idx * mn + col];
                __builtin_memcpy(&bits, &v, 4);
                vals[i] = bits;
            }
        }
        uint32_t packed = (vals[0] >> 23u) | (vals[1] >> 15u) | (vals[2] >> 7u) | (vals[3] << 1u);
        output[pk * mn + col] = (int)packed;
    }
}

// Helper: launch UE8M0 pack kernel, returns workspace size needed
static void launch_ue8m0_pack(const float* input, int* output,
                               int mn, int sf_k, cudaStream_t stream) {
    int packed_sf_k = (sf_k + 3) / 4;
    int threads = std::min(mn, 256);
    dim3 grid((mn + threads - 1) / threads, packed_sf_k);
    pack_fp32_to_ue8m0<<<grid, threads, 0, stream>>>(input, output, mn, sf_k);
}

// ── SM100 FP8 heuristic ───────────────────────────────────────────

struct SM100FP8Config {
    int block_m, block_n, block_k;
    int num_stages;
    int num_multicast;
    int swizzle_a, swizzle_b, swizzle_cd;
    int smem_size;
};

static SM100FP8Config select_sm100_fp8_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;

    int block_ms[4] = {128, 256, 0, 0};
    int n_block_ms = 2;
    if (m <= 32) { block_ms[n_block_ms++] = 32; }
    if (m <= 64) { block_ms[n_block_ms++] = 64; }

    int block_ns[9] = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    int n_block_ns = 9;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align_up = [](int x, int a) { return ((x + a - 1) / a) * a; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bm == 0) continue;
            // SM100 FP8 legality checks
            if (bn % 16 != 0) continue;
            if (k <= 256 && (bn > 128 || bm > 128)) continue;
            // UTTCP constraint: (2*bn + sf_bm/32 + sf_bn/32) <= 512
            int sf_bm = align_up(bm, 128), sf_bn = align_up(bn, 128);
            if (2 * bn + sf_bm / 32 + sf_bn / 32 > 512) continue;

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

    // SM100: multicast only on B (mc_on_a=false), requires m % bm == 0
    int multicast = 1;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal = (m % best_bm == 0) && (ceil_div(m, best_bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM100Arch>(
        KernelKind::Kernel1D1D, MmaKindLocal::MXFP8FP4,
        best_bm, best_bn, block_k, multicast, false,
        1, 2, m, n, k, scfg);

    return SM100FP8Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_cd = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// ── SM100 FP8 kernel instantiations ───────────────────────────────

#define KERNEL_TYPE_SM100_FP8(BLOCK_M, BLOCK_N, STAGES, SWIZZLE_CD, NUM_MC) \
    deep_gemm::sm100_fp8_gemm_1d1d_impl<                                     \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                         \
        128, 128,                                                            \
        0, 0, 0,                                                             \
        BLOCK_M, BLOCK_N, 128, 1,                                            \
        128, 128, SWIZZLE_CD,                                                \
        STAGES, 128, 128,                                                    \
        NUM_MC, false, 132,                                                  \
        GemmType::Normal, false,                                             \
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,  \
        deep_gemm::EpilogueIdentity>

// bm=32
__attribute__((used)) static auto* _s100f_00 = &KERNEL_TYPE_SM100_FP8(32, 16, 32, 32, 1);
__attribute__((used)) static auto* _s100f_01 = &KERNEL_TYPE_SM100_FP8(32, 32, 27, 64, 1);
__attribute__((used)) static auto* _s100f_02 = &KERNEL_TYPE_SM100_FP8(32, 64, 18, 128, 1);
__attribute__((used)) static auto* _s100f_03 = &KERNEL_TYPE_SM100_FP8(32, 128, 12, 128, 1);
// bm=64
__attribute__((used)) static auto* _s100f_10 = &KERNEL_TYPE_SM100_FP8(64, 16, 27, 32, 1);
__attribute__((used)) static auto* _s100f_11 = &KERNEL_TYPE_SM100_FP8(64, 32, 18, 64, 1);
__attribute__((used)) static auto* _s100f_12 = &KERNEL_TYPE_SM100_FP8(64, 64, 12, 128, 1);
__attribute__((used)) static auto* _s100f_13 = &KERNEL_TYPE_SM100_FP8(64, 128, 8, 128, 1);
// bm=128
__attribute__((used)) static auto* _s100f_20 = &KERNEL_TYPE_SM100_FP8(128, 16, 10, 32, 1);
__attribute__((used)) static auto* _s100f_21 = &KERNEL_TYPE_SM100_FP8(128, 32, 8, 64, 1);
__attribute__((used)) static auto* _s100f_22 = &KERNEL_TYPE_SM100_FP8(128, 64, 6, 128, 1);
__attribute__((used)) static auto* _s100f_23 = &KERNEL_TYPE_SM100_FP8(128, 96, 5, 64, 1);
__attribute__((used)) static auto* _s100f_24 = &KERNEL_TYPE_SM100_FP8(128, 128, 4, 128, 1);
__attribute__((used)) static auto* _s100f_25 = &KERNEL_TYPE_SM100_FP8(128, 192, 3, 128, 1);
__attribute__((used)) static auto* _s100f_26 = &KERNEL_TYPE_SM100_FP8(128, 256, 3, 128, 1);
// bm=128 multicast
__attribute__((used)) static auto* _s100f_30 = &KERNEL_TYPE_SM100_FP8(128, 16, 10, 32, 2);
__attribute__((used)) static auto* _s100f_31 = &KERNEL_TYPE_SM100_FP8(128, 32, 8, 64, 2);
__attribute__((used)) static auto* _s100f_32 = &KERNEL_TYPE_SM100_FP8(128, 64, 6, 128, 2);
__attribute__((used)) static auto* _s100f_33 = &KERNEL_TYPE_SM100_FP8(128, 128, 4, 128, 2);
__attribute__((used)) static auto* _s100f_34 = &KERNEL_TYPE_SM100_FP8(128, 256, 3, 128, 2);
// bm=256
__attribute__((used)) static auto* _s100f_40 = &KERNEL_TYPE_SM100_FP8(256, 16, 5, 32, 1);
__attribute__((used)) static auto* _s100f_41 = &KERNEL_TYPE_SM100_FP8(256, 32, 4, 64, 1);
__attribute__((used)) static auto* _s100f_42 = &KERNEL_TYPE_SM100_FP8(256, 64, 3, 128, 1);
__attribute__((used)) static auto* _s100f_43 = &KERNEL_TYPE_SM100_FP8(256, 128, 2, 128, 1);

// ── SM100 FP8 dispatch ────────────────────────────────────────────

static const void* get_sm100_fp8_kernel(const SM100FP8Config& cfg) {
    #define MATCH_S100F(BM, BN, ST, SCD, MC) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_cd == SCD && cfg.num_multicast == MC) \
            return (const void*)&KERNEL_TYPE_SM100_FP8(BM, BN, ST, SCD, MC);

    MATCH_S100F(32, 16, 32, 32, 1) MATCH_S100F(32, 32, 27, 64, 1)
    MATCH_S100F(32, 64, 18, 128, 1) MATCH_S100F(32, 128, 12, 128, 1)
    MATCH_S100F(64, 16, 27, 32, 1) MATCH_S100F(64, 32, 18, 64, 1)
    MATCH_S100F(64, 64, 12, 128, 1) MATCH_S100F(64, 128, 8, 128, 1)
    MATCH_S100F(128, 16, 10, 32, 1) MATCH_S100F(128, 32, 8, 64, 1)
    MATCH_S100F(128, 64, 6, 128, 1) MATCH_S100F(128, 96, 5, 64, 1)
    MATCH_S100F(128, 128, 4, 128, 1) MATCH_S100F(128, 192, 3, 128, 1)
    MATCH_S100F(128, 256, 3, 128, 1)
    MATCH_S100F(128, 16, 10, 32, 2) MATCH_S100F(128, 32, 8, 64, 2)
    MATCH_S100F(128, 64, 6, 128, 2) MATCH_S100F(128, 128, 4, 128, 2)
    MATCH_S100F(128, 256, 3, 128, 2)
    MATCH_S100F(256, 16, 5, 32, 1) MATCH_S100F(256, 32, 4, 64, 1)
    MATCH_S100F(256, 64, 3, 128, 1) MATCH_S100F(256, 128, 2, 128, 1)

    #undef MATCH_S100F
    return nullptr;
}

// ── SM100 FP8 implementation ──────────────────────────────────────

/// SM100 FP8 GEMM (1D1D): accepts FP32 scales, internally packs to UE8M0.
/// scale_a: [ceil(K/128), align(M, 4)] FP32 (MN-major, M contiguous)
/// scale_b: [ceil(K/128), align(N, 4)] FP32 (MN-major, N contiguous)
static int sm100_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    int M, int N, int K, void* stream
) {
    cudaGetLastError();
    auto cfg = select_sm100_fp8_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_sm100_fp8_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto s = static_cast<cudaStream_t>(stream);
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };
    // TMA alignment: align to 16/sizeof(int)=4 for int32
    auto align_tma = [](int x) { return ((x + 3) / 4) * 4; };

    int sfa_mn = align4(M);
    int sfb_mn = align4(N);
    int sf_k = ceil_div(K, 128);
    int packed_sf_k = ceil_div(sf_k, 4);

    // Allocate workspace for packed UE8M0 scaling factors
    int sfa_packed_mn = align_tma(sfa_mn);
    int sfb_packed_mn = align_tma(sfb_mn);
    size_t ws_sfa = (size_t)packed_sf_k * sfa_packed_mn * 4;
    size_t ws_sfb = (size_t)packed_sf_k * sfb_packed_mn * 4;

    int* packed_sfa = nullptr;
    int* packed_sfb = nullptr;
    cudaMallocAsync(&packed_sfa, ws_sfa, s);
    cudaMallocAsync(&packed_sfb, ws_sfb, s);
    if (!packed_sfa || !packed_sfb) {
        if (packed_sfa) cudaFreeAsync(packed_sfa, s);
        if (packed_sfb) cudaFreeAsync(packed_sfb, s);
        return -2;
    }

    // Zero the workspace (in case sf_k is not multiple of 4)
    cudaMemsetAsync(packed_sfa, 0, ws_sfa, s);
    cudaMemsetAsync(packed_sfb, 0, ws_sfb, s);

    // Pack FP32 → UE8M0
    launch_ue8m0_pack((const float*)scale_a, packed_sfa, sfa_mn, sf_k, s);
    launch_ue8m0_pack((const float*)scale_b, packed_sfb, sfb_mn, sf_k, s);

    // TMA for A [M, K] FP8, K-major
    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    // TMA for B [K, N] FP8, K-major (col-major weight)
    auto tma_b = make_2d_tma_u8(B, K, N, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    // TMA for D [M, N] BF16
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_cd > 0 ? cfg.swizzle_cd / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cd_store_bm, N, cfg.swizzle_cd);
    // TMA for SFA: packed UE8M0 [packed_sf_k, sfa_packed_mn] int32, MN-major
    // inner=sfa_packed_mn (M values), outer=packed_sf_k
    auto tma_sfa = make_2d_tma_f32(packed_sfa, sfa_packed_mn, packed_sf_k, cfg.block_m, 1, sfa_packed_mn, 0);
    // TMA for SFB: similar
    auto tma_sfb = make_2d_tma_f32(packed_sfb, sfb_packed_mn, packed_sf_k, cfg.block_n, 1, sfb_packed_mn, 0);

    int* grouped_layout = nullptr;
    uint32_t um = M, un = N, uk = K;
    void* args[] = {
        &grouped_layout, &um, &un, &uk,
        &tma_a, &tma_b, &tma_sfa, &tma_sfb, &tma_d
    };

    int ret = launch_kernel(kernel_ptr, 256, cfg.smem_size, cfg.num_multicast, args, s);

    cudaFreeAsync(packed_sfa, s);
    cudaFreeAsync(packed_sfb, s);
    return ret;
}

// ════════════════════════════════════════════════════════════════════
// SM100 M-Grouped Contiguous FP8 GEMM
// Same kernel template, GemmType::MGroupedContiguous, block_m=128.
// ════════════════════════════════════════════════════════════════════

static SM100FP8Config select_sm100_fp8_grouped_config(int m, int n, int k, int num_sms) {
    const int block_k = 128;
    const int bm = 128; // fixed for contiguous

    int block_ns[9] = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    int n_block_ns = 9;

    int best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align_up = [](int x, int a) { return ((x + a - 1) / a) * a; };

    for (int j = 0; j < n_block_ns; j++) {
        int bn = block_ns[j];
        if (bn % 16 != 0) continue;
        int sf_bm = align_up(bm, 128), sf_bn = align_up(bn, 128);
        if (2 * bn + sf_bm / 32 + sf_bn / 32 > 512) continue;

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

    int multicast = 1;
    if (m >= 512 && num_sms % 2 == 0) {
        bool legal = (m % bm == 0) && (ceil_div(m, bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM100Arch>(
        KernelKind::Kernel1D1D, MmaKindLocal::MXFP8FP4,
        bm, best_bn, block_k, multicast, false,
        1, 2, m, n, k, scfg);

    return SM100FP8Config{
        .block_m = bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_cd = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

#define KERNEL_TYPE_SM100_FP8_GROUPED(BLOCK_N, STAGES, SWIZZLE_CD, NUM_MC) \
    deep_gemm::sm100_fp8_gemm_1d1d_impl<                                     \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                         \
        128, 128,                                                            \
        0, 0, 0,                                                             \
        128, BLOCK_N, 128, 1,                                                \
        128, 128, SWIZZLE_CD,                                                \
        STAGES, 128, 128,                                                    \
        NUM_MC, false, 132,                                                  \
        GemmType::MGroupedContiguous, false,                                 \
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,  \
        deep_gemm::EpilogueIdentity>

__attribute__((used)) static auto* _s100fg_00 = &KERNEL_TYPE_SM100_FP8_GROUPED(16, 10, 32, 1);
__attribute__((used)) static auto* _s100fg_01 = &KERNEL_TYPE_SM100_FP8_GROUPED(32, 8, 64, 1);
__attribute__((used)) static auto* _s100fg_02 = &KERNEL_TYPE_SM100_FP8_GROUPED(64, 6, 128, 1);
__attribute__((used)) static auto* _s100fg_03 = &KERNEL_TYPE_SM100_FP8_GROUPED(96, 5, 64, 1);
__attribute__((used)) static auto* _s100fg_04 = &KERNEL_TYPE_SM100_FP8_GROUPED(128, 4, 128, 1);
__attribute__((used)) static auto* _s100fg_05 = &KERNEL_TYPE_SM100_FP8_GROUPED(192, 3, 128, 1);
__attribute__((used)) static auto* _s100fg_06 = &KERNEL_TYPE_SM100_FP8_GROUPED(256, 3, 128, 1);
// multicast
__attribute__((used)) static auto* _s100fg_10 = &KERNEL_TYPE_SM100_FP8_GROUPED(32, 8, 64, 2);
__attribute__((used)) static auto* _s100fg_11 = &KERNEL_TYPE_SM100_FP8_GROUPED(64, 6, 128, 2);
__attribute__((used)) static auto* _s100fg_12 = &KERNEL_TYPE_SM100_FP8_GROUPED(128, 4, 128, 2);
__attribute__((used)) static auto* _s100fg_13 = &KERNEL_TYPE_SM100_FP8_GROUPED(256, 3, 128, 2);

static const void* get_sm100_fp8_grouped_kernel(const SM100FP8Config& cfg) {
    #define MATCH_S100FG(BN, ST, SCD, MC) \
        if (cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_cd == SCD && cfg.num_multicast == MC) \
            return (const void*)&KERNEL_TYPE_SM100_FP8_GROUPED(BN, ST, SCD, MC);

    MATCH_S100FG(16, 10, 32, 1) MATCH_S100FG(32, 8, 64, 1)
    MATCH_S100FG(64, 6, 128, 1) MATCH_S100FG(96, 5, 64, 1)
    MATCH_S100FG(128, 4, 128, 1) MATCH_S100FG(192, 3, 128, 1)
    MATCH_S100FG(256, 3, 128, 1)
    MATCH_S100FG(32, 8, 64, 2) MATCH_S100FG(64, 6, 128, 2)
    MATCH_S100FG(128, 4, 128, 2) MATCH_S100FG(256, 3, 128, 2)

    #undef MATCH_S100FG
    return nullptr;
}

static int sm100_m_grouped_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    void* grouped_layout,
    int M, int N, int K, int num_groups, void* stream
) {
    cudaGetLastError();
    auto cfg = select_sm100_fp8_grouped_config(M, N, K, g_num_sms);
    auto kernel_ptr = get_sm100_fp8_grouped_kernel(cfg);
    if (!kernel_ptr) return -1;

    auto s = static_cast<cudaStream_t>(stream);
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };

    int sfa_mn = align4(M);
    int sfb_mn = align4(N);
    int sf_k = ceil_div(K, 128);
    int packed_sf_k = ceil_div(sf_k, 4);

    int* packed_sfa = nullptr;
    int* packed_sfb = nullptr;
    size_t ws_sfa = (size_t)packed_sf_k * sfa_mn * 4;
    size_t ws_sfb = (size_t)packed_sf_k * sfb_mn * 4;
    cudaMallocAsync(&packed_sfa, ws_sfa, s);
    cudaMallocAsync(&packed_sfb, ws_sfb, s);
    if (!packed_sfa || !packed_sfb) {
        if (packed_sfa) cudaFreeAsync(packed_sfa, s);
        if (packed_sfb) cudaFreeAsync(packed_sfb, s);
        return -2;
    }
    cudaMemsetAsync(packed_sfa, 0, ws_sfa, s);
    cudaMemsetAsync(packed_sfb, 0, ws_sfb, s);
    launch_ue8m0_pack((const float*)scale_a, packed_sfa, sfa_mn, sf_k, s);
    launch_ue8m0_pack((const float*)scale_b, packed_sfb, sfb_mn, sf_k, s);

    // Contiguous: A flat [total_M, K], B grouped [G, N, K]
    auto tma_a = make_2d_tma_u8(A, K, M, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_u8(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_cd > 0 ? cfg.swizzle_cd / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M, d_smem_inner, cd_store_bm, N, cfg.swizzle_cd);
    auto tma_sfa = make_2d_tma_f32(packed_sfa, sfa_mn, packed_sf_k, cfg.block_m, 1, sfa_mn, 0);
    auto tma_sfb = make_2d_tma_f32(packed_sfb, sfb_mn, packed_sf_k, cfg.block_n, 1, sfb_mn, 0);

    int* layout_ptr = (int*)grouped_layout;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &layout_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_sfa, &tma_sfb, &tma_d };

    int ret = launch_kernel(kernel_ptr, 256, cfg.smem_size, cfg.num_multicast, args, s);
    cudaFreeAsync(packed_sfa, s);
    cudaFreeAsync(packed_sfb, s);
    return ret;
}

// ════════════════════════════════════════════════════════════════════
// SM100 M-Grouped Masked FP8 GEMM
// ════════════════════════════════════════════════════════════════════

static SM100FP8Config select_sm100_fp8_masked_config(int expected_m, int n, int k, int num_groups, int num_sms) {
    const int block_k = 128;
    int block_ms[2] = {64, 128};
    int n_block_ms = 2;

    int block_ns[9] = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    int n_block_ns = 9;

    int best_bm = 0, best_bn = 0, best_waves = 0, best_last = 0;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align_up = [](int x, int a) { return ((x + a - 1) / a) * a; };

    for (int i = 0; i < n_block_ms; i++) {
        for (int j = 0; j < n_block_ns; j++) {
            int bm = block_ms[i], bn = block_ns[j];
            if (bn % 16 != 0) continue;
            int sf_bm = align_up(bm, 128), sf_bn = align_up(bn, 128);
            if (2 * bn + sf_bm / 32 + sf_bn / 32 > 512) continue;

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

    // SM100 masked: multicast only on B
    int multicast = 1;
    if (expected_m >= 512 && num_sms % 2 == 0) {
        bool legal = (expected_m % best_bm == 0) && (ceil_div(expected_m, best_bm) % 2 == 0);
        if (legal) multicast = 2;
    }

    SmemConfig scfg;
    int best_stages = select_num_stages<SM100Arch>(
        KernelKind::Kernel1D1D, MmaKindLocal::MXFP8FP4,
        best_bm, best_bn, block_k, multicast, false,
        1, 2, expected_m, n, k, scfg);

    return SM100FP8Config{
        .block_m = best_bm, .block_n = best_bn, .block_k = block_k,
        .num_stages = best_stages,
        .num_multicast = multicast,
        .swizzle_a = scfg.swizzle_a, .swizzle_b = scfg.swizzle_b, .swizzle_cd = scfg.swizzle_cd,
        .smem_size = scfg.smem_size,
    };
}

// SM100 masked FP8: kNumGroups must match
#define KERNEL_TYPE_SM100_FP8_MASKED(BLOCK_M, BLOCK_N, STAGES, SWIZZLE_CD, NUM_MC, NGROUPS) \
    deep_gemm::sm100_fp8_gemm_1d1d_impl<                                     \
        cute::UMMA::Major::K, cute::UMMA::Major::K,                         \
        128, 128,                                                            \
        0, 0, 0,                                                             \
        BLOCK_M, BLOCK_N, 128, NGROUPS,                                      \
        128, 128, SWIZZLE_CD,                                                \
        STAGES, 128, 128,                                                    \
        NUM_MC, false, 132,                                                  \
        GemmType::MGroupedMasked, false,                                     \
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,  \
        deep_gemm::EpilogueIdentity>

#define SM100_FP8_MASKED_CONFIGS(G, TAG) \
    __attribute__((used)) static auto* _s100fm_##TAG##_00 = &KERNEL_TYPE_SM100_FP8_MASKED(64, 16, 27, 32, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_01 = &KERNEL_TYPE_SM100_FP8_MASKED(64, 32, 18, 64, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_02 = &KERNEL_TYPE_SM100_FP8_MASKED(64, 64, 12, 128, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_03 = &KERNEL_TYPE_SM100_FP8_MASKED(64, 128, 8, 128, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_04 = &KERNEL_TYPE_SM100_FP8_MASKED(128, 16, 10, 32, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_05 = &KERNEL_TYPE_SM100_FP8_MASKED(128, 32, 8, 64, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_06 = &KERNEL_TYPE_SM100_FP8_MASKED(128, 64, 6, 128, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_07 = &KERNEL_TYPE_SM100_FP8_MASKED(128, 128, 4, 128, 1, G); \
    __attribute__((used)) static auto* _s100fm_##TAG##_08 = &KERNEL_TYPE_SM100_FP8_MASKED(128, 256, 3, 128, 1, G);

SM100_FP8_MASKED_CONFIGS(2,  g02)
SM100_FP8_MASKED_CONFIGS(4,  g04)
SM100_FP8_MASKED_CONFIGS(8,  g08)
SM100_FP8_MASKED_CONFIGS(16, g16)
SM100_FP8_MASKED_CONFIGS(32, g32)
SM100_FP8_MASKED_CONFIGS(64, g64)

static const void* get_sm100_fp8_masked_kernel(const SM100FP8Config& cfg, int num_groups) {
    #define MATCH_S100FM(BM, BN, ST, SCD, G) \
        if (cfg.block_m == BM && cfg.block_n == BN && cfg.num_stages == ST && \
            cfg.swizzle_cd == SCD && cfg.num_multicast == 1 && num_groups == G) \
            return (const void*)&KERNEL_TYPE_SM100_FP8_MASKED(BM, BN, ST, SCD, 1, G);

    #define MATCH_S100FM_ALL_G(BM, BN, ST, SCD) \
        MATCH_S100FM(BM, BN, ST, SCD, 2) MATCH_S100FM(BM, BN, ST, SCD, 4) \
        MATCH_S100FM(BM, BN, ST, SCD, 8) MATCH_S100FM(BM, BN, ST, SCD, 16) \
        MATCH_S100FM(BM, BN, ST, SCD, 32) MATCH_S100FM(BM, BN, ST, SCD, 64)

    MATCH_S100FM_ALL_G(64, 16, 27, 32)
    MATCH_S100FM_ALL_G(64, 32, 18, 64)
    MATCH_S100FM_ALL_G(64, 64, 12, 128)
    MATCH_S100FM_ALL_G(64, 128, 8, 128)
    MATCH_S100FM_ALL_G(128, 16, 10, 32)
    MATCH_S100FM_ALL_G(128, 32, 8, 64)
    MATCH_S100FM_ALL_G(128, 64, 6, 128)
    MATCH_S100FM_ALL_G(128, 128, 4, 128)
    MATCH_S100FM_ALL_G(128, 256, 3, 128)

    #undef MATCH_S100FM_ALL_G
    #undef MATCH_S100FM
    return nullptr;
}

static int sm100_m_grouped_masked_fp8_gemm(
    void* A, void* B, void* D,
    void* scale_a, void* scale_b,
    void* masked_m,
    int M, int N, int K, int num_groups, int expected_m, void* stream
) {
    cudaGetLastError();
    auto cfg = select_sm100_fp8_masked_config(expected_m, N, K, num_groups, g_num_sms);
    auto kernel_ptr = get_sm100_fp8_masked_kernel(cfg, num_groups);
    if (!kernel_ptr) return -1;

    auto s = static_cast<cudaStream_t>(stream);
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    auto align4 = [](int x) { return ((x + 3) / 4) * 4; };

    // For masked: SFA is per-group [G, sf_k, align(M,4)]
    int sfa_mn = align4(M);
    int sfb_mn = align4(N);
    int sf_k = ceil_div(K, 128);
    int packed_sf_k = ceil_div(sf_k, 4);

    int* packed_sfa = nullptr;
    int* packed_sfb = nullptr;
    // SFA: num_groups * packed_sf_k * sfa_mn
    size_t ws_sfa = (size_t)packed_sf_k * num_groups * sfa_mn * 4;
    size_t ws_sfb = (size_t)packed_sf_k * num_groups * sfb_mn * 4;
    cudaMallocAsync(&packed_sfa, ws_sfa, s);
    cudaMallocAsync(&packed_sfb, ws_sfb, s);
    if (!packed_sfa || !packed_sfb) {
        if (packed_sfa) cudaFreeAsync(packed_sfa, s);
        if (packed_sfb) cudaFreeAsync(packed_sfb, s);
        return -2;
    }
    cudaMemsetAsync(packed_sfa, 0, ws_sfa, s);
    cudaMemsetAsync(packed_sfb, 0, ws_sfb, s);
    // Pack each group's SFA/SFB
    launch_ue8m0_pack((const float*)scale_a, packed_sfa, sfa_mn * num_groups, sf_k, s);
    launch_ue8m0_pack((const float*)scale_b, packed_sfb, sfb_mn * num_groups, sf_k, s);

    // Masked: A[G,M,K], B[G,N,K], D[G,M,N]
    auto tma_a = make_2d_tma_u8(A, K, M * num_groups, cfg.block_k, cfg.block_m, K, cfg.swizzle_a);
    auto tma_b = make_2d_tma_u8(B, K, N * num_groups, cfg.block_k, cfg.block_n, K, cfg.swizzle_b);
    int cd_store_bm = std::min(cfg.block_m, 128);
    int d_smem_inner = cfg.swizzle_cd > 0 ? cfg.swizzle_cd / 2 : cfg.block_n;
    auto tma_d = make_2d_tma(D, N, M * num_groups, d_smem_inner, cd_store_bm, N, cfg.swizzle_cd);
    // SFA: [packed_sf_k * num_groups, sfa_mn], SFB similar
    auto tma_sfa = make_2d_tma_f32(packed_sfa, sfa_mn, packed_sf_k * num_groups, cfg.block_m, 1, sfa_mn, 0);
    auto tma_sfb = make_2d_tma_f32(packed_sfb, sfb_mn, packed_sf_k * num_groups, cfg.block_n, 1, sfb_mn, 0);

    int* mask_ptr = (int*)masked_m;
    uint32_t um = M, un = N, uk = K;
    void* args[] = { &mask_ptr, &um, &un, &uk, &tma_a, &tma_b, &tma_sfa, &tma_sfb, &tma_d };

    int ret = launch_kernel(kernel_ptr, 256, cfg.smem_size, cfg.num_multicast, args, s);
    cudaFreeAsync(packed_sfa, s);
    cudaFreeAsync(packed_sfb, s);
    return ret;
}
