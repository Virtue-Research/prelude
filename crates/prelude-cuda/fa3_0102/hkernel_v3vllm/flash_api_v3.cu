// Raw-pointer, torch-free C entry for the vendored vllm-flash-attn hopper kernel.
// Mirrors the params setup + persistent varlen scheduler launch of vllm-fa's
// flash_api.cpp (set_params_fprop + scheduler-metadata + prepare_varlen_num_blocks +
// run_mha_fwd), restricted to the candle-fa3-0102 use case: bf16, headdim 128, sm90,
// varlen (cu_seqlens), causal or full, GQA. num_splits = 1 (no split-KV).
//
// The candle binding allocates: softmax_lse (num_heads*total_q f32) and a zeroed int32
// scheduler-metadata scratch of size >= 1 + round_up(b,4)*2, and passes their pointers.

#include "flash.h"
#include "tile_size.h"
#include "heuristics.h"
#include "static_switch.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cutlass/numeric_types.h>

// Defined by the vendored instantiation .cu (flash_fwd_hdim128_bf16[_packgqa]_sm90.cu).
template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split,
          bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

// Defined in flash_prepare_scheduler.cu.
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl);

#ifdef FA3_CLOCK_PROBE
extern "C" void fa3_read_probe(unsigned long long*, unsigned long long*, unsigned*);
#endif

static inline int round_up_headdim(int d) {
    if (d <= 64) return 64;
    if (d <= 96) return 96;
    if (d <= 128) return 128;
    if (d <= 192) return 192;
    return 256;
}

extern "C" void run_mha_v3(
    void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr, void *softmax_lse_ptr,
    const int *cu_seqlens_q, const int *cu_seqlens_k,
    int *sched_metadata,           // zeroed int32 buffer, size >= 1 + round_up(b,4)*2
    uint32_t q_row_stride, uint32_t k_row_stride, uint32_t v_row_stride, uint32_t o_row_stride,
    uint32_t q_head_stride, uint32_t k_head_stride, uint32_t v_head_stride, uint32_t o_head_stride,
    uint32_t b, uint32_t h, uint32_t h_k, uint32_t d,
    uint32_t max_seqlen_q, uint32_t max_seqlen_k,
    uint32_t total_q, uint32_t total_k,
    float softmax_scale, int is_causal, int is_bf16) {

    Flash_fwd_params params{};
    params.is_bf16 = is_bf16 != 0;
    params.is_e4m3 = false;

    params.q_ptr = q_ptr; params.k_ptr = k_ptr; params.v_ptr = v_ptr; params.o_ptr = o_ptr;
    params.q_row_stride = q_row_stride; params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride; params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride; params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride; params.o_head_stride = o_head_stride;
    params.v_dim_stride = 1;  // contiguous last dim
    // varlen: batch strides unused (cu_seqlens drives addressing)
    params.cu_seqlens_q = const_cast<int *>(cu_seqlens_q);
    params.cu_seqlens_k = const_cast<int *>(cu_seqlens_k);
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.b = b; params.h = h; params.h_k = h_k;
    params.seqlen_q = max_seqlen_q; params.seqlen_k = max_seqlen_k;
    params.total_q = total_q; params.total_k = total_k;  // varlen: total token counts (TMA layout)
    auto rum = [](int x, int m) { return (x + m - 1) / m * m; };
    params.seqlen_q_rounded = rum(max_seqlen_q, 128);
    params.seqlen_k_rounded = rum(max_seqlen_k, 128);
    params.d = d; params.d_rounded = round_up_headdim(d);
    params.dv = d; params.dv_rounded = params.d_rounded;

    params.scale_softmax = softmax_scale;
    params.softcap = 0.f;
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = uint8_t(255);
    params.rp_dropout = 1.f;

    // causal == window_right=0, window_left<0 (matches set_params_fprop)
    params.is_causal = is_causal != 0;
    params.is_local = false;
    params.window_size_left = params.is_causal ? int(max_seqlen_k) - 1 : -1;
    params.window_size_right = params.is_causal ? 0 : -1;

    // Cache device props (cudaGetDeviceProperties is ~ms; called per forward would dominate).
    static int s_arch = 0, s_num_sm = 0;
    if (s_num_sm == 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        s_arch = prop.major * 10 + prop.minor;
        s_num_sm = prop.multiProcessorCount;
    }
    params.arch = s_arch;
    params.num_sm = s_num_sm;

    params.page_size = 1; params.page_table = nullptr;
    // CP (context parallelism) disabled: world_size MUST be 1, not 0. The mainloop divides
    // by cp_world_size when computing n_block_min_causal_local_mask; 0 makes that garbage
    // (negative), so EVERY n_block ran the masked slow path -> ~34% uniform slowdown while
    // staying bit-correct. This was the entire "residual 1.2x vs vLLM" gap.
    params.cp_world_size = 1;
    params.cp_rank = 0;
    params.cp_tot_seqused_k = nullptr;
    params.num_splits = 1;                       // no split-KV (vLLM also uses 1 here)
    // PackGQA=true matches vLLM's auto-pick for GQA and is faster (274.6 vs 286.5 us/batch
    // on the tg dataset) now that the cp_world_size=0 mask bug is fixed. (The old "PackGQA
    // slower" measurement was an artifact of that bug: the always-masked path penalized
    // PackGQA disproportionately.)
    params.pack_gqa = h != h_k;
    { const char *pg = getenv("V3_PACKGQA"); if (pg) params.pack_gqa = (pg[0] == '1'); }
    // PDL (Programmatic Dependent Launch): overlap prepare->main kernel dependency, as vLLM
    // does for small batch (prepare_varlen_pdl = b <= PREPARE_VARLEN_MAX_BATCHES_1CTA).
    params.prepare_varlen_pdl = true;
    { const char *p = getenv("V3_NOPDL"); if (p && p[0]=='1') params.prepare_varlen_pdl = false; }
    params.varlen_sort_batches = false;
    params.head_swizzle = params.is_causal || params.is_local;

    // Scheduler-metadata layout (mirrors flash_api.cpp, num_splits==1 => no dynamic split):
    //   [ prepare_seqlen_q (b_rounded) | (head_swizzle? num_nheads_in_l2 (b_rounded)) | semaphore(1) ]
    int b_rounded = rum(int(b), 4);
    int num_prepare_batch_vectors = 1 + (params.head_swizzle ? 1 : 0);
    int head_swizzle_offset = b_rounded * (num_prepare_batch_vectors - 1);
    int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    params.prepare_seqlen_q_ptr = sched_metadata;
    params.num_splits_dynamic_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? sched_metadata + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = sched_metadata + tile_count_semaphore_offset;

    cudaStream_t stream = 0;

    // Do NOT call prepare_varlen_num_blocks here: run_mha_fwd_ launches it internally
    // (flash_fwd_launch_template.h, `if (Varlen && !skip_scheduler_metadata_computation)`)
    // with the kernel's own correct kBlockM/N. Calling it here too caused a redundant 2nd
    // prepare launch with possibly-mismatched block sizes. Leave skip_* = false (default).
    BOOL_SWITCH(params.pack_gqa, PackGQA, [&] {
        run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, PackGQA>(params, stream);
    });

#ifdef FA3_CLOCK_PROBE
    cudaStreamSynchronize(stream);
    static unsigned long long s[256], e[256]; static unsigned mm[256];
    fa3_read_probe(s, e, mm);
    unsigned long long durs[256]; int n = 0;
    for (int i = 0; i < 256; i++) if (e[i] > s[i] && s[i] > 0) durs[n++] = e[i] - s[i];
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (durs[j] < durs[i]) { auto t=durs[i]; durs[i]=durs[j]; durs[j]=t; }
    if (n > 0) fprintf(stderr, "[probe] active_CTAs=%d  per-CTA dur(clk): min=%llu p50=%llu p95=%llu max=%llu  max/p50=%.2f\n",
                       n, durs[0], durs[n/2], durs[(n*95)/100], durs[n-1], (double)durs[n-1]/(double)durs[n/2]);
#endif
}

// ───────────────────────────────────────────────────────────────────────────
// Prelude serving entry: stream-aware + paged-KV-capable. Field-for-field port
// of vLLM 0.22's host logic (vllm-project/flash-attention@bce2942,
// hopper/flash_api.cpp: mha_fwd + set_params_fprop + get_pagedkv_tma +
// get_pack_gqa), restricted to bf16 / hdim128 / sm90 / fwd / num_splits=1.
//
// Paged convention (mirrors mha_fwd exactly):
//   k/v cache:  (num_pages, page_size, h_k, d), contiguous last dim
//   page_table: (b, max_num_pages_per_seq) int32, contiguous last dim
//   seqused_k:  (b,) int32, REQUIRED when paged; cu_seqlens_k must be NULL
//               ("If cu_seqlens_k is passed in, then page table is not supported")
//   total_k        = b * page_size        (mha_fwd: batch_size * k.size(1))
//   k_batch_stride = k.stride(0)          (page stride = page_size*h_k*d)
//   seqlen_k       = max_seqlen_k
// Dispatch (mha_fwd run_mha_fwd PAGEDKV_SWITCH/PACKGQA_SWITCH):
//   pagedkv_tma  = page_size % kBlockN == 0 && seqlen_q*(h/h_k) > kBlockM
//   page_table && !pagedkv_tma  -> PagedKVNonTMA=true kernel, PackGQA forced true
//   otherwise                   -> PagedKVNonTMA=false kernel (TMA reads pages)

// Copy of mha_fwd's get_pagedkv_tma (params.page_table/leftpad_k/knew_ptr/is_local,
// d_rounded, seqlen_q, h, h_k must already be set).
static inline bool get_pagedkv_tma_v3(Flash_fwd_params const& params) {
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr || params.is_local) { return false; }
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(
        params.d_rounded, params.dv_rounded, params.is_causal, params.is_local,
        params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/,
        false /*paged_kv_non_TMA*/, params.softcap > 0.f, use_one_mma_wg(params));
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower.
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

extern "C" void run_mha_v3_prelude(
    void *stream_ptr,              // cudaStream_t to launch on (prelude model stream)
    void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr, void *softmax_lse_ptr,
    const int *cu_seqlens_q,       // (b+1,) int32, required
    const int *cu_seqlens_k,       // (b+1,) int32 for non-paged varlen K; NULL when paged
    const int *seqused_k,          // (b,) int32; required when paged, optional otherwise
    const int *page_table,         // (b, max_num_pages_per_seq) int32; NULL = non-paged
    int *sched_metadata,           // zeroed int32 buffer, size >= 1 + round_up(b,4)*2
    uint32_t page_size, uint32_t num_pages, uint32_t page_table_batch_stride,
    uint64_t k_batch_stride, uint64_t v_batch_stride,  // page stride when paged; 0 otherwise
    uint32_t q_row_stride, uint32_t k_row_stride, uint32_t v_row_stride, uint32_t o_row_stride,
    uint32_t q_head_stride, uint32_t k_head_stride, uint32_t v_head_stride, uint32_t o_head_stride,
    uint32_t b, uint32_t h, uint32_t h_k, uint32_t d,
    uint32_t max_seqlen_q, uint32_t max_seqlen_k,
    uint32_t total_q, uint32_t total_k,
    float softmax_scale, int is_causal_in, int is_bf16,
    // Q-prologue fusion (in-kernel per-head RMSNorm + RoPE on raw Q).
    // Active when qnorm_weight != NULL; rotary tables are (max_pos, d/2)
    // non-interleaved (rotate-half), Q positions = seqused_k - seqlen_q + i.
    // The caller MUST route K through a norm+rope path with the same
    // convention (see fa3_q_prologue_deadlock_fix report: K-path mismatch
    // produces deterministic garbage).
    const void *rotary_cos, const void *rotary_sin,
    const void *qnorm_weight, float qnorm_eps) {

    bool const paged_KV = page_table != nullptr;
    Flash_fwd_params params{};
    params.is_bf16 = is_bf16 != 0;
    params.is_e4m3 = false;

    params.q_ptr = q_ptr; params.k_ptr = k_ptr; params.v_ptr = v_ptr; params.o_ptr = o_ptr;
    params.q_row_stride = q_row_stride; params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride; params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride; params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride; params.o_head_stride = o_head_stride;
    params.v_dim_stride = 1;  // contiguous last dim
    // set_params_fprop: k/v batch strides only when cu_seqlens_k == NULL (paged or dense).
    if (cu_seqlens_k == nullptr) {
        params.k_batch_stride = k_batch_stride;
        params.v_batch_stride = v_batch_stride;
    }
    params.cu_seqlens_q = const_cast<int *>(cu_seqlens_q);
    params.cu_seqlens_k = const_cast<int *>(cu_seqlens_k);
    params.seqused_q = nullptr;
    params.seqused_k = const_cast<int *>(seqused_k);
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.b = b; params.h = h; params.h_k = h_k;
    params.b_k = b;  // mha_fwd: batch_size_k = page_table.size(0) when paged
    int const seqlen_q = max_seqlen_q;
    int const seqlen_k = max_seqlen_k;  // mha_fwd: max_seqlen_k_.value()
    params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
    params.total_q = total_q; params.total_k = total_k;
    auto rum = [](int x, int m) { return (x + m - 1) / m * m; };
    params.seqlen_q_rounded = rum(seqlen_q, 128);
    params.seqlen_k_rounded = rum(seqlen_k, 128);
    params.d = d; params.d_rounded = round_up_headdim(d);
    params.dv = d; params.dv_rounded = params.d_rounded;

    params.scale_softmax = softmax_scale;
    params.softcap = 0.f;
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = uint8_t(255);
    params.rp_dropout = 1.f;

    // mha_fwd window/causal normalization (window starts at (-1, -1)):
    bool is_causal = is_causal_in != 0;
    int window_size_left = -1, window_size_right = -1;
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    // causal=true is the same as causal=false in this case, EXCEPT hdim 128 + pagedKV
    // where causal keeps kBlockN=128 (better for paged TMA).
    if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
        if ((int(d) <= 64 || int(d) > 128) || !paged_KV) { is_causal = false; }
    }
    if (is_causal) { window_size_right = 0; }
    is_causal = window_size_left < 0 && window_size_right == 0;
    // set_params_fprop:
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    static int s_arch = 0, s_num_sm = 0;
    if (s_num_sm == 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        s_arch = prop.major * 10 + prop.minor;
        s_num_sm = prop.multiProcessorCount;
    }
    params.arch = s_arch;
    params.num_sm = s_num_sm;

    // Q-prologue fusion: in-kernel RMSNorm+RoPE on Q (mainloop runtime branch
    // gated on qnorm_weight_ptr != NULL && rotary_dim > 0).
    params.rotary_cos_ptr = const_cast<void *>(rotary_cos);
    params.rotary_sin_ptr = const_cast<void *>(rotary_sin);
    params.qnorm_weight_ptr = const_cast<void *>(qnorm_weight);
    params.qnorm_eps = qnorm_eps;
    params.rotary_dim = qnorm_weight != nullptr ? int(d) : 0;
    params.is_rotary_interleaved = false;
    params.seqlens_rotary = nullptr;

    // Paged KV (mha_fwd ordering: page params set BEFORE get_pagedkv_tma).
    if (paged_KV) {
        params.page_table = const_cast<int *>(page_table);
        params.page_table_batch_stride = page_table_batch_stride;
    }
    params.page_size = paged_KV ? int(page_size) : 1;
    params.num_pages = paged_KV ? int(num_pages) : 0;
    params.pagedkv_tma = get_pagedkv_tma_v3(params);
    bool const paged_nontma = paged_KV && !params.pagedkv_tma;

    // cp_world_size MUST be 1 (0 => div-by-zero garbage in the causal-mask bound =>
    // every KV block runs the masked slow path; see report section 8).
    params.cp_world_size = 1;
    params.cp_rank = 0;
    params.cp_tot_seqused_k = nullptr;
    params.num_splits = 1;
    // get_pack_gqa: PagedKVNonTMA forces PackGQA=true (only that instantiation is
    // compiled); otherwise should_pack_gqa(varlen_q=true, ...) == true, matching vLLM
    // serving. V3_PACKGQA=0/1 overrides the non-forced case.
    params.pack_gqa = true;
    if (!paged_nontma) {
        const char *pg = getenv("V3_PACKGQA");
        if (pg) params.pack_gqa = (pg[0] == '1');
    }
    // mha_fwd: prepare_varlen_pdl = use_prepare_varlen && b <= PREPARE_VARLEN_MAX_BATCHES_1CTA(992)
    params.prepare_varlen_pdl = b <= 992;
    { const char *p = getenv("V3_NOPDL"); if (p && p[0]=='1') params.prepare_varlen_pdl = false; }
    params.varlen_sort_batches = false;
    params.head_swizzle = params.is_causal || params.is_local;

    // Scheduler-metadata layout (mirrors flash_api.cpp, num_splits==1):
    //   [ prepare_seqlen_q (b_rounded) | (head_swizzle? num_nheads_in_l2 (b_rounded)) | semaphore(1) ]
    int b_rounded = rum(int(b), 4);
    int num_prepare_batch_vectors = 1 + (params.head_swizzle ? 1 : 0);
    int head_swizzle_offset = b_rounded * (num_prepare_batch_vectors - 1);
    int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    params.prepare_seqlen_q_ptr = sched_metadata;
    params.num_splits_dynamic_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? sched_metadata + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = sched_metadata + tile_count_semaphore_offset;

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // prepare_varlen_num_blocks is launched inside run_mha_fwd_ (skip_* = false).
    if (paged_nontma) {
        // mha_fwd: PAGEDKV_SWITCH(true) + PackGQA forced true.
        run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, true, false, true>(params, stream);
    } else {
        BOOL_SWITCH(params.pack_gqa, PackGQA, [&] {
            run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, PackGQA>(params, stream);
        });
    }
}
