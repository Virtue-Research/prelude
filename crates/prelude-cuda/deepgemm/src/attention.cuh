// FP8 MQA attention kernels — SM90 + SM100 auto-dispatch.
//
// Based on deepseek-ai/DeepGEMM (MIT license).
//
// Provides:
//   - fp8_mqa_logits: prefill-phase MQA logit computation
//   - fp8_paged_mqa_logits: decode-phase paged MQA logits
//   - paged_mqa_logits_metadata: scheduling metadata for paged MQA
//   - clean_logits: fill -inf for out-of-range KV positions

#pragma once

#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>
#include <deep_gemm/impls/smxx_clean_logits.cuh>
#include <deep_gemm/impls/sm100_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

// ── 3D TMA descriptor for paged KV cache ──────────────────────────

static CUtensorMap make_3d_tma_u8(void* data,
                                   int dim0, int dim1, int dim2,
                                   int smem0, int smem1,
                                   int stride1_bytes, int swizzle_mode) {
    ensure_driver_api();
    CUtensorMap tmap{};
    cuuint64_t gmem_dims[3] = {(cuuint64_t)dim0, (cuuint64_t)dim1, (cuuint64_t)dim2};
    cuuint32_t smem_dims[3] = {(cuuint32_t)(swizzle_mode != 0 ? swizzle_mode : smem0),
                               (cuuint32_t)smem1, 1};
    cuuint64_t gmem_strides[2] = {(cuuint64_t)(dim0), (cuuint64_t)(stride1_bytes)};
    cuuint32_t elem_strides[3] = {1, 1, 1};
    p_cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3, data, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode_to_enum(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tmap;
}

// ── Kernel instantiation via __attribute__((used)) ────────────────
// Force template instantiation so kernels are present in the binary.
// Uses the same pattern as GEMM kernels (sm90_bf16.cuh).

// SM90 MQA logits: block_qh=128, block_kv=256, stages=3/3, tma=128, math=512
#define SM90_MQA_K(NH, HD, COMP) \
    __attribute__((used)) static auto* _sm90_mqa_##NH##_##HD##_##COMP = \
        &sm90_fp8_mqa_logits<NH, HD, COMP, 128/NH, 256, 3, 3, 128, 512>;

SM90_MQA_K(8,  64,  false) SM90_MQA_K(8,  64,  true)
SM90_MQA_K(8,  128, false) SM90_MQA_K(8,  128, true)
SM90_MQA_K(16, 64,  false) SM90_MQA_K(16, 64,  true)
SM90_MQA_K(16, 128, false) SM90_MQA_K(16, 128, true)
SM90_MQA_K(32, 64,  false) SM90_MQA_K(32, 64,  true)
SM90_MQA_K(64, 64,  false) SM90_MQA_K(64, 64,  true)
#undef SM90_MQA_K

// SM100 MQA logits: specialized=128, math=256
#define SM100_MQA_K(NH, HD, COMP) \
    __attribute__((used)) static auto* _sm100_mqa_##NH##_##HD##_##COMP = \
        &sm100_fp8_mqa_logits<NH, HD, COMP, 128/NH, 256, 3, 3, 128, 256>;

SM100_MQA_K(8,  64,  false) SM100_MQA_K(8,  64,  true)
SM100_MQA_K(8,  128, false) SM100_MQA_K(8,  128, true)
SM100_MQA_K(16, 64,  false) SM100_MQA_K(16, 64,  true)
SM100_MQA_K(16, 128, false) SM100_MQA_K(16, 128, true)
SM100_MQA_K(32, 64,  false) SM100_MQA_K(32, 64,  true)
SM100_MQA_K(64, 64,  false) SM100_MQA_K(64, 64,  true)
#undef SM100_MQA_K

// SM90 paged MQA: next_n=1, block_kv=64, stages=3/3, split_kv=256, tma=128, math=512
#define SM90_PMQA_K(NH, HD, CL2D) \
    __attribute__((used)) static auto* _sm90_pmqa_##NH##_##HD##_##CL2D = \
        &sm90_fp8_paged_mqa_logits<1, NH, HD, 64, CL2D, 3, 3, 256, 128, 512>;

SM90_PMQA_K(8,  64,  false) SM90_PMQA_K(8,  64,  true)
SM90_PMQA_K(8,  128, false) SM90_PMQA_K(8,  128, true)
SM90_PMQA_K(16, 64,  false) SM90_PMQA_K(16, 64,  true)
SM90_PMQA_K(16, 128, false) SM90_PMQA_K(16, 128, true)
SM90_PMQA_K(32, 64,  false) SM90_PMQA_K(32, 64,  true)
SM90_PMQA_K(64, 64,  false) SM90_PMQA_K(64, 64,  true)
#undef SM90_PMQA_K

// SM100 paged MQA: kNumHeads*kNextN must be 32/64/128, so with next_n=1
// only num_heads ∈ {32,64} valid. split_kv=512, stages=3/4, spec=128, math=512
#define SM100_PMQA_K(NH, HD, CL2D) \
    __attribute__((used)) static auto* _sm100_pmqa_##NH##_##HD##_##CL2D = \
        &sm100_fp8_paged_mqa_logits<1, NH, HD, 64, CL2D, 3, 4, 512, 128, 512>;

SM100_PMQA_K(32, 64,  false) SM100_PMQA_K(32, 64,  true)
SM100_PMQA_K(64, 64,  false) SM100_PMQA_K(64, 64,  true)
#undef SM100_PMQA_K

// Clean logits: next_n=1, block_kv=256, 4 warps
__attribute__((used)) static auto* _clean_logits_1 = &smxx_clean_logits<1, 256, 4>;

// ── MQA Logits dispatch ───────────────────────────────────────────

static const void* get_sm90_mqa_kernel(int nh, int hd, bool comp) {
    #define M(NH, HD, COMP) \
        if (nh == NH && hd == HD && comp == COMP) \
            return (const void*)&sm90_fp8_mqa_logits<NH, HD, COMP, 128/NH, 256, 3, 3, 128, 512>;
    M(8,64,false) M(8,64,true) M(8,128,false) M(8,128,true)
    M(16,64,false) M(16,64,true) M(16,128,false) M(16,128,true)
    M(32,64,false) M(32,64,true) M(64,64,false) M(64,64,true)
    #undef M
    return nullptr;
}

static const void* get_sm100_mqa_kernel(int nh, int hd, bool comp) {
    #define M(NH, HD, COMP) \
        if (nh == NH && hd == HD && comp == COMP) \
            return (const void*)&sm100_fp8_mqa_logits<NH, HD, COMP, 128/NH, 256, 3, 3, 128, 256>;
    M(8,64,false) M(8,64,true) M(8,128,false) M(8,128,true)
    M(16,64,false) M(16,64,true) M(16,128,false) M(16,128,true)
    M(32,64,false) M(32,64,true) M(64,64,false) M(64,64,true)
    #undef M
    return nullptr;
}

static const void* get_sm90_paged_mqa_kernel(int nh, int hd, bool cl2d) {
    #define M(NH, HD, CL2D) \
        if (nh == NH && hd == HD && cl2d == CL2D) \
            return (const void*)&sm90_fp8_paged_mqa_logits<1, NH, HD, 64, CL2D, 3, 3, 256, 128, 512>;
    M(8,64,false) M(8,64,true) M(8,128,false) M(8,128,true)
    M(16,64,false) M(16,64,true) M(16,128,false) M(16,128,true)
    M(32,64,false) M(32,64,true) M(64,64,false) M(64,64,true)
    #undef M
    return nullptr;
}

static const void* get_sm100_paged_mqa_kernel(int nh, int hd, bool cl2d) {
    #define M(NH, HD, CL2D) \
        if (nh == NH && hd == HD && cl2d == CL2D) \
            return (const void*)&sm100_fp8_paged_mqa_logits<1, NH, HD, 64, CL2D, 3, 4, 512, 128, 512>;
    M(32,64,false) M(32,64,true) M(64,64,false) M(64,64,true)
    #undef M
    return nullptr;
}

// ── Shared memory computation ─────────────────────────────────────

static int compute_mqa_logits_smem_sm90(int num_heads, int head_dim, int block_kv) {
    int block_q = 128 / num_heads;
    int smem_q_per = block_q * num_heads * head_dim; // FP8 = 1 byte
    int smem_weight_per = block_q * num_heads * 4;   // FP32
    int smem_kv_per = block_kv * head_dim;
    int smem_kv_scale_per = block_kv * 4;
    return 3 * smem_q_per + 3 * smem_kv_per + 3 * smem_weight_per + 3 * smem_kv_scale_per
           + (3 * 2 + 3 * 2) * 8;
}

static int compute_mqa_logits_smem_sm100(int num_heads, int head_dim, int block_kv) {
    int block_q = 128 / num_heads;
    int smem_q_per = block_q * num_heads * head_dim;
    int smem_weight_per = block_q * num_heads * 4;
    int smem_kv_per = block_kv * head_dim;
    int smem_kv_scale_per = align_up(block_kv * 4, 512);
    int num_wg = 2; // 256/128
    return 3 * smem_q_per + 3 * smem_weight_per + 3 * smem_kv_per + 3 * smem_kv_scale_per
           + (3 * 2 + 3 * 2 + num_wg * 2) * 8 + 4;
}

static int compute_paged_mqa_smem_sm90(int num_heads, int head_dim, int block_kv) {
    int swizzle_alignment = head_dim * 8;
    int smem_q_per = 1 * num_heads * head_dim;
    int aligned_weight_per = align_up(1 * num_heads * 4, swizzle_alignment);
    int smem_q_pipe = 3 * (smem_q_per + aligned_weight_per) + align_up(3 * 8 * 2, swizzle_alignment);
    int smem_kv_per = block_kv * head_dim;
    int aligned_kv_scale_per = align_up(block_kv * 4, swizzle_alignment);
    int smem_kv_pipe = 3 * (smem_kv_per + aligned_kv_scale_per) + align_up(3 * 8 * 2, swizzle_alignment);
    return smem_q_pipe + 4 * smem_kv_pipe; // 4 math warpgroups
}

static int compute_paged_mqa_smem_sm100(int num_heads, int head_dim, int block_kv) {
    int split_kv = 512;
    int num_wg = 4;
    int smem_q_per = 1 * num_heads * head_dim;
    int smem_kv_per = split_kv * head_dim;
    int smem_kv_scale_per = split_kv * 4;
    int smem_weight_per = 1 * num_heads * 4;
    int smem_barriers = (3 + 4) * 2 * 8;
    int smem_umma = num_wg * 2 * 8;
    return 3 * (smem_q_per + smem_weight_per) + 4 * (smem_kv_per + smem_kv_scale_per)
           + smem_barriers + smem_umma + 4;
}

// ── TMA creation helpers ──────────────────────────────────────────

// get_tma_aligned_size_local is defined in layout.cuh (included before this file)

struct MQALogitsTMA {
    CUtensorMap tma_q, tma_kv, tma_kv_scales, tma_weights;
};

static MQALogitsTMA make_mqa_logits_tma(
    void* q, void* kv, void* kv_scales, void* weights,
    int seq_len, int seq_len_kv, int num_heads, int head_dim, int block_kv
) {
    MQALogitsTMA tma;
    int block_qh = 128;
    int sw = std::min(head_dim, 128);
    tma.tma_q = make_2d_tma_u8(q, head_dim, seq_len * num_heads, head_dim, block_qh, head_dim, sw);
    tma.tma_kv = make_2d_tma_u8(kv, head_dim, seq_len_kv, head_dim, block_kv, head_dim, sw);
    int tma_slkv = get_tma_aligned_size_local(seq_len_kv, (int)sizeof(float));
    tma.tma_kv_scales = make_2d_tma_f32(kv_scales, tma_slkv, 1, block_kv, 1, tma_slkv, 0);
    int block_q = block_qh / num_heads;
    tma.tma_weights = make_2d_tma_f32(weights, num_heads, seq_len, num_heads, block_q, num_heads, 0);
    return tma;
}

struct PagedMQALogitsTMA {
    CUtensorMap tma_q, tma_kv, tma_kv_scales, tma_weights;
};

static PagedMQALogitsTMA make_paged_mqa_logits_tma(
    void* q, void* kv_cache, void* kv_scales, void* weights,
    int batch_size, int next_n, int num_heads, int head_dim,
    int num_kv_blocks, int block_kv, int kv_stride_bytes
) {
    PagedMQALogitsTMA tma;
    int sw = std::min(head_dim, 128);
    tma.tma_q = make_2d_tma_u8(q, head_dim, batch_size * next_n * num_heads,
                                head_dim, next_n * num_heads, head_dim, sw);
    tma.tma_kv = make_3d_tma_u8(kv_cache, head_dim, block_kv, num_kv_blocks,
                                 head_dim, block_kv, kv_stride_bytes, sw);
    int kv_scale_stride = kv_stride_bytes / (int)sizeof(float);
    tma.tma_kv_scales = make_2d_tma_f32(kv_scales, block_kv, num_kv_blocks,
                                         block_kv, 1, kv_scale_stride, 0);
    tma.tma_weights = make_2d_tma_f32(weights, next_n * num_heads, batch_size,
                                       next_n * num_heads, 1, next_n * num_heads, 0);
    return tma;
}

// ── MQA Logits launch ─────────────────────────────────────────────

static int sm90_fp8_mqa_logits_launch(
    void* q, void* kv, void* kv_scales, void* weights,
    void* cu_seq_len_k_start, void* cu_seq_len_k_end, void* logits,
    int seq_len, int seq_len_kv, int max_seqlen_k,
    int num_heads, int head_dim, int stride_logits, void* stream
) {
    bool comp = (max_seqlen_k > 0);
    auto kp = get_sm90_mqa_kernel(num_heads, head_dim, comp);
    if (!kp) return -1;
    int smem = compute_mqa_logits_smem_sm90(num_heads, head_dim, 256);
    auto tma = make_mqa_logits_tma(q, kv, kv_scales, weights, seq_len, seq_len_kv, num_heads, head_dim, 256);
    uint32_t sl = seq_len, slkv = seq_len_kv, msk = max_seqlen_k;
    uint64_t strl = stride_logits;
    void* args[] = {&sl, &slkv, &msk, &strl, &cu_seq_len_k_start, &cu_seq_len_k_end,
                    &logits, &tma.tma_q, &tma.tma_kv, &tma.tma_kv_scales, &tma.tma_weights};
    return launch_kernel(kp, 640, smem, 1, args, (cudaStream_t)stream);
}

static int sm100_fp8_mqa_logits_launch(
    void* q, void* kv, void* kv_scales, void* weights,
    void* cu_seq_len_k_start, void* cu_seq_len_k_end, void* logits,
    int seq_len, int seq_len_kv, int max_seqlen_k,
    int num_heads, int head_dim, int stride_logits, void* stream
) {
    bool comp = (max_seqlen_k > 0);
    auto kp = get_sm100_mqa_kernel(num_heads, head_dim, comp);
    if (!kp) return -1;
    int smem = compute_mqa_logits_smem_sm100(num_heads, head_dim, 256);
    auto tma = make_mqa_logits_tma(q, kv, kv_scales, weights, seq_len, seq_len_kv, num_heads, head_dim, 256);
    uint32_t sl = seq_len, slkv = seq_len_kv, msk = max_seqlen_k;
    uint64_t strl = stride_logits;
    void* args[] = {&sl, &slkv, &msk, &strl, &cu_seq_len_k_start, &cu_seq_len_k_end,
                    &logits, &tma.tma_q, &tma.tma_kv, &tma.tma_kv_scales, &tma.tma_weights};
    return launch_kernel(kp, 384, smem, 1, args, (cudaStream_t)stream);
}

// ── Paged MQA Logits launch ───────────────────────────────────────

static int sm90_fp8_paged_mqa_logits_launch(
    void* q, void* kv_cache, void* kv_scales, void* weights,
    void* ctx_lens, void* logits, void* block_table, void* sched_meta,
    int batch_size, int num_heads, int head_dim, int num_kv_blocks, int block_kv,
    bool cl2d, int kv_stride_bytes, int logits_stride, int block_table_stride, void* stream
) {
    auto kp = get_sm90_paged_mqa_kernel(num_heads, head_dim, cl2d);
    if (!kp) return -1;
    int smem = compute_paged_mqa_smem_sm90(num_heads, head_dim, block_kv);
    auto tma = make_paged_mqa_logits_tma(q, kv_cache, kv_scales, weights,
                                          batch_size, 1, num_heads, head_dim,
                                          num_kv_blocks, block_kv, kv_stride_bytes);
    uint32_t bs = batch_size;
    uint64_t ls = logits_stride, bts = block_table_stride;
    void* args[] = {&bs, &ls, &bts, &ctx_lens, &logits, &block_table, &sched_meta,
                    &tma.tma_q, &tma.tma_kv, &tma.tma_kv_scales, &tma.tma_weights};
    return launch_kernel(kp, 640, smem, 1, args, (cudaStream_t)stream);
}

static int sm100_fp8_paged_mqa_logits_launch(
    void* q, void* kv_cache, void* kv_scales, void* weights,
    void* ctx_lens, void* logits, void* block_table, void* sched_meta,
    int batch_size, int num_heads, int head_dim, int num_kv_blocks, int block_kv,
    bool cl2d, int kv_stride_bytes, int logits_stride, int block_table_stride, void* stream
) {
    auto kp = get_sm100_paged_mqa_kernel(num_heads, head_dim, cl2d);
    if (!kp) return -1;
    int smem = compute_paged_mqa_smem_sm100(num_heads, head_dim, block_kv);
    auto tma = make_paged_mqa_logits_tma(q, kv_cache, kv_scales, weights,
                                          batch_size, 1, num_heads, head_dim,
                                          num_kv_blocks, block_kv, kv_stride_bytes);
    uint32_t bs = batch_size;
    uint64_t ls = logits_stride, bts = block_table_stride;
    void* args[] = {&bs, &ls, &bts, &ctx_lens, &logits, &block_table, &sched_meta,
                    &tma.tma_q, &tma.tma_kv, &tma.tma_kv_scales, &tma.tma_weights};
    return launch_kernel(kp, 640, smem, 1, args, (cudaStream_t)stream);
}

// ── Paged MQA Metadata — runtime-parameterized ────────────────────

__global__ void paged_mqa_metadata_kernel(
    uint32_t batch_size, uint32_t next_n, bool is_context_lens_2d,
    uint32_t split_kv, uint32_t num_sms,
    const uint32_t* __restrict__ context_lens,
    uint32_t* __restrict__ schedule_metadata
) {
    extern __shared__ uint32_t smem_meta_[];
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t aligned_batch = ((batch_size + 31) / 32) * 32;

    uint32_t sum = 0;
    for (uint32_t k = 0; k < aligned_batch / 32; k++) {
        uint32_t q_idx = k * 32 + lane_idx;
        uint32_t lens_idx = is_context_lens_2d ? q_idx * next_n + next_n - 1 : q_idx;
        uint32_t ctx_len = (q_idx < batch_size) ? __ldg(context_lens + lens_idx) : 0;
        uint32_t num_segs = (ctx_len + split_kv - 1) / split_kv;

        uint32_t x = num_segs;
        for (uint32_t off = 1; off < 32; off <<= 1) {
            uint32_t y = __shfl_up_sync(0xffffffff, x, off);
            x += (lane_idx >= off ? y : 0);
        }
        x += sum;
        smem_meta_[k * 32 + lane_idx] = x;
        sum = __shfl_sync(0xffffffff, x, 31);
    }

    uint32_t q = sum / num_sms, r = sum % num_sms;
    for (uint32_t sm_idx = lane_idx; sm_idx <= num_sms; sm_idx += 32) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
        uint32_t q_idx = 0;
        while (q_idx < batch_size && smem_meta_[q_idx] <= seg_starts)
            ++q_idx;
        uint32_t kv_split_idx = (q_idx == 0) ? seg_starts : seg_starts - smem_meta_[q_idx - 1];
        __syncwarp();
        schedule_metadata[sm_idx * 2] = q_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

static int paged_mqa_metadata_launch(
    void* context_lens, void* schedule_metadata,
    int batch_size, int next_n, bool is_context_lens_2d,
    int split_kv, int num_sms, void* stream
) {
    int aligned_batch = ((batch_size + 31) / 32) * 32;
    int smem = aligned_batch * (int)sizeof(uint32_t);
    paged_mqa_metadata_kernel<<<1, 32, smem, (cudaStream_t)stream>>>(
        (uint32_t)batch_size, (uint32_t)next_n, is_context_lens_2d,
        (uint32_t)split_kv, (uint32_t)num_sms,
        (const uint32_t*)context_lens, (uint32_t*)schedule_metadata);
    if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); return -2; }
    return 0;
}

// ── Clean Logits launch ───────────────────────────────────────────

static int clean_logits_launch(
    void* cu_seq_len_k_start, void* cu_seq_len_k_end, void* logits,
    int seq_len, int seq_len_kv, int stride_logits, int next_n, void* stream
) {
    if (next_n != 1) return -1;
    const void* kp = (const void*)&smxx_clean_logits<1, 256, 4>;
    uint32_t sl = seq_len, slkv = seq_len_kv;
    uint64_t strl = stride_logits;
    void* args[] = {&sl, &slkv, &strl, &cu_seq_len_k_start, &cu_seq_len_k_end, &logits};
    return launch_kernel(kp, 128, 256 * (int)sizeof(float), 1, args, (cudaStream_t)stream);
}
