// Thin C wrapper for cuLA kernels — strips ATen/PyTorch dependency.
// Exposes extern "C" functions accepting raw pointers for Rust FFI.

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/arch/arch.h>

// ── SM90: Fused KDA prefill ──────────────────────────────────────────

#include "kda/sm90/prefill_kernel.hpp"

extern "C" int cula_kda_fwd_prefill_sm90(
    cudaStream_t stream,
    void* output,
    float* output_state,
    const void* q,
    const void* k,
    const void* v,
    const float* input_state,
    const float* alpha,
    const float* beta,
    const int32_t* cu_seqlens,
    uint8_t* workspace,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    int safe_gate,
    int32_t sm_count,
    int32_t num_k_heads)
{
    using bf16 = cute::bfloat16_t;
    using Sm90 = cutlass::arch::Sm90;

    kda::sm90::launch_kda_fwd_prefill_kernel<Sm90, bf16, bf16, float>(
        stream,
        reinterpret_cast<bf16*>(output),
        output_state,
        reinterpret_cast<bf16 const*>(q),
        reinterpret_cast<bf16 const*>(k),
        reinterpret_cast<bf16 const*>(v),
        input_state,
        alpha,
        beta,
        cu_seqlens,
        workspace,
        num_seqs,
        num_heads,
        head_size,
        total_seqlen,
        scale,
        static_cast<bool>(safe_gate),
        sm_count,
        num_k_heads);

    return 0;
}

// ── SM100: Chunked intra-attention ───────────────────────────────────

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 || !defined(__CUDA_ARCH__)
// Only include SM100 code for fat binary or host-side compilation
#ifdef CULA_SM100_ENABLED

#include "kda/sm100/kda_fwd_common.cuh"

extern "C" int cula_chunk_kda_fwd_intra_sm100(
    cudaStream_t stream,
    const void* q,
    const void* k,
    const void* g,
    const void* beta,
    const int32_t* cu_seqlens,
    const int32_t* chunk_indices,
    void* Aqk_out,
    void* Akk_out,
    int* tile_counter,
    float scale,
    int chunk_size,
    int total_q_len,
    int b,
    int h,
    int d,
    int num_tiles,
    int use_tf32_inverse,
    int unified_gref,
    int num_sm)
{
    KDA_fwd_intra_params params;
    params.total_q_len = total_q_len;
    params.b = b;
    params.h = h;
    params.d = d;
    params.chunk_size = chunk_size;
    params.scale = scale;
    params.use_tf32_inverse = static_cast<bool>(use_tf32_inverse);
    params.unified_gref = static_cast<bool>(unified_gref);
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k);
    params.g_ptr = const_cast<void*>(g);
    params.beta_ptr = const_cast<void*>(beta);
    params.cu_seqlens_ptr = const_cast<void*>(static_cast<const void*>(cu_seqlens));
    params.chunk_indices_ptr = const_cast<void*>(static_cast<const void*>(chunk_indices));
    params.Aqk_out_ptr = Aqk_out;
    params.Akk_out_ptr = Akk_out;
    params.shape_Akk = cute::make_shape(total_q_len, chunk_size, h);
    params.stride_Akk = cute::make_stride(chunk_size * h, cute::_1{}, chunk_size);
    params.num_sm = num_sm;
    params.tile_scheduler_params =
        StaticPersistentTileScheduler::Params{num_tiles, h, num_sm, tile_counter};

    kda::sm100::run_kda_fwd_intra_sm100(params, stream);
    return 0;
}

extern "C" int cula_chunk_kda_fwd_recomp_wu_sm100(
    cudaStream_t stream,
    const void* k,
    const void* v,
    const void* beta,
    const void* A,
    const void* g,
    const int32_t* cu_seqlens,
    const int32_t* chunk_indices,
    void* w_out,
    void* u_out,
    void* kg_out,
    int chunk_size,
    int total_len,
    int b,
    int h,
    int d,
    int num_tiles,
    int num_sm)
{
    KDA_fwd_recomp_w_u_params params;
    params.total_len = total_len;
    params.b = b;
    params.h = h;
    params.d = d;
    params.chunk_size = chunk_size;
    params.k_ptr = const_cast<void*>(k);
    params.v_ptr = const_cast<void*>(v);
    params.beta_ptr = const_cast<void*>(beta);
    params.A_ptr = const_cast<void*>(A);
    params.g_ptr = const_cast<void*>(g);
    params.cu_seqlens_ptr = const_cast<void*>(static_cast<const void*>(cu_seqlens));
    params.chunk_indices_ptr = const_cast<void*>(static_cast<const void*>(chunk_indices));
    params.w_out_ptr = w_out;
    params.u_out_ptr = u_out;
    params.kg_out_ptr = kg_out;
    params.shape_wukg = cute::make_shape(total_len, d, h);
    params.stride_wukg = cute::make_stride(d * h, cute::_1{}, d);
    params.num_sm = num_sm;
    params.tile_scheduler_params =
        StaticPersistentTileScheduler::Params{num_tiles, h, num_sm, nullptr};

    kda::sm100::run_kda_fwd_recomp_w_u_sm100(params, stream);
    return 0;
}

#endif // CULA_SM100_ENABLED
#endif // SM100 guard
