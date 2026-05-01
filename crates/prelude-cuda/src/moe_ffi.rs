//! FFI bindings for the MOE WMMA kernel (compiled from moe_wmma.cu).

use std::ffi::c_void;

unsafe extern "C" {
    pub fn moe_gemm_wmma(
        input: *const c_void,
        weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        expert_counts: *mut i32,
        expert_offsets: *mut i32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32,
        is_prefill: bool,
        stream: i64,
    );

    /// In-place swap of gate/up halves in expert weights.
    pub fn moe_swap_gate_up_inplace(
        data: *mut c_void,
        num_experts: i32,
        inter: i32,
        hidden: i32,
        stream: i64,
    );

    /// Warp-reduction GEMV kernel for MoE decode (size_m small).
    ///
    /// Same signature as `moe_gemm_wmma` minus the expert_counts / expert_offsets
    /// scratch (this kernel indexes via sorted_token_ids + expert_ids directly
    /// per block, so it needs no per-expert layout scan). `dtype` mirrors
    /// the WMMA convention (0=fp16, 1=bf16).
    pub fn moe_gemv(
        input: *const c_void,
        weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32,
        stream: i64,
    );


    /// GPU sort of expert assignments using thrust::sort_by_key.
    pub fn moe_sort_expert_assignments(
        expert_ids_in: *const u32,
        n: i32,
        sorted_experts: *mut u32,
        sorted_tokens: *mut u32,
        stream: i64,
    );

    /// Per-expert prefix-sum offsets from sorted expert ids. Reused by
    /// the SM100 grouped GEMM dispatch so it can call the CUTLASS
    /// grouped path without re-implementing the histogram.
    pub fn moe_compute_expert_offsets_light(
        sorted_expert_ids: *const i32,
        size_m: i32,
        num_experts: i32,
        expert_counts_tmp: *mut i32,
        expert_offsets: *mut i32,
        stream: i64,
    );
}
