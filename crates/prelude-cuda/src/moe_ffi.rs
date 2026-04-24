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
}
