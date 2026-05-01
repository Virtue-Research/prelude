//! Minimal Rust FFI bindings to Intel oneDNN, plus a rayon-backed
//! threadpool adapter. Static link only — the crate build compiles
//! oneDNN as a static library and links it into the final Rust binary, so
//! there is no `libdnnl.so` at runtime.
//!
//! ## What this crate exposes
//!
//! - Raw `extern "C"` declarations for the subset of oneDNN primitives we
//!   need: F32 matmul, BF16/S8/F8 brgemm micro-kernel paths, packed-weight
//!   lifetime helpers, post-op fused kernels.
//! - Three `#[no_mangle]` functions (`rayon_parallel_for`,
//!   `rayon_get_num_threads`, `rayon_get_in_parallel`) that the C++ side
//!   calls back into. They are implemented here because oneDNN's
//!   `THREADPOOL` runtime requires a user-provided parallel scheduler, and
//!   we delegate that scheduling to rayon.
//!
//! ## What this crate does NOT expose
//!
//! No safe Rust wrapper — everything here is `unsafe extern "C"` function
//! pointers. Build your own thin safe layer on top (see e.g.
//! `prelude-cpu`'s `onednn::ops` module).
//!
//! ## Environment variables
//!
//! - `ONEDNN_SOURCE_DIR` — path to a oneDNN source checkout. Defaults to
//!   `$CARGO_WORKSPACE/third_party/oneDNN` (prelude workspace case).

use std::ffi::c_void;

unsafe extern "C" {
    pub fn onednn_init();
    pub fn onednn_cleanup();
    pub fn onednn_set_num_threads(num_threads: i32);
    pub fn onednn_bind_threads(cpu_ids: *const i32, num_threads: i32);

    // ── F32 matmul (oneDNN primitive cache) ──────────────────────────

    pub fn onednn_f32_linear(
        input: *const c_void, weight: *const c_void, output: *mut c_void,
        m: i64, k: i64, n: i64,
    );

    pub fn onednn_f32_matmul(
        a: *const c_void, b: *const c_void, c: *mut c_void,
        m: i64, k: i64, n: i64,
    );

    // ── F32 packed weights (oneDNN blocked format) ────────────────────

    pub fn onednn_f32_pack_weights(
        weight: *const c_void, k: i64, n: i64, ref_m: i64,
    ) -> *mut c_void;

    pub fn onednn_f32_linear_packed(
        input: *const c_void, packed_weight: *mut c_void,
        output: *mut c_void, m: i64,
    );

    pub fn onednn_packed_weights_destroy(pw: *mut c_void);
    // ── BRGeMM micro-kernel ──────────────────────────────────────────

    pub fn brgemm_available() -> i32;

    pub fn brgemm_bf16_pack(
        weight: *const c_void, k: i64, n: i64,
    ) -> *mut c_void;

    pub fn brgemm_bf16_pack_destroy(pw: *mut c_void);

    pub fn brgemm_bf16_linear(
        input: *const c_void, pw: *mut c_void, output: *mut c_void,
        m: i64, n_total: i64, n_start: i64, n_end: i64,
    );

    pub fn brgemm_bf16_linear_fused_silu_mul(
        input: *const c_void, pw: *mut c_void, output: *mut c_void,
        m: i64, dim: i64, n_start: i64, n_end: i64,
    );

    /// Release AMX/brgemm HW context after attention N-block loop.
    pub fn brgemm_attn_release();

    /// QK^T GEMM for attention: scores = Q @ K^T * sm_scale
    pub fn brgemm_qk_gemm(
        q_bf16: *const u16,
        k_bf16: *const u16,
        scores_f32: *mut f32,
        m: i64, n: i64, head_dim: i64,
        q_stride: i64, k_stride: i64, ldc: i64, sm_scale: f32,
    );

    /// Score @ V accumulation for attention: C_f32 += scores_bf16 @ V_vnni
    pub fn brgemm_score_v_accum(
        scores_f32: *const f32,
        v_bf16: *const u16,
        c_f32: *mut f32,
        m: i64, k: i64, n: i64, lda: i64, v_stride: i64,
    );

    // ── INT8 BRGeMM ──────────────────────────────────────────────────

    pub fn brgemm_s8_available() -> i32;

    pub fn brgemm_s8_pack(
        weight: *const i8, scales: *const f32, k: i64, n: i64,
    ) -> *mut c_void;

    pub fn brgemm_s8_pack_destroy(pw: *mut c_void);

    pub fn brgemm_quantize_bf16_s8(
        input_bf16: *const c_void, out_s8: *mut i8, m: i64, k: i64,
    ) -> f32;

    pub fn brgemm_s8_linear(
        input_s8: *const i8, a_scale: f32,
        pw: *mut c_void, output_bf16: *mut c_void,
        m: i64, n_total: i64, n_start: i64, n_end: i64,
    );

    // ── FP8 BRGeMM ──────────────────────────────────────────────────

    pub fn brgemm_f8_available() -> i32;

    pub fn brgemm_f8e4m3_pack(
        weight_f8: *const c_void, scales: *const f32, k: i64, n: i64,
    ) -> *mut c_void;

    pub fn brgemm_f8_pack_destroy(pw: *mut c_void);

    pub fn brgemm_quantize_bf16_f8e4m3(
        input_bf16: *const c_void, out_f8: *mut c_void, m: i64, k: i64,
    ) -> f32;

    pub fn brgemm_f8e4m3_linear(
        input_f8: *const c_void, a_scale: f32,
        pw: *mut c_void, output_bf16: *mut c_void,
        m: i64, n_total: i64, n_start: i64, n_end: i64,
    );

    // ── BRGeMM post-ops ─────────────────────────────────────────────

    pub fn brgemm_bf16_linear_postops(
        input: *const c_void, pw: *mut c_void,
        output: *mut c_void, bias_bf16: *const c_void,
        postop_flags: i32,
        m: i64, n_total: i64, n_start: i64, n_end: i64,
    );
}

// ── Rayon callback exports ──────────────────────────────────────────────
// Called by oneDNN's RayonThreadPool adapter (C++ side) to dispatch
// parallel work to Rust's rayon thread pool.

/// C-compatible callback type: fn(chunk_id, total_chunks, context)
type ParallelForBody = unsafe extern "C" fn(i32, i32, *mut c_void);

/// Dispatch `n` parallel tasks to rayon. Called from C++ RayonThreadPool::parallel_for().
#[unsafe(no_mangle)]
pub extern "C" fn rayon_parallel_for(n: i32, body: ParallelForBody, context: *mut c_void) {
    use rayon::prelude::*;
    let ctx = context as usize; // usize is Send
    (0..n).into_par_iter().for_each(|i| {
        unsafe { body(i, n, ctx as *mut c_void) };
    });
}

/// Return the number of rayon worker threads.
#[unsafe(no_mangle)]
pub extern "C" fn rayon_get_num_threads() -> i32 {
    rayon::current_num_threads() as i32
}

/// Return 1 if the calling thread is inside a rayon parallel region, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn rayon_get_in_parallel() -> i32 {
    if rayon::current_thread_index().is_some() { 1 } else { 0 }
}
