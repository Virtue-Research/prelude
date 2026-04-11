//! Rust bindings to cuLA linear-attention CUDA kernels. Statically linked,
//! no libtorch / libpython runtime dependency.
//!
//! Two kernel paths, gated by feature flags:
//!
//! * **C++ CUTLASS kernels** (always enabled, compiled directly via nvcc):
//!   - SM90: KDA fused prefill (varlen, bf16)
//!   - SM100: KDA chunked intra-attention + recompute W/U
//!
//! * **CuTe DSL kernels** (feature = `dsl`, on by default):
//!   - Lightning Attention prefill (chunkwise decay)
//!   - Lightning Attention decode (single token)
//!   - Chunk delta-H (inter-chunk state update)
//!   - Forward output computation
//!   - KDA decode (single token, varlen / dense, up to 4-way GQA)
//!
//!   These are AOT-compiled into `.o` files via Python + `cute.compile(...)`
//!   and called through the `TVMSafeCallFn` convention, which is why the
//!   `dsl` feature pulls in `tvm-static-ffi`. Disable the feature to skip the
//!   Python build step and drop the tvm-static-ffi dependency.

#[cfg(feature = "dsl")]
pub mod dsl;

use std::ffi::c_void;

// ── C++ CUTLASS kernel FFI ───────────────────────────────────────────

unsafe extern "C" {
    fn cula_kda_fwd_prefill_sm90(
        stream: *const c_void,
        output: *mut c_void,
        output_state: *mut f32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        input_state: *const f32,
        alpha: *const f32,
        beta: *const f32,
        cu_seqlens: *const i32,
        workspace: *mut u8,
        num_seqs: i32,
        num_heads: i32,
        head_size: i32,
        total_seqlen: i64,
        scale: f32,
        safe_gate: i32,
        sm_count: i32,
    ) -> i32;

    fn cula_chunk_kda_fwd_intra_sm100(
        stream: *const c_void,
        q: *const c_void,
        k: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        aqk_out: *mut c_void,
        akk_out: *mut c_void,
        tile_counter: *mut i32,
        scale: f32,
        chunk_size: i32,
        total_q_len: i32,
        b: i32,
        h: i32,
        d: i32,
        num_tiles: i32,
        use_tf32_inverse: i32,
        unified_gref: i32,
        num_sm: i32,
    ) -> i32;

    fn cula_chunk_kda_fwd_recomp_wu_sm100(
        stream: *const c_void,
        k: *const c_void,
        v: *const c_void,
        beta: *const c_void,
        a: *const c_void,
        g: *const c_void,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        w_out: *mut c_void,
        u_out: *mut c_void,
        kg_out: *mut c_void,
        chunk_size: i32,
        total_len: i32,
        b: i32,
        h: i32,
        d: i32,
        num_tiles: i32,
        num_sm: i32,
    ) -> i32;
}

/// KDA fused prefill on SM90 (Hopper).
///
/// Q/K/V: bf16 `[packed_seq, H, D]`, cu_seqlens: i32 `[num_seqs+1]`.
/// Output: bf16 `[packed_seq, H, D]`, output_state: f32 `[num_seqs, H, D, D]`.
///
/// # Safety
/// All pointers must be valid CUDA device pointers on the same device.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_fwd_prefill_sm90(
    stream: *const c_void,
    output: *mut c_void,
    output_state: *mut f32,
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    input_state: Option<*const f32>,
    alpha: Option<*const f32>,
    beta: Option<*const f32>,
    cu_seqlens: *const i32,
    workspace: *mut u8,
    num_seqs: i32,
    num_heads: i32,
    head_size: i32,
    total_seqlen: i64,
    scale: f32,
    safe_gate: bool,
    sm_count: i32,
) -> Result<(), String> {
    let ret = unsafe {
        cula_kda_fwd_prefill_sm90(
            stream,
            output,
            output_state,
            q, k, v,
            input_state.unwrap_or(std::ptr::null()),
            alpha.unwrap_or(std::ptr::null()),
            beta.unwrap_or(std::ptr::null()),
            cu_seqlens,
            workspace,
            num_seqs, num_heads, head_size, total_seqlen,
            scale, safe_gate as i32, sm_count,
        )
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("cuLA KDA prefill SM90 failed (code {code})")),
    }
}

/// KDA chunked intra-attention on SM100 (Blackwell).
///
/// # Safety
/// All pointers must be valid CUDA device pointers on the same device.
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_kda_fwd_intra_sm100(
    stream: *const c_void,
    q: *const c_void,
    k: *const c_void,
    g: *const c_void,
    beta: *const c_void,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    aqk_out: *mut c_void,
    akk_out: *mut c_void,
    tile_counter: *mut i32,
    scale: f32,
    chunk_size: i32,
    total_q_len: i32,
    b: i32,
    h: i32,
    d: i32,
    num_tiles: i32,
    use_tf32_inverse: bool,
    unified_gref: bool,
    num_sm: i32,
) -> Result<(), String> {
    let ret = unsafe {
        cula_chunk_kda_fwd_intra_sm100(
            stream, q, k, g, beta,
            cu_seqlens, chunk_indices,
            aqk_out, akk_out, tile_counter,
            scale, chunk_size, total_q_len,
            b, h, d, num_tiles,
            use_tf32_inverse as i32, unified_gref as i32, num_sm,
        )
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("cuLA KDA intra SM100 failed (code {code})")),
    }
}

/// KDA recompute W & U on SM100 (Blackwell).
///
/// # Safety
/// All pointers must be valid CUDA device pointers on the same device.
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_kda_fwd_recomp_wu_sm100(
    stream: *const c_void,
    k: *const c_void,
    v: *const c_void,
    beta: *const c_void,
    a: *const c_void,
    g: *const c_void,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    w_out: *mut c_void,
    u_out: *mut c_void,
    kg_out: *mut c_void,
    chunk_size: i32,
    total_len: i32,
    b: i32,
    h: i32,
    d: i32,
    num_tiles: i32,
    num_sm: i32,
) -> Result<(), String> {
    let ret = unsafe {
        cula_chunk_kda_fwd_recomp_wu_sm100(
            stream, k, v, beta, a, g,
            cu_seqlens, chunk_indices,
            w_out, u_out, kg_out,
            chunk_size, total_len,
            b, h, d, num_tiles, num_sm,
        )
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("cuLA KDA recomp W/U SM100 failed (code {code})")),
    }
}
