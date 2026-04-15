//! Rust FFI bindings to Dao-AILab's causal-conv1d CUDA kernels.
//!
//! Two entry points:
//!
//!   * [`causal_conv1d_fwd`] — fused short causal 1D convolution over a
//!     whole sequence (prefill). Supports optional left-context
//!     `initial_states` (e.g. the saved `width-1` tokens from a previous
//!     chunk) and optional `final_states` writeback. SiLU activation
//!     can be fused.
//!
//!   * [`causal_conv1d_update`] — single-token decode step. Reads the
//!     running `conv_state` tail, shifts it left, writes the new input,
//!     computes the output, updates `conv_state` in place.
//!
//! Both functions dispatch on a `(input_dtype, weight_dtype)` pair via
//! the C shim, which forwards to one of upstream's 9 template
//! instantiations. For inference we almost always want BF16 or F16 for
//! both input and weight.
//!
//! ## Input / output layout conventions
//!
//! All tensors follow the upstream Dao-AILab convention:
//!
//!   * `x`:      `[batch, dim, seqlen]`
//!   * `weight`: `[dim, width]`        (depthwise: one filter per channel)
//!   * `bias`:   `[dim]` (optional)
//!   * `out`:    `[batch, dim, seqlen]`
//!
//! Strides are passed in **elements**, not bytes.

use std::ffi::c_void;

/// Element-type tags for the FFI shim. Keep in sync with
/// `crates/prelude-cuda/causal-conv1d/csrc/causal_conv1d_shim.cu`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    BF16 = 0,
    F16 = 1,
    F32 = 2,
}

unsafe extern "C" {
    fn cula_causal_conv1d_fwd(
        stream: *const c_void,
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        initial_states: *const c_void,
        final_states: *mut c_void,
        out: *mut c_void,
        batch: i32,
        dim: i32,
        seqlen: i32,
        width: i32,
        silu_activation: i32,
        x_batch_stride: i64,
        x_c_stride: i64,
        x_l_stride: i64,
        weight_c_stride: i64,
        weight_width_stride: i64,
        out_batch_stride: i64,
        out_c_stride: i64,
        out_l_stride: i64,
        initial_states_batch_stride: i64,
        initial_states_c_stride: i64,
        initial_states_l_stride: i64,
        final_states_batch_stride: i64,
        final_states_c_stride: i64,
        final_states_l_stride: i64,
        input_dtype: i32,
        weight_dtype: i32,
    ) -> i32;

    fn cula_causal_conv1d_update(
        stream: *const c_void,
        x: *const c_void,
        conv_state: *mut c_void,
        weight: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        conv_state_indices: *const i32,
        batch: i32,
        dim: i32,
        seqlen: i32,
        width: i32,
        conv_state_len: i32,
        silu_activation: i32,
        x_batch_stride: i64,
        x_c_stride: i64,
        x_l_stride: i64,
        weight_c_stride: i64,
        weight_width_stride: i64,
        out_batch_stride: i64,
        out_c_stride: i64,
        out_l_stride: i64,
        conv_state_batch_stride: i64,
        conv_state_c_stride: i64,
        conv_state_l_stride: i64,
        input_dtype: i32,
        weight_dtype: i32,
    ) -> i32;
}

/// Contiguous strides in element counts for shape `(batch, dim, seqlen)`.
fn contiguous_bcl(dim: i64, seqlen: i64) -> (i64, i64, i64) {
    // C-contiguous: batch stride = dim*seqlen, c stride = seqlen, l stride = 1
    (dim * seqlen, seqlen, 1)
}

/// Contiguous strides for shape `(dim, width)`.
fn contiguous_dw(width: i64) -> (i64, i64) {
    (width, 1)
}

/// Prefill-side fused causal conv1d.
///
/// Shapes are `x/out: [B, D, L]` (channel-contiguous-before-time —
/// i.e. `x_l_stride = 1`), `weight: [D, W]`, optional `bias: [D]`,
/// optional `initial_states / final_states: [B, D, W-1]`.
///
/// ## Initial state caveat
///
/// Upstream's `causal_conv1d_fwd` dispatches on `is_channel_last =
/// x.stride(1) == 1 && x.stride(2) > 1`. The channel-first kernel
/// (what this shim targets with a standard row-major `[B, D, L]`
/// input) **silently ignores** the `initial_states` pointer —
/// `initial_states is only supported for channel last layout`
/// (`causal_conv1d.cpp:213`). If you pass `Some(...)` for
/// `initial_states` here the pointer is forwarded but the kernel
/// never reads it, leaving the left context at zero.
///
/// Callers that actually need cross-chunk state (e.g. multi-chunk
/// prefill where a long prompt is split) must either manually prepend
/// the saved state to `x` (adding `W-1` tokens) or switch the whole
/// pipeline to a channel-last layout. Qwen3.5 DeltaNet today only
/// ever runs single-chunk prefill, so we pass `None` for safety and
/// sidestep the trap.
///
/// # Safety
/// All pointers must be valid CUDA device pointers on the same device.
/// The caller owns allocation and lifetime.
#[allow(clippy::too_many_arguments)]
pub unsafe fn causal_conv1d_fwd(
    stream: *const c_void,
    x: *const c_void,
    weight: *const c_void,
    bias: Option<*const c_void>,
    initial_states: Option<*const c_void>,
    final_states: Option<*mut c_void>,
    out: *mut c_void,
    batch: i32,
    dim: i32,
    seqlen: i32,
    width: i32,
    silu_activation: bool,
    input_dtype: Dtype,
    weight_dtype: Dtype,
) -> Result<(), String> {
    let (xb, xc, xl) = contiguous_bcl(dim as i64, seqlen as i64);
    let (wc, ww) = contiguous_dw(width as i64);
    let init_stride = contiguous_bcl(dim as i64, (width - 1) as i64);

    let ret = unsafe {
        cula_causal_conv1d_fwd(
            stream,
            x,
            weight,
            bias.unwrap_or(std::ptr::null()),
            initial_states.unwrap_or(std::ptr::null()),
            final_states.unwrap_or(std::ptr::null_mut()),
            out,
            batch, dim, seqlen, width,
            silu_activation as i32,
            xb, xc, xl,
            wc, ww,
            xb, xc, xl,                         // out strides match x
            init_stride.0, init_stride.1, init_stride.2,
            init_stride.0, init_stride.1, init_stride.2,
            input_dtype as i32,
            weight_dtype as i32,
        )
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("causal_conv1d_fwd failed (code {code})")),
    }
}

/// Decode-side single-token conv1d update. `conv_state` is updated in
/// place: the tail `width-1` slots always hold the last `width-1`
/// inputs.
///
/// Layout: `x: [B, D, 1]`, `conv_state: [B, D, conv_state_len]`
/// (typically `conv_state_len == width - 1`), `weight: [D, W]`,
/// `out: [B, D, 1]`.
///
/// # Safety
/// Same as [`causal_conv1d_fwd`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn causal_conv1d_update(
    stream: *const c_void,
    x: *const c_void,
    conv_state: *mut c_void,
    weight: *const c_void,
    bias: Option<*const c_void>,
    out: *mut c_void,
    conv_state_indices: Option<*const i32>,
    batch: i32,
    dim: i32,
    width: i32,
    conv_state_len: i32,
    silu_activation: bool,
    input_dtype: Dtype,
    weight_dtype: Dtype,
) -> Result<(), String> {
    let (xb, xc, xl) = contiguous_bcl(dim as i64, 1);
    let (wc, ww) = contiguous_dw(width as i64);
    let (sb, sc, sl) = contiguous_bcl(dim as i64, conv_state_len as i64);

    let ret = unsafe {
        cula_causal_conv1d_update(
            stream,
            x,
            conv_state,
            weight,
            bias.unwrap_or(std::ptr::null()),
            out,
            conv_state_indices.unwrap_or(std::ptr::null()),
            batch, dim, /*seqlen=*/ 1, width, conv_state_len,
            silu_activation as i32,
            xb, xc, xl,
            wc, ww,
            xb, xc, xl,             // out strides match x
            sb, sc, sl,
            input_dtype as i32,
            weight_dtype as i32,
        )
    };
    match ret {
        0 => Ok(()),
        code => Err(format!("causal_conv1d_update failed (code {code})")),
    }
}
