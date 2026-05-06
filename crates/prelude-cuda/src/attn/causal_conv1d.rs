//! Thin Rust wrapper around `causal_conv1d::causal_conv1d_fwd` /
//! `causal_conv1d_update` — the Dao-AILab mamba depthwise-causal-conv1d
//! kernels. Used by Qwen3.5 / Qwen3-next DeltaNet's `conv1d_prefill` /
//! `conv1d_decode` paths.
//!
//! The heavy lifting (kernel templates + .cu compile) is in the
//! `causal-conv1d` crate under `crates/prelude-cuda/causal-conv1d/`;
//! this module bridges candle tensors to the raw-pointer C ABI.

use candle_core::backend::BackendStorage;
use causal_conv1d::{self as ffi, Dtype};
use cudarc::driver::DevicePtr;
use half::{bf16, f16};
use prelude_core::tensor::{DType, Result, Tensor, bail};
use std::ffi::c_void;

/// Map a candle DType to the causal-conv1d shim's tag.
fn dtype_tag(dt: DType) -> Option<Dtype> {
    match dt {
        DType::BF16 => Some(Dtype::BF16),
        DType::F16 => Some(Dtype::F16),
        DType::F32 => Some(Dtype::F32),
        _ => None,
    }
}

/// Fused prefill conv1d.
///
/// - `x`: `[B, D, L]` (channel-before-time, matches Dao-AILab
///   convention — caller transposes if needed)
/// - `weight`: `[D, W]`, same dtype class as `x`
/// - `bias`: optional `[D]`
/// - `initial_states`: optional `[B, D, W-1]` left-context from a prior
///   chunk
/// - `silu_activation`: fuse a SiLU tail if true
///
/// Returns `[B, D, L]` same dtype as `x`. Caller saves the last `W-1`
/// columns of the raw input `x` as the new conv_state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    initial_states: Option<&Tensor>,
    silu_activation: bool,
) -> Result<Option<Tensor>> {
    if !x.device().is_cuda() {
        return Ok(None);
    }

    let (batch, dim, seqlen) = x.dims3()?;
    let (wd, width) = weight.dims2()?;
    if wd != dim {
        bail!("causal_conv1d_fwd: weight dim {wd} != x dim {dim}");
    }
    // Upstream specializes on widths 2..=4. Anything else falls back.
    if !(2..=4).contains(&width) {
        return Ok(None);
    }

    let Some(input_dtype) = dtype_tag(x.dtype()) else {
        return Ok(None);
    };
    let Some(weight_dtype) = dtype_tag(weight.dtype()) else {
        return Ok(None);
    };

    if let Some(init) = initial_states {
        let init_dims = init.dims();
        if init_dims != [batch, dim, width - 1] {
            bail!(
                "causal_conv1d_fwd: initial_states shape {:?} != [{batch}, {dim}, {}]",
                init_dims,
                width - 1
            );
        }
        if init.dtype() != x.dtype() {
            bail!(
                "causal_conv1d_fwd: initial_states dtype {:?} != x dtype {:?}",
                init.dtype(),
                x.dtype()
            );
        }
    }

    let x_c = x.contiguous()?;
    let weight_c = weight.contiguous()?;
    let bias_c = match bias {
        Some(b) => Some(b.contiguous()?),
        None => None,
    };
    let init_c = match initial_states {
        Some(s) => Some(s.contiguous()?),
        None => None,
    };
    let out = Tensor::zeros((batch, dim, seqlen), x.dtype(), x.device())?;

    // Pull device + stream from the output storage.
    let (out_storage, _) = out.storage_and_layout();
    let cuda_dev = match &*out_storage {
        candle_core::Storage::Cuda(s) => s.device().clone(),
        _ => bail!("causal_conv1d_fwd: output not on CUDA"),
    };
    drop(out_storage);
    let stream = cuda_dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *const c_void;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("causal_conv1d_fwd: tensor not on CUDA"),
            };
            let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64 as *mut c_void
        }};
    }

    // Extract pointers. Dispatch by dtype to pick the right cudarc slice.
    // The `cuda_ptr!` macro uses `?` which must propagate to this outer
    // fn's Result return — so we can't put it inside a closure like
    // `.map(|b| cuda_ptr!(...))`. Hand-expand each Option.
    let (x_ptr, w_ptr, out_ptr, bias_ptr, init_ptr) = match x.dtype() {
        DType::BF16 => {
            let xp = cuda_ptr!(x_c, bf16);
            let wp = cuda_ptr!(weight_c, bf16);
            let op = cuda_ptr!(out, bf16);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, bf16)),
                None => None,
            };
            let ip = match init_c.as_ref() {
                Some(i) => Some(cuda_ptr!(i, bf16)),
                None => None,
            };
            (xp, wp, op, bp, ip)
        }
        DType::F16 => {
            let xp = cuda_ptr!(x_c, f16);
            let wp = cuda_ptr!(weight_c, f16);
            let op = cuda_ptr!(out, f16);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, f16)),
                None => None,
            };
            let ip = match init_c.as_ref() {
                Some(i) => Some(cuda_ptr!(i, f16)),
                None => None,
            };
            (xp, wp, op, bp, ip)
        }
        DType::F32 => {
            let xp = cuda_ptr!(x_c, f32);
            let wp = cuda_ptr!(weight_c, f32);
            let op = cuda_ptr!(out, f32);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, f32)),
                None => None,
            };
            let ip = match init_c.as_ref() {
                Some(i) => Some(cuda_ptr!(i, f32)),
                None => None,
            };
            (xp, wp, op, bp, ip)
        }
        _ => unreachable!("dtype_tag guarded above"),
    };

    unsafe {
        ffi::causal_conv1d_fwd(
            stream_ptr,
            x_ptr as *const c_void,
            w_ptr as *const c_void,
            bias_ptr.map(|p| p as *const c_void),
            init_ptr.map(|p| p as *const c_void),
            /*final_states=*/ None,
            out_ptr,
            batch as i32,
            dim as i32,
            seqlen as i32,
            width as i32,
            silu_activation,
            input_dtype,
            weight_dtype,
        )
        .map_err(candle_core::Error::msg)?;
    }

    // Keep source tensors alive until the kernel launch has captured pointers.
    drop(x_c);
    drop(weight_c);
    drop(bias_c);
    drop(init_c);

    Ok(Some(out))
}

/// Fused prefill conv1d for channel-last `[B, L, D]` input/output.
pub(crate) fn try_fwd_channellast(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    initial_states: Option<&Tensor>,
    silu_activation: bool,
) -> Result<Option<Tensor>> {
    if !x.device().is_cuda() {
        return Ok(None);
    }

    let (batch, seqlen, dim) = x.dims3()?;
    let (wd, width) = weight.dims2()?;
    if wd != dim {
        bail!("causal_conv1d_fwd_channellast: weight dim {wd} != x dim {dim}");
    }
    if !(2..=4).contains(&width) {
        return Ok(None);
    }

    let Some(input_dtype) = dtype_tag(x.dtype()) else {
        return Ok(None);
    };
    let Some(weight_dtype) = dtype_tag(weight.dtype()) else {
        return Ok(None);
    };

    if let Some(init) = initial_states {
        let init_dims = init.dims();
        if init_dims != [batch, width - 1, dim] {
            bail!(
                "causal_conv1d_fwd_channellast: initial_states shape {:?} != [{batch}, {}, {dim}]",
                init_dims,
                width - 1
            );
        }
        if init.dtype() != x.dtype() {
            bail!(
                "causal_conv1d_fwd_channellast: initial_states dtype {:?} != x dtype {:?}",
                init.dtype(),
                x.dtype()
            );
        }
    }

    let x_c = x.contiguous()?;
    let weight_c = weight.contiguous()?;
    let bias_c = match bias {
        Some(b) => Some(b.contiguous()?),
        None => None,
    };
    let init_c = match initial_states {
        Some(s) => Some(s.contiguous()?),
        None => None,
    };
    let out = Tensor::zeros((batch, seqlen, dim), x.dtype(), x.device())?;

    let (out_storage, _) = out.storage_and_layout();
    let cuda_dev = match &*out_storage {
        candle_core::Storage::Cuda(s) => s.device().clone(),
        _ => bail!("causal_conv1d_fwd_channellast: output not on CUDA"),
    };
    drop(out_storage);
    let stream = cuda_dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *const c_void;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("causal_conv1d_fwd_channellast: tensor not on CUDA"),
            };
            let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64 as *mut c_void
        }};
    }

    let (x_ptr, w_ptr, out_ptr, bias_ptr, init_ptr) = match x.dtype() {
        DType::BF16 => {
            let xp = cuda_ptr!(x_c, bf16);
            let wp = cuda_ptr!(weight_c, bf16);
            let op = cuda_ptr!(out, bf16);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, bf16)),
                None => None,
            };
            let ip = match init_c.as_ref() {
                Some(i) => Some(cuda_ptr!(i, bf16)),
                None => None,
            };
            (xp, wp, op, bp, ip)
        }
        DType::F16 => {
            let xp = cuda_ptr!(x_c, f16);
            let wp = cuda_ptr!(weight_c, f16);
            let op = cuda_ptr!(out, f16);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, f16)),
                None => None,
            };
            let ip = match init_c.as_ref() {
                Some(i) => Some(cuda_ptr!(i, f16)),
                None => None,
            };
            (xp, wp, op, bp, ip)
        }
        DType::F32 => {
            let xp = cuda_ptr!(x_c, f32);
            let wp = cuda_ptr!(weight_c, f32);
            let op = cuda_ptr!(out, f32);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, f32)),
                None => None,
            };
            let ip = match init_c.as_ref() {
                Some(i) => Some(cuda_ptr!(i, f32)),
                None => None,
            };
            (xp, wp, op, bp, ip)
        }
        _ => unreachable!("dtype_tag guarded above"),
    };

    unsafe {
        ffi::causal_conv1d_fwd_channellast(
            stream_ptr,
            x_ptr as *const c_void,
            w_ptr as *const c_void,
            bias_ptr.map(|p| p as *const c_void),
            init_ptr.map(|p| p as *const c_void),
            /*final_states=*/ None,
            out_ptr,
            batch as i32,
            dim as i32,
            seqlen as i32,
            width as i32,
            silu_activation,
            input_dtype,
            weight_dtype,
        )
        .map_err(candle_core::Error::msg)?;
    }

    drop(x_c);
    drop(weight_c);
    drop(bias_c);
    drop(init_c);

    Ok(Some(out))
}

/// Single-token decode step. `conv_state` is updated in place.
///
/// When `conv_state_indices` is `Some(&indices)`, `conv_state` is a pool
/// tensor `[pool_size, dim, state_len]` and `indices` is `[batch]` I32
/// mapping each batch element to its pool slot. The kernel indexes into
/// the pool using these indices instead of assuming `conv_state` batch
/// dim matches `x` batch dim.
pub(crate) fn try_update(
    x: &Tensor,
    conv_state: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    silu_activation: bool,
    conv_state_indices: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    if !x.device().is_cuda() {
        return Ok(None);
    }

    let (batch, dim) = x.dims2()?;
    let (wd, width) = weight.dims2()?;
    if wd != dim {
        bail!("causal_conv1d_update: weight dim {wd} != x dim {dim}");
    }
    if !(2..=4).contains(&width) {
        return Ok(None);
    }

    let (sb, sd, sl) = conv_state.dims3()?;
    if conv_state_indices.is_none() && (sb != batch || sd != dim) {
        bail!(
            "causal_conv1d_update: conv_state shape {:?} doesn't match ({batch}, {dim}, *)",
            (sb, sd, sl)
        );
    }
    if conv_state_indices.is_some() && sd != dim {
        bail!(
            "causal_conv1d_update: conv_state dim {:?} doesn't match x dim {dim}",
            sd
        );
    }

    let Some(input_dtype) = dtype_tag(x.dtype()) else {
        return Ok(None);
    };
    let Some(weight_dtype) = dtype_tag(weight.dtype()) else {
        return Ok(None);
    };
    if conv_state.dtype() != x.dtype() {
        bail!(
            "causal_conv1d_update: conv_state dtype {:?} != x dtype {:?}",
            conv_state.dtype(),
            x.dtype()
        );
    }

    // Reshape x from [B, D] to [B, D, 1] without a copy if possible.
    let x_3d = x.unsqueeze(2)?.contiguous()?; // [B, D, 1]
    let weight_c = weight.contiguous()?;
    let bias_c = match bias {
        Some(b) => Some(b.contiguous()?),
        None => None,
    };
    // conv_state is mutated in-place. It must be contiguous; if the
    // caller passes a view, force-copy.
    let state_c = conv_state.contiguous()?;

    let out = Tensor::zeros((batch, dim, 1), x.dtype(), x.device())?;

    let (out_storage, _) = out.storage_and_layout();
    let cuda_dev = match &*out_storage {
        candle_core::Storage::Cuda(s) => s.device().clone(),
        _ => bail!("causal_conv1d_update: output not on CUDA"),
    };
    drop(out_storage);
    let stream = cuda_dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *const c_void;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("causal_conv1d_update: tensor not on CUDA"),
            };
            let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64 as *mut c_void
        }};
    }

    let (x_ptr, w_ptr, out_ptr, state_ptr, bias_ptr) = match x.dtype() {
        DType::BF16 => {
            let xp = cuda_ptr!(x_3d, bf16);
            let wp = cuda_ptr!(weight_c, bf16);
            let op = cuda_ptr!(out, bf16);
            let sp = cuda_ptr!(state_c, bf16);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, bf16)),
                None => None,
            };
            (xp, wp, op, sp, bp)
        }
        DType::F16 => {
            let xp = cuda_ptr!(x_3d, f16);
            let wp = cuda_ptr!(weight_c, f16);
            let op = cuda_ptr!(out, f16);
            let sp = cuda_ptr!(state_c, f16);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, f16)),
                None => None,
            };
            (xp, wp, op, sp, bp)
        }
        DType::F32 => {
            let xp = cuda_ptr!(x_3d, f32);
            let wp = cuda_ptr!(weight_c, f32);
            let op = cuda_ptr!(out, f32);
            let sp = cuda_ptr!(state_c, f32);
            let bp = match bias_c.as_ref() {
                Some(b) => Some(cuda_ptr!(b, f32)),
                None => None,
            };
            (xp, wp, op, sp, bp)
        }
        _ => unreachable!("dtype_tag guarded above"),
    };

    // Extract conv_state_indices GPU pointer if provided.
    // Accepts both I32 and U32 tensors (same binary representation for slot values).
    let indices_ptr: Option<*const i32> = match conv_state_indices {
        Some(idx_tensor) => {
            let idx_c = idx_tensor.contiguous()?;
            let (idx_storage, idx_layout) = idx_c.storage_and_layout();
            let idx_cuda = match &*idx_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("causal_conv1d_update: conv_state_indices not on CUDA"),
            };
            let ptr = match idx_tensor.dtype() {
                DType::U32 => {
                    let idx_slice = idx_cuda
                        .as_cuda_slice::<u32>()?
                        .slice(idx_layout.start_offset()..);
                    let (p, _guard) = idx_slice.device_ptr(&stream);
                    p as u64 as *const i32 // safe: u32 and i32 same size, values < 2^31
                }
                DType::I64 => {
                    bail!("causal_conv1d_update: conv_state_indices must be U32 or I32, got I64");
                }
                _ => {
                    bail!(
                        "causal_conv1d_update: conv_state_indices must be U32, got {:?}",
                        idx_tensor.dtype()
                    );
                }
            };
            Some(ptr)
        }
        None => None,
    };

    unsafe {
        ffi::causal_conv1d_update(
            stream_ptr,
            x_ptr as *const c_void,
            state_ptr,
            w_ptr as *const c_void,
            bias_ptr.map(|p| p as *const c_void),
            out_ptr,
            indices_ptr,
            batch as i32,
            dim as i32,
            width as i32,
            sl as i32,
            silu_activation,
            input_dtype,
            weight_dtype,
        )
        .map_err(candle_core::Error::msg)?;
    }

    drop(x_3d);
    drop(weight_c);
    drop(bias_c);

    // Copy the updated state back into the caller's original tensor if
    // we had to realize a contiguous copy. Tensor::slice_set would do
    // this, but simpler: just have the caller always pass a
    // contiguous state — which our Qwen3.5 DeltaNet already does via
    // `.contiguous()?` after pool loads.
    //
    // If we did copy (contiguous() returned a fresh tensor), the
    // caller's original view is now stale. Document: callers should
    // keep `state_c` as the canonical state. We return the output +
    // expect the caller to either (a) pass a contiguous state and
    // reuse it directly or (b) swap their state handle to `state_c`.
    //
    // Qwen3.5's deltanet_varlen_pooled always loads conv_state from
    // the pool and writes it back via `slice_set`, so this is fine —
    // we just need to hand `state_c` back to them.
    // … but the trait signature returns only the output. For now,
    // mutation happens through the raw pointer, which works when the
    // input `conv_state` is already contiguous. Document the precondition.
    drop(state_c);

    // Output is [B, D, 1] — squeeze the trailing 1 for the caller.
    out.squeeze(2).map(Some).map_err(candle_core::Error::from)
}
