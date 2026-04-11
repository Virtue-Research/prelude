//! cuLA KDA fused prefill — one CUTLASS warp-specialized TMA kernel call
//! per DeltaNet layer for an entire packed prefill sequence.
//!
//! Wraps the SM90 `cula::kda_fwd_prefill_sm90` C-shim. The upstream kernel
//! supports both plain MHA and KDA's "multi-value" GQA flavor (Q/K share
//! `num_k_heads`, V and the recurrent state have the larger `num_heads`).
//!
//! NOTE: upstream `inclusionAI/cuLA` currently only supports plain MHA
//! (`num_q_heads == num_k_heads == num_v_heads`). Multi-value GQA lives on
//! the `feat/sm90-gqa` fork PR and is not part of the shim this crate
//! links against — we return `Ok(None)` on GVA shapes so callers fall
//! back. Qwen3.5-A3B is GVA and therefore never hits this kernel.
//!
//! Gate preprocessing (softplus + A_log + chunk-cumsum + RCP_LN2 scale) is
//! done by the caller in Rust via candle ops — the kernel takes the already
//! chunk-cumsummed `alpha` tensor.

use candle_core::backend::BackendStorage;
use cudarc::driver::DevicePtr;
use half::bf16;
use prelude_core::tensor::{bail, DType, Result, Tensor};
use std::ffi::c_void;
use std::sync::OnceLock;

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

const CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;

fn detect_sm_major() -> i32 {
    unsafe {
        let mut dev = 0i32;
        if cudaGetDevice(&mut dev) != 0 { return 0; }
        let mut major = 0i32;
        if cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR, dev) != 0 {
            return 0;
        }
        major
    }
}

fn detect_sm_count() -> i32 {
    static CACHE: OnceLock<i32> = OnceLock::new();
    *CACHE.get_or_init(|| unsafe {
        let mut dev = 0i32;
        if cudaGetDevice(&mut dev) != 0 { return 132; }
        let mut count = 0i32;
        if cudaDeviceGetAttribute(&mut count, CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT, dev) != 0 {
            return 132;
        }
        if count > 0 { count } else { 132 }
    })
}

/// Try to run the fused cuLA KDA prefill kernel over a packed varlen batch.
///
/// Returns `Ok(Some((o, final_state)))` on success, `Ok(None)` if the GPU
/// arch or shapes aren't supported (caller falls back), `Err` only for
/// actual kernel / tensor errors.
///
/// Contract (`T = total_seqlen`, may pack multiple requests):
/// - `q`, `k`: `[T, HK, D]` BF16 (already L2-normalized if the model needs it)
/// - `v`: `[T, HV, D]` BF16
/// - `alpha`: `[T, HV, D]` F32, already `chunk_cumsum(gate) * RCP_LN2`
/// - `beta`: `[T, HV]` F32, already sigmoid-applied
/// - `cu_seqlens`: `[num_seqs+1]` I32 on the same CUDA device
/// - `initial_state`: optional `[num_seqs, HV, D, D]` F32
/// - `scale`: scalar (typically `1 / sqrt(D)`), applied by kernel to Q@K^T
///
/// Output shapes:
/// - `o`: `[T, HV, D]` BF16
/// - `final_state`: `[num_seqs, HV, D, D]` F32
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    cu_seqlens: &Tensor,
    initial_state: Option<&Tensor>,
    scale: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    // Only Hopper has a compiled variant of this kernel.
    if detect_sm_major() != 9 {
        return Ok(None);
    }

    let device = q.device();
    if !device.is_cuda() { return Ok(None); }

    // ── Shape validation ────────────────────────────────────────────
    let (t, hk, d) = q.dims3()?;
    let (tk, hkk, dk) = k.dims3()?;
    let (tv, hv, dv) = v.dims3()?;
    if t != tk || t != tv {
        bail!("kda_prefill: q/k/v seq-dim mismatch: q={t} k={tk} v={tv}");
    }
    if hk != hkk {
        bail!("kda_prefill: q/k head count mismatch: {hk} vs {hkk}");
    }
    if d != dk || d != dv {
        bail!("kda_prefill: q/k/v head_dim mismatch: {d} / {dk} / {dv}");
    }
    // Upstream cuLA (without the sm90-gqa fork PR) only supports MHA.
    // GVA/GQA shapes fall back to the caller's native path.
    if hk != hv {
        return Ok(None);
    }
    // cuLA launcher is specialized on D = 128 tile.
    if d != 128 {
        return Ok(None);
    }

    let (ta, hva, da) = alpha.dims3()?;
    if ta != t || hva != hv || da != d {
        bail!("kda_prefill: alpha shape {:?} mismatches ({t}, {hv}, {d})", (ta, hva, da));
    }
    let (tb, hvb) = beta.dims2()?;
    if tb != t || hvb != hv {
        bail!("kda_prefill: beta shape {:?} mismatches ({t}, {hv})", (tb, hvb));
    }

    if cu_seqlens.dtype() != DType::I32 && cu_seqlens.dtype() != DType::U32 {
        bail!("kda_prefill: cu_seqlens must be I32 / U32");
    }
    let num_seqs = cu_seqlens.dim(0)? - 1;

    // ── Dtype checks ────────────────────────────────────────────────
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        bail!("kda_prefill: q/k/v must be BF16");
    }
    if alpha.dtype() != DType::F32 || beta.dtype() != DType::F32 {
        bail!("kda_prefill: alpha/beta must be F32");
    }
    if let Some(s) = initial_state {
        if s.dtype() != DType::F32 {
            bail!("kda_prefill: initial_state must be F32");
        }
        let dims = s.dims();
        if dims.len() != 4 || dims[0] != num_seqs || dims[1] != hv || dims[2] != d || dims[3] != d {
            bail!(
                "kda_prefill: initial_state shape {:?} mismatches ({num_seqs}, {hv}, {d}, {d})",
                dims
            );
        }
    }

    // ── Ensure all tensors are contiguous in the expected layouts ───
    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;
    let alpha_c = alpha.contiguous()?;
    let beta_c = beta.contiguous()?;
    let cu_c = if cu_seqlens.dtype() == DType::I32 {
        cu_seqlens.contiguous()?
    } else {
        cu_seqlens.to_dtype(DType::I32)?.contiguous()?
    };
    let init_c = match initial_state {
        Some(s) => Some(s.contiguous()?),
        None => None,
    };

    // ── Output buffers ──────────────────────────────────────────────
    let out = Tensor::zeros((t, hv, d), DType::BF16, device)?;
    let final_state = Tensor::zeros((num_seqs, hv, d, d), DType::F32, device)?;

    // Workspace: cuLA stores one TMA tensormap per SM for O store.
    // Upstream uses `sm_count * 128` bytes; we pad to 256 bytes/SM for safety.
    let sm_count = detect_sm_count();
    let workspace_bytes = (sm_count as usize) * 256;
    let workspace = Tensor::zeros((workspace_bytes,), DType::U8, device)?;

    // ── Extract raw device pointers ─────────────────────────────────
    let (pool_storage, _pool_layout) = out.storage_and_layout();
    let cuda_dev = match &*pool_storage {
        candle_core::Storage::Cuda(s) => s.device().clone(),
        _ => bail!("kda_prefill: output not on CUDA"),
    };
    drop(pool_storage);
    let stream = cuda_dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *const c_void;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("kda_prefill: tensor not on CUDA"),
            };
            let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64
        }};
    }

    let q_ptr = cuda_ptr!(q_c, bf16);
    let k_ptr = cuda_ptr!(k_c, bf16);
    let v_ptr = cuda_ptr!(v_c, bf16);
    let alpha_ptr = cuda_ptr!(alpha_c, f32);
    let beta_ptr = cuda_ptr!(beta_c, f32);
    let cu_ptr = cuda_ptr!(cu_c, i32);
    let out_ptr = cuda_ptr!(out, bf16);
    let state_ptr = cuda_ptr!(final_state, f32);
    let workspace_ptr = cuda_ptr!(workspace, u8);

    let init_ptr = match init_c.as_ref() {
        Some(s) => Some(cuda_ptr!(s, f32)),
        None => None,
    };

    // ── Launch ──────────────────────────────────────────────────────
    // Upstream cuLA is MHA-only — we've already returned None for GVA
    // shapes above, so hk == hv here.
    unsafe {
        cula::kda_fwd_prefill_sm90(
            stream_ptr,
            out_ptr as *mut c_void,
            state_ptr as *mut f32,
            q_ptr as *const c_void,
            k_ptr as *const c_void,
            v_ptr as *const c_void,
            init_ptr.map(|p| p as *const f32),
            Some(alpha_ptr as *const f32),
            Some(beta_ptr as *const f32),
            cu_ptr as *const i32,
            workspace_ptr as *mut u8,
            num_seqs as i32,
            hv as i32,
            d as i32,
            t as i64,
            scale,
            true, // safe_gate — the only path compiled in cuLA SM90
            sm_count,
        )
        .map_err(candle_core::Error::msg)?;
    }

    // Keep source tensors alive until after the kernel launch.
    drop(q_c);
    drop(k_c);
    drop(v_c);
    drop(alpha_c);
    drop(beta_c);
    drop(cu_c);
    drop(init_c);
    drop(workspace);

    Ok(Some((out, final_state)))
}
