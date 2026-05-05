//! Gated DeltaNet (GDN) prefill.
//!
//! SM90 uses FlashInfer's Hopper WGMMA `gdn_prefill` utility. SM10x uses
//! Prelude's in-tree recurrent BF16 kernel with the same linear-space
//! DeltaNet semantics. An experimental FlashInfer CuTe DSL Blackwell AOT path
//! is compiled when available, but stays behind `PRELUDE_GDN_SM100_AOT=1`
//! until its TVM FFI launch path is fully validated.
//!
//! ## Semantics
//!
//! Unlike cuLA's KDA prefill, this kernel takes **linear-space**
//! per-step decay `alpha = exp(g_scalar)` and a scalar-per-head layout
//! `[total_seq, num_sab_heads]`. No per-element broadcast, no chunk
//! cumsum, no `RCP_LN2` rescaling, no `safe_gate` clamp. It natively
//! matches HF transformers' `chunk_gated_delta_rule` math, so wiring
//! Qwen3.5 through it should be bit-exact (within BF16) to the HF
//! reference.
//!
//! ## Shapes (all varlen; `num_sab_heads = max(num_q_heads, num_v_heads)`)
//!
//! - `q`, `k`: `[packed_seq, num_q/k_heads, head_dim]` BF16
//! - `v`: `[packed_seq, num_v_heads, head_dim]` BF16
//! - `alpha`: `[packed_seq, num_sab_heads]` F32 (= per-step linear decay)
//! - `beta`:  `[packed_seq, num_sab_heads]` F32 (= sigmoid(raw))
//! - `cu_seqlens`: `[num_seqs+1]` **I64** (note: not I32 like cuLA)
//! - `initial_state`: optional `[num_seqs, num_sab_heads, head_dim, head_dim]` F32
//!
//! Output:
//! - `output`: `[packed_seq, num_sab_heads, head_dim]` BF16
//! - `output_state`: `[num_seqs, num_sab_heads, head_dim, head_dim]` F32

use candle_core::Shape;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::DevicePtr;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use flashinfer::KernelRegistry;
use flashinfer::types::{
    DLDataType, DLDevice, DLTensor, KDLBFLOAT, KDLCUDA, KDLFLOAT, KDLINT, KDLUINT, TVMFFIAny,
};
use half::bf16;
use prelude_core::tensor::{DType, DeviceExt, Result, Tensor, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

use crate::{MOD_GDN_PREFILL_RECURRENT, PTX_GDN_PREFILL_RECURRENT};

const BF16_DT: DLDataType = DLDataType {
    code: KDLBFLOAT,
    bits: 16,
    lanes: 1,
};
const F32_DT: DLDataType = DLDataType {
    code: KDLFLOAT,
    bits: 32,
    lanes: 1,
};
const I8_DT: DLDataType = DLDataType {
    code: KDLINT,
    bits: 8,
    lanes: 1,
};
const I32_DT: DLDataType = DLDataType {
    code: KDLINT,
    bits: 32,
    lanes: 1,
};
const I64_DT: DLDataType = DLDataType {
    code: KDLINT,
    bits: 64,
    lanes: 1,
};
const U8_DT: DLDataType = DLDataType {
    code: KDLUINT,
    bits: 8,
    lanes: 1,
};

fn registry() -> &'static KernelRegistry {
    static REG: OnceLock<KernelRegistry> = OnceLock::new();
    REG.get_or_init(KernelRegistry::new)
}

fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn gpu_dl(
    data: *mut c_void,
    dev_id: i32,
    dtype: DLDataType,
    shape: &[i64],
    strides: &[i64],
) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice {
            device_type: KDLCUDA,
            device_id: dev_id,
        },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

/// Drive the FlashInfer `gdn_prefill` utility.
///
/// Returns `Ok(Some((o, final_state)))` on success, `Ok(None)` if the
/// registry doesn't have the kernel (unsupported arch or build-time
/// unavailable), `Err` for shape/dtype/CUDA errors.
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
    let reg = registry();
    // FlashInfer's AOT GDN module is SM90-only; Blackwell is served by the
    // custom recurrent kernel below. Avoid calling the TVM FFI entrypoint on
    // other architectures because FlashInfer launcher errors can terminate the
    // process before Rust can recover.
    let arch = reg.arch();
    if !(arch == 90 || (100..110).contains(&arch)) {
        return Ok(None);
    }

    let device = q.device();
    if !device.is_cuda() {
        return Ok(None);
    }

    // ── Shape validation ────────────────────────────────────────────
    let (t, hq, d) = q.dims3()?;
    let (tk, hk, dk) = k.dims3()?;
    let (tv, hv, dv) = v.dims3()?;
    if t != tk || t != tv {
        bail!("gdn_prefill: q/k/v seq-dim mismatch: q={t} k={tk} v={tv}");
    }
    if d != dk || d != dv {
        bail!("gdn_prefill: q/k/v head_dim mismatch: {d} / {dk} / {dv}");
    }
    // num_sab_heads = max(num_q, num_v); grouping axis depends on GVA/GQA.
    let num_sab_heads = hq.max(hv);
    if hq >= hv {
        // GQA: num_k == num_v, num_q is multiple of num_v.
        if hk != hv || hq % hk != 0 {
            bail!(
                "gdn_prefill: GQA requires num_k==num_v and num_q%num_k==0; got q={hq} k={hk} v={hv}"
            );
        }
    } else {
        // GVA (Qwen3.5 / Qwen3-next): num_q == num_k, num_v is multiple of num_q.
        if hq != hk || hv % hq != 0 {
            bail!(
                "gdn_prefill: GVA requires num_q==num_k and num_v%num_q==0; got q={hq} k={hk} v={hv}"
            );
        }
    }
    // Kernel is specialized on head_dim=128 (Qwen3.5 / Qwen3-next).
    if d != 128 {
        return Ok(None);
    }

    let (ta, hva) = alpha.dims2()?;
    if ta != t || hva != num_sab_heads {
        bail!(
            "gdn_prefill: alpha shape {:?} mismatches ({t}, {num_sab_heads})",
            (ta, hva)
        );
    }
    let (tb, hvb) = beta.dims2()?;
    if tb != t || hvb != num_sab_heads {
        bail!(
            "gdn_prefill: beta shape {:?} mismatches ({t}, {num_sab_heads})",
            (tb, hvb)
        );
    }

    // cu_seqlens must be I64 — FlashInfer's kernel enforces this.
    if cu_seqlens.dtype() != DType::I64 {
        bail!(
            "gdn_prefill: cu_seqlens must be I64 (got {:?})",
            cu_seqlens.dtype()
        );
    }
    let num_seqs = cu_seqlens.dim(0)? - 1;

    // ── Dtype checks ────────────────────────────────────────────────
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        bail!("gdn_prefill: q/k/v must be BF16");
    }
    if alpha.dtype() != DType::F32 || beta.dtype() != DType::F32 {
        bail!("gdn_prefill: alpha/beta must be F32");
    }
    if let Some(s) = initial_state {
        if s.dtype() != DType::F32 {
            bail!("gdn_prefill: initial_state must be F32");
        }
        let dims = s.dims();
        if dims.len() != 4
            || dims[0] != num_seqs
            || dims[1] != num_sab_heads
            || dims[2] != d
            || dims[3] != d
        {
            bail!(
                "gdn_prefill: initial_state shape {:?} mismatches ({num_seqs}, {num_sab_heads}, {d}, {d})",
                dims
            );
        }
    }

    if (100..110).contains(&arch) {
        if sm100_aot_enabled() {
            if let Some(result) = try_prefill_blackwell_aot(
                reg,
                q,
                k,
                v,
                alpha,
                beta,
                cu_seqlens,
                initial_state,
                scale,
                num_seqs,
                num_sab_heads,
            )? {
                return Ok(Some(result));
            }
        }
        return try_prefill_recurrent_blackwell(
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            initial_state,
            scale,
            num_seqs,
            num_sab_heads,
        );
    }

    let Some(gdn_fn) = reg.get_utility("gdn_prefill") else {
        return Ok(None);
    };

    // ── Materialize contiguous inputs ───────────────────────────────
    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;
    let alpha_c = alpha.contiguous()?;
    let beta_c = beta.contiguous()?;
    let cu_c = cu_seqlens.contiguous()?;
    let init_c = match initial_state {
        Some(s) => Some(s.contiguous()?),
        None => None,
    };

    // ── Output buffers ──────────────────────────────────────────────
    let out = Tensor::zeros((t, num_sab_heads, d), DType::BF16, device)?;
    // Kernel unconditionally writes final state — allocate even if the
    // caller doesn't care. Shape `[num_seqs, num_sab_heads, D, D]` F32.
    let final_state = Tensor::zeros((num_seqs, num_sab_heads, d, d), DType::F32, device)?;

    // Workspace: kernel stores one TMA tensormap per SM on Hopper.
    // Upstream's Python wrapper uses `sm_count * 128` bytes; we pad to
    // 256/SM for safety. 132 * 256 = 33792 on H200, trivial vs 128 MB
    // we'd over-provision otherwise.
    let sm_count = detect_sm_count();
    let workspace_bytes = (sm_count as usize) * 256;
    let workspace = Tensor::zeros((workspace_bytes,), DType::U8, device)?;

    // ── Extract device pointers (flashinfer-style storage_and_layout) ──
    let (out_storage, _) = out.storage_and_layout();
    let cuda_dev = match &*out_storage {
        candle_core::Storage::Cuda(s) => s.device().clone(),
        _ => bail!("gdn_prefill: output allocated on non-CUDA device"),
    };
    drop(out_storage);
    let stream = cuda_dev.cuda_stream();
    let dev_id = device.ordinal() as i32;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("gdn_prefill: tensor not on CUDA"),
            };
            let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64 as *mut c_void
        }};
    }

    let q_ptr = cuda_ptr!(q_c, bf16);
    let k_ptr = cuda_ptr!(k_c, bf16);
    let v_ptr = cuda_ptr!(v_c, bf16);
    let alpha_ptr = cuda_ptr!(alpha_c, f32);
    let beta_ptr = cuda_ptr!(beta_c, f32);
    let cu_ptr = cuda_ptr!(cu_c, i64);
    let out_ptr = cuda_ptr!(out, bf16);
    let state_ptr = cuda_ptr!(final_state, f32);
    let ws_ptr = cuda_ptr!(workspace, u8);
    let init_ptr = init_c.as_ref().map(|s| {
        let (storage, layout) = s.storage_and_layout();
        let cuda = match &*storage {
            candle_core::Storage::Cuda(c) => c,
            _ => panic!("gdn_prefill: init state not on CUDA"),
        };
        let slice = cuda
            .as_cuda_slice::<f32>()
            .expect("gdn_prefill: init state slice")
            .slice(layout.start_offset()..);
        let (ptr, _guard) = slice.device_ptr(&stream);
        ptr as u64 as *mut c_void
    });

    // ── Build DLTensors ─────────────────────────────────────────────
    let q_shape = [t as i64, hq as i64, d as i64];
    let q_strides = contiguous_strides(&q_shape);
    let k_shape = [t as i64, hk as i64, d as i64];
    let k_strides = contiguous_strides(&k_shape);
    let v_shape = [t as i64, hv as i64, d as i64];
    let v_strides = contiguous_strides(&v_shape);
    let o_shape = [t as i64, num_sab_heads as i64, d as i64];
    let o_strides = contiguous_strides(&o_shape);
    let state_shape = [num_seqs as i64, num_sab_heads as i64, d as i64, d as i64];
    let state_strides = contiguous_strides(&state_shape);
    let ab_shape = [t as i64, num_sab_heads as i64];
    let ab_strides = contiguous_strides(&ab_shape);
    let cu_shape = [(num_seqs + 1) as i64];
    let cu_strides = [1i64];
    let ws_shape = [workspace_bytes as i64];
    let ws_strides = [1i64];

    let dl_q = gpu_dl(q_ptr, dev_id, BF16_DT, &q_shape, &q_strides);
    let dl_k = gpu_dl(k_ptr, dev_id, BF16_DT, &k_shape, &k_strides);
    let dl_v = gpu_dl(v_ptr, dev_id, BF16_DT, &v_shape, &v_strides);
    let dl_o = gpu_dl(out_ptr, dev_id, BF16_DT, &o_shape, &o_strides);
    let dl_state = gpu_dl(state_ptr, dev_id, F32_DT, &state_shape, &state_strides);
    let dl_alpha = gpu_dl(alpha_ptr, dev_id, F32_DT, &ab_shape, &ab_strides);
    let dl_beta = gpu_dl(beta_ptr, dev_id, F32_DT, &ab_shape, &ab_strides);
    let dl_cu = gpu_dl(cu_ptr, dev_id, I64_DT, &cu_shape, &cu_strides);
    let dl_ws = gpu_dl(ws_ptr, dev_id, U8_DT, &ws_shape, &ws_strides);
    let dl_init = init_ptr.map(|ptr| gpu_dl(ptr, dev_id, F32_DT, &state_shape, &state_strides));

    let raw_stream = unsafe { stream.cu_stream() } as *mut c_void;
    reg.set_stream(dev_id, raw_stream);

    // The vendored flashinfer gdn_prefill is AOT-compiled with
    // `enable_checkpointing=false`, which drops the trailing 3
    // checkpoint args from the kernel signature. The compiled kernel
    // expects 11 args; passing 14 trips a tvm::ffi arity check.
    //
    // gdn_prefill(output, output_state, q, k, v, cu_seqlens,
    //             input_state?, alpha?, beta?, scale, workspace)
    let args = [
        TVMFFIAny::dltensor(&dl_o),
        TVMFFIAny::dltensor(&dl_state),
        TVMFFIAny::dltensor(&dl_q),
        TVMFFIAny::dltensor(&dl_k),
        TVMFFIAny::dltensor(&dl_v),
        TVMFFIAny::dltensor(&dl_cu),
        match dl_init.as_ref() {
            Some(dl) => TVMFFIAny::dltensor(dl),
            None => TVMFFIAny::none(),
        },
        TVMFFIAny::dltensor(&dl_alpha),
        TVMFFIAny::dltensor(&dl_beta),
        TVMFFIAny::float64(scale as f64),
        TVMFFIAny::dltensor(&dl_ws),
    ];

    unsafe {
        reg.call(gdn_fn, &args)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer gdn_prefill: {e}")))?;
    }

    // Keep source tensors alive until kernel launch has captured pointers.
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

fn sm100_aot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PRELUDE_GDN_SM100_AOT")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
fn try_prefill_blackwell_aot(
    reg: &KernelRegistry,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    cu_seqlens: &Tensor,
    initial_state: Option<&Tensor>,
    scale: f32,
    num_seqs: usize,
    num_sab_heads: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    let (t, hq, d) = q.dims3()?;
    let (_, hk, _) = k.dims3()?;
    let (_, hv, _) = v.dims3()?;
    if d != 128 {
        return Ok(None);
    }
    if hq >= hv {
        if hk != hv || hq % hv != 0 || num_sab_heads != hq {
            return Ok(None);
        }
    } else if hk != hq || hv % hq != 0 || num_sab_heads != hv {
        return Ok(None);
    }

    let name = format!("gdn_prefill_sm100_bf16_h{hq}_hv{hv}");
    let Some(kernel) = reg.get_utility(&name) else {
        return Ok(None);
    };

    let device = q.device();
    let cuda_dev = match device {
        prelude_core::tensor::Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let dev_id = device.ordinal() as i32;
    let stream = cuda_dev.cuda_stream();

    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;
    let alpha_c = alpha.contiguous()?;
    let beta_c = beta.contiguous()?;
    let cu_i32 = cu_seqlens.to_dtype(DType::U32)?.contiguous()?;
    let init_c = match initial_state {
        Some(s) => s.contiguous()?,
        None => Tensor::zeros((num_seqs, num_sab_heads, d, d), DType::F32, device)?,
    };

    let out = Tensor::zeros((t, num_sab_heads, d), DType::BF16, device)?;
    let final_state = Tensor::zeros((num_seqs, num_sab_heads, d, d), DType::F32, device)?;
    let workspace_bytes = detect_sm_count() as usize * 4 * 128; // q/k/v/o TMA descriptors per persistent CTA.
    let workspace = Tensor::zeros((workspace_bytes,), DType::U8, device)?;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("gdn_prefill_sm100: tensor not on CUDA"),
            };
            let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64 as *mut c_void
        }};
    }

    let q_ptr = cuda_ptr!(q_c, bf16);
    let k_ptr = cuda_ptr!(k_c, bf16);
    let v_ptr = cuda_ptr!(v_c, bf16);
    let alpha_ptr = cuda_ptr!(alpha_c, f32);
    let beta_ptr = cuda_ptr!(beta_c, f32);
    let cu_ptr = cuda_ptr!(cu_i32, u32);
    let init_ptr = cuda_ptr!(init_c, f32);
    let out_ptr = cuda_ptr!(out, bf16);
    let state_ptr = cuda_ptr!(final_state, f32);
    let ws_ptr = cuda_ptr!(workspace, u8);

    let q_shape = [t as i64, hq as i64, d as i64];
    let q_strides = contiguous_strides(&q_shape);
    let k_shape = [t as i64, hk as i64, d as i64];
    let k_strides = contiguous_strides(&k_shape);
    let v_shape = [t as i64, hv as i64, d as i64];
    let v_strides = contiguous_strides(&v_shape);
    let o_shape = [t as i64, num_sab_heads as i64, d as i64];
    let o_strides = contiguous_strides(&o_shape);
    let state_shape = [num_seqs as i64, num_sab_heads as i64, d as i64, d as i64];
    let state_strides = contiguous_strides(&state_shape);
    let ab_shape = [t as i64, num_sab_heads as i64];
    let ab_strides = contiguous_strides(&ab_shape);
    let cu_shape = [(num_seqs + 1) as i64];
    let cu_strides = [1i64];
    let ws_shape = [workspace_bytes as i64];
    let ws_strides = [1i64];

    let dl_q = gpu_dl(q_ptr, dev_id, BF16_DT, &q_shape, &q_strides);
    let dl_k = gpu_dl(k_ptr, dev_id, BF16_DT, &k_shape, &k_strides);
    let dl_v = gpu_dl(v_ptr, dev_id, BF16_DT, &v_shape, &v_strides);
    let dl_alpha = gpu_dl(alpha_ptr, dev_id, F32_DT, &ab_shape, &ab_strides);
    let dl_beta = gpu_dl(beta_ptr, dev_id, F32_DT, &ab_shape, &ab_strides);
    let dl_o = gpu_dl(out_ptr, dev_id, BF16_DT, &o_shape, &o_strides);
    let dl_cu = gpu_dl(cu_ptr, dev_id, I32_DT, &cu_shape, &cu_strides);
    let dl_init = gpu_dl(init_ptr, dev_id, F32_DT, &state_shape, &state_strides);
    let dl_state = gpu_dl(state_ptr, dev_id, F32_DT, &state_shape, &state_strides);
    let dl_ws = gpu_dl(ws_ptr, dev_id, I8_DT, &ws_shape, &ws_strides);

    let raw_stream = unsafe { stream.cu_stream() } as *mut c_void;
    reg.set_stream(dev_id, raw_stream);
    let args = [
        TVMFFIAny::dltensor(&dl_q),
        TVMFFIAny::dltensor(&dl_k),
        TVMFFIAny::dltensor(&dl_v),
        TVMFFIAny::dltensor(&dl_alpha),
        TVMFFIAny::dltensor(&dl_beta),
        TVMFFIAny::dltensor(&dl_o),
        TVMFFIAny::dltensor(&dl_cu),
        TVMFFIAny::dltensor(&dl_init),
        TVMFFIAny::dltensor(&dl_state),
        TVMFFIAny::none(), // output_checkpoints
        TVMFFIAny::none(), // cu_checkpoints
        TVMFFIAny::int32(0),
        TVMFFIAny::float64(scale as f64),
        TVMFFIAny::dltensor(&dl_ws),
        TVMFFIAny::opaque_ptr(raw_stream),
    ];

    unsafe {
        reg.call(kernel, &args)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer GDN SM100: {e}")))?;
    }

    drop(q_c);
    drop(k_c);
    drop(v_c);
    drop(alpha_c);
    drop(beta_c);
    drop(cu_i32);
    drop(init_c);
    drop(workspace);

    Ok(Some((out, final_state)))
}

#[allow(clippy::too_many_arguments)]
fn try_prefill_recurrent_blackwell(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    cu_seqlens: &Tensor,
    initial_state: Option<&Tensor>,
    scale: f32,
    num_seqs: usize,
    num_sab_heads: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    let (t, hq, d) = q.dims3()?;
    let (_, hk, _) = k.dims3()?;
    let (_, hv, _) = v.dims3()?;
    if hq != hk || hv < hq || hv % hq != 0 || num_sab_heads != hv || d != 128 {
        return Ok(None);
    }

    let device = q.device();
    let cuda_dev = match device {
        prelude_core::tensor::Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };

    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;
    let alpha_c = alpha.contiguous()?;
    let beta_c = beta.contiguous()?;
    let cu_c = cu_seqlens.contiguous()?;
    let init_c = match initial_state {
        Some(s) => s.contiguous()?,
        None => Tensor::zeros((1,), DType::F32, device)?,
    };

    let (q_storage, q_layout) = q_c.storage_and_layout();
    let (k_storage, k_layout) = k_c.storage_and_layout();
    let (v_storage, v_layout) = v_c.storage_and_layout();
    let (alpha_storage, alpha_layout) = alpha_c.storage_and_layout();
    let (beta_storage, beta_layout) = beta_c.storage_and_layout();
    let (cu_storage, cu_layout) = cu_c.storage_and_layout();
    let (init_storage, init_layout) = init_c.storage_and_layout();

    macro_rules! cuda_slice {
        ($storage:expr, $layout:expr, $ty:ty, $name:literal) => {{
            let cuda = match &*$storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("gdn_prefill_recurrent: {} not on CUDA", $name),
            };
            cuda.as_cuda_slice::<$ty>()?.slice($layout.start_offset()..)
        }};
    }

    let q_slice = cuda_slice!(q_storage, q_layout, bf16, "q");
    let k_slice = cuda_slice!(k_storage, k_layout, bf16, "k");
    let v_slice = cuda_slice!(v_storage, v_layout, bf16, "v");
    let alpha_slice = cuda_slice!(alpha_storage, alpha_layout, f32, "alpha");
    let beta_slice = cuda_slice!(beta_storage, beta_layout, f32, "beta");
    let cu_slice = cuda_slice!(cu_storage, cu_layout, i64, "cu_seqlens");
    let init_slice = cuda_slice!(init_storage, init_layout, f32, "initial_state");

    let mut out_buf = unsafe { cuda_dev.alloc::<bf16>(t * hv * d) }?;
    let mut state_buf = unsafe { cuda_dev.alloc::<f32>(num_seqs * hv * d * d) }?;

    let func = cuda_dev.get_or_load_custom_func(
        "gdn_prefill_recurrent_bf16",
        MOD_GDN_PREFILL_RECURRENT,
        PTX_GDN_PREFILL_RECURRENT,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (hv as u32, d as u32, num_seqs as u32),
        block_dim: (d as u32, 1, 1),
        shared_mem_bytes: ((d + 31) / 32 * std::mem::size_of::<f32>()) as u32,
    };

    let num_seqs_i = num_seqs as i32;
    let hq_i = hq as i32;
    let hv_i = hv as i32;
    let d_i = d as i32;
    let has_initial = if initial_state.is_some() { 1i32 } else { 0i32 };

    let mut builder = func.builder();
    builder.arg(&q_slice);
    builder.arg(&k_slice);
    builder.arg(&v_slice);
    builder.arg(&alpha_slice);
    builder.arg(&beta_slice);
    builder.arg(&cu_slice);
    builder.arg(&init_slice);
    builder.arg(&mut out_buf);
    builder.arg(&mut state_buf);
    builder.arg(&num_seqs_i);
    builder.arg(&hq_i);
    builder.arg(&hv_i);
    builder.arg(&d_i);
    builder.arg(&scale);
    builder.arg(&has_initial);
    unsafe { builder.launch(cfg) }.w()?;

    drop(q_storage);
    drop(k_storage);
    drop(v_storage);
    drop(alpha_storage);
    drop(beta_storage);
    drop(cu_storage);
    drop(init_storage);

    let out = tensor_from_bf16(out_buf, device, (t, hv, d))?;
    let final_state = tensor_from_f32(state_buf, device, (num_seqs, hv, d, d))?;
    Ok(Some((out, final_state)))
}

fn tensor_from_bf16(
    slice: CudaSlice<bf16>,
    dev: &prelude_core::tensor::Device,
    shape: impl Into<Shape>,
) -> Result<Tensor> {
    let cuda_dev = match dev {
        prelude_core::tensor::Device::Cuda(d) => d.clone(),
        _ => bail!("gdn_prefill_recurrent: output not on CUDA"),
    };
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, cuda_dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape.into(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn tensor_from_f32(
    slice: CudaSlice<f32>,
    dev: &prelude_core::tensor::Device,
    shape: impl Into<Shape>,
) -> Result<Tensor> {
    let cuda_dev = match dev {
        prelude_core::tensor::Device::Cuda(d) => d.clone(),
        _ => bail!("gdn_prefill_recurrent: output not on CUDA"),
    };
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, cuda_dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape.into(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

// ── Helpers ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

const CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;

fn detect_sm_count() -> i32 {
    static CACHE: OnceLock<i32> = OnceLock::new();
    *CACHE.get_or_init(|| unsafe {
        let mut dev = 0i32;
        if cudaGetDevice(&mut dev) != 0 {
            return 132;
        }
        let mut count = 0i32;
        if cudaDeviceGetAttribute(&mut count, CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT, dev) != 0 {
            return 132;
        }
        if count > 0 { count } else { 132 }
    })
}
