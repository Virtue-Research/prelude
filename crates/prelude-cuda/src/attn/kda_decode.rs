//! cuLA KDA fused batched decode — one kernel call per layer for all
//! requests in a decode batch.
//!
//! Mirrors `cula.kda.kda_decode.kda_decode` from upstream cuLA: loads each
//! request's recurrent state from the pool via its slot id, applies the
//! gated delta rule (L2-norm + softplus gate + beta + state update +
//! output) in one fused kernel, and writes the updated state back in place.
//!
//! Only the varlen layout is used here — matches the `[1, N, ...]` packing
//! that `deltanet_varlen_pooled` naturally has in the single-token decode
//! case. The kernel launcher reads `N` and `pool_size` from the tensor
//! shapes at runtime, so one compiled variant per `(H, HV, V)` serves any
//! batch size.

use candle_core::backend::BackendStorage;
use cudarc::driver::DevicePtr;
use half::bf16;
use prelude_core::tensor::{bail, DType, DeviceExt, Result, Tensor, D};
use cula::dsl::{
    DLDataType, DLDevice, DLTensor, DslKernelRegistry, TVMFFIAny, KDLBFLOAT, KDLCUDA, KDLFLOAT,
    KDLINT,
};
use std::ffi::c_void;

const BF16_DT: DLDataType = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
const F32_DT: DLDataType = DLDataType { code: KDLFLOAT, bits: 32, lanes: 1 };
const I32_DT: DLDataType = DLDataType { code: KDLINT, bits: 32, lanes: 1 };

/// Delegate to cuLA's own singleton registry — cuLA knows which
/// `lookup_dsl` function to bind to, so we don't duplicate the
/// `OnceLock` dance here.
fn registry() -> &'static DslKernelRegistry {
    cula::dsl::registry()
}

fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn gpu_dl(
    data: *mut c_void, dev_id: i32, dtype: DLDataType,
    shape: &[i64], strides: &[i64],
) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice { device_type: KDLCUDA, device_id: dev_id },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

/// Small-batch kernels process up to this many requests. Above that the
/// large-batch kernel is a better fit (matches cuLA's own split point).
const SMALL_BATCH_THRESHOLD: usize = 32;

/// Build the DSL kernel key that matches the symbols we AOT-compiled in
/// `compile_kernels.py::compile_kda_decode`. We only register `varlen`
/// variants here because the wrapper always submits a `[1, N, ...]`
/// packing.
fn kernel_name(variant: &str, h: usize, hv: usize, v: usize, arch: u32) -> String {
    format!(
        "cula_kda_decode_{variant}_h{h}_hv{hv}_v{v}_l2norm_sm{arch}"
    )
}

/// True iff the current CUDA device has an AOT-compiled `kda_decode`
/// kernel variant. Mirrors the arch gate in `try_decode` but is cheap
/// enough for hot-path callers to query before they start mutating
/// the DeltaNet pool's conv_state — running conv1d into the pool and
/// then discovering kda is unavailable would advance conv_state twice
/// (once in the fused fast path, once in the sequential fallback) and
/// produce repeated tokens at decode.
pub(crate) fn supported_on_current_arch() -> bool {
    let arch = registry().arch();
    arch == 90 || arch == 100
}


/// Try to run the fused cuLA `kda_decode` kernel over a decode batch.
///
/// Returns `Ok(Some(o))` if the kernel executed — `o` has shape `[N, HV, V]`
/// in BF16. Returns `Ok(None)` if no compiled variant matches this GPU arch
/// or this model's `(H, HV, V)`, so the caller can fall back to the
/// sequential per-request loop. Returns `Err` only for actual kernel /
/// tensor errors.
///
/// Contract (inputs come straight from the per-token projections and
/// layer parameters — the kernel does L2 norm + softplus gate + beta
/// sigmoid internally):
///
/// - `q`, `k`: `[N, H, K]` BF16, one token per request
/// - `v`: `[N, HV, V]` BF16
/// - `a_raw`: `[N, HV]` BF16 (output of `in_proj_a`, scalar per v-head)
/// - `b_raw`: `[N, HV]` BF16 (output of `in_proj_b`, scalar per v-head)
/// - `a_log`: `[HV]` F32 layer parameter
/// - `dt_bias_raw`: `[HV]` (any float dtype) — scalar per v-head
/// - `pool_state`: the full pool tensor `[pool_size, HV, V, K]` F32
/// - `slot_ids_gpu`: `[N]` I32 on the same device as everything else
///
/// The kernel expects per-key-dim `a: [N, HV, K]` and per-key-dim
/// `dt_bias: [HV, K]`. We broadcast the scalar-per-head tensors across K
/// before the call. Broadcasting a constant across K is mathematically
/// equivalent to scalar decay: `gate[k] = exp(-exp(A_log[hv]) *
/// softplus(a[hv] + dt_bias[hv]))` is the same value for every `k`, and
/// the state update `state[h, v, k] *= gate[k]` reduces to a uniform
/// decay per head.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a_raw: &Tensor,
    b_raw: &Tensor,
    a_log: &Tensor,
    dt_bias_raw: &Tensor,
    pool_state: &Tensor,
    slot_ids_gpu: &Tensor,
) -> Result<Option<Tensor>> {
    let reg = registry();
    let arch = reg.arch();

    // Our AOT build only covers Hopper (SM90) and Blackwell (SM100) for
    // kda_decode today. Everything else falls through to the composed path.
    if arch != 90 && arch != 100 {
        return Ok(None);
    }

    // ── Shape + dtype validation ─────────────────────────────────────
    let (n, h, key_dim) = q.dims3()?;
    let (nk, hk, key_dim_k) = k.dims3()?;
    let (nv, hv, val_dim) = v.dims3()?;
    if n != nk || n != nv {
        bail!("kda_decode: q/k/v batch dim mismatch: q={n} k={nk} v={nv}");
    }
    if h != hk {
        bail!("kda_decode: q/k head count mismatch: {h} vs {hk}");
    }
    if key_dim != key_dim_k {
        bail!("kda_decode: q/k head_dim mismatch: {key_dim} vs {key_dim_k}");
    }
    // The compiled kernel is specialized on K = TILE_K = 128.
    if key_dim != 128 {
        return Ok(None);
    }

    // GQA (H < HV) is supported by the kernel itself via `i_h = i_hv //
    // (HV // H)`, as long as a variant with the right head counts was
    // AOT-compiled. The caller is responsible for only routing supported
    // combos through here (see `deltanet_decode_batched_fused`).

    // Look up the variant that matches this (HV, V, arch). `small_varlen`
    // uses a tighter SMEM tile and is faster for N < 32.
    let variant = if n < SMALL_BATCH_THRESHOLD { "small_varlen" } else { "large_varlen" };
    let name = kernel_name(variant, h, hv, val_dim, arch);
    let Some(kernel) = reg.get("kda_decode", &name) else {
        return Ok(None);
    };

    // Everything must already be BF16/F32 and on the same CUDA device.
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        bail!("kda_decode: q/k/v must be BF16");
    }
    if a_raw.dtype() != DType::BF16 || b_raw.dtype() != DType::BF16 {
        bail!("kda_decode: a_raw/b_raw must be BF16");
    }
    if a_log.dtype() != DType::F32 || pool_state.dtype() != DType::F32 {
        bail!("kda_decode: a_log / pool_state must be F32");
    }
    if slot_ids_gpu.dtype() != DType::U32 && slot_ids_gpu.dtype() != DType::I64 {
        bail!("kda_decode: slot_ids must be U32 or I64");
    }

    let device = q.device();
    if !device.is_cuda() {
        return Ok(None);
    }
    let dev_id = device.ordinal() as i32;

    // Pool shape check: [pool_size, HV, V, K].
    let pool_dims = pool_state.dims();
    if pool_dims.len() != 4
        || pool_dims[1] != hv
        || pool_dims[2] != val_dim
        || pool_dims[3] != key_dim
    {
        bail!(
            "kda_decode: pool_state shape {:?} doesn't match (_, {hv}, {val_dim}, {key_dim})",
            pool_dims
        );
    }

    // ── Shape rewrites to match the varlen launcher's expectations ──
    //
    //  q/k:         [N, H, K]    ->  [1, N, H, K]
    //  v:           [N, HV, V]   ->  [1, N, HV, V]
    //  a:           [N, HV]      ->  [N, HV, K]    (broadcast across K)
    //  b:           [N, HV]      ->  [N, HV]       (no change)
    //  dt_bias:     [HV]         ->  [HV, K] F32   (cast + broadcast across K)
    //  slot_ids:    [N] U32      ->  [N] I32       (cast if needed)
    //  output:      [1, N, HV, V] BF16             (allocated here)

    let q4 = q.unsqueeze(0)?.contiguous()?;  // [1, N, H, K]
    let k4 = k.unsqueeze(0)?.contiguous()?;
    let v4 = v.unsqueeze(0)?.contiguous()?;

    // Expand a/dt_bias across the K axis so the per-head scalar becomes
    // the per-key-dim constant the kernel expects. `expand` yields a
    // non-contiguous view, `contiguous()` materialises the repeated data.
    let a3 = a_raw
        .unsqueeze(D::Minus1)?
        .broadcast_as((n, hv, key_dim))?
        .contiguous()?; // [N, HV, K] BF16
    let b2 = b_raw.contiguous()?; // [N, HV] BF16

    let dt_bias2 = dt_bias_raw
        .to_dtype(DType::F32)?
        .reshape((hv,))?
        .unsqueeze(D::Minus1)?
        .broadcast_as((hv, key_dim))?
        .contiguous()?; // [HV, K] F32
    let a_log1 = a_log.reshape((hv,))?.contiguous()?; // [HV] F32

    let slot_ids_i32 = if slot_ids_gpu.dtype() == DType::I64 {
        slot_ids_gpu.to_dtype(DType::I64)? // candle lacks direct i64→i32, stage through
            .to_dtype(DType::U32)?
            .contiguous()?
    } else {
        slot_ids_gpu.to_dtype(DType::U32)?.contiguous()?
    };

    // cu_seqlens = [0, 1, 2, ..., N] I32 on the same device.
    let cu_seqlens_host: Vec<i32> = (0..=n as i32).collect();
    let cu_seqlens_gpu =
        Tensor::from_vec(cu_seqlens_host, (n + 1,), device)?; // I32 on device

    // Output buffer: [1, N, HV, V] BF16.
    let out = unsafe {
        Tensor::zeros((1, n, hv, val_dim), DType::BF16, device)?
    };

    // Grab raw device pointers — the macro drops its RwLockReadGuard at
    // the end of the macro scope, same pattern flashinfer.rs uses. The
    // underlying storage stays alive because the Tensor bindings below
    // are held across the kernel call.
    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty, $stream:expr) => {{
            let (storage, layout) = $t.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("kda_decode: tensor not on CUDA"),
            };
            let slice = cuda
                .as_cuda_slice::<$ty>()?
                .slice(layout.start_offset()..);
            let (ptr, _guard) = slice.device_ptr($stream);
            ptr as u64 as *mut c_void
        }};
    }

    // Need a CUDA stream + device id to convert slices into raw pointers.
    let (pool_storage, _pool_layout) = pool_state.storage_and_layout();
    let cuda_dev = match &*pool_storage {
        candle_core::Storage::Cuda(s) => s.device().clone(),
        _ => bail!("kda_decode: pool_state not on CUDA"),
    };
    drop(pool_storage);
    let stream = cuda_dev.cuda_stream();

    let q_ptr = cuda_ptr!(q4, bf16, &stream);
    let k_ptr = cuda_ptr!(k4, bf16, &stream);
    let v_ptr = cuda_ptr!(v4, bf16, &stream);
    let a_ptr = cuda_ptr!(a3, bf16, &stream);
    let b_ptr = cuda_ptr!(b2, bf16, &stream);
    let a_log_ptr = cuda_ptr!(a_log1, f32, &stream);
    let dt_bias_ptr = cuda_ptr!(dt_bias2, f32, &stream);
    let pool_ptr = cuda_ptr!(pool_state, f32, &stream);
    let slot_ids_ptr = cuda_ptr!(slot_ids_i32, u32, &stream);
    let cu_seqlens_ptr = cuda_ptr!(cu_seqlens_gpu, i32, &stream);
    let o_ptr = cuda_ptr!(out, bf16, &stream);

    // ── Build DLTensors (shapes + strides) ───────────────────────────
    let q_shape = [1i64, n as i64, h as i64, key_dim as i64];
    let q_strides = contiguous_strides(&q_shape);
    let v_shape = [1i64, n as i64, hv as i64, val_dim as i64];
    let v_strides = contiguous_strides(&v_shape);
    let a_shape = [n as i64, hv as i64, key_dim as i64];
    let a_strides = contiguous_strides(&a_shape);
    let b_shape = [n as i64, hv as i64];
    let b_strides = contiguous_strides(&b_shape);
    let a_log_shape = [hv as i64];
    let a_log_strides = [1i64];
    let dt_bias_shape = [hv as i64, key_dim as i64];
    let dt_bias_strides = contiguous_strides(&dt_bias_shape);
    let pool_shape: Vec<i64> = pool_dims.iter().map(|&d| d as i64).collect();
    let pool_strides = contiguous_strides(&pool_shape);
    let slot_shape = [n as i64];
    let slot_strides = [1i64];
    let cu_shape = [(n + 1) as i64];
    let cu_strides = [1i64];

    let dl_q = gpu_dl(q_ptr, dev_id, BF16_DT, &q_shape, &q_strides);
    let dl_k = gpu_dl(k_ptr, dev_id, BF16_DT, &q_shape, &q_strides);
    let dl_v = gpu_dl(v_ptr, dev_id, BF16_DT, &v_shape, &v_strides);
    let dl_a = gpu_dl(a_ptr, dev_id, BF16_DT, &a_shape, &a_strides);
    let dl_b = gpu_dl(b_ptr, dev_id, BF16_DT, &b_shape, &b_strides);
    let dl_a_log = gpu_dl(a_log_ptr, dev_id, F32_DT, &a_log_shape, &a_log_strides);
    let dl_dt_bias = gpu_dl(dt_bias_ptr, dev_id, F32_DT, &dt_bias_shape, &dt_bias_strides);
    let dl_pool = gpu_dl(pool_ptr, dev_id, F32_DT, &pool_shape, &pool_strides);
    let dl_slot = gpu_dl(slot_ids_ptr, dev_id, I32_DT, &slot_shape, &slot_strides);
    let dl_cu = gpu_dl(cu_seqlens_ptr, dev_id, I32_DT, &cu_shape, &cu_strides);
    let dl_o = gpu_dl(o_ptr, dev_id, BF16_DT, &v_shape, &v_strides);

    // The launcher signature (after Constexpr elision) is:
    //   (cu_seqlens, q, k, v, a, b, A_log, dt_bias, h0_source, h0_indices,
    //    o, stream)
    // cuLA's `run_small_batch_varlen` takes `stream: cuda.CUstream` as a
    // regular (non-`EnvStream`) param, so the TVM FFI wrapper expects it
    // as an opaque-ptr runtime arg. We also call `set_stream` so any
    // env-based lookups still work.
    let raw_stream = unsafe { stream.cu_stream() } as *mut c_void;
    reg.set_stream(dev_id, raw_stream);
    let args = [
        TVMFFIAny::dltensor(&dl_cu),
        TVMFFIAny::dltensor(&dl_q),
        TVMFFIAny::dltensor(&dl_k),
        TVMFFIAny::dltensor(&dl_v),
        TVMFFIAny::dltensor(&dl_a),
        TVMFFIAny::dltensor(&dl_b),
        TVMFFIAny::dltensor(&dl_a_log),
        TVMFFIAny::dltensor(&dl_dt_bias),
        TVMFFIAny::dltensor(&dl_pool),
        TVMFFIAny::dltensor(&dl_slot),
        TVMFFIAny::dltensor(&dl_o),
        TVMFFIAny::opaque_ptr(raw_stream),
    ];

    // Use `call_tvm_ffi` directly rather than `DslKernelRegistry::call_kernel`,
    // because the helper in `cula` masks the real TVM error with
    // "TVM FFI internal failure" when the kernel didn't touch CUDA. We
    // want the actual message that TVM FFI stashed via TVMFFIErrorSet.
    unsafe {
        tvm_static_ffi::call_tvm_ffi(kernel, &args)
            .map_err(|e| candle_core::Error::Msg(format!("cuLA kda_decode: {e}")))?;
    }

    // Output is [1, N, HV, V]; callers want [N, HV, V].
    Ok(Some(out.squeeze(0)?))
}
