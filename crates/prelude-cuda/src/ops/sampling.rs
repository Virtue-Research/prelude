//! GPU-accelerated batched sampling via FlashInfer kernels.
//!
//! Replaces per-sequence CPU sampling with a single GPU kernel call
//! across the entire decode batch, eliminating N GPU→CPU syncs per step.

use candle_core::backend::BackendStorage;
use cudarc::driver::DevicePtr;
use prelude_core::tensor::{DType, Result, Tensor, bail};
use std::ffi::c_void;

use flashinfer::loader::{KernelRegistry, TVMSafeCallFn};
use flashinfer::types::*;

// ── Kernel lookup (cached) ─────────────────────────────────────────

fn get_sampling_from_logits() -> Option<TVMSafeCallFn> {
    static FUNC: std::sync::OnceLock<Option<TVMSafeCallFn>> = std::sync::OnceLock::new();
    *FUNC.get_or_init(|| {
        let registry = KernelRegistry::new();
        let f = registry.get_utility("sampling_from_logits");
        if f.is_some() {
            tracing::debug!("FlashInfer sampling_from_logits kernel available");
        }
        f
    })
}

fn get_top_k_top_p_sampling() -> Option<TVMSafeCallFn> {
    static FUNC: std::sync::OnceLock<Option<TVMSafeCallFn>> = std::sync::OnceLock::new();
    *FUNC.get_or_init(|| {
        let registry = KernelRegistry::new();
        let f = registry.get_utility("top_k_top_p_sampling_from_probs");
        if f.is_some() {
            tracing::debug!("FlashInfer top_k_top_p_sampling kernel available");
        }
        f
    })
}

fn get_softmax() -> Option<TVMSafeCallFn> {
    static FUNC: std::sync::OnceLock<Option<TVMSafeCallFn>> = std::sync::OnceLock::new();
    *FUNC.get_or_init(|| {
        let registry = KernelRegistry::new();
        registry.get_utility("softmax")
    })
}

// ── Helpers ────────────────────────────────────────────────────────

/// Helper: extract candle CUDA storage or bail.
fn as_candle_cuda<'a>(
    storage: &'a std::sync::RwLockReadGuard<'a, candle_core::Storage>,
    ctx: &str,
) -> Result<&'a candle_core::CudaStorage> {
    match &**storage {
        candle_core::Storage::Cuda(s) => Ok(s),
        _ => bail!("{ctx}: requires CUDA"),
    }
}

/// Extract raw CUDA device pointer from a candle tensor.
fn tensor_to_dl(t: &Tensor, shapes: &[i64], dt: DLDataType) -> Result<DLTensor> {
    let (storage, layout) = t.storage_and_layout();
    let cuda = as_candle_cuda(&storage, "gpu_sampling")?;
    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();
    let (base_ptr, elem_offset) = match t.dtype() {
        DType::F32 => {
            let s = cuda.as_cuda_slice::<f32>()?;
            (s.device_ptr(&stream).0, layout.start_offset())
        }
        DType::I64 => {
            let s = cuda.as_cuda_slice::<i64>()?;
            (s.device_ptr(&stream).0, layout.start_offset())
        }
        DType::U32 => {
            let s = cuda.as_cuda_slice::<u32>()?;
            (s.device_ptr(&stream).0, layout.start_offset())
        }
        DType::U8 => {
            let s = cuda.as_cuda_slice::<u8>()?;
            (s.device_ptr(&stream).0, layout.start_offset())
        }
        dt => bail!("gpu_sampling: unsupported dtype {dt:?}"),
    };
    let data_ptr = base_ptr + (elem_offset * t.dtype().size_in_bytes()) as u64;
    Ok(DLTensor {
        data: data_ptr as *mut c_void,
        device: DLDevice {
            device_type: KDLCUDA,
            device_id: 0,
        },
        ndim: shapes.len() as i32,
        dtype: dt,
        shape: shapes.as_ptr(),
        strides: std::ptr::null(),
        byte_offset: 0,
    })
}

/// Set TVM-FFI CUDA stream to match the candle device stream.
fn set_stream(t: &Tensor) -> Result<()> {
    let (storage, _) = t.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("gpu_sampling: requires CUDA tensor"),
    };
    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();
    let raw_stream = stream.cu_stream() as *mut c_void;
    let registry = KernelRegistry::new();
    registry.set_stream(0, raw_stream);
    Ok(())
}

// ── Public API ─────────────────────────────────────────────────────

/// Batched sampling from logits on GPU via FlashInfer.
///
/// Input: `logits` `[batch_size, vocab_size]` F32 (on GPU)
/// Returns: `[batch_size]` I32 tensor of sampled token IDs (on GPU).
///
/// When `deterministic=true`, uses deterministic sampling (functionally
/// equivalent to argmax for greedy decoding).
pub fn sample_from_logits(logits: &Tensor, deterministic: bool) -> Result<Tensor> {
    let func = get_sampling_from_logits().ok_or_else(|| {
        candle_core::Error::Msg("FlashInfer sampling_from_logits not available".into())
    })?;

    let logits = logits.to_dtype(DType::F32)?.contiguous()?;
    let (batch_size, _vocab_size) = logits.dims2()?;
    let device = logits.device();

    // Allocate output tensor: [batch_size] I32
    let output = Tensor::zeros((batch_size,), DType::I64, device)?;

    set_stream(&logits)?;

    let f32_dt = DLDataType {
        code: KDLFLOAT,
        bits: 32,
        lanes: 1,
    };
    let i64_dt = DLDataType {
        code: KDLINT,
        bits: 64,
        lanes: 1,
    };

    let logits_shape: Vec<i64> = logits.dims().iter().map(|&d| d as i64).collect();
    let output_shape = [batch_size as i64];

    let dl_logits = tensor_to_dl(&logits, &logits_shape, f32_dt)?;
    let dl_output = tensor_to_dl(&output, &output_shape, i64_dt)?;

    // sampling_from_logits(logits, output, maybe_indices, deterministic,
    //                      maybe_seed_arr, seed_val, maybe_offset_arr, offset_val)
    let args = [
        TVMFFIAny::dltensor(&dl_logits),    // logits [bs, vocab] F32
        TVMFFIAny::dltensor(&dl_output),    // output [bs] I64
        TVMFFIAny::none(),                  // maybe_indices (None)
        TVMFFIAny::bool_val(deterministic), // deterministic
        TVMFFIAny::none(),                  // maybe_seed_arr (None)
        TVMFFIAny::int64(42),               // seed_val
        TVMFFIAny::none(),                  // maybe_offset_arr (None)
        TVMFFIAny::int64(0),                // offset_val
    ];

    unsafe {
        tvm_static_ffi::call_tvm_ffi(func, &args)
            .map_err(|e| candle_core::Error::Msg(format!("sampling_from_logits: {e}")))?;
    }

    // FlashInfer outputs I64 — convert to U32 for token IDs
    output.to_dtype(DType::U32)
}

/// Batched top-k + top-p sampling from logits on GPU.
///
/// Steps: softmax(logits) → top_k_top_p_sampling_from_probs
pub fn top_k_top_p_sample(
    logits: &Tensor,
    top_k: i64,
    top_p: f64,
    deterministic: bool,
) -> Result<Tensor> {
    let softmax_fn = get_softmax()
        .ok_or_else(|| candle_core::Error::Msg("FlashInfer softmax not available".into()))?;
    let sample_fn = get_top_k_top_p_sampling().ok_or_else(|| {
        candle_core::Error::Msg("FlashInfer top_k_top_p_sampling not available".into())
    })?;

    let logits = logits.to_dtype(DType::F32)?.contiguous()?;
    let (batch_size, vocab_size) = logits.dims2()?;
    let device = logits.device();

    set_stream(&logits)?;

    let f32_dt = DLDataType {
        code: KDLFLOAT,
        bits: 32,
        lanes: 1,
    };
    let i64_dt = DLDataType {
        code: KDLINT,
        bits: 64,
        lanes: 1,
    };
    let u8_dt = DLDataType {
        code: KDLUINT,
        bits: 8,
        lanes: 1,
    };

    // Allocate workspace and output tensors
    let probs = Tensor::zeros((batch_size, vocab_size), DType::F32, device)?;
    let output = Tensor::zeros((batch_size,), DType::I64, device)?;
    let valid = Tensor::zeros((batch_size,), DType::U8, device)?;
    // Workspace for softmax — needs batch_size * sizeof(float) per FlashInfer docs
    let workspace = Tensor::zeros((batch_size * 4,), DType::U8, device)?;

    let logits_shape: Vec<i64> = logits.dims().iter().map(|&d| d as i64).collect();
    let probs_shape = logits_shape.clone();
    let output_shape = [batch_size as i64];
    let valid_shape = [batch_size as i64];
    let ws_shape = [(batch_size * 4) as i64];

    let dl_ws = tensor_to_dl(&workspace, &ws_shape, u8_dt)?;
    let dl_logits = tensor_to_dl(&logits, &logits_shape, f32_dt)?;
    let dl_probs = tensor_to_dl(&probs, &probs_shape, f32_dt)?;
    let dl_output = tensor_to_dl(&output, &output_shape, i64_dt)?;
    let dl_valid = tensor_to_dl(&valid, &valid_shape, u8_dt)?;

    // Step 1: softmax(workspace, logits, output_probs, maybe_temp_arr, temp_val, enable_pdl)
    let softmax_args = [
        TVMFFIAny::dltensor(&dl_ws),     // workspace
        TVMFFIAny::dltensor(&dl_logits), // logits
        TVMFFIAny::dltensor(&dl_probs),  // output (probs)
        TVMFFIAny::none(),               // maybe_temperature_arr (None)
        TVMFFIAny::float64(1.0),         // temperature_val
        TVMFFIAny::bool_val(false),      // enable_pdl
    ];

    unsafe {
        tvm_static_ffi::call_tvm_ffi(softmax_fn, &softmax_args)
            .map_err(|e| candle_core::Error::Msg(format!("softmax: {e}")))?;
    }

    // Step 2: top_k_top_p_sampling_from_probs(probs, output, valid, maybe_indices,
    //         maybe_top_k_arr, top_k_val, maybe_top_p_arr, top_p_val,
    //         deterministic, maybe_seed_arr, seed_val, maybe_offset_arr, offset_val)
    let sample_args = [
        TVMFFIAny::dltensor(&dl_probs),     // probs [bs, vocab] F32
        TVMFFIAny::dltensor(&dl_output),    // output [bs] I64
        TVMFFIAny::dltensor(&dl_valid),     // valid [bs] bool
        TVMFFIAny::none(),                  // maybe_indices (None)
        TVMFFIAny::none(),                  // maybe_top_k_arr (None = use scalar)
        TVMFFIAny::int64(top_k),            // top_k_val
        TVMFFIAny::none(),                  // maybe_top_p_arr (None = use scalar)
        TVMFFIAny::float64(top_p),          // top_p_val
        TVMFFIAny::bool_val(deterministic), // deterministic
        TVMFFIAny::none(),                  // maybe_seed_arr (None)
        TVMFFIAny::int64(42),               // seed_val
        TVMFFIAny::none(),                  // maybe_offset_arr (None)
        TVMFFIAny::int64(0),                // offset_val
    ];

    unsafe {
        tvm_static_ffi::call_tvm_ffi(sample_fn, &sample_args)
            .map_err(|e| candle_core::Error::Msg(format!("top_k_top_p_sampling: {e}")))?;
    }

    output.to_dtype(DType::U32)
}
