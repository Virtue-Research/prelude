use super::*;

pub(crate) fn candle_err(e: candle_core::Error) -> EngineError {
    EngineError::Internal(format!("candle error: {e}"))
}

pub(crate) fn init_cpu_runtime_if_needed(device: &Device, runtime: &crate::config::RuntimeConfig) {
    if !device.is_cpu() {
        let _ = runtime;
        return;
    }

    // Initialize oneDNN (for BF16 GEMM).
    // oneDNN uses THREADPOOL runtime (rayon-backed). No OpenMP, no contention.
    #[cfg(feature = "onednn")]
    {
        crate::ops::onednn::init();
        tracing::info!("oneDNN initialized (THREADPOOL runtime, rayon-backed)");
    }

    // Initialize NUMA-aware rayon pool for cpu_ops kernels
    {
        let numa_report = crate::ops::cpu::numa::init_numa_rayon_pool();
        tracing::info!(numa_report, "cpu_ops NUMA rayon pool initialized");
    }
}

pub(crate) fn select_device(
    runtime: &crate::config::RuntimeConfig,
) -> Result<(Device, DType), EngineError> {
    let requested = runtime.device.to_ascii_lowercase();

    let device = match requested.as_str() {
        "cpu" => Device::Cpu,
        "auto" => Device::cuda_if_available(0).map_err(candle_err)?,
        s if s.starts_with("cuda:") => {
            let ordinal = s
                .trim_start_matches("cuda:")
                .parse::<usize>()
                .map_err(|e| {
                    EngineError::InvalidRequest(format!("invalid PRELUDE_DEVICE: {e}"))
                })?;
            Device::new_cuda(ordinal).map_err(candle_err)?
        }
        "cuda" => Device::new_cuda(0).map_err(candle_err)?,
        other => {
            return Err(EngineError::InvalidRequest(format!(
                "invalid PRELUDE_DEVICE '{other}', expected auto|cpu|cuda|cuda:N"
            )))
        }
    };

    let dtype = match runtime.dtype.as_deref() {
        Some("f32" | "F32" | "float32") => DType::F32,
        Some("bf16" | "BF16" | "bfloat16") => DType::BF16,
        _ if device.is_cuda() => {
            if device.supports_bf16() { DType::BF16 } else { DType::F32 }
        }
        // CPU: auto-select BF16 when a BF16 GEMM backend is available.
        #[cfg(feature = "onednn")]
        _ => DType::BF16,
        #[cfg(not(feature = "onednn"))]
        _ => DType::F32,
    };
    info!(requested_device = %requested, is_cuda = device.is_cuda(), dtype = ?dtype, "selected device");
    Ok((device, dtype))
}
