use super::*;

pub(crate) fn tensor_err(e: impl std::fmt::Display) -> EngineError {
    EngineError::Internal(format!("tensor error: {e}"))
}

pub(crate) fn init_runtime(_device: &Device, _runtime: &crate::config::RuntimeConfig) {
    // Device-specific initialization (oneDNN, NUMA pool, GPU GEMM registration)
    // is handled by device crates (prelude-cpu, prelude-cuda) in their ops factories.
}

pub(crate) fn select_device(
    runtime: &crate::config::RuntimeConfig,
) -> Result<(Device, DType), EngineError> {
    let requested = runtime.device.to_ascii_lowercase();

    let device = match requested.as_str() {
        "cpu" => Device::Cpu,
        "auto" => Device::Cuda(0),
        s if s.starts_with("cuda:") => {
            let ordinal = s
                .trim_start_matches("cuda:")
                .parse::<usize>()
                .map_err(|e| {
                    EngineError::InvalidRequest(format!("invalid PRELUDE_DEVICE: {e}"))
                })?;
            Device::Cuda(ordinal)
        }
        "cuda" => Device::Cuda(0),
        other => {
            return Err(EngineError::InvalidRequest(format!(
                "invalid PRELUDE_DEVICE '{other}', expected auto|cpu|cuda|cuda:N"
            )))
        }
    };

    let dtype = match runtime.dtype.as_deref() {
        Some("f32" | "F32" | "float32") => DType::F32,
        Some("bf16" | "BF16" | "bfloat16") => DType::BF16,
        _ if device.is_cuda() => DType::BF16,
        // CPU: BF16 with oneDNN BRGeMM backend
        _ => DType::BF16,
    };
    info!(requested_device = %requested, is_cuda = device.is_cuda(), dtype = ?dtype, "selected device");
    Ok((device, dtype))
}
