use candle_core::{DType, Device, Result, Tensor};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use half::bf16;
use std::ffi::c_void;
use std::time::Instant;

/// GPU GEMM benchmark: candle BF16 matmul (cuBLAS) vs DeepGEMM
pub fn bench() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        println!("No CUDA device available, skipping GPU GEMM bench");
        return Ok(());
    }

    let shapes: &[(usize, usize, usize)] = &[
        // Decode
        (1, 4096, 4096),
        (1, 11008, 4096),
        (4, 4096, 4096),
        (32, 4096, 4096),
        // Prefill
        (128, 4096, 4096),
        (128, 11008, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (512, 11008, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
    ];

    let warmup = 5;
    let repeats = 20;

    println!("{:<22} {:>12} {:>12} {:>8}", "Shape", "cuBLAS(us)", "DeepGEMM(us)", "Ratio");
    println!("{}", "-".repeat(58));

    for &(m, n, k) in shapes {
        let input = Tensor::randn(0f32, 0.1, (m, k), &device)?.to_dtype(DType::BF16)?;
        let weight = Tensor::randn(0f32, 0.1, (n, k), &device)?.to_dtype(DType::BF16)?;
        let weight_t = weight.t()?;

        // ── candle matmul (cuBLAS) ──
        for _ in 0..warmup {
            let _ = input.matmul(&weight_t)?;
        }
        device.synchronize()?;
        let start = Instant::now();
        for _ in 0..repeats {
            let _ = input.matmul(&weight_t)?;
        }
        device.synchronize()?;
        let cublas_us = start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0;

        // ── DeepGEMM ──
        #[cfg(feature = "deepgemm")]
        let deepgemm_us = {
            let cuda_dev = match &device {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = cuda_dev.cuda_stream();
            let raw_stream = unsafe { stream.cu_stream() as *mut c_void };

            let input_ptr = get_bf16_ptr(&input, &stream)?;
            let weight_ptr = get_bf16_ptr(&weight, &stream)?;
            let output = Tensor::zeros((m, n), DType::BF16, &device)?;
            let output_ptr = get_bf16_ptr(&output, &stream)?;

            for _ in 0..warmup {
                unsafe {
                    prelude_deepgemm::bf16_gemm(
                        input_ptr, weight_ptr, output_ptr as *mut _,
                        m as i32, n as i32, k as i32, raw_stream,
                    ).ok();
                }
            }
            device.synchronize()?;
            let start = Instant::now();
            for _ in 0..repeats {
                unsafe {
                    prelude_deepgemm::bf16_gemm(
                        input_ptr, weight_ptr, output_ptr as *mut _,
                        m as i32, n as i32, k as i32, raw_stream,
                    ).ok();
                }
            }
            device.synchronize()?;
            Some(start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
        };
        #[cfg(not(feature = "deepgemm"))]
        let deepgemm_us: Option<f64> = None;

        let label = format!("M={m:<5} N={n:<5} K={k}");
        if let Some(dg) = deepgemm_us {
            let ratio = dg / cublas_us;
            let marker = if ratio <= 1.05 { "✓" } else if ratio <= 1.2 { "~" } else { "" };
            println!("{label:<22} {cublas_us:>12.1} {dg:>12.1} {ratio:>7.2}x {marker}");
        } else {
            println!("{label:<22} {cublas_us:>12.1} {:>12} {:>8}", "N/A", "");
        }
    }

    Ok(())
}

fn get_bf16_ptr(
    t: &Tensor,
    stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> Result<*mut c_void> {
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<bf16>()?.slice(layout.start_offset()..);
    let (ptr, _guard) = unsafe { slice.device_ptr(stream) };
    Ok(ptr as u64 as *mut c_void)
}
