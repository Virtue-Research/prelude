use candle_core::{DType, Device, Result, Tensor};
use std::ffi::c_void;
use super::common::*;

// ── Correctness verification ─────────────────────────────────────────
// Compares all GPU GEMM backends against CPU F32 naive matmul.

pub fn verify(filter: &[String], device: &Device, _cublas: Option<&CublasHandle>) -> Result<()> {
    let has_cublas = cfg!(feature = "bench-cublas");
    let has_cutlass = cfg!(feature = "cutlass-gemm");

    // Test a representative subset of shapes
    let test_tokens = vec![(1, "M=1"), (16, "M=16"), (128, "M=128"), (512, "M=512")];
    let mut total = 0;
    let mut passed = 0;

    for model in &models() {
        if !filter.is_empty() && !filter.iter().any(|f| model.name.contains(f.as_str())) {
            continue;
        }

        println!("━━━ {} ━━━", model.name);
        let (h, i, v) = (model.hidden, model.intermediate, model.vocab);
        let layers: Vec<(&str, usize, usize)> = vec![
            ("qkvo", h, h), ("gate/up", i, h), ("down", h, i), ("lm_head", v, h),
        ];

        for (m, m_label) in &test_tokens {
            for (layer_label, n, k) in &layers {
                if *layer_label == "lm_head" && *m > 16 { continue; }

                let input_gpu = Tensor::randn(0f32, 0.02, (*m, *k), &Device::Cpu)?
                    .to_dtype(DType::BF16)?.to_device(device)?;
                let weight_gpu = Tensor::randn(0f32, 0.02, (*n, *k), &Device::Cpu)?
                    .to_dtype(DType::BF16)?.to_device(device)?;

                // CPU F64 reference — true ground truth (no GPU rounding)
                let ref_f64 = cpu_f64_reference(&input_gpu, &weight_gpu)?;
                let ref_f32 = ref_f64.to_dtype(DType::F32)?;
                let weight_t_gpu = weight_gpu.t()?;

                // Collect results from all backends
                let mut results: Vec<(&str, Tensor)> = Vec::new();

                // dispatch path (DeepGEMM → CUTLASS)
                let dispatch_out = input_gpu.matmul(&weight_t_gpu)?;
                results.push(("dispatch", dispatch_out));

                // CUTLASS direct (SM90)
                #[cfg(feature = "cutlass-gemm")]
                {
                    let out = run_cutlass(&input_gpu, &weight_gpu, device)?;
                    results.push(("cutlass", out));
                }

                // SM80 configs
                #[cfg(feature = "cutlass-gemm")]
                for (cfg_id, cfg_name) in [(0, "sm80:32s4"), (1, "sm80:64s3"), (2, "sm80:64s4")] {
                    let out = run_cutlass_sm80(&input_gpu, &weight_gpu, device, cfg_id)?;
                    results.push((cfg_name, out));
                }

                // cuBLAS
                if let Some(cb) = _cublas {
                    let out = run_cublas(cb, &input_gpu, &weight_gpu, device)?;
                    results.push(("cublas", out));
                }

                // Verify each against reference
                // BF16 GEMM tolerance: max_abs < 1.0 (BF16 has ~0.8% relative error per mul-add,
                // accumulated over K=1024-30720 gives significant absolute error)
                let threshold = 1.0f32;
                let mut all_ok = true;

                print!("  {m_label:<6} {layer_label:<8}");
                for (name, out) in &results {
                    let (max_abs, mean_abs) = check_against_ref(&ref_f32, out)?;
                    let ok = max_abs < threshold;
                    if !ok { all_ok = false; }
                    print!("  {name}={:.3}{}", max_abs, if ok { "" } else { " FAIL" });
                }

                total += 1;
                if all_ok { passed += 1; }
                println!("  {}", if all_ok { "OK" } else { "FAIL" });
            }
        }
        println!();
    }

    println!("Correctness: {passed}/{total} passed (threshold: max_abs < 1.0 vs CPU F64)");
    if passed < total {
        println!("WARNING: {} shapes failed correctness check!", total - passed);
    }
    Ok(())
}

// ── Helper: run a single GEMM and return output tensor ───────────────

#[cfg(feature = "cutlass-gemm")]
fn run_cutlass(input: &Tensor, weight: &Tensor, device: &Device) -> Result<Tensor> {
    let m = input.dim(0)?;
    let k = input.dim(1)?;
    let n = weight.dim(0)?;
    let cuda_dev = match device { Device::Cuda(d) => d, _ => unreachable!() };
    let stream = cuda_dev.cuda_stream();
    let output = Tensor::zeros((m, n), DType::BF16, device)?;
    let input_ptr = get_raw_ptr(input, &stream)?;
    let weight_ptr = get_raw_ptr(weight, &stream)?;
    let output_ptr = get_raw_ptr(&output, &stream)?;
    let raw_stream = unsafe { stream.cu_stream() };
    let (n_cub, m_cub, k_cub) = (n as i32, m as i32, k as i32);
    unsafe {
        prelude_cutlass_gemm::gemm_dispatch(
            weight_ptr as *const c_void, input_ptr as *const c_void,
            output_ptr as *mut c_void,
            n_cub, m_cub, k_cub, 1, k_cub, k_cub, n_cub, 0, 0, 0,
            true, false, 0, raw_stream as *const c_void,
        ).map_err(|e| candle_core::Error::Msg(e))?;
    }
    device.synchronize()?;
    Ok(output)
}

#[cfg(feature = "cutlass-gemm")]
fn run_cutlass_sm80(input: &Tensor, weight: &Tensor, device: &Device, config: i32) -> Result<Tensor> {
    let m = input.dim(0)?;
    let k = input.dim(1)?;
    let n = weight.dim(0)?;
    let cuda_dev = match device { Device::Cuda(d) => d, _ => unreachable!() };
    let stream = cuda_dev.cuda_stream();
    let output = Tensor::zeros((m, n), DType::BF16, device)?;
    let input_ptr = get_raw_ptr(input, &stream)?;
    let weight_ptr = get_raw_ptr(weight, &stream)?;
    let output_ptr = get_raw_ptr(&output, &stream)?;
    let raw_stream = unsafe { stream.cu_stream() };
    let (n_cub, m_cub, k_cub) = (n as i32, m as i32, k as i32);
    unsafe {
        prelude_cutlass_gemm::gemm_sm80(
            weight_ptr as *const c_void, input_ptr as *const c_void,
            output_ptr as *mut c_void,
            n_cub, m_cub, k_cub, 0, config, raw_stream as *const c_void,
        ).map_err(|e| candle_core::Error::Msg(e))?;
    }
    device.synchronize()?;
    Ok(output)
}

#[cfg(feature = "bench-cublas")]
fn run_cublas(cb: &CublasHandle, input: &Tensor, weight: &Tensor, device: &Device) -> Result<Tensor> {
    use cudarc_bench::cublas::sys;
    let m = input.dim(0)?;
    let k = input.dim(1)?;
    let n = weight.dim(0)?;
    let cuda_dev = match device { Device::Cuda(d) => d, _ => unreachable!() };
    let stream = cuda_dev.cuda_stream();
    let output = Tensor::zeros((m, n), DType::BF16, device)?;
    let input_ptr = get_raw_ptr(input, &stream)?;
    let weight_ptr = get_raw_ptr(weight, &stream)?;
    let output_ptr = get_raw_ptr(&output, &stream)?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        cudarc_bench::cublas::result::gemm_strided_batched_ex(
            cb.handle,
            sys::cublasOperation_t::CUBLAS_OP_T, sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            (&alpha) as *const f32 as *const _,
            weight_ptr as *const _, sys::cudaDataType_t::CUDA_R_16BF, k as i32, (n * k) as i64,
            input_ptr as *const _, sys::cudaDataType_t::CUDA_R_16BF, k as i32, (m * k) as i64,
            (&beta) as *const f32 as *const _,
            output_ptr as *mut _, sys::cudaDataType_t::CUDA_R_16BF, n as i32, (m * n) as i64,
            1, sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        ).map_err(|e| candle_core::Error::Msg(format!("cublas: {e:?}")))?;
    }
    device.synchronize()?;
    Ok(output)
}

// ── Performance benchmark ────────────────────────────────────────────

pub fn bench(filter: &[String], device: &Device, _cublas: Option<&CublasHandle>) -> Result<()> {
    let has_cublas = cfg!(feature = "bench-cublas");
    let has_cutlass = cfg!(feature = "cutlass-gemm");

    for model in &models() {
        if !filter.is_empty() && !filter.iter().any(|f| model.name.contains(f.as_str())) {
            continue;
        }

        println!("━━━ {} (H={}, I={}, V={}) ━━━", model.name, model.hidden, model.intermediate, model.vocab);
        let sm80_names = ["32s4", "64s3", "64s4", "64s3U", "256x", "64s5"];
        print!("{:<12} {:<8} {:>9}", "tokens", "layer", "dispatch");
        if has_cutlass {
            print!(" {:>9}", "cutlass");
            for name in &sm80_names { print!(" {:>7}", name); }
        }
        if has_cublas { print!(" {:>9}", "cublas"); }
        print!(" {:>7}", "TFLOPS");
        if has_cublas { print!(" {:>6}", "d/cub"); }
        if has_cutlass && has_cublas { print!(" {:>6} {:>6}", "c/cub", "80/cub"); }
        println!();
        println!("{}", "─".repeat(140));

        let (h, i, v) = (model.hidden, model.intermediate, model.vocab);
        let layers: Vec<(&str, usize, usize)> = vec![
            ("qkvo", h, h), ("gate/up", i, h), ("down", h, i), ("lm_head", v, h),
        ];

        for (tokens, tok_label) in &token_counts() {
            let m = *tokens;
            for (layer_label, n, k) in &layers {
                if *layer_label == "lm_head" && m > 64 { continue; }
                if *layer_label == "lm_head" && !["decode", "batch4"].contains(tok_label) { continue; }

                let input = Tensor::randn(0f32, 0.1, (m, *k), &Device::Cpu)?.to_dtype(DType::BF16)?.to_device(device)?;
                let weight = Tensor::randn(0f32, 0.1, (*n, *k), &Device::Cpu)?.to_dtype(DType::BF16)?.to_device(device)?;
                let weight_t = weight.t()?;

                let dispatch_us = bench_fn(|| input.matmul(&weight_t), device)?;

                #[cfg(feature = "cutlass-gemm")]
                let (cutlass_us, sm80_cfgs) = {
                    let c = bench_cutlass(&input, &weight, device)?;
                    let mut cfgs = Vec::new();
                    for cfg in 0..6 {
                        cfgs.push(bench_cutlass_sm80(&input, &weight, device, cfg)?);
                    }
                    (Some(c), Some(cfgs))
                };
                #[cfg(not(feature = "cutlass-gemm"))]
                let (cutlass_us, sm80_cfgs): (Option<f64>, Option<Vec<f64>>) = (None, None);

                let cublas_us = match _cublas {
                    Some(cb) => Some(bench_cublas(cb, &input, &weight, device)?),
                    None => None,
                };

                let flops = 2.0 * m as f64 * *n as f64 * *k as f64;
                let tflops = flops / (dispatch_us * 1e-6) / 1e12;

                print!("{tok_label:<12} {layer_label:<8} {dispatch_us:>9.1}");
                if let Some(c) = cutlass_us { print!(" {c:>9.1}"); }
                if let Some(cfgs) = &sm80_cfgs { for s in cfgs { print!(" {s:>7.1}"); } }
                if let Some(c) = cublas_us { print!(" {c:>9.1}"); }
                print!(" {tflops:>6.1}T");

                if let Some(cub) = cublas_us {
                    let r = dispatch_us / cub;
                    print!(" {:>5.2}x{:<2}", r, ratio_marker(r));
                }
                if let (Some(cut), Some(cub)) = (cutlass_us, cublas_us) {
                    let r = cut / cub;
                    print!(" {:>5.2}x{:<2}", r, ratio_marker(r));
                }
                // Best SM80 config vs cuBLAS
                if let (Some(cfgs), Some(cub)) = (&sm80_cfgs, cublas_us) {
                    let best = cfgs.iter().cloned().fold(f64::MAX, f64::min);
                    let r = best / cub;
                    print!(" {:>5.2}x{:<2}", r, ratio_marker(r));
                }
                println!();
            }
        }
        println!();
    }
    Ok(())
}

// ── CUTLASS SM90 direct (bypasses DeepGEMM) ─────────────────────────

#[cfg(feature = "cutlass-gemm")]
fn bench_cutlass(input: &Tensor, weight: &Tensor, device: &Device) -> Result<f64> {
    let m = input.dim(0)?;
    let k = input.dim(1)?;
    let n = weight.dim(0)?;
    let cuda_dev = match device { Device::Cuda(d) => d, _ => unreachable!() };
    let stream = cuda_dev.cuda_stream();
    let output = Tensor::zeros((m, n), DType::BF16, device)?;
    let input_ptr = get_raw_ptr(input, &stream)?;
    let weight_ptr = get_raw_ptr(weight, &stream)?;
    let output_ptr = get_raw_ptr(&output, &stream)?;
    let raw_stream = unsafe { stream.cu_stream() };

    let (n_cub, m_cub, k_cub) = (n as i32, m as i32, k as i32);

    bench_raw(|| unsafe {
        prelude_cutlass_gemm::gemm_dispatch(
            weight_ptr as *const c_void, input_ptr as *const c_void,
            output_ptr as *mut c_void,
            n_cub, m_cub, k_cub, 1,
            k_cub, k_cub, n_cub,
            0, 0, 0,
            true, false, 0,
            raw_stream as *const c_void,
        ).ok();
    }, device)
}

// ── CUTLASS SM80 forced (benchmarks universal fallback) ─────────────

#[cfg(feature = "cutlass-gemm")]
fn bench_cutlass_sm80(input: &Tensor, weight: &Tensor, device: &Device, config: i32) -> Result<f64> {
    let m = input.dim(0)?;
    let k = input.dim(1)?;
    let n = weight.dim(0)?;
    let cuda_dev = match device { Device::Cuda(d) => d, _ => unreachable!() };
    let stream = cuda_dev.cuda_stream();
    let output = Tensor::zeros((m, n), DType::BF16, device)?;
    let input_ptr = get_raw_ptr(input, &stream)?;
    let weight_ptr = get_raw_ptr(weight, &stream)?;
    let output_ptr = get_raw_ptr(&output, &stream)?;
    let raw_stream = unsafe { stream.cu_stream() };

    let (n_cub, m_cub, k_cub) = (n as i32, m as i32, k as i32);

    bench_raw(|| unsafe {
        prelude_cutlass_gemm::gemm_sm80(
            weight_ptr as *const c_void, input_ptr as *const c_void,
            output_ptr as *mut c_void,
            n_cub, m_cub, k_cub, 0, config,
            raw_stream as *const c_void,
        ).ok();
    }, device)
}

// ── cuBLAS direct (main-branch candle baseline) ─────────────────────

#[cfg(feature = "bench-cublas")]
fn bench_cublas(cb: &CublasHandle, input: &Tensor, weight: &Tensor, device: &Device) -> Result<f64> {
    use cudarc_bench::cublas::sys;

    let m = input.dim(0)?;
    let k = input.dim(1)?;
    let n = weight.dim(0)?;
    let cuda_dev = match device { Device::Cuda(d) => d, _ => unreachable!() };
    let stream = cuda_dev.cuda_stream();
    let output = Tensor::zeros((m, n), DType::BF16, device)?;
    let input_ptr = get_raw_ptr(input, &stream)?;
    let weight_ptr = get_raw_ptr(weight, &stream)?;
    let output_ptr = get_raw_ptr(&output, &stream)?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let handle = cb.handle;

    bench_raw(|| unsafe {
        cudarc_bench::cublas::result::gemm_strided_batched_ex(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_T, sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            (&alpha) as *const f32 as *const _,
            weight_ptr as *const _, sys::cudaDataType_t::CUDA_R_16BF, k as i32, (n * k) as i64,
            input_ptr as *const _, sys::cudaDataType_t::CUDA_R_16BF, k as i32, (m * k) as i64,
            (&beta) as *const f32 as *const _,
            output_ptr as *mut _, sys::cudaDataType_t::CUDA_R_16BF, n as i32, (m * n) as i64,
            1, sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        ).ok();
    }, device)
}

#[cfg(not(feature = "bench-cublas"))]
fn bench_cublas(_cb: &CublasHandle, _input: &Tensor, _weight: &Tensor, _device: &Device) -> Result<f64> {
    Ok(0.0)
}

#[cfg(not(feature = "bench-cublas"))]
fn run_cublas(_cb: &CublasHandle, _input: &Tensor, _weight: &Tensor, _device: &Device) -> Result<Tensor> {
    candle_core::bail!("bench-cublas feature not enabled")
}
