use candle_core::{DType, Device, Result, Tensor};
use candle_core::backend::BackendStorage;
use half::bf16;
use std::time::Instant;

// ── Correctness verification ─────────────────────────────────────────

/// Compute CPU F64 reference: cast BF16→F64 on CPU, matmul → F64 result.
/// This is the true ground truth (no GPU rounding, no TensorOp approximation).
pub fn cpu_f64_reference(input_bf16: &Tensor, weight_bf16: &Tensor) -> Result<Tensor> {
    let input_f64 = input_bf16.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
    let weight_f64 = weight_bf16.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
    input_f64.matmul(&weight_f64.t()?)
}

/// Compare a GPU BF16 result against a reference (CPU F64 cast to F32).
/// Returns (max_abs_error, mean_abs_error).
pub fn check_against_ref(reference_f32: &Tensor, gpu_result: &Tensor) -> Result<(f32, f32)> {
    let ref_flat: Vec<f32> = reference_f32.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let test_flat: Vec<f32> = gpu_result.to_dtype(DType::F32)?.to_device(&Device::Cpu)?
        .flatten_all()?.to_vec1()?;
    assert_eq!(ref_flat.len(), test_flat.len());

    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    for (r, t) in ref_flat.iter().zip(test_flat.iter()) {
        let err = (r - t).abs();
        max_abs = max_abs.max(err);
        sum_abs += err as f64;
    }
    let mean_abs = (sum_abs / ref_flat.len() as f64) as f32;
    Ok((max_abs, mean_abs))
}

// ── Model definitions ───────────────────────────────────────────────

pub struct ModelDims {
    pub name: &'static str,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
}

pub fn models() -> Vec<ModelDims> {
    vec![
        ModelDims { name: "Qwen3-0.6B",    hidden: 1024, intermediate: 3072,  num_heads: 16, num_kv_heads: 8,  head_dim: 64,  vocab: 151936 },
        ModelDims { name: "Qwen3-1.7B",    hidden: 2048, intermediate: 5632,  num_heads: 16, num_kv_heads: 8,  head_dim: 128, vocab: 151936 },
        ModelDims { name: "Qwen3-4B",      hidden: 2560, intermediate: 6912,  num_heads: 32, num_kv_heads: 8,  head_dim: 80,  vocab: 151936 },
        ModelDims { name: "Qwen3-8B",      hidden: 4096, intermediate: 11008, num_heads: 32, num_kv_heads: 8,  head_dim: 128, vocab: 151936 },
        ModelDims { name: "Qwen3-14B",     hidden: 5120, intermediate: 13696, num_heads: 40, num_kv_heads: 8,  head_dim: 128, vocab: 151936 },
        ModelDims { name: "Qwen3-32B",     hidden: 5120, intermediate: 25600, num_heads: 64, num_kv_heads: 8,  head_dim: 128, vocab: 151936 },
        ModelDims { name: "Qwen3-MoE-A3B", hidden: 2048, intermediate: 1408,  num_heads: 16, num_kv_heads: 8,  head_dim: 128, vocab: 151936 },
        ModelDims { name: "Gemma3-1B",     hidden: 2304, intermediate: 9216,  num_heads: 8,  num_kv_heads: 4,  head_dim: 256, vocab: 262144 },
        ModelDims { name: "Gemma3-4B",     hidden: 3072, intermediate: 11520, num_heads: 32, num_kv_heads: 16, head_dim: 128, vocab: 262144 },
        ModelDims { name: "Gemma3-27B",    hidden: 3840, intermediate: 30720, num_heads: 32, num_kv_heads: 16, head_dim: 128, vocab: 262144 },
    ]
}

pub fn token_counts() -> Vec<(usize, &'static str)> {
    vec![
        (1, "decode"), (4, "batch4"), (16, "batch16"), (64, "batch64"),
        (128, "prefill128"), (512, "prefill512"), (2048, "prefill2k"),
        (7, "unalign7"), (33, "unalign33"), (99, "unalign99"),
    ]
}

// ── Timing utilities ────────────────────────────────────────────────

pub const WARMUP: usize = 10;
pub const REPEATS: usize = 50;

/// Benchmark a function that returns a Tensor (e.g. Tensor::matmul)
pub fn bench_fn(f: impl Fn() -> Result<Tensor>, device: &Device) -> Result<f64> {
    for _ in 0..WARMUP { let _ = f()?; }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..REPEATS { let _ = f()?; }
    device.synchronize()?;
    Ok(start.elapsed().as_nanos() as f64 / REPEATS as f64 / 1000.0)
}

/// Benchmark a raw FFI call (no return value)
pub fn bench_raw(f: impl Fn(), device: &Device) -> Result<f64> {
    for _ in 0..WARMUP { f(); }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..REPEATS { f(); }
    device.synchronize()?;
    Ok(start.elapsed().as_nanos() as f64 / REPEATS as f64 / 1000.0)
}

// ── GPU pointer extraction ──────────────────────────────────────────

pub fn get_raw_ptr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> Result<u64> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<bf16>()?.slice(layout.start_offset()..);
    let (ptr, _guard) = unsafe { slice.device_ptr(stream) };
    Ok(ptr as u64)
}

// ── cuBLAS handle management ────────────────────────────────────────

pub struct CublasHandle {
    #[cfg(feature = "bench-cublas")]
    pub handle: cudarc_bench::cublas::sys::cublasHandle_t,
}

impl CublasHandle {
    #[cfg(feature = "bench-cublas")]
    pub fn new(device: &Device) -> Result<Self> {
        let cuda_dev = match device { Device::Cuda(d) => d, _ => candle_core::bail!("not CUDA") };
        let stream = cuda_dev.cuda_stream();
        let h = cudarc_bench::cublas::result::create_handle()
            .map_err(|e| candle_core::Error::Msg(format!("cublas create: {e:?}")))?;
        unsafe {
            cudarc_bench::cublas::result::set_stream(h, stream.cu_stream() as *mut _)
                .map_err(|e| candle_core::Error::Msg(format!("cublas set_stream: {e:?}")))?;
        }
        Ok(Self { handle: h })
    }

    #[cfg(not(feature = "bench-cublas"))]
    pub fn new(_device: &Device) -> Result<Self> {
        candle_core::bail!("bench-cublas feature not enabled")
    }
}

#[cfg(feature = "bench-cublas")]
impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe { cudarc_bench::cublas::result::destroy_handle(self.handle) }.ok();
    }
}

// ── Output formatting ───────────────────────────────────────────────

pub fn ratio_marker(r: f64) -> &'static str {
    if r <= 1.05 { "✓" }
    else if r <= 1.3 { "~" }
    else if r >= 5.0 { "✗✗" }
    else if r >= 2.0 { "✗" }
    else { "" }
}
