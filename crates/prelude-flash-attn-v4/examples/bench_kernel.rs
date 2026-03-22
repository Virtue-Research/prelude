//! FA4 kernel microbenchmark.
//!
//! Measures per-kernel latency at various seq_len / head_dim / GQA configs.
//!
//! Usage:
//!   CUDA_VISIBLE_DEVICES=1 cargo run -p prelude-flash-attn-v4 --example bench_kernel --release
//!
//! Environment variables:
//!   FA4_BENCH_WARMUP=5     Number of warmup iterations
//!   FA4_BENCH_REPEATS=20   Number of timed iterations

use prelude_flash_attn_v4::{KernelKey, KernelRegistry};
use std::ffi::c_void;
use std::time::Instant;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

fn cuda_check(code: i32, msg: &str) {
    if code != 0 {
        let err = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(code)) };
        panic!("{msg}: CUDA error {code}: {}", err.to_string_lossy());
    }
}

struct GpuBuf(*mut c_void);
impl GpuBuf {
    fn alloc(bytes: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "malloc");
        cuda_check(unsafe { cudaMemset(ptr, 0, bytes) }, "memset");
        Self(ptr)
    }
}
impl Drop for GpuBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.0) };
    }
}

struct CudaEvent(*mut c_void);
impl CudaEvent {
    fn new() -> Self {
        let mut e = std::ptr::null_mut();
        cuda_check(unsafe { cudaEventCreate(&mut e) }, "event create");
        Self(e)
    }
    fn record(&self) {
        cuda_check(unsafe { cudaEventRecord(self.0, std::ptr::null_mut()) }, "event record");
    }
    fn sync(&self) {
        cuda_check(unsafe { cudaEventSynchronize(self.0) }, "event sync");
    }
    fn elapsed_ms(&self, start: &CudaEvent) -> f32 {
        let mut ms = 0.0f32;
        cuda_check(unsafe { cudaEventElapsedTime(&mut ms, start.0, self.0) }, "elapsed");
        ms
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0) };
    }
}

struct BenchConfig {
    name: &'static str,
    head_dim: usize,
    num_heads_q: usize,
    num_heads_k: usize,
}

fn main() {
    let warmup: usize = std::env::var("FA4_BENCH_WARMUP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let repeats: usize = std::env::var("FA4_BENCH_REPEATS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(20);

    let registry = KernelRegistry::new();

    // Model configs: Qwen3 family (GQA) + MHA baseline
    let configs = [
        BenchConfig { name: "MHA-h128",   head_dim: 128, num_heads_q: 8,  num_heads_k: 8 },
        BenchConfig { name: "MHA-h64",    head_dim: 64,  num_heads_q: 8,  num_heads_k: 8 },
        BenchConfig { name: "Qwen3-0.6B", head_dim: 128, num_heads_q: 16, num_heads_k: 8 },
        BenchConfig { name: "Qwen3-4B",   head_dim: 128, num_heads_q: 32, num_heads_k: 8 },
        BenchConfig { name: "Qwen3-32B",  head_dim: 128, num_heads_q: 64, num_heads_k: 8 },
    ];

    let seq_lens: &[usize] = &[128, 256, 512, 1024, 2048, 4096, 8192];

    // Print header
    println!(
        "{:<14} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Config", "SeqLen", "Med(us)", "Min(us)", "Max(us)", "Tok/s", "TFLOPS"
    );
    println!("{}", "-".repeat(74));

    for cfg in &configs {
        let gqa_ratio = cfg.num_heads_q / cfg.num_heads_k;
        let key = KernelKey::new(cfg.head_dim as u32, gqa_ratio as u32, true, false);
        let func = match registry.get(&key) {
            Some(f) => f,
            None => {
                println!("{:<14} SKIP (kernel not compiled: {:?})", cfg.name, key);
                continue;
            }
        };

        for &seq_len in seq_lens {
            let total_tokens = seq_len;
            let q_elems = total_tokens * cfg.num_heads_q * cfg.head_dim;
            let k_elems = total_tokens * cfg.num_heads_k * cfg.head_dim;

            let q_gpu = GpuBuf::alloc(q_elems * 2);
            let k_gpu = GpuBuf::alloc(k_elems * 2);
            let v_gpu = GpuBuf::alloc(k_elems * 2);
            let o_gpu = GpuBuf::alloc(q_elems * 2);
            let cu_gpu = GpuBuf::alloc(2 * 4);

            // Upload cu_seqlens
            let cu_data: [i32; 2] = [0, total_tokens as i32];
            cuda_check(
                unsafe { cudaMemcpy(cu_gpu.0, cu_data.as_ptr() as _, 8, 1) },
                "memcpy cu",
            );
            cuda_check(unsafe { cudaDeviceSynchronize() }, "sync init");

            let q_shape: [i64; 3] = [total_tokens as _, cfg.num_heads_q as _, cfg.head_dim as _];
            let k_shape: [i64; 3] = [total_tokens as _, cfg.num_heads_k as _, cfg.head_dim as _];
            let o_shape = q_shape;
            let lse_shape: [i64; 2] = [cfg.num_heads_q as _, total_tokens as _];
            let cu_shape: [i64; 1] = [2];
            let softmax_scale = 1.0 / (cfg.head_dim as f32).sqrt();

            let run_kernel = |_| {
                unsafe {
                    prelude_flash_attn_v4::fa4_varlen_fwd(
                        &registry, func,
                        q_gpu.0, k_gpu.0, v_gpu.0, o_gpu.0,
                        std::ptr::null_mut(),
                        softmax_scale,
                        std::ptr::null_mut(),
                        cu_gpu.0, cu_gpu.0,
                        &q_shape, &k_shape, &o_shape, &lse_shape, &cu_shape,
                        0, None, None,
                        None, None,
                    )
                    .expect("kernel failed");
                }
            };

            // Warmup
            for i in 0..warmup {
                run_kernel(i);
            }
            cuda_check(unsafe { cudaDeviceSynchronize() }, "sync warmup");

            // Timed runs using CUDA events for accurate GPU timing
            let mut times_us = Vec::with_capacity(repeats);
            for i in 0..repeats {
                let start_ev = CudaEvent::new();
                let end_ev = CudaEvent::new();

                start_ev.record();
                run_kernel(i);
                end_ev.record();
                end_ev.sync();

                let ms = end_ev.elapsed_ms(&start_ev);
                times_us.push((ms * 1000.0) as f64); // us
            }

            times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = times_us[times_us.len() / 2];
            let min = times_us[0];
            let max = times_us[times_us.len() - 1];

            // Throughput: tokens per second (using median latency)
            let tokens_per_sec = if median > 0.0 {
                total_tokens as f64 / (median / 1e6)
            } else {
                0.0
            };

            // TFLOPS: 2 * seq_len * seq_len * num_heads_q * head_dim / latency
            // (forward attention FLOPs approximation)
            let flops = 2.0 * (seq_len as f64).powi(2) * cfg.num_heads_q as f64
                * cfg.head_dim as f64;
            let tflops = if median > 0.0 {
                flops / (median / 1e6) / 1e12
            } else {
                0.0
            };

            println!(
                "{:<14} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10.2}",
                cfg.name, seq_len, median, min, max, tokens_per_sec, tflops
            );
        }
        println!();
    }
}
