//! FA4 kernel benchmark vs cuBLAS naive baseline.
//!
//! Run: cargo run -p prelude-flash-attn-v4 --example bench_kernel --release
//!
//! cuBLAS baseline = two cublasGemmStridedBatchedEx calls (Q@K^T + S@V).
//! No fused softmax, no causal mask skip. FA4 speedup grows with seq_len because:
//!   - cuBLAS needs O(seq^2) scratch for S matrix
//!   - cuBLAS doesn't fuse softmax (+30% overhead not counted)
//!   - cuBLAS computes full attention (FA4 skips half for causal)

use prelude_flash_attn_v4::{KernelKey, KernelRegistry};
use std::ffi::c_void;

// ── CUDA FFI ────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

// ── cuBLAS FFI ──────────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUDA_R_16BF: i32 = 14;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT: i32 = -1;

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> i32;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> i32;
    fn cublasGemmStridedBatchedEx(
        handle: cublasHandle_t,
        transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: *const c_void,
        a: *const c_void, a_type: i32, lda: i32, stride_a: i64,
        b: *const c_void, b_type: i32, ldb: i32, stride_b: i64,
        beta: *const c_void,
        c: *mut c_void, c_type: i32, ldc: i32, stride_c: i64,
        batch: i32,
        compute_type: i32, algo: i32,
    ) -> i32;
}

// ── Helpers ─────────────────────────────────────────────────────────

fn cuda_check(code: i32, msg: &str) {
    if code != 0 {
        let err = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(code)) };
        panic!("{msg}: CUDA error {code}: {}", err.to_string_lossy());
    }
}

fn gpu_alloc(bytes: usize) -> *mut c_void {
    let mut ptr = std::ptr::null_mut();
    cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "malloc");
    cuda_check(unsafe { cudaMemset(ptr, 0, bytes) }, "memset");
    ptr
}

/// Measure kernel time in ms using CUDA events.
fn cuda_bench<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f32 {
    for _ in 0..warmup { f(); }
    cuda_check(unsafe { cudaDeviceSynchronize() }, "sync warmup");

    let mut start: *mut c_void = std::ptr::null_mut();
    let mut end: *mut c_void = std::ptr::null_mut();
    unsafe {
        cudaEventCreate(&mut start);
        cudaEventCreate(&mut end);
        cudaEventRecord(start, std::ptr::null_mut());
    }
    for _ in 0..iters { f(); }
    unsafe {
        cudaEventRecord(end, std::ptr::null_mut());
        cudaEventSynchronize(end);
        let mut ms = 0.0f32;
        cudaEventElapsedTime(&mut ms, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        ms / iters as f32
    }
}

/// Naive attention via cuBLAS: S = Q @ K^T, O = S @ V (no softmax, no causal mask).
fn cublas_naive_attention(
    handle: cublasHandle_t,
    q: *const c_void, k: *const c_void, v: *const c_void,
    s: *mut c_void, o: *mut c_void,
    seq: i32, dim: i32, heads: i32,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let ap = &alpha as *const f32 as *const c_void;
    let bp = &beta as *const f32 as *const c_void;

    unsafe {
        // S[heads, seq, seq] = Q[heads, seq, dim] @ K[heads, seq, dim]^T
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq, seq, dim,
            ap,
            q, CUDA_R_16BF, dim, (seq * dim) as i64,
            k, CUDA_R_16BF, dim, (seq * dim) as i64,
            bp,
            s, CUDA_R_16BF, seq, (seq * seq) as i64,
            heads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
        );

        // O[heads, seq, dim] = S[heads, seq, seq] @ V[heads, seq, dim]
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, seq, seq,
            ap,
            v, CUDA_R_16BF, dim, (seq * dim) as i64,
            s, CUDA_R_16BF, seq, (seq * seq) as i64,
            bp,
            o, CUDA_R_16BF, dim, (seq * dim) as i64,
            heads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
        );
    }
}

// ── Config ──────────────────────────────────────────────────────────

struct BenchConfig {
    name: &'static str,
    head_dim: usize,
    num_heads_q: usize,
    num_heads_k: usize,
}

const WARMUP: usize = 5;
const ITERS: usize = 20;

// ── Bench: Non-paged varlen ─────────────────────────────────────────

fn bench_varlen(registry: &KernelRegistry, cublas_handle: cublasHandle_t) {
    println!("\n=== FA4 Varlen (causal, BF16) vs cuBLAS ===");
    println!("{:<14} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "Config", "SeqLen", "FA4(ms)", "cuBLAS*", "TFLOPS", "speedup");
    println!("{}", "-".repeat(66));

    let configs = [
        BenchConfig { name: "MHA-h128",   head_dim: 128, num_heads_q: 8,  num_heads_k: 8 },
        BenchConfig { name: "MHA-h64",    head_dim: 64,  num_heads_q: 8,  num_heads_k: 8 },
        BenchConfig { name: "Qwen3-0.6B", head_dim: 128, num_heads_q: 16, num_heads_k: 8 },
        BenchConfig { name: "Qwen3-4B",   head_dim: 128, num_heads_q: 32, num_heads_k: 8 },
        BenchConfig { name: "Qwen3-32B",  head_dim: 128, num_heads_q: 64, num_heads_k: 8 },
    ];

    let seq_lens: &[usize] = &[128, 256, 512, 1024, 2048, 4096, 8192];

    for cfg in &configs {
        let gqa_ratio = cfg.num_heads_q / cfg.num_heads_k;
        let key = KernelKey::new(cfg.head_dim as u32, gqa_ratio as u32, true, false);
        let func = match registry.get(&key) {
            Some(f) => f,
            None => {
                println!("{:<14} SKIP (kernel not compiled)", cfg.name);
                continue;
            }
        };

        for &seq_len in seq_lens {
            let q_elems = seq_len * cfg.num_heads_q * cfg.head_dim;
            let k_elems = seq_len * cfg.num_heads_k * cfg.head_dim;

            let q_gpu = gpu_alloc(q_elems * 2);
            let k_gpu = gpu_alloc(k_elems * 2);
            let v_gpu = gpu_alloc(k_elems * 2);
            let o_gpu = gpu_alloc(q_elems * 2);
            let cu_data: [i32; 2] = [0, seq_len as i32];
            let cu_gpu = gpu_alloc(8);
            cuda_check(
                unsafe { cudaMemcpy(cu_gpu, cu_data.as_ptr() as _, 8, 1) },
                "memcpy cu",
            );
            cuda_check(unsafe { cudaDeviceSynchronize() }, "sync init");

            let q_shape: [i64; 3] = [seq_len as _, cfg.num_heads_q as _, cfg.head_dim as _];
            let k_shape: [i64; 3] = [seq_len as _, cfg.num_heads_k as _, cfg.head_dim as _];
            let o_shape = q_shape;
            let lse_shape: [i64; 2] = [cfg.num_heads_q as _, seq_len as _];
            let cu_shape: [i64; 1] = [2];
            let softmax_scale = 1.0 / (cfg.head_dim as f32).sqrt();

            // FA4 kernel
            let fa4_ms = cuda_bench(WARMUP, ITERS, || {
                unsafe {
                    prelude_flash_attn_v4::fa4_varlen_fwd(
                        registry, func,
                        q_gpu, k_gpu, v_gpu, o_gpu,
                        std::ptr::null_mut(),
                        softmax_scale,
                        std::ptr::null_mut(),
                        cu_gpu, cu_gpu,
                        &q_shape, &k_shape, &o_shape, &lse_shape, &cu_shape,
                        0, None, None, None, None,
                    )
                    .expect("kernel failed");
                }
            });

            // cuBLAS baseline
            let s_size = cfg.num_heads_q * seq_len * seq_len;
            let s_buf = gpu_alloc(s_size * 2);
            let o_cub = gpu_alloc(q_elems * 2);
            let cub_ms = cuda_bench(WARMUP, ITERS, || {
                cublas_naive_attention(
                    cublas_handle,
                    q_gpu, k_gpu, v_gpu, s_buf, o_cub,
                    seq_len as i32, cfg.head_dim as i32, cfg.num_heads_q as i32,
                );
            });

            // TFLOPS: 2 * seq^2 * num_heads_q * head_dim / latency (forward attention)
            let flops = 2.0 * (seq_len as f64).powi(2) * cfg.num_heads_q as f64 * cfg.head_dim as f64;
            let tflops = flops / (fa4_ms as f64 / 1e3) / 1e12;

            let speedup = cub_ms / fa4_ms;
            println!(
                "{:<14} {:>8} {:>9.3}ms {:>9.3}ms {:>10.2} {:>7.1}x",
                cfg.name, seq_len, fa4_ms, cub_ms, tflops, speedup
            );

            unsafe {
                cudaFree(q_gpu); cudaFree(k_gpu); cudaFree(v_gpu);
                cudaFree(o_gpu); cudaFree(cu_gpu);
                cudaFree(s_buf); cudaFree(o_cub);
            }
        }
        println!();
    }
}

// ── Bench: Paged KV ─────────────────────────────────────────────────

fn bench_paged(registry: &KernelRegistry) {
    println!("\n=== FA4 Paged KV (causal, BF16) ===");
    println!("{:<14} {:>8} {:>8} {:>10} {:>10}",
        "Config", "Q-len", "KV-len", "FA4(ms)", "TFLOPS");
    println!("{}", "-".repeat(56));

    let configs = [
        BenchConfig { name: "Qwen3-4B",  head_dim: 128, num_heads_q: 32, num_heads_k: 8 },
        BenchConfig { name: "Qwen3-32B", head_dim: 128, num_heads_q: 64, num_heads_k: 8 },
    ];

    // Prefill: Q tokens > 1, paged KV cache
    let prefill_cases: &[(usize, usize)] = &[
        (128, 512), (128, 2048), (256, 2048), (256, 4096), (512, 4096),
    ];

    for cfg in &configs {
        let gqa_ratio = cfg.num_heads_q / cfg.num_heads_k;
        let key = KernelKey::new(cfg.head_dim as u32, gqa_ratio as u32, true, false)
            .with_paged(true);
        let func = match registry.get(&key) {
            Some(f) => f,
            None => {
                println!("{:<14} SKIP (paged kernel not compiled)", cfg.name);
                continue;
            }
        };

        let block_size: usize = 128;

        for &(q_len, kv_len) in prefill_cases {
            let q_elems = q_len * cfg.num_heads_q * cfg.head_dim;

            let num_blocks = (kv_len + block_size - 1) / block_size;
            let cache_elems = num_blocks * block_size * cfg.num_heads_k * cfg.head_dim;

            let q_gpu = gpu_alloc(q_elems * 2);
            let k_gpu = gpu_alloc(cache_elems * 2);
            let v_gpu = gpu_alloc(cache_elems * 2);
            let o_gpu = gpu_alloc(q_elems * 2);

            let cu_seqlens_q: [i32; 2] = [0, q_len as i32];
            let seqused_k: [i32; 1] = [kv_len as i32];
            let page_table: Vec<i32> = (0..num_blocks as i32).collect();

            let cu_q_gpu = gpu_alloc(8);
            let sk_gpu = gpu_alloc(4);
            let pt_gpu = gpu_alloc(num_blocks * 4);

            unsafe {
                cudaMemcpy(cu_q_gpu, cu_seqlens_q.as_ptr() as _, 8, 1);
                cudaMemcpy(sk_gpu, seqused_k.as_ptr() as _, 4, 1);
                cudaMemcpy(pt_gpu, page_table.as_ptr() as _, num_blocks * 4, 1);
                cudaDeviceSynchronize();
            }

            let q_shape: [i64; 3] = [q_len as _, cfg.num_heads_q as _, cfg.head_dim as _];
            let k_shape: [i64; 4] = [num_blocks as _, block_size as _, cfg.num_heads_k as _, cfg.head_dim as _];
            let o_shape: [i64; 3] = q_shape;
            let lse_shape: [i64; 2] = [cfg.num_heads_q as _, q_len as _];
            let cu_q_shape: [i64; 1] = [2];
            let sk_shape: [i64; 1] = [1];
            let pt_shape: [i64; 2] = [1, num_blocks as _];
            let softmax_scale = 1.0 / (cfg.head_dim as f32).sqrt();

            registry.set_stream(0, std::ptr::null_mut());

            let fa4_ms = cuda_bench(WARMUP, ITERS, || {
                unsafe {
                    prelude_flash_attn_v4::fa4_varlen_paged_fwd(
                        registry, func,
                        q_gpu, k_gpu, v_gpu, o_gpu,
                        std::ptr::null_mut(),
                        softmax_scale,
                        std::ptr::null_mut(),
                        cu_q_gpu, sk_gpu, pt_gpu,
                        &q_shape, &k_shape, &o_shape, &lse_shape,
                        &cu_q_shape, &sk_shape, &pt_shape,
                        0, None, None,
                    )
                    .expect("paged kernel failed");
                }
            });

            // TFLOPS: 2 * q_len * kv_len * num_heads_q * head_dim / latency
            let flops = 2.0 * q_len as f64 * kv_len as f64 * cfg.num_heads_q as f64 * cfg.head_dim as f64;
            let tflops = flops / (fa4_ms as f64 / 1e3) / 1e12;

            println!(
                "{:<14} {:>8} {:>8} {:>9.3}ms {:>10.2}",
                cfg.name, q_len, kv_len, fa4_ms, tflops
            );

            unsafe {
                cudaFree(q_gpu); cudaFree(k_gpu); cudaFree(v_gpu);
                cudaFree(o_gpu); cudaFree(cu_q_gpu); cudaFree(sk_gpu); cudaFree(pt_gpu);
            }
        }
        println!();
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let registry = KernelRegistry::new();
    println!("FA4 Benchmark — SM{}", registry.arch());

    let mut cublas_handle: cublasHandle_t = std::ptr::null_mut();
    unsafe { assert_eq!(cublasCreate_v2(&mut cublas_handle), 0, "cublasCreate failed"); }

    bench_varlen(&registry, cublas_handle);
    bench_paged(&registry);

    unsafe { cublasDestroy_v2(cublas_handle); }
}
