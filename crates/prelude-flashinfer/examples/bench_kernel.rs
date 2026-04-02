//! Benchmark FlashInfer attention kernels vs cuBLAS naive baseline.
//!
//! Run: cargo run -p prelude-flashinfer --example bench_kernel --release
//!
//! cuBLAS baseline = two cublasGemmStridedBatchedEx calls (Q@K^T + S@V).
//! This is what candle does without FlashInfer: no fused softmax, no causal
//! mask skip. FlashInfer speedup grows with seq_len because:
//!   - cuBLAS needs O(seq^2) scratch for S matrix
//!   - cuBLAS doesn't fuse softmax (+30% overhead not counted)
//!   - cuBLAS computes full attention (FlashInfer skips half for causal)

use prelude_flashinfer::types::*;
use prelude_flashinfer::*;
use std::ffi::c_void;
use std::time::Instant;

// ── CUDA FFI ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

// ── cuBLAS via dlopen (no link-time dependency) ─────────────────────

#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUDA_R_16BF: i32 = 14;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT: i32 = -1;

type FnCreate = unsafe extern "C" fn(*mut cublasHandle_t) -> i32;
type FnDestroy = unsafe extern "C" fn(cublasHandle_t) -> i32;
type FnGemmEx = unsafe extern "C" fn(
    cublasHandle_t, i32, i32, i32, i32, i32,
    *const c_void,
    *const c_void, i32, i32, i64,
    *const c_void, i32, i32, i64,
    *const c_void,
    *mut c_void, i32, i32, i64,
    i32, i32, i32,
) -> i32;

struct CuBlas {
    create: FnCreate,
    destroy: FnDestroy,
    gemm: FnGemmEx,
}

impl CuBlas {
    fn load() -> Option<Self> {
        unsafe extern "C" {
            fn dlopen(filename: *const i8, flags: i32) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
        }
        unsafe {
            let lib = dlopen(b"libcublas.so\0".as_ptr() as _, 0x101); // RTLD_LAZY|RTLD_GLOBAL
            if lib.is_null() { return None; }
            Some(Self {
                create: std::mem::transmute(dlsym(lib, b"cublasCreate_v2\0".as_ptr() as _)),
                destroy: std::mem::transmute(dlsym(lib, b"cublasDestroy_v2\0".as_ptr() as _)),
                gemm: std::mem::transmute(dlsym(lib, b"cublasGemmStridedBatchedEx\0".as_ptr() as _)),
            })
        }
    }
}

const BF16_DT: DLDataType = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
const FP32_DT: DLDataType = DLDataType { code: KDLFLOAT, bits: 32, lanes: 1 };
const I32_DT: DLDataType = DLDataType { code: KDLINT, bits: 32, lanes: 1 };
const U8_DT: DLDataType = DLDataType { code: KDLUINT, bits: 8, lanes: 1 };
// FP8 E4M3FN: DLPack code=10 (kDLFloat8_e4m3fn), bits=8
const FP8_DT: DLDataType = DLDataType { code: 10, bits: 8, lanes: 1 };

fn strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn gpu_dl(data: *mut c_void, dtype: DLDataType, shape: &[i64], st: &[i64]) -> DLTensor {
    DLTensor {
        data, device: DLDevice { device_type: KDLCUDA, device_id: 0 },
        ndim: shape.len() as i32, dtype,
        shape: shape.as_ptr(), strides: st.as_ptr(), byte_offset: 0,
    }
}

fn cpu_dl(data: *mut c_void, dtype: DLDataType, shape: &[i64], st: &[i64]) -> DLTensor {
    DLTensor {
        data, device: DLDevice { device_type: KDLCPU, device_id: 0 },
        ndim: shape.len() as i32, dtype,
        shape: shape.as_ptr(), strides: st.as_ptr(), byte_offset: 0,
    }
}

fn gpu_alloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe { assert_eq!(cudaMalloc(&mut ptr, size), 0); cudaMemset(ptr, 0, size); }
    ptr
}

/// Measure kernel time in ms using CUDA events.
fn cuda_bench<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f32 {
    for _ in 0..warmup { f(); }
    unsafe { cudaDeviceSynchronize(); }

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

struct Workspace {
    float_ws: *mut c_void,
    int_ws: *mut c_void,
    pinned_ws: *mut c_void,
    float_size: usize,
    int_size: usize,
}

impl Workspace {
    fn new() -> Self {
        let fs = 384 * 1024 * 1024;
        let is = 16 * 1024 * 1024;
        let fw = gpu_alloc(fs);
        let iw = gpu_alloc(is);
        let layout = std::alloc::Layout::from_size_align(is, 64).unwrap();
        let pw = unsafe { std::alloc::alloc_zeroed(layout) as *mut c_void };
        Self { float_ws: fw, int_ws: iw, pinned_ws: pw, float_size: fs, int_size: is }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.float_ws);
            cudaFree(self.int_ws);
            let layout = std::alloc::Layout::from_size_align(self.int_size, 64).unwrap();
            std::alloc::dealloc(self.pinned_ws as *mut u8, layout);
        }
    }
}

// ── cuBLAS naive attention baseline ───────────────────────────────────
// This is what candle does without FlashInfer: two cublasGemmStridedBatchedEx
// calls for Q@K^T and S@V, with no fused softmax or causal masking.

/// Naive attention via cuBLAS: S = Q @ K^T, O = S @ V (no softmax, no causal mask).
/// Input layout: Q/K/V are [num_heads, seq_len, head_dim] contiguous (head-major).
/// Returns time in ms for the two GEMMs combined.
fn cublas_naive_attention(
    cublas: &CuBlas, handle: cublasHandle_t,
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
        // cuBLAS col-major: treat row-major [seq, dim] as col-major [dim, seq]
        // transa=T on Q → Q^T col-major = Q row-major [seq, dim]
        // transb=N on K → K col-major [dim, seq] = K^T row-major [seq, dim]
        // Result C col-major [seq, seq] = Q row @ K^T row ✓
        (cublas.gemm)(
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
        // transa=N on V col-major [dim, seq] → V^T in formula sense
        // transb=T on S col-major [seq, seq] → S^T = S (symmetric-ish for perf)
        // Result C col-major [dim, seq] = V_col @ S_col^T
        //   = V^T_row @ S_row → O^T in row-major, but FLOPS are the same
        (cublas.gemm)(
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

// ── Bench: Prefill ───────────────────────────────────────────────────

fn bench_prefill_backend(
    reg: &KernelRegistry, ws: &Workspace, cublas: &CuBlas, cublas_handle: cublasHandle_t,
    backend: Backend, label: &str,
) {
    println!("\n{:=<80}", format!("= Prefill {label} (causal, BF16) vs cuBLAS "));
    println!("{:<10} {:<6} {:<6} {:>10} {:>10} {:>8}",
        "seq_len", "heads", "hdim", label, "cuBLAS*", "speedup");

    let configs = [
        (512, 32, 128),
        (1024, 32, 128),
        (2048, 32, 128),
        (4096, 32, 128),
    ];

    for (seq_len, num_heads, head_dim) in configs {
        let key = PrefillKey {
            dtype: KernelDtype::BF16,
            head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
            sliding_window: false, logits_soft_cap: false, backend,
        };
        let variant = match reg.get_prefill(&key) {
            Some(v) => v,
            None => { println!("{:<10} {:<6} {:<6} (no variant)", seq_len, num_heads, head_dim); continue; }
        };

        let batch_size = 1i64;
        let num_kv_heads = num_heads;
        let total = (seq_len * num_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;

        let q = gpu_alloc(total * 2);
        let k = gpu_alloc(kv_total * 2);
        let v = gpu_alloc(kv_total * 2);
        let o = gpu_alloc(total * 2);

        let cu_q: [i32; 2] = [0, seq_len as i32];
        let cu_k: [i32; 2] = [0, seq_len as i32];
        let kvl: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_alloc(8);
        let cu_k_gpu = gpu_alloc(8);
        unsafe {
            cudaMemcpy(cu_q_gpu, cu_q.as_ptr() as *const c_void, 8, 1);
            cudaMemcpy(cu_k_gpu, cu_k.as_ptr() as *const c_void, 8, 1);
        }

        let fws_s = [ws.float_size as i64]; let fws_st = [1i64];
        let iws_s = [ws.int_size as i64]; let iws_st = [1i64];
        let cu_s = [2i64]; let cu_st = [1i64];
        let kvl_s = [1i64]; let kvl_st = [1i64];
        let q_s = [seq_len as i64, num_heads as i64, head_dim as i64]; let q_st = strides(&q_s);
        let k_s = [seq_len as i64, num_kv_heads as i64, head_dim as i64]; let k_st = strides(&k_s);

        let dl_fws = gpu_dl(ws.float_ws, U8_DT, &fws_s, &fws_st);
        let dl_iws = gpu_dl(ws.int_ws, U8_DT, &iws_s, &iws_st);
        let dl_pws = cpu_dl(ws.pinned_ws, U8_DT, &iws_s, &iws_st);
        let dl_cuq_cpu = cpu_dl(cu_q.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk_cpu = cpu_dl(cu_k.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
        let dl_q = gpu_dl(q, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o, BF16_DT, &q_s, &q_st);

        reg.set_stream(0, std::ptr::null_mut());

        // Plan
        let plan_args = if backend == Backend::FA2 {
            vec![
                TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
                TVMFFIAny::dltensor(&dl_pws),
                TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_cuk_cpu),
                TVMFFIAny::dltensor(&dl_kvl),
                TVMFFIAny::int64(seq_len as i64), TVMFFIAny::int64(batch_size),
                TVMFFIAny::int64(num_heads as i64), TVMFFIAny::int64(num_kv_heads as i64),
                TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
                TVMFFIAny::int64(head_dim as i64), TVMFFIAny::int64(head_dim as i64),
                TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
                TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
            ]
        } else {
            vec![
                TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
                TVMFFIAny::dltensor(&dl_pws),
                TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_cuk_cpu),
                TVMFFIAny::dltensor(&dl_kvl),
                TVMFFIAny::int64(seq_len as i64), TVMFFIAny::int64(batch_size),
                TVMFFIAny::int64(num_heads as i64), TVMFFIAny::int64(num_kv_heads as i64),
                TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
                TVMFFIAny::int64(head_dim as i64), TVMFFIAny::int64(head_dim as i64),
                TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
            ]
        };
        let plan_info = unsafe { reg.call(variant.plan, &plan_args).expect("plan failed") };
        unsafe { cudaDeviceSynchronize(); } // wait for plan GPU work

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = if backend == Backend::FA2 {
            vec![
                TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), plan_info,
                TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_cuk_gpu),
                TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(),
                TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                TVMFFIAny::bool_val(false),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale),
                TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4), TVMFFIAny::int64(0),
            ]
        } else {
            vec![
                TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), plan_info,
                TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_cuk_gpu),
                TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(),
                TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                TVMFFIAny::bool_val(false),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale), TVMFFIAny::float64(1.0),
                TVMFFIAny::int64(0),       // token_pos_in_items_len
            ]
        };

        let fi_ms = cuda_bench(3, 20, || {
            unsafe { reg.call(variant.ragged_run, &run_args).unwrap(); }
        });

        // cuBLAS baseline: two GEMMs (Q@K^T + S@V), no softmax, no causal mask
        let s_size = (num_heads * seq_len * seq_len) as usize;
        let s_buf = gpu_alloc(s_size * 2);
        let o_cub = gpu_alloc(total * 2);
        let cub_ms = cuda_bench(3, 20, || {
            cublas_naive_attention(
                cublas, cublas_handle,
                q, k, v, s_buf, o_cub,
                seq_len as i32, head_dim as i32, num_heads as i32,
            );
        });

        let speedup = cub_ms / fi_ms;
        println!("{:<10} {:<6} {:<6} {:>9.3}ms {:>9.3}ms {:>7.1}x",
            seq_len, num_heads, head_dim, fi_ms, cub_ms, speedup);

        unsafe {
            cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
            cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
            cudaFree(s_buf); cudaFree(o_cub);
        }
    }
}

fn bench_prefill(reg: &KernelRegistry, ws: &Workspace) {
    let cublas = CuBlas::load().expect("failed to dlopen libcublas.so");
    let mut cublas_handle: cublasHandle_t = std::ptr::null_mut();
    unsafe { assert_eq!((cublas.create)(&mut cublas_handle), 0, "cublasCreate failed"); }

    // FA2 (SM80+)
    bench_prefill_backend(reg, ws, &cublas, cublas_handle, Backend::FA2, "FA2");

    // FA3 (SM90+, Hopper TMA)
    if reg.arch() >= 90 {
        bench_prefill_backend(reg, ws, &cublas, cublas_handle, Backend::FA3, "FA3");
    }

    // FP8 E4M3 prefill (SM90+)
    if reg.arch() >= 90 {
        let key = FP8PrefillKey { head_dim: 128, sliding_window: false };
        if let Some(variant) = reg.get_fp8_prefill(&key) {
            println!("\n{:=<80}", "= Prefill FP8 E4M3 (causal, SM90) vs FA3 BF16 ");
            println!("{:<10} {:<6} {:<6} {:>10} {:>10} {:>8}",
                "seq_len", "heads", "hdim", "FP8", "FA3-BF16", "ratio");

            let fa3_key = PrefillKey {
                dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
                sliding_window: false, logits_soft_cap: false, backend: Backend::FA3,
            };
            let fa3_variant = reg.get_prefill(&fa3_key);

            for seq_len in [512i64, 1024, 2048, 4096] {
                let num_heads = 32i64;
                let head_dim = 128i64;
                let batch_size = 1i64;
                let total = (seq_len * num_heads * head_dim) as usize;

                // FP8 data: 1 byte per element (vs 2 for BF16)
                // Fill with small random FP8 values (non-zero needed for TMA descriptor)
                let fp8_data: Vec<u8> = (0..total).map(|i| ((i % 120) + 1) as u8).collect();
                let q_fp8 = gpu_alloc(total);
                let k_fp8 = gpu_alloc(total);
                let v_fp8 = gpu_alloc(total);
                let o_bf16 = gpu_alloc(total * 2);  // output is BF16
                unsafe {
                    cudaMemcpy(q_fp8, fp8_data.as_ptr() as *const c_void, total, 1);
                    cudaMemcpy(k_fp8, fp8_data.as_ptr() as *const c_void, total, 1);
                    cudaMemcpy(v_fp8, fp8_data.as_ptr() as *const c_void, total, 1);
                }

                let cu_q: [i32; 2] = [0, seq_len as i32];
                let cu_k: [i32; 2] = [0, seq_len as i32];
                let kvl: [i32; 1] = [seq_len as i32];
                let cu_q_gpu = gpu_alloc(8);
                let cu_k_gpu = gpu_alloc(8);
                unsafe {
                    cudaMemcpy(cu_q_gpu, cu_q.as_ptr() as *const c_void, 8, 1);
                    cudaMemcpy(cu_k_gpu, cu_k.as_ptr() as *const c_void, 8, 1);
                }

                let fws_s = [ws.float_size as i64]; let fws_st = [1i64];
                let iws_s = [ws.int_size as i64]; let iws_st = [1i64];
                let cu_s = [2i64]; let cu_st = [1i64];
                let kvl_s = [1i64]; let kvl_st = [1i64];
                let q_s = [seq_len, num_heads, head_dim]; let q_st = strides(&q_s);
                let o_s = q_s; let o_st = q_st.clone();

                let dl_fws = gpu_dl(ws.float_ws, U8_DT, &fws_s, &fws_st);
                let dl_iws = gpu_dl(ws.int_ws, U8_DT, &iws_s, &iws_st);
                let dl_pws = cpu_dl(ws.pinned_ws, U8_DT, &iws_s, &iws_st);
                let dl_cuq_cpu = cpu_dl(cu_q.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
                let dl_cuk_cpu = cpu_dl(cu_k.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
                let dl_kvl = cpu_dl(kvl.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
                let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
                let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
                let dl_q = gpu_dl(q_fp8, FP8_DT, &q_s, &q_st);
                let dl_k = gpu_dl(k_fp8, FP8_DT, &q_s, &q_st);
                let dl_v = gpu_dl(v_fp8, FP8_DT, &q_s, &q_st);
                let dl_o = gpu_dl(o_bf16, BF16_DT, &o_s, &o_st);

                reg.set_stream(0, std::ptr::null_mut());

                // Plan (same as FA3)
                let plan_args = vec![
                    TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
                    TVMFFIAny::dltensor(&dl_pws),
                    TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_cuk_cpu),
                    TVMFFIAny::dltensor(&dl_kvl),
                    TVMFFIAny::int64(seq_len), TVMFFIAny::int64(batch_size),
                    TVMFFIAny::int64(num_heads), TVMFFIAny::int64(num_heads),
                    TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
                    TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
                    TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
                ];
                let plan_info = unsafe { reg.call(variant.plan, &plan_args).expect("FP8 plan failed") };
                unsafe { cudaDeviceSynchronize(); }

                let sm_scale = 1.0 / (head_dim as f64).sqrt();
                // FP8 run args: same structure as FA3 but with scale params instead of softcap/scale_v
                let run_args = vec![
                    TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), plan_info,
                    TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                    TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_cuk_gpu),
                    TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(),
                    TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                    TVMFFIAny::bool_val(false),
                    TVMFFIAny::none(),           // maybe_scale_q
                    TVMFFIAny::none(),           // maybe_scale_k
                    TVMFFIAny::none(),           // maybe_scale_v
                    TVMFFIAny::float64(sm_scale), // sm_scale
                    TVMFFIAny::float64(1.0),     // scale_q_scalar
                    TVMFFIAny::float64(1.0),     // scale_k_scalar
                    TVMFFIAny::float64(1.0),     // scale_v_scalar
                ];

                let fp8_ms = cuda_bench(3, 20, || {
                    unsafe { reg.call(variant.ragged_run, &run_args).unwrap(); }
                });

                // Compare with FA3 BF16
                let fa3_ms = if let Some(ref fa3) = fa3_variant {
                    let q_bf = gpu_alloc(total * 2);
                    let k_bf = gpu_alloc(total * 2);
                    let v_bf = gpu_alloc(total * 2);
                    let o_bf = gpu_alloc(total * 2);
                    let dl_q2 = gpu_dl(q_bf, BF16_DT, &q_s, &q_st);
                    let dl_k2 = gpu_dl(k_bf, BF16_DT, &q_s, &q_st);
                    let dl_v2 = gpu_dl(v_bf, BF16_DT, &q_s, &q_st);
                    let dl_o2 = gpu_dl(o_bf, BF16_DT, &o_s, &o_st);

                    let plan2 = vec![
                        TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
                        TVMFFIAny::dltensor(&dl_pws),
                        TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_cuk_cpu),
                        TVMFFIAny::dltensor(&dl_kvl),
                        TVMFFIAny::int64(seq_len), TVMFFIAny::int64(batch_size),
                        TVMFFIAny::int64(num_heads), TVMFFIAny::int64(num_heads),
                        TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
                        TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
                        TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
                    ];
                    let pi2 = unsafe { reg.call(fa3.plan, &plan2).expect("FA3 plan failed") };
                    unsafe { cudaDeviceSynchronize(); }

                    let run2 = vec![
                        TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), pi2,
                        TVMFFIAny::dltensor(&dl_q2), TVMFFIAny::dltensor(&dl_k2), TVMFFIAny::dltensor(&dl_v2),
                        TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_cuk_gpu),
                        TVMFFIAny::dltensor(&dl_o2), TVMFFIAny::none(),
                        TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                        TVMFFIAny::bool_val(false),
                        TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                        TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale), TVMFFIAny::float64(1.0),
                        TVMFFIAny::int64(0), // token_pos_in_items_len
                    ];
                    let ms = cuda_bench(3, 20, || {
                        unsafe { reg.call(fa3.ragged_run, &run2).unwrap(); }
                    });
                    unsafe { cudaFree(q_bf); cudaFree(k_bf); cudaFree(v_bf); cudaFree(o_bf); }
                    ms
                } else {
                    0.0
                };

                let ratio = if fa3_ms > 0.0 { fa3_ms / fp8_ms } else { 0.0 };
                println!("{:<10} {:<6} {:<6} {:>9.3}ms {:>9.3}ms {:>7.1}x",
                    seq_len, num_heads, head_dim, fp8_ms, fa3_ms, ratio);

                unsafe {
                    cudaFree(q_fp8); cudaFree(k_fp8); cudaFree(v_fp8); cudaFree(o_bf16);
                    cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
                }
            }
        }
    }

    println!("\n  * cuBLAS = two GEMMs only (no softmax, no causal mask)");

    unsafe { (cublas.destroy)(cublas_handle); }
}

// ── Bench: Decode ────────────────────────────────────────────────────

fn bench_decode(reg: &KernelRegistry, ws: &Workspace) {
    println!("\n{:=<80}", "= Decode (paged, BF16) vs cuBLAS naive ");
    println!("{:<8} {:<8} {:<6} {:<6} {:>10} {:>10} {:>8}",
        "batch", "kv_len", "heads", "hdim", "FlashInfer", "cuBLAS", "speedup");

    let cublas = CuBlas::load().expect("failed to dlopen libcublas.so");
    let mut cublas_handle: cublasHandle_t = std::ptr::null_mut();
    unsafe { assert_eq!((cublas.create)(&mut cublas_handle), 0); }

    let configs = [
        (1, 512, 32, 128),
        (8, 512, 32, 128),
        (32, 512, 32, 128),
        (64, 512, 32, 128),
    ];

    for (batch_size, kv_len, num_heads, head_dim) in configs {
        let key = DecodeKey {
            dtype: KernelDtype::BF16,
            head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
            sliding_window: false, logits_soft_cap: false,
        };
        let variant = match reg.get_decode(&key) {
            Some(v) => v,
            None => { println!("  Skipping: no variant"); continue; }
        };

        let page_size = 16i64;
        let num_pages_per_seq = (kv_len as i64 + page_size - 1) / page_size;
        let total_pages = num_pages_per_seq * batch_size as i64;
        let num_kv_heads = num_heads as i64;

        let q_elems = (batch_size as i64 * num_heads as i64 * head_dim as i64) as usize;
        let q = gpu_alloc(q_elems * 2);
        let o = gpu_alloc(q_elems * 2);
        let kv_elems = (total_pages * page_size * num_kv_heads * head_dim as i64) as usize;
        let k_cache = gpu_alloc(kv_elems * 2);
        let v_cache = gpu_alloc(kv_elems * 2);

        let mut kv_indptr_cpu: Vec<i32> = vec![0];
        let mut kv_indices: Vec<i32> = Vec::new();
        for b in 0..batch_size as i32 {
            kv_indptr_cpu.push(kv_indptr_cpu.last().unwrap() + num_pages_per_seq as i32);
            for p in 0..num_pages_per_seq as i32 {
                kv_indices.push(b * num_pages_per_seq as i32 + p);
            }
        }
        let last_len = kv_len as i32 % page_size as i32;
        let kv_last: Vec<i32> = vec![if last_len == 0 { page_size as i32 } else { last_len }; batch_size];

        let kvi_gpu = gpu_alloc(kv_indices.len() * 4);
        let kv_indptr_gpu = gpu_alloc(kv_indptr_cpu.len() * 4);
        let kv_last_gpu = gpu_alloc(kv_last.len() * 4);
        unsafe {
            cudaMemcpy(kvi_gpu, kv_indices.as_ptr() as *const c_void, kv_indices.len() * 4, 1);
            cudaMemcpy(kv_indptr_gpu, kv_indptr_cpu.as_ptr() as *const c_void, kv_indptr_cpu.len() * 4, 1);
            cudaMemcpy(kv_last_gpu, kv_last.as_ptr() as *const c_void, kv_last.len() * 4, 1);
        }

        let fws_s = [ws.float_size as i64]; let fws_st = [1i64];
        let iws_s = [ws.int_size as i64]; let iws_st = [1i64];
        let dl_fws = gpu_dl(ws.float_ws, U8_DT, &fws_s, &fws_st);
        let dl_iws = gpu_dl(ws.int_ws, U8_DT, &iws_s, &iws_st);
        let dl_pws = cpu_dl(ws.pinned_ws, U8_DT, &iws_s, &iws_st);

        let indptr_s = [batch_size as i64 + 1]; let indptr_st = [1i64];
        let dl_indptr_cpu = cpu_dl(kv_indptr_cpu.as_ptr() as *mut c_void, I32_DT, &indptr_s, &indptr_st);

        let empty_s = [0i64]; let empty_st = [1i64];
        let dl_eq = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
        let dl_ek = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws), TVMFFIAny::dltensor(&dl_indptr_cpu),
            TVMFFIAny::int64(batch_size as i64), TVMFFIAny::int64(num_heads as i64),
            TVMFFIAny::int64(num_kv_heads), TVMFFIAny::int64(page_size),
            TVMFFIAny::bool_val(false), TVMFFIAny::int64(-1),
            TVMFFIAny::float64(0.0), TVMFFIAny::int64(head_dim as i64),
            TVMFFIAny::int64(head_dim as i64), TVMFFIAny::dltensor(&dl_eq),
            TVMFFIAny::dltensor(&dl_ek),
        ];
        let plan_info = unsafe { reg.call(variant.plan, &plan_args).expect("decode plan failed") };

        let q_s = [batch_size as i64, num_heads as i64, head_dim as i64]; let q_st = strides(&q_s);
        let kv_s = [total_pages, page_size, num_kv_heads, head_dim as i64]; let kv_st = strides(&kv_s);
        let kvi_s = [kv_indices.len() as i64]; let kvi_st = [1i64];
        let bs_s = [batch_size as i64]; let bs_st = [1i64];

        let dl_q = gpu_dl(q, BF16_DT, &q_s, &q_st);
        let dl_o = gpu_dl(o, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_cache, BF16_DT, &kv_s, &kv_st);
        let dl_v = gpu_dl(v_cache, BF16_DT, &kv_s, &kv_st);
        let dl_kvi = gpu_dl(kvi_gpu, I32_DT, &kvi_s, &kvi_st);
        let dl_kv_indptr = gpu_dl(kv_indptr_gpu, I32_DT, &indptr_s, &indptr_st);
        let dl_kv_last = gpu_dl(kv_last_gpu, I32_DT, &bs_s, &bs_st);

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_kv_indptr), TVMFFIAny::dltensor(&dl_kvi),
            TVMFFIAny::dltensor(&dl_kv_last),
            TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(),
            TVMFFIAny::int64(0), TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
            TVMFFIAny::none(), TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
        ];

        let fi_ms = cuda_bench(3, 50, || {
            unsafe { reg.call(variant.run, &run_args).unwrap(); }
        });

        // cuBLAS baseline: decode = batched GEMV
        // Q[batch*heads, 1, dim] @ K[batch*heads, kv_len, dim]^T → S[batch*heads, 1, kv_len]
        // S[batch*heads, 1, kv_len] @ V[batch*heads, kv_len, dim] → O[batch*heads, 1, dim]
        let bh = batch_size as i32 * num_heads as i32;
        let sq = 1i32; // single query token
        let kv = kv_len as i32;
        let d = head_dim as i32;
        let cub_q = gpu_alloc((bh as usize * d as usize) * 2);
        let cub_k = gpu_alloc((bh as usize * kv as usize * d as usize) * 2);
        let cub_v = gpu_alloc((bh as usize * kv as usize * d as usize) * 2);
        let cub_s = gpu_alloc((bh as usize * kv as usize) * 2);
        let cub_o = gpu_alloc((bh as usize * d as usize) * 2);

        let cub_ms = cuda_bench(3, 50, || {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let ap = &alpha as *const f32 as *const c_void;
            let bp = &beta as *const f32 as *const c_void;
            unsafe {
                // S[bh, 1, kv] = Q[bh, 1, d] @ K[bh, kv, d]^T
                (cublas.gemm)(
                    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    sq, kv, d, ap,
                    cub_q, CUDA_R_16BF, d, (sq * d) as i64,
                    cub_k, CUDA_R_16BF, d, (kv * d) as i64,
                    bp,
                    cub_s, CUDA_R_16BF, sq, (sq * kv) as i64,
                    bh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
                );
                // O[bh, 1, d] = S[bh, 1, kv] @ V[bh, kv, d]
                (cublas.gemm)(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    d, sq, kv, ap,
                    cub_v, CUDA_R_16BF, d, (kv * d) as i64,
                    cub_s, CUDA_R_16BF, sq, (sq * kv) as i64,
                    bp,
                    cub_o, CUDA_R_16BF, d, (sq * d) as i64,
                    bh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
                );
            }
        });

        let speedup = cub_ms / fi_ms;
        println!("{:<8} {:<8} {:<6} {:<6} {:>9.3}ms {:>9.3}ms {:>7.1}x",
            batch_size, kv_len, num_heads, head_dim, fi_ms, cub_ms, speedup);

        unsafe {
            cudaFree(q); cudaFree(o); cudaFree(k_cache); cudaFree(v_cache);
            cudaFree(kvi_gpu); cudaFree(kv_indptr_gpu); cudaFree(kv_last_gpu);
            cudaFree(cub_q); cudaFree(cub_k); cudaFree(cub_v); cudaFree(cub_s); cudaFree(cub_o);
        }
    }

    unsafe { (cublas.destroy)(cublas_handle); }
}

// ── Bench: Utility kernels ───────────────────────────────────────────

fn bench_utilities(reg: &KernelRegistry) {
    println!("\n{:=<80}", "= Utility Kernels ");
    // Reset any lingering CUDA errors from previous benchmarks
    unsafe { cudaDeviceSynchronize(); }

    // Softmax
    if let Some(softmax) = reg.get_utility("softmax") {
        let batch = 64i64;
        let vocab = 32000i64;
        let n = (batch * vocab) as usize;
        let logits = gpu_alloc(n * 2);  // BF16
        let out = gpu_alloc(n * 4);     // FP32
        let ws_buf = gpu_alloc(64 * 1024 * 1024);

        let ws_s = [64 * 1024 * 1024i64]; let ws_st = [1i64];
        let s = [batch, vocab]; let st = strides(&s);
        let out_s = s; let out_st = strides(&out_s);

        let dl_ws = gpu_dl(ws_buf, U8_DT, &ws_s, &ws_st);
        let dl_in = gpu_dl(logits, BF16_DT, &s, &st);
        let dl_out = gpu_dl(out, FP32_DT, &out_s, &out_st);

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        let args = [
            TVMFFIAny::dltensor(&dl_ws), TVMFFIAny::dltensor(&dl_in),
            TVMFFIAny::dltensor(&dl_out), TVMFFIAny::none(),
            TVMFFIAny::float64(1.0), TVMFFIAny::bool_val(false),
        ];

        let ms = cuda_bench(5, 100, || {
            unsafe { reg.call(softmax, &args).unwrap(); }
        });
        let gb = (n * 2 + n * 4) as f64 / 1e9;
        let bw = gb / (ms as f64 * 1e-3);
        println!("  softmax     ({batch}×{vocab}): {ms:.3} ms, {bw:.1} GB/s");

        unsafe { cudaFree(logits); cudaFree(out); cudaFree(ws_buf); }
    }

    // RMSNorm
    if let Some(rmsnorm) = reg.get_utility("rmsnorm") {
        let tokens = 2048i64;
        let hidden = 4096i64;
        let n = (tokens * hidden) as usize;
        let input = gpu_alloc(n * 2);
        let weight = gpu_alloc(hidden as usize * 2);
        let output = gpu_alloc(n * 2);

        let in_s = [tokens, hidden]; let in_st = strides(&in_s);
        let w_s = [hidden]; let w_st = [1i64];

        let dl_out = gpu_dl(output, BF16_DT, &in_s, &in_st);
        let dl_in = gpu_dl(input, BF16_DT, &in_s, &in_st);
        let dl_w = gpu_dl(weight, BF16_DT, &w_s, &w_st);

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        let args = [
            TVMFFIAny::dltensor(&dl_out), TVMFFIAny::dltensor(&dl_in),
            TVMFFIAny::dltensor(&dl_w), TVMFFIAny::float64(1e-6),
            TVMFFIAny::bool_val(false),
        ];

        let ms = cuda_bench(5, 100, || {
            unsafe { reg.call(rmsnorm, &args).unwrap(); }
        });
        let gb = (n * 2 * 2 + hidden as usize * 2) as f64 / 1e9;
        let bw = gb / (ms as f64 * 1e-3);
        println!("  rmsnorm     ({tokens}×{hidden}): {ms:.3} ms, {bw:.1} GB/s");

        unsafe { cudaFree(input); cudaFree(weight); cudaFree(output); }
    }

    // Fused Add RMSNorm
    if let Some(fused) = reg.get_utility("fused_add_rmsnorm") {
        let tokens = 2048i64;
        let hidden = 4096i64;
        let n = (tokens * hidden) as usize;
        let input = gpu_alloc(n * 2);
        let residual = gpu_alloc(n * 2);
        let weight = gpu_alloc(hidden as usize * 2);

        let in_s = [tokens, hidden]; let in_st = strides(&in_s);
        let w_s = [hidden]; let w_st = [1i64];

        let dl_in = gpu_dl(input, BF16_DT, &in_s, &in_st);
        let dl_res = gpu_dl(residual, BF16_DT, &in_s, &in_st);
        let dl_w = gpu_dl(weight, BF16_DT, &w_s, &w_st);

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        let args = [
            TVMFFIAny::dltensor(&dl_in), TVMFFIAny::dltensor(&dl_res),
            TVMFFIAny::dltensor(&dl_w), TVMFFIAny::float64(1e-6),
            TVMFFIAny::bool_val(false),
        ];

        let ms = cuda_bench(5, 100, || {
            unsafe { reg.call(fused, &args).unwrap(); }
        });
        let gb = (n * 2 * 3 + hidden as usize * 2) as f64 / 1e9; // read input+residual+weight, write input+residual
        let bw = gb / (ms as f64 * 1e-3);
        println!("  fused_add_rms ({tokens}×{hidden}): {ms:.3} ms, {bw:.1} GB/s");

        unsafe { cudaFree(input); cudaFree(residual); cudaFree(weight); }
    }

    // Activation fusions
    for (name, label) in [
        ("silu_and_mul", "silu*mul"),
        ("gelu_and_mul", "gelu*mul"),
        ("gelu_tanh_and_mul", "gelu_t*mul"),
    ] {
        if let Some(act_fn) = reg.get_utility(name) {
            let tokens = 2048i64;
            let hidden = 4096i64;
            let input_dim = hidden * 2; // input is [tokens, 2*hidden]
            let n_in = (tokens * input_dim) as usize;
            let n_out = (tokens * hidden) as usize;
            let input = gpu_alloc(n_in * 2);
            let output = gpu_alloc(n_out * 2);

            let in_s = [tokens, input_dim]; let in_st = strides(&in_s);
            let out_s = [tokens, hidden]; let out_st = strides(&out_s);

            let dl_in = gpu_dl(input, BF16_DT, &in_s, &in_st);
            let dl_out = gpu_dl(output, BF16_DT, &out_s, &out_st);

            unsafe { reg.set_stream(0, std::ptr::null_mut()); }

            let args = [
                TVMFFIAny::dltensor(&dl_out),
                TVMFFIAny::dltensor(&dl_in),
                TVMFFIAny::bool_val(false), // enable_pdl
            ];

            let ms = cuda_bench(5, 100, || {
                unsafe { reg.call(act_fn, &args).unwrap(); }
            });
            // Read 2*hidden BF16 per token, write hidden BF16 per token
            let gb = (n_in * 2 + n_out * 2) as f64 / 1e9;
            let bw = gb / (ms as f64 * 1e-3);
            println!("  {label:13} ({tokens}×{hidden}): {ms:.3} ms, {bw:.1} GB/s");

            unsafe { cudaFree(input); cudaFree(output); }
        }
    }

    // FP4 KV cache quantization/dequantization
    if let (Some(quant_fn), Some(dequant_fn)) = (
        reg.get_utility("nvfp4_kv_quant"),
        reg.get_utility("nvfp4_kv_dequant"),
    ) {
        let m = 2048i64;   // tokens
        let k = 4096i64;   // head_dim * num_kv_heads (e.g. 128 * 32)
        let n = (m * k) as usize;

        let input = gpu_alloc(n * 2);           // BF16
        let fp4_out = gpu_alloc(n / 2);          // packed FP4
        let scales = gpu_alloc((m * k / 16) as usize); // FP8 block scales
        let global_scale_val: f32 = 1.0;
        let gs = gpu_alloc(4);
        unsafe { cudaMemcpy(gs, &global_scale_val as *const f32 as *const c_void, 4, 1); }
        let output = gpu_alloc(n * 2);           // BF16 dequantized

        let in_s = [m, k]; let in_st = strides(&in_s);
        let fp4_s = [m, k / 2]; let fp4_st = strides(&fp4_s);
        let sc_s = [m, k / 16]; let sc_st = strides(&sc_s);
        let gs_s = [1i64]; let gs_st = [1i64];
        let out_s = [m, k]; let out_st = strides(&out_s);

        let dl_in = gpu_dl(input, BF16_DT, &in_s, &in_st);
        let dl_fp4 = gpu_dl(fp4_out, U8_DT, &fp4_s, &fp4_st);
        let dl_sc = gpu_dl(scales, U8_DT, &sc_s, &sc_st);
        let dl_gs = gpu_dl(gs, FP32_DT, &gs_s, &gs_st);
        let dl_out = gpu_dl(output, BF16_DT, &out_s, &out_st);

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        let quant_args = [
            TVMFFIAny::dltensor(&dl_in), TVMFFIAny::dltensor(&dl_gs),
            TVMFFIAny::dltensor(&dl_fp4), TVMFFIAny::dltensor(&dl_sc),
        ];
        let dequant_args = [
            TVMFFIAny::dltensor(&dl_fp4), TVMFFIAny::dltensor(&dl_sc),
            TVMFFIAny::dltensor(&dl_gs), TVMFFIAny::dltensor(&dl_out),
        ];

        let ms_q = cuda_bench(5, 100, || {
            unsafe { reg.call(quant_fn, &quant_args).unwrap(); }
        });
        let ms_dq = cuda_bench(5, 100, || {
            unsafe { reg.call(dequant_fn, &dequant_args).unwrap(); }
        });
        let gb_in = (n * 2) as f64 / 1e9;
        let gb_fp4 = (n / 2 + (m * k / 16) as usize) as f64 / 1e9;
        let bw_q = gb_in / (ms_q as f64 * 1e-3);
        let bw_dq = (gb_fp4 + gb_in) / (ms_dq as f64 * 1e-3);
        let ratio = (n * 2) as f64 / (n as f64 / 2.0 + (m * k / 16) as f64);
        println!("  fp4_quant    ({m}×{k}): {ms_q:.3} ms, {bw_q:.1} GB/s");
        println!("  fp4_dequant  ({m}×{k}): {ms_dq:.3} ms, {bw_dq:.1} GB/s, compression={ratio:.1}x");

        unsafe { cudaFree(input); cudaFree(fp4_out); cudaFree(scales); cudaFree(gs); cudaFree(output); }
    }

    // MoE routing (DeepSeek V3 fused top-k)
    if let Some(moe_fn) = reg.get_utility("NoAuxTc") {
        let num_tokens = 2048i64;
        let num_experts = 256i64;
        let topk = 8i64;
        let n_group = 8i64;
        let topk_group = 4i64;

        // scores: [tokens, experts] BF16, bias: [experts] BF16
        let scores = gpu_alloc((num_tokens * num_experts) as usize * 2);
        let bias = gpu_alloc(num_experts as usize * 2);
        let topk_values = gpu_alloc((num_tokens * topk) as usize * 2);
        let topk_indices = gpu_alloc((num_tokens * topk) as usize * 4); // int32

        let sc_s = [num_tokens, num_experts]; let sc_st = strides(&sc_s);
        let bi_s = [num_experts]; let bi_st = [1i64];
        let tv_s = [num_tokens, topk]; let tv_st = strides(&tv_s);
        let ti_s = tv_s; let ti_st = tv_st.clone();

        let dl_scores = gpu_dl(scores, BF16_DT, &sc_s, &sc_st);
        let dl_bias = gpu_dl(bias, BF16_DT, &bi_s, &bi_st);
        let dl_values = gpu_dl(topk_values, BF16_DT, &tv_s, &tv_st);
        let dl_indices = gpu_dl(topk_indices, I32_DT, &ti_s, &ti_st);

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        let args = [
            TVMFFIAny::dltensor(&dl_scores),
            TVMFFIAny::dltensor(&dl_bias),
            TVMFFIAny::int64(n_group),
            TVMFFIAny::int64(topk_group),
            TVMFFIAny::int64(topk),
            TVMFFIAny::float64(1.0),          // routed_scaling_factor
            TVMFFIAny::dltensor(&dl_values),
            TVMFFIAny::dltensor(&dl_indices),
            TVMFFIAny::bool_val(false),        // launch_with_pdl
        ];

        let ms = cuda_bench(5, 100, || {
            unsafe { reg.call(moe_fn, &args).unwrap(); }
        });
        let gb = ((num_tokens * num_experts) as usize * 2  // read scores
                 + num_experts as usize * 2                 // read bias
                 + (num_tokens * topk) as usize * (2 + 4)  // write values + indices
        ) as f64 / 1e9;
        let bw = gb / (ms as f64 * 1e-3);
        println!("  moe_routing   ({num_tokens} tokens, {num_experts} experts, top{topk}): {ms:.3} ms, {bw:.1} GB/s");

        unsafe { cudaFree(scores); cudaFree(bias); cudaFree(topk_values); cudaFree(topk_indices); }
    }
}

// ── New module benchmarks ─────────────────────────────────────────────

fn bench_gemm(reg: &KernelRegistry) {
    println!("\n{:=<80}", "= GEMM Kernels ");

    // BF16 GEMM (SM100+ via TGV GEMM module)
    if let Some(gemm) = reg.get_utility("bf16_gemm") {
        for &(m, n, k) in &[(1i64, 4096, 4096), (32, 4096, 4096), (128, 4096, 11008), (512, 4096, 11008)] {
            let a = gpu_alloc((m * k) as usize * 2);
            let b = gpu_alloc((k * n) as usize * 2);
            let d = gpu_alloc((m * n) as usize * 2);

            let a_s = [m, k]; let a_st = strides(&a_s);
            let b_s = [n, k]; let b_st = strides(&b_s);  // NT layout
            let d_s = [m, n]; let d_st = strides(&d_s);

            let dl_a = gpu_dl(a, BF16_DT, &a_s, &a_st);
            let dl_b = gpu_dl(b, BF16_DT, &b_s, &b_st);
            let dl_d = gpu_dl(d, BF16_DT, &d_s, &d_st);

            unsafe { reg.set_stream(0, std::ptr::null_mut()); }

            let args = [
                TVMFFIAny::dltensor(&dl_a), TVMFFIAny::dltensor(&dl_b),
                TVMFFIAny::dltensor(&dl_d), TVMFFIAny::int64(-1),
            ];

            let ms = cuda_bench(5, 50, || {
                unsafe { reg.call(gemm, &args).unwrap(); }
            });
            let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms as f64 * 1e-3) / 1e12;
            println!("  bf16_gemm   ({m}×{n}×{k}): {ms:.3} ms, {tflops:.1} TFLOPS");

            unsafe { cudaFree(a); cudaFree(b); cudaFree(d); }
        }
    }

    // TGV GEMM (low-latency decode)
    if let Some(gemm) = reg.get_utility("tgv_gemm") {
        for &(m, n, k) in &[(1i64, 4096, 4096), (1, 4096, 11008), (1, 11008, 4096)] {
            let a = gpu_alloc((m * k) as usize * 2);
            let b = gpu_alloc((k * n) as usize * 2);
            let d = gpu_alloc((m * n) as usize * 2);

            let a_s = [m, k]; let a_st = strides(&a_s);
            let b_s = [n, k]; let b_st = strides(&b_s);
            let d_s = [m, n]; let d_st = strides(&d_s);

            let dl_a = gpu_dl(a, BF16_DT, &a_s, &a_st);
            let dl_b = gpu_dl(b, BF16_DT, &b_s, &b_st);
            let dl_d = gpu_dl(d, BF16_DT, &d_s, &d_st);

            let args = [
                TVMFFIAny::dltensor(&dl_a), TVMFFIAny::dltensor(&dl_b),
                TVMFFIAny::dltensor(&dl_d), TVMFFIAny::int64(-1),
            ];

            let ms = cuda_bench(5, 100, || {
                unsafe { reg.call(gemm, &args).unwrap(); }
            });
            let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms as f64 * 1e-3) / 1e12;
            println!("  tgv_gemm    ({m}×{n}×{k}): {ms:.3} ms, {tflops:.1} TFLOPS");

            unsafe { cudaFree(a); cudaFree(b); cudaFree(d); }
        }
    } else {
        println!("  tgv_gemm: not compiled");
    }

}

fn bench_topk(reg: &KernelRegistry) {
    if let Some(topk) = reg.get_utility("radix_topk") {
        println!("\n{:=<80}", "= TopK ");
        for &(batch, vocab, k) in &[(64i64, 32000i64, 8i64), (256, 151936, 8)] {
            let n = (batch * vocab) as usize;
            let input = gpu_alloc(n * 4);  // FP32 scores
            let vals = gpu_alloc((batch * k) as usize * 4);
            let idxs = gpu_alloc((batch * k) as usize * 4);

            let in_s = [batch, vocab]; let in_st = strides(&in_s);
            let out_s = [batch, k]; let out_st = strides(&out_s);

            let dl_in = gpu_dl(input, FP32_DT, &in_s, &in_st);
            let dl_vals = gpu_dl(vals, FP32_DT, &out_s, &out_st);
            let dl_idxs = gpu_dl(idxs, I32_DT, &out_s, &out_st);

            unsafe { reg.set_stream(0, std::ptr::null_mut()); }

            let args = [
                TVMFFIAny::dltensor(&dl_in),
                TVMFFIAny::dltensor(&dl_vals),
                TVMFFIAny::dltensor(&dl_idxs),
            ];

            let ms = cuda_bench(5, 100, || {
                unsafe { reg.call(topk, &args).unwrap(); }
            });
            let gb = n as f64 * 4.0 / 1e9;
            let bw = gb / (ms as f64 * 1e-3);
            println!("  radix_topk  ({batch}×{vocab}, k={k}): {ms:.3} ms, {bw:.1} GB/s");

            unsafe { cudaFree(input); cudaFree(vals); cudaFree(idxs); }
        }
    }
}

// ── POD (Prefill-On-Decode) benchmark ──────────────────────────────────
// Compare POD single kernel launch vs separate paged prefill + decode.

fn bench_pod(reg: &KernelRegistry, ws: &Workspace) {
    let pod_key = PodKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: 128,
        head_dim_vo: 128,
    };
    let pod_variant = match reg.get_pod(&pod_key) {
        Some(v) => v,
        None => {
            println!("\n{:=<80}", "= POD Benchmark: SKIPPED (not compiled) ");
            return;
        }
    };
    let prefill_variant = reg.get_prefill(&PrefillKey {
        dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
        sliding_window: false, logits_soft_cap: false, backend: Backend::FA2,
    }).unwrap();
    let decode_variant = reg.get_decode(&DecodeKey {
        dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
        sliding_window: false, logits_soft_cap: false,
    }).unwrap();

    println!("\n{:=<80}", "= POD Benchmark (BF16 h128) ");
    println!("{:<50} {:>10} {:>10} {:>8}",
        "Config", "POD(µs)", "Sep(µs)", "Speedup");
    println!("{}", "-".repeat(82));

    // Configs: (num_qo_heads, num_kv_heads, decode_batch, prefill_q_len, kv_len_decode)
    // NOTE: FA2 paged prefill has a known issue with 32 heads + seq_len >= 64
    //       ("illegal instruction"), so we use 2 heads for larger configs.
    let configs = [
        (2, 2, 1, 8, 32,       "1 prefill(8) + 1 decode(kv32) [2h]"),
        (32, 8, 1, 8, 128,     "1 prefill(8) + 1 decode(kv128) [32h GQA]"),
        (2, 2, 1, 64, 512,     "1 prefill(64) + 1 decode(kv512) [2h]"),
        (2, 2, 1, 128, 2048,   "1 prefill(128) + 1 decode(kv2048) [2h]"),
        (2, 2, 16, 32, 512,    "1 prefill(32) + 16 decode(kv512) [2h]"),
        (2, 2, 64, 32, 1024,   "1 prefill(32) + 64 decode(kv1024) [2h]"),
        (2, 2, 128, 64, 2048,  "1 prefill(64) + 128 decode(kv2048) [2h]"),
    ];

    for (nqh, nkh, dec_batch, pq_len, kv_d, label) in configs {
        let num_qo_heads = nqh as i64;
        let num_kv_heads = nkh as i64;
        let head_dim = 128i64;
        let page_size = 16i64;

        // Prefill: 1 request, pq_len tokens of Q, kv = pq_len
        let kv_len_p = pq_len as i64;
        let num_pages_p = (kv_len_p + page_size - 1) / page_size;

        // Decode: dec_batch requests, Q=1 each, kv = kv_d
        let kv_len_d = kv_d as i64;
        let num_pages_d_per = (kv_len_d + page_size - 1) / page_size;
        let dec_batch = dec_batch as i64;
        let total_pages = num_pages_p + num_pages_d_per * dec_batch;

        unsafe {
            // Shared KV cache
            let kv_elems = (total_pages * page_size * num_kv_heads * head_dim) as usize;
            let k_cache = gpu_alloc(kv_elems * 2);
            let v_cache = gpu_alloc(kv_elems * 2);
            cudaMemset(k_cache, 1, kv_elems * 2);
            cudaMemset(v_cache, 1, kv_elems * 2);

            // Prefill Q/O
            let q_p_elems = (kv_len_p * num_qo_heads * head_dim) as usize;
            let q_p = gpu_alloc(q_p_elems * 2);
            let o_p = gpu_alloc(q_p_elems * 2);
            cudaMemset(q_p, 1, q_p_elems * 2);

            // Decode Q/O (batched)
            let q_d_elems = (dec_batch * num_qo_heads * head_dim) as usize;
            let q_d = gpu_alloc(q_d_elems * 2);
            let o_d = gpu_alloc(q_d_elems * 2);
            cudaMemset(q_d, 1, q_d_elems * 2);

            // Page tables for prefill (1 request)
            let qo_indptr_p: Vec<i32> = vec![0, kv_len_p as i32];
            let kv_indptr_p: Vec<i32> = vec![0, num_pages_p as i32];
            let kv_indices_p: Vec<i32> = (0..num_pages_p as i32).collect();
            let kv_last_p = vec![
                if kv_len_p % page_size == 0 { page_size as i32 } else { (kv_len_p % page_size) as i32 }
            ];

            // Page tables for decode (dec_batch requests)
            let mut qo_indptr_d: Vec<i32> = vec![0];
            let mut kv_indptr_d: Vec<i32> = vec![0];
            let mut kv_indices_d: Vec<i32> = Vec::new();
            let mut kv_last_d: Vec<i32> = Vec::new();
            for b in 0..dec_batch as i32 {
                qo_indptr_d.push(qo_indptr_d.last().unwrap() + 1);
                kv_indptr_d.push(kv_indptr_d.last().unwrap() + num_pages_d_per as i32);
                let base = num_pages_p as i32 + b * num_pages_d_per as i32;
                for p in 0..num_pages_d_per as i32 {
                    kv_indices_d.push(base + p);
                }
                let rem = kv_len_d % page_size;
                kv_last_d.push(if rem == 0 { page_size as i32 } else { rem as i32 });
            }

            // Upload page tables
            let qo_indptr_p_gpu = gpu_alloc(qo_indptr_p.len() * 4);
            let kv_indptr_p_gpu = gpu_alloc(kv_indptr_p.len() * 4);
            let kv_indices_p_gpu = gpu_alloc(kv_indices_p.len() * 4);
            let kv_last_p_gpu = gpu_alloc(kv_last_p.len() * 4);
            cudaMemcpy(qo_indptr_p_gpu, qo_indptr_p.as_ptr() as _, qo_indptr_p.len() * 4, 1);
            cudaMemcpy(kv_indptr_p_gpu, kv_indptr_p.as_ptr() as _, kv_indptr_p.len() * 4, 1);
            cudaMemcpy(kv_indices_p_gpu, kv_indices_p.as_ptr() as _, kv_indices_p.len() * 4, 1);
            cudaMemcpy(kv_last_p_gpu, kv_last_p.as_ptr() as _, kv_last_p.len() * 4, 1);

            let qo_indptr_d_gpu = gpu_alloc(qo_indptr_d.len() * 4);
            let kv_indptr_d_gpu = gpu_alloc(kv_indptr_d.len() * 4);
            let kv_indices_d_gpu = gpu_alloc(kv_indices_d.len() * 4);
            let kv_last_d_gpu = gpu_alloc(kv_last_d.len() * 4);
            cudaMemcpy(qo_indptr_d_gpu, qo_indptr_d.as_ptr() as _, qo_indptr_d.len() * 4, 1);
            cudaMemcpy(kv_indptr_d_gpu, kv_indptr_d.as_ptr() as _, kv_indptr_d.len() * 4, 1);
            cudaMemcpy(kv_indices_d_gpu, kv_indices_d.as_ptr() as _, kv_indices_d.len() * 4, 1);
            cudaMemcpy(kv_last_d_gpu, kv_last_d.as_ptr() as _, kv_last_d.len() * 4, 1);

            // SM-aware scheduling buffer
            let num_sm = 132i64;
            let sched = gpu_alloc((num_sm as usize + 2) * 4);
            cudaMemset(sched, 0, (num_sm as usize + 2) * 4);

            reg.set_stream(0, std::ptr::null_mut());

            let sm_scale = 1.0 / (head_dim as f64).sqrt();

            // POD needs SEPARATE workspace buffers for prefill and decode sides
            // (they run concurrently on different SMs, shared workspace = corruption)
            let fws_size = ws.float_size;
            let iws_size = ws.int_size;
            let fws_s = [fws_size as i64]; let fws_st = strides(&fws_s);
            let iws_s = [iws_size as i64]; let iws_st = strides(&iws_s);
            // Prefill side workspace
            let dl_fws_p = gpu_dl(ws.float_ws, U8_DT, &fws_s, &fws_st);
            let dl_iws_p = gpu_dl(ws.int_ws, U8_DT, &iws_s, &iws_st);
            let dl_pws = cpu_dl(ws.pinned_ws, U8_DT, &iws_s, &iws_st);
            // Decode side workspace (separate allocation)
            let fws_d_ptr = gpu_alloc(fws_size);
            let iws_d_ptr = gpu_alloc(iws_size);
            let dl_fws_d = gpu_dl(fws_d_ptr, U8_DT, &fws_s, &fws_st);
            let dl_iws_d = gpu_dl(iws_d_ptr, U8_DT, &iws_s, &iws_st);
            // Pinned workspace for decode-side plan
            let pws_layout = std::alloc::Layout::from_size_align(iws_size, 64).unwrap();
            let pws_d_ptr = std::alloc::alloc_zeroed(pws_layout) as *mut c_void;
            let dl_pws_d = cpu_dl(pws_d_ptr, U8_DT, &iws_s, &iws_st);
            let kv_s = [total_pages, page_size, num_kv_heads, head_dim];
            let kv_st = strides(&kv_s);
            let dl_k = gpu_dl(k_cache, BF16_DT, &kv_s, &kv_st);
            let dl_v = gpu_dl(v_cache, BF16_DT, &kv_s, &kv_st);
            let sched_s = [num_sm + 2];
            let sched_st = strides(&sched_s);
            let dl_sched = gpu_dl(sched, I32_DT, &sched_s, &sched_st);

            // Prefill tensors
            let qp_s = [kv_len_p, num_qo_heads, head_dim]; let qp_st = strides(&qp_s);
            let dl_qp = gpu_dl(q_p, BF16_DT, &qp_s, &qp_st);
            let dl_op = gpu_dl(o_p, BF16_DT, &qp_s, &qp_st);
            let mk = |ptr, len: usize| { let s = [len as i64]; let st = strides(&s); gpu_dl(ptr, I32_DT, &s, &st) };
            let dl_qo_indptr_p = mk(qo_indptr_p_gpu, qo_indptr_p.len());
            let dl_kv_indptr_p = mk(kv_indptr_p_gpu, kv_indptr_p.len());
            let dl_kv_indices_p = mk(kv_indices_p_gpu, kv_indices_p.len());
            let dl_kv_last_p = mk(kv_last_p_gpu, kv_last_p.len());

            // Decode tensors
            let qd_s = [dec_batch, num_qo_heads, head_dim]; let qd_st = strides(&qd_s);
            let dl_qd = gpu_dl(q_d, BF16_DT, &qd_s, &qd_st);
            let dl_od = gpu_dl(o_d, BF16_DT, &qd_s, &qd_st);
            let dl_qo_indptr_d = mk(qo_indptr_d_gpu, qo_indptr_d.len());
            let dl_kv_indptr_d = mk(kv_indptr_d_gpu, kv_indptr_d.len());
            let dl_kv_indices_d = mk(kv_indices_d_gpu, kv_indices_d.len());
            let dl_kv_last_d = mk(kv_last_d_gpu, kv_last_d.len());

            // ── Plans ──
            let cu_p_s = [2i64]; let cu_p_st = strides(&cu_p_s);
            let kvl_p_s = [1i64]; let kvl_p_st = strides(&kvl_p_s);
            let kvl_data_p = [kv_len_p as i32];
            let dl_cuq_p = cpu_dl(qo_indptr_p.as_ptr() as _, I32_DT, &cu_p_s, &cu_p_st);
            let dl_cuk_p = cpu_dl(kv_indptr_p.as_ptr() as _, I32_DT, &cu_p_s, &cu_p_st);
            let dl_kvl_p = cpu_dl(kvl_data_p.as_ptr() as _, I32_DT, &kvl_p_s, &kvl_p_st);
            let plan_p = reg.call(prefill_variant.plan, &[
                TVMFFIAny::dltensor(&dl_fws_p), TVMFFIAny::dltensor(&dl_iws_p), TVMFFIAny::dltensor(&dl_pws),
                TVMFFIAny::dltensor(&dl_cuq_p), TVMFFIAny::dltensor(&dl_cuk_p), TVMFFIAny::dltensor(&dl_kvl_p),
                TVMFFIAny::int64(kv_len_p), TVMFFIAny::int64(1),
                TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
                TVMFFIAny::int64(page_size), TVMFFIAny::bool_val(false),
                TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
                TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
                TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
            ]).unwrap();

            // Decode-side plan (prefill plan with Q=1)
            let cu_d_s = [(dec_batch + 1) as i64]; let cu_d_st = strides(&cu_d_s);
            let kvl_d_s = [dec_batch]; let kvl_d_st = strides(&kvl_d_s);
            let kvl_data_d: Vec<i32> = vec![kv_len_d as i32; dec_batch as usize];
            let dl_cuq_d = cpu_dl(qo_indptr_d.as_ptr() as _, I32_DT, &cu_d_s, &cu_d_st);
            let dl_cuk_d = cpu_dl(kv_indptr_d.as_ptr() as _, I32_DT, &cu_d_s, &cu_d_st);
            let dl_kvl_d = cpu_dl(kvl_data_d.as_ptr() as _, I32_DT, &kvl_d_s, &kvl_d_st);
            let plan_d = reg.call(prefill_variant.plan, &[
                TVMFFIAny::dltensor(&dl_fws_d), TVMFFIAny::dltensor(&dl_iws_d), TVMFFIAny::dltensor(&dl_pws_d),
                TVMFFIAny::dltensor(&dl_cuq_d), TVMFFIAny::dltensor(&dl_cuk_d), TVMFFIAny::dltensor(&dl_kvl_d),
                TVMFFIAny::int64(kv_len_d), TVMFFIAny::int64(dec_batch),
                TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
                TVMFFIAny::int64(page_size), TVMFFIAny::bool_val(false),
                TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
                TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
                TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
            ]).unwrap();

            // ── POD run args (prefill side uses _p workspace, decode side uses _d) ──
            let pod_args = [
                TVMFFIAny::dltensor(&dl_fws_p), TVMFFIAny::dltensor(&dl_iws_p), plan_p,
                TVMFFIAny::dltensor(&dl_qp), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_qo_indptr_p), TVMFFIAny::dltensor(&dl_kv_indptr_p),
                TVMFFIAny::dltensor(&dl_kv_indices_p), TVMFFIAny::dltensor(&dl_kv_last_p),
                TVMFFIAny::dltensor(&dl_op), TVMFFIAny::none(),
                TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale), TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
                TVMFFIAny::dltensor(&dl_fws_d), TVMFFIAny::dltensor(&dl_iws_d), plan_d,
                TVMFFIAny::dltensor(&dl_qd), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_qo_indptr_d), TVMFFIAny::dltensor(&dl_kv_indptr_d),
                TVMFFIAny::dltensor(&dl_kv_indices_d), TVMFFIAny::dltensor(&dl_kv_last_d),
                TVMFFIAny::dltensor(&dl_od), TVMFFIAny::none(),
                TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale), TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
                TVMFFIAny::bool_val(false), TVMFFIAny::dltensor(&dl_sched),
            ];

            // ── Separate: paged prefill + decode (sequential, share _p workspace) ──
            let sep_prefill_args = [
                TVMFFIAny::dltensor(&dl_fws_p), TVMFFIAny::dltensor(&dl_iws_p), plan_p,
                TVMFFIAny::dltensor(&dl_qp), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_qo_indptr_p), TVMFFIAny::dltensor(&dl_kv_indptr_p),
                TVMFFIAny::dltensor(&dl_kv_indices_p), TVMFFIAny::dltensor(&dl_kv_last_p),
                TVMFFIAny::dltensor(&dl_op), TVMFFIAny::none(),
                TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale), TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
                TVMFFIAny::int64(0),
            ];

            // Decode plan (using FlashInfer decode, not prefill)
            let dl_indptr_d_cpu = cpu_dl(kv_indptr_d.as_ptr() as _, I32_DT, &cu_d_s, &cu_d_st);
            let empty_s = [0i64]; let empty_st = strides(&empty_s);
            let dl_eq = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
            let dl_ek = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
            let dec_plan = reg.call(decode_variant.plan, &[
                TVMFFIAny::dltensor(&dl_fws_p), TVMFFIAny::dltensor(&dl_iws_p), TVMFFIAny::dltensor(&dl_pws),
                TVMFFIAny::dltensor(&dl_indptr_d_cpu),
                TVMFFIAny::int64(dec_batch), TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
                TVMFFIAny::int64(page_size), TVMFFIAny::bool_val(false),
                TVMFFIAny::int64(-1), TVMFFIAny::float64(0.0),
                TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
                TVMFFIAny::dltensor(&dl_eq), TVMFFIAny::dltensor(&dl_ek),
            ]).unwrap();
            let sep_decode_args = [
                TVMFFIAny::dltensor(&dl_fws_p), TVMFFIAny::dltensor(&dl_iws_p), dec_plan,
                TVMFFIAny::dltensor(&dl_qd), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_kv_indptr_d), TVMFFIAny::dltensor(&dl_kv_indices_d),
                TVMFFIAny::dltensor(&dl_kv_last_d),
                TVMFFIAny::dltensor(&dl_od), TVMFFIAny::none(),
                TVMFFIAny::int64(0), TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
                TVMFFIAny::none(), TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale),
                TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
            ];

            // ── Run benchmarks (skip config if kernel fails) ──
            // Validate POD first
            if let Err(e) = reg.call(pod_variant.run, &pod_args) {
                cudaDeviceSynchronize(); // clear error state
                println!("{:<50} {:>10} {:>10} {:>8}", label, "FAIL", "-", "-");
                eprintln!("  POD error: {e}");
                cudaFree(k_cache); cudaFree(v_cache);
                cudaFree(q_p); cudaFree(o_p); cudaFree(q_d); cudaFree(o_d);
                cudaFree(qo_indptr_p_gpu); cudaFree(kv_indptr_p_gpu); cudaFree(kv_indices_p_gpu); cudaFree(kv_last_p_gpu);
                cudaFree(qo_indptr_d_gpu); cudaFree(kv_indptr_d_gpu); cudaFree(kv_indices_d_gpu); cudaFree(kv_last_d_gpu);
                cudaFree(sched); cudaFree(fws_d_ptr); cudaFree(iws_d_ptr);
                std::alloc::dealloc(pws_d_ptr as *mut u8, pws_layout);
                continue;
            }
            cudaDeviceSynchronize();

            let pod_ms = cuda_bench(10, 200, || {
                reg.call(pod_variant.run, &pod_args).unwrap();
            });

            // Validate separate path (reset CUDA error state first)
            cudaDeviceSynchronize(); cudaGetLastError();
            let sep_ok = reg.call(prefill_variant.paged_run, &sep_prefill_args).is_ok() && {
                cudaDeviceSynchronize();
                reg.call(decode_variant.run, &sep_decode_args).is_ok()
            };
            cudaDeviceSynchronize(); cudaGetLastError(); // clear any sticky errors

            if !sep_ok {
                let pod_us = pod_ms * 1000.0;
                println!("{:<50} {:>10.1} {:>10} {:>8}", label, pod_us, "SEP_ERR", "-");
                cudaFree(k_cache); cudaFree(v_cache);
                cudaFree(q_p); cudaFree(o_p); cudaFree(q_d); cudaFree(o_d);
                cudaFree(qo_indptr_p_gpu); cudaFree(kv_indptr_p_gpu); cudaFree(kv_indices_p_gpu); cudaFree(kv_last_p_gpu);
                cudaFree(qo_indptr_d_gpu); cudaFree(kv_indptr_d_gpu); cudaFree(kv_indices_d_gpu); cudaFree(kv_last_d_gpu);
                cudaFree(sched); cudaFree(fws_d_ptr); cudaFree(iws_d_ptr);
                std::alloc::dealloc(pws_d_ptr as *mut u8, pws_layout);
                continue;
            }

            let sep_ms = cuda_bench(10, 200, || {
                reg.call(prefill_variant.paged_run, &sep_prefill_args).ok();
                reg.call(decode_variant.run, &sep_decode_args).ok();
            });

            let pod_us = pod_ms * 1000.0;
            let sep_us = sep_ms * 1000.0;
            let speedup = sep_us / pod_us;
            println!("{:<50} {:>10.1} {:>10.1} {:>7.2}x", label, pod_us, sep_us, speedup);

            cudaFree(k_cache); cudaFree(v_cache);
            cudaFree(q_p); cudaFree(o_p); cudaFree(q_d); cudaFree(o_d);
            cudaFree(qo_indptr_p_gpu); cudaFree(kv_indptr_p_gpu); cudaFree(kv_indices_p_gpu); cudaFree(kv_last_p_gpu);
            cudaFree(qo_indptr_d_gpu); cudaFree(kv_indptr_d_gpu); cudaFree(kv_indices_d_gpu); cudaFree(kv_last_d_gpu);
            cudaFree(sched); cudaFree(fws_d_ptr); cudaFree(iws_d_ptr);
            std::alloc::dealloc(pws_d_ptr as *mut u8, pws_layout);
        }
    }
}

fn main() {
    println!("{:=<80}", "= FlashInfer Attention Benchmarks ");

    let reg = KernelRegistry::new();
    println!("GPU: SM{}, backend: {:?}", reg.arch(), reg.default_backend());

    bench_utilities(&reg);
    bench_gemm(&reg);
    bench_topk(&reg);

    let ws = Workspace::new();
    bench_prefill(&reg, &ws);
    bench_decode(&reg, &ws);
    bench_pod(&reg, &ws);

    println!("\nDone.");
}
