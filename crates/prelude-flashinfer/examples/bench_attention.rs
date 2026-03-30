//! Benchmark FlashInfer attention kernels.
//!
//! Run: cargo run -p prelude-flashinfer --example bench_attention --release
//!
//! Benchmarks:
//! 1. Prefill throughput (ragged, causal) — vary seq_len
//! 2. Decode throughput (paged) — vary batch_size, kv_len
//! 3. MLA decode throughput — DeepSeek V2/V3 config
//! 4. Utility kernel latency (softmax, rmsnorm)

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
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

const BF16_DT: DLDataType = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
const FP32_DT: DLDataType = DLDataType { code: KDLFLOAT, bits: 32, lanes: 1 };
const I32_DT: DLDataType = DLDataType { code: KDLINT, bits: 32, lanes: 1 };
const U8_DT: DLDataType = DLDataType { code: KDLUINT, bits: 8, lanes: 1 };

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

// ── Bench: Prefill ───────────────────────────────────────────────────

fn bench_prefill(reg: &KernelRegistry, ws: &Workspace) {
    println!("\n=== Prefill (ragged, causal, BF16) ===");
    println!("{:<10} {:<6} {:<6} {:<10} {:<12}", "seq_len", "heads", "hdim", "ms", "TFLOPS");

    // Use FA2 for benchmarks — more portable, works on SM80+
    let backend = Backend::FA2;
    let configs = [
        (256, 32, 128),
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
            None => { println!("  Skipping ({seq_len}, {num_heads}, {head_dim}): no variant"); continue; }
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

        unsafe { reg.set_stream(0, std::ptr::null_mut()); }

        // Plan: FA2 has 19 args, FA3 has 16 args
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
            // FA3/SM90 plan: 16 args (no fixed_split, disable_split, colocated_ctas)
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

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        // Run: FA2 has 25 args, FA3 has 21 args
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
            // FA3/SM90: base 14 args + additional 7 (prefix_len, token_pos, max_item, scale_v, softcap, sm_scale, scale_v_scalar)
            vec![
                TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), plan_info,
                TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
                TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_cuk_gpu),
                TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(),
                TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1),
                TVMFFIAny::bool_val(false),
                // additional: prefix_len, token_pos, max_item, scale_v
                TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
                TVMFFIAny::float64(0.0),       // logits_soft_cap
                TVMFFIAny::float64(sm_scale),  // sm_scale
                TVMFFIAny::float64(1.0),       // scale_v_scalar
            ]
        };

        let ms = cuda_bench(3, 20, || {
            unsafe { reg.call(variant.ragged_run, &run_args).unwrap(); }
        });

        // FLOPS: 2 * batch * num_heads * seq_len^2 * head_dim (self-attention, causal ≈ half)
        let flops = 2.0 * batch_size as f64 * num_heads as f64 * (seq_len as f64).powi(2)
            * head_dim as f64 / 2.0;  // causal mask halves
        let tflops = flops / (ms as f64 * 1e-3) / 1e12;

        println!("{:<10} {:<6} {:<6} {:<10.3} {:<12.2}", seq_len, num_heads, head_dim, ms, tflops);

        unsafe {
            cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
            cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
        }
    }
}

// ── Bench: Decode ────────────────────────────────────────────────────

fn bench_decode(reg: &KernelRegistry, ws: &Workspace) {
    println!("\n=== Decode (paged, BF16) ===");
    println!("{:<8} {:<8} {:<6} {:<6} {:<10} {:<12}", "batch", "kv_len", "heads", "hdim", "ms", "TFLOPS");

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

        let ms = cuda_bench(3, 50, || {
            unsafe { reg.call(variant.run, &run_args).unwrap(); }
        });

        let flops = 2.0 * batch_size as f64 * num_heads as f64 * kv_len as f64 * head_dim as f64;
        let tflops = flops / (ms as f64 * 1e-3) / 1e12;

        println!("{:<8} {:<8} {:<6} {:<6} {:<10.3} {:<12.2}",
            batch_size, kv_len, num_heads, head_dim, ms, tflops);

        unsafe {
            cudaFree(q); cudaFree(o); cudaFree(k_cache); cudaFree(v_cache);
            cudaFree(kvi_gpu); cudaFree(kv_indptr_gpu); cudaFree(kv_last_gpu);
        }
    }
}

// ── Bench: Utility kernels ───────────────────────────────────────────

fn bench_utilities(reg: &KernelRegistry) {
    println!("\n=== Utility Kernels ===");
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
}

fn main() {
    println!("=== FlashInfer Attention Benchmarks ===");

    let reg = KernelRegistry::new();
    println!("GPU: SM{}, backend: {:?}", reg.arch(), reg.default_backend());

    bench_utilities(&reg);

    let ws = Workspace::new();
    bench_prefill(&reg, &ws);
    bench_decode(&reg, &ws);

    println!("\nDone.");
}
