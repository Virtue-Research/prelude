//! Benchmark Prelude's vendored FlashInfer CUTLASS fused MoE runner for the
//! TopicGuard Qwen3-MoE production shape.
//!
//! Run:
//!   cargo run -p flashinfer --example bench_cutlass_moe_topicguard --release

use flashinfer::KernelRegistry;
use flashinfer::types::*;
use std::env;
use std::ffi::c_void;

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

const BF16_DT: DLDataType = DLDataType {
    code: KDLBFLOAT,
    bits: 16,
    lanes: 1,
};
const FP32_DT: DLDataType = DLDataType {
    code: KDLFLOAT,
    bits: 32,
    lanes: 1,
};
const I32_DT: DLDataType = DLDataType {
    code: KDLINT,
    bits: 32,
    lanes: 1,
};

const HIDDEN_SIZE: i64 = 2048;
const INTERMEDIATE_SIZE: i64 = 768;
const NUM_EXPERTS: i64 = 64;
const TOP_K: i64 = 4;

fn parse_tokens() -> Vec<i64> {
    env::var("MOE_TOKENS")
        .unwrap_or_else(|_| "32,64,128,256,512,1024,2048,4096".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect()
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn strides(shape: &[i64]) -> Vec<i64> {
    let mut out = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * shape[i + 1];
    }
    out
}

fn gpu_dl(data: *mut c_void, dtype: DLDataType, shape: &[i64], st: &[i64]) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice {
            device_type: KDLCUDA,
            device_id: 0,
        },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: st.as_ptr(),
        byte_offset: 0,
    }
}

fn check_cuda(rc: i32, what: &str) {
    assert_eq!(rc, 0, "{what} failed with cuda status {rc}");
}

fn gpu_alloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(cudaMalloc(&mut ptr, size), "cudaMalloc");
        check_cuda(cudaMemset(ptr, 0, size), "cudaMemset");
    }
    ptr
}

fn gpu_upload<T>(data: &[T]) -> *mut c_void {
    let bytes = std::mem::size_of_val(data);
    let ptr = gpu_alloc(bytes);
    unsafe {
        check_cuda(
            cudaMemcpy(ptr, data.as_ptr() as *const c_void, bytes, 1),
            "cudaMemcpy H2D",
        );
    }
    ptr
}

fn cuda_bench_us<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f32 {
    for _ in 0..warmup {
        f();
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");
    }

    let mut start: *mut c_void = std::ptr::null_mut();
    let mut end: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(cudaEventCreate(&mut start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&mut end), "cudaEventCreate end");
        check_cuda(
            cudaEventRecord(start, std::ptr::null_mut()),
            "cudaEventRecord start",
        );
    }
    for _ in 0..iters {
        f();
    }
    unsafe {
        check_cuda(
            cudaEventRecord(end, std::ptr::null_mut()),
            "cudaEventRecord end",
        );
        check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end");
        let mut ms = 0.0f32;
        check_cuda(
            cudaEventElapsedTime(&mut ms, start, end),
            "cudaEventElapsedTime",
        );
        check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
        check_cuda(cudaEventDestroy(end), "cudaEventDestroy end");
        ms * 1000.0 / iters as f32
    }
}

fn main() {
    let reg = KernelRegistry::new();
    println!(
        "backend=prelude_flashinfer_cutlass device_sm={} dtype=bf16 experts={} topk={} hidden={} inter={}",
        reg.arch(),
        NUM_EXPERTS,
        TOP_K,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE
    );

    if reg.get_utility("init").is_none() {
        println!("CUTLASS fused MoE kernel not compiled; skipping");
        return;
    }
    reg.set_stream(0, std::ptr::null_mut());

    let runner = flashinfer::moe::FusedMoeRunner::new()
        .expect("failed to create FlashInfer CUTLASS FusedMoeRunner");

    let warmup = parse_env_usize("MOE_WARMUP", 5);
    let iters = parse_env_usize("MOE_ITERS", 50);

    let w1_elems = (NUM_EXPERTS * 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE) as usize;
    let w2_elems = (NUM_EXPERTS * HIDDEN_SIZE * INTERMEDIATE_SIZE) as usize;
    let w1 = gpu_alloc(w1_elems * 2);
    let w2 = gpu_alloc(w2_elems * 2);
    let w1_s = [NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE];
    let w2_s = [NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE];
    let w1_st = strides(&w1_s);
    let w2_st = strides(&w2_s);
    let dl_w1 = gpu_dl(w1, BF16_DT, &w1_s, &w1_st);
    let dl_w2 = gpu_dl(w2, BF16_DT, &w2_s, &w2_st);

    for tokens in parse_tokens() {
        let input_elems = (tokens * HIDDEN_SIZE) as usize;
        let expert_elems = (tokens * TOP_K) as usize;
        let input = gpu_alloc(input_elems * 2);
        let output = gpu_alloc(input_elems * 2);

        let expert_ids: Vec<i32> = (0..expert_elems)
            .map(|i| ((i * 37 + 13) % NUM_EXPERTS as usize) as i32)
            .collect();
        let expert_weights: Vec<f32> = vec![1.0 / TOP_K as f32; expert_elems];
        let expert_ids_gpu = gpu_upload(&expert_ids);
        let expert_weights_gpu = gpu_upload(&expert_weights);

        let out_s = [tokens, HIDDEN_SIZE];
        let in_s = [tokens, HIDDEN_SIZE];
        let expert_s = [tokens, TOP_K];
        let out_st = strides(&out_s);
        let in_st = strides(&in_s);
        let expert_st = strides(&expert_s);

        let dl_output = gpu_dl(output, BF16_DT, &out_s, &out_st);
        let dl_input = gpu_dl(input, BF16_DT, &in_s, &in_st);
        let dl_expert_ids = gpu_dl(expert_ids_gpu, I32_DT, &expert_s, &expert_st);
        let dl_expert_weights = gpu_dl(expert_weights_gpu, FP32_DT, &expert_s, &expert_st);

        let us = cuda_bench_us(warmup, iters, || unsafe {
            runner
                .run_moe(
                    &dl_output,
                    &dl_input,
                    &dl_expert_ids,
                    &dl_expert_weights,
                    &dl_w1,
                    &dl_w2,
                    1,
                    0,
                    1,
                    0,
                )
                .expect("run_moe failed");
        });
        println!("cutlass tokens={tokens:5} direct={us:.2} us");

        unsafe {
            cudaFree(input);
            cudaFree(output);
            cudaFree(expert_ids_gpu);
            cudaFree(expert_weights_gpu);
        }
    }

    unsafe {
        cudaFree(w1);
        cudaFree(w2);
    }
}
