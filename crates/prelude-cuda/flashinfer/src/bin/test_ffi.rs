//! Minimal FFI test: call FlashInfer prefill plan + ragged_run directly.
//! Run: cargo run -p flashinfer --bin test_ffi --release

use flashinfer::types::*;
use flashinfer::*;
use std::ffi::c_void;

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const BF16_DT: DLDataType = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
const I32_DT: DLDataType = DLDataType { code: KDLINT, bits: 32, lanes: 1 };
const U8_DT: DLDataType = DLDataType { code: KDLUINT, bits: 8, lanes: 1 };

/// Compute row-major contiguous strides for a given shape.
fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn gpu_dl(data: *mut c_void, dtype: DLDataType, shape: &[i64], strides: &[i64]) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice { device_type: KDLCUDA, device_id: 0 },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

fn cpu_dl(data: *mut c_void, dtype: DLDataType, shape: &[i64], strides: &[i64]) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice { device_type: KDLCPU, device_id: 0 },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

fn main() {
    println!("=== FlashInfer FFI Test ===");

    let reg = KernelRegistry::new();
    println!("GPU arch: SM{}", reg.arch());

    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: 128,
        head_dim_vo: 128,
        sliding_window: false,
        logits_soft_cap: false,
        backend: Backend::FA2,
    };
    let variant = reg.get_prefill(&key).expect("FA2 prefill variant not found");
    println!("Found variant: plan={:p} ragged_run={:p}",
        variant.plan as *const (), variant.ragged_run as *const ());

    unsafe {
        // ── Workspace buffers ──
        let mut float_ws: *mut c_void = std::ptr::null_mut();
        let mut int_ws: *mut c_void = std::ptr::null_mut();
        let ws_size = 128 * 1024 * 1024usize;
        let iws_size = 8 * 1024 * 1024usize;
        assert_eq!(cudaMalloc(&mut float_ws, ws_size), 0);
        assert_eq!(cudaMalloc(&mut int_ws, iws_size), 0);
        cudaMemset(float_ws, 0, ws_size);
        cudaMemset(int_ws, 0, iws_size);

        // Pinned CPU workspace
        let pinned_layout = std::alloc::Layout::from_size_align(iws_size, 64).unwrap();
        let pinned_ws = std::alloc::alloc_zeroed(pinned_layout) as *mut c_void;
        assert!(!pinned_ws.is_null());
        println!("Workspace allocated");

        // ── Test params ──
        let batch_size = 1i64;
        let seq_len = 4i64;
        let num_qo_heads = 2i64;
        let num_kv_heads = 2i64;
        let head_dim = 128i64;
        let total_tokens = seq_len;

        // ── Allocate GPU tensors ──
        let q_size = (total_tokens * num_qo_heads * head_dim * 2) as usize;
        let kv_size = (total_tokens * num_kv_heads * head_dim * 2) as usize;

        let mut q_ptr: *mut c_void = std::ptr::null_mut();
        let mut k_ptr: *mut c_void = std::ptr::null_mut();
        let mut v_ptr: *mut c_void = std::ptr::null_mut();
        let mut o_ptr: *mut c_void = std::ptr::null_mut();
        assert_eq!(cudaMalloc(&mut q_ptr, q_size), 0);
        assert_eq!(cudaMalloc(&mut k_ptr, kv_size), 0);
        assert_eq!(cudaMalloc(&mut v_ptr, kv_size), 0);
        assert_eq!(cudaMalloc(&mut o_ptr, q_size), 0);
        cudaMemset(q_ptr, 0, q_size);
        cudaMemset(k_ptr, 0, kv_size);
        cudaMemset(v_ptr, 0, kv_size);
        cudaMemset(o_ptr, 0, q_size);

        // CPU indptrs for plan (FlashInfer plan reads on CPU!)
        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];

        // GPU indptrs for run (kernels read on GPU)
        let mut cu_q_gpu: *mut c_void = std::ptr::null_mut();
        let mut cu_k_gpu: *mut c_void = std::ptr::null_mut();
        let indptr_gpu_size = ((batch_size + 1) * 4) as usize;
        assert_eq!(cudaMalloc(&mut cu_q_gpu, indptr_gpu_size), 0);
        assert_eq!(cudaMalloc(&mut cu_k_gpu, indptr_gpu_size), 0);
        cudaMemcpy(cu_q_gpu, cu_q_data.as_ptr() as *const c_void, indptr_gpu_size, CUDA_MEMCPY_H2D);
        cudaMemcpy(cu_k_gpu, cu_k_data.as_ptr() as *const c_void, indptr_gpu_size, CUDA_MEMCPY_H2D);

        println!("Tensors allocated");

        // ── Build DLTensors with explicit strides ──
        // TVM's TensorView::stride() requires non-null strides!
        let fws_s: [i64; 1] = [ws_size as i64];
        let fws_st = contiguous_strides(&fws_s);
        let iws_s: [i64; 1] = [iws_size as i64];
        let iws_st = contiguous_strides(&iws_s);
        let cu_s: [i64; 1] = [batch_size + 1];
        let cu_st = contiguous_strides(&cu_s);
        let kvl_s: [i64; 1] = [batch_size];
        let kvl_st = contiguous_strides(&kvl_s);
        let q_s: [i64; 3] = [total_tokens, num_qo_heads, head_dim];
        let q_st = contiguous_strides(&q_s);
        let k_s: [i64; 3] = [total_tokens, num_kv_heads, head_dim];
        let k_st = contiguous_strides(&k_s);

        let dl_fws = gpu_dl(float_ws, U8_DT, &fws_s, &fws_st);
        let dl_iws = gpu_dl(int_ws, U8_DT, &iws_s, &iws_st);
        let dl_pws = cpu_dl(pinned_ws, U8_DT, &iws_s, &iws_st);
        // CPU tensors for plan
        let dl_cuq_cpu = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk_cpu = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        // GPU tensors for run
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);

        reg.set_stream(0, std::ptr::null_mut());

        // ── Plan (19 args for FA2) ──
        println!("\nCalling prefill plan...");
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws),   // float_workspace (GPU)
            TVMFFIAny::dltensor(&dl_iws),   // int_workspace (GPU)
            TVMFFIAny::dltensor(&dl_pws),   // pinned_workspace (CPU)
            TVMFFIAny::dltensor(&dl_cuq_cpu), // qo_indptr (CPU!)
            TVMFFIAny::dltensor(&dl_cuk_cpu), // kv_indptr (CPU!)
            TVMFFIAny::dltensor(&dl_kvl),   // kv_len_arr (CPU!)
            TVMFFIAny::int64(total_tokens), // total_num_rows
            TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads),
            TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1),            // page_size (ragged=1)
            TVMFFIAny::bool_val(false),     // enable_cuda_graph
            TVMFFIAny::int64(head_dim),     // head_dim_qk
            TVMFFIAny::int64(head_dim),     // head_dim_vo
            TVMFFIAny::bool_val(true),      // causal
            TVMFFIAny::int64(-1),           // window_left
            TVMFFIAny::int64(-1),           // fixed_split_size
            TVMFFIAny::bool_val(false),     // disable_split_kv
            TVMFFIAny::int64(0),            // num_colocated_ctas
        ];
        let plan_info = match reg.call(variant.plan, &plan_args) {
            Ok(info) => {
                println!("Plan OK! type_index={}", info.type_index);
                info
            }
            Err(e) => {
                eprintln!("Plan FAILED: {e}");
                return;
            }
        };

        // ── Ragged Run (25 args) ──
        let sm_scale = 1.0f64 / (head_dim as f64).sqrt();
        println!("\nCalling ragged_run...");
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),         // float_workspace
            TVMFFIAny::dltensor(&dl_iws),         // int_workspace
            plan_info,                             // plan_info (opaque Array<int64_t>)
            TVMFFIAny::dltensor(&dl_q),           // q
            TVMFFIAny::dltensor(&dl_k),           // k
            TVMFFIAny::dltensor(&dl_v),           // v
            TVMFFIAny::dltensor(&dl_cuq_gpu),     // qo_indptr (GPU)
            TVMFFIAny::dltensor(&dl_cuk_gpu),     // kv_indptr (GPU)
            TVMFFIAny::dltensor(&dl_o),           // output
            TVMFFIAny::none(),                    // maybe_lse
            TVMFFIAny::int64(1),                  // mask_mode = Causal
            TVMFFIAny::int64(0),                  // layout = NHD
            TVMFFIAny::int64(-1),                 // window_left
            TVMFFIAny::bool_val(false),           // enable_pdl
            // FA2 additional params:
            TVMFFIAny::none(),                    // maybe_custom_mask
            TVMFFIAny::none(),                    // maybe_mask_indptr
            TVMFFIAny::none(),                    // maybe_alibi_slopes
            TVMFFIAny::none(),                    // maybe_prefix_len_ptr
            TVMFFIAny::none(),                    // maybe_token_pos_in_items_ptr
            TVMFFIAny::none(),                    // maybe_max_item_len_ptr
            TVMFFIAny::float64(0.0),              // logits_soft_cap
            TVMFFIAny::float64(sm_scale),         // sm_scale
            TVMFFIAny::float64(1.0),              // rope_rcp_scale (Python default=1.0)
            TVMFFIAny::float64(1e4),              // rope_rcp_theta (Python default=1e4)
            TVMFFIAny::int64(0),                  // token_pos_in_items_len
        ];
        match reg.call(variant.ragged_run, &run_args) {
            Ok(_) => {
                cudaDeviceSynchronize();
                println!("Ragged run OK!");
            }
            Err(e) => eprintln!("Ragged run FAILED: {e}"),
        }

        // ── Decode plan test ──
        let decode_key = DecodeKey {
            dtype: KernelDtype::BF16,
            head_dim_qk: 128,
            head_dim_vo: 128,
            sliding_window: false,
            logits_soft_cap: false,
        };
        if let Some(decode_var) = reg.get_decode(&decode_key) {
            println!("\nCalling decode plan...");
            let dec_indptr: [i32; 2] = [0, 1];
            let dec_s: [i64; 1] = [2];
            let dec_st = contiguous_strides(&dec_s);
            let dl_dec_indptr = cpu_dl(dec_indptr.as_ptr() as *mut c_void, I32_DT, &dec_s, &dec_st);

            // Empty tensors for q/k dtype detection
            let empty_s: [i64; 1] = [0];
            let empty_st: [i64; 1] = [1];
            let dl_eq = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
            let dl_ek = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);

            let decode_plan_args = [
                TVMFFIAny::dltensor(&dl_fws),
                TVMFFIAny::dltensor(&dl_iws),
                TVMFFIAny::dltensor(&dl_pws),
                TVMFFIAny::dltensor(&dl_dec_indptr), // kv_indptr (CPU!)
                TVMFFIAny::int64(1),                 // batch_size
                TVMFFIAny::int64(num_qo_heads),
                TVMFFIAny::int64(num_kv_heads),
                TVMFFIAny::int64(16),                // page_size
                TVMFFIAny::bool_val(false),          // cuda_graph
                TVMFFIAny::int64(-1),                // window_left
                TVMFFIAny::float64(0.0),             // logits_soft_cap
                TVMFFIAny::int64(head_dim),
                TVMFFIAny::int64(head_dim),
                TVMFFIAny::dltensor(&dl_eq),
                TVMFFIAny::dltensor(&dl_ek),
            ];
            match reg.call(decode_var.plan, &decode_plan_args) {
                Ok(info) => println!("Decode plan OK! type_index={}", info.type_index),
                Err(e) => println!("Decode plan error: {e}"),
            }
        }

        // Cleanup
        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr); cudaFree(o_ptr);
        cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
        cudaFree(float_ws); cudaFree(int_ws);
        println!("\nDone!");
    }
}
