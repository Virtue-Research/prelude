//! Correctness tests for FlashInfer AOT kernels.
//!
//! Validates prefill, decode, MLA, and utility kernel outputs against
//! CPU reference implementations.

use prelude_flashinfer::types::*;
use prelude_flashinfer::*;
use std::ffi::c_void;

// ── CUDA FFI ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

const H2D: i32 = 1;
const D2H: i32 = 2;

const BF16_DT: DLDataType = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
const FP32_DT: DLDataType = DLDataType { code: KDLFLOAT, bits: 32, lanes: 1 };
const I32_DT: DLDataType = DLDataType { code: KDLINT, bits: 32, lanes: 1 };
const U8_DT: DLDataType = DLDataType { code: KDLUINT, bits: 8, lanes: 1 };

// ── Helpers ──────────────────────────────────────────────────────────

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

/// Allocate zeroed GPU memory.
fn gpu_alloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        assert_eq!(cudaMalloc(&mut ptr, size), 0, "cudaMalloc failed");
        cudaMemset(ptr, 0, size);
    }
    ptr
}

/// Upload data to GPU, returning device pointer.
fn gpu_upload<T: Copy>(data: &[T]) -> *mut c_void {
    let size = data.len() * std::mem::size_of::<T>();
    let ptr = gpu_alloc(size);
    unsafe {
        assert_eq!(
            cudaMemcpy(ptr, data.as_ptr() as *const c_void, size, H2D),
            0
        );
    }
    ptr
}

/// Download data from GPU.
fn gpu_download<T: Copy + Default>(ptr: *mut c_void, count: usize) -> Vec<T> {
    let size = count * std::mem::size_of::<T>();
    let mut result = vec![T::default(); count];
    unsafe {
        assert_eq!(
            cudaMemcpy(result.as_mut_ptr() as *mut c_void, ptr, size, D2H),
            0
        );
    }
    result
}

/// Convert f32 to BF16 (truncate).
fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Convert BF16 to f32.
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ── Workspace helper ─────────────────────────────────────────────────

struct Workspace {
    float_ws: *mut c_void,
    int_ws: *mut c_void,
    pinned_ws: *mut c_void,
    float_size: usize,
    int_size: usize,
}

impl Workspace {
    fn new() -> Self {
        let float_size = 128 * 1024 * 1024;
        let int_size = 8 * 1024 * 1024;
        let float_ws = gpu_alloc(float_size);
        let int_ws = gpu_alloc(int_size);
        let pinned_layout = std::alloc::Layout::from_size_align(int_size, 64).unwrap();
        let pinned_ws = unsafe { std::alloc::alloc_zeroed(pinned_layout) as *mut c_void };
        Self { float_ws, int_ws, pinned_ws, float_size, int_size }
    }

    fn dl_float(&self) -> (DLTensor, [i64; 1], [i64; 1]) {
        let s = [self.float_size as i64];
        let st = [1i64];
        (gpu_dl(self.float_ws, U8_DT, &s, &st), s, st)
    }

    fn dl_int(&self) -> (DLTensor, [i64; 1], [i64; 1]) {
        let s = [self.int_size as i64];
        let st = [1i64];
        (gpu_dl(self.int_ws, U8_DT, &s, &st), s, st)
    }

    fn dl_pinned(&self) -> (DLTensor, [i64; 1], [i64; 1]) {
        let s = [self.int_size as i64];
        let st = [1i64];
        (cpu_dl(self.pinned_ws, U8_DT, &s, &st), s, st)
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

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn registry_creation() {
    let reg = KernelRegistry::new();
    let arch = reg.arch();
    assert!(arch >= 70, "Expected GPU with SM >= 70, got SM{arch}");
    println!("GPU: SM{arch}, default backend: {:?}", reg.default_backend());
}

#[test]
fn prefill_variant_lookup() {
    let reg = KernelRegistry::new();

    // FA2 BF16 h128 — should always exist
    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: 128,
        head_dim_vo: 128,
        sliding_window: false,
        logits_soft_cap: false,
        backend: Backend::FA2,
    };
    assert!(reg.get_prefill(&key).is_some(), "FA2 BF16 h128 prefill not found");

    // FA3 BF16 h128 — SM90+ only
    if reg.arch() >= 90 {
        let key_fa3 = PrefillKey { backend: Backend::FA3, ..key };
        assert!(reg.get_prefill(&key_fa3).is_some(), "FA3 BF16 h128 prefill not found");
    }
}

#[test]
fn decode_variant_lookup() {
    let reg = KernelRegistry::new();
    let key = DecodeKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: 128,
        head_dim_vo: 128,
        sliding_window: false,
        logits_soft_cap: false,
    };
    assert!(reg.get_decode(&key).is_some(), "BF16 h128 decode not found");
}

#[test]
fn mla_decode_variant_lookup() {
    let reg = KernelRegistry::new();
    let key = MLADecodeKey {
        dtype: KernelDtype::BF16,
        head_dim_ckv: 512,
        head_dim_kpe: 64,
    };
    assert!(reg.get_mla_decode(&key).is_some(), "MLA decode BF16 c512k64 not found");
}

#[test]
fn mla_paged_variant_lookup() {
    let reg = KernelRegistry::new();
    let key = MLAPagedKey {
        dtype: KernelDtype::BF16,
        head_dim_ckv: 512,
        head_dim_kpe: 64,
    };
    assert!(reg.get_mla_paged(&key).is_some(), "MLA paged BF16 c512k64 not found");
}

#[test]
fn utility_kernel_lookup() {
    let reg = KernelRegistry::new();

    // Sampling
    assert!(reg.get_utility("softmax").is_some(), "softmax not found");
    assert!(reg.get_utility("sampling_from_probs").is_some(), "sampling_from_probs not found");
    assert!(reg.get_utility("top_k_sampling_from_probs").is_some(), "top_k not found");
    assert!(reg.get_utility("top_p_sampling_from_probs").is_some(), "top_p not found");

    // Norm
    assert!(reg.get_utility("rmsnorm").is_some(), "rmsnorm not found");
    assert!(reg.get_utility("fused_add_rmsnorm").is_some(), "fused_add_rmsnorm not found");

    // RoPE
    assert!(reg.get_utility("apply_rope").is_some(), "apply_rope not found");
    assert!(reg.get_utility("apply_rope_pos_ids_cos_sin_cache").is_some(), "rope cos_sin not found");

    // Page
    assert!(reg.get_utility("append_paged_kv_cache").is_some(), "append_paged_kv not found");

    // Cascade
    assert!(reg.get_utility("merge_state").is_some(), "merge_state not found");
}

#[test]
fn prefill_ragged_causal_bf16() {
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let batch_size = 1i64;
    let seq_len = 8i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;

    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: head_dim as u32,
        head_dim_vo: head_dim as u32,
        sliding_window: false,
        logits_soft_cap: false,
        backend: Backend::FA2,
    };
    let variant = reg.get_prefill(&key).expect("Prefill variant not found");

    unsafe {
        // Generate Q, K, V with known pattern (small values for BF16 precision)
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..total).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let k_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.01 * ((i + 3) as f32 % 10.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.02 * ((i + 7) as f32 % 10.0))).collect();

        let q_ptr = gpu_upload(&q_bf16);
        let k_ptr = gpu_upload(&k_bf16);
        let v_ptr = gpu_upload(&v_bf16);
        let o_ptr = gpu_alloc(total * 2);

        // Indptrs
        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let cu_k_gpu = gpu_upload(&cu_k_data);

        // Build DLTensors
        let (dl_fws, fws_s, fws_st) = ws.dl_float();
        let (dl_iws, iws_s, iws_st) = ws.dl_int();
        let (dl_pws, _, _) = ws.dl_pinned();
        let cu_s = [batch_size + 1];
        let cu_st = contiguous_strides(&cu_s);
        let kvl_s = [batch_size];
        let kvl_st = contiguous_strides(&kvl_s);
        let q_s = [seq_len, num_qo_heads, head_dim];
        let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim];
        let k_st = contiguous_strides(&k_s);

        let dl_cuq_cpu = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk_cpu = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);

        reg.set_stream(0, std::ptr::null_mut());

        // Plan
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq_cpu),
            TVMFFIAny::dltensor(&dl_cuk_cpu),
            TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len),
            TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads),
            TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1),           // page_size (ragged=1)
            TVMFFIAny::bool_val(false),    // cuda_graph
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true),     // causal
            TVMFFIAny::int64(-1),          // window_left
            TVMFFIAny::int64(-1),          // fixed_split_size
            TVMFFIAny::bool_val(false),    // disable_split_kv
            TVMFFIAny::int64(0),           // num_colocated_ctas
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("Prefill plan failed");

        // Ragged run
        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_cuk_gpu),
            TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),             // maybe_lse
            TVMFFIAny::int64(1),           // mask_mode = Causal
            TVMFFIAny::int64(0),           // layout = NHD
            TVMFFIAny::int64(-1),          // window_left
            TVMFFIAny::bool_val(false),    // enable_pdl
            TVMFFIAny::none(),             // custom_mask
            TVMFFIAny::none(),             // mask_indptr
            TVMFFIAny::none(),             // alibi_slopes
            TVMFFIAny::none(),             // prefix_len_ptr
            TVMFFIAny::none(),             // token_pos_in_items_ptr
            TVMFFIAny::none(),             // max_item_len_ptr
            TVMFFIAny::float64(0.0),       // logits_soft_cap
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),       // rope_rcp_scale
            TVMFFIAny::float64(1e4),       // rope_rcp_theta
            TVMFFIAny::int64(0),           // token_pos_in_items_len
        ];
        reg.call(variant.ragged_run, &run_args)
            .expect("Prefill ragged_run failed");
        cudaDeviceSynchronize();

        // Read output
        let output_bf16 = gpu_download::<u16>(o_ptr, total);
        let output_f32: Vec<f32> = output_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        // Basic sanity: output should not be all zeros (attention produces nonzero output)
        let sum: f32 = output_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Prefill output is all zeros");

        // For causal attention: first token attends only to itself → output should be V[0]
        // (after softmax of a single logit, weight is 1.0)
        // Check first token output ≈ V[0] for head 0
        let v_f32: Vec<f32> = v_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let mut max_diff: f32 = 0.0;
        for d in 0..head_dim as usize {
            let got = output_f32[d]; // token 0, head 0, dim d (NHD layout)
            let expected = v_f32[d]; // V token 0, head 0, dim d
            max_diff = max_diff.max((got - expected).abs());
        }
        assert!(max_diff < 0.05,
            "First token output should ≈ V[0] for causal attention, max_diff={max_diff}");

        println!("Prefill causal BF16: PASS (max_diff={max_diff:.6})");

        // Cleanup
        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr); cudaFree(o_ptr);
        cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
    }
}

#[test]
fn decode_plan_and_run() {
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let batch_size = 1i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;
    let page_size = 16i64;
    let kv_len: i64 = 32;
    let num_pages = (kv_len + page_size - 1) / page_size;
    let total_pages = num_pages * batch_size;

    let key = DecodeKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: head_dim as u32,
        head_dim_vo: head_dim as u32,
        sliding_window: false,
        logits_soft_cap: false,
    };
    let variant = reg.get_decode(&key).expect("Decode variant not found");

    unsafe {
        // Q: [batch, num_qo_heads, head_dim]
        let q_elems = (batch_size * num_qo_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..q_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 7.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let o_ptr = gpu_alloc(q_elems * 2);

        // KV cache: [total_pages, page_size, num_kv_heads, head_dim]
        let kv_elems = (total_pages * page_size * num_kv_heads * head_dim) as usize;
        let k_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.005 * (i as f32 % 11.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 5.0))).collect();
        let k_cache = gpu_upload(&k_bf16);
        let v_cache = gpu_upload(&v_bf16);

        // Page table: kv_indptr [batch+1], kv_indices [total_pages], kv_last_page_len [batch]
        let mut kv_indptr: Vec<i32> = vec![0];
        let mut kv_indices: Vec<i32> = Vec::new();
        for b in 0..batch_size as i32 {
            kv_indptr.push(kv_indptr.last().unwrap() + num_pages as i32);
            for p in 0..num_pages as i32 {
                kv_indices.push(b * num_pages as i32 + p);
            }
        }
        let kv_last_page_len = vec![kv_len as i32 % page_size as i32; batch_size as usize];
        let kv_last_page_len = if kv_last_page_len[0] == 0 {
            vec![page_size as i32; batch_size as usize]
        } else {
            kv_last_page_len
        };

        let kvi_ptr = gpu_upload(&kv_indices);
        let kv_indptr_cpu = kv_indptr.clone();

        // Build DLTensors
        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();

        let indptr_s = [batch_size + 1];
        let indptr_st = contiguous_strides(&indptr_s);
        let dl_indptr_cpu = cpu_dl(kv_indptr_cpu.as_ptr() as *mut c_void, I32_DT, &indptr_s, &indptr_st);

        let empty_s = [0i64];
        let empty_st = [1i64];
        let dl_eq = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
        let dl_ek = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);

        reg.set_stream(0, std::ptr::null_mut());

        // Plan
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_indptr_cpu),
            TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads),
            TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(page_size),
            TVMFFIAny::bool_val(false),    // cuda_graph
            TVMFFIAny::int64(-1),          // window_left
            TVMFFIAny::float64(0.0),       // logits_soft_cap
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::dltensor(&dl_eq),
            TVMFFIAny::dltensor(&dl_ek),
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("Decode plan failed");

        // Run
        let q_s = [batch_size, num_qo_heads, head_dim];
        let q_st = contiguous_strides(&q_s);
        let o_s = q_s;
        let o_st = q_st.clone();
        let kv_s = [total_pages, page_size, num_kv_heads, head_dim];
        let kv_st = contiguous_strides(&kv_s);
        let kvi_s = [kv_indices.len() as i64];
        let kvi_st = contiguous_strides(&kvi_s);
        let kv_indptr_gpu = gpu_upload(&kv_indptr);
        let kv_last_gpu = gpu_upload(&kv_last_page_len);
        let kvlp_s = [batch_size];
        let kvlp_st = contiguous_strides(&kvlp_s);

        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &o_s, &o_st);
        let dl_k = gpu_dl(k_cache, BF16_DT, &kv_s, &kv_st);
        let dl_v = gpu_dl(v_cache, BF16_DT, &kv_s, &kv_st);
        let dl_kvi = gpu_dl(kvi_ptr, I32_DT, &kvi_s, &kvi_st);
        let kv_indptr_s = [batch_size + 1];
        let kv_indptr_st = contiguous_strides(&kv_indptr_s);
        let dl_kv_indptr = gpu_dl(kv_indptr_gpu, I32_DT, &kv_indptr_s, &kv_indptr_st);
        let dl_kv_last = gpu_dl(kv_last_gpu, I32_DT, &kvlp_s, &kvlp_st);

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        // Run signature: float_ws, int_ws, plan_info, q, paged_k, paged_v,
        //   kv_indptr, kv_indices, kv_last_page_len, o, maybe_lse,
        //   layout, window_left, enable_pdl, [additional: alibi, softcap, sm_scale, rope...]
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k),          // paged_k_cache
            TVMFFIAny::dltensor(&dl_v),           // paged_v_cache
            TVMFFIAny::dltensor(&dl_kv_indptr),
            TVMFFIAny::dltensor(&dl_kvi),
            TVMFFIAny::dltensor(&dl_kv_last),
            TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),              // maybe_lse
            TVMFFIAny::int64(0),            // layout = NHD
            TVMFFIAny::int64(-1),           // window_left
            TVMFFIAny::bool_val(false),     // enable_pdl
            TVMFFIAny::none(),              // alibi_slopes
            TVMFFIAny::float64(0.0),        // logits_soft_cap
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),        // rope_rcp_scale
            TVMFFIAny::float64(1e4),        // rope_rcp_theta
        ];
        reg.call(variant.run, &run_args)
            .expect("Decode run failed");
        cudaDeviceSynchronize();

        // Verify: output should be nonzero
        let out_bf16 = gpu_download::<u16>(o_ptr, q_elems);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Decode output is all zeros");

        // Verify: no NaN/Inf
        assert!(
            out_f32.iter().all(|v| v.is_finite()),
            "Decode output contains NaN or Inf"
        );

        println!("Decode paged BF16: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(o_ptr);
        cudaFree(k_cache); cudaFree(v_cache);
        cudaFree(kvi_ptr); cudaFree(kv_indptr_gpu); cudaFree(kv_last_gpu);
    }
}

#[test]
fn rmsnorm_utility() {
    let reg = KernelRegistry::new();
    let rmsnorm = reg.get_utility("rmsnorm").expect("rmsnorm not found");

    let hidden = 256i64;
    let tokens = 4i64;

    // Generate input and weight
    let n = (tokens * hidden) as usize;
    let input_bf16: Vec<u16> = (0..n).map(|i| f32_to_bf16(0.5 + 0.01 * (i as f32 % 20.0))).collect();
    let weight_bf16: Vec<u16> = (0..hidden as usize).map(|i| f32_to_bf16(1.0 + 0.001 * i as f32)).collect();

    unsafe {
        let input_ptr = gpu_upload(&input_bf16);
        let weight_ptr = gpu_upload(&weight_bf16);
        let output_ptr = gpu_alloc(n * 2);

        let in_s = [tokens, hidden];
        let in_st = contiguous_strides(&in_s);
        let w_s = [hidden];
        let w_st = contiguous_strides(&w_s);

        let dl_out = gpu_dl(output_ptr, BF16_DT, &in_s, &in_st);
        let dl_in = gpu_dl(input_ptr, BF16_DT, &in_s, &in_st);
        let dl_w = gpu_dl(weight_ptr, BF16_DT, &w_s, &w_st);

        reg.set_stream(0, std::ptr::null_mut());

        let args = [
            TVMFFIAny::dltensor(&dl_out),
            TVMFFIAny::dltensor(&dl_in),
            TVMFFIAny::dltensor(&dl_w),
            TVMFFIAny::float64(1e-6),    // eps
            TVMFFIAny::bool_val(false),   // enable_pdl
        ];
        reg.call(rmsnorm, &args).expect("rmsnorm failed");
        cudaDeviceSynchronize();

        // Read output and compare with CPU reference
        let out_bf16 = gpu_download::<u16>(output_ptr, n);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let in_f32: Vec<f32> = input_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let w_f32: Vec<f32> = weight_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        // CPU reference: rmsnorm(x) = x / sqrt(mean(x^2) + eps) * weight
        let eps = 1e-6f64;
        let mut max_diff: f32 = 0.0;
        for t in 0..tokens as usize {
            let row = &in_f32[t * hidden as usize..(t + 1) * hidden as usize];
            let mean_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
            let rms = (mean_sq + eps).sqrt();
            for d in 0..hidden as usize {
                let expected = (row[d] as f64 / rms * w_f32[d] as f64) as f32;
                let got = out_f32[t * hidden as usize + d];
                max_diff = max_diff.max((got - expected).abs());
            }
        }
        assert!(max_diff < 0.05, "RMSNorm max_diff={max_diff} exceeds tolerance");
        println!("RMSNorm: PASS (max_diff={max_diff:.6})");

        cudaFree(input_ptr); cudaFree(weight_ptr); cudaFree(output_ptr);
    }
}

#[test]
fn softmax_utility() {
    let reg = KernelRegistry::new();
    let softmax = reg.get_utility("softmax").expect("softmax not found");

    let batch = 4i64;
    let vocab = 128i64;
    let n = (batch * vocab) as usize;

    // Generate logits
    let logits_bf16: Vec<u16> = (0..n).map(|i| f32_to_bf16(0.1 * (i as f32 % 20.0) - 1.0)).collect();

    unsafe {
        let logits_ptr = gpu_upload(&logits_bf16);
        let output_ptr = gpu_alloc(n * 4); // FP32 output
        let ws_ptr = gpu_alloc(8 * 1024 * 1024); // workspace

        let logit_s = [batch, vocab];
        let logit_st = contiguous_strides(&logit_s);
        let out_s = [batch, vocab];
        let out_st = contiguous_strides(&out_s);
        let ws_s = [8 * 1024 * 1024i64];
        let ws_st = [1i64];

        let dl_ws = gpu_dl(ws_ptr, U8_DT, &ws_s, &ws_st);
        let dl_logits = gpu_dl(logits_ptr, BF16_DT, &logit_s, &logit_st);
        let dl_out = gpu_dl(output_ptr, FP32_DT, &out_s, &out_st);

        reg.set_stream(0, std::ptr::null_mut());

        let args = [
            TVMFFIAny::dltensor(&dl_ws),
            TVMFFIAny::dltensor(&dl_logits),
            TVMFFIAny::dltensor(&dl_out),
            TVMFFIAny::none(),             // temperature_arr
            TVMFFIAny::float64(1.0),       // temperature_val
            TVMFFIAny::bool_val(false),    // enable_pdl
        ];
        reg.call(softmax, &args).expect("softmax failed");
        cudaDeviceSynchronize();

        // Read output and check row sums ≈ 1.0
        let out_f32 = gpu_download::<f32>(output_ptr, n);
        for b in 0..batch as usize {
            let row = &out_f32[b * vocab as usize..(b + 1) * vocab as usize];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Softmax row {b} sum={sum}, expected ≈ 1.0");
            assert!(row.iter().all(|&v| v >= 0.0 && v.is_finite()),
                "Softmax row {b} has negative or non-finite values");
        }

        println!("Softmax: PASS");

        cudaFree(logits_ptr); cudaFree(output_ptr); cudaFree(ws_ptr);
    }
}
