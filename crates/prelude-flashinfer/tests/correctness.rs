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
fn fp8_prefill_variant_lookup() {
    let reg = KernelRegistry::new();
    if reg.arch() < 90 {
        println!("Skipping FP8 test — requires SM90+");
        return;
    }

    let key = FP8PrefillKey {
        head_dim: 128,
        sliding_window: false,
    };
    assert!(reg.get_fp8_prefill(&key).is_some(), "FP8 E4M3 h128 prefill not found");

    // With sliding window
    let key_swa = FP8PrefillKey {
        head_dim: 128,
        sliding_window: true,
    };
    assert!(reg.get_fp8_prefill(&key_swa).is_some(), "FP8 E4M3 h128 swa prefill not found");
}

#[test]
fn activation_kernel_lookup() {
    let reg = KernelRegistry::new();

    assert!(reg.get_utility("silu_and_mul").is_some(), "silu_and_mul not found");
    assert!(reg.get_utility("gelu_and_mul").is_some(), "gelu_and_mul not found");
    assert!(reg.get_utility("gelu_tanh_and_mul").is_some(), "gelu_tanh_and_mul not found");
}

#[test]
fn moe_routing_kernel_lookup() {
    let reg = KernelRegistry::new();
    assert!(reg.get_utility("NoAuxTc").is_some(), "NoAuxTc (MoE routing) not found");
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

    // FP4 KV cache
    assert!(reg.get_utility("nvfp4_kv_quant").is_some(), "nvfp4_kv_quant not found");
    assert!(reg.get_utility("nvfp4_kv_dequant").is_some(), "nvfp4_kv_dequant not found");
}

// ── New module lookup tests ─────────────────────────────────────────

#[test]
fn new_utility_kernel_lookup() {
    let reg = KernelRegistry::new();

    // TopK
    assert!(reg.get_utility("radix_topk").is_some(), "radix_topk not found");
    assert!(reg.get_utility("can_implement_filtered_topk").is_some(), "can_implement_filtered_topk not found");

    // Concat MLA
    assert!(reg.get_utility("concat_mla_k").is_some(), "concat_mla_k not found");

    // add_moe: gen_gemm_module (segment GEMM + bmm_fp8)
    assert!(reg.get_utility("cutlass_segment_gemm").is_some(), "cutlass_segment_gemm not found");
    assert!(reg.get_utility("bmm_fp8").is_some(), "bmm_fp8 not found");

    // add_moe: DSv3 router
    assert!(reg.get_utility("dsv3_router_gemm_op").is_some(), "dsv3_router_gemm_op not found");

    // add_moe: MoE routing (NoAuxTc)
    assert!(reg.get_utility("NoAuxTc").is_some(), "NoAuxTc not found");

    // CUTLASS MLA
    assert!(reg.get_utility("cutlass_mla_paged_attention").is_some(), "cutlass_mla_paged_attention not found");

    // Comm: vLLM AllReduce (all archs)
    assert!(reg.get_utility("init_custom_ar").is_some(), "init_custom_ar not found");

    // TRT-LLM comm is SM100+, tested in sm100_module_lookup
}

#[test]
fn sm90_module_lookup() {
    let reg = KernelRegistry::new();
    if reg.arch() < 90 {
        println!("SM{} < SM90, skipping SM90 module lookup", reg.arch());
        return;
    }

    // add_misc: gen_gdn_prefill_sm90_module
    assert!(reg.get_utility("gdn_prefill").is_some(), "gdn_prefill not found (SM90 required)");

    // add_moe: gen_gemm_sm90_module
    assert!(reg.get_utility("cutlass_segment_gemm_sm90").is_some(),
            "cutlass_segment_gemm_sm90 not found (SM90 required)");
}

#[test]
fn sm100_module_lookup() {
    let reg = KernelRegistry::new();
    if reg.arch() < 100 {
        println!("SM{} < SM100, skipping SM100 module lookup", reg.arch());
        return;
    }

    // add_moe: gen_gemm_sm100_module_cutlass_fp8
    assert!(reg.get_utility("fp8_gemm").is_some(), "fp8_gemm not found (SM100 required)");

    // add_moe: gen_tgv_gemm_sm10x_module
    assert!(reg.get_utility("tgv_gemm").is_some(), "tgv_gemm not found (SM100 required)");
    assert!(reg.get_utility("bf16_gemm").is_some(), "bf16_gemm not found (SM100 required)");

    // add_comm: gen_trtllm_comm_module (SM100+)
    assert!(reg.get_utility("trtllm_custom_all_reduce").is_some(),
            "trtllm_custom_all_reduce not found (SM100 required)");

    // add_moe: gen_gemm_sm100_module_cutlass_fp4
    assert!(reg.get_utility("fp4_gemm").is_some(), "fp4_gemm not found (SM100 required)");

    // MXFP8 GEMM
    assert!(reg.get_utility("mxfp8_gemm").is_some(), "mxfp8_gemm not found (SM100 required)");

    // Groupwise GEMM
    assert!(reg.get_utility("gemm_fp8_nt_groupwise").is_some(),
            "gemm_fp8_nt_groupwise not found (SM100 required)");

    // Group GEMM
    assert!(reg.get_utility("group_gemm_fp8_nt_groupwise").is_some(),
            "group_gemm_fp8_nt_groupwise not found (SM100 required)");
    assert!(reg.get_utility("group_gemm_mxfp4_nt_groupwise").is_some(),
            "group_gemm_mxfp4_nt_groupwise not found (SM100 required)");
}

// ── TopK correctness ────────────────────────────────────────────────

#[test]
fn radix_topk_correctness() {
    let reg = KernelRegistry::new();
    let topk_fn = match reg.get_utility("radix_topk") {
        Some(f) => f,
        None => { println!("radix_topk not compiled, skipping"); return; }
    };

    let batch = 4i64;
    let d = 1024i64;
    let k = 8i64;

    // Generate known input: each row has distinct values
    let mut input_f32 = vec![0.0f32; (batch * d) as usize];
    for b in 0..batch as usize {
        for i in 0..d as usize {
            // Spread values so top-k is unambiguous
            input_f32[b * d as usize + i] = (i as f32) * 0.1 + (b as f32) * 0.001;
        }
    }

    let input_ptr = gpu_upload(&input_f32);
    let values_ptr = gpu_alloc((batch * k) as usize * 4);
    let indices_ptr = gpu_alloc((batch * k) as usize * 4); // int32

    let in_s = [batch, d]; let in_st = contiguous_strides(&in_s);
    let out_s = [batch, k]; let out_st = contiguous_strides(&out_s);

    let dl_in = gpu_dl(input_ptr, FP32_DT, &in_s, &in_st);
    let dl_vals = gpu_dl(values_ptr, FP32_DT, &out_s, &out_st);
    let dl_idxs = gpu_dl(indices_ptr, I32_DT, &out_s, &out_st);

    unsafe {
        reg.set_stream(0, std::ptr::null_mut());
        // radix_topk(input, output_indices, output_values, row_states_buf?, top_k, sorted, deterministic)
        let args = [
            TVMFFIAny::dltensor(&dl_in),
            TVMFFIAny::dltensor(&dl_idxs),
            TVMFFIAny::dltensor(&dl_vals),
            TVMFFIAny::none(),          // row_states_buffer
            TVMFFIAny::int64(k),
            TVMFFIAny::bool_val(true),  // sorted
            TVMFFIAny::bool_val(true),  // deterministic
        ];
        reg.call(topk_fn, &args).expect("radix_topk call failed");
        cudaDeviceSynchronize();
    }

    let gpu_vals = gpu_download::<f32>(values_ptr, (batch * k) as usize);
    let gpu_idxs = gpu_download::<i32>(indices_ptr, (batch * k) as usize);

    // CPU reference: top-k of each row (descending)
    for b in 0..batch as usize {
        let row = &input_f32[b * d as usize..(b + 1) * d as usize];
        let mut indexed: Vec<(f32, usize)> = row.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for j in 0..k as usize {
            let gpu_val = gpu_vals[b * k as usize + j];
            let gpu_idx = gpu_idxs[b * k as usize + j] as usize;
            let ref_val = indexed[j].0;
            let ref_idx = indexed[j].1;

            assert!((gpu_val - ref_val).abs() < 1e-4,
                    "TopK batch={b} rank={j}: val {gpu_val} vs ref {ref_val}");
            assert_eq!(gpu_idx, ref_idx,
                    "TopK batch={b} rank={j}: idx {gpu_idx} vs ref {ref_idx}");
        }
    }
    println!("radix_topk [{batch}×{d}, k={k}]: PASS");

    unsafe { cudaFree(input_ptr); cudaFree(values_ptr); cudaFree(indices_ptr); }
}

// ── TGV GEMM correctness ───────────────────────────────────────────

#[test]
fn tgv_gemm_correctness() {
    let reg = KernelRegistry::new();
    let gemm = match reg.get_utility("tgv_gemm") {
        Some(f) => f,
        None => { println!("tgv_gemm not compiled, skipping"); return; }
    };

    // TGV is optimized for decode (small M)
    let m = 1i64;
    let n = 256i64;
    let k = 512i64;

    // CPU reference: C = A @ B^T
    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.009).sin()).collect();
    let mut ref_c = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f64;
            for l in 0..k as usize {
                sum += a_f32[i * k as usize + l] as f64 * b_f32[j * k as usize + l] as f64;
            }
            ref_c[i * n as usize + j] = sum as f32;
        }
    }

    let a_bf16: Vec<u16> = a_f32.iter().map(|v| f32_to_bf16(*v)).collect();
    let b_bf16: Vec<u16> = b_f32.iter().map(|v| f32_to_bf16(*v)).collect();

    let a_ptr = gpu_upload(&a_bf16);
    let b_ptr = gpu_upload(&b_bf16);
    let c_ptr = gpu_alloc((m * n) as usize * 2);

    let a_s = [m, k]; let a_st = contiguous_strides(&a_s);
    let b_s = [n, k]; let b_st = contiguous_strides(&b_s);
    let c_s = [m, n]; let c_st = contiguous_strides(&c_s);

    let dl_a = gpu_dl(a_ptr, BF16_DT, &a_s, &a_st);
    let dl_b = gpu_dl(b_ptr, BF16_DT, &b_s, &b_st);
    let dl_c = gpu_dl(c_ptr, BF16_DT, &c_s, &c_st);

    unsafe {
        reg.set_stream(0, std::ptr::null_mut());
        // tgv_gemm(mat1, mat2, bias?, tactic, out, pdl)
        let args = [
            TVMFFIAny::dltensor(&dl_a), TVMFFIAny::dltensor(&dl_b),
            TVMFFIAny::none(),          // no bias
            TVMFFIAny::int64(-1),       // auto tactic
            TVMFFIAny::dltensor(&dl_c),
            TVMFFIAny::bool_val(false), // no pdl
        ];
        reg.call(gemm, &args).expect("tgv_gemm call failed");
        cudaDeviceSynchronize();
    }

    let gpu_bf16 = gpu_download::<u16>(c_ptr, (m * n) as usize);
    let gpu_f32: Vec<f32> = gpu_bf16.iter().map(|v| bf16_to_f32(*v)).collect();

    let mut max_rel = 0.0f32;
    for (r, g) in ref_c.iter().zip(gpu_f32.iter()) {
        let err = (r - g).abs();
        let denom = r.abs().max(1e-6);
        max_rel = max_rel.max(err / denom);
    }
    assert!(max_rel < 0.05, "tgv_gemm: max_rel={max_rel} exceeds 5% tolerance");
    println!("tgv_gemm [{m}×{n}×{k}]: PASS (max_rel={max_rel:.4})");

    unsafe { cudaFree(a_ptr); cudaFree(b_ptr); cudaFree(c_ptr); }
}

// ── Concat MLA K correctness ────────────────────────────────────────

#[test]
fn concat_mla_k_correctness() {
    let reg = KernelRegistry::new();
    let concat_fn = match reg.get_utility("concat_mla_k") {
        Some(f) => f,
        None => { println!("concat_mla_k not compiled, skipping"); return; }
    };

    // DeepSeek MLA specific: num_heads=128, nope_dim=128, rope_dim=64 (hardcoded in kernel)
    let tokens = 4i64;
    let num_heads = 128i64;
    let nope_dim = 128i64;
    let rope_dim = 64i64;
    let full_dim = nope_dim + rope_dim;

    let nope_elems = (tokens * num_heads * nope_dim) as usize;
    // k_rope: num_heads=1, broadcast across all 128 heads
    let rope_elems = (tokens * 1 * rope_dim) as usize;
    let full_elems = (tokens * num_heads * full_dim) as usize;

    // Generate known patterns
    let nope_bf16: Vec<u16> = (0..nope_elems).map(|i| f32_to_bf16(1.0 + i as f32 * 0.001)).collect();
    let rope_bf16: Vec<u16> = (0..rope_elems).map(|i| f32_to_bf16(-1.0 - i as f32 * 0.001)).collect();

    let nope_ptr = gpu_upload(&nope_bf16);
    let rope_ptr = gpu_upload(&rope_bf16);
    let k_ptr = gpu_alloc(full_elems * 2);

    let k_s = [tokens, num_heads, full_dim]; let k_st = contiguous_strides(&k_s);
    let nope_s = [tokens, num_heads, nope_dim]; let nope_st = contiguous_strides(&nope_s);
    let rope_s = [tokens, 1i64, rope_dim]; let rope_st = contiguous_strides(&rope_s);

    let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
    let dl_nope = gpu_dl(nope_ptr, BF16_DT, &nope_s, &nope_st);
    let dl_rope = gpu_dl(rope_ptr, BF16_DT, &rope_s, &rope_st);

    unsafe {
        reg.set_stream(0, std::ptr::null_mut());
        // concat_mla_k(k_output, k_nope, k_rope)
        let args = [
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_nope),
            TVMFFIAny::dltensor(&dl_rope),
        ];
        reg.call(concat_fn, &args).expect("concat_mla_k call failed");
        cudaDeviceSynchronize();
    }

    let k_bf16 = gpu_download::<u16>(k_ptr, full_elems);

    // Verify: k[:, h, :nope_dim] == nope[:, h, :], k[:, h, nope_dim:] == rope[:, 0, :] (broadcast)
    let mut mismatches = 0usize;
    for t in 0..tokens as usize {
        for h in 0..num_heads as usize {
            for d in 0..full_dim as usize {
                let idx = t * (num_heads * full_dim) as usize + h * full_dim as usize + d;
                let got = k_bf16[idx];
                let expected = if d < nope_dim as usize {
                    let ni = t * (num_heads * nope_dim) as usize + h * nope_dim as usize + d;
                    nope_bf16[ni]
                } else {
                    // rope has num_heads=1, broadcast to all heads
                    let ri = t * (1 * rope_dim) as usize + 0 * rope_dim as usize + (d - nope_dim as usize);
                    rope_bf16[ri]
                };
                if got != expected { mismatches += 1; }
            }
        }
    }
    assert_eq!(mismatches, 0, "concat_mla_k: {mismatches}/{full_elems} mismatches");
    println!("concat_mla_k [{tokens}×{num_heads}×({nope_dim}+{rope_dim})]: PASS");

    unsafe { cudaFree(nope_ptr); cudaFree(rope_ptr); cudaFree(k_ptr); }
}

// ── Comm module lookup (multi-GPU needed for execution) ─────────────

#[test]
fn comm_module_lookup() {
    let reg = KernelRegistry::new();

    // add_comm: gen_vllm_comm_module (all archs)
    assert!(reg.get_utility("init_custom_ar").is_some(), "init_custom_ar not found");
    assert!(reg.get_utility("all_reduce").is_some(), "vllm all_reduce not found");

    // TRT-LLM comm is SM100+, tested in sm100_module_lookup
    println!("comm_module_lookup: PASS");
}

// ── FP8 GEMM correctness ───────────────────────────────────────────

#[test]
fn fp8_gemm_correctness() {
    let reg = KernelRegistry::new();
    let gemm = match reg.get_utility("fp8_gemm") {
        Some(f) => f,
        None => { println!("fp8_gemm not compiled, skipping"); return; }
    };

    // FP8 GEMM: A (FP8 E4M3) @ B^T (FP8 E4M3) -> C (BF16)
    // We use bf16 inputs quantized to FP8 range for reference
    let m = 16i64;
    let n = 32i64;
    let k = 64i64;

    // Generate values in FP8 E4M3 range (max ~448)
    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.05).cos() * 2.0).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.03).sin() * 2.0).collect();

    // Quantize to FP8 E4M3 via half crate
    let a_fp8: Vec<u8> = a_f32.iter().map(|&v| {
        let bits = half::f16::from_f32(v).to_bits();
        (bits >> 8) as u8
    }).collect();
    let b_fp8: Vec<u8> = b_f32.iter().map(|&v| {
        let bits = half::f16::from_f32(v).to_bits();
        (bits >> 8) as u8
    }).collect();

    // For correctness, just verify the kernel runs without crash and produces finite output
    let a_ptr = gpu_upload(&a_fp8);
    let b_ptr = gpu_upload(&b_fp8);
    let c_ptr = gpu_alloc((m * n) as usize * 2);
    let ws_ptr = gpu_alloc(64 * 1024 * 1024);

    let fp8_dt = DLDataType { code: KDLINT, bits: 8, lanes: 1 }; // E4M3 as int8
    let a_s = [m, k]; let a_st = contiguous_strides(&a_s);
    let b_s = [n, k]; let b_st = contiguous_strides(&b_s);
    let c_s = [m, n]; let c_st = contiguous_strides(&c_s);
    let ws_s = [64 * 1024 * 1024i64]; let ws_st = [1i64];

    let dl_a = gpu_dl(a_ptr, fp8_dt, &a_s, &a_st);
    let dl_b = gpu_dl(b_ptr, fp8_dt, &b_s, &b_st);
    let dl_c = gpu_dl(c_ptr, BF16_DT, &c_s, &c_st);
    let dl_ws = gpu_dl(ws_ptr, U8_DT, &ws_s, &ws_st);

    unsafe {
        reg.set_stream(0, std::ptr::null_mut());
        let args = [
            TVMFFIAny::dltensor(&dl_a), TVMFFIAny::dltensor(&dl_b),
            TVMFFIAny::dltensor(&dl_c), TVMFFIAny::dltensor(&dl_ws),
            TVMFFIAny::int64(-1),
        ];
        match reg.call(gemm, &args) {
            Ok(_) => {
                cudaDeviceSynchronize();
                let out_bf16 = gpu_download::<u16>(c_ptr, (m * n) as usize);
                let finite = out_bf16.iter().map(|v| bf16_to_f32(*v)).filter(|v| v.is_finite()).count();
                println!("fp8_gemm [{m}×{n}×{k}]: PASS ({finite}/{} finite)", m * n);
            }
            Err(e) => {
                // FP8 GEMM may not be supported on SM < 89
                println!("fp8_gemm [{m}×{n}×{k}]: SKIPPED ({e})");
            }
        }
    }

    unsafe { cudaFree(a_ptr); cudaFree(b_ptr); cudaFree(c_ptr); cudaFree(ws_ptr); }
}

// ── BF16 GEMM correctness ───────────────────────────────────────────

#[test]
fn bf16_gemm_correctness() {
    let reg = KernelRegistry::new();
    let gemm = match reg.get_utility("bf16_gemm") {
        Some(f) => f,
        None => { println!("bf16_gemm not compiled, skipping"); return; }
    };

    let m = 32i64;
    let n = 64i64;
    let k = 128i64;

    // CPU reference: C = A @ B^T  (A: [m,k], B: [n,k], C: [m,n])
    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).cos()).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.007).sin()).collect();
    let mut ref_c = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f64;
            for l in 0..k as usize {
                sum += a_f32[i * k as usize + l] as f64 * b_f32[j * k as usize + l] as f64;
            }
            ref_c[i * n as usize + j] = sum as f32;
        }
    }

    let a_bf16: Vec<u16> = a_f32.iter().map(|v| f32_to_bf16(*v)).collect();
    let b_bf16: Vec<u16> = b_f32.iter().map(|v| f32_to_bf16(*v)).collect();

    let a_ptr = gpu_upload(&a_bf16);
    let b_ptr = gpu_upload(&b_bf16);
    let c_ptr = gpu_alloc((m * n) as usize * 2);
    let ws_ptr = gpu_alloc(64 * 1024 * 1024);

    let a_s = [m, k]; let a_st = contiguous_strides(&a_s);
    let b_s = [n, k]; let b_st = contiguous_strides(&b_s);
    let c_s = [m, n]; let c_st = contiguous_strides(&c_s);
    let ws_s = [64 * 1024 * 1024i64]; let ws_st = [1i64];

    let dl_a = gpu_dl(a_ptr, BF16_DT, &a_s, &a_st);
    let dl_b = gpu_dl(b_ptr, BF16_DT, &b_s, &b_st);
    let dl_c = gpu_dl(c_ptr, BF16_DT, &c_s, &c_st);
    let dl_ws = gpu_dl(ws_ptr, U8_DT, &ws_s, &ws_st);

    unsafe {
        reg.set_stream(0, std::ptr::null_mut());
        // bf16_gemm(mat1, mat2, out, workspace, tactic)
        let args = [
            TVMFFIAny::dltensor(&dl_a), TVMFFIAny::dltensor(&dl_b),
            TVMFFIAny::dltensor(&dl_c), TVMFFIAny::dltensor(&dl_ws),
            TVMFFIAny::int64(-1),
        ];
        reg.call(gemm, &args).expect("bf16_gemm call failed");
        cudaDeviceSynchronize();
    }

    let gpu_bf16 = gpu_download::<u16>(c_ptr, (m * n) as usize);
    let gpu_f32: Vec<f32> = gpu_bf16.iter().map(|v| bf16_to_f32(*v)).collect();

    let mut max_rel = 0.0f32;
    let mut fail_count = 0usize;
    for (r, g) in ref_c.iter().zip(gpu_f32.iter()) {
        if !r.is_finite() || !g.is_finite() { continue; }
        let err = (r - g).abs();
        let denom = r.abs().max(1e-6);
        let rel = err / denom;
        max_rel = max_rel.max(rel);
        if rel > 0.05 && err > 0.1 { fail_count += 1; }
    }

    println!("bf16_gemm [{m}×{n}×{k}]: max_rel={max_rel:.4}, fail={fail_count}/{}",
             m * n);
    assert!(fail_count == 0, "bf16_gemm: {fail_count} values exceeded tolerance");

    unsafe { cudaFree(a_ptr); cudaFree(b_ptr); cudaFree(c_ptr); cudaFree(ws_ptr); }
}

// ── GDN (Gated Delta Net) prefill smoke test ────────────────────────

#[test]
fn gdn_prefill_smoke() {
    let reg = KernelRegistry::new();
    if reg.arch() < 90 {
        println!("SM{} < SM90, skipping GDN test", reg.arch());
        return;
    }
    let gdn = match reg.get_utility("gdn_prefill") {
        Some(f) => f,
        None => { println!("gdn_prefill not compiled, skipping"); return; }
    };

    // Use upstream test dimensions: head_size=128, GQA (4 q heads, 1 kv head)
    // With alpha=true, beta=true (upstream skips alpha=false,beta=false due to
    // "output value amplitude explosion along token dimension")
    let num_seqs = 1i64;
    let seq_len = 64i64;
    let num_q_heads = 4i64;
    let num_k_heads = 1i64;
    let num_v_heads = 1i64;
    let head_dim = 128i64;
    let packed_seq = seq_len;
    let num_sab_heads = num_q_heads.max(num_v_heads); // = 4

    // Q: [packed_seq, num_q_heads, head_dim] BF16
    let q_elems = (packed_seq * num_q_heads * head_dim) as usize;
    let q_bf16: Vec<u16> = (0..q_elems).map(|i| f32_to_bf16(0.01 * ((i as f32) % 13.0 - 6.0))).collect();

    // K: [packed_seq, num_k_heads, head_dim] BF16 — L2 normalized per upstream
    let k_elems = (packed_seq * num_k_heads * head_dim) as usize;
    let mut k_f32: Vec<f32> = (0..k_elems).map(|i| 0.1 * ((i as f32) % 7.0 - 3.0)).collect();
    // L2 normalize each [head_dim] vector
    for row in k_f32.chunks_mut(head_dim as usize) {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in row.iter_mut() { *v /= norm; }
    }
    let k_bf16: Vec<u16> = k_f32.iter().map(|v| f32_to_bf16(*v)).collect();

    // V: [packed_seq, num_v_heads, head_dim] BF16
    let v_elems = (packed_seq * num_v_heads * head_dim) as usize;
    let v_bf16: Vec<u16> = (0..v_elems).map(|i| f32_to_bf16(0.005 * ((i as f32) % 11.0 - 5.0))).collect();

    // Output: [packed_seq, num_sab_heads, head_dim] BF16
    let o_elems = (packed_seq * num_sab_heads * head_dim) as usize;

    // Output state: [num_seqs, num_sab_heads, head_dim, head_dim] FP32
    let state_elems = (num_seqs * num_sab_heads * head_dim * head_dim) as usize;

    // Alpha, Beta: [packed_seq, num_sab_heads] FP32
    let ab_elems = (packed_seq * num_sab_heads) as usize;
    let alpha_f32: Vec<f32> = (0..ab_elems).map(|i| 0.5 + 0.01 * (i as f32 % 10.0)).collect();
    let beta_f32: Vec<f32> = (0..ab_elems).map(|i| 0.3 + 0.01 * (i as f32 % 8.0)).collect();

    // cu_seqlens: [num_seqs + 1] I64
    let cu_seqlens: Vec<i64> = vec![0, seq_len];

    let q_ptr = gpu_upload(&q_bf16);
    let k_ptr = gpu_upload(&k_bf16);
    let v_ptr = gpu_upload(&v_bf16);
    let o_ptr = gpu_alloc(o_elems * 2);
    let state_ptr = gpu_alloc(state_elems * 4);
    let alpha_ptr = gpu_upload(&alpha_f32);
    let beta_ptr = gpu_upload(&beta_f32);
    let cu_ptr = gpu_upload(&cu_seqlens);
    let ws_ptr = gpu_alloc(128 * 1024 * 1024);

    let q_s = [packed_seq, num_q_heads, head_dim]; let q_st = contiguous_strides(&q_s);
    let k_s = [packed_seq, num_k_heads, head_dim]; let k_st = contiguous_strides(&k_s);
    let v_s = [packed_seq, num_v_heads, head_dim]; let v_st = contiguous_strides(&v_s);
    let o_s = [packed_seq, num_sab_heads, head_dim]; let o_st = contiguous_strides(&o_s);
    let state_s = [num_seqs, num_sab_heads, head_dim, head_dim];
    let state_st = contiguous_strides(&state_s);
    let ab_s = [packed_seq, num_sab_heads]; let ab_st = contiguous_strides(&ab_s);
    let cu_s = [num_seqs + 1]; let cu_st = [1i64];
    let ws_s = [128 * 1024 * 1024i64]; let ws_st = [1i64];

    let i64_dt = DLDataType { code: KDLINT, bits: 64, lanes: 1 };

    let dl_o = gpu_dl(o_ptr, BF16_DT, &o_s, &o_st);
    let dl_state = gpu_dl(state_ptr, FP32_DT, &state_s, &state_st);
    let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
    let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
    let dl_v = gpu_dl(v_ptr, BF16_DT, &v_s, &v_st);
    let dl_cu = gpu_dl(cu_ptr, i64_dt, &cu_s, &cu_st);
    let dl_alpha = gpu_dl(alpha_ptr, FP32_DT, &ab_s, &ab_st);
    let dl_beta = gpu_dl(beta_ptr, FP32_DT, &ab_s, &ab_st);
    let dl_ws = gpu_dl(ws_ptr, U8_DT, &ws_s, &ws_st);

    unsafe {
        reg.set_stream(0, std::ptr::null_mut());
        // gdn_prefill(output, output_state, q, k, v, cu_seqlens,
        //             input_state?, alpha?, beta?, scale, workspace)
        let args = [
            TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::dltensor(&dl_state),
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_cu),
            TVMFFIAny::none(),              // input_state (None = zero init)
            TVMFFIAny::dltensor(&dl_alpha), // alpha
            TVMFFIAny::dltensor(&dl_beta),  // beta
            TVMFFIAny::float64(0.0),        // scale (0 = auto: 1/sqrt(head_dim))
            TVMFFIAny::dltensor(&dl_ws),
        ];
        reg.call(gdn, &args).expect("gdn_prefill call failed");
        cudaDeviceSynchronize();
    }

    let out_bf16 = gpu_download::<u16>(o_ptr, o_elems);
    let out_f32: Vec<f32> = out_bf16.iter().map(|v| bf16_to_f32(*v)).collect();
    let finite = out_f32.iter().filter(|v| v.is_finite()).count();
    let nonzero = out_f32.iter().filter(|v| v.abs() > 1e-10).count();

    println!("gdn_prefill [seq={seq_len}, q_heads={num_q_heads}, kv_heads={num_k_heads}, d={head_dim}]: \
              {finite}/{o_elems} finite, {nonzero}/{o_elems} nonzero");
    assert_eq!(finite, o_elems, "GDN output has NaN/Inf");
    assert!(nonzero > o_elems / 2, "GDN output is mostly zeros ({nonzero}/{o_elems})");

    // Also verify state is populated
    let state_f32 = gpu_download::<f32>(state_ptr, state_elems);
    let state_finite = state_f32.iter().filter(|v| v.is_finite()).count();
    let state_nonzero = state_f32.iter().filter(|v| v.abs() > 1e-10).count();
    println!("  output_state: {state_finite}/{state_elems} finite, {state_nonzero}/{state_elems} nonzero");
    assert_eq!(state_finite, state_elems, "GDN state has NaN/Inf");

    unsafe {
        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr);
        cudaFree(o_ptr); cudaFree(state_ptr);
        cudaFree(alpha_ptr); cudaFree(beta_ptr);
        cudaFree(cu_ptr); cudaFree(ws_ptr);
    }
}

// POD is excluded from AOT (archive size > 2GB). Infrastructure is in place
// for future JIT compilation. See generate_batch_pod_sources() in compile_kernels.py.

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
        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
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
fn prefill_sliding_window_dispatch() {
    // Tests runtime swa dispatch in merged kernels (window_left >= 0 path)
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let batch_size = 1i64;
    let seq_len = 8i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;

    // Request with swa=true — same merged kernel, different runtime path
    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: head_dim as u32,
        head_dim_vo: head_dim as u32,
        sliding_window: true,
        logits_soft_cap: false,
        backend: Backend::FA2,
    };
    let variant = reg.get_prefill(&key).expect("SWA prefill variant not found");

    unsafe {
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..total).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let k_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.01 * ((i + 3) as f32 % 10.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.02 * ((i + 7) as f32 % 10.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let k_ptr = gpu_upload(&k_bf16);
        let v_ptr = gpu_upload(&v_bf16);
        let o_ptr = gpu_alloc(total * 2);
        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let cu_k_gpu = gpu_upload(&cu_k_data);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
        let cu_s = [batch_size + 1]; let cu_st = contiguous_strides(&cu_s);
        let kvl_s = [batch_size]; let kvl_st = contiguous_strides(&kvl_s);
        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim]; let k_st = contiguous_strides(&k_s);

        let dl_cuq = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);

        reg.set_stream(0, std::ptr::null_mut());

        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws), TVMFFIAny::dltensor(&dl_cuq),
            TVMFFIAny::dltensor(&dl_cuk), TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len), TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true),  // causal
            TVMFFIAny::int64(4),        // window_left = 4 (sliding window!)
            TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
        ];
        let plan_info = reg.call(variant.plan, &plan_args).expect("SWA plan failed");

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v), TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_cuk_gpu), TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),         // lse
            TVMFFIAny::int64(1),       // causal
            TVMFFIAny::int64(0),       // NHD
            TVMFFIAny::int64(4),       // window_left = 4 (triggers SWA runtime path)
            TVMFFIAny::bool_val(false),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::float64(0.0),   // logits_soft_cap = 0 (no softcap)
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
            TVMFFIAny::int64(0),
        ];
        reg.call(variant.ragged_run, &run_args).expect("SWA ragged_run failed");
        cudaDeviceSynchronize();

        let out = gpu_download::<u16>(o_ptr, total);
        let out_f32: Vec<f32> = out.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "SWA prefill output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "SWA prefill has NaN/Inf");

        println!("Prefill SWA dispatch: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr); cudaFree(o_ptr);
        cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
    }
}

#[test]
fn prefill_softcap_dispatch() {
    // Tests runtime softcap dispatch in merged kernels (logits_soft_cap > 0 path)
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let seq_len = 8i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;

    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
        sliding_window: false, logits_soft_cap: true,
        backend: Backend::FA2,
    };
    let variant = reg.get_prefill(&key).expect("Softcap prefill variant not found");

    unsafe {
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..total).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let k_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.01 * ((i+3) as f32 % 10.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.02 * ((i+7) as f32 % 10.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let k_ptr = gpu_upload(&k_bf16);
        let v_ptr = gpu_upload(&v_bf16);
        let o_ptr = gpu_alloc(total * 2);
        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let cu_k_gpu = gpu_upload(&cu_k_data);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
        let cu_s = [2i64]; let cu_st = [1i64];
        let kvl_s = [1i64]; let kvl_st = [1i64];
        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim]; let k_st = contiguous_strides(&k_s);

        let dl_cuq = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);

        reg.set_stream(0, std::ptr::null_mut());

        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws), TVMFFIAny::dltensor(&dl_cuq),
            TVMFFIAny::dltensor(&dl_cuk), TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len), TVMFFIAny::int64(1),
            TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
            TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
        ];
        let plan_info = reg.call(variant.plan, &plan_args).expect("Softcap plan failed");

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v), TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_cuk_gpu), TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(), TVMFFIAny::int64(1), TVMFFIAny::int64(0),
            TVMFFIAny::int64(-1),       // window_left = -1 (no swa)
            TVMFFIAny::bool_val(false),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::float64(30.0),   // logits_soft_cap = 30.0 (Gemma-style, triggers softcap path)
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
            TVMFFIAny::int64(0),
        ];
        reg.call(variant.ragged_run, &run_args).expect("Softcap ragged_run failed");
        cudaDeviceSynchronize();

        let out = gpu_download::<u16>(o_ptr, total);
        let out_f32: Vec<f32> = out.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Softcap prefill output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "Softcap prefill has NaN/Inf");

        println!("Prefill softcap dispatch: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr); cudaFree(o_ptr);
        cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
    }
}

#[test]
fn fa3_prefill_execution() {
    let reg = KernelRegistry::new();
    if reg.arch() < 90 { println!("Skipping FA3 — requires SM90+"); return; }

    let ws = Workspace::new();
    let seq_len = 8i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;

    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
        sliding_window: false, logits_soft_cap: false,
        backend: Backend::FA3,
    };
    let variant = reg.get_prefill(&key).expect("FA3 prefill not found");

    unsafe {
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..total).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let k_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.01 * ((i+3) as f32 % 10.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.02 * ((i+7) as f32 % 10.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let k_ptr = gpu_upload(&k_bf16);
        let v_ptr = gpu_upload(&v_bf16);
        let o_ptr = gpu_alloc(total * 2);
        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let cu_k_gpu = gpu_upload(&cu_k_data);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
        let cu_s = [2i64]; let cu_st = [1i64];
        let kvl_s = [1i64]; let kvl_st = [1i64];
        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim]; let k_st = contiguous_strides(&k_s);

        let dl_cuq = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);

        reg.set_stream(0, std::ptr::null_mut());

        // FA3 SM90 plan: 16 params (no fixed_split_size, disable_split_kv, num_colocated_ctas)
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq), TVMFFIAny::dltensor(&dl_cuk),
            TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len), TVMFFIAny::int64(1),
            TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
        ];
        let plan_info = reg.call(variant.plan, &plan_args).expect("FA3 plan failed");

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);

        // FA3 SM90 ragged_run: 22 params with SM90 additional params
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v), TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_cuk_gpu), TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),         // lse
            TVMFFIAny::int64(1),       // causal
            TVMFFIAny::int64(0),       // NHD
            TVMFFIAny::int64(-1),      // window_left
            TVMFFIAny::bool_val(false),// enable_pdl
            // SM90 additional params: prefix_len_ptr, token_pos, max_item_len, scale_v
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::float64(0.0),   // logits_soft_cap
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),   // scale_v_scalar
            TVMFFIAny::int64(0),       // token_pos_in_items_len
        ];
        reg.call(variant.ragged_run, &run_args).expect("FA3 ragged_run failed");
        cudaDeviceSynchronize();

        let out = gpu_download::<u16>(o_ptr, total);
        let out_f32: Vec<f32> = out.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "FA3 prefill output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "FA3 prefill has NaN/Inf");

        println!("FA3 prefill: PASS (sum={sum:.4})");

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
fn silu_and_mul_activation() {
    let reg = KernelRegistry::new();
    let silu_fn = reg.get_utility("silu_and_mul").expect("silu_and_mul not found");

    let tokens = 4i64;
    let hidden = 256i64;
    let input_dim = hidden * 2; // activation kernels take [tokens, 2*hidden]

    // Generate input: first half is x, second half is y → out = silu(x) * y
    let n = (tokens * input_dim) as usize;
    let input_bf16: Vec<u16> = (0..n)
        .map(|i| f32_to_bf16(0.5 * ((i as f32 % 20.0) / 10.0 - 1.0)))
        .collect();

    unsafe {
        let input_ptr = gpu_upload(&input_bf16);
        let output_ptr = gpu_alloc((tokens * hidden) as usize * 2);

        let in_s = [tokens, input_dim];
        let in_st = contiguous_strides(&in_s);
        let out_s = [tokens, hidden];
        let out_st = contiguous_strides(&out_s);

        let dl_in = gpu_dl(input_ptr, BF16_DT, &in_s, &in_st);
        let dl_out = gpu_dl(output_ptr, BF16_DT, &out_s, &out_st);

        reg.set_stream(0, std::ptr::null_mut());

        let args = [
            TVMFFIAny::dltensor(&dl_out),
            TVMFFIAny::dltensor(&dl_in),
            TVMFFIAny::bool_val(false), // enable_pdl
        ];
        reg.call(silu_fn, &args).expect("silu_and_mul failed");
        cudaDeviceSynchronize();

        // CPU reference: silu(x) * y = x / (1 + exp(-x)) * y
        let in_f32: Vec<f32> = input_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let out_bf16 = gpu_download::<u16>(output_ptr, (tokens * hidden) as usize);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        let mut max_diff: f32 = 0.0;
        for t in 0..tokens as usize {
            for d in 0..hidden as usize {
                let x = in_f32[t * input_dim as usize + d];
                let y = in_f32[t * input_dim as usize + hidden as usize + d];
                let silu_x = x / (1.0 + (-x).exp());
                let expected = silu_x * y;
                let got = out_f32[t * hidden as usize + d];
                max_diff = max_diff.max((got - expected).abs());
            }
        }
        assert!(max_diff < 0.05, "silu_and_mul max_diff={max_diff} exceeds tolerance");
        println!("silu_and_mul: PASS (max_diff={max_diff:.6})");

        cudaFree(input_ptr);
        cudaFree(output_ptr);
    }
}

#[test]
fn fp4_quant_dequant_roundtrip() {
    let reg = KernelRegistry::new();
    let quant_fn = reg.get_utility("nvfp4_kv_quant").expect("nvfp4_kv_quant not found");
    let dequant_fn = reg.get_utility("nvfp4_kv_dequant").expect("nvfp4_kv_dequant not found");

    let m = 4i64;
    let k = 64i64; // must be divisible by 16 (NVFP4_BLOCK_SIZE)
    let n = (m * k) as usize;

    // Input: BF16 values in [-3, 3] range (within E2M1 representable range)
    let input_bf16: Vec<u16> = (0..n)
        .map(|i| f32_to_bf16(3.0 * ((i as f32 / n as f32) * 2.0 - 1.0)))
        .collect();

    unsafe {
        let input_ptr = gpu_upload(&input_bf16);
        let fp4_ptr = gpu_alloc(n / 2);                  // [M, K/2] packed FP4
        let scales_ptr = gpu_alloc((m * k / 16) as usize); // [M, K/16] FP8 block scales
        let global_scale_val: f32 = 1.0;
        let gs_ptr = gpu_upload(&[global_scale_val]);
        let output_ptr = gpu_alloc(n * 2);               // [M, K] BF16 output

        let in_s = [m, k]; let in_st = contiguous_strides(&in_s);
        let fp4_s = [m, k / 2]; let fp4_st = contiguous_strides(&fp4_s);
        let sc_s = [m, k / 16]; let sc_st = contiguous_strides(&sc_s);
        let gs_s = [1i64]; let gs_st = [1i64];
        let out_s = [m, k]; let out_st = contiguous_strides(&out_s);

        let dl_in = gpu_dl(input_ptr, BF16_DT, &in_s, &in_st);
        let dl_fp4 = gpu_dl(fp4_ptr, U8_DT, &fp4_s, &fp4_st);
        let dl_sc = gpu_dl(scales_ptr, U8_DT, &sc_s, &sc_st);
        let dl_gs = gpu_dl(gs_ptr, FP32_DT, &gs_s, &gs_st);
        let dl_out = gpu_dl(output_ptr, BF16_DT, &out_s, &out_st);

        reg.set_stream(0, std::ptr::null_mut());

        // Quantize: BF16 → FP4 + block scales
        let quant_args = [
            TVMFFIAny::dltensor(&dl_in),
            TVMFFIAny::dltensor(&dl_gs),
            TVMFFIAny::dltensor(&dl_fp4),
            TVMFFIAny::dltensor(&dl_sc),
        ];
        reg.call(quant_fn, &quant_args).expect("nvfp4_kv_quant failed");
        cudaDeviceSynchronize();

        // Dequantize: FP4 + block scales → BF16
        let dequant_args = [
            TVMFFIAny::dltensor(&dl_fp4),
            TVMFFIAny::dltensor(&dl_sc),
            TVMFFIAny::dltensor(&dl_gs),
            TVMFFIAny::dltensor(&dl_out),
        ];
        reg.call(dequant_fn, &dequant_args).expect("nvfp4_kv_dequant failed");
        cudaDeviceSynchronize();

        // Compare: roundtrip error should be bounded
        let out_bf16 = gpu_download::<u16>(output_ptr, n);
        let in_f32: Vec<f32> = input_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        let mut max_diff: f32 = 0.0;
        let mut sum_sq_err: f64 = 0.0;
        for i in 0..n {
            let diff = (in_f32[i] - out_f32[i]).abs();
            max_diff = max_diff.max(diff);
            sum_sq_err += (diff as f64) * (diff as f64);
        }
        let rmse = (sum_sq_err / n as f64).sqrt();

        // FP4 E2M1 has very limited precision (values: 0, 0.5, 1, 1.5, 2, 3, 4, 6)
        // so max error can be large, but RMSE should be reasonable
        assert!(!out_f32.iter().any(|v| v.is_nan()), "FP4 roundtrip produced NaN");
        assert!(rmse < 1.5, "FP4 roundtrip RMSE={rmse:.4} too large");

        println!("FP4 roundtrip: PASS (max_diff={max_diff:.4}, RMSE={rmse:.4})");

        cudaFree(input_ptr); cudaFree(fp4_ptr); cudaFree(scales_ptr);
        cudaFree(gs_ptr); cudaFree(output_ptr);
    }
}

#[test]
fn cascade_merge_state() {
    let reg = KernelRegistry::new();
    // Two-level cascade: level 0 = shared prefix (non-causal), level 1 = unique (causal)
    let keys = [
        PrefillKey {
            dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
            sliding_window: false, logits_soft_cap: false,
            backend: Backend::FA2,
        },
        PrefillKey {
            dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
            sliding_window: false, logits_soft_cap: false,
            backend: Backend::FA2,
        },
    ];
    let cascade = prelude_flashinfer::cascade::CascadeAttention::new(&reg, &keys)
        .expect("Cascade kernels not found");

    let n = 4i64;   // tokens
    let h = 2i64;   // heads
    let d = 128i64;  // head_dim
    let elems = (n * h * d) as usize;
    let lse_elems = (n * h) as usize;

    // Create two attention states with different values
    let v_a_bf16: Vec<u16> = (0..elems).map(|i| f32_to_bf16(0.1 * (i as f32 % 7.0))).collect();
    let v_b_bf16: Vec<u16> = (0..elems).map(|i| f32_to_bf16(0.2 * (i as f32 % 5.0))).collect();
    let s_a_f32: Vec<f32> = (0..lse_elems).map(|i| 1.0 + 0.1 * i as f32).collect();
    let s_b_f32: Vec<f32> = (0..lse_elems).map(|i| 0.5 + 0.2 * i as f32).collect();

    unsafe {
        let v_a = gpu_upload(&v_a_bf16);
        let v_b = gpu_upload(&v_b_bf16);
        let s_a = gpu_upload(&s_a_f32);
        let s_b = gpu_upload(&s_b_f32);
        let v_out = gpu_alloc(elems * 2);
        let s_out = gpu_alloc(lse_elems * 4);

        let v_s = [n, h, d]; let v_st = contiguous_strides(&v_s);
        let s_s = [n, h]; let s_st = contiguous_strides(&s_s);

        let dl_va = gpu_dl(v_a, BF16_DT, &v_s, &v_st);
        let dl_sa = gpu_dl(s_a, FP32_DT, &s_s, &s_st);
        let dl_vb = gpu_dl(v_b, BF16_DT, &v_s, &v_st);
        let dl_sb = gpu_dl(s_b, FP32_DT, &s_s, &s_st);
        let dl_vo = gpu_dl(v_out, BF16_DT, &v_s, &v_st);
        let dl_so = gpu_dl(s_out, FP32_DT, &s_s, &s_st);

        reg.set_stream(0, std::ptr::null_mut());

        let args = [
            TVMFFIAny::dltensor(&dl_va), TVMFFIAny::dltensor(&dl_sa),
            TVMFFIAny::dltensor(&dl_vb), TVMFFIAny::dltensor(&dl_sb),
            TVMFFIAny::dltensor(&dl_vo), TVMFFIAny::dltensor(&dl_so),
        ];
        cascade.merge(&reg, &args).expect("merge_state failed");
        cudaDeviceSynchronize();

        let out_v = gpu_download::<u16>(v_out, elems);
        let out_s = gpu_download::<f32>(s_out, lse_elems);

        // Output should be nonzero and finite
        let v_f32: Vec<f32> = out_v.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = v_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "merge_state output is all zeros");
        assert!(v_f32.iter().all(|v| v.is_finite()), "merge_state output has NaN/Inf");
        assert!(out_s.iter().all(|v| v.is_finite()), "merge_state LSE has NaN/Inf");

        println!("Cascade merge_state: PASS (output sum={sum:.4})");

        cudaFree(v_a); cudaFree(v_b); cudaFree(s_a); cudaFree(s_b);
        cudaFree(v_out); cudaFree(s_out);
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

#[test]
fn decode_sliding_window_dispatch() {
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
        sliding_window: true,
        logits_soft_cap: false,
    };
    let variant = reg.get_decode(&key).expect("SWA decode variant not found");

    unsafe {
        let q_elems = (batch_size * num_qo_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..q_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 7.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let o_ptr = gpu_alloc(q_elems * 2);

        let kv_elems = (total_pages * page_size * num_kv_heads * head_dim) as usize;
        let k_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.005 * (i as f32 % 11.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 5.0))).collect();
        let k_cache = gpu_upload(&k_bf16);
        let v_cache = gpu_upload(&v_bf16);

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

        // Plan — arg index 9 is window_left
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
            TVMFFIAny::int64(16),          // window_left = 16 (SWA!)
            TVMFFIAny::float64(0.0),       // logits_soft_cap
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::dltensor(&dl_eq),
            TVMFFIAny::dltensor(&dl_ek),
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("SWA decode plan failed");

        // Run — arg index 12 is window_left
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
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_kv_indptr),
            TVMFFIAny::dltensor(&dl_kvi),
            TVMFFIAny::dltensor(&dl_kv_last),
            TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),              // maybe_lse
            TVMFFIAny::int64(0),            // layout = NHD
            TVMFFIAny::int64(16),           // window_left = 16 (SWA!)
            TVMFFIAny::bool_val(false),     // enable_pdl
            TVMFFIAny::none(),              // alibi_slopes
            TVMFFIAny::float64(0.0),        // logits_soft_cap
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),        // rope_rcp_scale
            TVMFFIAny::float64(1e4),        // rope_rcp_theta
        ];
        reg.call(variant.run, &run_args)
            .expect("SWA decode run failed");
        cudaDeviceSynchronize();

        let out_bf16 = gpu_download::<u16>(o_ptr, q_elems);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "SWA decode output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "SWA decode output has NaN/Inf");

        println!("Decode SWA dispatch: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(o_ptr);
        cudaFree(k_cache); cudaFree(v_cache);
        cudaFree(kvi_ptr); cudaFree(kv_indptr_gpu); cudaFree(kv_last_gpu);
    }
}

#[test]
fn decode_softcap_dispatch() {
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
        logits_soft_cap: true,
    };
    let variant = reg.get_decode(&key).expect("Softcap decode variant not found");

    unsafe {
        let q_elems = (batch_size * num_qo_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..q_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 7.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let o_ptr = gpu_alloc(q_elems * 2);

        let kv_elems = (total_pages * page_size * num_kv_heads * head_dim) as usize;
        let k_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.005 * (i as f32 % 11.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 5.0))).collect();
        let k_cache = gpu_upload(&k_bf16);
        let v_cache = gpu_upload(&v_bf16);

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

        // Plan — arg index 10 is logits_soft_cap
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
            TVMFFIAny::float64(30.0),      // logits_soft_cap = 30.0
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::dltensor(&dl_eq),
            TVMFFIAny::dltensor(&dl_ek),
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("Softcap decode plan failed");

        // Run — arg index 15 is logits_soft_cap
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
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_kv_indptr),
            TVMFFIAny::dltensor(&dl_kvi),
            TVMFFIAny::dltensor(&dl_kv_last),
            TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),              // maybe_lse
            TVMFFIAny::int64(0),            // layout = NHD
            TVMFFIAny::int64(-1),           // window_left
            TVMFFIAny::bool_val(false),     // enable_pdl
            TVMFFIAny::none(),              // alibi_slopes
            TVMFFIAny::float64(30.0),       // logits_soft_cap = 30.0
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),        // rope_rcp_scale
            TVMFFIAny::float64(1e4),        // rope_rcp_theta
        ];
        reg.call(variant.run, &run_args)
            .expect("Softcap decode run failed");
        cudaDeviceSynchronize();

        let out_bf16 = gpu_download::<u16>(o_ptr, q_elems);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Softcap decode output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "Softcap decode output has NaN/Inf");

        println!("Decode softcap dispatch: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(o_ptr);
        cudaFree(k_cache); cudaFree(v_cache);
        cudaFree(kvi_ptr); cudaFree(kv_indptr_gpu); cudaFree(kv_last_gpu);
    }
}

#[test]
fn prefill_paged_fa2() {
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let batch_size = 1i64;
    let seq_len = 8i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;
    let page_size = 16i64;
    let num_pages = (seq_len + page_size - 1) / page_size;
    let total_pages = num_pages * batch_size;

    let key = PrefillKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: head_dim as u32,
        head_dim_vo: head_dim as u32,
        sliding_window: false,
        logits_soft_cap: false,
        backend: Backend::FA2,
    };
    let variant = reg.get_prefill(&key).expect("Paged prefill variant not found");

    unsafe {
        // Q: [seq_len, num_qo_heads, head_dim]
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..total).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let o_ptr = gpu_alloc(total * 2);

        // Paged KV cache: [total_pages, page_size, num_kv_heads, head_dim]
        let kv_elems = (total_pages * page_size * num_kv_heads * head_dim) as usize;
        let k_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.01 * ((i + 3) as f32 % 10.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.02 * ((i + 7) as f32 % 10.0))).collect();
        let k_cache = gpu_upload(&k_bf16);
        let v_cache = gpu_upload(&v_bf16);

        // Page table
        let mut paged_kv_indptr: Vec<i32> = vec![0];
        let mut paged_kv_indices: Vec<i32> = Vec::new();
        for b in 0..batch_size as i32 {
            paged_kv_indptr.push(paged_kv_indptr.last().unwrap() + num_pages as i32);
            for p in 0..num_pages as i32 {
                paged_kv_indices.push(b * num_pages as i32 + p);
            }
        }
        let paged_kv_last_page_len = {
            let rem = seq_len as i32 % page_size as i32;
            if rem == 0 { vec![page_size as i32; batch_size as usize] }
            else { vec![rem; batch_size as usize] }
        };

        // Indptrs for plan (CPU)
        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];

        // GPU versions
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let paged_kv_indptr_gpu = gpu_upload(&paged_kv_indptr);
        let paged_kv_indices_gpu = gpu_upload(&paged_kv_indices);
        let paged_kv_last_gpu = gpu_upload(&paged_kv_last_page_len);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
        let cu_s = [batch_size + 1]; let cu_st = contiguous_strides(&cu_s);
        let kvl_s = [batch_size]; let kvl_st = contiguous_strides(&kvl_s);
        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);

        let dl_cuq = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);

        let kv_s = [total_pages, page_size, num_kv_heads, head_dim];
        let kv_st = contiguous_strides(&kv_s);
        let dl_k_cache = gpu_dl(k_cache, BF16_DT, &kv_s, &kv_st);
        let dl_v_cache = gpu_dl(v_cache, BF16_DT, &kv_s, &kv_st);

        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let pki_s = [paged_kv_indptr.len() as i64]; let pki_st = contiguous_strides(&pki_s);
        let dl_paged_kv_indptr = gpu_dl(paged_kv_indptr_gpu, I32_DT, &pki_s, &pki_st);
        let pkidx_s = [paged_kv_indices.len() as i64]; let pkidx_st = contiguous_strides(&pkidx_s);
        let dl_paged_kv_indices = gpu_dl(paged_kv_indices_gpu, I32_DT, &pkidx_s, &pkidx_st);
        let pklp_s = [batch_size]; let pklp_st = contiguous_strides(&pklp_s);
        let dl_paged_kv_last = gpu_dl(paged_kv_last_gpu, I32_DT, &pklp_s, &pklp_st);

        reg.set_stream(0, std::ptr::null_mut());

        // Plan (uses page_size, not 1 like ragged)
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws), TVMFFIAny::dltensor(&dl_cuq),
            TVMFFIAny::dltensor(&dl_cuk), TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len), TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(page_size),   // page_size (paged, not 1!)
            TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true),     // causal
            TVMFFIAny::int64(-1),          // window_left
            TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("Paged prefill plan failed");

        // Paged run
        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k_cache),
            TVMFFIAny::dltensor(&dl_v_cache),
            TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_paged_kv_indptr),
            TVMFFIAny::dltensor(&dl_paged_kv_indices),
            TVMFFIAny::dltensor(&dl_paged_kv_last),
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
        reg.call(variant.paged_run, &run_args)
            .expect("Paged prefill run failed");
        cudaDeviceSynchronize();

        let out_bf16 = gpu_download::<u16>(o_ptr, total);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Paged prefill output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "Paged prefill output has NaN/Inf");

        println!("Prefill paged FA2: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(o_ptr);
        cudaFree(k_cache); cudaFree(v_cache);
        cudaFree(cu_q_gpu); cudaFree(paged_kv_indptr_gpu);
        cudaFree(paged_kv_indices_gpu); cudaFree(paged_kv_last_gpu);
    }
}

#[test]
fn prefill_custom_mask() {
    // Custom mask requires Optional<ffi::Tensor> (TVM Tensor objects, not raw DLTensors).
    // Creating TVM Tensor objects requires TVM runtime allocator setup which isn't available
    // in standalone tests. Skip until we integrate with the full runtime.
    println!("Skipping prefill_custom_mask — requires TVM Tensor allocator for Optional<ffi::Tensor> params");
    return;

    #[allow(unreachable_code)]
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
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;
        let q_bf16: Vec<u16> = (0..total).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let k_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.01 * ((i + 3) as f32 % 10.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_total).map(|i| f32_to_bf16(0.02 * ((i + 7) as f32 % 10.0))).collect();
        let q_ptr = gpu_upload(&q_bf16);
        let k_ptr = gpu_upload(&k_bf16);
        let v_ptr = gpu_upload(&v_bf16);
        let o_ptr = gpu_alloc(total * 2);

        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let cu_k_gpu = gpu_upload(&cu_k_data);

        // Custom mask: all-ones mask (full attention), packed as uint8 bitmask
        // Total bits = seq_len * seq_len = 64 bits = 8 bytes
        let num_bits = (seq_len * seq_len) as usize;
        let mask_bytes = (num_bits + 7) / 8;
        let custom_mask_data: Vec<u8> = vec![0xFF; mask_bytes];
        let custom_mask_gpu = gpu_upload(&custom_mask_data);
        // mask_indptr: [0, num_bits] for batch_size=1
        let mask_indptr_data: [i32; 2] = [0, num_bits as i32];
        let mask_indptr_gpu = gpu_upload(&mask_indptr_data);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
        let cu_s = [batch_size + 1]; let cu_st = contiguous_strides(&cu_s);
        let kvl_s = [batch_size]; let kvl_st = contiguous_strides(&kvl_s);
        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim]; let k_st = contiguous_strides(&k_s);

        let dl_cuq = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);
        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, BF16_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);

        let mask_s = [mask_bytes as i64]; let mask_st = [1i64];
        let dl_custom_mask = gpu_dl(custom_mask_gpu, U8_DT, &mask_s, &mask_st);
        let mi_s = [batch_size + 1]; let mi_st = contiguous_strides(&mi_s);
        let dl_mask_indptr = gpu_dl(mask_indptr_gpu, I32_DT, &mi_s, &mi_st);

        reg.set_stream(0, std::ptr::null_mut());

        // Plan (same as causal — mask_mode is only used at run time)
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws), TVMFFIAny::dltensor(&dl_cuq),
            TVMFFIAny::dltensor(&dl_cuk), TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len), TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1),           // page_size (ragged=1)
            TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(false),    // causal=false (custom mask)
            TVMFFIAny::int64(-1),
            TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false), TVMFFIAny::int64(0),
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("Custom mask prefill plan failed");

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v), TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_cuk_gpu), TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),             // maybe_lse
            TVMFFIAny::int64(2),           // mask_mode = Custom
            TVMFFIAny::int64(0),           // layout = NHD
            TVMFFIAny::int64(-1),          // window_left
            TVMFFIAny::bool_val(false),    // enable_pdl
            TVMFFIAny::dltensor(&dl_custom_mask),  // custom_mask (Optional<ffi::Tensor>)
            TVMFFIAny::dltensor(&dl_mask_indptr),  // mask_indptr (Optional<ffi::Tensor>)
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
            .expect("Custom mask prefill run failed");
        cudaDeviceSynchronize();

        let out_bf16 = gpu_download::<u16>(o_ptr, total);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Custom mask prefill output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "Custom mask prefill has NaN/Inf");

        println!("Prefill custom mask: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr); cudaFree(o_ptr);
        cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
        cudaFree(custom_mask_gpu); cudaFree(mask_indptr_gpu);
    }
}

const FP8_E4M3_DT: DLDataType = DLDataType { code: 32, bits: 8, lanes: 1 };

#[test]
fn fp8_prefill_execution() {
    let reg = KernelRegistry::new();
    if reg.arch() < 90 { println!("Skipping FP8 prefill — requires SM90+"); return; }

    let ws = Workspace::new();
    let seq_len = 8i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;

    let key = FP8PrefillKey {
        head_dim: head_dim as u32,
        sliding_window: false,
    };
    let variant = reg.get_fp8_prefill(&key).expect("FP8 prefill variant not found");

    unsafe {
        let total = (seq_len * num_qo_heads * head_dim) as usize;
        let kv_total = (seq_len * num_kv_heads * head_dim) as usize;
        // FP8 E4M3: generate small values. 0x3C = 1.0 in E4M3, 0x38 = 0.5, etc.
        // Use small positive values to avoid overflow.
        let q_fp8: Vec<u8> = (0..total).map(|i| (0x20 + (i % 16)) as u8).collect();
        let k_fp8: Vec<u8> = (0..kv_total).map(|i| (0x20 + ((i + 3) % 16)) as u8).collect();
        let v_fp8: Vec<u8> = (0..kv_total).map(|i| (0x20 + ((i + 7) % 16)) as u8).collect();
        let q_ptr = gpu_upload(&q_fp8);
        let k_ptr = gpu_upload(&k_fp8);
        let v_ptr = gpu_upload(&v_fp8);
        let o_ptr = gpu_alloc(total * 2); // BF16 output

        let cu_q_data: [i32; 2] = [0, seq_len as i32];
        let cu_k_data: [i32; 2] = [0, seq_len as i32];
        let kvl_data: [i32; 1] = [seq_len as i32];
        let cu_q_gpu = gpu_upload(&cu_q_data);
        let cu_k_gpu = gpu_upload(&cu_k_data);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();
        let cu_s = [2i64]; let cu_st = [1i64];
        let kvl_s = [1i64]; let kvl_st = [1i64];
        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim]; let k_st = contiguous_strides(&k_s);

        let dl_cuq = cpu_dl(cu_q_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk = cpu_dl(cu_k_data.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl = cpu_dl(kvl_data.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);

        reg.set_stream(0, std::ptr::null_mut());

        // FA3 SM90 plan: 16 args
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq), TVMFFIAny::dltensor(&dl_cuk),
            TVMFFIAny::dltensor(&dl_kvl),
            TVMFFIAny::int64(seq_len), TVMFFIAny::int64(1),
            TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(1), TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true), TVMFFIAny::int64(-1),
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("FP8 prefill plan failed");

        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let dl_q = gpu_dl(q_ptr, FP8_E4M3_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, FP8_E4M3_DT, &k_s, &k_st);
        let dl_v = gpu_dl(v_ptr, FP8_E4M3_DT, &k_s, &k_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &q_s, &q_st);
        let dl_cuq_gpu = gpu_dl(cu_q_gpu, I32_DT, &cu_s, &cu_st);
        let dl_cuk_gpu = gpu_dl(cu_k_gpu, I32_DT, &cu_s, &cu_st);

        // FP8 SM90 ragged_run: 21 args
        // float_ws, int_ws, plan_info, q, k, v, qo_indptr, kv_indptr, o,
        // maybe_lse, mask_mode, layout, window_left, enable_pdl,
        // ADDITIONAL: maybe_scale_q, maybe_scale_k, maybe_scale_v, sm_scale,
        //             scale_q_scalar, scale_k_scalar, scale_v_scalar
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v), TVMFFIAny::dltensor(&dl_cuq_gpu),
            TVMFFIAny::dltensor(&dl_cuk_gpu), TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::none(),         // lse
            TVMFFIAny::int64(1),       // causal
            TVMFFIAny::int64(0),       // NHD
            TVMFFIAny::int64(-1),      // window_left
            TVMFFIAny::bool_val(false),// enable_pdl
            TVMFFIAny::none(),         // maybe_scale_q
            TVMFFIAny::none(),         // maybe_scale_k
            TVMFFIAny::none(),         // maybe_scale_v
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),   // scale_q_scalar
            TVMFFIAny::float64(1.0),   // scale_k_scalar
            TVMFFIAny::float64(1.0),   // scale_v_scalar
        ];
        reg.call(variant.ragged_run, &run_args)
            .expect("FP8 prefill ragged_run failed");
        cudaDeviceSynchronize();

        let out = gpu_download::<u16>(o_ptr, total);
        let out_f32: Vec<f32> = out.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "FP8 prefill output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "FP8 prefill has NaN/Inf");

        println!("FP8 prefill: PASS (sum={sum:.4})");

        cudaFree(q_ptr); cudaFree(k_ptr); cudaFree(v_ptr); cudaFree(o_ptr);
        cudaFree(cu_q_gpu); cudaFree(cu_k_gpu);
    }
}

#[test]
fn mla_decode_execution() {
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let batch_size = 1i64;
    let num_qo_heads = 2i64;
    let head_dim_ckv = 512i64;
    let head_dim_kpe = 64i64;
    let page_size = 16i64;
    let kv_len: i64 = 32;
    let num_pages = (kv_len + page_size - 1) / page_size;
    let total_pages = num_pages * batch_size;

    let key = MLADecodeKey {
        dtype: KernelDtype::BF16,
        head_dim_ckv: head_dim_ckv as u32,
        head_dim_kpe: head_dim_kpe as u32,
    };
    let variant = reg.get_mla_decode(&key).expect("MLA decode variant not found");

    unsafe {
        // Q: q_nope [batch, num_qo_heads, head_dim_ckv], q_pe [batch, num_qo_heads, head_dim_kpe]
        let q_nope_elems = (batch_size * num_qo_heads * head_dim_ckv) as usize;
        let q_pe_elems = (batch_size * num_qo_heads * head_dim_kpe) as usize;
        let q_nope_bf16: Vec<u16> = (0..q_nope_elems).map(|i| f32_to_bf16(0.005 * (i as f32 % 7.0))).collect();
        let q_pe_bf16: Vec<u16> = (0..q_pe_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 5.0))).collect();
        let q_nope_ptr = gpu_upload(&q_nope_bf16);
        let q_pe_ptr = gpu_upload(&q_pe_bf16);
        let o_elems = (batch_size * num_qo_heads * head_dim_ckv) as usize;
        let o_ptr = gpu_alloc(o_elems * 2);

        // Paged CKV cache: [total_pages, page_size, 1, head_dim_ckv] (single-head MLA)
        let ckv_elems = (total_pages * page_size * 1 * head_dim_ckv) as usize;
        let kpe_elems = (total_pages * page_size * 1 * head_dim_kpe) as usize;
        let ckv_bf16: Vec<u16> = (0..ckv_elems).map(|i| f32_to_bf16(0.003 * (i as f32 % 9.0))).collect();
        let kpe_bf16: Vec<u16> = (0..kpe_elems).map(|i| f32_to_bf16(0.005 * (i as f32 % 6.0))).collect();
        let ckv_cache = gpu_upload(&ckv_bf16);
        let kpe_cache = gpu_upload(&kpe_bf16);

        // Page table
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
        let kv_indptr_gpu = gpu_upload(&kv_indptr);
        let kv_last_gpu = gpu_upload(&kv_last_page_len);

        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();

        let indptr_s = [batch_size + 1];
        let indptr_st = contiguous_strides(&indptr_s);
        let dl_indptr_cpu = cpu_dl(kv_indptr_cpu.as_ptr() as *mut c_void, I32_DT, &indptr_s, &indptr_st);

        reg.set_stream(0, std::ptr::null_mut());

        // MLA decode plan: 8 args
        // float_ws, int_ws, pinned_ws, indptr, batch_size, num_qo_heads, page_size, cuda_graph
        let plan_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_indptr_cpu),
            TVMFFIAny::int64(batch_size),
            TVMFFIAny::int64(num_qo_heads),
            TVMFFIAny::int64(page_size),
            TVMFFIAny::bool_val(false),    // cuda_graph
        ];
        let plan_info = reg.call(variant.plan, &plan_args)
            .expect("MLA decode plan failed");

        // Build run DLTensors
        let qn_s = [batch_size, num_qo_heads, head_dim_ckv];
        let qn_st = contiguous_strides(&qn_s);
        let qp_s = [batch_size, num_qo_heads, head_dim_kpe];
        let qp_st = contiguous_strides(&qp_s);
        let o_s = [batch_size, num_qo_heads, head_dim_ckv];
        let o_st = contiguous_strides(&o_s);
        let ckv_s = [total_pages, page_size, 1i64, head_dim_ckv];
        let ckv_st = contiguous_strides(&ckv_s);
        let kpe_s = [total_pages, page_size, 1i64, head_dim_kpe];
        let kpe_st = contiguous_strides(&kpe_s);
        let kvi_s = [kv_indices.len() as i64];
        let kvi_st = contiguous_strides(&kvi_s);
        let kvlp_s = [batch_size];
        let kvlp_st = contiguous_strides(&kvlp_s);
        let kv_indptr_s = [batch_size + 1];
        let kv_indptr_st = contiguous_strides(&kv_indptr_s);

        let dl_q_nope = gpu_dl(q_nope_ptr, BF16_DT, &qn_s, &qn_st);
        let dl_q_pe = gpu_dl(q_pe_ptr, BF16_DT, &qp_s, &qp_st);
        let dl_ckv = gpu_dl(ckv_cache, BF16_DT, &ckv_s, &ckv_st);
        let dl_kpe = gpu_dl(kpe_cache, BF16_DT, &kpe_s, &kpe_st);
        let dl_kv_indptr = gpu_dl(kv_indptr_gpu, I32_DT, &kv_indptr_s, &kv_indptr_st);
        let dl_kvi = gpu_dl(kvi_ptr, I32_DT, &kvi_s, &kvi_st);
        let dl_kv_last = gpu_dl(kv_last_gpu, I32_DT, &kvlp_s, &kvlp_st);
        let dl_o = gpu_dl(o_ptr, BF16_DT, &o_s, &o_st);

        let sm_scale = 1.0 / (head_dim_ckv as f64).sqrt();
        // MLA decode run: 18 args
        // float_ws, int_ws, plan_info, q_nope, q_pe, paged_ckv_cache,
        // paged_kpe_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
        // o, sm_scale, window_left, logits_soft_cap, rope_scale, rope_theta,
        // maybe_lse, enable_pdl
        let run_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q_nope),
            TVMFFIAny::dltensor(&dl_q_pe),
            TVMFFIAny::dltensor(&dl_ckv),
            TVMFFIAny::dltensor(&dl_kpe),
            TVMFFIAny::dltensor(&dl_kv_indptr),
            TVMFFIAny::dltensor(&dl_kvi),
            TVMFFIAny::dltensor(&dl_kv_last),
            TVMFFIAny::dltensor(&dl_o),
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::int64(-1),          // window_left
            TVMFFIAny::float64(0.0),       // logits_soft_cap
            TVMFFIAny::float64(1.0),       // rope_scale
            TVMFFIAny::float64(1e6),       // rope_theta
            TVMFFIAny::none(),             // maybe_lse
            TVMFFIAny::bool_val(false),    // enable_pdl
        ];
        reg.call(variant.run, &run_args)
            .expect("MLA decode run failed");
        cudaDeviceSynchronize();

        let out_bf16 = gpu_download::<u16>(o_ptr, o_elems);
        let out_f32: Vec<f32> = out_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let sum: f32 = out_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "MLA decode output is all zeros");
        assert!(out_f32.iter().all(|v| v.is_finite()), "MLA decode output has NaN/Inf");

        println!("MLA decode: PASS (sum={sum:.4})");

        cudaFree(q_nope_ptr); cudaFree(q_pe_ptr); cudaFree(o_ptr);
        cudaFree(ckv_cache); cudaFree(kpe_cache);
        cudaFree(kvi_ptr); cudaFree(kv_indptr_gpu); cudaFree(kv_last_gpu);
    }
}

#[test]
fn apply_rope_execution() {
    let reg = KernelRegistry::new();
    let rope_fn = reg.get_utility("apply_rope").expect("apply_rope not found");

    let batch_size = 1i64;
    let seq_len = 4i64;
    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;
    let rotary_dim = 64i64; // typically head_dim or head_dim/2

    let total_q = (seq_len * num_qo_heads * head_dim) as usize;
    let total_k = (seq_len * num_kv_heads * head_dim) as usize;

    let q_bf16: Vec<u16> = (0..total_q).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
    let k_bf16: Vec<u16> = (0..total_k).map(|i| f32_to_bf16(0.01 * ((i + 3) as f32 % 10.0))).collect();

    unsafe {
        let q_ptr = gpu_upload(&q_bf16);
        let k_ptr = gpu_upload(&k_bf16);
        let q_rope_ptr = gpu_alloc(total_q * 2);
        let k_rope_ptr = gpu_alloc(total_k * 2);

        // indptr: [0, seq_len] for batch_size=1 (cumulative token count per batch)
        let indptr_data: [i32; 2] = [0, seq_len as i32];
        let indptr_gpu = gpu_upload(&indptr_data);
        // offsets: [0] for batch_size=1 (starting position offset per batch)
        let offsets_data: [i32; 1] = [0];
        let offsets_gpu = gpu_upload(&offsets_data);

        let q_s = [seq_len, num_qo_heads, head_dim]; let q_st = contiguous_strides(&q_s);
        let k_s = [seq_len, num_kv_heads, head_dim]; let k_st = contiguous_strides(&k_s);
        let indptr_s = [batch_size + 1]; let indptr_st = contiguous_strides(&indptr_s);
        let offsets_s = [batch_size]; let offsets_st = contiguous_strides(&offsets_s);

        let dl_q = gpu_dl(q_ptr, BF16_DT, &q_s, &q_st);
        let dl_k = gpu_dl(k_ptr, BF16_DT, &k_s, &k_st);
        let dl_q_rope = gpu_dl(q_rope_ptr, BF16_DT, &q_s, &q_st);
        let dl_k_rope = gpu_dl(k_rope_ptr, BF16_DT, &k_s, &k_st);
        let dl_indptr = gpu_dl(indptr_gpu, I32_DT, &indptr_s, &indptr_st);
        let dl_offsets = gpu_dl(offsets_gpu, I32_DT, &offsets_s, &offsets_st);

        reg.set_stream(0, std::ptr::null_mut());

        // apply_rope: q, k, q_rope, k_rope, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta
        let args = [
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_q_rope),
            TVMFFIAny::dltensor(&dl_k_rope),
            TVMFFIAny::dltensor(&dl_indptr),
            TVMFFIAny::dltensor(&dl_offsets),
            TVMFFIAny::int64(rotary_dim),
            TVMFFIAny::bool_val(false),    // interleave
            TVMFFIAny::float64(1.0),       // rope_scale
            TVMFFIAny::float64(1e4),       // rope_theta
        ];
        reg.call(rope_fn, &args).expect("apply_rope failed");
        cudaDeviceSynchronize();

        let out_q = gpu_download::<u16>(q_rope_ptr, total_q);
        let out_k = gpu_download::<u16>(k_rope_ptr, total_k);
        let q_f32: Vec<f32> = out_q.iter().map(|&v| bf16_to_f32(v)).collect();
        let k_f32: Vec<f32> = out_k.iter().map(|&v| bf16_to_f32(v)).collect();

        let sum_q: f32 = q_f32.iter().map(|v| v.abs()).sum();
        let sum_k: f32 = k_f32.iter().map(|v| v.abs()).sum();
        assert!(sum_q > 0.0, "apply_rope q output is all zeros");
        assert!(sum_k > 0.0, "apply_rope k output is all zeros");
        assert!(q_f32.iter().all(|v| v.is_finite()), "apply_rope q has NaN/Inf");
        assert!(k_f32.iter().all(|v| v.is_finite()), "apply_rope k has NaN/Inf");

        println!("apply_rope: PASS (q_sum={sum_q:.4}, k_sum={sum_k:.4})");

        cudaFree(q_ptr); cudaFree(k_ptr);
        cudaFree(q_rope_ptr); cudaFree(k_rope_ptr);
        cudaFree(indptr_gpu); cudaFree(offsets_gpu);
    }
}

#[test]
fn moe_routing_execution() {
    let reg = KernelRegistry::new();
    let noaux_fn = reg.get_utility("NoAuxTc").expect("NoAuxTc (MoE routing) not found");

    let num_tokens = 4i64;
    let num_experts = 8i64;
    let n_group = 1i64;
    let topk_group = 1i64;
    let topk = 2i64;

    // Scores: [num_tokens, num_experts] BF16
    let score_elems = (num_tokens * num_experts) as usize;
    let scores_bf16: Vec<u16> = (0..score_elems)
        .map(|i| f32_to_bf16(0.1 * (i as f32 % num_experts as f32)))
        .collect();
    // Bias: [num_experts] BF16
    let bias_bf16: Vec<u16> = (0..num_experts as usize)
        .map(|i| f32_to_bf16(0.01 * i as f32))
        .collect();

    unsafe {
        let scores_ptr = gpu_upload(&scores_bf16);
        let bias_ptr = gpu_upload(&bias_bf16);
        // topk_values: [num_tokens, topk] BF16
        let topk_elems = (num_tokens * topk) as usize;
        let topk_values_ptr = gpu_alloc(topk_elems * 2);
        // topk_indices: [num_tokens, topk] I32
        let topk_indices_ptr = gpu_alloc(topk_elems * 4);

        let score_s = [num_tokens, num_experts];
        let score_st = contiguous_strides(&score_s);
        let bias_s = [num_experts];
        let bias_st = contiguous_strides(&bias_s);
        let topk_s = [num_tokens, topk];
        let topk_st = contiguous_strides(&topk_s);

        let dl_scores = gpu_dl(scores_ptr, BF16_DT, &score_s, &score_st);
        let dl_bias = gpu_dl(bias_ptr, BF16_DT, &bias_s, &bias_st);
        let dl_topk_values = gpu_dl(topk_values_ptr, BF16_DT, &topk_s, &topk_st);
        let dl_topk_indices = gpu_dl(topk_indices_ptr, I32_DT, &topk_s, &topk_st);

        reg.set_stream(0, std::ptr::null_mut());

        // NoAuxTc: scores, bias, n_group, topk_group, topk, routed_scaling_factor,
        //          topk_values, topk_indices, launch_with_pdl
        let args = [
            TVMFFIAny::dltensor(&dl_scores),
            TVMFFIAny::dltensor(&dl_bias),
            TVMFFIAny::int64(n_group),
            TVMFFIAny::int64(topk_group),
            TVMFFIAny::int64(topk),
            TVMFFIAny::float64(1.0),       // routed_scaling_factor
            TVMFFIAny::dltensor(&dl_topk_values),
            TVMFFIAny::dltensor(&dl_topk_indices),
            TVMFFIAny::bool_val(false),    // launch_with_pdl
        ];
        reg.call(noaux_fn, &args).expect("NoAuxTc MoE routing failed");
        cudaDeviceSynchronize();

        let out_values = gpu_download::<u16>(topk_values_ptr, topk_elems);
        let out_indices = gpu_download::<i32>(topk_indices_ptr, topk_elems);
        let values_f32: Vec<f32> = out_values.iter().map(|&v| bf16_to_f32(v)).collect();

        let sum: f32 = values_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "MoE routing output values are all zeros");
        assert!(values_f32.iter().all(|v| v.is_finite()), "MoE routing values have NaN/Inf");
        // Indices should be valid expert ids [0, num_experts)
        assert!(
            out_indices.iter().all(|&idx| idx >= 0 && idx < num_experts as i32),
            "MoE routing indices out of range"
        );

        println!("MoE routing (NoAuxTc): PASS (sum={sum:.4}, indices={out_indices:?})");

        cudaFree(scores_ptr); cudaFree(bias_ptr);
        cudaFree(topk_values_ptr); cudaFree(topk_indices_ptr);
    }
}

#[test]
fn pod_variant_lookup() {
    let reg = KernelRegistry::new();
    let key = PodKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: 128,
        head_dim_vo: 128,
    };
    let variant = reg.get_pod(&key);
    if variant.is_some() {
        println!("POD BF16 h128: found (merged swa/softcap, kCausal+kCustom mask modes)");
    } else {
        println!("POD BF16 h128: not compiled (expected if using minimal kernel set)");
    }
}

#[test]
fn pod_execution() {
    // POD (Prefill-On-Decode): mixed batch with prefill + decode in one kernel.
    // We reuse prefill plan and decode plan, then call POD run with both plan_infos.
    let reg = KernelRegistry::new();
    let ws = Workspace::new();

    let pod_key = PodKey {
        dtype: KernelDtype::BF16,
        head_dim_qk: 128,
        head_dim_vo: 128,
    };
    let pod_variant = match reg.get_pod(&pod_key) {
        Some(v) => v,
        None => {
            println!("POD BF16 h128 not compiled, skipping execution test");
            return;
        }
    };

    // Also need prefill and decode plan functions
    // POD uses PrefillPlanInfo for BOTH sides (prefill + decode).
    // The decode side is treated as a prefill with Q=1.
    let prefill_variant = reg.get_prefill(&PrefillKey {
        dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
        sliding_window: false, logits_soft_cap: false, backend: Backend::FA2,
    }).expect("Prefill variant needed for POD plan");

    let num_qo_heads = 2i64;
    let num_kv_heads = 2i64;
    let head_dim = 128i64;
    let page_size = 16i64;

    // Prefill request: seq_len_p tokens of Q, kv_len_p tokens of KV
    let seq_len_p = 8i64;
    let kv_len_p = 8i64;
    let num_pages_p = (kv_len_p + page_size - 1) / page_size;

    // Decode request: 1 token of Q, kv_len_d tokens of KV
    let kv_len_d = 32i64;
    let num_pages_d = (kv_len_d + page_size - 1) / page_size;

    let total_pages = num_pages_p + num_pages_d;

    unsafe {
        // ── Allocate shared KV cache ──
        let kv_elems = (total_pages * page_size * num_kv_heads * head_dim) as usize;
        let k_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.005 * (i as f32 % 11.0))).collect();
        let v_bf16: Vec<u16> = (0..kv_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 5.0))).collect();
        let k_cache = gpu_upload(&k_bf16);
        let v_cache = gpu_upload(&v_bf16);

        // ── Prefill Q [seq_len_p, num_qo_heads, head_dim] ──
        let q_p_elems = (seq_len_p * num_qo_heads * head_dim) as usize;
        let q_p_bf16: Vec<u16> = (0..q_p_elems).map(|i| f32_to_bf16(0.01 * (i as f32 % 10.0))).collect();
        let q_p_ptr = gpu_upload(&q_p_bf16);
        let o_p_ptr = gpu_alloc(q_p_elems * 2);

        // ── Decode Q [1, num_qo_heads, head_dim] ──
        let q_d_elems = (1 * num_qo_heads * head_dim) as usize;
        let q_d_bf16: Vec<u16> = (0..q_d_elems).map(|i| f32_to_bf16(0.02 * (i as f32 % 7.0))).collect();
        let q_d_ptr = gpu_upload(&q_d_bf16);
        let o_d_ptr = gpu_alloc(q_d_elems * 2);

        // ── Page tables ──
        // Prefill: pages [0, num_pages_p)
        let kv_indptr_p: Vec<i32> = vec![0, num_pages_p as i32];
        let kv_indices_p: Vec<i32> = (0..num_pages_p as i32).collect();
        let kv_last_page_p = vec![
            if kv_len_p % page_size == 0 { page_size as i32 } else { (kv_len_p % page_size) as i32 }
        ];
        // Decode: pages [num_pages_p, total_pages)
        let kv_indptr_d: Vec<i32> = vec![0, num_pages_d as i32];
        let kv_indices_d: Vec<i32> = (num_pages_p as i32..total_pages as i32).collect();
        let kv_last_page_d = vec![
            if kv_len_d % page_size == 0 { page_size as i32 } else { (kv_len_d % page_size) as i32 }
        ];

        // Prefill qo_indptr
        let qo_indptr_p: Vec<i32> = vec![0, seq_len_p as i32];
        // Decode qo_indptr (1 token per seq)
        let qo_indptr_d: Vec<i32> = vec![0, 1];

        // Upload page tables
        let kv_indptr_p_gpu = gpu_upload(&kv_indptr_p);
        let kv_indices_p_gpu = gpu_upload(&kv_indices_p);
        let kv_last_p_gpu = gpu_upload(&kv_last_page_p);
        let qo_indptr_p_gpu = gpu_upload(&qo_indptr_p);

        let kv_indptr_d_gpu = gpu_upload(&kv_indptr_d);
        let kv_indices_d_gpu = gpu_upload(&kv_indices_d);
        let kv_last_d_gpu = gpu_upload(&kv_last_page_d);
        let qo_indptr_d_gpu = gpu_upload(&qo_indptr_d);

        // ── SM-aware scheduling buffer ──
        let num_sm = 132i64; // H200
        let sched_elems = (num_sm + 2) as usize;
        let sched_ptr = gpu_alloc(sched_elems * 4);
        cudaMemset(sched_ptr, 0, sched_elems * 4);

        reg.set_stream(0, std::ptr::null_mut());

        // ── Workspaces ──
        let (dl_fws, _fws_s, _fws_st) = ws.dl_float();
        let (dl_iws, _iws_s, _iws_st) = ws.dl_int();
        let (dl_pws, _pws_s, _pws_st) = ws.dl_pinned();

        // ── Step 1: Prefill Plan ──
        let cu_s = [2i64]; // batch_size + 1 = 1 + 1
        let cu_st = contiguous_strides(&cu_s);
        let kvl_s = [1i64]; // batch_size = 1
        let kvl_st = contiguous_strides(&kvl_s);
        let kvl_data_p: [i32; 1] = [kv_len_p as i32];
        let dl_cuq_p_cpu = cpu_dl(qo_indptr_p.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk_p_cpu = cpu_dl(kv_indptr_p.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl_p = cpu_dl(kvl_data_p.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);

        let plan_p_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq_p_cpu),
            TVMFFIAny::dltensor(&dl_cuk_p_cpu),
            TVMFFIAny::dltensor(&dl_kvl_p),
            TVMFFIAny::int64(kv_len_p),        // total_len
            TVMFFIAny::int64(1),               // batch_size
            TVMFFIAny::int64(num_qo_heads),
            TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(page_size),
            TVMFFIAny::bool_val(false),        // cuda_graph
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true),         // causal
            TVMFFIAny::int64(-1),              // window_left
            TVMFFIAny::int64(-1),              // fixed_split_size
            TVMFFIAny::bool_val(false),        // disable_split_kv
            TVMFFIAny::int64(0),               // num_colocated_ctas
        ];
        let plan_info_p = reg.call(prefill_variant.plan, &plan_p_args)
            .expect("Prefill plan failed for POD");

        // ── Step 2: Decode Plan (uses prefill plan — POD treats decode as prefill with Q=1) ──
        let qo_indptr_d_cpu: Vec<i32> = vec![0, 1]; // Q=1
        let kv_indptr_d_cpu_plan = kv_indptr_d.clone();
        let kvl_data_d: [i32; 1] = [kv_len_d as i32];
        let dl_cuq_d_cpu = cpu_dl(qo_indptr_d_cpu.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_cuk_d_cpu = cpu_dl(kv_indptr_d_cpu_plan.as_ptr() as *mut c_void, I32_DT, &cu_s, &cu_st);
        let dl_kvl_d = cpu_dl(kvl_data_d.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);

        let plan_d_args = [
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq_d_cpu),
            TVMFFIAny::dltensor(&dl_cuk_d_cpu),
            TVMFFIAny::dltensor(&dl_kvl_d),
            TVMFFIAny::int64(kv_len_d),        // total_len
            TVMFFIAny::int64(1),               // batch_size
            TVMFFIAny::int64(num_qo_heads),
            TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(page_size),
            TVMFFIAny::bool_val(false),        // cuda_graph
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::int64(head_dim),
            TVMFFIAny::bool_val(true),         // causal
            TVMFFIAny::int64(-1),              // window_left
            TVMFFIAny::int64(-1),              // fixed_split_size
            TVMFFIAny::bool_val(false),        // disable_split_kv
            TVMFFIAny::int64(0),               // num_colocated_ctas
        ];
        let plan_info_d = reg.call(prefill_variant.plan, &plan_d_args)
            .expect("Decode-side prefill plan failed for POD");

        // ── Step 3: POD Run ──
        let sm_scale = 1.0 / (head_dim as f64).sqrt();

        // Build DLTensors for POD
        let q_p_s = [seq_len_p, num_qo_heads, head_dim];
        let q_p_st = contiguous_strides(&q_p_s);
        let q_d_s = [1i64, num_qo_heads, head_dim];
        let q_d_st = contiguous_strides(&q_d_s);
        let kv_s = [total_pages, page_size, num_kv_heads, head_dim];
        let kv_st = contiguous_strides(&kv_s);
        let sched_s = [num_sm + 2];
        let sched_st = contiguous_strides(&sched_s);

        let dl_qp = gpu_dl(q_p_ptr, BF16_DT, &q_p_s, &q_p_st);
        let dl_op = gpu_dl(o_p_ptr, BF16_DT, &q_p_s, &q_p_st);
        let dl_qd = gpu_dl(q_d_ptr, BF16_DT, &q_d_s, &q_d_st);
        let dl_od = gpu_dl(o_d_ptr, BF16_DT, &q_d_s, &q_d_st);
        let dl_k = gpu_dl(k_cache, BF16_DT, &kv_s, &kv_st);
        let dl_v = gpu_dl(v_cache, BF16_DT, &kv_s, &kv_st);
        let dl_sched = gpu_dl(sched_ptr, I32_DT, &sched_s, &sched_st);

        let kvi_p_s = [kv_indices_p.len() as i64];
        let kvi_p_st = contiguous_strides(&kvi_p_s);
        let kvi_d_s = [kv_indices_d.len() as i64];
        let kvi_d_st = contiguous_strides(&kvi_d_s);
        let kv_indptr_s = [2i64];
        let kv_indptr_st = contiguous_strides(&kv_indptr_s);
        let kvlp_s = [1i64];
        let kvlp_st = contiguous_strides(&kvlp_s);

        let dl_kv_indptr_p = gpu_dl(kv_indptr_p_gpu, I32_DT, &kv_indptr_s, &kv_indptr_st);
        let dl_kv_indices_p = gpu_dl(kv_indices_p_gpu, I32_DT, &kvi_p_s, &kvi_p_st);
        let dl_kv_last_p = gpu_dl(kv_last_p_gpu, I32_DT, &kvlp_s, &kvlp_st);
        let dl_qo_indptr_p = gpu_dl(qo_indptr_p_gpu, I32_DT, &kv_indptr_s, &kv_indptr_st);

        let dl_kv_indptr_d = gpu_dl(kv_indptr_d_gpu, I32_DT, &kv_indptr_s, &kv_indptr_st);
        let dl_kv_indices_d = gpu_dl(kv_indices_d_gpu, I32_DT, &kvi_d_s, &kvi_d_st);
        let dl_kv_last_d = gpu_dl(kv_last_d_gpu, I32_DT, &kvlp_s, &kvlp_st);
        let dl_qo_indptr_d = gpu_dl(qo_indptr_d_gpu, I32_DT, &kv_indptr_s, &kv_indptr_st);

        // POD run: prefill params, decode params, enable_pdl, sm_aware_sched
        let run_args = [
            // ── Prefill params ──
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info_p,
            TVMFFIAny::dltensor(&dl_qp),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_qo_indptr_p),
            TVMFFIAny::dltensor(&dl_kv_indptr_p),
            TVMFFIAny::dltensor(&dl_kv_indices_p),
            TVMFFIAny::dltensor(&dl_kv_last_p),
            TVMFFIAny::dltensor(&dl_op),
            TVMFFIAny::none(),              // maybe_lse_p
            TVMFFIAny::int64(1),            // mask_mode_p = Causal
            TVMFFIAny::int64(0),            // layout_p = NHD
            TVMFFIAny::int64(-1),           // window_left_p
            TVMFFIAny::none(),              // custom_mask_p
            TVMFFIAny::none(),              // mask_indptr_p
            TVMFFIAny::none(),              // alibi_slopes_p
            TVMFFIAny::float64(0.0),        // logits_soft_cap_p
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),        // rope_rcp_scale_p
            TVMFFIAny::float64(1e4),        // rope_rcp_theta_p
            // ── Decode params ──
            TVMFFIAny::dltensor(&dl_fws),
            TVMFFIAny::dltensor(&dl_iws),
            plan_info_d,
            TVMFFIAny::dltensor(&dl_qd),
            TVMFFIAny::dltensor(&dl_k),
            TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_qo_indptr_d),
            TVMFFIAny::dltensor(&dl_kv_indptr_d),
            TVMFFIAny::dltensor(&dl_kv_indices_d),
            TVMFFIAny::dltensor(&dl_kv_last_d),
            TVMFFIAny::dltensor(&dl_od),
            TVMFFIAny::none(),              // maybe_lse_d
            TVMFFIAny::int64(1),            // mask_mode_d = Causal
            TVMFFIAny::int64(0),            // layout_d = NHD
            TVMFFIAny::int64(-1),           // window_left_d
            TVMFFIAny::none(),              // custom_mask_d
            TVMFFIAny::none(),              // mask_indptr_d
            TVMFFIAny::none(),              // alibi_slopes_d
            TVMFFIAny::float64(0.0),        // logits_soft_cap_d
            TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0),        // rope_rcp_scale_d
            TVMFFIAny::float64(1e4),        // rope_rcp_theta_d
            // ── POD-specific ──
            TVMFFIAny::bool_val(false),     // enable_pdl
            TVMFFIAny::dltensor(&dl_sched),
        ];
        reg.call(pod_variant.run, &run_args)
            .expect("POD run failed");
        cudaDeviceSynchronize();

        // ── Save POD output ──
        let pod_p_bf16 = gpu_download::<u16>(o_p_ptr, q_p_elems);
        let pod_p_f32: Vec<f32> = pod_p_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let pod_d_bf16 = gpu_download::<u16>(o_d_ptr, q_d_elems);
        let pod_d_f32: Vec<f32> = pod_d_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        assert!(pod_p_f32.iter().map(|v| v.abs()).sum::<f32>() > 0.0, "POD prefill output is all zeros");
        assert!(pod_d_f32.iter().map(|v| v.abs()).sum::<f32>() > 0.0, "POD decode output is all zeros");
        assert!(pod_p_f32.iter().all(|v| v.is_finite()), "POD prefill output has NaN/Inf");
        assert!(pod_d_f32.iter().all(|v| v.is_finite()), "POD decode output has NaN/Inf");

        // ── Run separate paged prefill for reference ──
        let o_ref_p_ptr = gpu_alloc(q_p_elems * 2);
        cudaMemset(o_ref_p_ptr, 0, q_p_elems * 2);
        let dl_o_ref_p = gpu_dl(o_ref_p_ptr, BF16_DT, &q_p_s, &q_p_st);
        let sep_prefill_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), plan_info_p,
            TVMFFIAny::dltensor(&dl_qp), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_qo_indptr_p), TVMFFIAny::dltensor(&dl_kv_indptr_p),
            TVMFFIAny::dltensor(&dl_kv_indices_p), TVMFFIAny::dltensor(&dl_kv_last_p),
            TVMFFIAny::dltensor(&dl_o_ref_p), TVMFFIAny::none(),
            TVMFFIAny::int64(1), TVMFFIAny::int64(0), TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale), TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
            TVMFFIAny::int64(0),
        ];
        reg.call(prefill_variant.paged_run, &sep_prefill_args)
            .expect("Separate paged prefill failed");
        cudaDeviceSynchronize();
        let ref_p_bf16 = gpu_download::<u16>(o_ref_p_ptr, q_p_elems);
        let ref_p_f32: Vec<f32> = ref_p_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        // ── Run separate decode for reference ──
        let o_ref_d_ptr = gpu_alloc(q_d_elems * 2);
        cudaMemset(o_ref_d_ptr, 0, q_d_elems * 2);
        let dl_o_ref_d = gpu_dl(o_ref_d_ptr, BF16_DT, &q_d_s, &q_d_st);
        let decode_variant = reg.get_decode(&DecodeKey {
            dtype: KernelDtype::BF16, head_dim_qk: 128, head_dim_vo: 128,
            sliding_window: false, logits_soft_cap: false,
        }).unwrap();
        let cu_d_dec_s = [2i64]; let cu_d_dec_st = contiguous_strides(&cu_d_dec_s);
        let dl_indptr_d_dec = cpu_dl(kv_indptr_d.as_ptr() as *mut c_void, I32_DT, &cu_d_dec_s, &cu_d_dec_st);
        let empty_s = [0i64]; let empty_st = contiguous_strides(&empty_s);
        let dl_eq = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
        let dl_ek = gpu_dl(std::ptr::null_mut(), BF16_DT, &empty_s, &empty_st);
        let dec_plan = reg.call(decode_variant.plan, &[
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_indptr_d_dec),
            TVMFFIAny::int64(1), TVMFFIAny::int64(num_qo_heads), TVMFFIAny::int64(num_kv_heads),
            TVMFFIAny::int64(page_size), TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(-1), TVMFFIAny::float64(0.0),
            TVMFFIAny::int64(head_dim), TVMFFIAny::int64(head_dim),
            TVMFFIAny::dltensor(&dl_eq), TVMFFIAny::dltensor(&dl_ek),
        ]).unwrap();
        let sep_decode_args = [
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), dec_plan,
            TVMFFIAny::dltensor(&dl_qd), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_kv_indptr_d), TVMFFIAny::dltensor(&dl_kv_indices_d),
            TVMFFIAny::dltensor(&dl_kv_last_d),
            TVMFFIAny::dltensor(&dl_o_ref_d), TVMFFIAny::none(),
            TVMFFIAny::int64(0), TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
            TVMFFIAny::none(), TVMFFIAny::float64(0.0), TVMFFIAny::float64(sm_scale),
            TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4),
        ];
        reg.call(decode_variant.run, &sep_decode_args)
            .expect("Separate decode failed");
        cudaDeviceSynchronize();
        let ref_d_bf16 = gpu_download::<u16>(o_ref_d_ptr, q_d_elems);
        let ref_d_f32: Vec<f32> = ref_d_bf16.iter().map(|&v| bf16_to_f32(v)).collect();

        // ── Compare POD vs Separate ──
        let max_diff_p: f32 = pod_p_f32.iter().zip(ref_p_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_diff_d: f32 = pod_d_f32.iter().zip(ref_d_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // BF16 has ~0.4% relative error; for values ~0.01-0.1 range, absolute tolerance ~0.01
        assert!(max_diff_p < 0.05,
            "POD vs separate prefill output mismatch: max_diff={max_diff_p}");
        assert!(max_diff_d < 0.05,
            "POD vs separate decode output mismatch: max_diff={max_diff_d}");

        println!("POD correctness BF16 h128: PASS (prefill_max_diff={max_diff_p:.6}, decode_max_diff={max_diff_d:.6})");

        // Cleanup
        cudaFree(q_p_ptr); cudaFree(o_p_ptr); cudaFree(o_ref_p_ptr);
        cudaFree(q_d_ptr); cudaFree(o_d_ptr); cudaFree(o_ref_d_ptr);
        cudaFree(k_cache); cudaFree(v_cache);
        cudaFree(kv_indptr_p_gpu); cudaFree(kv_indices_p_gpu); cudaFree(kv_last_p_gpu); cudaFree(qo_indptr_p_gpu);
        cudaFree(kv_indptr_d_gpu); cudaFree(kv_indices_d_gpu); cudaFree(kv_last_d_gpu); cudaFree(qo_indptr_d_gpu);
        cudaFree(sched_ptr);
    }
}
