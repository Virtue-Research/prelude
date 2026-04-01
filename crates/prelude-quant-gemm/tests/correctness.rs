//! Correctness tests for GPU quant kernels vs llama.cpp CPU reference.
//!
//! Run: cargo test -p prelude-quant-gemm --release

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};
use prelude_quant_gemm::GgmlType;

// ── GPU helpers ─────────────────────────────────────────────────────────

struct Gpu {
    stream: Arc<CudaStream>,
}

impl Gpu {
    fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.new_stream().ok()?;
        Some(Self { stream })
    }

    fn stream_ptr(&self) -> *const c_void {
        self.stream.cu_stream() as *const c_void
    }

    fn upload<T: cudarc::driver::DeviceRepr>(&self, data: &[T]) -> CudaSlice<T> {
        self.stream.clone_htod(data).unwrap()
    }

    fn alloc_zeros<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> CudaSlice<T> {
        self.stream.alloc_zeros(len).unwrap()
    }

    fn download_f32(&self, d: &CudaSlice<f32>) -> Vec<f32> {
        self.stream.clone_dtoh(d).unwrap()
    }

    fn download_bf16_as_f32(&self, d: &CudaSlice<half::bf16>) -> Vec<f32> {
        let bf16_vec: Vec<half::bf16> = self.stream.clone_dtoh(d).unwrap();
        bf16_vec.iter().map(|v| v.to_f32()).collect()
    }

    fn sync(&self) { self.stream.synchronize().unwrap(); }
}

// ── Block metadata ──────────────────────────────────────────────────────

fn block_bytes(t: GgmlType) -> usize {
    match t {
        GgmlType::Q4_0   => 18,  GgmlType::Q4_1   => 20,
        GgmlType::Q5_0   => 22,  GgmlType::Q5_1   => 24,
        GgmlType::Q8_0   => 34,  GgmlType::Q2K    => 84,
        GgmlType::Q3K    => 110, GgmlType::Q4K    => 144,
        GgmlType::Q5K    => 176, GgmlType::Q6K    => 210,
        GgmlType::IQ4NL  => 18,  GgmlType::IQ4XS  => 136,
        GgmlType::IQ3S   => 110, GgmlType::IQ3XXS => 98,
        GgmlType::IQ2S   => 82,  GgmlType::IQ2XS  => 74,
        GgmlType::IQ2XXS => 66,  GgmlType::IQ1S   => 50,
        GgmlType::IQ1M   => 56,  GgmlType::MXFP4  => 17,
        GgmlType::NVFP4  => 36,
    }
}

fn block_elems(t: GgmlType) -> usize {
    match t {
        GgmlType::IQ4NL | GgmlType::MXFP4 | GgmlType::Q4_0 | GgmlType::Q4_1
        | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => 32,
        GgmlType::NVFP4 => 64,
        _ => 256,
    }
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u8
    }).collect()
}

// ── CPU reference ───────────────────────────────────────────────────────

fn cpu_dequantize(raw: &[u8], num_elements: usize, t: GgmlType) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    unsafe {
        prelude_quant_gemm::dequantize_ref(
            raw.as_ptr() as *const c_void,
            out.as_mut_ptr(),
            num_elements as i64,
            t,
        );
    }
    out
}

// ── Q8_1 quantization (CPU) ─────────────────────────────────────────────

const QK8_1: usize = 32;
const Q8_1_BLOCK_SIZE: usize = 36;

fn quantize_f32_to_q8_1(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK8_1, 0);
    let num_blocks = x.len() / QK8_1;
    let mut out = vec![0u8; num_blocks * Q8_1_BLOCK_SIZE];
    for ib in 0..num_blocks {
        let blk = &x[ib * QK8_1..(ib + 1) * QK8_1];
        let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut sum = 0i32;
        let off = ib * Q8_1_BLOCK_SIZE;
        for j in 0..QK8_1 {
            let q = (blk[j] * id).round() as i8;
            out[off + 4 + j] = q as u8;
            sum += q as i32;
        }
        let d_f16 = half::f16::from_f32(d);
        let s_f16 = half::f16::from_f32(d * sum as f32);
        out[off..off + 2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
        out[off + 2..off + 4].copy_from_slice(&s_f16.to_bits().to_le_bytes());
    }
    out
}

// ── MMVQ correctness test ───────────────────────────────────────────────

fn test_mmvq(t: GgmlType, label: &str, n: usize, k: usize, gpu: &Gpu) {
    let qk = block_elems(t);
    assert_eq!(k % qk, 0);
    let total_bytes = n * (k / qk) * block_bytes(t);

    let raw_w = random_bytes(total_bytes, 42 + t as u64);
    let w_f32 = cpu_dequantize(&raw_w, n * k, t);
    let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let ref_y: Vec<f32> = (0..n).map(|i| {
        w_f32[i * k..(i + 1) * k].iter().zip(x_f32.iter()).map(|(w, x)| w * x).sum()
    }).collect();

    let q8_data = quantize_f32_to_q8_1(&x_f32);
    let d_w = gpu.upload(&raw_w);
    let d_q8 = gpu.upload(&q8_data);
    let mut d_y: CudaSlice<f32> = gpu.alloc_zeros(n);

    unsafe {
        let (wp, _g1) = d_w.device_ptr(&gpu.stream);
        let (qp, _g2) = d_q8.device_ptr(&gpu.stream);
        let (yp, _g3) = d_y.device_ptr_mut(&gpu.stream);
        prelude_quant_gemm::mul_mat_vec_q(
            wp as *const c_void,
            qp as *const c_void,
            yp as *mut f32,
            n as i64, k as i64, t, gpu.stream_ptr(),
        );
    }
    gpu.sync();

    let gpu_y = gpu.download_f32(&d_y);

    let mut max_rel: f32 = 0.0;
    let mut fail_count = 0usize;
    for (r, g) in ref_y.iter().zip(gpu_y.iter()) {
        let err = (r - g).abs();
        let denom = r.abs().max(1e-6);
        let rel = err / denom;
        max_rel = max_rel.max(rel);
        if rel > 0.2 && err > 1.0 { fail_count += 1; }
    }

    println!("  {label:>6} [{n:>4}×{k:>5}]  {}  max_rel={max_rel:.4}  fail={fail_count}/{n}",
        if fail_count == 0 { "PASS" } else { "FAIL" });
    assert_eq!(fail_count, 0, "{label} [{n}×{k}]: {fail_count}/{n} failed (max_rel={max_rel})");
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn dequantize_ref_smoke() {
    let formats: &[(GgmlType, &str)] = &[
        (GgmlType::Q4_0, "Q4_0"), (GgmlType::Q4K, "Q4_K"),
        (GgmlType::IQ4NL, "IQ4NL"), (GgmlType::IQ4XS, "IQ4XS"),
        (GgmlType::IQ3S, "IQ3S"), (GgmlType::IQ3XXS, "IQ3XXS"),
        (GgmlType::IQ2S, "IQ2S"), (GgmlType::IQ2XS, "IQ2XS"),
        (GgmlType::IQ2XXS, "IQ2XXS"), (GgmlType::IQ1S, "IQ1S"),
        (GgmlType::IQ1M, "IQ1M"), (GgmlType::MXFP4, "MXFP4"),
        (GgmlType::NVFP4, "NVFP4"),
    ];
    for &(t, label) in formats {
        let qk = block_elems(t);
        let n = 256;
        if n % qk != 0 { continue; }
        let raw = random_bytes((n / qk) * block_bytes(t), 99 + t as u64);
        let out = cpu_dequantize(&raw, n, t);
        let finite = out.iter().filter(|v| v.is_finite()).count();
        println!("  {label:>6}: {n} elements, {finite} finite");
        assert!(finite > n / 2, "{label}: too many non-finite ({finite}/{n})");
    }
}

#[test]
fn mmvq_correctness() {
    let gpu = match Gpu::new() {
        Some(g) => g,
        None => { eprintln!("No CUDA device, skipping"); return; }
    };

    let formats: &[(GgmlType, &str)] = &[
        (GgmlType::Q4_0,   "Q4_0"),
        (GgmlType::Q4K,    "Q4_K"),
        (GgmlType::Q6K,    "Q6_K"),
        (GgmlType::IQ4NL,  "IQ4NL"),
        (GgmlType::IQ4XS,  "IQ4XS"),
        (GgmlType::IQ3S,   "IQ3S"),
        (GgmlType::IQ3XXS, "IQ3XX"),
        (GgmlType::IQ2S,   "IQ2S"),
        (GgmlType::IQ2XS,  "IQ2XS"),
        (GgmlType::IQ2XXS, "IQ2XX"),
        (GgmlType::IQ1S,   "IQ1S"),
        (GgmlType::IQ1M,   "IQ1M"),
        (GgmlType::MXFP4,  "MXFP4"),
        (GgmlType::NVFP4,  "NVFP4"),
    ];

    println!("\n=== MMVQ Correctness (GPU vs llama.cpp CPU) ===\n");
    for &(n, k) in &[(64usize, 1024usize), (128, 4096)] {
        for &(t, label) in formats {
            if k % block_elems(t) != 0 { continue; }
            test_mmvq(t, label, n, k, &gpu);
        }
        println!();
    }
}

// ── GPU dequantize correctness: gpu_dequantize vs dequantize_ref ────────

fn test_gpu_dequantize(t: GgmlType, label: &str, num_elements: usize, gpu: &Gpu) {
    let qk = block_elems(t);
    assert_eq!(num_elements % qk, 0);
    let total_bytes = (num_elements / qk) * block_bytes(t);

    let raw = random_bytes(total_bytes, 77 + t as u64);

    // CPU reference
    let ref_f32 = cpu_dequantize(&raw, num_elements, t);

    // GPU dequantize → BF16
    let d_input = gpu.upload(&raw);
    let mut d_output: CudaSlice<half::bf16> = gpu.alloc_zeros(num_elements);

    unsafe {
        let (ip, _g1) = d_input.device_ptr(&gpu.stream);
        let (op, _g2) = d_output.device_ptr_mut(&gpu.stream);
        prelude_quant_gemm::gpu_dequantize(
            ip as *const c_void,
            op as *mut c_void,
            num_elements as i64,
            t,
            gpu.stream_ptr(),
        );
    }
    gpu.sync();

    let gpu_f32 = gpu.download_bf16_as_f32(&d_output);

    // Compare (BF16 has ~7-bit mantissa, so some error expected)
    let mut max_abs: f32 = 0.0;
    let mut fail_count = 0usize;
    for (r, g) in ref_f32.iter().zip(gpu_f32.iter()) {
        let err = (r - g).abs();
        max_abs = max_abs.max(err);
        let tol = 0.02 + 0.02 * r.abs().max(g.abs());
        if err > tol { fail_count += 1; }
    }

    println!("  {label:>6} [{num_elements:>6}]  {}  max_abs={max_abs:.4}  fail={fail_count}/{num_elements}",
        if fail_count == 0 { "PASS" } else { "FAIL" });
    assert_eq!(fail_count, 0,
        "{label} [{num_elements}]: {fail_count}/{num_elements} failed (max_abs={max_abs})");
}

#[test]
fn gpu_dequantize_correctness() {
    let gpu = match Gpu::new() {
        Some(g) => g,
        None => { eprintln!("No CUDA device, skipping"); return; }
    };

    let all_formats: &[(GgmlType, &str)] = &[
        (GgmlType::Q4_0,   "Q4_0"),
        (GgmlType::Q4_1,   "Q4_1"),
        (GgmlType::Q5_0,   "Q5_0"),
        (GgmlType::Q5_1,   "Q5_1"),
        (GgmlType::Q8_0,   "Q8_0"),
        (GgmlType::Q2K,    "Q2_K"),
        (GgmlType::Q3K,    "Q3_K"),
        (GgmlType::Q4K,    "Q4_K"),
        (GgmlType::Q5K,    "Q5_K"),
        (GgmlType::Q6K,    "Q6_K"),
        (GgmlType::IQ4NL,  "IQ4NL"),
        (GgmlType::IQ4XS,  "IQ4XS"),
        (GgmlType::IQ3S,   "IQ3S"),
        (GgmlType::IQ3XXS, "IQ3XX"),
        (GgmlType::IQ2S,   "IQ2S"),
        (GgmlType::IQ2XS,  "IQ2XS"),
        (GgmlType::IQ2XXS, "IQ2XX"),
        (GgmlType::IQ1S,   "IQ1S"),
        (GgmlType::IQ1M,   "IQ1M"),
        (GgmlType::MXFP4,  "MXFP4"),
        (GgmlType::NVFP4,  "NVFP4"),
    ];

    println!("\n=== GPU Dequantize Correctness (GPU BF16 vs llama.cpp CPU F32) ===\n");
    for &n in &[256usize, 1024, 4096] {
        for &(t, label) in all_formats {
            if n % block_elems(t) != 0 { continue; }
            test_gpu_dequantize(t, label, n, &gpu);
        }
        println!();
    }
}

// ── Tiled MMQ correctness: GPU tiled MMQ vs CPU dequant + f32 matmul ────

fn test_tiled_mmq(t: GgmlType, label: &str, m: usize, n: usize, k: usize, gpu: &Gpu) {
    let qk = block_elems(t);
    assert_eq!(k % qk, 0);
    let total_w_bytes = n * (k / qk) * block_bytes(t);

    let raw_w = random_bytes(total_w_bytes, 55 + t as u64);
    let w_f32 = cpu_dequantize(&raw_w, n * k, t);

    // CPU reference: f32 matmul Y[M,N] = X[M,K] @ W[N,K]^T
    let x_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.007).cos()).collect();
    let mut ref_y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..k {
                sum += x_f32[i * k + l] as f64 * w_f32[j * k + l] as f64;
            }
            ref_y[i * n + j] = sum as f32;
        }
    }

    // GPU: BF16 activations → quantize Q8_1 → tiled MMQ
    let x_bf16: Vec<half::bf16> = x_f32.iter().map(|v| half::bf16::from_f32(*v)).collect();
    let d_w = gpu.upload(&raw_w);
    let d_x = gpu.upload(&x_bf16);

    // Q8_1_MMQ buffer (conservative size)
    let ne00_padded = ((k + 511) / 512) * 512;
    let q8_buf_size = m * (ne00_padded / 128) * 144 + m * 144;
    let mut d_q8: CudaSlice<u8> = gpu.alloc_zeros(q8_buf_size);
    let mut d_y: CudaSlice<f32> = gpu.alloc_zeros(m * n);

    unsafe {
        let (xp, _g1) = d_x.device_ptr(&gpu.stream);
        let (qp, _g2) = d_q8.device_ptr_mut(&gpu.stream);
        let (wp, _g3) = d_w.device_ptr(&gpu.stream);
        let (yp, _g4) = d_y.device_ptr_mut(&gpu.stream);
        prelude_quant_gemm::quantize_q8_1(
            xp as *const c_void, qp as *mut c_void,
            m as i64, k as i64, t, gpu.stream_ptr(),
        );
        prelude_quant_gemm::mul_mat_q(
            wp as *const c_void, qp as *const c_void, yp as *mut f32,
            m as i64, n as i64, k as i64,
            t, 0, gpu.stream_ptr(),
        );
    }
    gpu.sync();

    let gpu_y = gpu.download_f32(&d_y);

    let mut max_rel: f32 = 0.0;
    let mut fail_count = 0usize;
    for (r, g) in ref_y.iter().zip(gpu_y.iter()) {
        let err = (r - g).abs();
        let denom = r.abs().max(1e-6);
        let rel = err / denom;
        max_rel = max_rel.max(rel);
        // Double quantization (weight + Q8_1 activation) = higher tolerance
        if rel > 0.3 && err > 2.0 { fail_count += 1; }
    }

    let total = m * n;
    println!("  {label:>5} [{m}×{n}×{k}]  {}  max_rel={max_rel:.4}  fail={fail_count}/{total}",
        if fail_count == 0 { "PASS" } else { "FAIL" });
    assert_eq!(fail_count, 0,
        "{label} [{m}×{n}×{k}]: {fail_count}/{total} failed (max_rel={max_rel})");
}

#[test]
fn tiled_mmq_correctness() {
    let gpu = match Gpu::new() {
        Some(g) => g,
        None => { eprintln!("No CUDA device, skipping"); return; }
    };

    let formats: &[(GgmlType, &str)] = &[
        (GgmlType::Q4_0, "Q4_0"),
        (GgmlType::Q4K,  "Q4_K"),
        (GgmlType::Q6K,  "Q6_K"),
    ];

    println!("\n=== Tiled MMQ Correctness (GPU vs CPU dequant+matmul) ===\n");
    for &(m, n, k) in &[(32usize, 256, 1024), (64, 512, 4096)] {
        for &(t, label) in formats {
            if k % block_elems(t) != 0 { continue; }
            test_tiled_mmq(t, label, m, n, k, &gpu);
        }
        println!();
    }
}
