//! Quantized GEMM benchmark: MMVQ (decode M=1) + Tiled MMQ (prefill M>1).
//!
//! Run: cargo run -p prelude-quant-gemm --bin bench_kernel --release

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};
use prelude_quant_gemm::GgmlType;

// ── GPU context ─────────────────────────────────────────────────────────

struct Gpu {
    stream: Arc<CudaStream>,
    compute_cap: i32,
}

impl Gpu {
    fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.new_stream().ok()?;
        // Get compute capability from device properties
        let cc = 80; // default; TODO: query from device
        Some(Self { stream, compute_cap: cc })
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

    fn sync(&self) { self.stream.synchronize().unwrap(); }
}

// ── Timing ──────────────────────────────────────────────────────────────

const WARMUP: usize = 10;
const REPEATS: usize = 100;

fn bench_us(mut f: impl FnMut(), gpu: &Gpu) -> f64 {
    for _ in 0..WARMUP { f(); }
    gpu.sync();
    let start = Instant::now();
    for _ in 0..REPEATS { f(); }
    gpu.sync();
    start.elapsed().as_nanos() as f64 / REPEATS as f64 / 1000.0
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

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let gpu = Gpu::new().expect("No CUDA device found");
    println!("prelude-quant-gemm benchmark\n");

    // ── MMVQ benchmark (decode M=1) ─────────────────────────────────────

    println!("╔═══════════════════════════════════════════╗");
    println!("║  MMVQ Throughput (decode M=1, llama.cpp)  ║");
    println!("╚═══════════════════════════════════════════╝\n");
    println!("  W[N,K] @ x[K] → y[N]");
    println!("  warmup={WARMUP} repeats={REPEATS}\n");

    let mmvq_formats: &[(GgmlType, &str)] = &[
        (GgmlType::Q4_0,   "Q4_0"),
        (GgmlType::Q4K,    "Q4_K"),
        (GgmlType::Q6K,    "Q6_K"),
        (GgmlType::IQ4NL,  "IQ4NL"),
        (GgmlType::IQ4XS,  "IQ4XS"),
        (GgmlType::IQ3S,   "IQ3S"),
        (GgmlType::IQ2XS,  "IQ2XS"),
        (GgmlType::MXFP4,  "MXFP4"),
    ];

    let sizes: &[(usize, usize, &str)] = &[
        (4096,  4096,  "4K×4K"),
        (11008, 4096,  "11K×4K"),
        (4096,  11008, "4K×11K"),
    ];

    for &(n, k, label) in sizes {
        print!("  {label:>7}");
        for &(t, name) in mmvq_formats {
            let qk = block_elems(t);
            if k % qk != 0 { print!("  {name}=skip"); continue; }

            let total_w = n * (k / qk) * block_bytes(t);
            let raw_w = random_bytes(total_w, 42 + t as u64);
            let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
            let q8_data = quantize_f32_to_q8_1(&x_f32);

            let d_w = gpu.upload(&raw_w);
            let d_q8 = gpu.upload(&q8_data);
            let mut d_y: CudaSlice<f32> = gpu.alloc_zeros(n);

            let us = bench_us(|| unsafe {
                let (wp, _g1) = d_w.device_ptr(&gpu.stream);
                let (qp, _g2) = d_q8.device_ptr(&gpu.stream);
                let (yp, _g3) = d_y.device_ptr_mut(&gpu.stream);
                prelude_quant_gemm::mul_mat_vec_q(
                    wp as *const c_void,
                    qp as *const c_void,
                    yp as *mut f32,
                    n as i64, k as i64, t, gpu.stream_ptr(),
                );
            }, &gpu);

            print!("  {name}={us:.0}us");
        }
        println!();
    }

    // ── Tiled MMQ benchmark (prefill M>1) ───────────────────────────────

    println!("\n╔═══════════════════════════════════════════╗");
    println!("║  Tiled MMQ Throughput (prefill, llama.cpp) ║");
    println!("╚═══════════════════════════════════════════╝\n");
    println!("  Y[M,N] = X[M,K] @ W[N,K]^T");
    println!("  warmup={WARMUP} repeats={REPEATS}\n");

    let mmq_formats: &[(GgmlType, &str)] = &[
        (GgmlType::Q4_0, "Q4_0"),
        (GgmlType::Q4K,  "Q4_K"),
    ];

    let mmq_sizes: &[(usize, usize, usize, &str)] = &[
        (32,  4096, 4096,  "32×4K×4K"),
        (128, 4096, 4096,  "128×4K×4K"),
        (512, 4096, 4096,  "512×4K×4K"),
    ];

    for &(m, n, k, label) in mmq_sizes {
        print!("  {label:>12}");
        for &(t, name) in mmq_formats {
            let qk = block_elems(t);
            if k % qk != 0 { continue; }

            let total_w = n * (k / qk) * block_bytes(t);
            let raw_w = random_bytes(total_w, 42 + t as u64);
            let x_bf16: Vec<half::bf16> = (0..m * k)
                .map(|i| half::bf16::from_f32(((i as f32) * 0.007).cos()))
                .collect();

            let d_w = gpu.upload(&raw_w);
            let d_x = gpu.upload(&x_bf16);
            let q8_mmq_bytes = m * k * 2; // conservative
            let mut d_q8: CudaSlice<u8> = gpu.alloc_zeros(q8_mmq_bytes);
            let mut d_y: CudaSlice<f32> = gpu.alloc_zeros(m * n);

            let us = bench_us(|| unsafe {
                let (xp, _g1) = d_x.device_ptr(&gpu.stream);
                let (qp, _g2) = d_q8.device_ptr_mut(&gpu.stream);
                let (wp, _g3) = d_w.device_ptr(&gpu.stream);
                let (yp, _g4) = d_y.device_ptr_mut(&gpu.stream);
                prelude_quant_gemm::quantize_q8_1(
                    xp as *const c_void, qp as *mut c_void,
                    m as i64, k as i64, t, gpu.stream_ptr(),
                );
                prelude_quant_gemm::mul_mat_q(
                    wp as *const c_void, qp as *const c_void,
                    yp as *mut f32,
                    m as i64, n as i64, k as i64,
                    t, gpu.compute_cap, gpu.stream_ptr(),
                );
            }, &gpu);

            let tflops = 2.0 * m as f64 * n as f64 * k as f64 / us / 1e6;
            print!("  {name}={us:.0}us({tflops:.1}TF)");
        }
        println!();
    }
}
