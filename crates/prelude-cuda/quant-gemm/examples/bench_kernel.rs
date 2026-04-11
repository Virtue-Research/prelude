//! Quantized GEMM benchmark: MMVQ (decode M=1) + Tiled MMQ (prefill M>1).
//!
//! Run: cargo run -p quant-gemm --example bench_kernel --release

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};
use quant_gemm::GgmlType;

// ── GPU context ─────────────────────────────────────────────────────────

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

    fn sync(&self) { self.stream.synchronize().unwrap(); }
}

// ── Timing ──────────────────────────────────────────────────────────────

const WARMUP: usize = 10;
const REPEATS: usize = 100;

fn bench_us(mut f: impl FnMut(), gpu: &Gpu) -> f64 {
    for _ in 0..WARMUP { f(); }
    gpu.sync();
    let t = Instant::now();
    for _ in 0..REPEATS { f(); }
    gpu.sync();
    t.elapsed().as_nanos() as f64 / REPEATS as f64 / 1000.0
}

fn tflops(m: usize, n: usize, k: usize, us: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12
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

// ── Model shapes (same as cutlass/deepgemm benchmarks) ──────────────────

fn models() -> Vec<(&'static str, usize, usize)> {
    vec![
        ("Qwen3-0.6B", 1024, 3072),
        ("Qwen3-8B",   4096, 11008),
    ]
}

fn layers(h: usize, i: usize) -> Vec<(&'static str, usize, usize)> {
    vec![("qkvo", h, h), ("gate/up", i, h), ("down", h, i)]
}

// ── Format groups ───────────────────────────────────────────────────────

const MMVQ_FORMATS: &[(GgmlType, &str)] = &[
    (GgmlType::Q4_0,  "Q4_0"), (GgmlType::Q4K,  "Q4_K"), (GgmlType::Q6K,  "Q6_K"),
    (GgmlType::Q8_0,  "Q8_0"), (GgmlType::IQ4XS, "IQ4X"), (GgmlType::IQ3S, "IQ3S"),
    (GgmlType::IQ2XS, "IQ2X"), (GgmlType::IQ1M,  "IQ1M"), (GgmlType::MXFP4,"MXF4"),
    (GgmlType::NVFP4, "NVF4"),
];

// All formats with MMQ template instantiation (excludes IQ1_M which has no MMQ upstream).
const MMQ_FORMATS: &[(GgmlType, &str)] = &[
    (GgmlType::Q4_0,  "Q4_0"), (GgmlType::Q4K,  "Q4_K"), (GgmlType::Q6K,  "Q6_K"),
    (GgmlType::Q8_0,  "Q8_0"), (GgmlType::IQ4XS, "IQ4X"), (GgmlType::IQ3S, "IQ3S"),
    (GgmlType::IQ2XS, "IQ2X"), (GgmlType::IQ1S,  "IQ1S"), (GgmlType::MXFP4,"MXF4"),
    (GgmlType::NVFP4, "NVF4"),
];

// ── MMVQ helper ─────────────────────────────────────────────────────────

fn bench_mmvq(t: GgmlType, n: usize, k: usize, gpu: &Gpu) -> f64 {
    let qk = block_elems(t);
    if k % qk != 0 { return 0.0; }
    let raw_w = random_bytes(n * (k / qk) * block_bytes(t), 42 + t as u64);
    let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let q8_data = quantize_f32_to_q8_1(&x_f32);
    let d_w = gpu.upload(&raw_w);
    let d_q8 = gpu.upload(&q8_data);
    let mut d_y: CudaSlice<f32> = gpu.alloc_zeros(n);

    bench_us(|| unsafe {
        let (wp, _g1) = d_w.device_ptr(&gpu.stream);
        let (qp, _g2) = d_q8.device_ptr(&gpu.stream);
        let (yp, _g3) = d_y.device_ptr_mut(&gpu.stream);
        quant_gemm::mul_mat_vec_q(
            wp as *const c_void, qp as *const c_void, yp as *mut f32,
            n as i64, k as i64, t, gpu.stream_ptr(),
        );
    }, gpu)
}

// ── Tiled MMQ helper ────────────────────────────────────────────────────

fn bench_mmq(t: GgmlType, m: usize, n: usize, k: usize, gpu: &Gpu) -> f64 {
    let qk = block_elems(t);
    if k % qk != 0 { return 0.0; }
    let raw_w = random_bytes(n * (k / qk) * block_bytes(t), 42 + t as u64);
    let x_bf16: Vec<half::bf16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.007).cos()))
        .collect();
    let d_w = gpu.upload(&raw_w);
    let d_x = gpu.upload(&x_bf16);
    let ne00_padded = ((k + 511) / 512) * 512;
    let q8_bytes = m * ne00_padded * 36 / 32 + 128 * 144;
    let mut d_q8: CudaSlice<u8> = gpu.alloc_zeros(q8_bytes);
    let mut d_y: CudaSlice<f32> = gpu.alloc_zeros(m * n);

    bench_us(|| unsafe {
        let (xp, _g1) = d_x.device_ptr(&gpu.stream);
        let (qp, _g2) = d_q8.device_ptr_mut(&gpu.stream);
        let (wp, _g3) = d_w.device_ptr(&gpu.stream);
        let (yp, _g4) = d_y.device_ptr_mut(&gpu.stream);
        quant_gemm::quantize_q8_1(
            xp as *const c_void, qp as *mut c_void,
            m as i64, k as i64, t, gpu.stream_ptr(),
        );
        quant_gemm::mul_mat_q(
            wp as *const c_void, qp as *const c_void, yp as *mut f32,
            m as i64, n as i64, k as i64, t, 0, gpu.stream_ptr(),
        );
    }, gpu)
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let gpu = Gpu::new().expect("No CUDA device found");

    // ── MMVQ: decode (M=1), one format per column ───────────────────────

    println!("\n{:=<100}", "= Quantized MMVQ: decode (M=1) ");

    let fmt_header: String = MMVQ_FORMATS.iter().map(|(_, n)| format!("{n:>8}")).collect::<Vec<_>>().join("");
    println!("{:<10} {:<8}{fmt_header}", "tokens", "layer");
    println!("{}", "-".repeat(18 + MMVQ_FORMATS.len() * 8));

    for (name, h, i) in models() {
        println!("--- {name} (H={h}, I={i}) ---");
        for (layer, n, k) in layers(h, i) {
            print!("{:<10} {layer:<8}", "decode");
            for &(t, _) in MMVQ_FORMATS {
                let us = bench_mmvq(t, n, k, &gpu);
                if us == 0.0 { print!("{:>8}", "-"); }
                else         { print!("{us:>8.1}"); }
            }
            println!();
        }
    }

    gpu.sync();

    // ── Tiled MMQ: prefill (M>1), one format per column ─────────────────

    println!("\n{:=<100}", "= Quantized MMQ: prefill ");

    let fmt_header: String = MMQ_FORMATS.iter().map(|(_, n)| format!("{n:>9}")).collect::<Vec<_>>().join("");
    println!("{:<10} {:<8}{fmt_header}  TFLOPS", "tokens", "layer");
    println!("{}", "-".repeat(18 + MMQ_FORMATS.len() * 9 + 8));

    for (name, h, i) in models() {
        println!("--- {name} (H={h}, I={i}) ---");
        for &m in &[32usize, 128, 512] {
            let tok_label = format!("pre{m}");
            for (layer, n, k) in layers(h, i) {
                print!("{tok_label:<10} {layer:<8}");
                let mut first_tf = 0.0;
                for &(t, _) in MMQ_FORMATS {
                    let us = bench_mmq(t, m, n, k, &gpu);
                    if us == 0.0 { print!("{:>9}", "-"); }
                    else {
                        print!("{us:>9.1}");
                        if first_tf == 0.0 { first_tf = tflops(m, n, k, us); }
                    }
                }
                // Show TFLOPS once (all formats have similar throughput)
                if first_tf > 0.0 { print!("  {first_tf:>5.1}T"); }
                println!();
            }
        }
    }
}
