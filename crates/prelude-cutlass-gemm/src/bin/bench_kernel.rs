//! CUTLASS GEMM performance benchmark — SM90 vs SM80 vs cuBLAS/cuBLASLt.
//!
//! Run:  cargo run -p prelude-cutlass-gemm --bin bench_kernel --release

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{CudaBlas, Gemm as _, GemmConfig, StridedBatchedConfig, sys};
use cudarc::cublaslt::{self, CudaBlasLT, MatmulShared as _};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};

// ── GPU context ─────────────────────────────────────────────────────────

struct Gpu {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    blas_lt: CudaBlasLT,
}

impl Gpu {
    fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.new_stream().ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let blas_lt = CudaBlasLT::new(stream.clone()).ok()?;
        Some(Self { stream, blas, blas_lt })
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
const REPEATS: usize = 50;

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

fn ratio_marker(r: f64) -> &'static str {
    if r <= 1.05 { " " } else if r <= 1.3 { "~" } else if r >= 2.0 { "!" } else { "" }
}

// ── Random data ─────────────────────────────────────────────────────────

fn rand_bf16(len: usize) -> Vec<half::bf16> {
    use rand::RngExt;
    let mut rng = rand::rng();
    (0..len).map(|_| half::bf16::from_f32(rng.random_range(-0.5f32..0.5f32))).collect()
}

fn rand_fp16(len: usize) -> Vec<half::f16> {
    use rand::RngExt;
    let mut rng = rand::rng();
    (0..len).map(|_| half::f16::from_f32(rng.random_range(-0.5f32..0.5f32))).collect()
}

fn rand_f32(len: usize) -> Vec<f32> {
    use rand::RngExt;
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(-0.5f32..0.5f32)).collect()
}

// ── Model shapes ────────────────────────────────────────────────────────

fn models() -> Vec<(&'static str, usize, usize, usize)> {
    // (name, hidden, intermediate, vocab)
    vec![
        ("Qwen3-0.6B", 1024, 3072, 151936),
        ("Qwen3-8B", 4096, 11008, 151936),
    ]
}

fn token_counts() -> Vec<(usize, &'static str)> {
    vec![(1, "decode"), (4, "batch4"), (128, "pre128"), (512, "pre512")]
}

// ── CUTLASS call wrapper ────────────────────────────────────────────────

fn cutlass_gemm_bf16(
    weight: &CudaSlice<half::bf16>,
    input: &CudaSlice<half::bf16>,
    output: &mut CudaSlice<half::bf16>,
    m: usize, n: usize, k: usize,
    gpu: &Gpu,
) {
    let (wp, _g1) = weight.device_ptr(&gpu.stream);
    let (ip, _g2) = input.device_ptr(&gpu.stream);
    let (op, _g3) = output.device_ptr_mut(&gpu.stream);
    unsafe {
        prelude_cutlass_gemm::gemm_dispatch(
            wp as *const c_void, ip as *const c_void, op as *mut c_void,
            n as i32, m as i32, k as i32, 1,
            k as i32, k as i32, n as i32, 0, 0, 0,
            true, false, 0, gpu.stream_ptr(),
        ).ok();
    }
}

fn cutlass_sm80_bf16(
    weight: &CudaSlice<half::bf16>,
    input: &CudaSlice<half::bf16>,
    output: &mut CudaSlice<half::bf16>,
    m: usize, n: usize, k: usize,
    config: i32, gpu: &Gpu,
) {
    let (wp, _g1) = weight.device_ptr(&gpu.stream);
    let (ip, _g2) = input.device_ptr(&gpu.stream);
    let (op, _g3) = output.device_ptr_mut(&gpu.stream);
    unsafe {
        prelude_cutlass_gemm::gemm_sm80(
            wp as *const c_void, ip as *const c_void, op as *mut c_void,
            n as i32, m as i32, k as i32, 0, config, gpu.stream_ptr(),
        ).ok();
    }
}

/// Generic CUTLASS dispatch — works for any dtype by taking raw pointers.
fn cutlass_dispatch(
    weight: *const c_void, input: *const c_void, output: *mut c_void,
    m: usize, n: usize, k: usize, batch: usize,
    stride_a: i64, stride_b: i64, stride_d: i64,
    dtype: u32, gpu: &Gpu,
) {
    unsafe {
        prelude_cutlass_gemm::gemm_dispatch(
            weight, input, output,
            n as i32, m as i32, k as i32, batch as i32,
            k as i32, k as i32, n as i32,
            stride_a, stride_b, stride_d,
            true, false, dtype, gpu.stream_ptr(),
        ).ok();
    }
}

fn cublas_gemm_bf16(
    weight: &CudaSlice<half::bf16>,
    input: &CudaSlice<half::bf16>,
    output: &mut CudaSlice<half::bf16>,
    m: usize, n: usize, k: usize,
    gpu: &Gpu,
) {
    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_T,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: k as i32,
        alpha: half::bf16::from_f32(1.0),
        lda: k as i32, ldb: k as i32,
        beta: half::bf16::from_f32(0.0),
        ldc: n as i32,
    };
    unsafe { gpu.blas.gemm(cfg, weight, input, output).unwrap(); }
}

fn cublas_gemm_fp16(
    weight: &CudaSlice<half::f16>, input: &CudaSlice<half::f16>,
    output: &mut CudaSlice<half::f16>, m: usize, n: usize, k: usize, gpu: &Gpu,
) {
    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_T,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: k as i32,
        alpha: half::f16::from_f32(1.0), lda: k as i32, ldb: k as i32,
        beta: half::f16::from_f32(0.0), ldc: n as i32,
    };
    unsafe { gpu.blas.gemm(cfg, weight, input, output).unwrap(); }
}

fn cublas_gemm_f32(
    weight: &CudaSlice<f32>, input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>, m: usize, n: usize, k: usize, gpu: &Gpu,
) {
    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_T,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: k as i32,
        alpha: 1.0f32, lda: k as i32, ldb: k as i32,
        beta: 0.0f32, ldc: n as i32,
    };
    unsafe { gpu.blas.gemm(cfg, weight, input, output).unwrap(); }
}

// ── Simple bench table printer ──────────────────────────────────────────

fn print_header(title: &str) {
    println!("\n{:=<80}", format!("= {title} "));
    print!("{:<10} {:<8} {:>9} {:>9} {:>7} {:>7}", "tokens", "layer", "SM90", "cuBLAS", "vs_cub", "TFLOPS");
    println!();
    println!("{}", "-".repeat(80));
}

// ============================================================================
// All benchmarks in a single test to avoid GPU memory contention from parallel execution.
// Run with: cargo test -p prelude-cutlass-gemm --release --test performance -- --nocapture
// ============================================================================

fn main() {
    let gpu = match Gpu::new() {
        Some(g) => g,
        None => { eprintln!("No CUDA device, skipping"); return; }
    };
    bench_bf16_gemm(&gpu);
    bench_fp16_gemm(&gpu);
    bench_f32_gemm(&gpu);
    bench_fp8_gemm(&gpu);
    bench_batched_bf16(&gpu);
}

// BF16 performance benchmark
// ============================================================================

fn bench_bf16_gemm(gpu: &Gpu) {

    println!("\n{:=<110}", "= BF16 GEMM: CUTLASS SM90 vs SM80 vs cuBLAS ");
    //                                              ──── SM80 fallback ────
    print!("{:<10} {:<8} {:>9}", "", "", "");
    print!("  {:─^23}", " SM80 fallback ");
    println!();
    print!("{:<10} {:<8} {:>9}", "tokens", "layer", "SM90");
    print!(" {:>7} {:>7} {:>7}", "K32s4", "K64s3", "K64s4");
    print!(" {:>9} {:>7} {:>7}", "cuBLAS", "vs_cub", "TFLOPS");
    println!();
    println!("{}", "-".repeat(110));

    for (name, h, i, v) in &models() {
        println!("--- {name} (H={h}, I={i}) ---");
        let layers: Vec<(&str, usize, usize)> = vec![
            ("qkvo", *h, *h), ("gate/up", *i, *h), ("down", *h, *i), ("lm_head", *v, *h),
        ];

        for (m, tok_label) in &token_counts() {
            let m = *m;
            for (layer, n, k) in &layers {
                if *layer == "lm_head" && m > 4 { continue; }

                let input = gpu.upload(&rand_bf16(m * *k));
                let weight = gpu.upload(&rand_bf16(*n * *k));
                let mut out = gpu.alloc_zeros::<half::bf16>(m * *n);

                let sm90_us = bench_us(|| cutlass_gemm_bf16(&weight, &input, &mut out, m, *n, *k, &gpu), &gpu);

                let mut sm80_times = Vec::new();
                for cfg in 0..3i32 {
                    let us = bench_us(|| cutlass_sm80_bf16(&weight, &input, &mut out, m, *n, *k, cfg, &gpu), &gpu);
                    sm80_times.push(us);
                }

                let cub_us = bench_us(|| cublas_gemm_bf16(&weight, &input, &mut out, m, *n, *k, &gpu), &gpu);

                let tf = tflops(m, *n, *k, sm90_us);
                let r = sm90_us / cub_us;

                print!("{tok_label:<10} {layer:<8} {sm90_us:>9.1}");
                for t in &sm80_times { print!(" {t:>7.1}"); }
                print!(" {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T", r, ratio_marker(r));
                println!();
            }
        }
        println!();
    }
}

// ============================================================================
// FP16 performance benchmark
// ============================================================================

fn bench_fp16_gemm(gpu: &Gpu) {
    print_header("FP16 GEMM: CUTLASS SM90 vs cuBLAS");

    for (name, h, i, _v) in &models() {
        println!("--- {name} ---");
        let layers: Vec<(&str, usize, usize)> = vec![("qkvo", *h, *h), ("gate/up", *i, *h), ("down", *h, *i)];
        for (m, tok_label) in &token_counts() {
            let m = *m;
            for (layer, n, k) in &layers {
                let input = gpu.upload(&rand_fp16(m * *k));
                let weight = gpu.upload(&rand_fp16(*n * *k));
                let mut out = gpu.alloc_zeros::<half::f16>(m * *n);

                let sm90_us = bench_us(|| {
                    let (wp, _g1) = weight.device_ptr(&gpu.stream);
                    let (ip, _g2) = input.device_ptr(&gpu.stream);
                    let (op, _g3) = out.device_ptr_mut(&gpu.stream);
                    cutlass_dispatch(wp as _, ip as _, op as _, m, *n, *k, 1, 0, 0, 0, 1, &gpu);
                }, &gpu);
                let cub_us = bench_us(|| cublas_gemm_fp16(&weight, &input, &mut out, m, *n, *k, &gpu), &gpu);

                let tf = tflops(m, *n, *k, sm90_us);
                let r = sm90_us / cub_us;
                println!("{tok_label:<10} {layer:<8} {sm90_us:>9.1} {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T", r, ratio_marker(r));
            }
        }
        println!();
    }
}

// ============================================================================
// F32/TF32 performance benchmark
// ============================================================================

fn bench_f32_gemm(gpu: &Gpu) {
    print_header("F32/TF32 GEMM: CUTLASS SM90 vs cuBLAS");

    for (name, h, i, _v) in &models() {
        println!("--- {name} ---");
        let layers: Vec<(&str, usize, usize)> = vec![("qkvo", *h, *h), ("gate/up", *i, *h), ("down", *h, *i)];
        for (m, tok_label) in &token_counts() {
            let m = *m;
            for (layer, n, k) in &layers {
                let input = gpu.upload(&rand_f32(m * *k));
                let weight = gpu.upload(&rand_f32(*n * *k));
                let mut out = gpu.alloc_zeros::<f32>(m * *n);

                let sm90_us = bench_us(|| {
                    let (wp, _g1) = weight.device_ptr(&gpu.stream);
                    let (ip, _g2) = input.device_ptr(&gpu.stream);
                    let (op, _g3) = out.device_ptr_mut(&gpu.stream);
                    cutlass_dispatch(wp as _, ip as _, op as _, m, *n, *k, 1, 0, 0, 0, 2, &gpu);
                }, &gpu);
                let cub_us = bench_us(|| cublas_gemm_f32(&weight, &input, &mut out, m, *n, *k, &gpu), &gpu);

                let tf = tflops(m, *n, *k, sm90_us);
                let r = sm90_us / cub_us;
                println!("{tok_label:<10} {layer:<8} {sm90_us:>9.1} {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T", r, ratio_marker(r));
            }
        }
        println!();
    }
}

// ============================================================================
// FP8 performance benchmark (SM90 CUTLASS vs cuBLASLt)
// ============================================================================

/// FP8 E4M3 GEMM via cuBLASLt raw API with per-tensor scaling.
/// Input: FP8 E4M3, Output: BF16 (standard cuBLASLt FP8 pattern).
fn cublaslt_fp8_gemm(
    weight: &CudaSlice<u8>, input: &CudaSlice<u8>,
    output_bf16: &mut CudaSlice<half::bf16>,
    scale_a: &CudaSlice<f32>,
    scale_b: &CudaSlice<f32>,
    m: usize, n: usize, k: usize, gpu: &Gpu,
) -> bool {
    use cublaslt::{result as lt, sys as lt_sys};
    let dt_fp8 = lt_sys::cudaDataType_t::CUDA_R_8F_E4M3;
    let dt_bf16 = lt_sys::cudaDataType_t::CUDA_R_16BF;
    let (cm, cn, ck) = (n as u64, m as u64, k as u64);

    let a_layout = lt::create_matrix_layout(dt_fp8, ck, cm, k as i64).unwrap();
    let b_layout = lt::create_matrix_layout(dt_fp8, ck, cn, k as i64).unwrap();
    let c_layout = lt::create_matrix_layout(dt_bf16, cm, cn, n as i64).unwrap();
    let desc = lt::create_matmul_desc(
        lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        lt_sys::cudaDataType_t::CUDA_R_32F,
    ).unwrap();
    let transa: i32 = 1;
    let (sa, _g_sa) = scale_a.device_ptr(&gpu.stream);
    let (sb, _g_sb) = scale_b.device_ptr(&gpu.stream);
    unsafe {
        lt::set_matmul_desc_attribute(desc,
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&transa) as *const _ as *const _, 4).unwrap();
        lt::set_matmul_desc_attribute(desc,
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            (&sa) as *const _ as *const _, std::mem::size_of::<u64>()).unwrap();
        lt::set_matmul_desc_attribute(desc,
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            (&sb) as *const _ as *const _, std::mem::size_of::<u64>()).unwrap();
    }

    let pref = lt::create_matmul_pref().unwrap();
    let ws_size: usize = 33_554_432;
    unsafe {
        lt::set_matmul_pref_attribute(pref,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&ws_size) as *const _ as *const _, 8).unwrap();
    }
    let heuristic = match unsafe {
        lt::get_matmul_algo_heuristic(
            *gpu.blas_lt.handle(), desc, a_layout, b_layout, c_layout, c_layout, pref,
        )
    } {
        Ok(h) => h,
        Err(_) => {
            unsafe {
                lt::destroy_matmul_desc(desc).ok();
                lt::destroy_matrix_layout(a_layout).ok();
                lt::destroy_matrix_layout(b_layout).ok();
                lt::destroy_matrix_layout(c_layout).ok();
                lt::destroy_matmul_pref(pref).ok();
            }
            return false;
        }
    };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let (a, _g1) = weight.device_ptr(&gpu.stream);
    let (b, _g2) = input.device_ptr(&gpu.stream);
    let (c, _g3) = output_bf16.device_ptr_mut(&gpu.stream);
    let ws = gpu.alloc_zeros::<u8>(ws_size);
    let (w, _g4) = ws.device_ptr(&gpu.stream);
    unsafe {
        lt::matmul(
            *gpu.blas_lt.handle(), desc,
            (&alpha) as *const _ as *const _, (&beta) as *const _ as *const _,
            a as *const _, a_layout, b as *const _, b_layout,
            c as *const _, c_layout, c as *mut _, c_layout,
            (&heuristic.algo) as *const _, w as *mut _, ws_size,
            gpu.stream.cu_stream() as *mut _,
        ).unwrap();
    }
    unsafe {
        lt::destroy_matmul_desc(desc).unwrap();
        lt::destroy_matrix_layout(a_layout).unwrap();
        lt::destroy_matrix_layout(b_layout).unwrap();
        lt::destroy_matrix_layout(c_layout).unwrap();
        lt::destroy_matmul_pref(pref).unwrap();
    }
    true
}

fn bench_fp8_gemm(gpu: &Gpu) {
    print_header("FP8 E4M3 GEMM: CUTLASS SM90 vs cuBLASLt");

    for (name, h, i, _v) in &models() {
        println!("--- {name} ---");
        let layers: Vec<(&str, usize, usize)> = vec![("qkvo", *h, *h), ("gate/up", *i, *h), ("down", *h, *i)];
        for (m, tok_label) in &token_counts() {
            let m = *m;
            for (layer, n, k) in &layers {
                // Generate f32 data → compute per-tensor scale → quantize to FP8
                let a_f32 = rand_f32(m * *k);
                let b_f32 = rand_f32(*n * *k);
                let amax_a = a_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12);
                let amax_b = b_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12);
                let scale_a = 448.0 / amax_a;  // scale to fill FP8 E4M3 range
                let scale_b = 448.0 / amax_b;
                let descale_a = 1.0 / scale_a;  // cuBLASLt multiplies output by descale_a * descale_b
                let descale_b = 1.0 / scale_b;

                let fp8_data: Vec<u8> = a_f32.iter()
                    .map(|&x| float8::F8E4M3::from_f32(x * scale_b).to_bits()).collect();
                let fp8_weight: Vec<u8> = b_f32.iter()
                    .map(|&x| float8::F8E4M3::from_f32(x * scale_a).to_bits()).collect();
                let input = gpu.upload(&fp8_data);
                let weight = gpu.upload(&fp8_weight);
                let scale_a_gpu = gpu.upload(&[descale_a]);
                let scale_b_gpu = gpu.upload(&[descale_b]);
                let mut out = gpu.alloc_zeros::<u8>(m * *n);

                // Check if CUTLASS supports this shape
                let res = {
                    let (wp, _g1) = weight.device_ptr(&gpu.stream);
                    let (ip, _g2) = input.device_ptr(&gpu.stream);
                    let (op, _g3) = out.device_ptr_mut(&gpu.stream);
                    unsafe {
                        prelude_cutlass_gemm::gemm_dispatch(
                            wp as _, ip as _, op as _,
                            *n as i32, m as i32, *k as i32, 1,
                            *k as i32, *k as i32, *n as i32, 0, 0, 0,
                            true, false, 3, gpu.stream_ptr(),
                        )
                    }
                };
                if res.is_err() {
                    println!("{tok_label:<10} {layer:<8} (skipped)");
                    continue;
                }

                let sm90_us = bench_us(|| {
                    let (wp, _g1) = weight.device_ptr(&gpu.stream);
                    let (ip, _g2) = input.device_ptr(&gpu.stream);
                    let (op, _g3) = out.device_ptr_mut(&gpu.stream);
                    cutlass_dispatch(wp as _, ip as _, op as _, m, *n, *k, 1, 0, 0, 0, 3, &gpu);
                }, &gpu);

                let mut out_bf16 = gpu.alloc_zeros::<half::bf16>(m * *n);
                let cub_ok = cublaslt_fp8_gemm(&weight, &input, &mut out_bf16, &scale_a_gpu, &scale_b_gpu, m, *n, *k, &gpu);
                gpu.sync();

                let tf = tflops(m, *n, *k, sm90_us);
                if cub_ok {
                    let cub_us = bench_us(|| {
                        cublaslt_fp8_gemm(&weight, &input, &mut out_bf16, &scale_a_gpu, &scale_b_gpu, m, *n, *k, &gpu);
                    }, &gpu);
                    let r = sm90_us / cub_us;
                    println!("{tok_label:<10} {layer:<8} {sm90_us:>9.1} {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T", r, ratio_marker(r));
                } else {
                    println!("{tok_label:<10} {layer:<8} {sm90_us:>9.1}       N/A         {tf:>6.1}T");
                }
            }
        }
        println!();
    }
}

// ============================================================================
// Batched BF16 performance benchmark
// ============================================================================

fn cublas_batched_bf16(
    weight: &CudaSlice<half::bf16>, input: &CudaSlice<half::bf16>,
    output: &mut CudaSlice<half::bf16>,
    batch: usize, m: usize, n: usize, k: usize, gpu: &Gpu,
) {
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_T,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha: half::bf16::from_f32(1.0),
            lda: k as i32, ldb: k as i32,
            beta: half::bf16::from_f32(0.0),
            ldc: n as i32,
        },
        batch_size: batch as i32,
        stride_a: (n * k) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };
    unsafe { gpu.blas.gemm_strided_batched(cfg, weight, input, output).unwrap(); }
}

fn bench_batched_bf16(gpu: &Gpu) {
    println!("\n{:=<90}", "= Batched BF16 GEMM: CUTLASS SM90 vs cuBLAS ");
    print!("{:<8} {:<6} {:<6} {:<6} {:>9} {:>9} {:>7} {:>7}",
        "batch", "M", "N", "K", "SM90", "cuBLAS", "vs_cub", "TFLOPS");
    println!();
    println!("{}", "-".repeat(90));

    let shapes = [
        (2, 16, 1024, 1024), (4, 8, 1024, 1024), (8, 1, 1024, 1024),
        (2, 128, 3072, 1024), (4, 32, 3072, 1024),
    ];
    for (batch, m, n, k) in &shapes {
        let total_a = batch * m * k;
        let total_b = batch * n * k;
        let total_o = batch * m * n;
        let input = gpu.upload(&rand_bf16(total_a));
        let weight = gpu.upload(&rand_bf16(total_b));
        let mut out = gpu.alloc_zeros::<half::bf16>(total_o);

        let stride_a = (n * k) as i64;
        let stride_b = (m * k) as i64;
        let stride_d = (m * n) as i64;

        let sm90_us = bench_us(|| {
            let (wp, _g1) = weight.device_ptr(&gpu.stream);
            let (ip, _g2) = input.device_ptr(&gpu.stream);
            let (op, _g3) = out.device_ptr_mut(&gpu.stream);
            cutlass_dispatch(wp as _, ip as _, op as _, *m, *n, *k, *batch, stride_a, stride_b, stride_d, 0, &gpu);
        }, &gpu);

        let cub_us = bench_us(|| {
            cublas_batched_bf16(&weight, &input, &mut out, *batch, *m, *n, *k, &gpu);
        }, &gpu);

        let tf = tflops(*batch * *m, *n, *k, sm90_us);
        let r = sm90_us / cub_us;
        println!("{batch:<8} {m:<6} {n:<6} {k:<6} {sm90_us:>9.1} {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T", r, ratio_marker(r));
    }
}
