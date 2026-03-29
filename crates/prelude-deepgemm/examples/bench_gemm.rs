//! DeepGEMM BF16/FP8 GEMM performance benchmark — DeepGEMM vs cuBLAS/cuBLASLt.
//!
//! Run:  cargo run -p prelude-deepgemm --example bench_gemm --release

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{CudaBlas, Gemm as _, GemmConfig, sys};
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

    fn stream_ptr(&self) -> *mut c_void {
        self.stream.cu_stream() as *mut c_void
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

// ── DeepGEMM call wrapper ───────────────────────────────────────────────

fn deepgemm_bf16(
    input: &CudaSlice<half::bf16>,
    weight: &CudaSlice<half::bf16>,
    output: &mut CudaSlice<half::bf16>,
    m: usize, n: usize, k: usize,
    gpu: &Gpu,
) {
    let (ip, _g1) = input.device_ptr(&gpu.stream);
    let (wp, _g2) = weight.device_ptr(&gpu.stream);
    let (op, _g3) = output.device_ptr_mut(&gpu.stream);
    unsafe {
        prelude_deepgemm::bf16_gemm(
            ip as *mut c_void, wp as *mut c_void, op as *mut c_void,
            m as i32, n as i32, k as i32,
            gpu.stream_ptr(),
        ).ok();
    }
}

// ── cuBLAS call wrapper ─────────────────────────────────────────────────

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

// ============================================================================
// Main benchmark
// ============================================================================

fn main() {
    let gpu = match Gpu::new() {
        Some(g) => g,
        None => { eprintln!("No CUDA device, skipping"); return; }
    };

    bench_bf16(&gpu);
    bench_fp8(&gpu);
    bench_grouped(&gpu);
    bench_masked(&gpu);
    bench_masked_fp8(&gpu);
    bench_acc(&gpu);
}

fn bench_bf16(gpu: &Gpu) {
    println!("\n{:=<80}", "= BF16 GEMM: DeepGEMM vs cuBLAS ");
    print!("{:<10} {:<8} {:>9} {:>9} {:>7} {:>7} {:>20}",
        "tokens", "layer", "DeepGEMM", "cuBLAS", "vs_cub", "TFLOPS", "config");
    println!();
    println!("{}", "-".repeat(80));

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

                // Query kernel config for display
                let (bm, bn, stages, _smem) = prelude_deepgemm::query_config(
                    m as i32, *n as i32, *k as i32,
                );

                let dg_us = bench_us(
                    || deepgemm_bf16(&input, &weight, &mut out, m, *n, *k, &gpu),
                    &gpu,
                );

                let cub_us = bench_us(
                    || cublas_gemm_bf16(&weight, &input, &mut out, m, *n, *k, &gpu),
                    &gpu,
                );

                let tf = tflops(m, *n, *k, dg_us);
                let r = dg_us / cub_us;
                let cfg_str = format!("{bm}x{bn}/s{stages}");
                println!(
                    "{tok_label:<10} {layer:<8} {dg_us:>9.1} {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T  {cfg_str:>16}",
                    r, ratio_marker(r),
                );
            }
        }
        println!();
    }
}

// ============================================================================
// FP8 E4M3 GEMM: DeepGEMM vs cuBLASLt
// ============================================================================

/// Quantize f32 → FP8 E4M3 with per-tensor scaling.
/// Returns (fp8_bytes, descale) where descale = amax / 448.
fn quantize_fp8_per_tensor(data: &[f32]) -> (Vec<u8>, f32) {
    let amax = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12);
    let scale = 448.0 / amax;
    let descale = 1.0 / scale;
    let fp8: Vec<u8> = data.iter().map(|&x| {
        let scaled = (x * scale).max(-448.0).min(448.0);
        half::f16::from_f32(scaled).to_bits().to_be_bytes()[0]
    }).collect();
    (fp8, descale)
}

/// Quantize f32 → FP8 E4M3 with per-token scaling for DeepGEMM (gran_k=128).
/// Returns (fp8_bytes, scales in MN-major [align4(rows), k_groups]).
fn quantize_fp8_per_token(data: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let gran_k = 128usize;
    let k_groups = (cols + gran_k - 1) / gran_k;
    let aligned_rows = (rows + 3) / 4 * 4;
    let mut fp8 = vec![0u8; rows * cols];
    let mut scales = vec![0.0f32; k_groups * aligned_rows];

    for r in 0..rows {
        for kg in 0..k_groups {
            let k_start = kg * gran_k;
            let k_end = (k_start + gran_k).min(cols);
            let mut amax = 0.0f32;
            for c in k_start..k_end {
                amax = amax.max(data[r * cols + c].abs());
            }
            let scale_val = if amax > 1e-12 { amax / 448.0 } else { 1.0 };
            scales[kg * aligned_rows + r] = scale_val;
            let inv = 1.0 / scale_val;
            for c in k_start..k_end {
                let v = (data[r * cols + c] * inv).max(-448.0).min(448.0);
                fp8[r * cols + c] = half::f16::from_f32(v).to_bits().to_be_bytes()[0];
            }
        }
    }
    (fp8, scales)
}

fn deepgemm_fp8(
    input_fp8: &CudaSlice<u8>, weight_fp8: &CudaSlice<u8>,
    output_bf16: &mut CudaSlice<half::bf16>,
    sfa: &CudaSlice<f32>, sfb: &CudaSlice<f32>,
    m: usize, n: usize, k: usize, gpu: &Gpu,
) {
    let (ip, _g1) = input_fp8.device_ptr(&gpu.stream);
    let (wp, _g2) = weight_fp8.device_ptr(&gpu.stream);
    let (op, _g3) = output_bf16.device_ptr_mut(&gpu.stream);
    let (sap, _g4) = sfa.device_ptr(&gpu.stream);
    let (sbp, _g5) = sfb.device_ptr(&gpu.stream);
    unsafe {
        prelude_deepgemm::fp8_gemm(
            ip as *mut c_void, wp as *mut c_void, op as *mut c_void,
            sap as *mut c_void, sbp as *mut c_void,
            m as i32, n as i32, k as i32,
            gpu.stream_ptr(),
        ).ok();
    }
}

/// FP8 E4M3 GEMM via cuBLASLt with per-tensor scaling. Output: BF16.
fn cublaslt_fp8(
    weight: &CudaSlice<u8>, input: &CudaSlice<u8>,
    output_bf16: &mut CudaSlice<half::bf16>,
    scale_a: &CudaSlice<f32>, scale_b: &CudaSlice<f32>,
    m: usize, n: usize, k: usize, gpu: &Gpu,
) -> bool {
    use cublaslt::{result as lt, sys as lt_sys};
    let dt_fp8 = lt_sys::cudaDataType_t::CUDA_R_8F_E4M3;
    let dt_bf16 = lt_sys::cudaDataType_t::CUDA_R_16BF;

    let a_layout = lt::create_matrix_layout(dt_fp8, k as u64, n as u64, k as i64).unwrap();
    let b_layout = lt::create_matrix_layout(dt_fp8, k as u64, m as u64, k as i64).unwrap();
    let c_layout = lt::create_matrix_layout(dt_bf16, n as u64, m as u64, n as i64).unwrap();
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

// ── Grouped GEMM wrappers ──────────────────────────────────────────────

fn deepgemm_grouped(
    input: &CudaSlice<half::bf16>,
    weight: &CudaSlice<half::bf16>,
    output: &mut CudaSlice<half::bf16>,
    layout: &CudaSlice<i32>,
    m: usize, n: usize, k: usize, num_groups: usize,
    gpu: &Gpu,
) {
    let (ip, _g1) = input.device_ptr(&gpu.stream);
    let (wp, _g2) = weight.device_ptr(&gpu.stream);
    let (op, _g3) = output.device_ptr_mut(&gpu.stream);
    let (lp, _g4) = layout.device_ptr(&gpu.stream);
    unsafe {
        prelude_deepgemm::m_grouped_bf16_gemm(
            ip as *mut c_void, wp as *mut c_void, op as *mut c_void,
            lp as *mut c_void,
            m as i32, n as i32, k as i32,
            num_groups as i32,
            gpu.stream_ptr(),
        ).ok();
    }
}

fn bench_fp8(gpu: &Gpu) {
    println!("\n{:=<80}", "= FP8 E4M3 GEMM: DeepGEMM vs cuBLASLt ");
    print!("{:<10} {:<8} {:>9} {:>9} {:>7} {:>7} {:>20}",
        "tokens", "layer", "DeepGEMM", "cuBLASLt", "vs_cub", "TFLOPS", "config");
    println!();
    println!("{}", "-".repeat(80));

    for (name, h, i, _v) in &models() {
        println!("--- {name} (H={h}, I={i}) ---");
        let layers: Vec<(&str, usize, usize)> = vec![
            ("qkvo", *h, *h), ("gate/up", *i, *h), ("down", *h, *i),
        ];

        for (m, tok_label) in &token_counts() {
            let m = *m;
            for (layer, n, k) in &layers {
                // Generate data and quantize
                let a_f32 = rand_f32(m * *k);
                let b_f32 = rand_f32(*n * *k);

                // Per-tensor scaling for cuBLASLt
                let (a_fp8_pt, descale_a) = quantize_fp8_per_tensor(&a_f32);
                let (b_fp8_pt, descale_b) = quantize_fp8_per_tensor(&b_f32);
                let scale_a_cub = gpu.upload(&[descale_a]);
                let scale_b_cub = gpu.upload(&[descale_b]);

                // Per-token scaling for DeepGEMM
                let (a_fp8_tk, sfa) = quantize_fp8_per_token(&a_f32, m, *k);
                let (b_fp8_tk, sfb) = quantize_fp8_per_token(&b_f32, *n, *k);

                let input_dg = gpu.upload(&a_fp8_tk);
                let weight_dg = gpu.upload(&b_fp8_tk);
                let sfa_gpu = gpu.upload(&sfa);
                let sfb_gpu = gpu.upload(&sfb);
                let mut out_bf16_dg = gpu.alloc_zeros::<half::bf16>(m * *n);

                let (bm, bn, stages, _) = prelude_deepgemm::query_fp8_config(m as i32, *n as i32, *k as i32);

                let dg_us = bench_us(|| {
                    deepgemm_fp8(&input_dg, &weight_dg, &mut out_bf16_dg, &sfa_gpu, &sfb_gpu, m, *n, *k, gpu);
                }, gpu);

                // cuBLASLt FP8
                let input_cub = gpu.upload(&a_fp8_pt);
                let weight_cub = gpu.upload(&b_fp8_pt);
                let mut out_bf16 = gpu.alloc_zeros::<half::bf16>(m * *n);
                let cub_ok = cublaslt_fp8(&weight_cub, &input_cub, &mut out_bf16,
                    &scale_a_cub, &scale_b_cub, m, *n, *k, gpu);
                gpu.sync();

                let tf = tflops(m, *n, *k, dg_us);
                let cfg_str = format!("{bm}x{bn}/s{stages}");

                if cub_ok {
                    let cub_us = bench_us(|| {
                        cublaslt_fp8(&weight_cub, &input_cub, &mut out_bf16,
                            &scale_a_cub, &scale_b_cub, m, *n, *k, gpu);
                    }, gpu);
                    let r = dg_us / cub_us;
                    println!("{tok_label:<10} {layer:<8} {dg_us:>9.1} {cub_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T  {cfg_str:>16}",
                        r, ratio_marker(r));
                } else {
                    println!("{tok_label:<10} {layer:<8} {dg_us:>9.1}       N/A         {tf:>6.1}T  {cfg_str:>16}");
                }
            }
        }
        println!();
    }
}

// ============================================================================
// Grouped GEMM: DeepGEMM grouped vs G separate cuBLAS GEMMs
// ============================================================================

fn bench_grouped(gpu: &Gpu) {
    println!("\n{:=<80}", "= Grouped BF16 GEMM: DeepGEMM vs G×cuBLAS ");
    print!("{:<6} {:<6} {:<5} {:<5} {:>9} {:>9} {:>7} {:>7} {:>16}",
        "G", "M/grp", "N", "K", "grouped", "G×cuBLAS", "vs_cub", "TFLOPS", "config");
    println!();
    println!("{}", "-".repeat(80));

    // MoE-typical shapes: (num_groups, m_per_group, N, K)
    let shapes: Vec<(usize, usize, usize, usize)> = vec![
        (4, 128, 4096, 1024),
        (4, 256, 4096, 1024),
        (8, 128, 7168, 4096),
        (8, 256, 7168, 4096),
        (4, 128, 4096, 4096),
        (4, 256, 4096, 4096),
        (8, 128, 4096, 4096),
    ];

    for (num_groups, m_per_group, n, k) in &shapes {
        let num_groups = *num_groups;
        let m_per_group = *m_per_group;
        let n = *n;
        let k = *k;
        let total_m = num_groups * m_per_group;

        // Prepare data
        let a_data = rand_bf16(total_m * k);
        let b_data = rand_bf16(num_groups * n * k);

        // grouped_layout: [total_m] int32
        let mut layout_data = vec![0i32; total_m];
        for g in 0..num_groups {
            for r in 0..m_per_group {
                layout_data[g * m_per_group + r] = g as i32;
            }
        }

        let a_gpu = gpu.upload(&a_data);
        let b_gpu = gpu.upload(&b_data);
        let layout_gpu = gpu.upload(&layout_data);
        let mut out_grouped = gpu.alloc_zeros::<half::bf16>(total_m * n);

        let (bm, bn, stages, _) = prelude_deepgemm::query_grouped_config(
            total_m as i32, n as i32, k as i32,
        );

        // Benchmark grouped GEMM
        let grp_us = bench_us(|| {
            deepgemm_grouped(&a_gpu, &b_gpu, &mut out_grouped, &layout_gpu,
                total_m, n, k, num_groups, gpu);
        }, gpu);

        // Benchmark G separate cuBLAS GEMMs
        let mut outs_separate: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|_| gpu.alloc_zeros::<half::bf16>(m_per_group * n))
            .collect();

        // Upload per-group slices of A and B separately for fair comparison
        let a_groups: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|g| {
                let start = g * m_per_group * k;
                let end = start + m_per_group * k;
                gpu.upload(&a_data[start..end])
            })
            .collect();
        let b_groups: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|g| {
                let start = g * n * k;
                let end = start + n * k;
                gpu.upload(&b_data[start..end])
            })
            .collect();

        let sep_us = bench_us(|| {
            for g in 0..num_groups {
                cublas_gemm_bf16(&b_groups[g], &a_groups[g], &mut outs_separate[g],
                    m_per_group, n, k, gpu);
            }
        }, gpu);

        let tf = tflops(total_m, n, k, grp_us);
        let r = grp_us / sep_us;
        let cfg_str = format!("{bm}x{bn}/s{stages}");
        println!("{num_groups:<6} {m_per_group:<6} {n:<5} {k:<5} {grp_us:>9.1} {sep_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T  {cfg_str:>16}",
            r, ratio_marker(r));
    }
    println!();
}

fn bench_masked(gpu: &Gpu) {
    println!("\n{:=<80}", "= Masked BF16 GEMM: DeepGEMM vs G×cuBLAS ");
    print!("{:<6} {:<6} {:<6} {:<5} {:<5} {:>9} {:>9} {:>7} {:>7}",
        "G", "padM", "actM", "N", "K", "masked", "G×cuBLAS", "vs_cub", "TFLOPS");
    println!();
    println!("{}", "-".repeat(80));

    // (num_groups, padded_m, actual_m, N, K)
    let shapes: Vec<(usize, usize, usize, usize, usize)> = vec![
        (4, 256, 128, 4096, 1024),
        (4, 256, 192, 4096, 1024),
        (8, 128, 128, 4096, 4096),
        (8, 128, 64, 7168, 4096),
        (8, 256, 128, 4096, 4096),
    ];

    for &(num_groups, padded_m, actual_m, n, k) in &shapes {
        let a_data = rand_bf16(num_groups * padded_m * k);
        let b_data = rand_bf16(num_groups * n * k);
        let masked_m: Vec<i32> = vec![actual_m as i32; num_groups];

        let a_gpu = gpu.upload(&a_data);
        let b_gpu = gpu.upload(&b_data);
        let mask_gpu = gpu.upload(&masked_m);
        let mut out_masked = gpu.alloc_zeros::<half::bf16>(num_groups * padded_m * n);

        let expected_m = actual_m;
        let msk_us = bench_us(|| {
            let (ap, _) = a_gpu.device_ptr(&gpu.stream);
            let (bp, _) = b_gpu.device_ptr(&gpu.stream);
            let (mp, _) = mask_gpu.device_ptr(&gpu.stream);
            let (op, _) = out_masked.device_ptr_mut(&gpu.stream);
            unsafe {
                prelude_deepgemm::m_grouped_masked_bf16_gemm(
                    ap as *mut c_void, bp as *mut c_void, op as *mut c_void,
                    mp as *mut c_void,
                    padded_m as i32, n as i32, k as i32,
                    num_groups as i32, expected_m as i32,
                    gpu.stream_ptr(),
                ).unwrap();
            }
        }, gpu);

        // Baseline: G separate cuBLAS GEMMs (each actual_m × N × K)
        let a_groups: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|g| {
                let start = g * padded_m * k;
                let end = start + actual_m * k;
                gpu.upload(&a_data[start..end])
            })
            .collect();
        let b_groups: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|g| {
                let start = g * n * k;
                let end = start + n * k;
                gpu.upload(&b_data[start..end])
            })
            .collect();
        let mut outs_sep: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|_| gpu.alloc_zeros::<half::bf16>(actual_m * n))
            .collect();

        let sep_us = bench_us(|| {
            for g in 0..num_groups {
                cublas_gemm_bf16(&b_groups[g], &a_groups[g], &mut outs_sep[g],
                    actual_m, n, k, gpu);
            }
        }, gpu);

        let total_flops_m = num_groups * actual_m;
        let tf = tflops(total_flops_m, n, k, msk_us);
        let r = msk_us / sep_us;
        println!("{num_groups:<6} {padded_m:<6} {actual_m:<6} {n:<5} {k:<5} {msk_us:>9.1} {sep_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T",
            r, ratio_marker(r));
    }
    println!();
}

fn bench_masked_fp8(gpu: &Gpu) {
    println!("\n{:=<80}", "= Masked FP8 GEMM: DeepGEMM vs G×cuBLAS(BF16) ");
    print!("{:<6} {:<6} {:<6} {:<5} {:<5} {:>9} {:>9} {:>7} {:>7}",
        "G", "padM", "actM", "N", "K", "masked", "G×cuBLAS", "vs_cub", "TFLOPS");
    println!();
    println!("{}", "-".repeat(80));

    // K must be multiple of 128
    let shapes: Vec<(usize, usize, usize, usize, usize)> = vec![
        (4, 256, 128, 1024, 1024),
        (8, 128, 128, 4096, 4096),
        (8, 128, 64, 4096, 4096),
    ];

    for &(num_groups, padded_m, actual_m, n, k) in &shapes {
        let a_f32 = rand_f32(num_groups * padded_m * k);
        let b_f32 = rand_f32(num_groups * n * k);

        let (a_fp8, sfa) = quantize_fp8_per_token(&a_f32, num_groups * padded_m, k);
        let (b_fp8, sfb) = quantize_fp8_per_token(&b_f32, num_groups * n, k);
        let masked_m: Vec<i32> = vec![actual_m as i32; num_groups];

        let a_gpu = gpu.upload(&a_fp8);
        let b_gpu = gpu.upload(&b_fp8);
        let sfa_gpu = gpu.upload(&sfa);
        let sfb_gpu = gpu.upload(&sfb);
        let mask_gpu = gpu.upload(&masked_m);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(num_groups * padded_m * n);

        let expected_m = actual_m;
        let msk_us = bench_us(|| {
            let (ap, _) = a_gpu.device_ptr(&gpu.stream);
            let (bp, _) = b_gpu.device_ptr(&gpu.stream);
            let (sfap, _) = sfa_gpu.device_ptr(&gpu.stream);
            let (sfbp, _) = sfb_gpu.device_ptr(&gpu.stream);
            let (mp, _) = mask_gpu.device_ptr(&gpu.stream);
            let (op, _) = out_gpu.device_ptr_mut(&gpu.stream);
            unsafe {
                prelude_deepgemm::m_grouped_masked_fp8_gemm(
                    ap as *mut c_void, bp as *mut c_void, op as *mut c_void,
                    sfap as *mut c_void, sfbp as *mut c_void,
                    mp as *mut c_void,
                    padded_m as i32, n as i32, k as i32,
                    num_groups as i32, expected_m as i32,
                    gpu.stream_ptr(),
                ).unwrap();
            }
        }, gpu);

        // Baseline: G × cuBLAS BF16 (actual_m per group)
        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let a_groups: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|g| gpu.upload(&a_bf16[g*padded_m*k..g*padded_m*k+actual_m*k]))
            .collect();
        let b_groups: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|g| gpu.upload(&b_bf16[g*n*k..(g+1)*n*k]))
            .collect();
        let mut outs_sep: Vec<CudaSlice<half::bf16>> = (0..num_groups)
            .map(|_| gpu.alloc_zeros::<half::bf16>(actual_m * n))
            .collect();
        let sep_us = bench_us(|| {
            for g in 0..num_groups {
                cublas_gemm_bf16(&b_groups[g], &a_groups[g], &mut outs_sep[g],
                    actual_m, n, k, gpu);
            }
        }, gpu);

        let total_flops_m = num_groups * actual_m;
        let tf = tflops(total_flops_m, n, k, msk_us);
        let r = msk_us / sep_us;
        println!("{num_groups:<6} {padded_m:<6} {actual_m:<6} {n:<5} {k:<5} {msk_us:>9.1} {sep_us:>9.1} {:>5.2}x{:<1} {tf:>6.1}T",
            r, ratio_marker(r));
    }
    println!();
}

fn bench_acc(gpu: &Gpu) {
    println!("\n{:=<80}", "= BF16 GEMM + Acc: DeepGEMM D(FP32)+=A@B ");
    print!("{:<6} {:<5} {:<5} {:>12} {:>7}",
        "M", "N", "K", "acc_us", "TFLOPS");
    println!();
    println!("{}", "-".repeat(50));

    let shapes = vec![
        (1, 1024, 1024), (4, 4096, 4096), (128, 4096, 4096),
    ];

    for &(m, n, k) in &shapes {
        let a_data = rand_bf16(m * k);
        let b_data = rand_bf16(n * k);
        let c_data = rand_f32(m * n);

        let a_gpu = gpu.upload(&a_data);
        let b_gpu = gpu.upload(&b_data);
        let c_gpu = gpu.upload(&c_data);
        let mut d_gpu = gpu.alloc_zeros::<f32>(m * n);

        let acc_us = bench_us(|| {
            let (ap, _) = a_gpu.device_ptr(&gpu.stream);
            let (bp, _) = b_gpu.device_ptr(&gpu.stream);
            let (cp, _) = c_gpu.device_ptr(&gpu.stream);
            let (dp, _) = d_gpu.device_ptr_mut(&gpu.stream);
            unsafe {
                prelude_deepgemm::bf16_gemm_acc(
                    ap as *mut c_void, bp as *mut c_void,
                    cp as *mut c_void, dp as *mut c_void,
                    m as i32, n as i32, k as i32,
                    gpu.stream_ptr(),
                ).unwrap();
            }
        }, gpu);

        let tf = tflops(m, n, k, acc_us);
        println!("{m:<6} {n:<5} {k:<5} {acc_us:>12.1} {tf:>6.1}T");
    }
    println!();
}
