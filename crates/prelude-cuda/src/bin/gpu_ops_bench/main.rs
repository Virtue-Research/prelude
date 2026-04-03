// GPU kernel micro-benchmark suite.
//
// Compares our dispatch path vs cuBLAS/reference for GEMM, attention, etc.
//
// Usage:
//   # All benchmarks, all models:
//   CUDA_VISIBLE_DEVICES=1 cargo run -p prelude-core --bin gpu_ops_bench --release \
//       --features flashinfer-v4,cutlass-gemm,deepgemm,bench-cublas,onednn
//
//   # Filter by model (substring match):
//   ... -- 8B
//
//   # Run specific benchmark only:
//   ... -- --gemm
//   ... -- --gemm 8B
//   ... -- --quant

#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

mod common;
mod gemm;

use candle_core::{Device, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Parse: --bench flags and model filters
    // Quant benchmarks moved to: cargo run -p prelude-quant-gemm --bin bench_kernel --release
    let mut run_gemm = false;
    let mut run_verify = false;
    let mut filter = Vec::new();

    for arg in &args {
        match arg.as_str() {
            "--gemm" => run_gemm = true,
            "--verify" => run_verify = true,
            "--" => {},  // cargo passes this as separator
            s if s.starts_with("--") => eprintln!("Unknown flag: {s}"),
            _ => filter.push(arg.clone()),
        }
    }

    // Default: run all benchmarks if none specified
    let run_all = !run_gemm && !run_verify;
    if run_all { run_gemm = true; }

    println!("GPU Kernel Micro-Benchmark{}\n",
        if filter.is_empty() { String::new() }
        else { format!(" (filter: {})", filter.join(", ")) });

    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        println!("No CUDA device, skipping");
        return Ok(());
    }

    // Initialize CUDA ops (registers GEMM dispatch)
    #[cfg(any(feature = "cutlass-gemm", feature = "deepgemm"))]
    let _ops = prelude_cuda::create_cuda_ops();

    // Shared cuBLAS handle
    #[cfg(feature = "bench-cublas")]
    let cublas = Some(common::CublasHandle::new(&device)?);
    #[cfg(not(feature = "bench-cublas"))]
    let cublas: Option<common::CublasHandle> = None;

    let has_deepgemm = cfg!(feature = "deepgemm");
    let has_cutlass = cfg!(feature = "cutlass-gemm");
    let has_cublas = cublas.is_some();

    println!("Backends:");
    println!("  dispatch = Tensor::matmul → {}{}",
        if has_deepgemm { "DeepGEMM → " } else { "" },
        if has_cutlass { "CUTLASS" } else { "none" });
    if has_cutlass { println!("  cutlass  = CUTLASS direct (bypassing DeepGEMM)"); }
    if has_cublas { println!("  cublas   = cuBLAS (main-branch candle baseline)"); }
    println!("  warmup={} repeats={}\n", common::WARMUP, common::REPEATS);

    if run_verify {
        println!("╔══════════════════════════════════════╗");
        println!("║       GEMM Correctness Verify        ║");
        println!("╚══════════════════════════════════════╝\n");
        gemm::verify(&filter, &device, cublas.as_ref())?;
    }

    if run_gemm {
        println!("╔══════════════════════════════════════╗");
        println!("║            GEMM Benchmark            ║");
        println!("╚══════════════════════════════════════╝\n");
        gemm::bench(&filter, &device, cublas.as_ref())?;

        println!("d/cub = dispatch/cuBLAS  c/cub = cutlass/cuBLAS  (1.0=parity <1=faster)");
        println!("✓ ≤1.05x  ~ ≤1.3x  ✗ >2x  ✗✗ >5x");
    }

    // Quant benchmarks (dequantize, MMVQ, tiled MMQ) moved to prelude-quant-gemm:
    //   cargo test -p prelude-quant-gemm --release
    //   cargo run -p prelude-quant-gemm --bin bench_kernel --release

    Ok(())
}
