//! Micro-benchmark for GPU GEMM: candle (cuBLAS) vs DeepGEMM.
//!
//! Usage:
//!   CUDA_VISIBLE_DEVICES=1 cargo run -p prelude-core --bin gpu_ops_bench --release --features deepgemm

mod gemm;

fn main() -> candle_core::Result<()> {
    println!("GPU Ops Benchmark\n");

    gemm::bench()?;

    Ok(())
}
