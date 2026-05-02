//! Probe binary: run DeepGEMM SM100 grouped BF16 GEMM on a chosen shape
//! to track down which (BM, BN, stages, multicast) tuple hangs B300.
//!
//! Usage:
//!   cargo run -p deepgemm --example grouped_probe --release -- M_PER_GROUP N K NUM_GROUPS
//!
//! Tiny default exists for `compute-sanitizer` runs:
//!   cargo run -p deepgemm --example grouped_probe --release
//!
//! Wrap with compute-sanitizer to find the actual fault:
//!   /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
//!       target/release/examples/grouped_probe 128 768 2048 8

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m_per_group: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(128);
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let num_groups: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let total_m = m_per_group * num_groups;

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream: Arc<CudaStream> = ctx.new_stream().expect("stream");

    let (driver_v, arch) = deepgemm::query_device();
    let (bm, bn, st, smem) = deepgemm::query_grouped_config(total_m as i32, n as i32, k as i32);
    eprintln!("driver={driver_v} arch=SM{arch}");
    eprintln!("shape: total_M={total_m} M_per_group={m_per_group} N={n} K={k} G={num_groups}");
    eprintln!("config: bm={bm} bn={bn} stages={st} smem_bytes={smem}");

    // Zero-fill buffers — we only care about kernel launch, not output values.
    let a_zero = vec![half::bf16::from_f32(0.0); total_m * k];
    let b_zero = vec![half::bf16::from_f32(0.0); num_groups * n * k];
    let mut layout = vec![0i32; total_m];
    for g in 0..num_groups {
        for r in 0..m_per_group {
            layout[g * m_per_group + r] = g as i32;
        }
    }

    let a_gpu = stream.clone_htod(&a_zero).unwrap();
    let b_gpu = stream.clone_htod(&b_zero).unwrap();
    let lay_gpu = stream.clone_htod(&layout).unwrap();
    let mut d_gpu = stream.alloc_zeros::<half::bf16>(total_m * n).unwrap();

    eprintln!("launching kernel ...");
    let result = {
        let (ap, _g1) = a_gpu.device_ptr(&stream);
        let (bp, _g2) = b_gpu.device_ptr(&stream);
        let (lp, _g3) = lay_gpu.device_ptr(&stream);
        let (dp, _g4) = d_gpu.device_ptr_mut(&stream);
        let stream_ptr = stream.cu_stream() as *mut c_void;
        unsafe {
            deepgemm::m_grouped_bf16_gemm(
                ap as *mut c_void, bp as *mut c_void, dp as *mut c_void,
                lp as *mut c_void,
                total_m as i32, n as i32, k as i32,
                num_groups as i32,
                stream_ptr,
            )
        }
    };
    match result {
        Ok(()) => eprintln!("kernel launch returned Ok"),
        Err(e) => { eprintln!("kernel launch returned Err: {e}"); std::process::exit(1); }
    }

    eprintln!("syncing stream ...");
    stream.synchronize().unwrap();
    eprintln!("sync done — kernel completed without hang");
}
