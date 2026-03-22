//! DeepGEMM correctness test — BF16 GEMM output vs CPU F32 reference.
//! Tolerance: atol=1e-2 + rtol=1e-2 (same as FA4 correctness tests).
use half::bf16;
use std::ffi::c_void;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    fn cudaFree(devPtr: *mut c_void) -> i32;
}

const H2D: i32 = 1;
const D2H: i32 = 2;

/// CPU reference: D[M,N] = A[M,K] @ B[K,N] in F32
/// A row-major [M,K], B col-major [K,N] (= weight [N,K] row-major, transposed)
fn cpu_bf16_gemm(a: &[bf16], b: &[bf16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                // A[i,l] row-major: a[i*k + l]
                // B col-major [K,N] = weight[N,K] transposed: b[j*k + l]
                sum += a[i * k + l].to_f32() * b[j * k + l].to_f32();
            }
            d[i * n + j] = sum;
        }
    }
    d
}

fn test_shape(m: usize, n: usize, k: usize) {
    let (bm, bn, stages, smem) = prelude_deepgemm::query_config(m as i32, n as i32, k as i32);
    print!("  M={m:<5} N={n:<5} K={k:<5} block({bm}x{bn}) stg={stages} ");

    // Generate deterministic BF16 data
    let a_data: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin() * 0.1))
        .collect();
    let b_data: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos() * 0.1))
        .collect();

    // CPU reference
    let ref_d = cpu_bf16_gemm(&a_data, &b_data, m, n, k);

    // GPU
    let mut a_gpu = std::ptr::null_mut();
    let mut b_gpu = std::ptr::null_mut();
    let mut d_gpu = std::ptr::null_mut();
    unsafe {
        assert_eq!(cudaMalloc(&mut a_gpu, m * k * 2), 0);
        assert_eq!(cudaMalloc(&mut b_gpu, n * k * 2), 0);
        assert_eq!(cudaMalloc(&mut d_gpu, m * n * 2), 0);
        assert_eq!(cudaMemcpy(a_gpu, a_data.as_ptr() as _, m * k * 2, H2D), 0);
        assert_eq!(cudaMemcpy(b_gpu, b_data.as_ptr() as _, n * k * 2, H2D), 0);
        cudaDeviceSynchronize();
    }

    let result = unsafe {
        prelude_deepgemm::bf16_gemm(
            a_gpu, b_gpu, d_gpu,
            m as i32, n as i32, k as i32,
            std::ptr::null_mut(),
        )
    };

    let sync = unsafe { cudaDeviceSynchronize() };
    let last = unsafe { cudaGetLastError() };

    if result.is_err() || sync != 0 || last != 0 {
        let msg = result.err().unwrap_or_else(|| {
            let e = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(last)) };
            format!("CUDA: {}", e.to_string_lossy())
        });
        println!("FAIL: {msg}");
        unsafe { cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(d_gpu); }
        return;
    }

    // Download result
    let mut d_bf16 = vec![bf16::ZERO; m * n];
    unsafe {
        cudaMemcpy(d_bf16.as_mut_ptr() as _, d_gpu, m * n * 2, D2H);
        cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(d_gpu);
    }

    // Compare using atol + rtol * |expected| (same as FA4 correctness tests)
    let atol = 1e-2f32;
    let rtol = 1e-2f32;
    let mut max_diff = 0.0f32;
    let mut mismatches = 0usize;
    for i in 0..m * n {
        let got = d_bf16[i].to_f32();
        let exp = ref_d[i];
        let diff = (got - exp).abs();
        let tol = atol + rtol * exp.abs();
        if diff > tol { mismatches += 1; }
        max_diff = max_diff.max(diff);
    }

    let pass = mismatches == 0;
    println!(
        "max_diff={max_diff:.6} mismatch={mismatches}/{} {}",
        m * n,
        if pass { "OK" } else { "FAIL" }
    );
}

fn main() {
    println!("DeepGEMM BF16 GEMM correctness test\n");

    // Small shapes (fast CPU reference)
    test_shape(1, 256, 256);
    test_shape(4, 256, 256);
    test_shape(32, 256, 256);
    test_shape(128, 256, 256);

    // Medium shapes
    test_shape(1, 4096, 4096);
    test_shape(4, 4096, 4096);
    test_shape(32, 4096, 4096);
    test_shape(128, 4096, 4096);

    // Large prefill (skip CPU reference for very large — too slow)
    test_shape(256, 4096, 4096);
    test_shape(512, 4096, 4096);

    println!("\nDone.");
}
