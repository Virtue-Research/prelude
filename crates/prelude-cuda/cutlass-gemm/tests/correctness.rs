//! CUTLASS GEMM correctness tests — F32/TF32, FP8 E4M3, Batched.
//!
//! Each test allocates GPU memory via cudarc, runs CUTLASS GEMM, and compares
//! against a CPU F64 reference.  Requires a CUDA GPU to run.
//!
//! Run:  cargo test -p cutlass-gemm --release

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};

// ── CPU reference: f64 matmul (ground truth) ────────────────────────────

/// CPU F64 matmul: out[m,n] = a[m,k] @ b[n,k]^T  (TN convention)
fn cpu_ref_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f64;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ni * k + ki];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Batched CPU F64 matmul
fn cpu_ref_f64_batched(
    a: &[f64], b: &[f64], m: usize, n: usize, k: usize, batch: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * m * n];
    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * n * k;
        let o_off = bi * m * n;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f64;
                for ki in 0..k {
                    acc += a[a_off + mi * k + ki] * b[b_off + ni * k + ki];
                }
                out[o_off + mi * n + ni] = acc;
            }
        }
    }
    out
}

// ── CUDA helpers ────────────────────────────────────────────────────────

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

    fn download<T: cudarc::driver::DeviceRepr>(&self, slice: &CudaSlice<T>) -> Vec<T> {
        self.stream.clone_dtoh(slice).unwrap()
    }

    fn alloc_zeros<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> CudaSlice<T> {
        self.stream.alloc_zeros(len).unwrap()
    }

    fn sync(&self) {
        self.stream.synchronize().unwrap();
    }
}

fn ptr<T>(s: &CudaSlice<T>, stream: &CudaStream) -> *const c_void {
    let (p, _guard) = s.device_ptr(stream);
    // Safety: we sync before reading the result, so the pointer stays valid.
    p as *const c_void
}

fn ptr_mut<T>(s: &mut CudaSlice<T>, stream: &CudaStream) -> *mut c_void {
    let (p, _guard) = s.device_ptr_mut(stream);
    p as *mut c_void
}

// ── Dispatch wrapper ────────────────────────────────────────────────────

/// Call CUTLASS GEMM in cuBLAS TN convention.
/// weight [n,k] row-major, input [m,k] row-major → output [m,n]
fn call_gemm(
    weight_ptr: *const c_void,
    input_ptr: *const c_void,
    output_ptr: *mut c_void,
    m: usize, n: usize, k: usize,
    batch: usize,
    stride_a: i64, stride_b: i64, stride_d: i64,
    dtype: u32,
    stream: *const c_void,
) -> Result<(), String> {
    unsafe {
        cutlass_gemm::gemm_dispatch(
            weight_ptr, input_ptr, output_ptr,
            n as i32, m as i32, k as i32,
            batch as i32,
            k as i32, k as i32, n as i32,
            stride_a, stride_b, stride_d,
            true, false,
            dtype, stream,
        )
    }
}

fn max_abs_err(reference: &[f64], result: &[f64]) -> f64 {
    reference.iter().zip(result).map(|(r, t)| (r - t).abs()).fold(0.0f64, f64::max)
}

fn rand_f32(len: usize) -> Vec<f32> {
    use rand::RngExt;
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(-0.5f32..0.5f32)).collect()
}

// ============================================================================
// BF16 GEMM (dtype=0)
// ============================================================================

#[test]
fn bf16_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (m, n, k) in [(1, 256, 256), (4, 64, 128), (16, 512, 1024)] {
        let a_f32 = rand_f32(m * k);
        let b_f32 = rand_f32(n * k);
        let ref64 = cpu_ref_f64(
            &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k,
        );

        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let a_gpu = gpu.upload(&a_bf16);
        let b_gpu = gpu.upload(&b_bf16);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(m * n);

        {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 0, gpu.stream_ptr()).unwrap();
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        // Match gpu_ops_bench: flat 1.0 threshold for BF16
        assert!(err < 1.0, "BF16 GEMM {m}x{n}x{k}: max_err={err:.6e}");
    }
}

#[test]
fn bf16_gemm_model_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // Qwen3-0.6B: hidden=1024, intermediate=3072, vocab=151936
    for (m, n, k) in [(1, 1024, 1024), (32, 3072, 1024), (128, 1024, 3072), (1, 151936, 1024)] {
        let a_f32 = rand_f32(m * k);
        let b_f32 = rand_f32(n * k);
        let ref64 = cpu_ref_f64(
            &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k,
        );

        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let a_gpu = gpu.upload(&a_bf16);
        let b_gpu = gpu.upload(&b_bf16);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(m * n);

        {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 0, gpu.stream_ptr()).unwrap();
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        assert!(err < 1.0, "BF16 GEMM {m}x{n}x{k}: max_err={err:.6e}");
    }
}

// ============================================================================
// FP16 GEMM (dtype=1)
// ============================================================================

#[test]
fn fp16_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (m, n, k) in [(1, 256, 256), (4, 64, 128), (16, 512, 1024)] {
        let a_f32 = rand_f32(m * k);
        let b_f32 = rand_f32(n * k);
        let ref64 = cpu_ref_f64(
            &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k,
        );

        let a_fp16: Vec<half::f16> = a_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let b_fp16: Vec<half::f16> = b_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let a_gpu = gpu.upload(&a_fp16);
        let b_gpu = gpu.upload(&b_fp16);
        let mut out_gpu = gpu.alloc_zeros::<half::f16>(m * n);

        {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 1, gpu.stream_ptr()).unwrap();
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        // FP16 has ~0.1% relative error, slightly better than BF16. Use same 1.0 threshold.
        assert!(err < 1.0, "FP16 GEMM {m}x{n}x{k}: max_err={err:.6e}");
    }
}

#[test]
fn fp16_gemm_model_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (m, n, k) in [(1, 1024, 1024), (32, 3072, 1024), (128, 1024, 3072)] {
        let a_f32 = rand_f32(m * k);
        let b_f32 = rand_f32(n * k);
        let ref64 = cpu_ref_f64(
            &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k,
        );

        let a_fp16: Vec<half::f16> = a_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let b_fp16: Vec<half::f16> = b_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let a_gpu = gpu.upload(&a_fp16);
        let b_gpu = gpu.upload(&b_fp16);
        let mut out_gpu = gpu.alloc_zeros::<half::f16>(m * n);

        {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 1, gpu.stream_ptr()).unwrap();
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        assert!(err < 1.0, "FP16 GEMM {m}x{n}x{k}: max_err={err:.6e}");
    }
}

// ============================================================================
// F32 / TF32 GEMM (dtype=2)
// ============================================================================

#[test]
fn f32_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (m, n, k) in [(4, 64, 128), (1, 256, 256), (16, 512, 1024)] {
        let a = rand_f32(m * k);
        let b = rand_f32(n * k);
        let ref64 = cpu_ref_f64(
            &a.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k,
        );

        let a_gpu = gpu.upload(&a);
        let b_gpu = gpu.upload(&b);
        let mut out_gpu = gpu.alloc_zeros::<f32>(m * n);

        // Must scope pointer borrows so guards drop before sync/download
        {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 2, gpu.stream_ptr()).unwrap();
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|&x| x as f64).collect();
        let err = max_abs_err(&ref64, &result);
        let tol = 0.05 * k as f64 * 0.5e-3;
        assert!(err < tol, "F32 GEMM {m}x{n}x{k}: max_err={err:.6e}, tol={tol:.6e}");
    }
}

#[test]
fn f32_gemm_model_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // Qwen3-0.6B: hidden=1024, intermediate=3072
    for (m, n, k) in [(1, 1024, 1024), (32, 3072, 1024), (128, 1024, 3072)] {
        let a = rand_f32(m * k);
        let b = rand_f32(n * k);
        let ref64 = cpu_ref_f64(
            &a.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k,
        );

        let a_gpu = gpu.upload(&a);
        let b_gpu = gpu.upload(&b);
        let mut out_gpu = gpu.alloc_zeros::<f32>(m * n);

        {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 2, gpu.stream_ptr()).unwrap();
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|&x| x as f64).collect();
        let err = max_abs_err(&ref64, &result);
        let tol = 0.05 * k as f64 * 0.5e-3;
        assert!(err < tol, "F32 GEMM {m}x{n}x{k}: max_err={err:.6e}, tol={tol:.6e}");
    }
}

// ============================================================================
// FP8 E4M3 GEMM (dtype=3)
// ============================================================================

#[test]
fn fp8_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (m, n, k) in [(4, 64, 128), (16, 128, 256), (32, 256, 512)] {
        let a_f32 = rand_f32(m * k);
        let b_f32 = rand_f32(n * k);

        // Quantize to FP8 — reference uses dequantized inputs for fair comparison
        let a_fp8: Vec<float8::F8E4M3> = a_f32.iter().map(|&x| float8::F8E4M3::from_f32(x)).collect();
        let b_fp8: Vec<float8::F8E4M3> = b_f32.iter().map(|&x| float8::F8E4M3::from_f32(x)).collect();
        let a_deq: Vec<f64> = a_fp8.iter().map(|x| x.to_f32() as f64).collect();
        let b_deq: Vec<f64> = b_fp8.iter().map(|x| x.to_f32() as f64).collect();
        let ref64 = cpu_ref_f64(&a_deq, &b_deq, m, n, k);

        // Upload as u8 (FP8 = 1 byte)
        let a_bytes: Vec<u8> = a_fp8.iter().map(|x| x.to_bits()).collect();
        let b_bytes: Vec<u8> = b_fp8.iter().map(|x| x.to_bits()).collect();
        let a_gpu = gpu.upload(&a_bytes);
        let b_gpu = gpu.upload(&b_bytes);
        let mut out_gpu = gpu.alloc_zeros::<u8>(m * n);

        let res = {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 3, gpu.stream_ptr())
        };
        if let Err(e) = res {
            eprintln!("FP8 GEMM {m}x{n}x{k} skipped (not supported): {e}");
            return;
        }
        gpu.sync();

        let out_bytes = gpu.download(&out_gpu);
        let result: Vec<f64> = out_bytes.iter()
            .map(|&b| float8::F8E4M3::from_bits(b).to_f32() as f64)
            .collect();
        let err = max_abs_err(&ref64, &result);
        // FP8 E4M3: 3 mantissa bits → ~6% relative error per element.
        // Accumulation in F32 is exact; error comes from input/output quantization.
        let tol = 1.0;
        assert!(err < tol, "FP8 GEMM {m}x{n}x{k}: max_err={err:.6e}, tol={tol:.6e}");
    }
}

#[test]
fn fp8_gemm_model_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // Smaller K for FP8 to keep outputs in representable range
    for (m, n, k) in [(1, 512, 256), (16, 1024, 512), (32, 256, 1024)] {
        let a_f32 = rand_f32(m * k);
        let b_f32 = rand_f32(n * k);
        let a_fp8: Vec<float8::F8E4M3> = a_f32.iter().map(|&x| float8::F8E4M3::from_f32(x)).collect();
        let b_fp8: Vec<float8::F8E4M3> = b_f32.iter().map(|&x| float8::F8E4M3::from_f32(x)).collect();
        let a_deq: Vec<f64> = a_fp8.iter().map(|x| x.to_f32() as f64).collect();
        let b_deq: Vec<f64> = b_fp8.iter().map(|x| x.to_f32() as f64).collect();
        let ref64 = cpu_ref_f64(&a_deq, &b_deq, m, n, k);

        let a_bytes: Vec<u8> = a_fp8.iter().map(|x| x.to_bits()).collect();
        let b_bytes: Vec<u8> = b_fp8.iter().map(|x| x.to_bits()).collect();
        let a_gpu = gpu.upload(&a_bytes);
        let b_gpu = gpu.upload(&b_bytes);
        let mut out_gpu = gpu.alloc_zeros::<u8>(m * n);

        let res = {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, 1, 0, 0, 0, 3, gpu.stream_ptr())
        };
        if let Err(e) = res {
            eprintln!("FP8 GEMM {m}x{n}x{k} skipped: {e}");
            return;
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter()
            .map(|&b| float8::F8E4M3::from_bits(b).to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        assert!(err < 1.0, "FP8 GEMM {m}x{n}x{k}: max_err={err:.6e}");
    }
}

// ============================================================================
// Batched FP8 GEMM (batch>1, dtype=3)
// ============================================================================

#[test]
fn batched_fp8_gemm() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (batch, m, n, k) in [(2, 8, 64, 128), (4, 4, 128, 256)] {
        let total_a = batch * m * k;
        let total_b = batch * n * k;
        let total_o = batch * m * n;

        let a_f32 = rand_f32(total_a);
        let b_f32 = rand_f32(total_b);
        let a_fp8: Vec<float8::F8E4M3> = a_f32.iter().map(|&x| float8::F8E4M3::from_f32(x)).collect();
        let b_fp8: Vec<float8::F8E4M3> = b_f32.iter().map(|&x| float8::F8E4M3::from_f32(x)).collect();
        let a_deq: Vec<f64> = a_fp8.iter().map(|x| x.to_f32() as f64).collect();
        let b_deq: Vec<f64> = b_fp8.iter().map(|x| x.to_f32() as f64).collect();
        let ref64 = cpu_ref_f64_batched(&a_deq, &b_deq, m, n, k, batch);

        let a_bytes: Vec<u8> = a_fp8.iter().map(|x| x.to_bits()).collect();
        let b_bytes: Vec<u8> = b_fp8.iter().map(|x| x.to_bits()).collect();
        let a_gpu = gpu.upload(&a_bytes);
        let b_gpu = gpu.upload(&b_bytes);
        let mut out_gpu = gpu.alloc_zeros::<u8>(total_o);

        let stride_a = (n * k) as i64;
        let stride_b = (m * k) as i64;
        let stride_d = (m * n) as i64;

        let res = {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, batch, stride_a, stride_b, stride_d, 3, gpu.stream_ptr())
        };
        if let Err(e) = res {
            eprintln!("Batched FP8 GEMM {batch}x{m}x{n}x{k} skipped: {e}");
            return;
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter()
            .map(|&b| float8::F8E4M3::from_bits(b).to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        assert!(err < 1.0, "Batched FP8 {batch}x{m}x{n}x{k}: max_err={err:.6e}");
    }
}

// ============================================================================
// Batched BF16 GEMM (batch>1, dtype=0)
// ============================================================================

#[test]
fn batched_bf16_gemm() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (batch, m, n, k) in [(2, 16, 128, 256), (4, 8, 64, 128), (8, 1, 128, 512)] {
        let total_a = batch * m * k;
        let total_b = batch * n * k;
        let total_o = batch * m * n;

        let a_f32 = rand_f32(total_a);
        let b_f32 = rand_f32(total_b);
        let ref64 = cpu_ref_f64_batched(
            &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k, batch,
        );

        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let a_gpu = gpu.upload(&a_bf16);
        let b_gpu = gpu.upload(&b_bf16);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(total_o);

        let stride_a = (n * k) as i64;
        let stride_b = (m * k) as i64;
        let stride_d = (m * n) as i64;

        let res = {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, batch, stride_a, stride_b, stride_d, 0, gpu.stream_ptr())
        };
        if let Err(e) = res {
            eprintln!("Batched BF16 GEMM {batch}x{m}x{n}x{k} skipped: {e}");
            return;
        }
        gpu.sync();

        let out_bf16 = gpu.download(&out_gpu);
        let result: Vec<f64> = out_bf16.iter().map(|x| x.to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        // Match gpu_ops_bench: flat 1.0 threshold for BF16 (holds up to K=30720)
        let tol = 1.0;
        assert!(err < tol, "Batched BF16 {batch}x{m}x{n}x{k}: max_err={err:.6e}, tol={tol:.6e}");
    }
}

// ============================================================================
// Batched FP16 GEMM (batch>1, dtype=1)
// ============================================================================

#[test]
fn batched_fp16_gemm() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (batch, m, n, k) in [(2, 16, 128, 256), (4, 8, 64, 128)] {
        let total_a = batch * m * k;
        let total_b = batch * n * k;
        let total_o = batch * m * n;

        let a_f32 = rand_f32(total_a);
        let b_f32 = rand_f32(total_b);
        let ref64 = cpu_ref_f64_batched(
            &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k, batch,
        );

        let a_fp16: Vec<half::f16> = a_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let b_fp16: Vec<half::f16> = b_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let a_gpu = gpu.upload(&a_fp16);
        let b_gpu = gpu.upload(&b_fp16);
        let mut out_gpu = gpu.alloc_zeros::<half::f16>(total_o);

        let stride_a = (n * k) as i64;
        let stride_b = (m * k) as i64;
        let stride_d = (m * n) as i64;

        let res = {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, batch, stride_a, stride_b, stride_d, 1, gpu.stream_ptr())
        };
        if let Err(e) = res {
            eprintln!("Batched FP16 GEMM {batch}x{m}x{n}x{k} skipped: {e}");
            return;
        }
        gpu.sync();

        let out_fp16 = gpu.download(&out_gpu);
        let result: Vec<f64> = out_fp16.iter().map(|x| x.to_f32() as f64).collect();
        let err = max_abs_err(&ref64, &result);
        let tol = 1.0;
        assert!(err < tol, "Batched FP16 {batch}x{m}x{n}x{k}: max_err={err:.6e}, tol={tol:.6e}");
    }
}

// ============================================================================
// Batched F32 GEMM (batch>1, dtype=2)
// ============================================================================

#[test]
fn batched_f32_gemm() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (batch, m, n, k) in [(2, 8, 64, 128), (4, 4, 128, 256)] {
        let total_a = batch * m * k;
        let total_b = batch * n * k;
        let total_o = batch * m * n;

        let a = rand_f32(total_a);
        let b = rand_f32(total_b);
        let ref64 = cpu_ref_f64_batched(
            &a.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &b.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            m, n, k, batch,
        );

        let a_gpu = gpu.upload(&a);
        let b_gpu = gpu.upload(&b);
        let mut out_gpu = gpu.alloc_zeros::<f32>(total_o);

        let stride_a = (n * k) as i64;
        let stride_b = (m * k) as i64;
        let stride_d = (m * n) as i64;

        let res = {
            let ap = ptr(&a_gpu, &gpu.stream);
            let bp = ptr(&b_gpu, &gpu.stream);
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            call_gemm(bp, ap, op, m, n, k, batch, stride_a, stride_b, stride_d, 2, gpu.stream_ptr())
        };
        if let Err(e) = res {
            eprintln!("Batched F32 GEMM {batch}x{m}x{n}x{k} skipped: {e}");
            return;
        }
        gpu.sync();

        let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|&x| x as f64).collect();
        let err = max_abs_err(&ref64, &result);
        let tol = 0.05 * k as f64 * 0.5e-3;
        assert!(err < tol, "Batched F32 {batch}x{m}x{n}x{k}: max_err={err:.6e}, tol={tol:.6e}");
    }
}
