//! DeepGEMM BF16/FP8 GEMM correctness tests — GPU output vs CPU F64 reference.
//!
//! Run:  cargo test -p prelude-deepgemm --release

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};

// ── CPU reference: f64 matmul (ground truth) ────────────────────────────

/// CPU F64 matmul: out[m,n] = a[m,k] @ b[n,k]^T
/// A row-major [M,K], B row-major [N,K] (weight), used transposed.
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

    fn stream_ptr(&self) -> *mut c_void {
        self.stream.cu_stream() as *mut c_void
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

/// Call DeepGEMM bf16_gemm.
/// weight [N,K] row-major (used as col-major B), input [M,K] row-major → output [M,N]
fn call_deepgemm(
    input_ptr: *mut c_void,
    weight_ptr: *mut c_void,
    output_ptr: *mut c_void,
    m: usize, n: usize, k: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    unsafe {
        prelude_deepgemm::bf16_gemm(
            input_ptr, weight_ptr, output_ptr,
            m as i32, n as i32, k as i32,
            stream,
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

fn run_test(m: usize, n: usize, k: usize, gpu: &Gpu) {
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
        let (ap, _g1) = a_gpu.device_ptr(&gpu.stream);
        let (bp, _g2) = b_gpu.device_ptr(&gpu.stream);
        let (op, _g3) = out_gpu.device_ptr_mut(&gpu.stream);
        call_deepgemm(
            ap as *mut c_void, bp as *mut c_void, op as *mut c_void,
            m, n, k, gpu.stream_ptr(),
        ).unwrap();
    }
    gpu.sync();

    let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let err = max_abs_err(&ref64, &result);
    assert!(err < 1.0, "DeepGEMM BF16 {m}x{n}x{k}: max_err={err:.6e}");
}

// ============================================================================
// BF16 GEMM — small shapes
// ============================================================================

#[test]
fn bf16_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (m, n, k) in [(1, 256, 256), (4, 64, 128), (16, 512, 1024)] {
        run_test(m, n, k, &gpu);
    }
}

// ============================================================================
// BF16 GEMM — model-realistic shapes (Qwen3-0.6B)
// ============================================================================

#[test]
fn bf16_gemm_model_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // Qwen3-0.6B: hidden=1024, intermediate=3072, vocab=151936
    for (m, n, k) in [(1, 1024, 1024), (32, 3072, 1024), (128, 1024, 3072), (1, 151936, 1024)] {
        run_test(m, n, k, &gpu);
    }
}

// ============================================================================
// BF16 GEMM — decode shapes (M=1..32 where DeepGEMM excels)
// ============================================================================

#[test]
fn bf16_gemm_decode_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for m in [1, 4, 8, 16, 32] {
        run_test(m, 4096, 4096, &gpu);
    }
}

// ============================================================================
// BF16 GEMM — prefill shapes (large M)
// ============================================================================

#[test]
fn bf16_gemm_prefill_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for m in [64, 128, 256, 512] {
        run_test(m, 4096, 4096, &gpu);
    }
}

// ============================================================================
// BF16 GEMM — query_config sanity
// ============================================================================

#[test]
fn query_config_returns_valid() {
    let shapes = [(1, 4096, 4096), (32, 4096, 4096), (512, 4096, 4096)];
    for (m, n, k) in shapes {
        let (bm, bn, stages, smem) = prelude_deepgemm::query_config(m, n, k);
        assert!(bm > 0, "block_m should be > 0 for M={m}");
        assert!(bn > 0, "block_n should be > 0 for M={m}");
        assert!(stages > 0, "stages should be > 0 for M={m}");
        assert!(smem > 0, "smem should be > 0 for M={m}");
    }
}

// ============================================================================
// FP8 E4M3 GEMM — per-token 1D scaling, FP32 output
// ============================================================================

/// Quantize f32 to FP8 E4M3 with per-token scaling (granularity=128).
/// Returns (fp8_bytes, scale_factors in MN-major layout).
/// scale_factors: [ceil(K/128), M] with M contiguous.
fn quantize_to_fp8(data: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let gran_k = 128usize;
    let k_groups = (cols + gran_k - 1) / gran_k;
    // TMA requires inner dim (rows) aligned to 4 for FP32
    let aligned_rows = (rows + 3) / 4 * 4;
    let mut fp8_data = vec![0u8; rows * cols];
    // MN-major: scale[kg * aligned_rows + row], padded to aligned_rows
    let mut scales = vec![0.0f32; k_groups * aligned_rows];

    for r in 0..rows {
        for kg in 0..k_groups {
            let k_start = kg * gran_k;
            let k_end = (k_start + gran_k).min(cols);
            // Compute amax for this group
            let mut amax = 0.0f32;
            for c in k_start..k_end {
                amax = amax.max(data[r * cols + c].abs());
            }
            let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            scales[kg * aligned_rows + r] = scale;
            // Quantize
            for c in k_start..k_end {
                let val = data[r * cols + c] / scale;
                let clamped = val.max(-448.0).min(448.0);
                // Simple FP8 E4M3 conversion via f32→f8 rounding
                let fp8 = half::f16::from_f32(clamped); // approximate: f16 as proxy
                // Use raw byte truncation for FP8 E4M3
                let bits = fp8.to_bits();
                fp8_data[r * cols + c] = (bits >> 8) as u8; // top byte approximation
            }
        }
    }
    (fp8_data, scales)
}

/// Dequantize FP8 data back to f64 for reference computation.
fn dequantize_fp8_f64(fp8_bytes: &[u8], scales: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let gran_k = 128usize;
    let k_groups = (cols + gran_k - 1) / gran_k;
    let aligned_rows = (rows + 3) / 4 * 4;
    let mut result = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for kg in 0..k_groups {
            let scale = scales[kg * aligned_rows + r] as f64;
            let k_start = kg * gran_k;
            let k_end = (k_start + gran_k).min(cols);
            for c in k_start..k_end {
                // Reconstruct: reinterpret byte as FP8 E4M3, convert to f32, multiply by scale
                let fp8_val = fp8_e4m3_to_f32(fp8_bytes[r * cols + c]);
                result[r * cols + c] = fp8_val as f64 * scale;
            }
        }
    }
    result
}

/// Convert FP8 E4M3 byte to f32.
fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    // E4M3: sign(1) + exponent(4) + mantissa(3), bias=7, no inf, max=448
    let sign = ((bits >> 7) & 1) as f32;
    let exp = ((bits >> 3) & 0xF) as i32;
    let mant = (bits & 0x7) as f32;

    if exp == 0 {
        // Subnormal: (-1)^s * 2^(-6) * (0.mantissa)
        let val = (mant / 8.0) * (1.0 / 64.0); // 2^(-6) = 1/64
        if sign > 0.0 { -val } else { val }
    } else if exp == 15 && mant == 7.0 {
        // NaN
        f32::NAN
    } else {
        // Normal: (-1)^s * 2^(exp-7) * (1 + mantissa/8)
        let val = (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7);
        if sign > 0.0 { -val } else { val }
    }
}

fn run_fp8_test(m: usize, n: usize, k: usize, gpu: &Gpu) {
    // K must be multiple of 128 for FP8 scaling
    assert!(k % 128 == 0, "K must be multiple of 128 for FP8");

    let a_f32 = rand_f32(m * k);
    let b_f32 = rand_f32(n * k);

    // Quantize to FP8 with per-token scaling
    let (a_fp8, scale_a) = quantize_to_fp8(&a_f32, m, k);
    let (b_fp8, scale_b) = quantize_to_fp8(&b_f32, n, k);

    // CPU reference using dequantized values
    let a_deq = dequantize_fp8_f64(&a_fp8, &scale_a, m, k);
    let b_deq = dequantize_fp8_f64(&b_fp8, &scale_b, n, k);
    let ref64 = cpu_ref_f64(&a_deq, &b_deq, m, n, k);

    // Upload to GPU, run kernel, download result — all in a scope so GPU memory is freed
    let result: Vec<f64> = {
        let a_gpu = gpu.upload(&a_fp8);
        let b_gpu = gpu.upload(&b_fp8);
        let sfa_gpu = gpu.upload(&scale_a);
        let sfb_gpu = gpu.upload(&scale_b);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(m * n);

        {
            let (ap, _g1) = a_gpu.device_ptr(&gpu.stream);
            let (bp, _g2) = b_gpu.device_ptr(&gpu.stream);
            let (sfap, _g3) = sfa_gpu.device_ptr(&gpu.stream);
            let (sfbp, _g4) = sfb_gpu.device_ptr(&gpu.stream);
            let (op, _g5) = out_gpu.device_ptr_mut(&gpu.stream);
            unsafe {
                prelude_deepgemm::fp8_gemm(
                    ap as *mut c_void, bp as *mut c_void, op as *mut c_void,
                    sfap as *mut c_void, sfbp as *mut c_void,
                    m as i32, n as i32, k as i32,
                    gpu.stream_ptr(),
                ).unwrap();
            }
        }
        gpu.sync();
        gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect()
    }; // GPU slices dropped here

    let err = max_abs_err(&ref64, &result);
    let tol = 1.0;
    assert!(err < tol, "DeepGEMM FP8 {m}x{n}x{k}: max_err={err:.6e}");
}

#[test]
fn fp8_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // K must be multiple of 128
    for (m, n, k) in [(4, 128, 128), (16, 256, 256), (32, 512, 256)] {
        run_fp8_test(m, n, k, &gpu);
    }
}

#[test]
fn fp8_gemm_model_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // Qwen3-0.6B/8B: K must be multiple of 128 for FP8
    for (m, n, k) in [
        (1, 1024, 1024), (32, 3072, 1024), (128, 1024, 3072),
        (512, 4096, 4096), (512, 11008, 4096),
    ] {
        run_fp8_test(m, n, k, &gpu);
    }
}


#[test]
fn query_fp8_config_returns_valid() {
    let shapes = [(1, 4096, 4096), (32, 4096, 4096), (512, 4096, 4096)];
    for (m, n, k) in shapes {
        let (bm, bn, stages, smem) = prelude_deepgemm::query_fp8_config(m, n, k);
        assert!(bm > 0, "FP8 block_m should be > 0 for M={m}");
        assert!(bn > 0, "block_n should be > 0 for M={m}");
        assert!(stages > 0, "stages should be > 0 for M={m}");
        assert!(smem > 0, "smem should be > 0 for M={m}");
    }
}

// ============================================================================
// M-Grouped Contiguous BF16 GEMM (MoE)
// ============================================================================

/// CPU reference for grouped GEMM: per-group matmul.
/// A [total_M, K], B [G, N, K] row-major, D [total_M, N].
/// ms[g] = number of rows in group g, each aligned to 128.
fn cpu_grouped_ref_f64(
    a: &[f64], b: &[f64], ms: &[usize], n: usize, k: usize,
) -> Vec<f64> {
    let total_m: usize = ms.iter().sum();
    let mut out = vec![0.0f64; total_m * n];
    let mut offset = 0;
    for (g, &mi) in ms.iter().enumerate() {
        let a_group = &a[(offset * k)..((offset + mi) * k)];
        let b_group = &b[(g * n * k)..((g + 1) * n * k)];
        let ref_group = cpu_ref_f64(a_group, b_group, mi, n, k);
        out[(offset * n)..((offset + mi) * n)].copy_from_slice(&ref_group);
        offset += mi;
    }
    out
}

fn run_grouped_test(ms: &[usize], n: usize, k: usize, gpu: &Gpu) {
    let num_groups = ms.len();
    let total_m: usize = ms.iter().sum();

    let a_f32 = rand_f32(total_m * k);
    let b_f32 = rand_f32(num_groups * n * k);

    let ref64 = cpu_grouped_ref_f64(
        &a_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &b_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        ms, n, k,
    );

    // Build grouped_layout: [total_M] int32, grouped_layout[r] = group_id
    let mut grouped_layout = vec![0i32; total_m];
    let mut offset = 0;
    for (g, &mi) in ms.iter().enumerate() {
        for r in 0..mi {
            grouped_layout[offset + r] = g as i32;
        }
        offset += mi;
    }

    let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let result: Vec<f64> = {
        let a_gpu = gpu.upload(&a_bf16);
        let b_gpu = gpu.upload(&b_bf16);
        let layout_gpu = gpu.upload(&grouped_layout);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(total_m * n);

        {
            let (ap, _g1) = a_gpu.device_ptr(&gpu.stream);
            let (bp, _g2) = b_gpu.device_ptr(&gpu.stream);
            let (lp, _g3) = layout_gpu.device_ptr(&gpu.stream);
            let (op, _g4) = out_gpu.device_ptr_mut(&gpu.stream);
            unsafe {
                prelude_deepgemm::m_grouped_bf16_gemm(
                    ap as *mut c_void, bp as *mut c_void, op as *mut c_void,
                    lp as *mut c_void,
                    total_m as i32, n as i32, k as i32,
                    num_groups as i32,
                    gpu.stream_ptr(),
                ).unwrap();
            }
        }
        gpu.sync();
        gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect()
    };

    let err = max_abs_err(&ref64, &result);
    assert!(err < 1.0,
        "Grouped GEMM G={num_groups} M={total_m} N={n} K={k}: max_err={err:.6e}");
}

#[test]
fn grouped_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // 2 groups, each 128 rows (aligned to 128)
    run_grouped_test(&[128, 128], 256, 256, &gpu);
    // 4 groups, each 128 rows
    run_grouped_test(&[128, 128, 128, 128], 128, 128, &gpu);
}

#[test]
fn grouped_gemm_moe_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // DeepSeek-V3 MoE: 8 experts, hidden=7168, intermediate=4096
    for (ms, n, k) in [
        (vec![128; 4], 4096, 1024),
        (vec![128; 8], 7168, 4096),
        (vec![256; 4], 4096, 4096),
    ] {
        run_grouped_test(&ms, n, k, &gpu);
    }
}

#[test]
fn grouped_gemm_unequal_groups() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // Unequal per-group M sizes (all aligned to 128)
    run_grouped_test(&[128, 256, 128], 1024, 1024, &gpu);
    run_grouped_test(&[256, 128, 384, 128], 4096, 1024, &gpu);
}

#[test]
fn query_grouped_config_returns_valid() {
    let shapes = [(256, 4096, 4096), (1024, 7168, 4096), (512, 4096, 1024)];
    for (m, n, k) in shapes {
        let (bm, bn, stages, smem) = prelude_deepgemm::query_grouped_config(m, n, k);
        assert_eq!(bm, 128, "Grouped block_m should always be 128");
        assert!(bn > 0, "block_n should be > 0 for M={m}");
        assert!(stages > 0, "stages should be > 0 for M={m}");
        assert!(smem > 0, "smem should be > 0 for M={m}");
    }
}

// ============================================================================
// M-Grouped Contiguous FP8 GEMM
// ============================================================================

fn run_fp8_grouped_test(ms: &[usize], n: usize, k: usize, gpu: &Gpu) {
    assert!(k % 128 == 0, "K must be multiple of 128 for FP8");
    let num_groups = ms.len();
    let total_m: usize = ms.iter().sum();

    let a_f32 = rand_f32(total_m * k);
    let b_f32 = rand_f32(num_groups * n * k);

    // Quantize to FP8 with per-token scaling
    let (a_fp8, scale_a) = quantize_to_fp8(&a_f32, total_m, k);
    let (b_fp8, scale_b) = quantize_to_fp8(&b_f32, num_groups * n, k);

    // CPU reference using dequantized values, per-group matmul
    let a_deq = dequantize_fp8_f64(&a_fp8, &scale_a, total_m, k);
    let b_deq = dequantize_fp8_f64(&b_fp8, &scale_b, num_groups * n, k);
    let ref64 = cpu_grouped_ref_f64(&a_deq, &b_deq, ms, n, k);

    // Build grouped_layout
    let mut grouped_layout = vec![0i32; total_m];
    let mut offset = 0;
    for (g, &mi) in ms.iter().enumerate() {
        for r in 0..mi { grouped_layout[offset + r] = g as i32; }
        offset += mi;
    }

    let result: Vec<f64> = {
        let a_gpu = gpu.upload(&a_fp8);
        let b_gpu = gpu.upload(&b_fp8);
        let sfa_gpu = gpu.upload(&scale_a);
        let sfb_gpu = gpu.upload(&scale_b);
        let layout_gpu = gpu.upload(&grouped_layout);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(total_m * n);

        {
            let (ap, _) = a_gpu.device_ptr(&gpu.stream);
            let (bp, _) = b_gpu.device_ptr(&gpu.stream);
            let (sfap, _) = sfa_gpu.device_ptr(&gpu.stream);
            let (sfbp, _) = sfb_gpu.device_ptr(&gpu.stream);
            let (lp, _) = layout_gpu.device_ptr(&gpu.stream);
            let (op, _) = out_gpu.device_ptr_mut(&gpu.stream);
            unsafe {
                prelude_deepgemm::m_grouped_fp8_gemm(
                    ap as *mut c_void, bp as *mut c_void, op as *mut c_void,
                    sfap as *mut c_void, sfbp as *mut c_void,
                    lp as *mut c_void,
                    total_m as i32, n as i32, k as i32,
                    num_groups as i32,
                    gpu.stream_ptr(),
                ).unwrap();
            }
        }
        gpu.sync();
        gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect()
    };

    let err = max_abs_err(&ref64, &result);
    assert!(err < 1.0,
        "Grouped FP8 G={num_groups} M={total_m} N={n} K={k}: max_err={err:.6e}");
}

#[test]
fn fp8_grouped_gemm_small() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    // K must be multiple of 128
    run_fp8_grouped_test(&[128, 128], 256, 256, &gpu);
    run_fp8_grouped_test(&[128, 128, 128, 128], 128, 128, &gpu);
}

#[test]
fn fp8_grouped_gemm_moe_shapes() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    for (ms, n, k) in [
        (vec![128; 4], 1024, 1024),
        (vec![128; 8], 4096, 4096),
    ] {
        run_fp8_grouped_test(&ms, n, k, &gpu);
    }
}
