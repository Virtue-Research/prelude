//! cuLA KDA correctness tests — SM90 fused prefill.
//!
//! Each test allocates GPU memory via cudarc, runs cuLA KDA prefill, and compares
//! against a CPU F64 reference implementing the gated delta rule recurrence.
//!
//! Run:  cargo test -p prelude-cula --release

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits};

// ── CPU reference: KDA gated delta rule (F64 ground truth) ──────────
//
// Per-token recurrence (simplified, no gate/alpha for basic test):
//   state_t = state_{t-1} + beta_t * (k_t ⊗ v_t)
//   output_t = scale * (q_t @ state_t)
//
// With alpha (forget gate):
//   state_t = alpha_t * state_{t-1} + beta_t * (k_t ⊗ v_t)
//   output_t = scale * (q_t @ state_t)
//
// Shapes:
//   Q, K, V: [total_seq, H, D] (packed varlen)
//   state: [H, D, D] per sequence
//   cu_seqlens: [num_seqs + 1]

/// CPU reference for KDA prefill (F64).
/// Returns (output [total_seq, H, D], final_states [num_seqs, H, D, D]).
fn cpu_kda_prefill_f64(
    q: &[f64], k: &[f64], v: &[f64],
    alpha: Option<&[f64]>,  // [total_seq, H, D] or None
    beta: Option<&[f64]>,   // [total_seq, H] or None
    cu_seqlens: &[i32],
    h: usize, d: usize,
    scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let num_seqs = cu_seqlens.len() - 1;
    let total_seq = *cu_seqlens.last().unwrap() as usize;
    let mut output = vec![0.0f64; total_seq * h * d];
    let mut final_states = vec![0.0f64; num_seqs * h * d * d];

    for seq in 0..num_seqs {
        let start = cu_seqlens[seq] as usize;
        let end = cu_seqlens[seq + 1] as usize;

        // state: [H, D, D]
        let mut state = vec![0.0f64; h * d * d];

        for t in start..end {
            for hi in 0..h {
                // Get beta for this token/head (default 1.0)
                let b = beta.map_or(1.0, |betas| betas[t * h + hi]);

                // Update state: state[hi] += beta * (k_t ⊗ v_t)
                // With alpha: state[hi] = alpha * state[hi] + beta * (k_t ⊗ v_t)
                for di in 0..d {
                    let k_val = k[(t * h + hi) * d + di];

                    // Apply alpha (forget gate) if present
                    if let Some(alphas) = alpha {
                        let a = alphas[(t * h + hi) * d + di];
                        for dj in 0..d {
                            state[(hi * d + di) * d + dj] *= a;
                        }
                    }

                    for dj in 0..d {
                        let v_val = v[(t * h + hi) * d + dj];
                        state[(hi * d + di) * d + dj] += b * k_val * v_val;
                    }
                }

                // output_t = scale * (q_t @ state[hi])
                for di in 0..d {
                    let mut acc = 0.0f64;
                    for dk in 0..d {
                        let q_val = q[(t * h + hi) * d + dk];
                        acc += q_val * state[(hi * dk) * d + di];
                    }
                    // Fix: q_val * state[hi, dk, di] where state is [H, D_k, D_v]
                    output[(t * h + hi) * d + di] = scale * acc;
                }
            }
        }

        // Save final state
        let state_off = seq * h * d * d;
        final_states[state_off..state_off + h * d * d].copy_from_slice(&state);
    }

    // Recompute output correctly: output[t,h,dv] = scale * sum_dk(q[t,h,dk] * state[h,dk,dv])
    output.fill(0.0);
    for seq in 0..num_seqs {
        let start = cu_seqlens[seq] as usize;
        let end = cu_seqlens[seq + 1] as usize;
        let mut state = vec![0.0f64; h * d * d];

        for t in start..end {
            for hi in 0..h {
                let b = beta.map_or(1.0, |betas| betas[t * h + hi]);

                for dk in 0..d {
                    if let Some(alphas) = alpha {
                        let a = alphas[(t * h + hi) * d + dk];
                        for dv in 0..d {
                            state[(hi * d + dk) * d + dv] *= a;
                        }
                    }
                    let k_val = k[(t * h + hi) * d + dk];
                    for dv in 0..d {
                        let v_val = v[(t * h + hi) * d + dv];
                        state[(hi * d + dk) * d + dv] += b * k_val * v_val;
                    }
                }

                for dv in 0..d {
                    let mut acc = 0.0f64;
                    for dk in 0..d {
                        acc += q[(t * h + hi) * d + dk] * state[(hi * d + dk) * d + dv];
                    }
                    output[(t * h + hi) * d + dv] = scale * acc;
                }
            }
        }

        let state_off = seq * h * d * d;
        final_states[state_off..state_off + h * d * d].copy_from_slice(&state);
    }

    (output, final_states)
}

// ── CUDA helpers ────────────────────────────────────────────────────

struct Gpu {
    stream: Arc<CudaStream>,
}

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

fn detect_sm() -> (i32, i32) {
    unsafe {
        let mut dev = 0i32;
        cudaGetDevice(&mut dev);
        let (mut major, mut minor) = (0i32, 0i32);
        cudaDeviceGetAttribute(&mut major, 75, dev); // ComputeCapabilityMajor
        cudaDeviceGetAttribute(&mut minor, 76, dev); // ComputeCapabilityMinor
        (major, minor)
    }
}

fn detect_sm_count() -> i32 {
    unsafe {
        let mut dev = 0i32;
        cudaGetDevice(&mut dev);
        let mut count = 0i32;
        cudaDeviceGetAttribute(&mut count, 16, dev); // MultiprocessorCount
        if count > 0 { count } else { 132 }
    }
}

impl Gpu {
    fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let (major, _) = detect_sm();
        if major < 9 {
            eprintln!("cuLA tests require SM90+, skipping (SM{major}x)");
            return None;
        }
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
    p as *const c_void
}

fn ptr_mut<T>(s: &mut CudaSlice<T>, stream: &CudaStream) -> *mut c_void {
    let (p, _guard) = s.device_ptr_mut(stream);
    p as *mut c_void
}

fn rand_f32(len: usize, scale: f32) -> Vec<f32> {
    use rand::RngExt;
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(-scale..scale)).collect()
}

fn max_abs_err(reference: &[f64], result: &[f64]) -> f64 {
    reference.iter().zip(result).map(|(r, t)| (r - t).abs()).fold(0.0f64, f64::max)
}

fn mean_abs_err(reference: &[f64], result: &[f64]) -> f64 {
    let sum: f64 = reference.iter().zip(result).map(|(r, t)| (r - t).abs()).sum();
    sum / reference.len() as f64
}

// ── Tests ───────────────────────────────────────────────────────────

/// Basic correctness: single sequence with alpha (forget gate) and beta (update gate).
/// KDA SM90 kernel requires alpha + beta + safe_gate=true.
///
/// Recurrence: state_t = alpha_t * state_{t-1} + beta_t * (k_t ⊗ v_t)
///             output_t = scale * (q_t @ state_t)
#[test]
fn kda_prefill_sm90_basic() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    let h = 4;
    let d = 128; // KDA kernel requires D=128 (TileShape<64,64,128>)
    let seq_len = 64; // Keep short for CPU reference tractability
    let total_seq = seq_len;
    let scale = 1.0 / (d as f32).sqrt();

    let q_f32 = rand_f32(total_seq * h * d, 0.1);
    let k_f32 = rand_f32(total_seq * h * d, 0.1);
    let v_f32 = rand_f32(total_seq * h * d, 0.1);
    // Alpha (forget gate): per [total_seq, H, D], values near 1.0 for mild decay
    let alpha_f32: Vec<f32> = rand_f32(total_seq * h * d, 0.05)
        .iter().map(|&x| 0.95 + x.abs()).collect();
    // Beta (update gate): per [total_seq, H], small positive values
    let beta_f32: Vec<f32> = rand_f32(total_seq * h, 0.1)
        .iter().map(|x| x.abs() + 0.01).collect();
    let cu_seqlens: Vec<i32> = vec![0, seq_len as i32];

    // CPU reference
    let q_f64: Vec<f64> = q_f32.iter().map(|&x| x as f64).collect();
    let k_f64: Vec<f64> = k_f32.iter().map(|&x| x as f64).collect();
    let v_f64: Vec<f64> = v_f32.iter().map(|&x| x as f64).collect();
    let alpha_f64: Vec<f64> = alpha_f32.iter().map(|&x| x as f64).collect();
    let beta_f64: Vec<f64> = beta_f32.iter().map(|&x| x as f64).collect();
    let (ref_output, ref_state) = cpu_kda_prefill_f64(
        &q_f64, &k_f64, &v_f64,
        Some(&alpha_f64), Some(&beta_f64),
        &cu_seqlens, h, d, scale as f64,
    );

    // GPU: convert to BF16 (Q/K/V) and F32 (alpha/beta)
    let q_bf16: Vec<half::bf16> = q_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let k_bf16: Vec<half::bf16> = k_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let v_bf16: Vec<half::bf16> = v_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let q_gpu = gpu.upload(&q_bf16);
    let k_gpu = gpu.upload(&k_bf16);
    let v_gpu = gpu.upload(&v_bf16);
    let alpha_gpu = gpu.upload(&alpha_f32);
    let beta_gpu = gpu.upload(&beta_f32);
    let cu_gpu = gpu.upload(&cu_seqlens);
    let mut out_gpu = gpu.alloc_zeros::<half::bf16>(total_seq * h * d);
    let mut state_gpu = gpu.alloc_zeros::<f32>(1 * h * d * d);
    let mut workspace_gpu = gpu.alloc_zeros::<u8>(64 * 1024 * 1024); // 64MB

    {
        let qp = ptr(&q_gpu, &gpu.stream);
        let kp = ptr(&k_gpu, &gpu.stream);
        let vp = ptr(&v_gpu, &gpu.stream);
        let ap = ptr(&alpha_gpu, &gpu.stream) as *const f32;
        let bp = ptr(&beta_gpu, &gpu.stream) as *const f32;
        let cup = ptr(&cu_gpu, &gpu.stream) as *const i32;
        let op = ptr_mut(&mut out_gpu, &gpu.stream);
        let sp = ptr_mut(&mut state_gpu, &gpu.stream) as *mut f32;
        let wp = ptr_mut(&mut workspace_gpu, &gpu.stream) as *mut u8;

        unsafe {
            prelude_cula::kda_fwd_prefill_sm90(
                gpu.stream_ptr(),
                op, sp,
                qp, kp, vp,
                None,     // no input state
                Some(ap), // alpha (forget gate)
                Some(bp), // beta (update gate)
                cup, wp,
                1,     // num_seqs
                h as i32, d as i32,
                total_seq as i64,
                scale, true, // safe_gate required
                detect_sm_count(),
            ).unwrap();
        }
    }
    gpu.sync();

    let result_bf16 = gpu.download(&out_gpu);
    let result: Vec<f64> = result_bf16.iter().map(|x| x.to_f32() as f64).collect();
    let result_state_f32 = gpu.download(&state_gpu);
    let result_state: Vec<f64> = result_state_f32.iter().map(|&x| x as f64).collect();

    let out_err = max_abs_err(&ref_output, &result);
    let out_mean = mean_abs_err(&ref_output, &result);
    let state_err = max_abs_err(&ref_state, &result_state);

    eprintln!("KDA basic: output max_err={out_err:.6e} mean_err={out_mean:.6e}");
    eprintln!("KDA basic: state max_err={state_err:.6e}");

    // cuLA tolerance: output atol=5e-3, state atol=8e-3
    assert!(out_err < 5e-3, "KDA output max_err={out_err:.6e} (tol=5e-3)");
    assert!(state_err < 8e-3, "KDA state max_err={state_err:.6e} (tol=8e-3)");
}

/// Multiple sequences (varlen) test.
#[test]
fn kda_prefill_sm90_varlen() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    let h = 4;
    let d = 128;
    let seq_lens = [64, 128, 64]; // Must be multiples of 64 (tile size)
    let total_seq: usize = seq_lens.iter().sum();
    let num_seqs = seq_lens.len();
    let scale = 1.0 / (d as f32).sqrt();

    let mut cu_seqlens = vec![0i32];
    let mut acc = 0i32;
    for &l in &seq_lens {
        acc += l as i32;
        cu_seqlens.push(acc);
    }

    let q_f32 = rand_f32(total_seq * h * d, 0.1);
    let k_f32 = rand_f32(total_seq * h * d, 0.1);
    let v_f32 = rand_f32(total_seq * h * d, 0.1);
    let alpha_f32: Vec<f32> = rand_f32(total_seq * h * d, 0.05)
        .iter().map(|&x| 0.95 + x.abs()).collect();
    let beta_f32: Vec<f32> = rand_f32(total_seq * h, 0.1)
        .iter().map(|x| x.abs() + 0.01).collect();

    // CPU reference
    let q_f64: Vec<f64> = q_f32.iter().map(|&x| x as f64).collect();
    let k_f64: Vec<f64> = k_f32.iter().map(|&x| x as f64).collect();
    let v_f64: Vec<f64> = v_f32.iter().map(|&x| x as f64).collect();
    let alpha_f64: Vec<f64> = alpha_f32.iter().map(|&x| x as f64).collect();
    let beta_f64: Vec<f64> = beta_f32.iter().map(|&x| x as f64).collect();
    let (ref_output, ref_state) = cpu_kda_prefill_f64(
        &q_f64, &k_f64, &v_f64,
        Some(&alpha_f64), Some(&beta_f64),
        &cu_seqlens, h, d, scale as f64,
    );

    // GPU
    let q_bf16: Vec<half::bf16> = q_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let k_bf16: Vec<half::bf16> = k_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let v_bf16: Vec<half::bf16> = v_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let q_gpu = gpu.upload(&q_bf16);
    let k_gpu = gpu.upload(&k_bf16);
    let v_gpu = gpu.upload(&v_bf16);
    let alpha_gpu = gpu.upload(&alpha_f32);
    let beta_gpu = gpu.upload(&beta_f32);
    let cu_gpu = gpu.upload(&cu_seqlens);
    let mut out_gpu = gpu.alloc_zeros::<half::bf16>(total_seq * h * d);
    let mut state_gpu = gpu.alloc_zeros::<f32>(num_seqs * h * d * d);
    let mut workspace_gpu = gpu.alloc_zeros::<u8>(64 * 1024 * 1024);

    {
        let qp = ptr(&q_gpu, &gpu.stream);
        let kp = ptr(&k_gpu, &gpu.stream);
        let vp = ptr(&v_gpu, &gpu.stream);
        let ap = ptr(&alpha_gpu, &gpu.stream) as *const f32;
        let bp = ptr(&beta_gpu, &gpu.stream) as *const f32;
        let cup = ptr(&cu_gpu, &gpu.stream) as *const i32;
        let op = ptr_mut(&mut out_gpu, &gpu.stream);
        let sp = ptr_mut(&mut state_gpu, &gpu.stream) as *mut f32;
        let wp = ptr_mut(&mut workspace_gpu, &gpu.stream) as *mut u8;

        unsafe {
            prelude_cula::kda_fwd_prefill_sm90(
                gpu.stream_ptr(),
                op, sp,
                qp, kp, vp,
                None, Some(ap), Some(bp),
                cup, wp,
                num_seqs as i32,
                h as i32, d as i32,
                total_seq as i64,
                scale, true,
                detect_sm_count(),
            ).unwrap();
        }
    }
    gpu.sync();

    let result: Vec<f64> = gpu.download(&out_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let result_state: Vec<f64> = gpu.download(&state_gpu).iter().map(|&x| x as f64).collect();

    let out_err = max_abs_err(&ref_output, &result);
    let state_err = max_abs_err(&ref_state, &result_state);

    eprintln!("KDA varlen: output max_err={out_err:.6e}, state max_err={state_err:.6e}");
    assert!(out_err < 5e-3, "KDA varlen output max_err={out_err:.6e} (tol=5e-3)");
    // State accumulates over longer sequences (128 tokens) → higher BF16 error
    assert!(state_err < 2e-2, "KDA varlen state max_err={state_err:.6e} (tol=2e-2)");
}

// ============================================================================
// Performance benchmark (not a correctness test — prints timing)
// ============================================================================

#[test]
fn kda_prefill_sm90_perf() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    let configs = [
        // (H, D, seq_len)
        (32, 128, 512),
        (32, 128, 1024),
        (32, 128, 2048),
        (64, 128, 1024),
    ];

    const WARMUP: usize = 5;
    const REPEATS: usize = 20;

    for (h, d, seq_len) in configs {
        let total_seq = seq_len;
        let scale = 1.0 / (d as f32).sqrt();
        let cu_seqlens: Vec<i32> = vec![0, seq_len as i32];

        let q_f32 = rand_f32(total_seq * h * d, 0.1);
        let k_f32 = rand_f32(total_seq * h * d, 0.1);
        let v_f32 = rand_f32(total_seq * h * d, 0.1);

        let q_bf16: Vec<half::bf16> = q_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let k_bf16: Vec<half::bf16> = k_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let v_bf16: Vec<half::bf16> = v_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        let q_gpu = gpu.upload(&q_bf16);
        let k_gpu = gpu.upload(&k_bf16);
        let v_gpu = gpu.upload(&v_bf16);
        let cu_gpu = gpu.upload(&cu_seqlens);
        let mut out_gpu = gpu.alloc_zeros::<half::bf16>(total_seq * h * d);
        let mut state_gpu = gpu.alloc_zeros::<f32>(1 * h * d * d);
        let mut workspace_gpu = gpu.alloc_zeros::<u8>(64 * 1024 * 1024);

        // Alpha/beta for perf test (kernel requires them)
        let alpha_f32: Vec<f32> = (0..total_seq * h * d).map(|_| 0.98f32).collect();
        let beta_f32: Vec<f32> = (0..total_seq * h).map(|_| 0.1f32).collect();
        let alpha_gpu = gpu.upload(&alpha_f32);
        let beta_gpu = gpu.upload(&beta_f32);

        let mut run = || {
            let qp = ptr(&q_gpu, &gpu.stream);
            let kp = ptr(&k_gpu, &gpu.stream);
            let vp = ptr(&v_gpu, &gpu.stream);
            let ap = ptr(&alpha_gpu, &gpu.stream) as *const f32;
            let bp = ptr(&beta_gpu, &gpu.stream) as *const f32;
            let cup = ptr(&cu_gpu, &gpu.stream) as *const i32;
            let op = ptr_mut(&mut out_gpu, &gpu.stream);
            let sp = ptr_mut(&mut state_gpu, &gpu.stream) as *mut f32;
            let wp = ptr_mut(&mut workspace_gpu, &gpu.stream) as *mut u8;
            unsafe {
                prelude_cula::kda_fwd_prefill_sm90(
                    gpu.stream_ptr(),
                    op, sp, qp, kp, vp,
                    None, Some(ap), Some(bp),
                    cup, wp,
                    1, h as i32, d as i32, total_seq as i64,
                    scale, true, detect_sm_count(),
                ).unwrap();
            }
        };

        // Warmup
        for _ in 0..WARMUP { run(); }
        gpu.sync();

        let start = std::time::Instant::now();
        for _ in 0..REPEATS { run(); }
        gpu.sync();
        let elapsed_us = start.elapsed().as_nanos() as f64 / REPEATS as f64 / 1000.0;

        eprintln!(
            "KDA prefill H={h} D={d} seq={seq_len}: {elapsed_us:.1}us ({:.2}ms)",
            elapsed_us / 1000.0
        );
    }
}
