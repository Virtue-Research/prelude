//! Correctness tests for the Dao-AILab causal-conv1d bindings.
//!
//! Each test allocates GPU memory via cudarc, runs our `extern "C"`
//! shim, and compares against a CPU F64 reference implementation of
//! the same short causal 1D conv. Requires a CUDA GPU to run — if
//! `CudaContext::new` fails (no GPU / driver) the tests skip quietly.
//!
//! Run:  cargo test -p causal-conv1d --release

use std::ffi::c_void;
use std::sync::Arc;

use causal_conv1d::{causal_conv1d_fwd, causal_conv1d_update, Dtype};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

// ─── CPU reference ──────────────────────────────────────────────────

/// Depthwise short causal 1D conv over a whole sequence.
///
/// `x: [B, D, L]`, `w: [D, K]`, optional `bias: [D]`, optional
/// `init: [B, D, K-1]` for left context. Computes
///   `out[b, c, t] = sum_{k=0..K-1} x_padded[b, c, t+k] * w[c, k] + bias[c]`
/// where `x_padded[:, :, 0..K-1]` is the left context (from `init` or
/// zeros) and `x_padded[:, :, K-1..]` is the input. Optional SiLU tail.
fn cpu_ref_fwd(
    x: &[f64],
    w: &[f64],
    bias: Option<&[f64]>,
    init: Option<&[f64]>,
    b: usize,
    d: usize,
    l: usize,
    k: usize,
    silu: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; b * d * l];
    for bi in 0..b {
        for c in 0..d {
            for t in 0..l {
                let mut acc = bias.map(|v| v[c]).unwrap_or(0.0);
                for ki in 0..k {
                    // Position in the logical "padded" sequence: `t + ki`
                    // where 0..K-1 is left context and K-1..K-1+L is x.
                    // For a causal conv `out[t]` depends on
                    // `x[t], x[t-1], ..., x[t-K+1]`, so we read from
                    // the padded index `t + ki - (K-1) + (K-1) = t + ki`
                    // and offset into init for the pad region.
                    let padded_idx = (t as isize) + (ki as isize) - (k as isize - 1);
                    let val = if padded_idx < 0 {
                        // left context: init[bi, c, (K-1) + padded_idx]
                        let init_off = (k as isize - 1 + padded_idx) as usize;
                        init.map(|i| i[bi * d * (k - 1) + c * (k - 1) + init_off])
                            .unwrap_or(0.0)
                    } else {
                        x[bi * d * l + c * l + padded_idx as usize]
                    };
                    acc += val * w[c * k + ki];
                }
                if silu {
                    acc = acc / (1.0 + (-acc).exp());
                }
                out[bi * d * l + c * l + t] = acc;
            }
        }
    }
    out
}

/// Single-token decode step.
///
/// Given `x: [B, D]` (the new token), `state: [B, D, K-1]` (the last
/// K-1 inputs), and `w: [D, K]`, compute the conv1d output for the new
/// token and return the updated state (shifted left by one with `x`
/// appended). The state input is **not mutated** — tests compare the
/// kernel's in-place update against the CPU reference's returned copy.
fn cpu_ref_update(
    x: &[f64],
    state: &[f64],
    w: &[f64],
    bias: Option<&[f64]>,
    b: usize,
    d: usize,
    k: usize,
    silu: bool,
) -> (Vec<f64>, Vec<f64>) {
    let k_minus_1 = k - 1;
    let mut out = vec![0.0f64; b * d];
    let mut new_state = vec![0.0f64; b * d * k_minus_1];
    for bi in 0..b {
        for c in 0..d {
            // "Window" = state[bi, c, :] ++ [x[bi, c]], size K.
            let mut window = vec![0.0f64; k];
            for i in 0..k_minus_1 {
                window[i] = state[bi * d * k_minus_1 + c * k_minus_1 + i];
            }
            window[k_minus_1] = x[bi * d + c];

            // Conv output at the new token.
            let mut acc = bias.map(|v| v[c]).unwrap_or(0.0);
            for ki in 0..k {
                acc += window[ki] * w[c * k + ki];
            }
            if silu {
                acc = acc / (1.0 + (-acc).exp());
            }
            out[bi * d + c] = acc;

            // New state = window[1..K] = old state shifted + new x at tail.
            for i in 0..k_minus_1 {
                new_state[bi * d * k_minus_1 + c * k_minus_1 + i] = window[i + 1];
            }
        }
    }
    (out, new_state)
}

// ─── CUDA helpers ───────────────────────────────────────────────────

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
    fn alloc_bf16(&self, data: &[f64]) -> CudaSlice<half::bf16> {
        let bf: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x as f32)).collect();
        self.upload(&bf)
    }
    fn sync(&self) {
        self.stream.synchronize().unwrap();
    }
}

fn ptr<T>(s: &CudaSlice<T>, stream: &CudaStream) -> *const c_void {
    let (p, _g) = s.device_ptr(stream);
    p as *const c_void
}
fn ptr_mut<T>(s: &mut CudaSlice<T>, stream: &CudaStream) -> *mut c_void {
    let (p, _g) = s.device_ptr_mut(stream);
    p as *mut c_void
}

fn max_abs(ref_: &[f64], got: &[f64]) -> f64 {
    ref_.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
    // Simple LCG so the tests stay deterministic without pulling in
    // `rand` as a heavy dep.
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            u - 0.5
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────

/// BF16 forward with width=4 (Qwen3.5 DeltaNet shape) and no
/// initial_states. Tests basic conv + L2 tail, no SiLU.
#[test]
fn fwd_bf16_width4_no_init() {
    let gpu = match Gpu::new() { Some(g) => g, None => { eprintln!("no GPU, skipping"); return; } };
    let (b, d, l, k) = (1usize, 64usize, 32usize, 4usize);

    let x_f64 = rand_vec(b * d * l, 1);
    let w_f64 = rand_vec(d * k, 2);
    let ref_out = cpu_ref_fwd(&x_f64, &w_f64, None, None, b, d, l, k, false);

    let x_gpu = gpu.alloc_bf16(&x_f64);
    let w_gpu = gpu.alloc_bf16(&w_f64);
    let mut o_gpu = gpu.alloc_bf16(&vec![0.0f64; b * d * l]);
    unsafe {
        causal_conv1d_fwd(
            gpu.stream_ptr(),
            ptr(&x_gpu, &gpu.stream),
            ptr(&w_gpu, &gpu.stream),
            None,
            None,
            None,
            ptr_mut(&mut o_gpu, &gpu.stream),
            b as i32, d as i32, l as i32, k as i32,
            false,
            Dtype::BF16, Dtype::BF16,
        )
        .unwrap();
    }
    gpu.sync();

    let got: Vec<f64> = gpu.download(&o_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let err = max_abs(&ref_out, &got);
    eprintln!("fwd_bf16_width4_no_init: max_abs_err={err:.6e}");
    assert!(err < 5e-2, "causal_conv1d_fwd max_abs_err={err}");
}

/// BF16 forward + fused SiLU (the path Qwen3.5 uses in prefill).
///
/// We pass `initial_states = None`. Upstream's channel-first kernel
/// path silently **ignores** `initial_states` — only the channel-last
/// layout variant honors it (see `causal_conv1d.cpp:213`:
/// `"initial_states is only supported for channel last layout"`).
/// Our Rust shim feeds tensors in channel-first layout (stride_l=1),
/// so passing an `initial_states` pointer would look like it works
/// but the kernel would actually leave the left context at zero. If
/// we start needing multi-chunk prefill on Qwen3.5 (current serving
/// only uses single-chunk), we'll need to either transpose x to
/// channel-last layout before the call or add a separate kernel path.
#[test]
fn fwd_bf16_silu_no_init() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    let (b, d, l, k) = (1usize, 128usize, 64usize, 4usize);

    let x_f64 = rand_vec(b * d * l, 3);
    let w_f64 = rand_vec(d * k, 4);
    let ref_out = cpu_ref_fwd(&x_f64, &w_f64, None, None, b, d, l, k, true);

    let x_gpu = gpu.alloc_bf16(&x_f64);
    let w_gpu = gpu.alloc_bf16(&w_f64);
    let mut o_gpu = gpu.alloc_bf16(&vec![0.0f64; b * d * l]);
    unsafe {
        causal_conv1d_fwd(
            gpu.stream_ptr(),
            ptr(&x_gpu, &gpu.stream),
            ptr(&w_gpu, &gpu.stream),
            None,
            None,
            None,
            ptr_mut(&mut o_gpu, &gpu.stream),
            b as i32, d as i32, l as i32, k as i32,
            true,
            Dtype::BF16, Dtype::BF16,
        )
        .unwrap();
    }
    gpu.sync();

    let got: Vec<f64> = gpu.download(&o_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let err = max_abs(&ref_out, &got);
    eprintln!("fwd_bf16_silu_no_init: max_abs_err={err:.6e}");
    assert!(err < 5e-2, "causal_conv1d_fwd(silu) max_abs_err={err}");
}

/// Regression guard: passing `initial_states` through the shim in
/// **channel-first** layout is silently ignored by upstream's kernel.
/// This test asserts the current broken behavior so we notice if/when
/// upstream starts honoring it, or if/when we switch to channel-last.
///
/// The assertion is deliberately "kernel output == no-init reference"
/// — i.e. any non-zero `initial_states` we pass should NOT affect the
/// output through this code path.
#[test]
fn fwd_bf16_initial_states_silently_ignored_in_channel_first() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    let (b, d, l, k) = (1usize, 64usize, 32usize, 4usize);

    let x_f64 = rand_vec(b * d * l, 10);
    let w_f64 = rand_vec(d * k, 11);
    // Provide a non-zero init — if the kernel honored it, output would
    // differ from the ignore-init reference.
    let init_f64 = rand_vec(b * d * (k - 1), 12);
    let ref_no_init = cpu_ref_fwd(&x_f64, &w_f64, None, None, b, d, l, k, false);

    let x_gpu = gpu.alloc_bf16(&x_f64);
    let w_gpu = gpu.alloc_bf16(&w_f64);
    let init_gpu = gpu.alloc_bf16(&init_f64);
    let mut o_gpu = gpu.alloc_bf16(&vec![0.0f64; b * d * l]);
    unsafe {
        causal_conv1d_fwd(
            gpu.stream_ptr(),
            ptr(&x_gpu, &gpu.stream),
            ptr(&w_gpu, &gpu.stream),
            None,
            Some(ptr(&init_gpu, &gpu.stream)),
            None,
            ptr_mut(&mut o_gpu, &gpu.stream),
            b as i32, d as i32, l as i32, k as i32,
            false,
            Dtype::BF16, Dtype::BF16,
        )
        .unwrap();
    }
    gpu.sync();

    let got: Vec<f64> = gpu.download(&o_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let err = max_abs(&ref_no_init, &got);
    eprintln!("init_ignored: max_abs_err_vs_no_init={err:.6e}");
    // Kernel output should equal the no-init reference within BF16 noise
    // (any larger deviation would mean the init started being honored —
    // update this test if that happens).
    assert!(
        err < 5e-2,
        "channel-first kernel unexpectedly honored initial_states: {err}"
    );
}

/// Single-token decode update. Verifies both the returned output and
/// the in-place conv_state mutation.
#[test]
fn update_bf16_width4() {
    let gpu = match Gpu::new() { Some(g) => g, None => return };
    let (b, d, k) = (2usize, 128usize, 4usize);

    let x_f64 = rand_vec(b * d, 6);
    let state_f64 = rand_vec(b * d * (k - 1), 7);
    let w_f64 = rand_vec(d * k, 8);
    let (ref_out, ref_new_state) = cpu_ref_update(&x_f64, &state_f64, &w_f64, None, b, d, k, false);

    let x_gpu = gpu.alloc_bf16(&x_f64);
    let mut state_gpu = gpu.alloc_bf16(&state_f64);
    let w_gpu = gpu.alloc_bf16(&w_f64);
    let mut o_gpu = gpu.alloc_bf16(&vec![0.0f64; b * d]);

    unsafe {
        causal_conv1d_update(
            gpu.stream_ptr(),
            ptr(&x_gpu, &gpu.stream),
            ptr_mut(&mut state_gpu, &gpu.stream),
            ptr(&w_gpu, &gpu.stream),
            None,
            ptr_mut(&mut o_gpu, &gpu.stream),
            None,
            b as i32, d as i32, k as i32,
            (k - 1) as i32,
            false,
            Dtype::BF16, Dtype::BF16,
        )
        .unwrap();
    }
    gpu.sync();

    let got_out: Vec<f64> = gpu.download(&o_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let got_state: Vec<f64> = gpu.download(&state_gpu).iter().map(|x| x.to_f32() as f64).collect();
    let out_err = max_abs(&ref_out, &got_out);
    let state_err = max_abs(&ref_new_state, &got_state);
    eprintln!("update_bf16_width4: out_err={out_err:.6e} state_err={state_err:.6e}");
    assert!(out_err < 5e-2, "update out max_abs_err={out_err}");
    assert!(state_err < 5e-2, "update state max_abs_err={state_err}");
}
