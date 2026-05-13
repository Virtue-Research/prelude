//! Multi-block argmax over a contiguous `[B, V]` logits tensor.
//!
//! Candle's stock `Tensor::argmax` dispatches to `fast_argmax_bf16` /
//! `_f32` (see `kernels_src/candle/reduce.cu`), which launches **one
//! block per output row** and walks the reduce axis through
//! `get_strided_index` even when the tensor is contiguous. At the
//! LM-head — small batch (B≤32), wide vocab (V≈150K) — that leaves
//! 80-110 of the GPU's SMs idle and shows up as ~25% of GPU time in our
//! profiles.
//!
//! This module wraps a two-pass kernel (`fast_argmax_vocab.cu`) that
//! fans out across `blocks_per_row` blocks per row, reads logits with
//! plain linear addressing, and reduces the partials in a single warp.
//! Tie-break (smaller index wins) matches candle / NumPy / PyTorch.
//!
//! Only the BF16 and F32, 2-D contiguous, reduce-last-dim case is
//! handled here; callers should fall back to `tensor.argmax(D::Minus1)`
//! for anything else.

use std::sync::OnceLock;

use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Device, Result, Shape, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_FAST_ARGMAX_VOCAB, PTX_FAST_ARGMAX_VOCAB};

const BLOCK_SIZE: u32 = 256;

// CUDA driver attribute id — keep aligned with the value used in
// `attn::gdn_prefill::detect_sm_count`. See `cuda_runtime_api.h`.
const CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

fn detect_sm_count() -> u32 {
    static CACHE: OnceLock<u32> = OnceLock::new();
    *CACHE.get_or_init(|| unsafe {
        let mut dev = 0i32;
        if cudaGetDevice(&mut dev) != 0 {
            return 114;
        }
        let mut count = 0i32;
        if cudaDeviceGetAttribute(&mut count, CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT, dev) != 0
            || count <= 0
        {
            return 114;
        }
        count as u32
    })
}

/// Pick how many blocks to fan out per row. We want enough blocks to
/// keep all SMs busy when the batch is small, but not so many that the
/// pass-2 reduction outgrows a single warp (32 threads) or that the
/// per-block chunk shrinks below the launch overhead.
fn pick_blocks_per_row(b: u32, v: u32, num_sms: u32) -> u32 {
    // Tiny vocab: a single block is already enough — fall through to
    // pass-2 immediately, no fan-out.
    if v <= 4096 {
        return 1;
    }
    // Keep at least ~2K elements per block so the launch cost doesn't
    // dominate.
    let max_by_chunk = v.div_ceil(2048).max(1);
    // Spread out enough to fill the device when B is small. With B
    // rows × bpr blocks we want B*bpr >= num_sms.
    let want_from_sms = num_sms.div_ceil(b.max(1));
    let bpr = want_from_sms.min(max_by_chunk).min(32);
    bpr.max(1)
}

/// Argmax over the last dim of a contiguous `[B, V]` tensor.
///
/// Returns `[B]` U32. Errors propagate up; callers expecting a
/// best-effort fallback should map them to candle's stock argmax.
pub fn fast_argmax_vocab(logits: &Tensor) -> Result<Tensor> {
    // ── Shape / dtype gate ──────────────────────────────────────────
    let dims = logits.dims();
    if dims.len() != 2 {
        candle_core::bail!(
            "fast_argmax_vocab: expected 2-D tensor, got {:?}",
            dims
        );
    }
    let (b, v) = (dims[0] as u32, dims[1] as u32);
    if b == 0 || v == 0 {
        return Tensor::zeros((b as usize,), DType::U32, logits.device());
    }
    match logits.dtype() {
        DType::BF16 | DType::F32 => {}
        other => candle_core::bail!(
            "fast_argmax_vocab: dtype {:?} not supported (BF16/F32 only)",
            other
        ),
    }

    // ── Device plumbing ─────────────────────────────────────────────
    let dev: Device = logits.device().clone();
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("fast_argmax_vocab: requires a CUDA device"),
    };

    let logits_c = logits.contiguous()?;
    let (logits_storage, logits_layout) = logits_c.storage_and_layout();
    let logits_cuda = match &*logits_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("fast_argmax_vocab: logits not on CUDA"),
    };

    // ── Launch geometry ─────────────────────────────────────────────
    // Best-effort SM count for the heuristic — only affects throughput,
    // never correctness. Falls back to 114 (H100 PCIe) on query failure.
    let bpr = pick_blocks_per_row(b, v, detect_sm_count());
    let chunk_size = v.div_ceil(bpr);

    // ── Output / scratch buffers ────────────────────────────────────
    let n_partials = (b * bpr) as usize;
    let mut partials_val = unsafe { cuda_dev.alloc::<f32>(n_partials) }?;
    let mut partials_idx = unsafe { cuda_dev.alloc::<u32>(n_partials) }?;
    let mut out_idx = unsafe { cuda_dev.alloc::<u32>(b as usize) }?;

    // ── Pass 1: chunked local argmax ────────────────────────────────
    let cfg1 = LaunchConfig {
        grid_dim: (b, bpr, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    match logits.dtype() {
        DType::BF16 => {
            let slice = logits_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(logits_layout.start_offset()..);
            let func = cuda_dev.get_or_load_custom_func(
                "fast_argmax_vocab_pass1_bf16",
                MOD_FAST_ARGMAX_VOCAB,
                PTX_FAST_ARGMAX_VOCAB,
            )?;
            let mut builder = func.builder();
            builder.arg(&slice);
            builder.arg(&mut partials_val);
            builder.arg(&mut partials_idx);
            builder.arg(&b);
            builder.arg(&v);
            builder.arg(&bpr);
            builder.arg(&chunk_size);
            unsafe { builder.launch(cfg1) }.w()?;
        }
        DType::F32 => {
            let slice = logits_cuda
                .as_cuda_slice::<f32>()?
                .slice(logits_layout.start_offset()..);
            let func = cuda_dev.get_or_load_custom_func(
                "fast_argmax_vocab_pass1_f32",
                MOD_FAST_ARGMAX_VOCAB,
                PTX_FAST_ARGMAX_VOCAB,
            )?;
            let mut builder = func.builder();
            builder.arg(&slice);
            builder.arg(&mut partials_val);
            builder.arg(&mut partials_idx);
            builder.arg(&b);
            builder.arg(&v);
            builder.arg(&bpr);
            builder.arg(&chunk_size);
            unsafe { builder.launch(cfg1) }.w()?;
        }
        _ => unreachable!("validated above"),
    }

    drop(logits_storage);

    // ── Pass 2: reduce partials per row ─────────────────────────────
    // Single warp per row reduces up to 32 partials.
    let cfg2 = LaunchConfig {
        grid_dim: (b, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let func2 = cuda_dev.get_or_load_custom_func(
        "fast_argmax_vocab_pass2",
        MOD_FAST_ARGMAX_VOCAB,
        PTX_FAST_ARGMAX_VOCAB,
    )?;
    let mut builder = func2.builder();
    builder.arg(&partials_val);
    builder.arg(&partials_idx);
    builder.arg(&mut out_idx);
    builder.arg(&b);
    builder.arg(&bpr);
    unsafe { builder.launch(cfg2) }.w()?;

    // ── Wrap output back into a candle Tensor ───────────────────────
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out_idx, cuda_dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        Shape::from((b as usize,)),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, Device};

    /// xorshift64 — deterministic, no dep, plenty of entropy for a
    /// correctness test.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn rand_unit(state: &mut u64) -> f32 {
        // Map xorshift bits to a uniform float in [-10, 10).
        // Mask to the 23-bit mantissa first, then OR the 1.0 exponent —
        // OR-ing the raw random word would set the exponent bits and
        // randomly produce inf/NaN.
        let u = xorshift64(state);
        let mantissa = (u as u32) & 0x007F_FFFF;
        let bits = mantissa | 0x3F80_0000;
        (f32::from_bits(bits) - 1.5) * 20.0
    }

    fn make_random_logits(b: usize, v: usize, seed: u64, dev: &Device) -> Tensor {
        // Generate random f32 then cast to bf16 (the rounding noise is
        // identical for both kernels, so the comparison stays exact for
        // non-tied entries — which is overwhelmingly the case).
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let data: Vec<f32> = (0..b * v).map(|_| rand_unit(&mut state)).collect();
        Tensor::from_vec(data, (b, v), dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    /// Ground-truth CPU argmax over a `[B, V]` BF16 tensor.
    /// Returns the smallest column index per row achieving the max
    /// (same tie-break as candle / NumPy / our kernel).
    fn cpu_argmax(logits: &Tensor, b: usize, v: usize) -> Vec<u32> {
        let host: Vec<half::bf16> = logits.flatten_all().unwrap().to_vec1().unwrap();
        (0..b)
            .map(|row| {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = 0u32;
                for col in 0..v {
                    let v_bf16 = host[row * v + col].to_f32();
                    if v_bf16 > best_val {
                        best_val = v_bf16;
                        best_idx = col as u32;
                    }
                }
                best_idx
            })
            .collect()
    }

    fn run_and_compare(b: usize, v: usize, seed: u64) {
        crate::register();
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let logits = make_random_logits(b, v, seed, &dev);

        let cpu_ref = cpu_argmax(&logits, b, v);

        let candle_idx: Vec<u32> = logits
            .argmax(D::Minus1)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();

        let fast_idx: Vec<u32> = fast_argmax_vocab(&logits)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();

        // Print all three for visibility — candle and CPU disagree on
        // several common (B, V) combinations, so this serves as a
        // running record of where candle's argmax misbehaves.
        eprintln!(
            "B={b} V={v} seed={seed}\n  cpu    = {cpu_ref:?}\n  candle = {candle_idx:?}\n  fast   = {fast_idx:?}",
        );
        assert_eq!(
            fast_idx, cpu_ref,
            "fast_argmax disagreed with CPU reference (B={b} V={v} seed={seed})"
        );
    }

    // NOTE: candle's stock `fast_argmax_bf16` (in `reduce.cu`) is buggy
    // on several common shapes — it disagrees with the CPU reference
    // for V ∈ {4096, 65537, 100003, …}, returning indices off by tens
    // of thousands. So we compare against the CPU reference, not
    // candle. The eprintln below documents that disagreement (search
    // `cpu = … candle = …` in the test output).

    #[test]
    fn fast_argmax_vs_cpu_small() {
        run_and_compare(1, 4096, 1);
        run_and_compare(1, 4097, 2);
        run_and_compare(4, 1024, 3);
    }

    #[test]
    fn fast_argmax_vs_cpu_lm_head() {
        // Qwen3 vocab — the actual hot path.
        run_and_compare(1, 151_936, 11);
        run_and_compare(4, 151_936, 12);
        run_and_compare(16, 151_936, 13);
        run_and_compare(32, 151_936, 14);
    }

    #[test]
    fn fast_argmax_vs_cpu_uneven_chunk() {
        // V not divisible by typical block counts — exercises the
        // chunk_size rounding in pass-1.
        run_and_compare(2, 65_537, 21);
        run_and_compare(3, 100_003, 22);
    }

    /// One-hot input: the max is at a known column index per row.
    /// Sanity-checks both passes deterministically without depending on
    /// candle's argmax as a reference.
    #[test]
    fn fast_argmax_one_hot() {
        crate::register();
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let b = 4usize;
        let v = 151_936usize;
        let want: [usize; 4] = [0, 1, 75_000, 151_935];

        let mut data = vec![0.0f32; b * v];
        for (row, &col) in want.iter().enumerate() {
            data[row * v + col] = 1.0;
        }
        let logits = Tensor::from_vec(data, (b, v), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let got: Vec<u32> = fast_argmax_vocab(&logits)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let want_u32: Vec<u32> = want.iter().map(|&x| x as u32).collect();
        assert_eq!(got, want_u32, "one-hot argmax mismatch");
    }
}
