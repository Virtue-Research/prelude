//! Device-side batched EOS / stop-token check.
//!
//! Compares each sampled token in a `[B] u32` tensor against a small
//! set of EOS ids and emits a `[B] u8` done bitmap (1 = finished,
//! 0 = continue). Producing the EOS decision on the device lets the
//! decode loop fold it into the same async stream as the sampler and
//! sync the bitmap only periodically (every K steps) instead of once
//! per step, which is the dominant host-side stall today.
//!
//! Kernel: `crates/prelude-cuda/src/kernels/kernels_src/sample/check_eos.cu`.
//!
//! This file ships only the wrapper — the existing host-side
//! `Engine::is_eos` callers are unchanged. PR-3 / PR-4 in the
//! `Phase 3` series will route the decode loop through this kernel.

use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Device, Result, Shape, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_CHECK_EOS, PTX_CHECK_EOS};

const BLOCK_SIZE: u32 = 256;

/// Device-side EOS check.
///
/// * `tokens` — `[B]` U32 tensor of sampled token ids.
/// * `eos_ids` — `[E]` U32 tensor of stop ids. `E == 0` is allowed
///   (every output bit will be 0).
///
/// Returns a `[B]` U8 tensor; bit `b` is `1` iff `tokens[b] == eos_ids[i]`
/// for some `i`.
pub fn check_eos(tokens: &Tensor, eos_ids: &Tensor) -> Result<Tensor> {
    // ── Shape + dtype validation ────────────────────────────────────
    if tokens.dims().len() != 1 {
        candle_core::bail!(
            "check_eos: tokens must be 1-D, got {:?}",
            tokens.dims()
        );
    }
    if eos_ids.dims().len() != 1 {
        candle_core::bail!(
            "check_eos: eos_ids must be 1-D, got {:?}",
            eos_ids.dims()
        );
    }
    if tokens.dtype() != DType::U32 {
        candle_core::bail!("check_eos: tokens must be U32, got {:?}", tokens.dtype());
    }
    if eos_ids.dtype() != DType::U32 {
        candle_core::bail!(
            "check_eos: eos_ids must be U32, got {:?}",
            eos_ids.dtype()
        );
    }

    let b = tokens.dims()[0] as u32;
    let e = eos_ids.dims()[0] as u32;
    if b == 0 {
        return Tensor::zeros((0,), DType::U8, tokens.device());
    }

    // ── Device plumbing ─────────────────────────────────────────────
    let dev: Device = tokens.device().clone();
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("check_eos: requires a CUDA device"),
    };

    let tokens_c = tokens.contiguous()?;
    let eos_c = eos_ids.contiguous()?;
    let (tok_storage, tok_layout) = tokens_c.storage_and_layout();
    let (eos_storage, eos_layout) = eos_c.storage_and_layout();
    let tok_cuda = match &*tok_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("check_eos: tokens not on CUDA"),
    };
    let eos_cuda = match &*eos_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("check_eos: eos_ids not on CUDA"),
    };

    let tok_slice = tok_cuda
        .as_cuda_slice::<u32>()?
        .slice(tok_layout.start_offset()..);
    let eos_slice = eos_cuda
        .as_cuda_slice::<u32>()?
        .slice(eos_layout.start_offset()..);

    // ── Output buffer ───────────────────────────────────────────────
    let mut done = unsafe { cuda_dev.alloc::<u8>(b as usize) }?;

    // ── Launch ──────────────────────────────────────────────────────
    let block = BLOCK_SIZE.min(b.max(1));
    let grid = b.div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = cuda_dev.get_or_load_custom_func(
        "check_eos_u32",
        MOD_CHECK_EOS,
        PTX_CHECK_EOS,
    )?;
    let mut builder = func.builder();
    builder.arg(&tok_slice);
    builder.arg(&eos_slice);
    builder.arg(&mut done);
    builder.arg(&b);
    builder.arg(&e);
    unsafe { builder.launch(cfg) }.w()?;

    drop(tok_storage);
    drop(eos_storage);

    let storage = candle_core::CudaStorage::wrap_cuda_slice(done, cuda_dev);
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

    #[test]
    fn check_eos_single_match() {
        crate::register();
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let tokens = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (5,), &dev).unwrap();
        let eos_ids = Tensor::from_vec(vec![3u32], (1,), &dev).unwrap();
        let done = check_eos(&tokens, &eos_ids).unwrap();
        let got: Vec<u8> = done.to_vec1().unwrap();
        assert_eq!(got, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn check_eos_multi_match() {
        crate::register();
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Qwen3 has two EOS-ish tokens: <|im_end|>=151645, <|endoftext|>=151643.
        let tokens =
            Tensor::from_vec(vec![151645u32, 7, 151643, 151644, 0], (5,), &dev).unwrap();
        let eos_ids = Tensor::from_vec(vec![151645u32, 151643], (2,), &dev).unwrap();
        let done = check_eos(&tokens, &eos_ids).unwrap();
        let got: Vec<u8> = done.to_vec1().unwrap();
        assert_eq!(got, vec![1, 0, 1, 0, 0]);
    }

    #[test]
    fn check_eos_empty_eos_table() {
        crate::register();
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let tokens = Tensor::from_vec(vec![1u32, 2, 3], (3,), &dev).unwrap();
        let eos_ids = Tensor::from_vec(Vec::<u32>::new(), (0,), &dev).unwrap();
        let done = check_eos(&tokens, &eos_ids).unwrap();
        let got: Vec<u8> = done.to_vec1().unwrap();
        assert_eq!(got, vec![0, 0, 0]);
    }

    #[test]
    fn check_eos_large_batch() {
        crate::register();
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // B = 1000 to exercise multi-block grid (BLOCK_SIZE=256 → 4 blocks).
        // Use an EOS id that's outside `0..B` so the default fill values
        // don't accidentally match it.
        let b = 1000usize;
        let eos_id = 999_999u32;
        let mut data: Vec<u32> = (0..b as u32).collect();
        data[7] = eos_id;
        data[256] = eos_id; // boundary case across blocks
        data[999] = eos_id;
        let tokens = Tensor::from_vec(data, (b,), &dev).unwrap();
        let eos_ids = Tensor::from_vec(vec![eos_id], (1,), &dev).unwrap();
        let done = check_eos(&tokens, &eos_ids).unwrap();
        let got: Vec<u8> = done.to_vec1().unwrap();
        assert_eq!(got.len(), b);
        for (i, &bit) in got.iter().enumerate() {
            let expected = if i == 7 || i == 256 || i == 999 { 1 } else { 0 };
            assert_eq!(bit, expected, "mismatch at index {i}");
        }
    }
}
