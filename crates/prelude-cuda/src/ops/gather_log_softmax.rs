//! Rust launcher for the fused `gather_log_softmax` CUDA kernel.
//!
//! Used by the prompt_logprobs path in the engine to pull out
//! `O(num_tokens)` per-token logprobs from a `[num_tokens, vocab_size]`
//! logits matrix without materialising the full log_softmax.
//!
//! Asymptote matches vLLM's fused Triton kernel
//! (`vllm/v1/worker/gpu/sample/logprob.py::_topk_log_softmax_kernel`):
//! two full-vocab reads, one tiny scalar write per token, zero
//! `[T, V]` temporaries. See
//! `crates/prelude-cuda/src/kernels/kernels_src/logprobs/gather_log_softmax.cu`
//! for the kernel.

use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Device, Result, Shape, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_GATHER_LOG_SOFTMAX, PTX_GATHER_LOG_SOFTMAX};

/// Block size for the reduction. 512 threads = 16 warps, plenty of
/// parallelism for a ~150K-wide vocab while keeping shared memory use
/// tiny (block_reduce_sum/max need `(blockDim.x / 32) = 16` floats).
const BLOCK_SIZE: u32 = 512;

/// Fused gather + log_softmax.
///
/// Computes `out[t] = logits[t, target_ids[t]] - logsumexp(logits[t])`
/// for each `t ∈ [0, num_tokens)` without materialising the full
/// `[num_tokens, vocab_size]` log_softmax matrix on the device.
///
/// ## Inputs
///
/// * `logits`: `[num_tokens, vocab_size]` BF16 or F32. The kernel
///   always reduces in F32 regardless of input dtype — a ~152K-wide
///   logsumexp in BF16 would lose precision badly.
/// * `target_ids`: `[num_tokens]` U32 — the token index to gather at
///   each row. IDs ≥ `vocab_size` are safely clamped to `-inf` so a
///   malformed caller can't OOB-read into device memory.
///
/// ## Output
///
/// * `[num_tokens]` F32 — one logprob per row.
///
/// ## Math
///
/// Standard numerically-stable log_softmax:
///
/// ```text
/// max = max(logits[t])
/// lse = max + log(sum(exp(logits[t] - max)))
/// out[t] = logits[t, target_ids[t]] - lse
/// ```
///
/// Two-pass reduction (max, then exp-sum) — one block per token, one
/// warp-level reduction per pass via `block_reduce_max` /
/// `block_reduce_sum` from `common.cuh`.
pub fn gather_log_softmax(logits: &Tensor, target_ids: &Tensor) -> Result<Tensor> {
    // ── Shape + dtype validation ────────────────────────────────────
    let (num_tokens, vocab_size) = logits.dims2()?;
    if target_ids.dims1()? != num_tokens {
        candle_core::bail!(
            "gather_log_softmax: target_ids length {} != num_tokens {}",
            target_ids.dims1()?,
            num_tokens
        );
    }
    if !matches!(logits.dtype(), DType::BF16 | DType::F32) {
        candle_core::bail!(
            "gather_log_softmax: logits must be BF16 or F32, got {:?}",
            logits.dtype()
        );
    }
    if target_ids.dtype() != DType::U32 {
        candle_core::bail!(
            "gather_log_softmax: target_ids must be U32, got {:?}",
            target_ids.dtype()
        );
    }
    if num_tokens == 0 {
        return Tensor::zeros((0,), DType::F32, logits.device());
    }
    if vocab_size == 0 {
        candle_core::bail!("gather_log_softmax: vocab_size must be > 0");
    }

    // ── Device / storage plumbing ───────────────────────────────────
    let dev: Device = logits.device().clone();
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gather_log_softmax: requires CUDA device"),
    };

    let logits_c = logits.contiguous()?;
    let target_c = target_ids.contiguous()?;

    let (logits_storage, logits_layout) = logits_c.storage_and_layout();
    let (target_storage, target_layout) = target_c.storage_and_layout();

    macro_rules! cuda_of {
        ($s:expr, $name:literal) => {{
            match &*$s {
                candle_core::Storage::Cuda(cuda) => cuda,
                _ => candle_core::bail!("gather_log_softmax: {} not on CUDA", $name),
            }
        }};
    }
    let logits_cuda = cuda_of!(logits_storage, "logits");
    let target_cuda = cuda_of!(target_storage, "target_ids");

    let target_slice = target_cuda
        .as_cuda_slice::<u32>()?
        .slice(target_layout.start_offset()..);

    // ── Output buffer ───────────────────────────────────────────────
    let mut out = unsafe { cuda_dev.alloc::<f32>(num_tokens) }?;

    // ── Launch config ───────────────────────────────────────────────
    // shared mem = (blockDim.x / 32) floats for the block-reduce
    // helpers. 512/32 = 16 floats = 64 bytes. (Also hosts the tiny
    // row_max / row_lse scalar broadcasts which are declared as
    // static `__shared__` in the kernel, not via extern smem.)
    let shared_mem_bytes = (BLOCK_SIZE / 32) * 4;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes,
    };

    let nt_u32 = num_tokens as u32;
    let v_u32 = vocab_size as u32;

    // Dispatch on logits dtype. BF16 and F32 use separate entry points
    // so the BF16 path can do the per-element `__bfloat162float`
    // conversion inline without a runtime branch on every load.
    match logits.dtype() {
        DType::F32 => {
            let logits_slice = logits_cuda
                .as_cuda_slice::<f32>()?
                .slice(logits_layout.start_offset()..);
            let func = cuda_dev.get_or_load_custom_func(
                "gather_log_softmax_f32",
                MOD_GATHER_LOG_SOFTMAX,
                PTX_GATHER_LOG_SOFTMAX,
            )?;
            let mut builder = func.builder();
            builder.arg(&logits_slice);
            builder.arg(&target_slice);
            builder.arg(&mut out);
            builder.arg(&nt_u32);
            builder.arg(&v_u32);
            unsafe { builder.launch(cfg) }.w()?;
        }
        DType::BF16 => {
            let logits_slice = logits_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(logits_layout.start_offset()..);
            let func = cuda_dev.get_or_load_custom_func(
                "gather_log_softmax_bf16",
                MOD_GATHER_LOG_SOFTMAX,
                PTX_GATHER_LOG_SOFTMAX,
            )?;
            let mut builder = func.builder();
            builder.arg(&logits_slice);
            builder.arg(&target_slice);
            builder.arg(&mut out);
            builder.arg(&nt_u32);
            builder.arg(&v_u32);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => unreachable!("validated above"),
    }

    drop(logits_storage);
    drop(target_storage);

    // ── Wrap output back into a candle Tensor ──────────────────────
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, cuda_dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        Shape::from((num_tokens,)),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}
