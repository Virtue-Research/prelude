//! Rust launcher for the fused `gdn_post_conv` CUDA kernel.
//!
//! Collapses the Qwen3.5 DeltaNet post-conv1d prep chain (QKV split +
//! L2 norm on Q/K + softplus-and-exp gate + sigmoid beta) into a
//! single kernel launch per layer. See `kernels_src/gdn/post_conv.cu`
//! for the kernel itself and the `delta_rule_prefill_gdn` path in
//! `models/qwen3_5.rs` for the caller.

use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Device, Result, Shape, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_GDN_POST_CONV, PTX_GDN_POST_CONV};

/// Number of prefill tokens processed by one CUDA block. Must divide
/// evenly into `32` (our warp size * 1) when multiplied by
/// `head_dim / 32`, and keep `BLOCK_T * head_dim ≤ 1024` (max threads
/// per block on Hopper). BLOCK_T=4 gives 4 × 128 = 512 threads per
/// block on Qwen3.5, which maps cleanly to 16 warps.
const BLOCK_T: i32 = 4;

/// L2-norm epsilon — matches `l2_normalize_last_dim`'s Rust reference
/// which does `x / (sqrt(sum_sq) + 1e-12)`.
const L2_EPS: f32 = 1e-12;

/// Fused post-conv1d GDN prep. Returns `(q, k, v, alpha, beta)`.
///
/// ## Inputs
///
/// * `mixed_qkv`: `[L, HK*K*2 + HV*V]` BF16, channel layout
///   `[Q | K | V]`. This is the output of the causal_conv1d kernel.
/// * `a_raw`, `b_raw`: `[L, HV]` BF16 — the raw scalar-per-head gate
///   inputs from the in_proj_a / in_proj_b projections.
/// * `a_log`: `[HV]` F32 — the learned log-decay parameter.
/// * `dt_bias`: `[HV]` F32 — the learned delta-t bias. (Caller must
///   pre-cast from the checkpoint dtype; the kernel does not cast.)
/// * `num_k_heads` (`HK`), `num_v_heads` (`HV`), `head_dim`: shape
///   metadata.
///
/// ## Outputs
///
/// * `q`: `[L, HK, head_dim]` BF16 L2-normalised along the last dim
/// * `k`: `[L, HK, head_dim]` BF16 L2-normalised along the last dim
/// * `v`: `[L, HV, head_dim]` BF16 raw (just a strided copy out of
///   `mixed_qkv`)
/// * `alpha`: `[L, HV]` F32, `= exp(-exp(A_log) * softplus(a + dt_bias))`
///   — the linear-space per-step decay that flashinfer `gdn_prefill`
///   consumes directly as `g`.
/// * `beta`: `[L, HV]` F32, `= sigmoid(b_raw)`.
///
/// All input tensors must be on the same CUDA device. Shapes and
/// dtypes are asserted; mismatches `bail!` instead of silently doing
/// the wrong thing.
#[allow(clippy::too_many_arguments)]
pub fn gdn_post_conv(
    mixed_qkv: &Tensor,
    a_raw: &Tensor,
    b_raw: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    // ── Shape + dtype validation ────────────────────────────────────
    let (l, qkv_dim) = mixed_qkv.dims2()?;
    let expected_qkv_dim = num_k_heads * head_dim * 2 + num_v_heads * head_dim;
    if qkv_dim != expected_qkv_dim {
        candle_core::bail!(
            "gdn_post_conv: mixed_qkv channel dim {} != expected {} (2*HK*D + HV*D)",
            qkv_dim,
            expected_qkv_dim
        );
    }
    let (la, hva) = a_raw.dims2()?;
    let (lb, hvb) = b_raw.dims2()?;
    if la != l || lb != l || hva != num_v_heads || hvb != num_v_heads {
        candle_core::bail!(
            "gdn_post_conv: a_raw/b_raw shape mismatch — expected [{l}, {num_v_heads}]"
        );
    }
    if a_log.dim(0)? != num_v_heads || dt_bias.dim(0)? != num_v_heads {
        candle_core::bail!("gdn_post_conv: a_log/dt_bias must have shape [{num_v_heads}]");
    }
    if mixed_qkv.dtype() != DType::BF16
        || a_raw.dtype() != DType::BF16
        || b_raw.dtype() != DType::BF16
    {
        candle_core::bail!("gdn_post_conv: mixed_qkv / a_raw / b_raw must be BF16");
    }
    if a_log.dtype() != DType::F32 || dt_bias.dtype() != DType::F32 {
        candle_core::bail!("gdn_post_conv: a_log / dt_bias must be F32");
    }
    // head_dim must be a multiple of 32 (warp size) so the in-kernel
    // per-token warp reductions cover the full dim cleanly, and must
    // satisfy `BLOCK_T * head_dim ≤ 1024` (SM90 max threads/block).
    if head_dim % 32 != 0 || head_dim == 0 {
        candle_core::bail!(
            "gdn_post_conv: head_dim must be a nonzero multiple of 32 (got {head_dim})"
        );
    }
    let block_t = BLOCK_T as usize;
    if block_t * head_dim > 1024 {
        candle_core::bail!(
            "gdn_post_conv: BLOCK_T * head_dim = {} exceeds 1024 threads/block",
            block_t * head_dim
        );
    }

    // ── Device / storage plumbing ───────────────────────────────────
    let dev: Device = mixed_qkv.device().clone();
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gdn_post_conv: requires CUDA device"),
    };

    // Inputs must be contiguous. The caller already feeds us a
    // contiguous mixed_qkv out of causal_conv1d; a_raw/b_raw come from
    // the projection heads which are contiguous; A_log/dt_bias are
    // layer params.
    let mixed_c = mixed_qkv.contiguous()?;
    let a_c = a_raw.contiguous()?;
    let b_c = b_raw.contiguous()?;
    let a_log_c = a_log.contiguous()?;
    let dt_bias_c = dt_bias.contiguous()?;

    let (mixed_storage, mixed_layout) = mixed_c.storage_and_layout();
    let (a_storage, a_layout) = a_c.storage_and_layout();
    let (b_storage, b_layout) = b_c.storage_and_layout();
    let (alog_storage, alog_layout) = a_log_c.storage_and_layout();
    let (dt_storage, dt_layout) = dt_bias_c.storage_and_layout();

    macro_rules! cuda_of {
        ($s:expr, $name:literal) => {{
            match &*$s {
                candle_core::Storage::Cuda(cuda) => cuda,
                _ => candle_core::bail!("gdn_post_conv: {} not on CUDA", $name),
            }
        }};
    }
    let mixed_cuda = cuda_of!(mixed_storage, "mixed_qkv");
    let a_cuda = cuda_of!(a_storage, "a_raw");
    let b_cuda = cuda_of!(b_storage, "b_raw");
    let alog_cuda = cuda_of!(alog_storage, "A_log");
    let dt_cuda = cuda_of!(dt_storage, "dt_bias");

    let mixed_slice = mixed_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(mixed_layout.start_offset()..);
    let a_slice = a_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(a_layout.start_offset()..);
    let b_slice = b_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(b_layout.start_offset()..);
    let alog_slice = alog_cuda
        .as_cuda_slice::<f32>()?
        .slice(alog_layout.start_offset()..);
    let dt_slice = dt_cuda
        .as_cuda_slice::<f32>()?
        .slice(dt_layout.start_offset()..);

    // ── Output allocations ─────────────────────────────────────────
    let qk_elems = l * num_k_heads * head_dim;
    let v_elems = l * num_v_heads * head_dim;
    let ab_elems = l * num_v_heads;
    let mut q_out = unsafe { cuda_dev.alloc::<half::bf16>(qk_elems) }?;
    let mut k_out = unsafe { cuda_dev.alloc::<half::bf16>(qk_elems) }?;
    let mut v_out = unsafe { cuda_dev.alloc::<half::bf16>(v_elems) }?;
    let mut alpha_out = unsafe { cuda_dev.alloc::<f32>(ab_elems) }?;
    let mut beta_out = unsafe { cuda_dev.alloc::<f32>(ab_elems) }?;

    // ── Launch config ──────────────────────────────────────────────
    let grid_x = l.div_ceil(block_t) as u32;
    let grid_y = (num_k_heads + num_v_heads) as u32;
    let block_dim = (block_t * head_dim) as u32;
    // shared mem: `block_t * (warps_per_token * 2 + 2)` floats
    //             = partial sums + per-token (scale_q, scale_k)
    let warps_per_token = (head_dim / 32) as u32;
    let shared_mem_floats = block_t as u32 * (warps_per_token * 2 + 2);
    let shared_mem_bytes = shared_mem_floats * 4;

    let func = cuda_dev.get_or_load_custom_func(
        "gdn_post_conv_bf16",
        MOD_GDN_POST_CONV,
        PTX_GDN_POST_CONV,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes,
    };

    let l_i = l as i32;
    let hk_i = num_k_heads as i32;
    let hv_i = num_v_heads as i32;
    let hd_i = head_dim as i32;
    let block_t_i = BLOCK_T;
    let eps = L2_EPS;

    let mut builder = func.builder();
    builder.arg(&mixed_slice);
    builder.arg(&a_slice);
    builder.arg(&b_slice);
    builder.arg(&alog_slice);
    builder.arg(&dt_slice);
    builder.arg(&mut q_out);
    builder.arg(&mut k_out);
    builder.arg(&mut v_out);
    builder.arg(&mut alpha_out);
    builder.arg(&mut beta_out);
    builder.arg(&l_i);
    builder.arg(&hk_i);
    builder.arg(&hv_i);
    builder.arg(&hd_i);
    builder.arg(&block_t_i);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }.w()?;

    // Drop storage guards now that the launch has captured pointers.
    drop(mixed_storage);
    drop(a_storage);
    drop(b_storage);
    drop(alog_storage);
    drop(dt_storage);

    // ── Wrap outputs back into candle Tensors ──────────────────────
    let q = tensor_from_bf16(q_out, &dev, (l, num_k_heads, head_dim))?;
    let k = tensor_from_bf16(k_out, &dev, (l, num_k_heads, head_dim))?;
    let v = tensor_from_bf16(v_out, &dev, (l, num_v_heads, head_dim))?;
    let alpha = tensor_from_f32(alpha_out, &dev, (l, num_v_heads))?;
    let beta = tensor_from_f32(beta_out, &dev, (l, num_v_heads))?;
    Ok((q, k, v, alpha, beta))
}

fn tensor_from_bf16(
    slice: cudarc::driver::CudaSlice<half::bf16>,
    dev: &Device,
    shape: impl Into<Shape>,
) -> Result<Tensor> {
    let cuda_dev = match dev {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gdn_post_conv: output not on CUDA"),
    };
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, cuda_dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape.into(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn tensor_from_f32(
    slice: cudarc::driver::CudaSlice<f32>,
    dev: &Device,
    shape: impl Into<Shape>,
) -> Result<Tensor> {
    let cuda_dev = match dev {
        Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gdn_post_conv: output not on CUDA"),
    };
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, cuda_dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape.into(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}
