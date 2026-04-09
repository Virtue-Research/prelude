use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::{DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use prelude_core::tensor::{bail, DType, Result, Tensor};

use crate::{MOD_MOE_ROUTING, PTX_MOE_ROUTING};

/// Helper: extract candle CUDA storage or bail.
fn as_candle_cuda<'a>(
    storage: &'a std::sync::RwLockReadGuard<'a, candle_core::Storage>,
    ctx: &str,
) -> Result<&'a candle_core::CudaStorage> {
    match &**storage {
        candle_core::Storage::Cuda(s) => Ok(s),
        _ => candle_core::bail!("{ctx}: requires CUDA"),
    }
}

/// Fused MoE routing: softmax + top-k selection + weight normalization + sort.
/// Replaces ~8 separate kernel launches with a single CUDA kernel.
///
/// Input: `router_logits` [num_tokens, num_experts] BF16 (output of gate linear)
/// Returns: (topk_weights, topk_ids, sorted_expert_ids, sorted_token_ids)
///   - topk_weights: [num_tokens, topk] F32 (normalized)
///   - topk_ids: [num_tokens, topk] U32 (expert IDs)
///   - sorted_expert_ids: [num_tokens * topk] U32 (sorted by expert ID)
///   - sorted_token_ids: [num_tokens * topk] U32 (indices for moe_gemm)
pub fn fused_moe_routing(
    router_logits: &Tensor, // [num_tokens, num_experts] BF16
    topk: usize,
    norm_topk_prob: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (rl_storage, rl_layout) = router_logits.storage_and_layout();
    let rl_cuda = as_candle_cuda(&rl_storage, "fused_moe_routing")?;
    if rl_cuda.dtype() != DType::BF16 {
        bail!("fused_moe_routing: requires BF16 input");
    }

    let shape = rl_layout.shape();
    let dims = shape.dims();
    let (num_tokens, num_experts) = if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        bail!("fused_moe_routing: expected 2D input, got {:?}", dims);
    };

    let dev = rl_cuda.device().clone();
    let rl_slice = rl_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(rl_layout.start_offset()..);

    let topk_weights = unsafe { dev.alloc::<f32>(num_tokens * topk) }?;
    let topk_ids = unsafe { dev.alloc::<u32>(num_tokens * topk) }?;
    let sorted_expert_ids = unsafe { dev.alloc::<u32>(num_tokens * topk) }?;
    let sorted_token_ids = unsafe { dev.alloc::<u32>(num_tokens * topk) }?;

    // Shared memory: num_experts floats (softmax vals) + num_experts u32 (indices) + 16 floats (warp reduce)
    let shared_mem = (num_experts * 4 + num_experts * 4 + 16 * 4) as u32;
    // Block size: enough threads to cover num_experts (e.g., 128 threads for 128 experts)
    let block_size = if num_experts <= 128 { 128u32 } else { 256u32 };

    let func =
        dev.get_or_load_custom_func("fused_moe_routing_bf16", MOD_MOE_ROUTING, PTX_MOE_ROUTING)?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = func.builder();
    builder.arg(&rl_slice);
    builder.arg(&topk_weights);
    builder.arg(&topk_ids);
    builder.arg(&sorted_expert_ids);
    builder.arg(&sorted_token_ids);
    let num_tokens_val = num_tokens as u32;
    let num_experts_val = num_experts as u32;
    let topk_val = topk as u32;
    let norm_val = norm_topk_prob;
    builder.arg(&num_tokens_val);
    builder.arg(&num_experts_val);
    builder.arg(&topk_val);
    builder.arg(&norm_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(rl_storage);

    // Wrap outputs as tensors
    let tw_storage = candle_core::CudaStorage::wrap_cuda_slice(topk_weights, dev.clone());
    let tw_tensor = Tensor::from_storage(candle_core::Storage::Cuda(tw_storage), (num_tokens, topk), candle_core::op::BackpropOp::none(), false);
    let ti_storage = candle_core::CudaStorage::wrap_cuda_slice(topk_ids, dev.clone());
    let ti_tensor = Tensor::from_storage(candle_core::Storage::Cuda(ti_storage), (num_tokens * topk,), candle_core::op::BackpropOp::none(), false);
    let se_storage = candle_core::CudaStorage::wrap_cuda_slice(sorted_expert_ids, dev.clone());
    let se_tensor = Tensor::from_storage(candle_core::Storage::Cuda(se_storage), (num_tokens * topk,), candle_core::op::BackpropOp::none(), false);
    let st_storage = candle_core::CudaStorage::wrap_cuda_slice(sorted_token_ids, dev.clone());
    let st_tensor = Tensor::from_storage(candle_core::Storage::Cuda(st_storage), (num_tokens * topk,), candle_core::op::BackpropOp::none(), false);

    Ok((tw_tensor, ti_tensor, se_tensor, st_tensor))
}

/// WMMA-based MoE GEMM kernel.
///
/// Dispatches tokens to experts using sorted indices and performs
/// batched matrix multiply with optional topk_weights scaling.
pub fn moe_gemm_wmma(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use crate::device::GpuDType;
    use crate::moe_ffi as ffi;
    use half::{bf16, f16};

    fn cuda_fwd<T: GpuDType + DeviceRepr + candle_core::cuda_backend::CudaDType>(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );

        let (input_g, input_l) = input.storage_and_layout();
        let input_cuda = match &*input_g {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("moe_gemm: input requires CUDA"),
        };
        let input_s = input_cuda.as_cuda_slice::<T>()?.slice(input_l.start_offset()..);

        let (weights_g, weights_l) = weights.storage_and_layout();
        let weights_cuda = match &*weights_g {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("moe_gemm: weights requires CUDA"),
        };
        let weights_s = weights_cuda.as_cuda_slice::<T>()?.slice(weights_l.start_offset()..);

        let (sti_g, sti_l) = sorted_token_ids.storage_and_layout();
        let sti_cuda = match &*sti_g {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("moe_gemm: sorted_token_ids requires CUDA"),
        };
        let sti = sti_cuda.as_cuda_slice::<u32>()?.slice(sti_l.start_offset()..);

        let (ei_g, ei_l) = experts_ids.storage_and_layout();
        let ei_cuda = match &*ei_g {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("moe_gemm: experts_ids requires CUDA"),
        };
        let ei = ei_cuda.as_cuda_slice::<u32>()?.slice(ei_l.start_offset()..);

        let dev = input_cuda.device().clone();
        let stream = dev.cuda_stream();

        let topk_weights_ptr = if let Some(tw) = topk_weights {
            let (tw_g, tw_l) = tw.storage_and_layout();
            let tw_cuda = match &*tw_g {
                candle_core::Storage::Cuda(s) => s,
                _ => candle_core::bail!("moe_gemm: topk_weights requires CUDA"),
            };
            let tw_s = tw_cuda.as_cuda_slice::<f32>()?.slice(tw_l.start_offset()..);
            tw_s.device_ptr(&stream).0 as *const f32
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let cu_stream = stream.cu_stream() as i64;
        use core::ffi::c_void;

        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => candle_core::bail!("moe_gemm_wmma only accepts f16/bf16 inputs"),
        };

        unsafe {
            ffi::moe_gemm_wmma(
                input_s.device_ptr(&stream).0 as *const c_void,
                weights_s.device_ptr(&stream).0 as *const c_void,
                sti.device_ptr(&stream).0 as *const i32,
                ei.device_ptr(&stream).0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(&stream).0 as *mut c_void,
                expert_counts.device_ptr(&stream).0 as *mut i32,
                expert_offsets.device_ptr(&stream).0 as *mut i32,
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32,
                is_prefill,
                cu_stream,
            );
        }

        drop(input_g);
        drop(weights_g);
        drop(sti_g);
        drop(ei_g);

        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            (size_m, size_n),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(input, weights, topk_weights, sorted_token_ids, experts_ids, topk, is_prefill),
        DType::BF16 => cuda_fwd::<bf16>(input, weights, topk_weights, sorted_token_ids, experts_ids, topk, is_prefill),
        _ => bail!("moe_gemm only accepts f16/bf16 inputs"),
    }
}
