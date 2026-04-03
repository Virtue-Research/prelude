use candle_core::backend::BackendStorage;
use crate::{MOD_MOE_ROUTING, PTX_MOE_ROUTING};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};

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
    let rl_cuda = match &*rl_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_moe_routing: requires CUDA"),
    };
    if rl_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_moe_routing: requires BF16 input");
    }

    let shape = rl_layout.shape();
    let dims = shape.dims();
    let (num_tokens, num_experts) = if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        candle_core::bail!("fused_moe_routing: expected 2D input, got {:?}", dims);
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
    use candle_core::op::BackpropOp;

    let tw_storage = candle_core::CudaStorage::wrap_cuda_slice(topk_weights, dev.clone());
    let tw_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(tw_storage),
        (num_tokens, topk),
        BackpropOp::none(),
        false,
    );

    let ti_storage = candle_core::CudaStorage::wrap_cuda_slice(topk_ids, dev.clone());
    let ti_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(ti_storage),
        (num_tokens * topk,),
        BackpropOp::none(),
        false,
    );

    let se_storage = candle_core::CudaStorage::wrap_cuda_slice(sorted_expert_ids, dev.clone());
    let se_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(se_storage),
        (num_tokens * topk,),
        BackpropOp::none(),
        false,
    );

    let st_storage = candle_core::CudaStorage::wrap_cuda_slice(sorted_token_ids, dev.clone());
    let st_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(st_storage),
        (num_tokens * topk,),
        BackpropOp::none(),
        false,
    );

    Ok((tw_tensor, ti_tensor, se_tensor, st_tensor))
}

