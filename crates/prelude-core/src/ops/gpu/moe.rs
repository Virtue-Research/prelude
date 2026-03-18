use candle_core::backend::BackendStorage;
use super::{MOD_MOE_DOWN, MOD_MOE_GATEUP, MOD_MOE_ROUTING, PTX_MOE_DOWN, PTX_MOE_GATEUP,
            PTX_MOE_ROUTING};
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

// ── Fused MoE Decode Kernels ─────────────────────────────────────────

/// Fused MoE gate+up projection with SiLU activation for single-token decode.
/// Replaces: gate moe_gemm + up moe_gemm + fused_silu_mul (5 kernel calls -> 1).
///
/// - `input`: `[hidden_size]` or `[1, hidden_size]` BF16
/// - `gate_w`: `[num_experts, inter_size, hidden_size]` BF16
/// - `up_w`: `[num_experts, inter_size, hidden_size]` BF16
/// - `expert_ids`: `[num_active]` U32 (expert indices from routing)
///
/// Returns: `[num_active, inter_size]` BF16
pub fn moe_decode_gateup_silu(
    input: &Tensor,
    gate_w: &Tensor,
    up_w: &Tensor,
    expert_ids: &Tensor,
) -> Result<Tensor> {
    let (inp_storage, inp_layout) = input.storage_and_layout();
    let (gw_storage, gw_layout) = gate_w.storage_and_layout();
    let (uw_storage, uw_layout) = up_w.storage_and_layout();
    let (eid_storage, eid_layout) = expert_ids.storage_and_layout();

    let inp_cuda = match &*inp_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_gateup_silu: input requires CUDA"),
    };
    let gw_cuda = match &*gw_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_gateup_silu: gate_w requires CUDA"),
    };
    let uw_cuda = match &*uw_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_gateup_silu: up_w requires CUDA"),
    };
    let eid_cuda = match &*eid_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_gateup_silu: expert_ids requires CUDA"),
    };

    if inp_cuda.dtype() != DType::BF16 || gw_cuda.dtype() != DType::BF16 {
        candle_core::bail!("moe_decode_gateup_silu: requires BF16");
    }

    let dev = inp_cuda.device().clone();

    // Extract dimensions
    let hidden_size = inp_layout.shape().elem_count(); // flat [hidden_size]
    let gw_dims = gw_layout.shape().dims();
    let inter_size = gw_dims[1]; // gate_w is [num_experts, inter_size, hidden_size]
    let num_active = eid_layout.shape().elem_count();

    let inp_slice = inp_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(inp_layout.start_offset()..);
    let gw_slice = gw_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(gw_layout.start_offset()..);
    let uw_slice = uw_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(uw_layout.start_offset()..);
    let eid_slice = eid_cuda
        .as_cuda_slice::<u32>()?
        .slice(eid_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(num_active * inter_size) }?;

    // Grid: (ceil(inter_size/4), num_active), Block: 128
    let grid_x = (inter_size as u32 + 3) / 4;
    let grid_y = num_active as u32;
    let shared_mem = (hidden_size * 2) as u32; // BF16 input vector in shared memory

    let func = dev.get_or_load_custom_func(
        "moe_decode_gateup_silu_bf16",
        MOD_MOE_GATEUP,
        PTX_MOE_GATEUP,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = func.builder();
    builder.arg(&inp_slice);
    builder.arg(&gw_slice);
    builder.arg(&uw_slice);
    builder.arg(&eid_slice);
    builder.arg(&out);
    let hidden_size_val = hidden_size as u32;
    let inter_size_val = inter_size as u32;
    let num_active_val = num_active as u32;
    builder.arg(&hidden_size_val);
    builder.arg(&inter_size_val);
    builder.arg(&num_active_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(inp_storage);
    drop(gw_storage);
    drop(uw_storage);
    drop(eid_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (num_active, inter_size),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(out_tensor)
}

/// Fused MoE down projection with weighted expert reduction for single-token decode.
/// Replaces: down moe_gemm + reshape + weighted sum (3 kernel calls -> 1).
///
/// - `intermediate`: `[num_active, inter_size]` BF16
/// - `down_w`: `[num_experts, hidden_size, inter_size]` BF16
/// - `expert_ids`: `[num_active]` U32
/// - `topk_weights`: `[1, num_active]` or `[num_active]` F32
///
/// Returns: `[1, hidden_size]` BF16 (weighted sum across experts)
pub fn moe_decode_down_reduce(
    intermediate: &Tensor,
    down_w: &Tensor,
    expert_ids: &Tensor,
    topk_weights: &Tensor,
) -> Result<Tensor> {
    let (int_storage, int_layout) = intermediate.storage_and_layout();
    let (dw_storage, dw_layout) = down_w.storage_and_layout();
    let (eid_storage, eid_layout) = expert_ids.storage_and_layout();
    let (tw_storage, tw_layout) = topk_weights.storage_and_layout();

    let int_cuda = match &*int_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_down_reduce: intermediate requires CUDA"),
    };
    let dw_cuda = match &*dw_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_down_reduce: down_w requires CUDA"),
    };
    let eid_cuda = match &*eid_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_down_reduce: expert_ids requires CUDA"),
    };
    let tw_cuda = match &*tw_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("moe_decode_down_reduce: topk_weights requires CUDA"),
    };

    if int_cuda.dtype() != DType::BF16 || dw_cuda.dtype() != DType::BF16 {
        candle_core::bail!("moe_decode_down_reduce: requires BF16 for intermediate/down_w");
    }
    if tw_cuda.dtype() != DType::F32 {
        candle_core::bail!("moe_decode_down_reduce: topk_weights must be F32");
    }

    let dev = int_cuda.device().clone();

    // Extract dimensions
    let dw_dims = dw_layout.shape().dims();
    let hidden_size = dw_dims[1]; // down_w is [num_experts, hidden_size, inter_size]
    let inter_size = dw_dims[2];
    let num_active = eid_layout.shape().elem_count();

    let int_slice = int_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(int_layout.start_offset()..);
    let dw_slice = dw_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(dw_layout.start_offset()..);
    let eid_slice = eid_cuda
        .as_cuda_slice::<u32>()?
        .slice(eid_layout.start_offset()..);
    let tw_slice = tw_cuda
        .as_cuda_slice::<f32>()?
        .slice(tw_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(hidden_size) }?;

    // Grid: (ceil(hidden_size/4), 1), Block: 128
    let grid_x = (hidden_size as u32 + 3) / 4;

    let func =
        dev.get_or_load_custom_func("moe_decode_down_reduce_bf16", MOD_MOE_DOWN, PTX_MOE_DOWN)?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&int_slice);
    builder.arg(&dw_slice);
    builder.arg(&eid_slice);
    builder.arg(&tw_slice);
    builder.arg(&out);
    let hidden_size_val = hidden_size as u32;
    let inter_size_val = inter_size as u32;
    let num_active_val = num_active as u32;
    builder.arg(&hidden_size_val);
    builder.arg(&inter_size_val);
    builder.arg(&num_active_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(int_storage);
    drop(dw_storage);
    drop(eid_storage);
    drop(tw_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (1, hidden_size),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(out_tensor)
}
