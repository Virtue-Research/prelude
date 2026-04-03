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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::kernels::ffi;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
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
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => candle_core::bail!("moe_gemm_wmma only accepts f16/bf16 inputs"),
        };

        let (input_s, _) = input.storage_and_layout();
        let input_s = match &*input_s {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle_core::bail!("input must be a cuda tensor"),
        };

        let (weights_s, _) = weights.storage_and_layout();
        let weights_s = match &*weights_s {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle_core::bail!("weight must be a cuda tensor"),
        };

        let (sti, _) = sorted_token_ids.storage_and_layout();
        let sti = match &*sti {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle_core::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (ei, _) = experts_ids.storage_and_layout();
        let ei = match &*ei {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle_core::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(tw) = topk_weights {
            let (tw_s, _) = tw.storage_and_layout();
            let tw_s = match &*tw_s {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("topk_weights must be a cuda tensor"),
            };
            tw_s.device_ptr(tw_s.stream()).0 as *const f32
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            ffi::moe_gemm_wmma(
                input_s.device_ptr(input_s.stream()).0 as *const c_void,
                weights_s.device_ptr(weights_s.stream()).0 as *const c_void,
                sti.device_ptr(sti.stream()).0 as *const i32,
                ei.device_ptr(ei.stream()).0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void,
                expert_counts.device_ptr(expert_counts.stream()).0 as *mut i32,
                expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32,
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32,
                is_prefill,
                stream,
            );
        }

        use candle_core::op::BackpropOp;
        let output = candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle_core::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(input, weights, topk_weights, sorted_token_ids, experts_ids, topk, is_prefill),
        DType::BF16 => cuda_fwd::<bf16>(input, weights, topk_weights, sorted_token_ids, experts_ids, topk, is_prefill),
        _ => candle_core::bail!("moe_gemm only accepts f16/bf16 inputs"),
    }
}

