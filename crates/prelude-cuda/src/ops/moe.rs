use crate::device::{self as cb, CuResultExt, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::{MOD_MOE_ROUTING, PTX_MOE_ROUTING};
use prelude_core::tensor::{bail, DType, Result, Tensor};

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
    let (rl_storage, rl_layout) = cb::storage_and_layout(&router_logits);
    let rl_cuda = cb::as_cuda(&rl_storage, "fused_moe_routing")?;
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

    let stream = rl_cuda.stream.clone();
    let rl_slice = rl_cuda
        .as_slice::<half::bf16>()?
        .slice(rl_layout.start_offset()..);

    let topk_weights = unsafe { stream.alloc::<f32>(num_tokens * topk) }.ce()?;
    let topk_ids = unsafe { stream.alloc::<u32>(num_tokens * topk) }.ce()?;
    let sorted_expert_ids = unsafe { stream.alloc::<u32>(num_tokens * topk) }.ce()?;
    let sorted_token_ids = unsafe { stream.alloc::<u32>(num_tokens * topk) }.ce()?;

    // Shared memory: num_experts floats (softmax vals) + num_experts u32 (indices) + 16 floats (warp reduce)
    let shared_mem = (num_experts * 4 + num_experts * 4 + 16 * 4) as u32;
    // Block size: enough threads to cover num_experts (e.g., 128 threads for 128 experts)
    let block_size = if num_experts <= 128 { 128u32 } else { 256u32 };

    let func =
        crate::device::get_or_load_func(rl_cuda.device(), "fused_moe_routing_bf16", MOD_MOE_ROUTING, PTX_MOE_ROUTING)?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = stream.launch_builder(&func);
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
    unsafe { builder.launch(cfg) }.ce()?;

    drop(rl_storage);

    // Wrap outputs as tensors
    let tw_tensor = cb::tensor_from_cuda(topk_weights, stream.clone(), (num_tokens, topk));
    let ti_tensor = cb::tensor_from_cuda(topk_ids, stream.clone(), (num_tokens * topk,));
    let se_tensor = cb::tensor_from_cuda(sorted_expert_ids, stream.clone(), (num_tokens * topk,));
    let st_tensor = cb::tensor_from_cuda(sorted_token_ids, stream.clone(), (num_tokens * topk,));

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

    fn cuda_fwd<T: GpuDType + DeviceRepr>(
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
        let stream = cb::tensor_stream(input)?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => bail!("moe_gemm_wmma only accepts f16/bf16 inputs"),
        };

        let (input_s, _) = cb::storage_and_layout(&input);
        let input_s = cb::as_cuda(&input_s, "input")?.as_slice::<T>()?;

        let (weights_s, _) = cb::storage_and_layout(&weights);
        let weights_s = cb::as_cuda(&weights_s, "weight")?.as_slice::<T>()?;

        let (sti, _) = cb::storage_and_layout(&sorted_token_ids);
        let sti = cb::as_cuda(&sti, "sorted_token_ids")?.as_slice::<u32>()?;

        let (ei, _) = cb::storage_and_layout(&experts_ids);
        let ei = cb::as_cuda(&ei, "experts_ids")?.as_slice::<u32>()?;

        let topk_weights_ptr = if let Some(tw) = topk_weights {
            let (tw_s, _) = cb::storage_and_layout(&tw);
            let tw_s = cb::as_cuda(&tw_s, "topk_weights")?.as_slice::<f32>()?;
            tw_s.device_ptr(tw_s.stream()).0 as *const f32
        } else {
            std::ptr::null()
        };

        let output = unsafe { stream.alloc::<T>(size_m * size_n) }.ce()?;
        let expert_counts = unsafe { stream.alloc::<u32>(num_experts) }.ce()?;
        let expert_offsets = unsafe { stream.alloc::<u32>(num_experts + 1) }.ce()?;

        let cu_stream = stream.cu_stream() as i64;
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
                cu_stream,
            );
        }

        let output = cb::tensor_from_cuda(output, stream.clone(), (size_m, size_n));
        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(input, weights, topk_weights, sorted_token_ids, experts_ids, topk, is_prefill),
        DType::BF16 => cuda_fwd::<bf16>(input, weights, topk_weights, sorted_token_ids, experts_ids, topk, is_prefill),
        _ => bail!("moe_gemm only accepts f16/bf16 inputs"),
    }
}
