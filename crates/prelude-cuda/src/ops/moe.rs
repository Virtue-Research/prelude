use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::{DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use prelude_core::tensor::{bail, DType, Result, Tensor};

use crate::moe_ffi as ffi;
use crate::{MOD_MOE_ROUTING, PTX_MOE_ROUTING};

use std::sync::OnceLock;

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

        let cu_stream = stream.cu_stream() as i64;
        use core::ffi::c_void;

        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => candle_core::bail!("moe_gemm_wmma only accepts f16/bf16 inputs"),
        };

        // GEMV fast path for very-small-M MoE (single-token decode).
        //
        // The WMMA kernel's M-tile is 32 and launches `num_experts *
        // ceil(size_n / 32)` blocks independent of size_m — so its
        // launch overhead is fixed while Tensor Cores churn. The GEMV
        // kernel launches `size_m * size_n` blocks, one dot product
        // each: fine when size_m ≤ ~8, but past that the block-launch
        // cost of `size_m * size_n` blocks (≈ 128 * 2048 = 256K for
        // batch=32 × top=4) outweighs the Tensor Core slack.
        //
        // Empirically for this model (Qwen3-MoE, moe_inter=768) GEMV
        // only beats WMMA at batch=1–2 (size_m = 4–8). For larger
        // batched decode we stay on WMMA with is_prefill=false.
        // Override via PRELUDE_MOE_GEMV_THRESHOLD to tune.
        let gemv_threshold: usize = std::env::var("PRELUDE_MOE_GEMV_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);
        let use_gemv = size_m <= gemv_threshold;
        // Always take the no-thrust offsets path: `calculate_expert_offsets_light`
        // works for num_experts ≤ 1024 and is CUDA-graph-capturable. The thrust
        // variant is faster only when num_experts is much larger than that,
        // which no current MoE uses (Qwen3-MoE: 64, DeepSeek v2: 160).
        let is_prefill_for_offsets = false;
        let _ = is_prefill;

        if use_gemv {
            unsafe {
                ffi::moe_gemv(
                    input_s.device_ptr(&stream).0 as *const c_void,
                    weights_s.device_ptr(&stream).0 as *const c_void,
                    sti.device_ptr(&stream).0 as *const i32,
                    ei.device_ptr(&stream).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(&stream).0 as *mut c_void,
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    data_type as i32,
                    cu_stream,
                );
            }
        } else {
            let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
            let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;
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
                    is_prefill_for_offsets,
                    cu_stream,
                );
            }
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

/// GPU-accelerated sort of expert assignments using thrust::sort_by_key.
/// Input:  expert_ids [n] U32 (flat, unsorted)
/// Output: (sorted_expert_ids [n] U32, sorted_token_ids [n] U32)
pub fn moe_sort_experts_gpu(expert_ids: &Tensor) -> Result<(Tensor, Tensor)> {
    let n = expert_ids.elem_count();
    let (storage, layout) = expert_ids.storage_and_layout();
    let cuda = as_candle_cuda(&storage, "moe_sort_experts_gpu")?;
    if cuda.dtype() != DType::U32 {
        bail!("moe_sort_experts_gpu: requires U32 input, got {:?}", cuda.dtype());
    }
    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();
    let src = cuda.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);

    let sorted_experts = unsafe { dev.alloc::<u32>(n) }?;
    let sorted_tokens = unsafe { dev.alloc::<u32>(n) }?;

    let cu_stream = stream.cu_stream() as i64;
    unsafe {
        ffi::moe_sort_expert_assignments(
            src.device_ptr(&stream).0 as *const u32,
            n as i32,
            sorted_experts.device_ptr(&stream).0 as *mut u32,
            sorted_tokens.device_ptr(&stream).0 as *mut u32,
            cu_stream,
        );
    }

    drop(storage);

    let se_storage = candle_core::CudaStorage::wrap_cuda_slice(sorted_experts, dev.clone());
    let st_storage = candle_core::CudaStorage::wrap_cuda_slice(sorted_tokens, dev);
    let se = Tensor::from_storage(candle_core::Storage::Cuda(se_storage), (n,), candle_core::op::BackpropOp::none(), false);
    let st = Tensor::from_storage(candle_core::Storage::Cuda(st_storage), (n,), candle_core::op::BackpropOp::none(), false);
    Ok((se, st))
}

/// In-place swap gate/up halves in expert weights for CUTLASS Swiglu.
/// Converts [E, gate|up, H] → [E, up|gate, H] using 2MB temp on GPU.
/// Call once at model load time.
pub fn swap_gate_up_inplace(w1: &Tensor, inter: usize) -> Result<()> {
    let (storage, layout) = w1.storage_and_layout();
    let cuda = as_candle_cuda(&storage, "swap_gate_up")?;
    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();
    let slice = cuda.as_cuda_slice::<half::bf16>()?;
    let base = slice.device_ptr(&stream).0;
    let offset = (layout.start_offset() * 2) as u64; // bf16 = 2 bytes
    let data_ptr = (base + offset) as *mut std::ffi::c_void;
    let dims = w1.dims();
    let num_experts = dims[0] as i32;
    let hidden = dims[2] as i32;
    let cu_stream = unsafe { stream.cu_stream() } as i64;
    unsafe {
        ffi::moe_swap_gate_up_inplace(data_ptr, num_experts, inter as i32, hidden, cu_stream);
    }
    drop(storage);
    Ok(())
}

// ── CUTLASS Fused MoE forward ─────────────────────────────────────

/// Get or create the CUTLASS fused MoE runner singleton.
fn get_cutlass_moe_runner() -> Option<&'static flashinfer::moe::FusedMoeRunner> {
    static RUNNER: OnceLock<Option<flashinfer::moe::FusedMoeRunner>> = OnceLock::new();
    RUNNER.get_or_init(|| {
        match flashinfer::moe::FusedMoeRunner::new() {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::warn!("CUTLASS fused MoE unavailable: {e}");
                None
            }
        }
    }).as_ref()
}

/// CUTLASS fused MoE forward: converts candle tensors to DLTensor and calls the runner.
pub fn cutlass_fused_moe_forward(
    input: &Tensor,            // [n_tokens, hidden] BF16
    experts_per_tok: &Tensor,  // [n_tokens, topk] U32
    topk_weights: &Tensor,     // [n_tokens, topk] F32
    w1: &Tensor,               // [num_experts, 2*inter, hidden] BF16
    w2: &Tensor,               // [num_experts, hidden, inter] BF16
) -> Result<Tensor> {
    use flashinfer::types::*;

    let runner = get_cutlass_moe_runner()
        .ok_or_else(|| candle_core::Error::Msg("CUTLASS fused MoE runner not available".into()))?;

    let (n_tokens, hidden) = input.dims2()?;
    let device = input.device();

    // Allocate output tensor
    let output = Tensor::zeros((n_tokens, hidden), input.dtype(), device)?;

    // Ensure contiguous layout
    let input = input.contiguous()?;
    let experts_i32 = experts_per_tok.contiguous()?;
    let topk_weights = topk_weights.contiguous()?;
    let w2 = w2.contiguous()?;

    // w1 must be in [up|gate] order (swapped at load time via swap_gate_up_inplace).
    let w1 = w1.contiguous()?;

    // Build DLTensors from candle tensors.
    // Extracts the raw CUDA device pointer via candle's CudaStorage.
    // Extract raw CUDA device pointer from a candle tensor.
    // GPU device addresses are stable — the pointer remains valid after
    // dropping the storage guard.
    fn tensor_to_dl(t: &Tensor, shapes: &[i64], dt: DLDataType) -> Result<DLTensor> {
        let (storage, layout) = t.storage_and_layout();
        let cuda = as_candle_cuda(&storage, "cutlass_fused_moe")?;
        let dev = cuda.device().clone();
        let stream = dev.cuda_stream();
        // CudaSlice::device_ptr returns the base of the RAW allocation.
        // layout.start_offset() is the ELEMENT offset for this tensor's view.
        let (base_ptr, elem_offset) = match t.dtype() {
            DType::BF16 => {
                let s = cuda.as_cuda_slice::<half::bf16>()?;
                (s.device_ptr(&stream).0, layout.start_offset())
            }
            DType::F32 => {
                let s = cuda.as_cuda_slice::<f32>()?;
                (s.device_ptr(&stream).0, layout.start_offset())
            }
            DType::U32 => {
                let s = cuda.as_cuda_slice::<u32>()?;
                (s.device_ptr(&stream).0, layout.start_offset())
            }
            dt => bail!("cutlass_fused_moe: unsupported dtype {dt:?}"),
        };
        let data_ptr = base_ptr + (elem_offset * t.dtype().size_in_bytes()) as u64;
        Ok(DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: DLDevice { device_type: KDLCUDA, device_id: 0 },
            ndim: shapes.len() as i32,
            dtype: dt,
            shape: shapes.as_ptr(),
            strides: std::ptr::null(),
            byte_offset: 0,
        })
    }

    let bf16_dt = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
    let f32_dt = DLDataType { code: KDLFLOAT, bits: 32, lanes: 1 };
    let i32_dt = DLDataType { code: KDLINT, bits: 32, lanes: 1 };

    let to_shape = |t: &Tensor| -> Vec<i64> { t.dims().iter().map(|&d| d as i64).collect() };

    let out_shape = to_shape(&output);
    let in_shape = to_shape(&input);
    let ept_shape = to_shape(&experts_i32);
    let tw_shape = to_shape(&topk_weights);
    let w1_shape = to_shape(&w1);
    let w2_shape = to_shape(&w2);

    let dl_out = tensor_to_dl(&output, &out_shape, bf16_dt)?;
    let dl_in = tensor_to_dl(&input, &in_shape, bf16_dt)?;
    let dl_ept = tensor_to_dl(&experts_i32, &ept_shape, i32_dt)?;
    let dl_tw = tensor_to_dl(&topk_weights, &tw_shape, f32_dt)?;
    let dl_w1 = tensor_to_dl(&w1, &w1_shape, bf16_dt)?;
    let dl_w2 = tensor_to_dl(&w2, &w2_shape, bf16_dt)?;

    // Set CUDA stream for TVM-FFI (must match the stream used by the engine)
    let registry = flashinfer::KernelRegistry::new();
    let (storage, _) = input.storage_and_layout();
    let cuda_storage = as_candle_cuda(&storage, "cutlass_fused_moe_stream")?;
    let dev = cuda_storage.device().clone();
    let stream = dev.cuda_stream();
    let raw_stream = unsafe { stream.cu_stream() } as *mut std::ffi::c_void;
    registry.set_stream(0, raw_stream);
    drop(storage);

    {
        let (w1_stor, w1_layout) = w1.storage_and_layout();
        let w1_off = w1_layout.start_offset();
        drop(w1_stor);
        tracing::debug!(
            "CUTLASS MoE: n={n_tokens} h={hidden} w1={:?} w2={:?} \
             w1_ptr={:?} w1_offset={w1_off}",
            w1.dims(), w2.dims(), dl_w1.data,
        );
    }

    unsafe {
        runner.run_moe(&dl_out, &dl_in, &dl_ept, &dl_tw, &dl_w1, &dl_w2, 1, 0, 1, 0)
    }.map_err(|e| candle_core::Error::Msg(format!("CUTLASS fused MoE failed: {e}")))?;

    Ok(output)
}
