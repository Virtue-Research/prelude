use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtr};
use prelude_core::cache::block_manager::BlockManager;
use prelude_core::engine::{EngineError, OwnedBatchDecodeSeq};
use prelude_core::tensor::{DType, Device, Tensor};

use crate::device::GpuDType;

/// Pre-allocated GPU buffers for one captured graph.
/// All tensor addresses are fixed at allocation time and never change.
pub(super) struct DecodeGraphBuffers {
    pub(super) batch_size: usize,
    pub(super) packed_input: Tensor,   // (bs,) U32
    pub(super) cu_seqlens_q: Tensor,   // (bs+1,) U32 - fixed [0,1,...,N], never updated
    pub(super) cu_seqlens_k: Tensor,   // (bs+1,) U32
    pub(super) position_ids: Tensor,   // (bs,) U32
    pub(super) slot_mapping: Tensor,   // (bs,) I64
    pub(super) block_tables: Tensor,   // (bs, max_blocks) U32
    pub(super) q_seq_lens: Vec<usize>, // [1; bs] - CPU, fixed
    pub(super) max_blocks: usize,
    pub(super) max_seqlen_k: usize,
    // FlashInfer metadata buffers - pre-allocated with fixed GPU addresses.
    pub(super) fi_indptr: Tensor,        // (bs+1,) I32
    pub(super) fi_indices: Tensor,       // (bs * max_blocks,) I32
    pub(super) fi_last_page_len: Tensor, // (bs,) I32
    // DeltaNet slot IDs - pre-allocated for hybrid models.
    pub(super) deltanet_slots: Option<Tensor>, // (bs,) U32
}

impl DecodeGraphBuffers {
    pub(super) fn allocate(
        batch_size: usize,
        max_blocks: usize,
        max_seqlen_k: usize,
        has_deltanet: bool,
        device: &Device,
    ) -> Result<Self, EngineError> {
        let cu_q: Vec<u32> = (0..=batch_size as u32).collect();

        let max_total_pages = batch_size * max_blocks;
        let (fi_indptr, fi_indices, fi_last_page_len) =
            crate::attn::flashinfer::allocate_fi_graph_meta(batch_size, max_total_pages, device)
                .map_err(super::tensor_err)?;

        let deltanet_slots = if has_deltanet {
            Some(Tensor::zeros((batch_size,), DType::U32, device).map_err(super::tensor_err)?)
        } else {
            None
        };

        Ok(Self {
            batch_size,
            packed_input: Tensor::zeros((batch_size,), DType::U32, device)
                .map_err(super::tensor_err)?,
            cu_seqlens_q: Tensor::from_vec(cu_q, (batch_size + 1,), device)
                .map_err(super::tensor_err)?,
            cu_seqlens_k: Tensor::zeros((batch_size + 1,), DType::U32, device)
                .map_err(super::tensor_err)?,
            position_ids: Tensor::zeros((batch_size,), DType::U32, device)
                .map_err(super::tensor_err)?,
            slot_mapping: Tensor::zeros((batch_size,), DType::I64, device)
                .map_err(super::tensor_err)?,
            block_tables: Tensor::zeros((batch_size, max_blocks), DType::U32, device)
                .map_err(super::tensor_err)?,
            q_seq_lens: vec![1usize; batch_size],
            max_blocks,
            max_seqlen_k,
            fi_indptr,
            fi_indices,
            fi_last_page_len,
            deltanet_slots,
        })
    }
}

/// CPU-side data computed during buffer update, reused for FlashInfer plan
/// to avoid redundant GPU-to-CPU copies.
pub(super) struct CpuBatchData {
    pub(super) cu_seqlens_k: Vec<u32>,
    pub(super) block_tables: Vec<Vec<u32>>,
}

/// Update all pre-allocated buffers from the current decode batch.
/// Returns CPU-side data for reuse by FlashInfer plan computation.
pub(super) fn update_buffers(
    buffers: &DecodeGraphBuffers,
    seqs: &[OwnedBatchDecodeSeq],
    block_size: usize,
    stream: &Arc<CudaStream>,
) -> Result<CpuBatchData, EngineError> {
    let bs = seqs.len();
    debug_assert_eq!(bs, buffers.batch_size);

    let tokens: Vec<u32> = seqs.iter().map(|s| s.token).collect();
    unsafe { update_tensor(&buffers.packed_input, &tokens, stream)? };

    let mut cu_k: Vec<u32> = Vec::with_capacity(bs + 1);
    cu_k.push(0);
    for s in seqs {
        cu_k.push(cu_k.last().unwrap() + s.context_len as u32);
    }
    unsafe { update_tensor(&buffers.cu_seqlens_k, &cu_k, stream)? };

    let positions: Vec<u32> = seqs.iter().map(|s| s.position as u32).collect();
    unsafe { update_tensor(&buffers.position_ids, &positions, stream)? };

    let slots: Vec<i64> = seqs
        .iter()
        .map(|s| BlockManager::slot(&s.block_table, s.position, block_size))
        .collect();
    unsafe { update_tensor(&buffers.slot_mapping, &slots, stream)? };

    let max_blocks = buffers.max_blocks;
    let mut flat_bt: Vec<u32> = Vec::with_capacity(bs * max_blocks);
    let per_seq_bt: Vec<Vec<u32>> = seqs.iter().map(|s| s.block_table.clone()).collect();
    for s in seqs {
        flat_bt.extend_from_slice(&s.block_table);
        flat_bt.resize(flat_bt.len() + max_blocks - s.block_table.len(), 0);
    }
    unsafe { update_tensor(&buffers.block_tables, &flat_bt, stream)? };

    if let Some(ref dn_buf) = buffers.deltanet_slots {
        let dn_slots: Vec<u32> = seqs.iter().map(|s| s.deltanet_slot.unwrap_or(0)).collect();
        unsafe { update_tensor(dn_buf, &dn_slots, stream)? };
    }

    Ok(CpuBatchData {
        cu_seqlens_k: cu_k,
        block_tables: per_seq_bt,
    })
}

/// Write host data into a pre-allocated GPU tensor without new allocation.
///
/// Safety: caller must ensure no concurrent access to this tensor's storage.
/// This is called from the single GPU worker thread for CUDA graph buffer updates.
unsafe fn update_tensor<T: GpuDType + candle_core::cuda_backend::CudaDType>(
    tensor: &Tensor,
    data: &[T],
    stream: &Arc<CudaStream>,
) -> Result<(), EngineError> {
    debug_assert!(
        data.len() <= tensor.elem_count(),
        "update_tensor: data len {} exceeds tensor elem_count {}",
        data.len(),
        tensor.elem_count(),
    );
    // Safety: single GPU worker thread, no concurrent access to these graph-owned buffers.
    // Use raw CUDA memcpy with the device pointer from the tensor's CudaSlice.
    // We only need a read lock - the GPU write doesn't mutate the Rust struct.
    let (guard, _layout) = tensor.storage_and_layout();
    match &*guard {
        prelude_core::tensor::Storage::Cuda(cs) => {
            let slice = <T as candle_core::cuda_backend::CudaDType>::as_cuda_slice(cs)
                .map_err(|e| EngineError::Internal(format!("as_cuda_slice: {e}")))?;
            let (dev_ptr, _g) = slice.device_ptr(stream);
            let raw_stream = stream.cu_stream();
            unsafe {
                cudarc::driver::result::memcpy_htod_async(dev_ptr, data, raw_stream)
                    .map_err(|e| EngineError::Internal(format!("memcpy_htod: {e}")))?;
            }
        }
        _ => {
            return Err(EngineError::Internal(
                "update_tensor: expected CUDA storage".into(),
            ));
        }
    }
    Ok(())
}
