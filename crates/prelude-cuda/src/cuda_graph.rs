//! CUDA graph capture and replay for decode (Q=1) steps.
//!
//! Captures `model.forward()` into replayable CUDA graphs keyed by
//! `batch_size`. Pre-allocates fixed-size GPU input buffers and updates
//! them via `memcpy_htod` before each replay — no tensor allocation per step.
//!
//! Eager capture: graphs are captured for all batch sizes at startup.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtr};
use prelude_core::tensor::{DType, Device, Tensor};

use prelude_core::cache::block_manager::BlockManager;
use prelude_core::config::EngineConfig;
use prelude_core::engine::{Engine, EngineError, OwnedBatchDecodeSeq};
use prelude_core::models::commons::{BatchAttnContext, PagedKvBatchContext};

use crate::device::GpuDType;

fn tensor_err(e: prelude_core::tensor::Error) -> EngineError {
    EngineError::Internal(format!("tensor: {e}"))
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers for one captured graph.
/// All tensor addresses are fixed at allocation time and never change.
struct DecodeGraphBuffers {
    batch_size: usize,
    packed_input: Tensor,   // (bs,) U32
    cu_seqlens_q: Tensor,   // (bs+1,) U32 — fixed [0,1,...,N], never updated
    cu_seqlens_k: Tensor,   // (bs+1,) U32
    position_ids: Tensor,   // (bs,) U32
    slot_mapping: Tensor,   // (bs,) I64
    block_tables: Tensor,   // (bs, max_blocks) U32
    q_seq_lens: Vec<usize>, // [1; bs] — CPU, fixed
    max_blocks: usize,
    max_seqlen_k: usize,
    // FlashInfer metadata buffers — pre-allocated with fixed GPU addresses.
    fi_indptr: Tensor,         // (bs+1,) I32
    fi_indices: Tensor,        // (bs * max_blocks,) I32
    fi_last_page_len: Tensor,  // (bs,) I32
}

impl DecodeGraphBuffers {
    fn allocate(
        batch_size: usize,
        max_blocks: usize,
        max_seqlen_k: usize,
        device: &Device,
    ) -> Result<Self, EngineError> {
        let cu_q: Vec<u32> = (0..=batch_size as u32).collect();

        let max_total_pages = batch_size * max_blocks;
        let (fi_indptr, fi_indices, fi_last_page_len) =
            crate::attn::flashinfer::allocate_fi_graph_meta(batch_size, max_total_pages, device)
                .map_err(tensor_err)?;

        Ok(Self {
            batch_size,
            packed_input: Tensor::zeros((batch_size,), DType::U32, device).map_err(tensor_err)?,
            cu_seqlens_q: Tensor::from_vec(cu_q, (batch_size + 1,), device).map_err(tensor_err)?,
            cu_seqlens_k: Tensor::zeros((batch_size + 1,), DType::U32, device).map_err(tensor_err)?,
            position_ids: Tensor::zeros((batch_size,), DType::U32, device).map_err(tensor_err)?,
            slot_mapping: Tensor::zeros((batch_size,), DType::I64, device).map_err(tensor_err)?,
            block_tables: Tensor::zeros((batch_size, max_blocks), DType::U32, device).map_err(tensor_err)?,
            q_seq_lens: vec![1usize; batch_size],
            max_blocks,
            max_seqlen_k,
            fi_indptr,
            fi_indices,
            fi_last_page_len,
        })
    }
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
        data.len(), tensor.elem_count(),
    );
    // Safety: single GPU worker thread, no concurrent access to these graph-owned buffers.
    // Use raw CUDA memcpy with the device pointer from the tensor's CudaSlice.
    // We only need a read lock — the GPU write doesn't mutate the Rust struct.
    let (guard, _layout) = tensor.storage_and_layout();
    match &*guard {
        prelude_core::tensor::Storage::Cuda(cs) => {
            let slice = <T as candle_core::cuda_backend::CudaDType>::as_cuda_slice(cs)
                .map_err(|e| EngineError::Internal(format!("as_cuda_slice: {e}")))?;
            let (dev_ptr, _g) = slice.device_ptr(stream);
            let raw_stream = stream.cu_stream();
            cudarc::driver::result::memcpy_htod_async(dev_ptr, data, raw_stream)
                .map_err(|e| EngineError::Internal(format!("memcpy_htod: {e}")))?;
        }
        _ => return Err(EngineError::Internal("update_tensor: expected CUDA storage".into())),
    }
    Ok(())
}

/// CPU-side data computed during buffer update, reused for FlashInfer plan
/// to avoid redundant GPU→CPU copies.
struct CpuBatchData {
    cu_seqlens_k: Vec<u32>,
    block_tables: Vec<Vec<u32>>,
}

/// Update all pre-allocated buffers from the current decode batch.
/// Returns CPU-side data for reuse by FlashInfer plan computation.
fn update_buffers(
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

    Ok(CpuBatchData {
        cu_seqlens_k: cu_k,
        block_tables: per_seq_bt,
    })
}

// ---------------------------------------------------------------------------
// Captured graph
// ---------------------------------------------------------------------------

struct CapturedGraph {
    graph: cudarc::driver::CudaGraph,
    buffers: DecodeGraphBuffers,
    output: Tensor,
}

// ---------------------------------------------------------------------------
// Graph cache
// ---------------------------------------------------------------------------

/// Cache of captured CUDA graphs, keyed by `batch_size`.
///
/// Owned by the GPU worker thread — no `Arc`/`Mutex` needed.
pub(crate) struct DecodeGraphCache {
    graphs: HashMap<usize, CapturedGraph>,
    max_bs: usize,
    block_size: usize,
    enabled: bool,
}

impl DecodeGraphCache {
    pub fn new(
        config: &EngineConfig,
        block_size: usize,
        has_deltanet: bool,
    ) -> Self {
        let enabled = config.runtime.cuda_graph && !has_deltanet;
        let max_bs = config.runtime.cuda_graph_max_bs;

        if enabled {
            tracing::info!(max_bs, block_size, "CUDA graph decode enabled");
        }

        Self {
            graphs: HashMap::new(),
            max_bs,
            block_size,
            enabled,
        }
    }

    /// Eagerly capture graphs for all batch sizes at startup.
    pub fn warmup_all(&mut self, engine: &Engine) {
        if !self.enabled {
            return;
        }

        let t0 = std::time::Instant::now();
        let total = self.max_bs;
        let mut captured = 0;

        for bs in 1..=self.max_bs {
            // Allocate for up to 8192 seqlen (just buffer size, not a constraint)
            match self.capture(engine, bs, 8192) {
                Ok(()) => {
                    captured += 1;
                    tracing::debug!(batch_size = bs, progress = format!("{captured}/{total}"),
                        "CUDA graph captured");
                }
                Err(e) => {
                    tracing::warn!(batch_size = bs, error = %e, "CUDA graph warmup failed");
                }
            }
        }
        let elapsed_ms = t0.elapsed().as_millis();
        tracing::info!(captured, total, elapsed_ms, "CUDA graph warmup complete");
    }

    /// Try to replay a captured graph for this decode batch.
    /// Returns `None` if not eligible; caller falls back to eager.
    pub fn try_replay(
        &mut self,
        engine: &Engine,
        seqs: &[OwnedBatchDecodeSeq],
    ) -> Option<Result<Tensor, EngineError>> {
        if !self.enabled {
            return None;
        }

        let bs = seqs.len();
        if bs == 0 || bs > self.max_bs {
            return None;
        }

        // Skip DeltaNet sequences (hybrid models)
        if seqs.iter().any(|s| s.deltanet_slot.is_some()) {
            return None;
        }

        let captured = match self.graphs.get(&bs) {
            Some(c) => c,
            None => return None,
        };

        // Check block table fits
        let actual_max_blocks = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(0);
        if actual_max_blocks > captured.buffers.max_blocks {
            return None;
        }

        let stream = match Self::get_stream(&engine.executor.device) {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };

        // Update input buffers (returns CPU-side data for plan reuse)
        let cpu_data = match update_buffers(&captured.buffers, seqs, self.block_size, &stream) {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        // Pre-compute FlashInfer plan outside the graph.
        // Uses CPU-side data from update_buffers to avoid GPU→CPU syncs.
        {
            let pool = match engine.cache.paged_pool.as_ref() {
                Some(p) => p,
                None => return Some(Err(EngineError::Internal("paged pool unavailable".into()))),
            };
            let key_caches = pool.active_key_caches();
            let num_qo_heads = engine.executor.config.num_attention_heads;
            let head_dim = engine.executor.config.head_dim;

            if let Err(e) = crate::attn::flashinfer::precompute_paged_plan_replay(
                (bs, num_qo_heads, head_dim),
                &key_caches[0],
                &cpu_data.cu_seqlens_k,
                &cpu_data.block_tables,
                self.block_size,
                &captured.buffers.fi_indptr,
                &captured.buffers.fi_indices,
                &captured.buffers.fi_last_page_len,
            ).map_err(tensor_err) {
                return Some(Err(e));
            }
        }

        // Replay
        match captured.graph.launch() {
            Ok(()) => Some(Ok(captured.output.clone())),
            Err(e) => Some(Err(EngineError::Internal(format!(
                "CUDA graph replay failed: {e}"
            )))),
        }
    }

    /// Capture a CUDA graph for the given batch size.
    fn capture(
        &mut self,
        engine: &Engine,
        batch_size: usize,
        seqlen_budget: usize,
    ) -> Result<(), EngineError> {
        let device = &engine.executor.device;
        let pool = engine.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("CUDA graph capture requires paged pool".into())
        })?;
        let stream = Self::get_stream(device)?;

        let max_blocks = (seqlen_budget + self.block_size - 1) / self.block_size;
        let buffers = DecodeGraphBuffers::allocate(batch_size, max_blocks, seqlen_budget, device)?;

        let mut model = engine
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;

        let key_caches = pool.active_key_caches();
        let value_caches = pool.active_value_caches();

        // Build decode context and run model.forward().
        macro_rules! decode_forward {
            ($model:expr, manage_fi_cache = $manage:expr) => {{
                let paged_kv = PagedKvBatchContext {
                    key_caches: &key_caches,
                    value_caches: &value_caches,
                    slot_mapping: &buffers.slot_mapping,
                    block_tables: &buffers.block_tables,
                    cu_seqlens_k: &buffers.cu_seqlens_k,
                    max_seqlen_k: buffers.max_seqlen_k,
                };
                let mut ctx = BatchAttnContext {
                    ops: engine.executor.ops,
                    cu_seqlens_q: &buffers.cu_seqlens_q,
                    max_seqlen_q: 1,
                    position_ids: &buffers.position_ids,
                    seq_lens: &buffers.q_seq_lens,
                    paged_kv: Some(&paged_kv),
                    deltanet_pool: None,
                    deltanet_slots: None,
                };
                if $manage { engine.executor.ops.begin_forward(); }
                let result = $model
                    .forward(&buffers.packed_input, &mut ctx)
                    .map_err(tensor_err);
                if $manage { engine.executor.ops.end_forward(); }
                result
            }};
        }

        // Warmup (eager) — ensures kernel plans are initialized
        let _ = decode_forward!(model, manage_fi_cache = true)?;
        stream
            .synchronize()
            .map_err(|e| EngineError::Internal(format!("warmup sync: {e}")))?;

        // Pre-compute FlashInfer plan outside graph
        {
            let num_qo_heads = engine.executor.config.num_attention_heads;
            let head_dim = engine.executor.config.head_dim;
            crate::attn::flashinfer::precompute_paged_plan_capture(
                (batch_size, num_qo_heads, head_dim),
                &key_caches[0],
                &buffers.cu_seqlens_q,
                &buffers.block_tables,
                &buffers.cu_seqlens_k,
                1.0 / (head_dim as f32).sqrt(),
                &buffers.fi_indptr,
                &buffers.fi_indices,
                &buffers.fi_last_page_len,
            ).map_err(tensor_err)?;
        }

        // Capture
        tracing::debug!(batch_size, "CUDA graph: begin_capture");
        stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| EngineError::Internal(format!("begin_capture: {e}")))?;

        // Plan is pre-populated — skip begin/end_forward during capture
        tracing::debug!(batch_size, "CUDA graph: running model.forward inside capture");
        let output = decode_forward!(model, manage_fi_cache = false)?;
        let output = output.squeeze(1).map_err(tensor_err)?;

        tracing::debug!(batch_size, "CUDA graph: end_capture");
        let graph = stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| EngineError::Internal(format!("end_capture: {e}")))?
            .ok_or_else(|| EngineError::Internal("end_capture returned None".into()))?;

        drop(model);

        self.graphs.insert(batch_size, CapturedGraph { graph, buffers, output });
        Ok(())
    }

    fn get_stream(device: &Device) -> Result<Arc<CudaStream>, EngineError> {
        let cuda_dev = device
            .as_cuda_device()
            .map_err(|e| EngineError::Internal(format!("as_cuda_device: {e}")))?;
        Ok(cuda_dev.cuda_stream())
    }
}
