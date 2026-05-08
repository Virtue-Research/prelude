//! CUDA graph capture and replay for decode (Q=1) steps.
//!
//! Captures `model.forward()` into replayable CUDA graphs keyed by
//! `batch_size`. Pre-allocates fixed-size GPU input buffers and updates
//! them via `memcpy_htod` before each replay — no tensor allocation per step.
//!
//! Eager capture: graphs are captured for all batch sizes at startup.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use prelude_core::tensor::{Device, Tensor};

use prelude_core::config::EngineConfig;
use prelude_core::engine::{Engine, EngineError, OwnedBatchDecodeSeq};
use prelude_core::models::commons::{BatchAttnContext, PagedKvBatchContext};

mod buffers;

use self::buffers::{DecodeGraphBuffers, update_buffers};

fn tensor_err(e: prelude_core::tensor::Error) -> EngineError {
    EngineError::Internal(format!("tensor: {e}"))
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
        model_supports_graph: bool,
    ) -> Self {
        let enabled = config.runtime.cuda_graph && !has_deltanet && model_supports_graph;
        let max_bs = config.runtime.cuda_graph_max_bs;

        if enabled {
            tracing::info!(max_bs, block_size, "CUDA graph decode enabled");
        } else if config.runtime.cuda_graph && !model_supports_graph {
            tracing::info!(
                "CUDA graph decode disabled: model does not support graph capture \
                 (e.g. MoE with dynamic expert shapes)"
            );
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
                    tracing::debug!(
                        batch_size = bs,
                        progress = format!("{captured}/{total}"),
                        "CUDA graph captured"
                    );
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
            )
            .map_err(tensor_err)
            {
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

        let has_deltanet = engine.cache.deltanet_pool.is_some();
        let max_blocks = (seqlen_budget + self.block_size - 1) / self.block_size;
        let buffers = DecodeGraphBuffers::allocate(
            batch_size,
            max_blocks,
            seqlen_budget,
            has_deltanet,
            device,
        )?;

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
                let mut dn_pool_guard = engine
                    .cache
                    .deltanet_pool
                    .as_ref()
                    .map(|m| m.lock().unwrap());
                let dn_pool_ref = dn_pool_guard.as_deref_mut();
                let dn_slots_vec: Option<Vec<u32>> = buffers.deltanet_slots.as_ref().map(|t| {
                    t.to_vec1::<u32>()
                        .unwrap_or_else(|_| vec![0u32; batch_size])
                });
                let dn_slots_ref = dn_slots_vec.as_deref();
                let mut ctx = BatchAttnContext {
                    ops: engine.executor.ops,
                    cu_seqlens_q: &buffers.cu_seqlens_q,
                    max_seqlen_q: 1,
                    position_ids: &buffers.position_ids,
                    seq_lens: &buffers.q_seq_lens,
                    paged_kv: Some(&paged_kv),
                    deltanet_pool: dn_pool_ref,
                    deltanet_slots: dn_slots_ref,
                    deltanet_state_is_zero: None,
                    deltanet_slots_gpu: buffers.deltanet_slots.as_ref(),
                };
                if $manage {
                    engine.executor.ops.begin_forward();
                }
                let result = $model
                    .forward(&buffers.packed_input, &mut ctx)
                    .map_err(tensor_err);
                if $manage {
                    engine.executor.ops.end_forward();
                }
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
            )
            .map_err(tensor_err)?;
        }

        // Capture
        tracing::debug!(batch_size, "CUDA graph: begin_capture");
        stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| EngineError::Internal(format!("begin_capture: {e}")))?;

        // Plan is pre-populated — skip begin/end_forward during capture
        tracing::debug!(
            batch_size,
            "CUDA graph: running model.forward inside capture"
        );
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

        self.graphs.insert(
            batch_size,
            CapturedGraph {
                graph,
                buffers,
                output,
            },
        );
        Ok(())
    }

    fn get_stream(device: &Device) -> Result<Arc<CudaStream>, EngineError> {
        let cuda_dev = device
            .as_cuda_device()
            .map_err(|e| EngineError::Internal(format!("as_cuda_device: {e}")))?;
        Ok(cuda_dev.cuda_stream())
    }
}
