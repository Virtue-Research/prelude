//! CUDA graph capture and replay for decode (Q=1) steps.
//!
//! Captures `model.forward()` into replayable CUDA graphs keyed by
//! `(batch_size, seqlen_bucket)`. Pre-allocates fixed-size GPU input buffers
//! and updates them via `memcpy_htod` before each replay.
//!
//! Eager capture: graphs are captured for all (batch_size, seqlen_bucket)
//! combinations at startup.
//!
//! **FlashInfer mode** (`flashinfer` feature): no seqlen bucketing needed —
//! KV lengths are passed via device tensors, not scalar kernel args. Graph key
//! is just `batch_size`. Plan is computed outside the graph (before
//! capture/replay), graph only captures run() calls.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::cuda_backend::CudaDType;
use candle_core::{DType, Device, Storage, Tensor};

use crate::cache::block_manager::BlockManager;
use crate::config::EngineConfig;
use crate::engine::{Engine, EngineError, OwnedBatchDecodeSeq};
use crate::models::layers::{BatchAttnContext, PagedKvBatchContext};

fn candle_err(e: candle_core::Error) -> EngineError {
    EngineError::Internal(format!("candle: {e}"))
}

/// Seqlen buckets for graph capture (FA3 only — FlashInfer doesn't need them).
const SEQLEN_BUCKETS: &[usize] = &[256, 512, 1024, 2048, 4096];

/// Round `max_seqlen_k` up to the next bucket. Returns `None` if it exceeds
/// the largest bucket (caller should fall back to eager).
fn seqlen_bucket(max_seqlen_k: usize) -> Option<usize> {
    SEQLEN_BUCKETS.iter().copied().find(|&b| b >= max_seqlen_k)
}

/// Whether FlashInfer is the active attention backend at compile time.
const USE_FLASHINFER: bool = cfg!(feature = "flashinfer");

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
    // Updated via cudaMemcpyAsync before capture/replay so the graph's run()
    // always references the same addresses.
    #[cfg(feature = "flashinfer")]
    fi_indptr: Tensor,         // (bs+1,) I32
    #[cfg(feature = "flashinfer")]
    fi_indices: Tensor,        // (bs * max_blocks,) I32
    #[cfg(feature = "flashinfer")]
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

        #[cfg(feature = "flashinfer")]
        let (fi_indptr, fi_indices, fi_last_page_len) = {
            let max_total_pages = batch_size * max_blocks;
            crate::models::layers::allocate_fi_graph_meta(batch_size, max_total_pages, device)
                .map_err(candle_err)?
        };

        Ok(Self {
            batch_size,
            packed_input: Tensor::zeros((batch_size,), DType::U32, device).map_err(candle_err)?,
            cu_seqlens_q: Tensor::from_vec(cu_q, (batch_size + 1,), device).map_err(candle_err)?,
            cu_seqlens_k: Tensor::zeros((batch_size + 1,), DType::U32, device)
                .map_err(candle_err)?,
            position_ids: Tensor::zeros((batch_size,), DType::U32, device).map_err(candle_err)?,
            slot_mapping: Tensor::zeros((batch_size,), DType::I64, device).map_err(candle_err)?,
            block_tables: Tensor::zeros((batch_size, max_blocks), DType::U32, device)
                .map_err(candle_err)?,
            q_seq_lens: vec![1usize; batch_size],
            max_blocks,
            max_seqlen_k,
            #[cfg(feature = "flashinfer")]
            fi_indptr,
            #[cfg(feature = "flashinfer")]
            fi_indices,
            #[cfg(feature = "flashinfer")]
            fi_last_page_len,
        })
    }
}

/// Write host data into a pre-allocated GPU tensor without new allocation.
unsafe fn update_tensor<T: CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    tensor: &Tensor,
    data: &[T],
    stream: &Arc<CudaStream>,
) -> Result<(), EngineError> {
    debug_assert!(
        data.len() <= tensor.elem_count(),
        "update_tensor: data len {} exceeds tensor elem_count {}",
        data.len(), tensor.elem_count(),
    );
    let (mut guard, _layout) = tensor.storage_mut_and_layout();
    match *guard {
        Storage::Cuda(ref mut cs) => {
            let slice = T::as_cuda_slice_mut(cs)
                .map_err(|e| EngineError::Internal(format!("as_cuda_slice_mut: {e}")))?;
            stream
                .memcpy_htod(data, slice)
                .map_err(|e| EngineError::Internal(format!("memcpy_htod: {e}")))?;
            Ok(())
        }
        _ => Err(EngineError::Internal(
            "expected CUDA storage for graph buffer".into(),
        )),
    }
}

/// Update all pre-allocated buffers from the current decode batch.
fn update_buffers(
    buffers: &DecodeGraphBuffers,
    seqs: &[OwnedBatchDecodeSeq],
    block_size: usize,
    stream: &Arc<CudaStream>,
) -> Result<(), EngineError> {
    let bs = seqs.len();
    debug_assert_eq!(bs, buffers.batch_size);

    // packed_input
    let tokens: Vec<u32> = seqs.iter().map(|s| s.token).collect();
    unsafe { update_tensor(&buffers.packed_input, &tokens, stream)? };

    // cu_seqlens_k
    let mut cu_k: Vec<u32> = Vec::with_capacity(bs + 1);
    cu_k.push(0);
    for s in seqs {
        cu_k.push(cu_k.last().unwrap() + s.context_len as u32);
    }
    unsafe { update_tensor(&buffers.cu_seqlens_k, &cu_k, stream)? };

    // position_ids
    let positions: Vec<u32> = seqs.iter().map(|s| s.position as u32).collect();
    unsafe { update_tensor(&buffers.position_ids, &positions, stream)? };

    // slot_mapping
    let slots: Vec<i64> = seqs
        .iter()
        .map(|s| BlockManager::slot(&s.block_table, s.position, block_size))
        .collect();
    unsafe { update_tensor(&buffers.slot_mapping, &slots, stream)? };

    // block_tables: flatten + pad to max_blocks
    let max_blocks = buffers.max_blocks;
    let mut flat_bt: Vec<u32> = Vec::with_capacity(bs * max_blocks);
    for s in seqs {
        flat_bt.extend_from_slice(&s.block_table);
        flat_bt.resize(flat_bt.len() + max_blocks - s.block_table.len(), 0);
    }
    unsafe { update_tensor(&buffers.block_tables, &flat_bt, stream)? };

    Ok(())
}

// ---------------------------------------------------------------------------
// Captured graph
// ---------------------------------------------------------------------------

/// One captured CUDA graph plus its pre-allocated I/O buffers.
struct CapturedGraph {
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    buffers: DecodeGraphBuffers,
    /// Output tensor — points to GPU memory written by the graph.
    /// On replay, the graph writes to the same address.
    output: Tensor,
}

/// Graph cache key: (batch_size, seqlen_bucket).
/// FlashInfer mode: seqlen_bucket is always 0 (unused).
type GraphKey = (usize, usize);

// ---------------------------------------------------------------------------
// Graph cache
// ---------------------------------------------------------------------------

/// Cache of captured CUDA graphs, keyed by `(batch_size, seqlen_bucket)`.
///
/// Owned by the GPU worker thread — no `Arc`/`Mutex` needed.
pub(crate) struct DecodeGraphCache {
    graphs: HashMap<GraphKey, CapturedGraph>,
    max_bs: usize,
    block_size: usize,
    enabled: bool,
}

impl DecodeGraphCache {
    /// Create a new (empty) graph cache from engine configuration.
    pub fn new(
        config: &EngineConfig,
        block_size: usize,
        has_deltanet: bool,
    ) -> Self {
        let enabled = config.runtime.cuda_graph && !has_deltanet;
        let max_bs = config.runtime.cuda_graph_max_bs;

        if enabled {
            tracing::info!(
                max_bs,
                block_size,
                use_flashinfer = USE_FLASHINFER,
                "CUDA graph decode enabled"
            );
        }

        Self {
            graphs: HashMap::new(),
            max_bs,
            block_size,
            enabled,
        }
    }

    /// Eagerly capture graphs for all combinations at startup.
    pub fn warmup_all(&mut self, engine: &Engine) {
        if !self.enabled {
            return;
        }

        let t0 = std::time::Instant::now();
        let batch_sizes: Vec<usize> = (1..=self.max_bs).collect();

        if USE_FLASHINFER {
            // FlashInfer: no seqlen bucketing — key by batch_size only.
            // Use a large max_seqlen_k for buffer allocation.
            let total = batch_sizes.len();
            let mut captured = 0;
            for &bs in &batch_sizes {
                let key = (bs, 0);
                if self.graphs.contains_key(&key) {
                    continue;
                }
                // Allocate for up to 8192 seqlen (just buffer size, not a constraint)
                match self.capture(engine, bs, 8192) {
                    Ok(()) => {
                        captured += 1;
                        tracing::debug!(batch_size = bs, progress = format!("{captured}/{total}"),
                            "CUDA graph captured (FlashInfer, no seqlen bucket)");
                    }
                    Err(e) => {
                        tracing::warn!(batch_size = bs, error = %e, "CUDA graph warmup failed");
                    }
                }
            }
            let elapsed_ms = t0.elapsed().as_millis();
            tracing::info!(captured, total, elapsed_ms, "CUDA graph warmup complete (FlashInfer)");
        } else {
            // FA3: seqlen bucketing
            let total = batch_sizes.len() * SEQLEN_BUCKETS.len();
            let mut captured = 0;
            for &bs in &batch_sizes {
                for &bucket in SEQLEN_BUCKETS {
                    let key = (bs, bucket);
                    if self.graphs.contains_key(&key) {
                        continue;
                    }
                    match self.capture(engine, bs, bucket) {
                        Ok(()) => {
                            captured += 1;
                            tracing::debug!(batch_size = bs, seqlen_bucket = bucket,
                                progress = format!("{captured}/{total}"), "CUDA graph captured");
                        }
                        Err(e) => {
                            tracing::warn!(batch_size = bs, seqlen_bucket = bucket,
                                error = %e, "CUDA graph warmup capture failed");
                        }
                    }
                }
            }
            let elapsed_ms = t0.elapsed().as_millis();
            tracing::info!(captured, total, elapsed_ms, "CUDA graph warmup complete");
        }
    }

    /// Try to replay a captured graph for this decode batch.
    ///
    /// Returns `None` if CUDA graphs are disabled or the batch is not
    /// eligible. The caller should fall back to eager execution.
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

        // Skip if any sequence uses DeltaNet (hybrid models)
        if seqs.iter().any(|s| s.deltanet_slot.is_some()) {
            return None;
        }

        let actual_max_seqlen_k = seqs.iter().map(|s| s.context_len).max().unwrap_or(0);

        let key = if USE_FLASHINFER {
            // FlashInfer: no seqlen bucketing
            (bs, 0)
        } else {
            // FA3: seqlen bucketing
            let bucket = match seqlen_bucket(actual_max_seqlen_k) {
                Some(b) => b,
                None => return None,
            };
            (bs, bucket)
        };

        let captured = match self.graphs.get(&key) {
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

        // Update input buffers
        if let Err(e) = update_buffers(&captured.buffers, seqs, self.block_size, &stream) {
            return Some(Err(e));
        }

        // FlashInfer: pre-compute plan OUTSIDE the graph, then launch.
        // Plan writes scheduling data to workspace; graph replays run() which reads it.
        // Uses pre-allocated metadata buffers (fi_indptr/fi_indices/fi_last_page_len)
        // so the graph's run() references the same GPU addresses as during capture.
        #[cfg(feature = "flashinfer")]
        {
            let pool = match engine.cache.paged_pool.as_ref() {
                Some(p) => p,
                None => return Some(Err(EngineError::Internal("paged pool unavailable".into()))),
            };
            let key_caches = pool.active_key_caches();
            let num_qo_heads = engine.executor.config.num_attention_heads;
            let head_dim = engine.executor.config.head_dim;

            if let Err(e) = crate::models::layers::fi_precompute_paged_plan_graphed(
                (bs, num_qo_heads, head_dim),
                &key_caches[0],
                &captured.buffers.cu_seqlens_q,
                &captured.buffers.block_tables,
                &captured.buffers.cu_seqlens_k,
                1.0 / (head_dim as f32).sqrt(),
                &captured.buffers.fi_indptr,
                &captured.buffers.fi_indices,
                &captured.buffers.fi_last_page_len,
            ).map_err(candle_err) {
                return Some(Err(e));
            }
            // Plan cache populated with fixed-address metadata — graph replay is safe
        }

        // Replay the graph.
        match captured.graph.launch() {
            Ok(()) => Some(Ok(captured.output.clone())),
            Err(e) => Some(Err(EngineError::Internal(format!(
                "CUDA graph replay failed: {e}"
            )))),
        }
    }

    /// Capture a CUDA graph for the given batch size and seqlen bucket.
    fn capture(
        &mut self,
        engine: &Engine,
        batch_size: usize,
        seqlen_bucket: usize,
    ) -> Result<(), EngineError> {
        let device = &engine.executor.device;
        let pool = engine.cache.paged_pool.as_ref().ok_or_else(|| {
            EngineError::Internal("CUDA graph capture requires paged pool".into())
        })?;
        let stream = Self::get_stream(device)?;

        let max_blocks = (seqlen_bucket + self.block_size - 1) / self.block_size;

        // Allocate fixed-size buffers
        let buffers =
            DecodeGraphBuffers::allocate(batch_size, max_blocks, seqlen_bucket, device)?;

        let mut model = engine
            .executor
            .model
            .lock()
            .map_err(|e| EngineError::Internal(format!("model lock poisoned: {e}")))?;

        let key_caches = pool.active_key_caches();
        let value_caches = pool.active_value_caches();

        // Build paged decode context and run model.forward().
        // `manage_fi_cache`: whether to call begin/end_forward for FlashInfer plan cache.
        // During capture, plan cache is pre-populated, so we skip begin/end.
        macro_rules! decode_forward {
            ($model:expr, manage_fi_cache = $manage:expr) => {{
                let paged_kv = PagedKvBatchContext {
                    key_caches,
                    value_caches,
                    slot_mapping: &buffers.slot_mapping,
                    block_tables: &buffers.block_tables,
                    cu_seqlens_k: &buffers.cu_seqlens_k,
                    max_seqlen_k: buffers.max_seqlen_k,
                };
                let mut ctx = BatchAttnContext {
                    cu_seqlens_q: &buffers.cu_seqlens_q,
                    max_seqlen_q: 1,
                    position_ids: &buffers.position_ids,
                    seq_lens: &buffers.q_seq_lens,
                    paged_kv: Some(&paged_kv),
                    deltanet_pool: None,
                    deltanet_slots: None,
                };
                #[cfg(feature = "flashinfer")]
                if $manage { crate::models::layers::fi_begin_forward(); }
                let result = $model
                    .forward(&buffers.packed_input, &mut ctx)
                    .map_err(candle_err);
                #[cfg(feature = "flashinfer")]
                if $manage { crate::models::layers::fi_end_forward(); }
                result
            }};
        }

        // ── Warmup (eager, no capture) ──
        // Ensures cuBLAS/cuDNN plans are initialized, no JIT during capture.
        let _ = decode_forward!(model, manage_fi_cache = true)?;
        stream
            .synchronize()
            .map_err(|e| EngineError::Internal(format!("warmup sync: {e}")))?;

        // ── Pre-capture plan (FlashInfer only) ──
        // Compute plan OUTSIDE the graph so plan's GPU workspace writes are
        // not baked into the graph. The graph only captures run() calls.
        // Uses pre-allocated metadata buffers so the captured run() kernels
        // reference fixed GPU addresses that remain valid during replay.
        #[cfg(feature = "flashinfer")]
        {
            let num_qo_heads = engine.executor.config.num_attention_heads;
            let head_dim = engine.executor.config.head_dim;
            crate::models::layers::fi_precompute_paged_plan_graphed(
                (batch_size, num_qo_heads, head_dim),
                &key_caches[0],
                &buffers.cu_seqlens_q,
                &buffers.block_tables,
                &buffers.cu_seqlens_k,
                1.0 / (head_dim as f32).sqrt(),
                &buffers.fi_indptr,
                &buffers.fi_indices,
                &buffers.fi_last_page_len,
            ).map_err(candle_err)?;
            // Plan cache populated with fixed-address metadata — capture will only call run()
        }

        // ── Capture ──
        stream
            .begin_capture(
                candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
            .map_err(|e| EngineError::Internal(format!("begin_capture: {e}")))?;

        // FlashInfer: plan cache is pre-populated, skip begin/end_forward (would reset cache).
        // FA3: no pre-computed plan, use normal begin/end_forward.
        let output = decode_forward!(model, manage_fi_cache = !USE_FLASHINFER)?;
        let output = output.squeeze(1).map_err(candle_err)?;

        let graph = stream
            .end_capture(
                candle_core::cuda_backend::cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| EngineError::Internal(format!("end_capture: {e}")))?
            .ok_or_else(|| EngineError::Internal("end_capture returned None".into()))?;

        drop(model);

        let key = if USE_FLASHINFER { (batch_size, 0) } else { (batch_size, seqlen_bucket) };
        self.graphs.insert(
            key,
            CapturedGraph {
                graph,
                buffers,
                output,
            },
        );

        Ok(())
    }

    /// Get the CUDA stream from the device.
    fn get_stream(device: &Device) -> Result<Arc<CudaStream>, EngineError> {
        let cuda_dev = device
            .as_cuda_device()
            .map_err(|e| EngineError::Internal(format!("as_cuda_device: {e}")))?;
        Ok(cuda_dev.cuda_stream())
    }
}
