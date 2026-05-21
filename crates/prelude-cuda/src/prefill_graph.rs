//! Mixed prefill/decode CUDA graph capture/replay.
//!
//! `DecodeGraphCache` covers the pure-decode (Q=1 per slot) hot path
//! with one graph per batch size. For mixed prefill+decode batches —
//! which dominate TTFT — we need a separate cache keyed on
//! `(num_tokens, num_requests)` where the packed-input + cu_seqlens
//! shapes vary per call.
//!
//! ## Approach (post-investigation of vLLM v0.20 cu130)
//!
//! vLLM's `cudagraph_mode=FULL_AND_PIECEWISE` splits the model at
//! `unified_attention_with_output` via torch.compile and captures
//! every non-attention piece into its own CUDA graph. That requires
//! per-layer per-piece per-bucket capture (~48*2*N graphs) and the
//! associated buffer-stitching machinery.
//!
//! prelude doesn't have torch.compile, so we take the simpler route
//! that's still tractable: capture the *whole* `forward_hidden_states`
//! call at a discrete set of `num_tokens` buckets. Attention's kernel
//! reads `cu_seqlens_q` / `cu_seqlens_k` from GPU tensors at runtime,
//! so the same captured graph handles any seq-length distribution
//! that fits the bucket. `last_token_select` + `lm_head` run eager
//! after replay (they depend on per-request CPU `seq_lens`).
//!
//! ## Buffer set per bucket
//!
//! - `packed_input`  `(bucket_num_tokens,)` U32
//! - `position_ids`  `(bucket_num_tokens,)` U32
//! - `slot_mapping`  `(bucket_num_tokens,)` I64
//! - `cu_seqlens_q`  `(bucket_num_reqs + 1,)` U32 — updated each replay
//! - `cu_seqlens_k`  `(bucket_num_reqs + 1,)` U32 — updated each replay
//! - `block_tables`  `(bucket_num_reqs, max_blocks)` U32
//! - FlashInfer plan buffers (`fi_indptr`, `fi_indices`, `fi_last_page_len`)
//!
//! All addresses are fixed at allocation time; on replay we
//! `memcpy_htod` the actual contents into the same buffers, recompute
//! the FA plan, then launch the captured graph.
//!
//! ## Dispatch
//!
//! At call time we round `num_tokens` up to the smallest captured
//! bucket and pad `num_requests` similarly. Padded request slots get
//! zero-length sequences (`cu_seqlens_q[i] == cu_seqlens_q[i+1]`); the
//! FA inner loop emits no rows for those, so the captured kernel can
//! be shared cleanly across actual `num_requests` values up to the
//! bucket maximum. Token-side padding past the actual `num_tokens` is
//! written from a dummy slot reused across the padded region.
//!
//! Cache lookup tries the exact `(num_tokens_bucket, num_reqs_bucket)`
//! pair first; if not yet captured, the bench-path falls through to
//! eager and we capture lazily on the first hit — same pattern as
//! `CUDAGraphWrapper.concrete_cudagraph_entries` in vLLM.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtr};
use prelude_core::cache::block_manager::BlockManager;
use prelude_core::config::EngineConfig;
use prelude_core::engine::EngineError;
use prelude_core::engine::executor::StepRequest;
use prelude_core::tensor::{DType, Device, Tensor};

use crate::device::GpuDType;

fn tensor_err(e: prelude_core::tensor::Error) -> EngineError {
    EngineError::Internal(format!("tensor: {e}"))
}

/// `(num_tokens_bucket, num_reqs_bucket)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BucketKey {
    pub num_tokens: usize,
    pub num_reqs: usize,
}

/// Sparse `num_tokens` buckets. Mirrors vLLM's
/// `compile_ranges_endpoints=[8192]` — finer at the low end where
/// chunked-prefill steps live, coarser higher up.
const TOKEN_BUCKETS: &[usize] = &[
    64, 128, 256, 512, 1024, 2048, 4096, 8192,
];

/// Sparse `num_requests` buckets.
const REQ_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// Returns true if `cuda_graph` is on and the model is graphable.
fn enabled_from_config(config: &EngineConfig, model_supports_graph: bool) -> bool {
    config.runtime.cuda_graph && model_supports_graph
}

/// Pre-allocated fixed-address GPU buffers for one mixed prefill graph.
pub(crate) struct MixedGraphBuffers {
    pub(crate) key: BucketKey,
    pub(crate) packed_input: Tensor,   // (num_tokens,) U32
    pub(crate) position_ids: Tensor,   // (num_tokens,) U32
    pub(crate) slot_mapping: Tensor,   // (num_tokens,) I64
    pub(crate) cu_seqlens_q: Tensor,   // (num_reqs+1,) U32
    pub(crate) cu_seqlens_k: Tensor,   // (num_reqs+1,) U32
    pub(crate) block_tables: Tensor,   // (num_reqs, max_blocks) U32
    pub(crate) max_blocks: usize,
    pub(crate) max_seqlen_k_capture: usize,
    pub(crate) fi_indptr: Tensor,        // (num_reqs+1,) I32
    pub(crate) fi_indices: Tensor,       // (num_reqs * max_blocks,) I32
    pub(crate) fi_last_page_len: Tensor, // (num_reqs,) I32
}

impl MixedGraphBuffers {
    pub(crate) fn allocate(
        key: BucketKey,
        max_blocks: usize,
        max_seqlen_k_capture: usize,
        device: &Device,
    ) -> Result<Self, prelude_core::engine::EngineError> {
        let max_total_pages = key.num_reqs * max_blocks;
        let (fi_indptr, fi_indices, fi_last_page_len) =
            crate::attn::flashinfer::allocate_fi_graph_meta(
                key.num_reqs,
                max_total_pages,
                device,
            )
            .map_err(tensor_err)?;

        Ok(Self {
            key,
            packed_input: Tensor::zeros((key.num_tokens,), DType::U32, device).map_err(tensor_err)?,
            position_ids: Tensor::zeros((key.num_tokens,), DType::U32, device).map_err(tensor_err)?,
            slot_mapping: Tensor::zeros((key.num_tokens,), DType::I64, device).map_err(tensor_err)?,
            cu_seqlens_q: Tensor::zeros((key.num_reqs + 1,), DType::U32, device).map_err(tensor_err)?,
            cu_seqlens_k: Tensor::zeros((key.num_reqs + 1,), DType::U32, device).map_err(tensor_err)?,
            block_tables: Tensor::zeros((key.num_reqs, max_blocks), DType::U32, device).map_err(tensor_err)?,
            max_blocks,
            max_seqlen_k_capture,
            fi_indptr,
            fi_indices,
            fi_last_page_len,
        })
    }
}

/// One captured `forward_hidden_states` graph at a specific bucket.
pub(crate) struct CapturedMixedGraph {
    pub(crate) graph: cudarc::driver::CudaGraph,
    pub(crate) buffers: MixedGraphBuffers,
    /// Hidden output, shape `(num_tokens, hidden_size)`. Caller slices
    /// `[0..actual_num_tokens]` before running `last_token_select` +
    /// `lm_head` eager.
    pub(crate) hidden_output: Tensor,
}

/// Cache of captured mixed prefill+decode graphs.
///
/// Owned by the GPU worker thread (same lifetime/threading model as
/// `DecodeGraphCache`).
pub(crate) struct MixedGraphCache {
    graphs: HashMap<BucketKey, CapturedMixedGraph>,
    /// Sorted ascending.
    token_buckets: Vec<usize>,
    req_buckets: Vec<usize>,
    enabled: bool,
}

impl MixedGraphCache {
    pub(crate) fn new(config: &EngineConfig, model_supports_graph: bool) -> Self {
        let enabled = enabled_from_config(config, model_supports_graph);
        if enabled {
            tracing::info!(
                token_buckets = ?TOKEN_BUCKETS,
                req_buckets = ?REQ_BUCKETS,
                "MixedGraphCache enabled — lazy capture on first use"
            );
        }
        Self {
            graphs: HashMap::new(),
            token_buckets: TOKEN_BUCKETS.to_vec(),
            req_buckets: REQ_BUCKETS.to_vec(),
            enabled,
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Round (num_tokens, num_reqs) up to the smallest captured pair.
    pub(crate) fn round_up(&self, num_tokens: usize, num_reqs: usize) -> Option<BucketKey> {
        let nt = self.token_buckets.iter().copied().find(|&b| b >= num_tokens)?;
        let nr = self.req_buckets.iter().copied().find(|&b| b >= num_reqs)?;
        Some(BucketKey {
            num_tokens: nt,
            num_reqs: nr,
        })
    }

    /// Look up a captured graph for an exact bucket key.
    pub(crate) fn get(&self, key: BucketKey) -> Option<&CapturedMixedGraph> {
        self.graphs.get(&key)
    }

    pub(crate) fn insert(&mut self, captured: CapturedMixedGraph) {
        let key = captured.buffers.key;
        self.graphs.insert(key, captured);
    }

    pub(crate) fn contains(&self, key: BucketKey) -> bool {
        self.graphs.contains_key(&key)
    }
}

/// Resolve the bucket key and return whether a captured graph exists
/// for it. Cheap pre-check on the dispatch hot path.
pub(crate) fn try_pick_bucket(
    cache: &MixedGraphCache,
    num_tokens: usize,
    num_reqs: usize,
) -> Option<BucketKey> {
    if !cache.is_enabled() {
        return None;
    }
    cache.round_up(num_tokens, num_reqs)
}

pub(crate) fn get_stream(device: &Device) -> Result<Arc<CudaStream>, EngineError> {
    let cuda_dev = device
        .as_cuda_device()
        .map_err(|e| EngineError::Internal(format!("as_cuda_device: {e}")))?;
    Ok(cuda_dev.cuda_stream())
}

/// CPU-side data returned by `update_buffers` for reuse downstream
/// (specifically, the FlashInfer plan precompute).
pub(crate) struct PrefillCpuData {
    pub(crate) cu_seqlens_k: Vec<u32>,
    pub(crate) block_tables: Vec<Vec<u32>>,
}

/// Populate the bucket's pre-allocated GPU buffers from a real prefill
/// step's request list. Padding strategy:
///
/// - Tokens past `real_total_tokens` get replicas of the first request's
///   first-slot (so `slot_mapping` / `position_ids` / `packed_input`
///   point at well-formed cache entries — the FA kernel does work on
///   them but the result is sliced away).
/// - Request slots past `requests.len()` get zero-length sequences
///   (`cu_seqlens_q[i] == cu_seqlens_q[i-1]`), which makes the FA inner
///   loop a no-op for those slots. Their `block_tables` row gets the
///   first request's block table to keep address resolution safe.
pub(crate) fn update_buffers(
    buffers: &MixedGraphBuffers,
    requests: &[StepRequest],
    block_size: usize,
    stream: &Arc<CudaStream>,
) -> Result<PrefillCpuData, EngineError> {
    let num_reqs = requests.len();
    let bucket_num_reqs = buffers.key.num_reqs;
    let bucket_num_tokens = buffers.key.num_tokens;
    debug_assert!(
        num_reqs > 0 && num_reqs <= bucket_num_reqs,
        "update_buffers: num_reqs {num_reqs} must be in 1..={bucket_num_reqs}"
    );

    let real_total_tokens: usize = requests.iter().map(|r| r.tokens.len()).sum();
    debug_assert!(
        real_total_tokens > 0 && real_total_tokens <= bucket_num_tokens,
        "update_buffers: real_total_tokens {real_total_tokens} must be in 1..={bucket_num_tokens}"
    );

    // ── packed_input + position_ids + slot_mapping (length bucket_num_tokens) ──
    let mut packed_tokens: Vec<u32> = Vec::with_capacity(bucket_num_tokens);
    let mut position_ids: Vec<u32> = Vec::with_capacity(bucket_num_tokens);
    let mut slot_mapping: Vec<i64> = Vec::with_capacity(bucket_num_tokens);

    for r in requests {
        for (i, &tok) in r.tokens.iter().enumerate() {
            packed_tokens.push(tok);
            position_ids.push((r.position_start + i) as u32);
            slot_mapping.push(BlockManager::slot(
                &r.block_table,
                r.position_start + i,
                block_size,
            ));
        }
    }
    // Pad with a dummy slot from requests[0]
    let dummy_token = requests[0].tokens.first().copied().unwrap_or(0);
    let dummy_pos = requests[0].position_start as u32;
    let dummy_slot = BlockManager::slot(
        &requests[0].block_table,
        requests[0].position_start,
        block_size,
    );
    for _ in real_total_tokens..bucket_num_tokens {
        packed_tokens.push(dummy_token);
        position_ids.push(dummy_pos);
        slot_mapping.push(dummy_slot);
    }

    unsafe { update_tensor(&buffers.packed_input, &packed_tokens, stream)? };
    unsafe { update_tensor(&buffers.position_ids, &position_ids, stream)? };
    unsafe { update_tensor(&buffers.slot_mapping, &slot_mapping, stream)? };

    // ── cu_seqlens_q (length bucket_num_reqs + 1) ──
    let mut cu_q: Vec<u32> = Vec::with_capacity(bucket_num_reqs + 1);
    cu_q.push(0);
    for r in requests {
        cu_q.push(cu_q.last().unwrap() + r.tokens.len() as u32);
    }
    let last_q = *cu_q.last().unwrap();
    for _ in num_reqs..bucket_num_reqs {
        cu_q.push(last_q);
    }
    unsafe { update_tensor(&buffers.cu_seqlens_q, &cu_q, stream)? };

    // ── cu_seqlens_k (length bucket_num_reqs + 1) ──
    let mut cu_k: Vec<u32> = Vec::with_capacity(bucket_num_reqs + 1);
    cu_k.push(0);
    for r in requests {
        cu_k.push(cu_k.last().unwrap() + r.context_len as u32);
    }
    let last_k = *cu_k.last().unwrap();
    for _ in num_reqs..bucket_num_reqs {
        cu_k.push(last_k);
    }
    unsafe { update_tensor(&buffers.cu_seqlens_k, &cu_k, stream)? };

    // ── block_tables (bucket_num_reqs × max_blocks) ──
    let max_blocks = buffers.max_blocks;
    let mut flat_bt: Vec<u32> = Vec::with_capacity(bucket_num_reqs * max_blocks);
    let mut per_seq_bt: Vec<Vec<u32>> = Vec::with_capacity(bucket_num_reqs);
    for r in requests {
        flat_bt.extend_from_slice(&r.block_table);
        flat_bt.resize(flat_bt.len() + max_blocks - r.block_table.len(), 0);
        per_seq_bt.push(r.block_table.clone());
    }
    let dummy_bt = &requests[0].block_table;
    for _ in num_reqs..bucket_num_reqs {
        flat_bt.extend_from_slice(dummy_bt);
        flat_bt.resize(flat_bt.len() + max_blocks - dummy_bt.len(), 0);
        per_seq_bt.push(dummy_bt.clone());
    }
    unsafe { update_tensor(&buffers.block_tables, &flat_bt, stream)? };

    Ok(PrefillCpuData {
        cu_seqlens_k: cu_k,
        block_tables: per_seq_bt,
    })
}

/// memcpy host data into a pre-allocated GPU tensor without realloc.
/// Mirrors `cuda_graph::buffers::update_tensor` (same safety preconditions).
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
