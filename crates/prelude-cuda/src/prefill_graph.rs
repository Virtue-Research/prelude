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

use cudarc::driver::CudaStream;
use prelude_core::config::EngineConfig;
use prelude_core::tensor::{DType, Device, Tensor};

fn tensor_err(e: prelude_core::tensor::Error) -> prelude_core::engine::EngineError {
    prelude_core::engine::EngineError::Internal(format!("tensor: {e}"))
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

#[allow(dead_code)]
pub(crate) fn get_stream(device: &Device) -> Result<Arc<CudaStream>, prelude_core::engine::EngineError> {
    let cuda_dev = device
        .as_cuda_device()
        .map_err(|e| prelude_core::engine::EngineError::Internal(format!("as_cuda_device: {e}")))?;
    Ok(cuda_dev.cuda_stream())
}
