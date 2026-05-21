//! Piecewise CUDA graph capture/replay for prefill steps.
//!
//! Closes the prefill TTFT gap with vLLM's `cudagraph_mode=FULL_AND_PIECEWISE`:
//! vLLM splits the model at attention via `torch.compile` and CUDA-graphs
//! every non-attention piece; prelude historically ran prefill fully eager
//! and paid the per-layer kernel-launch overhead 48 times.
//!
//! ## Design
//!
//! Prelude doesn't have `torch.compile`, so we hand-split the model into
//! pieces in the model code itself (see `Qwen3Attention::{forward_pre_attn,
//! forward_attention, forward_post_attn}` and the matching MoE decoder
//! layer methods). The runtime composition per layer is:
//!
//! ```text
//! pre_piece  = input_norm + QKV proj + QK-norm + RoPE + paged KV write
//! attention  = paged_attention (eager — cu_seqlens varies per call)
//! post_piece = O proj + post-attn norm + MoE FFN
//! ```
//!
//! Both pieces are *shape-stable in `num_tokens`*: every kernel inside
//! depends only on the packed token count, never on per-sequence
//! boundaries. So a graph captured at `num_tokens = 4096` replays for
//! any prefill packed to that bucket — extra padding rows are inert.
//!
//! ## Capture strategy (lazy, like vLLM's `CUDAGraphWrapper`)
//!
//! - Capture sizes are picked from a sparse list (1, 32, 64, 128, 256,
//!   512, 1024, 2048, 4096, 8192) — every dispatch rounds up to the
//!   smallest enclosing bucket.
//! - We capture on **first encounter** of a `(bucket, layer_idx, piece)`
//!   triple, not at startup. Saves multi-minute warmup over the ~1800
//!   graph instantiations that exhaustive capture would imply, and only
//!   pays for buckets the real workload actually hits.
//!
//! ## Shared persistent buffers per bucket
//!
//! Each bucket owns one set of fixed-address tensors that every layer's
//! pieces read from / write to:
//!
//! - `hidden`   `[bucket, hidden_size]`   layer input + post-piece output
//! - `residual` `[bucket, hidden_size]`   running residual carry-in/out
//! - `q`        `[bucket, num_heads, head_dim]`     pre-piece Q output
//! - `k` / `v`  `[bucket, num_kv_heads, head_dim]`  pre-piece K, V
//! - `attn_out` `[bucket, num_heads, head_dim]`     attention output
//!
//! The captured graphs reference these addresses directly. Between
//! layers, the post-piece writes back into `hidden` so layer i+1's
//! pre-piece reads the same buffer — no buffer ping-pong needed.
//!
//! ## Dispatch flow at runtime
//!
//! ```text
//! prefill step with num_tokens = N:
//!   bucket = round_up(N, capture_sizes)
//!   buffers = ensure_buffers(bucket)
//!   pad N → bucket (zero-fill the extra rows in hidden)
//!   for layer in 0..num_layers:
//!     pre_graph[bucket][layer].replay()      # reads hidden, writes q,k,v
//!     attn_out_eager = attention_eager(q, k, v, ctx)   # reads true cu_seqlens
//!     # ensure attn_out lives at the captured address (memcpy if needed)
//!     post_graph[bucket][layer].replay()     # reads attn_out + residual, writes hidden
//!   slice hidden[..N] back to the caller
//! ```
//!
//! Attention itself is the only thing not graphed (its kernel registry
//! is keyed on per-sequence varlen state).
//!
//! ## Wiring (TODO: subsequent commits)
//!
//! 1. **Buffer allocation** (`BucketBuffers::allocate`) — fix sizes from
//!    the model config (`hidden_size`, `num_heads`, `num_kv_heads`,
//!    `head_dim`) and pre-allocate on the model device.
//! 2. **Graph capture for one piece** (`PrefillGraphCache::capture_pre` /
//!    `capture_post`) — modelled after `DecodeGraphCache::capture` in
//!    `cuda_graph.rs`: warmup forward, `begin_capture`, run the piece
//!    against the bucket buffers, `end_capture`.
//! 3. **Replay** (`PrefillGraphCache::try_replay_pre` / `_post`) — copy
//!    real-shape inputs into the bucket buffers, launch the captured
//!    graph, return a view into the bucket output buffer.
//! 4. **Engine wiring** (`crates/prelude-core/src/engine/model_runner/
//!    paged_mixed.rs::batch_mixed_paged` or equivalent) — when the
//!    request mix is pure-prefill and the cache is enabled, dispatch
//!    through `forward_with_prefill_graph` instead of the eager forward.
//!
//! Initial wiring will only support `Qwen3MoeModelForCausalLM`; other
//! architectures (`Qwen3ModelForCausalLM`, hybrid Qwen3.5) follow in
//! later changes once the cache shape is proven.

use std::collections::HashMap;

use prelude_core::config::EngineConfig;

/// Bucket sizes for prefill `num_tokens` capture, in increasing order.
/// Dispatch rounds the actual `num_tokens` up to the smallest enclosing
/// bucket. Values mirror vLLM's `compile_ranges_endpoints=[8192]` plus
/// a finer-grained low end since prelude's chunked prefill produces
/// smaller mixed steps than vLLM's prefill-only chunks.
pub(crate) const DEFAULT_PREFILL_BUCKETS: &[usize] = &[
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
];

/// Which half of a decoder layer this graph belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PiecePhase {
    /// `input_norm` + QKV + QK-norm + RoPE + (paged) KV write.
    PreAttn,
    /// O proj + post-attn norm + MoE / dense FFN.
    PostAttn,
}

/// Cache key: which layer, which phase, which bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphKey {
    pub layer_idx: usize,
    pub phase: PiecePhase,
    pub bucket: usize,
}

/// Per-bucket pre-allocated input/output tensors. Shared across layers
/// — layer i's post-piece writes `hidden`, layer i+1's pre-piece reads
/// it back. See module-level docs for the buffer-layout invariants the
/// captured graphs rely on.
///
/// (Skeleton — fields will be filled by `BucketBuffers::allocate` in the
/// follow-up commit that wires capture.)
#[allow(dead_code)]
pub(crate) struct BucketBuffers {
    pub bucket: usize,
    // hidden:    Tensor [bucket, hidden_size]
    // residual:  Tensor [bucket, hidden_size]
    // q:         Tensor [bucket, num_heads, head_dim]
    // k:         Tensor [bucket, num_kv_heads, head_dim]
    // v:         Tensor [bucket, num_kv_heads, head_dim]
    // attn_out:  Tensor [bucket, num_heads, head_dim]
    // slot_mapping: Tensor [bucket] (i64, indirection table written
    //               with real slot indices before each replay)
    // position_ids: Tensor [bucket] (u32, same pattern)
}

/// Captured `CudaGraph` for one (layer, phase, bucket) triple.
///
/// (Skeleton — the `cudarc::driver::CudaGraph` handle and the output
/// `Tensor` views land in the capture commit.)
#[allow(dead_code)]
pub(crate) struct CapturedPiece {
    pub key: GraphKey,
    // graph: cudarc::driver::CudaGraph,
    // output_view: Tensor,  // view into the bucket's hidden / q-k-v
}

/// Piecewise prefill graph cache. One per engine.
///
/// **Lifecycle**: constructed by the engine after model load, lives as
/// long as the model. `try_replay_pre` / `try_replay_post` are called
/// during prefill steps; if the bucket isn't captured yet, the call
/// returns `None` and the caller falls back to eager.
///
/// **Thread-safety**: owned by the GPU worker thread (mirrors
/// `DecodeGraphCache`); no `Arc<Mutex<…>>` wrapping needed.
pub(crate) struct PrefillGraphCache {
    /// Per-bucket pre-allocated buffer set (allocated lazily on first
    /// use of a bucket).
    buffers: HashMap<usize, BucketBuffers>,
    /// Captured graphs, keyed by `(layer_idx, phase, bucket)`.
    graphs: HashMap<GraphKey, CapturedPiece>,
    /// Sparse capture sizes; runtime rounds up to the smallest of these
    /// that fits the actual `num_tokens`.
    buckets: Vec<usize>,
    /// Master enable flag (mirrors `cuda_graph` + `model_supports_graph`
    /// in `DecodeGraphCache`).
    enabled: bool,
}

#[allow(dead_code)]
impl PrefillGraphCache {
    pub(crate) fn new(config: &EngineConfig, model_supports_graph: bool) -> Self {
        let enabled = config.runtime.cuda_graph && model_supports_graph;
        if enabled {
            tracing::info!(
                buckets = ?DEFAULT_PREFILL_BUCKETS,
                "PrefillGraphCache enabled — lazy capture on first use"
            );
        }
        Self {
            buffers: HashMap::new(),
            graphs: HashMap::new(),
            buckets: DEFAULT_PREFILL_BUCKETS.to_vec(),
            enabled,
        }
    }

    /// Round `num_tokens` up to the smallest captured bucket that fits.
    /// Returns `None` if the workload exceeds the largest bucket (caller
    /// falls back to eager).
    pub(crate) fn round_up_bucket(&self, num_tokens: usize) -> Option<usize> {
        self.buckets.iter().copied().find(|&b| b >= num_tokens)
    }

    /// Stub — returns `None` until the capture/replay machinery lands.
    /// Once wired, this either replays the captured pre-piece graph for
    /// layer `layer_idx` at `bucket`, or captures and replays on first
    /// encounter.
    pub(crate) fn try_replay_pre(
        &mut self,
        _layer_idx: usize,
        _bucket: usize,
        // hidden, residual inputs go here
    ) -> Option<()> {
        if !self.enabled {
            return None;
        }
        // TODO(yz): capture/replay impl
        None
    }

    /// Stub — symmetric for the post-piece.
    pub(crate) fn try_replay_post(
        &mut self,
        _layer_idx: usize,
        _bucket: usize,
        // attn_out, residual inputs go here
    ) -> Option<()> {
        if !self.enabled {
            return None;
        }
        // TODO(yz): capture/replay impl
        None
    }
}
