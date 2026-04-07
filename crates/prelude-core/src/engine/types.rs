use crate::types::{ClassifyRequest, EmbedRequest, GenerateRequest, TokenLogprobInfo};
use crate::tensor::Tensor;
use crate::engine::sampling::LogitsProcessor;

// ── Model dispatch ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskKind {
    Generate,
    Classify,
    Embed,
}

impl TaskKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Generate => "generation",
            Self::Classify => "classification",
            Self::Embed => "embeddings",
        }
    }

    pub(crate) fn endpoint_hint(self) -> &'static str {
        match self {
            Self::Generate => "/v1/completions or /v1/chat/completions",
            Self::Classify => "/v1/classify",
            Self::Embed => "/v1/embeddings",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TaskOverride {
    #[default]
    Auto,
    Generate,
    Classify,
    Embed,
}

impl TaskOverride {
    pub(crate) fn resolve(self, detected: TaskKind) -> TaskKind {
        match self {
            Self::Auto => detected,
            Self::Generate => TaskKind::Generate,
            Self::Classify => TaskKind::Classify,
            Self::Embed => TaskKind::Embed,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingPooling {
    #[default]
    LastToken,
    Mean,
    Cls,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingActivation {
    #[default]
    Identity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingNormalization {
    #[default]
    None,
    L2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingDenseLayerSpec {
    pub module_path: String,
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub activation: EmbeddingActivation,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbeddingSemantics {
    pub pooling: EmbeddingPooling,
    pub normalization: EmbeddingNormalization,
    pub dense_layers: Vec<EmbeddingDenseLayerSpec>,
}

impl EmbeddingSemantics {
    pub fn output_dim(&self, hidden_size: usize) -> usize {
        self.dense_layers
            .last()
            .map(|layer| layer.out_features)
            .unwrap_or(hidden_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsBackend {
    Safetensors,
    Gguf,
}

/// Common model configuration fields shared by all architectures.
/// Extracted once at parse time; engine code uses direct field access.
#[derive(Clone, Debug)]
pub struct CommonModelConfig {
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeCaps {
    pub supports_kv_cache: bool,
    pub supports_prefix_cache: bool,
    pub supports_paged_attn: bool,
    pub supports_varlen: bool,
    pub supports_deltanet: bool,
    pub supports_cuda_graph: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub task: TaskKind,
    pub arch_name: &'static str,
    pub backend: WeightsBackend,
}

// ── Batch execution planning ────────────────────────────────────────────

/// Execution dispatch for `Engine` direct-call path (non-scheduled).
///
/// Used by `Engine::plan_generate_batch` → `generate_prepared_batch`.
/// In production, `ScheduledEngine` routes requests to batch/continuous runtimes
/// before reaching this enum — multi-token decode on CPU goes through
/// `cpu_continuous_generation_loop`, not `MultiTokenDecode`.
///
/// This enum is primarily exercised by `PRELUDE_NO_SCHEDULER=1` (debug mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionKind {
    CudaPrefillOnly,
    CpuPrefillOnly,
    /// GPU multi-token decode via paged attention (Engine direct-call only).
    MultiTokenDecode,
}

/// A generation request that has already been prepared by the queue stage.
///
/// The scheduler constructs these once so the engine hot path can avoid
/// repeated tokenization and request normalization.
pub struct PreparedGenerateRequest {
    pub request_idx: usize,
    pub request: GenerateRequest,
    pub prompt_tokens: Vec<u32>,
    pub max_new: usize,
    pub is_greedy: bool,
    pub logits_processor: LogitsProcessor,
}

/// Logical prefix reuse candidate discovered during queue-side planning.
///
/// This is still pure planning data: no block IDs or tensor handles yet.
#[derive(Clone, Debug)]
pub(crate) struct PrefixReuseCandidate {
    pub(crate) common_prefix_tokens: Vec<u32>,
    pub(crate) min_prompt_len: usize,
}

/// Queue-side logical prefill plan.
#[derive(Clone, Debug)]
pub(crate) struct PrefillPlan {
    pub(crate) execution_kind: ExecutionKind,
    pub(crate) seq_lens: Vec<usize>,
    pub(crate) all_same_len: bool,
    pub(crate) all_greedy: bool,
    pub(crate) force_varlen: bool,
    pub(crate) prefix_reuse: Option<PrefixReuseCandidate>,
}

/// Read-only prefix reuse resolved against the current paged prefix cache state.
///
/// This still happens before any block manager mutation; the actual block
/// allocation stays in the execution layer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResolvedPrefixReuse {
    pub(crate) cached_len: usize,
    pub(crate) cached_block_ids: Vec<u32>,
}

/// Per-request logical block needs derived from sequence length + prefix reuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheAllocationPlanEntry {
    pub(crate) prompt_len: usize,
    pub(crate) suffix_len: usize,
    pub(crate) total_blocks: usize,
    pub(crate) new_blocks: usize,
}

/// Planned paged block usage before touching the block manager.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheAllocationPlan {
    pub(crate) prefix_reuse: ResolvedPrefixReuse,
    pub(crate) entries: Vec<CacheAllocationPlanEntry>,
    pub(crate) max_total_blocks: usize,
}

/// Logical decode plan. The decode path still starts with a prefill step,
/// so it reuses the same prefill metadata.
#[derive(Clone, Debug)]
pub(crate) struct DecodePlan {
    pub(crate) initial_prefill: PrefillPlan,
}

/// A prepared generation batch plan produced before execution.
#[derive(Clone, Debug)]
pub enum GenerateBatchPlan {
    Prefill(PrefillPlan),
    Decode(DecodePlan),
}

/// A prepared generation batch plus its queue-side logical plan.
pub struct PreparedGenerateBatch {
    pub plan: GenerateBatchPlan,
    pub items: Vec<PreparedGenerateRequest>,
}

// ── Batch item types ────────────────────────────────────────────────────

/// Pre-tokenized batch item for scheduler batch processing.
/// Generic over the request type (ClassifyRequest, EmbedRequest).
pub struct PreTokenizedBatchItem<R> {
    pub request_idx: usize,
    pub request: R,
    pub token_ids: Vec<Vec<u32>>,
    pub total_tokens: u32,
}

pub type PreTokenizedClassifyItem = PreTokenizedBatchItem<ClassifyRequest>;
pub type PreTokenizedEmbedItem = PreTokenizedBatchItem<EmbedRequest>;

/// Result of batch prefill for a single request.
/// Block table is retained (NOT freed) for subsequent streaming decode.
#[derive(Debug)]
pub struct BatchPrefillResult {
    pub first_token: u32,
    pub block_table: Vec<u32>,
    pub prompt_len: usize,
    pub prefill_ms: f32,
    /// DeltaNet pool slot for hybrid models (Qwen3.5, Qwen3-Next). None for non-hybrid models.
    pub deltanet_slot: Option<u32>,
    /// Logprobs for the first token (populated when request.logprobs is Some).
    pub first_token_logprobs: Option<TokenLogprobInfo>,
    /// Per-prompt-token logprobs (populated when request.prompt_logprobs is Some).
    pub prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>,
}

/// One sequence in a batched decode step (Q=1 per sequence).
pub struct BatchDecodeSeq<'a> {
    pub token: u32,
    pub position: usize,
    pub context_len: usize, // = position + 1
    pub block_table: &'a [u32],
    /// DeltaNet pool slot for hybrid models.
    pub deltanet_slot: Option<u32>,
}

/// Owned version of [`BatchDecodeSeq`] for sending through the GPU queue.
pub struct OwnedBatchDecodeSeq {
    pub token: u32,
    pub position: usize,
    pub context_len: usize,
    pub block_table: Vec<u32>,
    pub deltanet_slot: Option<u32>,
}

impl OwnedBatchDecodeSeq {
    pub fn as_borrowed(&self) -> BatchDecodeSeq<'_> {
        BatchDecodeSeq {
            token: self.token,
            position: self.position,
            context_len: self.context_len,
            block_table: &self.block_table,
            deltanet_slot: self.deltanet_slot,
        }
    }
}

// ── Paged KV pool ───────────────────────────────────────────────────────

pub struct PagedKvPool {
    pub(crate) key_caches: Vec<Tensor>,
    pub(crate) value_caches: Vec<Tensor>,
    pub(crate) key_caches_flash: Vec<Tensor>,
    pub(crate) value_caches_flash: Vec<Tensor>,
    pub block_size: usize,
}

impl PagedKvPool {
    /// Returns the active key caches for the compiled attention backend.
    pub fn active_key_caches(&self) -> &[Tensor] {
        &self.key_caches_flash
    }

    /// Returns the active value caches for the compiled attention backend.
    pub fn active_value_caches(&self) -> &[Tensor] {
        &self.value_caches_flash
    }
}
