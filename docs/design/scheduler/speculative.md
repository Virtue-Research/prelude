# Speculative Decoding

Back to [main design doc](README.md).


Engine-level orchestration for draft-then-verify decoding. Increases decode throughput
by drafting N candidate tokens cheaply, then verifying them in one target model forward pass.

## File Layout

```
prelude-core/src/
├── engine/
│   ├── speculative/                           # Speculative decoding orchestration
│   │   ├── mod.rs                             # SpecDecodeRunner: draft → verify → accept loop
│   │   ├── proposer.rs                        # trait DraftProposer + concrete implementations
│   │   ├── rejection.rs                       # Rejection sampling (strict, probabilistic)
│   │   └── tree.rs                            # Tree attention mask construction (EAGLE/Medusa)
│   └── model_runner/
│       └── mod.rs                             # ModelRunner::execute — handles spec decode batches
```

## Architecture

```
Engine main loop (run::ar)
    │
    ▼
SpecDecodeRunner
    │
    ├── 1. Draft: proposer.propose(hidden_states, last_token) → draft_token_ids
    │       (small model / EAGLE head / n-gram lookup — cheap, N tokens)
    │
    ├── 2. Verify: target model forward with all N+1 positions in one pass
    │       (paged_attention with MaskType::Custom for tree, or Causal for linear)
    │
    ├── 3. Accept: rejection_sample(target_logits, draft_logits) → k accepted tokens
    │       (strict: stop at first mismatch; probabilistic: ratio-based)
    │
    └── 4. Update: advance k tokens, recompute slot_mapping for next step
            (no KV cache rollback — rejected slots use PADDING_SLOT_ID = -1)
```

## DraftProposer Trait

```rust
// prelude-core/src/engine/speculative/proposer.rs

/// Minimal interface for draft token generation. All methods return draft token IDs.
/// Implementations are pluggable — engine doesn't know which method is used.
trait DraftProposer: Send + Sync {
    /// Generate N draft tokens given the target model's last hidden states and token.
    /// Returns [batch_size, num_draft_tokens] tensor of token IDs.
    fn propose(
        &mut self,
        target_hidden_states: &Tensor,   // from target model's last forward
        last_token_ids: &Tensor,         // [batch_size] last verified tokens
        positions: &Tensor,              // [batch_size] current positions
        num_draft_tokens: usize,
        ops: &Ops,
    ) -> Result<Tensor>;

    /// For tree-based methods (EAGLE/Medusa): generate a tree of candidates.
    /// Returns one tensor per tree level. Default: single-path (no tree).
    fn propose_tree(
        &mut self,
        target_hidden_states: &Tensor,
        last_token_ids: &Tensor,
        positions: &Tensor,
        tree_config: &TreeConfig,
        ops: &Ops,
    ) -> Result<Vec<Tensor>> {
        // Default: linear chain, wrap propose() output as single-path tree
        let drafts = self.propose(target_hidden_states, last_token_ids, positions,
                                   tree_config.total_tokens, ops)?;
        Ok(vec![drafts])
    }

    /// Name for logging/metrics.
    fn name(&self) -> &str;
}
```

### Concrete Implementations

```rust
// prelude-core/src/engine/speculative/proposer.rs

/// Small autoregressive model as draft. Separate weights, shared Ops.
struct DraftModelProposer {
    model: Box<dyn Model>,   // small model (e.g., Llama-68M)
    // Uses the same Ops as target — attention/GEMM dispatched to same device
}

/// EAGLE: target model's hidden states → lightweight head → draft tokens.
/// No separate model weights — just a small projection head.
struct EagleProposer {
    head: EagleHead,         // 1-2 transformer layers + LM head
    // Receives target hidden states, doesn't re-embed tokens
}

/// N-gram: lookup draft tokens from prompt/output history. No model, no GPU.
struct NgramProposer {
    window_size: usize,      // match window (e.g., 4-12)
    // Pure CPU lookup, zero GPU cost
}

/// Medusa: multiple parallel heads predict tokens at different positions.
struct MedusaProposer {
    heads: Vec<MedusaHead>,  // one head per speculative position
    // All heads run in parallel on same hidden states
}
```

**Design choice (learn from vLLM, avoid SGLang):**
- vLLM: clean `SpecDecodeBaseProposer` with single `propose()` interface. Each impl is a standalone class.
- SGLang: `EAGLEWorker` inherits from `TpModelWorker`, mixing proposer logic with worker lifecycle.
  Harder to test, harder to swap.
- **We follow vLLM**: proposer is a pure function object. It takes inputs, returns draft tokens.
  No worker lifecycle, no scheduler coupling.

## Rejection Sampling

```rust
// prelude-core/src/engine/speculative/rejection.rs

/// Rejection sampling strategy. Determines how many draft tokens are accepted.
enum RejectionMethod {
    /// Stop at first mismatch. Simple, deterministic.
    /// target_token[i] != draft_token[i] → reject i and all after.
    Strict,
    /// Accept with probability min(1, p_target / p_draft).
    /// Can accept tokens even when draft != target (if target prob is high enough).
    /// Better acceptance rate, slightly more compute.
    Probabilistic,
}

/// Compare target logits against draft tokens, return number of accepted tokens per request.
fn rejection_sample(
    target_logits: &Tensor,      // [batch, num_draft+1, vocab]
    draft_token_ids: &Tensor,    // [batch, num_draft]
    draft_logits: Option<&Tensor>, // [batch, num_draft, vocab] (for probabilistic)
    method: RejectionMethod,
    sampling_params: &SamplingParams,
) -> Result<RejectionResult>;

struct RejectionResult {
    /// Number of accepted tokens per request.
    num_accepted: Vec<usize>,           // [batch]
    /// Accepted token IDs (verified + bonus token from target).
    accepted_token_ids: Vec<Vec<u32>>,  // [batch][0..num_accepted+1]
}
```

**Design choice (learn from vLLM):**
- vLLM has three rejection methods (strict, probabilistic, synthetic) as separate Triton kernels.
  Clean separation — easy to benchmark and swap.
- SGLang bakes verification into `eagle_info.verify()`, tightly coupled with batch state.
- **We follow vLLM**: rejection sampling is a standalone function. Input: logits. Output: accepted count.
  No batch state mutation.

## Tree Attention (EAGLE/Medusa)

```rust
// prelude-core/src/engine/speculative/tree.rs

/// Tree structure for multi-path speculation.
/// Each node is a candidate token; children are alternative continuations.
struct TreeConfig {
    /// Tree topology: list of (parent_index) per node.
    /// Root is implicit (index 0 = last verified token).
    /// Example: [(0,), (0,), (1,), (1,), (2,)] = 2 branches from root, 
    ///          each with 2-3 children.
    parent_indices: Vec<usize>,
    total_tokens: usize,
}

/// Build attention mask for tree verification.
/// Each token attends to its ancestors in the tree, not a simple causal pattern.
fn build_tree_mask(tree: &TreeConfig) -> Tensor {
    // Output: [tree_len, tree_len], values 0.0 (attend) or -inf (mask)
    // Passed to paged_attention as MaskType::Custom(mask)
    let mut mask = Tensor::full(&[tree.total_tokens + 1, tree.total_tokens + 1], f32::NEG_INFINITY);
    for (child, &parent) in tree.parent_indices.iter().enumerate() {
        // child attends to parent and all ancestors (transitive)
        mask[child + 1][parent] = 0.0;
        // ... propagate ancestor connections
    }
    mask
}
```

**Design choice:**
- vLLM: pre-computed static tree bias, dynamically sliced per drafting level. Predictable latency.
- SGLang: runtime tree construction every verification pass, with compression modes (bitpacking).
  More flexible but less predictable.
- **We follow vLLM**: pre-compute tree mask once at config time. Slice per level during drafting.
  `MaskType::Custom(Tensor)` in `AttentionOps` handles the rest — no new ops needed.

## KV Cache: No Rollback

```rust
// prelude-core/src/engine/speculative/mod.rs — inside SpecDecodeRunner

/// After rejection: rejected tokens' slots already have PADDING_SLOT_ID = -1.
/// reshape_and_cache skips -1 slots. No explicit rollback needed.
///
/// Next step: recompute slot_mapping from the accepted position.
/// The accepted tokens' KV entries are valid in cache. Rejected entries
/// were never written (or written to padding slots).
fn update_after_rejection(
    &mut self,
    result: &RejectionResult,
    kv: &mut PagedKvCtx,
    scheduler: &mut ArScheduler,
) {
    for (req_idx, &num_accepted) in result.num_accepted.iter().enumerate() {
        // Advance request by num_accepted + 1 tokens (accepted + bonus)
        scheduler.advance_request(req_idx, num_accepted + 1);
        // Slot mapping recomputed from new positions — no KV cache modification
    }
}
```

**Design choice (both vLLM and SGLang agree):**
- Neither does KV cache rollback. vLLM uses slot mapping recomputation.
  SGLang uses paged cache duplication for tree branches.
- **We use slot mapping recomputation** (simpler). `reshape_and_cache` with `PADDING_SLOT_ID = -1`
  already skips invalid slots. No new mechanism needed.

## Configuration

```rust
// prelude-core/src/engine/config.rs

struct SpeculativeConfig {
    /// Draft method.
    method: SpecMethod,
    /// Number of draft tokens per step.
    num_draft_tokens: usize,        // e.g., 3-7
    /// Rejection sampling strategy.
    rejection_method: RejectionMethod,
    /// Tree topology for EAGLE/Medusa (None for linear drafting).
    tree_config: Option<TreeConfig>,
}

enum SpecMethod {
    /// Separate small draft model.
    DraftModel { model_path: String },
    /// EAGLE: hidden state reuse + lightweight head.
    Eagle { head_path: String },
    /// EAGLE-3: improved EAGLE with better tree topology.
    Eagle3 { head_path: String },
    /// N-gram: prompt/output history lookup, no model.
    Ngram { window_size: usize },
    /// Medusa: parallel heads at different positions.
    Medusa { head_path: String },
}
```

**Design choice (learn from vLLM, avoid SGLang):**
- vLLM: single `--speculative-config` JSON dict. Clean, one argument.
- SGLang: 15+ scattered `--speculative-*` CLI args. Hard to maintain.
- **We follow vLLM**: single config struct, one CLI argument.

## Scheduler Integration

The scheduler doesn't know about speculative decoding internals. It only sees:
- `request.num_computed_tokens` — advanced by `num_accepted + 1` after verification
- `request.spec_token_ids` — optional, set by engine for token budget tracking

The `SpecDecodeRunner` sits between scheduler and model runner:

```
ArScheduler.step()
    → ScheduledBatch (same as non-spec decode)
        → SpecDecodeRunner.execute(batch)
            → proposer.propose()          # draft N tokens
            → model_runner.execute()       # verify all N+1 positions
            → rejection_sample()           # accept k ≤ N
            → StepResult (k+1 tokens per request)
        → ArScheduler.update(result)
```

The scheduler's token budget accounts for speculative tokens:
`max_tokens_per_step` includes draft tokens during verification passes.

## Impact on Ops Design

| Concern | Where it lives | Impact on ops |
|---------|---------------|---------------|
| Draft model forward | SpecDecodeRunner | None — same `Ops` as target |
| Target verification | SpecDecodeRunner | None — chunked prefill via `paged_attention` |
| Tree attention mask | `MaskType::Custom(Tensor)` | Already in `AttentionOps` |
| KV cache for rejected tokens | `slot_mapping` with -1 | `reshape_and_cache` skips -1 slots |
| Rejection sampling | `rejection.rs` | Not an op trait concern (pure CPU logic) |
| EAGLE hidden states | Model exposes last hidden states | Model-level, no ops changes |
