# Constrained Decoding (Structured Output)

Back to [main design doc](README.md).


Grammar-based token filtering for structured output. Forces model output to conform to
JSON schema, regex, EBNF grammar, or choice list. Applied as a logits mask before sampling.

## File Layout

```
prelude-core/src/
├── engine/
│   ├── sampling/                              # Sampling pipeline
│   │   ├── mod.rs                             # Sampler: logits → token IDs
│   │   ├── grammar.rs                         # GrammarManager: async compile + bitmask fill
│   │   └── logits_processor.rs                # LogitsProcessor trait (grammar is one impl)
│   └── config.rs                              # StructuredOutputConfig in request params
```

## Architecture

```
Request arrives with constraint (JSON schema / regex / grammar)
    │
    ▼
GrammarManager.compile(constraint)             # async, overlaps with prior step's GPU work
    │  ThreadPool: compile grammar → FSM / GrammarMatcher
    │  Request waits until compilation completes
    ▼
Scheduler schedules request normally
    │
    ▼
Executor: model.forward() → logits          # GPU compute
    │
    ▼
GrammarManager.fill_bitmask(logits, batch)     # CPU: per-request bitmask of allowed tokens
    │  Each request's grammar state → bitmask[req][vocab_size]
    │  Parallel fill for large batches (> 128 requests)
    ▼
Sampler: logits.masked_fill_(~bitmask, -inf)   # GPU: zero out disallowed tokens
    │  Then standard temperature / top_k / top_p / sample
    ▼
GrammarManager.accept_tokens(sampled_ids)      # Advance FSM state per request
    │
    ▼
Next step (or finish if grammar reaches accept state)
```

## Grammar Backend Trait

```rust
// prelude-core/src/engine/sampling/grammar.rs

/// Pluggable grammar engine. Compiles constraints into token-level matchers.
/// Default implementation uses llguidance (pure Rust, MIT license).
trait GrammarBackend: Send + Sync {
    /// Compile a constraint specification into a grammar matcher.
    /// This can be expensive (10-100ms for complex JSON schemas).
    /// Called asynchronously on a thread pool.
    fn compile(&self, spec: &ConstraintSpec) -> Result<Box<dyn GrammarMatcher>>;

    /// Allocate a bitmask tensor for batch_size requests.
    /// Shape: [batch_size, ceil(vocab_size / 32)] as int32 (packed bits).
    fn allocate_bitmask(&self, batch_size: usize, vocab_size: usize) -> Tensor;
}

/// Per-request grammar state. Tracks FSM position, fills bitmask.
trait GrammarMatcher: Send {
    /// Fill bitmask row at `batch_index` with allowed next tokens.
    fn fill_bitmask(&self, bitmask: &mut Tensor, batch_index: usize);

    /// Accept token and advance FSM state.
    fn accept_token(&mut self, token_id: u32) -> bool;

    /// Rollback FSM state by k tokens (for speculative decoding rejection).
    fn rollback(&mut self, k: usize);

    /// True if grammar has reached an accept state (output is complete).
    fn is_terminated(&self) -> bool;

    /// Reset to initial state (for reuse across requests).
    fn reset(&mut self);
}

/// What the user specifies in the request.
enum ConstraintSpec {
    /// JSON schema (string or parsed). Most common for API serving.
    JsonSchema(String),
    /// Regex pattern. For structured fields (phone, email, etc.).
    Regex(String),
    /// EBNF grammar. For programming languages, custom formats.
    Grammar(String),
    /// Fixed choice list. Simplest constraint.
    Choice(Vec<String>),
}
```

**Design choice (learn from both):**
- vLLM: `StructuredOutputBackend` + `StructuredOutputGrammar` — clean abstraction, 4 pluggable backends.
- SGLang: `BaseGrammarBackend` + `BaseGrammarObject` — similar abstraction, with jump-forward optimization.
- **We take**: vLLM's clean interface names + SGLang's `rollback()` method (needed for spec decode).
  Keep the trait minimal — `fill_bitmask` + `accept_token` + `rollback` + `is_terminated`.

### Concrete Backend

```rust
// prelude-core/src/engine/sampling/grammar.rs

/// LLGuidance: pure Rust constrained decoding engine (MIT license, Microsoft).
/// Earley parser + derivative-based regex. ~50μs per token for 128k tokenizers.
/// Zero startup overhead (no pre-computation needed).
/// Powers OpenAI Structured Outputs. Integrated in llama.cpp, vLLM, SGLang.
///
/// Direct Rust crate dependency — no C++ FFI, no third_party/ submodule.
struct LlguidanceBackend {
    tokenizer: llguidance::LLTokenizer,
    cache: LruCache<String, llguidance::CompiledGrammar>,
}
```

## GrammarManager

```rust
// prelude-core/src/engine/sampling/grammar.rs

/// Manages grammar compilation and bitmask filling for a batch of requests.
/// Owned by the engine, called between model forward and sampling.
struct GrammarManager {
    backend: Box<dyn GrammarBackend>,
    /// Thread pool for async grammar compilation.
    compile_pool: ThreadPool,
    /// Pre-allocated bitmask tensor (reused across steps).
    bitmask: Tensor,  // [max_batch_size, ceil(vocab_size / 32)]
    /// Active grammar matchers, keyed by request ID.
    matchers: HashMap<RequestId, Box<dyn GrammarMatcher>>,
}

impl GrammarManager {
    /// Submit grammar compilation for a new request. Non-blocking.
    /// Returns immediately; compilation happens on thread pool.
    fn compile_async(&mut self, req_id: RequestId, spec: ConstraintSpec) {
        let backend = &self.backend;
        self.compile_pool.spawn(move || backend.compile(&spec));
    }

    /// Fill bitmask for all active grammar requests in the batch.
    /// Called after model.forward(), before sampling.
    fn fill_bitmask(&self, batch: &ScheduledBatch) -> &Tensor {
        // For each request with active grammar:
        //   matcher.fill_bitmask(&mut self.bitmask, batch_index)
        // For requests without grammar:
        //   bitmask[batch_index] = all 1s (no constraint)
        //
        // Parallel fill when batch > 128 requests (like vLLM).
        &self.bitmask
    }

    /// Accept sampled tokens, advance FSM state.
    fn accept_tokens(&mut self, sampled: &[(RequestId, u32)]) {
        for (req_id, token_id) in sampled {
            if let Some(matcher) = self.matchers.get_mut(req_id) {
                matcher.accept_token(*token_id);
                if matcher.is_terminated() {
                    self.matchers.remove(req_id);
                }
            }
        }
    }

    /// Rollback grammar state for rejected speculative tokens.
    fn rollback(&mut self, req_id: &RequestId, num_tokens: usize) {
        if let Some(matcher) = self.matchers.get_mut(req_id) {
            matcher.rollback(num_tokens);
        }
    }
}
```

## Integration with Sampling Pipeline

```
// prelude-core/src/engine/executor/mod.rs

Executor::execute(&batch)
    │
    ├── ops.begin_forward()
    ├── model.forward() → logits             # GPU
    ├── ops.end_forward()
    │
    ├── grammar_manager.fill_bitmask(&batch)  # CPU (parallel for large batches)
    │
    ├── apply_bitmask(&logits, &bitmask)      # GPU: logits[~bitmask] = -inf
    │
    ├── sample(&logits, &sampling_params)     # GPU: temperature → top_k → top_p → sample
    │
    ├── grammar_manager.accept_tokens(&sampled)  # CPU: advance FSM
    │
    └── → StepResult
```

**Key: async compilation overlaps with GPU compute.**
When a new request with a grammar arrives, compilation starts on the thread pool.
The request isn't scheduled until compilation completes. This means:
- Compilation latency is hidden behind other requests' GPU work
- No blocking on the hot path

## Batching: Heterogeneous Constraints

Each request can have a different constraint (or no constraint). Within the same batch:

```
Request 0: JSON schema {"name": str, "age": int}
Request 1: regex r"\d{3}-\d{4}"
Request 2: no constraint
Request 3: choice ["yes", "no", "maybe"]
```

The bitmask tensor has one row per request. `fill_bitmask` fills each row independently:
- Request 0: llguidance JSON matcher → row 0
- Request 1: regex FSM → row 1
- Request 2: all 1s → row 2 (no filtering)
- Request 3: only tokens starting "yes"/"no"/"maybe" → row 3

This is standard in both vLLM and SGLang. No special handling needed.

## Speculative Decoding Integration

When speculative decoding is active, draft tokens must also satisfy the grammar.
Two options:

1. **Validate drafts against grammar** (vLLM approach):
   After proposer generates draft tokens, validate each against the grammar.
   Invalid drafts are replaced with -1 (padding). Only valid drafts go to verification.

2. **Rollback on rejection** (both systems):
   If verification rejects draft tokens, rollback the grammar FSM state
   by the number of rejected tokens.

```rust
// prelude-core/src/engine/speculative/mod.rs

// After rejection sampling:
for (req_id, num_rejected) in rejections {
    grammar_manager.rollback(&req_id, num_rejected);
}
```

## Configuration

```rust
// prelude-core/src/engine/config.rs — part of request-level SamplingParams

struct SamplingParams {
    // ── Sampling method ──────────────────────────────
    pub temperature: f32,              // default: 1.0. ≤ 1e-7 → argmax (greedy)
    pub top_p: f32,                    // default: 1.0. Nucleus sampling threshold
    pub top_k: Option<u32>,           // None = disabled. Filter to top-k before top-p
    pub min_p: Option<f32>,           // None = disabled. Drop tokens with p < min_p * p_max

    // ── Logits processors (applied before sampling) ──
    pub repetition_penalty: Option<f32>,  // default: None. Divides/multiplies logits of seen tokens
    pub presence_penalty: Option<f32>,    // default: None. Subtracts fixed value for seen tokens
    pub frequency_penalty: Option<f32>,   // default: None. Subtracts count * penalty for seen tokens

    // ── Structured output ────────────────────────────
    pub constraint: Option<ConstraintSpec>,

    pub seed: Option<u64>,
}

// Engine-level config for grammar backend
struct GrammarConfig {
    /// Backend to use. Default: llguidance (pure Rust, always available).
    pub backend: String,           // "llguidance" | "auto"
    /// Max concurrent grammar compilations.
    pub compile_workers: usize,    // default: num_cpus / 2
    /// Grammar compilation cache size.
    pub cache_size_mb: usize,      // default: 256
}
```

**Design choice (learn from vLLM, avoid SGLang):**
- vLLM: `StructuredOutputsParams` dataclass with mutual exclusivity validation. Clean.
- SGLang: scattered `json_schema`, `regex`, `ebnf` as flat fields. No validation.
- **We use**: single `ConstraintSpec` enum — compiler enforces mutual exclusivity.
  One field in `SamplingParams`, not four.

## Full Sampling Pipeline

The sampling pipeline is a sequence of logits processors followed by a sampling method.
All processors conform to the OpenAI API specification — no custom behavior.

```
model.forward() → logits [batch, vocab]
    │
    ▼
1. Penalty processors (CPU, per-request)
    │  repetition_penalty: logits[t] /= penalty  (if logits[t] > 0, else *= penalty)
    │  presence_penalty:   logits[t] -= penalty   (for each unique seen token t)
    │  frequency_penalty:  logits[t] -= count[t] * penalty
    │
    ▼
2. Grammar bitmask (CPU → GPU)
    │  logits[~bitmask] = -inf
    │
    ▼
3. Temperature scaling
    │  logits /= temperature
    │
    ▼
4. Sampling method (one of):
    │  argmax          — temperature ≤ 1e-7, deterministic
    │  top_k → top_p   — filter to top-k, then nucleus, then multinomial
    │  top_p           — nucleus sampling, then multinomial
    │  min_p           — drop tokens with p < min_p * p_max, then multinomial
    │
    ▼
5. Grammar accept (CPU)
    │  advance FSM state per sampled token
    │
    ▼
sampled token IDs
```

### Penalty Semantics (OpenAI API spec)

All penalties operate on the set of tokens already generated by the request (`output_tokens`).

**repetition_penalty** (llama.cpp / HuggingFace convention, not OpenAI):
```
for token in set(output_tokens):
    if logits[token] > 0: logits[token] /= penalty
    else:                 logits[token] *= penalty
```
Typical range: 1.0 (disabled) to 1.3. Values < 1.0 encourage repetition.

**presence_penalty** (OpenAI convention):
```
for token in set(output_tokens):
    logits[token] -= penalty
```
Range: -2.0 to 2.0. Positive = discourage, negative = encourage reuse.

**frequency_penalty** (OpenAI convention):
```
for token in output_tokens:
    count[token] += 1
for token, c in count:
    logits[token] -= c * penalty
```
Range: -2.0 to 2.0. Scales with how often a token appeared.

These are trivial per-token logit modifications (~20 lines each). No GPU kernel needed —
the penalty loop touches at most `len(output_tokens)` entries in a vocab-sized logits tensor.
Applied on CPU before any GPU sampling.

### min_p Sampling

Filters tokens whose probability is below `min_p * max_probability`:

```
probs = softmax(logits / temperature)
threshold = min_p * max(probs)
probs[probs < threshold] = 0
sample from renormalized probs
```

Advantage over top-p: adapts to the model's confidence. When the model is confident
(one token dominates), min_p filters aggressively. When uncertain (flat distribution),
min_p preserves more candidates. This avoids top-p's failure mode of including too many
low-quality tokens when the model is confident.

### GPU Sampling (Future)

FlashInfer provides GPU-accelerated sampling kernels, already compiled in `prelude-flashinfer`:
- `top_k_sampling_from_probs`
- `top_p_sampling_from_probs`
- `top_k_top_p_sampling_from_probs`
- `min_p_sampling_from_probs`
- `chain_speculative_sampling`

Currently unused — sampling runs on CPU via Rust `rand`. The CPU path is sufficient for
typical batch sizes (< 256 requests). GPU sampling becomes worthwhile when:
- Batch size > 256 (sorting vocab_size=150K per request gets expensive)
- Speculative decoding with tree attention (many candidates to sample/verify per step)

Integration path: expose FlashInfer sampling functions via FFI in `prelude-flashinfer`,
call from `Sampler` when batch size exceeds threshold. Penalties remain on CPU regardless
(they touch few entries and need access to per-request `output_tokens`).
