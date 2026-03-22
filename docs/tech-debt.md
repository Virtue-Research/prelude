# Prelude Technical Debt & Refactoring TODO

This document tracks code quality issues, technical debt, and areas needing redesign.

---

## Priority Legend

- **P0 (Critical)**: Blocks scaling, causes bugs, must fix immediately
- **P1 (High)**: Significant impact, fix in next sprint
- **P2 (Medium)**: Noticeable impact, fix next release
- **P3 (Low)**: Minor issues, cleanup when convenient

---

## 1. Dead Code & Cleanup

### ~~P2: Debug Static Variables (qwen3/mod.rs:14-19)~~ KEPT INTENTIONALLY
Six global `AtomicBool` debug switches used by debugging tools (`qwen3_batch_consistency`, `qwen3_layer_diff`):
```rust
static DEBUG_DISABLE_FAST_RMSNORM: AtomicBool = ...
static DEBUG_DISABLE_FUSED_QKNORM_ROPE: AtomicBool = ...
// ... 4 more
```
**Status**: These are intentionally kept for runtime debugging of numerical drift issues.
The `Ordering::Relaxed` atomic loads have negligible overhead.

### ~~P3: Unused Wrapper Methods (candle_engine.rs)~~ DONE
```rust
pub fn generate_sync_pub() { self.generate_sync() }
pub fn generate_batch_sync_pub() { self.generate_batch_sync() }
```
**Status**: Removed in commit 37622f7.

### P3: TODO Comments
- `candle_engine.rs:3368`: "TODO: proper per-sequence KV extraction for batched paged attention"
**Action**: Implement or create GitHub issue.

---

## 2. Duplicated Code

### ~~P0: ModelConfig/ModelVariant Dispatch Pattern~~ DONE
**Status**: Fixed in commit a02865e. Introduced three dispatch macros:
- `dispatch_config!`: for ModelConfig field access
- `dispatch_model!`: for ModelVariant method dispatch (all variants)
- `dispatch_generation_model!`: for generation-only models (Dense, Moe)

Reduced ~140 lines of repetitive dispatch code. Adding a new model now requires
updating the 3 macros instead of 15+ locations.

### ~~P1: Hardcoded max_new_tokens Clamping~~ DONE
**Status**: Simplified in commit 2fb6395. Removed redundant clamp in engine layer,
API layer now uses default of 4096 with no hardcoded upper limit. Engine auto-clamps
to `max_position_embeddings - prompt_len`.

### ~~P1: Batch Result Logging (scheduled_engine.rs)~~ DONE
**Status**: Fixed in commit 37622f7. Removed `format!()` from tracing debug fields.

### P2: Prefill KV Scattering Logic (candle_engine.rs:2010-2085)
~60 lines duplicated for cached vs non-cached block paths.

**Solution**: Extract to helper function with `start_pos` and `num_tokens` parameters.

---

## 3. Non-Scalable Design

### P0: Scheduler State Machine (scheduled_engine.rs)
**Problem**:
- ~800 line main loop with interlocking state variables
- `pending_gpu`, `pending_classify_gpu`, `pending_embed_gpu`, `pending_embed_gpu_2`
- Implicit state machine, hard to reason about

**Solution**:
```rust
enum SchedulerState {
    Idle,
    PendingGenerate { handle, inflight },
    PendingClassify { handle, inflight },
    PendingEmbed { slots: [Option<EmbedSlot>; 2] },
}

impl Scheduler {
    fn transition(&mut self, event: Event) -> State { ... }
}
```

### P1: Feature Flag Complexity
**Problem**: Exponential code paths from feature combinations:
- `#[cfg(feature = "flash-attn")]`
- `#[cfg(feature = "flash-attn-v3")]`
- `#[cfg(feature = "paged-attn")]`
- `#[cfg(feature = "cuda")]`

**Solution**: Use runtime dispatch or reduce to orthogonal feature sets.

### P1: Prefix Cache Coupling
**Problem**: Tightly integrated into `CandleEngine` with logic scattered across:
- `try_prefix_cache_match()`
- `try_prefix_cache_insert()`
- `try_prefix_cache_insert_paged()`
- `scatter_prefill_kv_to_paged_pool()`

**Solution**: Extract to pluggable `PrefixCacheProvider` trait.

### P1: Model Loader Registry (Task / Arch / Backend Split)
**Current state**:
- `load.rs` still owns the high-level loading flow
- model construction now lives behind a dedicated builder function instead of an inline giant `match`
- this is better, but still centralized: adding a new model still requires editing config parsing and builder registration paths

**Problem**:
- current model selection mixes three different axes in one place:
  - task kind (`generate`, `classify`, `embed`)
  - architecture kind (`qwen3`, `gemma3`, `qwen3_next`, `qwen3_5`, etc.)
  - backend / weight format (`safetensors`, `gguf`, future async loaders / quantized backends)
- this keeps `load.rs` and config parsing as the central choke point for every new model
- as the supported matrix grows, the code will keep expanding in a non-local way

**Target design**:
```rust
enum TaskKind {
    Generate,
    Classify,
    Embed,
}

enum ArchKind {
    Qwen3,
    Qwen3Moe,
    Gemma3,
    Qwen3Next,
    Qwen3_5,
}

enum BackendKind {
    Safetensors,
    Gguf,
}

struct ModelDescriptor {
    task: TaskKind,
    arch: ArchKind,
    backend: BackendKind,
}
```

```rust
trait ModelFactory {
    fn build(
        &self,
        descriptor: &ModelDescriptor,
        assets: ModelAssets,
    ) -> Result<ModelVariant, EngineError>;
}
```

**Migration plan**:
1. Keep `load.rs` as orchestration only: fetch config, weights, tokenizer, device
2. Parse model metadata into a `ModelDescriptor`
3. Route model construction through a registry/factory layer
4. Move per-architecture builders into separate files/modules
5. Make `GGUF` a backend concern instead of a peer to task/arch variants

**Why this matters**:
- adding a new model should be local to one architecture/backend module
- async loading can be introduced at the loader/assets layer without entangling runtime engine logic
- task semantics and backend capabilities become explicit instead of being inferred from enum variant names

---

## 4. Hardcoded Values

### P1: Extract to Configuration

| Value | Location | Description |
|-------|----------|-------------|
| `4096` | multiple | max_new_tokens clamp |
| `42` | multiple | Default random seed |
| `0.5`, `0.4` | scheduled_engine.rs:71-72 | EWMA smoothing factors |
| `1000.0` | scheduled_engine.rs:64 | Initial arrival rate |
| `10000.0` | scheduled_engine.rs:86 | Max instant rate cap |
| `128` | server/main.rs:313 | Default max_tokens |
| `2048` | server/main.rs:313 | Max tokens upper bound |
| `0.7` | server/main.rs:309 | Default temperature |
| `256` | gemma3/mod.rs:42 | query_pre_attn_scalar |
| `262144` | gemma3/mod.rs:46 | Gemma3 vocab_size default |

**Solution**:
```rust
pub struct InferenceConfig {
    pub max_new_tokens: u32,
    pub default_seed: u64,
    pub scheduler: SchedulerConfig,
    pub sampling: SamplingConfig,
}
```

---

## 5. Error Handling

### ~~P0: Panicking Unwraps~~ DONE
**Status**: Fixed in commit 4b9a54b. Replaced 13 `.unwrap()` calls in `candle_engine.rs`:
- `output_tokens.last().unwrap()` → `ok_or_else(|| EngineError::Internal(...))?`
- `cu_seqlens.last().unwrap()` → `unwrap_or(&0)`
- `layer_kvs.unwrap()` → match pattern

### ~~P1: Silent Error Suppression~~ PARTIALLY DONE
**Status**: Partially fixed in commit 9a3136a. Added logging for streaming tokenization errors.
Remaining: `let _ = cache.insert(...)` calls are intentional (cache failures are non-fatal).

### P2: Vague Error Messages
**Example**: `"model lock poisoned: {e}"` - doesn't explain recovery

**Solution**: Add context: request ID, operation, recovery suggestions.

---

## 6. Inconsistent Patterns

### P2: Request/Response Naming
**Current**:
- `GenerateRequest` → `GenerateResult`
- `ClassifyRequest` → `ClassifyResult` AND `ClassificationRequest` → `ClassificationResult`
- `EmbedRequest` → `EmbedResult` AND `EmbeddingRequest` → `EmbeddingResponse`

**Solution**: Standardize to `*Request` → `*Response` for HTTP types, `*Result` for internal types.

### P2: Error Type Mixing
**Current**:
- Some: `fn x() -> Result<T, EngineError>`
- Others: `fn x() -> candle_core::Result<T>`

**Solution**: Use `EngineError` at API boundary, `candle_core::Result` internally with `.map_err(candle_err)?`.

### P3: Model Architecture Pattern
**Current**:
- qwen3: `ModelForCausalLM`, `ForEmbedding`
- gemma3: `ForCausalLM` (missing "Model" prefix)

**Solution**: Standardize naming convention.

---

## 7. Performance Issues

### P1: Excessive Cloning (32+ occurrences in candle_engine.rs)
**Hot paths**:
- `candle_engine.rs:2013`: `cached_paged_blocks.clone()`
- `candle_engine.rs:1807, 1801`: Embedding vectors cloned per batch item
- `candle_engine.rs:1572`: Probability scores cloned

**Solution**: Use references, or `into_iter()` for owned values.

### P2: String Formatting in Logging
**Problem**: `format!("{:.2}")` evaluates even when log level disabled.

**Solution**: Use tracing field syntax or lazy formatting.

### P2: Lock Contention
**Problem**: Multiple mutex locks on `model`, `block_manager`, `prefix_cache`.

**Solution**: Consider lock-free structures or reduce critical section scope.

### P3: Atomic Flag Checking (qwen3/mod.rs)
**Problem**: 6 atomic loads per forward pass for debug flags.

**Solution**: Use compile-time feature flags or batch checks.

---

## 8. Missing Features

### P1: Request Tracing
- No distributed tracing support (OpenTelemetry)
- Request IDs not propagated to all error paths

### P2: Metrics
- No Prometheus metrics endpoint
- Missing: request latency histograms, batch size distributions, cache hit rates

### P2: Health Checks
- Basic `/health` endpoint exists
- Missing: readiness vs liveness, GPU memory status

### P3: Configuration Validation
- No startup validation of config values
- Silent failures on invalid configurations

---

## Refactoring Roadmap

### Phase 1: Critical Fixes (1-2 weeks)
- [ ] Extract hardcoded constants to config
- [ ] Fix critical unwrap() calls (top 10 by frequency)
- [ ] Add basic error context to EngineError

### Phase 2: Model Dispatch Refactor (2-3 weeks)
- [ ] Design ModelForward trait hierarchy
- [ ] Migrate ModelVariant to trait-based dispatch
- [ ] Reduce match arms from 8+ to 1

### Phase 3: Scheduler Redesign (2 weeks)
- [ ] Formalize scheduler state machine
- [ ] Extract task types to enum
- [ ] Add proper cancellation support

### Phase 4: Cleanup (ongoing)
- [ ] Remove debug static variables
- [ ] Standardize naming conventions
- [ ] Add missing tests for error paths

---

## Contributing

When fixing issues from this list:
1. Create a GitHub issue referencing this document
2. Add tests for the fix
3. Update this document when complete
