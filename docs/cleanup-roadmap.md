# Codebase Cleanup Roadmap

This roadmap tracks the cleanup pass started after the Qwen3.5 fast-path work.
The goal is to reduce accidental complexity without changing model behavior or
performance-critical execution paths.

## Principles

- Preserve behavior first. Small refactors must stay covered by `cargo check`
  or focused tests before moving to larger reshapes.
- Move model facts to model/config types. Scheduler, cache, and executor code
  should not carry model-specific hard-coded layer patterns.
- Keep hot paths explicit. Performance-sensitive fallback chains should be
  documented as policy, not hidden as ad hoc conditionals.
- Prefer seams that match ownership: model metadata in `models`, scheduling
  policy in `scheduler`, runtime cache mechanics in `cache`, backend kernels in
  backend crates.

## Current Audit

### Repository Layout

- `crates/prelude-core`: central engine, scheduler, model registry, model
  implementations, tensor abstraction, and shared ops traits. Highest cleanup
  leverage and highest regression risk.
- `crates/prelude-cuda`: CUDA ops, attention backends, graph capture, kernel
  build glue, and backend-specific examples/tests. High performance sensitivity;
  avoid stylistic churn without benchmarks.
- `crates/prelude-cpu`: CPU ops, quantization, attention, OneDNN integration.
  Large but mostly isolated from recent Qwen3.5 serving work.
- `crates/prelude-server`: API routes, SSE, auth, request adaptation. Good
  candidate for request/response normalization cleanup after core stabilizes.
- `tests` and `benchmark`: valuable behavioral harnesses, but include debug
  scripts and duplicated endpoint helpers that should be separated from
  regression tests.
- `third_party` and vendored backend folders: leave untouched except for build
  integration boundaries.

### Folder Inventory

- `prelude-core/src/engine`: loading, execution, sampling, model runner, AR
  runtime, tokenizer, and weight loading. Cleanup should split orchestration
  from resource ownership. `engine/loading.rs` also contains embedding module
  discovery and GGUF tokenizer heuristics that should become focused helpers.
- `prelude-core/src/engine/model_runner`: paged prefill/decode/mixed paths and
  direct generation flows. Cleanup should keep GPU hot paths stable while
  extracting repeated DeltaNet slot and block-table preparation.
- `prelude-core/src/engine/run`: AR serving loop and request lifecycle. This is
  the largest operational knot: prepare, schedule, attach prefix cache, execute,
  emit, and free all live together.
- `prelude-core/src/models`: registry plus model implementations. Metadata,
  config parsing, layer implementations, and GGUF support should not live in one
  file per complex model.
- `prelude-core/src/models/commons`: reusable linear, activation, embedding, and
  attention context types. This is a good landing zone for shared model
  building blocks, but not for architecture-specific policy.
- `prelude-core/src/scheduler`: admission, scheduler state, tests, and cache
  components. Cleanup should separate scheduling policy from prefix-cache
  storage details.
- `prelude-core/src/scheduler/components/cache`: block manager, prefix trie,
  prefix resources, DeltaNet pool, and cache manager. Ownership/refcounting
  semantics need named types and focused tests.
- `prelude-core/src/ops/traits`: backend capability surface. Cleanup should
  avoid adding model-specific methods here unless they are truly backend
  primitives.
- `prelude-core/src/tensor`: tensor and quantized abstractions. Treat as a
  lower-level boundary; avoid cleanup churn unless it simplifies backend use.
- `prelude-cuda/src/attn`: FlashInfer, FA4, GDN prefill/decode, causal conv and
  KDA bindings. Needs dispatch/capability cleanup after benchmarks are in place.
- `prelude-cuda/src/ops`: CUDA op wrappers. Keep fallback order explicit and
  documented; avoid hidden model checks.
- `prelude-cuda/src/kernels`: in-repo CUDA kernels. Mostly leave untouched
  during Rust-level cleanup except for build integration.
- `prelude-cuda/prelude-kernelbuild`: kernel build/archive/NVCC orchestration.
  Good target for isolated cleanup because it has a small check surface.
- `prelude-cpu/src/ops`: CPU attention, GEMM, RMSNorm, RoPE, quantization, and
  buffer utilities. Defer until core/server cleanup is stable.
- `prelude-cpu/src/onednn`: OneDNN integration. Requires initialized submodule
  for validation.
- `prelude-server/src/routes`: OpenAI-compatible API adapters. Cleanup should
  consolidate duplicated request validation and response shaping.
- `tests/accuracy`, model-specific `tests/*`, and `benchmark`: separate
  regression fixtures from debug scripts and long-running benchmark utilities.

### Model Inventory

- `qwen3`: mature dense baseline. Keep as the reference for straightforward
  `CommonModelConfig` construction and registry shape.
- `qwen3_moe`: MoE model with mostly standard attention. Good candidate for
  sharing config/registry patterns with `qwen3`.
- `qwen3_5`: hybrid attention plus DeltaNet plus MoE/GGUF support. Highest
  cleanup priority; split metadata first, then GGUF, then attention/DeltaNet.
- `qwen3_next`: hybrid DeltaNet model with similar recurrent-state metadata.
  Keep helper semantics aligned with Qwen3.5 to avoid duplicate scheduler/cache
  special cases.
- `gemma3`: supports generation/classification/embedding layouts. Cleanup
  should focus on registry/build context boundaries, not changing model math.
- `gemma4`: similar registry/config cleanup surface to Gemma3, with fewer
  recent serving-path changes.

### High-Risk Modules

- `crates/prelude-core/src/models/qwen3_5.rs`: ~4k lines mixing config parsing,
  full attention, DeltaNet, MoE, pooled state paths, GGUF loading, and registry
  metadata. Needs staged extraction.
- `crates/prelude-core/src/engine/run/ar.rs`: ~2k lines mixing AR orchestration,
  preparation, prefix cache attach/insert, resource cleanup, and result
  streaming. Needs phase extraction around prefix cache and resource lifetime.
- `crates/prelude-core/src/scheduler/admission.rs`: scheduling policy now
  carries hybrid prefix-cache boundary logic. Move reusable-boundary concepts
  into cache/prefix planning types.
- `crates/prelude-core/src/scheduler/components/cache/prefix_cache.rs`: prefix
  trie, exact partial entries, DeltaNet state, eviction, and paged block
  ownership are tightly coupled. Needs clearer entry/resource abstractions.
- `crates/prelude-cuda/src/ops/gemm.rs` and attention backend selection: fallback
  policy is important but encoded in backend-specific conditionals. Needs a
  declarative capability/dispatch layer before adding more models.

### Model Layer Cleanup Targets

- `qwen3_5`: split into `config`, `attention`, `deltanet`, `mlp/moe`, `model`,
  `gguf`, and `meta` modules. First extract metadata/config helpers because
  they are easiest to verify.
- `qwen3_next`: align hybrid DeltaNet metadata helpers with Qwen3.5 so scheduler
  and cache logic see the same abstractions.
- `qwen3`, `qwen3_moe`, `gemma3`, `gemma4`: reduce repeated
  `CommonModelConfig` construction and keep model registry blocks thin.
- GGUF support: avoid returning dead metadata from registry results; only expose
  fields consumed by engine loading.

## Prioritized Plan

1. Foundation metadata cleanup.
   - Centralize `CommonModelConfig` construction.
   - Centralize hybrid DeltaNet pool-shape construction.
   - Remove unused GGUF load fields and stray review/dead variables.
   - Verification: core check and kernelbuild check.

2. Qwen3.5 module split.
   - Extract config/meta helpers without touching forward behavior.
   - Extract GGUF support into a sibling module.
   - Then split attention/DeltaNet implementations behind private module
     boundaries.
   - Add targeted compile checks after every extraction.

3. Prefix cache ownership cleanup.
   - Introduce typed prefix resource entries for full blocks vs partial pages.
   - Move exact-boundary hash/key logic out of `PrefixKvCache` internals.
   - Keep block-manager refcount changes explicit and covered by unit tests.

4. AR loop decomposition.
   - Move prefix attach/insert and DeltaNet restore into a dedicated helper.
   - Move request resource cleanup into one owner.
   - Keep streaming/result emission separate from scheduling mutation.

5. Scheduler policy cleanup.
   - Replace ad hoc hybrid boundary math with named prefix-boundary policy.
   - Expand scheduler unit tests around block-aligned and exact final-prefix
     cases.

6. Backend dispatch cleanup.
   - Document and encode GEMM/attention fallback policy as backend capabilities.
   - Avoid per-model special cases in backend dispatch unless the model config
     owns the capability.

7. Tests and benchmarks cleanup.
   - Separate debug scripts from regression tests.
   - Consolidate duplicated endpoint helpers.
   - Add long-run stress harness presets for Qwen3/Qwen3.5 TopicGuard workloads.

## First PR Scope

This PR intentionally starts with foundation cleanup only. It should be easy to
review and should not alter serving behavior:

- Add `CommonModelConfig::new` and
  `CommonModelConfig::with_uniform_physical_kv_layers`.
- Replace repeated model registry `CommonModelConfig` literals.
- Add model-owned `HybridAttentionPattern` and keep DeltaNet pool config focused
  on concrete pool dimensions.
- Use that layer-pattern helper from Qwen3.5 and Qwen3-Next metadata and GGUF
  shape logic.
- Remove unused `GgufLoadResult::deltanet` and related Qwen3.5 GGUF duplicate
  construction.
- Remove stray temporary/review markers and unused runtime-cap variables.
- Centralize prefix-aware scheduler prefill chunk selection so running and
  waiting admission paths cannot drift.
- Centralize model/runtime `PRELUDE_*` feature toggles in `EngineConfig`
  instead of reading environment variables inside model/cache modules.
- Share grouped-prefill batch slicing policy between Qwen3.5 and Qwen3-Next.
- Share packed-varlen sequence offset calculation between hybrid models.
- Refresh model-onboarding docs to match the current single-file inventory
  registry and shared config helpers.
