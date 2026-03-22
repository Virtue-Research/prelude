# Prelude OSS Gap Analysis for TEI/vLLM-Style Pooling Workloads

Last validated: 2026-03-11

## Executive Summary

Prelude is closer to a promising internal prototype than an OSS-ready competitor today.

The good news is that the core architectural bias is correct for the target market. The codebase already treats `classify` and `embed` as first-class batch workloads, keeps them on a dedicated batch runtime, and does not force them through the continuous-decode MVP path. That is the right substrate for a product aimed at small-model, non-continuous, single-forward inference.

The main blockers are not in the hot path. They are in the serving surface, OSS packaging, operational readiness, and scope discipline:

- The public server exposes multiple forward-only endpoints as stubs: `/v1/rerank`, `/v1/score`, `/v1/tokenize`, `/v1/detokenize`, and `/metrics`.
- The router also exposes unrelated 501 endpoints that dilute product positioning: `/v1/responses`, `/v1/messages`, `/v1/audio/transcriptions`, and `/v1/moderations`.
- The engine trait does not yet have first-class forward-only primitives for tokenize, score, or rerank, so the missing API surface is structural, not just routing work.
- README and server docs claim OpenTelemetry and metrics support, but the current server does not wire OTLP exporters and `/metrics` returns 501.
- OSS hygiene is incomplete: there is no root `LICENSE`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, or `SECURITY` file, there is no container/release story, and CI only covers CPU defaults.
- Public benchmark and release docs still assume private or internal environments, including private paths, specific H200 hosts, and sibling clones outside the repo.

The right v1 is not "beat vLLM everywhere." The right v1 is:

- Primary product: embeddings, classification, score/rerank, and single-forward or `max_tokens=1` completion benchmarking.
- Secondary compatibility layer: keep `/v1/completions` and `/v1/chat/completions` for easy adoption.
- Explicitly defer: broad continuous-decode parity, distributed serving, LoRA, and full OpenAI/Anthropic API breadth.

If the team keeps the product narrow and finishes the missing forward-only surface, Prelude can plausibly become a TEI-like serving framework with better control over the Rust/CUDA stack and a more credible pooling-first story than vLLM's current "pooling for convenience" posture.

## Current Strengths

### 1. The runtime already favors forward-only batching

Prelude already routes `classify` and `embed` through the batch runtime instead of the continuous-generation runtime.

Repo evidence:

- `docs/continuous-batching.md`
- `crates/prelude-core/src/runtime/batch_runtime.rs`
- `crates/prelude-core/src/runtime/engine.rs`

Why this matters:

- This is the correct execution bias for TEI-style workloads.
- The v1 roadmap should preserve this and avoid making pooling workloads subordinate to continuous decode work.

### 2. Classification and embedding already have dedicated engine paths

`classify` and `embed` are not thin wrappers over generation. They already have dedicated request types, pretokenized batch types, forward-only execution paths, and post-processing steps.

Repo evidence:

- `crates/prelude-core/src/task/classify.rs`
- `crates/prelude-core/src/task/embed.rs`
- `crates/prelude-core/src/types.rs`

Why this matters:

- This reduces the amount of refactoring needed to become a credible pooling server.
- The remaining missing work is mostly API completion, configurability, coverage, and operational polish.

### 3. The codebase already has meaningful batching and cache infrastructure

Prelude already ships:

- varlen prefill paths
- adaptive scheduler logic for batch workloads
- prefix cache and paged cache infrastructure
- CPU and GPU execution paths

Repo evidence:

- `docs/benchmark.md`
- `crates/prelude-core/src/task/tokenize.rs`
- `crates/prelude-core/src/cache/*`
- `crates/prelude-scheduler/src/*`

Why this matters:

- A forward-only product can reuse most of the scheduling and batching substrate instead of inventing a new runtime.
- The missing work is mostly on controls and public contracts rather than low-level batching primitives.

### 4. The API baseline is not empty

The current server already exposes:

- `/health`
- `/v1/models`
- `/v1/embeddings`
- `/v1/classify`
- `/v1/completions`
- `/v1/chat/completions`

Repo evidence:

- `crates/prelude-server/src/lib.rs`
- `crates/prelude-server/src/routes/classify.rs`
- `crates/prelude-server/src/routes/embeddings.rs`
- `crates/prelude-server/src/routes/completions.rs`
- `crates/prelude-server/src/routes/chat_completions.rs`

Why this matters:

- The project already has a public HTTP server shape, request/response types, and API tests.
- The gap is that the forward-only surface is incomplete and inconsistent.

### 5. The local test baseline is functional

The following commands were run locally on 2026-03-11:

- `cargo test -p prelude-core`
- `cargo test -p prelude-scheduler`
- `cargo test -p prelude-server --lib --tests`
- `cargo build -p prelude-server --release`

Observed result:

- All of the above passed.
- The build/test output is warning-heavy, especially in `prelude-core`.
- The warnings include both broad dead-code noise and Rust 2024 `unsafe_op_in_unsafe_fn` diagnostics in CPU kernels.

This is important for OSS readiness:

- Functional baseline: good.
- Contributor-facing quality bar: not yet acceptable.

## Competitive Baseline

### TEI

Official TEI positioning is already very close to the product Prelude should target.

What TEI provides today:

- A focused server for text embeddings and sequence classification
- token-based dynamic batching
- optimized inference using Flash Attention, Candle, and cuBLASLt
- production-ready observability claims, including OpenTelemetry and Prometheus
- Docker-first deployment and fast-boot positioning
- public routes for embeddings, rerankers, sequence classification, and batching examples
- support for SPLADE / sparse workflows

Primary sources:

- TEI repo README: <https://github.com/huggingface/text-embeddings-inference>
- TEI quick tour: <https://huggingface.co/docs/text-embeddings-inference/en/quick_tour>

Important implications:

- TEI is the right bar for product scope, deployment ergonomics, and forward-only runtime positioning.
- TEI is the strongest comparator for embeddings/classification/reranking, not vLLM.

### vLLM

vLLM is the broader compatibility and operability bar, not the ideal performance-positioning bar for this product.

What vLLM provides today that matters here:

- OpenAI-compatible serving plus extra forward-only endpoints:
  - `/tokenize`, `/detokenize`
  - `/pooling`
  - `/classify`
  - `/score`
  - `/rerank`, `/v1/rerank`, `/v2/rerank`
- support for pooling models and task conversion for many transformer families
- a real `/metrics` endpoint with Prometheus-compatible output
- production metrics and Grafana dashboard examples
- scalable deployment via Ray Serve LLM, including load balancing and back-pressure

Important nuance from vLLM's own docs:

- vLLM explicitly says pooling support exists "primarily for convenience" and is not guaranteed to improve performance over Hugging Face Transformers or Sentence Transformers directly.

Primary sources:

- vLLM OpenAI-compatible server docs: <https://docs.vllm.ai/en/stable/serving/openai_compatible_server/>
- vLLM pooling model docs: <https://docs.vllm.ai/en/stable/models/pooling_models/>
- vLLM production metrics: <https://docs.vllm.ai/en/stable/usage/metrics/>
- vLLM metrics design: <https://docs.vllm.ai/en/stable/design/metrics.html>

Important implications:

- vLLM is the right bar for API breadth and production observability.
- It is not the right performance narrative for a pooling-first product, which creates room for Prelude if the serving surface becomes complete.

### Recommended comparison stance

For this project, use the competitors differently:

- Use TEI as the primary comparator for product focus, benchmarks, deployment UX, and forward-only serving completeness.
- Use vLLM as the primary comparator for compatibility surface, observability, and operational maturity.

## Gap Matrix

| Area | Prelude today | TEI / vLLM baseline | Gap to close | Priority |
| --- | --- | --- | --- | --- |
| Product positioning | README mixes generation and pooling, with broad OpenAI-compatible framing | TEI is clearly embeddings/classification-first; vLLM is clearly a general inference platform | Narrow the public story to pooling-first workloads and treat generation as compatibility | P0 |
| Forward-only public APIs | `/v1/rerank`, `/v1/score`, `/v1/tokenize`, `/v1/detokenize`, `/metrics` are stubbed | TEI and vLLM expose working forward-only routes | Implement the missing routes or stop advertising them | P0 |
| Unrelated 501 routes on main server | `/v1/responses`, `/v1/messages`, `/v1/audio/transcriptions`, `/v1/moderations` are public but unimplemented | TEI does not dilute itself with unrelated APIs; vLLM's breadth is intentional and backed by features | Remove, feature-gate, or explicitly mark these as experimental and disabled by default | P0 |
| Engine trait surface | `InferenceEngine` supports `generate`, `classify`, and `embed` only | vLLM-equivalent forward-only APIs need tokenize, detokenize, score, rerank | Add first-class trait methods and request/response types | P0 |
| Forward-only runtime controls | CLI and env knobs are generation-oriented: batch size, wait ms, prefill tokens, total tokens | TEI-style serving usually exposes explicit max input and batching constraints; vLLM exposes broader serving controls | Add max input tokens, batch-token budgets, client batch caps, truncation policy, endpoint-specific concurrency/backpressure | P0 |
| Auth consistency | Auth middleware skips non-`/v1/` paths; `/classify` bypasses the normal API-key boundary | Public inference APIs should have a consistent auth model | Remove or protect non-`/v1/` aliases; define one stable security contract | P0 |
| Metrics and tracing | Docs mention OTEL and `/metrics`, but server does not export either in practice | TEI documents OTEL + Prometheus; vLLM exposes `/metrics` and dashboard guidance | Implement Prometheus metrics and either wire OTLP or remove the claim | P0 |
| Container / release packaging | No root Dockerfile or container docs for public use | TEI is Docker-first; vLLM is easy to run via Python or Docker | Add a public container story and release artifacts | P1 |
| OSS repo hygiene | No root `LICENSE`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, or `SECURITY` | Mature OSS servers ship these by default | Add basic OSS governance files and release support matrix | P1 |
| Model coverage for pooling | Publicly documented support is narrow and Qwen-heavy | TEI supports broad embedding and sequence-classification families; vLLM can convert many pooling models | Expand encoder/cross-encoder coverage and document the supported matrix clearly | P0 |
| Sparse / SPLADE path | No sparse embedding path or TEI-native sparse endpoints | TEI explicitly supports SPLADE workflows | Decide whether sparse support is v1 or deferred; if deferred, say so clearly | P1 |
| Benchmark reproducibility | Benchmark docs rely on sibling clones, private setups, and mixed internal assumptions | TEI and vLLM both have cleaner public install/run stories | Publish a clean-clone benchmark workflow with pinned public models and hardware assumptions | P0 |
| CI / release checks | `.github/workflows/ci.yml` is CPU-only and default-feature only | OSS competitors validate broader release surfaces | Add feature-matrix builds, clippy, smoke tests, and at least one reproducible GPU workflow | P1 |
| Build cleanliness | Current build/test output is warning-heavy | OSS competitors are expected to keep default builds relatively clean | Make warnings a tracked release debt and reduce them enough for contributor confidence | P1 |
| Documentation drift | Release and benchmark docs reference private paths, private hosts, sibling repos, and internal assumptions | Public docs should be self-contained and copy-pasteable | Scrub private references and split internal docs from public docs | P0 |

## Prioritized Backlog

The items below are written so they can be handed directly to a later engineer or agent.

### P0

#### 1. Finish the forward-only API surface

Why now:

- This is the single biggest product gap versus both TEI and vLLM.
- The current router advertises capabilities that do not exist.

Repo evidence:

- `crates/prelude-server/src/lib.rs`
- `crates/prelude-server/src/routes/rerank.rs`
- `crates/prelude-server/src/routes/score.rs`
- `crates/prelude-server/src/routes/tokenize.rs`
- `crates/prelude-server/src/routes/metrics.rs`
- `crates/prelude-core/src/engine/mod.rs`

Likely touchpoints:

- `crates/prelude-core/src/engine/mod.rs`
- `crates/prelude-core/src/types.rs`
- `crates/prelude-server/src/lib.rs`
- `crates/prelude-server/src/routes/{rerank,score,tokenize,metrics}.rs`
- `crates/prelude-server/tests/api.rs`

Acceptance criteria:

- `InferenceEngine` exposes first-class methods for `tokenize`, `detokenize`, `score`, and `rerank`.
- Stable HTTP request/response types exist for each route.
- The stub routes are replaced with working handlers.
- API tests cover single input, batch input, validation errors, and unsupported-model errors.

#### 2. Narrow or feature-gate the public server surface

Why now:

- The current route list makes the project look unfinished instead of focused.
- For OSS v1, extra 501s are a positioning liability.

Repo evidence:

- `crates/prelude-server/src/lib.rs`
- `crates/prelude-server/src/routes/messages.rs`
- `crates/prelude-server/src/routes/responses.rs`
- `crates/prelude-server/src/routes/audio.rs`
- `crates/prelude-server/src/routes/moderations.rs`

Likely touchpoints:

- `crates/prelude-server/src/lib.rs`
- README and server docs

Acceptance criteria:

- The default OSS server only exposes routes that are implemented and supported.
- Deferred routes are removed from the default build or hidden behind an explicit experimental flag.
- Public docs match the shipped surface exactly.

#### 3. Add forward-only serving controls that are separate from generation controls

Why now:

- Current controls are scheduler-centric and generation-oriented.
- Pooling workloads need explicit limit and batching contracts.

Repo evidence:

- `crates/prelude-server/src/main.rs`
- `crates/prelude-core/src/config.rs`
- `docs/server.md`

Required new controls:

- `max_input_tokens`
- `max_batch_tokens`
- `max_client_batch_size`
- per-endpoint concurrency or batch caps
- truncation or reject policy
- explicit backpressure / overload behavior

Likely touchpoints:

- `crates/prelude-core/src/config.rs`
- `crates/prelude-core/src/runtime/batch_runtime.rs`
- `crates/prelude-core/src/runtime/engine.rs`
- `crates/prelude-server/src/main.rs`
- `docs/server.md`

Acceptance criteria:

- All new controls are represented in config parsing, CLI/env docs, and API validation.
- Over-limit requests fail deterministically with stable JSON errors.
- Batch-runtime admission uses batch-token budgets, not only item counts.

#### 4. Fix auth consistency on the public inference surface

Why now:

- The current auth model is inconsistent and easy to misconfigure.

Repo evidence:

- `crates/prelude-server/src/auth.rs`
- `crates/prelude-server/src/lib.rs`

Current issue:

- Auth is enforced on `/v1/*`.
- Non-`/v1/` routes are skipped by middleware.
- `/classify` is publicly exposed outside `/v1/*`, which bypasses the normal API-key boundary.

Likely touchpoints:

- `crates/prelude-server/src/auth.rs`
- `crates/prelude-server/src/lib.rs`
- `crates/prelude-server/tests/api.rs`

Acceptance criteria:

- All public inference routes follow one consistent auth policy.
- Either remove non-`/v1/` inference aliases or protect them identically.
- Tests cover authenticated and unauthenticated access for every shipped route.

#### 5. Implement real observability or stop claiming it

Why now:

- README and docs already imply an observability story that does not exist in the running server.

Repo evidence:

- `README.md`
- `docs/server.md`
- `crates/prelude-server/src/routes/metrics.rs`
- `Cargo.toml`

Current issue:

- OTEL dependencies exist in workspace metadata.
- `docs/server.md` documents OTLP environment variables.
- `/metrics` exists in the router but returns 501.
- Code search shows no actual OpenTelemetry initialization or Prometheus exporter wiring.

Likely touchpoints:

- `crates/prelude-server/src/main.rs`
- `crates/prelude-server/src/routes/metrics.rs`
- new metrics/tracing module(s)
- `README.md`
- `docs/server.md`

Acceptance criteria:

- `/metrics` exports Prometheus-compatible counters, gauges, and histograms for requests, queue depth, batch size, token counts, and latency.
- OTLP tracing is either implemented end to end or removed from public docs.
- Public docs include at least one example scrape and dashboard/runbook path.

#### 6. Scrub internal/private assumptions from public docs and release guides

Why now:

- Current docs are not clean-clone reproducible.
- This is a blocker for outside contributors and benchmark reviewers.

Repo evidence:

- `docs/checklist.md`
- `docs/benchmark.md`
- `benchmark/README.md`
- `tests/accuracy/README.md`

Examples of drift:

- references to `together_h200`
- `/scratch/...` model paths
- sibling repos like `../vllm.rs`, `../llama.cpp`, and `../sglang-cpu`
- mixed internal benchmarking assumptions

Acceptance criteria:

- Public docs have no private hosts or private filesystem paths.
- Public docs do not require sibling repos unless clearly optional.
- There is a clean-clone quick start for CPU and GPU using public models only.

#### 7. Expand pooling model support and document the supported matrix honestly

Why now:

- The current public matrix is too narrow for a framework trying to compete with TEI for forward-only serving.

Repo evidence:

- `README.md`
- `crates/prelude-core/src/models/architectures/*`
- `docs/adding-a-model.md`
- `crates/prelude-core/src/engine/helpers.rs`

Current issue:

- Public support is heavily centered on Qwen-family models, plus limited Gemma3 classification support.
- There is no public sparse embedding path.
- There is no broad encoder/cross-encoder support story.

Acceptance criteria:

- Public docs list supported pooling model families explicitly.
- At least one high-quality public embedding model and one public reranker/cross-encoder are first-class, tested targets.
- Unsupported model families fail with clear errors instead of ambiguous behavior.

#### 8. Publish a public benchmark story that matches the product claim

Why now:

- A pooling-first product must prove itself on forward-only workloads, not on broad generation-only narratives.

Required public benchmark set:

- embeddings: `Qwen/Qwen3-Embedding-0.6B`
- rerank/classify: `BAAI/bge-reranker-base` or another public cross-encoder/reranker
- completion compatibility: `Qwen/Qwen3-0.6B` with `max_tokens=1`

Acceptance criteria:

- Same-model comparisons against TEI and vLLM are documented and reproducible.
- Hardware assumptions are explicit.
- Success metrics are explicit: throughput, p50/p95 latency, and error rate.

### P1

#### 9. Add basic OSS governance and packaging files

Why now:

- This is table-stakes for public OSS release.

Missing at repo root:

- `LICENSE`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`

Notes:

- `Cargo.toml` declares `license = "MIT"`, but there is no root license file.
- `crates/candle-core/LICENSE` does not solve the top-level repo requirement.

Acceptance criteria:

- Root governance and license files exist and match the intended release policy.
- README links to them.

#### 10. Add a real release and container story

Why now:

- TEI is Docker-first and easy to deploy.
- Prelude currently has no equivalent public artifact story.

Acceptance criteria:

- A public Dockerfile or image pipeline exists.
- README includes one copy-paste CPU and one GPU launch path.
- Release docs describe versioned artifacts and supported platforms.

#### 11. Harden CI beyond CPU defaults

Why now:

- Current CI only proves the narrowest build shape.

Repo evidence:

- `.github/workflows/ci.yml`

What is missing:

- clippy
- feature-matrix builds
- release build quality gates
- public-model smoke tests
- benchmark regression checks

Acceptance criteria:

- CI covers at least default, GPU-feature compile, and server tests.
- `cargo clippy --workspace --all-targets` is either green or tracked with a documented allowlist.

#### 12. Reduce warning volume to an OSS-acceptable level

Why now:

- Passing but warning-heavy builds make the project look unfinished.

Observed baseline on 2026-03-11:

- `cargo build -p prelude-server --release` passed but emitted 123 warnings from `prelude-core`.

Priority areas:

- Rust 2024 unsafe-op warnings in CPU kernels
- obvious dead code and unused imports in exported paths
- warning noise in benchmark/test binaries

Acceptance criteria:

- Default release build is materially cleaner.
- Remaining warnings are intentional and documented.

#### 13. Decide sparse support policy explicitly

Why now:

- TEI supports SPLADE-related sparse workflows.
- Prelude currently does not.

Acceptance criteria:

- Either sparse embeddings become a roadmap item with clear requirements, or the public docs explicitly say "dense embeddings/classification only" for v1.

### P2

#### 14. Broader generative parity

Deferred unless product scope changes:

- full continuous-decode parity
- speculative decoding
- broader OpenAI Responses or Anthropic Messages support
- multimodal serving

#### 15. Distributed and multi-GPU serving

Deferred for OSS v1:

- tensor parallel / multi-node serving
- Ray-like operational integrations
- cluster-level load balancing

#### 16. LoRA and adapter ecosystem support

Nice to have later, not required for a pooling-first v1.

## Recommended v1 Scope

### Ship in OSS v1

- `/health`
- `/v1/models`
- `/v1/embeddings`
- `/v1/classify`
- `/v1/score`
- `/v1/rerank`
- `/v1/tokenize`
- `/v1/detokenize`
- `/metrics`
- `/v1/completions` and `/v1/chat/completions` as compatibility routes, with public benchmarking centered on `max_tokens=1`

### Optional TEI-native parity after v1 blockers

- `/embed`
- `/predict`
- `/rerank`
- `/embed_sparse`

These matter for TEI compatibility, but they should come after the OpenAI/vLLM-style forward-only surface is complete and stable.

### Remove from default OSS positioning

- `/v1/responses`
- `/v1/messages`
- `/v1/audio/transcriptions`
- `/v1/moderations`
- broad claims around continuous generation as the primary differentiator

### Product statement to use publicly

"Prelude is a Rust-native inference server optimized for small-model, non-continuous, pooling-first workloads such as embeddings, classification, reranking, and other single-forward paths."

This is a stronger and more defensible v1 story than "general vLLM competitor."

## Verification Baseline

Verified locally on 2026-03-11:

- `cargo test -p prelude-core` -> pass
- `cargo test -p prelude-scheduler` -> pass
- `cargo test -p prelude-server --lib --tests` -> pass
- `cargo build -p prelude-server --release` -> pass

Release debt observed during verification:

- The build is warning-heavy.
- The warning surface is concentrated in `prelude-core`, especially dead code and `unsafe_op_in_unsafe_fn` diagnostics.
- CI does not yet prove non-default feature combinations or public GPU release workflows.

## Sources

### Official competitor sources

- Hugging Face TEI repo README: <https://github.com/huggingface/text-embeddings-inference>
- Hugging Face TEI quick tour: <https://huggingface.co/docs/text-embeddings-inference/en/quick_tour>
- vLLM OpenAI-compatible server docs: <https://docs.vllm.ai/en/stable/serving/openai_compatible_server/>
- vLLM pooling models docs: <https://docs.vllm.ai/en/stable/models/pooling_models/>
- vLLM production metrics: <https://docs.vllm.ai/en/stable/usage/metrics/>
- vLLM metrics design: <https://docs.vllm.ai/en/stable/design/metrics.html>

### Repo evidence used in this report

- `README.md`
- `Cargo.toml`
- `.github/workflows/ci.yml`
- `docs/server.md`
- `docs/benchmark.md`
- `docs/checklist.md`
- `docs/continuous-batching.md`
- `tests/accuracy/README.md`
- `benchmark/README.md`
- `crates/prelude-core/src/config.rs`
- `crates/prelude-core/src/engine/mod.rs`
- `crates/prelude-core/src/runtime/batch_runtime.rs`
- `crates/prelude-core/src/runtime/engine.rs`
- `crates/prelude-core/src/task/classify.rs`
- `crates/prelude-core/src/task/embed.rs`
- `crates/prelude-core/src/types.rs`
- `crates/prelude-server/src/lib.rs`
- `crates/prelude-server/src/auth.rs`
- `crates/prelude-server/src/main.rs`
- `crates/prelude-server/src/routes/classify.rs`
- `crates/prelude-server/src/routes/embeddings.rs`
- `crates/prelude-server/src/routes/rerank.rs`
- `crates/prelude-server/src/routes/score.rs`
- `crates/prelude-server/src/routes/tokenize.rs`
- `crates/prelude-server/src/routes/metrics.rs`
- `crates/prelude-server/src/routes/messages.rs`
- `crates/prelude-server/src/routes/responses.rs`
