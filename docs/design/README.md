# Prelude Architecture

## File Layout

```
crates/
├── prelude-server/                                # Binary — standalone HTTP server (composition root)
│   └── src/
│       └── main.rs                                # Engine::new(config) → engine.serve()
│
├── prelude-dynamo/                                # Binary — NVIDIA Dynamo backend (alternative entry point)
│   └── src/
│       ├── main.rs                                # Worker::from_settings() → DistributedRuntime → endpoint.start()
│       └── lib.rs                                 # impl AsyncEngine for PreludeBackend (wraps Engine)
│
├── prelude-core/                                  # Core: ops, models, engine, scheduler
│   └── src/
│       ├── lib.rs                                 # pub use engine::Engine;
│       │
│       ├── ops/                                   # Op trait definitions + built-in implementations
│       │   ├── mod.rs                             # register_backend(), select_ops(), thread-local Ops context
│       │   └── traits/                            # Trait definitions — the shared contract
│       │       ├── mod.rs                         # re-exports all traits
│       │       ├── ops.rs                         # unified Ops trait — flat API, fused ops return Option
│       │       └── attention.rs                   # VarlenParams, PagedParams, MaskType
│       │
│       ├── tensor/                                # Re-exports candle-core types (Tensor, DType, Device, etc.)
│       │
│       ├── engine/                                # Engine — the public API + execution loops
│       │   ├── mod.rs                             # pub struct Engine, trait InferenceEngine
│       │   ├── config.rs                          # EngineConfig — mode, device, scheduler, spec decode, grammar
│       │   ├── weight_loader.rs                    # WeightLoader: safetensors + GGUF → Tensor by name
│       │   ├── executor.rs                         # trait Executor { submit(batch) -> Handle, collect(Handle) -> Output }
│       │   ├── run/                               # Scheduling-paradigm loops (device-agnostic, all call Executor)
│       │   │   ├── ar.rs                          # AR LLM: scheduler.step → submit prefill/decode → sample
│       │   │   ├── dllm.rs                        # Diffusion LLM: iterative demasking loop
│       │   │   ├── diffusion.rs                   # Image/video: denoising loop
│       │   │   └── tts.rs                         # TTS: multi-stage pipeline
│       │   ├── speculative/                       # Speculative decoding — see scheduler/speculative.md
│       │   │   ├── mod.rs                         # SpecDecodeRunner: draft → verify → accept loop
│       │   │   ├── proposer.rs                    # trait DraftProposer (EAGLE/DraftModel/Ngram/Medusa)
│       │   │   ├── rejection.rs                   # Rejection sampling (strict, probabilistic)
│       │   │   └── tree.rs                        # Tree attention mask construction
│       │   └── sampling/                          # Sampling orchestration — see scheduler/constrained_decoding.md
│       │       ├── mod.rs                         # Sampler: penalties → grammar → ops.sampling → token IDs
│       │       ├── grammar.rs                     # GrammarManager: async compile + bitmask fill
│       │       └── logits_processor.rs            # LogitsProcessor trait (penalties, grammar are impls)
│       │
│       ├── scheduler/                             # Scheduling decisions (pure CPU, no GPU)
│       │   ├── mod.rs                             # re-exports
│       │   ├── ar.rs                              # ArScheduler — continuous batching for AR LLMs
│       │   ├── dllm.rs                            # DllmScheduler — block-level demasking for diffusion LLMs
│       │   ├── diffusion.rs                       # DiffusionScheduler — denoising loop for image/video
│       │   ├── oneshot.rs                         # OneShotScheduler — embed, classify, prefill-only
│       │   ├── tts.rs                             # TtsPipelineScheduler — multi-stage TTS streaming
│       │   ├── types.rs                           # ScheduledBatch, ScheduledRequest, StepResult, etc.
│       │   └── components/                        # Reusable components, used by schedulers as needed
│       │       ├── cache/                          # KV cache management subsystem
│       │       │   ├── mod.rs                     # re-exports
│       │       │   ├── block_manager.rs           # BlockManager — block alloc/free + ref counting
│       │       │   ├── prefix_cache.rs            # PrefixKvCache — block-level hash-trie + LRU eviction
│       │       │   ├── prefix_index.rs            # PrefixMatchIndex — tensor-free prefix matching algorithm
│       │       │   ├── kv_buf.rs                  # KvBuf — per-sequence KV buffer (non-paged path)
│       │       │   └── deltanet_pool.rs           # DeltaNetPool — pre-alloc recurrent state for hybrid models
│       │       └── request_queue.rs               # RequestQueue — FCFS / priority / cache-aware ordering
│       │
│       ├── disaggregated/                         # Multi-instance deployment (skip for single-machine)
│       │   ├── pd/                                # Prefill/Decode separation
│       │   │   ├── coordinator.rs                 # Cross-instance request routing
│       │   │   └── kv_transfer.rs                 # KV cache cross-instance transfer protocol
│       │   └── afd/                               # Attention-FFN separation
│       │       └── ffn_follower.rs                # Passive FFN process
│       │
│       └── models/                                # Model implementations
│           ├── mod.rs                             # trait ModelForward, model_config! macro
│           ├── layers/                            # Shared building blocks (weight containers + utilities)
│           │   ├── mod.rs                         # Context structs (BatchAttnContext, PagedKvContext, etc.)
│           │   ├── linear.rs                      # Linear, RmsNorm, NaiveLinear, QuantFormat registry
│           │   ├── embedding.rs                   # Embedding (vocab → hidden)
│           │   └── attn_utils.rs                  # RotaryEmbedding, qknorm_rope_varlen, attention dispatch
│           ├── registry.rs                        # ModelRegistry: inventory-based auto-registration (ArchSpec trait)
│           ├── qwen3.rs                           # Qwen3 (GQA + QK-norm) — self-contained, calls ops.xxx()
│           ├── qwen3_moe.rs                       # Qwen3-MoE (SparseMoeBlock + shared DecoderLayer)
│           ├── qwen3_5.rs                         # Qwen3.5 hybrid (gated attention + DeltaNet)
│           ├── gemma3.rs                          # Gemma3 (softcap, sliding window)
│           ├── llama.rs                           # Llama-3 (also: Phi3, InternLM3, Yi, Mistral)
│           ├── deepseek_v3.rs                     # DeepSeek-V3 (MLA + MoE + EP)
│           ├── flux.rs                            # Flux DiT (double/single stream, joint attention, AdaLN)
│           ├── whisper.rs                         # Whisper (encoder-decoder, cross-attention)
│           ├── bge.rs                             # BGE/GTE (encoder-only, embedding)
│           └── ...                                # llada2, qwen3_tts, qwen3_omni, hunyuan_video, ...
│
├── third_party/                               # All vendored third-party source (git submodules)
│   ├── flashinfer/                            # FlashInfer (attention, SM80/SM90)
│   ├── flash-attention/                       # FA4 (attention, TVM AOT, SM90+)
│   ├── DeepGEMM/                              # DeepGEMM (BF16/FP8 GEMM, SM90+)
│   ├── cutlass/                               # CUTLASS (GEMM + conv, header-only)
│   ├── tvm-ffi/                               # TVM FFI runtime (used by FA4, FlashInfer, cuLA)
│   ├── cuLA/                                  # cuLA (attention library)
│   ├── llama.cpp/                             # llama.cpp (GGUF quant kernels source)
│   ├── oneDNN/                                # OneDNN (GEMM + conv + fused ops, CPU)
│   ├── composable_kernel/                     # (planned) CK (GEMM + attention, ROCm)
│   ├── aiter/                                 # (planned) aiter (flash attention, ROCm gfx942/950)
│   ├── nccl/                                  # (planned) NCCL (collective communication, CUDA)
│   ├── rccl/                                  # (planned) RCCL (collective communication, ROCm)
│   └── uccl/                                  # (planned) UCCL-EP (MoE expert-parallel, cross-device)
│
├── plugins/                                       # Device-agnostic FFI crates
│   └── prelude-mooncake/                          # Mooncake Transfer Engine (KV cache transport)
│       ├── build.rs                               # bindgen generates FFI from transfer_engine_c.h
│       └── src/lib.rs                             # impl KvTransfer for MooncakeTransfer
│
├── prelude-cuda/                              # CUDA device impl (Ops + Executor)
│   ├── src/
│   │   ├── lib.rs                             # register() registers CudaOps + CudaExecutor with priority/probe
│   │   ├── device.rs                          # CUDA runtime: CudaStorage, stream/device registry, PTX loading
│   │   ├── cuda_ops.rs                        # struct CudaOps, impl all 9 op traits
│   │   ├── quant_backends.rs                  # GPU QuantFormat registration (inventory, priority=100)
│   │   ├── executor.rs                        # CudaExecutor: GPU queue, CUDA graph (planned)
│   │   ├── ops/                               # Kernel wrapper modules
│   │   │   ├── mod.rs                         # PTX loading, module exports
│   │   │   ├── elementwise.rs                 # vectorized_add, fast_silu_mul
│   │   │   ├── rmsnorm.rs                     # fast_rmsnorm, fused_add_rmsnorm
│   │   │   ├── rope.rs                        # fused_qknorm_rope_varlen
│   │   │   ├── kv_cache.rs                    # fused_knorm_rope_kv_cache_write
│   │   │   ├── moe.rs                         # MoE routing ops
│   │   │   ├── gemm.rs                        # GEMM wrappers
│   │   │   ├── quant.rs                       # quantization ops
│   │   │   └── tiled_mmq.rs                   # tiled MMQ kernels (GGUF quantized GEMM)
│   │   ├── attn/                              # Attention backends (impl AttentionOps)
│   │   │   ├── mod.rs                         # module exports
│   │   │   ├── flash_v4.rs                    # FA4 CuTeDSL wrappers (SM90+)
│   │   │   └── flashinfer.rs                  # FlashInfer wrappers (SM80+)
│   │   └── kernels/kernels_src/               # .cu source files (organized by subdirectory)
│   │
│   ├── fa4/                                   # build.rs compiles third_party/flash-attention/
│   ├── flashinfer/                            # build.rs compiles third_party/flashinfer/
│   ├── deepgemm/                              # build.rs compiles third_party/DeepGEMM/
│   ├── cutlass-gemm/                          # BF16/FP16 GEMM via third_party/cutlass/
│   ├── quant-gemm/                            # GGUF quantized GEMM + dequant (llama.cpp MMQ kernels)
│   ├── cula/                                  # cuLA attention via third_party/cuLA/
│   ├── nccl/                                  # build.rs links third_party/nccl/
│   └── uccl-ep/                               # build.rs compiles third_party/uccl/ep/
│
├── prelude-rocm/                              # ROCm device impl (Ops + Executor)
│   ├── src/
│   │   ├── lib.rs                             # register() registers RocmOps + RocmExecutor with priority/probe
│   │   ├── rocm_ops.rs                        # struct RocmOps, enum RocmArch (gfx942/950/1100/1200)
│   │   ├── executor.rs                          # RocmExecutor: HIP graphs, GPU queue
│   │   └── ...                                # attention, gemm, comm, fused, norm, activation, conv
│   │
│   ├── ck/                                    # build.rs compiles third_party/composable_kernel/
│   ├── aiter/                                 # build.rs compiles third_party/aiter/
│   ├── rccl/                                  # build.rs links third_party/rccl/
│   └── uccl-ep/                               # build.rs compiles third_party/uccl/ep/
│
├── prelude-metal/                             # Metal device impl (Apple Silicon)
│   ├── src/
│   │   ├── lib.rs                             # register() registers MetalOps + MetalExecutor with priority/probe
│   │   ├── metal_ops.rs                       # struct MetalOps (unified memory, simdgroup mm)
│   │   ├── executor.rs                          # MetalExecutor: Metal command buffer encoding
│   │   └── ...                                # attention, gemm, fused, norm, activation, conv
│   └── shaders/                               # MSL compute shaders (*.metal)
│
├── prelude-vulkan/                            # Vulkan device impl (cross-vendor)
│   ├── src/
│   │   ├── lib.rs                             # register() registers VulkanOps + VulkanExecutor with priority/probe
│   │   ├── vulkan_ops.rs                      # struct VulkanOps (cooperative_matrix, subgroup_size)
│   │   ├── executor.rs                          # VulkanExecutor: Vulkan command buffer + compute pipeline
│   │   └── ...                                # attention, gemm, norm, activation, conv
│   └── shaders/                               # GLSL → SPIR-V compute shaders
│
├── prelude-tpu/                               # TPU device impl (XLA/Pallas)
│   └── src/
│       ├── lib.rs                             # register() registers TpuOps + TpuExecutor with priority/probe
│       ├── tpu_ops.rs                         # struct TpuOps (PjrtClient, compiled_cache)
│       ├── executor.rs                          # TpuExecutor: XLA trace + compile cache
│       └── ...                                # attention, gemm, session (FusedOps: all None, XLA auto-fuses)
│
└── prelude-cpu/                               # CPU device impl
    ├── src/
    │   ├── lib.rs                             # register() registers CpuOps + CpuExecutor with priority/probe
    │   ├── cpu_ops.rs                         # struct CpuOps, impl all 9 op traits
    │   ├── executor.rs                          # CpuExecutor: simple block_in_place execution
    │   ├── ops/                               # CPU kernel implementations
    │   │   ├── attention/                     # AVX-512 / DPBF16 optimized attention
    │   │   ├── quant/                         # GGUF quantized matmul (Q4_0, Q4_K, Q6_K, IQ4_NL, ...)
    │   │   ├── rmsnorm.rs                     # Vectorized RMSNorm
    │   │   ├── rope.rs                        # RoPE (in-place, cos_sin_cache)
    │   │   ├── silu_mul.rs                    # Fused SiLU×Mul
    │   │   └── gemm.rs                        # OneDNN GEMM + dequant fallback
    │   └── linear_backends.rs                 # OnednnLinear, quant format registration (inventory)
    │
    └── onednn-ffi/                            # OneDNN C++ FFI (compiled by build.rs)
```

**Reading guide:**

- **`prelude-server/`** — binary crate, standalone composition root. Has zero device-specific code.
  Device crates register their Ops + Executor via `register()` at startup with priority/probe.
  Server just calls `register()` then creates `Engine::new(config)`.

- **`prelude-dynamo/`** — alternative binary crate for running as an NVIDIA Dynamo backend.
  Links `dynamo-runtime` + `prelude-core`. Implements Dynamo's `AsyncEngine` trait by wrapping
  `Engine`. Dynamo handles multi-node routing, P/D disaggregation orchestration, and KV transfers.
  Prelude handles single-worker inference. See [integration.md](integration.md).

- **`prelude-core/src/ops/`** — three layers:
  - `traits/` — trait definitions. `OpsBundle` provides a **flat API** (`ops.exp()`, `ops.rms_norm()`,
    `ops.varlen_attention()`, `ops.fused_add_rmsnorm()`). Models call `ops.xxx()` for everything —
    no `ops.attn.xxx()` nesting. Fused ops try device kernel → auto-fallback to composed.
    `TensorOps` uses `base()` delegation: device backends override only what they need.
  - `composed/` — `ComposedOps`: default impls for NormOps, ActivationOps, ConvOps, AttentionOps
    by composing TensorOps primitives. Pure logic, no device dependency.
  - Basic tensor ops (matmul, element-wise, cast, reduce) are handled by candle-core natively.
    The Ops trait only covers fused/inference-specific ops that candle doesn't provide.

- **`prelude-core/src/models/commons/`** — shared weight containers and utilities.
  `Linear`, `RmsNorm`, `Embedding`, `RotaryEmbedding`, context structs (`BatchAttnContext`,
  `PagedKvContext`). NOT module-level abstractions — models are self-contained (like vLLM),
  directly call `ops.xxx()` for compute.

- **`prelude-core/src/engine/executor.rs`** — trait `Executor` defines how scheduled batches are
  executed on a device. Core provides device-agnostic scheduling loops in `engine/run/`
  (batch mode, continuous batching). Device crates implement `Executor` with device-specific
  optimizations (GPU queue, CUDA graphs, HIP graphs, Metal command buffers).

- **`prelude-core/src/models/`** — model implementations. **Self-contained first** (like vLLM): each
  model file has its own structs and forward logic, 1:1 mapping to HuggingFace transformers.
  Models call `ops.xxx()` directly for compute. `models/commons/` only shares what's **universally
  common** across all models: weight containers (`Linear`, `Embedding`), `RotaryEmbedding`, and
  context structs. Model-specific components stay in model files — no forced abstraction.

- **`prelude-{cuda,rocm,metal,vulkan,tpu,cpu}/`** — one crate per device target. Each
  provides Ops (kernel dispatch) + Executor (execution strategy), registered at startup via `register()`.
  Features are **additive** — `--features cuda,rocm` builds both. Runtime probe auto-detects GPU.

- **`plugins/`** — device-agnostic FFI crates.
  `prelude-mooncake` (KV cache transport) uses `bindgen` to wrap Mooncake's Transfer Engine C API.
  Feature-gated — only built when standalone disaggregated serving is enabled.

- **`third_party/`** — all vendored third-party source (git submodules). Source only,
  not Cargo crates. Cross-device libraries (UCCL-EP) live here and are compiled by
  multiple device crates. NCCL/RCCL are **dlopen'd** at runtime.

- **`prelude-cuda/{fa4,flashinfer,...}/`** — kernel FFI sub-crates. Each has a `build.rs`
  that compiles from `third_party/` with the device toolchain (nvcc/hipcc), plus Rust
  FFI bindings. Only consumed by the parent device impl.

## Principles

1. **Subsystem isolation:** A person working on one subsystem should be able to do their job
   without reading or understanding any other subsystem's code. Model devs don't read kernel code.
   Kernel devs don't read model code. Device backend devs don't read other backends.
   The trait signatures are the complete contract between subsystems.

2. **Kernel optimization reach:** When a kernel optimization is added, as many models as possible
   should benefit automatically without per-model code changes. This is achieved via `OpsBundle` —
   add a fused kernel once, all models calling `ops.xxx()` benefit. O(1) change → O(N) benefit.

## Goals

1. Model code is device-agnostic: no `#[cfg(feature = "cuda")]` in models.
2. Model code is kernel-agnostic: models never reference FA4, FlashInfer, CUTLASS, etc.
3. Each operation independently dispatches to the best available kernel for the device + parameters.
4. Multi-device: CUDA, ROCm, TPU, Vulkan, CPU share the same model code.
5. Multi-model: AR (LLM), diffusion, TTS, vision all use the same op traits.
6. Fusion is transparent: OpsBundle handles fused/fallback dispatch, models call `ops.xxx()` without branching.

## Layering

```
Engine              starts             Run loop (core: batch or continuous batching)
Run loop            calls              Executor::submit/collect (device crate, via register())
Run loop            calls              Scheduler (core, pure CPU scheduling decisions)
Executor            calls              Model code (prepare tensors → model.forward())
Model code          calls              OpsBundle flat API (ops.rms_norm, ops.varlen_attention, ...)
OpsBundle           dispatches to      ComposedOps (default) or device overrides (CudaOps, ...)
ComposedOps         composes           candle tensor ops → norm, conv, attention, activation
Basic tensor ops    provided by        candle-core (CUDA + CPU backends, registered GEMM dispatch)
Fused ops           try device kernel  → auto-fallback to composed ops (transparent to model)
Device ops          dispatches to      Kernel libraries (FA4, FlashInfer, DeepGEMM, CUTLASS, CK, ...)
```

**Two trait boundaries** separate core from device crates:

1. **`OpsBundle`** (kernel dispatch) — flat API: `ops.exp()`, `ops.rms_norm()`,
   `ops.varlen_attention()`, `ops.fused_add_rmsnorm()`. Models call `ops.xxx()` for everything.
   Internally: `ComposedOps` provides defaults by composing TensorOps primitives.
   `TensorOps` uses `base()` delegation — device backends override only what they need.
   Fused ops try device kernel → auto-fallback to composed. All decision logic is in OpsBundle,
   not in model code.

2. **`Executor`** (device execution) — defines HOW batches are submitted to and collected from
   a device. `submit()` is non-blocking (GPU queues work, CPU runs inline). `collect()` awaits
   completion. Core's run loops use submit/collect to naturally get double-buffering on GPU
   (prepare batch N+1 while device runs batch N) and sequential execution on CPU.

**Models are self-contained** (like vLLM): each model file defines its own attention/MLP/decoder
structs, 1:1 mapping to HuggingFace transformers. Models call `ops.xxx()` directly for compute.
Shared weight containers (`Linear`, `Embedding`, `RotaryEmbedding`) live in `models/commons/`.
**When a new fused kernel is added, OpsBundle is updated once — all models benefit automatically.**

**Design principle: `Linear` is a parameter carrier, OpsBundle is the decision maker.**
`Linear` holds weights + optional LoRA state. When fusion is needed (e.g. fused QKV+LoRA projection),
`Linear` passes its weights to OpsBundle (`ops.qkv_projection(x, weights, lora_state)`), and
OpsBundle handles fused/fallback/device dispatch. All decision logic lives in OpsBundle, never
in `Linear` or model code. This ensures LoRA, quantization, and fusion work transparently across
all devices without model-level branching.

## Document Index

### ops/ — Op Trait System
| Section | File |
|---------|------|
| **Tensor Type** | [ops/tensor.md](ops/tensor.md) |
| **Tensor Layout Conventions** | [ops/tensor_layout.md](ops/tensor_layout.md) |
| **Op Traits** | [ops/op_traits.md](ops/op_traits.md) |
| **Ops Bundle** | [ops/ops_bundle.md](ops/ops_bundle.md) |
| **Session Lifecycle** | [ops/session_lifecycle.md](ops/session_lifecycle.md) |
| **Device Capability Matrix** | [ops/device_capability.md](ops/device_capability.md) |

### models/ — Model Architecture
| Section | File |
|---------|------|
| **LoRA (Low-Rank Adaptation)** | [models/lora.md](models/lora.md) |
| **Distributed Execution** | [models/distributed.md](models/distributed.md) |

### device/ — Device Implementations
| Section | File |
|---------|------|
| **Device Implementations** | [device/device_impls.md](device/device_impls.md) |
| **Runtime Dependencies** | [device/runtime_deps.md](device/runtime_deps.md) |
| **Construction** | [device/construction.md](device/construction.md) |
| **Non-Softmax Token Mixers & Hybrid Cache** | [device/token_mixers.md](device/token_mixers.md) |

### scheduler/ — Scheduler & Engine
| Section | File |
|---------|------|
| **Scheduler Architecture** | [scheduler/README.md](scheduler/README.md) |
| **AR Scheduler** | [scheduler/ar.md](scheduler/ar.md) |
| **DLLM Scheduler** | [scheduler/dllm.md](scheduler/dllm.md) |
| **Diffusion Scheduler** | [scheduler/diffusion.md](scheduler/diffusion.md) |
| **TTS Scheduler** | [scheduler/tts.md](scheduler/tts.md) |
| **OneShot Scheduler** | [scheduler/oneshot.md](scheduler/oneshot.md) |
| **Speculative Decoding** | [scheduler/speculative.md](scheduler/speculative.md) |
| **Sampling & Constrained Decoding** | [scheduler/constrained_decoding.md](scheduler/constrained_decoding.md) |
| **Disaggregated Serving** | [scheduler/disaggregated.md](scheduler/disaggregated.md) |
| **Design Comparisons** | [scheduler/design-comparisons.md](scheduler/design-comparisons.md) |
| **Examples** | [scheduler/examples.md](scheduler/examples.md) |
| **Workflows** | [scheduler/workflows.md](scheduler/workflows.md) |

### Top-level
| Section | File |
|---------|------|
| **Model Code Pattern** | [model_code.md](model_code.md) |
| **Model Examples (25)** | [model_examples.md](model_examples.md) |
| **Subsystem Independence** | [subsystem_independence.md](subsystem_independence.md) |
| **External Integration (Dynamo & Mooncake)** | [integration.md](integration.md) |
| **Summary** | [summary.md](summary.md) |
| **Model Registry & Loading** | [../model_registry.md](../model_registry.md) |

## Dependency Graph

```
prelude-server (binary, standalone HTTP server — zero device-specific code)
    ├── prelude-core                  (Ops trait, models, engine, scheduler)
    │       ├── candle-core               (tensor backend: storage, matmul, basic ops)
    │       ├── candle-nn                 (softmax, rotary emb, etc.)
    │       └── llguidance                (constrained decoding, pure Rust)
    ├── plugins/prelude-mooncake      (impl KvTransfer, wraps Mooncake Transfer Engine C API)
    ├── prelude-cuda                  (feature-gated, additive — fused ops + Executor)
    │       ├── prelude-core
    │       ├── candle-core (features = ["cuda"])
    │       ├── fa4/, flashinfer/, deepgemm/, cutlass-gemm/, quant-gemm/, cula/
    │       └── (each sub-crate compiles from third_party/)

prelude-dynamo (binary, NVIDIA Dynamo backend — alternative entry point)
    ├── prelude-core                  (same core library, used as library)
    ├── dynamo-runtime                (Dynamo's service discovery, transport, NIXL)
    ├── prelude-cuda / prelude-rocm / prelude-cpu  (device crates, same as standalone)
    └── (NO prelude-mooncake — Dynamo owns KV transfer via NIXL/Mooncake)

third_party/                          (git submodules, source only — not Cargo crates)
    ├── flashinfer/                   (compiled by prelude-cuda/flashinfer/build.rs)
    ├── flash-attention/              (compiled by prelude-cuda/fa4/build.rs)
    ├── tvm-ffi/                      (TVM FFI runtime, used by FA4/FlashInfer/cuLA build.rs)
    ├── composable_kernel/            (compiled by prelude-rocm/ck/build.rs)
    ├── uccl/                         (compiled by prelude-cuda AND prelude-rocm — cross-device)
    └── ...
```

**Key rules:**
- `prelude-core` depends on candle-core (Rust tensor library) for basic tensor ops and
  llguidance (pure Rust) for constrained decoding. The `cuda` feature is gated — CPU-only
  builds don't pull in CUDA types. `ComposedOps` composes candle tensor ops into higher-level
  ops — pure logic. `bare_ops()` provides minimal Ops defaults as lowest-priority fallback.
- Device crates register Ops + Executor via explicit `register()` calls at startup with priority/probe.
- Ops trait: flat API, models call `ops.xxx()` for fused/inference-specific ops. Basic ops
  (matmul, cast, element-wise) go through candle-core directly. GEMM dispatch is registered
  to route candle's matmul through CUTLASS/DeepGEMM.
- Fused ops return `Option` — `None` auto-falls back to composed defaults.
- Engine calls `select_backend()` internally — picks highest-priority backend whose `probe()` returns true.
- Device features are **additive**, not exclusive — `--features cuda,rocm` builds both.
- NCCL/RCCL are **dlopen'd** at runtime (not statically linked), avoiding symbol conflicts.
- Cross-device libraries (UCCL-EP) have separate sub-crates per device with shared
  `third_party/` source; FFI bindings are ~400 lines each (API stable, duplication acceptable).
- `prelude-dynamo` links `dynamo-runtime` — this is a Dynamo-specific binary, not a core dependency.
  `prelude-core` has NO knowledge of Dynamo. The integration is purely at the binary crate level.
- `prelude-mooncake` is feature-gated (`--features mooncake`). Single-machine and Dynamo-backend
  deployments don't build it. Only standalone multi-node disaggregated serving needs it.

## Build Targets

Standalone binaries cover all platforms. Install script auto-detects and downloads the right one.

| Binary | Build Platform | Device Backends | Features |
|--------|---------------|-----------------|----------|
| `prelude-linux-x86_64` | Linux | CUDA + ROCm + Vulkan + CPU | `--features cuda,rocm,vulkan` |
| `prelude-darwin-aarch64` | macOS | Metal + CPU | `--features metal` |
| `prelude-dynamo` | Linux | CUDA + ROCm + CPU | links `dynamo-runtime` |

Runtime auto-detection selects the best available backend:
```rust
// prelude-server/src/main.rs
// Device crates register at startup with priority/probe.
prelude_cpu::register();
#[cfg(feature = "cuda")]
prelude_cuda::register();
let engine = Engine::new(&config);
engine.serve().await;
```

Metal and CUDA/ROCm cannot coexist in one binary (macOS SDK vs Linux GPU toolchains).

## Dependency Summary

```
prelude-server            →  standalone binary: Engine::new(config), Axum HTTP, zero device code
prelude-dynamo            →  Dynamo backend binary: wraps Engine as AsyncEngine, links dynamo-runtime
plugins/prelude-mooncake  →  impl KvTransfer via Mooncake Transfer Engine (feature-gated, multi-node only)
prelude-core/ops          →  Ops trait (flat API) + composed defaults (candle tensor ops)
prelude-core/engine       →  Engine, Executor trait, run loops, Sampler, GrammarManager (llguidance)
prelude-core/models       →  self-contained models (call ops.xxx()), layers/ (Linear, Embedding, RoPE)
prelude-core/scheduler    →  ArScheduler, BlockManager, PrefixCache, DeltaNetPool
prelude-{device}/         →  hot-path Ops override + Executor impl + kernel sub-crates
```

Each layer only knows the layer directly below it. Models don't know devices.
Modules don't know models. Device impls don't know each other.

