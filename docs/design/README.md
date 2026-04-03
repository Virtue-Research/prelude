# Prelude Architecture

## File Layout

```
crates/
├── prelude-server/                                # Binary — composition root
│   └── src/
│       └── main.rs                                # create_ops → Engine::new(ops) → engine.serve()
│
├── prelude-core/                                  # Core: traits, modules, models, engine, scheduler (pure Rust)
│   └── src/
│       ├── lib.rs                                 # pub use engine::Engine;
│       │
│       ├── ops/                                   # Op trait definitions + CPU impl
│       │   ├── mod.rs                             # pub struct Ops { attn, kv_cache, gemm, norm, ... }
│       │   ├── traits/                            # Trait definitions — the shared contract
│       │   │   ├── mod.rs                         # re-exports all traits
│       │   │   ├── bundle.rs                      # Ops bundle struct
│       │   │   ├── attention.rs                   # trait AttentionOps, VarlenParams, PagedParams, MaskType
│       │   │   ├── kv_cache.rs                    # trait KvCacheOps, CacheSlotSpec, LayerCacheSpec
│       │   │   ├── gemm.rs                        # trait GemmOps, QuantScheme
│       │   │   ├── norm.rs                        # trait NormOps (rms_norm, layer_norm, group_norm)
│       │   │   ├── activation.rs                  # trait ActivationOps (silu, gelu, softmax)
│       │   │   ├── conv.rs                        # trait ConvOps (conv1d, conv2d, conv_transpose1d)
│       │   │   ├── comm.rs                        # trait CommOps (all_reduce, all_gather, all_to_all, send/recv)
│       │   │   ├── fused.rs                       # trait FusedOps — all methods default { None }
│       │   │   └── session.rs                     # trait OpsSession (begin/end_forward, precompute_paged_plan)
│       │   ├── cpu_ops.rs                         # CpuOps implementation
│       │   ├── cpu/                               # CPU kernel implementations (attention, GEMM, quant, etc.)
│       │   └── onednn/                            # OneDNN FFI bindings
│       │
│       ├── tensor.rs                              # candle_core abstraction layer (re-exports Tensor, Device, etc.)
│       │
│       ├── modules/                               # Shared modules — fusion/fallback logic lives here
│       │   ├── mod.rs                             # re-exports
│       │   ├── norm.rs                            # residual_norm, residual_layer_norm, adaln_zero, adaln_continuous
│       │   ├── attn_utils.rs                      # qk_norm_rope, knorm_rope_cache_write, apply_rope, split_qkv
│       │   ├── mlp.rs                             # gated_mlp, gelu_mlp
│       │   ├── linear.rs                          # struct Linear (unified: TP + quant + LoRA), apply_lora
│       │   └── moe.rs                             # moe_layer (Local / ExpertParallel / Disaggregated)
│       │
│       ├── engine/                                # Engine — the public API
│       │   ├── mod.rs                             # pub struct Engine: new(), serve(), generate(), embed()
│       │   ├── config.rs                          # EngineConfig — mode, device, scheduler, spec decode, grammar
│       │   ├── run.rs                             # run::ar(), run::dllm(), run::diffusion(), ... main loops
│       │   ├── model_runner/                      # ScheduledBatch → tensors → model.forward() → sample
│       │   │   ├── mod.rs                         # ModelRunner core logic
│       │   │   └── cuda_graph.rs                  # Graph capture/replay (CUDA/HIP)
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
│       │   └── components/                        # Optional components, used by schedulers as needed
│       │       ├── block_allocator.rs             # BlockAllocator — paged KV cache block management
│       │       ├── prefix_cache.rs                # PrefixCache — radix tree for KV prefix sharing
│       │       └── request_queue.rs               # RequestQueue — FCFS / priority / cache-aware ordering
│       │
│       ├── disaggregated/                         # Multi-instance deployment (skip for single-machine)
│       │   ├── pd/                                # Prefill/Decode separation
│       │   │   ├── coordinator.rs                 # Cross-worker request routing
│       │   │   └── kv_transfer.rs                 # KV cache cross-worker transfer protocol
│       │   └── afd/                               # Attention-FFN separation
│       │       └── ffn_follower.rs                # Passive FFN process
│       │
│       └── models/                                # Model implementations — see ../model_registry.md
│           ├── mod.rs                             # trait Model { fn forward(.., ops: &Ops) }
│           ├── registry.rs                        # ModelRegistry: inventory-based auto-registration
│           ├── weight_loader.rs                   # WeightLoader: safetensors/GGUF → model struct
│           ├── config.rs                          # ModelConfig: parsed from HF config.json
│           ├── qwen3.rs                           # Qwen3 (GQA + QK-norm + MoE)
│           ├── qwen3_moe.rs                       # Qwen3-MoE layers (EP / AFD via modules::moe_layer)
│           ├── qwen35.rs                          # Qwen3.5 hybrid (DeltaNet + softmax per-layer dispatch)
│           ├── llama.rs                           # Llama-3 (also: Phi3, InternLM3, Yi, Mistral)
│           ├── gemma3.rs                          # Gemma3 (softcap, sliding window, alternating attention)
│           ├── deepseek_v3.rs                     # DeepSeek-V3 (MLA + MoE + EP)
│           ├── flux.rs                            # Flux DiT (double/single stream, joint attention, AdaLN)
│           ├── hunyuan_video.rs                   # HunyuanVideo (spatial + temporal attention, AdaLN)
│           ├── whisper.rs                         # Whisper (encoder-decoder, cross-attention)
│           ├── bge.rs                             # BGE/GTE (encoder-only, embedding)
│           ├── llada2.rs                          # LLaDA2 (diffusion LLM, bidirectional demasking)
│           ├── qwen3_tts.rs                       # Qwen3-TTS (talker + code predictor + Code2Wav)
│           └── qwen3_omni.rs                      # Qwen3-Omni (vision encoder + audio encoder + thinker)
│
├── third_party/                               # All vendored third-party source (git submodules)
│   ├── flashinfer/                            # FlashInfer (attention, SM80/SM90)
│   ├── flash-attention/                       # FA4 (attention, TVM AOT, SM90+)
│   ├── DeepGEMM/                              # DeepGEMM (BF16/FP8 GEMM, SM90+)
│   ├── cutlass/                               # CUTLASS (GEMM + conv, header-only)
│   ├── composable_kernel/                     # CK (GEMM + attention, ROCm, header-only)
│   ├── aiter/                                 # aiter (flash attention, ROCm gfx942/950)
│   ├── nccl/                                  # NCCL (collective communication, CUDA)
│   ├── rccl/                                  # RCCL (collective communication, ROCm)
│   ├── uccl/                                  # UCCL-EP (MoE expert-parallel, cross-device)
│   ├── onednn/                                # OneDNN (GEMM + conv + fused ops, CPU)
│   └── xgrammar/                              # xgrammar (constrained decoding grammar engine)
│
├── plugins/                                       # Device-agnostic C++ FFI crates
│   └── prelude-xgrammar/                          # Constrained decoding (compiles third_party/xgrammar/)
│       ├── build.rs                               # cc crate compiles xgrammar C++
│       └── src/lib.rs                             # impl GrammarBackend for XGrammarBackend
│
├── prelude-cuda/                              # CUDA device impl
│   ├── src/                                   # CudaOps — dispatch layer
│   │   ├── lib.rs                             # pub fn create_ops(config) -> Ops
│   │   ├── cuda_ops.rs                        # struct CudaOps, select_attention_backend(), trait impls
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
│   │   │   ├── flash_v4.rs                    # FA4 CuTeDSL wrappers
│   │   │   ├── flash_v3.rs                    # FA3 Hopper wrappers
│   │   │   ├── flash_v2.rs                    # FA2 Ampere+ wrappers
│   │   │   ├── flashinfer.rs                  # FlashInfer wrappers
│   │   │   └── paged.rs                       # paged cache ops
│   │   └── kernels/kernels_src/               # .cu source files (organized by subdirectory)
│   │
│   ├── fa4/                                   # build.rs compiles third_party/flash-attention/
│   ├── flashinfer/                            # build.rs compiles third_party/flashinfer/
│   ├── deepgemm/                              # build.rs compiles third_party/DeepGEMM/
│   ├── cutlass-gemm/                          # BF16/FP16 GEMM via third_party/cutlass/
│   ├── quant-gemm/                            # GGUF quantized GEMM + dequant (llama.cpp MMQ kernels)
│   ├── nccl/                                  # build.rs links third_party/nccl/
│   └── uccl-ep/                               # build.rs compiles third_party/uccl/ep/
│
├── prelude-rocm/                              # ROCm device impl
│   ├── src/                                   # RocmOps — dispatch layer
│   │   ├── lib.rs                             # pub fn create_ops(config) -> Ops
│   │   ├── rocm_ops.rs                        # struct RocmOps, enum RocmArch (gfx942/950/1100/1200)
│   │   ├── attention.rs                       # impl AttentionOps: aiter → CK flash attn fallback
│   │   ├── gemm.rs                            # impl GemmOps: CK GEMM, FP8 FNUZ/E4M3 auto-select
│   │   ├── comm.rs                            # impl CommOps: RCCL + UCCL-EP fused dispatch
│   │   ├── fused.rs                           # impl FusedOps: fused_add_rmsnorm (HIP kernel)
│   │   └── ...                                # norm, activation, conv, session, graph (HIP graphs)
│   │
│   ├── ck/                                    # build.rs compiles third_party/composable_kernel/
│   ├── aiter/                                 # build.rs compiles third_party/aiter/
│   ├── rccl/                                  # build.rs links third_party/rccl/
│   └── uccl-ep/                               # build.rs compiles third_party/uccl/ep/
│
├── prelude-metal/                             # Metal device impl (Apple Silicon)
│   ├── src/                                   # MetalOps — dispatch layer
│   │   ├── lib.rs                             # pub fn create_ops(config) -> Ops
│   │   ├── metal_ops.rs                       # struct MetalOps (unified memory, simdgroup mm)
│   │   ├── attention.rs                       # impl AttentionOps: MSL flash attn (varlen only)
│   │   ├── gemm.rs                            # impl GemmOps: simdgroup matmul + in-shader Q4K dequant
│   │   ├── fused.rs                           # impl FusedOps: fused_add_rmsnorm, fused_silu_mul
│   │   └── ...                                # norm, activation, conv
│   └── shaders/                               # MSL compute shaders (*.metal)
│
├── prelude-vulkan/                            # Vulkan device impl (cross-vendor)
│   ├── src/                                   # VulkanOps — dispatch layer
│   │   ├── lib.rs                             # pub fn create_ops(config) -> Ops
│   │   ├── vulkan_ops.rs                      # struct VulkanOps (cooperative_matrix, subgroup_size)
│   │   ├── attention.rs                       # impl AttentionOps: GLSL flash attn (scalar + coopmat)
│   │   ├── gemm.rs                            # impl GemmOps: tiled / coopmat + in-shader Q4 dequant
│   │   └── ...                                # norm, activation, conv
│   └── shaders/                               # GLSL → SPIR-V compute shaders
│
├── prelude-tpu/                               # TPU device impl (XLA/Pallas)
│   └── src/
│       ├── lib.rs                             # pub fn create_ops(config) -> Ops
│       ├── tpu_ops.rs                         # struct TpuOps (PjrtClient, compiled_cache)
│       ├── attention.rs                       # impl AttentionOps: Pallas flash attn + ragged_paged
│       ├── gemm.rs                            # impl GemmOps: XLA dot_general (MXU)
│       ├── session.rs                         # impl OpsSession: XLA trace begin/end + compile cache
│       └── ...                                # FusedOps: all None (XLA auto-fuses)
│
└── prelude-cpu/                               # CPU device impl
    ├── src/                                   # CpuOps — dispatch layer
    │   ├── lib.rs                             # pub fn create_ops(config) -> Ops
    │   ├── cpu_ops.rs                         # struct CpuOps
    │   ├── attention.rs                       # impl AttentionOps: matmul-based SDPA (no paged)
    │   ├── gemm.rs                            # impl GemmOps: OneDNN GEMM + dequant fallback
    │   ├── fused.rs                           # impl FusedOps: fused_add_rmsnorm (vectorized)
    │   └── ...                                # norm, activation, conv
    │
    └── onednn/                                # build.rs compiles/links third_party/onednn/
```

**Reading guide:**

- **`prelude-server/`** — binary crate, composition root. Runtime `detect_gpu()` selects
  the device backend, calls `prelude_cuda::create_ops()` (or rocm/metal/...) and passes
  `Ops` + `GrammarBackend` to `Engine::new()`. Everything else is device-agnostic.

- **`prelude-core/src/ops/`** — the shared contract. Trait definitions live in `ops/traits/`,
  with `CpuOps` implementation in `ops/cpu_ops.rs` and CPU kernels in `ops/cpu/`.
  Model devs, module devs, device impl devs all start with the trait signatures.
  `tensor.rs` provides the candle_core abstraction layer. **No dependency on any device crate.**

- **`prelude-core/src/modules/`** — shared modules. Contain fusion/fallback logic
  (`FusedOps` match + `None` fallback). Models compose these instead of calling raw ops.
  One optimization in a module → all models that use it benefit.

- **`prelude-core/src/models/`** — model implementations. Device-agnostic, kernel-agnostic.
  Zero `#[cfg]` flags. Only depend on `modules/` and `ops/` traits.

- **`prelude-{cuda,rocm,metal,vulkan,tpu,cpu}/`** — one crate per device target. Each
  implements all op traits and exports `create_ops(config) -> Ops`. Features are
  **additive** — `--features cuda,rocm` builds both. Runtime auto-detects GPU.

- **`plugins/`** — device-agnostic C++ FFI crates that implement prelude-core traits.
  Currently only `prelude-xgrammar` (constrained decoding). Uses `cc` crate, no GPU toolchain.

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
   should benefit automatically without per-model code changes. This is achieved via shared
   modules — optimize one building block, all models that use it benefit. O(1) change → O(N) benefit.

## Goals

1. Model code is device-agnostic: no `#[cfg(feature = "cuda")]` in models.
2. Model code is kernel-agnostic: models never reference FA4, FlashInfer, CUTLASS, etc.
3. Each operation independently dispatches to the best available kernel for the device + parameters.
4. Multi-device: CUDA, ROCm, TPU, Vulkan, CPU share the same model code.
5. Multi-model: AR (LLM), diffusion, TTS, vision all use the same op traits.
6. Fusion is explicit: models control fusion boundaries, not the dispatch layer.

## Layering

```
Model code          composes      Modules (shared layers)
Modules     call          Op traits (+ FusedOps fallback logic)
Op traits           implemented by    Device ops (CudaOps, RocmOps, CpuOps, ...)
Device ops          dispatches to     Kernel libraries (FA4, FlashInfer, DeepGEMM, CUTLASS, CK, XLA, ...)
```

**Modules** are shared layer implementations (e.g., `ResidualNormBlock`, `GatedMLP`, `AttentionBlock`).
They contain the `FusedOps` match/fallback logic. Models compose modules instead of calling
raw ops. **When a new fused kernel is added, the building block is updated once, and all models
that use it benefit automatically.** This is how one kernel optimization reaches many models.

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

### modules/ — Shared Modules
| Section | File |
|---------|------|
| **Module Catalog & Usage** | [modules/modules.md](modules/modules.md) |
| **LoRA (Low-Rank Adaptation)** | [modules/lora.md](modules/lora.md) |
| **Distributed Execution** | [modules/distributed.md](modules/distributed.md) |

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
| **Summary** | [summary.md](summary.md) |
| **Model Registry & Loading** | [../model_registry.md](../model_registry.md) |

## Dependency Graph

```
prelude-server (binary, composition root)
    ├── prelude-core                  (traits, modules, models, engine, scheduler — pure Rust)
    ├── plugins/prelude-xgrammar      (impl GrammarBackend, compiles third_party/xgrammar/)
    ├── prelude-cuda                  (feature-gated, additive — can enable multiple)
    │       ├── prelude-core              (for trait definitions)
    │       ├── fa4/, flashinfer/, deepgemm/, nccl/, uccl-ep/
    │       └── (each sub-crate compiles from third_party/)
    ├── prelude-rocm                  (feature-gated, additive)
    │       ├── prelude-core
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-cpu                   (always included as fallback)

third_party/                          (git submodules, source only — not Cargo crates)
    ├── flashinfer/                   (compiled by prelude-cuda/flashinfer/build.rs)
    ├── composable_kernel/            (compiled by prelude-rocm/ck/build.rs)
    ├── uccl/                         (compiled by prelude-cuda AND prelude-rocm — cross-device)
    ├── xgrammar/                     (compiled by plugins/prelude-xgrammar/build.rs)
    └── ...
```

**Key rules:**
- `prelude-core` depends on NO device crate and compiles NO C++ (pure Rust leaf).
- Engine receives `Ops` + `GrammarBackend` via dependency injection at startup.
- Device features are **additive**, not exclusive — `--features cuda,rocm` builds both.
- NCCL/RCCL are **dlopen'd** at runtime (not statically linked), avoiding symbol conflicts.
- Cross-device libraries (UCCL-EP) have separate sub-crates per device with shared
  `third_party/` source; FFI bindings are ~400 lines each (API stable, duplication acceptable).

## Build Targets

Two binaries cover all platforms. Install script auto-detects and downloads the right one.

| Binary | Build Platform | Device Backends | Features |
|--------|---------------|-----------------|----------|
| `prelude-linux-x86_64` | Linux | CUDA + ROCm + Vulkan + CPU | `--features cuda,rocm,vulkan` |
| `prelude-darwin-aarch64` | macOS | Metal + CPU | `--features metal` |

Runtime auto-detection selects the best available backend:
```rust
// prelude-server/src/main.rs
let ops = match detect_gpu() {
    Gpu::Nvidia => prelude_cuda::create_ops(&config),
    Gpu::Amd    => prelude_rocm::create_ops(&config),
    Gpu::None   => prelude_cpu::create_ops(&config),
};
let grammar = prelude_xgrammar::create_backend();
let engine = Engine::new(ops, grammar, &config);
```

Metal and CUDA/ROCm cannot coexist in one binary (macOS SDK vs Linux GPU toolchains).

## Dependency Summary

```
prelude-server            →  composition root: detect GPU, create_ops, Engine::new
plugins/prelude-xgrammar  →  impl GrammarBackend (device-agnostic C++ FFI)
prelude-core/models       →  device-agnostic model code, calls modules + Ops
prelude-core/modules      →  shared layers (Linear, residual_norm, moe_layer), fusion fallback
prelude-core/ops          →  trait definitions in ops/traits/ (AttentionOps, GemmOps, FusedOps, ...) + CpuOps impl
prelude-core/engine       →  Engine, ModelRunner, SpecDecodeRunner, Sampler, GrammarManager
prelude-core/scheduler    →  ArScheduler, DllmScheduler, BlockAllocator, PrefixCache
prelude-{device}/         →  device impl (CudaOps, RocmOps, ...), kernel sub-crates
```

Each layer only knows the layer directly below it. Models don't know devices.
Modules don't know models. Device impls don't know each other.

