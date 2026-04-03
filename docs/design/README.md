# Prelude Architecture

## File Layout

```
crates/
├── prelude-server/                                # Binary — composition root
│   └── src/
│       └── main.rs                                # Engine::new(config) → engine.serve()
│
├── prelude-core/                                  # Core: traits, modules, models, engine, scheduler (pure Rust)
│   └── src/
│       ├── lib.rs                                 # pub use engine::Engine;
│       │
│       ├── ops/                                   # Op trait definitions + built-in fallback
│       │   ├── mod.rs                             # register_cpu_ops/gpu_ops, select_ops(&device)
│       │   ├── traits/                            # Trait definitions — the shared contract
│       │   │   ├── mod.rs                         # re-exports all traits
│       │   │   ├── bundle.rs                      # Ops bundle struct (all 9 Arc<dyn Trait> fields)
│       │   │   ├── attention.rs                   # trait AttentionOps, VarlenParams, PagedParams, MaskType
│       │   │   ├── kv_cache.rs                    # trait KvCacheOps, CacheSlotSpec, LayerCacheSpec
│       │   │   ├── gemm.rs                        # trait GemmOps, QuantScheme
│       │   │   ├── norm.rs                        # trait NormOps (rms_norm, layer_norm, group_norm)
│       │   │   ├── activation.rs                  # trait ActivationOps (silu, gelu, softmax)
│       │   │   ├── conv.rs                        # trait ConvOps (conv1d, conv2d, conv_transpose1d)
│       │   │   ├── comm.rs                        # trait CommOps (all_reduce, all_gather, all_to_all, send/recv)
│       │   │   ├── fused.rs                       # trait FusedOps — all methods default { None }
│       │   │   └── session.rs                     # trait OpsSession (begin/end_forward, precompute_paged_plan)
│       │   └── naive_ops.rs                       # Built-in fallback: matmul SDPA, candle ops (no device dep)
│       │
│       ├── tensor.rs                              # candle_core abstraction layer (re-exports Tensor, Device, etc.)
│       │
│       ├── modules/                               # Shared modules — fusion/fallback logic lives here
│       │   ├── mod.rs                             # re-exports, PagedKvContext, BatchAttnContext
│       │   ├── norm.rs                            # fast_rms_norm, fused_add_rmsnorm, fast_silu_mul, debug flags
│       │   ├── attn_utils.rs                      # RotaryEmbedding, fused_qkv_projection, qknorm_rope_varlen,
│       │   │                                      #   varlen_attention, paged_attention, windowed, bidirectional
│       │   ├── mlp.rs                             # GatedMlp (gate_proj, up_proj, down_proj)
│       │   ├── linear.rs                          # struct Linear (unified: TP + quant + LoRA), RmsNorm
│       │   ├── transformer_block.rs               # TransformerBlock (pre-norm decoder: norm→attn→fused_add_norm→MLP)
│       │   └── moe.rs                             # moe_layer (Local / ExpertParallel / Disaggregated)
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
│       └── models/                                # Model implementations — see ../model_registry.md
│           ├── mod.rs                             # trait Model { fn forward(.., ops: &Ops) }
│           ├── registry.rs                        # ModelRegistry: inventory-based auto-registration (ArchSpec trait)
│           │                                      #   ArchSpec: parse_config, build_model, build_model_gguf, runtime_caps
│           ├── qwen3.rs                           # Qwen3: layers + forward + ArchSpec + GGUF (GQA + QK-norm)
│           ├── qwen3_moe.rs                       # Qwen3-MoE (EP / AFD via modules::moe_layer)
│           ├── qwen3_5.rs                         # Qwen3.5 hybrid (DeltaNet + softmax per-layer dispatch)
│           ├── gemma3.rs                          # Gemma3 (softcap, sliding window, alternating attention)
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
├── prelude-cuda/                              # CUDA device impl (Ops + Executor)
│   ├── src/
│   │   ├── lib.rs                             # ctor auto-registers CudaOps + CudaExecutor at link time
│   │   ├── cuda_ops.rs                        # struct CudaOps, impl all 9 op traits
│   │   ├── executor.rs                          # CudaExecutor: GPU queue, double buffering, CUDA graph
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
│   │   ├── lib.rs                             # ctor auto-registers RocmOps + RocmExecutor
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
│   │   ├── lib.rs                             # ctor auto-registers MetalOps + MetalExecutor
│   │   ├── metal_ops.rs                       # struct MetalOps (unified memory, simdgroup mm)
│   │   ├── executor.rs                          # MetalExecutor: Metal command buffer encoding
│   │   └── ...                                # attention, gemm, fused, norm, activation, conv
│   └── shaders/                               # MSL compute shaders (*.metal)
│
├── prelude-vulkan/                            # Vulkan device impl (cross-vendor)
│   ├── src/
│   │   ├── lib.rs                             # ctor auto-registers VulkanOps + VulkanExecutor
│   │   ├── vulkan_ops.rs                      # struct VulkanOps (cooperative_matrix, subgroup_size)
│   │   ├── executor.rs                          # VulkanExecutor: Vulkan command buffer + compute pipeline
│   │   └── ...                                # attention, gemm, norm, activation, conv
│   └── shaders/                               # GLSL → SPIR-V compute shaders
│
├── prelude-tpu/                               # TPU device impl (XLA/Pallas)
│   └── src/
│       ├── lib.rs                             # ctor auto-registers TpuOps + TpuExecutor
│       ├── tpu_ops.rs                         # struct TpuOps (PjrtClient, compiled_cache)
│       ├── executor.rs                          # TpuExecutor: XLA trace + compile cache
│       └── ...                                # attention, gemm, session (FusedOps: all None, XLA auto-fuses)
│
└── prelude-cpu/                               # CPU device impl
    ├── src/
    │   ├── lib.rs                             # ctor auto-registers CpuOps + CpuExecutor
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

- **`prelude-server/`** — binary crate, composition root. Has zero device-specific code.
  Device crates auto-register their `Ops` + `Executor` via `ctor` at link time.
  Server just creates `Engine::new(config)` — engine calls `select_ops(&device)` internally.

- **`prelude-core/src/ops/`** — the shared contract. Trait definitions live in `ops/traits/`,
  with `naive_ops.rs` as the built-in fallback (matmul-based attention, candle ops).
  Device crates register optimized implementations via `register_cpu_ops()` / `register_gpu_ops()`.
  `tensor.rs` provides the candle_core abstraction layer. **No dependency on any device crate.**

- **`prelude-core/src/engine/executor.rs`** — trait `Executor` defines how scheduled batches are
  executed on a device. Core provides device-agnostic scheduling loops in `engine/run/`
  (batch mode, continuous batching). Device crates implement `Executor` with device-specific
  optimizations (GPU queue, CUDA graphs, HIP graphs, Metal command buffers).

- **`prelude-core/src/modules/`** — shared modules. Contain fusion/fallback logic
  (`FusedOps` match + `None` fallback). Models compose these instead of calling raw ops.
  One optimization in a module → all models that use it benefit.

- **`prelude-core/src/models/`** — model implementations. Device-agnostic, kernel-agnostic.
  Zero `#[cfg]` flags. Only depend on `modules/` and `ops/` traits. Each model is one flat
  file containing layers, forward, and `ArchSpec` (registry metadata). GGUF loading lives
  in `loading/gguf/`, not in model files.

- **`prelude-{cuda,rocm,metal,vulkan,tpu,cpu}/`** — one crate per device target. Each
  implements `Ops` (kernel dispatch) + `Executor` (execution strategy), auto-registered via `ctor`.
  Features are **additive** — `--features cuda,rocm` builds both. Runtime auto-detects GPU.

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
Engine              starts             Run loop (core: batch or continuous batching)
Run loop            calls              Executor::submit/collect (device crate, via ctor)
Run loop            calls              Scheduler (core, pure CPU scheduling decisions)
Executor            calls              Model code (prepare tensors → model.forward())
Model code          composes          Modules (shared layers)
Modules             call              Op traits (+ FusedOps fallback logic)
Op traits           implemented by    Device ops (CudaOps, RocmOps, CpuOps, ...)
Device ops          dispatches to     Kernel libraries (FA4, FlashInfer, DeepGEMM, CUTLASS, CK, XLA, ...)
```

**Two trait boundaries** separate core from device crates:

1. **`Ops`** (kernel dispatch) — defines WHAT operations are available (attention, GEMM, norm, ...).
   Device crates implement with optimized kernels. Models and modules call through this.

2. **`Executor`** (device execution) — defines HOW batches are submitted to and collected from
   a device. `submit()` is non-blocking (GPU queues work, CPU runs inline). `collect()` awaits
   completion. Core's run loops use submit/collect to naturally get double-buffering on GPU
   (prepare batch N+1 while device runs batch N) and sequential execution on CPU.

**Modules** are shared layer implementations (e.g., `TransformerBlock`, `GatedMlp`, `Linear`).
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
prelude-server (binary, composition root — zero device-specific code)
    ├── prelude-core                  (traits, modules, models, engine, scheduler — pure Rust)
    ├── plugins/prelude-xgrammar      (impl GrammarBackend, compiles third_party/xgrammar/)
    ├── prelude-cuda                  (feature-gated, additive — Ops + Executor, ctor auto-registers)
    │       ├── prelude-core              (for trait definitions)
    │       ├── fa4/, flashinfer/, deepgemm/, cutlass-gemm/, quant-gemm/, cula/
    │       └── (each sub-crate compiles from third_party/)
    ├── prelude-rocm                  (feature-gated, additive)
    │       ├── prelude-core
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-cpu                   (always included as fallback)
            └── prelude-core

third_party/                          (git submodules, source only — not Cargo crates)
    ├── flashinfer/                   (compiled by prelude-cuda/flashinfer/build.rs)
    ├── flash-attention/              (compiled by prelude-cuda/fa4/build.rs)
    ├── tvm-ffi/                      (TVM FFI runtime, used by FA4/FlashInfer/cuLA build.rs)
    ├── composable_kernel/            (compiled by prelude-rocm/ck/build.rs)
    ├── uccl/                         (compiled by prelude-cuda AND prelude-rocm — cross-device)
    ├── xgrammar/                     (compiled by plugins/prelude-xgrammar/build.rs)
    └── ...
```

**Key rules:**
- `prelude-core` depends on NO device crate and compiles NO C++ (pure Rust leaf).
- Device crates auto-register `Ops` + `Executor` via `ctor` at link time — no explicit init in server.
- Engine calls `select_ops(&device)` internally to pick the best registered implementation.
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
// Device crates (prelude-cuda, prelude-cpu, ...) auto-register via ctor at link time.
// Server has ZERO device-specific code — no imports from device crates, no cfg gates.
let engine = Engine::new(&config);
engine.serve().await;
```

Metal and CUDA/ROCm cannot coexist in one binary (macOS SDK vs Linux GPU toolchains).

## Dependency Summary

```
prelude-server            →  composition root: Engine::new(config), zero device code
plugins/prelude-xgrammar  →  impl GrammarBackend (device-agnostic C++ FFI)
prelude-core/models       →  device-agnostic model code, calls modules + Ops
prelude-core/modules      →  shared layers (Linear, TransformerBlock, GatedMlp), fusion fallback
prelude-core/ops          →  trait definitions (AttentionOps, GemmOps, FusedOps, ...) + naive_ops fallback
prelude-core/engine       →  Engine, Executor trait, run loops (batch/continuous), Sampler
prelude-core/scheduler    →  ArScheduler, BlockManager, PrefixCache, DeltaNetPool
prelude-{device}/         →  Ops impl + Executor impl (execution strategy), kernel sub-crates
```

Each layer only knows the layer directly below it. Models don't know devices.
Modules don't know models. Device impls don't know each other.

