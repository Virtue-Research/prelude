# Developer Guide

This page is the starting point for contributors. It covers the repository layout, design principles, system layering, and where to look for what.

## Design Principles

1. **Subsystem isolation:** Each subsystem is fully self-contained. A contributor working on one subsystem should not need to read or understand any other. Model devs don't read kernel code; kernel devs don't read model code. The trait signatures are the complete contract between subsystems. In practice: model code is device-agnostic (no `#[cfg(feature = "cuda")]`) and kernel-agnostic (no direct references to FA4, FlashInfer, or CUTLASS).

2. **Kernel optimization reach:** Adding a kernel optimization should benefit as many models as possible without per-model changes. This is achieved via `OpsBundle` — add a fused kernel once, and every model calling `ops.xxx()` benefits automatically. O(1) change → O(N) benefit.

3. **Multi-device, multi-modal:** CUDA, ROCm, TPU, Vulkan, and CPU all share the same model code. AR (LLM), diffusion, TTS, and vision all use the same op traits. Each operation independently dispatches to the best available kernel for the active device and parameters; fusion is transparent to model code.

## Repository Layout

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
├── prelude-core/                                  # Core: ops, models, engine, scheduler (pure Rust, CubeCL)
│   └── src/
│       ├── lib.rs                                 # pub use engine::Engine;
│       │
│       ├── ops/                                   # Op trait definitions + built-in implementations
│       │   ├── mod.rs                             # register_backend(), select_ops(), thread-local Ops context
│       │   ├── traits/                            # Trait definitions — the shared contract
│       │   │   ├── mod.rs                         # re-exports all traits
│       │   │   ├── bundle.rs                      # OpsBundle — flat API (ops.exp, ops.rms_norm, ops.fused_add_rmsnorm)
│       │   │   │                                  #   Fused ops: try device kernel → fallback to composed
│       │   │   ├── tensor_ops.rs                  # trait TensorOps — base() delegation, device overrides only what it needs
│       │   │   ├── attention.rs                   # trait AttentionOps, VarlenParams, PagedParams, MaskType
│       │   │   ├── kv_cache.rs                    # trait KvCacheOps, CacheSlotSpec
│       │   │   ├── gemm.rs                        # trait GemmOps (quantized_matmul, grouped_gemm — no plain matmul)
│       │   │   ├── norm.rs                        # trait NormOps (rms_norm, layer_norm, group_norm)
│       │   │   ├── activation.rs                  # trait ActivationOps (silu, gelu, softmax, sigmoid)
│       │   │   ├── conv.rs                        # trait ConvOps (conv1d, conv2d, conv_transpose1d)
│       │   │   ├── comm.rs                        # trait CommOps (all_reduce, all_gather, all_to_all)
│       │   │   ├── fused.rs                       # trait FusedOps — all methods return Option, default None
│       │   │   └── session.rs                     # trait OpsSession (begin/end_forward)
│       │   ├── composed/                          # ComposedOps — default impls via TensorOps composition
│       │   │   ├── mod.rs                         # ComposedOps struct + Activation/Gemm/stubs
│       │   │   ├── attention.rs                   # varlen SDPA (matmul + softmax, causal, GQA)
│       │   │   ├── conv.rs                        # conv1d/conv2d (im2col + matmul), conv_transpose1d
│       │   │   └── norm.rs                        # rms_norm, layer_norm, group_norm
│       │   └── primitives/                        # TensorOps implementation via CubeCL
│       │       ├── mod.rs                         # CubeCLTensorOps<R: Runtime> + default_cpu_ops()
│       │       ├── elementwise.rs                 # Op family traits + LinearView kernels (burn-cubecl pattern)
│       │       ├── reduce.rs                      # cubek::reduce wrapper
│       │       └── matmul.rs                      # cubek::matmul wrapper
│       │
│       ├── tensor/                                # Own tensor library (Storage, Layout, DType, Device, Error)
│       │
│       ├── engine/                                # Engine — the public API + execution loops
│       │   ├── mod.rs                             # pub struct Engine, trait InferenceEngine
│       │   ├── config.rs                          # EngineConfig — mode, device, scheduler, spec decode, grammar
│       │   ├── weight_loader.rs                   # WeightLoader: safetensors + GGUF → Tensor by name
│       │   ├── executor.rs                        # trait Executor { submit(batch) -> Handle, collect(Handle) -> Output }
│       │   ├── run/                               # Scheduling-paradigm loops (device-agnostic, all call Executor)
│       │   │   ├── ar.rs                          # AR LLM: scheduler.step → submit prefill/decode → sample
│       │   │   ├── dllm.rs                        # Diffusion LLM: iterative demasking loop
│       │   │   ├── diffusion.rs                   # Image/video: denoising loop
│       │   │   └── tts.rs                         # TTS: multi-stage pipeline
│       │   ├── speculative/                       # Speculative decoding
│       │   │   ├── mod.rs                         # SpecDecodeRunner: draft → verify → accept loop
│       │   │   ├── proposer.rs                    # trait DraftProposer (EAGLE/DraftModel/Ngram/Medusa)
│       │   │   ├── rejection.rs                   # Rejection sampling (strict, probabilistic)
│       │   │   └── tree.rs                        # Tree attention mask construction
│       │   └── sampling/                          # Sampling orchestration
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
│       │       ├── cache/                         # KV cache management subsystem
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
├── plugins/                                   # Device-agnostic FFI crates
│   └── prelude-mooncake/                      # Mooncake Transfer Engine (KV cache transport)
│       ├── build.rs                           # bindgen generates FFI from transfer_engine_c.h
│       └── src/lib.rs                         # impl KvTransfer for MooncakeTransfer
│
├── prelude-cuda/                              # CUDA device impl (Ops + Executor)
│   ├── src/
│   │   ├── lib.rs                             # register() registers CudaOps + CudaExecutor with priority/probe
│   │   ├── device.rs                          # CUDA runtime: CudaStorage, stream/device registry, PTX loading
│   │   ├── cuda_ops.rs                        # struct CudaOps, impl all 9 op traits
│   │   ├── quant_backends.rs                  # GPU QuantFormat registration (inventory, priority=100)
│   │   ├── executor.rs                        # CudaExecutor: GPU queue, CUDA graph
│   │   ├── ops/                               # Kernel wrapper modules
│   │   │   ├── elementwise.rs                 # vectorized_add, fast_silu_mul
│   │   │   ├── rmsnorm.rs                     # fast_rmsnorm, fused_add_rmsnorm
│   │   │   ├── rope.rs                        # fused_qknorm_rope_varlen
│   │   │   ├── kv_cache.rs                    # fused_knorm_rope_kv_cache_write
│   │   │   ├── moe.rs                         # MoE routing ops
│   │   │   ├── gemm.rs                        # GEMM wrappers
│   │   │   ├── quant.rs                       # quantization ops
│   │   │   └── tiled_mmq.rs                   # tiled MMQ kernels (GGUF quantized GEMM)
│   │   ├── attn/                              # Attention backends (impl AttentionOps)
│   │   │   ├── flash_v4.rs                    # FA4 CuTeDSL wrappers (SM90+)
│   │   │   └── flashinfer.rs                  # FlashInfer wrappers (SM80+)
│   │   └── kernels/kernels_src/               # .cu source files
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
│   │   ├── executor.rs                        # RocmExecutor: HIP graphs, GPU queue
│   │   └── ...                                # attention, gemm, comm, fused, norm, activation, conv
│   ├── ck/                                    # build.rs compiles third_party/composable_kernel/
│   ├── aiter/                                 # build.rs compiles third_party/aiter/
│   ├── rccl/                                  # build.rs links third_party/rccl/
│   └── uccl-ep/                               # build.rs compiles third_party/uccl/ep/
│
├── prelude-metal/                             # Metal device impl (Apple Silicon)
│   ├── src/
│   │   ├── lib.rs                             # register() registers MetalOps + MetalExecutor with priority/probe
│   │   ├── metal_ops.rs                       # struct MetalOps (unified memory, simdgroup mm)
│   │   ├── executor.rs                        # MetalExecutor: Metal command buffer encoding
│   │   └── ...                                # attention, gemm, fused, norm, activation, conv
│   └── shaders/                               # MSL compute shaders (*.metal)
│
├── prelude-vulkan/                            # Vulkan device impl (cross-vendor)
│   ├── src/
│   │   ├── lib.rs                             # register() registers VulkanOps + VulkanExecutor with priority/probe
│   │   ├── vulkan_ops.rs                      # struct VulkanOps (cooperative_matrix, subgroup_size)
│   │   ├── executor.rs                        # VulkanExecutor: Vulkan command buffer + compute pipeline
│   │   └── ...                                # attention, gemm, norm, activation, conv
│   └── shaders/                               # GLSL → SPIR-V compute shaders
│
├── prelude-tpu/                               # TPU device impl (XLA/Pallas)
│   └── src/
│       ├── lib.rs                             # register() registers TpuOps + TpuExecutor with priority/probe
│       ├── tpu_ops.rs                         # struct TpuOps (PjrtClient, compiled_cache)
│       ├── executor.rs                        # TpuExecutor: XLA trace + compile cache
│       └── ...                                # attention, gemm, session (FusedOps: all None, XLA auto-fuses)
│
└── prelude-cpu/                               # CPU device impl
    ├── src/
    │   ├── lib.rs                             # register() registers CpuOps + CpuExecutor with priority/probe
    │   ├── cpu_ops.rs                         # struct CpuOps, impl all 9 op traits
    │   ├── executor.rs                        # CpuExecutor: simple block_in_place execution
    │   ├── ops/                               # CPU kernel implementations
    │   │   ├── attention/                     # AVX-512 / DPBF16 optimized attention
    │   │   ├── quant/                         # GGUF quantized matmul (Q4_0, Q4_K, Q6_K, IQ4_NL, ...)
    │   │   ├── rmsnorm.rs                     # Vectorized RMSNorm
    │   │   ├── rope.rs                        # RoPE (in-place, cos_sin_cache)
    │   │   ├── silu_mul.rs                    # Fused SiLU×Mul
    │   │   └── gemm.rs                        # OneDNN GEMM + dequant fallback
    │   └── linear_backends.rs                 # OnednnLinear, quant format registration (inventory)
    └── onednn-ffi/                            # OneDNN C++ FFI (compiled by build.rs)
```
<!-- 
**Key crates explained:**

- **`prelude-server/`** — binary crate, standalone composition root. Has zero device-specific code.
  Device crates register their Ops + Executor via `register()` at startup with priority/probe.
  Server just calls `register()` then creates `Engine::new(config)`.

- **`prelude-dynamo/`** — alternative binary for running as an NVIDIA Dynamo backend.
  Links `dynamo-runtime` + `prelude-core`. Implements Dynamo's `AsyncEngine` trait by wrapping
  `Engine`. Dynamo handles multi-node routing, P/D disaggregation orchestration, and KV transfers.
  Prelude handles single-worker inference.

- **`prelude-core/src/ops/`** — three layers:
  - `traits/` — trait definitions. `OpsBundle` provides a **flat API** (`ops.exp()`, `ops.rms_norm()`,
    `ops.varlen_attention()`, `ops.fused_add_rmsnorm()`). Models call `ops.xxx()` for everything —
    no `ops.attn.xxx()` nesting. Fused ops try device kernel → auto-fallback to composed.
    `TensorOps` uses `base()` delegation: device backends override only what they need.
  - `composed/` — `ComposedOps`: default impls for NormOps, ActivationOps, ConvOps, AttentionOps
    by composing TensorOps primitives. Pure logic, no device dependency.
  - `primitives/` — `CubeCLTensorOps<R: Runtime>`: TensorOps via CubeCL. Element-wise ops use
    burn-cubecl patterns (op family traits, LinearView, launch_unchecked). Reduce/matmul via cubek.
    CubeCL CPU runtime serves as lowest-priority fallback.

- **`prelude-core/src/models/`** — model implementations. **Self-contained first** (like vLLM): each
  model file has its own structs and forward logic, 1:1 mapping to HuggingFace transformers.
  Models call `ops.xxx()` directly for compute. `models/commons/` only shares what's universally
  common: weight containers (`Linear`, `Embedding`), `RotaryEmbedding`, and context structs.
  Model-specific components stay in model files — no forced abstraction.

- **`prelude-core/src/engine/executor.rs`** — trait `Executor` defines how scheduled batches are
  executed on a device. Core provides device-agnostic scheduling loops in `engine/run/`
  (batch mode, continuous batching). Device crates implement `Executor` with device-specific
  optimizations (GPU queue, CUDA graphs, HIP graphs, Metal command buffers).

- **`prelude-{cuda,rocm,metal,vulkan,tpu,cpu}/`** — one crate per device target. Each provides
  Ops (kernel dispatch) + Executor (execution strategy), registered at startup via `register()`.
  Features are **additive** — `--features cuda,rocm` builds both. Runtime probe auto-detects GPU.

- **`third_party/`** — all vendored third-party source (git submodules). Source only, not Cargo
  crates. Cross-device libraries (UCCL-EP) live here and are compiled by multiple device crates.
  NCCL/RCCL are **dlopen'd** at runtime.

- **`prelude-cuda/{fa4,flashinfer,...}/`** — kernel FFI sub-crates. Each has a `build.rs` that
  compiles from `third_party/` with the device toolchain (nvcc/hipcc), plus Rust FFI bindings.
  Only consumed by the parent device impl. -->

## Layering

<!-- TODO: layering is not very clear -->

```
Engine              starts             Run loop (core: batch or continuous batching)
Run loop            calls              Executor::submit/collect (device crate, via register())
Run loop            calls              Scheduler (core, pure CPU scheduling decisions)
Executor            calls              Model code (prepare tensors → model.forward())
Model code          calls              OpsBundle flat API (ops.rms_norm, ops.varlen_attention, ...)
OpsBundle           dispatches to      ComposedOps (default) or device overrides (CudaOps, ...)
ComposedOps         composes           TensorOps primitives → norm, conv, attention, activation
TensorOps           implemented by     CubeCL primitives (CUDA/ROCm/Vulkan/Metal/CPU) or XLA (TPU)
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

<!-- **`Linear` is a parameter carrier, OpsBundle is the decision maker.**
`Linear` holds weights + optional LoRA state. When fusion is needed (e.g. fused QKV+LoRA projection),
`Linear` passes its weights to OpsBundle (`ops.qkv_projection(x, weights, lora_state)`), and
OpsBundle handles fused/fallback/device dispatch. All decision logic lives in OpsBundle, never
in `Linear` or model code. This ensures LoRA, quantization, and fusion work transparently across
all devices without model-level branching. -->

## Dependency Graph

```
prelude-server (binary, standalone HTTP server — zero device-specific code)
    ├── prelude-core                  (OpsBundle, ComposedOps, CubeCLTensorOps<R>, models, engine, scheduler)
    │       ├── cubecl                    (pure Rust: IR + TensorOps primitives, generic over runtime)
    │       ├── cubek                     (pure Rust: pre-built CubeCL kernels for reduce + matmul)
    │       └── llguidance                (constrained decoding, pure Rust)
    ├── plugins/prelude-mooncake      (impl KvTransfer, wraps Mooncake Transfer Engine C API)
    ├── prelude-cuda                  (feature-gated, additive — hot-path overrides + Executor)
    │       ├── prelude-core
    │       ├── cubecl (features = ["cuda"])   (enables CubeCL CUDA runtime for TensorOps)
    │       ├── fa4/, flashinfer/, deepgemm/, cutlass-gemm/, quant-gemm/, cula/
    │       └── (each sub-crate compiles from third_party/)
    ├── prelude-rocm                  (feature-gated, additive)
    │       ├── prelude-core
    │       ├── cubecl (features = ["hip"])
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-tpu                   (feature-gated — XLATensorOps + hot-path overrides)
            ├── prelude-core              (uses ComposedOps, same pattern as all other backends)
            └── pjrt C API               (XLA runtime, dlopen libpjrt_tpu.so)

prelude-dynamo (binary, NVIDIA Dynamo backend — alternative entry point)
    ├── prelude-core
    ├── dynamo-runtime                (Dynamo's service discovery, transport, NIXL)
    ├── prelude-cuda / prelude-rocm / prelude-cpu
    └── (NO prelude-mooncake — Dynamo owns KV transfer via NIXL/Mooncake)

third_party/ (git submodules, source only — not Cargo crates)
    ├── flashinfer/                   (compiled by prelude-cuda/flashinfer/build.rs)
    ├── flash-attention/              (compiled by prelude-cuda/fa4/build.rs)
    ├── tvm-ffi/                      (TVM FFI runtime, used by FA4/FlashInfer/cuLA build.rs)
    ├── composable_kernel/            (compiled by prelude-rocm/ck/build.rs)
    ├── uccl/                         (compiled by prelude-cuda AND prelude-rocm — cross-device)
    └── ...
```
<!-- 
**Key rules:**

- `prelude-core` compiles no C++ and has no device types. It depends on CubeCL (pure Rust) for
  TensorOps primitives and llguidance (pure Rust) for constrained decoding. CubeCL's CPU runtime
  serves as fallback when no device crate is linked.
- Device crates register Ops + Executor via explicit `register()` calls at startup with priority/probe.
- Device features are **additive**, not exclusive — `--features cuda,rocm` builds both.
- NCCL/RCCL are **dlopen'd** at runtime (not statically linked), avoiding symbol conflicts.
- `prelude-mooncake` is feature-gated (`--features mooncake`). Only standalone multi-node
  disaggregated serving needs it. Single-machine and Dynamo-backend deployments don't build it. -->

## Key Subsystems

| Subsystem | Location | Description |
|-----------|----------|-------------|
| Engine hierarchy | `prelude-core/src/engine/mod.rs` | `InferenceEngine` trait, `Engine`, `ScheduledEngine` |
| Schedulers | `prelude-core/src/scheduler/` | Per-paradigm schedulers (AR, diffusion, TTS, embed/classify) |
| KV cache | `prelude-core/src/scheduler/components/cache/` | `BlockManager`, `PrefixKvCache`, paged + prefix caching |
| GPU queue | `prelude-cuda/src/executor.rs` | Single-threaded FIFO queue draining `GpuPacket`s |
| Attention backends | `prelude-cuda/src/attn/` | FA4, FlashInfer, FA3, FA2, CPU — modular dispatch |
| Ops traits | `prelude-core/src/ops/traits/` | `OpsBundle`, `AttentionOps`, `GemmOps`, `NormOps`, etc. |
| Model registry | `prelude-core/src/models/registry.rs` | `inventory`-based auto-registration via `ArchSpec` |

When working on sub-systems:

| Subsystem | Needs to know | Does NOT need to know |
|-----------|--------------|----------------------|
| **Model impl** (Qwen3, Flux, TTS) | Module APIs (`Linear`, `residual_norm`, ...), `OpsBundle` | Any device impl, kernel library, engine |
| **Modules** (Linear, residual_norm, gated_mlp, ...) | Op trait signatures, `FusedOps` match pattern | Device internals, model specifics |
| **Device crate** (CudaOps+Executor, RocmOps+Executor, ...) | Op traits, Executor trait, kernel library APIs | Model code, other devices |
| **Kernel wrapper** (FA4, FlashInfer, DeepGEMM) | Kernel library C API | Op traits, model code, other wrappers |
| **Comm backend** (NCCL, RCCL, XLA coll.) | `CommOps` trait, communication library API | Model code, attention kernels |
| **Engine/Scheduler** | `Executor` trait, `OpsSession`, model `forward()` signature | Kernel implementations, device internals |
| **KV Cache Manager** | Block allocation logic, `block_tables`/`slot_mapping` layout | Attention kernels, device code |

<!-- **Three design choices that enable this:**

1. **`FusedOps` default methods** — adding a new fusion only touches the trait definition (1 line with `{ None }` default) and the device that implements it. Other devices don't change. Model developer adds call site + fallback. No cross-team coordination needed beyond agreeing on the method signature.

2. **Tensor layout conventions** — formalized in the doc (above), not just comments. Every device implementation accepts and returns canonical layouts. Device-internal transformations (TPU 128-byte padding, Metal transpose) are invisible to callers.

3. **Graph capture is Executor-internal** — CUDA graph capture/replay lives in `CudaExecutor`, not in `OpsSession` or the engine's run loops. The engine calls `executor.submit()` / `executor.collect()` — whether that internally uses graph replay or individual kernel launches is the Executor's decision. Model code and schedulers never know. -->

## Integration 

- [integration](design/integration.md) — Integrate with external frameworks.

## Common Contribution Paths

<!-- TODO: what else to contribution -->

- [add](design/add.md) — Details about how to add model, kernal backend, and schedular mechanisms.

## Design Docs

For deeper internals, see the design docs:

- [Architecture Overview](design/overview.md) — request flow, engine hierarchy
- [Scheduler](design/schedular.md) — continuous batching, KV management
- [Models](design/models.md) — how models are structured and registered
- [Ops and Modules](design/ops.md) — the three-layer ops system
- [Devices](design/devices.md) — device crate structure and backend dispatch
