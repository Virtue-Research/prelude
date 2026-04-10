# Developer Guide

This page is the starting point for contributors. It covers the repository layout, design principles, system layering, and where to look for what.

## Design Principles

1. **Subsystem isolation:** Each subsystem is fully self-contained. A contributor working on one subsystem should not need to read or understand any other. Model devs don't read kernel code; kernel devs don't read model code. The trait signatures are the complete contract between subsystems. In practice: model code is device-agnostic (no `#[cfg(feature = "cuda")]`) and kernel-agnostic (no direct references to FA4, FlashInfer, or CUTLASS).

2. **Kernel optimization reach:** Adding a kernel optimization should benefit as many models as possible without per-model changes. This is achieved via the `Ops` trait (`&'static dyn Ops`) — add a fused kernel once, and every model calling `ops.xxx()` benefits automatically. O(1) change → O(N) benefit.

3. **Multi-device, multi-modal:** CUDA, ROCm, TPU, Vulkan, and CPU all share the same model code. AR (LLM), diffusion, TTS, and vision all use the same op traits. Each operation independently dispatches to the best available kernel for the active device and parameters; fusion is transparent to model code.

## Repository Layout

```
crates/
├── prelude-server/                                # Binary — standalone HTTP server (composition root)
│   └── src/
│       ├── main.rs                                # Engine::new(config) → engine.serve()
│       └── routes/                               # HTTP route handlers
│           ├── chat_completions.rs               # POST /v1/chat/completions
│           ├── completions.rs                    # POST /v1/completions
│           ├── embeddings.rs                     # POST /v1/embeddings
│           └── classify.rs                       # POST /v1/classify
│
├── prelude-core/                                  # Core: ops, models, engine, scheduler (pure Rust)
│   └── src/
│       ├── lib.rs
│       ├── config.rs                             # EngineConfig — model path, device, scheduler knobs
│       ├── types.rs                              # Shared types (GenerateRequest, GenerateResult, ...)
│       │
│       ├── ops/                                   # Op trait definitions + built-in implementations
│       │   ├── mod.rs                             # register_backend(), select_ops()
│       │   ├── traits/                            # Trait definitions — the shared contract between models and devices
│       │   │   ├── ops.rs                         # `Ops` trait — flat API (ops.rms_norm, ops.varlen_attention, ...)
│       │   │   │                                  #   Fused ops return Option: try device kernel → fallback to composed
│       │   │   ├── attention.rs                   # AttentionOps, VarlenParams, PagedParams, MaskType
│       │   │   ├── norm.rs                        # NormOps (rms_norm, layer_norm)
│       │   │   └── conv.rs                        # ConvOps (conv1d, conv2d)
│       │   ├── cubecl_backend/                    # CubeCL-based TensorOps primitives (CPU fallback path)
│       │   │   ├── mod.rs                         # CubeCLBackend
│       │   │   ├── elementwise.rs                 # Elementwise ops via CubeCL
│       │   │   ├── matmul.rs                      # matmul via CubeCL
│       │   │   └── reduce.rs                      # reduce ops via CubeCL
│       │   └── device_backend/                    # Device backend registration
│       │       └── mod.rs                         # DeviceBackend trait, register_backend()
│       │
│       ├── tensor/                                # Own tensor library (Storage, Layout, DType)
│       │   ├── storage.rs                         # TensorStorage — raw buffer + device location
│       │   ├── layout.rs                          # TensorLayout — strides + offset
│       │   ├── shape.rs                           # Shape
│       │   ├── safetensors.rs                     # SafeTensors file loading
│       │   └── quantized/                         # GGUF quantized tensor support
│       │       ├── gguf_file.rs                   # GGUF file parsing
│       │       └── k_quants.rs                    # K-quant dequantization tables
│       │
│       ├── engine/                                # Engine — public API + execution loops
│       │   ├── engine.rs                          # Engine: owns model, tokenizer, cache, device
│       │   ├── scheduled.rs                       # ScheduledEngine: channel + ar_loop spawn
│       │   ├── executor.rs                        # trait Executor { submit(batch) → Handle, collect(Handle) → Output }
│       │   ├── planner.rs                         # Builds ForwardBatch from scheduler decisions
│       │   ├── weight_loader.rs                   # WeightLoader: safetensors + GGUF → Tensor by name
│       │   ├── tokenizer.rs                       # Tokenizer wrapper
│       │   ├── loading.rs                         # Model loading orchestration (weights + config)
│       │   ├── device.rs                          # Device selection and initialization
│       │   ├── model_runner/                      # Per-paradigm batch construction + output handling
│       │   │   ├── generate.rs                    # AR generation runner
│       │   │   ├── paged_prefill.rs               # Varlen prefill with paged KV
│       │   │   ├── paged_decode.rs                # Paged decode (one token per sequence)
│       │   │   ├── prefill.rs                     # Non-paged prefill (embed/classify)
│       │   │   ├── embed.rs                       # Embedding extraction
│       │   │   └── classify.rs                    # Classification
│       │   ├── run/                               # Scheduling-paradigm loops (device-agnostic)
│       │   │   ├── ar.rs                          # AR LLM: scheduler.step → submit prefill/decode → sample
│       │   │   ├── dllm.rs                        # Diffusion LLM: iterative demasking loop
│       │   │   ├── diffusion.rs                   # Image/video: denoising loop
│       │   │   └── tts.rs                         # TTS: multi-stage pipeline
│       │   ├── speculative/                       # Speculative decoding
│       │   │   ├── mod.rs                         # SpecDecodeRunner: draft → verify → accept loop
│       │   │   ├── proposer.rs                    # trait DraftProposer (EAGLE/DraftModel/Ngram/Medusa)
│       │   │   ├── rejection.rs                   # Rejection sampling
│       │   │   └── tree.rs                        # Tree attention mask construction
│       │   └── sampling/                          # Sampling orchestration
│       │       ├── mod.rs                         # Sampler: penalties → grammar → token IDs
│       │       ├── grammar.rs                     # GrammarManager: async compile + bitmask fill
│       │       └── logits_processor.rs            # LogitsProcessor trait
│       │
│       ├── scheduler/                             # Scheduling decisions (pure CPU, no GPU)
│       │   ├── mod.rs
│       │   ├── state.rs                           # Sequence state machine (Waiting/Prefilling/Decoding/Finished)
│       │   ├── preemption.rs                      # Preemption logic (evict KV blocks, requeue)
│       │   ├── admission.rs                       # Request admission control
│       │   ├── adaptive.rs                        # Adaptive batching + chunked prefill
│       │   ├── dllm.rs                            # DllmScheduler — diffusion LLM demasking
│       │   ├── diffusion.rs                       # DiffusionScheduler — image/video denoising
│       │   ├── oneshot.rs                         # OneShotScheduler — embed, classify, prefill-only
│       │   ├── tts.rs                             # TtsPipelineScheduler — multi-stage TTS
│       │   └── components/                        # Reusable scheduler components
│       │       ├── request_queue.rs               # RequestQueue — FCFS / priority / cache-aware ordering
│       │       └── cache/                         # KV cache management subsystem
│       │           ├── manager.rs                 # CacheManager — top-level cache coordinator
│       │           ├── block_manager.rs           # BlockManager — block alloc/free + ref counting
│       │           ├── prefix_cache.rs            # PrefixKvCache — hash-trie + LRU eviction
│       │           ├── prefix_index.rs            # PrefixMatchIndex — tensor-free LPM algorithm
│       │           ├── kv_buf.rs                  # KvBuf — per-sequence KV buffer (non-paged path)
│       │           └── deltanet_pool.rs           # DeltaNetPool — recurrent state for hybrid models
│       │
│       └── models/                                # Model implementations
│           ├── forward.rs                         # trait ModelForward
│           ├── registry.rs                        # ModelRegistry: inventory-based auto-registration
│           ├── config.rs                          # Model config parsing (from HuggingFace config.json)
│           ├── commons/                           # Shared building blocks (weight containers + utilities)
│           │   ├── linear.rs                      # Linear, RmsNorm, NaiveLinear, QuantFormat registry
│           │   ├── embedding.rs                   # Embedding (vocab → hidden)
│           │   ├── attn_utils.rs                  # RotaryEmbedding, attention dispatch helpers
│           │   └── activation.rs                  # Activation functions
│           ├── qwen3.rs                           # Qwen3 (GQA + QK-norm)
│           ├── qwen3_moe.rs                       # Qwen3-MoE (SparseMoeBlock)
│           ├── qwen3_5.rs                         # Qwen3.5 hybrid (gated attention + DeltaNet)
│           ├── qwen3_next.rs                      # Qwen3-Next
│           ├── gemma3.rs                          # Gemma3 (softcap, sliding window)
│           └── gemma4.rs                          # Gemma4
│
├── prelude-cuda/                                  # CUDA device impl (Ops + Executor)
│   ├── src/
│   │   ├── lib.rs                                 # register() — registers CudaOps + CudaExecutor with priority/probe
│   │   ├── device.rs                              # CUDA runtime: CudaStorage, stream/device registry, PTX loading
│   │   ├── cuda_ops.rs                            # struct CudaOps, impl all op traits
│   │   ├── executor.rs                            # CudaExecutor: GPU queue, CUDA graph capture/replay
│   │   ├── cuda_graph.rs                          # CUDA graph utilities
│   │   ├── quant_backends.rs                      # GPU QuantFormat registration
│   │   ├── attn/                                  # Attention backends
│   │   │   ├── flash_v4.rs                        # FA4 wrappers (SM90+, BF16, best-effort prefill)
│   │   │   └── flashinfer.rs                      # FlashInfer wrappers (SM80+): FA2 decode, FA3 SM90+
│   │   │                                          #   plan() cached per forward pass across all layers
│   │   ├── ops/                                   # Kernel wrapper modules
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
│   ├── flashinfer/                                # build.rs compiles third_party/flashinfer/
│   ├── deepgemm/                                  # build.rs compiles third_party/DeepGEMM/
│   ├── cutlass-gemm/                              # BF16/FP16 GEMM via third_party/cutlass/
│   ├── quant-gemm/                                # GGUF quantized GEMM (llama.cpp MMQ kernels)
│   ├── cula/                                      # cuLA attention via third_party/cuLA/
│   └── tvm-ffi/                                   # TVM FFI runtime (local copy, used by fa4 + flashinfer)
│
└── prelude-cpu/                                   # CPU device impl
    ├── src/
    │   ├── lib.rs                                 # register() — registers CpuOps + CpuExecutor
    │   ├── cpu_ops.rs                             # struct CpuOps, impl all op traits
    │   ├── executor.rs                            # CpuExecutor: block_in_place execution
    │   ├── linear_backends.rs                     # OnednnLinear, quant format registration
    │   ├── attn_cpu.rs                            # CPU attention dispatch
    │   ├── ops/                                   # CPU kernel implementations
    │   │   ├── attention/                         # AVX-512 / DPBF16 optimized attention
    │   │   │   ├── avx512.rs
    │   │   │   └── dpbf16.rs
    │   │   ├── quant/                             # GGUF quantized matmul (Q4_0 through Q6_K, IQ4_NL)
    │   │   │   ├── q4_0.rs, q4_k.rs, q6_k.rs, ...
    │   │   │   └── neon/                          # ARM NEON quant kernels
    │   │   ├── gemm.rs                            # OneDNN GEMM
    │   │   ├── rmsnorm.rs                         # Vectorized RMSNorm
    │   │   ├── rope.rs                            # RoPE (in-place, cos_sin_cache)
    │   │   └── silu_mul.rs                        # Fused SiLU×Mul
    │   └── onednn/                                # OneDNN Rust bindings
    │       ├── ffi.rs
    │       └── ops.rs
    └── onednn-ffi/                                # OneDNN C++ FFI (compiled by build.rs)

third_party/                                       # Vendored third-party source (git submodules, not Cargo crates)
├── flashinfer/                                    # FlashInfer (attention, SM80/SM90)
├── flash-attention/                               # FA4 (attention, TVM AOT)
├── DeepGEMM/                                      # DeepGEMM (BF16/FP8 GEMM, SM90+)
├── cutlass/                                       # CUTLASS (GEMM + conv, header-only)
├── tvm-ffi/                                       # TVM FFI runtime (used by FA4/FlashInfer builds)
├── cuLA/                                          # cuLA (attention library)
├── llama.cpp/                                     # llama.cpp (GGUF quant kernel source)
└── oneDNN/                                        # OneDNN (GEMM + fused ops, CPU)
```

## Layering

<!-- TODO: layering is not very clear -->

```
Engine              starts             Run loop (core: batch or continuous batching)
Run loop            calls              Executor::submit/collect (device crate, via register())
Run loop            calls              Scheduler (core, pure CPU scheduling decisions)
Model call          calls              Model code (prepare tensors → model.forward())
Model code          calls              ops.xxx() (flat API: ops.rms_norm, ops.varlen_attention, ...)
                                         ├── fused path:    try device kernel → fallback to composed
                                         ├── composed path: TensorOps primitives (CubeCL / XLA)
                                         └── device override: CudaOps / RocmOps / CpuOps / ...
Device ops          dispatches to      Kernel libraries (FA4, FlashInfer, DeepGEMM, CUTLASS, CK, ...)
```

**Two trait boundaries** separate core from device crates:

1. **`Ops` trait** (kernel dispatch) — flat API: `ops.exp()`, `ops.rms_norm()`,
   `ops.varlen_attention()`, `ops.fused_add_rmsnorm()`. Models call `ops.xxx()` for everything.
   Internally: `ComposedOps` provides defaults by composing TensorOps primitives.
   `TensorOps` uses `base()` delegation — device backends override only what they need.
   Fused ops try device kernel → auto-fallback to composed. All decision logic is in the `Ops`
   implementation, not in model code. (`OpsBundle` is a conceptual label for this dispatch layer —
   there is no struct or trait by that name; the real type is `&'static dyn Ops`.)

2. **`Model call`** (device execution) — defines HOW batches are submitted to and collected from
   a device. `submit()` is non-blocking (GPU queues work, CPU runs inline). `collect()` awaits
   completion. Core's run loops use submit/collect to naturally get double-buffering on GPU
   (prepare batch N+1 while device runs batch N) and sequential execution on CPU.

<!-- **`Linear` is a parameter carrier, `ops` is the decision maker.**
`Linear` holds weights + optional LoRA state. When fusion is needed (e.g. fused QKV+LoRA projection),
`Linear` passes its weights to ops (`ops.qkv_projection(x, weights, lora_state)`), and
the `Ops` implementation handles fused/fallback/device dispatch. All decision logic lives in the
`Ops` impl, never in `Linear` or model code. This ensures LoRA, quantization, and fusion work
transparently across all devices without model-level branching. -->

## Engine Architecture

The following figures show the key components inside `prelude-core` and how a generation request flows through them.

### Component Map

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ScheduledEngine  (prelude-core/src/engine/mod.rs)                                              │
│  Public handle held by the caller. Spawns ar_loop on creation.                                  │
│  Sends GenerateRequests in via mpsc channel; streams tokens back via response channel.           │
│                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  ar_loop  (prelude-core/src/engine/run/ar.rs)                                           │   │
│  │  Single async task per engine. Runs the continuous-batching loop until all seqs finish.  │   │
│  │                                                                                         │   │
│  │  ┌────────────────────────────────────┐   ┌──────────────────────────────────────────┐ │   │
│  │  │  ArScheduler                       │   │  Engine  (engine/mod.rs)                 │ │   │
│  │  │  prelude-core/src/scheduler/ar.rs  │   │                                          │ │   │
│  │  │                                    │   │  Tokenizer                               │ │   │
│  │  │  Pure CPU — no device knowledge.   │   │  Converts text ↔ token IDs.              │ │   │
│  │  │                                    │   │                                          │ │   │
│  │  │  Waiting queue                     │   │  ModelExecutor                           │ │   │
│  │  │  Running list                      │   │  ├─ model: Box<dyn ModelForward>         │ │   │
│  │  │  Finished list                     │   │  │   models/{qwen3,gemma3,llama,...}.rs  │ │   │
│  │  │                                    │   │  │   Device-agnostic: calls ops.xxx()    │ │   │
│  │  │  schedule_step()                   │   │  │   for all compute, no kernel refs.    │ │   │
│  │  │  → select seqs to prefill/decode   │   │  ├─ device, dtype, config               │ │   │
│  │  │  → enforce three budgets:          │   │  └─ ops: &'static dyn Ops              │ │   │
│  │  │    max_running_requests            │   │      ops/traits/ops.rs                 │ │   │
│  │  │    max_prefill_tokens              │   │      Flat API (ops.rms_norm,            │ │   │
│  │  │    max_total_kv_tokens             │   │      ops.varlen_attention, ...)         │ │   │
│  │  │                                    │   │      Fused ops: try device kernel →     │ │   │
│  │  │  Sequence state machine:           │   │      auto-fallback to composed.         │ │   │
│  │  │  Waiting → Prefilling              │   │                                          │ │   │
│  │  │  Prefilling → Decoding             │   │  CacheManager                            │ │   │
│  │  │  Decoding → Finished               │   │  ├─ PagedKvPool  (KV tensors on device)  │ │   │
│  │  │  Decoding → Waiting                │   │  ├─ BlockManager (alloc/free, ref-count) │ │   │
│  │  │  (preemption: KV blocks evicted)   │   │  │   cache/block_manager.rs             │ │   │
│  │  │                                    │   │  └─ PrefixKvCache (LPM + LRU eviction)  │ │   │
│  │  │  RequestQueue                      │   │      cache/prefix_cache.rs              │ │   │
│  │  │  FCFS / priority / cache-aware     │   │      Matches prompt prefixes to cached  │ │   │
│  │  │  components/request_queue.rs       │   │      KV blocks; reuses them directly.   │ │   │
│  │  └────────────────────────────────────┘   └──────────────────────────────────────────┘ │   │
│  │                                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Executor  (trait: prelude-core/src/engine/executor.rs)                         │   │   │
│  │  │                                                                                 │   │   │
│  │  │  submit(ForwardBatch) → ExecutionHandle   non-blocking; GPU queues work item    │   │   │
│  │  │  handle.await         → ModelOutput       tokio awaits device completion        │   │   │
│  │  │  Enables double-buffering: prepare batch N+1 while device runs batch N.         │   │   │
│  │  │                                                                                 │   │   │
│  │  │  ┌──────────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │   │   │
│  │  │  │  CudaExecutor            │  │  RocmExecutor        │  │  CpuExecutor     │  │   │   │
│  │  │  │  prelude-cuda/src/       │  │  prelude-rocm/src/   │  │  prelude-cpu/    │  │   │   │
│  │  │  │  executor.rs             │  │  executor.rs         │  │  src/executor.rs │  │   │   │
│  │  │  │                          │  │                      │  │                  │  │   │   │
│  │  │  │  Single-threaded GPU     │  │  HIP command queue   │  │  block_in_place  │  │   │   │
│  │  │  │  queue draining          │  │  HIP graph capture   │  │  Sequential,     │  │   │   │
│  │  │  │  GpuPackets.             │  │  and replay.         │  │  no graph.       │  │   │   │
│  │  │  │  CUDA graph capture      │  │                      │  │                  │  │   │   │
│  │  │  │  and replay for decode.  │  │                      │  │                  │  │   │   │
│  │  │  └──────────────────────────┘  └──────────────────────┘  └──────────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Kernel dispatch via `ops.xxx()` (called from model.forward() for every transformer layer):

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  Linear::forward()  →  ops.matmul()  →  CudaOps::matmul()                          │
  │  (models/layers/linear.rs)              (prelude-cuda/src/ops/gemm.rs)              │
  │                                                                                     │
  │  Covers: Q/K/V projections, output projection, MLP gate/up/down, lm_head (~7×/layer)│
  │                                                                                     │
  │  ├─ Try DeepGEMM      prelude-cuda/deepgemm/       SM90+ only, BF16, non-batched   │
  │  └─ Fallback CUTLASS  prelude-cuda/cutlass-gemm/   SM80+, BF16/FP16/F32            │
  └─────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  ops.varlen_attention()  →  CudaOps::varlen_attention()   [prefill: Q > 1]          │
  │  ops.paged_attention()   →  CudaOps::paged_attention()    [decode:  Q = 1]          │
  │  (ops/traits/attention.rs)  (prelude-cuda/src/cuda_ops.rs)                          │
  │                                                                                     │
  │  ├─ Try FA4         prelude-cuda/src/attn/flash_v4.rs     SM80+, BF16 best-effort  │
  │  └─ Fallback        prelude-cuda/src/attn/flashinfer.rs   SM80+ FA2 / SM90+ FA3    │
  │       FlashInfer plan() runs once per forward, cached across all layers.            │
  └─────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  Fused ops (all return Option — None means unsupported, auto-falls back):           │
  │  ops.fused_add_rmsnorm()       →  residual + norm in one kernel                    │
  │  ops.fused_silu_mul()          →  SiLU gate × up-projection in one kernel          │
  │  ops.fused_qknorm_rope()       →  Q/K norm + RoPE in one kernel                   │
  │  ops.fused_knorm_rope_cache_write()  →  K norm + RoPE + KV cache scatter           │
  │  Implementations: prelude-cuda/src/ops/{rmsnorm,rope,kv_cache,elementwise}.rs      │
  └─────────────────────────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
  API call: engine.generate(GenerateRequest)
       │
       │  tokenize + build sampling params
       │
       ▼
  ScheduledEngine ──channel──▶  ar_loop
                                    │
                    ┌───────────────▶│◀──────────────────────────────┐
                    │               │  loop until all done           │
                    │               │                                │
                    │    ┌──────────▼──────────────────────────┐     │
                    │    │  1. Drain new requests from channel  │     │
                    │    │     Scheduler::add_request(seq)      │     │
                    │    └──────────┬──────────────────────────┘     │
                    │               │                                │
                    │    ┌──────────▼──────────────────────────┐     │
                    │    │  2. Scheduler::schedule_step()       │     │
                    │    │     select sequences to prefill/decode│    │
                    │    └──────────┬──────────────────────────┘     │
                    │               │                                │
                    │    ┌──────────▼──────────────────────────┐     │
                    │    │  3. Engine: build ForwardBatch       │     │
                    │    │                                      │     │
                    │    │  Prefill  varlen attention           │     │
                    │    │  • find common prefix (cache reuse) │     │
                    │    │  • allocate KV blocks               │     │
                    │    │  • model.forward(packed_tokens)     │     │
                    │    │  • sample first output token        │     │
                    │    │                                      │     │
                    │    │  Decode   paged attention           │     │
                    │    │  • one token per running sequence   │     │
                    │    │  • model.forward(single tokens)     │     │
                    │    │  • read KV from paged block tables  │     │
                    │    └──────────┬──────────────────────────┘     │
                    │               │                                │
                    │    ┌──────────▼──────────────────────────┐     │
                    │    │  4. Executor::submit(batch)          │     │
                    │    │     → ExecutionHandle (non-blocking) │     │
                    │    │     handle.await → ModelOutput       │     │
                    │    └──────────┬──────────────────────────┘     │
                    │               │                                │
                    │    ┌──────────▼──────────────────────────┐     │
                    │    │  5. Sample + stop-condition check    │     │
                    │    │     LogitsProcessor (temp, top-p/k) │     │
                    │    │     EOS / stop string / max tokens  │     │
                    │    └──────────┬──────────────────────────┘     │
                    │               │                                │
                    │    ┌──────────▼──────────────────────────┐     │
                    │    │  6. Stream token / deliver result    │     │
                    │    │     → StreamEvent::Token             │     │
                    │    │     → GenerateResult on finish       │     │
                    │    └──────────┬──────────────────────────┘     │
                    │               │                                │
                    └───────────────┘────────────────────────────────┘
```

## Dependency Graph

```
prelude-server (binary, standalone HTTP server — zero device-specific code)
    ├── prelude-core              (ops traits, cubecl_backend, models, engine, scheduler)
    │       └── cubecl                (pure Rust: IR + TensorOps primitives, generic over runtime)
    ├── prelude-cuda              (feature-gated — CudaOps + CudaExecutor + kernel sub-crates)
    │       ├── prelude-core
    │       ├── cubecl (features = ["cuda"])   (enables CubeCL CUDA runtime for TensorOps)
    │       └── fa4/, flashinfer/, deepgemm/, cutlass-gemm/, quant-gemm/, cula/, tvm-ffi/
    │           (each sub-crate has a build.rs that compiles from third_party/)
    └── prelude-cpu               (always linked — CpuOps + CpuExecutor, lowest-priority fallback)
            ├── prelude-core
            └── onednn-ffi/           (OneDNN C++ FFI, compiled by build.rs from third_party/oneDNN/)

third_party/ (git submodules, source only — not Cargo crates)
    ├── flashinfer/               (compiled by prelude-cuda/flashinfer/build.rs)
    ├── flash-attention/          (compiled by prelude-cuda/fa4/build.rs)
    ├── DeepGEMM/                 (compiled by prelude-cuda/deepgemm/build.rs)
    ├── cutlass/                  (header-only, used by prelude-cuda/cutlass-gemm/)
    ├── tvm-ffi/                  (TVM FFI runtime, used by FA4/FlashInfer/cuLA builds)
    ├── cuLA/                     (compiled by prelude-cuda/cula/build.rs)
    ├── llama.cpp/                (GGUF quant kernels, compiled by prelude-cuda/quant-gemm/)
    └── oneDNN/                   (compiled by prelude-cpu/onednn-ffi/build.rs)
```

## Key Subsystems

| Subsystem | Location | Description |
|-----------|----------|-------------|
| Engine | `prelude-core/src/engine/engine.rs` | `Engine`: owns model, tokenizer, cache, device |
| ScheduledEngine | `prelude-core/src/engine/scheduled.rs` | Public handle; spawns ar_loop, owns request channel |
| Executor trait | `prelude-core/src/engine/executor.rs` | `submit(batch) → Handle`, `handle.await → Output` |
| AR run loop | `prelude-core/src/engine/run/ar.rs` | Continuous-batching loop: drain → schedule → submit → sample |
| Model runner | `prelude-core/src/engine/model_runner/` | Per-paradigm batch construction (prefill, decode, embed, classify) |
| Schedulers | `prelude-core/src/scheduler/` | State machine, preemption, admission, chunked prefill |
| KV cache | `prelude-core/src/scheduler/components/cache/` | `BlockManager`, `PrefixKvCache`, paged + prefix caching |
| Ops traits | `prelude-core/src/ops/traits/ops.rs` | `Ops` trait — flat API, `AttentionOps`, `NormOps`, `ConvOps` |
| CubeCL backend | `prelude-core/src/ops/cubecl_backend/` | TensorOps primitives via CubeCL (CPU fallback) |
| Model registry | `prelude-core/src/models/registry.rs` | `inventory`-based auto-registration |
| Model commons | `prelude-core/src/models/commons/` | `Linear`, `Embedding`, `RotaryEmbedding`, activations |
| CUDA ops | `prelude-cuda/src/cuda_ops.rs` | `CudaOps`: impl all op traits, dispatches to kernel wrappers |
| GPU executor | `prelude-cuda/src/executor.rs` | Single-threaded GPU queue, CUDA graph capture/replay |
| Attention backends | `prelude-cuda/src/attn/` | FA4 (SM90+), FlashInfer FA2/FA3 (SM80+) |
| GEMM dispatch | `prelude-cuda/src/ops/gemm.rs` | DeepGEMM (SM90+) → CUTLASS (SM80+) fallback chain |

When working on sub-systems:

| Subsystem | Needs to know | Does NOT need to know |
|-----------|--------------|----------------------|
| **Model impl** (Qwen3, Flux, TTS) | Module APIs (`Linear`, `residual_norm`, ...), `ops.xxx()` (`&dyn Ops`) | Any device impl, kernel library, engine |
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

- [Integration](integration.md) — Integrate with external frameworks.

## Common Contribution Paths

- [Add a component](add.md) — Details about how to add model, kernel backend, new devices, and scheduler mechanisms.

## Design Docs

For deeper internals, see the design docs:

- [Architecture Overview](design/overview.md) — request flow, engine hierarchy
- [Scheduler](design/scheduler.md) — continuous batching, KV management
- [Models](design/models.md) — how models are structured and registered
- [Ops and Modules](design/ops.md) — the three-layer ops system
- [Devices](design/devices.md) — device crate structure and backend dispatch
