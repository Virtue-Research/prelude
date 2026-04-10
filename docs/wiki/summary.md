## Summary

| Concern | Solution |
|---------|----------|
| FA4 can't decode | `CudaOps::paged_attention` routes Q=1 to FlashInfer |
| Fusion control | `FusedOps` trait + OpsBundle flat API encapsulates fallback logic (model code never sees Option) |
| Kernel → multi-model reach | OpsBundle: one fused kernel update → all models calling `ops.xxx()` benefit |
| Multi-device | Traits implemented per device, model code unchanged |
| Multi-model | Same `AttentionOps` for causal/bidirectional/cross-attention via `MaskType` + `VarlenParams` |
| DeltaNet/Mamba | Not `AttentionOps`; model-owned, implemented in each model's DecoderLayer |
| FlashInfer plan cache | `OpsSession::begin_forward()` / `end_forward()` |
| CUDA graphs | `CudaOps::precompute_paged_plan_graphed` (device-specific, not in shared trait) |
| KV cache write timing | `KvCacheOps::reshape_and_cache` separate from `AttentionOps` |
| KV cache sharing (YOCO) | `ModelForward::kv_cache_sharing()` → `CacheManager` aliases tensors. Shared layers skip `reshape_and_cache`. Prefill: model forward stores/passes K/V between layers. Decode: cache aliasing handles it. No new kernel/primitive needed — pure model-level routing. |
| MLA head_dim asymmetry | Derived from tensor shapes, not params |
| Chunked prefill | `paged_attention` with `max_seqlen_q > 1`, varlen kernel handles mixed Q lengths |
| AdaLN (diffusion) | `FusedOps::fused_adaln_zero` / `fused_scale_shift`, `None` fallback to separate ops |
| Quantized inference | `GemmOps::quantized_matmul` with `QuantScheme` dispatch; ComposedOps fallback: dequant → matmul |
| Metal (Apple) | `MetalOps`: flash attn + quantized matmul via MSL; no paged attention |
| Vulkan (cross-vendor) | `VulkanOps`: flash attn + quantized matmul via SPIR-V; edge/mobile focus |
| TPU (XLA) | `TpuOps`: static shapes, Pallas attention, XLA auto-fuses element-wise chains |
| ROCm arch variation | `RocmArch` enum (gfx942/950/1100), FP8 format auto-selected per arch |
| Tensor parallelism | `CommOps` trait (all_reduce, all_gather), attention ops are TP-agnostic |
| Expert parallelism | `CommOps::all_to_all` for dispatch/combine, `GemmOps::grouped_gemm` for local compute |
| Attention-FFN disaggregation | MoE layer with `MoeMode::Disaggregated`, `CommOps::send/recv` for hidden state transfer |
| KV transfer (standalone) | `plugins/prelude-mooncake` wraps Mooncake Transfer Engine (RDMA, NVLink, TCP, topology-aware) |
| KV transfer (Dynamo) | Dynamo owns transfer via NIXL/Mooncake; Prelude exposes block memory via `get_block_memory_info()` |
| Dynamo backend | `prelude-dynamo` binary implements `AsyncEngine` trait, wraps `Engine` — native Rust, no Python |
| Sequence parallelism | `CommOps::reduce_scatter` + `all_gather` around local attention |
| Multi-LoRA serving | `FusedOps::fused_lora_matmul` (BGMV/Punica), fallback to per-adapter matmul |
| LoRA + fusion | `Linear` is parameter carrier; `ops.qkv_projection(x, weights, lora_state)` handles fused QKV+LoRA kernel → fallback to separate matmul+LoRA. All decision logic in OpsBundle, Linear just passes weights. |
| Basic tensor ops | candle-core (CUDA + CPU backends). Matmul routes through registered GEMM dispatch (CUTLASS/DeepGEMM) |
| ComposedOps | Composes candle tensor ops → NormOps, ActivationOps, AttentionOps defaults. All backends inherit |
| OpsBundle flat API | `ops.exp()`, `ops.rms_norm()`, `ops.varlen_attention()`, `ops.fused_add_rmsnorm()` — single layer, no nesting |
| Fused fallback | `ops.fused_add_rmsnorm()` tries device kernel → auto-fallback to composed add + rms_norm |
| Candle tensor backend | candle-core handles basic tensor ops (matmul, cast, element-wise). Device crates register GEMM dispatch and override fused ops |
| Hot-path overrides | Device crates override: GEMM (CUTLASS/CK), Attention (FlashInfer/aiter/Pallas), KV cache |
| Constrained decoding | llguidance (pure Rust, Earley parser, ~50μs/token, MIT license) |
| RL: GPU precision | Batch invariance + FP32 logprob + TIS algorithmic correction |
| RL: TPU true on-policy | XLA determinism — same ops + same sharding → bit-wise zero logprob diff |
| RL: weight hot-update | `/update_weights_from_tensor`, `/flush_cache`, `/pause_generation` API |
| Speculative decoding | Engine-level; tree attention via `MaskType::Custom(Tensor)` |
| KV cache quantization | `cache_slot_spec` for layout query, encode/decode device-internal in `reshape_and_cache` / `paged_attention` |
