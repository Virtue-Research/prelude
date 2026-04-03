## Summary

| Concern | Solution |
|---------|----------|
| FA4 can't decode | `CudaOps::paged_attention` routes Q=1 to FlashInfer |
| Fusion control | `FusedOps` trait + modules encapsulate fallback logic |
| Kernel → multi-model reach | Modules: one optimization update → all models using that block benefit |
| Multi-device | Traits implemented per device, model code unchanged |
| Multi-model | Same `AttentionOps` for causal/bidirectional/cross-attention via `MaskType` + `VarlenParams` |
| DeltaNet/Mamba | Not `AttentionOps`; model-owned, closure-injected into TransformerBlock |
| FlashInfer plan cache | `OpsSession::begin_forward()` / `end_forward()` |
| CUDA graphs | `CudaOps::precompute_paged_plan_graphed` (device-specific, not in shared trait) |
| KV cache write timing | `KvCacheOps::reshape_and_cache` separate from `AttentionOps` |
| MLA head_dim asymmetry | Derived from tensor shapes, not params |
| Chunked prefill | `paged_attention` with `max_seqlen_q > 1`, varlen kernel handles mixed Q lengths |
| AdaLN (diffusion) | `FusedOps::fused_adaln_zero` / `fused_scale_shift`, `None` fallback to separate ops |
| Quantized inference | `GemmOps::quantized_matmul` with `QuantScheme` dispatch |
| Metal (Apple) | `MetalOps`: flash attn + quantized matmul via MSL; no paged attention |
| Vulkan (cross-vendor) | `VulkanOps`: flash attn + quantized matmul via SPIR-V; edge/mobile focus |
| TPU (XLA) | `TpuOps`: static shapes, Pallas attention, XLA auto-fuses element-wise chains |
| ROCm arch variation | `RocmArch` enum (gfx942/950/1100), FP8 format auto-selected per arch |
| Tensor parallelism | `CommOps` trait (all_reduce, all_gather), attention ops are TP-agnostic |
| Expert parallelism | `CommOps::all_to_all` for dispatch/combine, `GemmOps::grouped_gemm` for local compute |
| Attention-FFN disaggregation | `modules::moe_layer` with `MoeMode::Disaggregated`, `CommOps::send/recv` for hidden state transfer |
| Sequence parallelism | `CommOps::reduce_scatter` + `all_gather` around local attention |
| Multi-LoRA serving | `FusedOps::fused_lora_matmul` (BGMV/Punica), fallback to per-adapter matmul |
| Speculative decoding | Engine-level; tree attention via `MaskType::Custom(Tensor)` |
| KV cache quantization | `cache_slot_spec` for layout query, encode/decode device-internal in `reshape_and_cache` / `paged_attention` |
