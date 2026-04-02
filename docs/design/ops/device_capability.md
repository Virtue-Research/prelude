## Device Capability Matrix

What each device supports today. "Planned" = not yet implemented but feasible.

| Capability | CUDA | ROCm | Metal | Vulkan | TPU | CPU |
|------------|------|------|-------|--------|-----|-----|
| **varlen_attention** | FA4 / FlashInfer | CK / aiter | MSL flash attn | GLSL flash attn | Pallas | matmul SDPA |
| **paged_attention** | FlashInfer | CK / aiter | — | — | Pallas ragged | — |
| **matmul** | DeepGEMM / CUTLASS | CK GEMM | simdgroup mm | tiled / coopmat | XLA dot_general | BLAS |
| **quantized_matmul** | DeepGEMM FP8, CUTLASS INT8 | CK FP8 | in-shader dequant (Q4-Q8, IQ) | in-shader dequant (Q4-Q8, IQ) | XLA INT8/FP8 | dequant + BLAS |
| **rms_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **layer_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **group_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **conv1d / conv2d / conv_transpose1d** | CUTLASS conv / custom | CK conv / custom | MSL shader | GLSL shader | XLA conv | fallback |
| **fused_add_rmsnorm** | FlashInfer kernel | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **fused_adaln_zero** | Triton/CUDA kernel | — (planned) | — (planned) | — | XLA auto-fuse | — |
| **fused_qknorm_rope** | FlashInfer kernel | — | — | — | — | — |
| **fused_lora_matmul** | BGMV/Punica kernel | — (planned) | — | — | XLA custom op | — |
| **CommOps** | NCCL / custom AR / StepMesh | RCCL | — (single device) | — (single device) | XLA collective | — (single device) |
| **send/recv (AFD)** | NCCL P2P / StepMesh / RDMA | RCCL P2P | — | — | — | — |
| **OpsSession** | FlashInfer plan cache | no-op | no-op | no-op | XLA compile cache | no-op |
| **CUDA graphs** | yes | HIP graphs (6.1+) | — | — | — | — |
| **BFloat16** | SM80+ | all CDNA | Apple6+/Metal3+ | extension req'd | native | optional |
| **FP8** | SM89+ | gfx942 (FNUZ), gfx950 (E4M3) | — | — | v5e+ | — |
| **KV cache quant** | TurboQuant (device-internal) | — (planned) | — | — | — | — |

**Key insight:** The trait interface is the same across all devices. The difference is which
methods return real results vs errors, and which `FusedOps` return `Some` vs `None`.
Model code never changes — the dispatch layer absorbs all device differences.
