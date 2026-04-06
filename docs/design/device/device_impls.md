## Device Implementations

Each device crate provides two things:
1. **`OpsBundle`** — overrides **hot-path** op traits with optimized kernels. Non-hot-path ops fall through
   to **ComposedOps** — default implementations that compose TensorOps primitives.
2. **`Executor`** — implements `Executor` trait (submit/collect) with device-specific execution strategy

Both are auto-registered via `ctor` at link time. See [construction.md](construction.md).

### Tiered Ops Architecture

```
Layer 1: TensorOps primitives (unary, binary, reduce, cast, contiguous, matmul, ...)
          └── Provided by: CubeCL (CUDA/ROCm/Vulkan/Metal/CPU) or XLA (TPU)

Layer 2: ComposedOps (prelude-core, pure composition, no device dependency)
          └── Composes TensorOps → NormOps, ActivationOps, ConvOps defaults
          └── e.g., rms_norm = sqr → mean → rsqrt → mul

Layer 3: Device Ops (device crate, optimized overrides)
          └── Override any op at any level: TensorOps, ComposedOps, or FusedOps
```

**What each device crate provides:**

| | TensorOps primitives | Hot-path overrides | ComposedOps |
|--|---------------------|-------------------|------------|
| **CUDA** | CubeCL `<CudaRuntime>` | CUTLASS/DeepGEMM, FlashInfer/FA4, NCCL | inherits from core |
| **ROCm** | CubeCL `<HipRuntime>` | CK GEMM, aiter attention, RCCL | inherits from core |
| **Vulkan** | CubeCL `<WgpuRuntime>` | SPIR-V flash attn, cooperative matmul | inherits from core |
| **Metal** | CubeCL `<WgpuRuntime>` | MSL flash attn, simdgroup matmul | inherits from core |
| **TPU** | XLA (`XLATensorOps`) | Pallas attention, XLA dot_general | inherits from core |
| **CPU** | CubeCL `<CpuRuntime>` | oneDNN GEMM, AVX-512 attention | inherits from core |

**All backends follow the same pattern.** No exceptions. Each:
1. Provides TensorOps primitives (CubeCL or XLA)
2. Overrides hot-path ops (GEMM, Attention, KV cache)
3. Optionally overrides FusedOps, NormOps, ActivationOps, quantized ops, or even individual TensorOps methods
4. Inherits ComposedOps for everything else

ComposedOps provides correct-but-slow defaults for quantized operations too:
- `quantized_matmul`: dequantize weights to BF16/FP16, then standard matmul
- `QuantFormat` (GGUF): dequantize packed weights to float tensor, then standard Linear

Device crates override with native quantized kernels for performance:
- CUDA: DeepGEMM FP8, CUTLASS INT8, tiled MMQ for GGUF (operates on packed format directly)
- ROCm: CK FP8 GEMM
- TPU: XLA INT8/FP8 matmul
- Metal/Vulkan: in-shader dequant + compute
- CPU: AVX vec_dot on packed GGUF blocks

**Adding a new device backend (e.g., ROCm):**
- MUST implement: TensorOps (via `CubeCLTensorOps::<HipRuntime>`), GemmOps, AttentionOps, KvCacheOps, Executor
- SHOULD implement: FusedOps (fused_add_rmsnorm, fused_silu_mul for performance)
- CAN skip: NormOps, ActivationOps, ConvOps — ComposedOps handles them automatically

### CUDA

```rust
// prelude-cuda/src/cuda_ops.rs

struct CudaOps {
    fa4: Option<FA4Registry>,       // vendored, AOT-compiled
    fi: FlashInferRegistry,          // vendored, AOT-compiled
    deepgemm: Option<DeepGemmRegistry>,  // vendored, AOT-compiled
    cutlass: Option<CutlassHandle>,  // vendored, header-only, AOT-compiled
    fi_workspace: FlashInferWorkspace,
}
```

No cuBLAS, no cuDNN. All kernels are vendored and statically linked.
Runtime dependency: NVIDIA driver only (libcuda.so).

Dispatch logic is explicit if-else, not a capability system:

```rust
// prelude-cuda/src/attention.rs

impl AttentionOps for CudaOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        // FA4: best SM90+ prefill
        if let Some(fa4) = &self.fa4 {
            if let Some(func) = fa4.get(&fa4_key(q, k, params)) {
                return fa4_varlen(fa4, func, q, k, v, params);
            }
        }
        // FlashInfer: FA3 on SM90+, FA2 on SM80
        fi_varlen(&self.fi, &self.fi_workspace, q, k, v, params)
    }

    fn paged_attention(&self, q, key_cache, value_cache, params) -> Result<Tensor> {
        if params.max_seqlen_q == 1 {
            // Decode: always FlashInfer (FA4 can't do Q=1)
            return fi_paged_decode(&self.fi, &self.fi_workspace, q, key_cache, value_cache, params);
        }
        // Prefill over paged cache: try FA4, fallback FlashInfer
        if let Some(fa4) = &self.fa4 {
            if let Some(func) = fa4.get(&fa4_paged_key(q, params)) {
                return fa4_paged(fa4, func, q, key_cache, value_cache, params);
            }
        }
        fi_paged_prefill(&self.fi, &self.fi_workspace, q, key_cache, value_cache, params)
    }
}

impl GemmOps for CudaOps {
    fn matmul(&self, a, b) -> Result<Tensor> {
        // DeepGEMM: SM90+ BF16/FP8
        if let Some(dg) = &self.deepgemm {
            if let Ok(out) = dg.try_gemm(a, b) { return Ok(out); }
        }
        // CUTLASS: SM80+ (vendored, no cuBLAS needed)
        self.cutlass.gemm(a, b)
    }
}

// CudaOps overrides all fusions — only lists the ones it supports.
// Unlisted methods inherit the default `{ None }`.
impl FusedOps for CudaOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(gpu::fused_add_rmsnorm(..)) }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { Some(gpu::fused_silu_mul(..)) }
    fn fused_qknorm_rope(&self, ..) -> Option<Result<..>> { Some(gpu::fused_qknorm_rope(..)) }
    fn fused_knorm_rope_cache_write(&self, ..) -> Option<Result<..>> { Some(gpu::fused_knorm_rope_kv_write(..)) }
    fn fused_adaln_zero(&self, ..) -> Option<Result<..>> { Some(gpu::fused_adaln_zero(..)) }
    fn fused_scale_shift(&self, ..) -> Option<Result<..>> { Some(gpu::fused_scale_shift(..)) }
    fn fused_lora_matmul(&self, ..) -> Option<Result<..>> { Some(gpu::bgmv_lora(..)) }
}
```

### ROCm

All kernels vendored and AOT-compiled with hipcc at build time.
Runtime dependency: AMD driver + HIP runtime (libamdhip64.so) only. No hipBLAS, no hipDNN.

Key constraints:
- **Wave size**: CDNA (MI300/MI350) = wave64, RDNA (RX 7000/9000) = wave32.
- **LDS (shared memory)**: MI300 = 64KB, MI350 = 160KB. Kernel tile sizes must vary.
- **FP8 format**: MI300 (gfx942) uses FNUZ, MI350 (gfx950) uses E4M3. Both need support.
- **Flash attention**: aiter kernels (gfx942/gfx950 only), CK flash attn for other CDNA.

```rust
// prelude-rocm/src/

struct RocmOps {
    arch: RocmArch,                       // gfx942, gfx950, gfx1100, ...
    ck_flash: Option<CKFlashAttnHandle>,  // CK flash attention (vendored, CDNA only)
    ck_gemm: CKGemmHandle,               // CK GEMM (vendored, header-only, AOT-compiled)
    aiter: Option<AiterHandle>,           // aiter flash attention (vendored, gfx942/950)
}

/// AMD GPU architecture. Determines available kernels and tuning.
enum RocmArch {
    Gfx942,   // MI300/MI325X (CDNA3, wave64, 64KB LDS, FP8 FNUZ)
    Gfx950,   // MI350 (CDNA4, wave64, 160KB LDS, FP8 E4M3+FNUZ)
    Gfx1100,  // RX 7900 (RDNA3, wave32, WMMA, no MFMA)
    Gfx1200,  // RX 9000 (RDNA4, wave32)
}

impl AttentionOps for RocmOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        // aiter: best on MI300/MI350
        if let Some(aiter) = &self.aiter {
            return aiter.varlen(q, k, v, params);
        }
        // CK flash attention: CDNA fallback (vendored)
        if let Some(ck) = &self.ck_flash {
            return ck.varlen(q, k, v, params);
        }
        bail!("flash attention requires CK or aiter on ROCm")
    }

    fn paged_attention(&self, q, key_cache, value_cache, params) -> Result<Tensor> {
        if let Some(aiter) = &self.aiter {
            return aiter.paged(q, key_cache, value_cache, params);
        }
        if let Some(ck) = &self.ck_flash {
            return ck.paged(q, key_cache, value_cache, params);
        }
        bail!("paged attention requires CK or aiter on ROCm")
    }
}

impl GemmOps for RocmOps {
    fn matmul(&self, a, b) -> Result<Tensor> {
        // CK GEMM: vendored, AOT-compiled (no hipBLAS needed)
        self.ck_gemm.gemm(a, b)
    }
    fn quantized_matmul(&self, a, b, scale_a, scale_b, quant) -> Result<Tensor> {
        match quant {
            QuantScheme::Fp8E4m3 => match self.arch {
                RocmArch::Gfx942 => self.ck_gemm.fp8_fnuz_gemm(a, b, scale_a, scale_b),
                RocmArch::Gfx950 => self.ck_gemm.fp8_e4m3_gemm(a, b, scale_a, scale_b),
                _ => bail!("FP8 GEMM requires MI300+ (gfx942+)"),
            },
            _ => self.ck_gemm.quantized_gemm(a, b, scale_a, scale_b, quant),
        }
    }
}

// Only override what ROCm supports — everything else inherits default `{ None }`.
impl FusedOps for RocmOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(hip::fused_add_rmsnorm(..)) }
}
```

### Metal (Apple Silicon)

Metal is mature for on-device inference (llama.cpp has 67 Metal ops including flash attention).
Key characteristics:
- **Unified memory**: CPU/GPU share address space. No explicit transfers.
- **Simdgroup** (warp equivalent) = 32 threads. Simdgroup matrix multiply on Apple7+.
- **No streams**: single command queue, concurrency via operator reordering + memory barriers.
- **BFloat16**: Metal3+ / Apple6+ (M1 and later). No FP64.
- **Quantization**: excellent — Q4_0/1, Q5_0/1, Q8_0, Q4_K/Q5_K/Q6_K, IQ4_NL, MXFP4 all in MSL.
- **No paged KV cache** in current llama.cpp Metal. Contiguous KV only.
- **No CUDA-like tensor cores**, but simdgroup_multiply_accumulate provides cooperative matmul.

```rust
// prelude-metal/src/

struct MetalOps {
    device: MetalDevice,       // MTLDevice handle
    has_simdgroup_mm: bool,    // Apple7+ — cooperative matmul
    has_bfloat: bool,          // Metal3+ / Apple6+
    max_threadgroup_mem: usize, // typically 32KB
}

impl AttentionOps for MetalOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        // Metal flash attention (MSL compute shader, simdgroup-based tiling)
        metal::flash_attn_varlen(q, k, v, params)
    }
    fn paged_attention(&self, ..) -> Result<Tensor> {
        // Paged KV not yet implemented on Metal.
        // For on-device: use varlen_attention with contiguous KV.
        bail!("paged attention not supported on Metal — use varlen_attention with contiguous KV")
    }
}

impl GemmOps for MetalOps {
    fn matmul(&self, a, b) -> Result<Tensor> {
        if self.has_simdgroup_mm {
            metal::simdgroup_matmul(a, b)  // cooperative matrix path
        } else {
            metal::scalar_matmul(a, b)      // fallback
        }
    }
    fn quantized_matmul(&self, a, b, scale_a, scale_b, quant) -> Result<Tensor> {
        // Metal has excellent quantized matmul — in-shader dequant + compute
        metal::quantized_matmul(a, b, scale_a, scale_b, quant)
    }
}

impl NormOps for MetalOps {
    fn rms_norm(&self, x, weight, eps) -> Result<Tensor> { metal::rms_norm(x, weight, eps) }
    fn layer_norm(&self, x, weight, bias, eps) -> Result<Tensor> { metal::layer_norm(x, weight, bias, eps) }
    fn group_norm(&self, x, weight, bias, groups, eps) -> Result<Tensor> { metal::group_norm(x, weight, bias, groups, eps) }
}

// Metal: override what MSL shaders exist for. Rest inherits default `{ None }`.
impl FusedOps for MetalOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(metal::fused_add_rmsnorm(..)) }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { Some(metal::fused_silu_mul(..)) }
}

impl OpsSession for MetalOps {
    fn begin_forward(&self) {}   // no plan cache needed
    fn end_forward(&self) {}
    fn precompute_paged_plan(&self, ..) -> Result<()> { Ok(()) }
}
```

### Vulkan

Cross-vendor GPU compute (AMD, Intel, Nvidia, Qualcomm, mobile).
llama.cpp has a production-ready Vulkan backend with 151 GLSL compute shaders.

Key characteristics:
- **SPIR-V shaders**: GLSL → SPIR-V at build time. Specialization constants for tile sizes.
- **Subgroup operations**: warp-like, but subgroup size varies (32 Nvidia/AMD, 16 Intel Arc).
- **Cooperative matrix**: optional extension (VK_KHR_cooperative_matrix) — Nvidia only currently.
- **Shared memory**: explicit `shared` arrays in compute shaders (~96KB max per workgroup).
- **Descriptor sets**: buffer bindings per pipeline (overhead vs CUDA kernel args).
- **No paged attention**: confirmed. No vLLM/sglang Vulkan support either.
- **Flash attention**: yes — scalar path + cooperative matrix paths (Nvidia).
- **Quantization**: full coverage (Q4_0 through IQ4_NL, MXFP4).
- **BFloat16**: requires VK_KHR_shader_bfloat16 extension (not universal).

Best for edge/mobile inference with quantized models. Not suited for data center serving
(no paged attention, no async compute guarantees, descriptor binding overhead).

```rust
// prelude-vulkan/src/

struct VulkanOps {
    device: VulkanDevice,
    has_cooperative_matrix: bool,  // VK_KHR_cooperative_matrix (Nvidia)
    has_bfloat16: bool,           // VK_KHR_shader_bfloat16
    subgroup_size: u32,           // 32 (Nvidia/AMD), 16 (Intel Arc), 64 (AMD GCN)
    max_workgroup_shared_mem: u32, // typically 32-96KB
}

impl AttentionOps for VulkanOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        if self.has_cooperative_matrix {
            vk::flash_attn_coopmat(q, k, v, params)
        } else {
            vk::flash_attn_scalar(q, k, v, params)
        }
    }
    fn paged_attention(&self, ..) -> Result<Tensor> {
        bail!("paged attention not supported on Vulkan")
    }
}

impl GemmOps for VulkanOps {
    fn matmul(&self, a, b) -> Result<Tensor> {
        // Dispatch by matrix size: small/medium/large tile strategies
        if self.has_cooperative_matrix {
            vk::matmul_coopmat(a, b)       // tensor-core-like path
        } else {
            vk::matmul_tiled(a, b)          // scalar tiled matmul
        }
    }
    fn quantized_matmul(&self, a, b, scale_a, scale_b, quant) -> Result<Tensor> {
        // In-shader dequant + compute (same approach as Metal)
        vk::quantized_matmul(a, b, scale_a, scale_b, quant)
    }
}

// Vulkan: fuse simple element-wise chains. Rest inherits default `{ None }`.
impl FusedOps for VulkanOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(vk::fused_add_rmsnorm(..)) }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { Some(vk::fused_silu_mul(..)) }
}

impl OpsSession for VulkanOps {
    fn begin_forward(&self) {}   // no plan cache
    fn end_forward(&self) {}
    fn precompute_paged_plan(&self, ..) -> Result<()> { Ok(()) }
}
```

### TPU (via XLA/Pallas)

Fundamentally different execution model from all other devices:
- **No imperative execution**: all ops compiled to XLA HLO IR, then optimized and executed.
- **Static shapes required**: batch and sequence dimensions must be padded to fixed sizes.
- **Paged attention supported**: via JAX Pallas kernels (custom TPU kernels in XLA).
- **Head size alignment**: must be multiple of 128 bytes (MXU constraint).
- **BF16 native**: recommended dtype. FP16 emulated (slower). No FP64.
- **No custom GPU kernels**: everything goes through XLA ops or Pallas.
- **Compilation caching**: expensive first-run compilation, fast replays. OpsSession maps to this.
- **SPMD**: distributed execution via sharding annotations, not explicit all-reduce.

The trait interface works for TPU — the TpuOps implementation internally:
1. Builds XLA computation graphs from op calls.
2. Pads dynamic shapes to static sizes.
3. Caches compiled HLO programs keyed by shape signature.
4. Executes via PJRT (Portable JAX Runtime).

```rust
// prelude-tpu/src/

struct TpuOps {
    pjrt_client: PjrtClient,              // XLA runtime
    pallas_attn: PallasFlashAttn,         // Pallas flash attention kernel
    compiled_cache: CompiledProgramCache,  // HLO → compiled executable cache
    page_size: usize,                      // 16-256, varies by max_model_len
}

impl AttentionOps for TpuOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        // Pallas flash attention — pad heads to 128-byte alignment
        let q = self.pad_head_dim(q)?;
        let k = self.pad_head_dim(k)?;
        let v = self.pad_head_dim(v)?;
        self.pallas_attn.varlen(q, k, v, params)
    }

    fn paged_attention(&self, q, key_cache, value_cache, params) -> Result<Tensor> {
        // TPU supports paged attention via Pallas ragged_paged_attention
        let q = self.pad_head_dim(q)?;
        self.pallas_attn.ragged_paged(q, key_cache, value_cache, params)
    }
}

impl GemmOps for TpuOps {
    fn matmul(&self, a, b) -> Result<Tensor> {
        // XLA dot_general — compiled and optimized for MXU (128x128 systolic array)
        xla::dot_general(a, b)
    }
    fn quantized_matmul(&self, a, b, scale_a, scale_b, quant) -> Result<Tensor> {
        match quant {
            QuantScheme::Int8 => xla::int8_matmul(a, b, scale_a, scale_b),
            QuantScheme::Fp8E4m3 => xla::fp8_matmul(a, b, scale_a, scale_b),
            _ => bail!("W4A16/W4A4 not natively supported on TPU — use INT8 or FP8"),
        }
    }
}

// TPU: all default `{ None }`. XLA auto-fuses the fallback path during HLO compilation.
impl FusedOps for TpuOps {}

impl OpsSession for TpuOps {
    fn begin_forward(&self) {
        // Mark start of XLA tracing scope.
        // Ops called between begin/end are traced into a single HLO program.
        self.compiled_cache.begin_trace();
    }
    fn end_forward(&self) {
        // Compile traced program (or retrieve from cache), execute on TPU.
        self.compiled_cache.end_trace_and_execute();
    }
    fn precompute_paged_plan(&self, block_tables, cu_seqlens_k, block_size) -> Result<()> {
        // Pallas KV cache update — write scheduling metadata.
        // Uses TPU VMEM (software-managed scratch) for DMA-efficient page table access.
        self.pallas_attn.precompute_plan(block_tables, cu_seqlens_k, block_size)
    }
}
```

**Why FusedOps returns `None` on TPU:** XLA's HLO optimizer automatically fuses
element-wise op sequences (add + norm, silu * mul, etc.) during compilation.
The model's explicit fallback path (calling separate NormOps + ActivationOps)
produces the same fused kernel after XLA compilation. No manual fusion needed.

### CPU

```rust
// prelude-cpu/src/

struct CpuOps;

impl AttentionOps for CpuOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        cpu::varlen_attention(q, k, v, params)  // matmul-based SDPA
    }
    fn paged_attention(&self, ..) -> Result<Tensor> {
        bail!("paged attention not supported on CPU")
    }
}

// CPU: only fused_add_rmsnorm (vectorized). Rest inherits default `{ None }`.
impl FusedOps for CpuOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(cpu::fused_add_rmsnorm(..)) }
}

impl OpsSession for CpuOps {
    fn begin_forward(&self) {}   // no-op
    fn end_forward(&self) {}     // no-op
    fn precompute_paged_plan(&self, ..) -> Result<()> { Ok(()) }
}
```
