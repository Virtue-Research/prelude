# Devices

Each device crate provides two things: `Ops` (kernel dispatch) and `Executor` (execution strategy). Both register at startup via `register()` with a priority and probe function — `prelude-core` has zero device knowledge and no device-specific types.

See [Ops and Modules](ops.md) for the trait definitions that device crates implement.

## Runtime Dependencies

The compiled binary runs with **only the GPU driver installed**. No SDK, no toolkit, no Python packages. Everything is vendored and AOT-compiled into the binary at build time.

| Device | Runtime dependency | Build dependency (not needed at runtime) |
|--------|-------------------|----------------------------------------|
| **CUDA** | NVIDIA driver (`libcuda.so`) | CUDA toolkit (`nvcc`) for AOT compilation |
| **ROCm** | AMD driver + HIP runtime (`libamdhip64.so`) | ROCm SDK (`hipcc`) for AOT compilation |
| **Metal** | macOS (`Metal.framework` built-in) | Xcode command line tools |
| **Vulkan** | GPU driver (Vulkan ICD) | Vulkan SDK (`glslc`) for SPIR-V compilation |
| **TPU** | PJRT runtime (`libpjrt_tpu.so`), dlopen'd at runtime | JAX/XLA build tools |
| **CPU** | None | C compiler |

**No cuBLAS, no cuDNN, no hipBLAS, no hipDNN.** These are SDK components, not driver components. All kernels are vendored and AOT-compiled:

- GEMM: DeepGEMM (vendored) → CUTLASS (vendored, header-only)
- Attention: FA4 (vendored, TVM AOT) → FlashInfer (vendored, AOT)
- ROCm: CK (header-only, vendored) for GEMM + attention; aiter (vendored, AOT) for flash attention

NCCL and RCCL are **dlopen'd at runtime** — not statically linked — to avoid symbol conflicts when both CUDA and ROCm backends are compiled into the same binary.

**TPU exception:** TPU has no public low-level API. All kernel generation happens inside XLA JIT. The binary dlopen's `libpjrt_tpu.so` at runtime. GCP TPU VMs have PJRT pre-installed — no Python, JAX, or SDK needed.

**Two binaries cover all platforms.** Device features are additive — multiple backends compile into a single binary. Runtime auto-detection selects the best available backend.

```bash
# Linux: all GPU backends in one binary
cargo build --features cuda,rocm,vulkan    # → prelude-linux-x86_64

# macOS: Metal + CPU (CUDA/ROCm not available on macOS)
cargo build --features metal               # → prelude-darwin-aarch64
```

## File Structure

Every device crate follows the same layout. `prelude-cuda` is the most complete example:

```
prelude-cuda/
├── src/
│   ├── lib.rs                  # register() — registers CudaOps + CudaExecutor with priority/probe
│   ├── device.rs               # CUDA runtime: storage, stream/device registry, PTX loading
│   ├── cuda_ops.rs             # struct CudaOps, impl Ops trait
│   ├── executor.rs             # CudaExecutor: GPU queue, CUDA graph capture/replay
│   ├── tensor_ops_kernels.rs   # kernel launcher wrappers for tensor primitives
│   ├── cuda_graph.rs           # CUDA graph capture/replay helpers
│   ├── quant_backends.rs       # quantized matmul backend selection
│   ├── moe_ffi.rs              # MoE FFI integration
│   ├── ops/                    # Kernel wrapper modules (one file per op category)
│   │   ├── gemm.rs             # DeepGEMM → CUTLASS fallback dispatch + register_gpu_gemm
│   │   ├── kv_cache.rs         # fused_knorm_rope_kv_cache_write, kv_append, scatter
│   │   ├── rmsnorm.rs          # fused_add_rmsnorm
│   │   ├── elementwise.rs      # fused_silu_mul, add, etc.
│   │   ├── rope.rs             # fused_qknorm_rope
│   │   ├── quant.rs            # quantized op wrappers
│   │   ├── moe.rs              # MoE routing + gateup/down kernels
│   │   ├── tiled_mmq.rs        # tiled quantized matmul
│   │   └── mod.rs
│   ├── attn/                   # Attention backends
│   │   ├── flash_v4.rs         # FA4 dispatch (SM90+)
│   │   ├── flashinfer.rs       # FlashInfer dispatch (SM80+, paged)
│   │   └── mod.rs
│   └── kernels/
│       └── kernels_src/        # .cu source files, organised by category
│           ├── elementwise/    # add.cu, silu_mul.cu
│           ├── normalization/  # rmsnorm.cu, add_rmsnorm.cu
│           ├── kvcache/        # append.cu, scatter_kv_cache.cu, knorm_rope_kv_write.cu
│           ├── rope/           # qknorm_rope.cu
│           ├── moe/            # routing.cu, gateup.cu, down.cu
│           ├── candle/         # ported candle kernels (unary, binary, reduce, cast, …)
│           └── common/         # shared headers (common.cuh, vec_utils.cuh)
├── fa4/                    # build.rs compiles third_party/flash-attention/
├── flashinfer/             # build.rs compiles third_party/flashinfer/
├── deepgemm/               # build.rs compiles third_party/DeepGEMM/
├── cutlass-gemm/           # BF16/FP16 GEMM via third_party/cutlass/
├── quant-gemm/             # GGUF quantized GEMM (llama.cpp MMQ kernels)
├── cula/                   # cuLA attention via third_party/cuLA/
├── nccl/                   # links third_party/nccl/ (dlopen'd at runtime)
└── uccl-ep/                # compiles third_party/uccl/ep/
```

The minimal shape every device crate shares:

```
prelude-{device}/
├── src/
│   ├── lib.rs              # register() — required entry point
│   ├── {device}_ops.rs     # struct {Device}Ops, impl Ops for {Device}Ops
│   └── executor.rs         # struct {Device}Executor, impl Executor trait
└── {kernel-lib}/           # one sub-crate per vendored kernel library (has its own build.rs)
```

`lib.rs` is the only required file — it calls `register()`. The single `Ops` trait provides default composed implementations for everything not overridden. Each kernel sub-crate (e.g., `fa4/`, `ck/`) has its own `build.rs` that compiles from `third_party/`.

## Tiered Ops Architecture

There is a single `Ops` trait in `prelude-core`. All methods have defaults, so a device crate only overrides what it has optimised kernels for:

```
Layer 1: Tensor primitives (unary, binary, reduce, cast, matmul, ...)
          └── Default: delegates to `default_impl()` (CubeCL or bare-ops fallback)
          └── Override: device crate replaces with its own CUDA/HIP/MSL kernel

Layer 2: Composed ops (default impls in the trait, pure composition)
          └── e.g., rms_norm default = sqr → mean → rsqrt → mul
          └── Override: device crate replaces with a fused kernel

Layer 3: Fused ops (default returns `None` — composed path used automatically)
          └── Override: device crate returns `Some(...)` when a fused kernel exists
```

| | Tensor primitives | Hot-path overrides | Composed defaults |
|--|------------------|-------------------|-------------------|
| **CUDA** | CubeCL `<CudaRuntime>` | CUTLASS/DeepGEMM, FlashInfer/FA4, NCCL | inherits from core |
| **ROCm** | CubeCL `<HipRuntime>` | CK GEMM, aiter attention, RCCL | inherits from core |
| **Vulkan** | CubeCL `<WgpuRuntime>` | SPIR-V flash attn, cooperative matmul | inherits from core |
| **Metal** | CubeCL `<WgpuRuntime>` | MSL flash attn, simdgroup matmul | inherits from core |
| **TPU** | XLA (`XLATensorOps`) | Pallas attention, XLA dot_general | inherits from core |
| **CPU** | CubeCL `<CpuRuntime>` | oneDNN GEMM, AVX-512 attention | inherits from core |

All backends follow the same pattern — no exceptions:
1. Implement `impl Ops for {Device}Ops`. Override tensor primitives (via CubeCL or XLA).
2. Override hot-path methods (matmul, varlen_attention, paged_attention, KV cache).
3. Optionally override fused methods (`fused_add_rmsnorm`, `fused_silu_mul`, etc.) — unlisted fused methods auto-return `None` and fall through to composed.
4. Everything else inherits default composed implementations from `trait Ops`.

## Backend Registration

Device crates register at startup via explicit `register()` calls. Core's registry uses **priority + probe** for automatic backend selection.

```rust
// prelude-cuda/src/lib.rs

fn cuda_probe() -> bool {
    device::cuda_device(0).is_ok()   // hardware detection lives in the device crate
}

pub fn register() {
    prelude_core::ops::register_backend(prelude_core::ops::OpsBackend {
        name: "cuda",
        priority: 100,               // GPU backends: high priority
        probe: cuda_probe,
        supports: |d| d.is_cuda(),
        create_ops: cuda_ops::cuda_ops,
    });
    prelude_core::engine::executor::register_executor(
        prelude_core::engine::executor::ExecutorBackend {
            name: "cuda",
            priority: 100,
            probe: cuda_probe,
            supports: |d| d.is_cuda(),
            create: |engine| Box::new(executor::CudaExecutor::new(engine)),
        },
    );
}
```

```rust
// prelude-cpu/src/lib.rs

pub fn register() {
    prelude_core::ops::register_backend(prelude_core::ops::OpsBackend {
        name: "cpu",
        priority: 10,                // lowest — always the fallback
        probe: || true,              // CPU is always available
        supports: |d| d.is_cpu(),
        create_ops: cpu_ops::cpu_ops,
    });
    prelude_core::engine::executor::register_executor(
        prelude_core::engine::executor::ExecutorBackend {
            name: "cpu",
            priority: 10,
            probe: || true,
            supports: |d| d.is_cpu(),
            create: |engine| Box::new(executor::CpuExecutor::new(engine)),
        },
    );
}
```

ROCm uses `priority: 100`, Metal uses `priority: 100`, Vulkan uses `priority: 50`.

`select_ops()` in core filters backends by `supports(device)`, then by `probe()`, and picks the highest priority. **`prelude-core` has no `DeviceType` enum and no hardware detection code** — that logic lives entirely in the device crates.

CUDA and ROCm both register at priority 100. On a machine with an NVIDIA GPU, the CUDA `probe()` returns true and ROCm's returns false. The reverse on AMD hardware. Neither → falls back to CPU (priority 10, always true).

The server registers backends explicitly at startup:

```rust
// prelude-server/src/main.rs

fn main() {
    prelude_cpu::register();
    #[cfg(feature = "cuda")]
    prelude_cuda::register();

    let engine = Engine::new(&EngineConfig::from_env());
    engine.serve();
}
```

## Per-Device Implementation Details

### CUDA

Runtime dependency: NVIDIA driver only (`libcuda.so`). No cuBLAS, no cuDNN.

`CudaOps` is a unit struct. GEMM dispatch tables are registered once at init via `register_gpu_gemm()` rather than stored as fields.

```rust
pub struct CudaOps;
```

Dispatch is explicit if-else, not a capability system. All methods are part of the single `impl Ops for CudaOps`:

```rust
impl Ops for CudaOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        // FA4: best SM90+ prefill (flash_v4.rs)
        if let Some(func) = fa4_lookup(q, k, params) {
            return fa4_varlen(func, q, k, v, params);
        }
        // FlashInfer: FA3 on SM90+, FA2 on SM80 (flashinfer.rs)
        fi_varlen(q, k, v, params)
    }

    fn paged_attention(&self, q, key_cache, value_cache, params) -> Result<Tensor> {
        if params.max_seqlen_q == 1 {
            // Decode: always FlashInfer (FA4 can't do Q=1)
            return fi_paged_decode(q, key_cache, value_cache, params);
        }
        // Prefill over paged cache: try FA4, fallback FlashInfer
        if let Some(func) = fa4_paged_lookup(q, params) {
            return fa4_paged(func, q, key_cache, value_cache, params);
        }
        fi_paged_prefill(q, key_cache, value_cache, params)
    }

    fn matmul(&self, a, b) -> Result<Tensor> {
        // DeepGEMM (SM90+) → CUTLASS fallback (ops/gemm.rs)
        gpu_gemm_dispatch(a, b)
    }

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

Runtime dependency: AMD driver + HIP runtime (`libamdhip64.so`) only. No hipBLAS, no hipDNN. All kernels vendored and AOT-compiled with `hipcc` at build time.

Key hardware constraints:
- **Wave size**: CDNA (MI300/MI350) = wave64, RDNA (RX 7000/9000) = wave32.
- **LDS (shared memory)**: MI300 = 64KB, MI350 = 160KB. Kernel tile sizes must vary.
- **FP8 format**: MI300 (gfx942) uses FNUZ, MI350 (gfx950) uses E4M3. Both need support.
- **Flash attention**: aiter kernels (gfx942/gfx950 only), CK flash attn for other CDNA.

```rust
struct RocmOps {
    arch: RocmArch,
    ck_flash: Option<CKFlashAttnHandle>,  // CK flash attention (vendored, CDNA only)
    ck_gemm: CKGemmHandle,               // CK GEMM (vendored, header-only, AOT-compiled)
    aiter: Option<AiterHandle>,           // aiter flash attention (vendored, gfx942/950)
}

enum RocmArch {
    Gfx942,   // MI300/MI325X (CDNA3, wave64, 64KB LDS, FP8 FNUZ)
    Gfx950,   // MI350 (CDNA4, wave64, 160KB LDS, FP8 E4M3+FNUZ)
    Gfx1100,  // RX 7900 (RDNA3, wave32, WMMA, no MFMA)
    Gfx1200,  // RX 9000 (RDNA4, wave32)
}

// Only override what ROCm supports — everything else inherits defaults from `trait Ops`.
impl Ops for RocmOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        if let Some(aiter) = &self.aiter { return aiter.varlen(q, k, v, params); }
        if let Some(ck) = &self.ck_flash { return ck.varlen(q, k, v, params); }
        bail!("flash attention requires CK or aiter on ROCm")
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

    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(hip::fused_add_rmsnorm(..)) }
}
```

### Metal (Apple Silicon)

Key characteristics:
- **Unified memory**: CPU/GPU share address space — no explicit transfers.
- **Simdgroup** (warp equivalent) = 32 threads. Simdgroup matrix multiply on Apple7+.
- **No paged KV cache**: contiguous KV only. `paged_attention` returns an error.
- **BFloat16**: Metal3+ / Apple6+ (M1 and later).
- **Quantization**: excellent — Q4_0/1, Q5_0/1, Q8_0, Q4_K/Q5_K/Q6_K, IQ4_NL, MXFP4 all in MSL.

```rust
struct MetalOps {
    device: MetalDevice,
    has_simdgroup_mm: bool,     // Apple7+ — cooperative matmul
    has_bfloat: bool,           // Metal3+ / Apple6+
    max_threadgroup_mem: usize, // typically 32KB
}

impl Ops for MetalOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        metal::flash_attn_varlen(q, k, v, params)
    }
    fn paged_attention(&self, ..) -> Result<Tensor> {
        bail!("paged attention not supported on Metal — use varlen_attention with contiguous KV")
    }

    fn matmul(&self, a, b) -> Result<Tensor> {
        if self.has_simdgroup_mm { metal::simdgroup_matmul(a, b) }
        else { metal::scalar_matmul(a, b) }
    }

    // MSL shaders exist for these norms — override the composed defaults.
    fn rms_norm(&self, x, weight, eps) -> Result<Tensor> { metal::rms_norm(x, weight, eps) }
    fn layer_norm(&self, x, weight, bias, eps) -> Result<Tensor> { metal::layer_norm(x, weight, bias, eps) }
}
```

Best for on-device inference on Apple Silicon. Not suited for data center serving (no paged attention).

### Vulkan

Cross-vendor GPU compute (AMD, Intel, Nvidia, Qualcomm, mobile).

Key characteristics:
- **SPIR-V shaders**: GLSL → SPIR-V at build time. Specialization constants for tile sizes.
- **Subgroup size** varies: 32 (Nvidia/AMD), 16 (Intel Arc), 64 (AMD GCN).
- **Cooperative matrix**: optional extension (`VK_KHR_cooperative_matrix`) — Nvidia only currently.
- **No paged attention**: not supported. `paged_attention` returns an error.
- **BFloat16**: requires `VK_KHR_shader_bfloat16` (not universal).

```rust
struct VulkanOps {
    device: VulkanDevice,
    has_cooperative_matrix: bool,   // VK_KHR_cooperative_matrix (Nvidia)
    has_bfloat16: bool,
    subgroup_size: u32,
    max_workgroup_shared_mem: u32,
}

impl Ops for VulkanOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        if self.has_cooperative_matrix { vk::flash_attn_coopmat(q, k, v, params) }
        else { vk::flash_attn_scalar(q, k, v, params) }
    }
    fn paged_attention(&self, ..) -> Result<Tensor> {
        bail!("paged attention not supported on Vulkan")
    }
}
```

Best for edge/mobile inference with quantized models. Not suited for data center serving (no paged attention, descriptor binding overhead).

### TPU (via XLA/Pallas)

Fundamentally different execution model from all other devices:
- **No imperative execution**: all ops compiled to XLA HLO IR, then optimized and executed.
- **Static shapes required**: batch and sequence dimensions must be padded to fixed sizes.
- **Paged attention supported**: via JAX Pallas kernels (`ragged_paged_attention`).
- **Head size alignment**: must be a multiple of 128 bytes (MXU constraint).
- **BF16 native**: recommended dtype. FP16 emulated. No FP64.
- **SPMD**: distributed execution via sharding annotations, not explicit all-reduce.

The trait interface works for TPU: `TpuOps` internally builds XLA computation graphs from op calls, pads dynamic shapes to static sizes, caches compiled HLO programs keyed by shape signature, and executes via PJRT.

```rust
struct TpuOps {
    pjrt_client: PjrtClient,
    pallas_attn: PallasFlashAttn,
    compiled_cache: CompiledProgramCache,
    page_size: usize,
}

impl Ops for TpuOps {
    // Fused ops: all inherit the default `{ None }`.
    // XLA auto-fuses the composed fallback path during HLO compilation — no manual fusion needed.

    fn begin_forward(&self) {
        // Mark start of XLA tracing scope. Ops called between begin/end are
        // traced into a single HLO program.
        self.compiled_cache.begin_trace();
    }
    fn end_forward(&self) {
        // Compile traced program (or retrieve from cache), execute on TPU.
        self.compiled_cache.end_trace_and_execute();
    }
}
```

Fused op methods return `None` on TPU because XLA's HLO optimizer automatically fuses element-wise op sequences (add + norm, silu * mul, etc.) during compilation. The model's explicit fallback path produces the same fused kernel after XLA compilation — no manual fusion needed.

### CPU

```rust
pub struct CpuOps;

impl Ops for CpuOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        cpu::varlen_attention(q, k, v, params)  // matmul-based SDPA
    }
    fn paged_attention(&self, ..) -> Result<Tensor> {
        bail!("paged attention not supported on CPU")
    }

    // CPU: fused_add_rmsnorm and fused_silu_mul are vectorized. Rest inherits default `{ None }`.
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(cpu::fused_add_rmsnorm(..)) }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { Some(cpu::fused_silu_mul(..)) }
}
```

## Non-Softmax Token Mixers (Hybrid Models)

DeltaNet, Mamba, RWKV, and RetNet use recurrent state — not KV cache. Their forward signature is fundamentally different from softmax attention:

```
Softmax attention: (Q, K, V, mask) → O
DeltaNet:          (x, conv_state, recurrent_state) → (o, conv_state', recurrent_state')
Mamba:             (x, conv_state, ssm_state) → (o, conv_state', ssm_state')
```

These do **not** go through `Ops::varlen_attention`. Models own their token mixer implementations. `TransformerBlock` handles this via closure injection — it is agnostic to the token mixer type:

```rust
// TransformerBlock is agnostic to token mixer type
fn forward<A, M>(&self, x: &Tensor, attn_fn: A, mlp_fn: M) -> Result<Tensor>
where A: FnOnce(&Tensor) -> Result<Tensor>
{
    let h = ops.rms_norm(x, &self.ln1_weight, eps)?;
    let h = attn_fn(&h)?;  // softmax attention OR DeltaNet OR Mamba
    // ...
}

// The model decides per layer
match self.layer_type(i) {
    Softmax  => block.forward(x, |h| ops.varlen_attention(h, ...)),
    DeltaNet => block.forward(x, |h| self.deltanet[i].forward(h, ...)),
}
```

If a recurrent mixer needs multi-device support in the future, it gets its own trait (`LinearAttentionOps` or `RecurrentOps`) — it does not share a trait with softmax attention.

### Cache Allocation for Hybrid Models

Models mixing softmax attention + recurrent layers need different cache types per layer. The model declares what cache each layer needs via `LayerCacheSpec`:

```rust
enum LayerCacheSpec {
    /// Standard KV cache (softmax attention). Paged.
    Attention {
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    },
    /// Recurrent state (Mamba, DeltaNet, RWKV). Fixed size per request — no paging needed.
    Recurrent {
        state_shapes: Vec<Vec<usize>>,
        state_dtypes: Vec<DType>,
    },
    /// No cache (diffusion, embedding, encoder).
    None,
}

trait Model {
    fn cache_specs(&self) -> Vec<LayerCacheSpec>;
    fn forward(&mut self, x: &Tensor, ctx: &BatchState, ops: &dyn Ops, cache: &Cache) -> Result<Tensor>;
}
```

The engine groups layers by spec and allocates one pool per group. Attention groups use `BlockAllocator` + `PrefixCache` (paged). Recurrent groups use a fixed-size buffer (no paging — state size is constant per request).

**Design rationale vs alternatives:**
- vLLM uses a `KVCacheSpec` class hierarchy + `HybridKVCacheCoordinator` with a fixed-point algorithm. Sophisticated but complex ("support for >2 types not implemented").
- SGLang uses separate pool classes (`HybridLinearKVPool` + `MambaPool`), no unified coordinator.
- We use a 3-variant enum. Engine groups layers and allocates per group. Simpler than vLLM, more unified than SGLang.

### Prefix Cache for Hybrid Models

A prefix cache hit is valid only if **all** layer groups have the cached state at that token position. The effective hit length is the minimum across all groups:

```
Layer 0 (attention): prefix cache hit at token 512 ✓
Layer 1 (Mamba):     state available at token 512 ✓  → overall hit at 512
Layer 2 (attention): prefix cache hit at token 512 ✓
Layer 3 (Mamba):     state available at token 256 ✗  → overall hit at 256 (minimum)
```

Pure-attention models skip this check entirely.

## Dependency Graph

```
prelude-server (binary)
    ├── prelude-core               (Ops trait + default impls, models, engine — no C++, no device types)
    │       ├── cubecl                 (pure Rust: IR + TensorOps primitives, generic over CubeRuntime)
    │       └── llguidance             (constrained decoding, pure Rust)
    ├── prelude-cuda               (feature-gated, register() at startup)
    │       ├── prelude-core
    │       ├── cubecl (features=["cuda"])
    │       └── fa4/, flashinfer/, deepgemm/, cutlass-gemm/, nccl/, uccl-ep/
    ├── prelude-rocm               (feature-gated, register() at startup)
    │       ├── prelude-core
    │       ├── cubecl (features=["hip"])
    │       └── ck/, aiter/, rccl/, uccl-ep/
    ├── prelude-metal              (feature-gated, register() at startup)
    │       ├── prelude-core
    │       └── cubecl (features=["wgpu"])
    ├── prelude-vulkan             (feature-gated, register() at startup)
    │       ├── prelude-core
    │       └── cubecl (features=["wgpu"])
    ├── prelude-tpu                (feature-gated, XLATensorOps instead of CubeCL)
    │       ├── prelude-core
    │       └── pjrt C API (dlopen libpjrt_tpu.so)
    └── prelude-cpu                (always linked — lowest-priority fallback)
            ├── prelude-core
            └── cubecl (features=["cpu"]) + oneDNN
```

Key properties:
- `prelude-core` compiles no C++ and has no device-specific types.
- Device features are **additive** — `--features cuda,rocm` compiles both; runtime probe picks the right one.
- NCCL/RCCL are dlopen'd at runtime — no symbol conflicts when CUDA + ROCm are in the same binary.
- Adding a new device crate requires **zero changes** to `prelude-core` or any model file.
