# Ops and Modules

## Overview

Three layers, clear separation:

- **`Tensor`** — a data handle. Holds storage, layout, dtype, and device. No compute.
- **`Ops` trait** — the device contract. `prelude-core` defines one unified trait; device crates (`prelude-cuda`, `prelude-cpu`, …) implement it and register via the backend registry.
- **Backend registry** — at startup, device crates call `register_backend()`. `ops_for(device)` picks the highest-priority probe-passing backend for each device kind.

Model code never branches on device, fusion support, or quantization. All of that is resolved inside the `Ops` implementation selected at runtime.

---

## File Structure

```
crates/
├── prelude-core/src/
│   ├── tensor/                   # Tensor type, layout, shape, storage, DType, quantized formats
│   │   ├── mod.rs
│   │   ├── layout.rs             # strides, contiguity, view arithmetic
│   │   ├── shape.rs              # Shape, Dim, D helpers
│   │   ├── storage.rs            # CpuStorage, DeviceStorage, Storage enum (Device / CubeCL)
│   │   ├── with_dtype.rs
│   │   ├── safetensors.rs
│   │   └── quantized/            # GGUF k-quant types (K_Q4, K_Q6, …)
│   │       ├── mod.rs
│   │       ├── k_quants.rs
│   │       └── gguf_file.rs
│   └── ops/
│       ├── mod.rs                # Ops registry + backend selection + thread-local context
│       ├── traits/               # The unified Ops trait + helper types
│       │   ├── mod.rs
│       │   ├── attention.rs      # varlen_attention default impl, VarlenParams, PagedParams, MaskType
│       │   ├── conv.rs           # conv1d/conv2d default impls
│       │   ├── norm.rs           # rms_norm/layer_norm/group_norm default impls
│       │   └── ops.rs            # The Ops trait — all methods in one place
│       ├── cubecl_backend/       # CubeCL tensor primitives (validation path)
│       │   ├── mod.rs
│       │   ├── elementwise.rs
│       │   ├── matmul.rs
│       │   └── reduce.rs
│       └── device_backend/       # Pure-Rust tensor primitives (default path)
│           └── mod.rs
│
├── prelude-cuda/src/
│   ├── cuda_ops.rs               # CudaOps: implements the Ops trait
│   ├── cuda_graph.rs             # CUDA graph capture/replay (device-specific extension)
│   ├── tensor_ops_kernels.rs     # Low-level CUDA kernel dispatch
│   ├── attn/
│   │   ├── mod.rs
│   │   ├── flashinfer.rs         # FlashInfer attention backend
│   │   └── flash_v4.rs           # FA4 attention backend
│   ├── ops/
│   │   ├── mod.rs
│   │   ├── elementwise.rs        # Element-wise CUDA kernels
│   │   ├── gemm.rs               # DeepGEMM / CUTLASS dispatch
│   │   ├── kv_cache.rs           # reshape_and_cache, TurboQuant encoding
│   │   ├── moe.rs                # grouped_gemm, ep_dispatch_fused / ep_combine_fused
│   │   ├── quant.rs              # quantized_matmul (FP8, INT8, W4A16, W4A4)
│   │   ├── rmsnorm.rs            # fused_add_rmsnorm
│   │   ├── rope.rs               # fused_qknorm_rope, fused_knorm_rope_cache_write
│   │   └── tiled_mmq.rs          # Tiled mixed-precision matmul
│   └── kernels/                  # Raw CUDA kernel sources (.cu / .cuh)
│       └── kernels_src/
│           ├── candle/
│           ├── common/
│           ├── elementwise/
│           ├── kvcache/
│           ├── moe/
│           ├── normalization/
│           └── rope/
│
└── prelude-cpu/src/
    └── ops/
        ├── mod.rs
        ├── attention/            # Tiled BF16 SDPA, AVX-512 path, online softmax
        │   ├── mod.rs
        │   ├── avx512.rs
        │   ├── buffers.rs
        │   ├── common.rs
        │   ├── small.rs          # Small-sequence optimized path
        │   └── dpbf16.rs         # Dot-product BF16 path
        ├── gemm.rs               # oneDNN dispatch + scalar fallback
        ├── quant/                # GGUF k-quant dequant + matmul (Q2_K … Q6_K, IQ4_NL)
        │   └── ...
        ├── rmsnorm.rs
        ├── rope.rs
        └── silu_mul.rs
```

**Rule:** the `Ops` trait is defined in `prelude-core/src/ops/traits/ops.rs`. Device implementations live in `prelude-{cuda,cpu}/src/`. Model code only imports from `prelude-core`.

---

## Tensor

`Tensor` carries storage, layout, dtype, and device. It does not perform computation — all compute goes through the `Ops` trait.

```rust
// prelude-core/src/tensor/mod.rs

pub struct Tensor {
    storage: Arc<RwLock<Storage>>,  // device memory (Device or CubeCL path)
    layout: Layout,                 // shape + strides + offset
    dtype: DType,                   // BF16, FP16, FP32, FP8, U8, U32, I64, …
    device: Device,                 // Cpu or Cuda(n)
    id: TensorId,                   // unique identifier for debugging
}
```

Storage is held in an `Arc<RwLock<Storage>>` shared across views. `Storage` is either `Storage::Device(DeviceStorage)` (the default path) or `Storage::CubeCL(CubeCLStorage)` (the CubeCL validation path). Memory is freed when the last `Arc` drops.

### What Tensor CAN do (metadata only, no compute)

```rust
impl Tensor {
    // Shape queries
    pub fn shape(&self) -> &Shape;
    pub fn dims(&self) -> &[usize];
    pub fn dim<DD: Dim>(&self, d: DD) -> Result<usize>;
    pub fn elem_count(&self) -> usize;
    pub fn dtype(&self) -> DType;
    pub fn is_contiguous(&self) -> bool;

    // View operations — no data copy, new layout only, shared storage
    pub fn reshape(&self, s: impl ShapeWithOneHole) -> Result<Tensor>;
    pub fn narrow<DD: Dim>(&self, dim: DD, start: usize, len: usize) -> Result<Tensor>;
    pub fn squeeze<DD: Dim>(&self, dim: DD) -> Result<Tensor>;
    pub fn unsqueeze<DD: Dim>(&self, dim: DD) -> Result<Tensor>;
    pub fn transpose<D1: Dim, D2: Dim>(&self, d1: D1, d2: D2) -> Result<Tensor>;
    pub fn chunk<DD: Dim>(&self, n: usize, dim: DD) -> Result<Vec<Tensor>>;

    // Construction
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self>;

    // Raw pointer access (unsafe, via Ops — for FFI to kernels)
    unsafe fn data_ptr(&self) -> Result<*const u8>;     // via ops_for(device)
    unsafe fn data_ptr_mut(&self) -> Result<*mut u8>;
}
```

### What Tensor CANNOT do

```
❌ tensor.matmul(&other)         →  tensor.matmul(&other)  (routes through Ops internally)
❌ tensor.softmax(dim)           →  tensor.softmax(dim)     (routes through Ops internally)
❌ tensor.to_device(cuda)        →  ops.to_device(&tensor, &device)
❌ tensor.to_dtype(bf16)         →  ops.cast(&tensor, DType::BF16)
```

**Why:** computation requires knowing the device and dispatching to the right kernel. `Tensor` methods that perform compute (e.g. `tensor.matmul()`) are thin wrappers that call `ops_for(&self.device).matmul(...)` internally — they don't contain any compute themselves.

### Memory Management

**Allocation** goes through `Tensor::zeros(shape, dtype, device)`, which calls `ops_for(device).zeros(...)`. The device backend allocates memory and wraps it in the appropriate `Storage` variant.

**Deallocation** is automatic. `Storage::Device(DeviceStorage)` holds a trait object whose `Drop` implementation calls the device-specific free (e.g. `cudaFree`). `Tensor` clones share the `Arc<RwLock<Storage>>`; memory is freed when the last clone drops.

**Views** — `narrow()`, `reshape()`, etc. create a new `Tensor` with a different `Layout` but the same `Arc<RwLock<Storage>>`. The underlying memory is freed only when all views are dropped.

**Host↔device transfer:**

```rust
// via the Ops trait
ops.to_device(&cpu_tensor, &Device::Cuda(0))
```

### Operator Overloads

For ergonomics, `Tensor` supports `+`, `-`, `*`, `/` etc. via Rust operator overloading. These dispatch through `ops_for(&self.device)`, which checks a **thread-local `THREAD_OPS`** first (set by `with_ops()` / `forward_scope()` before each forward pass), then falls back to the device registry:

```rust
impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> {
        crate::ops::ops_for(&self.device).binary(self, rhs, BinaryOp::Add)
    }
}
```

### DType

```rust
// prelude-core/src/tensor/mod.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,        // packed quantized (TurboQuant, GGUF)
    U32,
    I16,       // some quantization formats
    I32,       // integer indices
    I64,
    BF16,
    F16,
    F32,
    F64,
    F8E4M3,    // FP8 (SM89+, gfx942+)
}
```

### BatchState

Per-batch runtime state passed to model forward and `Linear::forward`. Separate from `Ops` (per-device, static) and model weights (per-model, static).

```rust
// prelude-core/src/models/commons/mod.rs

pub struct BatchState<'a> {
    /// Per-token LoRA adapter index. None = LoRA not active for this batch.
    pub adapter_ids: Option<&'a Tensor>,   // [batch_size], -1 = no LoRA for this token
}
```

`Linear::forward` reads `batch_state.adapter_ids` to decide whether to apply LoRA. When `None`, the LoRA step is skipped entirely with zero overhead. Models that don't use LoRA still receive `BatchState` and forward it to `Linear` unchanged.

---

## The Ops Trait

All compute goes through a single `Ops` trait defined in `prelude-core/src/ops/traits/ops.rs`. Device crates implement it and register via `register_backend()`. Models receive a `&dyn Ops` reference — no branching on device.

```rust
// prelude-core/src/ops/traits/ops.rs

pub trait Ops: Send + Sync {
    fn default_impl(&self) -> &dyn Ops;  // terminal: return self; overlay: return inner backend

    // Tensor primitives (called by Tensor methods)
    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> { ... }
    fn binary(&self, a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> { ... }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { ... }
    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor> { ... }
    fn to_device(&self, x: &Tensor, device: &Device) -> Result<Tensor> { ... }
    // ... reduce, cat, gather, scatter_add, etc.

    // Normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> { ... }
    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> { ... }
    fn group_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, num_groups: usize, eps: f32) -> Result<Tensor> { ... }

    // Activation
    fn silu(&self, x: &Tensor) -> Result<Tensor> { ... }
    fn gelu(&self, x: &Tensor) -> Result<Tensor> { ... }
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> { ... }
    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> { ... }
    fn log_softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> { ... }

    // Convolution
    fn conv1d(&self, ...) -> Result<Tensor> { ... }
    fn conv2d(&self, ...) -> Result<Tensor> { ... }

    // GEMM (quantized / grouped)
    fn quantized_matmul(&self, a: &Tensor, b: &Tensor, sa: Option<&Tensor>, sb: Option<&Tensor>, q: QuantScheme) -> Result<Tensor> { ... }
    fn grouped_gemm(&self, input: &Tensor, weights: &Tensor, st: &Tensor, se: &Tensor, nt: &Tensor) -> Result<Tensor> { ... }

    // Attention
    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> { ... }
    fn paged_attention(&self, q: &Tensor, kc: &Tensor, vc: &Tensor, p: &PagedParams) -> Result<Tensor> { ... }

    // KV cache
    fn cache_slot_spec(&self, head_dim: usize, dtype: DType) -> CacheSlotSpec { ... }
    fn reshape_and_cache(&self, key: &Tensor, value: &Tensor, kc: &Tensor, vc: &Tensor, sm: &Tensor) -> Result<()> { ... }

    // Communication (single-device defaults are identity / no-ops)
    fn world_size(&self) -> usize { 1 }
    fn rank(&self) -> usize { 0 }
    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor> { ... }
    fn all_gather(&self, x: &Tensor, dim: usize) -> Result<Tensor> { ... }
    // ...

    // Fused ops (return None = no kernel, caller falls back)
    fn fused_add_rmsnorm(&self, ...) -> Option<Result<(Tensor, Tensor)>> { None }
    fn fused_silu_mul(&self, ...) -> Option<Result<Tensor>> { None }
    fn fused_qknorm_rope(&self, ...) -> Option<Result<(Tensor, Tensor)>> { None }
    // ...

    // Composed wrappers (try fused → fallback to separate ops automatically)
    fn add_rmsnorm(&self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> { ... }
    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> { ... }
    fn gelu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> { ... }

    // Session (per-forward-pass state — no-op defaults)
    fn begin_forward(&self) {}
    fn end_forward(&self) {}
    fn precompute_paged_plan(&self, ...) -> Result<()> { Ok(()) }
}
```

**Backend registry** (`prelude-core/src/ops/mod.rs`):

```rust
pub struct OpsBackend {
    pub name: &'static str,
    pub priority: u32,
    pub probe: fn() -> bool,           // returns true if usable on this machine
    pub supports: fn(&Device) -> bool,  // returns true for the device kinds it handles
    pub create_ops: fn() -> &'static dyn Ops,
}

pub fn register_backend(entry: OpsBackend) { ... }  // called at startup by device crates
pub fn ops_for(device: &Device) -> &'static dyn Ops { ... }  // THREAD_OPS first, then registry
```

**Thread-local context** (`with_ops` / `forward_scope`): before each forward pass the engine calls `forward_scope(ops, || model.forward(...))`, which sets `THREAD_OPS` for the duration. `ops_for()` reads this first, so all `Tensor` operator overloads pick up the right backend without any parameter threading.

**Design principles:**

- **Single trait.** All methods — primitives, norms, attention, fused ops, comm — live on one `Ops` trait. Device crates override only what they have optimized kernels for; everything else inherits defaults.
- **Fused auto-fallback via composed wrappers.** `ops.add_rmsnorm()` tries `fused_add_rmsnorm()` first; if it returns `None`, falls back to `add + rms_norm`. Models call the composed wrapper and never see `Option`.
- **`Linear` is a parameter carrier.** Holds weights, quant info, and (future) LoRA state. Passes them to `ops.xxx()`. All dispatch decisions live in the `Ops` implementation.

**Model code example (Qwen3 attention):**

```rust
let (q, k, v) = self.fused_qkv_projection(x, ops)?;
let (q, k) = match ops.fused_qknorm_rope(&q, &k, &qw, &kw, cos, sin, pos, eps) {
    Some(result) => result?,
    None => {
        let q = ops.rms_norm(&q, &qw, eps)?;
        let k = ops.rms_norm(&k, &kw, eps)?;
        (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
    }
};

if let Some(kv) = paged_kv {
    ops.paged_attention(&q, kv.key_cache, kv.value_cache, &params)?
} else {
    ops.varlen_attention(&q, &k, &v, &params)?
}
```

Zero device branching. The `Ops` implementation absorbs all device differences.

---

## Op Method Groups

All methods live on the single `Ops` trait. They are documented here by logical group.

### Attention

```rust
// prelude-core/src/ops/traits/ops.rs

fn varlen_attention(
    &self,
    q: &Tensor, k: &Tensor, v: &Tensor,
    params: &VarlenParams,
) -> Result<Tensor>;

fn paged_attention(
    &self,
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    params: &PagedParams,
) -> Result<Tensor>;

pub struct VarlenParams<'a> {
    pub cu_seqlens_q: &'a Tensor,   // [batch+1] cumulative sequence offsets
    pub cu_seqlens_k: &'a Tensor,   // [batch+1] may differ (cross-attention)
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,       // Gemma2/3/4 logit capping
}

pub struct PagedParams<'a> {
    pub block_tables: &'a Tensor,   // [batch * max_blocks_per_seq] flattened block indices
    pub cu_seqlens_q: &'a Tensor,   // [batch+1]
    pub cu_seqlens_k: &'a Tensor,   // [batch+1]
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,
}

pub enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
    /// Custom additive bias on logits before softmax.
    /// Used for speculative decoding tree attention.
    Custom(Tensor),
}
```

**Design decisions:**

- **Two methods, not one.** Contiguous and paged have fundamentally different tensor layouts (`cu_seqlens_k` vs `block_tables + cu_seqlens_k`). A single method with `Option<PagedKvRef>` would make both signatures worse.

- **Cross-attention is `varlen_attention`.** Q from decoder, K/V from encoder — pass different `cu_seqlens_q` and `cu_seqlens_k`. No special method needed.

- **MLA head_dim asymmetry is derived from tensor shapes.** `VarlenParams` does not carry `head_dim`. The implementation inspects `Q.shape[2]` and `K.shape[2]` to select the correct kernel.

- **Decode is `paged_attention` with `max_seqlen_q=1`.** The implementation dispatches to a decode-specialized kernel or a varlen kernel that handles Q=1. Model code does not distinguish prefill vs decode.

- **Chunked prefill is `paged_attention` with `max_seqlen_q>1`.** Same call, different shapes.

- **Scheduling metadata uses `&Tensor`, not `&[u32]`.** `cu_seqlens` and `block_tables` may live on any device. Using `Tensor` lets the backend read them wherever they are.

### KV Cache

```rust
// prelude-core/src/ops/traits/ops.rs

fn cache_slot_spec(&self, head_dim: usize, dtype: DType) -> CacheSlotSpec {
    CacheSlotSpec { slot_size: head_dim, dtype }  // default: uncompressed
}

fn reshape_and_cache(
    &self,
    key: &Tensor, value: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    slot_mapping: &Tensor,      // [total_tokens] flat slot indices
) -> Result<()>;

pub struct CacheSlotSpec {
    pub slot_size: usize,   // elements (or bytes for packed formats) per head per token
    pub dtype: DType,       // bf16 standard; fp8 / u8 for quantized cache
}
```

**Why separate from attention:**
1. Not all models use KV cache (diffusion doesn't).
2. Models must control when cache writes happen relative to other fusions.
3. Some devices may not support paged KV at all.

**Encode/decode is device-internal.** `reshape_and_cache` handles encoding. `paged_attention` handles decoding (unpack + dequant) before running the attention kernel. Because `CudaOps` implements both, coordination is internal state — no new trait methods, no model changes.

### GEMM

```rust
// prelude-core/src/ops/traits/ops.rs

fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

/// Quantized matmul with explicit per-tensor/per-channel scaling.
/// Covers FP8 (DeepGEMM/CK), W4A16 (AWQ), W4A4 (Nunchaku/GPTQ), INT8 (SmoothQuant).
fn quantized_matmul(
    &self,
    a: &Tensor, b: &Tensor,
    scale_a: Option<&Tensor>,
    scale_b: Option<&Tensor>,
    quant: QuantScheme,
) -> Result<Tensor>;

/// Grouped GEMM for MoE: per-expert weights applied to routed tokens.
fn grouped_gemm(
    &self,
    input: &Tensor,
    weights: &Tensor,                   // [num_experts, N, K]
    sorted_token_ids: &Tensor,
    sorted_expert_ids: &Tensor,
    num_tokens_per_expert: &Tensor,
) -> Result<Tensor>;

pub enum QuantScheme {
    Fp8E4m3,                         // FP8 E4M3, per-tensor/per-token scaling
    W4A16 { group_size: usize },     // weight-only 4-bit, per-group scaling (AWQ, GPTQ)
    W4A4  { group_size: usize },     // 4-bit weights + 4-bit activations (Nunchaku SVD-Q)
    Int8,                            // INT8 symmetric (SmoothQuant)
}
```

### Normalization

```rust
// prelude-core/src/ops/traits/ops.rs — defaults compose from tensor ops
// prelude-core/src/ops/traits/norm.rs — default implementations

fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
fn group_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, num_groups: usize, eps: f32) -> Result<Tensor>;
```

LLMs use `rms_norm`. Diffusion models use `group_norm`. TTS/vision encoders use `layer_norm`. All three are needed for multi-modal support.

### Activation

```rust
// prelude-core/src/ops/traits/ops.rs — all composed from tensor ops by default

fn silu(&self, x: &Tensor) -> Result<Tensor>;
fn gelu(&self, x: &Tensor) -> Result<Tensor>;
fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor>;   // tanh approx, used by Flux/DiT
fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
fn sigmoid(&self, x: &Tensor) -> Result<Tensor>;
fn log_softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
```

### Convolution

```rust
// prelude-core/src/ops/traits/conv.rs — default implementations

fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
          stride: usize, padding: usize) -> Result<Tensor>;
fn conv_transpose1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
                    stride: usize, padding: usize, output_padding: usize) -> Result<Tensor>;
fn conv2d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
          stride: [usize; 2], padding: [usize; 2]) -> Result<Tensor>;
```

- **LLM:** `conv1d` — DeltaNet/Mamba causal conv.
- **TTS:** `conv1d` + `conv_transpose1d` — vocoder upsampling, encoder.
- **Diffusion:** `conv2d` — UNet spatial layers, DiT patch embedding.

### Communication

```rust
// prelude-core/src/ops/traits/ops.rs — single-device defaults are identity / no-ops

fn world_size(&self) -> usize { 1 }
fn rank(&self) -> usize { 0 }

fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor>;
fn all_gather(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
fn reduce_scatter(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
/// Used for Ulysses sequence parallelism and MoE expert routing.
fn all_to_all(&self, x: &Tensor, input_splits: &[usize], output_splits: &[usize]) -> Result<Tensor>;

/// Point-to-point send. Used for attention-FFN disaggregation.
fn send(&self, x: &Tensor, dst: RemoteTarget) -> Result<()>;
fn recv(&self, src: RemoteTarget) -> Result<Tensor>;

/// Fused MoE dispatch: quantize to FP8 + GPU-initiated send to expert owners.
/// None = not supported; model falls back to all_to_all.
fn ep_dispatch_fused(&self, x: &Tensor, topk_ids: &Tensor,
                     num_experts: usize, use_fp8: bool,
) -> Option<Result<(Tensor, Tensor)>> { None }

fn ep_combine_fused(&self, x: &Tensor, topk_weights: &Tensor,
                    topk_ids: &Tensor,
) -> Option<Result<Tensor>> { None }
```

**Device implementations:**
- **CUDA:** NCCL (dlopen'd at runtime), custom all-reduce for single-node TP (P2P), UCCL-EP for fused MoE dispatch.
- **ROCm:** RCCL (dlopen'd), QuickAllReduce on MI300, UCCL-EP.
- **TPU:** XLA collective ops (compiled into HLO).
- **Single-device** (Metal, Vulkan, CPU): identity passthrough (TP=1, no communication).

**How TP uses comm ops — attention never sees it:**

```rust
// prelude-core/src/models/commons/linear.rs — inside Linear::forward, TP step
// Note: TP is not yet implemented; this shows the intended design

match self.tp {
    TpMode::Row                             => ops.all_reduce_sum(&out),
    TpMode::Column { gather_output: true }  => ops.all_gather(&out, -1),
    _                                       => Ok(out),
}
```

The QKV projection uses `Linear { tp: Column { gather_output: false } }` — each rank gets `num_heads / TP` heads. The attention kernel computes on its shard locally. All-reduce happens only in the output projection `Linear { tp: Row }`. Attention ops never need to know about TP.

### Fused Ops

Fused kernels are **methods that return `Option`**. `None` = not supported on this device. The composed wrappers on `Ops` hold the fallback — models call the wrapper, not the raw fused method.

```rust
// prelude-core/src/ops/traits/ops.rs
// All default to { None } — devices only override what they support.
// Adding a new fusion requires NO changes to existing device implementations.

fn fused_add_rmsnorm(&self, residual: &Tensor, x: &Tensor,
                     weight: &Tensor, eps: f32,
) -> Option<Result<(Tensor, Tensor)>> { None }

fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor,
) -> Option<Result<Tensor>> { None }

fn fused_qknorm_rope(&self, q: &Tensor, k: &Tensor,
                     q_weight: &Tensor, k_weight: &Tensor,
                     cos: &Tensor, sin: &Tensor,
                     position_ids: &Tensor, eps: f32,
) -> Option<Result<(Tensor, Tensor)>> { None }

fn fused_knorm_rope_cache_write(&self, k: &Tensor, v: &Tensor, k_weight: &Tensor,
                                cos: &Tensor, sin: &Tensor, position_ids: &Tensor,
                                key_cache: &Tensor, value_cache: &Tensor,
                                slot_mapping: &Tensor, eps: f32,
) -> Option<Result<()>> { None }

fn fused_adaln_zero(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
                    scale: &Tensor, shift: &Tensor, gate: &Tensor, eps: f32,
) -> Option<Result<(Tensor, Tensor)>> { None }

fn fused_scale_shift(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
                     scale: &Tensor, shift: &Tensor, eps: f32,
) -> Option<Result<Tensor>> { None }

fn fused_lora_matmul(&self, x: &Tensor, base_weight: &Tensor,
                     lora_a: &Tensor, lora_b: &Tensor,
                     adapter_indices: &Tensor, lora_scale: f32,
) -> Option<Result<Tensor>> { None }

fn fused_moe_routing(&self, router_logits: &Tensor, topk: usize,
) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> { None }

fn fused_add(&self, a: &Tensor, b: &Tensor,
) -> Option<Result<Tensor>> { None }
```

**Composed wrappers** (models call these — `Option` never escapes to model code):

```rust
fn add_rmsnorm(&self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    if let Some(r) = self.fused_add_rmsnorm(residual, x, weight, eps) { return r; }
    let sum = (residual + x)?;
    let normed = self.rms_norm(&sum, weight, eps)?;
    Ok((sum, normed))
}

fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if let Some(r) = self.fused_silu_mul(gate, up) { return r; }
    gate.silu()?.broadcast_mul(up)
}
```

**Why `Option<Result<T>>` on the raw fused methods — model holds the fallback for non-wrapped fusions:**

```rust
// prelude-core/src/models/ — at a fusion boundary not covered by a composed wrapper

let (q, k) = match ops.fused_qknorm_rope(&q, &k, ...) {
    Some(result) => result?,         // device supports it — use fused kernel
    None => {                        // fallback to separate ops
        let q = ops.rms_norm(&q, &qw, eps)?;
        let k = ops.rms_norm(&k, &kw, eps)?;
        (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
    }
};
```

CUDA returns `Some` (fused kernel). CPU/Vulkan returns `None` (separate ops). The model is always correct.

**Why not encode fusion as hints inside `AttentionParams`:** Qwen3 forward runs `fused_knorm_rope_cache_write(k, v, cache, slots)` *before* `paged_attention(q, cache)`. If the cache write were a hint inside `paged_attention`, this two-step structure would be impossible. The model must own the boundary.

**Extensibility:** adding a new fusion = one new method with `{ None }` default. Zero changes to existing device implementations.

### Session (Per-Forward-Pass State)

Some devices have per-forward-pass state that must be managed — FlashInfer plan cache, XLA compilation cache, CUDA/HIP graph pre-allocation.

```rust
// prelude-core/src/ops/traits/ops.rs — all default to no-ops

fn begin_forward(&self) {}

fn end_forward(&self) {}

fn precompute_paged_plan(
    &self,
    q_shape: (usize, usize, usize),   // (total_tokens, num_heads, head_dim)
    key_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
) -> Result<()> { Ok(()) }
```

The engine calls `forward_scope(ops, || model.forward(...))` which calls `begin_forward()` and `end_forward()` around the model call. Devices without session state (CPU, Metal, Vulkan) inherit the no-op defaults.

### CUDA Graph Capture (Device-Specific Extension)

CUDA graphs require pre-allocated buffers at fixed GPU addresses. This does **not** belong in the shared `Ops` trait — the engine doesn't need to know about it, and `GraphMetaBuffers` is meaningless on Metal/Vulkan/TPU/CPU.

Instead, `CudaOps` exposes a device-specific graph capture API. The engine's CUDA graph runner downcasts to `CudaOps` (it already knows it's on CUDA) and calls device-specific methods:

```rust
// prelude-cuda/src/cuda_graph.rs — CUDA-specific extension, not in Ops trait

impl CudaOps {
    /// Allocate pre-sized GPU buffers for graph metadata.
    pub fn allocate_graph_buffers(&self, max_batch: usize, max_blocks: usize) -> GraphMetaBuffers;

    /// Pre-compute paged plan into fixed-address graph buffers (for capture/replay).
    pub fn precompute_paged_plan_graphed(
        &self,
        block_tables: &[u32],
        cu_seqlens_k: &[u32],
        block_size: usize,
        graph_buffers: &GraphMetaBuffers,
    ) -> Result<()>;
}

/// Pre-allocated GPU tensors with fixed addresses for CUDA graph replay.
pub struct GraphMetaBuffers {
    pub indptr:        Tensor,
    pub indices:       Tensor,
    pub last_page_len: Tensor,
}
```

**CUDA graph flow (engine's CUDA graph runner):**

```
Capture:
    let cuda_ops: &CudaOps = ops.downcast();
    let graph_bufs = cuda_ops.allocate_graph_buffers(max_batch, max_blocks);
    cuda_ops.precompute_paged_plan_graphed(.., &graph_bufs);   // outside capture
    stream.begin_capture();
    model.forward(..);                                          // captured
    stream.end_capture() → graph;

Replay:
    update_input_buffers(..);
    cuda_ops.precompute_paged_plan_graphed(.., &graph_bufs);   // update metadata
    graph.launch();
```

HIP graphs on ROCm follow the same pattern via `RocmOps`.

---

## Tensor Layout Conventions

All `Ops` methods use a shared set of canonical tensor layouts. Every device implementation must accept and produce tensors in these layouts. Devices that need different internal layouts (e.g., TPU 128-byte alignment) must transpose/pad internally and return canonical layout.

```
Model tensors (live on device):
  Q, K, V (attention):          [total_tokens, num_heads, head_dim]
  O (attention output):         [total_tokens, num_heads, head_dim_v]
  key_cache, value_cache:       [num_blocks, block_size, num_heads_k, head_dim]   (paged KV)
  Linear weights:               [out_features, in_features]                        (row-major)
  Norm weights:                 [hidden_dim]
  Bias:                         [out_features]
  Conv1d input:                 [batch, channels, length]
  Conv2d input:                 [batch, channels, height, width]

Scheduling metadata (Tensor, constructed by engine, lives wherever device needs it):
  cu_seqlens:                   [batch_size + 1]                 cumulative offsets
  block_tables:                 [batch_size * max_blocks_per_seq] flattened block indices
  slot_mapping:                 [total_tokens]                   flat slot indices
```

**Why packed varlen** (`[total_tokens, ...]` + `cu_seqlens`) over padded batch (`[batch, max_seq, ...]`):
- No wasted compute on padding tokens.
- Natural for continuous batching (variable-length sequences in one batch).
- Required by all major attention kernels (FlashAttention, FlashInfer, CK, Pallas).
- Diffusion uses this too: a batch of images with different token counts → single packed tensor.

**Device-internal exceptions (invisible to model code):**
- TPU pads `head_dim` to 128-byte alignment internally.
- Metal may transpose for `simdgroup_multiply_accumulate` efficiency.
- Vulkan may pad workgroup-aligned dimensions.

---

## Device Capability Matrix

| Capability | CUDA | ROCm | Metal | Vulkan | TPU | CPU |
|---|---|---|---|---|---|---|
| **varlen_attention** | FA4 / FlashInfer | CK / aiter | MSL flash attn | GLSL flash attn | Pallas | matmul SDPA |
| **paged_attention** | FlashInfer | CK / aiter | — | — | Pallas ragged | — |
| **matmul** | DeepGEMM / CUTLASS | CK GEMM | simdgroup mm | tiled / coopmat | XLA dot_general | BLAS |
| **quantized_matmul** | DeepGEMM FP8, CUTLASS INT8 | CK FP8 | in-shader dequant (Q4–Q8, IQ) | in-shader dequant | XLA INT8/FP8 | dequant + BLAS |
| **rms_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **layer_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **group_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **conv1d / conv2d / conv_transpose1d** | CUTLASS conv | CK conv | MSL shader | GLSL shader | XLA conv | fallback |
| **fused_add_rmsnorm** | FlashInfer kernel | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **fused_adaln_zero** | Triton/CUDA kernel | — (planned) | — (planned) | — | XLA auto-fuse | — |
| **fused_qknorm_rope** | FlashInfer kernel | — | — | — | — | — |
| **fused_lora_matmul** | BGMV/Punica kernel | — (planned) | — | — | XLA custom op | — |
| **comm ops** | NCCL / custom AR | RCCL / QuickAllReduce | — (single device) | — (single device) | XLA collective | — (single device) |
| **send/recv (AFD)** | NCCL P2P / RDMA | RCCL P2P | — | — | — | — |
| **session (begin/end_forward)** | FlashInfer plan cache | no-op | no-op | no-op | XLA compile cache | no-op |
| **CUDA/HIP graphs** | yes (32 graphs) | HIP graphs (6.1+) | — | — | — | — |
| **BFloat16** | SM80+ | all CDNA | Apple6+ / Metal3+ | extension req'd | native | optional |
| **FP8** | SM89+ | gfx942 (FNUZ), gfx950 (E4M3) | — | — | v5e+ | — |
| **KV cache quant** | TurboQuant (device-internal) | — (planned) | — | — | — | — |

**Key insight:** the trait interface is the same across all devices. The difference is which methods return real results vs errors, and which fused methods return `Some` vs `None`. Model code never changes — the dispatch layer absorbs all device differences.

---

## Tensor Primitive Dual Backend (Temporary)

The `Ops` trait currently has two parallel implementations for tensor primitives under validation:

```
prelude-core/src/ops/
  traits/              — the Ops trait definition (shared)
  cubecl_backend/      — CubeCL runtime, Storage::CubeCL
  device_backend/      — pure Rust, Storage::Device (default)
```

**Why two:** CubeCL is being validated as the long-term compute backend. During this period, both implementations run the same test suite, selected by `PRELUDE_TENSOR_BACKEND` env var (`"device"` or `"cubecl"`, default `"device"`).

**Rules:**
- The two paths are completely isolated. No bridge code, no cross-storage conversion.
- `Tensor::from_vec` creates the matching storage type based on the active backend.
- `Tensor::data_ptr()` / `as_slice<T>()` / `as_mut_slice<T>()` dispatch through `Ops` — backend-agnostic API for device crates.
- `prelude-cpu`'s high-level ops (attention, rmsnorm, etc.) use only these generic APIs, never reference a specific backend directly.

**End state:** once CubeCL stabilizes, delete one backend, rename the survivor to `primitives/`, and remove the env var switch. No other code changes needed — everything already goes through `Ops` dispatch.
