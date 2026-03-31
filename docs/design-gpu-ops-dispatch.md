# Ops Dispatch Architecture

## Principles

1. **Subsystem isolation:** A person working on one subsystem should be able to do their job
   without reading or understanding any other subsystem's code. Model devs don't read kernel code.
   Kernel devs don't read model code. Device backend devs don't read other backends.
   The trait signatures are the complete contract between subsystems.

2. **Kernel optimization reach:** When a kernel optimization is added, as many models as possible
   should benefit automatically without per-model code changes. This is achieved via shared
   building blocks — optimize one building block, all models that use it benefit. O(1) change → O(N) benefit.

## Goals

1. Model code is device-agnostic: no `#[cfg(feature = "cuda")]` in models.
2. Model code is kernel-agnostic: models never reference FA4, FlashInfer, cuBLAS, etc.
3. Each operation independently dispatches to the best available kernel for the device + parameters.
4. Multi-device: CUDA, ROCm, TPU, Vulkan, CPU share the same model code.
5. Multi-model: AR (LLM), diffusion, TTS, vision all use the same op traits.
6. Fusion is explicit: models control fusion boundaries, not the dispatch layer.

## Layering

```
Model code          composes      Building blocks (shared layers)
Building blocks     call          Op traits (+ FusedOps fallback logic)
Op traits           implemented by    Device ops (CudaOps, RocmOps, CpuOps, ...)
Device ops          dispatches to     Kernel libraries (FA4, FlashInfer, CK, cuBLAS, XLA, ...)
```

**Building blocks** are shared layer implementations (e.g., `ResidualNormBlock`, `GatedMLP`, `AttentionBlock`).
They contain the `FusedOps` match/fallback logic. Models compose building blocks instead of calling
raw ops. **When a new fused kernel is added, the building block is updated once, and all models
that use it benefit automatically.** This is how one kernel optimization reaches many models.

## Tensor Layout Conventions

All op traits use a shared set of tensor layout conventions. These are **part of the contract**
— every device implementation must accept and produce tensors in these layouts.
Implementations that need different internal layouts (e.g., TPU 128-byte alignment)
must transpose/pad internally and return the canonical layout.

```
Model data (Tensor — lives on device):
  Q, K, V (attention):     [total_tokens, num_heads, head_dim]       — varlen (packed batch)
  O (attention output):     [total_tokens, num_heads, head_dim_v]
  key_cache, value_cache:   [num_blocks, block_size, num_heads_k, head_dim]  — paged KV
  Linear weights:           [out_features, in_features]               — row-major
  Norm weights:             [hidden_dim]
  Bias:                     [out_features]
  Conv1d input:             [batch, channels, length]
  Conv2d input:             [batch, channels, height, width]

Scheduling metadata (&[u32] — plain host-side data, device uploads internally):
  cu_seqlens:               [batch_size + 1]                          — cumulative sequence offsets
  block_tables:             [batch_size * max_blocks_per_seq]         — flattened block indices
  slot_mapping:             [total_tokens]                            — flat slot indices
```

**Why scheduling metadata is `&[u32]`, not `Tensor`:**
Scheduling metadata (cu_seqlens, block_tables, slot_mapping) describes batch structure,
not model computation. It is constructed by the engine on the host. The trait boundary
uses `&[u32]` to keep the interface device-agnostic — no assumptions about where integer
metadata lives or what type it uses on the device side.

**Device implementations convert internally — no performance overhead:**
The `&[u32]` at the trait boundary does NOT mean the kernel sees host memory.
Each device impl converts to its optimal internal representation:
- **CUDA/ROCm**: maintains a pre-allocated GPU buffer internally. On each call,
  async memcpy from `&[u32]` into the GPU buffer (overlaps with kernel launch),
  then passes the GPU pointer to FlashInfer/FA4. Zero extra cost vs passing a Tensor
  — the memcpy was always needed (engine computes metadata on CPU).
- **Metal**: unified memory. The `&[u32]` slice is already GPU-accessible, zero-copy.
- **TPU**: folds the values into XLA trace as compile-time constants.
- **CPU/Vulkan**: uses the slice directly.

The trait says "what" (u32 scheduling data). The device impl decides "where" (GPU buffer,
unified memory, XLA constant). This is the same encapsulation as the rest of the design.

**Why packed varlen** (`[total_tokens, ...]` with `cu_seqlens`) instead of padded batch (`[batch, max_seq, ...]`):
- No wasted compute on padding tokens.
- Natural for continuous batching (variable-length sequences in one batch).
- Required by FlashAttention, FlashInfer, CK, Pallas — all major attention kernels.
- Diffusion uses this too: batch of images with different token counts → single packed tensor.

**Device-internal exceptions** (invisible to model code):
- TPU pads head_dim to 128-byte alignment internally.
- Metal may transpose for simdgroup_multiply_accumulate efficiency.
- Vulkan may pad workgroup-aligned dimensions.
These are implementation details — the trait boundary always uses canonical layouts.

## Op Traits

### Attention

```rust
trait AttentionOps: Send + Sync {
    /// Varlen attention over contiguous Q, K, V.
    ///
    /// Covers: LLM prefill (causal), diffusion self-attention (bidirectional),
    /// cross-attention (Q and K/V from different sources with different cu_seqlens),
    /// sliding window, softcap.
    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor>;

    /// Paged attention: Q attends to K/V in block cache.
    ///
    /// Covers: LLM decode (Q=1), chunked prefill (mixed Q lengths),
    /// LLM prefill with prefix cache reuse (Q>1, K/V already in cache).
    fn paged_attention(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        params: &PagedParams,
    ) -> Result<Tensor>;
}

/// Scheduling metadata uses plain `&[u32]`, not `Tensor`.
/// These are small integer arrays describing batch structure (not model data).
/// Device implementations handle GPU upload internally:
///   - CUDA/ROCm: pre-allocated GPU buffer, memcpy before kernel launch.
///   - Metal: unified memory, zero-copy wrap.
///   - TPU: included in XLA trace as constants.
///   - CPU/Vulkan: used directly.
struct VarlenParams<'a> {
    pub cu_seqlens_q: &'a [u32],  // [batch+1], cumulative sequence offsets
    pub cu_seqlens_k: &'a [u32],  // [batch+1], may differ from cu_seqlens_q (cross-attention)
    pub max_seqlen_q: u32,
    pub max_seqlen_k: u32,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,     // Gemma2/3 logit capping
}

struct PagedParams<'a> {
    pub block_tables: &'a [u32],  // [batch * max_blocks_per_seq], flattened block indices
    pub num_seqs: u32,
    pub max_blocks_per_seq: u32,
    pub cu_seqlens_q: &'a [u32],  // [batch+1]
    pub cu_seqlens_k: &'a [u32],  // [batch+1]
    pub max_seqlen_q: u32,
    pub max_seqlen_k: u32,
    pub scale: f32,
    pub mask: MaskType,
}

enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
    /// Custom attention mask tensor. Used for speculative decoding tree attention:
    /// each token attends to its ancestors in the draft tree, not a simple causal pattern.
    /// Mask shape: [max_seqlen_q, max_seqlen_k], values are 0.0 (attend) or -inf (mask).
    /// Passed to attention kernel as additive bias on logits (before softmax).
    Custom(Tensor),
}
```

**Design decisions:**

- **Two methods, not one.** Contiguous and paged have different tensor layouts (`cu_seqlens_k` vs `block_tables + seqused_k`). A single method with `Option<PagedKvRef>` conflates them and makes both signatures worse.

- **Cross-attention is varlen_attention.** Q from decoder, K/V from encoder — just different `cu_seqlens_q` and `cu_seqlens_k`. No special method needed.

- **head_dim asymmetry (MLA) is derived from tensor shapes.** `VarlenParams` does not carry head_dim. The implementation inspects Q shape `[_, _, head_dim_q]` and K shape `[_, _, head_dim_k]` to select the correct kernel.

- **Decode is paged_attention with max_seqlen_q=1.** The implementation dispatches to a decode-specialized kernel (FlashInfer decode) or a varlen kernel that handles Q=1 (FlashInfer FA3 prefill). Model code does not distinguish prefill vs decode — it's always `paged_attention()`.

- **Chunked prefill (mixed Q lengths in one batch) is paged_attention with max_seqlen_q>1.** The varlen kernel handles mixed Q lengths via `cu_seqlens_q`. No special API.

### KV Cache

```rust
trait KvCacheOps: Send + Sync {
    /// Write K/V to paged cache at given slot positions.
    ///
    /// Separate from attention because models control timing.
    /// Example: Qwen3 runs fused_knorm_rope_cache_write() before attention.
    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &[u32],   // scheduling metadata, same as cu_seqlens
    ) -> Result<()>;
}
```

Separate from `AttentionOps` because:
1. Not all models use KV cache (diffusion doesn't).
2. Models must control when cache writes happen relative to other fusions.
3. Some devices may not support paged KV at all (Vulkan, early TPU).

### GEMM

```rust
trait GemmOps: Send + Sync {
    /// Matrix multiply. Dispatch: DeepGEMM > CUTLASS > cuBLAS > rocBLAS > XLA > CPU BLAS.
    ///
    /// Dtype-aware: if inputs are FP8/INT8, the implementation routes to
    /// quantized GEMM kernels (DeepGEMM FP8, CUTLASS INT8, cuBLAS FP8, etc.).
    /// Scale factors are tensor metadata, not separate parameters.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Quantized matmul with explicit per-tensor/per-channel scaling.
    ///
    /// Covers FP8 (DeepGEMM/cuBLAS), W4A16 (AWQ), W4A4 (Nunchaku/GPTQ) variants.
    /// For weight-only quantization: a is activations (BF16/FP16), b is quantized weights.
    /// scale_a: per-token or per-tensor activation scale (None for weight-only quant).
    /// scale_b: per-channel or per-group weight scale.
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
        weights: &Tensor,                // [num_experts, N, K]
        sorted_token_ids: &Tensor,
        sorted_expert_ids: &Tensor,
        num_tokens_per_expert: &Tensor,
    ) -> Result<Tensor>;
}

/// Quantization scheme for quantized_matmul dispatch.
enum QuantScheme {
    /// FP8 E4M3 with per-tensor/per-token scaling (DeepGEMM, cuBLAS).
    Fp8E4m3,
    /// Weight-only 4-bit with per-group scaling (AWQ, GPTQ).
    W4A16 { group_size: usize },
    /// 4-bit weights + 4-bit activations (Nunchaku SVD-Q).
    W4A4 { group_size: usize },
    /// INT8 symmetric quantization (SmoothQuant).
    Int8,
}
```

### Normalization

```rust
trait NormOps: Send + Sync {
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
    fn group_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
                  num_groups: usize, eps: f32) -> Result<Tensor>;
}
```

- LLM uses `rms_norm`.
- Diffusion uses `group_norm`.
- TTS/vision uses `layer_norm`.
- All three are needed for multi-model support.

### Activation

```rust
trait ActivationOps: Send + Sync {
    fn silu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor>; // tanh approx, used by Flux/DiT
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
}
```

### Convolution

```rust
trait ConvOps: Send + Sync {
    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
              stride: usize, padding: usize) -> Result<Tensor>;
    fn conv2d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
              stride: [usize; 2], padding: [usize; 2]) -> Result<Tensor>;
}
```

- Diffusion: conv2d (UNet, DiT).
- TTS: conv1d (vocoder, encoder).
- LLM: conv1d (DeltaNet/Mamba causal conv).

### Communication (Distributed)

```rust
trait CommOps: Send + Sync {
    /// Sum-reduce tensor across all ranks in the tensor-parallel group.
    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor>;

    /// Concatenate tensor shards from all ranks along `dim`.
    fn all_gather(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    /// Reduce-scatter: reduce across ranks, then scatter result shards.
    fn reduce_scatter(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    /// All-to-all: each rank sends/receives different data to/from every other rank.
    /// Used for Ulysses sequence parallelism and MoE expert routing.
    fn all_to_all(&self, x: &Tensor, input_splits: &[usize], output_splits: &[usize]) -> Result<Tensor>;
}
```

**Device implementations:**
- **CUDA**: NCCL (default), custom all-reduce for single-node TP (P2P), symmetric memory on H100+.
- **ROCm**: RCCL (NCCL equivalent), custom all-reduce on MI300 (QuickAllReduce).
- **TPU**: XLA collective ops (compiled into HLO, hardware-optimized).
- **Single-device** (Metal, Vulkan, CPU): identity passthrough (TP=1, no communication needed).

**How TP uses CommOps:**

```rust
/// RowParallelLinear: shard input, local GEMM, all-reduce output.
fn row_parallel_forward(x: &Tensor, ops: &Ops) -> Result<Tensor> {
    let out = ops.gemm.matmul(x, &self.weight_shard)?;  // local GEMM
    ops.comm.all_reduce_sum(&out)                         // synchronize
}

/// ColumnParallelLinear: local GEMM on sharded weights, optionally all-gather.
fn col_parallel_forward(x: &Tensor, ops: &Ops) -> Result<Tensor> {
    let out = ops.gemm.matmul(x, &self.weight_shard)?;  // local GEMM
    if self.gather_output {
        ops.comm.all_gather(&out, /*dim=*/-1)             // reconstruct full output
    } else {
        Ok(out)  // keep sharded (e.g., QKV stays sharded for local attention)
    }
}
```

**Key insight: attention ops don't know about TP.** `QKVParallelLinear` is a `ColumnParallelLinear` that shards Q/K/V output across ranks. Each rank gets `num_heads / TP` heads. The attention kernel receives already-sharded Q/K/V and computes locally — no all-reduce inside attention. All-reduce happens in the `RowParallelLinear` output projection after attention.

### Fusion

Fused kernels are **separate ops that return `Option`**. `None` = not supported on this device, model falls back to separate ops. This is not a hint system — the model explicitly checks the return value.

```rust
/// All methods have default `{ None }` — devices only override what they support.
/// Adding a new fusion method requires NO changes to existing device implementations.
trait FusedOps: Send + Sync {
    /// Fused residual add + RMSNorm: computes (x + h) and rms_norm(x + h) in one kernel.
    fn fused_add_rmsnorm(
        &self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused SiLU(gate) * up.
    fn fused_silu_mul(
        &self, gate: &Tensor, up: &Tensor,
    ) -> Option<Result<Tensor>> { None }

    /// Fused QK-norm + RoPE: normalize Q and K, apply rotary embedding.
    fn fused_qknorm_rope(
        &self,
        q: &Tensor, k: &Tensor,
        q_weight: &Tensor, k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused K-norm + RoPE + KV cache write.
    fn fused_knorm_rope_cache_write(
        &self,
        k: &Tensor, v: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor, eps: f32,
    ) -> Option<Result<()>> { None }

    /// Fused Adaptive Layer Norm (AdaLN-Zero).
    ///
    /// The signature normalization op for diffusion transformers (DiT, Flux, HunyuanVideo).
    /// Computes: normed = layer_norm(x) * (1 + scale) + shift, gated = normed * gate.
    /// (scale, shift, gate) are derived from timestep embedding by the model.
    ///
    /// Without fusion: 4 memory-bound element-wise ops. With fusion: single kernel pass.
    fn fused_adaln_zero(
        &self,
        x: &Tensor,
        weight: &Tensor, bias: Option<&Tensor>,
        scale: &Tensor, shift: &Tensor, gate: &Tensor,
        eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused scale + shift (continuous AdaLN variant).
    ///
    /// Simpler form: layer_norm(x) * (1 + scale) + shift, no gate.
    /// Used in Flux final norm, Sana, etc.
    fn fused_scale_shift(
        &self,
        x: &Tensor,
        weight: &Tensor, bias: Option<&Tensor>,
        scale: &Tensor, shift: &Tensor,
        eps: f32,
    ) -> Option<Result<Tensor>> { None }

    /// Fused multi-LoRA matmul: y = base_weight @ x + scale * (lora_b @ lora_a @ x)
    ///
    /// Each token in the batch can use a different LoRA adapter (for multi-tenant serving).
    /// adapter_indices: [batch] mapping each token to its adapter (-1 = no LoRA).
    /// lora_a: [num_adapters, rank, in_features], lora_b: [num_adapters, out_features, rank].
    ///
    /// Without fusion: split batch by adapter, N separate matmuls, merge. O(N) kernel launches.
    /// With fusion (BGMV/Punica): single kernel handles all adapters. O(1) launch.
    fn fused_lora_matmul(
        &self,
        x: &Tensor,
        base_weight: &Tensor,
        lora_a: &Tensor,
        lora_b: &Tensor,
        adapter_indices: &Tensor,
        lora_scale: f32,
    ) -> Option<Result<Tensor>> { None }
}
```

**Why `Option<Result<T>>`:**

```rust
// Model code — fusion boundary is explicit
let (q, k) = match ops.fused.fused_qknorm_rope(&q, &k, ...) {
    Some(result) => result?,            // device supports it
    None => {                           // fallback to separate ops
        let q = ops.norm.rms_norm(&q, &qw, eps)?;
        let k = ops.norm.rms_norm(&k, &kw, eps)?;
        (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
    }
};
```

The model is always correct regardless of which device it runs on. CUDA returns `Some` (fused kernel), CPU/Vulkan returns `None` (separate ops).

**Why not put fusion hints in AttentionParams:**

Qwen3 forward does this:
```
step 1: fused_knorm_rope_cache_write(k, v, cache, slots)   // writes to cache
step 2: paged_attention(q, cache)                            // reads from cache
```

If cache write were a "hint" inside `paged_attention`, step 1 couldn't exist as a separate fused kernel. The model must control the boundary between cache write and attention.

**Extensibility:** New fusions are new methods on `FusedOps`. Devices that don't support them return `None`. No combinatorial explosion — each fusion is independent, not a combination of flags.

## Session Lifecycle

Some devices have per-forward-pass state that must be managed:
- **FlashInfer**: plan cache (expensive scheduling, computed once, reused across N layers)
- **XLA (TPU)**: compilation cache, trace-based execution
- **CUDA/HIP graphs**: pre-allocated buffers with fixed GPU addresses

```rust
trait OpsSession: Send + Sync {
    /// Initialize per-forward-pass state. Called before model.forward().
    fn begin_forward(&self);

    /// Clear per-forward-pass state. Called after model.forward().
    fn end_forward(&self);

    /// Pre-compute paged attention scheduling for the current batch.
    /// Converts block_tables → kernel-specific metadata (e.g., FlashInfer indptr/indices).
    /// Called once before model.forward(), reused across all N layers.
    fn precompute_paged_plan(
        &self,
        block_tables: &Tensor,
        cu_seqlens_k: &Tensor,
        block_size: usize,
    ) -> Result<()>;
}
```

**Devices without session state** (CPU, Metal, Vulkan) implement all methods as no-ops.

### Graph Capture (CUDA/HIP-internal concern)

CUDA graphs require pre-allocated buffers at fixed GPU addresses. This is a CUDA-specific
optimization that does NOT belong in the shared `OpsSession` trait — the engine doesn't
need to know about it.

Instead, `CudaOps` exposes a **device-specific** graph capture API. The engine's CUDA
graph runner downcasts to `CudaOps` (since it already knows it's on CUDA) and calls
device-specific methods:

```rust
/// CUDA-specific extension. Not in OpsSession trait.
impl CudaOps {
    /// Allocate pre-sized GPU buffers for graph metadata.
    pub fn allocate_graph_buffers(&self, max_batch: usize, max_blocks: usize) -> GraphMetaBuffers;

    /// Pre-compute paged plan into fixed-address graph buffers (for capture/replay).
    pub fn precompute_paged_plan_graphed(
        &self,
        block_tables: &Tensor,
        cu_seqlens_k: &Tensor,
        block_size: usize,
        graph_buffers: &GraphMetaBuffers,
    ) -> Result<()>;
}

/// Pre-allocated GPU tensors with fixed addresses for CUDA graph replay.
struct GraphMetaBuffers {
    pub indptr: Tensor,
    pub indices: Tensor,
    pub last_page_len: Tensor,
}
```

**CUDA graph flow** (engine's CUDA graph runner, not generic engine):

```
Capture:
    let cuda_ops: &CudaOps = ops.downcast();   // engine knows it's on CUDA
    let graph_bufs = cuda_ops.allocate_graph_buffers(max_batch, max_blocks);
    cuda_ops.precompute_paged_plan_graphed(.., &graph_bufs);    // outside capture
    stream.begin_capture();
    model.forward(..);                                           // captured
    stream.end_capture() → graph;

Replay:
    update_input_buffers(..);
    cuda_ops.precompute_paged_plan_graphed(.., &graph_bufs);    // update metadata
    graph.launch();
```

**Why not in OpsSession:** `GraphMetaBuffers` is meaningless on Metal/Vulkan/TPU/CPU.
Putting it in the shared trait forces every device to handle CUDA-specific types.
Keeping it as a `CudaOps` method means the engine's CUDA graph runner is the only code
that knows about it — and that runner already knows it's on CUDA.

**HIP graphs:** Same pattern. `RocmOps` has equivalent `precompute_paged_plan_graphed` method.
HIP 6.1+ supports graph capture/replay with the same API shape.

## Ops Bundle

Models receive a single struct:

```rust
struct Ops {
    pub attn: Arc<dyn AttentionOps>,
    pub kv_cache: Arc<dyn KvCacheOps>,
    pub gemm: Arc<dyn GemmOps>,
    pub norm: Arc<dyn NormOps>,
    pub act: Arc<dyn ActivationOps>,
    pub conv: Arc<dyn ConvOps>,
    pub comm: Arc<dyn CommOps>,
    pub fused: Arc<dyn FusedOps>,
    pub session: Arc<dyn OpsSession>,
}
```

All fields are always present. Devices that don't support an op category (e.g., Vulkan has no `paged_attention`) return errors from those methods. This is simpler than `Option<Arc<dyn ...>>` — the error message is more informative than a missing field.

## Device Implementations

### CUDA

```rust
struct CudaOps {
    fa4: Option<FA4Registry>,
    fi: FlashInferRegistry,
    deepgemm: Option<DeepGemmRegistry>,
    cutlass: Option<CutlassHandle>,
    cublas: CublasHandle,
    fi_workspace: FlashInferWorkspace,
}
```

Dispatch logic is explicit if-else, not a capability system:

```rust
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
        // DeepGEMM: SM90+ BF16
        if let Some(dg) = &self.deepgemm {
            if let Ok(out) = dg.try_gemm(a, b) { return Ok(out); }
        }
        // CUTLASS: SM80+
        if let Some(cutlass) = &self.cutlass {
            if let Ok(out) = cutlass.try_gemm(a, b) { return Ok(out); }
        }
        // cuBLAS: always available
        self.cublas.gemm(a, b)
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

HIP is largely a CUDA translation layer (llama.cpp recompiles all CUDA kernels as HIP),
but production inference uses dedicated AMD libraries: Composable Kernels (CK) for attention,
aiter for flash attention on MI300/MI350, hipBLAS for GEMM.

Key constraints:
- **Wave size**: CDNA (MI300/MI350) = wave64, RDNA (RX 7000/9000) = wave32.
- **LDS (shared memory)**: MI300 = 64KB, MI350 = 160KB. Kernel tile sizes must vary.
- **FP8 format**: MI300 (gfx942) uses FNUZ, MI350 (gfx950) uses E4M3. Both need support.
- **Flash attention**: aiter library (gfx942/gfx950 only), CK flash attn, or CUDA-translated kernels.

```rust
struct RocmOps {
    arch: RocmArch,                       // gfx942, gfx950, gfx1100, ...
    ck_flash: Option<CKFlashAttnHandle>,  // CK flash attention (CDNA only)
    aiter: Option<AiterHandle>,           // aiter flash attention (gfx942/950)
    hipblas: HipblasHandle,
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
        // CK flash attention: CDNA fallback
        if let Some(ck) = &self.ck_flash {
            return ck.varlen(q, k, v, params);
        }
        // hipBLAS SDPA: universal fallback
        hip::sdpa_varlen(q, k, v, params)
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
        self.hipblas.gemm(a, b)
    }
    fn quantized_matmul(&self, a, b, scale_a, scale_b, quant) -> Result<Tensor> {
        match quant {
            QuantScheme::Fp8E4m3 => match self.arch {
                RocmArch::Gfx942 => hip::fp8_fnuz_gemm(a, b, scale_a, scale_b),
                RocmArch::Gfx950 => hip::fp8_e4m3_gemm(a, b, scale_a, scale_b),
                _ => bail!("FP8 GEMM requires MI300+ (gfx942+)"),
            },
            _ => hip::quantized_gemm(a, b, scale_a, scale_b, quant),
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

## Non-Softmax Token Mixers

DeltaNet, Mamba, RWKV, RetNet use recurrent state, not KV cache. Their forward signature is fundamentally different from softmax attention:

```
Softmax attention: (Q, K, V, mask) → O
DeltaNet:          (x, conv_state, recurrent_state) → (o, conv_state', recurrent_state')
Mamba:             (x, conv_state, ssm_state) → (o, conv_state', ssm_state')
```

These do NOT go through `AttentionOps`. Models own their token mixer implementations. The `TransformerBlock` handles this via closure injection:

```rust
// TransformerBlock is agnostic to token mixer type
fn forward<A, M>(&self, x: &Tensor, attn_fn: A, mlp_fn: M) -> Result<Tensor>
where A: FnOnce(&Tensor) -> Result<Tensor>
{
    let h = ops.norm.rms_norm(x, &self.ln1_weight, eps)?;
    let h = attn_fn(&h)?;  // softmax attention OR DeltaNet OR Mamba
    // ...
}
```

The model decides per-layer:
```rust
match self.layer_type(i) {
    Softmax  => block.forward(x, |h| ops.attn.varlen_attention(h, ...)),
    DeltaNet => block.forward(x, |h| self.deltanet[i].forward(h, ...)),
}
```

If DeltaNet needs multi-device support in the future, it gets its own trait (`LinearAttentionOps` or `RecurrentOps`). It does not share a trait with softmax attention.

## Distributed Execution

### Tensor Parallelism (TP)

TP shards model weights across N GPUs. **Attention ops don't know about TP** — all parallelism is handled by the linear layers that surround them.

```
Input (full hidden_state, identical on all ranks)
    ↓
[ColumnParallelLinear] QKV projection (each rank: num_heads/TP heads)
    ↓
[Local attention kernel] — each rank computes on its shard, no communication
    ↓
[RowParallelLinear] O projection → all_reduce_sum across ranks
    ↓
Output (full hidden_state, identical on all ranks)
```

Model code with TP:

```rust
struct ParallelAttention {
    qkv_proj: ColumnParallelLinear,  // weight: [3*heads/TP*hdim, hidden]
    o_proj: RowParallelLinear,        // weight: [hidden, heads/TP*hdim]
}

impl ParallelAttention {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // ColumnParallel: local GEMM, output is sharded (heads/TP)
        let qkv = self.qkv_proj.forward(x, ops)?;      // no communication
        let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

        // Attention is LOCAL — each rank has its own Q/K/V shard
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;

        // RowParallel: local GEMM + all_reduce
        self.o_proj.forward(&o, ops)                     // all_reduce inside
    }
}

struct RowParallelLinear { weight_shard: Tensor }

impl RowParallelLinear {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        let out = ops.gemm.matmul(x, &self.weight_shard)?;
        ops.comm.all_reduce_sum(&out)
    }
}
```

**KV cache with TP:** Each rank holds `num_kv_heads / TP` heads in its cache. No communication during KV cache access — each rank reads its own shard.

**GQA + TP edge case:** When `num_kv_heads < TP` (e.g., 8 KV heads with TP=16), KV heads are replicated: each rank gets 1 KV head replicated from the global set. The attention kernel handles K→Q head broadcasting locally.

### Pipeline Parallelism (PP)

PP splits layers across stages. **Engine-level orchestration, not ops-level.**

```
Stage 0 (GPU 0): layers [0:16]  → send activations → Stage 1 (GPU 1): layers [16:32]
```

Each stage has its own `Ops` bundle (same device type). The engine manages send/recv of activations between stages. No impact on op traits.

### Sequence Parallelism (SP)

For long-sequence diffusion/video models. Two patterns:

**Ulysses (all-to-all):** Shard sequence dim across ranks, all-gather before attention, reduce-scatter after.

```rust
fn sp_attention(x: &Tensor, ops: &Ops) -> Result<Tensor> {
    let x_local = ops.comm.reduce_scatter(x, /*dim=*/0)?;  // shard sequence
    let o_local = ops.attn.varlen_attention(&q, &k, &v, &params)?;  // local attention
    ops.comm.all_gather(&o_local, /*dim=*/0)                 // reconstruct full sequence
}
```

**Ring attention:** Rotate K/V between neighbors. Each rank computes partial attention and accumulates. Requires custom attention loop — not expressible through `AttentionOps` alone. Model owns the ring loop and calls `CommOps` for send/recv between steps.

### Expert Parallelism (EP)

EP distributes **complete experts** across ranks for MoE models (vs TP which shards each expert's weights). A model with 256 experts on EP=8 gives each rank 32 experts.

**Three-phase pattern: dispatch → compute → combine**

```
Phase 1 — DISPATCH: Route tokens to expert-owning ranks (all-to-all)
    Router selects top-K experts per token
    All-to-all sends each token to the rank that owns its expert

Phase 2 — COMPUTE: Local grouped GEMM on owned experts
    Each rank runs grouped_gemm on its local experts
    Only processes tokens routed to its experts

Phase 3 — COMBINE: Send results back to original ranks (all-to-all)
    Reverse all-to-all returns expert outputs to token-owning ranks
    Results weighted by router scores and summed
```

In our design, EP is a **model-level building block** using `CommOps` + `GemmOps`:

```rust
struct MoELayer {
    gate: MoeGate,                     // router
    expert_weights: Tensor,            // [num_local_experts, N, K]
    ep_size: usize,                    // expert parallel world size
    ep_rank: usize,                    // this rank's EP index
}

impl MoELayer {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        // 1. Route: select top-K experts per token
        let (topk_ids, topk_weights) = self.gate.route(x)?;

        if self.ep_size > 1 {
            // 2. Dispatch: all-to-all sends tokens to expert owners
            let (recv_tokens, recv_meta) = ep_dispatch(x, &topk_ids, ops)?;

            // 3. Compute: local grouped GEMM on owned experts
            let expert_out = ops.gemm.grouped_gemm(
                &recv_tokens, &self.expert_weights,
                &recv_meta.sorted_ids, &recv_meta.expert_ids, &recv_meta.num_tokens,
            )?;

            // 4. Combine: all-to-all sends results back
            ep_combine(&expert_out, &recv_meta, &topk_weights, ops)
        } else {
            // EP=1: standard local MoE (same as Qwen3 example)
            ops.gemm.grouped_gemm(x, &self.expert_weights, ..)
        }
    }
}

/// EP dispatch: all-to-all to send tokens to expert-owning ranks.
fn ep_dispatch(x: &Tensor, topk_ids: &Tensor, ops: &Ops) -> Result<(Tensor, DispatchMeta)> {
    let (send_counts, recv_counts) = compute_dispatch_layout(topk_ids, ops.comm.ep_size())?;
    let recv_tokens = ops.comm.all_to_all(x, &send_counts, &recv_counts)?;
    Ok((recv_tokens, DispatchMeta { .. }))
}
```

**EP + TP combined:** When EP=8 and TP=2 on 16 GPUs, experts are distributed across 8 EP ranks, and each expert's weight is sharded across 2 TP ranks. After expert compute, results are all-reduced across the TP group.

**Multiple dispatch backends:** The `all_to_all` in `CommOps` is the base primitive. Production systems use specialized backends (DeepEP for NVLink+RDMA, FlashInfer, Mooncake for elastic EP) that fuse quantization + communication for higher throughput. These can be exposed as device-specific optimizations on `CudaOps`, similar to how `CudaOps::precompute_paged_plan_graphed` is CUDA-specific.

## LoRA (Low-Rank Adaptation)

Multi-LoRA serving: a single batch contains tokens from different LoRA adapters.
Each token maps to a different adapter (or no adapter).

**Core computation:** `y = W @ x + scale * (lora_b @ lora_a @ x)`

### Where LoRA Sits

LoRA is a `FusedOps` concern. The fused BGMV/Punica kernel handles all adapters in one launch:

```rust
struct LoRALinear {
    base_weight: Tensor,                 // [out, in] — shared, possibly quantized
    lora_a: Tensor,                      // [num_adapters, rank, in]
    lora_b: Tensor,                      // [num_adapters, out, rank]
    scale: f32,                          // alpha / rank
}

impl LoRALinear {
    fn forward(&self, x: &Tensor, adapter_indices: &Tensor, ops: &Ops) -> Result<Tensor> {
        match ops.fused.fused_lora_matmul(
            x, &self.base_weight, &self.lora_a, &self.lora_b, adapter_indices, self.scale,
        ) {
            Some(r) => r?,
            None => {
                // Fallback: base matmul + per-adapter LoRA (slow but correct)
                let base = ops.gemm.matmul(x, &self.base_weight)?;
                lora_fallback(&base, x, &self.lora_a, &self.lora_b, adapter_indices, self.scale, ops)
            }
        }
    }
}
```

- **CUDA:** `fused_lora_matmul` returns `Some` — Punica/BGMV kernel. O(1) kernel launch for all adapters.
- **Other devices:** returns `None` — fallback splits batch by adapter_id, runs N matmuls. Correct but slower.

### LoRA + Quantization

Base weight can be quantized (W4A16, FP8). LoRA weights are always FP16/BF16:

```rust
fn forward(&self, x: &Tensor, adapter_indices: &Tensor, ops: &Ops) -> Result<Tensor> {
    // Base: quantized matmul
    let base = ops.gemm.quantized_matmul(x, &self.base_weight_q4, None, Some(&self.scale), W4A16 { .. })?;
    // LoRA: FP16 matmul (separate, always full precision)
    let lora = lora_forward(x, &self.lora_a, &self.lora_b, adapter_indices, ops)?;
    Ok((&base + &lora)?)
}
```

### LoRA + TP

With tensor parallelism, `lora_a` is replicated across ranks, `lora_b` is sharded like the base weight. All-gather between A and B phases:

```rust
// In ColumnParallelLinear + LoRA:
let shrunk = matmul(x, &self.lora_a)?;               // local: x @ lora_a
let gathered = ops.comm.all_gather(&shrunk, -1)?;     // synchronize
let expanded = matmul(&gathered, &self.lora_b_shard)?; // local: shard of lora_b
```

## Speculative Decoding

Speculative decoding is **engine-level orchestration**. It does not change op traits
(except `MaskType::Custom` for tree attention masks).

### Flow

```
1. Draft model generates N candidate tokens autoregressively
   — uses ops.attn.paged_attention() with max_seqlen_q=1, same as normal decode

2. Target model verifies all N+1 positions in one forward pass
   — uses ops.attn.paged_attention() with max_seqlen_q=N+1 (chunked prefill)

3. Engine compares logits, accepts k ≤ N tokens via rejection sampling

4. KV cache: rejected tokens' slots marked with PADDING_SLOT_ID = -1
   — reshape_and_cache skips -1 slots, no explicit rollback needed
```

### Tree Attention (EAGLE/Medusa)

Tree-based speculation generates a tree of candidates (multiple branching paths).
Verification uses a custom attention mask where each token attends to its ancestors:

```rust
// Engine constructs tree mask: [tree_len, tree_len]
// 0.0 = attend, -inf = mask
let tree_mask = build_tree_mask(&draft_tree);

// Target model forward with custom mask
let params = PagedParams {
    mask: MaskType::Custom(tree_mask),  // not simple Causal
    ..
};
let logits = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
```

The attention kernel passes the custom mask as additive bias on logits (before softmax).
Flash Attention supports this natively via the `attn_bias` parameter.

### EAGLE: Hidden State Reuse

EAGLE's draft model takes the target model's hidden states as input (not re-embedded tokens).
This is model-level: the target model exposes intermediate hidden states, and the draft model
consumes them. No ops changes needed — both models call the same `Ops` interface.

### Impact on Ops Design

| Spec decode concern | Where it lives | Impact on ops |
|---------------------|---------------|---------------|
| Draft model forward | Engine | None — same `Ops` as normal inference |
| Target verification | Engine | None — chunked prefill via `paged_attention` |
| Tree attention mask | `MaskType::Custom(Tensor)` | Already in `AttentionOps` |
| KV cache rollback | Engine (slot_mapping with -1) | `reshape_and_cache` skips -1 slots |
| Rejection sampling | Engine (GPU kernel) | Not an op trait concern |

## Shared Building Blocks

Building blocks are **shared layer implementations** that contain fusion/fallback logic.
Models compose them instead of calling raw ops. This is how one kernel optimization
reaches all models automatically.

### Why Building Blocks

**Problem:** Without building blocks, every model must write its own fusion fallback:

```rust
// Qwen3 model code — manual fusion logic
let (residual, h) = match ops.fused.fused_add_rmsnorm(x, &res, &w, eps) {
    Some(r) => r?,
    None => { let r = (x + &res)?; let h = ops.norm.rms_norm(&r, &w, eps)?; (r, h) }
};
```

If 20 models have this pattern, adding `fused_add_rmsnorm` requires updating all 20.

**Solution:** The building block contains the logic once:

```rust
// blocks/norm.rs — written once, used by all models
pub fn residual_norm(
    residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32, ops: &Ops,
) -> Result<(Tensor, Tensor)> {
    match ops.fused.fused_add_rmsnorm(residual, x, weight, eps) {
        Some(r) => r,
        None => {
            let r = (residual + x)?;
            let h = ops.norm.rms_norm(&r, weight, eps)?;
            Ok((r, h))
        }
    }
}
```

Now the model just writes `blocks::residual_norm(&residual, &h, &w, eps, ops)?`.
When the kernel dev adds `fused_add_rmsnorm` to CudaOps, all 20 models benefit with zero changes.

### Building Block Catalog

```rust
// ── Normalization ───────────────────────────────────────────────

/// Residual add + RMSNorm. Fuses to 1 kernel on CUDA, 2 ops elsewhere.
pub fn residual_norm(residual, x, weight, eps, ops) -> (Tensor, Tensor);

/// Residual add + LayerNorm. Same pattern for diffusion models.
pub fn residual_layer_norm(residual, x, weight, bias, eps, ops) -> (Tensor, Tensor);

/// AdaLN-Zero: layer_norm + scale + shift + gate. Fuses to 1 kernel on CUDA.
pub fn adaln_zero(x, weight, bias, scale, shift, gate, eps, ops) -> (Tensor, Tensor);

/// AdaLN continuous: layer_norm + scale + shift (no gate).
pub fn adaln_continuous(x, weight, bias, scale, shift, eps, ops) -> Tensor;

// ── Attention helpers ───────────────────────────────────────────

/// QK-norm + RoPE. Fuses to 1 kernel on CUDA (FlashInfer fused_qknorm_rope).
pub fn qk_norm_rope(q, k, qw, kw, cos, sin, pos, eps, ops) -> (Tensor, Tensor);

/// K-norm + RoPE + KV cache write. Fuses to 1 kernel on CUDA.
pub fn knorm_rope_cache_write(k, v, kw, cos, sin, pos, cache_k, cache_v, slots, eps, ops) -> ();

// ── MLP ─────────────────────────────────────────────────────────

/// SiLU-gated MLP: silu(gate) * up → down. Fuses gate*up to 1 kernel.
pub fn gated_mlp(x, gate_proj, up_proj, down_proj, ops) -> Tensor;

/// GELU MLP (diffusion): gelu(fc1) → fc2. Uses gelu_approximate on CUDA.
pub fn gelu_mlp(x, fc1, fc2, ops) -> Tensor;

// ── Linear layers (TP-aware) ────────────────────────────────────

/// Column-parallel linear: sharded output, optional all-gather.
pub fn col_parallel(x, weight_shard, gather_output, ops) -> Tensor;

/// Row-parallel linear: sharded input, all-reduce output.
pub fn row_parallel(x, weight_shard, ops) -> Tensor;

/// LoRA-augmented linear: base matmul + multi-adapter LoRA.
/// Fuses to BGMV/Punica on CUDA, per-adapter fallback elsewhere.
pub fn lora_linear(x, base_weight, lora_a, lora_b, adapter_ids, scale, ops) -> Tensor;

// ── MoE ─────────────────────────────────────────────────────────

/// MoE layer: route → (optional EP dispatch) → grouped GEMM → (optional EP combine).
pub fn moe_layer(x, gate, expert_weights, ep_config, ops) -> Tensor;
```

### How Models Use Building Blocks

```rust
// Qwen3 layer — using building blocks, no fusion logic visible
fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    // 1. Pre-attention norm (building block handles fusion internally)
    let (residual, h) = blocks::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

    // 2. QKV projection (building block handles TP + optional LoRA)
    let qkv = blocks::col_parallel(&h, &self.qkv_shard, false, ops)?;
    let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

    // 3. QK-norm + RoPE (building block handles fusion internally)
    let (q, k) = blocks::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

    // 4. Attention (raw op — no fusion opportunity here)
    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
    let h = blocks::row_parallel(&o, &self.o_proj_shard, ops)?;

    // 5. Post-attention norm + MoE (building block handles EP + fusion)
    let (residual, h) = blocks::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
    let h = blocks::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
    Ok((&residual + &h)?)
}
```

```rust
// Flux double block — same building blocks, different composition
fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor, ops: &Ops) -> Result<..> {
    let (scale1, shift1, gate1, ..) = self.img_mod.forward(temb)?;

    // AdaLN-Zero (building block handles fusion internally)
    let (img_normed, img_gate) = blocks::adaln_zero(
        img, &self.img_ln, None, &scale1, &shift1, &gate1, eps, ops,
    )?;

    // QK-norm (same building block as Qwen3, different context)
    let img_q = ops.norm.rms_norm(&img_q, &self.img_q_norm, eps)?;
    // ...

    // Joint attention (raw op)
    let attn_out = ops.attn.varlen_attention(&q, &k, &v, &params)?;

    // GELU MLP (building block)
    let mlp_out = blocks::gelu_mlp(&img_mlp_in, &self.fc1, &self.fc2, ops)?;
    // ...
}
```

### Kernel Optimization Reach

When a kernel dev adds a new optimization, how many models benefit?

| Optimization | Changed in | Models that benefit |
|-------------|-----------|-------------------|
| Faster FlashInfer FA3 | `CudaOps::varlen_attention` | **All models** (every model calls attention) |
| `fused_add_rmsnorm` kernel | `CudaOps` + `blocks::residual_norm` | **All transformer models** (Qwen3, Llama, Gemma, ...) |
| `fused_adaln_zero` kernel | `CudaOps` + `blocks::adaln_zero` | **All diffusion models** (Flux, HunyuanVideo, Sana, ...) |
| `fused_qknorm_rope` kernel | `CudaOps` + `blocks::qk_norm_rope` | **All QK-norm models** (Qwen3, Gemma3, ...) |
| `fused_silu_mul` kernel | `CudaOps` + `blocks::gated_mlp` | **All SiLU-gated MLP models** (most LLMs) |
| DeepGEMM FP8 improvement | `CudaOps::matmul` | **All models** (every model does matmul) |
| BGMV LoRA kernel | `CudaOps` + `blocks::lora_linear` | **All LoRA-served models** |
| Better NCCL all-reduce | `CudaOps::all_reduce_sum` | **All TP-distributed models** |

**Rule of thumb:** Ops-level improvements (attention, matmul) reach ALL models. Building-block-level improvements (fusion) reach all models that use that building block. Both are O(1) changes for O(N) model benefit.

## Model Code Pattern

Models compose building blocks for common patterns and call raw ops for model-specific logic:

```rust
// LLM attention layer
fn forward(&self, x: &Tensor, ops: &Ops, kv: Option<&PagedKvCtx>) -> Result<Tensor> {
    let (q, k, v) = self.qkv_proj(x, ops)?;

    // Fused QK-norm + RoPE (explicit check, explicit fallback)
    let (q, k) = match ops.fused.fused_qknorm_rope(&q, &k, &qw, &kw, &cos, &sin, &pos, eps) {
        Some(r) => r?,
        None => {
            let q = ops.norm.rms_norm(&q, &qw, eps)?;
            let k = ops.norm.rms_norm(&k, &kw, eps)?;
            (apply_rope(&q, &cos, &sin, &pos)?, apply_rope(&k, &cos, &sin, &pos)?)
        }
    };

    let o = if let Some(kv) = kv {
        // Fused cache write (explicit check, explicit fallback)
        match ops.fused.fused_knorm_rope_cache_write(&k, &v, ..., &kv.cache, &kv.slots, ..) {
            Some(r) => r?,
            None => ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?,
        };
        ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?
    } else {
        ops.attn.varlen_attention(&q, &k, &v, &varlen_params)?
    };

    self.o_proj.forward(&o, ops)
}
```

```rust
// Diffusion self-attention — same AttentionOps, different params
fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
    let (q, k, v) = self.qkv_proj(x)?;
    let params = VarlenParams { mask: MaskType::Bidirectional, .. };
    ops.attn.varlen_attention(&q, &k, &v, &params)
}
```

```rust
// Diffusion cross-attention — Q from decoder, K/V from encoder
fn forward(&self, x: &Tensor, context: &Tensor, ops: &Ops) -> Result<Tensor> {
    let q = self.q_proj(x)?;
    let k = self.k_proj(context)?;
    let v = self.v_proj(context)?;
    let params = VarlenParams {
        cu_seqlens_q: /* decoder seqlens */,
        cu_seqlens_k: /* encoder seqlens */,
        mask: MaskType::Bidirectional,
        ..
    };
    ops.attn.varlen_attention(&q, &k, &v, &params)
}
```

## Construction

```rust
fn create_ops(device: &Device, config: &OpsConfig) -> Ops {
    match device.device_type() {
        DeviceType::Cuda => {
            let cuda = Arc::new(CudaOps::new(config));
            Ops {
                attn: cuda.clone(), kv_cache: cuda.clone(), gemm: cuda.clone(),
                norm: cuda.clone(), act: cuda.clone(), conv: cuda.clone(),
                comm: cuda.clone(), fused: cuda.clone(), session: cuda,
            }
        }
        DeviceType::Rocm => {
            let rocm = Arc::new(RocmOps::new(config));
            Ops {
                attn: rocm.clone(), kv_cache: rocm.clone(), gemm: rocm.clone(),
                norm: rocm.clone(), act: rocm.clone(), conv: rocm.clone(),
                comm: rocm.clone(), fused: rocm.clone(), session: rocm,
            }
        }
        DeviceType::Metal => {
            let metal = Arc::new(MetalOps::new(config));
            Ops {
                attn: metal.clone(), kv_cache: metal.clone(), gemm: metal.clone(),
                norm: metal.clone(), act: metal.clone(), conv: metal.clone(),
                comm: metal.clone(), fused: metal.clone(), session: metal,
            }
        }
        DeviceType::Vulkan => {
            let vk = Arc::new(VulkanOps::new(config));
            Ops {
                attn: vk.clone(), kv_cache: vk.clone(), gemm: vk.clone(),
                norm: vk.clone(), act: vk.clone(), conv: vk.clone(),
                comm: vk.clone(), fused: vk.clone(), session: vk,
            }
        }
        DeviceType::Tpu => {
            let tpu = Arc::new(TpuOps::new(config));
            Ops {
                attn: tpu.clone(), kv_cache: tpu.clone(), gemm: tpu.clone(),
                norm: tpu.clone(), act: tpu.clone(), conv: tpu.clone(),
                comm: tpu.clone(), fused: tpu.clone(), session: tpu,
            }
        }
        DeviceType::Cpu => {
            let cpu = Arc::new(CpuOps);
            Ops {
                attn: cpu.clone(), kv_cache: cpu.clone(), gemm: cpu.clone(),
                norm: cpu.clone(), act: cpu.clone(), conv: cpu.clone(),
                comm: cpu.clone(), fused: cpu.clone(), session: cpu,
            }
        }
    }
}
```

All fields populated for every device. Methods on unsupported ops return errors (`bail!("paged attention not supported on {device}")`) rather than panicking or silently degrading.

## Subsystem Independence

Each subsystem can be developed by one person who only knows the trait signatures (shared contract)
and their own internals. No cross-subsystem code reading required.

| Subsystem | Needs to know | Does NOT need to know |
|-----------|--------------|----------------------|
| **Model impl** (Qwen3, Flux, TTS) | Building block APIs, `Ops` bundle | Any device impl, kernel library, engine |
| **Building blocks** (residual_norm, gated_mlp, ...) | Op trait signatures, `FusedOps` match pattern | Device internals, model specifics |
| **CudaOps** | Op trait signatures, FA4/FlashInfer/DeepGEMM/cuBLAS APIs | Model code, other devices |
| **RocmOps** | Op trait signatures, CK/aiter/hipBLAS APIs | CUDA code, model code |
| **MetalOps** | Op trait signatures, Metal/MSL API | CUDA/ROCm code, model code |
| **VulkanOps** | Op trait signatures, Vulkan/SPIR-V API | Other devices, model code |
| **TpuOps** | Op trait signatures, XLA/Pallas API | Other devices, model code |
| **Kernel wrapper** (FA4, FlashInfer, DeepGEMM) | Kernel library C API | Op traits, model code, other wrappers |
| **Comm backend** (NCCL, RCCL, XLA coll.) | `CommOps` trait, communication library API | Model code, attention kernels |
| **Engine/Scheduler** | `OpsSession`, `PagedKvCtx`, model `forward()` signature | Kernel implementations, device internals |
| **KV Cache Manager** | Block allocation logic, `block_tables`/`slot_mapping` layout | Attention kernels, device code |

**Three design choices that enable this:**

1. **`FusedOps` default methods** — adding a new fusion only touches the trait definition (1 line with `{ None }` default) and the device that implements it. Other devices don't change. Model developer adds call site + fallback. No cross-team coordination needed beyond agreeing on the method signature.

2. **Tensor layout conventions** — formalized in the doc (above), not just comments. Every device implementation accepts and returns canonical layouts. Device-internal transformations (TPU 128-byte padding, Metal transpose) are invisible to callers.

3. **Graph capture is device-internal** — `GraphMetaBuffers` and CUDA graph capture/replay are `CudaOps` methods, not in `OpsSession`. The engine's generic path uses `session.begin_forward()` / `session.end_forward()`. Only the CUDA-specific graph runner (which already knows it's on CUDA) calls `CudaOps::precompute_paged_plan_graphed`.

## Device Capability Matrix

What each device supports today. "Planned" = not yet implemented but feasible.

| Capability | CUDA | ROCm | Metal | Vulkan | TPU | CPU |
|------------|------|------|-------|--------|-----|-----|
| **varlen_attention** | FA4 / FlashInfer | CK / aiter | MSL flash attn | GLSL flash attn | Pallas | matmul SDPA |
| **paged_attention** | FlashInfer | CK / aiter | — | — | Pallas ragged | — |
| **matmul** | DeepGEMM/CUTLASS/cuBLAS | hipBLAS | simdgroup mm | tiled / coopmat | XLA dot_general | BLAS |
| **quantized_matmul** | DeepGEMM FP8, CUTLASS INT8 | hipBLAS FP8 | in-shader dequant (Q4-Q8, IQ) | in-shader dequant (Q4-Q8, IQ) | XLA INT8/FP8 | dequant + BLAS |
| **rms_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **layer_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **group_norm** | fused CUDA | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **conv1d / conv2d** | cuDNN / custom | hipDNN | MSL shader | GLSL shader | XLA conv | fallback |
| **fused_add_rmsnorm** | FlashInfer kernel | HIP kernel | MSL shader | GLSL shader | XLA auto-fuse | vectorized |
| **fused_adaln_zero** | Triton/CUDA kernel | — (planned) | — (planned) | — | XLA auto-fuse | — |
| **fused_qknorm_rope** | FlashInfer kernel | — | — | — | — | — |
| **fused_lora_matmul** | BGMV/Punica kernel | — (planned) | — | — | XLA custom op | — |
| **CommOps** | NCCL / custom AR | RCCL | — (single device) | — (single device) | XLA collective | — (single device) |
| **OpsSession** | FlashInfer plan cache | no-op | no-op | no-op | XLA compile cache | no-op |
| **CUDA graphs** | yes | HIP graphs (6.1+) | — | — | — | — |
| **BFloat16** | SM80+ | all CDNA | Apple6+/Metal3+ | extension req'd | native | optional |
| **FP8** | SM89+ | gfx942 (FNUZ), gfx950 (E4M3) | — | — | v5e+ | — |

**Key insight:** The trait interface is the same across all devices. The difference is which
methods return real results vs errors, and which `FusedOps` return `Some` vs `None`.
Model code never changes — the dispatch layer absorbs all device differences.

## Summary

| Concern | Solution |
|---------|----------|
| FA4 can't decode | `CudaOps::paged_attention` routes Q=1 to FlashInfer |
| Fusion control | `FusedOps` trait + building blocks encapsulate fallback logic |
| Kernel → multi-model reach | Building blocks: one optimization update → all models using that block benefit |
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
| Sequence parallelism | `CommOps::reduce_scatter` + `all_gather` around local attention |
| Multi-LoRA serving | `FusedOps::fused_lora_matmul` (BGMV/Punica), fallback to per-adapter matmul |
| Speculative decoding | Engine-level; tree attention via `MaskType::Custom(Tensor)` |

## Model Examples

Concrete examples showing how real model architectures map onto this design.

### Example 1: Qwen3-32B (LLM with GQA + MoE, Building Blocks)

Standard AR LLM. 64 layers, GQA (40 Q heads / 8 KV heads), hdim 128, MoE (8 active / 128 total experts).
Uses building blocks — no fusion fallback logic in model code.

```rust
impl Qwen3Layer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // 1. Pre-attention: building block handles fused_add_rmsnorm internally
        let (residual, h) = blocks::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

        // 2. QKV projection (TP-aware)
        let qkv = blocks::col_parallel(&h, &self.qkv_shard, false, ops)?;
        let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

        // 3. QK-norm + RoPE: building block handles fused_qknorm_rope internally
        let (q, k) = blocks::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

        // 4. KV cache + attention (raw ops — no building block needed)
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
        let h = blocks::row_parallel(&o, &self.o_proj_shard, ops)?;

        // 5. Post-attention norm + MoE
        let (residual, h) = blocks::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = blocks::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
        Ok((&residual + &h)?)
    }
}
```

Key points:
- **No `match` / `None` / fallback** in model code. All fusion logic is inside building blocks.
- `paged_attention` handles both decode (Q=1) and chunked prefill (Q>1). Model doesn't distinguish.
- `blocks::moe_layer` handles EP dispatch/combine internally when `ep_config.ep_size > 1`.
- If a kernel dev adds `fused_add_rmsnorm` to CudaOps, this model benefits with zero changes.

### Example 2: Flux (Diffusion Transformer, Building Blocks)

DiT with joint text+image attention. 19 double-stream blocks + 38 single-stream blocks.
No KV cache, no paged attention, no causal masking.

```rust
impl FluxDoubleBlock {
    fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor, ops: &Ops) -> Result<(Tensor, Tensor)> {
        // Timestep modulation: MLP maps temb → 6 affine params per stream
        let (is1, ih1, ig1, is2, ih2, ig2) = self.img_mod.forward(temb)?;
        let (ts1, th1, tg1, ts2, th2, tg2) = self.txt_mod.forward(temb)?;

        // 1. AdaLN-Zero (building block handles fused_adaln_zero internally)
        let (img_n, img_gate) = blocks::adaln_zero(img, &self.img_ln, None, &is1, &ih1, &ig1, eps, ops)?;
        let (txt_n, txt_gate) = blocks::adaln_zero(txt, &self.txt_ln, None, &ts1, &th1, &tg1, eps, ops)?;

        // 2. QKV + QK-norm
        let (img_q, img_k, img_v) = self.img_qkv_proj(&img_n, ops)?;
        let (txt_q, txt_k, txt_v) = self.txt_qkv_proj(&txt_n, ops)?;
        let img_q = ops.norm.rms_norm(&img_q, &self.img_q_norm, eps)?;
        let img_k = ops.norm.rms_norm(&img_k, &self.img_k_norm, eps)?;
        let txt_q = ops.norm.rms_norm(&txt_q, &self.txt_q_norm, eps)?;
        let txt_k = ops.norm.rms_norm(&txt_k, &self.txt_k_norm, eps)?;

        // 3. Joint attention: concat text + image, bidirectional
        let q = cat(&[&txt_q, &img_q], 0)?;
        let k = cat(&[&txt_k, &img_k], 0)?;
        let v = cat(&[&txt_v, &img_v], 0)?;
        let attn_out = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask: MaskType::Bidirectional, ..
        })?;
        let (txt_attn, img_attn) = attn_out.split_at(txt_len)?;

        // 4. Output proj + gated residual
        let img = (img + &(ops.gemm.matmul(&img_attn, &self.img_out)? * &img_gate)?)?;
        let txt = (txt + &(ops.gemm.matmul(&txt_attn, &self.txt_out)? * &txt_gate)?)?;

        // 5. MLP sub-layer with AdaLN-Zero
        let (img_n2, img_gate2) = blocks::adaln_zero(&img, &self.img_ln2, None, &is2, &ih2, &ig2, eps, ops)?;
        let img = (&img + &(blocks::gelu_mlp(&img_n2, &self.img_fc1, &self.img_fc2, ops)? * &img_gate2)?)?;
        // (txt symmetric, omitted)

        Ok((img, txt))
    }
}
```

Key points:
- `blocks::adaln_zero` called 4x per double block. 19 blocks = 76 calls. Building block handles fused/fallback internally.
- Joint attention: concat text + image → `varlen_attention` with `MaskType::Bidirectional`. Same `AttentionOps` as LLM.
- No KV cache, no paged attention. Diffusion is stateless per denoising step.
- Same building blocks (`rms_norm`, `gelu_mlp`) as LLM — kernel optimizations on these benefit both.

### Example 3: Qwen3-TTS Code Predictor (Small Causal AR, No KV Cache)

5-layer dense transformer predicting residual codec codes. Re-prefills every step (no KV cache).
Very similar to a small LLM but uses `varlen_attention` instead of `paged_attention`.

```rust
struct CodePredictorLayer {
    ln: Tensor,
    q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear,
    q_norm: Tensor, k_norm: Tensor,
    gate_proj: Linear, up_proj: Linear, down_proj: Linear,
}

impl CodePredictorLayer {
    fn forward(&self, x: &Tensor, ops: &Ops, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = ops.norm.rms_norm(x, &self.ln, eps)?;

        // Separate Q/K/V projections (no fused QKV for numerical stability)
        let q = ops.gemm.matmul(&h, &self.q_proj)?;
        let k = ops.gemm.matmul(&h, &self.k_proj)?;
        let v = ops.gemm.matmul(&h, &self.v_proj)?;

        // QK norm + RoPE
        let q = ops.norm.rms_norm(&q, &self.q_norm, eps)?;
        let k = ops.norm.rms_norm(&k, &self.k_norm, eps)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;

        // Causal self-attention — no KV cache, re-prefill every AR step
        let params = VarlenParams {
            cu_seqlens_q: /* ... */,
            cu_seqlens_k: /* same as q */,
            mask: MaskType::Causal,
            ..
        };
        let o = ops.attn.varlen_attention(&q, &k, &v, &params)?;
        let o = ops.gemm.matmul(&o, &self.o_proj)?;

        // SiLU-gated MLP
        let h = ops.norm.rms_norm(&(x + &o)?, &self.ln2, eps)?;
        let gate = ops.gemm.matmul(&h, &self.gate_proj)?;
        let up = ops.gemm.matmul(&h, &self.up_proj)?;
        let h = match ops.fused.fused_silu_mul(&gate, &up) {
            Some(r) => r?,
            None => (ops.act.silu(&gate)? * &up)?,
        };
        let h = ops.gemm.matmul(&h, &self.down_proj)?;
        Ok((x + &o + &h)?)
    }
}
```

Key points:
- Same `varlen_attention` as Flux but with `MaskType::Causal`. Kernel dispatch is the same — implementation picks FA4/FlashInfer based on shape.
- No `paged_attention` or `KvCacheOps`. TTS codec predictor is too small (5 layers) to benefit from KV cache — re-prefill is faster.
- Same `Ops` bundle as LLM. The model simply doesn't call cache/paged methods.

### Example 4: Qwen3-Omni Thinker (Multimodal LLM with Vision+Audio Encoders)

Standard AR LLM core, but receives pre-computed embeddings from vision and audio encoders.
Each encoder is a separate stage with its own `Ops`.

```rust
// Stage 1: Vision encoder (runs independently, no KV cache)
struct VisionEncoder { /* ViT layers */ }

impl VisionEncoder {
    fn forward(&self, pixel_values: &Tensor, ops: &Ops) -> Result<Tensor> {
        let patches = self.patch_embed.forward(pixel_values)?;  // Conv3d → Linear
        let mut h = patches;
        for layer in &self.layers {
            let q = ops.gemm.matmul(&h, &layer.qkv)?;
            // Bidirectional self-attention over image patches (no causal mask)
            let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
                mask: MaskType::Bidirectional,
                ..
            })?;
            h = layer.mlp(&o, ops)?;
        }
        self.spatial_merge(&h)  // 2x2 patch pooling
    }
}

// Stage 2: Thinker (standard LLM with injected multimodal embeddings)
struct Thinker { /* Qwen3 MoE layers */ }

impl Thinker {
    fn forward(
        &self,
        input_ids: &Tensor,
        image_embeds: Option<&Tensor>,  // from vision encoder
        audio_embeds: Option<&Tensor>,  // from audio encoder
        ops: &Ops,
        kv: &PagedKvCtx,
    ) -> Result<Tensor> {
        // Merge multimodal embeddings into token sequence
        let mut h = self.embed_tokens(input_ids)?;
        if let Some(img) = image_embeds {
            h = merge_at_positions(&h, img, &image_positions)?;
        }
        if let Some(aud) = audio_embeds {
            h = merge_at_positions(&h, aud, &audio_positions)?;
        }

        // Standard LLM forward with KV cache (same as Qwen3-32B example)
        for layer in &self.layers {
            h = layer.forward(&h, ops, kv)?;
        }
        self.lm_head(&h)
    }
}
```

Key points:
- Vision/audio encoders use `varlen_attention` with `MaskType::Bidirectional`. Same attention trait as everything else.
- Encoders use `conv2d` (vision patch embed) or `conv1d` (audio feature extract) from `ConvOps`.
- Thinker is a normal LLM — the multimodal part is just embedding injection before the first layer.
- Each stage can run on different devices or with different `Ops` configs (e.g., encoder on a separate GPU).

### Example 5: Qwen3-Omni Talker + Code2Wav (TTS Streaming Pipeline)

Multi-stage streaming: Thinker outputs hidden states → Talker predicts codec layer-0 → Code Predictor fills remaining 15 RVQ layers → Code2Wav decodes waveform. Each stage has its own `Ops`.

```rust
// Stage 3: Talker (5-layer AR decoder, produces layer-0 codec codes)
// → Same pattern as Code Predictor (Example 3), uses varlen_attention + Causal

// Stage 4: Code2Wav (neural vocoder, streaming decode)
struct Code2Wav {
    decoder: Vec<UpsampleBlock>,  // ConvTranspose1d + ConvNeXt blocks
}

struct UpsampleBlock {
    upsample: ConvTranspose1d,
    convnext: Vec<ConvNeXtBlock>,  // depthwise conv1d + layer_norm + MLP
}

impl Code2Wav {
    fn decode_chunk(&self, codes: &Tensor, ops: &Ops) -> Result<Tensor> {
        let mut h = self.embed(codes)?;  // [B, hidden, T]
        for block in &self.decoder {
            // ConvTranspose1d upsampling (not in ConvOps — use raw kernel or extend)
            h = block.upsample.forward(&h)?;
            for cnx in &block.convnext {
                // Depthwise conv1d + LayerNorm + GELU + pointwise conv1d
                let residual = h.clone();
                h = ops.conv.conv1d(&h, &cnx.dw_weight, None, 1, cnx.padding)?;
                h = ops.norm.layer_norm(&h, &cnx.ln_weight, Some(&cnx.ln_bias), eps)?;
                h = ops.act.gelu(&h)?;
                h = ops.conv.conv1d(&h, &cnx.pw_weight, Some(&cnx.pw_bias), 1, 0)?;
                h = (&residual + &h)?;
            }
        }
        h  // [B, 1, num_samples] waveform
    }
}
```

Key points:
- Code2Wav uses `conv1d` + `layer_norm` + `gelu` — all in existing op traits.
- Streaming: engine feeds 10-token chunks to `decode_chunk()`. Each chunk produces a waveform segment. No attention at all — pure convolutional.
- Each pipeline stage gets its own `Ops` bundle. Code2Wav doesn't need `AttentionOps` or `KvCacheOps`, but they're still present in the bundle (methods would error if called).
- ConvTranspose1d (upsampling) is not in `ConvOps` — extend later if needed, or use raw kernel call.

### Example 6: FP8 Quantized Inference (DeepSeek-V3 with FP8 Weights)

Using `quantized_matmul` for FP8 GEMM with per-token activation scaling.

```rust
struct QuantizedLinear {
    weight_fp8: Tensor,     // [N, K] in FP8 E4M3
    weight_scale: Tensor,   // [N] per-channel scale
}

impl QuantizedLinear {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        // x: [B, K] in BF16. Quantize activation on-the-fly.
        let (x_fp8, act_scale) = quantize_per_token_fp8(x)?;
        ops.gemm.quantized_matmul(
            &x_fp8, &self.weight_fp8,
            Some(&act_scale), Some(&self.weight_scale),
            QuantScheme::Fp8E4m3,
        )
    }
}
```

On CUDA: routes to DeepGEMM FP8 (SM90+) or cuBLAS FP8.
On ROCm: routes to hipBLAS FP8 (gfx942 FNUZ or gfx950 E4M3 auto-selected).
On CPU: `quantized_matmul` dequantizes and falls back to BLAS.

### Example 7: Llama-3-8B on Apple M4 (Metal, Q4_K Quantized)

On-device inference with 4-bit quantized weights on Apple Silicon.
Uses Metal compute shaders (MSL) with simdgroup matrix multiply.

```rust
// Same model code as any other device — only Ops differ.
// Weights are loaded as Q4_K quantized tensors.

struct QuantizedLinear {
    weight_q4k: Tensor,      // [N, K] in Q4_K (4-bit with K-means, 32 elements per block)
    weight_scale: Tensor,    // per-group scales
}

impl QuantizedLinear {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        // Metal dequantizes in-shader during compute (no separate dequant pass)
        ops.gemm.quantized_matmul(
            x, &self.weight_q4k,
            None, Some(&self.weight_scale),
            QuantScheme::W4A16 { group_size: 32 },
        )
    }
}

// Model forward is identical to CUDA/ROCm version
fn forward(x: &Tensor, ops: &Ops) -> Result<Tensor> {
    let h = ops.norm.rms_norm(x, &self.ln, eps)?;
    let qkv = self.qkv_proj.forward(&h, ops)?;  // quantized matmul via Metal
    let (q, k, v) = split_qkv(&qkv, 32, 8);
    let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

    // Metal flash attention (MSL compute shader, simdgroup tiling)
    // No paged KV — uses contiguous varlen attention
    let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
        mask: MaskType::Causal,
        ..
    })?;
    // ...
}
```

Key points:
- **Same model code** as CUDA. The only difference is `create_ops(DeviceType::Metal, ..)`.
- **Unified memory**: no CPU→GPU transfers. Tensors allocated once, visible to both.
- **Q4_K quantized matmul**: Metal MSL shader dequantizes in-register during compute. No separate dequant kernel — more memory-efficient than CUDA's approach for small batch.
- **No paged attention**: Metal uses `varlen_attention` with contiguous KV. Acceptable for single-user on-device inference (no batching needed).
- **No fused_qknorm_rope**: `FusedOps` returns `None`, model falls back to separate `rms_norm` + `apply_rope`. Correct, just uses 2 Metal dispatches instead of 1.

### Example 8: Flux on Vulkan (Edge Diffusion, Cross-Vendor GPU)

Running Flux image generation on an Intel Arc or AMD RX GPU via Vulkan.
Same model code as CUDA Flux example — only device dispatch differs.

```rust
// Exactly the same FluxDoubleBlock code as Example 2.
// On Vulkan:
//   - fused_adaln_zero returns None → model uses layer_norm + scale + shift (3 shaders)
//   - varlen_attention uses GLSL flash attention shader (scalar or cooperative matrix)
//   - matmul uses tiled GLSL compute shader (or cooperative matrix on Nvidia)

fn flux_forward_on_vulkan(img: &Tensor, txt: &Tensor, temb: &Tensor, ops: &Ops) -> Result<..> {
    // AdaLN-Zero: fused_adaln_zero returns None on Vulkan
    let img_normed = match ops.fused.fused_adaln_zero(img, &ln, None, &scale, &shift, &gate, eps) {
        Some(r) => r?,
        None => {
            // Fallback: 3 separate Vulkan compute dispatches (layer_norm, scale, shift)
            // Performance penalty ~10-20% vs fused kernel, but correct.
            let n = ops.norm.layer_norm(img, &ln, None, eps)?;
            let n = &n * &(1.0 + &scale)? + &shift;
            (n, gate.clone())
        }
    };

    // Joint attention: same GLSL flash attention shader, bidirectional
    let q = cat(&[&txt_q, &img_q], 0)?;
    let k = cat(&[&txt_k, &img_k], 0)?;
    let v = cat(&[&txt_v, &img_v], 0)?;
    let attn_out = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
        mask: MaskType::Bidirectional,
        ..
    })?;
    // ...
}
```

Key points:
- **Same model code**. No `#[cfg(vulkan)]` anywhere.
- **Fusion degrades gracefully**: `fused_adaln_zero` → `None` → 3 separate shaders. ~10-20% slower per block, but diffusion is latency-tolerant (20-50 denoising steps dominate).
- **Flash attention works on Vulkan**: GLSL compute shader with configurable tile sizes via specialization constants. Performance ~60-70% of CUDA on comparable Nvidia hardware.
- **Quantized weights**: Vulkan supports Q4_0 through IQ4_NL in-shader dequant — important for running Flux on 8GB consumer GPUs.
- **Cross-vendor**: same binary runs on AMD, Intel, Nvidia, Qualcomm.

### Example 9: Llama-3-70B on TPU v5e (XLA, Paged Attention)

Data center inference on TPU with paged KV cache. Key difference:
static shapes and XLA compilation.

```rust
// Same model code. TpuOps handles XLA constraints internally.

fn forward(x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    let h = ops.norm.rms_norm(x, &self.ln, eps)?;      // → XLA rms_norm op
    let qkv = ops.gemm.matmul(&h, &self.qkv_proj)?;    // → XLA dot_general (MXU)
    let (q, k, v) = split_qkv(&qkv, 64, 8);
    let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

    // KV cache write
    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;

    // Paged attention — TPU supports this via Pallas
    // TpuOps internally pads head_dim to 128-byte alignment
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
        block_tables: kv.block_tables.clone(),
        max_seqlen_q: 1,  // decode
        ..
    })?;

    // MLP: fused_silu_mul returns None on TPU, but XLA auto-fuses the fallback
    let gate = ops.gemm.matmul(&h, &self.gate_proj)?;
    let up = ops.gemm.matmul(&h, &self.up_proj)?;
    let h = match ops.fused.fused_silu_mul(&gate, &up) {
        None => (ops.act.silu(&gate)? * &up)?,  // XLA fuses silu + mul automatically
        Some(r) => r?,
    };
    ops.gemm.matmul(&h, &self.down_proj)
}
```

Key points:
- **Same model code** as CUDA. `TpuOps` handles shape padding and XLA compilation internally.
- **Paged attention works on TPU** via Pallas `ragged_paged_attention`. Page size auto-computed (16 for long sequences, up to 256 for short).
- **FusedOps all return `None`** — intentionally. XLA's HLO optimizer fuses element-wise chains (silu+mul, add+rmsnorm) automatically during compilation. The model's fallback path produces the same fused kernel.
- **OpsSession maps to XLA compilation**: `begin_forward()` starts HLO tracing, `end_forward()` compiles and executes. Second call hits cache — near-zero overhead.
- **Static shapes**: `TpuOps` pads batch/seq to nearest power-of-2 internally. Model code doesn't know.
- **BF16 native**: TPU MXU is BF16. `matmul` on TPU is always BF16 accumulation → fastest path.

### Example 10: Qwen3-4B on MI300X (ROCm, FP8 FNUZ)

LLM inference on AMD MI300X with FP8 quantization. ROCm uses HIP flash attention
and hipBLAS with architecture-specific FP8 format.

```rust
// Same model code. RocmOps auto-selects FP8 FNUZ for gfx942.

fn forward(x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    let h = ops.norm.rms_norm(x, &self.ln, eps)?;     // HIP fused kernel

    // FP8 quantized matmul — RocmOps detects gfx942, uses FNUZ format
    let (h_fp8, act_scale) = quantize_per_token_fp8(h)?;
    let qkv = ops.gemm.quantized_matmul(
        &h_fp8, &self.qkv_fp8,
        Some(&act_scale), Some(&self.qkv_scale),
        QuantScheme::Fp8E4m3,  // RocmOps internally maps to FNUZ on gfx942
    )?;

    let (q, k, v) = split_qkv(&qkv, 32, 8);
    let (q, k) = apply_rope(&q, &k, &cos, &sin)?;
    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;

    // Paged attention — aiter flash attention on gfx942
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
    // ...
}
```

Key points:
- **Same model code** as CUDA FP8 example. `QuantScheme::Fp8E4m3` is device-agnostic.
- **FP8 FNUZ auto-selected**: `RocmOps` detects gfx942, maps E4M3 request to FNUZ format internally. On gfx950, it would use native E4M3. Model doesn't know.
- **aiter flash attention**: specialized for MI300/MI350. Falls back to CK on older AMD hardware.
- **HIP graphs**: supported on ROCm 6.1+. `OpsSession` manages HIP graph capture/replay same as CUDA graphs.

### Example 11: LLaDA2 (Diffusion LLM, Bidirectional Demasking)

Diffusion LLM: generates text by iteratively replacing [MASK] tokens with predicted tokens.
**Not autoregressive** — uses bidirectional attention (all tokens see all tokens).
The model architecture is a standard transformer, but with `MaskType::Bidirectional`.

```rust
// Model forward: identical structure to a standard LLM, just bidirectional attention.
impl LLaDA2Layer {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        let (residual, h) = blocks::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

        let qkv = blocks::col_parallel(&h, &self.qkv_shard, false, ops)?;
        let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

        // Bidirectional attention — every token attends to every token (no causal mask)
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask: MaskType::Bidirectional,   // ← only difference from AR LLM
            ..
        })?;

        let h = blocks::row_parallel(&o, &self.o_proj_shard, ops)?;
        let (residual, h) = blocks::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = blocks::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
        Ok((&residual + &h)?)
    }
}
```

The **denoising loop** is engine-level (not model-level):

```rust
// Engine's DLLM decode loop
fn dllm_generate(model: &LLaDA2, ops: &Ops, prompt_ids: &[u32], block_size: usize) -> Vec<u32> {
    // Start with prompt + block_size MASK tokens
    let mut ids = [prompt_ids, &vec![MASK_ID; block_size]].concat();

    // Iterative denoising: up to block_size iterations per block
    for _ in 0..block_size {
        // Full forward pass (bidirectional attention over all tokens)
        ops.session.begin_forward();
        let logits = model.forward(&embed(&ids), ops)?;  // [total_len, vocab]
        ops.session.end_forward();

        // Confidence thresholding: replace high-confidence masks
        let probs = ops.act.softmax(&logits, /*dim=*/-1)?;  // softmax over vocab
        let (max_probs, predicted) = probs.max(/*dim=*/-1)?;

        // Replace MASK positions where confidence > threshold
        for pos in mask_positions(&ids) {
            if max_probs[pos] > 0.95 {
                ids[pos] = predicted[pos];
            }
        }

        if !ids.contains(&MASK_ID) { break; }  // all masks replaced
    }
    ids[prompt_ids.len()..].to_vec()
}
```

Key points:
- **Same building blocks as Qwen3**: `residual_norm`, `col_parallel`, `row_parallel`, `moe_layer`. All kernel optimizations on these benefit LLaDA2 too.
- **Only attention mask differs**: `MaskType::Bidirectional` instead of `Causal`. The kernel is the same FlashAttention with `causal=false`.
- **No KV cache for generation** (recompute every iteration). Can optionally cache across denoising iterations for speed.
- **No paged attention**: uses `varlen_attention` since there's no incremental decode.
- **Engine owns the denoising loop**: model.forward() is called block_size times. Scheduler controls when to stop.

### Example 12: Flux Full Pipeline (Denoising Loop + CFG + VAE Decode)

Complete image generation pipeline. Shows engine-level orchestration and how different
sub-models (text encoder, DiT, VAE) each get their own `Ops`.

```rust
// Engine's Flux pipeline — NOT model code
fn flux_generate(
    prompt: &str,
    dit: &FluxDiT,                // DiT transformer (Example 2)
    text_encoder: &T5Encoder,     // text encoder (bidirectional attention)
    vae: &AutoencoderKL,          // VAE decoder (conv2d + group_norm)
    ops: &Ops,
    num_steps: usize,
    guidance_scale: f32,
) -> Result<Image> {
    // ── Stage 1: Text encoding (single forward pass) ────────────
    let text_embeds = text_encoder.forward(&tokenize(prompt), ops)?;
    let pooled = text_encoder.pool(&text_embeds)?;

    // ── Stage 2: Denoising loop (num_steps iterations) ──────────
    let mut latents = Tensor::randn(&[1, 16, h/8, w/8])?;  // random noise
    let timesteps = flow_match_schedule(num_steps);           // e.g., [1.0, 0.95, ..., 0.0]

    for i in 0..num_steps {
        let t = timesteps[i];
        let dt = timesteps[i] - timesteps[i + 1];
        let temb = timestep_embed(t)?;                        // sinusoidal → MLP

        ops.session.begin_forward();

        // Classifier-Free Guidance: 2x batch (conditional + unconditional)
        let latents_cfg = cat(&[&latents, &latents], 0)?;    // [2, 16, h/8, w/8]
        let text_cfg = cat(&[&text_embeds, &null_embeds], 0)?;

        // DiT forward (same FluxDoubleBlock as Example 2, processes 2x batch)
        let noise_pred = dit.forward(&latents_cfg, &text_cfg, &pooled, &temb, ops)?;

        ops.session.end_forward();

        // CFG: guided = uncond + scale * (cond - uncond)
        let (cond_pred, uncond_pred) = noise_pred.chunk(2, 0)?;
        let guided = (&uncond_pred + guidance_scale * &(&cond_pred - &uncond_pred)?)?;

        // Euler step: latents = latents + dt * guided
        latents = (&latents + dt * &guided)?;
    }

    // ── Stage 3: VAE decode (latent → RGB image) ────────────────
    let image = vae.decode(&latents, ops)?;
    Ok(image)
}

// VAE decoder: conv2d + group_norm + silu pipeline
impl AutoencoderKL {
    fn decode(&self, latents: &Tensor, ops: &Ops) -> Result<Tensor> {
        let mut h = ops.gemm.matmul(latents, &self.post_quant_conv)?;

        // ResNet blocks + upsampling
        for block in &self.decoder_blocks {
            // ResNet: group_norm → silu → conv2d → group_norm → silu → conv2d + residual
            let residual = h.clone();
            h = ops.norm.group_norm(&h, &block.norm1, Some(&block.bias1), 32, eps)?;
            h = ops.act.silu(&h)?;
            h = ops.conv.conv2d(&h, &block.conv1, Some(&block.conv1_bias), [1,1], [1,1])?;
            h = ops.norm.group_norm(&h, &block.norm2, Some(&block.bias2), 32, eps)?;
            h = ops.act.silu(&h)?;
            h = ops.conv.conv2d(&h, &block.conv2, Some(&block.conv2_bias), [1,1], [1,1])?;
            h = (&h + &residual)?;

            // Upsample (nearest + conv2d)
            if let Some(up) = &block.upsample {
                h = nearest_upsample_2x(&h)?;
                h = ops.conv.conv2d(&h, &up.conv, Some(&up.bias), [1,1], [1,1])?;
            }
        }

        // Final norm + conv
        h = ops.norm.group_norm(&h, &self.final_norm, Some(&self.final_bias), 32, eps)?;
        h = ops.act.silu(&h)?;
        ops.conv.conv2d(&h, &self.final_conv, Some(&self.final_conv_bias), [1,1], [1,1])
    }
}
```

Key points:
- **Three sub-models, same `Ops`**: text encoder (bidirectional attention), DiT (AdaLN + joint attention), VAE (conv2d + group_norm). All share `Ops`.
- **CFG is 2x batch**: conditional + unconditional latents batched together. DiT processes both in one forward pass. Engine splits output after.
- **Denoising loop is engine-level**: scheduler controls timesteps, model.forward() called num_steps times.
- **VAE decoder uses `ConvOps` + `NormOps` only**: no attention. Pure conv2d + group_norm + silu pipeline. Kernel optimizations on `conv2d` and `group_norm` benefit all diffusion VAE decoders.
- **Building blocks inside DiT**: `blocks::adaln_zero`, `blocks::gelu_mlp` — same as Example 2.

### Example 13: Multi-LoRA Serving (Llama-3-8B, 50 Concurrent Adapters)

Multi-tenant serving: each request uses a different LoRA adapter. A single batch contains
tokens from 50 different adapters. Uses `blocks::lora_linear` for fused BGMV dispatch.

```rust
impl LoRALlamaLayer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx, adapter_ids: &Tensor) -> Result<Tensor> {
        let (residual, h) = blocks::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

        // LoRA-augmented QKV: building block handles fused_lora_matmul internally
        let qkv = blocks::lora_linear(
            &h, &self.qkv_weight, &self.qkv_lora_a, &self.qkv_lora_b,
            adapter_ids, self.lora_scale, ops,
        )?;
        let (q, k, v) = split_qkv(&qkv, 32, 8);

        // Attention is unchanged — LoRA only affects linear layers
        let (q, k) = blocks::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;

        // LoRA-augmented output projection
        let h = blocks::lora_linear(
            &o, &self.o_proj_weight, &self.o_proj_lora_a, &self.o_proj_lora_b,
            adapter_ids, self.lora_scale, ops,
        )?;

        let (residual, h) = blocks::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = blocks::gated_mlp(&h, &self.gate, &self.up, &self.down, ops)?;
        Ok((&residual + &h)?)
    }
}
```

Key points:
- `adapter_ids: [batch]` maps each token to its LoRA adapter (-1 = no adapter).
- On CUDA: `fused_lora_matmul` uses BGMV/Punica kernel — O(1) kernel launch for all 50 adapters.
- On CPU/Metal: fallback splits batch by adapter, runs separate matmuls. Correct but slower.
- **Attention is identical to non-LoRA** — LoRA only wraps linear layers.

### Example 14: Qwen3.5 Hybrid (DeltaNet + Softmax, Per-Layer Dispatch)

Qwen3.5 uses DeltaNet (linear attention) for ~75% of layers and softmax attention for ~25%.
DeltaNet is model-owned (closure injection), softmax goes through `AttentionOps`.

```rust
impl Qwen35Model {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let mut h = self.embed(x)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = match self.layer_type(i) {
                LayerType::Softmax => {
                    // Standard attention via building blocks (same as Qwen3)
                    layer.forward_softmax(&h, ops, kv)?
                }
                LayerType::DeltaNet => {
                    // DeltaNet: model-owned, uses conv1d + recurrent state
                    // Still uses ops.norm and ops.gemm — only the mixer is different
                    let (residual, h) = blocks::residual_norm(&h, &layer.residual, &layer.ln1, eps, ops)?;

                    // Conv1d causal scan (DeltaNet-specific, not in AttentionOps)
                    let h = self.deltanet[i].causal_conv(&h, &self.conv_states[i])?;
                    // Recurrent state update (not expressible as attention)
                    let h = self.deltanet[i].recurrent_step(&h, &self.recurrent_states[i])?;

                    let h = ops.gemm.matmul(&h, &layer.o_proj)?;  // still uses GemmOps
                    let (residual, h) = blocks::residual_norm(&residual, &h, &layer.ln2, eps, ops)?;
                    let h = blocks::gated_mlp(&h, &layer.gate, &layer.up, &layer.down, ops)?;
                    (&residual + &h)?
                }
            };
        }
        Ok(h)
    }
}
```

Key points:
- **DeltaNet doesn't go through `AttentionOps`** — it has fundamentally different state (conv + recurrent).
- But DeltaNet layers **still use building blocks** for everything else: `residual_norm`, `gated_mlp`, `ops.gemm.matmul`. Kernel optimizations on these benefit DeltaNet layers too.
- Only the token mixer differs per layer. MLP, norms, projections are identical.

### Example 15: DeepSeek-V3 with EP (Expert Parallelism, 256 Experts on 8 GPUs)

MoE model with 256 routed experts distributed across 8 EP ranks (32 experts each).
Uses `blocks::moe_layer` which handles EP dispatch/combine internally.

```rust
impl DeepSeekV3Layer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // Attention (same as any other LLM)
        let (residual, h) = blocks::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;
        let (q, k, v) = self.qkv_mla(&h, ops)?;  // MLA: compressed KV, head_dim_v != head_dim_q
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
        let h = blocks::row_parallel(&o, &self.o_proj_shard, ops)?;

        // MoE with EP: building block handles dispatch → grouped GEMM → combine
        let (residual, h) = blocks::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = blocks::moe_layer(&h, &self.gate, &self.expert_weights, &EpConfig {
            ep_size: 8,
            ep_rank: self.ep_rank,
            num_local_experts: 32,
        }, ops)?;
        Ok((&residual + &h)?)
    }
}
```

`blocks::moe_layer` internally:
```rust
pub fn moe_layer(x: &Tensor, gate: &MoeGate, weights: &Tensor, ep: &EpConfig, ops: &Ops) -> Result<Tensor> {
    let (topk_ids, topk_weights) = gate.route(x)?;
    if ep.ep_size > 1 {
        // Phase 1: all-to-all dispatch tokens to expert owners
        let (recv, meta) = ep_dispatch(x, &topk_ids, ep, ops)?;
        // Phase 2: local grouped GEMM on owned experts
        let out = ops.gemm.grouped_gemm(&recv, weights, &meta.sorted_ids, ..)?;
        // Phase 3: all-to-all combine results back
        ep_combine(&out, &meta, &topk_weights, ops)
    } else {
        ops.gemm.grouped_gemm(x, weights, ..)
    }
}
```

Key points:
- Model code is clean — `blocks::moe_layer` hides all EP complexity.
- Same model code works for EP=1 (single GPU) and EP=8 (8 GPUs).
- `CommOps::all_to_all` used inside `ep_dispatch`/`ep_combine`.

### Example 16: Speculative Decoding (EAGLE Draft + Target Verify)

Engine-level orchestration. Both draft and target models use the same `Ops`.

```rust
// Engine's speculative decode loop (NOT model code)
fn speculative_step(
    draft_model: &Model, target_model: &Model,
    ops: &Ops, kv: &PagedKvCtx,
) -> Result<Vec<Token>> {
    // 1. Draft: generate N candidates autoregressively
    let mut draft_tokens = Vec::new();
    for _ in 0..N {
        let logits = draft_model.forward(&draft_input, ops, &draft_kv)?;
        let token = sample(&logits);
        draft_tokens.push(token);
    }

    // 2. Build tree mask for verification
    let tree_mask = build_tree_mask(&draft_tokens);  // [N+1, max_kv_len]

    // 3. Target: verify all candidates in one forward pass
    let target_params = PagedParams {
        max_seqlen_q: N + 1,                    // all draft tokens + 1 bonus
        mask: MaskType::Custom(tree_mask),       // tree attention mask
        ..
    };
    let target_logits = target_model.forward(&all_candidates, ops, &target_kv)?;

    // 4. Rejection sampling: accept k ≤ N tokens
    let accepted = rejection_sample(&draft_logits, &target_logits);

    // 5. KV cache: rejected slots already have -1 in slot_mapping → skipped by reshape_and_cache
    Ok(accepted)
}
```

Key points:
- **Both models use the same `Ops`** — draft and target share the same attention/GEMM kernels.
- **Tree attention** via `MaskType::Custom(Tensor)` — passed as additive bias to attention kernel.
- **No KV cache rollback needed** — rejected tokens have `slot_mapping = -1`, `reshape_and_cache` skips them.
- **Engine-only concern** — model code is unchanged.

### Example 17: DeepSeek-V3 MLA (Multi-Head Latent Attention, Asymmetric Head Dims)

MLA compresses KV into low-rank latents: `head_dim_q = 192` but `head_dim_kv = 128`.
The implementation derives this from tensor shapes — no special params needed.

```rust
impl DeepSeekMLA {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let (residual, h) = blocks::residual_norm(x, &self.residual, &self.ln, eps, ops)?;

        // Q projection: full dimension (192)
        let q = ops.gemm.matmul(&h, &self.q_proj)?;        // [total, nq_heads, 192]
        // KV projection: compressed latent dimension (128)
        let kv_compressed = ops.gemm.matmul(&h, &self.kv_proj)?;  // [total, nkv_heads, 128]
        let k = ops.gemm.matmul(&kv_compressed, &self.k_up)?;     // [total, nkv_heads, 128]
        let v = ops.gemm.matmul(&kv_compressed, &self.v_up)?;     // [total, nkv_heads, 128]

        // RoPE on partial dims: first 64 dims of Q get RoPE, rest are position-independent
        let (q_rope, q_nope) = q.split_at(/*dim=*/-1, 64)?;
        let q_rope = apply_rope(&q_rope, cos, sin)?;
        let q = cat(&[&q_rope, &q_nope], -1)?;              // [total, nq_heads, 192]
        let k = apply_rope(&k, cos, sin)?;                   // [total, nkv_heads, 128]

        // Attention: Q [_, _, 192] × K [_, _, 128] → head_dim asymmetry
        // Implementation inspects tensor shapes to select correct kernel variant
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
            mask: MaskType::Causal,
            ..
        })?;  // output: [total, nq_heads, 128] (head_dim_v, not head_dim_q)

        let h = ops.gemm.matmul(&o, &self.o_proj)?;
        blocks::row_parallel(&h, &self.o_proj_shard, ops)
    }
}
```

Key points:
- **head_dim_q (192) ≠ head_dim_kv (128)**. No special parameter — the attention impl reads `Q.shape[-1]` and `K.shape[-1]` to select the right kernel.
- **Output dim = head_dim_v (128)**, not head_dim_q. The attention kernel handles this internally.
- **Partial RoPE**: only first 64 dims of Q/K get rotary embedding. Rest is position-independent (absorbed latent). This is model-level, not ops-level.
- KV cache stores compressed 128-dim K/V, saving ~33% memory vs full 192-dim.

### Example 18: Whisper (Encoder-Decoder, Cross-Attention)

Speech recognition with separate audio encoder and text decoder.
Decoder uses cross-attention: Q from decoder, K/V from encoder. Same `varlen_attention`.

```rust
// Stage 1: Audio encoder (bidirectional, no KV cache)
impl WhisperEncoder {
    fn forward(&self, mel: &Tensor, ops: &Ops) -> Result<Tensor> {
        // Conv1d feature extraction
        let h = ops.conv.conv1d(mel, &self.conv1, Some(&self.conv1_bias), 1, 1)?;
        let h = ops.act.gelu(&h)?;
        let h = ops.conv.conv1d(&h, &self.conv2, Some(&self.conv2_bias), 2, 1)?;
        let h = ops.act.gelu(&h)?;

        // Transformer encoder with bidirectional attention
        for layer in &self.layers {
            let (residual, h_norm) = blocks::residual_layer_norm(&h, &h, &layer.ln1, None, eps, ops)?;
            let (q, k, v) = layer.qkv_proj(&h_norm, ops)?;
            let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
                mask: MaskType::Bidirectional,  // encoder: full attention
                ..
            })?;
            h = (&residual + &ops.gemm.matmul(&o, &layer.o_proj)?)?;
            // MLP
            let (residual, h_norm) = blocks::residual_layer_norm(&h, &h, &layer.ln2, None, eps, ops)?;
            h = (&residual + &layer.mlp(&h_norm, ops)?)?;
        }
        Ok(h)
    }
}

// Stage 2: Text decoder (causal self-attention + cross-attention to encoder)
impl WhisperDecoderLayer {
    fn forward(
        &self, x: &Tensor, encoder_out: &Tensor, ops: &Ops,
        kv: &PagedKvCtx,
        encoder_seqlens: &Tensor,  // cu_seqlens for encoder output
    ) -> Result<Tensor> {
        // Self-attention (causal, with KV cache)
        let (residual, h) = blocks::residual_layer_norm(x, x, &self.ln1, None, eps, ops)?;
        let (q, k, v) = self.self_attn_qkv(&h, ops)?;
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
            mask: MaskType::Causal,
            ..
        })?;
        let h = (&residual + &ops.gemm.matmul(&o, &self.self_o_proj)?)?;

        // Cross-attention: Q from decoder, K/V from encoder
        let (residual, h_norm) = blocks::residual_layer_norm(&h, &h, &self.ln2, None, eps, ops)?;
        let q = ops.gemm.matmul(&h_norm, &self.cross_q_proj)?;       // [dec_total, heads, hdim]
        let k = ops.gemm.matmul(encoder_out, &self.cross_k_proj)?;   // [enc_total, heads, hdim]
        let v = ops.gemm.matmul(encoder_out, &self.cross_v_proj)?;   // [enc_total, heads, hdim]
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: kv.cu_seqlens_q.clone(),  // decoder sequence lengths
            cu_seqlens_k: encoder_seqlens.clone(),    // encoder sequence lengths (different!)
            mask: MaskType::Bidirectional,            // cross-attention: no causal mask
            ..
        })?;
        let h = (&residual + &ops.gemm.matmul(&o, &self.cross_o_proj)?)?;

        // MLP
        let (residual, h_norm) = blocks::residual_layer_norm(&h, &h, &self.ln3, None, eps, ops)?;
        Ok((&residual + &self.mlp(&h_norm, ops)?)?)
    }
}
```

Key points:
- **Cross-attention is just `varlen_attention`** with different `cu_seqlens_q` (decoder) and `cu_seqlens_k` (encoder). No special method needed.
- Decoder has **both** self-attention (causal, paged KV cache) and cross-attention (bidirectional, contiguous K/V). Two different attention calls in one layer.
- Encoder K/V are computed once and reused across all decoder layers/steps. Model caches them.
- `layer_norm` (not `rms_norm`): Whisper uses pre-LN with LayerNorm. Same `NormOps`.

### Example 19: HunyuanVideo (Video Diffusion, Temporal + Spatial Attention)

Video generation DiT. Each block has two attention calls: spatial (within-frame) and
temporal (across-frame at same position). Both use `varlen_attention` with different cu_seqlens.

```rust
impl HunyuanVideoBlock {
    fn forward(
        &self, x: &Tensor, ops: &Ops,
        temb: &Tensor,              // timestep embedding
        num_frames: usize,          // T
        spatial_tokens: usize,      // H*W per frame
    ) -> Result<Tensor> {
        // x shape: [batch * T * H*W, hidden]  (packed varlen)
        let total = num_frames * spatial_tokens;

        // ── Spatial attention: each frame attends within itself ──
        let (norm_x, gate_s) = blocks::adaln_zero(x, &self.ln_s, None, &s1, &h1, &g1, eps, ops)?;
        let (q, k, v) = self.spatial_qkv(&norm_x, ops)?;

        // cu_seqlens: [0, H*W, 2*H*W, ..., T*H*W] — one "sequence" per frame
        let spatial_seqlens = (0..=num_frames).map(|i| i * spatial_tokens).collect();
        let o_spatial = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: Tensor::from_slice(&spatial_seqlens),
            cu_seqlens_k: Tensor::from_slice(&spatial_seqlens),
            max_seqlen_q: spatial_tokens,   // H*W
            max_seqlen_k: spatial_tokens,
            mask: MaskType::Bidirectional,
            ..
        })?;
        let x = (x + &(ops.gemm.matmul(&o_spatial, &self.s_out)? * &gate_s)?)?;

        // ── Temporal attention: same spatial position attends across frames ──
        let (norm_x, gate_t) = blocks::adaln_zero(&x, &self.ln_t, None, &s2, &h2, &g2, eps, ops)?;

        // Reshape: [batch, T, H*W, hidden] → transpose → [batch, H*W, T, hidden] → pack
        let x_temporal = rearrange_spatial_to_temporal(&norm_x, num_frames, spatial_tokens)?;
        let (q, k, v) = self.temporal_qkv(&x_temporal, ops)?;

        // cu_seqlens: [0, T, 2*T, ..., H*W*T] — one "sequence" per spatial position
        let temporal_seqlens = (0..=spatial_tokens).map(|i| i * num_frames).collect();
        let o_temporal = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: Tensor::from_slice(&temporal_seqlens),
            cu_seqlens_k: Tensor::from_slice(&temporal_seqlens),
            max_seqlen_q: num_frames,       // T
            max_seqlen_k: num_frames,
            mask: MaskType::Bidirectional,   // no causal across frames
            ..
        })?;

        // Transpose back and residual
        let o_temporal = rearrange_temporal_to_spatial(&o_temporal, num_frames, spatial_tokens)?;
        let x = (&x + &(ops.gemm.matmul(&o_temporal, &self.t_out)? * &gate_t)?)?;

        // MLP
        let (norm_x, gate_m) = blocks::adaln_zero(&x, &self.ln_m, None, &s3, &h3, &g3, eps, ops)?;
        let x = (&x + &(blocks::gelu_mlp(&norm_x, &self.fc1, &self.fc2, ops)? * &gate_m)?)?;
        Ok(x)
    }
}
```

Key points:
- **Spatial + temporal = two `varlen_attention` calls** with different `cu_seqlens`. No special "video attention" trait.
- Spatial: `cu_seqlens = [0, H*W, 2*H*W, ...]` — each frame is a separate sequence.
- Temporal: transpose tokens so frames become the sequence dim, then `cu_seqlens = [0, T, 2*T, ...]`.
- **Same building blocks as image diffusion** (`blocks::adaln_zero`, `blocks::gelu_mlp`). Kernel optimizations transfer.
- **Same `AttentionOps`** as LLM. Just different sequence packing via `cu_seqlens`.

### Example 20: Mistral + Gemma Variants (Sliding Window, Softcap)

Shows how `MaskType` and `VarlenParams` parameters cover model-specific attention patterns.
No new ops needed — just different parameter values.

```rust
// Mistral: sliding window attention (attend only to last 4096 tokens)
impl MistralLayer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // ... standard norm, QKV, RoPE ...
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
            mask: MaskType::SlidingWindow { left: 4096, right: 0 },  // causal + window
            ..
        })?;
        // ... standard output proj, MLP ...
    }
}

// Gemma2: softcap (logit capping at 30.0)
impl Gemma2Layer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // ... standard norm, QKV, RoPE ...

        // Alternating attention: global (even layers) + sliding window (odd layers)
        let mask = if self.layer_idx % 2 == 0 {
            MaskType::Causal
        } else {
            MaskType::SlidingWindow { left: 4096, right: 0 }
        };

        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask,
            softcap: Some(30.0),  // Gemma2 attention logit capping
            ..
        })?;
        // ...
    }
}

// Gemma3: softcap 50.0 + bidirectional prefix (for prompt caching)
impl Gemma3Layer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // ...
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
            softcap: Some(50.0),
            mask: MaskType::SlidingWindow { left: 1024, right: 0 },
            ..
        })?;
        // ...
    }
}
```

Key points:
- **Sliding window** is just a parameter: `MaskType::SlidingWindow { left: N, right: 0 }`.
- **Softcap** is just a parameter: `VarlenParams { softcap: Some(30.0) }`. FA4 and FlashInfer both support it natively.
- **Per-layer alternating attention** (Gemma2) is model logic, not ops logic.
- **No building block needed** for these — they're just different `VarlenParams` / `PagedParams` values.

### Example 21: BGE / GTE (Embedding Model, Encoder-Only for Retrieval)

BERT-like encoder-only model for text embedding / retrieval. Simplest use of the design:
bidirectional attention, no KV cache, no generation, no decoder.

```rust
impl BgeLayer {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        let (residual, h) = blocks::residual_layer_norm(x, x, &self.ln1, Some(&self.ln1_bias), eps, ops)?;
        let (q, k, v) = self.qkv_proj(&h, ops)?;
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask: MaskType::Bidirectional,
            ..
        })?;
        let h = (&residual + &ops.gemm.matmul(&o, &self.o_proj)?)?;
        let (residual, h) = blocks::residual_layer_norm(&h, &h, &self.ln2, Some(&self.ln2_bias), eps, ops)?;
        Ok((&residual + &blocks::gelu_mlp(&h, &self.fc1, &self.fc2, ops)?)?)
    }
}

// Usage: encode once, pool, return embedding
fn embed(text: &str, model: &BgeModel, ops: &Ops) -> Result<Tensor> {
    let ids = tokenize(text);
    ops.session.begin_forward();
    let hidden = model.forward(&ids, ops)?;
    ops.session.end_forward();
    mean_pool(&hidden)  // average over tokens → single vector
}
```

Key points:
- **Minimal use of the design**: `varlen_attention` + `layer_norm` + `gelu_mlp`. No KV cache, no paged, no fusion tricks.
- Same building blocks as everything else — `residual_layer_norm`, `gelu_mlp`.
- `layer_norm` with bias (BERT-style), not `rms_norm` (LLM-style). Both in `NormOps`.
- Runs on any device including CPU (no paged attention needed).

### Example 22: CUDA Graph Capture/Replay (Engine-Level)

Shows how the engine uses `OpsSession` + CUDA-specific graph capture.
This is engine code, not model code.

```rust
// Engine's CUDA graph runner (knows it's on CUDA)
fn setup_cuda_graph(
    model: &dyn Model,
    ops: &Ops,
    cuda_ops: &CudaOps,       // downcast from ops.session
    max_batch: usize,
) -> CudaGraph {
    // 1. Allocate fixed-address buffers
    let graph_bufs = cuda_ops.allocate_graph_buffers(max_batch, max_blocks_per_seq);
    let input_buf = cuda_ops.allocate_fixed_tensor([max_batch, hidden_dim]);

    // 2. Precompute plan with graph buffers (outside capture)
    ops.session.begin_forward();
    cuda_ops.precompute_paged_plan_graphed(&block_tables, &cu_seqlens_k, block_size, &graph_bufs);

    // 3. Capture
    let stream = cuda_ops.stream();
    stream.begin_capture();
    model.forward(&input_buf, ops, &kv_ctx);    // all kernel launches captured
    let graph = stream.end_capture();
    ops.session.end_forward();

    CudaGraph { graph, graph_bufs, input_buf }
}

fn replay_cuda_graph(
    graph: &CudaGraph,
    ops: &Ops,
    cuda_ops: &CudaOps,
    batch_input: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_k: &Tensor,
) -> Result<Tensor> {
    // Update fixed-address buffers (memcpy, no reallocation)
    graph.input_buf.copy_from(batch_input);
    ops.session.begin_forward();
    cuda_ops.precompute_paged_plan_graphed(block_tables, cu_seqlens_k, block_size, &graph.graph_bufs);
    graph.graph.launch();
    ops.session.end_forward();
    Ok(graph.output_buf.clone())
}
```

Key points:
- **Engine knows it's on CUDA** — downcasts to `CudaOps` for graph-specific methods.
- **`OpsSession::begin_forward/end_forward`** is generic (all devices). Graph capture is CUDA-specific.
- **Model code is unaware of graphs** — same `model.forward()` call whether captured or not.
- **Fixed-address buffers**: `allocate_graph_buffers` and `allocate_fixed_tensor` return GPU tensors at stable addresses that survive graph replay.
- **precompute_paged_plan_graphed**: updates FlashInfer metadata in graph buffers (outside capture, before each replay).

### Example 23: Adding a New Fused Kernel (Developer Workflow, Always Keep Last)

Scenario: kernel dev adds `fused_geglu` (GELU-gated MLP fusion) to CudaOps.
Shows the minimal change set.

```rust
// Step 1: Add method to FusedOps trait with default { None } (1 line)
trait FusedOps {
    // ... existing methods ...
    fn fused_geglu(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> { None }
}
// ← RocmOps, MetalOps, VulkanOps, TpuOps, CpuOps: ZERO changes (inherit None)

// Step 2: Override in CudaOps (the only device that has the kernel)
impl FusedOps for CudaOps {
    fn fused_geglu(&self, gate, up) -> Option<Result<Tensor>> {
        Some(triton::fused_geglu(gate, up))
    }
}

// Step 3: Update the building block (1 function change)
pub fn gelu_mlp(x: &Tensor, fc1: &Tensor, fc2: &Tensor, ops: &Ops) -> Result<Tensor> {
    let hidden = ops.gemm.matmul(x, fc1)?;
    let (gate, up) = hidden.chunk(2, -1)?;
    let h = match ops.fused.fused_geglu(&gate, &up) {
        Some(r) => r?,
        None => (ops.act.gelu(&gate)? * &up)?,  // fallback: separate ops
    };
    ops.gemm.matmul(&h, fc2)
}
// ← All models using blocks::gelu_mlp (Flux, Sana, all diffusion models) benefit automatically
```

Total changes: **3 locations** (trait def, CudaOps, building block).
Models changed: **0**.
Models that benefit: **all models using `blocks::gelu_mlp`**.
