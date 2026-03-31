# Ops Dispatch Architecture

## Goals

1. Model code is device-agnostic: no `#[cfg(feature = "cuda")]` in models.
2. Model code is kernel-agnostic: models never reference FA4, FlashInfer, cuBLAS, etc.
3. Each operation independently dispatches to the best available kernel for the device + parameters.
4. Multi-device: CUDA, ROCm, TPU, Vulkan, CPU share the same model code.
5. Multi-model: AR (LLM), diffusion, TTS, vision all use the same op traits.
6. Fusion is explicit: models control fusion boundaries, not the dispatch layer.

## Layering

```
Model code          calls         Op traits
Op traits           implemented by    Device ops (CudaOps, RocmOps, CpuOps, ...)
Device ops          dispatches to     Kernel libraries (FA4, FlashInfer, CK, cuBLAS, XLA, ...)
```

Models only see op traits. Device ops are constructed once at engine init and injected into models.

## Op Traits

### Attention

```rust
trait AttentionOps: Send + Sync {
    /// Varlen attention over contiguous Q, K, V.
    ///
    /// Covers: LLM prefill (causal), diffusion self-attention (bidirectional),
    /// cross-attention (Q and K/V from different sources with different cu_seqlens),
    /// sliding window, softcap.
    ///
    /// Q: [total_q, num_heads_q, head_dim]
    /// K: [total_k, num_heads_k, head_dim_k]   (head_dim_k may differ from head_dim for MLA)
    /// V: [total_k, num_heads_k, head_dim_v]
    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor>;

    /// Paged attention: Q attends to K/V in block cache.
    ///
    /// Covers: LLM decode (Q=1), chunked prefill (mixed Q lengths),
    /// LLM prefill with prefix cache reuse (Q>1, K/V already in cache).
    ///
    /// Q: [total_q, num_heads_q, head_dim]
    /// key_cache/value_cache: [num_blocks, block_size, num_heads_k, head_dim]
    fn paged_attention(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        params: &PagedParams,
    ) -> Result<Tensor>;
}

struct VarlenParams {
    pub cu_seqlens_q: Tensor,     // [batch+1]
    pub cu_seqlens_k: Tensor,     // [batch+1], may differ from cu_seqlens_q (cross-attention, prefill+cache)
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,     // Gemma2/3 logit capping
}

struct PagedParams {
    pub block_tables: Tensor,     // [batch, max_blocks_per_seq]
    pub cu_seqlens_q: Tensor,     // [batch+1]
    pub cu_seqlens_k: Tensor,     // [batch+1]
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
}

enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
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
        slot_mapping: &Tensor,
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
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

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

### Fusion

Fused kernels are **separate ops that return `Option`**. `None` = not supported on this device, model falls back to separate ops. This is not a hint system — the model explicitly checks the return value.

```rust
trait FusedOps: Send + Sync {
    /// Fused residual add + RMSNorm: computes (x + h) and rms_norm(x + h) in one kernel.
    fn fused_add_rmsnorm(
        &self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>>;

    /// Fused SiLU(gate) * up.
    fn fused_silu_mul(
        &self, gate: &Tensor, up: &Tensor,
    ) -> Option<Result<Tensor>>;

    /// Fused QK-norm + RoPE: normalize Q and K, apply rotary embedding.
    fn fused_qknorm_rope(
        &self,
        q: &Tensor, k: &Tensor,
        q_weight: &Tensor, k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>>;

    /// Fused K-norm + RoPE + KV cache write.
    fn fused_knorm_rope_cache_write(
        &self,
        k: &Tensor, v: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor, eps: f32,
    ) -> Option<Result<()>>;
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
- **CUDA graphs**: pre-allocated buffers with fixed GPU addresses
- **XLA (TPU)**: compilation cache, trace-based execution

```rust
trait OpsSession: Send + Sync {
    /// Initialize per-forward-pass state. Called before model.forward().
    fn begin_forward(&self);

    /// Clear per-forward-pass state. Called after model.forward().
    fn end_forward(&self);

    /// Pre-compute paged attention scheduling for CUDA graph capture.
    /// Writes metadata to pre-allocated buffers (fixed GPU addresses).
    fn precompute_paged_plan(
        &self,
        block_tables: &Tensor,
        cu_seqlens_k: &Tensor,
        block_size: usize,
        graph_buffers: Option<&GraphMetaBuffers>,
    ) -> Result<()>;
}

/// Pre-allocated GPU tensors for CUDA graph metadata.
struct GraphMetaBuffers {
    pub indptr: Tensor,
    pub indices: Tensor,
    pub last_page_len: Tensor,
}
```

**CUDA graph flow:**

```
Capture:
    session.precompute_paged_plan(..., Some(&graph_buffers))   // outside capture
    stream.begin_capture()
    model.forward(...)                                          // captured
    stream.end_capture() → graph

Replay:
    update_input_buffers(...)                                   // memcpy to fixed addresses
    session.precompute_paged_plan(..., Some(&graph_buffers))   // update metadata
    graph.launch()
```

**Devices without session state** (CPU, Vulkan) implement `begin_forward` / `end_forward` as no-ops.

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

impl FusedOps for CudaOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(gpu::fused_add_rmsnorm(..)) }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { Some(gpu::fused_silu_mul(..)) }
    fn fused_qknorm_rope(&self, ..) -> Option<Result<..>> { Some(gpu::fused_qknorm_rope(..)) }
    fn fused_knorm_rope_cache_write(&self, ..) -> Option<Result<..>> { Some(gpu::fused_knorm_rope_kv_write(..)) }
}
```

### ROCm

```rust
struct RocmOps {
    ck_flash: CKFlashAttnHandle,
    rocblas: RocblasHandle,
}

impl AttentionOps for RocmOps {
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> {
        self.ck_flash.varlen(q, k, v, params)
    }
    fn paged_attention(&self, q, key_cache, value_cache, params) -> Result<Tensor> {
        self.ck_flash.paged(q, key_cache, value_cache, params)
    }
}

impl FusedOps for RocmOps {
    // ROCm has no fused qknorm_rope kernel yet
    fn fused_qknorm_rope(&self, ..) -> Option<Result<..>> { None }
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> { Some(hip::fused_add_rmsnorm(..)) }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { None }
    fn fused_knorm_rope_cache_write(&self, ..) -> Option<Result<..>> { None }
}
```

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

impl FusedOps for CpuOps {
    fn fused_add_rmsnorm(&self, ..) -> Option<Result<..>> {
        Some(cpu::fused_add_rmsnorm(..))  // CPU has a vectorized version
    }
    // All others: None (fallback to separate ops)
    fn fused_qknorm_rope(&self, ..) -> Option<Result<..>> { None }
    fn fused_silu_mul(&self, ..) -> Option<Result<..>> { None }
    fn fused_knorm_rope_cache_write(&self, ..) -> Option<Result<..>> { None }
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

## Model Code Pattern

Models receive `&Ops` and use it for all device-dependent operations:

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
                attn: cuda.clone(),
                kv_cache: cuda.clone(),
                gemm: cuda.clone(),
                norm: cuda.clone(),
                act: cuda.clone(),
                conv: cuda.clone(),
                fused: cuda.clone(),
                session: cuda,
            }
        }
        DeviceType::Rocm => { /* similar with RocmOps */ }
        DeviceType::Cpu => { /* CpuOps */ }
    }
}
```

All fields populated for every device. Methods on unsupported ops return errors (`bail!("paged attention not supported on {device}")`) rather than panicking or silently degrading.

## Summary

| Concern | Solution |
|---------|----------|
| FA4 can't decode | `CudaOps::paged_attention` routes Q=1 to FlashInfer |
| Fusion control | `FusedOps` trait, model checks `Option` return |
| Multi-device | Traits implemented per device, model code unchanged |
| Multi-model | Same `AttentionOps` for causal/bidirectional/cross-attention via `MaskType` + `VarlenParams` |
| DeltaNet/Mamba | Not `AttentionOps`; model-owned, closure-injected into TransformerBlock |
| FlashInfer plan cache | `OpsSession::begin_forward()` / `end_forward()` |
| CUDA graphs | `precompute_paged_plan(graph_buffers)` with pre-allocated fixed-address tensors |
| KV cache write timing | `KvCacheOps::reshape_and_cache` separate from `AttentionOps` |
| MLA head_dim asymmetry | Derived from tensor shapes, not params |
| Chunked prefill | `paged_attention` with `max_seqlen_q > 1`, varlen kernel handles mixed Q lengths |
