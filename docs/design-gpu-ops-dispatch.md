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
    ) -> Option<Result<(Tensor, Tensor)>>; // (normed_and_shifted, gate) or (gated_output, normed)

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
    ) -> Option<Result<Tensor>>;
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
    fn fused_adaln_zero(&self, ..) -> Option<Result<..>> { Some(gpu::fused_adaln_zero(..)) }
    fn fused_scale_shift(&self, ..) -> Option<Result<..>> { Some(gpu::fused_scale_shift(..)) }
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
    fn fused_adaln_zero(&self, ..) -> Option<Result<..>> { None }  // TODO: CK may add this
    fn fused_scale_shift(&self, ..) -> Option<Result<..>> { None }
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
    fn fused_adaln_zero(&self, ..) -> Option<Result<..>> { None }
    fn fused_scale_shift(&self, ..) -> Option<Result<..>> { None }
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
| AdaLN (diffusion) | `FusedOps::fused_adaln_zero` / `fused_scale_shift`, `None` fallback to separate ops |
| Quantized inference | `GemmOps::quantized_matmul` with `QuantScheme` dispatch |

## Model Examples

Concrete examples showing how real model architectures map onto this design.

### Example 1: Qwen3-32B (LLM with GQA + MoE)

Standard AR LLM. 64 layers, GQA (40 Q heads / 8 KV heads), hdim 128, MoE (8 active / 128 total experts).

```rust
struct Qwen3Layer {
    ln1: Tensor, ln2: Tensor,
    qkv_proj: Linear, o_proj: Linear,
    gate_proj: Linear, up_proj: Linear, down_proj: Linear,
    gate: MoeGate,       // router
    expert_weights: Tensor, // [128, N, K]
}

impl Qwen3Layer {
    fn forward(&self, x: &Tensor, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // 1. Pre-attention norm + attention
        let (residual, h) = match ops.fused.fused_add_rmsnorm(x, &self.residual, &self.ln1, eps) {
            Some(r) => r?,
            None => {
                let r = (x + &self.residual)?;
                let h = ops.norm.rms_norm(&r, &self.ln1, eps)?;
                (r, h)
            }
        };
        let qkv = ops.gemm.matmul(&h, &self.qkv_proj)?;
        let (q, k, v) = split_qkv(&qkv, 40, 8);
        let (q, k) = apply_rope(&q, &k, &kv.cos, &kv.sin, &kv.positions)?;
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
        let h = ops.gemm.matmul(&o, &self.o_proj)?;

        // 2. Post-attention norm + MoE
        let (residual, h) = match ops.fused.fused_add_rmsnorm(&residual, &h, &self.ln2, eps) {
            Some(r) => r?,
            None => {
                let r = (&residual + &h)?;
                let h = ops.norm.rms_norm(&r, &self.ln2, eps)?;
                (r, h)
            }
        };
        let (indices, weights) = self.gate.route(&h)?;          // top-8 routing
        let moe_out = ops.gemm.grouped_gemm(                    // per-expert GEMM
            &h, &self.expert_weights, &indices, &expert_ids, &num_tokens,
        )?;
        Ok((&residual + &moe_out)?)
    }
}
```

Key points:
- `paged_attention` handles both decode (Q=1) and chunked prefill (Q>1). Model doesn't distinguish.
- `grouped_gemm` for MoE. Same op on CUDA (DeepGEMM grouped) and ROCm (CK grouped GEMM).
- Fused ops (`fused_add_rmsnorm`) have explicit fallback. Runs correctly on CPU.

### Example 2: Flux (Diffusion Transformer, Text-to-Image)

DiT with joint text+image attention. 19 double-stream blocks + 38 single-stream blocks.
No KV cache, no paged attention, no causal masking.

```rust
struct FluxDoubleBlock {
    img_ln: Tensor, txt_ln: Tensor,
    img_qkv: Linear, txt_qkv: Linear,
    img_out: Linear, txt_out: Linear,
    img_mlp: MLP, txt_mlp: MLP,
}

impl FluxDoubleBlock {
    fn forward(
        &self, img: &Tensor, txt: &Tensor,
        temb: &Tensor,  // timestep embedding → (scale, shift, gate) per sub-layer
        ops: &Ops,
    ) -> Result<(Tensor, Tensor)> {
        // 1. AdaLN-Zero on image stream
        let (img_scale1, img_shift1, img_gate1, img_scale2, img_shift2, img_gate2) =
            self.img_mod.forward(temb)?;  // MLP: temb → 6 modulation params
        let (txt_scale1, txt_shift1, txt_gate1, txt_scale2, txt_shift2, txt_gate2) =
            self.txt_mod.forward(temb)?;

        // 2. Norm + modulate image
        let img_normed = match ops.fused.fused_adaln_zero(
            img, &self.img_ln, None, &img_scale1, &img_shift1, &img_gate1, eps,
        ) {
            Some(r) => r?,
            None => {
                let n = ops.norm.layer_norm(img, &self.img_ln, None, eps)?;
                let n = &n * &(1.0 + &img_scale1)? + &img_shift1;  // modulate
                (n, img_gate1.clone())
            }
        };

        // 3. Q/K/V projections, RMSNorm on Q/K
        let img_qkv = ops.gemm.matmul(&img_normed.0, &self.img_qkv)?;
        let txt_qkv = ops.gemm.matmul(&txt_normed.0, &self.txt_qkv)?;
        let (img_q, img_k, img_v) = split_qkv(&img_qkv, num_heads, num_heads);
        let (txt_q, txt_k, txt_v) = split_qkv(&txt_qkv, num_heads, num_heads);
        let img_q = ops.norm.rms_norm(&img_q, &self.img_q_norm, eps)?;
        let img_k = ops.norm.rms_norm(&img_k, &self.img_k_norm, eps)?;
        let txt_q = ops.norm.rms_norm(&txt_q, &self.txt_q_norm, eps)?;
        let txt_k = ops.norm.rms_norm(&txt_k, &self.txt_k_norm, eps)?;

        // 4. Joint attention: concat text + image, single attention call
        let q = cat(&[&txt_q, &img_q], /*seq_dim=*/0)?;
        let k = cat(&[&txt_k, &img_k], /*seq_dim=*/0)?;
        let v = cat(&[&txt_v, &img_v], /*seq_dim=*/0)?;
        let params = VarlenParams {
            cu_seqlens_q: /* single sequence */,
            cu_seqlens_k: /* same */,
            max_seqlen_q: txt_len + img_len,
            max_seqlen_k: txt_len + img_len,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask: MaskType::Bidirectional,  // no causal mask for diffusion
            softcap: None,
        };
        let attn_out = ops.attn.varlen_attention(&q, &k, &v, &params)?;

        // 5. Split output, apply gate, residual
        let (txt_attn, img_attn) = attn_out.split_at(txt_len)?;
        let img_attn = ops.gemm.matmul(&img_attn, &self.img_out)?;
        let img = (img + &(&img_attn * &img_normed.1)?)?;  // residual + gate

        // 6. MLP with AdaLN-Zero (second sub-layer)
        let img_mlp_in = match ops.fused.fused_adaln_zero(
            &img, &self.img_ln2, None, &img_scale2, &img_shift2, &img_gate2, eps,
        ) {
            Some(r) => r?,
            None => { /* fallback: layer_norm + scale + shift */ }
        };
        let img_mlp_out = self.img_mlp.forward(&img_mlp_in.0, ops)?;
        let img = (&img + &(&img_mlp_out * &img_mlp_in.1)?)?;

        // (txt stream is symmetric, omitted for brevity)
        Ok((img, txt))
    }
}
```

Key points:
- `fused_adaln_zero` is called 4x per double block (2 per stream × 2 sub-layers). 19 blocks = 76 fused kernel calls vs 304 element-wise ops without fusion.
- Joint attention: model concatenates text + image tokens, calls `varlen_attention` with `MaskType::Bidirectional`. No special "joint attention" trait needed.
- No KV cache, no `paged_attention`, no `KvCacheOps`. Diffusion is stateless.
- `layer_norm` (not `rms_norm`): diffusion uses LayerNorm, LLM uses RMSNorm. Both in `NormOps`.
- On CPU/Vulkan: `fused_adaln_zero` returns `None`, model falls back to separate ops. Correct but slower.

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
On ROCm: routes to CK FP8 GEMM or hipBLASLt FP8.
On CPU: `quantized_matmul` dequantizes and falls back to BLAS.
