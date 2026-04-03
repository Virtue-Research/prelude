## Model Examples

Concrete examples showing how real model architectures map onto this design.

### Example 1: Qwen3-32B (LLM with GQA + MoE, Modules)

Standard AR LLM. 64 layers, GQA (40 Q heads / 8 KV heads), hdim 128, MoE (8 active / 128 total experts).
Uses modules — no fusion fallback logic in model code.

```rust
// prelude-core/src/models/qwen3.rs

impl Qwen3Layer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // 1. Pre-attention: module handles fused_add_rmsnorm internally
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

        // 2. QKV projection (TP-aware)
        let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
        let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

        // 3. QK-norm + RoPE: module handles fused_qknorm_rope internally
        let (q, k) = modules::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

        // 4. KV cache + attention (raw ops — no module needed)
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
        let h = self.o_proj.forward(&o, ctx, ops)?;

        // 5. Post-attention norm + MoE
        let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = modules::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
        Ok((&residual + &h)?)
    }
}
```

Key points:
- **No `match` / `None` / fallback** in model code. All fusion logic is inside modules.
- `paged_attention` handles both decode (Q=1) and chunked prefill (Q>1). Model doesn't distinguish.
- `modules::moe_layer` handles EP dispatch/combine internally when `ep_config.ep_size > 1`.
- If a kernel dev adds `fused_add_rmsnorm` to CudaOps, this model benefits with zero changes.

### Example 2: Flux (Diffusion Transformer, Modules)

DiT with joint text+image attention. 19 double-stream blocks + 38 single-stream blocks.
No KV cache, no paged attention, no causal masking.

```rust
// prelude-core/src/models/flux.rs

impl FluxDoubleBlock {
    fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<(Tensor, Tensor)> {
        // Timestep modulation: MLP maps temb → 6 affine params per stream
        let (is1, ih1, ig1, is2, ih2, ig2) = self.img_mod.forward(temb)?;
        let (ts1, th1, tg1, ts2, th2, tg2) = self.txt_mod.forward(temb)?;

        // 1. AdaLN-Zero (module handles fused_adaln_zero internally)
        let (img_n, img_gate) = modules::adaln_zero(img, &self.img_ln, None, &is1, &ih1, &ig1, eps, ops)?;
        let (txt_n, txt_gate) = modules::adaln_zero(txt, &self.txt_ln, None, &ts1, &th1, &tg1, eps, ops)?;

        // 2. QKV via Linear + QK-norm
        let img_qkv = self.img_qkv_proj.forward(&img_n, ctx, ops)?;
        let (img_q, img_k, img_v) = modules::split_qkv(&img_qkv, num_heads, num_heads);
        let txt_qkv = self.txt_qkv_proj.forward(&txt_n, ctx, ops)?;
        let (txt_q, txt_k, txt_v) = modules::split_qkv(&txt_qkv, num_heads, num_heads);
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

        // 4. Output proj via Linear + gated residual
        let img = (img + &(self.img_out.forward(&img_attn, ctx, ops)? * &img_gate)?)?;
        let txt = (txt + &(self.txt_out.forward(&txt_attn, ctx, ops)? * &txt_gate)?)?;

        // 5. MLP sub-layer with AdaLN-Zero
        let (img_n2, img_gate2) = modules::adaln_zero(&img, &self.img_ln2, None, &is2, &ih2, &ig2, eps, ops)?;
        let img = (&img + &(modules::gelu_mlp(&img_n2, &self.img_fc1, &self.img_fc2, ctx, ops)? * &img_gate2)?)?;
        // (txt symmetric, omitted)

        Ok((img, txt))
    }
}
```

Key points:
- `modules::adaln_zero` called 4x per double block. 19 blocks = 76 calls. Module handles fused/fallback internally.
- Joint attention: concat text + image → `varlen_attention` with `MaskType::Bidirectional`. Same `AttentionOps` as LLM.
- No KV cache, no paged attention. Diffusion is stateless per denoising step.
- Same modules (`rms_norm`, `gelu_mlp`) as LLM — kernel optimizations on these benefit both.

### Example 3: Qwen3-TTS Code Predictor (Small Causal AR, No KV Cache)

5-layer dense transformer predicting residual codec codes. Re-prefills every step (no KV cache).
Very similar to a small LLM but uses `varlen_attention` instead of `paged_attention`.

```rust
// prelude-core/src/models/qwen3_tts.rs

struct CodePredictorLayer {
    ln1: Tensor, ln2: Tensor,
    q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear,
    q_norm: Tensor, k_norm: Tensor,
    gate_proj: Linear, up_proj: Linear, down_proj: Linear,
}

impl CodePredictorLayer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = ops.norm.rms_norm(x, &self.ln1, eps)?;

        // Separate Q/K/V projections via Linear (no fused QKV for numerical stability)
        let q = self.q_proj.forward(&h, ctx, ops)?;
        let k = self.k_proj.forward(&h, ctx, ops)?;
        let v = self.v_proj.forward(&h, ctx, ops)?;

        // QK norm + RoPE
        let q = ops.norm.rms_norm(&q, &self.q_norm, eps)?;
        let k = ops.norm.rms_norm(&k, &self.k_norm, eps)?;
        let (q, k) = modules::apply_rope(&q, &k, cos, sin)?;

        // Causal self-attention — no KV cache, re-prefill every AR step
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask: MaskType::Causal, ..
        })?;
        let o = self.o_proj.forward(&o, ctx, ops)?;

        // SiLU-gated MLP via module
        let (residual, h) = modules::residual_norm(&(x + &o)?, &Tensor::default(), &self.ln2, eps, ops)?;
        let h = modules::gated_mlp(&h, &self.gate_proj, &self.up_proj, &self.down_proj, ctx, ops)?;
        Ok((&residual + &h)?)
    }
}
```

Key points:
- All projections go through `Linear` — TP/quant/LoRA transparent even for this small model.
  See [modules/modules.md](modules/modules.md) Linear section.
- `varlen_attention` with `MaskType::Causal` — same kernel as LLM, dispatch picks FA4/FlashInfer
  based on shape. See [ops/op_traits.md](ops/op_traits.md) Attention section.
- `modules::gated_mlp` handles fused silu_mul internally.
  See [modules/modules.md](modules/modules.md) MLP section.
- No `paged_attention` or `KvCacheOps` — too small (5 layers) for KV cache benefit.

### Example 4: Qwen3-Omni Thinker (Multimodal LLM with Vision+Audio Encoders)

Standard AR LLM core, but receives pre-computed embeddings from vision and audio encoders.
Each encoder is a separate stage with its own `Ops`.

```rust
// prelude-core/src/models/qwen3_omni.rs

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
// prelude-core/src/models/qwen3_tts.rs

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
            // ConvTranspose1d upsampling
            h = ops.conv.conv_transpose1d(&h, &block.upsample_weight, None, block.stride, block.padding, 0)?;
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
- `conv_transpose1d` (upsampling) is in `ConvOps` — used here for learned upsampling between ConvNeXt blocks.

### Example 6: FP8 Quantized Inference (DeepSeek-V3 with FP8 Weights)

Using `quantized_matmul` for FP8 GEMM with per-token activation scaling.

```rust
// prelude-core/src/modules/linear.rs — shared by all models

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

On CUDA: routes to DeepGEMM FP8 (SM90+) or CUTLASS FP8.
On ROCm: routes to CK FP8 (gfx942 FNUZ or gfx950 E4M3 auto-selected).
On CPU: `quantized_matmul` dequantizes and falls back to BLAS.

### Example 7: Llama-3-8B on Apple M4 (Metal, Q4_K Quantized)

On-device inference with 4-bit quantized weights on Apple Silicon.
Uses Metal compute shaders (MSL) with simdgroup matrix multiply.

```rust
// prelude-core/src/modules/linear.rs — same code, only Ops differ per device

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

// prelude-core/src/models/llama.rs — identical to CUDA/ROCm version
fn forward(x: &Tensor, ops: &Ops) -> Result<Tensor> {
    let h = ops.norm.rms_norm(x, &self.ln, eps)?;
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;  // quantized matmul via Metal
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
- **Same model code** as CUDA. The only difference is which device crate is linked.
- **Unified memory**: no CPU→GPU transfers. Tensors allocated once, visible to both.
- **Q4_K quantized matmul**: Metal MSL shader dequantizes in-register during compute. No separate dequant kernel — more memory-efficient than CUDA's approach for small batch.
- **No paged attention**: Metal uses `varlen_attention` with contiguous KV. Acceptable for single-user on-device inference (no batching needed).
- **No fused_qknorm_rope**: `FusedOps` returns `None`, model falls back to separate `rms_norm` + `apply_rope`. Correct, just uses 2 Metal dispatches instead of 1.

### Example 8: Flux on Vulkan (Edge Diffusion, Cross-Vendor GPU)

Running Flux image generation on an Intel Arc or AMD RX GPU via Vulkan.
Same model code as CUDA Flux example — only device dispatch differs.

```rust
// prelude-core/src/models/flux.rs — same code as Example 2, runs on Vulkan
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
// prelude-core/src/models/llama.rs — SAME code as CUDA, TpuOps handles XLA internally

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;       // → XLA dot_general (MXU)
    let (q, k, v) = modules::split_qkv(&qkv, 64, 8);
    let (q, k) = modules::apply_rope(&q, &k, &cos, &sin)?;

    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
        mask: MaskType::Causal, max_seqlen_q: 1, ..
    })?;
    let h = self.o_proj.forward(&o, ctx, ops)?;

    let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
    let h = modules::gated_mlp(&h, &self.gate_proj, &self.up_proj, &self.down_proj, ctx, ops)?;
    Ok((&residual + &h)?)
}
```

Key points:
- **Identical model code to CUDA** — not one line differs. `Linear.forward` dispatches
  to XLA `dot_general` instead of DeepGEMM. See [device/device_impls.md](device/device_impls.md) TPU section.
- `modules::gated_mlp` handles fused_silu_mul fallback. On TPU, `FusedOps` returns `None`
  — XLA auto-fuses the separate ops during HLO compilation.
  See [ops/op_traits.md](ops/op_traits.md) Fusion section and [device/device_impls.md](device/device_impls.md) TPU section.
- **Paged attention on TPU** via Pallas `ragged_paged_attention`.
  See [ops/op_traits.md](ops/op_traits.md) Attention section.
- **OpsSession**: `begin_forward()` starts HLO tracing, `end_forward()` compiles and executes.
  See [ops/session_lifecycle.md](ops/session_lifecycle.md).
- **Static shapes**: `TpuOps` pads batch/seq internally. Model code doesn't know.

### Example 10: Qwen3-4B on MI300X (ROCm, FP8 FNUZ)

LLM inference on AMD MI300X with FP8 quantization. ROCm uses HIP flash attention
and CK GEMM with architecture-specific FP8 format.

```rust
// prelude-core/src/models/qwen3.rs — SAME code as CUDA and TPU, not one line differs

fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
    let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

    // Linear handles FP8 quantization internally:
    // qkv_proj was loaded with quant: Some(QuantInfo { scheme: Fp8E4m3 })
    // Linear.forward calls ops.gemm.quantized_matmul under the hood
    let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
    let (q, k, v) = modules::split_qkv(&qkv, 32, 8);
    let (q, k) = modules::apply_rope(&q, &k, &cos, &sin)?;

    ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
    let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;
    let h = self.o_proj.forward(&o, ctx, ops)?;

    let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
    let h = modules::gated_mlp(&h, &self.gate_proj, &self.up_proj, &self.down_proj, ctx, ops)?;
    Ok((&residual + &h)?)
}
```

Key points:
- **Identical code to CUDA/TPU examples**. FP8 is configured at load time in `Linear`:
  `qkv_proj = Linear { quant: Some(QuantInfo { scheme: Fp8E4m3 }), .. }`.
  See [modules/modules.md](modules/modules.md) Linear section.
- `Linear.forward` internally calls `ops.gemm.quantized_matmul` — model code doesn't know.
  See [ops/op_traits.md](ops/op_traits.md) GEMM section.
- **FP8 FNUZ auto-selected**: `RocmOps` detects gfx942, maps E4M3 to FNUZ format internally.
  On gfx950, native E4M3. See [device/device_impls.md](device/device_impls.md) ROCm section.
- **aiter flash attention**: specialized for MI300/MI350, falls back to CK on older AMD.
  See [device/device_impls.md](device/device_impls.md) ROCm section.

### Example 11: LLaDA2 (Diffusion LLM, Bidirectional Demasking)

Diffusion LLM: generates text by iteratively replacing [MASK] tokens with predicted tokens.
**Not autoregressive** — uses bidirectional attention (all tokens see all tokens).
The model architecture is a standard transformer, but with `MaskType::Bidirectional`.

```rust
// prelude-core/src/models/llada2.rs

impl LLaDA2Layer {
    fn forward(&self, x: &Tensor, ops: &Ops) -> Result<Tensor> {
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

        let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
        let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);

        // Bidirectional attention — every token attends to every token (no causal mask)
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask: MaskType::Bidirectional,   // ← only difference from AR LLM
            ..
        })?;

        let h = self.o_proj.forward(&o, ctx, ops)?;
        let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = modules::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
        Ok((&residual + &h)?)
    }
}
```

The **denoising loop** is engine-level (not model-level):

```rust
// prelude-core/src/engine/run.rs — engine's DLLM decode loop
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
- **Same modules as Qwen3**: `residual_norm`, `Linear` (TP-aware), `moe_layer`. All kernel optimizations on these benefit LLaDA2 too.
- **Only attention mask differs**: `MaskType::Bidirectional` instead of `Causal`. The kernel is the same FlashAttention with `causal=false`.
- **No KV cache for generation** (recompute every iteration). Can optionally cache across denoising iterations for speed.
- **No paged attention**: uses `varlen_attention` since there's no incremental decode.
- **Engine owns the denoising loop**: model.forward() is called block_size times. Scheduler controls when to stop.

### Example 12: Flux Full Pipeline (Denoising Loop + CFG + VAE Decode)

Complete image generation pipeline. Shows engine-level orchestration and how different
sub-models (text encoder, DiT, VAE) each get their own `Ops`.

```rust
// prelude-core/src/engine/run.rs — engine's Flux pipeline, NOT model code
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
    fn decode(&self, latents: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
        let mut h = self.post_quant_proj.forward(latents, ctx, ops)?;

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
- **Modules inside DiT**: `modules::adaln_zero`, `modules::gelu_mlp` — same as Example 2.

### Example 13: Multi-LoRA Serving (Llama-3-8B, 50 Concurrent Adapters)

Multi-tenant serving: each request uses a different LoRA adapter. A single batch contains
tokens from 50 different adapters. `Linear` handles LoRA internally — model code is
**identical** to non-LoRA Llama (compare Example 1's Qwen3).

```rust
// prelude-core/src/models/llama.rs — SAME code as non-LoRA version

impl LlamaLayer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;

        // Linear handles LoRA internally (BGMV/Punica on CUDA, fallback elsewhere)
        let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
        let (q, k, v) = modules::split_qkv(&qkv, 32, 8);

        // Attention is unchanged — LoRA only affects Linear layers
        let (q, k) = modules::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
        let h = self.o_proj.forward(&o, ctx, ops)?;

        let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = modules::gated_mlp(&h, &self.gate, &self.up, &self.down, ctx, ops)?;
        Ok((&residual + &h)?)
    }
}
// LoRA is configured at load time: qkv_proj = Linear { lora: Some(LoRAInfo { .. }), .. }
// ctx.adapter_ids = [batch] maps each token to its adapter (-1 = no adapter)
```

Key points:
- `ctx.adapter_ids: [batch]` maps each token to its LoRA adapter (-1 = no adapter).
- On CUDA: `Linear::forward` calls `fused_lora_matmul` → BGMV/Punica kernel. O(1) launch for all 50 adapters.
- On CPU/Metal: fallback splits batch by adapter, runs separate matmuls. Correct but slower.
- **Attention is identical to non-LoRA** — LoRA only wraps linear layers.

### Example 14: Qwen3.5 Hybrid (DeltaNet + Softmax, Per-Layer Dispatch)

Qwen3.5 uses DeltaNet (linear attention) for ~75% of layers and softmax attention for ~25%.
DeltaNet is model-owned (closure injection), softmax goes through `AttentionOps`.

```rust
// prelude-core/src/models/qwen35.rs

impl Qwen35Model {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let mut h = self.embed(x)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = match self.layer_type(i) {
                LayerType::Softmax => {
                    // Standard attention via modules (same as Qwen3)
                    layer.forward_softmax(&h, ops, kv)?
                }
                LayerType::DeltaNet => {
                    // DeltaNet: model-owned, uses conv1d + recurrent state
                    // Still uses ops.norm and ops.gemm — only the mixer is different
                    let (residual, h) = modules::residual_norm(&h, &layer.residual, &layer.ln1, eps, ops)?;

                    // Conv1d causal scan (DeltaNet-specific, not in AttentionOps)
                    let h = self.deltanet[i].causal_conv(&h, &self.conv_states[i])?;
                    // Recurrent state update (not expressible as attention)
                    let h = self.deltanet[i].recurrent_step(&h, &self.recurrent_states[i])?;

                    let h = ops.gemm.matmul(&h, &layer.o_proj)?;  // still uses GemmOps
                    let (residual, h) = modules::residual_norm(&residual, &h, &layer.ln2, eps, ops)?;
                    let h = modules::gated_mlp(&h, &layer.gate, &layer.up, &layer.down, ctx, ops)?;
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
- But DeltaNet layers **still use modules** for everything else: `residual_norm`, `gated_mlp`, `ops.gemm.matmul`. Kernel optimizations on these benefit DeltaNet layers too.
- Only the token mixer differs per layer. MLP, norms, projections are identical.

### Example 15: DeepSeek-V3 with EP (Expert Parallelism, 256 Experts on 8 GPUs)

MoE model with 256 routed experts distributed across 8 EP ranks (32 experts each).
Uses `modules::moe_layer` which handles EP dispatch/combine internally.

```rust
// prelude-core/src/models/deepseek_v3.rs

impl DeepSeekV3Layer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        // Attention (same as any other LLM)
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;
        let (q, k, v) = self.qkv_mla(&h, ops)?;  // MLA: compressed KV, head_dim_v != head_dim_q
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
        let h = self.o_proj.forward(&o, ctx, ops)?;

        // MoE with EP: module handles dispatch → grouped GEMM → combine
        let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = modules::moe_layer(&h, &self.gate, &self.expert_weights, &EpConfig {
            ep_size: 8,
            ep_rank: self.ep_rank,
            num_local_experts: 32,
        }, ops)?;
        Ok((&residual + &h)?)
    }
}
```

`modules::moe_layer` internally:
```rust
// prelude-core/src/modules/moe.rs

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
- Model code is clean — `modules::moe_layer` hides all EP complexity.
- Same model code works for EP=1 (single GPU) and EP=8 (8 GPUs).
- `CommOps::all_to_all` used inside `ep_dispatch`/`ep_combine`.

### Example 16: Speculative Decoding (EAGLE Draft + Target Verify)

Engine-level orchestration. Both draft and target models use the same `Ops`.

```rust
// prelude-core/src/engine/run.rs — speculative decode loop, NOT model code
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
// prelude-core/src/models/deepseek_v3.rs

impl DeepSeekMLA {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln, eps, ops)?;

        // Q projection: full dimension (192)
        let q = self.q_proj.forward(&h, ctx, ops)?;            // [total, nq_heads, 192]
        // KV projection: compressed latent → up-project to K, V
        let kv_compressed = self.kv_proj.forward(&h, ctx, ops)?;  // [total, nkv_heads, 128]
        let k = self.k_up.forward(&kv_compressed, ctx, ops)?;     // [total, nkv_heads, 128]
        let v = self.v_up.forward(&kv_compressed, ctx, ops)?;     // [total, nkv_heads, 128]

        // RoPE on partial dims: first 64 dims of Q get RoPE, rest are position-independent
        let (q_rope, q_nope) = q.split_at(/*dim=*/-1, 64)?;
        let q_rope = modules::apply_rope(&q_rope, cos, sin)?;
        let q = cat(&[&q_rope, &q_nope], -1)?;              // [total, nq_heads, 192]
        let k = modules::apply_rope(&k, cos, sin)?;          // [total, nkv_heads, 128]

        // Attention + KV cache
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
            mask: MaskType::Causal, ..
        })?;  // output: [total, nq_heads, 128] (head_dim_v, not head_dim_q)

        let h = self.o_proj.forward(&o, ctx, ops)?;
        Ok((&residual + &h)?)
    }
}
```

Key points:
- All projections (q_proj, kv_proj, k_up, v_up, o_proj) go through `Linear` — TP/quant transparent.
  See [modules/modules.md](modules/modules.md) Linear section.
- **head_dim_q (192) ≠ head_dim_kv (128)** — `AttentionOps` derives head_dim from tensor shapes,
  no special parameter needed. See [ops/op_traits.md](ops/op_traits.md) Attention design decisions.
- **Output dim = head_dim_v (128)**, not head_dim_q. The attention kernel handles this internally.
- **Partial RoPE**: first 64 dims of Q/K get rotary embedding, rest position-independent.
  This is model-level logic in `forward()`, not an ops concern.
- KV cache stores compressed 128-dim K/V, saving ~33% memory vs full 192-dim.
  `KvCacheOps::cache_slot_spec` reflects the 128-dim slot size.
  See [ops/op_traits.md](ops/op_traits.md) KV Cache section.

### Example 18: Whisper (Encoder-Decoder, Cross-Attention)

Speech recognition with separate audio encoder and text decoder.
Decoder uses cross-attention: Q from decoder, K/V from encoder. Same `varlen_attention`.

```rust
// prelude-core/src/models/whisper.rs

impl WhisperEncoder {
    fn forward(&self, mel: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
        // Conv1d feature extraction (raw ops — conv is not a Linear)
        let h = ops.conv.conv1d(mel, &self.conv1, Some(&self.conv1_bias), 1, 1)?;
        let h = ops.act.gelu(&h)?;
        let h = ops.conv.conv1d(&h, &self.conv2, Some(&self.conv2_bias), 2, 1)?;
        let h = ops.act.gelu(&h)?;

        // Transformer encoder with bidirectional attention
        for layer in &self.layers {
            let (residual, h_norm) = modules::residual_layer_norm(&h, &h, &layer.ln1, None, eps, ops)?;
            let qkv = layer.qkv_proj.forward(&h_norm, ctx, ops)?;
            let (q, k, v) = modules::split_qkv(&qkv, num_heads, num_heads);
            let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
                mask: MaskType::Bidirectional, ..
            })?;
            h = (&residual + &layer.o_proj.forward(&o, ctx, ops)?)?;
            // MLP
            let (residual, h_norm) = modules::residual_layer_norm(&h, &h, &layer.ln2, None, eps, ops)?;
            h = (&residual + &modules::gelu_mlp(&h_norm, &layer.fc1, &layer.fc2, ctx, ops)?)?;
        }
        Ok(h)
    }
}

// Text decoder (causal self-attention + cross-attention to encoder)
impl WhisperDecoderLayer {
    fn forward(
        &self, x: &Tensor, encoder_out: &Tensor, ctx: &BatchState, ops: &Ops,
        kv: &PagedKvCtx, encoder_seqlens: &[u32],
    ) -> Result<Tensor> {
        // Self-attention (causal, with KV cache)
        let (residual, h) = modules::residual_layer_norm(x, x, &self.ln1, None, eps, ops)?;
        let qkv = self.self_qkv_proj.forward(&h, ctx, ops)?;
        let (q, k, v) = modules::split_qkv(&qkv, num_heads, num_heads);
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &PagedParams {
            mask: MaskType::Causal, ..
        })?;
        let h = (&residual + &self.self_o_proj.forward(&o, ctx, ops)?)?;

        // Cross-attention: Q from decoder, K/V from encoder
        let (residual, h_norm) = modules::residual_layer_norm(&h, &h, &self.ln2, None, eps, ops)?;
        let q = self.cross_q_proj.forward(&h_norm, ctx, ops)?;          // [dec_total, heads, hdim]
        let k = self.cross_k_proj.forward(encoder_out, ctx, ops)?;      // [enc_total, heads, hdim]
        let v = self.cross_v_proj.forward(encoder_out, ctx, ops)?;      // [enc_total, heads, hdim]
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: &kv.cu_seqlens_q,   // decoder sequence lengths
            cu_seqlens_k: encoder_seqlens,     // encoder sequence lengths (different!)
            mask: MaskType::Bidirectional,     // cross-attention: no causal mask
            ..
        })?;
        let h = (&residual + &self.cross_o_proj.forward(&o, ctx, ops)?)?;

        // MLP
        let (residual, h_norm) = modules::residual_layer_norm(&h, &h, &self.ln3, None, eps, ops)?;
        Ok((&residual + &modules::gelu_mlp(&h_norm, &self.fc1, &self.fc2, ctx, ops)?)?)
    }
}
```

Key points:
- All projections go through `Linear` — encoder and decoder share the same pattern.
  See [modules/modules.md](modules/modules.md) Linear section.
- **Cross-attention is `varlen_attention`** with different `cu_seqlens_q` (decoder) and
  `cu_seqlens_k` (encoder). No special method needed.
  See [ops/op_traits.md](ops/op_traits.md) Attention design decisions.
- Decoder has **both** self-attention (`paged_attention`, causal) and cross-attention
  (`varlen_attention`, bidirectional). Two different attention ops in one layer.
- Encoder K/V are computed once and reused across all decoder layers/steps.
- Conv1d uses raw `ops.conv` — not a projection, no `Linear` wrapper.
  See [ops/op_traits.md](ops/op_traits.md) Convolution section.
- `layer_norm` (not `rms_norm`): Whisper uses pre-LN with LayerNorm. Same `NormOps`.

### Example 19: HunyuanVideo (Video Diffusion, Temporal + Spatial Attention)

Video generation DiT. Each block has two attention calls: spatial (within-frame) and
temporal (across-frame at same position). Both use `varlen_attention` with different cu_seqlens.

```rust
// prelude-core/src/models/hunyuan_video.rs

impl HunyuanVideoBlock {
    fn forward(
        &self, x: &Tensor, ctx: &BatchState, ops: &Ops,
        temb: &Tensor, num_frames: usize, spatial_tokens: usize,
    ) -> Result<Tensor> {
        // x shape: [batch * T * H*W, hidden]  (packed varlen)

        // ── Spatial attention: each frame attends within itself ──
        let (norm_x, gate_s) = modules::adaln_zero(x, &self.ln_s, None, &s1, &h1, &g1, eps, ops)?;
        let qkv = self.spatial_qkv_proj.forward(&norm_x, ctx, ops)?;
        let (q, k, v) = modules::split_qkv(&qkv, num_heads, num_heads);

        let spatial_seqlens: Vec<u32> = (0..=num_frames).map(|i| (i * spatial_tokens) as u32).collect();
        let o_spatial = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: &spatial_seqlens,
            cu_seqlens_k: &spatial_seqlens,
            max_seqlen_q: spatial_tokens as u32,
            max_seqlen_k: spatial_tokens as u32,
            mask: MaskType::Bidirectional, ..
        })?;
        let x = (x + &(self.s_out.forward(&o_spatial, ctx, ops)? * &gate_s)?)?;

        // ── Temporal attention: same spatial position attends across frames ──
        let (norm_x, gate_t) = modules::adaln_zero(&x, &self.ln_t, None, &s2, &h2, &g2, eps, ops)?;
        let x_temporal = rearrange_spatial_to_temporal(&norm_x, num_frames, spatial_tokens)?;
        let qkv = self.temporal_qkv_proj.forward(&x_temporal, ctx, ops)?;
        let (q, k, v) = modules::split_qkv(&qkv, num_heads, num_heads);

        let temporal_seqlens: Vec<u32> = (0..=spatial_tokens).map(|i| (i * num_frames) as u32).collect();
        let o_temporal = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            cu_seqlens_q: &temporal_seqlens,
            cu_seqlens_k: &temporal_seqlens,
            max_seqlen_q: num_frames as u32,
            max_seqlen_k: num_frames as u32,
            mask: MaskType::Bidirectional, ..
        })?;
        let o_temporal = rearrange_temporal_to_spatial(&o_temporal, num_frames, spatial_tokens)?;
        let x = (&x + &(self.t_out.forward(&o_temporal, ctx, ops)? * &gate_t)?)?;

        // MLP
        let (norm_x, gate_m) = modules::adaln_zero(&x, &self.ln_m, None, &s3, &h3, &g3, eps, ops)?;
        let x = (&x + &(modules::gelu_mlp(&norm_x, &self.fc1, &self.fc2, ctx, ops)? * &gate_m)?)?;
        Ok(x)
    }
}
```

Key points:
- All projections (spatial_qkv_proj, temporal_qkv_proj, s_out, t_out, fc1, fc2) go through
  `Linear`. See [modules/modules.md](modules/modules.md) Linear section.
- **Spatial + temporal = two `varlen_attention` calls** with different `cu_seqlens`.
  No special "video attention" trait — same `AttentionOps` as LLM, just different packing.
  See [ops/op_traits.md](ops/op_traits.md) Attention section.
- Spatial: `cu_seqlens = [0, H*W, 2*H*W, ...]` — each frame is a separate sequence.
- Temporal: transpose tokens, `cu_seqlens = [0, T, 2*T, ...]` — each spatial position across frames.
- **Same modules as image diffusion** (`modules::adaln_zero`, `modules::gelu_mlp`). Kernel optimizations transfer.
- **Same `AttentionOps`** as LLM. Just different sequence packing via `cu_seqlens`.

### Example 20: Mistral + Gemma Variants (Sliding Window, Softcap)

Shows how `MaskType` and `VarlenParams` parameters cover model-specific attention patterns.
No new ops needed — just different parameter values.

```rust
// prelude-core/src/models/ — Mistral, Gemma2, Gemma3

impl MistralLayer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
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
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
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
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
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
- **No module needed** for these — they're just different `VarlenParams` / `PagedParams` values.

### Example 21: BGE / GTE (Embedding Model, Encoder-Only for Retrieval)

BERT-like encoder-only model for text embedding / retrieval. Simplest use of the design:
bidirectional attention, no KV cache, no generation, no decoder.

```rust
// prelude-core/src/models/bge.rs

impl BgeLayer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
        let (residual, h) = modules::residual_layer_norm(x, x, &self.ln1, Some(&self.ln1_bias), eps, ops)?;
        let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
        let (q, k, v) = modules::split_qkv(&qkv, num_heads, num_heads);
        let o = ops.attn.varlen_attention(&q, &k, &v, &VarlenParams {
            mask: MaskType::Bidirectional, ..
        })?;
        let h = (&residual + &self.o_proj.forward(&o, ctx, ops)?)?;
        let (residual, h) = modules::residual_layer_norm(&h, &h, &self.ln2, Some(&self.ln2_bias), eps, ops)?;
        Ok((&residual + &modules::gelu_mlp(&h, &self.fc1, &self.fc2, ctx, ops)?)?)
    }
}

// Usage: encode once, pool, return embedding
fn embed(text: &str, model: &BgeModel, ctx: &BatchState, ops: &Ops) -> Result<Tensor> {
    let ids = tokenize(text);
    ops.session.begin_forward();
    let hidden = model.forward(&ids, ctx, ops)?;
    ops.session.end_forward();
    mean_pool(&hidden)
}
```

Key points:
- All projections through `Linear`, MLP through `modules::gelu_mlp`.
  See [modules/modules.md](modules/modules.md).
- **Minimal use of the design**: `varlen_attention` + `layer_norm` + `gelu_mlp`.
  No KV cache, no paged, no fusion tricks.
- `layer_norm` with bias (BERT-style), not `rms_norm` (LLM-style). Both in `NormOps`.
  See [ops/op_traits.md](ops/op_traits.md) Normalization section.
- Runs on any device including CPU (no paged attention needed).

### Example 22: CUDA Graph Capture/Replay (Engine-Level)

Shows how the engine uses `OpsSession` + CUDA-specific graph capture.
This is engine code, not model code.

```rust
// prelude-core/src/engine/executor.rs
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
    block_tables: &[u32],
    cu_seqlens_k: &[u32],
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

### Example 23: TurboQuant KV Cache Compression (KV Cache Quantization, Zero Model Changes)

KV cache quantization that compresses K/V to 2-4 bits using vector quantization.
Based on [TurboQuant](https://arxiv.org/abs/2504.19874). Shows how `cache_slot_spec`
and device-internal encode/decode make this transparent to model code.

**Algorithm:** Random rotation (Fast Walsh-Hadamard Transform) → Lloyd-Max scalar quantization
→ bit-pack into uint8. Decode is the reverse. After decode, standard bf16 K/V feed into
unmodified attention kernels.

**Integration: 4 touch points, 0 model changes.**

```rust
// prelude-core/src/ops/mod.rs — Step 1: OpsConfig

struct OpsConfig {
    // ... existing fields ...
    pub kv_cache_quant: Option<KvCacheQuantConfig>,
}

enum KvCacheQuantConfig {
    TurboQuant {
        bits: u8,              // 2, 3, 4
        lite: bool,            // skip rotation for speed
        outlier_fraction: f32, // fraction of channels kept at bf16 (0.0 = none)
    },
    // Future: KV-FP8, KIVI, etc.
}
```

```rust
// prelude-cuda/src/cuda_ops.rs — Step 2: CudaOps stores TurboQuant state

struct CudaOps {
    fa4: Option<FA4Registry>,
    fi: FlashInferRegistry,
    deepgemm: Option<DeepGemmRegistry>,
    cutlass: Option<CutlassHandle>,
    fi_workspace: FlashInferWorkspace,
    tq: Option<TurboQuantState>,       // ← only addition to CudaOps
}

/// Pre-computed TurboQuant state (created once at model load).
struct TurboQuantState {
    bits: u8,
    lite: bool,
    codebook: Tensor,          // Lloyd-Max quantization levels [2^bits]
    sign_flips: Tensor,        // deterministic random signs for Hadamard rotation
    n_outlier_channels: usize, // channels kept at bf16
    slot_bytes: usize,         // pre-computed: outlier_bytes + packed_bytes + norm_bytes
}
```

```rust
// prelude-cuda/src/kv_cache.rs — Step 3: cache_slot_spec + encode

impl KvCacheOps for CudaOps {
    fn cache_slot_spec(&self, head_dim: usize, dtype: DType) -> CacheSlotSpec {
        match &self.tq {
            Some(tq) => CacheSlotSpec {
                slot_size: tq.slot_bytes,
                dtype: DType::U8,  // packed quantized representation
            },
            None => CacheSlotSpec { slot_size: head_dim, dtype },
        }
    }

    fn reshape_and_cache(&self, key, value, key_cache, value_cache, slot_mapping) -> Result<()> {
        match &self.tq {
            Some(tq) => {
                // Encode: bf16 K/V → rotate → quantize → bit-pack → write uint8 cache
                tq_encode_and_cache(tq, key, value, key_cache, value_cache, slot_mapping)
            }
            None => {
                // Standard bf16 cache write
                standard_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
            }
        }
    }
}
```

```rust
// prelude-cuda/src/attention.rs — Step 4: decode before attention

impl AttentionOps for CudaOps {
    fn paged_attention(&self, q, key_cache, value_cache, params) -> Result<Tensor> {
        match &self.tq {
            Some(tq) => {
                // Decode: unpack → codebook lookup → inverse Hadamard → bf16 K/V
                let (k_bf16, v_bf16) = tq_decode_cache(tq, key_cache, value_cache, params)?;
                // Standard attention on decoded bf16 — kernel is unchanged
                fi_paged_prefill(&self.fi, &self.fi_workspace, q, &k_bf16, &v_bf16, params)
            }
            None => {
                if params.max_seqlen_q == 1 {
                    fi_paged_decode(&self.fi, &self.fi_workspace, q, key_cache, value_cache, params)
                } else {
                    // ... standard FA4 / FlashInfer dispatch (same as before) ...
                    fi_paged_prefill(&self.fi, &self.fi_workspace, q, key_cache, value_cache, params)
                }
            }
        }
    }

    // varlen_attention: unchanged (TurboQuant only affects paged KV cache)
    fn varlen_attention(&self, q, k, v, params) -> Result<Tensor> { /* same as before */ }
}
```

```rust
// prelude-core/src/engine/weight_loader.rs — engine cache allocation
fn allocate_kv_cache(ops: &Ops, num_blocks: usize, block_size: usize,
                     num_kv_heads: usize, head_dim: usize, dtype: DType) -> (Tensor, Tensor) {
    let spec = ops.kv_cache.cache_slot_spec(head_dim, dtype);
    // TurboQuant: [num_blocks, block_size, num_kv_heads, slot_bytes] in u8
    // Standard:   [num_blocks, block_size, num_kv_heads, head_dim] in bf16
    let shape = [num_blocks, block_size, num_kv_heads, spec.slot_size];
    (Tensor::zeros(&shape, spec.dtype), Tensor::zeros(&shape, spec.dtype))
}
```

```rust
// prelude-core/src/models/qwen3.rs — ZERO changes from non-quantized
// This is the exact same Qwen3 forward from Example 1. Not one line differs.
impl Qwen3Layer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;
        let qkv = self.qkv_proj.forward(&h, ctx, ops)?;
        let (q, k, v) = split_qkv(&qkv, num_heads_per_rank, num_kv_heads_per_rank);
        let (q, k) = modules::qk_norm_rope(&q, &k, &self.qw, &self.kw, cos, sin, pos, eps, ops)?;

        // These two calls are identical to non-quantized mode.
        // CudaOps handles encode/decode internally.
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &paged_params)?;

        let h = self.o_proj.forward(&o, ctx, ops)?;
        let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        let h = modules::moe_layer(&h, &self.gate, &self.expert_weights, &self.ep_config, ops)?;
        Ok((&residual + &h)?)
    }
}
```

Key points:
- **4 touch points**: `OpsConfig` (config), `CudaOps` (state), `KvCacheOps` (encode + spec), `AttentionOps` (decode). Compare vLLM's 19+ touch points for the same feature.
- **0 model changes**: model code passes bf16 K/V to `reshape_and_cache` and calls `paged_attention` exactly as before. Encode/decode is device-internal.
- **No new registries, no new backends, no new base classes**: just `if let Some(tq)` branches inside existing CudaOps methods.
- **Other devices unaffected**: `RocmOps`, `MetalOps`, etc. inherit `cache_slot_spec` default (uncompressed). Adding TurboQuant to ROCm later is another `if let Some(tq)` in RocmOps — same pattern.
- **Modules unaffected**: `modules::residual_norm`, `modules::qk_norm_rope`, etc. don't touch cache. No changes needed.
- **`cache_slot_spec` is general**: future KV cache quantization methods (KV-FP8, KIVI, etc.) use the same `cache_slot_spec` + encode/decode pattern. The mechanism is not TurboQuant-specific.

**Contrast with vLLM:**

| Concern | vLLM | Prelude |
|---------|------|---------|
| Configuration | Literal type + dict + validation class (3 files) | `OpsConfig` field (1 struct) |
| Registration | Quant registry + backend enum + selector (3 files) | None — `CudaOps` internal |
| Attention integration | 5 conditional branches in Attention layer | `if let Some(tq)` in `paged_attention` |
| Cache layout | Custom `get_kv_cache_spec()` override | `cache_slot_spec()` default method |
| Model code | Unchanged | Unchanged |
| Other backends | Must handle `kv_cache_dtype` checks | Inherit default, unaffected |
| Total touch points | 19+ | 4 |

The difference is architectural: vLLM's attention backend is monolithic (cache + attention coupled), so a cache-only change requires a new backend. Prelude's `KvCacheOps` / `AttentionOps` separation means cache compression is a device-internal concern — the right abstraction boundary absorbs the change.

### Example 24: Attention-FFN Disaggregation (AFD, MoE on Separate GPUs)

DeepSeek-V3 with 256 experts. Attention layers on 8 GPUs (TP=8), expert layers on 32 FFN GPUs
(EP=32, 8 experts each). Shows how `modules::moe_layer` absorbs AFD without model code changes.

```rust
// prelude-core/src/models/deepseek_v3.rs — ZERO changes from non-AFD
// Same DeepSeek-V3 forward from Example 15. Not one line differs.
impl DeepSeekV3Layer {
    fn forward(&self, x: &Tensor, ctx: &BatchState, ops: &Ops, kv: &PagedKvCtx) -> Result<Tensor> {
        let (residual, h) = modules::residual_norm(x, &self.residual, &self.ln1, eps, ops)?;
        let (q, k, v) = self.qkv_mla(&h, ops)?;
        ops.kv_cache.reshape_and_cache(&k, &v, &kv.cache_k, &kv.cache_v, &kv.slots)?;
        let o = ops.attn.paged_attention(&q, &kv.cache_k, &kv.cache_v, &params)?;
        let h = self.o_proj.forward(&o, ctx, ops)?;

        let (residual, h) = modules::residual_norm(&residual, &h, &self.ln2, eps, ops)?;
        // This call is identical to non-AFD. MoeConfig determines behavior.
        let h = modules::moe_layer(&h, &self.gate, &self.expert_weights, &self.moe_config, ops)?;
        Ok((&residual + &h)?)
    }
}
```

```rust
// prelude-core/src/engine/config.rs — MoeConfig at load time

// Non-AFD deployment (all on same GPUs):
let moe_config = MoeConfig { mode: MoeMode::ExpertParallel(EpConfig { ep_size: 8, .. }) };

// AFD deployment (attention + FFN on separate GPUs):
// Attention side:
let moe_config = MoeConfig { mode: MoeMode::Disaggregated(AfdConfig {
    role: AfdRole::Attention,
    ffn_target: RemoteTarget::new("ffn-pool:9000"),
}) };
// FFN side:
let moe_config = MoeConfig { mode: MoeMode::Disaggregated(AfdConfig {
    role: AfdRole::Ffn,
    ffn_target: RemoteTarget::new("attn-pool:9000"),
}) };
```

```rust
// prelude-core/src/modules/moe.rs — inside modules::moe_layer
// Attention side: route → send → recv
// FFN side: recv → grouped_gemm → send
// See AFD section under "Expert Parallelism" for full implementation.
```

Key points:
- **Model code is the same** as Example 15 (DeepSeek-V3 with EP). Only `MoeConfig` differs.
- **Deployment choice, not code choice.** Switch from EP to AFD by changing config, not code.
- **`modules::moe_layer` absorbs three modes:** `Local`, `ExpertParallel`, `Disaggregated`.
  Model devs never see the difference.
- **Contrast with SGLang:** SGLang replaces the MoE class with `AFDATTNMoE` / `AFDFFNMoE`
  per model, and currently only supports Qwen3MoE. Our approach works for any model using
  `modules::moe_layer` — DeepSeek-V3, Qwen3-MoE, future MoE models — with zero per-model work.

### Example 25: Adding a New Fused Kernel (Developer Workflow, Always Keep Last)

Scenario: kernel dev adds `fused_geglu` (GELU-gated MLP fusion) to CudaOps.
Shows the minimal change set.

```rust
// Step 1: prelude-core/src/ops/traits/fused.rs — add method with default { None }
trait FusedOps {
    // ... existing methods ...
    fn fused_geglu(&self, gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> { None }
}
// ← RocmOps, MetalOps, VulkanOps, TpuOps, CpuOps: ZERO changes (inherit None)

// Step 2: prelude-cuda/src/fused.rs — override in CudaOps
impl FusedOps for CudaOps {
    fn fused_geglu(&self, gate, up) -> Option<Result<Tensor>> {
        Some(triton::fused_geglu(gate, up))
    }
}

// Step 3: prelude-core/src/modules/mlp.rs — update the module
pub fn gelu_mlp(x: &Tensor, fc1: &Tensor, fc2: &Tensor, ops: &Ops) -> Result<Tensor> {
    let hidden = ops.gemm.matmul(x, fc1)?;
    let (gate, up) = hidden.chunk(2, -1)?;
    let h = match ops.fused.fused_geglu(&gate, &up) {
        Some(r) => r?,
        None => (ops.act.gelu(&gate)? * &up)?,  // fallback: separate ops
    };
    ops.gemm.matmul(&h, fc2)
}
// ← All models using modules::gelu_mlp (Flux, Sana, all diffusion models) benefit automatically
```

Total changes: **3 locations** (trait def, CudaOps, module).
Models changed: **0**.
Models that benefit: **all models using `modules::gelu_mlp`**.
