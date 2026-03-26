//! Attention backend trait.
//!
//! Defines the `AttentionBackend` trait that all attention backends implement.
//! Models call methods on `dyn AttentionBackend` via `select_backend()`;
//! the `#[cfg]` feature gates are concentrated in that one factory function
//! instead of being scattered through every call site.

use candle_core::{Result, Tensor};

/// Unified interface for attention backends.
///
/// Each method corresponds to a free function in `attn/mod.rs`. Backends that
/// do not support a particular operation (e.g., CPU has no paged attention)
/// should return an error from the unsupported methods.
#[allow(clippy::too_many_arguments)]
pub(crate) trait AttentionBackend: Send + Sync {
    /// Human-readable backend name (e.g., "flash-attn-v4", "cpu").
    fn name(&self) -> &str;

    /// Causal varlen attention (non-paged).
    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor>;

    /// Varlen attention with sliding window.
    fn varlen_attention_windowed(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
        window_left: Option<usize>, window_right: Option<usize>,
    ) -> Result<Tensor>;

    /// Non-causal (bidirectional) varlen attention.
    fn varlen_attention_bidirectional(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor>;

    /// Paged varlen attention: read from paged KV cache (no KV write).
    fn varlen_attention_paged(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor>;

    /// Write K/V to paged cache.
    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()>;

    /// Whether this backend supports paged attention for prefill (Q > 1).
    /// FA2 only supports paged decode (Q=1); all other backends support both.
    fn supports_paged_prefill(&self) -> bool { true }
}

// ── FA4 backend ──────────────────────────────────────────────────────

#[cfg(feature = "flash-attn-v4")]
pub(crate) struct FlashAttnV4Backend;

#[cfg(feature = "flash-attn-v4")]
impl AttentionBackend for FlashAttnV4Backend {
    fn name(&self) -> &str { "flash-attn-v4" }

    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v4::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_windowed(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
        window_left: Option<usize>, window_right: Option<usize>,
    ) -> Result<Tensor> {
        super::flash_v4::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        )
    }

    fn varlen_attention_bidirectional(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v4::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_paged(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        let seqused_k = super::cu_seqlens_to_lens(cu_seqlens_k)?;
        super::flash_v4::varlen_paged(
            q, key_cache, value_cache, block_tables,
            cu_seqlens_q, &seqused_k, max_seqlen_q, max_seqlen_k,
            softmax_scale,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        super::paged::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    }
}

// ── FlashInfer backend ───────────────────────────────────────────────

#[cfg(feature = "flashinfer")]
pub(crate) struct FlashInferBackend;

#[cfg(feature = "flashinfer")]
impl AttentionBackend for FlashInferBackend {
    fn name(&self) -> &str { "flashinfer" }

    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flashinfer::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_windowed(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
        window_left: Option<usize>, window_right: Option<usize>,
    ) -> Result<Tensor> {
        super::flashinfer::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        )
    }

    fn varlen_attention_bidirectional(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flashinfer::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_paged(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flashinfer::varlen_paged(
            q, key_cache, value_cache, block_tables,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            softmax_scale,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        super::paged::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    }
}

// ── FA3 backend ──────────────────────────────────────────────────────

#[cfg(feature = "flash-attn-v3")]
pub(crate) struct FlashAttnV3Backend;

#[cfg(feature = "flash-attn-v3")]
impl AttentionBackend for FlashAttnV3Backend {
    fn name(&self) -> &str { "flash-attn-v3" }

    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v3::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_windowed(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
        window_left: Option<usize>, window_right: Option<usize>,
    ) -> Result<Tensor> {
        super::flash_v3::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        )
    }

    fn varlen_attention_bidirectional(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v3::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_paged(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v3::varlen_paged(
            q, key_cache, value_cache, block_tables,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            softmax_scale,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        super::paged::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    }
}

// ── FA2 backend ──────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
pub(crate) struct FlashAttnV2Backend;

#[cfg(feature = "flash-attn")]
impl AttentionBackend for FlashAttnV2Backend {
    fn name(&self) -> &str { "flash-attn-v2" }

    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v2::varlen_causal(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_windowed(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
        window_left: Option<usize>, window_right: Option<usize>,
    ) -> Result<Tensor> {
        super::flash_v2::varlen_windowed(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
            window_left, window_right,
        )
    }

    fn varlen_attention_bidirectional(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::flash_v2::varlen_bidirectional(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale,
        )
    }

    fn varlen_attention_paged(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
        _cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        _max_seqlen_q: usize, max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        // FA2 paged: decode-only (Q=1) via vLLM paged_attention kernel.
        let context_lens = super::cu_seqlens_to_lens(cu_seqlens_k)?;
        super::paged::decode_attention(
            q, key_cache, value_cache, block_tables,
            &context_lens, max_seqlen_k, softmax_scale,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        super::paged::reshape_and_cache_v1(key, value, key_cache, value_cache, slot_mapping)
    }

    fn supports_paged_prefill(&self) -> bool { false }
}

// ── CPU backend ──────────────────────────────────────────────────────

pub(crate) struct CpuBackend;

impl AttentionBackend for CpuBackend {
    fn name(&self) -> &str { "cpu" }

    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        _max_seqlen_q: usize, _max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    }

    fn varlen_attention_windowed(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
        _max_seqlen_q: usize, _max_seqlen_k: usize,
        softmax_scale: f32,
        _window_left: Option<usize>, _window_right: Option<usize>,
    ) -> Result<Tensor> {
        // CPU fallback: ignore window, use standard causal attention.
        super::cpu::varlen_causal(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    }

    fn varlen_attention_bidirectional(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        cu_seqlens_q: &Tensor, _cu_seqlens_k: &Tensor,
        _max_seqlen_q: usize, _max_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        super::cpu::varlen_bidirectional(q, k, v, cu_seqlens_q, softmax_scale)
    }

    fn varlen_attention_paged(
        &self,
        _q: &Tensor,
        _key_cache: &Tensor, _value_cache: &Tensor, _block_tables: &Tensor,
        _cu_seqlens_q: &Tensor, _cu_seqlens_k: &Tensor,
        _max_seqlen_q: usize, _max_seqlen_k: usize,
        _softmax_scale: f32,
    ) -> Result<Tensor> {
        candle_core::bail!("varlen_attention_paged is not supported on CPU")
    }

    fn reshape_and_cache(
        &self,
        _key: &Tensor, _value: &Tensor,
        _key_cache: &Tensor, _value_cache: &Tensor,
        _slot_mapping: &Tensor,
    ) -> Result<()> {
        candle_core::bail!("reshape_and_cache is not supported on CPU")
    }
}

// ── Backend selection ────────────────────────────────────────────────

/// Select the best available attention backend based on compiled features
/// and the device type.
///
/// Returns the cached attention backend for the given device class.
///
/// Priority: FA4 → FlashInfer → FA3 → FA2 → CPU.
/// The backend is resolved once on first call and cached for the lifetime of
/// the process — no allocation on subsequent calls.
pub(crate) fn select_backend(is_cuda: bool) -> &'static dyn AttentionBackend {
    use std::sync::OnceLock;

    static GPU_BACKEND: OnceLock<Box<dyn AttentionBackend>> = OnceLock::new();
    static CPU_BACKEND: CpuBackend = CpuBackend;

    if is_cuda {
        GPU_BACKEND.get_or_init(|| {
            let backend: Box<dyn AttentionBackend> = select_gpu_backend();
            tracing::info!(backend = backend.name(), "attention backend selected");
            backend
        }).as_ref()
    } else {
        &CPU_BACKEND
    }
}

/// Select the highest-priority GPU backend among those compiled in.
fn select_gpu_backend() -> Box<dyn AttentionBackend> {
    #[cfg(feature = "flash-attn-v4")]
    return Box::new(FlashAttnV4Backend);

    #[cfg(feature = "flashinfer")]
    return Box::new(FlashInferBackend);

    #[cfg(feature = "flash-attn-v3")]
    return Box::new(FlashAttnV3Backend);

    #[cfg(feature = "flash-attn")]
    return Box::new(FlashAttnV2Backend);

    #[allow(unreachable_code)]
    {
        tracing::warn!("no GPU attention backend compiled, falling back to CPU");
        Box::new(CpuBackend)
    }
}
