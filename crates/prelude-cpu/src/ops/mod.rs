//! Pure Rust CPU kernels — AVX-512 optimized, zero external dependencies.
//!
//! Each kernel file contains both the raw `&[u16]` kernel and the Tensor-level API.
//! Public Tensor functions are re-exported here for convenience.

pub mod attention;
pub mod bf16_utils;
pub mod buf_tensor;
pub mod cpu_float;
pub mod gemm;
pub mod gemm_pool;
pub mod numa;
pub mod quant;
pub mod rmsnorm;
pub mod rope;
pub mod silu_mul;
pub mod softmax;

use prelude_core::tensor::{Device, Result, Storage, Tensor};

// ── Re-exports: Tensor-level kernel API ─────────────────────────────────

pub use rmsnorm::{CpuRmsNorm, cpu_fused_add_rmsnorm, cpu_rmsnorm};
pub use silu_mul::{cpu_silu_and_mul, cpu_silu_and_mul_inplace};

// Attention: tiled kernels for BF16 and F32.
pub use self::attention_tensor::{
    cpu_decode_attention, cpu_decode_attention_f32, cpu_prefill_attention,
};

// RoPE
pub use self::rope_tensor::{cpu_rotary_embedding, cpu_rotary_embedding_with_positions};
pub use self::rope_tensor::{cpu_rotary_embedding_f32, cpu_rotary_embedding_f32_with_positions};

// ── Thread pool helpers ─────────────────────────────────────────────────

/// Number of threads in the rayon pool.
pub(crate) fn pool_size() -> usize {
    rayon::current_num_threads()
}

/// Adaptive parallelization check.
pub(crate) fn should_parallelize(
    total_elems: usize,
    num_rows: usize,
    min_elems_per_thread: usize,
) -> bool {
    if num_rows <= 1 {
        return false;
    }
    let ps = pool_size();
    let n_threads = ps.min(num_rows);
    let per_thread = total_elems / n_threads;
    per_thread >= min_elems_per_thread && total_elems >= min_elems_per_thread * ps
}

/// SGLang-style BF16 tolerance check.
#[cfg(test)]
pub(crate) fn max_sglang_violation(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(&a, &e)| {
            let diff = (a - e).abs();
            let tol = 1e-2 + 1e-2 * a.abs().max(e.abs());
            diff - tol
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

// ── BF16 tensor helpers (shared across kernel files) ────────────────────

/// Get a zero-copy `&[u16]` view of a CPU BF16 tensor (public).
pub fn tensor_as_u16_slice_pub(tensor: &Tensor) -> Result<&[u16]> {
    tensor_as_u16_slice(tensor)
}

/// Get a raw const pointer to a contiguous CPU tensor's data, reinterpreted as `*const T`.
/// The tensor's actual storage type must be layout-compatible with `T`.
/// Pointer is valid while the tensor is alive.
pub(crate) fn tensor_raw_ptr<T>(tensor: &Tensor) -> Result<*const T> {
    use prelude_core::tensor::CpuStorage;
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cpu(cpu) => {
            // Match on candle CpuStorage variants to get the raw data pointer.
            let ptr = match cpu {
                CpuStorage::U8(v) => v.as_ptr() as *const T,
                CpuStorage::U32(v) => v.as_ptr() as *const T,
                CpuStorage::I32(v) => v.as_ptr() as *const T,
                CpuStorage::I64(v) => v.as_ptr() as *const T,
                CpuStorage::BF16(v) => v.as_ptr() as *const T,
                CpuStorage::F16(v) => v.as_ptr() as *const T,
                CpuStorage::F32(v) => v.as_ptr() as *const T,
                CpuStorage::F64(v) => v.as_ptr() as *const T,
                _ => prelude_core::tensor::bail!("tensor_raw_ptr: unsupported storage variant"),
            };
            Ok(ptr)
        }
        _ => prelude_core::tensor::bail!("tensor_raw_ptr: expected CPU tensor"),
    }
}

/// Get a zero-copy `&[u16]` view of a CPU BF16 tensor.
/// Tensor must be contiguous.
///
/// # Safety (lifetime)
/// The returned slice borrows from the tensor's `Arc<Storage>`.
/// The data pointer is stable because CPU storage Vecs are never reallocated
/// after creation. The caller must keep the tensor alive.
/// Internal zero-copy slice (unsafe lifetime extension).
/// Pointer valid while tensor is alive. For perf-critical internal code only.
pub(crate) fn tensor_as_u16_slice(tensor: &Tensor) -> Result<&[u16]> {
    assert!(tensor.is_contiguous());
    unsafe {
        Ok(std::slice::from_raw_parts(
            tensor_raw_ptr::<u16>(tensor)?,
            tensor.elem_count(),
        ))
    }
}

pub(crate) fn tensor_as_bf16_slice(tensor: &Tensor) -> Result<&[half::bf16]> {
    assert!(tensor.is_contiguous());
    unsafe {
        Ok(std::slice::from_raw_parts(
            tensor_raw_ptr::<half::bf16>(tensor)?,
            tensor.elem_count(),
        ))
    }
}

pub(crate) fn tensor_as_f32_slice(tensor: &Tensor) -> Result<&[f32]> {
    assert!(tensor.is_contiguous());
    unsafe {
        Ok(std::slice::from_raw_parts(
            tensor_raw_ptr::<f32>(tensor)?,
            tensor.elem_count(),
        ))
    }
}

/// Get a raw mutable pointer to a contiguous CPU tensor's data, reinterpreted as `*mut T`.
/// The tensor's actual storage type must be layout-compatible with `T`.
/// Caller must ensure exclusive access (no other aliases).
pub(crate) fn tensor_data_ptr_mut<T>(tensor: &Tensor) -> Result<*mut T> {
    Ok(tensor_raw_ptr::<T>(tensor)? as *mut T)
}

/// Convert a `Vec<u16>` (BF16 bit patterns) back to a BF16 tensor.
pub(crate) fn u16_vec_to_bf16_tensor(
    buf: Vec<u16>,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let bf16_vec: Vec<half::bf16> = bytemuck::cast_vec(buf);
    Tensor::from_vec(bf16_vec, shape, device)
}

// ── Attention Tensor wrapper (BF16 only) ────────────────────────────────

mod attention_tensor {
    use prelude_core::tensor::{Result, Tensor};

    /// Prefill attention using optimized BF16 CPU kernels.
    /// For F32, use matmul SDPA via `cpu_varlen_attention` in layers/ops.rs.
    pub fn cpu_prefill_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_lens: &[usize],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f64,
    ) -> Result<Tensor> {
        debug_assert_eq!(
            q.dtype(),
            prelude_core::tensor::DType::BF16,
            "cpu_prefill_attention requires BF16"
        );
        let total_tokens = q.dims()[0];

        let q_cont = q.contiguous()?;
        let k_cont = k.contiguous()?;
        let v_cont = v.contiguous()?;
        let q_slice = super::tensor_as_u16_slice(&q_cont)?;
        let k_slice = super::tensor_as_u16_slice(&k_cont)?;
        let v_slice = super::tensor_as_u16_slice(&v_cont)?;

        let mut out_buf = vec![0u16; total_tokens * num_heads * head_dim];
        super::attention::prefill_attention_bf16(
            &mut out_buf,
            q_slice,
            k_slice,
            v_slice,
            seq_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale as f32,
        );

        super::u16_vec_to_bf16_tensor(out_buf, &[total_tokens, num_heads, head_dim], q.device())
    }

    /// Decode attention: single Q token against contiguous KV cache (BF16).
    ///
    /// q: `[1, num_heads, head_dim]`, k/v_cache: `[context_len, num_kv_heads, head_dim]`
    /// Returns: `[1, num_heads, head_dim]`
    pub fn cpu_decode_attention(
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
    ) -> Result<Tensor> {
        debug_assert_eq!(
            q.dtype(),
            prelude_core::tensor::DType::BF16,
            "cpu_decode_attention requires BF16"
        );

        let q_cont = q.contiguous()?;
        let k_cont = k_cache.contiguous()?;
        let v_cont = v_cache.contiguous()?;
        let q_slice = super::tensor_as_u16_slice(&q_cont)?;
        let k_slice = super::tensor_as_u16_slice(&k_cont)?;
        let v_slice = super::tensor_as_u16_slice(&v_cont)?;

        let mut out_buf = vec![0u16; num_heads * head_dim];
        // Identity mapping: position i → slot i (contiguous cache)
        let req_to_token: Vec<i32> = (0..context_len as i32).collect();
        let seq_lens = [context_len as i64];

        super::attention::decode_attention_bf16(
            &mut out_buf,
            q_slice,
            k_slice,
            v_slice,
            &req_to_token,
            &seq_lens,
            1,
            context_len,
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
        );

        super::u16_vec_to_bf16_tensor(out_buf, &[1, num_heads, head_dim], q.device())
    }

    /// Decode attention: single Q token against contiguous KV cache (F32).
    ///
    /// q: `[1, num_heads, head_dim]`, k/v_cache: `[context_len, num_kv_heads, head_dim]`
    /// Returns: `[1, num_heads, head_dim]`
    pub fn cpu_decode_attention_f32(
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
    ) -> Result<Tensor> {
        debug_assert_eq!(
            q.dtype(),
            prelude_core::tensor::DType::F32,
            "cpu_decode_attention_f32 requires F32"
        );

        let q_cont = q.contiguous()?;
        let k_cont = k_cache.contiguous()?;
        let v_cont = v_cache.contiguous()?;
        let q_slice = super::tensor_as_f32_slice(&q_cont)?;
        let k_slice = super::tensor_as_f32_slice(&k_cont)?;
        let v_slice = super::tensor_as_f32_slice(&v_cont)?;

        let mut out_buf = vec![0.0f32; num_heads * head_dim];
        let req_to_token: Vec<i32> = (0..context_len as i32).collect();
        let seq_lens = [context_len as i64];

        super::attention::decode_attention_f32(
            &mut out_buf,
            q_slice,
            k_slice,
            v_slice,
            &req_to_token,
            &seq_lens,
            1,
            context_len,
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
        );

        Tensor::from_vec(out_buf, &[1, num_heads, head_dim], q.device())
    }
}

// ── RoPE Tensor wrappers ────────────────────────────────────────────────

mod rope_tensor {
    use prelude_core::tensor::{Result, Tensor};

    /// Apply NeoX-style RoPE in-place to Q and K tensors (BF16).
    pub fn cpu_rotary_embedding(
        q: &Tensor,
        k: &Tensor,
        cos_sin_cache: &Tensor,
        offset: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len) = (q_dims[0], q_dims[1]);
        let positions: Vec<i64> = (0..batch_size)
            .flat_map(|_| (0..seq_len).map(|s| (offset + s) as i64))
            .collect();
        cpu_rotary_embedding_with_positions(
            q,
            k,
            cos_sin_cache,
            &positions,
            num_heads,
            num_kv_heads,
        )
    }

    /// Apply NeoX-style RoPE with explicit per-token positions (BF16).
    pub fn cpu_rotary_embedding_with_positions(
        q: &Tensor,
        k: &Tensor,
        cos_sin_cache: &Tensor,
        positions: &[i64],
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len, _nh, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let rotary_dim = cos_sin_cache.dims()[1];

        let q_2d = q
            .contiguous()?
            .reshape((batch_size * seq_len * num_heads * head_dim,))?;
        let k_2d = k
            .contiguous()?
            .reshape((batch_size * seq_len * num_kv_heads * head_dim,))?;
        let cache_slice = super::tensor_as_u16_slice(cos_sin_cache)?;

        {
            let q_n = batch_size * seq_len * num_heads * head_dim;
            let k_n = batch_size * seq_len * num_kv_heads * head_dim;
            let q_slice = unsafe {
                std::slice::from_raw_parts_mut(super::tensor_data_ptr_mut::<u16>(&q_2d)?, q_n)
            };
            let k_slice = unsafe {
                std::slice::from_raw_parts_mut(super::tensor_data_ptr_mut::<u16>(&k_2d)?, k_n)
            };

            super::rope::rope_neox_bf16(
                q_slice,
                k_slice,
                cache_slice,
                positions,
                batch_size,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
            );
        }

        Ok((q_2d.reshape(q.dims())?, k_2d.reshape(k.dims())?))
    }

    /// Apply NeoX-style RoPE in-place to Q and K tensors (F32).
    pub fn cpu_rotary_embedding_f32(
        q: &Tensor,
        k: &Tensor,
        cos_sin_cache: &Tensor,
        offset: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len) = (q_dims[0], q_dims[1]);
        let positions: Vec<i64> = (0..batch_size)
            .flat_map(|_| (0..seq_len).map(|s| (offset + s) as i64))
            .collect();
        cpu_rotary_embedding_f32_with_positions(
            q,
            k,
            cos_sin_cache,
            &positions,
            num_heads,
            num_kv_heads,
        )
    }

    /// Apply NeoX-style RoPE with explicit per-token positions (F32).
    pub fn cpu_rotary_embedding_f32_with_positions(
        q: &Tensor,
        k: &Tensor,
        cos_sin_cache: &Tensor,
        positions: &[i64],
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len, _nh, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let rotary_dim = cos_sin_cache.dims()[1];

        let q_2d = q
            .contiguous()?
            .reshape((batch_size * seq_len * num_heads * head_dim,))?;
        let k_2d = k
            .contiguous()?
            .reshape((batch_size * seq_len * num_kv_heads * head_dim,))?;
        let cache_slice = super::tensor_as_f32_slice(cos_sin_cache)?;

        {
            let q_n = batch_size * seq_len * num_heads * head_dim;
            let k_n = batch_size * seq_len * num_kv_heads * head_dim;
            let q_slice = unsafe {
                std::slice::from_raw_parts_mut(super::tensor_data_ptr_mut::<f32>(&q_2d)?, q_n)
            };
            let k_slice = unsafe {
                std::slice::from_raw_parts_mut(super::tensor_data_ptr_mut::<f32>(&k_2d)?, k_n)
            };

            super::rope::rope_neox_f32(
                q_slice,
                k_slice,
                cache_slice,
                positions,
                batch_size,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
            );
        }

        Ok((q_2d.reshape(q.dims())?, k_2d.reshape(k.dims())?))
    }
}
