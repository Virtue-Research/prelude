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

use prelude_core::tensor::{Device, Result, Tensor};

// ── Re-exports: Tensor-level kernel API ─────────────────────────────────

pub use rmsnorm::{cpu_rmsnorm, cpu_fused_add_rmsnorm, CpuRmsNorm};
pub use silu_mul::{cpu_silu_and_mul, cpu_silu_and_mul_inplace};

// Attention: tiled kernels for BF16 and F32.
pub use self::attention_tensor::{cpu_prefill_attention, cpu_decode_attention, cpu_decode_attention_f32};

// RoPE
pub use self::rope_tensor::{cpu_rotary_embedding, cpu_rotary_embedding_with_positions};
pub use self::rope_tensor::{cpu_rotary_embedding_f32, cpu_rotary_embedding_f32_with_positions};

// ── Thread pool helpers ─────────────────────────────────────────────────

/// Number of threads in the rayon pool.
pub(crate) fn pool_size() -> usize {
    rayon::current_num_threads()
}

/// Run `f(chunk_id)` for chunk_id in [0, n) in parallel via rayon.
pub(crate) fn parallel_for<F>(n: usize, f: &F)
where
    F: Fn(usize) + Sync,
{
    use rayon::prelude::*;
    (0..n).into_par_iter().for_each(|i| f(i));
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

/// Returns optimal thread count for given workload.
pub(crate) fn optimal_thread_count(
    total_elems: usize,
    num_rows: usize,
    min_elems_per_thread: usize,
) -> usize {
    let ps = pool_size();
    let max_useful = (total_elems / min_elems_per_thread).max(1);
    max_useful.min(num_rows).min(ps)
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

/// Get a zero-copy `&[u16]` view of a CPU BF16 tensor.
/// Tensor must be contiguous.
pub(crate) fn tensor_as_u16_slice(tensor: &Tensor) -> Result<&[u16]> {
    if !tensor.is_contiguous() {
        prelude_core::tensor::bail!(
            "tensor_as_u16_slice: tensor is not contiguous (shape={:?}, stride={:?}). \
             Call .contiguous()? first and keep the result alive.",
            tensor.dims(), tensor.stride()
        );
    }
    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    let n = layout.shape().elem_count();
    match &*storage {
        prelude_core::tensor::backend::Storage::Cpu(cpu) => {
            let slice = cpu.as_slice::<half::bf16>()?;
            let data = &slice[offset..offset + n];
            Ok(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len()) })
        }
        _ => prelude_core::tensor::bail!("cpu_ops: expected CPU storage"),
    }
}

/// Get a zero-copy `&[f32]` view of a CPU F32 tensor.
/// Tensor must be contiguous.
///
/// # Safety
/// Same lifetime contract as `tensor_as_u16_slice`: the returned slice is
/// valid as long as the originating `Tensor` is alive.
pub(crate) fn tensor_as_f32_slice(tensor: &Tensor) -> Result<&[f32]> {
    if !tensor.is_contiguous() {
        prelude_core::tensor::bail!(
            "tensor_as_f32_slice: tensor is not contiguous (shape={:?}, stride={:?})",
            tensor.dims(), tensor.stride()
        );
    }
    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    let n = layout.shape().elem_count();
    match &*storage {
        prelude_core::tensor::backend::Storage::Cpu(cpu) => {
            let slice = cpu.as_slice::<f32>()?;
            let data = &slice[offset..offset + n];
            Ok(unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) })
        }
        _ => prelude_core::tensor::bail!("tensor_as_f32_slice: expected CPU storage"),
    }
}

/// Get a mutable `&mut [u16]` view into a CPU BF16 tensor's storage.
///
/// # Safety
/// Caller must guarantee exclusive access to the storage region `[offset..offset+n]`.
pub(crate) unsafe fn extract_bf16_mut_u16(
    storage: &mut prelude_core::tensor::backend::Storage,
    offset: usize,
    n: usize,
) -> Result<&mut [u16]> {
    match storage {
        prelude_core::tensor::backend::Storage::Cpu(prelude_core::tensor::backend::CpuStorage::BF16(data)) => {
            let ptr = unsafe { data.as_mut_ptr().add(offset) as *mut u16 };
            Ok(unsafe { std::slice::from_raw_parts_mut(ptr, n) })
        }
        _ => prelude_core::tensor::bail!("cpu_ops: expected CPU BF16 storage"),
    }
}

/// Get a mutable `&mut [f32]` view into a CPU F32 tensor's storage.
///
/// # Safety
/// Caller must guarantee exclusive access to the storage region `[offset..offset+n]`.
pub(crate) unsafe fn extract_f32_mut(
    storage: &mut prelude_core::tensor::backend::Storage,
    offset: usize,
    n: usize,
) -> Result<&mut [f32]> {
    match storage {
        prelude_core::tensor::backend::Storage::Cpu(prelude_core::tensor::backend::CpuStorage::F32(data)) => {
            let ptr = unsafe { data.as_mut_ptr().add(offset) };
            Ok(unsafe { std::slice::from_raw_parts_mut(ptr, n) })
        }
        _ => prelude_core::tensor::bail!("cpu_ops: expected CPU F32 storage"),
    }
}

// ── Safe in-place mutation helpers ──────────────────────────────────────

/// In-place BF16 residual add: `tensor[..] += src[..]`.
///
/// This is safe because it acquires exclusive mutable access via candle's
/// `storage_mut_and_layout()` write lock. The caller must ensure this tensor
/// is not aliased by other tensors sharing the same storage (e.g., via reshape/narrow).
pub(crate) fn inplace_add_bf16(tensor: &Tensor, src: &[u16]) -> Result<()> {
    let n = src.len();
    // SAFETY: we hold the exclusive write lock for the duration of this function.
    let (mut storage, layout) = unsafe { tensor.storage_mut_and_layout() };
    // SAFETY: we hold the exclusive write lock via storage_mut_and_layout().
    // Caller guarantees no other tensor aliases this storage region.
    let data = unsafe { extract_bf16_mut_u16(&mut storage, layout.start_offset(), n)? };
    // SAFETY: both pointers are valid, non-overlapping, and cover `n` elements.
    unsafe {
        crate::raw_cpu::raw_residual_add_bf16(
            data.as_mut_ptr(),
            src.as_ptr(),
            n,
        );
    }
    Ok(())
}

/// In-place F32 residual add: `tensor[..] += src[..]`.
pub(crate) fn inplace_add_f32(tensor: &Tensor, src: &[f32]) -> Result<()> {
    let n = src.len();
    // SAFETY: we hold the exclusive write lock for the duration of this function.
    let (mut storage, layout) = unsafe { tensor.storage_mut_and_layout() };
    let data = unsafe { extract_f32_mut(&mut storage, layout.start_offset(), n)? };
    unsafe {
        crate::raw_cpu::raw_residual_add_f32(
            data.as_mut_ptr(),
            src.as_ptr(),
            n,
        );
    }
    Ok(())
}

/// Convert a `Vec<u16>` (BF16 bit patterns) back to a Candle BF16 tensor.
pub(crate) fn u16_vec_to_bf16_tensor(buf: Vec<u16>, shape: &[usize], device: &Device) -> Result<Tensor> {
    let bf16_vec: Vec<half::bf16> = bytemuck::cast_vec(buf);
    Tensor::from_vec(bf16_vec, shape, device)
}

// ── Attention Tensor wrapper (BF16 only) ────────────────────────────────

mod attention_tensor {
    use prelude_core::tensor::{Result, Tensor};

    /// Prefill attention using optimized BF16 CPU kernels.
    /// For F32, use matmul SDPA via `cpu_varlen_attention` in layers/ops.rs.
    pub fn cpu_prefill_attention(
        q: &Tensor, k: &Tensor, v: &Tensor,
        seq_lens: &[usize], num_heads: usize, num_kv_heads: usize,
        head_dim: usize, sm_scale: f64,
    ) -> Result<Tensor> {
        debug_assert_eq!(q.dtype(), prelude_core::tensor::DType::BF16, "cpu_prefill_attention requires BF16");
        let total_tokens = q.dims()[0];

        let q_cont = q.contiguous()?;
        let k_cont = k.contiguous()?;
        let v_cont = v.contiguous()?;
        let q_slice = super::tensor_as_u16_slice(&q_cont)?;
        let k_slice = super::tensor_as_u16_slice(&k_cont)?;
        let v_slice = super::tensor_as_u16_slice(&v_cont)?;

        let mut out_buf = vec![0u16; total_tokens * num_heads * head_dim];
        super::attention::prefill_attention_bf16(
            &mut out_buf, q_slice, k_slice, v_slice,
            seq_lens, num_heads, num_kv_heads, head_dim, sm_scale as f32,
        );

        super::u16_vec_to_bf16_tensor(out_buf, &[total_tokens, num_heads, head_dim], q.device())
    }

    /// Decode attention: single Q token against contiguous KV cache (BF16).
    ///
    /// q: `[1, num_heads, head_dim]`, k/v_cache: `[context_len, num_kv_heads, head_dim]`
    /// Returns: `[1, num_heads, head_dim]`
    pub fn cpu_decode_attention(
        q: &Tensor, k_cache: &Tensor, v_cache: &Tensor,
        context_len: usize, num_heads: usize, num_kv_heads: usize,
        head_dim: usize, sm_scale: f32,
    ) -> Result<Tensor> {
        debug_assert_eq!(q.dtype(), prelude_core::tensor::DType::BF16, "cpu_decode_attention requires BF16");

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
            &mut out_buf, q_slice, k_slice, v_slice,
            &req_to_token, &seq_lens,
            1, context_len, num_heads, num_kv_heads, head_dim, sm_scale,
        );

        super::u16_vec_to_bf16_tensor(out_buf, &[1, num_heads, head_dim], q.device())
    }

    /// Decode attention: single Q token against contiguous KV cache (F32).
    ///
    /// q: `[1, num_heads, head_dim]`, k/v_cache: `[context_len, num_kv_heads, head_dim]`
    /// Returns: `[1, num_heads, head_dim]`
    pub fn cpu_decode_attention_f32(
        q: &Tensor, k_cache: &Tensor, v_cache: &Tensor,
        context_len: usize, num_heads: usize, num_kv_heads: usize,
        head_dim: usize, sm_scale: f32,
    ) -> Result<Tensor> {
        debug_assert_eq!(q.dtype(), prelude_core::tensor::DType::F32, "cpu_decode_attention_f32 requires F32");

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
            &mut out_buf, q_slice, k_slice, v_slice,
            &req_to_token, &seq_lens,
            1, context_len, num_heads, num_kv_heads, head_dim, sm_scale,
        );

        Tensor::from_vec(out_buf, &[1, num_heads, head_dim], q.device())
    }
}

// ── RoPE Tensor wrappers ────────────────────────────────────────────────

mod rope_tensor {
    use prelude_core::tensor::{Result, Tensor};

    /// Apply NeoX-style RoPE in-place to Q and K tensors (BF16).
    pub fn cpu_rotary_embedding(
        q: &Tensor, k: &Tensor, cos_sin_cache: &Tensor,
        offset: usize, num_heads: usize, num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len) = (q_dims[0], q_dims[1]);
        let positions: Vec<i64> = (0..batch_size)
            .flat_map(|_| (0..seq_len).map(|s| (offset + s) as i64))
            .collect();
        cpu_rotary_embedding_with_positions(q, k, cos_sin_cache, &positions, num_heads, num_kv_heads)
    }

    /// Apply NeoX-style RoPE with explicit per-token positions (BF16).
    pub fn cpu_rotary_embedding_with_positions(
        q: &Tensor, k: &Tensor, cos_sin_cache: &Tensor,
        positions: &[i64], num_heads: usize, num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len, _nh, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let rotary_dim = cos_sin_cache.dims()[1];

        let q_2d = q.contiguous()?.reshape((batch_size * seq_len * num_heads * head_dim,))?;
        let k_2d = k.contiguous()?.reshape((batch_size * seq_len * num_kv_heads * head_dim,))?;
        let cache_slice = super::tensor_as_u16_slice(cos_sin_cache)?;

        {
            let (mut q_guard, q_layout) = unsafe { q_2d.storage_mut_and_layout() };
            let (mut k_guard, k_layout) = unsafe { k_2d.storage_mut_and_layout() };
            let q_slice = unsafe {
                super::extract_bf16_mut_u16(&mut q_guard, q_layout.start_offset(), batch_size * seq_len * num_heads * head_dim)?
            };
            let k_slice = unsafe {
                super::extract_bf16_mut_u16(&mut k_guard, k_layout.start_offset(), batch_size * seq_len * num_kv_heads * head_dim)?
            };

            super::rope::rope_neox_bf16(
                q_slice, k_slice, cache_slice, positions,
                batch_size, seq_len, num_heads, num_kv_heads, head_dim, rotary_dim,
            );
        }

        Ok((q_2d.reshape(q.dims())?, k_2d.reshape(k.dims())?))
    }

    /// Apply NeoX-style RoPE in-place to Q and K tensors (F32).
    pub fn cpu_rotary_embedding_f32(
        q: &Tensor, k: &Tensor, cos_sin_cache: &Tensor,
        offset: usize, num_heads: usize, num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len) = (q_dims[0], q_dims[1]);
        let positions: Vec<i64> = (0..batch_size)
            .flat_map(|_| (0..seq_len).map(|s| (offset + s) as i64))
            .collect();
        cpu_rotary_embedding_f32_with_positions(q, k, cos_sin_cache, &positions, num_heads, num_kv_heads)
    }

    /// Apply NeoX-style RoPE with explicit per-token positions (F32).
    pub fn cpu_rotary_embedding_f32_with_positions(
        q: &Tensor, k: &Tensor, cos_sin_cache: &Tensor,
        positions: &[i64], num_heads: usize, num_kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.dims();
        let (batch_size, seq_len, _nh, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let rotary_dim = cos_sin_cache.dims()[1];

        let q_2d = q.contiguous()?.reshape((batch_size * seq_len * num_heads * head_dim,))?;
        let k_2d = k.contiguous()?.reshape((batch_size * seq_len * num_kv_heads * head_dim,))?;
        let cache_slice = super::tensor_as_f32_slice(cos_sin_cache)?;

        {
            let (mut q_guard, q_layout) = unsafe { q_2d.storage_mut_and_layout() };
            let (mut k_guard, k_layout) = unsafe { k_2d.storage_mut_and_layout() };
            let q_slice = unsafe {
                super::extract_f32_mut(&mut q_guard, q_layout.start_offset(), batch_size * seq_len * num_heads * head_dim)?
            };
            let k_slice = unsafe {
                super::extract_f32_mut(&mut k_guard, k_layout.start_offset(), batch_size * seq_len * num_kv_heads * head_dim)?
            };

            super::rope::rope_neox_f32(
                q_slice, k_slice, cache_slice, positions,
                batch_size, seq_len, num_heads, num_kv_heads, head_dim, rotary_dim,
            );
        }

        Ok((q_2d.reshape(q.dims())?, k_2d.reshape(k.dims())?))
    }
}
