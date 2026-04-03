//! RMSNorm CPU kernels — AVX-512 BF16 + generic scalar for F16/F32.
//!
//! RMSNorm(x, w, eps) = x * w / sqrt(mean(x^2) + eps)
//!
//! Public API (Tensor level):
//! - `cpu_rmsnorm()` — standalone RMSNorm
//! - `cpu_fused_add_rmsnorm()` — fused residual-add + RMSNorm
//! - `CpuRmsNorm` — Module wrapper

use prelude_core::tensor::{DType, Module, Result, Tensor};

use super::bf16_utils::{bf16_to_f32, f32_to_bf16};
#[cfg(target_arch = "x86_64")]
use super::bf16_utils::{bf16x16_load_as_f32, f32x16_store_as_bf16};
use super::cpu_float::CpuFloat;

// ── Tensor-level API ────────────────────────────────────────────────────

/// RMS Normalization: BF16 uses AVX-512 kernel, other dtypes use generic scalar.
/// input: `[..., hidden_size]`, weight: `[hidden_size]`
/// Returns: same shape and dtype as input.
pub fn cpu_rmsnorm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = input.dims();
    let hidden_size = *dims.last().unwrap();
    let batch_size: usize = dims.iter().take(dims.len() - 1).product();

    match input.dtype() {
        DType::BF16 => {
            let input_2d = input.reshape((batch_size, hidden_size))?;
            let in_slice = super::tensor_as_u16_slice(&input_2d)?;
            let w_slice = super::tensor_as_u16_slice(weight)?;
            let mut out_buf = vec![0u16; batch_size * hidden_size];
            rmsnorm_bf16(&mut out_buf, in_slice, w_slice, batch_size, hidden_size, eps as f32);
            super::u16_vec_to_bf16_tensor(out_buf, dims, input.device())
        }
        DType::F32 => {
            let input_2d = input.reshape((batch_size, hidden_size))?.contiguous()?;
            let in_data = input_2d.flatten_all()?.to_vec1::<f32>()?;
            let w_data = weight.to_vec1::<f32>()?;
            let n = batch_size * hidden_size;
            let mut out_buf = vec![0f32; n];
            rmsnorm_f32(&mut out_buf, &in_data[..n], &w_data, batch_size, hidden_size, eps as f32);
            Tensor::from_vec(out_buf, dims, input.device())
        }
        _ => cpu_rmsnorm_typed::<half::f16>(input, weight, batch_size, hidden_size, eps, dims),
    }
}

fn cpu_rmsnorm_typed<T: CpuFloat>(
    input: &Tensor, weight: &Tensor,
    batch_size: usize, hidden_size: usize, eps: f64,
    dims: &[usize],
) -> Result<Tensor> {
    let input_t = input.to_dtype(T::DTYPE)?;
    let weight_t = weight.to_dtype(T::DTYPE)?;
    let input_2d = input_t.reshape((batch_size, hidden_size))?;
    let in_data = T::tensor_to_vec(&input_2d)?;
    let w_data = T::tensor_to_vec(&weight_t)?;
    let n = batch_size * hidden_size;
    let mut out_buf = vec![T::zero(); n];
    let eps_f32 = eps as f32;

    const RAYON_MIN_ELEMS: usize = 16384;
    if !super::should_parallelize(n, batch_size, RAYON_MIN_ELEMS) {
        rmsnorm_generic(&mut out_buf, &in_data[..n], &w_data, batch_size, hidden_size, eps_f32);
    } else {
        use rayon::prelude::*;
        let hs = hidden_size;
        out_buf.par_chunks_mut(hs)
            .zip(in_data[..n].par_chunks(hs))
            .for_each(|(out_row, in_row)| {
                rmsnorm_generic(out_row, in_row, &w_data, 1, hs, eps_f32);
            });
    }

    T::vec_to_tensor(out_buf, dims, input.device())
}

/// Fused residual-add + RMSNorm: BF16 uses AVX-512, other dtypes use generic scalar.
///
/// Returns `(residual_out, normalized)` where:
///   `residual_out = input + residual`
///   `normalized   = rmsnorm(residual_out, weight)`
pub fn cpu_fused_add_rmsnorm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let dims = input.dims();
    let hidden_size = *dims.last().unwrap();
    let batch_size: usize = dims.iter().take(dims.len() - 1).product();
    let n = batch_size * hidden_size;

    match input.dtype() {
        DType::BF16 => {
            let input_c = input.contiguous()?;
            let residual_c = residual.contiguous()?;
            let in_slice = super::tensor_as_u16_slice(&input_c)?;
            let res_slice = super::tensor_as_u16_slice(&residual_c)?;
            let w_slice = super::tensor_as_u16_slice(weight)?;
            let mut norm_buf = Vec::with_capacity(n);
            let mut res_buf = Vec::with_capacity(n);
            // SAFETY: fused_add_rmsnorm_bf16_oop writes every element before returning.
            unsafe { norm_buf.set_len(n); res_buf.set_len(n); }
            fused_add_rmsnorm_bf16_oop(
                &in_slice[..n], &res_slice[..n],
                &mut norm_buf, &mut res_buf,
                w_slice, batch_size, hidden_size, eps as f32,
            );
            let res_out = super::u16_vec_to_bf16_tensor(res_buf, dims, input.device())?;
            let norm_out = super::u16_vec_to_bf16_tensor(norm_buf, dims, input.device())?;
            Ok((res_out, norm_out))
        }
        DType::F32 => {
            let input_c = input.contiguous()?;
            let residual_c = residual.contiguous()?;
            let in_data = input_c.flatten_all()?.to_vec1::<f32>()?;
            let res_data = residual_c.flatten_all()?.to_vec1::<f32>()?;
            let w_data = weight.to_vec1::<f32>()?;
            let mut norm_buf = Vec::with_capacity(n);
            let mut res_buf = Vec::with_capacity(n);
            unsafe { norm_buf.set_len(n); res_buf.set_len(n); }
            fused_add_rmsnorm_f32_oop(
                &in_data[..n], &res_data[..n],
                &mut norm_buf, &mut res_buf,
                &w_data, batch_size, hidden_size, eps as f32,
            );
            let res_out = Tensor::from_vec(res_buf, dims, input.device())?;
            let norm_out = Tensor::from_vec(norm_buf, dims, input.device())?;
            Ok((res_out, norm_out))
        }
        _ => cpu_fused_add_rmsnorm_typed::<half::f16>(input, residual, weight, batch_size, hidden_size, n, eps, dims),
    }
}

fn cpu_fused_add_rmsnorm_typed<T: CpuFloat>(
    input: &Tensor, residual: &Tensor, weight: &Tensor,
    batch_size: usize, hidden_size: usize, n: usize, eps: f64,
    dims: &[usize],
) -> Result<(Tensor, Tensor)> {
    let input_t = input.to_dtype(T::DTYPE)?.contiguous()?;
    let residual_t = residual.to_dtype(T::DTYPE)?.contiguous()?;
    let weight_t = weight.to_dtype(T::DTYPE)?;
    let in_data = T::tensor_to_vec(&input_t)?;
    let res_data = T::tensor_to_vec(&residual_t)?;
    let w_data = T::tensor_to_vec(&weight_t)?;
    let mut norm_buf = vec![T::zero(); n];
    let mut res_buf = vec![T::zero(); n];
    let eps_f32 = eps as f32;

    // Higher threshold than GemmPool: rayon threads may be parked (~20-50µs wake-up vs ~5µs spin).
    const RAYON_MIN_ELEMS: usize = 16384;
    if !super::should_parallelize(n, batch_size, RAYON_MIN_ELEMS) {
        fused_add_rmsnorm_generic_oop(
            &in_data[..n], &res_data[..n], &mut norm_buf, &mut res_buf,
            &w_data, batch_size, hidden_size, eps_f32,
        );
    } else {
        // Parallel: split rows across rayon threads.
        // (GemmPool requires non-generic fn pointers; rayon handles generics via closures.)
        use rayon::prelude::*;
        let hs = hidden_size;
        norm_buf.par_chunks_mut(hs)
            .zip(res_buf.par_chunks_mut(hs))
            .zip(in_data[..n].par_chunks(hs))
            .zip(res_data[..n].par_chunks(hs))
            .for_each(|(((norm_row, res_row), in_row), res_in_row)| {
                fused_add_rmsnorm_generic_oop(in_row, res_in_row, norm_row, res_row, &w_data, 1, hs, eps_f32);
            });
    }

    let res_out = T::vec_to_tensor(res_buf, dims, input.device())?;
    let norm_out = T::vec_to_tensor(norm_buf, dims, input.device())?;
    Ok((res_out, norm_out))
}

/// CPU RMSNorm Module wrapper — dispatches by dtype.
#[derive(Debug, Clone)]
pub struct CpuRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl CpuRmsNorm {
    pub fn new(eps: f64, weight: Tensor) -> Self {
        Self { weight, eps }
    }
}

impl Module for CpuRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        cpu_rmsnorm(x, &self.weight, self.eps)
    }
}

// ── Raw kernel API (u16 slices, used by raw_cpu.rs) ─────────────────────

/// Minimum elements per thread to justify parallelization overhead.
/// With mimalloc (no mmap page faults), GemmPool dispatch is ~5µs.
/// At ~0.28ns/elem (AVX-512 2-pass fused), 2K elems/thread ≈ 0.6µs compute,
/// which is still worthwhile if total serial time exceeds dispatch overhead.
const MIN_ELEMS_PER_THREAD: usize = 2048;

/// RMSNorm for BF16 data.
///
/// `output`, `input`: `[batch_size * hidden_size]` as raw `u16` (BF16 bit pattern).
/// `weight`: `[hidden_size]` as raw `u16`.
pub fn rmsnorm_bf16(
    output: &mut [u16],
    input: &[u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch_size * hidden_size);
    debug_assert_eq!(output.len(), batch_size * hidden_size);
    debug_assert_eq!(weight.len(), hidden_size);

    let total = batch_size * hidden_size;
    if !super::should_parallelize(total, batch_size, MIN_ELEMS_PER_THREAD) {
        rmsnorm_impl(output, input, weight, batch_size, hidden_size, eps);
        return;
    }

    // Use GemmPool (spinning threads) instead of rayon to avoid contention
    #[repr(C)]
    struct Ctx {
        out_ptr: usize, in_ptr: usize, w_ptr: usize,
        batch_size: usize, hidden_size: usize, eps: f32,
    }
    unsafe fn work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const Ctx);
            let rows_per = (c.batch_size + n_threads - 1) / n_threads;
            let start = tid * rows_per;
            let end = (start + rows_per).min(c.batch_size);
            if start >= end { return; }
            let chunk = end - start;
            let off = start * c.hidden_size;
            let out = std::slice::from_raw_parts_mut((c.out_ptr as *mut u16).add(off), chunk * c.hidden_size);
            let inp = std::slice::from_raw_parts((c.in_ptr as *const u16).add(off), chunk * c.hidden_size);
            let w = std::slice::from_raw_parts(c.w_ptr as *const u16, c.hidden_size);
            rmsnorm_impl(out, inp, w, chunk, c.hidden_size, c.eps);
        }
    }
    let ctx = Ctx {
        out_ptr: output.as_mut_ptr() as usize, in_ptr: input.as_ptr() as usize,
        w_ptr: weight.as_ptr() as usize, batch_size, hidden_size, eps,
    };
    let pool = super::gemm_pool::gemm_pool();
    // Adaptive thread count: ensure each thread gets enough work to amortize dispatch overhead
    let max_by_work = total / MIN_ELEMS_PER_THREAD;
    let n = pool.num_threads().min(batch_size).min(max_by_work.max(1));
    unsafe { pool.dispatch(work, &ctx as *const Ctx as *const u8, n); }
}

/// RMSNorm for F32 data. AVX-512 kernel + GemmPool parallel dispatch.
fn rmsnorm_f32(
    output: &mut [f32], input: &[f32], weight: &[f32],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    let total = batch_size * hidden_size;
    if !super::should_parallelize(total, batch_size, MIN_ELEMS_PER_THREAD) {
        rmsnorm_f32_impl(output, input, weight, batch_size, hidden_size, eps);
        return;
    }
    #[repr(C)]
    struct Ctx {
        out_ptr: usize, in_ptr: usize, w_ptr: usize,
        batch_size: usize, hidden_size: usize, eps: f32,
    }
    unsafe fn work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const Ctx);
            let rows_per = (c.batch_size + n_threads - 1) / n_threads;
            let start = tid * rows_per;
            let end = (start + rows_per).min(c.batch_size);
            if start >= end { return; }
            let chunk = end - start;
            let off = start * c.hidden_size;
            let out = std::slice::from_raw_parts_mut((c.out_ptr as *mut f32).add(off), chunk * c.hidden_size);
            let inp = std::slice::from_raw_parts((c.in_ptr as *const f32).add(off), chunk * c.hidden_size);
            let w = std::slice::from_raw_parts(c.w_ptr as *const f32, c.hidden_size);
            rmsnorm_f32_impl(out, inp, w, chunk, c.hidden_size, c.eps);
        }
    }
    let ctx = Ctx {
        out_ptr: output.as_mut_ptr() as usize, in_ptr: input.as_ptr() as usize,
        w_ptr: weight.as_ptr() as usize, batch_size, hidden_size, eps,
    };
    let pool = super::gemm_pool::gemm_pool();
    let max_by_work = total / MIN_ELEMS_PER_THREAD;
    let n = pool.num_threads().min(batch_size).min(max_by_work.max(1));
    unsafe { pool.dispatch(work, &ctx as *const Ctx as *const u8, n); }
}

fn rmsnorm_f32_impl(
    output: &mut [f32], input: &[f32], weight: &[f32],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                return rmsnorm_f32_avx512(output, input, weight, batch_size, hidden_size, eps);
            }
        }
    }
    rmsnorm_generic(output, input, weight, batch_size, hidden_size, eps);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rmsnorm_f32_avx512(
    output: &mut [f32], input: &[f32], weight: &[f32],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    use core::arch::x86_64::*;
    let chunks = hidden_size / 16;
    let inv_n = 1.0f32 / hidden_size as f32;

    for b in 0..batch_size {
        let off = b * hidden_size;
        let in_ptr = input.as_ptr().add(off);
        let out_ptr = output.as_mut_ptr().add(off);
        let w_ptr = weight.as_ptr();

        let mut sum_sq = _mm512_setzero_ps();
        for i in 0..chunks {
            let x = _mm512_loadu_ps(in_ptr.add(i * 16));
            sum_sq = _mm512_fmadd_ps(x, x, sum_sq);
        }
        let mut total_sq = _mm512_reduce_add_ps(sum_sq);
        for j in (chunks * 16)..hidden_size {
            let v = *in_ptr.add(j);
            total_sq += v * v;
        }

        let scale = 1.0f32 / (total_sq * inv_n + eps).sqrt();
        let scale_v = _mm512_set1_ps(scale);

        for i in 0..chunks {
            let x = _mm512_loadu_ps(in_ptr.add(i * 16));
            let w = _mm512_loadu_ps(w_ptr.add(i * 16));
            let r = _mm512_mul_ps(_mm512_mul_ps(x, w), scale_v);
            _mm512_storeu_ps(out_ptr.add(i * 16), r);
        }
        for j in (chunks * 16)..hidden_size {
            output[off + j] = input[off + j] * weight[j] * scale;
        }
    }
}

/// Out-of-place fused residual-add + RMSNorm for BF16 data.
///
/// Reads from `input` and `residual`, writes to `norm_out` and `res_out`.
/// After call:
///   `res_out[j]  = input[j] + residual[j]`
///   `norm_out[j] = rmsnorm(res_out, weight)[j]`
///
/// Avoids the `to_vec()` copy of the in-place variant, which triggers mmap +
/// serial page faults for large allocations (>128 KB in glibc malloc).
fn fused_add_rmsnorm_bf16_oop(
    input: &[u16],
    residual: &[u16],
    norm_out: &mut [u16],
    res_out: &mut [u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch_size * hidden_size);
    debug_assert_eq!(residual.len(), batch_size * hidden_size);
    debug_assert_eq!(norm_out.len(), batch_size * hidden_size);
    debug_assert_eq!(res_out.len(), batch_size * hidden_size);
    debug_assert_eq!(weight.len(), hidden_size);

    let total = batch_size * hidden_size;
    if !super::should_parallelize(total, batch_size, MIN_ELEMS_PER_THREAD) {
        fused_add_rmsnorm_oop_impl(input, residual, norm_out, res_out, weight, batch_size, hidden_size, eps);
        return;
    }

    #[repr(C)]
    struct Ctx {
        in_ptr: usize, res_ptr: usize,
        nout_ptr: usize, rout_ptr: usize,
        w_ptr: usize,
        batch_size: usize, hidden_size: usize, eps: f32,
    }
    unsafe fn work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const Ctx);
            let rows_per = (c.batch_size + n_threads - 1) / n_threads;
            let start = tid * rows_per;
            let end = (start + rows_per).min(c.batch_size);
            if start >= end { return; }
            let chunk = end - start;
            let off = start * c.hidden_size;
            let inp = std::slice::from_raw_parts((c.in_ptr as *const u16).add(off), chunk * c.hidden_size);
            let res = std::slice::from_raw_parts((c.res_ptr as *const u16).add(off), chunk * c.hidden_size);
            let nout = std::slice::from_raw_parts_mut((c.nout_ptr as *mut u16).add(off), chunk * c.hidden_size);
            let rout = std::slice::from_raw_parts_mut((c.rout_ptr as *mut u16).add(off), chunk * c.hidden_size);
            let w = std::slice::from_raw_parts(c.w_ptr as *const u16, c.hidden_size);
            fused_add_rmsnorm_oop_impl(inp, res, nout, rout, w, chunk, c.hidden_size, c.eps);
        }
    }
    let ctx = Ctx {
        in_ptr: input.as_ptr() as usize, res_ptr: residual.as_ptr() as usize,
        nout_ptr: norm_out.as_mut_ptr() as usize, rout_ptr: res_out.as_mut_ptr() as usize,
        w_ptr: weight.as_ptr() as usize, batch_size, hidden_size, eps,
    };
    let pool = super::gemm_pool::gemm_pool();
    let max_by_work = total / MIN_ELEMS_PER_THREAD;
    let n = pool.num_threads().min(batch_size).min(max_by_work.max(1));
    unsafe { pool.dispatch(work, &ctx as *const Ctx as *const u8, n); }
}

/// Out-of-place fused residual-add + RMSNorm for F32 data.
/// Same structure as the BF16 variant: AVX-512 kernel + GemmPool parallel dispatch.
fn fused_add_rmsnorm_f32_oop(
    input: &[f32], residual: &[f32],
    norm_out: &mut [f32], res_out: &mut [f32],
    weight: &[f32],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    let total = batch_size * hidden_size;
    if !super::should_parallelize(total, batch_size, MIN_ELEMS_PER_THREAD) {
        fused_add_rmsnorm_f32_oop_impl(input, residual, norm_out, res_out, weight, batch_size, hidden_size, eps);
        return;
    }
    #[repr(C)]
    struct Ctx {
        in_ptr: usize, res_ptr: usize,
        nout_ptr: usize, rout_ptr: usize,
        w_ptr: usize,
        batch_size: usize, hidden_size: usize, eps: f32,
    }
    unsafe fn work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const Ctx);
            let rows_per = (c.batch_size + n_threads - 1) / n_threads;
            let start = tid * rows_per;
            let end = (start + rows_per).min(c.batch_size);
            if start >= end { return; }
            let chunk = end - start;
            let off = start * c.hidden_size;
            let inp = std::slice::from_raw_parts((c.in_ptr as *const f32).add(off), chunk * c.hidden_size);
            let res = std::slice::from_raw_parts((c.res_ptr as *const f32).add(off), chunk * c.hidden_size);
            let nout = std::slice::from_raw_parts_mut((c.nout_ptr as *mut f32).add(off), chunk * c.hidden_size);
            let rout = std::slice::from_raw_parts_mut((c.rout_ptr as *mut f32).add(off), chunk * c.hidden_size);
            let w = std::slice::from_raw_parts(c.w_ptr as *const f32, c.hidden_size);
            fused_add_rmsnorm_f32_oop_impl(inp, res, nout, rout, w, chunk, c.hidden_size, c.eps);
        }
    }
    let ctx = Ctx {
        in_ptr: input.as_ptr() as usize, res_ptr: residual.as_ptr() as usize,
        nout_ptr: norm_out.as_mut_ptr() as usize, rout_ptr: res_out.as_mut_ptr() as usize,
        w_ptr: weight.as_ptr() as usize, batch_size, hidden_size, eps,
    };
    let pool = super::gemm_pool::gemm_pool();
    let max_by_work = total / MIN_ELEMS_PER_THREAD;
    let n = pool.num_threads().min(batch_size).min(max_by_work.max(1));
    unsafe { pool.dispatch(work, &ctx as *const Ctx as *const u8, n); }
}

fn fused_add_rmsnorm_f32_oop_impl(
    input: &[f32], residual: &[f32],
    norm_out: &mut [f32], res_out: &mut [f32],
    weight: &[f32],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                return fused_add_rmsnorm_f32_oop_avx512(
                    input, residual, norm_out, res_out, weight, batch_size, hidden_size, eps,
                );
            }
        }
    }
    // Scalar fallback
    let inv_n = 1.0f32 / hidden_size as f32;
    for b in 0..batch_size {
        let off = b * hidden_size;
        let mut sum_sq = 0.0f32;
        for j in 0..hidden_size {
            let added = input[off + j] + residual[off + j];
            sum_sq += added * added;
        }
        let scale = 1.0 / (sum_sq * inv_n + eps).sqrt();
        for j in 0..hidden_size {
            let added = input[off + j] + residual[off + j];
            res_out[off + j] = added;
            norm_out[off + j] = added * weight[j] * scale;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_add_rmsnorm_f32_oop_avx512(
    input: &[f32], residual: &[f32],
    norm_out: &mut [f32], res_out: &mut [f32],
    weight: &[f32],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    use core::arch::x86_64::*;

    let chunks = hidden_size / 16;
    let inv_n = 1.0f32 / hidden_size as f32;

    for b in 0..batch_size {
        let off = b * hidden_size;
        let in_ptr = input.as_ptr().add(off);
        let res_ptr = residual.as_ptr().add(off);
        let nout_ptr = norm_out.as_mut_ptr().add(off);
        let rout_ptr = res_out.as_mut_ptr().add(off);
        let w_ptr = weight.as_ptr();

        // Pass 1: sum of squares
        let mut sum_sq = _mm512_setzero_ps();
        for i in 0..chunks {
            let x = _mm512_loadu_ps(in_ptr.add(i * 16));
            let r = _mm512_loadu_ps(res_ptr.add(i * 16));
            let added = _mm512_add_ps(x, r);
            sum_sq = _mm512_fmadd_ps(added, added, sum_sq);
        }
        let mut total_sq = _mm512_reduce_add_ps(sum_sq);
        for j in (chunks * 16)..hidden_size {
            let added = input[off + j] + residual[off + j];
            total_sq += added * added;
        }

        let scale = 1.0f32 / (total_sq * inv_n + eps).sqrt();
        let scale_v = _mm512_set1_ps(scale);

        // Pass 2: write to separate output buffers
        for i in 0..chunks {
            let x = _mm512_loadu_ps(in_ptr.add(i * 16));
            let r = _mm512_loadu_ps(res_ptr.add(i * 16));
            let added = _mm512_add_ps(x, r);
            let w = _mm512_loadu_ps(w_ptr.add(i * 16));
            _mm512_storeu_ps(rout_ptr.add(i * 16), added);
            let result = _mm512_mul_ps(_mm512_mul_ps(added, w), scale_v);
            _mm512_storeu_ps(nout_ptr.add(i * 16), result);
        }
        for j in (chunks * 16)..hidden_size {
            let added = input[off + j] + residual[off + j];
            res_out[off + j] = added;
            norm_out[off + j] = added * weight[j] * scale;
        }
    }
}

// ── Implementation dispatch (feature detect once, loop inside) ──────────

pub(crate) fn rmsnorm_impl(
    output: &mut [u16],
    input: &[u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                return rmsnorm_bf16_avx512(output, input, weight, batch_size, hidden_size, eps);
            }
        }
    }
    rmsnorm_bf16_scalar_via_generic(output, input, weight, batch_size, hidden_size, eps);
}

// ── Scalar fallback (delegates to generic kernels via CpuFloat) ─────────

fn rmsnorm_bf16_scalar_via_generic(
    output: &mut [u16],
    input: &[u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let out = unsafe { &mut *(output as *mut [u16] as *mut [half::bf16]) };
    let inp = unsafe { &*(input as *const [u16] as *const [half::bf16]) };
    let w = unsafe { &*(weight as *const [u16] as *const [half::bf16]) };
    rmsnorm_generic(out, inp, w, batch_size, hidden_size, eps);
}


// ── Generic scalar kernels (any CpuFloat dtype) ────────────────────────

pub(crate) fn rmsnorm_generic<T: CpuFloat>(
    output: &mut [T],
    input: &[T],
    weight: &[T],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let inv_n = 1.0f32 / hidden_size as f32;
    for b in 0..batch_size {
        let off = b * hidden_size;
        let row = &input[off..off + hidden_size];
        let out = &mut output[off..off + hidden_size];
        let mut sum_sq = 0.0f32;
        for j in 0..hidden_size {
            let v = row[j].to_f32();
            sum_sq += v * v;
        }
        let scale = 1.0 / (sum_sq * inv_n + eps).sqrt();
        for j in 0..hidden_size {
            out[j] = T::from_f32(row[j].to_f32() * weight[j].to_f32() * scale);
        }
    }
}

pub(crate) fn fused_add_rmsnorm_generic_oop<T: CpuFloat>(
    input: &[T],
    residual: &[T],
    norm_out: &mut [T],
    res_out: &mut [T],
    weight: &[T],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let inv_n = 1.0f32 / hidden_size as f32;
    for b in 0..batch_size {
        let off = b * hidden_size;
        let mut sum_sq = 0.0f32;
        for j in 0..hidden_size {
            let added = input[off + j].to_f32() + residual[off + j].to_f32();
            sum_sq += added * added;
        }
        let scale = 1.0 / (sum_sq * inv_n + eps).sqrt();
        for j in 0..hidden_size {
            let added = input[off + j].to_f32() + residual[off + j].to_f32();
            res_out[off + j] = T::from_f32(added);
            norm_out[off + j] = T::from_f32(added * weight[j].to_f32() * scale);
        }
    }
}

// ── AVX-512 implementation ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn rmsnorm_bf16_avx512(
    output: &mut [u16],
    input: &[u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    use core::arch::x86_64::*;

    let chunks = hidden_size / 16;
    let inv_n = 1.0f32 / hidden_size as f32;

    for b in 0..batch_size {
        let off = b * hidden_size;
        let in_ptr = unsafe { input.as_ptr().add(off) };
        let out_ptr = unsafe { output.as_mut_ptr().add(off) };
        let w_ptr = weight.as_ptr();

        // Pass 1: sum of squares in F32
        let mut sum_sq = _mm512_setzero_ps();
        for i in 0..chunks {
            let x = bf16x16_load_as_f32(unsafe { in_ptr.add(i * 16) });
            sum_sq = _mm512_fmadd_ps(x, x, sum_sq);
        }
        let mut total_sq = _mm512_reduce_add_ps(sum_sq);
        for j in (chunks * 16)..hidden_size {
            let v = bf16_to_f32(unsafe { *in_ptr.add(j) });
            total_sq += v * v;
        }

        let scale = 1.0f32 / (total_sq * inv_n + eps).sqrt();
        let scale_v = _mm512_set1_ps(scale);

        // Pass 2: output = input * weight * scale
        for i in 0..chunks {
            let x = bf16x16_load_as_f32(unsafe { in_ptr.add(i * 16) });
            let w = bf16x16_load_as_f32(unsafe { w_ptr.add(i * 16) });
            let r = _mm512_mul_ps(_mm512_mul_ps(x, w), scale_v);
            f32x16_store_as_bf16(unsafe { out_ptr.add(i * 16) }, r);
        }
        for j in (chunks * 16)..hidden_size {
            output[off + j] =
                f32_to_bf16(bf16_to_f32(input[off + j]) * bf16_to_f32(weight[j]) * scale);
        }
    }
}

// ── Out-of-place fused kernel ───────────────────────────────────────────

/// Out-of-place variant: reads from `input` and `residual`, writes to `norm_out` and `res_out`.
/// After: `res_out = input + residual`, `norm_out = rmsnorm(res_out, weight)`.
fn fused_add_rmsnorm_oop_impl(
    input: &[u16],
    residual: &[u16],
    norm_out: &mut [u16],
    res_out: &mut [u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                return fused_add_rmsnorm_oop_avx512(
                    input, residual, norm_out, res_out, weight, batch_size, hidden_size, eps,
                );
            }
        }
    }
    fused_add_rmsnorm_oop_scalar(input, residual, norm_out, res_out, weight, batch_size, hidden_size, eps);
}

fn fused_add_rmsnorm_oop_scalar(
    input: &[u16], residual: &[u16],
    norm_out: &mut [u16], res_out: &mut [u16],
    weight: &[u16],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    let inv_n = 1.0f32 / hidden_size as f32;
    for b in 0..batch_size {
        let off = b * hidden_size;
        let mut sum_sq = 0.0f32;
        for j in 0..hidden_size {
            let added = bf16_to_f32(input[off + j]) + bf16_to_f32(residual[off + j]);
            sum_sq += added * added;
        }
        let scale = 1.0 / (sum_sq * inv_n + eps).sqrt();
        for j in 0..hidden_size {
            let added = bf16_to_f32(input[off + j]) + bf16_to_f32(residual[off + j]);
            res_out[off + j] = f32_to_bf16(added);
            norm_out[off + j] = f32_to_bf16(added * bf16_to_f32(weight[j]) * scale);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn fused_add_rmsnorm_oop_avx512(
    input: &[u16], residual: &[u16],
    norm_out: &mut [u16], res_out: &mut [u16],
    weight: &[u16],
    batch_size: usize, hidden_size: usize, eps: f32,
) {
    use core::arch::x86_64::*;

    let chunks = hidden_size / 16;
    let inv_n = 1.0f32 / hidden_size as f32;

    for b in 0..batch_size {
        let off = b * hidden_size;
        let in_ptr = unsafe { input.as_ptr().add(off) };
        let res_ptr = unsafe { residual.as_ptr().add(off) };
        let nout_ptr = unsafe { norm_out.as_mut_ptr().add(off) };
        let rout_ptr = unsafe { res_out.as_mut_ptr().add(off) };
        let w_ptr = weight.as_ptr();

        // Pass 1: sum of squares
        let mut sum_sq = _mm512_setzero_ps();
        for i in 0..chunks {
            let x = bf16x16_load_as_f32(unsafe { in_ptr.add(i * 16) });
            let r = bf16x16_load_as_f32(unsafe { res_ptr.add(i * 16) });
            let added = _mm512_add_ps(x, r);
            sum_sq = _mm512_fmadd_ps(added, added, sum_sq);
        }
        let mut total_sq = _mm512_reduce_add_ps(sum_sq);
        for j in (chunks * 16)..hidden_size {
            let added = bf16_to_f32(input[off + j]) + bf16_to_f32(residual[off + j]);
            total_sq += added * added;
        }

        let scale = 1.0f32 / (total_sq * inv_n + eps).sqrt();
        let scale_v = _mm512_set1_ps(scale);

        // Pass 2: re-read input+residual (still in L1), write to separate output buffers
        for i in 0..chunks {
            let x = bf16x16_load_as_f32(unsafe { in_ptr.add(i * 16) });
            let r = bf16x16_load_as_f32(unsafe { res_ptr.add(i * 16) });
            let added = _mm512_add_ps(x, r);
            let w = bf16x16_load_as_f32(unsafe { w_ptr.add(i * 16) });
            f32x16_store_as_bf16(unsafe { rout_ptr.add(i * 16) }, added);
            let result = _mm512_mul_ps(_mm512_mul_ps(added, w), scale_v);
            f32x16_store_as_bf16(unsafe { nout_ptr.add(i * 16) }, result);
        }
        for j in (chunks * 16)..hidden_size {
            let added = bf16_to_f32(input[off + j]) + bf16_to_f32(residual[off + j]);
            res_out[off + j] = f32_to_bf16(added);
            norm_out[off + j] = f32_to_bf16(added * bf16_to_f32(weight[j]) * scale);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::bf16_utils::{make_bf16_vec, to_f32_vec, bf16_to_f32, f32_to_bf16};
    use super::super::max_sglang_violation;

    /// Reference RMSNorm in f64 for accuracy comparison.
    fn rmsnorm_f64_ref(input: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
        let n = input.len();
        let sum_sq: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let scale = 1.0 / (sum_sq / n as f64 + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| (x as f64 * w as f64 * scale) as f32)
            .collect()
    }

    #[test]
    fn test_rmsnorm_scalar_basic() {
        let hidden = 64;
        let input_f32: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01) - 0.32).collect();
        let weight_f32: Vec<f32> = (0..hidden).map(|i| 0.5 + i as f32 * 0.01).collect();
        let eps = 1e-6;

        let input = make_bf16_vec(&input_f32);
        let weight = make_bf16_vec(&weight_f32);
        let mut output = vec![0u16; hidden];

        rmsnorm_bf16_scalar_via_generic(&mut output, &input, &weight, 1, hidden, eps as f32);

        let input_rebf16: Vec<f32> = to_f32_vec(&input);
        let weight_rebf16: Vec<f32> = to_f32_vec(&weight);
        let expected = rmsnorm_f64_ref(&input_rebf16, &weight_rebf16, eps);
        let actual = to_f32_vec(&output);

        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "scalar rmsnorm worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }

    #[test]
    fn test_rmsnorm_dispatch_matches_scalar() {
        let hidden = 128;
        let batch = 4;
        let input_f32: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32 * 0.007) - 0.5).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden).map(|i| 0.8 + i as f32 * 0.003).collect();

        let input = make_bf16_vec(&input_f32);
        let weight = make_bf16_vec(&weight_f32);

        // Scalar
        let mut out_scalar = vec![0u16; batch * hidden];
        rmsnorm_bf16_scalar_via_generic(&mut out_scalar, &input, &weight, batch, hidden, 1e-6);

        // Dispatched
        let mut out_dispatch = vec![0u16; batch * hidden];
        rmsnorm_bf16(&mut out_dispatch, &input, &weight, batch, hidden, 1e-6);

        assert_eq!(out_scalar, out_dispatch, "dispatch should match scalar");
    }

    #[test]
    fn test_rmsnorm_large_batch_parallel() {
        let hidden = 128;
        let batch = 128; // above PARALLEL_BATCH_THRESHOLD
        let input_f32: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32 * 0.007) - 0.5).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden).map(|i| 0.8 + i as f32 * 0.003).collect();

        let input = make_bf16_vec(&input_f32);
        let weight = make_bf16_vec(&weight_f32);

        // Sequential (using impl directly)
        let mut out_seq = vec![0u16; batch * hidden];
        rmsnorm_impl(&mut out_seq, &input, &weight, batch, hidden, 1e-6);

        // Parallel (via public API)
        let mut out_par = vec![0u16; batch * hidden];
        rmsnorm_bf16(&mut out_par, &input, &weight, batch, hidden, 1e-6);

        assert_eq!(out_seq, out_par, "parallel should match sequential");
    }

    #[test]
    fn test_fused_add_rmsnorm_scalar() {
        let hidden = 64;
        let h_f32: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.01).collect();
        let res_f32: Vec<f32> = (0..hidden).map(|i| -(i as f32) * 0.005).collect();
        let weight_f32: Vec<f32> = vec![1.0; hidden];
        let eps = 1e-6f32;

        let input = make_bf16_vec(&h_f32);
        let residual = make_bf16_vec(&res_f32);
        let weight = make_bf16_vec(&weight_f32);
        let mut norm_out = vec![0u16; hidden];
        let mut res_out = vec![0u16; hidden];

        fused_add_rmsnorm_oop_scalar(&input, &residual, &mut norm_out, &mut res_out, &weight, 1, hidden, eps);

        let actual: Vec<f32> = (0..hidden).map(|j| bf16_to_f32(res_out[j])).collect();
        let expected: Vec<f32> = (0..hidden).map(|j| h_f32[j] + res_f32[j]).collect();
        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "fused residual worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }

    #[test]
    fn test_fused_add_rmsnorm_dispatch_matches_scalar() {
        let hidden = 128;
        let batch = 2;
        let n = batch * hidden;

        let h_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013) - 0.3).cos()).collect();
        let res_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1).sin()).collect();
        let weight_f32: Vec<f32> = (0..hidden).map(|i| 0.9 + i as f32 * 0.002).collect();
        let weight = make_bf16_vec(&weight_f32);

        // Scalar
        let inp = make_bf16_vec(&h_f32);
        let res = make_bf16_vec(&res_f32);
        let mut norm_s = vec![0u16; n];
        let mut res_s = vec![0u16; n];
        fused_add_rmsnorm_oop_scalar(&inp, &res, &mut norm_s, &mut res_s, &weight, batch, hidden, 1e-6);

        // Dispatched (via oop parallel path)
        let mut norm_d = vec![0u16; n];
        let mut res_d = vec![0u16; n];
        fused_add_rmsnorm_bf16_oop(&inp, &res, &mut norm_d, &mut res_d, &weight, batch, hidden, 1e-6);

        assert_eq!(norm_s, norm_d, "fused norm should match");
        assert_eq!(res_s, res_d, "fused residual should match");
    }

    /// Verify rmsnorm at realistic model dimensions against F64 reference.
    fn verify_rmsnorm_config(hidden: usize, batch: usize, label: &str) {
        let eps = 1e-6;
        let input_f32: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32 * 0.007) - 0.5).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden)
            .map(|i| 0.8 + (i as f32 * 0.003).sin() * 0.5)
            .collect();

        let input = make_bf16_vec(&input_f32);
        let weight = make_bf16_vec(&weight_f32);
        let mut output = vec![0u16; batch * hidden];

        rmsnorm_bf16(&mut output, &input, &weight, batch, hidden, eps as f32);

        // Verify each row against F64 reference
        for row in 0..batch {
            let row_in: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(input[row * hidden + d]))
                .collect();
            let row_w: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(weight[d]))
                .collect();
            let expected = rmsnorm_f64_ref(&row_in, &row_w, eps);
            let actual: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(output[row * hidden + d]))
                .collect();

            let violation = max_sglang_violation(&actual, &expected);
            assert!(
                violation <= 0.0,
                "{label} row={row} worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
            );
        }
    }

    #[test]
    fn test_rmsnorm_realistic_configs() {
        // Qwen3-0.6B: hidden=896
        verify_rmsnorm_config(896, 1, "0.6B batch=1");
        verify_rmsnorm_config(896, 64, "0.6B batch=64");
        // Qwen3-0.6B MLP intermediate: 4864
        verify_rmsnorm_config(4864, 1, "0.6B-mlp batch=1");
        verify_rmsnorm_config(4864, 16, "0.6B-mlp batch=16");
        // Qwen3-32B: hidden=7168
        verify_rmsnorm_config(7168, 1, "32B batch=1");
        verify_rmsnorm_config(7168, 16, "32B batch=16");
    }

    /// Verify fused_add_rmsnorm at realistic dimensions.
    fn verify_fused_config(hidden: usize, batch: usize, label: &str) {
        let eps = 1e-6;
        let h_f32: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32 * 0.013) - 0.3).cos())
            .collect();
        let res_f32: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32 * 0.007) + 0.1).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden)
            .map(|i| 0.9 + (i as f32 * 0.002).sin() * 0.3)
            .collect();
        let weight = make_bf16_vec(&weight_f32);

        let input = make_bf16_vec(&h_f32);
        let residual = make_bf16_vec(&res_f32);
        let n = batch * hidden;
        let mut norm_out = vec![0u16; n];
        let mut res_out = vec![0u16; n];
        fused_add_rmsnorm_bf16_oop(&input, &residual, &mut norm_out, &mut res_out, &weight, batch, hidden, eps as f32);

        // Verify res_out = h + res (SGLang tolerance)
        for row in 0..batch {
            let actual: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(res_out[row * hidden + d]))
                .collect();
            let expected: Vec<f32> = (0..hidden)
                .map(|d| {
                    let idx = row * hidden + d;
                    bf16_to_f32(make_bf16_vec(&h_f32)[idx])
                        + bf16_to_f32(make_bf16_vec(&res_f32)[idx])
                })
                .collect();
            let violation = max_sglang_violation(&actual, &expected);
            assert!(
                violation <= 0.0,
                "{label} residual row={row} worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
            );
        }

        // Verify rmsnorm(res_out) output against F64 reference (SGLang tolerance)
        for row in 0..batch {
            let row_res: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(res_out[row * hidden + d]))
                .collect();
            let row_w: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(weight[d]))
                .collect();
            let expected = rmsnorm_f64_ref(&row_res, &row_w, eps);
            let actual: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(norm_out[row * hidden + d]))
                .collect();

            let violation = max_sglang_violation(&actual, &expected);
            assert!(
                violation <= 0.0,
                "{label} rmsnorm row={row} worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
            );
        }
    }

    #[test]
    fn test_fused_add_rmsnorm_realistic_configs() {
        verify_fused_config(896, 1, "0.6B batch=1");
        verify_fused_config(896, 64, "0.6B batch=64");
        verify_fused_config(4864, 16, "0.6B-mlp batch=16");
        verify_fused_config(7168, 1, "32B batch=1");
        verify_fused_config(7168, 16, "32B batch=16");
    }

    /// Test cpu_fused_add_rmsnorm Tensor API across BF16, F16, and F32 dtypes.
    /// All three should produce consistent results against the F64 reference.
    fn verify_fused_tensor_api(hidden: usize, batch: usize, dtype: prelude_core::tensor::DType, label: &str) {
        let eps = 1e-6;
        let device = prelude_core::tensor::Device::Cpu;
        let n = batch * hidden;

        let h_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013) - 0.3).cos()).collect();
        let res_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1).sin()).collect();
        let w_f32: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32 * 0.002).sin() * 0.3).collect();

        let h_t = prelude_core::tensor::Tensor::from_vec(h_f32.clone(), (batch, hidden), &device).unwrap()
            .to_dtype(dtype).unwrap();
        let r_t = prelude_core::tensor::Tensor::from_vec(res_f32.clone(), (batch, hidden), &device).unwrap()
            .to_dtype(dtype).unwrap();
        let w_t = prelude_core::tensor::Tensor::from_vec(w_f32.clone(), (hidden,), &device).unwrap()
            .to_dtype(dtype).unwrap();

        let (res_out, norm_out) = super::cpu_fused_add_rmsnorm(&h_t, &r_t, &w_t, eps).unwrap();
        assert_eq!(res_out.dtype(), dtype);
        assert_eq!(norm_out.dtype(), dtype);

        let res_out_f32 = res_out.to_dtype(prelude_core::tensor::DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let norm_out_f32 = norm_out.to_dtype(prelude_core::tensor::DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for row in 0..batch {
            // Check residual = input + residual
            let actual_res: Vec<f32> = res_out_f32[row*hidden..(row+1)*hidden].to_vec();
            let expected_res: Vec<f32> = (0..hidden).map(|d| h_f32[row*hidden+d] + res_f32[row*hidden+d]).collect();
            let violation = max_sglang_violation(&actual_res, &expected_res);
            assert!(violation <= 0.0,
                "{label} residual row={row} violation={violation:.6}");

            // Check norm = rmsnorm(residual_out, weight)
            let expected_norm = rmsnorm_f64_ref(&actual_res, &w_f32, eps);
            let actual_norm: Vec<f32> = norm_out_f32[row*hidden..(row+1)*hidden].to_vec();
            let violation = max_sglang_violation(&actual_norm, &expected_norm);
            assert!(violation <= 0.0,
                "{label} norm row={row} violation={violation:.6}");
        }
    }

    #[test]
    fn test_fused_tensor_api_bf16() {
        verify_fused_tensor_api(128, 4, prelude_core::tensor::DType::BF16, "BF16 128x4");
        verify_fused_tensor_api(896, 1, prelude_core::tensor::DType::BF16, "BF16 896x1");
    }

    #[test]
    fn test_fused_tensor_api_f16() {
        verify_fused_tensor_api(128, 4, prelude_core::tensor::DType::F16, "F16 128x4");
        verify_fused_tensor_api(896, 1, prelude_core::tensor::DType::F16, "F16 896x1");
    }

    #[test]
    fn test_fused_tensor_api_f32() {
        verify_fused_tensor_api(128, 4, prelude_core::tensor::DType::F32, "F32 128x4");
        verify_fused_tensor_api(896, 1, prelude_core::tensor::DType::F32, "F32 896x1");
    }

    /// Verify F32 AVX-512 kernel matches scalar within tolerance.
    /// (Not bit-exact: AVX-512 `_mm512_reduce_add_ps` sums in different order than serial loop.)
    #[test]
    fn test_fused_f32_avx512_matches_scalar() {
        for &(hidden, batch) in &[(128, 2), (4864, 4), (7168, 1)] {
            let n = batch * hidden;
            let inp: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013) - 0.3).cos()).collect();
            let res: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1).sin()).collect();
            let w: Vec<f32> = (0..hidden).map(|i| 0.9 + i as f32 * 0.002).collect();

            // Scalar reference
            let mut norm_s = vec![0f32; n];
            let mut res_s = vec![0f32; n];
            fused_add_rmsnorm_generic_oop(&inp, &res, &mut norm_s, &mut res_s, &w, batch, hidden, 1e-6);

            // AVX-512 path
            let mut norm_d = vec![0f32; n];
            let mut res_d = vec![0f32; n];
            fused_add_rmsnorm_f32_oop_impl(&inp, &res, &mut norm_d, &mut res_d, &w, batch, hidden, 1e-6);

            // res_out = input + residual is exact (no reordering)
            assert_eq!(res_s, res_d, "F32 res mismatch at {batch}x{hidden}");
            // norm_out has rounding diff from sum_sq reduction order
            let violation = max_sglang_violation(&norm_d, &norm_s);
            assert!(violation <= 0.0,
                "F32 norm mismatch at {batch}x{hidden}: violation={violation:.6}");
        }
    }

    /// Verify F32 parallel dispatch matches serial result.
    #[test]
    fn test_fused_f32_parallel_matches_serial() {
        let hidden = 4864;
        let batch = 256;
        let n = batch * hidden;
        let inp: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013) - 0.3).cos()).collect();
        let res: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1).sin()).collect();
        let w: Vec<f32> = (0..hidden).map(|i| 0.9 + i as f32 * 0.002).collect();

        // Serial
        let mut norm_s = vec![0f32; n];
        let mut res_s = vec![0f32; n];
        fused_add_rmsnorm_f32_oop_impl(&inp, &res, &mut norm_s, &mut res_s, &w, batch, hidden, 1e-6);

        // Parallel (via dispatch wrapper)
        let mut norm_p = vec![0f32; n];
        let mut res_p = vec![0f32; n];
        fused_add_rmsnorm_f32_oop(&inp, &res, &mut norm_p, &mut res_p, &w, batch, hidden, 1e-6);

        assert_eq!(norm_s, norm_p, "F32 parallel norm mismatch");
        assert_eq!(res_s, res_p, "F32 parallel res mismatch");
    }
}
