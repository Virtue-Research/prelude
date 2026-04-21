//! RMSNorm CPU kernels — AVX-512 BF16 + generic scalar for F16/F32.
//!
//! RMSNorm(x, w, eps) = x * w / sqrt(mean(x^2) + eps)
//!
//! Public API (Tensor level):
//! - `cpu_rmsnorm()` — standalone RMSNorm
//! - `cpu_fused_add_rmsnorm()` — fused residual-add + RMSNorm
//! - `CpuRmsNorm` — Module wrapper

use candle_core::{DType, Module, Result, Tensor};

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
        DType::F16 => cpu_rmsnorm_typed::<half::f16>(input, weight, batch_size, hidden_size, eps, dims),
        _ => cpu_rmsnorm_typed::<f32>(input, weight, batch_size, hidden_size, eps, dims),
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
    rmsnorm_generic(&mut out_buf, &in_data[..n], &w_data, batch_size, hidden_size, eps as f32);
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
            let mut norm_buf: Vec<u16> = in_slice[..n].to_vec();
            let mut res_buf: Vec<u16> = res_slice[..n].to_vec();
            fused_add_rmsnorm_bf16(
                &mut norm_buf, &mut res_buf, w_slice, batch_size, hidden_size, eps as f32,
            );
            let res_out = super::u16_vec_to_bf16_tensor(res_buf, dims, input.device())?;
            let norm_out = super::u16_vec_to_bf16_tensor(norm_buf, dims, input.device())?;
            Ok((res_out, norm_out))
        }
        DType::F16 => cpu_fused_add_rmsnorm_typed::<half::f16>(input, residual, weight, batch_size, hidden_size, n, eps, dims),
        _ => cpu_fused_add_rmsnorm_typed::<f32>(input, residual, weight, batch_size, hidden_size, n, eps, dims),
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
    let mut norm_buf: Vec<T> = in_data[..n].to_vec();
    let mut res_buf: Vec<T> = res_data[..n].to_vec();
    fused_add_rmsnorm_generic(
        &mut norm_buf, &mut res_buf, &w_data, batch_size, hidden_size, eps as f32,
    );
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
/// On high core-count systems (64+ cores), pool-wide overhead is
/// ~50-100µs. At ~0.14ns/elem (AVX-512 2-pass), 16K elems/thread ≈ 2.3µs.
/// Combined with the pool-wide floor, this prevents over-parallelization.
const MIN_ELEMS_PER_THREAD: usize = 16384;

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

/// Fused residual-add + RMSNorm for BF16 data (in-place).
///
/// After call:
///   `residual[j] = input[j] + residual[j]`  (updated in-place)
///   `input[j]    = rmsnorm(residual, weight)[j]`  (overwritten in-place)
pub fn fused_add_rmsnorm_bf16(
    input: &mut [u16],
    residual: &mut [u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    debug_assert_eq!(input.len(), batch_size * hidden_size);
    debug_assert_eq!(residual.len(), batch_size * hidden_size);
    debug_assert_eq!(weight.len(), hidden_size);

    let total = batch_size * hidden_size;
    if !super::should_parallelize(total, batch_size, MIN_ELEMS_PER_THREAD) {
        fused_add_rmsnorm_impl(input, residual, weight, batch_size, hidden_size, eps);
        return;
    }

    #[repr(C)]
    struct Ctx {
        in_ptr: usize, res_ptr: usize, w_ptr: usize,
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
            let inp = std::slice::from_raw_parts_mut((c.in_ptr as *mut u16).add(off), chunk * c.hidden_size);
            let res = std::slice::from_raw_parts_mut((c.res_ptr as *mut u16).add(off), chunk * c.hidden_size);
            let w = std::slice::from_raw_parts(c.w_ptr as *const u16, c.hidden_size);
            fused_add_rmsnorm_impl(inp, res, w, chunk, c.hidden_size, c.eps);
        }
    }
    let ctx = Ctx {
        in_ptr: input.as_mut_ptr() as usize, res_ptr: residual.as_mut_ptr() as usize,
        w_ptr: weight.as_ptr() as usize, batch_size, hidden_size, eps,
    };
    let pool = super::gemm_pool::gemm_pool();
    let max_by_work = total / MIN_ELEMS_PER_THREAD;
    let n = pool.num_threads().min(batch_size).min(max_by_work.max(1));
    unsafe { pool.dispatch(work, &ctx as *const Ctx as *const u8, n); }
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

fn fused_add_rmsnorm_impl(
    input: &mut [u16],
    residual: &mut [u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                return fused_add_rmsnorm_bf16_avx512(
                    input,
                    residual,
                    weight,
                    batch_size,
                    hidden_size,
                    eps,
                );
            }
        }
    }
    fused_add_rmsnorm_bf16_scalar_via_generic(input, residual, weight, batch_size, hidden_size, eps);
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

fn fused_add_rmsnorm_bf16_scalar_via_generic(
    input: &mut [u16],
    residual: &mut [u16],
    weight: &[u16],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let inp = unsafe { &mut *(input as *mut [u16] as *mut [half::bf16]) };
    let res = unsafe { &mut *(residual as *mut [u16] as *mut [half::bf16]) };
    let w = unsafe { &*(weight as *const [u16] as *const [half::bf16]) };
    fused_add_rmsnorm_generic(inp, res, w, batch_size, hidden_size, eps);
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

pub(crate) fn fused_add_rmsnorm_generic<T: CpuFloat>(
    input: &mut [T],
    residual: &mut [T],
    weight: &[T],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let inv_n = 1.0f32 / hidden_size as f32;
    for b in 0..batch_size {
        let off = b * hidden_size;
        // Pass 1: sum_sq
        let mut sum_sq = 0.0f32;
        for j in 0..hidden_size {
            let added = input[off + j].to_f32() + residual[off + j].to_f32();
            sum_sq += added * added;
        }
        let scale = 1.0 / (sum_sq * inv_n + eps).sqrt();
        // Pass 2: write both
        for j in 0..hidden_size {
            let added = input[off + j].to_f32() + residual[off + j].to_f32();
            residual[off + j] = T::from_f32(added);
            input[off + j] = T::from_f32(added * weight[j].to_f32() * scale);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn fused_add_rmsnorm_bf16_avx512(
    input: &mut [u16],
    residual: &mut [u16],
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
        let in_ptr = unsafe { input.as_mut_ptr().add(off) };
        let res_ptr = unsafe { residual.as_mut_ptr().add(off) };
        let w_ptr = weight.as_ptr();

        // Pass 1: compute sum_sq only (don't write anything yet — saves F32 buffer)
        let mut sum_sq = _mm512_setzero_ps();
        for i in 0..chunks {
            let x = bf16x16_load_as_f32(unsafe { in_ptr.add(i * 16) as *const u16 });
            let r = bf16x16_load_as_f32(unsafe { res_ptr.add(i * 16) as *const u16 });
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

        // Pass 2: recompute added (cheap — input/residual still in L1), write both outputs
        for i in 0..chunks {
            let x = bf16x16_load_as_f32(unsafe { in_ptr.add(i * 16) as *const u16 });
            let r = bf16x16_load_as_f32(unsafe { res_ptr.add(i * 16) as *const u16 });
            let added = _mm512_add_ps(x, r);
            let w = bf16x16_load_as_f32(unsafe { w_ptr.add(i * 16) });
            f32x16_store_as_bf16(unsafe { res_ptr.add(i * 16) }, added);
            let result = _mm512_mul_ps(_mm512_mul_ps(added, w), scale_v);
            f32x16_store_as_bf16(unsafe { in_ptr.add(i * 16) }, result);
        }
        for j in (chunks * 16)..hidden_size {
            let added = bf16_to_f32(input[off + j]) + bf16_to_f32(residual[off + j]);
            residual[off + j] = f32_to_bf16(added);
            input[off + j] = f32_to_bf16(added * bf16_to_f32(weight[j]) * scale);
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

        let mut input = make_bf16_vec(&h_f32);
        let mut residual = make_bf16_vec(&res_f32);
        let weight = make_bf16_vec(&weight_f32);

        fused_add_rmsnorm_bf16_scalar_via_generic(&mut input, &mut residual, &weight, 1, hidden, eps);

        let actual: Vec<f32> = (0..hidden).map(|j| bf16_to_f32(residual[j])).collect();
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
        let mut in_s = make_bf16_vec(&h_f32);
        let mut res_s = make_bf16_vec(&res_f32);
        fused_add_rmsnorm_bf16_scalar_via_generic(&mut in_s, &mut res_s, &weight, batch, hidden, 1e-6);

        // Dispatched
        let mut in_d = make_bf16_vec(&h_f32);
        let mut res_d = make_bf16_vec(&res_f32);
        fused_add_rmsnorm_bf16(&mut in_d, &mut res_d, &weight, batch, hidden, 1e-6);

        assert_eq!(in_s, in_d, "fused input should match");
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

        let mut input = make_bf16_vec(&h_f32);
        let mut residual = make_bf16_vec(&res_f32);
        fused_add_rmsnorm_bf16(&mut input, &mut residual, &weight, batch, hidden, eps as f32);

        // Verify residual = h + res (SGLang tolerance)
        for row in 0..batch {
            let actual: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(residual[row * hidden + d]))
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

        // Verify rmsnorm(residual) output against F64 reference (SGLang tolerance)
        for row in 0..batch {
            let row_res: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(residual[row * hidden + d]))
                .collect();
            let row_w: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(weight[d]))
                .collect();
            let expected = rmsnorm_f64_ref(&row_res, &row_w, eps);
            let actual: Vec<f32> = (0..hidden)
                .map(|d| bf16_to_f32(input[row * hidden + d]))
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
}
