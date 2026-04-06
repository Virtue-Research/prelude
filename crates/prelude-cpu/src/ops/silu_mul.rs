//! SiLU×Mul CPU kernels — AVX-512 BF16 + generic scalar for F16/F32.
//!
//! SiLU(gate) × up = gate / (1 + exp(-gate)) × up
//!
//! Input layout: `[num_tokens, 2 * dim]` where first half is gate, second half is up.
//! Output: `[num_tokens, dim]`.

use prelude_core::tensor::{DType, Result, Tensor};
use super::bf16_utils::{bf16_to_f32, f32_to_bf16};
#[cfg(target_arch = "x86_64")]
use super::bf16_utils::{bf16x16_load_as_f32, f32x16_store_as_bf16};
use super::cpu_float::CpuFloat;

// ── Tensor-level API ────────────────────────────────────────────────────

/// Fused SiLU×Mul: BF16 uses AVX-512, other dtypes use generic scalar.
/// input: `[num_tokens, 2*dim]` (gate and up concatenated).
/// Returns: `[num_tokens, dim]`.
pub fn cpu_silu_and_mul(input: &Tensor) -> Result<Tensor> {
    let (n, d2) = input.dims2()?;
    if d2 % 2 != 0 {
        prelude_core::tensor::bail!("cpu_silu_and_mul: last dim must be even, got {d2}");
    }
    let dim = d2 / 2;

    match input.dtype() {
        DType::BF16 => {
            let input_2d = input.contiguous()?;
            let in_slice = super::tensor_as_u16_slice(&input_2d)?;
            let mut out_buf: Vec<u16> = vec![0u16; n * dim];
            silu_and_mul_bf16(&mut out_buf, in_slice, n, dim);
            super::u16_vec_to_bf16_tensor(out_buf, &[n, dim], input.device())
        }
        DType::F16 => cpu_silu_and_mul_typed::<half::f16>(input, n, dim),
        _ => cpu_silu_and_mul_typed::<f32>(input, n, dim),
    }
}

fn cpu_silu_and_mul_typed<T: CpuFloat>(
    input: &Tensor, n: usize, dim: usize,
) -> Result<Tensor> {
    let input_t = input.to_dtype(T::DTYPE)?.contiguous()?;
    let in_data = T::tensor_to_vec(&input_t)?;
    let mut out_buf = vec![T::zero(); n * dim];
    silu_and_mul_generic(&mut out_buf, &in_data[..n * 2 * dim], n, dim);
    T::vec_to_tensor(out_buf, &[n, dim], input.device())
}

/// In-place SiLU×Mul: overwrites gate portion of input tensor, returns narrowed view.
/// BF16 only (used in raw MLP path).
pub fn cpu_silu_and_mul_inplace(input: Tensor) -> Result<Tensor> {
    let (n, d2) = input.dims2()?;
    if d2 % 2 != 0 {
        prelude_core::tensor::bail!("cpu_silu_and_mul_inplace: last dim must be even, got {d2}");
    }
    let dim = d2 / 2;

    let data_ptr = unsafe { input.data_ptr_mut()? as *mut u16 };

    silu_and_mul_bf16_inplace(data_ptr, n, dim);
    input.narrow(1, 0, dim)
}

// ── Raw kernel API (u16 slices) ─────────────────────────────────────────

/// Fused SiLU×Mul for BF16 data.
///
/// `input`: `[num_tokens * 2 * dim]` as raw `u16` (BF16), gate and up concatenated per token.
/// `output`: `[num_tokens * dim]` as raw `u16`.
///
/// Uses the spinning GemmPool for parallelization (same threads as GEMM dispatch).
/// This avoids rayon's futex parking overhead that dominated for small batch sizes
/// (e.g., tokens=8 was 80µs serial vs SGLang 26µs with OpenMP threads already spinning).
pub fn silu_and_mul_bf16(
    output: &mut [u16],
    input: &[u16],
    num_tokens: usize,
    dim: usize,
) {
    debug_assert_eq!(input.len(), num_tokens * 2 * dim);
    debug_assert_eq!(output.len(), num_tokens * dim);

    if num_tokens <= 1 {
        silu_and_mul_impl(output, input, num_tokens, dim);
        return;
    }

    // Use spinning pool for multi-token batches (threads already warm from GEMM)
    let pool = super::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(num_tokens);

    #[repr(C)]
    struct SiluCtx {
        out_ptr: usize,
        in_ptr: usize,
        num_tokens: usize,
        dim: usize,
    }

    let ctx = SiluCtx {
        out_ptr: output.as_mut_ptr() as usize,
        in_ptr: input.as_ptr() as usize,
        num_tokens,
        dim,
    };

    unsafe fn silu_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const SiluCtx);
            let rows_per_thread = (ctx.num_tokens + n_threads - 1) / n_threads;
            let start = tid * rows_per_thread;
            let end = (start + rows_per_thread).min(ctx.num_tokens);
            if start >= end { return; }

            let chunk_tokens = end - start;
            let in_off = start * 2 * ctx.dim;
            let out_off = start * ctx.dim;
            let in_ptr = (ctx.in_ptr as *const u16).add(in_off);
            let out_ptr = (ctx.out_ptr as *mut u16).add(out_off);
            let in_slice = std::slice::from_raw_parts(in_ptr, chunk_tokens * 2 * ctx.dim);
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, chunk_tokens * ctx.dim);
            silu_and_mul_impl(out_slice, in_slice, chunk_tokens, ctx.dim);
        }
    }

    unsafe {
        pool.dispatch(
            silu_work,
            &ctx as *const SiluCtx as *const u8,
            n_threads,
        );
    }
}

/// In-place fused SiLU×Mul: overwrites gate portion with SiLU(gate) * up.
///
/// `data`: `[num_tokens * 2 * dim]` as raw u16 (BF16), gate||up per token.
/// After call, `data[t*2*dim .. t*2*dim+dim]` contains SiLU(gate)*up.
/// The up portion is untouched.
///
/// Eliminates the output allocation (~3MB at M=512) by writing in-place.
pub fn silu_and_mul_bf16_inplace(
    data: *mut u16,
    num_tokens: usize,
    dim: usize,
) {
    if num_tokens <= 1 {
        silu_and_mul_inplace_impl(data, num_tokens, dim);
        return;
    }

    let pool = super::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(num_tokens);

    #[repr(C)]
    struct SiluInplaceCtx {
        data_ptr: usize,
        num_tokens: usize,
        dim: usize,
    }

    let ctx = SiluInplaceCtx {
        data_ptr: data as usize,
        num_tokens,
        dim,
    };

    unsafe fn silu_inplace_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const SiluInplaceCtx);
            let rows_per_thread = (ctx.num_tokens + n_threads - 1) / n_threads;
            let start = tid * rows_per_thread;
            let end = (start + rows_per_thread).min(ctx.num_tokens);
            if start >= end { return; }

            let chunk_tokens = end - start;
            let ptr = (ctx.data_ptr as *mut u16).add(start * 2 * ctx.dim);
            silu_and_mul_inplace_impl(ptr, chunk_tokens, ctx.dim);
        }
    }

    unsafe {
        pool.dispatch(
            silu_inplace_work,
            &ctx as *const SiluInplaceCtx as *const u8,
            n_threads,
        );
    }
}

// ── In-place implementation dispatch ────────────────────────────────────

fn silu_and_mul_inplace_impl(data: *mut u16, num_tokens: usize, dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                return silu_and_mul_bf16_avx512_inplace(data, num_tokens, dim);
            }
        }
    }
    // Scalar fallback
    for t in 0..num_tokens {
        let row = unsafe { std::slice::from_raw_parts_mut(data.add(t * 2 * dim), 2 * dim) };
        for j in 0..dim {
            let g = bf16_to_f32(row[j]);
            let u = bf16_to_f32(row[dim + j]);
            let silu_g = g / (1.0 + (-g).exp());
            row[j] = f32_to_bf16(silu_g * u);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn silu_and_mul_bf16_avx512_inplace(
    data: *mut u16,
    num_tokens: usize,
    dim: usize,
) {
    use core::arch::x86_64::*;

    let chunks = dim / 16;

    for t in 0..num_tokens {
        // Safety: caller guarantees data points to [num_tokens * 2 * dim] valid u16 elements
        let gate_ptr = unsafe { data.add(t * 2 * dim) };
        let up_ptr = unsafe { gate_ptr.add(dim) };

        for i in 0..chunks {
            let g = bf16x16_load_as_f32(unsafe { gate_ptr.add(i * 16) });
            let u = bf16x16_load_as_f32(unsafe { up_ptr.add(i * 16) });

            let neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g);
            let exp_neg_g = exp_ps_avx512(neg_g);
            let one = _mm512_set1_ps(1.0);
            let denom = _mm512_add_ps(one, exp_neg_g);
            let silu_g = _mm512_div_ps(g, denom);

            let result = _mm512_mul_ps(silu_g, u);
            f32x16_store_as_bf16(unsafe { gate_ptr.add(i * 16) as *mut u16 }, result);
        }
        // Scalar remainder
        for j in (chunks * 16)..dim {
            unsafe {
                let g = bf16_to_f32(*gate_ptr.add(j));
                let u = bf16_to_f32(*up_ptr.add(j));
                let silu_g = g / (1.0 + (-g).exp());
                *(gate_ptr.add(j) as *mut u16) = f32_to_bf16(silu_g * u);
            }
        }
    }
}

// ── Implementation dispatch ──────────────────────────────────────────────

fn silu_and_mul_impl(output: &mut [u16], input: &[u16], num_tokens: usize, dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                return silu_and_mul_bf16_avx512(output, input, num_tokens, dim);
            }
        }
    }
    silu_and_mul_bf16_scalar_via_generic(output, input, num_tokens, dim);
}

// ── Scalar fallback (delegates to generic kernel via CpuFloat) ──────────

fn silu_and_mul_bf16_scalar_via_generic(output: &mut [u16], input: &[u16], num_tokens: usize, dim: usize) {
    let out = unsafe { &mut *(output as *mut [u16] as *mut [half::bf16]) };
    let inp = unsafe { &*(input as *const [u16] as *const [half::bf16]) };
    silu_and_mul_generic(out, inp, num_tokens, dim);
}

// ── Generic scalar kernel (any CpuFloat dtype) ─────────────────────────

pub(crate) fn silu_and_mul_generic<T: CpuFloat>(
    output: &mut [T],
    input: &[T],
    num_tokens: usize,
    dim: usize,
) {
    for t in 0..num_tokens {
        let gate_off = t * 2 * dim;
        let up_off = gate_off + dim;
        let out_off = t * dim;
        for j in 0..dim {
            let g = input[gate_off + j].to_f32();
            let u = input[up_off + j].to_f32();
            let silu_g = g / (1.0 + (-g).exp());
            output[out_off + j] = T::from_f32(silu_g * u);
        }
    }
}

// ── AVX-512 exp approximation ───────────────────────────────────────────

/// Fast vectorized exp(x) for 16 packed floats using Cody-Waite range reduction
/// + 5th-order Taylor polynomial. ~20 ULP accuracy (more than enough for BF16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
fn exp_ps_avx512(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;

    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm512_set1_ps(0.693145752);
    let ln2_lo = _mm512_set1_ps(1.42860677e-6);
    let half = _mm512_set1_ps(0.5);

    let x = _mm512_max_ps(x, _mm512_set1_ps(-87.33654));
    let x = _mm512_min_ps(x, _mm512_set1_ps(88.72284));

    let fx = _mm512_fmadd_ps(x, log2e, half);
    let n = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    let r = _mm512_sub_ps(x, _mm512_mul_ps(n, ln2_hi));
    let r = _mm512_sub_ps(r, _mm512_mul_ps(n, ln2_lo));

    let c1 = _mm512_set1_ps(1.0);
    let c2 = _mm512_set1_ps(0.5);
    let c3 = _mm512_set1_ps(0.16666666);
    let c4 = _mm512_set1_ps(0.04166666);
    let c5 = _mm512_set1_ps(0.00833333);

    let mut p = _mm512_fmadd_ps(c5, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c1);

    let n_i = _mm512_cvtps_epi32(n);
    let bias = _mm512_set1_epi32(127);
    let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(n_i, bias), 23));

    _mm512_mul_ps(p, pow2n)
}

// ── AVX-512 implementation ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn silu_and_mul_bf16_avx512(
    output: &mut [u16],
    input: &[u16],
    num_tokens: usize,
    dim: usize,
) {
    use core::arch::x86_64::*;

    let chunks = dim / 16;

    for t in 0..num_tokens {
        // Safety: input/output slices are validated by debug_assert in caller
        let gate_ptr = unsafe { input.as_ptr().add(t * 2 * dim) };
        let up_ptr = unsafe { gate_ptr.add(dim) };
        let out_ptr = unsafe { output.as_mut_ptr().add(t * dim) };

        for i in 0..chunks {
            let g = bf16x16_load_as_f32(unsafe { gate_ptr.add(i * 16) });
            let u = bf16x16_load_as_f32(unsafe { up_ptr.add(i * 16) });

            // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
            let neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g);
            let exp_neg_g = exp_ps_avx512(neg_g);
            let one = _mm512_set1_ps(1.0);
            let denom = _mm512_add_ps(one, exp_neg_g);
            let silu_g = _mm512_div_ps(g, denom);

            let result = _mm512_mul_ps(silu_g, u);
            f32x16_store_as_bf16(unsafe { out_ptr.add(i * 16) }, result);
        }
        // Scalar remainder
        for j in (chunks * 16)..dim {
            let g = bf16_to_f32(input[t * 2 * dim + j]);
            let u = bf16_to_f32(input[t * 2 * dim + dim + j]);
            let silu_g = g / (1.0 + (-g).exp());
            output[t * dim + j] = f32_to_bf16(silu_g * u);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::bf16_utils::{make_bf16_vec, to_f32_vec, bf16_to_f32};
    use super::super::max_sglang_violation;

    fn silu_f32(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn test_silu_and_mul_scalar_basic() {
        let dim = 64;
        // gate values around [-3, 3] (typical MLP range)
        let gate_f32: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) * 6.0 - 3.0).collect();
        let up_f32: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.01).collect();

        // Interleave: [gate..., up...]
        let mut input_f32 = gate_f32.clone();
        input_f32.extend_from_slice(&up_f32);
        let input = make_bf16_vec(&input_f32);
        let mut output = vec![0u16; dim];

        silu_and_mul_bf16_scalar_via_generic(&mut output, &input, 1, dim);

        let actual = to_f32_vec(&output);
        let expected: Vec<f32> = (0..dim)
            .map(|j| {
                silu_f32(bf16_to_f32(make_bf16_vec(&gate_f32)[j]))
                    * bf16_to_f32(make_bf16_vec(&up_f32)[j])
            })
            .collect();

        let violation = max_sglang_violation(&actual, &expected);
        assert!(
            violation <= 0.0,
            "scalar silu_and_mul worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
        );
    }

    #[test]
    fn test_silu_and_mul_dispatch_matches_scalar() {
        let dim = 128;
        let num_tokens = 4;
        let n = num_tokens * 2 * dim;

        let input_f32: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.013) - 0.5).sin() * 3.0)
            .collect();
        let input = make_bf16_vec(&input_f32);

        let mut out_scalar = vec![0u16; num_tokens * dim];
        silu_and_mul_bf16_scalar_via_generic(&mut out_scalar, &input, num_tokens, dim);

        let mut out_dispatch = vec![0u16; num_tokens * dim];
        silu_and_mul_bf16(&mut out_dispatch, &input, num_tokens, dim);

        assert_eq!(out_scalar, out_dispatch, "dispatch should match scalar");
    }

    #[test]
    fn test_silu_and_mul_large_batch() {
        let dim = 128;
        let num_tokens = 256; // above parallelization threshold
        let n = num_tokens * 2 * dim;

        let input_f32: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.007) - 0.3).cos() * 2.0)
            .collect();
        let input = make_bf16_vec(&input_f32);

        let mut out_seq = vec![0u16; num_tokens * dim];
        silu_and_mul_impl(&mut out_seq, &input, num_tokens, dim);

        let mut out_par = vec![0u16; num_tokens * dim];
        silu_and_mul_bf16(&mut out_par, &input, num_tokens, dim);

        assert_eq!(out_seq, out_par, "parallel should match sequential");
    }

    /// Verify SiLU×Mul at realistic MLP dimensions against F32 reference.
    fn verify_silu_config(dim: usize, num_tokens: usize, label: &str) {
        let n = num_tokens * 2 * dim;
        let input_f32: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.007) - 0.5).sin() * 3.0)
            .collect();
        let input = make_bf16_vec(&input_f32);
        let mut output = vec![0u16; num_tokens * dim];

        silu_and_mul_bf16(&mut output, &input, num_tokens, dim);

        // Verify against F32 reference per token (SGLang tolerance)
        for t in 0..num_tokens {
            let gate_off = t * 2 * dim;
            let up_off = gate_off + dim;
            let out_off = t * dim;

            let actual: Vec<f32> = (0..dim).map(|j| bf16_to_f32(output[out_off + j])).collect();
            let expected: Vec<f32> = (0..dim)
                .map(|j| {
                    let g = bf16_to_f32(input[gate_off + j]);
                    let u = bf16_to_f32(input[up_off + j]);
                    silu_f32(g) * u
                })
                .collect();
            let violation = max_sglang_violation(&actual, &expected);
            assert!(
                violation <= 0.0,
                "{label} token={t} worst violation={violation:.6} (SGLang atol=1e-2, rtol=1e-2)"
            );
        }
    }

    #[test]
    fn test_silu_realistic_configs() {
        // Qwen3-0.6B MLP intermediate: 4864
        verify_silu_config(4864, 1, "0.6B batch=1");
        verify_silu_config(4864, 16, "0.6B batch=16");
        verify_silu_config(4864, 64, "0.6B batch=64");
        // Qwen3-1.7B MLP intermediate: 8960
        verify_silu_config(8960, 1, "1.7B batch=1");
        verify_silu_config(8960, 16, "1.7B batch=16");
        // Qwen3-32B MLP intermediate: 38656
        verify_silu_config(38656, 1, "32B batch=1");
        verify_silu_config(38656, 4, "32B batch=4");
    }

    #[test]
    fn test_exp_accuracy() {
        // Test exp approximation against std lib
        let test_vals: Vec<f32> = (-200..200)
            .map(|i| i as f32 * 0.4) // range [-80, 80]
            .collect();

        for &x in &test_vals {
            let expected = x.exp();
            if expected.is_infinite() || expected == 0.0 {
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    use core::arch::x86_64::*;
                    let xv = _mm512_set1_ps(x);
                    let rv = exp_ps_avx512(xv);
                    let mut result = [0.0f32; 16];
                    _mm512_storeu_ps(result.as_mut_ptr(), rv);

                    let rel_err = ((result[0] - expected) / expected).abs();
                    assert!(
                        rel_err < 1e-5,
                        "exp({x}): expected {expected}, got {}, rel_err={rel_err}",
                        result[0]
                    );
                }
            }
        }
    }
}
