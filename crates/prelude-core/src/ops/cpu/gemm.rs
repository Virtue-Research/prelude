//! Custom BF16 GEMM for small M (1-16 tokens).
//!
//! Computes Output[M, N] = Input[M, K] × Weight[N, K]^T
//! where Weight is row-major [N, K] (standard nn.Linear convention).
//!
//! Strategy: partition N across threads. Each thread processes NR=4 weight
//! columns simultaneously per K-iteration, creating 4 independent DRAM streams
//! with software prefetch to hide memory latency.
//!
//! For M=1 (decode): pure memory-bandwidth-bound. Theoretical minimum on Xeon 8480+
//! (DDR5 ~250 GB/s) for gate_up (12.5 MB weights): ~50us.
//! v2 targets ~100-150us via NR=4 multi-stream + prefetch (was ~583us in v1).
//!
//! Uses AVX-512 BF16 `vdpbf16ps` instruction for 2× throughput over regular FMA.

/// N-register block size: process this many weight columns per K-iteration.
/// Creates NR independent DRAM streams for better memory-level parallelism.
const NR: usize = 4;

/// Prefetch distance in K-chunks (each chunk = 32 BF16 = 64 bytes = 1 cache line).
/// With NR=4 streams: each iteration prefetches 4 cache lines. At ~80ns DRAM latency
/// and ~4ns per dpbf16ps, PF=4 hides 4×4×4ns = 64ns of the 80ns latency.
/// Configurable via `CPU_GEMM_PREFETCH` env var (default 4).
fn prefetch_distance() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("CPU_GEMM_PREFETCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}

/// Trait to cast __m512i to __m512bh for dpbf16ps intrinsic.
#[cfg(target_arch = "x86_64")]
trait AsBf16 {
    fn as_bf16(self) -> core::arch::x86_64::__m512bh;
}

#[cfg(target_arch = "x86_64")]
impl AsBf16 for core::arch::x86_64::__m512i {
    #[inline(always)]
    fn as_bf16(self) -> core::arch::x86_64::__m512bh {
        unsafe { std::mem::transmute(self) }
    }
}

/// Threshold below which we use brgemm/spinning-pool GEMM instead of oneDNN packed.
/// Configurable via `CPU_GEMM_SMALL_M` env var (default: usize::MAX = always brgemm).
///
/// In micro-bench, brgemm beats oneDNN packed at all M up to 512 (228µs vs 305µs).
/// More importantly, in real inference oneDNN packed uses rayon which contends with
/// GemmPool spinning threads on the same cores, causing 5-10x slowdown vs micro-bench.
/// Keeping all GEMM on brgemm+GemmPool avoids the rayon↔GemmPool contention entirely.
pub fn small_m_threshold() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("CPU_GEMM_SMALL_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

/// Number of threads for the spinning GEMM pool.
/// Configurable via `CPU_GEMM_THREADS` env var. Default: min(48, physical_cores).
/// More threads = more DRAM bandwidth (each thread adds an independent memory stream).
/// With the spinning pool (no futex parking), 48 threads on Xeon 8480+ achieves
/// ~162 GB/s DRAM bandwidth, matching SGLang's OpenMP-based threading.
pub fn gemm_thread_count_pub() -> usize { gemm_thread_count() }

fn gemm_thread_count() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Some(v) = std::env::var("CPU_GEMM_THREADS").ok().and_then(|v| v.parse().ok()) {
            return v;
        }
        // Default: all physical cores across all NUMA nodes (override with CPU_GEMM_THREADS=N)
        crate::ops::cpu::numa::detect_all_physical_cores()
            .map(|c| c.len())
            .unwrap_or(rayon::current_num_threads())
    })
}

/// BF16 GEMM: out[M, N] = input[M, K] × weight[N, K]^T
///
/// - `input`: row-major [M, K] as &[u16] (BF16 bit patterns)
/// - `weight`: row-major [N, K] as &[u16] (BF16 bit patterns)
/// - `out`: row-major [M, N] as &mut [u16] (BF16 bit patterns), must be pre-allocated
/// - `m`, `k`, `n`: dimensions
///
/// Weight layout: weight[i * k + j] = W[i, j], so row i of W is weight[i*k .. (i+1)*k].
/// Output: out[r * n + c] = sum_j(input[r * k + j] * weight[c * k + j])
pub fn bf16_gemm_small_m(
    out: &mut [u16],
    input: &[u16],
    weight: &[u16],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(weight.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    if !is_x86_feature_detected!("avx512bf16") {
        bf16_gemm_scalar(out, input, weight, m, k, n);
        return;
    }

    let pool = super::gemm_pool::gemm_pool();
    let num_threads = pool.num_threads().min(n / NR.max(1)).max(1);
    let pf_dist = prefetch_distance();

    #[repr(C)]
    struct AvxGemmCtx {
        out_ptr: usize,
        in_ptr: usize,
        w_ptr: usize,
        m: usize,
        k: usize,
        n: usize,
        pf_dist: usize,
    }

    let ctx = AvxGemmCtx {
        out_ptr: out.as_mut_ptr() as usize,
        in_ptr: input.as_ptr() as usize,
        w_ptr: weight.as_ptr() as usize,
        m, k, n, pf_dist,
    };

    unsafe fn avx_gemm_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const AvxGemmCtx);
            let n_per_thread = (ctx.n + n_threads - 1) / n_threads;
            let n_start = tid * n_per_thread;
            let n_end = (n_start + n_per_thread).min(ctx.n);
            if n_start >= n_end { return; }

            bf16_gemm_tile_avx512_v2(
                ctx.out_ptr as *mut u16,
                ctx.in_ptr as *const u16,
                ctx.w_ptr as *const u16,
                ctx.m, ctx.k, ctx.n,
                n_start, n_end,
                ctx.pf_dist,
            );
        }
    }

    unsafe {
        pool.dispatch(
            avx_gemm_work,
            &ctx as *const AvxGemmCtx as *const u8,
            num_threads,
        );
    }
}

/// AVX-512 BF16 GEMM micro-kernel v2: NR=4 column blocking + software prefetch.
///
/// Key improvements over v1:
/// - Processes NR=4 weight columns per K-iteration → 4 independent DRAM streams
/// - Software prefetch hides ~100ns DRAM latency
/// - MR=4 × NR=4 register tiling for M>1 (16 accumulators, reuses each load)
/// - Input loaded once per K-chunk, reused across NR columns (input fits L1)
///
/// # Safety
/// - All pointers must be valid for the given dimensions.
/// - The N range [n_start, n_end) must not overlap with other threads' ranges.
#[target_feature(enable = "avx512f,avx512bf16")]
fn bf16_gemm_tile_avx512_v2(
    out: *mut u16,
    input: *const u16,
    weight: *const u16,
    m: usize,
    k: usize,
    n: usize,
    n_start: usize,
    n_end: usize,
    pf_dist: usize,
) {
    // Safety: caller guarantees all pointers valid for the given dimensions
    // and the N range [n_start, n_end) doesn't overlap with other threads
    unsafe {
    use core::arch::x86_64::*;

    let k_chunks = k / 32;

    // ── NR=4 blocked path ──────────────────────────────────────────────
    let mut col = n_start;
    while col + NR <= n_end {
        // MR=4 × NR=4 tiled: process 4 input rows × 4 weight columns
        let mut row = 0;
        while row + 4 <= m {
            // 16 accumulators: acc[mr][nr]
            let mut acc00 = _mm512_setzero_ps();
            let mut acc01 = _mm512_setzero_ps();
            let mut acc02 = _mm512_setzero_ps();
            let mut acc03 = _mm512_setzero_ps();
            let mut acc10 = _mm512_setzero_ps();
            let mut acc11 = _mm512_setzero_ps();
            let mut acc12 = _mm512_setzero_ps();
            let mut acc13 = _mm512_setzero_ps();
            let mut acc20 = _mm512_setzero_ps();
            let mut acc21 = _mm512_setzero_ps();
            let mut acc22 = _mm512_setzero_ps();
            let mut acc23 = _mm512_setzero_ps();
            let mut acc30 = _mm512_setzero_ps();
            let mut acc31 = _mm512_setzero_ps();
            let mut acc32 = _mm512_setzero_ps();
            let mut acc33 = _mm512_setzero_ps();

            let in0 = input.add(row * k);
            let in1 = input.add((row + 1) * k);
            let in2 = input.add((row + 2) * k);
            let in3 = input.add((row + 3) * k);
            let w0 = weight.add(col * k);
            let w1 = weight.add((col + 1) * k);
            let w2 = weight.add((col + 2) * k);
            let w3 = weight.add((col + 3) * k);

            for c in 0..k_chunks {
                let off = c * 32;

                // Software prefetch: load future K-chunks into L1
                if c + pf_dist < k_chunks {
                    let pf_off = (c + pf_dist) * 32;
                    _mm_prefetch(w0.add(pf_off) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(w1.add(pf_off) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(w2.add(pf_off) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(w3.add(pf_off) as *const i8, _MM_HINT_T0);
                }

                // Load 4 input rows (from L1 — input is small)
                let a0 = _mm512_loadu_si512(in0.add(off) as *const _).as_bf16();
                let a1 = _mm512_loadu_si512(in1.add(off) as *const _).as_bf16();
                let a2 = _mm512_loadu_si512(in2.add(off) as *const _).as_bf16();
                let a3 = _mm512_loadu_si512(in3.add(off) as *const _).as_bf16();

                // Load NR=4 weight columns (from DRAM — 4 independent streams)
                let wv0 = _mm512_loadu_si512(w0.add(off) as *const _).as_bf16();
                let wv1 = _mm512_loadu_si512(w1.add(off) as *const _).as_bf16();
                let wv2 = _mm512_loadu_si512(w2.add(off) as *const _).as_bf16();
                let wv3 = _mm512_loadu_si512(w3.add(off) as *const _).as_bf16();

                // 16 dpbf16ps: each input reused NR times, each weight reused MR times
                acc00 = _mm512_dpbf16_ps(acc00, a0, wv0);
                acc01 = _mm512_dpbf16_ps(acc01, a0, wv1);
                acc02 = _mm512_dpbf16_ps(acc02, a0, wv2);
                acc03 = _mm512_dpbf16_ps(acc03, a0, wv3);

                acc10 = _mm512_dpbf16_ps(acc10, a1, wv0);
                acc11 = _mm512_dpbf16_ps(acc11, a1, wv1);
                acc12 = _mm512_dpbf16_ps(acc12, a1, wv2);
                acc13 = _mm512_dpbf16_ps(acc13, a1, wv3);

                acc20 = _mm512_dpbf16_ps(acc20, a2, wv0);
                acc21 = _mm512_dpbf16_ps(acc21, a2, wv1);
                acc22 = _mm512_dpbf16_ps(acc22, a2, wv2);
                acc23 = _mm512_dpbf16_ps(acc23, a2, wv3);

                acc30 = _mm512_dpbf16_ps(acc30, a3, wv0);
                acc31 = _mm512_dpbf16_ps(acc31, a3, wv1);
                acc32 = _mm512_dpbf16_ps(acc32, a3, wv2);
                acc33 = _mm512_dpbf16_ps(acc33, a3, wv3);
            }

            // Reduce + scalar remainder + store
            let k_tail_start = k_chunks * 32;
            macro_rules! reduce_store {
                ($acc:expr, $in_ptr:expr, $mr:expr, $nr:expr) => {{
                    let mut sum = _mm512_reduce_add_ps($acc);
                    let w_row = weight.add((col + $nr) * k);
                    for r in k_tail_start..k {
                        sum += bf16_to_f32(*$in_ptr.add(r))
                            * bf16_to_f32(*w_row.add(r));
                    }
                    *out.add((row + $mr) * n + col + $nr) = f32_to_bf16(sum);
                }};
            }
            reduce_store!(acc00, in0, 0, 0);
            reduce_store!(acc01, in0, 0, 1);
            reduce_store!(acc02, in0, 0, 2);
            reduce_store!(acc03, in0, 0, 3);
            reduce_store!(acc10, in1, 1, 0);
            reduce_store!(acc11, in1, 1, 1);
            reduce_store!(acc12, in1, 1, 2);
            reduce_store!(acc13, in1, 1, 3);
            reduce_store!(acc20, in2, 2, 0);
            reduce_store!(acc21, in2, 2, 1);
            reduce_store!(acc22, in2, 2, 2);
            reduce_store!(acc23, in2, 2, 3);
            reduce_store!(acc30, in3, 3, 0);
            reduce_store!(acc31, in3, 3, 1);
            reduce_store!(acc32, in3, 3, 2);
            reduce_store!(acc33, in3, 3, 3);
            row += 4;
        }

        // Remaining M rows (< 4): 1 row × NR=4 columns
        while row < m {
            let in_row = input.add(row * k);
            let mut a0 = _mm512_setzero_ps();
            let mut a1 = _mm512_setzero_ps();
            let mut a2 = _mm512_setzero_ps();
            let mut a3 = _mm512_setzero_ps();

            let w0 = weight.add(col * k);
            let w1 = weight.add((col + 1) * k);
            let w2 = weight.add((col + 2) * k);
            let w3 = weight.add((col + 3) * k);

            for c in 0..k_chunks {
                let off = c * 32;

                if c + pf_dist < k_chunks {
                    let pf_off = (c + pf_dist) * 32;
                    _mm_prefetch(w0.add(pf_off) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(w1.add(pf_off) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(w2.add(pf_off) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(w3.add(pf_off) as *const i8, _MM_HINT_T0);
                }

                let iv = _mm512_loadu_si512(in_row.add(off) as *const _).as_bf16();
                let wv0 = _mm512_loadu_si512(w0.add(off) as *const _).as_bf16();
                let wv1 = _mm512_loadu_si512(w1.add(off) as *const _).as_bf16();
                let wv2 = _mm512_loadu_si512(w2.add(off) as *const _).as_bf16();
                let wv3 = _mm512_loadu_si512(w3.add(off) as *const _).as_bf16();

                a0 = _mm512_dpbf16_ps(a0, iv, wv0);
                a1 = _mm512_dpbf16_ps(a1, iv, wv1);
                a2 = _mm512_dpbf16_ps(a2, iv, wv2);
                a3 = _mm512_dpbf16_ps(a3, iv, wv3);
            }

            let k_tail_start = k_chunks * 32;
            macro_rules! reduce_store_1 {
                ($acc:expr, $nr:expr) => {{
                    let mut sum = _mm512_reduce_add_ps($acc);
                    let w_row = weight.add((col + $nr) * k);
                    for r in k_tail_start..k {
                        sum += bf16_to_f32(*in_row.add(r))
                            * bf16_to_f32(*w_row.add(r));
                    }
                    *out.add(row * n + col + $nr) = f32_to_bf16(sum);
                }};
            }
            reduce_store_1!(a0, 0);
            reduce_store_1!(a1, 1);
            reduce_store_1!(a2, 2);
            reduce_store_1!(a3, 3);
            row += 1;
        }
        col += NR;
    }

    // ── Remaining N columns (< NR) — single-column path ───────────────
    while col < n_end {
        let w_row = weight.add(col * k);
        let mut row = 0;
        while row + 4 <= m {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            let in0 = input.add(row * k);
            let in1 = input.add((row + 1) * k);
            let in2 = input.add((row + 2) * k);
            let in3 = input.add((row + 3) * k);

            for c in 0..k_chunks {
                let off = c * 32;
                if c + pf_dist < k_chunks {
                    _mm_prefetch(w_row.add((c + pf_dist) * 32) as *const i8, _MM_HINT_T0);
                }
                let wv = _mm512_loadu_si512(w_row.add(off) as *const _).as_bf16();
                acc0 = _mm512_dpbf16_ps(acc0, _mm512_loadu_si512(in0.add(off) as *const _).as_bf16(), wv);
                acc1 = _mm512_dpbf16_ps(acc1, _mm512_loadu_si512(in1.add(off) as *const _).as_bf16(), wv);
                acc2 = _mm512_dpbf16_ps(acc2, _mm512_loadu_si512(in2.add(off) as *const _).as_bf16(), wv);
                acc3 = _mm512_dpbf16_ps(acc3, _mm512_loadu_si512(in3.add(off) as *const _).as_bf16(), wv);
            }

            let k_tail_start = k_chunks * 32;
            let mut s0 = _mm512_reduce_add_ps(acc0);
            let mut s1 = _mm512_reduce_add_ps(acc1);
            let mut s2 = _mm512_reduce_add_ps(acc2);
            let mut s3 = _mm512_reduce_add_ps(acc3);
            for r in k_tail_start..k {
                let wf = bf16_to_f32(*w_row.add(r));
                s0 += bf16_to_f32(*in0.add(r)) * wf;
                s1 += bf16_to_f32(*in1.add(r)) * wf;
                s2 += bf16_to_f32(*in2.add(r)) * wf;
                s3 += bf16_to_f32(*in3.add(r)) * wf;
            }
            *out.add(row * n + col) = f32_to_bf16(s0);
            *out.add((row + 1) * n + col) = f32_to_bf16(s1);
            *out.add((row + 2) * n + col) = f32_to_bf16(s2);
            *out.add((row + 3) * n + col) = f32_to_bf16(s3);
            row += 4;
        }

        while row < m {
            let in_row = input.add(row * k);
            let mut acc = _mm512_setzero_ps();
            for c in 0..k_chunks {
                let off = c * 32;
                if c + pf_dist < k_chunks {
                    _mm_prefetch(w_row.add((c + pf_dist) * 32) as *const i8, _MM_HINT_T0);
                }
                let wv = _mm512_loadu_si512(w_row.add(off) as *const _);
                let av = _mm512_loadu_si512(in_row.add(off) as *const _);
                acc = _mm512_dpbf16_ps(acc, av.as_bf16(), wv.as_bf16());
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            for r in (k_chunks * 32)..k {
                sum += bf16_to_f32(*in_row.add(r)) * bf16_to_f32(*w_row.add(r));
            }
            *out.add(row * n + col) = f32_to_bf16(sum);
            row += 1;
        }
        col += 1;
    }
    } // unsafe
}

/// Scalar fallback for non-AVX512BF16 CPUs.
fn bf16_gemm_scalar(
    out: &mut [u16],
    input: &[u16],
    weight: &[u16],
    m: usize,
    k: usize,
    n: usize,
) {
    for row in 0..m {
        for col in 0..n {
            let mut sum: f32 = 0.0;
            for j in 0..k {
                let a = bf16_to_f32(input[row * k + j]);
                let b = bf16_to_f32(weight[col * k + j]);
                sum += a * b;
            }
            out[row * n + col] = f32_to_bf16(sum);
        }
    }
}

#[inline(always)]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    // Round-to-nearest-even
    let bits = v.to_bits();
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_bf16_simple(v: f32) -> u16 {
        f32_to_bf16(v)
    }

    fn ref_gemm(input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0f32;
                for j in 0..k {
                    sum += input[r * k + j] * weight[c * k + j];
                }
                out[r * n + c] = sum;
            }
        }
        out
    }

    fn make_rng(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 16) as f32 / 65536.0) * 2.0 - 1.0
        }
    }

    fn check_gemm(m: usize, k: usize, n: usize, tol: f32) {
        let mut rng = make_rng(42 + m as u32 * 7 + n as u32);
        let input_f32: Vec<f32> = (0..m * k).map(|_| rng()).collect();
        let weight_f32: Vec<f32> = (0..n * k).map(|_| rng()).collect();

        let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_simple(v)).collect();
        let weight_bf16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_bf16_simple(v)).collect();

        let mut out = vec![0u16; m * n];
        bf16_gemm_small_m(&mut out, &input_bf16, &weight_bf16, m, k, n);

        let input_bf16_f32: Vec<f32> = input_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let weight_bf16_f32: Vec<f32> = weight_bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        let ref_out = ref_gemm(&input_bf16_f32, &weight_bf16_f32, m, k, n);

        let mut max_diff: f32 = 0.0;
        for i in 0..m * n {
            let got = bf16_to_f32(out[i]);
            let expected = ref_out[i];
            let diff = (got - expected).abs();
            max_diff = max_diff.max(diff);
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "mismatch at [{}, {}]: got={got}, expected={expected}, diff={diff}, tol={t} (M={m} K={k} N={n})",
                i / n,
                i % n
            );
        }
    }

    #[test]
    fn test_bf16_gemm_small() {
        check_gemm(2, 64, 32, 0.05);
    }

    #[test]
    fn test_bf16_gemm_m1_large() {
        // Simulate decode: M=1, K=1024, N=6144 (gate_up)
        check_gemm(1, 1024, 6144, 0.1);
    }

    #[test]
    fn test_bf16_gemm_m4_nr_boundary() {
        // Test MR=4 × NR=4 tiling path
        check_gemm(4, 1024, 6144, 0.1);
    }

    #[test]
    fn test_bf16_gemm_m5_nr_boundary() {
        // M=5: exercises MR=4 path + 1 remaining row
        check_gemm(5, 1024, 6144, 0.1);
    }

    #[test]
    fn test_bf16_gemm_n_not_multiple_of_nr() {
        // N=6145: exercises NR=4 blocked path + 1 remaining column
        check_gemm(1, 1024, 6145, 0.1);
    }

    #[test]
    fn test_bf16_gemm_n_less_than_nr() {
        // N=3: only remaining columns path (no NR=4 blocks)
        check_gemm(2, 64, 3, 0.05);
    }

    #[test]
    fn test_bf16_gemm_k_not_multiple_of_32() {
        // K=100: exercises scalar remainder path
        check_gemm(2, 100, 32, 0.1);
    }

    #[test]
    fn test_bf16_gemm_m8() {
        // M=8: two full MR=4 blocks
        check_gemm(8, 1024, 3072, 0.1);
    }

    #[test]
    fn test_bf16_gemm_m16() {
        // M=16: max threshold, four MR=4 blocks
        check_gemm(16, 1024, 6144, 0.15);
    }
}
