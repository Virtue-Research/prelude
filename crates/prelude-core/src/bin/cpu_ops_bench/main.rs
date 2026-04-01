//! Micro-benchmark for pure Rust CPU kernels vs Candle baseline (and onednn if available).
//!
//! Usage:
//!   cargo run -p prelude-core --bin cpu_ops_bench --release
//!
//! With oneDNN GEMM:
//!   cargo run -p prelude-core --bin cpu_ops_bench --release --features onednn

#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

mod attention;
mod gemm;
mod quant;
mod rmsnorm;
mod rope;
mod silu_mul;

use candle_core::Result;
use candle_core::{DType, Tensor};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // Optional filter: pass kernel names to run only those (e.g. "gemm", "rmsnorm rope")
    let filter: Option<Vec<&str>> = if args.len() > 1 {
        Some(args[1..].iter().map(|s| s.as_str()).collect())
    } else {
        None
    };
    let run = |name: &str| -> bool {
        filter.as_ref().map_or(true, |f| f.iter().any(|&k| name.contains(k)))
    };

    let warmup = 50;
    let repeats = 2000;

    // Initialize NUMA-pinned rayon pool (must be before any rayon work)
    let numa_report = prelude_core::ops::cpu::numa::init_numa_rayon_pool();

    {
        prelude_core::ops::onednn::init();
        // oneDNN uses THREADPOOL runtime (rayon-backed), no OpenMP binding needed
    }

    println!("CPU Ops Micro-Benchmark");
    println!("  {numa_report}");
    println!("  warmup={warmup}, repeats={repeats}");
    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "  AVX-512F: {}  AVX-512BW: {}  AVX-512BF16: {}",
            if is_x86_feature_detected!("avx512f") { "yes" } else { "NO" },
            if is_x86_feature_detected!("avx512bw") { "yes" } else { "NO" },
            if is_x86_feature_detected!("avx512bf16") { "yes" } else { "NO" },
        );
    }

    let batches = &[1, 4, 16, 64, 256];
    // Qwen3 hidden sizes: 0.6B=896, 1.7B=1536, MLP-0.6B=4864, 32B=7168
    let hiddens = &[896, 1536, 4864, 7168];

    let mlp_dims = &[4864, 8960, 38656];
    let rope_configs: &[(usize, usize, usize)] = &[(128, 16, 8), (128, 32, 8), (128, 56, 14)];
    // Attention configs: (num_heads, num_kv_heads, head_dim, seq_len, num_seqs)
    let attn_extend_configs: &[(usize, usize, usize, usize, usize)] = &[
        // Qwen3-0.6B: 16 heads, 8 KV heads, 128 dim
        // Small seq_len (decode-like, no KV cache -- the real bottleneck!)
        (16, 8, 128, 1, 1),
        (16, 8, 128, 2, 1),
        (16, 8, 128, 4, 1),
        (16, 8, 128, 8, 1),
        (16, 8, 128, 16, 1),   // boundary: small path threshold (old)
        (16, 8, 128, 32, 1),
        (16, 8, 128, 64, 1),   // boundary: small path threshold (new)
        (16, 8, 128, 128, 1),
        (16, 8, 128, 512, 1),
        (16, 8, 128, 1024, 1),
        (16, 8, 128, 2048, 1),
        (16, 8, 128, 4096, 1),
        (16, 8, 128, 8192, 1),
        (16, 8, 128, 32, 4),
        (16, 8, 128, 8, 8),    // batch of short seqs: 8 requests x 8 tokens
        // Qwen3-1.7B: 16 heads, 4 KV heads, 128 dim
        (16, 4, 128, 128, 1),
        (16, 4, 128, 2048, 1),
        // Qwen3-32B: 64 heads, 8 KV heads, 128 dim
        (64, 8, 128, 32, 1),
        (64, 8, 128, 128, 1),
        (64, 8, 128, 1024, 1),
        (64, 8, 128, 4096, 1),
    ];
    let attn_decode_configs: &[(usize, usize, usize, usize, usize)] = &[
        // (num_heads, num_kv_heads, head_dim, cache_len, num_seqs)
        (16, 8, 128, 32, 1),
        (16, 8, 128, 128, 1),
        (16, 8, 128, 512, 1),
        (16, 8, 128, 1024, 1),
        (16, 8, 128, 2048, 1),
        (16, 8, 128, 4096, 1),
        (16, 8, 128, 8192, 1),
        (16, 8, 128, 128, 4),
        (64, 8, 128, 128, 1),
        (64, 8, 128, 2048, 1),
    ];
    let gemm_configs: &[(usize, usize, usize)] = &[
        // 0.6B: hidden=896, intermediate=4864
        (1, 896, 4864),
        (2, 896, 4864),
        (4, 896, 4864),
        (8, 896, 4864),
        (16, 896, 4864),
        (32, 896, 4864),
        (64, 896, 4864),
        (128, 896, 4864),   // boundary: brgemm vs oneDNN packed (new threshold)
        (256, 896, 4864),
        (512, 896, 4864),   // well above threshold: should stay on oneDNN packed
        // 1.7B: hidden=1536, intermediate=8960
        (1, 1536, 8960),
        (16, 1536, 8960),
        (64, 1536, 8960),
        (128, 1536, 8960),  // boundary: brgemm vs oneDNN packed
        // 8B: hidden=3584, intermediate=18944
        (1, 3584, 18944),
        (16, 3584, 18944),
        (64, 3584, 18944),
        (128, 3584, 18944), // boundary: brgemm vs oneDNN packed
        // 32B: hidden=7168, intermediate=38656
        (1, 7168, 38656),
        (4, 7168, 38656),
        (16, 7168, 38656),
        (64, 7168, 38656),
        (128, 7168, 38656), // boundary: brgemm vs oneDNN packed
    ];

    // -- RMSNorm --
    if run("rmsnorm") {
        println!("\n=== RMSNorm (BF16) ===");
        for &hidden in hiddens {
            for &batch in batches {
                rmsnorm::bench(hidden, batch, warmup, repeats)?;
            }
            println!();
        }
        println!("=== RMSNorm (F16 / F32) ===");
        for &hidden in hiddens {
            for &batch in &[1, 16, 256] {
                rmsnorm::bench_dtypes(hidden, batch, warmup, repeats)?;
            }
            println!();
        }
    }

    // -- Fused Add+RMSNorm --
    if run("fused") {
        println!("=== Fused Add+RMSNorm (BF16) ===");
        for &hidden in hiddens {
            for &batch in batches {
                rmsnorm::bench_fused(hidden, batch, warmup, repeats)?;
            }
            println!();
        }
        println!("=== Fused Add+RMSNorm (F16 / F32) ===");
        for &hidden in hiddens {
            for &batch in &[1, 16, 256] {
                rmsnorm::bench_fused_dtypes(hidden, batch, warmup, repeats)?;
            }
            println!();
        }
    }

    // -- SiLU*Mul --
    if run("silu") {
        println!("=== SiLU*Mul ===");
        for &dim in mlp_dims {
            for &batch in batches {
                silu_mul::bench(dim, batch, warmup, repeats)?;
            }
            println!();
        }
    }

    // -- RoPE --
    if run("rope") {
        println!("=== RoPE ===");
        for &(head_dim, num_heads, num_kv_heads) in rope_configs {
            for &batch in batches {
                rope::bench(head_dim, num_heads, num_kv_heads, batch, warmup, repeats)?;
            }
            println!();
        }
    }

    // -- Attention --
    if run("attention") {
        println!("=== Attention (extend/prefill) ===");
        for &(nh, nkv, hd, slen, nseq) in attn_extend_configs {
            // Adaptive repeats: fewer for long sequences (O(n^2) cost)
            let attn_repeats = if slen >= 4096 { 10 } else if slen >= 1024 { 50 } else { 200 };
            attention::bench_extend(nh, nkv, hd, slen, nseq, 5, attn_repeats)?;
        }
        println!();
        println!("=== Attention (decode) ===");
        for &(nh, nkv, hd, ctx_len, nseq) in attn_decode_configs {
            let attn_repeats = if ctx_len >= 4096 { 20 } else if ctx_len >= 1024 { 50 } else { 200 };
            attention::bench_decode(nh, nkv, hd, ctx_len, nseq, 5, attn_repeats)?;
        }
    }

    // -- GEMM --
    if run("gemm") {
        println!("=== GEMM (BF16 linear) ===");
        let gemm_repeats = 100; // GEMM is much slower, fewer repeats
        for &(m, k, n) in gemm_configs {
            gemm::bench(m, k, n, 5, gemm_repeats)?;
        }
    }

    // -- Quantized kernels --
    if run("quant") {
        // Precision (llama.cpp style: dot product error vs F32)
        quant::verify_dot_precision()?;

        // Dot product throughput
        println!("\n=== Quantized dot product throughput ===");
        for &k in &[256, 512, 1024, 2048, 4096] {
            quant::bench_dot(k, 100, 10000)?;
        }

        // Matmul throughput
        println!("\n=== Quantized matmul: Q4_0 / Q4_K vs F32 ===");
        let quant_repeats = 50;
        let quant_configs: &[(usize, usize, usize)] = &[
            (1, 1024, 1024),
            (1, 2048, 2048),
            (1, 4096, 4096),
            (4, 2048, 2048),
            (4, 4096, 4096),
            (16, 4096, 4096),
            (1, 1024, 4096),
            (1, 4096, 1024),
        ];
        for &(m, k, n) in quant_configs {
            quant::bench_matmul(m, k, n, 5, quant_repeats)?;
        }
    }

    // -- Numerical accuracy --
    if run("accuracy") {
        {
            println!("\n=== Numerical accuracy: GEMM backends vs candle F32 ===");
            println!("  Pass criteria: |a-b| <= atol + rtol*max(|a|,|b|) (atol=5e-2, rtol=5e-2)");
            for &(m, k, n) in gemm_configs {
                gemm::verify_accuracy(m, k, n)?;
            }
        }
    }

    Ok(())
}

// -- Shared benchmark helpers --

pub(crate) struct BenchResult {
    pub cpu_ops_us: f64,
    pub candle_us: f64,
    pub sgl_us: Option<f64>,
}

pub(crate) fn print_result(label: &str, hidden: usize, batch: usize, r: &BenchResult) {
    print!(
        "  {label} [{batch:>3}x{hidden:>4}]  cpu_ops={:>8.1}us  candle={:>8.1}us  ({:.2}x)",
        r.cpu_ops_us,
        r.candle_us,
        r.candle_us / r.cpu_ops_us,
    );
    if let Some(sgl) = r.sgl_us {
        print!("  sgl={sgl:>8.1}us  ({:.2}x)", sgl / r.cpu_ops_us);
    }
    println!();
}

// -- Shared accuracy helpers --

/// Accuracy comparison result following SGLang's atol+rtol convention.
pub(crate) struct AccuracyResult {
    pub max_abs_err: f32,
    pub max_rel_err: f32,
    pub fail_count: usize,
    pub total: usize,
}

/// Compare two tensors using SGLang-style tolerance: |a - b| <= atol + rtol * max(|a|, |b|).
pub(crate) fn compare_tensors(a: &Tensor, b: &Tensor, atol: f32, rtol: f32) -> Result<AccuracyResult> {
    let a_f32 = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b_f32 = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut fail_count = 0usize;
    for (&av, &bv) in a_f32.iter().zip(b_f32.iter()) {
        let abs_err = (av - bv).abs();
        let denom = av.abs().max(bv.abs());
        let rel_err = if denom > 0.0 { abs_err / denom } else { 0.0 };
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
        let tol = atol + rtol * denom;
        if abs_err > tol {
            fail_count += 1;
        }
    }
    Ok(AccuracyResult { max_abs_err, max_rel_err, fail_count, total: a_f32.len() })
}

pub(crate) fn print_accuracy(label: &str, hidden: usize, batch: usize, r: &AccuracyResult, atol: f32, rtol: f32) {
    let status = if r.fail_count == 0 { "PASS" } else { "FAIL" };
    println!(
        "  {label} [{batch:>3}x{hidden:>4}] {status}: max_abs={:.6}, max_rel={:.6}, fail={}/{} (atol={atol}, rtol={rtol})",
        r.max_abs_err, r.max_rel_err, r.fail_count, r.total,
    );
}
