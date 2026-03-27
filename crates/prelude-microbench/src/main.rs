mod baselines;
mod cpu;
mod hardware;
mod report;

use clap::Parser;
use report::BenchReport;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "prelude-microbench", about = "Prelude kernel micro-benchmarks")]
struct Cli {
    /// Benchmark filters (e.g. "quant", "gemm", "attention", "forward")
    filters: Vec<String>,

    /// Save results as JSON
    #[arg(long)]
    json: Option<PathBuf>,

    /// Compare against previous results JSON
    #[arg(long)]
    compare: Option<PathBuf>,

    /// Warmup iterations
    #[arg(long, default_value = "50")]
    warmup: usize,

    /// Repeat iterations
    #[arg(long, default_value = "2000")]
    repeats: usize,

    /// HuggingFace model directory (for forward benchmark).
    /// Auto-detects ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B if not set.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// GGUF model file for llama-bench comparison (for forward benchmark).
    /// Also reads GGUF_MODEL env var.
    #[arg(long)]
    gguf_model: Option<PathBuf>,
}

fn should_run(filters: &[String], name: &str) -> bool {
    filters.is_empty() || filters.iter().any(|f| name.contains(f.as_str()))
}

fn main() -> candle_core::Result<()> {
    let cli = Cli::parse();

    // Initialize NUMA-pinned rayon pool
    let numa_report = prelude_core::ops::cpu::numa::init_numa_rayon_pool();

    // Initialize oneDNN
    prelude_core::ops::onednn::init();

    // Detect hardware
    let hw = hardware::HardwareInfo::detect();
    println!("=== prelude-microbench ===");
    hw.print_header();
    println!("  {numa_report}");
    println!("  warmup={}, repeats={}", cli.warmup, cli.repeats);
    #[cfg(ggml_baseline)]
    println!("  ggml baseline: enabled");
    #[cfg(not(ggml_baseline))]
    println!("  ggml baseline: disabled (set GGML_SRC to enable)");
    println!();

    let mut report = BenchReport::new(hw);

    // ── CPU / Quantized kernels ──
    if should_run(&cli.filters, "quant") {
        cpu::quant::verify_dot_precision()?;
        cpu::quant::bench_dot(&mut report, cli.warmup, cli.repeats)?;
        cpu::quant::bench_matmul(&mut report, cli.warmup.min(5), cli.repeats.min(50))?;
    }

    // ── CPU / RMSNorm ──
    if should_run(&cli.filters, "rmsnorm") || should_run(&cli.filters, "fused") {
        cpu::rmsnorm::bench_all(&mut report, cli.warmup, cli.repeats)?;
    }

    // ── CPU / SiLU*Mul ──
    if should_run(&cli.filters, "silu") {
        cpu::silu_mul::bench_all(&mut report, cli.warmup, cli.repeats)?;
    }

    // ── CPU / RoPE ──
    if should_run(&cli.filters, "rope") {
        cpu::rope::bench_all(&mut report, cli.warmup, cli.repeats)?;
    }

    // ── CPU / Attention ──
    if should_run(&cli.filters, "attention") {
        cpu::attention::bench_all(&mut report, cli.warmup, cli.repeats)?;
    }

    // ── CPU / GEMM ──
    if should_run(&cli.filters, "gemm") {
        cpu::gemm::bench_all(&mut report, cli.warmup, cli.repeats)?;
    }

    // ── CPU / Forward (model E2E) ──
    if should_run(&cli.filters, "forward") {
        let gguf_model = cli.gguf_model.clone()
            .or_else(|| std::env::var("GGUF_MODEL").ok().map(PathBuf::from));
        cpu::forward::bench_forward(&mut report, cli.model_dir.as_deref(), gguf_model.as_deref())?;
    }

    // Save JSON
    if let Some(path) = &cli.json {
        report.save_json(path).map_err(candle_core::Error::wrap)?;
    }

    // Compare with baseline
    if let Some(path) = &cli.compare {
        report
            .print_comparison(path)
            .map_err(candle_core::Error::wrap)?;
    }

    Ok(())
}
