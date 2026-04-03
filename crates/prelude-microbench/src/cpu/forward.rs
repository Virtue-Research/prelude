//! Model forward pass profiling — E2E prefill + decode timing.
//!
//! Loads Qwen3-0.6B and runs forward passes, reporting prefill tok/s and decode tok/s.
//! Optionally compares against llama.cpp via `llama-bench`.

use prelude_core::tensor::{DType, Device, Result, Tensor, D};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

/// Auto-detect HF model directory for Qwen3-0.6B.
fn find_hf_model_dir(override_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = override_dir {
        if dir.join("config.json").exists() {
            return Some(dir.to_path_buf());
        }
        eprintln!("  WARNING: --model-dir {} has no config.json", dir.display());
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let base = format!("{home}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let entries = std::fs::read_dir(&base).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.join("config.json").exists() && p.join("model.safetensors").exists() {
            return Some(p);
        }
    }
    None
}

/// Auto-detect GGUF model file.
fn find_gguf_model(override_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = override_path {
        if p.exists() {
            return Some(p.to_path_buf());
        }
        eprintln!("  WARNING: --gguf-model {} not found", p.display());
        return None;
    }
    // Check common locations
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF"),
        "../llama.cpp/models".into(),
    ];
    for base in &candidates {
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.contains("Qwen3-0.6B") && name.ends_with(".gguf") {
                        return Some(p);
                    }
                }
            }
        }
        // Check snapshots subdirectory (HuggingFace Hub layout)
        let snapshots = PathBuf::from(base).join("snapshots");
        if let Ok(snap_entries) = std::fs::read_dir(&snapshots) {
            for snap in snap_entries.flatten() {
                if let Ok(files) = std::fs::read_dir(snap.path()) {
                    for f in files.flatten() {
                        let p = f.path();
                        if p.extension().is_some_and(|e| e == "gguf") {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Find llama-bench binary.
fn find_llama_bench() -> Option<PathBuf> {
    // Check env var
    if let Ok(path) = std::env::var("LLAMA_BENCH") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    // Check common locations
    let candidates = [
        "/usr/local/bin/llama-bench",
        "../llama.cpp/bin/llama-bench",
        "../llama.cpp/build/bin/llama-bench",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    // Check PATH via `which`
    if let Ok(output) = std::process::Command::new("which")
        .arg("llama-bench")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}

/// Parse llama-bench CSV output to extract pp (prefill) and tg (decode) tok/s.
/// CSV format: model,size,params,backend,ngl,threads,type_k,type_v,n_batch,n_ubatch,flash,test,t/s
fn parse_llama_bench_csv(csv: &str) -> Vec<(String, f64)> {
    let mut results = Vec::new();
    for line in csv.lines() {
        if line.starts_with("model") || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 13 {
            let test = fields[fields.len() - 2].trim();
            let tps: f64 = fields[fields.len() - 1].trim().parse().unwrap_or(0.0);
            if tps > 0.0 {
                results.push((test.to_string(), tps));
            }
        }
    }
    results
}

/// Run llama-bench and return results as (test_name, tok/s) pairs.
fn run_llama_bench(
    llama_bench: &Path,
    gguf_model: &Path,
    prompt_lens: &[usize],
    n_threads: usize,
) -> Option<Vec<(String, f64)>> {
    // Build pp args: -p 32,128,512
    let pp_str = prompt_lens
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");

    println!("\n  Running llama-bench...");
    println!("    binary: {}", llama_bench.display());
    println!("    model:  {}", gguf_model.display());
    println!("    threads: {n_threads}");

    let output = std::process::Command::new(llama_bench)
        .args([
            "-m", &gguf_model.to_string_lossy(),
            "-p", &pp_str,
            "-n", "16",        // decode tokens (match our decode_steps)
            "-r", "3",         // repeats
            "-t", &n_threads.to_string(),
            "-o", "csv",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("  llama-bench failed: {stderr}");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(parse_llama_bench_csv(&stdout))
}

pub fn bench_forward(
    report: &mut BenchReport,
    model_dir_override: Option<&Path>,
    gguf_model_override: Option<&Path>,
) -> Result<()> {
    let model_dir = match find_hf_model_dir(model_dir_override) {
        Some(p) => p,
        None => {
            println!("\n=== Forward profiling: SKIPPED (no HF model found) ===");
            println!("  Hint: download with `huggingface-cli download Qwen/Qwen3-0.6B`");
            println!("  Or specify --model-dir /path/to/model");
            return Ok(());
        }
    };

    println!("\n=== Forward pass profiling (Qwen3-0.6B, CPU) ===");
    println!("  Model: {}", model_dir.display());

    let device = Device::Cpu;
    let dtype = DType::BF16;

    // Load model
    let config: prelude_core::nn_ops::Qwen3Config = {
        let s = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(candle_core::Error::msg)?;
        serde_json::from_str(&s).map_err(candle_core::Error::msg)?
    };

    let safetensor_files: Vec<PathBuf> = std::fs::read_dir(&model_dir)
        .map_err(candle_core::Error::msg)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();

    let load_start = Instant::now();
    let vb = unsafe {
        prelude_core::loading::var_builder::VarBuilder::from_mmaped_safetensors(
            &safetensor_files, dtype, &device,
        )?
    };
    let mut model = prelude_core::models::qwen3::Qwen3ModelForCausalLM::new(&config, vb)?;
    println!(
        "  Loaded in {:.0}ms (layers={}, hidden={})",
        load_start.elapsed().as_millis(),
        config.num_hidden_layers,
        config.hidden_size
    );

    let repeats = 3;
    let prompt_lens = [32, 128, 512];
    let decode_steps = 16;

    // ── Run llama-bench first (if available) ──
    let llama_bench_bin = find_llama_bench();
    let gguf_model = find_gguf_model(gguf_model_override);
    let llama_results = match (&llama_bench_bin, &gguf_model) {
        (Some(bench), Some(gguf)) => {
            let n_threads = rayon::current_num_threads();
            run_llama_bench(bench, gguf, &prompt_lens, n_threads)
        }
        _ => {
            if llama_bench_bin.is_none() {
                println!("  llama-bench: not found (set LLAMA_BENCH or install to PATH)");
            }
            if gguf_model.is_none() {
                println!("  GGUF model: not found (set GGUF_MODEL or --gguf-model)");
            }
            None
        }
    };

    // Index llama-bench results by test name
    let llama_pp: std::collections::HashMap<usize, f64> = llama_results
        .as_ref()
        .map(|results| {
            results
                .iter()
                .filter_map(|(test, tps)| {
                    if test.starts_with("pp") {
                        test[2..].parse::<usize>().ok().map(|n| (n, *tps))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let llama_tg: std::collections::HashMap<usize, f64> = llama_results
        .as_ref()
        .map(|results| {
            results
                .iter()
                .filter_map(|(test, tps)| {
                    if test.starts_with("tg") {
                        test[2..].parse::<usize>().ok().map(|n| (n, *tps))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // ── Run our forward pass ──
    println!("\n  Prelude forward:");

    let ops = prelude_cpu::cpu_ops();

    for &prompt_len in &prompt_lens {
        let tokens: Vec<i64> = (0..prompt_len).map(|i| (i as i64 % 1000) + 100).collect();

        // Warmup
        model.clear_kv_cache();
        let input = Tensor::from_vec(tokens.clone(), (prompt_len,), &device)?;
        let _ = model.forward_with_cache(ops, &input, 0)?;

        // Prefill benchmark
        let mut prefill_ms = 0.0;
        for _ in 0..repeats {
            model.clear_kv_cache();
            let input = Tensor::from_vec(tokens.clone(), (prompt_len,), &device)?;
            let t0 = Instant::now();
            let _ = model.forward_with_cache(ops, &input, 0)?;
            prefill_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        prefill_ms /= repeats as f64;
        let prefill_tps = prompt_len as f64 / (prefill_ms / 1000.0);

        // Decode benchmark
        model.clear_kv_cache();
        let input = Tensor::from_vec(tokens.clone(), (prompt_len,), &device)?;
        let logits = model.forward_with_cache(ops, &input, 0)?;
        // Take last token's logits: [seq_len, vocab] -> [vocab]
        let last_logits = logits.get(logits.dim(0)? - 1)?;
        let mut next_token = last_logits.argmax(D::Minus1)?.to_vec0::<u32>()?;

        let mut decode_ms = 0.0;
        let offset_start = prompt_len;
        for step in 0..decode_steps {
            let next_input = Tensor::from_vec(vec![next_token as i64], (1,), &device)?;
            let t0 = Instant::now();
            let logits = model.forward_with_cache(ops, &next_input, offset_start + step)?;
            decode_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // logits is [1, vocab] for single token — get [vocab] then argmax
            let last = logits.get(logits.dim(0)? - 1)?;
            next_token = last.argmax(D::Minus1)?.to_vec0::<u32>()?;
        }
        let decode_per_token_ms = decode_ms / decode_steps as f64;
        let decode_tps = 1000.0 / decode_per_token_ms;

        // Print with optional llama.cpp comparison
        let llama_pp_tps = llama_pp.get(&prompt_len);
        let llama_tg_tps = llama_tg.get(&decode_steps);

        print!(
            "    ctx={prompt_len:>4}  prefill={prefill_ms:.1}ms ({prefill_tps:.0} tok/s)"
        );
        if let Some(&ll_tps) = llama_pp_tps {
            let ratio = prefill_tps / ll_tps;
            print!("  llama.cpp={ll_tps:.0} tok/s  ratio={ratio:.2}x");
        }
        println!();

        print!(
            "    {:>10}  decode={decode_per_token_ms:.1}ms/tok ({decode_tps:.1} tok/s)",
            ""
        );
        if let Some(&ll_tps) = llama_tg_tps {
            let ratio = decode_tps / ll_tps;
            print!("  llama.cpp={ll_tps:.1} tok/s  ratio={ratio:.2}x");
        }
        println!();

        // Report entries
        report.add(BenchEntry {
            category: "cpu/forward/prefill".into(),
            name: format!("ctx={prompt_len}"),
            ours_us: prefill_ms * 1000.0,
            baseline_name: llama_pp_tps.map(|_| "llama.cpp".into()),
            baseline_us: llama_pp_tps.map(|&tps| {
                // Convert tok/s to microseconds for prompt_len tokens
                (prompt_len as f64 / tps) * 1_000_000.0
            }),
            note: Some(format!("{prefill_tps:.0} tok/s")),
        });
        report.add(BenchEntry {
            category: "cpu/forward/decode".into(),
            name: format!("ctx={prompt_len}"),
            ours_us: decode_per_token_ms * 1000.0,
            baseline_name: llama_tg_tps.map(|_| "llama.cpp".into()),
            baseline_us: llama_tg_tps.map(|&tps| {
                // Convert tok/s to microseconds per token
                1_000_000.0 / tps
            }),
            note: Some(format!("{decode_tps:.1} tok/s")),
        });
    }

    Ok(())
}
