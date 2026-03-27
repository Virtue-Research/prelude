//! Model forward pass profiling — E2E prefill + decode timing.
//!
//! Loads Qwen3-0.6B and runs forward passes, reporting prefill tok/s and decode tok/s.

use candle_core::{DType, Device, Result, Tensor, D};
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

pub fn bench_forward(report: &mut BenchReport) -> Result<()> {
    // Find model
    let home = std::env::var("HOME").unwrap_or_default();
    let base = format!("{home}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let model_dir = match std::fs::read_dir(&base) {
        Ok(entries) => {
            let mut found = None;
            for entry in entries.flatten() {
                let p = entry.path();
                if p.join("config.json").exists() && p.join("model.safetensors").exists() {
                    found = Some(p);
                    break;
                }
            }
            match found {
                Some(p) => p,
                None => {
                    println!("\n=== Forward profiling: SKIPPED (no model found) ===");
                    return Ok(());
                }
            }
        }
        Err(_) => {
            println!("\n=== Forward profiling: SKIPPED (no model found) ===");
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

    let safetensor_files: Vec<std::path::PathBuf> = std::fs::read_dir(&model_dir)
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
    println!("  Loaded in {:.0}ms (layers={}, hidden={})",
        load_start.elapsed().as_millis(), config.num_hidden_layers, config.hidden_size);

    let repeats = 3;
    let prompt_lens = [32, 128, 512];
    let decode_steps = 16;

    for &prompt_len in &prompt_lens {
        // Generate dummy tokens
        let tokens: Vec<i64> = (0..prompt_len).map(|i| (i as i64 % 1000) + 100).collect();

        // Warmup
        model.clear_kv_cache();
        let input = Tensor::from_vec(tokens.clone(), (prompt_len,), &device)?;
        let _ = model.forward_with_cache(&input, 0)?;

        // Prefill benchmark
        let mut prefill_ms = 0.0;
        for _ in 0..repeats {
            model.clear_kv_cache();
            let input = Tensor::from_vec(tokens.clone(), (prompt_len,), &device)?;
            let t0 = Instant::now();
            let _ = model.forward_with_cache(&input, 0)?;
            prefill_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        prefill_ms /= repeats as f64;
        let prefill_tps = prompt_len as f64 / (prefill_ms / 1000.0);

        // Decode benchmark
        model.clear_kv_cache();
        let input = Tensor::from_vec(tokens.clone(), (prompt_len,), &device)?;
        let logits = model.forward_with_cache(&input, 0)?;
        let mut next_token = logits.argmax(D::Minus1)?.reshape((1,))?.to_vec1::<u32>()?[0];

        let mut decode_ms = 0.0;
        let offset_start = prompt_len;
        for step in 0..decode_steps {
            let next_input = Tensor::from_vec(vec![next_token as i64], (1,), &device)?;
            let t0 = Instant::now();
            let logits = model.forward_with_cache(&next_input, offset_start + step)?;
            decode_ms += t0.elapsed().as_secs_f64() * 1000.0;
            next_token = logits.argmax(D::Minus1)?.reshape((1,))?.to_vec1::<u32>()?[0];
        }
        let decode_per_token_ms = decode_ms / decode_steps as f64;
        let decode_tps = 1000.0 / decode_per_token_ms;

        println!("  ctx={prompt_len:>4}  prefill={prefill_ms:.1}ms ({prefill_tps:.0} tok/s)  decode={decode_per_token_ms:.1}ms/tok ({decode_tps:.1} tok/s)");

        report.add(BenchEntry {
            category: "cpu/forward/prefill".into(),
            name: format!("ctx={prompt_len}"),
            ours_us: prefill_ms * 1000.0,
            baseline_name: None, baseline_us: None,
            note: Some(format!("{prefill_tps:.0} tok/s")),
        });
        report.add(BenchEntry {
            category: "cpu/forward/decode".into(),
            name: format!("ctx={prompt_len}"),
            ours_us: decode_per_token_ms * 1000.0,
            baseline_name: None, baseline_us: None,
            note: Some(format!("{decode_tps:.1} tok/s")),
        });
    }

    Ok(())
}
