use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor, D};
use prelude_core::loading::var_builder::VarBuilder;
use candle_transformers::models::qwen3::{
    Config as Qwen3Config, ModelForCausalLM as BaselineQwen3,
};
use prelude_core::models::qwen3::Qwen3ModelForCausalLM;

#[derive(Debug, Clone)]
struct Args {
    model_path: PathBuf,
    device: String,
    batch_size: usize,
    ours_only: bool,
    decode_tokens: usize,
    repeats: usize,
    contexts: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct BenchResult {
    prefill_ms: f64,
    decode_ms: f64,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let config = load_config(&args.model_path)?;
    let (device, dtype, device_name) = select_device(&args.device)?;

    // Register CUTLASS/DeepGEMM GEMM dispatch (replaces cuBLAS)
    #[cfg(any(feature = "cutlass-gemm", feature = "deepgemm"))]
    crate::ops::gpu::gemm::register_gpu_gemm();
    let weight_files = find_safetensor_files(&args.model_path)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };

    let mut baseline = if args.ours_only {
        None
    } else {
        Some(BaselineQwen3::new(&config, vb.clone())?)
    };
    let mut custom = Qwen3ModelForCausalLM::new(&config, vb)?;

    println!("Model path: {}", args.model_path.display());
    println!("Device: {} (dtype {:?})", device_name, dtype);
    println!("Contexts: {:?}", args.contexts);
    println!("Batch size: {}", args.batch_size);
    println!(
        "Mode: {}",
        if args.ours_only {
            "ours-only"
        } else {
            "compare"
        }
    );
    println!(
        "Decode tokens: {}, repeats: {}",
        args.decode_tokens, args.repeats
    );
    println!();
    if args.ours_only {
        println!("ctx\tours_prefill\tours_tok/s");
    } else {
        println!("ctx\tbase_prefill\tours_prefill\tprefill_x\tbase_tok/s\tours_tok/s\tdecode_x");
    }

    for &ctx in &args.contexts {
        let prompt = synthetic_batch_prompt(args.batch_size, ctx, config.vocab_size);
        let custom_res = run_bench(
            &mut custom,
            &prompt,
            args.batch_size,
            ctx,
            args.decode_tokens,
            args.repeats,
            &device,
        )?;
        if args.ours_only {
            let decode_token_count = (args.batch_size * args.decode_tokens) as f64;
            let custom_tps = decode_token_count / (custom_res.decode_ms / 1000.0);
            println!(
                "{}\t{:.2} ms\t{:.2}",
                ctx, custom_res.prefill_ms, custom_tps
            );
            continue;
        }

        let Some(base_model) = baseline.as_mut() else {
            candle_core::bail!("internal error: baseline model missing in compare mode");
        };
        let base_res = run_bench(
            base_model,
            &prompt,
            args.batch_size,
            ctx,
            args.decode_tokens,
            args.repeats,
            &device,
        )?;

        let decode_token_count = (args.batch_size * args.decode_tokens) as f64;
        let base_tps = decode_token_count / (base_res.decode_ms / 1000.0);
        let custom_tps = decode_token_count / (custom_res.decode_ms / 1000.0);
        println!(
            "{}\t{:.2} ms\t{:.2} ms\t{:.2}x\t{:.2}\t\t{:.2}\t\t{:.2}x",
            ctx,
            base_res.prefill_ms,
            custom_res.prefill_ms,
            base_res.prefill_ms / custom_res.prefill_ms.max(1e-9),
            base_tps,
            custom_tps,
            custom_tps / base_tps.max(1e-9),
        );
    }

    Ok(())
}

fn sync_device(device: &Device) {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        // Force CUDA synchronization for accurate timing
        let _ = device.synchronize();
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
}

fn run_bench<M: ForwardModel>(
    model: &mut M,
    prompt_tokens: &[u32],
    batch_size: usize,
    prompt_len: usize,
    decode_tokens: usize,
    repeats: usize,
    device: &Device,
) -> Result<BenchResult> {
    // Warmup: 2 iterations to ensure GPU is warmed up
    for _ in 0..2 {
        model.clear_kv_cache();
        let input = Tensor::from_vec(prompt_tokens.to_vec(), (batch_size, prompt_len), device)?;
        let logits = model.forward(&input, 0)?;
        let _ = sample_argmax_batch(&logits)?;
    }
    sync_device(device);

    let mut prefill_ms = 0f64;
    let mut decode_ms = 0f64;

    for _ in 0..repeats {
        model.clear_kv_cache();

        let input = Tensor::from_vec(prompt_tokens.to_vec(), (batch_size, prompt_len), device)?;
        sync_device(device);
        let prefill_start = Instant::now();
        let mut logits = model.forward(&input, 0)?;
        sync_device(device);
        prefill_ms += prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut offset = prompt_len;
        let mut next_tokens = sample_argmax_batch(&logits)?;
        sync_device(device);
        let decode_start = Instant::now();
        for _ in 0..decode_tokens {
            let next_input = Tensor::from_vec(next_tokens, (batch_size, 1), device)?;
            logits = model.forward(&next_input, offset)?;
            next_tokens = sample_argmax_batch(&logits)?;
            offset += 1;
        }
        sync_device(device);
        decode_ms += decode_start.elapsed().as_secs_f64() * 1000.0;
    }

    Ok(BenchResult {
        prefill_ms: prefill_ms / repeats as f64,
        decode_ms: decode_ms / repeats as f64,
    })
}

trait ForwardModel {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>;
    fn clear_kv_cache(&mut self);
}

impl ForwardModel for BaselineQwen3 {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward(input, offset)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }
}

impl ForwardModel for Qwen3ModelForCausalLM {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        // forward_with_cache returns [L, vocab]; wrap to [1, L, vocab] for sample_argmax_batch
        let flat = input.reshape((input.elem_count(),))?;
        self.forward_with_cache(&flat, offset)?.unsqueeze(0)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }
}

fn sample_argmax_batch(logits: &Tensor) -> Result<Vec<u32>> {
    let (_, l, _) = logits.dims3()?;
    logits
        .narrow(1, l - 1, 1)?
        .squeeze(1)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()
}

fn synthetic_batch_prompt(batch_size: usize, len: usize, vocab_size: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(batch_size.saturating_mul(len));
    let modulus = vocab_size.saturating_sub(1).max(1);
    for b in 0..batch_size {
        for i in 0..len {
            out.push(((b.wrapping_mul(8191) + i * 31 + 17) % modulus) as u32);
        }
    }
    out
}

fn parse_args() -> Result<Args> {
    let mut model_path = std::env::var("PRELUDE_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/Qwen3-0.6B"));
    let mut device = std::env::var("PRELUDE_DEVICE").unwrap_or_else(|_| "auto".to_string());
    let mut batch_size = 1usize;
    let mut ours_only = false;
    let mut decode_tokens = 64usize;
    let mut repeats = 3usize;
    let mut contexts = vec![32, 128, 512, 1024, 2048];

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    candle_core::bail!("missing value for --model-path");
                };
                model_path = PathBuf::from(v);
            }
            "--device" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    candle_core::bail!("missing value for --device");
                };
                device = v.to_string();
            }
            "--batch-size" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    candle_core::bail!("missing value for --batch-size");
                };
                batch_size = v.parse::<usize>().map_err(candle_core::Error::msg)?;
                if batch_size == 0 {
                    candle_core::bail!("--batch-size must be >= 1");
                }
            }
            "--ours-only" => {
                ours_only = true;
            }
            "--decode" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    candle_core::bail!("missing value for --decode");
                };
                decode_tokens = v.parse::<usize>().map_err(candle_core::Error::msg)?;
            }
            "--repeats" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    candle_core::bail!("missing value for --repeats");
                };
                repeats = v.parse::<usize>().map_err(candle_core::Error::msg)?;
            }
            "--contexts" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    candle_core::bail!("missing value for --contexts");
                };
                contexts = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>().map_err(candle_core::Error::msg))
                    .collect::<Result<Vec<_>>>()?;
                if contexts.is_empty() {
                    candle_core::bail!("--contexts produced an empty list");
                }
            }
            other => {
                candle_core::bail!(
                    "unknown arg: {other}. valid: --model-path --device --batch-size --ours-only --decode --repeats --contexts"
                );
            }
        }
        i += 1;
    }

    Ok(Args {
        model_path,
        device,
        batch_size,
        ours_only,
        decode_tokens,
        repeats,
        contexts,
    })
}

fn select_device(requested: &str) -> Result<(Device, DType, String)> {
    let requested = requested.to_ascii_lowercase();
    let device = match requested.as_str() {
        "cpu" => Device::Cpu,
        "auto" => Device::cuda_if_available(0)?,
        "cuda" => Device::new_cuda(0)?,
        s if s.starts_with("cuda:") => {
            let ordinal = s
                .trim_start_matches("cuda:")
                .parse::<usize>()
                .map_err(|e| candle_core::Error::msg(format!("invalid --device: {e}")))?;
            Device::new_cuda(ordinal)?
        }
        other => {
            candle_core::bail!("invalid --device '{other}', expected auto|cpu|cuda|cuda:N")
        }
    };
    let dtype = if device.supports_bf16() {
        DType::BF16
    } else {
        DType::F32
    };
    let device_name = if device.is_cuda() {
        "cuda".to_string()
    } else {
        "cpu".to_string()
    };
    Ok((device, dtype, device_name))
}

fn load_config(model_path: &Path) -> Result<Qwen3Config> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path).map_err(|e| {
        candle_core::Error::msg(format!("failed to read {}: {e}", config_path.display()))
    })?;
    serde_json::from_str(&content).map_err(candle_core::Error::msg)
}

fn find_safetensor_files(model_path: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| candle_core::Error::msg(format!("failed to read index: {e}")))?;
        let index: serde_json::Value =
            serde_json::from_str(&content).map_err(candle_core::Error::msg)?;
        let mut files: Vec<PathBuf> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for filename in map.values().filter_map(|v| v.as_str()) {
                if seen.insert(filename.to_string()) {
                    files.push(model_path.join(filename));
                }
            }
        }
        return Ok(files);
    }

    let single = model_path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(model_path)
        .map_err(|e| candle_core::Error::msg(format!("failed to read dir: {e}")))?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        candle_core::bail!("no safetensors files found in {}", model_path.display());
    }
    Ok(files)
}
