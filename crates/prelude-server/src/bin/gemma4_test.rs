use prelude_core::engine::InferenceEngine;
use prelude_core::types::{GenerateRequest, PromptInput, SamplingParams, StopConfig};
use prelude_core::{Engine, TaskOverride};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Register CUDA ops before model loading
    #[cfg(feature = "cuda")]
    prelude_cuda::register();

    let args: Vec<String> = std::env::args().collect();

    let model_id = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("google/gemma-4-E2B-it");

    println!("Loading model: {}", model_id);
    let start = Instant::now();
    let engine_config = prelude_core::EngineConfig::from_env()
        .map_err(|e| format!("invalid engine config: {e}"))?;
    let engine = Engine::from_hf_hub_with_task(model_id, TaskOverride::Auto, engine_config)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

    run_text_test(&engine).await?;

    Ok(())
}

fn create_generate_request(prompt: &str, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        request_id: "test".to_string(),
        model: "test".to_string(),
        input: PromptInput::Text(prompt.to_string()),
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: None,
        },
        max_new_tokens: max_tokens,
        stop: StopConfig {
            strings: vec![],
            token_ids: vec![],
        },
        seed: Some(42),
        deadline_ms: None,
        logprobs: Some(5),
        prompt_logprobs: None,
    }
}

async fn run_text_test(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Gemma4 Text Generation Test ===\n");

    // Use max_tokens=1 for prefill-only validation (no paged KV cache needed)
    let prompts = [
        ("What is the capital of France?", 1),
        ("Explain quantum computing in one sentence.", 1),
        ("Write a haiku about the ocean.", 1),
    ];

    for (i, (prompt, max_tokens)) in prompts.iter().enumerate() {
        println!("Test {}: \"{}\" (max_tokens={})", i + 1, prompt, max_tokens);

        let request = create_generate_request(prompt, *max_tokens);
        let start = Instant::now();
        let result = engine.generate(request).await?;
        let elapsed = start.elapsed();

        println!("  Output: {:?}", result.output_text);
        println!("  Output token IDs: {:?}", result.output_token_ids);
        if let Some(lp) = &result.token_logprobs {
            for info in lp {
                println!("  Token: {} (id={}) logprob={:.4}", info.token, info.token_id, info.logprob);
                for (tid, tok, lp2) in &info.top_logprobs {
                    println!("    top: {tok:>20} (id={tid:>6}) logprob={lp2:.4}");
                }
            }
        }
        println!("  Prompt tokens: {}", result.usage.prompt_tokens);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        if result.usage.completion_tokens > 0 {
            println!(
                "  Tokens/sec: {:.2}",
                result.usage.completion_tokens as f64 / elapsed.as_secs_f64()
            );
        }
        println!();
    }

    // Determinism test: same seed should produce same output
    println!("Test 4: Determinism (same seed → same output)");
    let prompt = "The meaning of life is";
    let r1 = engine
        .generate(create_generate_request(prompt, 1))
        .await?;
    let r2 = engine
        .generate(create_generate_request(prompt, 1))
        .await?;
    println!("  Run 1: {:?}", r1.output_text);
    println!("  Run 2: {:?}", r2.output_text);
    if r1.output_text == r2.output_text {
        println!("  PASS: deterministic");
    } else {
        println!("  WARN: non-deterministic (may be expected with some backends)");
    }
    println!();

    // Longer context test
    println!("Test 5: Longer context");
    let long_prompt = "The history of artificial intelligence began in antiquity, \
        with myths, stories and rumors of artificial beings endowed with intelligence \
        or consciousness by master craftsmen. The seeds of modern AI were planted by \
        philosophers who attempted to describe the process of human thinking as the \
        mechanical manipulation of symbols. This work culminated in the invention of \
        the programmable digital computer in the 1940s, a machine based on the abstract \
        essence of mathematical reasoning. This device and the ideas behind it inspired \
        a handful of scientists to begin seriously discussing the possibility of building \
        an electronic brain. Summarize the above in one sentence:";

    let request = create_generate_request(long_prompt, 1);
    let start = Instant::now();
    let result = engine.generate(request).await?;
    let elapsed = start.elapsed();
    println!("  Prompt tokens: {}", result.usage.prompt_tokens);
    println!("  Output: {:?}", result.output_text);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    if result.usage.completion_tokens > 0 {
        println!(
            "  Tokens/sec: {:.2}",
            result.usage.completion_tokens as f64 / elapsed.as_secs_f64()
        );
    }

    Ok(())
}
