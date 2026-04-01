#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

use prelude_core::engine::InferenceEngine;
use prelude_core::types::{
    ClassificationInputs, ClassifyRequest, GenerateRequest, PromptInput, SamplingParams, StopConfig,
};
use prelude_core::{Engine, TaskOverride};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_id = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("google/gemma-3-1b-it");
    let test_type = args.get(2).map(|s| s.as_str()).unwrap_or("text");

    println!("Loading model: {}", model_id);
    let start = Instant::now();
    let engine_config = prelude_core::EngineConfig::from_env()
        .map_err(|e| format!("invalid engine config: {e}"))?;
    let engine = Engine::from_hf_hub_with_task(model_id, TaskOverride::Auto, engine_config)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

    match test_type {
        "classify" => run_classify_test(&engine).await?,
        "text" => run_text_test(&engine).await?,
        _ => {
            println!("Unknown test type: {}. Use 'text' or 'classify'", test_type);
        }
    }

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
        logprobs: None,
        prompt_logprobs: None,
    }
}

async fn run_text_test(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Text Generation Test ===\n");

    let prompt = "What is the capital of France?";

    // Test 1: max_tokens = 1 (single token generation)
    println!("Test 1: max_tokens = 1 (single token)");
    let request = create_generate_request(prompt, 1);

    let start = Instant::now();
    let result = engine.generate(request).await?;
    let elapsed = start.elapsed();
    println!("  Generated: {:?}", result.output_text);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Prompt tokens: {}", result.usage.prompt_tokens);
    println!("  Completion tokens: {}", result.usage.completion_tokens);
    println!();

    // Test 2: max_tokens = 32 (short generation)
    println!("Test 2: max_tokens = 32 (short generation)");
    let request = create_generate_request(prompt, 32);

    let start = Instant::now();
    let result = engine.generate(request).await?;
    let elapsed = start.elapsed();
    println!("  Generated: {:?}", result.output_text);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Tokens/sec: {:.2}",
        result.usage.completion_tokens as f64 / elapsed.as_secs_f64()
    );
    println!();

    // Test 3: max_tokens = 64 (longer generation)
    println!("Test 3: max_tokens = 64 (longer generation)");
    let request = create_generate_request(prompt, 64);

    let start = Instant::now();
    let result = engine.generate(request).await?;
    let elapsed = start.elapsed();
    println!("  Generated: {:?}", result.output_text);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Tokens/sec: {:.2}",
        result.usage.completion_tokens as f64 / elapsed.as_secs_f64()
    );
    println!();

    // Test 4: Longer prompt
    println!("Test 4: Longer prompt (256 tokens context)");
    let long_prompt = "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.";

    let request = create_generate_request(long_prompt, 32);

    let start = Instant::now();
    let result = engine.generate(request).await?;
    let elapsed = start.elapsed();
    println!("  Prompt tokens: {}", result.usage.prompt_tokens);
    println!("  Generated: {:?}", result.output_text);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Tokens/sec: {:.2}",
        result.usage.completion_tokens as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

async fn run_classify_test(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Classification Test ===\n");

    let texts = vec![
        "This is a wonderful product! I love it.",
        "This is terrible, I want my money back.",
        "The service was okay, nothing special.",
    ];

    let inputs = ClassificationInputs::Texts(texts.iter().map(|s| s.to_string()).collect());

    let request = ClassifyRequest {
        request_id: "test-1".to_string(),
        model: "test".to_string(),
        inputs,
    };

    let start = Instant::now();
    let result = engine.classify(request).await?;
    let elapsed = start.elapsed();

    println!("Classification results:");
    for (i, (text, classification)) in texts.iter().zip(result.results.iter()).enumerate() {
        println!("  [{}] \"{}\"", i, text);
        println!(
            "      Label: {:?}, Probs: {:?}",
            classification.label, classification.probs
        );
    }
    println!();
    println!("Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "Samples/sec: {:.2}",
        texts.len() as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
