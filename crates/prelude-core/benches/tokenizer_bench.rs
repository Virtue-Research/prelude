#[global_allocator]
static GLOBAL: bc_mimalloc::MiMalloc = bc_mimalloc::MiMalloc;

//! Benchmark: HuggingFace `tokenizers` vs `fastokens` — encode/decode speed + correctness.
//!
//! Usage:
//!   cargo bench -p prelude-core --bench tokenizer_bench --features hf_tokenizer -- --model Qwen/Qwen3-0.6B
//!
//! Custom model:
//!   cargo bench -p prelude-core --bench tokenizer_bench --features hf_tokenizer -- --model deepseek-ai/DeepSeek-V3

#[cfg(not(feature = "hf_tokenizer"))]
compile_error!("This benchmark requires the `hf_tokenizer` feature for HF comparison. Use: --features hf_tokenizer");

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Qwen/Qwen3-0.6B");

    println!("Tokenizer Benchmark: HuggingFace tokenizers vs fastokens");
    println!("Model: {model_id}\n");

    // --- Load tokenizers ---
    println!("Loading tokenizers...");
    let t0 = Instant::now();
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;

    let t_hf_start = Instant::now();
    let hf_tok = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("HF tokenizer load failed: {e}"))?;
    let hf_load_ms = t_hf_start.elapsed().as_secs_f64() * 1000.0;

    let t_ft_start = Instant::now();
    let ft_tok = fastokens::Tokenizer::from_file(&tokenizer_path)?;
    let ft_load_ms = t_ft_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  HF load: {hf_load_ms:.1}ms, fastokens load: {ft_load_ms:.1}ms ({:.1}x)",
        hf_load_ms / ft_load_ms
    );
    println!(
        "  Total download+load: {:.1}ms\n",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // --- Test inputs ---
    let inputs = build_test_inputs();

    // === Correctness (encode) ===
    println!("=== Correctness (encode) ===");
    let mut total_pass = 0;
    let mut total_fail = 0;
    for (label, text) in &inputs {
        let hf_ids = hf_encode(&hf_tok, text);
        let ft_ids = ft_tok.encode_with_special_tokens(text, true)?;

        if hf_ids == ft_ids {
            println!("  PASS {label:40} tokens={}", hf_ids.len());
            total_pass += 1;
        } else {
            println!(
                "  FAIL {label:40} hf={} ft={}",
                hf_ids.len(),
                ft_ids.len()
            );
            for (i, (h, f)) in hf_ids.iter().zip(ft_ids.iter()).enumerate() {
                if h != f {
                    println!("    first diff at pos {i}: hf={h} ft={f}");
                    break;
                }
            }
            if hf_ids.len() != ft_ids.len() {
                println!(
                    "    length mismatch: hf={} ft={}",
                    hf_ids.len(),
                    ft_ids.len()
                );
            }
            total_fail += 1;
        }
    }
    println!(
        "  Result: {total_pass} pass, {total_fail} fail / {} total\n",
        inputs.len()
    );

    // === Correctness (encode, no special tokens) ===
    println!("=== Correctness (encode, add_special_tokens=false) ===");
    let mut ns_pass = 0;
    let mut ns_fail = 0;
    for (label, text) in &inputs {
        let hf_ids: Vec<u32> = hf_tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        let ft_ids = ft_tok.encode_with_special_tokens(text.as_str(), false)?;
        if hf_ids == ft_ids {
            ns_pass += 1;
        } else {
            println!("  FAIL {label:40} hf={} ft={}", hf_ids.len(), ft_ids.len());
            ns_fail += 1;
        }
    }
    println!(
        "  Result: {ns_pass} pass, {ns_fail} fail / {} total\n",
        inputs.len()
    );

    // === Correctness (decode) ===
    println!("=== Correctness (decode) ===");
    let mut dec_pass = 0;
    let mut dec_fail = 0;
    for (label, text) in &inputs {
        let hf_ids = hf_encode(&hf_tok, text);
        let hf_decoded = hf_tok
            .decode(&hf_ids, true)
            .unwrap_or_else(|_| "<decode error>".to_string());
        let ft_decoded = ft_tok.decode(&hf_ids, true)?;

        if hf_decoded == ft_decoded {
            println!("  PASS {label:40} len={}", hf_decoded.len());
            dec_pass += 1;
        } else {
            println!("  FAIL {label:40}");
            let common = hf_decoded
                .chars()
                .zip(ft_decoded.chars())
                .take_while(|(a, b)| a == b)
                .count();
            println!("    first diff at char {common}");
            println!(
                "    hf: ...{}",
                &hf_decoded[hf_decoded.len().saturating_sub(40)..]
            );
            println!(
                "    ft: ...{}",
                &ft_decoded[ft_decoded.len().saturating_sub(40)..]
            );
            dec_fail += 1;
        }
    }
    println!(
        "  Result: {dec_pass} pass, {dec_fail} fail / {} total\n",
        inputs.len()
    );

    // === Correctness: streaming decode (token-by-token) ===
    println!("=== Correctness (streaming decode) ===");
    // Simulates the inference engine's streaming output path:
    // decode tokens one at a time, concatenate, compare to bulk decode
    let stream_tests = [
        ("medium_prose", &inputs[3].1),
        ("code_snippet", &inputs[4].1),
        ("chat_template", &inputs[9].1),
    ];
    for (label, text) in &stream_tests {
        let ids = hf_encode(&hf_tok, text);
        // Bulk decode
        let hf_bulk = hf_tok.decode(&ids, false).unwrap();
        let ft_bulk = ft_tok.decode(&ids, false)?;
        // Streaming decode: one token at a time
        let mut hf_stream = String::new();
        let mut ft_stream = String::new();
        for &id in &ids {
            hf_stream.push_str(&hf_tok.decode(&[id], false).unwrap());
            ft_stream.push_str(&ft_tok.decode(&[id], false)?);
        }
        let hf_ok = hf_bulk == hf_stream;
        let ft_ok = ft_bulk == ft_stream;
        let cross_ok = hf_bulk == ft_bulk;
        if hf_ok && ft_ok && cross_ok {
            println!("  PASS {label:40} tokens={}", ids.len());
        } else {
            println!("  FAIL {label:40} hf_stream={hf_ok} ft_stream={ft_ok} cross={cross_ok}");
        }
    }
    println!();

    // === Encode Speed (single) ===
    println!("=== Encode Speed (single) ===");
    println!(
        "  {:40} {:>8} {:>6} {:>10} {:>10} {:>8}",
        "input", "chars", "tokens", "hf(us)", "ft(us)", "speedup"
    );
    let warmup = 20;
    let repeats = 200;
    for (label, text) in &inputs {
        let n_tokens = hf_encode(&hf_tok, text).len();
        // Adaptive repeats for very long inputs
        let r = if text.len() > 100_000 { 5 } else if text.len() > 10_000 { 20 } else { repeats };

        for _ in 0..warmup.min(r) {
            let _ = hf_encode(&hf_tok, text);
            let _ = ft_tok.encode_with_special_tokens(text, true);
        }

        let start = Instant::now();
        for _ in 0..r {
            let _ = hf_encode(&hf_tok, text);
        }
        let hf_us = start.elapsed().as_nanos() as f64 / r as f64 / 1000.0;

        let start = Instant::now();
        for _ in 0..r {
            let _ = ft_tok.encode_with_special_tokens(text, true);
        }
        let ft_us = start.elapsed().as_nanos() as f64 / r as f64 / 1000.0;

        println!(
            "  {label:40} {:>8} {:>6} {:>10.1} {:>10.1} {:>7.1}x",
            text.len(),
            n_tokens,
            hf_us,
            ft_us,
            hf_us / ft_us
        );
    }

    // === Decode Speed (single) ===
    println!("\n=== Decode Speed (single) ===");
    println!(
        "  {:40} {:>8} {:>10} {:>10} {:>8}",
        "input", "tokens", "hf(us)", "ft(us)", "speedup"
    );
    for (label, text) in &inputs {
        let ids = hf_encode(&hf_tok, text);
        let n_tok = ids.len();
        let r = if n_tok > 10_000 { 10 } else { repeats };

        for _ in 0..warmup.min(r) {
            let _ = hf_tok.decode(&ids, true);
            let _ = ft_tok.decode(&ids, true);
        }

        let start = Instant::now();
        for _ in 0..r {
            let _ = hf_tok.decode(&ids, true);
        }
        let hf_us = start.elapsed().as_nanos() as f64 / r as f64 / 1000.0;

        let start = Instant::now();
        for _ in 0..r {
            let _ = ft_tok.decode(&ids, true);
        }
        let ft_us = start.elapsed().as_nanos() as f64 / r as f64 / 1000.0;

        println!(
            "  {label:40} {:>8} {:>10.1} {:>10.1} {:>7.1}x",
            n_tok, hf_us, ft_us, hf_us / ft_us
        );
    }

    // === Batch Encode Speed ===
    println!("\n=== Batch Encode Speed ===");
    let batch_sizes = [1, 4, 16, 64];
    let batch_text = &inputs[3].1; // "medium_prose"
    for &bs in &batch_sizes {
        let texts: Vec<&str> = (0..bs).map(|_| batch_text.as_str()).collect();
        let texts_string: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        for _ in 0..warmup {
            let _ = hf_tok.encode_batch(texts_string.clone(), true);
            let _ = ft_tok.encode_batch(&texts, true);
        }
        let r = if bs >= 64 { 50 } else { repeats };
        let start = Instant::now();
        for _ in 0..r {
            let _ = hf_tok.encode_batch(texts_string.clone(), true);
        }
        let hf_us = start.elapsed().as_nanos() as f64 / r as f64 / 1000.0;
        let start = Instant::now();
        for _ in 0..r {
            let _ = ft_tok.encode_batch(&texts, true);
        }
        let ft_us = start.elapsed().as_nanos() as f64 / r as f64 / 1000.0;
        println!(
            "  batch={bs:<3} x {chars} chars  hf={hf_us:>10.1}us  ft={ft_us:>10.1}us  {:.1}x",
            hf_us / ft_us,
            chars = batch_text.len(),
        );
    }

    // === Single Token Decode (logprobs hot path) ===
    println!("\n=== Single Token Decode (logprobs hot path) ===");
    let sample_ids: Vec<u32> = (100..200).collect();
    for &id in &sample_ids {
        let _ = hf_tok.decode(&[id], false);
        let _ = ft_tok.decode(&[id], false);
    }
    let single_repeats = 2000;
    let start = Instant::now();
    for _ in 0..single_repeats {
        for &id in &sample_ids {
            let _ = hf_tok.decode(&[id], false);
        }
    }
    let hf_us =
        start.elapsed().as_nanos() as f64 / (single_repeats * sample_ids.len()) as f64 / 1000.0;
    let start = Instant::now();
    for _ in 0..single_repeats {
        for &id in &sample_ids {
            let _ = ft_tok.decode(&[id], false);
        }
    }
    let ft_us =
        start.elapsed().as_nanos() as f64 / (single_repeats * sample_ids.len()) as f64 / 1000.0;
    println!(
        "  per-token decode: hf={hf_us:.2}us  ft={ft_us:.2}us  {:.1}x",
        hf_us / ft_us
    );

    // === Concurrent Encode (thread safety) ===
    println!("\n=== Concurrent Encode (8 threads) ===");
    {
        let ft_arc = Arc::new(ft_tok);
        let hf_arc = Arc::new(hf_tok);
        let text = inputs[3].1.clone(); // medium_prose
        let n_threads = 8;
        let per_thread = 100;

        // Verify correctness under contention
        let hf_ref = Arc::clone(&hf_arc);
        let expected = hf_encode(&hf_ref, &text);

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let ft = Arc::clone(&ft_arc);
                let t = text.clone();
                let exp = expected.clone();
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        let ids = ft.encode_with_special_tokens(&t, true).unwrap();
                        assert_eq!(ids, exp, "concurrent encode mismatch");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        println!(
            "  correctness: PASS ({n_threads} threads x {per_thread} encodes)"
        );

        // Speed: concurrent encode
        let ft_c = Arc::clone(&ft_arc);
        let text_c = text.clone();
        let start = Instant::now();
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let ft = Arc::clone(&ft_c);
                let t = text_c.clone();
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        let _ = ft.encode_with_special_tokens(&t, true);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let ft_total_us = start.elapsed().as_micros() as f64;
        let ft_per_us = ft_total_us / (n_threads * per_thread) as f64;

        let hf_c = Arc::clone(&hf_arc);
        let text_c2 = text.clone();
        let start = Instant::now();
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let hf = Arc::clone(&hf_c);
                let t = text_c2.clone();
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        let _ = hf_encode(&hf, &t);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let hf_total_us = start.elapsed().as_micros() as f64;
        let hf_per_us = hf_total_us / (n_threads * per_thread) as f64;

        println!(
            "  throughput:  hf={hf_per_us:.1}us/encode  ft={ft_per_us:.1}us/encode  {:.1}x",
            hf_per_us / ft_per_us
        );
    }

    println!("\nDone.");
    Ok(())
}


fn hf_encode(tok: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, true).unwrap().get_ids().to_vec()
}


fn build_test_inputs() -> Vec<(&'static str, String)> {
    vec![
        ("tiny (5 chars)", "Hello".to_string()),
        (
            "short_english (50)",
            "The quick brown fox jumps over the lazy dog nearby.".to_string(),
        ),
        (
            "short_chinese (50)",
            "今天天气真好，我们一起去公园散步吧，顺便买点水果回来做晚饭。".to_string(),
        ),
        (
            "medium_prose (500)",
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It achieves memory safety without garbage collection through its ownership system, \
             which enforces strict rules at compile time. The borrow checker ensures that references \
             are always valid, preventing data races and dangling pointers. Rust's type system and \
             pattern matching provide powerful abstractions without runtime overhead. The language \
             has gained significant adoption in areas like web services, embedded systems, and \
             operating system development. Companies like Mozilla, Google, Microsoft, and Amazon \
             have integrated Rust into their infrastructure for performance-critical components."
                .to_string(),
        ),
        (
            "code_snippet (400)",
            r#"fn main() {
    let mut v = Vec::new();
    for i in 0..100 {
        v.push(i * i);
    }
    let sum: i64 = v.iter().sum();
    println!("Sum of squares: {}", sum);

    let filtered: Vec<_> = v.iter().filter(|&&x| x > 50).collect();
    println!("Above 50: {:?}", &filtered[..5]);

    let map: std::collections::HashMap<i64, i64> = v.iter()
        .enumerate()
        .map(|(i, &val)| (i as i64, val))
        .collect();
    println!("Map size: {}", map.len());
}"#
            .to_string(),
        ),
        (
            "mixed_multilingual (600)",
            "Machine learning models process text through tokenization. \
             分词是自然语言处理的基础步骤。Le tokenisation est une étape fondamentale du NLP. \
             トークン化は自然言語処理の基本的なステップです。 \
             토큰화는 자연어 처리의 기본 단계입니다。\
             Die Tokenisierung ist ein grundlegender Schritt in der NLP. \
             Токенизация — это фундаментальный шаг в обработке естественного языка. \
             La tokenización es un paso fundamental en el procesamiento del lenguaje natural. \
             A tokenização é uma etapa fundamental no processamento de linguagem natural."
                .to_string(),
        ),
        (
            "long_repeat (2K)",
            "The quick brown fox jumps over the lazy dog. ".repeat(45),
        ),
        ("long_unique (4K)", build_long_unique_text(4_000)),
        ("very_long (8K)", build_long_unique_text(8_000)),
        (
            "chat_template",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nWhat is the capital of France? Please explain in detail \
             with historical context and modern significance.<|im_end|>\n\
             <|im_start|>assistant\n"
                .to_string(),
        ),
        // --- Long context tests ---
        ("long_32K", build_long_unique_text(32_000)),
        ("long_64K", build_long_unique_text(64_000)),
        ("long_200K", build_long_unique_text(200_000)),
        (
            "long_code_16K",
            build_long_code(16_000),
        ),
        (
            "multi_turn_chat_8K",
            build_multi_turn_chat(80),
        ),
        (
            "multi_turn_chat_32K",
            build_multi_turn_chat(320),
        ),
        (
            "long_chinese_32K",
            build_long_chinese(32_000),
        ),
    ]
}


fn build_long_unique_text(target_chars: usize) -> String {
    let sentences = [
        "Artificial intelligence is transforming how we interact with technology.",
        "Quantum computing promises to revolutionize cryptography and drug discovery.",
        "Climate change remains one of the most pressing challenges of our time.",
        "The James Webb Space Telescope has revealed unprecedented views of the cosmos.",
        "Renewable energy sources like solar and wind continue to grow exponentially.",
        "Neural networks can now generate human-quality text, images, and music.",
        "The global supply chain has become increasingly complex and interconnected.",
        "Advances in biotechnology are enabling personalized medicine approaches.",
        "Cybersecurity threats have evolved alongside our growing digital infrastructure.",
        "Space exploration companies are making access to orbit more affordable.",
        "Programming languages continue to evolve toward safety and expressiveness.",
        "The intersection of AI and robotics is creating new manufacturing paradigms.",
        "Edge computing brings processing power closer to where data is generated.",
        "Open source software has become the backbone of modern technology stacks.",
        "Graph neural networks are advancing our understanding of molecular structures.",
    ];
    let mut result = String::with_capacity(target_chars + 200);
    let mut i = 0;
    while result.len() < target_chars {
        result.push_str(sentences[i % sentences.len()]);
        result.push(' ');
        i += 1;
    }
    // Truncate at a valid UTF-8 char boundary
    let mut end = target_chars.min(result.len());
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    result.truncate(end);
    result
}


fn build_long_code(target_chars: usize) -> String {
    let snippets = [
        "fn process_data(input: &[u8]) -> Vec<u8> {\n    let mut output = Vec::new();\n    for &byte in input {\n        output.push(byte ^ 0xFF);\n    }\n    output\n}\n\n",
        "struct Config {\n    name: String,\n    value: i64,\n    enabled: bool,\n}\n\nimpl Config {\n    fn new(name: &str) -> Self {\n        Self { name: name.to_string(), value: 0, enabled: true }\n    }\n}\n\n",
        "#[derive(Debug, Clone)]\nenum Token {\n    Number(f64),\n    String(String),\n    Ident(String),\n    Punct(char),\n}\n\nfn tokenize(input: &str) -> Vec<Token> {\n    let mut tokens = Vec::new();\n    // TODO: implement\n    tokens\n}\n\n",
        "use std::collections::HashMap;\n\nfn count_words(text: &str) -> HashMap<&str, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n        *counts.entry(word).or_insert(0) += 1;\n    }\n    counts\n}\n\n",
        "async fn fetch_url(url: &str) -> Result<String, Box<dyn std::error::Error>> {\n    let response = reqwest::get(url).await?;\n    let body = response.text().await?;\n    Ok(body)\n}\n\n",
        "/// A simple binary search tree.\npub struct BstNode<T: Ord> {\n    value: T,\n    left: Option<Box<BstNode<T>>>,\n    right: Option<Box<BstNode<T>>>,\n}\n\nimpl<T: Ord> BstNode<T> {\n    pub fn insert(&mut self, val: T) {\n        if val < self.value {\n            match &mut self.left {\n                Some(left) => left.insert(val),\n                None => self.left = Some(Box::new(BstNode { value: val, left: None, right: None })),\n            }\n        }\n    }\n}\n\n",
    ];
    let mut result = String::with_capacity(target_chars + 500);
    let mut i = 0;
    while result.len() < target_chars {
        result.push_str(snippets[i % snippets.len()]);
        i += 1;
    }
    // Truncate at a valid UTF-8 char boundary
    let mut end = target_chars.min(result.len());
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    result.truncate(end);
    result
}


fn build_multi_turn_chat(turns: usize) -> String {
    let user_messages = [
        "What is the capital of France?",
        "Can you explain how neural networks work?",
        "Write me a Python function to sort a list.",
        "What are the benefits of Rust over C++?",
        "Explain quantum entanglement in simple terms.",
        "How does a transformer model work?",
        "What is the difference between TCP and UDP?",
        "Can you help me debug this code: for i in range(10): print(i",
    ];
    let assistant_messages = [
        "The capital of France is Paris. It has been the capital since the 10th century and is known for landmarks like the Eiffel Tower and the Louvre Museum.",
        "Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process information through weighted connections.",
        "Here's a Python sorting function:\n```python\ndef sort_list(lst):\n    return sorted(lst)\n```\nYou can also use `lst.sort()` for in-place sorting.",
        "Rust offers several advantages: memory safety without garbage collection, zero-cost abstractions, fearless concurrency, and a strong type system that catches bugs at compile time.",
        "Quantum entanglement is when two particles become linked so that measuring one instantly affects the other, regardless of distance. Einstein called it 'spooky action at a distance'.",
        "A transformer uses self-attention mechanisms to process all parts of the input simultaneously, rather than sequentially. This allows it to capture long-range dependencies efficiently.",
        "TCP provides reliable, ordered delivery with error checking and flow control. UDP is faster but unreliable — it just sends packets without guarantees. Use TCP for web/email, UDP for streaming/gaming.",
        "You're missing a closing parenthesis. It should be:\n```python\nfor i in range(10):\n    print(i)\n```",
    ];
    let mut result = String::new();
    for i in 0..turns {
        result.push_str("<|im_start|>user\n");
        result.push_str(user_messages[i % user_messages.len()]);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>assistant\n");
        result.push_str(assistant_messages[i % assistant_messages.len()]);
        result.push_str("<|im_end|>\n");
    }
    result
}


fn build_long_chinese(target_chars: usize) -> String {
    let paragraphs = [
        "人工智能技术正在深刻改变着我们的生活方式。从智能手机上的语音助手到自动驾驶汽车，从医疗诊断到金融风控，AI的应用已经渗透到社会的各个角落。",
        "机器学习是人工智能的核心技术之一。通过大量数据的训练，机器可以自动发现规律并做出预测。深度学习作为机器学习的子领域，利用多层神经网络处理复杂的模式识别任务。",
        "自然语言处理是AI领域最具挑战性的方向之一。理解人类语言不仅需要词汇和语法知识，还需要常识推理和上下文理解能力。大型语言模型的出现极大地推动了这一领域的发展。",
        "量子计算有望在未来彻底改变计算范式。与传统计算机使用比特不同，量子计算机使用量子比特，可以同时处于多个状态，从而在某些问题上实现指数级的加速。",
        "可持续发展是当今世界面临的重要议题。如何在经济增长与环境保护之间找到平衡，如何应对气候变化带来的挑战，这些问题需要全球合作共同解决。",
    ];
    let mut result = String::with_capacity(target_chars + 500);
    let mut i = 0;
    while result.len() < target_chars {
        result.push_str(paragraphs[i % paragraphs.len()]);
        result.push('\n');
        i += 1;
    }
    // Truncate at a valid UTF-8 char boundary
    let mut end = target_chars.min(result.len());
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    result.truncate(end);
    result
}
