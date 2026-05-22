INPUT_TOKENS=1900 OUTPUT_TOKENS=3 MAX_REQUESTS=400 CONCURRENCY=32 \
    MODEL=Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix \
    ./benchmark/genai-bench/bench.sh prelude vllm --gpu