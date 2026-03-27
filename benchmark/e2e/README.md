# E2E Benchmark

Docker 化的端到端推理性能对比，每个引擎跑在自己的容器里。

## 前置

```bash
pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git"
```

## 用法

```bash
# 所有引擎 (GPU)
./benchmark/e2e/bench.sh all

# 所有引擎 (CPU)
./benchmark/e2e/bench.sh all --cpu

# 单个引擎
./benchmark/e2e/bench.sh prelude
./benchmark/e2e/bench.sh vllm
./benchmark/e2e/bench.sh sglang
./benchmark/e2e/bench.sh vllm.rs
GGUF_MODEL=/path/to/model.gguf ./benchmark/e2e/bench.sh llama.cpp

# 多个引擎对比
./benchmark/e2e/bench.sh prelude vllm sglang

# CPU 模式
./benchmark/e2e/bench.sh prelude --cpu
./benchmark/e2e/bench.sh llama.cpp --cpu

# 自定义参数
INPUT_TOKENS=512 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=4 \
  ./benchmark/e2e/bench.sh prelude vllm
```

## 引擎和 Docker Image

| 引擎 | Image | GPU | CPU |
|------|-------|-----|-----|
| prelude | `prelude-dev` (根 Dockerfile) | Y | Y (`--device cpu`) |
| vllm | `vllm/vllm-openai:latest` | Y | Y (需自建 `vllm-cpu`) |
| sglang | `lmsysorg/sglang:latest` | Y | Y (需自建 `sglang-cpu`) |
| llama.cpp | `prelude-others` (Dockerfile.others) | Y (`-ngl 99`) | Y (`-ngl 0`) |
| vllm.rs | `prelude-others` (Dockerfile.others) | Y | N |

首次运行自动 build image，之后复用。

## CPU Image 构建 (一次性)

vLLM 和 SGLang 的 CPU 版需要从上游源码构建：

```bash
# vLLM CPU
cd /path/to/vllm
docker build -f docker/Dockerfile.cpu -t vllm-cpu --target vllm-openai .

# SGLang CPU
cd /path/to/sglang
docker build -f docker/xeon.Dockerfile -t sglang-cpu .
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `Qwen/Qwen3-0.6B` | HuggingFace 模型 |
| `GGUF_MODEL` | - | GGUF 文件路径 (llama.cpp 必需) |
| `INPUT_TOKENS` | `128` | 输入 token 数 |
| `OUTPUT_TOKENS` | `1` | 输出 token 数 (1 = prefill-only) |
| `MAX_REQUESTS` | `200` | 总请求数 |
| `CONCURRENCY` | `1` | 并发数 |
| `GPU` | `0` | GPU 编号 |
| `PORT` | `8000` | 服务端口 |
| `HF_TOKEN` | - | HuggingFace token (gated 模型) |
| `VLLM_RS_DIR` | `../vllm.rs` | vllm.rs 源码路径 |

## 结果

输出到 `benchmark/e2e/results/`，JSON 格式，文件名包含引擎和时间戳：

```
results/
├── prelude_20260327_120000.json
├── prelude-cpu_20260327_120500.json
├── vllm_20260327_121000.json
└── sglang_20260327_121500.json
```
