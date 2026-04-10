# Benchmark Results

## Setup

- **Date**: 2026-04-07
- **Model**: Qwen/Qwen3-0.6B (BF16)
- **Tool**: genai-bench 0.0.3
- **Engines**: AGInfer (native binary), vLLM 0.18.0 (Docker), SGLang (Docker)

| Setup | CPU | GPU |
|---|---|---|
| H200 | AMD EPYC 9575F 64-Core | NVIDIA H200 (single GPU) |
| B300 | Intel Xeon 6776P | NVIDIA B300 SXM6 AC (single GPU) |

SGLang and vLLM run in Docker containers (`lmsysorg/sglang:latest`, `vllm/vllm-openai:latest`). AGInfer runs as a native binary.

---

## Prefill-Only (128 in → 1 out, c=1, 100 requests)

| GPU | Engine | Startup (s) | TTFT (s) | E2E (s) | Input tok/s | RPM |
|---|---|---|---|---|---|---|
| H200 | **AGInfer** | **4** | **0.0068** | **0.0069** | **14,520.9** | **6,514.8** |
| H200 | vLLM | 60 | 0.0107 | 0.0108 | 9,926.3 | 4,452.4 |
| H200 | SGLang | 38 | 0.0407 | 0.0409 | 3,098.2 | 1,389.7 |
| B300 | vLLM | 108 | 0.0199 | 0.0201 | 5,756.2 | 2,580.7 |
| B300 | SGLang | 120 | 0.1404 | 0.1405 | 931.6 | 418.1 |

**H200**: AGInfer vs vLLM — **1.46× RPM, 1.57× lower TTFT, 15× faster startup**
**H200**: AGInfer vs SGLang — **4.69× RPM, 5.99× lower TTFT**

---

## Decode (32 in → 32 out, c=4, 400 requests)

| GPU | Engine | Startup (s) | TTFT (s) | TPOT (s) | Output tok/s | RPM |
|---|---|---|---|---|---|---|
| H200 | **AGInfer** | **2** | **0.0135** | 0.0441 | 92.0 | 172.5 |
| H200 | vLLM | 50 | 0.0215 | **0.0016** | 1,713.8 | 3,213.5 |
| B300 | vLLM | 110 | 0.0326 | 0.0028 | 995.7 | 1,866.9 |
| B300 | SGLang | 106 | 0.1851 | 0.0015 | 539.5 | 1,011.6 |

*H200 SGLang result missing — benchmark timed out.*

---

## Decode (128 in → 32 out, c=4, 400 requests)

| GPU | Engine | Startup (s) | TTFT (s) | TPOT (s) | Output tok/s | RPM |
|---|---|---|---|---|---|---|
| H200 | **AGInfer** | 14 | **0.0178** | 0.0442 | 91.4 | 171.3 |
| H200 | vLLM | 42 | 0.0295 | **0.0016** | 1,474.5 | 2,764.8 |
| H200 | SGLang | 32 | 0.0669 | 0.0018 | 983.5 | 1,844.0 |
| B300 | vLLM | 114 | 0.0465 | 0.0052 | 487.4 | 914.0 |
| B300 | SGLang | 100 | 0.1338 | 0.0014 | 691.4 | 1,296.3 |

---

## Analysis

### Prefill
AGInfer leads across the board: **1.46× vLLM and 4.69× SGLang** in RPM on H200. TTFT of 6.8ms beats vLLM (10.7ms) and SGLang (40.7ms). Startup is 4s vs 60s/38s — the native binary has no Python runtime to initialise.

### Decode
AGInfer's TPOT of ~44ms is significantly slower than vLLM (1.6ms) and SGLang (1.8ms). This is a known issue: the decode path runs `batch_decode_paged` eagerly every step with 6 tensor allocations and H2D copies per step. CUDA graph capture is implemented but currently blocked by a stream isolation issue (weight tensors created on a different stream than the capture stream). Fixing this is the top decode performance priority.

### B300 (Blackwell)
Both vLLM and SGLang show 33–70% of their H200 throughput on B300, reflecting immature Blackwell support rather than hardware capability. SGLang's default image failed entirely on B300 and required a CUDA 13 image; vLLM's decode TPOT regressed 3×. These B300 numbers should not be used for hardware comparisons until the frameworks mature. AGInfer B300 results are pending.

---

## Notes

- H200 run: GPUs 0, 1, 4–7 were occupied; only GPU 2 was idle
- B300 run: single GPU; original `lmsysorg/sglang:latest` (CUDA 12.9) failed with `no kernel image is available for execution on the device` (SM_120 not supported)
- All results use a single GPU per engine — no tensor parallelism
