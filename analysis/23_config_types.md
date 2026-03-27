# Prelude 基础数据结构分析：配置、类型与常量

## 文件清单

| 文件 | 职责 |
|------|------|
| `crates/prelude-core/src/config.rs` | 全局运行时配置，从 `PRELUDE_*` 环境变量解析 |
| `crates/prelude-core/src/constants.rs` | 全局常量（seed、GGUF 回退默认值、mock 引擎限制） |
| `crates/prelude-core/src/types.rs` | OpenAI 兼容 API 类型 + 内部引擎请求/结果类型 |
| `crates/prelude-core/src/scheduler/state.rs` | 调度器内部类型（Sequence, SamplingParams, FinishReason 等） |
| `crates/prelude-core/src/lib.rs` | 模块组织与 re-export |

---

## 一、配置系统 (`config.rs`)

### 设计原则

- 所有 `PRELUDE_*` 环境变量 **只在 `EngineConfig::from_env()` 中解析一次**，之后以值传递
- 全局静态配置通过 `OnceLock` 存储（`GLOBAL_RUNTIME`, `GLOBAL_CACHE`），供模型层代码静态访问
- 严禁在 config.rs 以外直接调用 `std::env::var("PRELUDE_*")`

### 全局访问器

```
static GLOBAL_RUNTIME: OnceLock<RuntimeConfig>
static GLOBAL_CACHE: OnceLock<CacheConfig>

init_global_config(&EngineConfig)     // 在 loading/mod.rs 中调用，启动时初始化
global_runtime() -> Option<&'static RuntimeConfig>   // 被 ops/gpu/kv_cache.rs, models/ 使用
global_cache_config() -> Option<&'static CacheConfig> // 被 cache/manager.rs 使用
```

### `EngineConfig` -- 顶层配置根节点

```rust
pub struct EngineConfig {
    pub cache: CacheConfig,
    pub sampling: SamplingDefaults,
    pub runtime: RuntimeConfig,
    pub adaptive: AdaptiveConfig,
}
```

- **构建**: `EngineConfig::from_env()` -- 解析环境变量，调用 `validate()`
- **使用位置**: `loading/mod.rs`（构建引擎）, `engine/engine.rs`, `runtime/cuda_graph.rs`, `prelude-server/src/main.rs`
- **验证规则**:
  - `paged_block_size > 0`
  - `prefix_block_size > 0`
  - `arrival_alpha` 在 (0, 1]
  - `gpu_alpha` 在 (0, 1]

### `CacheConfig` -- KV 缓存池配置

```rust
pub struct CacheConfig {
    pub paged_block_size: usize,       // 默认: flash-attn-v3/v4/flashinfer 时 128，否则 16
    pub paged_attn_blocks: usize,      // 默认 0（自动根据 GPU 显存计算）
    pub gpu_memory_utilization: f32,    // 硬编码 0.4（vLLM 用 0.9）
    pub prefix_cache_blocks: usize,    // 默认 0（禁用前缀缓存）
    pub prefix_block_size: usize,      // 默认 64
    pub deltanet_pool_slots: u32,      // 默认 8
}
```

| 环境变量 | 字段 | 默认值 | 说明 |
|----------|------|--------|------|
| `PRELUDE_PAGED_BLOCK_SIZE` | `paged_block_size` | 128 或 16 | **feature flag 条件编译**: `flash-attn-v3`/`flash-attn-v4`/`flashinfer` 启用时为 128 |
| `PRELUDE_PAGED_ATTN_BLOCKS` | `paged_attn_blocks` | 0 | 0 表示从 `gpu_memory_utilization` 自动推算 |
| （无环境变量） | `gpu_memory_utilization` | 0.4 | 硬编码，仅当 blocks=0 时生效 |
| `PRELUDE_PREFIX_CACHE_BLOCKS` | `prefix_cache_blocks` | 0 | 0=禁用前缀缓存 |
| `PRELUDE_PREFIX_BLOCK_SIZE` | `prefix_block_size` | 64 | |
| `PRELUDE_DELTANET_POOL_SLOTS` | `deltanet_pool_slots` | 8 | DeltaNet 线性 attention 状态池大小 |

- **使用位置**: `cache/manager.rs`

### `SamplingDefaults` -- 采样参数默认值

```rust
pub struct SamplingDefaults {
    pub temperature: f32,    // 默认 0.7
    pub top_p: f32,          // 默认 1.0
    pub max_new_tokens: u32, // 默认 4096
}
```

| 环境变量 | 默认值 |
|----------|--------|
| `PRELUDE_DEFAULT_TEMPERATURE` | 0.7 |
| `PRELUDE_DEFAULT_TOP_P` | 1.0 |
| `PRELUDE_DEFAULT_MAX_TOKENS` | 4096 |

- **使用位置**: 仅在 `EngineConfig` 内部聚合，目前没有被其他文件直接引用（通过 `EngineConfig.sampling` 间接使用）

### `RuntimeConfig` -- 运行时特性开关

```rust
pub struct RuntimeConfig {
    pub device: String,              // "auto"/"cpu"/"cuda"/"cuda:N"
    pub sync_timing: bool,           // CUDA 同步计时（profiling 用）
    pub force_varlen_prefill: bool,  // 强制变长 prefill 路径
    pub fused_kv_cache_write: bool,  // 融合 K-Norm + RoPE + KV 写入 kernel
    pub cpu_thread_bind: Option<String>,  // NUMA 绑核
    pub dtype: Option<String>,       // 数据类型覆盖（"f32"/"bf16"，None 自动选择）
    pub cuda_graph: bool,            // CUDA graph 加速 decode（默认 true）
    pub cuda_graph_max_bs: usize,    // CUDA graph 最大 batch size
}
```

**CLI 参数**（覆盖默认值）:

| CLI 参数 | 默认值 | 说明 |
|----------|--------|------|
| `--device` | `auto` | 设备选择：auto, cpu, cuda, cuda:N |
| `--cuda-graph` | `true` | CUDA graph 加速 decode |
| `--dtype` | None (自动) | 数据类型：f32 或 bf16 |

**环境变量**（内部调优，无 CLI 对应）:

| 环境变量 | 解析方式 | 默认值 |
|----------|----------|--------|
| `PRELUDE_SYNC_TIMING` | `parse_env_bool`（1/true/yes/on） | false |
| `PRELUDE_FORCE_VARLEN_PREFILL` | `parse_env_bool` | false |
| `PRELUDE_FUSED_KV_CACHE_WRITE` | `parse_env_bool_eq1`（仅 "1"） | false |
| `SGLANG_CPU_OMP_THREADS_BIND` | 字符串 | None |
| `PRELUDE_CUDA_GRAPH_MAX_BS` | usize | 32 |

注意: `dtype` 字段在 `from_env()` 中硬编码为 `None`，由 CLI `--dtype` 或 `engine/device.rs` 后续设置。

- **使用位置**: `engine/device.rs`（设备选择）, `ops/gpu/kv_cache.rs`（fused kernel 开关）

### `AdaptiveConfig` -- 自适应批调度参数

```rust
pub struct AdaptiveConfig {
    pub arrival_alpha: f64,      // 到达率 EWMA 平滑因子
    pub gpu_alpha: f64,          // GPU 时间 EWMA 平滑因子
    pub initial_lambda: f64,     // 冷启动初始到达率假设 (req/s)
    pub max_instant_rate: f64,   // 瞬时速率上限
}
```

| 环境变量 | 默认值 |
|----------|--------|
| `PRELUDE_ADAPTIVE_ARRIVAL_ALPHA` | 0.5 |
| `PRELUDE_ADAPTIVE_GPU_ALPHA` | 0.4 |
| `PRELUDE_ADAPTIVE_INITIAL_LAMBDA` | 1000.0 |
| `PRELUDE_ADAPTIVE_MAX_RATE` | 10000.0 |

- **使用位置**: `scheduler/adaptive.rs`

### 解析辅助函数

| 函数 | 语义 |
|------|------|
| `parse_env_usize(name, default)` | 解析 usize 环境变量 |
| `parse_env_u32(name, default)` | 解析 u32 |
| `parse_env_f32(name, default)` | 解析 f32 |
| `parse_env_f64(name, default)` | 解析 f64 |
| `parse_env_bool(name)` | 匹配 "1"/"true"/"yes"/"on"（大小写不敏感） |
| `parse_env_bool_eq1(name)` | 仅匹配 "1"（严格） |

---

## 二、常量 (`constants.rs`)

```rust
pub const DEFAULT_SEED: u64 = 42;
pub const GGUF_DEFAULT_VOCAB_SIZE: usize = 151936;           // Qwen3 词表大小
pub const GGUF_INTERMEDIATE_SIZE_MULTIPLIER: usize = 3;
pub const PSEUDO_ENGINE_MAX_TOKENS: u32 = 256;
```

| 常量 | 使用位置 |
|------|----------|
| `DEFAULT_SEED` | `engine/mod.rs`, `engine/forward/generate.rs`（采样 seed 回退） |
| `GGUF_DEFAULT_VOCAB_SIZE` | `models/qwen3/gguf.rs`（GGUF 元数据缺失时回退） |
| `GGUF_INTERMEDIATE_SIZE_MULTIPLIER` | `models/qwen3/gguf.rs`, `qwen3_5_gguf.rs`（计算 FFN 中间层大小） |
| `PSEUDO_ENGINE_MAX_TOKENS` | `engine/pseudo.rs`（mock 引擎最大生成 token 数） |

---

## 三、API 与引擎类型 (`types.rs`)

### 3.1 Re-export 关系

`types.rs` 从 `scheduler/state.rs` re-export 两个核心类型：
```rust
pub use crate::scheduler::{FinishReason, SamplingParams};
```

`lib.rs` 则 `pub use types::*`，使所有类型对外可见。

### 3.2 Logprobs 类型族

内部表示：

| 类型 | 用途 |
|------|------|
| `TokenLogprobInfo` | 引擎内部每个 token 的 logprob，包含 token 文本、ID、logprob、top_logprobs 列表 |

OpenAI Completions API 格式：

| 类型 | 用途 |
|------|------|
| `CompletionLogprobs` | `/v1/completions` 响应中的 logprobs（扁平 dict 格式） |
| `PromptLogprobEntry` | prompt_logprobs 中单个 token 的条目（vLLM 扩展） |

OpenAI Chat API 格式：

| 类型 | 用途 |
|------|------|
| `ChatCompletionLogprobs` | `/v1/chat/completions` 响应中的 logprobs（结构化列表） |
| `ChatCompletionLogprobContent` | 单个 token 的 logprob 详情 |
| `ChatCompletionTopLogprob` | top_logprobs 列表中的候选 token |

**转换链**: `TokenLogprobInfo`（引擎内部） --> `CompletionLogprobs` 或 `ChatCompletionLogprobs`（在 `prelude-server/src/logprobs.rs` 中转换）

### 3.3 Completion API 请求/响应

```
CompletionPrompt (enum: Single/Batch)
    └── CompletionRequest
            ├── model, prompt, max_tokens, temperature, top_p, stop
            ├── stream, stream_options, logprobs, prompt_logprobs, seed
            └── n, frequency_penalty, presence_penalty (TODO: 未实现)

CompletionResponse
    ├── CompletionChoice
    │       ├── text, finish_reason
    │       ├── logprobs: Option<CompletionLogprobs>
    │       └── prompt_logprobs, prompt_token_ids (vLLM 扩展)
    └── Usage { prompt_tokens, completion_tokens, total_tokens }
```

### 3.4 Chat Completion API 请求/响应

```
ChatMessage
    ├── role: String
    ├── content: ChatMessageContent (enum: Text/Other)
    ├── name, tool_call_id, tool_calls (扩展字段，目前验证中拒绝)
    └── validate_text_only() -- 仅允许纯文本消息

ChatCompletionRequest
    ├── model, messages: Vec<ChatMessage>
    ├── max_completion_tokens / max_tokens (新旧两个字段兼容)
    ├── temperature, top_p, stop, stream, stream_options
    ├── logprobs: Option<bool>, top_logprobs: Option<u32>
    └── n, frequency_penalty, presence_penalty, response_format (TODO)

ChatCompletionResponse
    ├── ChatCompletionChoice
    │       ├── message / delta: ChatMessageOut (非流式/流式)
    │       ├── finish_reason
    │       └── logprobs: Option<ChatCompletionLogprobs>
    └── Usage (Optional，流式最后一个 chunk 才有)
```

- `ChatMessageContent` 使用 `#[serde(untagged)]` 自动检测纯文本 vs 结构化内容
- `StreamOptions.include_usage` 控制流式最后是否返回 usage
- `ResponseFormat` 预留 JSON mode（标记 TODO）

### 3.5 流式事件

```rust
pub enum StreamEvent {
    Started,                                    // prefill 完成，开始生成
    Token { text, logprobs },                   // 每个 token 的增量
    Finished { finish_reason, usage, metrics },  // 生成结束
    Error { message },                          // 出错
}
```

使用位置：`runtime/gpu_continuous.rs`, `runtime/cpu_continuous.rs`, `engine/scheduled.rs`, `prelude-server/src/sse.rs`（SSE 推送）

### 3.6 内部引擎请求/结果

**生成任务**：

```
GenerateRequest
    ├── request_id, model
    ├── input: PromptInput (enum: Text/TokenIds)
    ├── sampling: SamplingParams (来自 scheduler)
    ├── max_new_tokens, stop: StopConfig, seed
    ├── deadline_ms (超时)
    └── logprobs, prompt_logprobs

GenerateResult
    ├── model, output_token_ids, output_text
    ├── finish_reason: FinishReason (来自 scheduler)
    ├── usage: Usage, metrics: DecodeMetrics
    └── token_logprobs, prompt_token_logprobs
```

**辅助类型**：
- `PromptInput` -- 支持文本或预 tokenize 的 token ID 输入
- `StopConfig` -- 停止词（字符串列表 + token ID 列表）
- `DecodeMetrics` -- 性能指标（TTFT, prefill, decode, total 毫秒）

### 3.7 分类 (Classification) API

```
ClassificationInput (enum, #[serde(untagged)])
    ├── Single(String)
    ├── Batch(Vec<String>)
    ├── TokenIds(Vec<u32>)
    └── BatchTokenIds(Vec<Vec<u32>>)

ClassificationRequest -- API 请求体
    ├── model, input, messages (互斥)
    └── get_inputs() -> ClassificationInputs

ClassificationInputs (enum) -- 内部标准化表示
    ├── Texts(Vec<String>)
    └── TokenIds(Vec<Vec<u32>>)

ClassifyRequest -- 内部引擎请求
ClassifyResult -- 内部引擎结果
ClassificationResult -- 单条分类结果 (index, label, probs, num_classes)
ClassificationUsage -- 分类 token 用量
ClassificationResponse -- API 响应体
```

使用位置：`engine/forward/classify.rs`, `runtime/gpu_batch.rs`, `prelude-server/src/routes/classify.rs`

### 3.8 嵌入 (Embedding) API

```
EmbeddingInput (enum, #[serde(untagged)])
    ├── Single/Batch/TokenIds/BatchTokenIds

EmbeddingRequest -- API 请求体
    ├── input, model, encoding_format, dimensions
    ├── get_inputs() -> ClassificationInputs (复用分类的内部类型!)
    └── validate_public_request()

EmbedRequest -- 内部引擎请求 (复用 ClassificationInputs)
EmbedResult -- 内部引擎结果
EmbeddingData -- 单个嵌入向量
EmbeddingValue (enum) -- Float 数组或 Base64 编码
EmbeddingObject -- 响应中单个嵌入对象
EmbeddingResponse -- API 响应体
EmbeddingUsage -- 嵌入 token 用量
```

注意：`EmbedRequest.inputs` 字段类型是 `ClassificationInputs`，嵌入和分类共用同一个内部输入抽象。

### 3.9 其他 API 类型

| 类型 | 用途 |
|------|------|
| `ModelListResponse` | `/v1/models` 响应 |
| `ModelCard` | 单个模型信息 |
| `ModelInfo` | 内部模型元数据 |
| `HealthResponse` | `/health` 端点响应 |
| `Usage` | Token 使用量（跨 API 共用） |

### 3.10 验证辅助函数

| 函数 | 用途 |
|------|------|
| `validate_single_choice(n)` | 确保 n=1 或 None |
| `reject_if_present(field, is_present)` | 拒绝当前版本未支持的字段 |

被 `CompletionRequest`、`ChatCompletionRequest`、`EmbeddingRequest` 的 `validate_public_request()` 使用。

---

## 四、调度器类型 (`scheduler/state.rs`)

### 4.1 核心调度器类型

```rust
pub struct SamplingParams {
    pub temperature: f32,          // 默认 0.7
    pub top_p: f32,                // 默认 1.0
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
}

pub enum FinishReason { Stop, Length, Eos, Cancelled }
pub enum SequenceStatus { Waiting, Prefilling, Decoding, Finished }
pub enum SeqFinishReason { Stop, Length, Eos, Abort(String) }
pub enum ForwardMode { Prefill, Decode, Mixed }
pub enum SchedulePolicy { Fcfs }
```

`FinishReason` 和 `SamplingParams` 通过 `types.rs` re-export 到 crate 根。

`SeqFinishReason` -> `FinishReason` 实现了 `From` trait 转换。

### 4.2 `Sequence` -- 单个推理请求

```rust
pub struct Sequence {
    pub request_id: String,
    pub status: SequenceStatus,
    pub input_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_new_tokens: u32,
    pub stop_strings: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub finish_reason: Option<SeqFinishReason>,
    pub arrival_time: Instant,
    pub priority: Option<i64>,
    pub kv_computed_len: usize,       // 已计算 KV 的 token 数（用于 chunked prefill）
    pub block_table: Vec<usize>,       // paged attention 块表
    pub preempt_count: u32,            // 被抢占次数
}
```

关键方法：`total_len()`, `remaining_tokens()`, `prefill_len()`, `is_finished()`

### 4.3 `SchedulerConfig` -- 调度器配置

```rust
pub struct SchedulerConfig {
    pub max_batch_size: usize,          // 32
    pub max_batch_wait_ms: u64,         // 5
    pub max_running_requests: usize,    // 256
    pub max_prefill_tokens: usize,      // 8192
    pub max_total_tokens: usize,        // 32768
    pub decode_reservation_cap: usize,  // 4096
    pub new_token_ratio: f32,           // 0.4
    pub min_new_token_ratio: f32,       // 0.1
    pub new_token_ratio_decay: f32,     // 0.002
    pub policy: SchedulePolicy,         // Fcfs
    pub mixed_chunked: bool,            // false
}
```

### 4.4 `SchedulerStep` / `SchedulerOutput`

```
SchedulerStep -- 调度决策
    ├── prefill_request_ids: Vec<String>
    ├── decode_request_ids: Vec<String>
    └── forward_mode: ForwardMode

SchedulerOutput -- 简化版（索引 + mode）
    ├── sequences: Vec<usize>
    └── forward_mode: ForwardMode
```

### 4.5 `Scheduler` -- 连续批处理调度器

```rust
pub struct Scheduler {
    config: SchedulerConfig,
    waiting_queue: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    effective_new_token_ratio: f32,
    tokens_in_use: usize,
    step_count: u64,
}
```

---

## 五、Feature Flags 与条件编译

### prelude-core 的 feature 列表

| Feature | 依赖 | 说明 |
|---------|------|------|
| `default` | `onednn` | 默认启用 OneDNN CPU 加速 |
| `cuda` | `candle-core/cuda`, `candle-nn/cuda`, `candle-transformers/cuda` | CUDA GPU 支持 |
| `flash-attn` | `cuda` + `paged-attn` + `candle-flash-attn` | Flash Attention v2 |
| `flash-attn-v3` | `cuda` + `candle-flash-attn-v3` | Flash Attention v3 (Hopper) |
| `flash-attn-v4` | `cuda` + `prelude-flash-attn-v4` | Flash Attention v4 (自研) |
| `flashinfer` | `cuda` + `prelude-flashinfer` | FlashInfer 后端 |
| `flashinfer-v4` | `flashinfer` + `flash-attn-v4` + `prelude-flashinfer/skip-tvm-ffi` | FlashInfer + FA4 组合 |
| `deepgemm` | `cuda` + `prelude-deepgemm` | DeepGeMM kernel |
| `cutlass-gemm` | `cuda` + `prelude-cutlass-gemm` | CUTLASS GeMM kernel |
| `paged-attn` | `candle-paged-attn` (optional dep) | Paged Attention |
| `onednn` | （无额外依赖） | OneDNN CPU 加速标记 |
| `ggml-quants` | `prelude-ggml-quants` (optional dep) | GGML 量化支持 |
| `hf_tokenizer` | `tokenizers` (optional dep) | HuggingFace tokenizer 支持 |

**条件编译在 config.rs 中的影响**：
- `CacheConfig::paged_block_size` 默认值：`cfg!(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))` 为 true 时用 128，否则 16

---

## 六、完整类型依赖图

```
EngineConfig (config.rs)
├── CacheConfig (config.rs)
│   └── 被 cache/manager.rs 使用
│   └── 通过 global_cache_config() 全局访问
├── SamplingDefaults (config.rs)
│   └── 提供 API 请求缺省参数
├── RuntimeConfig (config.rs)
│   └── 被 engine/device.rs, ops/gpu/kv_cache.rs, models/ 使用
│   └── 通过 global_runtime() 全局访问
└── AdaptiveConfig (config.rs)
    └── 被 scheduler/adaptive.rs 使用

--- 外部 API 请求流 ---

CompletionRequest (types.rs)                ChatCompletionRequest (types.rs)
├── CompletionPrompt                        ├── ChatMessage
│                                           │   └── ChatMessageContent
├── StreamOptions                           ├── StreamOptions
└── validate_public_request()               ├── ResponseFormat (TODO)
    |                                       └── validate_public_request()
    v                                           |
GenerateRequest (types.rs)                      v
├── PromptInput (Text/TokenIds)         GenerateRequest (types.rs)
├── SamplingParams (scheduler/state.rs)     （同左）
├── StopConfig
└── logprobs / prompt_logprobs

--- 引擎处理流 ---

GenerateRequest --> Sequence (scheduler/state.rs)
                    ├── SamplingParams
                    ├── SequenceStatus
                    ├── SeqFinishReason --> FinishReason
                    └── block_table (paged attention)

Scheduler (scheduler/state.rs)
├── SchedulerConfig
│   └── SchedulePolicy
├── Sequence (waiting_queue / running / finished)
├── SchedulerStep --> ForwardMode
└── SchedulerOutput

--- 引擎输出流 ---

StreamEvent (types.rs)
├── Token { text, logprobs: Option<TokenLogprobInfo> }
└── Finished { FinishReason, Usage, DecodeMetrics }

GenerateResult (types.rs)
├── FinishReason (re-export from scheduler)
├── Usage
├── DecodeMetrics
├── token_logprobs: Option<Vec<TokenLogprobInfo>>
└── prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>

--- API 响应格式化 ---

TokenLogprobInfo (types.rs)
├──→ CompletionLogprobs (Completions API)
└──→ ChatCompletionLogprobs (Chat API)
     └── ChatCompletionLogprobContent
         └── ChatCompletionTopLogprob

CompletionResponse                  ChatCompletionResponse
├── CompletionChoice                ├── ChatCompletionChoice
│   ├── CompletionLogprobs          │   ├── ChatMessageOut
│   └── PromptLogprobEntry          │   └── ChatCompletionLogprobs
└── Usage                           └── Usage

--- 分类/嵌入流 ---

ClassificationRequest ──→ ClassificationInputs ──→ ClassifyRequest
                          (Texts/TokenIds)           ├── ClassifyResult
                                                     │   └── ClassificationResult
                                                     └── ClassificationResponse
                                                         └── ClassificationUsage

EmbeddingRequest ──→ ClassificationInputs (复用!) ──→ EmbedRequest
                                                      ├── EmbedResult
                                                      │   └── EmbeddingData
                                                      └── EmbeddingResponse
                                                          ├── EmbeddingObject
                                                          │   └── EmbeddingValue
                                                          └── EmbeddingUsage

--- 常量依赖 ---

DEFAULT_SEED ──→ engine/mod.rs, engine/forward/generate.rs
GGUF_DEFAULT_VOCAB_SIZE ──→ models/qwen3/gguf.rs
GGUF_INTERMEDIATE_SIZE_MULTIPLIER ──→ qwen3_gguf.rs, qwen3_5_gguf.rs
PSEUDO_ENGINE_MAX_TOKENS ──→ engine/pseudo.rs
```

---

## 七、lib.rs Re-export 结构

```rust
pub mod cache;          // KV 缓存管理
pub mod config;         // 配置系统
pub mod constants;      // 全局常量
pub mod engine;         // 推理引擎
pub mod models;         // 模型架构
pub mod ops;            // GPU/CPU 算子
pub mod runtime;        // 运行时（continuous batching 等）
pub mod scheduler;      // 调度器
pub mod types;          // API 类型

// 精选 re-export
pub use cache::deltanet_pool;
pub use cache::prefix_cache;
pub use cache::prefix_index;
pub use config::EngineConfig;
pub use engine::{Engine, TaskOverride};
pub use engine::{EngineError, InferenceEngine, PseudoEngine};
pub use scheduler::scheduled_engine;
pub use scheduler::{ScheduledEngine, SchedulerConfig};
pub use types::*;           // 所有 types.rs 中的类型（包括 re-export 的 FinishReason, SamplingParams）
```

这意味着外部 crate (`prelude-server`) 可以直接 `use prelude_core::{GenerateRequest, FinishReason, EngineConfig, ...}` 而无需指定子模块路径。

---

## 八、关键设计模式总结

1. **环境变量集中解析**: 所有 `PRELUDE_*` 环境变量在 `EngineConfig::from_env()` 一次性读取，避免散落在各处
2. **全局静态访问 + 值传递并行**: 配置既通过参数传递（构造 Engine/CacheManager），也通过 `OnceLock` 提供全局访问（给深层模型代码用）
3. **类型分层**: 外部 API 类型（serde 序列化/反序列化）与内部引擎类型分离，通过 server 层转换
4. **复用策略**: 分类和嵌入共用 `ClassificationInputs` 内部表示
5. **Feature flag 驱动的默认值**: `paged_block_size` 根据 attention 后端 feature 选择不同默认值
6. **渐进式 API 支持**: 通过 `validate_public_request()` + `reject_if_present()` 对未实现功能显式拒绝，留有 TODO 标记
