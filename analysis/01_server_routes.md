# prelude-server crate 深度代码分析

本文档对 `prelude-server` crate 中的每个文件、函数、结构体、枚举、trait 及 impl 块进行详尽分析，涵盖功能描述、参数/返回类型、调用关系和数据流。

---

## 目录

1. [main.rs — 服务入口与引擎构建](#1-mainrs--服务入口与引擎构建)
2. [lib.rs — Router 构建与 Server 状态](#2-librs--router-构建与-server-状态)
3. [routes/mod.rs — 路由模块导出](#3-routesmodrs--路由模块导出)
4. [routes/generation_common.rs — 生成请求共用工具](#4-routesgeneration_commonrs--生成请求共用工具)
5. [routes/chat_completions.rs — Chat Completions 路由](#5-routeschat_completionsrs--chat-completions-路由)
6. [routes/completions.rs — Text Completions 路由](#6-routescompletionsrs--text-completions-路由)
7. [routes/embeddings.rs — Embeddings 路由](#7-routesembeddingsrs--embeddings-路由)
8. [routes/classify.rs — Classification 路由](#8-routesclassifyrs--classification-路由)
9. [routes/health.rs — 健康检查路由](#9-routeshealthrs--健康检查路由)
10. [routes/models.rs — 模型信息路由](#10-routesmodelsrs--模型信息路由)
11. [辅助模块概述](#11-辅助模块概述)
12. [全局调用图](#12-全局调用图)
13. [数据流总览](#13-数据流总览)

---

## 1. main.rs — 服务入口与引擎构建

文件路径: `crates/prelude-server/src/main.rs`

### 1.1 struct `Cli`

```rust
#[derive(Debug, Parser)]
#[command(author, version, about = "Prelude HTTP server")]
struct Cli { ... }
```

**描述**: 使用 `clap::Parser` 定义的命令行参数结构体，是整个服务器的配置入口。

**字段详解**:

| 字段 | 类型 | 默认值 | 作用 |
|---|---|---|---|
| `host` | `String` | `"0.0.0.0"` | 监听地址 |
| `port` | `u16` | `8000` | 监听端口 |
| `model` | `String` | `"Qwen/Qwen3-0.6B"` | HuggingFace 模型 repo ID |
| `model_path` | `Option<String>` | `None` | 本地模型路径（目录或 .gguf 文件） |
| `pseudo` | `bool` | `false` | 是否使用 mock 引擎 |
| `pseudo_latency_ms` | `u64` | `25` | mock 引擎每次延迟毫秒 |
| `max_batch_size` | `usize` | `32` | 动态 batching 最大请求数 |
| `max_batch_wait_ms` | `u64` | `5` | 动态 batching 最长等待时间 |
| `max_running_requests` | `usize` | `8` | 调度器最大并发请求数 |
| `max_prefill_tokens` | `usize` | `4096` | 单次调度步骤最大 prefill token 数 |
| `max_total_tokens` | `usize` | `32768` | 所有运行请求的总 token 上限 |
| `decode_reservation_cap` | `usize` | `4096` | 每请求未来 decode token 预留上限 |
| `task` | `CliTaskOverride` | `Auto` | 强制模型任务类型 |
| `api_key` | `Vec<String>` | `[]` | API 认证 key（可重复） |
| `cors_allow_origin` | `Vec<String>` | `[]` | 允许的 CORS origin |
| `dtype` | `Option<String>` | `None` | 覆盖数据类型（f32/bf16） |
| `gpu_memory_utilization` | `f32` | `0.4` | GPU 显存中用于 KV cache 的比例 |
| `cuda_graph` | `bool` | `true` | CUDA graph 加速 decode（`--cuda-graph false` 关闭） |
| `device` | `String` | `auto` | 设备选择：auto, cpu, cuda, cuda:N |

**被调用者**: `main()` 通过 `Cli::parse()` 解析。

---

### 1.2 enum `CliTaskOverride`

```rust
#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliTaskOverride { Auto, Classify, Embedding, Generation }
```

**描述**: CLI 层面的任务类型枚举，用于 `--task` 参数，映射到引擎核心的 `TaskOverride`。

**`impl From<CliTaskOverride> for TaskOverride`**: 一对一映射转换，`Embedding` -> `Embed`，其余名称一致。

---

### 1.3 `async fn main()`

**描述**: 程序入口，负责初始化日志、解析 CLI、构建引擎和 chat template、合并 API key、构建 router、绑定端口并启动 HTTP 服务。

**返回类型**: `anyhow::Result<()>`

**执行流程**:
1. 初始化 `tracing_subscriber`，默认过滤 `prelude_server=info,prelude_core=info,tower_http=info`，可通过 `RUST_LOG` 环境变量覆盖
2. `Cli::parse()` 解析命令行参数
3. `build_engine(&cli)` 构建推理引擎 -> `Arc<dyn InferenceEngine>`
4. `build_chat_template(&cli)` 加载 chat template -> `Option<Arc<ModelChatTemplate>>`
5. 合并 `--api-key` 参数和 `PRELUDE_API_KEY` 环境变量中的 API key
6. `prelude_server::build_router_with_options(...)` 构建 axum Router
7. 绑定 TCP 监听并通过 `axum::serve` 启动服务

**调用关系**:
- 调用 -> `build_engine`, `build_chat_template`, `build_router_with_options`
- 被调用 <- tokio runtime 入口

---

### 1.4 `fn build_engine(cli: &Cli) -> anyhow::Result<Arc<dyn InferenceEngine>>`

**描述**: 根据 CLI 配置构建推理引擎。支持三种模式：
1. **Pseudo 模式**: `--pseudo` 时创建 `PseudoEngine`（mock）
2. **直连模式**: `PRELUDE_NO_SCHEDULER=1` 时直接使用 `Engine`
3. **调度模式**（默认）: 用 `ScheduledEngine` 包装 `Engine`

**参数**: `cli: &Cli` — 命令行配置

**返回**: `Arc<dyn InferenceEngine>` — 多态推理引擎

**执行流程**:
1. 将 `cli.task` 转为 `TaskOverride`
2. 若 `cli.pseudo` 为 true，创建 `PseudoEngine` 并直接返回
3. 从环境变量加载 `EngineConfig`，用 CLI 参数覆盖 `device`、`dtype`、`gpu_memory_utilization`、`cuda_graph`
4. 根据 `model_path` 是否存在，选择 `Engine::from_local_path_with_task` 或 `Engine::from_hf_hub_with_task`
5. 检查 `PRELUDE_NO_SCHEDULER` 环境变量，若为 `"1"` 则直接返回 base engine
6. 构建 `SchedulerConfig` 并创建 `ScheduledEngine`

**调用关系**:
- 调用 -> `PseudoEngine::new`, `EngineConfig::from_env`, `Engine::from_local_path_with_task`, `Engine::from_hf_hub_with_task`, `ScheduledEngine::new`
- 被调用 <- `main()`

---

### 1.5 `fn load_chat_template_with_gguf_fallback(model: &str) -> anyhow::Result<Option<ModelChatTemplate>>`

**描述**: 从 HuggingFace Hub 加载 chat template，并对 GGUF 模型仓库进行回退处理。若原始 repo 没有 `tokenizer_config.json`，尝试去掉 `-GGUF`/`-gguf` 后缀再次查找 base model 的 template。

**参数**: `model: &str` — 模型仓库 ID

**返回**: `anyhow::Result<Option<ModelChatTemplate>>`

**执行流程**:
1. 调用 `ModelChatTemplate::from_hf_hub(model)` 尝试加载
2. 若失败或返回 None，检查 model 是否以 `-GGUF`/`-gguf` 结尾
3. 若是，去除后缀后再次调用 `ModelChatTemplate::from_hf_hub(base)`

**调用关系**:
- 调用 -> `ModelChatTemplate::from_hf_hub`
- 被调用 <- `build_chat_template`

---

### 1.6 `fn build_chat_template(cli: &Cli) -> anyhow::Result<Option<ModelChatTemplate>>`

**描述**: 构建 chat template 的统一入口。pseudo 模式返回 None；本地路径优先从本地加载，失败后回退 HF Hub；否则直接走 HF Hub（含 GGUF 回退）。

**参数**: `cli: &Cli`

**返回**: `anyhow::Result<Option<ModelChatTemplate>>`

**执行流程**:
1. `cli.pseudo` 时返回 `None`
2. 有 `model_path` -> `ModelChatTemplate::from_local_path` -> 若 None 则 `load_chat_template_with_gguf_fallback`
3. 无 `model_path` -> `load_chat_template_with_gguf_fallback`

**调用关系**:
- 调用 -> `ModelChatTemplate::from_local_path`, `load_chat_template_with_gguf_fallback`
- 被调用 <- `main()`

---

## 2. lib.rs — Router 构建与 Server 状态

文件路径: `crates/prelude-server/src/lib.rs`

### 2.1 struct `Server`

```rust
#[derive(Clone)]
pub struct Server {
    pub engine: Arc<dyn InferenceEngine>,
    pub chat_template: Option<Arc<ModelChatTemplate>>,
    pub started_at: Instant,
}
```

**描述**: axum 应用的共享状态，作为 `State<Server>` 注入到每个路由 handler 中。包含推理引擎、可选的 chat template 和服务启动时间戳。

**数据流**: 由 `build_router_with_options` 创建，通过 `.with_state(server)` 传入所有路由。

---

### 2.2 struct `RouterOptions`

```rust
#[derive(Clone, Debug, Default)]
pub struct RouterOptions {
    pub cors_allowed_origins: Vec<String>,
}
```

**描述**: Router 构建的可选配置，目前仅包含 CORS 允许的 origin 列表。

---

### 2.3 `pub fn build_router(...) -> Router`

**参数**:
- `engine: Arc<dyn InferenceEngine>` — 推理引擎
- `chat_template: Option<Arc<ModelChatTemplate>>` — chat template
- `api_keys: Vec<String>` — API key 列表

**返回**: `Router`

**描述**: 简化版 router 构建函数，使用默认 `RouterOptions`。内部调用 `build_router_with_options` 并 unwrap。

**调用关系**:
- 调用 -> `build_router_with_options`
- 被调用 <- 外部测试/集成代码

---

### 2.4 `pub fn build_router_with_options(...) -> Result<Router>`

**参数**:
- `engine: Arc<dyn InferenceEngine>`
- `chat_template: Option<Arc<ModelChatTemplate>>`
- `api_keys: Vec<String>`
- `options: RouterOptions`

**返回**: `Result<Router>`

**描述**: 构建完整的 axum Router，注册所有路由和中间件。

**路由注册**:

| 路径 | 方法 | Handler |
|---|---|---|
| `/health` | GET | `health` |
| `/v1/models` | GET | `list_models` |
| `/v1/models/{model}` | GET | `get_model` |
| `/v1/completions` | POST | `completions` |
| `/v1/chat/completions` | POST | `chat_completions` |
| `/v1/embeddings` | POST | `embeddings` |
| `/v1/classify` | POST | `classify` |

**中间件层（从内到外）**:
1. `auth::auth_middleware` — API key 认证（通过 `from_fn_with_state` 注入 `ApiKeys`）
2. `TraceLayer::new_for_http()` — HTTP 请求追踪
3. CORS layer（如果配置了 allowed origins）

**调用关系**:
- 调用 -> `auth::ApiKeys` 构造, `apply_cors`
- 被调用 <- `build_router`, `main()`

---

### 2.5 `fn apply_cors(router: Router, options: &RouterOptions) -> Result<Router>`

**描述**: 条件性地为 router 添加 CORS layer。若 `cors_allowed_origins` 为空则不添加。

**参数**:
- `router: Router` — 已构建的 router
- `options: &RouterOptions` — 配置

**返回**: `Result<Router>`

**执行流程**:
1. 若 origin 列表为空，直接返回 router
2. 将每个 origin 字符串解析为 `HeaderValue`
3. 构建 `CorsLayer`，允许 GET/POST 方法，mirror 请求头

**调用关系**:
- 被调用 <- `build_router_with_options`

---

## 3. routes/mod.rs — 路由模块导出

文件路径: `crates/prelude-server/src/routes/mod.rs`

**描述**: 纯粹的模块声明和 re-export 文件。声明了 6 个子模块并导出 7 个公开 handler 函数。

**导出项**:
- `chat_completions::chat_completions`
- `classify::classify`
- `completions::completions`
- `embeddings::embeddings`
- `health::health`
- `models::get_model`
- `models::list_models`

---

## 4. routes/generation_common.rs — 生成请求共用工具

文件路径: `crates/prelude-server/src/routes/generation_common.rs`

该模块为 `chat_completions` 和 `completions` 路由提供共享的常量、类型和辅助函数。

### 4.1 常量 `DEFAULT_MAX_NEW_TOKENS`

```rust
pub(crate) const DEFAULT_MAX_NEW_TOKENS: u32 = 4096;
```

**描述**: 当用户请求未指定 `max_tokens` 时使用的默认最大新生成 token 数。

**被使用**: `parse_chat_request`, `parse_completion_requests`

---

### 4.2 struct `ResponseMeta`

```rust
#[derive(Clone)]
pub(crate) struct ResponseMeta {
    pub id: String,
    pub created: i64,
    pub model: String,
}
```

**描述**: 封装 OpenAI 兼容响应所需的元信息：唯一 ID、创建时间戳和模型名。

#### `ResponseMeta::new(prefix: &str, model: impl Into<String>) -> Self`

**描述**: 创建 ResponseMeta 实例。ID 格式为 `{prefix}-{uuid}`，`created` 为当前 UTC 时间戳。

**参数**:
- `prefix` — ID 前缀（如 `"chatcmpl"`, `"cmpl"`）
- `model` — 模型名

**调用关系**:
- 被调用 <- `chat_stream`, `chat_batch`, `completions_stream`, `completions_batch`

---

### 4.3 `fn build_generate_request(...) -> GenerateRequest`

**描述**: 将 HTTP 请求参数转换为引擎层面的 `GenerateRequest`。是 chat 和 completion 路由共用的核心转换函数。

**参数**:

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | `String` | 模型名 |
| `input` | `PromptInput` | 输入文本（`PromptInput::Text`） |
| `max_new_tokens` | `u32` | 最大生成 token 数 |
| `temperature` | `Option<f32>` | 温度（默认 0.7） |
| `top_p` | `Option<f32>` | top-p（默认 1.0） |
| `stop` | `Option<Vec<String>>` | 停止字符串 |
| `seed` | `Option<u64>` | 随机种子 |
| `logprobs` | `Option<u32>` | 返回 top-k logprobs |
| `prompt_logprobs` | `Option<u32>` | 返回 prompt 级别 logprobs |

**返回**: `GenerateRequest`，其 `request_id` 格式为 `req-{uuid}`。

**调用关系**:
- 被调用 <- `parse_chat_request`, `parse_completion_requests`

---

### 4.4 `fn sse_json_event<T: Serialize>(value: &T) -> Result<Event, Infallible>`

**描述**: 将任意可序列化值转为 SSE `Event`。序列化结果作为 event 的 `data` 字段。

**调用关系**:
- 被调用 <- `chat_stream`, `completions_stream` 中的 mapper 闭包

---

### 4.5 `fn sse_done_event() -> Result<Event, Infallible>`

**描述**: 创建流式传输结束标记事件，data 为 `[DONE]`，符合 OpenAI SSE 协议。

**调用关系**:
- 被调用 <- `chat_stream`, `completions_stream` 中的 Finished/Error 分支

---

## 5. routes/chat_completions.rs — Chat Completions 路由

文件路径: `crates/prelude-server/src/routes/chat_completions.rs`

### 5.1 `pub async fn chat_completions(State, Json) -> Result<Response, ApiError>`

**描述**: `/v1/chat/completions` 的 axum handler。接收 OpenAI 格式的 chat completion 请求，校验参数，解析 chat template 渲染 prompt，然后分发到流式或批量处理。

**参数**:
- `State(server): State<Server>` — 服务器共享状态
- `Json(request): Json<ChatCompletionRequest>` — 请求体

**返回**: `Result<Response, ApiError>`

**执行流程**:
1. 调用 `request.validate_public_request()` 校验请求参数合法性
2. 读取 `stream` 和 `stream_options.include_usage` 标志
3. 调用 `parse_chat_request` 将 chat 消息通过 chat template 渲染为 prompt，构建 `GenerateRequest`
4. 记录请求日志
5. 若 `is_streaming` -> `chat_stream()`；否则 -> `chat_batch()`

**调用关系**:
- 调用 -> `parse_chat_request`, `chat_stream`, `chat_batch`
- 被调用 <- axum router（`POST /v1/chat/completions`）

---

### 5.2 `fn chat_stream(...)` (private)

`fn chat_stream(engine, request, include_usage) -> Result<Response, ApiError>`

**可见性**: 模块内私有函数，仅被 `chat_completions` 调用。

**描述**: 流式 chat completion 处理。创建 `ResponseMeta`，通过 `stream_sse` 将引擎的 `StreamEvent` 映射为 SSE 事件序列。

**参数**:
- `engine: Arc<dyn InferenceEngine>`
- `request: GenerateRequest`
- `include_usage: bool` — 是否在最后一个 chunk 包含 usage 信息

**返回**: `Result<Response, ApiError>`

**StreamEvent 映射逻辑**:

| StreamEvent | 产生的 SSE 事件 |
|---|---|
| `Started` | 1 个 chunk，delta 包含空 content 和 role="assistant" |
| `Token { text, logprobs }` | 1 个 chunk，delta 包含 text 内容和可选 logprobs |
| `Finished { finish_reason, usage, metrics }` | 1-3 个事件：finish chunk + 可选 usage chunk + `[DONE]` |
| `Error { message }` | 记录错误日志 + `[DONE]` |

**关键细节**:
- `Started` 事件发送一个空 content 的 delta，让客户端知道 assistant 已开始回复
- `Finished` 事件中，`include_usage=true` 时额外发送一个空 choices 但包含 usage 的 chunk（符合 OpenAI 规范）

**调用关系**:
- 调用 -> `ResponseMeta::new`, `stream_sse`, `sse_json_event`, `sse_done_event`, `to_chat_logprob_content`, `log_generation_metrics`
- 被调用 <- `chat_completions`

---

### 5.3 `async fn chat_batch(...)` (private)

`async fn chat_batch(engine, request) -> Result<Response, ApiError>`

**可见性**: 模块内私有函数，仅被 `chat_completions` 调用。

**描述**: 非流式 chat completion 处理。调用引擎的 `generate` 方法，等待完整结果后一次性返回。

**参数**:
- `engine: Arc<dyn InferenceEngine>`
- `request: GenerateRequest`

**返回**: `Result<Response, ApiError>`

**执行流程**:
1. `engine.generate(request).await` 获取完整结果
2. 将 `FinishReason` 转为 OpenAI 字符串格式
3. 记录性能指标日志
4. 转换 logprobs（如有）
5. 构建并返回 `ChatCompletionResponse` JSON

**关键细节**:
- `object` 字段为 `"chat.completion"`（非流式）vs `"chat.completion.chunk"`（流式）
- 非流式响应中 `message` 字段有值，`delta` 为 None

**调用关系**:
- 调用 -> `engine.generate`, `log_generation_metrics`, `to_chat_logprobs`, `ResponseMeta::new`
- 被调用 <- `chat_completions`

---

### 5.4 `fn parse_chat_request(...)` (private)

`fn parse_chat_request(request, chat_template) -> Result<GenerateRequest, ApiError>`

**可见性**: 模块内私有函数，仅被 `chat_completions` 调用。

**描述**: 将 `ChatCompletionRequest` 转换为引擎的 `GenerateRequest`。核心步骤是使用 chat template（Jinja2）将消息数组渲染为单一 prompt 字符串。

**参数**:
- `request: &ChatCompletionRequest` — 原始请求
- `chat_template: Option<&ModelChatTemplate>` — Jinja2 模板

**返回**: `Result<GenerateRequest, ApiError>`

**执行流程**:
1. 校验 messages 非空
2. 校验 chat_template 存在（不存在则返回 400 提示使用 `/v1/completions`）
3. 调用 `chat_template.render(&request.messages)` 渲染 prompt
4. 解析 logprobs 配置：`logprobs=true` 时取 `top_logprobs`（默认 0）
5. `max_tokens` 优先级: `max_completion_tokens` > `max_tokens` > `DEFAULT_MAX_NEW_TOKENS`
6. 调用 `build_generate_request` 构建引擎请求

**调用关系**:
- 调用 -> `chat_template.render`, `build_generate_request`
- 被调用 <- `chat_completions`

---

## 6. routes/completions.rs — Text Completions 路由

文件路径: `crates/prelude-server/src/routes/completions.rs`

### 6.1 `pub async fn completions(State, Json) -> Result<Response, ApiError>`

**描述**: `/v1/completions` 的 axum handler。支持单个和批量 prompt 输入，以及流式/非流式响应模式。

**参数**:
- `State(server): State<Server>`
- `Json(request): Json<CompletionRequest>`

**返回**: `Result<Response, ApiError>`

**执行流程**:
1. 调用 `request.validate_public_request()` 校验参数
2. 读取 `stream` 和 `include_usage` 标志
3. 调用 `parse_completion_requests` 将请求拆分为一个或多个 `GenerateRequest`
4. 若流式且请求 > 1 个 prompt，返回 400 错误（流式不支持批量）
5. 流式 -> `completions_stream()`；非流式 -> `completions_batch()`

**与 chat_completions 的差异**:
- 不需要 chat template，直接使用原始 prompt 文本
- 支持批量 prompt（`CompletionPrompt::Batch`）
- 流式模式禁止批量 prompt

**调用关系**:
- 调用 -> `parse_completion_requests`, `completions_stream`, `completions_batch`
- 被调用 <- axum router（`POST /v1/completions`）

---

### 6.2 `fn completions_stream(engine, request, include_usage) -> Result<Response, ApiError>`

**描述**: 流式 text completion 处理。

**参数**:
- `engine: Arc<dyn InferenceEngine>`
- `request: GenerateRequest`
- `include_usage: bool`

**返回**: `Result<Response, ApiError>`

**StreamEvent 映射逻辑**:

| StreamEvent | 产生的 SSE 事件 |
|---|---|
| `Started` | 空 vec（不发送事件） |
| `Token { text, logprobs }` | 1 个 chunk，包含 text 和可选 logprobs |
| `Finished` | 1-2 个 finish chunk + `[DONE]` |
| `Error` | 记录错误 + `[DONE]` |

**与 chat_stream 的差异**:
- `Started` 不发送初始 chunk（completion 无需 role 前缀）
- `object` 为 `"text_completion"` 而非 `"chat.completion.chunk"`
- 使用 `CompletionResponse` / `CompletionChoice` 而非 Chat 版本
- logprobs 使用 `to_completion_logprobs`（扁平格式）而非 chat 的结构化格式

**调用关系**:
- 调用 -> `ResponseMeta::new`, `stream_sse`, `sse_json_event`, `sse_done_event`, `to_completion_logprobs`, `log_generation_metrics`
- 被调用 <- `completions`

---

### 6.3 `async fn completions_batch(engine, requests) -> Result<Response, ApiError>`

**描述**: 非流式批量 text completion 处理。支持多个 prompt 的并行推理。

**参数**:
- `engine: Arc<dyn InferenceEngine>`
- `requests: Vec<GenerateRequest>`

**返回**: `Result<Response, ApiError>`

**执行流程**:
1. 记录开始时间
2. `engine.generate_batch(requests).await` 批量推理
3. 计算总耗时
4. `aggregate_usage` 汇总所有结果的 token 用量
5. 将每个结果转为 `CompletionChoice`，包含 logprobs 和 prompt_logprobs
6. 构建并返回 `CompletionResponse` JSON

**与 chat_batch 的差异**:
- 使用 `generate_batch` 而非 `generate`（支持多 prompt）
- 每个 choice 的 `index` 从 0 递增
- 支持 `prompt_logprobs` 和 `prompt_token_ids` 字段
- 使用 `aggregate_usage` 汇总所有请求的 usage

**调用关系**:
- 调用 -> `engine.generate_batch`, `aggregate_usage`, `to_completion_logprobs`, `to_prompt_logprobs_response`
- 被调用 <- `completions`

---

### 6.4 `fn parse_completion_requests(request) -> Result<Vec<GenerateRequest>, ApiError>`

**描述**: 将 `CompletionRequest` 解析为一个或多个 `GenerateRequest`。处理 `CompletionPrompt` 的 Single/Batch 两种变体。

**参数**: `request: &CompletionRequest`

**返回**: `Result<Vec<GenerateRequest>, ApiError>`

**执行流程**:
1. 将 `CompletionPrompt` 展开为 `Vec<&str>`
2. 校验 prompt 列表非空
3. 校验每个 prompt 非空白
4. 为每个 prompt 调用 `build_generate_request` 创建引擎请求

**调用关系**:
- 调用 -> `build_generate_request`
- 被调用 <- `completions`

---

## 7. routes/embeddings.rs — Embeddings 路由

文件路径: `crates/prelude-server/src/routes/embeddings.rs`

### 7.1 `fn encode_embedding_base64(embedding: &[f32]) -> String`

**描述**: 将 f32 向量编码为 base64 字符串。每个 f32 转为 4 字节 little-endian，整个 buffer 做 base64 编码。这是 OpenAI 兼容的嵌入向量二进制编码格式。

**参数**: `embedding: &[f32]` — 嵌入向量

**返回**: `String` — base64 编码的字符串

**实现细节**: 使用 `unsafe` 将 `&[f32]` 重新解释为 `&[u8]`（零拷贝），然后用 `base64::STANDARD` 编码。

**调用关系**:
- 被调用 <- `embeddings`（当 `encoding_format="base64"` 时）

---

### 7.2 `pub async fn embeddings(State, Json) -> Result<Json<EmbeddingResponse>, ApiError>`

**描述**: `/v1/embeddings` 的 axum handler。接收文本输入，调用引擎的 embed 方法获取嵌入向量，支持 float 和 base64 两种返回格式。

**参数**:
- `State(server): State<Server>`
- `Json(request): Json<EmbeddingRequest>`

**返回**: `Result<Json<EmbeddingResponse>, ApiError>`

**执行流程**:
1. 调用 `request.validate_public_request()` 校验参数
2. `request.get_inputs()` 获取输入文本列表
3. 检查 `encoding_format` 是否为 `"base64"`
4. 构建 `EmbedRequest`（包含 `request_id: "embed-{uuid}"`）
5. 记录请求日志
6. `server.engine.embed(embed_request).await` 执行推理
7. 记录完成日志（模型、数量、维度、token 数）
8. 将结果转为 `EmbeddingResponse`，根据 `encoding_format` 选择 `EmbeddingValue::Float` 或 `EmbeddingValue::Base64`

**调用关系**:
- 调用 -> `request.validate_public_request`, `request.get_inputs`, `engine.embed`, `encode_embedding_base64`
- 被调用 <- axum router（`POST /v1/embeddings`）

---

## 8. routes/classify.rs — Classification 路由

文件路径: `crates/prelude-server/src/routes/classify.rs`

### 8.1 `pub async fn classify(State, Json) -> Result<Json<ClassificationResponse>, ApiError>`

**描述**: `/v1/classify` 的 axum handler。接收文本输入，调用引擎的 classify 方法进行分类推理。这是 Prelude 的扩展 API（非 OpenAI 标准）。

**参数**:
- `State(server): State<Server>`
- `Json(request): Json<ClassificationRequest>`

**返回**: `Result<Json<ClassificationResponse>, ApiError>`

**执行流程**:
1. `request.get_inputs()` 获取输入并校验
2. 构建 `ClassifyRequest`（`request_id: "classify-{uuid}"`）
3. 记录请求日志
4. `server.engine.classify(classify_request).await` 执行分类
5. 记录完成日志
6. 构建 `ClassificationResponse`，包含:
   - `id`: `"classify-{uuid}"`（注意这里生成了新的 uuid，与 request_id 不同）
   - `object`: `"list"`
   - `created`: 当前时间戳
   - `usage`: prompt_tokens 和 total_tokens 相同，completion_tokens = 0

**调用关系**:
- 调用 -> `request.get_inputs`, `engine.classify`
- 被调用 <- axum router（`POST /v1/classify`）

---

## 9. routes/health.rs — 健康检查路由

文件路径: `crates/prelude-server/src/routes/health.rs`

### 9.1 `pub async fn health(State) -> Result<Json<HealthResponse>, ApiError>`

**描述**: `/health` 的 GET handler。返回服务状态、模型名和运行时间。

**参数**: `State(server): State<Server>`

**返回**: `Result<Json<HealthResponse>, ApiError>`

**执行流程**:
1. `server.engine.model_info().await` 获取模型信息
2. 构建 `HealthResponse`：
   - `status`: 固定为 `"ready"`
   - `model`: 模型 ID
   - `uptime_s`: 从 `server.started_at` 到当前的秒数（f64）

**调用关系**:
- 调用 -> `engine.model_info`
- 被调用 <- axum router（`GET /health`）

**注意**: 该路由被 `auth_middleware` 豁免认证检查。

---

## 10. routes/models.rs — 模型信息路由

文件路径: `crates/prelude-server/src/routes/models.rs`

### 10.1 `pub async fn list_models(State) -> Result<Json<ModelListResponse>, ApiError>`

**描述**: `/v1/models` 的 GET handler。列出引擎加载的所有模型。

**参数**: `State(server): State<Server>`

**返回**: `Result<Json<ModelListResponse>, ApiError>`

**执行流程**:
1. `server.engine.list_models().await` 获取模型列表
2. 将每个 model 转为 `ModelCard`（`object: "model"`）
3. 包装为 `ModelListResponse`（`object: "list"`）

**调用关系**:
- 调用 -> `engine.list_models`
- 被调用 <- axum router（`GET /v1/models`）

---

### 10.2 `pub async fn get_model(State, Path) -> Result<Json<ModelCard>, ApiError>`

**描述**: `/v1/models/{model}` 的 GET handler。根据 model ID 查找并返回单个模型信息。

**参数**:
- `State(server): State<Server>`
- `Path(model_id): Path<String>` — URL 路径中的模型 ID

**返回**: `Result<Json<ModelCard>, ApiError>`

**执行流程**:
1. `server.engine.list_models().await` 获取所有模型
2. 在列表中查找 `id == model_id` 的模型
3. 找到则返回 `ModelCard`；未找到则返回 404 错误

**调用关系**:
- 调用 -> `engine.list_models`
- 被调用 <- axum router（`GET /v1/models/{model}`）

---

## 11. 辅助模块概述

以下模块虽不在分析范围内的路由文件中，但被路由 handler 大量引用，此处简要说明。

### 11.1 auth.rs — API 认证中间件

- **struct `ApiKeys(Arc<Vec<String>>)`**: 包装 API key 列表
- **`auth_middleware`**: axum 中间件，检查 `Authorization: Bearer <key>` 头。`/health` 和 `/metrics` 路径豁免。无配置 key 时认证禁用
- **`unauthorized(message)`**: 返回 401 JSON 错误响应

### 11.2 error.rs — 错误类型

- **struct `ApiError`**: 包含 HTTP status code、错误消息和错误类型
- **`impl From<EngineError> for ApiError`**: 将引擎错误映射为 HTTP 错误：
  - `InvalidRequest` -> 400
  - `Unavailable` -> 503
  - `Internal` -> 500
- **`impl IntoResponse for ApiError`**: 输出 `{"error": {"message": "...", "type": "..."}}` 格式

### 11.3 sse.rs — SSE 流式传输

- **`stream_sse(engine, request, mapper)`**: 核心 SSE 流式传输函数。创建 mpsc channel，spawn 后台任务调用 `engine.generate_stream`，将 `StreamEvent` 通过用户提供的 `mapper` 转换为 SSE Event 流
- **struct `CancelOnDrop<S>`**: Stream wrapper，在 Drop 时调用 `engine.cancel(&rid)` 取消请求，防止客户端断开后继续消耗 GPU/CPU

### 11.4 logprobs.rs — Logprobs 转换

- **`to_completion_logprobs`**: 引擎 logprobs -> OpenAI `/v1/completions` 扁平格式（tokens/token_logprobs/text_offset/top_logprobs）
- **`to_chat_logprob_content`**: 单个 token 的 logprob -> chat 结构化格式
- **`to_chat_logprobs`**: 批量 token logprobs -> `ChatCompletionLogprobs`
- **`to_prompt_logprobs_response`**: prompt token logprobs -> vLLM 兼容格式（第一个位置为 None）

### 11.5 utils.rs — 工具函数

- **`log_generation_metrics`**: 记录生成指标日志（prompt/completion tokens、TTFT、decode 速度等）
- **`aggregate_usage`**: 汇总多个 `GenerateResult` 的 token 用量（使用 saturating_add 防溢出）

### 11.6 chat_template.rs — Chat Template 渲染

- **struct `ModelChatTemplate`**: 持有 Jinja2 模板字符串和 special tokens（bos/eos/pad/unk）
- **`from_local_path`**: 从本地目录加载（优先 `chat_template.jinja`，回退 `tokenizer_config.json`）
- **`from_hf_hub`**: 从 HuggingFace Hub 下载加载
- **`render`**: 使用 minijinja 渲染消息数组为 prompt 字符串
- 支持 `TokenizerConfig` 中的 Single/Named chat template 和 Plain/AddedToken 特殊 token 格式

---

## 12. 全局调用图

```
main()
  |-> build_engine()
  |     |-> PseudoEngine::new()            [pseudo 模式]
  |     |-> EngineConfig::from_env()
  |     |-> Engine::from_local_path_with_task()  [有 model_path]
  |     |-> Engine::from_hf_hub_with_task()      [无 model_path]
  |     |-> ScheduledEngine::new()               [默认模式]
  |
  |-> build_chat_template()
  |     |-> ModelChatTemplate::from_local_path()
  |     |-> load_chat_template_with_gguf_fallback()
  |           |-> ModelChatTemplate::from_hf_hub()
  |
  |-> build_router_with_options()
        |-> auth::ApiKeys()
        |-> apply_cors()
        |
        |=== 注册路由 ===|
        |
        |-> health()
        |     |-> engine.model_info()
        |
        |-> list_models()
        |     |-> engine.list_models()
        |
        |-> get_model()
        |     |-> engine.list_models()
        |
        |-> chat_completions()
        |     |-> parse_chat_request()
        |     |     |-> chat_template.render()
        |     |     |-> build_generate_request()
        |     |
        |     |-> chat_stream()              [stream=true]
        |     |     |-> ResponseMeta::new()
        |     |     |-> stream_sse()
        |     |           |-> engine.generate_stream()  [spawned task]
        |     |           |-> mapper 闭包:
        |     |                 |-> sse_json_event()
        |     |                 |-> sse_done_event()
        |     |                 |-> to_chat_logprob_content()
        |     |                 |-> log_generation_metrics()
        |     |
        |     |-> chat_batch()               [stream=false]
        |           |-> engine.generate()
        |           |-> log_generation_metrics()
        |           |-> to_chat_logprobs()
        |           |-> ResponseMeta::new()
        |
        |-> completions()
        |     |-> parse_completion_requests()
        |     |     |-> build_generate_request()  [per prompt]
        |     |
        |     |-> completions_stream()       [stream=true, 仅单 prompt]
        |     |     |-> ResponseMeta::new()
        |     |     |-> stream_sse()
        |     |           |-> engine.generate_stream()
        |     |           |-> mapper 闭包:
        |     |                 |-> sse_json_event()
        |     |                 |-> sse_done_event()
        |     |                 |-> to_completion_logprobs()
        |     |                 |-> log_generation_metrics()
        |     |
        |     |-> completions_batch()        [stream=false]
        |           |-> engine.generate_batch()
        |           |-> aggregate_usage()
        |           |-> to_completion_logprobs()
        |           |-> to_prompt_logprobs_response()
        |
        |-> embeddings()
        |     |-> request.get_inputs()
        |     |-> engine.embed()
        |     |-> encode_embedding_base64()  [base64 模式]
        |
        |-> classify()
              |-> request.get_inputs()
              |-> engine.classify()
```

---

## 13. 数据流总览

### 13.1 请求处理总体流程

```
HTTP 请求
  -> CorsLayer（可选）
  -> TraceLayer（HTTP 追踪日志）
  -> auth_middleware（API key 认证）
  -> axum 路由分发
  -> handler 函数
  -> 请求校验 (validate_public_request)
  -> 请求解析/转换 (parse_xxx)
  -> 引擎调用 (engine.generate/embed/classify)
  -> 响应构建 (OpenAI 兼容格式)
  -> HTTP 响应
```

### 13.2 Chat Completions 数据转换链

```
ChatCompletionRequest
  -> ChatCompletionRequest.messages: Vec<ChatMessage>
  -> ModelChatTemplate.render() [Jinja2]
  -> prompt: String
  -> build_generate_request()
  -> GenerateRequest { input: PromptInput::Text(prompt), ... }
  -> engine.generate() / engine.generate_stream()
  -> GenerateResult / StreamEvent
  -> ChatCompletionResponse { choices: [ChatCompletionChoice], usage, ... }
```

### 13.3 Text Completions 数据转换链

```
CompletionRequest
  -> CompletionPrompt::Single(String) | CompletionPrompt::Batch(Vec<String>)
  -> parse_completion_requests() [每个 prompt 一个 GenerateRequest]
  -> Vec<GenerateRequest>
  -> engine.generate_batch() / engine.generate_stream()
  -> Vec<GenerateResult> / StreamEvent
  -> CompletionResponse { choices: [CompletionChoice], usage, ... }
```

### 13.4 Embeddings 数据转换链

```
EmbeddingRequest
  -> request.get_inputs() -> Vec<String>
  -> EmbedRequest { inputs, model, request_id }
  -> engine.embed()
  -> EmbedResult { data: Vec<EmbeddingData>, dimensions, prompt_tokens }
  -> EmbeddingResponse { data: [EmbeddingObject { embedding: Float|Base64 }], usage }
```

### 13.5 Classification 数据转换链

```
ClassificationRequest
  -> request.get_inputs() -> Vec<(String, String)>
  -> ClassifyRequest { inputs, model, request_id }
  -> engine.classify()
  -> ClassifyResult { results, model, prompt_tokens }
  -> ClassificationResponse { data: results, usage }
```

### 13.6 流式传输（SSE）数据流

```
engine.generate_stream(request, tx)  [后台 tokio task]
  -> tx.send(StreamEvent)            [mpsc unbounded channel]
  -> rx.recv()                       [unfold 转 Stream]
  -> mapper(StreamEvent)             [用户闭包，映射为 Vec<Event>]
  -> flat_map                        [展平为 Event 流]
  -> CancelOnDrop 包装               [客户端断开时取消请求]
  -> Sse::new(stream)                [axum SSE 响应]
  -> HTTP chunked transfer           [text/event-stream]
```

### 13.7 InferenceEngine trait 方法使用映射

| 路由 | 引擎方法 |
|---|---|
| `GET /health` | `engine.model_info()` |
| `GET /v1/models` | `engine.list_models()` |
| `GET /v1/models/{model}` | `engine.list_models()` |
| `POST /v1/chat/completions` (非流式) | `engine.generate()` |
| `POST /v1/chat/completions` (流式) | `engine.generate_stream()` |
| `POST /v1/completions` (非流式) | `engine.generate_batch()` |
| `POST /v1/completions` (流式) | `engine.generate_stream()` |
| `POST /v1/embeddings` | `engine.embed()` |
| `POST /v1/classify` | `engine.classify()` |
| SSE 客户端断开 | `engine.cancel()` |
