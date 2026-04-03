# vLLM 设计分析 & Prelude 可借鉴 / 应避免的

## 1. 值得学习的设计

### 1.1 Quantization Registry — 量化方法可插拔注入

vLLM:
```python
# 24+ 种量化方法，通过 registry 注册
QUANTIZATION_METHODS = {
    "fp8": Fp8Config,
    "gptq": GPTQConfig,
    "awq": AWQConfig,
    ...
}

# Linear 层不感知具体量化方法
class LinearBase(nn.Module):
    def __init__(self, ..., quant_config):
        self.quant_method = quant_config.get_quant_method(self)
    def forward(self, x):
        return self.quant_method.apply(self, x)
```

Prelude 当前:
```rust
// LinearInner 硬编码三个变体，加新格式要改 enum
enum LinearInner {
    Gpu(GpuLinear),
    Cpu(OnednnLinear),
    Quantized(QuantizedWeight),  // 只支持 Q4_0
}
```

**要做的**: 用 `LinearBackend` trait + `QuantFormat` registry（`inventory` crate），
每种量化格式自注册，`Linear` 只做 dispatch。后续加 Q4_K_M、FP8、INT4 GEMM 不用动 Linear。

### 1.2 Multi-Group KV Cache Coordinator — 混合注意力架构的通用缓存管理

vLLM:
```python
# 一个模型可以混合多种 attention 类型，每组独立管理
class HybridKVCacheCoordinator:
    """支持 full attention + sliding window + Mamba state 混合"""
    groups: List[KVCacheGroup]  # 每组有独立 block pool

# KV cache spec 是声明式的
class FullAttentionSpec(KVCacheSpec): ...
class SlidingWindowSpec(KVCacheSpec): ...
class MambaSpec(KVCacheSpec): ...
```

Prelude 当前:
```rust
// DeltaNet pool 和 KV cache 是两套独立的管理器
// 加新的混合架构（如 Mamba + attention）需要再写一套 pool
struct DeltaNetPool { ... }     // 专门给 DeltaNet
struct CacheManager { ... }     // 专门给 KV attention
```

**要做的**: 当支持更多混合架构时（Mamba-2、RWKV + attention），
参考 vLLM 的 `KVCacheSpec` 声明式规范 + `HybridKVCacheCoordinator` 统一管理。
当前只有 DeltaNet 一种混合架构，不急。

### 1.3 Speculative Decoding 框架 — 模块化 draft + 验证

vLLM:
```python
# Draft model 可插拔
class DraftModel: ...           # 小模型 draft
class EagleModel: ...           # EAGLE 架构
class NgramProposer: ...        # 无模型统计 draft

# Scheduler 统一处理
request.num_computed_tokens     # 已验证的
request.num_tokens_with_spec    # 包含投机的

# 验证逻辑与主模型共享 attention backend
```

Prelude: 完全没有 speculative decoding。

**要做的**: decode 吞吐的关键优化。设计时直接按模块化思路来：
- `DraftProposer` trait（小模型、n-gram、EAGLE 可插拔）
- Scheduler 统一 token 进度跟踪
- 验证阶段复用主模型的 forward path

### 1.4 Executor 抽象 — 为 multi-GPU 预留接口

vLLM:
```python
class Executor(ABC):
    @staticmethod
    def get_class(config) -> Type[Executor]:
        return {"ray": RayExecutor, "mp": MultiprocExecutor, "uni": UniProcExecutor}[backend]

    def collective_rpc(self, method, args, non_block=False):
        """广播命令到所有 worker"""
```

Prelude: 单 GPU queue，没有 executor 抽象。

**要做的**: 即使不马上做 TP/PP，提前定义 `Executor` trait + 单进程默认实现。
后面加 multi-GPU 时改动最小。核心接口只需要：
```rust
pub trait Executor: Send + Sync {
    fn forward(&self, batch: &Batch) -> Result<Output>;
    fn num_workers(&self) -> usize;
}
```

### 1.5 Structured Output / Constrained Decoding

vLLM:
```python
# Sampling 阶段集成 grammar-based token filtering
class SamplingParams:
    guided_decoding: GuidedDecodingParams  # JSON schema / regex / grammar
    logits_processors: List[LogitsProcessor]

# GrammarOutput 提供 valid token mask
class GrammarOutput:
    token_bitmask: Tensor  # 每个 position 的合法 token 掩码
```

Prelude: 只有 temperature / top_k / top_p，没有 constrained decoding。

**要做的**: API serving 刚需（function calling、JSON output）。
参考 vLLM 的 `LogitsProcessor` 管道 + `GuidedDecodingParams`，
集成 outlines / lm-format-enforcer 或类似的 Rust 实现。

### 1.6 Content-Addressed Prefix Cache

vLLM:
```python
# Block 内容的 hash 做 content-addressed 匹配
BlockHash = NewType("BlockHash", bytes)  # SHA256 or xxHash

# 自动去重，不需要显式 trie 结构
block_pool.get_one_block(block_hash_with_group_id)
```

Prelude: hash-trie 按 token 序列匹配，block-level 粒度。

**当前方案已经 work**，但 vLLM 的 content-addressed 方案更简单。
可以后续对比性能再决定是否切换。优先级低。

---

## 2. 应该避免的设计

### 2.1 gpu_model_runner.py — 309KB 的上帝文件

vLLM 的 `gpu_model_runner.py` 有 **309KB / ~8000+ 行**，混合了：
- Forward pass 执行
- CUDA graph 编译和缓存
- KV cache 更新
- Speculative decoding 集成
- 动态 batching/ubatching
- LoRA mixin
- KV connector mixin
- Encoder CUDA graph

这是典型的"所有逻辑塞进一个文件"反模式。Prelude 的 `runtime/` 目录按职责拆分
（`gpu_queue.rs`、`gpu_batch.rs`、`gpu_continuous.rs`、`cuda_graph.rs`）是更好的做法。

**原则**: 保持 Prelude 现有的按职责拆分风格，不要让任何文件超过 1000 行。

### 2.2 Python 动态 dispatch 的性能开销

vLLM 大量使用运行时 dispatch：
```python
# 每次 forward 都经过多层 Python dispatch
self.quant_method.apply(self, x)     # 量化方法 dispatch
self.backend.forward(q, k, v, ...)   # attention backend dispatch
executor.collective_rpc(...)          # worker dispatch
```

每层 Python 函数调用 ~100-500ns，在 decode 阶段（每 token 调用 N 层 × 3-4 个 dispatch）
累积可观。SGLang 也有同样问题。

**原则**: Prelude 的 Rust trait object dispatch 只有 ~2ns（vtable 间接调用），
这是语言优势。但也要注意：
- 不要在 hot path 上用 `Box<dyn Trait>` 做小对象（如 per-token 操作），
  `enum` dispatch 更适合
- 对于 Linear/Attention 这种粗粒度组件，trait object 完全 OK

### 2.3 Request 状态机过度复杂

vLLM 的 Request 有 10+ 种状态：
```python
WAITING, WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
WAITING_FOR_REMOTE_KVS, WAITING_FOR_STREAMING_REQ,
RUNNING, PREEMPTED,
FINISHED_STOPPED, FINISHED_LENGTH_CAPPED,
FINISHED_ABORTED, FINISHED_IGNORED, ...
```

很多状态是为了 P/D disaggregation 和 structured output 加的，
增加了 scheduler 的复杂度和 bug 面。

**原则**: 状态机尽量简单。Prelude 当前的
`Waiting → Prefilling → Decoding → Done` 是够用的。
只在真正需要时才加状态（如 speculative decoding 加一个 `Verifying`），
不要预设"万一以后需要"的状态。

### 2.4 过度抽象 — 多层 Coordinator / Connector / Mixin

vLLM 的 KV cache 管理有 5 层抽象：
```
KVCacheManager
  → KVCacheCoordinator (3 种实现)
    → BlockPool
      → FreeKVCacheBlockQueue
        → BlockHashToBlockMap
```

加上 `KVConnector`（P/D disaggregation）、`KVCacheSpec`（cache 类型声明）、
`RequestBlockHasher`（per-request hash 计算），理解完整流程需要跳转 6-7 个文件。

**原则**: 不要为了"通用性"引入不必要的抽象层。Prelude 的 `CacheManager` +
`BlockManager` + `PrefixCache` 三层已经足够清晰。只在有 3+ 种具体实现时才引入抽象。

### 2.5 全局可变状态 (OnceLock / static)

vLLM 大量使用全局状态：
```python
# 全局 registry，运行时修改
_QUANTIZATION_METHODS = {}
_ATTENTION_BACKENDS = {}

# 全局配置
_current_vllm_config = None
```

Prelude 之前的 `register_gpu_gemm()` 也是这个反模式（已经用 `GpuLinear` 直接 dispatch 替代了）。

**原则**: 继续保持"构造时决定，运行时不变"的模式。
`inventory` crate 的注册是编译期完成的，比运行时注册更安全。

### 2.6 向后兼容负担 — v0/v1 双版本共存

vLLM 同时维护 v0 和 v1 两套架构：
```
vllm/attention/     # v0 attention
vllm/v1/attention/  # v1 attention (完全重写)
```

大量代码是 `if v1: ... else: ...` 或 `from vllm.v1 import ... if V1 else from vllm import ...`。
这让代码库极其难以理解和维护。

**原则**: Prelude 还在早期，不需要考虑向后兼容。大胆删除旧代码（比如刚删的
`LinearInner::Candle` 和 `prelude-ggml-quants`），不要保留 fallback path。

---

## 3. 与 SGLang 对比总结

| 方面 | SGLang 更好 | vLLM 更好 | Prelude 应该学哪个 |
|------|------------|----------|-------------------|
| 量化 registry | 类似 | 类似 | 模式相同，学任一家都行 |
| Attention backend | 运行时 registry | 运行时 registry | 已经有 trait，模式对了 |
| Model registry | EntryClass 约定 | HF class → arch 映射 | 已用 inventory，比两家都干净 |
| Speculative decoding | 有但不够模块化 | EAGLE + ngram + 模块化 | 学 vLLM |
| KV cache 混合类型 | 不支持 | HybridCoordinator | 学 vLLM（等需要时） |
| Executor 抽象 | 不如 vLLM | UniProc/MP/Ray 可插拔 | 学 vLLM |
| Structured output | 有 | 有 | 两家都有，挑一个 Rust 实现集成 |
| 代码质量 | 较好 | 差（上帝文件、v0/v1 共存） | 学 SGLang 的简洁性 |
| CPU 推理 | 不支持 | 有但质量一般 | Prelude 自己最强 |
| 性能优化深度 | 较深 | 较深但 Python 开销大 | Prelude 语言优势明显 |

---

## 4. 建议优先级

| 优先级 | 方向 | 来源 | 理由 |
|--------|------|------|------|
| **P0** | LinearBackend trait + QuantFormat registry | vLLM/SGLang | 正在加更多量化格式 |
| **P1** | Speculative decoding 框架 | vLLM | Decode 吞吐关键优化 |
| **P1** | Constrained decoding | vLLM | API serving 刚需 |
| **P2** | Executor trait | vLLM | 为 multi-GPU 打基础 |
| **P2** | KV cache spec 声明式 | vLLM | 等更多混合架构时再做 |
| **P3** | Content-addressed prefix cache | vLLM | 现有方案已 work |
