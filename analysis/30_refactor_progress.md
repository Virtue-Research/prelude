# 重构进度记录

## 已完成

### 目录重构 (Phase 0-5)
- [x] 删除死文件 (`cache/planner.rs`, `cache/paged.rs`)
- [x] 统一文件命名 (`engine_struct→engine`, `tokenize→tokenizer`, runtime 加 `gpu_`/`cpu_` 前缀)
- [x] `scheduler/scheduled_engine.rs` → `engine/scheduled.rs`
- [x] `models/layers/` → `models/common/`
- [x] `models/architectures/*/` → `models/*/` (扁平化)
- [x] `engine/load.rs` + `engine/weights.rs` → `loading/` 新模块
- [x] GGUF 文件归入对应模型目录 (`qwen3/gguf.rs`, `qwen3_5/gguf.rs`)
- [x] `Cargo.toml` 加 `autobins=false` + `required-features`

### 自有层栈 (P0)
- [x] 新建 `Linear` struct (auto-dispatch OneDNN/candle)
- [x] 新建 `RmsNorm` struct (AVX-512 CPU / candle GPU)
- [x] 重命名: `Qwen3Mlp→GatedMlp`, `Qwen3RotaryEmbedding→RotaryEmbedding`, `prepare_qkv_varlen→fused_qkv_projection`
- [x] 全部 5 个模型迁移到 `Linear::load()` / `RmsNorm::from_weight()`
- [x] 删除 `AccelLinear`, `AccelRmsNorm`, `wrap_linear()`, `wrap_rmsnorm()`
- [x] 删除 `Gemma3RmsNorm` wrapper
- [x] MoE gate 改用 `candle_nn::Linear` (小矩阵不需要加速)

### 新抽象组件 (P1)
- [x] `AttentionBackend` trait + 5 个后端 impl (FA4/FlashInfer/FA3/FA2/CPU)
- [x] `select_backend()` 集中 cfg 选择
- [x] `TransformerBlock` 通用 pre-norm decoder block

### 去除 candle 高层依赖 (完成)
- [x] 创建 `loading/var_builder.rs` — 独立 VarBuilder (从 candle-nn 复制, ~580 行)
- [x] 创建 `nn_ops.rs` — 替代 candle-nn/candle-transformers 的所有功能:
  - Embedding, CandleLinear, CandleRmsNorm
  - ops: softmax, sigmoid, silu, log_softmax
  - Activation enum
  - rotary_emb: rope, rope_thd
  - moe: moe_gemm
  - generation: LogitsProcessor, Sampling
  - Qwen3Config struct
- [x] 删除 VarBuilder 的 candle-nn bridge 代码
- [x] 切换全部 ~20 个文件的 import: candle_nn → nn_ops, candle_transformers → nn_ops
- [x] MoE gate 字段: `candle_nn::Linear` → `CandleLinear`
- [x] CpuRmsNorm::new 清理: 移除 `candle_nn::RmsNorm` 参数
- [x] common/linear.rs: `candle_nn::Linear` → `CandleLinear`, `candle_nn::RmsNorm` → `CandleRmsNorm`
- [x] ops/onednn/ops.rs: `candle_nn::Linear` → `CandleLinear`
- [x] 从 Cargo.toml 删除 `candle-nn` 依赖
- [x] `candle-transformers` 改为可选 (behind `candle-baseline` feature, 仅 qwen3/gguf.rs 和 benchmark)
- [x] 最终依赖: 仅 `candle-core` (~25K 行)

### 文档
- [x] `docs/skills/adding-a-model.md` 更新
- [x] `docs/skills/adding-an-attention-backend.md` 更新
- [x] `docs/architecture.md` 更新
- [x] `docs/benchmarking.md` 新增 micro-benchmark 章节
- [x] `analysis/` 全部旧路径更新

### 设计文档
- [x] `analysis/27_sglang_comparison.md` — SGLang 对比 + 改进方案
- [x] `analysis/28_trait_split_design.md` — ModelForward 拆分设计
- [x] `analysis/29_layer_stack_design.md` — 层栈设计

### 去除 candle-nn 依赖 + unsafe 减少 (commit 3e251df)
- [x] ~20 个文件的 import 切换: candle_nn/candle_transformers → nn_ops
- [x] Cargo.toml 删除 candle-nn，candle-transformers 改为可选 (candle-baseline feature)
- [x] 引入 bytemuck + half crate 替代 transmute
- [x] 引入 nix 0.31 替代 libc unsafe syscall
- [x] UnsafeCell → RefCell, Vec::set_len → vec![0;n], from_raw_parts_mut → par_chunks_mut
- [x] SIMD unsafe 拆细 (common.rs/avx512.rs/dpbf16.rs/rope.rs)
- [x] 集中 storage_mut_and_layout 到 inplace_add_bf16/f32 safe helper

### AttentionBackend + TransformerBlock 接入 (commit 13571e9)
- [x] attn/mod.rs 自由函数 → AttentionBackend trait 委托 (-170 行 cfg 分发)
- [x] TransformerBlock 重设计 (闭包注入 attn + mlp)
- [x] Qwen3 + Qwen3-MoE 接入 TransformerBlock

### TransformerBlock 扩展 (当前 session, 未 commit)
- [x] Qwen3.5 接入 TransformerBlock (field destructuring 解决 DeltaNet &mut 借用)
- [x] Qwen3-Next 接入 TransformerBlock (同上)
- [x] Gemma3 保持自定义 (4-norm 架构不同，符合 SGLang 做法)
- 净减少: -61 行

---

## 当前进行：SGLang 优秀模式借鉴 (4 项)

### unsafe 减少 (已完成大部分)
- [x] transmute u16↔bf16: bytemuck::cast_vec 替代 (~9 处)
- [x] Vec::set_len: 改 vec![0; n] (~4 处)
- [x] NUMA syscall: 用 nix 0.31 crate (~2 处)
- [x] UnsafeCell: RefCell 替代 (raw_cpu.rs SCRATCH/SCRATCH_F32)
- [x] from_raw_parts_mut: par_chunks_mut 替代 (cpu attention)
- [x] SIMD unsafe 拆细: 只保留 load/store/pointer 的 unsafe 块
- [x] storage_mut_and_layout: 集中到 inplace_add_bf16/f32 safe helper
- [ ] GemmPool: 改 closure 替代函数指针 (~55 处)
- [ ] CpuTensor: 改存 &[u16] 替代 *const u16 (~12 处)
- [ ] CUDA graph update_tensor: 缩小 unsafe 范围 (~7 处)
- [ ] FFI: 新 crate 统一用 cxx

### AttentionBackend + TransformerBlock 接入 (已完成)
- [x] attn/mod.rs 的 5 个自由函数委托到 AttentionBackend trait
- [x] 消除 ~170 行重复 cfg 分发链
- [x] TransformerBlock 重设计: 不持有 MLP，用闭包注入 attn 和 mlp
- [x] Qwen3 接入 TransformerBlock
- [x] Qwen3-MoE 接入 TransformerBlock
- [x] Qwen3.5 接入 TransformerBlock (DeltaNet 用 field destructuring 解决借用)
- [x] Qwen3-Next 接入 TransformerBlock (同上)
- [x] Gemma3 保持自定义 (4-norm 架构不同，符合 SGLang 做法)

---

## 当前进行：SGLang 优秀模式借鉴 (4 项)

参考 SGLang 的优秀抽象，同时避免其技术债（scheduler god object、ForwardBatch 79 字段、
155 处 global、13+ monkey-patch、模型 copy-paste、1517 个 TODO/FIXME）。

### R1. ModelForward Trait Split
- 设计文档: `analysis/28_trait_split_design.md`
- 拆为: `ModelForward` (core) + `LogitsSplitModel` + `KvCacheModel` + `ClassifierModel` + `EmbeddingModel`
- 用 `as_xxx()` accessor 桥接 (Approach A)
- 改动范围: ~12 文件，~200-300 行 net
- Phase 1: 定义新 trait (non-breaking)
- Phase 2: 在 10 个 impl 上实现 sub-trait
- Phase 3: 迁移消费端 (generate.rs, classify.rs, embed.rs, prefill.rs)
- Phase 4: 从 ModelForward 移除旧方法
- 状态: [x] 完成 (commit e3567f9)

### R2. Runtime Attention Backend Selection
- 现状: `#[cfg(feature)]` 编译时选择，每个 binary 只支持一种 backend
- 目标: 运行时根据 GPU 能力选择 (H100→FA4, A100→FlashInfer, 旧卡→FA2, CPU→CPU)
- 改法: `select_backend()` 从 cfg 分支改为 runtime GPU probe
- 保持 AttentionBackend trait 精简 (5 方法)，不学 SGLang 的 21 个 backend × 11 种 forward 变体
- 状态: [x] 完成 (commit 843ff5d)

### R3. Model Registry via `inventory` Crate
- 现状: 添加新模型需改 3 个文件 (mod.rs, registry.rs, meta.rs)
- 目标: 每个模型文件自包含，`inventory::submit!` 自动注册
- 借鉴 SGLang 的 EntryClass 约定，但用编译时注册替代 Python 的运行时 import
- 状态: [x] 完成 (commit 9bb4ffd)

### R4. Quantization Method Injection
- 现状: 每个模型有独立 gguf.rs (800+ 行)，重复实现整个模型
- 目标: `Linear` 支持量化变体 (GGUF Q4/Q8, FP8)，模型代码不变
- 借鉴 SGLang 的 `QuantizationConfig.get_quant_method()` 模式
- 用 Rust enum variant 而非 Python 的 trait object，避免运行时开销
- 依赖 R3 (registry) 完成后更容易接入
- [x] Linear 支持 Quantized(QMatMul) variant
- [x] 标准 Qwen3.5 加 forward_with_cache (双模式: varlen + sequential)
- [x] GGUF 模型改为标准模型的 thin wrapper，删除全部重复层定义
- [x] gguf.rs: 1132 → 681 行 (-40%)
- 状态: [x] 完成 (commit 1346719)

### 扩展点 (留好不实现)
- [ ] 不假设单模型 (Stage trait for Omni)
- [ ] 不假设纯文本 I/O (ModalityInput/ModelOutput)
- [ ] 不假设 AR only (decode Q != 1 for dLLM)
- [ ] 不假设 causal only (自定义 attention mask for dLLM)
- [ ] Linear::with_tp() for tensor parallel
- [ ] Linear LoRA 支持
- [ ] NCCL dlopen for TP
- [ ] UCCL-P2P/EP 静态链接 for disaggregated + MoE EP
