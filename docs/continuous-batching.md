# Continuous Batching MVP

## Context

当前实现目标不是一步到位做出单次 `mixed forward` kernel，而是先把 multi-token generation 从 batch flush 模式切到 iteration-level 调度，并且允许新请求在已有 decode 进行时继续 admission。

这一步已经落地的核心变化：
1. `ScheduledEngine` 现在按任务路由 scheduler，而不是把所有任务塞进同一个 generation loop
2. multi-token generation 有了独立的 `continuous-generation` runtime
3. runtime 复用了已有的 `batch_prefill_paged` 和 `batch_decode_paged`，而不是重写执行层

## 当前路由规则

- `classify` / `embed` -> `batch-runtime`
- prepared `max_new <= 1` 的 generation -> `batch-runtime`
- prepared `max_new > 1` 且 paged decode 可用 -> `continuous-generation`
- prepared `max_new > 1` 但能力缺失 -> fallback 到 `batch-runtime`

## 当前实现形态

### 1. Batch Scheduler

保留原本的 batch-and-dispatch 主循环，职责变成：
- classify/embed 的稳定批处理路径
- prefill-only generation
- continuous runtime 不可用时的 generation fallback

### 2. Continuous Generation Runtime

新增独立 runtime，持有 per-sequence state：
- prepared request
- response channel
- pending token
- block table
- prompt length / next decode position
- streamed text offset
- finish reason / token logprobs / timing

它用 `Scheduler` 做 iteration-level 决策：
- `Scheduler::add_request()`
- `Scheduler::schedule_step()`
- `Scheduler::on_token_generated()`
- `Scheduler::finish_request()`

### 3. Mixed Scheduler Step

`SchedulerStep` 现在可以同时携带：
- `prefill_request_ids`
- `decode_request_ids`

也就是说，同一个 iteration 可以：
- 准入新的 prefill 请求
- 继续已有 running 序列的 decode

当前这个 “mixed” 是 **调度层 mixed**，不是 **单次 GPU forward mixed**。

## Execution Model

当前 continuous runtime 每轮大致做下面几件事：

```text
1. drain channel，收进新的 prepared generation request
2. scheduler.schedule_step()
3. 对本轮 admission 的请求，调用 batch_prefill_paged
4. 对本轮 decode 的请求，调用 batch_decode_paged
5. sample token / stop detection / streaming emit
6. finished request 释放 block table 和 deltanet slot
```

关键点：
- prefill 和 decode 可以在同一个 iteration 决策里同时发生
- 但执行上仍是两个 helper：一个 prefill helper，一个 decode helper
- streaming 和 non-streaming 共享同一套 state 推进逻辑

## 资源处理

MVP 版本的资源策略是偏保守的：
- prefill admission 会先按当前空闲 block 粗略裁剪 batch，避免直接把 batch_prefill_paged 打爆
- decode 前按需给活跃序列追加 paged block
- block exhaustion 会结束该序列或推迟新的 prefill，而不是直接做 scheduler-level preemption

## 已完成的验证

- 默认构建下：`cargo test -p prelude-core scheduler --lib`
- paged decode 构建下：`cargo test -p prelude-core scheduler --lib --features paged-attn,flash-attn-v3`

当前覆盖的回归点：
- classify/embed 仍留在 batch scheduler
- prepared prefill-only generation 仍走 batch scheduler
- prepared multi-token generation 走 continuous runtime
- `SchedulerStep` 能在 running decode 存在时发出 mixed step

## 当前限制

这份实现是 MVP，不是最终形态：

1. 还没有单次 `forward_mixed_step()`，所以 prefill 和 decode 仍不是一个真正的 GPU forward
2. continuous runtime 还没有接 block-manager 驱动的 preemption
3. 资源 admission 目前是粗粒度保守策略，不是完整的 cache-aware scheduling
4. continuous runtime 只覆盖 paged decode 路径

## 下一步

更合理的下一步是：
- 把 prefill + decode 收敛成真正的 single-forward mixed executor
- 把 block-manager 和 preemption 接进 continuous runtime
- 让 scheduler 的 memory budget 不再依赖保守近似
