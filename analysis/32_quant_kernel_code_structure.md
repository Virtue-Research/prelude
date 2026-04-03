# 32. 量化 Kernel 代码结构设计

## 设计原则

1. **每个 kernel 文件自包含** — 实现 + 单元测试在同一文件，review 时只需看一个文件
2. **测试分层** — kernel 精度 → matmul 精度 → Linear 集成 → 模型 e2e
3. **benchmark 独立** — 每个 kernel 有对应 benchmark，可独立运行
4. **naive reference 始终存在** — 所有优化 kernel 都有 scalar 参考实现做精度对照
5. **文件名即可定位** — 看到文件名就知道：什么格式、什么指令集、测试还是实现

## 目录结构

```
crates/prelude-core/src/ops/cpu/
├── mod.rs                          # CPU ops 统一入口
├── attention/                      # [已有] BF16 attention kernels
│   ├── mod.rs
│   ├── common.rs                   # 共享 SIMD helpers
│   ├── avx512.rs                   # weight×V accumulation
│   ├── dpbf16.rs                   # dpbf16ps dot products
│   └── tests/
│       ├── correctness.rs
│       ├── precision.rs
│       └── determinism.rs
│
├── quant/                          # [新] 量化 matmul kernels
│   ├── mod.rs                      # pub API: quantized_matmul()
│   ├── types.rs                    # Block 结构体定义 (Q4_0, Q8_0, Q4_K, Q8_K)
│   │
│   │  # ── 单格式 dot product kernels ──
│   ├── q4_0.rs                     # Q4_0 × Q8_0 dot product
│   │                               #   - scalar 参考实现
│   │                               #   - AVX2 实现
│   │                               #   - AVX-512 VNNI 实现 (cfg-gated)
│   │                               #   - dispatch 函数: 自动选最优
│   │                               #   - #[cfg(test)] 单元测试 (vs scalar)
│   │
│   ├── q4_k.rs                     # Q4_K × Q8_K dot product
│   │                               #   - scalar 参考 (含 6-bit scale unpacking)
│   │                               #   - AVX2 实现
│   │                               #   - AVX-512 实现
│   │                               #   - #[cfg(test)] 精度测试 + edge cases
│   │
│   ├── q8_0.rs                     # Q8_0 × Q8_0 dot product (纯 int8)
│   │
│   │  # ── 激活量化 ──
│   ├── quantize.rs                 # FP32/BF16 → Q8_0/Q8_K 激活量化
│   │                               #   - quantize_row_q8_0()
│   │                               #   - quantize_row_q8_k()
│   │                               #   - #[cfg(test)] round-trip 精度测试
│   │
│   │  # ── Matmul 编排 ──
│   ├── matmul.rs                   # 完整 matmul: 激活量化 + tiled dot + 线程并行
│   │                               #   - quantized_matmul(x: &Tensor, qweight: &QTensor) -> Tensor
│   │                               #   - 行并行调度 (rayon)
│   │                               #   - 输出 tiling (16×16)
│   │
│   │  # ── 测试 ──
│   └── tests/
│       ├── mod.rs                  # 共享 test helpers (gen data, tolerance)
│       ├── kernel_precision.rs     # 每种 kernel 的精度: SIMD vs scalar 参考
│       │                           #   - Q4_0 随机数据精度
│       │                           #   - Q4_K 随机数据精度
│       │                           #   - edge cases (全零, max scale, subnormals)
│       ├── matmul_correctness.rs   # matmul 级别: quantized_matmul vs F32 matmul
│       │                           #   - 小矩阵精确对比 (128×128)
│       │                           #   - 大矩阵统计精度 (4096×4096, max/mean error)
│       │                           #   - 不同形状 (M=1 decode, M=512 prefill)
│       └── matmul_determinism.rs   # 确定性: 多次运行 bit-exact
│
├── gemm.rs                         # [已有] BF16/F32 GEMM
├── gemm_pool.rs                    # [已有] 线程池
├── rmsnorm.rs                      # [已有]
├── rope.rs                         # [已有]
├── silu_mul.rs                     # [已有]
└── softmax.rs                      # [已有]

crates/prelude-core/src/bin/
├── cpu_ops_bench/
│   ├── main.rs                     # [已有] benchmark 入口
│   ├── attention.rs                # [已有]
│   ├── gemm.rs                     # [已有]
│   ├── rmsnorm.rs                  # [已有]
│   ├── rope.rs                     # [已有]
│   ├── silu_mul.rs                 # [已有]
│   └── quant.rs                    # [新] 量化 kernel benchmark
│                                   #   - dot product throughput (Q4_0, Q4_K)
│                                   #   - activation quantize throughput
│                                   #   - full matmul throughput vs F32
│                                   #   - 对比 candle QMatMul (dequant path)
│
├── quant_precision_test.rs         # [新] 独立精度测试 binary
│                                   #   - 加载真实 GGUF 模型权重
│                                   #   - Q4_K matmul vs F32 matmul 逐层对比
│                                   #   - 报告 max/mean/p99 误差
│
└── qwen3_bench.rs                  # [已有] e2e benchmark (添加 GGUF 路径)
```

## 每个文件的职责边界

### `types.rs` — 只有结构体，没有逻辑

```rust
// 所有 block 结构体，repr(C) 保证 GGUF 兼容
// 只定义类型 + bytemuck 安全转换
// 没有任何 SIMD 代码

#[repr(C)]
pub struct BlockQ4_0 { ... }

#[repr(C)]
pub struct BlockQ4K { ... }

// 安全的 &[u8] → &[BlockQ4_0] 转换
unsafe impl bytemuck::Pod for BlockQ4_0 {}
unsafe impl bytemuck::Zeroable for BlockQ4_0 {}
```

### `q4_0.rs` — 一个文件 = 一种 kernel 的全部

```rust
//! Q4_0 × Q8_0 dot product kernel.
//!
//! Scalar reference + AVX2 + AVX-512 VNNI implementations.
//! All SIMD variants tested against scalar for bit-level precision.

use super::types::*;

// ── Scalar 参考实现 (始终编译，测试用) ──────────────

pub fn vec_dot_q4_0_q8_0_scalar(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    // 最简单的实现，不做任何优化
    // 这是精度 ground truth
}

// ── AVX2 实现 ──────────────────────────────────────

#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_q8_0_avx2(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    // ...
}

// ── AVX-512 VNNI 实现 ─────────────────────────────

#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn vec_dot_q4_0_q8_0_avx512vnni(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    // ...
}

// ── Auto-dispatch ─────────────────────────────────

/// 自动选择最快的实现
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    if is_x86_feature_detected!("avx512vnni") {
        return unsafe { vec_dot_q4_0_q8_0_avx512vnni(x, y) };
    }
    if is_x86_feature_detected!("avx2") {
        return unsafe { vec_dot_q4_0_q8_0_avx2(x, y) };
    }
    vec_dot_q4_0_q8_0_scalar(x, y)
}

// ── 单元测试 ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_matches_scalar() {
        // 随机数据，AVX2 结果 vs scalar 结果
        // 允许 FP 累加顺序不同导致的微小误差 (< 1e-5 relative)
    }

    #[test]
    fn avx512vnni_matches_scalar() { ... }

    #[test]
    fn edge_case_all_zeros() { ... }

    #[test]
    fn edge_case_max_scale() { ... }

    #[test]
    fn edge_case_single_block() { ... }

    #[test]
    fn edge_case_large_k() { ... }  // K=4096 (typical hidden_size)
}
```

### `matmul.rs` — 编排层

```rust
//! Quantized matrix multiplication: x @ W^T where W is quantized.
//!
//! Flow: quantize_activations(x) → tiled dot product → F32 output
//! Thread parallelism: row-parallel via rayon.

/// Main entry point: quantized matmul replacing QMatMul::forward
pub fn quantized_matmul(
    x: &Tensor,           // [M, K] FP32 or BF16 activations
    qweight: &QTensor,    // [N, K] quantized weights
) -> Result<Tensor> {     // [M, N] F32 output
    // 1. 判断量化格式 (Q4_0, Q4_K, etc.)
    // 2. 量化激活 → Q8_0 or Q8_K
    // 3. 行并行 tiled matmul
    // 4. 包装为 Tensor 返回
}
```

### `tests/kernel_precision.rs` — SIMD vs scalar 交叉验证

```rust
//! Cross-validate all SIMD kernels against scalar reference.
//! Tests every format × every ISA combination.

#[test]
fn q4_0_avx2_precision() {
    // 1000 random vectors, compare AVX2 vs scalar
    // Report max relative error
    // Assert < 1e-5
}

#[test]
fn q4_k_avx2_precision() {
    // Q4_K has complex 6-bit scale unpacking
    // Extra attention to scale precision
}
```

### `tests/matmul_correctness.rs` — matmul 级别验证

```rust
//! Verify quantized_matmul produces correct results
//! by comparing against F32 matmul (dequant → F32 → matmul).

#[test]
fn small_matmul_q4_0() {
    // M=4, N=128, K=128
    // quantized_matmul vs dequant+F32 matmul
    // Exact comparison (should match closely)
}

#[test]
fn decode_shape_q4_k() {
    // M=1 (decode), N=4096, K=4096
    // This is the hot path for token generation
}

#[test]
fn prefill_shape_q4_k() {
    // M=512, N=4096, K=4096
    // Prefill shape, reports statistical error
}

#[test]
fn error_stats_q4_k_large() {
    // M=32, N=4096, K=11008 (intermediate_size)
    // Report: max_abs_error, mean_abs_error, p99_rel_error
    // Assert max_rel_error < 0.01
}
```

### `bin/cpu_ops_bench/quant.rs` — 性能 benchmark

```rust
//! Quantized kernel benchmarks.
//!
//! Reports: throughput (GOPS), bandwidth (GB/s), comparison vs F32.

pub fn run_benchmarks() {
    bench_dot_product_q4_0(4096);      // K=4096
    bench_dot_product_q4_k(4096);
    bench_quantize_activations(4096);
    bench_matmul_q4_k(1, 4096, 4096);    // decode
    bench_matmul_q4_k(512, 4096, 4096);  // prefill
    bench_matmul_q4_k_vs_f32(1, 4096, 4096);  // speedup ratio
    bench_matmul_q4_k_vs_qmatmul(1, 4096, 4096); // vs candle dequant
}

fn bench_dot_product_q4_0(k: usize) {
    // 单次 dot product latency + throughput
    // 报告: ns/block, GOPS, GB/s (带宽利用率)
    // 分别测 scalar / AVX2 / AVX-512 VNNI
}

fn bench_matmul_q4_k_vs_f32(m: usize, n: usize, k: usize) {
    // 1. F32 matmul (candle)
    // 2. quantized_matmul (our kernel)
    // 3. QMatMul (candle dequant path)
    // 报告: ms, speedup, 带宽利用率
}
```

### `bin/quant_precision_test.rs` — 真实权重精度测试

```rust
//! Load a real GGUF model and verify layer-by-layer precision.
//!
//! Usage: cargo run --bin quant_precision_test -- model.gguf
//!
//! Reports per-layer: max_error, mean_error, p99_error
//! between quantized_matmul and dequant→F32→matmul.

fn main() {
    let path = std::env::args().nth(1).expect("usage: quant_precision_test <model.gguf>");

    // 1. 加载 GGUF
    // 2. 对每个 Linear 层:
    //    a. 生成随机输入 x
    //    b. result_quant = our quantized_matmul(x, layer.weight)
    //    c. result_f32 = dequant(layer.weight) → F32 matmul
    //    d. 比较 result_quant vs result_f32
    // 3. 报告汇总统计
}
```

### e2e 测试 — 模型级别

e2e 精度测试不需要新文件，扩展 `qwen3_bench.rs`：

```rust
// 在 qwen3_bench.rs 中添加 GGUF 路径:
// 1. 加载 GGUF 模型 (使用新的 quantized_matmul path)
// 2. 加载 F16 模型 (标准 path)
// 3. 相同 prompt → 比较 logits
// 4. 报告 top-1 match rate, KL divergence
// 5. 测 decode throughput (tok/s)
```

## 测试层次总结

```
Layer 1: Kernel Unit Tests (cargo test)
  ├── SIMD vs scalar 精度 (每种格式 × 每种 ISA)
  ├── Edge cases (zeros, max, single block)
  └── 确定性 (多次运行 bit-exact)

Layer 2: Matmul Correctness (cargo test)
  ├── 小矩阵精确对比
  ├── 大矩阵统计精度 (max/mean/p99 error)
  └── 不同形状 (M=1 decode, M=512 prefill)

Layer 3: Kernel Benchmarks (cargo run --bin cpu_ops_bench)
  ├── dot product throughput per format
  ├── activation quantize throughput
  ├── full matmul throughput
  └── vs F32 / vs candle QMatMul 对比

Layer 4: Real Weight Precision (cargo run --bin quant_precision_test)
  ├── 加载真实 GGUF 权重
  ├── 逐层 quantized_matmul vs F32 对比
  └── 汇总统计 (max/mean/p99)

Layer 5: E2E Model (cargo run --bin qwen3_bench)
  ├── GGUF vs F16 logit 对比
  ├── Top-1 match rate, KL divergence
  └── Decode throughput (tok/s)
```

## Review 友好性

每个 PR 可以独立 review 一个层次：

| PR | 范围 | Review 重点 |
|---|---|---|
| PR1 | `types.rs` | Block 结构体是否和 GGUF 兼容 |
| PR2 | `q4_0.rs` + tests | scalar 实现正确性 + AVX2 vs scalar |
| PR3 | `q4_k.rs` + tests | 6-bit scale unpacking 正确性 |
| PR4 | `quantize.rs` + tests | 激活量化 round-trip 精度 |
| PR5 | `matmul.rs` + correctness tests | 编排逻辑 + 线程安全 |
| PR6 | `Linear` 集成 | QMatMul → quantized_matmul 替换 |
| PR7 | bench + e2e | 性能数据 + 模型精度 |

每个 PR 小于 300 行实现 + 测试，容易 review。
