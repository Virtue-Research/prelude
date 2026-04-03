# Quantized Kernel Benchmark Results

**Date**: 2026-03-26
**Machine**: AMD Ryzen 9 7950X (16C/32T), 64GB DDR5
**CPU Features**: AVX-512F, AVX-512BW, AVX-512BF16
**Build**: `--release`, rayon pool = 16 threads pinned to NUMA node 0

## 1. Precision (llama.cpp style)

Test: `|quant_dot - f32_dot| / length`, threshold = 0.02
Test vector: `0.1 + 2.0 * cos(i + offset)`, size = 4096

| Format | Our Error | llama.cpp Ref | Status |
|--------|-----------|---------------|--------|
| Q4_0   | 0.001142  | 0.001143      | PASS   |
| Q4_K   | 0.001849  | 0.002425      | PASS   |
| Q4_0 (overflow) | 0.002003 | 0.002286 | PASS |
| Q4_K (overflow) | 0.000000 | 0.004850 | PASS |

精度与 llama.cpp reference 完全一致（Q4_0）或更好（Q4_K）。

## 2. Dot Product Throughput

| K | Q4_0 | Q4_K | Q4_K/Q4_0 |
|---|------|------|-----------|
| 256 | 0.02 us | 0.01 us | 2.0x |
| 512 | 0.04 us | 0.02 us | 2.0x |
| 1024 | 0.09 us | 0.03 us | 3.0x |
| 2048 | 0.19 us | 0.06 us | 3.2x |
| 4096 | 0.39 us | 0.12 us | 3.3x |

Q4_K dot product 比 Q4_0 快 2-3x（更大 block = 更好的 SIMD 利用率）。

## 3. Matmul Throughput (vs F32)

| M | K | N | F32 (us) | Q4_0 (us) | Q4_K (us) | Q4_0/F32 | Q4_K/F32 |
|---|---|---|----------|-----------|-----------|----------|----------|
| 1 | 1024 | 1024 | 47 | 105 | 38 | 0.45x | 1.23x |
| 1 | 2048 | 2048 | 197 | 443 | 141 | 0.44x | 1.39x |
| 1 | 4096 | 4096 | 1779 | 1810 | 556 | 0.98x | **3.20x** |
| 4 | 2048 | 2048 | 636 | 592 | 202 | 1.07x | **3.15x** |
| 4 | 4096 | 4096 | 3391 | 2666 | 813 | 1.27x | **4.17x** |
| 16 | 4096 | 4096 | 3690 | 5112 | 2353 | 0.72x | 1.57x |
| 1 | 1024 | 4096 | 181 | 545 | 197 | 0.33x | 0.91x |
| 1 | 4096 | 1024 | 217 | 534 | 198 | 0.41x | 1.09x |

### 分析

- **Q4_K matmul 在 decode (M=1, K≥4096) 和 small prefill (M=4) 最有优势**，3-4x faster than F32
- **Q4_0 matmul 在多数场景下比 F32 慢**——AVX2 kernel 效率不够，需要优化
- **Q4_K 一致优于 Q4_0**——256-element block 让 SIMD 更高效
- M=16 时 Q4_K 仍有 1.57x 优势，但大 M 时 compute-bound 压过 memory 优势

### 待优化

1. Q4_0 AVX2 kernel 性能差（比 F32 还慢），需要对照 llama.cpp 优化
2. Q4_K AVX2 的 min 计算用了 scalar 循环，应改 SIMD（hadd + madd）
3. 缺少 AVX-512 实现（当前机器支持，但 kernel 只用 AVX2）
4. 对比标杆：需要与 llama.cpp 的 ggml kernel 直接对比（TODO: benches/quant_vs_ggml.rs）

### 权重内存节省

| Format | Weight Size (4096×4096) | vs F32 |
|--------|------------------------|--------|
| F32    | 67.1 MB                | 1x     |
| Q4_0   | 9.4 MB                 | 7.1x   |
| Q4_K   | 9.4 MB                 | 7.1x   |

两种量化格式都是 4.5 bpw，内存节省相同（7.1x），但 Q4_K 精度和性能都更好。
