# 31. 量化 Matmul Kernel 设计 — 从 llama.cpp 学习并用 Rust 重写

## 现状问题

`Linear::Quantized(QMatMul)` 当前走 candle 的 dequant → F32 → matmul 路径：

```
GGUF Q4_K block: 152 bytes (256 values)
    → dequant to F32: 1024 bytes (6.7× 膨胀)
    → F32 matmul (candle)
```

这完全失去了量化的意义：
- 内存带宽浪费 6.7×
- 无法利用 VNNI/AMX 等整数 SIMD 指令
- 推理速度远不如 llama.cpp

## 目标

直接在量化数据上做 SIMD dot product，不 dequant：

```
GGUF Q4_K block: 152 bytes → AVX2/AVX-512 内核 → F32 结果
                              ↑ 不产生中间 F32 权重
```

## 技术方案：参考 llama.cpp，Rust 重写

### 为什么不 vendor llama.cpp 的 C 代码

| Vendor (FFI) | Rust 重写 |
|---|---|
| 带入 ggml 类型系统，不兼容 Linear/Tensor | 直接集成到 Linear::Quantized |
| FFI 边界开销 + unsafe | 无 FFI，类型安全 |
| 30+ 量化格式全带入 | 只实现需要的 3-4 种 |
| 难以和 attention pipeline fuse | 可精确控制优化 |
| 已有 `prelude-ggml-quants` FFI 全模型委托 | 这是 kernel 级别，不需要整模型 FFI |

SIMD intrinsics 在 C 和 Rust 里完全一样（`_mm256_maddubs_epi16` 等），翻译是机械性的。

### 量化格式优先级

| 格式 | 精度 | 用途 | 优先级 |
|------|------|------|--------|
| **Q4_K_M** | 4.75 bpw | 最常用的 GGUF 格式 | P0 |
| **Q8_0** | 8.5 bpw | 激活量化 / 高精度权重 | P0 |
| **Q4_0** | 4.5 bpw | 简单基线格式 | P1 |
| Q6_K | 6.5 bpw | 高精度变体 | P2 |
| Q5_K_M | 5.5 bpw | 中间精度 | P2 |
| Q2_K / IQ | 2-3 bpw | 极端压缩 | P3 |

覆盖 Q4_K_M + Q8_0 即可处理 90% 的 GGUF 模型。

## Block 数据结构

### Q4_0 (基线，32 元素一组)

```rust
/// Q4_0: 32 values → 18 bytes (4.5 bpw)
/// Scale: 1× FP16, Data: 32 nibbles packed into 16 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ4_0 {
    d: u16,        // FP16 scale (delta)
    qs: [u8; 16],  // 32 × 4-bit values packed as nibble pairs
}
// sizeof = 18 bytes
```

### Q8_0 (激活量化，32 元素一组)

```rust
/// Q8_0: 32 values → 34 bytes (8.5 bpw)
/// Scale: 1× FP16, Data: 32 signed int8 values
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ8_0 {
    d: u16,        // FP16 scale (delta)
    qs: [i8; 32],  // 32 × signed 8-bit values
}
// sizeof = 34 bytes
```

### Q4_K (高质量 4-bit，256 元素超级块)

```rust
/// Q4_K: 256 values → 144 bytes (4.5 bpw) — K-quant variant
/// 8 sub-blocks of 32 elements, each with 6-bit scale + 6-bit min
/// Super-block: FP16 d (scale-of-scales) + FP16 dmin (scale-of-mins)
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ4K {
    d: u16,           // FP16: super-block scale for quantized scales
    dmin: u16,        // FP16: super-block scale for quantized mins
    scales: [u8; 12], // 8 × 6-bit scales + 8 × 6-bit mins, packed
    qs: [u8; 128],    // 256 × 4-bit values packed as nibble pairs
}
// sizeof = 144 bytes
```

### Q8_K (配套 K-quant 的 8-bit 中间格式，256 元素)

```rust
/// Q8_K: 256 values → 292 bytes — used as activation quantization for K-quants
/// FP32 scale (higher precision), 256 × int8, plus 16 partial sums for dmin compensation
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ8K {
    d: f32,            // FP32 scale (not FP16!)
    qs: [i8; 256],     // 256 × signed 8-bit values
    bsums: [i16; 16],  // partial sums per 16-element sub-block (for dmin term)
}
// sizeof = 292 bytes
```

## Kernel 实现计划

### Q4_0 × Q8_0 Dot Product (AVX2)

llama.cpp 的核心循环（翻译为 Rust）：

```rust
/// Q4_0 · Q8_0 dot product, processing 32 elements per block.
/// Returns f32 result.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_q8_0_avx2(
    x: *const BlockQ4_0,  // quantized weights
    y: *const BlockQ8_0,  // quantized activations
    nb: usize,            // number of blocks
) -> f32 {
    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let xb = &*x.add(ib);
        let yb = &*y.add(ib);

        // 1. Load 16 bytes of Q4_0 nibbles → unpack to 32 bytes
        let qx = bytes_from_nibbles_32(xb.qs.as_ptr());

        // 2. Offset from [0..15] to [-8..+7]
        let off = _mm256_set1_epi8(8);
        let qx = _mm256_sub_epi8(qx, off);

        // 3. Load 32 signed bytes of Q8_0
        let qy = _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i);

        // 4. Integer dot product: 32 × i8×i8 → 8 × i32
        let dot = mul_sum_i8_pairs_float(qx, qy);

        // 5. Multiply by combined scale (d_q4 × d_q8)
        let d = _mm256_set1_ps(fp16_to_f32(xb.d) * fp16_to_f32(yb.d));

        // 6. FMA accumulate
        acc = _mm256_fmadd_ps(d, dot, acc);
    }

    hsum_float_8(acc)
}
```

关键辅助函数：

```rust
/// Unpack 16 bytes of nibble pairs → 32 bytes in [0..15]
#[inline(always)]
unsafe fn bytes_from_nibbles_32(ptr: *const u8) -> __m256i {
    let raw = _mm_loadu_si128(ptr as *const __m128i);
    let hi = _mm_srli_epi16(raw, 4);
    let lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
    // Interleave lo/hi nibbles
    _mm256_set_m128i(
        _mm_unpackhi_epi8(lo, hi),
        _mm_unpacklo_epi8(lo, hi),
    )
}

/// 32 × i8 × i8 → 8 × f32 (using maddubs + madd chain)
#[inline(always)]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    // maddubs needs unsigned × signed, so: |x| · sign(x)*y
    let ax = _mm256_sign_epi8(x, x);       // abs(x)
    let sy = _mm256_sign_epi8(y, x);       // y with x's sign
    let dot = _mm256_maddubs_epi16(ax, sy); // u8×s8 → s16, adjacent pairs summed
    let sum32 = _mm256_madd_epi16(dot, _mm256_set1_epi16(1)); // s16→s32
    _mm256_cvtepi32_ps(sum32)
}
```

### Q4_K × Q8_K Dot Product (AVX2)

更复杂 — 需要处理 6-bit packed scales 和 dmin 项：

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_k_q8_k_avx2(
    x: *const BlockQ4K,
    y: *const BlockQ8K,
    nb: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let m4 = _mm256_set1_epi8(0xF);

    for ib in 0..nb {
        let xb = &*x.add(ib);
        let yb = &*y.add(ib);

        let d = fp16_to_f32(xb.d) * yb.d;
        let dmin = fp16_to_f32(xb.dmin) * yb.d;

        // 1. Unpack 6-bit scales from 12 bytes → 8 scale + 8 min values
        let (scales, mins) = unpack_q4k_scales(&xb.scales);

        // 2. dmin contribution: sum(min[j] × bsums[j]) for j=0..8
        let mins_sum = dot_scales_bsums(&mins, &yb.bsums);
        acc = _mm256_sub_ps(acc, _mm256_set1_ps(dmin * mins_sum as f32));

        // 3. Process 4 groups of 64 elements
        let mut sumi = _mm256_setzero_si256();
        let q4 = xb.qs.as_ptr();
        let q8 = yb.qs.as_ptr();

        for j in 0..4 {
            // Load 32 bytes Q4_K, unpack low/high nibbles
            let q4bits = _mm256_loadu_si256(q4.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            // Load corresponding Q8_K chunks
            let q8l = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
            let q8h = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);

            // Dot product with per-sub-block scales
            let p16l = _mm256_madd_epi16(
                _mm256_set1_epi16(scales[2 * j] as i16),
                _mm256_maddubs_epi16(q4l, q8l),
            );
            let p16h = _mm256_madd_epi16(
                _mm256_set1_epi16(scales[2 * j + 1] as i16),
                _mm256_maddubs_epi16(q4h, q8h),
            );
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(sumi),
            acc,
        );
    }

    hsum_float_8(acc)
}
```

### VNNI 变体

AVX-512 VNNI 用 `dpbssd` 替代 `maddubs` + `madd` 链：

```rust
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn mul_sum_i8_pairs_avx512vnni(x: __m512i, y: __m512i) -> __m512 {
    let zero = _mm512_setzero_si512();
    let sum32 = _mm512_dpbssd_epi32(zero, x, y);  // 一条指令完成
    _mm512_cvtepi32_ps(sum32)
}
```

### AMX 变体 (未来)

AMX 用 tile 级别的矩阵乘法，一条 `_tile_dpbssd` 处理 8×16 的 matmul：

```rust
// 概念代码 — AMX 需要 tile 配置
unsafe fn matmul_q4_amx(/* ... */) {
    // 配置 tiles
    _tile_loadconfig(&tc);

    // Load weight tile (8 rows × 64 bytes)
    _tile_loadd(TMM0, weight_ptr, stride);

    // Load activation tile (16 rows × 32 bytes)
    _tile_loadd(TMM2, act_ptr, stride);

    // 一条指令: 8×16 matmul (128 i8×i8→i32 products)
    _tile_dpbssd(TMM4, TMM2, TMM0);

    // Store result (16 rows × 16 i32 values)
    _tile_stored(TMM4, result_ptr, stride);
}
```

throughput 约 AVX2 的 2.6×。

## Matmul 调度设计

### 激活量化

llama.cpp 的模式：权重存 Q4_K，前向时把 FP32 激活量化为 Q8_K，然后做 Q4_K × Q8_K dot product：

```
x (FP32/BF16) → quantize_row_q8_k(x) → Q8_K
W (Q4_K from GGUF, 已存储)

result = vec_dot_q4_k_q8_k(W, x_q8k)
```

激活量化非常快（vectorized per-block: 找 max → scale → round）。

### 线程并行

llama.cpp 用行并行 — 每个线程计算输出矩阵的不同行：

```
Output [M, N]:
  Thread 0: rows [0, M/nthread)
  Thread 1: rows [M/nthread, 2*M/nthread)
  ...
```

这和我们的 GemmPool / rayon 模式一致。

### 集成到 Linear::Quantized

当前路径：
```
Linear::Quantized(QMatMul) → QMatMul::forward(x)
  → candle dequant → F32 → F32 matmul
```

目标路径：
```
Linear::Quantized(QMatMul) → quantized_matmul(x, qweight)
  → quantize_activations(x) → Q8_K
  → tiled_q4k_q8k_matmul(qweight, x_q8k)  // 直接在量化数据上
  → F32 output
```

替换 `QMatMul::forward` 的内部实现，或在 `LinearInner::Quantized` 分支里直接调我们的 kernel。

## CPU Flash Attention 分块优化

llama.cpp 的 CPU flash attention 按 tile 做，在 tile 内部 dequant KV：

```
per-thread scratch:
  Q_tile:  [Q_TILE × DK]     float
  KQ:      [Q_TILE × KV_TILE] float (attention scores)
  V_tile:  [KV_TILE × DV]     float (dequant from Q8 KV cache)
  VKQ:     [Q_TILE × DV]      float (output accumulator)

for each KV tile:
  1. dequant K_tile from cache → K_f32
  2. QK = Q_tile @ K_f32^T
  3. softmax(QK)
  4. dequant V_tile from cache → V_f32
  5. VKQ += QK @ V_f32
```

所有 dequant 数据留在 L1/L2，不会冲刷到 DRAM。

这个可以和我们的 `cpu_prefill_attention` / `cpu_decode_attention` 结合，未来支持量化 KV cache。

## 实施阶段

### Phase 1: Block 结构体 + Q4_0 × Q8_0 kernel
- 定义 `BlockQ4_0`, `BlockQ8_0` (兼容 GGUF 格式)
- 实现 `vec_dot_q4_0_q8_0_avx2`
- 实现 `quantize_row_q8_0` (FP32 → Q8_0 激活量化)
- 基准测试 vs candle QMatMul

### Phase 2: Q4_K × Q8_K kernel
- 定义 `BlockQ4K`, `BlockQ8K`
- 实现 `vec_dot_q4_k_q8_k_avx2`（含 6-bit scale unpacking）
- 实现 `quantize_row_q8_k`
- 集成到 `Linear::Quantized` 的 forward 路径

### Phase 3: AVX-512 / VNNI 变体
- `vec_dot_q4_0_q8_0_avx512vnni` (dpbssd)
- `vec_dot_q4_k_q8_k_avx512` (512-bit 宽度)
- Runtime dispatch：检测 CPU 能力选择最优路径

### Phase 4: Matmul 封装
- `quantized_matmul(x: &Tensor, qweight: &QTensor) -> Tensor`
- 行并行 (rayon / GemmPool)
- 替换 `QMatMul::forward` 或 `LinearInner::Quantized` 分支
- 基准测试 vs llama.cpp（目标：性能在 80% 以内）

### Phase 5 (未来): 高级优化
- AMX 支持 (`_tile_dpbssd`)
- Weight repack (预解码 6-bit scales)
- KV cache 量化 (Q8 KV + flash attention tile 内 dequant)
- Q6_K / Q5_K_M 支持

## 参考文件

| 文件 | 内容 |
|------|------|
| `llama.cpp/ggml/src/ggml-common.h:170-355` | Block 结构体定义 |
| `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:540-700` | Q4_0 AVX2 kernel |
| `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:1740-1830` | Q4_K AVX2 kernel |
| `llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp:200-260` | AMX tile 配置 |
| `llama.cpp/ggml/src/ggml-cpu/ops.cpp:8400-8900` | CPU flash attention tiling |
| `llama.cpp/ggml/src/ggml-cpu/repack.cpp` | Weight 预处理 (4836 行) |
| `prelude/crates/prelude-core/src/ops/cpu/attention/dpbf16.rs` | 我们的 BF16 SIMD 参考 |
| `prelude/crates/prelude-core/src/models/common/linear.rs` | Linear::Quantized 集成点 |

## 性能目标

| 格式 | 当前 (candle dequant) | 目标 (native kernel) | llama.cpp |
|------|---------------------|---------------------|-----------|
| Q4_K_M 7B decode | ~15 tok/s | ~40 tok/s | ~50 tok/s |
| Q4_K_M 7B prefill | ~200 tok/s | ~800 tok/s | ~1000 tok/s |
| Q8_0 7B decode | ~12 tok/s | ~35 tok/s | ~40 tok/s |

目标是达到 llama.cpp 80% 的性能，同时保持代码可维护性和 Rust 类型安全。
