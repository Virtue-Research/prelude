# NVTX Profiling

## 原理

用 NVIDIA NVTX (Tools Extension) 的 `nvtxRangePush/Pop` 在代码关键点打标注。
`nsys profile` 运行程序时，这些标注出现在 CPU timeline 上，形成嵌套层级。
不开 `nvtx` feature 时所有宏编译为空，零开销。

## 标注层级

```
forward
├─ layer[0]
│  ├─ attention
│  │  ├─ qkv_gemm
│  │  ├─ norm_rope
│  │  ├─ attn_compute
│  │  └─ o_proj
│  └─ mlp
│     ├─ gate_up+silu
│     └─ down_gemm
├─ layer[1]
│  ...
```

标注位置：
- `layer[i]` — `qwen3/mod.rs` 层循环
- `attention` / `mlp` — `transformer_block.rs`
- 各 kernel — `raw_cpu.rs` 的 `raw_attention_forward` 和 `raw_mlp_forward`

## 使用

```bash
# 编译（需要服务器上有 CUDA toolkit）
cargo build -p prelude-microbench --release --features prelude-core/nvtx

# 运行 profiling
nsys profile -o forward_profile ./target/release/prelude-microbench forward

# 查看结果（GUI）
nsys-ui forward_profile.nsys-rep

# 命令行统计（按 NVTX range name 聚合）
nsys stats --report nvtx_pushpop_trace forward_profile.nsys-rep

# 导出 CSV
nsys stats --report nvtx_pushpop_trace --format csv -o forward_stats forward_profile.nsys-rep

# 导出 SQLite（可自定义 SQL 查询）
nsys export --type sqlite -o forward.sqlite forward_profile.nsys-rep
```

## 实现

`crates/prelude-core/src/profiling.rs` 里两个宏：

```rust
nvtx_push!("attention");       // 静态字符串，零分配
nvtx_push!("layer[{}]", i);    // 格式化字符串，一次 CString 分配
nvtx_pop!();
```

底层是 FFI 调 `libnvToolsExt.so` 的 `nvtxRangePushA` / `nvtxRangePop`。
`build.rs` 在 `nvtx` feature 开启时链接 `libnvToolsExt`。

## 添加新标注

在需要的地方加 import 和 push/pop：

```rust
use crate::profiling::{nvtx_push, nvtx_pop};

nvtx_push!("my_kernel");
// ... 计算 ...
nvtx_pop!();
```

不开 feature 时这些行编译为空。
