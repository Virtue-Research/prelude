# prelude-cpu

**Internal**: CPU backend for the Prelude inference engine.

## Why this is not a standalone crate

Unlike the sibling GPU-kernel crates (`cula`, `flash-attn-v4`, `deepgemm`,
`flashinfer`, `quant-gemm`, `cutlass-gemm`), which are self-contained
Rust wrappers around upstream CUDA libraries, `prelude-cpu`:

- **Hard-depends on [`prelude-core`](../prelude-core)** for the `Tensor`
  type, the `Ops` trait, and the backend registration / executor
  machinery. It cannot compile without it.
- **Implements** the `Ops` / `Executor` traits for CPU execution — it is
  an *implementation* of a prelude-core interface, not a general-purpose
  kernel library.
- **Contains prelude-specific attention, linear, and quant code**
  (`attn_cpu.rs`, `linear_backends.rs`, `ops/quant`, and oneDNN wrappers)
  — all of which assume the prelude model graph / dispatch conventions.

Publishing this crate on its own wouldn't give external users anything
useful; they'd have to also pull in prelude-core and its downstream
model / scheduler code, which effectively is "all of prelude".

## The `onednn-ffi` crate

The raw oneDNN bindings + CMake build logic live in a standalone
[`onednn-ffi`](../onednn-ffi) crate (a sibling workspace member).
`prelude-cpu` depends on it behind the `onednn` feature and keeps only
the **safe Rust wrappers** (`src/onednn/ops.rs`) that talk to
`prelude_core::Tensor`. That crate is publishable on its own — anyone
wanting raw oneDNN primitives + a rayon-backed threadpool adapter can
`cargo add onednn-ffi` without pulling in the rest of prelude.

## Layout

```
prelude-cpu/
├── Cargo.toml           # package "prelude-cpu"
└── src/
    ├── lib.rs                 # crate entry
    ├── cpu_ops.rs             # implements prelude_core::ops::Ops for CPU
    ├── executor.rs            # prelude_core::engine::Executor impl
    ├── attn_cpu.rs            # SDPA + paged attention on CPU
    ├── linear_backends.rs     # BF16 / quant Linear backends (routed through oneDNN)
    ├── ops/                   # BF16 attention, GEMM, RMSNorm, RoPE, quant kernels
    └── onednn/
        ├── mod.rs             # re-exports `onednn-ffi` as `ffi`, pulls in `ops`
        └── ops.rs             # safe Rust wrappers using `prelude_core::Tensor`
```

## Build requirements

When the `onednn` feature is enabled (the default), building
`prelude-cpu` transitively builds [`onednn-ffi`](../onednn-ffi), which
in turn CMake-builds oneDNN from source. That means you need:

- A C/C++ toolchain
- CMake 3.18+
- POSIX threads
- The `third_party/oneDNN` submodule (or `ONEDNN_SOURCE_DIR` env var
  pointing at any oneDNN checkout)

First build takes a few minutes while oneDNN compiles. Subsequent builds
are fast — CMake handles incremental rebuilds inside the `onednn-ffi`
crate's `OUT_DIR`.

## Running CPU kernel benchmarks

```bash
cargo run -p prelude-cpu --bin cpu_ops_bench --release
cargo run -p prelude-cpu --bin cpu_ops_bench --release -- quant
cargo run -p prelude-cpu --bin cpu_ops_bench --release -- gemm
```

## License

Apache-2.0 (same as the rest of the prelude workspace).
oneDNN is Apache-2.0 (Intel).
