# onednn-ffi

Minimal Rust FFI bindings to [Intel oneDNN](https://github.com/uxlfoundation/oneDNN)
(matmul + brgemm micro-kernels), with a **rayon-backed threadpool adapter**.
Static link only — no `libdnnl.so` at runtime.

## Scope

This crate is a thin `-sys`-style wrapper that exposes only the oneDNN
surface prelude-cpu actually needs. It is **not** a general-purpose Rust
binding for all of oneDNN.

| | |
|---|---|
| Upstream | [uxlfoundation/oneDNN](https://github.com/uxlfoundation/oneDNN) |
| Primitives exposed | `MATMUL`, `REORDER`, and the experimental `ukernel` (brgemm) API |
| Numeric formats | F32, BF16, INT8 (W8A8), FP8 E4M3 |
| CPU runtime | `THREADPOOL` — calls back into Rust-side rayon scheduler |
| Linking | Static: both `libdnnl.a` and `libonednn_ffi.a` are archived into the Rust binary. No `libdnnl.so`, no OpenMP runtime |
| Runtime deps | `libc`, `libstdc++` |

## Architecture

```
┌───────────────────────────────┐       C ABI       ┌────────────────────┐
│ Consumer crate                │ ────────────────→ │ libonednn_ffi.a    │
│ (e.g. prelude-cpu/src/onednn/ │                   │ (src/onednn_ffi.   │
│  ops.rs — safe wrapper)       │                   │  cpp, our C++ FFI) │
└───────────────────────────────┘                   └────────────────────┘
         ▲                                                     │
         │ rayon_parallel_for,                                  │ links statically
         │ rayon_get_num_threads                                ▼
         │ (Rust callbacks, exported                  ┌────────────────────┐
         │  from this crate)                          │ libdnnl.a          │
         │                                            │ (oneDNN, MATMUL +  │
         └────────────────────────────────────────────│  REORDER only)     │
                                                      └────────────────────┘
```

The reason the rayon callbacks live in this crate rather than in the
consumer: oneDNN's `THREADPOOL` runtime is configured to look up a set of
`extern "C"` symbols at **link time**, not at runtime. Those symbols must
be present in the final binary or `libonednn_ffi.a` fails to link. Putting
them here means every consumer of this crate gets a working threadpool
wiring for free.

If you want a different scheduler (Tokio, Crossbeam, plain `std::thread`),
vendor this crate and replace the three functions in `src/lib.rs`. The
signatures are fixed by oneDNN; the body is yours.

## Features

None. The crate is a single compile target.

## Build requirements

| Required | For | Default |
|---|---|---|
| **CMake ≥ 3.18** | Driving the oneDNN + FFI build | on `$PATH` |
| **C++17 toolchain** | Compiling oneDNN itself (and `onednn_ffi.cpp`) | `$CXX` / gcc / clang |
| **oneDNN source** | The full oneDNN source tree — checked out at the `third_party/oneDNN` submodule if you're inside the prelude workspace | `$WORKSPACE/third_party/oneDNN` |

First build is ~2-5 minutes (CMake configure + full oneDNN compile of
matmul + reorder primitives). Subsequent builds are instant because CMake
handles incremental compilation.

### Environment variable overrides

| Variable | Purpose | Default |
|---|---|---|
| `ONEDNN_SOURCE_DIR` | Path to any oneDNN checkout (use this to build this crate standalone, outside the prelude workspace) | `$CARGO_WORKSPACE/third_party/oneDNN` |
| `NUM_JOBS` / `CARGO_BUILD_JOBS` | Parallelism for `cmake --build` | `available_parallelism()` |

## Usage

Pure FFI — callers must build their own safe wrapper.

```rust
use onednn_ffi::{onednn_init, onednn_bf16_linear};
use std::ffi::c_void;

unsafe {
    onednn_init(); // call once, idempotent

    onednn_bf16_linear(
        input as *const c_void,
        weight as *const c_void,
        output as *mut c_void,
        /* m */ 1, /* k */ 4096, /* n */ 4096,
    );
}
```

See [`src/lib.rs`](src/lib.rs) for the complete list of exported
primitives: F32 linear, BF16 / S8 / F8E4M3 brgemm, packed-weight
management, fused SiLU×Mul, and post-op (bias / GELU / ReLU) paths.

For an example of a safe Rust wrapper that pins primitive caches, manages
packed weights, and dispatches by dtype, see
`prelude-cpu/src/onednn/ops.rs` in the prelude workspace.

## Rayon callbacks

This crate exports three `#[no_mangle]` functions on the Rust side that
oneDNN's `THREADPOOL` runtime calls back into:

| Symbol | Purpose |
|---|---|
| `rayon_parallel_for(n, body, ctx)` | Dispatch `n` work items to rayon's global pool |
| `rayon_get_num_threads()` | Return the rayon worker count |
| `rayon_get_in_parallel()` | `1` if the caller is already inside a rayon parallel region, `0` otherwise |

## License

Apache-2.0. The vendored oneDNN source is Apache-2.0 (Intel Corporation /
UXL Foundation).
