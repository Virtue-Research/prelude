# tvm-static-ffi

Minimal ABI for calling statically-linked CuTeDSL / cutlass-dsl AOT kernels
from Rust. Implements only the subset of the TVM FFI surface that is needed
to invoke `__tvm_ffi_<name>` symbols produced by
`cute.compile(...).export_to_c(...)` —— no dynamic loading, no function
registry, no object system.

## What this crate is (and is not)

| | `tvm-static-ffi` (this crate) | `tvm-ffi` (upstream) |
|---|---|---|
| Target kernels | AOT `.o` archives linked into the binary | `.so` loaded at runtime via TVM function registry |
| Dependencies | cudart, libbacktrace, stdc++ | libtvm_ffi.so, Python runtime |
| API surface | `DLTensor`, `TVMFFIAny`, `TVMSafeCallFn`, `call_tvm_ffi` | full TVM object system, macros, type traits |
| LoC | ~140 lines Rust + ~220 lines build.rs | ~3100 lines Rust across 12 files |
| Use case | shipping a single binary with fused GPU kernels baked in | wrapping Python-registered TVM functions from Rust |

The two crates serve different scenarios and can coexist. If you need
full TVM runtime integration, use the upstream
[`tvm-ffi`](https://crates.io/crates/tvm-ffi) crate.

## What you get

```rust
use tvm_static_ffi::{
    DLTensor, DLDataType, DLDevice, TVMFFIAny, TVMSafeCallFn,
    call_tvm_ffi, KDLCUDA, KDLBFLOAT,
};

unsafe extern "C" {
    // Exported by your AOT-compiled kernel (`cute.compile → export_to_c`
    // with CUTE_DSL_ENABLE_TVM_FFI=1).
    fn __tvm_ffi_my_kernel(
        handle: *mut std::ffi::c_void,
        args: *const TVMFFIAny,
        num_args: i32,
        ret: *mut TVMFFIAny,
    ) -> i32;
}

// Pack a DLTensor and dispatch.
let dl = DLTensor { /* ... */ };
let args = [TVMFFIAny::dltensor(&dl)];
unsafe { call_tvm_ffi(__tvm_ffi_my_kernel, &args)? };
```

See `src/lib.rs` for the full list of exposed types and helpers.

## Build requirements

| | |
|---|---|
| **CUDA toolkit** | Any version supported by nvcc used by consumers. This crate links `cudart_static`. |
| **tvm-ffi C++ runtime** | Vendored from `third_party/tvm-ffi` (apache/tvm-ffi). Only the error-retrieval helper is actually invoked; the rest is linked in because the submodule ships as a single unit. |
| **libbacktrace** | Built from `third_party/tvm-ffi/3rdparty/libbacktrace`, required by the TVM runtime code. |

### Environment variable overrides

| Variable | Purpose | Default |
|---|---|---|
| `TVM_FFI_ROOT` | Path to a checkout of apache/tvm-ffi | `$CARGO_WORKSPACE/third_party/tvm-ffi` |
| `CUDA_PATH` | CUDA toolkit root | `/opt/cuda`, `/usr/local/cuda`, `/usr/lib/x86_64-linux-gnu` |

`TVM_FFI_ROOT` is what lets you build this crate outside the prelude
workspace — point it at any `apache/tvm-ffi` checkout and it will
compile the needed C++ translation units.

## Used by

- [`cula`](../cula) — cuLA CuTeDSL AOT kernels
- [`flash-attn-v4`](../fa4) — Flash Attention v4 CuTeDSL AOT kernels
- [`flashinfer`](../flashinfer) — FlashInfer AOT kernels

## License

Apache-2.0. The vendored tvm-ffi C++ runtime is Apache-2.0 (Apache
Software Foundation).
