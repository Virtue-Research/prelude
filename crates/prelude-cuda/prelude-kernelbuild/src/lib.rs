//! Shared build-script helpers for Prelude's CUDA kernel crates.
//!
//! This crate is a **build dependency** — it is linked into consumer crates'
//! `build.rs` scripts, not into their runtime binaries. It exists so the
//! CUDA toolkit discovery, nvcc invocation, Python venv management, and
//! TVM-FFI dispatch-table codegen aren't copied verbatim across every kernel
//! subcrate (`cula`, `fa4`, `flashinfer`, `deepgemm`, `cutlass-gemm`, and
//! `prelude-cuda`'s own PTX pipeline).
//!
//! ## Modules
//!
//! * [`nvcc`] — CUDA toolkit discovery, arch probing, single-file nvcc
//!   compile helpers for `.cu → .ptx` and `.cu → .o`, CUDA runtime linking,
//!   workspace/submodule helpers.
//! * [`venv`] — Python virtualenv provisioning, `uv`/pip wrapping, CUDA
//!   wheel index detection, importability checks.
//! * [`dispatch`] — TVM-FFI dispatch-table codegen: manifest parsing,
//!   extern-block emission, stub generators for when no kernels compile.
//! * [`archive`] — `.o` collection, `ar rcs` + whole-archive linking.
//! * [`dsl`] — CuTeDSL compile driver: per-arch Python invocation,
//!   script-hash cache invalidation, sticky failure markers.
//!
//! ## Design constraints
//!
//! Every heavy kernel crate's `build.rs` has to compile this crate first, so
//! keep dependencies minimal (`anyhow + serde_json + sha2 + hex`). Don't
//! pull in `tokio`, `clap`, or anything else that would balloon build-time
//! compile costs across the workspace.

pub mod archive;
pub mod dispatch;
pub mod dsl;
pub mod log;
pub mod nvcc;
pub mod venv;

// Re-export the log macro at crate root for convenient `use`.
pub use crate::log::__build_log_inner as _build_log_inner;
