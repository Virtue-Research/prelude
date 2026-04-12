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

/// Absolute path to this crate's `scripts/` directory, which contains
/// the shared `dsl_driver.py` module that consumer compile scripts
/// import. Consumer build scripts should call this from their
/// `build.rs` and pass the result as the `PRELUDE_KB_SCRIPTS_DIR` env
/// var when spawning their Python compile script, so the script can
/// `sys.path.insert(0, os.environ["PRELUDE_KB_SCRIPTS_DIR"])` before
/// importing.
///
/// Resolved via `CARGO_MANIFEST_DIR` at the build-support library's
/// compile time — so the path is baked into the consumer crate's
/// `build.rs` binary and always points at a concrete checkout of
/// prelude-kernelbuild regardless of where the consumer lives in the
/// workspace.
pub fn scripts_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts")
}
