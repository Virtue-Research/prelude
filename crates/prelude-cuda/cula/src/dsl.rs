//! CuTe DSL kernel dispatch for cuLA.
//!
//! All the registry machinery now lives in the shared
//! [`prelude_cutedsl`] crate. This module is a thin facade that:
//!
//!   1. Re-exports the runtime types under `cula::dsl::*` so existing
//!      downstream callers (`use cula::dsl::DslKernelRegistry`, etc.)
//!      keep compiling without changes.
//!   2. Includes the build-generated dispatch table from `OUT_DIR`,
//!      which defines `lookup_dsl(kernel_type, key, arch)` and the
//!      `unsafe extern "C"` declarations for every compiled variant.
//!   3. Owns a lazy singleton [`DslKernelRegistry`] wired to cuLA's
//!      own `lookup_dsl`, exposed via [`registry()`]. Consumers just
//!      call `cula::dsl::registry()` to get a `&'static` reference
//!      rather than constructing their own.

use std::sync::OnceLock;

// Runtime re-exports from the shared crate — keeps
// `use cula::dsl::{DslKernelRegistry, TVMFFIAny, ...}` working.
pub use prelude_cutedsl::{
    DLDataType, DLDevice, DLTensor, DslKernelRegistry, KDLBFLOAT, KDLCUDA, KDLFLOAT, KDLINT,
    LookupFn, TVMFFIAny, TVMSafeCallFn, contiguous_strides, detect_gpu_arch, make_dltensor,
};

// Build-generated dispatch table. Defines
// `pub(crate) fn lookup_dsl(_kernel_type: &str, key: &str, arch: u32) -> Option<TVMSafeCallFn>`
// plus the `unsafe extern "C" { fn __tvm_ffi_<name>(...); ... }` block.
include!(concat!(env!("OUT_DIR"), "/cula_dsl_dispatch.rs"));

/// Global lazy registry for cuLA's DSL kernels. Every call returns the
/// same `&'static` instance — the underlying compute-capability probe
/// only runs once. Consumer crates (e.g. prelude-cuda's
/// `attn/kda_decode.rs`) should use this accessor rather than
/// constructing their own `DslKernelRegistry::new(...)` so the
/// singleton stays consistent across call sites.
pub fn registry() -> &'static DslKernelRegistry {
    static REG: OnceLock<DslKernelRegistry> = OnceLock::new();
    REG.get_or_init(|| DslKernelRegistry::new(lookup_dsl as LookupFn))
}
