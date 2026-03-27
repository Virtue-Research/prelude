//! Zero-overhead NVTX profiling annotations.
//!
//! Compile with `--features nvtx` to enable. Without the feature, all macros
//! are no-ops (zero code emitted).
//!
//! Usage:
//!   cargo build --features nvtx
//!   nsys profile ./target/release/prelude-microbench forward

#[cfg(feature = "nvtx")]
pub mod ffi {
    use std::ffi::c_char;
    use std::ffi::c_int;

    unsafe extern "C" {
        pub fn nvtxRangePushA(message: *const c_char) -> c_int;
        pub fn nvtxRangePop() -> c_int;
    }
}

/// Push a named NVTX range. No-op without `nvtx` feature.
///
/// Static literal: `nvtx_push!("attention")` — zero allocation.
/// Formatted: `nvtx_push!("layer[{}]", i)` — one CString allocation.
macro_rules! nvtx_push {
    ($name:literal) => {
        #[cfg(feature = "nvtx")]
        #[allow(unused_unsafe)]
        // SAFETY: pointer is a valid null-terminated literal produced by concat!
        unsafe { $crate::profiling::ffi::nvtxRangePushA(concat!($name, "\0").as_ptr().cast()) };
    };
    ($fmt:literal, $($arg:tt)*) => {
        #[cfg(feature = "nvtx")]
        {
            let s = std::ffi::CString::new(format!($fmt, $($arg)*)).unwrap();
            #[allow(unused_unsafe)]
            // SAFETY: CString guarantees a valid null-terminated pointer
            unsafe { $crate::profiling::ffi::nvtxRangePushA(s.as_ptr()) };
        }
    };
}

/// Pop the current NVTX range. No-op without `nvtx` feature.
macro_rules! nvtx_pop {
    () => {
        #[cfg(feature = "nvtx")]
        #[allow(unused_unsafe)]
        unsafe { $crate::profiling::ffi::nvtxRangePop() };
    };
}

pub(crate) use nvtx_push;
pub(crate) use nvtx_pop;
