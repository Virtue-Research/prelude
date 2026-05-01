//! Shared `build_log!` macro that mirrors the pattern every kernel crate
//! had inlined: stderr + `cargo:warning=` so messages show up both during
//! interactive builds and in `cargo` output.

/// Internal implementation of the `build_log!` macro. Consumers should use
/// the macro, not this function, so the caller's `CARGO_PKG_NAME` shows up
/// in the prefix.
#[doc(hidden)]
pub fn __build_log_inner(pkg: &str, msg: &str) {
    eprintln!("  [{pkg}] {msg}");
    println!("cargo:warning={msg}");
}

/// Log a build-script message. Emits to both stderr (for interactive
/// builds) and `cargo:warning=` (so it shows up even in noisy workspace
/// builds). Prefix includes the caller crate's name.
///
/// ```ignore
/// use prelude_kernelbuild::build_log;
/// build_log!("compiling {} kernels", count);
/// ```
#[macro_export]
macro_rules! build_log {
    ($($arg:tt)*) => {{
        let _msg = format!($($arg)*);
        $crate::_build_log_inner(env!("CARGO_PKG_NAME"), &_msg);
    }};
}
