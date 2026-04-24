/// Shared build.rs utilities.
///
/// Include from any subcrate's build.rs:
///   include!("../../../build_log.rs");  // adjust depth as needed
///
/// Provides:
///   build_log!("message")          — log to stderr + cargo warnings
///   track_submodule("cuLA")        — rerun when submodule updates

macro_rules! build_log {
    ($($arg:tt)*) => {{
        let _msg = format!($($arg)*);
        eprintln!("  [{}] {_msg}", env!("CARGO_PKG_NAME"));
        println!("cargo:warning={}", _msg);
    }};
}

/// Track a third_party submodule so build.rs reruns when `git submodule update` pulls new code.
///
/// Usage: `track_submodule("cuLA");`
///
/// Works by watching `.git/modules/third_party/<name>/HEAD` which changes on every checkout.
fn track_submodule(name: &str) {
    // Find workspace root (walk up from CARGO_MANIFEST_DIR until we find .git/)
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut dir = manifest.as_path();
    loop {
        if dir.join(".git").is_dir() {
            let head = dir.join(format!(".git/modules/third_party/{name}/HEAD"));
            if head.exists() {
                println!("cargo:rerun-if-changed={}", head.display());
            }
            return;
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => return,
        }
    }
}
