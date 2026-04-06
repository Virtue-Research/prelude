//! Build script for prelude-tvm-ffi.
//!
//! Compiles the vendored TVM FFI C++ runtime from third_party/tvm-ffi.
//! This is shared by FA4, FlashInfer, and cuLA kernel crates — compiled once,
//! linked into the final binary once.
//!
//! Also compiles:
//! - libbacktrace (required by tvm-ffi's backtrace.cc)
//! - tvm_error_helper.cc (Rust-callable C wrapper for TVM error messages)
//! - Links cuda_dialect_runtime_static.a if found

use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    // tvm-ffi lives in workspace third_party/
    let workspace_root = manifest_dir
        .parent().unwrap()  // crates/prelude-cuda/
        .parent().unwrap()  // crates/
        .parent().unwrap(); // workspace root
    let tvm_ffi_dir = workspace_root.join("third_party/tvm-ffi");
    let tvm_src = tvm_ffi_dir.join("src");
    let tvm_include = tvm_ffi_dir.join("include");
    let dlpack_include = tvm_ffi_dir.join("3rdparty/dlpack/include");

    if !tvm_src.exists() {
        anyhow::bail!(
            "third_party/tvm-ffi not found. Run: git submodule update --init third_party/tvm-ffi"
        );
    }

    // ── Phase 1: Compile TVM FFI C++ source ────────────────────────
    let cc_files: Vec<PathBuf> = walkdir(&tvm_src, "cc")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            !name.contains("win") && !name.contains("testing")
        })
        .collect();

    eprintln!("prelude-tvm-ffi: compiling {} C++ source files", cc_files.len());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .pic(true)
        .include(&tvm_include)
        .include(&dlpack_include)
        .define("TVM_FFI_EXPORTS", None)
        .define("NDEBUG", None)
        .warnings(false);

    for f in &cc_files {
        build.file(f);
    }

    // tvm_error_helper.cc — Rust-callable C wrapper for TVM error messages
    let error_helper = manifest_dir.join("src/tvm_error_helper.cc");
    if error_helper.exists() {
        build.file(&error_helper);
    }

    // ── Phase 2: Compile libbacktrace ──────────────────────────────
    compile_libbacktrace(&tvm_ffi_dir)?;

    // Use +whole-archive so kernel .o files can resolve TVM symbols at link time.
    build.link_lib_modifier("+whole-archive");
    build
        .try_compile("tvm_ffi_static")
        .context("Failed to compile vendored tvm_ffi")?;

    // ── Phase 3: Link cuda_dialect_runtime_static.a ────────────────
    let venv_dir = workspace_root.join("target/fa4-venv");
    if let Some(p) = find_file_recursive(&venv_dir, "libcuda_dialect_runtime_static.a") {
        println!("cargo:rustc-link-search=native={}", p.parent().unwrap().display());
        println!("cargo:rustc-link-lib=static:+whole-archive=cuda_dialect_runtime_static");
    }

    // ── Phase 4: Link CUDA runtime ─────────────────────────────────
    for candidate in [
        "/opt/cuda/targets/x86_64-linux/lib",
        "/opt/cuda/lib64",
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        if Path::new(candidate).join("libcudart.so").exists() {
            println!("cargo:rustc-link-search=native={candidate}");
            break;
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cuda");   // CUDA Driver API
    println!("cargo:rustc-link-lib=dylib=rt");      // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=dl");      // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

/// Compile libbacktrace from third_party/tvm-ffi/3rdparty/libbacktrace.
fn compile_libbacktrace(tvm_ffi_dir: &Path) -> Result<()> {
    let bt_dir = tvm_ffi_dir.join("3rdparty/libbacktrace");
    if !bt_dir.exists() {
        anyhow::bail!(
            "libbacktrace not found. Run: git submodule update --init --recursive third_party/tvm-ffi"
        );
    }

    let out = PathBuf::from(env::var("OUT_DIR")?);
    let config_dir = out.join("libbacktrace");
    std::fs::create_dir_all(&config_dir)?;

    let config_h = if cfg!(target_os = "linux") {
        r#"
#define BACKTRACE_ELF_SIZE 64
#define BACKTRACE_SUPPORTED 1
#define BACKTRACE_SUPPORTS_THREADS 1
#define BACKTRACE_USES_MALLOC 0
#define HAVE_ATOMIC_FUNCTIONS 1
#define HAVE_SYNC_FUNCTIONS 1
#define HAVE_DL_ITERATE_PHDR 1
#define HAVE_DLFCN_H 1
#define HAVE_FCNTL 1
#define HAVE_LINK_H 1
#define HAVE_LSTAT 1
#define HAVE_READLINK 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_DECL_STRNLEN 1
#define HAVE_DECL_GETPAGESIZE 0
"#
    } else if cfg!(target_os = "macos") {
        r#"
#define BACKTRACE_ELF_SIZE 64
#define HAVE_ATOMIC_FUNCTIONS 1
#define HAVE_DLFCN_H 1
#define HAVE_FCNTL 1
#define HAVE_MACH_O_DYLD_H 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_DECL_STRNLEN 1
#define HAVE_DECL_GETPAGESIZE 0
"#
    } else {
        r#"
#define HAVE_ATOMIC_FUNCTIONS 1
#define HAVE_DECL_STRNLEN 1
#define HAVE_DECL_GETPAGESIZE 0
"#
    };
    std::fs::write(config_dir.join("config.h"), config_h)?;

    let core_files = ["backtrace.c", "dwarf.c", "fileline.c", "posix.c",
                      "sort.c", "state.c", "alloc.c", "read.c", "mmapio.c", "mmap.c"];
    let format_file = if cfg!(target_os = "macos") { "macho.c" } else { "elf.c" };

    let mut build = cc::Build::new();
    build
        .opt_level(2)
        .pic(true)
        .include(&bt_dir)
        .include(&config_dir)
        .define("_GNU_SOURCE", None)
        .warnings(false);

    for name in &core_files {
        let f = bt_dir.join(name);
        if f.exists() {
            build.file(&f);
        }
    }
    let fmt = bt_dir.join(format_file);
    if fmt.exists() {
        build.file(&fmt);
    }

    build.try_compile("backtrace")
        .context("Failed to compile libbacktrace")?;

    Ok(())
}

/// Recursively find a file by name under a directory.
fn find_file_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_file_recursive(&path, name) {
                    return Some(found);
                }
            } else if path.file_name().is_some_and(|n| n == name) {
                return Some(path);
            }
        }
    }
    None
}

/// Recursively find files with given extension.
fn walkdir(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(walkdir(&path, ext));
            } else if path.extension().is_some_and(|e| e == ext) {
                result.push(path);
            }
        }
    }
    result
}
