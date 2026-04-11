//! Build script for the `onednn-ffi` crate.
//!
//! Invokes CMake against `csrc/CMakeLists.txt`, which builds:
//!   - `libdnnl.a` from the oneDNN source tree (matmul + reorder primitives only)
//!   - `libonednn_ffi.a` — our thin C++ wrapper that pins the primitive cache and
//!     exposes a flat C ABI for the Rust side
//!
//! Both archives are static-linked into the final Rust binary — no libdnnl.so
//! is required at runtime. Only the standard C++ runtime stays dynamic.
//!
//! Source discovery (first match wins):
//!   - `$ONEDNN_SOURCE_DIR`     — explicit override, any oneDNN checkout
//!   - workspace layout         — `$WORKSPACE/third_party/oneDNN` (zero-config for prelude)
//!
//! Parallelism:
//!   - `$NUM_JOBS` / `$CARGO_BUILD_JOBS` honored by cargo are forwarded to
//!     `cmake --build --parallel`, falling back to `available_parallelism()`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn configured_parallel_jobs() -> Option<usize> {
    for key in ["NUM_JOBS", "CARGO_BUILD_JOBS"] {
        if let Ok(value) = env::var(key)
            && let Ok(jobs) = value.parse::<usize>()
            && jobs > 0
        {
            return Some(jobs);
        }
    }
    std::thread::available_parallelism()
        .ok()
        .map(std::num::NonZeroUsize::get)
}

fn run_cmake_build(build_dir: &Path) -> std::io::Result<std::process::ExitStatus> {
    let mut cmd = Command::new("cmake");
    cmd.arg("--build").arg(build_dir);
    if let Some(jobs) = configured_parallel_jobs() {
        cmd.arg("--parallel").arg(jobs.to_string());
    }
    cmd.status()
}

fn locate_onednn_source(manifest_dir: &Path) -> PathBuf {
    // Explicit override wins.
    if let Ok(p) = env::var("ONEDNN_SOURCE_DIR") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    // Fallback: prelude workspace layout
    // (crates/onednn-ffi/build.rs → ../../ = workspace root → third_party/oneDNN).
    manifest_dir
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // workspace root
        .join("third_party/oneDNN")
}

fn main() {
    println!("cargo:rerun-if-env-changed=ONEDNN_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=NUM_JOBS");
    println!("cargo:rerun-if-env-changed=CARGO_BUILD_JOBS");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("onednn-build");
    let ffi_lib = build_dir.join("libonednn_ffi.a");
    // With `add_subdirectory(${ONEDNN_SOURCE_DIR})` the produced libdnnl.a lives
    // at `<build>/onednn/src/libdnnl.a`.
    let dnnl_lib_dir = build_dir.join("onednn/src");

    let onednn_source = locate_onednn_source(&manifest_dir);
    if !onednn_source.join("CMakeLists.txt").exists() {
        panic!(
            "oneDNN source not found at {}. Either run `git submodule update \
             --init third_party/oneDNN` inside the prelude workspace, or set \
             ONEDNN_SOURCE_DIR=/path/to/oneDNN to build this crate standalone.",
            onednn_source.display()
        );
    }

    // Configure once per OUT_DIR (first build only).
    if !build_dir.join("CMakeCache.txt").exists() {
        let jobs = configured_parallel_jobs().unwrap_or(1);
        eprintln!(
            "onednn-ffi: configuring CMake with {} job(s) (first build only)",
            jobs
        );
        std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

        let status = Command::new("cmake")
            .args([
                "-S",
                csrc_dir.to_str().unwrap(),
                "-B",
                build_dir.to_str().unwrap(),
                "-DCMAKE_BUILD_TYPE=Release",
                &format!("-DONEDNN_SOURCE_DIR={}", onednn_source.display()),
            ])
            .status();
        match status {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                panic!(
                    "cmake not found. Install cmake to build onednn-ffi, or \
                     pre-build manually with: bash {}/build.sh",
                    csrc_dir.display()
                );
            }
            Err(e) => panic!("cmake configure: {}", e),
            Ok(s) => assert!(s.success(), "cmake configure failed"),
        }
    }

    // Incremental build every time (fast if nothing changed).
    let status = run_cmake_build(&build_dir).expect("cmake --build failed to spawn");
    assert!(status.success(), "cmake build failed");
    assert!(
        ffi_lib.exists(),
        "cmake build succeeded but libonednn_ffi.a not found at {}",
        ffi_lib.display()
    );

    // Link order: our FFI wrapper first, then oneDNN, then stdc++.
    // Both archives use +whole-archive because the Rust side holds function
    // pointers into them — dead-code elim would otherwise drop unreferenced
    // primitive entry points.
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}", dnnl_lib_dir.display());
    println!("cargo:rustc-link-lib=static=onednn_ffi");
    println!("cargo:rustc-link-lib=static=dnnl");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Change tracking: any of these trigger a rebuild.
    println!(
        "cargo:rerun-if-changed={}",
        csrc_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        csrc_dir.join("src/onednn_ffi.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        csrc_dir.join("src/amx_gemm.c").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        csrc_dir.join("include/onednn_ffi.h").display()
    );
}
