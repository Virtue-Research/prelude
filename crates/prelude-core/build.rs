fn configured_parallel_jobs() -> Option<usize> {
    for key in ["NUM_JOBS", "CARGO_BUILD_JOBS"] {
        if let Ok(value) = std::env::var(key)
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

fn run_cmake_build(build_dir: &std::path::Path) -> std::io::Result<std::process::ExitStatus> {
    let mut cmd = std::process::Command::new("cmake");
    cmd.arg("--build").arg(build_dir);
    if let Some(jobs) = configured_parallel_jobs() {
        cmd.arg("--parallel").arg(jobs.to_string());
    }
    cmd.status()
}

fn main() {
    // PTX kernel compilation moved to prelude-cuda crate.

    // ── oneDNN FFI (static linking — no .so needed at runtime) ──
    {
        use std::path::{Path, PathBuf};
        use std::process::Command;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();

        let ffi_dir: PathBuf = if let Ok(dir) = std::env::var("ONEDNN_FFI_DIR") {
            PathBuf::from(dir)
        } else {
            workspace_root.join("crates/onednn-ffi")
        };

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("onednn-build");
        let ffi_lib = build_dir.join("libonednn_ffi.a");
        let dnnl_lib_dir = build_dir.join("_deps/onednn-build/src");

        // Configure cmake if build dir doesn't exist yet (first build)
        if !build_dir.join("CMakeCache.txt").exists() {
            let jobs = configured_parallel_jobs().unwrap_or(1);
            println!(
                "cargo:warning=Configuring oneDNN FFI with cmake (first build downloads oneDNN ~200MB)..."
            );
            println!("cargo:warning=building oneDNN FFI with {} job(s)", jobs);

            std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

            let status = Command::new("cmake")
                .args([
                    "-S",
                    ffi_dir.to_str().unwrap(),
                    "-B",
                    build_dir.to_str().unwrap(),
                    "-DCMAKE_BUILD_TYPE=Release",
                ])
                .status();
            match status {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    panic!(
                        "cmake not found. Install cmake to auto-build onednn-ffi, \
                         or pre-build with: bash {}/build.sh",
                        ffi_dir.display()
                    );
                }
                Err(e) => panic!("cmake failed: {}", e),
                Ok(s) => assert!(s.success(), "cmake configure failed"),
            }
        }

        // Always run incremental build (fast if nothing changed)
        {
            let status = run_cmake_build(&build_dir).expect("cmake --build failed");
            assert!(status.success(), "cmake build failed");

            assert!(
                ffi_lib.exists(),
                "cmake build succeeded but libonednn_ffi.a not found at {}",
                ffi_lib.display()
            );
        }

        // Static link: our wrapper + oneDNN + system libs
        // Order matters: onednn_ffi depends on dnnl
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!(
            "cargo:rustc-link-search=native={}",
            dnnl_lib_dir.display()
        );
        println!("cargo:rustc-link-lib=static=onednn_ffi");
        println!("cargo:rustc-link-lib=static=dnnl");
        // C++ standard library (oneDNN is C++)
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // No OpenMP needed — oneDNN uses THREADPOOL runtime (rayon-backed)

        println!("cargo:rerun-if-env-changed=ONEDNN_FFI_DIR");
        println!("cargo:rerun-if-changed={}", ffi_lib.display());
        println!(
            "cargo:rerun-if-changed={}",
            ffi_dir.join("CMakeLists.txt").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            ffi_dir.join("src/onednn_ffi.cpp").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            ffi_dir.join("src/amx_gemm.c").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            ffi_dir.join("include/onednn_ffi.h").display()
        );
    }

    // ── NVTX profiling (link libnvToolsExt when feature is enabled) ──
    #[cfg(feature = "nvtx")]
    {
        let cuda_root = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
        println!("cargo:rustc-link-lib=dylib=nvToolsExt");
    }
}
