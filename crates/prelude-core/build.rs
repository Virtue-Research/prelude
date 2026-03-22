#[cfg(feature = "onednn")]
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

#[cfg(feature = "onednn")]
fn run_cmake_build(build_dir: &std::path::Path) -> std::io::Result<std::process::ExitStatus> {
    let mut cmd = std::process::Command::new("cmake");
    cmd.arg("--build").arg(build_dir);
    if let Some(jobs) = configured_parallel_jobs() {
        cmd.arg("--parallel").arg(jobs.to_string());
    }
    cmd.status()
}

fn main() {
    // ── Compile custom CUDA kernels to PTX (loaded at runtime via cudarc) ──
    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        use std::process::Command;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let kernels_dir = PathBuf::from(&manifest_dir).join("src/ops/gpu/kernels");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        // Find nvcc
        let cuda_root = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let nvcc = PathBuf::from(&cuda_root).join("bin/nvcc");

        // Detect GPU compute capability, default to sm_80 (Ampere)
        let compute_cap = detect_compute_cap().unwrap_or(80);
        let arch_flag = format!("-arch=sm_{compute_cap}");

        // Include path for common headers
        let include_flag = format!("-I{}", kernels_dir.display());

        // Define kernel modules: (category, filename, output_name)
        let kernel_modules = [
            ("elementwise", "add.cu", "add"),
            ("elementwise", "silu_mul.cu", "silu_mul"),
            ("normalization", "rmsnorm.cu", "rmsnorm"),
            ("normalization", "add_rmsnorm.cu", "add_rmsnorm"),
            ("rope", "qknorm_rope.cu", "qknorm_rope"),
            ("moe", "routing.cu", "moe_routing"),
            ("moe", "gateup.cu", "moe_gateup"),
            ("moe", "down.cu", "moe_down"),
            ("kvcache", "append.cu", "kv_append"),
            ("kvcache", "knorm_rope_kv_write.cu", "knorm_rope_kv_write"),
            ("kvcache", "scatter_kv_cache.cu", "scatter_kv_cache"),
        ];

        // Track all source files for rerun-if-changed
        println!("cargo:rerun-if-changed=src/ops/gpu/kernels/common/common.cuh");
        println!("cargo:rerun-if-changed=src/ops/gpu/kernels/common/vec_utils.cuh");

        for (category, filename, output_name) in kernel_modules.iter() {
            let kernel_src = kernels_dir.join(category).join(filename);
            let ptx_path = out_dir.join(format!("{}.ptx", output_name));

            println!("cargo:rerun-if-changed={}", kernel_src.display());

            let status = Command::new(&nvcc)
                .args([
                    "--ptx",
                    kernel_src.to_str().unwrap(),
                    "-o",
                    ptx_path.to_str().unwrap(),
                    &arch_flag,
                    &include_flag,
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr", // Allow constexpr in device code
                ])
                .status()
                .unwrap_or_else(|e| panic!("Failed to run nvcc at {}: {}", nvcc.display(), e));

            assert!(
                status.success(),
                "nvcc PTX compilation of {} failed",
                filename
            );
        }
    }

    // ── oneDNN FFI (static linking — no .so needed at runtime) ──
    #[cfg(feature = "onednn")]
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

        let build_dir = ffi_dir.join("build");
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

}

#[cfg(feature = "cuda")]
fn detect_compute_cap() -> Option<u32> {
    // Try __CUDA_ARCH_LIST__ env var first, then nvidia-smi
    if let Ok(arch_list) = std::env::var("CUDA_ARCH_LIST") {
        if let Some(cap) = arch_list
            .split(',')
            .filter_map(|s| {
                let s = s.trim().replace('.', "");
                s.parse::<u32>().ok()
            })
            .max()
        {
            return Some(cap);
        }
    }

    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .next()?
        .trim()
        .replace('.', "")
        .parse::<u32>()
        .ok()
}
