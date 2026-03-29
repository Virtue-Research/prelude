//! Build script: compile DeepGEMM BF16/FP8 GEMM wrapper with nvcc.
//!
//! Requires:
//! - CUDA Toolkit with nvcc (sm_90a support; sm_100a for Blackwell)
//! - CUTLASS headers (auto-cloned from GitHub)
//! - DeepGEMM headers (auto-cloned from GitHub)

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_BRANCH: &str = "main";

const DEEPGEMM_REPO: &str = "https://github.com/deepseek-ai/DeepGEMM.git";
const DEEPGEMM_BRANCH: &str = "main";

fn main() {
    println!("cargo:rerun-if-changed=src/deepgemm_wrapper.cu");
    println!("cargo:rerun-if-changed=src/common.cuh");
    println!("cargo:rerun-if-changed=src/sm90_bf16.cuh");
    println!("cargo:rerun-if-changed=src/sm90_fp8.cuh");
    println!("cargo:rerun-if-changed=src/sm100_bf16.cuh");
    println!("cargo:rerun-if-changed=src/sm100_fp8.cuh");
    println!("cargo:rerun-if-changed=src/attention.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // 1. Ensure CUTLASS headers
    let cutlass_dir = ensure_cutlass(&out_dir);
    let cutlass_include = cutlass_dir.join("include");

    // 2. Ensure DeepGEMM headers (cloned from upstream, patched for fat binary)
    let deepgemm_dir = ensure_deepgemm(&out_dir);
    let deepgemm_include = deepgemm_dir.join("deep_gemm/include");

    // 3. Find CUDA toolkit
    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }

    // 4. Compile deepgemm_wrapper.cu
    let common_args = [
        "-std=c++20", "-O3", "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
    ];
    let src_dir = manifest_dir.join("src");
    let include_args = [
        "-I", cutlass_include.to_str().unwrap(),
        "-I", deepgemm_include.to_str().unwrap(),
        "-I", &format!("{}/include", cuda_path.display()),
        "-I", src_dir.to_str().unwrap(),
    ];

    // Compile for SM90a always. Add SM100a if toolkit supports it (fat binary).
    // SM90 kernel bodies are guarded by __CUDA_ARCH__ >= 900 && < 1000,
    // SM100 kernel bodies by __CUDA_ARCH__ >= 1000, so both coexist safely.
    let cu_src = manifest_dir.join("src/deepgemm_wrapper.cu");
    let obj = out_dir.join("deepgemm_wrapper.o");

    let sm100 = nvcc_supports_sm100(&nvcc);
    if sm100 {
        println!("cargo:warning=DeepGEMM: compiling for SM90a + SM100a (fat binary)");
    } else {
        println!("cargo:warning=DeepGEMM: compiling for SM90a only (CUDA toolkit lacks SM100 support)");
    }

    let mut nvcc_cmd = Command::new(&nvcc);
    nvcc_cmd.args(&common_args)
        .arg("-gencode=arch=compute_90a,code=sm_90a");
    if sm100 {
        nvcc_cmd.arg("-gencode=arch=compute_100a,code=sm_100a");
    }
    let status = nvcc_cmd
        .args(&include_args)
        .args(["-c", cu_src.to_str().unwrap(), "-o", obj.to_str().unwrap()])
        .status()
        .expect("Failed to run nvcc");
    if !status.success() {
        panic!("nvcc compilation failed for deepgemm_wrapper.cu");
    }

    // 4. Create static archive via nvcc --lib (preserves CUDA fatbin sections)
    let lib = out_dir.join("libdeepgemm.a");
    let status = Command::new(&nvcc)
        .arg("--lib")
        .args(["-o", lib.to_str().unwrap()])
        .arg(&obj)
        .status()
        .expect("Failed to run nvcc --lib");
    if !status.success() {
        panic!("nvcc --lib failed");
    }

    // 5. Link
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=deepgemm");

    // CUDA runtime + driver
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
    if cuda_targets_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_lib.display());
    }
    // Stubs for CUDA driver API (cuTensorMapEncodeTiled)
    let cuda_stubs = cuda_path.join("lib64/stubs");
    if cuda_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_stubs.display());
    }
    let cuda_targets_stubs = cuda_path.join("targets/x86_64-linux/lib/stubs");
    if cuda_targets_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_stubs.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");   // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=dl");   // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=cuda"); // CUDA driver API (cuTensorMapEncodeTiled)
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn ensure_cutlass(out_dir: &Path) -> PathBuf {
    let cutlass_dir = out_dir.join("cutlass");

    if cutlass_dir.join("include/cutlass/cutlass.h").exists() {
        return cutlass_dir;
    }

    println!("cargo:warning=Cloning CUTLASS main branch (headers only)...");

    if cutlass_dir.exists() {
        std::fs::remove_dir_all(&cutlass_dir).ok();
    }

    let status = Command::new("git")
        .args([
            "clone", "--depth", "1", "--branch", CUTLASS_BRANCH,
            "--single-branch", "--no-checkout", CUTLASS_REPO,
        ])
        .arg(&cutlass_dir)
        .status()
        .expect("git clone failed");

    if !status.success() {
        panic!("Failed to clone CUTLASS repo");
    }

    Command::new("git")
        .args(["sparse-checkout", "init", "--cone"])
        .current_dir(&cutlass_dir)
        .status().ok();
    Command::new("git")
        .args(["sparse-checkout", "set", "include"])
        .current_dir(&cutlass_dir)
        .status().ok();
    Command::new("git")
        .args(["checkout"])
        .current_dir(&cutlass_dir)
        .status().ok();

    if !cutlass_dir.join("include/cutlass/cutlass.h").exists() {
        panic!("CUTLASS headers not found after clone");
    }

    println!("cargo:warning=CUTLASS headers ready");
    cutlass_dir
}

fn ensure_deepgemm(out_dir: &Path) -> PathBuf {
    let dg_dir = out_dir.join("deepgemm");

    if dg_dir.join("deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh").exists() {
        return dg_dir;
    }

    println!("cargo:warning=Cloning DeepGEMM (headers only)...");

    if dg_dir.exists() {
        std::fs::remove_dir_all(&dg_dir).ok();
    }

    let status = Command::new("git")
        .args([
            "clone", "--depth", "1", "--branch", DEEPGEMM_BRANCH,
            "--single-branch", "--no-checkout", DEEPGEMM_REPO,
        ])
        .arg(&dg_dir)
        .status()
        .expect("git clone failed");
    if !status.success() {
        panic!("Failed to clone DeepGEMM repo");
    }

    Command::new("git")
        .args(["sparse-checkout", "init", "--cone"])
        .current_dir(&dg_dir)
        .status().ok();
    Command::new("git")
        .args(["sparse-checkout", "set", "deep_gemm/include"])
        .current_dir(&dg_dir)
        .status().ok();
    Command::new("git")
        .args(["checkout"])
        .current_dir(&dg_dir)
        .status().ok();

    if !dg_dir.join("deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh").exists() {
        panic!("DeepGEMM headers not found after clone");
    }

    // Patch SM90 kernel arch guards for fat binary (SM90a + SM100a) compilation.
    // Upstream uses `__CUDA_ARCH__ >= 900` which includes SM100, but SM90 wgmma
    // instructions are invalid on SM100. Narrow to `>= 900 && < 1000`.
    for file in [
        "deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh",
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh",
    ] {
        let path = dg_dir.join(file);
        if let Ok(content) = std::fs::read_to_string(&path) {
            let patched = content.replace(
                "(defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))",
                "(defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000))",
            );
            if patched != content {
                std::fs::write(&path, patched).expect("Failed to patch DeepGEMM header");
                println!("cargo:warning=Patched {file} for fat binary arch guard");
            }
        }
    }

    println!("cargo:warning=DeepGEMM headers ready");
    dg_dir
}

fn find_cuda() -> PathBuf {
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        if Path::new(p).join("bin/nvcc").exists() {
            return PathBuf::from(p);
        }
    }
    panic!("CUDA toolkit not found. Set CUDA_PATH env var.");
}

fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    let output = Command::new(nvcc)
        .args(["--list-gpu-arch"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).contains("compute_100")
        }
        _ => false,
    }
}

