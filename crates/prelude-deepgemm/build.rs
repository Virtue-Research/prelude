//! Build script: compile DeepGEMM BF16/FP8 GEMM wrapper with nvcc.
//!
//! Requires:
//! - CUDA Toolkit with nvcc (sm_90a support; sm_100a for Blackwell)
//! - CUTLASS headers (auto-cloned from GitHub)
//! - Vendored DeepGEMM headers in vendor/deep_gemm/

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_BRANCH: &str = "main";

fn main() {
    println!("cargo:rerun-if-changed=src/deepgemm_wrapper.cu");
    println!("cargo:rerun-if-changed=src/deepgemm_sm100.cu");
    println!("cargo:rerun-if-changed=vendor/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // 1. Ensure CUTLASS headers (needed by DeepGEMM kernel)
    let cutlass_dir = ensure_cutlass(&out_dir);
    let cutlass_include = cutlass_dir.join("include");

    // 2. Find CUDA toolkit
    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }

    // 3. Compile deepgemm_wrapper.cu (SM90a)
    let vendor_include = manifest_dir.join("vendor");
    let common_args = [
        "-std=c++20", "-O3", "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
    ];
    let include_args = [
        "-I", cutlass_include.to_str().unwrap(),
        "-I", vendor_include.to_str().unwrap(),
        "-I", &format!("{}/include", cuda_path.display()),
    ];

    // SM90 wrapper always compiles for compute_90a (H100/H200).
    // SM100 wrapper always compiles for compute_100a (Blackwell) in a separate unit.
    // No GPU detection needed — both are compiled unconditionally if toolkit supports them.
    println!("cargo:warning=DeepGEMM: compiling SM90a kernels");

    let cu_src = manifest_dir.join("src/deepgemm_wrapper.cu");
    let obj_sm90 = out_dir.join("deepgemm_wrapper.o");
    let status = Command::new(&nvcc)
        .args(&common_args)
        .arg("-gencode=arch=compute_90a,code=sm_90a")
        .args(&include_args)
        .args(["-c", cu_src.to_str().unwrap(), "-o", obj_sm90.to_str().unwrap()])
        .status()
        .expect("Failed to run nvcc");
    if !status.success() {
        panic!("nvcc compilation failed for deepgemm_wrapper.cu");
    }

    // 4. Compile deepgemm_sm100.cu (SM100a) — separate compilation unit
    //    SM100 kernels use Blackwell-specific instructions incompatible with SM90.
    let cu_sm100 = manifest_dir.join("src/deepgemm_sm100.cu");
    let obj_sm100 = out_dir.join("deepgemm_sm100.o");
    let sm100_ok = if nvcc_supports_sm100(&nvcc) {
        println!("cargo:warning=DeepGEMM: compiling SM100 (Blackwell) support");
        let status = Command::new(&nvcc)
            .args(&common_args)
            .arg("-gencode=arch=compute_100a,code=sm_100a")
            .args(&include_args)
            .args(["-c", cu_sm100.to_str().unwrap(), "-o", obj_sm100.to_str().unwrap()])
            .status()
            .expect("Failed to run nvcc for SM100");
        status.success()
    } else {
        println!("cargo:warning=DeepGEMM: SM100 not supported by this CUDA toolkit, skipping");
        false
    };

    // 5. Create static archive via nvcc --lib (preserves CUDA fatbin sections)
    let lib = out_dir.join("libdeepgemm.a");
    let mut lib_cmd = Command::new(&nvcc);
    lib_cmd.arg("--lib").args(["-o", lib.to_str().unwrap()]).arg(&obj_sm90);
    if sm100_ok {
        lib_cmd.arg(&obj_sm100);
    }
    let status = lib_cmd.status().expect("Failed to run nvcc --lib");
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

