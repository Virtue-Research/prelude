//! Build script: compile CUTLASS 3.x BF16+F32 GEMM wrapper with nvcc.
//!
//! Requires:
//! - CUDA Toolkit with nvcc (SM80+ support)
//! - CUTLASS headers (auto-cloned from GitHub)

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_BRANCH: &str = "main";

fn main() {
    println!("cargo:rerun-if-changed=src/cutlass_wrapper.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // 1. Ensure CUTLASS headers
    let cutlass_dir = ensure_cutlass(&out_dir);
    let cutlass_include = cutlass_dir.join("include");

    // 2. Find CUDA toolkit
    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }

    // 3. Compile cutlass_wrapper.cu
    // SM90 native for 3.x CollectiveBuilder (TMA + warp-specialized)
    // SM80 for 3.x manual CollectiveMma (cp.async + TensorOp)
    let cu_src = manifest_dir.join("src/cutlass_wrapper.cu");
    let obj = out_dir.join("cutlass_wrapper.o");

    println!("cargo:warning=CUTLASS GEMM: compiling 3.x for SM90a + SM80");

    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            "-Xcompiler", "-fPIC",
            // SM90a: CUTLASS 3.x CollectiveBuilder (TMA + warp-specialized)
            "-gencode=arch=compute_90a,code=sm_90a",
            // SM80: CUTLASS 3.x manual CollectiveMma (Ampere TensorOp)
            "-gencode=arch=compute_80,code=sm_80",
            "-I", cutlass_include.to_str().unwrap(),
            "-I", cutlass_dir.join("tools/util/include").to_str().unwrap(),
            "-I", &format!("{}/include", cuda_path.display()),
            "-c", cu_src.to_str().unwrap(),
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("nvcc compilation failed for cutlass_wrapper.cu");
    }

    // 4. Create static archive
    let lib = out_dir.join("libcutlass_gemm.a");
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
    println!("cargo:rustc-link-lib=static=cutlass_gemm");

    // CUDA runtime (static) + driver (dynamic)
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
    if cuda_targets_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_lib.display());
    }
    // Static cudart — no libcudart.so dependency at runtime
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");  // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=dl");  // required by cudart_static
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
        .args(["sparse-checkout", "set", "include", "tools/util/include"])
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
