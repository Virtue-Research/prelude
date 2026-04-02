//! Build script: compile llama.cpp MMQ/MMVQ/dequantize kernels.
//!
//! Auto-clones llama.cpp main branch for kernel headers (same pattern as prelude-cutlass-gemm).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp.git";
const LLAMA_CPP_BRANCH: &str = "master";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest_dir.join("csrc");

    // Rerun triggers (our FFI glue files)
    println!("cargo:rerun-if-changed=csrc/mmq_ffi.cu");
    println!("cargo:rerun-if-changed=csrc/mmq_ffi.h");
    println!("cargo:rerun-if-changed=csrc/dequantize_ffi.c");
    println!("cargo:rerun-if-changed=csrc/dequantize_gpu.cu");
    println!("cargo:rerun-if-changed=csrc/iq_tables.cuh");

    // 1. Ensure llama.cpp headers
    let llama_dir = ensure_llama_cpp(&out_dir);
    let ggml_include = llama_dir.join("ggml/include");
    let ggml_src = llama_dir.join("ggml/src");
    let ggml_cuda = ggml_src.join("ggml-cuda");

    // 2. Find CUDA
    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }

    // 3. Detect GPU compute capability
    let compute_cap = detect_compute_cap().unwrap_or(80);
    let gencode = format!(
        "-gencode=arch=compute_{compute_cap},code=sm_{compute_cap}"
    );

    println!(
        "cargo:warning=prelude-quant-gemm: compiling tiled MMQ for sm_{compute_cap}"
    );

    // 4. Compile mmq_ffi.cu → mmq_ffi.o
    //    Include paths match upstream llama.cpp CMake:
    //      -I ggml/include       (ggml.h, ggml-backend.h, ggml-cuda.h)
    //      -I ggml/src           (ggml-common.h, ggml-impl.h)
    //      -I ggml/src/ggml-cuda (common.cuh, mmq.cuh, vecdotq.cuh, vendors/)
    let obj = out_dir.join("mmq_ffi.o");
    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-Xcompiler", "-fPIC",
            &gencode,
            "-I", csrc.to_str().unwrap(),
            "-I", ggml_include.to_str().unwrap(),
            "-I", ggml_src.to_str().unwrap(),
            "-I", ggml_cuda.to_str().unwrap(),
            "-c",
            csrc.join("mmq_ffi.cu").to_str().unwrap(),
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("nvcc compilation of mmq_ffi.cu failed");
    }

    // 5. Compile dequantize_ffi.c (pure CPU, for correctness testing)
    let deq_obj = out_dir.join("dequantize_ffi.o");
    let cc = env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let status = Command::new(&cc)
        .args([
            "-c", "-O2", "-fPIC",
            "-I", csrc.to_str().unwrap(),
            "-I", ggml_include.to_str().unwrap(),
            "-I", ggml_src.to_str().unwrap(),
            csrc.join("dequantize_ffi.c").to_str().unwrap(),
            "-o", deq_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to compile dequantize_ffi.c");

    if !status.success() {
        panic!("gcc compilation of dequantize_ffi.c failed");
    }

    // 6. Compile dequantize_gpu.cu → dequantize_gpu.o (GPU dequant kernels)
    let deq_gpu_obj = out_dir.join("dequantize_gpu.o");
    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-Xcompiler", "-fPIC",
            &gencode,
            "-I", csrc.to_str().unwrap(),
            "-c",
            csrc.join("dequantize_gpu.cu").to_str().unwrap(),
            "-o", deq_gpu_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run nvcc for dequantize_gpu.cu");

    if !status.success() {
        panic!("nvcc compilation of dequantize_gpu.cu failed");
    }

    // 7. Create static archive (CUDA + CPU objects)
    let lib = out_dir.join("libprelude_quant_gemm.a");
    let status = Command::new(&nvcc)
        .args(["--lib", "-o", lib.to_str().unwrap()])
        .arg(&obj)
        .arg(&deq_obj)
        .arg(&deq_gpu_obj)
        .status()
        .expect("Failed to run nvcc --lib");

    if !status.success() {
        panic!("nvcc --lib failed");
    }

    // 8. Link
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=prelude_quant_gemm");

    // CUDA runtime (static) — no libcudart.so needed at runtime
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
    if cuda_targets_lib.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_targets_lib.display()
        );
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn ensure_llama_cpp(out_dir: &Path) -> PathBuf {
    let llama_dir = out_dir.join("llama.cpp");

    // Check if already cloned with kernel headers
    if llama_dir.join("ggml/src/ggml-cuda/common.cuh").exists() {
        return llama_dir;
    }

    println!("cargo:warning=Cloning llama.cpp main branch (CUDA kernels only)...");

    if llama_dir.exists() {
        std::fs::remove_dir_all(&llama_dir).ok();
    }

    let status = Command::new("git")
        .args([
            "clone", "--depth", "1", "--branch", LLAMA_CPP_BRANCH,
            "--single-branch", "--no-checkout", LLAMA_CPP_REPO,
        ])
        .arg(&llama_dir)
        .status()
        .expect("git clone failed");

    if !status.success() {
        panic!("Failed to clone llama.cpp repo");
    }

    // Sparse checkout: only ggml headers + CUDA kernel sources
    Command::new("git")
        .args(["sparse-checkout", "init", "--cone"])
        .current_dir(&llama_dir)
        .status().ok();
    Command::new("git")
        .args(["sparse-checkout", "set", "ggml/include", "ggml/src/ggml-common.h", "ggml/src/ggml-impl.h", "ggml/src/ggml-cuda"])
        .current_dir(&llama_dir)
        .status().ok();
    Command::new("git")
        .args(["checkout"])
        .current_dir(&llama_dir)
        .status().ok();

    if !llama_dir.join("ggml/src/ggml-cuda/common.cuh").exists() {
        panic!("llama.cpp CUDA headers not found after clone");
    }

    println!("cargo:warning=llama.cpp CUDA kernel headers ready");
    llama_dir
}

fn find_cuda() -> PathBuf {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(p) = env::var(var) {
            return PathBuf::from(p);
        }
    }
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        if Path::new(p).join("bin/nvcc").exists() {
            return PathBuf::from(p);
        }
    }
    panic!("CUDA toolkit not found. Set CUDA_HOME or CUDA_PATH.");
}

fn detect_compute_cap() -> Option<u32> {
    if let Ok(arch_list) = env::var("CUDA_ARCH_LIST") {
        if let Some(cap) = arch_list
            .split(',')
            .filter_map(|s| s.trim().replace('.', "").parse::<u32>().ok())
            .max()
        {
            return Some(cap);
        }
    }
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .replace('.', "")
        .parse::<u32>()
        .ok()
}
