//! Build script: compile llama.cpp MMQ/MMVQ/dequantize kernels.
//!
//! Uses third_party/llama.cpp submodule for kernel headers.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Inlined workspace helper (keep this crate self-contained) ──────

macro_rules! build_log {
    ($($arg:tt)*) => {{
        let _msg = format!($($arg)*);
        eprintln!("  [{}] {_msg}", env!("CARGO_PKG_NAME"));
        println!("cargo:warning={}", _msg);
    }};
}

#[allow(dead_code)]
fn track_submodule(name: &str) {
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
    track_submodule("llama.cpp");

    // 1. Ensure llama.cpp headers (third_party/llama.cpp submodule)
    let workspace_root = PathBuf::from(&manifest_dir).join("../../..");
    let llama_dir = workspace_root.join("third_party/llama.cpp");
    if !llama_dir.join("ggml/src/ggml-cuda/common.cuh").exists() {
        panic!(
            "third_party/llama.cpp submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/llama.cpp"
        );
    }
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

    build_log!("compiling tiled MMQ for sm_{compute_cap}");

    // 4-6. Compile all sources in parallel (3 independent compilations)
    let obj = out_dir.join("mmq_ffi.o");
    let deq_obj = out_dir.join("dequantize_ffi.o");
    let deq_gpu_obj = out_dir.join("dequantize_gpu.o");

    let cc_compiler = env::var("CC").unwrap_or_else(|_| "gcc".to_string());

    std::thread::scope(|s| {
        // mmq_ffi.cu (CUDA)
        let h1 = s.spawn(|| {
            let status = Command::new(&nvcc)
                .args([
                    "-std=c++17", "-O3", "--use_fast_math",
                    "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
                    &gencode,
                    "-I", csrc.to_str().unwrap(),
                    "-I", ggml_include.to_str().unwrap(),
                    "-I", ggml_src.to_str().unwrap(),
                    "-I", ggml_cuda.to_str().unwrap(),
                    "-c", csrc.join("mmq_ffi.cu").to_str().unwrap(),
                    "-o", obj.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to run nvcc");
            assert!(status.success(), "nvcc compilation of mmq_ffi.cu failed");
        });

        // dequantize_ffi.c (pure CPU)
        let h2 = s.spawn(|| {
            let status = Command::new(&cc_compiler)
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
            assert!(status.success(), "gcc compilation of dequantize_ffi.c failed");
        });

        // dequantize_gpu.cu (CUDA)
        let h3 = s.spawn(|| {
            let status = Command::new(&nvcc)
                .args([
                    "-std=c++17", "-O3", "--use_fast_math",
                    "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
                    &gencode,
                    "-I", csrc.to_str().unwrap(),
                    "-c", csrc.join("dequantize_gpu.cu").to_str().unwrap(),
                    "-o", deq_gpu_obj.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to run nvcc for dequantize_gpu.cu");
            assert!(status.success(), "nvcc compilation of dequantize_gpu.cu failed");
        });

        h1.join().unwrap();
        h2.join().unwrap();
        h3.join().unwrap();
    });

    // 7. Create static archive (CUDA + CPU objects)
    let lib = out_dir.join("libquant_gemm.a");
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
    println!("cargo:rustc-link-lib=static=quant_gemm");

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
