//! Build script: compile CUTLASS 3.x BF16+F32 GEMM wrapper with nvcc.
//!
//! Requires:
//! - CUDA Toolkit with nvcc (SM80+ support)
//! - CUTLASS headers (third_party/cutlass submodule)

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

include!("../../../build_log.rs");

fn main() {
    println!("cargo:rerun-if-changed=src/cutlass_wrapper.cu");
    println!("cargo:rerun-if-changed=src/naive_gemm.cu");
    track_submodule("cutlass");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // 1. Ensure CUTLASS headers (third_party/cutlass submodule)
    let workspace_root = PathBuf::from(&manifest_dir).join("../../..");
    let cutlass_dir = workspace_root.join("third_party/cutlass");
    if !cutlass_dir.join("include/cutlass/cutlass.h").exists() {
        panic!(
            "third_party/cutlass submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/cutlass"
        );
    }
    let cutlass_include = cutlass_dir.join("include");

    // 2. Find CUDA toolkit
    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }

    // 3. Compile cutlass_wrapper.cu (CUTLASS template kernels)
    // SM90 native for 3.x CollectiveBuilder (TMA + warp-specialized)
    // SM80 for 3.x manual CollectiveMma (cp.async + TensorOp)
    let cu_src = manifest_dir.join("src/cutlass_wrapper.cu");
    let obj = out_dir.join("cutlass_wrapper.o");

    let sm100 = nvcc_supports_sm100(&nvcc);
    let sm103 = nvcc_supports_sm103(&nvcc);
    if sm103 {
        build_log!("compiling 3.x for SM80 + SM90a + SM100a + SM103a (fat binary)");
    } else if sm100 {
        build_log!("compiling 3.x for SM80 + SM90a + SM100a (fat binary)");
    } else {
        build_log!("compiling 3.x for SM80 + SM90a");
    }

    let mut nvcc_cmd = Command::new(&nvcc);
    nvcc_cmd.args([
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "-Xcompiler", "-fPIC",
        // SM80: CUTLASS 3.x manual CollectiveMma (Ampere TensorOp)
        "-gencode=arch=compute_80,code=sm_80",
        // SM90a: CUTLASS 3.x CollectiveBuilder (TMA + warp-specialized)
        "-gencode=arch=compute_90a,code=sm_90a",
    ]);
    if sm100 {
        nvcc_cmd.arg("-gencode=arch=compute_100a,code=sm_100a");
    }
    if sm103 {
        // B300 is SM103, sm_100a cubin doesn't run on it and "a" PTX is not
        // forward-compatible across the Blackwell family — need a native
        // sm_103a cubin. See prelude-deepgemm build.rs for the same fix.
        nvcc_cmd.arg("-gencode=arch=compute_103a,code=sm_103a");
    }
    let status = nvcc_cmd
        .args([
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

    // 3b. Compile naive_gemm.cu — separate TU, no CUTLASS headers.
    // Isolated to avoid CUTLASS template machinery corrupting simple kernels.
    let naive_src = manifest_dir.join("src/naive_gemm.cu");
    let naive_obj = out_dir.join("naive_gemm.o");

    let mut naive_cmd = Command::new(&nvcc);
    naive_cmd.args([
        "-std=c++17",
        "-O3",
        "-Xcompiler", "-fPIC",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90a,code=sm_90a",
    ]);
    if sm100 {
        naive_cmd.arg("-gencode=arch=compute_100a,code=sm_100a");
    }
    if sm103 {
        naive_cmd.arg("-gencode=arch=compute_103a,code=sm_103a");
    }
    let status = naive_cmd
        .args([
            "-I", &format!("{}/include", cuda_path.display()),
            "-c", naive_src.to_str().unwrap(),
            "-o", naive_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run nvcc for naive_gemm.cu");

    if !status.success() {
        panic!("nvcc compilation failed for naive_gemm.cu");
    }

    // 4. Create static archive from both object files
    let lib = out_dir.join("libcutlass_gemm.a");
    let status = Command::new(&nvcc)
        .arg("--lib")
        .args(["-o", lib.to_str().unwrap()])
        .arg(&obj)
        .arg(&naive_obj)
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

fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("compute_100"))
        .unwrap_or(false)
}

fn nvcc_supports_sm103(nvcc: &Path) -> bool {
    Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("compute_103"))
        .unwrap_or(false)
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
