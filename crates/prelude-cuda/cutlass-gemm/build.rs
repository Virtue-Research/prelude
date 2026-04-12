//! Build script: compile CUTLASS 3.x BF16+F32 GEMM wrapper with nvcc.
//!
//! Two translation units land in the final `libcutlass_gemm.a`:
//!
//!   1. `cutlass_wrapper.cu` — CUTLASS 3.x template kernels. Fat-binary
//!      across SM80 (Ampere CollectiveMma) + SM90a (Hopper TMA
//!      CollectiveBuilder) + optionally SM100a (Blackwell) when nvcc
//!      supports it.
//!   2. `naive_gemm.cu` — simple reference kernels in a separate TU to
//!      keep CUTLASS template machinery from corrupting them.
//!
//! Requires the `third_party/cutlass` submodule and a CUDA toolkit with
//! nvcc. All the toolkit discovery / arch probing / submodule tracking /
//! CUDA runtime linking is centralized in `prelude_kernelbuild::nvcc`.

use std::path::PathBuf;
use std::process::Command;

use prelude_kernelbuild::build_log;
use prelude_kernelbuild::nvcc::{
    compile_cu_to_obj, find_cuda, link_cuda_runtime_static, locate_source, nvcc_path,
    nvcc_supports_sm100, track_submodule, ObjCompile,
};

fn main() {
    println!("cargo:rerun-if-changed=src/cutlass_wrapper.cu");
    println!("cargo:rerun-if-changed=src/naive_gemm.cu");
    println!("cargo:rerun-if-changed=build.rs");
    track_submodule("cutlass");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.join("../../..");

    // CUTLASS headers (third_party/cutlass submodule, with env override)
    let cutlass_dir = locate_source(
        "CUTLASS_ROOT",
        "cutlass",
        "include/cutlass/cutlass.h",
        &workspace_root.join("third_party/cutlass"),
    );

    let cuda_path = find_cuda();
    let nvcc = nvcc_path(&cuda_path);
    let sm100 = nvcc_supports_sm100(&nvcc);

    if sm100 {
        build_log!("compiling 3.x for SM80 + SM90a + SM100a (fat binary)");
    } else {
        build_log!("compiling 3.x for SM80 + SM90a");
    }

    let common_gencodes: Vec<String> = {
        let mut v = vec![
            "-gencode=arch=compute_80,code=sm_80".into(),
            "-gencode=arch=compute_90a,code=sm_90a".into(),
        ];
        if sm100 {
            v.push("-gencode=arch=compute_100a,code=sm_100a".into());
        }
        v
    };

    // ── TU 1: CUTLASS template wrapper ──────────────────────────────
    let wrapper_src = manifest_dir.join("src/cutlass_wrapper.cu");
    let wrapper_obj = out_dir.join("cutlass_wrapper.o");
    let mut wrapper_opts = ObjCompile::new(&wrapper_src, &wrapper_obj)
        .include(cutlass_dir.join("include"))
        .include(cutlass_dir.join("tools/util/include"))
        .include(cuda_path.join("include"))
        .cpp_std("-std=c++17");
    for g in &common_gencodes {
        wrapper_opts = wrapper_opts.gencode(g.clone());
    }
    compile_cu_to_obj(&nvcc, &wrapper_opts);

    // ── TU 2: Naive reference kernels ───────────────────────────────
    let naive_src = manifest_dir.join("src/naive_gemm.cu");
    let naive_obj = out_dir.join("naive_gemm.o");
    let mut naive_opts = ObjCompile::new(&naive_src, &naive_obj)
        .include(cuda_path.join("include"))
        .cpp_std("-std=c++17");
    for g in &common_gencodes {
        naive_opts = naive_opts.gencode(g.clone());
    }
    compile_cu_to_obj(&nvcc, &naive_opts);

    // ── Archive both objects into libcutlass_gemm.a ─────────────────
    let lib = out_dir.join("libcutlass_gemm.a");
    let status = Command::new(&nvcc)
        .arg("--lib")
        .args(["-o", lib.to_str().unwrap()])
        .arg(&wrapper_obj)
        .arg(&naive_obj)
        .status()
        .expect("Failed to run nvcc --lib");
    if !status.success() {
        panic!("nvcc --lib failed for libcutlass_gemm.a");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cutlass_gemm");
    link_cuda_runtime_static(&cuda_path);
}
