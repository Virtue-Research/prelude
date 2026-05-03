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
    nvcc_supports_sm100, nvcc_supports_sm103, track_submodule, ObjCompile,
};

fn main() {
    println!("cargo:rerun-if-changed=src/cutlass_wrapper.cu");
    println!("cargo:rerun-if-changed=src/naive_gemm.cu");
    println!("cargo:rerun-if-changed=src/grouped_moe_sm100.cu");
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
    let sm103 = nvcc_supports_sm103(&nvcc);

    match (sm100, sm103) {
        (true, true) => build_log!("compiling 3.x for SM80 + SM90a + SM100a + SM103a (fat binary)"),
        (true, false) => build_log!("compiling 3.x for SM80 + SM90a + SM100a (fat binary)"),
        _ => build_log!("compiling 3.x for SM80 + SM90a"),
    }

    let common_gencodes: Vec<String> = {
        let mut v = vec![
            "-gencode=arch=compute_80,code=sm_80".into(),
            "-gencode=arch=compute_90a,code=sm_90a".into(),
        ];
        if sm100 {
            v.push("-gencode=arch=compute_100a,code=sm_100a".into());
        }
        if sm103 {
            v.push("-gencode=arch=compute_103a,code=sm_103a".into());
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

    // ── TU 3: SM100 grouped GEMM for MoE (Blackwell only) ──────────
    let grouped_obj = if sm100 {
        let grouped_src = manifest_dir.join("src/grouped_moe_sm100.cu");
        let grouped_obj = out_dir.join("grouped_moe_sm100.o");
        let mut grouped_opts = ObjCompile::new(&grouped_src, &grouped_obj)
            .include(cutlass_dir.join("include"))
            .include(cutlass_dir.join("tools/util/include"))
            .include(cuda_path.join("include"))
            .cpp_std("-std=c++17");
        // Only emit SM100/SM103 cubins — the kernel uses tcgen05/UMMA
        // and won't compile against earlier arches.
        grouped_opts = grouped_opts.gencode("-gencode=arch=compute_100a,code=sm_100a");
        if sm103 {
            grouped_opts = grouped_opts.gencode("-gencode=arch=compute_103a,code=sm_103a");
        }
        compile_cu_to_obj(&nvcc, &grouped_opts);
        Some(grouped_obj)
    } else {
        None
    };

    // ── Archive objects into libcutlass_gemm.a ─────────────────────
    let lib = out_dir.join("libcutlass_gemm.a");
    let mut nvcc_lib = Command::new(&nvcc);
    nvcc_lib.arg("--lib")
        .args(["-o", lib.to_str().unwrap()])
        .arg(&wrapper_obj)
        .arg(&naive_obj);
    if let Some(ref obj) = grouped_obj {
        nvcc_lib.arg(obj);
    }
    let status = nvcc_lib.status().expect("Failed to run nvcc --lib");
    if !status.success() {
        panic!("nvcc --lib failed for libcutlass_gemm.a");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cutlass_gemm");
    link_cuda_runtime_static(&cuda_path);
}
