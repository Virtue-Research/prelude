//! Build script for `prelude-cuda`: compiles the custom in-tree CUDA
//! kernels under `src/kernels/kernels_src/` and the standalone MOE WMMA
//! static lib. CUDA toolkit discovery, compute-cap detection, and nvcc
//! invocation all live in `prelude_kernelbuild::nvcc`.
//!
//! Two output artifacts:
//!
//!   1. A set of `.ptx` files (one per kernel) placed in `OUT_DIR`, loaded
//!      at runtime via cudarc's `get_or_load_custom_func`. These are
//!      framework-agnostic — each is a single `.cu → .ptx` via nvcc.
//!
//!   2. `libmoe_wmma.a` — a static lib built from `candle/moe_wmma.cu`
//!      that gets whole-archive linked into prelude-cuda so the WMMA
//!      intrinsics are callable through FFI. This one needs `-fPIC` and
//!      links `libcudart.so` dynamically (we already depend on cudarc's
//!      dylib lookup for all the PTX kernels anyway).

use std::path::PathBuf;
use std::sync::Arc;

use prelude_kernelbuild::nvcc::{
    ObjCompile, PtxCompile, compile_cu_to_obj, compile_cu_to_ptx, detect_compute_cap, find_cuda,
    link_cublas_dynamic, link_cuda_runtime_dynamic, nvcc_path,
};

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = PathBuf::from(&manifest_dir).join("src/kernels/kernels_src");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let cuda_root = find_cuda();
    let nvcc = nvcc_path(&cuda_root);

    // Default to sm_80 when no GPU is visible (cross-compile on CI).
    let compute_cap = detect_compute_cap().unwrap_or(80);

    // Define kernel modules: (category, filename, output_name)
    let kernel_modules = [
        // Our own optimized kernels
        ("elementwise", "add.cu", "add"),
        ("elementwise", "silu_mul.cu", "silu_mul"),
        ("fp8", "quantize.cu", "fp8_quantize"),
        ("normalization", "rmsnorm.cu", "rmsnorm"),
        ("normalization", "rmsnorm_gated.cu", "rmsnorm_gated"),
        ("rope", "qknorm_rope.cu", "qknorm_rope"),
        ("moe", "routing.cu", "moe_routing"),
        ("moe", "shared_expert_gate.cu", "shared_expert_gate"),
        ("moe", "gateup.cu", "moe_gateup"),
        ("moe", "down.cu", "moe_down"),
        ("kvcache", "append.cu", "kv_append"),
        ("kvcache", "knorm_rope_kv_write.cu", "knorm_rope_kv_write"),
        ("kvcache", "scatter_kv_cache.cu", "scatter_kv_cache"),
        ("gdn", "post_conv.cu", "gdn_post_conv"),
        ("gdn", "prefill_recurrent.cu", "gdn_prefill_recurrent"),
        ("logprobs", "gather_log_softmax.cu", "gather_log_softmax"),
        ("sample", "fast_argmax_vocab.cu", "fast_argmax_vocab"),
        ("sample", "check_eos.cu", "check_eos"),
    ];

    // Track common headers for rerun-if-changed
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/common/common.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/common/vec_utils.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/candle/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/candle/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/candle/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=build.rs");

    for &(category, filename, _) in kernel_modules.iter() {
        let kernel_src = kernels_dir.join(category).join(filename);
        println!("cargo:rerun-if-changed={}", kernel_src.display());
    }

    // ── Phase 1: PTX kernels (parallel nvcc --ptx invocations) ──────
    let nvcc = Arc::new(nvcc);
    let kernels_dir_arc = Arc::new(kernels_dir.clone());
    let out_dir_arc = Arc::new(out_dir.clone());

    let handles: Vec<_> = kernel_modules
        .iter()
        .map(|&(category, filename, output_name)| {
            let nvcc = nvcc.clone();
            let kernels_dir = kernels_dir_arc.clone();
            let out_dir = out_dir_arc.clone();
            std::thread::spawn(move || {
                let src = kernels_dir.join(category).join(filename);
                let ptx = out_dir.join(format!("{output_name}.ptx"));

                // `common/` headers live one level above the per-category
                // subdir (e.g. `elementwise/add.cu` includes
                // `common/common.cuh`). Pass the kernels root as -I so the
                // relative include resolves.
                let opts =
                    PtxCompile::new(&src, &ptx, compute_cap).include(kernels_dir.as_ref().clone());

                compile_cu_to_ptx(&nvcc, &opts);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("nvcc PTX compilation thread panicked");
    }

    // ── Phase 2: MOE WMMA + GEMV static lib (FFI, not PTX) ──────────
    let moe_wmma_src = kernels_dir.join("candle").join("moe_wmma.cu");
    let moe_gemv_src = kernels_dir.join("candle").join("moe_gemv.cu");
    if moe_wmma_src.exists() {
        println!("cargo:rerun-if-changed={}", moe_wmma_src.display());

        let moe_wmma_obj = out_dir.join("moe_wmma.o");
        let mut opts = ObjCompile::new(&moe_wmma_src, &moe_wmma_obj)
            .include(kernels_dir.join("candle"))
            .gencode(format!("-arch=sm_{compute_cap}"))
            .cpp_std("-std=c++17");
        if compute_cap < 80 {
            opts = opts.define("-DNO_BF16_KERNEL");
        }
        compile_cu_to_obj(&nvcc, &opts);

        let mut obj_paths = vec![moe_wmma_obj];

        if moe_gemv_src.exists() {
            println!("cargo:rerun-if-changed={}", moe_gemv_src.display());
            let moe_gemv_obj = out_dir.join("moe_gemv.o");
            let mut gemv_opts = ObjCompile::new(&moe_gemv_src, &moe_gemv_obj)
                .include(kernels_dir.join("candle"))
                .gencode(format!("-arch=sm_{compute_cap}"))
                .cpp_std("-std=c++17");
            if compute_cap < 80 {
                gemv_opts = gemv_opts.define("-DNO_BF16_KERNEL");
            }
            compile_cu_to_obj(&nvcc, &gemv_opts);
            obj_paths.push(moe_gemv_obj);
        }

        // Archive both objects into libmoe_wmma.a so the linker
        // treats it as a normal static lib dependency.
        let moe_lib = out_dir.join("libmoe_wmma.a");
        let mut ar_args: Vec<String> =
            vec!["rcs".to_string(), moe_lib.to_str().unwrap().to_string()];
        for o in &obj_paths {
            ar_args.push(o.to_str().unwrap().to_string());
        }
        let status = std::process::Command::new("ar")
            .args(&ar_args)
            .status()
            .unwrap_or_else(|e| panic!("Failed to create libmoe_wmma.a: {e}"));
        assert!(status.success(), "ar failed for moe_wmma");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=moe_wmma");
    }

    // The Rust GEMM fallback owns raw CUDA runtime + cuBLAS FFI bindings.
    // Dynamic links are intentional: cudarc already requires CUDA shared
    // libraries at runtime, and this keeps the final server link explicit.
    link_cuda_runtime_dynamic(&cuda_root);
    link_cublas_dynamic(&cuda_root);

    // ── fa3-0102: prebuilt vendored vLLM 0.22 FA3 hopper kernel ─────
    // Static lib produced out-of-band by candle-fa3-0102's
    // _build_v3_kernel_prelude.sh (CUDA 13.2 + CUTLASS 3.8, sm90a).
    // Built automatically here on first use if missing (~3 min).
    if std::env::var("CARGO_FEATURE_FA3_0102").is_ok() {
        println!("cargo:rerun-if-env-changed=FA3_0102_PRELUDE_PREBUILT_DIR");
        println!("cargo:rerun-if-env-changed=CUDA_HOME_FA3");
        println!("cargo:rerun-if-env-changed=CUTLASS38");
        let lib_dir = std::env::var("FA3_0102_PRELUDE_PREBUILT_DIR")
            .unwrap_or_else(|_| "/data/xueying/fa3_0102_prelude_prebuilt".to_string());
        let lib = PathBuf::from(&lib_dir).join("libprelude_fa3_0102.a");
        if !lib.exists() {
            // The fa3-0102 kernel needs CUDA 13.2 + CUTLASS 3.8 to compile (or a
            // prebuilt archive). These are NOT available on a stock CUDA box or in
            // CI, so before shelling out we preflight the toolchain and fail with
            // an actionable message instead of a cryptic "script failed" panic.
            // Override paths via CUDA_HOME_FA3 / CUTLASS38 / FA3_0102_PRELUDE_PREBUILT_DIR.
            let cuda_home = std::env::var("CUDA_HOME_FA3")
                .unwrap_or_else(|_| "/usr/local/cuda-13.2".to_string());
            let cutlass = std::env::var("CUTLASS38")
                .unwrap_or_else(|_| "/data/xueying/cutlass38/include".to_string());
            let missing: Vec<&str> = [
                (PathBuf::from(&cuda_home).join("bin/nvcc").exists(), "CUDA 13.2 toolkit (set CUDA_HOME_FA3)"),
                (PathBuf::from(&cutlass).join("cutlass/cutlass.h").exists(), "CUTLASS 3.8 headers (set CUTLASS38)"),
            ]
            .iter()
            .filter_map(|(ok, name)| if *ok { None } else { Some(*name) })
            .collect();
            assert!(
                missing.is_empty(),
                "feature `fa3-0102` is enabled but the kernel cannot be built here.\n\
                 Missing build prerequisites: {missing:?}\n\
                 Provide a prebuilt archive via FA3_0102_PRELUDE_PREBUILT_DIR=<dir containing libprelude_fa3_0102.a>,\n\
                 or set CUDA_HOME_FA3 (CUDA 13.2) and CUTLASS38 (CUTLASS 3.8 include dir) so it can compile.\n\
                 If you do not need this backend, build without `--features fa3-0102`."
            );
            let script = PathBuf::from(&manifest_dir)
                .join("../candle-fa3-0102/_build_v3_kernel_prelude.sh");
            let status = std::process::Command::new("bash")
                .arg(&script)
                .status()
                .unwrap_or_else(|e| panic!("fa3-0102 kernel build failed to start: {e}"));
            assert!(
                status.success() && lib.exists(),
                "fa3-0102 prebuilt lib missing and {} failed (expected {})",
                script.display(),
                lib.display()
            );
        }
        // Track the prebuilt archive so out-of-band kernel rebuilds force a relink.
        println!("cargo:rerun-if-changed={}", lib.display());
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=static=prelude_fa3_0102");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // ── Phase 3: Link cuda_dialect_runtime_static.a ─────────────────
    //
    // The MLIR-generated .o files in cuLA/FA4's DSL kernel archives
    // reference `_cudaGetDevice`, `cuda_dialect_init_library_once`, etc.
    // These are provided by `libcuda_dialect_runtime_static.a` which
    // ships inside the `nvidia_cutlass_dsl` Python package.
    //
    // This MUST happen in prelude-cuda's build.rs (not tvm-static-ffi's)
    // because prelude-cuda builds AFTER cuLA and FA4, so their Python
    // venvs (where the .a lives) are guaranteed to exist. tvm-static-ffi
    // builds BEFORE them and would race on a fresh clone.
    //
    // Search both the cuLA venv and the FA4 venv — whichever exists
    // first wins (they install the same cutlass-dsl version).
    link_cuda_dialect_runtime(&out_dir, std::path::Path::new(&manifest_dir));
}

fn link_cuda_dialect_runtime(_out_dir: &std::path::Path, manifest_dir: &std::path::Path) {
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Candidate venv locations — cuLA's venv is inside its OUT_DIR,
    // FA4's is at a stable workspace-level path.
    let fa4_venv = workspace_root.join("target/fa4-venv");

    // Also scan cuLA's OUT_DIR-based venv. We can't know the exact
    // hash Cargo picks, but we can walk the build dir.
    let candidates: Vec<std::path::PathBuf> = vec![fa4_venv];

    // Also look in cuLA build dirs
    let cula_build_dir = workspace_root.join("target");
    for profile in ["debug", "release", "dist"] {
        let build_dir = cula_build_dir.join(profile).join("build");
        if let Ok(entries) = std::fs::read_dir(&build_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("cula-") || name_str.starts_with("prelude-cula-") {
                    let venv = entry.path().join("out/cula-venv");
                    if venv.exists() {
                        let lib = venv.join("lib/python3.12/site-packages/nvidia_cutlass_dsl/lib");
                        if lib.join("libcuda_dialect_runtime_static.a").exists() {
                            println!("cargo:rustc-link-search=native={}", lib.display());
                            println!(
                                "cargo:rustc-link-lib=static:+whole-archive=cuda_dialect_runtime_static"
                            );
                            return;
                        }
                    }
                }
            }
        }
    }

    // Check FA4 venv
    for candidate in &candidates {
        if let Some(found) = find_file_recursive(candidate, "libcuda_dialect_runtime_static.a") {
            println!(
                "cargo:rustc-link-search=native={}",
                found.parent().unwrap().display()
            );
            println!("cargo:rustc-link-lib=static:+whole-archive=cuda_dialect_runtime_static");
            return;
        }
    }

    // Not found — warn but don't fail. CPU-only builds won't have it.
    eprintln!(
        "  [prelude-cuda] WARNING: libcuda_dialect_runtime_static.a not found; \
               dist/LTO builds with CuTeDSL kernels may fail to link"
    );
}

fn find_file_recursive(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
    }
    None
}
