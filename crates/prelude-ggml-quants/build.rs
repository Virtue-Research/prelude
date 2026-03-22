use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let llama_cpp_dir = env::var("LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            let workspace_root = manifest.parent().unwrap().parent().unwrap();
            workspace_root.join("../llama.cpp")
        });

    if !llama_cpp_dir.join("include/llama.h").exists() {
        println!(
            "cargo:warning=llama.cpp not found at {}. Set LLAMA_CPP_DIR env var.",
            llama_cpp_dir.display()
        );
        println!("cargo:warning=Building stub only — GGUF inference will panic at runtime.");
        cc::Build::new()
            .file("csrc/stub.c")
            .compile("llama_ffi");
        return;
    }

    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-changed=csrc/llama_ffi.c");
    println!("cargo:rerun-if-changed=csrc/stub.c");

    let build_dir = llama_cpp_dir.join("build-prelude");

    // CPU-only build — GPU GGUF inference uses FlashInfer/FA4 via dequant, not llama.cpp CUDA.
    // Build llama.cpp as static libraries via cmake
    if !build_dir.join("src/libllama.a").exists() {
        println!("cargo:warning=Building llama.cpp static libraries (CPU-only, first time)...");

        let cmake_args = vec![
            "-B", build_dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_CPU=ON",
            "-DGGML_NATIVE=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_METAL=OFF",
            "-DGGML_VULKAN=OFF",
            "-DGGML_SYCL=OFF",
            "-DGGML_OPENMP=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ];

        let status = Command::new("cmake")
            .current_dir(&llama_cpp_dir)
            .args(cmake_args)
            .status()
            .expect("failed to run cmake configure");
        assert!(status.success(), "cmake configure failed");

        // Only build the llama target (includes ggml)
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "8".to_string());

        let status = Command::new("cmake")
            .args(["--build", build_dir.to_str().unwrap(), "-j", &nproc, "--target", "llama"])
            .status()
            .expect("failed to run cmake build");
        assert!(status.success(), "cmake build failed");
    }

    // Compile our thin C wrapper (uses llama.h)
    cc::Build::new()
        .file("csrc/llama_ffi.c")
        .include(llama_cpp_dir.join("include"))
        .include(llama_cpp_dir.join("ggml/include"))
        .opt_level(2)
        .warnings(false)
        .compile("llama_ffi");

    // Link static libraries
    let lib_dir = build_dir.join("src");
    let ggml_dir = build_dir.join("ggml/src");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", ggml_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml");

    // System libraries required by ggml
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");  // OpenMP
    println!("cargo:rustc-link-lib=dylib=m");

    // AMX backend (if built)
    let amx_lib = ggml_dir.join("libggml-amx.a");
    if amx_lib.exists() {
        println!("cargo:rustc-link-lib=static=ggml-amx");
    }

}
