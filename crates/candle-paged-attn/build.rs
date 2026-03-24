use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/pagedattention.cu");
    println!("cargo:rerun-if-changed=kernels/reshape_and_cache_kernel.cu");

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());

    let mut builder = bindgen_cuda::Builder::default()
        .kernel_paths_glob("kernels/*.cu")
        .include_paths_glob("kernels")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math");

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    builder.build_lib(build_dir.join("libpagedattention.a"));

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
