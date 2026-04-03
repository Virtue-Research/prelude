fn main() {
    // oneDNN FFI compilation moved to prelude-cpu crate.

    // ── NVTX profiling (link libnvToolsExt when feature is enabled) ──
    #[cfg(feature = "nvtx")]
    {
        let cuda_root = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
        println!("cargo:rustc-link-lib=dylib=nvToolsExt");
    }
}
