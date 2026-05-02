fn main() {
    // oneDNN FFI compilation moved to prelude-cpu crate.

    // ── NVTX profiling (link libnvToolsExt when feature is enabled) ──
    //
    // CUDA 13 dropped the standalone libnvToolsExt.so from the toolkit
    // (only header-only nvtx3 ships now), so we have to find a copy from
    // a pip-installed nvidia-nvtx package as a fallback. PRELUDE_NVTX_LIB
    // can override the search path.
    #[cfg(feature = "nvtx")]
    {
        let mut search_dirs: Vec<String> = Vec::new();

        if let Ok(p) = std::env::var("PRELUDE_NVTX_LIB") {
            search_dirs.push(p);
        }
        let cuda_root = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        search_dirs.push(format!("{}/lib64", cuda_root));

        // Probe common pip-package install paths.
        for base in [
            std::env::var("HOME").unwrap_or_default() + "/.cache/uv/archive-v0",
            "/scratch".into(),
        ] {
            if let Ok(entries) = std::fs::read_dir(&base) {
                for ent in entries.flatten() {
                    let p = ent.path().join("nvidia/nvtx/lib");
                    if p.join("libnvToolsExt.so.1").exists() {
                        search_dirs.push(p.to_string_lossy().into_owned());
                    }
                }
            }
        }

        // Probe each search dir for a usable nvtx lib. CUDA 13 toolkits
        // ship no `libnvToolsExt.so` (only header-only `nvtx3`), and pip
        // packages ship only the versioned `.so.1`. Symlink the first
        // versioned hit into OUT_DIR so the bare `-lnvToolsExt` link
        // works regardless of the source.
        let mut linked = false;
        let out_dir = std::env::var("OUT_DIR").unwrap();
        for d in &search_dirs {
            println!("cargo:rustc-link-search=native={}", d);
            if !linked {
                let so1 = std::path::Path::new(d).join("libnvToolsExt.so.1");
                if so1.exists() {
                    let dst = std::path::Path::new(&out_dir).join("libnvToolsExt.so");
                    let _ = std::fs::remove_file(&dst);
                    if std::os::unix::fs::symlink(&so1, &dst).is_ok() {
                        println!("cargo:rustc-link-search=native={}", out_dir);
                        for ds in &search_dirs {
                            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ds);
                        }
                        linked = true;
                    }
                }
            }
        }
        println!("cargo:rustc-link-lib=dylib=nvToolsExt");
    }
}
