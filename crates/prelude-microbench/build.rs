use std::env;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ggml_baseline)");

    // Link against pre-built ggml-cpu library for baseline comparison.
    // In Docker: GGML_LIB points to llama.cpp build directory.
    // Locally: not set → ggml baseline disabled.
    let ggml_lib = match env::var("GGML_LIB") {
        Ok(p) => p,
        Err(_) => {
            println!("cargo:warning=GGML_LIB not set — ggml baseline disabled");
            return;
        }
    };

    println!("cargo:rustc-cfg=ggml_baseline");
    println!("cargo:rerun-if-env-changed=GGML_LIB");

    // Link ggml-cpu static library and its dependencies
    println!("cargo:rustc-link-search=native={}/ggml/src/ggml-cpu", ggml_lib);
    println!("cargo:rustc-link-search=native={}/ggml/src", ggml_lib);
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=stdc++");
}
