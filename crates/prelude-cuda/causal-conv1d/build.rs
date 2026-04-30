//! Build script for the `causal-conv1d` crate — Rust bindings to
//! Dao-AILab's causal-conv1d CUDA kernels, statically linked (no
//! libtorch, no pip install required at runtime).
//!
//! ## What gets compiled
//!
//! Three CUDA sources go through nvcc:
//!   1. `csrc/causal_conv1d_shim.cu`        (our `extern "C"` wrapper)
//!   2. `$SRC/csrc/causal_conv1d_fwd.cu`    (upstream template defs)
//!   3. `$SRC/csrc/causal_conv1d_update.cu` (upstream template defs)
//!
//! We intentionally skip `causal_conv1d_bwd.cu` (backward — not used in
//! inference) and `causal_conv1d.cpp` (the torch::Tensor dispatch layer
//! which drags in all of libtorch).
//!
//! ## Include path ordering
//!
//! Upstream .cu files `#include <c10/util/BFloat16.h>` and friends. We
//! put `csrc/c10_compat/` on the nvcc include path **before** any real
//! torch include so those includes resolve to our 3-file shim that
//! typedefs `c10::BFloat16 → __nv_bfloat16` and defines the two
//! `C10_CUDA_*` logging macros upstream uses.
//!
//! ## Source discovery
//!
//! * `CAUSAL_CONV1D_ROOT` — explicit path to a checkout, if set
//! * default — `$CARGO_WORKSPACE/third_party/causal-conv1d`
//! * marker — `csrc/causal_conv1d.h` (the framework-agnostic header)
//!
//! ## Targeted GPU archs
//!
//! SM80 and SM90 by default (Ampere + Hopper + Ada). Override via
//! `CAUSAL_CONV1D_ARCH_LIST` if you need something narrower/wider.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

macro_rules! build_log {
    ($($arg:tt)*) => {{
        let _msg = format!($($arg)*);
        eprintln!("  [{}] {_msg}", env!("CARGO_PKG_NAME"));
        println!("cargo:warning={}", _msg);
    }};
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/causal_conv1d_shim.cu");
    println!("cargo:rerun-if-changed=csrc/c10_compat/c10/util/BFloat16.h");
    println!("cargo:rerun-if-changed=csrc/c10_compat/c10/util/Half.h");
    println!("cargo:rerun-if-changed=csrc/c10_compat/c10/cuda/CUDAException.h");
    println!("cargo:rerun-if-env-changed=CAUSAL_CONV1D_ROOT");
    println!("cargo:rerun-if-env-changed=CAUSAL_CONV1D_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    track_submodule("causal-conv1d");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let workspace_root = manifest_dir.join("../../..");
    let src_dir = locate_source(
        "CAUSAL_CONV1D_ROOT",
        "causal-conv1d",
        "csrc/causal_conv1d.h",
        &workspace_root.join("third_party/causal-conv1d"),
    )?;

    let cuda_path = find_cuda()?;
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        bail!("nvcc not found at {}", nvcc.display());
    }

    // Compile the three .cu files with shared nvcc flags.
    let arch_list = env::var("CAUSAL_CONV1D_ARCH_LIST").unwrap_or_else(|_| {
        // SM80 Ampere, SM89 Ada, SM90 Hopper, SM100/103 Blackwell. The
        // upstream causal_conv1d kernels are depthwise-conv vectorised
        // copies — they don't touch wgmma/MMA, so the same source
        // compiles cleanly for every SM target. Without an explicit
        // sm_103 cubin a B300 launch fails with "no kernel image is
        // available" because sm_90a PTX is arch-conditional and won't
        // JIT-forward to sm_103.
        "80;89;90;100;103".to_string()
    });
    let mut arch_flags: Vec<String> = Vec::new();
    for raw in arch_list.split(';') {
        let a = raw.trim();
        if a.is_empty() {
            continue;
        }
        let suffix = if a == "90" { "a" } else { "" };
        arch_flags.push(format!("-gencode=arch=compute_{a}{suffix},code=sm_{a}{suffix}"));
    }

    let shim_src = manifest_dir.join("csrc/causal_conv1d_shim.cu");
    let fwd_src = src_dir.join("csrc/causal_conv1d_fwd.cu");
    let update_src = src_dir.join("csrc/causal_conv1d_update.cu");

    for src in [&shim_src, &fwd_src, &update_src] {
        if !src.exists() {
            bail!("expected source at {}", src.display());
        }
    }
    println!("cargo:rerun-if-changed={}", fwd_src.display());
    println!("cargo:rerun-if-changed={}", update_src.display());

    // Include path order matters: c10_compat first so its headers
    // shadow the real torch ones the upstream .cu files ask for.
    let c10_compat = manifest_dir.join("csrc/c10_compat");
    let upstream_csrc = src_dir.join("csrc");
    let include_args: Vec<String> = vec![
        format!("-I{}", c10_compat.display()),
        format!("-I{}", upstream_csrc.display()),
        format!("-I{}/include", cuda_path.display()),
    ];

    let common_flags: Vec<&str> = vec![
        "-std=c++17",
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        // Upstream's `causal_conv1d_common.h` uses uint8_t/16_t/32_t/64_t
        // without including <cstdint>. In the torch build this works
        // because one of ATen's headers pulls cstdint in first; we have
        // no such header, so force-include it via nvcc.
        "--pre-include=cstdint",
        "-Xcompiler",
        "-fPIC",
        // Keep warnings visible but don't fail the build on upstream's
        // existing `__host__ __device__` annotation warnings.
        "-Xcudafe",
        "--diag_suppress=20012",
    ];

    let mut obj_files: Vec<PathBuf> = Vec::new();
    for src in [shim_src.as_path(), fwd_src.as_path(), update_src.as_path()] {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.o"));
        build_log!("[nvcc] {}", src.file_name().unwrap().to_string_lossy());

        let mut cmd = Command::new(&nvcc);
        cmd.args(&arch_flags)
            .args(&common_flags)
            .args(&include_args)
            .arg("-c")
            .arg(src)
            .arg("-o")
            .arg(&obj);
        let status = cmd
            .status()
            .with_context(|| format!("nvcc failed to start for {}", src.display()))?;
        if !status.success() {
            bail!("nvcc failed for {}", src.display());
        }
        obj_files.push(obj);
    }

    // Archive the three .o files into a single static library.
    let lib = out_dir.join("libcausal_conv1d.a");
    if lib.exists() {
        std::fs::remove_file(&lib)?;
    }
    let mut cmd = Command::new(&nvcc);
    cmd.arg("--lib")
        .args(["-o", lib.to_str().unwrap()]);
    for obj in &obj_files {
        cmd.arg(obj);
    }
    let status = cmd.status().context("nvcc --lib failed to start")?;
    if !status.success() {
        bail!("nvcc --lib failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=causal_conv1d");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    build_log!("linked libcausal_conv1d.a ({} objects)", obj_files.len());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Helpers (inlined — this crate must be able to build standalone)
// ─────────────────────────────────────────────────────────────────────

fn track_submodule(name: &str) {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut dir = manifest.as_path();
    loop {
        if dir.join(".git").exists() {
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

fn locate_source(env_var: &str, name: &str, marker: &str, default: &Path) -> Result<PathBuf> {
    if let Ok(explicit) = env::var(env_var) {
        let p = PathBuf::from(explicit);
        if !p.join(marker).exists() {
            bail!("{env_var}={} does not contain {marker}", p.display());
        }
        return Ok(p);
    }
    if default.join(marker).exists() {
        return Ok(default.to_path_buf());
    }
    bail!(
        "{name} source not found.\n  \
         Either set {env_var}=/path/to/{name} or run \
         `git submodule update --init third_party/{name}` inside the prelude workspace.\n  \
         Tried: {}",
        default.display()
    )
}

fn find_cuda() -> Result<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        let p = PathBuf::from(p);
        if p.join("bin/nvcc").exists() {
            return Ok(p);
        }
    }
    for candidate in ["/usr/local/cuda", "/opt/cuda"] {
        let p = PathBuf::from(candidate);
        if p.join("bin/nvcc").exists() {
            return Ok(p);
        }
    }
    bail!("CUDA toolkit not found. Set CUDA_PATH or install at /usr/local/cuda");
}
