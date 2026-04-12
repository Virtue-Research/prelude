//! CUDA toolkit discovery, nvcc invocation, and workspace/filesystem
//! helpers shared across every kernel crate's `build.rs`.
//!
//! Before this module existed, every kernel crate under
//! `crates/prelude-cuda/*` had its own near-identical copies of:
//!
//!   * `find_cuda`          — locate the CUDA toolkit via `CUDA_HOME` /
//!                            `CUDA_PATH` / `/usr/local/cuda` fallback
//!   * `detect_compute_cap` — probe the local GPU's SM via `nvidia-smi` or
//!                            `CUDA_ARCH_LIST`
//!   * `nvcc_supports_sm100`— parse `nvcc --list-gpu-arch` to gate Blackwell
//!                            kernels
//!   * `track_submodule`    — emit `cargo:rerun-if-changed` for a submodule's
//!                            `.git/modules/<name>/HEAD` so re-pointing a
//!                            submodule re-runs the build script
//!   * `locate_source`      — resolve a third-party source root via env var
//!                            override + workspace fallback + marker-file
//!                            sanity check
//!   * `file_hash`          — SHA-256(first 16 hex chars) for cache keys
//!   * `link_cuda_runtime`  — emit the standard `cargo:rustc-link-*` lines
//!                            for libcudart_static + its companions
//!
//! All of the above are now pub fns in this module. Consumer crates do
//! `use prelude_kernelbuild::nvcc::{find_cuda, ...};` and drop their inline
//! copies.
//!
//! Single-file compile helpers:
//!
//!   * [`compile_cu_to_ptx`] — used by `prelude-cuda/build.rs` for its
//!     ~20 custom CUDA kernels that get loaded via cudarc at runtime.
//!   * [`compile_cu_to_obj`] — used by `cutlass-gemm`, `deepgemm`, and
//!     `cula`'s Phase 1 for their wrapper-style `.cu → .o → .a` pipelines.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::build_log;

// ─────────────────────────────────────────────────────────────────────
// CUDA toolkit discovery
// ─────────────────────────────────────────────────────────────────────

/// Locate the CUDA toolkit root. Checks `CUDA_HOME`, then `CUDA_PATH`,
/// then common install paths (`/usr/local/cuda`, `/opt/cuda`), and finally
/// panics with a clear error if none are found.
pub fn find_cuda() -> PathBuf {
    if let Ok(p) = env::var("CUDA_HOME") {
        return PathBuf::from(p);
    }
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        if Path::new(p).join("bin/nvcc").exists() {
            return PathBuf::from(p);
        }
    }
    panic!(
        "CUDA toolkit not found. Set CUDA_HOME or CUDA_PATH to point at \
         a directory containing bin/nvcc."
    );
}

/// Return the nvcc path for a given CUDA root, panicking with a clear
/// message if nvcc is missing. Every consumer crate does this exact check;
/// centralizing it gives uniform error messages.
pub fn nvcc_path(cuda_root: &Path) -> PathBuf {
    let nvcc = cuda_root.join("bin/nvcc");
    if !nvcc.exists() {
        panic!(
            "nvcc not found at {}. Is CUDA_HOME/CUDA_PATH pointing at a \
             real CUDA toolkit install?",
            nvcc.display()
        );
    }
    nvcc
}

/// Probe the local GPU's compute capability as an integer (e.g. `90` for
/// Hopper SM90). Honors `CUDA_ARCH_LIST=8.0,9.0` override. Returns `None`
/// if no GPU is present and the env var is unset — callers should fall
/// back to a sensible default like 80 (Ampere) for cross-compiles on
/// CPU-only build hosts.
pub fn detect_compute_cap() -> Option<u32> {
    if let Ok(arch_list) = env::var("CUDA_ARCH_LIST") {
        if let Some(cap) = arch_list
            .split(',')
            .filter_map(|s| {
                let s = s.trim().replace('.', "");
                s.parse::<u32>().ok()
            })
            .max()
        {
            return Some(cap);
        }
    }

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .next()?
        .trim()
        .replace('.', "")
        .parse::<u32>()
        .ok()
}

/// Like [`detect_compute_cap`] but returns the arch as an `sm_XX` string,
/// honoring a fallback for CPU-only build hosts.
pub fn detect_gpu_arch_str(default: &str) -> String {
    match detect_compute_cap() {
        Some(cap) => format!("sm_{cap}"),
        None => default.to_string(),
    }
}

/// Check whether the given nvcc supports Blackwell (`compute_100`) codegen.
/// Used to gate SM100 kernel compilation on older toolchains.
pub fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("compute_100"))
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────
// Workspace / submodule helpers
// ─────────────────────────────────────────────────────────────────────

/// Emit a `cargo:rerun-if-changed` line for a git submodule's HEAD file,
/// so re-pointing `third_party/<name>` via `git submodule update` triggers
/// a rebuild. Walks up from `CARGO_MANIFEST_DIR` looking for a `.git`
/// directory, which means the crate must live inside a git checkout that
/// has `third_party/<name>` as a submodule. When run outside such a
/// workspace (e.g. standalone crate build) the function is a no-op.
pub fn track_submodule(name: &str) {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut dir = manifest.as_path();
    loop {
        if dir.join(".git").is_dir() {
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

/// Resolve a third-party source root with env-var override + workspace
/// fallback + marker-file sanity check.
///
/// * `env_var`  — name of the env var consumers can set to override the
///                default path (e.g. `CULA_ROOT`)
/// * `name`     — human-readable name for error messages
/// * `marker`   — relative file that must exist inside the resolved dir as
///                a sanity check (e.g. `include/cutlass/cutlass.h`)
/// * `fallback` — default path when the env var is unset (typically
///                `third_party/<name>` relative to the workspace root)
///
/// Panics with a helpful message on miss so builds fail loudly instead of
/// silently compiling with broken include paths.
pub fn locate_source(env_var: &str, name: &str, marker: &str, fallback: &Path) -> PathBuf {
    let chosen = match env::var(env_var) {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => fallback.to_path_buf(),
    };
    if !chosen.join(marker).exists() {
        let hint = if env::var(env_var).is_ok() {
            format!(
                "{env_var}={} points at a directory that is missing `{marker}`. \
                 Check that it really is a {name} checkout.",
                chosen.display()
            )
        } else {
            format!(
                "Expected {name} at {} (missing `{marker}`). Either run \
                 `git submodule update --init third_party/{name}` inside the \
                 prelude workspace, or set {env_var}=/path/to/{name} to build \
                 this crate standalone.",
                chosen.display()
            )
        };
        panic!("{hint}");
    }
    chosen
}

/// SHA-256 of a file, truncated to 16 hex chars. Used as a cache key for
/// pre-compiled kernel archives so touching the compile script invalidates
/// the cache.
pub fn file_hash(path: &Path) -> Option<String> {
    use sha2::Digest;
    let content = std::fs::read(path).ok()?;
    let hash = sha2::Sha256::digest(&content);
    Some(hex::encode(hash)[..16].to_string())
}

// ─────────────────────────────────────────────────────────────────────
// Linking helpers
// ─────────────────────────────────────────────────────────────────────

/// Emit the `cargo:rustc-link-*` directives every kernel crate needs to
/// pull in the CUDA runtime + its standard companion libs (rt, dl, stdc++).
/// Prefers the static runtime (`libcudart_static.a`) so the consumer
/// binary doesn't need `libcudart.so` at runtime.
pub fn link_cuda_runtime_static(cuda_path: &Path) {
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
    if cuda_targets_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_lib.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Dynamic variant of [`link_cuda_runtime_static`] — links against
/// `libcudart.so`. Only used by the top-level `prelude-cuda` build (which
/// already depends on libcudart being available for cudarc's runtime).
pub fn link_cuda_runtime_dynamic(cuda_path: &Path) {
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
}

// ─────────────────────────────────────────────────────────────────────
// nvcc compile helpers
// ─────────────────────────────────────────────────────────────────────

/// Options for a single `.cu → .ptx` compile. Built via the builder
/// pattern so call sites stay readable when there are lots of flags.
#[derive(Debug, Clone)]
pub struct PtxCompile<'a> {
    pub src: &'a Path,
    pub out_ptx: &'a Path,
    /// Compute capability, e.g. `80` for `-arch=sm_80`.
    pub compute_cap: u32,
    /// Include directories. Passed as `-I<path>`.
    pub includes: Vec<PathBuf>,
    /// Extra raw nvcc flags, appended after the defaults.
    pub extra_flags: Vec<String>,
}

impl<'a> PtxCompile<'a> {
    pub fn new(src: &'a Path, out_ptx: &'a Path, compute_cap: u32) -> Self {
        Self {
            src,
            out_ptx,
            compute_cap,
            includes: Vec::new(),
            extra_flags: Vec::new(),
        }
    }

    pub fn include(mut self, dir: impl Into<PathBuf>) -> Self {
        self.includes.push(dir.into());
        self
    }

    pub fn extra_flag(mut self, flag: impl Into<String>) -> Self {
        self.extra_flags.push(flag.into());
        self
    }
}

/// Compile one `.cu` file to `.ptx` via nvcc. Panics on failure — build
/// scripts don't have a meaningful recovery path for a busted nvcc invoke.
pub fn compile_cu_to_ptx(nvcc: &Path, opts: &PtxCompile<'_>) {
    let mut cmd = Command::new(nvcc);
    cmd.arg("--ptx")
        .arg(opts.src)
        .arg("-o")
        .arg(opts.out_ptx)
        .arg(format!("-arch=sm_{}", opts.compute_cap))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr");
    for inc in &opts.includes {
        cmd.arg(format!("-I{}", inc.display()));
    }
    for flag in &opts.extra_flags {
        cmd.arg(flag);
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to run nvcc at {}: {e}", nvcc.display()));
    if !status.success() {
        panic!(
            "nvcc PTX compilation failed for {}",
            opts.src.file_name().map(|s| s.to_string_lossy()).unwrap_or_default()
        );
    }
}

/// Options for a single `.cu → .o` compile. Supports multiple `-gencode`
/// arch lines for fat binaries and arbitrary `-D` defines.
#[derive(Debug, Clone)]
pub struct ObjCompile<'a> {
    pub src: &'a Path,
    pub out_obj: &'a Path,
    /// Include directories.
    pub includes: Vec<PathBuf>,
    /// One or more `-gencode=arch=...,code=...` arg pairs. Each string is
    /// passed to nvcc verbatim.
    pub gencodes: Vec<String>,
    /// Preprocessor `-D` defines.
    pub defines: Vec<String>,
    /// C++ standard flag, e.g. `-std=c++17` or `-std=c++20`. Defaults to
    /// `-std=c++20` if unset (most CUTLASS code needs it).
    pub cpp_std: Option<String>,
    /// Extra raw nvcc flags.
    pub extra_flags: Vec<String>,
    /// Optimization level flag. Defaults to `-O3`.
    pub opt_level: Option<String>,
    /// Emit `-Xcompiler -fPIC`. Defaults true (needed for static archives
    /// linked into Rust lib crates).
    pub fpic: bool,
}

impl<'a> ObjCompile<'a> {
    pub fn new(src: &'a Path, out_obj: &'a Path) -> Self {
        Self {
            src,
            out_obj,
            includes: Vec::new(),
            gencodes: Vec::new(),
            defines: Vec::new(),
            cpp_std: None,
            extra_flags: Vec::new(),
            opt_level: None,
            fpic: true,
        }
    }

    pub fn include(mut self, dir: impl Into<PathBuf>) -> Self {
        self.includes.push(dir.into());
        self
    }

    pub fn gencode(mut self, gencode: impl Into<String>) -> Self {
        self.gencodes.push(gencode.into());
        self
    }

    pub fn define(mut self, define: impl Into<String>) -> Self {
        self.defines.push(define.into());
        self
    }

    pub fn cpp_std(mut self, std: impl Into<String>) -> Self {
        self.cpp_std = Some(std.into());
        self
    }

    pub fn extra_flag(mut self, flag: impl Into<String>) -> Self {
        self.extra_flags.push(flag.into());
        self
    }
}

/// Compile one `.cu` file to a relocatable `.o` object file via nvcc.
pub fn compile_cu_to_obj(nvcc: &Path, opts: &ObjCompile<'_>) {
    let src_name = opts
        .src
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let primary_arch = opts.gencodes.first().cloned().unwrap_or_else(|| "default".into());
    build_log!("[nvcc] {src_name} ({primary_arch})");

    let mut cmd = Command::new(nvcc);
    cmd.arg(opts.cpp_std.as_deref().unwrap_or("-std=c++20"))
        .arg(opts.opt_level.as_deref().unwrap_or("-O3"))
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda");
    if opts.fpic {
        cmd.arg("-Xcompiler").arg("-fPIC");
    }
    for g in &opts.gencodes {
        cmd.arg(g);
    }
    for d in &opts.defines {
        cmd.arg(d);
    }
    for inc in &opts.includes {
        cmd.arg(format!("-I{}", inc.display()));
    }
    for flag in &opts.extra_flags {
        cmd.arg(flag);
    }
    cmd.arg("-c").arg(opts.src).arg("-o").arg(opts.out_obj);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to run nvcc at {}: {e}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed for {}", opts.src.display());
    }
    build_log!("[nvcc] {src_name} ✓");
}
