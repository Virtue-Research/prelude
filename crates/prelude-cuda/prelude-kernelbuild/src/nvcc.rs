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

pub const PRELUDE_CUDA_ARCHS_ENV: &str = "PRELUDE_CUDA_ARCHS";

// ─────────────────────────────────────────────────────────────────────
// CUDA toolkit discovery
// ─────────────────────────────────────────────────────────────────────

/// Locate the CUDA toolkit root. Checks `CUDA_HOME`, then `CUDA_PATH`,
/// then platform-specific default install paths, and panics with a clear
/// error if none are found.
///
/// Platform defaults:
///   Linux:   `/usr/local/cuda`, `/opt/cuda`
///   Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`
pub fn find_cuda() -> PathBuf {
    if let Ok(p) = env::var("CUDA_HOME") {
        return PathBuf::from(p);
    }
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }

    #[cfg(not(target_os = "windows"))]
    {
        for p in ["/usr/local/cuda", "/opt/cuda"] {
            if Path::new(p).join("bin/nvcc").exists() {
                return PathBuf::from(p);
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let base = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if base.is_dir() {
            if let Ok(entries) = std::fs::read_dir(base) {
                let mut versions: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.join("bin/nvcc.exe").exists())
                    .collect();
                versions.sort();
                if let Some(latest) = versions.pop() {
                    return latest;
                }
            }
        }
    }

    panic!(
        "CUDA toolkit not found. Set CUDA_HOME or CUDA_PATH to point at \
         a directory containing bin/nvcc{}.",
        if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        }
    );
}

/// Return the nvcc binary path for a given CUDA root. On Windows the
/// binary is `bin/nvcc.exe`.
pub fn nvcc_path(cuda_root: &Path) -> PathBuf {
    let name = if cfg!(target_os = "windows") {
        "bin/nvcc.exe"
    } else {
        "bin/nvcc"
    };
    let nvcc = cuda_root.join(name);
    if !nvcc.exists() {
        panic!(
            "nvcc not found at {}. Is CUDA_HOME/CUDA_PATH pointing at a \
             real CUDA toolkit install?",
            nvcc.display()
        );
    }
    nvcc
}

/// Parse an architecture list such as `sm_90,sm_103`, `90;103`, or
/// `8.9 9.0`. Returns normalized compute capabilities (`90`, `103`, ...).
pub fn parse_cuda_arch_list(value: &str) -> Vec<u32> {
    let mut archs: Vec<u32> = value
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter_map(parse_cuda_arch)
        .collect();
    archs.sort_unstable();
    archs.dedup();
    archs
}

fn parse_cuda_arch(raw: &str) -> Option<u32> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    let s = s
        .strip_prefix("sm_")
        .or_else(|| s.strip_prefix("sm"))
        .or_else(|| s.strip_prefix("compute_"))
        .or_else(|| s.strip_prefix("compute"))
        .unwrap_or(&s);
    let s = s.strip_suffix('a').unwrap_or(s);
    if let Some((major, minor)) = s.split_once('.') {
        let major = major.parse::<u32>().ok()?;
        let minor = minor.parse::<u32>().ok()?;
        return Some(major * 10 + minor);
    }
    s.parse::<u32>().ok()
}

pub fn cuda_archs_from_env(env_name: &str) -> Option<Vec<u32>> {
    let value = env::var(env_name).ok()?;
    if value.trim().is_empty() {
        return None;
    }
    let archs = parse_cuda_arch_list(&value);
    if archs.is_empty() {
        panic!(
            "{env_name}={value:?} did not contain any CUDA archs. \
             Use values like sm_90,sm_103 or 90;103."
        );
    }
    Some(archs)
}

/// Return the user-requested CUDA arch list, if present. Prelude's
/// workspace-wide override takes priority over CUDA_ARCH_LIST because it is
/// meant to control AOT fatbin size, not just the single local GPU probe.
pub fn requested_cuda_archs() -> Option<Vec<u32>> {
    cuda_archs_from_env(PRELUDE_CUDA_ARCHS_ENV).or_else(|| cuda_archs_from_env("CUDA_ARCH_LIST"))
}

pub fn requested_cuda_archs_or(default: &[u32]) -> Vec<u32> {
    requested_cuda_archs().unwrap_or_else(|| default.to_vec())
}

/// Same as [`requested_cuda_archs_or`], but lets a crate-specific env var
/// win over the workspace-wide setting.
pub fn requested_cuda_archs_or_with(crate_env: &str, default: &[u32]) -> Vec<u32> {
    cuda_archs_from_env(crate_env)
        .or_else(requested_cuda_archs)
        .unwrap_or_else(|| default.to_vec())
}

pub fn supported_cuda_archs(nvcc: &Path, default: &[u32]) -> Vec<u32> {
    let requested = requested_cuda_archs();
    supported_cuda_archs_inner(nvcc, requested, default)
}

pub fn supported_cuda_archs_with(nvcc: &Path, crate_env: &str, default: &[u32]) -> Vec<u32> {
    let requested = cuda_archs_from_env(crate_env).or_else(requested_cuda_archs);
    supported_cuda_archs_inner(nvcc, requested, default)
}

fn supported_cuda_archs_inner(
    nvcc: &Path,
    requested: Option<Vec<u32>>,
    default: &[u32],
) -> Vec<u32> {
    let user_requested = requested.is_some();
    let mut archs = requested.unwrap_or_else(|| default.to_vec());
    archs.sort_unstable();
    archs.dedup();

    let unsupported: Vec<u32> = archs
        .iter()
        .copied()
        .filter(|arch| !nvcc_supports_arch(nvcc, *arch))
        .collect();

    if user_requested && !unsupported.is_empty() {
        panic!(
            "Requested CUDA arch(s) {unsupported:?}, but {} does not support them",
            nvcc.display()
        );
    }

    archs.retain(|arch| nvcc_supports_arch(nvcc, *arch));
    if archs.is_empty() {
        panic!(
            "No supported CUDA archs remain. Requested/default arch list: {default:?}; nvcc: {}",
            nvcc.display()
        );
    }
    archs
}

pub fn cuda_arch_list_string(archs: &[u32]) -> String {
    archs
        .iter()
        .map(|arch| format!("sm_{arch}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn cutlass_gencode(arch: u32) -> String {
    let suffix = if matches!(arch, 90 | 100 | 103) {
        "a"
    } else {
        ""
    };
    format!("-gencode=arch=compute_{arch}{suffix},code=sm_{arch}{suffix}")
}

pub fn plain_gencode(arch: u32) -> String {
    format!("-gencode=arch=compute_{arch},code=sm_{arch}")
}

pub fn nvcc_supports_arch(nvcc: &Path, arch: u32) -> bool {
    let needle = format!("compute_{arch}");
    Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains(&needle))
        .unwrap_or(false)
}

/// Probe the local GPU's compute capability as an integer (e.g. `90` for
/// Hopper SM90). Honors `PRELUDE_CUDA_ARCHS` and `CUDA_ARCH_LIST`
/// overrides. Returns `None` if no GPU is present and the env vars are
/// unset — callers should fall back to a sensible default like 80 (Ampere)
/// for cross-compiles on CPU-only build hosts.
pub fn detect_compute_cap() -> Option<u32> {
    if let Some(cap) = requested_cuda_archs().and_then(|archs| archs.into_iter().max()) {
        return Some(cap);
    }

    // nvidia-smi is in PATH on Linux; on Windows it lives in the
    // driver directory and may or may not be in PATH.
    let smi = if cfg!(target_os = "windows") {
        "nvidia-smi.exe"
    } else {
        "nvidia-smi"
    };
    let output = Command::new(smi)
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
///
/// Setting `PRELUDE_FORCE_SM90_ONLY=1` forces this to return `false`,
/// skipping Blackwell codegen even on a Blackwell-capable toolchain.
/// Useful when CUDA 12.8's `ptxas` can't assemble certain Blackwell
/// kernel templates that the headers still emit (e.g. cuLA's
/// `kda_fwd_sm100` hits "Vector type too large, exceeds 128 bit
/// limit"); the workaround is to bypass the SM100 cubins entirely on
/// Hopper deployments and rely on the SM90 fallback.
pub fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    if std::env::var_os("PRELUDE_FORCE_SM90_ONLY").is_some() {
        return false;
    }
    nvcc_supports_arch(nvcc, 100)
}

/// Check whether the given nvcc supports Blackwell-Ultra (`compute_103`).
/// Required for B300 — SM100a PTX does NOT JIT-forward to SM103, so without
/// a native sm_103a cubin every B300 launch hits "no kernel image
/// available" or returns -2 from the dispatch.
///
/// Also gated by `PRELUDE_FORCE_SM90_ONLY=1`.
pub fn nvcc_supports_sm103(nvcc: &Path) -> bool {
    if std::env::var_os("PRELUDE_FORCE_SM90_ONLY").is_some() {
        return false;
    }
    nvcc_supports_arch(nvcc, 103)
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
/// pull in the CUDA runtime + its standard companion libs. Prefers the
/// static runtime so the consumer binary doesn't need the CUDA runtime
/// shared library at run time.
///
/// Platform-specific:
///   Linux:   `libcudart_static.a` + `-lrt -ldl -lstdc++`
///   Windows: `cudart_static.lib` (no rt/dl, stdc++ is MSVC CRT)
pub fn link_cuda_runtime_static(cuda_path: &Path) {
    emit_cuda_lib_search_paths(cuda_path);
    println!("cargo:rustc-link-lib=static=cudart_static");

    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// Dynamic variant of [`link_cuda_runtime_static`] — links against
/// `libcudart.so` / `cudart.lib`. Only used by the top-level
/// `prelude-cuda` build (which already depends on the dynamic runtime
/// for cudarc).
pub fn link_cuda_runtime_dynamic(cuda_path: &Path) {
    emit_cuda_lib_search_paths(cuda_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
}

/// Link against cuBLAS dynamically. Callers using raw cuBLAS FFI need this
/// even when cudarc is configured for fallback dynamic loading, because Rust's
/// final link still has to resolve the extern symbols.
pub fn link_cublas_dynamic(cuda_path: &Path) {
    emit_cuda_lib_search_paths(cuda_path);
    println!("cargo:rustc-link-lib=dylib=cublas");
}

/// Emit `cargo:rustc-link-search=native=` for the CUDA lib directories
/// that exist on this platform.
fn emit_cuda_lib_search_paths(cuda_path: &Path) {
    #[cfg(not(target_os = "windows"))]
    {
        let cuda_lib = cuda_path.join("lib64");
        if cuda_lib.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        }
        let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
        if cuda_targets_lib.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                cuda_targets_lib.display()
            );
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows CUDA Toolkit: lib/x64/ holds .lib files
        let cuda_lib = cuda_path.join("lib/x64");
        if cuda_lib.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Incremental-compile (per-file mtime) helpers
// ─────────────────────────────────────────────────────────────────────

/// Env var that forces a full recompile of every kernel, bypassing the
/// per-file mtime skip. Set `PRELUDE_KERNEL_FORCE_REBUILD=1` when you
/// suspect the incremental logic missed a dependency (or want a clean
/// timing run).
pub const FORCE_REBUILD_ENV: &str = "PRELUDE_KERNEL_FORCE_REBUILD";

fn force_rebuild() -> bool {
    matches!(
        env::var(FORCE_REBUILD_ENV).ok().as_deref(),
        Some(v) if !v.is_empty() && v != "0"
    )
}

/// Last-modified time of a path, or `None` if it can't be stat-ed.
fn mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Sidecar makefile-style depfile we ask nvcc to emit next to each output,
/// e.g. `foo.ptx` → `foo.ptx.d`. It records the source plus every header
/// nvcc pulled in, which is how a header edit invalidates the cached output.
fn depfile_path(output: &Path) -> PathBuf {
    let mut name = output
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(".d");
    output.with_file_name(name)
}

/// Parse a make-style depfile (as emitted by `nvcc -MD -MF`) into its list
/// of prerequisite paths.
///
/// The format is one logical rule using `\`-newline continuations:
///
/// ```text
/// out.ptx : a.cu \
///     header_a.cuh \
///     header_b.h
/// ```
///
/// Everything up to the first unescaped `:` (the target) is dropped, line
/// continuations are joined, and `\ ` escaped spaces inside paths are
/// preserved.
fn parse_depfile(contents: &str) -> Vec<PathBuf> {
    // Join `\`-newline (and `\`-CRLF) continuations into one stream.
    let joined = contents.replace("\\\r\n", " ").replace("\\\n", " ");
    // Drop the target: everything before the first ':'.
    let prereqs = match joined.split_once(':') {
        Some((_, rest)) => rest,
        None => joined.as_str(),
    };

    let mut deps = Vec::new();
    let mut current = String::new();
    let mut chars = prereqs.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // `\ ` is an escaped space embedded in a path.
            '\\' if chars.peek() == Some(&' ') => {
                chars.next();
                current.push(' ');
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    deps.push(PathBuf::from(std::mem::take(&mut current)));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        deps.push(PathBuf::from(current));
    }
    deps
}

/// Return `true` when `output` can be reused without recompiling: it exists,
/// is at least as new as `src`, and is newer than every prerequisite recorded
/// in its sidecar depfile.
///
/// Returns `false` (i.e. "must rebuild") whenever the output, the depfile, or
/// any prerequisite is missing/unreadable/newer — the safe default is always
/// to recompile so a stale or first-time build can never be silently skipped.
fn output_is_fresh(output: &Path, src: &Path) -> bool {
    let out_mtime = match mtime(output) {
        Some(t) => t,
        None => return false,
    };
    match mtime(src) {
        Some(t) if t > out_mtime => return false,
        Some(_) => {}
        // Source vanished — let nvcc produce the real error.
        None => return false,
    }

    // Without a depfile we have no record of which headers were pulled in,
    // so we conservatively rebuild (this also covers the very first build).
    let depfile = depfile_path(output);
    let contents = match std::fs::read_to_string(&depfile) {
        Ok(c) => c,
        Err(_) => return false,
    };
    for dep in parse_depfile(&contents) {
        match mtime(&dep) {
            Some(t) if t > out_mtime => return false,
            Some(_) => {}
            // A prerequisite moved/disappeared since the last build: rebuild.
            None => return false,
        }
    }
    true
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
    let src_name = opts
        .src
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Per-file mtime skip: if the .ptx is newer than its source and every
    // header recorded in the sidecar depfile, there is nothing to do.
    if !force_rebuild() && output_is_fresh(opts.out_ptx, opts.src) {
        build_log!("[nvcc] {src_name} (ptx) ↺ up-to-date, skipping");
        return;
    }

    let depfile = depfile_path(opts.out_ptx);
    let mut cmd = Command::new(nvcc);
    cmd.arg("--ptx")
        .arg(opts.src)
        .arg("-o")
        .arg(opts.out_ptx)
        .arg(format!("-arch=sm_{}", opts.compute_cap))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        // Emit a make-style depfile so the next build can mtime-skip when
        // neither the source nor any included header has changed.
        .arg("-MD")
        .arg("-MF")
        .arg(&depfile);
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
            opts.src
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default()
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
    let primary_arch = opts
        .gencodes
        .first()
        .cloned()
        .unwrap_or_else(|| "default".into());

    // Per-file mtime skip: reuse the .o when it is newer than its source and
    // every header recorded in the sidecar depfile.
    if !force_rebuild() && output_is_fresh(opts.out_obj, opts.src) {
        build_log!("[nvcc] {src_name} ({primary_arch}) ↺ up-to-date, skipping");
        return;
    }

    build_log!("[nvcc] {src_name} ({primary_arch})");

    let depfile = depfile_path(opts.out_obj);
    let mut cmd = Command::new(nvcc);
    cmd.arg(opts.cpp_std.as_deref().unwrap_or("-std=c++20"))
        .arg(opts.opt_level.as_deref().unwrap_or("-O3"))
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        // Emit a make-style depfile to drive the next build's mtime skip.
        .arg("-MD")
        .arg("-MF")
        .arg(&depfile);
    // -fPIC is Linux-only (ELF position-independent code). Windows
    // COFF is always position-independent so the flag doesn't exist.
    if opts.fpic && !cfg!(target_os = "windows") {
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

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn depfile_path_appends_d() {
        assert_eq!(depfile_path(Path::new("/o/foo.ptx")), Path::new("/o/foo.ptx.d"));
        assert_eq!(depfile_path(Path::new("/o/bar.o")), Path::new("/o/bar.o.d"));
    }

    #[test]
    fn parse_depfile_drops_target_and_joins_continuations() {
        let dep = "out.ptx : a.cu \\\n    /usr/include/foo.h \\\n    hdr.cuh\n";
        let deps = parse_depfile(dep);
        assert_eq!(
            deps,
            vec![
                PathBuf::from("a.cu"),
                PathBuf::from("/usr/include/foo.h"),
                PathBuf::from("hdr.cuh"),
            ]
        );
    }

    #[test]
    fn parse_depfile_handles_escaped_spaces() {
        let dep = "out.o: /path/with\\ space/h.cuh src.cu\n";
        let deps = parse_depfile(dep);
        assert_eq!(
            deps,
            vec![
                PathBuf::from("/path/with space/h.cuh"),
                PathBuf::from("src.cu"),
            ]
        );
    }

    #[test]
    fn fresh_output_skips_stale_output_rebuilds() {
        let dir = std::env::temp_dir().join(format!("prelude_inc_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("k.cu");
        let hdr = dir.join("k.cuh");
        let out = dir.join("k.ptx");
        let depf = depfile_path(&out);

        std::fs::write(&src, b"src").unwrap();
        std::fs::write(&hdr, b"hdr").unwrap();
        std::fs::write(&out, b"ptx").unwrap();
        std::fs::write(&depf, format!("{} : {} {}\n", out.display(), src.display(), hdr.display()))
            .unwrap();

        // Output is newest → fresh.
        let now = std::time::SystemTime::now();
        for p in [&src, &hdr] {
            set_mtime(p, now - Duration::from_secs(10));
        }
        set_mtime(&out, now);
        assert!(output_is_fresh(&out, &src), "output newer than inputs must be fresh");

        // Touch a header newer than the output → stale.
        set_mtime(&hdr, now + Duration::from_secs(10));
        assert!(!output_is_fresh(&out, &src), "header edit must invalidate output");

        // No depfile at all → always rebuild.
        std::fs::remove_file(&depf).unwrap();
        assert!(!output_is_fresh(&out, &src), "missing depfile must rebuild");

        std::fs::remove_dir_all(&dir).ok();
    }

    fn set_mtime(path: &Path, t: std::time::SystemTime) {
        let ft = filetime_from(t);
        // Use libc-free approach via utimensat is overkill for a test; fall
        // back to the filetime-style trick by re-writing then setting via the
        // standard library is not available, so shell out to `touch -d`.
        let secs = ft;
        let _ = Command::new("touch")
            .arg("-d")
            .arg(format!("@{secs}"))
            .arg(path)
            .status();
    }

    fn filetime_from(t: std::time::SystemTime) -> i64 {
        match t.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_secs() as i64,
            Err(e) => -(e.duration().as_secs() as i64),
        }
    }

    /// Try to locate an nvcc without panicking (unlike [`find_cuda`]), so the
    /// end-to-end test can no-op on hosts that lack a CUDA toolkit.
    fn try_find_nvcc() -> Option<PathBuf> {
        for var in ["CUDA_HOME", "CUDA_PATH"] {
            if let Ok(p) = env::var(var) {
                let n = Path::new(&p).join("bin/nvcc");
                if n.exists() {
                    return Some(n);
                }
            }
        }
        for p in ["/usr/local/cuda", "/opt/cuda"] {
            let n = Path::new(p).join("bin/nvcc");
            if n.exists() {
                return Some(n);
            }
        }
        None
    }

    #[test]
    fn end_to_end_compile_then_skip_then_rebuild() {
        let Some(nvcc) = try_find_nvcc() else {
            eprintln!("skipping: no nvcc on this host");
            return;
        };
        // sm_80 (Ampere) is supported by every CUDA toolkit we build with.
        if !nvcc_supports_arch(&nvcc, 80) {
            eprintln!("skipping: nvcc lacks sm_80");
            return;
        }

        let dir =
            std::env::temp_dir().join(format!("prelude_inc_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("k.cu");
        let hdr = dir.join("k.cuh");
        let out = dir.join("k.ptx");
        std::fs::write(&hdr, b"#define BUMP(v) ((v) + 1.0f)\n").unwrap();
        std::fs::write(
            &src,
            b"#include \"k.cuh\"\nextern \"C\" __global__ void k(float* x){ x[threadIdx.x] = BUMP(x[threadIdx.x]); }\n",
        )
        .unwrap();

        let opts = PtxCompile::new(&src, &out, 80).include(&dir);

        // 1) First call compiles and produces the .ptx + sidecar depfile.
        compile_cu_to_ptx(&nvcc, &opts);
        assert!(out.exists(), "first compile must produce the ptx");
        assert!(depfile_path(&out).exists(), "first compile must write a depfile");
        let mtime1 = mtime(&out).unwrap();

        // 2) Nothing changed → must be skipped (output mtime unchanged).
        std::thread::sleep(Duration::from_millis(1100));
        compile_cu_to_ptx(&nvcc, &opts);
        assert_eq!(mtime(&out).unwrap(), mtime1, "unchanged inputs must skip recompile");

        // 3) Editing the *header* must trigger a rebuild (newer ptx mtime).
        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(&hdr, b"#define BUMP(v) ((v) + 2.0f)\n").unwrap();
        compile_cu_to_ptx(&nvcc, &opts);
        assert!(
            mtime(&out).unwrap() > mtime1,
            "a header edit must force a recompile"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
