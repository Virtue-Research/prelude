//! Python virtualenv provisioning for crate build scripts.
//!
//! `cula` and `fa4` both do the same dance: look for `uv`, fall back to
//! `python3 -m venv`, install torch with a CUDA-matching wheel index,
//! install kernel-specific packages, then check importability. Before
//! this module they each had ~150 lines of inline copy-paste that
//! drifted over time. Now both go through [`PythonVenv`].
//!
//! The API is deliberately thin — it's a small wrapper around
//! `uv pip` / `python3 -m pip` commands, not a reimplementation of pip's
//! resolver. Consumers still own the decision of which packages to
//! install in what order and in which phases.
//!
//! ## Typical usage
//!
//! ```ignore
//! use prelude_kernelbuild::venv::{PythonVenv, InstallOpts};
//!
//! let venv = PythonVenv::ensure(&out_dir.join("foo-venv"))?;
//!
//! // Install torch with a CUDA-matching wheel index (must come first
//! // when the next package needs torch available at setup-time).
//! let torch_index = prelude_kernelbuild::venv::detect_torch_cuda_index();
//! venv.pip_install(
//!     &["torch"],
//!     InstallOpts::new().extra_index_url(torch_index.as_deref()),
//! )?;
//!
//! // Install kernel-specific packages.
//! venv.pip_install(
//!     &["nvidia-cutlass-dsl>=4.4.2", "einops"],
//!     InstallOpts::new(),
//! )?;
//!
//! // Verify what the consumer actually needs.
//! if !venv.check_import("import cutlass") {
//!     anyhow::bail!("cutlass not importable after install");
//! }
//!
//! let python = venv.python_path();
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::build_log;

/// Detect the local CUDA toolkit's major.minor release from `nvcc
/// --version` and return the matching PyTorch wheel index URL. Returns
/// `None` if nvcc is missing or the release banner can't be parsed.
///
/// Example: CUDA 12.8 → `https://download.pytorch.org/whl/cu128`.
pub fn detect_torch_cuda_index() -> Option<String> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let release = text.split("release ").nth(1)?;
    let version = release.split(',').next()?.trim();
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 2 {
        let cu_tag = format!("{}{}", parts[0], parts[1]);
        Some(format!("https://download.pytorch.org/whl/cu{cu_tag}"))
    } else {
        None
    }
}

/// Whether `uv` is callable on PATH. Cached lookup — we only probe once
/// per build-script run since uv doesn't disappear mid-compile.
pub fn has_uv() -> bool {
    Command::new("uv")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A managed Python virtualenv rooted at a specific directory. Cheap to
/// construct via [`PythonVenv::ensure`] — subsequent calls with the same
/// directory reuse the existing venv if it's already there.
///
/// The venv itself has no opinion about which packages are installed;
/// consumers call [`PythonVenv::pip_install`] to layer packages on.
pub struct PythonVenv {
    venv_dir: PathBuf,
    python: PathBuf,
    has_uv: bool,
}

impl PythonVenv {
    /// Create the venv directory if it doesn't already exist. Prefers
    /// `uv venv --python 3.12` for speed, falling back to
    /// `python3 -m venv` when uv isn't on PATH. If a venv already lives
    /// at this path (i.e. `bin/python3` exists) it's reused as-is.
    pub fn ensure(venv_dir: &Path) -> Result<Self, String> {
        let python = venv_dir.join("bin/python3");
        let has_uv = has_uv();

        if !python.exists() {
            build_log!("creating Python venv at {}", venv_dir.display());
            if has_uv {
                let status = Command::new("uv")
                    .args(["venv", venv_dir.to_str().unwrap(), "--python", "3.12"])
                    .status()
                    .map_err(|e| format!("uv venv spawn failed: {e}"))?;
                if !status.success() {
                    return Err("uv venv creation failed".into());
                }
            } else {
                let status = Command::new("python3")
                    .args(["-m", "venv", venv_dir.to_str().unwrap()])
                    .status()
                    .map_err(|e| format!("python3 -m venv spawn failed: {e}"))?;
                if !status.success() {
                    return Err(
                        "python3 -m venv failed; install uv or a working python3".into(),
                    );
                }
            }
            if !python.exists() {
                return Err(format!(
                    "venv created at {} but {} is missing",
                    venv_dir.display(),
                    python.display()
                ));
            }
        }

        Ok(Self {
            venv_dir: venv_dir.to_path_buf(),
            python,
            has_uv,
        })
    }

    /// Absolute path to the venv's `python3` binary. Pass this as the
    /// interpreter when invoking user scripts so they pick up the venv's
    /// installed packages without PATH gymnastics.
    pub fn python_path(&self) -> &Path {
        &self.python
    }

    /// Path to the venv root. Mostly useful for logging.
    pub fn venv_dir(&self) -> &Path {
        &self.venv_dir
    }

    /// Install one or more packages into the venv. Prefers `uv pip`
    /// (faster, better resolver), falls back to `python -m pip` when uv
    /// isn't available.
    ///
    /// `packages` can be any mix of pip-accepted specs: wheels (`torch`),
    /// versioned specs (`nvidia-cutlass-dsl>=4.4.2`), git URLs
    /// (`flash-linear-attention@git+...`), or local paths.
    pub fn pip_install(&self, packages: &[&str], opts: InstallOpts<'_>) -> Result<(), String> {
        if packages.is_empty() {
            return Ok(());
        }
        build_log!(
            "installing {} package{} ({})",
            packages.len(),
            if packages.len() == 1 { "" } else { "s" },
            packages.join(", ")
        );

        let venv_python_str = self.python.to_string_lossy().to_string();

        let mut cmd = if self.has_uv {
            let mut c = Command::new("uv");
            c.args(["pip", "install", "--python", &venv_python_str]);
            c
        } else {
            let pip = self.venv_dir.join("bin/pip");
            let mut c = Command::new(pip);
            c.arg("install");
            c
        };

        if opts.no_build_isolation {
            cmd.arg("--no-build-isolation");
        }
        if let Some(idx) = opts.extra_index_url {
            cmd.args(["--extra-index-url", idx]);
        }
        for pkg in packages {
            cmd.arg(pkg);
        }

        let output = cmd.output().map_err(|e| format!("pip install spawn failed: {e}"))?;
        if !output.status.success() {
            // Surface the last few stderr lines — pip tracebacks are huge
            // and build-log truncation chops off the useful part first.
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr
                .lines()
                .rev()
                .take(10)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                build_log!("[pip] {line}");
            }
            return Err(format!(
                "pip install failed (exit {}): {}",
                output.status,
                packages.join(" ")
            ));
        }
        Ok(())
    }

    /// Run `python -c <code>` inside the venv and return `true` iff the
    /// interpreter exits with code 0. Used to probe whether a
    /// consumer-specific import is satisfied after install. Example:
    /// `venv.check_import("import cula; from cula.kda import kda_decode")`.
    pub fn check_import(&self, code: &str) -> bool {
        Command::new(&self.python)
            .args(["-c", code])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// Locate the `nvidia_cutlass_dsl` package's `lib/` directory inside a
/// Python environment. Returns the path to the directory containing
/// `libcuda_dialect_runtime_static.a` — which must be linked by any
/// crate whose `.o` files were produced by `cute.compile → export_to_c`
/// (both cuLA and FA4 DSL kernels need it).
///
/// The `cuda_dialect_runtime` library provides the MLIR-generated shims
/// (`_cudaGetDevice`, `cuda_dialect_init_library_once`, etc.) that the
/// CuTeDSL code generator emits as external references. Without it, fat
/// LTO builds fail with ~500 undefined symbols from
/// `libcula_dsl_kernels.a` / `libfa4_kernels.a`.
///
/// `python` is the venv's Python binary (typically from
/// `PythonVenv::python_path()`). Returns `None` if the package isn't
/// installed or the lib dir doesn't exist.
pub fn find_cutlass_dsl_lib(python: &Path) -> Option<PathBuf> {
    let output = Command::new(python)
        .args(["-c", "import nvidia_cutlass_dsl; print(nvidia_cutlass_dsl.__path__[0])"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let pkg_root = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let lib_dir = pkg_root.join("lib");
    if lib_dir.join("libcuda_dialect_runtime_static.a").exists() {
        Some(lib_dir)
    } else {
        None
    }
}

/// Emit `cargo:rustc-link-*` directives to link the
/// `cuda_dialect_runtime_static` library from the cutlass-dsl package.
/// Call this after DSL kernel `.o` files have been archived.
///
/// Returns `true` if the library was found and the directives were
/// emitted, `false` if the library wasn't found (caller should decide
/// whether to warn or error).
pub fn link_cutlass_dsl_runtime(python: &Path) -> bool {
    if let Some(lib_dir) = find_cutlass_dsl_lib(python) {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=cuda_dialect_runtime_static");
        build_log!(
            "linked libcuda_dialect_runtime_static.a from {}",
            lib_dir.display()
        );
        true
    } else {
        build_log!(
            "WARNING: nvidia_cutlass_dsl lib/ not found — cuda_dialect_runtime \
             will be missing, dist/LTO builds may fail with undefined symbols"
        );
        false
    }
}

/// Options for a single [`PythonVenv::pip_install`] call. Default is an
/// empty struct — opt into extras via the builder methods.
#[derive(Debug, Default, Clone, Copy)]
pub struct InstallOpts<'a> {
    /// Extra pip index URL (usually a CUDA-matching PyTorch wheel index
    /// produced by [`detect_torch_cuda_index`]).
    pub extra_index_url: Option<&'a str>,
    /// Pass `--no-build-isolation` — required for source installs when
    /// the sdist's build backend assumes torch is already present in the
    /// current env (common for CUTLASS-heavy projects).
    pub no_build_isolation: bool,
}

impl<'a> InstallOpts<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extra_index_url(mut self, url: Option<&'a str>) -> Self {
        self.extra_index_url = url;
        self
    }

    pub fn no_build_isolation(mut self, no_isolation: bool) -> Self {
        self.no_build_isolation = no_isolation;
        self
    }
}
