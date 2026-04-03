//! Build script for prelude-cula: cuLA linear attention kernels.
//!
//! Phase 1: Compile C++ CUTLASS 3.x kernels (SM90/SM100) via nvcc → static archive.
//! Phase 2: AOT compile CuTe DSL kernels via Python → .o files → static archive.
//! Phase 3: Compile vendored TVM FFI (needed by DSL kernel symbols).
//! Phase 4: Generate Rust dispatch code from manifest.json.

use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cula_wrapper.cu");
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-changed=kernels/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.join("../../..");

    // Ensure submodules
    let cula_dir = workspace_root.join("third_party/cuLA");
    if !cula_dir.join("csrc/kda/sm90/prefill_kernel.hpp").exists() {
        panic!(
            "third_party/cuLA submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/cuLA"
        );
    }
    let cutlass_dir = workspace_root.join("third_party/cutlass");
    if !cutlass_dir.join("include/cutlass/cutlass.h").exists() {
        panic!(
            "third_party/cutlass submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/cutlass"
        );
    }

    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }
    let sm100 = nvcc_supports_sm100(&nvcc);

    // ── Phase 1: C++ CUTLASS kernels ─────────────────────────────────
    let cpp_objects = compile_cpp_kernels(
        &nvcc, &cula_dir, &cutlass_dir, &cuda_path, &out_dir, &manifest_dir, sm100,
    );

    // Archive C++ objects → libcula_cpp.a
    if !cpp_objects.is_empty() {
        let lib = out_dir.join("libcula_cpp.a");
        let mut cmd = Command::new(&nvcc);
        cmd.arg("--lib").args(["-o", lib.to_str().unwrap()]);
        for obj in &cpp_objects {
            cmd.arg(obj);
        }
        let status = cmd.status().expect("Failed to run nvcc --lib");
        if !status.success() {
            panic!("nvcc --lib failed for libcula_cpp.a");
        }
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=cula_cpp");
    }

    // ── Phase 2: CuTe DSL kernels ────────────────────────────────────
    let kernels_dir = manifest_dir.join("kernels");
    let has_dsl_kernels = compile_dsl_kernels(&kernels_dir, &manifest_dir, &cula_dir, &out_dir);

    if has_dsl_kernels {
        link_dsl_kernel_objects(&kernels_dir, &out_dir);
    }

    // ── Phase 3: TVM FFI ─────────────────────────────────────────────
    if has_dsl_kernels {
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap();
        let tvm_ffi_dir = workspace_root.join("third_party/tvm-ffi");
        if !tvm_ffi_dir.join("src").exists() {
            panic!("third_party/tvm-ffi not found. Run: git submodule update --init --recursive third_party/tvm-ffi");
        }
        compile_tvm_ffi_static(&tvm_ffi_dir, &manifest_dir, &out_dir, &cuda_path);
    }

    // ── Phase 4: Generate dispatch code ──────────────────────────────
    generate_dispatch_code(&kernels_dir, &out_dir, has_dsl_kernels);

    // ── Link CUDA runtime ────────────────────────────────────────────
    link_cuda_runtime(&cuda_path);
}

// ─────────────────────────────────────────────────────────────────────
// Phase 1: C++ CUTLASS kernels
// ─────────────────────────────────────────────────────────────────────

fn compile_cpp_kernels(
    nvcc: &Path,
    cula_dir: &Path,
    cutlass_dir: &Path,
    cuda_path: &Path,
    out_dir: &Path,
    manifest_dir: &Path,
    sm100: bool,
) -> Vec<PathBuf> {
    let include_args: Vec<String> = vec![
        format!("-I{}", cutlass_dir.join("include").display()),
        format!("-I{}", cutlass_dir.join("tools/util/include").display()),
        format!("-I{}", cula_dir.join("csrc").display()),
        format!("-I{}", cula_dir.join("csrc/kerutils/include").display()),
        format!("-I{}/include", cuda_path.display()),
    ];

    let sm90_sources = vec![
        cula_dir.join("csrc/kda/sm90/kda_fwd_sm90.cu"),
        cula_dir.join("csrc/kda/sm90/kda_fwd_sm90_safe_gate.cu"),
    ];
    let wrapper_src = manifest_dir.join("src/cula_wrapper.cu");

    let mut objects = Vec::new();

    // SM90 kernel sources
    for src in &sm90_sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.o"));
        nvcc_compile(nvcc, src, &obj, &include_args, &[
            "-gencode=arch=compute_90a,code=sm_90a",
        ], &["-DCULA_SM90A_ENABLED"]);
        objects.push(obj);
    }

    // SM90 wrapper
    {
        let obj = out_dir.join("cula_wrapper_sm90.o");
        nvcc_compile(nvcc, &wrapper_src, &obj, &include_args, &[
            "-gencode=arch=compute_90a,code=sm_90a",
        ], &["-DCULA_SM90A_ENABLED"]);
        objects.push(obj);
    }

    // SM100
    if sm100 {
        println!("cargo:warning=cuLA: compiling SM100 KDA kernels");
        let sm100_src = cula_dir.join("csrc/kda/sm100/kda_fwd_sm100.cu");
        let obj = out_dir.join("kda_fwd_sm100.o");
        nvcc_compile(nvcc, &sm100_src, &obj, &include_args, &[
            "-gencode=arch=compute_100a,code=sm_100a",
        ], &[]);
        objects.push(obj);

        let obj = out_dir.join("cula_wrapper_sm100.o");
        nvcc_compile(nvcc, &wrapper_src, &obj, &include_args, &[
            "-gencode=arch=compute_100a,code=sm_100a",
        ], &["-DCULA_SM100_ENABLED"]);
        objects.push(obj);
    } else {
        println!("cargo:warning=cuLA: SM100 not supported by nvcc, skipping Blackwell kernels");
    }

    objects
}

fn nvcc_compile(
    nvcc: &Path, src: &Path, obj: &Path,
    include_args: &[String], arch_args: &[&str], defines: &[&str],
) {
    let mut cmd = Command::new(nvcc);
    cmd.args([
        "-std=c++20", "-O3",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-Xcompiler", "-fPIC",
    ]);
    for a in arch_args {
        cmd.arg(a);
    }
    for d in defines {
        cmd.arg(d);
    }
    for inc in include_args {
        cmd.arg(inc);
    }
    cmd.args(["-c", src.to_str().unwrap(), "-o", obj.to_str().unwrap()]);
    let status = cmd.status().expect("Failed to run nvcc");
    if !status.success() {
        panic!("nvcc failed for {}", src.display());
    }
}

// ─────────────────────────────────────────────────────────────────────
// Phase 2: CuTe DSL kernels (AOT via Python, same pattern as FA4)
// ─────────────────────────────────────────────────────────────────────

fn compile_dsl_kernels(
    kernels_dir: &Path,
    manifest_dir: &Path,
    cula_dir: &Path,
    out_dir: &Path,
) -> bool {
    let script = manifest_dir.join("scripts/compile_kernels.py");

    // Check if pre-compiled kernels exist and are up-to-date
    if let Some(hash) = file_hash(&script) {
        if kernels_dir.join("manifest.json").exists() && has_obj_files(kernels_dir) {
            let manifest_str = std::fs::read_to_string(kernels_dir.join("manifest.json"))
                .unwrap_or_default();
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                let stored = manifest.get("script_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if stored == hash {
                    println!("cargo:warning=cuLA DSL: using cached kernels");
                    return true;
                }
                println!("cargo:warning=cuLA DSL: compile_kernels.py changed, recompiling...");
                clear_obj_files(kernels_dir);
            }
        }
    }

    println!("cargo:warning=cuLA DSL: attempting AOT compilation...");

    if !script.exists() {
        println!("cargo:warning=cuLA DSL: no compile script, skipping DSL kernels");
        return false;
    }

    let python = match ensure_python_env(out_dir) {
        Ok(p) => p,
        Err(e) => {
            println!("cargo:warning=cuLA DSL: Python env failed: {e}, skipping DSL kernels");
            return false;
        }
    };

    // Detect GPU arch
    let arch = detect_gpu_arch().unwrap_or_else(|| "sm_90".to_string());
    println!("cargo:warning=cuLA DSL: AOT compiling for {arch}...");

    let workers = env::var("PRELUDE_CULA_WORKERS").unwrap_or_else(|_| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        n.min(8).to_string()
    });

    let status = Command::new(&python)
        .arg(&script)
        .arg("--output-dir").arg(kernels_dir)
        .args(["-j", &workers])
        .env("PYTHONPATH", cula_dir)
        .env("CUTE_DSL_ARCH", format!("{arch}a"))
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=cuLA DSL: compilation succeeded");
            has_obj_files(kernels_dir)
        }
        Ok(_) => {
            println!("cargo:warning=cuLA DSL: compilation failed");
            false
        }
        Err(e) => {
            println!("cargo:warning=cuLA DSL: failed to run script: {e}");
            false
        }
    }
}

fn link_dsl_kernel_objects(kernels_dir: &Path, out_dir: &Path) {
    let obj_files: Vec<PathBuf> = std::fs::read_dir(kernels_dir)
        .ok().into_iter().flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "o"))
        .collect();

    if obj_files.is_empty() {
        return;
    }

    println!(
        "cargo:warning=cuLA DSL: archiving {} kernel .o files",
        obj_files.len()
    );

    let archive_path = out_dir.join("libcula_dsl_kernels.a");
    let mut cmd = Command::new("ar");
    cmd.arg("rcs").arg(&archive_path);
    for obj in &obj_files {
        cmd.arg(obj);
    }
    let status = cmd.status().expect("Failed to run ar");
    if !status.success() {
        panic!("ar failed to create libcula_dsl_kernels.a");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=cula_dsl_kernels");
}

// ─────────────────────────────────────────────────────────────────────
// Phase 3: TVM FFI (from third_party/tvm-ffi)
// ─────────────────────────────────────────────────────────────────────

fn compile_tvm_ffi_static(tvm_ffi_dir: &Path, manifest_dir: &Path, out_dir: &Path, _cuda_path: &Path) {
    let tvm_src = tvm_ffi_dir.join("src");
    let tvm_include = tvm_ffi_dir.join("include");
    let dlpack_include = tvm_ffi_dir.join("3rdparty/dlpack/include");

    let cc_files: Vec<PathBuf> = walkdir(&tvm_src, "cc")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            !name.contains("win") && !name.contains("testing")
        })
        .collect();

    println!("cargo:warning=cuLA: compiling TVM FFI ({} files)", cc_files.len());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .pic(true)
        .include(&tvm_include)
        .include(&dlpack_include)
        .define("TVM_FFI_EXPORTS", None)
        .define("NDEBUG", None)
        .warnings(false);

    for f in &cc_files {
        build.file(f);
    }

    // TVM error helper
    let error_helper = manifest_dir.join("src/tvm_error_helper.cc");
    if error_helper.exists() {
        build.file(&error_helper);
    }

    build.link_lib_modifier("+whole-archive");
    build.try_compile("cula_tvm_ffi_static")
        .expect("Failed to compile TVM FFI for cuLA");

    // Compile libbacktrace (backtrace.cc references it)
    compile_libbacktrace(tvm_ffi_dir, out_dir);

    // cuda_dialect_runtime_static.a — from fa4-venv or cula-venv
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap();
    let cuda_dialect_lib = "libcuda_dialect_runtime_static.a";
    let cuda_dialect_dir = find_file_recursive(&workspace_root.join("target/fa4-venv"), cuda_dialect_lib)
        .or_else(|| find_file_recursive(&out_dir.join("cula-venv"), cuda_dialect_lib))
        .map(|p| p.parent().unwrap().to_path_buf());
    if let Some(dir) = cuda_dialect_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=cuda_dialect_runtime_static");
    } else {
        println!("cargo:warning=cuLA: cuda_dialect_runtime_static.a not found");
    }
}

fn compile_libbacktrace(tvm_ffi_dir: &Path, out_dir: &Path) {
    let bt_dir = tvm_ffi_dir.join("3rdparty/libbacktrace");
    if !bt_dir.exists() {
        panic!("libbacktrace not found. Run: git submodule update --init --recursive third_party/tvm-ffi");
    }

    let config_dir = out_dir.join("libbacktrace");
    std::fs::create_dir_all(&config_dir).unwrap();

    let config_h = if cfg!(target_os = "linux") {
        "#define BACKTRACE_ELF_SIZE 64\n#define HAVE_ATOMIC_FUNCTIONS 1\n#define HAVE_DL_ITERATE_PHDR 1\n\
         #define HAVE_DLFCN_H 1\n#define HAVE_FCNTL 1\n#define HAVE_LINK_H 1\n#define HAVE_LSTAT 1\n\
         #define HAVE_READLINK 1\n#define HAVE_SYS_MMAN_H 1\n#define HAVE_DECL_STRNLEN 1\n#define HAVE_DECL_GETPAGESIZE 0\n"
    } else if cfg!(target_os = "macos") {
        "#define BACKTRACE_ELF_SIZE 64\n#define HAVE_ATOMIC_FUNCTIONS 1\n#define HAVE_DLFCN_H 1\n\
         #define HAVE_FCNTL 1\n#define HAVE_MACH_O_DYLD_H 1\n#define HAVE_SYS_MMAN_H 1\n\
         #define HAVE_DECL_STRNLEN 1\n#define HAVE_DECL_GETPAGESIZE 0\n"
    } else {
        "#define HAVE_ATOMIC_FUNCTIONS 1\n#define HAVE_DECL_STRNLEN 1\n#define HAVE_DECL_GETPAGESIZE 0\n"
    };
    std::fs::write(config_dir.join("config.h"), config_h).unwrap();

    let core_files = ["backtrace.c", "dwarf.c", "fileline.c", "posix.c",
                      "sort.c", "state.c", "alloc.c", "read.c", "mmapio.c", "mmap.c"];
    let format_file = if cfg!(target_os = "macos") { "macho.c" } else { "elf.c" };

    let mut build = cc::Build::new();
    build.opt_level(2).pic(true).include(&bt_dir).include(&config_dir)
        .define("_GNU_SOURCE", None).warnings(false);

    for name in &core_files {
        let f = bt_dir.join(name);
        if f.exists() { build.file(&f); }
    }
    let fmt = bt_dir.join(format_file);
    if fmt.exists() { build.file(&fmt); }

    build.try_compile("cula_backtrace").expect("Failed to compile libbacktrace for cuLA");
}

// ─────────────────────────────────────────────────────────────────────
// Phase 4: Generate Rust dispatch code
// ─────────────────────────────────────────────────────────────────────

fn generate_dispatch_code(kernels_dir: &Path, out_dir: &Path, has_kernels: bool) {
    let dispatch_path = out_dir.join("cula_dsl_dispatch.rs");

    if !has_kernels {
        std::fs::write(
            &dispatch_path,
            "pub(crate) fn lookup_dsl(_kernel_type: &str, _key: &str, _arch: u32) \
             -> Option<crate::dsl::TVMSafeCallFn> { None }\n",
        ).unwrap();
        return;
    }

    let manifest_path = kernels_dir.join("manifest.json");
    let manifest_str = match std::fs::read_to_string(&manifest_path) {
        Ok(s) => s,
        Err(_) => {
            std::fs::write(
                &dispatch_path,
                "pub(crate) fn lookup_dsl(_kernel_type: &str, _key: &str, _arch: u32) \
                 -> Option<crate::dsl::TVMSafeCallFn> { None }\n",
            ).unwrap();
            return;
        }
    };

    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)
        .expect("Failed to parse cuLA manifest.json");
    let variants = manifest["variants"].as_array()
        .expect("manifest.json missing 'variants' array");

    let mut code = String::new();
    writeln!(code, "// AUTO-GENERATED by build.rs — do not edit").unwrap();
    writeln!(code).unwrap();

    // Extern declarations
    writeln!(code, "unsafe extern \"C\" {{").unwrap();
    for variant in variants {
        let symbol = variant["symbol"].as_str().unwrap();
        writeln!(
            code,
            "    fn {symbol}(handle: *mut std::ffi::c_void, args: *const crate::dsl::TVMFFIAny, \
             num_args: i32, ret: *mut crate::dsl::TVMFFIAny) -> i32;"
        ).unwrap();
    }
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Lookup by name
    writeln!(
        code,
        "pub(crate) fn lookup_dsl(_kernel_type: &str, key: &str, _arch: u32) \
         -> Option<crate::dsl::TVMSafeCallFn> {{"
    ).unwrap();
    writeln!(code, "    match key {{").unwrap();
    for variant in variants {
        let name = variant["name"].as_str().unwrap();
        let symbol = variant["symbol"].as_str().unwrap();
        writeln!(code, "        \"{name}\" => Some({symbol}),").unwrap();
    }
    writeln!(code, "        _ => None,").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();

    std::fs::write(&dispatch_path, &code).unwrap();
    println!(
        "cargo:warning=cuLA DSL: generated dispatch with {} variants",
        variants.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────

fn link_cuda_runtime(cuda_path: &Path) {
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

fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("compute_100"))
        .unwrap_or(false)
}

fn find_cuda() -> PathBuf {
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        if Path::new(p).join("bin/nvcc").exists() {
            return PathBuf::from(p);
        }
    }
    panic!("CUDA toolkit not found. Set CUDA_PATH env var.");
}

fn file_hash(path: &Path) -> Option<String> {
    use sha2::Digest;
    let content = std::fs::read(path).ok()?;
    let hash = sha2::Sha256::digest(&content);
    Some(hex::encode(hash)[..16].to_string())
}

fn has_obj_files(dir: &Path) -> bool {
    std::fs::read_dir(dir).ok().into_iter().flatten()
        .any(|e| e.ok().map(|e| e.path().extension().is_some_and(|x| x == "o")).unwrap_or(false))
}

fn clear_obj_files(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|x| x == "o") {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    let _ = std::fs::remove_file(dir.join("manifest.json"));
}

fn walkdir(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(walkdir(&path, ext));
            } else if path.extension().is_some_and(|e| e == ext) {
                result.push(path);
            }
        }
    }
    result
}

fn find_file_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().map_or(false, |n| n == name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, name) {
                return Some(found);
            }
        }
    }
    None
}

fn detect_gpu_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output().ok()?;
    if !output.status.success() { return None; }
    let cap = String::from_utf8_lossy(&output.stdout);
    let cap = cap.trim().lines().next()?;
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() == 2 {
        let major: u32 = parts[0].parse().ok()?;
        let minor: u32 = parts[1].parse().ok()?;
        Some(format!("sm_{}{}", major, minor))
    } else {
        None
    }
}

fn ensure_python_env(out_dir: &Path) -> Result<String, String> {
    // Check if system python has cutlass
    for candidate in ["python3", "python"] {
        if check_cutlass(candidate) {
            return Ok(candidate.to_string());
        }
    }

    // Create venv
    let venv_dir = out_dir.join("cula-venv");
    let venv_python = venv_dir.join("bin/python3");

    if check_cutlass(venv_python.to_str().unwrap_or("")) {
        return Ok(venv_python.to_string_lossy().into_owned());
    }

    println!("cargo:warning=cuLA: creating Python venv for DSL kernel compilation...");

    let venv_created = Command::new("uv")
        .args(["venv", venv_dir.to_str().unwrap(), "--python", "3.12"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !venv_created {
        let status = Command::new("python3")
            .args(["-m", "venv", venv_dir.to_str().unwrap()])
            .status()
            .map_err(|e| format!("Failed to create venv: {e}"))?;
        if !status.success() {
            return Err("Failed to create Python venv".to_string());
        }
    }

    // Detect CUDA version for PyTorch wheel
    let torch_index = detect_torch_cuda_index();

    println!("cargo:warning=cuLA: installing Python deps...");

    let venv_python_str = venv_python.to_string_lossy().to_string();
    let has_uv = Command::new("uv")
        .arg("--version").output()
        .map(|o| o.status.success()).unwrap_or(false);

    let base_pkgs = &["nvidia-cutlass-dsl>=4.4.2", "torch"];

    let status = if has_uv {
        let mut cmd = Command::new("uv");
        cmd.args(["pip", "install", "--python", &venv_python_str]);
        if let Some(ref idx) = torch_index {
            cmd.args(["--extra-index-url", idx]);
        }
        cmd.args(base_pkgs);
        cmd.status().map_err(|e| format!("uv pip install failed: {e}"))?
    } else {
        let pip = venv_dir.join("bin/pip");
        let mut cmd = Command::new(&pip);
        cmd.arg("install");
        if let Some(ref idx) = torch_index {
            cmd.args(["--extra-index-url", idx]);
        }
        cmd.args(base_pkgs);
        cmd.status().map_err(|e| format!("pip install failed: {e}"))?
    };

    if !status.success() {
        return Err("Failed to install Python deps".to_string());
    }

    if !check_cutlass(venv_python.to_str().unwrap_or("")) {
        return Err("cutlass not importable after install".to_string());
    }

    Ok(venv_python.to_string_lossy().into_owned())
}

fn check_cutlass(python: &str) -> bool {
    Command::new(python)
        .args(["-c", "import cutlass"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn detect_torch_cuda_index() -> Option<String> {
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
