"""Shared helpers for per-crate CuTeDSL AOT compile scripts.

Before this module, `cula/scripts/compile_kernels.py` and
`fa4/scripts/compile_kernels.py` each had their own copies of
`_verify_symbol`, manifest writing, script-hash invalidation,
ProcessPool worker plumbing, and the standard argparse CLI — about
~150 lines of near-identical glue per crate that drifted slightly
over time. Now both import from here.

## How consumers pick this up

Consumer crates' `build.rs` sets `PRELUDE_KB_SCRIPTS_DIR` to the
absolute path of this directory (computed at build time via
`prelude_kernelbuild::scripts_dir()`) before spawning the Python
compile script. The compile script then puts the dir on `sys.path`
and imports from here:

```python
import os, sys
sys.path.insert(0, os.environ["PRELUDE_KB_SCRIPTS_DIR"])
from dsl_driver import (
    verify_symbol, compute_script_hash, run_parallel,
    write_manifest, load_existing_manifest, standard_argparse,
)
```

We deliberately keep this as a single loose .py file (no package, no
install) to avoid dragging PyPI packaging into the build flow. Every
consumer crate still owns its own `compile_kernels.py` with the
per-kernel knowledge (variant matrix, `cute.compile` arg setup,
per-kernel result formatting); this file just hosts the common
plumbing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


# ─────────────────────────────────────────────────────────────────────
# Symbol verification
# ─────────────────────────────────────────────────────────────────────

def verify_symbol(obj_path: Path, expected_name: str) -> None:
    """Ensure `obj_path` exports `__tvm_ffi_{expected_name}`.

    When `cute.compile → export_to_c` runs inside an anonymous-lambda
    context, it sometimes emits `__tvm_ffi_func` as the exported
    wrapper symbol instead of the crate-specific mangled name. We use
    `objcopy --redefine-sym` to rename it in place so the dispatch
    table's `extern "C" { fn __tvm_ffi_<name>() }` declarations can
    still find it.

    All failures are logged but non-fatal — some exotic export paths
    land neither of the expected symbols, in which case the linker
    will surface the real problem later via "undefined reference".
    This function's job is only to catch the common anonymous-lambda
    case and rename it.
    """
    try:
        result = subprocess.run(
            ["nm", "-g", str(obj_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        symbols = result.stdout
        expected = f"__tvm_ffi_{expected_name}"
        if expected in symbols:
            print(f"    Symbol OK: {expected}")
        elif "__tvm_ffi_func" in symbols:
            print(f"    WARNING: found __tvm_ffi_func, renaming to {expected}")
            subprocess.run(
                ["objcopy", f"--redefine-sym=__tvm_ffi_func={expected}", str(obj_path)],
                check=True,
                timeout=10,
            )
            print("    Renamed OK")
        else:
            print(f"    WARNING: neither {expected} nor __tvm_ffi_func found in symbols")
    except Exception as e:
        print(f"    Symbol check skipped: {e}")


# ─────────────────────────────────────────────────────────────────────
# Script-hash cache invalidation
# ─────────────────────────────────────────────────────────────────────

def compute_script_hash(script_path: Path) -> str:
    """SHA-256 prefix (16 hex chars) of the compile script's bytes.

    Used as a cache key so any edit to `compile_kernels.py`
    invalidates the `.o` files from the previous build — this is the
    build.rs-side mechanism (`prelude_kernelbuild::nvcc::file_hash`)
    for detecting changes. Consumers store the hash in
    `manifest.json` under `script_hash`; build.rs compares the stored
    hash against the current file hash on each build and clears the
    `.o` files on mismatch.
    """
    return hashlib.sha256(script_path.read_bytes()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────
# Manifest IO
# ─────────────────────────────────────────────────────────────────────

def load_existing_manifest(manifest_path: Path) -> list[dict]:
    """Load `manifest.json`'s `variants` array, or return `[]` if the
    file is missing or corrupt.

    Used by multi-arch incremental compilation: a consumer calls this
    at the top of `main()`, filters out the variants for the current
    arch, compiles new ones, merges, then writes a fresh manifest.
    This way compiling for sm_100 after sm_90 doesn't drop the
    existing sm_90 entries.
    """
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            return data.get("variants", [])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return []


def write_manifest(
    manifest_path: Path,
    variants: list[dict],
    script_hash: str,
    archs: Optional[list[int]] = None,
) -> None:
    """Serialise `variants` + `script_hash` + optional `archs` +
    `cutlass_version` (if importable) into `manifest.json`.

    `archs` is optional — cula's dispatch codegen reads per-variant
    `arch` fields directly and ignores the top-level list, fa4 uses
    the list for multi-arch incremental book-keeping. Pass it when
    you care, leave as `None` otherwise.
    """
    manifest: dict[str, Any] = {
        "variants": variants,
        "script_hash": script_hash,
    }
    if archs is not None:
        manifest["archs"] = archs
    try:
        import cutlass  # type: ignore[import-not-found]
        manifest["cutlass_version"] = cutlass.__version__
    except (ImportError, AttributeError):
        pass

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
# Parallel / sequential task runner
# ─────────────────────────────────────────────────────────────────────

def run_parallel(
    tasks: Iterable[Any],
    worker_fn: Callable[[Any], Optional[dict]],
    max_workers: int = 1,
) -> list[dict]:
    """Run `worker_fn(task)` for each task and collect the non-None
    results.

    `max_workers == 1` runs sequentially (no process pool) — this is
    the right choice for CuTeDSL compiles because each call touches
    CUDA during placeholder-tensor setup, and forking after CUDA
    init blows up with *"Cannot re-initialize CUDA in forked
    subprocess"*. fa4 can get away with workers > 1 because its
    `_patch_flash_attn_import` defers CUDA init until after the pool
    fork; cuLA's kernels import cutlass eagerly so they're
    single-worker.

    `worker_fn` must return a dict (a variant manifest entry) or
    `None` to skip the result (e.g. a compile that failed but
    shouldn't abort the whole build).
    """
    tasks_list = list(tasks)

    if max_workers > 1:
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(worker_fn, t): t for t in tasks_list}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    # Sequential path — always safe, required for CUDA-touching compiles.
    results_seq: list[dict] = []
    for task in tasks_list:
        result = worker_fn(task)
        if result:
            results_seq.append(result)
    return results_seq


# ─────────────────────────────────────────────────────────────────────
# argparse boilerplate
# ─────────────────────────────────────────────────────────────────────

def standard_argparse(
    description: str,
    default_output_dir: Path,
    default_workers: int = 1,
    add_prototype: bool = True,
    add_arch: bool = True,
) -> argparse.ArgumentParser:
    """Build a pre-configured argparse parser with the common CLI
    shape every compile_kernels.py uses:

    * `--output-dir` → where `.o` files and `manifest.json` land
    * `-j, --workers` → parallelism for [`run_parallel`]
    * `--arch` → target SM arch (optional, some scripts auto-detect
      from env vars instead)
    * `--prototype` → compile a single smoke-test variant instead of
      the full matrix (optional, used by cuLA to sanity-check a
      minimal spec in < 30s)

    Consumers can add their own crate-specific flags on top of the
    returned parser before calling `parser.parse_args()`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Where to write .o files and manifest.json",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel workers (default: %(default)d). Each worker is a subprocess "
        "with its own CUDA context and compile cache.",
    )
    if add_arch:
        parser.add_argument(
            "--arch",
            type=int,
            default=None,
            help="SM arch (e.g. 90, 100). Default: auto-detect from env.",
        )
    if add_prototype:
        parser.add_argument(
            "--prototype",
            action="store_true",
            help="Compile a minimal smoke-test variant set instead of the full matrix.",
        )
    return parser
