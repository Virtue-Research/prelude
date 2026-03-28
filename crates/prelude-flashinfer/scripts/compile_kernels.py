#!/usr/bin/env python3
"""AOT compile FlashInfer kernels to .o files for static linking.

Usage:
    python compile_kernels.py [--flashinfer-src DIR] [--out-dir DIR] [-j N]
                              [--archs sm_80,sm_90] [--head-dims 64,96,128]
                              [--dtypes bf16] [--dry-run]

Generates:
  - .o files for each kernel variant
  - manifest.json listing all variants and their exported symbols
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Variant configuration ──────────────────────────────────────────────

DTYPE_MAP = {
    "bf16": ("nv_bfloat16", "bfloat16"),
    "fp16": ("nv_half", "float16"),
}

MASK_MODES = {
    0: "MaskMode::kNone",
    1: "MaskMode::kCausal",
    2: "MaskMode::kCustom",
    3: "MaskMode::kMultiItemScoring",
}

# All mask modes referenced by DISPATCH_MASK_MODE must be compiled.
NEEDED_MASK_MODES = [0, 1, 2, 3]


def variant_id(kind: str, backend: str, dtype: str, hdim_qk: int, hdim_vo: int,
               swa: bool = False, softcap: bool = False) -> str:
    """Generate a unique variant identifier used for symbol naming."""
    name = f"fi_{kind}_{backend}_{dtype}_h{hdim_qk}"
    if hdim_qk != hdim_vo:
        name += f"v{hdim_vo}"
    if swa:
        name += "_swa"
    if softcap:
        name += "_cap"
    return name


# ── Source generation ──────────────────────────────────────────────────

def generate_batch_decode_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim_qk: int, hdim_vo: int,
    swa: bool, softcap: bool,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for a batch_decode variant.
    Returns (source_paths, extra_nvcc_flags).
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    # Render config .inc
    with open(csrc / "batch_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    variant_name = (
        f"DefaultAttention<false, {str(swa).lower()}, "
        f"{str(softcap).lower()}, false>"
    )
    kwargs = {
        "variant_decl": "#include<flashinfer/attention/variants.cuh>",
        "variant_name": variant_name,
        "dtype_q": dtype_c, "dtype_kv": dtype_c, "dtype_o": dtype_c,
        "idtype": "int32_t",
        "head_dim_qk": hdim_qk, "head_dim_vo": hdim_vo,
        "pos_encoding_mode": "PosEncodingMode::kNone",
        "use_sliding_window": str(swa).lower(),
        "use_logits_soft_cap": str(softcap).lower(),
        **dict(zip(
            ("additional_params_decl", "additional_func_params", "additional_params_setter"),
            _default_decode_params(),
        )),
    }
    config_str = config_templ.render(**kwargs)
    (out / "batch_decode_config.inc").write_text(config_str)

    # Render kernel instantiation
    with open(csrc / "batch_decode_kernel_inst.jinja") as f:
        kernel_templ = jinja2.Template(f.read())
    kernel_src = kernel_templ.render(**kwargs)
    (out / "batch_decode_kernel.cu").write_text(kernel_src)

    sources = [out / "batch_decode_kernel.cu"]

    # batch_decode.cu defines BatchDecodeWithPagedKVCacheRun — rename per variant
    # to avoid duplicate symbols across AOT variants.
    renames = {
        "BatchDecodeWithPagedKVCacheRun": f"{vid}_BatchDecodeWithPagedKVCacheRun",
        "BatchDecodeWithPagedKVCachePlan": f"{vid}_BatchDecodeWithPagedKVCachePlan",
    }
    _copy_with_renames(csrc / "batch_decode.cu", out / "batch_decode.cu", renames)
    sources.append(out / "batch_decode.cu")

    # Generate modified binding with variant-specific symbol names
    binding_src = _generate_renamed_binding(
        csrc / "batch_decode_jit_binding.cu",
        "batch_decode_config.inc",
        {
            "plan": f"__tvm_ffi_{vid}_plan",
            "run": f"__tvm_ffi_{vid}_run",
        },
    )
    # Also rename the internal function calls in the binding
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    binding_path = out / "batch_decode_binding.cu"
    binding_path.write_text(binding_src)
    sources.append(binding_path)

    return sources, []


def generate_batch_prefill_fa2_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim_qk: int, hdim_vo: int,
    swa: bool, softcap: bool,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for a batch_prefill FA2 variant."""
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    variant_name = (
        f"DefaultAttention<false, {str(swa).lower()}, "
        f"{str(softcap).lower()}, false>"
    )
    kwargs = {
        "variant_decl": "#include<flashinfer/attention/variants.cuh>",
        "variant_name": variant_name,
        "dtype_q": dtype_c, "dtype_kv": dtype_c, "dtype_o": dtype_c,
        "idtype": "int32_t",
        "head_dim_qk": hdim_qk, "head_dim_vo": hdim_vo,
        "pos_encoding_mode": "PosEncodingMode::kNone",
        "use_sliding_window": str(swa).lower(),
        "use_logits_soft_cap": str(softcap).lower(),
        "use_fp16_qk_reduction": "false",
        **dict(zip(
            ("additional_params_decl", "additional_func_params", "additional_params_setter"),
            _default_prefill_params(),
        )),
    }

    # Config
    with open(csrc / "batch_prefill_customize_config.jinja") as f:
        config_str = jinja2.Template(f.read()).render(**kwargs)
    (out / "batch_prefill_config.inc").write_text(config_str)

    # Kernel instantiations for needed mask modes
    with open(csrc / "batch_prefill_paged_kernel_inst.jinja") as f:
        paged_templ = jinja2.Template(f.read())
    with open(csrc / "batch_prefill_ragged_kernel_inst.jinja") as f:
        ragged_templ = jinja2.Template(f.read())

    sources = []
    for mm in NEEDED_MASK_MODES:
        for templ, prefix in [(paged_templ, "paged"), (ragged_templ, "ragged")]:
            fname = f"batch_prefill_{prefix}_kernel_mask_{mm}.cu"
            src = templ.render(mask_mode=MASK_MODES[mm], **kwargs)
            (out / fname).write_text(src)
            sources.append(out / fname)

    # batch_prefill.cu defines BatchPrefillWith{Ragged,Paged}KVCacheRun — rename per variant
    renames = {
        "BatchPrefillWithKVCachePlan": f"{vid}_BatchPrefillWithKVCachePlan",
        "BatchPrefillWithRaggedKVCacheRun": f"{vid}_BatchPrefillWithRaggedKVCacheRun",
        "BatchPrefillWithPagedKVCacheRun": f"{vid}_BatchPrefillWithPagedKVCacheRun",
    }
    _copy_with_renames(csrc / "batch_prefill.cu", out / "batch_prefill.cu", renames)
    sources.append(out / "batch_prefill.cu")

    # Generate modified binding
    binding_src = _generate_renamed_binding(
        csrc / "batch_prefill_jit_binding.cu",
        "batch_prefill_config.inc",
        {
            "plan": f"__tvm_ffi_{vid}_plan",
            "ragged_run": f"__tvm_ffi_{vid}_ragged_run",
            "paged_run": f"__tvm_ffi_{vid}_paged_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_prefill_binding.cu").write_text(binding_src)
    sources.append(out / "batch_prefill_binding.cu")

    return sources, []


def generate_batch_prefill_fa3_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim_qk: int, hdim_vo: int,
    swa: bool, softcap: bool,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for a batch_prefill FA3 (SM90) variant."""
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    variant_name = f"DefaultAttention<{str(softcap).lower()}>"
    kwargs = {
        "variant_decl": "#include<flashinfer/attention/hopper/variants.cuh>",
        "variant_name": variant_name,
        "dtype_q": dtype_c, "dtype_kv": dtype_c, "dtype_o": dtype_c,
        "idtype": "int32_t",
        "head_dim_qk": hdim_qk, "head_dim_vo": hdim_vo,
        "pos_encoding_mode": "PosEncodingMode::kNone",
        "use_sliding_window": str(swa).lower(),
        "use_logits_soft_cap": str(softcap).lower(),
        "use_fp16_qk_reduction": "false",
        **dict(zip(
            ("additional_params_decl", "additional_func_params", "additional_params_setter"),
            _default_prefill_sm90_params(),
        )),
    }

    # Config
    with open(csrc / "batch_prefill_sm90_customize_config.jinja") as f:
        config_str = jinja2.Template(f.read()).render(**kwargs)
    (out / "batch_prefill_sm90_config.inc").write_text(config_str)

    # Kernel instantiations
    with open(csrc / "batch_prefill_paged_sm90_kernel_inst.jinja") as f:
        paged_templ = jinja2.Template(f.read())
    with open(csrc / "batch_prefill_ragged_sm90_kernel_inst.jinja") as f:
        ragged_templ = jinja2.Template(f.read())

    sources = []
    for mm in NEEDED_MASK_MODES:
        for templ, prefix in [(paged_templ, "paged"), (ragged_templ, "ragged")]:
            fname = f"batch_prefill_{prefix}_sm90_kernel_mask_{mm}.cu"
            src = templ.render(mask_mode=MASK_MODES[mm], **kwargs)
            (out / fname).write_text(src)
            sources.append(out / fname)

    # batch_prefill_sm90.cu defines BatchPrefillWith{Ragged,Paged}KVCacheSM90Run — rename per variant
    renames = {
        "BatchPrefillWithKVCacheSM90Plan": f"{vid}_BatchPrefillWithKVCacheSM90Plan",
        "BatchPrefillWithRaggedKVCacheSM90Run": f"{vid}_BatchPrefillWithRaggedKVCacheSM90Run",
        "BatchPrefillWithPagedKVCacheSM90Run": f"{vid}_BatchPrefillWithPagedKVCacheSM90Run",
    }
    _copy_with_renames(csrc / "batch_prefill_sm90.cu", out / "batch_prefill_sm90.cu", renames)
    sources.append(out / "batch_prefill_sm90.cu")

    # Generate modified binding
    binding_src = _generate_renamed_binding(
        csrc / "batch_prefill_sm90_jit_binding.cu",
        "batch_prefill_sm90_config.inc",
        {
            "plan": f"__tvm_ffi_{vid}_plan",
            "ragged_run": f"__tvm_ffi_{vid}_ragged_run",
            "paged_run": f"__tvm_ffi_{vid}_paged_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_prefill_sm90_binding.cu").write_text(binding_src)
    sources.append(out / "batch_prefill_sm90_binding.cu")

    return sources, ["-gencode", "arch=compute_90a,code=sm_90a"]


def generate_page_sources(
    fi_src: Path, gen_dir: Path,
) -> Tuple[List[Path], List[str]]:
    """Generate page.cu (append_paged_kv_cache) with renamed symbols."""
    csrc = fi_src / "csrc"
    out = gen_dir / "fi_page"
    out.mkdir(parents=True, exist_ok=True)

    # Read original and add TVM FFI export at the bottom
    original = (csrc / "page.cu").read_text()
    # The original page.cu doesn't use TVM_FFI_DLL_EXPORT_TYPED_FUNC.
    # The binding is in flashinfer_page_binding.cu. Let's check:
    binding_path = csrc / "flashinfer_page_binding.cu"
    if binding_path.exists():
        shutil.copy2(csrc / "page.cu", out / "page.cu")
        shutil.copy2(binding_path, out / "flashinfer_page_binding.cu")
        return [out / "page.cu", out / "flashinfer_page_binding.cu"], []
    else:
        # page.cu has the functions directly, we'll just compile it
        shutil.copy2(csrc / "page.cu", out / "page.cu")
        return [out / "page.cu"], []


# ── Symbol renaming ────────────────────────────────────────────────────

def _copy_with_renames(src: Path, dst: Path, renames: Dict[str, str]) -> None:
    """Copy a .cu file, prepending #define macros to rename symbols."""
    defines = "\n".join(f"#define {old} {new}" for old, new in renames.items())
    original = src.read_text()
    dst.write_text(f"// AOT variant rename\n{defines}\n\n{original}")


def _generate_renamed_binding(
    original_path: Path, config_inc: str,
    symbol_map: Dict[str, str],
) -> str:
    """Read the original binding .cu and replace TVM_FFI_DLL_EXPORT_TYPED_FUNC
    with variant-specific export names.

    Instead of: TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, FuncName)
    We generate: extern "C" { int __tvm_ffi_VARIANT_plan(...) { ... } }
    by using a macro rename trick.
    """
    src = original_path.read_text()

    # Replace the config include to point to local generated config
    # (the original includes e.g. "batch_decode_config.inc" which is in the same dir)

    # Replace each TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, Func) with renamed version
    for orig_name, new_symbol in symbol_map.items():
        # Extract just the function name part from __tvm_ffi_VARIANT_plan
        new_export = new_symbol.replace("__tvm_ffi_", "")
        old_line = f'TVM_FFI_DLL_EXPORT_TYPED_FUNC({orig_name},'
        new_line = f'TVM_FFI_DLL_EXPORT_TYPED_FUNC({new_export},'
        src = src.replace(old_line, new_line)

    return src


# ── Default additional params (matching FlashInfer's DefaultAttention) ──
# Must match flashinfer/jit/attention/utils.py::generate_additional_params()
# - params_decl: raw pointer types in the kernel Params struct
# - func_params: Optional<ffi::Tensor> in the TVM FFI binding signature
# - params_setter: extract raw pointers from tensors into params struct


def _generate_additional_params(
    tensor_names: List[str],
    tensor_dtypes: List[str],
    scalar_names: List[str],
    scalar_dtypes: List[str],
    is_sm90: bool = False,
) -> Tuple[str, str, str]:
    """Generate (decl, func_params, setter) matching FlashInfer's format."""
    # Decl: raw pointers + scalars inside AdditionalParams struct
    decl_lines = []
    for dtype, var in zip(tensor_dtypes, tensor_names):
        decl_lines.append(f"{dtype}* {var};")
    for dtype, var in zip(scalar_dtypes, scalar_names):
        decl_lines.append(f"{dtype} {var};")
    decl = "\n".join(decl_lines)

    # Func params: Optional<ffi::Tensor> for maybe_* tensors, ffi::Tensor otherwise
    func_parts = []
    for var in tensor_names:
        if var.startswith("maybe"):
            func_parts.append(f"Optional<ffi::Tensor> {var}")
        else:
            func_parts.append(f"ffi::Tensor {var}")
    for dtype, var in zip(scalar_dtypes, scalar_names):
        func_parts.append(f"{dtype} {var}")
    func_params = ", " + ", ".join(func_parts) if func_parts else ""

    # Setter: extract raw pointers, with backslash line continuation
    prefix = "params.additional_params." if is_sm90 else "params."
    setter_lines = []
    for dtype, var in zip(tensor_dtypes, tensor_names):
        if var.startswith("maybe"):
            setter_lines.append(
                f"{prefix}{var} = {var} ? static_cast<{dtype}*>({var}.value().data_ptr()): nullptr;"
            )
        else:
            setter_lines.append(
                f"{prefix}{var} = static_cast<{dtype}*>({var}.data_ptr());"
            )
    for var in scalar_names:
        setter_lines.append(f"{prefix}{var} = {var};")
    setter = " \\\n".join(setter_lines)

    return decl, func_params, setter


def _default_decode_params():
    return _generate_additional_params(
        tensor_names=["maybe_alibi_slopes"],
        tensor_dtypes=["float"],
        scalar_names=["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        scalar_dtypes=["double", "double", "double", "double"],
    )


def _default_prefill_params():
    return _generate_additional_params(
        tensor_names=[
            "maybe_custom_mask", "maybe_mask_indptr", "maybe_alibi_slopes",
            "maybe_prefix_len_ptr", "maybe_token_pos_in_items_ptr", "maybe_max_item_len_ptr",
        ],
        tensor_dtypes=["uint8_t", "int32_t", "float", "uint32_t", "uint16_t", "uint16_t"],
        scalar_names=["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta",
                       "token_pos_in_items_len"],
        scalar_dtypes=["double", "double", "double", "double", "int64_t"],
    )


def _default_prefill_sm90_params():
    return _generate_additional_params(
        tensor_names=[
            "maybe_prefix_len_ptr", "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr", "maybe_scale_v",
        ],
        tensor_dtypes=["uint32_t", "uint16_t", "uint16_t", "float"],
        scalar_names=["logits_soft_cap", "sm_scale", "scale_v_scalar"],
        scalar_dtypes=["double", "double", "double"],
        is_sm90=True,
    )


# ── Compilation ────────────────────────────────────────────────────────

def compile_source(
    src: Path, out_dir: Path, fi_src: Path,
    arch_flags: List[str], extra_flags: List[str],
) -> Optional[Path]:
    """Compile a single .cu file to .o using nvcc."""
    obj = out_dir / (src.stem + ".o")
    if obj.exists() and obj.stat().st_mtime > src.stat().st_mtime:
        return obj  # up-to-date

    include_dirs = [
        str(fi_src / "include"),
        str(fi_src / "csrc"),
        str(src.parent),  # for config .inc files
    ]
    # CUTLASS headers (needed for SM90/Hopper kernels)
    cutlass_base = fi_src / "3rdparty" / "cutlass"
    for sub in ["include", "tools/util/include"]:
        p = cutlass_base / sub
        if p.exists():
            include_dirs.append(str(p))

    # Find TVM FFI headers
    tvm_ffi_include = _find_tvm_ffi_include(fi_src)
    if tvm_ffi_include:
        include_dirs.append(str(tvm_ffi_include))

    cmd = [
        "nvcc", "-c", str(src), "-o", str(obj),
        "-std=c++17", "--expt-relaxed-constexpr",
        "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
        "-O3", "--use_fast_math",
        "-Xcompiler", "-fPIC",
    ]
    for d in include_dirs:
        cmd += ["-I", d]
    cmd += arch_flags
    cmd += extra_flags

    print(f"  nvcc {src.name} -> {obj.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {src.name}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None
    return obj


def _find_tvm_ffi_include(fi_src: Path) -> Optional[Path]:
    """Find TVM FFI include directory."""
    # Check if flashinfer has 3rdparty/tvm_ffi
    p = fi_src / "3rdparty" / "tvm_ffi" / "include"
    if p.exists():
        return p
    # Check FA4's vendored copy (same workspace)
    script_dir = Path(__file__).resolve().parent.parent  # prelude-flashinfer crate
    fa4_vendor = script_dir.parent / "prelude-flash-attn-v4" / "vendor" / "tvm_ffi" / "include"
    if fa4_vendor.exists():
        return fa4_vendor
    # Check installed tvm_ffi package
    try:
        import tvm_ffi
        pkg_dir = Path(tvm_ffi.__file__).parent
        inc = pkg_dir / "include"
        if inc.exists():
            return inc
    except ImportError:
        pass
    return None


# ── Variant matrix ─────────────────────────────────────────────────────

def build_variant_matrix(
    archs: List[int], head_dims: List[int], dtypes: List[str],
) -> List[dict]:
    """Build the list of kernel variants to compile.

    Aligned with FlashInfer's official default config (flashinfer/aot.py):
      - FA2: decode + prefill, swa × softcap combinations
      - FA3: prefill only (decode uses prefill on SM90), swa × softcap
      - Asymmetric head_dim (192,128) for FA3
      - FP16 + BF16

    FP8 E4M3 KV cache variants are NOT included (requires mixed-precision
    dtype_q != dtype_kv support in our compilation infrastructure).
    """
    variants = []
    sm80_flags = ["-gencode", "arch=compute_80,code=compute_80",
                  "-gencode", "arch=compute_90,code=sm_90"]
    sm90_flags = ["-gencode", "arch=compute_90a,code=sm_90a"]
    has_sm90 = any(a >= 90 for a in archs)

    swa_softcap_combos = [(False, False), (True, False), (False, True), (True, True)]

    # Standard symmetric head_dim variants
    for dtype in dtypes:
        dtype_c, dtype_name = DTYPE_MAP[dtype]
        for hdim in head_dims:
            # Batch decode FA2
            for swa, softcap in swa_softcap_combos:
                vid = variant_id("decode", "fa2", dtype, hdim, hdim, swa=swa, softcap=softcap)
                variants.append({
                    "vid": vid, "kind": "decode", "backend": "fa2",
                    "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                    "hdim_qk": hdim, "hdim_vo": hdim,
                    "swa": swa, "softcap": softcap,
                    "arch_flags": sm80_flags,
                })

            # Batch prefill FA2
            for swa, softcap in swa_softcap_combos:
                vid = variant_id("prefill", "fa2", dtype, hdim, hdim, swa=swa, softcap=softcap)
                variants.append({
                    "vid": vid, "kind": "prefill_fa2", "backend": "fa2",
                    "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                    "hdim_qk": hdim, "hdim_vo": hdim,
                    "swa": swa, "softcap": softcap,
                    "arch_flags": sm80_flags,
                })

            # Batch prefill FA3 (SM90+ only, HEAD_DIM_VO must be 64/128/256)
            if has_sm90 and hdim in (64, 128, 256):
                for swa, softcap in swa_softcap_combos:
                    vid = variant_id("prefill", "fa3", dtype, hdim, hdim, swa=swa, softcap=softcap)
                    variants.append({
                        "vid": vid, "kind": "prefill_fa3", "backend": "fa3",
                        "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                        "hdim_qk": hdim, "hdim_vo": hdim,
                        "swa": swa, "softcap": softcap,
                        "arch_flags": sm90_flags,
                    })

    # Asymmetric head_dim: FA3 (192,128) — used by some models
    if has_sm90:
        for dtype in dtypes:
            dtype_c, dtype_name = DTYPE_MAP[dtype]
            for swa, softcap in swa_softcap_combos:
                vid = variant_id("prefill", "fa3", dtype, 192, 128, swa=swa, softcap=softcap)
                variants.append({
                    "vid": vid, "kind": "prefill_fa3", "backend": "fa3",
                    "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                    "hdim_qk": 192, "hdim_vo": 128,
                    "swa": swa, "softcap": softcap,
                    "arch_flags": sm90_flags,
                })

    return variants


# ── Main ───────────────────────────────────────────────────────────────

def _do_compile(job, obj_dir, fi_src):
    """Module-level wrapper for ProcessPoolExecutor (must be picklable)."""
    src, arch_flags, extra_flags, vinfo = job
    var_obj_dir = obj_dir / vinfo["vid"]
    var_obj_dir.mkdir(parents=True, exist_ok=True)
    return compile_source(src, var_obj_dir, fi_src, arch_flags, extra_flags)


def main():
    parser = argparse.ArgumentParser(description="AOT compile FlashInfer kernels")
    parser.add_argument("--flashinfer-src", type=Path,
                        default=os.environ.get("FLASHINFER_SRC"),
                        help="Path to FlashInfer source")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for .o files and manifest")
    parser.add_argument("-j", "--workers", type=int, default=4,
                        help="Parallel compilation workers")
    parser.add_argument("--archs", type=str, default="sm_80,sm_90",
                        help="Target SM architectures")
    parser.add_argument("--head-dims", type=str, default="64,96,128,192,256",
                        help="Head dimensions to compile (must match build.rs defaults)")
    parser.add_argument("--dtypes", type=str, default="bf16,fp16",
                        help="Data types to compile (must match build.rs defaults)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print variants without compiling")
    args = parser.parse_args()

    if not args.flashinfer_src:
        print("ERROR: --flashinfer-src or FLASHINFER_SRC env var required",
              file=sys.stderr)
        sys.exit(1)

    fi_src = args.flashinfer_src.resolve()
    if not (fi_src / "csrc").exists():
        print(f"ERROR: {fi_src}/csrc not found", file=sys.stderr)
        sys.exit(1)

    # Add flashinfer to Python path for jinja2 template access
    sys.path.insert(0, str(fi_src))

    out_dir = args.out_dir.resolve()
    gen_dir = out_dir / "gen"
    obj_dir = out_dir / "obj"
    gen_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    archs = [int(a.replace("sm_", "")) for a in args.archs.split(",")]
    head_dims = [int(d) for d in args.head_dims.split(",")]
    dtypes = args.dtypes.split(",")

    variants = build_variant_matrix(archs, head_dims, dtypes)
    print(f"FlashInfer AOT: {len(variants)} variants, archs={archs}, "
          f"head_dims={head_dims}, dtypes={dtypes}")

    if args.dry_run:
        for v in variants:
            print(f"  {v['vid']} ({v['kind']})")
        return

    # Phase 1: Generate source files
    print("Phase 1: Generating source files...")
    compile_jobs = []  # (source_path, arch_flags, extra_flags, variant_info)

    for v in variants:
        if v["kind"] == "decode":
            sources, extra = generate_batch_decode_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
                v["swa"], v["softcap"],
            )
        elif v["kind"] == "prefill_fa2":
            sources, extra = generate_batch_prefill_fa2_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
                v["swa"], v["softcap"],
            )
        elif v["kind"] == "prefill_fa3":
            sources, extra = generate_batch_prefill_fa3_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
                v["swa"], v["softcap"],
            )
        else:
            continue

        for src in sources:
            compile_jobs.append((src, v["arch_flags"], extra, v))

    # Also add page module
    page_sources, page_extra = generate_page_sources(fi_src, gen_dir)
    for src in page_sources:
        compile_jobs.append((src, ["-gencode", "arch=compute_80,code=compute_80", "-gencode", "arch=compute_90,code=sm_90"], page_extra,
                             {"vid": "fi_page", "kind": "page"}))

    print(f"Phase 2: Compiling {len(compile_jobs)} source files with {args.workers} workers...")

    # Phase 2: Compile
    obj_files = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_do_compile, job, obj_dir, fi_src): job for job in compile_jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            if result is None:
                failed.append(job[0].name)
            else:
                obj_files.append(str(result))

    if failed:
        print(f"\nFAILED to compile {len(failed)} files:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Write manifest
    manifest = {"variants": [], "objects": obj_files}
    for v in variants:
        entry = {
            "vid": v["vid"],
            "kind": v["kind"],
            "backend": v["backend"],
            "dtype": v["dtype"],
            "hdim_qk": v["hdim_qk"],
            "hdim_vo": v["hdim_vo"],
            "swa": v["swa"],
            "softcap": v["softcap"],
        }
        if v["kind"] == "decode":
            entry["symbols"] = {
                "plan": f"__tvm_ffi_{v['vid']}_plan",
                "run": f"__tvm_ffi_{v['vid']}_run",
            }
        elif v["kind"].startswith("prefill"):
            entry["symbols"] = {
                "plan": f"__tvm_ffi_{v['vid']}_plan",
                "ragged_run": f"__tvm_ffi_{v['vid']}_ragged_run",
                "paged_run": f"__tvm_ffi_{v['vid']}_paged_run",
            }
        manifest["variants"].append(entry)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(obj_files)} objects, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
