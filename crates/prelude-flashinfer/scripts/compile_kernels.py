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
    "e4m3": ("__nv_fp8_e4m3", "fp8_e4m3"),
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


def generate_batch_prefill_fp8_fa3_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim: int,
    swa: bool,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for an FP8 E4M3 prefill FA3 (SM90) variant.

    FP8 prefill uses the same config template as FA3 but with:
      - FP8 kernel inst templates (batch_prefill_fp8_{paged,ragged}_sm90_kernel_inst.jinja)
      - FP8 wrapper (batch_prefill_fp8_sm90.cu)
      - DefaultFP8Attention variant (per-head Q/K/V scales)
      - Symmetric head_dim only (HEAD_DIM_QK == HEAD_DIM_VO)
      - DTypeQ == DTypeKV (both FP8)
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    # FP8 output dtype: bf16 if input is e4m3 with bf16 output, but the CUDA
    # type for output is determined by the "output" dtype.  For FP8 kernels the
    # output is always bf16 or fp16.
    # We decide: e4m3 Q/KV → bf16 output (most common inference path).
    dtype_o_c = "nv_bfloat16"

    variant_name = "DefaultFP8Attention"
    kwargs = {
        "variant_decl": "#include<flashinfer/attention/hopper/variants.cuh>",
        "variant_name": variant_name,
        "dtype_q": dtype_c, "dtype_kv": dtype_c, "dtype_o": dtype_o_c,
        "idtype": "int32_t",
        "head_dim_qk": hdim, "head_dim_vo": hdim,  # must be symmetric for FP8
        "pos_encoding_mode": "PosEncodingMode::kNone",
        "use_sliding_window": str(swa).lower(),
        "use_logits_soft_cap": "false",  # FP8 variant doesn't support logits_soft_cap
        "use_fp16_qk_reduction": "false",
        **dict(zip(
            ("additional_params_decl", "additional_func_params", "additional_params_setter"),
            _default_prefill_fp8_sm90_params(),
        )),
    }

    # Config (same template as FA3)
    with open(csrc / "batch_prefill_sm90_customize_config.jinja") as f:
        config_str = jinja2.Template(f.read()).render(**kwargs)
    (out / "batch_prefill_sm90_config.inc").write_text(config_str)

    # FP8-specific kernel instantiations
    with open(csrc / "batch_prefill_fp8_paged_sm90_kernel_inst.jinja") as f:
        paged_templ = jinja2.Template(f.read())
    with open(csrc / "batch_prefill_fp8_ragged_sm90_kernel_inst.jinja") as f:
        ragged_templ = jinja2.Template(f.read())

    sources = []
    for mm in NEEDED_MASK_MODES:
        for templ, prefix in [(paged_templ, "paged"), (ragged_templ, "ragged")]:
            fname = f"batch_prefill_fp8_{prefix}_sm90_kernel_mask_{mm}.cu"
            src = templ.render(mask_mode=MASK_MODES[mm], **kwargs)
            (out / fname).write_text(src)
            sources.append(out / fname)

    # FP8 wrapper (batch_prefill_fp8_sm90.cu) — rename per variant
    renames = {
        "BatchPrefillWithKVCacheSM90Plan": f"{vid}_BatchPrefillWithKVCacheSM90Plan",
        "BatchPrefillWithRaggedKVCacheSM90Run": f"{vid}_BatchPrefillWithRaggedKVCacheSM90Run",
        "BatchPrefillWithPagedKVCacheSM90Run": f"{vid}_BatchPrefillWithPagedKVCacheSM90Run",
    }
    _copy_with_renames(csrc / "batch_prefill_fp8_sm90.cu", out / "batch_prefill_fp8_sm90.cu", renames)
    sources.append(out / "batch_prefill_fp8_sm90.cu")

    # Binding (same as FA3)
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
    (out / "batch_prefill_fp8_sm90_binding.cu").write_text(binding_src)
    sources.append(out / "batch_prefill_fp8_sm90_binding.cu")

    return sources, ["-gencode", "arch=compute_90a,code=sm_90a"]


def generate_batch_decode_mla_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str,
    head_dim_ckv: int, head_dim_kpe: int,
    swa: bool, softcap: bool,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for a batch_decode MLA variant (SM80 cute backend)."""
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    # qo_tile_len: 128 for ckv<=256, 64 otherwise
    qo_tile_len = 128 if head_dim_ckv <= 256 else 64

    variant_name = (
        f"DefaultAttention<false, {str(swa).lower()}, "
        f"{str(softcap).lower()}, false>"
    )
    with open(csrc / "batch_decode_mla_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_c,
        dtype_idx="int32_t",
        use_sliding_window=str(swa).lower(),
        use_logits_soft_cap=str(softcap).lower(),
        head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe,
        qo_tile_len=qo_tile_len,
    )
    (out / "mla_config.inc").write_text(config_str)

    # cute SM80 backend only supports FP16; use standard plan/run for BF16
    sources = []
    renames = {
        "BatchDecodeWithPagedKVCachePlanMLA": f"{vid}_BatchDecodeWithPagedKVCachePlanMLA",
        "BatchDecodeWithPagedKVCacheRunMLA": f"{vid}_BatchDecodeWithPagedKVCacheRunMLA",
    }
    if dtype_c == "nv_half":
        # FP16: use cute SM80 (single file with plan + run)
        _copy_with_renames(csrc / "batch_decode_mla_cute_sm80.cu",
                           out / "batch_decode_mla_cute_sm80.cu", renames)
        sources.append(out / "batch_decode_mla_cute_sm80.cu")
    else:
        # BF16: use standard plan + run
        _copy_with_renames(csrc / "batch_decode_mla_plan.cu",
                           out / "batch_decode_mla_plan.cu", renames)
        _copy_with_renames(csrc / "batch_decode_mla_run.cu",
                           out / "batch_decode_mla_run.cu", renames)
        sources.append(out / "batch_decode_mla_plan.cu")
        sources.append(out / "batch_decode_mla_run.cu")

    # Binding
    binding_src = _generate_renamed_binding(
        csrc / "batch_decode_mla_binding.cu",
        "mla_config.inc",
        {
            "plan": f"__tvm_ffi_{vid}_plan",
            "run": f"__tvm_ffi_{vid}_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_decode_mla_binding.cu").write_text(binding_src)
    sources.append(out / "batch_decode_mla_binding.cu")

    return sources, []


def generate_batch_mla_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str,
    head_dim_ckv: int, head_dim_kpe: int,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for batch MLA paged attention (FA2 backend)."""
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    with open(csrc / "batch_mla_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_c,
        dtype_idx="int32_t",
        head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe,
    )
    (out / "batch_mla_config.inc").write_text(config_str)

    # Plan + Run source
    sources = []
    renames = {
        "BatchMLAPagedAttentionPlan": f"{vid}_BatchMLAPagedAttentionPlan",
        "BatchMLAPagedAttentionRun": f"{vid}_BatchMLAPagedAttentionRun",
    }
    _copy_with_renames(csrc / "batch_mla_plan.cu", out / "batch_mla_plan.cu", renames)
    _copy_with_renames(csrc / "batch_mla_run.cu", out / "batch_mla_run.cu", renames)
    sources.append(out / "batch_mla_plan.cu")
    sources.append(out / "batch_mla_run.cu")

    # Binding
    binding_src = _generate_renamed_binding(
        csrc / "batch_mla_binding.cu",
        "batch_mla_config.inc",
        {
            "plan": f"__tvm_ffi_{vid}_plan",
            "run": f"__tvm_ffi_{vid}_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_mla_binding.cu").write_text(binding_src)
    sources.append(out / "batch_mla_binding.cu")

    return sources, []


def _generate_activation_source(fi_src: Path, gen_dir: Path) -> Tuple[List[Path], List[str], dict]:
    """Generate a combined activation .cu file for AOT compilation.

    Creates silu_and_mul, gelu_and_mul, gelu_tanh_and_mul as TVM FFI functions.
    These are JIT-generated in upstream FlashInfer — we create them statically.
    """
    out = gen_dir / "fi_activation"
    out.mkdir(parents=True, exist_ok=True)

    # Find tvm_ffi_utils.h location
    tvm_utils_include = fi_src / "csrc"

    activation_cu = r'''
#include <flashinfer/activation.cuh>
#include <cuda_runtime.h>
#include "tvm_ffi_utils.h"

using namespace flashinfer;

// ── Activation function definitions ──────────────────────────────

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}

__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}

// ── Launcher template ────────────────────────────────────────────

template <float (*Activation)(const float&)>
static void launch_act_and_mul(TensorView out, TensorView input) {
  int d = input.size(input.ndim() - 1) / 2;
  int64_t num_tokens = input.numel() / input.size(input.ndim() - 1);

  cudaSetDevice(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 grid(num_tokens);
    dim3 block(std::min((uint32_t)(d / vec_size), 1024U));

    auto kernel = activation::act_and_mul_kernel<c_type, Activation>;
    kernel<<<grid, block, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()),
        static_cast<const c_type*>(input.data_ptr()), d);
    return true;
  });
}

// ── TVM FFI exports ──────────────────────────────────────────────

void silu_and_mul(TensorView out, TensorView input) {
  launch_act_and_mul<silu>(out, input);
}

void gelu_and_mul(TensorView out, TensorView input) {
  launch_act_and_mul<gelu>(out, input);
}

void gelu_tanh_and_mul(TensorView out, TensorView input) {
  launch_act_and_mul<gelu_tanh>(out, input);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul, silu_and_mul);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gelu_and_mul, gelu_and_mul);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gelu_tanh_and_mul, gelu_tanh_and_mul);
'''

    src_path = out / "activation.cu"
    src_path.write_text(activation_cu)

    symbols = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
    vinfo = {
        "vid": "fi_activation", "kind": "activation",
        "symbols": {s: f"__tvm_ffi_{s}" for s in symbols},
    }
    return [src_path], [], vinfo


def generate_utility_sources(
    fi_src: Path, gen_dir: Path,
) -> List[Tuple[List[Path], List[str], dict]]:
    """Generate source files for non-templated utility kernels.

    Returns list of (sources, extra_flags, variant_info) tuples.
    """
    csrc = fi_src / "csrc"
    results = []

    # Each utility module: (dir_name, source_files, binding_file, kind, symbols)
    modules = [
        ("fi_page", ["page.cu"], "flashinfer_page_binding.cu", "page",
         ["append_paged_kv_cache", "append_paged_mla_kv_cache"]),
        ("fi_sampling", ["sampling.cu", "renorm.cu"], "flashinfer_sampling_binding.cu", "sampling",
         ["softmax", "sampling_from_probs", "sampling_from_logits",
          "top_k_sampling_from_probs", "top_p_sampling_from_probs",
          "min_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
          "top_k_renorm_probs", "top_p_renorm_probs", "top_k_mask_logits",
          "chain_speculative_sampling"]),
        ("fi_norm", ["norm.cu"], "flashinfer_norm_binding.cu", "norm",
         ["rmsnorm", "fused_add_rmsnorm", "gemma_rmsnorm", "gemma_fused_add_rmsnorm",
          "rmsnorm_quant", "fused_add_rmsnorm_quant"]),
        ("fi_rope", ["rope.cu"], "flashinfer_rope_binding.cu", "rope",
         ["apply_rope", "apply_llama31_rope", "apply_rope_pos_ids",
          "apply_llama31_rope_pos_ids", "apply_rope_pos_ids_cos_sin_cache",
          "rope_quantize", "rope_quantize_append_paged_kv_cache"]),
        ("fi_cascade", ["cascade.cu"], "flashinfer_cascade_binding.cu", "cascade",
         ["merge_state", "merge_state_in_place", "merge_states"]),
        # MoE routing kernel + all TRT-LLM common utilities (auto-discovered)
        ("fi_moe_routing",
         ["fused_moe/noAuxTcKernels.cu"]
         + [f"nv_internal/cpp/common/{f.name}"
            for f in sorted((csrc / "nv_internal" / "cpp" / "common").iterdir())
            if f.suffix in (".cpp", ".cu")],
         None, "moe_routing",
         ["NoAuxTc"]),
    ]

    for dir_name, src_files, binding_file, kind, symbols in modules:
        out = gen_dir / dir_name
        out.mkdir(parents=True, exist_ok=True)
        sources = []
        for src_file in src_files:
            src_path = csrc / src_file
            if src_path.exists():
                if kind == "norm" and src_file == "norm.cu":
                    # Patch: exclude layernorm (requires TensorRT-LLM headers)
                    src_text = src_path.read_text()
                    src_text = src_text.replace(
                        '#include <flashinfer/norm.cuh>',
                        '#define FLASHINFER_NORM_NO_LAYERNORM\n#include <flashinfer/norm.cuh>'
                    )
                    # Remove layernorm function body
                    lines = src_text.split('\n')
                    filtered = []
                    skip = False
                    for line in lines:
                        if 'void layernorm(' in line:
                            skip = True
                        if skip and line.strip() == '}':
                            skip = False
                            continue
                        if not skip:
                            filtered.append(line)
                    (out / src_file).write_text('\n'.join(filtered))
                elif kind == "norm" and src_file == binding_file:
                    # Remove layernorm binding
                    src_text = src_path.read_text()
                    lines = [l for l in src_text.split('\n')
                             if 'layernorm' not in l.lower()
                             or 'rmsnorm' in l.lower()]
                    (out / src_file).write_text('\n'.join(lines))
                    continue  # already handled
                else:
                    # Handle subdirectory sources (e.g. fused_moe/foo.cu → foo.cu)
                    dst_name = Path(src_file).name
                    shutil.copy2(src_path, out / dst_name)
                sources.append(out / Path(src_file).name)
        if binding_file is None:
            pass  # source file already contains TVM FFI export
        else:
            binding_path = csrc / binding_file
            if binding_path.exists() and not (out / binding_file).exists():
                if kind == "norm":
                    # Remove layernorm binding export
                    src_text = binding_path.read_text()
                    lines = [l for l in src_text.split('\n')
                             if 'layernorm' not in l]
                    (out / binding_file).write_text('\n'.join(lines))
                else:
                    shutil.copy2(binding_path, out / binding_file)
                sources.append(out / binding_file)

        vinfo = {
            "vid": dir_name, "kind": kind,
            "symbols": {s: f"__tvm_ffi_{s}" for s in symbols},
        }
        results.append((sources, [], vinfo))

    return results


def generate_page_sources(
    fi_src: Path, gen_dir: Path,
) -> Tuple[List[Path], List[str]]:
    """Generate page.cu (append_paged_kv_cache) with renamed symbols.
    DEPRECATED: use generate_utility_sources() instead.
    """
    csrc = fi_src / "csrc"
    out = gen_dir / "fi_page"
    out.mkdir(parents=True, exist_ok=True)

    binding_path = csrc / "flashinfer_page_binding.cu"
    if binding_path.exists():
        shutil.copy2(csrc / "page.cu", out / "page.cu")
        shutil.copy2(binding_path, out / "flashinfer_page_binding.cu")
        return [out / "page.cu", out / "flashinfer_page_binding.cu"], []
    else:
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


def _default_prefill_fp8_sm90_params():
    """Additional params for FP8 E4M3 prefill (SM90+ only).

    Uses per-head Q/K/V scale tensors plus scalar fallbacks.
    Matches FlashInfer's DefaultFP8Attention variant.
    """
    return _generate_additional_params(
        tensor_names=[
            "maybe_scale_q", "maybe_scale_k", "maybe_scale_v",
        ],
        tensor_dtypes=["float", "float", "float"],
        scalar_names=["sm_scale", "scale_q_scalar", "scale_k_scalar", "scale_v_scalar"],
        scalar_dtypes=["double", "double", "double", "double"],
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
        str(fi_src / "csrc" / "nv_internal"),          # vendored TRT-LLM impl headers
        str(fi_src / "csrc" / "nv_internal" / "include"),  # vendored TRT-LLM public headers
        str(fi_src / "csrc" / "fused_moe"),            # MoE helper headers
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
        "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_FP8",
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
    mla_dims: List[Tuple[int, int]] = None,
) -> List[dict]:
    """Build the list of kernel variants to compile.

    Aligned with FlashInfer's official default config (flashinfer/aot.py):
      - FA2: decode + prefill, swa × softcap combinations
      - FA3: prefill only (decode uses prefill on SM90), swa × softcap
      - Asymmetric head_dim (192,128) for FA3
      - MLA decode + paged: (head_dim_ckv, head_dim_kpe) combos
      - FP16 + BF16

    FP8 E4M3 variants: SM90-only, symmetric head_dim, DTypeQ==DTypeKV==FP8.
    """
    if mla_dims is None:
        mla_dims = [(512, 64)]  # DeepSeek V2/V3 default

    variants = []
    sm80_flags = ["-gencode", "arch=compute_80,code=compute_80",
                  "-gencode", "arch=compute_90a,code=sm_90a"]
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

    # ── FP8 E4M3 prefill FA3 (SM90+ only) ──────────────────────────
    # Both Q and KV are FP8 E4M3, output is BF16. Symmetric head_dim only.
    # No logits_soft_cap support in FP8 variant.
    if has_sm90:
        fp8_dtype_c, fp8_dtype_name = DTYPE_MAP["e4m3"]
        for hdim in head_dims:
            if hdim not in (64, 128, 256):
                continue  # FA3 only supports 64/128/256 for HEAD_DIM_VO
            for swa in [False, True]:
                vid = f"fi_prefill_fp8_fa3_e4m3_h{hdim}"
                if swa:
                    vid += "_swa"
                variants.append({
                    "vid": vid, "kind": "prefill_fp8", "backend": "fa3",
                    "dtype": "e4m3", "dtype_c": fp8_dtype_c, "dtype_name": fp8_dtype_name,
                    "hdim_qk": hdim, "hdim_vo": hdim,
                    "swa": swa, "softcap": False,
                    "arch_flags": sm90_flags,
                })

    # ── MLA variants ──────────────────────────────────────────────────
    for dtype in dtypes:
        dtype_c, dtype_name = DTYPE_MAP[dtype]
        for ckv, kpe in mla_dims:
            # MLA decode (cute SM80 backend, supports SM80+)
            # Only compile (no-swa, no-softcap) since MLA rarely uses these
            vid = f"fi_mla_decode_{dtype}_c{ckv}k{kpe}"
            variants.append({
                "vid": vid, "kind": "mla_decode", "backend": "fa2",
                "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                "head_dim_ckv": ckv, "head_dim_kpe": kpe,
                "swa": False, "softcap": False,
                "arch_flags": sm80_flags,
            })

            # MLA paged attention (FA2, supports SM80+)
            vid = f"fi_mla_paged_{dtype}_c{ckv}k{kpe}"
            variants.append({
                "vid": vid, "kind": "mla_paged", "backend": "fa2",
                "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                "head_dim_ckv": ckv, "head_dim_kpe": kpe,
                "swa": False, "softcap": False,
                "arch_flags": sm80_flags,
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
    parser.add_argument("-j", "--workers", type=int,
                        default=os.cpu_count() or 8,
                        help="Parallel compilation workers (default: all CPUs)")
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

    # Parse MLA dims (e.g., "512x64,256x64")
    mla_dims_str = os.environ.get("PRELUDE_FLASHINFER_MLA_DIMS", "512x64")
    mla_dims = []
    for d in mla_dims_str.split(","):
        d = d.strip()
        if d:
            ckv, kpe = d.split("x")
            mla_dims.append((int(ckv), int(kpe)))

    variants = build_variant_matrix(archs, head_dims, dtypes, mla_dims)
    print(f"FlashInfer AOT: {len(variants)} variants, archs={archs}, "
          f"head_dims={head_dims}, dtypes={dtypes}, mla_dims={mla_dims}")

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
        elif v["kind"] == "prefill_fp8":
            sources, extra = generate_batch_prefill_fp8_fa3_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"],
                v["swa"],
            )
        elif v["kind"] == "mla_decode":
            sources, extra = generate_batch_decode_mla_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["head_dim_ckv"], v["head_dim_kpe"],
                v["swa"], v["softcap"],
            )
        elif v["kind"] == "mla_paged":
            sources, extra = generate_batch_mla_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["head_dim_ckv"], v["head_dim_kpe"],
            )
        else:
            continue

        for src in sources:
            compile_jobs.append((src, v["arch_flags"], extra, v))

    # Utility kernels (non-templated, compiled once)
    # Use sm_90a (arch-specific) to satisfy arch_condition.h check in CUDA 12.9+
    sm80_flags = ["-gencode", "arch=compute_80,code=compute_80",
                  "-gencode", "arch=compute_90a,code=sm_90a"]
    utility_variants = generate_utility_sources(fi_src, gen_dir)

    # Activation fusions (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul)
    act_sources, act_extra, act_vinfo = _generate_activation_source(fi_src, gen_dir)
    utility_variants.append((act_sources, act_extra, act_vinfo))

    for sources, extra, vinfo in utility_variants:
        for src in sources:
            compile_jobs.append((src, sm80_flags, extra, vinfo))

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
            "backend": v.get("backend", "fa2"),
            "dtype": v["dtype"],
        }
        if v["kind"] in ("decode", "prefill_fa2", "prefill_fa3", "prefill_fp8"):
            entry["hdim_qk"] = v["hdim_qk"]
            entry["hdim_vo"] = v["hdim_vo"]
            entry["swa"] = v["swa"]
            entry["softcap"] = v["softcap"]
        if v["kind"] in ("mla_decode", "mla_paged"):
            entry["head_dim_ckv"] = v["head_dim_ckv"]
            entry["head_dim_kpe"] = v["head_dim_kpe"]

        if v["kind"] == "decode":
            entry["symbols"] = {
                "plan": f"__tvm_ffi_{v['vid']}_plan",
                "run": f"__tvm_ffi_{v['vid']}_run",
            }
        elif v["kind"].startswith("prefill"):  # prefill_fa2, prefill_fa3, prefill_fp8
            entry["symbols"] = {
                "plan": f"__tvm_ffi_{v['vid']}_plan",
                "ragged_run": f"__tvm_ffi_{v['vid']}_ragged_run",
                "paged_run": f"__tvm_ffi_{v['vid']}_paged_run",
            }
        elif v["kind"] in ("mla_decode", "mla_paged"):
            entry["symbols"] = {
                "plan": f"__tvm_ffi_{v['vid']}_plan",
                "run": f"__tvm_ffi_{v['vid']}_run",
            }
        manifest["variants"].append(entry)

    # Utility modules
    for _, _, vinfo in utility_variants:
        manifest["variants"].append({
            "vid": vinfo["vid"],
            "kind": vinfo["kind"],
            "symbols": vinfo["symbols"],
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(obj_files)} objects, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
