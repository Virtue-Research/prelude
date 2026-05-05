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
) -> Tuple[List[Path], List[str]]:
    """Generate MERGED source files for a batch_decode variant.

    All swa/softcap combos in ONE compilation unit per (dtype, hdim).
    4 swa/cap × 1 kernel = 4 template instantiations (decode has no mask_mode dispatch).
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    additional = dict(zip(
        ("additional_params_decl", "additional_func_params", "additional_params_setter"),
        _default_decode_params(),
    ))

    # ── Merged config.inc ──────────────────────────────────────────
    with open(csrc / "batch_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        variant_decl="#include<flashinfer/attention/variants.cuh>",
        variant_name="DefaultAttention<false, false, false, false>",  # placeholder
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_c,
        idtype="int32_t",
        head_dim_qk=hdim_qk, head_dim_vo=hdim_vo,
        pos_encoding_mode="PosEncodingMode::kNone",
        use_sliding_window="false", use_logits_soft_cap="false",
        **additional,
    )
    # Replace DISPATCH_context with runtime swa/softcap dispatch
    old_config = config_str
    config_str = config_str.replace(
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, '
        'POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, '
        'AttentionVariant, Params, ...) { \\\n'
        '  using AttentionVariant = DefaultAttention<false, false, false, false>; \\\n'
        '  __VA_ARGS__(); \\\n'
        '}',
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, '
        'POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, '
        'AttentionVariant, Params, ...) { \\\n'
        '  auto _run = [&]<bool _SWA, bool _CAP>() { \\\n'
        '    using AttentionVariant = DefaultAttention<false, _SWA, _CAP, false>; \\\n'
        '    __VA_ARGS__(); \\\n'
        '  }; \\\n'
        '  bool _swa = (window_left >= 0); \\\n'
        '  bool _cap = (logits_soft_cap > 0.0); \\\n'
        '  if (!_swa && !_cap) _run.template operator()<false, false>(); \\\n'
        '  else if (_swa && !_cap) _run.template operator()<true, false>(); \\\n'
        '  else if (!_swa && _cap) _run.template operator()<false, true>(); \\\n'
        '  else _run.template operator()<true, true>(); \\\n'
        '}',
    )
    assert config_str != old_config, "DISPATCH_context replacement failed in batch_decode — upstream macro may have changed"
    (out / "batch_decode_config.inc").write_text(config_str)

    # ── Merged kernel instantiations (4 swa/cap combos) ──────────
    swa_cap_combos = [(False, False), (True, False), (False, True), (True, True)]
    lines = [
        '#include <flashinfer/attention/decode.cuh>',
        '#include "batch_decode_config.inc"',
        '',
        'using namespace flashinfer;',
        '',
        'namespace flashinfer {',
        '',
    ]
    for swa, cap in swa_cap_combos:
        var = f"DefaultAttention<false, {str(swa).lower()}, {str(cap).lower()}, false>"
        lines.append(
            f"template cudaError_t "
            f"BatchDecodeWithPagedKVCacheDispatched<{hdim_qk}, "
            f"PosEncodingMode::kNone, {var}, Params>"
            f"(Params params, {dtype_c}* tmp_v, float* tmp_s, "
            f"bool enable_pdl, cudaStream_t stream);"
        )
    lines.extend(['', '};'])

    kernel_path = out / "merged_kernels.cu"
    kernel_path.write_text('\n'.join(lines))
    sources = [kernel_path]

    # ── batch_decode.cu (dispatch) ──
    # Include decode.cuh so that BatchDecodeWithPagedKVCacheKernel is defined (not
    # just forward-declared via scheduler.cuh).  WorkEstimationDispatched takes a
    # function pointer to the kernel; without the definition in this TU the SM90a
    # cubin linker fails with "undefined hidden symbol".
    renames = {
        "BatchDecodeWithPagedKVCacheRun": f"{vid}_BatchDecodeWithPagedKVCacheRun",
        "BatchDecodeWithPagedKVCachePlan": f"{vid}_BatchDecodeWithPagedKVCachePlan",
    }
    _copy_with_renames(csrc / "batch_decode.cu", out / "batch_decode.cu", renames,
                       extra_includes=["flashinfer/attention/decode.cuh"])
    sources.append(out / "batch_decode.cu")

    # ── Binding ──
    binding_src = _generate_renamed_binding(
        csrc / "batch_decode_jit_binding.cu",
        "batch_decode_config.inc",
        {
            "plan": f"__tvm_ffi_{vid}_plan",
            "run": f"__tvm_ffi_{vid}_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_decode_binding.cu").write_text(binding_src)
    sources.append(out / "batch_decode_binding.cu")

    return sources, []


def generate_batch_prefill_fa2_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim_qk: int, hdim_vo: int,
) -> Tuple[List[Path], List[str]]:
    """Generate MERGED source files for a batch_prefill FA2 variant.

    All swa/softcap/mask_mode/layout combos in ONE compilation unit per (dtype, hdim).
    This is the deepgemm-style approach: 96 template instantiations in one .cu,
    with runtime dispatch for swa/softcap/mask_mode.
    Reduces FA2 prefill from 400 .o files to ~30.
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    additional = dict(zip(
        ("additional_params_decl", "additional_func_params", "additional_params_setter"),
        _default_prefill_params(),
    ))

    # ── Merged config.inc ──────────────────────────────────────────
    # Same as upstream but DISPATCH_context dispatches swa/softcap at runtime.
    with open(csrc / "batch_prefill_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    # Render with (swa=false, cap=false) as dummy — we override DISPATCH_context below
    config_str = config_templ.render(
        variant_decl="#include<flashinfer/attention/variants.cuh>",
        variant_name="DefaultAttention<use_custom_mask, false, false, false>",  # placeholder
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_c,
        idtype="int32_t",
        head_dim_qk=hdim_qk, head_dim_vo=hdim_vo,
        pos_encoding_mode="PosEncodingMode::kNone",
        use_sliding_window="false", use_logits_soft_cap="false",
        use_fp16_qk_reduction="false",
        **additional,
    )
    # Replace the hardcoded DISPATCH_context with runtime swa/softcap dispatch
    old_config = config_str
    config_str = config_str.replace(
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, '
        'POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, '
        'AttentionVariant, RaggedParams, PagedParams, ...) \\\n'
        '  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \\\n'
        '    constexpr auto use_custom_mask = MASK_MODE == MaskMode::kCustom; \\\n'
        '    using AttentionVariant = DefaultAttention<use_custom_mask, false, false, false>; \\\n'
        '    __VA_ARGS__(); \\\n'
        '  })',
        # Merged: runtime dispatch swa/softcap via window_left and logits_soft_cap
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, '
        'POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, '
        'AttentionVariant, RaggedParams, PagedParams, ...) \\\n'
        '  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \\\n'
        '    constexpr auto use_custom_mask = MASK_MODE == MaskMode::kCustom; \\\n'
        '    auto _run = [&]<bool _SWA, bool _CAP>() { \\\n'
        '      using AttentionVariant = DefaultAttention<use_custom_mask, _SWA, _CAP, false>; \\\n'
        '      __VA_ARGS__(); \\\n'
        '    }; \\\n'
        '    bool _swa = (window_left >= 0); \\\n'
        '    bool _cap = (logits_soft_cap > 0.0); \\\n'
        '    if (!_swa && !_cap) _run.template operator()<false, false>(); \\\n'
        '    else if (_swa && !_cap) _run.template operator()<true, false>(); \\\n'
        '    else if (!_swa && _cap) _run.template operator()<false, true>(); \\\n'
        '    else _run.template operator()<true, true>(); \\\n'
        '  })',
    )
    assert config_str != old_config, "DISPATCH_context replacement failed in batch_prefill_fa2 — upstream macro may have changed"
    (out / "batch_prefill_config.inc").write_text(config_str)

    # ── Merged kernel instantiations (ONE .cu for all combos) ──────
    # 4 swa/cap × 4 mask_modes × 2 layouts × 3 CTA_TILE_Q = 96 instantiations
    lines = [
        '#include <flashinfer/attention/prefill.cuh>',
        '#include "batch_prefill_config.inc"',
        '',
        'namespace flashinfer {',
        '',
    ]
    swa_cap_combos = [(False, False), (True, False), (False, True), (True, True)]
    for mm_val, mm_name in MASK_MODES.items():
        for swa, cap in swa_cap_combos:
            var = f"DefaultAttention<{mm_name} == MaskMode::kCustom, {str(swa).lower()}, {str(cap).lower()}, false>"
            for cta in [16, 64, 128]:
                for func, params in [("BatchPrefillWithPagedKVCacheDispatched", "PagedParams"),
                                     ("BatchPrefillWithRaggedKVCacheDispatched", "RaggedParams")]:
                    lines.append(
                        f"template cudaError_t {func}<"
                        f"/*CTA_TILE_Q=*/{cta}, {hdim_qk}, {hdim_vo}, "
                        f"PosEncodingMode::kNone, false, {mm_name}, "
                        f"{var}, {params}>"
                        f"({params} params, {dtype_c}* tmp_v, float* tmp_s, "
                        f"bool enable_pdl, cudaStream_t stream);"
                    )
        lines.append('')
    lines.append('};  // namespace flashinfer')

    kernel_path = out / "merged_kernels.cu"
    kernel_path.write_text('\n'.join(lines))

    sources = [kernel_path]

    # ── batch_prefill.cu (dispatch) — reuse upstream with renamed symbols ──
    renames = {
        "BatchPrefillWithKVCachePlan": f"{vid}_BatchPrefillWithKVCachePlan",
        "BatchPrefillWithRaggedKVCacheRun": f"{vid}_BatchPrefillWithRaggedKVCacheRun",
        "BatchPrefillWithPagedKVCacheRun": f"{vid}_BatchPrefillWithPagedKVCacheRun",
    }
    _copy_with_renames(csrc / "batch_prefill.cu", out / "batch_prefill.cu", renames)
    sources.append(out / "batch_prefill.cu")

    # ── Binding ──
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
) -> Tuple[List[Path], List[str]]:
    """Generate MERGED source files for a batch_prefill FA3 (SM90) variant.

    All swa/softcap/mask_mode combos in ONE compilation unit per (dtype, hdim).
    FA3 variant: DefaultAttention<softcap>, USE_SLIDING_WINDOW is a separate template param.
    2 swa × 2 softcap × 4 mask × 2 scheduler × 2 layout = 64 instantiations.
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    additional = dict(zip(
        ("additional_params_decl", "additional_func_params", "additional_params_setter"),
        _default_prefill_sm90_params(),
    ))

    # ── Merged config.inc ──────────────────────────────────────────
    with open(csrc / "batch_prefill_sm90_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        variant_decl="#include<flashinfer/attention/hopper/variants.cuh>",
        variant_name="DefaultAttention<false>",  # placeholder
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_c,
        idtype="int32_t",
        head_dim_qk=hdim_qk, head_dim_vo=hdim_vo,
        pos_encoding_mode="PosEncodingMode::kNone",
        use_sliding_window="false", use_logits_soft_cap="false",
        use_fp16_qk_reduction="false",
        **additional,
    )
    # Replace DISPATCH_context: runtime swa/softcap dispatch
    # FA3: variant = DefaultAttention<CAP>, USE_SLIDING_WINDOW is separate template param
    # batch_prefill_sm90.cu uses USE_SLIDING_WINDOW directly in Dispatched<..., USE_SLIDING_WINDOW, ...>
    # So we need USE_SLIDING_WINDOW as a compile-time bool visible in __VA_ARGS__
    old_config = config_str
    config_str = config_str.replace(
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, '
        'USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, ...) \\\n'
        '  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { '
        'using AttentionVariant = DefaultAttention<false>; __VA_ARGS__();})',
        # Merged: runtime dispatch swa/softcap
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, '
        '_USE_SLIDING_WINDOW, _USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, ...) \\\n'
        '  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \\\n'
        '    auto _run = [&]<bool USE_SLIDING_WINDOW, bool _CAP>() { \\\n'
        '      using AttentionVariant = DefaultAttention<_CAP>; \\\n'
        '      __VA_ARGS__(); \\\n'
        '    }; \\\n'
        '    bool _swa = (window_left >= 0); \\\n'
        '    bool _cap = (logits_soft_cap > 0.0); \\\n'
        '    if (!_swa && !_cap) _run.template operator()<false, false>(); \\\n'
        '    else if (_swa && !_cap) _run.template operator()<true, false>(); \\\n'
        '    else if (!_swa && _cap) _run.template operator()<false, true>(); \\\n'
        '    else _run.template operator()<true, true>(); \\\n'
        '  })',
    )
    assert config_str != old_config, "DISPATCH_context replacement failed in batch_prefill_fa3 — upstream macro may have changed"
    (out / "batch_prefill_sm90_config.inc").write_text(config_str)

    # ── Merged kernel instantiations ──────────────────────────────
    # 2 swa × 2 softcap × 4 mask × 2 scheduler × 2 layout = 64 instantiations
    lines = [
        '#include <flashinfer/attention/hopper/prefill_sm90.cuh>',
        '#include "batch_prefill_sm90_config.inc"',
        '',
        'namespace flashinfer {',
        '',
    ]
    swa_cap_combos = [(False, False), (True, False), (False, True), (True, True)]
    for mm_name in MASK_MODES.values():
        for swa, cap in swa_cap_combos:
            var = f"DefaultAttention<{str(cap).lower()}>"
            for same_sched in ["true", "false"]:
                for func, params in [("BatchPrefillWithPagedKVCacheDispatched", "PagedParams"),
                                     ("BatchPrefillWithRaggedKVCacheDispatched", "RaggedParams")]:
                    lines.append(
                        f"template cudaError_t {func}"
                        f"<{hdim_qk}, {hdim_vo}, {mm_name}, "
                        f"/*USE_SLIDING_WINDOW=*/{str(swa).lower()}, "
                        f"/*SAME_SCHEDULER_FOR_ALL_HEADS=*/{same_sched}, "
                        f"{var}, {params}>"
                        f"({params}& params, bool enable_pdl, cudaStream_t stream);"
                    )
        lines.append('')
    lines.append('};  // namespace flashinfer')

    kernel_path = out / "merged_kernels.cu"
    kernel_path.write_text('\n'.join(lines))
    sources = [kernel_path]

    # ── batch_prefill_sm90.cu (dispatch) ──
    renames = {
        "BatchPrefillWithKVCacheSM90Plan": f"{vid}_BatchPrefillWithKVCacheSM90Plan",
        "BatchPrefillWithRaggedKVCacheSM90Run": f"{vid}_BatchPrefillWithRaggedKVCacheSM90Run",
        "BatchPrefillWithPagedKVCacheSM90Run": f"{vid}_BatchPrefillWithPagedKVCacheSM90Run",
    }
    _copy_with_renames(csrc / "batch_prefill_sm90.cu", out / "batch_prefill_sm90.cu", renames)
    sources.append(out / "batch_prefill_sm90.cu")

    # ── Binding ──
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
) -> Tuple[List[Path], List[str]]:
    """Generate MERGED source files for FP8 E4M3 prefill FA3 (SM90).

    All swa/mask combos in ONE compilation unit per hdim.
    FP8: no softcap, only swa varies. Variant is always DefaultFP8Attention.
    2 swa × 4 mask × 2 scheduler × 2 layout = 32 instantiations.
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    dtype_o_c = "nv_bfloat16"
    additional = dict(zip(
        ("additional_params_decl", "additional_func_params", "additional_params_setter"),
        _default_prefill_fp8_sm90_params(),
    ))

    # ── Merged config.inc ──────────────────────────────────────────
    with open(csrc / "batch_prefill_sm90_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        variant_decl="#include<flashinfer/attention/hopper/variants.cuh>",
        variant_name="DefaultFP8Attention",
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_o_c,
        idtype="int32_t",
        head_dim_qk=hdim, head_dim_vo=hdim,
        pos_encoding_mode="PosEncodingMode::kNone",
        use_sliding_window="false", use_logits_soft_cap="false",
        use_fp16_qk_reduction="false",
        **additional,
    )
    # Replace DISPATCH_context: runtime swa dispatch (no softcap for FP8)
    old_config = config_str
    config_str = config_str.replace(
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, '
        'USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, ...) \\\n'
        '  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { '
        'using AttentionVariant = DefaultFP8Attention; __VA_ARGS__();})',
        '#define DISPATCH_context('
        'DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, '
        '_USE_SLIDING_WINDOW, _USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, ...) \\\n'
        '  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \\\n'
        '    auto _run = [&]<bool USE_SLIDING_WINDOW>() { \\\n'
        '      using AttentionVariant = DefaultFP8Attention; \\\n'
        '      __VA_ARGS__(); \\\n'
        '    }; \\\n'
        '    if (window_left >= 0) _run.template operator()<true>(); \\\n'
        '    else _run.template operator()<false>(); \\\n'
        '  })',
    )
    assert config_str != old_config, "DISPATCH_context replacement failed in batch_prefill_fp8 — upstream macro may have changed"
    (out / "batch_prefill_sm90_config.inc").write_text(config_str)

    # ── Merged kernel instantiations ──────────────────────────────
    # 2 swa × 4 mask × 2 scheduler × 2 layout = 32 instantiations
    lines = [
        '#include <flashinfer/attention/hopper/quantization/prefill_sm90.cuh>',
        '#include "batch_prefill_sm90_config.inc"',
        '',
        'namespace flashinfer {',
        '',
    ]
    for mm_name in MASK_MODES.values():
        for swa in [False, True]:
            for same_sched in ["true", "false"]:
                for func, params in [("BatchFP8PrefillWithPagedKVCacheDispatched", "PagedParams"),
                                     ("BatchFP8PrefillWithRaggedKVCacheDispatched", "RaggedParams")]:
                    lines.append(
                        f"template cudaError_t {func}"
                        f"<{hdim}, {mm_name}, "
                        f"/*USE_SLIDING_WINDOW=*/{str(swa).lower()}, "
                        f"/*SAME_SCHEDULER_FOR_ALL_HEADS=*/{same_sched}, "
                        f"DefaultFP8Attention, {params}>"
                        f"({params}& params, bool enable_pdl, cudaStream_t stream);"
                    )
        lines.append('')
    lines.append('};  // namespace flashinfer')

    kernel_path = out / "merged_kernels.cu"
    kernel_path.write_text('\n'.join(lines))
    sources = [kernel_path]

    # ── FP8 dispatch ──
    renames = {
        "BatchPrefillWithKVCacheSM90Plan": f"{vid}_BatchPrefillWithKVCacheSM90Plan",
        "BatchPrefillWithRaggedKVCacheSM90Run": f"{vid}_BatchPrefillWithRaggedKVCacheSM90Run",
        "BatchPrefillWithPagedKVCacheSM90Run": f"{vid}_BatchPrefillWithPagedKVCacheSM90Run",
    }
    _copy_with_renames(csrc / "batch_prefill_fp8_sm90.cu", out / "batch_prefill_fp8_sm90.cu", renames)
    sources.append(out / "batch_prefill_fp8_sm90.cu")

    # ── Binding ──
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
static void launch_act_and_mul(TensorView out, TensorView input, bool enable_pdl) {
  int d = input.size(input.ndim() - 1) / 2;
  int64_t num_tokens = input.numel() / input.size(input.ndim() - 1);

  cudaSetDevice(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);

    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = activation::act_and_mul_kernel<c_type, Activation>;
    cudaLaunchKernelEx(&config, kernel,
        static_cast<c_type*>(out.data_ptr()),
        static_cast<c_type*>(input.data_ptr()), d);

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "Failed to launch kernel: " << cudaGetErrorString(err);
    return true;
  });
}

// ── TVM FFI exports ──────────────────────────────────────────────

void silu_and_mul(TensorView out, TensorView input, bool enable_pdl) {
  launch_act_and_mul<silu>(out, input, enable_pdl);
}

void gelu_and_mul(TensorView out, TensorView input, bool enable_pdl) {
  launch_act_and_mul<gelu>(out, input, enable_pdl);
}

void gelu_tanh_and_mul(TensorView out, TensorView input, bool enable_pdl) {
  launch_act_and_mul<gelu_tanh>(out, input, enable_pdl);
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

    # ── Module list ─────────────────────────────────────────────────
    #
    # Organized to match upstream flashinfer/aot.py gen_all_modules().
    # Each comment references the upstream function and section.
    #
    # Format: (dir_name, source_files, binding_file, kind, symbols)
    #   dir_name:     output subdirectory name
    #   source_files: .cu files relative to csrc/
    #   binding_file: TVM-FFI binding .cu (or None if exports are inline)
    #   kind:         category for manifest (used by build.rs dispatch codegen)
    #   symbols:      exported TVM-FFI function names
    #
    modules = [
        # ── add_misc: gen_page_module() ──────────────────────────────
        ("fi_page", ["page.cu"], "flashinfer_page_binding.cu", "page",
         ["append_paged_kv_cache", "append_paged_mla_kv_cache"]),

        # ── add_misc: gen_sampling_module() ──────────────────────────
        ("fi_sampling", ["sampling.cu", "renorm.cu"],
         "flashinfer_sampling_binding.cu", "sampling",
         ["softmax", "sampling_from_probs", "sampling_from_logits",
          "top_k_sampling_from_probs", "top_p_sampling_from_probs",
          "min_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
          "top_k_renorm_probs", "top_p_renorm_probs", "top_k_mask_logits",
          "chain_speculative_sampling"]),

        # ── add_misc: gen_norm_module() ──────────────────────────────
        ("fi_norm", ["norm.cu"], "flashinfer_norm_binding.cu", "norm",
         ["rmsnorm", "fused_add_rmsnorm", "gemma_rmsnorm", "gemma_fused_add_rmsnorm",
          "rmsnorm_quant", "fused_add_rmsnorm_quant"]),

        # ── add_misc: gen_rope_module() ──────────────────────────────
        ("fi_rope", ["rope.cu"], "flashinfer_rope_binding.cu", "rope",
         ["apply_rope", "apply_llama31_rope", "apply_rope_pos_ids",
          "apply_llama31_rope_pos_ids", "apply_rope_pos_ids_cos_sin_cache",
          "rope_quantize", "rope_quantize_append_paged_kv_cache"]),

        # ── add_misc: gen_cascade_module() ───────────────────────────
        ("fi_cascade", ["cascade.cu"], "flashinfer_cascade_binding.cu", "cascade",
         ["merge_state", "merge_state_in_place", "merge_states"]),

        # ── add_misc: gen_quantization_module() ──────────────────────
        ("fi_quantization", ["quantization.cu"],
         "flashinfer_quantization_binding.cu", "quantization",
         ["packbits", "segment_packbits"]),

        # ── add_misc: gen_topk_module() ──────────────────────────────
        ("fi_topk", ["topk.cu"], "flashinfer_topk_binding.cu", "topk",
         ["radix_topk", "radix_topk_page_table_transform",
          "radix_topk_ragged_transform", "can_implement_filtered_topk"]),

        # ── add_misc: gen_fp4_kv_dequantization_module() ─────────────
        ("fi_fp4_dequant", ["fp4_kv_dequantization.cu"], None, "fp4",
         ["nvfp4_kv_dequant"]),

        # ── add_misc: gen_fp4_kv_quantization_module() ───────────────
        # SM100+ for HW path, software fallback for SM80+.
        ("fi_fp4_quant", ["fp4_kv_quantization.cu"], None, "fp4",
         ["nvfp4_kv_quant"]),

        # ── add_misc: concat_mla (no upstream aot.py entry) ─────────
        ("fi_concat_mla", ["concat_mla.cu"], None, "mla",
         ["concat_mla_k"]),

        # ── add_misc: CUTLASS MLA paged attention ────────────────────
        ("fi_cutlass_mla", ["cutlass_mla.cu"], "flashinfer_mla_binding.cu", "mla",
         ["cutlass_mla_paged_attention"]),

        # ── add_moe: gen_gemm_module() ───────────────────────────────
        # Base GEMM: segment GEMM + BMM FP8. All archs.
        # Upstream links -lcublas -lcublasLt (bmm_fp8 calls cuBLAS).
        ("fi_gemm", ["group_gemm.cu", "bmm_fp8.cu"],
         "flashinfer_gemm_binding.cu", "gemm",
         ["cutlass_segment_gemm", "bmm_fp8"]),

        # ── add_moe: DSv3 / ML3 router GEMM ─────────────────────────
        ("fi_dsv3_router", ["dsv3_router_gemm.cu"], None, "moe",
         ["dsv3_router_gemm_op", "ml3_router_gemm_op"]),

        # ── add_moe: MoE routing kernel (NoAuxTc) + nv_internal utils ─
        ("fi_moe_routing",
         ["fused_moe/noAuxTcKernels.cu"]
         + [f"nv_internal/cpp/common/{f.name}"
            for f in sorted((csrc / "nv_internal" / "cpp" / "common").iterdir())
            if f.suffix in (".cpp", ".cu")],
         None, "moe_routing",
         ["NoAuxTc"]),

        # ── add_comm: gen_vllm_comm_module() ─────────────────────────
        # All archs. Needs -lcuda (CUDA Driver API, comes with driver).
        ("fi_vllm_allreduce", ["vllm_custom_all_reduce.cu"], None, "comm",
         ["get_graph_buffer_ipc_meta", "register_graph_buffers", "dispose",
          "meta_size", "register_buffer", "init_custom_ar", "all_reduce"]),
    ]

    for dir_name, src_files, binding_file, kind, symbols in modules:
        out = gen_dir / dir_name
        out.mkdir(parents=True, exist_ok=True)
        sources = []
        for src_file in src_files:
            src_path = csrc / src_file
            if src_path.exists():
                if kind == "norm" and src_file == "norm.cu":
                    # Patch: exclude layernorm (requires TensorRT-LLM headers
                    # that the upstream norm.cuh pulls in for LayerNorm).
                    src_text = src_path.read_text()
                    # Remove layernorm function body using brace-counting
                    # so we correctly skip nested braces (macros, lambdas).
                    lines = src_text.split('\n')
                    filtered = []
                    skip = False
                    brace_depth = 0
                    seen_open = False
                    for line in lines:
                        if not skip and 'void layernorm(' in line:
                            skip = True
                            brace_depth = 0
                            seen_open = False
                        if skip:
                            brace_depth += line.count('{') - line.count('}')
                            if line.count('{') > 0:
                                seen_open = True
                            if seen_open and brace_depth <= 0:
                                skip = False
                            continue
                        filtered.append(line)
                    (out / src_file).write_text('\n'.join(filtered))
                elif kind == "fp4" and src_file == "fp4_kv_quantization.cu":
                    # Patch: replace __trap() fallback with software E2M1 for pre-SM100.
                    # The upstream guards E2M1 with #if __CUDA_ARCH__ >= 1000 / #else __trap() / #endif.
                    # We replace the __trap() branch with a correct software implementation.
                    src_text = src_path.read_text()
                    trap_marker = '__trap();\n  return 0;'
                    assert trap_marker in src_text, (
                        "FP4 __trap() replacement failed in fp4_kv_quantization.cu — "
                        "upstream E2M1 __trap() marker not found, upstream may have changed"
                    )
                    sw_fallback = (
                        '// Software E2M1 round-nearest with saturation (prelude AOT, pre-SM100)\n'
                        '  auto sw_e2m1 = [](float v) -> uint8_t {\n'
                        '    float av = fabsf(v);\n'
                        '    uint8_t sign = v < 0.0f ? 0x8u : 0u;\n'
                        '    uint8_t mag;\n'
                        '    if (av < 0.25f) mag = 0;\n'
                        '    else if (av < 0.75f) mag = 1;\n'
                        '    else if (av < 1.25f) mag = 2;\n'
                        '    else if (av < 1.75f) mag = 3;\n'
                        '    else if (av < 2.5f) mag = 4;\n'
                        '    else if (av < 3.5f) mag = 5;\n'
                        '    else if (av < 5.0f) mag = 6;\n'
                        '    else mag = 7;\n'
                        '    return sign | mag;\n'
                        '  };\n'
                        '  uint32_t val = 0;\n'
                        '  #pragma unroll\n'
                        '  for (int i = 0; i < 4; i++) {\n'
                        '    uint8_t lo = sw_e2m1(array[i].x);\n'
                        '    uint8_t hi = sw_e2m1(array[i].y);\n'
                        '    val |= ((uint32_t)((hi << 4) | lo)) << (i * 8);\n'
                        '  }\n'
                        '  return val;'
                    )
                    old_text = src_text
                    src_text = src_text.replace(trap_marker, sw_fallback, 1)
                    assert src_text != old_text, (
                        "FP4 __trap() replacement failed in fp4_kv_quantization.cu — "
                        "replace() did not modify the source"
                    )
                    (out / src_file).write_text(src_text)
                else:
                    # Handle subdirectory sources (e.g. fused_moe/foo.cu → foo.cu)
                    dst_name = Path(src_file).name
                    shutil.copy2(src_path, out / dst_name)
                sources.append(out / Path(src_file).name)
        if binding_file is None:
            pass  # source file already contains TVM FFI export
        else:
            binding_path = csrc / binding_file
            if binding_path.exists():
                # Bug fix: always add the binding to the sources list so it gets
                # compiled. The previous `and not (out / binding_file).exists()`
                # short-circuit would skip the append on incremental builds
                # where the staged binding .cu was already present, leaving the
                # TVM FFI exports (e.g. __tvm_ffi_sampling_from_probs) undefined.
                if not (out / binding_file).exists():
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

def _copy_with_renames(src: Path, dst: Path, renames: Dict[str, str],
                       extra_includes: Optional[List[str]] = None) -> None:
    """Copy a .cu file, prepending #define macros to rename symbols."""
    defines = "\n".join(f"#define {old} {new}" for old, new in renames.items())
    includes = ""
    if extra_includes:
        includes = "\n".join(f"#include <{h}>" for h in extra_includes) + "\n"
    original = src.read_text()
    dst.write_text(f"// AOT variant rename\n{defines}\n{includes}\n{original}")


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
        scalar_names=["logits_soft_cap", "sm_scale", "scale_v_scalar",
                       "token_pos_in_items_len"],
        scalar_dtypes=["double", "double", "double", "int64_t"],
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
    # Prefix .o with variant dir name so filenames are unique across variants
    # (ar archives use basename only, so duplicates would collide).
    obj = out_dir / (out_dir.name + "_" + src.stem + ".o")
    if obj.exists() and obj.stat().st_mtime > src.stat().st_mtime:
        return obj  # up-to-date

    include_dirs = [
        str(fi_src / "include"),
        str(fi_src / "csrc"),
        str(fi_src / "csrc" / "nv_internal"),          # vendored TRT-LLM impl headers
        str(fi_src / "csrc" / "nv_internal" / "include"),  # vendored TRT-LLM public headers
        str(fi_src / "csrc" / "nv_internal" / "tensorrt_llm" / "kernels" / "cutlass_kernels" / "include"),  # MOE GEMM headers
        str(fi_src / "csrc" / "nv_internal" / "tensorrt_llm" / "cutlass_extensions" / "include"),  # CUTLASS extensions
        str(fi_src / "csrc" / "fused_moe"),            # MoE helper headers
        str(fi_src / "csrc" / "fused_moe" / "cutlass_backend"),  # CUTLASS MOE kernels
        str(fi_src / "csrc" / "fused_moe" / "trtllm_backend"),  # TRT-LLM MOE routing
        str(src.parent),  # for config .inc files
    ]
    # spdlog headers (needed for comm modules)
    spdlog_include = fi_src / "3rdparty" / "spdlog" / "include"
    if spdlog_include.exists():
        include_dirs.append(str(spdlog_include))
    # CUTLASS headers (needed for SM90/Hopper kernels)
    cutlass_base = fi_src / "3rdparty" / "cutlass"
    for sub in ["include", "tools/util/include"]:
        p = cutlass_base / sub
        if p.exists():
            include_dirs.append(str(p))
    # FlashInfer v0.6.9+ vendors CCCL and expects it ahead of CTK-bundled
    # headers, matching flashinfer.jit.cpp_ext.get_cccl_includes().
    cccl_base = fi_src / "3rdparty" / "cccl"
    for sub in ["cub", "libcudacxx/include", "thrust"]:
        p = cccl_base / sub
        if p.exists():
            include_dirs.append(str(p))

    # Find TVM FFI headers (+ dlpack sub-dependency)
    tvm_ffi_include = _find_tvm_ffi_include(fi_src)
    if tvm_ffi_include:
        include_dirs.append(str(tvm_ffi_include))
        dlpack_include = tvm_ffi_include.parent / "3rdparty" / "dlpack" / "include"
        if dlpack_include.exists():
            include_dirs.append(str(dlpack_include))

    cmd = [
        "nvcc", "-c", str(src), "-o", str(obj),
        "-std=c++20", "--expt-relaxed-constexpr",
        "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_FP8",
        "-DNDEBUG",
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


def _configure_flashinfer_jit_env(fi_src: Path, gen_dir: Path):
    """Point FlashInfer JIT helpers at the source checkout and AOT gen dir."""
    from flashinfer.jit import env as jit_env

    jit_env.FLASHINFER_CSRC_DIR = fi_src / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = fi_src / "include"
    jit_env.FLASHINFER_GEN_SRC_DIR = gen_dir / "flashinfer_jit"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        fi_src / "3rdparty" / "cutlass" / "include",
        fi_src / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = fi_src / "3rdparty" / "spdlog" / "include"
    jit_env.CCCL_INCLUDE_DIRS = [
        fi_src / "3rdparty" / "cccl" / "cub",
        fi_src / "3rdparty" / "cccl" / "libcudacxx" / "include",
        fi_src / "3rdparty" / "cccl" / "thrust",
    ]
    jit_env.FLASHINFER_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)


def _jit_spec_compile_flags(spec) -> List[str]:
    flags = []
    if spec.extra_cuda_cflags:
        flags.extend(str(flag) for flag in spec.extra_cuda_cflags if flag)
    # gen_cutlass_fused_moe_* puts FAST_BUILD in extra_cflags. We compile
    # every source with nvcc in this AOT path, so carry preprocessor defines.
    if spec.extra_cflags:
        flags.extend(str(flag) for flag in spec.extra_cflags if str(flag).startswith("-D"))
    if spec.extra_include_dirs:
        for include_dir in spec.extra_include_dirs:
            flags.extend(["-I", str(include_dir)])
    return flags


_BLACKWELL_CUTLASS_MOE_QUANT_FLAGS = {
    "-DENABLE_FP8",
    "-DENABLE_FP8_BLOCK_SCALE",
    "-DENABLE_FP4",
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
}


_BLACKWELL_CUTLASS_MOE_COMMON_SOURCES = {
    "moe_gemm_tma_warp_specialized_input.cu",
    "moe_gemm_kernels_bf16_bf16.cu",
    "cutlass_heuristic.cpp",
    "lora.cpp",
    "flashinfer_cutlass_fused_moe_binding.cu",
    "deepgemm_jit_setup.cu",
    "cutlass_fused_moe_instantiation.cu",
}


def _cutlass_moe_blackwell_compile_flags(flags: List[str]) -> List[str]:
    """Keep Prelude's Blackwell fused-MoE AOT build on the dense BF16 path."""
    filtered = [flag for flag in flags if flag not in _BLACKWELL_CUTLASS_MOE_QUANT_FLAGS]
    filtered.append("-DPRELUDE_FLASHINFER_BLACKWELL_BF16_ONLY")
    return filtered


def _should_compile_cutlass_moe_blackwell_source(src: Path, archs: List[int]) -> bool:
    # TopicGuard/Qwen3 production weights are dense BF16. The upstream
    # SM100/SM103 JIT spec is broad and also pulls in fp16, uint, FP8/FP4 and
    # older generated kernels. Several of those translation units compile very
    # slowly or stall under static AOT, so keep the AOT archive to the BF16
    # dense path plus the generated kernels for the requested Blackwell arch.
    name = src.name
    if name in _BLACKWELL_CUTLASS_MOE_COMMON_SOURCES:
        return True
    if not name.startswith("cutlass_kernel_file_gemm_grouped_"):
        return False
    if any(a >= 103 and a < 120 for a in archs):
        # The BF16 dispatcher still references upstream Sm100 launcher
        # instantiations for some tile/cluster choices, while the SM103
        # generated files cover Blackwell-specific grouped kernels.
        return "_sm103_" in name or "_sm100_" in name
    return "_sm100_" in name


def _add_cutlass_fused_moe_from_jit_spec(compile_jobs, fi_src: Path, gen_dir: Path,
                                         archs: List[int], vinfo: dict) -> int:
    _configure_flashinfer_jit_env(fi_src, gen_dir)
    from flashinfer.jit.fused_moe import (
        gen_cutlass_fused_moe_sm100_module,
        gen_cutlass_fused_moe_sm103_module,
    )

    if any(a >= 103 and a < 120 for a in archs):
        spec = gen_cutlass_fused_moe_sm103_module(use_fast_build=True)
    else:
        spec = gen_cutlass_fused_moe_sm100_module(use_fast_build=True)

    flags = _cutlass_moe_blackwell_compile_flags(_jit_spec_compile_flags(spec))
    shared_common_dir = fi_src / "csrc" / "nv_internal" / "cpp" / "common"
    compiled = 0
    skipped = 0
    for src in spec.sources:
        src = Path(src)
        # The monolithic AOT archive already includes TRT-LLM common support
        # objects through fi_moe_routing. Upstream JIT specs are standalone and
        # include them per module; whole-archive static linking would duplicate
        # symbols such as tensorrt_llm::common::getIntEnv.
        if src.parent == shared_common_dir and src.suffix in (".cpp", ".cu"):
            continue
        if not _should_compile_cutlass_moe_blackwell_source(src, archs):
            skipped += 1
            continue
        compile_jobs.append((src, [], flags, vinfo))
        compiled += 1
    if skipped:
        print(f"  Blackwell CUTLASS MoE: skipped {skipped} non-production AOT sources")
    return compiled


def _find_tvm_ffi_include(fi_src: Path) -> Optional[Path]:
    """Find TVM FFI include directory from third_party/tvm-ffi."""
    script_dir = Path(__file__).resolve().parent.parent  # prelude-flashinfer crate
    workspace_root = script_dir.parent.parent.parent  # crates/prelude-cuda/flashinfer -> workspace
    p = workspace_root / "third_party" / "tvm-ffi" / "include"
    if p.exists():
        return p
    return None


# ── Variant matrix ─────────────────────────────────────────────────────

def fa2_arch_flags(archs: List[int]) -> List[str]:
    # Keep a PTX fallback for forward-compatible FA2 dispatch, but only emit
    # native SM90 cubins when SM90 was explicitly requested.
    flags = ["-gencode", "arch=compute_80,code=compute_80"]
    if 90 in archs:
        flags += ["-gencode", "arch=compute_90a,code=sm_90a"]
    return flags


def flashinfer_cuda_arch_list(archs: List[int]) -> str:
    return " ".join(f"{arch // 10}.{arch % 10}" for arch in sorted(set(archs)))


def blackwell_arch_flags(archs: List[int]) -> List[str]:
    flags = []
    for arch in sorted(set(archs)):
        if arch >= 100:
            suffix = f"{arch}a"
            flags += ["-gencode", f"arch=compute_{suffix},code=sm_{suffix}"]
    return flags or ["-gencode", "arch=compute_100a,code=sm_100a"]


def gdn_prefill_arch_flags(archs: List[int]) -> List[str]:
    """Architectures for the FlashInfer GDN prefill kernel.

    The implementation lives under FlashInfer's Hopper path and uses WGMMA,
    so it is SM90-only. Blackwell does not assemble these WGMMA instructions
    for sm_10x targets; Prelude's CUDA backend provides its own fallback there.
    """
    flags = []
    unique_archs = sorted(set(archs))
    if 90 in unique_archs:
        flags += ["-gencode", "arch=compute_90a,code=sm_90a"]
    return flags


GDN_BLACKWELL_HEAD_PAIRS = [
    (16, 16),  # Qwen3.5 dense 0.8B/2B
    (16, 32),  # Qwen3.5 dense 4B, Qwen3.5/3.6 MoE, Qwen3-Next
    (16, 48),  # Qwen3.5 27B class
    (16, 64),  # Qwen3.5 large class without TP
    (32, 32),
    (64, 64),
]


def verify_exported_tvm_symbol(obj_path: Path, name: str) -> None:
    symbol = f"__tvm_ffi_{name}"
    result = subprocess.run(
        ["nm", "-g", str(obj_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nm failed for {obj_path}: {result.stderr.strip()}")
    if symbol not in result.stdout:
        raise RuntimeError(f"{obj_path} does not export {symbol}")


def compile_gdn_blackwell_variant(
    obj_path: Path,
    hq: int,
    hv: int,
    dtype: str,
) -> bool:
    """AOT-export FlashInfer's SM100 CuTe DSL GDN prefill kernel.

    The exported TVM-FFI function takes the same runtime args as
    ``GatedDeltaNetChunkedKernel.__call__``:

      q, k, v, gate, beta, output, cu_seqlens, initial_state, output_state,
      output_checkpoints, cu_checkpoints, checkpoint_every_n_tokens, scale,
      workspace, stream

    We specialize only on IO dtype and head counts. Token count, batch size and
    pool size stay dynamic, matching upstream's Python cache key.
    """
    if obj_path.exists():
        print(f"  {obj_path.name}: already exists, skipping")
        return True

    try:
        import cutlass
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda_drv
        import torch
        from cutlass.cute.runtime import from_dlpack
        from flashinfer.gdn_kernels.blackwell.gated_delta_net_chunked import (
            GatedDeltaNetChunkedKernel,
        )
        from flashinfer.cute_dsl.utils import get_num_sm
    except Exception as e:
        print(f"  Blackwell GDN import failed: {e}")
        return False

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
        cutlass_dtype = cutlass.BFloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
        cutlass_dtype = cutlass.Float16
    else:
        return False

    try:
        device = torch.device("cuda")
        num_sm = get_num_sm(device)
        max_active_clusters = num_sm
        seq_len = 64
        batch = 1
        head_dim = 128

        q = torch.zeros((seq_len, hq, head_dim), dtype=torch_dtype, device=device)
        k = torch.zeros((seq_len, hq, head_dim), dtype=torch_dtype, device=device)
        v = torch.zeros((seq_len, hv, head_dim), dtype=torch_dtype, device=device)
        gate = torch.zeros((seq_len, max(hq, hv)), dtype=torch.float32, device=device)
        beta = torch.zeros_like(gate)
        output = torch.zeros((seq_len, max(hq, hv), head_dim), dtype=torch_dtype, device=device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        initial_state = torch.zeros(
            (batch, max(hq, hv), head_dim, head_dim),
            dtype=torch.float32,
            device=device,
        )
        output_state = torch.zeros_like(initial_state)
        workspace = torch.empty(
            GatedDeltaNetChunkedKernel.get_workspace_size(
                num_sm, batch, hq, hv, True
            ),
            dtype=torch.int8,
            device=device,
        )

        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=cutlass_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=cutlass.Float32,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(128, 64, 128),
            mma_tiler_qkv=(128, 64, 64),
            mma_tiler_kv=(128, 128, 64),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=hq >= hv,
            use_initial_state=True,
            store_final_state=True,
            enable_checkpoints=False,
            is_persistent=True,
        )

        def dyn(t, mode, stride_order, assumed_align=16, divisibility=1):
            cute_t = from_dlpack(t, assumed_align=assumed_align)
            cute_t.mark_compact_shape_dynamic(
                mode=mode, stride_order=stride_order, divisibility=divisibility
            )
            return cute_t

        q_t = dyn(q, 0, (0, 1, 2))
        k_t = dyn(k, 0, (0, 1, 2))
        v_t = dyn(v, 0, (0, 1, 2))
        gate_t = dyn(gate, 0, (0, 1))
        beta_t = dyn(beta, 0, (0, 1))
        output_t = dyn(output, 0, (0, 1, 2))
        cu_t = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        init_t = (
            from_dlpack(initial_state, assumed_align=16)
            .mark_layout_dynamic()
            .mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=head_dim
            )
        )
        state_t = (
            from_dlpack(output_state, assumed_align=16)
            .mark_layout_dynamic()
            .mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=head_dim
            )
        )
        workspace_t = from_dlpack(workspace, assumed_align=16)
        stream = cuda_drv.CUstream(torch.cuda.current_stream(device=device).cuda_stream)

        compiled = cute.compile(
            gdn,
            q_t,
            k_t,
            v_t,
            gate_t,
            beta_t,
            output_t,
            cu_t,
            init_t,
            state_t,
            None,
            None,
            0,
            head_dim ** -0.5,
            workspace_t,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        compiled.export_to_c(
            object_file_path=str(obj_path),
            function_name=f"gdn_prefill_sm100_{dtype}_h{hq}_hv{hv}",
        )
        print(f"  Exported {obj_path.name} ({obj_path.stat().st_size // 1024}KB)")
        verify_exported_tvm_symbol(obj_path, f"gdn_prefill_sm100_{dtype}_h{hq}_hv{hv}")
        return True
    except Exception as e:
        print(f"  ERROR compiling Blackwell GDN h{hq}/hv{hv}/{dtype}: {e}")
        import traceback

        traceback.print_exc()
        return False


def guard_gdn_outer_dispatch_instantiations(source: str) -> str:
    """Gate GDN outer dispatch instantiations by selected AOT dtypes.

    FlashInfer's dispatcher source explicitly instantiates both fp16 and bf16
    wrappers. Prelude production builds usually pass PRELUDE_FLASHINFER_DTYPES
    to compile only the dtype used by the loaded model, so keep the copied
    dispatcher source consistent with the generated inner instantiation files.
    """
    fp16_marker = (
        "template void launch_delta_rule_prefill_kernel"
        "<cutlass::arch::Sm90, half, half, float>("
    )
    bf16_marker = (
        "template void\n"
        "launch_delta_rule_prefill_kernel"
        "<cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>("
    )
    namespace_end = "\n\n}  // namespace flat"

    fp16_start = source.find(fp16_marker)
    bf16_start = source.find(bf16_marker)
    end = source.find(namespace_end)
    if fp16_start < 0 or bf16_start < 0 or end < 0:
        raise RuntimeError("unexpected GDN dispatcher source layout")

    fp16_block = source[fp16_start:bf16_start]
    bf16_block = source[bf16_start:end]
    return (
        source[:fp16_start]
        + "#if defined(PRELUDE_GDN_ENABLE_FP16)\n"
        + fp16_block.rstrip()
        + "\n#endif\n\n"
        + "#if defined(PRELUDE_GDN_ENABLE_BF16)\n"
        + bf16_block.rstrip()
        + "\n#endif"
        + source[end:]
    )


def guard_gdn_launcher_dispatch(source: str) -> str:
    """Gate GDN launcher dtype dispatch by selected AOT dtypes."""
    start_marker = "  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dtype, DType, [&] {\n"
    end_marker = "  });\n}"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("unexpected GDN launcher source layout")
    end += len("  });\n")

    replacement = r'''  int dev_id;
  cudaGetDevice(&dev_id);
  int device_major;
  cudaDeviceGetAttribute(&device_major, cudaDevAttrComputeCapabilityMajor, dev_id);

#if defined(FLAT_SM90A_ENABLED)
  if (device_major != 9 && device_major != 10) {
    std::ostringstream err_msg;
    err_msg << "delta rule kernel supports SM90/SM10x only; got device major version "
            << device_major;
    FLASHINFER_ERROR(err_msg.str());
    return;
  }
#else
  FLASHINFER_ERROR("SM90/SM10x GDN prefill is not enabled, delta rule kernel is not built");
  return;
#endif

#define PRELUDE_GDN_LAUNCH(DType)                                                        \
  do {                                                                                    \
    flat::launch_delta_rule_prefill_kernel<cutlass::arch::Sm90, DType, DType, float>(     \
        stream, static_cast<DType*>(output), static_cast<float*>(output_state),           \
        static_cast<DType const*>(q), static_cast<DType const*>(k),                       \
        static_cast<DType const*>(v), static_cast<float const*>(input_state),             \
        static_cast<float const*>(alpha), static_cast<float const*>(beta), cu_seqlens,    \
        workspace_buffer, num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads,   \
        head_size, packed_seq, scale, sm_count, static_cast<float*>(state_checkpoints),   \
        checkpoint_cu_starts, static_cast<int32_t>(checkpoint_every_n_tokens));           \
    return;                                                                               \
  } while (false)

  if (dtype == dl_bfloat16) {
#if defined(PRELUDE_GDN_ENABLE_BF16)
    PRELUDE_GDN_LAUNCH(nv_bfloat16);
#else
    FLASHINFER_ERROR("bf16 GDN prefill was not AOT-compiled");
    return;
#endif
  }

  if (dtype == dl_float16) {
#if defined(PRELUDE_GDN_ENABLE_FP16)
    PRELUDE_GDN_LAUNCH(half);
#else
    FLASHINFER_ERROR("fp16 GDN prefill was not AOT-compiled");
    return;
#endif
  }

#undef PRELUDE_GDN_LAUNCH

  std::ostringstream err_msg;
  err_msg << "unsupported GDN prefill dtype code=" << static_cast<int>(dtype.code)
          << " bits=" << static_cast<int>(dtype.bits);
  FLASHINFER_ERROR(err_msg.str());
  return;
'''

    return source[:start] + replacement + source[end:]


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
    sm80_flags = fa2_arch_flags(archs)
    sm90_flags = ["-gencode", "arch=compute_90a,code=sm_90a"]
    has_sm90 = 90 in archs

    swa_softcap_combos = [(False, False), (True, False), (False, True), (True, True)]

    # Standard symmetric head_dim variants
    # FA2 decode + prefill: MERGED — one variant per (dtype, hdim), swa/softcap dispatched at runtime
    for dtype in dtypes:
        dtype_c, dtype_name = DTYPE_MAP[dtype]
        for hdim in head_dims:
            # Batch decode FA2 (merged: all swa/softcap in one compilation unit)
            vid = variant_id("decode", "fa2", dtype, hdim, hdim)
            variants.append({
                "vid": vid, "kind": "decode", "backend": "fa2",
                "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                "hdim_qk": hdim, "hdim_vo": hdim,
                "arch_flags": sm80_flags,
            })

            # Batch prefill FA2 (merged: all swa/softcap/mask in one compilation unit)
            vid = variant_id("prefill", "fa2", dtype, hdim, hdim)
            variants.append({
                "vid": vid, "kind": "prefill_fa2", "backend": "fa2",
                "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                "hdim_qk": hdim, "hdim_vo": hdim,
                "arch_flags": sm80_flags,
            })

            # Batch prefill FA3 (SM90+, merged: all swa/softcap in one compilation unit)
            if has_sm90 and hdim in (64, 128, 256):
                vid = variant_id("prefill", "fa3", dtype, hdim, hdim)
                variants.append({
                    "vid": vid, "kind": "prefill_fa3", "backend": "fa3",
                    "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                    "hdim_qk": hdim, "hdim_vo": hdim,
                    "arch_flags": sm90_flags,
                })

    # Asymmetric head_dim: FA3 (192,128) — merged
    if has_sm90:
        for dtype in dtypes:
            dtype_c, dtype_name = DTYPE_MAP[dtype]
            vid = variant_id("prefill", "fa3", dtype, 192, 128)
            variants.append({
                "vid": vid, "kind": "prefill_fa3", "backend": "fa3",
                "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                "hdim_qk": 192, "hdim_vo": 128,
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
            # Merged: all swa in one compilation unit per hdim
            vid = f"fi_prefill_fp8_fa3_e4m3_h{hdim}"
            variants.append({
                "vid": vid, "kind": "prefill_fp8", "backend": "fa3",
                "dtype": "e4m3", "dtype_c": fp8_dtype_c, "dtype_name": fp8_dtype_name,
                "hdim_qk": hdim, "hdim_vo": hdim,
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
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = flashinfer_cuda_arch_list(archs)

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
            )
        elif v["kind"] == "prefill_fa2":
            sources, extra = generate_batch_prefill_fa2_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
            )
        elif v["kind"] == "prefill_fa3":
            sources, extra = generate_batch_prefill_fa3_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
            )
        elif v["kind"] == "prefill_fp8":
            sources, extra = generate_batch_prefill_fp8_fa3_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"],
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
    sm80_flags = fa2_arch_flags(archs)
    utility_variants = generate_utility_sources(fi_src, gen_dir)

    # Activation fusions (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul)
    act_sources, act_extra, act_vinfo = _generate_activation_source(fi_src, gen_dir)
    utility_variants.append((act_sources, act_extra, act_vinfo))

    # SM100 (Blackwell) FMHA — CUTLASS-based FMHA for SM100+.
    # Requires: CUDA 12.8+, PRELUDE_FLASHINFER_ARCHS containing sm_100.
    # The upstream kernels use a different architecture (CUTLASS SM100 FMHA runner)
    # with runtime dispatch for head_dim and mask_mode.
    # Source files: csrc/fmha_cutlass_sm100.cu, csrc/blackwell_fmha_plan.cu,
    #               csrc/fmha_cutlass_sm100_binding.cu
    has_sm100 = any(a >= 100 for a in archs)
    if has_sm100:
        sm100_csrc = fi_src / "csrc"
        sm100_out = gen_dir / "fi_fmha_sm100"
        sm100_out.mkdir(parents=True, exist_ok=True)
        sm100_sources = []
        sm100_flags = blackwell_arch_flags(archs)
        for src_file in ["fmha_cutlass_sm100.cu", "blackwell_fmha_plan.cu"]:
            src_path = sm100_csrc / src_file
            if src_path.exists():
                shutil.copy2(src_path, sm100_out / src_file)
                sm100_sources.append(sm100_out / src_file)
        # Rename exports to avoid collision with other plan/run symbols
        binding_path = sm100_csrc / "fmha_cutlass_sm100_binding.cu"
        if binding_path.exists():
            bsrc = binding_path.read_text()
            bsrc = bsrc.replace(
                'TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,',
                'TVM_FFI_DLL_EXPORT_TYPED_FUNC(fmha_sm100_run,',
            ).replace(
                'TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan,',
                'TVM_FFI_DLL_EXPORT_TYPED_FUNC(fmha_sm100_plan,',
            )
            (sm100_out / "fmha_cutlass_sm100_binding.cu").write_text(bsrc)
            sm100_sources.append(sm100_out / "fmha_cutlass_sm100_binding.cu")
        sm100_vinfo = {
            "vid": "fi_fmha_sm100", "kind": "fmha_sm100",
            "symbols": {
                "fmha_sm100_run": "__tvm_ffi_fmha_sm100_run",
                "fmha_sm100_plan": "__tvm_ffi_fmha_sm100_plan",
            },
        }
        for src in sm100_sources:
            compile_jobs.append((src, sm100_flags, [], sm100_vinfo))
        print(f"  SM100 FMHA: {len(sm100_sources)} sources (requires CUDA 12.8+)")

    # ── SM90 modules (matching upstream aot.py has_sm90 blocks) ────────
    has_sm90 = 90 in archs
    sm90_flags = ["-gencode", "arch=compute_90a,code=sm_90a"]

    gdn_flags = gdn_prefill_arch_flags(archs)
    gdn_dtypes = []
    if "fp16" in dtypes:
        gdn_dtypes.append("half")
    if "bf16" in dtypes:
        gdn_dtypes.append("nv_bfloat16")
    if gdn_flags and gdn_dtypes:
        import jinja2

        csrc = fi_src / "csrc"
        mod_out = gen_dir / "fi_gdn"
        mod_out.mkdir(parents=True, exist_ok=True)
        sources = []
        gdn_extra_flags = ["-DFLAT_SM90A_ENABLED"]
        if "half" in gdn_dtypes:
            gdn_extra_flags.append("-DPRELUDE_GDN_ENABLE_FP16")
        if "nv_bfloat16" in gdn_dtypes:
            gdn_extra_flags.append("-DPRELUDE_GDN_ENABLE_BF16")

        with open(csrc / "gdn_prefill_sm90_kernel_inst.jinja") as f:
            templ = jinja2.Template(f.read())

        # gen_gdn_prefill module: 64 kernel instantiations via jinja
        # NOTE: upstream a1166dc added a 6th template bool `enable_checkpointing`
        # (commit 08ab45d, state checkpointing in chunk_gated_delta_rule). We
        # compile both variants so the runtime dispatcher in
        # `prefill_kernel_delta_rule_sm90.cu::launch_delta_rule_prefill_kernel`
        # can pick based on `checkpoint_every_n_tokens > 0`. Compiling only
        # one side produces a link error when the other branch of the
        # `DISPATCH_GBAI` macro references an un-instantiated template.
        for dtype in gdn_dtypes:
            for is_gva in ["false", "true"]:
                for needs_beta in ["false", "true"]:
                    for needs_alpha in ["false", "true"]:
                        for init_state in ["false", "true"]:
                            for enable_checkpointing in ["false", "true"]:
                                params = dict(dtype=dtype, is_gva=is_gva,
                                              needs_beta=needs_beta,
                                              needs_alpha=needs_alpha,
                                              init_state=init_state,
                                              enable_checkpointing=enable_checkpointing)
                                ckpt_tag = "c1" if enable_checkpointing == "true" else "c0"
                                fname = (f"gdn_prefill_kernel_{dtype}_g{is_gva}"
                                         f"b{needs_beta}a{needs_alpha}i{init_state}"
                                         f"{ckpt_tag}.cu")
                                fpath = mod_out / fname
                                fpath.write_text(templ.render(**params))
                                sources.append(fpath)

        for sf in ["gdn_prefill_launcher.cu", "prefill_kernel_delta_rule_sm90.cu"]:
            sp = csrc / sf
            if sp.exists():
                dst = mod_out / Path(sf).name
                if sf == "gdn_prefill_launcher.cu":
                    launcher = guard_gdn_launcher_dispatch(sp.read_text())
                    dst.write_text(launcher)
                elif sf == "prefill_kernel_delta_rule_sm90.cu":
                    dispatcher = guard_gdn_outer_dispatch_instantiations(sp.read_text())
                    dst.write_text(dispatcher)
                else:
                    shutil.copy2(sp, dst)
                sources.append(dst)

        vinfo_gdn = {
            "vid": "fi_gdn",
            "kind": "gdn",
            "symbols": {"gdn_prefill": "__tvm_ffi_gdn_prefill"},
        }
        for src in sources:
            compile_jobs.append((src, gdn_flags, gdn_extra_flags, vinfo_gdn))
        utility_variants.append(([], [], vinfo_gdn))
        print(f"  GDN prefill: {len(sources)} sources ({','.join(gdn_dtypes)}, SM90/SM10x)")

    gdn_blackwell_vinfo = None
    if has_sm100 and "bf16" in dtypes:
        symbols = {
            f"gdn_prefill_sm100_bf16_h{hq}_hv{hv}":
                f"__tvm_ffi_gdn_prefill_sm100_bf16_h{hq}_hv{hv}"
            for hq, hv in GDN_BLACKWELL_HEAD_PAIRS
        }
        gdn_blackwell_vinfo = {
            "vid": "fi_gdn_blackwell",
            "kind": "gdn",
            "symbols": symbols,
        }
        print(
            "  Blackwell GDN prefill: "
            f"{len(GDN_BLACKWELL_HEAD_PAIRS)} BF16 CuTe DSL variants"
        )

    if has_sm90:
        import jinja2
        sm90_modules = []
        csrc = fi_src / "csrc"

        def _add_sm90_jinja(vid, kind, template_name, instances, extra_sources,
                            binding_file, symbols, extra_flags=None):
            """Render jinja template for each instance, copy extra sources, compile all."""
            mod_out = gen_dir / vid
            mod_out.mkdir(parents=True, exist_ok=True)
            sources = []
            with open(csrc / template_name) as f:
                templ = jinja2.Template(f.read())
            for params, fname in instances:
                src_text = templ.render(**params)
                fpath = mod_out / fname
                fpath.write_text(src_text)
                sources.append(fpath)
            for sf in extra_sources:
                sp = csrc / sf
                if sp.exists():
                    shutil.copy2(sp, mod_out / Path(sf).name)
                    sources.append(mod_out / Path(sf).name)
            if binding_file:
                bp = csrc / binding_file
                if bp.exists():
                    shutil.copy2(bp, mod_out / Path(binding_file).name)
                    sources.append(mod_out / Path(binding_file).name)
            vinfo = {"vid": vid, "kind": kind,
                     "symbols": {s: f"__tvm_ffi_{s}" for s in symbols}}
            ef = extra_flags or []
            for src in sources:
                compile_jobs.append((src, sm90_flags, ef, vinfo))
            sm90_modules.append(vinfo)
            return len(sources)

        # gen_gemm_sm90_module: 6 dtype pair instantiations via jinja
        cutlass_dtype_map = {
            "float16": "cutlass::half_t", "bfloat16": "cutlass::bfloat16_t",
            "float8_e4m3fn": "cutlass::float_e4m3_t", "float8_e5m2": "cutlass::float_e5m2_t",
        }
        gemm90_dtype_pairs = [
            ("float16", "float16"), ("bfloat16", "bfloat16"),
            ("float8_e4m3fn", "float16"), ("float8_e5m2", "float16"),
            ("float8_e4m3fn", "bfloat16"), ("float8_e5m2", "bfloat16"),
        ]
        gemm90_instances = []
        for dtype_in, dtype_out in gemm90_dtype_pairs:
            params = {"dtype_in": cutlass_dtype_map[dtype_in],
                      "dtype_out": cutlass_dtype_map[dtype_out]}
            fname = f"group_gemm_sm90_{dtype_in}_{dtype_out}.cu"
            gemm90_instances.append((params, fname))
        n = _add_sm90_jinja(
            "fi_gemm_sm90", "gemm", "group_gemm_sm90_kernel_inst.jinja",
            gemm90_instances,
            ["group_gemm_sm90.cu"], "flashinfer_gemm_sm90_binding.cu",
            ["cutlass_segment_gemm_sm90"],
            ["-DCUTLASS_ENABLE_GDC_FOR_SM90=1"],
        )
        print(f"  SM90 GEMM: {n} sources (6 dtype pair instantiations)")

        # ── CUTLASS Fused MoE SM90 (BF16 only) ─────────────────────────
        # Generates CUTLASS template instantiation files, then compiles
        # the fused MoE binding + BF16 GEMM kernels.  This replaces the
        # WMMA-based per-expert dispatch with a single CUTLASS-based
        # fused routing + GEMM1 + activation + GEMM2 kernel.
        from flashinfer.jit.gemm.cutlass.generate_kernels import generate_gemm_operations

        cutlass_moe_gen = gen_dir / "cutlass_moe_90"
        cutlass_moe_gen.mkdir(parents=True, exist_ok=True)
        generate_gemm_operations(str(cutlass_moe_gen), "90;90-real")

        nv_moe = csrc / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm"
        # Compile ALL dtype kernel files (matching FlashInfer JIT).
        # This ensures all template instantiations the binding references
        # are available at link time.
        cutlass_moe_srcs_abs = [
            nv_moe / "moe_gemm_tma_warp_specialized_input.cu",
            nv_moe / "moe_gemm_kernels_bf16_bf16.cu",
            nv_moe / "moe_gemm_kernels_fp16_fp16.cu",
            nv_moe / "moe_gemm_kernels_bf16_uint4.cu",
            nv_moe / "moe_gemm_kernels_bf16_uint8.cu",
            nv_moe / "moe_gemm_kernels_bf16_fp8.cu",
            nv_moe / "moe_gemm_kernels_bf16_fp4.cu",
            nv_moe / "moe_gemm_kernels_fp16_uint4.cu",
            nv_moe / "moe_gemm_kernels_fp16_uint8.cu",
            nv_moe / "moe_gemm_kernels_fp16_fp4.cu",
            nv_moe / "moe_gemm_kernels_fp8_fp8.cu",
            nv_moe / "moe_gemm_kernels_fp8_uint4.cu",
            nv_moe / "moe_gemm_kernels_fp8_fp4.cu",
            nv_moe / "moe_gemm_kernels_fp4_fp4.cu",
            nv_moe / "moe_gemm_kernels_fp32_fp32.cu",
            csrc / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/"
                   "fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
            csrc / "fused_moe/cutlass_backend/deepgemm_jit_setup.cu",
            # Support files (NOT already compiled in fi_moe_routing)
            csrc / "nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
            csrc / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
            csrc / "nv_internal/tensorrt_llm/kernels/lora/lora.cpp",
        ]

        # Patch the binding to guard FP16 runner creation behind ENABLE_FP16
        # (not defined in our BF16-only build). Upstream always compiles FP16
        # which drags in NONE-fusion generated kernels that hit a template bug.
        # Only switch_output_type needs patching (not dataType() which is just
        # a type mapping).
        mod_out = gen_dir / "fi_cutlass_moe_sm90"
        mod_out.mkdir(parents=True, exist_ok=True)
        # No patching needed — compile all dtype variants to match JIT

        cutlass_moe_extra = [
            # Must match FlashInfer JIT flags exactly to avoid template bugs
            "-std=c++17",  # override c++20 — CUTLASS templates need c++17
            "-DCOMPILE_HOPPER_TMA_GEMMS",
            "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP8_BLOCK_SCALE",
            "-DENABLE_FP4",
            "-DUSING_OSS_CUTLASS_MOE_GEMM",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DFLASHINFER_ENABLE_FP8_E8M0",
            "-DFLASHINFER_ENABLE_FP4_E2M1",
            "-DFAST_BUILD",
            f"-I{cutlass_moe_gen}",
        ]

        cutlass_moe_compiled = 0
        vinfo_moe = {
            "vid": "fi_cutlass_moe_sm90",
            "kind": "cutlass_moe",
            "symbols": {"init": "__tvm_ffi_init"},
        }

        # Compile static source files from original locations
        for sp in cutlass_moe_srcs_abs:
            if sp.exists():
                compile_jobs.append((sp, sm90_flags, cutlass_moe_extra, vinfo_moe))
                cutlass_moe_compiled += 1

        # Compile binding from original location (all dtypes)
        binding_src = csrc / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu"
        compile_jobs.append((binding_src, sm90_flags, cutlass_moe_extra, vinfo_moe))
        cutlass_moe_compiled += 1

        # Compile cutlass_fused_moe_instantiation from original location
        inst_src = csrc / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu"
        if inst_src.exists():
            compile_jobs.append((inst_src, sm90_flags, cutlass_moe_extra, vinfo_moe))
            cutlass_moe_compiled += 1

        # Compile ALL generated CUTLASS kernel instantiation files
        # (matching FlashInfer JIT — compile everything to avoid linker issues)
        for gf in sorted(cutlass_moe_gen.rglob("*.generated.cu")):
            compile_jobs.append((gf, sm90_flags, cutlass_moe_extra, vinfo_moe))
            cutlass_moe_compiled += 1

        sm90_modules.append(vinfo_moe)
        print(f"  SM90 CUTLASS MoE: {cutlass_moe_compiled} sources (BF16 only)")

        for vinfo in sm90_modules:
            utility_variants.append(([], [], vinfo))

    # ── SM100 modules (matching upstream aot.py has_sm100 blocks) ──────
    sm100_flags = blackwell_arch_flags(archs)
    if has_sm100:
        import jinja2
        sm100_modules = []
        csrc = fi_src / "csrc"

        def _add_sm100_module(vid, src_files, binding_file, symbols, kind="gemm", extra_flags=None):
            mod_out = gen_dir / vid
            mod_out.mkdir(parents=True, exist_ok=True)
            mod_sources = []
            for sf in src_files:
                sp = csrc / sf
                if sp.exists():
                    shutil.copy2(sp, mod_out / Path(sf).name)
                    mod_sources.append(mod_out / Path(sf).name)
            if binding_file:
                bp = csrc / binding_file
                if bp.exists():
                    shutil.copy2(bp, mod_out / Path(binding_file).name)
                    mod_sources.append(mod_out / Path(binding_file).name)
            vinfo = {"vid": vid, "kind": kind,
                     "symbols": {s: f"__tvm_ffi_{s}" for s in symbols}}
            ef = extra_flags or []
            for src in mod_sources:
                compile_jobs.append((src, sm100_flags, ef, vinfo))
            sm100_modules.append(vinfo)

        # gen_gemm_sm100_module_cutlass_fp8: FP8 GEMM via jinja. Production
        # BF16-only builds do not call these kernels, and they dominate AOT
        # build time on Blackwell, so only emit them for explicit FP8 builds.
        if any(dtype.lower().startswith("fp8") for dtype in dtypes):
            fp8_out = gen_dir / "fi_fp8_gemm_cutlass"
            fp8_out.mkdir(parents=True, exist_ok=True)
            fp8_sources = []
            # Copy main dispatch file
            sp = csrc / "fp8_gemm_cutlass.cu"
            if sp.exists():
                shutil.copy2(sp, fp8_out / "fp8_gemm_cutlass.cu")
                fp8_sources.append(fp8_out / "fp8_gemm_cutlass.cu")
            # Generate SM100 instantiation files from upstream jinja
            with open(csrc / "fp8_gemm_cutlass.jinja") as f:
                fp8_templ = jinja2.Template(f.read())
            cta_configs = [(64,64,128), (64,128,128), (64,256,128),
                           (128,64,128), (128,128,128), (128,256,128)]
            for cta_m, cta_n, cta_k in cta_configs:
                for dtype in ["__nv_bfloat16", "half"]:
                    src_text = fp8_templ.render(type=dtype, cta_m=cta_m, cta_n=cta_n, cta_k=cta_k)
                    fname = f"fp8_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                    fpath = fp8_out / fname
                    fpath.write_text(src_text)
                    fp8_sources.append(fpath)
            fp8_vinfo = {"vid": "fi_fp8_gemm_cutlass", "kind": "gemm",
                         "symbols": {"fp8_gemm": "__tvm_ffi_fp8_gemm",
                                     "fp8_gemm_tactic_num": "__tvm_ffi_fp8_gemm_tactic_num"}}
            fp8_extra = ["-DENABLE_BF16", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"]
            for src in fp8_sources:
                compile_jobs.append((src, sm100_flags, fp8_extra, fp8_vinfo))
            sm100_modules.append(fp8_vinfo)
            print(f"  SM100 FP8 GEMM: {len(fp8_sources)} sources (12 jinja + 1 dispatch)")
        else:
            print("  SM100 FP8 GEMM: skipped for non-FP8 build")

        # gen_tgv_gemm_sm10x_module: TGV decode GEMM via jinja (11 tile configs per dtype)
        with open(csrc / "tgv_gemm.jinja") as f:
            tgv_templ = jinja2.Template(f.read())
        tgv_cta_configs = [
            (64, 8, 6), (64, 8, 8), (64, 8, 10), (64, 8, 12),
            (64, 16, 6), (64, 16, 8), (64, 16, 10),
            (64, 32, 6), (64, 32, 8),
            (64, 64, 6),
            (128, 16, 6),
        ]
        tgv_dtypes = [dt for dt in ["bf16", "fp16"] if dt in dtypes]
        for tgv_dtype in tgv_dtypes:
            tgv_out = gen_dir / f"fi_tgv_gemm_{tgv_dtype}"
            tgv_out.mkdir(parents=True, exist_ok=True)
            tgv_sources = []
            # Dispatch TU (tgv_gemm.cu) defines torch_ext::tgv_gemm /
            # bf16_gemm + TVM_FFI exports — compile it once (under bf16) so
            # we don't get duplicate symbols when linked into a single static
            # archive. fp16 variant just provides the extra kernel tile .o's.
            if tgv_dtype == "bf16":
                sp = csrc / "tgv_gemm.cu"
                if sp.exists():
                    shutil.copy2(sp, tgv_out / "tgv_gemm.cu")
                    tgv_sources.append(tgv_out / "tgv_gemm.cu")
            for cta_m, cta_n, dma_stage in tgv_cta_configs:
                src_text = tgv_templ.render(cta_m=cta_m, cta_n=cta_n,
                                            dma_stage=dma_stage, dtype=tgv_dtype)
                fname = f"tgv_gemm_{tgv_dtype}_{cta_m}x{cta_n}_{dma_stage}.cu"
                fpath = tgv_out / fname
                fpath.write_text(src_text)
                tgv_sources.append(fpath)
            # Upstream tgv_gemm.cu only emits TVM_FFI_DLL_EXPORT for tgv_gemm
            # and tgv_gemm_tactic_num; bf16_gemm / bf16_gemm_tactic_num are
            # C++ wrappers without TVM FFI exports, so don't advertise them.
            tgv_vinfo = {"vid": f"fi_tgv_gemm_{tgv_dtype}", "kind": "gemm",
                         "symbols": {"tgv_gemm": "__tvm_ffi_tgv_gemm",
                                     "tgv_gemm_tactic_num": "__tvm_ffi_tgv_gemm_tactic_num"}}
            tgv_extra = ["-DCUTLASS_ENABLE_GDC_FOR_SM100=1"]
            for src in tgv_sources:
                compile_jobs.append((src, sm100_flags, tgv_extra, tgv_vinfo))
            sm100_modules.append(tgv_vinfo)
        print(f"  SM100 TGV GEMM: {len(tgv_dtypes)} dtypes × {len(tgv_cta_configs)+1} sources each")

        # gen_gemm_sm100_module: groupwise GEMM + binding
        _add_sm100_module("fi_gemm_sm100",
                          ["gemm_groupwise_sm100.cu"], "gemm_sm100_binding.cu",
                          ["gemm_fp8_nt_groupwise"])
        # gen_gemm_sm100_module_cutlass_fp4: FP4 GEMM (DISABLED)
        # The upstream jinja instantiations for genericFp4GemmKernelLauncher
        # are not expanded here, so the dispatch TU leaves undefined symbols.
        # Not needed for BF16/FP16 models.
        # _add_sm100_module("fi_fp4_gemm_sm100",
        #                   ["fp4_gemm_cutlass.cu"], None,
        #                   ["fp4_gemm", "fp4_gemm_tactic_num"])
        # gen_gemm_sm100_module_cutlass_mxfp8: MXFP8 GEMM (DISABLED — same
        # reason as FP4: template instantiations not generated).
        # _add_sm100_module("fi_mxfp8_gemm",
        #                   ["mxfp8_gemm_cutlass.cu"], None,
        #                   ["mxfp8_gemm", "mxfp8_gemm_tactic_num"])
        # Group GEMM SM100 (FP8 + MXFP4)
        _add_sm100_module("fi_group_gemm_sm100",
                          ["group_gemm_fp8_groupwise_sm100.cu",
                           "group_gemm_mxfp4_groupwise_sm100.cu"],
                          "group_gemm_sm100_binding.cu",
                          ["group_gemm_fp8_nt_groupwise", "group_gemm_mxfp4_nt_groupwise"])

        # ── CUTLASS Fused MoE Blackwell (BF16 path used by Qwen3 MoE) ──
        #
        # The Rust runner looks up a generic "init" utility symbol. Avoid
        # emitting both SM90 and Blackwell fused-MoE modules in one archive
        # until the loader grows arch-specific symbol names; production B300
        # builds pass only sm_103, so this is the path that matters.
        if not has_sm90:
            cutlass_moe_arch = 103 if any(a >= 103 and a < 120 for a in archs) else 100
            cutlass_moe_vinfo = {
                "vid": f"fi_cutlass_moe_sm{cutlass_moe_arch}",
                "kind": "cutlass_moe",
                "symbols": {"init": "__tvm_ffi_init"},
            }
            cutlass_moe_compiled = _add_cutlass_fused_moe_from_jit_spec(
                compile_jobs, fi_src, gen_dir, archs, cutlass_moe_vinfo,
            )
            sm100_modules.append(cutlass_moe_vinfo)
            print(f"  SM{cutlass_moe_arch} CUTLASS MoE: {cutlass_moe_compiled} sources (upstream JIT spec)")

        # Prelude's static AOT runtime does not load FlashInfer's TRT-LLM
        # allreduce/MNNVL modules. Keeping them out of single-node builds cuts
        # compile time and avoids widening the static link surface.

        for vinfo in sm100_modules:
            utility_variants.append(([], [], vinfo))
        print(f"  SM100: {len(sm100_modules)} modules total")

    # ── SM120 conditional modules ─────────────────────────────────────
    has_sm120 = any(a >= 120 for a in archs)
    if has_sm120:
        sm120_flags = ["-gencode", "arch=compute_120a,code=sm_120a"]
        sm120_modules = []
        csrc = fi_src / "csrc"

        def _add_sm120_module(vid, src_files, binding_file, symbols):
            mod_out = gen_dir / vid
            mod_out.mkdir(parents=True, exist_ok=True)
            mod_sources = []
            for sf in src_files:
                sp = csrc / sf
                if sp.exists():
                    shutil.copy2(sp, mod_out / Path(sf).name)
                    mod_sources.append(mod_out / Path(sf).name)
            if binding_file:
                bp = csrc / binding_file
                if bp.exists():
                    shutil.copy2(bp, mod_out / binding_file)
                    mod_sources.append(mod_out / binding_file)
            vinfo = {"vid": vid, "kind": "gemm",
                     "symbols": {s: f"__tvm_ffi_{s}" for s in symbols}}
            for src in mod_sources:
                compile_jobs.append((src, sm120_flags, [], vinfo))
            sm120_modules.append(vinfo)

        # FP4 GEMM SM120 (DISABLED — see SM100 comment)
        # _add_sm120_module("fi_fp4_gemm_sm120",
        #                   ["fp4_gemm_cutlass_sm120.cu"], None,
        #                   ["fp4_gemm", "fp4_gemm_tactic_num"])
        # Groupwise GEMM SM120
        _add_sm120_module("fi_gemm_groupwise_sm120",
                          ["gemm_groupwise_sm120.cu"], "gemm_sm120_binding.cu",
                          ["gemm_fp8_nt_groupwise"])
        # Group GEMM SM120 (FP8 + NVFP4 + MXFP4)
        _add_sm120_module("fi_group_gemm_sm120",
                          ["group_gemm_fp8_groupwise_sm120.cu",
                           "group_gemm_nvfp4_groupwise_sm120.cu",
                           "group_gemm_mxfp4_groupwise_sm120.cu"],
                          "group_gemm_sm120_binding.cu",
                          ["group_gemm_fp8_nt_groupwise",
                           "group_gemm_nvfp4_nt_groupwise",
                           "group_gemm_mxfp4_nt_groupwise"])

        for vinfo in sm120_modules:
            utility_variants.append(([], [], vinfo))
        print(f"  SM120 GEMM: {len(sm120_modules)} modules")

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
        # Tolerate failures for generated CUTLASS MoE template instantiations.
        # Some BF16 tile shapes hit a template bug in moe_gemm_tma_ws_launcher.inl
        # (FP8 block-scale parameter type mismatch). The runner has fallback tiles.
        moe_generated = [f for f in failed if f.endswith(".generated.cu")]
        critical = [f for f in failed if f not in moe_generated]
        if critical:
            print(f"\nFAILED to compile {len(critical)} critical files:", file=sys.stderr)
            for f in critical:
                print(f"  {f}", file=sys.stderr)
            sys.exit(1)
        if moe_generated:
            print(f"\n  Skipped {len(moe_generated)} non-critical CUTLASS MoE tile variants")

    if gdn_blackwell_vinfo is not None:
        gdn_bw_dir = obj_dir / "fi_gdn_blackwell"
        gdn_bw_ok = []
        for hq, hv in GDN_BLACKWELL_HEAD_PAIRS:
            name = f"gdn_prefill_sm100_bf16_h{hq}_hv{hv}"
            obj_path = gdn_bw_dir / f"{name}.o"
            if compile_gdn_blackwell_variant(obj_path, hq, hv, "bf16"):
                obj_files.append(str(obj_path))
                gdn_bw_ok.append(name)
        missing = sorted(set(gdn_blackwell_vinfo["symbols"].keys()) - set(gdn_bw_ok))
        if missing:
            print("\nFAILED to compile Blackwell GDN variants:", file=sys.stderr)
            for name in missing:
                print(f"  {name}", file=sys.stderr)
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
            # Merged: swa/softcap dispatched at runtime, not in key
            entry["hdim_qk"] = v["hdim_qk"]
            entry["hdim_vo"] = v["hdim_vo"]
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

    # SM100 FMHA (if compiled)
    if has_sm100:
        manifest["variants"].append({
            "vid": sm100_vinfo["vid"],
            "kind": sm100_vinfo["kind"],
            "symbols": sm100_vinfo["symbols"],
        })

    if gdn_blackwell_vinfo is not None:
        manifest["variants"].append({
            "vid": gdn_blackwell_vinfo["vid"],
            "kind": gdn_blackwell_vinfo["kind"],
            "symbols": gdn_blackwell_vinfo["symbols"],
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(obj_files)} objects, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
