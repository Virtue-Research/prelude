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

# POD only needs causal + custom (spec decode). kNone and kMultiItemScoring
# are meaningless in mixed prefill+decode batches.
POD_MASK_MODES = [1, 2]  # kCausal, kCustom


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
    renames = {
        "BatchDecodeWithPagedKVCacheRun": f"{vid}_BatchDecodeWithPagedKVCacheRun",
        "BatchDecodeWithPagedKVCachePlan": f"{vid}_BatchDecodeWithPagedKVCachePlan",
    }
    _copy_with_renames(csrc / "batch_decode.cu", out / "batch_decode.cu", renames)
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


def generate_batch_pod_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim_qk: int, hdim_vo: int,
    swa_p: bool, softcap_p: bool, swa_d: bool, softcap_d: bool,
) -> Tuple[List[Path], List[str]]:
    """Generate source files for a batch POD (Prefill+Decode) variant (unmerged, legacy).

    POD dispatches both prefill and decode within a single kernel launch
    using SM-aware scheduling. Uses FA2 kernels (SM80+).
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    variant_name_p = (
        f"DefaultAttention<use_custom_mask_p, {str(swa_p).lower()}, "
        f"{str(softcap_p).lower()}, false>"
    )
    variant_name_d = (
        f"DefaultAttention<use_custom_mask_d, {str(swa_d).lower()}, "
        f"{str(softcap_d).lower()}, false>"
    )
    kwargs = {
        "dtype_q": dtype_c, "dtype_kv": dtype_c, "dtype_o": dtype_c,
        "idtype": "int32_t",
        "head_dim_qk": hdim_qk, "head_dim_vo": hdim_vo,
        "pos_encoding_mode_p": "PosEncodingMode::kNone",
        "pos_encoding_mode_d": "PosEncodingMode::kNone",
        "use_sliding_window_p": str(swa_p).lower(),
        "use_logits_soft_cap_p": str(softcap_p).lower(),
        "use_sliding_window_d": str(swa_d).lower(),
        "use_logits_soft_cap_d": str(softcap_d).lower(),
        "use_fp16_qk_reduction": "false",
        "variant_name_p": variant_name_p,
        "variant_name_d": variant_name_d,
    }

    # Config
    with open(csrc / "batch_pod_customize_config.jinja") as f:
        config_str = jinja2.Template(f.read()).render(**kwargs)
    (out / "batch_pod_config.inc").write_text(config_str)

    # Kernel instantiations for all mask mode combos (4×4 = 16 files)
    with open(csrc / "batch_pod_kernel_inst.jinja") as f:
        kernel_templ = jinja2.Template(f.read())

    sources = []
    for mm_p in NEEDED_MASK_MODES:
        for mm_d in NEEDED_MASK_MODES:
            fname = f"batch_pod_kernel_mask_{mm_p}p_{mm_d}d.cu"
            src = kernel_templ.render(
                mask_mode_p=MASK_MODES[mm_p],
                mask_mode_d=MASK_MODES[mm_d],
                **kwargs,
            )
            (out / fname).write_text(src)
            sources.append(out / fname)

    # batch_pod.cu — rename the main function per variant
    renames = {
        "batch_pod_with_kv_cache_tensor": f"{vid}_batch_pod_with_kv_cache_tensor",
    }
    _copy_with_renames(csrc / "batch_pod.cu", out / "batch_pod.cu", renames)
    sources.append(out / "batch_pod.cu")

    # Binding — TVM FFI export with variant-specific symbol
    binding_src = _generate_renamed_binding(
        csrc / "batch_pod_jit_binding.cu",
        "batch_pod_config.inc",
        {
            "batch_pod_with_kv_cache_tensor": f"__tvm_ffi_{vid}_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_pod_binding.cu").write_text(binding_src)
    sources.append(out / "batch_pod_binding.cu")

    return sources, []


def generate_batch_pod_merged_sources(
    fi_src: Path, gen_dir: Path, vid: str,
    dtype_c: str, dtype_name: str, hdim_qk: int, hdim_vo: int,
) -> Tuple[List[Path], List[str]]:
    """Generate MERGED source files for a batch POD (Prefill+Decode) variant.

    All swa/softcap/mask_mode combos in ONE compilation unit per (dtype, hdim).
    Same merging approach as FA2 prefill: swa/softcap dispatched at runtime via
    lambda templates, mask modes via explicit template instantiation.

    For inference: swa_p==swa_d and softcap_p==softcap_d (same model), so we
    dispatch on 2 booleans (swa, softcap) → 4 combos.

    Per CTA_TILE_Q file: 4 swa/cap × 16 mask pairs = 64 instantiations.
    Split into 3 files by CTA_TILE_Q → 64 instantiations each (like FA3 prefill).
    """
    import jinja2

    csrc = fi_src / "csrc"
    out = gen_dir / vid
    out.mkdir(parents=True, exist_ok=True)

    # ── Merged config.inc ──────────────────────────────────────────
    # Render with dummy swa/softcap (overridden by DISPATCH_context below)
    with open(csrc / "batch_pod_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        dtype_q=dtype_c, dtype_kv=dtype_c, dtype_o=dtype_c,
        idtype="int32_t",
        head_dim_qk=hdim_qk, head_dim_vo=hdim_vo,
        pos_encoding_mode_p="PosEncodingMode::kNone",
        pos_encoding_mode_d="PosEncodingMode::kNone",
        use_sliding_window_p="false", use_logits_soft_cap_p="false",
        use_sliding_window_d="false", use_logits_soft_cap_d="false",
        use_fp16_qk_reduction="false",
        variant_name_p="DefaultAttention<use_custom_mask_p, false, false, false>",
        variant_name_d="DefaultAttention<use_custom_mask_d, false, false, false>",
    )

    # Replace DISPATCH_context to add runtime swa/softcap lambda dispatch.
    # Original: nested DISPATCH_MASK_MODE only.
    # Merged: DISPATCH_MASK_MODE + runtime swa/softcap via lambda.
    old_dispatch = (
        '#define DISPATCH_context(MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK,    \\\n'
        '            USE_SLIDING_WINDOW_P, USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, ...)   \\\n'
        '  DISPATCH_MASK_MODE(mask_mode_p, MASK_MODE_P, {                                    \\\n'
        '    DISPATCH_MASK_MODE(mask_mode_d, MASK_MODE_D, {                                  \\\n'
        '      __VA_ARGS__();                                                                \\\n'
        '    });                                                                             \\\n'
        '});'
    )
    # POD uses window_left_p/logits_soft_cap_p (not window_left/logits_soft_cap).
    # For inference: swa_p==swa_d, softcap_p==softcap_d, so dispatch on _p variant.
    new_dispatch = (
        '#define DISPATCH_context(MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK,    \\\n'
        '            USE_SLIDING_WINDOW_P, USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, ...)   \\\n'
        '  DISPATCH_MASK_MODE(mask_mode_p, MASK_MODE_P, {                                    \\\n'
        '    DISPATCH_MASK_MODE(mask_mode_d, MASK_MODE_D, {                                  \\\n'
        '      auto _run = [&]<bool _SWA, bool _CAP>() {                                    \\\n'
        '        constexpr auto USE_SLIDING_WINDOW_P = _SWA;                                 \\\n'
        '        constexpr auto USE_SLIDING_WINDOW_D = _SWA;                                 \\\n'
        '        constexpr auto USE_LOGITS_SOFT_CAP_P = _CAP;                                \\\n'
        '        constexpr auto USE_LOGITS_SOFT_CAP_D = _CAP;                                \\\n'
        '        constexpr auto use_custom_mask_p = MASK_MODE_P == MaskMode::kCustom;         \\\n'
        '        constexpr auto use_custom_mask_d = MASK_MODE_D == MaskMode::kCustom;         \\\n'
        '        __VA_ARGS__();                                                              \\\n'
        '      };                                                                            \\\n'
        '      bool _swa = (window_left_p >= 0);                                             \\\n'
        '      bool _cap = (logits_soft_cap_p > 0.0);                                        \\\n'
        '      if (!_swa && !_cap) _run.template operator()<false, false>();                  \\\n'
        '      else if (_swa && !_cap) _run.template operator()<true, false>();               \\\n'
        '      else if (!_swa && _cap) _run.template operator()<false, true>();               \\\n'
        '      else _run.template operator()<true, true>();                                   \\\n'
        '    });                                                                             \\\n'
        '});'
    )
    old_config = config_str
    config_str = config_str.replace(old_dispatch, new_dispatch)
    assert config_str != old_config, (
        "DISPATCH_context replacement failed in batch_pod — upstream macro may have changed"
    )

    # Remove the constexpr swa/softcap lines (they're now inside the lambda)
    for line in [
        "constexpr auto USE_LOGITS_SOFT_CAP_P = false;",
        "constexpr auto USE_SLIDING_WINDOW_P = false;",
        "constexpr auto USE_LOGITS_SOFT_CAP_D = false;",
        "constexpr auto USE_SLIDING_WINDOW_D = false;",
    ]:
        config_str = config_str.replace(line, f"// merged: {line.strip()} (dispatched at runtime)")

    # Override DISPATCH_MASK_MODE to only emit kCausal + kCustom cases.
    # The upstream macro generates switch cases for all 4 mask modes, but POD only
    # needs kCausal (normal serving) and kCustom (spec decode). Without this override,
    # the linker would require template instantiations for kNone and kMultiItemScoring
    # which we don't compile.
    config_str += r"""

// Override DISPATCH_MASK_MODE: POD only needs kCausal + kCustom
#ifdef DISPATCH_MASK_MODE
#undef DISPATCH_MASK_MODE
#endif
#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)                          \
  switch (mask_mode) {                                                         \
    case MaskMode::kCausal: {                                                  \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal;                        \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case MaskMode::kCustom: {                                                  \
      constexpr MaskMode MASK_MODE = MaskMode::kCustom;                        \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "POD only supports kCausal and kCustom mask modes, got: "     \
              << int(mask_mode);                                               \
      throw flashinfer::Error(__FUNCTION__, __FILE__, __LINE__, err_msg.str());\
    }                                                                          \
  }
"""

    (out / "batch_pod_config.inc").write_text(config_str)

    # ── Merged kernel instantiations (split by CTA_TILE_Q) ────────
    # POD only needs kCausal + kCustom mask modes (POD_MASK_MODES).
    # Per file: 4 swa/cap combos × 4 mask pairs (2×2) = 16 instantiations
    # 3 files (CTA_TILE_Q ∈ {16, 64, 128}) → 48 total
    swa_cap_combos = [(False, False), (True, False), (False, True), (True, True)]
    pod_masks = {k: v for k, v in MASK_MODES.items() if k in POD_MASK_MODES}

    sources = []
    for cta in [16, 64, 128]:
        lines = [
            '#include <flashinfer/attention/default_prefill_params.cuh>',
            '#include <flashinfer/attention/default_decode_params.cuh>',
            '#include <flashinfer/attention/variants.cuh>',
            '#include <flashinfer/attention/scheduler.cuh>',
            '#include <flashinfer/attention/mask.cuh>',
            '#include <flashinfer/attention/batch_pod.cuh>',
            '#include <flashinfer/pos_enc.cuh>',
            '#include <flashinfer/utils.cuh>',
            '#include <flashinfer/page.cuh>',
            '',
            '#include "batch_pod_config.inc"',
            '',
            'using namespace flashinfer;',
            '',
            'namespace flashinfer {',
            '',
        ]
        for _, mm_p_name in pod_masks.items():
            for _, mm_d_name in pod_masks.items():
                for swa, cap in swa_cap_combos:
                    var_p = (
                        f"DefaultAttention<{mm_p_name} == MaskMode::kCustom, "
                        f"{str(swa).lower()}, {str(cap).lower()}, false>"
                    )
                    var_d = (
                        f"DefaultAttention<{mm_d_name} == MaskMode::kCustom, "
                        f"{str(swa).lower()}, {str(cap).lower()}, false>"
                    )
                    lines.append(
                        f"template cudaError_t BatchPODWithKVCacheTensorDispatched<"
                        f"{hdim_qk}, {hdim_vo}, PosEncodingMode::kNone, "
                        f"false, /*CTA_TILE_Q_P=*/{cta}, {mm_p_name}, "
                        f"/*CTA_TILE_Q_D=*/16, {mm_d_name}, "
                        f"{var_p}, {var_d}, PrefillParams, DecodeParams>"
                        f"(PrefillParams prefill_params, {dtype_c}* tmp_v_p, float* tmp_s_p, "
                        f"DecodeParams decode_params, {dtype_c}* tmp_v_d, float* tmp_s_d, "
                        f"bool enable_pdl, cudaStream_t stream, int* sm_aware_sched);"
                    )
            lines.append('')
        lines.append('};  // namespace flashinfer')

        kernel_path = out / f"merged_kernels_cta{cta}.cu"
        kernel_path.write_text('\n'.join(lines))
        sources.append(kernel_path)

    # ── batch_pod.cu (dispatch) — reuse upstream with renamed symbols ──
    renames = {
        "batch_pod_with_kv_cache_tensor": f"{vid}_batch_pod_with_kv_cache_tensor",
    }
    _copy_with_renames(csrc / "batch_pod.cu", out / "batch_pod.cu", renames)
    sources.append(out / "batch_pod.cu")

    # ── Binding — TVM FFI export with variant-specific symbol ──
    binding_src = _generate_renamed_binding(
        csrc / "batch_pod_jit_binding.cu",
        "batch_pod_config.inc",
        {
            "batch_pod_with_kv_cache_tensor": f"__tvm_ffi_{vid}_run",
        },
    )
    for old, new in renames.items():
        binding_src = binding_src.replace(old, new)
    (out / "batch_pod_binding.cu").write_text(binding_src)
    sources.append(out / "batch_pod_binding.cu")

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
        # FP4 KV cache dequantization (SM80+, LUT-based)
        ("fi_fp4_dequant", ["fp4_kv_dequantization.cu"], None, "fp4",
         ["nvfp4_kv_dequant"]),
        # FP4 KV cache quantization (SM100+ for HW path, software fallback for SM80+)
        ("fi_fp4_quant", ["fp4_kv_quantization.cu"], None, "fp4",
         ["nvfp4_kv_quant"]),
        # Quantization utilities: packbits / segment_packbits (GPU kernels for sparse mask packing)
        ("fi_quantization", ["quantization.cu"], "flashinfer_quantization_binding.cu", "quantization",
         ["packbits", "segment_packbits"]),
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
                    # Patch: exclude layernorm (requires TensorRT-LLM headers
                    # that the upstream norm.cuh pulls in for LayerNorm).
                    src_text = src_path.read_text()
                    src_text = src_text.replace(
                        '#include <flashinfer/norm.cuh>',
                        '#define FLASHINFER_NORM_NO_LAYERNORM\n#include <flashinfer/norm.cuh>'
                    )
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

    # ── POD (Prefill+Decode) mixed batching ─────────────────────────
    # MERGED: all swa/softcap/mask_mode combos in one compilation unit per (dtype, hdim).
    # Split by CTA_TILE_Q into 3 .cu files, each with 64 instantiations (like FA3 prefill).
    # Assumes swa_p==swa_d, softcap_p==softcap_d (same model in continuous batching).
    # SM80 only: POD is FA2-based, on SM90+ separate FA3 prefill + FA2 decode is faster.
    sm80_only_flags = ["-gencode", "arch=compute_80,code=compute_80"]
    for dtype in dtypes:
        dtype_c, dtype_name = DTYPE_MAP[dtype]
        for hdim in head_dims:
            vid = variant_id("pod", "fa2", dtype, hdim, hdim)
            variants.append({
                "vid": vid, "kind": "pod_merged", "backend": "fa2",
                "dtype": dtype, "dtype_c": dtype_c, "dtype_name": dtype_name,
                "hdim_qk": hdim, "hdim_vo": hdim,
                "arch_flags": sm80_only_flags,
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
        elif v["kind"] == "pod":
            sources, extra = generate_batch_pod_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
                v["swa_p"], v["softcap_p"],
                v["swa_d"], v["softcap_d"],
            )
        elif v["kind"] == "pod_merged":
            sources, extra = generate_batch_pod_merged_sources(
                fi_src, gen_dir, v["vid"],
                v["dtype_c"], v["dtype_name"],
                v["hdim_qk"], v["hdim_vo"],
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
        sm100_flags = ["-gencode", "arch=compute_100a,code=sm_100a"]
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
        if v["kind"] in ("decode", "prefill_fa2", "prefill_fa3", "prefill_fp8", "pod_merged"):
            # Merged: swa/softcap dispatched at runtime, not in key
            entry["hdim_qk"] = v["hdim_qk"]
            entry["hdim_vo"] = v["hdim_vo"]
        if v["kind"] in ("mla_decode", "mla_paged"):
            entry["head_dim_ckv"] = v["head_dim_ckv"]
            entry["head_dim_kpe"] = v["head_dim_kpe"]
        if v["kind"] == "pod":
            entry["hdim_qk"] = v["hdim_qk"]
            entry["hdim_vo"] = v["hdim_vo"]
            entry["swa_p"] = v["swa_p"]
            entry["softcap_p"] = v["softcap_p"]
            entry["swa_d"] = v["swa_d"]
            entry["softcap_d"] = v["softcap_d"]

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
        elif v["kind"] in ("pod", "pod_merged"):
            entry["symbols"] = {
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

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(obj_files)} objects, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
