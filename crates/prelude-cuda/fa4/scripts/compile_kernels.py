#!/usr/bin/env python3
"""
AOT compile FlashAttention-4 CuTeDSL kernels to .o object files.

Uses the same export path as FA4's own JITPersistentCache:
  compile → export_to_c(.o)

Each variant gets a unique function_name so that all .o files can be
statically linked into a single binary without symbol conflicts.
The TVM FFI entry point becomes __tvm_ffi_{variant_name} per variant.

Requirements (build machine only):
    pip install nvidia-cutlass-dsl quack-kernels torch  # CUDA torch

Usage (called automatically by build.rs):
    PYTHONPATH=/path/to/flash-attention \
    FLASH_ATTENTION_ARCH=sm_90 \
        python3 compile_kernels.py [--output-dir ../kernels] [--prototype]
"""

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Shared helpers from prelude-kernelbuild — symbol verification,
# manifest IO, parallel runner, standard argparse. Loaded via the
# PRELUDE_KB_SCRIPTS_DIR env var set by build.rs.
sys.path.insert(0, os.environ["PRELUDE_KB_SCRIPTS_DIR"])
from dsl_driver import (  # noqa: E402
    compute_script_hash,
    run_parallel,
    standard_argparse,
    verify_symbol,
    write_manifest,
)


def _patch_flash_attn_import():
    """Pre-register a fake flash_attn module to skip C extension imports."""
    if 'flash_attn' in sys.modules:
        return
    pythonpath = os.environ.get('PYTHONPATH', '')
    for p in pythonpath.split(':'):
        candidate = os.path.join(p, 'flash_attn')
        if os.path.isdir(candidate):
            mod = types.ModuleType('flash_attn')
            mod.__path__ = [candidate]
            mod.__package__ = 'flash_attn'
            mod.__file__ = os.path.join(candidate, '__init__.py')
            sys.modules['flash_attn'] = mod
            return
    raise ImportError("flash_attn not found in PYTHONPATH")

_patch_flash_attn_import()
os.environ.setdefault("FLASH_ATTENTION_FAKE_TENSOR", "1")
# Disable CLC work-stealing scheduler (new compile_key dimension since upstream 98024f9).
# CLC is SM90-only and opt-in; our AOT kernels use the default persistent scheduler.
os.environ.setdefault("FA_CLC", "0")
os.environ.setdefault("FA_DISABLE_2CTA", "0")


def get_arch_from_env():
    arch_str = os.environ.get("FLASH_ATTENTION_ARCH", "sm_90")
    return int("".join(c for c in arch_str.split("_")[1] if c.isdigit()))


@dataclass
class VariantSpec:
    """Specification for one kernel variant to compile."""
    head_dim: int
    head_dim_v: int          # usually = head_dim; DeepSeek MLA uses (192, 128)
    gqa_ratio: int
    causal: bool
    window: bool
    softcap: Optional[float]
    paged: bool
    paged_non_tma: bool      # if paged: True=cp.async (page_size!=tile_n), False=TMA
    dtype: str               # "bf16" or "fp16"
    has_seqused_q: bool      # whether seqused_q is passed (prefix cache Q trimming)


def compile_variant(spec: VariantSpec, arch: int, output_dir: Path):
    """Compile one FA4 kernel variant → .o object file."""
    pack_gqa = spec.gqa_ratio > 1

    # Build unique name
    parts = [f"fa4_fwd_hdim{spec.head_dim}"]
    if spec.head_dim_v != spec.head_dim:
        parts.append(f"hdimv{spec.head_dim_v}")
    parts.append(f"gqa{spec.gqa_ratio}")
    parts.append(spec.dtype)
    parts.append("causal" if spec.causal else "noncausal")
    if spec.window:
        parts.append("window")
    if pack_gqa:
        parts.append("packgqa")
    if spec.softcap is not None:
        parts.append(f"softcap{int(spec.softcap)}")
    if spec.paged:
        parts.append("paged")
    if spec.paged_non_tma:
        parts.append("cpasync")
    if spec.has_seqused_q:
        parts.append("seqused_q")
    parts.append(f"sm{arch}")
    name = "_".join(parts)

    obj_path = output_dir / f"{name}.o"

    if obj_path.exists():
        print(f"  {name}: already exists, skipping")
        return _variant_result(spec, name, pack_gqa, arch, obj_path)

    print(f"  Compiling {name}...")

    try:
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd
        import flash_attn.cute.interface as _fa4_iface

        # Monkey-patch FA4 to support (head_dim=512, head_dim_v=256) on SM90.
        # Upstream caps at 256; Gemma4 needs 512 for full-attention layers.
        # The V-split approach: head_dim=512 is the K-reduction dim (MMA iterates),
        # head_dim_v=256 is the output dim (within MMA N≤256 limit).
        _orig_validate = _fa4_iface._validate_head_dims
        def _patched_validate(head_dim, head_dim_v, compute_capability, alignment):
            if compute_capability == 9 and head_dim <= 512 and head_dim_v <= 256:
                assert head_dim % alignment == 0 and head_dim_v % alignment == 0
                return
            _orig_validate(head_dim, head_dim_v, compute_capability, alignment)
        _fa4_iface._validate_head_dims = _patched_validate

        _orig_tile_fwd = _fa4_iface._tile_size_fwd_sm90
        def _patched_tile_fwd(head_dim, head_dim_v, is_causal, is_local, **kwargs):
            if head_dim > 256:
                return _fa4_iface.FwdConfig(64, 32, True, False)
            return _orig_tile_fwd(head_dim, head_dim_v, is_causal, is_local, **kwargs)
        _fa4_iface._tile_size_fwd_sm90 = _patched_tile_fwd

        torch_dtype = torch.bfloat16 if spec.dtype == "bf16" else torch.float16

        num_heads_k = 8
        num_heads_q = num_heads_k * spec.gqa_ratio

        if spec.paged:
            # Paged mode: Q is varlen [total_q, Hq, D], K/V are paged [num_pages, page_size, Hk, Dv]
            batch_size = 1
            # TMA requires page_size == tile_n. tile_n depends on head_dim (SM90):
            #   hdim≤128 → 128, hdim≤192 → 112, hdim256 → 80
            # cp.async: any page_size != tile_n triggers the non-TMA code path.
            # The actual page_size is a runtime value from K tensor shape, not baked in.
            if spec.paged_non_tma:
                page_size = 1  # arbitrary != tile_n, triggers cp.async compile key
            else:
                # TMA: page_size must equal tile_n
                if spec.head_dim <= 128:
                    page_size = 128
                elif spec.head_dim <= 192:
                    page_size = 112
                elif spec.head_dim <= 256:
                    page_size = 80
                else:
                    page_size = 64
            num_pages = 2
            max_pages_per_seq = 2
            total_q = 128

            q = torch.empty(total_q, num_heads_q, spec.head_dim, dtype=torch_dtype, device="cuda")
            k = torch.empty(num_pages, page_size, num_heads_k, spec.head_dim, dtype=torch_dtype, device="cuda")
            v = torch.empty(num_pages, page_size, num_heads_k, spec.head_dim_v, dtype=torch_dtype, device="cuda")
            cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
            page_table = torch.zeros(batch_size, max_pages_per_seq, dtype=torch.int32, device="cuda")
            seqused_k = torch.tensor([page_size * max_pages_per_seq], dtype=torch.int32, device="cuda")

            kwargs = dict(
                causal=spec.causal,
                softmax_scale=1.0 / (spec.head_dim ** 0.5),
                cu_seqlens_q=cu_seqlens_q,
                page_table=page_table,
                seqused_k=seqused_k,
            )

            if spec.has_seqused_q:
                kwargs["seqused_q"] = torch.tensor([total_q], dtype=torch.int32, device="cuda")
        else:
            # Non-paged varlen mode: rank-3 tensors [total_tokens, num_heads, head_dim]
            total_tokens = 128
            q = torch.empty(total_tokens, num_heads_q, spec.head_dim, dtype=torch_dtype, device="cuda")
            k = torch.empty(total_tokens, num_heads_k, spec.head_dim, dtype=torch_dtype, device="cuda")
            v = torch.empty(total_tokens, num_heads_k, spec.head_dim_v, dtype=torch_dtype, device="cuda")
            cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

            kwargs = dict(
                causal=spec.causal,
                softmax_scale=1.0 / (spec.head_dim ** 0.5),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
            )

            if spec.has_seqused_q:
                kwargs["seqused_q"] = torch.tensor([total_tokens], dtype=torch.int32, device="cuda")

        if pack_gqa:
            kwargs["pack_gqa"] = True

        if spec.window:
            kwargs["window_size_left"] = 256
            kwargs["window_size_right"] = 0 if spec.causal else 256

        if spec.softcap is not None:
            kwargs["softcap"] = spec.softcap

        # Trigger JIT compilation
        _flash_attn_fwd(q, k, v, **kwargs)

        # Extract compiled function from cache
        cache = _flash_attn_fwd.compile_cache
        if not hasattr(cache, 'cache') or not cache.cache:
            print(f"    ERROR: compile_cache is empty after compilation")
            return None

        compiled_fn = list(cache.cache.values())[-1]

        compiled_fn.export_to_c(
            object_file_path=str(obj_path),
            function_name=name,
        )
        print(f"    Exported .o: {obj_path} ({obj_path.stat().st_size // 1024}KB)")

        verify_symbol(obj_path, name)

        cache.cache.clear()

        return _variant_result(spec, name, pack_gqa, arch, obj_path)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def _variant_result(spec: VariantSpec, name: str, pack_gqa: bool, arch: int, obj_path: Path):
    return {
        "name": name,
        "head_dim": spec.head_dim,
        "head_dim_v": spec.head_dim_v,
        "gqa_ratio": spec.gqa_ratio,
        "dtype": spec.dtype,
        "causal": spec.causal,
        "window": spec.window,
        "pack_gqa": pack_gqa,
        "softcap": spec.softcap,
        "paged": spec.paged,
        "paged_non_tma": spec.paged_non_tma,
        "has_seqused_q": spec.has_seqused_q,
        "arch": arch,
        "obj_path": str(obj_path),
        "symbol": f"__tvm_ffi_{name}",
    }


def _arch_constraints(arch: int):
    """Return arch-specific constraints mirroring _validate_head_dims and
    constructor asserts in flash_attn/cute/interface.py.

    Source of truth (interface.py):
      _validate_head_dims:
        SM90 (cap==9):     hdim 8-256, hdim_v 8-256
        SM100/110 (cap in [10,11]): hdim 8-128 + DeepSeek (192,128)
        SM80/SM120:        no validation (unconstrained)
      Constructor asserts:
        SM80:   no paged, no split_kv
        SM90:   no split_kv, paged supported
        SM100:  paged supported, split_kv supported
        SM120:  no paged, no split_kv
    """
    cap = arch // 10
    if cap == 9:
        # SM90: hdim 8-256 native, hdim 512 via V-split, paged yes
        return dict(
            head_dims=[64, 96, 128, 192, 256],
            supports_paged=True,
            supports_deepseek_mla=True,  # (192,128) within 8-256 range
            supports_hdim512_vsplit=True,  # Gemma4: (512,256) with V-split dispatch
        )
    elif cap in [10, 11]:
        # SM100/SM110: hdim 8-128 + DeepSeek (192,128), paged yes
        return dict(
            head_dims=[64, 96, 128],
            supports_paged=True,
            supports_deepseek_mla=True,
        )
    elif cap == 12:
        # SM120: no hdim validation in upstream, no paged, no DeepSeek tested
        return dict(
            head_dims=[64, 96, 128],
            supports_paged=False,
            supports_deepseek_mla=False,
        )
    else:
        # SM80 and others: conservative
        return dict(
            head_dims=[64, 96, 128],
            supports_paged=False,
            supports_deepseek_mla=False,
        )


def build_variant_matrix(arch: int, prototype=False):
    """Build the variant matrix for a specific SM architecture.

    Mirrors upstream compile_key dimensions from flash_attn/cute/interface.py.
    Filters out variants known to be unsupported per arch to avoid wasted compiles.
    """
    constraints = _arch_constraints(arch)
    head_dims = constraints["head_dims"]
    supports_paged = constraints["supports_paged"]
    supports_deepseek_mla = constraints["supports_deepseek_mla"]

    # Include odd ratios needed by common HF models:
    #   gqa=5: Qwen3-14B (40/8)
    #   gqa=6: Qwen3-32B (48/8)
    #   gqa=7: Llama-3-70B (56/8)
    gqa_ratios = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
    causal_modes = [True, False]
    window_modes = [False, True]
    dtypes = ["bf16", "fp16"]
    softcap_values = [30.0, 50.0]  # Gemma2=30, Gemma3=50
    softcap_gqa_limit = 2

    if prototype:
        return [
            VariantSpec(128, 128, 4, True, False, None, False, False, "bf16", False),
            VariantSpec(128, 128, 4, True, False, None, True, False, "bf16", False),
        ]

    variants = []

    # 1. Non-paged base: hdims × gqa × causal × window × dtype
    for dtype in dtypes:
        for hd in head_dims:
            for gqa in gqa_ratios:
                for c in causal_modes:
                    for w in window_modes:
                        variants.append(VariantSpec(hd, hd, gqa, c, w, None, False, False, dtype, False))

    # 2. Softcap: bf16 only, GQA ≤ 2 (Gemma models)
    for hd in head_dims:
        for gqa in gqa_ratios:
            if gqa > softcap_gqa_limit:
                continue
            for c in causal_modes:
                for w in window_modes:
                    for sc in softcap_values:
                        variants.append(VariantSpec(hd, hd, gqa, c, w, sc, False, False, "bf16", False))

    if supports_paged:
        # 3. Paged TMA (page_size=tile_n): both dtypes, causal only
        for dtype in dtypes:
            for hd in head_dims:
                for gqa in gqa_ratios:
                    variants.append(VariantSpec(hd, hd, gqa, True, False, None, True, False, dtype, False))

        # 4. Paged cp.async (page_size!=tile_n): bf16, causal only
        for hd in head_dims:
            for gqa in gqa_ratios:
                variants.append(VariantSpec(hd, hd, gqa, True, False, None, True, True, "bf16", False))

        # 5. Paged + seqused_q (prefix cache Q trimming): bf16, TMA, causal only
        for hd in head_dims:
            for gqa in gqa_ratios:
                variants.append(VariantSpec(hd, hd, gqa, True, False, None, True, False, "bf16", True))

    # 6. DeepSeek MLA shape (hdim_qk=192, hdim_v=128): bf16, non-paged
    if supports_deepseek_mla:
        for gqa in gqa_ratios:
            for c in causal_modes:
                variants.append(VariantSpec(192, 128, gqa, c, False, None, False, False, "bf16", False))

    # 7. Gemma4 hdim=512 via V-split: compile (hdim_qk=512, hdim_v=256).
    #    At runtime, V is split into two 256-wide halves, the kernel is called
    #    twice, and results are concatenated.  Q@K^T uses hdim=512 as the
    #    reduction dimension (K-mode), which SM90 MMA handles via iteration.
    if constraints.get("supports_hdim512_vsplit"):
        for gqa in gqa_ratios:
            for c in causal_modes:
                for w in window_modes:
                    variants.append(VariantSpec(512, 256, gqa, c, w, None, False, False, "bf16", False))

    return variants


def _compile_worker(task):
    """Worker function for parallel compilation. Runs in a subprocess."""
    spec, arch, output_dir = task
    return compile_variant(spec, arch, output_dir)


def main():
    parser = standard_argparse(
        description="AOT compile FA4 kernels",
        default_output_dir=Path(__file__).parent.parent / "kernels",
        default_workers=1,
    )
    args = parser.parse_args()

    arch = args.arch or get_arch_from_env()
    os.environ["FLASH_ATTENTION_ARCH"] = f"sm_{arch}"
    os.environ["CUTE_DSL_ARCH"] = f"sm_{arch}a"

    print(f"Target: SM{arch}")
    print(f"Output: {args.output_dir}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Multi-arch incremental: keep variants from other archs already in
    # the manifest so that compiling sm_100 after sm_90 doesn't drop
    # the existing sm_90 entries.
    from dsl_driver import load_existing_manifest  # noqa: E402
    existing = load_existing_manifest(args.output_dir / "manifest.json")
    other_arch_variants = [v for v in existing if v.get("arch") != arch]

    variants = build_variant_matrix(arch, prototype=args.prototype)

    print(f"Compiling {len(variants)} kernel variants for SM{arch} (workers={args.workers})...\n")

    tasks = [(spec, arch, args.output_dir) for spec in variants]
    results = run_parallel(tasks, _compile_worker, max_workers=args.workers)

    # Merge: other archs + this arch
    all_variants = other_arch_variants + results
    all_archs = sorted(set(v["arch"] for v in all_variants))

    write_manifest(
        manifest_path=args.output_dir / "manifest.json",
        variants=all_variants,
        script_hash=compute_script_hash(Path(__file__)),
        archs=all_archs,
    )

    # Summary
    n_paged = sum(1 for v in results if v.get("paged"))
    n_softcap = sum(1 for v in results if v.get("softcap") is not None)
    n_fp16 = sum(1 for v in results if v.get("dtype") == "fp16")
    n_seqused_q = sum(1 for v in results if v.get("has_seqused_q"))
    n_deepseek = sum(1 for v in results if v.get("head_dim_v") != v.get("head_dim"))
    n_base = len(results) - n_paged - n_softcap - n_deepseek

    print(f"\n=== Results: {len(results)}/{len(variants)} compiled for SM{arch} ===")
    print(f"  Non-paged base: {n_base} (FP16: {n_fp16})")
    print(f"  Softcap: {n_softcap}")
    print(f"  Paged: {n_paged} (seqused_q: {n_seqused_q})")
    print(f"  DeepSeek MLA (192,128): {n_deepseek}")
    if all_archs != [arch]:
        print(f"  Multi-arch total: {len(all_variants)} variants across {all_archs}")
    if len(results) < len(variants):
        failed = len(variants) - len(results)
        print(f"  Failed: {failed} (expected for unsupported combos)")
        sys.exit(1)


if __name__ == "__main__":
    main()
