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

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path


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


def get_arch_from_env():
    arch_str = os.environ.get("FLASH_ATTENTION_ARCH", "sm_90")
    return int("".join(c for c in arch_str.split("_")[1] if c.isdigit()))


def compile_variant(head_dim, gqa_ratio, causal, window, arch, output_dir,
                    dtype="bf16", softcap=None):
    """Compile one FA4 kernel variant → .o object file.

    Each variant gets a unique function_name so that __tvm_ffi_{name} symbols
    don't conflict when all .o files are statically linked together.

    Args:
        head_dim: Head dimension (64, 128, 256)
        gqa_ratio: num_heads_q / num_heads_k (1=MHA, 2/4/8/16/32=GQA)
        causal: Causal masking
        window: Sliding window attention (uses window_size_left/right)
        arch: SM architecture (90, 100, 120, etc.)
        output_dir: Output directory for .o files
        dtype: "bf16" or "fp16"
        softcap: Optional attention logit soft-capping value (e.g. 30.0, 50.0)
    """
    # pack_gqa is auto-derived: always True when GQA > 1
    pack_gqa = gqa_ratio > 1

    causal_str = "causal" if causal else "noncausal"
    window_str = "_window" if window else ""
    packgqa_str = "_packgqa" if pack_gqa else ""
    softcap_str = f"_softcap{int(softcap)}" if softcap is not None else ""
    name = (f"fa4_fwd_hdim{head_dim}_gqa{gqa_ratio}_{dtype}_{causal_str}"
            f"{window_str}{packgqa_str}{softcap_str}_sm{arch}")
    obj_path = output_dir / f"{name}.o"

    if obj_path.exists():
        print(f"  {name}: already exists, skipping")
        return _variant_result(name, head_dim, gqa_ratio, causal, window,
                               pack_gqa, softcap, arch, dtype, obj_path)

    print(f"  Compiling {name}...")

    try:
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        num_heads_k = 8
        num_heads_q = num_heads_k * gqa_ratio
        total_tokens = 128

        # Varlen mode: rank-3 tensors [total_tokens, num_heads, head_dim]
        q = torch.empty(total_tokens, num_heads_q, head_dim, dtype=torch_dtype, device="cuda")
        k = torch.empty(total_tokens, num_heads_k, head_dim, dtype=torch_dtype, device="cuda")
        v = torch.empty(total_tokens, num_heads_k, head_dim, dtype=torch_dtype, device="cuda")
        cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

        kwargs = dict(
            causal=causal,
            softmax_scale=1.0 / (head_dim ** 0.5),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )

        # pack_gqa: auto-enabled for GQA > 1 if upstream supports it
        if pack_gqa:
            try:
                kwargs["pack_gqa"] = True
            except Exception:
                pass  # upstream auto-derives from tensor shapes

        if window:
            kwargs["window_size_left"] = 256
            kwargs["window_size_right"] = 0 if causal else 256

        if softcap is not None:
            kwargs["softcap"] = softcap

        # Trigger JIT compilation
        _flash_attn_fwd(q, k, v, **kwargs)

        # Extract compiled function from cache
        cache = _flash_attn_fwd.compile_cache
        if not hasattr(cache, 'cache') or not cache.cache:
            print(f"    ERROR: compile_cache is empty after compilation")
            return None

        # Get the last compiled function (the one we just triggered)
        compiled_fn = list(cache.cache.values())[-1]

        # Export .o with unique function_name per variant.
        # This makes __tvm_ffi_{name} the entry symbol instead of __tvm_ffi_func,
        # so all .o files can be statically linked without symbol conflicts.
        compiled_fn.export_to_c(
            object_file_path=str(obj_path),
            function_name=name,
        )
        print(f"    Exported .o: {obj_path} ({obj_path.stat().st_size // 1024}KB)")

        # Verify the .o has the expected unique symbol
        _verify_symbol(obj_path, name)

        # Clear cache so next variant starts fresh
        cache.cache.clear()

        return _variant_result(name, head_dim, gqa_ratio, causal, window,
                               pack_gqa, softcap, arch, dtype, obj_path)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def _verify_symbol(obj_path, name):
    """Verify the .o has __tvm_ffi_{name} and NOT __tvm_ffi_func."""
    try:
        result = subprocess.run(
            ["nm", "-g", str(obj_path)],
            capture_output=True, text=True, timeout=10,
        )
        symbols = result.stdout
        expected = f"__tvm_ffi_{name}"
        if expected in symbols:
            print(f"    Symbol OK: {expected}")
        elif "__tvm_ffi_func" in symbols:
            # Fallback: rename with objcopy
            print(f"    WARNING: found __tvm_ffi_func, renaming to {expected}")
            subprocess.run([
                "objcopy",
                f"--redefine-sym=__tvm_ffi_func={expected}",
                str(obj_path),
            ], check=True, timeout=10)
            print(f"    Renamed OK")
        else:
            print(f"    WARNING: neither {expected} nor __tvm_ffi_func found in symbols")
    except Exception as e:
        print(f"    Symbol check skipped: {e}")


def _variant_result(name, head_dim, gqa_ratio, causal, window,
                    pack_gqa, softcap, arch, dtype, obj_path):
    return {
        "name": name, "head_dim": head_dim, "gqa_ratio": gqa_ratio,
        "dtype": dtype, "causal": causal, "window": window,
        "pack_gqa": pack_gqa, "softcap": softcap,
        "arch": arch, "obj_path": str(obj_path),
        "symbol": f"__tvm_ffi_{name}",
    }


def build_variant_matrix(prototype=False):
    """Build the full variant matrix.

    Base matrix: 3 hdim × 6 gqa × 2 causal × 2 window = 72 variants (no softcap)
    Softcap variants: only for GQA ≤ 2 (Gemma uses GQA=1)
        3 hdim × 2 gqa(1,2) × 2 causal × 2 window × 2 softcap(30,50) = 48 variants
    Total: 72 + 48 = 120 variants per arch.
    """
    head_dims = [64, 128, 256]
    gqa_ratios = [1, 2, 4, 8, 16, 32]
    causal_modes = [True, False]
    window_modes = [False, True]
    softcap_values = [30.0, 50.0]  # Gemma2=30, Gemma3=50
    softcap_gqa_limit = 2  # only compile softcap for GQA ≤ this

    if prototype:
        return [(128, 4, True, False, None)]  # Single: hdim128, GQA=4, causal, no window, no softcap

    variants = []

    # Base matrix: no softcap (all GQA combos)
    for hd in head_dims:
        for gqa in gqa_ratios:
            for c in causal_modes:
                for w in window_modes:
                    variants.append((hd, gqa, c, w, None))

    # Softcap variants: only for GQA ≤ softcap_gqa_limit
    for hd in head_dims:
        for gqa in gqa_ratios:
            if gqa > softcap_gqa_limit:
                continue
            for c in causal_modes:
                for w in window_modes:
                    for sc in softcap_values:
                        variants.append((hd, gqa, c, w, sc))

    return variants


def load_existing_manifest(output_dir):
    """Load existing manifest to support incremental multi-arch compilation."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            return data.get("variants", [])
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def main():
    parser = argparse.ArgumentParser(description="AOT compile FA4 kernels")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "kernels")
    parser.add_argument("--arch", type=int, default=None,
                        help="SM arch (e.g. 90, 120). Default: auto-detect from env")
    parser.add_argument("--prototype", action="store_true",
                        help="Single variant (hdim128, bf16, causal)")
    args = parser.parse_args()

    arch = args.arch or get_arch_from_env()
    os.environ["FLASH_ATTENTION_ARCH"] = f"sm_{arch}"
    os.environ["CUTE_DSL_ARCH"] = f"sm_{arch}a"

    print(f"Target: SM{arch}")
    print(f"Output: {args.output_dir}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing variants from other archs (supports incremental multi-arch builds:
    # run once per arch, each run adds to the manifest without clobbering the other arch)
    existing = load_existing_manifest(args.output_dir)
    # Keep variants from OTHER archs
    other_arch_variants = [v for v in existing if v.get("arch") != arch]

    variants = build_variant_matrix(prototype=args.prototype)

    print(f"Compiling {len(variants)} kernel variants for SM{arch}...\n")

    results = []
    for head_dim, gqa_ratio, causal, window, softcap in variants:
        result = compile_variant(head_dim, gqa_ratio, causal, window, arch,
                                 args.output_dir, softcap=softcap)
        if result:
            results.append(result)

    # Merge: other archs + this arch
    all_variants = other_arch_variants + results
    all_archs = sorted(set(v["arch"] for v in all_variants))

    manifest = {"archs": all_archs, "variants": all_variants}
    try:
        import cutlass
        manifest["cutlass_version"] = cutlass.__version__
    except (ImportError, AttributeError):
        pass

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Results: {len(results)}/{len(variants)} compiled for SM{arch} ===")
    print(f"  Base variants (no softcap): {sum(1 for v in results if v['softcap'] is None)}")
    print(f"  Softcap variants: {sum(1 for v in results if v['softcap'] is not None)}")
    if all_archs != [arch]:
        print(f"  Multi-arch total: {len(all_variants)} variants across {all_archs}")
    if len(results) < len(variants):
        sys.exit(1)


if __name__ == "__main__":
    main()
