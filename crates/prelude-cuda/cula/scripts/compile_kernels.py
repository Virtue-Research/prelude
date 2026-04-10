#!/usr/bin/env python3
"""
AOT compile cuLA CuTe DSL kernels to .o object files.

Uses the same export path as FA4: cute.compile → export_to_c(.o).
Each variant gets a unique function_name (__tvm_ffi_{name}).

Requirements (build machine only):
    pip install nvidia-cutlass-dsl torch

Usage (called automatically by build.rs):
    PYTHONPATH=/path/to/cuLA \
    CUTE_DSL_ARCH=sm_90a \
        python3 compile_kernels.py [--output-dir ../kernels] [-j N]
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_arch_from_env():
    arch_str = os.environ.get("CUTE_DSL_ARCH", "sm_90a")
    return int("".join(c for c in arch_str.split("_")[1] if c.isdigit()))


# ---------------------------------------------------------------------------
# Variant specs
# ---------------------------------------------------------------------------

@dataclass
class LightningPrefillSpec:
    """Lightning Attention prefill (chunkwise decay)."""
    H: int
    D: int
    chunk_size: int
    has_initial_state: bool
    output_final_state: bool
    scale: float = 1.0


@dataclass
class LightningDecodeSpec:
    """Lightning Attention decode (single token)."""
    B: int
    H: int
    K: int  # head dim
    scale: float = 1.0


@dataclass
class ChunkDeltaHSpec:
    """Chunk delta-H (inter-chunk state update for KDA)."""
    H: int
    K: int
    V: int
    chunk_size: int
    is_varlen: bool
    persistent: bool


@dataclass
class FwdOSpec:
    """Forward output computation for KDA."""
    H: int
    K: int
    V: int
    chunk_size: int
    scale: float
    is_varlen: bool
    persistent: bool


@dataclass
class KdaDecodeSpec:
    """KDA single-token decode (CuTe DSL, Hopper+)."""
    H: int     # query heads
    HV: int    # value heads (equal to H for MHA, >H for GQA)
    K: int     # head_k_dim (must equal TILE_K=128 per the kernel)
    V: int     # head_v_dim
    variant: str  # one of: small_dense, small_varlen, large_dense, large_varlen
    use_qk_l2norm: bool
    softplus_beta: float = 1.0
    softplus_threshold: float = 20.0


# ---------------------------------------------------------------------------
# Compilation functions
# ---------------------------------------------------------------------------

def compile_lightning_prefill(spec: LightningPrefillSpec, arch: int, output_dir: Path):
    parts = [
        "cula_la_prefill",
        f"h{spec.H}",
        f"d{spec.D}",
        f"cs{spec.chunk_size}",
    ]
    if spec.has_initial_state:
        parts.append("initstate")
    if spec.output_final_state:
        parts.append("finalstate")
    parts.append(f"sm{arch}")
    name = "_".join(parts)
    obj_path = output_dir / f"{name}.o"

    if obj_path.exists():
        print(f"  {name}: already exists, skipping")
        return _result(name, arch, obj_path, "lightning_prefill", spec.__dict__)

    print(f"  Compiling {name}...")
    try:
        from cula.ops.lightning_attn import _compile_single_variant

        compiled_fn = _compile_single_variant(
            spec.has_initial_state,
            spec.output_final_state,
            spec.H,
            spec.D,
            spec.scale,
            spec.chunk_size,
        )

        compiled_fn.export_to_c(
            object_file_path=str(obj_path),
            function_name=name,
        )
        print(f"    Exported .o: {obj_path} ({obj_path.stat().st_size // 1024}KB)")
        _verify_symbol(obj_path, name)
        return _result(name, arch, obj_path, "lightning_prefill", spec.__dict__)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compile_chunk_delta_h(spec: ChunkDeltaHSpec, arch: int, output_dir: Path):
    parts = [
        "cula_delta_h",
        f"h{spec.H}",
        f"k{spec.K}",
        f"v{spec.V}",
        f"cs{spec.chunk_size}",
    ]
    if spec.is_varlen:
        parts.append("varlen")
    if spec.persistent:
        parts.append("persistent")
    parts.append(f"sm{arch}")
    name = "_".join(parts)
    obj_path = output_dir / f"{name}.o"

    if obj_path.exists():
        print(f"  {name}: already exists, skipping")
        return _result(name, arch, obj_path, "chunk_delta_h", spec.__dict__)

    print(f"  Compiling {name}...")
    try:
        from cula.ops.chunk_delta_h import _compile_delta_h_variant

        compiled_fn = _compile_delta_h_variant(
            spec.is_varlen,
            spec.persistent,
            spec.H,
            spec.K,
            spec.V,
            spec.chunk_size,
            use_fast_math=True,
        )

        compiled_fn.export_to_c(
            object_file_path=str(obj_path),
            function_name=name,
        )
        print(f"    Exported .o: {obj_path} ({obj_path.stat().st_size // 1024}KB)")
        _verify_symbol(obj_path, name)
        return _result(name, arch, obj_path, "chunk_delta_h", spec.__dict__)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compile_kda_decode(spec: KdaDecodeSpec, arch: int, output_dir: Path):
    """AOT-compile one kda_decode variant to a .o file.

    The launcher (`run_small_batch`, `run_large_batch`, etc. in
    `cula.kda.kda_decode`) deletes `B`, `T`, `K`, `use_initial_state`,
    `cu_seqlens` inside the jit body, so the compiled kernel can serve any
    batch size / pool size at runtime. We only need to specialize on
    (variant, H, HV, V, use_qk_l2norm) plus the scalar constants.
    """
    import importlib

    parts = [
        "cula_kda_decode",
        spec.variant,
        f"h{spec.H}",
        f"hv{spec.HV}",
        f"v{spec.V}",
    ]
    if spec.use_qk_l2norm:
        parts.append("l2norm")
    parts.append(f"sm{arch}")
    name = "_".join(parts)
    obj_path = output_dir / f"{name}.o"

    if obj_path.exists():
        print(f"  {name}: already exists, skipping")
        return _result(name, arch, obj_path, "kda_decode", spec.__dict__)

    print(f"  Compiling {name}...")
    try:
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda_drv
        import torch
        from cutlass.cute.runtime import from_dlpack

        kda_decode_mod = importlib.import_module("cula.kda.kda_decode")
        TILE_K = kda_decode_mod.TILE_K
        assert spec.K == TILE_K, f"kda_decode kernel requires K={TILE_K}, got K={spec.K}"

        run_small, run_small_varlen, run_large, run_large_varlen = \
            kda_decode_mod._create_jit_functions()
        variant_map = {
            "small_dense":  (run_small,         False),
            "small_varlen": (run_small_varlen,  True),
            "large_dense":  (run_large,         False),
            "large_varlen": (run_large_varlen,  True),
        }
        kernel_func, is_varlen = variant_map[spec.variant]

        # Placeholder tensors — shapes match the launcher's expected layouts.
        # The launcher reads the runtime batch size from `h0_indices` and the
        # pool size from `h0_source`, so fixed placeholder dims work.
        N_placeholder = 1
        pool_placeholder = 1
        if is_varlen:
            q = torch.zeros(1, N_placeholder, spec.H,  spec.K, dtype=torch.bfloat16, device="cuda")
            k = torch.zeros(1, N_placeholder, spec.H,  spec.K, dtype=torch.bfloat16, device="cuda")
            v = torch.zeros(1, N_placeholder, spec.HV, spec.V, dtype=torch.bfloat16, device="cuda")
            a = torch.zeros(N_placeholder, spec.HV, spec.K, dtype=torch.bfloat16, device="cuda")
            b = torch.zeros(N_placeholder, spec.HV,          dtype=torch.bfloat16, device="cuda")
            o = torch.zeros(1, N_placeholder, spec.HV, spec.V, dtype=torch.bfloat16, device="cuda")
        else:
            q = torch.zeros(N_placeholder, 1, spec.H,  spec.K, dtype=torch.bfloat16, device="cuda")
            k = torch.zeros(N_placeholder, 1, spec.H,  spec.K, dtype=torch.bfloat16, device="cuda")
            v = torch.zeros(N_placeholder, 1, spec.HV, spec.V, dtype=torch.bfloat16, device="cuda")
            a = torch.zeros(N_placeholder, 1, spec.HV, spec.K, dtype=torch.bfloat16, device="cuda")
            b = torch.zeros(N_placeholder, 1, spec.HV,          dtype=torch.bfloat16, device="cuda")
            o = torch.zeros(N_placeholder, 1, spec.HV, spec.V, dtype=torch.bfloat16, device="cuda")
        A_log    = torch.zeros(spec.HV,             dtype=torch.float32, device="cuda")
        dt_bias  = torch.zeros(spec.HV, spec.K,     dtype=torch.float32, device="cuda")
        h0_src   = torch.zeros(pool_placeholder, spec.HV, spec.V, spec.K,
                               dtype=torch.float32, device="cuda")
        h0_idx   = torch.zeros(N_placeholder,       dtype=torch.int32,   device="cuda")
        cu_sql   = torch.zeros(N_placeholder + 1,   dtype=torch.int32,   device="cuda")

        # Mark the dims that vary at runtime as dynamic. Batch size `N` and
        # pool_size must be dynamic or the TVM FFI wrapper bakes the
        # placeholder values into the exported kernel as shape asserts.
        # `mark_compact_shape_dynamic(mode=i, stride_order=...)` marks dim
        # `i` dynamic while fixing the innermost stride-1 axis.
        cu_sql_t = (
            from_dlpack(cu_sql, assumed_align=16)
            .mark_compact_shape_dynamic(mode=0, stride_order=cu_sql.dim_order())
        )
        if is_varlen:
            # q/k/v/o are [1, N, H, K] or [1, N, HV, V] — dim 1 is the batch.
            q_t = from_dlpack(q, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=1, stride_order=q.dim_order())
            k_t = from_dlpack(k, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=1, stride_order=k.dim_order())
            v_t = from_dlpack(v, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=1, stride_order=v.dim_order())
            o_t = from_dlpack(o, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=1, stride_order=o.dim_order())
            # a is [N, HV, K], b is [N, HV] — dim 0 is the batch.
            a_t = from_dlpack(a, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order())
            b_t = from_dlpack(b, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=b.dim_order())
        else:
            # dense layout: q/k/v/o = [N, 1, ...], a = [N, 1, HV, K], b = [N, 1, HV]
            q_t = from_dlpack(q, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=q.dim_order())
            k_t = from_dlpack(k, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=k.dim_order())
            v_t = from_dlpack(v, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=v.dim_order())
            o_t = from_dlpack(o, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=o.dim_order())
            a_t = from_dlpack(a, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order())
            b_t = from_dlpack(b, assumed_align=16) \
                .mark_compact_shape_dynamic(mode=0, stride_order=b.dim_order())
        A_log_t = from_dlpack(A_log, assumed_align=16)
        dt_bias_t = from_dlpack(dt_bias, assumed_align=16)
        # h0_source: pool_size (dim 0) is dynamic, HV/V/K are fixed by spec.
        h0_src_t = (
            from_dlpack(h0_src, assumed_align=16)
            .mark_compact_shape_dynamic(mode=0, stride_order=h0_src.dim_order())
        )
        h0_idx_t = (
            from_dlpack(h0_idx, assumed_align=16)
            .mark_compact_shape_dynamic(mode=0, stride_order=h0_idx.dim_order())
        )

        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

        compiled = cute.compile(
            kernel_func,
            cu_sql_t, q_t, k_t, v_t, a_t, b_t, A_log_t, dt_bias_t, h0_src_t, h0_idx_t, o_t,
            softplus_beta=spec.softplus_beta,
            softplus_threshold=spec.softplus_threshold,
            scale=spec.K ** -0.5,
            B=1 if is_varlen else N_placeholder,
            T=N_placeholder if is_varlen else 1,
            H=spec.H, HV=spec.HV, K=spec.K, V=spec.V,
            use_initial_state=True,
            use_qk_l2norm=spec.use_qk_l2norm,
            stream=stream,
        )

        # When CUTE_DSL_ENABLE_TVM_FFI=1 is set (in build.rs), `cute.compile`
        # returns a `TVMFFIJitCompiledFunctionBase` whose `export_to_c` keeps
        # the older kwargs: `object_file_path` + `function_name`. This matches
        # the other compile_* functions in this file.
        compiled.export_to_c(
            object_file_path=str(obj_path),
            function_name=name,
        )
        print(f"    Exported .o: {obj_path} ({obj_path.stat().st_size // 1024}KB)")
        _verify_symbol(obj_path, name)
        return _result(name, arch, obj_path, "kda_decode", spec.__dict__)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compile_fwd_o(spec: FwdOSpec, arch: int, output_dir: Path):
    parts = [
        "cula_fwd_o",
        f"h{spec.H}",
        f"k{spec.K}",
        f"v{spec.V}",
        f"cs{spec.chunk_size}",
    ]
    if spec.is_varlen:
        parts.append("varlen")
    if spec.persistent:
        parts.append("persistent")
    parts.append(f"sm{arch}")
    name = "_".join(parts)
    obj_path = output_dir / f"{name}.o"

    if obj_path.exists():
        print(f"  {name}: already exists, skipping")
        return _result(name, arch, obj_path, "fwd_o", spec.__dict__)

    print(f"  Compiling {name}...")
    try:
        from cula.ops.fwd_o import _compile_fwd_o_variant

        compiled_fn = _compile_fwd_o_variant(
            spec.is_varlen,
            spec.persistent,
            spec.H,
            spec.K,
            spec.V,
            spec.scale,
            spec.chunk_size,
            use_fast_math=True,
        )

        compiled_fn.export_to_c(
            object_file_path=str(obj_path),
            function_name=name,
        )
        print(f"    Exported .o: {obj_path} ({obj_path.stat().st_size // 1024}KB)")
        _verify_symbol(obj_path, name)
        return _result(name, arch, obj_path, "fwd_o", spec.__dict__)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Variant matrix
# ---------------------------------------------------------------------------

def build_variant_matrix(arch: int, prototype: bool = False):
    """Build the variant matrix for AOT compilation.

    Conservative: only compile configs needed for known model architectures.
    Expand as new models are added.
    """
    # Common head dims for linear attention models
    head_dims = [128]
    num_heads_list = [32, 64]
    chunk_sizes = [64]

    if prototype:
        return {
            "lightning_prefill": [
                LightningPrefillSpec(H=32, D=128, chunk_size=64,
                                     has_initial_state=False, output_final_state=True),
            ],
            "chunk_delta_h": [
                ChunkDeltaHSpec(H=32, K=128, V=128, chunk_size=64,
                                is_varlen=False, persistent=False),
            ],
            "fwd_o": [
                FwdOSpec(H=32, K=128, V=128, chunk_size=64, scale=1.0,
                         is_varlen=False, persistent=False),
            ],
        }

    la_prefill = []
    delta_h = []
    fwd_o_specs = []
    kda_decode_specs = []

    for H in num_heads_list:
        for D in head_dims:
            for cs in chunk_sizes:
                # Lightning Attention prefill: all state combos
                for init_state in [False, True]:
                    for final_state in [False, True]:
                        la_prefill.append(LightningPrefillSpec(
                            H=H, D=D, chunk_size=cs,
                            has_initial_state=init_state,
                            output_final_state=final_state,
                        ))
                # Chunk delta-H: varlen and non-varlen
                for varlen in [False, True]:
                    for persistent in [False, True]:
                        delta_h.append(ChunkDeltaHSpec(
                            H=H, K=D, V=D, chunk_size=cs,
                            is_varlen=varlen, persistent=persistent,
                        ))
                        fwd_o_specs.append(FwdOSpec(
                            H=H, K=D, V=D, chunk_size=cs, scale=1.0,
                            is_varlen=varlen, persistent=persistent,
                        ))
        # KDA decode: one kernel per (H, HV, K=128, V=128) × 4 variants.
        # Kernel launcher reads runtime batch size / pool size from tensors,
        # so we only specialize on head counts + variant.
        for variant in ("small_dense", "small_varlen", "large_dense", "large_varlen"):
            # MHA: H == HV
            kda_decode_specs.append(KdaDecodeSpec(
                H=H, HV=H, K=128, V=128, variant=variant, use_qk_l2norm=True,
            ))

    # GQA variants we care about. Add as new models land.
    # Qwen3.5-35B-A3B: linear_num_key_heads=16, linear_num_value_heads=32
    for variant in ("small_dense", "small_varlen", "large_dense", "large_varlen"):
        kda_decode_specs.append(KdaDecodeSpec(
            H=16, HV=32, K=128, V=128, variant=variant, use_qk_l2norm=True,
        ))

    # KDA decode works on Hopper+ (the CuTe DSL kernel targets SM90+).
    result = {"kda_decode": kda_decode_specs}

    # chunk_delta_h and fwd_o use tcgen05 ops → SM100+ only.
    # Lightning Attention prefill also requires SM100+.
    if arch >= 100:
        result["chunk_delta_h"] = delta_h
        result["fwd_o"] = fwd_o_specs
        result["lightning_prefill"] = la_prefill
    else:
        print(f"  Note: skipping chunk_delta_h/fwd_o/lightning_prefill "
              f"(require SM100+, target is SM{arch})")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(name, arch, obj_path, kernel_type, params):
    return {
        "name": name,
        "arch": arch,
        "obj_path": str(obj_path),
        "symbol": f"__tvm_ffi_{name}",
        "kernel_type": kernel_type,
        **{k: v for k, v in params.items()},
    }


def _verify_symbol(obj_path, name):
    try:
        result = subprocess.run(
            ["nm", "-g", str(obj_path)],
            capture_output=True, text=True, timeout=10,
        )
        expected = f"__tvm_ffi_{name}"
        if expected in result.stdout:
            print(f"    Symbol OK: {expected}")
        elif "__tvm_ffi_func" in result.stdout:
            print(f"    WARNING: found __tvm_ffi_func, renaming to {expected}")
            subprocess.run([
                "objcopy",
                f"--redefine-sym=__tvm_ffi_func={expected}",
                str(obj_path),
            ], check=True, timeout=10)
            print(f"    Renamed OK")
        else:
            print(f"    WARNING: neither {expected} nor __tvm_ffi_func found")
    except Exception as e:
        print(f"    Symbol check skipped: {e}")


def _compile_worker(task):
    kernel_type, spec, arch, output_dir = task
    if kernel_type == "lightning_prefill":
        return compile_lightning_prefill(spec, arch, output_dir)
    elif kernel_type == "chunk_delta_h":
        return compile_chunk_delta_h(spec, arch, output_dir)
    elif kernel_type == "fwd_o":
        return compile_fwd_o(spec, arch, output_dir)
    elif kernel_type == "kda_decode":
        return compile_kda_decode(spec, arch, output_dir)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AOT compile cuLA CuTe DSL kernels")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "kernels")
    parser.add_argument("--arch", type=int, default=None)
    parser.add_argument("--prototype", action="store_true",
                        help="Minimal variant set for testing")
    parser.add_argument("-j", "--workers", type=int, default=1)
    args = parser.parse_args()

    arch = args.arch or get_arch_from_env()
    os.environ["CUTE_DSL_ARCH"] = f"sm_{arch}a"

    print(f"cuLA AOT: target SM{arch}")
    print(f"Output: {args.output_dir}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variant_matrix(arch, prototype=args.prototype)

    # Flatten into task list
    tasks = []
    for kernel_type, specs in variants.items():
        for spec in specs:
            tasks.append((kernel_type, spec, arch, args.output_dir))

    print(f"Compiling {len(tasks)} kernel variants (workers={args.workers})...\n")

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_compile_worker, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        results = []
        for task in tasks:
            result = _compile_worker(task)
            if result:
                results.append(result)

    # Write manifest
    import hashlib
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    manifest = {
        "archs": [arch],
        "variants": results,
        "script_hash": script_hash,
    }
    try:
        import cutlass
        manifest["cutlass_version"] = cutlass.__version__
    except (ImportError, AttributeError):
        pass

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Results: {len(results)}/{len(tasks)} compiled for SM{arch} ===")
    by_type = {}
    for r in results:
        t = r["kernel_type"]
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    if len(results) < len(tasks):
        failed = len(tasks) - len(results)
        print(f"  Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
