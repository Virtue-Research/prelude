"""
vLLM kernel micro-benchmarks.

Runs inside vllm Docker image, outputs JSON for comparison.
Usage: python3 bench_kernels.py --json /results/vllm.json
"""

import argparse
import json
import time
import torch

def get_hardware_info():
    info = {"framework": "vllm"}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info

def bench_kernel(fn, warmup=50, repeats=200):
    """Benchmark a CUDA kernel, returns time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / repeats * 1e6

def run_benchmarks():
    results = []

    # Add benchmarks here as needed. Example:
    #
    # from vllm._custom_ops import cutlass_scaled_mm
    #
    # def bench_w8a8():
    #     ...
    #     us = bench_kernel(lambda: cutlass_scaled_mm(...))
    #     results.append({
    #         "category": "gpu/gemm/w8a8",
    #         "name": "M=1 K=4096 N=4096",
    #         "ours_us": us,
    #     })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Output JSON path")
    args = parser.parse_args()

    report = {
        "hardware": get_hardware_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": run_benchmarks(),
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {args.json}")
    else:
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
