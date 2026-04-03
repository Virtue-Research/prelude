"""
SGLang kernel micro-benchmarks.

Runs inside sglang Docker image, outputs JSON for comparison.
Usage: python3 bench_kernels.py --json /results/sglang.json
"""

import argparse
import json
import time
import torch

def get_hardware_info():
    info = {"framework": "sglang"}
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
    return elapsed / repeats * 1e6  # microseconds

def run_benchmarks():
    results = []

    # Add benchmarks here as needed. Example:
    #
    # from sgl_kernel import moe_align_block_size
    #
    # def bench_moe():
    #     ...
    #     us = bench_kernel(lambda: moe_align_block_size(...))
    #     results.append({
    #         "category": "gpu/moe/align",
    #         "name": "topk=8 experts=64",
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
