#!/usr/bin/env python3
"""Utility helpers for bench.sh — vllm.rs launcher, metrics extraction, summary table."""

import argparse
import csv
import json
import os
import sys


def check_engine(
    engine: str,
    prelude_bin: str,
    vllm_rs_bin: str,
    vllm_cpu_venv: str,
    llama_cpp_bin: str = "",
    gguf_model: str = "",
):
    """Check if an engine's dependencies are available. Exit 0 if ready, 1 if not."""
    checks = {
        "prelude": lambda: (
            os.path.isfile(prelude_bin),
            f"binary not built ({prelude_bin})",
        ),
        "prelude-gguf": lambda: (
            os.path.isfile(prelude_bin) and os.path.isfile(gguf_model),
            f"binary not built ({prelude_bin})"
            if not os.path.isfile(prelude_bin)
            else f"GGUF model not found ({gguf_model})",
        ),
        "vllm.rs": lambda: (
            os.path.isfile(vllm_rs_bin),
            f"binary not built ({vllm_rs_bin})",
        ),
        "vllm": lambda: (_try_import("vllm"), "vllm not installed"),
        "vllm-cpu": lambda: (
            os.path.isfile(os.path.join(vllm_cpu_venv, "bin", "python3"))
            and os.system(f'"{vllm_cpu_venv}/bin/python3" -c "import vllm" 2>/dev/null')
            == 0,
            f"venv or vllm not found ({vllm_cpu_venv})",
        ),
        "sglang": lambda: (_try_import("sglang"), "sglang not installed"),
        "sglang-cpu": lambda: (
            os.system("docker image inspect sglang-cpu:latest >/dev/null 2>&1") == 0,
            "Docker image sglang-cpu:latest not found",
        ),
        "llama.cpp": lambda: (
            os.path.isfile(llama_cpp_bin) and os.path.isfile(gguf_model),
            f"llama-server not found ({llama_cpp_bin})"
            if not os.path.isfile(llama_cpp_bin)
            else f"GGUF model not found ({gguf_model})",
        ),
    }
    if engine not in checks:
        return
    ok, reason = checks[engine]()
    if not ok:
        print(reason)
        sys.exit(1)


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _fallback_from_individual(data: dict) -> dict:
    """Compute metrics from individual_request_metrics when aggregated stats are null.

    genai-bench filters out TPOT/e2e when output_latency < 1ms (all tokens arrive
    at once), leaving aggregated stats as null. We recompute from per-request data.
    """
    reqs = data.get("individual_request_metrics", [])
    valid = [r for r in reqs if r.get("error_code") is None and r.get("e2e_latency") is not None]
    if not valid:
        return {}

    import statistics

    e2e_vals = [r["e2e_latency"] for r in valid]
    tpot_vals = [
        r["output_latency"] / (r["num_output_tokens"] - 1)
        for r in valid
        if r.get("output_latency") is not None and r.get("num_output_tokens", 0) > 1
    ]
    in_tokens = sum(r.get("num_input_tokens", 0) for r in valid)
    out_tokens = sum(r.get("num_output_tokens", 0) for r in valid)
    total_time = sum(e2e_vals)

    return {
        "tpot_mean": statistics.mean(tpot_vals) if tpot_vals else None,
        "e2e_mean": statistics.mean(e2e_vals),
        "in_tps": in_tokens / total_time if total_time > 0 else 0,
        "out_tps": out_tokens / total_time if total_time > 0 else 0,
        "rps": len(valid) / total_time if total_time > 0 else 0,
    }


def extract_metrics(
    json_file: str, engine: str, device: str, startup_s: str, csv_file: str
):
    """Extract benchmark metrics from genai-bench JSON results and append to CSV."""
    with open(json_file) as f:
        data = json.load(f)

    agg = data.get("aggregated_metrics", {})
    stats = agg.get("stats", {})

    def fmt(val, precision=4):
        return f"{val:.{precision}f}" if isinstance(val, (int, float)) else "N/A"

    ttft = fmt(stats.get("ttft", {}).get("mean"))
    tpot = fmt(stats.get("tpot", {}).get("mean"))
    e2e = fmt(stats.get("e2e_latency", {}).get("mean"))
    in_tps = fmt(agg.get("mean_input_throughput_tokens_per_s"), 1)
    out_tps = fmt(agg.get("mean_output_throughput_tokens_per_s"), 1)
    rps = agg.get("requests_per_second", 0)
    rpm = fmt(rps * 60, 1) if isinstance(rps, (int, float)) else "N/A"

    # Fallback to individual request data when aggregated stats are null
    # (genai-bench filters TPOT/e2e when output_latency < 1ms)
    if tpot == "N/A" or e2e == "N/A" or in_tps == "0.0":
        fb = _fallback_from_individual(data)
        if fb:
            if tpot == "N/A":
                tpot = fmt(fb.get("tpot_mean"))
            if e2e == "N/A":
                e2e = fmt(fb.get("e2e_mean"))
            if in_tps == "0.0":
                in_tps = fmt(fb.get("in_tps"), 1)
            if out_tps == "0.0":
                out_tps = fmt(fb.get("out_tps"), 1)
            if rpm == "0.0":
                rpm = fmt(fb.get("rps", 0) * 60, 1)

    with open(csv_file, "a") as cf:
        cf.write(
            f"{engine},{device},{startup_s},{ttft},{tpot},{e2e},{in_tps},{out_tps},{rpm}\n"
        )


def print_summary(
    csv_file: str,
    model: str,
    traffic: str,
    input_tokens: str,
    output_tokens: str,
    concurrency: str,
    max_requests: str,
    has_gpu: str,
    cpu_name: str = "",
    gpu_name: str = "",
    gpu_count: str = "0",
):
    """Print formatted summary table from CSV results."""
    COLS = [
        ("Engine", 14, "s"),
        ("Device", 7, "s"),
        ("Start(s)", 8, "s"),
        ("TTFT(s)", 8, "s"),
        ("TPOT(s)", 8, "s"),
        ("E2E(s)", 8, "s"),
        ("In t/s", 8, "s"),
        ("Out t/s", 8, "s"),
        ("RPM", 6, "s"),
    ]

    def sep(left, mid, right):
        return left + mid.join("─" * (w + 2) for _, w, _ in COLS) + right

    def row(vals):
        cells = []
        for (_, w, align), v in zip(COLS, vals):
            if align == "s":
                cells.append(f" {v:<{w}} ")
            else:
                cells.append(f" {v:>{w}} ")
        return "│" + "│".join(cells) + "│"

    print()
    print(sep("┌", "┬", "┐"))
    print(row([name for name, _, _ in COLS]))
    print(sep("├", "┼", "┤"))

    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for fields in reader:
            print(row(fields))

    print(sep("└", "┴", "┘"))
    print()
    if cpu_name:
        print(f"  CPU:          {cpu_name}")
    if gpu_name and int(gpu_count) > 0:
        print(f"  GPU:          {gpu_count}x {gpu_name}")
    print(f"  Model:        {model}")
    print(f"  Traffic:      {traffic}  (input={input_tokens}, output={output_tokens})")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Requests:     {max_requests}")
    print()
    print(f"  CSV:          {csv_file}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark utilities")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser(
        "check-engine", help="Check if engine dependencies are available"
    )
    p.add_argument("--engine", required=True)
    p.add_argument("--prelude-bin", default="")
    p.add_argument("--vllm-rs-bin", default="")
    p.add_argument("--vllm-cpu-venv", default="")
    p.add_argument("--llama-cpp-bin", default="")
    p.add_argument("--gguf-model", default="")

    p = sub.add_parser("extract-metrics", help="Extract genai-bench metrics to CSV")
    p.add_argument("--json-file", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--startup-s", default="N/A")
    p.add_argument("--csv-file", required=True)

    p = sub.add_parser("print-summary", help="Print formatted results table")
    p.add_argument("--csv-file", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--traffic", required=True)
    p.add_argument("--input-tokens", required=True)
    p.add_argument("--output-tokens", required=True)
    p.add_argument("--concurrency", required=True)
    p.add_argument("--max-requests", required=True)
    p.add_argument("--has-gpu", required=True)
    p.add_argument("--cpu-name", default="")
    p.add_argument("--gpu-name", default="")
    p.add_argument("--gpu-count", default="0")

    args = parser.parse_args()

    if args.command == "check-engine":
        check_engine(
            args.engine,
            args.prelude_bin,
            args.vllm_rs_bin,
            args.vllm_cpu_venv,
            args.llama_cpp_bin,
            args.gguf_model,
        )
    elif args.command == "extract-metrics":
        extract_metrics(
            args.json_file, args.engine, args.device, args.startup_s, args.csv_file
        )
    elif args.command == "print-summary":
        print_summary(
            args.csv_file,
            args.model,
            args.traffic,
            args.input_tokens,
            args.output_tokens,
            args.concurrency,
            args.max_requests,
            args.has_gpu,
            args.cpu_name,
            args.gpu_name,
            args.gpu_count,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
