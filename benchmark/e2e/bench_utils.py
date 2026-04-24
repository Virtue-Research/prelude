#!/usr/bin/env python3
"""Benchmark utilities — metrics extraction and summary table for bench.sh."""

import argparse
import csv
import json
import statistics
import sys


def extract_metrics(json_file: str, engine: str, device: str, startup_s: str, csv_file: str):
    """Extract benchmark metrics from genai-bench JSON and append to CSV."""
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

    # Fallback: recompute from individual requests when aggregated stats are null
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
        cf.write(f"{engine},{device},{startup_s},{ttft},{tpot},{e2e},{in_tps},{out_tps},{rpm}\n")


def _fallback_from_individual(data: dict) -> dict:
    """Compute metrics from individual_request_metrics when aggregated stats are null."""
    reqs = data.get("individual_request_metrics", [])
    valid = [r for r in reqs if r.get("error_code") is None and r.get("e2e_latency") is not None]
    if not valid:
        return {}

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


def print_summary(csv_file: str, model: str, traffic: str, input_tokens: str,
                   output_tokens: str, concurrency: str, max_requests: str,
                   cpu_name: str = "", gpu_name: str = "", gpu_count: str = "0"):
    """Print formatted summary table from CSV."""
    COLS = [
        ("Engine", 14), ("Device", 7), ("Start(s)", 8), ("TTFT(s)", 8),
        ("TPOT(s)", 8), ("E2E(s)", 8), ("In t/s", 8), ("Out t/s", 8), ("RPM", 6),
    ]

    sep = lambda l, m, r: l + m.join("─" * (w + 2) for _, w in COLS) + r
    row = lambda vals: "│" + "│".join(f" {v:<{w}} " for (_, w), v in zip(COLS, vals)) + "│"

    print()
    print(sep("┌", "┬", "┐"))
    print(row([name for name, _ in COLS]))
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
    print(f"\n  CSV:          {csv_file}\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("extract-metrics")
    p.add_argument("--json-file", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--startup-s", default="N/A")
    p.add_argument("--csv-file", required=True)

    p = sub.add_parser("print-summary")
    p.add_argument("--csv-file", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--traffic", required=True)
    p.add_argument("--input-tokens", required=True)
    p.add_argument("--output-tokens", required=True)
    p.add_argument("--concurrency", required=True)
    p.add_argument("--max-requests", required=True)
    p.add_argument("--cpu-name", default="")
    p.add_argument("--gpu-name", default="")
    p.add_argument("--gpu-count", default="0")

    args = parser.parse_args()
    if args.command == "extract-metrics":
        extract_metrics(args.json_file, args.engine, args.device, args.startup_s, args.csv_file)
    elif args.command == "print-summary":
        print_summary(args.csv_file, args.model, args.traffic, args.input_tokens,
                       args.output_tokens, args.concurrency, args.max_requests,
                       args.cpu_name, args.gpu_name, args.gpu_count)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
