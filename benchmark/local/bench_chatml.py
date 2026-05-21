#!/usr/bin/env python3
"""
Chat benchmark that reads a ChatML JSONL (`{"messages": [...]}` per line) and
POSTs each row to /v1/chat/completions. Sweeps concurrency levels and reports
throughput/latency.

Usage:
  bench_chatml.py --url http://localhost:8000 --model NAME \
      --dataset /path/to/labeled_272k_chatml.jsonl \
      --samples 1000 --max-tokens 3 --concurrency 1,8,32,64,128 --warmup 5
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import aiohttp


def load_chatml(path: Path, n: int, seed: int) -> list[list[dict]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                continue
            # Drop trailing assistant turns so the model has something to predict.
            while msgs and msgs[-1].get("role") == "assistant":
                msgs.pop()
            if not msgs:
                continue
            rows.append(msgs)
            if len(rows) >= max(n * 4, n + 64):
                break
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def pct(s: list[float], p: float) -> float:
    if not s:
        return 0.0
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


async def one_request(session, url, model, messages, max_tokens, timeout):
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            raw = await resp.read()
            if resp.status >= 400:
                return None, raw[:300].decode("utf-8", "replace")
            payload = json.loads(raw)
    except Exception as e:
        return None, str(e)[:300]
    lat = (time.perf_counter() - t0) * 1000
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return (
        lat,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    ), None


async def run_level(url, model, rows, conc, max_tokens, timeout):
    sem = asyncio.Semaphore(conc)
    lats: list[float] = []
    in_tok = 0
    out_tok = 0
    errors = 0
    error_samples: list[str] = []

    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def worker(msgs):
            async with sem:
                return await one_request(session, url, model, msgs, max_tokens, timeout)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker(m)) for m in rows]
        done = 0
        for coro in asyncio.as_completed(tasks):
            ok, err = await coro
            if ok is None:
                errors += 1
                if len(error_samples) < 3:
                    error_samples.append(err or "")
            else:
                lat, pt, ct = ok
                lats.append(lat)
                in_tok += pt
                out_tok += ct
            done += 1
            if done % 100 == 0:
                print(f"\r  progress {done}/{len(rows)}", end="", flush=True)
        total = time.perf_counter() - t0
    print(f"\r  progress {len(rows)}/{len(rows)} done in {total:.2f}s")
    if error_samples:
        for s in error_samples:
            print(f"  err: {s}")
    lats.sort()
    return {
        "conc": conc,
        "reqs": len(rows),
        "ok": len(lats),
        "fail": errors,
        "time_s": total,
        "rps": len(lats) / total if total > 0 else 0.0,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "in_tok_per_s": in_tok / total if total > 0 else 0.0,
        "out_tok_per_s": out_tok / total if total > 0 else 0.0,
        "avg_ms": statistics.mean(lats) if lats else 0.0,
        "p50_ms": pct(lats, 0.50),
        "p95_ms": pct(lats, 0.95),
        "p99_ms": pct(lats, 0.99),
        "min_ms": lats[0] if lats else 0.0,
        "max_ms": lats[-1] if lats else 0.0,
    }


def fmt_table(rows):
    head = f"{'conc':>5} {'reqs':>5} {'fail':>5} {'rps':>9} {'in-tok/s':>10} {'out-tok/s':>10} {'avg(ms)':>9} {'p50':>8} {'p95':>8} {'p99':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['conc']:>5} {r['reqs']:>5} {r['fail']:>5} "
            f"{r['rps']:>9.2f} {r['in_tok_per_s']:>10.0f} {r['out_tok_per_s']:>10.0f} "
            f"{r['avg_ms']:>9.2f} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} {r['p99_ms']:>8.2f}"
        )


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=3)
    ap.add_argument("--concurrency", default="1,8,32,64,128")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(f"loading {args.samples} rows from {args.dataset} ...")
    rows = load_chatml(Path(args.dataset), args.samples, args.seed)
    if len(rows) < args.samples:
        print(f"WARN: only {len(rows)} usable rows in dataset")
    if not rows:
        print("ERROR: no rows loaded")
        sys.exit(1)
    print(f"loaded {len(rows)} rows; first row has {len(rows[0])} messages, "
          f"system len={len(rows[0][0].get('content',''))} chars")

    # warmup (sequential)
    print(f"\nwarming up {args.warmup} requests...")
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(args.warmup):
            ok, err = await one_request(
                session, args.url, args.model, rows[i % len(rows)],
                args.max_tokens, args.timeout,
            )
            if ok is None:
                print(f"  warmup {i+1}: ERROR {err}")
                sys.exit(1)
            print(f"  warmup {i+1}: {ok[0]:.1f}ms in={ok[1]} out={ok[2]}")

    results = []
    for c in concs:
        print(f"\n--- concurrency={c} ---")
        # use the same 1000 rows each level so prefix cache observation is consistent
        r = await run_level(
            args.url, args.model, rows, c, args.max_tokens, args.timeout,
        )
        results.append(r)
        print(
            f"  {r['ok']}/{r['reqs']} ok, {r['fail']} fail | rps={r['rps']:.2f} "
            f"in-tok/s={r['in_tok_per_s']:.0f} out-tok/s={r['out_tok_per_s']:.0f} "
            f"avg={r['avg_ms']:.1f}ms p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms"
        )
        if c != concs[-1]:
            await asyncio.sleep(2)

    print("\n========= SUMMARY =========")
    fmt_table(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nsaved results to {args.output}")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
