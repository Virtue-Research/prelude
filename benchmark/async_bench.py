"""Minimal concurrent bench: hammer /v1/chat/completions and report ms latencies.

Output mirrors the genai-bench table I gave earlier so the user can compare 1:1.
"""

import argparse, asyncio, json, os, random, statistics, time
import aiohttp


async def one_request(session, url, model, prompt, max_tokens, decode_priority_max_remaining_tokens, stats):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    if decode_priority_max_remaining_tokens is not None:
        payload["decode_priority_max_remaining_tokens"] = decode_priority_max_remaining_tokens
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                stats["errors"] += 1
                stats["error_codes"].setdefault(resp.status, 0)
                stats["error_codes"][resp.status] += 1
                return
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                if obj.get("usage"):
                    completion_tokens = obj["usage"].get("completion_tokens", 0)
                    prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                # Count tokens via choices.delta.content if present
        t1 = time.perf_counter()
        e2e = t1 - t0
        if ttft is None:
            ttft = e2e
        # TPOT = (e2e - ttft) / (output_tokens - 1) if output_tokens > 1 else 0
        if completion_tokens > 1:
            tpot = (e2e - ttft) / (completion_tokens - 1)
        else:
            tpot = 0.0
        stats["ttft"].append(ttft)
        stats["tpot"].append(tpot)
        stats["e2e"].append(e2e)
        stats["input_tokens"].append(prompt_tokens)
        stats["output_tokens"].append(completion_tokens)
        stats["completed"] += 1
    except Exception as e:
        stats["errors"] += 1
        stats["error_codes"].setdefault(repr(type(e).__name__), 0)
        stats["error_codes"][repr(type(e).__name__)] += 1


async def worker(name, queue, session, url, model, max_tokens, decode_priority_max_remaining_tokens, stats):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        await one_request(session, url, model, item, max_tokens, decode_priority_max_remaining_tokens, stats)
        queue.task_done()


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="JSON list of prompt strings")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--max-tokens", type=int, default=3)
    ap.add_argument("--decode-priority-max-remaining-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=0, help="warmup requests before measurement")
    args = ap.parse_args()

    random.seed(args.seed)
    prompts = json.load(open(args.dataset))
    print(f"[bench] loaded {len(prompts)} prompts from {args.dataset}")
    print(f"[bench] url={args.url} model={args.model} c={args.concurrency} n={args.requests} max_tokens={args.max_tokens} decode_priority_max_remaining_tokens={args.decode_priority_max_remaining_tokens}")

    # Use TCPConnector with enough sockets for concurrency
    conn = aiohttp.TCPConnector(limit=args.concurrency * 2, limit_per_host=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        if args.warmup > 0:
            print(f"[bench] warmup {args.warmup} requests …")
            wstats = {"ttft": [], "tpot": [], "e2e": [], "input_tokens": [], "output_tokens": [],
                      "completed": 0, "errors": 0, "error_codes": {}}
            wq = asyncio.Queue()
            for _ in range(args.warmup):
                wq.put_nowait(random.choice(prompts))
            for _ in range(min(args.concurrency, args.warmup)):
                wq.put_nowait(None)
            workers = [asyncio.create_task(worker(f"w{i}", wq, session, args.url, args.model, args.max_tokens, args.decode_priority_max_remaining_tokens, wstats))
                       for i in range(min(args.concurrency, args.warmup))]
            await wq.join()
            for w in workers:
                w.cancel()

        stats = {"ttft": [], "tpot": [], "e2e": [], "input_tokens": [], "output_tokens": [],
                 "completed": 0, "errors": 0, "error_codes": {}}
        q = asyncio.Queue()
        for _ in range(args.requests):
            q.put_nowait(random.choice(prompts))
        for _ in range(args.concurrency):
            q.put_nowait(None)

        t_start = time.perf_counter()
        workers = [asyncio.create_task(worker(f"w{i}", q, session, args.url, args.model, args.max_tokens, args.decode_priority_max_remaining_tokens, stats))
                   for i in range(args.concurrency)]
        # progress ticker
        async def tick():
            last = 0
            while True:
                await asyncio.sleep(5)
                done = stats["completed"]
                rps = (done - last) / 5
                last = done
                print(f"  [{time.perf_counter() - t_start:5.1f}s] completed={done} errors={stats['errors']} rps_5s={rps:.1f}")
                if done + stats["errors"] >= args.requests:
                    return
        ticker = asyncio.create_task(tick())
        await q.join()
        ticker.cancel()
        t_end = time.perf_counter()
        for w in workers:
            w.cancel()

    duration = t_end - t_start
    total_in = sum(stats["input_tokens"])
    total_out = sum(stats["output_tokens"])
    completed = stats["completed"]

    def line(label, vals):
        if not vals:
            print(f"  {label}: (no data)")
            return
        print(f"  {label}: mean={statistics.mean(vals)*1000:7.1f} p50={pct(vals,50)*1000:7.1f} "
              f"p90={pct(vals,90)*1000:7.1f} p99={pct(vals,99)*1000:7.1f} max={max(vals)*1000:7.1f}")

    print()
    print(f"=== Results ===")
    print(f"  duration: {duration*1000:.0f} ms")
    print(f"  completed: {completed}  errors: {stats['errors']}  error_codes: {stats['error_codes']}")
    print(f"  RPS: {completed / duration:.1f}")
    print(f"  input throughput: {total_in / duration:.1f} tok/s")
    print(f"  output throughput: {total_out / duration:.1f} tok/s")
    print(f"  input tokens: mean={statistics.mean(stats['input_tokens']):.1f} "
          f"p50={pct(stats['input_tokens'],50):.0f} p99={pct(stats['input_tokens'],99):.0f} max={max(stats['input_tokens'])}")
    print(f"  output tokens: mean={statistics.mean(stats['output_tokens']):.1f}")
    print()
    print("  Latencies (ms):")
    line("ttft       ", stats["ttft"])
    line("tpot       ", stats["tpot"])
    line("e2e_latency", stats["e2e"])

    # Also dump JSON
    out = {
        "duration_ms": duration * 1000,
        "completed": completed,
        "errors": stats["errors"],
        "rps": completed / duration if duration > 0 else 0,
        "input_throughput_tok_s": total_in / duration if duration > 0 else 0,
        "output_throughput_tok_s": total_out / duration if duration > 0 else 0,
        "ttft_ms": {"mean": statistics.mean(stats["ttft"])*1000 if stats["ttft"] else None,
                    "p50": pct(stats["ttft"], 50)*1000 if stats["ttft"] else None,
                    "p90": pct(stats["ttft"], 90)*1000 if stats["ttft"] else None,
                    "p99": pct(stats["ttft"], 99)*1000 if stats["ttft"] else None,
                    "max": max(stats["ttft"])*1000 if stats["ttft"] else None},
        "tpot_ms": {"mean": statistics.mean(stats["tpot"])*1000 if stats["tpot"] else None,
                    "p50": pct(stats["tpot"], 50)*1000 if stats["tpot"] else None,
                    "p90": pct(stats["tpot"], 90)*1000 if stats["tpot"] else None,
                    "p99": pct(stats["tpot"], 99)*1000 if stats["tpot"] else None,
                    "max": max(stats["tpot"])*1000 if stats["tpot"] else None},
        "e2e_ms":  {"mean": statistics.mean(stats["e2e"])*1000 if stats["e2e"] else None,
                    "p50": pct(stats["e2e"], 50)*1000 if stats["e2e"] else None,
                    "p90": pct(stats["e2e"], 90)*1000 if stats["e2e"] else None,
                    "p99": pct(stats["e2e"], 99)*1000 if stats["e2e"] else None,
                    "max": max(stats["e2e"])*1000 if stats["e2e"] else None},
        "input_tokens_mean": statistics.mean(stats["input_tokens"]) if stats["input_tokens"] else None,
        "output_tokens_mean": statistics.mean(stats["output_tokens"]) if stats["output_tokens"] else None,
    }
    out_path = os.environ.get("BENCH_OUT", "async_bench_out.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {out_path}")


asyncio.run(main())
