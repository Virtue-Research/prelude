#!/usr/bin/env python3
"""
Benchmark Suite

Tasks:
  classify   /v1/classify         batch mode (multiple texts per request)
  embed      /v1/embeddings       batch mode
  complete   /v1/completions      prefill-only (max_tokens=1) or generation (max_tokens>1)
  mix        /v1/completions      prefill-only + generation mixed by weight

Data:
  synthetic  (default) pure random text; KV cache cannot reuse across requests
             --prompt-tokens N      exact N tokens per prompt (needs tokenizer)
             --prompt-tokens N-M    uniform random in [N, M] (needs tokenizer)
  realworld  HF dataset; must have a "question" string column
             --dataset openai/gsm8k (default)

Features:
  --prefix                use chat API with shared system prompt (prefix cache test)
  --prefix-tokens N|N-M   system prompt token length (exact or range)
  --num-unique-prefixes N number of distinct system prompts to rotate (default: 1)
  --streaming             SSE streaming for complete/mix (enables TTFT/TPOT metrics)

Repeat & aggregate:
  --rerun N               repeat the benchmark N times and report mean/std

Config file:
  --config config.toml    load all parameters from TOML (no other args allowed)

Usage examples:
  python benchmark/benchmark.py classify \\
    --url http://localhost:8000 --model mymodel \\
    --concurrency 1,4,16,64 --batch-size 20

  python benchmark/benchmark.py classify \\
    --data realworld --dataset openai/gsm8k \\
    --url http://localhost:8000 --model mymodel

  python benchmark/benchmark.py embed \\
    --prompt-tokens 10-512 --url http://localhost:8000 --model mymodel

  python benchmark/benchmark.py complete \\
    --max-tokens 1 --prefix --prefix-tokens 512 --num-unique-prefixes 3 \\
    --url http://localhost:8000 --model mymodel --concurrency 1,4,16,64

  python benchmark/benchmark.py complete \\
    --max-tokens 32 --streaming \\
    --url http://localhost:8000 --model mymodel --concurrency 1,4,16

  python benchmark/benchmark.py mix \\
    --prefill-weight 0.7 --gen-weight 0.3 --max-tokens 32 --rerun 3 \\
    --url http://localhost:8000 --model mymodel --concurrency 1,4,16

  python benchmark/benchmark.py --config bench.toml
"""

import argparse
import asyncio
import json
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import aiohttp


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "test"
DEFAULT_TIMEOUT = 120
DEFAULT_DATASET = "openai/gsm8k"

# Default system prompt for prefix cache testing when --prefix-tokens is not set
SYSTEM_PROMPT_TEMPLATE = """You are a content safety classifier. Your task is to analyze the given content and determine if it is safe or unsafe.

Guidelines:
- Answer with ONLY "true" (unsafe) or "false" (safe)
- No explanation needed
- Consider context and intent

Categories to check:
- Hate speech or discrimination
- Violence or threats
- Adult content
- Misinformation
- Illegal activities

Output: Answer with just "true" or "false"."""


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result from one concurrency level of a benchmark run."""
    task: str
    concurrency: int
    requests: int
    successful: int
    failed: int
    total_time_s: float
    throughput_rps: float
    throughput_items_per_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_tokens_per_s: float = 0.0
    output_tokens_per_s: float = 0.0
    avg_latency_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    extra: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def check_server(url: str) -> dict:
    """Check server health (sync, called once at startup)."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                try:
                    return json.loads(resp.read())
                except json.JSONDecodeError:
                    return {"status": "ready"}
            return {"status": "error", "code": resp.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_model_served(url: str, model: str, timeout_s: int = 5) -> dict:
    """Verify the requested model is advertised by /v1/models."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "unknown", "message": f"model list unavailable: {e}"}

    models = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not models:
        return {"status": "unknown", "message": "no model ids in /v1/models"}
    if model in models:
        return {"status": "ok", "models": models}
    return {"status": "mismatch", "requested": model, "served_models": models}


def generate_random_text(min_chars: int = 64, max_chars: int = 256,
                         rng: Optional[random.Random] = None) -> str:
    """Generate random ASCII text. Used for synthetic data without token control."""
    if rng is None:
        rng = random.Random()
    chars = string.ascii_letters + string.digits + " .,!?;:-"
    length = rng.randint(min_chars, max_chars)
    return "".join(rng.choice(chars) for _ in range(length))


def compute_latency_stats(latencies: list[float]) -> dict:
    """Compute latency percentile statistics from a list of ms values."""
    if not latencies:
        return {}
    s = sorted(latencies)
    n = len(s)
    return {
        "avg": statistics.mean(s),
        "p50": statistics.median(s),
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "min": s[0],
        "max": s[-1],
    }


def parse_csv_ints(value: str) -> list[int]:
    """Parse a comma-separated list of integers (e.g. '1,4,16,64')."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one integer")
    return [int(p) for p in parts]


def parse_prompt_tokens(spec: str) -> "int | tuple[int, int]":
    """
    Parse --prompt-tokens / --prefix-tokens argument.
      '128'    → 128          exact token count
      '10-512' → (10, 512)   uniform random in [10, 512]
    """
    spec = spec.strip()
    if "-" in spec:
        lo_str, hi_str = spec.split("-", 1)
        lo, hi = int(lo_str.strip()), int(hi_str.strip())
        if lo < 1 or hi < lo:
            raise ValueError(f"invalid range '{spec}': must satisfy 1 ≤ lo ≤ hi")
        return (lo, hi)
    n = int(spec)
    if n < 1:
        raise ValueError("prompt-tokens must be ≥ 1")
    return n


def load_tokenizer(model: str):
    """Load a HuggingFace tokenizer for exact token-length generation."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Exact token-length control requires the 'transformers' package. "
            "Install with: pip install transformers"
        ) from e
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def build_one_token_units(tokenizer) -> list[str]:
    """Return a pool of single-word strings that each tokenize to exactly one token."""
    candidates = [
        " a", " b", " c", " d", " e", " f", " g", " h", " i", " j",
        " k", " l", " m", " n", " o", " p", " q", " r", " s", " t",
        " u", " v", " w", " x", " y", " z",
        " hello", " world", " foo", " bar", " baz",
        " yes", " no", " maybe",
    ]
    units = [u for u in candidates
             if len(tokenizer.encode(u, add_special_tokens=False)) == 1]
    if not units:
        raise RuntimeError("No stable 1-token units found for this tokenizer")
    return units


def generate_exact_token_text(tokenizer, one_token_units: list[str],
                               target_tokens: int, rng: random.Random,
                               retries: int = 8) -> str:
    """Generate a string that tokenizes to exactly `target_tokens` tokens."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be ≥ 1")
    for _ in range(retries):
        text = "".join(
            one_token_units[rng.randrange(len(one_token_units))]
            for _ in range(target_tokens)
        )
        if len(tokenizer.encode(text, add_special_tokens=False)) == target_tokens:
            return text
    raise RuntimeError(
        f"Could not generate an exact {target_tokens}-token string after {retries} retries"
    )


def format_result_table(results: list[BenchmarkResult]) -> str:
    """Format a list of BenchmarkResult as a plain-text table."""
    has_in_tok = any(r.input_tokens_per_s > 0 for r in results)
    has_out_tok = any(r.output_tokens_per_s > 0 for r in results)
    header = (
        f"{'Conc':>6} {'Reqs':>6} {'Fail':>5} "
        f"{'RPS':>10} "
        + (f"{'In-tok/s':>11} " if has_in_tok else "")
        + (f"{'Out-tok/s':>11} " if has_out_tok else "")
        + f"{'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.concurrency:>6} {r.successful:>6} {r.failed:>5} "
            f"{r.throughput_rps:>10.2f} "
            + (f"{r.input_tokens_per_s:>11.0f} " if has_in_tok else "")
            + (f"{r.output_tokens_per_s:>11.0f} " if has_out_tok else "")
            + f"{r.avg_latency_ms:>10.2f} {r.p50_ms:>10.2f} {r.p95_ms:>10.2f}"
        )
    return "\n".join(rows)


def aggregate_rerun_results(all_runs: list[list[BenchmarkResult]]) -> list[BenchmarkResult]:
    """
    Aggregate results across multiple reruns by concurrency level.
    Returns one BenchmarkResult per concurrency level with mean values and
    throughput std/CV stored in the extra dict.
    """
    from collections import defaultdict
    by_conc: dict[int, list[BenchmarkResult]] = defaultdict(list)
    for run in all_runs:
        for r in run:
            by_conc[r.concurrency].append(r)

    aggregated = []
    for conc in sorted(by_conc):
        rows = by_conc[conc]

        def avg(vals: list[float]) -> float:
            return statistics.mean(vals)

        rps_vals = [r.throughput_rps for r in rows]
        rps_mean = avg(rps_vals)
        rps_std = statistics.pstdev(rps_vals) if len(rps_vals) > 1 else 0.0
        rps_cv = rps_std / rps_mean if rps_mean > 0 else 0.0

        aggregated.append(BenchmarkResult(
            task=rows[0].task,
            concurrency=conc,
            requests=rows[0].requests,
            successful=int(round(avg([r.successful for r in rows]))),
            failed=int(round(avg([r.failed for r in rows]))),
            total_time_s=avg([r.total_time_s for r in rows]),
            throughput_rps=rps_mean,
            throughput_items_per_s=avg([r.throughput_items_per_s for r in rows]),
            avg_latency_ms=avg([r.avg_latency_ms for r in rows]),
            p50_ms=avg([r.p50_ms for r in rows]),
            p95_ms=avg([r.p95_ms for r in rows]),
            p99_ms=avg([r.p99_ms for r in rows]),
            min_ms=avg([r.min_ms for r in rows]),
            max_ms=avg([r.max_ms for r in rows]),
            extra={
                "rerun_count": len(rows),
                "throughput_rps_std": rps_std,
                "throughput_rps_cv": rps_cv,
            },
        ))
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_realworld_dataset(dataset: str, max_samples: int, seed: int = 42) -> list[str]:
    """
    Load prompts from a HuggingFace dataset.
    The dataset must have a 'question' string column.
    Defaults to 'openai/gsm8k' (train split, 'main' config).
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset '{dataset}'...")
    try:
        if dataset == DEFAULT_DATASET:
            ds = hf_load(dataset, "main", split="train")
        else:
            try:
                ds = hf_load(dataset, split="test")
            except Exception:
                ds = hf_load(dataset, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset '{dataset}': {e}")
        sys.exit(1)

    if "question" not in ds.column_names:
        available = ", ".join(ds.column_names)
        print(f"ERROR: dataset '{dataset}' has no 'question' column. "
              f"Available columns: {available}")
        sys.exit(1)

    questions = [str(row["question"]) for row in ds if row.get("question")]
    if not questions:
        print(f"ERROR: dataset '{dataset}' has no non-empty 'question' entries")
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(questions)
    # Repeat if the dataset is smaller than requested
    while len(questions) < max_samples:
        questions = questions * 2
    questions = questions[:max_samples]
    print(f"  Loaded {len(questions)} prompts from '{dataset}'")
    return questions


def build_prompts(
    count: int,
    data_type: str,
    prompt_tokens_spec: "None | int | tuple[int, int]",
    dataset: str,
    model: str,
    seed: int,
    rng: random.Random,
) -> list[str]:
    """
    Generate `count` prompts according to the data configuration.

    data_type == "synthetic":
      - prompt_tokens_spec is None         → random ASCII chars (no tokenizer needed)
      - prompt_tokens_spec is int N        → exactly N tokens each
      - prompt_tokens_spec is (lo, hi)     → uniform random token count in [lo, hi]

    data_type == "realworld":
      - load from HF dataset, 'question' column; token spec is ignored
    """
    if data_type == "realworld":
        return load_realworld_dataset(dataset, max_samples=count, seed=seed)

    # synthetic
    if prompt_tokens_spec is None:
        return [generate_random_text(64, 256, rng) for _ in range(count)]

    tokenizer = load_tokenizer(model)
    units = build_one_token_units(tokenizer)
    prompts = []
    for _ in range(count):
        if isinstance(prompt_tokens_spec, tuple):
            lo, hi = prompt_tokens_spec
            target = rng.randint(lo, hi)
        else:
            target = prompt_tokens_spec
        prompts.append(generate_exact_token_text(tokenizer, units, target, rng))
    return prompts


def build_prefix_prompts(
    num_unique: int,
    prefix_tokens_spec: "None | int | tuple[int, int]",
    model: str,
    rng: random.Random,
) -> list[str]:
    """
    Generate `num_unique` distinct system prompts for prefix cache testing.
    If prefix_tokens_spec is None, use the default SYSTEM_PROMPT_TEMPLATE
    (all copies identical when num_unique == 1; random text for num_unique > 1).
    """
    if prefix_tokens_spec is None:
        if num_unique == 1:
            return [SYSTEM_PROMPT_TEMPLATE]
        # Generate distinct random prefixes for rotation
        return [generate_random_text(300, 500, rng) for _ in range(num_unique)]

    tokenizer = load_tokenizer(model)
    units = build_one_token_units(tokenizer)
    prefixes = []
    for _ in range(num_unique):
        if isinstance(prefix_tokens_spec, tuple):
            lo, hi = prefix_tokens_spec
            target = rng.randint(lo, hi)
        else:
            target = prefix_tokens_spec
        prefixes.append(generate_exact_token_text(tokenizer, units, target, rng))
    return prefixes


# ─────────────────────────────────────────────────────────────────────────────
# Async API Client
# ─────────────────────────────────────────────────────────────────────────────

class AsyncAPIClient:
    """
    Async HTTP client for the inference server.
    Endpoints are hardcoded per task; base_url is the server root.
    Uses force_close=True to avoid connection buffering artifacts on TTFT measurements.
    """

    def __init__(self, base_url: str, model: str,
                 timeout: int = DEFAULT_TIMEOUT, max_connections: int = 0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        connector = aiohttp.TCPConnector(limit=max_connections, force_close=True)
        self.session = aiohttp.ClientSession(connector=connector, timeout=self.timeout)

    @staticmethod
    async def _read_body(resp: aiohttp.ClientResponse) -> bytes:
        """Read response body; raise on HTTP errors. JSON parsing done by caller."""
        raw = await resp.read()
        if resp.status >= 400:
            snippet = raw.decode("utf-8", errors="replace").strip().replace("\n", " ")
            raise RuntimeError(f"HTTP {resp.status}: {snippet[:500]}")
        return raw

    async def close(self):
        await self.session.close()

    async def classify(self, texts: list[str]) -> tuple[float, dict]:
        start = time.perf_counter()
        async with self.session.post(
            f"{self.base_url}/v1/classify",
            json={"model": self.model, "input": texts},
        ) as resp:
            raw = await self._read_body(resp)
        return (time.perf_counter() - start) * 1000, json.loads(raw)

    async def embed(self, texts: list[str]) -> tuple[float, dict]:
        start = time.perf_counter()
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": texts},
        ) as resp:
            raw = await self._read_body(resp)
        return (time.perf_counter() - start) * 1000, json.loads(raw)

    async def complete(self, prompt: str, max_tokens: int = 1) -> tuple[float, dict]:
        start = time.perf_counter()
        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json={"model": self.model, "prompt": prompt,
                  "max_tokens": max_tokens, "temperature": 0.0},
        ) as resp:
            raw = await self._read_body(resp)
        return (time.perf_counter() - start) * 1000, json.loads(raw)

    async def complete_streaming(self, prompt: str, max_tokens: int) -> dict:
        start = time.perf_counter()
        ttft: Optional[float] = None
        tokens = 0
        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json={"model": self.model, "prompt": prompt,
                  "max_tokens": max_tokens, "temperature": 0.0, "stream": True},
        ) as resp:
            resp.raise_for_status()
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        if parsed.get("choices") and parsed["choices"][0].get("text"):
                            if ttft is None:
                                ttft = (time.perf_counter() - start) * 1000
                            tokens += 1
                    except json.JSONDecodeError:
                        pass
        total_ms = (time.perf_counter() - start) * 1000
        tpot_ms = (total_ms - ttft) / (tokens - 1) if tokens > 1 and ttft is not None else 0.0
        return {"ttft_ms": ttft or total_ms, "total_ms": total_ms,
                "tokens": tokens, "tpot_ms": tpot_ms}

    async def chat(self, messages: list[dict], max_tokens: int = 1) -> tuple[float, dict]:
        start = time.perf_counter()
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.0},
        ) as resp:
            raw = await self._read_body(resp)
        return (time.perf_counter() - start) * 1000, json.loads(raw)

    async def chat_streaming(self, messages: list[dict], max_tokens: int) -> dict:
        start = time.perf_counter()
        ttft: Optional[float] = None
        tokens = 0
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.0, "stream": True},
        ) as resp:
            resp.raise_for_status()
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        choices = parsed.get("choices", [])
                        if choices and choices[0].get("delta", {}).get("content"):
                            if ttft is None:
                                ttft = (time.perf_counter() - start) * 1000
                            tokens += 1
                    except json.JSONDecodeError:
                        pass
        total_ms = (time.perf_counter() - start) * 1000
        tpot_ms = (total_ms - ttft) / (tokens - 1) if tokens > 1 and ttft is not None else 0.0
        return {"ttft_ms": ttft or total_ms, "total_ms": total_ms,
                "tokens": tokens, "tpot_ms": tpot_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_batch_benchmark(
    task: str,                        # "classify" or "embed"
    url: str,
    model: str,
    concurrency_levels: list[int],
    requests_per_level: list[int],
    prompts: list[str],               # pre-generated; len == warmup*bs + sum(rpl)*bs
    batch_size: int,
    warmup: int,
    timeout: int,
) -> list[BenchmarkResult]:
    """Classify or embed benchmark: multiple texts batched into each HTTP request."""
    results: list[BenchmarkResult] = []

    print(f"\n{'='*70}")
    print(f"  Task: {task}")
    print(f"  Requests/level: {requests_per_level}  Batch size: {batch_size}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"{'='*70}")

    async def call(client: AsyncAPIClient, batch: list[str]) -> tuple[float, int]:
        if task == "classify":
            lat, resp = await client.classify(batch)
        else:
            lat, resp = await client.embed(batch)
        return lat, len(resp.get("data", []))

    # Warmup
    print(f"\nWarming up ({warmup} requests)...")
    client = AsyncAPIClient(url, model, timeout)
    try:
        for i in range(warmup):
            batch = prompts[i * batch_size:(i + 1) * batch_size]
            lat, _ = await call(client, batch)
            print(f"  Warmup {i + 1}/{warmup}: {lat:.1f}ms")
    except Exception as e:
        print(f"  Warmup error: {e}")
    await client.close()

    cumulative_reqs = 0
    for level_idx, conc in enumerate(concurrency_levels):
        rpl = requests_per_level[level_idx]
        if level_idx > 0:
            print(f"\n  Cooling down (2s)...")
            await asyncio.sleep(2)

        print(f"\n--- Concurrency: {conc}  Requests: {rpl} ---")

        latencies: list[float] = []
        items_processed = 0
        errors = 0
        error_msgs: list[str] = []
        completed = 0

        client = AsyncAPIClient(url, model, timeout, max_connections=0)
        sem = asyncio.Semaphore(conc)
        level_offset = warmup * batch_size + cumulative_reqs * batch_size

        async def worker(idx: int, _offset: int = level_offset,
                         _client: AsyncAPIClient = client) -> tuple[float, int, bool, str]:
            async with sem:
                try:
                    start = _offset + idx * batch_size
                    batch = _prompts[start:start + batch_size]
                    lat, items = await call(_client, batch)
                    return lat, items, True, ""
                except Exception as e:
                    return 0.0, 0, False, str(e)

        _prompts = prompts  # captured in worker closure

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(rpl)]
        for coro in asyncio.as_completed(tasks):
            lat, items, ok, err = await coro
            if ok:
                latencies.append(lat)
                items_processed += items
            else:
                errors += 1
                if err and err not in error_msgs:
                    error_msgs.append(err)
            completed += 1
            if completed % 50 == 0:
                print(f"\r  Progress: {completed}/{rpl}", end="", flush=True)

        total_time = time.perf_counter() - t0
        await client.close()
        print(f"\r  Progress: {rpl}/{rpl} - Done")
        if error_msgs:
            print(f"  Errors ({errors}): {'; '.join(error_msgs[:3])}")

        cumulative_reqs += rpl
        stats = compute_latency_stats(latencies)
        result = BenchmarkResult(
            task=task,
            concurrency=conc,
            requests=rpl,
            successful=len(latencies),
            failed=errors,
            total_time_s=total_time,
            throughput_rps=len(latencies) / total_time if total_time > 0 else 0.0,
            throughput_items_per_s=items_processed / total_time if total_time > 0 else 0.0,
            avg_latency_ms=stats.get("avg", 0.0),
            p50_ms=stats.get("p50", 0.0),
            p95_ms=stats.get("p95", 0.0),
            p99_ms=stats.get("p99", 0.0),
            min_ms=stats.get("min", 0.0),
            max_ms=stats.get("max", 0.0),
        )
        results.append(result)
        print(f"  Throughput: {result.throughput_rps:.2f} req/s  "
              f"| {result.throughput_items_per_s:.2f} items/s")
        print(f"  Latency:    avg={result.avg_latency_ms:.1f}ms  "
              f"p50={result.p50_ms:.1f}ms  p95={result.p95_ms:.1f}ms")

    return results


async def run_complete_benchmark(
    url: str,
    model: str,
    concurrency_levels: list[int],
    requests_per_level: list[int],
    prompts: list[str],               # pre-generated; len == warmup + sum(rpl)
    prefixes: Optional[list[str]],    # None = no prefix; list = rotate these system prompts
    max_tokens: int,
    streaming: bool,
    warmup: int,
    timeout: int,
) -> list[BenchmarkResult]:
    """
    Complete benchmark.
      prefixes=None  → /v1/completions with plain prompt
      prefixes=list  → /v1/chat/completions with [system, user] messages
      streaming=True and max_tokens>1 → SSE; collects TTFT and TPOT
    """
    use_chat = prefixes is not None
    do_stream = streaming and max_tokens > 1

    label_parts = ["complete"]
    if use_chat:
        label_parts.append("prefix")
    if do_stream:
        label_parts.append("streaming")
    task_label = "+".join(label_parts)

    results: list[BenchmarkResult] = []

    print(f"\n{'='*70}")
    print(f"  Task: complete  max_tokens={max_tokens}  "
          f"prefix={'yes (' + str(len(prefixes)) + ' unique)' if use_chat else 'no'}  "
          f"streaming={do_stream}")
    print(f"  Requests/level: {requests_per_level}  Concurrency: {concurrency_levels}")
    print(f"{'='*70}")

    async def send(client: AsyncAPIClient, idx: int, prompt: str) -> dict:
        """Dispatch one request; returns a metrics dict."""
        if use_chat:
            msgs = [
                {"role": "system", "content": prefixes[idx % len(prefixes)]},
                {"role": "user",   "content": prompt},
            ]
            if do_stream:
                return await client.chat_streaming(msgs, max_tokens)
            lat, resp = await client.chat(msgs, max_tokens)
            usage = resp.get("usage", {})
            return {"total_ms": lat, "ttft_ms": lat, "tpot_ms": 0.0,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)}
        else:
            if do_stream:
                return await client.complete_streaming(prompt, max_tokens)
            lat, resp = await client.complete(prompt, max_tokens)
            usage = resp.get("usage", {})
            return {"total_ms": lat, "ttft_ms": lat, "tpot_ms": 0.0,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)}

    # Warmup
    print(f"\nWarming up ({warmup} requests)...")
    client = AsyncAPIClient(url, model, timeout)
    try:
        for i in range(warmup):
            m = await send(client, i, prompts[i])
            print(f"  Warmup {i + 1}/{warmup}: {m['total_ms']:.1f}ms")
    except Exception as e:
        print(f"  Warmup error: {e}")
    await client.close()

    cumulative_reqs = 0
    for level_idx, conc in enumerate(concurrency_levels):
        rpl = requests_per_level[level_idx]
        if level_idx > 0:
            print(f"\n  Cooling down (2s)...")
            await asyncio.sleep(2)

        print(f"\n--- Concurrency: {conc}  Requests: {rpl} ---")

        total_lats: list[float] = []
        ttft_list: list[float] = []
        tpot_list: list[float] = []
        prompt_token_counts: list[int] = []
        completion_token_counts: list[int] = []
        errors = 0
        completed = 0

        client = AsyncAPIClient(url, model, timeout, max_connections=0)
        sem = asyncio.Semaphore(conc)
        level_offset = warmup + cumulative_reqs
        _prompts = prompts

        async def worker(idx: int, _off: int = level_offset,
                         _client: AsyncAPIClient = client) -> tuple[dict, bool]:
            async with sem:
                try:
                    prompt = _prompts[(_off + idx) % len(_prompts)]
                    return await send(_client, idx, prompt), True
                except Exception as e:
                    return {"error": str(e)}, False

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(rpl)]
        for coro in asyncio.as_completed(tasks):
            m, ok = await coro
            if ok:
                total_lats.append(m["total_ms"])
                ttft_list.append(m["ttft_ms"])
                if m.get("tpot_ms", 0.0) > 0:
                    tpot_list.append(m["tpot_ms"])
                prompt_token_counts.append(m.get("prompt_tokens", 0))
                completion_token_counts.append(m.get("completion_tokens", m.get("tokens", 0)))
            else:
                errors += 1
            completed += 1
            if completed % 50 == 0:
                print(f"\r  Progress: {completed}/{rpl}", end="", flush=True)

        total_time = time.perf_counter() - t0
        await client.close()
        print(f"\r  Progress: {rpl}/{rpl} - Done")

        cumulative_reqs += rpl
        total_input = sum(prompt_token_counts)
        total_output = sum(completion_token_counts)
        lat_stats = compute_latency_stats(total_lats)
        ttft_stats = compute_latency_stats(ttft_list) if do_stream else {}
        tpot_stats = compute_latency_stats(tpot_list) if tpot_list else {}

        extra: dict = {"max_tokens": max_tokens}
        if do_stream and ttft_stats:
            extra.update({
                "ttft_avg_ms": ttft_stats["avg"],
                "ttft_p50_ms": ttft_stats["p50"],
                "ttft_p95_ms": ttft_stats["p95"],
            })
        if tpot_stats:
            extra.update({
                "tpot_avg_ms": tpot_stats["avg"],
                "tpot_p50_ms": tpot_stats["p50"],
                "tpot_p95_ms": tpot_stats["p95"],
            })

        result = BenchmarkResult(
            task=task_label,
            concurrency=conc,
            requests=rpl,
            successful=len(total_lats),
            failed=errors,
            total_time_s=total_time,
            throughput_rps=len(total_lats) / total_time if total_time > 0 else 0.0,
            input_tokens=total_input,
            output_tokens=total_output,
            input_tokens_per_s=total_input / total_time if total_time > 0 else 0.0,
            output_tokens_per_s=total_output / total_time if total_time > 0 else 0.0,
            avg_latency_ms=lat_stats.get("avg", 0.0),
            p50_ms=lat_stats.get("p50", 0.0),
            p95_ms=lat_stats.get("p95", 0.0),
            p99_ms=lat_stats.get("p99", 0.0),
            min_ms=lat_stats.get("min", 0.0),
            max_ms=lat_stats.get("max", 0.0),
            extra=extra,
        )
        results.append(result)

        tps_parts = []
        if total_input > 0:
            tps_parts.append(f"{result.input_tokens_per_s:.0f} in-tok/s")
        if total_output > 0:
            tps_parts.append(f"{result.output_tokens_per_s:.0f} out-tok/s")
        tps_str = f" | {' | '.join(tps_parts)}" if tps_parts else ""
        print(f"  Throughput: {result.throughput_rps:.2f} req/s{tps_str}")
        print(f"  Latency:    avg={result.avg_latency_ms:.1f}ms  "
              f"p50={result.p50_ms:.1f}ms  p95={result.p95_ms:.1f}ms")
        if do_stream and ttft_stats:
            print(f"  TTFT:       avg={ttft_stats['avg']:.1f}ms  p95={ttft_stats['p95']:.1f}ms")
        if tpot_stats:
            print(f"  TPOT:       avg={tpot_stats['avg']:.2f}ms  p95={tpot_stats['p95']:.2f}ms")

    return results


async def run_mix_benchmark(
    url: str,
    model: str,
    concurrency_levels: list[int],
    requests_per_level: list[int],
    prompts: list[str],               # shared data source
    prefixes: Optional[list[str]],
    prefill_weight: float,
    gen_weight: float,
    max_tokens: int,
    streaming: bool,
    warmup: int,
    timeout: int,
    seed: int,
) -> list[BenchmarkResult]:
    """
    Mix benchmark: prefill-only (max_tokens=1) and generation (max_tokens=max_tokens)
    requests issued concurrently, sampled by weight from the same prompts.
    """
    total_w = prefill_weight + gen_weight
    prefill_ratio = prefill_weight / total_w
    use_chat = prefixes is not None
    do_stream = streaming and max_tokens > 1

    results: list[BenchmarkResult] = []

    print(f"\n{'='*70}")
    print(f"  Task: mix  prefill={prefill_ratio:.0%}  gen={1 - prefill_ratio:.0%}  "
          f"max_tokens={max_tokens}  streaming={do_stream}")
    print(f"  Prefix: {'yes (' + str(len(prefixes)) + ' unique)' if use_chat else 'no'}  "
          f"Requests/level: {requests_per_level}")
    print(f"{'='*70}")

    async def send_prefill(client: AsyncAPIClient, idx: int, prompt: str) -> dict:
        if use_chat:
            msgs = [{"role": "system", "content": prefixes[idx % len(prefixes)]},
                    {"role": "user",   "content": prompt}]
            lat, _ = await client.chat(msgs, max_tokens=1)
        else:
            lat, _ = await client.complete(prompt, max_tokens=1)
        return {"total_ms": lat, "kind": "prefill"}

    async def send_gen(client: AsyncAPIClient, idx: int, prompt: str) -> dict:
        if use_chat:
            msgs = [{"role": "system", "content": prefixes[idx % len(prefixes)]},
                    {"role": "user",   "content": prompt}]
            if do_stream:
                m = await client.chat_streaming(msgs, max_tokens)
            else:
                lat, resp = await client.chat(msgs, max_tokens)
                usage = resp.get("usage", {})
                m = {"total_ms": lat, "ttft_ms": lat, "tpot_ms": 0.0,
                     "prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0)}
        else:
            if do_stream:
                m = await client.complete_streaming(prompt, max_tokens)
            else:
                lat, resp = await client.complete(prompt, max_tokens)
                usage = resp.get("usage", {})
                m = {"total_ms": lat, "ttft_ms": lat, "tpot_ms": 0.0,
                     "prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0)}
        m["kind"] = "gen"
        return m

    # Warmup
    print(f"\nWarming up ({warmup} requests)...")
    client = AsyncAPIClient(url, model, timeout)
    try:
        for i in range(warmup):
            prompt = prompts[i % len(prompts)]
            if i % 2 == 0:
                m = await send_prefill(client, i, prompt)
            else:
                m = await send_gen(client, i, prompt)
            print(f"  Warmup {i + 1}/{warmup}: {m['total_ms']:.1f}ms ({m['kind']})")
    except Exception as e:
        print(f"  Warmup error: {e}")
    await client.close()

    cumulative_reqs = 0
    for level_idx, conc in enumerate(concurrency_levels):
        rpl = requests_per_level[level_idx]
        if level_idx > 0:
            print(f"\n  Cooling down (2s)...")
            await asyncio.sleep(2)

        print(f"\n--- Concurrency: {conc}  Requests: {rpl} ---")

        # Pre-assign request kinds for determinism
        kind_rng = random.Random(seed + level_idx)
        kinds = ["prefill" if kind_rng.random() < prefill_ratio else "gen"
                 for _ in range(rpl)]

        lats: dict[str, list[float]] = {"prefill": [], "gen": []}
        prompt_token_counts: list[int] = []
        completion_token_counts: list[int] = []
        errors = 0
        completed = 0

        client = AsyncAPIClient(url, model, timeout, max_connections=0)
        sem = asyncio.Semaphore(conc)
        level_offset = warmup + cumulative_reqs
        _prompts = prompts

        async def worker(idx: int, _off: int = level_offset,
                         _client: AsyncAPIClient = client) -> tuple[dict, bool]:
            async with sem:
                try:
                    prompt = _prompts[(_off + idx) % len(_prompts)]
                    if kinds[idx] == "prefill":
                        return await send_prefill(_client, idx, prompt), True
                    else:
                        return await send_gen(_client, idx, prompt), True
                except Exception as e:
                    return {"error": str(e), "kind": kinds[idx]}, False

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(rpl)]
        for coro in asyncio.as_completed(tasks):
            m, ok = await coro
            if ok:
                lats[m["kind"]].append(m["total_ms"])
                prompt_token_counts.append(m.get("prompt_tokens", 0))
                completion_token_counts.append(m.get("completion_tokens", m.get("tokens", 0)))
            else:
                errors += 1
            completed += 1
            if completed % 50 == 0:
                print(f"\r  Progress: {completed}/{rpl}", end="", flush=True)

        total_time = time.perf_counter() - t0
        await client.close()
        print(f"\r  Progress: {rpl}/{rpl} - Done")

        cumulative_reqs += rpl
        for kind, kind_lats in lats.items():
            if kind_lats:
                s = compute_latency_stats(kind_lats)
                print(f"  {kind:>8}: {len(kind_lats):4d} reqs  "
                      f"avg={s['avg']:.1f}ms  p95={s['p95']:.1f}ms")

        all_lats = lats["prefill"] + lats["gen"]
        total_input = sum(prompt_token_counts)
        total_output = sum(completion_token_counts)
        stats = compute_latency_stats(all_lats)
        result = BenchmarkResult(
            task="mix",
            concurrency=conc,
            requests=rpl,
            successful=len(all_lats),
            failed=errors,
            total_time_s=total_time,
            throughput_rps=len(all_lats) / total_time if total_time > 0 else 0.0,
            input_tokens=total_input,
            output_tokens=total_output,
            input_tokens_per_s=total_input / total_time if total_time > 0 else 0.0,
            output_tokens_per_s=total_output / total_time if total_time > 0 else 0.0,
            avg_latency_ms=stats.get("avg", 0.0),
            p50_ms=stats.get("p50", 0.0),
            p95_ms=stats.get("p95", 0.0),
            p99_ms=stats.get("p99", 0.0),
            min_ms=stats.get("min", 0.0),
            max_ms=stats.get("max", 0.0),
            extra={
                "prefill_count": len(lats["prefill"]),
                "gen_count": len(lats["gen"]),
                "prefill_ratio": round(prefill_ratio, 4),
                "max_tokens": max_tokens,
            },
        )
        results.append(result)
        print(f"  Overall:  {result.throughput_rps:.2f} req/s  "
              f"avg={result.avg_latency_ms:.1f}ms  p95={result.p95_ms:.1f}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TOML Config
# ─────────────────────────────────────────────────────────────────────────────

def load_toml_config(path: str) -> dict:
    """Load a TOML config file. Requires Python ≥ 3.11 or 'pip install tomli'."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            print("ERROR: TOML support requires Python ≥ 3.11 or: pip install tomli")
            sys.exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)


def toml_to_namespace(cfg: dict):
    """Convert a flat+nested TOML config dict into an argparse-like SimpleNamespace."""
    import types
    args = types.SimpleNamespace()

    args.task = cfg.get("task") or cfg.get("command")
    if not args.task:
        print("ERROR: TOML config must have a 'task' field "
              "(classify | embed | complete | mix)")
        sys.exit(1)

    args.url = cfg.get("url", DEFAULT_URL)
    args.model = cfg.get("model", DEFAULT_MODEL)
    conc = cfg.get("concurrency", [1, 4, 16, 64])
    args.concurrency = conc if isinstance(conc, list) else parse_csv_ints(str(conc))
    reqs = cfg.get("requests", 200)
    args.requests = reqs if isinstance(reqs, list) else [reqs]
    args.rerun = cfg.get("rerun", 1)
    args.warmup = cfg.get("warmup", 5)
    args.seed = cfg.get("seed", 42)
    args.timeout = cfg.get("timeout", DEFAULT_TIMEOUT)
    args.output = cfg.get("output", None)
    args.batch_size = cfg.get("batch_size", 20)
    args.max_tokens = cfg.get("max_tokens", 1)
    args.streaming = cfg.get("streaming", False)
    args.prefill_weight = cfg.get("prefill_weight", 0.7)
    args.gen_weight = cfg.get("gen_weight", 0.3)

    data_sec = cfg.get("data", {})
    args.data = data_sec.get("type", "synthetic") if isinstance(data_sec, dict) else "synthetic"
    args.dataset = data_sec.get("dataset", DEFAULT_DATASET) if isinstance(data_sec, dict) else DEFAULT_DATASET
    pt = data_sec.get("prompt_tokens") if isinstance(data_sec, dict) else None
    args.prompt_tokens = str(pt) if pt is not None else None

    prefix_sec = cfg.get("prefix", {})
    if isinstance(prefix_sec, dict) and prefix_sec.get("enabled", False):
        args.prefix = True
        pt2 = prefix_sec.get("prefix_tokens")
        args.prefix_tokens = str(pt2) if pt2 is not None else None
        args.num_unique_prefixes = prefix_sec.get("num_unique", 1)
    else:
        args.prefix = False
        args.prefix_tokens = None
        args.num_unique_prefixes = 1

    return args


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

async def async_main():
    # Special-case: --config <path> [--model X] [--url Y]
    # TOML sets all params; --model and --url on the CLI override the TOML values.
    if len(sys.argv) >= 3 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
        args = toml_to_namespace(load_toml_config(config_path))
        # Allow --model and --url overrides after --config
        extra = sys.argv[3:]
        i = 0
        while i < len(extra):
            if extra[i] == "--model" and i + 1 < len(extra):
                args.model = extra[i + 1]; i += 2
            elif extra[i] == "--url" and i + 1 < len(extra):
                args.url = extra[i + 1]; i += 2
            else:
                print(f"ERROR: only --model and --url are allowed alongside --config (got '{extra[i]}')")
                sys.exit(1)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--config":
        print("Usage: benchmark.py --config <path.toml> [--model NAME] [--url URL]")
        sys.exit(1)
    else:
        parser = argparse.ArgumentParser(
            description="Inference Benchmark Suite",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        subparsers = parser.add_subparsers(dest="task")

        # ── Shared argument groups ──────────────────────────────────────────
        def add_common(p):
            p.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
            p.add_argument("--model", default=DEFAULT_MODEL,
                           help="Model name sent in the request body")
            p.add_argument("--concurrency", default="1,4,8,16,32,64,128",
                           help="Comma-separated concurrency levels to sweep")
            p.add_argument("--requests", default="200",
                           help="Requests per concurrency level: single int or comma-separated list matching --concurrency")
            p.add_argument("--rerun", type=int, default=1,
                           help="Repeat the full benchmark N times and aggregate")
            p.add_argument("--warmup", type=int, default=5,
                           help="Warmup requests before timing")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                           help="Per-request timeout in seconds")
            p.add_argument("--output", "-o", default=None,
                           help="Write results to this JSON file")

        def add_data(p):
            p.add_argument("--data", choices=["synthetic", "realworld"],
                           default="synthetic",
                           help="Input data source")
            p.add_argument("--dataset", default=DEFAULT_DATASET,
                           help="HF dataset for --data realworld; "
                                "must have a 'question' string column")
            p.add_argument("--prompt-tokens", default=None, metavar="N|N-M",
                           help="Token length for synthetic data: "
                                "'128' (exact) or '10-512' (uniform range). "
                                "Requires --model to be a valid HF model path.")

        def add_prefix(p):
            p.add_argument("--prefix", action="store_true",
                           help="Use chat API with a shared system prompt "
                                "(tests prefix cache effectiveness)")
            p.add_argument("--prefix-tokens", default=None, metavar="N|N-M",
                           help="System prompt token length (exact or range)")
            p.add_argument("--num-unique-prefixes", type=int, default=1,
                           help="Number of distinct system prompts to rotate "
                                "(default: 1 = all requests share one prefix)")

        # ── classify ───────────────────────────────────────────────────────
        p = subparsers.add_parser("classify", help="POST /v1/classify benchmark")
        add_common(p)
        add_data(p)
        p.add_argument("--batch-size", type=int, default=20,
                       help="Texts per HTTP request")

        # ── embed ──────────────────────────────────────────────────────────
        p = subparsers.add_parser("embed", help="POST /v1/embeddings benchmark")
        add_common(p)
        add_data(p)
        p.add_argument("--batch-size", type=int, default=20,
                       help="Texts per HTTP request")

        # ── complete ───────────────────────────────────────────────────────
        p = subparsers.add_parser("complete",
                                  help="POST /v1/completions benchmark "
                                       "(max_tokens=1 → prefill-only; >1 → generation)")
        add_common(p)
        add_data(p)
        add_prefix(p)
        p.add_argument("--max-tokens", type=int, default=1,
                       help="Decode length (1 = prefill-only, >1 = generation)")
        p.add_argument("--streaming", action="store_true",
                       help="SSE streaming output; collects TTFT and TPOT "
                            "(only meaningful with --max-tokens > 1)")

        # ── mix ────────────────────────────────────────────────────────────
        p = subparsers.add_parser("mix",
                                  help="Mixed prefill-only + generation benchmark")
        add_common(p)
        add_data(p)
        add_prefix(p)
        p.add_argument("--prefill-weight", type=float, default=0.7,
                       help="Relative weight for prefill-only requests")
        p.add_argument("--gen-weight", type=float, default=0.3,
                       help="Relative weight for generation requests")
        p.add_argument("--max-tokens", type=int, default=32,
                       help="Decode length for generation requests")
        p.add_argument("--streaming", action="store_true",
                       help="SSE streaming for generation requests")

        args = parser.parse_args()
        if not args.task:
            parser.print_help()
            sys.exit(1)

        args.concurrency = parse_csv_ints(args.concurrency)
        args.requests = parse_csv_ints(str(args.requests))

        # Fill defaults for attributes not present on all subcommands
        for attr, default in [
            ("batch_size", 20), ("max_tokens", 1), ("streaming", False),
            ("prefix", False), ("prefix_tokens", None), ("num_unique_prefixes", 1),
            ("prefill_weight", 0.7), ("gen_weight", 0.3),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)

    # ── Expand requests to match concurrency levels ─────────────────────────
    n_levels = len(args.concurrency)
    if len(args.requests) == 1:
        args.requests = args.requests * n_levels
    elif len(args.requests) != n_levels:
        print(f"ERROR: --requests has {len(args.requests)} values but "
              f"--concurrency has {n_levels} levels. "
              "Provide a single value or one per concurrency level.")
        sys.exit(1)

    # ── Validation ────────────────────────────────────────────────────────────
    if args.rerun < 1:
        print("ERROR: --rerun must be ≥ 1")
        sys.exit(1)
    if args.max_tokens < 1:
        print("ERROR: --max-tokens must be ≥ 1")
        sys.exit(1)
    if args.task not in ("classify", "embed", "complete", "mix"):
        print(f"ERROR: unknown task '{args.task}'. "
              "Choose from: classify, embed, complete, mix")
        sys.exit(1)
    if getattr(args, "data", "synthetic") == "realworld" and args.prompt_tokens:
        print("ERROR: --prompt-tokens is not compatible with --data realworld "
              "(real data has its own token lengths)")
        sys.exit(1)
    if getattr(args, "streaming", False) and args.max_tokens == 1:
        print("WARNING: --streaming has no effect when --max-tokens=1 "
              "(prefill-only; there is nothing to stream)")

    prompt_tokens_spec = None
    if args.prompt_tokens:
        try:
            prompt_tokens_spec = parse_prompt_tokens(args.prompt_tokens)
        except ValueError as e:
            print(f"ERROR: --prompt-tokens: {e}")
            sys.exit(1)

    prefix_tokens_spec = None
    if getattr(args, "prefix", False) and getattr(args, "prefix_tokens", None):
        try:
            prefix_tokens_spec = parse_prompt_tokens(args.prefix_tokens)
        except ValueError as e:
            print(f"ERROR: --prefix-tokens: {e}")
            sys.exit(1)

    # ── Server check ──────────────────────────────────────────────────────────
    print(f"Checking server at {args.url}...")
    health = check_server(args.url)
    if health.get("status") == "error":
        print(f"ERROR: Server not ready: {health}")
        sys.exit(1)
    print(f"Server ready: {health}")

    model_check = check_model_served(args.url, args.model,
                                     timeout_s=min(args.timeout, 10))
    if model_check.get("status") == "mismatch":
        served = ", ".join(model_check.get("served_models", [])[:8])
        print(f"ERROR: model '{args.model}' not served. Available: {served}")
        sys.exit(1)
    if model_check.get("status") == "ok":
        print(f"Model check: '{args.model}' confirmed")
    else:
        print(f"Model check: skipped ({model_check.get('message')})")

    # ── Prepare data ──────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    batch_size = args.batch_size if args.task in ("classify", "embed") else 1
    items_per_req = batch_size
    total_prompts = (args.warmup + sum(args.requests)) * items_per_req

    print(f"\nPreparing {total_prompts} prompts (data={args.data})...")
    prompts = build_prompts(
        count=total_prompts,
        data_type=getattr(args, "data", "synthetic"),
        prompt_tokens_spec=prompt_tokens_spec,
        dataset=getattr(args, "dataset", DEFAULT_DATASET),
        model=args.model,
        seed=args.seed,
        rng=rng,
    )

    prefixes: Optional[list[str]] = None
    if getattr(args, "prefix", False):
        num_unique = getattr(args, "num_unique_prefixes", 1)
        print(f"Preparing {num_unique} system prompt(s)...")
        prefixes = build_prefix_prompts(
            num_unique=num_unique,
            prefix_tokens_spec=prefix_tokens_spec,
            model=args.model,
            rng=rng,
        )

    # ── Run (with optional rerun) ─────────────────────────────────────────────
    async def run_once() -> list[BenchmarkResult]:
        if args.task in ("classify", "embed"):
            return await run_batch_benchmark(
                task=args.task, url=args.url, model=args.model,
                concurrency_levels=args.concurrency,
                requests_per_level=args.requests,
                prompts=prompts, batch_size=batch_size,
                warmup=args.warmup, timeout=args.timeout,
            )
        elif args.task == "complete":
            return await run_complete_benchmark(
                url=args.url, model=args.model,
                concurrency_levels=args.concurrency,
                requests_per_level=args.requests,
                prompts=prompts, prefixes=prefixes,
                max_tokens=args.max_tokens, streaming=args.streaming,
                warmup=args.warmup, timeout=args.timeout,
            )
        else:  # mix
            return await run_mix_benchmark(
                url=args.url, model=args.model,
                concurrency_levels=args.concurrency,
                requests_per_level=args.requests,
                prompts=prompts, prefixes=prefixes,
                prefill_weight=args.prefill_weight,
                gen_weight=args.gen_weight,
                max_tokens=args.max_tokens, streaming=args.streaming,
                warmup=args.warmup, timeout=args.timeout,
                seed=args.seed,
            )

    all_runs: list[list[BenchmarkResult]] = []
    for run_idx in range(args.rerun):
        if run_idx > 0:
            print(f"\n{'='*70}")
            print(f"  Rerun {run_idx + 1} / {args.rerun}")
            print(f"{'='*70}")
            await asyncio.sleep(2)
        all_runs.append(await run_once())

    final = aggregate_rerun_results(all_runs) if args.rerun > 1 else all_runs[0]

    # ── Summary ───────────────────────────────────────────────────────────────
    rerun_str = f"  rerun={args.rerun}" if args.rerun > 1 else ""
    print(f"\n{'='*70}")
    print(f"  Summary  task={args.task}{rerun_str}")
    print(f"{'='*70}\n")
    print(format_result_table(final))

    if args.rerun > 1 and any(r.extra.get("throughput_rps_cv") for r in final):
        print("\n  Stability (CV = std/mean of RPS across reruns):")
        for r in final:
            cv = r.extra.get("throughput_rps_cv", 0.0)
            std = r.extra.get("throughput_rps_std", 0.0)
            print(f"    conc={r.concurrency:>4}  RPS={r.throughput_rps:.2f} "
                  f"± {std:.2f}  CV={cv:.3f}")

    # ── Save output ───────────────────────────────────────────────────────────
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "task": args.task,
            "rerun": args.rerun,
            "results": [asdict(r) for r in final],
        }
        if args.rerun > 1:
            output_data["all_runs"] = [[asdict(r) for r in run] for run in all_runs]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
