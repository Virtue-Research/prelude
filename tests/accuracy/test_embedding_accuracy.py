#!/usr/bin/env python3
"""Reference test for embedding models.

Compares Prelude `/v1/embeddings` responses against HuggingFace hidden-state
last-token embeddings on a fixed prompt set.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BINARY = "target/release/prelude-server"
DEFAULT_PORT = 8099
HEALTH_TIMEOUT = 300
PROCESS_SHUTDOWN_TIMEOUT = 10


def spec_inputs(spec: dict) -> list[str]:
    if isinstance(spec["input"], list):
        return spec["input"]
    return [spec["input"]]


def generate_golden(model_path: str, prompts: list[dict], output_path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    results = {}
    for spec in prompts:
        texts = spec_inputs(spec)
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**encoded)

        hidden = outputs.last_hidden_state.float().cpu()
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        embeddings = hidden[torch.arange(hidden.shape[0]), lengths].numpy()
        prompt_tokens = int(encoded["attention_mask"].sum().item())

        results[spec["id"]] = {
            "prompt_tokens": prompt_tokens,
            "dimensions": int(embeddings.shape[1]),
            "data": [
                {
                    "index": idx,
                    "embedding": embedding.tolist(),
                }
                for idx, embedding in enumerate(embeddings)
            ],
        }

    golden = {
        "model": model_path,
        "generator": "transformers",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(golden, handle, indent=2, ensure_ascii=False)
    print(f"Golden written: {output_path}")
    return golden


def start_server(binary: str, model: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PRELUDE_DEVICE", "cpu")
    env.setdefault("RUST_LOG", "prelude_core=info")

    cmd = [
        binary,
        "--model",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    print(f"Starting server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def wait_for_health(
    url: str,
    proc: subprocess.Popen | None = None,
    timeout: int = HEALTH_TIMEOUT,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            return False
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def stop_process(proc: subprocess.Popen) -> str:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    try:
        stdout, _ = proc.communicate(timeout=PROCESS_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate(timeout=PROCESS_SHUTDOWN_TIMEOUT)
    return stdout.decode("utf-8", errors="replace") if stdout else ""


def query_server(url: str, model: str, spec: dict) -> dict:
    response = requests.post(
        f"{url}/v1/embeddings",
        json={"model": model, "input": spec["input"]},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    denom = np.linalg.norm(lhs_arr) * np.linalg.norm(rhs_arr)
    if denom == 0.0:
        return 1.0
    return float(np.dot(lhs_arr, rhs_arr) / denom)


def compare(spec: dict, golden_entry: dict, response: dict) -> tuple[bool, list[str]]:
    problems = []
    actual = response["data"]
    expected = golden_entry["data"]
    actual_dimensions_match = True

    if len(actual) != len(expected):
        problems.append(f"expected {len(expected)} embeddings, got {len(actual)}")
        return False, problems

    if response["usage"]["prompt_tokens"] != golden_entry["prompt_tokens"]:
        problems.append(
            f"{spec['id']} prompt token mismatch: {response['usage']['prompt_tokens']} != {golden_entry['prompt_tokens']}"
        )

    for idx, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if len(actual_item["embedding"]) != golden_entry["dimensions"]:
            problems.append(f"{spec['id']}[{idx}] dimension mismatch")
            actual_dimensions_match = False
            continue
        similarity = cosine_similarity(expected_item["embedding"], actual_item["embedding"])
        if similarity < 0.999:
            problems.append(
                f"{spec['id']}[{idx}] cosine similarity too low: {similarity:.6f}"
            )

    if actual_dimensions_match:
        expected_matrix = np.asarray(
            [[cosine_similarity(a["embedding"], b["embedding"]) for b in expected] for a in expected],
            dtype=np.float32,
        )
        actual_matrix = np.asarray(
            [[cosine_similarity(a["embedding"], b["embedding"]) for b in actual] for a in actual],
            dtype=np.float32,
        )
        if np.max(np.abs(expected_matrix - actual_matrix)) > 1e-3:
            problems.append(f"{spec['id']} pairwise similarity matrix drifted")

    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding accuracy test")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model path or HF repo ID")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="prelude-server binary")
    parser.add_argument("--server-url", help="Pre-started server URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument(
        "--prompts",
        default=str(SCRIPT_DIR / "embed_prompts.json"),
        help="Prompt specification JSON",
    )
    parser.add_argument(
        "--golden",
        default=str(SCRIPT_DIR / "golden" / "qwen3-embedding-0.6b.json"),
        help="Golden JSON path",
    )
    parser.add_argument("--generate-golden", action="store_true")
    args = parser.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    golden_path = Path(args.golden)
    if args.generate_golden:
        generate_golden(args.model, prompts, golden_path)

    if not golden_path.exists():
        print(f"Golden file not found: {golden_path}")
        return 1

    with open(golden_path, "r", encoding="utf-8") as handle:
        golden = json.load(handle)

    proc = None
    try:
        url = args.server_url
        if not url:
            proc = start_server(args.binary, args.model, args.port)
            url = f"http://localhost:{args.port}"
            if not wait_for_health(url, proc):
                output = stop_process(proc)
                proc = None
                print(f"Server failed to start:\n{output}")
                return 1

        failures = []
        for spec in prompts:
            response = query_server(url, args.model, spec)
            ok, problems = compare(spec, golden["results"][spec["id"]], response)
            if ok:
                print(f"[PASS] {spec['id']}")
            else:
                print(f"[FAIL] {spec['id']}")
                failures.extend(problems)

        if failures:
            for problem in failures:
                print(problem)
            return 1
        return 0
    finally:
        if proc:
            stop_process(proc)


if __name__ == "__main__":
    sys.exit(main())
