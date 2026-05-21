#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
from urllib import error, request

from huggingface_hub import hf_hub_download


MODEL_ID = "Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix"
DATASET_ID = "Virtue-AI-HUB/PolicyGuardData"
DATA_FILE = "chat_prompts.jsonl"
OUTPUT_PATH = Path("scratch_runs/policyguard_first10_vllm_outputs.jsonl")
NUM_SAMPLES = 10
MAX_TOKENS = 3
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.environ.get("VLLM_MODEL", MODEL_ID)
REQUEST_TIMEOUT_S = 600
STOP = ["<|im_end|>"]


def load_samples():
    data_path = hf_hub_download(
        DATASET_ID,
        DATA_FILE,
        repo_type="dataset",
        token=True,
    )

    samples = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= NUM_SAMPLES:
                break
            row = json.loads(line)
            messages = row["messages"]
            gold = None
            input_messages = []
            for message in messages:
                if message.get("role") == "assistant":
                    gold = message.get("content", "")
                    break
                input_messages.append(
                    {
                        "role": message["role"],
                        "content": message.get("content", ""),
                    }
                )
            samples.append(
                {
                    "index": idx,
                    "input_messages": input_messages,
                    "gold": gold,
                }
            )
    return samples


def generate_one(messages):
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stop": STOP,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{VLLM_BASE_URL.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {VLLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc

    prediction = (parsed["choices"][0]["message"].get("content") or "").strip()
    usage = parsed.get("usage") or {}
    return {
        "prediction": prediction,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples()

    print(f"Loaded {len(samples)} samples from {DATASET_ID}/{DATA_FILE}")
    print(f"Calling vLLM server at {VLLM_BASE_URL} with model={VLLM_MODEL!r}")

    rows = []
    started = time.time()
    for sample in samples:
        t0 = time.time()
        result = generate_one(sample["input_messages"])
        row = {
            "index": sample["index"],
            "gold": sample["gold"],
            "elapsed_s": round(time.time() - t0, 3),
            **result,
        }
        rows.append(row)
        print(f"[{row['index']}] {row['elapsed_s']}s pred={row['prediction']!r}")

    elapsed = time.time() - started

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(rows)} outputs in {elapsed:.2f}s")
    print(f"Wrote {OUTPUT_PATH}")
    for row in rows:
        print(f"[{row['index']}] pred={row['prediction']!r}")
        print(f"    gold={row['gold']!r}")


if __name__ == "__main__":
    main()
