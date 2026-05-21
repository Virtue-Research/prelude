#!/usr/bin/env python3
import json
import time
from pathlib import Path
from urllib import error, request

from huggingface_hub import hf_hub_download


MODEL_ID = "Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix"
DATASET_ID = "Virtue-AI-HUB/PolicyGuardData"
DATA_FILE = "chat_prompts.jsonl"
SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
OUTPUT_PATH = Path("scratch_runs/policyguard_first10_prelude_outputs.jsonl")
NUM_SAMPLES = 10


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
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"].strip()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    print(f"Loaded {len(samples)} samples from {DATASET_ID}/{DATA_FILE}")

    rows = []
    started = time.time()
    for sample in samples:
        t0 = time.time()
        prediction = generate_one(sample["input_messages"])
        row = {
            "index": sample["index"],
            "prediction": prediction,
            "gold": sample["gold"],
            "elapsed_s": round(time.time() - t0, 3),
        }
        rows.append(row)
        print(f"[{row['index']}] {row['elapsed_s']}s pred={row['prediction']!r}")

    elapsed = time.time() - started
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(rows)} outputs in {elapsed:.2f}s")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
