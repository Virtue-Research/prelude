#!/usr/bin/env bash
set -euo pipefail

DATASET_URL="https://huggingface.co/datasets/Virtue-AI-HUB/PolicyGuardData/resolve/main/jingyang/labeled_272k_20260513_compact_nouserprefix_chatml.jsonl"
OUT_FILE="hf_downloads/labeled_272k_20260513_compact_nouserprefix_chatml.jsonl"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Export a Hugging Face token first." >&2
  exit 1
fi

mkdir -p "hf_downloads"
curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" "${DATASET_URL}" -o "${OUT_FILE}"

python3 - "${OUT_FILE}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
groups = set()
label_pat = re.compile(r"^(?:true)(\d+)\|")

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"Invalid JSON on line {i}.", file=sys.stderr)
            raise SystemExit(1)

        # ChatML compact format: label is embedded in assistant content, e.g. "true3|..."
        msgs = row.get("messages", [])
        assistant = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        m = label_pat.match(assistant)
        if m:
            groups.add(int(m.group(1)))

print(len(groups))
PY