#!/usr/bin/env python3
"""On-the-fly accuracy check for Qwen/Qwen3.5-35B-A3B.

Loads HuggingFace transformers as the reference, runs every prompt in
`prompts.json` through both HF and the Prelude server at `--server-url`,
and compares the decoded strings for an exact match. No golden file —
each run is self-contained, so there's nothing to drift or regenerate
when the chat template or server behavior changes.

Usage:
    python tests/qwen3_5_moe/test_accuracy.py --server-url http://localhost:8099
"""

import argparse
import json
import sys
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_JSON = Path(__file__).parent / "prompts.json"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def render_prompt(tokenizer, spec: dict) -> str:
    """Single source of truth for prompt text on both sides."""
    if spec.get("type") == "completion":
        return spec["prompt"]
    return tokenizer.apply_chat_template(
        spec["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )


def hf_generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    ids = out[0, prompt_len:].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


def server_generate(server_url: str, model_id: str, spec: dict) -> str:
    body = {
        "model": model_id,
        "max_tokens": spec["max_tokens"],
        "temperature": 0.0,
    }
    if spec.get("type") == "completion":
        body["prompt"] = spec["prompt"]
        r = requests.post(f"{server_url}/v1/completions", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    body["messages"] = spec["messages"]
    r = requests.post(f"{server_url}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    # Our server splits reasoning models' output on `</think>` into a
    # separate field (matching vLLM's `ChatMessage.reasoning`; the older
    # alias `reasoning_content` is still accepted). Reconstruct the raw
    # tokenizer.decode() byte stream so we can diff against HF directly.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    if reasoning is None:
        return content
    if content:
        return f"{reasoning}</think>{content}"
    return reasoning


def strip_stop_tokens(text: str) -> str:
    """Both sides sometimes decode trailing special tokens. Drop them."""
    for tok in STOP_TOKENS:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx]
    return text.rstrip("\n")


def first_diff(a: str, b: str) -> str:
    """Human-readable pointer to the first character that differs."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"diff@char {i}: hf={a[i:i+24]!r} server={b[i:i+24]!r}"
    if len(a) != len(b):
        shorter = min(len(a), len(b))
        return f"length differs at char {shorter} (hf={len(a)}, server={len(b)})"
    return "no diff"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-url", default="http://localhost:8099",
                   help="Base URL of a running Prelude server")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF model id for the reference load (must match the server)")
    args = p.parse_args()

    prompts = json.loads(PROMPTS_JSON.read_text())["prompts"]

    # Fail fast if the server isn't up — HF loading is slow, don't
    # waste minutes only to find we can't diff anything.
    try:
        requests.get(f"{args.server_url}/v1/models", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"Server at {args.server_url} not reachable: {e}", file=sys.stderr)
        return 1
    print(f"Server up at {args.server_url}")

    print(f"Loading {args.model} as HF reference...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # Torch 2.10 grouped_mm has a data-ptr alignment bug on MoE layers;
        # forcing per-expert eager execution dodges it.
        # https://github.com/pytorch/pytorch/issues/154557
        experts_implementation="eager",
    ).eval()
    print(f"Loaded on {next(model.parameters()).device}")
    print()

    passed = 0
    for spec in prompts:
        pid = spec["id"]
        prompt = render_prompt(tokenizer, spec)
        hf_text = strip_stop_tokens(hf_generate(model, tokenizer, prompt, spec["max_tokens"]))
        server_text = strip_stop_tokens(server_generate(args.server_url, args.model, spec))

        if hf_text == server_text:
            print(f"  [{pid}] MATCH")
            passed += 1
        else:
            print(f"  [{pid}] FAIL")
            print(f"    hf:     {hf_text!r}")
            print(f"    server: {server_text!r}")
            print(f"    {first_diff(hf_text, server_text)}")

    total = len(prompts)
    print()
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
