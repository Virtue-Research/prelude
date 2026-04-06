#!/usr/bin/env python3
"""Minimal debug script: compare rust-infer logprobs vs transformers token-by-token.

Usage:
    # Start server first, then:
    python tests/accuracy/debug_logprobs.py --url http://localhost:8001

    # Or auto-start:
    python tests/accuracy/debug_logprobs.py --binary target/release/prelude-server --model Qwen/Qwen3-0.6B
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The quick brown fox jumps over the"
MAX_TOKENS = 10
TOP_K = 5


def generate_reference(model, tokenizer, prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )

    output_ids = outputs.sequences[0, prompt_len:].tolist()

    tokens = []
    for step, logits_t in enumerate(outputs.logits):
        logits = logits_t.squeeze(0).float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        tid = output_ids[step]
        tok_str = tokenizer.decode([tid])
        tok_lp = log_probs[tid].item()
        topk_vals, topk_ids = torch.topk(log_probs, TOP_K)
        top = [(tokenizer.decode([i.item()]), i.item(), v.item()) for i, v in zip(topk_ids, topk_vals)]
        tokens.append({"id": tid, "token": tok_str, "logprob": tok_lp, "top": top})

    return tokens


def query_server(url, model, prompt, max_tokens):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": TOP_K,
    }
    resp = requests.post(f"{url}/v1/completions", json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Print raw response for inspection
    print("=== Raw API response ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    choice = data["choices"][0]
    text = choice["text"]
    lp = choice.get("logprobs")

    if not lp:
        print("ERROR: No logprobs in response!")
        return []

    tokens = []
    for i, tok_str in enumerate(lp["tokens"]):
        tok_lp = lp["token_logprobs"][i]
        top_map = lp["top_logprobs"][i] if i < len(lp["top_logprobs"]) else {}
        top = sorted(top_map.items(), key=lambda x: -x[1])
        tokens.append({"token": tok_str, "logprob": tok_lp, "top": top})

    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Step 1: transformers reference
    print(f"=== Transformers reference (dtype={torch_dtype}) ===")
    print(f"Prompt: {PROMPT!r}")
    print(f"Max tokens: {MAX_TOKENS}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
    ref_model.eval()
    ref_tokens = generate_reference(ref_model, tokenizer, PROMPT, MAX_TOKENS)
    del ref_model

    print("Reference tokens:")
    for i, t in enumerate(ref_tokens):
        print(f"  [{i}] id={t['id']:6d} token={t['token']!r:15s} logprob={t['logprob']:.6f}")
        for tok_str, tid, lp in t["top"][:3]:
            print(f"        top: {tok_str!r:15s} id={tid:6d} lp={lp:.6f}")
    print()

    # Step 2: rust-infer server
    proc = None
    url = args.url
    try:
        if not url:
            if not args.binary:
                print("ERROR: --url or --binary required")
                sys.exit(1)
            env = {**os.environ, "PRELUDE_DEVICE": "cpu"}
            cmd = [args.binary, "--host", "0.0.0.0", "--port", str(args.port),
                   "--model", args.model]
            print(f"Starting: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            url = f"http://localhost:{args.port}"
            for _ in range(120):
                try:
                    if requests.get(f"{url}/health", timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(1)
            else:
                print("Server failed to start")
                sys.exit(1)
            print("Server ready.\n")

        server_tokens = query_server(url, args.model, PROMPT, MAX_TOKENS)
    finally:
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)

    # Step 3: side-by-side comparison
    print("=== Token-by-token comparison ===")
    print(f"{'idx':>3s}  {'ref_token':>15s} {'ref_lp':>10s}  {'srv_token':>15s} {'srv_lp':>10s}  {'lp_diff':>10s}  match")
    print("-" * 90)

    min_len = min(len(ref_tokens), len(server_tokens))
    for i in range(max(len(ref_tokens), len(server_tokens))):
        ref = ref_tokens[i] if i < len(ref_tokens) else None
        srv = server_tokens[i] if i < len(server_tokens) else None

        ref_tok = ref["token"] if ref else "---"
        ref_lp = f"{ref['logprob']:.6f}" if ref else "---"
        srv_tok = srv["token"] if srv else "---"
        srv_lp = f"{srv['logprob']:.6f}" if srv else "---"

        if ref and srv:
            diff = abs(ref["logprob"] - srv["logprob"])
            match = "OK" if ref["token"] == srv["token"] and diff < 0.01 else "MISMATCH"
            diff_str = f"{diff:.6f}"
        else:
            match = "MISSING"
            diff_str = "---"

        print(f"{i:3d}  {ref_tok!r:>15s} {ref_lp:>10s}  {srv_tok!r:>15s} {srv_lp:>10s}  {diff_str:>10s}  {match}")


if __name__ == "__main__":
    main()
