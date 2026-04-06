#!/usr/bin/env python3
"""Debug script: compare SGLang CPU logprobs vs transformers token-by-token.

Usage:
    # Connect to already-running SGLang server:
    python tests/accuracy/debug_sglang.py --url http://localhost:8090 --model Qwen/Qwen3-0.6B

    # Auto-start SGLang server (requires sglang installed in current venv):
    python tests/accuracy/debug_sglang.py --model Qwen/Qwen3-0.6B

    # Use Docker (sglang-cpu image):
    python tests/accuracy/debug_sglang.py --model Qwen/Qwen3-0.6B --docker
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
    resp = requests.post(f"{url}/v1/completions", json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    print("=== Raw API response ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    choice = data["choices"][0]
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


def wait_for_server(url, timeout=300):
    """Wait for server health endpoint."""
    for _ in range(timeout):
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        # Also try /v1/models (SGLang uses this)
        try:
            r = requests.get(f"{url}/v1/models", timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="URL of running SGLang server")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--docker", action="store_true", help="Use sglang-cpu Docker image")
    parser.add_argument("--ref-dtype", default="bf16", choices=["f32", "bf16"],
                        help="dtype for transformers reference (default: bf16 to match SGLang CPU)")
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.ref_dtype == "bf16" else torch.float32

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

    # Step 2: SGLang server
    proc = None
    container_id = None
    url = args.url

    try:
        if not url:
            url = f"http://localhost:{args.port}"

            if args.docker:
                print(f"Starting SGLang CPU via Docker on port {args.port}...")
                cmd = [
                    "docker", "run", "--rm", "-d",
                    "-p", f"{args.port}:30000",
                    "-v", os.path.expanduser("~/.cache/huggingface:/root/.cache/huggingface"),
                    "sglang-cpu:latest",
                    "bash", "-c",
                    f"source /opt/.venv/bin/activate && "
                    f"python -m sglang.launch_server "
                    f"--model {args.model} --host 0.0.0.0 --port 30000 "
                    f"--dtype bfloat16 --device cpu"
                ]
                print(f"  {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Docker start failed: {result.stderr}")
                    sys.exit(1)
                container_id = result.stdout.strip()
                print(f"  Container: {container_id[:12]}")
            else:
                print(f"Starting SGLang server on port {args.port}...")
                cmd = [
                    sys.executable, "-m", "sglang.launch_server",
                    "--model", args.model,
                    "--host", "0.0.0.0", "--port", str(args.port),
                    "--dtype", "bfloat16", "--device", "cpu",
                ]
                print(f"  {' '.join(cmd)}")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            print(f"Waiting for server at {url}...")
            if not wait_for_server(url, timeout=300):
                print("Server failed to start within 300s")
                sys.exit(1)
            print("Server ready.\n")

        server_tokens = query_server(url, args.model, PROMPT, MAX_TOKENS)
    finally:
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        if container_id:
            print(f"Stopping container {container_id[:12]}...")
            subprocess.run(["docker", "kill", container_id], capture_output=True)

    # Step 3: side-by-side comparison
    print("=== Token-by-token comparison ===")
    print(f"{'idx':>3s}  {'ref_token':>15s} {'ref_lp':>10s}  {'sgl_token':>15s} {'sgl_lp':>10s}  {'lp_diff':>10s}  match")
    print("-" * 90)

    for i in range(max(len(ref_tokens), len(server_tokens))):
        ref = ref_tokens[i] if i < len(ref_tokens) else None
        srv = server_tokens[i] if i < len(server_tokens) else None

        ref_tok = ref["token"] if ref else "---"
        ref_lp = f"{ref['logprob']:.6f}" if ref else "---"
        srv_tok = srv["token"] if srv else "---"
        srv_lp = f"{srv['logprob']:.6f}" if srv else "---"

        if ref and srv:
            diff = abs(ref["logprob"] - srv["logprob"])
            match = "OK" if ref["token"] == srv["token"] and diff < 0.1 else "MISMATCH"
            diff_str = f"{diff:.6f}"
        else:
            match = "MISSING"
            diff_str = "---"

        print(f"{i:3d}  {ref_tok!r:>15s} {ref_lp:>10s}  {srv_tok!r:>15s} {srv_lp:>10s}  {diff_str:>10s}  {match}")


if __name__ == "__main__":
    main()
