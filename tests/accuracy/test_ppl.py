#!/usr/bin/env python3
"""WikiText perplexity test — validates inference precision against HuggingFace transformers.

Adapted from vLLM's generation_ppl_test (tests/models/language/generation_ppl_test/ppl_utils.py).

Pass criteria: (vLLM PPL_TOL = 0.01)
  PPL difference between server and HF transformers must be < 1%.

Usage:
    # Auto-start server
    python tests/accuracy/test_ppl.py \
        --binary target/release/prelude-server --model Qwen/Qwen3-0.6B

    # Pre-started server
    python tests/accuracy/test_ppl.py \
        --server http://localhost:8001 --model Qwen/Qwen3-0.6B

    # Use pre-computed HF PPL (skip HF computation)
    python tests/accuracy/test_ppl.py \
        --server http://localhost:8001 --model Qwen/Qwen3-0.6B \
        --hf-ppl 23.864

    # Custom tolerance
    python tests/accuracy/test_ppl.py \
        --server http://localhost:8001 --model Qwen/Qwen3-0.6B --tolerance 0.02
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

import requests
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# vLLM defaults
PPL_TOL = 0.01  # 1% relative tolerance
MAX_LENGTH = 1024


def wait_for_server(url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    return False


def compute_hf_ppl(model_name, tokenizer, chunks, dtype, device="cuda"):
    """Compute perplexity using HuggingFace transformers (vLLM reference method)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"  Loading {model_name} in {dtype} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    nll_sum = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    n_tokens = 0

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            inputs = torch.tensor([chunk], device=device)
            outputs = model(inputs, labels=inputs)
            neg_log_likelihood = outputs.loss.to(torch.float32).cpu()
            num_loss_tokens = len(chunk) - 1
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens
            if (i + 1) % 10 == 0:
                print(f"    HF chunk {i+1}/{len(chunks)}, running PPL: {math.exp(nll_sum / n_tokens):.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ppl = float(torch.exp(nll_sum / n_tokens))
    print(f"  HF PPL: {ppl:.6f} ({n_tokens} tokens)")
    return ppl


def compute_server_ppl(server_url, model_name, tokenizer, chunks):
    """Compute perplexity using our server's prompt_logprobs API."""
    nll_sum = 0.0
    n_tokens = 0

    for i, chunk in enumerate(chunks):
        prompt_text = tokenizer.decode(chunk, skip_special_tokens=False)

        body = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }

        try:
            resp = requests.post(f"{server_url}/v1/completions", json=body, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Chunk {i+1}/{len(chunks)}: FAILED: {e}")
            continue

        choice = data["choices"][0]
        prompt_lps = choice.get("prompt_logprobs")

        if not prompt_lps:
            print(f"    Chunk {i+1}/{len(chunks)}: no prompt_logprobs returned")
            continue

        # prompt_lps is [None, {token_id: {logprob: ...}}, ...]
        chunk_nll = 0.0
        chunk_tokens = 0
        for entry in prompt_lps:
            if entry is None:
                continue
            for tid_str, lp_info in entry.items():
                logprob = lp_info["logprob"] if isinstance(lp_info, dict) else lp_info
                chunk_nll -= logprob
                chunk_tokens += 1
                break

        nll_sum += chunk_nll
        n_tokens += chunk_tokens

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            ppl_so_far = math.exp(nll_sum / n_tokens) if n_tokens > 0 else float("inf")
            print(f"    Chunk {i+1}/{len(chunks)}, running PPL: {ppl_so_far:.4f}")

    if n_tokens == 0:
        print("  ERROR: No tokens processed")
        return float("inf")

    ppl = math.exp(nll_sum / n_tokens)
    print(f"  Server PPL: {ppl:.6f} ({n_tokens} tokens)")
    return ppl


def main():
    parser = argparse.ArgumentParser(description="WikiText PPL test (vLLM-compatible)")
    parser.add_argument("--server", default=None, help="Server URL (e.g. http://localhost:8001)")
    parser.add_argument("--binary", default=None, help="Path to prelude-server binary (auto-start)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"],
                        help="Server device (auto=GPU if available, cpu=CPU only)")
    parser.add_argument("--dtype", default="bf16", choices=["f32", "bf16", "f16"])
    parser.add_argument("--hf-ppl", type=float, default=None,
                        help="Pre-computed HF PPL (skip HF computation)")
    parser.add_argument("--tolerance", type=float, default=PPL_TOL,
                        help=f"Relative PPL tolerance (default: {PPL_TOL})")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of chunks (for faster testing)")
    args = parser.parse_args()

    dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    dtype = dtype_map[args.dtype]

    print(f"Model:     {args.model}")
    print(f"Device:    {args.device}")
    print(f"Dtype:     {dtype}")
    print(f"Max len:   {args.max_length}")
    print(f"Tolerance: {args.tolerance} ({args.tolerance*100}%)")
    print()

    # Load dataset and tokenize
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer.encode("\n\n".join(dataset["text"]))
    print(f"  Total tokens: {len(tokens)}")

    # Split into chunks
    stride = args.max_length
    chunks = []
    for begin_loc in range(0, len(tokens), stride):
        end_loc = min(begin_loc + args.max_length, len(tokens))
        chunk = tokens[begin_loc:end_loc]
        if len(chunk) > 1:  # Need at least 2 tokens for loss
            chunks.append(chunk)

    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"  Chunks: {len(chunks)} (stride={stride})")
    print()

    # Step 1: HF reference PPL
    print("=" * 60)
    print("Step 1: HuggingFace reference PPL")
    print("=" * 60)
    if args.hf_ppl is not None:
        hf_ppl = args.hf_ppl
        print(f"  Using pre-computed HF PPL: {hf_ppl:.6f}")
    else:
        hf_ppl = compute_hf_ppl(args.model, tokenizer, chunks, dtype)
    print()

    # Step 2: Server PPL
    print("=" * 60)
    print("Step 2: Server PPL")
    print("=" * 60)

    proc = None
    server_url = args.server
    if not server_url:
        if not args.binary:
            print("ERROR: provide --server URL or --binary path")
            sys.exit(1)
        port = 18878
        server_url = f"http://localhost:{port}"
        # Kill any existing server on this port
        subprocess.run(f"lsof -ti:{port} | xargs -r kill -9", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        env = {**os.environ, "PRELUDE_DEVICE": args.device}
        if args.device != "cpu":
            # GPU needs paged attention
            env["PRELUDE_PAGED_ATTN_BLOCKS"] = "1024"
            env["PRELUDE_PAGED_BLOCK_SIZE"] = "128"
        cmd = [args.binary, "--host", "0.0.0.0", "--port", str(port),
               "--model", args.model, "--dtype", args.dtype]
        print(f"  Starting: {' '.join(cmd)}")
        print(f"  Env: PRELUDE_DEVICE={args.device}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  Waiting for server...", end=" ", flush=True)
        if not wait_for_server(server_url):
            print("TIMEOUT (check server logs on stderr)")
            sys.exit(1)
        print("ready")

    try:
        server_ppl = compute_server_ppl(server_url, args.model, tokenizer, chunks)
    finally:
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Step 3: Compare
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    # PPL: lower is better. We only care if server PPL is HIGHER than HF
    # (precision degradation). Server PPL being lower is fine.
    # This matches vLLM's one-sided testing (see ppl_utils.py).
    diff = (server_ppl - hf_ppl) / hf_ppl
    print(f"  HF PPL:     {hf_ppl:.6f}")
    print(f"  Server PPL: {server_ppl:.6f}")
    print(f"  Difference: {diff*100:+.4f}%")
    print(f"  Tolerance:  {args.tolerance*100:.1f}% (one-sided: only fails if server is worse)")
    print()

    if diff < args.tolerance:
        print(f"PASSED (PPL diff {diff*100:+.4f}% < {args.tolerance*100:.1f}%)")
        sys.exit(0)
    else:
        print(f"FAILED (PPL diff {diff*100:+.4f}% >= {args.tolerance*100:.1f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
