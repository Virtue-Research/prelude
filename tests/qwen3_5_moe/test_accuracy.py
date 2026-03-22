#!/usr/bin/env python3
"""Accuracy test for Qwen3.5 MoE hybrid model (35B-A3B).

Compares Prelude server output against HuggingFace transformers golden reference.

Pass criteria (same as run_accuracy_test.py):
  L0: Exact text match → PASS
  L1: At first token divergence, server's token is in ref's top-5 → PASS (soft)
  L2: Logprob cosine similarity (informational)

Usage:
    # Step 1: Generate golden reference (one-time)
    python tests/qwen3_5_moe/test_accuracy.py --generate-golden \
        --model /path/to/models/qwen3.5-35b-a3b

    # Step 2: Test against pre-started server
    python tests/qwen3_5_moe/test_accuracy.py --server-url http://localhost:8001

    # Step 2 (alt): Auto-start server and test
    python tests/qwen3_5_moe/test_accuracy.py \
        --model /path/to/models/qwen3.5-35b-a3b

    # Combined: generate golden + test (slow but one-shot)
    python tests/qwen3_5_moe/test_accuracy.py --generate-golden \
        --model /path/to/models/qwen3.5-35b-a3b \
        --server-url http://localhost:8001
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

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_BINARY = "target/release/prelude-server"
DEFAULT_PORT = 8099
HEALTH_TIMEOUT = 600  # 10 min — 35B MoE model is large
TOP_K_LOGPROBS = 5


# ── Golden reference generation ──

def build_chatml_prompt(messages: list[dict]) -> str:
    """Build ChatML prompt matching prelude server's chat_to_engine_request()."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def resolve_dtype(dtype_str: str):
    """Resolve dtype string to torch dtype."""
    import torch
    if dtype_str == "f32":
        return torch.float32, "f32"
    elif dtype_str == "bf16":
        return torch.bfloat16, "bf16"
    elif dtype_str == "auto":
        return "auto", "bf16"  # Most modern models default to BF16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def generate_golden(model_path: str, prompts: list[dict], output_path: Path,
                    dtype_str: str = "auto"):
    """Generate golden references using HuggingFace transformers."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype, dtype_label = resolve_dtype(dtype_str)
    print(f"Loading {model_path} in {dtype_label} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype,
        # Workaround: torch 2.10 grouped_mm has alignment bug (data_ptr not 16-byte aligned).
        # See: https://github.com/pytorch/pytorch/issues/154557
        #      https://github.com/pytorch/pytorch/issues/159378
        # Use per-expert loop instead.
        experts_implementation="eager",
    )
    model.eval()
    actual_dtype = next(model.parameters()).dtype
    print(f"Model loaded in {time.time() - t0:.1f}s (dtype={actual_dtype})")

    results = {}
    for spec in prompts:
        pid = spec["id"]
        if spec.get("type") == "completion":
            prompt_text = spec["prompt"]
        else:
            prompt_text = build_chatml_prompt(spec["messages"])

        inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        print(f"  [{pid}] generating (prompt={prompt_len} tokens, max_new={spec['max_tokens']})...",
              end=" ", flush=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=spec["max_tokens"],
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )

        output_ids = outputs.sequences[0, prompt_len:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=False)

        # Extract per-step logprobs for L1/L2 comparison
        token_logprobs = []
        for step_idx, logits_tensor in enumerate(outputs.logits):
            logits = logits_tensor.squeeze(0).float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_id = output_ids[step_idx]
            topk_vals, topk_ids = torch.topk(log_probs, TOP_K_LOGPROBS)
            token_logprobs.append({
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "logprob": log_probs[token_id].item(),
                "top_logprobs": [
                    {
                        "token_id": tid.item(),
                        "token": tokenizer.decode([tid.item()]),
                        "logprob": lp.item(),
                    }
                    for tid, lp in zip(topk_ids, topk_vals)
                ],
            })

        results[pid] = {
            "output_token_ids": output_ids,
            "output_text": output_text,
            "token_logprobs": token_logprobs,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(output_ids),
        }
        print(f"{len(output_ids)} tokens: {output_text[:60]!r}")

    golden = {
        "model": DEFAULT_MODEL_ID,
        "dtype": str(actual_dtype),
        "generator": "transformers",
        "generator_version": transformers.__version__,
        "torch_version": torch.__version__,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nGolden written: {output_path}")


# ── Server management ──

def start_server(binary: str, model: str, port: int) -> subprocess.Popen:
    """Start prelude-server with Qwen3.5 MoE model."""
    env = os.environ.copy()
    env.setdefault("PRELUDE_DEVICE", "cpu")
    env.setdefault("RUST_LOG", "prelude_core=info")

    cmd = [
        binary,
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def wait_for_health(url: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    """Wait for server to become healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


# ── Server queries ──

def query_server(url: str, model_id: str, spec: dict, use_logprobs: bool) -> dict:
    """Query server with a prompt spec, return raw response."""
    if spec.get("type") == "completion":
        body = {
            "model": model_id, "prompt": spec["prompt"],
            "max_tokens": spec["max_tokens"], "temperature": 0.0,
        }
        if use_logprobs:
            body["logprobs"] = TOP_K_LOGPROBS
        resp = requests.post(f"{url}/v1/completions", json=body, timeout=300)
    else:
        body = {
            "model": model_id, "messages": spec["messages"],
            "max_tokens": spec["max_tokens"], "temperature": 0.0, "stream": False,
        }
        if use_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = TOP_K_LOGPROBS
        resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()


def extract_text(response: dict, spec: dict) -> str:
    if spec.get("type") == "completion":
        return response["choices"][0]["text"]
    return response["choices"][0]["message"]["content"]


def extract_logprobs(response: dict, spec: dict):
    """Extract (token_logprobs, top_logprobs_list, tokens) from server response."""
    if spec.get("type") == "completion":
        lp = response["choices"][0].get("logprobs")
        if not lp:
            return [], [], []
        return lp.get("token_logprobs", []), lp.get("top_logprobs", []), lp.get("tokens", [])
    else:
        lp = response["choices"][0].get("logprobs")
        if not lp:
            return [], [], []
        content = lp.get("content", [])
        return (
            [item["logprob"] for item in content],
            [{e["token"]: e["logprob"] for e in item.get("top_logprobs", [])} for item in content],
            [item["token"] for item in content],
        )


# ── Comparison logic ──

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(dot / norm)


STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]


def strip_stop_tokens(text: str) -> str:
    """Strip trailing stop/special tokens that the server removes from responses."""
    while True:
        text = text.rstrip("\n")
        found = False
        for tok in STOP_TOKENS:
            if text.endswith(tok):
                text = text[: -len(tok)]
                found = True
        if not found:
            break
    return text


def compare(ref: dict, response: dict, spec: dict, use_logprobs: bool) -> dict:
    """Compare server response against golden reference."""
    actual_text = extract_text(response, spec)
    ref_text = ref["output_text"]
    ref_lps = ref.get("token_logprobs", [])

    # Strip stop tokens from golden text — server strips them per API convention.
    ref_text = strip_stop_tokens(ref_text)
    actual_text = strip_stop_tokens(actual_text)

    # Truncate golden logprobs to match stripped text length (remove stop token entries).
    if ref_lps:
        stripped_ids = []
        for lp in ref_lps:
            tok = lp.get("token", "")
            if tok.strip() in STOP_TOKENS:
                break
            stripped_ids.append(lp)
        ref_lps = stripped_ids

    result = {
        "text_match": actual_text == ref_text,
        "cross_contained": None,
        "cos_sim": None,
        "max_lp_diff": None,
        "first_diff_char": None,
        "ref_text": ref_text,
        "actual_text": actual_text,
    }

    # L0: Exact text match
    if result["text_match"]:
        result["passed"] = True
        return result

    # Find first diff char
    for i, (a, b) in enumerate(zip(actual_text, ref_text)):
        if a != b:
            result["first_diff_char"] = i
            break
    else:
        result["first_diff_char"] = min(len(actual_text), len(ref_text))

    # L1: Bidirectional cross-containment
    actual_lps, actual_top_lps, actual_tokens = extract_logprobs(response, spec)

    if ref_lps and actual_tokens:
        cross_ok = True
        for step in range(min(len(ref_lps), len(actual_tokens))):
            ref_token = ref_lps[step].get("token", "")
            srv_token = actual_tokens[step]
            if ref_token != srv_token:
                # Direction 1: server's token in ref's top-k
                ref_top_tokens = {t["token"] for t in ref_lps[step].get("top_logprobs", [])}
                srv_in_ref = srv_token in ref_top_tokens
                # Direction 2: ref's token in server's top-k (dict keys are token strings)
                srv_top_tokens = set()
                if actual_top_lps and step < len(actual_top_lps) and actual_top_lps[step]:
                    srv_top_tokens = set(actual_top_lps[step].keys())
                ref_in_srv = ref_token in srv_top_tokens if srv_top_tokens else None
                if srv_in_ref and (ref_in_srv is None or ref_in_srv):
                    detail = f"step {step}: bidir ok"
                    if ref_in_srv is None:
                        detail += " (srv top-k unavailable, fwd only)"
                    result["diverge_detail"] = detail
                elif srv_in_ref:
                    result["diverge_detail"] = f"step {step}: '{srv_token}' in ref top-5, but '{ref_token}' NOT in srv top-5"
                else:
                    cross_ok = False
                    result["diverge_detail"] = f"step {step}: '{srv_token}' NOT in ref top-5"
                break
        result["cross_contained"] = cross_ok

    result["passed"] = result["text_match"] or (result["cross_contained"] is True)

    # L2: Logprob metrics (informational)
    if use_logprobs and ref_lps and actual_lps:
        min_len = min(len(ref_lps), len(actual_lps))
        if min_len > 0:
            ref_vec = [ref_lps[i]["logprob"] for i in range(min_len)]
            act_vec = list(actual_lps[:min_len])
            result["cos_sim"] = cosine_similarity(ref_vec, act_vec)
            result["max_lp_diff"] = max(abs(r - a) for r, a in zip(ref_vec, act_vec))

    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 accuracy test")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help="HF repo ID or local model directory")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="prelude-server binary")
    parser.add_argument("--server-url", help="Pre-started server URL (skip auto-start)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--golden-path", help="Path to golden JSON (default: auto)")
    parser.add_argument("--generate-golden", action="store_true",
                        help="Generate golden reference from HF transformers")
    parser.add_argument("--no-logprobs", action="store_true",
                        help="Skip logprob comparison")
    parser.add_argument("--dtype", default="auto", choices=["auto", "f32", "bf16"],
                        help="HF reference dtype (default: auto → bf16 for modern models)")
    args = parser.parse_args()

    _, dtype_label = resolve_dtype(args.dtype)
    prompts_path = SCRIPT_DIR / "prompts.json"
    golden_path = Path(args.golden_path) if args.golden_path else SCRIPT_DIR / "golden" / f"qwen3.5-35b-a3b-{dtype_label}.json"

    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts = prompts_data["prompts"]
    use_logprobs = not args.no_logprobs

    print(f"Model:    {args.model}")
    print(f"Prompts:  {len(prompts)} test cases")
    print(f"Logprobs: {'yes' if use_logprobs else 'no'}")
    print(f"Criteria: L0=exact text, L1=top-5 cross-containment")
    print()

    # ── Step 1: Golden reference ──
    if args.generate_golden:
        print("=" * 60)
        print("Generating golden reference from HF transformers")
        print("=" * 60)
        generate_golden(args.model, prompts, golden_path, args.dtype)
        print()

    # If only generating golden (no server test), exit
    if args.generate_golden and not args.server_url:
        print("Golden generated. Re-run with --server-url to test.")
        sys.exit(0)

    # Load golden
    if not golden_path.exists():
        print(f"Golden not found at {golden_path}, generating...")
        generate_golden(args.model, prompts, golden_path, args.dtype)

    with open(golden_path) as f:
        golden = json.load(f)
    print(f"Golden:   {golden_path}")
    print(f"          generated={golden.get('generated_at', '?')}, "
          f"transformers={golden.get('generator_version', '?')}")
    print()

    # ── Step 2: Server ──
    proc = None
    url = args.server_url

    if not url:
        if not os.path.exists(args.binary):
            print(f"ERROR: binary not found: {args.binary}")
            sys.exit(1)
        proc = start_server(args.binary, args.model, args.port)
        url = f"http://localhost:{args.port}"

    try:
        if proc:
            print(f"Waiting for server at {url} (timeout={HEALTH_TIMEOUT}s)...", flush=True)
            if not wait_for_health(url):
                out = proc.stdout.read().decode() if proc.stdout else ""
                print(f"Server failed to start. Output:\n{out[-3000:]}")
                sys.exit(1)
            print("Server ready.\n")
        else:
            print(f"Checking server at {url}...", end=" ", flush=True)
            if not wait_for_health(url, timeout=5):
                print("NOT REACHABLE")
                sys.exit(1)
            print("ok\n")

        # ── Step 3: Test ──
        print("=" * 60)
        print("Testing prelude vs transformers golden")
        print("=" * 60)

        results = []
        for spec in prompts:
            pid = spec["id"]
            ref = golden["results"].get(pid)
            if not ref:
                print(f"  [{pid}] SKIP (no golden)")
                results.append({"id": pid, "passed": None})
                continue

            print(f"  [{pid}]", end="", flush=True)

            try:
                response = query_server(url, args.model, spec, use_logprobs)
            except Exception as e:
                print(f" ERR({e})")
                results.append({"id": pid, "passed": False, "error": str(e)})
                continue

            cmp = compare(ref, response, spec, use_logprobs)

            # Status label
            if cmp["text_match"]:
                status = "MATCH"
            elif cmp.get("cross_contained"):
                status = "CLOSE"
            else:
                status = "FAIL"

            # Metrics
            parts = []
            if cmp["cos_sim"] is not None:
                parts.append(f"cos={cmp['cos_sim']:.4f}")
            if cmp["max_lp_diff"] is not None:
                parts.append(f"lp={cmp['max_lp_diff']:.4f}")
            detail = f" ({', '.join(parts)})" if parts else ""
            print(f" {status}{detail}")

            # Show diff details for non-exact matches
            if not cmp["text_match"]:
                c = cmp.get("first_diff_char", 0)
                print(f"    ref: {cmp['ref_text'][:80]!r}")
                print(f"    got: {cmp['actual_text'][:80]!r}")
                if c is not None:
                    print(f"    diff@char {c}")
                if cmp.get("diverge_detail"):
                    print(f"    L1: {cmp['diverge_detail']}")

            results.append({"id": pid, "passed": cmp["passed"], **cmp})

    finally:
        if proc:
            print("\nStopping server...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary")
    print("-" * 60)

    tested = [r for r in results if r.get("passed") is not None]
    passed = sum(1 for r in tested if r["passed"])
    exact = sum(1 for r in tested if r.get("text_match", False))
    close = sum(1 for r in tested if r.get("cross_contained") is True and not r.get("text_match", False))
    failed = len(tested) - passed

    cos_sims = [r["cos_sim"] for r in tested if r.get("cos_sim") is not None]

    print(f"  Total:  {len(tested)} prompts")
    print(f"  Exact:  {exact}")
    print(f"  Close:  {close} (token in ref top-5)")
    print(f"  Failed: {failed}")
    if cos_sims:
        print(f"  Cosine: min={min(cos_sims):.4f}, avg={sum(cos_sims)/len(cos_sims):.4f}, max={max(cos_sims):.4f}")

    print(f"\nResult: {passed}/{len(tested)} passed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
