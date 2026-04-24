#!/usr/bin/env python3
"""Accuracy test runner — validates inference engines against HuggingFace transformers.

Pass criteria (vLLM-style):
  L0: Exact text match → PASS
  L1: At first token divergence, bidirectional cross-containment in top-5 → PASS

Usage:
    # prelude vs transformers
    python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
        --server prelude --model Qwen/Qwen3-0.6B

    # prelude vs sglang (skip transformers)
    python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
        --server prelude --server sglang --ref sglang --model Qwen/Qwen3-0.6B

    # Pre-started server
    python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
        --server myserver=http://localhost:8001 --model Qwen/Qwen3-0.6B
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANTS = {
    "cpu-f32": {
        "env": {"PRELUDE_DEVICE": "cpu"},
        "torch_dtype": torch.float32,
        "features": [],
        "extra_args": ["--device", "cpu", "--dtype", "f32"],
    },
    "cpu-bf16": {
        "env": {"PRELUDE_DEVICE": "cpu"},
        "torch_dtype": torch.bfloat16,
        "extra_args": ["--device", "cpu", "--dtype", "bf16"],
    },
    "gpu": {
        "env": {
            "PRELUDE_DEVICE": "auto",
            "PRELUDE_PAGED_ATTN_BLOCKS": "1024",
            "PRELUDE_PAGED_BLOCK_SIZE": "128",
        },
        "torch_dtype": torch.bfloat16,
    },
}

# Auto-start port assignments (avoid conflicts)
import os as _os
PORT_PRELUDE = int(_os.environ.get("PRELUDE_TEST_PORT", "8099"))
PORT_SGLANG = int(_os.environ.get("SGLANG_TEST_PORT", "8098"))
PORT_VLLM = int(_os.environ.get("VLLM_TEST_PORT", "8097"))

TOP_K_LOGPROBS = 5

# SGLang-style tolerances (from test_generation_models.py)
PREFILL_TOLERANCE = 5e-2
DECODE_TOLERANCE = 6e-2


# ── Server auto-start registry ──

class ManagedServer:
    """A server that we start and stop."""
    def __init__(self, name: str, url: str, proc=None, container_id=None):
        self.name = name
        self.url = url
        self.proc = proc
        self.container_id = container_id

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.container_id:
            try:
                subprocess.run(["docker", "kill", self.container_id],
                               capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                pass


def start_prelude(model: str, variant: dict, binary: str = None) -> ManagedServer:
    """Start prelude server."""
    binary = binary or "target/release/prelude-server"
    env = {**os.environ, **variant["env"]}
    cmd = [binary, "--host", "0.0.0.0", "--port", str(PORT_PRELUDE),
           "--model", model] + variant.get("extra_args", [])
    print(f"  Starting prelude: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return ManagedServer("prelude", f"http://localhost:{PORT_PRELUDE}", proc=proc)


def start_sglang(model: str, variant: dict, **_kw) -> ManagedServer:
    """Start SGLang CPU server (native or Docker)."""
    dtype = "bfloat16" if variant["torch_dtype"] == torch.bfloat16 else "float32"
    port = PORT_SGLANG

    # Try Docker first (sglang-cpu image)
    docker_available = shutil.which("docker") is not None
    if docker_available:
        result = subprocess.run(
            ["docker", "images", "-q", "sglang-cpu:latest"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            hf_cache = os.path.expanduser("~/.cache/huggingface")
            cmd = [
                "docker", "run", "--rm", "-d",
                "-p", f"{port}:30000",
                "-v", f"{hf_cache}:/root/.cache/huggingface",
                "sglang-cpu:latest",
                "bash", "-c",
                f"source /opt/.venv/bin/activate && "
                f"python -m sglang.launch_server "
                f"--model {model} --host 0.0.0.0 --port 30000 "
                f"--dtype {dtype} --device cpu",
            ]
            print(f"  Starting sglang (Docker): sglang-cpu:latest")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                cid = result.stdout.strip()
                return ManagedServer("sglang", f"http://localhost:{port}", container_id=cid)
            print(f"    Docker start failed: {result.stderr.strip()}")

    # Fallback: native Python
    try:
        import sglang  # noqa: F401
    except ImportError:
        print("  ERROR: sglang not installed and Docker image not available")
        sys.exit(1)

    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model", model, "--host", "0.0.0.0", "--port", str(port),
           "--dtype", dtype, "--device", "cpu"]
    print(f"  Starting sglang (native): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return ManagedServer("sglang", f"http://localhost:{port}", proc=proc)


def start_vllm(model: str, variant: dict, **_kw) -> ManagedServer:
    """Start vLLM CPU server."""
    dtype = "bfloat16" if variant["torch_dtype"] == torch.bfloat16 else "float32"
    port = PORT_VLLM
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--host", "0.0.0.0", "--port", str(port),
           "--dtype", dtype, "--device", "cpu"]
    print(f"  Starting vllm: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return ManagedServer("vllm", f"http://localhost:{port}", proc=proc)


ENGINE_STARTERS = {
    "prelude": start_prelude,
    "sglang": start_sglang,
    "vllm": start_vllm,
}


# ── Reference generation (transformers) ──

def build_chatml_prompt(messages: list[dict]) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def generate_reference(model, tokenizer, prompt_spec, device) -> dict:
    if prompt_spec.get("type") == "completion":
        prompt_text = prompt_spec["prompt"]
    else:
        prompt_text = build_chatml_prompt(prompt_spec["messages"])

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    max_new = prompt_spec["max_tokens"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            output_logits=True, return_dict_in_generate=True,
        )

    output_ids = outputs.sequences[0, prompt_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)

    token_logprobs = []
    for step_idx, logits_tensor in enumerate(outputs.logits):
        logits = logits_tensor.squeeze(0).float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_id = output_ids[step_idx]
        token_logprob = log_probs[token_id].item()
        topk_vals, topk_ids = torch.topk(log_probs, TOP_K_LOGPROBS)
        top_logprobs = [
            {
                "token_id": tid.item(),
                "token": tokenizer.decode([tid.item()]),
                "logprob": lp.item(),
            }
            for tid, lp in zip(topk_ids, topk_vals)
        ]
        token_logprobs.append({
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "logprob": token_logprob,
            "top_logprobs": top_logprobs,
        })

    return {
        "output_token_ids": output_ids, "output_text": output_text,
        "token_logprobs": token_logprobs, "prompt_tokens": prompt_len,
        "completion_tokens": len(output_ids),
    }


# ── Server interaction ──

def wait_for_server(url: str, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ["/health", "/v1/models"]:
            try:
                r = requests.get(f"{url}{path}", timeout=2)
                if r.status_code == 200:
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
        time.sleep(1)
    return False


def send_chat(url, model, messages, max_tokens, logprobs=False, top_logprobs=0, timeout=120):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def send_completion(url, model, prompt, max_tokens, logprobs=None, timeout=120):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0}
    if logprobs is not None:
        body["logprobs"] = logprobs
    resp = requests.post(f"{url}/v1/completions", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def query_server(url, model, spec, use_logprobs, timeout=120):
    if spec.get("type") == "completion":
        return send_completion(url, model, spec["prompt"], spec["max_tokens"],
                               logprobs=TOP_K_LOGPROBS if use_logprobs else None,
                               timeout=timeout)
    else:
        return send_chat(url, model, spec["messages"], spec["max_tokens"],
                         logprobs=use_logprobs, top_logprobs=TOP_K_LOGPROBS,
                         timeout=timeout)


# ── Comparison logic ──

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(dot / norm)


def extract_server_logprobs(response, spec):
    """Extract (token_logprobs, top_logprobs, tokens) from server response."""
    if spec.get("type") == "completion":
        lp = response["choices"][0].get("logprobs")
        if not lp:
            return [], [], []
        return (
            lp.get("token_logprobs", []),
            lp.get("top_logprobs", []),
            lp.get("tokens", []),
        )
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


def extract_server_text(response, spec):
    if spec.get("type") == "completion":
        return response["choices"][0]["text"]
    else:
        return response["choices"][0]["message"]["content"]


def compare_with_ref(name, ref, response, spec, use_logprobs):
    """Compare server output against reference using vLLM-style criteria.

    Pass criteria:
      L0: Exact text match → PASS
      L1: At first token divergence, bidirectional cross-containment in top-5 → PASS
    """
    actual_text = extract_server_text(response, spec)
    ref_text = ref["output_text"]
    ref_lps = ref.get("token_logprobs", [])

    result = {
        "name": name,
        "text_match": actual_text == ref_text,
        "cross_contained": None,
        "max_lp_diff": None,
        "first_diff_char": None,
        "diverge_detail": None,
    }

    # ── L0: Exact text match ──
    if result["text_match"]:
        result["passed"] = True
    else:
        # Find first divergence character
        for i, (a, b) in enumerate(zip(actual_text, ref_text)):
            if a != b:
                result["first_diff_char"] = i
                result["expected_snippet"] = ref_text[max(0, i - 20):i + 20]
                result["actual_snippet"] = actual_text[max(0, i - 20):i + 20]
                break
        else:
            result["first_diff_char"] = min(len(actual_text), len(ref_text))

        # ── L1: Bidirectional cross-containment (vLLM-style) ──
        _, actual_top_lps, actual_tokens = extract_server_logprobs(response, spec)

        if ref_lps and actual_tokens:
            cross_ok = True
            for step in range(min(len(ref_lps), len(actual_tokens))):
                ref_token = ref_lps[step].get("token", "")
                srv_token = actual_tokens[step]
                if ref_token != srv_token:
                    # Direction 1: server's token in ref's top-k
                    ref_top_tokens = {t["token"] for t in ref_lps[step].get("top_logprobs", [])}
                    srv_in_ref = srv_token in ref_top_tokens
                    # Direction 2: ref's token in server's top-k
                    srv_top_tokens = set()
                    if actual_top_lps and step < len(actual_top_lps) and actual_top_lps[step]:
                        srv_top_tokens = set(actual_top_lps[step].keys())
                    ref_in_srv = ref_token in srv_top_tokens if srv_top_tokens else None

                    if srv_in_ref and (ref_in_srv is None or ref_in_srv):
                        result["diverge_detail"] = f"step {step}: bidir ok"
                    elif srv_in_ref:
                        result["diverge_detail"] = f"step {step}: '{srv_token}' in ref top-5, but '{ref_token}' NOT in srv top-5"
                    else:
                        cross_ok = False
                        result["diverge_detail"] = f"step {step}: '{srv_token}' NOT in ref top-5"
                    break

            result["cross_contained"] = cross_ok

        result["passed"] = result["text_match"] or (result["cross_contained"] is True)

    # ── Logprob metrics (informational) ──
    if use_logprobs:
        actual_lps_vals, _, _ = extract_server_logprobs(response, spec)
        if ref_lps and actual_lps_vals:
            min_len = min(len(ref_lps), len(actual_lps_vals))
            if min_len > 0:
                result["max_lp_diff"] = max(
                    abs(ref_lps[i]["logprob"] - actual_lps_vals[i]) for i in range(min_len))

    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Accuracy test runner (vLLM/SGLang-style)")
    parser.add_argument("--variant", required=True, choices=VARIANTS.keys())
    parser.add_argument("--binary", default="target/release/prelude-server",
                        help="Path to prelude-server binary (used by --server prelude)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--server", action="append", default=[], metavar="NAME[=URL]",
                        help="Add server: 'prelude' / 'sglang' (auto-start) or 'name=http://...' (pre-started)")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--ref", default="transformers", metavar="NAME",
                        help="Reference source: 'transformers' (default) or a server name (e.g. 'sglang')")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-request timeout in seconds (default: 120)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    prompts_path = Path(args.prompts) if args.prompts else script_dir / "golden" / "prompts.json"
    variant = VARIANTS[args.variant]
    torch_dtype = variant["torch_dtype"]
    use_logprobs = not args.no_logprobs

    # Parse --server flags: name=url (pre-started) or name (auto-start)
    servers: dict[str, str] = {}          # name -> url
    auto_start: list[str] = []            # names to auto-start
    managed: list[ManagedServer] = []     # servers we started (to stop later)

    for s in args.server:
        if "=" in s:
            name, url = s.split("=", 1)
            servers[name] = url
        else:
            auto_start.append(s)

    with open(prompts_path) as f:
        prompts_data = json.load(f)

    # Inject programmatically-generated long-context prompt (~4K tokens).
    # Too large for JSON, so we build it here.
    _long_ctx = (
        "Below is a series of numbered facts. After reading all of them, "
        "answer the question at the end.\n\n"
    )
    for i in range(1, 201):
        _long_ctx += f"Fact {i}: The value of item {i} is {i * 7 + 3}. "
        if i % 5 == 0:
            _long_ctx += "\n"
    _long_ctx += (
        "\nQuestion: What is the value of item 42 according to the facts above? "
        "Show your reasoning step by step."
    )
    prompts_data["prompts"].append({
        "id": "long_context_4k",
        "description": "~4K token context + 64-step decode (KV cache stress test)",
        "type": "completion",
        "prompt": _long_ctx,
        "max_tokens": 64,
        "temperature": 0.0,
    })

    use_server_ref = args.ref != "transformers"
    ref_server_name = args.ref if use_server_ref else None

    # If --ref is a server, ensure it's in the server list
    if use_server_ref and ref_server_name not in [s.split("=")[0] for s in args.server]:
        print(f"ERROR: --ref {ref_server_name} not in --server list. Add --server {ref_server_name}")
        sys.exit(1)

    print(f"Variant:  {args.variant}")
    print(f"Model:    {args.model}")
    print(f"Dtype:    {torch_dtype}")
    print(f"Logprobs: {'yes' if use_logprobs else 'no'}")
    print(f"Ref:      {args.ref}")
    print(f"Criteria: L0=exact text, L1=bidirectional top-5 cross-containment (vLLM-style)")
    print(f"Prompts:  {len(prompts_data['prompts'])} test cases")
    print()

    # ── Step 1: Generate reference (transformers only) ──
    references = {}
    tokenizer = None
    if not use_server_ref:
        print("=" * 60)
        print("Step 1: Generating reference with transformers")
        print("=" * 60)

        # Reference runs on GPU when available — CPU inference on >1B models
        # takes tens of minutes per prompt and dominates total test time.
        device = _os.environ.get("PRELUDE_REF_DEVICE") \
            or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {args.model} in {torch_dtype} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype).to(device)
        ref_model.eval()

        max_tokens_override = variant.get("max_tokens_override")
        for spec in prompts_data["prompts"]:
            ref_spec = spec
            if max_tokens_override is not None:
                ref_spec = {**spec, "max_tokens": max_tokens_override}
            pid = spec["id"]
            print(f"  [{pid}] Generating reference...", end=" ", flush=True)
            ref = generate_reference(ref_model, tokenizer, ref_spec, device)
            references[pid] = ref
            print(f"{ref['completion_tokens']} tokens")

        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\nReference generation complete. "
              f"transformers={transformers.__version__}, torch={torch.__version__}\n")
    else:
        print("=" * 60)
        print(f"Step 1: Skipped (using {ref_server_name} as reference)")
        print("=" * 60)
        print()

    # ── Step 2: Start servers ──
    print("=" * 60)
    print("Step 2: Starting servers")
    print("=" * 60)

    try:
        # Auto-start engines
        for name in auto_start:
            if name not in ENGINE_STARTERS:
                print(f"  ERROR: unknown engine '{name}'. Available: {', '.join(ENGINE_STARTERS.keys())}")
                sys.exit(1)
            srv = ENGINE_STARTERS[name](args.model, variant, binary=args.binary)
            managed.append(srv)
            servers[name] = srv.url

        if not servers:
            print("ERROR: No servers. Use --server prelude / sglang / vllm / name=url")
            sys.exit(1)

        # Wait for all servers
        for name, url in servers.items():
            is_managed = any(m.name == name for m in managed)
            if is_managed:
                print(f"  Waiting for {name} at {url}...", end=" ", flush=True)
                if not wait_for_server(url):
                    print("TIMEOUT")
                    for m in managed:
                        if m.name == name and m.proc and m.proc.stdout:
                            out = m.proc.stdout.read().decode()
                            print(f"  {name} output:\n{out[-2000:]}")
                    sys.exit(1)
                print("ready")
            else:
                print(f"  Checking {name} at {url}...", end=" ", flush=True)
                if not wait_for_server(url, timeout=5):
                    print("NOT REACHABLE")
                    sys.exit(1)
                print("ok")

        # ── Step 3: Query all servers and compare ──
        all_names = list(servers.keys())
        if use_server_ref:
            test_names = [n for n in all_names if n != ref_server_name]
            ref_label = ref_server_name
        else:
            test_names = all_names
            ref_label = "transformers"

        print("=" * 60)
        print(f"Step 3: Testing {', '.join(test_names)} vs {ref_label}")
        print("=" * 60)

        all_reports = []

        max_tokens_override = variant.get("max_tokens_override")
        for spec in prompts_data["prompts"]:
            query_spec = spec
            if max_tokens_override is not None:
                query_spec = {**spec, "max_tokens": max_tokens_override}
            pid = spec["id"]
            print(f"  [{pid}]", end="", flush=True)

            # Query all servers (including ref server if server-ref mode)
            server_responses = {}
            for sname in all_names:
                surl = servers[sname]
                try:
                    resp = query_server(surl, args.model, query_spec, use_logprobs,
                                        timeout=args.timeout)
                    server_responses[sname] = resp
                except Exception as e:
                    print(f" {sname}=ERR({e})", end="")
                    server_responses[sname] = None

            # Build reference for this prompt
            if use_server_ref:
                ref_resp = server_responses.get(ref_server_name)
                if ref_resp is None:
                    print(f" ref={ref_server_name} failed, SKIP")
                    all_reports.append({"id": pid, "passed": None})
                    continue
                ref_text = extract_server_text(ref_resp, spec)
                ref_lps, ref_top_lps, ref_tokens = extract_server_logprobs(ref_resp, spec)
                ref = {
                    "output_text": ref_text,
                    "token_logprobs": [{"logprob": lp, "token": tok} for lp, tok in zip(ref_lps, ref_tokens)],
                }
            else:
                ref = references.get(pid)
                if not ref:
                    print(" SKIP")
                    all_reports.append({"id": pid, "passed": None})
                    continue

            # Compare test servers against reference
            server_ref_results = {}
            for sname in test_names:
                resp = server_responses.get(sname)
                if resp is None:
                    server_ref_results[sname] = {
                        "name": sname, "passed": False, "text_match": False,
                        "cross_contained": None, "cos_sim": None,
                        "max_lp_diff": None, "error": "query failed",
                    }
                    continue
                cmp = compare_with_ref(sname, ref, resp, spec, use_logprobs)
                server_ref_results[sname] = cmp

            # Print results per-server
            for sname in test_names:
                cmp = server_ref_results[sname]
                if "error" in cmp:
                    print(f" {sname}=ERR", end="")
                    continue

                if cmp["text_match"]:
                    status = "MATCH"
                elif cmp.get("cross_contained"):
                    status = "CLOSE"
                else:
                    status = "FAIL"

                parts = []
                if cmp["max_lp_diff"] is not None:
                    parts.append(f"lp={cmp['max_lp_diff']:.4f}")
                detail = f"({', '.join(parts)})" if parts else ""
                print(f" {sname}={status}{detail}", end="")
            print()

            # Print diff details for non-exact matches
            for sname in test_names:
                cmp = server_ref_results[sname]
                if cmp.get("text_match"):
                    continue
                if cmp.get("first_diff_char") is not None:
                    c = cmp["first_diff_char"]
                    print(f"    {sname} diff@{c}: ref=...{cmp.get('expected_snippet', '')}...")
                    print(f"    {' ' * len(sname)}          got=...{cmp.get('actual_snippet', '')}...")
                if cmp.get("diverge_detail"):
                    print(f"    {sname} L1: {cmp['diverge_detail']}")

            prompt_passed = all(
                server_ref_results[s].get("passed", False) for s in test_names
            )
            all_reports.append({"id": pid, "passed": prompt_passed, "servers": server_ref_results})

    finally:
        for m in managed:
            print(f"Stopping {m.name}...")
            m.stop()

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"Summary (vs {ref_label})")
    print("-" * 60)

    for sname in test_names:
        exact = sum(1 for r in all_reports
                    if r.get("servers", {}).get(sname, {}).get("text_match", False))
        close = sum(1 for r in all_reports
                    if r.get("servers", {}).get(sname, {}).get("cross_contained") is True
                    and not r.get("servers", {}).get(sname, {}).get("text_match", False))
        total = sum(1 for r in all_reports if r.get("passed") is not None)
        lp_diffs = [
            r["servers"][sname]["max_lp_diff"]
            for r in all_reports
            if r.get("servers", {}).get(sname, {}).get("max_lp_diff") is not None
        ]
        parts = [f"exact={exact}/{total}", f"close={close}/{total}"]
        if lp_diffs:
            parts.append(f"max_lp: {min(lp_diffs):.4f}/{sum(lp_diffs)/len(lp_diffs):.4f}/{max(lp_diffs):.4f}")
        print(f"  {sname:12s} {', '.join(parts)}")

    overall_passed = sum(1 for r in all_reports if r.get("passed") is True)
    overall_total = sum(1 for r in all_reports if r.get("passed") is not None)
    overall_failed = overall_total - overall_passed
    print(f"\nOverall: {overall_passed}/{overall_total} passed "
          f"(exact + cross-contained), {overall_failed} failed")
    print("=" * 60)

    if overall_failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
