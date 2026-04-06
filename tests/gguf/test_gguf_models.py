#!/usr/bin/env python3
"""Test universal GGUF support by downloading small GGUF models and running inference."""

import json
import os
import signal
import subprocess
import sys
import time

import requests
from huggingface_hub import hf_hub_download, list_repo_files

# Small GGUF models to test (repo, filename_pattern, model_id for tokenizer, arch)
MODELS = [
    {
        "name": "qwen3",
        "repo": "Qwen/Qwen3-0.6B-GGUF",
        "tokenizer_repo": "Qwen/Qwen3-0.6B",
        "arch": "qwen3",
    },
    {
        "name": "qwen2",
        "repo": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "tokenizer_repo": "Qwen/Qwen2-0.5B-Instruct",
        "arch": "qwen2",
    },
    {
        "name": "llama",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "tokenizer_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "arch": "llama",
    },
    {
        "name": "gemma3",
        "repo": "google/gemma-3-1b-it-qat-q4_0-gguf",
        "tokenizer_repo": "google/gemma-3-1b-it",
        "arch": "gemma3",
    },
    {
        "name": "phi3",
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "tokenizer_repo": "microsoft/Phi-3.5-mini-instruct",
        "arch": "phi3",
    },
]

GGUF_DIR = os.environ.get("GGUF_DIR", os.path.join("temp", "gguf-test"))
BINARY = os.environ.get("BINARY", "target/release/prelude-server")
PORT = 8099
PROMPT = "What is 2+2? Answer briefly:"


def find_smallest_gguf(repo: str) -> str:
    """Find the smallest GGUF file in a repo (prefer Q4_K_M, then Q8_0, then any)."""
    files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
    if not files:
        raise RuntimeError(f"No GGUF files in {repo}")
    # Prefer small quantizations
    for pattern in ["Q4_K_M", "Q4_K_S", "Q4_0", "Q8_0"]:
        matches = [f for f in files if pattern in f]
        if matches:
            return matches[0]
    return files[0]


def download_model(model: dict) -> str:
    """Download GGUF file and tokenizer, return path to GGUF file."""
    model_dir = os.path.join(GGUF_DIR, model["name"])
    os.makedirs(model_dir, exist_ok=True)

    # Find and download GGUF
    gguf_name = find_smallest_gguf(model["repo"])
    gguf_path = os.path.join(model_dir, gguf_name)
    if not os.path.exists(gguf_path):
        print(f"  Downloading {model['repo']}/{gguf_name}...")
        hf_hub_download(model["repo"], gguf_name, local_dir=model_dir)
    else:
        print(f"  Already have {gguf_name}")

    # Download tokenizer.json next to GGUF
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"  Downloading tokenizer from {model['tokenizer_repo']}...")
        hf_hub_download(model["tokenizer_repo"], "tokenizer.json", local_dir=model_dir)

    return gguf_path


def start_server(gguf_path: str) -> subprocess.Popen:
    """Start the rust-infer server with a GGUF model."""
    env = os.environ.copy()
    env["PRELUDE_DEVICE"] = "cpu"
    env["RUST_LOG"] = "prelude_core=info"
    cmd = [
        BINARY,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--model", gguf_path,
    ]
    print(f"  Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return proc


def wait_for_server(timeout: int = 60) -> bool:
    """Wait for server health endpoint."""
    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def test_inference() -> tuple[bool, str]:
    """Send a completion request and check we get a non-empty response."""
    url = f"http://127.0.0.1:{PORT}/v1/completions"
    payload = {
        "model": "test",
        "prompt": PROMPT,
        "max_tokens": 20,
        "temperature": 0,
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        text = data.get("choices", [{}])[0].get("text", "")
        if not text.strip():
            return False, f"Empty response: {json.dumps(data)[:200]}"
        return True, text.strip()
    except Exception as e:
        return False, str(e)


def kill_server(proc: subprocess.Popen):
    """Kill server process."""
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait()


def main():
    results = []
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {model['name']} (arch={model['arch']})")
        print(f"{'='*60}")

        # Download
        try:
            gguf_path = download_model(model)
        except Exception as e:
            print(f"  SKIP: download failed: {e}")
            results.append((model["name"], "SKIP", str(e)))
            continue

        # Start server
        proc = start_server(gguf_path)
        try:
            print("  Waiting for server...")
            if not wait_for_server(timeout=120):
                # Capture server output for debugging
                proc.terminate()
                stdout, _ = proc.communicate(timeout=5)
                output = stdout.decode()[-2000:] if stdout else ""
                print(f"  FAIL: server didn't start")
                print(f"  Server output:\n{output}")
                results.append((model["name"], "FAIL", "server timeout"))
                continue

            print("  Server ready. Testing inference...")
            ok, text = test_inference()
            if ok:
                print(f"  PASS: {text[:80]}")
                results.append((model["name"], "PASS", text[:80]))
            else:
                print(f"  FAIL: {text[:200]}")
                results.append((model["name"], "FAIL", text[:200]))
        finally:
            kill_server(proc)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = 0
    for name, status, detail in results:
        icon = "✓" if status == "PASS" else ("⊘" if status == "SKIP" else "✗")
        print(f"  {icon} {name:12s} {status:5s}  {detail[:60]}")
        if status == "PASS":
            passed += 1
    total = len([r for r in results if r[1] != "SKIP"])
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
