#!/usr/bin/env python3
"""
GGUF integration test for Prelude.

Downloads Qwen3-0.6B Q8_0 GGUF from HuggingFace and validates inference.

Usage:
    # Auto-download GGUF and run test (starts/stops server automatically)
    python tests/gguf/test_gguf.py

    # Use a pre-downloaded GGUF file
    python tests/gguf/test_gguf.py --gguf-path /path/to/qwen3-0.6b-q8_0.gguf

    # Use a pre-started server
    python tests/gguf/test_gguf.py --server-url http://localhost:8001

    # Custom binary
    python tests/gguf/test_gguf.py --binary target/release/prelude-server
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_GGUF_REPO = "Qwen/Qwen3-0.6B-GGUF"
DEFAULT_GGUF_FILE = "Qwen3-0.6B-Q8_0.gguf"
DEFAULT_BINARY = "target/release/prelude-server"
DEFAULT_PORT = 8099
HEALTH_TIMEOUT = 120  # seconds


def download_gguf(repo: str, filename: str, cache_dir: str) -> str:
    """Download GGUF file from HuggingFace Hub, return local path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=cache_dir,
    )
    print(f"GGUF file: {path}")
    return path


def start_server(binary: str, gguf_path: str, port: int) -> subprocess.Popen:
    """Start prelude-server with GGUF model."""
    env = os.environ.copy()
    env["PRELUDE_DEVICE"] = "cpu"
    env["RUST_LOG"] = "prelude_core=info"

    cmd = [
        binary,
        "--model", gguf_path,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_health(url: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    """Wait for server health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server healthy at {url}")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def test_completions(url: str) -> dict:
    """Test /v1/completions endpoint."""
    payload = {
        "model": DEFAULT_MODEL_ID,
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0,
    }
    print(f"\nTest: /v1/completions")
    print(f"  Prompt: {payload['prompt']!r}")
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["text"]
    print(f"  Output: {text!r}")
    assert len(text.strip()) > 0, "Empty response from completions"
    return data


def test_chat_completions(url: str) -> dict:
    """Test /v1/chat/completions endpoint."""
    payload = {
        "model": DEFAULT_MODEL_ID,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "max_tokens": 10,
        "temperature": 0,
    }
    print(f"\nTest: /v1/chat/completions")
    print(f"  Message: {payload['messages'][0]['content']!r}")
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    print(f"  Output: {text!r}")
    assert len(text.strip()) > 0, "Empty response from chat completions"
    return data


def test_models(url: str) -> dict:
    """Test /v1/models endpoint."""
    print(f"\nTest: /v1/models")
    r = requests.get(f"{url}/v1/models", timeout=10)
    r.raise_for_status()
    data = r.json()
    models = data.get("data", [])
    print(f"  Models: {[m['id'] for m in models]}")
    assert len(models) > 0, "No models returned"
    return data


def main():
    parser = argparse.ArgumentParser(description="GGUF integration test")
    parser.add_argument("--gguf-path", help="Path to pre-downloaded GGUF file")
    parser.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO, help="HF repo for GGUF")
    parser.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE, help="GGUF filename in repo")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model ID for tokenizer")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="prelude-server binary")
    parser.add_argument("--server-url", help="Pre-started server URL (skip server management)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory")
    args = parser.parse_args()

    # Resolve GGUF path
    gguf_path = args.gguf_path
    if not gguf_path and not args.server_url:
        print(f"Downloading GGUF: {args.gguf_repo}/{args.gguf_file}")
        gguf_path = download_gguf(args.gguf_repo, args.gguf_file, args.cache_dir)

    # Start or connect to server
    proc = None
    url = args.server_url
    if not url:
        if not os.path.exists(args.binary):
            print(f"Error: binary not found: {args.binary}")
            sys.exit(1)
        proc = start_server(args.binary, gguf_path, args.port)
        url = f"http://localhost:{args.port}"

    try:
        if proc:
            if not wait_for_health(url):
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                print(f"Server failed to start. stderr:\n{stderr}")
                sys.exit(1)

        # Run tests
        passed = 0
        failed = 0

        for test_fn in [test_models, test_completions, test_chat_completions]:
            try:
                test_fn(url)
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1

        print(f"\n{'='*40}")
        print(f"Results: {passed} passed, {failed} failed")
        if failed > 0:
            sys.exit(1)
        print("All GGUF tests passed!")

    finally:
        if proc:
            print("\nStopping server...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
