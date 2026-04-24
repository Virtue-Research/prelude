#!/usr/bin/env python3
"""Gemma 4 endpoint accuracy wrapper.

Defaults to the public E2B model for completion testing.
Uses the shared test_endpoint_accuracy.py suite to compare against HF.

Usage (two GPUs — server and HF reference on separate devices):

    PRELUDE_SERVER_DEVICE=cuda:4 PRELUDE_REF_DEVICE=cuda:5 \
    python3 tests/gemma4/test_accuracy.py --endpoint completion

Single GPU (only works for small models):

    CUDA_VISIBLE_DEVICES=4 PRELUDE_DEVICE=auto \
    python3 tests/gemma4/test_accuracy.py --endpoint completion
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ENDPOINT_SUITE = REPO_ROOT / "tests" / "accuracy" / "test_endpoint_accuracy.py"

DEFAULT_COMPLETION_MODEL = "google/gemma-4-E2B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--endpoint",
        choices=["completion", "all"],
        default="completion",
        help="Endpoint family to test (Gemma4 currently supports completion only)",
    )
    parser.add_argument(
        "--binary",
        default=str(REPO_ROOT / "target" / "release" / "prelude-server"),
        help="Path to prelude-server binary",
    )
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--output", help="Report output path")
    parser.add_argument(
        "--cases",
        default=str(REPO_ROOT / "tests" / "accuracy" / "benchmark_guide_cases.json"),
    )
    parser.add_argument("--server-url", help="Pre-started Prelude URL")
    parser.add_argument("--completion-model", default=DEFAULT_COMPLETION_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cmd = [
        sys.executable,
        str(ENDPOINT_SUITE),
        "--endpoint", args.endpoint,
        "--binary", args.binary,
        "--port", str(args.port),
        "--cases", args.cases,
        "--completion-model", args.completion_model,
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    if args.server_url:
        cmd.extend(["--server-url", args.server_url])

    print(f"Gemma 4 accuracy suite")
    print(f"  completion: {args.completion_model}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
