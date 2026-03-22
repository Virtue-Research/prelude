#!/usr/bin/env python3
"""Gemma 3 endpoint accuracy wrapper with public smoke defaults.

Defaults:
  completion: nmaroulis/tiny-random-Gemma3ForCausalLM
  classify:   nmaroulis/tiny-random-Gemma3ForSequenceClassification
  embedding:  michaelfeil/embeddinggemma-300m

Canonical Google repos remain opt-in via --use-canonical-google-models or the
PRELUDE_GEMMA3_CANONICAL_* environment variables.
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

PUBLIC_COMPLETION_MODEL = "nmaroulis/tiny-random-Gemma3ForCausalLM"
PUBLIC_CLASSIFY_MODEL = "nmaroulis/tiny-random-Gemma3ForSequenceClassification"
PUBLIC_EMBEDDING_MODEL = "michaelfeil/embeddinggemma-300m"

CANONICAL_COMPLETION_MODEL = "google/gemma-3-1b-it"
CANONICAL_EMBEDDING_MODEL = "google/embeddinggemma-300m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        choices=["completion", "classify", "embedding", "all"],
        default="all",
        help="Endpoint family to test",
    )
    parser.add_argument(
        "--binary",
        default=str(REPO_ROOT / "target" / "release" / "prelude-server"),
        help="Path to prelude-server binary for auto-start mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Port for auto-started Prelude",
    )
    parser.add_argument("--output", help="Report path for the delegated accuracy suite")
    parser.add_argument(
        "--cases",
        default=str(REPO_ROOT / "tests" / "accuracy" / "benchmark_guide_cases.json"),
        help="Path to the shared cases JSON file",
    )
    parser.add_argument(
        "--server-url",
        help="Pre-started Prelude URL. Valid only with a single --endpoint value.",
    )
    parser.add_argument(
        "--use-canonical-google-models",
        action="store_true",
        help="Use gated canonical Google completion/embedding repos instead of the public smoke defaults",
    )
    parser.add_argument(
        "--completion-model",
        help="Override completion model id/path",
    )
    parser.add_argument(
        "--classify-model",
        help="Override classify model id/path",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override embedding model id/path",
    )
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.use_canonical_google_models:
        default_completion = os.environ.get(
            "PRELUDE_GEMMA3_CANONICAL_COMPLETION_MODEL",
            CANONICAL_COMPLETION_MODEL,
        )
        default_embedding = os.environ.get(
            "PRELUDE_GEMMA3_CANONICAL_EMBEDDING_MODEL",
            CANONICAL_EMBEDDING_MODEL,
        )
    else:
        default_completion = os.environ.get(
            "PRELUDE_GEMMA3_COMPLETION_MODEL",
            PUBLIC_COMPLETION_MODEL,
        )
        default_embedding = os.environ.get(
            "PRELUDE_GEMMA3_EMBEDDING_MODEL",
            PUBLIC_EMBEDDING_MODEL,
        )

    default_classify = os.environ.get(
        "PRELUDE_GEMMA3_CLASSIFY_MODEL",
        PUBLIC_CLASSIFY_MODEL,
    )
    if args.use_canonical_google_models:
        default_classify = os.environ.get(
            "PRELUDE_GEMMA3_CANONICAL_CLASSIFY_MODEL",
            default_classify,
        )

    return (
        args.completion_model or default_completion,
        args.classify_model or default_classify,
        args.embedding_model or default_embedding,
    )


def main() -> int:
    args = parse_args()
    completion_model, classify_model, embedding_model = resolve_models(args)

    cmd = [
        sys.executable,
        str(ENDPOINT_SUITE),
        "--endpoint",
        args.endpoint,
        "--binary",
        args.binary,
        "--port",
        str(args.port),
        "--cases",
        args.cases,
        "--completion-model",
        completion_model,
        "--classify-model",
        classify_model,
        "--embedding-model",
        embedding_model,
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    if args.server_url:
        cmd.extend(["--server-url", args.server_url])

    print("Gemma 3 accuracy suite")
    print(f"  completion: {completion_model}")
    print(f"  classify:   {classify_model}")
    print(f"  embedding:  {embedding_model}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
