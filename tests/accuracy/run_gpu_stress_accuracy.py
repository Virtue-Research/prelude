#!/usr/bin/env python3
"""Generate and optionally run a broader GPU-only endpoint accuracy sweep."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

import test_endpoint_accuracy as suite

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_CASES_OUTPUT = REPO_ROOT / "temp" / "accuracy" / "gpu-stress-cases.json"
DEFAULT_REPORT_OUTPUT = REPO_ROOT / "temp" / "accuracy" / "gpu-stress-report.json"
DEFAULT_GPU = "2"
DEFAULT_PAGED_BLOCKS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO_ROOT / suite.DEFAULT_BINARY))
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--samples-per-endpoint", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paged-blocks", type=int, default=DEFAULT_PAGED_BLOCKS)
    parser.add_argument("--cases-output", default=str(DEFAULT_CASES_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args()


def normalize_chunk(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def corpus_files() -> list[Path]:
    files: list[Path] = []
    files.append(REPO_ROOT / "README.md")
    for base in [
        REPO_ROOT / "docs",
        REPO_ROOT / "crates" / "prelude-core" / "src",
        REPO_ROOT / "crates" / "prelude-server" / "src",
        REPO_ROOT / "tests" / "accuracy",
    ]:
        if not base.exists():
            continue
        for suffix in ("*.md", "*.rs", "*.py", "*.json"):
            files.extend(sorted(base.rglob(suffix)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if any(
            part in {"target", ".git", "temp", "__pycache__"} for part in path.parts
        ):
            continue
        if path not in seen and path.is_file():
            deduped.append(path)
            seen.add(path)
    return deduped


def load_corpus() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in corpus_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for raw_chunk in re.split(r"\n\s*\n+", content):
            chunk = normalize_chunk(raw_chunk)
            if len(chunk) < 40 or len(chunk) > 4000:
                continue
            if chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(
                {
                    "text": chunk,
                    "source": str(path.relative_to(REPO_ROOT)),
                    "kind": path.suffix.lstrip(".") or "text",
                }
            )
    if not chunks:
        raise RuntimeError("failed to build stress corpus")
    return chunks


def length_targets(total: int) -> list[int]:
    cycle = [
        10000,
        8192,
        4096,
        4096,
        2048,
        2048,
        1024,
        1024,
        1024,
        512,
        512,
        512,
        256,
        256,
        256,
        128,
        128,
        64,
        64,
        32,
    ]
    return [cycle[index % len(cycle)] for index in range(total)]


def style_chunk(chunk: dict[str, str], style: str) -> str:
    text = chunk["text"]
    source = chunk["source"]
    if style == "section":
        return f"Source: {source}\n\n{text}"
    if style == "bullet":
        return f"- {text}"
    if style == "dialogue":
        return f"user: summarize this\nassistant: {text}"
    if style == "json":
        escaped = json.dumps(text)
        return f'{{"source": {json.dumps(source)}, "content": {escaped}}}'
    if style == "code":
        return f"```text\n{text}\n```"
    return text


def build_text(
    tokenizer,
    corpus: list[dict[str, str]],
    target_tokens: int,
    rng: random.Random,
    style: str,
) -> tuple[str, int, list[str]]:
    parts: list[str] = []
    sources: list[str] = []
    while True:
        chunk = rng.choice(corpus)
        parts.append(style_chunk(chunk, style))
        sources.append(chunk["source"])
        text = "\n\n".join(parts)
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        if token_count >= target_tokens:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            if token_count > target_tokens + 128:
                encoded = encoded[:target_tokens]
                text = tokenizer.decode(
                    encoded,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                token_count = len(tokenizer.encode(text, add_special_tokens=True))
            return text, token_count, sources


def text_to_token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=True))


def batch_targets(base_target: int, batch_size: int) -> list[int]:
    cap = min(base_target, 2048)
    if batch_size == 1:
        return [cap]
    factors = [1.0, 0.75, 0.5, 0.25]
    return [max(32, int(cap * factors[index])) for index in range(batch_size)]


def make_completion_case(
    index: int,
    tokenizer,
    corpus: list[dict[str, str]],
    target_tokens: int,
    rng: random.Random,
) -> dict[str, Any]:
    variant = index % 4
    style = ["plain", "section", "bullet", "json"][index % 4]
    if target_tokens >= 4096:
        max_tokens = 1
    elif target_tokens >= 1024:
        max_tokens = 2
    else:
        max_tokens = 4
    if variant == 1:
        prompts = []
        prompt_tokens = []
        sources: list[list[str]] = []
        for local_index, prompt_target in enumerate(batch_targets(target_tokens, 3)):
            text, actual_tokens, item_sources = build_text(
                tokenizer,
                corpus,
                prompt_target,
                random.Random(rng.randint(0, 10**9)),
                style,
            )
            prompts.append(text)
            prompt_tokens.append(actual_tokens)
            sources.append(item_sources)
        return {
            "id": f"completion_batch_{index:03d}",
            "prompt": prompts,
            "max_tokens": max_tokens,
            "meta": {
                "variant": "batch_prompt",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": prompt_tokens,
                "sources": sources,
            },
        }

    text, actual_tokens, sources = build_text(
        tokenizer, corpus, target_tokens, rng, style
    )
    case: dict[str, Any] = {
        "id": f"completion_single_{index:03d}",
        "prompt": text,
        "max_tokens": max_tokens,
        "meta": {
            "variant": "single_prompt",
            "target_prompt_tokens": target_tokens,
            "actual_prompt_tokens": actual_tokens,
            "sources": sources,
        },
    }
    if variant == 2:
        case["stop"] = ["\n\nuser:"]
        case["meta"]["variant"] = "single_prompt_stop"
    return case


def make_classify_case(
    index: int,
    tokenizer,
    corpus: list[dict[str, str]],
    target_tokens: int,
    rng: random.Random,
) -> dict[str, Any]:
    variant = index % 7
    style = ["plain", "section", "dialogue", "json", "code", "bullet", "plain"][
        index % 7
    ]
    if variant == 0:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, target_tokens, rng, style
        )
        return {
            "id": f"classify_text_{index:03d}",
            "input": text,
            "meta": {
                "variant": "input_text",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    if variant == 1:
        items = []
        prompt_tokens = []
        sources = []
        for prompt_target in batch_targets(target_tokens, 3):
            text, actual_tokens, item_sources = build_text(
                tokenizer,
                corpus,
                prompt_target,
                random.Random(rng.randint(0, 10**9)),
                style,
            )
            items.append(text)
            prompt_tokens.append(actual_tokens)
            sources.append(item_sources)
        return {
            "id": f"classify_batch_text_{index:03d}",
            "input": items,
            "meta": {
                "variant": "input_batch_text",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": prompt_tokens,
                "sources": sources,
            },
        }
    if variant == 2:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, target_tokens, rng, style
        )
        return {
            "id": f"classify_token_ids_{index:03d}",
            "input": text_to_token_ids(tokenizer, text),
            "meta": {
                "variant": "input_token_ids",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    if variant == 3:
        texts = []
        ids = []
        prompt_tokens = []
        sources = []
        for prompt_target in batch_targets(target_tokens, 2):
            text, actual_tokens, item_sources = build_text(
                tokenizer,
                corpus,
                prompt_target,
                random.Random(rng.randint(0, 10**9)),
                style,
            )
            texts.append(text)
            ids.append(text_to_token_ids(tokenizer, text))
            prompt_tokens.append(actual_tokens)
            sources.append(item_sources)
        return {
            "id": f"classify_batch_token_ids_{index:03d}",
            "input": ids,
            "meta": {
                "variant": "input_batch_token_ids",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": prompt_tokens,
                "sources": sources,
            },
        }
    if variant == 4:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, min(target_tokens, 4096), rng, "dialogue"
        )
        return {
            "id": f"classify_messages_{index:03d}",
            "messages": [
                {
                    "role": "system",
                    "content": "Classify the relevance of the user content.",
                },
                {"role": "user", "content": text},
            ],
            "meta": {
                "variant": "messages",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    if variant == 5:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, target_tokens, rng, style
        )
        return {
            "id": f"classify_source_text_{index:03d}",
            "source_text": text,
            "meta": {
                "variant": "source_text",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    texts = []
    prompt_tokens = []
    sources = []
    for prompt_target in batch_targets(target_tokens, 2):
        text, actual_tokens, item_sources = build_text(
            tokenizer,
            corpus,
            prompt_target,
            random.Random(rng.randint(0, 10**9)),
            style,
        )
        texts.append(text)
        prompt_tokens.append(actual_tokens)
        sources.append(item_sources)
    return {
        "id": f"classify_source_texts_{index:03d}",
        "source_texts": texts,
        "meta": {
            "variant": "source_texts",
            "target_prompt_tokens": target_tokens,
            "actual_prompt_tokens": prompt_tokens,
            "sources": sources,
        },
    }


def make_embedding_case(
    index: int,
    tokenizer,
    corpus: list[dict[str, str]],
    target_tokens: int,
    rng: random.Random,
) -> dict[str, Any]:
    variant = index % 6
    style = ["plain", "section", "code", "json", "bullet", "plain"][index % 6]
    if variant == 0:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, target_tokens, rng, style
        )
        return {
            "id": f"embedding_text_{index:03d}",
            "input": text,
            "meta": {
                "variant": "input_text",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    if variant == 1:
        texts = []
        prompt_tokens = []
        sources = []
        for prompt_target in batch_targets(target_tokens, 3):
            text, actual_tokens, item_sources = build_text(
                tokenizer,
                corpus,
                prompt_target,
                random.Random(rng.randint(0, 10**9)),
                style,
            )
            texts.append(text)
            prompt_tokens.append(actual_tokens)
            sources.append(item_sources)
        return {
            "id": f"embedding_batch_text_{index:03d}",
            "input": texts,
            "meta": {
                "variant": "input_batch_text",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": prompt_tokens,
                "sources": sources,
            },
        }
    if variant == 2:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, target_tokens, rng, style
        )
        return {
            "id": f"embedding_token_ids_{index:03d}",
            "input": text_to_token_ids(tokenizer, text),
            "meta": {
                "variant": "input_token_ids",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    if variant == 3:
        ids = []
        prompt_tokens = []
        sources = []
        for prompt_target in batch_targets(target_tokens, 2):
            text, actual_tokens, item_sources = build_text(
                tokenizer,
                corpus,
                prompt_target,
                random.Random(rng.randint(0, 10**9)),
                style,
            )
            ids.append(text_to_token_ids(tokenizer, text))
            prompt_tokens.append(actual_tokens)
            sources.append(item_sources)
        return {
            "id": f"embedding_batch_token_ids_{index:03d}",
            "input": ids,
            "meta": {
                "variant": "input_batch_token_ids",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": prompt_tokens,
                "sources": sources,
            },
        }
    if variant == 4:
        text, actual_tokens, sources = build_text(
            tokenizer, corpus, target_tokens, rng, style
        )
        return {
            "id": f"embedding_source_text_{index:03d}",
            "source_text": text,
            "meta": {
                "variant": "source_text",
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
                "sources": sources,
            },
        }
    texts = []
    prompt_tokens = []
    sources = []
    for prompt_target in batch_targets(target_tokens, 2):
        text, actual_tokens, item_sources = build_text(
            tokenizer,
            corpus,
            prompt_target,
            random.Random(rng.randint(0, 10**9)),
            style,
        )
        texts.append(text)
        prompt_tokens.append(actual_tokens)
        sources.append(item_sources)
    return {
        "id": f"embedding_source_texts_{index:03d}",
        "source_texts": texts,
        "meta": {
            "variant": "source_texts",
            "target_prompt_tokens": target_tokens,
            "actual_prompt_tokens": prompt_tokens,
            "sources": sources,
        },
    }


def build_case_matrix(
    samples_per_endpoint: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    corpus = load_corpus()

    completion_tokenizer = AutoTokenizer.from_pretrained(suite.DEFAULT_COMPLETION_MODEL)
    classify_tokenizer = AutoTokenizer.from_pretrained(suite.DEFAULT_CLASSIFY_MODEL)
    embedding_tokenizer = AutoTokenizer.from_pretrained(suite.DEFAULT_EMBEDDING_MODEL)
    suite.ensure_padding_token(completion_tokenizer)
    suite.ensure_padding_token(classify_tokenizer)
    suite.ensure_padding_token(embedding_tokenizer)

    targets = length_targets(samples_per_endpoint)
    return {
        "completion": [
            make_completion_case(
                index,
                completion_tokenizer,
                corpus,
                targets[index],
                random.Random(rng.randint(0, 10**9)),
            )
            for index in range(samples_per_endpoint)
        ],
        "classify": [
            make_classify_case(
                index,
                classify_tokenizer,
                corpus,
                targets[index],
                random.Random(rng.randint(0, 10**9)),
            )
            for index in range(samples_per_endpoint)
        ],
        "embedding": [
            make_embedding_case(
                index,
                embedding_tokenizer,
                corpus,
                targets[index],
                random.Random(rng.randint(0, 10**9)),
            )
            for index in range(samples_per_endpoint)
        ],
    }


def save_cases(path: Path, case_matrix: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")


def run_suite(args: argparse.Namespace, cases_path: Path) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PRELUDE_DEVICE"] = "auto"
    env["PRELUDE_PAGED_ATTN_BLOCKS"] = str(args.paged_blocks)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test_endpoint_accuracy.py"),
        "--endpoint",
        "all",
        "--binary",
        args.binary,
        "--cases",
        str(cases_path),
        "--output",
        args.output,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, env=env, cwd=REPO_ROOT).returncode


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases_output)
    case_matrix = build_case_matrix(args.samples_per_endpoint, args.seed)
    save_cases(cases_path, case_matrix)
    print(f"Wrote stress cases: {cases_path}")
    if args.generate_only:
        return 0
    return run_suite(args, cases_path)


if __name__ == "__main__":
    raise SystemExit(main())
