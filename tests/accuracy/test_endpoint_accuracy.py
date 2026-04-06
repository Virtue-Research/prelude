#!/usr/bin/env python3
"""Unified benchmark-guide endpoint accuracy suite.

This suite compares Prelude against HuggingFace reference execution for:
  - /v1/completions
  - /v1/classify
  - /v1/embeddings

References are generated on the fly and a JSON report with raw numeric values is
always written to disk.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from embedding_reference import (
    EmbeddingSemantics,
    apply_embedding_modules,
    load_embedding_semantics,
)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CASES = SCRIPT_DIR / "benchmark_guide_cases.json"
DEFAULT_BINARY = "target/release/prelude-server"
DEFAULT_PORT = 8099
HEALTH_TIMEOUT = 300
PROCESS_SHUTDOWN_TIMEOUT = 10
DEFAULT_COMPLETION_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CLASSIFY_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K_LOGPROBS = 5
CLASSIFY_COSINE_THRESHOLD = 0.999
EMBED_COSINE_THRESHOLD = 0.999
EMBED_NORM_DRIFT_THRESHOLD = 1e-3
EMBED_PAIRWISE_DRIFT_THRESHOLD = 1e-3


@dataclass
class ManagedServer:
    url: str
    proc: subprocess.Popen[bytes] | None = None


@dataclass
class ReferenceRuntime:
    device: torch.device
    torch_dtype: torch.dtype
    device_label: str
    dtype_label: str


@dataclass
class CompletionReferenceContext:
    model_id: str
    runtime: ReferenceRuntime
    tokenizer: Any
    model: Any
    eos_ids: set[int]


@dataclass
class ClassifyReferenceContext:
    model_id: str
    runtime: ReferenceRuntime
    tokenizer: Any
    model: Any


@dataclass
class EmbeddingReferenceContext:
    model_id: str
    runtime: ReferenceRuntime
    tokenizer: Any
    semantics: EmbeddingSemantics
    model: Any


_REFERENCE_CONTEXTS: dict[tuple[str, str, str], Any] = {}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def default_output_path() -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("temp") / "accuracy" / f"benchmark-guide-endpoint-accuracy-{stamp}.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_report(path: Path, report: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Report written: {path}")


def resolve_reference_runtime() -> ReferenceRuntime:
    # PRELUDE_REF_DEVICE: GPU for HF reference model (e.g. "cuda:1").
    # Falls back to PRELUDE_DEVICE for single-GPU setups.
    requested = os.environ.get("PRELUDE_REF_DEVICE",
                os.environ.get("PRELUDE_DEVICE", "cpu")).strip().lower()
    if requested != "cpu" and torch.cuda.is_available():
        device = torch.device(requested if ":" in requested else "cuda")
        return ReferenceRuntime(
            device=device,
            torch_dtype=torch.bfloat16,
            device_label=str(device),
            dtype_label="bfloat16",
        )
    return ReferenceRuntime(
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        device_label="cpu",
        dtype_label="float32",
    )


def reference_runtime_key(runtime: ReferenceRuntime) -> str:
    return f"{runtime.device_label}:{runtime.dtype_label}"


def release_reference_model(model: Any, runtime: ReferenceRuntime) -> None:
    del model
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()


def ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token


def wait_for_server(
    url: str,
    proc: subprocess.Popen[bytes] | None = None,
    timeout: int = HEALTH_TIMEOUT,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            return False
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def stop_process(proc: subprocess.Popen[bytes]) -> str:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    try:
        stdout, _ = proc.communicate(timeout=PROCESS_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate(timeout=PROCESS_SHUTDOWN_TIMEOUT)
    return stdout.decode("utf-8", errors="replace") if stdout else ""


def server_task(endpoint: str) -> str:
    if endpoint == "completion":
        return "generation"
    if endpoint == "classify":
        return "classify"
    if endpoint == "embedding":
        return "embedding"
    raise ValueError(f"unknown endpoint: {endpoint}")


def start_server(
    binary: str,
    model: str,
    port: int,
    endpoint: str,
) -> ManagedServer:
    env = os.environ.copy()
    # PRELUDE_SERVER_DEVICE: GPU for the Prelude server.
    # When running HF reference on a separate GPU (PRELUDE_REF_DEVICE),
    # set PRELUDE_SERVER_DEVICE to isolate them.
    server_device = os.environ.get("PRELUDE_SERVER_DEVICE",
                    os.environ.get("PRELUDE_DEVICE", "cpu"))
    env["PRELUDE_DEVICE"] = server_device
    # If a specific CUDA device is requested (e.g. "cuda:1"), set CUDA_VISIBLE_DEVICES
    # so the server process only sees that GPU.
    if server_device.startswith("cuda:"):
        gpu_idx = server_device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        env["PRELUDE_DEVICE"] = "auto"
    host = "0.0.0.0"
    selected_port = choose_auto_start_port(host, port)
    if selected_port != port:
        print(f"Requested port {port} is busy; using free port {selected_port} instead")

    cmd = [
        binary,
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(selected_port),
        "--task",
        server_task(endpoint),
    ]
    print(f"Starting Prelude for {model}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    url = f"http://localhost:{selected_port}"
    if not wait_for_server(url, proc):
        output = stop_process(proc)
        raise RuntimeError(f"server failed to start for {model}:\n{output}")
    return ManagedServer(url=url, proc=proc)


def close_server(server: ManagedServer | None) -> None:
    if server and server.proc:
        stop_process(server.proc)


def encode_text(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=True))


def effective_pad_token_id(tokenizer) -> int:
    return int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_auto_start_port(host: str, requested_port: int) -> int:
    if is_port_free(host, requested_port):
        return requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def normalize_token_id_case_input(raw_input: Any) -> list[list[int]]:
    if (
        isinstance(raw_input, list)
        and raw_input
        and all(isinstance(item, int) for item in raw_input)
    ):
        return [list(map(int, raw_input))]
    if isinstance(raw_input, list):
        return [list(map(int, ids)) for ids in raw_input]
    raise ValueError(f"unsupported token-id input: {raw_input!r}")


def lookup_label(id2label: dict[Any, Any] | None, label_index: int) -> str:
    if not id2label:
        return f"LABEL_{label_index}"
    return str(
        id2label.get(
            label_index, id2label.get(str(label_index), f"LABEL_{label_index}")
        )
    )


def build_padded_token_batch(
    token_ids: list[list[int]],
    device: torch.device,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in token_ids)
    input_rows = []
    mask_rows = []
    for ids in token_ids:
        padding = [pad_token_id] * (max_len - len(ids))
        input_rows.append(ids + padding)
        mask_rows.append([1] * len(ids) + [0] * len(padding))
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    return input_ids, attention_mask


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    denom = np.linalg.norm(lhs_arr) * np.linalg.norm(rhs_arr)
    if denom == 0.0:
        return 1.0 if np.allclose(lhs_arr, rhs_arr) else 0.0
    return float(np.dot(lhs_arr, rhs_arr) / denom)


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def find_first_diff_char(actual: str, expected: str) -> int | None:
    for idx, (lhs, rhs) in enumerate(zip(actual, expected)):
        if lhs != rhs:
            return idx
    if len(actual) != len(expected):
        return min(len(actual), len(expected))
    return None


def format_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def infer_server_completion_tokens(choice: dict[str, Any]) -> int:
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return len(choice.get("text", ""))


def build_completion_request(case: dict[str, Any], model: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": case["prompt"],
        "max_tokens": int(case["max_tokens"]),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "logprobs": TOP_K_LOGPROBS,
        "seed": 0,
    }
    if "stop" in case:
        body["stop"] = case["stop"]
    return body


def get_completion_reference_context(
    model_id: str,
    runtime: ReferenceRuntime,
) -> CompletionReferenceContext:
    cache_key = ("completion", model_id, reference_runtime_key(runtime))
    cached = _REFERENCE_CONTEXTS.get(cache_key)
    if cached is not None:
        return cached

    print(
        f"  Loading HF completion reference: {model_id} ({runtime.device_label}/{runtime.dtype_label})"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=runtime.torch_dtype,
    ).to(runtime.device)
    model.eval()
    torch.manual_seed(0)

    eos_cfg = model.generation_config.eos_token_id
    eos_ids = set()
    if isinstance(eos_cfg, int):
        eos_ids.add(int(eos_cfg))
    elif isinstance(eos_cfg, (list, tuple)):
        eos_ids.update(int(token_id) for token_id in eos_cfg)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    context = CompletionReferenceContext(
        model_id=model_id,
        runtime=runtime,
        tokenizer=tokenizer,
        model=model,
        eos_ids=eos_ids,
    )
    _REFERENCE_CONTEXTS[cache_key] = context
    return context


def close_completion_reference_context(
    model_id: str,
    runtime: ReferenceRuntime,
) -> None:
    cache_key = ("completion", model_id, reference_runtime_key(runtime))
    context = _REFERENCE_CONTEXTS.pop(cache_key, None)
    if context is not None:
        release_reference_model(context.model, runtime)


def load_completion_reference(
    model_id: str,
    runtime: ReferenceRuntime,
    prompts: list[str],
    max_tokens: int,
    stop_strings: list[str],
) -> dict[str, Any]:
    context = get_completion_reference_context(model_id, runtime)
    tokenizer = context.tokenizer
    model = context.model
    eos_ids = context.eos_ids

    items: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    with torch.no_grad():
        for index, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(context.runtime.device)
            attention_mask = encoded["attention_mask"].to(context.runtime.device)
            prompt_tokens = int(attention_mask.sum().item())
            total_prompt_tokens += prompt_tokens

            generated_ids: list[int] = []
            token_logprobs: list[dict[str, Any]] = []
            finish_reason = "length"
            stop_hit = False

            current_ids = input_ids
            current_mask = attention_mask
            for _ in range(max_tokens):
                outputs = model(input_ids=current_ids, attention_mask=current_mask)
                logits = outputs.logits[:, -1, :].squeeze(0).float()
                log_probs = torch.log_softmax(logits, dim=-1)
                token_id = int(torch.argmax(log_probs).item())

                top_vals, top_ids = torch.topk(
                    log_probs, k=min(TOP_K_LOGPROBS, log_probs.shape[-1])
                )
                token_logprobs.append(
                    {
                        "token_id": token_id,
                        "token": tokenizer.decode(
                            [token_id], skip_special_tokens=False
                        ),
                        "logprob": float(log_probs[token_id].item()),
                        "top_logprobs": [
                            {
                                "token_id": int(top_id.item()),
                                "token": tokenizer.decode(
                                    [int(top_id.item())],
                                    skip_special_tokens=False,
                                ),
                                "logprob": float(top_val.item()),
                            }
                            for top_id, top_val in zip(top_ids, top_vals)
                        ],
                    }
                )
                generated_ids.append(token_id)

                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if token_id in eos_ids:
                    finish_reason = "stop"
                    break
                if stop_strings and any(stop in output_text for stop in stop_strings):
                    finish_reason = "stop"
                    stop_hit = True
                    break

                next_token = torch.tensor(
                    [[token_id]], dtype=torch.long, device=context.runtime.device
                )
                current_ids = torch.cat([current_ids, next_token], dim=1)
                next_mask = torch.ones(
                    (1, 1), dtype=current_mask.dtype, device=context.runtime.device
                )
                current_mask = torch.cat([current_mask, next_mask], dim=1)

            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion_tokens = len(generated_ids)
            total_completion_tokens += completion_tokens
            items.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens,
                    "output_text": output_text,
                    "output_token_ids": generated_ids,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "stop_hit": stop_hit,
                    "token_logprobs": token_logprobs,
                }
            )

    return {
        "model": model_id,
        "runtime": {
            "device": runtime.device_label,
            "dtype": runtime.dtype_label,
        },
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "items": items,
    }


def query_completion(url: str, request_body: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{url}/v1/completions", json=request_body, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(
            f"completion request failed with {response.status_code}: {response.text.strip()}"
        )
    return response.json()


def compare_completion_item(
    ref_item: dict[str, Any],
    actual_choice: dict[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    issues: list[str] = []
    actual_text = actual_choice.get("text", "")
    text_match = actual_text == ref_item["output_text"]

    actual_logprobs = actual_choice.get("logprobs") or {}
    actual_tokens = actual_logprobs.get("tokens") or []
    actual_token_logprobs = actual_logprobs.get("token_logprobs") or []
    actual_top_logprobs = actual_logprobs.get("top_logprobs") or []

    cross_contained: bool | None = None
    divergence_detail = None
    if ref_item["token_logprobs"] and actual_tokens:
        cross_contained = True
        for step in range(min(len(ref_item["token_logprobs"]), len(actual_tokens))):
            ref_token = ref_item["token_logprobs"][step]["token"]
            actual_token = actual_tokens[step]
            if ref_token != actual_token:
                ref_top = {
                    entry["token"]
                    for entry in ref_item["token_logprobs"][step]["top_logprobs"]
                }
                server_top_entry = (
                    actual_top_logprobs[step] if step < len(actual_top_logprobs) else {}
                )
                server_top = (
                    set(server_top_entry.keys())
                    if isinstance(server_top_entry, dict)
                    else set()
                )
                server_in_ref = actual_token in ref_top
                ref_in_server = ref_token in server_top if server_top else None
                cross_contained = server_in_ref and (
                    ref_in_server is None or ref_in_server
                )
                if cross_contained:
                    divergence_detail = (
                        f"step {step}: bidirectional top-{TOP_K_LOGPROBS} containment"
                    )
                elif server_in_ref:
                    divergence_detail = (
                        f"step {step}: server token '{actual_token}' is in HF top-{TOP_K_LOGPROBS}, "
                        f"but HF token '{ref_token}' is not in server top-{TOP_K_LOGPROBS}"
                    )
                else:
                    divergence_detail = f"step {step}: server token '{actual_token}' is not in HF top-{TOP_K_LOGPROBS}"
                break

    actual_completion_tokens = infer_server_completion_tokens(actual_choice)
    completion_tokens_match = actual_completion_tokens == ref_item["completion_tokens"]
    if not completion_tokens_match:
        issues.append(
            f"completion_tokens mismatch: {actual_completion_tokens} != {ref_item['completion_tokens']}"
        )

    actual_finish_reason = actual_choice.get("finish_reason")
    finish_reason_match = actual_finish_reason == ref_item["finish_reason"]
    if not finish_reason_match:
        issues.append(
            f"finish_reason mismatch: {actual_finish_reason} != {ref_item['finish_reason']}"
        )

    if not text_match and cross_contained is not True:
        issues.append("text mismatch without top-k cross-containment")

    min_len = min(len(ref_item["token_logprobs"]), len(actual_token_logprobs))
    cosine = None
    lp_max_diff = None
    if min_len > 0:
        ref_vec = [ref_item["token_logprobs"][idx]["logprob"] for idx in range(min_len)]
        act_vec = [float(actual_token_logprobs[idx]) for idx in range(min_len)]
        cosine = cosine_similarity(ref_vec, act_vec)
        lp_max_diff = max_abs_diff(ref_vec, act_vec)

    metrics = {
        "index_match": actual_choice.get("index") == ref_item["index"],
        "text_match": text_match,
        "cross_contained": cross_contained,
        "completion_tokens_match": completion_tokens_match,
        "finish_reason_match": finish_reason_match,
        "first_diff_char": find_first_diff_char(actual_text, ref_item["output_text"]),
        "divergence_detail": divergence_detail,
        "logprob_cosine_similarity": cosine,
        "max_logprob_diff": lp_max_diff,
        "server_completion_tokens": actual_completion_tokens,
        "reference_completion_tokens": ref_item["completion_tokens"],
    }
    if not metrics["index_match"]:
        issues.append(
            f"choice index mismatch: {actual_choice.get('index')} != {ref_item['index']}"
        )

    passed = (
        metrics["index_match"]
        and completion_tokens_match
        and finish_reason_match
        and (text_match or cross_contained is True)
    )
    return passed, metrics, issues


def evaluate_completion_case(
    case: dict[str, Any],
    model_id: str,
    runtime: ReferenceRuntime,
    url: str,
) -> dict[str, Any]:
    prompts = case["prompt"] if isinstance(case["prompt"], list) else [case["prompt"]]
    stop_strings = list(case.get("stop", []))
    request_body = build_completion_request(case, model_id)
    reference = load_completion_reference(
        model_id,
        runtime,
        prompts,
        int(case["max_tokens"]),
        stop_strings,
    )
    response = query_completion(url, request_body)

    issues: list[str] = []
    if response["usage"]["prompt_tokens"] != reference["usage"]["prompt_tokens"]:
        issues.append(
            "prompt_tokens mismatch: "
            f"{response['usage']['prompt_tokens']} != {reference['usage']['prompt_tokens']}"
        )
    if (
        response["usage"]["completion_tokens"]
        != reference["usage"]["completion_tokens"]
    ):
        issues.append(
            "completion_tokens mismatch: "
            f"{response['usage']['completion_tokens']} != {reference['usage']['completion_tokens']}"
        )

    actual_choices = response.get("choices") or []
    if len(actual_choices) != len(reference["items"]):
        issues.append(
            f"choice count mismatch: {len(actual_choices)} != {len(reference['items'])}"
        )

    item_reports: list[dict[str, Any]] = []
    for index, ref_item in enumerate(reference["items"]):
        if index >= len(actual_choices):
            item_reports.append(
                {
                    "index": index,
                    "passed": False,
                    "issues": ["missing server choice"],
                }
            )
            continue
        item_passed, metrics, item_issues = compare_completion_item(
            ref_item, actual_choices[index]
        )
        item_reports.append(
            {
                "index": index,
                "passed": item_passed,
                "issues": item_issues,
                "metrics": metrics,
            }
        )
        issues.extend(f"item[{index}]: {issue}" for issue in item_issues)

    passed = not issues and all(item["passed"] for item in item_reports)
    return {
        "id": case["id"],
        "endpoint": "completion",
        "request": request_body,
        "hf_reference": reference,
        "prelude_response": response,
        "metrics": {
            "usage_prompt_tokens_match": response["usage"]["prompt_tokens"]
            == reference["usage"]["prompt_tokens"],
            "usage_completion_tokens_match": response["usage"]["completion_tokens"]
            == reference["usage"]["completion_tokens"],
            "items": item_reports,
        },
        "issues": issues,
        "passed": passed,
    }


def build_classify_request(
    case: dict[str, Any], tokenizer, model: str
) -> dict[str, Any]:
    if "messages" in case:
        return {"model": model, "messages": case["messages"]}
    if "input" in case:
        return {"model": model, "input": case["input"]}
    if "source_text" in case:
        return {"model": model, "input": encode_text(tokenizer, case["source_text"])}
    if "source_texts" in case:
        return {
            "model": model,
            "input": [encode_text(tokenizer, text) for text in case["source_texts"]],
        }
    raise ValueError(f"unsupported classify case shape: {case['id']}")


def query_classify(url: str, request_body: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{url}/v1/classify", json=request_body, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(
            f"classify request failed with {response.status_code}: {response.text.strip()}"
        )
    return response.json()


def get_classify_reference_context(
    model_id: str,
    runtime: ReferenceRuntime,
) -> ClassifyReferenceContext:
    cache_key = ("classify", model_id, reference_runtime_key(runtime))
    cached = _REFERENCE_CONTEXTS.get(cache_key)
    if cached is not None:
        return cached

    print(
        f"  Loading HF classify reference: {model_id} ({runtime.device_label}/{runtime.dtype_label})"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_padding_token(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=runtime.torch_dtype,
    ).to(runtime.device)
    model.eval()

    context = ClassifyReferenceContext(
        model_id=model_id,
        runtime=runtime,
        tokenizer=tokenizer,
        model=model,
    )
    _REFERENCE_CONTEXTS[cache_key] = context
    return context


def close_classify_reference_context(
    model_id: str,
    runtime: ReferenceRuntime,
) -> None:
    cache_key = ("classify", model_id, reference_runtime_key(runtime))
    context = _REFERENCE_CONTEXTS.pop(cache_key, None)
    if context is not None:
        release_reference_model(context.model, runtime)


def load_classify_reference(
    model_id: str,
    runtime: ReferenceRuntime,
    case: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    context = get_classify_reference_context(model_id, runtime)
    tokenizer = context.tokenizer
    model = context.model

    if "messages" in case:
        texts = [format_messages(case["messages"])]
        input_ids = None
        attention_mask = None
    elif "input" in case:
        raw_input = case["input"]
        if isinstance(raw_input, str):
            texts = [raw_input]
            input_ids = None
            attention_mask = None
        else:
            texts = raw_input if raw_input and isinstance(raw_input[0], str) else None
            if texts is None:
                token_ids = normalize_token_id_case_input(raw_input)
                input_ids, attention_mask = build_padded_token_batch(
                    token_ids,
                    context.runtime.device,
                    effective_pad_token_id(tokenizer),
                )
                texts = None
            else:
                input_ids = None
                attention_mask = None
    elif "source_text" in case:
        token_ids = [encode_text(tokenizer, case["source_text"])]
        texts = None
        input_ids, attention_mask = build_padded_token_batch(
            token_ids,
            context.runtime.device,
            effective_pad_token_id(tokenizer),
        )
    elif "source_texts" in case:
        token_ids = [encode_text(tokenizer, text) for text in case["source_texts"]]
        texts = None
        input_ids, attention_mask = build_padded_token_batch(
            token_ids,
            context.runtime.device,
            effective_pad_token_id(tokenizer),
        )
    else:
        raise ValueError(f"unsupported classify case shape: {case['id']}")

    if texts is not None:
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(context.runtime.device)
        attention_mask = encoded["attention_mask"].to(context.runtime.device)

    with torch.no_grad():
        logits = (
            model(input_ids=input_ids, attention_mask=attention_mask)
            .logits.float()
            .cpu()
            .numpy()
        )

    results = []
    prompt_tokens = int(attention_mask.sum().item())
    for index, row in enumerate(logits):
        label_index = int(np.argmax(row))
        label = lookup_label(getattr(model.config, "id2label", None), label_index)
        results.append(
            {
                "index": index,
                "label": label,
                "probs": row.tolist(),
                "num_classes": int(row.shape[0]),
            }
        )

    return {
        "model": model_id,
        "runtime": {
            "device": runtime.device_label,
            "dtype": runtime.dtype_label,
        },
        "prompt_tokens": prompt_tokens,
        "results": results,
    }, tokenizer


def evaluate_classify_case(
    case: dict[str, Any],
    model_id: str,
    runtime: ReferenceRuntime,
    url: str,
) -> dict[str, Any]:
    reference, tokenizer = load_classify_reference(model_id, runtime, case)
    request_body = build_classify_request(case, tokenizer, model_id)
    response = query_classify(url, request_body)

    issues: list[str] = []
    if response["usage"]["prompt_tokens"] != reference["prompt_tokens"]:
        issues.append(
            f"prompt_tokens mismatch: {response['usage']['prompt_tokens']} != {reference['prompt_tokens']}"
        )

    actual_items = response.get("data") or []
    if len(actual_items) != len(reference["results"]):
        issues.append(
            f"result count mismatch: {len(actual_items)} != {len(reference['results'])}"
        )

    item_reports: list[dict[str, Any]] = []
    for index, ref_item in enumerate(reference["results"]):
        if index >= len(actual_items):
            item_reports.append(
                {"index": index, "passed": False, "issues": ["missing server result"]}
            )
            continue

        actual_item = actual_items[index]
        actual_probs = actual_item.get("probs", [])
        probs_len_match = len(actual_probs) == len(ref_item["probs"])
        logits_cosine = (
            cosine_similarity(ref_item["probs"], actual_probs)
            if probs_len_match
            else None
        )
        logits_max_diff = (
            max_abs_diff(ref_item["probs"], actual_probs) if probs_len_match else None
        )
        label_match = actual_item.get("label") == ref_item["label"]
        num_classes_match = actual_item.get("num_classes") == ref_item["num_classes"]
        index_match = actual_item.get("index") == ref_item["index"]
        actual_argmax = int(np.argmax(actual_probs)) if actual_probs else None
        expected_argmax = int(np.argmax(ref_item["probs"]))
        argmax_match = actual_argmax == expected_argmax

        item_issues = []
        if not label_match:
            item_issues.append(
                f"label mismatch: {actual_item.get('label')} != {ref_item['label']}"
            )
        if not num_classes_match:
            item_issues.append(
                f"num_classes mismatch: {actual_item.get('num_classes')} != {ref_item['num_classes']}"
            )
        if not probs_len_match:
            item_issues.append(
                f"logit vector length mismatch: {len(actual_probs)} != {len(ref_item['probs'])}"
            )
        if not index_match:
            item_issues.append(
                f"index mismatch: {actual_item.get('index')} != {ref_item['index']}"
            )
        if not argmax_match:
            item_issues.append(f"argmax mismatch: {actual_argmax} != {expected_argmax}")
        if logits_cosine is not None and logits_cosine < CLASSIFY_COSINE_THRESHOLD:
            item_issues.append(
                f"logits cosine similarity below threshold: {logits_cosine:.6f} < {CLASSIFY_COSINE_THRESHOLD:.3f}"
            )

        item_passed = not item_issues
        item_reports.append(
            {
                "index": index,
                "passed": item_passed,
                "issues": item_issues,
                "metrics": {
                    "index_match": index_match,
                    "label_match": label_match,
                    "num_classes_match": num_classes_match,
                    "probs_len_match": probs_len_match,
                    "argmax_match": argmax_match,
                    "logits_cosine_similarity": logits_cosine,
                    "max_abs_diff": logits_max_diff,
                },
            }
        )
        issues.extend(f"item[{index}]: {issue}" for issue in item_issues)

    passed = not issues and all(item["passed"] for item in item_reports)
    return {
        "id": case["id"],
        "endpoint": "classify",
        "request": request_body,
        "hf_reference": reference,
        "prelude_response": response,
        "metrics": {
            "usage_prompt_tokens_match": response["usage"]["prompt_tokens"]
            == reference["prompt_tokens"],
            "items": item_reports,
        },
        "issues": issues,
        "passed": passed,
    }


def build_embedding_request(
    case: dict[str, Any], tokenizer, model: str
) -> dict[str, Any]:
    if "input" in case:
        return {"model": model, "input": case["input"]}
    if "source_text" in case:
        return {"model": model, "input": encode_text(tokenizer, case["source_text"])}
    if "source_texts" in case:
        return {
            "model": model,
            "input": [encode_text(tokenizer, text) for text in case["source_texts"]],
        }
    raise ValueError(f"unsupported embedding case shape: {case['id']}")


def query_embedding(url: str, request_body: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{url}/v1/embeddings", json=request_body, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(
            f"embedding request failed with {response.status_code}: {response.text.strip()}"
        )
    return response.json()


def get_embedding_reference_context(
    model_id: str,
    runtime: ReferenceRuntime,
) -> EmbeddingReferenceContext:
    cache_key = ("embedding", model_id, reference_runtime_key(runtime))
    cached = _REFERENCE_CONTEXTS.get(cache_key)
    if cached is not None:
        return cached

    print(
        f"  Loading HF embedding reference: {model_id} ({runtime.device_label}/{runtime.dtype_label})"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_padding_token(tokenizer)
    semantics = load_embedding_semantics(model_id, device=runtime.device)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=runtime.torch_dtype,
    ).to(runtime.device)
    model.eval()

    context = EmbeddingReferenceContext(
        model_id=model_id,
        runtime=runtime,
        tokenizer=tokenizer,
        semantics=semantics,
        model=model,
    )
    _REFERENCE_CONTEXTS[cache_key] = context
    return context


def close_embedding_reference_context(
    model_id: str,
    runtime: ReferenceRuntime,
) -> None:
    cache_key = ("embedding", model_id, reference_runtime_key(runtime))
    context = _REFERENCE_CONTEXTS.pop(cache_key, None)
    if context is not None:
        release_reference_model(context.model, runtime)


def load_embedding_reference(
    model_id: str,
    runtime: ReferenceRuntime,
    case: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    context = get_embedding_reference_context(model_id, runtime)
    tokenizer = context.tokenizer
    semantics = context.semantics
    model = context.model

    if "input" in case:
        raw_input = case["input"]
        if isinstance(raw_input, str):
            texts = [raw_input]
            input_ids = None
            attention_mask = None
        else:
            texts = raw_input if raw_input and isinstance(raw_input[0], str) else None
            if texts is None:
                token_ids = normalize_token_id_case_input(raw_input)
                input_ids, attention_mask = build_padded_token_batch(
                    token_ids,
                    context.runtime.device,
                    effective_pad_token_id(tokenizer),
                )
                texts = None
            else:
                input_ids = None
                attention_mask = None
    elif "source_text" in case:
        token_ids = [encode_text(tokenizer, case["source_text"])]
        texts = None
        input_ids, attention_mask = build_padded_token_batch(
            token_ids,
            context.runtime.device,
            effective_pad_token_id(tokenizer),
        )
    elif "source_texts" in case:
        token_ids = [encode_text(tokenizer, text) for text in case["source_texts"]]
        texts = None
        input_ids, attention_mask = build_padded_token_batch(
            token_ids,
            context.runtime.device,
            effective_pad_token_id(tokenizer),
        )
    else:
        raise ValueError(f"unsupported embedding case shape: {case['id']}")

    if texts is not None:
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(context.runtime.device)
        attention_mask = encoded["attention_mask"].to(context.runtime.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state.float()
        embeddings = (
            apply_embedding_modules(hidden_states, attention_mask, semantics)
            .cpu()
            .numpy()
        )

    prompt_tokens = int(attention_mask.sum().item())
    items = []
    for index, embedding in enumerate(embeddings):
        vector = embedding.tolist()
        items.append(
            {
                "index": index,
                "embedding": vector,
                "norm": float(np.linalg.norm(np.asarray(vector, dtype=np.float32))),
            }
        )

    return {
        "model": model_id,
        "runtime": {
            "device": runtime.device_label,
            "dtype": runtime.dtype_label,
        },
        "semantics": {
            "pooling": semantics.pooling,
            "normalize": semantics.normalize,
            "modules": semantics.modules,
        },
        "prompt_tokens": prompt_tokens,
        "dimensions": int(embeddings.shape[1]),
        "items": items,
    }, tokenizer


def pairwise_cosine_matrix(items: list[dict[str, Any]], key: str) -> list[list[float]]:
    vectors = [item[key] for item in items]
    return [[cosine_similarity(lhs, rhs) for rhs in vectors] for lhs in vectors]


def evaluate_embedding_case(
    case: dict[str, Any],
    model_id: str,
    runtime: ReferenceRuntime,
    url: str,
) -> dict[str, Any]:
    reference, tokenizer = load_embedding_reference(model_id, runtime, case)
    request_body = build_embedding_request(case, tokenizer, model_id)
    response = query_embedding(url, request_body)

    issues: list[str] = []
    if response["usage"]["prompt_tokens"] != reference["prompt_tokens"]:
        issues.append(
            f"prompt_tokens mismatch: {response['usage']['prompt_tokens']} != {reference['prompt_tokens']}"
        )

    actual_items = response.get("data") or []
    if len(actual_items) != len(reference["items"]):
        issues.append(
            f"embedding count mismatch: {len(actual_items)} != {len(reference['items'])}"
        )

    item_reports: list[dict[str, Any]] = []
    for index, ref_item in enumerate(reference["items"]):
        if index >= len(actual_items):
            item_reports.append(
                {
                    "index": index,
                    "passed": False,
                    "issues": ["missing server embedding"],
                }
            )
            continue

        actual_item = actual_items[index]
        actual_embedding = actual_item.get("embedding", [])
        actual_norm = float(
            np.linalg.norm(np.asarray(actual_embedding, dtype=np.float32))
        )
        dimension_match = len(actual_embedding) == reference["dimensions"]
        index_match = actual_item.get("index") == ref_item["index"]
        cosine = (
            cosine_similarity(ref_item["embedding"], actual_embedding)
            if dimension_match
            else None
        )
        norm_drift = abs(actual_norm - ref_item["norm"]) if dimension_match else None
        drift = (
            max_abs_diff(ref_item["embedding"], actual_embedding)
            if dimension_match
            else None
        )

        item_issues = []
        if not dimension_match:
            item_issues.append(
                f"dimension mismatch: {len(actual_embedding)} != {reference['dimensions']}"
            )
        if not index_match:
            item_issues.append(
                f"index mismatch: {actual_item.get('index')} != {ref_item['index']}"
            )
        if cosine is not None and cosine < EMBED_COSINE_THRESHOLD:
            item_issues.append(
                f"cosine similarity below threshold: {cosine:.6f} < {EMBED_COSINE_THRESHOLD:.3f}"
            )
        if norm_drift is not None and norm_drift > EMBED_NORM_DRIFT_THRESHOLD:
            item_issues.append(
                f"norm drift above threshold: {norm_drift:.6f} > {EMBED_NORM_DRIFT_THRESHOLD:.1e}"
            )

        item_passed = not item_issues
        item_reports.append(
            {
                "index": index,
                "passed": item_passed,
                "issues": item_issues,
                "metrics": {
                    "index_match": index_match,
                    "dimension_match": dimension_match,
                    "cosine_similarity": cosine,
                    "norm_drift": norm_drift,
                    "max_abs_diff": drift,
                },
            }
        )
        issues.extend(f"item[{index}]: {issue}" for issue in item_issues)

    actual_dimensions_uniform = all(
        len(item.get("embedding", [])) == reference["dimensions"] for item in actual_items
    )
    actual_matrix = (
        pairwise_cosine_matrix(actual_items, "embedding")
        if actual_items and actual_dimensions_uniform
        else []
    )
    reference_matrix = (
        pairwise_cosine_matrix(reference["items"], "embedding")
        if reference["items"]
        else []
    )
    pairwise_drift = None
    if (
        actual_matrix
        and reference_matrix
        and len(actual_matrix) == len(reference_matrix)
    ):
        pairwise_drift = max(
            abs(actual_matrix[row][col] - reference_matrix[row][col])
            for row in range(len(actual_matrix))
            for col in range(len(actual_matrix[row]))
        )
        if pairwise_drift > EMBED_PAIRWISE_DRIFT_THRESHOLD:
            issues.append(
                f"pairwise similarity drift above threshold: {pairwise_drift:.6f} > {EMBED_PAIRWISE_DRIFT_THRESHOLD:.1e}"
            )

    passed = not issues and all(item["passed"] for item in item_reports)
    return {
        "id": case["id"],
        "endpoint": "embedding",
        "request": request_body,
        "hf_reference": {
            **reference,
            "pairwise_similarity_matrix": reference_matrix,
        },
        "prelude_response": response,
        "metrics": {
            "usage_prompt_tokens_match": response["usage"]["prompt_tokens"]
            == reference["prompt_tokens"],
            "pairwise_similarity_matrix": actual_matrix,
            "pairwise_similarity_max_abs_diff": pairwise_drift,
            "items": item_reports,
        },
        "issues": issues,
        "passed": passed,
    }


def selected_endpoints(endpoint_arg: str) -> list[str]:
    if endpoint_arg == "all":
        return ["completion", "classify", "embedding"]
    return [endpoint_arg]


def endpoint_model(endpoint: str, args: argparse.Namespace) -> str:
    if endpoint == "completion":
        return args.completion_model
    if endpoint == "classify":
        return args.classify_model
    if endpoint == "embedding":
        return args.embedding_model
    raise ValueError(f"unknown endpoint: {endpoint}")


def close_reference_context(
    endpoint: str, model_id: str, runtime: ReferenceRuntime
) -> None:
    if endpoint == "completion":
        close_completion_reference_context(model_id, runtime)
    elif endpoint == "classify":
        close_classify_reference_context(model_id, runtime)
    elif endpoint == "embedding":
        close_embedding_reference_context(model_id, runtime)
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")


def evaluate_endpoint(
    endpoint: str,
    cases: list[dict[str, Any]],
    model_id: str,
    args: argparse.Namespace,
    runtime: ReferenceRuntime,
) -> dict[str, Any]:
    print(f"\n[{endpoint}] model={model_id}")
    server: ManagedServer | None = None
    endpoint_report: dict[str, Any] = {
        "model": model_id,
        "cases": [],
        "passed": False,
        "case_count": len(cases),
        "passed_cases": 0,
        "server_url": args.server_url
        if args.server_url and args.endpoint != "all"
        else None,
    }

    try:
        if args.server_url:
            url = args.server_url
            if not wait_for_server(url, timeout=5):
                raise RuntimeError(f"server is not reachable at {url}")
        else:
            server = start_server(args.binary, model_id, args.port, endpoint)
            url = server.url
            endpoint_report["server_url"] = url

        for case in cases:
            print(f"  - {case['id']} ... ", end="", flush=True)
            try:
                if endpoint == "completion":
                    case_report = evaluate_completion_case(case, model_id, runtime, url)
                elif endpoint == "classify":
                    case_report = evaluate_classify_case(case, model_id, runtime, url)
                elif endpoint == "embedding":
                    case_report = evaluate_embedding_case(case, model_id, runtime, url)
                else:
                    raise ValueError(f"unsupported endpoint: {endpoint}")
            except Exception as exc:
                case_report = {
                    "id": case["id"],
                    "endpoint": endpoint,
                    "request": None,
                    "hf_reference": None,
                    "prelude_response": None,
                    "metrics": {},
                    "issues": [str(exc)],
                    "passed": False,
                    "error": str(exc),
                }
            endpoint_report["cases"].append(case_report)
            if case_report["passed"]:
                endpoint_report["passed_cases"] += 1
                print("PASS")
            else:
                print("FAIL")
                for issue in case_report["issues"]:
                    print(f"      {issue}")

        endpoint_report["passed"] = (
            endpoint_report["passed_cases"] == endpoint_report["case_count"]
        )
        return endpoint_report
    finally:
        close_server(server)
        close_reference_context(endpoint, model_id, runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark-guide endpoint accuracy suite"
    )
    parser.add_argument(
        "--endpoint",
        choices=["completion", "classify", "embedding", "all"],
        default="all",
        help="Endpoint family to test",
    )
    parser.add_argument(
        "--binary",
        default=DEFAULT_BINARY,
        help="Path to prelude-server binary for auto-start mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for auto-started Prelude",
    )
    parser.add_argument(
        "--output",
        help="Report path (default: temp/accuracy/benchmark-guide-endpoint-accuracy-<timestamp>.json)",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="Path to the cases JSON file",
    )
    parser.add_argument(
        "--completion-model",
        default=DEFAULT_COMPLETION_MODEL,
        help="HF repo ID or local path for completion testing",
    )
    parser.add_argument(
        "--classify-model",
        default=DEFAULT_CLASSIFY_MODEL,
        help="HF repo ID or local path for classification testing",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="HF repo ID or local path for embedding testing",
    )
    parser.add_argument(
        "--server-url",
        help="Pre-started Prelude URL. Valid only with a single --endpoint value.",
    )
    args = parser.parse_args()
    if args.server_url and args.endpoint == "all":
        parser.error(
            "--server-url may only be used with a single endpoint, not --endpoint all"
        )
    return args


def main() -> int:
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output_path()
    cases_path = Path(args.cases)
    runtime = resolve_reference_runtime()

    with open(cases_path, "r", encoding="utf-8") as handle:
        case_matrix = json.load(handle)

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "script": str(SCRIPT_DIR / "test_endpoint_accuracy.py"),
        "cases_file": str(cases_path),
        "selected_endpoint": args.endpoint,
        "models": {
            "completion": args.completion_model,
            "classify": args.classify_model,
            "embedding": args.embedding_model,
        },
        "reference_runtime": {
            "device": runtime.device_label,
            "dtype": runtime.dtype_label,
        },
        "thresholds": {
            "completion_top_k_logprobs": TOP_K_LOGPROBS,
            "classify_cosine_similarity": CLASSIFY_COSINE_THRESHOLD,
            "embedding_cosine_similarity": EMBED_COSINE_THRESHOLD,
            "embedding_norm_drift": EMBED_NORM_DRIFT_THRESHOLD,
            "embedding_pairwise_similarity_drift": EMBED_PAIRWISE_DRIFT_THRESHOLD,
        },
        "notes": {
            "benchmark_guide_classifier": "The suite defaults to the public classifier model tomaarsen/Qwen3-Reranker-0.6B-seq-cls.",
            "embedding_reference": "Uses sentence-transformers semantics from repo artifacts (pooling + ordered Dense layers + optional normalization), not the repo's raw last-token shortcut.",
        },
        "endpoints": {},
        "overall": {
            "passed": False,
            "endpoint_count": 0,
            "passed_endpoints": 0,
            "case_count": 0,
            "passed_cases": 0,
        },
    }

    exit_code = 0
    try:
        for endpoint in selected_endpoints(args.endpoint):
            model_id = endpoint_model(endpoint, args)
            endpoint_cases = case_matrix[endpoint]
            try:
                endpoint_report = evaluate_endpoint(
                    endpoint, endpoint_cases, model_id, args, runtime
                )
            except Exception as exc:
                endpoint_report = {
                    "model": model_id,
                    "passed": False,
                    "case_count": len(endpoint_cases),
                    "passed_cases": 0,
                    "cases": [],
                    "error": str(exc),
                }
                print(f"[{endpoint}] ERROR: {exc}")
            report["endpoints"][endpoint] = endpoint_report
            report["overall"]["endpoint_count"] += 1
            report["overall"]["case_count"] += endpoint_report.get("case_count", 0)
            report["overall"]["passed_cases"] += endpoint_report.get("passed_cases", 0)
            if endpoint_report.get("passed"):
                report["overall"]["passed_endpoints"] += 1
            else:
                exit_code = 1

        report["overall"]["passed"] = (
            report["overall"]["passed_endpoints"] == report["overall"]["endpoint_count"]
        )
    finally:
        save_report(output_path, report)

    print("\nSummary")
    for endpoint, endpoint_report in report["endpoints"].items():
        passed_cases = endpoint_report.get("passed_cases", 0)
        case_count = endpoint_report.get("case_count", 0)
        status = "PASS" if endpoint_report.get("passed") else "FAIL"
        print(f"  {endpoint:10s} {status} {passed_cases}/{case_count}")
        if endpoint_report.get("error"):
            print(f"    error: {endpoint_report['error']}")

    if not report["overall"]["passed"]:
        print(
            f"\nOverall: FAIL {report['overall']['passed_cases']}/{report['overall']['case_count']} cases passed"
        )
    else:
        print(
            f"\nOverall: PASS {report['overall']['passed_cases']}/{report['overall']['case_count']} cases passed"
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
