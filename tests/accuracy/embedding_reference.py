#!/usr/bin/env python3
"""Sentence-transformers embedding reference helpers for accuracy tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers.utils.hub import cached_file

IDENTITY_ACTIVATIONS = {
    "identity",
    "Identity",
    "torch.nn.Identity",
    "torch.nn.modules.linear.Identity",
}


@dataclass
class DenseLayerSpec:
    name: str
    path: str
    in_features: int
    out_features: int
    activation: str
    weight: torch.Tensor
    bias: torch.Tensor | None


@dataclass
class EmbeddingSemantics:
    pooling: str
    normalize: bool
    modules: list[str]
    dense_layers: list[DenseLayerSpec]


def _resolve_model_artifact(model_id: str, relative_path: str) -> Path:
    local_root = Path(model_id)
    if local_root.exists():
        artifact_path = local_root / relative_path
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"missing local embedding artifact: {artifact_path}"
            )
        return artifact_path

    resolved = cached_file(model_id, relative_path)
    if resolved is None:
        raise FileNotFoundError(
            f"failed to resolve embedding artifact {relative_path} for {model_id}"
        )
    return Path(resolved)


def _load_json_artifact(model_id: str, relative_path: str) -> dict:
    return json.loads(
        _resolve_model_artifact(model_id, relative_path).read_text(encoding="utf-8")
    )


def _module_idx(entry: dict) -> int:
    try:
        return int(entry.get("idx", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid sentence-transformers module idx: {entry!r}") from exc


def _parse_pooling(model_id: str, module_path: str) -> str:
    pooling_cfg = _load_json_artifact(model_id, f"{module_path}/config.json")
    if pooling_cfg.get("pooling_mode_lasttoken"):
        return "lasttoken"
    if pooling_cfg.get("pooling_mode_mean_tokens"):
        return "mean"
    if pooling_cfg.get("pooling_mode_cls_token"):
        return "cls"
    raise RuntimeError(
        f"unsupported embedding pooling for {model_id}: "
        f"{json.dumps(pooling_cfg, sort_keys=True)}"
    )


def _parse_dense_layer(
    model_id: str,
    module_name: str,
    module_path: str,
    device: torch.device | None,
) -> DenseLayerSpec:
    dense_cfg = _load_json_artifact(model_id, f"{module_path}/config.json")
    activation = str(
        dense_cfg.get("activation_function") or dense_cfg.get("activation") or "Identity"
    )
    if activation not in IDENTITY_ACTIVATIONS:
        raise RuntimeError(
            f"unsupported dense activation for {model_id}/{module_path}: {activation}"
        )

    tensors = load_file(str(_resolve_model_artifact(model_id, f"{module_path}/model.safetensors")))
    weight = tensors.get("linear.weight")
    if weight is None:
        raise RuntimeError(
            f"missing linear.weight for embedding dense module {model_id}/{module_path}"
        )

    in_features = int(dense_cfg["in_features"])
    out_features = int(dense_cfg["out_features"])
    if tuple(weight.shape) != (out_features, in_features):
        raise RuntimeError(
            f"dense weight shape mismatch for {model_id}/{module_path}: "
            f"{tuple(weight.shape)} != {(out_features, in_features)}"
        )

    bias = tensors.get("linear.bias")
    if device is not None:
        weight = weight.to(device=device, dtype=torch.float32)
        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)
    else:
        weight = weight.to(dtype=torch.float32)
        if bias is not None:
            bias = bias.to(dtype=torch.float32)

    return DenseLayerSpec(
        name=module_name,
        path=module_path,
        in_features=in_features,
        out_features=out_features,
        activation=activation,
        weight=weight,
        bias=bias,
    )


def load_embedding_semantics(
    model_id: str,
    *,
    device: torch.device | None = None,
) -> EmbeddingSemantics:
    try:
        modules = _load_json_artifact(model_id, "modules.json")
    except FileNotFoundError:
        return EmbeddingSemantics(
            pooling="lasttoken",
            normalize=False,
            modules=[],
            dense_layers=[],
        )

    modules = sorted(modules, key=_module_idx)
    module_types = [str(entry["type"]) for entry in modules]
    pooling = "lasttoken"
    normalize = False
    dense_layers: list[DenseLayerSpec] = []

    for entry in modules:
        module_type = str(entry["type"])
        if normalize and not module_type.endswith(".Normalize"):
            raise RuntimeError(
                "unsupported sentence-transformers module order: "
                "Normalize must be the final module"
            )
        module_name = str(entry.get("name", entry.get("idx", "")))
        module_path = str(entry.get("path") or "")
        if module_type.endswith(".Transformer"):
            continue
        if module_type.endswith(".Pooling"):
            pooling = _parse_pooling(model_id, module_path)
            continue
        if module_type.endswith(".Dense"):
            dense_layers.append(
                _parse_dense_layer(model_id, module_name, module_path, device)
            )
            continue
        if module_type.endswith(".Normalize"):
            if normalize:
                raise RuntimeError(
                    "unsupported sentence-transformers module order: "
                    "multiple Normalize modules are not supported"
                )
            normalize = True
            continue
        raise RuntimeError(
            f"unsupported sentence-transformers module for {model_id}: {module_type}"
        )

    return EmbeddingSemantics(
        pooling=pooling,
        normalize=normalize,
        modules=module_types,
        dense_layers=dense_layers,
    )


def pool_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    semantics: EmbeddingSemantics,
) -> torch.Tensor:
    mask = attention_mask.to(hidden_states.device)
    if semantics.pooling == "lasttoken":
        lengths = mask.sum(dim=1) - 1
        return hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device), lengths
        ]
    if semantics.pooling == "mean":
        expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(
            1.0
        )
    if semantics.pooling == "cls":
        return hidden_states[:, 0, :]
    raise RuntimeError(f"unsupported pooling mode: {semantics.pooling}")


def apply_embedding_modules(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    semantics: EmbeddingSemantics,
) -> torch.Tensor:
    embeddings = pool_embeddings(hidden_states, attention_mask, semantics)
    for layer in semantics.dense_layers:
        weight = layer.weight
        bias = layer.bias
        if weight.device != embeddings.device:
            weight = weight.to(device=embeddings.device)
        if weight.dtype != embeddings.dtype:
            weight = weight.to(dtype=embeddings.dtype)
        if bias is not None and bias.device != embeddings.device:
            bias = bias.to(device=embeddings.device)
        if bias is not None and bias.dtype != embeddings.dtype:
            bias = bias.to(dtype=embeddings.dtype)
        embeddings = F.linear(embeddings, weight, bias)

    if semantics.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings
