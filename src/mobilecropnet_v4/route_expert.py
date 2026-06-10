from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .data import SUBJECT_MODE_COUNT, SUBJECT_MODE_VOCAB, load_image_tensor_letterbox


def _build_route_expert_model(
    *,
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> nn.Module:
    config = dict(config or {})
    backbone = str(config.get("backbone") or "")
    if not backbone:
        raise ValueError("route expert payload has no backbone config")
    labels = list(config.get("labels") or SUBJECT_MODE_VOCAB)
    if len(labels) != SUBJECT_MODE_COUNT:
        raise ValueError(f"route expert label count mismatch: got {len(labels)}, expected {SUBJECT_MODE_COUNT}")
    if labels != list(SUBJECT_MODE_VOCAB):
        raise ValueError(f"route expert labels do not match SUBJECT_MODE_VOCAB: {labels}")
    import timm

    model = timm.create_model(backbone, pretrained=False, num_classes=SUBJECT_MODE_COUNT)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_route_expert_checkpoint(checkpoint: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(Path(checkpoint), map_location="cpu")
    config = dict(ckpt.get("config") or {})
    state = ckpt.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"route expert checkpoint has no model_state: {checkpoint}")
    model = _build_route_expert_model(config=config, state=state, device=device)
    return model, config, ckpt


def load_embedded_route_experts(
    checkpoint_payload: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[list[tuple[nn.Module, dict[str, Any]]], list[float] | None]:
    payloads = checkpoint_payload.get("embedded_route_experts") or []
    if not isinstance(payloads, list):
        payloads = []
    experts: list[tuple[nn.Module, dict[str, Any]]] = []
    weights: list[float] = []
    for idx, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            continue
        config = dict(payload.get("config") or {})
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise ValueError(f"embedded route expert #{idx} has no model_state")
        model = _build_route_expert_model(config=config, state=state, device=device)
        experts.append((model, config))
        weights.append(float(payload.get("weight", 1.0)))
    return experts, weights if experts else None


def route_expert_logits_from_paths(
    model: nn.Module,
    config: dict[str, Any],
    image_paths: Sequence[str | Path],
    *,
    device: torch.device,
) -> torch.Tensor:
    input_size = int(config.get("input_size") or 384)
    image_mean = tuple(float(v) for v in config.get("image_mean", (0.485, 0.456, 0.406)))
    image_std = tuple(float(v) for v in config.get("image_std", (0.229, 0.224, 0.225)))
    tensors = [
        load_image_tensor_letterbox(Path(path), input_size, image_mean=image_mean, image_std=image_std)[0]
        for path in image_paths
    ]
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        return model(batch)


def route_expert_ensemble_logits_from_paths(
    experts: Sequence[tuple[nn.Module, dict[str, Any]]],
    image_paths: Sequence[str | Path],
    *,
    device: torch.device,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    if not experts:
        raise ValueError("route expert ensemble is empty")
    if weights is None:
        weights = [1.0 for _ in experts]
    if len(weights) != len(experts):
        raise ValueError(f"route expert weight count mismatch: {len(weights)} != {len(experts)}")
    logits = []
    normalized_weights = []
    for (model, config), weight in zip(experts, weights):
        w = float(weight)
        if w <= 0.0:
            continue
        logits.append(route_expert_logits_from_paths(model, config, image_paths, device=device) * w)
        normalized_weights.append(w)
    if not logits:
        raise ValueError("route expert ensemble has no positive weights")
    return torch.stack(logits, dim=0).sum(dim=0) / float(sum(normalized_weights))


def override_outputs_route_logits(outputs: dict[str, Any], route_logits: torch.Tensor) -> dict[str, Any]:
    outputs = dict(outputs)
    outputs["route_logits"] = route_logits
    outputs["route_fine_logits"] = route_logits
    return outputs
