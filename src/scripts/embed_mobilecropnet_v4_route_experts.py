#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import SUBJECT_MODE_COUNT, SUBJECT_MODE_VOCAB  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 checkpoint에 route expert를 내장한다.")
    parser.add_argument("--mobilecropnet_checkpoint", required=True, type=Path)
    parser.add_argument("--route_expert_checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--route_expert_weights", default=None)
    parser.add_argument("--output_checkpoint", required=True, type=Path)
    parser.add_argument("--note", default="")
    return parser


def _parse_weights(raw: str | None, count: int) -> list[float]:
    if not raw:
        return [1.0 for _ in range(count)]
    weights = [float(item) for item in str(raw).split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError(f"route_expert_weights count mismatch: {len(weights)} != {count}")
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("at least one route expert weight must be positive")
    return weights


def _load_route_expert_payload(path: Path, *, weight: float) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    config = dict(ckpt.get("config") or {})
    labels = list(config.get("labels") or SUBJECT_MODE_VOCAB)
    if labels != list(SUBJECT_MODE_VOCAB):
        raise ValueError(f"route expert labels do not match SUBJECT_MODE_VOCAB: {path}")
    state = ckpt.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"route expert checkpoint has no model_state: {path}")
    if len(labels) != SUBJECT_MODE_COUNT:
        raise ValueError(f"route expert label count mismatch: {len(labels)} != {SUBJECT_MODE_COUNT}")
    return {
        "source_checkpoint": str(path),
        "weight": float(weight),
        "config": config,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": ckpt.get("metrics"),
        "model_state": state,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "model_state"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights = _parse_weights(args.route_expert_weights, len(args.route_expert_checkpoints))
    mobile_ckpt = torch.load(args.mobilecropnet_checkpoint, map_location="cpu")
    embedded = [
        _load_route_expert_payload(path, weight=weight)
        for path, weight in zip(args.route_expert_checkpoints, weights)
    ]
    mobile_ckpt["embedded_route_experts"] = embedded
    mobile_ckpt["embedded_route_expert_summary"] = {
        "source_mobilecropnet_checkpoint": str(args.mobilecropnet_checkpoint),
        "route_expert_checkpoints": [str(path) for path in args.route_expert_checkpoints],
        "route_expert_weights": weights,
        "note": str(args.note or ""),
    }
    train_config = dict(mobile_ckpt.get("train_config") or {})
    train_config["embedded_route_expert_summary"] = _json_safe(mobile_ckpt["embedded_route_expert_summary"])
    mobile_ckpt["train_config"] = train_config
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mobile_ckpt, args.output_checkpoint)
    summary = {
        "state": "completed",
        "output_checkpoint": str(args.output_checkpoint),
        "embedded_route_expert_summary": mobile_ckpt["embedded_route_expert_summary"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
