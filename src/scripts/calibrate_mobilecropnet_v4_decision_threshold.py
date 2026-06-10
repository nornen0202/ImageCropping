#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a MobileCropNet v4 checkpoint and fix decision_source_logit_threshold from validation sweep metrics."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run_dir", required=True, type=Path, help="Training run directory containing summary.json/metrics.json.")
    parser.add_argument("--output_checkpoint", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("best_balanced", "best_min_recall", "explicit"),
        default="best_balanced",
        help="Which threshold sweep point to materialize in the checkpoint config.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Required when --mode explicit is used.")
    parser.add_argument(
        "--runtime_use_decision_source_action_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record whether deployment inference should use decision_source_logit as the baseline-vs-crop action gate.",
    )
    parser.add_argument("--note", default="")
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_row(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        row = summary.get("best_row") or summary.get("last_row")
        if isinstance(row, dict):
            return row
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        rows = _load_json(metrics_path)
        if isinstance(rows, list) and rows:
            return max(rows, key=lambda row: float(row.get("selection_score", 0.0) or 0.0))
    raise FileNotFoundError(f"no usable summary.json or metrics.json found under {run_dir}")


def _float_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"required metric missing: {key}")
    return float(value)


def _threshold_from_row(row: dict[str, Any], *, mode: str, explicit_threshold: float | None) -> tuple[float, str]:
    if mode == "explicit":
        if explicit_threshold is None:
            raise ValueError("--threshold is required when --mode explicit is used")
        return float(explicit_threshold), "explicit"
    val = row.get("val")
    if not isinstance(val, dict):
        raise KeyError("best row does not contain a val metrics dictionary")
    if mode == "best_min_recall":
        return _float_metric(val, "decision_source_threshold_sweep_best_min_recall_threshold"), (
            "decision_source_threshold_sweep_best_min_recall_threshold"
        )
    return _float_metric(val, "decision_source_threshold_sweep_best_threshold"), (
        "decision_source_threshold_sweep_best_threshold"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    row = _best_row(args.run_dir)
    threshold, metric_key = _threshold_from_row(row, mode=args.mode, explicit_threshold=args.threshold)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be a dict payload: {args.checkpoint}")
    model_config = dict(ckpt.get("model_config") or {})
    old_threshold = float(model_config.get("decision_source_logit_threshold", 0.0) or 0.0)
    model_config["decision_source_logit_threshold"] = float(threshold)
    ckpt["model_config"] = model_config

    calibration = {
        "source_checkpoint": str(args.checkpoint),
        "source_run_dir": str(args.run_dir),
        "mode": str(args.mode),
        "metric_key": metric_key,
        "old_decision_source_logit_threshold": old_threshold,
        "decision_source_logit_threshold": float(threshold),
        "runtime_use_decision_source_action_gate": bool(args.runtime_use_decision_source_action_gate),
        "best_epoch": row.get("epoch"),
        "best_selection_score": row.get("selection_score"),
        "note": str(args.note or ""),
        "selected_val_metrics": {
            key: (row.get("val") or {}).get(key)
            for key in (
                "baseline_preserve_top1_hit",
                "crop_action_top1_hit",
                "top_return_hit",
                "decision_conditioned_baseline_preserve_top1_hit",
                "decision_conditioned_crop_action_top1_hit",
                "decision_conditioned_top_return_hit",
                "decision_source_threshold_sweep_best_balanced_acc",
                "decision_source_threshold_sweep_best_base_recall",
                "decision_source_threshold_sweep_best_crop_recall",
                "decision_source_threshold_sweep_top_return_hit",
                "decision_source_threshold_sweep_best_min_recall",
                "decision_source_threshold_sweep_best_min_recall_top_return_hit",
            )
        },
    }
    ckpt["decision_source_threshold_calibration"] = calibration
    train_config = dict(ckpt.get("train_config") or {})
    train_config["decision_source_threshold_calibration"] = _json_safe(calibration)
    train_config["runtime_use_decision_source_action_gate"] = bool(args.runtime_use_decision_source_action_gate)
    ckpt["train_config"] = train_config
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output_checkpoint)
    summary = {
        "state": "completed",
        "output_checkpoint": str(args.output_checkpoint),
        "decision_source_threshold_calibration": calibration,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
