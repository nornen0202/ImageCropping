#!/usr/bin/env python3
"""Backfill MobileCropNet v4 train-gate decisions from durable summaries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "route_balanced_acc": 0.52,
    "subject_box_iou": 0.50,
    "subject_box_valid_best_balanced_acc": 0.55,
    "proposal_recall_0_5": 0.93,
    "generated_align_top1_hit": 0.90,
}


def _truthy(value: Any) -> bool:
    return str(value or "0").strip().lower() in {"1", "true", "yes", "on"}


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _pick_surface_key(
    suffix: str,
    *,
    val: dict[str, Any],
    use_source_gate: bool,
    use_source_mixture: bool,
    use_decision_conditioned: bool,
) -> str:
    source_gate_key = f"source_gate_{suffix}"
    if use_source_gate and val.get(source_gate_key) is not None:
        return source_gate_key
    source_key = f"source_mixture_{suffix}"
    if use_source_mixture and val.get(source_key) is not None:
        return source_key
    conditioned_key = f"decision_conditioned_{suffix}"
    if use_decision_conditioned and val.get(conditioned_key) is not None:
        return conditioned_key
    return suffix


def build_decision(run_dir: Path, *, manual_backfill: bool) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    config_path = run_dir / "quality_relaxed_joint_launch_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary.get("best_row") or summary.get("last_row") or {}
    val = row.get("val") or {}
    if not isinstance(val, dict):
        raise ValueError(f"summary row has no val metrics: {summary_path}")

    thresholds = {
        "route_balanced_acc": _float_config(config, "train_gate_route_balanced_min", DEFAULT_THRESHOLDS["route_balanced_acc"]),
        "subject_box_iou": _float_config(config, "train_gate_subject_iou_min", DEFAULT_THRESHOLDS["subject_box_iou"]),
        "subject_box_valid_best_balanced_acc": _float_config(
            config,
            "train_gate_subject_valid_balanced_min",
            DEFAULT_THRESHOLDS["subject_box_valid_best_balanced_acc"],
        ),
        "proposal_recall_0_5": _float_config(config, "train_gate_proposal_recall_min", DEFAULT_THRESHOLDS["proposal_recall_0_5"]),
        "generated_align_top1_hit": _float_config(
            config,
            "train_gate_generated_align_min",
            DEFAULT_THRESHOLDS["generated_align_top1_hit"],
        ),
    }

    return_target_mode = str(config.get("return_target_mode") or "score").lower()
    use_decision_conditioned = _truthy(config.get("return_score_decision_conditioned"))
    use_source_mixture = _truthy(config.get("return_score_source_mixture"))
    use_source_gate = _truthy(config.get("return_score_source_gate"))

    crop_action_key = _pick_surface_key(
        "crop_action_top1_hit",
        val=val,
        use_source_gate=use_source_gate,
        use_source_mixture=use_source_mixture,
        use_decision_conditioned=use_decision_conditioned,
    )
    top_return_key = _pick_surface_key(
        "top_return_hit",
        val=val,
        use_source_gate=use_source_gate,
        use_source_mixture=use_source_mixture,
        use_decision_conditioned=use_decision_conditioned,
    )
    baseline_key = _pick_surface_key(
        "baseline_preserve_top1_hit",
        val=val,
        use_source_gate=use_source_gate,
        use_source_mixture=use_source_mixture,
        use_decision_conditioned=use_decision_conditioned,
    )
    thresholds[crop_action_key] = _float_config(config, "train_gate_action_top1_min", 0.96)
    thresholds[top_return_key] = _float_config(config, "train_gate_top_return_min", 0.58)
    metrics = {key: val.get(key) for key in thresholds}

    baseline_target_rate = val.get("baseline_preserve_target_rate")
    raw_top_key = None
    raw_gate_mode = str(config.get("train_gate_raw_top_return_mode") or "auto").strip().lower()
    if return_target_mode in {"decision_source", "decision_consistent", "action_source"}:
        raw_top_key = _pick_surface_key(
            "raw_top_return_hit",
            val=val,
            use_source_gate=use_source_gate,
            use_source_mixture=use_source_mixture,
            use_decision_conditioned=use_decision_conditioned,
        )
        apply_raw_gate = val.get(raw_top_key) is not None
        if raw_gate_mode in {"0", "false", "no", "off", "never", "none", "disabled"}:
            apply_raw_gate = False
        elif raw_gate_mode in {"auto", "mixed", "nonempty_baseline"}:
            baseline_rate_min = _float_config(config, "train_gate_baseline_target_rate_min", 0.05)
            apply_raw_gate = apply_raw_gate and (
                baseline_target_rate is None or float(baseline_target_rate) >= baseline_rate_min
            )
        elif raw_gate_mode not in {"1", "true", "yes", "on", "always"}:
            raise ValueError(f"unsupported train_gate_raw_top_return_mode={raw_gate_mode!r}")
        if apply_raw_gate:
            thresholds[raw_top_key] = _float_config(config, "train_gate_top_return_min", 0.58)
            metrics[raw_top_key] = val.get(raw_top_key)

    baseline_rate_min = _float_config(config, "train_gate_baseline_target_rate_min", 0.05)
    if baseline_target_rate is not None and float(baseline_target_rate) >= baseline_rate_min:
        thresholds[baseline_key] = _float_config(config, "train_gate_baseline_preserve_min", 0.955)
        metrics[baseline_key] = val.get(baseline_key)
        metrics["baseline_preserve_target_rate"] = baseline_target_rate

    failures = {
        key: {"value": metrics.get(key), "threshold": threshold}
        for key, threshold in thresholds.items()
        if metrics.get(key) is None or float(metrics.get(key)) < threshold
    }
    passed = not failures
    state = "train_gate_passed" if passed else "excluded_train_gate"
    if manual_backfill:
        state = f"{state}_manual_backfill"
    return {
        "state": state,
        "run_name": config.get("run_name") or run_dir.name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": str(summary_path),
        "config": str(config_path) if config_path.exists() else None,
        "best_epoch": summary.get("best_epoch"),
        "best_selection_score": summary.get("best_selection_score"),
        "thresholds": thresholds,
        "metrics": metrics,
        "failures": failures,
        "return_target_mode": return_target_mode,
        "raw_top_return_gate_applied": bool(raw_top_key is not None and raw_top_key in thresholds),
        "raw_top_return_gate_mode": raw_gate_mode,
        "next_action": "run embedded no-prior direct/head/qualitative gate" if passed else "skip GPU gate and mark excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_manual_suffix", action="store_true")
    parser.add_argument("--write_early_stop", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    decision_path = run_dir / "train_gate_decision.json"
    if decision_path.exists() and not args.overwrite:
        raise FileExistsError(f"decision already exists: {decision_path}")
    payload = build_decision(run_dir, manual_backfill=not args.no_manual_suffix)
    decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / ".train_gate_pass").write_text(
        "1\n" if payload["state"].startswith("train_gate_passed") else "0\n",
        encoding="utf-8",
    )
    if args.write_early_stop and not payload["state"].startswith("train_gate_passed"):
        (run_dir / "early_stop_reason.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
