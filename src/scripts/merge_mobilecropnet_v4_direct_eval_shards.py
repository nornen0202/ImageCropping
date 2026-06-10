#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.eval_utils import summarize_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 Product-AR direct eval shard 결과를 병합한다.")
    parser.add_argument("--shard_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_rows", type=int, default=0)
    parser.add_argument("--allow_duplicate_same_payload", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            yield row


def prediction_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))


def main() -> int:
    args = parse_args()
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    shard_summaries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    by_target_ar: dict[str, list[dict[str, float]]] = defaultdict(list)
    duplicate_count = 0
    input_row_count = 0
    missing_shards: list[str] = []
    first_metrics: dict[str, Any] | None = None

    for shard_dir in args.shard_dirs:
        pred_path = shard_dir / "predictions.jsonl"
        metrics_path = shard_dir / "metrics.json"
        progress_path = shard_dir / "progress.json"
        shard_summary: dict[str, Any] = {
            "shard_dir": str(shard_dir),
            "predictions_jsonl": str(pred_path),
            "metrics_json": str(metrics_path),
            "progress_json": str(progress_path),
            "exists": pred_path.exists(),
            "row_count": 0,
        }
        if metrics_path.exists():
            shard_metrics = read_json(metrics_path)
            shard_summary["metrics"] = shard_metrics
            if first_metrics is None:
                first_metrics = shard_metrics
        if progress_path.exists():
            shard_summary["progress"] = read_json(progress_path)
        if not pred_path.exists():
            missing_shards.append(str(shard_dir))
            shard_summaries.append(shard_summary)
            continue
        for row in read_jsonl(pred_path):
            input_row_count += 1
            shard_summary["row_count"] += 1
            key = prediction_key(row)
            previous = rows_by_key.get(key)
            if previous is not None:
                duplicate_count += 1
                if not args.allow_duplicate_same_payload or previous != row:
                    raise ValueError(f"duplicate direct eval row with different payload: {key}")
                continue
            rows_by_key[key] = row
        shard_summaries.append(shard_summary)

    if missing_shards:
        raise SystemExit(f"missing shard prediction files: {missing_shards}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_key[key] for key in sorted(rows_by_key.keys())]
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                metric_row = {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                metric_rows.append(metric_row)
                by_target_ar[str(row.get("target_ar", "FREE"))].append(metric_row)

    merged_count = len(rows)
    expected = int(args.expected_rows)
    complete = expected <= 0 or merged_count == expected
    base = first_metrics or {}
    summary = {
        "checkpoint": base.get("checkpoint"),
        "route_expert_checkpoint": base.get("route_expert_checkpoint"),
        "route_expert_checkpoints": base.get("route_expert_checkpoints", []),
        "route_expert_weights": base.get("route_expert_weights"),
        "route_expert_source": base.get("route_expert_source"),
        "embedded_route_expert_count": base.get("embedded_route_expert_count"),
        "eval_jsonl": base.get("eval_jsonl"),
        "selection_policy": base.get("selection_policy"),
        "proposal_top_m": base.get("proposal_top_m"),
        "runtime_subject_prior_mode": base.get("runtime_subject_prior_mode"),
        "exact_target_ar_postprocess": base.get("exact_target_ar_postprocess"),
        "enforce_baseline_decision_gate": base.get("enforce_baseline_decision_gate"),
        "baseline_decision_confidence_threshold": base.get("baseline_decision_confidence_threshold"),
        "baseline_score_margin": base.get("baseline_score_margin"),
        "subject_valid_threshold": base.get("subject_valid_threshold"),
        "subject_valid_policy": base.get("subject_valid_policy"),
        "subject_box_target_source": base.get("subject_box_target_source"),
        "image_count": int(merged_count),
        "expected_rows": expected,
        "complete": bool(complete),
        "input_row_count": int(input_row_count),
        "duplicate_count": int(duplicate_count),
        "metrics": summarize_metrics(metric_rows),
        "by_target_ar": {
            key: {"image_count": len(value), **summarize_metrics(value)}
            for key, value in sorted(by_target_ar.items())
        },
        "merge": {
            "state": "completed" if complete else "incomplete",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_dir": str(args.output_dir),
            "shards": shard_summaries,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary["merge"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not complete:
        raise SystemExit(f"merged rows {merged_count} != expected_rows {expected}")
    print(json.dumps(summary["merge"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
