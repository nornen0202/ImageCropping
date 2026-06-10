#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import box_iou_xyxy, safe_float
from mobilecropnet_v4.eval_utils import mean, summarize_metrics, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MobileCropNet v4 predictions against replay baselines/oracles.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _best_positive(row: dict[str, Any]) -> dict[str, Any] | None:
    best = row.get("best_positive")
    if isinstance(best, dict):
        return best
    positives = row.get("positive_records") or []
    return max(positives, key=lambda item: safe_float(item.get("score")), default=None)


def _select_candidate(candidates: Sequence[dict[str, Any]], method: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    if method == "v4_model":
        return max(candidates, key=lambda c: safe_float(c.get("model_utility")))
    if method == "baseline_candidate":
        base = [c for c in candidates if safe_float(c.get("is_base")) > 0]
        return base[0] if base else candidates[0]
    if method == "teacher_score_oracle":
        return max(candidates, key=lambda c: safe_float(c.get("score_target")))
    if method == "positive_oracle":
        positives = [c for c in candidates if safe_float(c.get("positive_target")) > 0]
        return max(positives, key=lambda c: safe_float(c.get("score_target")), default=max(candidates, key=lambda c: safe_float(c.get("score_target"))))
    if method in {"proposal_top1", "proposal_top1_target_ar"}:
        return None
    raise ValueError(f"unknown method: {method}")


def _method_metrics(row: dict[str, Any], method: str) -> dict[str, float]:
    best = _best_positive(row)
    best_box = best.get("bbox_norm_xyxy") if isinstance(best, dict) else None
    candidates = row.get("candidates") or []
    if method in {"proposal_top1", "proposal_top1_target_ar"}:
        proposal = row.get("proposal_top1_target_ar") if method == "proposal_top1_target_ar" else (row.get("proposals") or [{}])[0]
        box = proposal.get("bbox_norm_xyxy") if isinstance(proposal, dict) else None
        return {
            "top1_iou_to_best_positive": box_iou_xyxy(box, best_box) if isinstance(box, list) and isinstance(best_box, list) else 0.0,
            "candidate_top1_hit": 0.0,
            "candidate_top1_exact_best": 0.0,
            "utility_regret": 0.0,
            "target_ar_log_error": safe_float(proposal.get("target_ar_log_error"), 0.0) if isinstance(proposal, dict) else 0.0,
            "target_ar_compatible": float(bool(proposal.get("target_ar_compatible", True))) if isinstance(proposal, dict) else 0.0,
        }
    chosen = _select_candidate(candidates, method)
    if chosen is None:
        return {}
    best_score = max([safe_float(c.get("score_target")) for c in candidates], default=0.0)
    chosen_score = safe_float(chosen.get("score_target"))
    best_candidate = max(candidates, key=lambda c: safe_float(c.get("score_target")), default={})
    return {
        "candidate_top1_hit": float(safe_float(chosen.get("positive_target")) > 0.0),
        "candidate_top1_exact_best": float(str(chosen.get("candidate_id")) == str(best_candidate.get("candidate_id"))),
        "top1_iou_to_best_positive": box_iou_xyxy(chosen.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), best_box) if isinstance(best_box, list) else 0.0,
        "chosen_label_score": chosen_score,
        "utility_regret": max(0.0, best_score - chosen_score),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# MobileCropNet v4 Method Comparison", "", f"- predictions_jsonl: `{summary['predictions_jsonl']}`", f"- image_count: {summary['image_count']}", "", "| method | top1_hit | exact_best | iou_to_best | utility_regret |", "| --- | ---: | ---: | ---: | ---: |"]
    for method, metrics in summary["methods"].items():
        lines.append(
            "| {method} | {hit:.6f} | {exact:.6f} | {iou:.6f} | {regret:.6f} |".format(
                method=method,
                hit=float(metrics.get("candidate_top1_hit", 0.0)),
                exact=float(metrics.get("candidate_top1_exact_best", 0.0)),
                iou=float(metrics.get("top1_iou_to_best_positive", 0.0)),
                regret=float(metrics.get("utility_regret", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `baseline_candidate` is the current label pipeline baseline crop, usually the full/max-area crop.",
            "- `teacher_score_oracle` is an upper bound within the replay candidate set, not a deployable cropper.",
            "- `proposal_top1` evaluates the learned proposal head raw top-1 without candidate replay ranking.",
            "- `proposal_top1_target_ar` evaluates the proposal selected after target-AR compatibility filtering.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(args.predictions_jsonl)
    methods = ["v4_model", "baseline_candidate", "teacher_score_oracle", "positive_oracle", "proposal_top1", "proposal_top1_target_ar"]
    detail_rows = []
    summary_methods = {}
    for method in methods:
        metric_rows = []
        for row in rows:
            metrics = _method_metrics(row, method)
            metric_rows.append(metrics)
            detail_rows.append({"image_id": row.get("image_id"), "target_ar": row.get("target_ar"), "method": method, "metrics": metrics})
        summary_methods[method] = summarize_metrics(metric_rows)
        summary_methods[method]["image_count"] = len(metric_rows)
    summary = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "image_count": len(rows),
        "methods": summary_methods,
        "method_order_by_top1_hit": sorted(methods, key=lambda m: mean([summary_methods[m].get("candidate_top1_hit", 0.0)]), reverse=True),
    }
    (args.output_dir / "comparison_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "comparison_details.jsonl", detail_rows)
    _write_report(args.output_dir / "COMPARISON_REPORT.md", summary)
    print(json.dumps(summary_methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
