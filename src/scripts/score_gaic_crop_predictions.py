#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet.eval_utils import annotation_candidates_for_record, box_iou_xyxy, load_label_crop_records, mean, safe_float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score arbitrary crop prediction JSONL against GAIC-like positive labels.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--label_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--id_field", default="image_id")
    parser.add_argument("--target_ar_field", default="target_ar")
    parser.add_argument("--bbox_field", default="bbox_norm_xyxy", help="Dot path, e.g. bbox_norm_xyxy or top_candidate.bbox_norm_xyxy")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dot_get(row: dict[str, Any], field: str) -> Any:
    cur: Any = row
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _norm_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    vals = [max(0.0, min(1.0, safe_float(v))) for v in value[:4]]
    x1, y1, x2, y2 = vals
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _build_label_index(label_json: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = load_label_crop_records(label_json)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        image = record.image
        image_id = str(image.get("sstk_image_id", image.get("id")))
        target_ar = str(image.get("target_ar", "FREE"))
        candidates = annotation_candidates_for_record(
            record.annotations,
            width=int(image["width"]),
            height=int(image["height"]),
        )
        positives = [cand for cand in candidates if int(cand.get("gt_flag", 0)) == 1]
        best_positive = max(positives, key=lambda cand: safe_float(cand.get("label_score")), default=None)
        entry = {
            "image": image,
            "target_ar": target_ar,
            "candidates": candidates,
            "positives": positives,
            "best_positive": best_positive,
        }
        index[(image_id, target_ar)] = entry
        index[(str(image.get("id")), target_ar)] = entry
        index[(str(image.get("file_name")), target_ar)] = entry
    return index


def _score_one(pred: dict[str, Any], label_entry: dict[str, Any], box: list[float]) -> dict[str, Any]:
    candidates = label_entry["candidates"]
    positives = label_entry["positives"]
    best_positive = label_entry["best_positive"]
    best_positive_box = best_positive.get("bbox_norm_xyxy") if isinstance(best_positive, dict) else None
    nearest_candidate = max(candidates, key=lambda cand: box_iou_xyxy(box, cand.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])), default=None)
    nearest_positive = max(positives, key=lambda cand: box_iou_xyxy(box, cand.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])), default=None)
    iou_best = box_iou_xyxy(box, best_positive_box) if best_positive_box else 0.0
    iou_any_pos = box_iou_xyxy(box, nearest_positive.get("bbox_norm_xyxy")) if isinstance(nearest_positive, dict) else 0.0
    nearest_score = safe_float(nearest_candidate.get("label_score"), 0.0) if isinstance(nearest_candidate, dict) else 0.0
    nearest_positive_score = safe_float(nearest_positive.get("label_score"), 0.0) if isinstance(nearest_positive, dict) else 0.0
    oracle_score = safe_float(best_positive.get("label_score"), 0.0) if isinstance(best_positive, dict) else 0.0
    return {
        "image_id": str(pred.get("image_id", "")),
        "target_ar": str(label_entry["target_ar"]),
        "bbox_norm_xyxy": box,
        "model_score": pred.get("score", pred.get("model_score")),
        "iou_to_best_positive": iou_best,
        "max_iou_to_positive": iou_any_pos,
        "iou_to_best_positive_ge_0_5": float(iou_best >= 0.5),
        "iou_to_best_positive_ge_0_7": float(iou_best >= 0.7),
        "max_iou_to_positive_ge_0_5": float(iou_any_pos >= 0.5),
        "max_iou_to_positive_ge_0_7": float(iou_any_pos >= 0.7),
        "nearest_label_score": nearest_score,
        "nearest_positive_score": nearest_positive_score,
        "oracle_label_score": oracle_score,
        "nearest_score_regret": max(0.0, oracle_score - nearest_score),
        "nearest_positive_score_regret": max(0.0, oracle_score - nearest_positive_score),
        "nearest_candidate_id": nearest_candidate.get("candidate_id") if isinstance(nearest_candidate, dict) else "",
        "nearest_positive_candidate_id": nearest_positive.get("candidate_id") if isinstance(nearest_positive, dict) else "",
        "best_positive_candidate_id": best_positive.get("candidate_id") if isinstance(best_positive, dict) else "",
        "source_prediction": pred,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "iou_to_best_positive",
        "max_iou_to_positive",
        "iou_to_best_positive_ge_0_5",
        "iou_to_best_positive_ge_0_7",
        "max_iou_to_positive_ge_0_5",
        "max_iou_to_positive_ge_0_7",
        "nearest_label_score",
        "nearest_positive_score",
        "oracle_label_score",
        "nearest_score_regret",
        "nearest_positive_score_regret",
    ]
    return {key: mean([safe_float(row.get(key), 0.0) for row in rows]) for key in keys}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GAIC Crop Prediction Scoring",
        "",
        f"- model_name: `{summary.get('model_name', '')}`",
        f"- prediction_count: {summary['prediction_count']}",
        f"- matched_count: {summary['matched_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in sorted(summary["metrics"].items()):
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.extend(["", "## By Target AR", "", "| target_ar | count | max_iou_to_positive | nearest_positive_score |", "| --- | ---: | ---: | ---: |"])
    for ar, row in sorted(summary["by_target_ar"].items()):
        lines.append(
            f"| {ar} | {int(row['count'])} | {float(row.get('max_iou_to_positive', 0.0)):.6f} | {float(row.get('nearest_positive_score', 0.0)):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_json = args.label_json
    labels = _build_label_index(label_json)
    predictions = _read_jsonl(args.predictions_jsonl)
    scored: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for pred in predictions:
        image_id = str(_dot_get(pred, args.id_field) or "")
        target_ar = str(_dot_get(pred, args.target_ar_field) or "FREE")
        box = _norm_box(_dot_get(pred, args.bbox_field))
        label_entry = labels.get((image_id, target_ar))
        if label_entry is None or box is None:
            skipped.append({"image_id": image_id, "target_ar": target_ar, "reason": "missing_label_or_box", "prediction": pred})
            continue
        scored.append(_score_one(pred, label_entry, box))

    by_ar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_ar[str(row["target_ar"])].append(row)
    by_target_ar = {}
    for ar, rows in by_ar.items():
        ar_summary = _summarize(rows)
        ar_summary["count"] = len(rows)
        by_target_ar[ar] = ar_summary
    summary = {
        "model_name": args.model_name,
        "predictions_jsonl": str(args.predictions_jsonl),
        "label_json": str(label_json),
        "prediction_count": len(predictions),
        "matched_count": len(scored),
        "skipped_count": len(skipped),
        "metrics": _summarize(scored),
        "by_target_ar": by_target_ar,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "scored_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if skipped:
        with (args.output_dir / "skipped_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in skipped:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_report(args.output_dir / "PREDICTION_EVAL_REPORT.md", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
