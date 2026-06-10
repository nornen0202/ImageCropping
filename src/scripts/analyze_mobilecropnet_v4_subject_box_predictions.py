#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze MobileCropNet v4 subject-box confidence calibration from prediction JSONL.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--thresholds", default="0.05:0.95:0.01", help="start:end:step threshold sweep.")
    parser.add_argument("--target_negative_acc", type=float, default=0.70)
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _thresholds(spec: str) -> list[float]:
    parts = [float(part) for part in str(spec).split(":")]
    if len(parts) != 3:
        raise ValueError("--thresholds must be start:end:step")
    start, end, step = parts
    if step <= 0:
        raise ValueError("threshold step must be positive")
    values: list[float] = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(float(cur), 6))
        cur += step
    return values


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            subject = rec.get("subject_box") or {}
            predicted = subject.get("predicted") or {}
            conf = _safe_float(predicted.get("confidence"), 0.0)
            target_valid = _safe_float(subject.get("target_valid"), 0.0)
            iou = _safe_float(subject.get("iou_to_teacher"), 0.0)
            rows.append({"confidence": conf, "target_valid": 1.0 if target_valid >= 0.5 else 0.0, "iou_to_teacher": iou})
    return rows


def _metrics(rows: list[dict[str, float]], threshold: float) -> dict[str, float]:
    if not rows:
        return {}
    tp = fp = tn = fn = 0
    valid_ious: list[float] = []
    for row in rows:
        target = row["target_valid"] >= 0.5
        pred = row["confidence"] >= threshold
        if target:
            valid_ious.append(row["iou_to_teacher"])
        if pred and target:
            tp += 1
        elif pred and not target:
            fp += 1
        elif (not pred) and target:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    pos_total = tp + fn
    neg_total = tn + fp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, pos_total)
    specificity = tn / max(1, neg_total)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": float(threshold),
        "count": float(total),
        "positive_count": float(pos_total),
        "negative_count": float(neg_total),
        "acc": float((tp + tn) / max(1, total)),
        "precision": float(precision),
        "positive_recall": float(recall),
        "negative_acc": float(specificity),
        "balanced_acc": float(0.5 * (recall + specificity)),
        "f1": float(f1),
        "mean_confidence": float(sum(row["confidence"] for row in rows) / max(1, total)),
        "mean_iou_valid": float(sum(valid_ious) / max(1, len(valid_ious))),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    selected = payload.get("selected", {})
    fixed = payload.get("fixed_0_5", {})
    neg = payload.get("selected_target_negative_acc", {})
    lines = [
        "# Subject Box Confidence Calibration",
        "",
        f"- predictions: `{payload.get('predictions_jsonl')}`",
        f"- count: `{int(payload.get('count', 0))}`",
        f"- positive_count: `{int(payload.get('positive_count', 0))}`",
        f"- negative_count: `{int(payload.get('negative_count', 0))}`",
        "",
        "| selection | threshold | acc | pos recall | neg acc | balanced acc | f1 | mean IoU(valid) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in [("fixed_0.5", fixed), ("best_balanced", selected), ("target_negative_acc", neg)]:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{_safe_float(row.get('threshold')):.3f}",
                    f"{_safe_float(row.get('acc')):.6f}",
                    f"{_safe_float(row.get('positive_recall')):.6f}",
                    f"{_safe_float(row.get('negative_acc')):.6f}",
                    f"{_safe_float(row.get('balanced_acc')):.6f}",
                    f"{_safe_float(row.get('f1')):.6f}",
                    f"{_safe_float(row.get('mean_iou_valid')):.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows(args.predictions_jsonl)
    sweep = [_metrics(rows, threshold) for threshold in _thresholds(args.thresholds)]
    fixed = min(sweep, key=lambda row: abs(row["threshold"] - 0.5)) if sweep else {}
    selected = max(sweep, key=lambda row: (row.get("balanced_acc", 0.0), row.get("f1", 0.0))) if sweep else {}
    target_neg = [row for row in sweep if row.get("negative_acc", 0.0) >= float(args.target_negative_acc)]
    selected_target_neg = max(target_neg, key=lambda row: (row.get("positive_recall", 0.0), row.get("balanced_acc", 0.0))) if target_neg else selected
    payload = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "count": len(rows),
        "positive_count": int(sum(1 for row in rows if row["target_valid"] >= 0.5)),
        "negative_count": int(sum(1 for row in rows if row["target_valid"] < 0.5)),
        "fixed_0_5": fixed,
        "selected": selected,
        "target_negative_acc": float(args.target_negative_acc),
        "selected_target_negative_acc": selected_target_neg,
        "sweep": sweep,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "subject_box_confidence_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output_dir / "subject_box_confidence_calibration.md", payload)
    print(json.dumps({"output_dir": str(args.output_dir), "selected": selected, "target_negative_acc": selected_target_neg}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
