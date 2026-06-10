#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import _subject_box_supervision_valid_from_row, _subject_prior_box_from_row, box_iou_xyxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="이미지 입력만 사용하는 YOLO 내부 subject-prior smoke 평가.")
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--weights", type=Path, default=Path("src/scripts/yolov8n.pt"))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_unique_images", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress_log_interval", type=int, default=25)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _image_path(row: dict[str, Any], project_root: Path) -> Path:
    raw = row.get("image_path") or row.get("path") or row.get("file_name")
    if raw is None:
        raise ValueError(f"row has no image_path: image_id={row.get('image_id')!r}")
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _clip_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box[:4]]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _area(box: list[float]) -> float:
    b = _clip_box(box)
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _union(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    xs1, ys1, xs2, ys2 = zip(*[_clip_box(box) for box in boxes])
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _threshold_metrics(rows: list[dict[str, float]], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        target = row["target_valid"] >= 0.5
        pred = row["confidence"] >= threshold
        if pred and target:
            tp += 1
        elif pred and not target:
            fp += 1
        elif (not pred) and target:
            fn += 1
        else:
            tn += 1
    pos = tp + fn
    neg = tn + fp
    return {
        "threshold": float(threshold),
        "acc": float((tp + tn) / max(1, tp + fp + tn + fn)),
        "positive_recall": float(tp / max(1, pos)),
        "negative_acc": float(tn / max(1, neg)),
        "balanced_acc": float(0.5 * ((tp / max(1, pos)) + (tn / max(1, neg)))),
    }


def _summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    valid = [row for row in rows if row["target_valid"] >= 0.5]
    sweep = [_threshold_metrics(rows, round(t / 100.0, 2)) for t in range(5, 96, 5)]
    best = max(sweep, key=lambda row: (row["balanced_acc"], row["positive_recall"])) if sweep else {}
    return {
        "row_count": len(rows),
        "positive_count": len(valid),
        "mean_iou_valid": _safe_mean([row["iou"] for row in valid]),
        "hit_0_5_valid": _safe_mean([1.0 if row["iou"] >= 0.5 else 0.0 for row in valid]),
        "mean_confidence": _safe_mean([row["confidence"] for row in rows]),
        "fixed_0_5": _threshold_metrics(rows, 0.5) if rows else {},
        "best_balanced": best,
        "threshold_sweep": sweep,
    }


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "phase": "load",
                "eval_jsonl": str(args.eval_jsonl),
                "weights": str(args.weights),
                "started_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    from ultralytics import YOLO

    rows = _read_jsonl(args.eval_jsonl)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_paths: dict[str, Path] = {}
    for row in rows:
        image_id = str(row.get("image_id") or row.get("image_path"))
        if len(by_image) >= int(args.max_unique_images) and image_id not in by_image:
            continue
        by_image[image_id].append(row)
        image_paths[image_id] = _image_path(row, args.project_root)

    model = YOLO(str(args.weights))
    metric_rows: list[dict[str, float]] = []
    pred_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    status_path.write_text(
        json.dumps({"state": "running", "phase": "predict", "unique_images": len(by_image), "last_update_time_unix": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for idx, (image_id, group) in enumerate(by_image.items(), start=1):
        path = image_paths[image_id]
        result = model.predict(source=str(path), imgsz=int(args.imgsz), conf=float(args.conf), device=str(args.device), verbose=False)[0]
        h, w = result.orig_shape
        detections: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            clss = result.boxes.cls.detach().cpu().tolist()
            names = result.names
            for box_px, conf, cls_id in zip(xyxy, confs, clss):
                box = _clip_box([box_px[0] / max(1, w), box_px[1] / max(1, h), box_px[2] / max(1, w), box_px[3] / max(1, h)])
                detections.append(
                    {
                        "box": box,
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": str(names.get(int(cls_id), int(cls_id))) if isinstance(names, dict) else str(int(cls_id)),
                        "area": _area(box),
                    }
                )
        detections.sort(key=lambda det: (float(det["confidence"]) * math.sqrt(max(1e-8, float(det["area"]))), float(det["confidence"])), reverse=True)
        top1 = detections[0]["box"] if detections else None
        union_top3 = _union([det["box"] for det in detections[:3]])
        pred_box = union_top3 if union_top3 is not None and _area(union_top3) <= 0.75 else top1
        pred_conf = max([float(det["confidence"]) for det in detections], default=0.0)
        for row in group:
            teacher = _subject_prior_box_from_row(row)
            target_valid = _subject_box_supervision_valid_from_row(row, has_box=teacher is not None)
            iou = box_iou_xyxy(pred_box, teacher) if pred_box is not None and teacher is not None and target_valid > 0.0 else 0.0
            metric = {
                "target_valid": float(target_valid),
                "confidence": float(pred_conf),
                "iou": float(iou),
                "detected": float(pred_box is not None),
            }
            metric_rows.append(metric)
            pred_rows.append(
                {
                    "image_id": image_id,
                    "target_ar": row.get("target_ar", "FREE"),
                    "subject_mode": (row.get("routing") or {}).get("subject_mode") if isinstance(row.get("routing"), dict) else None,
                    "teacher_box": teacher,
                    "pred_box": pred_box,
                    "confidence": pred_conf,
                    "detections": detections[:8],
                    "metrics": metric,
                }
            )
        if args.progress_log_interval > 0 and (idx == 1 or idx % int(args.progress_log_interval) == 0):
            status_path.write_text(
                json.dumps(
                    {
                        "state": "running",
                        "phase": "predict",
                        "processed_unique_images": idx,
                        "unique_images": len(by_image),
                        "elapsed_sec": time.monotonic() - started,
                        "last_update_time_unix": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    summary = {
        "state": "completed",
        "eval_jsonl": str(args.eval_jsonl),
        "weights": str(args.weights),
        "unique_images": len(by_image),
        "metrics": _summarize(metric_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in pred_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    status_path.write_text(json.dumps({"state": "completed", "phase": "completed", "summary_path": str(args.output_dir / "summary.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
