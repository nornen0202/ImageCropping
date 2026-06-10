#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    MobileCropNetV4BatchDataset,
    box_iou_xyxy,
    letterbox_to_original_box,
    mobilecropnet_v4_collate,
    original_to_letterbox_box,
)
from mobilecropnet_v4.eval_utils import infer_image_norm, infer_input_size, load_mobilecropnet_v4_checkpoint, summarize_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 내부 subject detector + MobileCropNet crop 모델 composite no-prior 평가.")
    parser.add_argument("--yolo_checkpoint", type=Path, required=True)
    parser.add_argument("--crop_checkpoint", type=Path, required=True)
    parser.add_argument("--eval_jsonl", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=2400)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--yolo_conf", type=float, default=0.05)
    parser.add_argument("--progress_log_interval", type=int, default=25)
    return parser


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _route_balanced_acc(rows: list[dict[str, float]]) -> float:
    recalls = []
    for route_id in sorted({int(row["route_target"]) for row in rows}):
        subset = [row for row in rows if int(row["route_target"]) == route_id]
        if subset:
            recalls.append(sum(float(row["route_correct"]) for row in subset) / float(len(subset)))
    return float(sum(recalls) / max(1, len(recalls)))


def _image_path_from_dataset_row(dataset: MobileCropNetV4BatchDataset, idx: int) -> Path:
    row = dataset.rows[idx]
    raw = row.get("image_path") or row.get("path") or row.get("file_name")
    path = Path(str(raw))
    return path if path.is_absolute() else dataset.project_root / path


def _yolo_subject_box(model: Any, image_path: Path, *, imgsz: int, conf: float, device: str) -> tuple[list[float] | None, float]:
    result = model.predict(source=str(image_path), imgsz=int(imgsz), conf=float(conf), device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) <= 0:
        return None, 0.0
    h, w = result.orig_shape
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    detections: list[tuple[list[float], float, float]] = []
    for box_px, score in zip(xyxy, confs):
        box = [
            max(0.0, min(1.0, float(box_px[0]) / max(1, w))),
            max(0.0, min(1.0, float(box_px[1]) / max(1, h))),
            max(0.0, min(1.0, float(box_px[2]) / max(1, w))),
            max(0.0, min(1.0, float(box_px[3]) / max(1, h))),
        ]
        box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        detections.append((box, float(score), area))
    detections.sort(key=lambda item: (item[1] * math.sqrt(max(1e-8, item[2])), item[1]), reverse=True)
    top_boxes = [item[0] for item in detections[:3]]
    union = [
        min(box[0] for box in top_boxes),
        min(box[1] for box in top_boxes),
        max(box[2] for box in top_boxes),
        max(box[3] for box in top_boxes),
    ]
    union_area = max(0.0, union[2] - union[0]) * max(0.0, union[3] - union[1])
    pred_box = union if union_area <= 0.75 else detections[0][0]
    return pred_box, max(float(item[1]) for item in detections)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "phase": "load",
                "yolo_checkpoint": str(args.yolo_checkpoint),
                "crop_checkpoint": str(args.crop_checkpoint),
                "eval_jsonl": str(args.eval_jsonl),
                "started_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    from ultralytics import YOLO

    yolo_model = YOLO(str(args.yolo_checkpoint))
    crop_model, crop_ckpt = load_mobilecropnet_v4_checkpoint(args.crop_checkpoint, device=device)
    crop_input_size = infer_input_size(crop_ckpt, None)
    crop_mean, crop_std = infer_image_norm(crop_ckpt)
    crop_k = int(crop_ckpt.get("model_config", {}).get("candidate_k", 24))
    crop_ds = MobileCropNetV4BatchDataset(
        jsonl_path=args.eval_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=crop_input_size,
        candidate_k=crop_k,
        image_mean=crop_mean,
        image_std=crop_std,
        max_rows=args.max_rows,
    )
    loader = DataLoader(
        crop_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )

    metric_rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []
    yolo_cache: dict[str, tuple[list[float] | None, float]] = {}
    started = time.monotonic()
    processed = 0
    with torch.no_grad():
        row_offset = 0
        for batch_idx, batch_raw in enumerate(loader, start=1):
            crop_batch = _to_device(batch_raw, device)
            prior_boxes = []
            prior_valid = []
            yolo_boxes_orig: list[list[float] | None] = []
            for local_idx, image_id in enumerate(batch_raw["image_id"]):
                dataset_idx = row_offset + local_idx
                cache_key = str(image_id)
                if cache_key not in yolo_cache:
                    yolo_cache[cache_key] = _yolo_subject_box(
                        yolo_model,
                        _image_path_from_dataset_row(crop_ds, dataset_idx),
                        imgsz=args.yolo_imgsz,
                        conf=args.yolo_conf,
                        device="0" if device.type == "cuda" else "cpu",
                    )
                pred_orig, conf = yolo_cache[cache_key]
                yolo_boxes_orig.append(pred_orig)
                if pred_orig is None:
                    prior_boxes.append([0.0, 0.0, 0.0, 0.0])
                    prior_valid.append(0.0)
                else:
                    prior_boxes.append(original_to_letterbox_box(pred_orig, batch_raw["transform"][local_idx]))
                    prior_valid.append(float(conf))
            row_offset += len(batch_raw["image_id"])
            crop_prior_box_tensor = torch.tensor(prior_boxes, dtype=crop_batch["image"].dtype, device=device)
            crop_prior_valid_tensor = torch.tensor(prior_valid, dtype=crop_batch["image"].dtype, device=device)
            outputs = crop_model(
                crop_batch["image"],
                crop_batch["boxes"],
                crop_batch["valid"],
                crop_batch["target_ar_id"],
                crop_batch["image_ar_log"],
                crop_batch.get("candidate_is_base"),
                crop_batch.get("box_meta"),
                crop_batch.get("letterbox_content_box"),
                crop_prior_box_tensor,
                crop_prior_valid_tensor,
                crop_prior_valid_tensor,
                0.0,
            )
            route_pred = outputs["route_logits"].argmax(dim=1).detach().cpu()
            utility = torch.sigmoid(outputs["utility_logits"]).detach().cpu()
            for idx in range(route_pred.shape[0]):
                target_valid = float(batch_raw["subject_box_valid"][idx].item())
                target_box_orig = letterbox_to_original_box(batch_raw["subject_box_target"][idx].tolist(), batch_raw["transform"][idx])
                pred_box_orig = yolo_boxes_orig[idx]
                subject_iou = box_iou_xyxy(pred_box_orig, target_box_orig) if pred_box_orig is not None and target_valid > 0.0 else 0.0
                top_idx = int(utility[idx].argmax().item())
                top_box_orig = batch_raw["boxes_orig"][idx, top_idx].tolist()
                best_positive_iou = 0.0
                for pos_idx, pos_valid in enumerate(batch_raw["positive_valid"][idx].tolist()):
                    if float(pos_valid) <= 0.0:
                        continue
                    best_positive_iou = max(best_positive_iou, box_iou_xyxy(top_box_orig, batch_raw["positive_boxes_orig"][idx, pos_idx].tolist()))
                route_target = int(batch_raw["route_target"][idx].item())
                metrics = {
                    "route_target": float(route_target),
                    "route_pred": float(route_pred[idx].item()),
                    "route_correct": float(int(route_pred[idx].item()) == route_target),
                    "subject_box_iou": float(subject_iou),
                    "subject_box_pred_conf": float(prior_valid[idx]),
                    "subject_box_target_valid": float(target_valid),
                    "subject_box_valid_acc": float((float(prior_valid[idx]) >= 0.5) == (target_valid >= 0.5)),
                    "top1_iou_to_positive": float(best_positive_iou),
                    "top1_hit": float(best_positive_iou >= 0.5),
                }
                metric_rows.append(metrics)
                prediction_rows.append({"image_id": batch_raw["image_id"][idx], "target_ar": batch_raw["target_ar"][idx], "metrics": metrics, "subject_box": {"predicted_norm_xyxy": pred_box_orig, "teacher_norm_xyxy": target_box_orig}})
            processed += int(route_pred.shape[0])
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                status_path.write_text(json.dumps({"state": "running", "phase": "eval", "processed": processed, "elapsed_sec": time.monotonic() - started, "last_update_time_unix": time.time()}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "state": "completed",
        "yolo_checkpoint": str(args.yolo_checkpoint),
        "crop_checkpoint": str(args.crop_checkpoint),
        "eval_jsonl": str(args.eval_jsonl),
        "image_count": len(metric_rows),
        "metrics": {**summarize_metrics(metric_rows), "route_balanced_acc": _route_balanced_acc(metric_rows)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    status_path.write_text(json.dumps({"state": "completed", "phase": "completed", "summary_path": str(args.output_dir / "metrics.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
