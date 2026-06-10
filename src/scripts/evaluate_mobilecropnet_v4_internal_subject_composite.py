#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mobilecropnet_v4.data import (
    MobileCropNetV4BatchDataset,
    box_iou_xyxy,
    letterbox_to_original_box,
    mobilecropnet_v4_collate,
    original_to_letterbox_box,
)
from mobilecropnet_v4.eval_utils import infer_image_norm, infer_input_size, load_mobilecropnet_v4_checkpoint, summarize_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="내부 subject 모델을 route/crop 모델 prior로 쓰는 no-prior composite smoke 평가.")
    parser.add_argument("--subject_checkpoint", type=Path, required=True)
    parser.add_argument("--crop_checkpoint", type=Path, required=True)
    parser.add_argument("--eval_jsonl", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=2400)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
                "subject_checkpoint": str(args.subject_checkpoint),
                "crop_checkpoint": str(args.crop_checkpoint),
                "eval_jsonl": str(args.eval_jsonl),
                "started_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    subject_model, subject_ckpt = load_mobilecropnet_v4_checkpoint(args.subject_checkpoint, device=device)
    crop_model, crop_ckpt = load_mobilecropnet_v4_checkpoint(args.crop_checkpoint, device=device)
    subject_input_size = infer_input_size(subject_ckpt, None)
    crop_input_size = infer_input_size(crop_ckpt, None)
    subject_mean, subject_std = infer_image_norm(subject_ckpt)
    crop_mean, crop_std = infer_image_norm(crop_ckpt)
    subject_k = int(subject_ckpt.get("model_config", {}).get("candidate_k", 32))
    crop_k = int(crop_ckpt.get("model_config", {}).get("candidate_k", 24))

    subject_ds = MobileCropNetV4BatchDataset(
        jsonl_path=args.eval_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=subject_input_size,
        candidate_k=subject_k,
        image_mean=subject_mean,
        image_std=subject_std,
        max_rows=args.max_rows,
    )
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
    subject_loader = DataLoader(
        subject_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )
    crop_loader = DataLoader(
        crop_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )

    metric_rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    processed = 0
    status_path.write_text(
        json.dumps({"state": "running", "phase": "eval", "last_update_time_unix": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with torch.no_grad():
        for batch_idx, (subject_batch_raw, crop_batch_raw) in enumerate(zip(subject_loader, crop_loader), start=1):
            if subject_batch_raw["image_id"] != crop_batch_raw["image_id"]:
                raise RuntimeError("subject/crop dataloader order mismatch")
            subject_batch = _to_device(subject_batch_raw, device)
            crop_batch = _to_device(crop_batch_raw, device)
            subject_outputs = subject_model(
                subject_batch["image"],
                None,
                None,
                subject_batch["target_ar_id"],
                subject_batch["image_ar_log"],
                None,
                None,
                subject_batch.get("letterbox_content_box"),
                None,
                None,
                None,
            )
            subject_pred_lb = subject_outputs["pred_subject_box"].detach().cpu()
            subject_pred_valid = subject_outputs["pred_subject_valid"].detach().cpu().view(-1)
            crop_prior_boxes = []
            for idx in range(subject_pred_lb.shape[0]):
                pred_orig = letterbox_to_original_box(subject_pred_lb[idx].tolist(), subject_batch_raw["transform"][idx])
                crop_prior_boxes.append(original_to_letterbox_box(pred_orig, crop_batch_raw["transform"][idx]))
            crop_prior_box_tensor = torch.tensor(crop_prior_boxes, dtype=crop_batch["image"].dtype, device=device)
            crop_prior_valid_tensor = subject_pred_valid.to(device=device, dtype=crop_batch["image"].dtype)
            crop_outputs = crop_model(
                crop_batch["image"],
                None,
                None,
                crop_batch["target_ar_id"],
                crop_batch["image_ar_log"],
                None,
                None,
                crop_batch.get("letterbox_content_box"),
                crop_prior_box_tensor,
                crop_prior_valid_tensor,
                crop_prior_valid_tensor,
                0.0,
            )
            route_pred = crop_outputs["route_logits"].argmax(dim=1).detach().cpu()
            utility = torch.sigmoid(crop_outputs["utility_logits"]).detach().cpu()
            for idx in range(route_pred.shape[0]):
                target_valid = float(crop_batch_raw["subject_box_valid"][idx].item())
                target_box_orig = letterbox_to_original_box(
                    crop_batch_raw["subject_box_target"][idx].tolist(),
                    crop_batch_raw["transform"][idx],
                )
                pred_box_orig = letterbox_to_original_box(subject_pred_lb[idx].tolist(), subject_batch_raw["transform"][idx])
                subject_iou = box_iou_xyxy(pred_box_orig, target_box_orig) if target_valid > 0.0 else 0.0
                top_idx = int(utility[idx].argmax().item())
                top_box_orig = crop_batch_raw["boxes_orig"][idx, top_idx].tolist()
                best_positive_iou = 0.0
                for pos_idx, pos_valid in enumerate(crop_batch_raw["positive_valid"][idx].tolist()):
                    if float(pos_valid) <= 0.0:
                        continue
                    best_positive_iou = max(best_positive_iou, box_iou_xyxy(top_box_orig, crop_batch_raw["positive_boxes_orig"][idx, pos_idx].tolist()))
                route_target = int(crop_batch_raw["route_target"][idx].item())
                metrics = {
                    "route_target": float(route_target),
                    "route_pred": float(route_pred[idx].item()),
                    "route_correct": float(int(route_pred[idx].item()) == route_target),
                    "subject_box_iou": float(subject_iou),
                    "subject_box_pred_conf": float(subject_pred_valid[idx].item()),
                    "subject_box_target_valid": float(target_valid),
                    "subject_box_valid_acc": float((float(subject_pred_valid[idx].item()) >= 0.5) == (target_valid >= 0.5)),
                    "top1_iou_to_positive": float(best_positive_iou),
                    "top1_hit": float(best_positive_iou >= 0.5),
                }
                metric_rows.append(metrics)
                prediction_rows.append(
                    {
                        "image_id": crop_batch_raw["image_id"][idx],
                        "target_ar": crop_batch_raw["target_ar"][idx],
                        "metrics": metrics,
                        "subject_box": {
                            "predicted_norm_xyxy": pred_box_orig,
                            "teacher_norm_xyxy": target_box_orig,
                        },
                    }
                )
            processed += int(route_pred.shape[0])
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                status_path.write_text(
                    json.dumps(
                        {
                            "state": "running",
                            "phase": "eval",
                            "last_update_time_unix": time.time(),
                            "batches": batch_idx,
                            "processed": processed,
                            "elapsed_sec": time.monotonic() - started,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    summary = {
        "state": "completed",
        "subject_checkpoint": str(args.subject_checkpoint),
        "crop_checkpoint": str(args.crop_checkpoint),
        "eval_jsonl": str(args.eval_jsonl),
        "image_count": len(metric_rows),
        "metrics": {
            **summarize_metrics(metric_rows),
            "route_balanced_acc": _route_balanced_acc(metric_rows),
        },
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
