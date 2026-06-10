#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import MobileCropNetV4BatchDataset, SUBJECT_BOX_TARGET_SOURCES, mobilecropnet_v4_collate
from mobilecropnet_v4.eval_utils import (
    SUBJECT_VALID_POLICIES,
    batch_predictions,
    infer_image_norm,
    infer_input_size,
    infer_subject_valid_threshold,
    load_mobilecropnet_v4_checkpoint,
    per_record_metrics,
    summarize_metrics,
    write_jsonl,
)
from mobilecropnet_v4.route_expert import (
    load_embedded_route_experts,
    load_route_expert_checkpoint,
    override_outputs_route_logits,
    route_expert_ensemble_logits_from_paths,
    route_expert_logits_from_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MobileCropNet v4 on current batch JSONL labels.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--image_mean", default=None, help="Comma-separated RGB mean. Defaults to checkpoint train_config/backbone pretrained_cfg.")
    parser.add_argument("--image_std", default=None, help="Comma-separated RGB std. Defaults to checkpoint train_config/backbone pretrained_cfg.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--save_topk", type=int, default=8)
    parser.add_argument("--subject_valid_threshold", type=float, default=None)
    parser.add_argument("--subject_valid_policy", choices=SUBJECT_VALID_POLICIES, default="confidence")
    parser.add_argument("--subject_box_target_source", choices=SUBJECT_BOX_TARGET_SOURCES, default=None)
    parser.add_argument("--route_expert_checkpoint", type=Path, default=None)
    parser.add_argument("--route_expert_checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--route_expert_weights", default=None, help="Comma-separated weights for route expert ensemble.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress_log_interval", type=int, default=25)
    parser.add_argument(
        "--progress_json",
        type=Path,
        default=None,
        help="Durable progress JSON path. Defaults to <output_dir>/progress.json.",
    )
    return parser


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    enriched = dict(payload)
    enriched["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        "# MobileCropNet v4 Evaluation",
        "",
        f"- checkpoint: `{summary.get('checkpoint')}`",
        f"- eval_jsonl: `{summary.get('eval_jsonl')}`",
        f"- subject_valid_threshold: `{summary.get('subject_valid_threshold', 0.5)}`",
        f"- subject_valid_policy: `{summary.get('subject_valid_policy', 'confidence')}`",
        f"- subject_box_target_source: `{summary.get('subject_box_target_source', 'legacy')}`",
        f"- image_count: {summary.get('image_count', 0)}",
        "",
        "## Key Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)):
            lines.append(f"| {key} | {float(value):.6f} |")
    lines.extend(["", "## By Target AR", "", "| target_ar | images | top1_hit | ndcg@5 | srcc | proposal_r@5_iou0.5 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for ar, row in sorted(summary.get("by_target_ar", {}).items()):
        lines.append(
            "| {ar} | {count} | {top1:.6f} | {ndcg:.6f} | {srcc:.6f} | {prop:.6f} |".format(
                ar=ar,
                count=int(row.get("image_count", 0)),
                top1=float(row.get("candidate_top1_hit", 0.0)),
                ndcg=float(row.get("ndcg_at_5", 0.0)),
                srcc=float(row.get("srcc", 0.0)),
                prop=float(row.get("proposal_recall_at_5_iou_0_5", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.checkpoint, device=device)
    route_expert_paths: list[Path] = []
    if args.route_expert_checkpoint is not None:
        route_expert_paths.append(args.route_expert_checkpoint)
    route_expert_paths.extend(args.route_expert_checkpoints or [])
    route_experts: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    route_expert_source = "none"
    embedded_route_expert_count = 0
    for checkpoint in route_expert_paths:
        expert_model, expert_config, _ = load_route_expert_checkpoint(checkpoint, device=device)
        route_experts.append((expert_model, expert_config))
    if route_experts:
        route_expert_source = "external_checkpoint"
    route_expert_weights = None
    if args.route_expert_weights:
        route_expert_weights = [float(item) for item in str(args.route_expert_weights).split(",") if item.strip()]
    if not route_experts:
        route_experts, route_expert_weights = load_embedded_route_experts(ckpt, device=device)
        embedded_route_expert_count = len(route_experts)
        if route_experts:
            route_expert_source = "embedded_checkpoint"
    input_size = infer_input_size(ckpt, args.input_size)
    image_mean, image_std = infer_image_norm(ckpt, explicit_mean=args.image_mean, explicit_std=args.image_std)
    subject_valid_threshold = infer_subject_valid_threshold(ckpt, args.subject_valid_threshold)
    subject_box_target_source = str(
        args.subject_box_target_source
        or ckpt.get("train_config", {}).get("subject_box_target_source")
        or ckpt.get("dataset_summary", {}).get("subject_box_target_source")
        or "legacy"
    )
    candidate_k = int(args.candidate_k or ckpt.get("model_config", {}).get("candidate_k", 24))
    dataset = MobileCropNetV4BatchDataset(
        jsonl_path=args.eval_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        max_rows=args.max_rows,
        subject_box_target_source=subject_box_target_source,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )
    predictions: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    by_ar: dict[str, list[dict[str, float]]] = defaultdict(list)
    model.eval()
    progress_path = args.progress_json or (args.output_dir / "progress.json")
    started_at = time.monotonic()
    processed = 0
    _write_progress(
        progress_path,
        {
            "state": "running",
            "phase": "replay_eval",
            "checkpoint": str(args.checkpoint),
            "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
            "route_expert_checkpoints": [str(path) for path in route_expert_paths],
            "eval_jsonl": str(args.eval_jsonl),
            "output_dir": str(args.output_dir),
            "image_count": len(dataset),
            "batch_size": int(args.batch_size),
            "processed_batches": 0,
            "processed_images": 0,
            "metrics_json": str(args.output_dir / "metrics.json"),
            "predictions_jsonl": str(args.output_dir / "predictions.jsonl"),
        },
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            tensor_batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(
                tensor_batch["image"],
                tensor_batch["boxes"],
                tensor_batch["valid"],
                tensor_batch["target_ar_id"],
                tensor_batch["image_ar_log"],
                tensor_batch["candidate_is_base"],
                tensor_batch["box_meta"],
                tensor_batch.get("letterbox_content_box"),
                tensor_batch.get("subject_prior_box"),
                tensor_batch.get("subject_prior_valid"),
                tensor_batch.get("subject_prior_reliability"),
            )
            if route_experts:
                if len(route_experts) == 1:
                    expert_model, expert_config = route_experts[0]
                    expert_logits = route_expert_logits_from_paths(
                        expert_model,
                        expert_config,
                        batch["image_path"],
                        device=device,
                    )
                else:
                    expert_logits = route_expert_ensemble_logits_from_paths(
                        route_experts,
                        batch["image_path"],
                        device=device,
                        weights=route_expert_weights,
                    )
                outputs = override_outputs_route_logits(outputs, expert_logits)
            for pred in batch_predictions(
                outputs=outputs,
                batch=batch,
                save_topk=args.save_topk,
                subject_valid_threshold=subject_valid_threshold,
                subject_valid_policy=str(args.subject_valid_policy),
            ):
                metrics = per_record_metrics(
                    pred,
                    subject_valid_conf_threshold=subject_valid_threshold,
                    subject_valid_policy=str(args.subject_valid_policy),
                )
                pred["metrics"] = metrics
                predictions.append(pred)
                metric_rows.append(metrics)
                by_ar[str(pred.get("target_ar", "FREE"))].append(metrics)
            processed += int(tensor_batch["image"].shape[0])
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                elapsed = max(1e-6, time.monotonic() - started_at)
                payload = {
                    "state": "running",
                    "phase": "replay_eval",
                    "checkpoint": str(args.checkpoint),
                    "output_dir": str(args.output_dir),
                    "image_count": len(dataset),
                    "processed_batches": batch_idx,
                    "processed_images": processed,
                    "samples_per_sec": round(processed / elapsed, 4),
                    "elapsed_sec": round(elapsed, 3),
                    "metrics_json": str(args.output_dir / "metrics.json"),
                    "predictions_jsonl": str(args.output_dir / "predictions.jsonl"),
                }
                _write_progress(progress_path, payload)
                print(json.dumps({"event": "replay_eval_progress", **payload}, ensure_ascii=False), file=sys.stderr, flush=True)

    by_target_ar = {}
    for ar, rows in by_ar.items():
        summarized = summarize_metrics(rows)
        summarized["image_count"] = len(rows)
        by_target_ar[ar] = summarized
    summary = {
        "checkpoint": str(args.checkpoint),
        "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
        "route_expert_checkpoints": [str(path) for path in route_expert_paths],
        "route_expert_weights": route_expert_weights,
        "route_expert_source": route_expert_source,
        "embedded_route_expert_count": embedded_route_expert_count,
        "eval_jsonl": str(args.eval_jsonl),
        "image_count": len(predictions),
        "subject_valid_threshold": float(subject_valid_threshold),
        "subject_valid_policy": str(args.subject_valid_policy),
        "subject_box_target_source": subject_box_target_source,
        "dataset_summary": dataset.summary(),
        "metrics": summarize_metrics(metric_rows),
        "by_target_ar": by_target_ar,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    _write_report(args.output_dir / "EVAL_REPORT.md", summary)
    _write_progress(
        progress_path,
        {
            "state": "completed",
            "phase": "replay_eval",
            "checkpoint": str(args.checkpoint),
            "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
            "route_expert_checkpoints": [str(path) for path in route_expert_paths],
            "eval_jsonl": str(args.eval_jsonl),
            "output_dir": str(args.output_dir),
            "image_count": len(predictions),
            "processed_batches": len(loader),
            "processed_images": len(predictions),
            "metrics_json": str(args.output_dir / "metrics.json"),
            "predictions_jsonl": str(args.output_dir / "predictions.jsonl"),
        },
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
