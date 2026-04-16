#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import MobileCropNetV4BatchDataset, mobilecropnet_v4_collate
from mobilecropnet_v4.eval_utils import (
    batch_predictions,
    infer_input_size,
    load_mobilecropnet_v4_checkpoint,
    per_record_metrics,
    summarize_metrics,
    write_jsonl,
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--save_topk", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        "# MobileCropNet v4 Evaluation",
        "",
        f"- checkpoint: `{summary.get('checkpoint')}`",
        f"- eval_jsonl: `{summary.get('eval_jsonl')}`",
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
    input_size = infer_input_size(ckpt, args.input_size)
    candidate_k = int(args.candidate_k or ckpt.get("model_config", {}).get("candidate_k", 24))
    dataset = MobileCropNetV4BatchDataset(
        jsonl_path=args.eval_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        max_rows=args.max_rows,
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
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(
                tensor_batch["image"],
                tensor_batch["boxes"],
                tensor_batch["valid"],
                tensor_batch["target_ar_id"],
                tensor_batch["image_ar_log"],
                tensor_batch["candidate_is_base"],
                tensor_batch["box_meta"],
            )
            for pred in batch_predictions(outputs=outputs, batch=batch, save_topk=args.save_topk):
                metrics = per_record_metrics(pred)
                pred["metrics"] = metrics
                predictions.append(pred)
                metric_rows.append(metrics)
                by_ar[str(pred.get("target_ar", "FREE"))].append(metrics)

    by_target_ar = {}
    for ar, rows in by_ar.items():
        summarized = summarize_metrics(rows)
        summarized["image_count"] = len(rows)
        by_target_ar[ar] = summarized
    summary = {
        "checkpoint": str(args.checkpoint),
        "eval_jsonl": str(args.eval_jsonl),
        "image_count": len(predictions),
        "dataset_summary": dataset.summary(),
        "metrics": summarize_metrics(metric_rows),
        "by_target_ar": by_target_ar,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    _write_report(args.output_dir / "EVAL_REPORT.md", summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
