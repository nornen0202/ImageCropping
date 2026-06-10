#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet.eval_utils import (
    acc_k_of_top_n,
    average_precision_binary,
    box_iou_xyxy,
    infer_input_size,
    load_label_crop_records,
    load_mobilecropnet_checkpoint,
    mean,
    pearson_corr,
    roc_auc_binary,
    safe_float,
    score_crop_record,
    spearman_corr,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MobileCropNet on current GAIC-like crop labels.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label_json", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--candidate_source", choices=["label", "static"], default="label")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--static_k", type=int, default=None)
    parser.add_argument("--static_area_prior", type=float, default=None)
    parser.add_argument("--static_area_penalty", type=float, default=0.0)
    parser.add_argument("--static_full_penalty", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--target_ar_filter", default=None)
    parser.add_argument("--save_topk", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _safe_metric(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _topk_any_positive(candidates: list[dict[str, Any]], ranked: list[int], k: int) -> float:
    if not ranked:
        return 0.0
    top = ranked[: min(k, len(ranked))]
    return float(any(int(candidates[idx].get("gt_flag", 0)) == 1 for idx in top))


def _topk_max_iou(candidates: list[dict[str, Any]], ranked: list[int], best_label: dict[str, Any] | None, k: int) -> float:
    if not ranked or not isinstance(best_label, dict):
        return 0.0
    label_box = best_label.get("bbox_norm_xyxy")
    if not label_box:
        return 0.0
    return max(
        box_iou_xyxy(candidates[idx].get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), label_box)
        for idx in ranked[: min(k, len(ranked))]
    )


def _per_image_metrics(pred: dict[str, Any]) -> dict[str, float]:
    candidates = pred["candidates"]
    ranked = pred["ranked_indices"]
    source = str(pred["candidate_source"])
    best_label = pred.get("best_label_candidate") if isinstance(pred.get("best_label_candidate"), dict) else None
    top_idx = ranked[0] if ranked else -1
    top = candidates[top_idx] if top_idx >= 0 else {}
    metrics: dict[str, float] = {
        "top1_iou_to_best_label": safe_float(pred.get("top_iou_to_best_label")),
        "top1_iou_ge_0_5": float(safe_float(pred.get("top_iou_to_best_label")) >= 0.5),
        "top1_iou_ge_0_7": float(safe_float(pred.get("top_iou_to_best_label")) >= 0.7),
        "top3_max_iou_to_best_label": _topk_max_iou(candidates, ranked, best_label, 3),
        "top5_max_iou_to_best_label": _topk_max_iou(candidates, ranked, best_label, 5),
    }
    if best_label:
        metrics["oracle_label_score"] = safe_float(best_label.get("label_score"), 0.0)
        metrics["decision_acc"] = float(int(pred.get("decision_id", -1)) == int(best_label.get("decision_id", -2)))
        metrics["route_acc"] = float(int(pred.get("route_id", -1)) == int(best_label.get("subject_mode_id", -2)))
        if source == "label":
            metrics["top1_exact_best_label"] = float(
                str(top.get("candidate_id", "")) == str(best_label.get("candidate_id", ""))
            )
            metrics["top1_label_score"] = safe_float(top.get("label_score"), 0.0)
            metrics["label_score_regret"] = max(0.0, metrics["oracle_label_score"] - metrics["top1_label_score"])

    if source == "label":
        labels = [int(cand.get("gt_flag", 0)) for cand in candidates]
        label_scores = [safe_float(cand.get("label_score"), 0.0) for cand in candidates]
        model_scores = [safe_float(cand.get("model_score"), 0.0) for cand in candidates]
        metrics.update(
            {
                "candidate_top1_hit": float(labels[top_idx] == 1) if top_idx >= 0 else 0.0,
                "candidate_recall_at_3": _topk_any_positive(candidates, ranked, 3),
                "candidate_recall_at_5": _topk_any_positive(candidates, ranked, 5),
                "srcc": spearman_corr(model_scores, label_scores),
                "pcc": pearson_corr(model_scores, label_scores),
                "average_precision": average_precision_binary(labels, model_scores),
                "roc_auc": roc_auc_binary(labels, model_scores),
                "acc1_of_top5": acc_k_of_top_n(label_scores, model_scores, k=1, n=5),
                "acc1_of_top10": acc_k_of_top_n(label_scores, model_scores, k=1, n=10),
                "acc5_of_top10": acc_k_of_top_n(label_scores, model_scores, k=5, n=10),
            }
        )
    return {key: _safe_metric(value) for key, value in metrics.items()}


def _summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row.keys()})
    return {key: mean([row[key] for row in rows if key in row]) for key in keys}


def _prediction_row(pred: dict[str, Any], metrics: dict[str, float], *, save_topk: int) -> dict[str, Any]:
    ranked = pred["ranked_indices"]
    candidates = pred["candidates"]
    top_candidates = [candidates[idx] for idx in ranked[: min(save_topk, len(ranked))]]
    return {
        "image_id": pred["image_id"],
        "file_name": pred["file_name"],
        "target_ar": pred["target_ar"],
        "candidate_source": pred["candidate_source"],
        "decision_id": pred["decision_id"],
        "route_id": pred["route_id"],
        "anchor_box": pred["anchor_box"],
        "top_candidate": pred["top_candidate"],
        "best_label_candidate": pred["best_label_candidate"],
        "top_candidates": top_candidates,
        "metrics": metrics,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# MobileCropNet GAIC Evaluation",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- label_json: `{summary['label_json']}`",
        f"- candidate_source: `{summary['candidate_source']}`",
        f"- image_count: {summary['image_count']}",
        f"- candidate_count: {summary['candidate_count']}",
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
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- `candidate_top1_hit`: model top-1 candidate is a GAIC-like positive label.",
            "- `candidate_recall_at_K`: at least one positive label appears in model top-K candidates.",
            "- `srcc`/`pcc`: per-image rank/linear correlation between model scores and label scores.",
            "- `accK_of_topN`: GAIC-style proxy; model top-K candidates are checked against label-score top-N candidates.",
            "- `top*_iou_to_best_label`: geometric agreement with the highest-score positive label candidate.",
            "",
            "## By Target AR",
            "",
            "| target_ar | images | candidate_top1_hit | srcc | pcc | top1_iou_to_best_label |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ar, row in sorted(summary.get("by_target_ar", {}).items()):
        lines.append(
            "| {ar} | {count} | {top1:.6f} | {srcc:.6f} | {pcc:.6f} | {iou:.6f} |".format(
                ar=ar,
                count=int(row.get("image_count", 0)),
                top1=float(row.get("candidate_top1_hit", 0.0)),
                srcc=float(row.get("srcc", 0.0)),
                pcc=float(row.get("pcc", 0.0)),
                iou=float(row.get("top1_iou_to_best_label", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_checkpoint(args.checkpoint, device=device)
    input_size = infer_input_size(ckpt, args.input_size)
    label_json = args.label_json
    records = load_label_crop_records(
        label_json,
        max_images=args.max_images,
        target_ar_filter=args.target_ar_filter,
    )

    metric_rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []
    by_ar_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    all_labels: list[int] = []
    all_model_scores: list[float] = []
    candidate_count = 0

    for record in records:
        pred = score_crop_record(
            model=model,
            record=record,
            image_root=args.image_root,
            input_size=input_size,
            device=device,
            candidate_source=args.candidate_source,
            static_k=args.static_k,
            static_area_prior=args.static_area_prior,
            static_area_penalty=args.static_area_penalty,
            static_full_penalty=args.static_full_penalty,
        )
        metrics = _per_image_metrics(pred)
        metric_rows.append(metrics)
        by_ar_rows[str(pred["target_ar"])].append(metrics)
        prediction_rows.append(_prediction_row(pred, metrics, save_topk=args.save_topk))
        candidate_count += len(pred["candidates"])
        if args.candidate_source == "label":
            all_labels.extend(int(cand.get("gt_flag", 0)) for cand in pred["candidates"])
            all_model_scores.extend(safe_float(cand.get("model_score"), 0.0) for cand in pred["candidates"])

    summary_metrics = _summarize_metric_rows(metric_rows)
    if args.candidate_source == "label":
        summary_metrics["global_average_precision"] = average_precision_binary(all_labels, all_model_scores)
        summary_metrics["global_roc_auc"] = roc_auc_binary(all_labels, all_model_scores)
    by_target_ar = {}
    for ar, rows in by_ar_rows.items():
        row = _summarize_metric_rows(rows)
        row["image_count"] = len(rows)
        by_target_ar[ar] = row

    summary = {
        "checkpoint": str(args.checkpoint),
        "label_json": str(label_json),
        "image_root": str(args.image_root),
        "candidate_source": args.candidate_source,
        "static_area_prior": args.static_area_prior,
        "static_area_penalty": args.static_area_penalty,
        "static_full_penalty": args.static_full_penalty,
        "input_size": input_size,
        "image_count": len(records),
        "candidate_count": candidate_count,
        "metrics": summary_metrics,
        "by_target_ar": by_target_ar,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_report(args.output_dir / "EVAL_REPORT.md", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
