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

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    MobileCropNetV4BatchDataset,
    SUBJECT_MODE_VOCAB,
    SUBJECT_BOX_TARGET_SOURCES,
    TARGET_AR_VOCAB,
    letterbox_to_original_box,
    mobilecropnet_v4_collate,
)
from mobilecropnet_v4.eval_utils import (
    AR_COMPATIBILITY_LOG_ERROR_THRESHOLD,
    RUNTIME_BASELINE_DECISION_CONFIDENCE_THRESHOLD,
    RUNTIME_BASELINE_SCORE_MARGIN,
    SUBJECT_VALID_POLICIES,
    build_runtime_baselines,
    execute_runtime_decision,
    infer_image_norm,
    infer_input_size,
    infer_subject_valid_threshold,
    load_mobilecropnet_v4_checkpoint,
    select_generated_proposal_index,
    subject_valid_from_policy,
    summarize_metrics,
)
from mobilecropnet_v4.route_expert import (
    load_embedded_route_experts,
    load_route_expert_checkpoint,
    override_outputs_route_logits,
    route_expert_ensemble_logits_from_paths,
    route_expert_logits_from_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deployed Product AR inference against Product AR/full labels.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--image_mean", default=None)
    parser.add_argument("--image_std", default=None)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--row_shard_index", type=int, default=0)
    parser.add_argument("--row_shard_count", type=int, default=1)
    parser.add_argument("--selection_policy", choices=["utility_top1", "proposal_top1", "proposal_topk_rerank"], default="proposal_topk_rerank")
    parser.add_argument("--proposal_top_m", type=int, default=8)
    parser.add_argument("--runtime_subject_prior_mode", choices=["none", "teacher"], default="none")
    parser.add_argument("--exact_target_ar_postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use_decision_source_action_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use calibrated decision_source_logit to choose baseline-vs-crop final runtime action. Defaults to checkpoint train_config.",
    )
    parser.add_argument("--enforce_baseline_decision_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline_decision_confidence_threshold", type=float, default=RUNTIME_BASELINE_DECISION_CONFIDENCE_THRESHOLD)
    parser.add_argument("--baseline_score_margin", type=float, default=RUNTIME_BASELINE_SCORE_MARGIN)
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


def _target_ar_value(target_ar: str) -> float | None:
    values = {"1:1": 1.0, "3:4": 3.0 / 4.0, "4:3": 4.0 / 3.0, "16:9": 16.0 / 9.0, "9:16": 9.0 / 16.0}
    return values.get(str(target_ar))


def _crop_ar(box: list[float], *, width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw = max(1e-6, (x2 - x1) * float(max(1, int(width))))
    bh = max(1e-6, (y2 - y1) * float(max(1, int(height))))
    return float(bw / bh)


def _target_ar_error(box: list[float], *, width: int, height: int, target_ar: str) -> tuple[float, float]:
    crop_ar = _crop_ar(box, width=width, height=height)
    target = _target_ar_value(target_ar)
    if target is None:
        return crop_ar, 0.0
    return crop_ar, float(abs(math.log(max(1e-6, crop_ar) / max(1e-6, target))))


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-8, aa + bb - inter))


def _proposal_rows(
    *,
    proposal_boxes: list[list[float]],
    proposal_scores: list[float],
    utility_scores: list[float],
    return_scores: list[float] | None = None,
    transform: Any,
    width: int,
    height: int,
    target_ar: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, box_lb in enumerate(proposal_boxes):
        orig_box = letterbox_to_original_box(box_lb, transform)
        crop_ar, ar_error = _target_ar_error(orig_box, width=width, height=height, target_ar=target_ar)
        rows.append(
            {
                "proposal_id": idx,
                "bbox_norm_xyxy": orig_box,
                "score": float(proposal_scores[idx]),
                "utility_score": float(utility_scores[idx]),
                "return_score": float(return_scores[idx]) if return_scores is not None and idx < len(return_scores) else float(utility_scores[idx]),
                "proposal_score": float(proposal_scores[idx]),
                "crop_ar": crop_ar,
                "target_ar_log_error": ar_error,
                "target_ar_compatible": bool(_target_ar_value(target_ar) is None or ar_error <= AR_COMPATIBILITY_LOG_ERROR_THRESHOLD),
            }
        )
    return rows


def _best_positive_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score", 0.0)))


def _per_record_metrics(
    *,
    row: dict[str, Any],
    selected: dict[str, Any],
    proposal_rows: list[dict[str, Any]],
    predicted_decision: int,
    predicted_route: int,
    label_decision: int,
    label_route: int,
) -> dict[str, float]:
    positive_rows = list(row.get("positive_records") or [])
    risk_rows = [cand for cand in (row.get("candidate_records") or []) if float(cand.get("risk_target", 0.0)) > 0.0]
    best_positive = _best_positive_row(positive_rows)
    selected_box = list(selected.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])
    positive_ious = [_box_iou(selected_box, list(item.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])) for item in positive_rows]
    risk_ious = [_box_iou(selected_box, list(item.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])) for item in risk_rows]
    ranked_proposal_rows = sorted(
        proposal_rows,
        key=lambda prop: (float(prop.get("proposal_score", prop.get("score", 0.0))), float(prop.get("utility_score", 0.0))),
        reverse=True,
    )
    prop_positive_ious = [
        max((_box_iou(list(prop.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0]), list(item.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])) for item in positive_rows), default=0.0)
        for prop in ranked_proposal_rows
    ]
    compatible_prop_ious = [prop_positive_ious[idx] for idx, prop in enumerate(ranked_proposal_rows) if bool(prop.get("target_ar_compatible", True))]
    action_source = str(selected.get("action_source", ""))
    label_is_baseline = int(label_decision != 2)
    final_is_baseline = int(action_source.startswith("baseline"))
    return {
        "final_positive_hit_iou_0_5": float(max(positive_ious, default=0.0) >= 0.5),
        "final_best_iou_to_positive": float(max(positive_ious, default=0.0)),
        "final_iou_to_best_positive": float(_box_iou(selected_box, list(best_positive.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])) if best_positive else 0.0),
        "final_risk_hit_iou_0_5": float(max(risk_ious, default=0.0) >= 0.5),
        "final_risk_overlap_iou": float(max(risk_ious, default=0.0)),
        "final_target_ar_log_error": float(selected.get("target_ar_log_error", 0.0)),
        "final_target_ar_compatible": float(bool(selected.get("target_ar_compatible", True))),
        "final_exact_target_ar_postprocessed": float(bool(selected.get("exact_ar_postprocessed", False))),
        "final_action_is_baseline": float(final_is_baseline),
        "final_action_is_full": float(action_source.startswith("baseline_full")),
        "final_action_is_minimal": float(action_source.startswith("baseline_minimal")),
        "final_action_is_crop": float(not action_source.startswith("baseline")),
        "executor_override_to_crop": float(bool(selected.get("executor_override_to_crop", False))),
        "label_action_is_baseline": float(label_is_baseline),
        "action_source_acc": float(final_is_baseline == label_is_baseline),
        "baseline_preserve_hit": float(final_is_baseline) if label_is_baseline else 0.0,
        "crop_action_hit": float(not final_is_baseline) if not label_is_baseline else 0.0,
        "proposal_best_iou_at_1": float(max(prop_positive_ious[:1], default=0.0)),
        "proposal_best_iou_at_5": float(max(prop_positive_ious[:5], default=0.0)),
        "proposal_positive_recall_at_1_iou_0_5": float(max(prop_positive_ious[:1], default=0.0) >= 0.5),
        "proposal_positive_recall_at_5_iou_0_5": float(max(prop_positive_ious[:5], default=0.0) >= 0.5),
        "proposal_target_ar_best_iou_at_5": float(max(compatible_prop_ious[:5], default=0.0)),
        "proposal_target_ar_recall_at_5_iou_0_5": float(max(compatible_prop_ious[:5], default=0.0) >= 0.5),
        "decision_acc": float(int(predicted_decision == label_decision)),
        "route_acc": float(int(predicted_route == label_route)),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MobileCropNet v4 Product-AR 직접 평가 보고서",
        "",
        f"- checkpoint: `{summary.get('checkpoint')}`",
        f"- eval_jsonl: `{summary.get('eval_jsonl')}`",
        f"- selection_policy: `{summary.get('selection_policy')}`",
        f"- proposal_top_m: `{summary.get('proposal_top_m')}`",
        f"- runtime_subject_prior_mode: `{summary.get('runtime_subject_prior_mode')}`",
        f"- exact_target_ar_postprocess: `{summary.get('exact_target_ar_postprocess')}`",
        f"- enforce_baseline_decision_gate: `{summary.get('enforce_baseline_decision_gate')}`",
        f"- baseline_decision_confidence_threshold: `{summary.get('baseline_decision_confidence_threshold')}`",
        f"- baseline_score_margin: `{summary.get('baseline_score_margin')}`",
        f"- subject_valid_threshold: `{summary.get('subject_valid_threshold', 0.5)}`",
        f"- subject_valid_policy: `{summary.get('subject_valid_policy', 'confidence')}`",
        f"- subject_box_target_source: `{summary.get('subject_box_target_source', 'legacy')}`",
        f"- 평가 이미지 수: `{summary.get('image_count', 0)}`",
        "",
        "## 핵심 지표",
        "",
        "| 지표 | 값 |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("metrics") or {}).items()):
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.extend(["", "## Target AR별 결과", "", "| target_ar | 이미지 | hit@0.5 | best_iou | route_acc | decision_acc |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for target_ar, payload in sorted((summary.get("by_target_ar") or {}).items()):
        lines.append(
            f"| {target_ar} | {int(payload.get('image_count', 0))} | {float(payload.get('final_positive_hit_iou_0_5', 0.0)):.6f} | "
            f"{float(payload.get('final_best_iou_to_positive', 0.0)):.6f} | {float(payload.get('route_acc', 0.0)):.6f} | {float(payload.get('decision_acc', 0.0)):.6f} |"
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
    use_decision_source_action_gate = (
        bool(args.use_decision_source_action_gate)
        if args.use_decision_source_action_gate is not None
        else bool(ckpt.get("train_config", {}).get("runtime_use_decision_source_action_gate", False))
    )
    candidate_k = int(args.candidate_k or max(32, int(ckpt.get("model_config", {}).get("candidate_k", 24))))
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
    row_shard_count = max(1, int(args.row_shard_count))
    row_shard_index = int(args.row_shard_index)
    if row_shard_index < 0 or row_shard_index >= row_shard_count:
        raise SystemExit(f"--row_shard_index must be in [0, {row_shard_count - 1}], got {row_shard_index}")
    original_image_count = len(dataset)
    if row_shard_count > 1:
        dataset.records = [
            row for row_index, row in enumerate(dataset.records) if (row_index % row_shard_count) == row_shard_index
        ]
        if not dataset.records:
            raise SystemExit(
                f"row shard {row_shard_index}/{row_shard_count} has no records after filtering {original_image_count} rows"
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )

    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    by_target_ar: dict[str, list[dict[str, float]]] = defaultdict(list)
    model.eval()
    started_at = time.monotonic()
    processed = 0
    progress_path = args.progress_json or (args.output_dir / "progress.json")
    _write_progress(
        progress_path,
        {
            "state": "running",
            "phase": "direct_eval",
            "checkpoint": str(args.checkpoint),
            "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
            "route_expert_checkpoints": [str(path) for path in route_expert_paths],
            "eval_jsonl": str(args.eval_jsonl),
            "output_dir": str(args.output_dir),
            "image_count": len(dataset),
            "original_image_count": int(original_image_count),
            "row_shard_index": int(row_shard_index),
            "row_shard_count": int(row_shard_count),
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
            subject_prior_box = tensor_batch.get("subject_prior_box") if args.runtime_subject_prior_mode == "teacher" else None
            subject_prior_valid = tensor_batch.get("subject_prior_valid") if args.runtime_subject_prior_mode == "teacher" else None
            subject_prior_reliability = tensor_batch.get("subject_prior_reliability") if args.runtime_subject_prior_mode == "teacher" else None
            outputs = model(
                tensor_batch["image"],
                None,
                None,
                tensor_batch["target_ar_id"],
                tensor_batch["image_ar_log"],
                None,
                None,
                tensor_batch.get("letterbox_content_box"),
                subject_prior_box,
                subject_prior_valid,
                subject_prior_reliability,
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
            utility = torch.sigmoid(outputs["utility_logits"]).detach().cpu()
            selection_logits = outputs.get(
                "source_gate_return_logits",
                outputs.get(
                    "source_mixture_return_logits",
                    outputs.get("decision_conditioned_return_logits", outputs.get("return_logits", outputs["utility_logits"])),
                ),
            )
            selection_utility = torch.sigmoid(selection_logits).detach().cpu()
            proposal = torch.sigmoid(outputs["proposal_logits"]).detach().cpu()
            proposal_boxes = outputs["proposal_boxes"].detach().cpu()
            decision_logits_cpu = outputs["decision_logits"].detach().cpu()
            decision_source_logit = outputs.get("decision_source_logit")
            decision_source_raw_logit = outputs.get("decision_source_raw_logit", decision_source_logit)
            decision_source_threshold = outputs.get("decision_source_logit_threshold")
            decision_source_logit_cpu = (
                decision_source_logit.detach().cpu() if torch.is_tensor(decision_source_logit) else None
            )
            decision_source_raw_logit_cpu = (
                decision_source_raw_logit.detach().cpu() if torch.is_tensor(decision_source_raw_logit) else None
            )
            source_gate_logit = outputs.get("source_gate_logit")
            source_gate_logit_cpu = source_gate_logit.detach().cpu() if torch.is_tensor(source_gate_logit) else None
            action_gate_signal = "decision_source_logit" if decision_source_logit_cpu is not None else None
            action_gate_logit_cpu = decision_source_logit_cpu
            action_gate_raw_logit_cpu = decision_source_raw_logit_cpu
            if action_gate_logit_cpu is None and source_gate_logit_cpu is not None:
                action_gate_signal = "source_gate_logit"
                action_gate_logit_cpu = source_gate_logit_cpu
                action_gate_raw_logit_cpu = source_gate_logit_cpu
            decision_source_threshold_value = (
                float(decision_source_threshold.detach().cpu().view(-1)[0].item())
                if torch.is_tensor(decision_source_threshold) and decision_source_threshold.numel() >= 1
                else None
            )
            baseline_utility = outputs.get(
                "runtime_baseline_source_gate_return_logits",
                outputs.get(
                    "runtime_baseline_source_mixture_return_logits",
                    outputs.get(
                        "runtime_baseline_decision_conditioned_return_logits",
                        outputs.get("runtime_baseline_return_logits", outputs.get("runtime_baseline_utility_logits")),
                    ),
                ),
            )
            baseline_utility_cpu = baseline_utility.detach().cpu() if torch.is_tensor(baseline_utility) else None
            pred_subject_boxes = outputs.get("pred_subject_box")
            pred_subject_valid = outputs.get("pred_subject_valid")
            pred_subject_boxes_cpu = pred_subject_boxes.detach().cpu() if torch.is_tensor(pred_subject_boxes) else None
            pred_subject_valid_cpu = pred_subject_valid.detach().cpu() if torch.is_tensor(pred_subject_valid) else None
            route_ids = outputs["route_logits"].argmax(dim=1).detach().cpu()
            decision_ids = outputs["decision_logits"].argmax(dim=1).detach().cpu()
            for idx in range(utility.shape[0]):
                target_ar = str(batch["target_ar"][idx])
                proposal_rows = _proposal_rows(
                    proposal_boxes=proposal_boxes[idx].tolist(),
                    proposal_scores=proposal[idx].tolist(),
                    utility_scores=utility[idx].tolist(),
                    return_scores=selection_utility[idx].tolist(),
                    transform=batch["transform"][idx],
                    width=int(batch["width"][idx]),
                    height=int(batch["height"][idx]),
                    target_ar=target_ar,
                )
                selected_idx = select_generated_proposal_index(
                    utility_scores=selection_utility[idx].tolist(),
                    proposal_scores=proposal[idx].tolist(),
                    proposals=proposal_rows,
                    selection_policy=args.selection_policy,
                    proposal_top_m=int(args.proposal_top_m),
                )
                baseline_rows = build_runtime_baselines(
                    width=int(batch["width"][idx]),
                    height=int(batch["height"][idx]),
                    target_ar=target_ar,
                )
                if baseline_utility_cpu is not None:
                    for base_idx, label in enumerate(("full", "minimal")):
                        baseline_rows[label]["score"] = float(torch.sigmoid(baseline_utility_cpu[idx, base_idx]).item())
                        baseline_rows[label]["utility_score"] = baseline_rows[label]["score"]
                selected = execute_runtime_decision(
                    decision_id=int(decision_ids[idx].item()),
                    target_ar=target_ar,
                    width=int(batch["width"][idx]),
                    height=int(batch["height"][idx]),
                    proposals=proposal_rows,
                    selected_proposal_index=selected_idx,
                    exact_target_ar_postprocess=args.exact_target_ar_postprocess,
                    baseline_rows=baseline_rows,
                    decision_logits=decision_logits_cpu[idx].tolist(),
                    decision_source_logit=float(action_gate_logit_cpu[idx].item())
                    if action_gate_logit_cpu is not None
                    else None,
                    decision_source_raw_logit=float(action_gate_raw_logit_cpu[idx].item())
                    if action_gate_raw_logit_cpu is not None
                    else None,
                    decision_source_logit_threshold=decision_source_threshold_value,
                    use_decision_source_action_gate=bool(use_decision_source_action_gate),
                    enforce_baseline_decision_gate=args.enforce_baseline_decision_gate,
                    baseline_decision_confidence_threshold=args.baseline_decision_confidence_threshold,
                    baseline_score_margin=args.baseline_score_margin,
                )
                selected["action_gate_signal"] = action_gate_signal
                if 0 <= selected_idx < len(proposal_rows):
                    selected["proposal_score"] = float(proposal[idx, selected_idx].item())
                    selected["utility_score"] = float(utility[idx, selected_idx].item())
                    selected["return_score"] = float(selection_utility[idx, selected_idx].item())
                subject_box: dict[str, Any] = {}
                target_valid_tensor = batch.get("subject_box_valid", batch.get("subject_prior_valid"))
                target_box_tensor = batch.get("subject_box_target", batch.get("subject_prior_box"))
                if pred_subject_boxes_cpu is not None and pred_subject_valid_cpu is not None:
                    pred_lb = [float(v) for v in pred_subject_boxes_cpu[idx].tolist()]
                    pred_conf = float(pred_subject_valid_cpu[idx].item())
                    route_idx = int(route_ids[idx].item())
                    route_label = SUBJECT_MODE_VOCAB[route_idx] if 0 <= route_idx < len(SUBJECT_MODE_VOCAB) else ""
                    pred_valid = subject_valid_from_policy(
                        confidence=pred_conf,
                        threshold=float(subject_valid_threshold),
                        route_label=route_label,
                        policy=str(args.subject_valid_policy),
                    )
                    subject_box["predicted"] = {
                        "bbox_letterbox_xyxy": pred_lb,
                        "bbox_norm_xyxy": letterbox_to_original_box(pred_lb, batch["transform"][idx]),
                        "confidence": pred_conf,
                        "valid_threshold": float(subject_valid_threshold),
                        "valid_policy": str(args.subject_valid_policy),
                        "route_label": route_label,
                        "confidence_valid": bool(pred_conf >= float(subject_valid_threshold)),
                        "valid": bool(pred_valid),
                    }
                if torch.is_tensor(target_valid_tensor) and torch.is_tensor(target_box_tensor):
                    target_valid = float(target_valid_tensor[idx].item())
                    target_lb = [float(v) for v in target_box_tensor[idx].tolist()]
                    subject_box["target_valid"] = target_valid
                    if target_valid > 0.0:
                        subject_box["teacher"] = {
                            "bbox_letterbox_xyxy": target_lb,
                            "bbox_norm_xyxy": letterbox_to_original_box(target_lb, batch["transform"][idx]),
                            "confidence": float(batch["subject_prior_reliability"][idx].item()) if torch.is_tensor(batch.get("subject_prior_reliability")) else None,
                        }
                if isinstance(subject_box.get("predicted"), dict) and isinstance(subject_box.get("teacher"), dict):
                    subject_box["iou_to_teacher"] = _box_iou(subject_box["predicted"]["bbox_norm_xyxy"], subject_box["teacher"]["bbox_norm_xyxy"])
                row = {
                    "image_id": batch["image_id"][idx],
                    "image_path": batch["image_path"][idx],
                    "target_ar": target_ar,
                    "positive_records": batch["positive_records"][idx],
                    "candidate_records": batch["candidate_records"][idx],
                }
                metrics = _per_record_metrics(
                    row=row,
                    selected=selected,
                    proposal_rows=proposal_rows,
                    predicted_decision=int(selected.get("runtime_decision_id", int(decision_ids[idx].item()))),
                    predicted_route=int(route_ids[idx].item()),
                    label_decision=int(batch["decision_target"][idx].item()),
                    label_route=int(batch["route_target"][idx].item()),
                )
                if subject_box:
                    target_valid = float(subject_box.get("target_valid", 0.0) or 0.0)
                    pred_conf = float(subject_box.get("predicted", {}).get("confidence", 0.0)) if isinstance(subject_box.get("predicted"), dict) else 0.0
                    pred_positive = bool(subject_box.get("predicted", {}).get("valid", False)) if isinstance(subject_box.get("predicted"), dict) else False
                    target_positive = target_valid >= 0.5
                    metrics["subject_box_target_valid"] = target_valid
                    metrics["subject_box_pred_conf"] = pred_conf
                    metrics["subject_box_valid_threshold"] = float(subject_valid_threshold)
                    metrics["subject_box_valid_acc"] = float(pred_positive == target_positive)
                    if target_positive:
                        metrics["subject_box_valid_positive_acc"] = float(pred_positive)
                    else:
                        metrics["subject_box_valid_negative_acc"] = float(not pred_positive)
                    if "iou_to_teacher" in subject_box:
                        metrics["subject_box_iou_to_teacher"] = float(subject_box["iou_to_teacher"])
                rows.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "target_ar": target_ar,
                        "selection_policy": args.selection_policy,
                        "runtime_subject_prior_mode": args.runtime_subject_prior_mode,
                        "exact_target_ar_postprocess": bool(args.exact_target_ar_postprocess),
                        "enforce_baseline_decision_gate": bool(args.enforce_baseline_decision_gate),
                        "baseline_decision_confidence_threshold": float(args.baseline_decision_confidence_threshold),
                        "baseline_score_margin": float(args.baseline_score_margin),
                        "subject_valid_threshold": float(subject_valid_threshold),
                        "subject_valid_policy": str(args.subject_valid_policy),
                        "subject_box_target_source": subject_box_target_source,
                        "selected": selected,
                        "runtime_baselines": baseline_rows,
                        "selected_proposal_index": int(selected_idx),
                        "proposal_rows": proposal_rows,
                        "metrics": metrics,
                        "subject_box": subject_box,
                    }
                )
                metric_rows.append(metrics)
                by_target_ar[target_ar].append(metrics)
            processed += int(utility.shape[0])
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                elapsed = max(1e-6, time.monotonic() - started_at)
                payload = {
                    "state": "running",
                    "phase": "direct_eval",
                    "checkpoint": str(args.checkpoint),
                    "output_dir": str(args.output_dir),
                    "image_count": len(dataset),
                    "original_image_count": int(original_image_count),
                    "row_shard_index": int(row_shard_index),
                    "row_shard_count": int(row_shard_count),
                    "processed_batches": batch_idx,
                    "processed_images": processed,
                    "samples_per_sec": round(processed / elapsed, 4),
                    "elapsed_sec": round(elapsed, 3),
                    "metrics_json": str(args.output_dir / "metrics.json"),
                    "predictions_jsonl": str(args.output_dir / "predictions.jsonl"),
                }
                _write_progress(progress_path, payload)
                print(
                    json.dumps({"event": "direct_eval_progress", **payload}, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )

    summary = {
        "checkpoint": str(args.checkpoint),
        "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
        "route_expert_checkpoints": [str(path) for path in route_expert_paths],
        "route_expert_weights": route_expert_weights,
        "route_expert_source": route_expert_source,
        "embedded_route_expert_count": embedded_route_expert_count,
        "eval_jsonl": str(args.eval_jsonl),
        "selection_policy": args.selection_policy,
        "proposal_top_m": int(args.proposal_top_m),
        "runtime_subject_prior_mode": args.runtime_subject_prior_mode,
        "exact_target_ar_postprocess": bool(args.exact_target_ar_postprocess),
        "use_decision_source_action_gate": bool(use_decision_source_action_gate),
        "enforce_baseline_decision_gate": bool(args.enforce_baseline_decision_gate),
        "baseline_decision_confidence_threshold": float(args.baseline_decision_confidence_threshold),
        "baseline_score_margin": float(args.baseline_score_margin),
        "subject_valid_threshold": float(subject_valid_threshold),
        "subject_valid_policy": str(args.subject_valid_policy),
        "subject_box_target_source": subject_box_target_source,
        "image_count": len(rows),
        "original_image_count": int(original_image_count),
        "row_shard_index": int(row_shard_index),
        "row_shard_count": int(row_shard_count),
        "dataset_summary": dataset.summary(),
        "metrics": summarize_metrics(metric_rows),
        "by_target_ar": {
            key: {"image_count": len(value), **summarize_metrics(value)}
            for key, value in sorted(by_target_ar.items())
        },
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "metrics.json").write_text(summary_text, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_report(args.output_dir / "EVAL_REPORT.md", summary)
    _write_progress(
        progress_path,
        {
            "state": "completed",
            "phase": "direct_eval",
            "checkpoint": str(args.checkpoint),
            "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
            "route_expert_checkpoints": [str(path) for path in route_expert_paths],
            "eval_jsonl": str(args.eval_jsonl),
            "output_dir": str(args.output_dir),
            "image_count": len(rows),
            "original_image_count": int(original_image_count),
            "row_shard_index": int(row_shard_index),
            "row_shard_count": int(row_shard_count),
            "processed_batches": len(loader),
            "processed_images": len(rows),
            "metrics_json": str(args.output_dir / "metrics.json"),
            "summary_json": str(args.output_dir / "summary.json"),
            "predictions_jsonl": str(args.output_dir / "predictions.jsonl"),
        },
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
