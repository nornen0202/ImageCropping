#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    DECISION_VOCAB,
    SUBJECT_BOX_TARGET_SOURCES,
    SUBJECT_MODE_VOCAB,
    MobileCropNetV4BatchDataset,
    box_iou_xyxy,
    mobilecropnet_v4_collate,
)
from mobilecropnet_v4.eval_utils import (
    SUBJECT_VALID_POLICIES,
    batch_predictions,
    infer_image_norm,
    infer_input_size,
    load_mobilecropnet_v4_checkpoint,
    per_record_metrics,
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
    parser = argparse.ArgumentParser(description="Audit MobileCropNet v4 release-gate heads on a fixed validation JSONL.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--candidate_k", type=int)
    parser.add_argument("--image_mean")
    parser.add_argument("--image_std")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=2400)
    parser.add_argument("--save_topk", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--route_gate", type=float, default=0.52)
    parser.add_argument("--subject_iou_gate", type=float, default=0.50)
    parser.add_argument("--valid_conf_threshold", "--subject_valid_threshold", type=float, default=0.50)
    parser.add_argument("--subject_valid_policy", choices=SUBJECT_VALID_POLICIES, default="confidence")
    parser.add_argument("--subject_box_target_source", choices=SUBJECT_BOX_TARGET_SOURCES, default=None)
    parser.add_argument("--route_expert_checkpoint", type=Path, default=None)
    parser.add_argument("--route_expert_checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--route_expert_weights", default=None, help="Comma-separated weights for route expert ensemble.")
    return parser


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _label(vocab: list[str] | tuple[str, ...], idx: int) -> str:
    return vocab[idx] if 0 <= idx < len(vocab) else f"class_{idx}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _confusion_payload(confusion: list[list[int]], labels: list[str]) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []
    total_correct = 0
    total = 0
    recalls = []
    precisions = []
    for idx, label in enumerate(labels):
        row_total = int(sum(confusion[idx]))
        col_total = int(sum(confusion[r][idx] for r in range(len(labels))))
        correct = int(confusion[idx][idx])
        total_correct += correct
        total += row_total
        recall = float(correct / row_total) if row_total else None
        precision = float(correct / col_total) if col_total else None
        if recall is not None:
            recalls.append(recall)
        if precision is not None:
            precisions.append(precision)
        per_class.append(
            {
                "class_id": idx,
                "label": label,
                "target_count": row_total,
                "pred_count": col_total,
                "correct": correct,
                "recall": recall,
                "precision": precision,
            }
        )
    return {
        "labels": labels,
        "matrix_target_by_pred": confusion,
        "accuracy": float(total_correct / total) if total else 0.0,
        "balanced_accuracy": _mean(recalls),
        "macro_precision": _mean(precisions),
        "per_class": per_class,
    }


def _bin_key(value: float) -> str:
    clipped = max(0.0, min(0.999999, float(value)))
    left = int(clipped * 10) / 10.0
    right = left + 0.1
    return f"{left:.1f}-{right:.1f}"


def _subject_valid_calibration_payload(
    rows: list[tuple[float, bool, str]],
    *,
    threshold: float,
    policy: str = "confidence",
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for conf, target_pos, route_label in rows:
        pred_pos = subject_valid_from_policy(
            confidence=float(conf),
            threshold=float(threshold),
            route_label=str(route_label),
            policy=str(policy),
        )
        if pred_pos and target_pos:
            tp += 1
        elif pred_pos and not target_pos:
            fp += 1
        elif (not pred_pos) and target_pos:
            fn += 1
        else:
            tn += 1
    total = max(1, tp + fp + tn + fn)
    positive_total = max(1, tp + fn)
    negative_total = max(1, tn + fp)
    positive_acc = float(tp / positive_total)
    negative_acc = float(tn / negative_total)
    return {
        "threshold": float(threshold),
        "policy": str(policy),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float((tp + tn) / total),
        "precision": float(tp / max(1, tp + fp)),
        "positive_accuracy": positive_acc,
        "negative_accuracy": negative_acc,
        "balanced_accuracy": float((positive_acc + negative_acc) / 2.0),
        "release_gate_pass": bool(positive_acc >= 0.50 and negative_acc >= 0.60 and ((tp + tn) / total) >= 0.55),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    route = summary["route"]
    subject = summary["subject_box"]
    failure = summary["failure_taxonomy"]
    lines = [
        "# MobileCropNet v4.0 릴리스 헤드 감사 보고서",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- eval_jsonl: `{summary['eval_jsonl']}`",
        f"- subject_box_target_source: `{summary.get('subject_box_target_source', 'legacy')}`",
        f"- 평가 이미지 수: `{summary['image_count']}`",
        "",
        "## Gate 스냅샷",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| route accuracy | {route['accuracy']:.6f} |",
        f"| route balanced accuracy | {route['balanced_accuracy']:.6f} |",
        f"| subject valid IoU mean | {subject['valid_iou_mean']:.6f} |",
        f"| subject valid accuracy | {subject['valid_accuracy']:.6f} |",
        f"| subject valid positive acc | {_safe_float(subject.get('valid_positive_acc')):.6f} |",
        f"| subject valid negative acc | {_safe_float(subject.get('valid_negative_acc')):.6f} |",
        f"| recommended valid threshold | {_safe_float(subject.get('recommended_valid_conf_threshold')):.6f} |",
        f"| subject valid policy | `{subject.get('valid_policy', 'confidence')}` |",
        f"| top1 hit | {_safe_float(metrics.get('candidate_top1_hit')):.6f} |",
        f"| top1 IoU to best positive | {_safe_float(metrics.get('top1_iou_to_best_positive')):.6f} |",
        f"| proposal target-AR top1 compatible | {_safe_float(metrics.get('proposal_top1_target_ar_compatible')):.6f} |",
        f"| checklist agreement | {_safe_float(metrics.get('explain_label_agreement_when_applicable')):.6f} |",
        "",
        "## Route 클래스별 결과",
        "",
        "| route | target | pred | recall | precision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in route["per_class"]:
        recall = "-" if row["recall"] is None else f"{row['recall']:.6f}"
        precision = "-" if row["precision"] is None else f"{row['precision']:.6f}"
        lines.append(f"| `{row['label']}` | {row['target_count']} | {row['pred_count']} | {recall} | {precision} |")
    lines.extend(
        [
            "",
            "## 실패 Taxonomy",
            "",
            "| bucket | 건수 | 비율 |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, row in sorted(failure.items()):
        lines.append(f"| `{key}` | {row['count']} | {row['rate']:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    status_path.write_text(json.dumps({"state": "running", "checkpoint": str(args.checkpoint)}, indent=2) + "\n", encoding="utf-8")

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

    n_route = len(SUBJECT_MODE_VOCAB)
    n_decision = len(DECISION_VOCAB)
    route_conf = [[0 for _ in range(n_route)] for _ in range(n_route)]
    decision_conf = [[0 for _ in range(n_decision)] for _ in range(n_decision)]
    route_confidence: list[float] = []
    metric_rows: list[dict[str, float]] = []
    target_ar_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    subject_iou_values: list[float] = []
    subject_conf_values: list[float] = []
    subject_valid_calibration_rows: list[tuple[float, bool, str]] = []
    subject_valid_correct = 0
    subject_valid_total = 0
    subject_valid_tp = subject_valid_fp = subject_valid_fn = 0
    subject_by_route: dict[str, list[float]] = defaultdict(list)
    subject_conf_bins: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "target_positive": 0, "pred_positive": 0, "iou_sum": 0.0})
    failures = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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
            route_logits = outputs["route_logits"].detach()
            route_pred = route_logits.argmax(dim=1).cpu()
            route_prob = torch.softmax(route_logits.float(), dim=1).max(dim=1).values.cpu()
            route_target = batch["route_target"].cpu()
            decision_pred = outputs["decision_logits"].argmax(dim=1).detach().cpu()
            decision_target = batch["decision_target"].cpu()
            pred_subject = outputs.get("pred_subject_box")
            pred_subject_valid = outputs.get("pred_subject_valid")
            if torch.is_tensor(pred_subject):
                pred_subject = pred_subject.detach().cpu()
            if torch.is_tensor(pred_subject_valid):
                pred_subject_valid = pred_subject_valid.detach().cpu()
            target_subject = batch.get("subject_box_target")
            target_subject_valid = batch.get("subject_box_valid")
            preds = batch_predictions(
                outputs=outputs,
                batch=batch,
                save_topk=args.save_topk,
                subject_valid_threshold=float(args.valid_conf_threshold),
                subject_valid_policy=str(args.subject_valid_policy),
            )
            for i, pred in enumerate(preds):
                rt = int(route_target[i].item())
                rp = int(route_pred[i].item())
                dt = int(decision_target[i].item())
                dp = int(decision_pred[i].item())
                route_conf[rt][rp] += 1
                decision_conf[dt][dp] += 1
                route_confidence.append(float(route_prob[i].item()))

                route_label = _label(SUBJECT_MODE_VOCAB, rp)
                metrics = per_record_metrics(
                    pred,
                    subject_valid_conf_threshold=float(args.valid_conf_threshold),
                    subject_valid_policy=str(args.subject_valid_policy),
                )
                metric_rows.append(metrics)
                target_ar_rows[str(pred.get("target_ar", "FREE"))].append(metrics)

                subj_valid = float(target_subject_valid[i].item()) if torch.is_tensor(target_subject_valid) else 0.0
                subj_conf = float(pred_subject_valid[i].item()) if torch.is_tensor(pred_subject_valid) else 0.0
                subj_pred_pos = subject_valid_from_policy(
                    confidence=subj_conf,
                    threshold=float(args.valid_conf_threshold),
                    route_label=route_label,
                    policy=str(args.subject_valid_policy),
                )
                subj_target_pos = subj_valid >= 0.5
                subject_conf_values.append(subj_conf)
                subject_valid_calibration_rows.append((subj_conf, subj_target_pos, route_label))
                subject_valid_correct += int(subj_pred_pos == subj_target_pos)
                subject_valid_total += 1
                if subj_pred_pos and subj_target_pos:
                    subject_valid_tp += 1
                elif subj_pred_pos and not subj_target_pos:
                    subject_valid_fp += 1
                elif (not subj_pred_pos) and subj_target_pos:
                    subject_valid_fn += 1
                subj_iou = 0.0
                if subj_target_pos and torch.is_tensor(pred_subject) and torch.is_tensor(target_subject):
                    subj_iou = box_iou_xyxy([float(v) for v in pred_subject[i].tolist()], [float(v) for v in target_subject[i].tolist()])
                    subject_iou_values.append(subj_iou)
                    subject_by_route[_label(SUBJECT_MODE_VOCAB, rt)].append(subj_iou)
                bin_row = subject_conf_bins[_bin_key(subj_conf)]
                bin_row["count"] += 1
                bin_row["target_positive"] += int(subj_target_pos)
                bin_row["pred_positive"] += int(subj_pred_pos)
                bin_row["iou_sum"] += subj_iou

                def add_failure(key: str) -> None:
                    failures[key] += 1
                    if len(examples[key]) < 12:
                        examples[key].append(
                            {
                                "image_id": pred.get("image_id"),
                                "target_ar": pred.get("target_ar"),
                                "route_target": _label(SUBJECT_MODE_VOCAB, rt),
                                "route_pred": _label(SUBJECT_MODE_VOCAB, rp),
                                "decision_target": _label(DECISION_VOCAB, dt),
                                "decision_pred": _label(DECISION_VOCAB, dp),
                                "subject_iou": subj_iou,
                                "subject_conf": subj_conf,
                                "top1_hit": metrics.get("candidate_top1_hit"),
                                "top1_iou": metrics.get("top1_iou_to_best_positive"),
                            }
                        )

                if rp != rt:
                    add_failure("route_mismatch")
                if dp != dt:
                    add_failure("decision_mismatch")
                if subj_target_pos and subj_iou < float(args.subject_iou_gate):
                    add_failure("low_subject_iou")
                if subj_pred_pos != subj_target_pos:
                    add_failure("subject_valid_mismatch")
                if _safe_float(metrics.get("candidate_top1_hit")) < 0.5:
                    add_failure("top1_not_positive")
                if _safe_float(metrics.get("top1_iou_to_best_positive")) < 0.5:
                    add_failure("low_top1_iou")
                if _safe_float(metrics.get("proposal_target_ar_recall_at_5_iou_0_5")) < 0.5:
                    add_failure("low_target_ar_proposal_recall")
                if _safe_float(metrics.get("proposal_top1_target_ar_compatible"), 1.0) < 0.5:
                    add_failure("target_ar_incompatible")
                if _safe_float(metrics.get("explain_label_agreement_when_applicable"), 1.0) < 0.5:
                    add_failure("checklist_low_agreement")

            status_path.write_text(
                json.dumps(
                    {
                        "state": "running",
                        "processed_batches": batch_idx + 1,
                        "processed_images": len(metric_rows),
                        "checkpoint": str(args.checkpoint),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    route_summary = _confusion_payload(route_conf, list(SUBJECT_MODE_VOCAB))
    decision_summary = _confusion_payload(decision_conf, list(DECISION_VOCAB))
    metrics = summarize_metrics(metric_rows)
    by_target_ar = {
        ar: {"image_count": len(rows), **summarize_metrics(rows)}
        for ar, rows in sorted(target_ar_rows.items())
    }
    image_count = len(metric_rows)
    subject_bins = {}
    for key, row in sorted(subject_conf_bins.items()):
        count = int(row["count"])
        subject_bins[key] = {
            "count": count,
            "target_positive_rate": float(row["target_positive"] / count) if count else 0.0,
            "pred_positive_rate": float(row["pred_positive"] / count) if count else 0.0,
            "valid_iou_mean": float(row["iou_sum"] / max(1, row["target_positive"])),
        }
    valid_threshold_sweep = [
        _subject_valid_calibration_payload(
            subject_valid_calibration_rows,
            threshold=threshold / 100.0,
            policy=str(args.subject_valid_policy),
        )
        for threshold in range(30, 91, 5)
    ]
    current_valid_calibration = _subject_valid_calibration_payload(
        subject_valid_calibration_rows,
        threshold=float(args.valid_conf_threshold),
        policy=str(args.subject_valid_policy),
    )
    passing_valid_thresholds = [row for row in valid_threshold_sweep if row["release_gate_pass"]]
    recommended_valid_threshold = max(
        passing_valid_thresholds or valid_threshold_sweep,
        key=lambda row: (
            1 if row["release_gate_pass"] else 0,
            row["balanced_accuracy"],
            -abs(float(row["threshold"]) - float(args.valid_conf_threshold)),
        ),
    )
    subject_summary = {
        "target_positive_count": int(sum(1 for v in subject_iou_values)),
        "valid_conf_threshold": float(args.valid_conf_threshold),
        "valid_policy": str(args.subject_valid_policy),
        "valid_iou_mean": _mean(subject_iou_values),
        "pred_conf_mean": _mean(subject_conf_values),
        "valid_accuracy": current_valid_calibration["accuracy"],
        "valid_precision": current_valid_calibration["precision"],
        "valid_recall": current_valid_calibration["positive_accuracy"],
        "valid_positive_acc": current_valid_calibration["positive_accuracy"],
        "valid_negative_acc": current_valid_calibration["negative_accuracy"],
        "valid_balanced_acc": current_valid_calibration["balanced_accuracy"],
        "valid_release_gate_pass": current_valid_calibration["release_gate_pass"],
        "recommended_valid_conf_threshold": recommended_valid_threshold["threshold"],
        "recommended_valid_calibration": recommended_valid_threshold,
        "valid_threshold_sweep": valid_threshold_sweep,
        "by_route_target": {
            label: {"target_positive_count": len(values), "valid_iou_mean": _mean(values)}
            for label, values in sorted(subject_by_route.items())
        },
        "confidence_bins": subject_bins,
    }
    failure_payload = {
        key: {"count": int(count), "rate": float(count / max(1, image_count))}
        for key, count in sorted(failures.items(), key=lambda item: (-item[1], item[0]))
    }
    release_gate = {
        "route_balanced_pass": bool(route_summary["balanced_accuracy"] >= float(args.route_gate)),
        "subject_iou_pass": bool(subject_summary["valid_iou_mean"] >= float(args.subject_iou_gate)),
        "subject_valid_calibration_pass": bool(subject_summary["valid_release_gate_pass"]),
        "route_gate_threshold": float(args.route_gate),
        "subject_iou_gate_threshold": float(args.subject_iou_gate),
        "subject_valid_gate_thresholds": {
            "accuracy_min": 0.55,
            "positive_acc_min": 0.50,
            "negative_acc_min": 0.60,
        },
        "deploy_candidate": False,
    }
    release_gate["deploy_candidate"] = bool(
        release_gate["route_balanced_pass"]
        and release_gate["subject_iou_pass"]
        and release_gate["subject_valid_calibration_pass"]
    )
    summary = {
        "state": "completed",
        "checkpoint": str(args.checkpoint),
        "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
        "route_expert_checkpoints": [str(path) for path in route_expert_paths],
        "route_expert_weights": route_expert_weights,
        "route_expert_source": route_expert_source,
        "embedded_route_expert_count": embedded_route_expert_count,
        "eval_jsonl": str(args.eval_jsonl),
        "image_count": image_count,
        "input_size": input_size,
        "candidate_k": candidate_k,
        "subject_box_target_source": subject_box_target_source,
        "dataset_summary": dataset.summary(),
        "metrics": metrics,
        "by_target_ar": by_target_ar,
        "route": route_summary,
        "route_confidence_mean": _mean(route_confidence),
        "decision": decision_summary,
        "subject_box": subject_summary,
        "failure_taxonomy": failure_payload,
        "failure_examples": examples,
        "release_gate": release_gate,
    }
    (args.output_dir / "release_head_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(args.output_dir / "RELEASE_HEAD_AUDIT.md", summary)
    status_path.write_text(json.dumps({"state": "completed", "image_count": image_count, "summary": str(args.output_dir / "release_head_audit_summary.json")}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "completed", "route_balanced_acc": route_summary["balanced_accuracy"], "subject_iou": subject_summary["valid_iou_mean"], "top1_hit": metrics.get("candidate_top1_hit")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
