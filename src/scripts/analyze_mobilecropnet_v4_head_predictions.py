#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import CHECKLIST_CLASS_KEYS, CHECKLIST_CLASS_VOCABS, CHECKLIST_SCORE_KEYS, DECISION_VOCAB, SUBJECT_MODE_VOCAB, WHY_TAG_VOCAB
from mobilecropnet_v4.eval_utils import box_iou_xyxy, safe_float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 보조 head/근거/정책/proposal 예측을 분석한다.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--eval_jsonl", type=Path, default=None, help="Optional source batch JSONL to recover row-level teacher routing/policy metadata.")
    parser.add_argument("--top_k_failures", type=int, default=200)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Counter):
        return dict(value)
    raise TypeError(f"unsupported type for json serialization: {type(value)!r}")


def _row_key(image_id: Any, target_ar: Any) -> tuple[str, str]:
    return str(image_id or ""), str(target_ar or "FREE")


def _decision_label_from_row(row: dict[str, Any]) -> str:
    decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    raw = decision.get("decision_id", decision.get("decision_type", 2))
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        label = str(raw or "").strip()
        if label in DECISION_VOCAB:
            return label
        idx = 2
    idx = max(0, min(len(DECISION_VOCAB) - 1, idx))
    return DECISION_VOCAB[idx]


def _subject_mode_label_from_row(row: dict[str, Any]) -> str:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    label = str(routing.get("subject_mode") or "")
    if label in SUBJECT_MODE_VOCAB:
        return label
    try:
        idx = int(routing.get("subject_mode_id", 0))
    except (TypeError, ValueError):
        idx = 0
    idx = max(0, min(len(SUBJECT_MODE_VOCAB) - 1, idx))
    return SUBJECT_MODE_VOCAB[idx]


def _load_eval_row_map(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_jsonl(path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
        flags = routing.get("flags") if isinstance(routing.get("flags"), dict) else {}
        subject_overlay = routing.get("subject_support_overlay") if isinstance(routing.get("subject_support_overlay"), dict) else {}
        out[_row_key(row.get("image_id"), row.get("target_ar"))] = {
            "subject_mode_label": _subject_mode_label_from_row(row),
            "decision_label": _decision_label_from_row(row),
            "policy_id": str(routing.get("policy_id") or ""),
            "router_rule_id": str(routing.get("router_rule_id") or routing.get("rule_id") or ""),
            "subject_prior_bbox_norm_xyxy": routing.get("subject_prior_bbox_norm_xyxy"),
            "subject_reliability": safe_float(flags.get("subject_reliability"), safe_float(subject_overlay.get("subject_reliability"), 0.0)),
            "subject_support_trust_tier": str(flags.get("subject_support_trust_tier") or subject_overlay.get("support_trust_tier") or ""),
            "subject_repr_type": str(flags.get("subject_repr_type") or subject_overlay.get("subject_repr_type") or ""),
            "policy_head_row": row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {},
            "routing_row": routing,
        }
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = float(x) - mx
        dy = float(y) - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
    if denom <= 1e-12:
        return None
    return num / denom


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 1e-12:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _score_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _proposal_subject_alignment(
    pred: dict[str, Any],
    eval_meta: dict[str, Any] | None,
) -> dict[str, float | None]:
    subject_box = eval_meta.get("subject_prior_bbox_norm_xyxy") if isinstance(eval_meta, dict) else None
    proposal = pred.get("proposal_top1_target_ar") if isinstance(pred.get("proposal_top1_target_ar"), dict) else None
    selected = pred.get("top_candidate") if isinstance(pred.get("top_candidate"), dict) else None
    best_positive = pred.get("best_positive") if isinstance(pred.get("best_positive"), dict) else None
    if not isinstance(subject_box, list):
        return {
            "proposal_to_subject_iou": None,
            "selected_to_subject_iou": None,
            "best_positive_to_subject_iou": None,
            "proposal_subject_hit_iou_0_5": None,
        }
    proposal_box = proposal.get("bbox_norm_xyxy") if isinstance(proposal, dict) else None
    selected_box = selected.get("bbox_norm_xyxy") if isinstance(selected, dict) else None
    best_box = best_positive.get("bbox_norm_xyxy") if isinstance(best_positive, dict) else None
    proposal_iou = box_iou_xyxy(proposal_box, subject_box) if isinstance(proposal_box, list) else None
    selected_iou = box_iou_xyxy(selected_box, subject_box) if isinstance(selected_box, list) else None
    best_iou = box_iou_xyxy(best_box, subject_box) if isinstance(best_box, list) else None
    return {
        "proposal_to_subject_iou": proposal_iou,
        "selected_to_subject_iou": selected_iou,
        "best_positive_to_subject_iou": best_iou,
        "proposal_subject_hit_iou_0_5": 1.0 if proposal_iou is not None and proposal_iou >= 0.5 else 0.0 if proposal_iou is not None else None,
    }


def _summarize_binary_counts(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    route = summary.get("route", {})
    decision = summary.get("decision", {})
    proposal = summary.get("proposal", {})
    why_tags = summary.get("why_tags", {})
    lines = [
        "# MobileCropNet v4 Head 분석 보고서",
        "",
        f"- 예측 JSONL: `{summary.get('predictions_jsonl')}`",
        f"- 평가 원본 JSONL: `{summary.get('eval_jsonl') or ''}`",
        f"- 이미지 수: {summary.get('image_count', 0)}",
        "",
        "## 요약",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| route_acc | {safe_float(route.get('accuracy'), 0.0):.6f} |",
        f"| decision_acc | {safe_float(decision.get('accuracy'), 0.0):.6f} |",
        f"| proposal_r@5_iou0.5 | {safe_float(proposal.get('proposal_recall_at_5_iou_0_5'), 0.0):.6f} |",
        f"| proposal_subject_hit@0.5 | {safe_float(proposal.get('proposal_subject_hit_iou_0_5'), 0.0):.6f} |",
        f"| proposal_subject_iou_mean | {safe_float(proposal.get('proposal_to_subject_iou_mean'), 0.0):.6f} |",
        f"| selected_subject_iou_mean | {safe_float(proposal.get('selected_to_subject_iou_mean'), 0.0):.6f} |",
        f"| why_tag_micro_f1 | {safe_float(why_tags.get('micro', {}).get('f1'), 0.0):.6f} |",
        "",
        "## 주체 모드별 Proposal",
        "",
        "| 주체 모드 | 이미지 | route_acc | proposal_r@5 | proposal_subject_iou | selected_subject_iou | best_positive_subject_iou |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, row in sorted((summary.get("by_subject_mode") or {}).items()):
        lines.append(
            "| {mode} | {count} | {route_acc:.6f} | {prop_r5:.6f} | {prop_iou:.6f} | {sel_iou:.6f} | {best_iou:.6f} |".format(
                mode=mode,
                count=int(row.get("image_count", 0)),
                route_acc=safe_float(row.get("route_acc"), 0.0),
                prop_r5=safe_float(row.get("proposal_recall_at_5_iou_0_5"), 0.0),
                prop_iou=safe_float(row.get("proposal_to_subject_iou_mean"), 0.0),
                sel_iou=safe_float(row.get("selected_to_subject_iou_mean"), 0.0),
                best_iou=safe_float(row.get("best_positive_to_subject_iou_mean"), 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Checklist 라벨 일치도",
            "",
            "| 항목 | acc | applicability_precision | applicability_recall |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, row in sorted((summary.get("checklist") or {}).items()):
        lines.append(
            "| {key} | {acc:.6f} | {prec:.6f} | {rec:.6f} |".format(
                key=key,
                acc=safe_float(row.get("label_accuracy"), 0.0),
                prec=safe_float(row.get("applicability", {}).get("precision"), 0.0),
                rec=safe_float(row.get("applicability", {}).get("recall"), 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## 세부 점수 상관",
            "",
            "| 항목 | mae | pearson | 개수 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, row in sorted((summary.get("detail_scores") or {}).items()):
        lines.append(
            "| {key} | {mae:.6f} | {pearson:.6f} | {count} |".format(
                key=key,
                mae=safe_float(row.get("mae"), 0.0),
                pearson=safe_float(row.get("pearson"), 0.0),
                count=int(row.get("count", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Macro 회귀",
            "",
            "| 항목 | mae | pearson | 개수 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, row in sorted((summary.get("macro") or {}).items()):
        lines.append(
            "| {key} | {mae:.6f} | {pearson:.6f} | {count} |".format(
                key=key,
                mae=safe_float(row.get("mae"), 0.0),
                pearson=safe_float(row.get("pearson"), 0.0),
                count=int(row.get("count", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Why 태그",
            "",
            "| 태그 | precision | recall | f1 | support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in sorted((summary.get("why_tags") or {}).get("per_tag", {}).items()):
        lines.append(
            "| {key} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {support} |".format(
                key=key,
                precision=safe_float(row.get("precision"), 0.0),
                recall=safe_float(row.get("recall"), 0.0),
                f1=safe_float(row.get("f1"), 0.0),
                support=int(row.get("support", 0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _read_jsonl(args.predictions_jsonl)
    eval_map = _load_eval_row_map(args.eval_jsonl)

    route_total = decision_total = 0
    route_ok = decision_ok = 0
    route_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    decision_confusion: dict[str, Counter[str]] = defaultdict(Counter)

    checklist_label_hits: dict[str, int] = Counter()
    checklist_label_total: dict[str, int] = Counter()
    checklist_app_counts: dict[str, dict[str, int]] = {key: {"tp": 0, "fp": 0, "fn": 0} for key in CHECKLIST_CLASS_KEYS}

    detail_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    macro_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    why_tag_tp: Counter[str] = Counter()
    why_tag_fp: Counter[str] = Counter()
    why_tag_fn: Counter[str] = Counter()
    why_micro_tp = why_micro_fp = why_micro_fn = 0

    proposal_metrics: dict[str, list[float]] = defaultdict(list)
    subject_mode_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    failure_rows: list[dict[str, Any]] = []

    for pred in predictions:
        key = _row_key(pred.get("image_id"), pred.get("target_ar"))
        eval_meta = eval_map.get(key, {})
        target_mode = str(eval_meta.get("subject_mode_label") or "")
        pred_mode = str(pred.get("subject_mode_label") or "")
        if target_mode:
            route_total += 1
            route_ok += int(target_mode == pred_mode)
            route_confusion[target_mode][pred_mode] += 1

        target_decision = str(eval_meta.get("decision_label") or "")
        pred_decision = str(pred.get("decision_label") or "")
        if target_decision:
            decision_total += 1
            decision_ok += int(target_decision == pred_decision)
            decision_confusion[target_decision][pred_decision] += 1

        top = pred.get("top_candidate") if isinstance(pred.get("top_candidate"), dict) else {}
        metrics = pred.get("metrics") if isinstance(pred.get("metrics"), dict) else {}

        model_detail = top.get("model_detailed_checklist") if isinstance(top.get("model_detailed_checklist"), dict) else {}
        class_targets = top.get("checklist_class_target") if isinstance(top.get("checklist_class_target"), list) else []
        class_valid = top.get("checklist_class_valid") if isinstance(top.get("checklist_class_valid"), list) else []
        applicability_target = top.get("checklist_applicability_target") if isinstance(top.get("checklist_applicability_target"), list) else []
        applicability_valid = top.get("checklist_applicability_valid") if isinstance(top.get("checklist_applicability_valid"), list) else []
        for idx, key_name in enumerate(CHECKLIST_CLASS_KEYS):
            item = model_detail.get(key_name) if isinstance(model_detail, dict) else None
            if not isinstance(item, dict):
                continue
            model_applicable = bool(item.get("model_applicable", item.get("available", False)))
            if idx < len(applicability_target) and idx < len(applicability_valid) and safe_float(applicability_valid[idx]) > 0:
                target_applicable = safe_float(applicability_target[idx]) > 0.0
                if model_applicable and target_applicable:
                    checklist_app_counts[key_name]["tp"] += 1
                elif model_applicable and not target_applicable:
                    checklist_app_counts[key_name]["fp"] += 1
                elif (not model_applicable) and target_applicable:
                    checklist_app_counts[key_name]["fn"] += 1
            if idx < len(class_targets) and idx < len(class_valid) and safe_float(class_valid[idx]) > 0 and bool(item.get("available", False)):
                teacher_labels = top.get("teacher_checklist_labels", {})
                target_label = None
                if isinstance(teacher_labels, dict):
                    target_label = str(teacher_labels.get(key_name) or "")
                if not target_label:
                    try:
                        target_idx = int(class_targets[idx])
                    except (TypeError, ValueError):
                        target_idx = 0
                    vocab = CHECKLIST_CLASS_VOCABS[key_name]
                    target_idx = max(0, min(max(0, len(vocab) - 1), target_idx))
                    target_label = vocab[target_idx] if vocab else ""
                pred_label = str(item.get("label") or "")
                if target_label:
                    checklist_label_total[key_name] += 1
                    checklist_label_hits[key_name] += int(pred_label == target_label)

        detail_targets = top.get("detail_score_target") if isinstance(top.get("detail_score_target"), list) else []
        detail_valid = top.get("detail_score_valid") if isinstance(top.get("detail_score_valid"), list) else []
        model_detail_scores = top.get("model_detail_scores") if isinstance(top.get("model_detail_scores"), dict) else {}
        for idx, key_name in enumerate(CHECKLIST_SCORE_KEYS):
            if idx >= len(detail_targets) or idx >= len(detail_valid) or safe_float(detail_valid[idx]) <= 0:
                continue
            pred_value = _score_or_none(model_detail_scores.get(key_name))
            target_value = _score_or_none(detail_targets[idx])
            if pred_value is None or target_value is None:
                continue
            detail_pairs[key_name].append((pred_value, target_value))

        macro_target = top.get("macro_target") if isinstance(top.get("macro_target"), list) else []
        model_checklist = top.get("model_checklist") if isinstance(top.get("model_checklist"), dict) else {}
        for idx, key_name in enumerate(("A_macro", "S_macro", "C_macro", "T_macro")):
            if idx >= len(macro_target):
                continue
            head_key = ("aesthetic", "subject", "composition", "technical")[idx]
            pred_value = _score_or_none((model_checklist.get(head_key) or {}).get("score"))
            target_value = _score_or_none(macro_target[idx])
            if pred_value is None or target_value is None:
                continue
            macro_pairs[key_name].append((pred_value, target_value))

        teacher_tags = {str(tag) for tag in (top.get("teacher_why_tags") or [])}
        model_tags = {str(item.get("tag")) for item in (top.get("model_why_tags") or []) if isinstance(item, dict) and item.get("tag")}
        for tag in WHY_TAG_VOCAB:
            in_teacher = tag in teacher_tags
            in_model = tag in model_tags
            if in_teacher and in_model:
                why_tag_tp[tag] += 1
                why_micro_tp += 1
            elif in_model and not in_teacher:
                why_tag_fp[tag] += 1
                why_micro_fp += 1
            elif in_teacher and not in_model:
                why_tag_fn[tag] += 1
                why_micro_fn += 1

        alignment = _proposal_subject_alignment(pred, eval_meta)
        for metric_name in (
            "proposal_to_subject_iou",
            "selected_to_subject_iou",
            "best_positive_to_subject_iou",
            "proposal_subject_hit_iou_0_5",
        ):
            metric_value = alignment.get(metric_name)
            if metric_value is not None:
                proposal_metrics[metric_name].append(float(metric_value))

        proposal_r5 = safe_float(metrics.get("proposal_recall_at_5_iou_0_5"), 0.0)
        subject_mode_bucket = target_mode or pred_mode or "unknown"
        row_summary = {
            "route_acc": 1.0 if target_mode and target_mode == pred_mode else 0.0 if target_mode else math.nan,
            "decision_acc": 1.0 if target_decision and target_decision == pred_decision else 0.0 if target_decision else math.nan,
            "proposal_recall_at_5_iou_0_5": proposal_r5,
            "proposal_to_subject_iou": safe_float(alignment.get("proposal_to_subject_iou"), math.nan),
            "selected_to_subject_iou": safe_float(alignment.get("selected_to_subject_iou"), math.nan),
            "best_positive_to_subject_iou": safe_float(alignment.get("best_positive_to_subject_iou"), math.nan),
        }
        subject_mode_rows[subject_mode_bucket].append(row_summary)

        route_miss = float(bool(target_mode) and target_mode != pred_mode)
        decision_miss = float(bool(target_decision) and target_decision != pred_decision)
        checklist_agreement = safe_float(metrics.get("explain_label_agreement_when_applicable"), 0.0)
        severity = (
            3.0 * route_miss
            + 2.0 * decision_miss
            + (1.0 - safe_float(metrics.get("candidate_top1_hit"), 0.0))
            + (1.0 - proposal_r5)
            + (1.0 - checklist_agreement)
            + (1.0 - safe_float(alignment.get("proposal_subject_hit_iou_0_5"), 0.0))
        )
        failure_rows.append(
            {
                "severity": severity,
                "image_id": pred.get("image_id"),
                "image_path": pred.get("image_path"),
                "target_ar": pred.get("target_ar"),
                "teacher_subject_mode": target_mode,
                "pred_subject_mode": pred_mode,
                "teacher_decision": target_decision,
                "pred_decision": pred_decision,
                "policy_id": eval_meta.get("policy_id"),
                "subject_support_trust_tier": eval_meta.get("subject_support_trust_tier"),
                "subject_reliability": eval_meta.get("subject_reliability"),
                "candidate_top1_hit": safe_float(metrics.get("candidate_top1_hit"), 0.0),
                "proposal_recall_at_5_iou_0_5": proposal_r5,
                "explain_label_agreement_when_applicable": checklist_agreement,
                "top1_iou_to_best_positive": safe_float(metrics.get("top1_iou_to_best_positive"), 0.0),
                "proposal_to_subject_iou": alignment.get("proposal_to_subject_iou"),
                "selected_to_subject_iou": alignment.get("selected_to_subject_iou"),
                "best_positive_to_subject_iou": alignment.get("best_positive_to_subject_iou"),
                "top_candidate_bbox_norm_xyxy": top.get("bbox_norm_xyxy"),
                "proposal_top1_target_ar_bbox_norm_xyxy": (pred.get("proposal_top1_target_ar") or {}).get("bbox_norm_xyxy"),
                "subject_prior_bbox_norm_xyxy": eval_meta.get("subject_prior_bbox_norm_xyxy"),
                "teacher_checklist_labels": top.get("teacher_checklist_labels"),
                "model_detailed_checklist": top.get("model_detailed_checklist"),
                "teacher_why_tags": top.get("teacher_why_tags"),
                "model_why_tags": top.get("model_why_tags"),
            }
        )

    checklist_summary = {}
    for key_name in CHECKLIST_CLASS_KEYS:
        counts = checklist_app_counts[key_name]
        checklist_summary[key_name] = {
            "label_accuracy": checklist_label_hits[key_name] / max(1, checklist_label_total[key_name]),
            "label_correct": checklist_label_hits[key_name],
            "label_total": checklist_label_total[key_name],
            "applicability": _summarize_binary_counts(counts["tp"], counts["fp"], counts["fn"]),
        }

    detail_summary = {}
    for key_name, pairs in detail_pairs.items():
        preds = [p for p, _ in pairs]
        targets = [t for _, t in pairs]
        detail_summary[key_name] = {
            "count": len(pairs),
            "mae": mean(abs(p - t) for p, t in pairs),
            "pearson": _pearson(preds, targets),
        }

    macro_summary = {}
    for key_name, pairs in macro_pairs.items():
        preds = [p for p, _ in pairs]
        targets = [t for _, t in pairs]
        macro_summary[key_name] = {
            "count": len(pairs),
            "mae": mean(abs(p - t) for p, t in pairs),
            "pearson": _pearson(preds, targets),
        }

    why_per_tag = {}
    for tag in WHY_TAG_VOCAB:
        why_per_tag[tag] = {
            "support": why_tag_tp[tag] + why_tag_fn[tag],
            **_summarize_binary_counts(why_tag_tp[tag], why_tag_fp[tag], why_tag_fn[tag]),
        }

    by_subject_mode = {}
    for mode, rows in subject_mode_rows.items():
        route_values = [row["route_acc"] for row in rows if math.isfinite(row["route_acc"])]
        decision_values = [row["decision_acc"] for row in rows if math.isfinite(row["decision_acc"])]
        proposal_values = [row["proposal_recall_at_5_iou_0_5"] for row in rows if math.isfinite(row["proposal_recall_at_5_iou_0_5"])]
        proposal_subject_values = [row["proposal_to_subject_iou"] for row in rows if math.isfinite(row["proposal_to_subject_iou"])]
        selected_subject_values = [row["selected_to_subject_iou"] for row in rows if math.isfinite(row["selected_to_subject_iou"])]
        best_subject_values = [row["best_positive_to_subject_iou"] for row in rows if math.isfinite(row["best_positive_to_subject_iou"])]
        by_subject_mode[mode] = {
            "image_count": len(rows),
            "route_acc": _mean_or_none(route_values),
            "decision_acc": _mean_or_none(decision_values),
            "proposal_recall_at_5_iou_0_5": _mean_or_none(proposal_values),
            "proposal_to_subject_iou_mean": _mean_or_none(proposal_subject_values),
            "selected_to_subject_iou_mean": _mean_or_none(selected_subject_values),
            "best_positive_to_subject_iou_mean": _mean_or_none(best_subject_values),
        }

    summary = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl is not None else None,
        "image_count": len(predictions),
        "route": {
            "accuracy": route_ok / max(1, route_total),
            "total": route_total,
            "correct": route_ok,
            "confusion": {key: dict(value) for key, value in sorted(route_confusion.items())},
        },
        "decision": {
            "accuracy": decision_ok / max(1, decision_total),
            "total": decision_total,
            "correct": decision_ok,
            "confusion": {key: dict(value) for key, value in sorted(decision_confusion.items())},
        },
        "checklist": checklist_summary,
        "detail_scores": detail_summary,
        "macro": macro_summary,
        "why_tags": {
            "micro": _summarize_binary_counts(why_micro_tp, why_micro_fp, why_micro_fn),
            "per_tag": why_per_tag,
        },
        "proposal": {
            "proposal_recall_at_5_iou_0_5": _mean_or_none([safe_float((row.get("metrics") or {}).get("proposal_recall_at_5_iou_0_5"), 0.0) for row in predictions]),
            "proposal_to_subject_iou_mean": _mean_or_none(proposal_metrics.get("proposal_to_subject_iou", [])),
            "selected_to_subject_iou_mean": _mean_or_none(proposal_metrics.get("selected_to_subject_iou", [])),
            "best_positive_to_subject_iou_mean": _mean_or_none(proposal_metrics.get("best_positive_to_subject_iou", [])),
            "proposal_subject_hit_iou_0_5": _mean_or_none(proposal_metrics.get("proposal_subject_hit_iou_0_5", [])),
        },
        "by_subject_mode": by_subject_mode,
        "top_failures": sorted(failure_rows, key=lambda row: float(row.get("severity", 0.0)), reverse=True)[: max(0, int(args.top_k_failures))],
    }

    (args.output_dir / "head_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    with (args.output_dir / "head_analysis_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in summary["top_failures"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_report(args.output_dir / "HEAD_ANALYSIS_REPORT.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
