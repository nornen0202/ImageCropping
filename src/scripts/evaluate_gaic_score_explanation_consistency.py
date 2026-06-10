#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from mobilecropnet_v4.gaic_benchmark import per_image_gaic_metrics, summarize_metric_rows


DEFAULT_SCORE_FIELDS = (
    "crop_utility_raw",
    "crop_utility_prob",
    "score_policy",
    "score_policy_sigmoid_z_local",
    "score_policy_z_local",
    "score_rank_pct",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def sigmoid(value: float) -> float:
    x = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-x))


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_rows(path: Optional[Path], *, protocol: str) -> List[Dict[str, Any]]:
    if path is None:
        return []
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("protocol", "")) == str(protocol):
                rows.append(row)
    return rows


def row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("protocol", "")),
        str(row.get("image_id", "")),
        str(row.get("candidate_id", "")),
        str(row.get("gt_annotation_id", "")),
    )


def load_prediction_scores(specs: Sequence[str]) -> Dict[str, Dict[Tuple[str, str, str, str], float]]:
    out: Dict[str, Dict[Tuple[str, str, str, str], float]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"prediction spec must be NAME=PATH, got {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        scores = out.setdefault(name, {})
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scores[row_key(row)] = safe_float(row.get("score"), 0.0)
    return out


def quality_label(score: float, *, good: float = 0.68, ok: float = 0.45) -> str:
    value = float(score)
    if value >= float(good):
        return "good"
    if value >= float(ok):
        return "ok"
    return "bad"


def penalty_label(value: float, *, warn: float, fail: float) -> str:
    v = float(value)
    if v >= float(fail):
        return "fail"
    if v >= float(warn):
        return "warn"
    return "pass"


def bbox_area(row: Dict[str, Any]) -> float:
    area = row.get("profile_bbox_area")
    if area is not None:
        return clamp(safe_float(area, 1.0), 0.0, 1.0)
    box = row.get("bbox_norm_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 1.0
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box[:4]]
    return clamp(max(0.0, x2 - x1) * max(0.0, y2 - y1), 0.0, 1.0)


def area_label(area: float) -> str:
    if area < 0.12:
        return "too_tight"
    if area > 0.90:
        return "near_full"
    if 0.35 <= area <= 0.78:
        return "balanced"
    return "acceptable"


def build_explanation_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    a_macro = clamp(safe_float(row.get("profile_A_macro"), 0.0), 0.0, 1.0)
    s_macro = clamp(safe_float(row.get("profile_S_macro"), 0.0), 0.0, 1.0)
    c_macro = clamp(safe_float(row.get("profile_C_macro"), 0.0), 0.0, 1.0)
    t_macro = clamp(safe_float(row.get("profile_T_macro"), 0.0), 0.0, 1.0)
    active_terms = []
    for key, value in (
        ("A", a_macro),
        ("S", s_macro),
        ("C", c_macro),
        ("T", t_macro),
    ):
        if safe_int(row.get(f"profile_{key}_active"), 1 if key != "T" else 0) > 0:
            active_terms.append(value)
    macro_mean = sum(active_terms) / len(active_terms) if active_terms else (a_macro + s_macro + c_macro) / 3.0

    safety_total = max(0.0, safe_float(row.get("profile_safety_penalty_total"), 0.0))
    safety_hard = max(0.0, safe_float(row.get("profile_safety_penalty_hard"), 0.0))
    safety_soft = max(0.0, safe_float(row.get("profile_safety_penalty_soft"), 0.0))
    p_cut = max(0.0, safe_float(row.get("profile_component_p_cut"), 0.0))
    p_text = max(0.0, safe_float(row.get("profile_component_p_text"), 0.0))
    p_ar = max(0.0, safe_float(row.get("profile_component_p_ar_free"), 0.0))
    r_comp = safe_float(row.get("profile_component_r_comp"), 0.0)
    r_headroom = safe_float(row.get("profile_component_r_headroom"), 0.0)
    r_lookroom = safe_float(row.get("profile_component_r_lookroom"), 0.0)
    r_horizon = safe_float(row.get("profile_component_r_horizon"), 0.0)
    r_sym = safe_float(row.get("profile_component_r_sym"), 0.0)
    r_context = safe_float(row.get("profile_component_r_context"), 0.0)
    aesthetic = clamp(safe_float(row.get("profile_component_aesthetic_norm"), a_macro), 0.0, 1.0)
    align = clamp(0.5 + 0.5 * safe_float(row.get("profile_component_cosine_img_text"), 0.0), 0.0, 1.0)
    area = bbox_area(row)
    log_ar_abs = abs(math.log(max(1e-6, safe_float(row.get("profile_bbox_ar"), 1.0))))

    safety_norm = clamp(safety_total / 10.5, 0.0, 1.0)
    safety_hard_norm = clamp(safety_hard / 2.55, 0.0, 1.0)
    safety_soft_norm = clamp(safety_soft / 7.95, 0.0, 1.0)
    p_cut_norm = clamp(p_cut / 5.2, 0.0, 1.0)
    p_text_norm = clamp(p_text / 1.0, 0.0, 1.0)
    p_ar_norm = clamp(p_ar / 0.20, 0.0, 1.0)

    reward = (
        0.12 * clamp(r_comp, 0.0, 1.0)
        + 0.08 * clamp(r_headroom, 0.0, 1.0)
        + 0.08 * clamp(r_lookroom, 0.0, 1.0)
        + 0.06 * clamp(r_horizon, 0.0, 1.0)
        + 0.05 * clamp(r_sym, 0.0, 1.0)
        + 0.08 * clamp(r_context, -1.0, 1.0)
        + 0.04 * aesthetic
        + 0.03 * align
    )
    penalty = (
        0.60 * safety_norm
        + 0.45 * safety_hard_norm
        + 0.16 * safety_soft_norm
        + 0.55 * p_cut_norm
        + 0.18 * p_text_norm
        + 0.18 * p_ar_norm
        + 0.06 * min(log_ar_abs, 2.0)
    )
    area_bonus = 0.08 * math.exp(-0.5 * ((math.log(max(area, 1e-6)) - math.log(0.62)) / 0.48) ** 2)
    raw = 0.64 * macro_mean + reward + area_bonus - penalty
    explanation_score = clamp(sigmoid(4.0 * (raw - 0.48)), 0.0, 1.0)

    labels = {
        "aesthetic": quality_label(a_macro),
        "subject": quality_label(s_macro),
        "composition": quality_label(c_macro),
        "technical": quality_label(t_macro) if safe_int(row.get("profile_T_active"), 0) > 0 else "not_applicable",
        "safety": penalty_label(safety_total, warn=2.5, fail=6.0),
        "hard_safety": penalty_label(safety_hard, warn=1.0, fail=2.0),
        "subject_cut": penalty_label(p_cut, warn=1.2, fail=2.5),
        "text_cut": penalty_label(p_text, warn=0.18, fail=0.45),
        "freeform_ar": penalty_label(p_ar, warn=0.10, fail=0.18),
        "area": area_label(area),
        "headroom": quality_label(clamp(r_headroom, 0.0, 1.0), good=0.58, ok=0.32),
        "lookroom": quality_label(clamp(r_lookroom, 0.0, 1.0), good=0.58, ok=0.32),
        "horizon": quality_label(clamp(r_horizon, 0.0, 1.0), good=0.58, ok=0.32),
        "symmetry": quality_label(clamp(r_sym, 0.0, 1.0), good=0.58, ok=0.32),
        "context": quality_label(clamp(0.5 + 0.5 * r_context, 0.0, 1.0), good=0.58, ok=0.32),
    }
    warnings: List[str] = []
    if labels["hard_safety"] == "fail" or labels["safety"] == "fail":
        warnings.append("safety_penalty_fail")
    if labels["subject_cut"] == "fail":
        warnings.append("subject_cut_fail")
    if labels["text_cut"] == "fail":
        warnings.append("text_cut_fail")
    if labels["freeform_ar"] == "fail":
        warnings.append("freeform_ar_fail")
    if labels["area"] in {"too_tight", "near_full"}:
        warnings.append(f"area_{labels['area']}")
    if explanation_score < 0.35:
        warnings.append("low_explanation_score")

    fatal = any(
        label == "fail"
        for label in (labels["hard_safety"], labels["safety"], labels["subject_cut"], labels["text_cut"], labels["freeform_ar"])
    )
    pass_labels = sum(1 for label in labels.values() if label in {"good", "ok", "pass", "balanced", "acceptable", "not_applicable"})
    explanation_confidence = clamp(0.35 + 0.45 * explanation_score + 0.20 * (pass_labels / max(1, len(labels))) - (0.25 if fatal else 0.0))
    return {
        "explanation_score": float(explanation_score),
        "explanation_confidence": float(explanation_confidence),
        "checklist_labels": labels,
        "warnings": warnings,
        "fatal_flag": bool(fatal),
        "macro_mean": float(macro_mean),
        "penalty_sum": float(penalty),
        "reward_sum": float(reward),
    }


def grouped_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("image_id", ""))].append(row)
    for image_rows in grouped.values():
        image_rows.sort(key=lambda r: (safe_int(r.get("gt_annotation_id"), 0), str(r.get("candidate_id", ""))))
    return dict(sorted(grouped.items()))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ax = sum(a) / len(a)
    bx = sum(b) / len(b)
    num = sum((x - ax) * (y - bx) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ax) ** 2 for x in a))
    db = math.sqrt(sum((y - bx) ** 2 for y in b))
    return float(num / (da * db)) if da > 0.0 and db > 0.0 else 0.0


def ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), idx))
    out = [0.0] * len(values)
    n = len(values)
    start = 0
    while start < n:
        end = start + 1
        while end < n and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        avg = (float(start) + float(end - 1)) * 0.5
        for pos in range(start, end):
            out[order[pos]] = avg
        start = end
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    return pearson(ranks(a), ranks(b))


def rank_pct_by_image(rows: Sequence[Dict[str, Any]], scores: Sequence[float]) -> List[float]:
    out = [0.0] * len(rows)
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for idx, (row, score) in enumerate(zip(rows, scores)):
        grouped[str(row.get("image_id", ""))].append((idx, float(score)))
    for pairs in grouped.values():
        n = len(pairs)
        if n <= 1:
            for idx, _ in pairs:
                out[idx] = 0.5
            continue
        sorted_pairs = sorted(pairs, key=lambda item: (item[1], item[0]))
        denom = max(1.0, float(n - 1))
        for rank, (idx, _) in enumerate(sorted_pairs):
            out[idx] = float(rank) / denom
    return out


def method_scores(
    rows: Sequence[Dict[str, Any]],
    *,
    field: Optional[str] = None,
    prediction_scores: Optional[Dict[Tuple[str, str, str, str], float]] = None,
) -> List[float]:
    if field is not None:
        return [safe_float(row.get(field), 0.0) for row in rows]
    if prediction_scores is not None:
        return [safe_float(prediction_scores.get(row_key(row), 0.0), 0.0) for row in rows]
    raise ValueError("field or prediction_scores is required")


def adjusted_scores(
    rows: Sequence[Dict[str, Any]],
    base_scores: Sequence[float],
    explanations: Sequence[Dict[str, Any]],
    *,
    lambda_expl: float,
    fatal_penalty: float,
    low_expl_penalty: float,
    hard_gate: bool,
) -> List[float]:
    base_rank = rank_pct_by_image(rows, base_scores)
    out: List[float] = []
    for base, expl in zip(base_rank, explanations):
        fatal = bool(expl.get("fatal_flag", False))
        expl_score = safe_float(expl.get("explanation_score"), 0.0)
        if hard_gate and fatal:
            out.append(-1.0)
            continue
        score = (1.0 - float(lambda_expl)) * float(base) + float(lambda_expl) * expl_score
        if fatal:
            score -= float(fatal_penalty)
        if expl_score < 0.35:
            score -= float(low_expl_penalty)
        out.append(float(score))
    return out


def score_objective(metrics: Dict[str, float]) -> float:
    return (
        safe_float(metrics.get("srcc"), 0.0)
        + safe_float(metrics.get("pcc"), 0.0)
        + safe_float(metrics.get("acc1_of_top10"), 0.0)
        + safe_float(metrics.get("acc4_of_top10"), 0.0)
        + 0.10 * safe_float(metrics.get("top1_mos"), 0.0)
        - 0.10 * safe_float(metrics.get("top1_mos_regret"), 0.0)
    )


def evaluate_method(
    rows: Sequence[Dict[str, Any]],
    scores: Sequence[float],
    explanations: Sequence[Dict[str, Any]],
    *,
    method: str,
    family: str,
    base_method: str = "",
) -> Dict[str, Any]:
    grouped = grouped_rows(rows)
    row_by_key = {row_key(row): idx for idx, row in enumerate(rows)}
    gaic_metric_rows: List[Dict[str, float]] = []
    align_srcc: List[float] = []
    align_pcc: List[float] = []
    top1_expl_scores: List[float] = []
    top1_conf: List[float] = []
    top1_pass: List[float] = []
    top1_fatal: List[float] = []
    top1_contra: List[float] = []
    high_score_contra_rates: List[float] = []
    mos_top_by_expl: List[float] = []
    for image_rows in grouped.values():
        indices = [row_by_key[row_key(row)] for row in image_rows]
        mos = [safe_float(rows[idx].get("mos"), 0.0) for idx in indices]
        image_scores = [float(scores[idx]) for idx in indices]
        expl_scores = [safe_float(explanations[idx].get("explanation_score"), 0.0) for idx in indices]
        metrics = per_image_gaic_metrics(mos, image_scores)
        if metrics:
            gaic_metric_rows.append(metrics)
        align_srcc.append(spearman(image_scores, expl_scores))
        align_pcc.append(pearson(image_scores, expl_scores))
        best_idx = max(range(len(indices)), key=lambda local_idx: (image_scores[local_idx], -local_idx))
        expl = explanations[indices[best_idx]]
        expl_score = safe_float(expl.get("explanation_score"), 0.0)
        fatal = bool(expl.get("fatal_flag", False))
        confidence = safe_float(expl.get("explanation_confidence"), 0.0)
        top1_expl_scores.append(expl_score)
        top1_conf.append(confidence)
        top1_fatal.append(1.0 if fatal else 0.0)
        top1_pass.append(1.0 if expl_score >= 0.45 and not fatal else 0.0)
        top1_contra.append(1.0 if fatal or expl_score < 0.35 else 0.0)
        cutoff_count = max(1, int(math.ceil(0.10 * len(indices))))
        top_score_indices = sorted(range(len(indices)), key=lambda local_idx: image_scores[local_idx], reverse=True)[:cutoff_count]
        contra = 0
        for local_idx in top_score_indices:
            item = explanations[indices[local_idx]]
            if bool(item.get("fatal_flag", False)) or safe_float(item.get("explanation_score"), 0.0) < 0.35:
                contra += 1
        high_score_contra_rates.append(float(contra) / float(max(1, len(top_score_indices))))
        expl_best_idx = max(range(len(indices)), key=lambda local_idx: (expl_scores[local_idx], -local_idx))
        mos_top_by_expl.append(float(mos[expl_best_idx]))

    score_metrics = summarize_metric_rows(gaic_metric_rows)
    consistency = {
        "score_explanation_srcc": float(sum(align_srcc) / len(align_srcc)) if align_srcc else 0.0,
        "score_explanation_pcc": float(sum(align_pcc) / len(align_pcc)) if align_pcc else 0.0,
        "top1_explanation_score": float(sum(top1_expl_scores) / len(top1_expl_scores)) if top1_expl_scores else 0.0,
        "top1_explanation_confidence": float(sum(top1_conf) / len(top1_conf)) if top1_conf else 0.0,
        "top1_checklist_pass_rate": float(sum(top1_pass) / len(top1_pass)) if top1_pass else 0.0,
        "top1_fatal_rate": float(sum(top1_fatal) / len(top1_fatal)) if top1_fatal else 0.0,
        "top1_contradiction_rate": float(sum(top1_contra) / len(top1_contra)) if top1_contra else 0.0,
        "high_score_contradiction_rate": float(sum(high_score_contra_rates) / len(high_score_contra_rates)) if high_score_contra_rates else 0.0,
        "explanation_top1_mos": float(sum(mos_top_by_expl) / len(mos_top_by_expl)) if mos_top_by_expl else 0.0,
    }
    joint_objective = (
        score_objective(score_metrics)
        + 0.20 * consistency["top1_checklist_pass_rate"]
        + 0.10 * consistency["top1_explanation_confidence"]
        - 0.45 * consistency["top1_contradiction_rate"]
        - 0.35 * consistency["top1_fatal_rate"]
    )
    return {
        "method": method,
        "family": family,
        "base_method": base_method,
        "image_count": len(gaic_metric_rows),
        "candidate_count": len(rows),
        "metrics": score_metrics,
        "consistency": consistency,
        "score_objective": round(score_objective(score_metrics), 9),
        "joint_objective": round(joint_objective, 9),
    }


def build_method_specs(score_fields: Sequence[str], prediction_scores: Dict[str, Dict[Tuple[str, str, str, str], float]]) -> List[Dict[str, Any]]:
    specs = [{"method": field, "family": "score_field", "field": field} for field in score_fields]
    specs.extend({"method": name, "family": "prediction", "prediction_name": name} for name in sorted(prediction_scores))
    return specs


def build_adapter_grid(base_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    profiles = [
        ("expl_blend10", 0.10, 0.00, 0.00, False),
        ("expl_blend25", 0.25, 0.00, 0.00, False),
        ("expl_blend40", 0.40, 0.00, 0.00, False),
        ("soft_gate25", 0.25, 0.25, 0.12, False),
        ("soft_gate40", 0.40, 0.30, 0.15, False),
        ("hard_gate10", 0.10, 0.00, 0.00, True),
        ("hard_gate25", 0.25, 0.00, 0.00, True),
    ]
    out: List[Dict[str, Any]] = []
    for base in base_specs:
        base_method = str(base["method"])
        for name, lam, fatal_penalty, low_penalty, hard_gate in profiles:
            out.append(
                {
                    "method": f"{base_method}__{name}",
                    "family": "explanation_adapter",
                    "base_method": base_method,
                    "lambda_expl": float(lam),
                    "fatal_penalty": float(fatal_penalty),
                    "low_expl_penalty": float(low_penalty),
                    "hard_gate": bool(hard_gate),
                }
            )
    return out


def split_scores_for_base(
    rows: Sequence[Dict[str, Any]],
    spec: Dict[str, Any],
    prediction_scores: Dict[str, Dict[Tuple[str, str, str, str], float]],
) -> List[float]:
    if "field" in spec:
        return method_scores(rows, field=str(spec["field"]))
    return method_scores(rows, prediction_scores=prediction_scores[str(spec["prediction_name"])])


def evaluate_split(
    rows: Sequence[Dict[str, Any]],
    *,
    method_specs: Sequence[Dict[str, Any]],
    adapter_specs: Sequence[Dict[str, Any]],
    prediction_scores: Dict[str, Dict[Tuple[str, str, str, str], float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[Dict[str, Any]]]:
    explanations = [build_explanation_payload(row) for row in rows]
    results: List[Dict[str, Any]] = []
    scores_by_method: Dict[str, List[float]] = {}
    for spec in method_specs:
        scores = split_scores_for_base(rows, spec, prediction_scores)
        scores_by_method[str(spec["method"])] = scores
        results.append(
            evaluate_method(
                rows,
                scores,
                explanations,
                method=str(spec["method"]),
                family=str(spec["family"]),
                base_method=str(spec.get("base_method", "")),
            )
        )
    for spec in adapter_specs:
        base_method = str(spec["base_method"])
        if base_method not in scores_by_method:
            continue
        scores = adjusted_scores(
            rows,
            scores_by_method[base_method],
            explanations,
            lambda_expl=safe_float(spec.get("lambda_expl"), 0.0),
            fatal_penalty=safe_float(spec.get("fatal_penalty"), 0.0),
            low_expl_penalty=safe_float(spec.get("low_expl_penalty"), 0.0),
            hard_gate=bool(spec.get("hard_gate", False)),
        )
        scores_by_method[str(spec["method"])] = scores
        results.append(
            evaluate_method(
                rows,
                scores,
                explanations,
                method=str(spec["method"]),
                family=str(spec["family"]),
                base_method=base_method,
            )
        )
    results.sort(key=lambda item: safe_float(item.get("joint_objective"), 0.0), reverse=True)
    return results, scores_by_method, explanations


def result_table(title: str, rows: Sequence[Dict[str, Any]], *, limit: int) -> List[str]:
    lines = [
        f"### {title}",
        "",
        "| method | family | base | joint | PCC | SRCC | Acc1/10 | top1 MOS | regret | expl SRCC | pass | contradiction | fatal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        metrics = row.get("metrics", {})
        consistency = row.get("consistency", {})
        lines.append(
            "| {method} | {family} | {base} | {joint:.6f} | {pcc:.6f} | {srcc:.6f} | {acc:.6f} | {top:.6f} | {reg:.6f} | {es:.6f} | {pass_rate:.6f} | {contra:.6f} | {fatal:.6f} |".format(
                method=row.get("method", ""),
                family=row.get("family", ""),
                base=row.get("base_method", ""),
                joint=safe_float(row.get("joint_objective"), 0.0),
                pcc=safe_float(metrics.get("pcc"), 0.0),
                srcc=safe_float(metrics.get("srcc"), 0.0),
                acc=safe_float(metrics.get("acc1_of_top10"), 0.0),
                top=safe_float(metrics.get("top1_mos"), 0.0),
                reg=safe_float(metrics.get("top1_mos_regret"), 0.0),
                es=safe_float(consistency.get("score_explanation_srcc"), 0.0),
                pass_rate=safe_float(consistency.get("top1_checklist_pass_rate"), 0.0),
                contra=safe_float(consistency.get("top1_contradiction_rate"), 0.0),
                fatal=safe_float(consistency.get("top1_fatal_rate"), 0.0),
            )
        )
    lines.append("")
    return lines


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def label_rows_for_method(
    rows: Sequence[Dict[str, Any]],
    scores: Sequence[float],
    explanations: Sequence[Dict[str, Any]],
    *,
    method: str,
) -> Iterable[Dict[str, Any]]:
    for row, score, expl in zip(rows, scores, explanations):
        yield {
            "image_id": str(row.get("image_id", "")),
            "candidate_id": str(row.get("candidate_id", "")),
            "gt_annotation_id": row.get("gt_annotation_id"),
            "protocol": row.get("protocol"),
            "mos": safe_float(row.get("mos"), 0.0),
            "score": float(score),
            "method": method,
            "bbox_norm_xyxy": row.get("bbox_norm_xyxy"),
            "explanation_score": safe_float(expl.get("explanation_score"), 0.0),
            "explanation_confidence": safe_float(expl.get("explanation_confidence"), 0.0),
            "fatal_flag": bool(expl.get("fatal_flag", False)),
            "checklist_labels": expl.get("checklist_labels", {}),
            "warnings": expl.get("warnings", []),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAIC score/explanation consistency and explanation-aware score adapters.")
    parser.add_argument("--train_candidate_eval_jsonl", type=Path, default=None)
    parser.add_argument("--val_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--test_candidate_eval_jsonl", type=Path, default=None)
    parser.add_argument("--protocol", default="Gc", choices=["Gc", "Ge"])
    parser.add_argument("--score_fields", default=",".join(DEFAULT_SCORE_FIELDS))
    parser.add_argument("--prediction_jsonl", action="append", default=[], help="Optional NAME=PATH prediction JSONL.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--eval_top_n_train_test", type=int, default=20)
    parser.add_argument("--write_label_jsonl", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    score_fields = parse_csv(args.score_fields)
    prediction_scores = load_prediction_scores(args.prediction_jsonl)
    split_rows = {
        "train": read_rows(args.train_candidate_eval_jsonl, protocol=str(args.protocol)) if args.train_candidate_eval_jsonl else [],
        "val": read_rows(args.val_candidate_eval_jsonl, protocol=str(args.protocol)),
        "test": read_rows(args.test_candidate_eval_jsonl, protocol=str(args.protocol)) if args.test_candidate_eval_jsonl else [],
    }
    if not split_rows["val"]:
        raise RuntimeError("val split has no rows for requested protocol")

    method_specs = build_method_specs(score_fields, prediction_scores)
    adapter_specs = build_adapter_grid(method_specs)

    split_results: Dict[str, List[Dict[str, Any]]] = {}
    split_scores: Dict[str, Dict[str, List[float]]] = {}
    split_explanations: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, rows in split_rows.items():
        if not rows:
            split_results[split_name] = []
            split_scores[split_name] = {}
            split_explanations[split_name] = []
            continue
        results, scores_by_method, explanations = evaluate_split(
            rows,
            method_specs=method_specs,
            adapter_specs=adapter_specs,
            prediction_scores=prediction_scores,
        )
        split_results[split_name] = results
        split_scores[split_name] = scores_by_method
        split_explanations[split_name] = explanations

    val_ranked = split_results["val"]
    selected_methods = [str(item["method"]) for item in val_ranked[: max(1, int(args.eval_top_n_train_test))]]
    selected_by_val = str(val_ranked[0]["method"])
    selected_summary: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        selected_summary[split_name] = [item for item in split_results.get(split_name, []) if str(item.get("method")) in selected_methods]
        selected_summary[split_name].sort(key=lambda item: selected_methods.index(str(item.get("method"))) if str(item.get("method")) in selected_methods else 999)

    payload = {
        "protocol": str(args.protocol),
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "score_fields": score_fields,
        "prediction_methods": sorted(prediction_scores.keys()),
        "selected_method_by_val_joint_objective": selected_by_val,
        "val_results": val_ranked,
        "selected_train_results": selected_summary["train"],
        "selected_val_results": selected_summary["val"],
        "selected_test_results": selected_summary["test"],
    }
    write_json(output_dir / "score_explanation_consistency_summary.json", payload)

    lines = [
        "# GAIC Score Explanation Consistency Report",
        "",
        f"- protocol: `{args.protocol}`",
        f"- rows: train={len(split_rows['train'])}, val={len(split_rows['val'])}, test={len(split_rows['test'])}",
        f"- selected_method_by_val_joint_objective: `{selected_by_val}`",
        "",
    ]
    lines.extend(result_table("Val Methods Ranked By Joint Objective", val_ranked, limit=int(args.top_k)))
    if selected_summary["train"]:
        lines.extend(result_table("Val-Selected Methods On Train", selected_summary["train"], limit=int(args.top_k)))
    if selected_summary["test"]:
        lines.extend(result_table("Val-Selected Methods On Test", selected_summary["test"], limit=int(args.top_k)))
    (output_dir / "SCORE_EXPLANATION_CONSISTENCY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    if int(args.write_label_jsonl):
        for split_name in ("val", "test"):
            rows = split_rows.get(split_name, [])
            if not rows:
                continue
            method = selected_by_val
            scores = split_scores[split_name].get(method)
            if scores is None:
                continue
            write_jsonl(
                output_dir / f"explanation_labels_{split_name}_{method}.jsonl",
                label_rows_for_method(rows, scores, split_explanations[split_name], method=method),
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "protocol": str(args.protocol),
                "selected_method_by_val_joint_objective": selected_by_val,
                "row_counts": {split: len(rows) for split, rows in split_rows.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
