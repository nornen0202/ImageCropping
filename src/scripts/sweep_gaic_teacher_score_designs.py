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


BASELINE_FIELDS = (
    "crop_utility_raw",
    "score_policy",
    "score_rank",
    "crop_utility_prob",
    "score_policy_sigmoid_z_local",
    "score_policy_z_local",
    "score_rank_pct",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def has_profile_fields(rows: Sequence[Dict[str, Any]]) -> bool:
    required = {
        "profile_A_macro",
        "profile_S_macro",
        "profile_C_macro",
        "profile_area_log_prior",
        "profile_safety_penalty_total",
    }
    return bool(rows) and required.issubset(rows[0].keys())


def grouped_metric_summary(rows: Sequence[Dict[str, Any]], scores: Sequence[float], *, method: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[str(row.get("image_id", ""))].append((row, float(score)))

    metric_rows: List[Dict[str, float]] = []
    for image_id in sorted(grouped):
        pairs = sorted(
            grouped[image_id],
            key=lambda pair: (safe_int(pair[0].get("gt_annotation_id"), 0), str(pair[0].get("candidate_id", ""))),
        )
        mos = [safe_float(row.get("mos"), 0.0) for row, _ in pairs]
        image_scores = [score for _, score in pairs]
        metrics = per_image_gaic_metrics(mos, image_scores)
        if metrics:
            metric_rows.append(metrics)
    return {
        "method": method,
        "image_count": len(metric_rows),
        "candidate_count": len(rows),
        "metrics": summarize_metric_rows(metric_rows),
    }


def objective(metrics: Dict[str, float]) -> float:
    return (
        safe_float(metrics.get("srcc"), 0.0)
        + safe_float(metrics.get("pcc"), 0.0)
        + safe_float(metrics.get("acc1_of_top10"), 0.0)
        + safe_float(metrics.get("acc4_of_top10"), 0.0)
        + 0.10 * safe_float(metrics.get("top1_mos"), 0.0)
        - 0.10 * safe_float(metrics.get("top1_mos_regret"), 0.0)
    )


def evaluate_field(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    result = grouped_metric_summary(rows, [safe_float(row.get(field), 0.0) for row in rows], method=field)
    result["family"] = "score_field"
    result["objective"] = round(objective(result["metrics"]), 9)
    return result


def macro_value(row: Dict[str, Any], name: str, weight: float) -> Tuple[float, float]:
    active = safe_int(row.get(f"profile_{name}_active", 1), 1)
    if active <= 0 or weight <= 0.0:
        return 0.0, 0.0
    return weight * safe_float(row.get(f"profile_{name}_macro"), 0.0), weight


def component_design_score(row: Dict[str, Any], spec: Dict[str, Any]) -> float:
    num = 0.0
    den = 0.0
    for name, key in (("A", "w_a"), ("S", "w_s"), ("C", "w_c"), ("T", "w_t")):
        value, weight = macro_value(row, name, safe_float(spec.get(key), 0.0))
        num += value
        den += weight
    macro = 0.0 if den <= 0.0 else num / den
    area = safe_float(row.get("profile_area_log_prior"), 0.0)
    safety = safe_float(row.get("profile_safety_penalty_total"), 0.0)
    soft = safe_float(row.get("profile_safety_penalty_soft"), 0.0)
    hard = safe_float(row.get("profile_safety_penalty_hard"), 0.0)
    safety_mix = safe_float(spec.get("safety_total_scale"), 1.0) * safety
    safety_mix += safe_float(spec.get("safety_soft_extra"), 0.0) * soft
    safety_mix += safe_float(spec.get("safety_hard_extra"), 0.0) * hard
    return (
        macro
        + safe_float(spec.get("area_scale"), 0.0) * area
        - safety_mix
        + safe_float(spec.get("aesthetic_extra"), 0.0) * safe_float(row.get("profile_component_aesthetic_norm"), 0.0)
        + safe_float(spec.get("align_extra"), 0.0) * (0.5 + 0.5 * safe_float(row.get("profile_component_cosine_img_text"), 0.0))
        - safe_float(spec.get("cut_penalty", spec.get("cut_extra", 0.0)), 0.0) * safe_float(row.get("profile_component_p_cut"), 0.0)
        - safe_float(spec.get("text_penalty", spec.get("text_extra", 0.0)), 0.0) * safe_float(row.get("profile_component_p_text"), 0.0)
        - safe_float(spec.get("free_ar_penalty", spec.get("free_ar_extra", 0.0)), 0.0) * safe_float(row.get("profile_component_p_ar_free"), 0.0)
        + safe_float(spec.get("comp_extra"), 0.0) * safe_float(row.get("profile_component_r_comp"), 0.0)
        + safe_float(spec.get("headroom_extra"), 0.0) * safe_float(row.get("profile_component_r_headroom"), 0.0)
        + safe_float(spec.get("lookroom_extra"), 0.0) * safe_float(row.get("profile_component_r_lookroom"), 0.0)
        + safe_float(spec.get("horizon_extra"), 0.0) * safe_float(row.get("profile_component_r_horizon"), 0.0)
        + safe_float(spec.get("sym_extra"), 0.0) * safe_float(row.get("profile_component_r_sym"), 0.0)
        + safe_float(spec.get("context_extra"), 0.0) * safe_float(row.get("profile_component_r_context"), 0.0)
        + mid_area_bonus(row, spec)
        - safe_float(spec.get("ar_extreme_penalty"), 0.0) * abs(math.log(max(1e-6, safe_float(row.get("profile_bbox_ar"), 1.0))))
    )


def mode_key(row: Dict[str, Any]) -> str:
    for key in ("mode_bucket", "subject_mode"):
        value = str(row.get(key, "") or "").strip().lower()
        if value:
            return value
    return "default"


def field_blend_score(row: Dict[str, Any], spec: Dict[str, Any]) -> float:
    weights = spec.get("field_weights", {})
    if not isinstance(weights, dict) or not weights:
        return 0.0
    num = 0.0
    den = 0.0
    for field, weight_value in weights.items():
        weight = safe_float(weight_value, 0.0)
        if weight == 0.0:
            continue
        num += weight * safe_float(row.get(str(field)), 0.0)
        den += abs(weight)
    if safe_float(spec.get("normalize_field_weights"), 1.0) > 0.0 and den > 0.0:
        return num / den
    return num


def raw_score_for_row(row: Dict[str, Any], spec: Dict[str, Any]) -> float:
    formula = str(spec.get("formula", "component"))
    if formula == "field_blend":
        return field_blend_score(row, spec)
    if formula == "mode_aware":
        mode_specs = spec.get("mode_specs", {})
        if isinstance(mode_specs, dict):
            key = mode_key(row)
            sub_spec = mode_specs.get(key) or mode_specs.get(key.split(":", 1)[0]) or mode_specs.get("default")
            if isinstance(sub_spec, dict):
                merged = dict(spec)
                merged.update(sub_spec)
                merged.pop("mode_specs", None)
                merged["formula"] = "component"
                return raw_score_for_row(row, merged)
        return component_design_score(row, spec)
    if formula == "hybrid":
        component = component_design_score(row, spec)
        field = field_blend_score(row, spec)
        return safe_float(spec.get("component_scale"), 1.0) * component + safe_float(spec.get("field_scale"), 1.0) * field
    return component_design_score(row, spec)


def local_transform_scores(rows: Sequence[Dict[str, Any]], raw_scores: Sequence[float], transform: str) -> List[float]:
    transform = str(transform or "none")
    if transform in ("", "none", "raw"):
        return [float(v) for v in raw_scores]

    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for idx, (row, score) in enumerate(zip(rows, raw_scores)):
        grouped[str(row.get("image_id", ""))].append((idx, float(score)))
    out = [0.0 for _ in raw_scores]
    for pairs in grouped.values():
        values = [score for _, score in pairs]
        n = len(values)
        if n <= 1:
            for idx, _ in pairs:
                out[idx] = 0.5 if transform == "rank_pct" else 0.0
            continue
        mean = sum(values) / float(n)
        var = sum((v - mean) ** 2 for v in values) / float(n)
        std = math.sqrt(max(var, 1e-12))
        if transform in ("z", "sigmoid_z"):
            for idx, score in pairs:
                z = (score - mean) / std
                out[idx] = 1.0 / (1.0 + math.exp(-z)) if transform == "sigmoid_z" else z
        elif transform == "rank_pct":
            sorted_pairs = sorted(pairs, key=lambda item: (item[1], item[0]))
            denom = max(1.0, float(n - 1))
            for rank, (idx, _) in enumerate(sorted_pairs):
                out[idx] = float(rank) / denom
        else:
            raise ValueError(f"unsupported post_transform: {transform}")
    return out


def scores_for_spec(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> List[float]:
    raw_scores = [raw_score_for_row(row, spec) for row in rows]
    return local_transform_scores(rows, raw_scores, str(spec.get("post_transform", "none")))


def mid_area_bonus(row: Dict[str, Any], spec: Dict[str, float]) -> float:
    scale = safe_float(spec.get("mid_area_bonus"), 0.0)
    if scale == 0.0:
        return 0.0
    log_area = safe_float(row.get("profile_bbox_log_area"), math.log(0.65))
    target = safe_float(spec.get("mid_area_log_target"), math.log(0.65))
    width = max(1e-6, safe_float(spec.get("mid_area_log_width"), 0.45))
    return scale * math.exp(-0.5 * ((log_area - target) / width) ** 2)


def default_designs() -> List[Dict[str, Any]]:
    seeds = [
        ("profile_default_replay", 1.0, 1.0, 1.0, 0.10, 1.00, 1.00, 0.0, 0.0, 0.0),
        ("gaic_no_area_soft_safety", 1.0, 1.0, 1.0, 0.00, 0.00, 0.55, 0.0, 0.0, 0.0),
        ("gaic_aesthetic_composition", 1.45, 0.45, 1.25, 0.00, 0.10, 0.35, 0.10, 0.0, 0.0),
        ("gaic_subject_composition", 0.65, 1.35, 1.05, 0.00, 0.10, 0.65, 0.0, 0.0, 0.0),
        ("gaic_composition_first", 0.35, 0.45, 1.65, 0.00, 0.00, 0.35, 0.0, 0.0, 0.05),
        ("gaic_aesthetic_align", 1.20, 0.25, 0.80, 0.00, 0.00, 0.25, 0.15, 0.08, 0.0),
        ("gaic_subject_safety_gate", 0.55, 1.55, 0.95, 0.00, 0.05, 0.90, 0.0, 0.0, 0.0),
        ("gaic_rule_comp_context", 0.35, 0.35, 1.55, 0.00, 0.00, 0.35, 0.0, 0.0, 0.12),
        ("gaic_portrait_subject_rules", 0.45, 1.45, 0.85, 0.00, 0.05, 0.65, 0.0, 0.0, 0.0),
        ("gaic_soft_safety_mid_area", 0.95, 0.80, 1.25, 0.00, 0.00, 0.35, 0.05, 0.0, 0.0),
    ]
    designs: List[Dict[str, Any]] = []
    for name, w_a, w_s, w_c, w_t, area, safety, aest_extra, align_extra, ctx_extra in seeds:
        designs.append(
            {
                "name": name,
                "family": "component_weight_seed",
                "w_a": w_a,
                "w_s": w_s,
                "w_c": w_c,
                "w_t": w_t,
                "area_scale": area,
                "safety_total_scale": safety,
                "aesthetic_extra": aest_extra,
                "align_extra": align_extra,
                "context_extra": ctx_extra,
            }
        )
    designs[-3].update({"comp_extra": 0.08, "horizon_extra": 0.08, "sym_extra": 0.05})
    designs[-2].update({"cut_penalty": 0.06, "headroom_extra": 0.10, "lookroom_extra": 0.08})
    designs[-1].update({"mid_area_bonus": 0.10, "mid_area_log_target": math.log(0.65), "mid_area_log_width": 0.45})

    component_families = [
        {
            "name": "family_composition_rules_no_area",
            "family": "component_rule_design",
            "w_a": 0.50,
            "w_s": 0.35,
            "w_c": 1.70,
            "w_t": 0.0,
            "area_scale": 0.0,
            "safety_total_scale": 0.35,
            "comp_extra": 0.10,
            "horizon_extra": 0.08,
            "sym_extra": 0.06,
            "context_extra": 0.06,
        },
        {
            "name": "family_subject_crop_rules",
            "family": "component_rule_design",
            "w_a": 0.55,
            "w_s": 1.45,
            "w_c": 0.95,
            "w_t": 0.0,
            "area_scale": 0.05,
            "safety_total_scale": 0.55,
            "cut_penalty": 0.08,
            "headroom_extra": 0.12,
            "lookroom_extra": 0.10,
        },
        {
            "name": "family_aesthetic_alignment",
            "family": "component_rule_design",
            "w_a": 1.35,
            "w_s": 0.25,
            "w_c": 0.85,
            "w_t": 0.0,
            "area_scale": 0.0,
            "safety_total_scale": 0.25,
            "aesthetic_extra": 0.18,
            "align_extra": 0.10,
            "text_penalty": 0.04,
        },
        {
            "name": "family_freeform_geometry",
            "family": "component_rule_design",
            "w_a": 0.85,
            "w_s": 0.70,
            "w_c": 1.20,
            "w_t": 0.0,
            "area_scale": 0.0,
            "safety_total_scale": 0.45,
            "free_ar_penalty": 0.06,
            "mid_area_bonus": 0.12,
            "mid_area_log_target": math.log(0.62),
            "mid_area_log_width": 0.42,
            "ar_extreme_penalty": 0.04,
        },
        {
            "name": "family_text_safe_composition",
            "family": "component_rule_design",
            "w_a": 0.70,
            "w_s": 0.55,
            "w_c": 1.35,
            "w_t": 0.0,
            "area_scale": 0.0,
            "safety_total_scale": 0.40,
            "text_penalty": 0.07,
            "context_extra": 0.07,
            "comp_extra": 0.06,
        },
    ]
    designs.extend(component_families)

    field_blends = [
        {
            "name": "blend_prob_policy_rank",
            "family": "field_blend",
            "formula": "field_blend",
            "field_weights": {"crop_utility_prob": 0.45, "score_policy_sigmoid_z_local": 0.35, "score_rank_pct": 0.20},
        },
        {
            "name": "blend_raw_prob_rank",
            "family": "field_blend",
            "formula": "field_blend",
            "field_weights": {"crop_utility_raw": 0.45, "crop_utility_prob": 0.35, "score_rank_pct": 0.20},
        },
        {
            "name": "blend_policy_z_rank",
            "family": "field_blend",
            "formula": "field_blend",
            "field_weights": {"score_policy_sigmoid_z_local": 0.45, "score_policy_z_local": 0.20, "score_rank_pct": 0.35},
        },
        {
            "name": "blend_policy_prob_raw",
            "family": "field_blend",
            "formula": "field_blend",
            "field_weights": {"score_policy_sigmoid_z_local": 0.30, "crop_utility_prob": 0.40, "crop_utility_raw": 0.30},
        },
    ]
    designs.extend(field_blends)

    for base in list(component_families[:4]) + [designs[0], designs[1]]:
        for transform in ("rank_pct", "sigmoid_z"):
            variant = dict(base)
            variant["name"] = f"{base['name']}__{transform}"
            variant["family"] = "component_local_transform"
            variant["post_transform"] = transform
            designs.append(variant)

    hybrids = []
    for base_name, base in (
        ("default", designs[0]),
        ("composition", component_families[0]),
        ("subject", component_families[1]),
        ("freeform", component_families[3]),
    ):
        for field_name, weights in (
            ("prob_rank", {"crop_utility_prob": 0.55, "score_rank_pct": 0.45}),
            ("policy_sigmoid", {"score_policy_sigmoid_z_local": 0.60, "crop_utility_prob": 0.40}),
        ):
            spec = dict(base)
            spec.update(
                {
                    "name": f"hybrid_{base_name}_{field_name}",
                    "family": "component_field_hybrid",
                    "formula": "hybrid",
                    "component_scale": 0.55,
                    "field_scale": 0.45,
                    "field_weights": weights,
                }
            )
            hybrids.append(spec)
    designs.extend(hybrids)

    mode_default = {
        "w_a": 0.85,
        "w_s": 0.80,
        "w_c": 1.20,
        "w_t": 0.0,
        "area_scale": 0.05,
        "safety_total_scale": 0.45,
        "cut_penalty": 0.06,
        "free_ar_penalty": 0.04,
    }
    mode_specs = {
        "default": mode_default,
        "portrait": {
            "w_a": 0.50,
            "w_s": 1.55,
            "w_c": 1.05,
            "area_scale": 0.05,
            "safety_total_scale": 0.55,
            "cut_penalty": 0.12,
            "headroom_extra": 0.10,
            "lookroom_extra": 0.08,
        },
        "person": {
            "w_a": 0.50,
            "w_s": 1.55,
            "w_c": 1.05,
            "area_scale": 0.05,
            "safety_total_scale": 0.55,
            "cut_penalty": 0.12,
            "headroom_extra": 0.10,
            "lookroom_extra": 0.08,
        },
        "scene": {
            "w_a": 0.80,
            "w_s": 0.25,
            "w_c": 1.65,
            "area_scale": 0.0,
            "safety_total_scale": 0.35,
            "context_extra": 0.10,
            "horizon_extra": 0.10,
            "sym_extra": 0.05,
        },
        "text": {
            "w_a": 0.65,
            "w_s": 0.55,
            "w_c": 1.35,
            "area_scale": 0.0,
            "safety_total_scale": 0.45,
            "text_penalty": 0.10,
            "context_extra": 0.08,
        },
        "copyspace": {
            "w_a": 0.75,
            "w_s": 0.55,
            "w_c": 1.40,
            "area_scale": 0.0,
            "safety_total_scale": 0.35,
            "context_extra": 0.12,
            "comp_extra": 0.08,
        },
    }
    designs.append(
        {
            "name": "mode_aware_subject_scene_text_v1",
            "family": "mode_aware_component",
            "formula": "mode_aware",
            **mode_default,
            "mode_specs": mode_specs,
        }
    )
    designs.append(
        {
            "name": "mode_aware_subject_scene_text_v1__sigmoid_z",
            "family": "mode_aware_component",
            "formula": "mode_aware",
            "post_transform": "sigmoid_z",
            **mode_default,
            "mode_specs": mode_specs,
        }
    )

    for w_a in (0.6, 1.0, 1.4):
        for w_s in (0.4, 0.9, 1.3):
            for w_c in (0.8, 1.2, 1.6):
                for area in (0.0, 0.15, 0.35):
                    for safety in (0.25, 0.55, 0.85):
                        designs.append(
                            {
                                "name": f"grid_a{w_a:g}_s{w_s:g}_c{w_c:g}_area{area:g}_safe{safety:g}",
                                "family": "macro_area_safety_grid",
                                "w_a": w_a,
                                "w_s": w_s,
                                "w_c": w_c,
                                "w_t": 0.0,
                                "area_scale": area,
                                "safety_total_scale": safety,
                            }
                        )
    return designs


def evaluate_design(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    scores = scores_for_spec(rows, spec)
    result = grouped_metric_summary(rows, scores, method=str(spec["name"]))
    result["family"] = str(spec.get("family", spec.get("formula", "component")))
    result["design"] = {k: v for k, v in spec.items() if k != "name"}
    result["objective"] = round(objective(result["metrics"]), 9)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def report_table(title: str, results: Sequence[Dict[str, Any]], limit: int = 20) -> List[str]:
    lines = [
        f"### {title}",
        "",
        "| method | family | images | objective | PCC | SRCC | Acc1/10 | top1 MOS | regret |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results[:limit]:
        metrics = result["metrics"]
        lines.append(
            "| {method} | {family} | {images} | {obj:.6f} | {pcc:.6f} | {srcc:.6f} | {acc:.6f} | {top:.6f} | {reg:.6f} |".format(
                method=result["method"],
                family=result.get("family", result.get("design", {}).get("family", "")),
                images=result["image_count"],
                obj=safe_float(result.get("objective"), objective(metrics)),
                pcc=safe_float(metrics.get("pcc")),
                srcc=safe_float(metrics.get("srcc")),
                acc=safe_float(metrics.get("acc1_of_top10")),
                top=safe_float(metrics.get("top1_mos")),
                reg=safe_float(metrics.get("top1_mos_regret")),
            )
        )
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep GAIC teacher score profile/design formulas from candidate_eval rows.")
    parser.add_argument("--train_candidate_eval_jsonl", default="")
    parser.add_argument("--val_candidate_eval_jsonl", required=True)
    parser.add_argument("--test_candidate_eval_jsonl", default="")
    parser.add_argument("--protocol", default="Gc", choices=["Gc", "Ge"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--eval_top_n_train_test", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_rows(Path(args.train_candidate_eval_jsonl), protocol=args.protocol) if args.train_candidate_eval_jsonl else []
    val_rows = read_rows(Path(args.val_candidate_eval_jsonl), protocol=args.protocol)
    test_rows = read_rows(Path(args.test_candidate_eval_jsonl), protocol=args.protocol) if args.test_candidate_eval_jsonl else []

    if not val_rows:
        raise SystemExit("val_candidate_eval_jsonl has no rows for requested protocol")

    baseline_val = []
    for field in BASELINE_FIELDS:
        if field in val_rows[0]:
            result = evaluate_field(val_rows, field)
            baseline_val.append(result)
    baseline_val.sort(key=lambda item: item["objective"], reverse=True)

    baseline_train = []
    if train_rows:
        for field in BASELINE_FIELDS:
            if field in train_rows[0]:
                baseline_train.append(evaluate_field(train_rows, field))
        baseline_train.sort(key=lambda item: item["objective"], reverse=True)

    baseline_test = []
    if test_rows:
        for field in BASELINE_FIELDS:
            if field in test_rows[0]:
                baseline_test.append(evaluate_field(test_rows, field))
        baseline_test.sort(key=lambda item: item["objective"], reverse=True)

    design_val: List[Dict[str, Any]] = []
    profile_fields_available = has_profile_fields(val_rows)
    if profile_fields_available:
        design_val = [evaluate_design(val_rows, spec) for spec in default_designs()]
        design_val.sort(key=lambda item: item["objective"], reverse=True)

    best_result = design_val[0] if design_val and design_val[0]["objective"] >= baseline_val[0]["objective"] else baseline_val[0]
    best_method = best_result["method"]
    best_spec = best_result.get("design", {}) if "design" in best_result else {"field": best_method}

    design_train: List[Dict[str, Any]] = []
    design_test: List[Dict[str, Any]] = []
    if design_val and (train_rows or test_rows):
        by_name = {str(spec["name"]): spec for spec in default_designs()}
        top_names = [str(item["method"]) for item in design_val[: max(1, int(args.eval_top_n_train_test))]]
        top_specs = [by_name[name] for name in top_names if name in by_name]
        if train_rows:
            design_train = [evaluate_design(train_rows, spec) for spec in top_specs]
            design_train.sort(key=lambda item: item["objective"], reverse=True)
        if test_rows:
            design_test = [evaluate_design(test_rows, spec) for spec in top_specs]
            design_test.sort(key=lambda item: item["objective"], reverse=True)

    payload: Dict[str, Any] = {
        "protocol": args.protocol,
        "profile_fields_available": profile_fields_available,
        "row_counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "best_method_by_val": best_method,
        "best_spec_by_val": best_spec,
        "baseline_train": baseline_train,
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "design_train": design_train,
        "design_val": design_val,
        "design_test": design_test,
    }

    if train_rows:
        payload["best_train"] = (
            evaluate_design(train_rows, {"name": best_method, **best_spec})
            if profile_fields_available and "field" not in best_spec
            else evaluate_field(train_rows, str(best_spec["field"]))
        )
    if test_rows:
        payload["best_test"] = (
            evaluate_design(test_rows, {"name": best_method, **best_spec})
            if profile_fields_available and "field" not in best_spec
            else evaluate_field(test_rows, str(best_spec["field"]))
        )

    out_dir = Path(args.output_dir)
    write_json(out_dir / "score_design_sweep_summary.json", payload)

    lines = [
        "# GAIC Teacher Score Design Sweep",
        "",
        f"- protocol: `{args.protocol}`",
        f"- profile_fields_available: `{profile_fields_available}`",
        f"- rows: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}",
        f"- best_method_by_val: `{best_method}`",
        "",
    ]
    lines.extend(report_table("Baseline Fields On Val", baseline_val, limit=args.top_k))
    if baseline_train:
        lines.extend(report_table("Baseline Fields On Train", baseline_train, limit=args.top_k))
    if baseline_test:
        lines.extend(report_table("Baseline Fields On Test", baseline_test, limit=args.top_k))
    if design_val:
        lines.extend(report_table("Score Designs On Val", design_val, limit=args.top_k))
    if design_train:
        lines.extend(report_table("Top-Val Score Designs On Train", design_train, limit=args.top_k))
    if design_test:
        lines.extend(report_table("Top-Val Score Designs On Test", design_test, limit=args.top_k))
    if "best_train" in payload:
        lines.extend(report_table("Best Method On Train", [payload["best_train"]], limit=1))
    if "best_test" in payload:
        lines.extend(report_table("Best Method On Test", [payload["best_test"]], limit=1))
    (out_dir / "SCORE_DESIGN_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "best_method_by_val": best_method}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
