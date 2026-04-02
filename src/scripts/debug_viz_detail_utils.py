from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def round_opt(value: Any, digits: int = 6) -> Optional[float]:
    parsed = maybe_float(value)
    if parsed is None:
        return None
    return round(float(parsed), digits)


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def load_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "na"


def checklist_label(checklist: Dict[str, Any], checklist_labels: Dict[str, Any], key: str) -> str:
    direct = checklist_labels.get(key)
    if direct not in (None, ""):
        return normalize_label(direct)
    return normalize_label(safe_dict(checklist.get(key)).get("label"))


def checklist_value(checklist: Dict[str, Any], key: str, subkey: str = "value") -> Optional[float]:
    return round_opt(safe_dict(checklist.get(key)).get(subkey))


def dominant_composition_anchor(
    third_dist: Optional[float],
    phi_dist: Optional[float],
    center_dist: Optional[float],
) -> str:
    candidates = [
        ("thirds", third_dist),
        ("phi", phi_dist),
        ("center", center_dist),
    ]
    valid = [(name, value) for name, value in candidates if value is not None]
    if not valid:
        return "na"
    valid.sort(key=lambda item: (item[1], item[0]))
    return valid[0][0]


def extract_raw_candidate_detail(candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = copy.deepcopy(safe_dict(candidate))
    checklist = safe_dict(raw.get("checklist"))
    checklist_labels = safe_dict(raw.get("checklist_labels"))
    scores = safe_dict(raw.get("scores"))
    score_components = safe_dict(scores.get("components"))
    macro_scores = safe_dict(raw.get("macro_scores"))
    macro_components = safe_dict(raw.get("macro_components"))
    safety_bundle = safe_dict(scores.get("safety_penalty_components"))
    why_tags = [str(tag) for tag in safe_list(raw.get("why_tags"))]
    reject_tags = [str(tag) for tag in safe_list(raw.get("reject_tags"))]

    third_dist = checklist_value(checklist, "third_dist")
    phi_dist = checklist_value(checklist, "phi_dist")
    center_dist = checklist_value(checklist, "center_dist")
    headroom_ratio = checklist_value(checklist, "headroom")
    lookroom_ratio = checklist_value(checklist, "lookroom")
    subject_coverage_ratio = checklist_value(checklist, "subject_coverage")
    subject_scale_ratio = checklist_value(checklist, "subject_scale")
    text_keep_ratio = checklist_value(checklist, "text_keep_ratio")
    horizon_y = checklist_value(checklist, "horizon")
    horizon_visible_ratio = checklist_value(checklist, "horizon", "visible_ratio")
    context_value = checklist_value(checklist, "context")
    teacher_rho = checklist_value(checklist, "teacher_consensus")
    symmetry_score = round_opt(
        macro_components.get("C_sym", score_components.get("r_sym")),
    )

    return {
        "candidate_id": str(raw.get("candidate_id", "")),
        "source": str(raw.get("source", "")),
        "bbox_norm_xyxy": list(raw.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
        "score_detail": {
            "score_prob": round_opt(
                first_nonempty(
                    raw.get("score_policy_sigmoid_z_local"),
                    raw.get("score_sigmoid_z_local"),
                ),
                9,
            ),
            "score_raw_rank": round_opt(scores.get("rank")),
            "score_raw_policy": round_opt(scores.get("policy", scores.get("final"))),
            "score_raw_policy_base": round_opt(scores.get("policy_base", scores.get("policy", scores.get("final")))),
            "score_raw_policy_safe": round_opt(scores.get("policy_safe", scores.get("policy", scores.get("final")))),
            "score_raw_rank_macro": round_opt(scores.get("rank_macro")),
            "area_log_prior": round_opt(scores.get("area_log_prior")),
            "safety_penalty_total": round_opt(scores.get("safety_penalty_total")),
            "safety_penalty_soft": round_opt(scores.get("safety_penalty_soft")),
            "safety_penalty_hard": round_opt(scores.get("safety_penalty_hard")),
        },
        "macro_scores": {
            key: round_opt(macro_scores.get(key))
            for key in ("A_macro", "S_macro", "C_macro", "T_macro")
        },
        "macro_components": {
            key: round_opt(macro_components.get(key))
            for key in (
                "A_aesthetic",
                "A_align",
                "S_cov",
                "S_scale",
                "S_support_structure",
                "S_border",
                "S_softcut_quality",
                "C_comp",
                "C_place",
                "C_comp_linear",
                "C_place_margin",
                "C_headroom",
                "C_lookroom",
                "C_horizon_y",
                "C_sym",
                "C_context",
                "C_copyspace",
                "T_teacher",
            )
        },
        "checklist_labels": {
            "subject_coverage": checklist_label(checklist, checklist_labels, "subject_coverage"),
            "subject_scale": checklist_label(checklist, checklist_labels, "subject_scale"),
            "headroom": checklist_label(checklist, checklist_labels, "headroom"),
            "lookroom": checklist_label(checklist, checklist_labels, "lookroom"),
            "face_cut": checklist_label(checklist, checklist_labels, "face_cut"),
            "joint_cut": checklist_label(checklist, checklist_labels, "joint_cut"),
            "text_keep": checklist_label(checklist, checklist_labels, "text_keep_ratio"),
            "copyspace": checklist_label(checklist, checklist_labels, "copyspace"),
            "horizon_state": normalize_label(safe_dict(checklist.get("horizon")).get("state", safe_dict(checklist.get("horizon")).get("label"))),
            "context": checklist_label(checklist, checklist_labels, "context"),
            "third_dist": checklist_label(checklist, checklist_labels, "third_dist"),
            "phi_dist": checklist_label(checklist, checklist_labels, "phi_dist"),
            "center_dist": checklist_label(checklist, checklist_labels, "center_dist"),
            "teacher_consensus": checklist_label(checklist, checklist_labels, "teacher_consensus"),
            "crop_tightness": checklist_label(checklist, checklist_labels, "crop_tightness"),
            "roll": checklist_label(checklist, checklist_labels, "roll"),
            "ar": checklist_label(checklist, checklist_labels, "ar"),
        },
        "checklist_scores": {
            "subject_coverage_ratio": subject_coverage_ratio,
            "subject_scale_ratio": subject_scale_ratio,
            "headroom_ratio": headroom_ratio,
            "lookroom_ratio": lookroom_ratio,
            "text_keep_ratio": text_keep_ratio,
            "third_dist": third_dist,
            "phi_dist": phi_dist,
            "center_dist": center_dist,
            "horizon_y": horizon_y,
            "horizon_visible_ratio": horizon_visible_ratio,
            "context_value": round_opt(score_components.get("context_value", context_value)),
            "teacher_rho": round_opt(score_components.get("teacher_rho", teacher_rho)),
            "symmetry_score": symmetry_score,
            "copyspace_blank_ratio_keep": round_opt(score_components.get("copyspace_blank_ratio_keep")),
        },
        "composition_focus": {
            "dominant_anchor": dominant_composition_anchor(third_dist, phi_dist, center_dist),
            "placement_family_best": str(score_components.get("placement_family_best", "na")),
            "placement_family_margin": round_opt(score_components.get("placement_family_margin")),
            "placement_score": round_opt(score_components.get("r_place")),
            "placement_linear": round_opt(score_components.get("r_comp_linear")),
            "placement_reward_third": round_opt(score_components.get("placement_reward_third")),
            "placement_reward_phi": round_opt(score_components.get("placement_reward_phi")),
            "placement_reward_center": round_opt(score_components.get("placement_reward_center")),
            "placement_score_third": round_opt(score_components.get("placement_score_third")),
            "placement_score_phi": round_opt(score_components.get("placement_score_phi")),
            "placement_score_center": round_opt(score_components.get("placement_score_center")),
            "active_tags": [
                tag for tag in ("rule_of_thirds", "phi_grid", "centered_subject", "symmetry")
                if tag in why_tags
            ],
        },
        "safety_penalty": {
            "total": round_opt(safety_bundle.get("total")),
            "soft_total": round_opt(safety_bundle.get("soft_total")),
            "hard_total": round_opt(safety_bundle.get("hard_total")),
            "soft_components": {
                key: round_opt(value)
                for key, value in safe_dict(safety_bundle.get("soft_components")).items()
            },
            "hard_components": {
                key: round_opt(value)
                for key, value in safe_dict(safety_bundle.get("hard_components")).items()
            },
            "signals": {
                key: round_opt(value)
                for key, value in safe_dict(safety_bundle.get("signals")).items()
            },
        },
        "why_tags": why_tags,
        "reject_tags": reject_tags,
        "why_text_template": str(raw.get("why_text_template", "")).strip(),
    }


def iter_teacher_ar_candidates(ar_result: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    bucket_keys = [
        "cheap_top_m",
        "selected_topk",
        "hard_negatives",
        "also_considered_rejected",
    ]
    single_keys = [
        "best_candidate",
        "baseline_candidate",
        "best",
        "baseline_maxarea_subject",
    ]
    for key in bucket_keys:
        for candidate in safe_list(ar_result.get(key)):
            cand = safe_dict(candidate)
            if cand:
                yield cand
    for key in single_keys:
        cand = safe_dict(ar_result.get(key))
        if cand:
            yield cand


def build_teacher_detail_index(path: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    if not path.exists():
        return {}
    by_image: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for row in load_jsonl_rows(path):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        results_by_ar = safe_dict(safe_dict(row.get("teacher_scorer")).get("results_by_ar"))
        if not results_by_ar:
            continue
        image_map = by_image.setdefault(image_id, {})
        for target_ar, ar_result in results_by_ar.items():
            ar_map = image_map.setdefault(str(target_ar), {})
            for candidate in iter_teacher_ar_candidates(safe_dict(ar_result)):
                candidate_id = str(candidate.get("candidate_id", "")).strip()
                if not candidate_id or candidate_id in ar_map:
                    continue
                ar_map[candidate_id] = extract_raw_candidate_detail(candidate)
    return by_image
