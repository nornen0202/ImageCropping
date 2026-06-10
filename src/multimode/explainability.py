from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Mapping, Sequence

from .mode_catalog import MODE_SPEC_BY_NAME


SCHEMA_VERSION = "multimode_explainability_v1"

CHECKLIST_CLASS_KEYS = (
    "subject_coverage",
    "subject_scale",
    "headroom",
    "lookroom",
    "face_cut",
    "joint_cut",
    "copyspace",
    "context",
    "third_dist",
    "phi_dist",
    "center_dist",
    "ar",
    "horizon_state",
)

WHY_TAG_ORDER = (
    "avoid_face_cut",
    "avoid_person_cut",
    "symmetry",
    "ar_fits_well",
    "ar_choice_freeform",
    "balanced_crop",
    "wide_crop",
    "tight_crop",
    "context_preserved",
    "context_loss",
    "rule_of_thirds",
    "phi_grid",
    "centered_subject",
    "subject_preserved",
    "subject_poor",
    "subject_marginal",
    "subject_scale_ideal",
    "subject_scale_loose",
    "subject_scale_tight",
    "subject_terms_neutralized",
    "head_top_safe",
    "headroom_ok",
    "headroom_violation",
    "lookroom_ok",
    "lookroom_violation",
    "copy_space_kept",
    "copy_space_lost",
    "needs_leveling",
    "horizon_on_target",
    "horizon_off_target",
    "ar_extreme_penalty",
)

PERSON_MODES = {"single_person_center", "single_person_rot", "group_center", "group_rot", "face"}
GROUP_MODES = {"group_center", "group_rot"}
OBJECT_MODES = {"object_single_center", "object_single_rot", "object_multi_center", "object_multi_rot"}
ROT_MODES = {"single_person_rot", "group_rot", "object_single_rot", "object_multi_rot"}
CENTER_MODES = {"single_person_center", "group_center", "face", "object_single_center", "object_multi_center", "landscape"}


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0, default: float | None = None) -> float | None:
    parsed = _safe_float(value, default)
    if parsed is None:
        return None
    return max(float(lo), min(float(hi), float(parsed)))


def _round(value: Any, ndigits: int = 6) -> float | None:
    parsed = _safe_float(value, None)
    if parsed is None:
        return None
    return round(float(parsed), int(ndigits))


def _strength_from_distance(value: Any, *, scale: float = 0.30) -> float | None:
    parsed = _safe_float(value, None)
    if parsed is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - float(parsed) / max(1e-6, float(scale)))), 6)


def _component(components: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    return _safe_float(components.get(key), default)


def _mode_family(mode_name: str) -> str:
    if mode_name == "landscape":
        return "scene_general"
    if mode_name.startswith("group"):
        return "portrait_group"
    if mode_name in {"single_person_center", "single_person_rot", "face"}:
        return "portrait_single"
    if mode_name.startswith("object_single"):
        return "object_single"
    if mode_name.startswith("object_multi"):
        return "object_multi"
    return "other_ambiguous"


def _is_subject_applicable(mode_name: str, attributes: Mapping[str, Any], components: Mapping[str, Any]) -> bool:
    if mode_name == "landscape":
        return int(_safe_float(components.get("landscape_subject_safe_active"), 0.0) or 0) == 1
    return mode_name in PERSON_MODES or mode_name in OBJECT_MODES


def _coverage_label(value: float | None) -> str:
    if value is None:
        return "na"
    if value >= 0.98:
        return "excellent"
    if value >= 0.90:
        return "good"
    if value >= 0.70:
        return "marginal"
    return "poor"


def _scale_label(mode_name: str, subject_ratio: float | None) -> str:
    if subject_ratio is None or mode_name not in MODE_SPEC_BY_NAME:
        return "na"
    spec = MODE_SPEC_BY_NAME[mode_name]
    target = float(spec.subject_ratio_target)
    sigma = float(spec.subject_ratio_sigma)
    loose_cut = max(0.0, target - 0.80 * sigma)
    tight_cut = min(1.0, target + 0.90 * sigma)
    if subject_ratio < loose_cut:
        return "too_loose"
    if subject_ratio > tight_cut:
        return "too_tight"
    return "ideal_scale"


def _headroom_label(mode_name: str, headroom_value: float | None) -> str:
    if mode_name not in PERSON_MODES or headroom_value is None:
        return "na"
    if mode_name == "face":
        tight_cut, loose_cut = 0.03, 0.26
    elif mode_name.startswith("group"):
        tight_cut, loose_cut = 0.035, 0.18
    else:
        tight_cut, loose_cut = 0.04, 0.20
    if headroom_value < tight_cut:
        return "headroom_tight"
    if headroom_value > loose_cut:
        return "headroom_loose"
    return "headroom_ok"


def _lookroom_label(mode_name: str, lookroom_value: float | None, gaze_dir: str) -> str:
    if mode_name not in PERSON_MODES or lookroom_value is None or str(gaze_dir) not in {"left", "right"}:
        return "na"
    if lookroom_value < 1.05:
        return "lookroom_insufficient"
    if lookroom_value > 2.50:
        return "lookroom_excessive"
    return "lookroom_adequate"


def _face_cut_label(mode_name: str, face_recall: float | None) -> str:
    if mode_name not in PERSON_MODES or face_recall is None:
        return "na"
    if face_recall >= 0.98:
        return "no_face_cut"
    if face_recall >= 0.90:
        return "face_cut_mild"
    return "face_cut"


def _joint_cut_label(mode_name: str, joint_cut_score: float | None) -> str:
    if mode_name not in PERSON_MODES or joint_cut_score is None:
        return "na"
    if joint_cut_score <= 0.05:
        return "no_joint_cut"
    if joint_cut_score <= 0.35:
        return "joint_cut_mild"
    return "joint_cut"


def _context_label(context_value: float | None) -> str:
    if context_value is None:
        return "na"
    if context_value < 0.20:
        return "context_poor"
    if context_value < 0.45:
        return "context_partial"
    if context_value <= 0.82:
        return "context_preserved"
    return "context_excessive"


def _third_label(mode_name: str, distance: float | None) -> str:
    if mode_name not in ROT_MODES or distance is None:
        return "na"
    if distance <= 0.075:
        return "rule_of_thirds_strong"
    return "rule_of_thirds_weak"


def _center_label(mode_name: str, distance: float | None) -> str:
    if mode_name not in CENTER_MODES or distance is None:
        return "na"
    if distance <= 0.125:
        return "center_comp_strong"
    return "center_comp_weak"


def _ar_label(target_ar: str, q_ar: float | None, ar_rel_error: float | None) -> str:
    if str(target_ar or "").strip().upper() == "FREE":
        return "ar_choice_freeform"
    if q_ar is not None and q_ar >= 0.97:
        return "ar_fits_well"
    if ar_rel_error is not None and ar_rel_error <= 0.035:
        return "ar_fits_well"
    return "na"


def _horizon_label(mode_name: str, q_scene: float | None) -> str:
    if mode_name != "landscape" or q_scene is None:
        return "na"
    if q_scene >= 0.75:
        return "strong"
    if q_scene >= 0.55:
        return "ok"
    return "weak"


def _copyspace_label(attributes: Mapping[str, Any], context_value: float | None) -> str:
    if int(_safe_float(attributes.get("copyspace_intent"), 0.0) or 0) != 1:
        return "na"
    if context_value is None:
        return "copyspace_partial"
    if context_value >= 0.60:
        return "copyspace_preserved"
    if context_value >= 0.35:
        return "copyspace_partial"
    return "copyspace_missing"


def _center_distance(components: Mapping[str, Any]) -> float | None:
    control = _component(components, "portrait_control_center_deviation", None)
    if control is not None:
        return abs(float(control))
    rx = _component(components, "anchor_rx", None)
    ry = _component(components, "anchor_ry", None)
    if rx is None or ry is None:
        return None
    return math.sqrt((float(rx) - 0.5) ** 2 + (float(ry) - 0.5) ** 2)


def _third_distance(components: Mapping[str, Any]) -> float | None:
    control = _component(components, "portrait_control_thirds_deviation", None)
    if control is not None:
        return abs(float(control))
    fallback = _component(components, "rot_thirds_deviation", None)
    if fallback is not None:
        return abs(float(fallback))
    rx = _component(components, "anchor_rx", None)
    if rx is None:
        return None
    return min(abs(float(rx) - (1.0 / 3.0)), abs(float(rx) - (2.0 / 3.0)))


def _ordered_tags(tags: set[str]) -> list[str]:
    return [tag for tag in WHY_TAG_ORDER if tag in tags]


def build_explainability_payload(
    *,
    mode_name: str,
    target_ar: str,
    score_mode: float,
    components: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
    hard_reject_reasons: Sequence[str] | None = None,
) -> Dict[str, Any]:
    attrs = attributes if isinstance(attributes, Mapping) else {}
    rejects = {str(reason) for reason in (hard_reject_reasons or [])}
    subject_applicable = _is_subject_applicable(mode_name, attrs, components)
    subject_coverage_ratio = _component(components, "q_subj", None)
    subject_ratio = _component(components, "subject_ratio", None)
    headroom_value = _component(components, "headroom_value", None)
    lookroom_value = _component(components, "lookroom_value", None)
    gaze_dir = str(components.get("gaze_dir") or "unknown")
    face_recall = _component(components, "face_recall", None)
    joint_cut_score = _component(components, "joint_cut_score", None)
    context_value = _component(components, "context_value", None)
    q_ar = _component(components, "q_ar", None)
    ar_rel_error = _component(components, "ar_rel_error", None)
    q_scene = _component(components, "q_scene", None)
    q_head = _component(components, "q_head", None)
    q_look = _component(components, "q_look", None)
    q_ctx = _component(components, "q_ctx", None)
    q_place = _component(components, "q_place", None)
    third_dist = _third_distance(components)
    center_dist = _center_distance(components)

    labels = {key: "na" for key in CHECKLIST_CLASS_KEYS}
    if subject_applicable:
        labels["subject_coverage"] = _coverage_label(subject_coverage_ratio)
        labels["subject_scale"] = _scale_label(mode_name, subject_ratio)
    labels["headroom"] = _headroom_label(mode_name, headroom_value)
    labels["lookroom"] = _lookroom_label(mode_name, lookroom_value, gaze_dir)
    labels["face_cut"] = _face_cut_label(mode_name, face_recall)
    labels["joint_cut"] = _joint_cut_label(mode_name, joint_cut_score)
    labels["copyspace"] = _copyspace_label(attrs, context_value)
    labels["context"] = _context_label(context_value)
    labels["third_dist"] = _third_label(mode_name, third_dist)
    labels["center_dist"] = _center_label(mode_name, center_dist)
    labels["ar"] = _ar_label(target_ar, q_ar, ar_rel_error)
    labels["horizon_state"] = _horizon_label(mode_name, q_scene)

    scores: Dict[str, float] = {"final_score": round(float(score_mode), 6)}
    if subject_coverage_ratio is not None:
        scores["subject_coverage_ratio"] = round(float(_clip(subject_coverage_ratio, default=0.0) or 0.0), 6)
    if subject_ratio is not None:
        scores["subject_scale_ratio"] = round(float(_clip(subject_ratio, default=0.0) or 0.0), 6)
    if headroom_value is not None:
        scores["headroom_ratio"] = round(float(headroom_value), 6)
        scores["headroom_ratio_norm"] = round(float(_clip(float(headroom_value) / 0.60, default=0.0) or 0.0), 6)
    if lookroom_value is not None:
        scores["lookroom_ratio"] = round(float(lookroom_value), 6)
        scores["lookroom_ratio_norm"] = round(float(_clip(float(lookroom_value) / 3.0, default=0.0) or 0.0), 6)
    if third_dist is not None:
        scores["third_dist"] = round(float(third_dist), 6)
        strength = _strength_from_distance(third_dist)
        if strength is not None:
            scores["third_strength"] = strength
    if center_dist is not None:
        scores["center_dist"] = round(float(center_dist), 6)
        strength = _strength_from_distance(center_dist)
        if strength is not None:
            scores["center_strength"] = strength
    if context_value is not None:
        scores["context_value"] = round(float(_clip(context_value, default=0.0) or 0.0), 6)
    if q_scene is not None:
        scores["horizon_visible_ratio"] = round(float(_clip(q_scene, default=0.0) or 0.0), 6)
    symmetry = _component(components, "symmetry_score", None)
    if symmetry is not None:
        scores["symmetry_score"] = round(float(_clip(symmetry, default=0.0) or 0.0), 6)
    if q_place is not None:
        scores["placement_score"] = round(float(_clip(q_place, default=0.0) or 0.0), 6)
        if mode_name in ROT_MODES:
            scores["placement_reward_third"] = scores["placement_score"]
        if mode_name in CENTER_MODES:
            scores["placement_reward_center"] = scores["placement_score"]
    if q_head is not None:
        scores["C_headroom"] = round(float(_clip(q_head, default=0.0) or 0.0), 6)
    if q_look is not None:
        scores["C_lookroom"] = round(float(_clip(q_look, default=0.0) or 0.0), 6)
    if q_ctx is not None:
        scores["C_context"] = round(float(_clip(q_ctx, default=0.0) or 0.0), 6)
    scores["safety_penalty_hard"] = 1.0 if rejects else 0.0
    scores["safety_penalty_soft"] = round(float(_clip(1.0 - float(score_mode), default=0.0) or 0.0), 6)

    tags: set[str] = set()
    if labels["subject_coverage"] in {"good", "excellent"}:
        tags.add("subject_preserved")
    elif labels["subject_coverage"] == "marginal":
        tags.add("subject_marginal")
    elif labels["subject_coverage"] == "poor":
        tags.add("subject_poor")
    if labels["subject_scale"] == "ideal_scale":
        tags.add("subject_scale_ideal")
    elif labels["subject_scale"] == "too_loose":
        tags.add("subject_scale_loose")
        tags.add("wide_crop")
    elif labels["subject_scale"] == "too_tight":
        tags.add("subject_scale_tight")
        tags.add("tight_crop")
    if labels["headroom"] == "headroom_ok":
        tags.add("headroom_ok")
        tags.add("head_top_safe")
    elif labels["headroom"] in {"headroom_tight", "headroom_loose"}:
        tags.add("headroom_violation")
    if labels["lookroom"] == "lookroom_adequate":
        tags.add("lookroom_ok")
    elif labels["lookroom"] in {"lookroom_insufficient", "lookroom_excessive"}:
        tags.add("lookroom_violation")
    if labels["face_cut"] == "no_face_cut":
        tags.add("avoid_face_cut")
    if labels["joint_cut"] == "no_joint_cut":
        tags.add("avoid_person_cut")
    if labels["context"] == "context_preserved":
        tags.add("context_preserved")
    elif labels["context"] in {"context_poor", "context_partial"}:
        tags.add("context_loss")
    if labels["third_dist"] == "rule_of_thirds_strong":
        tags.add("rule_of_thirds")
    if labels["center_dist"] == "center_comp_strong":
        tags.add("centered_subject")
    if labels["ar"] == "ar_fits_well":
        tags.add("ar_fits_well")
    elif labels["ar"] == "ar_choice_freeform":
        tags.add("ar_choice_freeform")
    if labels["copyspace"] == "copyspace_preserved":
        tags.add("copy_space_kept")
    elif labels["copyspace"] == "copyspace_missing":
        tags.add("copy_space_lost")
    if labels["horizon_state"] == "strong":
        tags.add("horizon_on_target")
    elif labels["horizon_state"] == "weak":
        tags.add("horizon_off_target")
    if "target_ar_mismatch" in rejects:
        tags.add("ar_extreme_penalty")
    if "landscape_subject_safe_suppressed_reason" in attrs or str(attrs.get("landscape_subject_safe_suppressed_reason") or ""):
        tags.add("subject_terms_neutralized")

    applicable = {
        "subject_coverage": int(subject_applicable),
        "subject_scale": int(subject_applicable),
        "headroom": int(mode_name in PERSON_MODES),
        "lookroom": int(mode_name in PERSON_MODES and gaze_dir in {"left", "right"}),
        "face_cut": int(mode_name in PERSON_MODES),
        "joint_cut": int(mode_name in PERSON_MODES),
        "copyspace": int(labels["copyspace"] != "na"),
        "context": 1,
        "third_dist": int(mode_name in ROT_MODES),
        "phi_dist": 0,
        "center_dist": int(mode_name in CENTER_MODES),
        "ar": 1,
        "horizon_state": int(mode_name == "landscape"),
    }

    non_na_labels = {key: value for key, value in labels.items() if value != "na"}
    return {
        "schema_version": SCHEMA_VERSION,
        "mode_family": _mode_family(mode_name),
        "label_policy_version": "thresholds_20260602_v1",
        "checklist_labels": labels,
        "checklist_scores": scores,
        "checklist_applicable": applicable,
        "checklist_non_na_count": len(non_na_labels),
        "why_tags": _ordered_tags(tags),
    }


def attach_explainability_to_attributes(
    *,
    mode_name: str,
    target_ar: str,
    score_mode: float,
    attributes: Dict[str, Any],
    hard_reject_reasons: Sequence[str] | None = None,
) -> Dict[str, Any]:
    out = dict(attributes)
    components = out.get("score_components") if isinstance(out.get("score_components"), Mapping) else {}
    payload = build_explainability_payload(
        mode_name=mode_name,
        target_ar=target_ar,
        score_mode=score_mode,
        components=components,
        attributes=out,
        hard_reject_reasons=hard_reject_reasons if hard_reject_reasons is not None else out.get("hard_reject_reasons") or [],
    )
    out["explainability_schema_version"] = payload["schema_version"]
    out["explainability_label_policy_version"] = payload["label_policy_version"]
    out["checklist_labels"] = payload["checklist_labels"]
    out["checklist_scores"] = payload["checklist_scores"]
    out["checklist_applicable"] = payload["checklist_applicable"]
    out["why_tags"] = payload["why_tags"]
    out["teacher_checklist_labels"] = payload["checklist_labels"]
    out["teacher_checklist_scores"] = payload["checklist_scores"]
    out["teacher_why_tags"] = payload["why_tags"]
    out["explainability"] = {
        "schema_version": payload["schema_version"],
        "label_policy_version": payload["label_policy_version"],
        "mode_family": payload["mode_family"],
        "checklist_non_na_count": payload["checklist_non_na_count"],
    }
    return out


def explainability_policy_payload() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "label_policy_version": "thresholds_20260602_v1",
        "strategy": "Train regression heads for continuous checklist_scores, then derive checklist_labels with fixed thresholds.",
        "checklist_class_keys": list(CHECKLIST_CLASS_KEYS),
        "why_tag_order": list(WHY_TAG_ORDER),
        "score_keys": {
            "final_score": "Copy of the annotation-level score_mode stored inside attributes.checklist_scores for attribute-only training adapters.",
            "component_scores": "Continuous regression targets derived from attributes.score_components.",
        },
        "thresholds": {
            "subject_coverage": {"excellent": "q_subj >= 0.98", "good": "0.90 <= q_subj < 0.98", "marginal": "0.70 <= q_subj < 0.90", "poor": "q_subj < 0.70"},
            "subject_scale": {"too_loose": "subject_ratio < target - 0.8*sigma", "ideal_scale": "target - 0.8*sigma <= subject_ratio <= target + 0.9*sigma", "too_tight": "subject_ratio > target + 0.9*sigma"},
            "headroom": {"headroom_tight": "person/group/face mode-specific lower cut", "headroom_ok": "inside mode-specific interval", "headroom_loose": "above mode-specific upper cut"},
            "lookroom": {"lookroom_insufficient": "lookroom_ratio < 1.05", "lookroom_adequate": "1.05 <= lookroom_ratio <= 2.50", "lookroom_excessive": "lookroom_ratio > 2.50"},
            "face_cut": {"no_face_cut": "face_recall >= 0.98", "face_cut_mild": "0.90 <= face_recall < 0.98", "face_cut": "face_recall < 0.90"},
            "joint_cut": {"no_joint_cut": "joint_cut_score <= 0.05", "joint_cut_mild": "0.05 < joint_cut_score <= 0.35", "joint_cut": "joint_cut_score > 0.35"},
            "context": {"context_poor": "context_value < 0.20", "context_partial": "0.20 <= context_value < 0.45", "context_preserved": "0.45 <= context_value <= 0.82", "context_excessive": "context_value > 0.82"},
            "third_dist": {"rule_of_thirds_strong": "thirds distance <= 0.075", "rule_of_thirds_weak": "thirds distance > 0.075"},
            "center_dist": {"center_comp_strong": "center distance <= 0.125", "center_comp_weak": "center distance > 0.125"},
            "ar": {"ar_choice_freeform": "target_ar == FREE", "ar_fits_well": "q_ar >= 0.97 or ar_rel_error <= 0.035"},
            "horizon_state": {"strong": "q_scene >= 0.75", "ok": "0.55 <= q_scene < 0.75", "weak": "q_scene < 0.55"},
        },
    }


def summarize_explainability_annotations(annotations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    present = 0
    label_counts: Dict[str, Counter[str]] = {key: Counter() for key in CHECKLIST_CLASS_KEYS}
    tag_counts: Counter[str] = Counter()
    non_na_sum = 0
    missing_score_components = 0
    for ann in annotations:
        total += 1
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), Mapping) else {}
        if not isinstance(attrs.get("score_components"), Mapping):
            missing_score_components += 1
        labels = attrs.get("checklist_labels") if isinstance(attrs.get("checklist_labels"), Mapping) else None
        if labels is None:
            continue
        present += 1
        for key in CHECKLIST_CLASS_KEYS:
            value = str(labels.get(key, "na") or "na")
            label_counts[key][value] += 1
            if value != "na":
                non_na_sum += 1
        for tag in attrs.get("why_tags") or []:
            tag_counts[str(tag)] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_count": total,
        "explainability_annotation_count": present,
        "missing_explainability_annotation_count": total - present,
        "coverage_rate": round(present / float(total), 6) if total else 0.0,
        "missing_score_components_count": missing_score_components,
        "mean_non_na_checklist_labels_per_annotation": round(non_na_sum / float(present), 6) if present else 0.0,
        "checklist_label_counts": {key: dict(counter) for key, counter in label_counts.items()},
        "why_tag_counts": dict(tag_counts),
    }
