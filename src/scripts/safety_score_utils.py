from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def parse_target_ar_value(target_ar: Any) -> Optional[float]:
    if target_ar is None:
        return None
    text = str(target_ar).strip().upper()
    if not text or text == "FREE":
        return None
    if ":" in text:
        left, right = text.split(":", 1)
        left_f = safe_float(left, 0.0)
        right_f = safe_float(right, 0.0)
        if right_f <= 0.0:
            return None
        return left_f / right_f
    value = safe_float(text, 0.0)
    return value if value > 0.0 else None


@dataclass(frozen=True)
class SafetyPenaltyConfig:
    area_min: float = 0.15
    area_max: float = 0.95
    ar_hard_eps: float = 0.01
    head_top_min_margin: float = 0.008

    lambda_area: float = 1.90
    lambda_ar: float = 1.75
    lambda_face: float = 1.20
    lambda_head_top: float = 1.45
    lambda_joint: float = 1.10
    lambda_lookroom: float = 0.70
    lambda_text: float = 0.85
    lambda_subject_border: float = 0.25

    bonus_area_hard: float = 0.95
    bonus_ar_hard: float = 0.95
    bonus_face_hard: float = 0.85
    bonus_head_top_hard: float = 0.95
    bonus_joint_hard: float = 0.75
    bonus_lookroom_hard: float = 0.65
    bonus_text_hard: float = 0.60

    ar_violation_scale: float = 0.05
    lookroom_scale_floor: float = 0.25
    text_soft_cap: float = 2.0
    severity_cap: float = 2.0
    joint_relax_lambda_scale: float = 0.772727273
    joint_relax_bonus_scale: float = 0.733333333
    head_top_relax_lambda_scale: float = 0.793103448
    head_top_relax_bonus_scale: float = 0.736842105
    subject_border_relax_lambda_scale: float = 0.60
    relax_joint_head_top_min_coverage: float = 0.90
    relax_subject_border_min_coverage: float = 0.85
    relax_joint_max_severe_count: int = 1
    relax_subject_border_max_joint_score: float = 0.10


def _normalized_overflow(value: float, limit: float) -> float:
    return max(0.0, float(value)) / max(1e-6, float(limit))


def compute_safety_penalty_bundle(
    *,
    candidate: Dict[str, Any],
    target_ar_value: Optional[float],
    cfg: SafetyPenaltyConfig,
) -> Dict[str, Any]:
    area = safe_float(candidate.get("area_ratio", 0.0), 0.0)
    ar = safe_float(candidate.get("ar", 0.0), 0.0)
    flags = candidate.get("flags", {}) if isinstance(candidate.get("flags"), dict) else {}
    checks = candidate.get("composition_checks", {}) if isinstance(candidate.get("composition_checks"), dict) else {}
    cutoff = checks.get("cutoff", {}) if isinstance(checks.get("cutoff"), dict) else {}
    head_top = checks.get("head_top", {}) if isinstance(checks.get("head_top"), dict) else {}
    lookroom = checks.get("lookroom", {}) if isinstance(checks.get("lookroom"), dict) else {}
    text = checks.get("text", {}) if isinstance(checks.get("text"), dict) else {}
    checklist = candidate.get("checklist", {}) if isinstance(candidate.get("checklist"), dict) else {}
    reject_tags = {str(tag) for tag in (candidate.get("reject_tags") or [])}

    area_below = _normalized_overflow(cfg.area_min - area, cfg.area_min)
    area_above = _normalized_overflow(area - cfg.area_max, 1.0 - cfg.area_max)
    area_severity = clamp(max(area_below, area_above), 0.0, cfg.severity_cap)

    ar_severity = 0.0
    if target_ar_value is not None and target_ar_value > 0.0:
        ar_gap = max(0.0, abs(ar - float(target_ar_value)) - float(cfg.ar_hard_eps))
        ar_severity = clamp(ar_gap / max(float(cfg.ar_violation_scale), float(cfg.ar_hard_eps)), 0.0, cfg.severity_cap)

    face_severity = 1.0 if bool(cutoff.get("face_cut", False)) or ("face_cut" in reject_tags) else 0.0
    face_cut = face_severity > 0.0

    joint_score = safe_float(cutoff.get("joint_cutoff_score", 0.0), 0.0)
    joint_severe_count = safe_float(cutoff.get("joint_severe_count", 0.0), 0.0)
    joint_severity = clamp(max(joint_score, 0.65 * joint_severe_count), 0.0, cfg.severity_cap)

    head_top_severity = 0.0
    if bool(head_top.get("activated", False)):
        min_required = safe_float(head_top.get("min_required_y1", 0.0), 0.0)
        crop_y1 = safe_float(head_top.get("crop_y1", 0.0), 0.0)
        head_top_deficit = max(0.0, crop_y1 - min_required)
        head_top_severity = clamp(
            head_top_deficit / max(float(cfg.head_top_min_margin), 1e-6),
            0.0,
            cfg.severity_cap,
        )

    lookroom_severity = 0.0
    look_value = lookroom.get("value")
    target_range = lookroom.get("target_range")
    if isinstance(target_range, (list, tuple)) and len(target_range) >= 2 and look_value is not None:
        lmin = safe_float(target_range[0], 0.0)
        lmax = safe_float(target_range[1], 0.0)
        scale = max(float(cfg.lookroom_scale_floor), lmax - lmin)
        overflow = max(0.0, lmin - safe_float(look_value, 0.0), safe_float(look_value, 0.0) - lmax)
        lookroom_severity = clamp(overflow / scale, 0.0, cfg.severity_cap)
    elif bool(lookroom.get("pass", True)) is False:
        lookroom_severity = 1.0

    text_keep_ratio = safe_float(text.get("text_keep_ratio", 1.0), 1.0)
    cut_box_ratio = safe_float(text.get("cut_box_ratio", 0.0), 0.0)
    severe_cut_ratio = safe_float(text.get("severe_cut_ratio", 0.0), 0.0)
    text_severity = 0.0
    if bool(text.get("available", False)):
        text_severity = clamp(
            max(1.0 - text_keep_ratio, 0.75 * cut_box_ratio, 1.10 * severe_cut_ratio),
            0.0,
            float(cfg.text_soft_cap),
        )

    subject_border_severity = 1.0 if bool(flags.get("subject_touch_border", False)) else 0.0
    coverage = safe_float(
        (checklist.get("subject_coverage") or {}).get("value", candidate.get("subject_coverage", 0.0)),
        0.0,
    )
    has_structural_reject = bool({"area_violation", "ar_violation"} & reject_tags)
    joint_relaxed = bool(
        coverage >= float(cfg.relax_joint_head_top_min_coverage)
        and joint_severe_count <= float(cfg.relax_joint_max_severe_count)
        and (not face_cut)
        and (not has_structural_reject)
    )
    head_top_relaxed = joint_relaxed
    subject_border_relaxed = bool(
        coverage >= float(cfg.relax_subject_border_min_coverage)
        and (not face_cut)
        and joint_score <= float(cfg.relax_subject_border_max_joint_score)
    )

    lambda_joint = float(cfg.lambda_joint) * (float(cfg.joint_relax_lambda_scale) if joint_relaxed else 1.0)
    bonus_joint = float(cfg.bonus_joint_hard) * (float(cfg.joint_relax_bonus_scale) if joint_relaxed else 1.0)
    lambda_head_top = float(cfg.lambda_head_top) * (float(cfg.head_top_relax_lambda_scale) if head_top_relaxed else 1.0)
    bonus_head_top = float(cfg.bonus_head_top_hard) * (float(cfg.head_top_relax_bonus_scale) if head_top_relaxed else 1.0)
    lambda_subject_border = float(cfg.lambda_subject_border) * (
        float(cfg.subject_border_relax_lambda_scale) if subject_border_relaxed else 1.0
    )

    soft_components = {
        "area_violation": float(cfg.lambda_area) * area_severity,
        "ar_violation": float(cfg.lambda_ar) * ar_severity,
        "face_cut": float(cfg.lambda_face) * face_severity,
        "head_top_cut": lambda_head_top * head_top_severity,
        "joint_cutoff": lambda_joint * joint_severity,
        "lookroom_cut": float(cfg.lambda_lookroom) * lookroom_severity,
        "text_cutoff": float(cfg.lambda_text) * text_severity,
        "subject_border_touch": lambda_subject_border * subject_border_severity,
    }

    hard_components = {
        "area_violation": float(cfg.bonus_area_hard) if "area_violation" in reject_tags else 0.0,
        "ar_violation": float(cfg.bonus_ar_hard) if "ar_violation" in reject_tags else 0.0,
        "face_cut": float(cfg.bonus_face_hard) if "face_cut" in reject_tags else 0.0,
        "head_top_cut": bonus_head_top if "head_top_cut" in reject_tags else 0.0,
        "joint_cutoff": bonus_joint if "joint_cutoff" in reject_tags else 0.0,
        "lookroom_cut": float(cfg.bonus_lookroom_hard) if "lookroom_cut" in reject_tags else 0.0,
        "text_cutoff": float(cfg.bonus_text_hard) if "text_cutoff" in reject_tags else 0.0,
    }

    soft_total = sum(soft_components.values())
    hard_total = sum(hard_components.values())
    total = soft_total + hard_total
    return {
        "total": round(float(total), 9),
        "soft_total": round(float(soft_total), 9),
        "hard_total": round(float(hard_total), 9),
        "soft_components": {key: round(float(value), 9) for key, value in soft_components.items()},
        "hard_components": {key: round(float(value), 9) for key, value in hard_components.items()},
        "signals": {
            "area_severity": round(float(area_severity), 9),
            "ar_severity": round(float(ar_severity), 9),
            "face_severity": round(float(face_severity), 9),
            "head_top_severity": round(float(head_top_severity), 9),
            "joint_severity": round(float(joint_severity), 9),
            "lookroom_severity": round(float(lookroom_severity), 9),
            "text_severity": round(float(text_severity), 9),
            "subject_border_severity": round(float(subject_border_severity), 9),
            "coverage": round(float(coverage), 9),
            "joint_relaxed": bool(joint_relaxed),
            "head_top_relaxed": bool(head_top_relaxed),
            "subject_border_relaxed": bool(subject_border_relaxed),
        },
    }
