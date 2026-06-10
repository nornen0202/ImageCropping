from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence


ROUTE_FAMILY_V2 = ("scene", "person_single", "person_group", "pet_dogcat")
PERSON_SHOT_TYPE = ("face_headshot", "upper_half_body", "full_body")
PLACEMENT_INTENT = ("center", "rot")
CONTEXT_INTENT = ("tight_subject", "balanced", "environmental")
SCHEMA_VERSION = "routing_v2_simple_v8_nofood_petstrict_shotstrict"

DOG_CLASS_ID = 16
CAT_CLASS_ID = 15
DOGCAT_CLASS_IDS = {CAT_CLASS_ID, DOG_CLASS_ID}
ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77}
FOOD_CLASS_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
TABLEWARE_CLASS_IDS = {39, 40, 41, 42, 43, 44, 45}
DOGCAT_PRESENT_SCORE_MIN = 0.45
DOGCAT_STRONG_SCORE_MIN = 0.70
ANIMAL_POSE_SPECIES_CONF_MIN = 0.50
ANIMAL_POSE_HEAD_CONF_MIN = 0.45
ANIMAL_POSE_CONF_MIN = 0.30
FULL_BODY_ANKLE_CONF_MIN = 0.30
FULL_BODY_MAX_FACE_RATIO = 0.13

HIERARCHICAL_VOCABS = {
    "route_family_v2": list(ROUTE_FAMILY_V2),
    "person_shot_type": list(PERSON_SHOT_TYPE),
    "placement_intent": list(PLACEMENT_INTENT),
    "context_intent": list(CONTEXT_INTENT),
    "mode_feasible": [False, True],
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def box_area(box: Any) -> float:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = [safe_float(v) for v in box[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _routing(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    value = row.get("routing")
    return value if isinstance(value, dict) else {}


def _router_signals(row: dict[str, Any] | None) -> dict[str, Any]:
    routing = _routing(row)
    value = routing.get("router_signals")
    return value if isinstance(value, dict) else {}


def iter_c2_instances(row: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    if not isinstance(row, dict):
        return
    for key in ("c2_seg", "c2_det"):
        value = row.get(key)
        if isinstance(value, dict):
            items = value.get("instances")
        else:
            items = value
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield item


def iter_primary_c2_instances(row: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    """Yield one C2 stream to avoid double-counting duplicated det/seg rows."""
    if not isinstance(row, dict):
        return
    for key in ("c2_seg", "c2_det"):
        value = row.get(key)
        if isinstance(value, dict):
            items = value.get("instances")
        else:
            items = value
        if isinstance(items, list) and items:
            for item in items:
                if isinstance(item, dict):
                    yield item
            return


def dogcat_evidence(row: dict[str, Any] | None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    score_max = 0.0
    area_sum = 0.0
    animal_count = 0
    animal_area_sum = 0.0
    for inst in iter_primary_c2_instances(row):
        class_id = safe_int(inst.get("class_id"), -1)
        score = clamp(safe_float(inst.get("score"), 0.0))
        area = clamp(safe_float(inst.get("area_ratio"), 0.0))
        if class_id in ANIMAL_CLASS_IDS:
            animal_count += 1
            animal_area_sum += area
        if class_id == DOG_CLASS_ID:
            counts["dog"] += 1
        elif class_id == CAT_CLASS_ID:
            counts["cat"] += 1
        else:
            continue
        score_max = max(score_max, score)
        area_sum += area
    species = None
    if counts.get("dog", 0) and counts.get("cat", 0):
        species = "dogcat_mixed"
    elif counts.get("dog", 0):
        species = "dog"
    elif counts.get("cat", 0):
        species = "cat"
    return {
        "dogcat_count": int(sum(counts.values())),
        "dog_count": int(counts.get("dog", 0)),
        "cat_count": int(counts.get("cat", 0)),
        "animal_count": int(animal_count),
        "species": species,
        "dogcat_score_max": round(score_max, 6),
        "dogcat_area_sum": round(clamp(area_sum), 6),
        "animal_area_sum": round(clamp(animal_area_sum), 6),
        "dogcat_present": bool(sum(counts.values()) > 0 and score_max >= DOGCAT_PRESENT_SCORE_MIN),
    }


def food_evidence(row: dict[str, Any] | None) -> dict[str, Any]:
    food_count = 0
    tableware_count = 0
    food_score_max = 0.0
    tableware_score_max = 0.0
    food_area_sum = 0.0
    tableware_area_sum = 0.0
    for inst in iter_c2_instances(row):
        class_id = safe_int(inst.get("class_id"), -1)
        score = clamp(safe_float(inst.get("score"), 0.0))
        area = clamp(safe_float(inst.get("area_ratio"), 0.0))
        if class_id in FOOD_CLASS_IDS:
            food_count += 1
            food_score_max = max(food_score_max, score)
            food_area_sum += area
        elif class_id in TABLEWARE_CLASS_IDS:
            tableware_count += 1
            tableware_score_max = max(tableware_score_max, score)
            tableware_area_sum += area
    present = bool(
        (food_count > 0 and food_score_max >= 0.35)
        or (tableware_count > 0 and tableware_score_max >= 0.80)
    )
    return {
        "food_count": int(food_count),
        "tableware_count": int(tableware_count),
        "food_score_max": round(food_score_max, 6),
        "tableware_score_max": round(tableware_score_max, 6),
        "food_area_sum": round(clamp(food_area_sum), 6),
        "tableware_area_sum": round(clamp(tableware_area_sum), 6),
        "food_present": present,
    }


def _keypoint_visible(keypoints: Sequence[Sequence[Any]], indices: Sequence[int], conf_thr: float = 0.15) -> bool:
    for index in indices:
        if index >= len(keypoints):
            continue
        kp = keypoints[index]
        if isinstance(kp, (list, tuple)) and len(kp) >= 3 and safe_float(kp[2], 0.0) >= conf_thr:
            return True
    return False


def _keypoint_max_conf(keypoints: Sequence[Sequence[Any]], indices: Sequence[int]) -> float:
    values = []
    for index in indices:
        if index >= len(keypoints):
            continue
        kp = keypoints[index]
        if isinstance(kp, (list, tuple)) and len(kp) >= 3:
            values.append(safe_float(kp[2], 0.0))
    return max(values) if values else 0.0


def _keypoint_max_y(keypoints: Sequence[Sequence[Any]], indices: Sequence[int]) -> float:
    values = []
    for index in indices:
        if index >= len(keypoints):
            continue
        kp = keypoints[index]
        if isinstance(kp, (list, tuple)) and len(kp) >= 3 and safe_float(kp[2], 0.0) > 0.0:
            values.append(safe_float(kp[1], 0.0))
    return max(values) if values else 0.0


def person_pose_signals(row: dict[str, Any] | None) -> dict[str, Any]:
    poses = row.get("c3_pose") if isinstance(row, dict) else None
    if not isinstance(poses, list) or not poses:
        return {
            "pose_available": False,
            "face_available": False,
            "face_ratio": 0.0,
            "has_eye": False,
            "has_shoulder": False,
            "has_hip": False,
            "has_knee": False,
            "has_ankle": False,
            "visible_group_count": 0,
            "pose_confidence": 0.0,
            "hip_conf_max": 0.0,
            "knee_conf_max": 0.0,
            "ankle_conf_max": 0.0,
            "ankle_y_norm": 0.0,
            "pose_bbox_height_ratio": 0.0,
            "lower_body_reliable": False,
        }
    best = None
    best_key = (-1.0, -1.0)
    for pose in poses:
        if not isinstance(pose, dict):
            continue
        bbox = pose.get("bbox")
        area = box_area(bbox)
        score = safe_float(pose.get("score"), 0.0)
        key = (score, area)
        if key > best_key:
            best = pose
            best_key = key
    if not isinstance(best, dict):
        return person_pose_signals(None)
    keypoints = best.get("keypoints")
    keypoints = keypoints if isinstance(keypoints, list) else []
    face = best.get("face") if isinstance(best.get("face"), dict) else {}
    face_box = face.get("bbox")
    bbox = best.get("bbox")
    face_ratio = box_area(face_box) / max(1.0e-8, box_area(bbox))
    image_h = safe_float((row or {}).get("height"), 0.0)
    image_w = safe_float((row or {}).get("width"), 0.0)
    if image_h <= 0.0 or image_w <= 0.0:
        size = (row or {}).get("size") if isinstance((row or {}).get("size"), dict) else {}
        image_h = safe_float(size.get("height"), image_h)
        image_w = safe_float(size.get("width"), image_w)
    has_eye = _keypoint_visible(keypoints, (1, 2))
    has_shoulder = _keypoint_visible(keypoints, (5, 6))
    has_hip = _keypoint_visible(keypoints, (11, 12))
    has_knee = _keypoint_visible(keypoints, (13, 14))
    has_ankle = _keypoint_visible(keypoints, (15, 16))
    hip_conf = _keypoint_max_conf(keypoints, (11, 12))
    knee_conf = _keypoint_max_conf(keypoints, (13, 14))
    ankle_conf = _keypoint_max_conf(keypoints, (15, 16))
    ankle_y = _keypoint_max_y(keypoints, (15, 16))
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in (bbox or [0.0, 0.0, 0.0, 0.0])[:4]]
    bbox_h = max(0.0, y2 - y1)
    pose_bbox_height_ratio = bbox_h / max(1.0, image_h) if image_h > 0.0 else 0.0
    ankle_y_norm = ankle_y / max(1.0, image_h) if image_h > 0.0 else 0.0
    lower_body_reliable = bool(
        ankle_conf >= FULL_BODY_ANKLE_CONF_MIN
        and face_ratio <= FULL_BODY_MAX_FACE_RATIO
        and (ankle_y_norm >= 0.62 or image_h <= 0.0)
    )
    visible_group_count = sum(int(v) for v in (has_eye, has_shoulder, has_hip, has_knee, has_ankle))
    return {
        "pose_available": True,
        "face_available": bool(face_box and box_area(face_box) > 0.0),
        "face_ratio": round(clamp(face_ratio), 6),
        "has_eye": bool(has_eye),
        "has_shoulder": bool(has_shoulder),
        "has_hip": bool(has_hip),
        "has_knee": bool(has_knee),
        "has_ankle": bool(has_ankle),
        "visible_group_count": int(visible_group_count),
        "pose_confidence": round(clamp(best_key[0]), 6),
        "hip_conf_max": round(clamp(hip_conf), 6),
        "knee_conf_max": round(clamp(knee_conf), 6),
        "ankle_conf_max": round(clamp(ankle_conf), 6),
        "ankle_y_norm": round(clamp(ankle_y_norm), 6),
        "pose_bbox_height_ratio": round(clamp(pose_bbox_height_ratio), 6),
        "lower_body_reliable": lower_body_reliable,
    }


def normalize_person_shot_type(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"face", "head", "headshot", "face_headshot"}:
        return "face_headshot"
    if text in {"upper", "upper_body", "bust", "chest", "half", "half_body", "waist", "upper_half_body", "upperhalf"}:
        return "upper_half_body"
    if text in {"three_quarter", "3q", "full", "full_body"}:
        return "full_body"
    return None


def environmental_portrait_evidence(row: dict[str, Any] | None) -> dict[str, Any]:
    routing = _routing(row)
    signals = _router_signals(row)
    pose = person_pose_signals(row)
    person_area = clamp(safe_float(signals.get("person_union_area_ratio"), 0.0))
    primary_area = clamp(safe_float(signals.get("c2_primary_area_ratio"), 0.0))
    scene_score = clamp(safe_float(signals.get("scene_score"), 0.0))
    foreground_mass = clamp(safe_float(signals.get("foreground_mass_ratio"), safe_float(signals.get("saliency_foreground_area_ratio"), 0.0)))
    entropy = clamp(safe_float(signals.get("saliency_entropy_norm"), 0.0))
    blank_ratio = clamp(safe_float(signals.get("blank_ratio_est"), safe_float((routing.get("copyspace") or {}).get("blank_ratio_est") if isinstance(routing.get("copyspace"), dict) else 0.0)))
    copyspace_signal = bool((routing.get("copyspace") or {}).get("gate_passed")) if isinstance(routing.get("copyspace"), dict) else False
    num_person = safe_float(signals.get("num_person"), 0.0)
    context_score = 0.0
    context_score += 0.34 * scene_score
    context_score += 0.18 * entropy
    context_score += 0.16 * clamp(1.0 - foreground_mass)
    context_score += 0.14 * blank_ratio
    context_score += 0.10 if copyspace_signal else 0.0
    context_score += 0.08 if pose["pose_available"] and person_area <= 0.28 else 0.0
    is_single_person = num_person <= 1.5 or str(routing.get("subject_mode") or "") == "portrait_single"
    area_ok = 0.025 <= max(person_area, primary_area) <= 0.30
    context_ok = context_score >= 0.50 or (scene_score >= 0.68 and max(person_area, primary_area) <= 0.36)
    environmental = bool(is_single_person and area_ok and context_ok)
    return {
        "environmental_portrait": environmental,
        "environmental_score": round(clamp(context_score), 6),
        "person_area": round(person_area, 6),
        "primary_area": round(primary_area, 6),
        "scene_score": round(scene_score, 6),
        "foreground_mass": round(foreground_mass, 6),
        "saliency_entropy_norm": round(entropy, 6),
        "blank_ratio_est": round(blank_ratio, 6),
        "copyspace_signal": bool(copyspace_signal),
        "area_ok": bool(area_ok),
        "context_ok": bool(context_ok),
    }


def _pose_based_person_shot_type(row: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    signals = person_pose_signals(row)
    if signals["lower_body_reliable"]:
        return "full_body", {"source": "pose_reliable_full_body", **signals}
    if signals["has_ankle"] or signals["has_knee"] or signals["has_hip"]:
        return "upper_half_body", {"source": "pose_partial_lower_body_upper_half_body", **signals}
    if signals["has_shoulder"]:
        return "upper_half_body", {"source": "pose_shoulder", **signals}
    if signals["face_available"] or signals["face_ratio"] >= 0.10:
        return "face_headshot", {"source": "pose_face", **signals}
    return "upper_half_body", {"source": "portrait_fallback_upper_half_body", **signals}


def infer_person_shot_type(
    row: dict[str, Any] | None,
    *,
    mode_name: str = "",
    attributes: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    attrs = attributes if isinstance(attributes, dict) else {}
    score_components = attrs.get("score_components") if isinstance(attrs.get("score_components"), dict) else {}
    if str(mode_name) == "face":
        return "face_headshot", {"source": "mode_face"}
    for value in (
        attrs.get("person_shot_type"),
        score_components.get("portrait_shot_type"),
        _routing(row).get("shot_type"),
    ):
        normalized = normalize_person_shot_type(value)
        if normalized:
            if normalized == "full_body":
                pose_signals = person_pose_signals(row)
                if not pose_signals.get("lower_body_reliable"):
                    return "upper_half_body", {
                        "source": "existing_full_body_demoted_weak_lower_body",
                        "raw": str(value),
                        **pose_signals,
                    }
                return normalized, {
                    "source": "existing_shot_type_pose_validated",
                    "raw": str(value),
                    **pose_signals,
                }
            return normalized, {"source": "existing_shot_type", "raw": str(value)}
    env = environmental_portrait_evidence(row)
    shot, info = _pose_based_person_shot_type(row)
    if env["environmental_portrait"]:
        info = {
            **info,
            "source": f"environmental_context_{info.get('source', 'pose')}",
            "environmental_context": True,
            **env,
        }
    return shot, info


def placement_from_mode(mode_name: str, fallback: str = "center") -> str:
    text = str(mode_name or "").strip().lower()
    if text.endswith("_rot") or "_rot_" in text:
        return "rot"
    return "center" if fallback not in PLACEMENT_INTENT else fallback


def context_from_family_and_shot(route_family: str, person_shot_type: str | None, placement: str) -> str:
    if route_family == "scene":
        return "balanced"
    if route_family == "person_group":
        return "balanced"
    if route_family == "person_single":
        return "tight_subject" if person_shot_type == "face_headshot" and placement == "center" else "balanced"
    if route_family == "pet_dogcat":
        return "tight_subject" if placement == "center" else "balanced"
    return "balanced"


def _vote_distribution(vote: dict[str, Any], key: str) -> tuple[str | None, float, float, float]:
    values = vote.get(f"{key}_votes") if isinstance(vote, dict) else None
    if not isinstance(values, dict) or not values:
        value = vote.get(key) if isinstance(vote, dict) else None
        return (str(value), 1.0, 1.0, 1.0) if value is not None else (None, 0.0, 0.0, 0.0)
    parsed: list[tuple[str, float]] = []
    for value, weight in values.items():
        parsed.append((str(value), max(0.0, safe_float(weight, 0.0))))
    parsed.sort(key=lambda item: item[1], reverse=True)
    if not parsed:
        return None, 0.0, 0.0, 0.0
    top, top_weight = parsed[0]
    total = sum(weight for _, weight in parsed)
    share = top_weight / total if total > 0.0 else 0.0
    return top, round(top_weight, 6), round(total, 6), round(share, 6)


def _strong_vote(vote: dict[str, Any], key: str, *, min_weight: float = 0.75, min_share: float = 0.60) -> tuple[str | None, float, float]:
    value, weight, _total, share = _vote_distribution(vote, key)
    if value is None:
        return None, 0.0, 0.0
    if weight >= min_weight and share >= min_share:
        return value, weight, share
    return None, weight, share


def _legacy_subject_mode(row: dict[str, Any] | None, attrs: dict[str, Any] | None = None) -> str:
    attrs = attrs if isinstance(attrs, dict) else {}
    routing = _routing(row)
    return str(
        attrs.get("route_mode")
        or attrs.get("source_route_mode")
        or routing.get("subject_mode")
        or ""
    ).strip()


def _route_family_from_mode_name(mode_name: str, feature_row: dict[str, Any] | None, attrs: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    mode = str(mode_name or "").strip()
    if mode == "landscape":
        return "scene", ["mode_landscape"]
    if mode == "face" or mode.startswith("single_person"):
        return "person_single", [f"mode_{mode}"]
    if mode.startswith("group"):
        return "person_group", [f"mode_{mode}"]
    if mode.startswith("object"):
        pet = dogcat_evidence(feature_row)
        food = food_evidence(feature_row)
        object_class_id = safe_int(attrs.get("object_class_id"), -1)
        if object_class_id in DOGCAT_CLASS_IDS or pet["dogcat_present"]:
            reasons.append("object_mode_dogcat_evidence")
            return "pet_dogcat", reasons
        if object_class_id in FOOD_CLASS_IDS or object_class_id in TABLEWARE_CLASS_IDS or food["food_present"]:
            reasons.append("object_mode_food_evidence_folded_to_scene_no_food_route")
            return "scene", reasons
        reasons.append("object_mode_folded_to_scene_simple_taxonomy")
        return "scene", reasons
    return "scene", ["unknown_mode_folded_to_scene"]


def derive_query_routing_v2_simple(
    *,
    mode_name: str,
    feature_row: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    positive_exists: bool = True,
    positive_score: float | None = None,
    no_positive_reason: str = "",
    pet_pose_head_available: bool | None = None,
) -> dict[str, Any]:
    attrs = attributes if isinstance(attributes, dict) else {}
    route_family, reasons = _route_family_from_mode_name(mode_name, feature_row, attrs)
    person_shot_type = None
    shot_info: dict[str, Any] = {}
    if route_family == "person_single":
        person_shot_type, shot_info = infer_person_shot_type(feature_row, mode_name=mode_name, attributes=attrs)
        reasons.append(f"shot_{person_shot_type}:{shot_info.get('source', '')}")
    placement = placement_from_mode(mode_name)
    context = context_from_family_and_shot(route_family, person_shot_type, placement)
    if route_family == "person_single" and bool(shot_info.get("environmental_context")):
        context = "environmental"
    if route_family == "pet_dogcat" and pet_pose_head_available is False:
        positive_exists = False
        no_positive_reason = f"{no_positive_reason};pet_pose_head_missing".strip(";")
        reasons.append("pet_pose_head_required_missing")
    routing_conf = _confidence(
        feature_row=feature_row,
        route_family=route_family,
        positive_exists=positive_exists,
        positive_score=positive_score,
        reasons=reasons,
        no_positive_reason=no_positive_reason,
        shot_info=shot_info,
    )
    support_policy = "not_applicable"
    if route_family == "person_single":
        support_policy = "active" if person_shot_type == "full_body" else "inactive"
    elif route_family == "person_group":
        support_policy = "weak"
    elif route_family == "pet_dogcat":
        support_policy = "animal_pose_required"
    return {
        "schema_version": SCHEMA_VERSION,
        "route_family_v2": route_family,
        "person_shot_type": person_shot_type,
        "placement_intent": placement,
        "context_intent": context,
        "mode_feasible": bool(positive_exists),
        "routing_confidence": routing_conf,
        "support_grounding_policy": support_policy,
        "teacher_reliability": "high" if routing_conf >= 0.75 else "medium" if routing_conf >= 0.60 else "low",
        "reasons": reasons,
        "no_positive_reason": str(no_positive_reason or ""),
        "environmental_portrait_evidence": environmental_portrait_evidence(feature_row) if route_family == "person_single" else None,
        "food_evidence": food_evidence(feature_row) if route_family == "food" else None,
        "pet_pose_head_available": pet_pose_head_available if route_family == "pet_dogcat" else None,
    }


def derive_image_routing_v2_simple(
    feature_row: dict[str, Any] | None,
    *,
    positive_vote_summary: dict[str, Any] | None = None,
    pet_pose_head_available: bool | None = None,
) -> dict[str, Any]:
    routing = _routing(feature_row)
    legacy_mode = str(routing.get("subject_mode") or "").strip()
    pet = dogcat_evidence(feature_row)
    food = food_evidence(feature_row)
    reasons: list[str] = []
    signals = _router_signals(feature_row)
    person_area = clamp(safe_float(signals.get("person_union_area_ratio"), 0.0))
    semantic_primary_family = str(
        signals.get("semantic_primary_family")
        or routing.get("subject_family")
        or routing.get("primary_subject_type")
        or ""
    ).strip().lower()
    pet_score = clamp(safe_float(pet.get("dogcat_score_max"), 0.0))
    pet_area = clamp(safe_float(pet.get("dogcat_area_sum"), 0.0))
    pet_pose_ok = pet_pose_head_available is not False
    pet_pose_strict_ok = pet_pose_head_available is True
    pet_detector_strong = bool(pet["dogcat_present"] and pet_score >= DOGCAT_STRONG_SCORE_MIN and pet_area >= 0.05)
    pet_detector_dominant = bool(
        pet_detector_strong
        and (
            semantic_primary_family == "animal"
            or (person_area <= 0.28 and pet_area >= max(0.10, 0.85 * person_area))
        )
    )
    pet_over_legacy_portrait = bool(
        legacy_mode == "portrait_single"
        and pet_pose_strict_ok
        and pet_detector_dominant
    )
    food_over_legacy_portrait_fold_to_scene = bool(
        legacy_mode == "portrait_single"
        and food["food_present"]
        and str(routing.get("shot_type") or "").strip().lower() in {"", "unknown"}
        and person_area <= 0.32
    )
    if pet_over_legacy_portrait:
        route_family = "pet_dogcat"
        reasons.append("dogcat_evidence_over_legacy_portrait")
    elif food_over_legacy_portrait_fold_to_scene:
        route_family = "scene"
        reasons.append("food_evidence_over_legacy_portrait_folded_to_scene_no_food_route")
    elif legacy_mode == "portrait_single":
        route_family = "person_single"
        reasons.append("legacy_portrait_single")
    elif legacy_mode == "portrait_group":
        route_family = "person_group"
        reasons.append("legacy_portrait_group")
    elif pet["dogcat_present"] and pet_pose_ok and legacy_mode in {"object_single", "object_multi", "scene_general", "other_ambiguous", ""}:
        route_family = "pet_dogcat"
        reasons.append("dogcat_evidence")
    elif pet["dogcat_present"] and pet_pose_head_available is False and legacy_mode in {"object_single", "object_multi", "scene_general", "other_ambiguous", ""}:
        route_family = "scene"
        reasons.append("dogcat_evidence_pose_head_missing_folded_to_scene")
    elif food["food_present"] and legacy_mode in {"object_single", "object_multi", "scene_general", "other_ambiguous", ""}:
        route_family = "scene"
        reasons.append("food_evidence_folded_to_scene_no_food_route")
    else:
        route_family = "scene"
        if legacy_mode.startswith("object"):
            reasons.append("legacy_object_folded_to_scene_simple_taxonomy")
        elif legacy_mode:
            reasons.append(f"legacy_{legacy_mode}")
        else:
            reasons.append("legacy_missing_scene_fallback")

    person_shot_type = None
    shot_info: dict[str, Any] = {}
    if route_family == "person_single":
        person_shot_type, shot_info = infer_person_shot_type(feature_row)
        reasons.append(f"shot_{person_shot_type}:{shot_info.get('source', '')}")

    vote = positive_vote_summary if isinstance(positive_vote_summary, dict) else {}
    placement_vote, placement_weight, placement_share = _strong_vote(vote, "placement_intent", min_weight=0.75, min_share=0.56)
    vote_family, vote_family_weight, _vote_family_total, vote_family_share = _vote_distribution(vote, "route_family_v2")
    if vote_family in ROUTE_FAMILY_V2 and vote_family != route_family:
        reasons.append(f"positive_vote_route_observed_only:{vote_family}:share={vote_family_share:.3f}")
    elif vote_family in ROUTE_FAMILY_V2:
        reasons.append(f"positive_vote_route_agreement_observed_only:share={vote_family_share:.3f}")
    placement = "center"
    if route_family != "person_single":
        person_shot_type = None
    context = context_from_family_and_shot(route_family, person_shot_type, placement)
    if route_family == "person_single" and bool(shot_info.get("environmental_context")):
        context = "environmental"
    routing_conf = _confidence(
        feature_row=feature_row,
        route_family=route_family,
        positive_exists=True,
        positive_score=None,
        reasons=reasons,
        no_positive_reason="",
        shot_info=shot_info,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "route_family_v2": route_family,
        "person_shot_type": person_shot_type,
        "placement_intent": placement,
        "context_intent": context,
        "mode_feasible": True,
        "routing_confidence": routing_conf,
        "support_grounding_policy": "active" if person_shot_type == "full_body" else "inactive" if route_family == "person_single" else "animal_pose_required" if route_family == "pet_dogcat" else "not_applicable",
        "teacher_reliability": "high" if routing_conf >= 0.75 else "medium" if routing_conf >= 0.60 else "low",
        "legacy_subject_mode": legacy_mode,
        "positive_vote_route_family_share": vote_family_share,
        "positive_vote_route_family_weight": vote_family_weight,
        "positive_vote_placement_share": placement_share,
        "positive_vote_placement_weight": placement_weight,
        "dogcat_evidence": pet,
        "food_evidence": food,
        "pet_pose_head_available": pet_pose_head_available if route_family == "pet_dogcat" or pet["dogcat_present"] else None,
        "person_pose_signals": person_pose_signals(feature_row),
        "environmental_portrait_evidence": environmental_portrait_evidence(feature_row) if route_family == "person_single" else None,
        "reasons": reasons,
    }


def _confidence(
    *,
    feature_row: dict[str, Any] | None,
    route_family: str,
    positive_exists: bool,
    positive_score: float | None,
    reasons: Sequence[str],
    no_positive_reason: str,
    shot_info: dict[str, Any] | None,
) -> float:
    routing = _routing(feature_row)
    signals = _router_signals(feature_row)
    conf = safe_float(routing.get("subject_mode_conf"), 0.58)
    if route_family == "pet_dogcat":
        pet = dogcat_evidence(feature_row)
        conf = max(conf, 0.68 + 0.18 * clamp(pet["dogcat_score_max"]) + 0.06 * clamp(pet["dogcat_area_sum"] * 4.0))
    if route_family.startswith("person"):
        pose = person_pose_signals(feature_row)
        if pose["pose_available"]:
            conf += 0.06
        if pose["visible_group_count"] >= 3:
            conf += 0.04
        if route_family == "person_single" and (shot_info or {}).get("source", "").startswith("pose"):
            conf += 0.05
        if route_family == "person_single" and bool((shot_info or {}).get("environmental_context")):
            conf += 0.06 + 0.08 * clamp(safe_float((shot_info or {}).get("environmental_score"), 0.0))
    if route_family == "scene":
        scene_score = safe_float(signals.get("scene_score"), 0.0)
        if scene_score >= 0.65:
            conf += 0.05
    if positive_score is not None:
        conf = 0.72 * conf + 0.28 * clamp(float(positive_score))
    if not positive_exists:
        conf -= 0.18
    if no_positive_reason:
        conf -= 0.06
    if any("folded_to_scene" in r for r in reasons):
        conf -= 0.18
    if bool(routing.get("subject_mode_conflict", False)):
        conf -= 0.12
    return round(clamp(conf, 0.05, 0.99), 6)


def flat_route_class_key(routing_v2: dict[str, Any]) -> str:
    family = str(routing_v2.get("route_family_v2") or "scene")
    if family not in ROUTE_FAMILY_V2:
        family = "scene"
    shot = str(routing_v2.get("person_shot_type") or "na")
    if family == "person_single" and shot not in PERSON_SHOT_TYPE:
        shot = "upper_half_body"
    elif family != "person_single":
        shot = "na"
    placement = str(routing_v2.get("placement_intent") or "center")
    context = str(routing_v2.get("context_intent") or "balanced")
    feasible = "feasible" if bool(routing_v2.get("mode_feasible", True)) else "infeasible"
    return f"{family}|{shot}|{placement}|{context}|{feasible}"


def hierarchical_target_ids(routing_v2: dict[str, Any]) -> dict[str, int]:
    family = str(routing_v2.get("route_family_v2") or "scene")
    shot = routing_v2.get("person_shot_type")
    placement = str(routing_v2.get("placement_intent") or "center")
    context = str(routing_v2.get("context_intent") or "balanced")
    feasible = bool(routing_v2.get("mode_feasible", True))
    return {
        "route_family_v2_id": ROUTE_FAMILY_V2.index(family) if family in ROUTE_FAMILY_V2 else 0,
        "person_shot_type_id": PERSON_SHOT_TYPE.index(str(shot)) if shot in PERSON_SHOT_TYPE else -1,
        "placement_intent_id": PLACEMENT_INTENT.index(placement) if placement in PLACEMENT_INTENT else 0,
        "context_intent_id": CONTEXT_INTENT.index(context) if context in CONTEXT_INTENT else 1,
        "mode_feasible_id": 1 if feasible else 0,
    }


def probe_features_from_feature_row(row: dict[str, Any] | None) -> dict[str, float]:
    routing = _routing(row)
    signals = _router_signals(row)
    pet = dogcat_evidence(row)
    food = food_evidence(row)
    pose = person_pose_signals(row)
    legacy_mode = str(routing.get("subject_mode") or "")
    features: dict[str, float] = {
        "subject_mode_conf": safe_float(routing.get("subject_mode_conf"), 0.0),
        "subject_mode_conflict": float(bool(routing.get("subject_mode_conflict", False))),
        "num_person": safe_float(signals.get("num_person"), 0.0),
        "num_person_c2": safe_float(signals.get("num_person_c2"), 0.0),
        "person_union_area_ratio": safe_float(signals.get("person_union_area_ratio"), 0.0),
        "c2_num_instances": safe_float(signals.get("c2_num_instances"), 0.0),
        "c2_primary_area_ratio": safe_float(signals.get("c2_primary_area_ratio"), 0.0),
        "foreground_mass_ratio": safe_float(signals.get("foreground_mass_ratio"), 0.0),
        "nonperson_object_count": safe_float(signals.get("nonperson_object_count"), 0.0),
        "nonperson_object_union_area_ratio": safe_float(signals.get("nonperson_object_union_area_ratio"), 0.0),
        "scene_score": safe_float(signals.get("scene_score"), 0.0),
        "saliency_foreground_area_ratio": safe_float(signals.get("saliency_foreground_area_ratio"), 0.0),
        "saliency_dominance_score": safe_float(signals.get("saliency_dominance_score"), 0.0),
        "saliency_top2_mass_ratio": safe_float(signals.get("saliency_top2_mass_ratio"), 0.0),
        "saliency_entropy_norm": safe_float(signals.get("saliency_entropy_norm"), 0.0),
        "dogcat_count": float(pet["dogcat_count"]),
        "dog_count": float(pet["dog_count"]),
        "cat_count": float(pet["cat_count"]),
        "dogcat_score_max": float(pet["dogcat_score_max"]),
        "dogcat_area_sum": float(pet["dogcat_area_sum"]),
        "animal_count": float(pet["animal_count"]),
        "animal_area_sum": float(pet["animal_area_sum"]),
        "food_count": float(food["food_count"]),
        "tableware_count": float(food["tableware_count"]),
        "food_score_max": float(food["food_score_max"]),
        "tableware_score_max": float(food["tableware_score_max"]),
        "food_area_sum": float(food["food_area_sum"]),
        "tableware_area_sum": float(food["tableware_area_sum"]),
        "food_present": float(bool(food["food_present"])),
        "pose_available": float(bool(pose["pose_available"])),
        "face_available": float(bool(pose["face_available"])),
        "face_ratio": float(pose["face_ratio"]),
        "has_eye": float(bool(pose["has_eye"])),
        "has_shoulder": float(bool(pose["has_shoulder"])),
        "has_hip": float(bool(pose["has_hip"])),
        "has_knee": float(bool(pose["has_knee"])),
        "has_ankle": float(bool(pose["has_ankle"])),
        "visible_group_count": float(pose["visible_group_count"]),
        "pose_confidence": float(pose["pose_confidence"]),
        "hip_conf_max": float(pose.get("hip_conf_max", 0.0)),
        "knee_conf_max": float(pose.get("knee_conf_max", 0.0)),
        "ankle_conf_max": float(pose.get("ankle_conf_max", 0.0)),
        "ankle_y_norm": float(pose.get("ankle_y_norm", 0.0)),
        "pose_bbox_height_ratio": float(pose.get("pose_bbox_height_ratio", 0.0)),
        "lower_body_reliable": float(bool(pose.get("lower_body_reliable", False))),
    }
    env = environmental_portrait_evidence(row)
    for key, value in env.items():
        if isinstance(value, bool):
            features[f"environmental_{key}"] = float(value)
        elif isinstance(value, (int, float)):
            features[f"environmental_{key}"] = float(value)
    for mode in ("portrait_single", "portrait_group", "object_single", "object_multi", "scene_general", "background_texture_copyspace", "other_ambiguous"):
        features[f"legacy_mode_{mode}"] = float(legacy_mode == mode)
    return features
