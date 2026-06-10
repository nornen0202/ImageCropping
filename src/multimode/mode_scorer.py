from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from portrait_composition import portrait_place_components
from subject_region import box_area, box_center, clip_box01, norm_box_xyxy

from .candidate_bank import parse_target_ar
from .mode_catalog import MODE_SPEC_BY_NAME, is_object_mode, is_person_mode, mode_alignment_bonus
from .query_builder import ModeQuery


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _recall(box: Sequence[float], crop: Sequence[float]) -> float:
    denom = max(1e-8, box_area(box))
    return _intersection_area(box, crop) / denom


def _gaussian_quality(value: float, target: float, sigma: float) -> float:
    sigma = max(1e-6, float(sigma))
    return float(math.exp(-((float(value) - float(target)) ** 2) / (2.0 * sigma * sigma)))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _teacher_score_quality(score: Optional[float]) -> float:
    if score is None:
        return 0.0
    # GAIC/CGS teacher scores in the current SSTK candidate pack are roughly
    # [-1.5, 4.5]. Map them to a bounded quality so landscape tau remains stable.
    return _clamp((float(score) + 1.5) / 6.0)


def _candidate_teacher_scores(candidate: Dict[str, Any], *, scope: str) -> List[float]:
    scope_norm = str(scope or "all").strip().lower()
    scores: List[float] = []
    provenance = candidate.get("teacher_provenance")
    if isinstance(provenance, list):
        for item in provenance:
            if not isinstance(item, dict):
                continue
            teacher_id = str(item.get("teacher_id") or "").strip().lower()
            if scope_norm == "gaic" and teacher_id != "gaic":
                continue
            score = _finite_float_or_none(item.get("teacher_score"))
            if score is not None:
                scores.append(score)
    direct_score = _finite_float_or_none(candidate.get("teacher_score"))
    if direct_score is not None:
        source = str(candidate.get("source") or "").strip().lower()
        teacher_id = str(candidate.get("teacher_id") or "").strip().lower()
        if scope_norm != "gaic" or source == "teacher:gaic" or teacher_id == "gaic":
            scores.append(direct_score)
    return scores


def _candidate_teacher_score(candidate: Dict[str, Any], *, scope: str) -> Optional[float]:
    scores = _candidate_teacher_scores(candidate, scope=scope)
    if not scores:
        return None
    return max(scores)


def _relative_anchor_position(crop: Sequence[float], anchor_box: Sequence[float]) -> Tuple[float, float]:
    cx, cy = box_center(anchor_box)
    x1, y1, x2, y2 = [float(v) for v in crop]
    crop_w = max(1e-8, x2 - x1)
    crop_h = max(1e-8, y2 - y1)
    return _clamp((cx - x1) / crop_w), _clamp((cy - y1) / crop_h)


def _placement_quality(mode_name: str, crop: Sequence[float], anchor_box: Sequence[float]) -> float:
    rx, ry = _relative_anchor_position(crop, anchor_box)
    sigma = 0.18
    if mode_name.endswith("_rot"):
        sigma = 0.12
        left = math.exp(-(((rx - (1.0 / 3.0)) ** 2) + ((ry - 0.5) ** 2)) / (2.0 * sigma * sigma))
        right = math.exp(-(((rx - (2.0 / 3.0)) ** 2) + ((ry - 0.5) ** 2)) / (2.0 * sigma * sigma))
        return float(max(left, right))
    return float(math.exp(-(((rx - 0.5) ** 2) + ((ry - 0.5) ** 2)) / (2.0 * sigma * sigma)))


def _rot_thirds_deviation(crop: Sequence[float], anchor_box: Sequence[float]) -> float:
    rx, _ = _relative_anchor_position(crop, anchor_box)
    return min(abs(rx - (1.0 / 3.0)), abs(rx - (2.0 / 3.0)))


def _context_quality(subject_ratio: float, target: float, sigma: float) -> Tuple[float, float]:
    context_value = _clamp(1.0 - float(subject_ratio))
    return _gaussian_quality(context_value, target, sigma), context_value


def _axis_occupancy(query: ModeQuery, crop: Sequence[float]) -> Tuple[float, float, float, float]:
    subj = query.anchor_bbox_norm_xyxy
    subj_w = max(1e-8, float(subj[2]) - float(subj[0]))
    subj_h = max(1e-8, float(subj[3]) - float(subj[1]))
    crop_w = max(1e-8, float(crop[2]) - float(crop[0]))
    crop_h = max(1e-8, float(crop[3]) - float(crop[1]))
    occ_x = subj_w / crop_w
    occ_y = subj_h / crop_h
    occ_dom = max(occ_x, occ_y)
    occ_min = min(occ_x, occ_y)
    return occ_x, occ_y, occ_dom, occ_min


def _subject_margins(query: ModeQuery, crop: Sequence[float]) -> Tuple[float, float, float, float, float]:
    subj = query.anchor_bbox_norm_xyxy
    crop_w = max(1e-8, float(crop[2]) - float(crop[0]))
    crop_h = max(1e-8, float(crop[3]) - float(crop[1]))
    local_left = (float(subj[0]) - float(crop[0])) / crop_w
    local_top = (float(subj[1]) - float(crop[1])) / crop_h
    local_right = (float(crop[2]) - float(subj[2])) / crop_w
    local_bottom = (float(crop[3]) - float(subj[3])) / crop_h
    outside = (
        max(0.0, -local_left)
        + max(0.0, -local_top)
        + max(0.0, -local_right)
        + max(0.0, -local_bottom)
    )
    return local_left, local_right, local_top, local_bottom, outside


def _source_edge_margin_inherent(
    query: ModeQuery,
    crop: Sequence[float],
    *,
    edge: str,
    core_recall: float,
    group_recall: float,
    face_recall: float,
) -> bool:
    crop_box = [float(v) for v in crop]
    anchor = [float(v) for v in query.anchor_bbox_norm_xyxy]
    core = [float(v) for v in query.core_bbox_norm_xyxy]
    envelope = [float(v) for v in query.envelope_bbox_norm_xyxy]
    if query.mode_name.startswith("group"):
        if group_recall < 0.90 or face_recall < 0.88:
            return False
    elif core_recall < 0.95 or face_recall < 0.90:
        return False
    if edge == "bottom":
        subject_edge = max(anchor[3], core[3], envelope[3])
        return crop_box[3] >= 0.985 and subject_edge >= 0.975
    if edge == "left":
        subject_edge = min(anchor[0], core[0], envelope[0])
        return crop_box[0] <= 0.015 and subject_edge <= 0.025
    if edge == "right":
        subject_edge = max(anchor[2], core[2], envelope[2])
        return crop_box[2] >= 0.985 and subject_edge >= 0.975
    return False


def _source_side_margin_relaxed(
    query: ModeQuery,
    crop: Sequence[float],
    *,
    margin_left: float,
    margin_right: float,
    threshold: float,
    core_recall: float,
    group_recall: float,
    face_recall: float,
) -> bool:
    bad_edges: List[str] = []
    if margin_left < float(threshold):
        bad_edges.append("left")
    if margin_right < float(threshold):
        bad_edges.append("right")
    if not bad_edges:
        return False
    return all(
        _source_edge_margin_inherent(
            query,
            crop,
            edge=edge,
            core_recall=core_recall,
            group_recall=group_recall,
            face_recall=face_recall,
        )
        for edge in bad_edges
    )


def _source_bottom_margin_relaxed(
    query: ModeQuery,
    crop: Sequence[float],
    *,
    core_recall: float,
    group_recall: float,
    face_recall: float,
) -> bool:
    return _source_edge_margin_inherent(
        query,
        crop,
        edge="bottom",
        core_recall=core_recall,
        group_recall=group_recall,
        face_recall=face_recall,
    )


def _group_center_axis_relaxed(
    query: ModeQuery,
    crop: Sequence[float],
    *,
    bottom_margin_relaxed: bool,
    occ_x: float,
    occ_y: float,
    group_recall: float,
    face_recall: float,
    joint_cut_score: float,
) -> bool:
    if query.mode_name != "group_center" or not bottom_margin_relaxed:
        return False
    crop_w = max(1e-8, float(crop[2]) - float(crop[0]))
    crop_h = max(1e-8, float(crop[3]) - float(crop[1]))
    return (
        crop_w >= 0.70
        and crop_h >= 0.70
        and occ_x <= 0.90
        and occ_y <= 0.96
        and group_recall >= 0.92
        and face_recall >= 0.90
        and joint_cut_score <= 0.40
    )


def _single_center_source_axis_relaxed(
    query: ModeQuery,
    *,
    bottom_margin_relaxed: bool,
    core_recall: float,
    face_recall: float,
    subject_ratio: float,
    occ_x: float,
    occ_y: float,
    portrait_place: Dict[str, Any],
) -> bool:
    if query.mode_name != "single_person_center":
        return False
    if not bottom_margin_relaxed or not bool(portrait_place.get("active", False)):
        return False
    shot_type = str(portrait_place.get("shot_type", "unknown") or "unknown")
    if shot_type not in {"half", "three_quarter"}:
        return False
    return (
        core_recall >= 0.98
        and face_recall >= 0.98
        and subject_ratio <= 0.66
        and occ_x <= 0.78
        and occ_y <= 0.86
        and _safe_float(portrait_place.get("q_place", 0.0), 0.0) >= 0.92
        and _safe_float(portrait_place.get("q_x", 0.0), 0.0) >= 0.90
        and _safe_float(portrait_place.get("q_eye_y", 0.0), 0.0) >= 0.90
    )


def _full_body_support_grounding_relaxed(
    *,
    query: ModeQuery,
    portrait_place: Dict[str, Any],
    core_recall: float,
    face_recall: float,
    joint_cut_score: float,
    intrusion_ratio: float,
    margin_bottom: float,
    subject_outside: float,
) -> bool:
    if query.mode_name not in {"single_person_center", "single_person_rot"}:
        return False
    if not bool(portrait_place.get("active", False)) or not bool(portrait_place.get("support_active", False)):
        return False
    if str(portrait_place.get("shot_type", "unknown") or "unknown") != "full":
        return False
    if str(portrait_place.get("support_point_source", "") or "") != "bbox_bottom_adjusted":
        return False
    return (
        core_recall >= 0.98
        and face_recall >= 0.96
        and joint_cut_score <= 0.08
        and intrusion_ratio <= 0.04
        and margin_bottom >= -0.003
        and subject_outside <= 0.003
        and _safe_float(portrait_place.get("q_support_y", 0.0), 0.0) >= 0.08
    )


def _is_group_portrait_secondary_exception(query: ModeQuery) -> bool:
    rule = str(query.attributes.get("secondary_person_exception_rule", "") or "")
    return rule.startswith("group_portrait_secondary_")


def _portrait_control_deviation(portrait_place: Dict[str, Any], mode_name: str) -> Tuple[Optional[float], Optional[float]]:
    rx_ctrl = portrait_place.get("rx_ctrl")
    if rx_ctrl is None:
        return None, None
    rx = _safe_float(rx_ctrl, 0.5)
    center_dev = abs(rx - 0.5)
    if not str(mode_name).endswith("_rot"):
        return center_dev, None
    direction_hint = str(portrait_place.get("direction_hint", "unknown") or "unknown")
    targets = [1.0 / 3.0, 2.0 / 3.0]
    if direction_hint == "right":
        targets = [1.0 / 3.0]
    elif direction_hint == "left":
        targets = [2.0 / 3.0]
    return center_dev, min(abs(rx - target) for target in targets)


def _strict_center_control_exception(
    *,
    query: ModeQuery,
    portrait_place: Dict[str, Any],
    core_recall: float,
    face_recall: float,
    joint_cut_score: float,
    intrusion_ratio: float,
    subject_outside: float,
) -> bool:
    """Allow only near-perfect full-body/support cases to escape stricter center gates."""

    if not bool(portrait_place.get("active", False)):
        return False
    if str(portrait_place.get("shot_type", "unknown") or "unknown") != "full":
        return False
    if not bool(portrait_place.get("support_active", False)):
        return False
    q_support = _safe_float(portrait_place.get("q_support_y"), 0.0)
    q_eye = _safe_float(portrait_place.get("q_eye_y"), 0.0)
    return (
        core_recall >= 0.99
        and face_recall >= 0.985
        and q_support >= 0.45
        and q_eye >= 0.55
        and joint_cut_score <= 0.04
        and intrusion_ratio <= 0.03
        and subject_outside <= 0.002
        and int(_safe_float(query.attributes.get("person_rank"), 0.0)) == 0
    )


def _strict_rot_center_like(center_deviation: Optional[float], thirds_deviation: Optional[float]) -> bool:
    if center_deviation is None or thirds_deviation is None:
        return False
    return center_deviation < 0.09 and thirds_deviation > 0.075


def _landscape_subject_safe_requirements(query: ModeQuery) -> Tuple[bool, float, float]:
    if int(_safe_float(query.attributes.get("landscape_subject_safe_enabled", 1), 1.0)) == 0:
        return False, 0.0, 0.0
    route_bucket = str(query.attributes.get("route_bucket", "") or "").strip().lower()
    route_family = str(query.route_family or "").strip().lower()
    route_mode = str(query.route_mode or "").strip().lower()
    core_area = box_area(query.core_bbox_norm_xyxy)
    active = (
        core_area >= 0.025
        and (
            route_bucket in {"human", "object"}
            or route_family in {"human", "person", "object", "animal"}
            or route_mode.startswith("portrait")
            or route_mode.startswith("object")
        )
    )
    target_ar = str(query.target_ar or "").strip().upper()
    if target_ar in {"9:16", "3:4"}:
        return active, 0.90, 0.68
    if target_ar in {"1:1"}:
        return active, 0.88, 0.66
    return active, 0.85, 0.62


def _axis_quality(mode_name: str, occ_x: float, occ_y: float) -> float:
    occ_dom = max(occ_x, occ_y)
    occ_min = min(occ_x, occ_y)
    if mode_name == "single_person_center":
        qx = _gaussian_quality(occ_x, 0.28, 0.14)
        qy = _gaussian_quality(occ_y, 0.62, 0.12)
        return _clamp(0.35 * qx + 0.65 * qy)
    if mode_name == "single_person_rot":
        qx = _gaussian_quality(occ_x, 0.32, 0.14)
        qy = _gaussian_quality(occ_y, 0.66, 0.12)
        return _clamp(0.35 * qx + 0.65 * qy)
    if mode_name == "group_center":
        qx = _gaussian_quality(occ_x, 0.62, 0.16)
        qy = _gaussian_quality(occ_y, 0.70, 0.14)
        return _clamp(0.45 * qx + 0.55 * qy)
    if mode_name == "group_rot":
        qx = _gaussian_quality(occ_x, 0.66, 0.16)
        qy = _gaussian_quality(occ_y, 0.74, 0.14)
        return _clamp(0.45 * qx + 0.55 * qy)
    if mode_name == "object_single_center":
        q_dom = _gaussian_quality(occ_dom, 0.82, 0.10)
        q_min = _gaussian_quality(occ_min, 0.38, 0.18)
        return _clamp(0.75 * q_dom + 0.25 * q_min)
    if mode_name == "object_single_rot":
        q_dom = _gaussian_quality(occ_dom, 0.84, 0.10)
        q_min = _gaussian_quality(occ_min, 0.34, 0.18)
        return _clamp(0.75 * q_dom + 0.25 * q_min)
    if mode_name == "object_multi_center":
        q_dom = _gaussian_quality(occ_dom, 0.86, 0.12)
        q_min = _gaussian_quality(occ_min, 0.40, 0.18)
        return _clamp(0.75 * q_dom + 0.25 * q_min)
    if mode_name == "object_multi_rot":
        q_dom = _gaussian_quality(occ_dom, 0.88, 0.12)
        q_min = _gaussian_quality(occ_min, 0.36, 0.18)
        return _clamp(0.75 * q_dom + 0.25 * q_min)
    return 1.0


def _actual_pixel_ar(crop: Sequence[float], image_ar: float) -> float:
    crop_w = max(1e-8, float(crop[2]) - float(crop[0]))
    crop_h = max(1e-8, float(crop[3]) - float(crop[1]))
    return float((crop_w / crop_h) * float(image_ar))


def _target_ar_error(query: ModeQuery, crop: Sequence[float], image_ar: float) -> Tuple[float, float]:
    if str(query.target_ar or "").strip().upper() == "FREE":
        return 1.0, 0.0
    target_pixel_ar = parse_target_ar(query.target_ar, image_ar)
    actual_pixel_ar = _actual_pixel_ar(crop, image_ar)
    rel_err = abs(actual_pixel_ar - target_pixel_ar) / max(1e-8, target_pixel_ar)
    return _gaussian_quality(rel_err, 0.0, 0.035), rel_err


def _scene_quality(feat_row: Dict[str, Any], crop: Sequence[float]) -> float:
    c5 = feat_row.get("c5_geom")
    if isinstance(c5, list):
        c5 = c5[0] if c5 else {}
    if not isinstance(c5, dict):
        c5 = {}
    horizon_conf = _safe_float(c5.get("horizon_conf", 0.0), 0.0)
    symmetry_score = _safe_float(c5.get("symmetry_score", 0.0), 0.0)
    horizon_roll = c5.get("horizon_roll") if isinstance(c5.get("horizon_roll"), dict) else {}
    horizon_y = horizon_roll.get("horizon_y_norm")
    crop_w = max(1e-8, float(crop[2]) - float(crop[0]))
    crop_h = max(1e-8, float(crop[3]) - float(crop[1]))
    wide_score = _clamp(box_area(crop) / 0.60)
    if horizon_y is None:
        horizon_visible = 0.5
    else:
        horizon_visible = 1.0 if float(crop[1]) <= float(horizon_y) <= float(crop[3]) else 0.0
    horizon_score = _clamp(0.6 * horizon_visible + 0.4 * horizon_conf)
    return _clamp(0.45 * horizon_score + 0.30 * symmetry_score + 0.25 * wide_score)


def _face_recall(query: ModeQuery, crop: Sequence[float]) -> float:
    if query.member_face_boxes_norm_xyxy:
        recalls = [_recall(face_box, crop) for face_box in query.member_face_boxes_norm_xyxy]
        if recalls:
            return sum(recalls) / float(len(recalls))
    if query.face_bbox_norm_xyxy is None:
        return 1.0
    return _recall(query.face_bbox_norm_xyxy, crop)


def _joint_visibility(query: ModeQuery, crop: Sequence[float]) -> Tuple[float, float]:
    visible = 0
    total = 0
    margin = 0.01
    x1, y1, x2, y2 = [float(v) for v in crop]
    for keypoints in query.member_keypoints_norm:
        for kp in keypoints:
            if len(kp) < 3 or float(kp[2]) < 0.15:
                continue
            total += 1
            x = float(kp[0])
            y = float(kp[1])
            inside = (x1 + margin) <= x <= (x2 - margin) and (y1 + margin) <= y <= (y2 - margin)
            if inside:
                visible += 1
    if total <= 0:
        return 1.0, 0.0
    ratio = visible / float(total)
    return ratio, 1.0 - ratio


def _headroom_quality(query: ModeQuery, crop: Sequence[float], mode_name: str) -> Tuple[float, Optional[float], bool]:
    if not query.head_y_norm:
        return 0.5, None, False
    if mode_name == "face":
        target = 0.16
        sigma_lo = 0.08
        sigma_hi = 0.16
    elif mode_name.startswith("single_person"):
        target = 0.12
        sigma_lo = 0.06
        sigma_hi = 0.14
    elif mode_name.startswith("group"):
        target = 0.10
        sigma_lo = 0.06
        sigma_hi = 0.16
    else:
        target = 0.12
        sigma_lo = 0.07
        sigma_hi = 0.14
    y_head = min(float(v) for v in query.head_y_norm)
    crop_h = max(1e-8, float(crop[3]) - float(crop[1]))
    ratio = (y_head - float(crop[1])) / crop_h
    sigma = sigma_lo if ratio <= target else sigma_hi
    return _gaussian_quality(ratio, target, sigma), ratio, True


def _lookroom_quality(query: ModeQuery, crop: Sequence[float]) -> Tuple[float, Optional[float], str]:
    if not query.gaze_entries:
        return 0.5, None, "unknown"
    best = max(query.gaze_entries, key=lambda entry: _safe_float(entry.get("conf", 0.0), 0.0))
    gaze_dir = str(best.get("gaze_dir", "unknown"))
    if gaze_dir not in {"left", "right"}:
        return 0.5, None, gaze_dir
    x1, _, x2, _ = [float(v) for v in crop]
    ax = _clamp(_safe_float(best.get("anchor_x", 0.5), 0.5), x1, x2)
    m_l = max(1e-8, ax - x1)
    m_r = max(1e-8, x2 - ax)
    fwd = m_r if gaze_dir == "right" else m_l
    back = m_l if gaze_dir == "right" else m_r
    ratio = (fwd + 1e-6) / (back + 1e-6)
    return _gaussian_quality(ratio, 1.45, 0.45), ratio, gaze_dir


def _other_person_intrusion(query: ModeQuery, crop: Sequence[float], feat_row: Dict[str, Any], width: int, height: int) -> float:
    if query.entity_type not in {"person", "face"}:
        return 0.0
    target_box = query.anchor_bbox_norm_xyxy
    intruding = 0.0
    for pose in feat_row.get("c3_pose", []) or []:
        if not isinstance(pose, dict):
            continue
        bbox = pose.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        person_box = norm_box_xyxy(bbox, width, height)
        if _intersection_area(person_box, target_box) / max(1e-8, min(box_area(person_box), box_area(target_box))) >= 0.70:
            continue
        intruding += _intersection_area(person_box, crop)
    return intruding / max(1e-8, box_area(crop))


def _face_body_leakage(query: ModeQuery, crop: Sequence[float]) -> float:
    if query.mode_name != "face" or not query.member_boxes_norm_xyxy:
        return 0.0
    person_box = query.member_boxes_norm_xyxy[0]
    head_box = query.head_bbox_norm_xyxy or query.face_bbox_norm_xyxy or query.anchor_bbox_norm_xyxy
    head_h = max(1e-8, float(head_box[3]) - float(head_box[1]))
    torso_y1 = min(float(person_box[3]), float(head_box[3]) + 0.15 * head_h)
    torso_box = [float(person_box[0]), torso_y1, float(person_box[2]), float(person_box[3])]
    if torso_box[3] <= torso_box[1]:
        return 0.0
    crop_area = max(1e-8, box_area(crop))
    return _clamp(_intersection_area(torso_box, crop) / crop_area)


def _group_quality(query: ModeQuery, crop: Sequence[float]) -> Tuple[float, float]:
    if not query.member_boxes_norm_xyxy:
        return 0.0, 0.0
    recalls = [_recall(box, crop) for box in query.member_boxes_norm_xyxy]
    avg_recall = sum(recalls) / float(len(recalls))
    if len(recalls) == 1:
        balance = 1.0
    else:
        mean = avg_recall
        variance = sum((rec - mean) ** 2 for rec in recalls) / float(len(recalls))
        balance = _clamp(1.0 - math.sqrt(variance) / 0.35)
    return avg_recall, balance


def _collect_object_boxes(feat_row: Dict[str, Any], width: int, height: int) -> List[List[float]]:
    boxes: List[List[float]] = []
    for inst in feat_row.get("c2_seg", []) or []:
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        class_id = int(_safe_float(inst.get("class_id", -1), -1))
        if class_id == 0:
            continue
        box = inst.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            norm = norm_box_xyxy(box, width, height)
            if box_area(norm) > 0.0:
                boxes.append(norm)
    return boxes


def _saliency_overlap(feat_row: Dict[str, Any], crop: Sequence[float]) -> Tuple[float, float]:
    c7 = feat_row.get("c7_saliency") if isinstance(feat_row.get("c7_saliency"), dict) else {}
    fg_box = c7.get("foreground_bbox_norm_xyxy")
    if not isinstance(fg_box, (list, tuple)) or len(fg_box) != 4:
        fg_box = c7.get("top_component_bbox_norm_xyxy")
    fg_area = _safe_float(c7.get("foreground_area_ratio"), 0.0)
    if not isinstance(fg_box, (list, tuple)) or len(fg_box) != 4:
        return 1.0, fg_area
    crop_area = max(1e-8, box_area(crop))
    return _intersection_area(crop, clip_box01(fg_box)) / crop_area, fg_area


def score_query_candidates(
    *,
    query: ModeQuery,
    candidates: Sequence[Dict[str, Any]],
    feat_row: Dict[str, Any],
    width: int,
    height: int,
    image_ar: float,
    landscape_score_policy: str = "composition",
    landscape_teacher_score_scope: str = "all",
    mode_intent_policy: str = "legacy",
) -> List[Dict[str, Any]]:
    spec = MODE_SPEC_BY_NAME[query.mode_name]
    landscape_score_policy_norm = str(landscape_score_policy or "composition").strip().lower()
    landscape_teacher_score_scope_norm = str(landscape_teacher_score_scope or "all").strip().lower()
    mode_intent_policy_norm = str(mode_intent_policy or "legacy").strip().lower()
    if landscape_score_policy_norm not in {"composition", "teacher_only", "teacher_subject_safe"}:
        raise ValueError(f"Unsupported landscape score policy: {landscape_score_policy}")
    if landscape_teacher_score_scope_norm not in {"all", "gaic"}:
        raise ValueError(f"Unsupported landscape teacher score scope: {landscape_teacher_score_scope}")
    if mode_intent_policy_norm not in {"legacy", "strict_v11"}:
        raise ValueError(f"Unsupported mode intent policy: {mode_intent_policy}")
    strict_intent = mode_intent_policy_norm == "strict_v11"
    scored: List[Dict[str, Any]] = []
    all_object_boxes = _collect_object_boxes(feat_row, width, height)
    alignment_bonus = mode_alignment_bonus(query.mode_name, query.route_mode, query.route_family)
    route_aligned = bool(int(_safe_float(query.attributes.get("route_aligned", 1.0), 1.0)))
    dominance_score = _safe_float(query.attributes.get("dominance_score", 1.0), 1.0)
    route_gate_tau = _safe_float(query.attributes.get("route_gate_tau", 0.0), 0.0)
    portrait_comp = query.attributes.get("portrait_comp") if isinstance(query.attributes.get("portrait_comp"), dict) else None
    for candidate in candidates:
        bbox = candidate.get("bbox_norm_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        crop = clip_box01(bbox)
        crop_area = max(1e-8, box_area(crop))
        core_recall = _recall(query.core_bbox_norm_xyxy, crop)
        env_recall = _recall(query.envelope_bbox_norm_xyxy, crop)
        secondary_recall = _recall(query.secondary_core_bbox_norm_xyxy, crop) if query.secondary_core_bbox_norm_xyxy else env_recall
        q_subj = _clamp(0.55 * core_recall + 0.25 * env_recall + 0.20 * secondary_recall)
        subject_ratio = _clamp(box_area(query.anchor_bbox_norm_xyxy) / crop_area)
        q_scale = _gaussian_quality(subject_ratio, spec.subject_ratio_target, spec.subject_ratio_sigma)
        q_place = _placement_quality(query.mode_name, crop, query.anchor_bbox_norm_xyxy)
        anchor_rx, anchor_ry = _relative_anchor_position(crop, query.anchor_bbox_norm_xyxy)
        portrait_place = portrait_place_components(crop, portrait_comp, query.mode_name)
        if query.mode_name in {"single_person_center", "single_person_rot", "group_center", "group_rot"} and bool(portrait_place.get("active", False)):
            q_place = _safe_float(portrait_place.get("q_place", q_place), q_place)
        occ_x, occ_y, occ_dom, occ_min = _axis_occupancy(query, crop)
        q_axis = _axis_quality(query.mode_name, occ_x, occ_y)
        margin_left, margin_right, margin_top, margin_bottom, subject_outside = _subject_margins(query, crop)
        control_center_deviation, control_thirds_deviation = _portrait_control_deviation(portrait_place, query.mode_name)
        q_ar, ar_rel_error = _target_ar_error(query, crop, image_ar)
        q_ctx, context_value = _context_quality(subject_ratio, spec.context_target, spec.context_sigma)
        q_scene = _scene_quality(feat_row, crop)
        q_head, headroom_value, headroom_activated = _headroom_quality(query, crop, query.mode_name)
        q_look, lookroom_value, gaze_dir = _lookroom_quality(query, crop)
        face_recall = _face_recall(query, crop)
        joint_visible_ratio, joint_cut_score = _joint_visibility(query, crop)
        intrusion_ratio = _other_person_intrusion(query, crop, feat_row, width, height)
        q_group_recall, q_group_balance = _group_quality(query, crop)
        q_group = _clamp(0.55 * q_group_recall + 0.25 * q_group_balance + 0.20 * env_recall)
        rot_thirds_deviation = _rot_thirds_deviation(crop, query.anchor_bbox_norm_xyxy) if query.mode_name.endswith("_rot") else 0.0
        face_body_leakage = _face_body_leakage(query, crop) if query.mode_name == "face" else 0.0
        route_dom_penalty = 0.0 if route_aligned else 0.04 * max(0.0, route_gate_tau - dominance_score + 0.04)
        saliency_crop_overlap, saliency_fg_area_ratio = _saliency_overlap(feat_row, crop)
        teacher_score_raw = _candidate_teacher_score(candidate, scope=landscape_teacher_score_scope_norm)
        q_teach = _teacher_score_quality(teacher_score_raw)
        bottom_margin_relaxed = _source_bottom_margin_relaxed(
            query,
            crop,
            core_recall=core_recall,
            group_recall=q_group_recall,
            face_recall=face_recall,
        )
        single_center_axis_relaxed = _single_center_source_axis_relaxed(
            query,
            bottom_margin_relaxed=bottom_margin_relaxed,
            core_recall=core_recall,
            face_recall=face_recall,
            subject_ratio=subject_ratio,
            occ_x=occ_x,
            occ_y=occ_y,
            portrait_place=portrait_place,
        )
        full_body_support_relaxed = _full_body_support_grounding_relaxed(
            query=query,
            portrait_place=portrait_place,
            core_recall=core_recall,
            face_recall=face_recall,
            joint_cut_score=joint_cut_score,
            intrusion_ratio=intrusion_ratio,
            margin_bottom=margin_bottom,
            subject_outside=subject_outside,
        )
        strict_center_exception = _strict_center_control_exception(
            query=query,
            portrait_place=portrait_place,
            core_recall=core_recall,
            face_recall=face_recall,
            joint_cut_score=joint_cut_score,
            intrusion_ratio=intrusion_ratio,
            subject_outside=subject_outside,
        )
        landscape_subject_safe_active, landscape_min_core_recall, landscape_min_q_subj = _landscape_subject_safe_requirements(query)

        object_recall = 0.0
        if is_object_mode(query.mode_name) and query.entity_type == "object":
            object_recall = max((_recall(box, crop) for box in all_object_boxes), default=0.0)

        hard_rejects: List[str] = []
        if ar_rel_error > 0.035:
            hard_rejects.append("target_ar_mismatch")
        if query.mode_name in {"single_person_center", "single_person_rot"}:
            person_rank = int(_safe_float(query.attributes.get("person_rank"), 0.0))
            secondary_exception = int(_safe_float(query.attributes.get("secondary_person_exception"), 0.0))
            if person_rank > 0:
                group_secondary_exception = _is_group_portrait_secondary_exception(query)
                if secondary_exception != 1:
                    hard_rejects.append("secondary_person_without_exception")
                else:
                    if subject_ratio < (0.22 if group_secondary_exception else 0.30):
                        hard_rejects.append("secondary_person_crop_dominance_miss")
                    if core_recall < (0.95 if group_secondary_exception else 0.98):
                        hard_rejects.append("secondary_person_core_recall_miss")
                    if intrusion_ratio > (0.16 if group_secondary_exception else 0.04):
                        hard_rejects.append("secondary_person_intrusion")
                    if occ_dom < (0.34 if group_secondary_exception else 0.45):
                        hard_rejects.append("secondary_person_too_loose")
            if face_recall < 0.96:
                hard_rejects.append("face_cut")
            if headroom_activated and headroom_value is not None and headroom_value < 0.04:
                hard_rejects.append("head_top_cut")
            if joint_cut_score > 0.35:
                hard_rejects.append("joint_cutoff")
            if core_recall < 0.90:
                hard_rejects.append("subject_coverage")
            if intrusion_ratio > 0.12:
                hard_rejects.append("other_person_intrusion")
            if gaze_dir in {"left", "right"} and lookroom_value is not None and lookroom_value < 1.05:
                hard_rejects.append("lookroom_cut")
            if query.mode_name.endswith("_rot") and rot_thirds_deviation > 0.11:
                hard_rejects.append("rot_placement_miss")
            if (
                bool(portrait_place.get("support_active", False))
                and _safe_float(portrait_place.get("q_support_y", 1.0), 1.0) < 0.16
                and not bottom_margin_relaxed
                and not full_body_support_relaxed
            ):
                hard_rejects.append("support_grounding_miss")
            if query.mode_name == "single_person_center":
                if bool(portrait_place.get("active", False)):
                    if _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.42:
                        hard_rejects.append("center_composition_miss")
                    if _safe_float(portrait_place.get("q_x", 1.0), 1.0) < 0.35:
                        hard_rejects.append("center_control_x_miss")
                    if control_center_deviation is not None and control_center_deviation > 0.20:
                        hard_rejects.append("center_control_far")
                    if strict_intent and not strict_center_exception:
                        q_place_strict = _safe_float(portrait_place.get("q_place", 1.0), 1.0)
                        q_x_strict = _safe_float(portrait_place.get("q_x", 1.0), 1.0)
                        rx_ctrl_strict = portrait_place.get("rx_ctrl")
                        strict_thirds_deviation = (
                            None
                            if rx_ctrl_strict is None
                            else min(
                                abs(_safe_float(rx_ctrl_strict, 0.5) - (1.0 / 3.0)),
                                abs(_safe_float(rx_ctrl_strict, 0.5) - (2.0 / 3.0)),
                            )
                        )
                        if q_place_strict < 0.56:
                            hard_rejects.append("center_composition_miss_strict")
                        if q_x_strict < 0.52:
                            hard_rejects.append("center_control_x_miss_strict")
                        if control_center_deviation is not None and control_center_deviation > 0.125:
                            hard_rejects.append("center_control_far_strict")
                        if (
                            control_center_deviation is not None
                            and strict_thirds_deviation is not None
                            and control_center_deviation >= 0.10
                            and strict_thirds_deviation <= 0.075
                        ):
                            hard_rejects.append("center_control_rot_like")
                    if (
                        bool(portrait_place.get("support_active", False))
                        and _safe_float(portrait_place.get("q_support_y", 1.0), 1.0) < 0.25
                        and not bottom_margin_relaxed
                        and not full_body_support_relaxed
                    ):
                        hard_rejects.append("center_support_grounding_miss")
                center_side_relaxed = _source_side_margin_relaxed(
                    query,
                    crop,
                    margin_left=margin_left,
                    margin_right=margin_right,
                    threshold=0.025,
                    core_recall=core_recall,
                    group_recall=q_group_recall,
                    face_recall=face_recall,
                )
                if (margin_left < 0.025 or margin_right < 0.025) and not center_side_relaxed:
                    hard_rejects.append("center_side_margin_tight")
                if bool(portrait_place.get("support_active", False)) and margin_bottom < 0.025 and not bottom_margin_relaxed:
                    hard_rejects.append("center_bottom_margin_tight")
                if subject_outside > 0.01:
                    hard_rejects.append("center_subject_outside")
                if occ_x > 0.72 and not single_center_axis_relaxed:
                    hard_rejects.append("center_horizontal_tight")
                if occ_y > 0.80 and not single_center_axis_relaxed:
                    hard_rejects.append("center_vertical_tight")
            if query.mode_name == "single_person_rot":
                if bool(portrait_place.get("active", False)):
                    if _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.45:
                        hard_rejects.append("rot_composition_miss")
                    if _safe_float(portrait_place.get("q_x", 1.0), 1.0) < 0.35:
                        hard_rejects.append("rot_control_x_miss")
                    if control_thirds_deviation is not None and control_thirds_deviation > 0.16:
                        hard_rejects.append("rot_control_thirds_miss")
                    if strict_intent:
                        if _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.54:
                            hard_rejects.append("rot_composition_miss_strict")
                        if control_thirds_deviation is not None and control_thirds_deviation > 0.105:
                            hard_rejects.append("rot_control_thirds_miss_strict")
                        if _strict_rot_center_like(control_center_deviation, control_thirds_deviation):
                            hard_rejects.append("rot_control_center_like")
                    if (
                        bool(portrait_place.get("support_active", False))
                        and _safe_float(portrait_place.get("q_support_y", 1.0), 1.0) < 0.35
                        and not bottom_margin_relaxed
                        and not full_body_support_relaxed
                    ):
                        hard_rejects.append("rot_support_grounding_miss")
                    rx_ctrl = portrait_place.get("rx_ctrl")
                    direction_hint = str(portrait_place.get("direction_hint", "unknown") or "unknown")
                    if rx_ctrl is not None and direction_hint == "right" and _safe_float(rx_ctrl, 0.5) < 0.20:
                        hard_rejects.append("rot_right_gaze_too_far_left")
                    if rx_ctrl is not None and direction_hint == "left" and _safe_float(rx_ctrl, 0.5) > 0.80:
                        hard_rejects.append("rot_left_gaze_too_far_right")
                rot_side_relaxed = _source_side_margin_relaxed(
                    query,
                    crop,
                    margin_left=margin_left,
                    margin_right=margin_right,
                    threshold=0.035,
                    core_recall=core_recall,
                    group_recall=q_group_recall,
                    face_recall=face_recall,
                )
                if (margin_left < 0.035 or margin_right < 0.035) and not rot_side_relaxed:
                    hard_rejects.append("rot_side_margin_tight")
                if bool(portrait_place.get("support_active", False)) and margin_bottom < 0.035 and not bottom_margin_relaxed:
                    hard_rejects.append("rot_bottom_margin_tight")
                if subject_outside > 0.01:
                    hard_rejects.append("rot_subject_outside")
            if occ_y > (0.84 if query.mode_name == "single_person_center" else 0.86) and not single_center_axis_relaxed:
                hard_rejects.append("vertical_tight")
            if occ_dom > (0.90 if query.mode_name == "single_person_center" else 0.92) and not single_center_axis_relaxed:
                hard_rejects.append("axis_tight")
            utility = (
                0.27 * q_subj
                + 0.10 * q_scale
                + (0.17 if query.mode_name == "single_person_center" else 0.18) * q_place
                + 0.12 * q_axis
                + 0.13 * q_head
                + 0.10 * q_look
                + 0.05 * q_ctx
                + 0.05 * q_ar
                + alignment_bonus
                - 0.25 * intrusion_ratio
                - 0.18 * joint_cut_score
            )
        elif query.mode_name in {"group_center", "group_rot"}:
            if q_group_recall < 0.90:
                hard_rejects.append("group_recall")
            if face_recall < 0.90:
                hard_rejects.append("member_face_cut")
            if joint_cut_score > 0.40:
                hard_rejects.append("member_joint_cut")
            if query.mode_name.endswith("_rot") and rot_thirds_deviation > 0.11:
                hard_rejects.append("rot_placement_miss")
            if query.mode_name == "group_center":
                if bool(portrait_place.get("active", False)) and _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.35:
                    hard_rejects.append("group_center_composition_miss")
                if strict_intent and bool(portrait_place.get("active", False)):
                    if _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.50:
                        hard_rejects.append("group_center_composition_miss_strict")
                    if control_center_deviation is not None and control_center_deviation > 0.16:
                        hard_rejects.append("group_center_control_far_strict")
                group_center_side_relaxed = _source_side_margin_relaxed(
                    query,
                    crop,
                    margin_left=margin_left,
                    margin_right=margin_right,
                    threshold=0.025,
                    core_recall=core_recall,
                    group_recall=q_group_recall,
                    face_recall=face_recall,
                )
                if (margin_left < 0.025 or margin_right < 0.025) and not group_center_side_relaxed:
                    hard_rejects.append("group_center_side_margin_tight")
                if margin_bottom < 0.025 and not bottom_margin_relaxed:
                    hard_rejects.append("group_center_bottom_margin_tight")
                if subject_outside > 0.01:
                    hard_rejects.append("group_center_subject_outside")
                group_axis_relaxed = _group_center_axis_relaxed(
                    query,
                    crop,
                    bottom_margin_relaxed=bottom_margin_relaxed,
                    occ_x=occ_x,
                    occ_y=occ_y,
                    group_recall=q_group_recall,
                    face_recall=face_recall,
                    joint_cut_score=joint_cut_score,
                )
                if (occ_x > 0.88 or occ_y > 0.88) and not group_axis_relaxed:
                    hard_rejects.append("group_center_axis_tight")
            if query.mode_name == "group_rot":
                if bool(portrait_place.get("active", False)) and _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.40:
                    hard_rejects.append("group_rot_composition_miss")
                if control_thirds_deviation is not None and control_thirds_deviation > 0.16:
                    hard_rejects.append("group_rot_control_thirds_miss")
                if strict_intent and bool(portrait_place.get("active", False)):
                    if _safe_float(portrait_place.get("q_place", 1.0), 1.0) < 0.50:
                        hard_rejects.append("group_rot_composition_miss_strict")
                    if control_thirds_deviation is not None and control_thirds_deviation > 0.105:
                        hard_rejects.append("group_rot_control_thirds_miss_strict")
                    if _strict_rot_center_like(control_center_deviation, control_thirds_deviation):
                        hard_rejects.append("group_rot_control_center_like")
                group_rot_side_relaxed = _source_side_margin_relaxed(
                    query,
                    crop,
                    margin_left=margin_left,
                    margin_right=margin_right,
                    threshold=0.025,
                    core_recall=core_recall,
                    group_recall=q_group_recall,
                    face_recall=face_recall,
                )
                if (margin_left < 0.025 or margin_right < 0.025) and not group_rot_side_relaxed:
                    hard_rejects.append("group_rot_side_margin_tight")
                if subject_outside > 0.01:
                    hard_rejects.append("group_rot_subject_outside")
            if occ_dom > (0.95 if query.mode_name == "group_center" else 0.97):
                hard_rejects.append("axis_tight")
            utility = (
                (0.24 if query.mode_name == "group_center" else 0.23) * q_group
                + (0.16 if query.mode_name == "group_center" else 0.15) * q_subj
                + (0.10 if query.mode_name == "group_center" else 0.09) * q_scale
                + (0.15 if query.mode_name == "group_center" else 0.20) * q_place
                + 0.08 * q_axis
                + (0.08 if query.mode_name == "group_center" else 0.06) * q_head
                + (0.11 if query.mode_name == "group_center" else 0.11) * q_ctx
                + 0.05 * q_ar
                + alignment_bonus
                - 0.15 * joint_cut_score
            )
        elif query.mode_name == "face":
            person_rank = int(_safe_float(query.attributes.get("person_rank"), 0.0))
            secondary_exception = int(_safe_float(query.attributes.get("secondary_person_exception"), 0.0))
            if person_rank > 0:
                group_secondary_exception = _is_group_portrait_secondary_exception(query)
                if secondary_exception != 1:
                    hard_rejects.append("secondary_face_without_exception")
                else:
                    if subject_ratio < (0.14 if group_secondary_exception else 0.22):
                        hard_rejects.append("secondary_face_crop_dominance_miss")
                    if core_recall < (0.95 if group_secondary_exception else 0.98):
                        hard_rejects.append("secondary_face_core_recall_miss")
                    if intrusion_ratio > (0.10 if group_secondary_exception else 0.03):
                        hard_rejects.append("secondary_face_intrusion")
                    if face_body_leakage > (0.22 if group_secondary_exception else 0.16):
                        hard_rejects.append("secondary_face_body_leakage")
            if face_recall < 0.95:
                hard_rejects.append("face_coverage")
            if core_recall < 0.92:
                hard_rejects.append("head_subject_coverage")
            if headroom_activated and headroom_value is not None and headroom_value < 0.03:
                hard_rejects.append("head_top_cut")
            if intrusion_ratio > 0.08:
                hard_rejects.append("other_person_intrusion")
            if subject_ratio < 0.12:
                hard_rejects.append("face_too_loose")
            if face_body_leakage > 0.22:
                hard_rejects.append("body_leakage")
            if q_place < 0.35:
                hard_rejects.append("face_center_placement_miss")
            if abs(float(anchor_rx) - 0.5) > 0.34:
                hard_rejects.append("face_center_x_miss")
            if strict_intent:
                if q_place < 0.55:
                    hard_rejects.append("face_center_placement_miss_strict")
                if abs(float(anchor_rx) - 0.5) > 0.22:
                    hard_rejects.append("face_center_x_miss_strict")
                if face_body_leakage > 0.25:
                    hard_rejects.append("body_leakage_strict")
            utility = (
                0.30 * q_subj
                + 0.28 * q_scale
                + 0.12 * q_place
                + 0.12 * q_head
                + 0.05 * q_look
                + 0.07 * q_ctx
                + 0.06 * q_ar
                + alignment_bonus
                - 0.30 * intrusion_ratio
                - 0.22 * face_body_leakage
            )
        elif query.mode_name == "landscape":
            candidate_source = str(candidate.get("source", "unknown"))
            scene_subtype_norm = str(query.attributes.get("scene_subtype", "") or "").strip().lower()
            route_mode_norm = str(query.route_mode or "").strip().lower()
            route_bucket_norm = str(query.attributes.get("route_bucket", "") or "").strip().lower()
            scene_or_interior = (
                route_bucket_norm == "scene"
                or route_mode_norm in {"scene_general", "background_texture_copyspace"}
                or "interior" in scene_subtype_norm
                or "indoor" in scene_subtype_norm
            )
            if not route_aligned and not candidate_source.startswith("teacher:"):
                hard_rejects.append("non_scene_landscape_without_public_teacher")
            if landscape_score_policy_norm in {"teacher_only", "teacher_subject_safe"} and not candidate_source.startswith("teacher:"):
                hard_rejects.append("landscape_teacher_only_non_teacher_candidate")
            if landscape_score_policy_norm in {"teacher_only", "teacher_subject_safe"} and teacher_score_raw is None:
                hard_rejects.append(f"landscape_no_{landscape_teacher_score_scope_norm}_teacher_score")
            if (
                scene_or_interior
                and landscape_teacher_score_scope_norm == "gaic"
                and teacher_score_raw is not None
                and q_teach < spec.tau_pos
            ):
                hard_rejects.append("landscape_low_gaic_qteach_scene_no_label")
            if (
                landscape_score_policy_norm == "teacher_subject_safe"
                and landscape_subject_safe_active
                and core_recall < landscape_min_core_recall
            ):
                hard_rejects.append("landscape_subject_core_recall_miss")
            if (
                landscape_score_policy_norm == "teacher_subject_safe"
                and landscape_subject_safe_active
                and q_subj < landscape_min_q_subj
            ):
                hard_rejects.append("landscape_subject_quality_miss")
            if not route_aligned and q_scale == 0.0 and q_ctx < 0.01:
                hard_rejects.append("non_scene_landscape_low_context")
            suppress_scene_subject_loss = bool(str(query.attributes.get("landscape_subject_safe_suppressed_reason", "") or ""))
            if q_subj < 0.45 and box_area(query.core_bbox_norm_xyxy) > 0.05 and not suppress_scene_subject_loss:
                hard_rejects.append("salient_subject_loss")
            if box_area(crop) < 0.15:
                hard_rejects.append("too_tight")
            if landscape_score_policy_norm == "teacher_only":
                utility = q_teach
            elif landscape_score_policy_norm == "teacher_subject_safe":
                if landscape_subject_safe_active:
                    subject_cut_penalty = max(0.0, landscape_min_core_recall - core_recall) + max(
                        0.0, landscape_min_q_subj - q_subj
                    )
                    subject_bonus = 0.010 * q_subj + 0.008 * core_recall
                else:
                    subject_cut_penalty = 0.0
                    subject_bonus = 0.0
                utility = (
                    q_teach
                    + subject_bonus
                    + 0.004 * q_scene
                    + 0.003 * q_place
                    - 0.040 * subject_cut_penalty
                )
            else:
                utility = (
                    0.10 * q_subj
                    + 0.10 * q_scale
                    + 0.14 * q_place
                    + 0.28 * q_scene
                    + 0.23 * q_ctx
                    + 0.05 * _clamp(context_value)
                    + 0.10 * q_ar
                    + alignment_bonus
                )
        elif query.mode_name in {"object_single_center", "object_single_rot"}:
            if core_recall < (0.92 if query.mode_name == "object_single_center" else 0.90):
                hard_rejects.append("object_truncation")
            if saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.35:
                hard_rejects.append("object_low_saliency_overlap")
            if strict_intent and saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.45:
                hard_rejects.append("object_low_saliency_overlap_strict")
            if query.mode_name == "object_single_center":
                if q_place < 0.30:
                    hard_rejects.append("object_center_placement_miss")
                if q_axis < 0.25:
                    hard_rejects.append("object_center_axis_miss")
                if saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.35:
                    hard_rejects.append("object_center_low_saliency_overlap")
                if strict_intent:
                    if q_place < 0.60:
                        hard_rejects.append("object_center_placement_miss_strict")
                    if abs(float(anchor_rx) - 0.5) > 0.18:
                        hard_rejects.append("object_center_x_miss_strict")
                    if saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.45:
                        hard_rejects.append("object_center_low_saliency_overlap_strict")
            if query.mode_name.endswith("_rot") and rot_thirds_deviation > 0.12:
                hard_rejects.append("rot_placement_miss")
            if strict_intent and query.mode_name.endswith("_rot"):
                center_dev_anchor = abs(float(anchor_rx) - 0.5)
                if rot_thirds_deviation > 0.10:
                    hard_rejects.append("rot_placement_miss_strict")
                if _strict_rot_center_like(center_dev_anchor, rot_thirds_deviation):
                    hard_rejects.append("rot_center_like_strict")
            if occ_dom > 0.98 or (occ_dom > 0.94 and q_axis < 0.20):
                hard_rejects.append("axis_tight")
            if not route_aligned and str(query.attributes.get("route_bucket", "")) == "human" and saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.35:
                hard_rejects.append("human_route_object_low_saliency_overlap")
            if (
                strict_intent
                and not route_aligned
                and str(query.attributes.get("route_bucket", "")) == "human"
                and saliency_fg_area_ratio < 0.85
                and saliency_crop_overlap < 0.50
            ):
                hard_rejects.append("human_route_object_low_saliency_overlap_strict")
            if not route_aligned and dominance_score < max(0.80, route_gate_tau):
                hard_rejects.append("route_dominance_miss")
            utility = (
                (0.30 if query.mode_name == "object_single_center" else 0.28) * q_subj
                + (0.14 if query.mode_name == "object_single_center" else 0.13) * q_scale
                + (0.14 if query.mode_name == "object_single_center" else 0.18) * q_place
                + 0.13 * q_axis
                + (0.10 if query.mode_name == "object_single_center" else 0.09) * q_ctx
                + 0.10 * q_scene
                + 0.07 * q_ar
                + alignment_bonus
                - route_dom_penalty
                - 0.12 * (1.0 - object_recall)
            )
        else:
            if q_group_recall < 0.88:
                hard_rejects.append("object_group_recall")
            if saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.35:
                hard_rejects.append("object_multi_low_saliency_overlap")
            if strict_intent and saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.45:
                hard_rejects.append("object_multi_low_saliency_overlap_strict")
            if query.mode_name == "object_multi_center":
                if q_place < 0.30:
                    hard_rejects.append("object_multi_center_placement_miss")
                if saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.35:
                    hard_rejects.append("object_multi_center_low_saliency_overlap")
                if strict_intent:
                    if q_place < 0.58:
                        hard_rejects.append("object_multi_center_placement_miss_strict")
                    if abs(float(anchor_rx) - 0.5) > 0.20:
                        hard_rejects.append("object_multi_center_x_miss_strict")
                    if saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.45:
                        hard_rejects.append("object_multi_center_low_saliency_overlap_strict")
            if query.mode_name.endswith("_rot") and rot_thirds_deviation > 0.12:
                hard_rejects.append("rot_placement_miss")
            if strict_intent and query.mode_name.endswith("_rot"):
                center_dev_anchor = abs(float(anchor_rx) - 0.5)
                if rot_thirds_deviation > 0.10:
                    hard_rejects.append("rot_placement_miss_strict")
                if _strict_rot_center_like(center_dev_anchor, rot_thirds_deviation):
                    hard_rejects.append("rot_center_like_strict")
            if not route_aligned and str(query.attributes.get("route_bucket", "")) == "human" and saliency_fg_area_ratio < 0.85 and saliency_crop_overlap < 0.35:
                hard_rejects.append("human_route_object_multi_low_saliency_overlap")
            if (
                strict_intent
                and not route_aligned
                and str(query.attributes.get("route_bucket", "")) == "human"
                and saliency_fg_area_ratio < 0.85
                and saliency_crop_overlap < 0.50
            ):
                hard_rejects.append("human_route_object_multi_low_saliency_overlap_strict")
            if occ_dom > 1.02 or (occ_dom > 0.96 and q_axis < 0.20):
                hard_rejects.append("axis_tight")
            if not route_aligned and dominance_score < max(0.82, route_gate_tau):
                hard_rejects.append("route_dominance_miss")
            utility = (
                (0.22 if query.mode_name == "object_multi_center" else 0.20) * q_subj
                + 0.22 * q_group
                + (0.10 if query.mode_name == "object_multi_center" else 0.09) * q_scale
                + (0.14 if query.mode_name == "object_multi_center" else 0.20) * q_place
                + 0.10 * q_axis
                + (0.14 if query.mode_name == "object_multi_center" else 0.14) * q_ctx
                + 0.08 * q_ar
                + alignment_bonus
                - route_dom_penalty
            )

        scored.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "bbox_norm_xyxy": [round(float(v), 6) for v in crop],
                "source": str(candidate.get("source", "unknown")),
                "score_mode": round(float(utility), 6),
                "hard_reject": bool(hard_rejects),
                "hard_reject_reasons": list(hard_rejects),
                "components": {
                    "q_subj": round(float(q_subj), 6),
                    "q_scale": round(float(q_scale), 6),
                    "q_axis": round(float(q_axis), 6),
                    "q_place": round(float(q_place), 6),
                    "q_ar": round(float(q_ar), 6),
                    "q_ctx": round(float(q_ctx), 6),
                    "q_scene": round(float(q_scene), 6),
                    "q_head": round(float(q_head), 6),
                    "q_look": round(float(q_look), 6),
                    "q_group": round(float(q_group), 6),
                    "q_teach": round(float(q_teach), 6),
                    "teacher_score_raw": None if teacher_score_raw is None else round(float(teacher_score_raw), 6),
                    "landscape_score_policy": landscape_score_policy_norm,
                    "landscape_teacher_score_scope": landscape_teacher_score_scope_norm,
                    "mode_intent_policy": mode_intent_policy_norm,
                    "landscape_subject_safe_active": int(bool(landscape_subject_safe_active)),
                    "landscape_min_core_recall": round(float(landscape_min_core_recall), 6),
                    "landscape_min_q_subj": round(float(landscape_min_q_subj), 6),
                    "landscape_subject_safe_enabled": int(
                        bool(int(_safe_float(query.attributes.get("landscape_subject_safe_enabled", 1), 1.0)))
                    ),
                    "landscape_subject_safe_suppressed_reason": str(
                        query.attributes.get("landscape_subject_safe_suppressed_reason", "") or ""
                    ),
                    "scene_subject_reliability": round(
                        float(_safe_float(query.attributes.get("scene_subject_reliability", 1.0), 1.0)),
                        6,
                    ),
                    "scene_subject_agreement_iou": round(
                        float(_safe_float(query.attributes.get("scene_subject_agreement_iou", 1.0), 1.0)),
                        6,
                    ),
                    "scene_subject_core_source": str(query.attributes.get("scene_subject_core_source", "") or ""),
                    "scene_subject_guidance_area": round(
                        float(_safe_float(query.attributes.get("scene_subject_guidance_area", 0.0), 0.0)),
                        6,
                    ),
                    "scene_subject_saliency_area": round(
                        float(_safe_float(query.attributes.get("scene_subject_saliency_area", 0.0), 0.0)),
                        6,
                    ),
                    "scene_subject_guidance_saliency_iou": round(
                        float(_safe_float(query.attributes.get("scene_subject_guidance_saliency_iou", 0.0), 0.0)),
                        6,
                    ),
                    "context_value": round(float(context_value), 6),
                    "subject_ratio": round(float(subject_ratio), 6),
                    "anchor_rx": round(float(anchor_rx), 6),
                    "anchor_ry": round(float(anchor_ry), 6),
                    "occ_x": round(float(occ_x), 6),
                    "occ_y": round(float(occ_y), 6),
                    "occ_dom": round(float(occ_dom), 6),
                    "occ_min": round(float(occ_min), 6),
                    "subject_margin_left": round(float(margin_left), 6),
                    "subject_margin_right": round(float(margin_right), 6),
                    "subject_margin_top": round(float(margin_top), 6),
                    "subject_margin_bottom": round(float(margin_bottom), 6),
                    "subject_outside": round(float(subject_outside), 6),
                    "core_recall": round(float(core_recall), 6),
                    "env_recall": round(float(env_recall), 6),
                    "face_recall": round(float(face_recall), 6),
                    "joint_visible_ratio": round(float(joint_visible_ratio), 6),
                    "joint_cut_score": round(float(joint_cut_score), 6),
                    "intrusion_ratio": round(float(intrusion_ratio), 6),
                    "face_body_leakage": round(float(face_body_leakage), 6),
                    "group_recall": round(float(q_group_recall), 6),
                    "group_balance": round(float(q_group_balance), 6),
                    "headroom_value": None if headroom_value is None else round(float(headroom_value), 6),
                    "lookroom_value": None if lookroom_value is None else round(float(lookroom_value), 6),
                    "gaze_dir": gaze_dir,
                    "alignment_bonus": round(float(alignment_bonus), 6),
                    "route_dominance_score": round(float(dominance_score), 6),
                    "route_gate_tau": round(float(route_gate_tau), 6),
                    "route_dom_penalty": round(float(route_dom_penalty), 6),
                    "ar_rel_error": round(float(ar_rel_error), 6),
                    "rot_thirds_deviation": round(float(rot_thirds_deviation), 6),
                    "portrait_control_center_deviation": (
                        None if control_center_deviation is None else round(float(control_center_deviation), 6)
                    ),
                    "portrait_control_thirds_deviation": (
                        None if control_thirds_deviation is None else round(float(control_thirds_deviation), 6)
                    ),
                    "saliency_crop_overlap": round(float(saliency_crop_overlap), 6),
                    "saliency_fg_area_ratio": round(float(saliency_fg_area_ratio), 6),
                    "portrait_q_x": None if not portrait_place.get("active", False) else portrait_place.get("q_x"),
                    "portrait_q_eye_y": None if not portrait_place.get("active", False) else portrait_place.get("q_eye_y"),
                    "portrait_q_support_y": None if not portrait_place.get("active", False) else portrait_place.get("q_support_y"),
                    "portrait_rx_ctrl": None if not portrait_place.get("active", False) else portrait_place.get("rx_ctrl"),
                    "portrait_ry_eye": None if not portrait_place.get("active", False) else portrait_place.get("ry_eye"),
                    "portrait_ry_support": None if not portrait_place.get("active", False) else portrait_place.get("ry_support"),
                    "portrait_direction_hint": portrait_place.get("direction_hint", "unknown"),
                    "portrait_shot_type": portrait_place.get("shot_type", "unknown"),
                    "portrait_source_bottom_margin_relaxed": int(bool(bottom_margin_relaxed)),
                    "portrait_full_body_support_grounding_relaxed": int(bool(full_body_support_relaxed)),
                    "portrait_strict_center_exception": int(bool(strict_center_exception)),
                    "portrait_source_side_margin_relaxed": int(
                        bool(
                            _source_side_margin_relaxed(
                                query,
                                crop,
                                margin_left=margin_left,
                                margin_right=margin_right,
                                threshold=0.035,
                                core_recall=core_recall,
                                group_recall=q_group_recall,
                                face_recall=face_recall,
                            )
                        )
                    ),
                    "portrait_single_center_axis_relaxed": int(bool(single_center_axis_relaxed)),
                    "portrait_group_axis_tight_relaxed": int(
                        bool(
                            _group_center_axis_relaxed(
                                query,
                                crop,
                                bottom_margin_relaxed=bottom_margin_relaxed,
                                occ_x=occ_x,
                                occ_y=occ_y,
                                group_recall=q_group_recall,
                                face_recall=face_recall,
                                joint_cut_score=joint_cut_score,
                            )
                        )
                    ),
                },
            }
        )
    scored.sort(key=lambda row: (float(row["score_mode"]), int(not row["hard_reject"])), reverse=True)
    return scored


def select_query_results(
    *,
    query: ModeQuery,
    scored_candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    spec = MODE_SPEC_BY_NAME[query.mode_name]
    best_positive: Optional[Dict[str, Any]] = None
    for candidate in scored_candidates:
        if bool(candidate.get("hard_reject", False)):
            continue
        if _safe_float(candidate.get("score_mode", 0.0), 0.0) >= spec.tau_pos:
            best_positive = dict(candidate)
            break
    selected_negatives: List[Dict[str, Any]] = []
    if best_positive is not None:
        positive_id = str(best_positive.get("candidate_id", ""))
    else:
        positive_id = ""
    hard_negative = next(
        (
            dict(candidate)
            for candidate in scored_candidates
            if bool(candidate.get("hard_reject", False)) and str(candidate.get("candidate_id", "")) != positive_id
        ),
        None,
    )
    if hard_negative is not None:
        hard_negative["negative_reason"] = ",".join(hard_negative.get("hard_reject_reasons", [])) or "hard_reject"
        selected_negatives.append(hard_negative)
    near_negative = next(
        (
            dict(candidate)
            for candidate in scored_candidates
            if not bool(candidate.get("hard_reject", False))
            and str(candidate.get("candidate_id", "")) != positive_id
            and _safe_float(candidate.get("score_mode", 0.0), 0.0) < spec.tau_pos
        ),
        None,
    )
    if near_negative is not None and str(near_negative.get("candidate_id", "")) not in {
        str(candidate.get("candidate_id", ""))
        for candidate in selected_negatives
    }:
        near_negative["negative_reason"] = "below_tau"
        selected_negatives.append(near_negative)
    return {
        "query_id": query.query_id,
        "mode_name": query.mode_name,
        "mode_id": spec.category_id,
        "tau_pos": spec.tau_pos,
        "positive": best_positive,
        "negatives": selected_negatives,
        "best_scored_candidate": dict(scored_candidates[0]) if scored_candidates else None,
        "candidate_count": len(scored_candidates),
    }
