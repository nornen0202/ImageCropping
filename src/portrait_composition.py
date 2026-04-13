from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from subject_region import box_area, box_center, clip_box01, union_boxes


KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _clip_point(point_xy: Sequence[float]) -> List[float]:
    return [
        _clamp(_safe_float(point_xy[0], 0.5), 0.0, 1.0),
        _clamp(_safe_float(point_xy[1], 0.5), 0.0, 1.0),
    ]


def _midpoint(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        0.5 * (_safe_float(a[0], 0.5) + _safe_float(b[0], 0.5)),
        0.5 * (_safe_float(a[1], 0.5) + _safe_float(b[1], 0.5)),
    ]


def _point_local_ratio(crop: Sequence[float], point_xy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in crop]
    crop_w = max(1e-8, x2 - x1)
    crop_h = max(1e-8, y2 - y1)
    return (
        (float(point_xy[0]) - x1) / crop_w,
        (float(point_xy[1]) - y1) / crop_h,
    )


def _gaussian_quality(value: float, target: float, sigma: float) -> float:
    sigma = max(1e-6, float(sigma))
    delta = float(value) - float(target)
    return float(math.exp(-(delta * delta) / (2.0 * sigma * sigma)))


def _asymmetric_gaussian(value: float, target: float, sigma_lo: float, sigma_hi: float) -> float:
    sigma = sigma_lo if float(value) <= float(target) else sigma_hi
    return _gaussian_quality(value, target, sigma)


def _normalized_box(box: Optional[Sequence[float]]) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    clipped = clip_box01(box)
    if box_area(clipped) <= 1e-8:
        return None
    return [float(v) for v in clipped]


def _keypoint_point(keypoints_norm: Sequence[Sequence[float]], index: int, conf_thr: float = 0.15) -> Optional[List[float]]:
    if index >= len(keypoints_norm):
        return None
    kp = keypoints_norm[index]
    if not isinstance(kp, (list, tuple)) or len(kp) < 3:
        return None
    conf = _safe_float(kp[2], 0.0)
    if conf < float(conf_thr):
        return None
    return [_clamp(_safe_float(kp[0], 0.5)), _clamp(_safe_float(kp[1], 0.5))]


def _single_or_midpoint(
    keypoints_norm: Sequence[Sequence[float]],
    left_idx: int,
    right_idx: int,
    conf_thr: float = 0.15,
) -> tuple[Optional[List[float]], float]:
    left = _keypoint_point(keypoints_norm, left_idx, conf_thr=conf_thr)
    right = _keypoint_point(keypoints_norm, right_idx, conf_thr=conf_thr)
    if left is not None and right is not None:
        return _midpoint(left, right), 1.0
    if left is not None:
        return list(left), 0.65
    if right is not None:
        return list(right), 0.65
    return None, 0.0


def _best_gaze_direction(gaze_entries: Sequence[Dict[str, Any]]) -> str:
    best_entry: Optional[Dict[str, Any]] = None
    best_conf = -1.0
    for entry in gaze_entries:
        if not isinstance(entry, dict):
            continue
        conf = _safe_float(entry.get("conf", 0.0), 0.0)
        if conf > best_conf:
            best_conf = conf
            best_entry = entry
    if best_entry is None:
        return "unknown"
    gaze_dir = str(best_entry.get("gaze_dir", "unknown")).strip().lower()
    if gaze_dir in {"left", "right", "center"}:
        return gaze_dir
    yaw_proxy = _safe_float(best_entry.get("yaw_proxy", 0.0), 0.0)
    if yaw_proxy >= 0.10:
        return "right"
    if yaw_proxy <= -0.10:
        return "left"
    return "unknown"


def _support_target_y(shot_type: str) -> Optional[float]:
    table = {
        "three_quarter": 0.92,
        "full": 0.95,
    }
    return table.get(str(shot_type or ""))


def _eye_target_y(shot_type: str) -> float:
    table = {
        "headshot": 0.38,
        "half": 0.33,
        "three_quarter": 0.28,
        "full": 0.22,
        "group": 0.30,
        "unknown": 0.34,
    }
    return float(table.get(str(shot_type or "unknown"), 0.34))


def _weights_for_shot(shot_type: str) -> Dict[str, float]:
    table = {
        "headshot": {"eye": 0.78, "torso": 0.22, "pelvis": 0.00},
        "half": {"eye": 0.58, "torso": 0.32, "pelvis": 0.10},
        "three_quarter": {"eye": 0.42, "torso": 0.33, "pelvis": 0.25},
        "full": {"eye": 0.20, "torso": 0.35, "pelvis": 0.45},
        "group": {"eye": 0.65, "torso": 0.35, "pelvis": 0.00},
        "unknown": {"eye": 0.46, "torso": 0.34, "pelvis": 0.20},
    }
    return dict(table.get(str(shot_type or "unknown"), table["unknown"]))


def _placement_y_for_shot(
    *,
    shot_type: str,
    eye_point: Optional[Sequence[float]],
    torso_point: Optional[Sequence[float]],
    support_point: Optional[Sequence[float]],
    support_active: bool,
) -> float:
    candidates: List[Tuple[float, float]] = []
    if eye_point is not None:
        eye_w = {
            "headshot": 0.85,
            "half": 0.72,
            "three_quarter": 0.48,
            "full": 0.22,
            "group": 0.70,
        }.get(str(shot_type or "unknown"), 0.62)
        candidates.append((float(eye_w), _safe_float(eye_point[1], 0.35)))
    if torso_point is not None:
        torso_w = {
            "headshot": 0.15,
            "half": 0.28,
            "three_quarter": 0.28,
            "full": 0.20,
            "group": 0.30,
        }.get(str(shot_type or "unknown"), 0.24)
        candidates.append((float(torso_w), _safe_float(torso_point[1], 0.50)))
    if support_active and support_point is not None:
        support_w = {
            "three_quarter": 0.24,
            "full": 0.58,
        }.get(str(shot_type or "unknown"), 0.0)
        if support_w > 0.0:
            candidates.append((float(support_w), _safe_float(support_point[1], 0.90)))
    if not candidates:
        return 0.5
    total = sum(weight for weight, _ in candidates)
    if total <= 1e-8:
        return 0.5
    return _clamp(sum(weight * value for weight, value in candidates) / total, 0.0, 1.0)


def _infer_single_shot_type(
    *,
    bbox_box: Sequence[float],
    face_box: Optional[Sequence[float]],
    keypoints_norm: Sequence[Sequence[float]],
) -> str:
    bbox_area = max(1e-8, box_area(bbox_box))
    face_ratio = box_area(face_box) / bbox_area if face_box is not None else 0.0
    has_hip = _keypoint_point(keypoints_norm, KP_LEFT_HIP) is not None or _keypoint_point(keypoints_norm, KP_RIGHT_HIP) is not None
    has_knee = _keypoint_point(keypoints_norm, KP_LEFT_KNEE) is not None or _keypoint_point(keypoints_norm, KP_RIGHT_KNEE) is not None
    has_ankle = _keypoint_point(keypoints_norm, KP_LEFT_ANKLE) is not None or _keypoint_point(keypoints_norm, KP_RIGHT_ANKLE) is not None
    if has_ankle:
        return "full"
    if has_knee and has_hip:
        return "three_quarter"
    if has_hip:
        return "half"
    if face_ratio >= 0.18:
        return "headshot"
    if face_ratio >= 0.10:
        return "half"
    return "unknown"


def build_single_portrait_spec(
    *,
    bbox_norm_xyxy: Sequence[float],
    face_bbox_norm_xyxy: Optional[Sequence[float]] = None,
    keypoints_norm: Optional[Sequence[Sequence[float]]] = None,
    gaze_entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    bbox_box = _normalized_box(bbox_norm_xyxy)
    if bbox_box is None:
        return None
    face_box = _normalized_box(face_bbox_norm_xyxy)
    keypoints = list(keypoints_norm or [])
    gaze_entries = list(gaze_entries or [])
    bbox_center_xy = list(box_center(bbox_box))
    bbox_h = max(1e-6, float(bbox_box[3]) - float(bbox_box[1]))

    eye_point, eye_conf = _single_or_midpoint(keypoints, KP_LEFT_EYE, KP_RIGHT_EYE)
    if eye_point is None:
        nose = _keypoint_point(keypoints, KP_NOSE)
        if nose is not None:
            eye_point = list(nose)
            eye_conf = 0.55
    if eye_point is None and face_box is not None:
        eye_point = list(box_center(face_box))
        eye_conf = 0.45
    if eye_point is None:
        eye_point = [bbox_center_xy[0], _clamp(float(bbox_box[1]) + 0.22 * bbox_h)]
        eye_conf = 0.10

    shoulder_mid, shoulder_conf = _single_or_midpoint(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hip_mid, hip_conf = _single_or_midpoint(keypoints, KP_LEFT_HIP, KP_RIGHT_HIP)
    torso_point = None
    if shoulder_mid is not None and hip_mid is not None:
        torso_point = _midpoint(shoulder_mid, hip_mid)
    elif shoulder_mid is not None:
        torso_point = list(shoulder_mid)
    elif hip_mid is not None:
        torso_point = list(hip_mid)
    if torso_point is None:
        torso_point = list(bbox_center_xy)

    pelvis_point = list(hip_mid) if hip_mid is not None else list(torso_point)
    support_point, support_conf = _single_or_midpoint(keypoints, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)

    shot_type = _infer_single_shot_type(bbox_box=bbox_box, face_box=face_box, keypoints_norm=keypoints)
    direction_hint = _best_gaze_direction(gaze_entries)
    weights = _weights_for_shot(shot_type)
    available = []
    if eye_point is not None:
        available.append(("eye", eye_point))
    if torso_point is not None:
        available.append(("torso", torso_point))
    if pelvis_point is not None:
        available.append(("pelvis", pelvis_point))
    weight_total = sum(weights.get(name, 0.0) for name, _ in available)
    if weight_total <= 1e-8:
        control_x = bbox_center_xy[0]
        control_y = bbox_center_xy[1]
    else:
        control_x = sum(weights.get(name, 0.0) * float(point[0]) for name, point in available) / weight_total
        control_y = sum(weights.get(name, 0.0) * float(point[1]) for name, point in available) / weight_total
    support_active = support_point is not None and shot_type in {"three_quarter", "full"}
    placement_anchor_y = _placement_y_for_shot(
        shot_type=shot_type,
        eye_point=eye_point,
        torso_point=torso_point,
        support_point=support_point,
        support_active=support_active,
    )
    placement_anchor_xy = [float(control_x), float(placement_anchor_y)]
    eye_target_y = _eye_target_y(shot_type)
    support_target_y = _support_target_y(shot_type)
    return {
        "kind": "single",
        "shot_type": shot_type,
        "direction_hint": direction_hint,
        "bbox_center_norm_xy": [round(float(bbox_center_xy[0]), 6), round(float(bbox_center_xy[1]), 6)],
        "eye_point_norm_xy": None if eye_point is None else [round(float(eye_point[0]), 6), round(float(eye_point[1]), 6)],
        "eye_conf": round(float(eye_conf), 6),
        "torso_point_norm_xy": None if torso_point is None else [round(float(torso_point[0]), 6), round(float(torso_point[1]), 6)],
        "pelvis_point_norm_xy": None if pelvis_point is None else [round(float(pelvis_point[0]), 6), round(float(pelvis_point[1]), 6)],
        "support_point_norm_xy": None if support_point is None else [round(float(support_point[0]), 6), round(float(support_point[1]), 6)],
        "support_conf": round(float(support_conf), 6),
        "control_point_norm_xy": [round(float(control_x), 6), round(float(control_y), 6)],
        "placement_anchor_norm_xy": [round(float(placement_anchor_xy[0]), 6), round(float(placement_anchor_xy[1]), 6)],
        "eye_target_y": round(float(eye_target_y), 6),
        "support_target_y": None if support_target_y is None else round(float(support_target_y), 6),
        "support_active": bool(support_active),
    }


def build_group_portrait_spec(
    *,
    member_boxes_norm_xyxy: Sequence[Sequence[float]],
    member_face_boxes_norm_xyxy: Optional[Sequence[Sequence[float]]] = None,
    member_keypoints_norm: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    gaze_entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    member_boxes = [_normalized_box(box) for box in member_boxes_norm_xyxy]
    member_boxes = [box for box in member_boxes if box is not None]
    if not member_boxes:
        return None
    union_box = union_boxes(member_boxes)
    if union_box is None or box_area(union_box) <= 1e-8:
        return None
    face_centers: List[List[float]] = []
    for face_box in member_face_boxes_norm_xyxy or []:
        norm = _normalized_box(face_box)
        if norm is not None:
            face_centers.append(list(box_center(norm)))
    if not face_centers and member_keypoints_norm:
        for keypoints in member_keypoints_norm:
            eye_point, _ = _single_or_midpoint(keypoints, KP_LEFT_EYE, KP_RIGHT_EYE)
            if eye_point is not None:
                face_centers.append(list(eye_point))
    if face_centers:
        eye_x = sum(point[0] for point in face_centers) / float(len(face_centers))
        eye_y = sum(point[1] for point in face_centers) / float(len(face_centers))
        eye_point = [_clamp(eye_x), _clamp(eye_y)]
    else:
        eye_point = list(box_center(union_box))
    torso_x = sum(box_center(box)[0] for box in member_boxes) / float(len(member_boxes))
    torso_y = sum(box_center(box)[1] for box in member_boxes) / float(len(member_boxes))
    torso_point = [_clamp(torso_x), _clamp(torso_y)]
    placement_y = 0.72 * eye_point[1] + 0.28 * torso_point[1]
    return {
        "kind": "group",
        "shot_type": "group",
        "direction_hint": _best_gaze_direction(gaze_entries or []),
        "bbox_center_norm_xy": [round(float(box_center(union_box)[0]), 6), round(float(box_center(union_box)[1]), 6)],
        "eye_point_norm_xy": [round(float(eye_point[0]), 6), round(float(eye_point[1]), 6)],
        "torso_point_norm_xy": [round(float(torso_point[0]), 6), round(float(torso_point[1]), 6)],
        "pelvis_point_norm_xy": None,
        "support_point_norm_xy": None,
        "support_conf": 0.0,
        "control_point_norm_xy": [round(float(eye_point[0]), 6), round(float(torso_point[1]), 6)],
        "placement_anchor_norm_xy": [round(float(eye_point[0]), 6), round(float(placement_y), 6)],
        "eye_target_y": round(float(_eye_target_y("group")), 6),
        "support_target_y": None,
        "support_active": False,
    }


def portrait_seed_layouts(spec: Optional[Dict[str, Any]], mode_name: str) -> List[Dict[str, Any]]:
    if not isinstance(spec, dict):
        return []
    control_point = spec.get("control_point_norm_xy")
    placement_anchor = spec.get("placement_anchor_norm_xy")
    eye_point = spec.get("eye_point_norm_xy")
    torso_point = spec.get("torso_point_norm_xy")
    support_point = spec.get("support_point_norm_xy")
    shot_type = str(spec.get("shot_type", "unknown") or "unknown")
    direction_hint = str(spec.get("direction_hint", "unknown") or "unknown")
    support_active = bool(spec.get("support_active", False))
    eye_target_y = _safe_float(spec.get("eye_target_y", _eye_target_y(shot_type)), _eye_target_y(shot_type))
    support_target_y = spec.get("support_target_y")
    x_targets = [0.5]
    if str(mode_name).endswith("_rot"):
        if direction_hint == "right":
            x_targets = [1.0 / 3.0]
        elif direction_hint == "left":
            x_targets = [2.0 / 3.0]
        else:
            x_targets = [1.0 / 3.0, 2.0 / 3.0]
    layouts: List[Dict[str, Any]] = []
    if eye_point is not None:
        layouts.append(
            {
                "family": "portrait_eye",
                "anchor_xy": list(eye_point),
                "placement_x_values": list(x_targets),
                "placement_y_values": [max(0.12, eye_target_y - 0.04), eye_target_y, min(0.90, eye_target_y + 0.04)],
                "priority": 1.0,
            }
        )
    if placement_anchor is not None:
        layouts.append(
            {
                "family": "portrait_anchor",
                "anchor_xy": list(placement_anchor),
                "placement_x_values": list(x_targets),
                "placement_y_values": [float(_clamp(_safe_float(placement_anchor[1], 0.5), 0.10, 0.95))],
                "priority": 0.88,
            }
        )
    if support_active and support_point is not None and support_target_y is not None:
        layouts.append(
            {
                "family": "portrait_support",
                "anchor_xy": list(support_point),
                "placement_x_values": list(x_targets),
                "placement_y_values": [max(0.75, float(support_target_y) - 0.03), float(support_target_y)],
                "priority": 0.96,
            }
        )
    elif torso_point is not None and str(spec.get("kind", "")) == "group":
        layouts.append(
            {
                "family": "group_faceband",
                "anchor_xy": list(torso_point),
                "placement_x_values": list(x_targets),
                "placement_y_values": [0.38, 0.46],
                "priority": 0.84,
            }
        )
    if not layouts and control_point is not None:
        layouts.append(
            {
                "family": "portrait_control",
                "anchor_xy": list(control_point),
                "placement_x_values": list(x_targets),
                "placement_y_values": [0.5],
                "priority": 0.70,
            }
        )
    return layouts


def portrait_place_components(crop: Sequence[float], spec: Optional[Dict[str, Any]], mode_name: str) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {
            "active": False,
            "q_place": 0.0,
            "q_x": 0.0,
            "q_eye_y": 0.0,
            "q_support_y": 0.0,
            "rx_ctrl": None,
            "ry_eye": None,
            "ry_support": None,
            "direction_hint": "unknown",
            "shot_type": "unknown",
            "support_active": False,
        }
    control_point = spec.get("control_point_norm_xy")
    eye_point = spec.get("eye_point_norm_xy")
    support_point = spec.get("support_point_norm_xy")
    shot_type = str(spec.get("shot_type", "unknown") or "unknown")
    direction_hint = str(spec.get("direction_hint", "unknown") or "unknown")
    support_active = bool(spec.get("support_active", False))
    target_x_values = [0.5]
    if str(mode_name).endswith("_rot"):
        if direction_hint == "right":
            target_x_values = [1.0 / 3.0]
        elif direction_hint == "left":
            target_x_values = [2.0 / 3.0]
        else:
            target_x_values = [1.0 / 3.0, 2.0 / 3.0]

    if control_point is not None:
        rx_ctrl, _ = _point_local_ratio(crop, control_point)
        sigma_x = 0.11 if str(mode_name).endswith("_center") else 0.09
        q_x = max(_gaussian_quality(rx_ctrl, target_x, sigma_x) for target_x in target_x_values)
    else:
        rx_ctrl = None
        q_x = 0.5

    eye_target_y = _safe_float(spec.get("eye_target_y", _eye_target_y(shot_type)), _eye_target_y(shot_type))
    if eye_point is not None:
        _, ry_eye = _point_local_ratio(crop, eye_point)
        q_eye_y = _asymmetric_gaussian(ry_eye, eye_target_y, 0.07, 0.10)
    else:
        ry_eye = None
        q_eye_y = 0.5

    support_target_y = spec.get("support_target_y")
    if support_active and support_point is not None and support_target_y is not None:
        _, ry_support = _point_local_ratio(crop, support_point)
        q_support_y = _asymmetric_gaussian(ry_support, float(support_target_y), 0.06, 0.04)
    else:
        ry_support = None
        q_support_y = 0.5

    shot_weights = {
        "headshot": {"x": 0.56, "eye": 0.44, "support": 0.00},
        "half": {"x": 0.54, "eye": 0.40, "support": 0.06},
        "three_quarter": {"x": 0.48, "eye": 0.28, "support": 0.24},
        "full": {"x": 0.42, "eye": 0.20, "support": 0.38},
        "group": {"x": 0.50, "eye": 0.50, "support": 0.00},
        "unknown": {"x": 0.54, "eye": 0.38, "support": 0.08},
    }.get(shot_type, {"x": 0.54, "eye": 0.38, "support": 0.08})
    if not support_active or support_point is None or support_target_y is None:
        shot_weights = dict(shot_weights)
        shot_weights["eye"] += shot_weights.get("support", 0.0)
        shot_weights["support"] = 0.0
    weight_sum = max(1e-8, sum(float(v) for v in shot_weights.values()))
    q_place = (
        float(shot_weights.get("x", 0.0)) * float(q_x)
        + float(shot_weights.get("eye", 0.0)) * float(q_eye_y)
        + float(shot_weights.get("support", 0.0)) * float(q_support_y)
    ) / weight_sum
    return {
        "active": True,
        "q_place": round(float(_clamp(q_place)), 6),
        "q_x": round(float(_clamp(q_x)), 6),
        "q_eye_y": round(float(_clamp(q_eye_y)), 6),
        "q_support_y": round(float(_clamp(q_support_y)), 6),
        "rx_ctrl": None if rx_ctrl is None else round(float(rx_ctrl), 6),
        "ry_eye": None if ry_eye is None else round(float(ry_eye), 6),
        "ry_support": None if ry_support is None else round(float(ry_support), 6),
        "direction_hint": direction_hint,
        "shot_type": shot_type,
        "support_active": bool(support_active),
    }

