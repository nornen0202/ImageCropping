from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCENE_LIKE_MODES = {
    "scene_general",
    "scene_landscape",
    "background_texture_copyspace",
    "text_document",
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clip_box01(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def box_wh(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.sqrt(((ax - bx) ** 2) + ((ay - by) ** 2))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [safe_float(v, 0.0) for v in a]
    bx1, by1, bx2, by2 = [safe_float(v, 0.0) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def norm_box_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    if len(box) != 4:
        raise ValueError(f"Expected 4-length box, got: {box}")
    vals = [safe_float(v, 0.0) for v in box]
    if max(abs(v) for v in vals) <= 1.5:
        return clip_box01(vals)
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    return clip_box01([vals[0] / w, vals[1] / h, vals[2] / w, vals[3] / h])


def union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    valid = [clip_box01(b) for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4 and box_area(b) > 0]
    if not valid:
        return None
    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)
    return clip_box01([x1, y1, x2, y2])


def centered_box(cx: float, cy: float, area_ratio: float, image_ar: float) -> List[float]:
    area_ratio = clamp(area_ratio, 1e-4, 0.95)
    image_ar = max(1e-6, float(image_ar))
    bw = math.sqrt(area_ratio / image_ar)
    bh = bw * image_ar
    return clip_box01([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh])


def expand_box_to_min_area(box: Sequence[float], min_area: float) -> List[float]:
    clipped = clip_box01(box)
    area = box_area(clipped)
    if area <= 0.0:
        return clipped
    target = max(area, float(min_area))
    if target <= area + 1e-8:
        return clipped
    scale = math.sqrt(target / max(area, 1e-8))
    cx, cy = box_center(clipped)
    bw, bh = box_wh(clipped)
    return clip_box01([cx - 0.5 * bw * scale, cy - 0.5 * bh * scale, cx + 0.5 * bw * scale, cy + 0.5 * bh * scale])


def blend_points(a: Tuple[float, float], b: Tuple[float, float], alpha: float) -> Tuple[float, float]:
    alpha = clamp(alpha, 0.0, 1.0)
    return (
        (1.0 - alpha) * float(a[0]) + alpha * float(b[0]),
        (1.0 - alpha) * float(a[1]) + alpha * float(b[1]),
    )


def _pick_norm_box(payload: Dict[str, Any], width: int, height: int, norm_key: str, abs_key: str) -> Optional[List[float]]:
    box = payload.get(norm_key)
    if isinstance(box, (list, tuple)) and len(box) == 4:
        norm = norm_box_xyxy(box, width, height)
        if box_area(norm) > 0:
            return norm
    box = payload.get(abs_key)
    if isinstance(box, (list, tuple)) and len(box) == 4:
        norm = norm_box_xyxy(box, width, height)
        if box_area(norm) > 0:
            return norm
    return None


def summarize_saliency_signal(c7_saliency: Optional[Dict[str, Any]], width: int, height: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "backend": "none",
        "requested_backend": "none",
        "foreground_area_ratio": 0.0,
        "blank_ratio_est": 1.0,
        "dominant_component_area_ratio": 0.0,
        "dominance_score": 0.0,
        "top2_mass_ratio": 0.0,
        "component_count": 0,
        "dispersion_score": 1.0,
        "entropy_norm": 1.0,
        "weighted_centroid_xy_norm": [0.5, 0.5],
        "foreground_bbox_norm_xyxy": None,
        "top_component_bbox_norm_xyxy": None,
        "top2_union_bbox_norm_xyxy": None,
        "components_topk_norm_xyxy": [],
        "dominant_subject_likely": False,
        "multi_subject_likely": False,
        "distributed_attention_likely": False,
        "scene_like_signal": False,
        "support_bbox_norm_xyxy": None,
        "support_kind": "none",
        "support_map_available": False,
        "support_grid_size": 0,
        "support_grid_encoding": "",
        "support_mass_grid_f16_b64": None,
        "support_occ_grid_u8_b64": None,
    }
    if not isinstance(c7_saliency, dict):
        return out

    out["backend"] = str(c7_saliency.get("backend", "none") or "none")
    out["requested_backend"] = str(c7_saliency.get("requested_backend", out["backend"]) or out["backend"])
    out["available"] = bool(c7_saliency.get("available", False))
    if not out["available"]:
        return out

    out["foreground_area_ratio"] = clamp(safe_float(c7_saliency.get("foreground_area_ratio", 0.0), 0.0), 0.0, 1.0)
    out["blank_ratio_est"] = clamp(
        safe_float(c7_saliency.get("blank_ratio_est", 1.0 - out["foreground_area_ratio"]), 1.0 - out["foreground_area_ratio"]),
        0.0,
        1.0,
    )
    out["dominant_component_area_ratio"] = clamp(
        safe_float(c7_saliency.get("dominant_component_area_ratio", 0.0), 0.0),
        0.0,
        1.0,
    )
    out["dominance_score"] = clamp(safe_float(c7_saliency.get("dominance_score", 0.0), 0.0), 0.0, 1.0)
    out["top2_mass_ratio"] = clamp(safe_float(c7_saliency.get("top2_mass_ratio", 0.0), 0.0), 0.0, 1.0)
    out["component_count"] = max(0, int(safe_float(c7_saliency.get("component_count", 0), 0.0)))
    out["dispersion_score"] = clamp(safe_float(c7_saliency.get("dispersion_score", 1.0), 1.0), 0.0, 1.0)
    out["entropy_norm"] = clamp(safe_float(c7_saliency.get("entropy_norm", 1.0), 1.0), 0.0, 1.0)

    centroid = c7_saliency.get("weighted_centroid_xy_norm")
    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        out["weighted_centroid_xy_norm"] = [
            clamp(safe_float(centroid[0], 0.5), 0.0, 1.0),
            clamp(safe_float(centroid[1], 0.5), 0.0, 1.0),
        ]

    out["foreground_bbox_norm_xyxy"] = _pick_norm_box(c7_saliency, width, height, "foreground_bbox_norm_xyxy", "foreground_bbox_xyxy")
    out["top_component_bbox_norm_xyxy"] = _pick_norm_box(c7_saliency, width, height, "top_component_bbox_norm_xyxy", "top_component_bbox_xyxy")
    out["top2_union_bbox_norm_xyxy"] = _pick_norm_box(c7_saliency, width, height, "top2_union_bbox_norm_xyxy", "top2_union_bbox_xyxy")
    components_topk = c7_saliency.get("components_topk", [])
    if isinstance(components_topk, list):
        for comp in components_topk[:3]:
            if not isinstance(comp, dict):
                continue
            box = comp.get("bbox_norm_xyxy") or comp.get("bbox_xyxy")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                norm = norm_box_xyxy(box, width, height)
                if box_area(norm) > 0:
                    out["components_topk_norm_xyxy"].append(norm)
    support_grid_size = int(safe_float(c7_saliency.get("support_grid_size", 0), 0.0))
    support_mass_grid = c7_saliency.get("support_mass_grid_f16_b64")
    support_occ_grid = c7_saliency.get("support_occ_grid_u8_b64")
    if support_grid_size > 0 and isinstance(support_mass_grid, str) and support_mass_grid and isinstance(support_occ_grid, str) and support_occ_grid:
        out["support_map_available"] = True
        out["support_grid_size"] = support_grid_size
        out["support_grid_encoding"] = str(c7_saliency.get("support_grid_encoding", "f16_base64") or "f16_base64")
        out["support_mass_grid_f16_b64"] = support_mass_grid
        out["support_occ_grid_u8_b64"] = support_occ_grid

    out["dominant_subject_likely"] = bool(
        out["dominance_score"] >= 0.55
        and 0.015 <= out["dominant_component_area_ratio"] <= 0.40
        and out["component_count"] <= 4
    )
    out["multi_subject_likely"] = bool(
        out["component_count"] >= 2
        and out["foreground_area_ratio"] >= 0.012
        and out["top2_mass_ratio"] >= 0.38
        and (out["component_count"] >= 3 or out["dominance_score"] <= 0.70)
    )
    out["distributed_attention_likely"] = bool(
        (
            out["foreground_area_ratio"] >= 0.08
            and (
                out["dominance_score"] < 0.45
                or out["component_count"] >= 3
                or out["dispersion_score"] >= 0.30
                or out["entropy_norm"] >= 0.55
            )
        )
        or (
            out["component_count"] >= 4
            and out["foreground_area_ratio"] >= 0.012
            and (
                out["dominance_score"] < 0.55
                or out["top2_mass_ratio"] < 0.60
                or out["entropy_norm"] >= 0.58
            )
        )
        or (
            out["component_count"] >= 6
            and out["foreground_area_ratio"] >= 0.02
        )
    )
    out["scene_like_signal"] = bool(
        out["distributed_attention_likely"]
        and out["foreground_area_ratio"] >= 0.03
        and out["blank_ratio_est"] <= 0.90
    )
    if out["distributed_attention_likely"]:
        out["support_bbox_norm_xyxy"] = (
            out["foreground_bbox_norm_xyxy"]
            or out["top2_union_bbox_norm_xyxy"]
            or out["top_component_bbox_norm_xyxy"]
        )
        out["support_kind"] = (
            "foreground_bbox"
            if out["foreground_bbox_norm_xyxy"] is not None
            else ("top2_union" if out["top2_union_bbox_norm_xyxy"] is not None else "top1")
        )
    elif out["multi_subject_likely"]:
        out["support_bbox_norm_xyxy"] = (
            out["top2_union_bbox_norm_xyxy"]
            or out["foreground_bbox_norm_xyxy"]
            or out["top_component_bbox_norm_xyxy"]
        )
        out["support_kind"] = (
            "top2_union"
            if out["top2_union_bbox_norm_xyxy"] is not None
            else ("foreground_bbox" if out["foreground_bbox_norm_xyxy"] is not None else "top1")
        )
    else:
        out["support_bbox_norm_xyxy"] = (
            out["top_component_bbox_norm_xyxy"]
            or out["top2_union_bbox_norm_xyxy"]
            or out["foreground_bbox_norm_xyxy"]
        )
        out["support_kind"] = (
            "top1"
            if out["top_component_bbox_norm_xyxy"] is not None
            else ("top2_union" if out["top2_union_bbox_norm_xyxy"] is not None else "foreground_bbox")
        )
    return out


def _conservative_scene_anchor(
    *,
    raw_anchor: Optional[Sequence[float]],
    saliency: Dict[str, Any],
    image_ar: float,
) -> List[float]:
    raw = clip_box01(raw_anchor) if isinstance(raw_anchor, (list, tuple)) and len(raw_anchor) == 4 else None
    raw_area = box_area(raw) if raw is not None else 0.0
    sal_box = saliency.get("support_bbox_norm_xyxy") or saliency.get("top2_union_bbox_norm_xyxy") or saliency.get("foreground_bbox_norm_xyxy")
    sal_box = clip_box01(sal_box) if isinstance(sal_box, (list, tuple)) and len(sal_box) == 4 else None
    sal_area = box_area(sal_box) if sal_box is not None else 0.0
    centroid_xy = saliency.get("weighted_centroid_xy_norm", [0.5, 0.5])
    sal_centroid = (
        clamp(safe_float(centroid_xy[0], 0.5), 0.0, 1.0),
        clamp(safe_float(centroid_xy[1], 0.5), 0.0, 1.0),
    )
    blank_ratio = clamp(safe_float(saliency.get("blank_ratio_est", 1.0), 1.0), 0.0, 1.0)
    fg_ratio = clamp(safe_float(saliency.get("foreground_area_ratio", 0.0), 0.0), 0.0, 1.0)
    center_pull = 0.60 if (blank_ratio >= 0.95 or fg_ratio <= 0.05) else 0.40
    anchor_cx, anchor_cy = blend_points(sal_centroid, (0.5, 0.5), center_pull)
    target_area = max(0.18, min(0.32, max(sal_area, 0.22 if saliency.get("available", False) else 0.25)))
    if raw is not None and raw_area >= 0.08:
        overlap = iou_xyxy(raw, sal_box or raw)
        if overlap >= 0.10:
            return expand_box_to_min_area(raw, min_area=max(target_area, raw_area))
    return centered_box(anchor_cx, anchor_cy, target_area, image_ar)


def _saliency_support_box(saliency: Dict[str, Any]) -> Optional[List[float]]:
    support = saliency.get("support_bbox_norm_xyxy") or saliency.get("foreground_bbox_norm_xyxy") or saliency.get("top2_union_bbox_norm_xyxy") or saliency.get("top_component_bbox_norm_xyxy")
    if isinstance(support, (list, tuple)) and len(support) == 4:
        clipped = clip_box01(support)
        if box_area(clipped) > 0:
            return clipped
    return None


def _raw_saliency_group(raw_anchor: Optional[Sequence[float]], sal_top1: Optional[Sequence[float]]) -> Optional[Dict[str, Any]]:
    if not (isinstance(raw_anchor, (list, tuple)) and len(raw_anchor) == 4):
        return None
    if not (isinstance(sal_top1, (list, tuple)) and len(sal_top1) == 4):
        return None
    raw = clip_box01(raw_anchor)
    sal = clip_box01(sal_top1)
    raw_area = box_area(raw)
    sal_area = box_area(sal)
    if raw_area <= 0.0 or sal_area <= 0.0:
        return None
    agreement_iou = iou_xyxy(raw, sal)
    centroid_dist = center_distance(raw, sal)
    union_box = union_boxes([raw, sal])
    if union_box is None:
        return None
    union_area = box_area(union_box)
    if agreement_iou >= 0.08:
        return None
    if centroid_dist <= 0.18:
        return None
    if max(raw_area, sal_area) >= 0.22:
        return None
    if union_area >= 0.45:
        return None
    return {
        "raw_box": raw,
        "saliency_box": sal,
        "union_box": union_box,
        "agreement_iou": round(agreement_iou, 6),
        "center_distance": round(centroid_dist, 6),
        "union_area": round(union_area, 6),
    }


def _decode_support_mass_grid(saliency: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(saliency, dict):
        return None
    grid_size = int(safe_float(saliency.get("support_grid_size", 0), 0.0))
    mass_b64 = saliency.get("support_mass_grid_f16_b64")
    if grid_size <= 0 or not isinstance(mass_b64, str) or not mass_b64:
        return None
    try:
        mass_grid = np.frombuffer(base64.b64decode(mass_b64.encode("ascii")), dtype=np.float16).astype(np.float32)
    except Exception:
        return None
    if mass_grid.size != grid_size * grid_size:
        return None
    mass_grid = mass_grid.reshape((grid_size, grid_size))
    mass_sum = float(mass_grid.sum())
    if mass_sum <= 1e-8:
        return None
    return mass_grid / mass_sum


def _support_map_mass_for_box(box: Optional[Sequence[float]], saliency: Dict[str, Any]) -> Optional[float]:
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    mass_grid = _decode_support_mass_grid(saliency)
    if mass_grid is None:
        return None
    grid_size = int(mass_grid.shape[0])
    x1, y1, x2, y2 = clip_box01(box)
    gx1 = max(0, min(grid_size, int(math.floor(x1 * grid_size))))
    gy1 = max(0, min(grid_size, int(math.floor(y1 * grid_size))))
    gx2 = max(gx1 + 1, min(grid_size, int(math.ceil(x2 * grid_size))))
    gy2 = max(gy1 + 1, min(grid_size, int(math.ceil(y2 * grid_size))))
    return float(clamp(float(mass_grid[gy1:gy2, gx1:gx2].sum()), 0.0, 1.0))


def resolve_effective_subject_region(
    *,
    width: int,
    height: int,
    subject_mode: str,
    subject_set: Optional[Dict[str, Any]],
    raw_anchor_box: Optional[Sequence[float]],
    c7_saliency: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    mode = str(subject_mode or "").strip().lower()
    subject_set = subject_set if isinstance(subject_set, dict) else {}
    primary_idx = int(safe_float(subject_set.get("primary_idx", -1), -1.0))
    raw_anchor = norm_box_xyxy(raw_anchor_box, width, height) if isinstance(raw_anchor_box, (list, tuple)) and len(raw_anchor_box) == 4 else None
    raw_area = box_area(raw_anchor) if raw_anchor is not None else 0.0
    saliency = summarize_saliency_signal(c7_saliency, width=width, height=height)
    image_ar = max(1e-6, float(width) / max(1.0, float(height)))

    fallback_box = raw_anchor or centered_box(0.5, 0.5, 0.25, image_ar)
    candidate_anchor_box = fallback_box
    effective_box = fallback_box
    support_box = fallback_box
    state = "dominant_subject"
    score_mode = "subject_preservation"
    source = "raw_anchor"
    repr_type = "detected_box" if raw_anchor is not None else "guard_proxy"
    placeholder_flag = raw_anchor is None
    subject_reliability = 0.85 if raw_area >= 0.08 else (0.65 if raw_area >= 0.04 else (0.40 if raw_area > 0.0 else 0.05))
    subject_source = "detector" if raw_anchor is not None else "proxy"
    attention_layout = "dominant" if raw_anchor is not None else "none"
    support_kind = "raw_anchor" if raw_anchor is not None else "proxy"
    support_map_enabled = False
    reasons: List[str] = []
    suggested_subject_mode = mode

    sal_top1 = saliency.get("top_component_bbox_norm_xyxy")
    sal_top2 = saliency.get("top2_union_bbox_norm_xyxy")
    sal_fg = saliency.get("foreground_bbox_norm_xyxy")
    sal_support = _saliency_support_box(saliency)
    sal_support_kind = str(saliency.get("support_kind", "none") or "none")
    tiny_raw = 0.0 < raw_area < 0.04
    sal_dominant = bool(saliency.get("dominant_subject_likely", False))
    sal_multi = bool(saliency.get("multi_subject_likely", False))
    sal_distributed = bool(saliency.get("distributed_attention_likely", False))
    scene_primary_none = (mode in SCENE_LIKE_MODES) and (primary_idx < 0)
    raw_sal_group = _raw_saliency_group(raw_anchor, sal_top1)
    raw_sal_agreement = (
        safe_float(raw_sal_group.get("agreement_iou", 0.0), 0.0)
        if isinstance(raw_sal_group, dict)
        else (iou_xyxy(raw_anchor, sal_top1) if raw_anchor is not None and isinstance(sal_top1, list) else 1.0)
    )
    raw_sal_distance = (
        safe_float(raw_sal_group.get("center_distance", 0.0), 0.0)
        if isinstance(raw_sal_group, dict)
        else (center_distance(raw_anchor, sal_top1) if raw_anchor is not None and isinstance(sal_top1, list) else 0.0)
    )
    raw_support_mass = _support_map_mass_for_box(raw_anchor, saliency)
    severe_disagreement = bool(
        raw_anchor is not None
        and isinstance(sal_top1, list)
        and (
            (raw_sal_agreement < 0.10 and raw_sal_distance > 0.20)
            or ((raw_support_mass is not None) and raw_support_mass < 0.05)
        )
    )
    support_fallback_reason = ""

    if severe_disagreement and not mode.startswith("portrait"):
        if sal_distributed or scene_primary_none:
            candidate_anchor_box = _conservative_scene_anchor(raw_anchor=raw_anchor, saliency=saliency, image_ar=image_ar)
            support_box = sal_support or candidate_anchor_box
            effective_box = support_box
            state = "distributed_attention" if sal_distributed else "no_dominant_subject"
            attention_layout = "distributed" if sal_distributed else "none"
            score_mode = "support_map" if bool(saliency.get("support_map_available", False)) else "neutralized"
            source = "detector_saliency_severe_disagreement_soft_anchor"
            repr_type = "soft_scene_support_map" if score_mode == "support_map" else "soft_scene_region"
            placeholder_flag = score_mode != "support_map"
            subject_reliability = 0.20 if score_mode == "support_map" else 0.12
            subject_source = "fusion"
            support_kind = sal_support_kind or "candidate_proxy"
            support_map_enabled = score_mode == "support_map"
            reasons.append("detector_saliency_severe_disagreement")
        else:
            fusion_box = union_boxes([raw_anchor, sal_top1]) or sal_support or sal_top1 or fallback_box
            candidate_anchor_box = expand_box_to_min_area(fusion_box, min_area=max(0.08, box_area(fusion_box)))
            support_box = sal_support or fusion_box
            effective_box = support_box
            state = "multi_subject" if (sal_multi or raw_sal_group is not None) else "no_dominant_subject"
            attention_layout = "multi_subject" if state == "multi_subject" else "none"
            score_mode = "support_map" if bool(saliency.get("support_map_available", False)) else "neutralized"
            source = "detector_saliency_severe_disagreement_fusion"
            repr_type = "fusion_support_map" if score_mode == "support_map" else "fusion_multi_subject"
            placeholder_flag = score_mode != "support_map"
            subject_reliability = 0.22 if score_mode == "support_map" else 0.15
            subject_source = "fusion"
            support_kind = "raw_saliency_union" if raw_sal_group is not None else (sal_support_kind or "saliency_support")
            support_map_enabled = score_mode == "support_map"
            reasons.append("detector_saliency_severe_disagreement")

    elif scene_primary_none:
        if isinstance(sal_top1, list) and sal_dominant and not sal_distributed and not sal_multi:
            candidate_anchor_box = sal_top1
            support_box = sal_support or sal_top2 or sal_top1
            effective_box = expand_box_to_min_area(support_box, min_area=max(0.05, box_area(sal_top1)))
            state = "saliency_subject"
            score_mode = "subject_preservation"
            source = "saliency_scene_top1"
            repr_type = "saliency_box"
            placeholder_flag = False
            subject_reliability = clamp(0.55 + 0.40 * safe_float(saliency.get("dominance_score", 0.0), 0.0), 0.55, 0.95)
            subject_source = "saliency"
            attention_layout = "dominant"
            support_kind = sal_support_kind or "top1"
            reasons.append("scene_primary_none_saliency_dominant")
        else:
            candidate_anchor_box = _conservative_scene_anchor(raw_anchor=raw_anchor, saliency=saliency, image_ar=image_ar)
            support_box = sal_support or candidate_anchor_box
            effective_box = support_box
            state = "distributed_attention" if sal_distributed else "no_dominant_subject"
            attention_layout = "distributed" if sal_distributed else ("multi_subject" if sal_multi else "none")
            score_mode = "support_map" if bool(saliency.get("support_map_available", False)) else "neutralized"
            source = "scene_soft_anchor"
            repr_type = "soft_scene_support_map" if score_mode == "support_map" else "soft_scene_region"
            placeholder_flag = False if score_mode == "support_map" else True
            if score_mode == "support_map":
                subject_reliability = clamp(
                    0.30
                    + 0.25 * safe_float(saliency.get("top2_mass_ratio", 0.0), 0.0)
                    + 0.15 * safe_float(saliency.get("foreground_area_ratio", 0.0), 0.0),
                    0.30,
                    0.58,
                )
                support_map_enabled = True
            else:
                subject_reliability = 0.22 if support_box != candidate_anchor_box else (0.18 if saliency.get("available", False) else 0.08)
            subject_source = "saliency" if saliency.get("available", False) else "proxy"
            support_kind = sal_support_kind or ("candidate_proxy" if support_box == candidate_anchor_box else "support_bbox")
            reasons.append("scene_primary_none_support_map" if score_mode == "support_map" else "scene_primary_none_soft_anchor")
    elif mode == "other_ambiguous" and tiny_raw:
        if isinstance(raw_sal_group, dict):
            group_box = raw_sal_group["union_box"]
            candidate_anchor_box = expand_box_to_min_area(group_box, min_area=max(0.08, raw_sal_group["union_area"]))
            support_box = group_box
            effective_box = group_box
            state = "multi_subject"
            attention_layout = "multi_subject"
            score_mode = "subject_preservation"
            source = "fusion_raw_saliency_union"
            repr_type = "fusion_multi_subject"
            placeholder_flag = False
            subject_reliability = clamp(0.52 + 0.18 * safe_float(saliency.get("dominance_score", 0.0), 0.0) + 0.10 * safe_float(saliency.get("top2_mass_ratio", 0.0), 0.0), 0.52, 0.78)
            subject_source = "fusion"
            support_kind = "raw_saliency_union"
            reasons.append("ambiguous_tiny_raw_saliency_group_union")
        elif sal_distributed and isinstance(sal_support, list):
            candidate_anchor_box = _conservative_scene_anchor(raw_anchor=raw_anchor, saliency=saliency, image_ar=image_ar)
            support_box = sal_support
            effective_box = support_box
            state = "distributed_attention"
            attention_layout = "distributed"
            score_mode = "support_map" if bool(saliency.get("support_map_available", False)) else "neutralized"
            source = "saliency_ambiguous_soft_anchor"
            suggested_subject_mode = "scene_general"
            repr_type = "soft_scene_support_map" if score_mode == "support_map" else "soft_scene_region"
            placeholder_flag = False if score_mode == "support_map" else True
            subject_reliability = 0.28 if score_mode == "support_map" else 0.20
            subject_source = "saliency"
            support_kind = sal_support_kind or "support_bbox"
            support_map_enabled = score_mode == "support_map"
            reasons.append("ambiguous_tiny_raw_saliency_support_map" if score_mode == "support_map" else "ambiguous_tiny_raw_saliency_soft_anchor")
        elif sal_dominant and isinstance(sal_top1, list):
            candidate_anchor_box = sal_top1
            support_box = sal_support or sal_top1
            effective_box = expand_box_to_min_area(support_box, min_area=max(0.04, raw_area))
            state = "saliency_subject"
            attention_layout = "dominant"
            score_mode = "subject_preservation"
            source = "saliency_ambiguous_top1"
            repr_type = "saliency_box"
            placeholder_flag = False
            subject_reliability = clamp(0.58 + 0.35 * safe_float(saliency.get("dominance_score", 0.0), 0.0), 0.58, 0.95)
            subject_source = "saliency"
            support_kind = sal_support_kind or "top1"
            reasons.append("ambiguous_tiny_raw_saliency_dominant")
        else:
            candidate_anchor_box = _conservative_scene_anchor(raw_anchor=raw_anchor, saliency=saliency, image_ar=image_ar)
            support_box = candidate_anchor_box
            effective_box = support_box
            state = "no_dominant_subject"
            attention_layout = "none"
            score_mode = "neutralized"
            source = "ambiguous_tiny_raw_conservative"
            suggested_subject_mode = "scene_general"
            repr_type = "soft_scene_region"
            placeholder_flag = True
            subject_reliability = 0.10
            subject_source = "proxy"
            support_kind = "candidate_proxy"
            reasons.append("ambiguous_tiny_raw_no_saliency_subject")
    elif tiny_raw and sal_dominant and isinstance(sal_top1, list):
        if isinstance(raw_sal_group, dict):
            group_box = raw_sal_group["union_box"]
            candidate_anchor_box = expand_box_to_min_area(group_box, min_area=max(0.08, raw_sal_group["union_area"]))
            support_box = group_box
            effective_box = group_box
            state = "multi_subject"
            attention_layout = "multi_subject"
            score_mode = "subject_preservation"
            source = "fusion_tiny_raw_saliency_union"
            repr_type = "fusion_multi_subject"
            placeholder_flag = False
            subject_reliability = clamp(0.54 + 0.18 * safe_float(saliency.get("dominance_score", 0.0), 0.0), 0.54, 0.78)
            subject_source = "fusion"
            support_kind = "raw_saliency_union"
            reasons.append("tiny_raw_saliency_group_union")
        elif (raw_anchor is None) or (iou_xyxy(raw_anchor, sal_top1) < 0.20) or (box_area(sal_top1) >= max(0.04, raw_area * 2.0)):
            candidate_anchor_box = sal_top1
            support_box = sal_support or sal_top1
            effective_box = expand_box_to_min_area(support_box, min_area=max(0.04, raw_area))
            state = "saliency_subject"
            attention_layout = "dominant"
            score_mode = "subject_preservation"
            source = "saliency_tiny_raw_override"
            repr_type = "saliency_box"
            placeholder_flag = False
            subject_reliability = clamp(0.56 + 0.35 * safe_float(saliency.get("dominance_score", 0.0), 0.0), 0.56, 0.95)
            subject_source = "saliency"
            support_kind = sal_support_kind or "top1"
            reasons.append("tiny_raw_saliency_override")

    candidate_support_mass = _support_map_mass_for_box(candidate_anchor_box, saliency)
    effective_support_mass = _support_map_mass_for_box(effective_box, saliency)
    support_area = box_area(support_box)
    should_support_fallback = bool(
        score_mode == "support_map"
        and support_map_enabled
        and (
            (effective_support_mass is not None and effective_support_mass < 0.15)
            or (
                effective_support_mass is not None
                and candidate_support_mass is not None
                and (effective_support_mass + 0.10) < candidate_support_mass
            )
            or (mode in SCENE_LIKE_MODES and support_area < 0.03)
        )
    )
    if should_support_fallback:
        if effective_support_mass is not None and effective_support_mass < 0.15:
            support_fallback_reason = "support_mass_lt_0_15"
        elif effective_support_mass is not None and candidate_support_mass is not None and (effective_support_mass + 0.10) < candidate_support_mass:
            support_fallback_reason = "support_mass_much_worse_than_candidate"
        else:
            support_fallback_reason = "support_area_too_small"
        support_box = candidate_anchor_box
        effective_box = candidate_anchor_box
        score_mode = "subject_preservation"
        support_map_enabled = False
        support_kind = "candidate_anchor_fallback"
        subject_reliability = min(subject_reliability, 0.25 if mode in SCENE_LIKE_MODES else 0.35)
        repr_type = "candidate_anchor_fallback"
        reasons.append(f"support_fallback:{support_fallback_reason}")
        effective_support_mass = candidate_support_mass

    candidate_anchor_box = clip_box01(candidate_anchor_box)
    support_box = clip_box01(support_box)
    effective_box = clip_box01(effective_box)
    anchor_cx, anchor_cy = box_center(candidate_anchor_box)
    anchor_w, anchor_h = box_wh(candidate_anchor_box)
    return {
        "available": True,
        "state": state,
        "score_mode": score_mode,
        "source": source,
        "reasons": reasons,
        "suggested_subject_mode": suggested_subject_mode,
        "raw_anchor_bbox_norm_xyxy": raw_anchor,
        "candidate_anchor_bbox_norm_xyxy": candidate_anchor_box,
        "candidate_anchor_centroid": [round(anchor_cx, 6), round(anchor_cy, 6)],
        "candidate_anchor_size": [round(anchor_w, 6), round(anchor_h, 6)],
        "support_bbox_norm_xyxy": support_box,
        "support_kind": support_kind,
        "support_map_enabled": bool(support_map_enabled),
        "effective_bbox_norm_xyxy": effective_box,
        "scoring_bbox_norm_xyxy": effective_box,
        "subject_repr_type": repr_type,
        "placeholder_flag": bool(placeholder_flag),
        "subject_reliability": round(float(clamp(subject_reliability, 0.0, 1.0)), 6),
        "subject_source": subject_source,
        "attention_layout": attention_layout,
        "subject_agreement_iou": round(float(clamp(raw_sal_agreement, 0.0, 1.0)), 6),
        "subject_center_distance": round(float(max(0.0, raw_sal_distance)), 6),
        "subject_disagreement_severe": bool(severe_disagreement),
        "raw_anchor_support_mass": None if raw_support_mass is None else round(float(raw_support_mass), 6),
        "candidate_anchor_support_mass": None if candidate_support_mass is None else round(float(candidate_support_mass), 6),
        "effective_support_mass": None if effective_support_mass is None else round(float(effective_support_mass), 6),
        "support_fallback_reason": support_fallback_reason,
        "saliency_summary": saliency,
    }
