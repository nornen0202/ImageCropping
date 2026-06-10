from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from portrait_composition import portrait_seed_layouts
from subject_region import box_area, box_center, clip_box01

from .mode_catalog import MODE_SPEC_BY_NAME
from .query_builder import ModeQuery


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_target_ar(target_ar: str, image_ar: float) -> float:
    target = str(target_ar or "").strip().upper()
    if not target or target == "FREE":
        return max(1e-6, float(image_ar))
    if ":" in target:
        left, right = target.split(":", 1)
        return max(1e-6, _safe_float(left, 1.0) / max(1e-6, _safe_float(right, 1.0)))
    return max(1e-6, _safe_float(target, image_ar))


def build_global_candidate_bank(candidate_row: Dict[str, Any], target_ar: str) -> List[Dict[str, Any]]:
    candidates_by_ar = candidate_row.get("candidates_by_ar") if isinstance(candidate_row.get("candidates_by_ar"), dict) else {}
    values = candidates_by_ar.get(target_ar, [])
    if not isinstance(values, list):
        return []
    return [dict(candidate) for candidate in values if isinstance(candidate, dict)]


def _fit_crop_to_box(
    *,
    anchor_cx: float,
    anchor_cy: float,
    envelope_box: Sequence[float],
    image_ar: float,
    crop_ar: float,
    placement_x: float,
    placement_y: float = 0.5,
    area_scale: float = 1.2,
) -> List[float]:
    env_w = max(1e-6, float(envelope_box[2]) - float(envelope_box[0]))
    env_h = max(1e-6, float(envelope_box[3]) - float(envelope_box[1]))
    base_area = max(1e-5, box_area(envelope_box) * max(1.0, float(area_scale)))
    crop_w = math.sqrt(base_area * float(crop_ar))
    crop_h = crop_w / max(1e-6, float(crop_ar))
    if crop_w < env_w:
        crop_w = env_w
        crop_h = crop_w / max(1e-6, float(crop_ar))
    if crop_h < env_h:
        crop_h = env_h
        crop_w = crop_h * float(crop_ar)
    if crop_w > 1.0:
        crop_w = 1.0
        crop_h = crop_w / max(1e-6, float(crop_ar))
    if crop_h > 1.0:
        crop_h = 1.0
        crop_w = crop_h * float(crop_ar)
    return _place_crop_with_dims(
        anchor_cx=anchor_cx,
        anchor_cy=anchor_cy,
        envelope_box=envelope_box,
        crop_w=crop_w,
        crop_h=crop_h,
        placement_x=placement_x,
        placement_y=placement_y,
    )


def _place_crop_with_dims(
    *,
    anchor_cx: float,
    anchor_cy: float,
    envelope_box: Sequence[float],
    crop_w: float,
    crop_h: float,
    placement_x: float,
    placement_y: float = 0.5,
) -> List[float]:
    crop_w = max(1e-6, min(1.0, float(crop_w)))
    crop_h = max(1e-6, min(1.0, float(crop_h)))
    x1 = anchor_cx - float(placement_x) * crop_w
    y1 = anchor_cy - float(placement_y) * crop_h
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    if x1 < 0.0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0.0:
        y2 -= y1
        y1 = 0.0
    if x2 > 1.0:
        x1 -= (x2 - 1.0)
        x2 = 1.0
    if y2 > 1.0:
        y1 -= (y2 - 1.0)
        y2 = 1.0
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    crop = clip_box01([x1, y1, x2, y2])
    ex1, ey1, ex2, ey2 = [float(v) for v in envelope_box]
    cx1, cy1, cx2, cy2 = [float(v) for v in crop]
    if cx1 > ex1:
        shift = cx1 - ex1
        crop = clip_box01([cx1 - shift, cy1, cx2 - shift, cy2])
    if cy1 > ey1:
        shift = cy1 - ey1
        crop = clip_box01([crop[0], cy1 - shift, crop[2], cy2 - shift])
    if crop[2] < ex2:
        shift = ex2 - crop[2]
        crop = clip_box01([crop[0] + shift, crop[1], crop[2] + shift, crop[3]])
    if crop[3] < ey2:
        shift = ey2 - crop[3]
        crop = clip_box01([crop[0], crop[1] + shift, crop[2], crop[3] + shift])
    return crop


def _maximal_inset_crop(crop_ar_norm: float) -> List[float]:
    crop_ar_norm = max(1e-6, float(crop_ar_norm))
    if crop_ar_norm >= 1.0:
        crop_w = 1.0
        crop_h = crop_w / crop_ar_norm
    else:
        crop_h = 1.0
        crop_w = crop_h * crop_ar_norm
    x1 = 0.5 * (1.0 - crop_w)
    y1 = 0.5 * (1.0 - crop_h)
    return clip_box01([x1, y1, x1 + crop_w, y1 + crop_h])


def _seed_scale_ladder(mode_name: str) -> List[float]:
    if mode_name == "face":
        return [1.0, 1.16, 1.34]
    if mode_name == "landscape":
        return [1.0, 1.18, 1.40, 1.70]
    if mode_name.startswith("single_person"):
        return [1.0, 1.14, 1.30, 1.48, 1.70]
    if mode_name.startswith("group"):
        return [1.0, 1.12, 1.26, 1.42, 1.60]
    if mode_name.startswith("object_single"):
        return [1.0, 1.12, 1.28, 1.46, 1.66]
    if mode_name.startswith("object_multi"):
        return [1.0, 1.10, 1.24, 1.38, 1.54]
    return [1.0, 1.15, 1.30, 1.50]


def _synthetic_seed_boxes(query: ModeQuery, image_ar: float) -> Iterable[Dict[str, Any]]:
    spec = MODE_SPEC_BY_NAME[query.mode_name]
    target_pixel_ar = parse_target_ar(query.target_ar, image_ar)
    crop_ar_norm = max(1e-6, float(target_pixel_ar) / max(1e-6, float(image_ar)))
    envelope = query.envelope_bbox_norm_xyxy
    expand_scales = _seed_scale_ladder(query.mode_name)
    portrait_layout_rows = portrait_seed_layouts(query.attributes.get("portrait_comp"), query.mode_name)
    if portrait_layout_rows and (query.mode_name.startswith("single_person") or query.mode_name.startswith("group")):
        for family_idx, layout in enumerate(portrait_layout_rows, start=1):
            anchor_xy = layout.get("anchor_xy")
            if not isinstance(anchor_xy, (list, tuple)) or len(anchor_xy) != 2:
                continue
            anchor_cx = _safe_float(anchor_xy[0], 0.5)
            anchor_cy = _safe_float(anchor_xy[1], 0.5)
            placement_x_values = layout.get("placement_x_values")
            if not isinstance(placement_x_values, list) or not placement_x_values:
                placement_x_values = [0.5]
            placement_y_values = layout.get("placement_y_values")
            if not isinstance(placement_y_values, list) or not placement_y_values:
                placement_y_values = [0.5]
            family = str(layout.get("family", f"portrait_{family_idx}") or f"portrait_{family_idx}")
            priority_boost = 2.0 * _safe_float(layout.get("priority", 0.0), 0.0)
            for placement_x in placement_x_values:
                for placement_y in placement_y_values:
                    px = _safe_float(placement_x, 0.5)
                    py = _safe_float(placement_y, 0.5)
                    base_crop = _fit_crop_to_box(
                        anchor_cx=anchor_cx,
                        anchor_cy=anchor_cy,
                        envelope_box=envelope,
                        image_ar=image_ar,
                        crop_ar=crop_ar_norm,
                        placement_x=px,
                        placement_y=py,
                        area_scale=1.0,
                    )
                    base_w = max(1e-6, float(base_crop[2]) - float(base_crop[0]))
                    base_h = max(1e-6, float(base_crop[3]) - float(base_crop[1]))
                    for scale_idx, expand_scale in enumerate(expand_scales, start=1):
                        crop = _place_crop_with_dims(
                            anchor_cx=anchor_cx,
                            anchor_cy=anchor_cy,
                            envelope_box=envelope,
                            crop_w=base_w * float(expand_scale),
                            crop_h=base_h * float(expand_scale),
                            placement_x=px,
                            placement_y=py,
                        )
                        yield {
                            "candidate_id": f"mm::{query.query_id}::seed_{family}_{int(round(px * 100)):02d}_{int(round(py * 100)):02d}_{scale_idx}",
                            "bbox_norm_xyxy": crop,
                            "ar": target_pixel_ar,
                            "area_ratio": box_area(crop),
                            "source": "multimode_seed",
                            "must_keep": False,
                            "priority": 92 + priority_boost - scale_idx,
                            "teacher_derived": False,
                            "source_lineage": ["multimode_seed", family],
                        }
    else:
        anchor_cx, anchor_cy = box_center(query.anchor_bbox_norm_xyxy)
        placements = [0.5]
        if spec.placement == "rot":
            placements = [1.0 / 3.0, 2.0 / 3.0]
        for placement_x in placements:
            base_crop = _fit_crop_to_box(
                anchor_cx=anchor_cx,
                anchor_cy=anchor_cy,
                envelope_box=envelope,
                image_ar=image_ar,
                crop_ar=crop_ar_norm,
                placement_x=placement_x,
                placement_y=0.5,
                area_scale=1.0,
            )
            base_w = max(1e-6, float(base_crop[2]) - float(base_crop[0]))
            base_h = max(1e-6, float(base_crop[3]) - float(base_crop[1]))
            for scale_idx, expand_scale in enumerate(expand_scales, start=1):
                crop = _place_crop_with_dims(
                    anchor_cx=anchor_cx,
                    anchor_cy=anchor_cy,
                    envelope_box=envelope,
                    crop_w=base_w * float(expand_scale),
                    crop_h=base_h * float(expand_scale),
                    placement_x=placement_x,
                    placement_y=0.5,
                )
                yield {
                    "candidate_id": f"mm::{query.query_id}::seed_{int(round(placement_x * 100)):02d}_{scale_idx}",
                    "bbox_norm_xyxy": crop,
                    "ar": target_pixel_ar,
                    "area_ratio": box_area(crop),
                    "source": "multimode_seed",
                    "must_keep": False,
                    "priority": 90 - scale_idx,
                    "teacher_derived": False,
                    "source_lineage": ["multimode_seed"],
                }
    if query.mode_name == "landscape":
        full_box = _maximal_inset_crop(crop_ar_norm)
        yield {
            "candidate_id": f"mm::{query.query_id}::seed_fullframe",
            "bbox_norm_xyxy": full_box,
            "ar": target_pixel_ar,
            "area_ratio": box_area(full_box),
            "source": "multimode_seed_full",
            "must_keep": True,
            "priority": 99,
            "teacher_derived": False,
            "source_lineage": ["multimode_seed_full"],
        }


def expand_query_candidates(
    *,
    base_candidates: Sequence[Dict[str, Any]],
    query: ModeQuery,
    image_ar: float,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for candidate in list(base_candidates) + list(_synthetic_seed_boxes(query, image_ar)):
        bbox = candidate.get("bbox_norm_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        key = tuple(round(_safe_float(v, 0.0), 6) for v in bbox)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(candidate))
    return merged
