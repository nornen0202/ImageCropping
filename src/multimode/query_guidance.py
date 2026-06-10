from __future__ import annotations

from typing import List, Sequence, Tuple

from subject_region import box_area, box_center, clip_box01

from .query_builder import ModeQuery


def _expanded_box(box: Sequence[float], scale_x: float, scale_y: float) -> List[float]:
    cx, cy = box_center(box)
    bw = max(1e-6, float(box[2]) - float(box[0]))
    bh = max(1e-6, float(box[3]) - float(box[1]))
    return clip_box01(
        [
            cx - 0.5 * bw * float(scale_x),
            cy - 0.5 * bh * float(scale_y),
            cx + 0.5 * bw * float(scale_x),
            cy + 0.5 * bh * float(scale_y),
        ]
    )


def _merge_query_boxes(primary: Sequence[float], secondary: Sequence[float] | None) -> List[float]:
    if secondary is None or box_area(secondary) <= 0.0:
        return [float(v) for v in primary]
    x1 = min(float(primary[0]), float(secondary[0]))
    y1 = min(float(primary[1]), float(secondary[1]))
    x2 = max(float(primary[2]), float(secondary[2]))
    y2 = max(float(primary[3]), float(secondary[3]))
    return clip_box01([x1, y1, x2, y2])


def build_query_guidance(query: ModeQuery) -> ModeQuery:
    if query.mode_name == "face" and query.face_bbox_norm_xyxy is not None:
        core = list(query.head_bbox_norm_xyxy or query.face_bbox_norm_xyxy)
        env = _expanded_box(core, 1.35, 1.55)
        query.anchor_bbox_norm_xyxy = list(core)
    elif query.mode_name.startswith("single_person"):
        core = list(query.anchor_bbox_norm_xyxy)
        portrait_comp = query.attributes.get("portrait_comp") if isinstance(query.attributes.get("portrait_comp"), dict) else {}
        shot_type = str(portrait_comp.get("shot_type", "unknown") or "unknown")
        if shot_type == "headshot":
            env = _expanded_box(core, 1.24, 1.30)
        elif shot_type == "half":
            env = _expanded_box(core, 1.20, 1.24)
        elif shot_type in {"three_quarter", "full"}:
            env = _expanded_box(core, 1.14, 1.22)
        else:
            env = _expanded_box(core, 1.18, 1.18)
    elif query.mode_name.startswith("group"):
        core = list(query.anchor_bbox_norm_xyxy)
        env = _expanded_box(core, 1.10, 1.14)
    elif query.mode_name.startswith("object_single"):
        core = list(query.anchor_bbox_norm_xyxy)
        env = _expanded_box(core, 1.12, 1.12)
    elif query.mode_name.startswith("object_multi"):
        core = list(query.anchor_bbox_norm_xyxy)
        env = _expanded_box(core, 1.08, 1.08)
    else:
        core = list(query.core_bbox_norm_xyxy)
        env = list(query.envelope_bbox_norm_xyxy)
    query.core_bbox_norm_xyxy = clip_box01(core)
    query.envelope_bbox_norm_xyxy = clip_box01(_merge_query_boxes(env, query.support_bbox_norm_xyxy))
    query.anchor_bbox_norm_xyxy = clip_box01(query.anchor_bbox_norm_xyxy or query.core_bbox_norm_xyxy)
    if box_area(query.envelope_bbox_norm_xyxy) < box_area(query.core_bbox_norm_xyxy):
        query.envelope_bbox_norm_xyxy = list(query.core_bbox_norm_xyxy)
    return query


def anchor_center_in_box(query: ModeQuery) -> Tuple[float, float]:
    return box_center(query.anchor_bbox_norm_xyxy)
