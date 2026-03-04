from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUBJECT_MODE_TO_POLICY: Dict[str, str] = {
    "portrait_single": "portrait_single_v1",
    "portrait_group": "portrait_group_v1",
    "object_single": "object_v1",
    "object_multi": "object_multi_v1",
    "scene_landscape": "scene_v1",
    "background_texture_copyspace": "copyspace_v1",
    "text_document": "text_v1",
    "other_ambiguous": "generic_v1",
}

PEOPLE_SUPER_CATS = {"people_single", "people_group", "lifestyle_people"}
TEXT_SUPER_CATS = {"documents_text"}
SCENE_SUPER_CATS = {"landscape_nature", "architecture_exterior", "indoor_interior"}
OBJECT_SUPER_CATS = {"animals", "food", "product_object", "transportation", "sports"}

TEXT_HINTS = (
    "text",
    "document",
    "typography",
    "poster",
    "banner",
    "lettering",
    "magazine",
    "newspaper",
)
GROUP_HINTS = ("group", "family", "team", "crowd", "couple", "together")
PORTRAIT_HINTS = ("portrait", "headshot", "person", "people", "selfie", "model")
COPYSPACE_HINTS = (
    "copy space",
    "copy-space",
    "negative space",
    "background",
    "texture",
    "pattern",
    "isolated",
    "minimal",
)
SCENE_HINTS = ("landscape", "cityscape", "interior", "architecture", "scenery", "panorama")
OBJECT_HINTS = ("animal", "pet", "product", "vehicle", "car", "food", "object")
BLANK_RATIO_COPYSPACE_MIN = 0.28


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _normalize_box_xyxy(box: Sequence[Any], width: int, height: int) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    w = max(1, int(width))
    h = max(1, int(height))
    x1, y1, x2, y2 = [_safe_float(v, 0.0) for v in box]
    x1 = _clamp(x1, 0.0, float(w))
    y1 = _clamp(y1, 0.0, float(h))
    x2 = _clamp(x2, 0.0, float(w))
    y2 = _clamp(y2, 0.0, float(h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = _box_area(a) + _box_area(b) - inter
    if ua <= 1e-8:
        return 0.0
    return float(inter / ua)


def _union_box(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    x1 = min(float(b[0]) for b in boxes)
    y1 = min(float(b[1]) for b in boxes)
    x2 = max(float(b[2]) for b in boxes)
    y2 = max(float(b[3]) for b in boxes)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        if "|" in raw_tags:
            vals = [x.strip().lower() for x in raw_tags.split("|")]
            return [x for x in vals if x]
        s = raw_tags.strip().lower()
        return [s] if s else []
    if isinstance(raw_tags, (list, tuple)):
        out = []
        for x in raw_tags:
            s = str(x).strip().lower()
            if s:
                out.append(s)
        return out
    try:
        vals = list(raw_tags)
    except Exception:
        s = str(raw_tags).strip().lower()
        return [s] if s else []
    out = []
    for x in vals:
        s = str(x).strip().lower()
        if s:
            out.append(s)
    return out


def _has_any_hint(tags_norm: Sequence[str], hints: Iterable[str]) -> bool:
    if not tags_norm:
        return False
    text = " | ".join(tags_norm)
    for h in hints:
        key = str(h).strip().lower()
        if not key:
            continue
        if key in text:
            return True
    return False


def _box_area_ratio_xyxy(box: Sequence[float], width: int, height: int) -> float:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    area = _box_area(box)
    return float(_clamp(area / (w * h), 0.0, 1.0))


def _person_union_box(c3_pose: Any, width: int, height: int) -> Optional[List[float]]:
    if not isinstance(c3_pose, list):
        return None
    boxes: List[List[float]] = []
    for one in c3_pose:
        if not isinstance(one, dict):
            continue
        b = _normalize_box_xyxy(one.get("bbox"), width, height)
        if b is not None:
            boxes.append(b)
    return _union_box(boxes)


def _center_dist_norm(box: Sequence[float], width: int, height: int) -> float:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    dx = cx - 0.5 * w
    dy = cy - 0.5 * h
    d = math.sqrt(dx * dx + dy * dy)
    max_d = math.sqrt((0.5 * w) ** 2 + (0.5 * h) ** 2) + 1e-6
    return float(_clamp(d / max_d, 0.0, 1.0))


def _border_touch(box: Sequence[float], width: int, height: int, margin_px: float = 2.0) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    touch = 0.0
    touch += 1.0 if x1 <= margin_px else 0.0
    touch += 1.0 if y1 <= margin_px else 0.0
    touch += 1.0 if x2 >= (w - margin_px) else 0.0
    touch += 1.0 if y2 >= (h - margin_px) else 0.0
    return float(_clamp(touch / 4.0, 0.0, 1.0))


def _area_pref(area_ratio: float, mu: float = 0.18, sigma: float = 0.12) -> float:
    ar = max(0.0, float(area_ratio))
    return float(math.exp(-((ar - mu) ** 2) / (2.0 * sigma * sigma + 1e-12)))


def _is_bg_like(area_ratio: float, border_touch: float, box: Sequence[float], width: int, height: int) -> bool:
    if area_ratio > 0.75 and border_touch >= 0.5:
        return True
    bw = max(1e-6, float(box[2]) - float(box[0]))
    bh = max(1e-6, float(box[3]) - float(box[1]))
    aspect = max(bw / bh, bh / bw)
    if area_ratio > 0.60 and aspect > 3.5 and border_touch >= 0.5:
        return True
    # Extra guard: almost full-image mask is treated as background-like.
    if area_ratio > 0.90:
        return True
    return False


def enrich_c2_topn(
    *,
    c2_seg: Any,
    c2_det: Any,
    c3_pose: Any,
    width: int,
    height: int,
    top_n: int = 5,
    union_top_m: int = 3,
    allow_det_proxy: bool = True,
) -> Dict[str, Any]:
    w = max(1, int(width))
    h = max(1, int(height))
    img_area = float(max(1, w * h))

    seg_list = c2_seg if isinstance(c2_seg, list) else []
    det_list = c2_det if isinstance(c2_det, list) else []

    det_boxes: List[Tuple[List[float], float, int]] = []
    for det in det_list:
        if not isinstance(det, dict):
            continue
        b = _normalize_box_xyxy(det.get("box"), w, h)
        if b is None:
            continue
        det_boxes.append(
            (
                b,
                _safe_float(det.get("score", 0.0), 0.0),
                int(_safe_float(det.get("class_id", -1), -1)),
            )
        )

    person_union = _person_union_box(c3_pose, w, h)

    instances: List[Dict[str, Any]] = []

    for seg in seg_list:
        if not isinstance(seg, dict):
            continue
        box = _normalize_box_xyxy(seg.get("box"), w, h)
        if box is None:
            continue
        area = _safe_float(seg.get("area"), _box_area(box))
        if area <= 1.0:
            area = _box_area(box)
        area_ratio = float(_clamp(area / img_area, 0.0, 1.0))
        center_dist = _center_dist_norm(box, w, h)
        border_touch = _border_touch(box, w, h)
        bg_like = _is_bg_like(area_ratio, border_touch, box, w, h)

        det_conf = 0.0
        det_class = int(_safe_float(seg.get("class_id", -1), -1))
        for db, ds, cid in det_boxes:
            iou = _iou_xyxy(box, db)
            if iou >= 0.25 and ds > det_conf:
                det_conf = ds
                det_class = cid

        person_overlap = _iou_xyxy(box, person_union) if person_union is not None else 0.0
        area_term = _area_pref(area_ratio)
        center_term = 1.0 - center_dist
        border_term = 1.0 - border_touch
        bg_pen = 1.0 if bg_like else 0.0

        importance = (
            0.80 * det_conf
            + 0.40 * area_term
            + 0.20 * center_term
            + 0.20 * border_term
            + 0.60 * person_overlap
            - 1.00 * bg_pen
        )
        if det_class == 0:
            importance += 0.20

        instances.append(
            {
                **seg,
                "box": [round(v, 3) for v in box],
                "class_id": int(det_class),
                "area": int(round(area)),
                "area_ratio": round(area_ratio, 6),
                "center_xy": [round(0.5 * (box[0] + box[2]), 3), round(0.5 * (box[1] + box[3]), 3)],
                "center_dist_norm": round(center_dist, 6),
                "border_touch": round(border_touch, 6),
                "bg_like": bool(bg_like),
                "importance_score": round(float(importance), 6),
                "importance_comp": {
                    "det": round(float(det_conf), 6),
                    "area_pref": round(float(area_term), 6),
                    "center": round(float(center_term), 6),
                    "border_pen": round(float(-border_touch), 6),
                    "person_overlap": round(float(person_overlap), 6),
                    "bg_penalty": float(bg_pen),
                },
                "source": str(seg.get("source", "seg")),
            }
        )

    if allow_det_proxy and len(instances) < max(1, int(top_n)):
        proxy_candidates: List[Dict[str, Any]] = []
        for db, ds, cid in det_boxes:
            area = _box_area(db)
            area_ratio = float(_clamp(area / img_area, 0.0, 1.0))
            center_dist = _center_dist_norm(db, w, h)
            border_touch = _border_touch(db, w, h)
            bg_like = _is_bg_like(area_ratio, border_touch, db, w, h)
            person_overlap = _iou_xyxy(db, person_union) if person_union is not None else 0.0
            area_term = _area_pref(area_ratio)
            center_term = 1.0 - center_dist
            border_term = 1.0 - border_touch
            bg_pen = 1.0 if bg_like else 0.0
            importance = (
                0.90 * ds
                + 0.30 * area_term
                + 0.15 * center_term
                + 0.15 * border_term
                + 0.50 * person_overlap
                - 1.00 * bg_pen
            )
            if cid == 0:
                importance += 0.20
            proxy_candidates.append(
                {
                    "box": [round(v, 3) for v in db],
                    "class_id": int(cid),
                    "score": round(float(ds), 6),
                    "mask_rle": None,
                    "area": int(round(area)),
                    "area_ratio": round(area_ratio, 6),
                    "center_xy": [round(0.5 * (db[0] + db[2]), 3), round(0.5 * (db[1] + db[3]), 3)],
                    "center_dist_norm": round(center_dist, 6),
                    "border_touch": round(border_touch, 6),
                    "bg_like": bool(bg_like),
                    "importance_score": round(float(importance), 6),
                    "importance_comp": {
                        "det": round(float(ds), 6),
                        "area_pref": round(float(area_term), 6),
                        "center": round(float(center_term), 6),
                        "border_pen": round(float(-border_touch), 6),
                        "person_overlap": round(float(person_overlap), 6),
                        "bg_penalty": float(bg_pen),
                    },
                    "source": "c2_det_proxy",
                }
            )
        proxy_candidates.sort(key=lambda x: float(x.get("importance_score", -1e9)), reverse=True)
        for p in proxy_candidates:
            pbox = p.get("box")
            if not isinstance(pbox, list):
                continue
            overlapped = False
            for ex in instances:
                ebox = ex.get("box")
                if isinstance(ebox, list) and _iou_xyxy(pbox, ebox) > 0.90:
                    overlapped = True
                    break
            if not overlapped:
                instances.append(p)
            if len(instances) >= max(1, int(top_n)) * 2:
                break

    instances.sort(key=lambda x: float(x.get("importance_score", -1e9)), reverse=True)
    if top_n > 0:
        instances = instances[: int(top_n)]

    c2_primary_idx = 0 if instances else -1
    gap_1_2 = 0.0
    if len(instances) >= 2:
        gap_1_2 = float(instances[0]["importance_score"]) - float(instances[1]["importance_score"])

    kept_for_union: List[List[float]] = []
    for inst in instances:
        box = inst.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        if bool(inst.get("bg_like", False)):
            continue
        kept_for_union.append([float(v) for v in box])
        if len(kept_for_union) >= max(1, int(union_top_m)):
            break
    if not kept_for_union:
        for inst in instances[: max(1, int(union_top_m))]:
            box = inst.get("box")
            if isinstance(box, list) and len(box) == 4:
                kept_for_union.append([float(v) for v in box])

    c2_union = _union_box(kept_for_union)
    if c2_union is not None:
        c2_union = [round(v, 3) for v in c2_union]

    stats = {
        "num_raw_masks": int(len(seg_list)),
        "num_raw_det": int(len(det_list)),
        "num_kept": int(len(instances)),
        "importance_gap_1_2": round(float(gap_1_2), 6),
        "max_area_ratio": round(max((_safe_float(i.get("area_ratio", 0.0), 0.0) for i in instances), default=0.0), 6),
    }

    return {
        "instances": instances,
        "c2_primary_idx": int(c2_primary_idx),
        "c2_union_box_xyxy": c2_union,
        "c2_topn": int(top_n),
        "c2_stats": stats,
    }


def _infer_mode(
    *,
    tags_norm: Sequence[str],
    super_cat: str,
    c3_person_count: int,
) -> Tuple[str, float, List[str], str]:
    sc = str(super_cat or "").strip().lower()
    reasons: List[str] = []
    if sc in TEXT_SUPER_CATS or _has_any_hint(tags_norm, TEXT_HINTS):
        reasons.append("text_signal")
        return "text_document", 0.95, reasons, "rule_text"
    if c3_person_count >= 2 or _has_any_hint(tags_norm, GROUP_HINTS):
        reasons.append("group_signal")
        return "portrait_group", 0.90, reasons, "rule_group"
    if c3_person_count == 1 or sc in PEOPLE_SUPER_CATS or _has_any_hint(tags_norm, PORTRAIT_HINTS):
        reasons.append("portrait_signal")
        return "portrait_single", 0.85, reasons, "rule_portrait"
    if _has_any_hint(tags_norm, COPYSPACE_HINTS):
        reasons.append("copyspace_signal")
        return "background_texture_copyspace", 0.80, reasons, "rule_copyspace"
    if sc in SCENE_SUPER_CATS or _has_any_hint(tags_norm, SCENE_HINTS):
        reasons.append("scene_signal")
        return "scene_landscape", 0.75, reasons, "rule_scene"
    if sc in OBJECT_SUPER_CATS or _has_any_hint(tags_norm, OBJECT_HINTS):
        reasons.append("object_signal")
        return "object_single", 0.60, reasons, "rule_object"
    reasons.append("fallback")
    return "other_ambiguous", 0.30, reasons, "rule_fallback"


def _fallback_mode_after_guard(
    *,
    tags_norm: Sequence[str],
    super_cat: str,
    text_signal: bool,
    copyspace_allowed: bool,
) -> Tuple[str, float, List[str], str]:
    sc = str(super_cat or "").strip().lower()
    reasons: List[str] = []
    if text_signal and (sc in TEXT_SUPER_CATS or _has_any_hint(tags_norm, TEXT_HINTS)):
        reasons.append("fallback_text_signal")
        return "text_document", 0.55, reasons, "fallback_text"
    if copyspace_allowed:
        reasons.append("fallback_copyspace_signal")
        return "background_texture_copyspace", 0.55, reasons, "fallback_copyspace"
    if sc in SCENE_SUPER_CATS or _has_any_hint(tags_norm, SCENE_HINTS):
        reasons.append("fallback_scene_signal")
        return "scene_landscape", 0.50, reasons, "fallback_scene"
    if sc in OBJECT_SUPER_CATS or _has_any_hint(tags_norm, OBJECT_HINTS):
        reasons.append("fallback_object_signal")
        return "object_single", 0.45, reasons, "fallback_object"
    reasons.append("fallback_ambiguous")
    return "other_ambiguous", 0.30, reasons, "fallback_ambiguous"


def route_subject_mode(
    *,
    tags_norm: Sequence[str],
    super_cat: str,
    c3_pose: Any,
    c2_instances: Sequence[Dict[str, Any]],
    c2_union_box_xyxy: Optional[Sequence[float]],
    c2_primary_idx: int,
    width: int,
    height: int,
    ocr_text_boxes_count: Optional[int] = None,
    text_overlay_likely: Optional[bool] = None,
    copy_space_flag: Optional[bool] = None,
    blank_ratio_est: Optional[float] = None,
    horizon_conf: Optional[float] = None,
    symmetry_score: Optional[float] = None,
) -> Dict[str, Any]:
    c3_list = c3_pose if isinstance(c3_pose, list) else []
    num_person = len(c3_list)
    mode, conf, reasons, base_rule_id = _infer_mode(tags_norm=tags_norm, super_cat=super_cat, c3_person_count=num_person)
    router_rule_id = base_rule_id

    primary_idx_seed = int(c2_primary_idx) if c2_primary_idx is not None else -1
    primary_idx_seed = primary_idx_seed if 0 <= primary_idx_seed < len(c2_instances) else (-1 if not c2_instances else 0)
    union_box_seed = (
        list(c2_union_box_xyxy)
        if isinstance(c2_union_box_xyxy, (list, tuple)) and len(c2_union_box_xyxy) == 4
        else None
    )

    c2_primary_area_ratio_seed = 0.0
    c2_primary_bg_like_seed = False
    if 0 <= primary_idx_seed < len(c2_instances):
        c2_primary_area_ratio_seed = _safe_float(c2_instances[primary_idx_seed].get("area_ratio", 0.0), 0.0)
        c2_primary_bg_like_seed = bool(c2_instances[primary_idx_seed].get("bg_like", False))

    has_text_hint = _has_any_hint(tags_norm, TEXT_HINTS)
    has_copyspace_tag = _has_any_hint(tags_norm, COPYSPACE_HINTS)
    text_boxes = max(0, int(_safe_float(ocr_text_boxes_count, 0.0)))
    text_overlay = bool(text_overlay_likely) if text_overlay_likely is not None else False
    text_signal = bool(has_text_hint or text_overlay or text_boxes > 0 or str(super_cat or "").strip().lower() in TEXT_SUPER_CATS)
    copyspace_signal = bool(copy_space_flag) if copy_space_flag is not None else bool(has_copyspace_tag)

    if blank_ratio_est is None:
        if isinstance(union_box_seed, list) and len(union_box_seed) == 4:
            blank_ratio = 1.0 - _box_area_ratio_xyxy(union_box_seed, width, height)
        elif c2_primary_area_ratio_seed > 0.0:
            blank_ratio = 1.0 - float(c2_primary_area_ratio_seed)
        else:
            blank_ratio = 0.0
    else:
        blank_ratio = _clamp(_safe_float(blank_ratio_est, 0.0), 0.0, 1.0)
    copyspace_allowed = bool(copyspace_signal and blank_ratio >= BLANK_RATIO_COPYSPACE_MIN)

    guard_reasons: List[str] = []

    # P0 guard #1: person_count=0 cannot route to portrait_*.
    if mode.startswith("portrait") and num_person <= 0:
        guard_reasons.append("guard_no_person_for_portrait")
        mode, conf, fb_reasons, fb_rule_id = _fallback_mode_after_guard(
            tags_norm=tags_norm,
            super_cat=super_cat,
            text_signal=text_signal,
            copyspace_allowed=copyspace_allowed,
        )
        reasons.extend(guard_reasons + fb_reasons)
        router_rule_id = f"{base_rule_id}|{fb_rule_id}"

    # P0 guard #2: no OCR/text evidence should not route text_document
    # (except explicit text super-category).
    sc = str(super_cat or "").strip().lower()
    if mode == "text_document" and (sc not in TEXT_SUPER_CATS) and (not text_signal):
        guard_reasons.append("guard_no_text_signal")
        mode, conf, fb_reasons, fb_rule_id = _fallback_mode_after_guard(
            tags_norm=tags_norm,
            super_cat=super_cat,
            text_signal=False,
            copyspace_allowed=copyspace_allowed,
        )
        reasons.extend(["guard_no_text_signal"] + fb_reasons)
        router_rule_id = f"{base_rule_id}|{fb_rule_id}"

    # P0 guard #3: copy-space tag only is insufficient when blank ratio is low.
    if mode == "background_texture_copyspace" and copyspace_signal and (blank_ratio < BLANK_RATIO_COPYSPACE_MIN):
        guard_reasons.append("guard_low_blank_ratio_for_copyspace")
        mode, conf, fb_reasons, fb_rule_id = _fallback_mode_after_guard(
            tags_norm=tags_norm,
            super_cat=super_cat,
            text_signal=text_signal,
            copyspace_allowed=False,
        )
        reasons.extend(["guard_low_blank_ratio_for_copyspace"] + fb_reasons)
        router_rule_id = f"{base_rule_id}|{fb_rule_id}"

    # Finalize subject source/union after guard-corrected mode.
    primary_idx = int(primary_idx_seed)
    primary_source = "c2" if primary_idx >= 0 else "none"
    union_box = list(union_box_seed) if isinstance(union_box_seed, list) else None
    multi_subject = False
    if mode.startswith("portrait"):
        primary_source = "c3" if num_person > 0 else primary_source
        if mode == "portrait_group":
            multi_subject = True
            c3_union = _person_union_box(c3_list, width, height)
            if c3_union is not None:
                union_box = [round(v, 3) for v in c3_union]
    elif mode.startswith("object"):
        if len(c2_instances) >= 2:
            s1 = _safe_float(c2_instances[0].get("importance_score", 0.0), 0.0)
            s2 = _safe_float(c2_instances[1].get("importance_score", 0.0), 0.0)
            iou12 = _iou_xyxy(c2_instances[0].get("box", [0, 0, 0, 0]), c2_instances[1].get("box", [0, 0, 0, 0]))
            if (s1 - s2) < 0.15 and iou12 < 0.75:
                mode = "object_multi"
                conf = max(conf, 0.70)
                reasons.append("top2_close_multi_override")
                multi_subject = True
                union_box = _union_box(
                    [
                        [float(v) for v in c2_instances[0].get("box", [0, 0, 0, 0])],
                        [float(v) for v in c2_instances[1].get("box", [0, 0, 0, 0])],
                    ]
                )
                if union_box is not None:
                    union_box = [round(v, 3) for v in union_box]
        if mode == "object_multi":
            multi_subject = True
    elif mode in {"scene_landscape", "background_texture_copyspace", "text_document"}:
        primary_idx = -1
        primary_source = "none"

    c2_primary_area_ratio = 0.0
    c2_primary_bg_like = False
    if 0 <= primary_idx < len(c2_instances):
        c2_primary_area_ratio = _safe_float(c2_instances[primary_idx].get("area_ratio", 0.0), 0.0)
        c2_primary_bg_like = bool(c2_instances[primary_idx].get("bg_like", False))

    flags = {
        "has_person": bool(num_person > 0),
        "has_text_heavy": bool(mode == "text_document"),
        "has_copyspace_tag": bool(has_copyspace_tag),
        "is_background_like": bool(0 <= primary_idx < len(c2_instances) and c2_primary_bg_like),
    }

    shot_type: Optional[str] = None
    if mode == "portrait_group":
        shot_type = "group"
    elif mode == "portrait_single":
        if num_person > 0:
            c3_union = _person_union_box(c3_list, width, height)
            if c3_union is not None:
                ar = _box_area(c3_union) / float(max(1, int(width) * int(height)))
                if ar < 0.18:
                    shot_type = "headshot"
                elif ar < 0.45:
                    shot_type = "half"
                else:
                    shot_type = "full"
            if shot_type is None:
                shot_type = "half"
        else:
            shot_type = "unknown"

    primary_subject_type = "other"
    if mode.startswith("portrait"):
        primary_subject_type = "person"
    elif mode.startswith("object"):
        primary_subject_type = "object"
    elif mode == "text_document":
        primary_subject_type = "text"
    elif mode in {"scene_landscape", "background_texture_copyspace"}:
        primary_subject_type = "scene"

    mode_conflict = False
    if mode.startswith("portrait") and not flags["has_person"] and sc not in PEOPLE_SUPER_CATS:
        mode_conflict = True
    if mode == "text_document" and sc not in TEXT_SUPER_CATS and not _has_any_hint(tags_norm, TEXT_HINTS):
        mode_conflict = True
    if guard_reasons:
        mode_conflict = True

    if not router_rule_id:
        router_rule_id = "rule_unknown"

    router_signals = {
        "super_cat": sc,
        "num_person": int(num_person),
        "c2_num_instances": int(len(c2_instances)),
        "c2_primary_bg_like": bool(c2_primary_bg_like),
        "c2_primary_area_ratio": round(float(c2_primary_area_ratio), 6),
        "has_text_hint": bool(has_text_hint),
        "ocr_text_boxes": int(text_boxes),
        "text_overlay_likely": bool(text_overlay),
        "text_signal": bool(text_signal),
        "has_copyspace_tag": bool(has_copyspace_tag),
        "copy_space_flag": bool(copyspace_signal),
        "blank_ratio_est": round(float(blank_ratio), 6),
        "blank_ratio_thr": float(BLANK_RATIO_COPYSPACE_MIN),
        "copyspace_allowed": bool(copyspace_allowed),
        "horizon_conf": round(float(_safe_float(horizon_conf, 0.0)), 6),
        "symmetry_score": round(float(_safe_float(symmetry_score, 0.0)), 6),
    }

    return {
        "subject_mode": mode,
        "subject_mode_conf": round(float(conf), 6),
        "subject_mode_reasons": reasons,
        "subject_mode_flags": flags,
        "subject_mode_conflict": bool(mode_conflict),
        "shot_type": shot_type,
        "primary_subject_type": primary_subject_type,
        "primary_subject_source": primary_source,
        "subject_set": {
            "num_person": int(num_person),
            "num_subject_inst": int(len(c2_instances)),
            "union_box_xyxy": union_box,
            "primary_idx": int(primary_idx),
            "c2_primary_area_ratio": round(float(c2_primary_area_ratio), 6),
            "c2_primary_bg_like": bool(c2_primary_bg_like),
            "multi_subject": bool(multi_subject or num_person >= 2),
        },
        "policy_id": SUBJECT_MODE_TO_POLICY.get(mode, "generic_v1"),
        "router_rule_id": str(router_rule_id),
        "router_signals": router_signals,
    }
