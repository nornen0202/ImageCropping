from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from saliency_semantic import build_saliency_semantic_summary
from subject_region import resolve_effective_subject_region, summarize_saliency_signal


SUBJECT_MODE_TO_POLICY: Dict[str, str] = {
    "portrait_single": "portrait_single_v1",
    "portrait_group": "portrait_group_v1",
    "object_single": "object_v1",
    "object_multi": "object_multi_v1",
    "scene_general": "scene_v1",
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
COPYSPACE_STRONG_HINTS = (
    "copy space",
    "copy-space",
    "negative space",
    "texture",
    "pattern",
    "isolated",
    "minimal",
    "minimalist",
    "white background",
    "blank background",
    "plain background",
    "empty background",
    "wallpaper",
)
COPYSPACE_WEAK_HINTS = ("background",)
COPYSPACE_HINTS = COPYSPACE_STRONG_HINTS + COPYSPACE_WEAK_HINTS
TEXT_STRONG_HINTS = (
    "document",
    "page",
    "sheet of paper",
    "piece of paper",
    "book",
    "notebook",
    "magazine",
    "newspaper",
    "menu",
    "poster",
    "brochure",
    "flyer",
    "letter",
    "receipt",
    "invoice",
    "passport",
    "certificate",
    "map",
)
TEXT_WEAK_HINTS = (
    "text",
    "typography",
    "banner",
    "billboard",
    "sign",
    "label",
    "screen",
    "display",
)
SCENE_HINTS = (
    "landscape",
    "cityscape",
    "interior",
    "architecture",
    "scenery",
    "panorama",
    "street",
    "road",
    "sidewalk",
    "beach",
    "ocean",
    "sea",
    "lake",
    "mountain",
    "forest",
    "skyline",
    "room",
    "water",
    "river",
    "park",
    "bridge",
    "tower",
)
OBJECT_HINTS = (
    "animal",
    "pet",
    "product",
    "vehicle",
    "car",
    "food",
    "object",
    "dog",
    "cat",
    "bird",
    "flower",
    "plant",
    "bottle",
    "cup",
    "phone",
    "watch",
    "bag",
    "shoe",
    "chair",
    "table",
    "fruit",
    "cake",
    "pizza",
)
CAPTION_PEOPLE_HINTS = (
    "person",
    "people",
    "woman",
    "man",
    "girl",
    "boy",
    "child",
    "children",
    "lady",
    "guy",
    "bride",
    "groom",
    "worker",
    "rider",
    "surfer",
    "skater",
    "player",
    "model",
)
CAPTION_GROUP_HINTS = GROUP_HINTS + (
    "friends",
    "men",
    "women",
    "children",
    "two people",
    "three people",
    "several people",
    "group of",
)
CAPTION_PORTRAIT_HINTS = PORTRAIT_HINTS + (
    "close up",
    "close-up",
    "face",
    "smiling at the camera",
    "looking at the camera",
)
BLANK_RATIO_COPYSPACE_MIN = 0.28
BLANK_RATIO_COPYSPACE_STRONG = 0.40
SCENE_SCORE_MIN = 0.35
OBJECT_DOMINANCE_MIN = 0.18
FOREGROUND_MASS_MIN = 0.12
CONTEXTUAL_TINY_HUMAN_SINGLE_AREA_MAX = 0.03
CONTEXTUAL_TINY_HUMAN_GROUP_AREA_MAX = 0.05
CONTEXTUAL_TINY_HUMAN_SCENE_SCORE_MIN = 0.30
CONTEXTUAL_TINY_HUMAN_BLANK_RATIO_MIN = 0.85
CONTEXTUAL_TINY_HUMAN_FOREGROUND_MASS_MAX = 0.10
PORTRAIT_TINY_SUBJECT_UNKNOWN_SHOT_MAX = 0.08
PERSON_CLASS_IDS = {0}
PERSON_BOX_DEDUP_IOU = 0.60
PERSON_BOX_DEDUP_CONTAINMENT = 0.82
PERSON_BOX_MAX_WIDTH_TO_HEIGHT = 1.35
PERSON_DET_SCORE_CONFIRM_MIN = 0.65
POSE_SCORE_CONFIRM_MIN = 0.55
POSE_SCORE_PERSON_MIN = 0.30
POSE_FACE_SCORE_MIN = 0.20
POSE_C2_SUPPORT_IOU_MIN = 0.10
PORTRAIT_OBJECT_OVERRIDE_PERSON_AREA_MAX = 0.12
PORTRAIT_OBJECT_OVERRIDE_DOM_MIN = 0.16
PORTRAIT_OBJECT_OVERRIDE_UNION_MIN = 0.14
PORTRAIT_OBJECT_OVERRIDE_REL_GAIN = 1.60
SCENE_SUBTYPE_UNKNOWN = "scene_general_unknown"
SCENE_HORIZON_TARGETS: Dict[str, List[float]] = {
    "scene_landscape_nature": [1.0 / 3.0, 2.0 / 3.0],
    "scene_reflection_symmetry": [0.5],
    "scene_city_architecture": [1.0 / 3.0, 0.5, 2.0 / 3.0],
    "scene_interior_architecture": [],
    "scene_structural_pattern": [],
    "scene_contextual_object": [],
    SCENE_SUBTYPE_UNKNOWN: [],
}

TRUST_TIER_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


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


def _intersection_ratio_to_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    denom = min(_box_area(a), _box_area(b))
    if denom <= 1e-8:
        return 0.0
    return float(inter / denom)


def _box_width_to_height(box: Sequence[float]) -> float:
    w = max(1e-6, float(box[2]) - float(box[0]))
    h = max(1e-6, float(box[3]) - float(box[1]))
    return float(w / h)


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


def _normalize_free_text(raw_text: Any) -> str:
    if raw_text is None:
        return ""
    text = str(raw_text).replace("\n", " ").replace("_", " ").replace("-", " ")
    return " ".join(text.split()).strip().lower()


def _text_has_any(text: str, hints: Iterable[str]) -> bool:
    blob = _normalize_free_text(text)
    if not blob:
        return False
    for hint in hints:
        key = _normalize_free_text(hint)
        if key and key in blob:
            return True
    return False


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(math.fsum(float(v) for v in values) / float(len(values)))


def _box_area_ratio_xyxy(box: Sequence[float], width: int, height: int) -> float:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    area = _box_area(box)
    return float(_clamp(area / (w * h), 0.0, 1.0))


def _infer_caption_semantics(caption_text: Any) -> Dict[str, Any]:
    text = _normalize_free_text(caption_text)
    has_people = _text_has_any(text, CAPTION_PEOPLE_HINTS)
    has_group = _text_has_any(text, CAPTION_GROUP_HINTS)
    has_portrait = _text_has_any(text, CAPTION_PORTRAIT_HINTS)
    has_scene = _text_has_any(text, SCENE_HINTS)
    has_object = _text_has_any(text, OBJECT_HINTS)
    has_text_strong = _text_has_any(text, TEXT_STRONG_HINTS)
    has_text_weak = _text_has_any(text, TEXT_WEAK_HINTS)
    has_copyspace = _text_has_any(text, COPYSPACE_STRONG_HINTS) or (
        "background" in text and _text_has_any(text, ("white", "blank", "plain", "empty", "isolated", "minimal"))
    )
    text_backed = bool(
        has_text_strong
        or _text_has_any(
            text,
            (
                "text on",
                "words on",
                "covered in text",
                "filled with text",
                "page of",
                "sign with text",
                "menu on",
            ),
        )
    )
    return {
        "available": bool(text),
        "text": text,
        "people_hint": bool(has_people),
        "group_hint": bool(has_group),
        "portrait_hint": bool(has_portrait or (has_people and _text_has_any(text, ("close up", "selfie", "face")))),
        "scene_hint": bool(has_scene),
        "object_hint": bool(has_object),
        "copyspace_hint": bool(has_copyspace),
        "text_hint": bool(has_text_strong or has_text_weak),
        "text_backed_evidence": bool(text_backed),
    }


def _summarize_object_layout(
    c2_instances: Sequence[Dict[str, Any]],
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    fg_rows: List[Dict[str, Any]] = []
    for inst in c2_instances:
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        box = _normalize_box_xyxy(inst.get("box"), width, height)
        if box is None:
            continue
        class_id = int(_safe_float(inst.get("class_id", -1), -1))
        if class_id in PERSON_CLASS_IDS:
            continue
        area_ratio = _safe_float(inst.get("area_ratio", _box_area_ratio_xyxy(box, width, height)), 0.0)
        importance = _safe_float(inst.get("importance_score", 0.0), 0.0)
        if area_ratio < 0.01 and importance < 0.20:
            continue
        fg_rows.append(
            {
                "box": [float(v) for v in box],
                "class_id": int(class_id),
                "area_ratio": float(_clamp(area_ratio, 0.0, 1.0)),
                "importance_score": float(importance),
            }
        )

    fg_rows.sort(
        key=lambda item: (float(item.get("importance_score", 0.0)), float(item.get("area_ratio", 0.0))),
        reverse=True,
    )
    top_boxes = [row["box"] for row in fg_rows[:2]]
    union_box = _union_box(top_boxes)
    union_area_ratio = (
        _box_area_ratio_xyxy(union_box, width, height)
        if isinstance(union_box, list) and len(union_box) == 4
        else 0.0
    )
    dominant_area_ratio = max((float(row.get("area_ratio", 0.0)) for row in fg_rows), default=0.0)
    top2_gap = 0.0
    top2_iou = 0.0
    if len(fg_rows) >= 2:
        top2_gap = float(fg_rows[0]["importance_score"]) - float(fg_rows[1]["importance_score"])
        top2_iou = _iou_xyxy(fg_rows[0]["box"], fg_rows[1]["box"])
    object_signal = bool(
        fg_rows
        and (
            dominant_area_ratio >= 0.05
            or union_area_ratio >= 0.10
            or max((float(row.get("importance_score", 0.0)) for row in fg_rows), default=0.0) >= 0.45
        )
    )
    object_multi_signal = bool(
        len(fg_rows) >= 2
        and union_area_ratio >= 0.12
        and top2_iou < 0.65
        and (top2_gap < 0.18 or dominant_area_ratio < 0.22)
    )
    copyspace_layout_signal = bool(
        dominant_area_ratio > 0.0
        and dominant_area_ratio <= 0.22
        and union_area_ratio > 0.0
        and union_area_ratio <= 0.30
    )
    return {
        "count": int(len(fg_rows)),
        "dominant_area_ratio": round(float(dominant_area_ratio), 6),
        "union_area_ratio": round(float(union_area_ratio), 6),
        "object_signal": bool(object_signal),
        "object_multi_signal": bool(object_multi_signal),
        "copyspace_layout_signal": bool(copyspace_layout_signal),
        "top2_iou": round(float(top2_iou), 6),
        "top2_gap": round(float(top2_gap), 6),
    }


def _summarize_public_teacher_support(public_teacher_proposals: Any) -> Dict[str, Any]:
    if not isinstance(public_teacher_proposals, dict):
        return {
            "available": False,
            "teacher_count": 0,
            "consensus_iou": 0.0,
            "median_area_ratio": 0.0,
            "union_area_ratio": 0.0,
            "center_spread": 0.0,
            "best_margin_side": "unknown",
            "best_margin_ratio": 0.0,
            "object_focus_signal": False,
            "scene_fullframe_signal": False,
            "copyspace_signal": False,
            "hint_category": "none",
        }

    top_boxes: List[List[float]] = []
    for teacher_id, block in public_teacher_proposals.items():
        if not isinstance(block, dict):
            continue
        free_form = block.get("free_form", [])
        if not isinstance(free_form, list) or not free_form:
            continue
        box = _normalize_box_xyxy(free_form[0].get("bbox_norm_xyxy"), 1, 1)
        if box is None:
            continue
        top_boxes.append([float(v) for v in box])
    if not top_boxes:
        return {
            "available": False,
            "teacher_count": 0,
            "consensus_iou": 0.0,
            "median_area_ratio": 0.0,
            "union_area_ratio": 0.0,
            "center_spread": 0.0,
            "best_margin_side": "unknown",
            "best_margin_ratio": 0.0,
            "object_focus_signal": False,
            "scene_fullframe_signal": False,
            "copyspace_signal": False,
            "hint_category": "none",
        }

    pairwise_ious: List[float] = []
    center_pairs: List[float] = []
    areas: List[float] = []
    union_box = _union_box(top_boxes)
    union_area_ratio = float(_box_area(union_box)) if isinstance(union_box, list) and len(union_box) == 4 else 0.0
    margins_by_side: Dict[str, List[float]] = {"left": [], "right": [], "top": [], "bottom": []}
    for idx, box in enumerate(top_boxes):
        area = _box_area(box)
        areas.append(float(_clamp(area, 0.0, 1.0)))
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        margins_by_side["left"].append(max(0.0, box[0]))
        margins_by_side["right"].append(max(0.0, 1.0 - box[2]))
        margins_by_side["top"].append(max(0.0, box[1]))
        margins_by_side["bottom"].append(max(0.0, 1.0 - box[3]))
        for jdx in range(idx + 1, len(top_boxes)):
            other = top_boxes[jdx]
            pairwise_ious.append(_iou_xyxy(box, other))
            ocx = 0.5 * (other[0] + other[2])
            ocy = 0.5 * (other[1] + other[3])
            center_pairs.append(math.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2))

    best_margin_side = "unknown"
    best_margin_ratio = 0.0
    if margins_by_side:
        best_margin_side, best_margin_ratio = max(
            ((side, _median(vals)) for side, vals in margins_by_side.items()),
            key=lambda item: item[1],
        )

    teacher_count = len(top_boxes)
    consensus_iou = _mean(pairwise_ious) if pairwise_ious else 1.0
    median_area_ratio = _median(areas)
    center_spread = max(center_pairs) if center_pairs else 0.0
    object_focus_signal = bool(
        teacher_count >= 2
        and 0.12 <= median_area_ratio <= 0.65
        and consensus_iou >= 0.35
        and center_spread <= 0.30
    )
    scene_fullframe_signal = bool(
        teacher_count >= 2
        and median_area_ratio >= 0.72
        and consensus_iou >= 0.35
    )
    copyspace_signal = bool(
        teacher_count >= 2
        and best_margin_ratio >= 0.16
        and median_area_ratio <= 0.82
    )
    hint_category = "none"
    if copyspace_signal:
        hint_category = "copyspace"
    elif object_focus_signal:
        hint_category = "object_focus"
    elif scene_fullframe_signal:
        hint_category = "scene_fullframe"
    return {
        "available": True,
        "teacher_count": int(teacher_count),
        "consensus_iou": round(float(consensus_iou), 6),
        "median_area_ratio": round(float(median_area_ratio), 6),
        "union_area_ratio": round(float(_clamp(union_area_ratio, 0.0, 1.0)), 6),
        "center_spread": round(float(center_spread), 6),
        "best_margin_side": str(best_margin_side),
        "best_margin_ratio": round(float(best_margin_ratio), 6),
        "object_focus_signal": bool(object_focus_signal),
        "scene_fullframe_signal": bool(scene_fullframe_signal),
        "copyspace_signal": bool(copyspace_signal),
        "hint_category": str(hint_category),
    }


def _max_foreground_area_ratio(c2_instances: Sequence[Dict[str, Any]]) -> float:
    best = 0.0
    for inst in c2_instances:
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        best = max(best, _safe_float(inst.get("area_ratio", 0.0), 0.0))
    return float(_clamp(best, 0.0, 1.0))


def _foreground_mass_ratio(union_box: Optional[Sequence[float]], width: int, height: int) -> float:
    if not isinstance(union_box, (list, tuple)) or len(union_box) != 4:
        return 0.0
    return _box_area_ratio_xyxy([float(v) for v in union_box], width, height)


def _infer_copyspace_side(union_box: Optional[Sequence[float]], width: int, height: int) -> str:
    if not isinstance(union_box, (list, tuple)) or len(union_box) != 4:
        return "unknown"
    x1, y1, x2, y2 = [float(v) for v in union_box]
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    margins = {
        "left": max(0.0, x1) / w,
        "right": max(0.0, w - x2) / w,
        "top": max(0.0, y1) / h,
        "bottom": max(0.0, h - y2) / h,
    }
    side, value = max(margins.items(), key=lambda kv: kv[1])
    return side if value >= 0.10 else "unknown"


def _infer_copyspace_quality(
    *,
    blank_ratio: float,
    tag_signal: bool,
    hint_signal: bool,
    gate_passed: bool,
    caption_signal: bool = False,
    teacher_signal: bool = False,
    perception_signal: bool = False,
) -> str:
    semantic_signal = bool(tag_signal or caption_signal or teacher_signal)
    if gate_passed and semantic_signal and blank_ratio >= BLANK_RATIO_COPYSPACE_STRONG:
        return "strong"
    if gate_passed:
        return "weak"
    if semantic_signal or hint_signal or perception_signal or blank_ratio >= BLANK_RATIO_COPYSPACE_MIN:
        return "pseudo"
    return "none"


def _infer_copyspace_mode_fired_by(
    *,
    tag_signal: bool,
    hint_signal: bool,
    blank_signal: bool,
    caption_signal: bool = False,
    teacher_signal: bool = False,
    perception_signal: bool = False,
) -> str:
    if teacher_signal and blank_signal:
        return "teacher+blank"
    if caption_signal and blank_signal:
        return "caption+blank"
    if tag_signal and blank_signal:
        return "tag+blank"
    if perception_signal and blank_signal:
        return "perception_layout"
    if blank_signal and hint_signal:
        return "fallback_hint"
    if teacher_signal:
        return "teacher_only"
    if caption_signal:
        return "caption_only"
    if tag_signal:
        return "tag_only"
    if blank_signal:
        return "blank_only"
    if hint_signal:
        return "fallback_hint"
    return "none"


def _infer_scene_subtype(
    *,
    tags_norm: Sequence[str],
    super_cat: str,
    horizon_exists_prob: float,
    horizon_conf: float,
    symmetry_score: float,
    largest_obj_area_ratio: float,
    foreground_mass_ratio: float,
) -> Tuple[str, float, float]:
    text = " | ".join(str(x).strip().lower() for x in tags_norm if str(x).strip())
    sc = str(super_cat or "").strip().lower()
    is_nature = sc == "landscape_nature" or any(
        k in text for k in ("landscape", "nature", "mountain", "beach", "sea", "scenery", "panorama")
    )
    is_city = sc == "architecture_exterior" or any(
        k in text for k in ("cityscape", "skyline", "architecture", "building", "urban", "downtown")
    )
    is_interior = sc == "indoor_interior" or any(
        k in text for k in ("interior", "indoor", "room", "lobby", "office", "corridor")
    )
    is_reflection = any(k in text for k in ("reflection", "reflective", "mirror", "mirrored", "lake", "water"))
    is_pattern = any(k in text for k in ("pattern", "texture", "geometric", "geometry", "abstract"))

    dominant_vertical_strength = 0.0
    if is_city or is_interior:
        dominant_vertical_strength = 0.60 + 0.30 * _clamp(symmetry_score, 0.0, 1.0)
    elif is_pattern:
        dominant_vertical_strength = 0.35 + 0.25 * _clamp(symmetry_score, 0.0, 1.0)

    subtype = SCENE_SUBTYPE_UNKNOWN
    conf = 0.35
    if is_reflection and symmetry_score >= 0.45:
        subtype = "scene_reflection_symmetry"
        conf = 0.75 + 0.10 * _clamp(symmetry_score, 0.0, 1.0)
    elif is_nature and horizon_exists_prob >= 0.15:
        subtype = "scene_landscape_nature"
        conf = 0.68 + 0.20 * _clamp(horizon_conf, 0.0, 1.0)
    elif is_interior:
        subtype = "scene_interior_architecture"
        conf = 0.66 + 0.15 * _clamp(dominant_vertical_strength, 0.0, 1.0)
    elif is_city:
        subtype = "scene_city_architecture"
        conf = 0.64 + 0.12 * _clamp(dominant_vertical_strength, 0.0, 1.0)
    elif is_pattern and largest_obj_area_ratio < 0.18:
        subtype = "scene_structural_pattern"
        conf = 0.62 + 0.15 * _clamp(symmetry_score, 0.0, 1.0)
    elif largest_obj_area_ratio >= 0.16 or foreground_mass_ratio >= 0.20:
        subtype = "scene_contextual_object"
        conf = 0.58 + 0.12 * _clamp(max(largest_obj_area_ratio, foreground_mass_ratio), 0.0, 1.0)
    elif is_nature:
        subtype = "scene_landscape_nature"
        conf = 0.55 + 0.15 * _clamp(horizon_conf, 0.0, 1.0)
    return subtype, float(_clamp(conf, 0.0, 0.99)), float(_clamp(dominant_vertical_strength, 0.0, 1.0))


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


def _dedupe_boxes(boxes: Sequence[Sequence[float]], iou_thr: float = PERSON_BOX_DEDUP_IOU) -> List[List[float]]:
    kept: List[List[float]] = []
    for box in boxes:
        cand = [float(v) for v in box]
        if any(
            _iou_xyxy(cand, prev) >= float(iou_thr)
            or _intersection_ratio_to_smaller(cand, prev) >= float(PERSON_BOX_DEDUP_CONTAINMENT)
            for prev in kept
        ):
            continue
        kept.append(cand)
    return kept


def _pose_support_summary_for_box(box: Sequence[float], c3_pose: Sequence[Dict[str, Any]], width: int, height: int) -> Dict[str, float]:
    match_count = 0
    max_overlap = 0.0
    max_pose_score = 0.0
    max_face_score = 0.0
    for pose in c3_pose:
        if not isinstance(pose, dict):
            continue
        pose_box = _normalize_box_xyxy(pose.get("bbox"), width, height)
        if pose_box is None:
            continue
        overlap = _iou_xyxy(box, pose_box)
        if overlap < POSE_C2_SUPPORT_IOU_MIN:
            continue
        match_count += 1
        max_overlap = max(max_overlap, overlap)
        max_pose_score = max(max_pose_score, _safe_float(pose.get("score", 0.0), 0.0))
        face = pose.get("face", {}) if isinstance(pose.get("face"), dict) else {}
        max_face_score = max(max_face_score, _safe_float(face.get("score", 0.0), 0.0))
    return {
        "match_count": float(match_count),
        "max_overlap": float(max_overlap),
        "max_pose_score": float(max_pose_score),
        "max_face_score": float(max_face_score),
    }


def _is_confirmed_person_box(
    box: Sequence[float],
    *,
    det_score: float,
    pose_support: Dict[str, float],
) -> bool:
    aspect_wh = _box_width_to_height(box)
    max_face_score = _safe_float(pose_support.get("max_face_score", 0.0), 0.0)
    max_pose_score = _safe_float(pose_support.get("max_pose_score", 0.0), 0.0)
    plausible_shape = bool(aspect_wh <= PERSON_BOX_MAX_WIDTH_TO_HEIGHT or max_face_score >= POSE_FACE_SCORE_MIN)
    pose_backed = bool(max_face_score >= POSE_FACE_SCORE_MIN or max_pose_score >= POSE_SCORE_CONFIRM_MIN)
    det_backed = bool(det_score >= PERSON_DET_SCORE_CONFIRM_MIN)
    return bool(plausible_shape and (pose_backed or det_backed))


def _is_confirmed_pose_box(box: Sequence[float], *, pose_score: float, face_score: float) -> bool:
    aspect_wh = _box_width_to_height(box)
    if face_score >= POSE_FACE_SCORE_MIN:
        return True
    return bool(pose_score >= POSE_SCORE_CONFIRM_MIN and aspect_wh <= PERSON_BOX_MAX_WIDTH_TO_HEIGHT)


def _person_signal_stats(
    c2_instances: Sequence[Dict[str, Any]],
    c3_pose: Any,
    width: int,
    height: int,
) -> Dict[str, Any]:
    c3_list = c3_pose if isinstance(c3_pose, list) else []
    c2_person_boxes_raw: List[List[float]] = []
    c2_person_boxes: List[List[float]] = []
    for inst in c2_instances:
        if not isinstance(inst, dict):
            continue
        if bool(inst.get("bg_like", False)):
            continue
        class_id = int(_safe_float(inst.get("class_id", -1), -1))
        if class_id not in PERSON_CLASS_IDS:
            continue
        box = _normalize_box_xyxy(inst.get("box"), width, height)
        if box is None:
            continue
        c2_person_boxes_raw.append(box)
        pose_support = _pose_support_summary_for_box(box, c3_list, width, height)
        det_score = _safe_float(inst.get("score", 0.0), 0.0)
        if _is_confirmed_person_box(box, det_score=det_score, pose_support=pose_support):
            c2_person_boxes.append(box)
    c2_person_boxes = _dedupe_boxes(c2_person_boxes)

    c3_supported_boxes: List[List[float]] = []
    for pose in c3_list:
        if not isinstance(pose, dict):
            continue
        box = _normalize_box_xyxy(pose.get("bbox"), width, height)
        if box is None:
            continue
        pose_score = _safe_float(pose.get("score", 0.0), 0.0)
        face = pose.get("face", {}) if isinstance(pose.get("face"), dict) else {}
        face_score = _safe_float(face.get("score", 0.0), 0.0)
        overlap = max((_iou_xyxy(box, pb) for pb in c2_person_boxes), default=0.0)
        if c2_person_boxes:
            if overlap < POSE_C2_SUPPORT_IOU_MIN:
                continue
        elif not _is_confirmed_pose_box(box, pose_score=pose_score, face_score=face_score):
            continue
        c3_supported_boxes.append(box)
    c3_supported_boxes = _dedupe_boxes(c3_supported_boxes)

    effective_boxes = c2_person_boxes if c2_person_boxes else c3_supported_boxes
    num_person_source = "c2_person" if c2_person_boxes else ("c3_pose" if c3_supported_boxes else "none")
    return {
        "num_person": int(len(effective_boxes)),
        "num_person_c2": int(len(c2_person_boxes)),
        "num_person_c2_raw": int(len(_dedupe_boxes(c2_person_boxes_raw))),
        "num_person_c3_supported": int(len(c3_supported_boxes)),
        "num_person_source": num_person_source,
        "person_union_box_xyxy": _union_box(effective_boxes),
    }


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
    group_signal: bool,
    portrait_signal: bool,
    text_mode_signal: bool,
    copyspace_gate_passed: bool,
    object_signal: bool,
    object_multi_signal: bool,
    largest_obj_area_ratio: float,
    foreground_mass_ratio: float,
    scene_signal: bool,
    scene_score: float,
) -> Tuple[str, float, List[str], str]:
    reasons: List[str] = []
    if text_mode_signal:
        reasons.append("text_signal")
        return "text_document", 0.95, reasons, "rule_text"
    if group_signal:
        reasons.append("group_signal")
        return "portrait_group", 0.90, reasons, "rule_group"
    if portrait_signal:
        reasons.append("portrait_signal")
        return "portrait_single", 0.85, reasons, "rule_portrait"
    if object_multi_signal and (
        largest_obj_area_ratio >= (OBJECT_DOMINANCE_MIN * 0.75)
        or foreground_mass_ratio >= FOREGROUND_MASS_MIN
    ):
        reasons.append("object_multi_signal")
        return "object_multi", 0.72, reasons, "rule_object_multi"
    if copyspace_gate_passed:
        reasons.append("copyspace_signal")
        return "background_texture_copyspace", 0.80, reasons, "rule_copyspace"
    if object_signal and (
        largest_obj_area_ratio >= OBJECT_DOMINANCE_MIN or foreground_mass_ratio >= FOREGROUND_MASS_MIN
    ):
        reasons.append("object_signal")
        return "object_single", 0.60, reasons, "rule_object"
    if scene_signal or scene_score >= SCENE_SCORE_MIN:
        reasons.append("scene_signal")
        return "scene_general", 0.75, reasons, "rule_scene"
    if object_signal:
        reasons.append("object_signal")
        return "object_single", 0.55, reasons, "rule_object"
    reasons.append("fallback")
    return "other_ambiguous", 0.30, reasons, "rule_fallback"


def _fallback_mode_after_guard(
    *,
    text_mode_signal: bool,
    copyspace_gate_passed: bool,
    object_signal: bool,
    object_multi_signal: bool,
    scene_signal: bool,
    scene_score: float,
) -> Tuple[str, float, List[str], str]:
    reasons: List[str] = []
    if text_mode_signal:
        reasons.append("fallback_text_signal")
        return "text_document", 0.55, reasons, "fallback_text"
    if copyspace_gate_passed:
        reasons.append("fallback_copyspace_signal")
        return "background_texture_copyspace", 0.55, reasons, "fallback_copyspace"
    if object_multi_signal:
        reasons.append("fallback_object_multi_signal")
        return "object_multi", 0.50, reasons, "fallback_object_multi"
    if scene_signal or scene_score >= SCENE_SCORE_MIN:
        reasons.append("fallback_scene_signal")
        return "scene_general", 0.50, reasons, "fallback_scene"
    if object_signal:
        reasons.append("fallback_object_signal")
        return "object_single", 0.45, reasons, "fallback_object"
    reasons.append("fallback_ambiguous")
    return "other_ambiguous", 0.30, reasons, "fallback_ambiguous"


def _should_fallback_tiny_human_to_scene(
    *,
    mode: str,
    num_person: int,
    person_union_area_ratio: float,
    blank_ratio: float,
    scene_score: float,
    foreground_mass_ratio: float,
    has_explicit_people_hint: bool,
) -> bool:
    if not str(mode or "").startswith("portrait"):
        return False
    if int(num_person) <= 0:
        return False
    if has_explicit_people_hint:
        return False
    area_thr = (
        CONTEXTUAL_TINY_HUMAN_GROUP_AREA_MAX
        if int(num_person) >= 2
        else CONTEXTUAL_TINY_HUMAN_SINGLE_AREA_MAX
    )
    return bool(
        person_union_area_ratio > 0.0
        and person_union_area_ratio <= float(area_thr)
        and blank_ratio >= float(CONTEXTUAL_TINY_HUMAN_BLANK_RATIO_MIN)
        and scene_score >= float(CONTEXTUAL_TINY_HUMAN_SCENE_SCORE_MIN)
        and foreground_mass_ratio <= float(CONTEXTUAL_TINY_HUMAN_FOREGROUND_MASS_MAX)
    )


def _should_override_portrait_to_object(
    *,
    mode: str,
    num_person: int,
    person_union_area_ratio: float,
    has_explicit_people_hint: bool,
    object_signal: bool,
    largest_obj_area_ratio: float,
    foreground_mass_ratio: float,
    nonperson_union_area_ratio: float,
    caption_object_hint: bool,
    teacher_object_focus_signal: bool,
) -> bool:
    if not str(mode or "").startswith("portrait"):
        return False
    if int(num_person) <= 0:
        return False
    if has_explicit_people_hint:
        return False
    if person_union_area_ratio <= 0.0 or person_union_area_ratio > float(PORTRAIT_OBJECT_OVERRIDE_PERSON_AREA_MAX):
        return False
    if not object_signal:
        return False
    object_dom = max(float(largest_obj_area_ratio), float(nonperson_union_area_ratio))
    object_support = max(float(foreground_mass_ratio), float(nonperson_union_area_ratio))
    strong_object = bool(
        object_dom >= float(PORTRAIT_OBJECT_OVERRIDE_DOM_MIN)
        or object_support >= float(PORTRAIT_OBJECT_OVERRIDE_UNION_MIN)
        or caption_object_hint
        or teacher_object_focus_signal
    )
    if not strong_object:
        return False
    return bool(
        object_dom >= max(float(PORTRAIT_OBJECT_OVERRIDE_DOM_MIN), person_union_area_ratio * float(PORTRAIT_OBJECT_OVERRIDE_REL_GAIN))
        or object_support >= max(float(PORTRAIT_OBJECT_OVERRIDE_UNION_MIN), person_union_area_ratio * (float(PORTRAIT_OBJECT_OVERRIDE_REL_GAIN) + 0.4))
    )


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
    c7_saliency: Optional[Dict[str, Any]] = None,
    ocr_text_boxes_count: Optional[int] = None,
    text_overlay_likely: Optional[bool] = None,
    copy_space_flag: Optional[bool] = None,
    blank_ratio_est: Optional[float] = None,
    horizon_conf: Optional[float] = None,
    symmetry_score: Optional[float] = None,
    ocr_backend_method: Optional[str] = None,
    caption_text: Optional[str] = None,
    public_teacher_proposals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    c3_list = c3_pose if isinstance(c3_pose, list) else []
    person_stats = _person_signal_stats(c2_instances=c2_instances, c3_pose=c3_list, width=width, height=height)
    num_person = int(person_stats["num_person"])
    person_union_box = person_stats.get("person_union_box_xyxy")
    person_union_area_ratio = (
        _box_area_ratio_xyxy(person_union_box, width, height)
        if isinstance(person_union_box, list) and len(person_union_box) == 4
        else 0.0
    )

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

    sc = str(super_cat or "").strip().lower()
    caption_semantics = _infer_caption_semantics(caption_text)
    object_layout = _summarize_object_layout(c2_instances=c2_instances, width=width, height=height)
    teacher_support = _summarize_public_teacher_support(public_teacher_proposals)
    caption_excerpt = str(caption_semantics.get("text", ""))[:160] if caption_semantics.get("available", False) else ""
    caption_people_hint = bool(caption_semantics.get("people_hint", False))
    caption_group_hint = bool(caption_semantics.get("group_hint", False))
    caption_portrait_hint = bool(caption_semantics.get("portrait_hint", False))
    caption_scene_hint = bool(caption_semantics.get("scene_hint", False))
    caption_object_hint = bool(caption_semantics.get("object_hint", False))
    caption_copyspace_signal = bool(caption_semantics.get("copyspace_hint", False))
    caption_text_backed_evidence = bool(caption_semantics.get("text_backed_evidence", False))
    text_boxes = max(0, int(_safe_float(ocr_text_boxes_count, 0.0)))

    has_text_hint = bool(_has_any_hint(tags_norm, TEXT_HINTS) or caption_semantics.get("text_hint", False))
    has_text_strong_hint = bool(
        _has_any_hint(tags_norm, TEXT_STRONG_HINTS) or caption_text_backed_evidence
    )
    has_copyspace_tag = _has_any_hint(tags_norm, COPYSPACE_HINTS)
    has_copyspace_strong_tag = _has_any_hint(tags_norm, COPYSPACE_STRONG_HINTS)
    has_scene_hint = bool(_has_any_hint(tags_norm, SCENE_HINTS) or caption_scene_hint)
    has_object_hint = bool(_has_any_hint(tags_norm, OBJECT_HINTS) or caption_object_hint)
    text_overlay = bool(text_overlay_likely) if text_overlay_likely is not None else False
    ocr_method = str(ocr_backend_method or "").strip().lower()
    ocr_available = ocr_method not in {"", "unknown", "unavailable", "disabled"}
    strong_text_evidence = bool(text_boxes > 0 or text_overlay or caption_text_backed_evidence)
    text_signal = bool(has_text_hint or text_overlay or text_boxes > 0 or sc in TEXT_SUPER_CATS)
    copyspace_hint_signal = bool(copy_space_flag) if copy_space_flag is not None else False
    teacher_copyspace_signal = bool(teacher_support.get("copyspace_signal", False))

    if blank_ratio_est is None:
        if isinstance(union_box_seed, list) and len(union_box_seed) == 4:
            blank_ratio = 1.0 - _box_area_ratio_xyxy(union_box_seed, width, height)
        elif c2_primary_area_ratio_seed > 0.0:
            blank_ratio = 1.0 - float(c2_primary_area_ratio_seed)
        else:
            blank_ratio = 0.0
    else:
        blank_ratio = _clamp(_safe_float(blank_ratio_est, 0.0), 0.0, 1.0)
    largest_obj_area_ratio = _max_foreground_area_ratio(c2_instances)
    foreground_mass_ratio = _foreground_mass_ratio(union_box_seed, width, height)
    saliency_signal = summarize_saliency_signal(c7_saliency, width=width, height=height)
    saliency_semantic_summary = build_saliency_semantic_summary(
        width=width,
        height=height,
        c2_instances=c2_instances,
        c3_pose=c3_list,
        c7_saliency=c7_saliency,
        ocr_text_boxes_count=text_boxes,
        caption_text=caption_text,
    )
    semantic_primary_family = str(saliency_semantic_summary.get("primary_family", "unknown") or "unknown")
    semantic_primary_subtype = str(saliency_semantic_summary.get("primary_subtype", "") or "")
    semantic_layout_structure = str(saliency_semantic_summary.get("layout_structure", "single") or "single")
    semantic_support_trust_tier = str(saliency_semantic_summary.get("support_trust_tier", "low") or "low")
    semantic_human_cluster_count = int(_safe_float(saliency_semantic_summary.get("human_cluster_count", 0), 0.0))
    semantic_primary_support_mass = _safe_float(saliency_semantic_summary.get("primary_support_mass", 0.0), 0.0)
    semantic_attributed_support_mass = _safe_float(saliency_semantic_summary.get("attributed_support_mass", 0.0), 0.0)
    semantic_trust_rank = TRUST_TIER_RANK.get(semantic_support_trust_tier, 0)
    if semantic_human_cluster_count > 0 and semantic_human_cluster_count < num_person:
        num_person = semantic_human_cluster_count
        human_union = saliency_semantic_summary.get("human_union_bbox_norm_xyxy")
        if isinstance(human_union, list) and len(human_union) == 4:
            person_union_box = human_union
            person_union_area_ratio = _box_area_ratio_xyxy(person_union_box, 1, 1)
    saliency_fg_ratio = float(saliency_signal.get("foreground_area_ratio", 0.0))
    saliency_blank_ratio = float(saliency_signal.get("blank_ratio_est", 1.0))
    if saliency_fg_ratio > 0.0:
        foreground_mass_ratio = max(foreground_mass_ratio, saliency_fg_ratio)
        blank_ratio = min(blank_ratio, saliency_blank_ratio)
    group_signal = bool(num_person >= 2 or _has_any_hint(tags_norm, GROUP_HINTS) or caption_group_hint)
    portrait_signal = bool(
        num_person == 1
        or sc in PEOPLE_SUPER_CATS
        or _has_any_hint(tags_norm, PORTRAIT_HINTS)
        or caption_people_hint
        or caption_portrait_hint
        or (semantic_primary_family == "human" and semantic_trust_rank >= TRUST_TIER_RANK["medium"])
    )
    has_explicit_people_hint = bool(
        sc in PEOPLE_SUPER_CATS
        or _has_any_hint(tags_norm, PORTRAIT_HINTS)
        or _has_any_hint(tags_norm, GROUP_HINTS)
        or caption_people_hint
        or caption_group_hint
    )
    scene_signal = bool(
        sc in SCENE_SUPER_CATS
        or has_scene_hint
        or teacher_support.get("scene_fullframe_signal", False)
        or saliency_signal.get("scene_like_signal", False)
    )
    perception_object_signal = bool(object_layout.get("object_signal", False))
    object_signal_explicit = bool(
        sc in OBJECT_SUPER_CATS
        or has_object_hint
        or teacher_support.get("object_focus_signal", False)
    )
    object_signal = bool(object_signal_explicit or (perception_object_signal and not scene_signal))
    perception_object_multi_signal = bool(object_layout.get("object_multi_signal", False))
    perception_copyspace_signal = bool(
        object_layout.get("copyspace_layout_signal", False)
        and blank_ratio >= max(BLANK_RATIO_COPYSPACE_STRONG, 0.45)
        and foreground_mass_ratio <= 0.28
        and num_person <= 0
        and not strong_text_evidence
    )
    copyspace_blank_signal = bool(blank_ratio >= BLANK_RATIO_COPYSPACE_MIN)
    copyspace_semantic_signal = bool(
        has_copyspace_strong_tag or copyspace_hint_signal or caption_copyspace_signal
    )
    teacher_copyspace_conflict = bool(
        caption_people_hint
        or caption_group_hint
        or caption_portrait_hint
        or caption_scene_hint
        or caption_object_hint
        or scene_signal
        or object_signal_explicit
        or strong_text_evidence
        or teacher_support.get("scene_fullframe_signal", False)
    )
    teacher_copyspace_eligible = bool(
        teacher_copyspace_signal
        and blank_ratio >= BLANK_RATIO_COPYSPACE_STRONG
        and not teacher_copyspace_conflict
    )
    copyspace_explicit_signal = bool(copyspace_semantic_signal or teacher_copyspace_eligible)
    copyspace_signal = bool(
        has_copyspace_tag
        or copyspace_hint_signal
        or caption_copyspace_signal
        or teacher_copyspace_signal
        or perception_copyspace_signal
    )
    copyspace_gate_passed = bool(
        copyspace_blank_signal
        and (
            copyspace_explicit_signal
            or (
                perception_copyspace_signal
                and not scene_signal
                and not perception_object_multi_signal
            )
        )
    )
    copyspace_mode_fired_by = _infer_copyspace_mode_fired_by(
        tag_signal=bool(has_copyspace_tag),
        hint_signal=bool(copyspace_hint_signal),
        blank_signal=bool(copyspace_blank_signal),
        caption_signal=bool(caption_copyspace_signal),
        teacher_signal=bool(teacher_copyspace_signal),
        perception_signal=bool(perception_copyspace_signal),
    )
    horizon_exists_prob = _clamp(_safe_float(horizon_conf, 0.0), 0.0, 1.0)
    scene_score = float(
        _clamp(
            0.45 * (1.0 if scene_signal else 0.0)
            + 0.20 * horizon_exists_prob
            + 0.20 * _clamp(_safe_float(symmetry_score, 0.0), 0.0, 1.0)
            + 0.15 * (1.0 - _clamp(largest_obj_area_ratio, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    scene_subtype = None
    scene_conf = 0.0
    dominant_vertical_strength = 0.0
    text_mode_signal = bool(
        strong_text_evidence
        and (
            sc in TEXT_SUPER_CATS
            or has_text_strong_hint
            or (has_text_hint and max(largest_obj_area_ratio, foreground_mass_ratio) >= 0.18)
        )
    )
    mode, conf, reasons, base_rule_id = _infer_mode(
        group_signal=group_signal,
        portrait_signal=portrait_signal,
        text_mode_signal=text_mode_signal,
        copyspace_gate_passed=copyspace_gate_passed,
        object_signal=object_signal,
        object_multi_signal=perception_object_multi_signal,
        largest_obj_area_ratio=largest_obj_area_ratio,
        foreground_mass_ratio=foreground_mass_ratio,
        scene_signal=scene_signal,
        scene_score=scene_score,
    )
    if mode == "scene_general":
        scene_subtype, scene_conf, dominant_vertical_strength = _infer_scene_subtype(
            tags_norm=tags_norm,
            super_cat=super_cat,
            horizon_exists_prob=horizon_exists_prob,
            horizon_conf=_safe_float(horizon_conf, 0.0),
            symmetry_score=_safe_float(symmetry_score, 0.0),
            largest_obj_area_ratio=largest_obj_area_ratio,
            foreground_mass_ratio=foreground_mass_ratio,
        )
    router_rule_id = base_rule_id

    guard_reasons: List[str] = []

    # P0 guard #1: person_count=0 cannot route to portrait_*.
    if mode.startswith("portrait") and num_person <= 0:
        guard_reasons.append("guard_no_person_for_portrait")
        mode, conf, fb_reasons, fb_rule_id = _fallback_mode_after_guard(
            text_mode_signal=text_mode_signal,
            copyspace_gate_passed=copyspace_gate_passed,
            object_signal=object_signal,
            object_multi_signal=perception_object_multi_signal,
            scene_signal=scene_signal,
            scene_score=scene_score,
        )
        reasons.extend(guard_reasons + fb_reasons)
        router_rule_id = f"{base_rule_id}|{fb_rule_id}"

    # P0 guard #2: text_document requires OCR/caption-backed evidence.
    if mode == "text_document" and not text_mode_signal:
        guard_reasons.append("guard_text_requires_backed_evidence")
        mode, conf, fb_reasons, fb_rule_id = _fallback_mode_after_guard(
            text_mode_signal=False,
            copyspace_gate_passed=copyspace_gate_passed,
            object_signal=object_signal,
            object_multi_signal=perception_object_multi_signal,
            scene_signal=scene_signal,
            scene_score=scene_score,
        )
        reasons.extend(["guard_text_requires_backed_evidence"] + fb_reasons)
        router_rule_id = f"{base_rule_id}|{fb_rule_id}"
    elif text_signal and not text_mode_signal and "guard_text_requires_backed_evidence" not in reasons:
        reasons.append("guard_text_requires_backed_evidence")
        guard_reasons.append("guard_text_requires_backed_evidence")

    # P0 guard #3: copy-space tag only is insufficient when blank ratio is low.
    if mode == "background_texture_copyspace" and copyspace_signal and (blank_ratio < BLANK_RATIO_COPYSPACE_MIN):
        guard_reasons.append("guard_low_blank_ratio_for_copyspace")
        mode, conf, fb_reasons, fb_rule_id = _fallback_mode_after_guard(
            text_mode_signal=text_mode_signal,
            copyspace_gate_passed=False,
            object_signal=object_signal,
            object_multi_signal=perception_object_multi_signal,
            scene_signal=scene_signal,
            scene_score=scene_score,
        )
        reasons.extend(["guard_low_blank_ratio_for_copyspace"] + fb_reasons)
        router_rule_id = f"{base_rule_id}|{fb_rule_id}"
    elif copyspace_signal and (blank_ratio < BLANK_RATIO_COPYSPACE_MIN) and "guard_low_blank_ratio_for_copyspace" not in reasons:
        reasons.append("guard_low_blank_ratio_for_copyspace")
        guard_reasons.append("guard_low_blank_ratio_for_copyspace")

    object_focus_guard = bool(
        mode == "background_texture_copyspace"
        and object_signal
        and (
            max(
                float(object_layout.get("dominant_area_ratio", 0.0)),
                float(object_layout.get("union_area_ratio", 0.0)),
            ) >= 0.18
            or object_signal_explicit
        )
        and not copyspace_semantic_signal
    )
    if object_focus_guard:
        guard_reasons.append("guard_object_focus_over_copyspace")
        reasons.append("guard_object_focus_over_copyspace")
        mode = "object_multi" if perception_object_multi_signal else "object_single"
        conf = max(float(conf), 0.66 if mode == "object_single" else 0.72)
        router_rule_id = f"{router_rule_id}|guard_object_focus_over_copyspace"

    if _should_override_portrait_to_object(
        mode=mode,
        num_person=num_person,
        person_union_area_ratio=person_union_area_ratio,
        has_explicit_people_hint=has_explicit_people_hint,
        object_signal=object_signal,
        largest_obj_area_ratio=float(object_layout.get("dominant_area_ratio", largest_obj_area_ratio)),
        foreground_mass_ratio=foreground_mass_ratio,
        nonperson_union_area_ratio=float(object_layout.get("union_area_ratio", 0.0)),
        caption_object_hint=caption_object_hint,
        teacher_object_focus_signal=bool(teacher_support.get("object_focus_signal", False)),
    ):
        guard_reasons.append("guard_contextual_person_object_focus")
        reasons.append("guard_contextual_person_object_focus")
        mode = "object_multi" if perception_object_multi_signal else "object_single"
        conf = max(float(conf), 0.68 if mode == "object_single" else 0.74)
        router_rule_id = f"{router_rule_id}|guard_contextual_person_object_focus"

    semantic_object_override = bool(
        semantic_trust_rank >= TRUST_TIER_RANK["medium"]
        and semantic_primary_family in {"object", "animal"}
        and semantic_primary_support_mass >= max(0.10, person_union_area_ratio * 0.90)
        and (
            not mode.startswith("portrait")
            or not has_explicit_people_hint
            or semantic_primary_support_mass >= max(0.14, person_union_area_ratio * 1.20)
        )
    )
    if semantic_object_override and mode not in {"text_document", "background_texture_copyspace"}:
        if mode != "scene_general" or semantic_primary_support_mass >= 0.14:
            guard_reasons.append("guard_saliency_semantic_primary_object")
            reasons.append("guard_saliency_semantic_primary_object")
            mode = "object_multi" if semantic_layout_structure == "multi" else "object_single"
            conf = max(float(conf), 0.70 if mode == "object_single" else 0.76)
            router_rule_id = f"{router_rule_id}|guard_saliency_semantic_primary_object"

    contextual_tiny_human_scene = False
    if _should_fallback_tiny_human_to_scene(
        mode=mode,
        num_person=num_person,
        person_union_area_ratio=person_union_area_ratio,
        blank_ratio=blank_ratio,
        scene_score=scene_score,
        foreground_mass_ratio=foreground_mass_ratio,
        has_explicit_people_hint=has_explicit_people_hint,
    ):
        contextual_tiny_human_scene = True
        guard_reasons.append("guard_tiny_human_contextual_scene")
        reasons.append("guard_tiny_human_contextual_scene")
        mode = "scene_general"
        conf = max(0.55, min(0.85, 0.45 + 0.50 * float(scene_score)))
        router_rule_id = f"{router_rule_id}|guard_tiny_human_contextual_scene"

    if (
        mode == "other_ambiguous"
        and bool(saliency_signal.get("scene_like_signal", False))
        and c2_primary_area_ratio_seed < 0.04
        and largest_obj_area_ratio < 0.04
    ):
        mode = "scene_general"
        conf = max(float(conf), 0.52)
        reasons.append("saliency_distributed_scene")
        router_rule_id = f"{router_rule_id}|saliency_distributed_scene"

    if mode == "scene_general":
        scene_subtype, scene_conf, dominant_vertical_strength = _infer_scene_subtype(
            tags_norm=tags_norm,
            super_cat=super_cat,
            horizon_exists_prob=horizon_exists_prob,
            horizon_conf=_safe_float(horizon_conf, 0.0),
            symmetry_score=_safe_float(symmetry_score, 0.0),
            largest_obj_area_ratio=largest_obj_area_ratio,
            foreground_mass_ratio=foreground_mass_ratio,
        )
    else:
        scene_subtype = None
        scene_conf = 0.0

    # Finalize subject source/union after guard-corrected mode.
    primary_idx = int(primary_idx_seed)
    primary_source = "c2" if primary_idx >= 0 else "none"
    union_box = list(union_box_seed) if isinstance(union_box_seed, list) else None
    multi_subject = False
    if mode.startswith("portrait"):
        if num_person > 0:
            primary_source = str(person_stats.get("num_person_source", "c3_pose"))
        if mode == "portrait_group":
            multi_subject = True
            person_union = person_stats.get("person_union_box_xyxy")
            if isinstance(person_union, list) and len(person_union) == 4:
                union_box = [round(float(v), 3) for v in person_union]
    elif mode.startswith("object"):
        nonperson_rows = [
            inst
            for inst in c2_instances
            if int(_safe_float(inst.get("class_id", -1), -1)) not in PERSON_CLASS_IDS
        ]
        if len(nonperson_rows) >= 2:
            s1 = _safe_float(nonperson_rows[0].get("importance_score", 0.0), 0.0)
            s2 = _safe_float(nonperson_rows[1].get("importance_score", 0.0), 0.0)
            iou12 = _iou_xyxy(nonperson_rows[0].get("box", [0, 0, 0, 0]), nonperson_rows[1].get("box", [0, 0, 0, 0]))
            if (s1 - s2) < 0.15 and iou12 < 0.75:
                mode = "object_multi"
                conf = max(conf, 0.70)
                reasons.append("top2_close_multi_override")
                multi_subject = True
                union_box = _union_box(
                    [
                        [float(v) for v in nonperson_rows[0].get("box", [0, 0, 0, 0])],
                        [float(v) for v in nonperson_rows[1].get("box", [0, 0, 0, 0])],
                    ]
                )
                if union_box is not None:
                    union_box = [round(v, 3) for v in union_box]
        if mode == "object_multi":
            multi_subject = True
    elif mode in {"scene_general", "background_texture_copyspace", "text_document"}:
        primary_idx = -1
        primary_source = "none"
        if contextual_tiny_human_scene:
            union_box = None

    c2_primary_area_ratio = 0.0
    c2_primary_bg_like = False
    if 0 <= primary_idx < len(c2_instances):
        c2_primary_area_ratio = _safe_float(c2_instances[primary_idx].get("area_ratio", 0.0), 0.0)
        c2_primary_bg_like = bool(c2_instances[primary_idx].get("bg_like", False))

    copyspace_side = _infer_copyspace_side(union_box_seed, width, height)
    if copyspace_side == "unknown" and teacher_support.get("best_margin_side", "unknown") != "unknown":
        copyspace_side = str(teacher_support.get("best_margin_side", "unknown"))

    flags = {
        "has_person": bool(num_person > 0),
        "has_text_heavy": bool(mode == "text_document"),
        "has_copyspace_tag": bool(has_copyspace_tag),
        "has_copyspace_hint": bool(copyspace_signal),
        "is_background_like": bool(0 <= primary_idx < len(c2_instances) and c2_primary_bg_like),
        "contextual_tiny_human": bool(contextual_tiny_human_scene),
        "caption_available": bool(caption_semantics.get("available", False)),
        "public_teacher_support_available": bool(teacher_support.get("available", False)),
    }

    shot_type: Optional[str] = None
    if mode == "portrait_group":
        shot_type = "group"
    elif mode == "portrait_single":
        if num_person > 0:
            portrait_union = person_union_box if isinstance(person_union_box, list) and len(person_union_box) == 4 else _person_union_box(c3_list, width, height)
            if portrait_union is not None:
                ar = _box_area(portrait_union) / float(max(1, int(width) * int(height)))
                if ar < PORTRAIT_TINY_SUBJECT_UNKNOWN_SHOT_MAX:
                    shot_type = "unknown"
                    if "guard_tiny_human_unknown_shot" not in reasons:
                        reasons.append("guard_tiny_human_unknown_shot")
                elif ar < 0.18:
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
    if semantic_primary_family == "human" or mode.startswith("portrait"):
        primary_subject_type = "person"
    elif semantic_primary_family in {"object", "animal"} or mode.startswith("object"):
        primary_subject_type = "object"
    elif semantic_primary_family == "text" or mode == "text_document":
        primary_subject_type = "text"
    elif semantic_primary_family == "scene_region" or mode in {"scene_general", "background_texture_copyspace"}:
        primary_subject_type = "scene"

    mode_conflict = False
    if mode.startswith("portrait") and not flags["has_person"] and sc not in PEOPLE_SUPER_CATS:
        mode_conflict = True
    if mode == "text_document" and not text_mode_signal:
        mode_conflict = True
    if guard_reasons:
        mode_conflict = True

    if not router_rule_id:
        router_rule_id = "rule_unknown"

    router_signals = {
        "super_cat": sc,
        "num_person": int(num_person),
        "num_person_c2": int(person_stats.get("num_person_c2", 0)),
        "num_person_c2_raw": int(person_stats.get("num_person_c2_raw", 0)),
        "num_person_c3_supported": int(person_stats.get("num_person_c3_supported", 0)),
        "num_person_source": str(person_stats.get("num_person_source", "none")),
        "person_union_area_ratio": round(float(person_union_area_ratio), 6),
        "c2_num_instances": int(len(c2_instances)),
        "c2_primary_bg_like": bool(c2_primary_bg_like),
        "c2_primary_area_ratio": round(float(c2_primary_area_ratio), 6),
        "caption_available": bool(caption_semantics.get("available", False)),
        "caption_excerpt": caption_excerpt,
        "caption_people_hint": bool(caption_semantics.get("people_hint", False)),
        "caption_group_hint": bool(caption_semantics.get("group_hint", False)),
        "caption_portrait_hint": bool(caption_semantics.get("portrait_hint", False)),
        "caption_scene_hint": bool(caption_semantics.get("scene_hint", False)),
        "caption_object_hint": bool(caption_semantics.get("object_hint", False)),
        "caption_copyspace_hint": bool(caption_semantics.get("copyspace_hint", False)),
        "caption_text_backed_evidence": bool(caption_semantics.get("text_backed_evidence", False)),
        "has_text_hint": bool(has_text_hint),
        "ocr_text_boxes": int(text_boxes),
        "ocr_backend_method": ocr_method or "unknown",
        "ocr_available": bool(ocr_available),
        "text_overlay_likely": bool(text_overlay),
        "text_evidence_strong": bool(strong_text_evidence),
        "text_mode_signal": bool(text_mode_signal),
        "text_signal": bool(text_signal),
        "has_copyspace_tag": bool(has_copyspace_tag),
        "copyspace_tag_signal": bool(has_copyspace_tag),
        "copyspace_hint_signal": bool(copyspace_hint_signal),
        "copyspace_caption_signal": bool(caption_copyspace_signal),
        "copyspace_teacher_signal": bool(teacher_copyspace_signal),
        "copyspace_perception_signal": bool(perception_copyspace_signal),
        "copy_space_flag": bool(copyspace_signal),
        "blank_ratio_est": round(float(blank_ratio), 6),
        "blank_ratio_thr": float(BLANK_RATIO_COPYSPACE_MIN),
        "copyspace_blank_signal": bool(copyspace_blank_signal),
        "copyspace_gate_passed": bool(copyspace_gate_passed),
        "copyspace_mode_fired_by": str(copyspace_mode_fired_by),
        "copyspace_allowed": bool(copyspace_gate_passed),
        "horizon_exists_prob": round(float(horizon_exists_prob), 6),
        "horizon_conf": round(float(_safe_float(horizon_conf, 0.0)), 6),
        "symmetry_score": round(float(_safe_float(symmetry_score, 0.0)), 6),
        "scene_tag_signal": bool(has_scene_hint),
        "scene_signal": bool(scene_signal),
        "scene_score": round(float(scene_score), 6),
        "largest_obj_area_ratio": round(float(largest_obj_area_ratio), 6),
        "foreground_mass_ratio": round(float(foreground_mass_ratio), 6),
        "nonperson_object_count": int(object_layout.get("count", 0)),
        "nonperson_object_union_area_ratio": round(float(object_layout.get("union_area_ratio", 0.0)), 6),
        "nonperson_object_dominant_area_ratio": round(float(object_layout.get("dominant_area_ratio", 0.0)), 6),
        "object_multi_candidate": bool(object_layout.get("object_multi_signal", False)),
        "dominant_vertical_strength": round(float(dominant_vertical_strength), 6),
        "public_teacher_available": bool(teacher_support.get("available", False)),
        "public_teacher_count": int(teacher_support.get("teacher_count", 0)),
        "public_teacher_hint_category": str(teacher_support.get("hint_category", "none")),
        "public_teacher_consensus_iou": round(float(teacher_support.get("consensus_iou", 0.0)), 6),
        "public_teacher_median_area_ratio": round(float(teacher_support.get("median_area_ratio", 0.0)), 6),
        "public_teacher_best_margin_side": str(teacher_support.get("best_margin_side", "unknown")),
        "public_teacher_best_margin_ratio": round(float(teacher_support.get("best_margin_ratio", 0.0)), 6),
        "public_teacher_copyspace_signal": bool(teacher_support.get("copyspace_signal", False)),
        "public_teacher_copyspace_eligible": bool(teacher_copyspace_eligible),
        "public_teacher_object_focus_signal": bool(teacher_support.get("object_focus_signal", False)),
        "public_teacher_scene_fullframe_signal": bool(teacher_support.get("scene_fullframe_signal", False)),
        "saliency_available": bool(saliency_signal.get("available", False)),
        "saliency_foreground_area_ratio": round(float(saliency_signal.get("foreground_area_ratio", 0.0)), 6),
        "saliency_blank_ratio_est": round(float(saliency_signal.get("blank_ratio_est", 1.0)), 6),
        "saliency_dominance_score": round(float(saliency_signal.get("dominance_score", 0.0)), 6),
        "saliency_top2_mass_ratio": round(float(saliency_signal.get("top2_mass_ratio", 0.0)), 6),
        "saliency_component_count": int(saliency_signal.get("component_count", 0)),
        "saliency_dispersion_score": round(float(saliency_signal.get("dispersion_score", 1.0)), 6),
        "saliency_entropy_norm": round(float(saliency_signal.get("entropy_norm", 1.0)), 6),
        "semantic_primary_family": semantic_primary_family,
        "semantic_primary_subtype": semantic_primary_subtype,
        "semantic_layout_structure": semantic_layout_structure,
        "semantic_support_trust_tier": semantic_support_trust_tier,
        "semantic_primary_support_mass": round(float(semantic_primary_support_mass), 6),
        "semantic_attributed_support_mass": round(float(semantic_attributed_support_mass), 6),
        "semantic_human_cluster_count": int(semantic_human_cluster_count),
    }
    copyspace = {
        "tag_signal": bool(has_copyspace_tag),
        "hint_signal": bool(copyspace_hint_signal),
        "caption_signal": bool(caption_copyspace_signal),
        "teacher_signal": bool(teacher_copyspace_signal),
        "teacher_eligible": bool(teacher_copyspace_eligible),
        "perception_signal": bool(perception_copyspace_signal),
        "blank_signal": bool(copyspace_blank_signal),
        "gate_passed": bool(copyspace_gate_passed),
        "mode_fired_by": str(copyspace_mode_fired_by),
        "side": str(copyspace_side),
        "blank_ratio_est": round(float(blank_ratio), 6),
        "blank_ratio_thr": float(BLANK_RATIO_COPYSPACE_MIN),
        "quality": _infer_copyspace_quality(
            blank_ratio=float(blank_ratio),
            tag_signal=bool(has_copyspace_tag),
            hint_signal=bool(copyspace_hint_signal),
            gate_passed=bool(copyspace_gate_passed),
            caption_signal=bool(caption_copyspace_signal),
            teacher_signal=bool(teacher_copyspace_signal),
            perception_signal=bool(perception_copyspace_signal),
        ),
    }

    primary_subject_exists = bool(
        num_person > 0
        or primary_idx >= 0
        or (
            union_box is not None
            and mode not in {"scene_general", "background_texture_copyspace", "text_document"}
        )
    )
    num_effective_subjects = int(
        num_person if num_person > 0 else (2 if multi_subject else (1 if primary_subject_exists else 0))
    )
    semantic_primary_box = saliency_semantic_summary.get("primary_cluster_bbox_norm_xyxy")
    semantic_union_box = saliency_semantic_summary.get("union_bbox_norm_xyxy")
    semantic_anchor_box: Optional[Sequence[float]] = None
    if semantic_trust_rank >= TRUST_TIER_RANK["medium"]:
        if mode.startswith("portrait") and semantic_primary_family == "human":
            semantic_anchor_box = semantic_union_box or semantic_primary_box
        elif mode.startswith("object") and semantic_primary_family in {"object", "animal"}:
            semantic_anchor_box = semantic_union_box if semantic_layout_structure == "multi" else semantic_primary_box
        elif mode in {"scene_general", "other_ambiguous"} and semantic_primary_family in {"human", "object", "animal"}:
            semantic_anchor_box = semantic_union_box or semantic_primary_box
    raw_anchor_box = semantic_anchor_box or union_box
    if raw_anchor_box is None and 0 <= primary_idx < len(c2_instances):
        raw_anchor_box = c2_instances[primary_idx].get("box")
    if raw_anchor_box is None:
        raw_anchor_box = person_union_box
    effective_subject_region = resolve_effective_subject_region(
        width=width,
        height=height,
        subject_mode=mode,
        subject_set={
            "primary_idx": int(primary_idx),
            "union_box_xyxy": union_box,
            "num_person": int(num_person),
        },
        raw_anchor_box=raw_anchor_box,
        c7_saliency=c7_saliency,
        saliency_semantic_summary=saliency_semantic_summary,
    )

    return {
        "subject_mode": mode,
        "subject_mode_conf": round(float(conf), 6),
        "scene_subtype": scene_subtype,
        "scene_conf": round(float(scene_conf), 6),
        "subject_family": semantic_primary_family,
        "subject_subtype": semantic_primary_subtype,
        "layout_structure": semantic_layout_structure,
        "support_trust_tier": semantic_support_trust_tier,
        "subject_mode_reasons": reasons,
        "subject_mode_flags": flags,
        "subject_mode_conflict": bool(mode_conflict),
        "shot_type": shot_type,
        "primary_subject_type": primary_subject_type,
        "primary_subject_source": primary_source,
        "subject_set": {
            "num_person": int(num_person),
            "num_c2_instances": int(len(c2_instances)),
            "num_subject_inst": int(len(c2_instances)),
            "num_effective_subjects": int(num_effective_subjects),
            "primary_subject_exists": bool(primary_subject_exists),
            "union_box_xyxy": union_box,
            "primary_idx": int(primary_idx),
            "c2_primary_area_ratio": round(float(c2_primary_area_ratio), 6),
            "c2_primary_bg_like": bool(c2_primary_bg_like),
            "person_union_area_ratio": round(float(person_union_area_ratio), 6),
            "multi_subject": bool(multi_subject or num_person >= 2 or semantic_layout_structure == "multi"),
        },
        "policy_id": SUBJECT_MODE_TO_POLICY.get(mode, "generic_v1"),
        "router_rule_id": str(router_rule_id),
        "router_signals": router_signals,
        "copyspace": copyspace,
        "saliency_semantic_summary": saliency_semantic_summary,
        "effective_subject_region": effective_subject_region,
    }
