from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2


CATEGORY_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 255),
    7: (255, 128, 0),
    8: (0, 128, 255),
    9: (128, 255, 0),
}

PERSON_CLASS_IDS = {0}
SUBJECT_OVERLAY_COLOR = (255, 255, 0)
SUPPORT_OVERLAY_COLOR = (0, 165, 255)
CONTROL_OVERLAY_COLOR = (255, 255, 255)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_box(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in vals):
        return None
    vals = [max(0.0, min(1.0, value)) for value in vals]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def _box_px(box_norm_xyxy: Sequence[float], width: int, height: int, padding_height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box_norm_xyxy]
    return (
        int(round(x1 * width)),
        int(round(y1 * height + padding_height)),
        int(round(x2 * width)),
        int(round(y2 * height + padding_height)),
    )


def _point_px(point_norm_xy: Sequence[float], width: int, height: int, padding_height: int) -> Tuple[int, int]:
    return (
        int(round(float(point_norm_xy[0]) * width)),
        int(round(float(point_norm_xy[1]) * height + padding_height)),
    )


def _feature_box_to_norm(box: Any, width: int, height: int) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in vals):
        return None
    if max(vals) > 1.5:
        vals = [
            vals[0] / max(1.0, float(width)),
            vals[1] / max(1.0, float(height)),
            vals[2] / max(1.0, float(width)),
            vals[3] / max(1.0, float(height)),
        ]
    return _norm_box(vals)


def _union_norm_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    normalized = [_norm_box(box) for box in boxes]
    valid = [box for box in normalized if box is not None]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _entity_index(entity_id: str, prefix: str) -> Optional[int]:
    if not entity_id.startswith(prefix):
        return None
    try:
        return int(entity_id[len(prefix) :])
    except ValueError:
        return None


def _load_feature_index(feature_jsonl: str, source_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    if not feature_jsonl:
        return {}
    path = Path(feature_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"feature_jsonl not found: {path}")
    features: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("image_id") or row.get("source_image_id") or "")
            if source_id not in source_ids:
                continue
            features[source_id] = row
            if len(features) >= len(source_ids):
                break
    return features


def _person_box_from_feature(
    feature_row: Optional[Dict[str, Any]],
    entity_id: str,
    width: int,
    height: int,
    *,
    prefer_face: bool,
) -> Optional[List[float]]:
    if not isinstance(feature_row, dict):
        return None
    idx = _entity_index(entity_id, "person_")
    c3_pose = feature_row.get("c3_pose") if isinstance(feature_row.get("c3_pose"), list) else []
    if idx is not None and 0 <= idx < len(c3_pose) and isinstance(c3_pose[idx], dict):
        pose = c3_pose[idx]
        if prefer_face:
            face = pose.get("face") if isinstance(pose.get("face"), dict) else {}
            face_box = _feature_box_to_norm(face.get("bbox_norm") or face.get("bbox"), width, height)
            if face_box is not None:
                return face_box
        pose_box = _feature_box_to_norm(pose.get("bbox"), width, height)
        if pose_box is not None:
            return pose_box
    person_boxes = []
    for inst in feature_row.get("c2_det") or []:
        if isinstance(inst, dict) and int(_safe_float(inst.get("class_id"), -1)) in PERSON_CLASS_IDS:
            box = _feature_box_to_norm(inst.get("box") or inst.get("bbox"), width, height)
            if box is not None:
                person_boxes.append(box)
    if idx is not None and 0 <= idx < len(person_boxes):
        return person_boxes[idx]
    return person_boxes[0] if person_boxes else None


def _group_box_from_feature(feature_row: Optional[Dict[str, Any]], width: int, height: int) -> Optional[List[float]]:
    if not isinstance(feature_row, dict):
        return None
    boxes: List[List[float]] = []
    for pose in feature_row.get("c3_pose") or []:
        if not isinstance(pose, dict):
            continue
        box = _feature_box_to_norm(pose.get("bbox"), width, height)
        if box is not None:
            boxes.append(box)
    if len(boxes) < 2:
        for inst in feature_row.get("c2_det") or []:
            if isinstance(inst, dict) and int(_safe_float(inst.get("class_id"), -1)) in PERSON_CLASS_IDS:
                box = _feature_box_to_norm(inst.get("box") or inst.get("bbox"), width, height)
                if box is not None:
                    boxes.append(box)
    subject_set = (feature_row.get("routing") or {}).get("subject_set") if isinstance(feature_row.get("routing"), dict) else {}
    if isinstance(subject_set, dict):
        route_box = _feature_box_to_norm(subject_set.get("union_box_xyxy"), width, height)
        if route_box is not None:
            boxes.append(route_box)
    return _union_norm_boxes(boxes)


def _object_boxes_from_feature(feature_row: Optional[Dict[str, Any]], width: int, height: int) -> List[List[float]]:
    if not isinstance(feature_row, dict):
        return []
    rows = []
    for source_key in ("c2_seg", "c2_det"):
        for inst in feature_row.get(source_key) or []:
            if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
                continue
            class_id = int(_safe_float(inst.get("class_id"), -1))
            if class_id < 0 or class_id in PERSON_CLASS_IDS:
                continue
            box = _feature_box_to_norm(inst.get("box") or inst.get("bbox"), width, height)
            if box is None:
                continue
            score = _safe_float(inst.get("importance_score"), _safe_float(inst.get("score"), 0.0))
            area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
            rows.append((score, area, box))
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: List[List[float]] = []
    for _, _, box in rows:
        if any(_iou_norm(box, existing) >= 0.88 for existing in out):
            continue
        out.append(box)
    return out


def _iou_norm(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _object_box_from_feature(feature_row: Optional[Dict[str, Any]], entity_id: str, width: int, height: int) -> Optional[List[float]]:
    boxes = _object_boxes_from_feature(feature_row, width, height)
    if entity_id == "object_multi_top2":
        return _union_norm_boxes(boxes[:2])
    idx = _entity_index(entity_id, "object_")
    if idx is not None and 0 <= idx < len(boxes):
        return boxes[idx]
    return boxes[0] if boxes else None


def _scene_box_from_feature(feature_row: Optional[Dict[str, Any]], width: int, height: int) -> Optional[List[float]]:
    if not isinstance(feature_row, dict):
        return None
    routing = feature_row.get("routing") if isinstance(feature_row.get("routing"), dict) else {}
    effective = routing.get("effective_subject_region") if isinstance(routing.get("effective_subject_region"), dict) else {}
    for key in ("guidance_envelope_bbox_norm_xyxy", "effective_bbox_norm_xyxy"):
        box = _feature_box_to_norm(effective.get(key), width, height)
        if box is not None:
            return box
    c7 = feature_row.get("c7_saliency") if isinstance(feature_row.get("c7_saliency"), dict) else {}
    for key in ("foreground_bbox_norm_xyxy", "top_component_bbox_norm_xyxy", "top2_union_bbox_norm_xyxy"):
        box = _feature_box_to_norm(c7.get(key), width, height)
        if box is not None:
            return box
    return None


def _subject_box_for_annotation(
    ann: Dict[str, Any],
    attrs: Dict[str, Any],
    feature_row: Optional[Dict[str, Any]],
    width: int,
    height: int,
) -> Optional[List[float]]:
    mode_name = str(ann.get("mode_name") or attrs.get("mode_name") or "")
    entity_id = str(ann.get("entity_id") or attrs.get("entity_id") or "")
    entity_type = str(attrs.get("entity_type") or ann.get("entity_type") or "")
    debug = attrs.get("subject_debug") if isinstance(attrs.get("subject_debug"), dict) else {}
    comps = attrs.get("score_components") if isinstance(attrs.get("score_components"), dict) else {}
    if mode_name == "landscape" or entity_type == "scene":
        safe_enabled = attrs.get("landscape_subject_safe_enabled", comps.get("landscape_subject_safe_enabled", 1))
        safe_active = attrs.get("landscape_subject_safe_active", comps.get("landscape_subject_safe_active", 0))
        suppressed = str(
            attrs.get(
                "landscape_subject_safe_suppressed_reason",
                comps.get("landscape_subject_safe_suppressed_reason", ""),
            )
            or ""
        )
        if str(safe_enabled) in {"0", "False", "false"} or str(safe_active) in {"0", "False", "false"} or suppressed:
            return None
    debug_keys: List[str]
    if mode_name == "face":
        debug_keys = ["head_bbox_norm_xyxy", "face_bbox_norm_xyxy", "core_bbox_norm_xyxy", "anchor_bbox_norm_xyxy"]
    elif mode_name == "landscape" or entity_type == "scene":
        debug_keys = ["core_bbox_norm_xyxy", "support_bbox_norm_xyxy", "anchor_bbox_norm_xyxy"]
    else:
        debug_keys = ["anchor_bbox_norm_xyxy", "core_bbox_norm_xyxy", "support_bbox_norm_xyxy"]
    for key in debug_keys:
        box = _norm_box(debug.get(key))
        if box is not None:
            return box
    if mode_name == "face" or entity_id.startswith("person_") or entity_type == "person":
        return _person_box_from_feature(feature_row, entity_id, width, height, prefer_face=(mode_name == "face"))
    if entity_type == "group" or entity_id == "group_all" or mode_name.startswith("group"):
        return _group_box_from_feature(feature_row, width, height)
    if entity_type in {"object", "object_multi"} or entity_id.startswith("object_"):
        return _object_box_from_feature(feature_row, entity_id, width, height)
    if mode_name == "landscape" or entity_type == "scene":
        return _scene_box_from_feature(feature_row, width, height)
    return None


def _draw_text_box(canvas: Any, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    if not text:
        return
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    y = max(th + baseline + 2, y)
    cv2.rectangle(canvas, (x, y - th - baseline - 3), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def _draw_subject_support_overlays(
    *,
    canvas: Any,
    anns: Sequence[Dict[str, Any]],
    feature_row: Optional[Dict[str, Any]],
    width: int,
    height: int,
    padding_height: int,
    alpha: float,
) -> Tuple[int, int]:
    overlay = canvas.copy()
    subject_rows: List[Tuple[List[float], str, Tuple[int, int, int]]] = []
    support_points: List[Tuple[List[float], str]] = []
    seen_subjects: Set[Tuple[str, Tuple[int, int, int, int]]] = set()
    seen_points: Set[Tuple[str, int, int]] = set()
    for ann in anns:
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        category_id = int(ann.get("category_id", -1))
        color = CATEGORY_COLORS.get(category_id, SUBJECT_OVERLAY_COLOR)
        mode_name = str(ann.get("mode_name") or attrs.get("mode_name") or "")
        entity_id = str(ann.get("entity_id") or attrs.get("entity_id") or "")
        subject_box = _subject_box_for_annotation(ann, attrs, feature_row, width, height)
        if subject_box is not None:
            rounded = tuple(int(round(value * 1000)) for value in subject_box)
            key = (f"{mode_name}:{entity_id}", rounded)
            if key not in seen_subjects:
                seen_subjects.add(key)
                subject_rows.append((subject_box, f"SUBJ {mode_name}:{entity_id}".rstrip(":"), color))
        if mode_name.startswith("single_person") or mode_name.startswith("group"):
            portrait_comp = attrs.get("portrait_comp") if isinstance(attrs.get("portrait_comp"), dict) else {}
            support_point = portrait_comp.get("support_point_norm_xy")
            if isinstance(support_point, (list, tuple)) and len(support_point) == 2:
                sx = int(round(_safe_float(support_point[0]) * 1000))
                sy = int(round(_safe_float(support_point[1]) * 1000))
                key = (entity_id, sx, sy)
                if key not in seen_points:
                    seen_points.add(key)
                    support_points.append(([float(support_point[0]), float(support_point[1])], f"SUPPORT {entity_id}".rstrip()))
    for box, _, color in subject_rows:
        x1, y1, x2, y2 = _box_px(box, width, height, padding_height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    for point, _ in support_points:
        cx, cy = _point_px(point, width, height, padding_height)
        radius = max(5, int(round(min(width, height) * 0.012)))
        cv2.circle(overlay, (cx, cy), radius, SUPPORT_OVERLAY_COLOR, -1)
    cv2.addWeighted(overlay, max(0.0, min(1.0, alpha)), canvas, 1.0 - max(0.0, min(1.0, alpha)), 0, canvas)
    for box, label, color in subject_rows:
        x1, y1, x2, y2 = _box_px(box, width, height, padding_height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        _draw_text_box(canvas, label, (x1, max(padding_height + 12, y1 - 4)), color)
    for point, label in support_points:
        cx, cy = _point_px(point, width, height, padding_height)
        radius = max(5, int(round(min(width, height) * 0.012)))
        cv2.circle(canvas, (cx, cy), radius + 2, (0, 0, 0), 2)
        cv2.circle(canvas, (cx, cy), radius, SUPPORT_OVERLAY_COLOR, 2)
        cv2.line(canvas, (cx - radius - 4, cy), (cx + radius + 4, cy), SUPPORT_OVERLAY_COLOR, 1)
        cv2.line(canvas, (cx, cy - radius - 4), (cx, cy + radius + 4), SUPPORT_OVERLAY_COLOR, 1)
        _draw_text_box(canvas, label, (cx + radius + 4, cy - radius - 2), SUPPORT_OVERLAY_COLOR)
    return len(subject_rows), len(support_points)


def _category_names(payload: Dict[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for row in payload.get("categories", []):
        if not isinstance(row, dict):
            continue
        result[int(row.get("id", -1))] = str(row.get("name") or row.get("mode_name") or row.get("id"))
    return result


def _source_id(row: Dict[str, Any]) -> str:
    return str(row.get("source_image_id") or Path(str(row.get("file_name") or "")).stem)


def _image_path(image_root: Path, image_row: Dict[str, Any]) -> Path:
    file_name = str(image_row.get("file_name") or "")
    if file_name:
        direct = image_root / file_name
        if direct.is_file():
            return direct
    source_id = _source_id(image_row)
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = image_root / f"{source_id}{suffix}"
        if candidate.is_file():
            return candidate
    return image_root / file_name


def _selected_image_ids(payload: Dict[str, Any], image_ids_path: str, samples: int, seed: int) -> List[int]:
    source_filter: Optional[set[str]] = None
    if image_ids_path:
        source_filter = {
            line.strip()
            for line in Path(image_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    selected: List[int] = []
    for row in payload.get("images", []):
        if not isinstance(row, dict):
            continue
        if source_filter is not None and _source_id(row) not in source_filter:
            continue
        selected.append(int(row["id"]))
    if source_filter is None and samples > 0 and len(selected) > samples:
        rng = random.Random(seed)
        selected = sorted(rng.sample(selected, samples))
    return selected


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize positive multimode label boxes.")
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_ids", default="", help="Optional text file of source image ids.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--category", default="all", help="'all' or a numeric category id.")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--feature_jsonl", default="", help="Optional precompute feature JSONL used to recover subject/support overlays.")
    parser.add_argument("--draw_subject_support", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_support_alpha", type=float, default=0.22)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    label_json = Path(args.label_json)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(label_json)
    image_by_id = {int(row["id"]): row for row in payload.get("images", []) if isinstance(row, dict)}
    category_names = _category_names(payload)
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if isinstance(ann, dict):
            annotations_by_image[int(ann.get("image_id", -1))].append(ann)

    target_category: Optional[int]
    if args.category == "all":
        target_category = None
    else:
        target_category = int(args.category)

    selected = _selected_image_ids(payload, args.image_ids, args.samples, args.seed)
    selected_source_ids = {_source_id(image_by_id[image_id]) for image_id in selected if image_id in image_by_id}
    feature_index = _load_feature_index(args.feature_jsonl, selected_source_ids) if args.feature_jsonl else {}
    category_count: Counter[int] = Counter()
    missing_images: List[str] = []
    written: List[str] = []
    subject_overlay_count = 0
    support_overlay_count = 0
    padding_height = 34

    for image_id in selected:
        image_row = image_by_id.get(image_id)
        if image_row is None:
            continue
        source_id = _source_id(image_row)
        path = _image_path(image_root, image_row)
        if not path.is_file():
            missing_images.append(source_id)
            continue
        image = cv2.imread(str(path))
        if image is None:
            missing_images.append(source_id)
            continue
        canvas = cv2.copyMakeBorder(image, padding_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in annotations_by_image.get(image_id, []):
            if int(ann.get("gt_flag", 0)) != 1:
                continue
            category_id = int(ann.get("category_id", -1))
            if target_category is not None and category_id != target_category:
                continue
            grouped[category_id].append(ann)
        selected_annotations: List[Dict[str, Any]] = []
        for category_id, rows in grouped.items():
            rows.sort(key=lambda row: _safe_float(row.get("score_mode", row.get("score", 1.0))), reverse=True)
            selected_annotations.extend(rows[: args.top_k])
        if args.draw_subject_support:
            subj_count, supp_count = _draw_subject_support_overlays(
                canvas=canvas,
                anns=selected_annotations,
                feature_row=feature_index.get(source_id),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                padding_height=padding_height,
                alpha=float(args.subject_support_alpha),
            )
            subject_overlay_count += subj_count
            support_overlay_count += supp_count
        for category_id, rows in grouped.items():
            rows.sort(key=lambda row: _safe_float(row.get("score_mode", row.get("score", 1.0))), reverse=True)
            color = CATEGORY_COLORS.get(category_id, (180, 180, 180))
            name = category_names.get(category_id, f"cat_{category_id}")
            for ann in rows[: args.top_k]:
                bbox = ann.get("bbox") or [0, 0, 0, 0]
                x, y, w, h = [_safe_float(v) for v in bbox[:4]]
                y += padding_height
                pt1 = (int(round(x)), int(round(y)))
                pt2 = (int(round(x + w)), int(round(y + h)))
                cv2.rectangle(canvas, pt1, pt2, color, 2)
                attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
                target_ar = str(attrs.get("target_ar") or ann.get("target_ar") or "")
                score = _safe_float(ann.get("score_mode", ann.get("score", 1.0)))
                label = f"{name} {target_ar} {score:.2f}".strip()
                text_org = (pt1[0], max(12, pt1[1] - 6))
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                bg_tl = (text_org[0], text_org[1] - th - baseline - 4)
                bg_br = (text_org[0] + tw + 4, text_org[1] + baseline + 2)
                cv2.rectangle(canvas, bg_tl, bg_br, (0, 0, 0), -1)
                cv2.putText(canvas, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                category_count[category_id] += 1
        out_path = out_dir / f"vis_{source_id}.jpg"
        cv2.imwrite(str(out_path), canvas)
        written.append(str(out_path))
        if len(written) % 50 == 0:
            print(f"processed {len(written)}/{len(selected)}")

    summary = {
        "label_json": str(label_json),
        "image_root": str(image_root),
        "out_dir": str(out_dir),
        "selected_image_count": len(selected),
        "written_image_count": len(written),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "top_k": int(args.top_k),
        "category_box_counts": {str(key): value for key, value in sorted(category_count.items())},
        "draw_subject_support": bool(args.draw_subject_support),
        "feature_jsonl": str(args.feature_jsonl),
        "feature_rows_loaded": len(feature_index),
        "subject_overlay_count": int(subject_overlay_count),
        "support_overlay_count": int(support_overlay_count),
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_dir.parent / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
