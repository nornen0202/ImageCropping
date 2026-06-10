from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .mode_catalog import BASE_CATEGORIES


def _bbox_to_xywh_pixels(bbox_norm_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1 = float(bbox_norm_xyxy[0]) * float(width)
    y1 = float(bbox_norm_xyxy[1]) * float(height)
    x2 = float(bbox_norm_xyxy[2]) * float(width)
    y2 = float(bbox_norm_xyxy[3]) * float(height)
    return [round(x1, 3), round(y1, 3), round(max(0.0, x2 - x1), 3), round(max(0.0, y2 - y1), 3)]


def build_coco_dataset(
    *,
    image_rows: Sequence[Dict[str, Any]],
    annotation_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return normalize_coco_image_rows({
        "images": list(image_rows),
        "categories": list(BASE_CATEGORIES),
        "annotations": list(annotation_rows),
    })


def build_image_entry(
    *,
    image_entry_id: int,
    source_image_id: str,
    file_name: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    return {
        "id": int(image_entry_id),
        "source_image_id": str(source_image_id),
        "file_name": str(file_name),
        "width": int(width),
        "height": int(height),
    }


def build_annotation_entry(
    *,
    annotation_id: int,
    image_entry_id: int,
    width: int,
    height: int,
    query_id: str,
    entity_id: str,
    mode_name: str,
    category_id: int,
    source_route_mode: str,
    bbox_norm_xyxy: Sequence[float],
    score_mode: float,
    gt_flag: int,
    is_best: int,
    attributes: Dict[str, Any],
) -> Dict[str, Any]:
    bbox = _bbox_to_xywh_pixels(bbox_norm_xyxy, width=width, height=height)
    return {
        "id": int(annotation_id),
        "image_id": int(image_entry_id),
        "category_id": int(category_id),
        "bbox": bbox,
        "area": round(float(bbox[2]) * float(bbox[3]), 3),
        "iscrowd": 0,
        "gt_flag": int(gt_flag),
        "query_id": str(query_id),
        "entity_id": str(entity_id),
        "mode_name": str(mode_name),
        "is_best": int(is_best),
        "score_mode": round(float(score_mode), 6),
        "source_route_mode": str(source_route_mode),
        "attributes": dict(attributes),
    }


def normalize_coco_image_rows(coco_dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse legacy per-target-AR image rows into one row per source image."""
    categories = [dict(row) for row in coco_dataset.get("categories", [])]
    raw_images = [dict(row) for row in coco_dataset.get("images", [])]
    raw_annotations = [dict(row) for row in coco_dataset.get("annotations", [])]
    old_id_to_new_id: Dict[int, int] = {}
    old_id_to_target_ar: Dict[int, str] = {}
    source_to_new_id: Dict[str, int] = {}
    images: List[Dict[str, Any]] = []

    for raw_image in raw_images:
        old_id = int(raw_image.get("id", len(old_id_to_new_id) + 1))
        source_image_id = str(raw_image.get("source_image_id", raw_image.get("id", old_id)))
        old_id_to_target_ar[old_id] = str(raw_image.get("target_ar", ""))
        if source_image_id not in source_to_new_id:
            new_id = len(images) + 1
            source_to_new_id[source_image_id] = new_id
            image_row = dict(raw_image)
            image_row["id"] = new_id
            image_row["source_image_id"] = source_image_id
            image_row.pop("target_ar", None)
            images.append(image_row)
        old_id_to_new_id[old_id] = source_to_new_id[source_image_id]

    annotations: List[Dict[str, Any]] = []
    for raw_annotation in raw_annotations:
        row = dict(raw_annotation)
        old_image_id = int(row.get("image_id", 0))
        if old_image_id in old_id_to_new_id:
            row["image_id"] = old_id_to_new_id[old_image_id]
        attributes = row.get("attributes")
        if isinstance(attributes, dict):
            attributes = dict(attributes)
            if "target_ar" not in attributes and old_image_id in old_id_to_target_ar:
                attributes["target_ar"] = old_id_to_target_ar[old_image_id]
            row["attributes"] = attributes
        elif old_image_id in old_id_to_target_ar:
            row["attributes"] = {"target_ar": old_id_to_target_ar[old_image_id]}
        annotations.append(row)

    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def build_target_ar_only_coco_dataset(coco_dataset: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_coco_image_rows(coco_dataset)
    images = [dict(row) for row in normalized.get("images", [])]
    categories = [dict(row) for row in coco_dataset.get("categories", [])]
    annotations: List[Dict[str, Any]] = []
    for annotation in normalized.get("annotations", []):
        row = dict(annotation)
        attributes = row.pop("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        target_ar = row.pop("target_ar", attributes.get("target_ar", ""))
        row["attributes"] = {"target_ar": str(target_ar)}
        annotations.append(row)
    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
