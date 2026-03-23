from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def relativize_path(path_str: Optional[str], cwd: Path) -> Optional[str]:
    if not path_str:
        return None
    path = Path(path_str)
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_image_size(path_str: Optional[str], cache: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    path = str(path_str or "")
    if path in cache:
        return cache[path]
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    with Image.open(image_path) as image:
        size = (int(image.width), int(image.height))
    cache[path] = size
    return size


def norm_xyxy_to_coco_bbox(bbox_norm_xyxy: Sequence[Any], width: int, height: int) -> List[float]:
    x1 = safe_float(bbox_norm_xyxy[0]) * width
    y1 = safe_float(bbox_norm_xyxy[1]) * height
    x2 = safe_float(bbox_norm_xyxy[2]) * width
    y2 = safe_float(bbox_norm_xyxy[3]) * height
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)]


def coco_area(bbox_xywh: Sequence[Any]) -> float:
    return round(max(0.0, safe_float(bbox_xywh[2])) * max(0.0, safe_float(bbox_xywh[3])), 3)


def build_common_image_entry(
    image_entry_id: int,
    record: Dict[str, Any],
    width: int,
    height: int,
    cwd: Path,
) -> Dict[str, Any]:
    return {
        "id": image_entry_id,
        "file_name": relativize_path(record.get("image_path"), cwd),
        "width": width,
        "height": height,
        "orig_image_id": record.get("image_id"),
        "target_ar": record.get("target_ar"),
        "label_generation": safe_dict(record.get("label_generation")),
        "routing": safe_dict(record.get("routing")),
        "baseline": safe_dict(record.get("baseline")),
        "teacher_meta": safe_dict(record.get("teacher_meta")),
    }


def build_batch_coco(records: Sequence[Dict[str, Any]], cwd: Path) -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    size_cache: Dict[str, Tuple[int, int]] = {}
    ann_id = 1
    for image_entry_id, record in enumerate(records, start=1):
        width, height = load_image_size(record.get("image_path"), size_cache)
        image_row = build_common_image_entry(image_entry_id, record, width, height, cwd)
        image_row["decision_target"] = safe_dict(record.get("decision_target"))
        image_row["matching_target_count"] = len(safe_list(record.get("matching_targets")))
        image_row["candidate_pool_count"] = len(safe_list(record.get("candidate_pool")))
        image_row["ignored_candidate_count"] = len(safe_list(record.get("ignored_candidates")))
        images.append(image_row)
        for target in safe_list(record.get("matching_targets")):
            bbox = norm_xyxy_to_coco_bbox(target.get("bbox_norm_xyxy", [0, 0, 1, 1]), width, height)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_entry_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "area": coco_area(bbox),
                    "iscrowd": 0,
                    "score": safe_float(safe_dict(target.get("score_targets")).get("score_prob", 0.0)),
                    "gt_flag": 1,
                    "candidate_id": target.get("candidate_id"),
                    "target_id": target.get("target_id"),
                    "score_targets": safe_dict(target.get("score_targets")),
                    "macro_targets": safe_dict(target.get("macro_targets")),
                    "checklist_labels": safe_dict(target.get("checklist_labels")),
                    "checklist_scores": safe_dict(target.get("checklist_scores")),
                    "applicable_mask": safe_dict(target.get("applicable_mask")),
                    "observed_mask": safe_dict(target.get("observed_mask")),
                    "derived_checklist_targets": safe_dict(target.get("derived_checklist_targets")),
                    "why_tags": safe_list(target.get("why_tags")),
                    "is_safe_high_score_leftover": bool(target.get("is_safe_high_score_leftover", False)),
                    "safe_leftover_policy_state": target.get("safe_leftover_policy_state"),
                }
            )
            ann_id += 1
    return {
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": [{"supercategory": "none", "id": 0, "name": "crop"}],
        "sstk_meta": {
            "source_schema": "sstk_conditional_detr_batch_v2",
            "image_unit": "(image_id, target_ar)",
            "note": "Each COCO image row corresponds to one conditioned crop sample, not one raw image across all aspect ratios.",
            "safe_leftover_policies": sorted(
                {
                    str(safe_dict(record.get("label_generation")).get("safe_leftover_policy", "keep_negative"))
                    for record in records
                }
            ),
        },
    }


def build_canonical_coco(records: Sequence[Dict[str, Any]], cwd: Path) -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    size_cache: Dict[str, Tuple[int, int]] = {}
    ann_id = 1
    for image_entry_id, record in enumerate(records, start=1):
        width, height = load_image_size(record.get("image_path"), size_cache)
        image_row = build_common_image_entry(image_entry_id, record, width, height, cwd)
        image_row["decision"] = safe_dict(record.get("decision"))
        image_row["candidate_count"] = len(safe_list(record.get("candidates")))
        images.append(image_row)
        for candidate in safe_list(record.get("candidates")):
            bbox = norm_xyxy_to_coco_bbox(candidate.get("bbox_norm_xyxy", [0, 0, 1, 1]), width, height)
            is_positive = bool(candidate.get("is_positive_candidate", False))
            is_unsafe = bool(candidate.get("is_unsafe_negative", False))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_entry_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "area": coco_area(bbox),
                    "iscrowd": 0,
                    "score": safe_float(safe_dict(candidate.get("score_targets")).get("score_raw_rank", 0.0)),
                    "gt_flag": 1 if is_positive and not is_unsafe else 0,
                    "candidate_id": candidate.get("candidate_id"),
                    "label_type": candidate.get("label_type"),
                    "is_positive_candidate": is_positive,
                    "is_soft_positive": bool(candidate.get("is_soft_positive", False)),
                    "is_hard_negative": bool(candidate.get("is_hard_negative", False)),
                    "is_unsafe_negative": is_unsafe,
                    "is_ignore_candidate": bool(candidate.get("is_ignore_candidate", False)),
                    "is_safe_high_score_leftover": bool(candidate.get("is_safe_high_score_leftover", False)),
                    "safe_leftover_policy_state": candidate.get("safe_leftover_policy_state"),
                    "score_targets": safe_dict(candidate.get("score_targets")),
                    "macro_targets": safe_dict(candidate.get("macro_targets")),
                    "checklist_labels": safe_dict(candidate.get("checklist_labels")),
                    "checklist_scores": safe_dict(candidate.get("checklist_scores")),
                    "applicable_mask": safe_dict(candidate.get("applicable_mask")),
                    "observed_mask": safe_dict(candidate.get("observed_mask")),
                    "why_tags": safe_list(candidate.get("why_tags")),
                    "reject_tags": safe_list(candidate.get("reject_tags")),
                }
            )
            ann_id += 1
    return {
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": [{"supercategory": "none", "id": 0, "name": "crop"}],
        "sstk_meta": {
            "source_schema": "sstk_detr_train_v2",
            "image_unit": "(image_id, target_ar)",
            "note": "Canonical COCO keeps all candidates as annotations and uses gt_flag to mark safe positive candidates.",
            "safe_leftover_policies": sorted(
                {
                    str(safe_dict(record.get("label_generation")).get("safe_leftover_policy", "keep_negative"))
                    for record in records
                }
            ),
        },
    }


def validate_coco(coco: Dict[str, Any], *, require_positive_gt: bool) -> Dict[str, Any]:
    images = safe_list(coco.get("images"))
    annotations = safe_list(coco.get("annotations"))
    image_ids = {safe_dict(image).get("id") for image in images}
    image_size_map = {safe_dict(image).get("id"): (safe_dict(image).get("width"), safe_dict(image).get("height")) for image in images}
    errors: List[str] = []
    ann_ids: set[int] = set()
    gt_counter = 0
    label_type_counter: Counter[str] = Counter()
    for annotation in annotations:
        ann = safe_dict(annotation)
        ann_id = ann.get("id")
        image_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if ann_id in ann_ids:
            errors.append(f"duplicate annotation id: {ann_id}")
        ann_ids.add(ann_id)
        if image_id not in image_ids:
            errors.append(f"annotation {ann_id} references missing image_id={image_id}")
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"annotation {ann_id} has invalid bbox")
            continue
        _, _, width, height = bbox
        if safe_float(width) <= 0.0 or safe_float(height) <= 0.0:
            errors.append(f"annotation {ann_id} has non-positive bbox size")
        img_w, img_h = image_size_map[image_id]
        x, y, w, h = [safe_float(value) for value in bbox]
        if x < 0 or y < 0 or x + w > safe_float(img_w) + 1e-3 or y + h > safe_float(img_h) + 1e-3:
            errors.append(f"annotation {ann_id} bbox exceeds image bounds")
        if safe_float(ann.get("gt_flag", 0.0)) > 0.0:
            gt_counter += 1
        label_type_counter[str(ann.get("label_type", ""))] += 1
    if require_positive_gt and gt_counter <= 0:
        errors.append("no positive gt_flag annotations found")
    return {
        "status": "ok" if not errors else "failed",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "gt_annotation_count": gt_counter,
        "label_type_counts": dict(label_type_counter),
        "error_count": len(errors),
        "errors": errors[:100],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SSTK Conditional-DETR labels to COCO instances JSON")
    parser.add_argument("--canonical_jsonl", required=True)
    parser.add_argument("--batch_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_records = load_jsonl(Path(args.canonical_jsonl))
    batch_records = load_jsonl(Path(args.batch_jsonl))

    canonical_coco = build_canonical_coco(canonical_records, cwd)
    batch_coco = build_batch_coco(batch_records, cwd)

    canonical_path = out_dir / "instances_conditional_detr_canonical.json"
    batch_path = out_dir / "instances_conditional_detr_batch.json"
    write_json(canonical_path, canonical_coco)
    write_json(batch_path, batch_coco)

    canonical_validation = validate_coco(canonical_coco, require_positive_gt=True)
    batch_validation = validate_coco(batch_coco, require_positive_gt=True)
    summary = {
        "canonical_json": str(canonical_path),
        "batch_json": str(batch_path),
        "canonical_validation": canonical_validation,
        "batch_validation": batch_validation,
    }
    write_json(out_dir / "coco_conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if canonical_validation["status"] != "ok" or batch_validation["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
