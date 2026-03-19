from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_xyxy_to_coco_bbox(bbox_norm_xyxy: Sequence[Any], width: int, height: int) -> List[float]:
    x1 = safe_float(bbox_norm_xyxy[0]) * width
    y1 = safe_float(bbox_norm_xyxy[1]) * height
    x2 = safe_float(bbox_norm_xyxy[2]) * width
    y2 = safe_float(bbox_norm_xyxy[3]) * height
    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    return [round(x1, 3), round(y1, 3), round(bbox_w, 3), round(bbox_h, 3)]


def coco_area(bbox_xywh: Sequence[Any]) -> float:
    return round(max(0.0, safe_float(bbox_xywh[2])) * max(0.0, safe_float(bbox_xywh[3])), 3)


def basename_only(path_str: Optional[str]) -> str:
    return Path(path_str or "").name


def load_image_size(image_path: str, size_cache: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    if image_path in size_cache:
        return size_cache[image_path]
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        size = (int(image.width), int(image.height))
    size_cache[image_path] = size
    return size


def build_gaic_like_instances(batch_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    size_cache: Dict[str, Tuple[int, int]] = {}
    ann_id = 1
    for image_entry_id, record in enumerate(batch_records, start=1):
        image_path = str(record.get("image_path") or "")
        width, height = load_image_size(image_path, size_cache)
        routing = safe_dict(record.get("routing"))
        decision_target = safe_dict(record.get("decision_target"))
        images.append(
            {
                "file_name": basename_only(image_path),
                "height": height,
                "width": width,
                "id": image_entry_id,
                "sstk_image_id": record.get("image_id"),
                "target_ar": record.get("target_ar"),
            }
        )
        for target in safe_list(record.get("matching_targets")):
            bbox = norm_xyxy_to_coco_bbox(target.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), width, height)
            score_targets = safe_dict(target.get("score_targets"))
            annotations.append(
                {
                    "area": coco_area(bbox),
                    "image_id": image_entry_id,
                    "bbox": bbox,
                    "category_id": 0,
                    "id": ann_id,
                    "score": safe_float(score_targets.get("score_prob", 0.0)),
                    "gt_flag": 1,
                    "iscrowd": 0,
                    "subject_mode_id": safe_int(routing.get("subject_mode_id"), -1),
                    "decision_id": safe_int(decision_target.get("decision_id"), -1),
                    "candidate_id": target.get("candidate_id"),
                    "target_ar": record.get("target_ar"),
                    "macro_targets": safe_dict(target.get("macro_targets")),
                    "label_type": "matching_target",
                    "is_hard_negative": False,
                    "is_unsafe_negative": False,
                }
            )
            ann_id += 1
        for candidate in safe_list(record.get("candidate_pool")):
            bbox = norm_xyxy_to_coco_bbox(candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), width, height)
            annotations.append(
                {
                    "area": coco_area(bbox),
                    "image_id": image_entry_id,
                    "bbox": bbox,
                    "category_id": 0,
                    "id": ann_id,
                    "score": safe_float(candidate.get("score_prob", candidate.get("score_rank_pct", 0.0))),
                    "gt_flag": 0,
                    "iscrowd": 0,
                    "subject_mode_id": safe_int(routing.get("subject_mode_id"), -1),
                    "decision_id": safe_int(decision_target.get("decision_id"), -1),
                    "candidate_id": candidate.get("candidate_id"),
                    "target_ar": record.get("target_ar"),
                    "label_type": candidate.get("label_type"),
                    "is_hard_negative": bool(candidate.get("is_hard_negative", False)),
                    "is_unsafe_negative": bool(candidate.get("is_unsafe_negative", False)),
                }
            )
            ann_id += 1
    return {
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": [{"supercategory": "none", "id": 0, "name": "crop"}],
    }


def validate_gaic_like_instances(
    payload: Dict[str, Any],
    *,
    expected_image_count: int,
    expected_annotation_count: int,
) -> Dict[str, Any]:
    images = safe_list(payload.get("images"))
    annotations = safe_list(payload.get("annotations"))
    categories = safe_list(payload.get("categories"))
    errors: List[str] = []
    warnings: List[str] = []
    image_ids = set()
    image_size_map: Dict[int, Tuple[int, int]] = {}
    file_name_with_dir_count = 0
    for image in images:
        row = safe_dict(image)
        image_id = safe_int(row.get("id"), -1)
        image_ids.add(image_id)
        image_size_map[image_id] = (safe_int(row.get("width")), safe_int(row.get("height")))
        if "/" in str(row.get("file_name", "")) or "\\" in str(row.get("file_name", "")):
            file_name_with_dir_count += 1
    if len(images) != expected_image_count:
        errors.append(f"image count mismatch: expected={expected_image_count}, got={len(images)}")
    if len(annotations) != expected_annotation_count:
        errors.append(f"annotation count mismatch: expected={expected_annotation_count}, got={len(annotations)}")
    if file_name_with_dir_count:
        errors.append(f"found {file_name_with_dir_count} images with non-basename file_name")
    if len(categories) != 1 or safe_dict(categories[0]).get("name") != "crop":
        errors.append("categories must contain exactly one 'crop' entry")

    ann_ids: set[int] = set()
    subject_mode_counter: Counter[int] = Counter()
    decision_counter: Counter[int] = Counter()
    gt_count = 0
    negative_count = 0
    score_values: List[float] = []
    label_type_counter: Counter[str] = Counter()
    for annotation in annotations:
        row = safe_dict(annotation)
        ann_id = safe_int(row.get("id"), -1)
        if ann_id in ann_ids:
            errors.append(f"duplicate annotation id: {ann_id}")
        ann_ids.add(ann_id)
        image_id = safe_int(row.get("image_id"), -1)
        if image_id not in image_ids:
            errors.append(f"annotation {ann_id} references missing image_id={image_id}")
            continue
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"annotation {ann_id} has invalid bbox")
            continue
        x, y, bbox_w, bbox_h = [safe_float(value) for value in bbox]
        width, height = image_size_map[image_id]
        if bbox_w <= 0.0 or bbox_h <= 0.0:
            errors.append(f"annotation {ann_id} has non-positive bbox size")
        if x < 0.0 or y < 0.0 or x + bbox_w > width + 1e-3 or y + bbox_h > height + 1e-3:
            errors.append(f"annotation {ann_id} bbox exceeds image bounds")
        gt_flag = safe_int(row.get("gt_flag"), 0)
        if gt_flag not in (0, 1):
            errors.append(f"annotation {ann_id} gt_flag must be 0 or 1")
        if gt_flag == 1:
            gt_count += 1
        else:
            negative_count += 1
        if "subject_mode_id" not in row or "decision_id" not in row or "candidate_id" not in row:
            errors.append(f"annotation {ann_id} missing required SSTK fields")
        subject_mode_counter[safe_int(row.get("subject_mode_id"), -1)] += 1
        decision_counter[safe_int(row.get("decision_id"), -1)] += 1
        label_type_counter[str(row.get("label_type", ""))] += 1
        score = safe_float(row.get("score"), -1.0)
        score_values.append(score)
        if not 0.0 <= score <= 1.0:
            warnings.append(f"annotation {ann_id} score outside [0,1]: {score}")
        macro_targets = safe_dict(row.get("macro_targets"))
        if gt_flag == 1:
            for macro_key in ("A_macro", "S_macro", "C_macro", "T_macro"):
                if macro_key not in macro_targets:
                    warnings.append(f"annotation {ann_id} missing macro target {macro_key}")
    return {
        "status": "ok" if not errors else "failed",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "gt_annotation_count": gt_count,
        "negative_annotation_count": negative_count,
        "subject_mode_id_counts": {str(key): value for key, value in sorted(subject_mode_counter.items())},
        "decision_id_counts": {str(key): value for key, value in sorted(decision_counter.items())},
        "label_type_counts": dict(label_type_counter),
        "score_range": {
            "min": min(score_values) if score_values else None,
            "max": max(score_values) if score_values else None,
        },
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
    }


def build_format_guide(
    gaic_reference: Dict[str, Any],
    converted_payload: Dict[str, Any],
    validation_summary: Dict[str, Any],
    out_json_path: Path,
) -> str:
    gaic_images = safe_list(gaic_reference.get("images"))
    gaic_annotations = safe_list(gaic_reference.get("annotations"))
    gaic_categories = safe_list(gaic_reference.get("categories"))
    sample_image = gaic_images[0] if gaic_images else {}
    sample_annotation = gaic_annotations[0] if gaic_annotations else {}
    sample_category = gaic_categories[0] if gaic_categories else {}
    converted_images = safe_list(converted_payload.get("images"))
    converted_annotations = safe_list(converted_payload.get("annotations"))
    converted_sample_image = converted_images[0] if converted_images else {}
    converted_sample_annotation = converted_annotations[0] if converted_annotations else {}
    lines = [
        "# GAIC instances_train Format Guide",
        "",
        "## 1. 원본 GAIC `instances_train.json` 포맷",
        "",
        "- top-level keys: `images`, `type`, `annotations`, `categories`",
        "- `type`는 보통 `instances`입니다.",
        "- `images[]`는 이미지 메타데이터 행이고, `annotations[]`는 해당 이미지에 속한 crop bbox 후보 행입니다.",
        "- `categories[]`는 단일 category `crop` 하나만 둡니다.",
        "",
        "### 1.1 GAIC top-level statistics",
        "",
        f"- images: `{len(gaic_images)}`",
        f"- annotations: `{len(gaic_annotations)}`",
        f"- categories: `{len(gaic_categories)}`",
        "",
        "### 1.2 GAIC sample rows",
        "",
        "sample `images[0]`:",
        "",
        "```json",
        json.dumps(sample_image, ensure_ascii=False, indent=2),
        "```",
        "",
        "sample `annotations[0]`:",
        "",
        "```json",
        json.dumps(sample_annotation, ensure_ascii=False, indent=2),
        "```",
        "",
        "sample `categories[0]`:",
        "",
        "```json",
        json.dumps(sample_category, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 2. 필드 해석",
        "",
        "| location | field | meaning |",
        "| --- | --- | --- |",
        "| images[] | `file_name` | 원본 이미지 파일명 |",
        "| images[] | `height`, `width` | 픽셀 단위 원본 크기 |",
        "| images[] | `id` | annotation에서 참조하는 image primary key |",
        "| annotations[] | `image_id` | 어느 image row에 속하는지 가리키는 foreign key |",
        "| annotations[] | `bbox` | `[x, y, w, h]` 형식의 픽셀 bbox |",
        "| annotations[] | `score` | crop quality score |",
        "| annotations[] | `gt_flag` | positive 여부. GAIC에는 0/1이 함께 존재 |",
        "| annotations[] | `iscrowd` | COCO 호환 필드, 일반적으로 0 |",
        "| annotations[] | `category_id` | `categories[]`의 `crop`를 참조 |",
        "",
        "- 즉 GAIC 포맷의 핵심은 `이미지 1행 + 그 이미지에 속한 복수 crop annotation` 구조다.",
        "- 하나의 이미지에 여러 annotation이 붙을 수 있고, 학습 시 점수 또는 `gt_flag`로 positive/negative를 구분한다.",
        "",
        "## 3. SSTK GAIC-like 변환본",
        "",
        f"- output json: `{out_json_path}`",
        f"- images: `{len(converted_images)}`",
        f"- annotations: `{len(converted_annotations)}`",
        f"- validation status: `{validation_summary.get('status')}`",
        "",
        "- SSTK 변환본은 GAIC와 같은 top-level skeleton을 유지하되, `matching_targets`는 `gt_flag=1`, `candidate_pool`은 `gt_flag=0`으로 함께 담았습니다.",
        "- 따라서 이 변환본은 safe positive와 negative/hard/unsafe 후보를 함께 보관하는 GAIC-style annotation set입니다.",
        "- GAIC 유사성을 위해 `file_name`은 경로를 제거하고 `파일명.확장자`만 저장했습니다.",
        "- SSTK 조건부 학습에 필요한 `subject_mode_id`, `decision_id`, `candidate_id`, `target_ar`는 annotation custom field로 추가했습니다.",
        "- `score_prob`는 별도 중복 필드로 두지 않고 GAIC 원본에 맞춰 `score` 하나로만 저장했습니다. positive는 `matching_target.score_prob`, negative는 `candidate_pool.score_prob`를 사용합니다.",
        "- `macro_targets`는 safe positive annotation에만 유지합니다. candidate_pool negative에는 일반적으로 존재하지 않습니다.",
        "",
        "sample converted `images[0]`:",
        "",
        "```json",
        json.dumps(converted_sample_image, ensure_ascii=False, indent=2),
        "```",
        "",
        "sample converted `annotations[0]`:",
        "",
        "```json",
        json.dumps(converted_sample_annotation, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 4. GAIC 원본과의 대응 관계",
        "",
        "| GAIC field | SSTK GAIC-like field | note |",
        "| --- | --- | --- |",
        "| `images[].file_name` | `images[].file_name` | basename only로 맞춤 |",
        "| `images[].id` | `images[].id` | `(image_id, target_ar)` 샘플 단위 id |",
        "| `annotations[].image_id` | `annotations[].image_id` | image foreign key |",
        "| `annotations[].bbox` | `annotations[].bbox` | 픽셀 `[x, y, w, h]` |",
        "| `annotations[].score` | `annotations[].score` | positive/negative 모두 SSTK local score를 매핑 |",
        "| `annotations[].gt_flag` | `annotations[].gt_flag` | `matching_targets=1`, `candidate_pool=0` |",
        "| `annotations[].category_id` | `annotations[].category_id` | `crop` 단일 category 유지 |",
        "| 없음 | `annotations[].subject_mode_id` | Conditional-DETR conditioning용 확장 필드 |",
        "| 없음 | `annotations[].decision_id` | policy supervision용 확장 필드 |",
        "| 없음 | `annotations[].candidate_id` | crop provenance 추적용 확장 필드 |",
        "| 없음 | `annotations[].target_ar` | aspect-ratio conditioned sample 식별용 확장 필드 |",
        "| 없음 | `annotations[].macro_targets` | safe positive auxiliary regression supervision용 확장 필드 |",
        "| 없음 | `annotations[].label_type` | `matching_target`, `near_negative`, `hard_negative`, `unsafe_negative` 등 provenance |",
        "| 없음 | `annotations[].is_hard_negative` | hard negative 여부 |",
        "| 없음 | `annotations[].is_unsafe_negative` | unsafe negative 여부 |",
        "",
        "## 5. 해석 시 주의점",
        "",
        "- GAIC 원본 `instances_train.json`처럼 positive/negative 후보가 함께 들어가지만, SSTK에서는 positive가 `matching_targets`, negative가 `candidate_pool`에서 왔다는 점이 다릅니다.",
        "- `score`는 positive/negative 모두 [0,1] 계열 SSTK local score를 사용하므로, GAIC 원본의 1~5대 점수 스케일과 직접 같지는 않습니다.",
        "- 이 변환본은 `GAIC-like`이지 완전한 GAIC 복제는 아닙니다. 동일한 outer format에 SSTK 전용 필드를 덧붙인 학습용 export입니다.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SSTK Conditional-DETR batch labels to a GAIC-like instances JSON with positives and candidate-pool negatives"
    )
    parser.add_argument("--batch_jsonl", required=True)
    parser.add_argument("--gaic_reference_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_summary_json", required=True)
    parser.add_argument("--out_guide_md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_jsonl = Path(args.batch_jsonl)
    gaic_reference_json = Path(args.gaic_reference_json)
    out_json = Path(args.out_json)
    out_summary_json = Path(args.out_summary_json)
    out_guide_md = Path(args.out_guide_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_guide_md.parent.mkdir(parents=True, exist_ok=True)

    batch_records = load_jsonl(batch_jsonl)
    payload = build_gaic_like_instances(batch_records)
    expected_annotation_count = sum(
        len(safe_list(record.get("matching_targets"))) + len(safe_list(record.get("candidate_pool")))
        for record in batch_records
    )
    validation_summary = validate_gaic_like_instances(
        payload,
        expected_image_count=len(batch_records),
        expected_annotation_count=expected_annotation_count,
    )
    reference = load_json(gaic_reference_json)
    write_json(out_json, payload)
    write_json(
        out_summary_json,
        {
            "out_json": str(out_json),
            "source_batch_jsonl": str(batch_jsonl),
            "gaic_reference_json": str(gaic_reference_json),
            "validation": validation_summary,
        },
    )
    out_guide_md.write_text(
        build_format_guide(reference, payload, validation_summary, out_json),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_summary_json": str(out_summary_json),
                "out_guide_md": str(out_guide_md),
                "validation": validation_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if validation_summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
