from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.progress_utils import progress_log


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _jsonl_image_ids(path: Path) -> set[str]:
    return {str(row.get("image_id", "")).strip() for row in _iter_jsonl(path) if str(row.get("image_id", "")).strip()}


def _image_root_ids(path: Path) -> set[str]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return {row.stem for row in path.rglob("*") if row.is_file() and row.suffix.lower() in suffixes}


def _source_image_ids(coco: Dict[str, Any]) -> set[str]:
    images = coco.get("images") if isinstance(coco.get("images"), list) else []
    return {str(row.get("source_image_id", "")).strip() for row in images if str(row.get("source_image_id", "")).strip()}


def _validate_mode_rows(summary: Dict[str, Any], query_status_jsonl: Path) -> Dict[str, Any]:
    mode_counter = Counter()
    positive_counter = Counter()
    row_count = 0
    with query_status_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row_count += 1
            row = json.loads(line)
            mode_name = str(row.get("mode_name", ""))
            mode_counter[mode_name] += 1
            if int(row.get("positive_exists", 0)) == 1:
                positive_counter[mode_name] += 1
    mode_summary = summary.get("mode_summary") if isinstance(summary.get("mode_summary"), dict) else {}
    mismatches = []
    for mode_name, payload in sorted(mode_summary.items()):
        if not isinstance(payload, dict):
            continue
        expected_queries = int(payload.get("query_count", 0))
        expected_positives = int(payload.get("positive_query_count", 0))
        if mode_counter.get(mode_name, 0) != expected_queries:
            mismatches.append(f"{mode_name}:query_count")
        if positive_counter.get(mode_name, 0) != expected_positives:
            mismatches.append(f"{mode_name}:positive_query_count")
    return {
        "row_count": row_count,
        "mode_query_counts": dict(mode_counter),
        "mode_positive_counts": dict(positive_counter),
        "mismatches": mismatches,
    }


def _summary_int(summary: Dict[str, Any], key: str, *fallback_path: str, default: int = -1) -> int:
    value = summary.get(key)
    if value is None and fallback_path:
        cursor: Any = summary
        for part in fallback_path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(part)
        value = cursor
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multimode training label outputs.")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--query_status_jsonl", required=True)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--features_jsonl", default="")
    parser.add_argument("--candidates_jsonl", default="")
    parser.add_argument("--expected_source_images", type=int, default=0)
    parser.add_argument("--allow_partial", type=int, default=1)
    parser.add_argument("--split_manifest_json", default="")
    parser.add_argument("--progress", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary_json = Path(args.summary_json)
    label_json = Path(args.label_json)
    query_status_jsonl = Path(args.query_status_jsonl)
    for path in (summary_json, label_json, query_status_jsonl):
        if not path.is_file():
            raise FileNotFoundError(f"required file missing: {path}")
    summary = _load_json(summary_json)
    label_payload = _load_json(label_json)
    mode_validation = _validate_mode_rows(summary, query_status_jsonl)
    query_row_count = int(mode_validation["row_count"])
    expected_query_count = _summary_int(summary, "query_count", "split_summaries", "mode_query_status", "rows")
    if expected_query_count != query_row_count:
        raise ValueError(
            f"query_count mismatch summary={expected_query_count} query_status_rows={query_row_count}"
        )
    images = label_payload.get("images") if isinstance(label_payload.get("images"), list) else []
    annotations = label_payload.get("annotations") if isinstance(label_payload.get("annotations"), list) else []
    categories = label_payload.get("categories") if isinstance(label_payload.get("categories"), list) else []
    expected_image_entries = _summary_int(
        summary,
        "source_image_entry_count",
        "split_summaries",
        "multimode_full",
        "images",
        default=_summary_int(summary, "coco_image_count"),
    )
    if expected_image_entries != len(images):
        raise ValueError("source image entry count mismatch")
    expected_annotation_count = _summary_int(summary, "annotation_count", "split_summaries", "multimode_full", "annotations")
    if expected_annotation_count != len(annotations):
        raise ValueError("annotation count mismatch")
    if len(categories) <= 0:
        raise ValueError("categories is empty")
    if mode_validation["mismatches"]:
        raise ValueError("mode summary mismatch: " + ",".join(mode_validation["mismatches"]))
    coverage_payload: Dict[str, Any] = {}
    source_ids = _source_image_ids(label_payload)
    expected_source_images = int(args.expected_source_images)
    if args.image_root:
        root_ids = _image_root_ids(Path(args.image_root))
        coverage_payload["image_root_count"] = len(root_ids)
        coverage_payload["source_images_missing_from_label_json"] = len(root_ids - source_ids)
        coverage_payload["label_images_missing_from_image_root"] = len(source_ids - root_ids)
        if expected_source_images > 0 and len(root_ids) != expected_source_images:
            raise ValueError(f"image root count mismatch expected={expected_source_images} actual={len(root_ids)}")
        if int(args.allow_partial) == 0 and root_ids - source_ids:
            raise ValueError(f"partial multimode labels: missing_source_images={len(root_ids - source_ids)}")
    elif expected_source_images > 0 and len(source_ids) != expected_source_images and int(args.allow_partial) == 0:
        raise ValueError(f"label source image count mismatch expected={expected_source_images} actual={len(source_ids)}")
    if args.features_jsonl:
        feature_ids = _jsonl_image_ids(Path(args.features_jsonl))
        coverage_payload["feature_unique_images"] = len(feature_ids)
        coverage_payload["label_images_missing_from_features"] = len(source_ids - feature_ids)
    if args.candidates_jsonl:
        candidate_ids = _jsonl_image_ids(Path(args.candidates_jsonl))
        coverage_payload["candidate_unique_images"] = len(candidate_ids)
        coverage_payload["label_images_missing_from_candidates"] = len(source_ids - candidate_ids)

    split_payload: Dict[str, Any] = {}
    if args.split_manifest_json:
        split_manifest_path = Path(args.split_manifest_json)
        split_manifest = _load_json(split_manifest_path)
        split_payload["split_manifest_json"] = str(split_manifest_path)
        split_ids: set[str] = set()
        for split_name, item in sorted((split_manifest.get("splits") or {}).items()):
            split_label_path = Path(str(item.get("label_json") or item.get("coco_json", "")))
            if not split_label_path.is_file():
                raise FileNotFoundError(f"split label JSON missing: {split_label_path}")
            split_label_payload = _load_json(split_label_path)
            ids = _source_image_ids(split_label_payload)
            overlap = split_ids & ids
            if overlap:
                raise ValueError(f"split image overlap detected split={split_name} overlap_count={len(overlap)}")
            split_ids |= ids
            split_payload[split_name] = {
                "image_count": len(ids),
                "annotation_count": len(
                    split_label_payload.get("annotations", [])
                    if isinstance(split_label_payload.get("annotations"), list)
                    else []
                ),
            }
        if split_ids != source_ids:
            raise ValueError(f"split union mismatch source={len(source_ids)} split_union={len(split_ids)}")
    payload = {
        "ok": True,
        "summary_json": str(summary_json),
        "label_json": str(label_json),
        "query_status_jsonl": str(query_status_jsonl),
        "query_count": query_row_count,
        "source_image_entry_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "mode_query_counts": mode_validation["mode_query_counts"],
        "mode_positive_counts": mode_validation["mode_positive_counts"],
        "coverage": coverage_payload,
        "splits": split_payload,
    }
    progress_log(
        f"multimode validation ok images={len(images)} annotations={len(annotations)} queries={query_row_count}",
        enabled=bool(args.progress),
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
