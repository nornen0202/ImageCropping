from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from label_artifacts.paths import LABEL_JSON_DIRNAME, MULTIMODE_LABEL_FILENAMES
from multimode.coco_writer import build_target_ar_only_coco_dataset, write_json, write_jsonl
from scripts.build_multimode_training_labels import _build_summary, _collect_label_stats, _write_csv, _write_split_outputs


TARGET_AR_ORDER = {"FREE": 0, "1:1": 1, "9:16": 2, "16:9": 3, "3:4": 4, "4:3": 5}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ids(path: Path) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _source_id(row: Dict[str, Any]) -> str:
    return str(row.get("source_image_id") or Path(str(row.get("file_name") or "")).stem)


def _annotation_target_ar(row: Dict[str, Any]) -> str:
    attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or row.get("target_ar") or "")


def _load_split_assignments(base_dir: Path) -> Dict[str, str]:
    split_dir = base_dir / "splits"
    assignments: Dict[str, str] = {}
    for split_name in ("train", "val"):
        path = split_dir / f"{split_name}_image_ids.txt"
        if not path.is_file():
            continue
        for source_id in _read_ids(path):
            assignments[source_id] = split_name
    return assignments


def _query_sort_key(row: Dict[str, Any], image_order: Dict[str, int]) -> tuple[int, int, str, str]:
    source_id = str(row.get("source_image_id") or "")
    return (
        image_order.get(source_id, 10**9),
        TARGET_AR_ORDER.get(str(row.get("target_ar") or ""), 999),
        str(row.get("mode_name") or ""),
        str(row.get("query_id") or ""),
    )


def merge_subset(
    *,
    base_dir: Path,
    subset_dir: Path,
    out_dir: Path,
    replace_ids_path: Path,
    merge_note: str,
) -> Dict[str, Any]:
    replace_ids = set(_read_ids(replace_ids_path))
    if not replace_ids:
        raise ValueError(f"replace id list is empty: {replace_ids_path}")

    label_dir = out_dir / LABEL_JSON_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    start_time = time.time()
    write_json(
        status_path,
        {
            "phase": "running",
            "base_dir": str(base_dir),
            "subset_dir": str(subset_dir),
            "out_dir": str(out_dir),
            "replace_id_count": len(replace_ids),
        },
    )

    base_label = _load_json(base_dir / LABEL_JSON_DIRNAME / MULTIMODE_LABEL_FILENAMES[("full", False)])
    subset_label = _load_json(subset_dir / LABEL_JSON_DIRNAME / MULTIMODE_LABEL_FILENAMES[("full", False)])
    base_summary = _load_json(base_dir / "summary.json")
    subset_summary = _load_json(subset_dir / "summary.json")

    base_images = [dict(row) for row in base_label.get("images", []) if isinstance(row, dict)]
    base_source_by_image_id = {int(row["id"]): _source_id(row) for row in base_images}
    base_image_by_source = {_source_id(row): dict(row) for row in base_images}
    next_image_id = max((int(row.get("id", 0)) for row in base_images), default=0) + 1

    subset_source_by_image_id = {
        int(row["id"]): _source_id(row)
        for row in subset_label.get("images", [])
        if isinstance(row, dict)
    }
    for row in subset_label.get("images", []):
        if not isinstance(row, dict):
            continue
        source_id = _source_id(row)
        if source_id not in replace_ids or source_id in base_image_by_source:
            continue
        new_row = dict(row)
        new_row["id"] = next_image_id
        next_image_id += 1
        base_images.append(new_row)
        base_image_by_source[source_id] = new_row
        base_source_by_image_id[int(new_row["id"])] = source_id

    image_id_by_source = {source_id: int(row["id"]) for source_id, row in base_image_by_source.items()}
    image_order = {source_id: idx for idx, source_id in enumerate(_source_id(row) for row in base_images)}

    kept_annotations: List[Dict[str, Any]] = []
    for ann in base_label.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        source_id = base_source_by_image_id.get(int(ann.get("image_id", -1)), "")
        if source_id in replace_ids:
            continue
        kept_annotations.append(dict(ann))

    replacement_annotations: List[Dict[str, Any]] = []
    subset_seen_sources = {source_id for source_id in subset_source_by_image_id.values() if source_id in replace_ids}
    replacement_annotation_sources = set()
    for ann in subset_label.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        source_id = subset_source_by_image_id.get(int(ann.get("image_id", -1)), "")
        if source_id not in replace_ids:
            continue
        if source_id not in image_id_by_source:
            continue
        new_ann = dict(ann)
        new_ann["image_id"] = image_id_by_source[source_id]
        replacement_annotations.append(new_ann)
        replacement_annotation_sources.add(source_id)

    missing_replacements = sorted(replace_ids - subset_seen_sources)
    replacement_without_annotations = sorted(subset_seen_sources - replacement_annotation_sources)
    merged_annotations = kept_annotations + replacement_annotations
    merged_annotations.sort(
        key=lambda ann: (
            image_order.get(base_source_by_image_id.get(int(ann.get("image_id", -1)), ""), 10**9),
            TARGET_AR_ORDER.get(_annotation_target_ar(ann), 999),
            str(ann.get("mode_name") or ""),
            int(ann.get("gt_flag", 0)) != 1,
            int(ann.get("id", 0)),
        )
    )
    for idx, ann in enumerate(merged_annotations, start=1):
        ann["id"] = idx

    categories = [dict(row) for row in base_label.get("categories", subset_label.get("categories", []))]
    merged_label = {
        "images": base_images,
        "annotations": merged_annotations,
        "categories": categories,
    }

    base_query_rows = [
        row
        for row in _iter_jsonl(base_dir / "mode_query_status.jsonl")
        if str(row.get("source_image_id") or "") not in replace_ids
    ]
    subset_query_rows = [
        row
        for row in _iter_jsonl(subset_dir / "mode_query_status.jsonl")
        if str(row.get("source_image_id") or "") in replace_ids
    ]
    query_status_rows = base_query_rows + subset_query_rows
    query_status_rows.sort(key=lambda row: _query_sort_key(row, image_order))

    full_label_json = label_dir / MULTIMODE_LABEL_FILENAMES[("full", False)]
    full_target_ar_only_label_json = label_dir / MULTIMODE_LABEL_FILENAMES[("full", True)]
    write_json(full_label_json, merged_label)
    write_json(full_target_ar_only_label_json, build_target_ar_only_coco_dataset(merged_label))
    write_json(out_dir / "categories.json", {"categories": categories})
    write_jsonl(out_dir / "mode_query_status.jsonl", query_status_rows)

    image_task_count = len({(str(row.get("source_image_id") or ""), str(row.get("target_ar") or "")) for row in query_status_rows})
    summary = _build_summary(
        image_count=int(base_summary.get("image_count", len(base_images))),
        image_task_count=image_task_count,
        image_rows=base_images,
        annotation_rows=merged_annotations,
        query_status_rows=query_status_rows,
    )
    summary["input"] = dict(base_summary.get("input") or {})
    summary["input"]["merge_base_dir"] = str(base_dir)
    summary["input"]["merge_subset_dir"] = str(subset_dir)
    summary["input"]["merge_replace_ids_path"] = str(replace_ids_path)
    summary["input"]["merge_note"] = merge_note
    summary["merge"] = {
        "base_dir": str(base_dir),
        "subset_dir": str(subset_dir),
        "replace_ids_path": str(replace_ids_path),
        "replace_image_count": len(replace_ids),
        "replacement_annotation_count": len(replacement_annotations),
        "replacement_query_count": len(subset_query_rows),
        "subset_summary": {
            "image_count": subset_summary.get("image_count"),
            "query_count": subset_summary.get("query_count"),
            "positive_query_count": subset_summary.get("positive_query_count"),
            "annotation_count": subset_summary.get("annotation_count"),
        },
        "missing_replacement_source_ids": missing_replacements[:200],
        "missing_replacement_count": len(missing_replacements),
        "replacement_without_annotation_source_ids": replacement_without_annotations[:200],
        "replacement_without_annotation_count": len(replacement_without_annotations),
    }

    stats = _collect_label_stats(
        image_rows=merged_label.get("images", []),
        annotation_rows=merged_label.get("annotations", []),
        query_status_rows=query_status_rows,
    )
    write_json(out_dir / "dataset_stats.json", stats)
    _write_csv(out_dir / "mode_stats.csv", stats["mode_stats"])
    _write_csv(out_dir / "target_ar_stats.csv", stats["target_ar_stats"])
    _write_csv(out_dir / "mode_target_ar_stats.csv", stats["mode_target_ar_stats"])

    split_assignments = _load_split_assignments(base_dir)
    if split_assignments:
        split_manifest = _write_split_outputs(
            out_dir=out_dir,
            coco_dataset=merged_label,
            query_status_rows=query_status_rows,
            split_assignments=split_assignments,
        )
        split_counter = Counter(split_assignments.values())
        summary["split"] = {
            "enabled": True,
            "train_ratio": float((base_summary.get("split") or {}).get("train_ratio", 0.9)),
            "seed": int((base_summary.get("split") or {}).get("seed", 20260506)),
            "train_image_count": int(split_counter.get("train", 0)),
            "val_image_count": int(split_counter.get("val", 0)),
            "manifest_json": str(out_dir / "splits" / "split_manifest.json"),
            "splits": split_manifest["splits"],
        }
    else:
        summary["split"] = {"enabled": False}

    write_json(out_dir / "summary.json", summary)
    write_json(
        status_path,
        {
            "phase": "completed",
            "elapsed_sec": round(time.time() - start_time, 3),
            "replace_image_count": len(replace_ids),
            "query_count": int(summary["query_count"]),
            "positive_annotation_count": int(summary["positive_annotation_count"]),
            "negative_annotation_count": int(summary["negative_annotation_count"]),
            "output_paths": {
                "summary_json": str(out_dir / "summary.json"),
                "label_json": str(full_label_json),
                "target_ar_only_label_json": str(full_target_ar_only_label_json),
                "query_status_jsonl": str(out_dir / "mode_query_status.jsonl"),
                "split_manifest_json": str(out_dir / "splits" / "split_manifest.json"),
            },
        },
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge regenerated multimode subset labels into a full base run.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--subset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--replace_ids", required=True)
    parser.add_argument("--merge_note", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = merge_subset(
        base_dir=Path(args.base_dir),
        subset_dir=Path(args.subset_dir),
        out_dir=Path(args.out_dir),
        replace_ids_path=Path(args.replace_ids),
        merge_note=str(args.merge_note),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
