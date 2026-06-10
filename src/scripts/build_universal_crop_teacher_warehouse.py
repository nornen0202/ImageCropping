#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.gaic_benchmark import load_gaic_annotation_records  # noqa: E402
from universal_crop_teacher.data import write_jsonl  # noqa: E402
from universal_crop_teacher.warehouse import (  # noqa: E402
    gaic_record_to_warehouse_row,
    public_task_to_warehouse_row,
    read_jsonl,
    summarize_warehouse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UniversalCropTeacher-H candidate warehouse JSONL.")
    parser.add_argument("--public_task_manifest_jsonl", type=Path, default=None)
    parser.add_argument("--gaic_annotations_json", type=Path, default=PROJECT_ROOT / "data/Publics/GAIC/annotations_json/instances_test.json")
    parser.add_argument("--gaic_annotations_jsons", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--gaic_image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC/images/train",
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/train",
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/val",
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
        ],
    )
    parser.add_argument("--include_public", type=int, default=1)
    parser.add_argument("--include_gaic", type=int, default=1)
    parser.add_argument("--max_public_tasks_per_dataset", type=int, default=0)
    parser.add_argument("--max_gaic_images", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=20260420)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--force_public_hash_split", type=int, default=0)
    parser.add_argument("--gaic_min_mos_gap", type=float, default=0.20)
    parser.add_argument("--gaic_max_pairs_per_task", type=int, default=256)
    parser.add_argument("--gaic_preserve_official_split", type=int, default=1)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", default="", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    rows: list[dict[str, Any]] = []
    public_seen_by_dataset: dict[str, int] = {}
    if int(args.include_public) > 0 and args.public_task_manifest_jsonl is not None:
        for task in read_jsonl(args.public_task_manifest_jsonl):
            dataset = str(task.get("dataset", "unknown")).lower()
            seen = public_seen_by_dataset.get(dataset, 0)
            if int(args.max_public_tasks_per_dataset) > 0 and seen >= int(args.max_public_tasks_per_dataset):
                continue
            row = public_task_to_warehouse_row(
                task,
                split_seed=int(args.split_seed),
                val_fraction=float(args.val_fraction),
                test_fraction=float(args.test_fraction),
                force_hash_split=bool(int(args.force_public_hash_split) > 0),
            )
            rows.append(row)
            public_seen_by_dataset[dataset] = seen + 1

    if int(args.include_gaic) > 0:
        gaic_annotation_jsons = [Path(path) for path in args.gaic_annotations_jsons if str(path).strip()]
        if not gaic_annotation_jsons and args.gaic_annotations_json is not None:
            gaic_annotation_jsons = [args.gaic_annotations_json]
        loaded_images = 0
        max_images_total = int(args.max_gaic_images) if int(args.max_gaic_images) > 0 else None
        for annotation_json in gaic_annotation_jsons:
            if not Path(annotation_json).exists():
                continue
            remaining = None if max_images_total is None else max(0, max_images_total - loaded_images)
            if remaining == 0:
                break
            records = load_gaic_annotation_records(annotation_json, image_roots=args.gaic_image_roots, max_images=remaining)
            loaded_images += len(records)
            for record in records:
                rows.append(
                    gaic_record_to_warehouse_row(
                        record,
                        split_seed=int(args.split_seed),
                        val_fraction=float(args.val_fraction),
                        test_fraction=float(args.test_fraction),
                        min_mos_gap=float(args.gaic_min_mos_gap),
                        max_pairs=int(args.gaic_max_pairs_per_task),
                        preserve_official_split=bool(int(args.gaic_preserve_official_split) > 0),
                    )
                )

    written = write_jsonl(args.output_jsonl.resolve(), rows)
    summary = summarize_warehouse(rows)
    summary.update(
        {
            "output_jsonl": str(args.output_jsonl.resolve()),
            "written_rows": int(written),
            "duration_sec": round(time.time() - start, 3),
            "inputs": {
                "public_task_manifest_jsonl": str(args.public_task_manifest_jsonl.resolve()) if args.public_task_manifest_jsonl else "",
                "gaic_annotations_json": str(args.gaic_annotations_json.resolve()) if args.gaic_annotations_json else "",
                "gaic_annotations_jsons": [str(Path(path).resolve()) for path in gaic_annotation_jsons],
                "include_public": bool(int(args.include_public) > 0),
                "include_gaic": bool(int(args.include_gaic) > 0),
                "max_public_tasks_per_dataset": int(args.max_public_tasks_per_dataset),
                "max_gaic_images": int(args.max_gaic_images),
                "force_public_hash_split": bool(int(args.force_public_hash_split) > 0),
                "gaic_preserve_official_split": bool(int(args.gaic_preserve_official_split) > 0),
            },
        }
    )
    if str(args.summary_json).strip():
        args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
