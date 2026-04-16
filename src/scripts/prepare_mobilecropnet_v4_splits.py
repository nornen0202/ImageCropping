#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare leakage-aware MobileCropNet v4 JSONL splits from current batch labels.")
    parser.add_argument("--label_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--batch_jsonl", type=Path, default=None)
    parser.add_argument("--coco_train_json", type=Path, default=None)
    parser.add_argument("--coco_test_json", type=Path, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("image_id", "")), str(row.get("target_ar", "FREE"))


def _load_coco_keys(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys: set[tuple[str, str]] = set()
    for image in payload.get("images", []):
        ar = str(image.get("target_ar", "FREE"))
        ids = [
            str(image.get("sstk_image_id", "")),
            str(image.get("id", "")),
            Path(str(image.get("file_name", ""))).stem,
        ]
        for image_id in ids:
            if image_id:
                keys.add((image_id, ar))
    return keys


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label_dir = args.label_dir
    batch_jsonl = args.batch_jsonl or label_dir / "train_conditional_detr_batch.jsonl"
    coco_train = args.coco_train_json or label_dir / "coco" / "instances_conditional_detr_batch_gaic_like_train.json"
    coco_test = args.coco_test_json or label_dir / "coco" / "instances_conditional_detr_batch_gaic_like_test.json"
    train_keys = _load_coco_keys(coco_train)
    test_keys = _load_coco_keys(coco_test)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    with batch_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _row_key(row)
            if key in test_keys:
                test_rows.append(row)
            elif key in train_keys or not train_keys:
                train_rows.append(row)
            else:
                unassigned_rows.append(row)
            if args.max_rows is not None and (len(train_rows) + len(test_rows) + len(unassigned_rows)) >= int(args.max_rows):
                break

    if not train_rows and unassigned_rows:
        train_rows = unassigned_rows
        unassigned_rows = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        grouped.setdefault(str(row.get("image_id", "")), []).append(row)
    image_ids = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * float(args.val_fraction)))) if len(image_ids) > 1 else 0
    val_ids = set(image_ids[:val_count])
    val_dev = [row for image_id in image_ids if image_id in val_ids for row in grouped[image_id]]
    train_dev = [row for image_id in image_ids if image_id not in val_ids for row in grouped[image_id]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "gaic_personv6_v4_train_dev.jsonl"
    val_path = args.output_dir / "gaic_personv6_v4_val_dev.jsonl"
    test_path = args.output_dir / "gaic_personv6_v4_test.jsonl"
    unassigned_path = args.output_dir / "gaic_personv6_v4_unassigned.jsonl"
    _write_jsonl(train_path, train_dev)
    _write_jsonl(val_path, val_dev)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(unassigned_path, unassigned_rows)

    summary = {
        "label_dir": str(label_dir),
        "batch_jsonl": str(batch_jsonl),
        "coco_train_json": str(coco_train) if coco_train.exists() else None,
        "coco_test_json": str(coco_test) if coco_test.exists() else None,
        "train_key_count": len(train_keys),
        "test_key_count": len(test_keys),
        "train_dev_rows": len(train_dev),
        "val_dev_rows": len(val_dev),
        "test_rows": len(test_rows),
        "unassigned_rows": len(unassigned_rows),
        "train_dev_path": str(train_path),
        "val_dev_path": str(val_path),
        "test_path": str(test_path),
        "unassigned_path": str(unassigned_path),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
