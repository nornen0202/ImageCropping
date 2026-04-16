#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert GAIC_v2 txt MOS annotations to COCO-style instances JSON.")
    parser.add_argument("--gaic_v2_root", type=Path, default=PROJECT_ROOT / "data/Publics/GAIC_v2")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json")
    parser.add_argument("--gt_threshold", type=float, default=4.0, help="gt_flag is 1 when MOS is greater than this value.")
    parser.add_argument("--skip_negative_mos", type=int, default=1, help="GAIC_v2 uses -2 for unrated crops; skip them by default.")
    return parser


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _parse_annotation_line(line: str) -> tuple[int, int, int, int, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    y1 = int(float(parts[0]))
    x1 = int(float(parts[1]))
    y2 = int(float(parts[2]))
    x2 = int(float(parts[3]))
    mos = float(parts[4])
    return y1, x1, y2, x2, mos


def convert_split(*, root: Path, split: str, gt_threshold: float, skip_negative_mos: bool) -> dict[str, Any]:
    ann_dir = root / "annotations" / split
    image_dir = root / "images" / split
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"annotation split directory not found: {ann_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image split directory not found: {image_dir}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    skipped_negative_mos = 0
    for txt_path in sorted(ann_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        image_id = int(txt_path.stem)
        image_path = image_dir / f"{txt_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"image not found for annotation {txt_path}: {image_path}")
        width, height = _image_size(image_path)
        images.append({"file_name": image_path.name, "height": height, "width": width, "id": image_id})
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            parsed = _parse_annotation_line(line)
            if parsed is None:
                continue
            y1, x1, y2, x2, mos = parsed
            if skip_negative_mos and float(mos) < 0.0:
                skipped_negative_mos += 1
                continue
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w <= 0 or h <= 0:
                raise ValueError(f"invalid crop box at {txt_path}:{line_no}: {line!r}")
            annotations.append(
                {
                    "area": int(w * h),
                    "image_id": image_id,
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "category_id": 0,
                    "id": ann_id,
                    "score": round(float(mos), 2),
                    "gt_flag": 1 if float(mos) > float(gt_threshold) else 0,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": [{"supercategory": "none", "id": 0, "name": "crop"}],
        "conversion_meta": {
            "source": "GAIC_v2 txt annotations",
            "split": split,
            "gt_threshold": float(gt_threshold),
            "skip_negative_mos": bool(skip_negative_mos),
            "skipped_negative_mos": int(skipped_negative_mos),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "gaic_v2_root": str(args.gaic_v2_root),
        "output_dir": str(args.output_dir),
        "gt_threshold": float(args.gt_threshold),
        "skip_negative_mos": bool(int(args.skip_negative_mos)),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        payload = convert_split(
            root=args.gaic_v2_root,
            split=split,
            gt_threshold=float(args.gt_threshold),
            skip_negative_mos=bool(int(args.skip_negative_mos)),
        )
        out_path = args.output_dir / f"instances_{split}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["splits"][split] = {
            "path": str(out_path),
            "image_count": len(payload["images"]),
            "annotation_count": len(payload["annotations"]),
            "positive_count": sum(1 for row in payload["annotations"] if int(row.get("gt_flag", 0)) == 1),
            "skipped_negative_mos": payload["conversion_meta"]["skipped_negative_mos"],
        }
    (args.output_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
