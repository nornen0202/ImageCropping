#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import _subject_box_supervision_valid_from_row, _subject_prior_box_from_row

ROUTE_CLASS_NAMES = {
    "object_single": 0,
    "object_multi": 1,
    "portrait_single": 2,
    "portrait_group": 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UCTR subject bbox label을 YOLO one-class detector dataset으로 변환한다.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--copy_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--class_mode", choices=["one", "route"], default="one")
    parser.add_argument("--include_negative_images", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _image_path(row: dict[str, Any], project_root: Path) -> Path:
    raw = row.get("image_path") or row.get("path") or row.get("file_name")
    if raw is None:
        raise ValueError(f"missing image_path for image_id={row.get('image_id')!r}")
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _box_to_yolo(box: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box[:4]]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return x1 + 0.5 * w, y1 + 0.5 * h, w, h


def _route_class_id(row: dict[str, Any], *, class_mode: str) -> int:
    if str(class_mode) == "one":
        return 0
    route = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    return int(ROUTE_CLASS_NAMES.get(str(route.get("subject_mode") or ""), 0))


def _pick_unique_subject_rows(
    rows: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    max_images: int,
    class_mode: str,
    include_negative_images: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        image_id = str(row.get("image_id") or row.get("image_path"))
        if image_id in seen or image_id in exclude_ids:
            continue
        box = _subject_prior_box_from_row(row)
        valid = _subject_box_supervision_valid_from_row(row, has_box=box is not None)
        if box is None or valid <= 0.0:
            if not include_negative_images:
                continue
            row = dict(row)
            row["_subject_box"] = None
            row["_subject_class"] = None
            out.append(row)
            seen.add(image_id)
            if max_images > 0 and len(out) >= max_images:
                break
            continue
        row = dict(row)
        row["_subject_box"] = box
        row["_subject_class"] = _route_class_id(row, class_mode=class_mode)
        out.append(row)
        seen.add(image_id)
        if max_images > 0 and len(out) >= max_images:
            break
    return out


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _write_split(rows: list[dict[str, Any]], *, split: str, output_dir: Path, project_root: Path, copy_mode: str) -> dict[str, Any]:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    manifest_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        src = _image_path(row, project_root)
        suffix = src.suffix or ".jpg"
        image_id = str(row.get("image_id") or f"{split}_{idx:06d}")
        dst = image_dir / f"{image_id}{suffix}"
        _link_or_copy(src, dst, mode=copy_mode)
        label_path = label_dir / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        box = row.get("_subject_box")
        class_id = row.get("_subject_class")
        if box is None or class_id is None:
            label_path.write_text("", encoding="utf-8")
        else:
            cx, cy, w, h = _box_to_yolo(box)
            label_path.write_text(f"{int(class_id)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n", encoding="utf-8")
        manifest_rows.append(
            {
                "image_id": image_id,
                "src": str(src),
                "image": str(dst),
                "label": str(label_path),
                "box": box,
                "class_id": class_id,
            }
        )
    (output_dir / f"{split}_manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"split": split, "image_count": len(manifest_rows), "image_dir": str(image_dir), "label_dir": str(label_dir)}


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_rows_raw = _read_jsonl(args.val_jsonl)
    val_rows = _pick_unique_subject_rows(
        val_rows_raw,
        exclude_ids=set(),
        max_images=int(args.max_val_images),
        class_mode=args.class_mode,
        include_negative_images=bool(args.include_negative_images),
    )
    val_ids = {str(row.get("image_id") or row.get("image_path")) for row in val_rows_raw}
    train_rows = _pick_unique_subject_rows(
        _read_jsonl(args.train_jsonl),
        exclude_ids=val_ids,
        max_images=int(args.max_train_images),
        class_mode=args.class_mode,
        include_negative_images=bool(args.include_negative_images),
    )
    train_summary = _write_split(train_rows, split="train", output_dir=args.output_dir, project_root=args.project_root, copy_mode=args.copy_mode)
    val_summary = _write_split(val_rows, split="val", output_dir=args.output_dir, project_root=args.project_root, copy_mode=args.copy_mode)
    names = ["subject"] if args.class_mode == "one" else [name for name, _idx in sorted(ROUTE_CLASS_NAMES.items(), key=lambda item: item[1])]
    yaml_path = args.output_dir / "subject.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {args.output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "state": "completed",
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "output_dir": str(args.output_dir),
        "yaml": str(yaml_path),
        "class_mode": str(args.class_mode),
        "include_negative_images": bool(args.include_negative_images),
        "names": names,
        "train": train_summary,
        "val": val_summary,
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
