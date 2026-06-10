#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from mobilecropnet.bank import build_static_micro_bank  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def resolve_image_path(image: dict[str, Any], image_roots: list[Path], split: str) -> str:
    file_name = str(image.get("file_name", "") or "").strip()
    image_id = str(image.get("sstk_image_id", image.get("id", ""))).strip()
    candidates: list[Path] = []
    for root in image_roots:
        if file_name:
            candidates.append(root / file_name)
            candidates.append(root / split / Path(file_name).name)
            candidates.append(root / Path(file_name).name)
        if image_id:
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(root / split / f"{image_id}{ext}")
                candidates.append(root / f"{image_id}{ext}")
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0]) if candidates else file_name


def iter_rows(
    coco: dict[str, Any],
    *,
    image_roots: list[Path],
    target_ars: list[str],
    split: str,
    protocol: str,
    static_k: int,
    max_images: int,
) -> Iterable[dict[str, Any]]:
    images = list(coco.get("images") or [])
    images.sort(key=lambda item: str(item.get("sstk_image_id", item.get("id", ""))))
    if int(max_images) > 0:
        images = images[: int(max_images)]
    for image in images:
        image_id = str(image.get("sstk_image_id", image.get("id", "")))
        width = int(image.get("width", 0) or 0)
        height = int(image.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        image_path = resolve_image_path(image, image_roots, split)
        for target_ar in target_ars:
            bank = build_static_micro_bank(image_width=width, image_height=height, target_ar=str(target_ar), k=int(static_k))
            for idx, candidate in enumerate(bank):
                row = {
                    "image_id": image_id,
                    "protocol": str(protocol),
                    "official_split": str(split),
                    "calibration_split": str(split),
                    "target_ar": str(target_ar),
                    "candidate_id": candidate.candidate_id,
                    "gt_annotation_id": "",
                    "gt_flag": 0,
                    "bbox_norm_xyxy": list(candidate.bbox_norm_xyxy),
                    "mos": 0.0,
                    "subject_mode": "",
                    "mode_bucket": "",
                    "source": candidate.source,
                    "candidate_index": idx,
                    "image_path": image_path,
                    "file_name": str(image.get("file_name", "")),
                    "image_width": width,
                    "image_height": height,
                }
                yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GAIC fixed-AR static candidate rows for T6 ranker scoring.")
    parser.add_argument("--annotations_json", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, action="append", required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--target_ars", default="1:1,3:4,4:3,16:9,9:16")
    parser.add_argument("--split", required=True)
    parser.add_argument("--protocol", default="Gc")
    parser.add_argument("--static_k", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_ars = parse_csv(args.target_ars)
    coco = read_json(args.annotations_json)
    rows = list(
        iter_rows(
            coco,
            image_roots=[Path(p) for p in args.image_root],
            target_ars=target_ars,
            split=str(args.split),
            protocol=str(args.protocol),
            static_k=int(args.static_k),
            max_images=int(args.max_images),
        )
    )
    count = write_jsonl(args.output_jsonl, rows)
    image_count = len({str(row.get("image_id", "")) for row in rows})
    by_ar: dict[str, int] = {}
    for row in rows:
        key = str(row.get("target_ar", ""))
        by_ar[key] = by_ar.get(key, 0) + 1
    summary = {
        "status": "ok",
        "annotations_json": str(args.annotations_json),
        "output_jsonl": str(args.output_jsonl),
        "split": str(args.split),
        "protocol": str(args.protocol),
        "target_ars": target_ars,
        "static_k": int(args.static_k),
        "image_count": int(image_count),
        "row_count": int(count),
        "by_target_ar": by_ar,
        "note": "Rows are candidate geometry/visual inputs for T6 fixed-AR scoring; MOS is intentionally unavailable.",
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
