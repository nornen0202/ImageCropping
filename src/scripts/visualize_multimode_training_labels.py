#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ar_dir_name(target_ar: str) -> str:
    return str(target_ar).replace(":", "x").replace("/", "_")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def box_xywh_to_xyxy(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    x, y, w, h = [safe_float(v) for v in bbox[:4]]
    return (int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))


def resolve_image_path(image: dict[str, Any], image_root: Path) -> Path:
    file_name = str(image.get("file_name", ""))
    source = str(image.get("source_image_id", ""))
    candidates = [
        image_root / file_name,
        image_root / f"{source}.jpg",
        image_root / "train" / f"{source}.jpg",
        image_root / "val" / f"{source}.jpg",
        image_root / "test" / f"{source}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def visualize_image(
    image: dict[str, Any],
    anns: Sequence[dict[str, Any]],
    *,
    image_root: Path,
    output_dir: Path,
    crop_dir: Path,
    max_ann_per_image: int,
) -> list[dict[str, Any]]:
    path = resolve_image_path(image, image_root)
    if not path.exists():
        return []
    with Image.open(path) as im:
        img = im.convert("RGB")
    by_ar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        by_ar[str(attrs.get("target_ar", "FREE"))].append(ann)
    records = []
    for target_ar, ar_anns in sorted(by_ar.items()):
        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)
        positives = [a for a in ar_anns if int(a.get("gt_flag", 0)) == 1]
        negatives = [a for a in ar_anns if int(a.get("gt_flag", 0)) == 0]
        selected = sorted(positives, key=lambda a: safe_float(a.get("score_mode"), 0.0), reverse=True)
        shown = selected[: max_ann_per_image] + negatives[: max(0, max_ann_per_image - len(selected[:max_ann_per_image]))]
        for ann in shown:
            pos = int(ann.get("gt_flag", 0)) == 1
            color = (0, 230, 80) if pos else (255, 80, 80)
            box = box_xywh_to_xyxy(ann.get("bbox") or [0, 0, 1, 1])
            stroke = 3 if pos else 2
            for offset in range(stroke):
                draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
            text = "{mode} {kind} s={score:.3f}".format(
                mode=str(ann.get("mode_name", "")),
                kind="POS" if pos else "NEG",
                score=safe_float(ann.get("score_mode"), 0.0),
            )
            draw_label(draw, (max(0, box[0] + 4), max(0, box[1] + 4)), text, color)
        ar_name = ar_dir_name(target_ar)
        out_ar = output_dir / "debug_viz" / ar_name
        crop_ar = crop_dir / "debug_best_pos_crops" / ar_name
        out_ar.mkdir(parents=True, exist_ok=True)
        crop_ar.mkdir(parents=True, exist_ok=True)
        source = str(image.get("source_image_id", image.get("id", "")))
        viz_path = out_ar / f"{source}_{ar_name}_multimode.jpg"
        canvas.save(viz_path, quality=92)
        for ann in selected[: max_ann_per_image]:
            box = box_xywh_to_xyxy(ann.get("bbox") or [0, 0, 1, 1])
            crop = img.crop(box)
            crop_path = crop_ar / f"{source}_{ar_name}_{ann.get('mode_name')}_{ann.get('id')}.jpg"
            crop.save(crop_path, quality=92)
        records.append({"source_image_id": source, "target_ar": target_ar, "viz_path": str(viz_path), "positive_count": len(positives)})
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize multimode annotation label JSON.")
    parser.add_argument("--multimode_label_json", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=32)
    parser.add_argument("--max_ann_per_image", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_json = args.multimode_label_json
    data = load_json(label_json)
    images = {int(img.get("id")): img for img in data.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[int(ann.get("image_id", -1))].append(ann)
    records = []
    for idx, image_id in enumerate(sorted(anns_by_image)):
        if int(args.max_images) > 0 and idx >= int(args.max_images):
            break
        image = images.get(image_id)
        if not image:
            continue
        records.extend(
            visualize_image(
                image,
                anns_by_image[image_id],
                image_root=args.image_root,
                output_dir=args.output_dir,
                crop_dir=args.output_dir,
                max_ann_per_image=int(args.max_ann_per_image),
            )
        )
    summary = {
        "multimode_label_json": str(label_json),
        "output_dir": str(args.output_dir),
        "image_count": len({r["source_image_id"] for r in records}),
        "visualization_count": len(records),
        "records": records[:20],
    }
    write_json(args.output_dir / "visualization_summary_multimode.json", summary)
    print(json.dumps({"status": "ok", "image_count": summary["image_count"], "visualization_count": len(records), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
