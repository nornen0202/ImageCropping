#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_image_path(image_id: str, image_roots: Sequence[Path]) -> Path | None:
    names = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png"]
    for root in image_roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
        for sub in ("train", "val", "test", "images"):
            for name in names:
                p = root / sub / name
                if p.exists():
                    return p
    return None


def norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [clamp(safe_float(v), 0.0, 1.0) for v in box[:4]]
    return (int(round(x1 * width)), int(round(y1 * height)), int(round(x2 * width)), int(round(y2 * height)))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    x, y = xy
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def grouped_by_image(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("image_id", ""))].append(row)
    return dict(sorted(grouped.items()))


def visualize_group(
    image_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    image_roots: Sequence[Path],
    out_dir: Path,
    crop_dir: Path,
    top_k: int,
    split: str,
) -> dict[str, Any] | None:
    path = resolve_image_path(image_id, image_roots)
    if path is None:
        return None
    with Image.open(path) as im:
        img = im.convert("RGB")
    width, height = img.size
    rows = sorted(rows, key=lambda r: (safe_float(r.get("score_rank_pct_by_image"), 0.0), safe_float(r.get("ranker_score"), 0.0)), reverse=True)
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    palette = [(0, 220, 80), (255, 205, 0), (0, 180, 255), (255, 80, 80), (210, 90, 255)]
    selected = next((r for r in rows if r.get("selected_by_ranker")), rows[0] if rows else None)
    for rank, row in enumerate(rows[: max(1, top_k)], start=1):
        color = palette[(rank - 1) % len(palette)]
        if row is selected:
            color = (0, 255, 80)
        box = norm_to_px(row.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        stroke = 4 if row is selected else 2
        for offset in range(stroke):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
        text = "T6#{rank} pct={pct:.3f} mos={mos:.2f} {tier}".format(
            rank=rank,
            pct=safe_float(row.get("score_rank_pct_by_image"), 0.0),
            mos=safe_float(row.get("mos"), 0.0),
            tier=str(row.get("label_quality_tier", "")),
        )
        draw_label(draw, (max(0, box[0] + 4), max(0, box[1] + 4)), text, color)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    viz_path = out_dir / f"{split}_{image_id}_t6_top{top_k}.jpg"
    canvas.save(viz_path, quality=92)
    crop_path = None
    if selected is not None:
        box = norm_to_px(selected.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        crop = img.crop(box)
        crop_path = crop_dir / f"{split}_{image_id}_selected_{selected.get('candidate_id')}.jpg"
        crop.save(crop_path, quality=92)
    return {
        "image_id": image_id,
        "viz_path": str(viz_path),
        "selected_crop_path": str(crop_path) if crop_path else None,
        "candidate_count": len(rows),
        "selected_candidate_id": None if selected is None else str(selected.get("candidate_id", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize T6 score-only training labels.")
    parser.add_argument("--t6_labels_jsonl", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max_images", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.t6_labels_jsonl)
    grouped = grouped_by_image(rows)
    out_dir = args.output_dir / "debug_viz" / str(args.split)
    crop_dir = args.output_dir / "debug_best_pos_crops" / str(args.split)
    records = []
    for idx, (image_id, image_rows) in enumerate(grouped.items()):
        if int(args.max_images) > 0 and idx >= int(args.max_images):
            break
        rec = visualize_group(
            image_id,
            image_rows,
            image_roots=[Path(p) for p in args.image_root],
            out_dir=out_dir,
            crop_dir=crop_dir,
            top_k=int(args.top_k),
            split=str(args.split),
        )
        if rec is not None:
            records.append(rec)
    summary = {
        "t6_labels_jsonl": str(args.t6_labels_jsonl),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "image_count": len(records),
        "top_k": int(args.top_k),
        "records": records[:20],
    }
    write_json(args.output_dir / f"visualization_summary_{args.split}.json", summary)
    print(json.dumps({"status": "ok", "image_count": len(records), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
