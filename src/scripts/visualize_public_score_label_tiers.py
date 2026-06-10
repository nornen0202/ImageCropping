#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.gaic_benchmark import load_gaic_annotation_records


TIER_COLORS: dict[str, tuple[int, int, int]] = {
    "main_positive": (36, 150, 80),
    "score_explanation_conflict": (224, 138, 34),
    "gaic_public_positive_sstk_fatal": (196, 45, 45),
    "normal_negative": (85, 110, 210),
    "low_weight_positive": (170, 130, 220),
    "hard_negative": (150, 40, 40),
    "candidate": (130, 130, 130),
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_map_from_annotations(annotation_json: Path, image_roots: Sequence[Path]) -> dict[str, str]:
    records = load_gaic_annotation_records(annotation_json, image_roots=image_roots)
    return {str(record["image_id"]): str(record.get("image_path", "")) for record in records if str(record.get("image_path", "")).strip()}


def norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [safe_float(v) for v in box[:4]]
    return (
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
        int(round(max(0.0, min(1.0, x2)) * width)),
        int(round(max(0.0, min(1.0, y2)) * height)),
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    tw = int(draw.textlength(text, font=font))
    th = int(getattr(font, "size", 18) * 1.35)
    draw.rectangle([x, max(0, y - th - 4), x + tw + 12, max(th + 4, y)], fill=color)
    draw.text((x + 6, max(0, y - th - 2)), text, fill=(255, 255, 255), font=font)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def panel_line(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += int(getattr(font, "size", 16) * 1.35) + 2
    return y


def grouped_by_image(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("image_id", ""))].append(row)
    for image_rows in grouped.values():
        image_rows.sort(
            key=lambda row: (
                bool(row.get("selected_by_ranker", False)),
                safe_float(row.get("score_rank_pct_by_image"), 0.0),
                safe_float(row.get("ranker_score"), 0.0),
            ),
            reverse=True,
        )
    return grouped


def best_row(rows: Sequence[dict[str, Any]], predicate) -> dict[str, Any] | None:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            safe_float(row.get("score_rank_pct_by_image"), 0.0),
            safe_float(row.get("ranker_score"), 0.0),
            safe_float(row.get("mos"), 0.0),
        ),
    )


def row_summary(row: dict[str, Any] | None, prefix: str) -> list[str]:
    if not row:
        return [f"{prefix}: none"]
    return [
        f"{prefix}: {row.get('candidate_id', '')}",
        f"  tier={row.get('label_quality_tier', '')} selected={bool(row.get('selected_by_ranker', False))}",
        f"  pct={safe_float(row.get('score_rank_pct_by_image')):.3f} raw={safe_float(row.get('ranker_score')):.3f} mos={safe_float(row.get('mos')):.2f}",
        f"  expl={safe_float(row.get('explanation_score')):.3f} conf={safe_float(row.get('explanation_confidence')):.3f} fatal={bool(row.get('fatal_flag', False))} contradiction={bool(row.get('contradiction_flag', False))}",
        f"  warnings={','.join(str(v) for v in (row.get('warnings') or [])[:6]) or 'none'}",
    ]


def render(
    *,
    image_path: Path,
    rows: Sequence[dict[str, Any]],
    focus: dict[str, Any],
    output_path: Path,
    max_side: int,
) -> None:
    with Image.open(image_path) as img:
        base = img.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(base.size)))
    if scale < 1.0:
        base = base.resize((int(round(base.width * scale)), int(round(base.height * scale))), Image.BILINEAR)
    panel_w = 720
    canvas_h = max(base.height, 1180)
    canvas = Image.new("RGB", (base.width + panel_w, canvas_h), (18, 18, 18))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(18, min(28, base.width // 48)))
    small = load_font(max(15, min(22, base.width // 62)))
    panel_font = load_font(22)
    px = base.width
    draw.rectangle([px, 0, canvas.width, canvas.height], fill=(22, 24, 27))
    line_w = max(3, base.width // 360)

    tier = str(focus.get("label_quality_tier", "candidate"))
    public_top = best_row(rows, lambda row: bool(row.get("selected_by_ranker", False)))
    main_pos = best_row(rows, lambda row: str(row.get("label_quality_tier", "")) == "main_positive")
    mos_best = max(rows, key=lambda row: safe_float(row.get("mos"), 0.0), default=None)
    overlays = [
        ("focus", focus, TIER_COLORS.get(tier, (230, 230, 230))),
        ("public top", public_top, (255, 220, 80)),
        ("main positive", main_pos, TIER_COLORS["main_positive"]),
        ("MOS best", mos_best, (70, 210, 230)),
    ]
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for name, row, color in overlays:
        if not row or not row.get("bbox_norm_xyxy"):
            continue
        box = [safe_float(v) for v in row["bbox_norm_xyxy"][:4]]
        key = (name, tuple(round(v, 5) for v in box))
        if key in seen:
            continue
        seen.add(key)
        xy = norm_to_px(box, base.width, base.height)
        draw.rectangle(xy, outline=color, width=line_w)
        draw_label(draw, (xy[0], xy[1]), f"{name} {safe_float(row.get('score_rank_pct_by_image')):.2f}", color, small)

    pad = 24
    y = pad
    draw.text((px + pad, y), f"Public score tier: {tier}", fill=TIER_COLORS.get(tier, (255, 255, 255)), font=panel_font)
    y += 40
    y = panel_line(draw, x=px + pad, y=y, text=f"image={focus.get('image_id', '')} split={focus.get('split', '')} mode={focus.get('subject_mode', '')}", font=small, fill=(230, 230, 230), max_width=panel_w - 2 * pad)
    y += 14
    for title, row in (
        ("FOCUS", focus),
        ("PUBLIC_TOP", public_top),
        ("MAIN_POSITIVE", main_pos),
        ("GAIC_MOS_BEST", mos_best),
    ):
        for line in row_summary(row, title):
            y = panel_line(draw, x=px + pad, y=y, text=line, font=small, fill=(225, 225, 225), max_width=panel_w - 2 * pad)
        y += 8
    labels = focus.get("checklist_labels") if isinstance(focus.get("checklist_labels"), dict) else {}
    if labels:
        y += 8
        draw.text((px + pad, y), "SSTK checklist labels", fill=(255, 255, 255), font=panel_font)
        y += 36
        for key in sorted(labels):
            y = panel_line(draw, x=px + pad, y=y, text=f"{key}: {labels[key]}", font=small, fill=(205, 220, 235), max_width=panel_w - 2 * pad)
            if y > canvas_h - 40:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
        thumb.thumbnail((420, 320), Image.BILINEAR)
        tile = Image.new("RGB", (420, 320), (18, 18, 18))
        tile.paste(thumb, ((420 - thumb.width) // 2, (320 - thumb.height) // 2))
        thumbs.append(tile)
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 320), (10, 10, 10))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 420, (idx // cols) * 320))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GAIC public-score + SSTK-explanation label tiers.")
    parser.add_argument("--labels_jsonl", type=Path, required=True)
    parser.add_argument("--annotations_json", type=Path, required=True)
    parser.add_argument("--image_roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tiers", default="main_positive,score_explanation_conflict,gaic_public_positive_sstk_fatal,normal_negative")
    parser.add_argument("--max_per_tier", type=int, default=40)
    parser.add_argument("--seed", type=int, default=260420)
    parser.add_argument("--max_side", type=int, default=1500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.labels_jsonl)
    by_image = grouped_by_image(rows)
    images = image_map_from_annotations(args.annotations_json, args.image_roots)
    rng = random.Random(args.seed)
    tiers = [part.strip() for part in str(args.tiers).split(",") if part.strip()]
    manifest: dict[str, Any] = {"labels_jsonl": str(args.labels_jsonl), "tiers": {}, "output_dir": str(args.output_dir)}
    for tier in tiers:
        tier_rows = [row for row in rows if str(row.get("label_quality_tier", "")) == tier and str(row.get("image_id", "")) in images]
        tier_rows.sort(key=lambda row: (bool(row.get("selected_by_ranker", False)), safe_float(row.get("score_rank_pct_by_image"), 0.0), safe_float(row.get("ranker_score"), 0.0)), reverse=True)
        if len(tier_rows) > int(args.max_per_tier):
            head = tier_rows[: max(1, int(args.max_per_tier) // 2)]
            tail = tier_rows[max(1, int(args.max_per_tier) // 2) :]
            rng.shuffle(tail)
            selected = head + tail[: int(args.max_per_tier) - len(head)]
        else:
            selected = tier_rows
        out_paths: list[Path] = []
        for idx, row in enumerate(selected, 1):
            image_id = str(row.get("image_id", ""))
            out = args.output_dir / tier / "overlays" / f"{idx:04d}_{image_id}_{row.get('candidate_id', '')}.png"
            render(image_path=Path(images[image_id]), rows=by_image[image_id], focus=row, output_path=out, max_side=int(args.max_side))
            out_paths.append(out)
        make_contact_sheet(out_paths, args.output_dir / tier / "contact_sheet.png")
        manifest["tiers"][tier] = {"available_rows": len(tier_rows), "visualized": len(out_paths), "contact_sheet": str(args.output_dir / tier / "contact_sheet.png")}
    write_json(args.output_dir / "visualization_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
