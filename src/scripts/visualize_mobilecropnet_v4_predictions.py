#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write high-quality PNG visualizations for MobileCropNet v4 predictions.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--max_side", type=int, default=1400)
    return parser


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _box_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
        int(round(max(0.0, min(1.0, x2)) * width)),
        int(round(max(0.0, min(1.0, y2)) * height)),
    )


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], size: tuple[int, int], color: tuple[int, int, int], label: str, font: ImageFont.ImageFont, width: int) -> None:
    w, h = size
    xy = _box_px(box, w, h)
    draw.rectangle(xy, outline=color, width=width)
    tw = int(draw.textlength(label, font=font))
    th = int(getattr(font, "size", 18) * 1.35)
    x1, y1, _, _ = xy
    bg = [x1, max(0, y1 - th - 4), min(w, x1 + tw + 12), max(th + 4, y1)]
    draw.rectangle(bg, fill=color)
    draw.text((bg[0] + 6, bg[1] + 2), label, fill=(255, 255, 255), font=font)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text) or "na"


def _render_row(row: dict[str, Any], output_path: Path, *, max_side: int) -> None:
    image_path = Path(str(row.get("image_path", "")))
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(canvas.size)))
    if scale < 1.0:
        canvas = canvas.resize((int(round(canvas.width * scale)), int(round(canvas.height * scale))), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, int(round(max(canvas.size) / 350)))
    font = _load_font(max(18, int(round(max(canvas.size) / 42))))
    small_font = _load_font(max(15, int(round(max(canvas.size) / 55))))

    best = row.get("best_positive") if isinstance(row.get("best_positive"), dict) else None
    top = row.get("top_candidate") if isinstance(row.get("top_candidate"), dict) else None
    base = None
    for cand in row.get("candidates", []):
        if float(cand.get("is_base", 0.0)) > 0:
            base = cand
            break
    proposal = row.get("proposals", [{}])[0] if row.get("proposals") else None
    if isinstance(best, dict) and best.get("bbox_norm_xyxy"):
        _draw_box(draw, best["bbox_norm_xyxy"], canvas.size, (36, 150, 80), f"label {float(best.get('score', 0.0)):.2f}", small_font, line_width)
    if isinstance(base, dict) and base.get("bbox_norm_xyxy"):
        _draw_box(draw, base["bbox_norm_xyxy"], canvas.size, (45, 102, 210), "baseline", small_font, line_width)
    if isinstance(proposal, dict) and proposal.get("bbox_norm_xyxy"):
        _draw_box(draw, proposal["bbox_norm_xyxy"], canvas.size, (224, 138, 34), f"proposal {float(proposal.get('score', 0.0)):.2f}", small_font, line_width)
    if isinstance(top, dict) and top.get("bbox_norm_xyxy"):
        _draw_box(draw, top["bbox_norm_xyxy"], canvas.size, (196, 45, 45), f"v4 {float(top.get('model_utility', 0.0)):.2f}", small_font, line_width)

    metrics = row.get("metrics", {})
    title = "{image_id} | {ar} | hit={hit:.0f} ndcg5={ndcg:.2f} pIoU5={piou:.2f}".format(
        image_id=row.get("image_id", ""),
        ar=row.get("target_ar", "FREE"),
        hit=float(metrics.get("candidate_top1_hit", 0.0)),
        ndcg=float(metrics.get("ndcg_at_5", 0.0)),
        piou=float(metrics.get("proposal_best_iou_at_5", 0.0)),
    )
    pad = max(12, int(round(canvas.width / 90)))
    text_w = int(draw.textlength(title, font=font))
    text_h = int(getattr(font, "size", 20) * 1.45)
    draw.rectangle([0, 0, min(canvas.width, text_w + 2 * pad), text_h + pad], fill=(0, 0, 0))
    draw.text((pad, pad // 2), title, fill=(255, 255, 255), font=font)
    canvas.save(output_path, format="PNG", compress_level=3)


def _make_contact_sheet(paths: list[Path], output_path: Path) -> None:
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
    sheet.save(output_path, format="PNG", compress_level=3)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.predictions_jsonl)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: min(args.sample_size, len(rows))]
    output_paths = []
    for idx, row in enumerate(selected, 1):
        safe_id = _safe_name(row.get("image_id", idx))
        safe_ar = _safe_name(row.get("target_ar", "FREE"))
        out = overlay_dir / f"{idx:04d}_{safe_id}_{safe_ar}.png"
        _render_row(row, out, max_side=args.max_side)
        output_paths.append(out)
    _make_contact_sheet(output_paths, args.output_dir / "contact_sheet.png")
    manifest = {"prediction_count": len(rows), "sample_count": len(output_paths), "overlay_dir": str(overlay_dir), "contact_sheet": str(args.output_dir / "contact_sheet.png")}
    (args.output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
