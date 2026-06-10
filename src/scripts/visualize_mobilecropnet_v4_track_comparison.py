#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    TARGET_AR_VOCAB,
    box_area,
    compute_letterbox_transform,
    letterbox_content_box,
    letterbox_to_original_box,
    load_image_tensor_letterbox,
    original_to_letterbox_box,
    target_ar_id,
    xyxy_to_cxcywh,
)
from mobilecropnet_v4.eval_utils import infer_image_norm, infer_input_size, load_mobilecropnet_v4_checkpoint
from mobilecropnet_v4.gaic_benchmark import load_gaic_annotation_records, order_desc


COLORS = [
    (196, 45, 45),
    (36, 150, 80),
    (45, 102, 210),
    (224, 138, 34),
    (170, 130, 220),
    (70, 210, 230),
]


@dataclass
class ModelSpec:
    name: str
    checkpoint: Path
    model: torch.nn.Module
    ckpt: dict[str, Any]
    input_size: int
    image_mean: Sequence[float]
    image_std: Sequence[float]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def box_meta(box: Sequence[float], *, is_base: float = 0.0) -> list[float]:
    cxcywh = xyxy_to_cxcywh(box)
    area = box_area(box)
    aspect = cxcywh[2] / max(1e-6, cxcywh[3])
    return [*box, *cxcywh, area, aspect, float(is_base)]


def parse_model_specs(values: Sequence[str], device: torch.device) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be NAME=CHECKPOINT, got {value}")
        name, raw_path = value.split("=", 1)
        checkpoint = Path(raw_path)
        model, ckpt = load_mobilecropnet_v4_checkpoint(checkpoint, device=device)
        model.eval()
        size = infer_input_size(ckpt, None)
        image_mean, image_std = infer_image_norm(ckpt)
        specs.append(ModelSpec(name=name.strip(), checkpoint=checkpoint, model=model, ckpt=ckpt, input_size=int(size), image_mean=image_mean, image_std=image_std))
    return specs


def collect_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.annotations_json is not None:
        records = load_gaic_annotation_records(args.annotations_json, image_roots=args.image_roots, max_images=args.max_images)
        for record in records:
            if str(record.get("image_path", "")).strip():
                rows.append(record)
        return rows
    paths: list[Path] = []
    for pattern in args.image_glob or []:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    for path in args.image or []:
        paths.append(Path(path))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen or not path.exists():
            continue
        seen.add(key)
        unique.append(path)
        if args.max_images is not None and len(unique) >= int(args.max_images):
            break
    for path in unique:
        with Image.open(path) as img:
            width, height = img.size
        rows.append({"image_id": path.stem, "file_name": path.name, "image_path": str(path), "width": width, "height": height, "candidates": []})
    return rows


def score_one_model(
    spec: ModelSpec,
    record: dict[str, Any],
    *,
    target_ar: str,
    device: torch.device,
) -> dict[str, Any]:
    image_path = Path(str(record.get("image_path", "")))
    image, transform = load_image_tensor_letterbox(image_path, spec.input_size, image_mean=spec.image_mean, image_std=spec.image_std)
    candidates = list(record.get("candidates") or [])
    boxes_arg = None
    valid_arg = None
    is_base_arg = None
    meta_arg = None
    candidate_payloads: list[dict[str, Any]] = []
    if candidates:
        boxes_orig = [list(c["bbox_norm_xyxy"]) for c in candidates]
        boxes_lb = [original_to_letterbox_box(box, transform) for box in boxes_orig]
        boxes_arg = torch.tensor([boxes_lb], dtype=torch.float32, device=device)
        valid_arg = torch.ones((1, len(boxes_lb)), dtype=torch.float32, device=device)
        is_base_arg = torch.zeros_like(valid_arg)
        meta_arg = torch.tensor([[box_meta(box) for box in boxes_lb]], dtype=torch.float32, device=device)
        candidate_payloads = candidates
    with torch.no_grad():
        outputs = spec.model(
            image.unsqueeze(0).to(device),
            boxes_arg,
            valid_arg,
            torch.tensor([target_ar_id(target_ar)], dtype=torch.long, device=device),
            torch.tensor([transform.image_ar_log], dtype=torch.float32, device=device),
            is_base_arg,
            meta_arg,
            torch.tensor([letterbox_content_box(transform)], dtype=torch.float32, device=device),
        )
    scores = torch.sigmoid(outputs["utility_logits"][0]).detach().cpu().tolist()
    if boxes_arg is not None:
        out_boxes_lb = boxes_arg.detach().cpu()[0].tolist()
    else:
        out_boxes_lb = outputs["proposal_boxes"].detach().cpu()[0].tolist()
    ranked = order_desc(scores)
    top_idx = ranked[0]
    top_box = letterbox_to_original_box(out_boxes_lb[top_idx], transform)
    payload: dict[str, Any] = {
        "name": spec.name,
        "checkpoint": str(spec.checkpoint),
        "score": float(scores[top_idx]),
        "rank": 1,
        "bbox_norm_xyxy": top_box,
        "candidate_index": int(top_idx),
    }
    if candidate_payloads:
        candidate = candidate_payloads[top_idx]
        payload.update(
            {
                "annotation_id": candidate.get("annotation_id"),
                "mos": safe_float(candidate.get("mos"), 0.0),
                "gt_flag": candidate.get("gt_flag"),
            }
        )
        mos = [safe_float(c.get("mos"), 0.0) for c in candidate_payloads]
        mos_order = order_desc(mos)
        payload["best_mos"] = mos[mos_order[0]] if mos_order else 0.0
        payload["mos_rank"] = int({idx: rank for rank, idx in enumerate(mos_order, 1)}.get(top_idx, len(mos)))
    return payload


def norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [safe_float(v) for v in box[:4]]
    return (
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
        int(round(max(0.0, min(1.0, x2)) * width)),
        int(round(max(0.0, min(1.0, y2)) * height)),
    )


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    tw = int(draw.textlength(text, font=font))
    th = int(getattr(font, "size", 18) * 1.35)
    draw.rectangle([x, max(0, y - th - 4), x + tw + 12, max(th + 4, y)], fill=color)
    draw.text((x + 6, max(0, y - th - 2)), text, fill=(255, 255, 255), font=font)


def panel_line(draw: ImageDraw.ImageDraw, *, x: int, y: int, text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], max_width: int) -> int:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines or [""]:
        draw.text((x, y), line, fill=fill, font=font)
        y += int(getattr(font, "size", 16) * 1.35) + 2
    return y


def render_record(record: dict[str, Any], results: list[dict[str, Any]], output_path: Path, *, max_side: int) -> None:
    image_path = Path(str(record.get("image_path", "")))
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(image.size)))
    if scale < 1.0:
        image = image.resize((int(round(image.width * scale)), int(round(image.height * scale))), Image.BILINEAR)
    panel_w = 720
    canvas_h = max(image.height, 740)
    canvas = Image.new("RGB", (image.width + panel_w, canvas_h), (18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(18, min(28, image.width // 48)))
    small = load_font(max(15, min(22, image.width // 62)))
    line_w = max(3, image.width // 360)
    for idx, result in enumerate(results):
        color = COLORS[idx % len(COLORS)]
        xy = norm_to_px(result.get("bbox_norm_xyxy", [0, 0, 1, 1]), image.width, image.height)
        draw.rectangle(xy, outline=color, width=line_w)
        draw_label(draw, (xy[0], xy[1]), f"{result['name']} {safe_float(result.get('score')):.2f}", color, small)
    px = image.width
    pad = 24
    draw.rectangle([px, 0, canvas.width, canvas.height], fill=(22, 24, 27))
    y = pad
    draw.text((px + pad, y), f"Track comparison: {record.get('image_id', '')}", fill=(255, 255, 255), font=font)
    y += 44
    for idx, result in enumerate(results):
        color = COLORS[idx % len(COLORS)]
        draw.rectangle([px + pad, y + 4, px + pad + 18, y + 22], fill=color)
        mos_text = ""
        if "mos" in result:
            mos_text = f" mos={safe_float(result.get('mos')):.2f} best={safe_float(result.get('best_mos')):.2f} rank={result.get('mos_rank', '')}"
        y = panel_line(
            draw,
            x=px + pad + 28,
            y=y,
            text=f"{result['name']}: score={safe_float(result.get('score')):.3f}{mos_text} box=[{','.join(f'{safe_float(v):.2f}' for v in result.get('bbox_norm_xyxy', [])[:4])}]",
            font=small,
            fill=(230, 230, 230),
            max_width=panel_w - 2 * pad - 28,
        )
        y += 10
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
    parser = argparse.ArgumentParser(description="Visualize side-by-side MobileCropNet v4 track inference results.")
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT. Repeat for each track.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--annotations_json", type=Path, default=None)
    parser.add_argument("--image_roots", nargs="*", type=Path, default=[])
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--image_glob", action="append", default=[])
    parser.add_argument("--target_ar", default="FREE", choices=TARGET_AR_VOCAB)
    parser.add_argument("--max_images", type=int, default=24)
    parser.add_argument("--max_side", type=int, default=1500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    specs = parse_model_specs(args.model, device)
    records = collect_images(args)
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for idx, record in enumerate(records[: int(args.max_images)], 1):
        results = [score_one_model(spec, record, target_ar=str(args.target_ar), device=device) for spec in specs]
        out = args.output_dir / "overlays" / f"{idx:04d}_{record.get('image_id', idx)}.png"
        render_record(record, results, out, max_side=int(args.max_side))
        paths.append(out)
        rows.append({"image_id": str(record.get("image_id", "")), "image_path": str(record.get("image_path", "")), "results": results, "overlay": str(out)})
    make_contact_sheet(paths, args.output_dir / "contact_sheet.png")
    manifest = {"model_count": len(specs), "image_count": len(rows), "rows": rows, "contact_sheet": str(args.output_dir / "contact_sheet.png")}
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "image_count": len(rows), "contact_sheet": str(args.output_dir / "contact_sheet.png")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
