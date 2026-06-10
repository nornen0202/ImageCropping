#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet.eval_utils import infer_input_size, load_label_crop_records, load_mobilecropnet_checkpoint, score_crop_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save qualitative MobileCropNet prediction overlays.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label_json", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--candidate_source", choices=["label", "static"], default="label")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--static_k", type=int, default=None)
    parser.add_argument("--static_area_prior", type=float, default=None)
    parser.add_argument("--static_area_penalty", type=float, default=0.0)
    parser.add_argument("--static_full_penalty", type=float, default=0.0)
    parser.add_argument("--sample_size", type=int, default=12)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--target_ar_filter", default=None)
    parser.add_argument("--save_topk", type=int, default=5)
    parser.add_argument("--image_format", choices=["png"], default="png")
    parser.add_argument("--font_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _box_px(box: list[float], *, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box[:4]]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    *,
    box: list[float],
    image_size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    width: int = 3,
) -> None:
    img_w, img_h = image_size
    px = _box_px(box, width=img_w, height=img_h)
    draw.rectangle(px, outline=color, width=width)
    if not label:
        return
    text_pad = max(4, int(width * 2))
    bbox_probe = draw.textbbox((0, 0), label, font=font)
    text_h = bbox_probe[3] - bbox_probe[1]
    text_pos = (px[0] + text_pad, max(0, px[1] - text_h - text_pad * 2))
    bbox = draw.textbbox(text_pos, label, font=font)
    pad = max(4, int(width * 1.5))
    bg = (max(0, color[0] - 50), max(0, color[1] - 50), max(0, color[2] - 50))
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text(text_pos, label, fill=(255, 255, 255), font=font)


def _draw_overlay(pred: dict[str, Any], *, output_path: Path, save_topk: int, font_scale: float) -> None:
    with Image.open(pred["image_path"]) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_size = canvas.size
    min_side = max(1, min(image_size))
    label_font = _load_font(max(18, int(min_side * 0.030 * float(font_scale))))
    title_font = _load_font(max(22, int(min_side * 0.036 * float(font_scale))))
    heavy_width = max(5, int(min_side * 0.008))
    mid_width = max(3, int(min_side * 0.005))
    thin_width = max(2, int(min_side * 0.004))
    ranked = pred["ranked_indices"]
    candidates = pred["candidates"]

    best_label = pred.get("best_label_candidate") if isinstance(pred.get("best_label_candidate"), dict) else None
    if best_label:
        _draw_labeled_box(
            draw,
            box=best_label["bbox_norm_xyxy"],
            image_size=image_size,
            color=(0, 180, 80),
            label=f"label best {best_label.get('label_score', 0.0):.3f}",
            font=label_font,
            width=heavy_width,
        )

    palette = [(230, 55, 55), (255, 175, 45), (255, 225, 65), (45, 130, 220), (145, 105, 230)]
    for rank, idx in reversed(list(enumerate(ranked[: min(save_topk, len(ranked))], 1))):
        cand = candidates[idx]
        color = palette[min(rank - 1, len(palette) - 1)]
        _draw_labeled_box(
            draw,
            box=cand["bbox_norm_xyxy"],
            image_size=image_size,
            color=color,
            label=f"m{rank} {cand.get('model_score', 0.0):.3f}",
            font=label_font,
            width=heavy_width if rank == 1 else thin_width,
        )

    anchor_box = pred.get("anchor_box")
    if isinstance(anchor_box, list) and len(anchor_box) >= 4:
        _draw_labeled_box(
            draw,
            box=anchor_box,
            image_size=image_size,
            color=(0, 190, 210),
            label="anchor",
            font=label_font,
            width=mid_width,
        )

    title = f"{pred['image_id']} ar={pred['target_ar']} src={pred['candidate_source']} iou={pred['top_iou_to_best_label']:.3f}"
    title_pad = max(8, int(min_side * 0.012))
    title_bbox = draw.textbbox((title_pad, title_pad), title, font=title_font)
    draw.rectangle(
        (title_pad // 2, title_pad // 2, title_bbox[2] + title_pad, title_bbox[3] + title_pad),
        fill=(0, 0, 0),
    )
    draw.text((title_pad, title_pad), title, fill=(255, 255, 255), font=title_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _make_contact_sheet(image_paths: list[Path], output_path: Path, *, thumb_width: int = 640, columns: int = 3) -> None:
    if not image_paths:
        return
    thumbs: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            scale = thumb_width / float(rgb.width)
            thumb = rgb.resize((thumb_width, max(1, round(rgb.height * scale))), Image.BILINEAR)
            thumbs.append(thumb)
    rows = (len(thumbs) + columns - 1) // columns
    cell_h = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_h), (20, 20, 20))
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_width
        y = (idx // columns) * cell_h
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)


def _prediction_row(pred: dict[str, Any], *, save_topk: int) -> dict[str, Any]:
    ranked = pred["ranked_indices"]
    candidates = pred["candidates"]
    return {
        "image_id": pred["image_id"],
        "file_name": pred["file_name"],
        "target_ar": pred["target_ar"],
        "candidate_source": pred["candidate_source"],
        "decision_id": pred["decision_id"],
        "route_id": pred["route_id"],
        "top_iou_to_best_label": pred["top_iou_to_best_label"],
        "top_candidate": pred["top_candidate"],
        "best_label_candidate": pred["best_label_candidate"],
        "top_candidates": [candidates[idx] for idx in ranked[: min(save_topk, len(ranked))]],
    }


def _safe_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_checkpoint(args.checkpoint, device=device)
    input_size = infer_input_size(ckpt, args.input_size)
    label_json = args.label_json
    records = load_label_crop_records(
        label_json,
        max_images=max(args.sample_size, 1) if not args.image_id else None,
        image_ids=args.image_id or None,
        target_ar_filter=args.target_ar_filter,
    )
    if not args.image_id:
        records = records[: max(0, int(args.sample_size))]

    rows: list[dict[str, Any]] = []
    saved_images: list[Path] = []
    for record in records:
        pred = score_crop_record(
            model=model,
            record=record,
            image_root=args.image_root,
            input_size=input_size,
            device=device,
            candidate_source=args.candidate_source,
            static_k=args.static_k,
            static_area_prior=args.static_area_prior,
            static_area_penalty=args.static_area_penalty,
            static_full_penalty=args.static_full_penalty,
        )
        safe_image_id = _safe_filename_part(pred["image_id"])
        safe_target_ar = _safe_filename_part(pred["target_ar"])
        out_path = args.output_dir / "overlays" / f"{safe_image_id}_{safe_target_ar}_{args.candidate_source}.png"
        _draw_overlay(pred, output_path=out_path, save_topk=args.save_topk, font_scale=args.font_scale)
        saved_images.append(out_path)
        row = _prediction_row(pred, save_topk=args.save_topk)
        row["overlay_path"] = str(out_path)
        rows.append(row)

    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    contact_sheet_path = args.output_dir / "contact_sheet.png"
    _make_contact_sheet(saved_images, contact_sheet_path)
    summary = {
        "checkpoint": str(args.checkpoint),
        "label_json": str(label_json),
        "image_root": str(args.image_root),
        "candidate_source": args.candidate_source,
        "static_area_prior": args.static_area_prior,
        "static_area_penalty": args.static_area_penalty,
        "static_full_penalty": args.static_full_penalty,
        "input_size": input_size,
        "sample_count": len(rows),
        "contact_sheet": str(contact_sheet_path),
        "overlay_dir": str(args.output_dir / "overlays"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
