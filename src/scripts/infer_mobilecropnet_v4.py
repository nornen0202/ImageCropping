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

from mobilecropnet_v4.data import (
    TARGET_AR_VOCAB,
    letterbox_to_original_box,
    load_image_tensor_letterbox,
    original_to_letterbox_box,
    target_ar_id,
)
from mobilecropnet_v4.eval_utils import load_mobilecropnet_v4_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MobileCropNet v4 inference on one image.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--target_ar", default="FREE", choices=TARGET_AR_VOCAB)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_png", type=Path, default=None)
    parser.add_argument("--candidate_boxes_json", type=Path, default=None, help="Optional JSON list of normalized xyxy boxes.")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _load_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_png(image_path: Path, rows: list[dict[str, Any]], output_path: Path) -> None:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    scale = min(1.0, 1400.0 / float(max(canvas.size)))
    if scale < 1.0:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(18, canvas.width // 55))
    width = max(3, canvas.width // 350)
    colors = [(196, 45, 45), (224, 138, 34), (45, 102, 210), (36, 150, 80)]
    for idx, row in enumerate(rows[:4]):
        box = row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
        x1, y1, x2, y2 = [float(v) for v in box]
        xy = [int(x1 * canvas.width), int(y1 * canvas.height), int(x2 * canvas.width), int(y2 * canvas.height)]
        color = colors[idx % len(colors)]
        draw.rectangle(xy, outline=color, width=width)
        label = f"#{idx + 1} {float(row.get('score', 0.0)):.2f}"
        tw = int(draw.textlength(label, font=font))
        th = int(getattr(font, "size", 18) * 1.35)
        draw.rectangle([xy[0], max(0, xy[1] - th - 4), xy[0] + tw + 10, xy[1]], fill=color)
        draw.text((xy[0] + 5, max(0, xy[1] - th - 2)), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.checkpoint, device=device)
    input_size = int(args.input_size or ckpt.get("train_config", {}).get("input_size", 256))
    image_tensor, transform = load_image_tensor_letterbox(args.image, input_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    ar = torch.tensor([target_ar_id(args.target_ar)], dtype=torch.long, device=device)
    image_ar = torch.tensor([transform.image_ar_log], dtype=torch.float32, device=device)

    boxes = None
    valid = None
    is_base = None
    if args.candidate_boxes_json is not None:
        raw_boxes = json.loads(args.candidate_boxes_json.read_text(encoding="utf-8"))
        padded = [original_to_letterbox_box(box, transform) for box in raw_boxes]
        boxes = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(0)
        valid = torch.ones((1, len(padded)), dtype=torch.float32, device=device)
        is_base = torch.zeros_like(valid)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, boxes, valid, ar, image_ar, is_base)
    scores = torch.sigmoid(outputs["utility_logits"][0]).detach().cpu().tolist()
    out_boxes = boxes.detach().cpu()[0].tolist() if boxes is not None else outputs["proposal_boxes"].detach().cpu()[0].tolist()
    ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    predictions = [
        {
            "rank": rank + 1,
            "score": float(scores[idx]),
            "bbox_letterbox_xyxy": [float(v) for v in out_boxes[idx]],
            "bbox_norm_xyxy": letterbox_to_original_box(out_boxes[idx], transform),
        }
        for rank, idx in enumerate(ranked[: args.topk])
    ]
    payload = {
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "target_ar": args.target_ar,
        "input_size": input_size,
        "predictions": predictions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_png is not None:
        _draw_png(args.image, predictions, args.output_png)
    print(json.dumps(payload["predictions"][: min(3, len(predictions))], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
