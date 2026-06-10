#!/usr/bin/env python3
"""Build SSTK T1/UCTR subject-mode target-AR best-crop visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "Implement_Docs" / "assets_mobilecropnet_training_data_master_20260424"
)
T1_VAL_JSONL = (
    PROJECT_ROOT
    / "data/SSTK/Full_10000/artifacts/"
    / "training_labels_t1_product_ar_260422_hashsplit_80_10_10_260422"
    / "val/train_conditional_detr_batch.jsonl"
)
UCTR_VAL_JSONL = (
    PROJECT_ROOT
    / "data/SSTK/Full_10000/artifacts/"
    / "training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422"
    / "val/train_conditional_detr_batch.jsonl"
)

AR_ORDER = ("FREE", "1:1", "4:3", "3:4", "16:9", "9:16")
INK = (27, 36, 48)
MUTED = (91, 100, 114)
GRID = (215, 221, 231)
PAPER = (248, 246, 240)
WHITE = (255, 255, 255)
BLUE = (49, 93, 140)
TEAL = (46, 140, 125)
ORANGE = (194, 106, 46)
RED = (181, 74, 74)


@dataclass(frozen=True)
class Representative:
    mode: str
    image_id: str
    label: str
    audit_note: str


# These examples were manually selected from the UCTR Product-AR validation split
# after rejecting visually inconsistent high-score route examples.
REPRESENTATIVES = (
    Representative(
        mode="object_single",
        image_id="sstk_image_689380903",
        label="single object",
        audit_note="single bird, no person flag",
    ),
    Representative(
        mode="object_multi",
        image_id="pond5_image_147935866",
        label="multi object",
        audit_note="dessert plate with multiple items",
    ),
    Representative(
        mode="portrait_single",
        image_id="sstk_image_777178726",
        label="single portrait",
        audit_note="one person exercising",
    ),
    Representative(
        mode="portrait_group",
        image_id="sstk_image_1896379945",
        label="group portrait",
        audit_note="two detected people, group route",
    ),
    Representative(
        mode="scene_general",
        image_id="sstk_image_1077646094",
        label="scene",
        audit_note="landscape scene, no detected subject",
    ),
    Representative(
        mode="background_texture_copyspace",
        image_id="sstk_image_2325090935",
        label="texture/copyspace",
        audit_note="repeating wood texture, background-like route",
    ),
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = _font(24, bold=True)
FONT_HEAD = _font(16, bold=True)
FONT_BODY = _font(13)
FONT_SMALL = _font(11)
FONT_TINY = _font(10)


def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def resolve_image_path(raw_path: str | None) -> Path:
    if not raw_path:
        raise ValueError("row has no image_path")
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    local = PROJECT_ROOT / path
    if local.exists():
        return local
    if raw_path.startswith("/group-volume/users/jaden.ju/Sources/ImageCropping/"):
        rel = raw_path.split("/group-volume/users/jaden.ju/Sources/ImageCropping/", 1)[1]
        local = PROJECT_ROOT / rel
        if local.exists():
            return local
    raise FileNotFoundError(raw_path)


def load_label_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_id = str(row.get("image_id"))
            target_ar = str(row.get("target_ar"))
            rows.setdefault(image_id, {})[target_ar] = row
    return rows


def score_value(target: dict[str, Any]) -> float:
    nested = target.get("score_targets") or {}
    value = (
        target.get("crop_utility_prob")
        if target.get("crop_utility_prob") is not None
        else nested.get("crop_utility_prob")
    )
    if value is None:
        value = target.get("score_prob") if target.get("score_prob") is not None else nested.get("score_prob")
    if value is None:
        return 0.0
    return float(value)


def best_target(row: dict[str, Any]) -> dict[str, Any]:
    targets = row.get("matching_targets") or []
    if not targets:
        raise ValueError(f"row has no matching_targets: {row.get('image_id')} {row.get('target_ar')}")
    return max(targets, key=score_value)


def target_scores(target: dict[str, Any]) -> dict[str, float | None]:
    nested = target.get("score_targets") or {}
    ext = target.get("external_score_teacher") or {}
    label_score = score_value(target)
    raw_external = nested.get("raw_external_ranker_score")
    if raw_external is None:
        raw_external = ext.get("ranker_score")
    sstk_score = nested.get("sstk_crop_utility_prob")
    if sstk_score is None:
        sstk_score = nested.get("sstk_score_prob")
    rank_pct = nested.get("crop_utility_rank_pct")
    if rank_pct is None:
        rank_pct = target.get("crop_utility_rank_pct")
    return {
        "label_score": label_score,
        "raw_external_ranker_score": None if raw_external is None else float(raw_external),
        "sstk_crop_utility_prob": None if sstk_score is None else float(sstk_score),
        "crop_utility_rank_pct": None if rank_pct is None else float(rank_pct),
    }


def norm_bbox(target: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = target.get("bbox_norm_xyxy")
    if not bbox or len(bbox) != 4:
        raise ValueError("target has no bbox_norm_xyxy")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 <= x1:
        x2 = clamp(x1 + 0.01, 0.0, 1.0)
    if y2 <= y1:
        y2 = clamp(y1 + 0.01, 0.0, 1.0)
    return x1, y1, x2, y2


def crop_by_norm_bbox(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = bbox
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))
    px1 = max(0, min(px1, w - 1))
    py1 = max(0, min(py1, h - 1))
    px2 = max(px1 + 1, min(px2, w))
    py2 = max(py1 + 1, min(py2, h))
    return image.crop((px1, py1, px2, py2))


def contain(image: Image.Image, size: tuple[int, int], fill: tuple[int, int, int] = WHITE) -> tuple[Image.Image, tuple[int, int, int, int]]:
    canvas = Image.new("RGB", size, fill)
    thumb = image.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    x = (size[0] - thumb.width) // 2
    y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas, (x, y, thumb.width, thumb.height)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    x, y = xy
    words = text.split()
    line = ""
    for word in words:
        test = word if not line else f"{line} {word}"
        if draw.textlength(test, font=font) <= width:
            line = test
            continue
        if line:
            draw.text((x, y), line, font=font, fill=fill)
            y += int(font.size * 1.25)
        line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)


def draw_bbox_on_thumbnail(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    thumb_box: tuple[int, int, int, int],
    bbox: list[float] | tuple[float, float, float, float] | None,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if not bbox or len(bbox) != 4:
        return
    tx, ty, tw, th = thumb_box
    ox, oy = origin
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = ox + tx + clamp(x1, 0.0, 1.0) * tw
    y1 = oy + ty + clamp(y1, 0.0, 1.0) * th
    x2 = ox + tx + clamp(x2, 0.0, 1.0) * tw
    y2 = oy + ty + clamp(y2, 0.0, 1.0) * th
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def draw_original_cell(
    canvas: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    image: Image.Image,
    row: dict[str, Any],
    rep: Representative,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    draw.rounded_rectangle((x, y, x + size[0], y + size[1]), radius=6, fill=WHITE, outline=GRID, width=1)
    view, thumb_box = contain(image, (size[0] - 24, 132), fill=(245, 245, 242))
    view_x = x + 12
    view_y = y + 10
    canvas.paste(view, (view_x, view_y))

    routing = row.get("routing") or {}
    ss = routing.get("subject_set") or {}
    draw.text((x + 12, y + 150), rep.label, font=FONT_HEAD, fill=INK)
    draw.text((x + 12, y + 172), rep.image_id, font=FONT_SMALL, fill=MUTED)
    draw.text((x + 12, y + 188), rep.mode, font=FONT_TINY, fill=MUTED)
    num_person = ss.get("num_person")
    num_eff = ss.get("num_effective_subjects")
    if num_person is not None or num_eff is not None:
        detail = f"persons {num_person} | eff {num_eff}"
    else:
        detail = "subject-set fields not present in this branch"
    draw.text((x + 12, y + 202), detail, font=FONT_TINY, fill=MUTED)
    draw_wrapped(draw, (x + 12, y + 216), rep.audit_note, size[0] - 24, FONT_TINY, MUTED)


def draw_crop_cell(
    canvas: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    image: Image.Image,
    row: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    target = best_target(row)
    bbox = norm_bbox(target)
    scores = target_scores(target)
    crop = crop_by_norm_bbox(image, bbox)

    border = TEAL if method == "T1" else ORANGE
    draw.rounded_rectangle((x, y, x + size[0], y + size[1]), radius=6, fill=WHITE, outline=GRID, width=1)
    view, _ = contain(crop, (size[0] - 20, 142), fill=(248, 248, 246))
    canvas.paste(view, (x + 10, y + 10))
    draw.rounded_rectangle((x + 10, y + 10, x + size[0] - 10, y + 152), radius=4, outline=border, width=3)

    target_ar = str(row.get("target_ar"))
    draw.text((x + 12, y + 162), f"AR {target_ar}", font=FONT_HEAD, fill=INK)
    if method == "T1":
        draw.text((x + 12, y + 185), f"score {scores['label_score']:.3f}", font=FONT_BODY, fill=INK)
    else:
        raw_external = scores["raw_external_ranker_score"]
        raw_text = "na" if raw_external is None else f"{raw_external:.3f}"
        draw.text((x + 12, y + 185), f"label {scores['label_score']:.2f}", font=FONT_BODY, fill=INK)
        draw.text((x + 12, y + 204), f"ranker {raw_text}", font=FONT_SMALL, fill=MUTED)
    x1, y1, x2, y2 = bbox
    return {
        "target_ar": target_ar,
        "bbox_norm_xyxy": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
        **scores,
    }


def validate_representatives(label_rows: dict[str, dict[str, dict[str, Any]]], label_name: str) -> None:
    missing: list[str] = []
    for rep in REPRESENTATIVES:
        armap = label_rows.get(rep.image_id)
        if not armap:
            missing.append(f"{label_name}:{rep.image_id}:all")
            continue
        for ar in AR_ORDER:
            if ar not in armap:
                missing.append(f"{label_name}:{rep.image_id}:{ar}")
        modes = {str((r.get("routing") or {}).get("subject_mode")) for r in armap.values()}
        if rep.mode not in modes:
            missing.append(f"{label_name}:{rep.image_id}:mode={modes}, expected={rep.mode}")
    if missing:
        raise RuntimeError("Missing or mismatched representative rows:\n" + "\n".join(missing))


def build_figure(
    label_rows: dict[str, dict[str, dict[str, Any]]],
    method: str,
    source_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_representatives(label_rows, method)
    col0 = 268
    col = 208
    row_h = 236
    header_h = 132
    footer_h = 44
    width = col0 + len(AR_ORDER) * col + 36
    height = header_h + len(REPRESENTATIVES) * row_h + footer_h
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    title = f"SSTK {method} subject-mode target-AR best crops"
    draw.text((24, 18), title, font=FONT_TITLE, fill=INK)
    draw.text(
        (24, 52),
        "Rows are manually audited representative SSTK images; each AR cell shows the selected positive crop and score.",
        font=FONT_BODY,
        fill=MUTED,
    )
    score_note = (
        "T1 score = native crop_utility_prob."
        if method == "T1"
        else "UCTR label = safe-converted crop_utility_prob/rank-pct; ranker = raw external ranker score."
    )
    draw.text((24, 74), score_note, font=FONT_SMALL, fill=MUTED)
    source_text = f"Source: {source_path.relative_to(PROJECT_ROOT)}"
    draw_wrapped(draw, (24, 92), source_text, width - 48, FONT_TINY, MUTED)

    x0 = 18
    y0 = header_h
    draw.text((x0 + 12, y0 - 22), "audited source", font=FONT_HEAD, fill=INK)
    for i, ar in enumerate(AR_ORDER):
        draw.text((x0 + col0 + i * col + 14, y0 - 22), f"best crop {ar}", font=FONT_HEAD, fill=INK)

    summary_rows: list[dict[str, Any]] = []
    for row_idx, rep in enumerate(REPRESENTATIVES):
        row_y = y0 + row_idx * row_h
        armap = label_rows[rep.image_id]
        free_row = armap["FREE"]
        image_path = resolve_image_path(str(free_row.get("image_path")))
        image = open_image(image_path)
        draw_original_cell(canvas, (x0, row_y), (col0 - 12, row_h - 12), image, free_row, rep)
        mode_summary = {
            "mode": rep.mode,
            "image_id": rep.image_id,
            "label": rep.label,
            "audit_note": rep.audit_note,
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "routing_subject_mode": (free_row.get("routing") or {}).get("subject_mode"),
            "subject_set": (free_row.get("routing") or {}).get("subject_set"),
            "target_ars": [],
        }
        for ar_idx, ar in enumerate(AR_ORDER):
            cell_x = x0 + col0 + ar_idx * col
            row_summary = draw_crop_cell(canvas, (cell_x, row_y), (col - 12, row_h - 12), image, armap[ar], method)
            mode_summary["target_ars"].append(row_summary)
        summary_rows.append(mode_summary)

    draw.line((24, height - 36, width - 24, height - 36), fill=GRID, width=1)
    draw.text(
        (24, height - 26),
        "Source thumbnails are plain audited images. AR cells are rendered as actual selected positive crop thumbnails.",
        font=FONT_TINY,
        fill=MUTED,
    )

    filename = f"fig_sstk_{method.lower()}_subject_mode_ar_best_crops.png"
    out_path = output_dir / filename
    canvas.save(out_path)
    return {
        "method": method,
        "source_path": str(source_path.relative_to(PROJECT_ROOT)),
        "figure_path": str(out_path.relative_to(PROJECT_ROOT)),
        "representatives": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t1-val-jsonl", type=Path, default=T1_VAL_JSONL)
    parser.add_argument("--uctr-val-jsonl", type=Path, default=UCTR_VAL_JSONL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t1_rows = load_label_rows(args.t1_val_jsonl)
    uctr_rows = load_label_rows(args.uctr_val_jsonl)
    summary = {
        "selection_policy": "manual visual audit of full-AR UCTR validation candidates, then reuse same image IDs for T1 and UCTR comparison",
        "ar_order": list(AR_ORDER),
        "figures": [
            build_figure(t1_rows, "T1", args.t1_val_jsonl, args.output_dir),
            build_figure(uctr_rows, "UCTR", args.uctr_val_jsonl, args.output_dir),
        ],
    }
    summary_path = args.output_dir / "fig_sstk_teacher_subject_mode_ar_best_crops_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"summary_path": str(summary_path), "figures": [f["figure_path"] for f in summary["figures"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
