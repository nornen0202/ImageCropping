#!/usr/bin/env python3
"""Build section-level evidence figures for the MobileCropNet data report."""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "Implement_Docs" / "assets_mobilecropnet_training_data_master_20260424"
)
PUBLIC_DATA_ROOT = PROJECT_ROOT / "data/Publics"
PUBLIC_CROPPER_GAIC_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts/unified_public_benchmark_20260420/public_cropper_full_grouped/public_cropper_gaic_predictions.jsonl"
)
PUBLIC_CROPPER_CGS_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts/unified_public_benchmark_20260420/public_cropper_full_grouped/public_cropper_cgs_predictions.jsonl"
)
EQUAL4_TEACHER_LEADERBOARD_JSON = (
    PROJECT_ROOT / "artifacts/unified_public_benchmark_20260423_equal4_teacher/equal4_teacher_leaderboard.json"
)
BATCH_JSONL = (
    PROJECT_ROOT
    / "data/SSTK/Full_10000/artifacts/training_labels/"
    / "260413_v1_multimode_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl"
)
CANDIDATE_JSONL = (
    PROJECT_ROOT
    / "data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260413_v1_multimode.jsonl"
)
MANIFEST_JSON = DEFAULT_OUTPUT_DIR / "figure_manifest.json"

INK = "#1B2430"
MUTED = "#5B6472"
GRID = "#D9DEE7"
PAPER = "#F7F3EA"
WHITE = "#FFFFFF"
BLUE = "#315D8C"
TEAL = "#2E8C7D"
ORANGE = "#C26A2E"
RED = "#B54A4A"
GOLD = "#C49A2C"
PURPLE = "#7357A6"
GREEN = "#4F8A53"
EXCLUDED_RESULT_MODES = {"background_texture_copyspace"}
LABEL_EXAMPLE_SCAN_LIMIT = 20000
PREFERRED_LABEL_EXAMPLE_IMAGE_IDS = {
    "subject_coverage_good": ["sstk_image_1157970946"],
    "subject_coverage_bad": ["sstk_image_1476557597"],
    "headroom_good": ["sstk_image_671450404"],
    "headroom_bad": ["sstk_image_2236985255"],
    "lookroom_good": ["sstk_image_2078623873"],
    "lookroom_bad": ["sstk_image_2078623873"],
}
EXCLUDED_PORTRAIT_CHECKLIST_IMAGE_IDS = {
    "sstk_image_94489720",
    "sstk_image_1706324077",
}
PREFERRED_JOINT_OVERLAY_IMAGE_IDS = [
    "sstk_image_476205934",
    "sstk_image_715944223",
    "sstk_image_87576424",
    "bigstock_image_71945659",
]
PREFERRED_SUPPORT_MAP_IMAGE_IDS = [
    "bigstock_image_71945659",
    "sstk_image_476205934",
    "sstk_image_715944223",
    "sstk_image_87576424",
    "sstk_image_1543389905",
    "pond5_image_81735906",
]
PREFERRED_C7_SUPPORT_EXAMPLE_IMAGE_IDS = [
    "bigstock_image_71945659",
    "sstk_image_476205934",
    "sstk_image_671450404",
    "pond5_image_95476634",
    "sstk_image_715944223",
    "sstk_image_87576424",
    "sstk_image_1543389905",
    "pond5_image_81735906",
]
PREFERRED_SUPPORT_MASS_ONLY_IMAGE_IDS = [
    "bigstock_image_71945659",
    "sstk_image_1336721579",
    "sstk_image_671450404",
    "pond5_image_95476634",
    "sstk_image_715944223",
    "sstk_image_87576424",
]
PREFERRED_ROLE_ROW_IMAGE_IDS = [
    "sstk_image_1242934243",
    "sstk_image_1336721579",
    "sstk_image_715944223",
    "sstk_image_1543389905",
]
PREFERRED_PORTRAIT_AUDIT_IMAGE_IDS = [
    "sstk_image_1242934243",
    "sstk_image_1276179112",
    "sstk_image_671450404",
    "pond5_image_95476634",
]
PREFERRED_GUIDANCE_IMAGE_IDS = [
    "bigstock_image_71945659",
    "sstk_image_671450404",
    "pond5_image_95476634",
    "sstk_image_1336721579",
    "sstk_image_715944223",
    "sstk_image_476205934",
]
CURATED_ROUTE_MODE_IMAGE_IDS = {
    "object_single": ["sstk_image_1157970946"],
    "portrait_single": ["sstk_image_671450404"],
    "portrait_group": ["pond5_image_95476634"],
    "scene_general": ["sstk_image_715944223"],
    "object_multi": ["sstk_image_1336721579"],
}
CURATED_ROUTE_IMAGE_IDS = {
    image_id for image_ids in CURATED_ROUTE_MODE_IMAGE_IDS.values() for image_id in image_ids
}


def read_json(path: Path, default: Any | None = None) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    for path in paths:
        p = PROJECT_ROOT / path if not isinstance(path, Path) else path
        if p.exists():
            return p
    return None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    fill: str = INK,
    size: int = 28,
    bold: bool = False,
    line_spacing: int = 6,
) -> int:
    fnt = font(size, bold=bold)
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        bbox = draw.textbbox((0, 0), trial, font=fnt)
        if bbox[2] - bbox[0] <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        bbox = draw.textbbox((x, y), line, font=fnt)
        y += bbox[3] - bbox[1] + line_spacing
    return y


def draw_tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    text_fill: str = WHITE,
    size: int = 22,
) -> tuple[int, int]:
    fnt = font(size, bold=True)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    w = bbox[2] - bbox[0] + 24
    h = bbox[3] - bbox[1] + 14
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=fill)
    draw.text((x + 12, y + 6), text, font=fnt, fill=text_fill)
    return x + w, y + h


def open_image(path: Path, size: tuple[int, int], fill: str = "#EFEAE1") -> Image.Image:
    img, _ = open_image_with_bbox(path, size, fill=fill)
    return img


def open_image_with_bbox(
    path: Path,
    size: tuple[int, int],
    fill: str = "#EFEAE1",
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return a letterboxed image and the actual image rectangle inside the canvas."""
    if not path.exists():
        img = Image.new("RGB", size, fill)
        d = ImageDraw.Draw(img)
        draw_wrapped(d, (28, 28), f"Missing artifact\n{rel(path)}", size[0] - 56, fill=RED, size=24, bold=True)
        return img, (0, 0, size[0], size[1])
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas, (x, y, x + img.width, y + img.height)


def first_png(rel_dir: str) -> Path | None:
    root = PROJECT_ROOT / rel_dir
    if not root.exists():
        return None
    files = sorted(p for p in root.glob("*.png") if p.name != "contact_sheet.png")
    return files[0] if files else None


def extract_visual_region(
    path: Path,
    size: tuple[int, int],
    fill: str = "#EFEAE1",
    max_width_ratio: float | None = None,
) -> Image.Image:
    """Load one overlay artifact and keep the image/box region, not a contact sheet."""
    if not path.exists():
        return open_image(path, size, fill=fill)
    img = Image.open(path).convert("RGB")
    if img.width > 900:
        ratio = max_width_ratio if max_width_ratio is not None else 0.56
        visual_width = min(int(img.width * ratio), 1050)
        img = img.crop((0, 0, visual_width, img.height))
    gray = img.convert("L")
    pix = gray.load()
    w, h = gray.size
    row_threshold = max(8, int(w * 0.07))
    col_threshold = max(8, int(h * 0.05))
    rows = []
    for yy in range(h):
        count = 0
        for xx in range(w):
            if pix[xx, yy] > 30:
                count += 1
        if count >= row_threshold:
            rows.append(yy)
    cols = []
    for xx in range(w):
        count = 0
        for yy in range(h):
            if pix[xx, yy] > 30:
                count += 1
        if count >= col_threshold:
            cols.append(xx)
    bbox = (min(cols), min(rows), max(cols) + 1, max(rows) + 1) if rows and cols else gray.point(lambda p: 255 if p > 30 else 0).getbbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        pad = 12
        img = img.crop(
            (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(img.width, x2 + pad),
                min(img.height, y2 + pad),
            )
        )
    img = ImageOps.contain(img, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_individual_visual_panels(
    path: Path,
    title: str,
    subtitle: str,
    sources: list[tuple[Any, ...]],
) -> None:
    img = Image.new("RGB", (1600, 1000), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(d, title, subtitle)
    n = max(1, len(sources))
    cols = min(3, n)
    rows = math.ceil(n / cols)
    panel_w = int((1460 - 34 * (cols - 1)) / cols)
    panel_h = 665 if rows == 1 else 360
    image_h = 510 if rows == 1 else 245
    start_x, start_y = 70, 175
    for idx, source in enumerate(sources):
        label, source_path, note, color = source[:4]
        crop_ratio = source[4] if len(source) >= 5 else None
        col = idx % cols
        row = idx // cols
        x = start_x + col * (panel_w + 34)
        y = start_y + row * (panel_h + 36)
        d.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=18, fill=WHITE, outline=GRID, width=2)
        draw_tag(d, (x + 20, y + 18), label, color, size=20)
        visual = (
            extract_visual_region(source_path, (panel_w - 40, image_h), max_width_ratio=crop_ratio)
            if source_path is not None
            else Image.new("RGB", (panel_w - 40, image_h), "#EFEAE1")
        )
        img.paste(visual, (x + 20, y + 68))
        draw_wrapped(d, (x + 22, y + 82 + image_h), note, panel_w - 44, fill=MUTED, size=19)
    save_pil(img, path)


def resolve_image_path(row: dict[str, Any]) -> Path | None:
    raw = row.get("image_path") or row.get("path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def absolute_box_to_norm(row: dict[str, Any], box: Any) -> list[float] | None:
    values = parse_norm_box(box)
    if values is None:
        return None
    if max(abs(v) for v in values) <= 1.5:
        return values
    image_path = resolve_image_path(row)
    if not image_path:
        return None
    with Image.open(image_path) as im:
        width, height = im.size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = values
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def norm_box_to_pixels(box: Any, area: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        return None
    left, top, right, bottom = area
    w = right - left
    h = bottom - top
    return (
        int(left + max(0.0, min(1.0, x1)) * w),
        int(top + max(0.0, min(1.0, y1)) * h),
        int(left + max(0.0, min(1.0, x2)) * w),
        int(top + max(0.0, min(1.0, y2)) * h),
    )


def norm_point_to_pixels(point: Any, area: tuple[int, int, int, int]) -> tuple[int, int] | None:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x, y = [float(v) for v in point]
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = area
    return (
        int(left + max(0.0, min(1.0, x)) * (right - left)),
        int(top + max(0.0, min(1.0, y)) * (bottom - top)),
    )


def hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        return (79, 138, 83, alpha)
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def fill_norm_box(
    canvas: Image.Image,
    box: Any,
    area: tuple[int, int, int, int],
    color: str,
    alpha: int = 62,
) -> None:
    pix = norm_box_to_pixels(box, area)
    if pix is None:
        return
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle(pix, fill=hex_to_rgba(color, alpha))


def decode_f16_grid(encoded: Any, grid_size: int) -> list[float] | None:
    if not encoded or not isinstance(encoded, str) or grid_size <= 0:
        return None
    try:
        raw = base64.b64decode(encoded)
    except Exception:
        return None
    expected = grid_size * grid_size
    if len(raw) < expected * 2:
        return None
    values: list[float] = []
    for idx in range(expected):
        try:
            value = struct.unpack_from("<e", raw, idx * 2)[0]
        except struct.error:
            return None
        if math.isfinite(value):
            values.append(max(0.0, float(value)))
        else:
            values.append(0.0)
    return values


def support_spec(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("routing", {}).get("subject_support_overlay", {}).get("support_spec") or {}


def is_support_map_row(row: dict[str, Any]) -> bool:
    if str(row.get("image_id")) in EXCLUDED_PORTRAIT_CHECKLIST_IMAGE_IDS:
        return False
    if str(row.get("routing", {}).get("subject_mode") or "unknown") in EXCLUDED_RESULT_MODES:
        return False
    support = row.get("routing", {}).get("subject_support_overlay", {}) or {}
    spec = support.get("support_spec") or {}
    return (
        support.get("mode") == "support_map"
        and bool(support.get("support_map_enabled"))
        and bool(spec.get("support_mass_grid_f16_b64") or spec.get("mass_grid_f16_b64"))
        and resolve_image_path(row) is not None
    )


def has_decoded_support_mass(row: dict[str, Any]) -> bool:
    if str(row.get("image_id")) in EXCLUDED_PORTRAIT_CHECKLIST_IMAGE_IDS:
        return False
    if str(row.get("routing", {}).get("subject_mode") or "unknown") in EXCLUDED_RESULT_MODES:
        return False
    spec = support_spec(row)
    return bool(spec.get("support_mass_grid_f16_b64") or spec.get("mass_grid_f16_b64")) and resolve_image_path(row) is not None


def support_repr_label(row: dict[str, Any]) -> str:
    support = row.get("routing", {}).get("subject_support_overlay", {}) or {}
    repr_type = str(support.get("subject_repr_type") or "subject_region")
    if is_support_map_row(row) or has_decoded_support_mass(row):
        return repr_type.replace("_", " ")
    return "bbox fallback"


def support_bbox_for_render(row: dict[str, Any]) -> list[float] | None:
    support = row.get("routing", {}).get("subject_support_overlay", {}) or {}
    spec = support.get("support_spec") or {}
    if is_support_map_row(row):
        return (
            parse_norm_box(spec.get("latent_support_bbox_norm_xyxy"))
            or parse_norm_box(spec.get("envelope_norm_xyxy"))
            or parse_norm_box(support.get("bbox_norm_xyxy"))
        )
    return parse_norm_box(support.get("bbox_norm_xyxy"))


def support_core_box_for_render(row: dict[str, Any]) -> list[float] | None:
    spec = support_spec(row)
    return parse_norm_box(spec.get("latent_support_core_bbox_norm_xyxy") or spec.get("core_bbox_norm_xyxy"))


def guidance_box_for_render(row: dict[str, Any]) -> list[float] | None:
    mode = str(row.get("routing", {}).get("subject_mode") or "")
    if is_support_map_row(row):
        return support_bbox_for_render(row)
    if mode in {"object_multi", "portrait_single", "portrait_group"}:
        return subject_union_norm(row) or parse_norm_box(row.get("routing", {}).get("subject_prior_bbox_norm_xyxy")) or support_bbox_for_render(row)
    return subject_union_norm(row) or parse_norm_box(row.get("routing", {}).get("subject_prior_bbox_norm_xyxy")) or support_bbox_for_render(row)


def guidance_label_for_row(row: dict[str, Any]) -> str:
    mode = str(row.get("routing", {}).get("subject_mode") or "")
    if is_support_map_row(row):
        return "guidance envelope"
    if mode == "object_multi":
        return "subject union"
    return "subject prior"


def draw_support_heatmap(
    canvas: Image.Image,
    row: dict[str, Any],
    area: tuple[int, int, int, int],
    alpha_max: int = 128,
) -> bool:
    if not has_decoded_support_mass(row):
        return False
    spec = support_spec(row)
    grid_size = int(spec.get("support_grid_size") or spec.get("grid_size") or 24)
    values = decode_f16_grid(
        spec.get("support_mass_grid_f16_b64") or spec.get("mass_grid_f16_b64"),
        grid_size,
    )
    if not values:
        return False
    peak = max(values)
    if peak <= 0:
        return False
    alpha = Image.new("L", (grid_size, grid_size), 0)
    alpha.putdata([min(alpha_max, int((value / peak) ** 0.65 * alpha_max)) for value in values])
    w = max(1, area[2] - area[0])
    h = max(1, area[3] - area[1])
    alpha = alpha.resize((w, h), Image.Resampling.BICUBIC)
    overlay = Image.new("RGBA", (w, h), hex_to_rgba(GREEN, 0))
    overlay.putalpha(alpha)
    canvas.paste(overlay, (area[0], area[1]), overlay)
    return True


def draw_support_centroid(draw: ImageDraw.ImageDraw, row: dict[str, Any], area: tuple[int, int, int, int]) -> None:
    point = support_spec(row).get("centroid_xy_norm")
    pix = norm_point_to_pixels(point, area)
    if pix is None:
        return
    x, y = pix
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=GOLD, outline=INK, width=2)


def draw_norm_box(
    draw: ImageDraw.ImageDraw,
    box: Any,
    area: tuple[int, int, int, int],
    color: str,
    label: str,
    width: int = 5,
    label_offset: tuple[int, int] = (4, 4),
) -> None:
    pix = norm_box_to_pixels(box, area)
    if pix is None:
        return
    draw.rectangle(pix, outline=color, width=width)
    if not label:
        return
    x1, y1, _, _ = pix
    dx, dy = label_offset
    draw_tag(draw, (x1 + dx, max(area[1] + 4, y1 + dy)), label, color, size=18)


def infer_profile_gaze(row: dict[str, Any]) -> tuple[str, list[float], list[float]] | None:
    flags = row.get("routing", {}).get("flags") or {}
    if not flags.get("is_profile_view"):
        return None
    support = support_visual_box(row)
    if support is None:
        return None
    x1, y1, x2, y2 = support
    cx = 0.5 * (x1 + x2)
    cy = max(0.18, min(0.55, y1 + 0.28 * max(0.1, y2 - y1)))
    if cx >= 0.54:
        direction = "left"
        start = [max(x1 + 0.08, cx - 0.02), cy]
        end = [max(0.02, start[0] - 0.22), cy]
    elif cx <= 0.46:
        direction = "right"
        start = [min(x2 - 0.08, cx + 0.02), cy]
        end = [min(0.98, start[0] + 0.22), cy]
    else:
        return None
    return direction, start, end


def draw_gaze_arrow(
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    area: tuple[int, int, int, int],
) -> None:
    gaze = infer_profile_gaze(row)
    if gaze is None:
        return
    direction, start_norm, end_norm = gaze
    start = norm_point_to_pixels(start_norm, area)
    end = norm_point_to_pixels(end_norm, area)
    if start is None or end is None:
        return
    draw.line((start[0], start[1], end[0], end[1]), fill=BLUE, width=5)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 18
    for delta in (math.radians(150), math.radians(-150)):
        hx = int(end[0] + head_len * math.cos(angle + delta))
        hy = int(end[1] + head_len * math.sin(angle + delta))
        draw.line((end[0], end[1], hx, hy), fill=BLUE, width=5)
    draw_tag(draw, (min(start[0], end[0]) + 4, max(area[1] + 4, min(start[1], end[1]) - 34)), f"gaze {direction}", BLUE, size=16)


def draw_norm_outline(
    draw: ImageDraw.ImageDraw,
    box: Any,
    area: tuple[int, int, int, int],
    color: str,
    width: int = 4,
) -> None:
    pix = norm_box_to_pixels(box, area)
    if pix is not None:
        draw.rectangle(pix, outline=color, width=width)


def offset_area(area: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    ox, oy = origin
    return area[0] + ox, area[1] + oy, area[2] + ox, area[3] + oy


def save_pil(img: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    img.save(path, optimize=True)


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    draw.text((54, 38), title, font=font(42, bold=True), fill=INK)
    if subtitle:
        draw_wrapped(draw, (56, 94), subtitle, 1460, fill=MUTED, size=25)


def box_card(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str = WHITE,
    outline: str = GRID,
    accent: str = BLUE,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rounded_rectangle(xyxy, radius=20, fill=fill, outline=outline, width=2)
    draw.rounded_rectangle((x1, y1, x1 + 12, y2), radius=8, fill=accent)
    draw.text((x1 + 28, y1 + 22), title, font=font(26, bold=True), fill=INK)
    draw_wrapped(draw, (x1 + 28, y1 + 64), body, x2 - x1 - 56, fill=MUTED, size=21)


def collect_public_dataset_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    try:
        from public_benchmark.adapters import iter_tasks
    except Exception as exc:  # pragma: no cover - defensive for partial checkouts.
        return {"error": f"public_benchmark.adapters unavailable: {exc}"}

    try:
        task = next(iter_tasks("fcdb", PUBLIC_DATA_ROOT, split="all"))
        examples["fcdb"] = {
            "dataset": "FCDB",
            "image_id": task.image_id,
            "image_path": task.image_path,
            "boxes": [
                {
                    "bbox": task.gt_boxes[0].bbox_xyxy_norm,
                    "label": "expert GT",
                    "color": GREEN,
                }
            ],
            "note": "single expert crop; no local MOS or pairwise density in this copy",
        }
    except Exception as exc:
        examples["fcdb"] = {"error": str(exc)}

    try:
        for task in iter_tasks("cpc", PUBLIC_DATA_ROOT, split="all"):
            candidates = [cand for cand in task.candidate_windows if cand.score is not None]
            if len(candidates) >= 3:
                candidates.sort(key=lambda cand: float(cand.score or 0.0), reverse=True)
                mid = candidates[len(candidates) // 2]
                low = candidates[-1]
                examples["cpc"] = {
                    "dataset": "CPC",
                    "image_id": task.image_id,
                    "image_path": task.image_path,
                    "boxes": [
                        {
                            "bbox": candidates[0].bbox_xyxy_norm,
                            "label": f"top mean={float(candidates[0].score or 0):.2f}",
                            "color": GREEN,
                        },
                        {
                            "bbox": mid.bbox_xyxy_norm,
                            "label": f"mid mean={float(mid.score or 0):.2f}",
                            "color": GOLD,
                        },
                        {
                            "bbox": low.bbox_xyxy_norm,
                            "label": f"low mean={float(low.score or 0):.2f}",
                            "color": RED,
                        },
                    ],
                    "note": f"{len(candidates)} candidate views; pairwise preferences are derived from annotator scores",
                }
                break
    except Exception as exc:
        examples["cpc"] = {"error": str(exc)}

    try:
        grouped: dict[str, list[Any]] = defaultdict(list)
        for task in iter_tasks("gnmc", PUBLIC_DATA_ROOT, split="all"):
            grouped[str(task.image_id)].append(task)
            if len(grouped[str(task.image_id)]) >= 3:
                tasks = grouped[str(task.image_id)]
                examples["gnmc"] = {
                    "dataset": "GNMC",
                    "image_id": tasks[0].image_id,
                    "image_path": tasks[0].image_path,
                    "boxes": [
                        {
                            "bbox": t.gt_boxes[0].bbox_xyxy_norm,
                            "label": f"AR {t.target_ar}",
                            "color": [GREEN, ORANGE, BLUE, PURPLE, RED][idx % 5],
                        }
                        for idx, t in enumerate(tasks[:5])
                    ],
                    "note": "same image has editor crops for multiple fixed aspect ratios",
                }
                break
    except Exception as exc:
        examples["gnmc"] = {"error": str(exc)}

    examples["gaic"] = collect_gaic_mos_example()
    return examples


def collect_gaic_mos_example() -> dict[str, Any]:
    for split in ("test", "val", "train"):
        ann_dir = PROJECT_ROOT / "data/Publics/GAIC_v2/annotations" / split
        img_dir = PROJECT_ROOT / "data/Publics/GAIC_v2/images" / split
        if not ann_dir.exists() or not img_dir.exists():
            continue
        for ann_path in sorted(ann_dir.glob("*.txt"))[:300]:
            image_path = img_dir / f"{ann_path.stem}.jpg"
            if not image_path.exists():
                continue
            rows: list[tuple[list[float], float]] = []
            with Image.open(image_path) as im:
                width, height = im.size
            for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    x1, y1, x2, y2, score = [float(v) for v in parts[:5]]
                except ValueError:
                    continue
                box = [
                    max(0.0, min(1.0, x1 / width)),
                    max(0.0, min(1.0, y1 / height)),
                    max(0.0, min(1.0, x2 / width)),
                    max(0.0, min(1.0, y2 / height)),
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                if 0.0 <= score <= 5.0:
                    rows.append((box, score))
            if len(rows) < 3:
                continue
            rows.sort(key=lambda item: item[1], reverse=True)
            if rows[0][1] < 4.0 or rows[0][1] - rows[-1][1] < 0.75:
                continue
            mid = rows[len(rows) // 2]
            return {
                "dataset": "GAIC",
                "image_id": ann_path.stem,
                "image_path": str(image_path.resolve()),
                "boxes": [
                    {"bbox": rows[0][0], "label": f"MOS {rows[0][1]:.2f}", "color": GREEN},
                    {"bbox": mid[0], "label": f"MOS {mid[1]:.2f}", "color": GOLD},
                    {"bbox": rows[-1][0], "label": f"MOS {rows[-1][1]:.2f}", "color": RED},
                ],
                "note": "dense candidate boxes with human MOS; high-MOS is not a product-safety label",
            }
    return {"error": "GAIC_v2 annotation example not found"}


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")).lower(),
        str(row.get("image_id", "")),
        "" if row.get("target_ar") is None else str(row.get("target_ar")),
    )


def task_key_from_example(dataset: str, example: dict[str, Any]) -> tuple[str, str, str] | None:
    if not example or example.get("error"):
        return None
    if dataset == "gnmc":
        box_label = str((example.get("boxes") or [{}])[0].get("label", ""))
        target_ar = box_label.replace("AR", "").strip() or "1:1"
    else:
        target_ar = ""
    return (dataset, str(example.get("image_id", "")), target_ar)


def load_prediction_rows(path: Path, wanted: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists() or not wanted:
        return rows
    for row in iter_jsonl(path) or []:
        key = prediction_key(row)
        if key in wanted and key not in rows:
            rows[key] = row
        if len(rows) >= len(wanted):
            break
    return rows


def collect_public_cropper_examples(public_dataset_examples: dict[str, Any]) -> list[dict[str, Any]]:
    wanted: set[tuple[str, str, str]] = set()
    image_lookup: dict[tuple[str, str, str], str] = {}
    for dataset in ("fcdb", "cpc", "gnmc"):
        example = public_dataset_examples.get(dataset) or {}
        key = task_key_from_example(dataset, example)
        if key:
            wanted.add(key)
            image_lookup[key] = str(example.get("image_path", ""))
    gaic_rows = load_prediction_rows(PUBLIC_CROPPER_GAIC_PREDICTIONS, wanted)
    cgs_rows = load_prediction_rows(PUBLIC_CROPPER_CGS_PREDICTIONS, wanted)
    out: list[dict[str, Any]] = []
    for key in sorted(wanted):
        if key not in gaic_rows or key not in cgs_rows:
            continue
        out.append(
            {
                "key": key,
                "dataset": key[0],
                "image_id": key[1],
                "target_ar": key[2] or None,
                "image_path": image_lookup.get(key),
                "public_cropper_gaic": gaic_rows[key],
                "public_cropper_cgs": cgs_rows[key],
            }
        )
    return out


def norm_box_area(box: Any) -> float:
    values = parse_norm_box(box)
    if values is None:
        return 0.0
    x1, y1, x2, y2 = values
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def norm_box_intersection_area(a: Any, b: Any) -> float:
    box_a = parse_norm_box(a)
    box_b = parse_norm_box(b)
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def union_norm_boxes(boxes: list[Any]) -> list[float] | None:
    parsed = [box for box in (parse_norm_box(box) for box in boxes) if box is not None]
    if not parsed:
        return None
    return [
        max(0.0, min(box[0] for box in parsed)),
        max(0.0, min(box[1] for box in parsed)),
        min(1.0, max(box[2] for box in parsed)),
        min(1.0, max(box[3] for box in parsed)),
    ]


def support_visual_box(row: dict[str, Any]) -> list[float] | None:
    support = row.get("routing", {}).get("subject_support_overlay", {})
    spec = support.get("support_spec") or {}
    return union_norm_boxes(
        [
            support.get("bbox_norm_xyxy"),
            support.get("latent_support_bbox_norm_xyxy"),
            support.get("latent_support_core_bbox_norm_xyxy"),
            spec.get("latent_support_bbox_norm_xyxy"),
            spec.get("envelope_norm_xyxy"),
        ]
    )


def support_in_crop_ratio(row: dict[str, Any], target: dict[str, Any]) -> float:
    support = row.get("routing", {}).get("subject_support_overlay", {}).get("bbox_norm_xyxy")
    crop = target.get("bbox_norm_xyxy")
    support_area = norm_box_area(support)
    if support_area <= 0:
        return 0.0
    return norm_box_intersection_area(support, crop) / support_area


def support_visual_in_crop_ratio(row: dict[str, Any], target: dict[str, Any]) -> float:
    support = support_visual_box(row)
    support_area = norm_box_area(support)
    if support_area <= 0:
        return 0.0
    return norm_box_intersection_area(support, target.get("bbox_norm_xyxy")) / support_area


def is_portrait_checklist_candidate(row: dict[str, Any]) -> bool:
    image_id = str(row.get("image_id") or "")
    if image_id in EXCLUDED_PORTRAIT_CHECKLIST_IMAGE_IDS:
        return False
    routing = row.get("routing", {})
    mode = str(routing.get("subject_mode") or "")
    flags = routing.get("flags") or {}
    return mode.startswith("portrait") and bool(flags.get("subject_mode_has_person", True))


def label_example_priority(
    key: str,
    row: dict[str, Any],
    value: str,
    support_ratio: float,
    visual_support_ratio: float,
) -> int:
    image_id = str(row.get("image_id") or "")
    if image_id in PREFERRED_LABEL_EXAMPLE_IMAGE_IDS.get(key, []):
        if key == "lookroom_bad":
            return int(abs(visual_support_ratio - 0.60) * 100)
        if key == "subject_coverage_bad":
            return int(abs(visual_support_ratio - 0.50) * 100)
        if key in {"subject_coverage_good", "lookroom_good", "headroom_good"}:
            return int((1.0 - visual_support_ratio) * 100)
        return 0
    base = 100
    if key == "subject_coverage_good":
        return base if value == "excellent" else base + 2
    if key == "subject_coverage_bad":
        return base if value in {"poor", "bad"} else base + 5
    return base


def load_context(max_rows: int) -> dict[str, Any]:
    manifest = read_json(MANIFEST_JSON, {})
    batch_rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    sample_by_mode: dict[str, dict[str, Any]] = {}
    row_with_roles: dict[str, Any] | None = None
    rows_with_support: list[dict[str, Any]] = []
    rows_with_support_map: list[dict[str, Any]] = []
    rows_for_guidance_only: list[dict[str, Any]] = []
    curated_rows_by_id: dict[str, dict[str, Any]] = {}
    seen_support_images: set[str] = set()
    seen_support_map_images: set[str] = set()
    seen_guidance_images: set[str] = set()
    label_examples: dict[str, dict[str, Any]] = {}
    curated_ids = (
        set(PREFERRED_JOINT_OVERLAY_IMAGE_IDS)
        | set(PREFERRED_SUPPORT_MAP_IMAGE_IDS)
        | set(PREFERRED_C7_SUPPORT_EXAMPLE_IMAGE_IDS)
        | set(PREFERRED_SUPPORT_MASS_ONLY_IMAGE_IDS)
        | set(PREFERRED_ROLE_ROW_IMAGE_IDS)
        | set(PREFERRED_PORTRAIT_AUDIT_IMAGE_IDS)
        | set(PREFERRED_GUIDANCE_IMAGE_IDS)
        | CURATED_ROUTE_IMAGE_IDS
    )

    def remember_support_rows(row: dict[str, Any], mode: str) -> None:
        image_id = str(row.get("image_id"))
        if mode in EXCLUDED_RESULT_MODES or not resolve_image_path(row):
            return
        support = row.get("routing", {}).get("subject_support_overlay")
        if support and image_id not in seen_support_images:
            rows_with_support.append(row)
            seen_support_images.add(image_id)
        if is_support_map_row(row) and image_id not in seen_support_map_images:
            rows_with_support_map.append(row)
            seen_support_map_images.add(image_id)
        if support and image_id not in seen_guidance_images and image_id not in EXCLUDED_PORTRAIT_CHECKLIST_IMAGE_IDS:
            rows_for_guidance_only.append(row)
            seen_guidance_images.add(image_id)

    def remember_curated_row(row: dict[str, Any]) -> None:
        image_id = str(row.get("image_id"))
        if image_id not in curated_ids or not resolve_image_path(row):
            return
        current = curated_rows_by_id.get(image_id)
        if current is None or row.get("target_ar") == "FREE":
            curated_rows_by_id[image_id] = row

    def remember_label_example(
        key: str,
        row: dict[str, Any],
        target: dict[str, Any],
        value: str,
        priority: int,
        support_ratio: float | None = None,
        visual_support_ratio: float | None = None,
    ) -> None:
        current = label_examples.get(key)
        if current is not None and current.get("priority", 99) <= priority:
            return
        if not resolve_image_path(row):
            return
        label_examples[key] = {
            "row": row,
            "target": target,
            "value": value,
            "priority": priority,
            "support_in_crop_ratio": support_ratio,
            "visual_support_in_crop_ratio": visual_support_ratio,
        }

    def scan_label_examples(row: dict[str, Any], mode: str) -> None:
        if mode in EXCLUDED_RESULT_MODES:
            return
        for target in row.get("matching_targets") or []:
            checklist = target.get("checklist_labels") or {}
            coverage = str(checklist.get("subject_coverage") or "").lower()
            support_ratio = support_in_crop_ratio(row, target)
            visual_support_ratio = support_visual_in_crop_ratio(row, target)
            if coverage in {"good", "excellent"} and support_ratio >= 0.97 and visual_support_ratio >= 0.95:
                remember_label_example(
                    "subject_coverage_good",
                    row,
                    target,
                    coverage,
                    label_example_priority("subject_coverage_good", row, coverage, support_ratio, visual_support_ratio),
                    support_ratio,
                    visual_support_ratio,
                )
            elif coverage in {"poor", "bad", "marginal"} and support_ratio <= 0.75 and visual_support_ratio <= 0.75:
                remember_label_example(
                    "subject_coverage_bad",
                    row,
                    target,
                    coverage,
                    label_example_priority("subject_coverage_bad", row, coverage, support_ratio, visual_support_ratio),
                    support_ratio,
                    visual_support_ratio,
                )

            if not is_portrait_checklist_candidate(row):
                continue
            headroom = str(checklist.get("headroom") or "").lower()
            if headroom == "headroom_ok":
                remember_label_example(
                    "headroom_good",
                    row,
                    target,
                    headroom,
                    label_example_priority("headroom_good", row, headroom, support_ratio, visual_support_ratio),
                    support_ratio,
                    visual_support_ratio,
                )
            elif headroom in {"headroom_tight", "headroom_loose"}:
                remember_label_example(
                    "headroom_bad",
                    row,
                    target,
                    headroom,
                    label_example_priority("headroom_bad", row, headroom, support_ratio, visual_support_ratio),
                    support_ratio,
                    visual_support_ratio,
                )

            lookroom = str(checklist.get("lookroom") or "").lower()
            if not (row.get("routing", {}).get("flags") or {}).get("is_profile_view"):
                continue
            if lookroom == "lookroom_adequate":
                remember_label_example(
                    "lookroom_good",
                    row,
                    target,
                    lookroom,
                    label_example_priority("lookroom_good", row, lookroom, support_ratio, visual_support_ratio),
                    support_ratio,
                    visual_support_ratio,
                )
            elif lookroom in {"lookroom_insufficient", "lookroom_excessive"}:
                remember_label_example(
                    "lookroom_bad",
                    row,
                    target,
                    lookroom,
                    label_example_priority("lookroom_bad", row, lookroom, support_ratio, visual_support_ratio),
                    support_ratio,
                    visual_support_ratio,
                )

    for row in iter_jsonl(BATCH_JSONL, limit=max_rows) or []:
        batch_rows.append(row)
        mode = str(row.get("routing", {}).get("subject_mode") or "unknown")
        remember_curated_row(row)
        if mode not in EXCLUDED_RESULT_MODES:
            route_counts[mode] += 1
            sample_by_mode.setdefault(mode, row)
        action_counts[str(row.get("decision_target", {}).get("decision_type") or "unknown")] += 1
        for role in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
            role_counts[role] += len(row.get(role) or [])
        remember_support_rows(row, mode)
        if row_with_roles is None and mode not in EXCLUDED_RESULT_MODES:
            total_extra = sum(len(row.get(role) or []) for role in ("candidate_pool", "ignored_candidates", "overflow_candidates"))
            if total_extra > 0 or len(row.get("matching_targets") or []) >= 3:
                row_with_roles = row
        scan_label_examples(row, mode)

    if max_rows < LABEL_EXAMPLE_SCAN_LIMIT:
        for row in iter_jsonl(BATCH_JSONL, limit=LABEL_EXAMPLE_SCAN_LIMIT) or []:
            mode = str(row.get("routing", {}).get("subject_mode") or "unknown")
            remember_curated_row(row)
            remember_support_rows(row, mode)
            scan_label_examples(row, mode)

    candidate_ar_counts: Counter[str] = Counter()
    candidate_source_counts: Counter[str] = Counter()
    candidate_examples: list[dict[str, Any]] = []
    for row in iter_jsonl(CANDIDATE_JSONL, limit=250) or []:
        candidate_examples.append(row)
        for ar, candidates in (row.get("candidates_by_ar") or {}).items():
            candidate_ar_counts[str(ar)] += len(candidates or [])
            for cand in candidates or []:
                source = str(cand.get("source") or "unknown")
                candidate_source_counts[source] += 1

    public_dataset_examples = collect_public_dataset_examples()
    public_cropper_examples = collect_public_cropper_examples(public_dataset_examples)

    return {
        "manifest": manifest,
        "batch_rows": batch_rows,
        "route_counts": route_counts,
        "action_counts": action_counts,
        "role_counts": role_counts,
        "sample_by_mode": sample_by_mode,
        "row_with_roles": row_with_roles or (batch_rows[0] if batch_rows else None),
        "rows_with_support": rows_with_support[:80],
        "rows_with_support_map": rows_with_support_map[:200],
        "rows_for_guidance_only": rows_for_guidance_only[:200],
        "curated_rows_by_id": curated_rows_by_id,
        "label_examples": label_examples,
        "candidate_ar_counts": candidate_ar_counts,
        "candidate_source_counts": candidate_source_counts,
        "candidate_examples": candidate_examples,
        "public_dataset_examples": public_dataset_examples,
        "public_cropper_examples": public_cropper_examples,
        "public_export": read_json(
            PROJECT_ROOT
            / "artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/public_score_label_export_summary.json",
            {},
        ),
        "public_distill": read_json(
            PROJECT_ROOT
            / "artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/public_score_distill_v2_label_convert_summary.json",
            {},
        ),
        "release_gate": read_json(
            PROJECT_ROOT
            / "artifacts/mobilecropnet_v4/unified_recovery_20260424/release_gate_base_strict_check/release_gate_manifest.json",
            {},
        ),
        "final_leaderboard": read_json(
            PROJECT_ROOT
            / "artifacts/mobilecropnet_v4/unified_recovery_20260424/final_deployment_leaderboard_base_strict_check/final_deployment_leaderboard.json",
            {},
        ),
        "subject_report": read_json(
            PROJECT_ROOT
            / "artifacts/mobilecropnet_v4/subject_box_head_uctr_20260423/subject_box_report_latest/subject_box_report.json",
            {},
        ),
        "equal4_teacher_leaderboard": read_json(EQUAL4_TEACHER_LEADERBOARD_JSON, {}),
    }


def plot_bar(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    color: str = BLUE,
    horizontal: bool = False,
    notes: str | None = None,
) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 17, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(12, 7), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(WHITE)
    if horizontal:
        ax.barh(labels, values, color=color, edgecolor=INK, linewidth=0.8)
        ax.set_xlabel(ylabel)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:,.0f}", va="center", color=INK, fontsize=10)
    else:
        ax.bar(labels, values, color=color, edgecolor=INK, linewidth=0.8)
        ax.set_ylabel(ylabel)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.tick_params(axis="x", rotation=25)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y" if not horizontal else "x", color=GRID, linewidth=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#B7BEC9")
    if notes:
        fig.text(0.03, 0.03, notes, color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.06 if notes else 0, 1, 1))
    ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_c1_c7_perception_stack(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1040), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "C1-C7 Perception Stack",
        "The same perception cache feeds routing, candidate construction, scoring, conversion repair, and student targets.",
    )
    stages = [
        ("C1", "Semantic context", "caption, text alignment, scene/object/person hints", BLUE),
        ("C2", "Object regions", "boxes, masks, foreground instances", TEAL),
        ("C3", "Person cues", "face, pose, gaze, group structure", ORANGE),
        ("C4", "OCR/text safety", "text regions, text-cut risk, reading safety", RED),
        ("C5", "Geometry", "horizon, symmetry, centrality, margins", GOLD),
        ("C6", "Portrait policy", "headroom, lookroom, face/joint safety", PURPLE),
        ("C7", "Support map", "saliency-support and subject-support attribution", GREEN),
    ]
    y = 160
    for idx, (code, title, body, color) in enumerate(stages):
        box_card(d, (70, y, 650, y + 88), f"{code}  {title}", body, accent=color)
        if idx < len(stages) - 1:
            d.line((360, y + 88, 360, y + 114), fill=MUTED, width=4)
            d.polygon([(360, y + 122), (350, y + 106), (370, y + 106)], fill=MUTED)
        y += 118

    downstream = [
        ("Routing", "subject_mode, route confidence, policy prior"),
        ("Candidate Bank", "baseline, AR candidates, subject/support proposals"),
        ("Teacher Scoring", "rank score, macro score, checklist, safety"),
        ("Safe Conversion", "positive selection, demotion, provenance"),
        ("Student Contract", "batch + listwise + pairwise + aux targets"),
    ]
    for i, (title, body) in enumerate(downstream):
        y0 = 185 + i * 145
        box_card(d, (900, y0, 1490, y0 + 108), title, body, fill="#FCFBF8", accent=[BLUE, TEAL, ORANGE, RED, GREEN][i])
        arrow = FancyArrowPatch(
            (650, y0 + 54),
            (900, y0 + 54),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.5,
            color=MUTED,
        )
        fig, ax = plt.subplots(figsize=(1, 1))
        plt.close(fig)
        # Draw the arrow manually to avoid mixing matplotlib and PIL canvases.
        d.line((650, y0 + 54, 880, y0 + 54), fill=MUTED, width=4)
        d.polygon([(900, y0 + 54), (878, y0 + 42), (878, y0 + 66)], fill=MUTED)

    counts = ctx["manifest"].get("sstk_main_batch", {}).get("counts", {})
    summary = [
        ("Batch rows", counts.get("conditional_detr_batch", 0)),
        ("Pairwise rows", counts.get("pairwise", 0)),
        ("Checklist candidates", counts.get("checklist", 0)),
    ]
    x = 760
    y = 875
    for label, value in summary:
        d.rounded_rectangle((x, y, x + 235, y + 88), radius=18, fill=WHITE, outline=GRID, width=2)
        d.text((x + 18, y + 16), f"{value:,.0f}", font=font(30, bold=True), fill=INK)
        d.text((x + 18, y + 55), label, font=font(18), fill=MUTED)
        x += 255
    save_pil(img, path)


def draw_annotation_dataset_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    title: str,
    example: dict[str, Any] | None,
    accent: str,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rounded_rectangle(xyxy, radius=18, fill=WHITE, outline=GRID, width=2)
    draw_tag(draw, (x1 + 18, y1 + 18), title, accent, size=20)
    if not example or example.get("error"):
        draw_wrapped(draw, (x1 + 24, y1 + 78), str((example or {}).get("error", "No example available.")), x2 - x1 - 48, fill=RED, size=22)
        return
    image_path = Path(str(example.get("image_path", "")))
    thumb, local_area = open_image_with_bbox(image_path, (x2 - x1 - 40, y2 - y1 - 155))
    canvas.paste(thumb, (x1 + 20, y1 + 62))
    area = offset_area(local_area, (x1 + 20, y1 + 62))
    is_gaic = "GAIC" in title
    legend_x = x1 + 22
    for box_record in example.get("boxes") or []:
        box = box_record.get("bbox")
        color = str(box_record.get("color") or accent)
        fill_norm_box(canvas, box, area, color, alpha=42)
        if is_gaic:
            draw_norm_outline(draw, box, area, color, width=4)
            next_x, _ = draw_tag(draw, (legend_x, y2 - 116), str(box_record.get("label") or "box"), color, size=15)
            legend_x = next_x + 8
        else:
            draw_norm_box(draw, box, area, color, str(box_record.get("label") or "box"), width=4)
    draw.text((x1 + 22, y2 - 78), f"{example.get('dataset')}  image_id={example.get('image_id')}", font=font(20, bold=True), fill=INK)
    draw_wrapped(draw, (x1 + 22, y2 - 48), str(example.get("note", "")), x2 - x1 - 44, fill=MUTED, size=17)


def fig_public_dataset_annotation_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1120), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Public crop benchmark dataset annotation examples",
        "Each panel shows the actual local dataset image and annotation type used by the benchmark adapter.",
    )
    examples = ctx.get("public_dataset_examples", {})
    panels = [
        ((70, 170, 770, 600), "FCDB: expert crop box", examples.get("fcdb"), BLUE),
        ((830, 170, 1530, 600), "CPC: scored candidate views", examples.get("cpc"), ORANGE),
        ((70, 650, 770, 1080), "GNMC: AR-conditioned GT", examples.get("gnmc"), PURPLE),
        ((830, 650, 1530, 1080), "GAIC: dense MOS crops", examples.get("gaic"), GREEN),
    ]
    for box, title, example, color in panels:
        draw_annotation_dataset_panel(img, d, box, title, example, color)
    save_pil(img, path)


def best_prediction_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [cand for cand in row.get("candidates") or [] if parse_norm_box(cand.get("bbox_xyxy_norm")) is not None]
    if not candidates:
        return None

    def key(cand: dict[str, Any]) -> tuple[int, float]:
        rank = cand.get("model_rank")
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = 10_000
        try:
            score = float(cand.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return (rank_value, -score)

    return sorted(candidates, key=key)[0]


def draw_public_cropper_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    method_label: str,
    row: dict[str, Any],
    image_path: Path,
    color: str,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rounded_rectangle(xyxy, radius=16, fill=WHITE, outline=GRID, width=2)
    draw_tag(draw, (x1 + 18, y1 + 16), method_label, color, size=18)
    thumb, local_area = open_image_with_bbox(image_path, (x2 - x1 - 36, y2 - y1 - 102))
    canvas.paste(thumb, (x1 + 18, y1 + 56))
    area = offset_area(local_area, (x1 + 18, y1 + 56))
    best = best_prediction_candidate(row)
    if best:
        draw_norm_box(draw, best.get("bbox_xyxy_norm"), area, color, "top-1", width=5)
        score = best.get("score")
        rank = best.get("model_rank")
        draw.text(
            (x1 + 20, y2 - 36),
            f"rank={rank}  score={float(score):.3f}" if isinstance(score, (int, float)) else f"rank={rank}",
            font=font(18, bold=True),
            fill=INK,
        )


def fig_public_cropper_inference_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1180), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Public cropper inference examples",
        "GAIC and CGS public croppers are candidate rankers; top-1 boxes can differ even on the same image-task.",
    )
    examples = ctx.get("public_cropper_examples") or []
    row_h = 300
    start_y = 175
    for idx, example in enumerate(examples[:3]):
        y = start_y + idx * 325
        dataset = str(example.get("dataset", "")).upper()
        ar = example.get("target_ar") or "FREE"
        image_path = Path(str(example.get("image_path", "")))
        d.text((65, y + 12), f"{dataset}  target_ar={ar}", font=font(24, bold=True), fill=INK)
        d.text((65, y + 42), f"image_id={example.get('image_id')}", font=font(17), fill=MUTED)
        draw_public_cropper_cell(
            img,
            d,
            (410, y, 955, y + row_h),
            "public_cropper_gaic",
            example.get("public_cropper_gaic") or {},
            image_path,
            ORANGE,
        )
        draw_public_cropper_cell(
            img,
            d,
            (990, y, 1535, y + row_h),
            "public_cropper_cgs",
            example.get("public_cropper_cgs") or {},
            image_path,
            BLUE,
        )
        box_card(
            d,
            (65, y + 58, 370, y + row_h),
            "Reading",
            "These boxes are inferred ranker outputs, not GT. They expose public aesthetic/ranking bias but do not encode MobileCropNet safety/action policy.",
            fill="#FCFBF8",
            accent=RED,
        )
    if not examples:
        draw_wrapped(d, (80, 190), "No public cropper prediction rows were matched to dataset examples.", 1400, fill=RED, size=28, bold=True)
    save_pil(img, path)


def subject_union_norm(row: dict[str, Any]) -> list[float] | None:
    subject_set = row.get("routing", {}).get("subject_set") or {}
    for key in ("union_box_xyxy", "bbox_norm_xyxy", "box_norm_xyxy"):
        box = subject_set.get(key)
        norm = absolute_box_to_norm(row, box)
        if norm is not None:
            return norm
    return None


def choose_joint_overlay_row(ctx: dict[str, Any]) -> dict[str, Any] | None:
    candidates = list(ctx.get("rows_with_support_map") or []) + list(ctx["rows_with_support"]) + ctx["batch_rows"]
    candidates = list((ctx.get("curated_rows_by_id") or {}).values()) + candidates
    preferred = {image_id: idx for idx, image_id in enumerate(PREFERRED_JOINT_OVERLAY_IMAGE_IDS)}
    candidates = sorted(
        candidates,
        key=lambda row: (
            preferred.get(str(row.get("image_id")), 999),
            0 if (row.get("routing", {}).get("subject_set") or {}).get("num_person", 0) else 1,
            0 if is_support_map_row(row) else 1,
        ),
    )
    for row in candidates:
        mode = str(row.get("routing", {}).get("subject_mode") or "unknown")
        if mode in EXCLUDED_RESULT_MODES:
            continue
        routing = row.get("routing", {})
        if (
            routing.get("subject_support_overlay", {}).get("bbox_norm_xyxy")
            and routing.get("subject_prior_bbox_norm_xyxy")
            and subject_union_norm(row)
            and resolve_image_path(row)
        ):
            return row
    for row in ctx["rows_with_support"]:
        if resolve_image_path(row):
            return row
    return None


def fig_c2_c3_c7_joint_overlay(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 980), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "C2/C3/C7 joint perception overlay",
        "C2 object/subject regions, C3 subject/person prior, and C7 support evidence are different teacher signals.",
    )
    row = choose_joint_overlay_row(ctx)
    if not row:
        draw_wrapped(d, (80, 190), "No row with C2/C3/C7 overlay evidence was available.", 1400, fill=RED, size=30, bold=True)
        save_pil(img, path)
        return
    image_path = resolve_image_path(row)
    thumb, local_area = open_image_with_bbox(image_path, (980, 640)) if image_path else (Image.new("RGB", (980, 640), "#EFEAE1"), (0, 0, 980, 640))
    img.paste(thumb, (70, 185))
    area = offset_area(local_area, (70, 185))
    routing = row.get("routing", {})
    c2_box = subject_union_norm(row)
    c3_box = routing.get("subject_prior_bbox_norm_xyxy")
    c7_box = support_bbox_for_render(row)
    draw_support_heatmap(img, row, area, alpha_max=118)
    fill_norm_box(img, c7_box, area, GREEN, alpha=36)
    draw_norm_box(d, c2_box, area, TEAL, "", width=6)
    draw_norm_box(d, c3_box, area, PURPLE, "", width=5)
    draw_norm_box(d, c7_box, area, GREEN, "", width=5)
    draw_norm_box(d, support_core_box_for_render(row), area, GOLD, "", width=3)
    draw_support_centroid(d, row, area)
    d.rounded_rectangle((70, 850, 1050, 920), radius=18, fill=WHITE, outline=GRID, width=2)
    d.text(
        (95, 870),
        f"image_id={row.get('image_id')}   mode={routing.get('subject_mode')}   target_ar={row.get('target_ar')}",
        font=font(24, bold=True),
        fill=INK,
    )

    legend = [
        ("C2 object/subject", "detected foreground or subject union used for candidate constraints", TEAL),
        ("C3 subject/person prior", "person/body-aware or detector-derived prior used by route-specific framing", PURPLE),
        ("C7 support evidence", "support heatmap/envelope that should remain visually preserved", GREEN),
    ]
    y = 205
    for title, body, color in legend:
        box_card(d, (1110, y, 1525, y + 145), title, body, accent=color)
        y += 172
    box_card(
        d,
        (1110, y, 1525, y + 190),
        "Interpretation",
        "Boxes are not interchangeable. A crop can satisfy one signal and still violate another, so audit overlays must preserve each source.",
        fill="#FCFBF8",
        accent=RED,
    )
    save_pil(img, path)


def fig_subject_support_map_region_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1060), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Subject support-map region examples",
        "Green heat is decoded support mass; the green outline is its support envelope. No crop box is drawn.",
    )
    rows = preferred_unique_rows(
        [
            row
            for row in list((ctx.get("curated_rows_by_id") or {}).values()) + list(ctx.get("rows_with_support_map", []))
            if is_support_map_row(row)
        ],
        PREFERRED_SUPPORT_MAP_IMAGE_IDS,
        4,
    )
    cell_w, cell_h = 700, 360
    for idx, row in enumerate(rows):
        col = idx % 2
        rr = idx // 2
        x = 75 + col * 760
        y = 185 + rr * 410
        d.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=18, fill=WHITE, outline=GRID, width=2)
        image_path = resolve_image_path(row)
        thumb, local_area = open_image_with_bbox(image_path, (cell_w - 40, 245)) if image_path else (Image.new("RGB", (cell_w - 40, 245), "#EFEAE1"), (0, 0, cell_w - 40, 245))
        img.paste(thumb, (x + 20, y + 20))
        area = offset_area(local_area, (x + 20, y + 20))
        draw_support_heatmap(img, row, area, alpha_max=138)
        support = support_bbox_for_render(row)
        core = support_core_box_for_render(row)
        draw_norm_box(d, support, area, GREEN, "", width=5)
        draw_norm_box(d, core, area, GOLD, "", width=3)
        draw_support_centroid(d, row, area)
        mode = row.get("routing", {}).get("subject_mode", "unknown")
        d.text((x + 24, y + 286), str(mode), font=font(24, bold=True), fill=INK)
        d.text(
            (x + 24, y + 320),
            f"image_id={row.get('image_id')}   {support_repr_label(row)}",
            font=font(18),
            fill=MUTED,
        )
    if not rows:
        draw_wrapped(d, (80, 190), "No support-map rows were available.", 1400, fill=RED, size=30, bold=True)
    save_pil(img, path)


def rows_for_modes(ctx: dict[str, Any], modes: list[str], max_n: int) -> list[dict[str, Any]]:
    rows = []
    curated_rows = ctx.get("curated_rows_by_id") or {}
    for mode in modes:
        row = None
        for image_id in CURATED_ROUTE_MODE_IMAGE_IDS.get(mode, []):
            candidate = curated_rows.get(image_id)
            if candidate and str(candidate.get("routing", {}).get("subject_mode") or "") == mode:
                row = candidate
                break
        if row is None:
            row = ctx["sample_by_mode"].get(mode)
        if row:
            rows.append(row)
    if len(rows) < max_n:
        for row in ctx["batch_rows"]:
            if str(row.get("routing", {}).get("subject_mode") or "unknown") in EXCLUDED_RESULT_MODES:
                continue
            if row not in rows:
                rows.append(row)
            if len(rows) >= max_n:
                break
    return rows[:max_n]


def preferred_unique_rows(rows: list[dict[str, Any]], preferred_ids: list[str], max_n: int) -> list[dict[str, Any]]:
    preferred = {image_id: idx for idx, image_id in enumerate(preferred_ids)}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: preferred.get(str(item.get("image_id")), 999)):
        image_id = str(row.get("image_id"))
        if image_id in seen or not resolve_image_path(row):
            continue
        out.append(row)
        seen.add(image_id)
        if len(out) >= max_n:
            break
    return out


def curated_source_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        list((ctx.get("curated_rows_by_id") or {}).values())
        + list(ctx.get("rows_with_support_map") or [])
        + list(ctx.get("rows_with_support") or [])
        + list(ctx.get("batch_rows") or [])
    )


def support_review_rows(ctx: dict[str, Any], max_n: int) -> list[dict[str, Any]]:
    return preferred_unique_rows(
        curated_source_rows(ctx),
        PREFERRED_C7_SUPPORT_EXAMPLE_IMAGE_IDS,
        max_n,
    )


def preferred_row(ctx: dict[str, Any], preferred_ids: list[str]) -> dict[str, Any] | None:
    rows = preferred_unique_rows(curated_source_rows(ctx), preferred_ids, 1)
    return rows[0] if rows else None


def choose_candidate_role_row(ctx: dict[str, Any]) -> dict[str, Any] | None:
    preferred = {image_id: idx for idx, image_id in enumerate(PREFERRED_ROLE_ROW_IMAGE_IDS)}

    def role_score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        image_id = str(row.get("image_id") or "")
        role_counts = {role: len(row.get(role) or []) for role in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates")}
        return (
            preferred.get(image_id, 999),
            0 if role_counts["overflow_candidates"] else 1,
            0 if role_counts["ignored_candidates"] else 1,
            0 if role_counts["candidate_pool"] else 1,
            -role_counts["matching_targets"],
        )

    for row in sorted(curated_source_rows(ctx), key=role_score):
        mode = str(row.get("routing", {}).get("subject_mode") or "")
        if mode in EXCLUDED_RESULT_MODES or not resolve_image_path(row):
            continue
        total_roles = sum(len(row.get(role) or []) for role in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"))
        if total_roles > 0:
            return row
    return ctx.get("row_with_roles")


def render_row_grid(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[dict[str, Any]],
    box_plan: str = "route",
) -> None:
    img = Image.new("RGB", (1600, 1060), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(d, title, subtitle)
    cell_w, cell_h = 480, 355
    start_x, start_y = 68, 165
    for idx, row in enumerate(rows[:6]):
        col = idx % 3
        r = idx // 3
        x = start_x + col * (cell_w + 30)
        y = start_y + r * (cell_h + 60)
        d.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=18, fill=WHITE, outline=GRID, width=2)
        img_path = resolve_image_path(row)
        if img_path:
            thumb, local_area = open_image_with_bbox(img_path, (cell_w - 36, 250))
        else:
            thumb = Image.new("RGB", (cell_w - 36, 250), "#EFEAE1")
            local_area = (0, 0, thumb.width, thumb.height)
        img.paste(thumb, (x + 18, y + 18))
        area = offset_area(local_area, (x + 18, y + 18))
        if box_plan == "support":
            draw_support_heatmap(img, row, area, alpha_max=108)
        if box_plan in {"route", "roles", "support"}:
            base_label = "" if box_plan == "support" else "base"
            winner_label = "" if box_plan == "support" else "win"
            draw_norm_box(d, row.get("baseline", {}).get("bbox_norm_xyxy"), area, "#7A8696", base_label, width=3)
            draw_norm_box(d, row.get("decision_target", {}).get("winner_post_gate_bbox"), area, ORANGE, winner_label, width=4)
        if box_plan == "support":
            support = support_bbox_for_render(row)
            prior = row.get("routing", {}).get("subject_prior_bbox_norm_xyxy")
            draw_norm_box(d, support, area, GREEN, "", width=4)
            draw_norm_box(d, prior, area, BLUE, "", width=3)
            draw_norm_box(d, support_core_box_for_render(row), area, GOLD, "", width=2)
            draw_support_centroid(d, row, area)
        mode = row.get("routing", {}).get("subject_mode", "unknown")
        action = row.get("decision_target", {}).get("decision_type", "unknown")
        ar = row.get("target_ar", "unknown")
        d.text((x + 20, y + 284), str(mode), font=font(22, bold=True), fill=INK)
        d.text((x + 20, y + 314), f"target_ar={ar}  action={action}", font=font(18), fill=MUTED)
    save_pil(img, path)


def fig_c7_subject_support_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1120), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "C7 subject-support overlay examples",
        "Heat=decoded support mass, gray=baseline, orange=winner crop, green=support, blue=prior, gold=core.",
    )
    rows = support_review_rows(ctx, 8)
    cell_w, cell_h = 360, 390
    start_x, start_y = 58, 165
    gap_x, gap_y = 28, 42
    image_size = (cell_w - 34, 230)
    for idx, row in enumerate(rows):
        col = idx % 4
        rr = idx // 4
        x = start_x + col * (cell_w + gap_x)
        y = start_y + rr * (cell_h + gap_y)
        d.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=18, fill=WHITE, outline=GRID, width=2)
        image_path = resolve_image_path(row)
        thumb, local_area = open_image_with_bbox(image_path, image_size) if image_path else (Image.new("RGB", image_size, "#EFEAE1"), (0, 0, image_size[0], image_size[1]))
        img.paste(thumb, (x + 17, y + 18))
        area = offset_area(local_area, (x + 17, y + 18))
        draw_support_heatmap(img, row, area, alpha_max=104)
        draw_norm_box(d, row.get("baseline", {}).get("bbox_norm_xyxy"), area, "#7A8696", "", width=2)
        draw_norm_box(d, row.get("decision_target", {}).get("winner_post_gate_bbox"), area, ORANGE, "", width=3)
        draw_norm_box(d, support_bbox_for_render(row), area, GREEN, "", width=4)
        draw_norm_box(d, row.get("routing", {}).get("subject_prior_bbox_norm_xyxy"), area, BLUE, "", width=3)
        draw_norm_box(d, support_core_box_for_render(row), area, GOLD, "", width=2)
        draw_support_centroid(d, row, area)
        routing = row.get("routing", {})
        subject_set = routing.get("subject_set") or {}
        mode = str(routing.get("subject_mode") or "unknown")
        action = row.get("decision_target", {}).get("decision_type", "unknown")
        d.text((x + 20, y + 268), mode, font=font(21, bold=True), fill=INK)
        d.text((x + 20, y + 298), f"image_id={row.get('image_id')}", font=font(15), fill=MUTED)
        d.text(
            (x + 20, y + 322),
            f"target_ar={row.get('target_ar')}  action={action}",
            font=font(15),
            fill=MUTED,
        )
        if mode.startswith("portrait"):
            d.text(
                (x + 20, y + 346),
                f"verified person subject: n={subject_set.get('num_person')}",
                font=font(14, bold=True),
                fill=GREEN,
            )
    if not rows:
        draw_wrapped(d, (80, 190), "No C7 support rows were available.", 1400, fill=RED, size=30, bold=True)
    save_pil(img, path)


def fig_decoded_support_mass_only_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 980), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Decoded support-mass only examples",
        "Only decoded support mass is overlaid. No bbox, crop, core, centroid, or prior outline is drawn.",
    )
    rows = preferred_unique_rows(
        [row for row in curated_source_rows(ctx) if has_decoded_support_mass(row)],
        PREFERRED_SUPPORT_MASS_ONLY_IMAGE_IDS,
        6,
    )
    cell_w, cell_h = 480, 330
    start_x, start_y = 62, 170
    gap_x, gap_y = 35, 52
    for idx, row in enumerate(rows):
        col = idx % 3
        rr = idx // 3
        x = start_x + col * (cell_w + gap_x)
        y = start_y + rr * (cell_h + gap_y)
        d.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=18, fill=WHITE, outline=GRID, width=2)
        image_path = resolve_image_path(row)
        thumb, local_area = open_image_with_bbox(image_path, (cell_w - 34, 220)) if image_path else (Image.new("RGB", (cell_w - 34, 220), "#EFEAE1"), (0, 0, cell_w - 34, 220))
        img.paste(thumb, (x + 17, y + 18))
        area = offset_area(local_area, (x + 17, y + 18))
        draw_support_heatmap(img, row, area, alpha_max=148)
        mode = row.get("routing", {}).get("subject_mode", "unknown")
        d.text((x + 20, y + 256), str(mode), font=font(22, bold=True), fill=INK)
        d.text((x + 20, y + 284), f"image_id={row.get('image_id')}", font=font(15), fill=MUTED)
        d.text((x + 20, y + 306), support_repr_label(row), font=font(15), fill=MUTED)
    if not rows:
        draw_wrapped(d, (80, 190), "No decoded support-mass rows were available.", 1400, fill=RED, size=30, bold=True)
    save_pil(img, path)


def fig_subject_guidance_clean_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 980), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Clean subject and crop-guidance examples",
        "Only subject/guidance evidence is drawn: no baseline, winner, or candidate crop boxes.",
    )
    source_rows = (
        list((ctx.get("curated_rows_by_id") or {}).values())
        + list(ctx.get("rows_for_guidance_only") or [])
        + list(ctx.get("rows_with_support_map") or [])
    )
    rows = preferred_unique_rows(source_rows, PREFERRED_GUIDANCE_IMAGE_IDS, 6)
    cell_w, cell_h = 480, 330
    start_x, start_y = 62, 170
    gap_x, gap_y = 35, 52
    for idx, row in enumerate(rows):
        col = idx % 3
        rr = idx // 3
        x = start_x + col * (cell_w + gap_x)
        y = start_y + rr * (cell_h + gap_y)
        d.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=18, fill=WHITE, outline=GRID, width=2)
        image_path = resolve_image_path(row)
        thumb, local_area = open_image_with_bbox(image_path, (cell_w - 40, 220)) if image_path else (Image.new("RGB", (cell_w - 40, 220), "#EFEAE1"), (0, 0, cell_w - 40, 220))
        img.paste(thumb, (x + 20, y + 20))
        area = offset_area(local_area, (x + 20, y + 20))
        if is_support_map_row(row):
            draw_support_heatmap(img, row, area, alpha_max=142)
            draw_norm_box(d, guidance_box_for_render(row), area, GREEN, "", width=5)
            draw_norm_box(d, support_core_box_for_render(row), area, GOLD, "", width=3)
            draw_support_centroid(d, row, area)
            signal = support_repr_label(row)
        else:
            guidance = guidance_box_for_render(row)
            fill_norm_box(img, guidance, area, BLUE, alpha=46)
            draw_norm_box(d, guidance, area, BLUE, "", width=5)
            signal = "subject union" if str(row.get("routing", {}).get("subject_mode") or "") in {"object_multi", "portrait_single", "portrait_group"} else "subject bbox"
        mode = row.get("routing", {}).get("subject_mode", "unknown")
        d.text((x + 24, y + 256), str(mode), font=font(24, bold=True), fill=INK)
        d.text((x + 24, y + 286), f"image_id={row.get('image_id')}", font=font(15), fill=MUTED)
        d.text((x + 24, y + 308), signal, font=font(15), fill=MUTED)
    if not rows:
        draw_wrapped(d, (80, 190), "No clean subject/guidance rows were available.", 1400, fill=RED, size=30, bold=True)
    save_pil(img, path)


def fig_route_collapse(path: Path, ctx: dict[str, Any]) -> None:
    manifest = ctx["manifest"]
    pair_counts = manifest.get("sstk_main_batch", {}).get("pair_count_by_mode", {})
    labels = [k for k in pair_counts.keys() if k not in EXCLUDED_RESULT_MODES]
    values = [pair_counts[k] for k in labels]
    plot_bar(
        path,
        "Route supervision distribution and collapse diagnosis",
        labels,
        values,
        "pairwise rows",
        color=ORANGE,
        horizontal=True,
        notes="Final shortlist evidence: 19/19 entries failed route-collapse hard gate; default portrait_single accuracy = 0.012.",
    )


def fig_candidate_bank_by_ar(path: Path, ctx: dict[str, Any]) -> None:
    counts = ctx["candidate_ar_counts"]
    labels = list(counts.keys())[:10]
    values = [counts[k] for k in labels]
    plot_bar(
        path,
        "Candidate bank size by target aspect ratio",
        labels,
        values,
        "candidates in first 250 candidate rows",
        color=TEAL,
        notes="Counts are sampled from the canonical SSTK candidate JSONL to show AR-conditioned bank expansion.",
    )


def fig_candidate_roles(path: Path, ctx: dict[str, Any]) -> None:
    row = choose_candidate_role_row(ctx)
    img = Image.new("RGB", (1600, 930), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Candidate role split in a training row",
        "Matching targets become positive anchors; pool/ignored/overflow candidates preserve contrast, ambiguity, and hard audit cases.",
    )
    if not row:
        draw_wrapped(d, (80, 180), "No batch row was available.", 1400, fill=RED, size=30, bold=True)
        save_pil(img, path)
        return
    image_path = resolve_image_path(row)
    if image_path:
        thumb, local_area = open_image_with_bbox(image_path, (660, 520))
    else:
        thumb = Image.new("RGB", (660, 520), "#EFEAE1")
        local_area = (0, 0, thumb.width, thumb.height)
    img.paste(thumb, (72, 180))
    area = offset_area(local_area, (72, 180))
    draw_norm_box(d, row.get("baseline", {}).get("bbox_norm_xyxy"), area, "#7A8696", "", width=4)
    targets = row.get("matching_targets") or []
    if targets:
        draw_norm_box(d, targets[0].get("bbox_norm_xyxy"), area, GREEN, "", width=5)
    if len(targets) > 1:
        draw_norm_box(d, targets[1].get("bbox_norm_xyxy"), area, TEAL, "", width=4)
    draw_norm_box(d, row.get("decision_target", {}).get("winner_post_gate_bbox"), area, ORANGE, "", width=5)
    legend = [("baseline", "#7A8696"), ("positive", GREEN), ("alt+", TEAL), ("winner", ORANGE)]
    lx = 78
    for label, color in legend:
        d.line((lx, 724, lx + 32, 724), fill=color, width=5)
        d.text((lx + 42, 714), label, font=font(18), fill=INK)
        lx += 150

    roles = [
        ("matching_targets", GREEN, "direct positive supervision"),
        ("candidate_pool", BLUE, "safe negatives / soft positives"),
        ("ignored_candidates", GOLD, "ambiguous candidates kept for audit"),
        ("overflow_candidates", RED, "unsafe or hard examples for mining"),
    ]
    y = 198
    for name, color, desc in roles:
        value = len(row.get(name) or [])
        d.rounded_rectangle((815, y, 1490, y + 120), radius=18, fill=WHITE, outline=GRID, width=2)
        draw_tag(d, (840, y + 26), name, color, size=20)
        d.text((1275, y + 22), f"{value:,}", font=font(38, bold=True), fill=INK)
        d.text((840, y + 76), desc, font=font(22), fill=MUTED)
        y += 142
    save_pil(img, path)


def fig_teacher_disagreement(path: Path, ctx: dict[str, Any]) -> None:
    export = ctx["public_export"].get("splits", {}).get("train", {}).get("raw_label_summary", {}).get("label_summary", {})
    metric_note = (
        f"train contradiction_rate={export.get('contradiction_rate', 0):.4f}, "
        f"fatal_rate={export.get('fatal_rate', 0):.4f}, "
        f"candidate_labels={export.get('rows', 0):,}"
    )
    draw_individual_visual_panels(
        path,
        "Public score versus SSTK safety semantics",
        "Representative individual overlays are selected from tier artifacts; concat contact sheets are not embedded.",
        [
            (
                "main_positive",
                first_png("artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/tier_viz/val/main_positive/overlays"),
                "Public score and SSTK semantics agree. This tier can become a strong positive after local grouping.",
                GREEN,
                0.54,
            ),
            (
                "score_conflict",
                first_png("artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/tier_viz/val/score_explanation_conflict/overlays"),
                "The external score is useful, but explanation or checklist disagreement prevents direct positive import.",
                RED,
                0.54,
            ),
            (
                "fatal_public+",
                first_png("artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/tier_viz/val/gaic_public_positive_sstk_fatal/overlays"),
                f"Fatal or unsafe candidates are demoted/capped. {metric_note}",
                RED,
                0.47,
            ),
        ],
    )


def fig_safe_conversion(path: Path, ctx: dict[str, Any]) -> None:
    export = ctx["public_export"].get("splits", {}).get("train", {}).get("raw_label_summary", {}).get("label_summary", {})
    distill = ctx["public_distill"].get("splits", {}).get("train", {})
    labels = ["raw candidates", "safe train pairs", "distill train pairs"]
    values = [
        float(export.get("rows", 0)),
        float(ctx["manifest"].get("gaic_conversion", {}).get("T6 corrected_v2b", {}).get("Train", {}).get("pairwise_rows", 0)),
        float(distill.get("pairwise_rows", 0)),
    ]
    plot_bar(
        path,
        "Safe conversion: raw public score is filtered before supervision",
        labels,
        values,
        "rows",
        color=RED,
        notes=(
            f"unsafe_score_cap={distill.get('unsafe_score_cap', 0.05)}; "
            f"raw_score_blend_weight={ctx['public_distill'].get('raw_score_blend_weight', 0.35)}."
        ),
    )


def fig_uctr_stage3_deep_ranker_flow(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1100), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "UCTR Stage3 deep-ranker scoring versus T1 native SSTK scoring",
        "T1 preserves product semantics; UCTR Stage3 imports a learned crop-ranking utility and must pass safe conversion before student supervision.",
    )
    left_x, right_x = 90, 875
    y0 = 175
    t1_cards = [
        ("C1-C7 SSTK evidence", "subject, route, checklist, safety, action and support annotations", BLUE),
        ("T1 native scorer", "hand/semantic product scoring with reject tags and decision type", TEAL),
        ("Native labels", "crop/minimal/full action, checklist, route, rationale, provenance", GREEN),
    ]
    uctr_cards = [
        ("Public crop data", "FCDB/CPC/GNMC geometry plus GAIC ranking or MOS supervision", PURPLE),
        ("Crop-aware encoder", "image crop, context, AR geometry and candidate features", ORANGE),
        ("Stage3 ranker head", "deep utility score trained with pair/list/ranking losses", RED),
    ]
    for i, (title, body, color) in enumerate(t1_cards):
        y = y0 + i * 155
        box_card(d, (left_x, y, left_x + 600, y + 118), title, body, accent=color)
        if i < len(t1_cards) - 1:
            d.line((left_x + 300, y + 118, left_x + 300, y + 148), fill=MUTED, width=4)
            d.polygon([(left_x + 300, y + 158), (left_x + 288, y + 138), (left_x + 312, y + 138)], fill=MUTED)
    for i, (title, body, color) in enumerate(uctr_cards):
        y = y0 + i * 155
        box_card(d, (right_x, y, right_x + 600, y + 118), title, body, accent=color)
        if i < len(uctr_cards) - 1:
            d.line((right_x + 300, y + 118, right_x + 300, y + 148), fill=MUTED, width=4)
            d.polygon([(right_x + 300, y + 158), (right_x + 288, y + 138), (right_x + 312, y + 138)], fill=MUTED)

    d.rounded_rectangle((410, 690, 1190, 820), radius=24, fill=WHITE, outline=GRID, width=2)
    draw_tag(d, (455, 724), "Safe Conversion", RED, size=24)
    draw_wrapped(
        d,
        (705, 714),
        "local grouping, fatal reject, contradiction cap, safe-positive selection, fallback to native SSTK when needed",
        430,
        fill=INK,
        size=22,
    )
    d.line((390, 630, 585, 690), fill=MUTED, width=4)
    d.polygon([(410, 636), (390, 630), (402, 655)], fill=MUTED)
    d.line((1175, 630, 1015, 690), fill=MUTED, width=4)
    d.polygon([(1157, 637), (1175, 630), (1163, 655)], fill=MUTED)

    output_cards = [
        ("Student score target", "external utility normalized only inside the image-targetAR candidate group", ORANGE),
        ("Product semantics", "route, action, support, checklist and reject fields remain separate targets", GREEN),
        ("Auditability", "teacher lineage records whether score came from T1, UCTR, public ensemble, or fallback", BLUE),
    ]
    x = 90
    for title, body, color in output_cards:
        box_card(d, (x, 885, x + 455, 1010), title, body, accent=color)
        x += 520
    save_pil(img, path)


def fig_training_row_contract(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 980), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Training row contract",
        "A batch row defines an image-targetAR task; sidecars densify local ranking without erasing route, safety, and action labels.",
    )
    columns = [
        ("Batch JSONL", ["image_id", "target_ar", "routing", "decision_target", "matching_targets"], BLUE),
        ("Candidate Roles", ["baseline", "candidate_pool", "ignored", "overflow", "provenance"], TEAL),
        ("Sidecars", ["listwise", "pairwise", "checklist", "regression", "decision"], ORANGE),
        ("Student Targets", ["utility", "proposal", "route", "subject box", "risk/rationale"], GREEN),
    ]
    x = 70
    for title, fields, color in columns:
        d.rounded_rectangle((x, 190, x + 330, 760), radius=26, fill=WHITE, outline=GRID, width=2)
        draw_tag(d, (x + 30, 225), title, color, size=22)
        y = 310
        for field in fields:
            d.rounded_rectangle((x + 35, y, x + 295, y + 62), radius=14, fill="#F6F8FA", outline="#E2E6EC", width=1)
            d.text((x + 55, y + 17), field, font=font(24), fill=INK)
            y += 84
        if x < 1120:
            d.line((x + 330, 475, x + 410, 475), fill=MUTED, width=4)
            d.polygon([(x + 430, 475), (x + 405, 462), (x + 405, 488)], fill=MUTED)
        x += 390
    counts = ctx["manifest"].get("sstk_main_batch", {}).get("counts", {})
    d.rounded_rectangle((120, 820, 1480, 910), radius=18, fill=WHITE, outline=GRID, width=2)
    d.text(
        (150, 847),
        (
            f"Canonical materialization: batch={counts.get('conditional_detr_batch', 0):,}, "
            f"listwise={counts.get('listwise', 0):,}, pairwise={counts.get('pairwise', 0):,}, "
            f"checklist/regression candidates={counts.get('checklist', 0):,}"
        ),
        font=font(27, bold=True),
        fill=INK,
    )
    save_pil(img, path)


def fig_batch_sidecar_join(path: Path, ctx: dict[str, Any]) -> None:
    counts = ctx["manifest"].get("sstk_main_batch", {}).get("counts", {})
    labels = ["batch", "listwise", "pairwise", "decision", "checklist"]
    values = [
        counts.get("conditional_detr_batch", 0),
        counts.get("listwise", 0),
        counts.get("pairwise", 0),
        counts.get("decision", 0),
        counts.get("checklist", 0),
    ]
    plot_bar(
        path,
        "Batch and sidecar join scale",
        labels,
        values,
        "rows / candidate targets",
        color=BLUE,
        notes="Dataset joins listwise and pairwise labels by (image_id, target_ar, candidate_id) when candidate slots are sampled.",
    )


def fig_product_ar_status(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 900), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Product-AR branch materialization status",
        "Smoke/fallback manifests and full SSTK Product-AR materialization must be interpreted separately.",
    )
    rows = [
        ("GAIC T1 local", "1 converted row / split", "smoke or fallback", RED),
        ("GAIC public ensemble local", "1 converted row + 10 pairs / split", "CUDA/fusion smoke", GOLD),
        ("SSTK UCTR stage3 local", "1 converted row + 10 pairs", "UCTR merge smoke", GOLD),
        ("SSTK public ensemble full", "48,766 rows, 671,635 pairs", "full label line", GREEN),
    ]
    y = 185
    for name, scale, status, color in rows:
        d.rounded_rectangle((90, y, 1510, y + 130), radius=22, fill=WHITE, outline=GRID, width=2)
        draw_tag(d, (125, y + 35), status, color, size=22)
        d.text((485, y + 28), name, font=font(30, bold=True), fill=INK)
        d.text((485, y + 72), scale, font=font(24), fill=MUTED)
        y += 155
    save_pil(img, path)


def fig_decision_distribution(path: Path, ctx: dict[str, Any]) -> None:
    counts = ctx["manifest"].get("sstk_main_batch", {}).get("decision_counts", {})
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    plot_bar(
        path,
        "SSTK decision/action distribution",
        labels,
        values,
        "decision rows",
        color=PURPLE,
        notes="The skew toward minimal_crop is product-relevant; all-crop Product-AR lines under-specify the action executor.",
    )


def fig_runtime_no_prior(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 940), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "No-prior runtime contract",
        "Teacher subject/support signals are training supervision, not inference inputs.",
    )
    left = [
        ("Training", "image + target_ar + teacher subject/support + route/checklist labels", GREEN),
        ("Supervision", "subject_box_target, route, action, risk, rationale", TEAL),
        ("Inference", "image + target_ar only", BLUE),
        ("Executor", "keep_full -> full baseline; minimal_crop -> AR baseline; crop -> selected proposal", ORANGE),
    ]
    x = 130
    for i, (title, body, color) in enumerate(left):
        y = 185 + i * 155
        box_card(d, (x, y, 645, y + 110), title, body, accent=color)
        if i < len(left) - 1:
            d.line((387, y + 110, 387, y + 145), fill=MUTED, width=4)
            d.polygon([(387, y + 155), (375, y + 135), (399, y + 135)], fill=MUTED)
    row = (support_review_rows(ctx, 1) or ctx["batch_rows"] or [None])[0]
    if row:
        image_path = resolve_image_path(row)
        if image_path:
            thumb, local_area = open_image_with_bbox(image_path, (720, 520))
        else:
            thumb = Image.new("RGB", (720, 520), "#EFEAE1")
            local_area = (0, 0, thumb.width, thumb.height)
        img.paste(thumb, (790, 225))
        area = offset_area(local_area, (790, 225))
        draw_support_heatmap(img, row, area, alpha_max=96)
        support = support_bbox_for_render(row)
        winner = row.get("decision_target", {}).get("winner_post_gate_bbox")
        draw_norm_box(d, support, area, GREEN, "", width=5)
        draw_norm_box(d, winner, area, ORANGE, "", width=5)
        legend_y = 758
        d.line((795, legend_y + 8, 830, legend_y + 8), fill=GREEN, width=5)
        d.text((840, legend_y), "teacher support", font=font(18), fill=INK)
        d.line((1045, legend_y + 8, 1080, legend_y + 8), fill=ORANGE, width=5)
        d.text((1090, legend_y), "runtime crop", font=font(18), fill=INK)
        draw_wrapped(
            d,
            (795, 805),
            "Training learns the hidden subject structure; runtime receives no external prior.",
            700,
            fill=INK,
            size=24,
            bold=True,
        )
    save_pil(img, path)


def fig_subject_box_no_prior(path: Path, ctx: dict[str, Any]) -> None:
    rows = ctx["subject_report"].get("runs") or ctx["subject_report"].get("rows") or []
    values = []
    labels = []
    for row in rows[:8]:
        label = row.get("profile") or row.get("run_name") or row.get("track_name") or f"run{len(labels)+1}"
        val = row.get("direct_subject_box_iou_to_teacher") or row.get("direct_subj_iou") or row.get("best_val_subject_box_iou")
        if val is not None:
            labels.append(str(label)[:22])
            values.append(float(val))
    if not labels:
        labels = ["GAIC plus_384", "GAIC rank_320", "SSTK plus_384", "SSTK rank_320"]
        values = [0.331153, 0.309458, 0.522159, 0.516106]
    plot_bar(
        path,
        "Subject-box supervision under no-prior deployment",
        labels,
        values,
        "subject IoU to teacher",
        color=GREEN,
        horizontal=True,
        notes="Subject-box improvement is promising, but does not by itself close route-collapse release gates.",
    )


def fig_quant_sstk_scale(path: Path, ctx: dict[str, Any]) -> None:
    counts = ctx["manifest"].get("sstk_main_batch", {}).get("counts", {})
    labels = ["batch", "pairwise", "listwise", "decision", "checklist"]
    values = [
        counts.get("conditional_detr_batch", 0),
        counts.get("pairwise", 0),
        counts.get("listwise", 0),
        counts.get("decision", 0),
        counts.get("checklist", 0),
    ]
    plot_bar(path, "SSTK factory supervision scale", labels, values, "rows", color=BLUE)


def fig_quant_gaic_lineage(path: Path, ctx: dict[str, Any]) -> None:
    gaic = ctx["manifest"].get("gaic_conversion", {})
    labels: list[str] = []
    means: list[float] = []
    for lineage, split_map in gaic.items():
        train = split_map.get("Train", {})
        labels.append(lineage.replace(" ", "\n"))
        means.append(float(train.get("candidate_count_mean", 0)))
    plot_bar(
        path,
        "GAIC free-form versus Product-AR candidate richness",
        labels,
        means,
        "mean candidates per train task",
        color=TEAL,
        notes="Corrected_v2b preserves broad free-form ranking; Product-AR narrows the candidate bank for deployment-style AR tasks.",
    )


def fig_quant_route(path: Path, ctx: dict[str, Any]) -> None:
    shortlist = ctx["manifest"].get("student_shortlist", {})
    labels = ["entries", "completed", "route collapse", "final gate pass"]
    values = [
        shortlist.get("entry_count", 0),
        shortlist.get("completed_entry_count", 0),
        shortlist.get("route_collapse_count", 0),
        shortlist.get("final_gate_pass_count", 0),
    ]
    plot_bar(path, "Student shortlist gate diagnostics", labels, values, "count", color=RED)


def fig_quant_subject_box(path: Path, ctx: dict[str, Any]) -> None:
    labels = ["GAIC +384", "GAIC r320", "SSTK +384", "SSTK r320"]
    values = [0.280450, 0.285814, 0.477671, 0.473988]
    direct = [0.331153, 0.309458, 0.522159, 0.516106]
    fig, ax = plt.subplots(figsize=(12, 7), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(WHITE)
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], values, width, label="Val IoU", color=GREEN, edgecolor=INK)
    ax.bar([i + width / 2 for i in x], direct, width, label="Direct IoU", color=BLUE, edgecolor=INK)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("IoU")
    ax.set_title("Subject-box v2 balanced-valid evidence", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def short_method_name(name: str) -> str:
    mapping = {
        "production_final_hybrid": "prod final hybrid",
        "universal_crop_teacher_h_stage3_gate128": "UCTR-H stage3",
        "universal_crop_teacher_h_stage2_fullhonest": "UCTR-H stage2",
        "public_cropper_ensemble_best": "public ensemble",
        "public_cropper_gaic": "public GAIC",
        "public_cropper_cgs": "public CGS",
        "teacher_proxy_compact_utility_pool": "teacher proxy",
        "sstk_teacher_t1_historical_gaic_plus_public_proxy": "SSTK T1 proxy",
    }
    return mapping.get(name, name.replace("_", " "))


def equal4_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = ctx.get("equal4_teacher_leaderboard", {}).get("rows") or []
    return sorted(rows, key=lambda row: float(row.get("rank_equal4_zscore", 999)))


SELECTED_UNIFIED_METHOD_NAMES = {
    "universal_crop_teacher_h_stage3_gate128": "Universal Crop Teacher-H\nStage3 Deep Ranker",
    "public_cropper_ensemble_best": "Public Cropper Ensemble\nGAIC + CGS",
    "public_cropper_gaic": "GAIC Public Cropper\nMOS Ranker",
    "public_cropper_cgs": "CGS Public Cropper\nComposition Ranker",
    "teacher_proxy_compact_utility_pool": "Compact Teacher Proxy\nUtility Pool",
}


def selected_unified_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    selected = set(SELECTED_UNIFIED_METHOD_NAMES)
    rows = [row for row in equal4_rows(ctx) if str(row.get("method")) in selected]
    return sorted(rows, key=lambda row: float(row.get("equal4_raw_mean", 0.0)), reverse=True)


def fig_unified_equal4_leaderboard(path: Path, ctx: dict[str, Any]) -> None:
    rows = equal4_rows(ctx)
    if not rows:
        plot_bar(path, "Unified public benchmark equal-4 leaderboard", ["missing"], [0.0], "score", color=RED)
        return
    labels = [short_method_name(str(row.get("method"))) for row in rows][::-1]
    z_values = [float(row.get("equal4_zscore", 0.0)) for row in rows][::-1]
    worst_values = [float(row.get("worst_dataset_z", 0.0)) for row in rows][::-1]
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(13, 8), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(WHITE)
    ax.barh([v + 0.18 for v in y], z_values, height=0.34, color=BLUE, edgecolor=INK, label="equal-4 z")
    ax.barh([v - 0.18 for v in y], worst_values, height=0.34, color=RED, edgecolor=INK, label="worst dataset z")
    ax.axvline(0, color=INK, linewidth=1.1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("z-score across compared teacher/public-cropper methods")
    ax.set_title("Unified public benchmark: teacher/public cropper equal-4 comparison", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    for spine in ax.spines.values():
        spine.set_color("#B7BEC9")
    fig.text(
        0.04,
        0.03,
        "Equal-4 averages FCDB IoU, CPC weighted pairwise, GNMC IoU, and GAIC primary with equal dataset weight; worst-dataset z exposes hidden collapse.",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_unified_equal4_raw_bar(path: Path, ctx: dict[str, Any]) -> None:
    rows = equal4_rows(ctx)
    if not rows:
        plot_bar(path, "Unified public benchmark equal-4 raw mean", ["missing"], [0.0], "score", color=RED)
        return
    labels = [short_method_name(str(row.get("method"))) for row in rows][::-1]
    values = [float(row.get("equal4_raw_mean", 0.0)) for row in rows][::-1]
    colors = []
    for row in rows[::-1]:
        family = str(row.get("family", ""))
        if "hybrid" in family:
            colors.append(BLUE)
        elif "uctr" in family:
            colors.append(TEAL)
        elif "public_cropper" in family:
            colors.append(ORANGE)
        else:
            colors.append(PURPLE)
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(13, 8), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(WHITE)
    ax.barh(y, values, height=0.58, color=colors, edgecolor=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 0.95)
    ax.set_xlabel("equal-4 raw mean")
    ax.set_title("Unified public benchmark: equal-4 raw mean comparison", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for yi, value in zip(y, values):
        ax.text(value + 0.008, yi, f"{value:.6f}", va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#B7BEC9")
    fig.text(
        0.04,
        0.03,
        "Raw mean = average of FCDB IoU, CPC weighted pairwise, GNMC IoU, and GAIC primary without cross-method z-normalization.",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_unified_selected_methods(path: Path, ctx: dict[str, Any]) -> None:
    rows = selected_unified_rows(ctx)
    if not rows:
        plot_bar(path, "Selected unified public benchmark methods", ["missing"], [0.0], "score", color=RED)
        return
    labels = [SELECTED_UNIFIED_METHOD_NAMES[str(row.get("method"))] for row in rows]
    raw_values = [float(row.get("equal4_raw_mean", 0.0)) for row in rows]
    metrics = [
        ("FCDB\nIoU", "fcdb_iou_top1"),
        ("CPC\nweighted", "cpc_weighted_pairwise"),
        ("GNMC\nIoU", "gnmc_iou_top1"),
        ("GAIC\nprimary", "gaic_primary"),
    ]
    matrix = [[float(row.get(key, 0.0)) for _, key in metrics] for row in rows]

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(16, 8), dpi=170, gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.patch.set_facecolor(PAPER)
    for ax in (ax_bar, ax_heat):
        ax.set_facecolor(WHITE)

    y = list(range(len(labels)))
    colors = [TEAL, ORANGE, ORANGE, ORANGE, PURPLE]
    ax_bar.barh(y, raw_values, color=colors[: len(rows)], edgecolor=INK, height=0.58)
    ax_bar.set_yticks(y, labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, 0.9)
    ax_bar.set_xlabel("equal-4 raw mean")
    ax_bar.set_title("Overall equal-4 raw mean", loc="left", fontweight="bold")
    ax_bar.grid(axis="x", color=GRID, linewidth=0.8)
    for yi, value in zip(y, raw_values):
        ax_bar.text(value + 0.008, yi, f"{value:.6f}", va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold")

    im = ax_heat.imshow(matrix, cmap="YlGnBu", vmin=0.50, vmax=0.93, aspect="auto")
    ax_heat.set_xticks(range(len(metrics)), [name for name, _ in metrics])
    ax_heat.set_yticks(range(len(labels)), [""] * len(labels))
    ax_heat.set_title("Dataset-primary metrics", loc="left", fontweight="bold")
    for i, row_values in enumerate(matrix):
        for j, value in enumerate(row_values):
            ax_heat.text(j, i, f"{value:.3f}", ha="center", va="center", color=INK, fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("primary metric", rotation=270, labelpad=14)
    fig.suptitle("Selected Unified Public Benchmark Methods", x=0.06, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(
        0.06,
        0.035,
        "Names are expanded for readability: UCTR-H is a learned deep ranker, public croppers are external candidate rankers, and the compact proxy is a utility baseline.",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_unified_metric_matrix(path: Path, ctx: dict[str, Any]) -> None:
    rows = equal4_rows(ctx)
    if not rows:
        plot_bar(path, "Unified public benchmark metric matrix", ["missing"], [0.0], "score", color=RED)
        return
    metrics = [
        ("FCDB IoU", "fcdb_iou_top1"),
        ("CPC weighted", "cpc_weighted_pairwise"),
        ("GNMC IoU", "gnmc_iou_top1"),
        ("GAIC primary", "gaic_primary"),
    ]
    labels = [short_method_name(str(row.get("method"))) for row in rows]
    values = [[float(row.get(key, 0.0)) for _, key in metrics] for row in rows]
    fig, ax = plt.subplots(figsize=(13, 8), dpi=170)
    fig.patch.set_facecolor(PAPER)
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.50, vmax=0.93, aspect="auto")
    ax.set_xticks(range(len(metrics)), [name for name, _ in metrics])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Dataset-primary metric matrix for teacher/public cropper methods", loc="left", fontweight="bold")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=INK, fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.set_ylabel("primary metric value", rotation=270, labelpad=16)
    fig.text(
        0.04,
        0.03,
        "UCTR/production hybrid dominate FCDB/GNMC geometry; public croppers remain strongest around GAIC-style aesthetic ranking.",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_qualitative_failure_taxonomy(path: Path, ctx: dict[str, Any]) -> None:
    draw_individual_visual_panels(
        path,
        "Qualitative audit pack: success and failure families",
        "Each panel is one representative overlay, not a downscaled contact sheet.",
        [
            (
                "route/action",
                first_png(
                    "artifacts/mobilecropnet_v4/route_gate_recover_20260424/full_runs/sstk_public/"
                    "mcn-sstk-public-routefix-balanced-288-20260424/viz_test/overlays"
                ),
                "Route/action overlays expose whether crop, baseline, subject box, and route prediction agree.",
                BLUE,
            ),
            (
                "Product-AR",
                first_png("artifacts/mobilecropnet_v4/product_ar_best_20260420/arx2_turbo256_test_images/overlays"),
                "Product-AR examples expose action diversity and AR-conditioned proposal behavior.",
                ORANGE,
            ),
            (
                "subject-box",
                first_png("artifacts/mobilecropnet_v4/subject_box_head_smoke_local_20260423_1/viz_subject_box_smoke/overlays"),
                "Subject-box overlays separate localization error from ranking or action error.",
                GREEN,
            ),
        ],
    )


def fig_subject_box_fp_fn(path: Path, ctx: dict[str, Any]) -> None:
    render_row_grid(
        path,
        "Subject-support examples for FP/FN review",
        "Curated review rows only: heat=support mass, gray=baseline, orange=winner crop, green=support envelope, blue=subject prior.",
        support_review_rows(ctx, 6),
        box_plan="support",
    )


def draw_checklist_example_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    title: str,
    example: dict[str, Any] | None,
    color: str,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rounded_rectangle(xyxy, radius=18, fill=WHITE, outline=GRID, width=2)
    draw_tag(draw, (x1 + 22, y1 + 18), title, color, size=20)
    if not example:
        draw_wrapped(draw, (x1 + 25, y1 + 90), "No example was found in the scanned rows.", x2 - x1 - 50, fill=RED, size=22)
        return
    row = example["row"]
    target = example["target"]
    image_path = resolve_image_path(row)
    image_size = (x2 - x1 - 44, max(120, y2 - y1 - 180))
    thumb, local_area = open_image_with_bbox(image_path, image_size) if image_path else (Image.new("RGB", image_size, "#EFEAE1"), (0, 0, image_size[0], image_size[1]))
    canvas.paste(thumb, (x1 + 22, y1 + 70))
    area = offset_area(local_area, (x1 + 22, y1 + 70))
    support = support_visual_box(row)
    fill_norm_box(canvas, support, area, GREEN, alpha=52)
    draw_norm_box(draw, support, area, GREEN, "", width=4)
    draw_norm_box(draw, target.get("bbox_norm_xyxy"), area, ORANGE, "crop", width=5)
    if "lookroom" in title:
        draw_gaze_arrow(draw, row, area)
    value = example.get("value", "unknown")
    mode = row.get("routing", {}).get("subject_mode", "unknown")
    support_ratio = example.get("visual_support_in_crop_ratio", example.get("support_in_crop_ratio"))
    support_info = row.get("routing", {}).get("subject_support_overlay", {})
    repr_type = str(support_info.get("subject_repr_type") or "support_region")
    ratio_text = f"visual-support-in-crop={float(support_ratio):.2f}" if support_ratio is not None else "visual-support-in-crop=n/a"
    draw.text((x1 + 24, y2 - 80), f"{value}   mode={mode}", font=font(19, bold=True), fill=INK)
    draw.text((x1 + 24, y2 - 52), f"{ratio_text}   green=C7 support envelope ({repr_type})", font=font(16), fill=MUTED)
    draw.text((x1 + 24, y2 - 26), f"image_id={row.get('image_id')}", font=font(16), fill=MUTED)


def fig_subject_coverage_good_bad_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 820), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Subject coverage: good versus bad crop examples",
        "Orange is the evaluated crop candidate; green is the C7 subject-support envelope used to judge preservation.",
    )
    examples = ctx.get("label_examples", {})
    draw_checklist_example_panel(
        img,
        d,
        (80, 185, 760, 745),
        "subject coverage: good",
        examples.get("subject_coverage_good"),
        GREEN,
    )
    draw_checklist_example_panel(
        img,
        d,
        (840, 185, 1520, 745),
        "subject coverage: bad",
        examples.get("subject_coverage_bad"),
        RED,
    )
    save_pil(img, path)


def fig_headroom_lookroom_good_bad_examples(path: Path, ctx: dict[str, Any]) -> None:
    img = Image.new("RGB", (1600, 1280), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "Headroom and lookroom checklist examples",
        "Each panel shows one candidate crop and the C7 subject-support envelope used to score portrait framing quality.",
    )
    examples = ctx.get("label_examples", {})
    panels = [
        ((70, 175, 770, 705), "headroom: good", examples.get("headroom_good"), GREEN),
        ((830, 175, 1530, 705), "headroom: bad", examples.get("headroom_bad"), RED),
        ((70, 750, 770, 1280), "lookroom: good", examples.get("lookroom_good"), GREEN),
        ((830, 750, 1530, 1280), "lookroom: bad", examples.get("lookroom_bad"), RED),
    ]
    for box, title, example, color in panels:
        draw_checklist_example_panel(img, d, box, title, example, color)
    save_pil(img, path)


def fig_deployment_leaderboard(path: Path, ctx: dict[str, Any]) -> None:
    leaderboard = ctx["final_leaderboard"]
    winners = leaderboard.get("winners", {})
    labels = [
        "best crop z",
        "head sanity",
        "direct align",
        "prop recall",
        "risk hit",
    ]
    values = [
        winners.get("best_crop_zscore", 0),
        winners.get("best_head_sanity_score", 0),
        winners.get("best_direct_alignment_score", 0),
        winners.get("best_prop_target_recall", 0),
        winners.get("best_risk_hit_rate", 0),
    ]
    plot_bar(
        path,
        "Deployment leaderboard signal summary",
        labels,
        [float(v or 0) for v in values],
        "score",
        color=BLUE,
        notes="These are selection signals, not release approval. The strict release gate remains blocked.",
    )


def fig_final_gate_matrix(path: Path, ctx: dict[str, Any]) -> None:
    release = ctx["release_gate"]
    img = Image.new("RGB", (1600, 1040), PAPER)
    d = ImageDraw.Draw(img)
    decision = release.get("decision", {})
    draw_title(
        d,
        "Final release gate matrix",
        "A model is deployable only when public quality, product alignment, route, subject-box, risk, and target-AR gates pass together.",
    )
    state = decision.get("state", "BLOCKED")
    draw_tag(d, (80, 155), state, RED if state == "BLOCKED" else GREEN, size=30)
    blockers = decision.get("blockers", [])[:7]
    y = 250
    for blocker in blockers:
        d.rounded_rectangle((90, y, 1510, y + 72), radius=14, fill=WHITE, outline=GRID, width=2)
        draw_tag(d, (115, y + 16), "FAIL", RED, size=18)
        draw_wrapped(d, (220, y + 17), str(blocker), 1220, fill=INK, size=22)
        y += 94
    if not blockers:
        draw_wrapped(d, (100, 260), "No blocker details were found in the release manifest.", 1400, fill=MUTED, size=26)
    d.text(
        (95, 965),
        f"entry_count={release.get('entry_count', 0)}  completed={release.get('completed_entry_count', 0)}  final_gate_pass={release.get('final_gate_pass_count', 0)}",
        font=font(27, bold=True),
        fill=INK,
    )
    save_pil(img, path)


def draw_portrait_audit_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    title: str,
    row: dict[str, Any] | None,
    note: str,
    color: str,
    *,
    show_crop: bool,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rounded_rectangle(xyxy, radius=18, fill=WHITE, outline=GRID, width=2)
    draw_tag(draw, (x1 + 20, y1 + 18), title, color, size=19)
    if not row:
        draw_wrapped(draw, (x1 + 24, y1 + 90), "No curated portrait row was found.", x2 - x1 - 48, fill=RED, size=22)
        return
    image_path = resolve_image_path(row)
    image_size = (x2 - x1 - 40, 500)
    thumb, local_area = open_image_with_bbox(image_path, image_size) if image_path else (Image.new("RGB", image_size, "#EFEAE1"), (0, 0, image_size[0], image_size[1]))
    canvas.paste(thumb, (x1 + 20, y1 + 66))
    area = offset_area(local_area, (x1 + 20, y1 + 66))
    routing = row.get("routing", {})
    support = support_bbox_for_render(row)
    prior = routing.get("subject_prior_bbox_norm_xyxy")
    union = subject_union_norm(row)
    fill_norm_box(canvas, support, area, GREEN, alpha=34)
    draw_norm_box(draw, support, area, GREEN, "", width=4)
    draw_norm_box(draw, prior, area, PURPLE, "", width=4)
    draw_norm_box(draw, union, area, TEAL, "", width=3)
    legend = [("support", GREEN), ("C3 prior", PURPLE), ("subject", TEAL)]
    if show_crop:
        draw_norm_box(draw, row.get("baseline", {}).get("bbox_norm_xyxy"), area, "#7A8696", "", width=3)
        draw_norm_box(draw, row.get("decision_target", {}).get("winner_post_gate_bbox"), area, ORANGE, "", width=5)
        legend.extend([("base", "#7A8696"), ("crop", ORANGE)])
    lx = x1 + 24
    ly = y1 + 574
    for label, item_color in legend:
        item_w = int(draw.textlength(label, font=font(14))) + 34
        if lx + item_w > x2 - 20:
            lx = x1 + 24
            ly += 22
        draw.line((lx, ly + 8, lx + 20, ly + 8), fill=item_color, width=4)
        draw.text((lx + 25, ly), label, font=font(14), fill=INK)
        lx += item_w + 10
    mode = routing.get("subject_mode", "unknown")
    action = row.get("decision_target", {}).get("decision_type", "unknown")
    draw.text((x1 + 24, y1 + 620), f"image_id={row.get('image_id')}", font=font(16), fill=MUTED)
    draw.text((x1 + 24, y1 + 646), f"mode={mode}   target_ar={row.get('target_ar')}", font=font(16), fill=MUTED)
    draw.text((x1 + 24, y1 + 672), f"action={action}", font=font(16), fill=MUTED)
    draw_wrapped(draw, (x1 + 24, y1 + 702), note, x2 - x1 - 48, fill=INK, size=16)


def fig_c3_c6_portrait_audit(path: Path, ctx: dict[str, Any]) -> None:
    portrait_single = preferred_row(ctx, ["sstk_image_1242934243"])
    portrait_group = preferred_row(ctx, ["sstk_image_1276179112"])
    action_portrait = preferred_row(ctx, ["sstk_image_671450404"])
    img = Image.new("RGB", (1600, 1000), PAPER)
    d = ImageDraw.Draw(img)
    draw_title(
        d,
        "C3/C6 portrait-sensitive audit evidence",
        "Curated portrait rows replace arbitrary first-PNG overlays, so subject-mode and visual content agree.",
    )
    panels = [
        (
            (70, 175, 548, 940),
            "C3 person prior",
            portrait_single,
            "Purple prior and teal subject union are person-centered; this panel is the canonical portrait_single evidence.",
            PURPLE,
            False,
        ),
        (
            (562, 175, 1040, 940),
            "C6 group framing",
            portrait_group,
            "The route is portrait_group and the selected crop keeps the people-framing region instead of a generic object crop.",
            BLUE,
            True,
        ),
        (
            (1054, 175, 1532, 940),
            "action executor",
            action_portrait,
            "A clean portrait row shows how baseline, winner crop, support, and portrait prior are audited together.",
            ORANGE,
            True,
        ),
    ]
    for box, title, row, note, color, show_crop in panels:
        draw_portrait_audit_cell(img, d, box, title, row, note, color, show_crop=show_crop)
    save_pil(img, path)


def parse_norm_box(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def validate_box_overlays(ctx: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Validate boxes rendered by this script, including letterbox-aware mapping."""
    checks: list[dict[str, Any]] = []

    def add_check(
        figure: str,
        row: dict[str, Any],
        box_name: str,
        box: Any,
        display_size: tuple[int, int],
    ) -> None:
        values = parse_norm_box(box)
        image_path = resolve_image_path(row)
        if image_path:
            _, image_area = open_image_with_bbox(image_path, display_size)
        else:
            image_area = (0, 0, display_size[0], display_size[1])
        pixel_box = norm_box_to_pixels(values, image_area) if values is not None else None
        valid = (
            values is not None
            and -1e-6 <= values[0] < values[2] <= 1.0 + 1e-6
            and -1e-6 <= values[1] < values[3] <= 1.0 + 1e-6
            and pixel_box is not None
            and image_area[0] <= pixel_box[0] < pixel_box[2] <= image_area[2]
            and image_area[1] <= pixel_box[1] < pixel_box[3] <= image_area[3]
        )
        checks.append(
            {
                "figure": figure,
                "image_id": row.get("image_id"),
                "target_ar": row.get("target_ar"),
                "subject_mode": row.get("routing", {}).get("subject_mode"),
                "box_name": box_name,
                "norm_box": values,
                "display_size": list(display_size),
                "image_area_excluding_letterbox": list(image_area),
                "pixel_box": list(pixel_box) if pixel_box else None,
                "valid": bool(valid),
                "source_image": rel(image_path) if image_path else None,
            }
        )

    for dataset, example in (ctx.get("public_dataset_examples") or {}).items():
        if not isinstance(example, dict) or example.get("error"):
            continue
        row_proxy = {
            "image_id": example.get("image_id"),
            "image_path": example.get("image_path"),
            "routing": {"subject_mode": f"public_{dataset}"},
            "target_ar": None,
        }
        for i, box_record in enumerate(example.get("boxes") or []):
            add_check(
                "fig_public_dataset_annotation_examples.png",
                row_proxy,
                f"{dataset}_annotation_{i}",
                box_record.get("bbox"),
                (660, 275),
            )

    for example in ctx.get("public_cropper_examples") or []:
        row_proxy = {
            "image_id": example.get("image_id"),
            "image_path": example.get("image_path"),
            "routing": {"subject_mode": f"public_{example.get('dataset')}"},
            "target_ar": example.get("target_ar"),
        }
        for method in ("public_cropper_gaic", "public_cropper_cgs"):
            best = best_prediction_candidate(example.get(method) or {})
            if best:
                add_check(
                    "fig_public_cropper_inference_examples.png",
                    row_proxy,
                    f"{method}_top1",
                    best.get("bbox_xyxy_norm"),
                    (509, 198),
                )

    c7_support_rows = support_review_rows(ctx, 8)
    support_rows = support_review_rows(ctx, 6)
    joint_row = choose_joint_overlay_row(ctx)
    if joint_row:
        for box_name, box in [
            ("c2_subject_union", subject_union_norm(joint_row)),
            ("c3_subject_prior", joint_row.get("routing", {}).get("subject_prior_bbox_norm_xyxy")),
            ("c7_subject_support", support_bbox_for_render(joint_row)),
            ("c7_subject_support_core", support_core_box_for_render(joint_row)),
        ]:
            if box is not None:
                add_check("fig_c2_c3_c7_joint_overlay.png", joint_row, box_name, box, (980, 640))

    for row in preferred_unique_rows(
        list((ctx.get("curated_rows_by_id") or {}).values()) + list(ctx.get("rows_with_support_map") or []),
        PREFERRED_SUPPORT_MAP_IMAGE_IDS,
        4,
    ):
        add_check(
            "fig_subject_support_map_region_examples.png",
            row,
            "subject_support_overlay",
            support_bbox_for_render(row),
            (660, 245),
        )

    for row in c7_support_rows:
        add_check(
            "fig_c7_subject_support_examples.png",
            row,
            "baseline",
            row.get("baseline", {}).get("bbox_norm_xyxy"),
            (326, 230),
        )
        add_check(
            "fig_c7_subject_support_examples.png",
            row,
            "winner_post_gate",
            row.get("decision_target", {}).get("winner_post_gate_bbox"),
            (326, 230),
        )
        add_check(
            "fig_c7_subject_support_examples.png",
            row,
            "subject_support_overlay",
            support_bbox_for_render(row),
            (326, 230),
        )
        core = support_core_box_for_render(row)
        if core is not None:
            add_check("fig_c7_subject_support_examples.png", row, "subject_support_core", core, (326, 230))
        prior = row.get("routing", {}).get("subject_prior_bbox_norm_xyxy")
        if prior is not None:
            add_check("fig_c7_subject_support_examples.png", row, "subject_prior", prior, (326, 230))

    for row in preferred_unique_rows(
        list((ctx.get("curated_rows_by_id") or {}).values())
        + list(ctx.get("rows_for_guidance_only") or [])
        + list(ctx.get("rows_with_support_map") or []),
        PREFERRED_GUIDANCE_IMAGE_IDS,
        6,
    ):
        box = guidance_box_for_render(row)
        add_check("fig_subject_guidance_clean_examples.png", row, "guidance_region", box, (440, 220))

    if support_rows:
        row = support_rows[0]
        add_check(
            "fig_runtime_no_prior_action_executor_example.png",
            row,
            "subject_support_overlay",
            support_bbox_for_render(row),
            (720, 520),
        )
        add_check(
            "fig_runtime_no_prior_action_executor_example.png",
            row,
            "winner_post_gate",
            row.get("decision_target", {}).get("winner_post_gate_bbox"),
            (720, 520),
        )

    for row in support_rows:
        add_check(
            "fig_subject_box_fp_fn_examples.png",
            row,
            "baseline",
            row.get("baseline", {}).get("bbox_norm_xyxy"),
            (444, 250),
        )
        add_check(
            "fig_subject_box_fp_fn_examples.png",
            row,
            "winner_post_gate",
            row.get("decision_target", {}).get("winner_post_gate_bbox"),
            (444, 250),
        )
        add_check(
            "fig_subject_box_fp_fn_examples.png",
            row,
            "subject_support_overlay",
            support_bbox_for_render(row),
            (444, 250),
        )

    for row in preferred_unique_rows(curated_source_rows(ctx), PREFERRED_PORTRAIT_AUDIT_IMAGE_IDS, 3):
        for box_name, box in [
            ("subject_support", support_bbox_for_render(row)),
            ("subject_prior", row.get("routing", {}).get("subject_prior_bbox_norm_xyxy")),
            ("subject_union", subject_union_norm(row)),
            ("winner_post_gate", row.get("decision_target", {}).get("winner_post_gate_bbox")),
        ]:
            if box is not None:
                add_check("fig_c3_c6_portrait_failure_panel.png", row, box_name, box, (438, 500))

    label_examples = ctx.get("label_examples", {})
    for key, figure, display_size in [
        ("subject_coverage_good", "fig_subject_coverage_good_bad_examples.png", (636, 380)),
        ("subject_coverage_bad", "fig_subject_coverage_good_bad_examples.png", (636, 380)),
        ("headroom_good", "fig_headroom_lookroom_good_bad_examples.png", (656, 350)),
        ("headroom_bad", "fig_headroom_lookroom_good_bad_examples.png", (656, 350)),
        ("lookroom_good", "fig_headroom_lookroom_good_bad_examples.png", (656, 350)),
        ("lookroom_bad", "fig_headroom_lookroom_good_bad_examples.png", (656, 350)),
    ]:
        example = label_examples.get(key)
        if not example:
            continue
        row = example["row"]
        add_check(
            figure,
            row,
            key,
            example["target"].get("bbox_norm_xyxy"),
            display_size,
        )
        support = support_visual_box(row)
        if support is not None:
            add_check(figure, row, f"{key}_support", support, display_size)

    role_row = choose_candidate_role_row(ctx)
    if role_row:
        add_check(
            "fig_matching_candidate_pool_ignored_overflow.png",
            role_row,
            "baseline",
            role_row.get("baseline", {}).get("bbox_norm_xyxy"),
            (660, 520),
        )
        targets = role_row.get("matching_targets") or []
        for i, target in enumerate(targets[:2]):
            add_check(
                "fig_matching_candidate_pool_ignored_overflow.png",
                role_row,
                f"matching_target_{i}",
                target.get("bbox_norm_xyxy"),
                (660, 520),
            )
        add_check(
            "fig_matching_candidate_pool_ignored_overflow.png",
            role_row,
            "winner_post_gate",
            role_row.get("decision_target", {}).get("winner_post_gate_bbox"),
            (660, 520),
        )

    invalid = [record for record in checks if not record["valid"]]
    payload = {
        "status": "pass" if not invalid else "fail",
        "checked_box_count": len(checks),
        "invalid_box_count": len(invalid),
        "mapping_policy": "normalized boxes are mapped to the actual image rectangle after ImageOps.contain; letterbox padding is excluded",
        "checks": checks,
    }
    write_json(output_dir / "box_overlay_validation.json", payload)
    return payload


def build_figures(output_dir: Path, max_rows: int, write_manifest_flag: bool) -> list[dict[str, str]]:
    ensure_dir(output_dir)
    ctx = load_context(max_rows=max_rows)
    figures: list[tuple[str, str, str, Any]] = [
        (
            "fig_public_dataset_annotation_examples.png",
            "Public crop benchmark dataset annotation examples",
            "section 2",
            lambda p: fig_public_dataset_annotation_examples(p, ctx),
        ),
        (
            "fig_public_cropper_inference_examples.png",
            "Public cropper inference examples",
            "section 2",
            lambda p: fig_public_cropper_inference_examples(p, ctx),
        ),
        (
            "fig_c1_c7_perception_stack_panel.png",
            "C1-C7 perception stack and downstream label use",
            "section 5",
            lambda p: fig_c1_c7_perception_stack(p, ctx),
        ),
        (
            "fig_c2_c3_c7_joint_overlay.png",
            "C2/C3/C7 joint perception overlay",
            "section 5",
            lambda p: fig_c2_c3_c7_joint_overlay(p, ctx),
        ),
        (
            "fig_subject_support_map_region_examples.png",
            "Subject support-map region examples",
            "section 5",
            lambda p: fig_subject_support_map_region_examples(p, ctx),
        ),
        (
            "fig_c3_c6_portrait_failure_panel.png",
            "Portrait-sensitive C3/C6 audit evidence",
            "section 5",
            lambda p: fig_c3_c6_portrait_audit(p, ctx),
        ),
        (
            "fig_c7_subject_support_examples.png",
            "C7 subject-support overlay examples",
            "section 5",
            lambda p: fig_c7_subject_support_examples(p, ctx),
        ),
        (
            "fig_decoded_support_mass_only_examples.png",
            "Decoded support-mass only examples",
            "section 5",
            lambda p: fig_decoded_support_mass_only_examples(p, ctx),
        ),
        (
            "fig_subject_guidance_clean_examples.png",
            "Clean subject and crop-guidance examples",
            "section 5",
            lambda p: fig_subject_guidance_clean_examples(p, ctx),
        ),
        (
            "fig_route_subject_mode_gallery.png",
            "Subject-mode routing examples",
            "section 6",
            lambda p: render_row_grid(
                p,
                "Subject-mode routing examples",
                "Each panel shows one routed image-task with baseline and selected crop overlay.",
                rows_for_modes(
                    ctx,
                    [
                        "object_single",
                        "portrait_single",
                        "portrait_group",
                        "scene_general",
                        "object_multi",
                    ],
                    5,
                ),
                box_plan="route",
            ),
        ),
        (
            "fig_route_subject_mode_plain_gallery.png",
            "Subject-mode routing examples without crop-box overlays",
            "section 6",
            lambda p: render_row_grid(
                p,
                "Subject-mode routing examples without crop boxes",
                "Each panel shows the routed image-task only; no winner or baseline crop box is drawn.",
                rows_for_modes(
                    ctx,
                    [
                        "object_single",
                        "portrait_single",
                        "portrait_group",
                        "scene_general",
                        "object_multi",
                    ],
                    5,
                ),
                box_plan="plain",
            ),
        ),
        ("fig_route_collapse_examples.png", "Route supervision distribution and collapse diagnostic", "section 6", lambda p: fig_route_collapse(p, ctx)),
        ("fig_candidate_bank_by_ar.png", "Candidate bank by AR", "section 7", lambda p: fig_candidate_bank_by_ar(p, ctx)),
        ("fig_matching_candidate_pool_ignored_overflow.png", "Candidate role split in a row", "section 7", lambda p: fig_candidate_roles(p, ctx)),
        ("fig_teacher_score_disagreement_examples.png", "Teacher score disagreement examples", "section 8", lambda p: fig_teacher_disagreement(p, ctx)),
        ("fig_uctr_stage3_deep_ranker_flow.png", "UCTR stage3 deep-ranker scoring flow", "section 8", lambda p: fig_uctr_stage3_deep_ranker_flow(p, ctx)),
        ("fig_safe_conversion_demoted_candidates.png", "Safe conversion demotion evidence", "section 8", lambda p: fig_safe_conversion(p, ctx)),
        ("fig_training_row_contract_example.png", "Training row contract", "section 9", lambda p: fig_training_row_contract(p, ctx)),
        ("fig_batch_pairwise_listwise_join.png", "Batch-sidecar join scale", "section 9", lambda p: fig_batch_sidecar_join(p, ctx)),
        ("fig_product_ar_branch_materialization_status.png", "Product-AR materialization status", "section 10", lambda p: fig_product_ar_status(p, ctx)),
        ("fig_decision_distribution_sstk_vs_product_ar.png", "Decision/action distribution", "section 10", lambda p: fig_decision_distribution(p, ctx)),
        ("fig_runtime_no_prior_action_executor_example.png", "No-prior runtime action executor", "section 12", lambda p: fig_runtime_no_prior(p, ctx)),
        ("fig_runtime_pred_subject_box_without_prior.png", "Subject-box supervision under no-prior deployment", "section 12", lambda p: fig_subject_box_no_prior(p, ctx)),
        ("fig_quant_sstk_factory_scale.png", "SSTK factory supervision scale", "section 13", lambda p: fig_quant_sstk_scale(p, ctx)),
        ("fig_quant_gaic_label_lineage_comparison.png", "GAIC label lineage comparison", "section 13", lambda p: fig_quant_gaic_lineage(p, ctx)),
        ("fig_unified_equal4_teacher_cropper_leaderboard.png", "Unified public benchmark equal-4 teacher/public cropper leaderboard", "section 13", lambda p: fig_unified_equal4_leaderboard(p, ctx)),
        ("fig_unified_equal4_raw_mean_bar.png", "Unified public benchmark equal-4 raw mean comparison", "section 13", lambda p: fig_unified_equal4_raw_bar(p, ctx)),
        ("fig_unified_public_benchmark_metric_matrix.png", "Unified public benchmark teacher/public cropper metric matrix", "section 13", lambda p: fig_unified_metric_matrix(p, ctx)),
        ("fig_unified_selected_methods_comparison.png", "Selected Unified Public Benchmark teacher/public cropper comparison", "section 13", lambda p: fig_unified_selected_methods(p, ctx)),
        ("fig_quant_route_collapse_diagnostics.png", "Route-collapse quantitative diagnostic", "section 13", lambda p: fig_quant_route(p, ctx)),
        ("fig_quant_subject_box_v1_v2.png", "Subject-box quantitative comparison", "section 13", lambda p: fig_quant_subject_box(p, ctx)),
        ("fig_qualitative_failure_taxonomy_panel.png", "Qualitative failure taxonomy panel", "section 14", lambda p: fig_qualitative_failure_taxonomy(p, ctx)),
        ("fig_subject_box_fp_fn_examples.png", "Subject-box FP/FN review examples", "section 14", lambda p: fig_subject_box_fp_fn(p, ctx)),
        ("fig_subject_coverage_good_bad_examples.png", "Subject coverage good/bad examples", "section 14", lambda p: fig_subject_coverage_good_bad_examples(p, ctx)),
        ("fig_headroom_lookroom_good_bad_examples.png", "Headroom/lookroom good/bad examples", "section 14", lambda p: fig_headroom_lookroom_good_bad_examples(p, ctx)),
        ("fig_deployment_leaderboard_summary.png", "Deployment leaderboard summary", "section 15", lambda p: fig_deployment_leaderboard(p, ctx)),
        ("fig_final_gate_pass_fail_matrix.png", "Final gate pass/fail matrix", "section 15", lambda p: fig_final_gate_matrix(p, ctx)),
    ]

    entries: list[dict[str, str]] = []
    for filename, caption, section, builder in figures:
        path = output_dir / filename
        builder(path)
        entries.append({"filename": filename, "path": rel(path), "caption": caption, "section": section})

    box_validation = validate_box_overlays(ctx, output_dir)
    if write_manifest_flag:
        manifest_path = output_dir / "figure_manifest.json"
        manifest = read_json(manifest_path, {})
        manifest["section_visuals"] = entries
        manifest["box_overlay_validation"] = {
            "path": rel(output_dir / "box_overlay_validation.json"),
            "status": box_validation["status"],
            "checked_box_count": box_validation["checked_box_count"],
            "invalid_box_count": box_validation["invalid_box_count"],
            "mapping_policy": box_validation["mapping_policy"],
        }
        manifest["section_visual_source_paths"] = {
            "batch_jsonl": rel(BATCH_JSONL),
            "candidate_jsonl": rel(CANDIDATE_JSONL),
            "public_score_export": "artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/public_score_label_export_summary.json",
            "public_score_distill": "artifacts/mobilecropnet_v4/public_score_sstk_explain_260420/public_score_distill_v2_label_convert_summary.json",
            "release_gate": "artifacts/mobilecropnet_v4/unified_recovery_20260424/release_gate_base_strict_check/release_gate_manifest.json",
            "final_leaderboard": "artifacts/mobilecropnet_v4/unified_recovery_20260424/final_deployment_leaderboard_base_strict_check/final_deployment_leaderboard.json",
        }
        write_json(manifest_path, manifest)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=800)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = build_figures(args.output_dir, max_rows=args.max_rows, write_manifest_flag=args.write_manifest)
    print(json.dumps({"output_dir": rel(args.output_dir), "figure_count": len(entries), "figures": entries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
