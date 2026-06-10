#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


MODE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "landscape": (31, 119, 180),
    "scene_center": (31, 119, 180),
    "scene_rot": (70, 130, 180),
    "single_person_center": (44, 160, 44),
    "single_person_rot": (0, 128, 128),
    "group_center": (214, 39, 40),
    "group_rot": (255, 80, 80),
    "face": (255, 193, 7),
    "object_single_center": (255, 127, 14),
    "object_single_rot": (255, 165, 80),
    "object_multi_center": (140, 86, 75),
    "object_multi_rot": (188, 189, 34),
    "person_single_face_headshot_center": (255, 99, 132),
    "person_single_face_headshot_rot": (255, 140, 160),
    "person_single_upper_half_body_center": (46, 204, 113),
    "person_single_upper_half_body_rot": (0, 166, 90),
    "person_single_full_body_center": (155, 89, 182),
    "person_single_full_body_rot": (171, 106, 210),
    "person_group_center": (214, 39, 120),
    "person_group_rot": (174, 68, 143),
    "pet_dogcat_center": (241, 196, 15),
    "pet_dogcat_rot": (230, 126, 34),
}
ROUTE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "scene": (64, 132, 178),
    "person_single": (34, 156, 100),
    "person_group": (132, 92, 190),
    "pet_dogcat": (224, 126, 34),
}

MODE_PRIORITY: Dict[str, int] = {
    "face": 0,
    "single_person_center": 1,
    "single_person_rot": 2,
    "group_center": 3,
    "group_rot": 4,
    "object_single_center": 5,
    "object_single_rot": 6,
    "object_multi_center": 7,
    "object_multi_rot": 8,
    "landscape": 9,
}

AR_PRIORITY: Dict[str, int] = {
    "FREE": 0,
    "1:1": 1,
    "9:16": 2,
    "16:9": 3,
    "3:4": 4,
    "4:3": 5,
}

SUBJECT_CORE_COLOR = (0, 220, 220)
SUBJECT_SUPPORT_COLOR = (255, 170, 0)
SUBJECT_MEMBER_COLOR = (255, 90, 200)
PANEL_BG = (244, 244, 244)
CARD_BG = (255, 255, 255)
MUTED = (90, 90, 90)


def _route_color(route_name: str, family: str = "") -> Tuple[int, int, int]:
    text = str(route_name or family or "scene")
    for key, color in ROUTE_COLORS.items():
        if text == key or text.startswith(f"{key}_"):
            return color
    return (80, 100, 120)


def _mode_color(mode_name: str) -> Tuple[int, int, int]:
    mode = str(mode_name or "")
    if mode in MODE_COLORS:
        return MODE_COLORS[mode]
    if mode.startswith("scene_"):
        return ROUTE_COLORS["scene"]
    if mode.startswith("person_single_face"):
        return (20, 184, 166)
    if mode.startswith("person_single_upper"):
        return (52, 152, 219)
    if mode.startswith("person_single_full"):
        return (46, 180, 105)
    if mode.startswith("person_group"):
        return ROUTE_COLORS["person_group"]
    if mode.startswith("pet_dogcat"):
        return ROUTE_COLORS["pet_dogcat"]
    return (120, 120, 120)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_by_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    max_lines: int,
) -> List[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    words: List[str] = []
    for chunk in text.split(" "):
        if _text_size(draw, chunk, font)[0] <= max_width:
            words.append(chunk)
            continue
        approx = max(4, int(len(chunk) * max_width / max(1, _text_size(draw, chunk, font)[0])))
        words.extend(textwrap.wrap(chunk, width=approx, break_long_words=True, break_on_hyphens=False))

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and words:
        joined = " ".join(words)
        if " ".join(lines) != joined:
            line = lines[-1]
            while line and _text_size(draw, line + "...", font)[0] > max_width:
                line = line[:-1]
            lines[-1] = (line.rstrip() + "...") if line else "..."
    return lines[:max_lines]


def _draw_text_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    xy: Tuple[int, int],
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    *,
    line_gap: int = 3,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        _, h = _text_size(draw, line, font)
        y += h + line_gap
    return y


def _source_id(image_row: Dict[str, Any]) -> str:
    return str(image_row.get("source_image_id") or Path(str(image_row.get("file_name") or "")).stem)


def _resolve_image_path(image_root: Path, image_row: Dict[str, Any]) -> Path:
    file_name = str(image_row.get("file_name") or "")
    if file_name:
        direct = image_root / file_name
        if direct.is_file():
            return direct
    source_id = _source_id(image_row)
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = image_root / f"{source_id}{suffix}"
        if candidate.is_file():
            return candidate
    return image_root / file_name


def _norm_box(box: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    vals = [_safe_float(value, float("nan")) for value in box]
    if not all(math.isfinite(value) for value in vals):
        return None
    vals = [max(0.0, min(1.0, value)) for value in vals]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return (vals[0], vals[1], vals[2], vals[3])


def _dedupe_norm_boxes(rows: Iterable[Tuple[str, Tuple[float, float, float, float]]]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    seen: set[Tuple[str, Tuple[int, int, int, int]]] = set()
    out: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for label, box in rows:
        rounded = tuple(int(round(value * 1000)) for value in box)
        key = (label, rounded)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, box))
    return out


def _subject_boxes(anns: Sequence[Dict[str, Any]]) -> Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]]:
    core_rows: List[Tuple[str, Tuple[float, float, float, float]]] = []
    support_rows: List[Tuple[str, Tuple[float, float, float, float]]] = []
    member_rows: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for ann in anns:
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        debug = attrs.get("subject_debug") if isinstance(attrs.get("subject_debug"), dict) else {}
        mode = str(ann.get("mode_name") or "unknown")
        entity = str(ann.get("entity_id") or "")
        label = f"{mode}:{entity}".rstrip(":")
        for key in ("core_bbox_norm_xyxy", "anchor_bbox_norm_xyxy"):
            box = _norm_box(debug.get(key))
            if box is not None:
                core_rows.append((label, box))
                break
        support = _norm_box(debug.get("support_bbox_norm_xyxy") or debug.get("envelope_bbox_norm_xyxy"))
        if support is not None:
            support_rows.append((label, support))
        members = debug.get("member_boxes_norm_xyxy")
        if isinstance(members, list):
            for idx, member in enumerate(members[:6]):
                box = _norm_box(member)
                if box is not None:
                    member_rows.append((f"{label}:m{idx + 1}", box))
    return {
        "core": _dedupe_norm_boxes(core_rows),
        "support": _dedupe_norm_boxes(support_rows),
        "member": _dedupe_norm_boxes(member_rows),
    }


def _ann_target_ar(ann: Dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or ann.get("target_ar") or "FREE")


def _ann_mode(ann: Dict[str, Any]) -> str:
    return str(ann.get("mode_name") or "unknown")


def _ann_score(ann: Dict[str, Any]) -> float:
    return _safe_float(ann.get("score_mode", ann.get("score", 0.0)))


def _routing_v2_from_ann(ann: Dict[str, Any]) -> Dict[str, Any]:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    routing = attrs.get("routing_v2_simple")
    return routing if isinstance(routing, dict) else {}


def _routing_v2_from_image(image_row: Dict[str, Any]) -> Dict[str, Any]:
    routing = image_row.get("routing_v2_simple")
    return routing if isinstance(routing, dict) else {}


def _flat_route_class(ann: Dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("flat_route_class") or "")


def _routing_label(routing: Dict[str, Any], *, include_conf: bool = True) -> str:
    if not routing:
        return "routing_v2=missing"
    family = str(routing.get("route_family_v2") or "na")
    shot = str(routing.get("person_shot_type") or "na")
    placement = str(routing.get("placement_intent") or "na")
    context = str(routing.get("context_intent") or "na")
    feasible = "feasible" if bool(routing.get("mode_feasible", True)) else "infeasible"
    label = f"{family}/{shot}/{placement}/{context}/{feasible}"
    if include_conf:
        label = f"{label}/conf={_safe_float(routing.get('routing_confidence'), 0.0):.3f}"
    return label


def _image_route_target(image_row: Dict[str, Any]) -> str:
    return str(image_row.get("image_route_name_no_placement") or "")


def _image_v16_summary_aux(image_row: Dict[str, Any]) -> str:
    return str(image_row.get("v16_mode_name") or "")


def _ann_sort_key(ann: Dict[str, Any]) -> Tuple[int, int, str, float]:
    mode = _ann_mode(ann)
    target_ar = _ann_target_ar(ann)
    return (AR_PRIORITY.get(target_ar, 99), MODE_PRIORITY.get(mode, 99), str(ann.get("query_id") or ""), -_ann_score(ann))


def _ann_best_key(ann: Dict[str, Any]) -> Tuple[float, int, float]:
    bbox = ann.get("bbox") or [0, 0, 0, 0]
    area = _safe_float(bbox[2]) * _safe_float(bbox[3]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 0.0
    return (_ann_score(ann), _safe_int(ann.get("is_best"), 0), area)


def _best_by_ar_mode(anns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ann in anns:
        if _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        key = (_ann_target_ar(ann), _ann_mode(ann))
        current = best.get(key)
        if current is None or _ann_best_key(ann) > _ann_best_key(current):
            best[key] = ann
    return sorted(best.values(), key=_ann_sort_key)


def _select_display_annotations(anns: Sequence[Dict[str, Any]], max_crops: int) -> List[Dict[str, Any]]:
    best_rows = _best_by_ar_mode(anns)
    if max_crops <= 0:
        return best_rows
    return best_rows[:max_crops]


def _fit_contain(image: Image.Image, box_w: int, box_h: int, bg: Tuple[int, int, int]) -> Tuple[Image.Image, float, int, int]:
    image = image.convert("RGB")
    scale = min(box_w / image.width, box_h / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), bg)
    ox = (box_w - new_w) // 2
    oy = (box_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def _scale_xywh_box(
    bbox: Sequence[Any],
    scale: float,
    ox: int,
    oy: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = [_safe_float(value) for value in bbox[:4]]
    x1 = int(round(max(0.0, min(float(image_w), x)) * scale + ox))
    y1 = int(round(max(0.0, min(float(image_h), y)) * scale + oy))
    x2 = int(round(max(0.0, min(float(image_w), x + w)) * scale + ox))
    y2 = int(round(max(0.0, min(float(image_h), y + h)) * scale + oy))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _scale_norm_box(
    box: Sequence[float],
    scale: float,
    ox: int,
    oy: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        int(round(x1 * image_w * scale + ox)),
        int(round(y1 * image_h * scale + oy)),
        int(round(x2 * image_w * scale + ox)),
        int(round(y2 * image_h * scale + oy)),
    )


def _draw_number_badge(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    tw, th = _text_size(draw, text, font)
    pad_x = 5
    pad_y = 3
    draw.rounded_rectangle((x, y, x + tw + pad_x * 2, y + th + pad_y * 2), radius=4, fill=(0, 0, 0), outline=color, width=2)
    draw.text((x + pad_x, y + pad_y - 1), text, fill=color, font=font)


def _draw_overlay_panel(
    *,
    source: Image.Image,
    source_id: str,
    selected_anns: Sequence[Dict[str, Any]],
    all_positive_count: int,
    ar_mode_group_count: int,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    title_font = _font(22, bold=True)
    small_font = _font(14)
    badge_font = _font(15, bold=True)
    header_h = 72
    image_box_h = panel_h - header_h
    panel = Image.new("RGB", (panel_w, panel_h), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), "Original + positive crop boxes + subject areas", fill=(20, 20, 20), font=title_font)
    draw.text(
        (12, 40),
        f"{source_id} | shown best crop/query-mode crops {len(selected_anns)} / groups {ar_mode_group_count} / positives {all_positive_count}",
        fill=MUTED,
        font=small_font,
    )
    fitted, scale, ox, oy = _fit_contain(source, panel_w - 20, image_box_h - 12, (238, 238, 238))
    img_x = 10
    img_y = header_h + 2
    panel.paste(fitted, (img_x, img_y))
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    boxes = _subject_boxes(selected_anns)
    image_w, image_h = source.size
    for _, box in boxes["core"]:
        px = _scale_norm_box(box, scale, img_x + ox, img_y + oy, image_w, image_h)
        area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
        alpha = 24 if area > 0.85 else 42
        odraw.rectangle(px, fill=(*SUBJECT_CORE_COLOR, alpha), outline=(*SUBJECT_CORE_COLOR, 210), width=3)
    for _, box in boxes["support"]:
        px = _scale_norm_box(box, scale, img_x + ox, img_y + oy, image_w, image_h)
        odraw.rectangle(px, outline=(*SUBJECT_SUPPORT_COLOR, 220), width=2)
    for _, box in boxes["member"]:
        px = _scale_norm_box(box, scale, img_x + ox, img_y + oy, image_w, image_h)
        odraw.rectangle(px, outline=(*SUBJECT_MEMBER_COLOR, 220), width=2)
    panel = Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(panel)
    legend_y = header_h + 12
    legend = "cyan=subject core, orange=support, magenta=member"
    legend_w, legend_h = _text_size(draw, legend, small_font)
    draw.rectangle((18, legend_y, min(panel_w - 16, 25 + legend_w + 8), legend_y + legend_h + 10), fill=(0, 0, 0))
    draw.text((25, legend_y + 4), legend, fill=(235, 235, 235), font=small_font)
    for idx, ann in enumerate(selected_anns, start=1):
        color = _mode_color(_ann_mode(ann))
        x1, y1, x2, y2 = _scale_xywh_box(ann.get("bbox") or [0, 0, 1, 1], scale, img_x + ox, img_y + oy, image_w, image_h)
        for offset in range(3):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        _draw_number_badge(draw, (max(img_x + 2, x1 + 4), max(img_y + 2, y1 + 4)), f"#{idx}", color, badge_font)
    draw.rectangle((img_x, img_y, img_x + fitted.width - 1, img_y + fitted.height - 1), outline=(210, 210, 210), width=1)
    return panel


def _short_query(ann: Dict[str, Any], source_id: str) -> str:
    query = str(ann.get("query_id") or "")
    prefix = f"{source_id}::"
    if query.startswith(prefix):
        query = query[len(prefix) :]
    if not query:
        query = f"{_ann_target_ar(ann)}::{_ann_mode(ann)}::{ann.get('entity_id', '')}".rstrip(":")
    return query


def _crop_ar_string(bbox: Sequence[Any]) -> str:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return "?"
    w = _safe_float(bbox[2])
    h = _safe_float(bbox[3])
    if w <= 0 or h <= 0:
        return "?"
    return f"{w / h:.2f}"


def _draw_crop_card(
    *,
    source: Image.Image,
    ann: Dict[str, Any],
    source_id: str,
    crop_idx: int,
    card_w: int,
    card_h: int,
) -> Image.Image:
    header_h = 62
    footer_h = 76
    pad = 8
    image_h = max(72, card_h - header_h - footer_h - pad * 2)
    card = Image.new("RGB", (card_w, card_h), CARD_BG)
    draw = ImageDraw.Draw(card)
    mode = _ann_mode(ann)
    color = _mode_color(mode)
    draw.rectangle((0, 0, card_w - 1, card_h - 1), outline=color, width=3)
    query_font = _font(14, bold=True)
    info_font = _font(13)
    query = f"#{crop_idx} {_short_query(ann, source_id)}"
    lines = _wrap_by_pixels(draw, query, query_font, card_w - pad * 2, max_lines=2)
    _draw_text_lines(draw, lines, (pad, 7), query_font, (20, 20, 20), line_gap=2)

    bbox = ann.get("bbox") or [0, 0, 1, 1]
    x, y, w, h = [_safe_float(value) for value in bbox[:4]]
    left = max(0, min(source.width, int(round(x))))
    top = max(0, min(source.height, int(round(y))))
    right = max(left + 1, min(source.width, int(round(x + w))))
    bottom = max(top + 1, min(source.height, int(round(y + h))))
    crop = source.crop((left, top, right, bottom))
    fitted, _, _, _ = _fit_contain(crop, card_w - pad * 2, image_h, (238, 238, 238))
    image_y = header_h + pad
    card.paste(fitted, (pad, image_y))
    draw.rectangle((pad, image_y, pad + fitted.width - 1, image_y + fitted.height - 1), outline=(220, 220, 220), width=1)

    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    candidate_source = str(attrs.get("candidate_source") or "")
    if len(candidate_source) > 26:
        candidate_source = candidate_source[:25] + "."
    footer_top = image_y + image_h + 7
    info1 = f"AR={_ann_target_ar(ann)} crop={_crop_ar_string(bbox)} score={_ann_score(ann):.3f}"
    routing = _routing_v2_from_ann(ann)
    route_short = ""
    if routing:
        route_short = (
            f"{routing.get('route_family_v2')}/{routing.get('person_shot_type') or 'na'}/"
            f"{routing.get('placement_intent')}/{routing.get('context_intent')}"
        )
    info2 = f"crop_mode={mode}"
    if route_short:
        info2 = f"{info2} | crop_route={route_short}"
    if candidate_source:
        info2 = f"{info2} | {candidate_source}"
    info1_lines = _wrap_by_pixels(draw, info1, info_font, card_w - pad * 2, max_lines=1)
    info2_lines = _wrap_by_pixels(draw, info2, info_font, card_w - pad * 2, max_lines=3)
    y_text = _draw_text_lines(draw, info1_lines, (pad, footer_top), info_font, (30, 30, 30), line_gap=2)
    _draw_text_lines(draw, info2_lines, (pad, y_text + 2), info_font, MUTED, line_gap=2)
    return card


def _draw_crop_grid(
    *,
    source: Image.Image,
    source_id: str,
    selected_anns: Sequence[Dict[str, Any]],
    positive_total: int,
    ar_mode_group_count: int,
    grid_w: int,
    crop_cols: int,
    card_h: int,
) -> Image.Image:
    title_font = _font(22, bold=True)
    small_font = _font(14)
    gap = 10
    header_h = 88
    cols = max(1, int(crop_cols))
    card_w = max(180, (grid_w - gap * (cols + 1)) // cols)
    rows = max(1, (len(selected_anns) + cols - 1) // cols)
    grid_h = header_h + gap + rows * (card_h + gap)
    grid = Image.new("RGB", (grid_w, grid_h), (250, 250, 250))
    draw = ImageDraw.Draw(grid)
    draw.text((12, 10), "Positive crop results: crop/query mode labels", fill=(20, 20, 20), font=title_font)
    subtitle = f"one best crop per target AR + crop/query mode | shown {len(selected_anns)} / groups {ar_mode_group_count} / positives {positive_total}"
    subtitle_lines = _wrap_by_pixels(draw, subtitle, small_font, grid_w - 24, max_lines=2)
    _draw_text_lines(draw, subtitle_lines, (12, 40), small_font, MUTED, line_gap=2)
    if not selected_anns:
        draw.text((12, header_h + 24), "No positive crop annotations.", fill=MUTED, font=small_font)
        return grid
    for idx, ann in enumerate(selected_anns, start=1):
        row = (idx - 1) // cols
        col = (idx - 1) % cols
        x = gap + col * (card_w + gap)
        y = header_h + gap + row * (card_h + gap)
        card = _draw_crop_card(source=source, ann=ann, source_id=source_id, crop_idx=idx, card_w=card_w, card_h=card_h)
        grid.paste(card, (x, y))
    return grid


def _mode_counts(anns: Sequence[Dict[str, Any]]) -> Counter[str]:
    return Counter(_ann_mode(ann) for ann in anns if _safe_int(ann.get("gt_flag"), 0) == 1)


def _target_ar_counts(anns: Sequence[Dict[str, Any]]) -> Counter[str]:
    return Counter(_ann_target_ar(ann) for ann in anns if _safe_int(ann.get("gt_flag"), 0) == 1)


def _counts_text(counter: Counter[str], max_items: int = 8) -> str:
    if not counter:
        return "none"
    pairs = sorted(counter.items(), key=lambda item: (-item[1], MODE_PRIORITY.get(item[0], 99), item[0]))[:max_items]
    return ", ".join(f"{key}:{value}" for key, value in pairs)


def _routing_counts(anns: Sequence[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for ann in anns:
        if _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        routing = _routing_v2_from_ann(ann)
        if routing:
            counter[_routing_label(routing, include_conf=False)] += 1
    return counter


def _flat_route_counts(anns: Sequence[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for ann in anns:
        if _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        flat = _flat_route_class(ann)
        if flat:
            counter[flat] += 1
    return counter


def _render_review_panel(
    *,
    image_row: Dict[str, Any],
    anns: Sequence[Dict[str, Any]],
    image_root: Path,
    output_path: Path,
    max_crops: int,
    crop_cols: int,
    card_h: int,
) -> Dict[str, Any]:
    source_id = _source_id(image_row)
    image_path = _resolve_image_path(image_root, image_row)
    positives = [ann for ann in anns if _safe_int(ann.get("gt_flag"), 0) == 1]
    selected = _select_display_annotations(positives, max_crops=max_crops)
    ar_mode_group_count = len(_best_by_ar_mode(positives))
    if not image_path.is_file():
        raise FileNotFoundError(f"source image not found for {source_id}: {image_path}")
    with Image.open(image_path) as im:
        source = im.convert("RGB")
    left_w = 820
    right_w = 950
    gap = 16
    overlay_h = 900
    left = _draw_overlay_panel(
        source=source,
        source_id=source_id,
        selected_anns=selected,
        all_positive_count=len(positives),
        ar_mode_group_count=ar_mode_group_count,
        panel_w=left_w,
        panel_h=overlay_h,
    )
    right = _draw_crop_grid(
        source=source,
        source_id=source_id,
        selected_anns=selected,
        positive_total=len(positives),
        ar_mode_group_count=ar_mode_group_count,
        grid_w=right_w,
        crop_cols=crop_cols,
        card_h=card_h,
    )
    header_h = 194
    body_h = max(left.height, right.height)
    canvas = Image.new("RGB", (left_w + right_w + gap * 3, header_h + body_h + gap), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28, bold=True)
    route_label_font = _font(16, bold=True)
    route_font = _font(30, bold=True)
    small_font = _font(15)
    mode_text = _counts_text(_mode_counts(positives))
    ar_text = _counts_text(_target_ar_counts(positives))
    routing = _routing_v2_from_image(image_row)
    routing_text = _routing_label(routing)
    image_route_text = _image_route_target(image_row) or str(routing.get("route_family_v2") or "")
    v16_summary_aux = _image_v16_summary_aux(image_row)
    route_text = _counts_text(_routing_counts(positives), max_items=3)
    content_w = left_w + right_w + gap * 3 - 36
    route_color = _route_color(image_route_text, str(routing.get("route_family_v2") or ""))
    draw.text((18, 10), f"{source_id}", fill=(20, 20, 20), font=title_font)
    route_bar = (18, 48, left_w + right_w + gap * 3 - 18, 96)
    draw.rounded_rectangle(route_bar, radius=8, fill=route_color, outline=(35, 35, 35), width=1)
    draw.rounded_rectangle((30, 58, 164, 86), radius=6, fill=(15, 26, 36))
    draw.text((43, 63), "IMAGE ROUTE", fill=(255, 255, 255), font=route_label_font)
    conf_text = f"conf={_safe_float(routing.get('routing_confidence'), 0.0):.3f}"
    conf_w, _ = _text_size(draw, conf_text, small_font)
    conf_x1 = route_bar[2] - conf_w - 28
    route_value = str(image_route_text or "unknown")
    route_lines = _wrap_by_pixels(draw, route_value, route_font, max(160, conf_x1 - 184 - 16), max_lines=1)
    _draw_text_lines(draw, route_lines, (184, 55), route_font, (255, 255, 255), line_gap=2)
    draw.rounded_rectangle((conf_x1, 60, route_bar[2] - 12, 84), radius=6, fill=(255, 255, 255))
    draw.text((conf_x1 + 8, 64), conf_text, fill=(35, 35, 35), font=small_font)
    mode_lines = _wrap_by_pixels(draw, f"positive crop/query modes: {mode_text}", small_font, content_w, max_lines=1)
    ar_lines = _wrap_by_pixels(draw, f"positive target ARs: {ar_text}", small_font, content_w, max_lines=1)
    metadata_lines = _wrap_by_pixels(
        draw,
        f"image metadata: {routing_text} | v16 summary aux: {v16_summary_aux}",
        small_font,
        content_w,
        max_lines=1,
    )
    positive_route_lines = _wrap_by_pixels(draw, f"positive crop routing groups: {route_text}", small_font, content_w, max_lines=1)
    _draw_text_lines(draw, metadata_lines, (18, 106), small_font, (45, 65, 85), line_gap=2)
    _draw_text_lines(draw, mode_lines, (18, 128), small_font, (55, 55, 55), line_gap=2)
    _draw_text_lines(draw, ar_lines, (18, 148), small_font, (55, 55, 55), line_gap=2)
    _draw_text_lines(draw, positive_route_lines, (18, 166), small_font, (45, 65, 85), line_gap=2)
    canvas.paste(left, (gap, header_h))
    canvas.paste(right, (left_w + gap * 2, header_h))
    draw.line((left_w + gap + gap // 2, header_h, left_w + gap + gap // 2, header_h + body_h), fill=(205, 205, 205), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    shown_queries = [_short_query(ann, source_id) for ann in selected]
    return {
        "source_image_id": source_id,
        "review_image": str(output_path),
        "source_image": str(image_path),
        "image_route_name_no_placement": image_route_text,
        "image_v16_summary_mode_aux": v16_summary_aux,
        "image_flat_route_class_aux": str(image_row.get("flat_route_class") or ""),
        "positive_total": len(positives),
        "ar_mode_group_count": ar_mode_group_count,
        "shown_crop_count": len(selected),
        "mode_counts": dict(_mode_counts(positives)),
        "target_ar_counts": dict(_target_ar_counts(positives)),
        "image_route_family_v2": str(routing.get("route_family_v2") or ""),
        "image_person_shot_type": str(routing.get("person_shot_type") or ""),
        "image_context_intent": str(routing.get("context_intent") or ""),
        "image_routing_confidence": _safe_float(routing.get("routing_confidence"), 0.0),
        "routing_v2_counts": dict(_routing_counts(positives)),
        "flat_route_counts": dict(_flat_route_counts(positives)),
        "shown_queries": shown_queries,
    }


def _select_image_ids(
    *,
    images: Dict[int, Dict[str, Any]],
    anns_by_image: Dict[int, List[Dict[str, Any]]],
    samples: int,
    per_bucket: int,
    seed: int,
) -> List[int]:
    positives_by_image = {
        image_id: [ann for ann in anns if _safe_int(ann.get("gt_flag"), 0) == 1]
        for image_id, anns in anns_by_image.items()
    }
    bucket_rows: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
    image_score: Dict[int, float] = defaultdict(float)
    for image_id, positives in positives_by_image.items():
        for ann in positives:
            key = (_ann_mode(ann), _ann_target_ar(ann))
            score = _ann_score(ann)
            bucket_rows[key].append((score, image_id))
            image_score[image_id] += score
    selected: List[int] = []
    selected_set: set[int] = set()
    for key in sorted(bucket_rows, key=lambda item: (MODE_PRIORITY.get(item[0], 99), AR_PRIORITY.get(item[1], 99), item)):
        rows = sorted(bucket_rows[key], key=lambda item: (-item[0], item[1]))
        picked = 0
        for _, image_id in rows:
            if image_id not in images or image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            picked += 1
            if samples > 0 and len(selected) >= samples:
                return selected
            if picked >= per_bucket:
                break
    rng = random.Random(seed)
    rest = []
    for image_id, positives in positives_by_image.items():
        if image_id in selected_set or image_id not in images:
            continue
        modes = {_ann_mode(ann) for ann in positives}
        target_ars = {_ann_target_ar(ann) for ann in positives}
        rest.append((len(modes), len(target_ars), len(positives), image_score.get(image_id, 0.0), rng.random(), image_id))
    rest.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4], item[5]))
    for *_, image_id in rest:
        selected.append(image_id)
        selected_set.add(image_id)
        if samples > 0 and len(selected) >= samples:
            break
    return selected


def _read_source_ids(path: Path) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _save_contact_sheet_chunk(
    *,
    rows: Sequence[Dict[str, Any]],
    thumbs: Sequence[Image.Image],
    out_path: Path,
    cols: int,
    thumb_w: int,
    cell_h: int,
) -> None:
    sheet = Image.new("RGB", (cols * thumb_w, ((len(thumbs) + cols - 1) // cols) * cell_h), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(20, bold=True)
    small_font = _font(13)
    for idx, (row, thumb) in enumerate(zip(rows, thumbs)):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * cell_h
        draw.text((x + 8, y + 6), str(row["source_image_id"]), fill=(20, 20, 20), font=title_font)
        draw.text((x + 8, y + 30), f"pos={row['positive_total']} shown={row['shown_crop_count']}", fill=MUTED, font=small_font)
        sheet.paste(thumb, (x, y + 52))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def _write_contact_sheets(rows: Sequence[Dict[str, Any]], out_path: Path, *, cols: int, thumb_w: int) -> List[str]:
    if not rows:
        return []
    thumbs: List[Image.Image] = []
    for row in rows:
        with Image.open(row["review_image"]) as im:
            image = im.convert("RGB")
        scale = thumb_w / image.width
        thumbs.append(image.resize((thumb_w, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS))
    cell_h = max(thumb.height for thumb in thumbs) + 58
    max_jpeg_dimension = 60000
    max_rows_per_sheet = max(1, max_jpeg_dimension // max(1, cell_h))
    max_items_per_sheet = max(cols, max_rows_per_sheet * cols)
    if len(rows) <= max_items_per_sheet:
        _save_contact_sheet_chunk(rows=rows, thumbs=thumbs, out_path=out_path, cols=cols, thumb_w=thumb_w, cell_h=cell_h)
        return [str(out_path)]
    paths: List[str] = []
    stem = out_path.stem
    suffix = out_path.suffix or ".jpg"
    for start in range(0, len(rows), max_items_per_sheet):
        end = min(len(rows), start + max_items_per_sheet)
        chunk_path = out_path.with_name(f"{stem}_part{len(paths) + 1:03d}{suffix}")
        _save_contact_sheet_chunk(
            rows=rows[start:end],
            thumbs=thumbs[start:end],
            out_path=chunk_path,
            cols=cols,
            thumb_w=thumb_w,
            cell_h=cell_h,
        )
        paths.append(str(chunk_path))
    return paths


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "source_image_id",
        "image_route_name_no_placement",
        "image_v16_summary_mode_aux",
        "image_flat_route_class_aux",
        "image_route_family_v2",
        "image_person_shot_type",
        "image_context_intent",
        "image_routing_confidence",
        "positive_total",
        "ar_mode_group_count",
        "shown_crop_count",
        "review_image",
        "source_image",
        "mode_counts_json",
        "target_ar_counts_json",
        "routing_v2_counts_json",
        "flat_route_counts_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_image_id": row.get("source_image_id", ""),
                    "image_route_name_no_placement": row.get("image_route_name_no_placement", ""),
                    "image_v16_summary_mode_aux": row.get("image_v16_summary_mode_aux", ""),
                    "image_flat_route_class_aux": row.get("image_flat_route_class_aux", ""),
                    "image_route_family_v2": row.get("image_route_family_v2", ""),
                    "image_person_shot_type": row.get("image_person_shot_type", ""),
                    "image_context_intent": row.get("image_context_intent", ""),
                    "image_routing_confidence": row.get("image_routing_confidence", ""),
                    "positive_total": row.get("positive_total", ""),
                    "ar_mode_group_count": row.get("ar_mode_group_count", ""),
                    "shown_crop_count": row.get("shown_crop_count", ""),
                    "review_image": row.get("review_image", ""),
                    "source_image": row.get("source_image", ""),
                    "mode_counts_json": json.dumps(row.get("mode_counts", {}), ensure_ascii=False, sort_keys=True),
                    "target_ar_counts_json": json.dumps(row.get("target_ar_counts", {}), ensure_ascii=False, sort_keys=True),
                    "routing_v2_counts_json": json.dumps(row.get("routing_v2_counts", {}), ensure_ascii=False, sort_keys=True),
                    "flat_route_counts_json": json.dumps(row.get("flat_route_counts", {}), ensure_ascii=False, sort_keys=True),
                }
            )


def _write_review_guide(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Multi-Mode Crop Review Guide",
        "",
        "이 시각화는 비교 기준 없이 multi-mode 라벨 자체를 정성 검수하기 위한 산출물이다.",
        "",
        "## 읽는 법",
        "",
        "- 왼쪽 패널: 원본 이미지 위에 선택된 positive crop box를 표시한다.",
        "- 오른쪽 패널: `(target AR, crop/query mode)` 조합별 score가 가장 높은 best crop 1개만 카드 형태로 보여준다.",
        "- 패널 상단의 컬러 `IMAGE ROUTE` 바는 image-level classifier target이며 placement를 포함하지 않는다.",
        "- 패널 상단의 `positive crop/query modes`와 crop card의 `crop_mode=`는 실제 crop annotation/query mode이다.",
        "- crop 카드 하단의 `crop_route=` 값은 annotation-level `route_family_v2/person_shot_type/placement_intent/context_intent`이며, 긴 텍스트는 카드 안에서 줄바꿈된다.",
        "- `v16 summary aux`는 image row의 보조 요약값이며 image-level primary target이 아니다.",
        "- crop 카드의 query/AR/score 텍스트는 crop 이미지 위가 아니라 별도 header/footer 영역에 배치했다.",
        "- cyan 영역은 subject core, orange box는 support/envelope, magenta box는 member subject를 의미한다.",
        "- crop box 번호와 오른쪽 crop 카드 번호가 대응한다.",
        "- mode별 crop/카드 경계선은 서로 구분되는 고정 팔레트를 사용한다.",
        "",
        "## 산출물",
        "",
        f"- selected panels: `{summary['panel_dir']}`",
        f"- contact sheets: `{summary['contact_sheets'][0] if summary.get('contact_sheets') else ''}`",
        f"- manifest JSON: `{summary['manifest_json']}`",
        f"- manifest CSV: `{summary['manifest_csv']}`",
        "",
        "## 요약",
        "",
        f"- selected images: {summary['selected_count']}",
        f"- written panels: {summary['written_count']}",
        f"- max crops per image: {summary['max_crops']}",
        f"- crop columns: {summary['crop_cols']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build qualitative crop review panels for multimode labels.")
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_ids", default="", help="Optional text file of source image ids to visualize in order.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--per_bucket", type=int, default=3, help="Initial sample quota per (mode, target_ar) bucket.")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max_crops", type=int, default=12)
    parser.add_argument("--crop_cols", type=int, default=3)
    parser.add_argument("--card_h", type=int, default=246)
    parser.add_argument("--contact_cols", type=int, default=2)
    parser.add_argument("--contact_thumb_w", type=int, default=760)
    parser.add_argument("--manifest_prefix", default="v11_crop_review")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    label_json = Path(args.label_json)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    panel_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(label_json)
    images = {int(row["id"]): row for row in payload.get("images", []) if isinstance(row, dict)}
    source_to_image_id = {_source_id(row): image_id for image_id, row in images.items()}
    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if isinstance(ann, dict):
            anns_by_image[int(ann.get("image_id", -1))].append(ann)

    if str(args.image_ids).strip():
        selected_ids = []
        for source_id in _read_source_ids(Path(args.image_ids)):
            image_id = source_to_image_id.get(source_id)
            if image_id is not None:
                selected_ids.append(image_id)
    else:
        selected_ids = _select_image_ids(
            images=images,
            anns_by_image=anns_by_image,
            samples=int(args.samples),
            per_bucket=max(1, int(args.per_bucket)),
            seed=int(args.seed),
        )

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for idx, image_id in enumerate(selected_ids, start=1):
        image_row = images.get(image_id)
        if image_row is None:
            continue
        source_id = _source_id(image_row)
        try:
            row = _render_review_panel(
                image_row=image_row,
                anns=anns_by_image.get(image_id, []),
                image_root=image_root,
                output_path=panel_dir / f"review_{source_id}.jpg",
                max_crops=int(args.max_crops),
                crop_cols=int(args.crop_cols),
                card_h=int(args.card_h),
            )
        except FileNotFoundError:
            missing.append(source_id)
            continue
        rows.append(row)
        if idx % 25 == 0:
            print(f"rendered {idx}/{len(selected_ids)}")

    contact_path = out_dir / f"{args.manifest_prefix}_contact_sheet.jpg"
    contact_sheets = _write_contact_sheets(
        rows,
        contact_path,
        cols=max(1, int(args.contact_cols)),
        thumb_w=max(320, int(args.contact_thumb_w)),
    )
    selected_ids_path = out_dir / f"{args.manifest_prefix}_selected_image_ids.txt"
    selected_ids_path.write_text("\n".join(row["source_image_id"] for row in rows) + "\n", encoding="utf-8")
    manifest_json = out_dir / f"{args.manifest_prefix}_manifest.json"
    manifest_csv = out_dir / f"{args.manifest_prefix}_manifest.csv"
    mode_totals: Counter[str] = Counter()
    ar_totals: Counter[str] = Counter()
    image_route_totals: Counter[str] = Counter()
    image_route_no_placement_totals: Counter[str] = Counter()
    image_shot_totals: Counter[str] = Counter()
    positive_route_totals: Counter[str] = Counter()
    flat_route_totals: Counter[str] = Counter()
    for row in rows:
        mode_totals.update(row.get("mode_counts", {}))
        ar_totals.update(row.get("target_ar_counts", {}))
        if row.get("image_route_family_v2"):
            image_route_totals[str(row.get("image_route_family_v2"))] += 1
        if row.get("image_route_name_no_placement"):
            image_route_no_placement_totals[str(row.get("image_route_name_no_placement"))] += 1
        if row.get("image_person_shot_type"):
            image_shot_totals[str(row.get("image_person_shot_type"))] += 1
        positive_route_totals.update(row.get("routing_v2_counts", {}))
        flat_route_totals.update(row.get("flat_route_counts", {}))
    summary = {
        "label_json": str(label_json),
        "image_root": str(image_root),
        "out_dir": str(out_dir),
        "panel_dir": str(panel_dir),
        "selected_count": len(selected_ids),
        "written_count": len(rows),
        "missing_count": len(missing),
        "missing_source_image_ids": missing,
        "max_crops": int(args.max_crops),
        "crop_cols": int(args.crop_cols),
        "card_h": int(args.card_h),
        "contact_sheets": contact_sheets,
        "selected_ids_path": str(selected_ids_path),
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "mode_positive_counts_in_selected": dict(sorted(mode_totals.items())),
        "target_ar_positive_counts_in_selected": dict(sorted(ar_totals.items())),
        "image_route_family_counts_in_selected": dict(sorted(image_route_totals.items())),
        "image_route_no_placement_counts_in_selected": dict(sorted(image_route_no_placement_totals.items())),
        "image_person_shot_counts_in_selected": dict(sorted(image_shot_totals.items())),
        "positive_routing_v2_counts_in_selected": dict(sorted(positive_route_totals.items())),
        "flat_route_counts_in_selected": dict(sorted(flat_route_totals.items())),
        "rows": rows,
    }
    _write_json(manifest_json, summary)
    _write_csv(manifest_csv, rows)
    guide_path = out_dir / "CROP_REVIEW_GUIDE_KO.md"
    _write_review_guide(guide_path, summary)
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "written_count": len(rows),
                "missing_count": len(missing),
                "contact_sheets": contact_sheets,
                "manifest_json": str(manifest_json),
                "review_guide": str(guide_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
