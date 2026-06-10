#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


FACE_REASONS = [
    "body_leakage_strict",
    "face_center_x_miss_strict",
    "face_too_loose",
    "head_subject_coverage",
    "head_top_cut",
]

PERSON_MODES = {"single_person_center", "single_person_rot", "group_center", "group_rot", "face"}

GUIDE_PROBLEM_SOURCES = {
    "sstk_image_1972270181": "legacy guide 4.2 Headroom Tight",
    "sstk_image_1633699804": "legacy guide 5.2 Lookroom Insufficient",
    "bigstock_image_428715728": "legacy guide 5.3 Lookroom Excessive",
}

REASON_TEXT = {
    "body_leakage_strict": {
        "threshold": "face_body_leakage > 0.25 in strict face mode",
        "meaning": "The crop keeps the face, but too much torso/body area leaks into a face-intent crop.",
        "metrics": ["face_body_leakage", "subject_ratio", "face_recall", "q_scale"],
    },
    "face_center_x_miss_strict": {
        "threshold": "abs(anchor_rx - 0.5) > 0.22",
        "meaning": "The face/head anchor is too far from the horizontal center for face-center intent.",
        "metrics": ["anchor_rx", "q_place", "portrait_control_center_deviation"],
    },
    "face_too_loose": {
        "threshold": "subject_ratio < 0.12",
        "meaning": "The face/head occupies too little of the crop, so the face crop becomes a loose scene crop.",
        "metrics": ["subject_ratio", "q_scale", "face_recall"],
    },
    "head_subject_coverage": {
        "threshold": "core_recall < 0.92",
        "meaning": "The head/face core is not sufficiently covered by the crop.",
        "metrics": ["core_recall", "face_recall", "q_subj"],
    },
    "head_top_cut": {
        "threshold": "headroom_value < 0.03 in face mode",
        "meaning": "The top of the head is too close to, or cut by, the crop top.",
        "metrics": ["headroom_value", "q_head", "face_recall"],
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                val = row.get(key)
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False, sort_keys=True)
                out[key] = val
            writer.writerow(out)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


FONT_TITLE = _font(28, bold=True)
FONT_H2 = _font(22, bold=True)
FONT = _font(17)
FONT_SMALL = _font(14)
FONT_MONO = _font(14)


def _text_w(font: ImageFont.ImageFont, text: str) -> float:
    try:
        return float(font.getlength(text))
    except Exception:
        return float(len(text) * 8)


def _wrap(text: str, max_px: int, font: ImageFont.ImageFont) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else f"{cur} {word}"
        if _text_w(font, nxt) <= max_px:
            cur = nxt
            continue
        if cur:
            lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_lines(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: Iterable[str], *, font: ImageFont.ImageFont, fill: tuple[int, int, int], gap: int = 5) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), str(line), font=font, fill=fill)
        y += int(font.size) + gap
    return y


def _source_id(image_row: dict[str, Any]) -> str:
    return str(image_row.get("source_image_id") or Path(str(image_row.get("file_name") or "")).stem)


def _resolve_image_path(images_dir: Path, image_row: dict[str, Any]) -> Path:
    file_name = str(image_row.get("file_name") or "")
    if file_name and (images_dir / file_name).exists():
        return images_dir / file_name
    sid = _source_id(image_row)
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = images_dir / f"{sid}{suffix}"
        if p.exists():
            return p
    return images_dir / file_name


def _fit(im: Image.Image, max_w: int, max_h: int) -> tuple[Image.Image, float]:
    scale = min(max_w / im.width, max_h / im.height)
    w = max(1, int(round(im.width * scale)))
    h = max(1, int(round(im.height * scale)))
    return im.resize((w, h), Image.Resampling.LANCZOS), scale


def _bbox_xyxy_from_ann(ann: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 1, 1])]
    return x, y, x + w, y + h


def _crop_norm_from_ann(ann: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _bbox_xyxy_from_ann(ann)
    return x1 / width, y1 / height, x2 / width, y2 / height


def _norm_box(box: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    vals = [_safe_float(v, float("nan")) for v in box]
    if not all(math.isfinite(v) for v in vals):
        return None
    x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in vals]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _norm_point(pt: Any) -> tuple[float, float] | None:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    x = _safe_float(pt[0], float("nan"))
    y = _safe_float(pt[1], float("nan"))
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _box_to_px(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    return box[0] * width, box[1] * height, box[2] * width, box[3] * height


def _draw_px_rect(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], offset: tuple[int, int], scale: float, color: tuple[int, int, int], width: int = 4) -> None:
    ox, oy = offset
    x1, y1, x2, y2 = xyxy
    draw.rectangle((ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale), outline=color, width=width)


def _draw_px_point(draw: ImageDraw.ImageDraw, xy: tuple[float, float], offset: tuple[int, int], scale: float, color: tuple[int, int, int], radius: int = 7) -> None:
    ox, oy = offset
    x, y = xy
    cx = ox + x * scale
    cy = oy + y * scale
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(20, 20, 20), width=2)


def _draw_px_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], offset: tuple[int, int], scale: float, color: tuple[int, int, int], width: int = 4) -> None:
    ox, oy = offset
    sx, sy = start
    ex, ey = end
    p1 = (ox + sx * scale, oy + sy * scale)
    p2 = (ox + ex * scale, oy + ey * scale)
    draw.line((p1, p2), fill=color, width=width)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    size = 16
    head = [
        p2,
        (p2[0] - ux * size + px * size * 0.45, p2[1] - uy * size + py * size * 0.45),
        (p2[0] - ux * size - px * size * 0.45, p2[1] - uy * size - py * size * 0.45),
    ]
    draw.polygon(head, fill=color)


def _contains(box: tuple[float, float, float, float] | None, pt: tuple[float, float] | None) -> bool:
    if box is None or pt is None:
        return False
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _ann_info(ann: dict[str, Any]) -> dict[str, Any]:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    comps = attrs.get("score_components") if isinstance(attrs.get("score_components"), dict) else {}
    subj = attrs.get("subject_debug") if isinstance(attrs.get("subject_debug"), dict) else {}
    portrait = attrs.get("portrait_comp") if isinstance(attrs.get("portrait_comp"), dict) else {}
    reasons = ann.get("hard_reject_reasons") or attrs.get("hard_reject_reasons") or []
    return {
        "attrs": attrs,
        "components": comps,
        "subject_debug": subj,
        "portrait_comp": portrait,
        "reasons": [str(x) for x in reasons],
    }


def _draw_subject_overlays(
    draw: ImageDraw.ImageDraw,
    *,
    ann: dict[str, Any] | None,
    status_attrs: dict[str, Any] | None,
    image_size: tuple[int, int],
    offset: tuple[int, int],
    scale: float,
    crop_xyxy: tuple[float, float, float, float] | None = None,
    gaze: bool = True,
) -> None:
    width, height = image_size
    if ann is not None:
        info = _ann_info(ann)
        subj = info["subject_debug"]
        portrait = info["portrait_comp"]
        comps = info["components"]
    else:
        subj = {}
        portrait = (status_attrs or {}).get("portrait_comp") if isinstance((status_attrs or {}).get("portrait_comp"), dict) else {}
        comps = {}

    def draw_norm_box(key: str, color: tuple[int, int, int], line: int = 3) -> None:
        box = _norm_box(subj.get(key))
        if box is None:
            return
        _draw_px_rect(draw, _box_to_px(box, width, height), offset, scale, color, line)

    draw_norm_box("core_bbox_norm_xyxy", (0, 210, 210), 4)
    draw_norm_box("face_bbox_norm_xyxy", (0, 120, 255), 4)
    draw_norm_box("head_bbox_norm_xyxy", (80, 160, 255), 3)
    pbox = _norm_box(portrait.get("bbox_norm_xyxy"))
    if pbox is not None:
        _draw_px_rect(draw, _box_to_px(pbox, width, height), offset, scale, (255, 170, 0), 3)
    if not pbox and status_attrs and isinstance(status_attrs.get("pose_bbox_norm_xyxy"), (list, tuple)):
        pbox = _norm_box(status_attrs.get("pose_bbox_norm_xyxy"))
        if pbox is not None:
            _draw_px_rect(draw, _box_to_px(pbox, width, height), offset, scale, (255, 170, 0), 3)

    points = [
        ("eye_point_norm_xy", (255, 230, 0)),
        ("control_point_norm_xy", (180, 80, 255)),
        ("support_point_norm_xy", (255, 90, 0)),
        ("torso_point_norm_xy", (60, 220, 90)),
        ("pelvis_point_norm_xy", (255, 130, 60)),
    ]
    for key, color in points:
        pt = _norm_point(portrait.get(key))
        if pt is not None:
            _draw_px_point(draw, (pt[0] * width, pt[1] * height), offset, scale, color, 7)

    if gaze:
        eye = _norm_point(portrait.get("eye_point_norm_xy"))
        gaze_dir = str(comps.get("gaze_dir") or portrait.get("direction_hint") or "unknown")
        if eye is not None and gaze_dir in {"left", "right"}:
            x = eye[0] * width
            y = eye[1] * height
            dx = 0.16 * width * (-1.0 if gaze_dir == "left" else 1.0)
            _draw_px_arrow(draw, (x, y), (x + dx, y), offset, scale, (255, 220, 0), 5)

    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        cx = 0.5 * (x1 + x2)
        _draw_px_rect(draw, crop_xyxy, offset, scale, (255, 45, 45), 5)
        ox, oy = offset
        draw.line((ox + cx * scale, oy + y1 * scale, ox + cx * scale, oy + y2 * scale), fill=(255, 45, 45), width=2)


def _relative_norm_box_to_crop(box: tuple[float, float, float, float], crop: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = crop
    cw = max(1e-8, x2 - x1)
    ch = max(1e-8, y2 - y1)
    rx1 = (box[0] - x1) / cw
    ry1 = (box[1] - y1) / ch
    rx2 = (box[2] - x1) / cw
    ry2 = (box[3] - y1) / ch
    rx1, ry1, rx2, ry2 = max(0.0, min(1.0, rx1)), max(0.0, min(1.0, ry1)), max(0.0, min(1.0, rx2)), max(0.0, min(1.0, ry2))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return rx1, ry1, rx2, ry2


def draw_annotation_panel(
    *,
    image_path: Path,
    image_row: dict[str, Any],
    ann: dict[str, Any],
    title: str,
    explanation: str,
    out_path: Path,
    reason: str | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    crop_xyxy = _bbox_xyxy_from_ann(ann)
    crop_box_int = tuple(int(round(v)) for v in crop_xyxy)
    crop = source.crop(crop_box_int)
    canvas = Image.new("RGB", (1650, 980), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, font=FONT_TITLE, fill=(20, 20, 20))
    draw.text((24, 54), _source_id(image_row), font=FONT, fill=(80, 80, 80))

    left_x, top_y = 24, 96
    fitted, scale = _fit(source, 720, 620)
    canvas.paste(fitted, (left_x, top_y))
    draw.rectangle((left_x, top_y, left_x + fitted.width - 1, top_y + fitted.height - 1), outline=(170, 170, 170), width=1)
    _draw_subject_overlays(draw, ann=ann, status_attrs=None, image_size=(width, height), offset=(left_x, top_y), scale=scale, crop_xyxy=crop_xyxy)

    crop_x, crop_y = 790, 96
    crop_fit, crop_scale = _fit(crop, 430, 620)
    canvas.paste(crop_fit, (crop_x, crop_y))
    draw.rectangle((crop_x, crop_y, crop_x + crop_fit.width - 1, crop_y + crop_fit.height - 1), outline=(170, 170, 170), width=1)
    crop_norm = _crop_norm_from_ann(ann, width, height)
    info = _ann_info(ann)
    subj = info["subject_debug"]
    for key, color in [
        ("core_bbox_norm_xyxy", (0, 210, 210)),
        ("face_bbox_norm_xyxy", (0, 120, 255)),
        ("head_bbox_norm_xyxy", (80, 160, 255)),
    ]:
        box = _norm_box(subj.get(key))
        if box is None:
            continue
        rbox = _relative_norm_box_to_crop(box, crop_norm)
        if rbox is None:
            continue
        _draw_px_rect(draw, _box_to_px(rbox, crop.width, crop.height), (crop_x, crop_y), crop_scale, color, 3)

    text_x, text_y = 1250, 96
    draw.text((text_x, text_y), "Gate explanation", font=FONT_H2, fill=(20, 20, 20))
    text_y += 36
    if reason:
        rt = REASON_TEXT.get(reason, {})
        text_y = _draw_lines(draw, (text_x, text_y), _wrap(f"Reason: {reason}", 350, FONT), font=FONT, fill=(180, 30, 30))
        text_y = _draw_lines(draw, (text_x, text_y), _wrap(f"Threshold: {rt.get('threshold', '-')}", 350, FONT_SMALL), font=FONT_SMALL, fill=(50, 50, 50))
        text_y = _draw_lines(draw, (text_x, text_y), _wrap(str(rt.get("meaning", "")), 350, FONT_SMALL), font=FONT_SMALL, fill=(70, 70, 70))
    text_y += 10
    attrs = info["attrs"]
    comps = info["components"]
    rows = [
        f"mode/ar: {ann.get('mode_name')} / {attrs.get('target_ar')}",
        f"gt: {ann.get('gt_flag')}  score: {ann.get('score_mode')}",
        f"candidate: {attrs.get('candidate_source')}",
        f"hard rejects: {', '.join(info['reasons']) or '-'}",
        f"negative: {attrs.get('negative_reason') or '-'}",
    ]
    if reason:
        for key in REASON_TEXT.get(reason, {}).get("metrics", []):
            rows.append(f"{key}: {comps.get(key)}")
    rows.extend(extra_lines or [])
    for row in rows:
        text_y = _draw_lines(draw, (text_x, text_y), _wrap(row, 350, FONT_MONO), font=FONT_MONO, fill=(20, 20, 20), gap=4)

    legend = "red=crop, cyan=core/head, blue=face/head, orange=person, yellow=gaze/eye, purple=control, orange dot=support"
    _draw_lines(draw, (24, 735), _wrap(explanation, 1190, FONT), font=FONT, fill=(40, 40, 40), gap=6)
    _draw_lines(draw, (24, 900), _wrap(legend, 1550, FONT_SMALL), font=FONT_SMALL, fill=(80, 80, 80), gap=4)
    canvas.save(out_path, quality=92)


def _reason_score(reason: str, ann: dict[str, Any]) -> tuple[float, ...]:
    comps = ((ann.get("attributes") or {}).get("score_components") or {})
    reasons = _ann_info(ann)["reasons"]
    score = _safe_float(ann.get("score_mode"), 0.0)
    ar = str(((ann.get("attributes") or {}).get("target_ar")) or "")
    ar_pref = 1.0 if ar in {"9:16", "3:4"} else 0.0
    clean = -float(len(reasons))
    if reason == "body_leakage_strict":
        return (clean, score, _safe_float(comps.get("face_body_leakage"), 0.0), ar_pref)
    if reason == "face_center_x_miss_strict":
        return (clean, score, abs(_safe_float(comps.get("anchor_rx"), 0.5) - 0.5), ar_pref)
    if reason == "face_too_loose":
        return (clean, score, -_safe_float(comps.get("subject_ratio"), 1.0), ar_pref)
    if reason == "head_subject_coverage":
        return (clean, score, -_safe_float(comps.get("core_recall"), 1.0), ar_pref)
    if reason == "head_top_cut":
        return (clean, score, -_safe_float(comps.get("headroom_value"), 1.0), ar_pref)
    return (score,)


def build_face_reason_visuals(label: dict[str, Any], images_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    images = {int(im["id"]): im for im in label["images"]}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in label["annotations"]:
        if str(ann.get("mode_name")) != "face" or int(ann.get("gt_flag") or 0) != 0:
            continue
        reasons = _ann_info(ann)["reasons"]
        for reason in FACE_REASONS:
            if reason in reasons:
                image_row = images[int(ann["image_id"])]
                if _resolve_image_path(images_dir, image_row).exists():
                    candidates[reason].append(ann)

    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for reason in FACE_REASONS:
        anns = sorted(candidates.get(reason, []), key=lambda a: _reason_score(reason, a), reverse=True)
        if not anns:
            continue
        ann = anns[0]
        image_row = images[int(ann["image_id"])]
        sid = _source_id(image_row)
        out = out_dir / f"face_reason_{reason}_{sid}.jpg"
        draw_annotation_panel(
            image_path=_resolve_image_path(images_dir, image_row),
            image_row=image_row,
            ann=ann,
            reason=reason,
            title=f"Face AR missing reason: {reason}",
            explanation="This panel uses an actual v14 negative face crop. It explains why a target-AR face query can be opened but fail to produce a positive label.",
            out_path=out,
        )
        info = _ann_info(ann)
        comps = info["components"]
        row = {
            "reason": reason,
            "source_image_id": sid,
            "annotation_id": ann.get("id"),
            "target_ar": info["attrs"].get("target_ar"),
            "score_mode": ann.get("score_mode"),
            "hard_rejects": info["reasons"],
            "panel": out.name,
            "face_body_leakage": comps.get("face_body_leakage"),
            "anchor_rx": comps.get("anchor_rx"),
            "subject_ratio": comps.get("subject_ratio"),
            "core_recall": comps.get("core_recall"),
            "headroom_value": comps.get("headroom_value"),
        }
        rows.append(row)
        panel_paths.append(out)

    make_contact_sheet(panel_paths, out_dir / "face_reason_explanation_contact_sheet.jpg", title="Face mode AR missing reason examples")
    _write_csv(out_dir / "face_reason_visual_manifest.csv", rows)
    return rows


def make_contact_sheet(paths: list[Path], out_path: Path, *, title: str) -> None:
    if not paths:
        return
    thumb_w = 760
    thumb_h = 452
    cols = 2
    rows = math.ceil(len(paths) / cols)
    header = 64
    sheet = Image.new("RGB", (cols * thumb_w + 36, header + rows * (thumb_h + 42) + 24), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 16), title, font=FONT_TITLE, fill=(20, 20, 20))
    for idx, path in enumerate(paths):
        im = Image.open(path).convert("RGB")
        fitted, _ = _fit(im, thumb_w - 20, thumb_h)
        x = 18 + (idx % cols) * thumb_w
        y = header + (idx // cols) * (thumb_h + 42)
        draw.text((x, y), path.stem, font=FONT_SMALL, fill=(60, 60, 60))
        sheet.paste(fitted, (x, y + 22))
        draw.rectangle((x, y + 22, x + fitted.width - 1, y + 22 + fitted.height - 1), outline=(190, 190, 190), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def _load_status_rows(path: Path, source_id: str) -> list[dict[str, Any]]:
    return [row for row in _iter_jsonl(path) if str(row.get("source_image_id")) == source_id]


def _load_precompute_pose(precompute_path: Path, source_id: str, width: int, height: int) -> dict[str, Any]:
    if not precompute_path.exists():
        return {}
    for row in _iter_jsonl(precompute_path):
        if str(row.get("image_id") or row.get("source_image_id")) != source_id:
            continue
        poses = row.get("c3_pose") if isinstance(row.get("c3_pose"), list) else []
        if not poses:
            return {}
        pose = poses[0]
        bbox = pose.get("bbox") if isinstance(pose, dict) else None
        out: dict[str, Any] = {}
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            out["pose_bbox_norm_xyxy"] = [float(bbox[0]) / width, float(bbox[1]) / height, float(bbox[2]) / width, float(bbox[3]) / height]
        return out
    return {}


def draw_person_gate_panel(
    *,
    image_path: Path,
    image_row: dict[str, Any],
    status_rows: list[dict[str, Any]],
    precompute_extra: dict[str, Any],
    out_path: Path,
) -> None:
    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    canvas = Image.new("RGB", (1650, 1080), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Person crop no-positive gates: sstk_image_1197510367", font=FONT_TITLE, fill=(20, 20, 20))
    draw.text((24, 54), "Diagnostic crops are illustrative. Final v14 label JSON did not keep person negative boxes for this query.", font=FONT, fill=(80, 80, 80))

    first_person = next((r for r in status_rows if str(r.get("mode_name")) in {"single_person_center", "single_person_rot"}), {})
    attrs = first_person.get("attributes") if isinstance(first_person.get("attributes"), dict) else {}
    attrs = dict(attrs)
    attrs.update(precompute_extra)
    portrait = attrs.get("portrait_comp") if isinstance(attrs.get("portrait_comp"), dict) else {}
    support = _norm_point(portrait.get("support_point_norm_xy"))
    control = _norm_point(portrait.get("control_point_norm_xy"))
    eye = _norm_point(portrait.get("eye_point_norm_xy"))
    pose_box = _norm_box(attrs.get("pose_bbox_norm_xyxy"))
    if pose_box is None:
        pose_box = (0.30, 0.25, 0.50, 0.82)

    panels = [
        {
            "title": "A. support grounding miss",
            "crop": (0.0, 0.0, 1.0, 1.0),
            "text": "support point should sit near lower target band (0.95 for full-body). Here it is much higher, so q_support_y becomes low.",
        },
        {
            "title": "B. center/rot placement miss",
            "crop": (0.05, 0.0, 0.72, 1.0),
            "text": "center wants control x near 0.5; rot wants a clean thirds placement. Ambiguous/shifted control placement lowers q_place.",
        },
        {
            "title": "C. side/bottom margin tight",
            "crop": (max(0.0, pose_box[0] - 0.005), max(0.0, pose_box[1] - 0.04), min(1.0, pose_box[2] + 0.005), min(1.0, pose_box[3] + 0.005)),
            "text": "side margin threshold is 0.025/0.035 and bottom margin threshold is 0.025/0.035. Tight crops can touch body edges.",
        },
        {
            "title": "D. coverage miss",
            "crop": (pose_box[0] + 0.03, pose_box[1] + 0.08, min(1.0, pose_box[2] - 0.01), min(1.0, pose_box[3] - 0.08)),
            "text": "subject coverage gates reject crops that cut too much of core/body, even if the resulting crop may look plausible as an upper-body portrait.",
        },
    ]

    cell_w, cell_h = 780, 420
    for idx, panel in enumerate(panels):
        x = 24 + (idx % 2) * (cell_w + 42)
        y = 96 + (idx // 2) * (cell_h + 58)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=8, fill=(255, 255, 255), outline=(210, 210, 210), width=1)
        draw.text((x + 14, y + 12), panel["title"], font=FONT_H2, fill=(20, 20, 20))
        view, scale = _fit(source, 430, 300)
        vx, vy = x + 14, y + 54
        canvas.paste(view, (vx, vy))
        draw.rectangle((vx, vy, vx + view.width - 1, vy + view.height - 1), outline=(190, 190, 190), width=1)
        crop_norm = panel["crop"]
        crop_xyxy = _box_to_px(crop_norm, width, height)
        _draw_px_rect(draw, crop_xyxy, (vx, vy), scale, (255, 45, 45), 5)
        if pose_box:
            _draw_px_rect(draw, _box_to_px(pose_box, width, height), (vx, vy), scale, (255, 170, 0), 3)
        for pt, color in [(eye, (255, 230, 0)), (control, (180, 80, 255)), (support, (255, 90, 0))]:
            if pt is not None:
                _draw_px_point(draw, (pt[0] * width, pt[1] * height), (vx, vy), scale, color, 7)
        if support is not None:
            x1, y1, x2, y2 = crop_xyxy
            target_y = y1 + 0.95 * (y2 - y1)
            draw.line((vx + x1 * scale, vy + target_y * scale, vx + x2 * scale, vy + target_y * scale), fill=(255, 90, 0), width=3)
            draw.text((vx + x1 * scale + 4, vy + target_y * scale - 18), "support target y=0.95", font=FONT_SMALL, fill=(255, 90, 0))
        tx = x + 466
        ty = y + 56
        _draw_lines(draw, (tx, ty), _wrap(panel["text"], 285, FONT), font=FONT, fill=(50, 50, 50), gap=6)
    reason_counter = Counter()
    for row in status_rows:
        if str(row.get("mode_name")) not in {"single_person_center", "single_person_rot"}:
            continue
        for part in str(row.get("no_positive_reason") or "").split(","):
            part = part.strip()
            if part:
                reason_counter[part] += 1
    summary = ", ".join(f"{k}:{v}" for k, v in reason_counter.most_common())
    _draw_lines(draw, (24, 995), _wrap(f"Actual Full v14 status reason counts for person queries: {summary}", 1560, FONT), font=FONT, fill=(30, 30, 30), gap=5)
    canvas.save(out_path, quality=92)


def build_person_gate_visuals(full_label: dict[str, Any], full_images_dir: Path, full_status_path: Path, full_precompute_path: Path, out_dir: Path) -> dict[str, Any]:
    images = {int(im["id"]): im for im in full_label["images"]}
    source_to_image = {_source_id(im): im for im in full_label["images"]}
    source_id = "sstk_image_1197510367"
    image_row = source_to_image[source_id]
    status_rows = _load_status_rows(full_status_path, source_id)
    precompute_extra = _load_precompute_pose(full_precompute_path, source_id, int(image_row["width"]), int(image_row["height"]))
    out = out_dir / "person_gate_sstk_image_1197510367_support_margin_coverage.jpg"
    draw_person_gate_panel(
        image_path=_resolve_image_path(full_images_dir, image_row),
        image_row=image_row,
        status_rows=status_rows,
        precompute_extra=precompute_extra,
        out_path=out,
    )
    rows = []
    for row in status_rows:
        if str(row.get("mode_name")) in {"single_person_center", "single_person_rot"}:
            rows.append(
                {
                    "source_image_id": source_id,
                    "target_ar": row.get("target_ar"),
                    "mode_name": row.get("mode_name"),
                    "best_score": row.get("best_score"),
                    "candidate_count": row.get("candidate_count"),
                    "negative_count": row.get("negative_count"),
                    "no_positive_reason": row.get("no_positive_reason"),
                }
            )
    _write_csv(out_dir / "person_gate_sstk_image_1197510367_status_rows.csv", rows)
    return {"panel": out.name, "rows": len(rows)}


def _ann_crop_contains(ann: dict[str, Any], image_row: dict[str, Any], pt: tuple[float, float] | None) -> bool:
    if pt is None:
        return False
    crop = _crop_norm_from_ann(ann, int(image_row["width"]), int(image_row["height"]))
    return _contains(crop, pt)


def audit_gaze_anchors(label: dict[str, Any], images_dir: Path, out_dir: Path) -> dict[str, Any]:
    images = {int(im["id"]): im for im in label["images"]}
    all_rows: list[dict[str, Any]] = []
    suspect_rows: list[dict[str, Any]] = []
    ann_by_id: dict[int, dict[str, Any]] = {}
    issue_counter: Counter[str] = Counter()

    for ann in label["annotations"]:
        mode = str(ann.get("mode_name") or "")
        if mode not in PERSON_MODES:
            continue
        ann_by_id[int(ann["id"])] = ann
        image_row = images[int(ann["image_id"])]
        info = _ann_info(ann)
        attrs = info["attrs"]
        comps = info["components"]
        subj = info["subject_debug"]
        portrait = info["portrait_comp"]
        sid = _source_id(image_row)
        eye = _norm_point(portrait.get("eye_point_norm_xy"))
        gaze_dir = str(comps.get("gaze_dir") or portrait.get("direction_hint") or "unknown")
        direction_hint = str(portrait.get("direction_hint") or "unknown")
        lookroom_value = comps.get("lookroom_value")
        face_box = _norm_box(subj.get("face_bbox_norm_xyxy"))
        head_box = _norm_box(subj.get("head_bbox_norm_xyxy"))
        core_box = _norm_box(subj.get("core_bbox_norm_xyxy"))
        member_faces = subj.get("member_face_boxes_norm_xyxy") if isinstance(subj.get("member_face_boxes_norm_xyxy"), list) else []
        issues: list[str] = []
        if sid in GUIDE_PROBLEM_SOURCES:
            issues.append("legacy_guide_sample")
        if gaze_dir in {"left", "right"} and eye is None:
            issues.append("side_gaze_missing_eye_anchor")
        if eye is not None and not _ann_crop_contains(ann, image_row, eye):
            issues.append("eye_anchor_outside_crop")
        if eye is not None and face_box is not None and not _contains(face_box, eye) and not _contains(head_box, eye):
            issues.append("eye_anchor_outside_face_head")
        if mode.startswith("group") and gaze_dir in {"left", "right"} and len(member_faces) > 1:
            issues.append("group_aggregate_gaze_anchor")
        if gaze_dir in {"left", "right"} and direction_hint in {"left", "right"} and gaze_dir != direction_hint:
            issues.append("gaze_direction_hint_conflict")
        lv = _safe_float(lookroom_value, float("nan"))
        if math.isfinite(lv) and (lv < 0.35 or lv > 5.0):
            issues.append("lookroom_extreme_ratio")
        if gaze_dir in {"left", "right"} and eye is not None and face_box is None and head_box is None:
            issues.append("side_gaze_without_face_head_box")
        if issues:
            for issue in issues:
                issue_counter[issue] += 1
        row = {
            "source_image_id": sid,
            "annotation_id": ann.get("id"),
            "mode_name": mode,
            "target_ar": attrs.get("target_ar"),
            "gt_flag": ann.get("gt_flag"),
            "score_mode": ann.get("score_mode"),
            "gaze_dir": gaze_dir,
            "direction_hint": direction_hint,
            "lookroom_value": lookroom_value,
            "eye_point_norm_xy": eye,
            "eye_inside_crop": int(_ann_crop_contains(ann, image_row, eye)),
            "eye_inside_face": int(_contains(face_box, eye)),
            "eye_inside_head": int(_contains(head_box, eye)),
            "eye_inside_core": int(_contains(core_box, eye)),
            "member_face_count": len(member_faces),
            "issues": issues,
        }
        all_rows.append(row)
        if issues:
            suspect_rows.append(row)

    _write_csv(out_dir / "gaze_anchor_audit_all_rows.csv", all_rows)
    _write_csv(out_dir / "gaze_anchor_audit_suspect_rows.csv", suspect_rows)

    selected_rows: list[dict[str, Any]] = []
    selected_sids: set[str] = set()
    selected_issues: set[str] = set()

    def add_row(row: dict[str, Any], *, allow_same_source: bool = False) -> None:
        sid = str(row.get("source_image_id"))
        ann_id = int(row.get("annotation_id"))
        if ann_id not in ann_by_id:
            return
        if not allow_same_source and sid in selected_sids:
            return
        selected_rows.append(row)
        selected_sids.add(sid)
        for issue in row.get("issues") or []:
            selected_issues.add(str(issue))

    for guide_sid in GUIDE_PROBLEM_SOURCES:
        guide_rows = [row for row in suspect_rows if row.get("source_image_id") == guide_sid]
        if not guide_rows:
            guide_rows = [row for row in all_rows if row.get("source_image_id") == guide_sid]
        if guide_rows:
            add_row(guide_rows[0], allow_same_source=False)

    for issue in issue_counter:
        if issue in selected_issues:
            continue
        rows_for_issue = [row for row in suspect_rows if issue in (row.get("issues") or []) and row.get("source_image_id") not in selected_sids]
        if rows_for_issue:
            add_row(rows_for_issue[0], allow_same_source=False)

    for row in suspect_rows:
        if len(selected_rows) >= 12:
            break
        add_row(row, allow_same_source=False)

    panel_paths: list[Path] = []
    seen_ann: set[int] = set()
    for row in selected_rows[:12]:
        ann = ann_by_id[int(row["annotation_id"])]
        issues = [str(x) for x in row.get("issues") or []]
        ann_id = int(ann["id"])
        if ann_id in seen_ann:
            continue
        seen_ann.add(ann_id)
        image_row = images[int(ann["image_id"])]
        sid = _source_id(image_row)
        out = out_dir / f"gaze_anchor_audit_{sid}_{ann_id}.jpg"
        draw_annotation_panel(
            image_path=_resolve_image_path(images_dir, image_row),
            image_row=image_row,
            ann=ann,
            reason=None,
            title=f"Gaze anchor audit: {sid}",
            explanation="This panel visualizes the gaze anchor used by lookroom. Yellow dot/arrow is the current anchor/direction; cyan/blue boxes show face/head/core when available.",
            out_path=out,
            extra_lines=[
                f"issues: {', '.join(issues)}",
                f"guide note: {GUIDE_PROBLEM_SOURCES.get(sid, '-')}",
            ],
        )
        panel_paths.append(out)
    make_contact_sheet(panel_paths, out_dir / "gaze_anchor_suspect_contact_sheet.jpg", title="Gaze anchor suspect examples")

    summary = {
        "total_person_mode_annotations": len(all_rows),
        "suspect_rows": len(suspect_rows),
        "issue_counts": dict(issue_counter.most_common()),
        "legacy_guide_sources": GUIDE_PROBLEM_SOURCES,
        "outputs": {
            "all_rows_csv": "gaze_anchor_audit_all_rows.csv",
            "suspect_rows_csv": "gaze_anchor_audit_suspect_rows.csv",
            "contact_sheet": "gaze_anchor_suspect_contact_sheet.jpg",
        },
    }
    with (out_dir / "gaze_anchor_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual explanations for review_jy_260604 follow-up report.")
    parser.add_argument("--phase_run_dir", type=Path, default=Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010"))
    parser.add_argument("--phase_images_dir", type=Path, default=Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/images"))
    parser.add_argument("--full_run_dir", type=Path, default=Path("data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010"))
    parser.add_argument("--full_images_dir", type=Path, default=Path("data/SSTK/Full_10000/images"))
    parser.add_argument("--full_precompute_path", type=Path, default=Path("data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_c7_saliency.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.phase_run_dir / "crop_review_v14_photo_primary/review_jy_260604_followup/assets_reason_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_label = _read_json(args.phase_run_dir / "label_json/multimode_labels_full.json")
    full_label = _read_json(args.full_run_dir / "label_json/multimode_labels_full.json")

    face_rows = build_face_reason_visuals(phase_label, args.phase_images_dir, out_dir)
    person_summary = build_person_gate_visuals(
        full_label,
        args.full_images_dir,
        args.full_run_dir / "mode_query_status.jsonl",
        args.full_precompute_path,
        out_dir,
    )
    gaze_summary = audit_gaze_anchors(phase_label, args.phase_images_dir, out_dir)

    status = {
        "output_dir": str(out_dir),
        "face_reason_rows": face_rows,
        "person_gate_summary": person_summary,
        "gaze_anchor_summary": gaze_summary,
    }
    with (out_dir / "visual_explanations_summary.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
