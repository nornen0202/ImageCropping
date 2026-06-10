#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


PERSON_MODES = {"single_person_center", "single_person_rot", "group_center", "group_rot", "face"}
GUIDE_RUN = "260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


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
FONT = _font(18)
FONT_SMALL = _font(15)
FONT_MONO = _font(15)


def _wrap_text(text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else f"{cur} {word}"
        if font.getlength(nxt) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_lines(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: Iterable[str], *, font: ImageFont.ImageFont, fill: tuple[int, int, int], line_gap: int = 6) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += int(font.size) + line_gap
    return y


def _image_xyxy_from_norm(box: list[float] | tuple[float, ...] | None, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not box or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return x1 * width, y1 * height, x2 * width, y2 * height


def _crop_norm_from_ann(ann: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    return (
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, (x + w) / width)),
        max(0.0, min(1.0, (y + h) / height)),
    )


def _fit_image(image: Image.Image, box_w: int, box_h: int) -> tuple[Image.Image, tuple[int, int], float]:
    scale = min(box_w / image.width, box_h / image.height)
    out_w = max(1, int(round(image.width * scale)))
    out_h = max(1, int(round(image.height * scale)))
    resized = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return resized, ((box_w - out_w) // 2, (box_h - out_h) // 2), scale


def _draw_rect(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], offset: tuple[int, int], scale: float, color: tuple[int, int, int], width: int = 4) -> None:
    x1, y1, x2, y2 = box
    ox, oy = offset
    draw.rectangle((ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale), outline=color, width=width)


def _draw_point(draw: ImageDraw.ImageDraw, pt: tuple[float, float], offset: tuple[int, int], scale: float, color: tuple[int, int, int], radius: int = 6) -> None:
    x, y = pt
    ox, oy = offset
    cx, cy = ox + x * scale, oy + y * scale
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(20, 20, 20), width=2)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], offset: tuple[int, int], scale: float, color: tuple[int, int, int], width: int = 4) -> None:
    sx, sy = start
    ex, ey = end
    ox, oy = offset
    p1 = (ox + sx * scale, oy + sy * scale)
    p2 = (ox + ex * scale, oy + ey * scale)
    draw.line((p1, p2), fill=color, width=width)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    size = 14
    head = [
        p2,
        (p2[0] - ux * size + px * size * 0.45, p2[1] - uy * size + py * size * 0.45),
        (p2[0] - ux * size - px * size * 0.45, p2[1] - uy * size - py * size * 0.45),
    ]
    draw.polygon(head, fill=color)


def _draw_safe_rect(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], *, fill: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = xy
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    if right <= left or bottom <= top:
        return
    draw.rectangle((left, top, right, bottom), fill=fill)


def _ann_hard_rejects(ann: dict[str, Any]) -> list[str]:
    attrs = ann.get("attributes") or {}
    reasons = ann.get("hard_reject_reasons")
    if reasons is None:
        reasons = attrs.get("hard_reject_reasons")
    return [str(x) for x in (reasons or [])]


def _ann_info(ann: dict[str, Any]) -> dict[str, Any]:
    attrs = ann.get("attributes") or {}
    comps = attrs.get("score_components") or {}
    labels = attrs.get("checklist_labels") or {}
    scores = attrs.get("checklist_scores") or {}
    applicable = attrs.get("checklist_applicable") or {}
    why = [str(x) for x in attrs.get("why_tags") or []]
    return {
        "target_ar": attrs.get("target_ar"),
        "labels": labels,
        "scores": scores,
        "applicable": applicable,
        "why": why,
        "components": comps,
        "subject_debug": attrs.get("subject_debug") or {},
        "portrait_comp": attrs.get("portrait_comp") or {},
        "candidate_source": attrs.get("candidate_source", ""),
        "candidate_id": attrs.get("candidate_id", ""),
        "route_mode": attrs.get("route_mode", ""),
        "hard_rejects": _ann_hard_rejects(ann),
    }


def _score_candidate(case_id: str, ann: dict[str, Any]) -> tuple[float, ...]:
    info = _ann_info(ann)
    comps = info["components"]
    labels = info["labels"]
    score = _safe_float(ann.get("score_mode"), 0.0) or 0.0
    q_subj = _safe_float(comps.get("q_subj"), 0.0) or 0.0
    q_head = _safe_float(comps.get("q_head"), 0.0) or 0.0
    q_look = _safe_float(comps.get("q_look"), 0.0) or 0.0
    hr = _safe_float(comps.get("headroom_value"), 0.0) or 0.0
    lr = _safe_float(comps.get("lookroom_value"), 0.0) or 0.0
    no_reject = 1.0 if not info["hard_rejects"] else 0.0
    is_best = 1.0 if ann.get("is_best") else 0.0
    if case_id == "headroom_tight":
        return (no_reject, is_best, q_subj, -abs(hr - 0.02), q_head, score)
    if case_id == "headroom_loose":
        return (no_reject, is_best, q_subj, -abs(hr - 0.24), q_head, score)
    if case_id == "lookroom_insufficient":
        return (no_reject, is_best, q_subj, -lr, score)
    if case_id == "lookroom_excessive":
        return (no_reject, is_best, q_subj, lr, score)
    if case_id == "lookroom_not_applicable":
        gaze_known = 1.0 if str(comps.get("gaze_dir") or "") in {"left", "right"} else 0.0
        return (no_reject, is_best, q_subj, -gaze_known, q_head, score)
    if labels.get("headroom") == "headroom_ok" and labels.get("lookroom") == "lookroom_adequate":
        return (no_reject, is_best, q_subj, q_head + q_look, score)
    return (no_reject, is_best, q_subj, q_head + q_look, score)


def _matches_case(case: dict[str, Any], ann: dict[str, Any]) -> bool:
    if ann.get("gt_flag") != 1:
        return False
    mode = str(ann.get("mode_name") or "")
    if mode not in case["modes"]:
        return False
    info = _ann_info(ann)
    labels = info["labels"]
    comps = info["components"]
    if case.get("headroom") is not None and labels.get("headroom") != case["headroom"]:
        return False
    if case.get("lookroom") is not None and labels.get("lookroom") != case["lookroom"]:
        return False
    if case.get("gaze_unknown") and str(comps.get("gaze_dir") or "unknown") in {"left", "right"}:
        return False
    if case.get("requires_eye") and not info["portrait_comp"].get("eye_point_norm_xy"):
        return False
    return True


def select_examples(data: dict[str, Any], image_root: Path) -> list[dict[str, Any]]:
    images = {int(im["id"]): im for im in data["images"]}
    anns = data["annotations"]
    cases = [
        {
            "case_id": "face_balanced_headroom_lookroom",
            "title": "Face mode: balanced headroom and adequate lookroom",
            "modes": {"face"},
            "headroom": "headroom_ok",
            "lookroom": "lookroom_adequate",
            "requires_eye": True,
        },
        {
            "case_id": "single_person_balanced_headroom_lookroom",
            "title": "Single-person mode: balanced headroom and adequate lookroom",
            "modes": {"single_person_center", "single_person_rot"},
            "headroom": "headroom_ok",
            "lookroom": "lookroom_adequate",
            "requires_eye": True,
        },
        {
            "case_id": "group_balanced_headroom_lookroom",
            "title": "Group mode: shared envelope with adequate lookroom",
            "modes": {"group_center", "group_rot"},
            "headroom": "headroom_ok",
            "lookroom": "lookroom_adequate",
            "requires_eye": True,
        },
        {
            "case_id": "headroom_tight",
            "title": "Headroom tight: crop top is close to the head line",
            "modes": PERSON_MODES,
            "headroom": "headroom_tight",
            "lookroom": None,
        },
        {
            "case_id": "headroom_loose",
            "title": "Headroom loose: top margin is larger than the mode threshold",
            "modes": PERSON_MODES,
            "headroom": "headroom_loose",
            "lookroom": None,
        },
        {
            "case_id": "lookroom_insufficient",
            "title": "Lookroom insufficient: gaze-forward margin is too short",
            "modes": PERSON_MODES,
            "headroom": None,
            "lookroom": "lookroom_insufficient",
            "requires_eye": True,
        },
        {
            "case_id": "lookroom_excessive",
            "title": "Lookroom excessive: gaze-forward margin dominates the crop",
            "modes": PERSON_MODES,
            "headroom": None,
            "lookroom": "lookroom_excessive",
            "requires_eye": True,
        },
        {
            "case_id": "lookroom_not_applicable",
            "title": "Lookroom not applicable: no reliable side-gaze direction",
            "modes": PERSON_MODES,
            "headroom": "headroom_ok",
            "lookroom": "na",
            "gaze_unknown": True,
        },
    ]
    selected: list[dict[str, Any]] = []
    used_images: set[int] = set()
    for case in cases:
        candidates = []
        for ann in anns:
            image_id = int(ann["image_id"])
            im = images[image_id]
            if not (image_root / im["file_name"]).exists():
                continue
            if not _matches_case(case, ann):
                continue
            penalty = -1.0 if image_id in used_images else 0.0
            candidates.append((_score_candidate(case["case_id"], ann) + (penalty,), ann))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        ann = candidates[0][1]
        used_images.add(int(ann["image_id"]))
        selected.append({"case": case, "annotation": ann, "image": images[int(ann["image_id"])]})
    return selected


def _draw_sample_panel(sample: dict[str, Any], image_root: Path, out_path: Path) -> dict[str, Any]:
    ann = sample["annotation"]
    imrec = sample["image"]
    case = sample["case"]
    info = _ann_info(ann)
    width, height = int(imrec["width"]), int(imrec["height"])
    image_path = image_root / imrec["file_name"]
    src = Image.open(image_path).convert("RGB")
    crop_norm = _crop_norm_from_ann(ann, width, height)
    crop_px = (
        int(round(crop_norm[0] * src.width)),
        int(round(crop_norm[1] * src.height)),
        int(round(crop_norm[2] * src.width)),
        int(round(crop_norm[3] * src.height)),
    )
    crop_px = (
        max(0, min(src.width - 1, crop_px[0])),
        max(0, min(src.height - 1, crop_px[1])),
        max(1, min(src.width, crop_px[2])),
        max(1, min(src.height, crop_px[3])),
    )
    crop_img = src.crop(crop_px)

    canvas = Image.new("RGB", (1800, 1020), (248, 249, 251))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, 1800, 66), fill=(34, 42, 54, 255))
    draw.text((28, 18), case["title"], font=FONT_TITLE, fill=(255, 255, 255, 255))

    full_box = (40, 104, 820, 684)
    crop_box = (860, 104, 1340, 684)
    text_box = (1380, 100, 1760, 940)
    for box, title in [(full_box, "Original + geometry"), (crop_box, "Actual crop result")]:
        x1, y1, x2, y2 = box
        draw.rounded_rectangle((x1 - 12, y1 - 42, x2 + 12, y2 + 12), radius=8, fill=(255, 255, 255, 255), outline=(210, 215, 222, 255), width=1)
        draw.text((x1, y1 - 34), title, font=FONT_H2, fill=(33, 37, 41, 255))

    full_thumb, full_off, full_scale = _fit_image(src, full_box[2] - full_box[0], full_box[3] - full_box[1])
    full_origin = (full_box[0] + full_off[0], full_box[1] + full_off[1])
    canvas.paste(full_thumb, full_origin)
    overlay = ImageDraw.Draw(canvas, "RGBA")
    crop_full_px = (crop_norm[0] * width, crop_norm[1] * height, crop_norm[2] * width, crop_norm[3] * height)
    _draw_rect(overlay, crop_full_px, full_origin, full_scale, (230, 57, 70), 5)

    subj = info["subject_debug"]
    head_box = _image_xyxy_from_norm(subj.get("head_bbox_norm_xyxy") or subj.get("face_bbox_norm_xyxy"), width, height)
    face_box = _image_xyxy_from_norm(subj.get("face_bbox_norm_xyxy"), width, height)
    if face_box:
        _draw_rect(overlay, face_box, full_origin, full_scale, (46, 196, 182), 3)
    if head_box:
        _draw_rect(overlay, head_box, full_origin, full_scale, (42, 111, 219), 4)

    portrait = info["portrait_comp"]
    comps = info["components"]
    eye_norm = portrait.get("eye_point_norm_xy")
    gaze_dir = str(comps.get("gaze_dir") or portrait.get("direction_hint") or "unknown")
    eye_px: tuple[float, float] | None = None
    if isinstance(eye_norm, list) and len(eye_norm) >= 2:
        eye_px = (float(eye_norm[0]) * width, float(eye_norm[1]) * height)
        _draw_point(overlay, eye_px, full_origin, full_scale, (255, 193, 7), 7)
        if gaze_dir in {"left", "right"}:
            direction = -1.0 if gaze_dir == "left" else 1.0
            end_x = max(0.0, min(float(width), eye_px[0] + direction * width * 0.14))
            _draw_arrow(overlay, eye_px, (end_x, eye_px[1]), full_origin, full_scale, (255, 193, 7), 5)

    if head_box:
        head_top = head_box[1]
    else:
        hv = _safe_float(comps.get("headroom_value"), None)
        head_top = crop_full_px[1] + hv * (crop_full_px[3] - crop_full_px[1]) if hv is not None else None
    if head_top is not None:
        x1, y1, x2, _ = crop_full_px
        head_line = max(float(y1), min(float(head_top), float(crop_full_px[3])))
        _draw_safe_rect(
            overlay,
            (
                full_origin[0] + x1 * full_scale,
                full_origin[1] + y1 * full_scale,
                full_origin[0] + x2 * full_scale,
                full_origin[1] + head_line * full_scale,
            ),
            fill=(42, 111, 219, 42),
        )
        overlay.line(
            (
                full_origin[0] + x1 * full_scale,
                full_origin[1] + head_line * full_scale,
                full_origin[0] + x2 * full_scale,
                full_origin[1] + head_line * full_scale,
            ),
            fill=(42, 111, 219, 220),
            width=4,
        )

    if eye_px and gaze_dir in {"left", "right"}:
        x1, y1, x2, _ = crop_full_px
        ax = max(x1, min(x2, eye_px[0]))
        y = full_origin[1] + (eye_px[1] + 22) * full_scale
        fwd = (ax, x2) if gaze_dir == "right" else (x1, ax)
        back = (x1, ax) if gaze_dir == "right" else (ax, x2)
        overlay.line((full_origin[0] + back[0] * full_scale, y, full_origin[0] + back[1] * full_scale, y), fill=(130, 130, 130, 210), width=5)
        overlay.line((full_origin[0] + fwd[0] * full_scale, y, full_origin[0] + fwd[1] * full_scale, y), fill=(39, 174, 96, 230), width=7)

    crop_thumb, crop_off, crop_scale = _fit_image(crop_img, crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    crop_origin = (crop_box[0] + crop_off[0], crop_box[1] + crop_off[1])
    canvas.paste(crop_thumb, crop_origin)
    overlay = ImageDraw.Draw(canvas, "RGBA")
    if head_box:
        rel_head = (
            head_box[0] - crop_px[0],
            head_box[1] - crop_px[1],
            head_box[2] - crop_px[0],
            head_box[3] - crop_px[1],
        )
        rel_head = (
            max(0.0, min(crop_img.width, rel_head[0])),
            max(0.0, min(crop_img.height, rel_head[1])),
            max(0.0, min(crop_img.width, rel_head[2])),
            max(0.0, min(crop_img.height, rel_head[3])),
        )
        if rel_head[2] > rel_head[0] and rel_head[3] > rel_head[1]:
            _draw_rect(overlay, rel_head, crop_origin, crop_scale, (42, 111, 219), 4)
            _draw_safe_rect(
                overlay,
                (
                    crop_origin[0],
                    crop_origin[1],
                    crop_origin[0] + crop_img.width * crop_scale,
                    crop_origin[1] + rel_head[1] * crop_scale,
                ),
                fill=(42, 111, 219, 46),
            )
            overlay.line(
                (
                    crop_origin[0],
                    crop_origin[1] + rel_head[1] * crop_scale,
                    crop_origin[0] + crop_img.width * crop_scale,
                    crop_origin[1] + rel_head[1] * crop_scale,
                ),
                fill=(42, 111, 219, 230),
                width=4,
            )
    if eye_px and gaze_dir in {"left", "right"}:
        rel_eye = (eye_px[0] - crop_px[0], eye_px[1] - crop_px[1])
        if 0 <= rel_eye[0] <= crop_img.width and 0 <= rel_eye[1] <= crop_img.height:
            _draw_point(overlay, rel_eye, crop_origin, crop_scale, (255, 193, 7), 7)
            direction = -1.0 if gaze_dir == "left" else 1.0
            end_x = max(0.0, min(float(crop_img.width), rel_eye[0] + direction * crop_img.width * 0.22))
            _draw_arrow(overlay, rel_eye, (end_x, rel_eye[1]), crop_origin, crop_scale, (255, 193, 7), 5)
            y = crop_origin[1] + (rel_eye[1] + 22) * crop_scale
            fwd = (rel_eye[0], crop_img.width) if gaze_dir == "right" else (0.0, rel_eye[0])
            back = (0.0, rel_eye[0]) if gaze_dir == "right" else (rel_eye[0], crop_img.width)
            overlay.line((crop_origin[0] + back[0] * crop_scale, y, crop_origin[0] + back[1] * crop_scale, y), fill=(130, 130, 130, 210), width=5)
            overlay.line((crop_origin[0] + fwd[0] * crop_scale, y, crop_origin[0] + fwd[1] * crop_scale, y), fill=(39, 174, 96, 230), width=7)

    labels = info["labels"]
    scores = info["scores"]
    applicable = info["applicable"]
    y = text_box[1]
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((text_box[0] - 16, y - 12, text_box[2] + 16, text_box[3]), radius=8, fill=(255, 255, 255, 255), outline=(210, 215, 222, 255), width=1)
    text_lines = [
        f"image: {imrec['source_image_id']}",
        f"mode: {ann.get('mode_name')}  AR: {info['target_ar']}",
        f"score_mode: {float(ann.get('score_mode', 0.0)):.6f}",
        f"gt_flag: {ann.get('gt_flag')}  is_best: {ann.get('is_best')}",
        "",
        f"headroom label: {labels.get('headroom')}",
        f"headroom applicable: {applicable.get('headroom')}",
        f"headroom_value: {comps.get('headroom_value')}",
        f"q_head / C_headroom: {comps.get('q_head')} / {scores.get('C_headroom')}",
        "",
        f"lookroom label: {labels.get('lookroom')}",
        f"lookroom applicable: {applicable.get('lookroom')}",
        f"gaze_dir: {gaze_dir}",
        f"lookroom_value: {comps.get('lookroom_value')}",
        f"q_look / C_lookroom: {comps.get('q_look')} / {scores.get('C_lookroom')}",
        "",
        f"why_tags: {', '.join(info['why']) if info['why'] else '(none)'}",
        f"rejects: {', '.join(info['hard_rejects']) if info['hard_rejects'] else '(none)'}",
        f"candidate: {info['candidate_source']}",
    ]
    for line in text_lines:
        if line == "":
            y += 12
            continue
        wrapped = _wrap_text(line, text_box[2] - text_box[0], FONT_MONO)
        y = _draw_lines(draw, (text_box[0], y), wrapped, font=FONT_MONO, fill=(32, 37, 43), line_gap=4)

    legend_y = 736
    legend_lines = [
        "Legend: red=crop, blue=head/headroom, cyan=face, yellow=eye/gaze, green=gaze-forward lookroom, gray=gaze-back margin.",
        "Headroom band is crop top to detected head top. Lookroom ratio is forward margin / backward margin from the gaze anchor.",
    ]
    draw.rounded_rectangle((40, legend_y - 18, 1340, 946), radius=8, fill=(255, 255, 255, 255), outline=(210, 215, 222, 255), width=1)
    y = legend_y
    for line in legend_lines:
        y = _draw_lines(draw, (64, y), _wrap_text(line, 1230, FONT), font=FONT, fill=(55, 65, 75), line_gap=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=94)
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "source_image_id": imrec["source_image_id"],
        "image_id": ann["image_id"],
        "annotation_id": ann["id"],
        "mode_name": ann.get("mode_name"),
        "target_ar": info["target_ar"],
        "score_mode": ann.get("score_mode"),
        "headroom_label": labels.get("headroom"),
        "headroom_value": comps.get("headroom_value"),
        "q_head": comps.get("q_head"),
        "lookroom_label": labels.get("lookroom"),
        "lookroom_value": comps.get("lookroom_value"),
        "q_look": comps.get("q_look"),
        "gaze_dir": gaze_dir,
        "why_tags": info["why"],
        "panel_path": str(out_path),
        "source_image": str(image_path),
    }


def _gaussian_quality(value: float, target: float, sigma: float) -> float:
    sigma = max(1e-6, float(sigma))
    return float(math.exp(-((float(value) - float(target)) ** 2) / (2.0 * sigma * sigma)))


def _single_person_headroom_label(value: float) -> str:
    if value < 0.04:
        return "headroom_tight"
    if value > 0.20:
        return "headroom_loose"
    return "headroom_ok"


def _lookroom_label(value: float) -> str:
    if value < 1.05:
        return "lookroom_insufficient"
    if value > 2.50:
        return "lookroom_excessive"
    return "lookroom_adequate"


def _same_image_variant_metrics(
    *,
    crop: tuple[float, float, float, float],
    head_y: float,
    gaze_anchor_x: float,
    gaze_y: float,
) -> dict[str, Any]:
    x1, y1, x2, y2 = crop
    crop_h = max(1e-8, y2 - y1)
    headroom = (float(head_y) - y1) / crop_h
    sigma = 0.06 if headroom <= 0.12 else 0.14
    q_head = _gaussian_quality(headroom, 0.12, sigma)
    ax = max(x1, min(x2, float(gaze_anchor_x)))
    back = max(1e-8, ax - x1)
    fwd = max(1e-8, x2 - ax)
    lookroom = (fwd + 1e-6) / (back + 1e-6)
    q_look = _gaussian_quality(lookroom, 1.45, 0.45)
    head_label = _single_person_headroom_label(headroom)
    look_label = _lookroom_label(lookroom)
    tags: list[str] = []
    if head_label == "headroom_ok":
        tags.extend(["headroom_ok", "head_top_safe"])
    else:
        tags.append("headroom_violation")
    tags.append("lookroom_ok" if look_label == "lookroom_adequate" else "lookroom_violation")
    return {
        "crop": [round(float(v), 3) for v in crop],
        "headroom_value": round(float(headroom), 6),
        "q_head": round(float(q_head), 6),
        "headroom_label": head_label,
        "lookroom_value": round(float(lookroom), 6),
        "q_look": round(float(q_look), 6),
        "lookroom_label": look_label,
        "gaze_dir": "right",
        "why_tags": tags,
        "head_y": round(float(head_y), 3),
        "gaze_anchor_x": round(float(gaze_anchor_x), 3),
        "gaze_y": round(float(gaze_y), 3),
    }


def _draw_same_image_variant_panel(
    *,
    src: Image.Image,
    source_image_id: str,
    title: str,
    note: str,
    crop: tuple[float, float, float, float],
    head_y: float,
    gaze_anchor_x: float,
    gaze_y: float,
    out_path: Path,
) -> dict[str, Any]:
    metrics = _same_image_variant_metrics(crop=crop, head_y=head_y, gaze_anchor_x=gaze_anchor_x, gaze_y=gaze_y)
    crop_px = (
        max(0, min(src.width - 1, int(round(crop[0])))),
        max(0, min(src.height - 1, int(round(crop[1])))),
        max(1, min(src.width, int(round(crop[2])))),
        max(1, min(src.height, int(round(crop[3])))),
    )
    crop_img = src.crop(crop_px)
    canvas = Image.new("RGB", (1700, 940), (248, 249, 251))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, 1700, 66), fill=(34, 42, 54, 255))
    draw.text((28, 18), title, font=FONT_TITLE, fill=(255, 255, 255, 255))

    full_box = (40, 112, 760, 650)
    crop_box = (810, 112, 1240, 650)
    text_box = (1280, 112, 1660, 850)
    for box, label in ((full_box, "Original + crop geometry"), (crop_box, "Crop result")):
        x1, y1, x2, y2 = box
        draw.rounded_rectangle((x1 - 12, y1 - 42, x2 + 12, y2 + 12), radius=8, fill=(255, 255, 255, 255), outline=(210, 215, 222, 255), width=1)
        draw.text((x1, y1 - 34), label, font=FONT_H2, fill=(33, 37, 41, 255))

    full_thumb, full_off, full_scale = _fit_image(src, full_box[2] - full_box[0], full_box[3] - full_box[1])
    full_origin = (full_box[0] + full_off[0], full_box[1] + full_off[1])
    canvas.paste(full_thumb, full_origin)
    overlay = ImageDraw.Draw(canvas, "RGBA")
    _draw_rect(overlay, crop, full_origin, full_scale, (230, 57, 70), 5)
    x1, y1, x2, y2 = crop
    head_line = max(y1, min(float(head_y), y2))
    _draw_safe_rect(
        overlay,
        (
            full_origin[0] + x1 * full_scale,
            full_origin[1] + y1 * full_scale,
            full_origin[0] + x2 * full_scale,
            full_origin[1] + head_line * full_scale,
        ),
        fill=(42, 111, 219, 46),
    )
    overlay.line(
        (
            full_origin[0] + x1 * full_scale,
            full_origin[1] + head_line * full_scale,
            full_origin[0] + x2 * full_scale,
            full_origin[1] + head_line * full_scale,
        ),
        fill=(42, 111, 219, 230),
        width=4,
    )
    _draw_point(overlay, (gaze_anchor_x, gaze_y), full_origin, full_scale, (255, 193, 7), 7)
    _draw_arrow(overlay, (gaze_anchor_x, gaze_y), (min(src.width, gaze_anchor_x + src.width * 0.14), gaze_y), full_origin, full_scale, (255, 193, 7), 5)
    ax = max(x1, min(x2, gaze_anchor_x))
    y_margin = full_origin[1] + (gaze_y + 24) * full_scale
    overlay.line((full_origin[0] + x1 * full_scale, y_margin, full_origin[0] + ax * full_scale, y_margin), fill=(130, 130, 130, 210), width=5)
    overlay.line((full_origin[0] + ax * full_scale, y_margin, full_origin[0] + x2 * full_scale, y_margin), fill=(39, 174, 96, 230), width=7)

    crop_thumb, crop_off, crop_scale = _fit_image(crop_img, crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    crop_origin = (crop_box[0] + crop_off[0], crop_box[1] + crop_off[1])
    canvas.paste(crop_thumb, crop_origin)
    overlay = ImageDraw.Draw(canvas, "RGBA")
    rel_head_y = max(0.0, min(float(crop_img.height), float(head_y) - crop_px[1]))
    _draw_safe_rect(
        overlay,
        (
            crop_origin[0],
            crop_origin[1],
            crop_origin[0] + crop_img.width * crop_scale,
            crop_origin[1] + rel_head_y * crop_scale,
        ),
        fill=(42, 111, 219, 46),
    )
    overlay.line(
        (
            crop_origin[0],
            crop_origin[1] + rel_head_y * crop_scale,
            crop_origin[0] + crop_img.width * crop_scale,
            crop_origin[1] + rel_head_y * crop_scale,
        ),
        fill=(42, 111, 219, 230),
        width=4,
    )
    rel_anchor = (gaze_anchor_x - crop_px[0], gaze_y - crop_px[1])
    if 0 <= rel_anchor[0] <= crop_img.width and 0 <= rel_anchor[1] <= crop_img.height:
        _draw_point(overlay, rel_anchor, crop_origin, crop_scale, (255, 193, 7), 7)
        _draw_arrow(
            overlay,
            rel_anchor,
            (min(crop_img.width, rel_anchor[0] + crop_img.width * 0.20), rel_anchor[1]),
            crop_origin,
            crop_scale,
            (255, 193, 7),
            5,
        )
        y2m = crop_origin[1] + (rel_anchor[1] + 24) * crop_scale
        ax_rel = max(0.0, min(float(crop_img.width), rel_anchor[0]))
        overlay.line((crop_origin[0], y2m, crop_origin[0] + ax_rel * crop_scale, y2m), fill=(130, 130, 130, 210), width=5)
        overlay.line((crop_origin[0] + ax_rel * crop_scale, y2m, crop_origin[0] + crop_img.width * crop_scale, y2m), fill=(39, 174, 96, 230), width=7)

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((text_box[0] - 16, text_box[1] - 12, text_box[2] + 16, text_box[3]), radius=8, fill=(255, 255, 255, 255), outline=(210, 215, 222, 255), width=1)
    lines = [
        f"source: {source_image_id}",
        "mode: single_person_center",
        "variant: synthetic crop" if "Actual" not in title else "variant: actual v14 label crop",
        f"crop_xyxy: {metrics['crop']}",
        "",
        f"headroom_value: {metrics['headroom_value']}",
        f"q_head: {metrics['q_head']}",
        f"headroom label: {metrics['headroom_label']}",
        "",
        f"lookroom_value: {metrics['lookroom_value']}",
        f"q_look: {metrics['q_look']}",
        f"lookroom label: {metrics['lookroom_label']}",
        f"gaze_dir: {metrics['gaze_dir']}",
        "",
        f"why_tags: {', '.join(metrics['why_tags'])}",
        "",
        note,
    ]
    y = text_box[1]
    for line in lines:
        if line == "":
            y += 12
            continue
        y = _draw_lines(draw, (text_box[0], y), _wrap_text(line, text_box[2] - text_box[0], FONT_MONO), font=FONT_MONO, fill=(32, 37, 43), line_gap=4)
    draw.rounded_rectangle((40, 718, 1240, 880), radius=8, fill=(255, 255, 255, 255), outline=(210, 215, 222, 255), width=1)
    legend = (
        "Synthetic variants reuse the real PhaseA image, subject/gaze anchors, and v14 scoring formulas. "
        "Only crop_xyxy is changed to show threshold behavior. These synthetic crops are explanation assets, not training labels."
    )
    _draw_lines(draw, (64, 744), _wrap_text(legend, 1120, FONT), font=FONT, fill=(55, 65, 75), line_gap=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=94)
    metrics.update({"title": title, "note": note, "panel_path": str(out_path)})
    return metrics


def _write_same_image_variant_appendix(data: dict[str, Any], image_root: Path, output_dir: Path) -> None:
    images = {int(im["id"]): im for im in data["images"]}
    ann = next((a for a in data["annotations"] if int(a.get("id", -1)) == 165313), None)
    if ann is None:
        return
    im = images[int(ann["image_id"])]
    src = Image.open(image_root / im["file_name"]).convert("RGB")
    attrs = ann.get("attributes") or {}
    comps = attrs.get("score_components") or {}
    portrait = attrs.get("portrait_comp") or {}
    x, y, w, h = [float(v) for v in ann["bbox"]]
    actual_crop = (x, y, x + w, y + h)
    headroom_value = _safe_float(comps.get("headroom_value"), 0.12) or 0.12
    head_y = actual_crop[1] + headroom_value * (actual_crop[3] - actual_crop[1])
    lookroom_value = _safe_float(comps.get("lookroom_value"), 1.45) or 1.45
    gaze_anchor_x = (actual_crop[2] + lookroom_value * actual_crop[0]) / (1.0 + lookroom_value)
    eye = portrait.get("eye_point_norm_xy") if isinstance(portrait.get("eye_point_norm_xy"), list) else [0.42, 0.18]
    gaze_y = float(eye[1]) * src.height
    y2_base = actual_crop[3]
    tight_hr = 0.025
    tight_y1 = (head_y - tight_hr * y2_base) / (1.0 - tight_hr)
    loose_hr = 0.24
    loose_y2 = head_y / loose_hr
    insufficient_ratio = 0.65
    insufficient_x2 = gaze_anchor_x + insufficient_ratio * (gaze_anchor_x - actual_crop[0])
    excessive_ratio = 3.0
    excessive_x1 = gaze_anchor_x - (actual_crop[2] - gaze_anchor_x) / excessive_ratio
    insufficient_ann = next(
        (
            a
            for a in data["annotations"]
            if int(a.get("image_id", -1)) == int(ann["image_id"])
            and int(a.get("id", -1)) != int(ann["id"])
            and str(((a.get("attributes") or {}).get("checklist_labels") or {}).get("lookroom")) == "lookroom_insufficient"
            and str(a.get("mode_name")) == "single_person_center"
        ),
        None,
    )
    if insufficient_ann is not None:
        ix, iy, iw, ih = [float(v) for v in insufficient_ann["bbox"]]
        insufficient_crop = (ix, iy, ix + iw, iy + ih)
        insufficient_note = "This uses an actual v14 negative candidate from the same image, not a synthetic box."
    else:
        insufficient_crop = (actual_crop[0], actual_crop[1], insufficient_x2, actual_crop[3])
        insufficient_note = "No matching same-image v14 candidate was found, so this synthetic box moves the right edge near the gaze anchor."
    variants = [
        (
            "actual_ok_adequate",
            "Actual v14 crop: headroom_ok + lookroom_adequate",
            "This is the real positive annotation used in section 5.1.",
            actual_crop,
        ),
        (
            "synthetic_headroom_tight",
            "Synthetic: headroom_tight with adequate lookroom",
            "Move crop top downward so the head line sits too close to the top.",
            (actual_crop[0], tight_y1, actual_crop[2], actual_crop[3]),
        ),
        (
            "synthetic_headroom_loose",
            "Synthetic: headroom_loose with adequate lookroom",
            "Keep the top but shorten the crop height so top margin becomes large as a crop-height ratio.",
            (actual_crop[0], actual_crop[1], actual_crop[2], loose_y2),
        ),
        (
            "actual_negative_lookroom_insufficient" if insufficient_ann is not None else "synthetic_lookroom_insufficient",
            "Actual negative: lookroom_insufficient with ok headroom" if insufficient_ann is not None else "Synthetic: lookroom_insufficient with ok headroom",
            insufficient_note,
            insufficient_crop,
        ),
        (
            "synthetic_lookroom_excessive",
            "Synthetic: lookroom_excessive with ok headroom",
            "Move the left edge near the subject so the rightward gaze-forward space dominates.",
            (excessive_x1, actual_crop[1], actual_crop[2], actual_crop[3]),
        ),
    ]
    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for idx, (variant_id, title, note, crop) in enumerate(variants, 1):
        crop = (
            max(0.0, min(float(src.width - 1), float(crop[0]))),
            max(0.0, min(float(src.height - 1), float(crop[1]))),
            max(1.0, min(float(src.width), float(crop[2]))),
            max(1.0, min(float(src.height), float(crop[3]))),
        )
        out_path = output_dir / f"phasea_v14_headlook_02_same_image_{idx:02d}_{variant_id}.jpg"
        row = _draw_same_image_variant_panel(
            src=src,
            source_image_id=str(im["source_image_id"]),
            title=title,
            note=note,
            crop=crop,
            head_y=head_y,
            gaze_anchor_x=gaze_anchor_x,
            gaze_y=gaze_y,
            out_path=out_path,
        )
        row.update({"variant_id": variant_id, "source_image_id": im["source_image_id"], "base_annotation_id": ann["id"]})
        rows.append(row)
        panel_paths.append(out_path)
    _write_contact_sheet(panel_paths, output_dir / "phasea_v14_headlook_02_same_image_variant_contact_sheet.jpg")
    with (output_dir / "phasea_v14_headlook_02_same_image_variant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_image_id": im["source_image_id"],
                "base_annotation_id": ann["id"],
                "base_annotation_note": "Only the first variant is an actual v14 annotation. Other variants are synthetic crop boxes for explanation.",
                "head_y": round(float(head_y), 3),
                "gaze_anchor_x": round(float(gaze_anchor_x), 3),
                "gaze_y": round(float(gaze_y), 3),
                "variants": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "phasea_v14_headlook_02_same_image_variant_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant_id",
                "source_image_id",
                "base_annotation_id",
                "crop",
                "headroom_value",
                "q_head",
                "headroom_label",
                "lookroom_value",
                "q_look",
                "lookroom_label",
                "why_tags",
                "panel_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in writer.fieldnames}
            out["why_tags"] = "|".join(row.get("why_tags") or [])
            writer.writerow(out)


def _write_contact_sheet(panel_paths: list[Path], out_path: Path) -> None:
    thumbs = []
    for p in panel_paths:
        img = Image.open(p).convert("RGB")
        thumb = img.resize((900, 510), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 900, rows * 510), (248, 249, 251))
    for idx, img in enumerate(thumbs):
        x = (idx % cols) * 900
        y = (idx // cols) * 510
        sheet.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=94)


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    all_counts: dict[str, Counter[str]] = {"headroom": Counter(), "lookroom": Counter()}
    pos_counts: dict[str, Counter[str]] = {"headroom": Counter(), "lookroom": Counter()}
    mode_pos = Counter()
    for ann in data["annotations"]:
        attrs = ann.get("attributes") or {}
        labels = attrs.get("checklist_labels") or {}
        for key in ("headroom", "lookroom"):
            all_counts[key][str(labels.get(key, "missing"))] += 1
            if ann.get("gt_flag") == 1:
                pos_counts[key][str(labels.get(key, "missing"))] += 1
        if ann.get("gt_flag") == 1:
            mode_pos[str(ann.get("mode_name"))] += 1
    return {
        "annotations_total": len(data.get("annotations", [])),
        "images_total": len(data.get("images", [])),
        "headroom_counts_all": dict(all_counts["headroom"]),
        "lookroom_counts_all": dict(all_counts["lookroom"]),
        "headroom_counts_positive": dict(pos_counts["headroom"]),
        "lookroom_counts_positive": dict(pos_counts["lookroom"]),
        "positive_mode_counts": dict(mode_pos),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build actual-image headroom/lookroom guide assets from PhaseA multimode v14 labels.")
    parser.add_argument("--phasea_root", type=Path, default=Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529"))
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("Implement_Docs/assets_phasea_v14_headroom_lookroom_guide"))
    args = parser.parse_args()

    run_dir = args.run_dir or args.phasea_root / "artifacts/training_labels_multimode" / GUIDE_RUN
    label_json = run_dir / "label_json/multimode_labels_full.json"
    image_root = args.phasea_root / "images"
    with label_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    selected = select_examples(data, image_root)
    rows = []
    panel_paths: list[Path] = []
    for idx, sample in enumerate(selected, 1):
        case_id = sample["case"]["case_id"]
        out_path = args.output_dir / f"phasea_v14_headlook_{idx:02d}_{case_id}.jpg"
        row = _draw_sample_panel(sample, image_root, out_path)
        rows.append(row)
        panel_paths.append(out_path)

    _write_contact_sheet(panel_paths, args.output_dir / "phasea_v14_headroom_lookroom_contact_sheet.jpg")
    _write_same_image_variant_appendix(data, image_root, args.output_dir)

    summary = _summary(data)
    summary["run_dir"] = str(run_dir)
    summary["label_json"] = str(label_json)
    summary["selected_count"] = len(rows)
    summary["selected_samples"] = rows
    with (args.output_dir / "phasea_v14_headroom_lookroom_guide_assets_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "phasea_v14_headroom_lookroom_guide_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "title",
                "source_image_id",
                "image_id",
                "annotation_id",
                "mode_name",
                "target_ar",
                "score_mode",
                "headroom_label",
                "headroom_value",
                "q_head",
                "lookroom_label",
                "lookroom_value",
                "q_look",
                "gaze_dir",
                "why_tags",
                "panel_path",
                "source_image",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["why_tags"] = "|".join(row.get("why_tags") or [])
            writer.writerow(out)
    print(json.dumps({"output_dir": str(args.output_dir), "selected_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
