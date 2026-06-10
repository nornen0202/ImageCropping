#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_simple import CAT_CLASS_ID, DOG_CLASS_ID, safe_float, safe_int  # noqa: E402


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_RUN_TAG = "260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2"
DEFAULT_LABEL_DIR = DEFAULT_PHASE_ROOT / "artifacts/training_labels" / DEFAULT_RUN_TAG
DEFAULT_MM_DIR = DEFAULT_PHASE_ROOT / "artifacts/training_labels_multimode" / DEFAULT_RUN_TAG
DEFAULT_FEATURE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl"
DEFAULT_ANIMAL_POSE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/animal_pose_260609_routing_v2_simple_gpu_full_ap10k_nms_v2/animal_pose.jsonl"

ROUTE_COLORS = {
    "scene": (70, 130, 180),
    "person_single": (32, 160, 100),
    "person_group": (155, 92, 210),
    "pet_dogcat": (230, 126, 34),
}
SHOT_COLORS = {
    "face_headshot": (20, 184, 166),
    "upper_half_body": (52, 152, 219),
    "full_body": (46, 204, 113),
    "environmental_portrait": (241, 196, 15),
}
ANIMAL_KP_COLOR = (255, 210, 40)
ANIMAL_HEAD_COLOR = (255, 90, 90)
ANIMAL_BODY_COLOR = (255, 150, 40)
POSITIVE_CROP_COLOR = (50, 220, 90)
SUBJECT_COLOR = (0, 210, 255)
TEXT_DARK = (24, 28, 32)
TEXT_MUTED = (92, 98, 106)

AP10K_EDGES = [
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("nose", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_front_paw"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_front_paw"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_back_paw"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_back_paw"),
    ("neck", "root_of_tail"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize routing_v2_simple labels and AP-10K animal pose artifacts.")
    parser.add_argument("--label_jsonl", type=Path, default=DEFAULT_LABEL_DIR / "routing_v2_v16_full.jsonl")
    parser.add_argument("--multimode_json", type=Path, default=DEFAULT_MM_DIR / "label_json/multimode_labels_full.json")
    parser.add_argument("--feature_jsonl", type=Path, default=DEFAULT_FEATURE_JSONL)
    parser.add_argument("--animal_pose_jsonl", type=Path, default=DEFAULT_ANIMAL_POSE_JSONL)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_PHASE_ROOT / "images")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_LABEL_DIR / "qualitative_viz")
    parser.add_argument("--max_per_bucket", type=int, default=5)
    parser.add_argument("--max_animal_pose", type=int, default=24)
    return parser


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
FONT_H2 = _font(20, bold=True)
FONT = _font(16)
FONT_SMALL = _font(13)
FONT_MONO = _font(13)


def _wrap_by_pixels(draw: ImageDraw.ImageDraw, text: Any, font: ImageFont.ImageFont, max_width: int, *, max_lines: int = 3) -> list[str]:
    words = str(text if text is not None else "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        pieces = [word]
        if draw.textlength(word, font=font) > max_width:
            approx = max(4, int(len(word) * max_width / max(1, draw.textlength(word, font=font))))
            pieces = textwrap.wrap(word, width=approx, break_long_words=True, break_on_hyphens=False) or [word]
        for piece in pieces:
            candidate = piece if not current else f"{current} {piece}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = piece
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        line = lines[-1]
        while line and draw.textlength(line + "...", font=font) > max_width:
            line = line[:-1]
        lines[-1] = f"{line.rstrip()}..." if line else "..."
    return lines[:max_lines]


def _draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], x: int, y: int, font: ImageFont.ImageFont, fill: tuple[int, int, int], *, gap: int = 3) -> int:
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + gap
    return y


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _load_features(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("image_id")): row for row in _iter_jsonl(path) if row.get("image_id")}


def _load_animal_pose(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(path):
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            continue
        by_image[image_id].append(row)
    for rows in by_image.values():
        rows.sort(key=lambda item: safe_float(item.get("pose_confidence"), 0.0), reverse=True)
    return dict(by_image)


def _load_multimode_annotations(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    anns = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    id_to_source = {
        int(image.get("id")): str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)
        for image in images
        if image.get("id") is not None
    }
    image_rows = {source_id: image for image_id, source_id in id_to_source.items() for image in images if int(image.get("id")) == image_id}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        source_id = id_to_source.get(int(ann.get("image_id", -1)))
        if not source_id:
            continue
        by_source[source_id].append(ann)
    for rows in by_source.values():
        rows.sort(key=lambda ann: _score_value(ann), reverse=True)
    return image_rows, dict(by_source)


def _score_value(ann: dict[str, Any]) -> float:
    for key in ("score_mode", "score", "final_score"):
        value = ann.get(key)
        if value is not None:
            return safe_float(value, 0.0)
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    scores = attrs.get("checklist_scores") if isinstance(attrs.get("checklist_scores"), dict) else {}
    return safe_float(scores.get("final_score"), 0.0)


def _resolve_image(path_text: str, image_root: Path, image_id: str) -> Path:
    path_text = str(path_text or "").strip()
    if path_text:
        path = Path(path_text)
        if path.exists() and path.is_file():
            return path
        candidate = image_root / path.name
        if candidate.exists() and candidate.is_file():
            return candidate
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_root / f"{image_id}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return image_root / path_text


def _fit(im: Image.Image, max_w: int, max_h: int) -> tuple[Image.Image, float]:
    scale = min(max_w / max(1, im.width), max_h / max(1, im.height))
    size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
    return im.resize(size, Image.Resampling.LANCZOS), scale


def _box_to_px(box: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    vals = [safe_float(v, float("nan")) for v in box[:4]]
    if not all(math.isfinite(v) for v in vals):
        return None
    if max(vals) <= 1.5:
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _ann_bbox_to_px(ann: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = ann.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    x, y, w, h = [safe_float(v, 0.0) for v in box[:4]]
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def _draw_rect(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], *, offset: tuple[int, int], scale: float, color: tuple[int, int, int], width: int = 4, label: str = "") -> None:
    ox, oy = offset
    x1, y1, x2, y2 = box
    rect = (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale)
    draw.rectangle(rect, outline=color, width=width)
    if label:
        tx, ty = rect[0] + 5, max(oy + 5, rect[1] - 22)
        tw = int(draw.textlength(label, font=FONT_SMALL)) + 10
        draw.rectangle((tx - 3, ty - 2, tx + tw, ty + 18), fill=color)
        draw.text((tx + 2, ty), label, font=FONT_SMALL, fill=(15, 15, 15))


def _draw_point(draw: ImageDraw.ImageDraw, xy: tuple[float, float], *, offset: tuple[int, int], scale: float, color: tuple[int, int, int], radius: int = 5) -> None:
    ox, oy = offset
    x, y = ox + xy[0] * scale, oy + xy[1] * scale
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(20, 20, 20), width=1)


def _draw_animal_pose(draw: ImageDraw.ImageDraw, pose: dict[str, Any], *, offset: tuple[int, int], scale: float) -> None:
    body = _box_to_px(pose.get("body_box_xyxy") or pose.get("bbox_xyxy"), 1, 1)
    head = _box_to_px(pose.get("head_box_xyxy"), 1, 1)
    if body:
        _draw_rect(draw, body, offset=offset, scale=scale, color=ANIMAL_BODY_COLOR, width=3, label=str(pose.get("species_hint") or "pet"))
    if head:
        _draw_rect(draw, head, offset=offset, scale=scale, color=ANIMAL_HEAD_COLOR, width=3, label="head")
    kps = pose.get("keypoints") if isinstance(pose.get("keypoints"), dict) else {}
    points: dict[str, tuple[float, float]] = {}
    for name, point in kps.items():
        if isinstance(point, (list, tuple)) and len(point) >= 3 and safe_float(point[2], 0.0) >= 0.15:
            points[str(name)] = (safe_float(point[0]), safe_float(point[1]))
    ox, oy = offset
    for a, b in AP10K_EDGES:
        if a in points and b in points:
            ax, ay = points[a]
            bx, by = points[b]
            draw.line((ox + ax * scale, oy + ay * scale, ox + bx * scale, oy + by * scale), fill=ANIMAL_KP_COLOR, width=3)
    for point in points.values():
        _draw_point(draw, point, offset=offset, scale=scale, color=ANIMAL_KP_COLOR, radius=5)


def _draw_subject_overlays(draw: ImageDraw.ImageDraw, feature: dict[str, Any] | None, routing: dict[str, Any], *, offset: tuple[int, int], scale: float, width: int, height: int) -> None:
    if not isinstance(feature, dict):
        return
    family = str(routing.get("route_family_v2") or "")
    if family.startswith("person"):
        poses = feature.get("c3_pose") if isinstance(feature.get("c3_pose"), list) else []
        for idx, pose in enumerate(poses[:4]):
            if isinstance(pose, dict):
                box = _box_to_px(pose.get("bbox"), width, height)
                if box:
                    _draw_rect(draw, box, offset=offset, scale=scale, color=SUBJECT_COLOR, width=2, label=f"person{idx}")
    elif family == "pet_dogcat":
        for inst in _iter_dogcat_instances(feature):
            box = _box_to_px(inst.get("box") or inst.get("bbox"), width, height)
            if box:
                species = "dog" if safe_int(inst.get("class_id"), -1) == DOG_CLASS_ID else "cat"
                _draw_rect(draw, box, offset=offset, scale=scale, color=SUBJECT_COLOR, width=2, label=f"det {species}")


def _iter_dogcat_instances(feature: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("c2_seg", "c2_det"):
        items = feature.get(key)
        if isinstance(items, dict):
            items = items.get("instances")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and safe_int(item.get("class_id"), -1) in {DOG_CLASS_ID, CAT_CLASS_ID}:
                yield item


def _select_samples(rows: list[dict[str, Any]], max_per_bucket: int) -> list[dict[str, Any]]:
    bucket_order = [
        ("scene", None, None),
        ("person_single", "face_headshot", None),
        ("person_single", "upper_half_body", None),
        ("person_single", "full_body", None),
        ("person_single", None, "environmental"),
        ("person_group", None, None),
        ("pet_dogcat", None, None),
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for family, shot, context in bucket_order:
        candidates = []
        for row in rows:
            routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
            if routing.get("route_family_v2") != family:
                continue
            if shot is not None and routing.get("person_shot_type") != shot:
                continue
            if context is not None and routing.get("context_intent") != context:
                continue
            candidates.append(row)
        candidates.sort(key=lambda row: (safe_float((row.get("routing_v2_simple") or {}).get("routing_confidence"), 0.0), safe_float(row.get("label_weight"), 0.0)), reverse=True)
        for row in candidates:
            image_id = str(row.get("image_id") or "")
            if image_id in used:
                continue
            selected.append(row)
            used.add(image_id)
            if sum(1 for x in selected if (x.get("routing_v2_simple") or {}).get("route_family_v2") == family and (shot is None or (x.get("routing_v2_simple") or {}).get("person_shot_type") == shot)) >= max_per_bucket:
                break
    return selected


def _matching_annotations(image_id: str, routing: dict[str, Any], anns_by_source: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for ann in anns_by_source.get(image_id, []):
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        ann_routing = attrs.get("routing_v2_simple") if isinstance(attrs.get("routing_v2_simple"), dict) else {}
        if ann_routing.get("route_family_v2") != routing.get("route_family_v2"):
            continue
        if routing.get("person_shot_type") is not None and ann_routing.get("person_shot_type") != routing.get("person_shot_type"):
            continue
        rows.append(ann)
    rows.sort(key=_score_value, reverse=True)
    return rows[:3]


def _draw_card(row: dict[str, Any], *, image_root: Path, feature: dict[str, Any] | None, anns_by_source: dict[str, list[dict[str, Any]]], animal_by_image: dict[str, list[dict[str, Any]]], output_path: Path) -> dict[str, Any]:
    image_id = str(row.get("image_id") or "")
    image_path = _resolve_image(str(row.get("image_path") or ""), image_root, image_id)
    im = Image.open(image_path).convert("RGB")
    fitted, scale = _fit(im, 920, 690)
    canvas = Image.new("RGB", (1360, 900), (246, 247, 249))
    image_offset = (24, 134)
    canvas.paste(fitted, image_offset)
    draw = ImageDraw.Draw(canvas)
    routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
    family = str(routing.get("route_family_v2") or "scene")
    shot = str(routing.get("person_shot_type") or "na")
    context = str(routing.get("context_intent") or "")
    image_route = str(row.get("image_route_name_no_placement") or family)
    route_color = ROUTE_COLORS.get(family, (120, 120, 120))
    draw.text((24, 12), image_id, font=FONT_TITLE, fill=TEXT_DARK)
    draw.rounded_rectangle((24, 56, 1336, 102), radius=8, fill=route_color, outline=(35, 35, 35), width=1)
    draw.rounded_rectangle((38, 65, 162, 93), radius=6, fill=(15, 26, 36))
    draw.text((48, 70), "IMAGE ROUTE", font=FONT_SMALL, fill=(255, 255, 255))
    draw.text((184, 62), image_route, font=FONT_TITLE, fill=(255, 255, 255))
    conf = f"conf={safe_float(routing.get('routing_confidence'), 0.0):.3f}"
    conf_w = int(draw.textlength(conf, font=FONT_SMALL)) + 18
    draw.rounded_rectangle((1336 - conf_w - 12, 67, 1324, 91), radius=6, fill=(255, 255, 255))
    draw.text((1336 - conf_w - 3, 71), conf, font=FONT_SMALL, fill=TEXT_DARK)
    draw.text((24, 108), f"image metadata: {family}/{shot}/{routing.get('placement_intent')}/{context}/{routing.get('mode_feasible')} | v16={row.get('v16_mode_name')}", font=FONT_SMALL, fill=TEXT_MUTED)
    _draw_subject_overlays(draw, feature, routing, offset=image_offset, scale=scale, width=im.width, height=im.height)
    for idx, ann in enumerate(_matching_annotations(image_id, routing, anns_by_source)):
        box = _ann_bbox_to_px(ann)
        if box:
            _draw_rect(draw, box, offset=image_offset, scale=scale, color=POSITIVE_CROP_COLOR, width=4, label=f"pos{idx+1} {safe_float(_score_value(ann), 0.0):.2f}")
    for pose in animal_by_image.get(image_id, [])[:3]:
        _draw_animal_pose(draw, pose, offset=image_offset, scale=scale)
    side_x = 980
    y = 146
    lines = [
        ("image_route", image_route),
        ("v16_mode_aux", row.get("v16_mode_name")),
        ("route_family_v2", family),
        ("person_shot_type", shot),
        ("placement_intent", routing.get("placement_intent")),
        ("context_intent", context),
        ("mode_feasible", routing.get("mode_feasible")),
        ("routing_confidence", f"{safe_float(routing.get('routing_confidence'), 0.0):.3f}"),
        ("teacher_reliability", routing.get("teacher_reliability")),
        ("support_policy", routing.get("support_grounding_policy")),
        ("flat_route_class", row.get("flat_route_class")),
    ]
    for key, value in lines:
        draw.text((side_x, y), str(key), font=FONT_SMALL, fill=TEXT_MUTED)
        y += 17
        value_font = FONT_H2 if key in {"image_route", "flat_route_class"} else FONT
        max_lines = 2 if key in {"image_route", "v16_mode_aux", "flat_route_class"} else 1
        y = _draw_lines(draw, _wrap_by_pixels(draw, value, value_font, 340, max_lines=max_lines), side_x, y, value_font, TEXT_DARK, gap=4)
        y += 8
    env = routing.get("environmental_portrait_evidence") if isinstance(routing.get("environmental_portrait_evidence"), dict) else {}
    if env:
        draw.text((side_x, y + 8), "environmental evidence", font=FONT_H2, fill=SHOT_COLORS.get("environmental_portrait", TEXT_DARK))
        y += 38
        for key in ("environmental_score", "person_area", "scene_score", "foreground_mass", "blank_ratio_est", "area_ok", "context_ok"):
            if key in env:
                draw.text((side_x, y), f"{key}: {env[key]}", font=FONT_SMALL, fill=TEXT_DARK)
                y += 19
    pet_rows = animal_by_image.get(image_id, [])
    if pet_rows:
        draw.text((side_x, y + 8), "animal pose", font=FONT_H2, fill=ROUTE_COLORS["pet_dogcat"])
        y += 38
        for pose in pet_rows[:4]:
            draw.text((side_x, y), f"{pose.get('species_hint')} pose={safe_float(pose.get('pose_confidence'), 0.0):.3f} head={((pose.get('visible_groups') or {}).get('head'))}", font=FONT_SMALL, fill=TEXT_DARK)
            y += 19
    reasons = routing.get("reasons") if isinstance(routing.get("reasons"), list) else []
    if reasons:
        draw.text((side_x, min(y + 10, 755)), "reasons", font=FONT_H2, fill=TEXT_DARK)
        y = min(y + 42, 787)
        for reason in reasons[:6]:
            y = _draw_lines(draw, _wrap_by_pixels(draw, f"- {reason}", FONT_SMALL, 340, max_lines=2), side_x, y, FONT_SMALL, TEXT_DARK, gap=2)
            y += 3
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return {
        "image_id": image_id,
        "image_route_name_no_placement": image_route,
        "route_family_v2": family,
        "person_shot_type": shot,
        "context_intent": context,
        "routing_confidence": safe_float(routing.get("routing_confidence"), 0.0),
        "flat_route_class": row.get("flat_route_class"),
        "card": str(output_path),
    }


def _animal_pose_card(image_id: str, poses: list[dict[str, Any]], *, image_root: Path, output_path: Path) -> dict[str, Any]:
    image_path = _resolve_image("", image_root, image_id)
    im = Image.open(image_path).convert("RGB")
    fitted, scale = _fit(im, 920, 760)
    canvas = Image.new("RGB", (1180, 840), (247, 247, 246))
    canvas.paste(fitted, (24, 58))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1180, 46), fill=ROUTE_COLORS["pet_dogcat"])
    draw.text((24, 10), f"{image_id} | AP-10K animal pose", font=FONT_H2, fill=(255, 255, 255))
    for pose in poses[:4]:
        _draw_animal_pose(draw, pose, offset=(24, 58), scale=scale)
    x, y = 970, 76
    for pose in poses[:6]:
        draw.text((x, y), f"{pose.get('species_hint')} conf={safe_float(pose.get('pose_confidence'), 0.0):.3f}", font=FONT_SMALL, fill=TEXT_DARK)
        y += 18
        groups = pose.get("visible_groups") if isinstance(pose.get("visible_groups"), dict) else {}
        draw.text((x, y), f"head={groups.get('head')} torso={groups.get('torso')}", font=FONT_SMALL, fill=TEXT_MUTED)
        y += 24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return {"image_id": image_id, "pose_rows": len(poses), "card": str(output_path)}


def _contact_sheet(paths: list[Path], output_path: Path, *, thumb_w: int = 340) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        scale = thumb_w / max(1, im.width)
        thumb = im.resize((thumb_w, max(1, int(round(im.height * scale)))), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    cols = 3
    gap = 12
    rows = math.ceil(len(thumbs) / cols)
    h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (238, 240, 243))
    for idx, thumb in enumerate(thumbs):
        x = gap + (idx % cols) * (thumb_w + gap)
        y = gap + (idx // cols) * (h + gap)
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _write_index(path: Path, routing_rows: list[dict[str, Any]], animal_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# routing_v2_simple qualitative visualization",
        "",
        f"- routing cards: {summary['routing_cards']}",
        f"- animal pose cards: {summary['animal_pose_cards']}",
        "",
        "## Routing Samples",
        "",
    ]
    for row in routing_rows:
        rel = Path(row["card"]).name
        lines.append(f"- `{row['image_id']}`: `{row['flat_route_class']}`, conf={row['routing_confidence']:.3f}  ")
        lines.append(f"  ![]({Path('routing_cards') / rel})")
    lines.extend(["", "## Animal Pose Samples", ""])
    for row in animal_rows:
        rel = Path(row["card"]).name
        lines.append(f"- `{row['image_id']}`: pose rows={row['pose_rows']}  ")
        lines.append(f"  ![]({Path('animal_pose_cards') / rel})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(_iter_jsonl(args.label_jsonl))
    features = _load_features(args.feature_jsonl)
    _image_rows, anns_by_source = _load_multimode_annotations(args.multimode_json)
    animal_by_image = _load_animal_pose(args.animal_pose_jsonl)

    selected = _select_samples(rows, max_per_bucket=max(1, int(args.max_per_bucket)))
    routing_manifest = []
    routing_paths: list[Path] = []
    for idx, row in enumerate(selected, start=1):
        image_id = str(row.get("image_id") or "")
        out = output_dir / "routing_cards" / f"{idx:03d}_{image_id}.jpg"
        routing_manifest.append(
            _draw_card(
                row,
                image_root=args.image_root,
                feature=features.get(image_id),
                anns_by_source=anns_by_source,
                animal_by_image=animal_by_image,
                output_path=out,
            )
        )
        routing_paths.append(out)

    animal_candidates = sorted(
        animal_by_image.items(),
        key=lambda item: (len(item[1]), max(safe_float(row.get("pose_confidence"), 0.0) for row in item[1])),
        reverse=True,
    )
    animal_manifest = []
    animal_paths: list[Path] = []
    for idx, (image_id, poses) in enumerate(animal_candidates[: max(0, int(args.max_animal_pose))], start=1):
        out = output_dir / "animal_pose_cards" / f"{idx:03d}_{image_id}.jpg"
        animal_manifest.append(_animal_pose_card(image_id, poses, image_root=args.image_root, output_path=out))
        animal_paths.append(out)

    _contact_sheet(routing_paths, output_dir / "routing_contact_sheet.jpg")
    _contact_sheet(animal_paths, output_dir / "animal_pose_contact_sheet.jpg")
    _write_csv(output_dir / "routing_sample_manifest.csv", routing_manifest)
    _write_csv(output_dir / "animal_pose_manifest.csv", animal_manifest)
    summary = {
        "state": "completed",
        "label_jsonl": str(args.label_jsonl),
        "multimode_json": str(args.multimode_json),
        "feature_jsonl": str(args.feature_jsonl),
        "animal_pose_jsonl": str(args.animal_pose_jsonl),
        "routing_cards": len(routing_manifest),
        "animal_pose_cards": len(animal_manifest),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_index(output_dir / "index.md", routing_manifest, animal_manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
