from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional at runtime
    Image = None
    ImageColor = None
    ImageDraw = None
    ImageFont = None

from multimode.candidate_bank import build_global_candidate_bank, expand_query_candidates
from multimode.coco_writer import build_annotation_entry, build_coco_dataset, build_image_entry, write_json, write_jsonl
from multimode.entity_atoms import build_entity_atoms
from multimode.mode_catalog import BASE_CATEGORIES
from multimode.mode_scorer import score_query_candidates, select_query_results
from multimode.query_builder import ModeQuery, build_mode_queries
from multimode.query_guidance import build_query_guidance
from scripts.progress_utils import ProgressTracker, count_nonempty_lines, progress_log


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MODE_DEBUG_COLORS = {
    "landscape": "#D4A017",
    "single_person_center": "#1F77B4",
    "single_person_rot": "#17A2B8",
    "group_center": "#2CA02C",
    "group_rot": "#3CB44B",
    "face": "#FF7F0E",
    "object_single_center": "#9467BD",
    "object_single_rot": "#8C564B",
    "object_multi_center": "#E377C2",
    "object_multi_rot": "#7F7F7F",
}
NEGATIVE_DEBUG_COLOR = "#D62728"
SUBJECT_DEBUG_COLOR = "#FFF4CC"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_jsonl(path: Path, *, key_field: Optional[str] = None, max_rows: int = 0, progress: bool = False) -> Any:
    if key_field:
        rows: Dict[str, Dict[str, Any]] = {}
    else:
        rows = []
    invalid_rows = 0
    total = count_nonempty_lines(path) if progress else None
    tracker = ProgressTracker(f"load:{path.name}", total=total, unit="rows", every=500, min_seconds=5.0, enabled=progress)
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                progress_log(
                    f"load:{path.name}: skip invalid json row line={idx}",
                    enabled=bool(progress),
                )
                continue
            if key_field:
                key = str(row.get(key_field, "")).strip()
                if key:
                    rows[key] = row
            else:
                rows.append(row)
            if max_rows > 0 and idx >= max_rows:
                break
            tracker.update(idx)
    tracker.finish(len(rows) if not isinstance(rows, list) else len(rows))
    if invalid_rows > 0:
        progress_log(f"load:{path.name}: invalid_rows={invalid_rows}", enabled=bool(progress))
    return rows


def _normalize_target_ars(text: str) -> List[str]:
    values = []
    for chunk in str(text or "").split(","):
        value = chunk.strip()
        if value:
            values.append(value)
    deduped: List[str] = []
    seen = set()
    for value in values:
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _read_image_ids_file(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    image_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                image_ids.add(value)
    return image_ids


def _build_image_index(image_root: Path, *, progress: bool) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    files = [path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    tracker = ProgressTracker("index_images", total=len(files), unit="files", every=500, min_seconds=5.0, enabled=progress)
    for idx, path in enumerate(files, start=1):
        stem = path.stem
        index.setdefault(stem, path)
        tracker.update(idx)
    tracker.finish(len(files))
    return index


def _resolve_image_path(
    *,
    image_id: str,
    image_root: Path,
    image_index: Dict[str, Path],
    candidate_row: Dict[str, Any],
    feat_row: Dict[str, Any],
) -> Tuple[Path, str]:
    for row in (candidate_row, feat_row):
        for key in ("image_path", "file_path", "file_name", "rel_path"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            raw = Path(value.strip())
            candidates = []
            if raw.is_absolute():
                candidates.append(raw)
            else:
                candidates.append(image_root / raw)
                candidates.append(Path(value.strip()))
            for candidate in candidates:
                if candidate.is_file():
                    try:
                        rel = str(candidate.relative_to(image_root))
                    except ValueError:
                        rel = candidate.name
                    return candidate, rel
    resolved = image_index.get(str(image_id))
    if resolved is None:
        raise FileNotFoundError(f"image file not found for image_id={image_id} under {image_root}")
    try:
        rel = str(resolved.relative_to(image_root))
    except ValueError:
        rel = resolved.name
    return resolved, rel


def _resolve_image_size(
    *,
    candidate_row: Dict[str, Any],
    feat_row: Dict[str, Any],
    image_path: Optional[Path],
) -> Tuple[int, int]:
    width = _safe_int(candidate_row.get("width"), 0) or _safe_int(candidate_row.get("width_parquet"), 0)
    height = _safe_int(candidate_row.get("height"), 0) or _safe_int(candidate_row.get("height_parquet"), 0)
    if width > 0 and height > 0:
        return width, height
    width = _safe_int(feat_row.get("width"), 0)
    height = _safe_int(feat_row.get("height"), 0)
    if width > 0 and height > 0:
        return width, height
    if image_path is None or Image is None:
        raise ValueError("image size unavailable and PIL is not usable")
    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def _query_status_row(
    *,
    source_image_id: str,
    target_ar: str,
    query: ModeQuery,
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    positive = selection.get("positive") if isinstance(selection.get("positive"), dict) else None
    best = selection.get("best_scored_candidate") if isinstance(selection.get("best_scored_candidate"), dict) else None
    negatives = selection.get("negatives") if isinstance(selection.get("negatives"), list) else []
    if positive is not None:
        decision = "positive"
        no_positive_reason = ""
    elif best is None:
        decision = "no_candidates"
        no_positive_reason = "no_candidates"
    elif bool(best.get("hard_reject", False)):
        reasons = best.get("hard_reject_reasons") or ["hard_reject"]
        decision = "no_positive"
        no_positive_reason = ",".join(str(reason) for reason in reasons)
    else:
        decision = "below_tau"
        no_positive_reason = "below_tau"
    return {
        "source_image_id": str(source_image_id),
        "target_ar": str(target_ar),
        "query_id": str(query.query_id),
        "mode_name": str(query.mode_name),
        "mode_id": int(query.mode_id),
        "entity_id": str(query.entity_id),
        "entity_type": str(query.entity_type),
        "route_mode": str(query.route_mode),
        "route_family": str(query.route_family),
        "decision": decision,
        "tau_pos": float(selection.get("tau_pos", 0.0)),
        "candidate_count": int(selection.get("candidate_count", 0)),
        "positive_exists": int(positive is not None),
        "positive_score": None if positive is None else float(positive.get("score_mode", 0.0)),
        "best_score": None if best is None else float(best.get("score_mode", 0.0)),
        "negative_count": len(negatives),
        "no_positive_reason": no_positive_reason,
        "attributes": dict(query.attributes),
    }


def _annotation_attributes(
    *,
    query: ModeQuery,
    selected_row: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    components = selected_row.get("components") if isinstance(selected_row.get("components"), dict) else {}
    payload = {
        "target_ar": str(query.target_ar),
        "entity_type": str(query.entity_type),
        "route_family": str(query.route_family),
        "route_mode": str(query.route_mode),
        "candidate_id": str(selected_row.get("candidate_id", "")),
        "candidate_source": str(selected_row.get("source", "")),
        "score_components": components,
    }
    payload.update(query.attributes)
    payload.update(extra)
    return payload


def _load_debug_font(font_size: int) -> Any:
    if ImageFont is None:
        return None
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _box_px(box_norm_xyxy: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = float(box_norm_xyxy[0]) * float(width)
    y1 = float(box_norm_xyxy[1]) * float(height)
    x2 = float(box_norm_xyxy[2]) * float(width)
    y2 = float(box_norm_xyxy[3]) * float(height)
    return x1, y1, x2, y2


def _draw_dashed_rectangle(
    draw: Any,
    coords: Tuple[float, float, float, float],
    *,
    color: str,
    width_px: int,
    dash_px: int,
    gap_px: int,
) -> None:
    x1, y1, x2, y2 = coords
    segments = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    for start, end in segments:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(abs(dx), abs(dy))
        if length <= 0:
            continue
        step = max(1.0, float(dash_px + gap_px))
        pos = 0.0
        while pos < length:
            seg_start = pos / length
            seg_end = min(length, pos + dash_px) / length
            sx = start[0] + dx * seg_start
            sy = start[1] + dy * seg_start
            ex = start[0] + dx * seg_end
            ey = start[1] + dy * seg_end
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width_px)
            pos += step


def _draw_double_rectangle(
    draw: Any,
    coords: Tuple[float, float, float, float],
    *,
    color: str,
    width_px: int,
    inset_px: int,
) -> None:
    x1, y1, x2, y2 = coords
    draw.rectangle(coords, outline=color, width=width_px)
    inner = (x1 + inset_px, y1 + inset_px, x2 - inset_px, y2 - inset_px)
    if inner[2] > inner[0] and inner[3] > inner[1]:
        draw.rectangle(inner, outline=color, width=max(2, width_px // 2))


def _draw_label(
    draw: Any,
    coords: Tuple[float, float, float, float],
    *,
    label: str,
    color: str,
    font: Any,
    width: int,
    height: int,
) -> None:
    x1, y1, _, _ = coords
    pad_x = max(6, int(round(min(width, height) * 0.006)))
    pad_y = max(4, int(round(min(width, height) * 0.004)))
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = max(1, int(round(right - left)))
        text_h = max(1, int(round(bottom - top)))
    else:
        text_w = max(1, int(round(len(label) * 8)))
        text_h = 16
    box_y1 = max(0, int(round(y1 - text_h - pad_y * 2 - 6)))
    box_y2 = box_y1 + text_h + pad_y * 2
    box_x1 = max(0, int(round(x1)))
    box_x2 = min(width, box_x1 + text_w + pad_x * 2)
    draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=8, fill=(0, 0, 0, 180), outline=color, width=2)
    rgb = ImageColor.getrgb(color) if ImageColor is not None else (255, 255, 255)
    draw.text((box_x1 + pad_x, box_y1 + pad_y), label, fill=(rgb[0], rgb[1], rgb[2], 255), font=font)


def _draw_box(
    draw: Any,
    box_norm_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
    color: str,
    label: str,
    font: Any,
    line_width: int,
    fill_rgba: Tuple[int, int, int, int] | None = None,
    dashed: bool = False,
    double_line: bool = False,
) -> None:
    coords = _box_px(box_norm_xyxy, width, height)
    if fill_rgba is not None:
        draw.rectangle(coords, fill=fill_rgba)
    if dashed:
        _draw_dashed_rectangle(
            draw,
            coords,
            color=color,
            width_px=line_width,
            dash_px=max(8, line_width * 4),
            gap_px=max(6, line_width * 3),
        )
    elif double_line:
        _draw_double_rectangle(
            draw,
            coords,
            color=color,
            width_px=line_width,
            inset_px=max(4, line_width + 1),
        )
    else:
        draw.rectangle(coords, outline=color, width=line_width)
    _draw_label(draw, coords, label=label, color=color, font=font, width=width, height=height)


def _subject_box_for_debug(query: ModeQuery) -> Optional[List[float]]:
    if query.mode_name == "landscape":
        return None
    if query.mode_name == "face" and query.face_bbox_norm_xyxy is not None:
        return [float(v) for v in (query.head_bbox_norm_xyxy or query.face_bbox_norm_xyxy)]
    return [float(v) for v in query.anchor_bbox_norm_xyxy]


def _draw_query_payloads(
    *,
    canvas: Any,
    width: int,
    height: int,
    query_payloads: Sequence[Dict[str, Any]],
    include_negatives: bool,
    single_mode_name: Optional[str] = None,
) -> Any:
    overlay_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_layer, "RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    base_scale = max(14, int(round(min(width, height) * 0.022)))
    font = _load_debug_font(base_scale)
    subject_line_width = max(5, int(round(min(width, height) * 0.006)))
    pos_line_width = max(5, int(round(min(width, height) * 0.005)))
    neg_line_width = max(3, int(round(min(width, height) * 0.0035)))

    filtered_payloads = query_payloads
    if single_mode_name is not None:
        filtered_payloads = [payload for payload in query_payloads if payload["query"].mode_name == single_mode_name]

    for payload in filtered_payloads:
        query = payload["query"]
        mode_color = MODE_DEBUG_COLORS.get(query.mode_name, "#1F77B4")
        rgb = ImageColor.getrgb(mode_color)
        overlay_rgba = (rgb[0], rgb[1], rgb[2], 76)
        subject_box = _subject_box_for_debug(query)
        if subject_box is None:
            continue
        _draw_box(
            overlay_draw,
            subject_box,
            width=width,
            height=height,
            color=SUBJECT_DEBUG_COLOR,
            label=f"{query.mode_name}:SUBJ",
            font=font,
            line_width=subject_line_width,
            fill_rgba=overlay_rgba,
            double_line=True,
        )
    composed = Image.alpha_composite(canvas, overlay_layer)
    draw = ImageDraw.Draw(composed, "RGBA")
    for payload in filtered_payloads:
        query = payload["query"]
        selection = payload["selection"]
        mode_color = MODE_DEBUG_COLORS.get(query.mode_name, "#1F77B4")
        subject_box = _subject_box_for_debug(query)
        if subject_box is not None:
            _draw_box(
                draw,
                subject_box,
                width=width,
                height=height,
                color=SUBJECT_DEBUG_COLOR,
                label=f"{query.mode_name}:SUBJ",
                font=font,
                line_width=subject_line_width,
                double_line=True,
            )
        positive = selection.get("positive") if isinstance(selection.get("positive"), dict) else None
        if positive is not None:
            _draw_box(
                draw,
                positive["bbox_norm_xyxy"],
                width=width,
                height=height,
                color=mode_color,
                label=f"{query.mode_name}:POS {positive['score_mode']:.3f}",
                font=font,
                line_width=pos_line_width,
            )
        if include_negatives:
            for idx, negative in enumerate(selection.get("negatives") or [], start=1):
                _draw_box(
                    draw,
                    negative["bbox_norm_xyxy"],
                    width=width,
                    height=height,
                    color=NEGATIVE_DEBUG_COLOR,
                    label=f"{query.mode_name}:NEG{idx}",
                    font=font,
                    line_width=neg_line_width,
                    dashed=True,
                )
    return composed


def _safe_filename_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "na"
    chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("._") or "na"


def _crop_box_px(box_norm_xyxy: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(box_norm_xyxy) < 4:
        return None
    x1 = max(0, min(width, int(math.floor(float(box_norm_xyxy[0]) * float(width)))))
    y1 = max(0, min(height, int(math.floor(float(box_norm_xyxy[1]) * float(height)))))
    x2 = max(0, min(width, int(math.ceil(float(box_norm_xyxy[2]) * float(width)))))
    y2 = max(0, min(height, int(math.ceil(float(box_norm_xyxy[3]) * float(height)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _write_best_positive_crop_images(
    *,
    source_rgb: Any,
    out_path: Path,
    width: int,
    height: int,
    query_payloads: Sequence[Dict[str, Any]],
) -> int:
    crop_root = out_path.parent.parent.parent / "debug_best_pos_crops" / out_path.parent.name
    written = 0
    image_token = _safe_filename_token(out_path.stem)
    for payload in query_payloads:
        query = payload["query"]
        selection = payload["selection"]
        positive = selection.get("positive") if isinstance(selection.get("positive"), dict) else None
        if positive is None:
            continue
        crop_box = _crop_box_px(positive.get("bbox_norm_xyxy", []), width=width, height=height)
        if crop_box is None:
            continue
        mode_token = _safe_filename_token(query.mode_name)
        entity_token = _safe_filename_token(query.entity_id)
        query_token = _safe_filename_token(query.query_id)
        score_token = f"{_safe_float(positive.get('score_mode'), 0.0):.3f}".replace(".", "p")
        crop_path = crop_root / mode_token / f"{image_token}__{mode_token}__{entity_token}__{query_token}__s{score_token}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        source_rgb.crop(crop_box).save(crop_path, quality=94)
        written += 1
    return written


def _write_debug_viz(
    *,
    image_path: Path,
    out_path: Path,
    width: int,
    height: int,
    query_payloads: Sequence[Dict[str, Any]],
) -> int:
    if Image is None or ImageDraw is None or ImageColor is None:
        return 0
    with Image.open(image_path) as source:
        base_canvas = source.convert("RGBA")
        source_rgb = source.convert("RGB")
    combined_with_neg = _draw_query_payloads(
        canvas=base_canvas.copy(),
        width=width,
        height=height,
        query_payloads=query_payloads,
        include_negatives=True,
    )
    combined_no_neg = _draw_query_payloads(
        canvas=base_canvas.copy(),
        width=width,
        height=height,
        query_payloads=query_payloads,
        include_negatives=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_with_neg.convert("RGB").save(out_path, quality=92)
    no_neg_path = out_path.parent.parent / f"{out_path.parent.name}_no_neg" / out_path.name
    no_neg_path.parent.mkdir(parents=True, exist_ok=True)
    combined_no_neg.convert("RGB").save(no_neg_path, quality=92)
    mode_names = sorted({payload["query"].mode_name for payload in query_payloads})
    for mode_name in mode_names:
        mode_canvas = _draw_query_payloads(
            canvas=base_canvas.copy(),
            width=width,
            height=height,
            query_payloads=query_payloads,
            include_negatives=True,
            single_mode_name=mode_name,
        )
        mode_path = out_path.parent.parent / f"{out_path.parent.name}_per_mode" / mode_name / out_path.name
        mode_path.parent.mkdir(parents=True, exist_ok=True)
        mode_canvas.convert("RGB").save(mode_path, quality=92)
    return _write_best_positive_crop_images(
        source_rgb=source_rgb,
        out_path=out_path,
        width=width,
        height=height,
        query_payloads=query_payloads,
    )


def _build_summary(
    *,
    image_count: int,
    image_task_count: int,
    image_rows: Sequence[Dict[str, Any]],
    annotation_rows: Sequence[Dict[str, Any]],
    query_status_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    mode_summary: Dict[str, Dict[str, Any]] = {}
    for spec in BASE_CATEGORIES:
        mode_name = str(spec["name"])
        mode_rows = [row for row in query_status_rows if str(row.get("mode_name")) == mode_name]
        positive_rows = [row for row in mode_rows if int(row.get("positive_exists", 0)) == 1]
        negative_annotations = [
            row for row in annotation_rows if str(row.get("mode_name")) == mode_name and int(row.get("gt_flag", 0)) == 0
        ]
        mode_summary[mode_name] = {
            "mode_id": int(spec["id"]),
            "query_count": len(mode_rows),
            "positive_query_count": len(positive_rows),
            "no_positive_query_count": max(0, len(mode_rows) - len(positive_rows)),
            "positive_query_rate": round(len(positive_rows) / float(len(mode_rows)), 6) if mode_rows else 0.0,
            "annotation_count": sum(1 for row in annotation_rows if str(row.get("mode_name")) == mode_name),
            "negative_annotation_count": len(negative_annotations),
        }
    ar_counter = Counter(str(row.get("target_ar", "")) for row in query_status_rows)
    positive_ar_counter = Counter(
        str(row.get("target_ar", "")) for row in query_status_rows if int(row.get("positive_exists", 0)) == 1
    )
    duplicate_positive_counter = Counter()
    for row in annotation_rows:
        if int(row.get("gt_flag", 0)) != 1:
            continue
        key = (int(row.get("image_id", -1)), tuple(round(float(v), 3) for v in row.get("bbox", [0, 0, 0, 0])))
        duplicate_positive_counter[key] += 1
    multi_positive_hits = sum(1 for count in duplicate_positive_counter.values() if count >= 2)
    return {
        "image_count": int(image_count),
        "image_task_count": int(image_task_count),
        "coco_image_count": len(image_rows),
        "annotation_count": len(annotation_rows),
        "positive_annotation_count": sum(1 for row in annotation_rows if int(row.get("gt_flag", 0)) == 1),
        "negative_annotation_count": sum(1 for row in annotation_rows if int(row.get("gt_flag", 0)) == 0),
        "query_count": len(query_status_rows),
        "positive_query_count": sum(1 for row in query_status_rows if int(row.get("positive_exists", 0)) == 1),
        "no_positive_query_count": sum(1 for row in query_status_rows if int(row.get("positive_exists", 0)) == 0),
        "same_crop_multi_positive_count": multi_positive_hits,
        "target_ar": {
            target_ar: {
                "query_count": int(count),
                "positive_query_count": int(positive_ar_counter.get(target_ar, 0)),
            }
            for target_ar, count in sorted(ar_counter.items())
        },
        "mode_summary": mode_summary,
    }


def _select_image_ids(
    *,
    feature_rows_by_id: Dict[str, Dict[str, Any]],
    candidate_rows_by_id: Dict[str, Dict[str, Any]],
    allow_ids: Optional[set[str]],
    max_images: int,
) -> List[str]:
    image_ids = sorted(set(feature_rows_by_id.keys()) & set(candidate_rows_by_id.keys()))
    if allow_ids is not None:
        image_ids = [image_id for image_id in image_ids if image_id in allow_ids]
    if max_images > 0:
        image_ids = image_ids[:max_images]
    return image_ids


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    features_jsonl = Path(args.features_jsonl)
    candidates_jsonl = Path(args.candidates_jsonl)
    out_dir = Path(args.out_dir)
    image_root = Path(args.image_root)
    target_ars = _normalize_target_ars(args.target_ars)
    if not target_ars:
        raise ValueError("target_ars is empty")

    feature_rows_by_id = _load_jsonl(features_jsonl, key_field="image_id", progress=bool(args.progress))
    candidate_rows_by_id = _load_jsonl(candidates_jsonl, key_field="image_id", progress=bool(args.progress))
    image_index = _build_image_index(image_root, progress=bool(args.progress))
    allow_ids = _read_image_ids_file(Path(args.image_ids_file) if args.image_ids_file else None)
    selected_image_ids = _select_image_ids(
        feature_rows_by_id=feature_rows_by_id,
        candidate_rows_by_id=candidate_rows_by_id,
        allow_ids=allow_ids,
        max_images=int(args.max_images),
    )

    image_rows: List[Dict[str, Any]] = []
    annotation_rows: List[Dict[str, Any]] = []
    query_status_rows: List[Dict[str, Any]] = []
    image_task_debug: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    image_task_count = 0
    annotation_id = 1
    image_entry_id = 1

    tracker = ProgressTracker(
        "multimode_build",
        total=len(selected_image_ids),
        unit="images",
        every=25,
        min_seconds=5.0,
        enabled=bool(args.progress),
    )
    for idx, image_id in enumerate(selected_image_ids, start=1):
        feat_row = feature_rows_by_id[image_id]
        candidate_row = candidate_rows_by_id[image_id]
        image_path, relative_file_name = _resolve_image_path(
            image_id=image_id,
            image_root=image_root,
            image_index=image_index,
            candidate_row=candidate_row,
            feat_row=feat_row,
        )
        width, height = _resolve_image_size(candidate_row=candidate_row, feat_row=feat_row, image_path=image_path)
        image_ar = _safe_float(candidate_row.get("image_ar"), float(width) / max(1.0, float(height)))
        atoms = build_entity_atoms(feat_row, width=width, height=height)
        routing = feat_row.get("routing") if isinstance(feat_row.get("routing"), dict) else {}

        for target_ar in target_ars:
            base_candidates = build_global_candidate_bank(candidate_row, target_ar)
            if not base_candidates:
                continue
            queries = [build_query_guidance(query) for query in build_mode_queries(image_id=image_id, target_ar=target_ar, atoms=atoms, routing=routing)]
            if not queries:
                continue
            image_rows.append(
                build_image_entry(
                    image_entry_id=image_entry_id,
                    source_image_id=image_id,
                    file_name=relative_file_name,
                    width=width,
                    height=height,
                    target_ar=target_ar,
                )
            )
            image_task_count += 1
            for query in queries:
                candidates = expand_query_candidates(base_candidates=base_candidates, query=query, image_ar=image_ar)
                scored = score_query_candidates(
                    query=query,
                    candidates=candidates,
                    feat_row=feat_row,
                    width=width,
                    height=height,
                    image_ar=image_ar,
                )
                selection = select_query_results(query=query, scored_candidates=scored)
                query_status_rows.append(
                    _query_status_row(
                        source_image_id=image_id,
                        target_ar=target_ar,
                        query=query,
                        selection=selection,
                    )
                )
                positive = selection.get("positive") if isinstance(selection.get("positive"), dict) else None
                if positive is None:
                    continue
                annotation_rows.append(
                    build_annotation_entry(
                        annotation_id=annotation_id,
                        image_entry_id=image_entry_id,
                        width=width,
                        height=height,
                        query_id=query.query_id,
                        entity_id=query.entity_id,
                        mode_name=query.mode_name,
                        category_id=query.mode_id,
                        source_route_mode=query.route_mode,
                        bbox_norm_xyxy=positive["bbox_norm_xyxy"],
                        score_mode=float(positive.get("score_mode", 0.0)),
                        gt_flag=1,
                        is_best=1,
                        attributes=_annotation_attributes(
                            query=query,
                            selected_row=positive,
                            extra={
                                "negative_reason": "",
                                "hard_reject_reasons": [],
                                "query_has_positive": 1,
                            },
                        ),
                    )
                )
                annotation_id += 1
                if int(args.include_optional_negatives) == 1:
                    for negative in selection.get("negatives") or []:
                        annotation_rows.append(
                            build_annotation_entry(
                                annotation_id=annotation_id,
                                image_entry_id=image_entry_id,
                                width=width,
                                height=height,
                                query_id=query.query_id,
                                entity_id=query.entity_id,
                                mode_name=query.mode_name,
                                category_id=query.mode_id,
                                source_route_mode=query.route_mode,
                                bbox_norm_xyxy=negative["bbox_norm_xyxy"],
                                score_mode=float(negative.get("score_mode", 0.0)),
                                gt_flag=0,
                                is_best=0,
                                attributes=_annotation_attributes(
                                    query=query,
                                    selected_row=negative,
                                    extra={
                                        "negative_reason": str(negative.get("negative_reason", "")),
                                        "hard_reject_reasons": list(negative.get("hard_reject_reasons") or []),
                                        "query_has_positive": 1,
                                    },
                                ),
                            )
                        )
                        annotation_id += 1
                if int(args.write_debug_viz) == 1:
                    image_task_debug[(image_id, target_ar)].append({"query": query, "selection": selection})
            if int(args.write_debug_viz) == 1 and (image_id, target_ar) not in image_task_debug:
                image_task_debug[(image_id, target_ar)] = []
            image_entry_id += 1
        tracker.update(idx)
    tracker.finish(len(selected_image_ids))

    coco_dataset = build_coco_dataset(image_rows=image_rows, annotation_rows=annotation_rows)
    summary = _build_summary(
        image_count=len(selected_image_ids),
        image_task_count=image_task_count,
        image_rows=image_rows,
        annotation_rows=annotation_rows,
        query_status_rows=query_status_rows,
    )
    summary["input"] = {
        "features_jsonl": str(features_jsonl),
        "candidates_jsonl": str(candidates_jsonl),
        "image_root": str(image_root),
        "target_ars": target_ars,
        "max_images": int(args.max_images),
        "include_optional_negatives": int(args.include_optional_negatives),
        "write_debug_viz": int(args.write_debug_viz),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "mode_query_status.jsonl", query_status_rows)
    write_json(out_dir / "coco" / "instances_multimode_training_labels.json", coco_dataset)
    write_json(out_dir / "categories.json", {"categories": BASE_CATEGORIES})

    if int(args.write_debug_viz) == 1 and image_task_debug:
        limit = int(args.debug_viz_limit)
        written = 0
        best_pos_crop_written = 0
        for (image_id, target_ar), payloads in sorted(image_task_debug.items()):
            if limit > 0 and written >= limit:
                break
            feat_row = feature_rows_by_id[image_id]
            candidate_row = candidate_rows_by_id[image_id]
            image_path, _ = _resolve_image_path(
                image_id=image_id,
                image_root=image_root,
                image_index=image_index,
                candidate_row=candidate_row,
                feat_row=feat_row,
            )
            width, height = _resolve_image_size(candidate_row=candidate_row, feat_row=feat_row, image_path=image_path)
            out_path = out_dir / "debug_viz" / target_ar.replace(":", "x") / f"{image_id}.jpg"
            best_pos_crop_written += _write_debug_viz(
                image_path=image_path,
                out_path=out_path,
                width=width,
                height=height,
                query_payloads=payloads,
            )
            written += 1
        summary["debug_viz_count"] = written
        summary["debug_best_pos_crop_count"] = best_pos_crop_written
        summary["debug_best_pos_crop_dir"] = str(out_dir / "debug_best_pos_crops")
        write_json(out_dir / "summary.json", summary)

    progress_log(
        f"multimode build done images={summary['image_count']} tasks={summary['image_task_count']} "
        f"queries={summary['query_count']} positives={summary['positive_query_count']}",
        enabled=bool(args.progress),
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimode training labels from routed features and candidates.")
    parser.add_argument("--features_jsonl", required=True)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_ars", default="FREE,1:1,9:16,16:9,3:4,4:3")
    parser.add_argument("--image_ids_file", default="")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--include_optional_negatives", type=int, default=1)
    parser.add_argument("--write_debug_viz", type=int, default=0)
    parser.add_argument("--debug_viz_limit", type=int, default=16)
    parser.add_argument("--progress", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
