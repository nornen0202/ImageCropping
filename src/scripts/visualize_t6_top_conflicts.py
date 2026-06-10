#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def resolve_image_path(image_id: str, image_roots: Sequence[Path]) -> Path | None:
    names = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png"]
    for root in image_roots:
        for name in names:
            path = root / name
            if path.exists():
                return path
        for sub in ("train", "val", "test", "images"):
            for name in names:
                path = root / sub / name
                if path.exists():
                    return path
    return None


def norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [clamp(safe_float(v), 0.0, 1.0) for v in box[:4]]
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def is_conflict(row: dict[str, Any]) -> bool:
    return bool(row.get("fatal_flag", False)) or bool(row.get("contradiction_flag", False)) or str(row.get("label_quality_tier", "")) == "hard_negative"


def is_hard_fatal(row: dict[str, Any]) -> bool:
    return bool(row.get("fatal_flag", False)) or str(row.get("label_quality_tier", "")) == "hard_negative"


def is_negative_for_review(row: dict[str, Any], *, predicate: str) -> bool:
    if predicate == "hard_fatal":
        return is_hard_fatal(row)
    if predicate == "conflict":
        return is_conflict(row)
    raise ValueError(f"unsupported conflict predicate: {predicate}")


def sort_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            1 if row.get("selected_by_ranker") else 0,
            safe_float(row.get("score_rank_pct_by_image"), 0.0),
            safe_float(row.get("ranker_score"), 0.0),
        ),
        reverse=True,
    )


def best_nonfatal(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for row in sort_rows(rows):
        if not is_hard_fatal(row):
            return row
    return None


def mos_best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            safe_float(row.get("mos"), 0.0),
            safe_float(row.get("score_rank_pct_by_image"), 0.0),
            safe_float(row.get("ranker_score"), 0.0),
        ),
    )


def normalized_training_positive(row: dict[str, Any]) -> dict[str, Any] | None:
    targets = list(row.get("matching_targets") or [])
    if not targets:
        return None
    target = dict(targets[0])
    teacher = dict(target.get("t6_teacher") or {})
    out = {
        "candidate_id": str(target.get("candidate_id", "")),
        "bbox_norm_xyxy": target.get("bbox_norm_xyxy"),
        "ranker_score": safe_float(teacher.get("ranker_score", target.get("score_targets", {}).get("t6_ranker_score", 0.0))),
        "score_rank_pct_by_image": safe_float(
            teacher.get("score_rank_pct_by_image", target.get("score_targets", {}).get("score_rank_pct", target.get("score_rank_pct", 0.0)))
        ),
        "mos": safe_float(teacher.get("mos", 0.0)),
        "label_quality_tier": str(teacher.get("label_quality_tier", target.get("training_bucket", ""))),
        "training_weight": safe_float(teacher.get("training_weight", target.get("candidate_weight", 1.0))),
        "fatal_flag": bool(teacher.get("fatal_flag", target.get("is_unsafe_negative", False))),
        "contradiction_flag": bool(teacher.get("contradiction_flag", False)),
        "explanation_score": safe_float(teacher.get("explanation_score", 0.0)),
        "explanation_confidence": safe_float(teacher.get("explanation_confidence", 0.0)),
        "warnings": list(teacher.get("warnings") or target.get("reject_tags") or []),
        "checklist_labels": teacher.get("checklist_labels") or {},
        "unsafe_after_converter": bool(teacher.get("unsafe_after_converter", target.get("is_unsafe_negative", False))),
        "is_positive_candidate": bool(target.get("is_positive_candidate", True)),
        "source": "corrected_v2b_training_label",
    }
    return out


def load_training_positive_map(path: Path) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for row in read_jsonl(path):
        out[str(row.get("image_id", ""))] = normalized_training_positive(row)
    return out


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        next_line = word if not current else f"{current} {word}"
        if len(next_line) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = next_line
    if current:
        lines.append(current)
    return lines or [""]


def short_box(box: Sequence[float]) -> str:
    return "[" + ",".join(f"{safe_float(v):.3f}" for v in list(box)[:4]) + "]"


def row_summary(row: dict[str, Any] | None, *, prefix: str) -> list[str]:
    if row is None:
        return [f"{prefix}: none"]
    warnings = row.get("warnings") or []
    checklist = row.get("checklist_labels") or {}
    important = []
    for key in ("subject_cut", "safety", "hard_safety", "headroom", "lookroom", "horizon", "context", "composition", "subject"):
        if key in checklist:
            important.append(f"{key}={checklist[key]}")
    lines = [
        f"{prefix}: {row.get('candidate_id')}",
        f"  pct={safe_float(row.get('score_rank_pct_by_image')):.4f} raw={safe_float(row.get('ranker_score')):.4f} mos={safe_float(row.get('mos')):.2f}",
        f"  tier={row.get('label_quality_tier')} weight={safe_float(row.get('training_weight')):.3f}",
        f"  fatal={bool(row.get('fatal_flag'))} contradiction={bool(row.get('contradiction_flag'))}",
        f"  unsafe_after_converter={bool(row.get('unsafe_after_converter')) if 'unsafe_after_converter' in row else 'n/a'}",
        f"  expl={safe_float(row.get('explanation_score')):.3f} conf={safe_float(row.get('explanation_confidence')):.3f}",
        f"  box={short_box(row.get('bbox_norm_xyxy') or [0, 0, 1, 1])}",
    ]
    if warnings:
        lines.extend(f"  warn: {line}" for line in wrap_text(",".join(str(v) for v in warnings), max_chars=52))
    if important:
        lines.extend(f"  chk: {line}" for line in wrap_text(", ".join(important), max_chars=52))
    return lines


def make_crop_thumb(
    image: Image.Image,
    row: dict[str, Any] | None,
    *,
    label: str,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> Image.Image:
    thumb = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(thumb)
    title_font = load_font(18)
    font = load_font(15)
    draw.rectangle((0, 0, width - 1, height - 1), outline=color, width=4)
    draw.text((10, 8), label, fill=color, font=title_font)
    if row is None:
        draw.text((10, 42), "none / image skipped", fill=(40, 40, 40), font=font)
        return thumb
    box = norm_to_px(row.get("bbox_norm_xyxy") or [0, 0, 1, 1], image.width, image.height)
    crop = image.crop(box)
    max_w = width - 20
    max_h = max(40, height - 92)
    crop.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    thumb.paste(crop, ((width - crop.width) // 2, 42))
    y = 48 + crop.height
    lines = [
        f"id={row.get('candidate_id')} mos={safe_float(row.get('mos')):.2f}",
        f"pct={safe_float(row.get('score_rank_pct_by_image')):.3f} raw={safe_float(row.get('ranker_score')):.3f}",
        f"tier={row.get('label_quality_tier')} fatal={bool(row.get('fatal_flag'))} conflict={bool(row.get('contradiction_flag'))}",
    ]
    for line in lines:
        if y > height - 18:
            break
        draw.text((10, y), line, fill=(25, 25, 25), font=font)
        y += 18
    return thumb


def draw_debug_panel(
    image: Image.Image,
    original_image: Image.Image,
    *,
    image_id: str,
    split: str,
    selected: dict[str, Any],
    alt: dict[str, Any] | None,
    training_positive: dict[str, Any] | None,
    best_mos_row: dict[str, Any] | None,
    rows: Sequence[dict[str, Any]],
    min_selected_mos: float,
) -> Image.Image:
    panel_width = max(560, min(820, int(image.width * 0.55)))
    strip_h = max(240, min(360, int(image.height * 0.32)))
    out = Image.new("RGB", (image.width + panel_width, image.height + strip_h), (245, 245, 245))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    font = load_font(max(15, min(22, image.width // 55)))
    title_font = load_font(max(18, min(26, image.width // 45)))
    x = image.width + 18
    y = 18
    line_h = int(getattr(font, "size", 16) * 1.35)
    title_h = int(getattr(title_font, "size", 20) * 1.45)

    title = f"{split} / {image_id} / GAIC MOS>={min_selected_mos:.1f} top1 conflict"
    draw.text((x, y), title, fill=(120, 0, 0), font=title_font)
    y += title_h

    for line in row_summary(selected, prefix="T6_TOP1_GAIC_POS_SSTK_NEG"):
        draw.text((x, y), line, fill=(25, 25, 25), font=font)
        y += line_h
    y += line_h // 2
    for line in row_summary(training_positive, prefix="TRAIN_LABEL_TOP1_POS"):
        draw.text((x, y), line, fill=(25, 80, 25), font=font)
        y += line_h
    y += line_h // 2
    for line in row_summary(best_mos_row, prefix="GAIC_MOS_BEST"):
        draw.text((x, y), line, fill=(0, 75, 110), font=font)
        y += line_h

    y += line_h // 2
    for line in row_summary(alt, prefix="BEST_NONFATAL_DEBUG"):
        draw.text((x, y), line, fill=(25, 70, 25), font=font)
        y += line_h

    y += line_h // 2
    draw.text((x, y), "Top candidates:", fill=(0, 0, 0), font=title_font)
    y += title_h
    for rank, row in enumerate(sort_rows(rows)[:8], start=1):
        mark = "!" if is_hard_fatal(row) else " "
        text = (
            f"{rank:02d}{mark} {row.get('candidate_id')} "
            f"pct={safe_float(row.get('score_rank_pct_by_image')):.3f} "
            f"mos={safe_float(row.get('mos')):.2f} "
            f"tier={row.get('label_quality_tier')}"
        )
        for wrapped in wrap_text(text, max_chars=64):
            if y + line_h > out.height - 12:
                return out
            draw.text((x, y), wrapped, fill=(60, 60, 60), font=font)
            y += line_h

    strip_y = image.height
    draw.rectangle((0, strip_y, out.width, out.height), fill=(232, 232, 232))
    gap = 12
    thumb_w = max(280, (out.width - gap * 4) // 3)
    thumb_h = strip_h - 18
    thumbs = [
        make_crop_thumb(original_image, selected, label="RED: GAIC positive but SSTK negative", color=(255, 0, 0), width=thumb_w, height=thumb_h),
        make_crop_thumb(original_image, training_positive, label="GREEN: actual train top-1 positive", color=(0, 190, 75), width=thumb_w, height=thumb_h),
        make_crop_thumb(original_image, best_mos_row, label="CYAN: best GAIC MOS crop", color=(0, 150, 220), width=thumb_w, height=thumb_h),
    ]
    x0 = gap
    for thumb in thumbs:
        out.paste(thumb, (x0, strip_y + 9))
        x0 += thumb_w + gap
    return out


def visualize_group(
    image_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    image_roots: Sequence[Path],
    output_dir: Path,
    top_k: int,
    min_selected_mos: float,
    conflict_predicate: str,
    training_positive: dict[str, Any] | None,
) -> dict[str, Any] | None:
    rows_sorted = sort_rows(rows)
    if not rows_sorted:
        return None
    selected = next((row for row in rows_sorted if row.get("selected_by_ranker")), rows_sorted[0])
    if safe_float(selected.get("mos"), 0.0) < float(min_selected_mos):
        return None
    if not is_negative_for_review(selected, predicate=conflict_predicate):
        return None

    image_path = resolve_image_path(image_id, image_roots)
    if image_path is None:
        return None
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    width, height = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font(max(15, min(22, width // 60)))
    palette = [(255, 40, 40), (0, 185, 75), (255, 190, 0), (0, 160, 230), (180, 80, 255), (255, 110, 0)]
    alt = best_nonfatal(rows_sorted)
    alt_id = str(alt.get("candidate_id")) if alt else ""
    training_positive_id = str(training_positive.get("candidate_id")) if training_positive else ""
    best_mos_row = mos_best(rows_sorted)
    best_mos_id = str(best_mos_row.get("candidate_id")) if best_mos_row else ""

    for rank, row in enumerate(rows_sorted[: max(1, top_k)], start=1):
        cid = str(row.get("candidate_id", ""))
        color = palette[(rank - 1) % len(palette)]
        stroke = 2
        if cid == str(selected.get("candidate_id")):
            color = (255, 0, 0)
            stroke = 6
        elif cid == training_positive_id:
            color = (0, 210, 80)
            stroke = 5
        elif cid == best_mos_id:
            color = (0, 160, 230)
            stroke = 5
        elif cid == alt_id:
            color = (255, 190, 0)
            stroke = 4
        box = norm_to_px(row.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        for offset in range(stroke):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
        tag = "FATAL" if is_hard_fatal(row) else "OK"
        label = f"#{rank} {tag} pct={safe_float(row.get('score_rank_pct_by_image')):.3f} mos={safe_float(row.get('mos')):.2f}"
        draw_label(draw, (max(0, box[0] + 4), max(0, box[1] + 4)), label, color, font)

    for special, color, stroke, label_prefix in (
        (training_positive, (0, 210, 80), 5, "TRAIN_POS"),
        (best_mos_row, (0, 160, 230), 5, "MOS_BEST"),
    ):
        if special is None:
            continue
        cid = str(special.get("candidate_id", ""))
        if cid in {str(row.get("candidate_id", "")) for row in rows_sorted[: max(1, top_k)]}:
            continue
        box = norm_to_px(special.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        for offset in range(stroke):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
        label = f"{label_prefix} mos={safe_float(special.get('mos')):.2f}"
        draw_label(draw, (max(0, box[0] + 4), max(0, box[1] + 4)), label, color, font)

    canvas = draw_debug_panel(
        overlay,
        image,
        image_id=image_id,
        split=split,
        selected=selected,
        alt=alt,
        training_positive=training_positive,
        best_mos_row=best_mos_row,
        rows=rows_sorted,
        min_selected_mos=float(min_selected_mos),
    )
    overlay_dir = output_dir / split / "overlays"
    crop_dir = output_dir / split / "selected_crops"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{split}_{image_id}_{selected.get('candidate_id')}_mos{safe_float(selected.get('mos')):.2f}_conflict_top.png"
    canvas.save(overlay_path, format="PNG", compress_level=3)

    selected_crop_path = crop_dir / f"{split}_{image_id}_{selected.get('candidate_id')}_selected.png"
    selected_box = norm_to_px(selected.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
    image.crop(selected_box).save(selected_crop_path, format="PNG", compress_level=3)
    alt_crop_path = None
    if alt is not None:
        alt_crop_path = crop_dir / f"{split}_{image_id}_{alt.get('candidate_id')}_best_nonfatal.png"
        alt_box = norm_to_px(alt.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        image.crop(alt_box).save(alt_crop_path, format="PNG", compress_level=3)
    training_positive_crop_path = None
    if training_positive is not None:
        training_positive_crop_path = crop_dir / f"{split}_{image_id}_{training_positive.get('candidate_id')}_training_positive.png"
        train_box = norm_to_px(training_positive.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        image.crop(train_box).save(training_positive_crop_path, format="PNG", compress_level=3)
    best_mos_crop_path = None
    if best_mos_row is not None:
        best_mos_crop_path = crop_dir / f"{split}_{image_id}_{best_mos_row.get('candidate_id')}_mos_best.png"
        best_box = norm_to_px(best_mos_row.get("bbox_norm_xyxy") or [0, 0, 1, 1], width, height)
        image.crop(best_box).save(best_mos_crop_path, format="PNG", compress_level=3)

    return {
        "split": split,
        "image_id": image_id,
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "selected_crop_path": str(selected_crop_path),
        "best_nonfatal_crop_path": str(alt_crop_path) if alt_crop_path else None,
        "training_positive_crop_path": str(training_positive_crop_path) if training_positive_crop_path else None,
        "gaic_mos_best_crop_path": str(best_mos_crop_path) if best_mos_crop_path else None,
        "candidate_count": len(rows_sorted),
        "min_selected_mos": float(min_selected_mos),
        "conflict_predicate": str(conflict_predicate),
        "selected": {
            "candidate_id": str(selected.get("candidate_id", "")),
            "bbox_norm_xyxy": selected.get("bbox_norm_xyxy"),
            "ranker_score": safe_float(selected.get("ranker_score")),
            "score_rank_pct_by_image": safe_float(selected.get("score_rank_pct_by_image")),
            "mos": safe_float(selected.get("mos")),
            "label_quality_tier": str(selected.get("label_quality_tier", "")),
            "training_weight": safe_float(selected.get("training_weight")),
            "fatal_flag": bool(selected.get("fatal_flag")),
            "contradiction_flag": bool(selected.get("contradiction_flag")),
            "explanation_score": safe_float(selected.get("explanation_score")),
            "explanation_confidence": safe_float(selected.get("explanation_confidence")),
            "warnings": list(selected.get("warnings") or []),
            "checklist_labels": selected.get("checklist_labels") or {},
        },
        "training_positive": None
        if training_positive is None
        else {
            "candidate_id": str(training_positive.get("candidate_id", "")),
            "bbox_norm_xyxy": training_positive.get("bbox_norm_xyxy"),
            "ranker_score": safe_float(training_positive.get("ranker_score")),
            "score_rank_pct_by_image": safe_float(training_positive.get("score_rank_pct_by_image")),
            "mos": safe_float(training_positive.get("mos")),
            "label_quality_tier": str(training_positive.get("label_quality_tier", "")),
            "training_weight": safe_float(training_positive.get("training_weight")),
            "fatal_flag": bool(training_positive.get("fatal_flag")),
            "contradiction_flag": bool(training_positive.get("contradiction_flag")),
            "unsafe_after_converter": bool(training_positive.get("unsafe_after_converter", False)),
            "explanation_score": safe_float(training_positive.get("explanation_score")),
            "explanation_confidence": safe_float(training_positive.get("explanation_confidence")),
            "warnings": list(training_positive.get("warnings") or []),
            "checklist_labels": training_positive.get("checklist_labels") or {},
        },
        "gaic_mos_best": None
        if best_mos_row is None
        else {
            "candidate_id": str(best_mos_row.get("candidate_id", "")),
            "bbox_norm_xyxy": best_mos_row.get("bbox_norm_xyxy"),
            "ranker_score": safe_float(best_mos_row.get("ranker_score")),
            "score_rank_pct_by_image": safe_float(best_mos_row.get("score_rank_pct_by_image")),
            "mos": safe_float(best_mos_row.get("mos")),
            "label_quality_tier": str(best_mos_row.get("label_quality_tier", "")),
            "training_weight": safe_float(best_mos_row.get("training_weight")),
            "fatal_flag": bool(best_mos_row.get("fatal_flag")),
            "contradiction_flag": bool(best_mos_row.get("contradiction_flag")),
            "explanation_score": safe_float(best_mos_row.get("explanation_score")),
            "explanation_confidence": safe_float(best_mos_row.get("explanation_confidence")),
            "warnings": list(best_mos_row.get("warnings") or []),
            "checklist_labels": best_mos_row.get("checklist_labels") or {},
        },
        "best_nonfatal": None
        if alt is None
        else {
            "candidate_id": str(alt.get("candidate_id", "")),
            "bbox_norm_xyxy": alt.get("bbox_norm_xyxy"),
            "ranker_score": safe_float(alt.get("ranker_score")),
            "score_rank_pct_by_image": safe_float(alt.get("score_rank_pct_by_image")),
            "mos": safe_float(alt.get("mos")),
            "label_quality_tier": str(alt.get("label_quality_tier", "")),
            "training_weight": safe_float(alt.get("training_weight")),
            "fatal_flag": bool(alt.get("fatal_flag")),
            "contradiction_flag": bool(alt.get("contradiction_flag")),
            "explanation_score": safe_float(alt.get("explanation_score")),
            "explanation_confidence": safe_float(alt.get("explanation_confidence")),
            "warnings": list(alt.get("warnings") or []),
            "checklist_labels": alt.get("checklist_labels") or {},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize T6 selected top-1 crops that are fatal/hard-negative.")
    parser.add_argument("--t6_labels_jsonl", type=Path, action="append", required=True, help="Split JSONL. Repeatable.")
    parser.add_argument("--split", action="append", required=True, help="Split name matching each --t6_labels_jsonl.")
    parser.add_argument("--image_root", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_images_per_split", type=int, default=0)
    parser.add_argument("--training_batch_jsonl", type=Path, action="append", default=None, help="Corrected MobileCropNet v4 batch label JSONL. Repeatable and aligned with --split.")
    parser.add_argument("--min_selected_mos", type=float, default=0.0, help="Only visualize selected top-1 crops whose GAIC MOS is at least this value.")
    parser.add_argument("--conflict_predicate", choices=["hard_fatal", "conflict"], default="hard_fatal", help="hard_fatal=fatal or hard_negative; conflict also includes contradiction_flag.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.t6_labels_jsonl) != len(args.split):
        raise SystemExit("--t6_labels_jsonl and --split must have the same count")
    training_maps: dict[str, dict[str, dict[str, Any] | None]] = {}
    if args.training_batch_jsonl:
        if len(args.training_batch_jsonl) != len(args.split):
            raise SystemExit("--training_batch_jsonl must match --split count when provided")
        for split, batch_path in zip(args.split, args.training_batch_jsonl):
            training_maps[str(split)] = load_training_positive_map(Path(batch_path))
    all_records: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split, jsonl_path in zip(args.split, args.t6_labels_jsonl):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(Path(jsonl_path)):
            grouped[str(row.get("image_id", ""))].append(row)
        records: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        training_map = training_maps.get(str(split), {})
        for image_id, rows in sorted(grouped.items()):
            rows_sorted = sort_rows(rows)
            if not rows_sorted:
                continue
            selected = next((row for row in rows_sorted if row.get("selected_by_ranker")), rows_sorted[0])
            if safe_float(selected.get("mos"), 0.0) < float(args.min_selected_mos):
                continue
            if is_negative_for_review(selected, predicate=str(args.conflict_predicate)):
                counters["selected_top_conflict_images"] += 1
                counters[f"selected_tier:{selected.get('label_quality_tier')}"] += 1
                if bool(selected.get("fatal_flag")):
                    counters["selected_fatal_flag"] += 1
                if bool(selected.get("contradiction_flag")):
                    counters["selected_contradiction_flag"] += 1
                if training_map and image_id not in training_map:
                    counters["missing_training_positive_image"] += 1
                elif training_map.get(image_id) is None:
                    counters["no_training_positive_image"] += 1
                if int(args.max_images_per_split) > 0 and len(records) >= int(args.max_images_per_split):
                    continue
                record = visualize_group(
                    image_id,
                    rows_sorted,
                    split=str(split),
                    image_roots=[Path(p) for p in args.image_root],
                    output_dir=Path(args.output_dir),
                    top_k=int(args.top_k),
                    min_selected_mos=float(args.min_selected_mos),
                    conflict_predicate=str(args.conflict_predicate),
                    training_positive=training_map.get(image_id),
                )
                if record is not None:
                    records.append(record)
        manifest_path = Path(args.output_dir) / str(split) / "fatal_top_manifest.jsonl"
        write_jsonl(manifest_path, records)
        split_summaries[str(split)] = {
            "t6_labels_jsonl": str(jsonl_path),
            "training_batch_jsonl": str(args.training_batch_jsonl[args.split.index(split)]) if args.training_batch_jsonl else None,
            "image_count": len(grouped),
            "visualized_count": len(records),
            "manifest_path": str(manifest_path),
            "counters": dict(counters),
        }
        all_records.extend(records)
    summary = {
        "status": "ok",
        "output_dir": str(args.output_dir),
        "top_k": int(args.top_k),
        "max_images_per_split": int(args.max_images_per_split),
        "min_selected_mos": float(args.min_selected_mos),
        "conflict_predicate": str(args.conflict_predicate),
        "split_summaries": split_summaries,
        "total_visualized": len(all_records),
        "sample_records": all_records[:20],
    }
    write_json(Path(args.output_dir) / "fatal_top_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
