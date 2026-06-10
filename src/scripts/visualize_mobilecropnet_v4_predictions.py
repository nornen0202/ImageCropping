#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import CHECKLIST_CLASS_KEYS, PERSON_ONLY_WHY_TAGS, checklist_applicability_by_mode, target_ar_value

NA_LABELS = {"", "na", "n/a", "none", "null", "nan"}
SCORE_ALIASES = {
    "subject_coverage_ratio": ("subject_coverage_ratio",),
    "subject_scale_ratio": ("subject_scale_ratio",),
    "headroom_ratio": ("headroom_ratio", "headroom_ratio_norm", "C_headroom"),
    "lookroom_ratio": ("lookroom_ratio", "lookroom_ratio_norm", "C_lookroom"),
    "third_dist": ("third_dist", "third_strength", "placement_reward_third"),
    "phi_dist": ("phi_dist", "phi_strength", "placement_reward_phi"),
    "center_dist": ("center_dist", "center_strength", "placement_reward_center"),
    "context_value": ("context_value", "C_context"),
    "symmetry_score": ("symmetry_score",),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write high-quality PNG visualizations for MobileCropNet v4 predictions.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--max_side", type=int, default=1400)
    return parser


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _box_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
        int(round(max(0.0, min(1.0, x2)) * width)),
        int(round(max(0.0, min(1.0, y2)) * height)),
    )


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], size: tuple[int, int], color: tuple[int, int, int], label: str, font: ImageFont.ImageFont, width: int) -> None:
    w, h = size
    xy = _box_px(box, w, h)
    draw.rectangle(xy, outline=color, width=width)
    tw = int(draw.textlength(label, font=font))
    th = int(getattr(font, "size", 18) * 1.35)
    x1, y1, _, _ = xy
    bg = [x1, max(0, y1 - th - 4), min(w, x1 + tw + 12), max(th + 4, y1)]
    draw.rectangle(bg, fill=color)
    draw.text((bg[0] + 6, bg[1] + 2), label, fill=(255, 255, 255), font=font)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_panel_line(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int,
) -> int:
    for line in _wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += int(getattr(font, "size", 16) * 1.35) + line_gap
    return y


def _score_label(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "na"


def _is_na_label(value: Any) -> bool:
    return str(value or "").strip().lower() in NA_LABELS


def _label_with_score(payload: Any, *, default_label: str = "na", na_display: str = "unlabeled") -> str:
    if not isinstance(payload, dict):
        return default_label
    raw_label = str(payload.get("label", default_label))
    label = str(payload.get("display_label") or raw_label)
    if label == "not_applicable":
        return "not_applicable"
    score = _score_label(payload.get("score"))
    if score == "na":
        return label
    if _is_na_label(raw_label):
        return f"{na_display} p={score}" if score != "na" else na_display
    return f"{label} {score}"


def _tag_text(tags: Any, limit: int = 6) -> str:
    if not isinstance(tags, list):
        return "na"
    rows = []
    for item in tags[:limit]:
        if isinstance(item, dict):
            rows.append(f"{item.get('tag', 'tag')} {_score_label(item.get('score'))}")
        else:
            rows.append(str(item))
    return ", ".join(rows) if rows else "na"


def _detail_item_text(items: dict[str, Any], key: str) -> str:
    value = items.get(key)
    if isinstance(value, dict):
        return _label_with_score(value)
    return "na"


def _is_label_key_applicable(row: dict[str, Any], key: str) -> bool:
    subject_mode = str(row.get("subject_mode_label", "other_ambiguous"))
    app = dict(zip(CHECKLIST_CLASS_KEYS, checklist_applicability_by_mode(subject_mode)))
    return bool(app.get(key, 0.0) > 0.0)


def _model_detail_item_text(row: dict[str, Any], items: dict[str, Any], key: str) -> str:
    if not _is_label_key_applicable(row, key):
        return "not_applicable"
    return _detail_item_text(items, key)


def _model_score_item_text(row: dict[str, Any], detailed: dict[str, Any], scores: dict[str, Any], detail_key: str, score_key: str) -> str:
    if not _is_label_key_applicable(row, detail_key):
        return "not_applicable"
    item = detailed.get(detail_key)
    if isinstance(item, dict) and str(item.get("display_label")) == "not_applicable":
        return "not_applicable"
    if isinstance(item, dict) and item.get("available") is False:
        return str(item.get("display_label") or "unavailable")
    return _score_item_text(scores, score_key)


def _score_item_text(scores: dict[str, Any], key: str) -> str:
    if not isinstance(scores, dict):
        return "na"
    for alias in SCORE_ALIASES.get(key, (key,)):
        if scores.get(alias) is not None:
            return _score_label(scores.get(alias), 2)
    return "na"


def _teacher_label_text(row: dict[str, Any], labels: dict[str, Any], key: str) -> str:
    if not _is_label_key_applicable(row, key):
        return "not_applicable"
    return str(labels.get(key, "na"))


def _teacher_score_text(row: dict[str, Any], labels: dict[str, Any], scores: dict[str, Any], label_key: str, score_key: str) -> str:
    if _teacher_label_text(row, labels, label_key) == "not_applicable":
        return "not_applicable"
    return _score_item_text(scores, score_key)


def _teacher_tag_text(row: dict[str, Any], tags: Any) -> str:
    if str(row.get("subject_mode_label", "")) not in {"object_single", "object_multi"}:
        return _tag_text(tags)
    if not isinstance(tags, list):
        return "na"
    filtered = []
    for tag in tags:
        name = str(tag.get("tag")) if isinstance(tag, dict) else str(tag)
        if name not in PERSON_ONLY_WHY_TAGS:
            filtered.append(tag)
    return _tag_text(filtered)


def _proposal_for_visualization(row: dict[str, Any]) -> dict[str, Any] | None:
    proposal = row.get("proposal_top1_target_ar")
    if isinstance(proposal, dict) and proposal.get("bbox_norm_xyxy"):
        return proposal
    proposals = [p for p in (row.get("proposals") or []) if isinstance(p, dict) and p.get("bbox_norm_xyxy")]
    if not proposals:
        return None
    if target_ar_value(row.get("target_ar")) is None:
        return proposals[0]
    compatible = [p for p in proposals if bool(p.get("target_ar_compatible", False))]
    if compatible:
        return compatible[0]
    return min(proposals, key=lambda p: float(p.get("target_ar_log_error") or 1e9))


def _subject_box_payload(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    subject_box = row.get("subject_box") if isinstance(row.get("subject_box"), dict) else {}
    payload = subject_box.get(key) if isinstance(subject_box, dict) else None
    if isinstance(payload, dict) and payload.get("bbox_norm_xyxy"):
        return payload
    return None


def _non_na_label_count(labels: Any) -> int:
    if not isinstance(labels, dict):
        return 0
    return sum(1 for value in labels.values() if not _is_na_label(value))


def _box_iou(a: Any, b: Any) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _teacher_detail_payload(row: dict[str, Any], top: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Any, str]:
    labels = top.get("teacher_checklist_labels")
    scores = top.get("teacher_checklist_scores")
    tags = top.get("teacher_why_tags")
    if _non_na_label_count(labels) > 0:
        return labels, scores if isinstance(scores, dict) else {}, tags, "source: model top crop"

    top_box = top.get("bbox_norm_xyxy")
    best_candidate: dict[str, Any] | None = None
    best_iou = -1.0
    best_non_na = -1
    candidates = []
    for bucket_key in ("candidates", "top_candidates"):
        for candidate in row.get(bucket_key) or []:
            if isinstance(candidate, dict):
                candidates.append(candidate)
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for candidate in candidates:
        c_labels = candidate.get("teacher_checklist_labels")
        non_na = _non_na_label_count(c_labels)
        if non_na <= 0:
            continue
        c_box = candidate.get("bbox_norm_xyxy")
        key = (str(candidate.get("candidate_id", "")), tuple(round(float(v), 5) for v in c_box[:4]) if isinstance(c_box, (list, tuple)) and len(c_box) >= 4 else ())
        if key in seen:
            continue
        seen.add(key)
        iou = _box_iou(top_box, c_box)
        if iou > best_iou or (abs(iou - best_iou) < 1e-6 and non_na > best_non_na):
            best_candidate = candidate
            best_iou = iou
            best_non_na = non_na

    if isinstance(best_candidate, dict):
        labels = best_candidate.get("teacher_checklist_labels")
        scores = best_candidate.get("teacher_checklist_scores")
        tags = best_candidate.get("teacher_why_tags")
        cid = best_candidate.get("candidate_id", "na")
        return (
            labels if isinstance(labels, dict) else {},
            scores if isinstance(scores, dict) else {},
            tags,
            f"source: nearest labeled candidate {cid} | IoU to top={max(0.0, best_iou):.2f}",
        )
    return {}, {}, None, "source: no teacher detail available for this image/AR"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text) or "na"


def _render_row(row: dict[str, Any], output_path: Path, *, max_side: int) -> None:
    image_path = Path(str(row.get("image_path", "")))
    with Image.open(image_path) as img:
        image_canvas = img.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(image_canvas.size)))
    if scale < 1.0:
        image_canvas = image_canvas.resize((int(round(image_canvas.width * scale)), int(round(image_canvas.height * scale))), Image.BILINEAR)
    panel_width = max(640, min(820, int(round(image_canvas.width * 0.52))))
    panel_min_height = 2100
    canvas_height = max(image_canvas.height, panel_min_height)
    canvas = Image.new("RGB", (image_canvas.width + panel_width, canvas_height), (18, 18, 18))
    canvas.paste(image_canvas, (0, 0))
    draw = ImageDraw.Draw(canvas)
    image_size = image_canvas.size
    line_width = max(3, int(round(max(image_size) / 350)))
    font = _load_font(max(18, min(30, int(round(max(image_size) / 42)))))
    small_font = _load_font(max(15, min(23, int(round(max(image_size) / 55)))))
    panel_font = _load_font(max(18, min(26, int(round(panel_width / 28)))))
    panel_small = _load_font(max(15, min(21, int(round(panel_width / 36)))))

    best = row.get("best_positive") if isinstance(row.get("best_positive"), dict) else None
    if best is None and isinstance(row.get("positive_records"), list) and row["positive_records"]:
        best = max(row["positive_records"], key=lambda item: float(item.get("score", 0.0))) if all(isinstance(item, dict) for item in row["positive_records"]) else None
    top = row.get("top_candidate") if isinstance(row.get("top_candidate"), dict) else None
    if top is None and isinstance(row.get("selected"), dict):
        top = row["selected"]
    base = None
    for cand in row.get("candidates", []):
        if float(cand.get("is_base", 0.0)) > 0:
            base = cand
            break
    proposal = _proposal_for_visualization(row)
    pred_subject = _subject_box_payload(row, "predicted")
    teacher_subject = _subject_box_payload(row, "teacher")
    box_rows: list[tuple[str, tuple[int, int, int], dict[str, Any] | None]] = [
        ("model top crop", (196, 45, 45), top),
        ("best positive label", (36, 150, 80), best),
        ("baseline candidate", (45, 102, 210), base),
        ("proposal top-1 target AR", (224, 138, 34), proposal if isinstance(proposal, dict) else None),
        ("pred subject box", (196, 64, 210), pred_subject),
        ("teacher subject box", (16, 188, 180), teacher_subject),
    ]
    if isinstance(best, dict) and best.get("bbox_norm_xyxy"):
        _draw_box(draw, best["bbox_norm_xyxy"], image_size, (36, 150, 80), f"label {float(best.get('score', 0.0)):.2f}", small_font, line_width)
    if isinstance(base, dict) and base.get("bbox_norm_xyxy"):
        _draw_box(draw, base["bbox_norm_xyxy"], image_size, (45, 102, 210), "baseline", small_font, line_width)
    if isinstance(proposal, dict) and proposal.get("bbox_norm_xyxy"):
        _draw_box(draw, proposal["bbox_norm_xyxy"], image_size, (224, 138, 34), f"proposal {float(proposal.get('score', 0.0)):.2f}", small_font, line_width)
    if isinstance(teacher_subject, dict) and teacher_subject.get("bbox_norm_xyxy"):
        _draw_box(draw, teacher_subject["bbox_norm_xyxy"], image_size, (16, 188, 180), f"t-subj {_score_label(teacher_subject.get('confidence'))}", small_font, line_width)
    if isinstance(pred_subject, dict) and pred_subject.get("bbox_norm_xyxy"):
        _draw_box(draw, pred_subject["bbox_norm_xyxy"], image_size, (196, 64, 210), f"m-subj {_score_label(pred_subject.get('confidence'))}", small_font, line_width)
    if isinstance(top, dict) and top.get("bbox_norm_xyxy"):
        _draw_box(draw, top["bbox_norm_xyxy"], image_size, (196, 45, 45), f"v4 {float(top.get('model_utility', 0.0)):.2f}", small_font, line_width)

    metrics = row.get("metrics", {})
    title = "{image_id} | {ar} | hit={hit:.0f} ndcg5={ndcg:.2f} pIoU5={piou:.2f}".format(
        image_id=row.get("image_id", ""),
        ar=row.get("target_ar", "FREE"),
        hit=float(metrics.get("candidate_top1_hit", 0.0)),
        ndcg=float(metrics.get("ndcg_at_5", 0.0)),
        piou=float(metrics.get("proposal_best_iou_at_5", 0.0)),
    )
    pad = max(12, int(round(canvas.width / 90)))
    text_w = int(draw.textlength(title, font=font))
    text_h = int(getattr(font, "size", 20) * 1.45)
    draw.rectangle([0, 0, min(image_canvas.width, text_w + 2 * pad), text_h + pad], fill=(0, 0, 0))
    draw.text((pad, pad // 2), title, fill=(255, 255, 255), font=font)

    px = image_canvas.width
    panel_pad = max(18, panel_width // 28)
    draw.rectangle([px, 0, canvas.width, canvas_height], fill=(20, 22, 24))
    draw.line([px, 0, px, canvas.height], fill=(235, 235, 235), width=max(2, line_width // 2))
    y = panel_pad
    content_w = panel_width - 2 * panel_pad
    draw.text((px + panel_pad, y), "MobileCropNet v4", fill=(255, 255, 255), font=font)
    y += int(getattr(font, "size", 20) * 1.55)
    y = _draw_panel_line(
        draw,
        x=px + panel_pad,
        y=y,
        text=f"image: {row.get('image_id', '')} | target_ar: {row.get('target_ar', 'FREE')}",
        font=panel_small,
        fill=(215, 220, 225),
        max_width=content_w,
        line_gap=2,
    )
    decision = str(row.get("decision_label", f"decision_{row.get('decision_id', 'na')}"))
    subject_mode = str(row.get("subject_mode_label", f"mode_{row.get('route_id', 'na')}"))
    y += panel_pad // 2
    y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=f"decision: {decision}", font=panel_font, fill=(255, 236, 179), max_width=content_w, line_gap=2)
    y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=f"subject mode: {subject_mode}", font=panel_font, fill=(186, 228, 255), max_width=content_w, line_gap=2)
    y += panel_pad // 2
    draw.text((px + panel_pad, y), "bbox legend", fill=(255, 255, 255), font=panel_font)
    y += int(getattr(panel_font, "size", 18) * 1.45)
    legend_rows = [
        ((196, 45, 45), "red: model top crop"),
        ((36, 150, 80), "green: best/positive label crop"),
        ((45, 102, 210), "blue: baseline/full or policy baseline"),
        ((224, 138, 34), "orange: proposal top-1 filtered by target AR"),
        ((196, 64, 210), "magenta: model-predicted subject bbox"),
        ((16, 188, 180), "cyan: teacher subject bbox target"),
    ]
    for color, text in legend_rows:
        draw.rectangle([px + panel_pad, y + 3, px + panel_pad + 18, y + 21], fill=color)
        y = _draw_panel_line(draw, x=px + panel_pad + 26, y=y, text=text, font=panel_small, fill=(225, 225, 225), max_width=content_w - 26, line_gap=1)
    y += panel_pad // 3
    draw.text((px + panel_pad, y), "bbox details", fill=(255, 255, 255), font=panel_font)
    y += int(getattr(panel_font, "size", 18) * 1.45)
    for name, color, payload in box_rows:
        if not isinstance(payload, dict) or not payload.get("bbox_norm_xyxy"):
            continue
        box = [float(v) for v in payload.get("bbox_norm_xyxy", [])[:4]]
        score = payload.get("model_utility", payload.get("score", payload.get("objectness", payload.get("model_score", "na"))))
        draw.rectangle([px + panel_pad, y + 3, px + panel_pad + 18, y + 21], fill=color)
        ar_text = ""
        if name.startswith("proposal"):
            ar_text = f" crop_ar={_score_label(payload.get('crop_ar'))} ar_err={_score_label(payload.get('target_ar_log_error'), 3)}"
        if name.endswith("subject box"):
            subject_meta = row.get("subject_box") if isinstance(row.get("subject_box"), dict) else {}
            if name.startswith("pred") and isinstance(subject_meta, dict) and subject_meta.get("iou_to_teacher") is not None:
                ar_text = f" iouT={_score_label(subject_meta.get('iou_to_teacher'), 3)}"
        y = _draw_panel_line(
            draw,
            x=px + panel_pad + 26,
            y=y,
            text=f"{name}: score={_score_label(score)}{ar_text} box=[{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}]",
            font=panel_small,
            fill=(225, 225, 225),
            max_width=content_w - 26,
            line_gap=1,
        )
    if isinstance(top, dict):
        y += panel_pad // 2
        explanation = top.get("model_explanation") if isinstance(top.get("model_explanation"), dict) else {}
        utility_text = _label_with_score(explanation.get("utility"), default_label="utility")
        positive_text = _label_with_score(explanation.get("positive"), default_label="positive")
        risk_text = _label_with_score(explanation.get("risk"), default_label="risk")
        y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=f"top crop: {utility_text}", font=panel_font, fill=(255, 255, 255), max_width=content_w, line_gap=2)
        y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=f"positive: {positive_text}", font=panel_small, fill=(210, 255, 216), max_width=content_w, line_gap=2)
        y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=f"risk: {risk_text}", font=panel_small, fill=(255, 210, 210), max_width=content_w, line_gap=2)
        y = _draw_panel_line(
            draw,
            x=px + panel_pad,
            y=y,
            text=f"candidate: {top.get('candidate_id', 'na')} | source: {top.get('source', 'na')} | label: {top.get('label_type', 'na')}",
            font=panel_small,
            fill=(220, 220, 220),
            max_width=content_w,
            line_gap=1,
        )
        checklist = top.get("model_checklist") if isinstance(top.get("model_checklist"), dict) else explanation.get("checklist")
        if isinstance(checklist, dict):
            y += panel_pad // 2
            draw.text((px + panel_pad, y), "macro checklist", fill=(255, 255, 255), font=panel_font)
            y += int(getattr(panel_font, "size", 18) * 1.45)
            for key in ("aesthetic", "subject", "composition", "technical"):
                if key in checklist:
                    y = _draw_panel_line(
                        draw,
                        x=px + panel_pad,
                        y=y,
                        text=f"- {key}: {_label_with_score(checklist[key])}",
                        font=panel_small,
                        fill=(230, 230, 230),
                        max_width=content_w,
                        line_gap=1,
                    )
        detailed = top.get("model_detailed_checklist") if isinstance(top.get("model_detailed_checklist"), dict) else explanation.get("detailed_checklist")
        detail_scores = top.get("model_detail_scores") if isinstance(top.get("model_detail_scores"), dict) else explanation.get("detail_scores")
        why_tags = top.get("model_why_tags") if isinstance(top.get("model_why_tags"), list) else explanation.get("why_tags")
        if isinstance(detailed, dict):
            y += panel_pad // 2
            draw.text((px + panel_pad, y), "model detail", fill=(255, 255, 255), font=panel_font)
            y += int(getattr(panel_font, "size", 18) * 1.45)
            for text in (
                f"composition: thirds={_model_detail_item_text(row, detailed, 'third_dist')} phi={_model_detail_item_text(row, detailed, 'phi_dist')} center={_model_detail_item_text(row, detailed, 'center_dist')}",
                f"framing: headroom={_model_detail_item_text(row, detailed, 'headroom')} lookroom={_model_detail_item_text(row, detailed, 'lookroom')}",
                f"subject: coverage={_model_detail_item_text(row, detailed, 'subject_coverage')} scale={_model_detail_item_text(row, detailed, 'subject_scale')}",
                f"safety/context: face={_model_detail_item_text(row, detailed, 'face_cut')} joint={_model_detail_item_text(row, detailed, 'joint_cut')} context={_model_detail_item_text(row, detailed, 'context')}",
                f"composition scores: third={_score_item_text(detail_scores, 'third_strength')} phi={_score_item_text(detail_scores, 'phi_strength')} center={_score_item_text(detail_scores, 'center_strength')} sym={_score_item_text(detail_scores, 'symmetry_score')}",
                f"framing/risk scores: head={_model_score_item_text(row, detailed, detail_scores, 'headroom', 'headroom_ratio')} look={_model_score_item_text(row, detailed, detail_scores, 'lookroom', 'lookroom_ratio')} soft={_score_item_text(detail_scores, 'safety_penalty_soft')} hard={_score_item_text(detail_scores, 'safety_penalty_hard')}",
                f"why: {_teacher_tag_text(row, why_tags)}",
            ):
                y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=text, font=panel_small, fill=(230, 230, 230), max_width=content_w, line_gap=1)
        teacher_labels, teacher_scores, teacher_tags, teacher_source = _teacher_detail_payload(row, top)
        if isinstance(teacher_labels, dict):
            y += panel_pad // 2
            draw.text((px + panel_pad, y), "label detail", fill=(255, 255, 255), font=panel_font)
            y += int(getattr(panel_font, "size", 18) * 1.45)
            y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=teacher_source, font=panel_small, fill=(205, 220, 235), max_width=content_w, line_gap=1)
            for text in (
                f"composition: thirds={_teacher_label_text(row, teacher_labels, 'third_dist')} phi={_teacher_label_text(row, teacher_labels, 'phi_dist')} center={_teacher_label_text(row, teacher_labels, 'center_dist')}",
                f"framing: headroom={_teacher_label_text(row, teacher_labels, 'headroom')} lookroom={_teacher_label_text(row, teacher_labels, 'lookroom')}",
                f"subject/context: coverage={_teacher_label_text(row, teacher_labels, 'subject_coverage')} scale={_teacher_label_text(row, teacher_labels, 'subject_scale')} context={_teacher_label_text(row, teacher_labels, 'context')}",
                f"raw scores: head={_teacher_score_text(row, teacher_labels, teacher_scores, 'headroom', 'headroom_ratio')} look={_teacher_score_text(row, teacher_labels, teacher_scores, 'lookroom', 'lookroom_ratio')} third={_teacher_score_text(row, teacher_labels, teacher_scores, 'third_dist', 'third_dist')} phi={_teacher_score_text(row, teacher_labels, teacher_scores, 'phi_dist', 'phi_dist')} center={_teacher_score_text(row, teacher_labels, teacher_scores, 'center_dist', 'center_dist')}",
                f"label tags: {_teacher_tag_text(row, teacher_tags)}",
            ):
                y = _draw_panel_line(draw, x=px + panel_pad, y=y, text=text, font=panel_small, fill=(205, 220, 235), max_width=content_w, line_gap=1)
    canvas.save(output_path, format="PNG", compress_level=3)


def _make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
        thumb.thumbnail((420, 320), Image.BILINEAR)
        tile = Image.new("RGB", (420, 320), (18, 18, 18))
        tile.paste(thumb, ((420 - thumb.width) // 2, (320 - thumb.height) // 2))
        thumbs.append(tile)
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 320), (10, 10, 10))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 420, (idx // cols) * 320))
    sheet.save(output_path, format="PNG", compress_level=3)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.predictions_jsonl)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: min(args.sample_size, len(rows))]
    output_paths = []
    for idx, row in enumerate(selected, 1):
        safe_id = _safe_name(row.get("image_id", idx))
        safe_ar = _safe_name(row.get("target_ar", "FREE"))
        out = overlay_dir / f"{idx:04d}_{safe_id}_{safe_ar}.png"
        _render_row(row, out, max_side=args.max_side)
        output_paths.append(out)
    _make_contact_sheet(output_paths, args.output_dir / "contact_sheet.png")
    manifest = {"prediction_count": len(rows), "sample_count": len(output_paths), "overlay_dir": str(overlay_dir), "contact_sheet": str(args.output_dir / "contact_sheet.png")}
    (args.output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
