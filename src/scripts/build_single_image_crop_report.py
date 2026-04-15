#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


TOP_COLORS = ["#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5"]
TEACHER_COLORS = {
    "gaic": "#8e24aa",
    "cacnet": "#00897b",
    "cgs": "#6d4c41",
}
GRID_COLORS = {
    "thirds": "#ffd54f",
    "phi": "#ab47bc",
    "center": "#29b6f6",
}
TILE_W = 520
TILE_H = 1040


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a single-image crop comparison report.")
    p.add_argument("--run_tag", required=True)
    p.add_argument("--data_root", required=True, help="e.g. data/SSTK/Test_100")
    p.add_argument("--image_id", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ars", default="FREE,1:1,16:9")
    p.add_argument("--teacher_scores_jsonl", default="")
    p.add_argument("--candidates_jsonl", default="")
    p.add_argument("--features_jsonl", default="")
    p.add_argument("--teacher_row_json", default="")
    p.add_argument("--candidate_row_json", default="")
    p.add_argument("--feature_row_json", default="")
    return p.parse_args()


def read_jsonl_one(path: Path, image_id: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("image_id", "")) == image_id:
                return rec
    raise FileNotFoundError(f"image_id={image_id} not found in {path}")


def read_json_row_or_jsonl(row_json: str, jsonl_path: Path, image_id: str) -> Dict[str, Any]:
    if str(row_json).strip():
        path = Path(row_json)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return read_jsonl_one(jsonl_path, image_id)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_optional_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def reward_dist(d: Optional[float], sigma: float) -> float:
    if d is None:
        return 0.0
    ss = max(1e-6, float(sigma))
    return float(math.exp(-((float(d) / ss) ** 2)))


def fmt_opt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "na"
    return f"{safe_float(v):.{digits}f}"


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [safe_float(v) for v in box_a]
    bx1, by1, bx2, by2 = [safe_float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def denorm_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    return (
        max(0, min(width - 1, int(round(x1 * width)))),
        max(0, min(height - 1, int(round(y1 * height)))),
        max(1, min(width, int(round(x2 * width)))),
        max(1, min(height, int(round(y2 * height)))),
    )


def clip_box01(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_wh(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    return (max(1e-6, x2 - x1), max(1e-6, y2 - y1))


def project_box_to_ar(box: Sequence[float], target_ar: float, prefer_expand: bool = True) -> List[float]:
    x1, y1, x2, y2 = clip_box01(box)
    cx, cy = box_center([x1, y1, x2, y2])
    w, h = box_wh([x1, y1, x2, y2])
    ar = w / h
    ar_t = max(target_ar, 1e-6)

    if abs(ar - ar_t) <= 1e-6:
        return [x1, y1, x2, y2]

    if ar < ar_t:
        if prefer_expand:
            w2 = min(h * ar_t, 1.0)
            h2 = w2 / ar_t
        else:
            h2 = min(w / ar_t, 1.0)
            w2 = h2 * ar_t
    else:
        if prefer_expand:
            h2 = min(w / ar_t, 1.0)
            w2 = h2 * ar_t
        else:
            w2 = min(h * ar_t, 1.0)
            h2 = w2 / ar_t

    if w2 > 1.0:
        w2 = 1.0
        h2 = w2 / ar_t
    if h2 > 1.0:
        h2 = 1.0
        w2 = h2 * ar_t

    x1n = cx - 0.5 * w2
    x2n = cx + 0.5 * w2
    y1n = cy - 0.5 * h2
    y2n = cy + 0.5 * h2

    if x1n < 0.0:
        x2n -= x1n
        x1n = 0.0
    elif x2n > 1.0:
        x1n -= (x2n - 1.0)
        x2n = 1.0

    if y1n < 0.0:
        y2n -= y1n
        y1n = 0.0
    elif y2n > 1.0:
        y1n -= (y2n - 1.0)
        y2n = 1.0
    return clip_box01([x1n, y1n, x2n, y2n])


def parse_ar_text(ar: str) -> Optional[float]:
    text = str(ar).strip().upper()
    if not text or text == "FREE":
        return None
    if ":" not in text:
        return None
    a, b = text.split(":", 1)
    try:
        aa = float(a)
        bb = float(b)
        if aa <= 0 or bb <= 0:
            return None
        return aa / bb
    except Exception:
        return None


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(26)
FONT_BODY = load_font(16)
FONT_SMALL = load_font(13)
FONT_TINY = load_font(11)


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    raw = str(text or "")
    if max_width <= 0:
        return ""
    if draw.textlength(raw, font=font) <= max_width:
        return raw
    ellipsis = "..."
    lo, hi = 0, len(raw)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = raw[:mid].rstrip() + ellipsis
        if draw.textlength(candidate, font=font) <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def short_label(metric: str, value: str) -> str:
    mapping = {
        "rule_of_thirds_strong": "3rds+",
        "rule_of_thirds_weak": "3rds-",
        "center_comp_strong": "ctr+",
        "center_comp_weak": "ctr-",
        "teacher_aligned": "teach+",
        "teacher_mismatch": "teach-",
        "teacher_no_consensus": "teach?",
        "balanced_crop": "bal",
        "wide_crop": "wide",
        "tight_crop": "tight",
        "headroom_ok": "head ok",
        "headroom_loose": "head loose",
        "headroom_tight": "head tight",
        "lookroom_adequate": "look ok",
        "lookroom_insufficient": "look-",
        "lookroom_excessive": "look+",
        "no_face_cut": "face ok",
        "face_cut": "face cut",
        "no_joint_cut": "joint ok",
        "joint_cut_mild": "joint m",
        "joint_cut_severe": "joint s",
        "excellent": "cov ex",
        "good": "cov gd",
        "marginal": "cov mg",
        "poor": "cov pr",
    }
    return mapping.get(value, f"{metric}:{value}")


def badge_color(metric: str, value: str) -> str:
    if value in {"face_cut", "joint_cut_severe", "lookroom_insufficient", "headroom_tight", "poor"}:
        return "#c62828"
    if value in {"joint_cut_mild", "teacher_mismatch", "context_too_loose", "context_too_tight", "good"}:
        return "#ef6c00"
    if value in {"teacher_aligned", "balanced_crop", "wide_crop", "excellent", "no_face_cut", "no_joint_cut"}:
        return "#2e7d32"
    return "#546e7a"


def draw_badges(draw: ImageDraw.ImageDraw, x: int, y: int, labels: Sequence[Tuple[str, str]], *, max_width: Optional[int] = None) -> int:
    cx = x
    for metric, value in labels:
        if not value:
            continue
        text = short_label(metric, value)
        if max_width is not None:
            remaining = max_width - (cx - x)
            if remaining <= 30:
                break
            text = fit_text(draw, text, FONT_TINY, max(16, remaining - 18))
        w = int(draw.textlength(text, font=FONT_TINY)) + 12
        if max_width is not None and (cx - x + w) > max_width:
            break
        draw.rounded_rectangle((cx, y, cx + w, y + 18), radius=5, fill=badge_color(metric, value))
        draw.text((cx + 6, y + 3), text, font=FONT_TINY, fill="white")
        cx += w + 6
    return cx


def compute_teacher_agreement(box_norm: Sequence[float], teacher_refs: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for ref in teacher_refs:
        teacher_id = str(ref.get("teacher_id", ""))
        ref_box = ref.get("bbox_norm_xyxy")
        if teacher_id and isinstance(ref_box, list):
            out.append((teacher_id, iou_xyxy(box_norm, ref_box)))
    return out


def map_box_into_crop(inner_box_norm: Sequence[float], crop_box_norm: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    ix1, iy1, ix2, iy2 = [safe_float(v) for v in inner_box_norm]
    cx1, cy1, cx2, cy2 = [safe_float(v) for v in crop_box_norm]
    inter_x1 = max(ix1, cx1)
    inter_y1 = max(iy1, cy1)
    inter_x2 = min(ix2, cx2)
    inter_y2 = min(iy2, cy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None
    cw = max(1e-6, cx2 - cx1)
    ch = max(1e-6, cy2 - cy1)
    return (
        (inter_x1 - cx1) / cw,
        (inter_y1 - cy1) / ch,
        (inter_x2 - cx1) / cw,
        (inter_y2 - cy1) / ch,
    )


def map_point_into_crop(point_norm: Sequence[float], crop_box_norm: Sequence[float]) -> Optional[Tuple[float, float]]:
    px, py = [safe_float(v) for v in point_norm]
    cx1, cy1, cx2, cy2 = [safe_float(v) for v in crop_box_norm]
    if px < cx1 or px > cx2 or py < cy1 or py > cy2:
        return None
    cw = max(1e-6, cx2 - cx1)
    ch = max(1e-6, cy2 - cy1)
    return ((px - cx1) / cw, (py - cy1) / ch)


def _draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, radius: int = 3) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=1)


def draw_grid(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    thirds_x = [x1 + w / 3.0, x1 + 2.0 * w / 3.0]
    thirds_y = [y1 + h / 3.0, y1 + 2.0 * h / 3.0]
    phi_x = [x1 + w * 0.382, x1 + w * 0.618]
    phi_y = [y1 + h * 0.382, y1 + h * 0.618]
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    for xx in thirds_x:
        draw.line((xx, y1, xx, y2), fill=GRID_COLORS["thirds"], width=1)
    for yy in thirds_y:
        draw.line((x1, yy, x2, yy), fill=GRID_COLORS["thirds"], width=1)
    for xx in phi_x:
        draw.line((xx, y1, xx, y2), fill=GRID_COLORS["phi"], width=1)
    for yy in phi_y:
        draw.line((x1, yy, x2, yy), fill=GRID_COLORS["phi"], width=1)
    draw.line((cx, y1, cx, y2), fill=GRID_COLORS["center"], width=1)
    draw.line((x1, cy, x2, cy), fill=GRID_COLORS["center"], width=1)
    for xx in thirds_x:
        for yy in thirds_y:
            _draw_marker(draw, xx, yy, GRID_COLORS["thirds"], radius=3)
    for xx in phi_x:
        for yy in phi_y:
            _draw_marker(draw, xx, yy, GRID_COLORS["phi"], radius=2)
    _draw_marker(draw, cx, cy, GRID_COLORS["center"], radius=4)


def draw_grid_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = [("thirds", "3rds"), ("phi", "phi"), ("center", "center")]
    cx = x
    for key, label in items:
        color = GRID_COLORS[key]
        draw.line((cx, y + 7, cx + 14, y + 7), fill=color, width=2)
        _draw_marker(draw, cx + 7, y + 7, color, radius=2)
        draw.text((cx + 20, y), label, font=FONT_TINY, fill=color)
        cx += 54


def short_metric_name(label: str) -> str:
    mapping = {
        "aesthetic": "aesth",
        "align": "align",
        "cov": "cov",
        "scale": "scale",
        "border": "border",
        "softcut_quality": "softcut",
        "comp": "comp",
        "headroom": "head",
        "lookroom": "look",
        "horizon_y": "horizon",
        "sym": "sym",
        "context": "context",
        "copyspace": "copy",
        "teacher": "teacher",
        "rank": "rank",
        "policy": "policy",
        "area": "area",
    }
    return mapping.get(label, label)


def summarize_provenance(item: Dict[str, Any]) -> str:
    teacher_prov = item.get("teacher_provenance")
    if isinstance(teacher_prov, list) and teacher_prov:
        parts = []
        for entry in teacher_prov[:2]:
            if not isinstance(entry, dict):
                continue
            tid = str(entry.get("teacher_id", "")).strip()
            stage = str(entry.get("teacher_stage", "")).strip()
            if tid:
                parts.append(f"{tid}:{stage}" if stage else tid)
        if parts:
            return "t=" + ",".join(parts)
    lineage = item.get("source_lineage")
    if isinstance(lineage, list) and lineage:
        return " + ".join(str(x) for x in lineage[:3])
    source_types = item.get("source_types")
    if isinstance(source_types, list) and source_types:
        return " + ".join(str(x) for x in source_types[:3])
    source = str(item.get("source", "")).strip()
    if source:
        return source
    return "na"


def verdict_color(value: str) -> str:
    if value in {"face_cut", "joint_cut_severe", "lookroom_insufficient", "headroom_tight", "teacher_mismatch", "poor"}:
        return "#c62828"
    if value in {"joint_cut_mild", "rule_of_thirds_weak", "center_comp_weak", "marginal", "good", "context_too_tight", "context_too_loose"}:
        return "#ef6c00"
    if value in {
        "no_face_cut",
        "no_joint_cut",
        "teacher_aligned",
        "rule_of_thirds_strong",
        "center_comp_strong",
        "excellent",
        "balanced_crop",
        "wide_crop",
        "ideal_scale",
        "headroom_ok",
        "lookroom_adequate",
    }:
        return "#2e7d32"
    return "#78909c"


def draw_verdict_strip(draw: ImageDraw.ImageDraw, x: int, y: int, labels: Dict[str, Any]) -> None:
    keys = [
        ("face_cut", "face"),
        ("joint_cut", "joint"),
        ("headroom", "head"),
        ("lookroom", "look"),
        ("teacher_consensus", "teach"),
        ("crop_tightness", "tight"),
    ]
    draw.text((x, y), "label strip", font=FONT_TINY, fill="#555555")
    cx = x
    yy = y + 14
    for key, short in keys:
        value = str(labels.get(key, "") or "")
        color = verdict_color(value)
        draw.rounded_rectangle((cx, yy, cx + 36, yy + 16), radius=4, fill=color)
        draw.text((cx + 4, yy + 3), short, font=FONT_TINY, fill="white")
        cx += 42


def draw_score_summary(draw: ImageDraw.ImageDraw, x: int, y: int, scores: Dict[str, Any]) -> None:
    draw.text((x, y), "score summary", font=FONT_TINY, fill="#555555")
    rows = [
        ("rank", safe_float(scores.get("rank", scores.get("final", 0.0)), 0.0)),
        ("policy", safe_float(scores.get("policy", scores.get("final", 0.0)), 0.0)),
        ("area", safe_float(scores.get("area_log_prior", 0.0), 0.0)),
    ]
    for idx, (label, val) in enumerate(rows):
        yy = y + 18 + idx * 16
        draw.text((x, yy), label, font=FONT_TINY, fill="#555555")
        draw.text((x + 48, yy), f"{val:.4f}", font=FONT_TINY, fill="#263238")


def draw_signed_mini_bars(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    rows: Sequence[Tuple[str, Optional[float], str]],
    *,
    bar_w: int = 104,
    zero_center: bool = True,
) -> None:
    draw.text((x, y), title, font=FONT_TINY, fill="#555555")
    for idx, (label, val, suffix) in enumerate(rows):
        yy = y + 16 + idx * 16
        draw.text((x, yy), label, font=FONT_TINY, fill="#555555")
        bx1 = x + 44
        by1 = yy + 2
        bx2 = bx1 + bar_w
        by2 = yy + 10
        draw.rectangle((bx1, by1, bx2, by2), outline="#bbbbbb", width=1)
        if zero_center:
            mid = bx1 + bar_w // 2
            draw.line((mid, by1, mid, by2), fill="#cfd8dc", width=1)
        if val is None:
            draw.text((bx2 + 6, yy), "na", font=FONT_TINY, fill="#777777")
            continue
        vf = clamp(safe_float(val, 0.0), -1.0, 1.0)
        if zero_center:
            mid = bx1 + bar_w // 2
            half = bar_w // 2
            fill = int(round(abs(vf) * half))
            if vf >= 0:
                draw.rectangle((mid, by1, mid + fill, by2), fill="#2e7d32")
            else:
                draw.rectangle((mid - fill, by1, mid, by2), fill="#c62828")
        else:
            fill = int(round(clamp((vf + 1.0) / 2.0, 0.0, 1.0) * bar_w))
            color = "#00897b" if vf >= 0 else "#c62828"
            draw.rectangle((bx1, by1, bx1 + fill, by2), fill=color)
        draw.text((bx2 + 6, yy), f"{vf:.2f}{suffix}", font=FONT_TINY, fill="#555555")


def draw_metric_rows(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    rows: Sequence[Tuple[str, Optional[float], str]],
    *,
    bar_w: int = 80,
    bar_color: str = "#1e88e5",
    label_w: int = 48,
    note_w: int = 58,
    row_h: int = 18,
) -> None:
    draw.text((x, y), title, font=FONT_TINY, fill="#555555")
    for idx, (label, val, note) in enumerate(rows):
        yy = y + 16 + idx * row_h
        draw.text((x, yy), fit_text(draw, label, FONT_TINY, label_w - 2), font=FONT_TINY, fill="#555555")
        bx1 = x + label_w
        bx2 = bx1 + bar_w
        draw.rectangle((bx1, yy + 2, bx2, yy + 10), outline="#bbbbbb", width=1)
        if val is None:
            draw.text((bx2 + 6, yy), fit_text(draw, note or "na", FONT_TINY, note_w), font=FONT_TINY, fill="#777777")
            continue
        vf = clamp(safe_float(val, 0.0), 0.0, 1.0)
        fill = int(round(vf * bar_w))
        draw.rectangle((bx1, yy + 2, bx1 + fill, yy + 10), fill=bar_color)
        draw.text((bx2 + 6, yy), fit_text(draw, note or f"{vf:.2f}", FONT_TINY, note_w), font=FONT_TINY, fill="#555555")


def draw_margin_ruler(draw: ImageDraw.ImageDraw, x: int, y: int, margins: Optional[Dict[str, float]]) -> None:
    draw.text((x, y), "margin ruler", font=FONT_TINY, fill="#555555")
    if not margins:
        draw.text((x + 72, y), "na", font=FONT_TINY, fill="#777777")
        return
    for idx, key in enumerate(["L", "R", "T", "B"]):
        val = clamp(safe_float(margins.get(key, 0.0), 0.0), 0.0, 1.0)
        bar_x = x + idx * 118
        bar_y = y + 16
        draw.text((bar_x, bar_y), key, font=FONT_TINY, fill="#555555")
        draw.rectangle((bar_x + 12, bar_y + 3, bar_x + 82, bar_y + 11), outline="#bbbbbb", width=1)
        draw.rectangle((bar_x + 12, bar_y + 3, bar_x + 12 + int(70 * val), bar_y + 11), fill="#00bcd4")
        draw.text((bar_x + 86, bar_y), f"{val*100:.0f}%", font=FONT_TINY, fill="#555555")


def effective_comp_weights_for_report(
    cfg: Dict[str, Any],
    route_global: Dict[str, Any],
    symmetry_score: float,
    horizon_active: bool,
) -> Dict[str, float]:
    flags = route_global.get("flags", {}) if isinstance(route_global.get("flags"), dict) else {}
    shot_type = str(route_global.get("shot_type", "")).strip().lower()
    portrait_category = str(route_global.get("portrait_category", "")).strip().lower()

    w_third = safe_float(cfg.get("w_third", 0.35), 0.35)
    w_phi = safe_float(cfg.get("w_phi", 0.15), 0.15)
    w_center = safe_float(cfg.get("w_center", 0.35), 0.35)
    w_hor = safe_float(cfg.get("w_horizon", 0.15), 0.15) if horizon_active else 0.0

    if portrait_category in {"formal_id", "corporate"}:
        w_center += 0.20
        w_third -= 0.10
        w_phi -= 0.05
    if portrait_category == "lifestyle":
        w_third += 0.10
        w_phi += 0.05
        w_center -= 0.05
    w_center += 0.30 * clamp(symmetry_score, 0.0, 1.0)
    if bool(flags.get("is_landscape_scene", False)):
        w_hor += 0.20
    if shot_type == "group":
        w_center += 0.05
    vals = {
        "third": max(0.0, w_third),
        "phi": max(0.0, w_phi),
        "center": max(0.0, w_center),
        "horizon": max(0.0, w_hor),
    }
    denom = sum(vals.values())
    if denom <= 1e-8:
        return {"third": 0.5, "phi": 0.0, "center": 0.5, "horizon": 0.0}
    return {k: v / denom for k, v in vals.items()}


def effective_lambdas_for_report(cfg: Dict[str, Any], route_global: Dict[str, Any]) -> Dict[str, float]:
    flags = dict(route_global.get("flags", {}) if isinstance(route_global.get("flags"), dict) else {})
    shot_type = str(route_global.get("shot_type", "")).strip().lower()
    policy_id = str(route_global.get("policy_id", "")).strip().lower()
    subject_mode = str(route_global.get("subject_mode", "")).strip().lower()
    has_people = int(route_global.get("num_people", 0) or 0) > 0

    lam = {
        "cov": safe_float(cfg.get("lambda_cov", 0.60), 0.60),
        "cut": safe_float(cfg.get("lambda_cut", 1.00), 1.00),
        "text": safe_float(cfg.get("lambda_text", 0.15), 0.15),
        "comp": safe_float(cfg.get("lambda_comp", 0.35), 0.35),
        "hr": safe_float(cfg.get("lambda_hr", 0.15), 0.15),
        "lr": safe_float(cfg.get("lambda_lr", 0.15), 0.15),
        "sym": safe_float(cfg.get("lambda_sym", 0.10), 0.10),
        "ctx": safe_float(cfg.get("lambda_ctx", 0.10), 0.10),
        "cs": safe_float(cfg.get("lambda_cs", 0.10), 0.10),
        "teach": safe_float(cfg.get("lambda_teach", 0.10), 0.10),
        "ar_free": safe_float(cfg.get("lambda_ar_free", 0.10), 0.10),
    }

    if not has_people:
        lam["hr"] = 0.0
        lam["lr"] = 0.0
    if bool(flags.get("has_copy_space", False)):
        lam["cs"] += 0.20
        lam["ctx"] += 0.10
    if bool(flags.get("is_landscape_scene", False)):
        lam["comp"] += 0.15
        lam["sym"] += 0.05
        lam["hr"] = 0.0
        lam["lr"] = 0.0
    if shot_type == "group":
        lam["lr"] = min(lam["lr"], 0.10)
    if bool(flags.get("is_product", False)) or bool(flags.get("is_isolated_packshot", False)):
        lam["ctx"] -= 0.10
        lam["comp"] -= 0.10
        lam["cut"] += 0.15

    if subject_mode in {"portrait_single", "portrait_group"} or policy_id in {"portrait_single_v1", "portrait_group_v1"}:
        if subject_mode == "portrait_group":
            lam["lr"] = min(lam["lr"], 0.12)
            lam["cut"] = max(lam["cut"], 1.10)
        else:
            lam["hr"] = max(lam["hr"], 0.25)
            lam["lr"] = max(lam["lr"], 0.20)
            lam["cut"] = max(lam["cut"], 1.05)
    elif subject_mode in {"object_single", "object_multi"} or policy_id in {"object_v1", "object_multi_v1"}:
        lam["cut"] = max(lam["cut"], 1.05)
        lam["ctx"] = max(lam["ctx"], 0.18)
        if subject_mode == "object_multi":
            lam["cov"] = max(lam["cov"], 0.55)
    elif subject_mode in {"scene_general", "scene_landscape"} or policy_id == "scene_v1":
        lam["cov"] = min(lam["cov"], 0.30)
        lam["comp"] = max(lam["comp"], 0.55)
        lam["ctx"] = max(lam["ctx"], 0.35)
        lam["hr"] = 0.0
        lam["lr"] = 0.0
    elif subject_mode == "background_texture_copyspace" or policy_id == "copyspace_v1":
        flags["has_copy_space"] = True
        lam["cs"] = max(lam["cs"], 0.35)
        lam["ctx"] = max(lam["ctx"], 0.30)
        lam["hr"] = 0.0
        lam["lr"] = 0.0
    elif subject_mode == "text_document" or policy_id == "text_v1":
        lam["text"] = max(lam["text"], 0.40)
        lam["cut"] = max(lam["cut"], 1.00)
        lam["hr"] = 0.0
        lam["lr"] = 0.0

    for k in list(lam.keys()):
        lam[k] = max(0.0, float(lam[k]))
    return lam


def area_weight_for_report(route_global: Dict[str, Any]) -> float:
    flags = route_global.get("flags", {}) if isinstance(route_global.get("flags"), dict) else {}
    subject_mode = str(route_global.get("subject_mode", "")).strip().lower()
    policy_id = str(route_global.get("policy_id", "")).strip().lower()
    w = 0.12 if (bool(flags.get("has_copy_space", False)) or bool(flags.get("is_landscape_scene", False))) else (0.06 if bool(flags.get("is_product", False)) else 0.10)
    if subject_mode in {"portrait_single", "portrait_group"} or policy_id in {"portrait_single_v1", "portrait_group_v1"}:
        w = max(w, 0.10)
    elif subject_mode in {"object_single", "object_multi"} or policy_id in {"object_v1", "object_multi_v1"}:
        w = max(w, 0.08)
    elif subject_mode in {"scene_general", "scene_landscape", "background_texture_copyspace"} or policy_id in {"scene_v1", "copyspace_v1"}:
        w = min(w, 0.08)
    return w


def extract_score_details(
    item: Dict[str, Any],
    route_global: Dict[str, Any],
    scorer_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    macro_scores = item.get("macro_scores", {}) if isinstance(item.get("macro_scores"), dict) else {}
    macro_components = item.get("macro_components", {}) if isinstance(item.get("macro_components"), dict) else {}
    macro_masks = item.get("macro_masks", {}) if isinstance(item.get("macro_masks"), dict) else {}
    subject_mode = str(item.get("subject_mode") or route_global.get("subject_mode", "")).strip().lower()

    a_weights = {
        "A_aesthetic": max(0.0, safe_float(scorer_cfg.get("a_macro_aesthetic_weight", 0.75), 0.75)),
        "A_align": max(0.0, safe_float(scorer_cfg.get("a_macro_align_weight", 0.25), 0.25)),
    }
    if subject_mode.startswith("scene_"):
        s_weights = {"S_cov": 0.45, "S_scale": 0.20, "S_border": 0.15, "S_softcut_quality": 0.20}
    elif subject_mode.startswith("portrait"):
        s_weights = {"S_cov": 0.40, "S_scale": 0.25, "S_border": 0.20, "S_softcut_quality": 0.15}
    else:
        s_weights = {"S_cov": 0.35, "S_scale": 0.25, "S_border": 0.20, "S_softcut_quality": 0.20}
    if subject_mode.startswith("portrait"):
        c_weights = {"C_comp": 0.20, "C_headroom": 0.30, "C_lookroom": 0.30, "C_sym": 0.10, "C_context": 0.10}
    elif subject_mode.startswith("scene_"):
        c_weights = {"C_comp": 0.20, "C_horizon_y": 0.30, "C_sym": 0.10, "C_context": 0.25, "C_copyspace": 0.0}
    elif subject_mode == "background_copyspace":
        c_weights = {"C_comp": 0.15, "C_context": 0.25, "C_copyspace": 0.45, "C_sym": 0.15}
    else:
        c_weights = {"C_comp": 0.35, "C_sym": 0.20, "C_context": 0.25, "C_headroom": 0.10, "C_lookroom": 0.10}
    t_weights = {"T_teacher": 1.0}
    rank_weights = {
        "A_macro": max(0.0, safe_float(scorer_cfg.get("rank_weight_a", 1.0), 1.0)),
        "S_macro": max(0.0, safe_float(scorer_cfg.get("rank_weight_s", 1.0), 1.0)),
        "C_macro": max(0.0, safe_float(scorer_cfg.get("rank_weight_c", 1.0), 1.0)),
        "T_macro": max(0.0, safe_float(scorer_cfg.get("rank_weight_t", 0.10), 0.10)),
    }

    def build_group_rows(keys: Sequence[str], weights: Dict[str, float]) -> List[Tuple[str, Optional[float], str]]:
        rows: List[Tuple[str, Optional[float], str]] = []
        for key in keys:
            val = macro_components.get(key)
            weight = max(0.0, float(weights.get(key, 0.0)))
            label = short_metric_name(key.split("_", 1)[1] if "_" in key else key)
            note = "na" if val is None else f"{safe_float(val, 0.0):.2f}"
            rows.append((label, val, note))
        return rows

    rank_rows: List[Tuple[str, Optional[float], str]] = []
    for key, mask_key in [("A_macro", "A_active"), ("S_macro", "S_active"), ("C_macro", "C_active"), ("T_macro", "T_active")]:
        val = macro_scores.get(key)
        if val is None or int(macro_masks.get(mask_key, 1)) <= 0:
            rank_rows.append((key[0], None, "na"))
            continue
        rank_rows.append((key[0], val, f"{safe_float(val, 0.0):.2f}"))

    policy_rows = [
        ("rank", safe_float(item.get("scores", {}).get("rank", item.get("scores", {}).get("final", 0.0)), 0.0), f"{safe_float(item.get('scores', {}).get('rank', item.get('scores', {}).get('final', 0.0)), 0.0):.2f}"),
        ("area", clamp(0.5 + 0.5 * safe_float(item.get("scores", {}).get("area_log_prior", 0.0), 0.0), 0.0, 1.0), f"{safe_float(item.get('scores', {}).get('area_log_prior', 0.0), 0.0):.3f}"),
        ("policy", safe_float(item.get("scores", {}).get("policy", item.get("scores", {}).get("final", 0.0)), 0.0), f"{safe_float(item.get('scores', {}).get('policy', item.get('scores', {}).get('final', 0.0)), 0.0):.2f}"),
    ]
    return {
        "rank_rows": rank_rows,
        "group_rows": {
            "A": build_group_rows(["A_aesthetic", "A_align"], a_weights),
            "S": build_group_rows(["S_cov", "S_scale", "S_border", "S_softcut_quality"], s_weights),
            "C": build_group_rows(["C_comp", "C_headroom", "C_lookroom", "C_horizon_y", "C_sym", "C_context", "C_copyspace"], c_weights),
            "T": build_group_rows(["T_teacher"], t_weights),
            "policy": policy_rows,
        },
        "rank_weights": rank_weights,
        "group_weights": {
            "A": a_weights,
            "S": s_weights,
            "C": c_weights,
            "T": t_weights,
        },
    }


def crop_to_tile(
    image: Image.Image,
    bbox_px: Tuple[int, int, int, int],
    title: str,
    subtitle: str,
    *,
    bbox_norm: Optional[Sequence[float]] = None,
    subject_box_norm: Optional[Sequence[float]] = None,
    face_center_norm: Optional[Sequence[float]] = None,
    teacher_agreement: Optional[Sequence[Tuple[str, float]]] = None,
    badge_specs: Optional[Sequence[Tuple[str, str]]] = None,
    scores: Optional[Dict[str, Any]] = None,
    components: Optional[Dict[str, Any]] = None,
    macro_scores: Optional[Dict[str, Any]] = None,
    score_details: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None,
    provenance_text: str = "",
) -> Image.Image:
    x1, y1, x2, y2 = bbox_px
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    canvas_w, canvas_h = TILE_W, TILE_H
    header_h = 88
    footer_h = 610
    body_h = canvas_h - header_h - footer_h
    tile = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((16, 10), fit_text(draw, title, FONT_BODY, canvas_w - 32), font=FONT_BODY, fill="black")
    draw.text((16, 30), fit_text(draw, subtitle, FONT_SMALL, canvas_w - 32), font=FONT_SMALL, fill="#333333")
    if badge_specs:
        draw_badges(draw, 16, 50, badge_specs[:6], max_width=canvas_w - 32)
    if provenance_text:
        prov = fit_text(draw, provenance_text, FONT_TINY, 160)
        w = int(draw.textlength(prov, font=FONT_TINY)) + 12
        draw.rounded_rectangle((canvas_w - w - 16, 12, canvas_w - 16, 30), radius=5, fill="#eceff1")
        draw.text((canvas_w - w - 10, 16), prov, font=FONT_TINY, fill="#37474f")
    if crop.width <= 0 or crop.height <= 0:
        return tile
    scale = min((canvas_w - 24) / crop.width, (body_h - 24) / crop.height)
    resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
    ox = (canvas_w - resized.width) // 2
    oy = header_h + (body_h - resized.height) // 2
    tile.paste(resized, (ox, oy))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((ox, oy, ox + resized.width, oy + resized.height), outline="#222222", width=2)
    draw_grid(draw, (ox, oy, ox + resized.width, oy + resized.height))
    margin_values: Optional[Dict[str, float]] = None
    if bbox_norm and subject_box_norm:
        rel = map_box_into_crop(subject_box_norm, bbox_norm)
        if rel is not None:
            sx1 = ox + rel[0] * resized.width
            sy1 = oy + rel[1] * resized.height
            sx2 = ox + rel[2] * resized.width
            sy2 = oy + rel[3] * resized.height
            draw.rectangle((sx1, sy1, sx2, sy2), outline="#00bcd4", width=2)
            crop_w = max(1e-6, safe_float(bbox_norm[2]) - safe_float(bbox_norm[0]))
            crop_h = max(1e-6, safe_float(bbox_norm[3]) - safe_float(bbox_norm[1]))
            sub_w = max(0.0, rel[2] - rel[0])
            sub_h = max(0.0, rel[3] - rel[1])
            margins = {
                "L": max(0.0, rel[0]),
                "R": max(0.0, 1.0 - rel[2]),
                "T": max(0.0, rel[1]),
                "B": max(0.0, 1.0 - rel[3]),
            }
            margin_values = margins
    if bbox_norm and face_center_norm:
        rel_pt = map_point_into_crop(face_center_norm, bbox_norm)
        if rel_pt is not None:
            fx = ox + rel_pt[0] * resized.width
            fy = oy + rel_pt[1] * resized.height
            draw.ellipse((fx - 4, fy - 4, fx + 4, fy + 4), fill="#ec407a", outline="white", width=1)
    footer_y = canvas_h - footer_h + 8
    if scores:
        draw_score_summary(draw, 16, footer_y + 4, scores)
    if teacher_agreement:
        base_x = 300
        base_y = footer_y + 4
        draw.text((base_x, base_y), "teacher agreement", font=FONT_TINY, fill="#555555")
        for idx, (teacher_id, iou) in enumerate(teacher_agreement):
            yy = base_y + 18 + idx * 18
            color = TEACHER_COLORS.get(teacher_id, "#666666")
            draw.text((base_x, yy), fit_text(draw, teacher_id, FONT_TINY, 42), font=FONT_TINY, fill=color)
            draw.rectangle((base_x + 46, yy + 3, base_x + 116, yy + 11), outline="#bbbbbb", width=1)
            draw.rectangle((base_x + 46, yy + 3, base_x + 46 + int(70 * min(1.0, iou)), yy + 11), fill=color)
            draw.text((base_x + 120, yy), f"{iou:.2f}", font=FONT_TINY, fill="#555555")
    if score_details:
        draw_metric_rows(draw, 128, footer_y + 4, "rank fuse A/S/C/T", score_details.get("rank_rows", []), bar_w=74, bar_color="#2e7d32", label_w=22, note_w=34, row_h=18)
        draw_metric_rows(draw, 16, footer_y + 104, "A detail", score_details.get("group_rows", {}).get("A", []), bar_w=70, bar_color="#5e35b1", label_w=44, note_w=34, row_h=18)
        draw_metric_rows(draw, 264, footer_y + 104, "S detail", score_details.get("group_rows", {}).get("S", []), bar_w=70, bar_color="#1e88e5", label_w=44, note_w=34, row_h=18)
        draw_metric_rows(draw, 16, footer_y + 232, "C detail", score_details.get("group_rows", {}).get("C", []), bar_w=80, bar_color="#ef6c00", label_w=52, note_w=34, row_h=18)
        draw_metric_rows(draw, 264, footer_y + 232, "T/policy", score_details.get("group_rows", {}).get("T", []) + score_details.get("group_rows", {}).get("policy", []), bar_w=70, bar_color="#00897b", label_w=44, note_w=34, row_h=18)
    if labels:
        draw_verdict_strip(draw, 138, footer_y + 388, labels)
    draw_margin_ruler(draw, 16, footer_y + 438, margin_values)
    draw_grid_legend(draw, 16, canvas_h - 22)
    draw.text((190, canvas_h - 20), "subject/face/teacher overlays + macro A/S/C/T detail", font=FONT_TINY, fill="#555555")
    return tile


def build_overlay(
    image: Image.Image,
    title: str,
    topk: Sequence[Dict[str, Any]],
    teacher_refs: Sequence[Dict[str, Any]],
    subject_box_norm: Optional[Sequence[float]],
    face_center_norm: Optional[Sequence[float]],
    out_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for idx, cand in enumerate(topk, start=1):
        color = TOP_COLORS[idx - 1]
        box = denorm_box(cand["bbox_norm_xyxy"], width, height)
        draw.rectangle(box, outline=color, width=4)
        label = f"T{idx}"
        tx, ty = box[0] + 6, max(0, box[1] - 24)
        draw.rectangle((tx - 4, ty - 2, tx + 44, ty + 18), fill=color)
        draw.text((tx, ty), label, font=FONT_BODY, fill="black")
    for ref in teacher_refs:
        color = TEACHER_COLORS.get(ref["teacher_id"], "#000000")
        box = denorm_box(ref["bbox_norm_xyxy"], width, height)
        draw.rectangle(box, outline=color, width=3)
        label = f"{ref['teacher_id']}:{ref['stage']}"
        tx, ty = max(0, box[0] + 6), min(height - 22, box[3] + 4)
        draw.rectangle((tx - 4, ty - 2, min(width, tx + 120), min(height, ty + 18)), fill=color)
        draw.text((tx, ty), label, font=FONT_SMALL, fill="white")
    if subject_box_norm:
        sbox = denorm_box(subject_box_norm, width, height)
        draw.rectangle(sbox, outline="#00bcd4", width=3)
    if face_center_norm:
        fx = int(round(safe_float(face_center_norm[0]) * width))
        fy = int(round(safe_float(face_center_norm[1]) * height))
        draw.ellipse((fx - 5, fy - 5, fx + 5, fy + 5), fill="#ec407a", outline="white", width=1)
    pad = 18
    top_band = Image.new("RGB", (canvas.width, 48), "white")
    band_draw = ImageDraw.Draw(top_band)
    band_draw.text((pad, 10), fit_text(band_draw, title, FONT_TITLE, canvas.width - pad * 2), font=FONT_TITLE, fill="black")
    merged = Image.new("RGB", (canvas.width, canvas.height + top_band.height), "white")
    merged.paste(top_band, (0, 0))
    merged.paste(canvas, (0, top_band.height))
    ensure_dir(out_path.parent)
    merged.save(out_path)


def build_panel(
    image: Image.Image,
    items: Sequence[Dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    cols = 2
    tile_w, tile_h = TILE_W, TILE_H
    gap = 20
    rows = math.ceil(len(items) / cols)
    panel_w = cols * tile_w + (cols + 1) * gap
    panel_h = rows * tile_h + (rows + 1) * gap + 60
    panel = Image.new("RGB", (panel_w, panel_h), "#f4f4f4")
    draw = ImageDraw.Draw(panel)
    draw.text((gap, 16), fit_text(draw, title, FONT_TITLE, panel_w - gap * 2), font=FONT_TITLE, fill="black")
    for idx, item in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = 60 + gap + row * (tile_h + gap)
        tile = crop_to_tile(
            image,
            item["bbox_px"],
            item["title"],
            item["subtitle"],
            bbox_norm=item.get("bbox_norm"),
            subject_box_norm=item.get("subject_box_norm"),
            face_center_norm=item.get("face_center_norm"),
            teacher_agreement=item.get("teacher_agreement"),
            scores=item.get("scores"),
            components=item.get("components"),
            macro_scores=item.get("macro_scores"),
            score_details=item.get("score_details"),
            labels=item.get("labels"),
            provenance_text=item.get("provenance_text", ""),
            badge_specs=[
                ("third_dist", str(item.get("labels", {}).get("third_dist", ""))),
                ("center_dist", str(item.get("labels", {}).get("center_dist", ""))),
                ("teacher_consensus", str(item.get("labels", {}).get("teacher_consensus", ""))),
                ("headroom", str(item.get("labels", {}).get("headroom", ""))),
                ("lookroom", str(item.get("labels", {}).get("lookroom", ""))),
                ("crop_tightness", str(item.get("labels", {}).get("crop_tightness", ""))),
            ],
        )
        panel.paste(tile, (x, y))
    ensure_dir(out_path.parent)
    panel.save(out_path)


def stringify_labels(labels: Dict[str, Any], keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        parts.append(f"{key}={labels.get(key, '')}")
    return ", ".join(parts)


def collect_scored_candidates(ar_res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in ["selected_topk", "hard_negatives", "also_considered_rejected"]:
        for cand in ar_res.get(key, []) or []:
            cid = str(cand.get("candidate_id", ""))
            if cid and cid not in out:
                out[cid] = cand
    return out


@lru_cache(maxsize=256)
def load_teacher_proposal_record(proposals_jsonl: str, image_id: str) -> Dict[str, Any]:
    path = Path(proposals_jsonl)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("image_id", "")) == image_id:
                return rec
    return {}


def proposal_candidate_sources(data_root: Path, run_tag: str) -> List[Tuple[str, Path]]:
    proposal_dir = data_root / "artifacts" / "public_teachers" / "proposals"
    current = proposal_dir / f"teacher_proposals_public_{run_tag}.jsonl"
    others = sorted(
        [p for p in proposal_dir.glob("teacher_proposals_public_*.jsonl") if p != current],
        key=lambda p: p.stem,
        reverse=True,
    )
    ordered: List[Tuple[str, Path]] = []
    if current.exists():
        ordered.append((run_tag, current))
    for path in others:
        tag = path.stem.replace("teacher_proposals_public_", "", 1)
        ordered.append((tag, path))
    return ordered


def best_teacher_refs(
    candidate_row: Dict[str, Any],
    ar: str,
    *,
    data_root: Path,
    run_tag: str,
    image_id: str,
    image_ar: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    teacher_candidates = candidate_row.get("teacher_candidates", []) or []
    for teacher_id in ["gaic", "cacnet", "cgs"]:
        matches = [x for x in teacher_candidates if x.get("teacher_id") == teacher_id and x.get("target_ar") == ar]
        if ar == "FREE":
            chosen = next((x for x in matches if x.get("stage") == "seed"), None)
        else:
            chosen = next((x for x in matches if x.get("stage") == "proj"), None)
            if chosen is None:
                chosen = next((x for x in matches if x.get("stage") == "seed"), None)
        if chosen is not None:
            out.append(
                {
                    "teacher_id": teacher_id,
                    "stage": str(chosen.get("stage", "")),
                    "bbox_norm_xyxy": chosen.get("bbox_norm_xyxy"),
                    "teacher_score": safe_float(chosen.get("teacher_score", 0.0)),
                    "ref_source": "candidate_row",
                    "ref_run_tag": run_tag,
                }
            )
            seen.add(teacher_id)

    if len(seen) == 3:
        return out

    target_ar = parse_ar_text(ar)
    for src_tag, path in proposal_candidate_sources(data_root, run_tag):
        record = load_teacher_proposal_record(str(path), image_id)
        payload = record.get("teacher_proposals", {}) if isinstance(record.get("teacher_proposals"), dict) else {}
        for teacher_id in ["gaic", "cacnet", "cgs"]:
            if teacher_id in seen:
                continue
            teacher_payload = payload.get(teacher_id, {})
            if not isinstance(teacher_payload, dict):
                continue
            free_form = teacher_payload.get("free_form", [])
            if not isinstance(free_form, list) or not free_form:
                continue
            seed = free_form[0]
            seed_box = seed.get("bbox_norm_xyxy")
            if not isinstance(seed_box, list) or len(seed_box) != 4:
                continue
            bbox = [safe_float(v) for v in seed_box]
            stage = "seed"
            if target_ar is not None:
                bbox = project_box_to_ar(bbox, max(target_ar / max(image_ar, 1e-6), 1e-6))
                stage = "proj"
            out.append(
                {
                    "teacher_id": teacher_id,
                    "stage": stage,
                    "bbox_norm_xyxy": [round(v, 6) for v in bbox],
                    "teacher_score": safe_float(seed.get("score", 0.0)),
                    "ref_source": "proposal_fallback",
                    "ref_run_tag": src_tag,
                }
            )
            seen.add(teacher_id)
        if len(seen) == 3:
            break
    return out


def nearest_scored_match(ref_box: Sequence[float], scored_pool: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_iou = -1.0
    for cand in scored_pool:
        box = cand.get("bbox_norm_xyxy")
        if not isinstance(box, list):
            continue
        iou = iou_xyxy(ref_box, box)
        if iou > best_iou:
            best = cand
            best_iou = iou
    if best is None:
        return None
    out = dict(best)
    out["_match_iou"] = best_iou
    return out


def make_report(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    artifacts = data_root / "artifacts"
    image_id = str(args.image_id)
    run_tag = str(args.run_tag)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "assets")

    teacher_scores_jsonl = Path(args.teacher_scores_jsonl) if str(args.teacher_scores_jsonl).strip() else artifacts / "teacher" / "scores" / f"teacher_scores_ar_{run_tag}.jsonl"
    candidates_jsonl = Path(args.candidates_jsonl) if str(args.candidates_jsonl).strip() else artifacts / "candidates" / f"candidates_ar_{run_tag}.jsonl"
    features_jsonl = Path(args.features_jsonl) if str(args.features_jsonl).strip() else artifacts / "precompute" / "feats_c2c3c5_v2_strict_enriched_routed.jsonl"

    teacher_row = read_json_row_or_jsonl(args.teacher_row_json, teacher_scores_jsonl, image_id)
    candidate_row = read_json_row_or_jsonl(args.candidate_row_json, candidates_jsonl, image_id)
    feature_row = read_json_row_or_jsonl(args.feature_row_json, features_jsonl, image_id)
    image_path = data_root / "images" / f"{image_id}.jpg"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    subject_box_norm = None
    subject_prior = candidate_row.get("subject_prior", {}) if isinstance(candidate_row.get("subject_prior"), dict) else {}
    if isinstance(subject_prior.get("bbox_norm_xyxy"), list) and len(subject_prior.get("bbox_norm_xyxy")) == 4:
        subject_box_norm = [safe_float(v) for v in subject_prior.get("bbox_norm_xyxy")]
    face_center_norm = None
    c3_pose = feature_row.get("c3_pose", []) if isinstance(feature_row.get("c3_pose"), list) else []
    if c3_pose and isinstance(c3_pose[0], dict):
        face = c3_pose[0].get("face", {}) if isinstance(c3_pose[0].get("face"), dict) else {}
        bbox_norm = face.get("bbox_norm")
        if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
            face_center_norm = [
                (safe_float(bbox_norm[0]) + safe_float(bbox_norm[2])) / 2.0,
                (safe_float(bbox_norm[1]) + safe_float(bbox_norm[3])) / 2.0,
            ]

    ars = [x.strip() for x in str(args.ars).split(",") if x.strip()]
    report_json: Dict[str, Any] = {
        "image_id": image_id,
        "run_tag": run_tag,
        "image_size": [width, height],
        "route_global": teacher_row.get("route_global", {}),
        "ars": {},
    }
    route_global = teacher_row.get("route_global", {}) if isinstance(teacher_row.get("route_global"), dict) else {}
    scorer_cfg = teacher_row.get("teacher_scorer", {}).get("config", {}) if isinstance(teacher_row.get("teacher_scorer", {}).get("config"), dict) else {}

    md_lines: List[str] = [
        f"# {image_id} crop 상세 비교 리포트",
        "",
        f"- run_tag: `{run_tag}`",
        f"- image: `{image_id}`",
        f"- image_size: `{width} x {height}`",
        f"- subject_mode: `{teacher_row.get('route_global', {}).get('subject_mode', '')}`",
        f"- policy_id: `{teacher_row.get('route_global', {}).get('policy_id', '')}`",
        "",
        "> 참고: public teacher box는 `FREE`에서는 `seed`, AR 고정(`1:1`, `16:9`)에서는 `proj`를 대표 box로 사용했습니다.",
        "> `selected_topk`는 strict score rank가 아니라 force/diversity/fill을 거친 `selection order`입니다. 따라서 top5의 final score가 top2보다 높아도 정상일 수 있습니다.",
        "> public teacher 행은 `proposal_injection.public_teacher_ref_eval`의 exact ref score를 우선 사용합니다. exact ref eval이 없는 구 산출물에서만 `가장 가까운 scored candidate` fallback을 사용합니다.",
        "",
        "## Score Formula",
        "",
        "```text",
        "A_macro = weighted_mean(A_aesthetic, A_align)",
        "S_macro = weighted_mean(S_cov, S_scale, S_border, S_softcut_quality)",
        "C_macro = weighted_mean(C_comp, C_headroom, C_lookroom, C_horizon_y, C_sym, C_context, C_copyspace)",
        "T_macro = teacher_consensus_score",
        "",
        "score_rank = weighted_mean(A_macro, S_macro, C_macro, T_macro)",
        "score_policy = score_rank + area_log_prior",
        "```",
        "",
        "> panel은 현재 scorer 기준으로만 표시합니다. `legacy`/old-path ladder는 시각화에서 제거했습니다.",
        "> 아래 AR별 설명에는 이 이미지의 reconstructed macro/rank weight를 같이 적었습니다.",
        "",
    ]

    label_keys = [
        "subject_coverage",
        "subject_scale",
        "face_cut",
        "joint_cut",
        "third_dist",
        "phi_dist",
        "center_dist",
        "headroom",
        "lookroom",
        "context",
        "teacher_consensus",
        "crop_tightness",
    ]

    for ar in ars:
        ar_res = teacher_row["teacher_scorer"]["results_by_ar"][ar]
        topk = list(ar_res.get("selected_topk", []) or [])[:5]
        stored_public_eval = ar_res.get("proposal_injection", {}).get("public_teacher_ref_eval", [])
        if isinstance(stored_public_eval, list) and stored_public_eval:
            teacher_refs = [
                {
                    "teacher_id": str(x.get("teacher_id", "")),
                    "stage": str(x.get("teacher_stage", "")),
                    "bbox_norm_xyxy": x.get("bbox_norm_xyxy"),
                    "teacher_score": safe_float(x.get("teacher_raw_score", 0.0)),
                    "exact_eval": x,
                    "ref_source": "public_teacher_ref_eval",
                    "ref_run_tag": run_tag,
                }
                for x in stored_public_eval
                if isinstance(x, dict)
            ]
        else:
            teacher_refs = best_teacher_refs(
                candidate_row,
                ar,
                data_root=data_root,
                run_tag=run_tag,
                image_id=image_id,
                image_ar=(width / max(1, height)),
            )
        if len({str(x.get("teacher_id", "")) for x in teacher_refs}) < 3:
            fallback_refs = best_teacher_refs(
                candidate_row,
                ar,
                data_root=data_root,
                run_tag=run_tag,
                image_id=image_id,
                image_ar=(width / max(1, height)),
            )
            existing = {str(x.get("teacher_id", "")) for x in teacher_refs}
            for ref in fallback_refs:
                teacher_id = str(ref.get("teacher_id", ""))
                if teacher_id and teacher_id not in existing:
                    teacher_refs.append(ref)
                    existing.add(teacher_id)
        scored_pool = collect_scored_candidates(ar_res)

        overlay_path = out_dir / "assets" / "overlays" / f"{ar.replace(':', 'x')}_overlay.jpg"
        build_overlay(
            image=image,
            title=f"{image_id} | {ar} | top1-top5 + teacher refs",
            topk=topk,
            teacher_refs=teacher_refs,
            subject_box_norm=subject_box_norm,
            face_center_norm=face_center_norm,
            out_path=overlay_path,
        )

        top_items: List[Dict[str, Any]] = []
        top_rows: List[Dict[str, Any]] = []
        for idx, cand in enumerate(topk, start=1):
            bbox_px = denorm_box(cand["bbox_norm_xyxy"], width, height)
            score_details = extract_score_details(cand, route_global, scorer_cfg)
            tile_path = out_dir / "assets" / "crops" / ar.replace(":", "x") / f"rank{idx}_{cand['candidate_id']}.jpg"
            tile = crop_to_tile(
                image,
                bbox_px,
                title=f"Top{idx} | {cand['source']}",
                subtitle=(
                    f"rank={safe_float(cand['scores'].get('rank', cand['scores'].get('final', 0.0))):.4f} | "
                    f"policy={safe_float(cand['scores'].get('policy', cand['scores'].get('final', 0.0))):.4f}"
                ),
                bbox_norm=cand["bbox_norm_xyxy"],
                subject_box_norm=subject_box_norm,
                face_center_norm=face_center_norm,
                teacher_agreement=compute_teacher_agreement(cand["bbox_norm_xyxy"], teacher_refs),
                scores=cand.get("scores", {}),
                components=cand.get("scores", {}).get("components", {}),
                macro_scores=cand.get("macro_scores", {}),
                score_details=score_details,
                labels=cand.get("checklist_labels", {}),
                provenance_text=summarize_provenance(cand),
                badge_specs=[
                    ("third_dist", str(cand.get("checklist_labels", {}).get("third_dist", ""))),
                    ("center_dist", str(cand.get("checklist_labels", {}).get("center_dist", ""))),
                    ("teacher_consensus", str(cand.get("checklist_labels", {}).get("teacher_consensus", ""))),
                    ("headroom", str(cand.get("checklist_labels", {}).get("headroom", ""))),
                    ("lookroom", str(cand.get("checklist_labels", {}).get("lookroom", ""))),
                    ("crop_tightness", str(cand.get("checklist_labels", {}).get("crop_tightness", ""))),
                ],
            )
            ensure_dir(tile_path.parent)
            tile.save(tile_path)
            top_items.append(
                {
                    "bbox_px": bbox_px,
                    "bbox_norm": cand["bbox_norm_xyxy"],
                    "title": f"Top{idx} | {cand['source']}",
                    "subtitle": (
                        f"rank={safe_float(cand['scores'].get('rank', cand['scores'].get('final', 0.0))):.4f} | "
                        f"policy={safe_float(cand['scores'].get('policy', cand['scores'].get('final', 0.0))):.4f}"
                    ),
                    "labels": cand.get("checklist_labels", {}),
                    "scores": cand.get("scores", {}),
                    "components": cand.get("scores", {}).get("components", {}),
                    "macro_scores": cand.get("macro_scores", {}),
                    "score_details": score_details,
                    "teacher_agreement": compute_teacher_agreement(cand["bbox_norm_xyxy"], teacher_refs),
                    "subject_box_norm": subject_box_norm,
                    "face_center_norm": face_center_norm,
                    "provenance_text": summarize_provenance(cand),
                }
            )
            comps = cand.get("scores", {}).get("components", {})
            top_rows.append(
                {
                    "selection_order": idx,
                    "candidate_id": str(cand.get("candidate_id", "")),
                    "source": str(cand.get("source", "")),
                    "bbox_norm_xyxy": cand.get("bbox_norm_xyxy"),
                    "cheap": safe_float(cand.get("scores", {}).get("cheap", 0.0)),
                    "expensive": safe_float(cand.get("scores", {}).get("expensive", 0.0)),
                    "expensive_source": str(comps.get("expensive_source", "")),
                    "rank": safe_float(cand.get("scores", {}).get("rank", cand.get("scores", {}).get("final", 0.0))),
                    "policy": safe_float(cand.get("scores", {}).get("policy", cand.get("scores", {}).get("final", 0.0))),
                    "final": safe_float(cand.get("scores", {}).get("final", 0.0)),
                    "final_legacy": safe_float(cand.get("scores", {}).get("final_legacy", 0.0)),
                    "area_log_prior": safe_float(cand.get("scores", {}).get("area_log_prior", 0.0)),
                    "cov": safe_float(comps.get("cov", 0.0)),
                    "p_cut": safe_float(comps.get("p_cut", 0.0)),
                    "r_comp": safe_float(comps.get("r_comp", 0.0)),
                    "r_teach": safe_float(comps.get("r_teach", 0.0)),
                    "aesthetic_norm": safe_float(comps.get("aesthetic_norm", 0.0)),
                    "cosine_img_text": safe_float(comps.get("cosine_img_text", 0.0)),
                    "teacher_rho": safe_float(comps.get("teacher_rho", 0.0)),
                    "macro_scores": cand.get("macro_scores", {}),
                    "labels": cand.get("checklist_labels", {}),
                    "why_tags": list(cand.get("why_tags", []) or []),
                    "image_rel": tile_path.relative_to(out_dir).as_posix(),
                }
            )
        if top_rows:
            ranked = sorted(top_rows, key=lambda r: r["rank"], reverse=True)
            final_rank_map = {row["candidate_id"]: rank for rank, row in enumerate(ranked, start=1)}
            for row in top_rows:
                row["final_rank_within_selected_topk"] = final_rank_map.get(row["candidate_id"])

        teacher_items: List[Dict[str, Any]] = []
        teacher_rows: List[Dict[str, Any]] = []
        for ref in teacher_refs:
            bbox_px = denorm_box(ref["bbox_norm_xyxy"], width, height)
            exact_eval = ref.get("exact_eval") if isinstance(ref.get("exact_eval"), dict) else None
            matched = exact_eval if exact_eval is not None else nearest_scored_match(ref["bbox_norm_xyxy"], scored_pool.values())
            score_details = extract_score_details(matched or {"scores": {"components": {}}, "checklist": {}, "composition_checks": {}, "area_ratio": 0.0}, route_global, scorer_cfg)
            tile_path = out_dir / "assets" / "crops" / ar.replace(":", "x") / f"teacher_{ref['teacher_id']}_{ref['stage']}.jpg"
            tile = crop_to_tile(
                image,
                bbox_px,
                title=f"{ref['teacher_id']} | {ref['stage']}",
                subtitle=(
                    f"teacher={ref['teacher_score']:.4f} | "
                    f"rank={fmt_opt(matched.get('scores', {}).get('rank') if matched else None)} | "
                    f"policy={fmt_opt(matched.get('scores', {}).get('policy') if matched else None)}"
                ),
                bbox_norm=ref["bbox_norm_xyxy"],
                subject_box_norm=subject_box_norm,
                face_center_norm=face_center_norm,
                teacher_agreement=compute_teacher_agreement(ref["bbox_norm_xyxy"], teacher_refs),
                scores=(matched.get("scores", {}) if matched else {}),
                components=(matched.get("scores", {}).get("components", {}) if matched else {}),
                macro_scores=(matched.get("macro_scores", {}) if matched else {}),
                score_details=score_details,
                labels=((matched or {}).get("checklist_labels", {})),
                provenance_text=summarize_provenance(matched or ref),
                badge_specs=[
                    ("third_dist", str((matched or {}).get("checklist_labels", {}).get("third_dist", ""))),
                    ("center_dist", str((matched or {}).get("checklist_labels", {}).get("center_dist", ""))),
                    ("teacher_consensus", str((matched or {}).get("checklist_labels", {}).get("teacher_consensus", ""))),
                    ("headroom", str((matched or {}).get("checklist_labels", {}).get("headroom", ""))),
                    ("lookroom", str((matched or {}).get("checklist_labels", {}).get("lookroom", ""))),
                    ("crop_tightness", str((matched or {}).get("checklist_labels", {}).get("crop_tightness", ""))),
                ],
            )
            ensure_dir(tile_path.parent)
            tile.save(tile_path)
            teacher_items.append(
                {
                    "bbox_px": bbox_px,
                    "bbox_norm": ref["bbox_norm_xyxy"],
                    "title": f"{ref['teacher_id']} | {ref['stage']}",
                    "subtitle": (
                        f"teacher={ref['teacher_score']:.4f} | "
                        f"rank={fmt_opt(matched.get('scores', {}).get('rank') if matched else None)} | "
                        f"policy={fmt_opt(matched.get('scores', {}).get('policy') if matched else None)}"
                    ),
                    "labels": (matched or {}).get("checklist_labels", {}),
                    "scores": (matched.get("scores", {}) if matched else {}),
                    "components": (matched.get("scores", {}).get("components", {}) if matched else {}),
                    "macro_scores": (matched.get("macro_scores", {}) if matched else {}),
                    "score_details": score_details,
                    "teacher_agreement": compute_teacher_agreement(ref["bbox_norm_xyxy"], teacher_refs),
                    "subject_box_norm": subject_box_norm,
                    "face_center_norm": face_center_norm,
                    "provenance_text": summarize_provenance(matched or ref),
                }
            )
            teacher_rows.append(
                {
                    "selection_order": "public_teacher_exact" if exact_eval is not None else "public_teacher_fallback",
                    "teacher_id": ref["teacher_id"],
                    "stage": ref["stage"],
                    "ref_source": ref.get("ref_source", "unknown"),
                    "ref_run_tag": ref.get("ref_run_tag", run_tag),
                    "teacher_score": ref["teacher_score"],
                    "bbox_norm_xyxy": ref["bbox_norm_xyxy"],
                    "match_iou": (1.0 if exact_eval is not None else safe_float(matched.get("_match_iou", 0.0))) if matched else None,
                    "matched_candidate_id": matched.get("candidate_id") if matched else None,
                    "matched_source": matched.get("source") if matched else None,
                    "matched_cheap": safe_float(matched.get("scores", {}).get("cheap", 0.0)) if matched else None,
                    "matched_expensive": safe_float(matched.get("scores", {}).get("expensive", 0.0)) if matched else None,
                    "matched_rank": safe_float(matched.get("scores", {}).get("rank", matched.get("scores", {}).get("final", 0.0))) if matched else None,
                    "matched_policy": safe_float(matched.get("scores", {}).get("policy", matched.get("scores", {}).get("final", 0.0))) if matched else None,
                    "matched_final": safe_float(matched.get("scores", {}).get("final", 0.0)) if matched else None,
                    "matched_final_legacy": safe_float(matched.get("scores", {}).get("final_legacy", 0.0)) if matched else None,
                    "matched_components": matched.get("scores", {}).get("components", {}) if matched else {},
                    "matched_macro_scores": matched.get("macro_scores", {}) if matched else {},
                    "matched_labels": matched.get("checklist_labels", {}) if matched else {},
                    "matched_why_tags": list(matched.get("why_tags", []) or []) if matched else [],
                    "image_rel": tile_path.relative_to(out_dir).as_posix(),
                }
            )

        panel_path = out_dir / "assets" / "panels" / f"{ar.replace(':', 'x')}_panel.jpg"
        build_panel(
            image=image,
            items=top_items + teacher_items,
            out_path=panel_path,
            title=f"{image_id} | {ar} | top1-top5 + teacher boxes",
        )

        report_json["ars"][ar] = {
            "decision": ar_res.get("decision", {}),
            "proposal_injection": ar_res.get("proposal_injection", {}),
            "topk": top_rows,
            "teacher_refs": teacher_rows,
            "overlay_rel": overlay_path.relative_to(out_dir).as_posix(),
            "panel_rel": panel_path.relative_to(out_dir).as_posix(),
        }
        macro_detail = extract_score_details(topk[0], route_global, scorer_cfg) if topk else {"rank_weights": {}, "group_weights": {}}

        md_lines.extend(
            [
                f"## {ar}",
                "",
                f"- decision: `{ar_res.get('decision', {}).get('decision_type', '')}`",
                f"- chosen_candidate_id: `{ar_res.get('decision', {}).get('chosen_candidate_id', '')}`",
                f"- delta_improve / tau_improve: `{safe_float(ar_res.get('decision', {}).get('delta_improve', 0.0)):.4f}` / `{safe_float(ar_res.get('decision', {}).get('tau_improve', 0.0)):.4f}`",
                f"- proposal injection: `num_teacher_candidates_ar={int(ar_res.get('proposal_injection', {}).get('num_teacher_candidates_ar', 0))}`, `candidate_top1_iou_to_teacher_seed={safe_float(ar_res.get('proposal_injection', {}).get('candidate_top1_iou_to_teacher_seed', 0.0)):.4f}`",
                f"- teacher consensus: `{json.dumps(ar_res.get('proposal_injection', {}).get('teacher_consensus', {}), ensure_ascii=False)}`",
                f"- visualization extras: `subject_box`, `face_center`, `teacher agreement strip`, `margin ruler`, `risk badges`, `macro A/S/C/T detail`",
                f"- public teacher refs in scorer payload: `{len(teacher_refs)}`",
                (f"- expensive path status: `{str(topk[0].get('scores', {}).get('components', {}).get('expensive_source', 'unknown'))}`"
                 if topk else "- expensive path status: `na`"),
                f"- reconstructed rank weights: `A={safe_float(macro_detail.get('rank_weights', {}).get('A_macro', 0.0)):.3f}, S={safe_float(macro_detail.get('rank_weights', {}).get('S_macro', 0.0)):.3f}, C={safe_float(macro_detail.get('rank_weights', {}).get('C_macro', 0.0)):.3f}, T={safe_float(macro_detail.get('rank_weights', {}).get('T_macro', 0.0)):.3f}`",
                f"- reconstructed A weights: `{json.dumps(macro_detail.get('group_weights', {}).get('A', {}), ensure_ascii=False)}`",
                f"- reconstructed S weights: `{json.dumps(macro_detail.get('group_weights', {}).get('S', {}), ensure_ascii=False)}`",
                f"- reconstructed C weights: `{json.dumps(macro_detail.get('group_weights', {}).get('C', {}), ensure_ascii=False)}`",
                f"- policy area_log_prior: `{safe_float(topk[0].get('scores', {}).get('area_log_prior', 0.0), 0.0):.4f}`" if topk else "- policy area_log_prior: `na`",
                "",
                f"### {ar} overlay",
                "",
                f"![]({overlay_path.relative_to(out_dir).as_posix()})",
                "",
                f"### {ar} contact sheet",
                "",
                "- grid color guide: `yellow=thirds`, `purple=phi-grid`, `cyan=center`",
                "- point markers: thirds/phi 교차점과 center 교차점을 같이 표시합니다.",
                "- `subject_box`: 하늘색 사각형으로 crop 내부 주피사체 영역을 표시합니다.",
                "- `face_center`: 분홍 점으로 얼굴 중심을 표시합니다.",
                "- `teacher agreement`: 각 teacher ref와의 IoU를 작은 bar로 표시합니다.",
                "- `margin ruler`: subject box 기준 좌/우/상/하 여백 비율을 표시합니다.",
                "- `score summary`: `rank / policy / area_log_prior`를 텍스트로 표시합니다.",
                "- `rank fuse A/S/C/T`: 최종 `score_rank`를 구성하는 macro 항의 가중 기여를 미니 bar로 표시합니다.",
                "- `A detail`: `A_aesthetic / A_align`의 내부 항을 표시합니다.",
                "- `expensive_source=disabled|missing` 이면 A detail은 참고용 0값만 표시되고 rank fusion에서는 제외됩니다.",
                "- `S detail`: `S_cov / S_scale / S_border / S_softcut_quality`를 표시합니다.",
                "- `C detail`: `C_comp / C_headroom / C_lookroom / C_horizon_y / C_sym / C_context / C_copyspace`를 표시합니다.",
                "- `T/policy`: `T_teacher`와 `policy(rank + area_log_prior)`를 함께 표시합니다.",
                "- `label strip`: `face / joint / head / look / teach / tight` verdict를 색으로 요약합니다.",
                "`green`: 양호/통과 계열",
                "`orange`: 경계/약한 위반 계열",
                "`red`: 명확한 리스크/불합격 계열",
                "`gray`: 해당 없음 또는 중립",
                "- `provenance badge`: 타일 우상단에 실제 lineage 요약을 표시합니다. `teacher_provenance`가 있으면 `t=<teacher_id>:<stage>`를 우선 표기하고, 없으면 `source_lineage`를 표시합니다.",
                ("> 참고: 이번 AR 결과에는 `public_teacher_ref_eval`이 없어 teacher agreement strip과 teacher box 비교가 비어 있습니다. "
                 "이는 current run에 public teacher proposal이 주입되지 않았다는 뜻입니다." if not teacher_refs else ""),
                "",
                f"![]({panel_path.relative_to(out_dir).as_posix()})",
                "",
                f"### {ar} selected_topk (selection order) + public teacher exact eval 비교",
                "",
                "- `sel_order`는 `selected_topk`에 저장된 선택 순서입니다. strict score rank가 아닙니다.",
                "- `final_rank_in_selected`는 현재 표시된 selected_topk 5개 내부에서 `score_rank`만 다시 정렬했을 때의 순위입니다.",
                "- `teacher:*` 행은 public teacher reference box 자체의 exact eval이며, 원래의 selected_topk 경쟁 과정에 직접 포함되지 않았을 수 있습니다.",
                "",
                "|row|sel_order|final_rank_in_selected|source|candidate_id|bbox_norm_xyxy|teacher_score|matched_iou|cheap|expensive|exp_source|rank|policy|area_log_prior|cov|p_cut|r_comp|r_teach|A_norm|cos|teacher_rho|A/S/C/T|labels|why_tags|crop image|",
                "|---|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
            ]
        )
        for row in top_rows:
            macro = row.get("macro_scores", {}) or {}
            md_lines.append(
                f"|top{row['selection_order']}|{row['selection_order']}|{int(row.get('final_rank_within_selected_topk') or 0)}|{row['source']}|`{row['candidate_id']}`|`{row['bbox_norm_xyxy']}`|"
                f"na|na|{row['cheap']:.4f}|{row['expensive']:.4f}|{row['expensive_source']}|{row['rank']:.4f}|{row['policy']:.4f}|{row['area_log_prior']:.4f}|{row['cov']:.4f}|{row['p_cut']:.4f}|"
                f"{row['r_comp']:.4f}|{row['r_teach']:.4f}|{row['aesthetic_norm']:.4f}|{row['cosine_img_text']:.4f}|"
                f"{row['teacher_rho']:.4f}|A={fmt_opt(macro.get('A_macro'))}, S={fmt_opt(macro.get('S_macro'))}, C={fmt_opt(macro.get('C_macro'))}, T={fmt_opt(macro.get('T_macro'))}|"
                f"{stringify_labels(row['labels'], label_keys)}|{', '.join(row['why_tags'][:6])}|"
                f"[img]({row['image_rel']})|"
            )
        for row in teacher_rows:
            matched_scores = {
                "cheap": row.get("matched_cheap"),
                "expensive": row.get("matched_expensive"),
                "rank": row.get("matched_rank"),
                "policy": row.get("matched_policy"),
                "final": row.get("matched_final"),
                "legacy": row.get("matched_final_legacy"),
            }
            matched_components = row.get("matched_components", {}) or {}
            matched_macro = row.get("matched_macro_scores", {}) or {}
            matched_labels = stringify_labels(row["matched_labels"], label_keys) if row["matched_labels"] else "na"
            matched_why = ", ".join(row.get("matched_why_tags", [])[:6]) if row.get("matched_why_tags") else f"matched:{row['matched_source'] or 'na'}"
            md_lines.append(
                f"|teacher:{row['teacher_id']}|{row['selection_order']}|0|public_teacher_ref|`{row['teacher_id']}:{row['stage']}`|`{row['bbox_norm_xyxy']}`|"
                f"{row['teacher_score']:.4f}|{fmt_opt(row['match_iou'])}|{fmt_opt(matched_scores['cheap'])}|"
                f"{fmt_opt(matched_scores['expensive'])}|{fmt_opt(matched_components.get('expensive_source'))}|{fmt_opt(matched_scores['rank'])}|{fmt_opt(matched_scores['policy'])}|na|"
                f"{fmt_opt(matched_components.get('cov'))}|{fmt_opt(matched_components.get('p_cut'))}|{fmt_opt(matched_components.get('r_comp'))}|"
                f"{fmt_opt(matched_components.get('r_teach'))}|{fmt_opt(matched_components.get('aesthetic_norm'))}|"
                f"{fmt_opt(matched_components.get('cosine_img_text'))}|{fmt_opt(matched_components.get('teacher_rho'))}|"
                f"A={fmt_opt(matched_macro.get('A_macro'))}, S={fmt_opt(matched_macro.get('S_macro'))}, C={fmt_opt(matched_macro.get('C_macro'))}, T={fmt_opt(matched_macro.get('T_macro'))}|"
                f"{matched_labels}|{matched_why}|[img]({row['image_rel']})|"
            )

        md_lines.extend(
            [
                "",
                f"### {ar} teacher box 비교",
                "",
                ("> 현재 run 산출물에는 이 AR의 public teacher exact ref가 없습니다." if not teacher_rows else ""),
                ("> 일부 teacher ref는 current run payload가 비어 있을 때 동일 image의 다른 complete run public-teacher proposal에서 보강됩니다." if any(str(r.get("ref_source", "")) == "proposal_fallback" for r in teacher_rows) else ""),
                "",
                "|teacher|stage|ref_source|ref_run_tag|teacher_score|bbox_norm_xyxy|best scored IoU|matched source|matched candidate_id|matched cheap|matched expensive|matched final|matched labels|crop image|",
                "|---|---|---|---|---:|---|---:|---|---|---:|---:|---:|---|---|",
            ]
        )
        for row in teacher_rows:
            md_lines.append(
                f"|{row['teacher_id']}|{row['stage']}|{row.get('ref_source','')}|{row.get('ref_run_tag','')}|{row['teacher_score']:.4f}|`{row['bbox_norm_xyxy']}`|"
                f"{fmt_opt(row['match_iou'])}|{row['matched_source'] or 'na'}|"
                f"`{row['matched_candidate_id'] or 'na'}`|"
                f"{fmt_opt(row['matched_cheap'])}|"
                f"{fmt_opt(row['matched_expensive'])}|"
                f"{fmt_opt(row['matched_final'])}|"
                f"{stringify_labels(row['matched_labels'], label_keys) if row['matched_labels'] else 'na'}|"
                f"[img]({row['image_rel']})|"
            )
        md_lines.append("")

    report_path = out_dir / f"REPORT_{image_id}_KO.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path = out_dir / "assets" / "analysis.json"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] report={report_path}")
    print(f"[done] analysis_json={json_path}")


def main() -> None:
    make_report(parse_args())


if __name__ == "__main__":
    main()
