from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_BEFORE_CANDIDATES = (
    "data/GAIC_v2/Test500_SSTK_LocalQF_Smoke10_B1/artifacts/candidates/"
    "candidates_ar_gaic_v2_test500_localqf_smoke10_b1.jsonl"
)
DEFAULT_AFTER_CANDIDATES = (
    "data/GAIC_v2/Test500_SSTK_LocalQF_C7_Smoke10_B1/artifacts/candidates/"
    "candidates_ar_gaic_v2_test500_localqf_c7_smoke10_b1.jsonl"
)
DEFAULT_BEFORE_TEACHER = (
    "data/GAIC_v2/Test500_SSTK_LocalQF_Smoke10_B1/artifacts/teacher/scores/"
    "teacher_scores_ar_gaic_v2_test500_localqf_smoke10_b1.jsonl"
)
DEFAULT_AFTER_TEACHER = (
    "data/GAIC_v2/Test500_SSTK_LocalQF_C7_Smoke10_B1/artifacts/teacher/scores/"
    "teacher_scores_ar_gaic_v2_test500_localqf_c7_smoke10_b1.jsonl"
)
DEFAULT_IMAGE_DIR = "data/GAIC_v2/Test500_SSTK_LocalQF_C7_Smoke10_B1/images"
DEFAULT_OUT_DIR = "Implement_Docs/assets_post_20260317_subject_c7"


PANEL_W = 560
PANEL_H = 420
LABEL_H = 208
GUTTER = 18
BG = (245, 247, 250)
INK = (24, 29, 36)
MUTED = (87, 96, 111)
SUBJECT = (16, 212, 216)
ENVELOPE = (125, 211, 252)
CORE = (245, 158, 11)
CROP = (228, 87, 46)
BEFORE = (121, 85, 72)
AFTER = (20, 120, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paired before/after C7 support-map visualizations for the post-2026-03-17 report."
    )
    parser.add_argument("--before_candidates", default=DEFAULT_BEFORE_CANDIDATES)
    parser.add_argument("--after_candidates", default=DEFAULT_AFTER_CANDIDATES)
    parser.add_argument("--before_teacher", default=DEFAULT_BEFORE_TEACHER)
    parser.add_argument("--after_teacher", default=DEFAULT_AFTER_TEACHER)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_samples", type=int, default=10)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def load_jsonl_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                out[image_id] = row
    return out


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clip_box(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1 = max(0.0, min(1.0, safe_float(box[0])))
    y1 = max(0.0, min(1.0, safe_float(box[1])))
    x2 = max(0.0, min(1.0, safe_float(box[2])))
    y2 = max(0.0, min(1.0, safe_float(box[3])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6:
        return None
    return [x1, y1, x2, y2]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def decode_support_grid(support_spec: Dict[str, Any]) -> Optional[np.ndarray]:
    grid_size = int(safe_float(support_spec.get("support_grid_size") or support_spec.get("grid_size"), 0))
    mass_b64 = support_spec.get("support_mass_grid_f16_b64") or support_spec.get("mass_grid_f16_b64")
    if grid_size <= 0 or not isinstance(mass_b64, str) or not mass_b64:
        return None
    try:
        grid = np.frombuffer(base64.b64decode(mass_b64.encode("ascii")), dtype=np.float16).astype(np.float32)
    except Exception:
        return None
    if grid.size != grid_size * grid_size:
        return None
    grid = grid.reshape((grid_size, grid_size))
    if float(grid.max()) <= 1e-8:
        return None
    return grid


def pick_first_box(*boxes: Any) -> Optional[List[float]]:
    for box in boxes:
        clipped = clip_box(box)
        if clipped is not None:
            return clipped
    return None


def extract_overlay(candidate_row: Dict[str, Any]) -> Dict[str, Any]:
    routing = safe_dict(candidate_row.get("routing"))
    subject_prior = safe_dict(candidate_row.get("subject_prior"))
    region = safe_dict(subject_prior.get("effective_subject_region"))
    guidance = safe_dict(subject_prior.get("crop_guidance_spec") or region.get("crop_guidance_spec"))
    support_spec = safe_dict(guidance.get("support_spec"))
    semantic_spec = safe_dict(guidance.get("semantic_spec"))
    guidance_spec = safe_dict(guidance.get("guidance_spec"))
    support_grid = decode_support_grid(support_spec)
    support_visual = bool(
        support_grid is not None
        or support_spec.get("mode") == "support_map"
        or region.get("score_mode") == "support_map"
        or region.get("support_map_enabled")
    )
    bbox = pick_first_box(
        region.get("effective_bbox_norm_xyxy"),
        subject_prior.get("effective_bbox_norm_xyxy"),
        subject_prior.get("bbox_norm_xyxy"),
        safe_dict(candidate_row.get("subject")).get("primary_box"),
    )
    if support_visual:
        # In support-map cases, prefer the support-preservation core/envelope.
        # Some historical rows keep a semantic/raw guidance core that can point
        # to a different object than the C7 support mass.
        core = pick_first_box(
            support_spec.get("core_bbox_norm_xyxy"),
            support_spec.get("latent_support_core_bbox_norm_xyxy"),
            region.get("latent_support_core_bbox_norm_xyxy"),
            guidance_spec.get("core_bbox_norm_xyxy"),
            region.get("guidance_core_bbox_norm_xyxy"),
            bbox,
        )
        envelope = pick_first_box(
            support_spec.get("envelope_norm_xyxy"),
            support_spec.get("latent_support_bbox_norm_xyxy"),
            region.get("latent_support_bbox_norm_xyxy"),
            region.get("support_bbox_norm_xyxy"),
            guidance_spec.get("envelope_bbox_norm_xyxy"),
            region.get("guidance_envelope_bbox_norm_xyxy"),
            bbox,
        )
    else:
        core = pick_first_box(
            guidance_spec.get("core_bbox_norm_xyxy"),
            region.get("guidance_core_bbox_norm_xyxy"),
            support_spec.get("core_bbox_norm_xyxy"),
            bbox,
        )
        envelope = pick_first_box(
            guidance_spec.get("envelope_bbox_norm_xyxy"),
            region.get("guidance_envelope_bbox_norm_xyxy"),
            support_spec.get("envelope_norm_xyxy"),
            bbox,
        )
    secondary = pick_first_box(
        guidance_spec.get("secondary_core_bbox_norm_xyxy"),
        region.get("guidance_secondary_core_bbox_norm_xyxy"),
    )
    return {
        "subject_mode": str(routing.get("subject_mode") or subject_prior.get("subject_mode") or ""),
        "policy_id": str(routing.get("policy_id") or subject_prior.get("policy_id") or ""),
        "score_mode": str(region.get("score_mode") or ""),
        "state": str(region.get("state") or ""),
        "subject_repr_type": str(region.get("subject_repr_type") or subject_prior.get("subject_repr_type") or ""),
        "support_trust_tier": str(
            region.get("support_trust_tier") or semantic_spec.get("support_trust_tier") or support_spec.get("support_trust_tier") or ""
        ),
        "support_map_enabled": bool(region.get("support_map_enabled") or support_spec.get("mode") == "support_map"),
        "support_hybrid_enabled": bool(region.get("support_hybrid_enabled") or support_spec.get("hybrid_enabled")),
        "guidance_preserves_latent_support": bool(region.get("guidance_preserves_latent_support") or support_spec.get("guidance_preserves_latent_support")),
        "bbox": bbox,
        "core": core,
        "envelope": envelope,
        "secondary": secondary,
        "support_grid": support_grid,
        "support_spec": support_spec,
    }


def candidate_score(candidate: Dict[str, Any]) -> float:
    scores = safe_dict(candidate.get("scores"))
    for key in ("crop_utility_raw", "final", "policy_safe", "policy"):
        if key in scores:
            return safe_float(scores.get(key), 0.0)
    return 0.0


def find_candidate_by_id(result: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    if not candidate_id:
        return None
    for key in ("baseline_candidate", "best_candidate"):
        candidate = safe_dict(result.get(key))
        if str(candidate.get("candidate_id", "")) == candidate_id:
            return candidate
    for key in ("selected_topk", "utility_pool", "rank_pool", "cheap_top_m", "hard_negatives", "also_considered_rejected"):
        value = result.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            candidate = safe_dict(item)
            if str(candidate.get("candidate_id", "")) == candidate_id:
                return candidate
    return None


def resolve_selected_candidate(result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    decision = safe_dict(result.get("decision"))
    chosen_id = str(decision.get("chosen_candidate_id", "") or "")
    chosen = find_candidate_by_id(result, chosen_id)
    if chosen is not None:
        return chosen, "decision.chosen_candidate_id"
    best = safe_dict(result.get("best_candidate"))
    if best:
        return best, "best_candidate_fallback"
    baseline = safe_dict(result.get("baseline_candidate"))
    if baseline:
        return baseline, "baseline_candidate_fallback"
    return {}, "missing"


def choose_ar(before_teacher: Dict[str, Any], after_teacher: Dict[str, Any]) -> str:
    before_results = safe_dict(safe_dict(before_teacher.get("teacher_scorer")).get("results_by_ar"))
    after_results = safe_dict(safe_dict(after_teacher.get("teacher_scorer")).get("results_by_ar"))
    best_ar = ""
    best_value = -1e9
    for ar, after_result in after_results.items():
        before_result = safe_dict(before_results.get(ar))
        before_selected, _ = resolve_selected_candidate(before_result)
        after_selected, _ = resolve_selected_candidate(safe_dict(after_result))
        before_box = clip_box(before_selected.get("bbox_norm_xyxy")) or [0.0, 0.0, 1.0, 1.0]
        after_box = clip_box(after_selected.get("bbox_norm_xyxy")) or [0.0, 0.0, 1.0, 1.0]
        labels = safe_dict(after_selected.get("checklist_labels"))
        non_na_support = sum(
            1
            for key in (
                "subject_coverage",
                "support_component_recall",
                "support_component_balance",
                "support_density",
                "support_structure",
                "support_extent_area",
            )
            if labels.get(key) and not str(labels.get(key)).endswith("_na")
        )
        source = str(after_selected.get("source", ""))
        visual_delta = 1.0 - box_iou(before_box, after_box)
        value = non_na_support + visual_delta
        if "guidance" in source or "support" in source or "saliency" in source:
            value += 3.0
        if ar in {"16:9", "9:16", "1:1"}:
            value += 0.15
        if value > best_value:
            best_value = value
            best_ar = str(ar)
    return best_ar or "1:1"


def norm_to_panel(box: Sequence[float], ox: int, oy: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return (
        int(round(ox + box[0] * w)),
        int(round(oy + box[1] * h)),
        int(round(ox + box[2] * w)),
        int(round(oy + box[3] * h)),
    )


def fit_image(image: Image.Image, size: Tuple[int, int]) -> Tuple[Image.Image, int, int, int, int]:
    target_w, target_h = size
    img = image.convert("RGB")
    scale = min(target_w / max(1, img.width), target_h / max(1, img.height))
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (232, 236, 242))
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, ox, oy, new_w, new_h


def draw_box(draw: ImageDraw.ImageDraw, panel_origin: Tuple[int, int], display_box: Tuple[int, int, int, int], color: Tuple[int, int, int], width: int = 3) -> None:
    draw.rectangle(display_box, outline=color, width=width)


def draw_support_heat(canvas: Image.Image, grid: Optional[np.ndarray], x: int, y: int, w: int, h: int) -> None:
    if grid is None or grid.size == 0:
        return
    max_mass = float(grid.max())
    if max_mass <= 1e-8:
        return
    alpha = np.sqrt(np.clip(grid / max_mass, 0.0, 1.0))
    alpha_u8 = np.asarray(np.clip(alpha * 175.0, 0.0, 255.0), dtype=np.uint8)
    alpha_img = Image.fromarray(alpha_u8, mode="L").resize((w, h), Image.Resampling.BILINEAR)
    heat = Image.new("RGBA", (w, h), (16, 212, 216, 0))
    heat.putalpha(alpha_img)
    canvas.alpha_composite(heat, (x, y))


def draw_multiline(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], lines: Sequence[str], font: ImageFont.ImageFont, fill: Tuple[int, int, int] = INK, line_h: int = 21) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h


def draw_legend(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    entries = [
        (CROP, "red: final decision crop"),
        (SUBJECT, "cyan: effective subject"),
        (ENVELOPE, "sky-blue: support/guidance envelope"),
        (CORE, "orange: support/guidance core"),
    ]
    start_x, start_y = x, y
    for idx, (color, text) in enumerate(entries):
        col = idx % 2
        row = idx // 2
        lx = start_x + col * 530
        ly = start_y + row * 22
        draw.rectangle((lx, ly + 3, lx + 14, ly + 17), outline=color, width=3)
        draw.text((lx + 20, ly + 2), text, font=font, fill=MUTED)


def crop_preview(image: Image.Image, box: Optional[Sequence[float]], size: Tuple[int, int]) -> Image.Image:
    if box is None:
        return Image.new("RGB", size, (232, 236, 242))
    x1 = int(round(box[0] * image.width))
    y1 = int(round(box[1] * image.height))
    x2 = int(round(box[2] * image.width))
    y2 = int(round(box[3] * image.height))
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width, x2))
    y2 = max(y1 + 1, min(image.height, y2))
    crop = image.crop((x1, y1, x2, y2))
    fitted, _, _, _, _ = fit_image(crop, size)
    return fitted


def support_label(candidate: Dict[str, Any], key: str) -> str:
    labels = safe_dict(candidate.get("checklist_labels"))
    value = labels.get(key)
    return str(value) if value is not None else "n/a"


def short_float(value: Any) -> str:
    return f"{safe_float(value):.3f}"


def render_sample(
    *,
    image_id: str,
    index: int,
    image_path: Path,
    before_candidate_row: Dict[str, Any],
    after_candidate_row: Dict[str, Any],
    before_teacher_row: Dict[str, Any],
    after_teacher_row: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    before_overlay = extract_overlay(before_candidate_row)
    after_overlay = extract_overlay(after_candidate_row)
    target_ar = choose_ar(before_teacher_row, after_teacher_row)
    before_result = safe_dict(safe_dict(before_teacher_row.get("teacher_scorer")).get("results_by_ar")).get(target_ar, {})
    after_result = safe_dict(safe_dict(after_teacher_row.get("teacher_scorer")).get("results_by_ar")).get(target_ar, {})
    before_best = safe_dict(safe_dict(before_result).get("best_candidate"))
    after_best = safe_dict(safe_dict(after_result).get("best_candidate"))
    before_selected, before_selected_source = resolve_selected_candidate(safe_dict(before_result))
    after_selected, after_selected_source = resolve_selected_candidate(safe_dict(after_result))
    before_crop = clip_box(before_selected.get("bbox_norm_xyxy"))
    after_crop = clip_box(after_selected.get("bbox_norm_xyxy"))
    crop_iou = box_iou(before_crop or [0, 0, 1, 1], after_crop or [0, 0, 1, 1])

    width = PANEL_W * 2 + GUTTER * 3
    height = LABEL_H + PANEL_H * 2 + GUTTER * 3
    canvas = Image.new("RGBA", (width, height), BG + (255,))
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)
    small_font = load_font(13)
    title_font = load_font(17)

    title = f"{index:02d}. {image_id} | target AR {target_ar} | support-map before/after"
    draw.text((GUTTER, 14), title, font=title_font, fill=INK)
    draw.text(
        (GUTTER, 38),
        "C7 heat is drawn only on the AFTER panel when support mass is available.",
        font=small_font,
        fill=MUTED,
    )
    draw_legend(draw, (GUTTER, 58), small_font)

    before_lines = [
        f"BEFORE no C7: {before_overlay['subject_mode']} / {before_overlay['score_mode']}",
        f"repr={before_overlay['subject_repr_type']} trust={before_overlay['support_trust_tier']}",
        f"crop src={before_selected.get('source', '')}",
        f"utility={short_float(candidate_score(before_selected))} support={support_label(before_selected, 'support_structure')}",
    ]
    after_lines = [
        f"AFTER C7: {after_overlay['subject_mode']} / {after_overlay['score_mode']}",
        f"repr={after_overlay['subject_repr_type']} trust={after_overlay['support_trust_tier']}",
        f"crop src={after_selected.get('source', '')}",
        f"utility={short_float(candidate_score(after_selected))} support={support_label(after_selected, 'support_structure')}",
    ]
    draw_multiline(draw, (GUTTER, 112), before_lines, font, BEFORE)
    draw_multiline(draw, (GUTTER + PANEL_W + GUTTER, 112), after_lines, font, AFTER)

    top_y = LABEL_H + GUTTER
    panels = [
        ("before", GUTTER, top_y, before_overlay, before_crop, before_best, False),
        ("after", GUTTER + PANEL_W + GUTTER, top_y, after_overlay, after_crop, after_best, True),
    ]
    for _, px, py, overlay, crop_box, _, use_heat in panels:
        fitted, ox, oy, disp_w, disp_h = fit_image(image, (PANEL_W, PANEL_H))
        canvas.alpha_composite(fitted.convert("RGBA"), (px, py))
        if use_heat:
            draw_support_heat(canvas, overlay.get("support_grid"), px + ox, py + oy, disp_w, disp_h)
        draw = ImageDraw.Draw(canvas)
        if overlay.get("envelope"):
            draw_box(draw, (px, py), norm_to_panel(overlay["envelope"], px + ox, py + oy, disp_w, disp_h), ENVELOPE, 2)
        if overlay.get("bbox"):
            draw_box(draw, (px, py), norm_to_panel(overlay["bbox"], px + ox, py + oy, disp_w, disp_h), SUBJECT, 3)
        if overlay.get("core"):
            draw_box(draw, (px, py), norm_to_panel(overlay["core"], px + ox, py + oy, disp_w, disp_h), CORE, 3)
        if overlay.get("secondary"):
            draw_box(draw, (px, py), norm_to_panel(overlay["secondary"], px + ox, py + oy, disp_w, disp_h), (52, 211, 153), 2)
        if crop_box:
            draw_box(draw, (px, py), norm_to_panel(crop_box, px + ox, py + oy, disp_w, disp_h), CROP, 4)

    crop_y = top_y + PANEL_H + GUTTER
    before_preview = crop_preview(image, before_crop, (PANEL_W, PANEL_H))
    after_preview = crop_preview(image, after_crop, (PANEL_W, PANEL_H))
    canvas.alpha_composite(before_preview.convert("RGBA"), (GUTTER, crop_y))
    canvas.alpha_composite(after_preview.convert("RGBA"), (GUTTER + PANEL_W + GUTTER, crop_y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((GUTTER, crop_y, GUTTER + PANEL_W, crop_y + 24), fill=(0, 0, 0, 160))
    draw.rectangle((GUTTER + PANEL_W + GUTTER, crop_y, GUTTER + PANEL_W * 2 + GUTTER, crop_y + 24), fill=(0, 0, 0, 160))
    draw.text((GUTTER + 8, crop_y + 5), "BEFORE selected crop", font=font, fill=(255, 255, 255))
    draw.text((GUTTER + PANEL_W + GUTTER + 8, crop_y + 5), "AFTER selected crop", font=font, fill=(255, 255, 255))
    footer = (
        f"crop_iou={crop_iou:.3f} | "
        f"after S_cov={safe_dict(after_selected.get('macro_components')).get('S_cov')} | "
        f"after support_mass_grid={'yes' if after_overlay.get('support_grid') is not None else 'no'}"
    )
    draw.text((GUTTER, height - 27), footer, font=small_font, fill=MUTED)

    filename = f"sample_{index:02d}_{image_id}_{target_ar.replace(':', 'x')}.png"
    out_path = out_dir / filename
    canvas.convert("RGB").save(out_path, quality=95)
    return {
        "index": index,
        "image_id": image_id,
        "target_ar": target_ar,
        "path": str(out_path),
        "before_source": str(before_selected.get("source", "")),
        "after_source": str(after_selected.get("source", "")),
        "before_selected_resolution": before_selected_source,
        "after_selected_resolution": after_selected_source,
        "before_chosen_candidate_id": str(safe_dict(before_result.get("decision")).get("chosen_candidate_id", "") or ""),
        "after_chosen_candidate_id": str(safe_dict(after_result.get("decision")).get("chosen_candidate_id", "") or ""),
        "before_best_candidate_id": str(before_best.get("candidate_id", "") or ""),
        "after_best_candidate_id": str(after_best.get("candidate_id", "") or ""),
        "before_utility": candidate_score(before_selected),
        "after_utility": candidate_score(after_selected),
        "crop_iou": crop_iou,
        "before_score_mode": before_overlay.get("score_mode"),
        "after_score_mode": after_overlay.get("score_mode"),
        "before_subject_repr": before_overlay.get("subject_repr_type"),
        "after_subject_repr": after_overlay.get("subject_repr_type"),
        "after_support_map": bool(after_overlay.get("support_grid") is not None),
        "after_support_structure": support_label(after_selected, "support_structure"),
        "after_subject_coverage": support_label(after_selected, "subject_coverage"),
    }


def build_contact_sheet(sample_paths: Sequence[Path], out_path: Path) -> None:
    thumbs: List[Image.Image] = []
    thumb_w, thumb_h = 420, 320
    for path in sample_paths:
        img = Image.open(path).convert("RGB")
        fitted, _, _, _, _ = fit_image(img, (thumb_w, thumb_h))
        thumbs.append(fitted)
    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * GUTTER, rows * thumb_h + (rows + 1) * GUTTER), BG)
    for i, thumb in enumerate(thumbs):
        row = i // cols
        col = i % cols
        x = GUTTER + col * (thumb_w + GUTTER)
        y = GUTTER + row * (thumb_h + GUTTER)
        sheet.paste(thumb, (x, y))
    sheet.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    before_candidates = load_jsonl_map(Path(args.before_candidates))
    after_candidates = load_jsonl_map(Path(args.after_candidates))
    before_teacher = load_jsonl_map(Path(args.before_teacher))
    after_teacher = load_jsonl_map(Path(args.after_teacher))
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_ids = sorted(set(before_candidates) & set(after_candidates) & set(before_teacher) & set(after_teacher))
    records: List[Dict[str, Any]] = []
    for image_id in common_ids[: max(0, int(args.max_samples))]:
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{image_id}.png"
        if not image_path.exists():
            continue
        records.append(
            render_sample(
                image_id=image_id,
                index=len(records) + 1,
                image_path=image_path,
                before_candidate_row=before_candidates[image_id],
                after_candidate_row=after_candidates[image_id],
                before_teacher_row=before_teacher[image_id],
                after_teacher_row=after_teacher[image_id],
                out_dir=out_dir,
            )
        )

    contact_path = out_dir / "contact_sheet_subject_c7_before_after.png"
    build_contact_sheet([Path(r["path"]) for r in records], contact_path)
    manifest = {
        "before_candidates": args.before_candidates,
        "after_candidates": args.after_candidates,
        "before_teacher": args.before_teacher,
        "after_teacher": args.after_teacher,
        "image_dir": args.image_dir,
        "num_samples": len(records),
        "contact_sheet": str(contact_path),
        "records": records,
    }
    manifest_path = out_dir / "subject_c7_before_after_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "num_samples": len(records), "manifest": str(manifest_path), "contact_sheet": str(contact_path)}, indent=2))


if __name__ == "__main__":
    main()
