#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mobilecropnet_v4.data import (  # noqa: E402
    DECISION_VOCAB,
    SUBJECT_MODE_VOCAB,
    TARGET_AR_VOCAB,
    box_area,
    letterbox_content_box,
    letterbox_to_original_box,
    load_image_tensor_letterbox,
    target_ar_id,
)
from mobilecropnet_v4.eval_utils import (  # noqa: E402
    class_label,
    detailed_explanation_from_logits,
    infer_image_norm,
    load_mobilecropnet_v4_checkpoint,
    quality_label,
    select_generated_proposal_index,
)
from mobilecropnet_v4.gaic_benchmark import load_gaic_annotation_records, order_desc  # noqa: E402
from scripts.evaluate_public_croppers_gaic_v2 import _score_gaic  # noqa: E402


AR_COLORS: dict[str, tuple[int, int, int]] = {
    "FREE": (226, 54, 54),
    "1:1": (44, 154, 84),
    "3:4": (45, 105, 214),
    "4:3": (229, 143, 39),
    "16:9": (156, 100, 214),
    "9:16": (48, 190, 205),
}
PUBLIC_COLOR = (245, 217, 70)
GT_COLOR = (235, 235, 235)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [clamp01(safe_float(v)) for v in box[:4]]
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def crop_ar(box: Sequence[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = [clamp01(safe_float(v)) for v in box[:4]]
    bw = max(1e-6, (x2 - x1) * float(width))
    bh = max(1e-6, (y2 - y1) * float(height))
    return bw / bh


def center_crop_for_ar(target_ar: str, width: int, height: int) -> list[float]:
    if target_ar == "FREE":
        return [0.0, 0.0, 1.0, 1.0]
    ar_values = {"1:1": 1.0, "3:4": 3.0 / 4.0, "4:3": 4.0 / 3.0, "16:9": 16.0 / 9.0, "9:16": 9.0 / 16.0}
    ar = ar_values[target_ar]
    image_ar = float(width) / float(max(1, height))
    if image_ar >= ar:
        crop_h = float(height)
        crop_w = crop_h * ar
    else:
        crop_w = float(width)
        crop_h = crop_w / ar
    x1 = (float(width) - crop_w) * 0.5 / float(width)
    y1 = (float(height) - crop_h) * 0.5 / float(height)
    x2 = x1 + crop_w / float(width)
    y2 = y1 + crop_h / float(height)
    return [clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)]


def collect_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.annotations_json is not None:
        records = load_gaic_annotation_records(args.annotations_json, image_roots=args.image_roots, max_images=args.max_images)
        return [row for row in records if str(row.get("image_path", "")).strip()]
    paths: list[Path] = []
    for pattern in args.image_glob or []:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    for path in args.image or []:
        paths.append(Path(path))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            with Image.open(path) as img:
                width, height = img.size
        except (OSError, UnidentifiedImageError):
            continue
        out.append({"image_id": path.stem, "file_name": path.name, "image_path": str(path), "width": width, "height": height, "candidates": []})
        if args.max_images is not None and len(out) >= int(args.max_images):
            break
    return out


def load_label_targets(label_jsonls: Sequence[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    by_image: dict[str, dict[str, dict[str, Any]]] = {}
    for path in label_jsonls:
        if path is None or not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = str(row.get("image_id", "")).strip()
                target_ar = str(row.get("target_ar", "")).strip()
                if not image_id or not target_ar:
                    continue
                routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
                decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
                by_image.setdefault(image_id, {})[target_ar] = {
                    "subject_mode": str(routing.get("subject_mode", "") or "na"),
                    "subject_mode_id": routing.get("subject_mode_id"),
                    "decision_type": str(decision.get("decision_type", "") or "na"),
                    "decision_id": decision.get("decision_id"),
                    "label_source": str(
                        ((row.get("label_generation") if isinstance(row.get("label_generation"), dict) else {}).get("source"))
                        or Path(path).parent.name
                    ),
                }
    return by_image


def summarize_detail(detail: dict[str, Any], *, max_items: int = 5) -> list[str]:
    rows: list[str] = []
    detailed = detail.get("detailed_checklist") if isinstance(detail.get("detailed_checklist"), dict) else {}
    priority = ("subject_coverage", "subject_scale", "headroom", "lookroom", "third_dist", "center_dist", "context", "copyspace", "horizon_state")
    for key in priority:
        payload = detailed.get(key)
        if not isinstance(payload, dict) or payload.get("available") is False:
            continue
        label = str(payload.get("display_label") or payload.get("label") or "")
        if not label or label in {"not_applicable", "unlabeled"}:
            continue
        score = payload.get("score")
        score_txt = "" if score is None else f" {safe_float(score):.2f}"
        rows.append(f"{key}:{label}{score_txt}")
        if len(rows) >= max_items:
            break
    why = detail.get("why_tags") if isinstance(detail.get("why_tags"), list) else []
    if why:
        tags = [str(item.get("tag", "")) for item in why if isinstance(item, dict) and item.get("tag")][:3]
        if tags:
            rows.append("why:" + ",".join(tags))
    return rows


def run_mobilecropnet_for_record(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    record: dict[str, Any],
    target_ars: Sequence[str],
    input_size: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    device: torch.device,
    selection_policy: str,
    proposal_top_m: int,
) -> dict[str, Any]:
    image_path = Path(str(record["image_path"]))
    image_tensor, transform = load_image_tensor_letterbox(image_path, input_size, image_mean=image_mean, image_std=image_std)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_ar = torch.tensor([transform.image_ar_log], dtype=torch.float32, device=device)
    content_box = torch.tensor([letterbox_content_box(transform)], dtype=torch.float32, device=device)
    results: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for target_ar in target_ars:
            outputs = model(
                image_tensor,
                None,
                None,
                torch.tensor([target_ar_id(target_ar)], dtype=torch.long, device=device),
                image_ar,
                None,
                None,
                content_box,
            )
            scores = torch.sigmoid(outputs["utility_logits"][0]).detach().cpu().tolist()
            proposal_scores = torch.sigmoid(outputs["proposal_logits"][0]).detach().cpu().tolist()
            boxes_lb = outputs["proposal_boxes"].detach().cpu()[0].tolist()
            proposal_rows: list[dict[str, Any]] = []
            for idx, padded_box in enumerate(boxes_lb):
                orig_box = letterbox_to_original_box(padded_box, transform)
                ar_val = crop_ar(orig_box, transform.original_width, transform.original_height)
                target_val = {"FREE": None, "1:1": 1.0, "3:4": 3.0 / 4.0, "4:3": 4.0 / 3.0, "16:9": 16.0 / 9.0, "9:16": 9.0 / 16.0}[target_ar]
                ar_error = 0.0 if target_val is None else abs(math.log(max(1e-6, ar_val) / max(1e-6, target_val)))
                proposal_rows.append(
                    {
                        "bbox_norm_xyxy": orig_box,
                        "target_ar_compatible": bool(target_val is None or ar_error <= 0.08),
                        "target_ar_log_error": ar_error,
                    }
                )
            top_idx = select_generated_proposal_index(
                utility_scores=scores,
                proposal_scores=proposal_scores,
                proposals=proposal_rows,
                selection_policy=selection_policy,
                proposal_top_m=proposal_top_m,
            )
            top_box = letterbox_to_original_box(boxes_lb[top_idx], transform)
            positives = torch.sigmoid(outputs["positive_logits"][0]).detach().cpu().tolist()
            risks = torch.sigmoid(outputs["risk_logits"][0]).detach().cpu().tolist()
            route_id = int(outputs["route_logits"].argmax(dim=1).detach().cpu()[0].item())
            decision_id = int(outputs["decision_logits"].argmax(dim=1).detach().cpu()[0].item())
            route_label = class_label(SUBJECT_MODE_VOCAB, route_id)
            detail = detailed_explanation_from_logits(
                checklist_logits=outputs["checklist_class_logits"].detach().cpu()[0][top_idx] if "checklist_class_logits" in outputs else None,
                checklist_applicability_logits=outputs["checklist_applicability_logits"].detach().cpu()[0][top_idx] if "checklist_applicability_logits" in outputs else None,
                detail_score_logits=outputs["detail_score_logits"].detach().cpu()[0][top_idx] if "detail_score_logits" in outputs else None,
                why_tag_logits=outputs["why_tag_logits"].detach().cpu()[0][top_idx] if "why_tag_logits" in outputs else None,
                subject_mode_label=route_label,
            )
            results[target_ar] = {
                "model": "MobileCropNet v4 public-score distill",
                "checkpoint": str(checkpoint_path),
                "target_ar": target_ar,
                "selection_source": selection_policy,
                "bbox_norm_xyxy": top_box,
                "score": float(scores[top_idx]),
                "positive_score": float(positives[top_idx]),
                "risk_score": float(risks[top_idx]),
                "proposal_score": float(proposal_scores[top_idx]),
                "decision_label": class_label(DECISION_VOCAB, decision_id),
                "subject_mode_label": route_label,
                "detail_summary": summarize_detail(detail),
                "detailed_checklist": detail.get("detailed_checklist", {}),
                "why_tags": detail.get("why_tags", []),
            }
    return results


def public_scores_for_records(records: list[dict[str, Any]], device: torch.device) -> dict[str, dict[str, Any]]:
    scores_by_image, coverage_by_image, metadata = _score_gaic(records, device=device)
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        candidates = list(record.get("candidates") or [])
        scores = list(scores_by_image.get(image_id) or [])
        if not candidates or len(scores) != len(candidates):
            continue
        ranked = order_desc(scores)
        top_idx = ranked[0]
        top = candidates[top_idx]
        payload = {
            "method": "GAIC public cropper",
            "bbox_norm_xyxy": top.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]),
            "score": float(scores[top_idx]),
            "candidate_index": int(top_idx),
            "candidate_id": top.get("candidate_id"),
            "annotation_id": top.get("annotation_id"),
            "source": top.get("source", "candidate"),
            "coverage": coverage_by_image.get(image_id, {}),
        }
        if "mos" in top:
            mos = [safe_float(c.get("mos"), 0.0) for c in candidates]
            mos_rank = {idx: rank for rank, idx in enumerate(order_desc(mos), 1)}
            payload["mos"] = safe_float(top.get("mos"), 0.0)
            payload["best_mos"] = max(mos) if mos else 0.0
            payload["mos_rank"] = mos_rank.get(top_idx, len(mos))
        out[image_id] = payload
    out["_metadata"] = metadata
    return out


def build_public_records_from_ar_bank(records: list[dict[str, Any]], mcn_rows: list[dict[str, Any]], target_ars: Sequence[str]) -> list[dict[str, Any]]:
    by_id = {str(row["image_id"]): row for row in mcn_rows}
    out: list[dict[str, Any]] = []
    for record in records:
        image_id = str(record.get("image_id", ""))
        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
        candidates: list[dict[str, Any]] = []
        mcn = by_id.get(image_id, {}).get("mobilecropnet", {})
        for target_ar in target_ars:
            if target_ar in mcn:
                candidates.append(
                    {
                        "candidate_id": f"mcn_{target_ar}",
                        "source": "mcn_ar_proposal",
                        "target_ar": target_ar,
                        "bbox_norm_xyxy": mcn[target_ar]["bbox_norm_xyxy"],
                    }
                )
            candidates.append(
                {
                    "candidate_id": f"center_{target_ar}",
                    "source": "center_ar_baseline",
                    "target_ar": target_ar,
                    "bbox_norm_xyxy": center_crop_for_ar(target_ar, width, height),
                }
            )
        out.append({**record, "candidates": candidates})
    return out


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    tw = int(draw.textlength(text, font=font))
    th = int(getattr(font, "size", 16) * 1.35)
    draw.rectangle([x, max(0, y - th - 4), x + tw + 12, y], fill=color)
    draw.text((x + 6, max(0, y - th - 2)), text, fill=(255, 255, 255), font=font)


def panel_line(draw: ImageDraw.ImageDraw, *, x: int, y: int, text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], max_width: int) -> int:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines or [""]:
        draw.text((x, y), line, fill=fill, font=font)
        y += int(getattr(font, "size", 16) * 1.35) + 2
    return y


def render_overlay(row: dict[str, Any], output_path: Path, *, max_side: int) -> None:
    image_path = Path(str(row["image_path"]))
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(image.size)))
    if scale < 1.0:
        image = image.resize((int(round(image.width * scale)), int(round(image.height * scale))), Image.BILINEAR)
    panel_w = 820
    canvas_h = max(image.height, 980)
    canvas = Image.new("RGB", (image.width + panel_w, canvas_h), (18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(18, min(28, image.width // 48)))
    small = load_font(max(14, min(20, image.width // 70)))
    line_w = max(3, image.width // 360)
    mcn = row.get("mobilecropnet", {})
    label_targets = row.get("label_targets") if isinstance(row.get("label_targets"), dict) else {}
    for target_ar, payload in mcn.items():
        color = AR_COLORS.get(target_ar, (200, 200, 200))
        xy = norm_to_px(payload.get("bbox_norm_xyxy", [0, 0, 1, 1]), image.width, image.height)
        draw.rectangle(xy, outline=color, width=line_w)
        draw_label(draw, (xy[0], xy[1]), f"MCN final {target_ar} {safe_float(payload.get('score')):.2f}", color, small)
    public = row.get("public_cropper")
    if isinstance(public, dict):
        xy = norm_to_px(public.get("bbox_norm_xyxy", [0, 0, 1, 1]), image.width, image.height)
        draw.rectangle(xy, outline=PUBLIC_COLOR, width=line_w + 1)
        draw_label(draw, (xy[0], max(0, xy[1] + 28)), f"GAIC public {safe_float(public.get('score')):.2f}", PUBLIC_COLOR, small)
    gt = row.get("gaic_best_mos")
    if isinstance(gt, dict):
        xy = norm_to_px(gt.get("bbox_norm_xyxy", [0, 0, 1, 1]), image.width, image.height)
        draw.rectangle(xy, outline=GT_COLOR, width=max(2, line_w - 1))
        draw_label(draw, (xy[0], max(0, xy[1] + 56)), f"GT MOS {safe_float(gt.get('mos')):.2f}", GT_COLOR, small)
    px = image.width
    pad = 24
    draw.rectangle([px, 0, canvas.width, canvas.height], fill=(22, 24, 27))
    y = pad
    y = panel_line(draw, x=px + pad, y=y, text=f"Deployed final crop vs GAIC public cropper | {row.get('image_id', '')}", font=font, fill=(255, 255, 255), max_width=panel_w - 2 * pad)
    y += 8
    y = panel_line(draw, x=px + pad, y=y, text="Legend: colored boxes are deployed final crops chosen from generated proposals for each target AR. Yellow is GAIC public cropper top crop. White is highest MOS annotation when available.", font=small, fill=(210, 210, 210), max_width=panel_w - 2 * pad)
    y += 10
    if label_targets:
        label_sources = sorted({str(payload.get("label_source", "")).strip() for payload in label_targets.values() if isinstance(payload, dict) and str(payload.get("label_source", "")).strip()})
        label_source_txt = ", ".join(label_sources) if label_sources else "attached"
        y = panel_line(draw, x=px + pad, y=y, text=f"Label sidecar: {label_source_txt}", font=small, fill=(170, 210, 170), max_width=panel_w - 2 * pad)
        y += 10
    if isinstance(public, dict):
        public_extra = ""
        if "mos" in public:
            public_extra = f" mos={safe_float(public.get('mos')):.2f} best={safe_float(public.get('best_mos')):.2f} rank={public.get('mos_rank')}"
        y = panel_line(draw, x=px + pad, y=y, text=f"GAIC public: score={safe_float(public.get('score')):.3f}{public_extra} source={public.get('source', '')}", font=small, fill=PUBLIC_COLOR, max_width=panel_w - 2 * pad)
        y += 10
    for target_ar, payload in mcn.items():
        color = AR_COLORS.get(target_ar, (230, 230, 230))
        draw.rectangle([px + pad, y + 4, px + pad + 18, y + 22], fill=color)
        ar_value = crop_ar(payload.get("bbox_norm_xyxy", [0, 0, 1, 1]), int(row.get("width", image.width)), int(row.get("height", image.height)))
        y = panel_line(
            draw,
            x=px + pad + 30,
            y=y,
            text=f"{target_ar}: final_score={safe_float(payload.get('score')):.3f} pos={safe_float(payload.get('positive_score')):.2f} risk={safe_float(payload.get('risk_score')):.2f} crop_ar={ar_value:.2f}",
            font=small,
            fill=(235, 235, 235),
            max_width=panel_w - 2 * pad - 30,
        )
        detail_rows = list(payload.get("detail_summary") or [])
        for detail in detail_rows[:3]:
            y = panel_line(draw, x=px + pad + 30, y=y, text=f"- {detail}", font=small, fill=(190, 210, 235), max_width=panel_w - 2 * pad - 30)
        y = panel_line(
            draw,
            x=px + pad + 30,
            y=y,
            text=f"model: decision={payload.get('decision_label')} subject_mode={payload.get('subject_mode_label')}",
            font=small,
            fill=(190, 235, 190),
            max_width=panel_w - 2 * pad - 30,
        )
        label_payload = label_targets.get(target_ar) if isinstance(label_targets.get(target_ar), dict) else None
        if label_payload is not None:
            y = panel_line(
                draw,
                x=px + pad + 30,
                y=y,
                text=f"label: decision={label_payload.get('decision_type')} subject_mode={label_payload.get('subject_mode')}",
                font=small,
                fill=(235, 205, 150),
                max_width=panel_w - 2 * pad - 30,
            )
        y += 8
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
        thumb.thumbnail((520, 360), Image.BILINEAR)
        tile = Image.new("RGB", (520, 360), (14, 14, 14))
        tile.paste(thumb, ((520 - thumb.width) // 2, (360 - thumb.height) // 2))
        thumbs.append(tile)
    cols = min(3, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 520, rows * 360), (8, 8, 8))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 520, (idx // cols) * 360))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=3)


def best_mos_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    candidates = list(record.get("candidates") or [])
    if not candidates:
        return None
    best = max(candidates, key=lambda c: safe_float(c.get("mos"), 0.0))
    return {"bbox_norm_xyxy": best.get("bbox_norm_xyxy"), "mos": safe_float(best.get("mos"), 0.0), "annotation_id": best.get("annotation_id")}


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]], public_metadata: dict[str, Any]) -> None:
    report_path = args.output_dir / "BEST_MODEL_PUBLIC_CROPPER_AR_COMPARISON_REPORT.md"
    dataset = "GAICv2 official test candidates" if args.annotations_json else "data/TestImages proposal-bank comparison"
    public_note = (
        "GAIC public cropper scoring was skipped for this Product AR inference visualization."
        if args.skip_public_cropper
        else "GAIC public cropper top crop is included as a yellow reference box."
    )
    lines = [
        "# MobileCropNet v4 Product AR Inference Visualization",
        "",
        f"- dataset: `{dataset}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- target ARs: `{', '.join(args.target_ars)}`",
        f"- image_count: `{len(rows)}`",
        f"- contact sheet: `{args.output_dir / 'contact_sheet.png'}`",
        f"- public cropper: `{public_note}`",
        f"- label sidecar: `{', '.join(str(path) for path in args.label_jsonl) if args.label_jsonl else 'none'}`",
        "",
        "## Interpretation",
        "",
        f"- MobileCropNet v4 배포 추론은 외부 candidate bank를 쓰지 않고, 내부 proposal generator가 만든 후보들 중 `{args.selection_policy}` 정책으로 최종 crop을 고른다.",
        f"- `proposal_topk_rerank`는 proposal objectness top-{int(args.proposal_top_m)} shortlist 안에서 utility top-1을 선택한다.",
        "- 본 시각화의 colored box는 각 target AR에 대해 사용자가 실제 보게 되는 deployed final crop이다.",
        "- 각 AR crop에는 final utility/positive/risk score와 detailed checklist가 붙고, 우측 패널에는 `model decision/subject mode`가 출력된다.",
        "- Product AR label JSONL을 함께 넘기면, 같은 `(image_id, target_ar)`에 대한 `label decision/subject mode`도 같이 표시한다.",
        "- GAIC public cropper는 선택적으로만 표시한다. Product AR cropper 품질 판단의 핵심은 AR별 crop geometry와 explanation이다.",
        "- `data/TestImages`에는 official candidate와 MOS annotation이 없으므로 정량 판단이 아니라 정성 sanity check로 해석한다.",
        "",
        "## Visual Legend",
        "",
        "| color | meaning |",
        "| --- | --- |",
        "| red | deployed final `FREE` crop |",
        "| green | deployed final `1:1` crop |",
        "| blue | deployed final `3:4` crop |",
        "| orange | deployed final `4:3` crop |",
        "| purple | deployed final `16:9` crop |",
        "| cyan | deployed final `9:16` crop |",
        "| yellow | GAIC public cropper top crop, only when public scoring is enabled |",
        "| white | GAIC highest MOS crop, GAICv2 only |",
        "",
        "## Public Cropper Metadata",
        "",
        "```json",
        json.dumps(public_metadata, ensure_ascii=False, indent=2)[:4000],
        "```",
        "",
        "## Rows",
        "",
        "| image_id | overlay | public source | FREE score | 1:1 score | 3:4 score | 4:3 score | 16:9 score | 9:16 score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        mcn = row.get("mobilecropnet", {})
        public = row.get("public_cropper")
        if not isinstance(public, dict):
            public = {}
        scores = [safe_float(mcn.get(ar, {}).get("score"), 0.0) for ar in args.target_ars]
        lines.append(
            f"| `{row.get('image_id')}` | `{row.get('overlay')}` | `{public.get('source', '')}` | "
            + " | ".join(f"{score:.3f}" for score in scores)
            + " |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MobileCropNet v4 best model AR outputs against GAIC public cropper.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--annotations_json", type=Path, default=None)
    parser.add_argument("--image_roots", nargs="*", type=Path, default=[])
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--image_glob", action="append", default=[])
    parser.add_argument("--label_jsonl", action="append", type=Path, default=[], help="Optional Product AR label JSONL sidecar(s) for panel comparison.")
    parser.add_argument("--target_ars", nargs="+", default=["FREE", "1:1", "3:4", "4:3", "16:9", "9:16"], choices=TARGET_AR_VOCAB)
    parser.add_argument("--max_images", type=int, default=24)
    parser.add_argument("--max_side", type=int, default=1500)
    parser.add_argument("--skip_public_cropper", action="store_true", help="Skip CUDA-only GAIC public cropper scoring and render MobileCropNet/GT boxes only.")
    parser.add_argument("--selection_policy", choices=["utility_top1", "proposal_top1", "proposal_topk_rerank"], default="utility_top1")
    parser.add_argument("--proposal_top_m", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.checkpoint, device=device)
    input_size = int(ckpt.get("train_config", {}).get("input_size", 320))
    image_mean, image_std = infer_image_norm(ckpt)
    records = collect_records(args)
    label_targets_by_image = load_label_targets(args.label_jsonl)
    rows: list[dict[str, Any]] = []
    for record in records[: int(args.max_images)]:
        image_id = str(record.get("image_id", ""))
        mcn = run_mobilecropnet_for_record(
            model=model,
            checkpoint_path=args.checkpoint,
            record=record,
            target_ars=args.target_ars,
            input_size=input_size,
            image_mean=image_mean,
            image_std=image_std,
            device=device,
            selection_policy=args.selection_policy,
            proposal_top_m=int(args.proposal_top_m),
        )
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(record.get("image_path", "")),
                "width": int(record.get("width", 0)),
                "height": int(record.get("height", 0)),
                "mobilecropnet": mcn,
                "gaic_best_mos": best_mos_payload(record),
                "label_targets": label_targets_by_image.get(image_id, {}),
            }
        )
    if args.skip_public_cropper:
        public_by_image = {"_metadata": {"skipped": True, "reason": "skip_public_cropper"}}
    elif args.annotations_json is not None:
        public_records = records[: int(args.max_images)]
        public_by_image = public_scores_for_records(public_records, device=device)
    else:
        public_records = build_public_records_from_ar_bank(records[: int(args.max_images)], rows, args.target_ars)
        public_by_image = public_scores_for_records(public_records, device=device)
    public_metadata = public_by_image.pop("_metadata", {})
    overlay_paths: list[Path] = []
    for idx, row in enumerate(rows, 1):
        row["public_cropper"] = public_by_image.get(str(row.get("image_id", "")))
        output_path = args.output_dir / "overlays" / f"{idx:04d}_{row.get('image_id', idx)}.png"
        row["overlay"] = str(output_path)
        render_overlay(row, output_path, max_side=int(args.max_side))
        overlay_paths.append(output_path)
    make_contact_sheet(overlay_paths, args.output_dir / "contact_sheet.png")
    manifest = {
        "checkpoint": str(args.checkpoint),
        "target_ars": list(args.target_ars),
        "selection_policy": args.selection_policy,
        "proposal_top_m": int(args.proposal_top_m),
        "image_count": len(rows),
        "contact_sheet": str(args.output_dir / "contact_sheet.png"),
        "public_cropper_metadata": public_metadata,
        "rows": rows,
    }
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args, rows, public_metadata)
    print(json.dumps({"status": "ok", "image_count": len(rows), "contact_sheet": str(args.output_dir / "contact_sheet.png")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
