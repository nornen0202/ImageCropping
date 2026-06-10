#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mobilecropnet_v4.eval_utils import infer_image_norm, load_mobilecropnet_v4_checkpoint  # noqa: E402
from scripts.visualize_mobilecropnet_v4_public_ar_comparison import (  # noqa: E402
    run_mobilecropnet_for_record,
    safe_float,
)

TRAINING_ASSET_DIR = PROJECT_ROOT / "Implement_Docs/assets_mobilecropnet_training_data_master_20260424"
V4_ASSET_DIR = PROJECT_ROOT / "Implement_Docs/assets_mobilecropnet_v4_master_20260427"
LABEL_JSONL = (
    PROJECT_ROOT
    / "data/SSTK/Full_10000/artifacts/training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422/test/"
    / "train_conditional_detr_batch.jsonl"
)
JOINT_MANIFEST = V4_ASSET_DIR / "joint_head_inference_gallery_20260427/fig13_joint_head_inference_gallery_manifest.json"
MCN_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_public_subjectprior_spot/"
    / "mcn-sstk-public-subjasync-r5bal-balanced-288-20260423-034145/best.pt"
)

TARGET_ARS = ("FREE", "1:1", "3:4", "4:3", "16:9", "9:16")
TEACHER_MODE_QUOTA = {
    "portrait_single": 2,
    "object_single": 2,
    "object_multi": 2,
    "scene_general": 2,
    "background_texture_copyspace": 2,
}
MODE_COLORS = {
    "portrait_single": (205, 70, 80),
    "portrait_group": (190, 90, 120),
    "object_single": (45, 145, 90),
    "object_multi": (42, 120, 198),
    "scene_general": (210, 130, 35),
    "background_texture_copyspace": (110, 95, 185),
}
CROP_COLOR = (232, 54, 54)
SUBJECT_COLOR = (20, 190, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher/MCN crop-head success galleries for report docs.")
    parser.add_argument("--label_jsonl", type=Path, default=LABEL_JSONL)
    parser.add_argument("--joint_manifest", type=Path, default=JOINT_MANIFEST)
    parser.add_argument("--mcn_checkpoint", type=Path, default=MCN_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selection_policy", choices=["utility_top1", "proposal_top1", "proposal_topk_rerank"], default="proposal_topk_rerank")
    parser.add_argument("--proposal_top_m", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT) if path.is_absolute() else path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    vals = [clamp01(safe_float(v)) for v in box[:4]]
    return (
        int(round(vals[0] * width)),
        int(round(vals[1] * height)),
        int(round(vals[2] * width)),
        int(round(vals[3] * height)),
    )


def valid_box(box: Any) -> list[float] | None:
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        vals = [clamp01(safe_float(v)) for v in box[:4]]
        if vals[2] > vals[0] and vals[3] > vals[1]:
            return vals
    return None


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = (235, 238, 242)) -> tuple[Image.Image, float, tuple[int, int]]:
    out = Image.new("RGB", size, bg)
    work = image.copy()
    work.thumbnail(size, Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
    offset = ((size[0] - work.width) // 2, (size[1] - work.height) // 2)
    out.paste(work, offset)
    scale = work.width / float(max(1, image.width))
    return out, scale, offset


def crop_norm(image: Image.Image, box: Sequence[float]) -> Image.Image:
    x1, y1, x2, y2 = norm_to_px(box, image.width, image.height)
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width, x2))
    y2 = max(y1 + 1, min(image.height, y2))
    return image.crop((x1, y1, x2, y2))


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], max_width: int) -> int:
    x, y = xy
    line = ""
    lines: list[str] = []
    for word in str(text).split():
        candidate = word if not line else f"{line} {word}"
        if draw.textlength(candidate, font=fnt) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    line_h = int(getattr(fnt, "size", 13) * 1.28)
    for item in lines[:7]:
        draw.text((x, y), item, fill=fill, font=fnt)
        y += line_h
    return y


def draw_box_on_fitted(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    *,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    x1, y1, x2, y2 = norm_to_px(box, original_size[0], original_size[1])
    mapped = (
        offset[0] + int(round(x1 * scale)),
        offset[1] + int(round(y1 * scale)),
        offset[0] + int(round(x2 * scale)),
        offset[1] + int(round(y2 * scale)),
    )
    draw.rectangle(mapped, outline=color, width=width)


def candidate_score(candidate: dict[str, Any]) -> float:
    scores = candidate.get("score_targets") if isinstance(candidate.get("score_targets"), dict) else {}
    macro = candidate.get("macro_targets") if isinstance(candidate.get("macro_targets"), dict) else {}
    penalty = candidate.get("safety_penalty") if isinstance(candidate.get("safety_penalty"), dict) else {}
    base = safe_float(scores.get("crop_utility_prob", scores.get("score_prob", candidate.get("crop_utility_prob"))))
    base += 0.15 * safe_float(macro.get("S_macro"))
    base += 0.10 * safe_float(macro.get("C_macro"))
    base -= 0.20 * safe_float(penalty.get("total"))
    return base


def best_teacher_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for section in ("matching_targets", "candidate_pool"):
        for candidate in row.get(section) or []:
            if isinstance(candidate, dict) and valid_box(candidate.get("bbox_norm_xyxy")) is not None:
                if section == "matching_targets" or candidate.get("is_positive_candidate") is True:
                    candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=candidate_score)


def teacher_row_score(row: dict[str, Any], candidate: dict[str, Any]) -> float:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    teacher = row.get("teacher_meta") if isinstance(row.get("teacher_meta"), dict) else {}
    label_gen = row.get("label_generation") if isinstance(row.get("label_generation"), dict) else {}
    score = candidate_score(candidate)
    score += 0.10 * safe_float(routing.get("route_conf"))
    score += 0.05 * safe_float(teacher.get("teacher_confidence"))
    score += 0.05 if label_gen.get("positive_reason") in {"external_best_safe", "t1_best_safe", "uctr_best_safe"} else 0.0
    return score


def select_teacher_samples(label_jsonl: Path) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_images: set[str] = set()
    for row in read_jsonl(label_jsonl):
        image_id = str(row.get("image_id", ""))
        image_path = PROJECT_ROOT / str(row.get("image_path", ""))
        if not image_id or not image_path.exists():
            continue
        routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
        mode = str(routing.get("subject_mode", ""))
        if mode not in TEACHER_MODE_QUOTA:
            continue
        candidate = best_teacher_candidate(row)
        if candidate is None:
            continue
        checklist = candidate.get("checklist_labels") if isinstance(candidate.get("checklist_labels"), dict) else {}
        if checklist.get("subject_coverage") in {"poor", "very_poor"}:
            continue
        payload = {
            "image_id": image_id,
            "image_path": str(image_path),
            "target_ar": str(row.get("target_ar", "")),
            "subject_mode": mode,
            "decision": str((row.get("decision_target") or {}).get("decision_type", "unknown")),
            "crop_box": valid_box(candidate.get("bbox_norm_xyxy")),
            "subject_box": valid_box(routing.get("subject_prior_bbox_norm_xyxy")),
            "checklist": checklist,
            "why_tags": [str(tag) for tag in (candidate.get("why_tags") or [])[:5]],
            "score": teacher_row_score(row, candidate),
            "candidate_id": candidate.get("candidate_id"),
        }
        if payload["crop_box"] is None:
            continue
        by_mode[mode].append(payload)
    selected: list[dict[str, Any]] = []
    for mode, quota in TEACHER_MODE_QUOTA.items():
        rows = sorted(by_mode.get(mode, []), key=lambda item: item["score"], reverse=True)
        picked = 0
        for row in rows:
            if row["image_id"] in seen_images:
                continue
            selected.append(row)
            seen_images.add(row["image_id"])
            picked += 1
            if picked >= quota:
                break
    return selected


def select_mcn_samples(joint_manifest: Path, *, max_samples: int = 10) -> list[dict[str, Any]]:
    manifest = json.loads(joint_manifest.read_text(encoding="utf-8"))
    rows = list(manifest.get("samples") or [])[:max_samples]
    selected: list[dict[str, Any]] = []
    for sample in rows:
        crops = [crop for crop in (sample.get("crop_results") or []) if isinstance(crop, dict)]
        if not crops:
            continue
        best = max(crops, key=lambda crop: safe_float(crop.get("final_iou_to_best_positive")))
        selected.append(
            {
                "image_id": str(sample.get("image_id", "")),
                "image_path": str(PROJECT_ROOT / str(sample.get("image_path", ""))),
                "target_ar": str(best.get("target_ar", "FREE")),
                "teacher_route": str(sample.get("teacher_route", "")),
                "model_route": str(best.get("model_route", sample.get("model_route_majority", ""))),
                "teacher_decision": str(best.get("teacher_decision", "")),
                "model_decision": str(best.get("model_decision", "")),
                "crop_box": valid_box(best.get("bbox_norm_xyxy")),
                "subject_box": valid_box((sample.get("teacher_subject_box") or {}).get("bbox_norm_xyxy")),
                "iou": safe_float(best.get("final_iou_to_best_positive")),
                "score": safe_float(best.get("utility_score")),
                "proposal_score": safe_float(best.get("proposal_score")),
            }
        )
    return selected


def attach_mcn_detail(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.mcn_checkpoint, device=device)
    input_size = int(ckpt.get("train_config", {}).get("input_size", 320))
    image_mean, image_std = infer_image_norm(ckpt)
    for row in selected:
        image = Image.open(row["image_path"])
        record = {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "width": image.width,
            "height": image.height,
        }
        decoded = run_mobilecropnet_for_record(
            model=model,
            checkpoint_path=args.mcn_checkpoint,
            record=record,
            target_ars=[row["target_ar"]],
            input_size=input_size,
            image_mean=image_mean,
            image_std=image_std,
            device=device,
            selection_policy=args.selection_policy,
            proposal_top_m=int(args.proposal_top_m),
        )
        payload = decoded.get(row["target_ar"], {})
        row["model_route_decode"] = payload.get("subject_mode_label", row.get("model_route"))
        row["model_decision_decode"] = payload.get("decision_label", row.get("model_decision"))
        row["detail_summary"] = list(payload.get("detail_summary") or [])[:4]
        row["why_tags"] = [str(item.get("tag")) for item in (payload.get("why_tags") or []) if isinstance(item, dict) and item.get("tag")][:5]
        if row.get("crop_box") is None:
            row["crop_box"] = valid_box(payload.get("bbox_norm_xyxy"))


def checklist_summary(checklist: dict[str, Any]) -> list[str]:
    keys = ("subject_coverage", "subject_scale", "headroom", "lookroom", "context", "copyspace")
    rows = []
    for key in keys:
        value = checklist.get(key)
        if value not in (None, "", "na"):
            rows.append(f"{key}:{value}")
    return rows[:4]


def render_card(sample: dict[str, Any], *, kind: str, size: tuple[int, int]) -> Image.Image:
    card = Image.new("RGB", size, (248, 249, 251))
    draw = ImageDraw.Draw(card)
    header_h = 34
    mode = str(sample.get("subject_mode") or sample.get("teacher_route") or sample.get("model_route") or "unknown")
    color = MODE_COLORS.get(mode, (86, 100, 116))
    draw.rectangle([0, 0, size[0], header_h], fill=color)
    title = f"{kind} | {mode} | {sample.get('image_id', '')}"
    draw.text((12, 8), title[:72], fill=(255, 255, 255), font=font(14, bold=True))

    image = Image.open(sample["image_path"]).convert("RGB")
    original_box = (12, 48, 252, 226)
    crop_box = (264, 48, 504, 226)
    orig_tile, scale, offset = fit_image(image, (original_box[2] - original_box[0], original_box[3] - original_box[1]), bg=(230, 233, 237))
    card.paste(orig_tile, (original_box[0], original_box[1]))
    odraw = ImageDraw.Draw(card)
    adjusted_offset = (original_box[0] + offset[0], original_box[1] + offset[1])
    if sample.get("crop_box"):
        draw_box_on_fitted(odraw, sample["crop_box"], original_size=image.size, scale=scale, offset=adjusted_offset, color=CROP_COLOR, width=3)
    if sample.get("subject_box"):
        draw_box_on_fitted(odraw, sample["subject_box"], original_size=image.size, scale=scale, offset=adjusted_offset, color=SUBJECT_COLOR, width=2)

    crop = crop_norm(image, sample.get("crop_box") or [0.0, 0.0, 1.0, 1.0])
    crop_tile, _, _ = fit_image(crop, (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), bg=(20, 22, 24))
    card.paste(crop_tile, (crop_box[0], crop_box[1]))
    draw.rectangle([original_box[0], original_box[1], original_box[2], original_box[3]], outline=(190, 198, 207), width=1)
    draw.rectangle([crop_box[0], crop_box[1], crop_box[2], crop_box[3]], outline=(190, 198, 207), width=1)
    draw.text((original_box[0], original_box[3] + 4), "original + crop/subject boxes", fill=(70, 78, 88), font=font(10))
    draw.text((crop_box[0], crop_box[3] + 4), "selected crop", fill=(70, 78, 88), font=font(10))

    x = 518
    y = 48
    text_font = font(12)
    small = font(10)
    if kind == "Teacher":
        y = draw_wrapped(draw, (x, y), f"AR {sample.get('target_ar')} | decision {sample.get('decision')} | score {safe_float(sample.get('score')):.3f}", text_font, (25, 35, 48), size[0] - x - 12)
        checks = checklist_summary(sample.get("checklist") or {})
        if checks:
            y = draw_wrapped(draw, (x, y + 6), "check: " + "; ".join(checks), small, (58, 72, 88), size[0] - x - 12)
        if sample.get("why_tags"):
            y = draw_wrapped(draw, (x, y + 6), "why: " + ", ".join(sample["why_tags"][:5]), small, (62, 78, 118), size[0] - x - 12)
        y = draw_wrapped(draw, (x, y + 6), f"candidate: {sample.get('candidate_id')}", small, (88, 94, 102), size[0] - x - 12)
    else:
        y = draw_wrapped(
            draw,
            (x, y),
            f"AR {sample.get('target_ar')} | route T/M {sample.get('teacher_route')}/{sample.get('model_route_decode', sample.get('model_route'))}",
            text_font,
            (25, 35, 48),
            size[0] - x - 12,
        )
        y = draw_wrapped(
            draw,
            (x, y + 4),
            f"decision T/M {sample.get('teacher_decision')}/{sample.get('model_decision_decode', sample.get('model_decision'))}",
            text_font,
            (25, 35, 48),
            size[0] - x - 12,
        )
        y = draw_wrapped(draw, (x, y + 4), f"score {safe_float(sample.get('score')):.3f} | IoU {safe_float(sample.get('iou')):.3f}", text_font, (25, 35, 48), size[0] - x - 12)
        if sample.get("detail_summary"):
            y = draw_wrapped(draw, (x, y + 6), "check: " + "; ".join(sample["detail_summary"][:4]), small, (58, 72, 88), size[0] - x - 12)
        if sample.get("why_tags"):
            y = draw_wrapped(draw, (x, y + 6), "why: " + ", ".join(sample["why_tags"][:5]), small, (62, 78, 118), size[0] - x - 12)
    return card


def render_gallery(samples: list[dict[str, Any]], *, kind: str, output_path: Path) -> None:
    card_w, card_h = 880, 250
    cols = 2
    rows = math.ceil(len(samples) / cols)
    header_h = 78
    pad = 18
    width = cols * card_w + (cols + 1) * pad
    height = header_h + rows * card_h + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), (235, 238, 242))
    draw = ImageDraw.Draw(canvas)
    title = "Teacher Crop + Head Success Gallery" if kind == "Teacher" else "MobileCropNet Crop + Head Success Gallery"
    subtitle = (
        "10 representative label rows with crop, subject mode, checklist, and why-tags"
        if kind == "Teacher"
        else "10 representative inference rows with crop, route/decision, checklist decode, and why-tags"
    )
    draw.text((pad, 18), title, fill=(18, 24, 32), font=font(25, bold=True))
    draw.text((pad, 50), subtitle, fill=(80, 88, 98), font=font(13))
    for idx, sample in enumerate(samples):
        col = idx % cols
        row = idx // cols
        x = pad + col * (card_w + pad)
        y = header_h + pad + row * (card_h + pad)
        canvas.paste(render_card(sample, kind=kind, size=(card_w, card_h)), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def copy_to_dirs(source: Path, target_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root_name, root in (("training", TRAINING_ASSET_DIR), ("v4", V4_ASSET_DIR)):
        dest = root / target_name
        if dest != source:
            shutil.copy2(source, dest)
        out[root_name] = {"path": rel(dest), "bytes": dest.stat().st_size}
    return out


def main() -> int:
    args = parse_args()
    started = time.time()
    teacher_samples = select_teacher_samples(args.label_jsonl)
    if len(teacher_samples) < 10:
        raise RuntimeError(f"teacher sample selection returned only {len(teacher_samples)} rows")
    teacher_samples = teacher_samples[:10]

    mcn_samples = select_mcn_samples(args.joint_manifest, max_samples=10)
    attach_mcn_detail(args, mcn_samples)

    teacher_v4 = V4_ASSET_DIR / "fig16_teacher_crop_head_success_gallery.png"
    mcn_v4 = V4_ASSET_DIR / "fig16_mcn_crop_head_success_gallery.png"
    render_gallery(teacher_samples, kind="Teacher", output_path=teacher_v4)
    render_gallery(mcn_samples, kind="MCN", output_path=mcn_v4)
    outputs = {
        "teacher": copy_to_dirs(teacher_v4, "fig_teacher_crop_head_success_gallery.png"),
        "mcn": copy_to_dirs(mcn_v4, "fig_mcn_crop_head_success_gallery.png"),
    }
    # Keep v4-specific aliases for section numbering.
    outputs["teacher"]["v4_alias"] = {"path": rel(teacher_v4), "bytes": teacher_v4.stat().st_size}
    outputs["mcn"]["v4_alias"] = {"path": rel(mcn_v4), "bytes": mcn_v4.stat().st_size}

    manifest = {
        "generated_at_kst": time.strftime("%Y-%m-%d %H:%M:%S KST", time.localtime()),
        "label_jsonl": rel(args.label_jsonl),
        "joint_manifest": rel(args.joint_manifest),
        "mcn_checkpoint": rel(args.mcn_checkpoint),
        "teacher_sample_count": len(teacher_samples),
        "mcn_sample_count": len(mcn_samples),
        "teacher_mode_counts": dict(Counter(sample["subject_mode"] for sample in teacher_samples)),
        "mcn_teacher_route_counts": dict(Counter(sample["teacher_route"] for sample in mcn_samples)),
        "outputs": outputs,
        "teacher_samples": teacher_samples,
        "mcn_samples": mcn_samples,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(V4_ASSET_DIR / "fig16_crop_head_success_gallery_manifest.json", manifest)
    write_json(TRAINING_ASSET_DIR / "fig_crop_head_success_gallery_manifest.json", manifest)
    print(json.dumps({"status": "ok", "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
