from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

from public_benchmark.staging import write_jsonl


COLORS = {
    "gt": (20, 170, 70),
    "top1": (220, 45, 45),
    "top1_pre": (245, 160, 25),
    "text": (255, 255, 255),
    "shadow": (0, 0, 0),
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:160] or "item"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_failure(row: dict[str, Any]) -> tuple[list[str], float]:
    reasons: list[str] = []
    severity = 0.0
    if _as_float(row.get("schema_issue_count")) > 0:
        reasons.append("schema_issue")
        severity += 10.0
    if _as_float(row.get("nan_score_count")) > 0:
        reasons.append("nan_score")
        severity += 8.0
    if row.get("coverage_at_07_preinject") is not None and _as_float(row.get("coverage_at_07_preinject")) < 1.0:
        reasons.append("candidate_recall_lt_0.7")
        severity += 6.0 + (1.0 - _as_float(row.get("coverage_at_07_preinject")))
    if row.get("iou_top1_preinject") is not None and _as_float(row.get("iou_top1_preinject")) < 0.7:
        reasons.append("top1_preinject_iou_lt_0.7")
        severity += 4.0 + (0.7 - _as_float(row.get("iou_top1_preinject")))
    if row.get("iou_top1") is not None and _as_float(row.get("iou_top1")) < 0.7:
        reasons.append("top1_iou_lt_0.7")
        severity += 3.0 + (0.7 - _as_float(row.get("iou_top1")))
    if row.get("gt_rank_at_5") is not None and _as_float(row.get("gt_rank_at_5")) < 1.0:
        reasons.append("gt_rank_miss_at_5")
        severity += 2.0
    if _as_float(row.get("ar_constraint_violation")) > 0:
        reasons.append("ar_constraint_violation")
        severity += 2.0
    if str(row.get("dataset", "")).lower() == "gnmc" and _as_float(row.get("keep_full")) > 0:
        reasons.append("gnmc_full_frame_selected")
        severity += 1.5
    return reasons, severity


def _draw_box(draw: ImageDraw.ImageDraw, box: list[float], width: int, height: int, color: tuple[int, int, int], label: str) -> None:
    x1 = int(round(max(0.0, min(1.0, float(box[0]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(box[1]))) * height))
    x2 = int(round(max(0.0, min(1.0, float(box[2]))) * width))
    y2 = int(round(max(0.0, min(1.0, float(box[3]))) * height))
    for offset in range(3):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
    tx, ty = x1 + 3, max(3, y1 - 16)
    draw.rectangle([tx - 2, ty - 2, tx + max(50, 7 * len(label)), ty + 14], fill=(0, 0, 0))
    draw.text((tx, ty), label, fill=color)


def _render_case(row: dict[str, Any], out_path: Path, reasons: list[str]) -> None:
    image_path = Path(str(row.get("image_path", "")))
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        max_side = 1100
        if max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for idx, box in enumerate(row.get("gt_boxes", []) or []):
            if isinstance(box, list) and len(box) == 4:
                _draw_box(draw, box, width, height, COLORS["gt"], f"gt{idx}")
        top1_pre = row.get("top1_preinject_bbox_xyxy_norm")
        if isinstance(top1_pre, list) and len(top1_pre) == 4:
            _draw_box(draw, top1_pre, width, height, COLORS["top1_pre"], "top1_pre")
        top1 = row.get("top1_bbox_xyxy_norm")
        if isinstance(top1, list) and len(top1) == 4:
            _draw_box(draw, top1, width, height, COLORS["top1"], "top1")
        title = f"{row.get('dataset')} {row.get('image_id')} ar={row.get('target_ar') or 'FREE'}"
        subtitle = ", ".join(reasons[:4])
        draw.rectangle([0, 0, width, 42], fill=(0, 0, 0))
        draw.text((8, 5), title, fill=COLORS["text"])
        draw.text((8, 23), subtitle, fill=COLORS["text"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, quality=92)


def build_failure_gallery(rows: list[dict[str, Any]], output_dir: Path, *, max_items: int = 24) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    cases: list[dict[str, Any]] = []
    for row in rows:
        reasons, severity = classify_failure(row)
        if not reasons:
            continue
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.exists():
            continue
        case = dict(row)
        case["failure_reasons"] = reasons
        case["failure_severity"] = severity
        cases.append(case)
    cases.sort(key=lambda item: (_as_float(item.get("failure_severity")), str(item.get("dataset")), str(item.get("image_id"))), reverse=True)
    selected = cases[: max(0, int(max_items))]

    rendered_rows: list[dict[str, Any]] = []
    for idx, case in enumerate(selected, start=1):
        name = "_".join(
            [
                f"{idx:03d}",
                _safe_name(str(case.get("dataset", ""))),
                _safe_name(str(case.get("image_id", ""))),
                _safe_name(str(case.get("target_ar") or "FREE")),
            ]
        )
        rel = Path("images") / f"{name}.jpg"
        out_path = output_dir / rel
        try:
            _render_case(case, out_path, list(case.get("failure_reasons", [])))
            case["overlay_path"] = str(out_path.resolve())
            rendered_rows.append(case)
        except Exception as exc:
            case["render_error"] = str(exc)

    cases_jsonl = output_dir / "failure_cases.jsonl"
    write_jsonl(cases_jsonl, rendered_rows)
    index_path = output_dir / "index.md"
    lines = ["# Public Benchmark Failure Gallery", ""]
    lines.append(f"- rendered_cases: `{len(rendered_rows)}`")
    lines.append(f"- candidate_failures: `{len(cases)}`")
    lines.append("")
    for case in rendered_rows:
        rel = Path(str(case["overlay_path"])).relative_to(output_dir)
        title = f"{case.get('dataset')} / {case.get('image_id')} / {case.get('target_ar') or 'FREE'}"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- reasons: `{', '.join(case.get('failure_reasons', []))}`")
        lines.append(f"- iou_top1_preinject: `{case.get('iou_top1_preinject')}`")
        lines.append(f"- gt_rank_at_5: `{case.get('gt_rank_at_5')}`")
        lines.append("")
        lines.append(f"![{title}]({rel.as_posix()})")
        lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "rendered_cases": len(rendered_rows),
        "candidate_failures": len(cases),
        "gallery_dir": str(output_dir.resolve()),
        "failure_cases_jsonl": str(cases_jsonl.resolve()),
        "index_md": str(index_path.resolve()),
    }
