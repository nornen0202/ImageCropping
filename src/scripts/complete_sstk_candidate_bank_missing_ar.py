#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_INPUT = DEFAULT_PHASE_ROOT / "artifacts/candidates/candidates_ar_260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010.jsonl"
DEFAULT_OUTPUT = DEFAULT_PHASE_ROOT / "artifacts/candidates/candidates_ar_260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010_v17_complete_260609.jsonl"
TARGET_ARS = ("FREE", "1:1", "4:3", "3:4", "16:9", "9:16")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _parse_target_ar(target_ar: str, image_ar: float) -> float:
    text = str(target_ar or "").strip().upper()
    if text == "FREE":
        return max(1e-6, float(image_ar))
    if ":" in text:
        left, right = text.split(":", 1)
        return max(1e-6, _safe_float(left, 1.0) / max(1e-6, _safe_float(right, 1.0)))
    return max(1e-6, _safe_float(text, image_ar))


def _clip_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _candidate_box_for_center(crop_w: float, crop_h: float, cx: float, cy: float) -> list[float]:
    x1 = float(cx) - 0.5 * float(crop_w)
    y1 = float(cy) - 0.5 * float(crop_h)
    x2 = x1 + float(crop_w)
    y2 = y1 + float(crop_h)
    if x1 < 0.0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0.0:
        y2 -= y1
        y1 = 0.0
    if x2 > 1.0:
        x1 -= x2 - 1.0
        x2 = 1.0
    if y2 > 1.0:
        y1 -= y2 - 1.0
        y2 = 1.0
    return _clip_box([x1, y1, x2, y2])


def _completion_candidates(row: dict[str, Any], target_ar: str) -> list[dict[str, Any]]:
    width = max(1.0, _safe_float(row.get("width"), _safe_float(row.get("width_parquet"), 1.0)))
    height = max(1.0, _safe_float(row.get("height"), _safe_float(row.get("height_parquet"), 1.0)))
    image_ar = width / height
    target_pixel_ar = _parse_target_ar(target_ar, image_ar)
    crop_ar_norm = max(1e-6, target_pixel_ar / max(1e-6, image_ar))
    if crop_ar_norm >= 1.0:
        crop_w = 1.0
        crop_h = 1.0 / crop_ar_norm
    else:
        crop_h = 1.0
        crop_w = crop_ar_norm
    centers_x = [0.5, 1.0 / 3.0, 2.0 / 3.0, 0.2, 0.8]
    centers_y = [0.5]
    out: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for idx, cx in enumerate(centers_x, start=1):
        for cy in centers_y:
            box = _candidate_box_for_center(crop_w, crop_h, cx, cy)
            key = tuple(round(v, 6) for v in box)
            if key in seen or _area(box) <= 0.0:
                continue
            seen.add(key)
            out.append(
                {
                    "candidate_id": f"{target_ar}_v17complete_maxarea_slide_i{idx:04d}",
                    "bbox_norm_xyxy": [round(v, 8) for v in box],
                    "ar": round(target_pixel_ar, 8),
                    "area_ratio": round(_area(box), 8),
                    "source": "completion_bank_maxarea_slide",
                    "source_lineage": ["candidate_bank_completion_v17", "precompute_geometry"],
                    "source_types": ["completion_bank_maxarea_slide"],
                    "must_keep": True,
                    "priority": 95 - idx,
                    "teacher_derived": False,
                    "teacher_ids": [],
                    "teacher_stages": [],
                    "teacher_provenance": [],
                }
            )
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete SSTK candidate bank rows that have empty AR buckets.")
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overview_json", type=Path, default=None)
    parser.add_argument("--status_json", type=Path, default=None)
    parser.add_argument("--target_ars", nargs="+", default=list(TARGET_ARS))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = time.time()
    overview_path = args.overview_json or args.output_jsonl.with_name(args.output_jsonl.stem + "_overview.json")
    status_path = args.status_json or args.output_jsonl.with_name(args.output_jsonl.stem + "_status.json")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_json(status_path, {"state": "running", "phase": "copy_and_complete", "input_jsonl": str(args.input_jsonl), "output_jsonl": str(args.output_jsonl)})
    rows = 0
    completed_rows = 0
    completed_candidates = 0
    empty_after = Counter()
    added_by_ar = Counter()
    total_by_ar = Counter()
    with args.input_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            by_ar = row.get("candidates_by_ar")
            if not isinstance(by_ar, dict):
                by_ar = {}
                row["candidates_by_ar"] = by_ar
            row_added = 0
            for target_ar in args.target_ars:
                existing = by_ar.get(target_ar)
                if isinstance(existing, list) and existing:
                    total_by_ar[target_ar] += len(existing)
                    continue
                generated = _completion_candidates(row, target_ar)
                by_ar[target_ar] = generated
                row_added += len(generated)
                completed_candidates += len(generated)
                added_by_ar[target_ar] += len(generated)
                total_by_ar[target_ar] += len(generated)
                if not generated:
                    empty_after[target_ar] += 1
            if row_added:
                completed_rows += 1
                stats = row.get("stats")
                if not isinstance(stats, dict):
                    stats = {}
                    row["stats"] = stats
                num_by_ar = stats.get("num_candidates_by_ar")
                if not isinstance(num_by_ar, dict):
                    num_by_ar = {}
                    stats["num_candidates_by_ar"] = num_by_ar
                for target_ar in args.target_ars:
                    num_by_ar[target_ar] = len(by_ar.get(target_ar) or [])
                stats["candidate_bank_completion_v17"] = {
                    "completed": True,
                    "added_candidates": row_added,
                    "policy": "deterministic max-area/thirds geometry candidates for previously empty AR buckets",
                }
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if rows % 500 == 0:
                _write_json(status_path, {"state": "running", "phase": "copy_and_complete", "rows": rows, "completed_rows": completed_rows})
    summary = {
        "state": "completed",
        "schema_version": "sstk_candidate_bank_completion_v17_20260609",
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "row_count": rows,
        "completed_row_count": completed_rows,
        "completed_candidate_count": completed_candidates,
        "added_by_ar": dict(added_by_ar),
        "total_by_ar": dict(total_by_ar),
        "empty_after_by_ar": dict(empty_after),
        "elapsed_sec": round(time.time() - start, 3),
        "notes": {
            "teacher_policy": "Completion candidates are generated candidate-bank artifacts. They do not fabricate GAIC/CGS scores; teacher_provenance remains empty unless public-teacher inference provides a real score.",
            "label_factory_policy": "Downstream label generation must read this completed bank and must not synthesize replacements for missing AR buckets internally.",
        },
    }
    _write_json(overview_path, summary)
    _write_json(status_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
