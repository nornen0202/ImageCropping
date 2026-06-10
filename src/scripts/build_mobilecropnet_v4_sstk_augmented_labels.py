#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import box_iou_xyxy, safe_float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SSTK-only perturbation and repair labels for MobileCropNet v4.")
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_pairs_per_row", type=int, default=4)
    parser.add_argument("--min_positive_score", type=float, default=0.65)
    return parser


def _clamp_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [max(0.0, min(1.0, safe_float(v))) for v in box[:4]]
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [x1, y1, x2, y2]


def _box(candidate: dict[str, Any]) -> list[float]:
    if isinstance(candidate.get("bbox_norm_xyxy"), list):
        return _clamp_box(candidate["bbox_norm_xyxy"])
    if isinstance(candidate.get("bbox_cxcywh"), list):
        cx, cy, w, h = [safe_float(v) for v in candidate["bbox_cxcywh"][:4]]
        return _clamp_box([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h])
    return [0.0, 0.0, 1.0, 1.0]


def _score(candidate: dict[str, Any]) -> float:
    for key in ("crop_utility_prob", "score_prob"):
        if candidate.get(key) is not None:
            return max(0.0, min(1.0, safe_float(candidate.get(key))))
    targets = candidate.get("score_targets")
    if isinstance(targets, dict):
        for key in ("crop_utility_prob", "score_prob"):
            if targets.get(key) is not None:
                return max(0.0, min(1.0, safe_float(targets.get(key))))
    return max(0.0, min(1.0, safe_float(candidate.get("score_rank_pct"), 0.0)))


def _perturbations(box: list[float]) -> list[tuple[str, list[float]]]:
    x1, y1, x2, y2 = _clamp_box(box)
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    out = [
        ("shift_right", [x1 + 0.25 * w, y1, x2 + 0.25 * w, y2]),
        ("shift_down", [x1, y1 + 0.25 * h, x2, y2 + 0.25 * h]),
        ("subject_truncate_left", [x1 + 0.18 * w, y1, x2, y2]),
        ("subject_truncate_top", [x1, y1 + 0.18 * h, x2, y2]),
        ("excessive_empty_margin", [cx - 0.75 * w, cy - 0.75 * h, cx + 0.75 * w, cy + 0.75 * h]),
        ("over_tight", [cx - 0.32 * w, cy - 0.32 * h, cx + 0.32 * w, cy + 0.32 * h]),
        ("aspect_squeeze_tall", [cx - 0.28 * w, y1, cx + 0.28 * w, y2]),
        ("aspect_squeeze_wide", [x1, cy - 0.28 * h, x2, cy + 0.28 * h]),
    ]
    return [(name, _clamp_box(candidate)) for name, candidate in out if box_iou_xyxy(box, _clamp_box(candidate)) < 0.92]


def _positive_candidates(row: dict[str, Any], min_score: float) -> list[dict[str, Any]]:
    candidates = []
    for candidate in row.get("matching_targets") or []:
        candidates.append(candidate)
    for candidate in row.get("candidate_pool") or []:
        unsafe = bool(candidate.get("is_hard_negative")) or bool(candidate.get("is_unsafe_negative")) or bool(candidate.get("is_overflow_candidate"))
        if not unsafe and _score(candidate) >= min_score:
            candidates.append(candidate)
    candidates.sort(key=_score, reverse=True)
    return candidates


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perturbation_path = args.output_dir / "perturbation_pairs.jsonl"
    repair_path = args.output_dir / "repair_targets.jsonl"
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "rows_seen": 0,
        "rows_with_pairs": 0,
        "perturbation_pairs": 0,
        "repair_targets": 0,
        "min_positive_score": float(args.min_positive_score),
        "max_pairs_per_row": int(args.max_pairs_per_row),
    }
    with args.input_jsonl.open("r", encoding="utf-8") as src, perturbation_path.open("w", encoding="utf-8") as pair_out, repair_path.open("w", encoding="utf-8") as repair_out:
        for line in src:
            if not line.strip():
                continue
            if args.max_rows is not None and summary["rows_seen"] >= int(args.max_rows):
                break
            row = json.loads(line)
            summary["rows_seen"] += 1
            positives = _positive_candidates(row, args.min_positive_score)
            wrote_for_row = False
            for candidate in positives[: max(1, int(args.max_pairs_per_row))]:
                pos_box = _box(candidate)
                candidate_id = str(candidate.get("candidate_id") or candidate.get("target_id") or "")
                for perturbation_type, neg_box in _perturbations(pos_box)[:1]:
                    payload = {
                        "image_id": str(row.get("image_id", "")),
                        "image_path": str(row.get("image_path", "")),
                        "target_ar": str(row.get("target_ar", "FREE")),
                        "source_candidate_id": candidate_id,
                        "positive_bbox_norm_xyxy": pos_box,
                        "negative_bbox_norm_xyxy": neg_box,
                        "perturbation_type": perturbation_type,
                        "label": 1,
                        "pair_weight": max(0.25, min(1.0, _score(candidate))),
                        "routing": row.get("routing", {}) if isinstance(row.get("routing"), dict) else {},
                    }
                    pair_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    repair_out.write(
                        json.dumps(
                            {
                                "image_id": payload["image_id"],
                                "image_path": payload["image_path"],
                                "target_ar": payload["target_ar"],
                                "source_candidate_id": candidate_id,
                                "damaged_bbox_norm_xyxy": neg_box,
                                "target_bbox_norm_xyxy": pos_box,
                                "perturbation_type": perturbation_type,
                                "repair_weight": payload["pair_weight"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    summary["perturbation_pairs"] += 1
                    summary["repair_targets"] += 1
                    wrote_for_row = True
            if wrote_for_row:
                summary["rows_with_pairs"] += 1
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
