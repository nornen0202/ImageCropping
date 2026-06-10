#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.score_mobilecropnet_v4_product_ar_candidates_with_public_cropper import (
    iter_label_rows,
    read_jsonl,
    summarize,
    write_json,
    write_jsonl,
)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")), str(row.get("candidate_id", "")))


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))


def normalize_group_scores(values: dict[str, float], mode: str) -> dict[str, float]:
    if not values:
        return {}
    scores = list(values.items())
    mode_key = str(mode).strip().lower()
    if mode_key == "rank_pct":
        ordered = sorted(scores, key=lambda item: (float(item[1]), item[0]), reverse=True)
        if len(ordered) == 1:
            return {ordered[0][0]: 1.0}
        denom = float(max(1, len(ordered) - 1))
        out: dict[str, float] = {}
        pos = 0
        while pos < len(ordered):
            end = pos + 1
            while end < len(ordered) and float(ordered[end][1]) == float(ordered[pos][1]):
                end += 1
            avg_rank = 0.5 * (float(pos + 1) + float(end))
            score = float(1.0 - (avg_rank - 1.0) / denom)
            for candidate_id, _ in ordered[pos:end]:
                out[candidate_id] = score
            pos = end
        return out
    if mode_key == "zscore":
        raw = [float(score) for _, score in scores]
        mean = float(sum(raw) / len(raw))
        var = float(sum((score - mean) ** 2 for score in raw) / len(raw))
        std = math.sqrt(max(0.0, var))
        if std <= 1e-12:
            return {candidate_id: 0.0 for candidate_id, _ in scores}
        return {candidate_id: float((score - mean) / std) for candidate_id, score in scores}
    raise ValueError(f"unsupported normalization mode: {mode}")


def fuse_scores(
    primary_rows: Sequence[dict[str, Any]],
    secondary_rows: Sequence[dict[str, Any]],
    *,
    normalization: str,
    primary_weight: float,
    secondary_weight: float,
) -> tuple[list[float], dict[str, Any]]:
    secondary_by_key = {_row_key(row): row for row in secondary_rows}
    grouped_primary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_secondary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        grouped_primary[_group_key(row)].append(row)
    for row in secondary_rows:
        grouped_secondary[_group_key(row)].append(row)

    fused_by_key: dict[tuple[str, str, str], float] = {}
    missing_secondary = 0
    for group_key, group_rows in grouped_primary.items():
        primary_scores = {str(row.get("candidate_id", "")): float(row.get("ranker_score", 0.0)) for row in group_rows}
        secondary_scores = {
            str(row.get("candidate_id", "")): float(row.get("ranker_score", 0.0))
            for row in grouped_secondary.get(group_key, [])
        }
        norm_primary = normalize_group_scores(primary_scores, normalization)
        norm_secondary = normalize_group_scores(secondary_scores, normalization) if secondary_scores else {}
        for row in group_rows:
            candidate_id = str(row.get("candidate_id", ""))
            score = float(primary_weight) * float(norm_primary.get(candidate_id, 0.0))
            if candidate_id in norm_secondary:
                score += float(secondary_weight) * float(norm_secondary[candidate_id])
            else:
                missing_secondary += 1
            fused_by_key[_row_key(row)] = float(score)

    fused = [float(fused_by_key.get(_row_key(row), 0.0)) for row in primary_rows]
    metadata = {
        "normalization": str(normalization),
        "primary_weight": float(primary_weight),
        "secondary_weight": float(secondary_weight),
        "primary_rows": len(primary_rows),
        "secondary_rows": len(secondary_rows),
        "group_count": len(grouped_primary),
        "missing_secondary_count": int(missing_secondary),
        "matched_secondary_count": int(sum(1 for row in primary_rows if _row_key(row) in secondary_by_key)),
    }
    return fused, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse two Product AR public-cropper score JSONLs into one ensemble score JSONL.")
    parser.add_argument("--primary_scores_jsonl", type=Path, required=True)
    parser.add_argument("--secondary_scores_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--method_name", default="public_cropper_ensemble")
    parser.add_argument("--score_source", default="public_cropper_ensemble_best")
    parser.add_argument("--normalization", choices=["rank_pct", "zscore"], default="rank_pct")
    parser.add_argument("--primary_weight", type=float, default=0.65)
    parser.add_argument("--secondary_weight", type=float, default=0.35)
    parser.add_argument("--high_score_threshold", type=float, default=0.85)
    parser.add_argument("--low_score_threshold", type=float, default=0.20)
    parser.add_argument("--contradiction_threshold", type=float, default=0.35)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary_rows = read_jsonl(args.primary_scores_jsonl, max_rows=int(args.max_rows))
    secondary_rows = read_jsonl(args.secondary_scores_jsonl, max_rows=int(args.max_rows))
    fused_scores, metadata = fuse_scores(
        primary_rows,
        secondary_rows,
        normalization=str(args.normalization),
        primary_weight=float(args.primary_weight),
        secondary_weight=float(args.secondary_weight),
    )
    written = write_jsonl(
        args.output_jsonl,
        iter_label_rows(
            primary_rows,
            fused_scores,
            method=str(args.method_name),
            score_source=str(args.score_source),
            high_threshold=float(args.high_score_threshold),
            low_threshold=float(args.low_score_threshold),
            contradiction_threshold=float(args.contradiction_threshold),
        ),
    )
    summary = {
        "status": "ok",
        "primary_scores_jsonl": str(args.primary_scores_jsonl),
        "secondary_scores_jsonl": str(args.secondary_scores_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "method_name": str(args.method_name),
        "score_source": str(args.score_source),
        "written_rows": int(written),
        "fusion": metadata,
        **summarize(args.output_jsonl),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
