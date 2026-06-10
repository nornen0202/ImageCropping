#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import quantiles
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_teacher_score(candidate: dict[str, Any]) -> float:
    for key in ("score_target", "crop_utility_prob", "score_prob", "raw_ranker_zscore_prob", "raw_ranker_minmax_prob"):
        if key in candidate:
            return safe_float(candidate.get(key), 0.0)
    return 0.0


def candidate_student_score(candidate: dict[str, Any]) -> float:
    for key in ("model_utility", "model_score", "utility"):
        if key in candidate:
            return safe_float(candidate.get(key), 0.0)
    return 0.0


def row_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidates", "candidate_pool", "proposal_rows"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [c for c in value if isinstance(c, dict) and str(c.get("candidate_id", c.get("proposal_id", "")))]
    return []


def candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or candidate.get("proposal_id") or "")


def high_threshold(scores: list[float], quantile: float) -> float:
    if not scores:
        return 1.0
    q = max(0.0, min(1.0, float(quantile)))
    if len(scores) < 4:
        return sorted(scores)[max(0, min(len(scores) - 1, int(round((len(scores) - 1) * q))))]
    return float(quantiles(scores, n=100, method="inclusive")[max(0, min(98, int(round(q * 100)) - 1))])


def mine_row(
    row: dict[str, Any],
    *,
    teacher_low_threshold: float,
    teacher_positive_threshold: float,
    student_high_quantile: float,
    max_pairs_per_image: int,
) -> list[dict[str, Any]]:
    candidates = row_candidates(row)
    if not candidates:
        return []
    student_scores = [candidate_student_score(c) for c in candidates]
    student_thr = high_threshold(student_scores, student_high_quantile)
    positives = [
        c
        for c in candidates
        if candidate_teacher_score(c) >= teacher_positive_threshold
        and not bool(c.get("is_unsafe_negative", False))
        and not bool(c.get("is_hard_negative", False))
    ]
    hard_negatives = [
        c
        for c in candidates
        if candidate_teacher_score(c) <= teacher_low_threshold and candidate_student_score(c) >= student_thr
    ]
    positives.sort(key=candidate_teacher_score, reverse=True)
    hard_negatives.sort(key=candidate_student_score, reverse=True)
    out: list[dict[str, Any]] = []
    for positive in positives[: max(1, min(4, len(positives)))]:
        pos_score = candidate_teacher_score(positive)
        for negative in hard_negatives:
            if candidate_id(positive) == candidate_id(negative):
                continue
            neg_score = candidate_teacher_score(negative)
            student_gap = candidate_student_score(negative) - candidate_student_score(positive)
            margin = max(0.01, pos_score - neg_score)
            out.append(
                {
                    "image_id": str(row.get("image_id", "")),
                    "target_ar": str(row.get("target_ar", "FREE")),
                    "candidate_id_a": candidate_id(positive),
                    "candidate_id_b": candidate_id(negative),
                    "label": 1,
                    "score_margin": margin,
                    "crop_utility_margin": margin,
                    "weight": max(0.5, min(1.0, margin + max(0.0, student_gap))),
                    "pair_type": "student_overscore_hard_negative",
                    "teacher_score_a": pos_score,
                    "teacher_score_b": neg_score,
                    "student_score_a": candidate_student_score(positive),
                    "student_score_b": candidate_student_score(negative),
                    "student_high_threshold": student_thr,
                }
            )
            if len(out) >= max_pairs_per_image:
                return out
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine MobileCropNet v4 student over-score hard-negative pairwise labels.")
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--teacher_low_threshold", type=float, default=0.25)
    parser.add_argument("--teacher_positive_threshold", type=float, default=0.75)
    parser.add_argument("--student_high_quantile", type=float, default=0.80)
    parser.add_argument("--max_pairs_per_image", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary = {"image_count": 0, "image_with_pairs": 0, "pair_count": 0}
    with args.predictions_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            summary["image_count"] += 1
            pairs = mine_row(
                row,
                teacher_low_threshold=args.teacher_low_threshold,
                teacher_positive_threshold=args.teacher_positive_threshold,
                student_high_quantile=args.student_high_quantile,
                max_pairs_per_image=max(1, int(args.max_pairs_per_image)),
            )
            if pairs:
                summary["image_with_pairs"] += 1
            for pair in pairs:
                summary["pair_count"] += 1
                dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
