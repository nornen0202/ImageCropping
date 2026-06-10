#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.evaluate_gaic_score_explanation_consistency import (  # noqa: E402
    build_explanation_payload,
    load_prediction_scores,
    read_rows,
    row_key,
    safe_float,
)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def group_indices(rows: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get("image_id", ""))].append(idx)
    return dict(sorted(groups.items()))


def rank_pct_by_group(rows: Sequence[dict[str, Any]], scores: Sequence[float]) -> list[float]:
    out = [0.0] * len(rows)
    for indices in group_indices(rows).values():
        if len(indices) <= 1:
            for idx in indices:
                out[idx] = 0.5
            continue
        sorted_indices = sorted(indices, key=lambda idx: (float(scores[idx]), idx))
        denom = float(max(1, len(sorted_indices) - 1))
        for rank, idx in enumerate(sorted_indices):
            out[idx] = float(rank) / denom
    return out


def selected_indices_by_group(rows: Sequence[dict[str, Any]], scores: Sequence[float]) -> set[int]:
    selected: set[int] = set()
    for indices in group_indices(rows).values():
        if not indices:
            continue
        selected.add(max(indices, key=lambda idx: (float(scores[idx]), -idx)))
    return selected


def quality_tier(*, selected: bool, rank_pct: float, explanation_score: float, confidence: float, fatal: bool, contradiction: bool) -> str:
    if fatal and rank_pct >= 0.85:
        return "hard_negative"
    if selected and not contradiction and confidence >= 0.60 and explanation_score >= 0.45:
        return "main_positive"
    if selected:
        return "low_weight_positive"
    if rank_pct >= 0.80 and contradiction:
        return "low_weight_high_score"
    return "candidate"


def training_weight(*, rank_pct: float, confidence: float, fatal: bool, contradiction: bool, selected: bool) -> float:
    rank_term = 0.20 + 0.80 * clamp(rank_pct)
    confidence_term = 0.25 + 0.75 * clamp(confidence)
    weight = rank_term * confidence_term
    if contradiction:
        weight *= 0.45
    if fatal:
        weight *= 0.35
    if selected:
        weight = max(weight, 0.35 if contradiction or fatal else 0.75)
    return float(clamp(weight, 0.05, 1.0))


def iter_label_rows(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    method: str,
    split_name: str,
    score_source: str,
    contradiction_threshold: float,
) -> Iterable[dict[str, Any]]:
    rank_pct = rank_pct_by_group(rows, scores)
    selected = selected_indices_by_group(rows, scores)
    for idx, (row, score) in enumerate(zip(rows, scores)):
        explanation = build_explanation_payload(row)
        explanation_score = safe_float(explanation.get("explanation_score"), 0.0)
        confidence = safe_float(explanation.get("explanation_confidence"), 0.0)
        fatal = bool(explanation.get("fatal_flag", False))
        contradiction = bool(fatal or explanation_score < float(contradiction_threshold))
        is_selected = idx in selected
        tier = quality_tier(
            selected=is_selected,
            rank_pct=rank_pct[idx],
            explanation_score=explanation_score,
            confidence=confidence,
            fatal=fatal,
            contradiction=contradiction,
        )
        yield {
            "image_id": str(row.get("image_id", "")),
            "candidate_id": str(row.get("candidate_id", "")),
            "gt_annotation_id": row.get("gt_annotation_id"),
            "split": split_name,
            "protocol": row.get("protocol"),
            "mos": safe_float(row.get("mos"), 0.0),
            "bbox_norm_xyxy": row.get("bbox_norm_xyxy"),
            "score_source": score_source,
            "ranker_method": method,
            "ranker_score": float(score),
            "score_rank_pct_by_image": float(rank_pct[idx]),
            "selected_by_ranker": bool(is_selected),
            "explanation_score": explanation_score,
            "explanation_confidence": confidence,
            "fatal_flag": fatal,
            "warnings": explanation.get("warnings", []),
            "contradiction_flag": contradiction,
            "checklist_labels": explanation.get("checklist_labels", {}),
            "label_quality_tier": tier,
            "training_weight": training_weight(
                rank_pct=rank_pct[idx],
                confidence=confidence,
                fatal=fatal,
                contradiction=contradiction,
                selected=is_selected,
            ),
        }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Counter:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            counts["rows"] += 1
            counts[f"tier:{row.get('label_quality_tier')}"] += 1
            if row.get("selected_by_ranker"):
                counts["selected"] += 1
            if row.get("contradiction_flag"):
                counts["contradiction"] += 1
            if row.get("fatal_flag"):
                counts["fatal"] += 1
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return counts


def summarize_labels(path: Path) -> dict[str, Any]:
    rows = 0
    selected = 0
    contradiction = 0
    fatal = 0
    weights: list[float] = []
    rank_pct_selected: list[float] = []
    tiers: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            tiers[str(row.get("label_quality_tier", ""))] += 1
            selected += 1 if row.get("selected_by_ranker") else 0
            contradiction += 1 if row.get("contradiction_flag") else 0
            fatal += 1 if row.get("fatal_flag") else 0
            weights.append(safe_float(row.get("training_weight"), 0.0))
            if row.get("selected_by_ranker"):
                rank_pct_selected.append(safe_float(row.get("score_rank_pct_by_image"), 0.0))
    return {
        "rows": rows,
        "selected": selected,
        "contradiction_rate": float(contradiction / rows) if rows else 0.0,
        "fatal_rate": float(fatal / rows) if rows else 0.0,
        "mean_training_weight": float(sum(weights) / len(weights)) if weights else 0.0,
        "mean_selected_rank_pct": float(sum(rank_pct_selected) / len(rank_pct_selected)) if rank_pct_selected else 0.0,
        "tiers": dict(sorted(tiers.items())),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GAIC ranker training labels with explanation quality signals.")
    parser.add_argument("--candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--prediction_jsonl", action="append", required=True, help="NAME=PATH prediction JSONL. Repeatable.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol", choices=["Gc", "Ge"], required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--score_source", default="deep_crop_ranker")
    parser.add_argument("--contradiction_threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_scores = load_prediction_scores(args.prediction_jsonl)
    if args.method not in prediction_scores:
        raise RuntimeError(f"method={args.method} not found in prediction_jsonl specs")
    rows = read_rows(args.candidate_eval_jsonl, protocol=str(args.protocol))
    scores = [safe_float(prediction_scores[str(args.method)].get(row_key(row)), 0.0) for row in rows]
    counts = write_jsonl(
        args.output_jsonl,
        iter_label_rows(
            rows,
            scores,
            method=str(args.method),
            split_name=str(args.split_name),
            score_source=str(args.score_source),
            contradiction_threshold=float(args.contradiction_threshold),
        ),
    )
    summary = {
        "method": str(args.method),
        "protocol": str(args.protocol),
        "split": str(args.split_name),
        "candidate_eval_jsonl": str(args.candidate_eval_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "counts": dict(sorted(counts.items())),
        "summary": summarize_labels(args.output_jsonl),
    }
    write_json(args.summary_json, summary)
    print(json.dumps({"status": "ok", **summary["summary"], "output_jsonl": str(args.output_jsonl)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
