#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mobilecropnet_v4.gaic_benchmark import DEFAULT_RETURN_K, DEFAULT_TOP_N, evaluate_scored_records, load_gaic_annotation_records
from scripts.evaluate_gaic_score_explanation_consistency import build_explanation_payload, read_rows, safe_float
try:
    from scripts.evaluate_public_croppers_gaic_v2 import _score_cgs, _score_gaic
except ModuleNotFoundError:
    from evaluate_public_croppers_gaic_v2 import _score_cgs, _score_gaic


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def group_indices(rows: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row.get("image_id", ""))].append(idx)
    return dict(sorted(grouped.items()))


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


def label_tier(
    *,
    selected: bool,
    rank_pct: float,
    explanation_score: float,
    confidence: float,
    fatal: bool,
    contradiction: bool,
    high_score_threshold: float,
    low_score_threshold: float,
) -> str:
    high_public = bool(selected or rank_pct >= float(high_score_threshold))
    low_public = bool(rank_pct <= float(low_score_threshold))
    explanation_good = bool((not fatal) and (not contradiction) and confidence >= 0.60 and explanation_score >= 0.45)
    sstk_negative = bool(fatal or contradiction or explanation_score < 0.45 or confidence < 0.45)
    if high_public and fatal:
        return "gaic_public_positive_sstk_fatal"
    if high_public and contradiction:
        return "score_explanation_conflict"
    if high_public and explanation_good:
        return "main_positive"
    if high_public:
        return "low_weight_positive"
    if low_public and sstk_negative:
        return "normal_negative"
    if fatal:
        return "hard_negative"
    return "candidate"


def training_weight(
    *,
    rank_pct: float,
    confidence: float,
    fatal: bool,
    contradiction: bool,
    selected: bool,
    tier: str,
) -> float:
    rank_term = 0.20 + 0.80 * clamp(rank_pct)
    confidence_term = 0.25 + 0.75 * clamp(confidence)
    weight = rank_term * confidence_term
    if tier == "main_positive":
        weight = max(weight, 0.85)
    elif tier == "score_explanation_conflict":
        weight = min(max(weight * 0.50, 0.35 if selected else 0.20), 0.60)
    elif tier == "gaic_public_positive_sstk_fatal":
        weight = min(max(weight * 0.30, 0.20 if selected else 0.10), 0.45)
    elif tier == "low_weight_positive":
        weight = min(max(weight * 0.65, 0.40 if selected else 0.20), 0.70)
    elif tier == "normal_negative":
        weight = max(weight, 0.55)
    else:
        if contradiction:
            weight *= 0.55
        if fatal:
            weight *= 0.40
        if selected:
            weight = max(weight, 0.35)
    return float(clamp(weight, 0.05, 1.0))


def scorer_for_method(method: str):
    if method == "gaic":
        return _score_gaic
    if method == "cgs":
        return _score_cgs
    raise ValueError(f"unsupported method for candidate score labels: {method}")


def score_records(
    *,
    records: Sequence[dict[str, Any]],
    method: str,
    device: torch.device,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], dict[str, Any]]:
    scorer = scorer_for_method(method)
    return scorer(records, device=device)


def score_map_from_records(records: Sequence[dict[str, Any]], scores_by_image: dict[str, Sequence[float]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        scores = list(scores_by_image.get(image_id) or [])
        candidates = list(record.get("candidates") or [])
        if len(scores) != len(candidates):
            continue
        for candidate, score in zip(candidates, scores):
            out[(image_id, str(candidate.get("annotation_id", "")))] = float(score)
    return out


def iter_label_rows(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    method: str,
    split_name: str,
    score_source: str,
    contradiction_threshold: float,
    high_score_threshold: float,
    low_score_threshold: float,
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
        tier = label_tier(
            selected=is_selected,
            rank_pct=rank_pct[idx],
            explanation_score=explanation_score,
            confidence=confidence,
            fatal=fatal,
            contradiction=contradiction,
            high_score_threshold=float(high_score_threshold),
            low_score_threshold=float(low_score_threshold),
        )
        yield {
            "image_id": str(row.get("image_id", "")),
            "candidate_id": str(row.get("candidate_id", "")),
            "gt_annotation_id": row.get("gt_annotation_id"),
            "split": split_name,
            "protocol": row.get("protocol"),
            "subject_mode": row.get("subject_mode"),
            "mode_bucket": row.get("mode_bucket"),
            "gt_flag": row.get("gt_flag"),
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
                tier=tier,
            ),
        }


def summarize_labels(path: Path) -> dict[str, Any]:
    rows = 0
    selected = 0
    contradiction = 0
    fatal = 0
    weights: list[float] = []
    rank_pct_selected: list[float] = []
    tiers: Counter[str] = Counter()
    selected_tiers: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            tier = str(row.get("label_quality_tier", ""))
            tiers[tier] += 1
            if row.get("selected_by_ranker"):
                selected += 1
                selected_tiers[tier] += 1
                rank_pct_selected.append(safe_float(row.get("score_rank_pct_by_image"), 0.0))
            contradiction += 1 if row.get("contradiction_flag") else 0
            fatal += 1 if row.get("fatal_flag") else 0
            weights.append(safe_float(row.get("training_weight"), 0.0))
    return {
        "rows": rows,
        "selected": selected,
        "contradiction_rate": float(contradiction / rows) if rows else 0.0,
        "fatal_rate": float(fatal / rows) if rows else 0.0,
        "mean_training_weight": float(sum(weights) / len(weights)) if weights else 0.0,
        "mean_selected_rank_pct": float(sum(rank_pct_selected) / len(rank_pct_selected)) if rank_pct_selected else 0.0,
        "tiers": dict(sorted(tiers.items())),
        "selected_tiers": dict(sorted(selected_tiers.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GAIC public cropper score labels with SSTK explanation payloads.")
    parser.add_argument("--annotations_json", type=Path, required=True)
    parser.add_argument("--candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--image_roots", nargs="+", type=Path, required=True)
    parser.add_argument("--method", choices=["gaic", "cgs"], default="gaic")
    parser.add_argument("--protocol", default="Gc")
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--score_source", default="gaic_public_cropper")
    parser.add_argument("--contradiction_threshold", type=float, default=0.35)
    parser.add_argument("--high_score_threshold", type=float, default=0.85)
    parser.add_argument("--low_score_threshold", type=float, default=0.20)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_gaic_annotation_records(args.annotations_json, image_roots=args.image_roots, max_images=args.max_images)
    records = [record for record in records if str(record.get("image_path", "")).strip()]
    device = torch.device(args.device)
    scores_by_image, coverage_by_image, scorer_metadata = score_records(records=records, method=str(args.method), device=device)
    score_map = score_map_from_records(records, scores_by_image)
    candidate_rows = read_rows(args.candidate_eval_jsonl, protocol=str(args.protocol))
    if args.max_images is not None:
        allowed = {str(record.get("image_id", "")) for record in records}
        candidate_rows = [row for row in candidate_rows if str(row.get("image_id", "")) in allowed]
    aligned_rows: list[dict[str, Any]] = []
    aligned_scores: list[float] = []
    missing: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = (str(row.get("image_id", "")), str(row.get("gt_annotation_id", "")))
        if key not in score_map:
            missing.append({"image_id": key[0], "gt_annotation_id": key[1], "candidate_id": row.get("candidate_id")})
            continue
        aligned_rows.append(row)
        aligned_scores.append(float(score_map[key]))
    if not aligned_rows:
        raise RuntimeError("No candidate rows could be aligned to public cropper scores.")
    written = write_jsonl(
        args.output_jsonl,
        iter_label_rows(
            aligned_rows,
            aligned_scores,
            method=str(args.method),
            split_name=str(args.split_name),
            score_source=str(args.score_source),
            contradiction_threshold=float(args.contradiction_threshold),
            high_score_threshold=float(args.high_score_threshold),
            low_score_threshold=float(args.low_score_threshold),
        ),
    )
    eval_summary = evaluate_scored_records(
        records,
        scores_by_image,
        method_name=str(args.method),
        coverage_by_image_id=coverage_by_image,
        return_k_values=DEFAULT_RETURN_K,
        top_n_values=DEFAULT_TOP_N,
    )
    summary = {
        "method": str(args.method),
        "score_source": str(args.score_source),
        "protocol": str(args.protocol),
        "split": str(args.split_name),
        "annotations_json": str(args.annotations_json),
        "candidate_eval_jsonl": str(args.candidate_eval_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "written_rows": int(written),
        "aligned_rows": len(aligned_rows),
        "missing_rows": len(missing),
        "missing_examples": missing[:50],
        "image_count": len(records),
        "candidate_count": sum(len(record.get("candidates") or []) for record in records),
        "label_summary": summarize_labels(args.output_jsonl),
        "gaic_benchmark_metrics": {k: v for k, v in eval_summary.items() if k != "per_image"},
        "scorer_metadata": scorer_metadata,
        "contradiction_threshold": float(args.contradiction_threshold),
        "high_score_threshold": float(args.high_score_threshold),
        "low_score_threshold": float(args.low_score_threshold),
    }
    write_json(args.summary_json, summary)
    print(json.dumps({"status": "ok", "output_jsonl": str(args.output_jsonl), **summary["label_summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
