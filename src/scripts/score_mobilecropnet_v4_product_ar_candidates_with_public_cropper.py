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

from scripts.evaluate_public_croppers_gaic_v2 import _score_cgs, _score_gaic


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def read_jsonl(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


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


def grouped_indices(rows: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get("score_group_id") or f"{row.get('image_id')}::{row.get('target_ar', 'FREE')}")].append(idx)
    return dict(sorted(groups.items()))


def rank_pct_by_group(rows: Sequence[dict[str, Any]], scores: Sequence[float]) -> list[float]:
    out = [0.0] * len(rows)
    for indices in grouped_indices(rows).values():
        if len(indices) <= 1:
            for idx in indices:
                out[idx] = 1.0
            continue
        sorted_indices = sorted(indices, key=lambda idx: (float(scores[idx]), idx))
        denom = float(max(1, len(sorted_indices) - 1))
        for rank, idx in enumerate(sorted_indices):
            out[idx] = float(rank) / denom
    return out


def selected_by_group(rows: Sequence[dict[str, Any]], scores: Sequence[float]) -> set[int]:
    out: set[int] = set()
    for indices in grouped_indices(rows).values():
        if indices:
            out.add(max(indices, key=lambda idx: (float(scores[idx]), -idx)))
    return out


def explanation_score(row: dict[str, Any]) -> tuple[float, float]:
    if row.get("sstk_hard_reject"):
        base = min(0.30, safe_float(row.get("sstk_crop_utility_prob"), 0.0))
        return clamp(base), 0.80
    macro = [
        clamp(safe_float(row.get("profile_A_macro"), 0.0)),
        clamp(safe_float(row.get("profile_S_macro"), 0.0)),
        clamp(safe_float(row.get("profile_C_macro"), 0.0)),
        clamp(safe_float(row.get("profile_T_macro"), 0.0)),
    ]
    known = [v for v in macro if v > 0.0]
    if known:
        return clamp(sum(known) / len(known)), 0.70
    score = clamp(safe_float(row.get("sstk_crop_utility_prob", row.get("sstk_score_prob", 0.0)), 0.0))
    return score, 0.45 if score <= 0.0 else 0.60


def label_tier(
    *,
    selected: bool,
    rank_pct: float,
    fatal: bool,
    contradiction: bool,
    expl_score: float,
    expl_conf: float,
    high_threshold: float,
    low_threshold: float,
) -> str:
    high_public = bool(selected or rank_pct >= float(high_threshold))
    low_public = bool(rank_pct <= float(low_threshold))
    explanation_good = bool((not fatal) and (not contradiction) and expl_conf >= 0.60 and expl_score >= 0.45)
    sstk_negative = bool(fatal or contradiction or expl_score < 0.45 or expl_conf < 0.45)
    if high_public and fatal:
        return "public_positive_sstk_fatal_review"
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


def training_weight(*, rank_pct: float, expl_conf: float, selected: bool, tier: str) -> float:
    weight = (0.20 + 0.80 * clamp(rank_pct)) * (0.30 + 0.70 * clamp(expl_conf))
    if tier == "main_positive":
        weight = max(weight, 0.85)
    elif tier == "score_explanation_conflict":
        weight = min(max(weight * 0.55, 0.35 if selected else 0.20), 0.65)
    elif tier == "public_positive_sstk_fatal_review":
        weight = min(max(weight * 0.25, 0.20 if selected else 0.10), 0.45)
    elif tier == "low_weight_positive":
        weight = min(max(weight * 0.70, 0.35 if selected else 0.20), 0.75)
    elif tier in {"normal_negative", "hard_negative"}:
        weight = max(weight, 0.70)
    return clamp(weight, 0.05, 1.0)


def build_records(
    rows: Sequence[dict[str, Any]],
    *,
    record_grouping: str = "image",
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    if record_grouping == "score_group":
        group_fn = lambda row: str(row.get("score_group_id") or f"{row.get('image_id')}::{row.get('target_ar', 'FREE')}")
    elif record_grouping == "image":
        group_fn = lambda row: str(row.get("image_id", ""))
    else:
        raise ValueError(f"unsupported record_grouping: {record_grouping}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[group_fn(row)].append(idx)
    records: list[dict[str, Any]] = []
    record_to_row_indices: dict[str, list[int]] = {}
    for group_id, indices in sorted(grouped.items()):
        first = rows[indices[0]]
        candidates = []
        for local_idx, row_idx in enumerate(indices):
            row = rows[row_idx]
            candidates.append(
                {
                    "annotation_id": local_idx,
                    "candidate_id": str(row.get("candidate_id", "")),
                    "bbox_norm_xyxy": row.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0],
                    "mos": 0.0,
                    "gt_flag": int(row.get("gt_flag", 0)),
                }
            )
        records.append(
            {
                "image_id": group_id,
                "file_name": Path(str(first.get("image_path", ""))).name,
                "width": int(first.get("image_width", 0) or 0),
                "height": int(first.get("image_height", 0) or 0),
                "image_path": str(first.get("image_path", "")),
                "candidates": candidates,
            }
        )
        record_to_row_indices[group_id] = list(indices)
    return records, record_to_row_indices


def score_records(records: Sequence[dict[str, Any]], *, method: str, device: torch.device):
    if method == "gaic":
        return _score_gaic(records, device=device)
    if method == "cgs":
        return _score_cgs(records, device=device)
    raise ValueError(f"unsupported public cropper method: {method}")


def flatten_scores(
    records: Sequence[dict[str, Any]],
    record_to_row_indices: dict[str, list[int]],
    rows: Sequence[dict[str, Any]],
    scores_by_record: dict[str, Sequence[float]],
) -> list[float]:
    out = [0.0] * len(rows)
    for record in records:
        group_id = str(record.get("image_id", ""))
        indices = record_to_row_indices.get(group_id, [])
        scores = list(scores_by_record.get(group_id) or [])
        if len(indices) != len(scores):
            continue
        for row_idx, score in zip(indices, scores):
            out[row_idx] = float(score)
    return out


def iter_label_rows(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    method: str,
    score_source: str,
    high_threshold: float,
    low_threshold: float,
    contradiction_threshold: float,
) -> Iterable[dict[str, Any]]:
    rank_pct = rank_pct_by_group(rows, scores)
    selected = selected_by_group(rows, scores)
    for idx, row in enumerate(rows):
        expl_score, expl_conf = explanation_score(row)
        fatal = bool(row.get("sstk_hard_reject") or row.get("sstk_is_hard_negative") or row.get("sstk_is_unsafe_negative"))
        contradiction = bool(fatal or expl_score < float(contradiction_threshold))
        tier = label_tier(
            selected=idx in selected,
            rank_pct=rank_pct[idx],
            fatal=fatal,
            contradiction=contradiction,
            expl_score=expl_score,
            expl_conf=expl_conf,
            high_threshold=float(high_threshold),
            low_threshold=float(low_threshold),
        )
        out = dict(row)
        out.update(
            {
                "score_source": str(score_source),
                "ranker_method": str(method),
                "ranker_score": float(scores[idx]),
                "score_rank_pct_by_image": float(rank_pct[idx]),
                "score_rank_pct_by_group": float(rank_pct[idx]),
                "selected_by_ranker": bool(idx in selected),
                "explanation_score": float(expl_score),
                "explanation_confidence": float(expl_conf),
                "fatal_flag": fatal,
                "contradiction_flag": contradiction,
                "warnings": list(row.get("sstk_reject_tags") or []),
                "label_quality_tier": tier,
                "training_weight": training_weight(rank_pct=rank_pct[idx], expl_conf=expl_conf, selected=idx in selected, tier=tier),
            }
        )
        yield out


def summarize(path: Path) -> dict[str, Any]:
    rows = 0
    images: set[str] = set()
    groups: set[str] = set()
    counters: Counter[str] = Counter()
    selected = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            images.add(str(row.get("image_id", "")))
            groups.add(str(row.get("score_group_id", "")))
            counters[f"target_ar::{row.get('target_ar', '')}"] += 1
            counters[f"tier::{row.get('label_quality_tier', '')}"] += 1
            if row.get("fatal_flag"):
                counters["fatal"] += 1
            if row.get("contradiction_flag"):
                counters["contradiction"] += 1
            if row.get("selected_by_ranker"):
                selected += 1
                counters[f"selected_tier::{row.get('label_quality_tier', '')}"] += 1
    return {
        "rows": rows,
        "image_count": len(images),
        "group_count": len(groups),
        "selected": selected,
        "counter": dict(sorted(counters.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Product AR candidate rows with a public GAIC/CGS cropper.")
    parser.add_argument("--candidate_rows_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--method", choices=["gaic", "cgs"], default="gaic")
    parser.add_argument("--score_source", default="gaic_public_cropper_product_ar")
    parser.add_argument("--high_score_threshold", type=float, default=0.85)
    parser.add_argument("--low_score_threshold", type=float, default=0.20)
    parser.add_argument("--contradiction_threshold", type=float, default=0.35)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument(
        "--record_grouping",
        choices=["image", "score_group"],
        default="image",
        help="Use image-level records for efficient public-cropper inference, while ranking is still computed by score_group_id.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.candidate_rows_jsonl, max_rows=int(args.max_rows))
    records, record_to_row_indices = build_records(rows, record_grouping=str(args.record_grouping))
    device = torch.device(str(args.device))
    scores_by_record, coverage_by_record, scorer_metadata = score_records(records, method=str(args.method), device=device)
    scores = flatten_scores(records, record_to_row_indices, rows, scores_by_record)
    written = write_jsonl(
        args.output_jsonl,
        iter_label_rows(
            rows,
            scores,
            method=str(args.method),
            score_source=str(args.score_source),
            high_threshold=float(args.high_score_threshold),
            low_threshold=float(args.low_score_threshold),
            contradiction_threshold=float(args.contradiction_threshold),
        ),
    )
    summary = {
        "status": "ok",
        "candidate_rows_jsonl": str(args.candidate_rows_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "method": str(args.method),
        "score_source": str(args.score_source),
        "record_grouping": str(args.record_grouping),
        "written_rows": int(written),
        "record_count": len(records),
        "coverage_count": len(coverage_by_record),
        "scorer_metadata": scorer_metadata,
        "high_score_threshold": float(args.high_score_threshold),
        "low_score_threshold": float(args.low_score_threshold),
        "contradiction_threshold": float(args.contradiction_threshold),
        **summarize(args.output_jsonl),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
