#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator


RISK_KEYS = ("is_hard_negative", "is_unsafe_negative", "is_ignore_candidate", "is_overflow_candidate")
SCORE_KEYS = ("crop_utility_prob", "score_prob", "crop_utility_rank_pct", "score_rank_pct")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _score(candidate: dict[str, Any]) -> float:
    for key in SCORE_KEYS:
        if key in candidate:
            return max(0.0, min(1.0, _safe_float(candidate.get(key))))
    targets = candidate.get("score_targets")
    if isinstance(targets, dict):
        for key in SCORE_KEYS:
            if key in targets:
                return max(0.0, min(1.0, _safe_float(targets.get(key))))
    return 0.0


def _candidate_id(candidate: dict[str, Any], fallback: str) -> str:
    cid = str(candidate.get("candidate_id") or "").strip()
    return cid or fallback


def _is_risky(candidate: dict[str, Any]) -> bool:
    return any(bool(candidate.get(key)) for key in RISK_KEYS)


def _collect_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
        for idx, candidate in enumerate(row.get(field) or []):
            if not isinstance(candidate, dict):
                continue
            cid = _candidate_id(candidate, f"{field}_{idx}")
            if cid in seen:
                continue
            seen.add(cid)
            item = dict(candidate)
            item["candidate_id"] = cid
            out.append(item)
    return out


def _row_pairs(
    row: dict[str, Any],
    *,
    max_pairs_per_row: int,
    min_score_gap: float,
    include_risk_pairs: bool,
) -> list[dict[str, Any]]:
    candidates = _collect_candidates(row)
    valid = [c for c in candidates if _score(c) > 0.0]
    if len(valid) < 2:
        return []
    safe = [c for c in valid if not _is_risky(c)]
    ranking_pool = safe or valid
    ranking_pool.sort(key=lambda c: (_score(c), str(c.get("candidate_id", ""))), reverse=True)
    best = ranking_pool[0]
    best_score = _score(best)
    rows: list[dict[str, Any]] = []
    opponents = [c for c in valid if str(c.get("candidate_id")) != str(best.get("candidate_id"))]
    opponents.sort(key=lambda c: (_is_risky(c), abs(best_score - _score(c)), _score(c)), reverse=True)
    for candidate in opponents:
        gap = max(0.0, best_score - _score(candidate))
        if gap < min_score_gap and not (_is_risky(candidate) and include_risk_pairs):
            continue
        pair_type = "top1_return_vs_risk" if _is_risky(candidate) else "top1_return_vs_candidate"
        weight = max(0.10, min(1.0, 0.35 + 4.0 * gap))
        if pair_type.startswith("top1"):
            weight = max(weight, 0.75)
        rows.append(
            {
                "image_id": row.get("image_id", row.get("image_path", "")),
                "target_ar": row.get("target_ar", "FREE"),
                "candidate_id_a": best.get("candidate_id"),
                "candidate_id_b": candidate.get("candidate_id"),
                "label": 1,
                "pair_type": pair_type,
                "score_margin": gap,
                "crop_utility_margin": gap,
                "weight": weight,
                "best_score": best_score,
                "other_score": _score(candidate),
            }
        )
        if len(rows) >= max_pairs_per_row:
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build return/action explicit pairwise labels from MobileCropNet v4 JSONL labels.")
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--max_pairs_per_row", type=int, default=8)
    parser.add_argument("--min_score_gap", type=float, default=0.02)
    parser.add_argument("--include_risk_pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    rows_read = 0
    pair_rows = 0
    row_with_pairs = 0
    score_gap_sum = 0.0
    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for row in _iter_jsonl(args.input_jsonl):
            if args.max_rows > 0 and rows_read >= int(args.max_rows):
                break
            rows_read += 1
            pairs = _row_pairs(
                row,
                max_pairs_per_row=max(1, int(args.max_pairs_per_row)),
                min_score_gap=max(0.0, float(args.min_score_gap)),
                include_risk_pairs=bool(args.include_risk_pairs),
            )
            if pairs:
                row_with_pairs += 1
            for pair in pairs:
                out_f.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
                pair_rows += 1
                score_gap_sum += _safe_float(pair.get("score_margin"), 0.0)
    summary = {
        "schema_version": "mobilecropnet_v4_return_pairwise_labels_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "rows_read": rows_read,
        "row_with_pairs": row_with_pairs,
        "pair_rows": pair_rows,
        "pairs_per_row_mean": float(pair_rows / max(1, rows_read)),
        "score_gap_mean": float(score_gap_sum / max(1, pair_rows)),
        "max_pairs_per_row": max(1, int(args.max_pairs_per_row)),
        "min_score_gap": max(0.0, float(args.min_score_gap)),
        "include_risk_pairs": bool(args.include_risk_pairs),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
