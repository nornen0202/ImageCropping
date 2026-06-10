#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mobilecropnet_v4.data import SUBJECT_MODE_VOCAB, target_ar_value
from scripts.build_mobilecropnet_v4_product_ar_score_labels import (
    build_listwise_rows,
    build_pairwise_rows,
    write_json,
    write_jsonl,
)
from scripts.export_mobilecropnet_v4_product_ar_candidates import clamp_box, safe_float


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def box_ar_error(box: Sequence[float] | None, target_ar: str) -> float:
    expected = target_ar_value(target_ar)
    if expected is None:
        return 0.0
    x1, y1, x2, y2 = clamp_box(box)
    ar = max(1e-6, x2 - x1) / max(1e-6, y2 - y1)
    return abs(math.log(ar / max(float(expected), 1e-6)))


def xyxy_to_cxcywh(box: Sequence[float] | None) -> list[float]:
    x1, y1, x2, y2 = clamp_box(box)
    return [0.5 * (x1 + x2), 0.5 * (y1 + y2), max(0.0, x2 - x1), max(0.0, y2 - y1)]


def iter_jsonl(path: Path, max_rows: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows > 0 and idx >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def subject_mode_id(value: str | None) -> int:
    text = str(value or "other_ambiguous")
    try:
        return SUBJECT_MODE_VOCAB.index(text)
    except ValueError:
        return SUBJECT_MODE_VOCAB.index("other_ambiguous")


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))


def _score_prob(row: dict[str, Any]) -> float:
    return clamp(safe_float(row.get("sstk_score_prob", row.get("score_prob", 0.0)), 0.0))


def _crop_prob(row: dict[str, Any]) -> float:
    return clamp(safe_float(row.get("sstk_crop_utility_prob", row.get("crop_utility_prob", _score_prob(row))), 0.0))


def _score_rank_pct(row: dict[str, Any]) -> float:
    return clamp(safe_float(row.get("sstk_score_rank_pct", row.get("score_rank_pct", 0.0)), 0.0))


def _crop_rank_pct(row: dict[str, Any]) -> float:
    return clamp(safe_float(row.get("sstk_crop_utility_rank_pct", row.get("crop_utility_rank_pct", _score_rank_pct(row))), 0.0))


def _candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    target_ar = str(row.get("target_ar", "FREE"))
    crop_prob = _crop_prob(row)
    score_prob = _score_prob(row)
    crop_rank = _crop_rank_pct(row)
    score_rank = _score_rank_pct(row)
    candidate = {
        "candidate_id": str(row.get("candidate_id", "")),
        "bbox_norm_xyxy": clamp_box(row.get("bbox_norm_xyxy")),
        "bbox_cxcywh": xyxy_to_cxcywh(row.get("bbox_norm_xyxy")),
        "score_prob": score_prob,
        "crop_utility_prob": crop_prob,
        "score_rank_pct": score_rank,
        "crop_utility_rank_pct": crop_rank,
        "rank_pct": crop_rank,
        "candidate_weight": clamp(safe_float(row.get("sstk_candidate_weight", 1.0), 1.0), 0.05, 1.0),
        "training_bucket": str(row.get("sstk_training_bucket", "")),
        "reject_tags": list(row.get("sstk_reject_tags") or []),
        "is_positive_candidate": bool(row.get("sstk_is_positive_candidate", False)),
        "is_hard_negative": bool(row.get("sstk_is_hard_negative", False)),
        "is_unsafe_negative": bool(row.get("sstk_is_unsafe_negative", False)),
        "is_ignore_candidate": bool(row.get("sstk_is_ignore_candidate", False)),
        "is_overflow_candidate": bool(row.get("sstk_is_overflow_candidate", False)),
        "checklist_labels": dict(row.get("checklist_labels") or {}),
        "checklist_scores": dict(row.get("checklist_scores") or {}),
        "why_tags": list(row.get("why_tags") or []),
        "macro_targets": {
            "A_macro": clamp(safe_float(row.get("profile_A_macro", 0.0), 0.0)),
            "S_macro": clamp(safe_float(row.get("profile_S_macro", 0.0), 0.0)),
            "C_macro": clamp(safe_float(row.get("profile_C_macro", 0.0), 0.0)),
            "T_macro": clamp(safe_float(row.get("profile_T_macro", 0.0), 0.0)),
        },
        "score_targets": {
            "score_prob": score_prob,
            "crop_utility_prob": crop_prob,
            "score_rank_pct": score_rank,
            "crop_utility_rank_pct": crop_rank,
            "rank_pct": crop_rank,
            "pseudo_mos_1to5": 1.0 + 4.0 * crop_prob,
            "target_ar_log_error": box_ar_error(row.get("bbox_norm_xyxy"), target_ar),
            "crop_utility_raw": crop_prob,
            "score_raw_policy": crop_prob,
        },
        "source_candidate_role": str(row.get("candidate_role", "")),
        "gt_flag": int(bool(row.get("gt_flag", 0))),
    }
    return candidate


def _sort_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(candidate.get("crop_utility_rank_pct", candidate.get("score_rank_pct", 0.0))),
        float(candidate.get("crop_utility_prob", candidate.get("score_prob", 0.0))),
        str(candidate.get("candidate_id", "")),
    )


def _source_routing_from_row(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("source_subject_mode") or row.get("subject_mode") or "other_ambiguous")
    routing = {
        "subject_mode": mode,
        "subject_mode_id": int(max(0, safe_float(row.get("source_subject_mode_id", subject_mode_id(mode)), subject_mode_id(mode)))),
        "route_conf": 1.0,
        "subject_mode_conf": 1.0,
        "mode_bucket": str(row.get("source_mode_bucket") or row.get("mode_bucket") or mode),
    }
    if row.get("source_policy_id"):
        routing["policy_id"] = str(row.get("source_policy_id"))
    if row.get("source_router_rule_id"):
        routing["router_rule_id"] = str(row.get("source_router_rule_id"))
    if isinstance(row.get("source_subject_prior_bbox_norm_xyxy"), list):
        routing["subject_prior_bbox_norm_xyxy"] = clamp_box(row.get("source_subject_prior_bbox_norm_xyxy"))
    reliability = safe_float(row.get("source_subject_reliability"), -1.0)
    if reliability >= 0.0:
        routing["flags"] = {"subject_reliability": clamp(reliability)}
    return routing


def _normalized_decision(
    source_decision: dict[str, Any],
    *,
    winner: dict[str, Any],
) -> dict[str, Any]:
    decision = dict(source_decision)
    decision.update(
        {
            "winner_post_gate_candidate_id": str(winner.get("candidate_id", "")),
            "winner_post_gate_bbox": winner.get("bbox_norm_xyxy"),
            "chosen_score_rank": float(winner.get("crop_utility_rank_pct", winner.get("score_rank_pct", 0.0))),
            "chosen_crop_utility_raw": float(winner.get("crop_utility_prob", winner.get("score_prob", 0.0))),
        }
    )
    if decision.get("decision_type") is None:
        decision["decision_type"] = "crop"
    if decision.get("decision_id") is None:
        decision["decision_id"] = 2
    return decision


def build_batch_rows(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    score_source_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[_group_key(row)].append(dict(row))

    out: list[dict[str, Any]] = []
    for (_, target_ar), rows in sorted(grouped.items()):
        first = rows[0]
        candidates = [_candidate_from_row(row) for row in rows]
        matching_targets = [c for c in candidates if bool(c.get("is_positive_candidate"))]
        ignored_candidates = [c for c in candidates if bool(c.get("is_ignore_candidate"))]
        overflow_candidates = [c for c in candidates if bool(c.get("is_overflow_candidate"))]
        candidate_pool = [
            c
            for c in candidates
            if not bool(c.get("is_positive_candidate"))
            and not bool(c.get("is_ignore_candidate"))
            and not bool(c.get("is_overflow_candidate"))
        ]
        matching_targets.sort(key=_sort_key, reverse=True)
        candidate_pool.sort(key=_sort_key, reverse=True)
        ignored_candidates.sort(key=_sort_key, reverse=True)
        overflow_candidates.sort(key=_sort_key, reverse=True)
        winner = matching_targets[0] if matching_targets else (candidate_pool[0] if candidate_pool else None)
        source_label_generation = dict(first.get("source_label_generation") or {})
        source_decision = dict(first.get("source_decision_target") or {})
        source_baseline = dict(first.get("source_baseline") or {})
        row = {
            "schema_version": "sstk_conditional_detr_batch_v3",
            "image_id": str(first.get("image_id", "")),
            "image_path": str(first.get("image_path", "")),
            "target_ar": target_ar,
            "label_generation": {
                "source": str(score_source_name),
                "source_parent": source_label_generation,
                "safe_leftover_policy": source_label_generation.get("safe_leftover_policy", "ignore"),
                "reconstructed_from_candidate_rows": True,
            },
            "routing": _source_routing_from_row(first),
            "matching_targets": matching_targets,
            "candidate_pool": candidate_pool,
            "ignored_candidates": ignored_candidates,
            "overflow_candidates": overflow_candidates,
            "teacher_meta": {
                "score_source": str(score_source_name),
                "candidate_count": len(candidates),
                "selectable_candidate_count": sum(1 for c in candidates if not bool(c.get("is_unsafe_negative"))),
                "reconstructed_from_candidate_rows": True,
            },
            "score_semantics": {
                "official_score_name": str(score_source_name),
                "official_score_field": "crop_utility_raw",
                "official_prob_name": str(score_source_name),
                "official_prob_field": "crop_utility_prob",
                "decision_semantics": "baseline_relative",
            },
        }
        if source_baseline:
            row["baseline"] = source_baseline
        if winner is not None:
            row["decision_target"] = _normalized_decision(source_decision, winner=winner)
        out.append(row)
    return out


def build_batch_row(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    score_source_name: str,
) -> dict[str, Any]:
    rows = build_batch_rows(candidate_rows, score_source_name=score_source_name)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one grouped row, got {len(rows)}")
    return rows[0]


def iter_grouped_candidate_rows(candidate_rows: Iterable[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[str, str] | None = None
    current_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        key = _group_key(row)
        if current_key is None:
            current_key = key
            current_rows = [dict(row)]
            continue
        if key == current_key:
            current_rows.append(dict(row))
            continue
        if key in seen_keys:
            raise ValueError(f"candidate rows are not grouped by image_id/target_ar; repeated key after flush: {key}")
        seen_keys.add(current_key)
        yield current_rows
        current_key = key
        current_rows = [dict(row)]
    if current_key is not None and current_rows:
        if current_key in seen_keys:
            raise ValueError(f"candidate rows are not grouped by image_id/target_ar; repeated key at final flush: {current_key}")
        yield current_rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    for row in rows:
        update_summary_counters(counters, row)
    return {
        "row_count": len(rows),
        "candidate_count_mean": float(counters["candidate_count"] / len(rows)) if rows else 0.0,
        "counter": dict(sorted(counters.items())),
    }


def update_summary_counters(counters: Counter[str], row: dict[str, Any]) -> None:
        counters[f"target_ar::{row.get('target_ar', 'FREE')}"] += 1
        candidates = (
            list(row.get("matching_targets") or [])
            + list(row.get("candidate_pool") or [])
            + list(row.get("ignored_candidates") or [])
            + list(row.get("overflow_candidates") or [])
        )
        counters["candidate_count"] += len(candidates)
        counters["positive_count"] += len(row.get("matching_targets") or [])
        counters["ignored_count"] += len(row.get("ignored_candidates") or [])
        counters["overflow_count"] += len(row.get("overflow_candidates") or [])


def write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild MobileCropNet v4 Product AR T1/base batch labels from flattened candidate rows.")
    parser.add_argument("--candidate_rows_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--output_pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--output_listwise_jsonl", type=Path, default=None)
    parser.add_argument("--score_source_name", default="sstk_teacher_t1_product_ar")
    parser.add_argument("--pairwise_score_margin", type=float, default=0.03)
    parser.add_argument("--max_pairs_per_image", type=int, default=48)
    parser.add_argument("--listwise_temperature", type=float, default=0.12)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row_count = 0
    counters: Counter[str] = Counter()
    pairwise_count = 0
    listwise_count = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_pairwise_jsonl is not None:
        args.output_pairwise_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_listwise_jsonl is not None:
        args.output_listwise_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        pairwise_handle = args.output_pairwise_jsonl.open("w", encoding="utf-8") if args.output_pairwise_jsonl is not None else None
        listwise_handle = args.output_listwise_jsonl.open("w", encoding="utf-8") if args.output_listwise_jsonl is not None else None
        try:
            grouped_rows = iter_grouped_candidate_rows(iter_jsonl(args.candidate_rows_jsonl, max_rows=int(args.max_rows)))
            for grouped in grouped_rows:
                row = build_batch_row(grouped, score_source_name=str(args.score_source_name))
                write_jsonl_line(output_handle, row)
                row_count += 1
                update_summary_counters(counters, row)
                if pairwise_handle is not None:
                    for pairwise_row in build_pairwise_rows(
                        [row],
                        source=str(args.score_source_name),
                        max_pairs_per_image=int(args.max_pairs_per_image),
                        score_margin=float(args.pairwise_score_margin),
                    ):
                        write_jsonl_line(pairwise_handle, pairwise_row)
                        pairwise_count += 1
                if listwise_handle is not None:
                    for listwise_row in build_listwise_rows([row], source=str(args.score_source_name), temperature=float(args.listwise_temperature)):
                        write_jsonl_line(listwise_handle, listwise_row)
                        listwise_count += 1
        finally:
            if pairwise_handle is not None:
                pairwise_handle.close()
            if listwise_handle is not None:
                listwise_handle.close()
    summary = {
        "status": "ok",
        "candidate_rows_jsonl": str(args.candidate_rows_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_pairwise_jsonl": str(args.output_pairwise_jsonl) if args.output_pairwise_jsonl is not None else None,
        "output_listwise_jsonl": str(args.output_listwise_jsonl) if args.output_listwise_jsonl is not None else None,
        "written_rows": int(row_count),
        "pairwise_rows": int(pairwise_count),
        "listwise_rows": int(listwise_count),
        "score_source_name": str(args.score_source_name),
        "row_count": int(row_count),
        "candidate_count_mean": float(counters["candidate_count"] / row_count) if row_count else 0.0,
        "counter": dict(sorted(counters.items())),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
