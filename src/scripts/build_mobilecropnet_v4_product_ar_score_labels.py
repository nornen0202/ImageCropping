#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import target_ar_value


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


def clamp_box(box: Sequence[float] | None) -> list[float]:
    if not box or len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [clamp(safe_float(v), 0.0, 1.0) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [x1, y1, x2, y2]


def xyxy_to_cxcywh(box: Sequence[float] | None) -> list[float]:
    x1, y1, x2, y2 = clamp_box(box)
    return [0.5 * (x1 + x2), 0.5 * (y1 + y2), max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_ar_error(box: Sequence[float] | None, target_ar: str) -> float:
    expected = target_ar_value(target_ar)
    if expected is None:
        return 0.0
    x1, y1, x2, y2 = clamp_box(box)
    ar = max(1e-6, x2 - x1) / max(1e-6, y2 - y1)
    return abs(math.log(ar / max(float(expected), 1e-6)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path, max_rows: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows > 0 and idx >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


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


def score_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")), str(row.get("candidate_id", "")))


def candidate_key(batch_row: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (str(batch_row.get("image_id", "")), str(batch_row.get("target_ar", "FREE")), str(candidate.get("candidate_id", "")))


def score_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if row.get(key) is not None:
            return safe_float(row.get(key), default)
    return float(default)


def candidate_score(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    if candidate.get(key) is not None:
        return safe_float(candidate.get(key), default)
    targets = candidate.get("score_targets")
    if isinstance(targets, dict) and targets.get(key) is not None:
        return safe_float(targets.get(key), default)
    return float(default)


def score_adjustment(
    scored_rows: list[dict[str, Any]],
    *,
    unsafe_score_cap: float,
    hard_reject_policy: str,
) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[(str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))].append(row)
    for rows in grouped.values():
        safe_rows = []
        for row in rows:
            fatal = bool(row.get("fatal_flag") or row.get("sstk_hard_reject"))
            if fatal and hard_reject_policy in {"cap_and_demote", "exclude_positive"}:
                continue
            safe_rows.append(row)
        if safe_rows:
            sorted_safe = sorted(
                safe_rows,
                key=lambda row: (
                    score_value(row, "ranker_score"),
                    score_value(row, "score_rank_pct_by_group", "score_rank_pct_by_image"),
                    str(row.get("candidate_id", "")),
                ),
            )
            denom = float(max(1, len(sorted_safe) - 1))
            for rank, row in enumerate(sorted_safe):
                adjusted = 1.0 if len(sorted_safe) == 1 else float(rank) / denom
                out[score_row_key(row)] = clamp(adjusted)
        for row in rows:
            key = score_row_key(row)
            if key in out:
                continue
            raw = clamp(score_value(row, "score_rank_pct_by_group", "score_rank_pct_by_image"))
            if bool(row.get("fatal_flag") or row.get("sstk_hard_reject")):
                raw = min(raw, float(unsafe_score_cap))
            out[key] = clamp(raw)
    return out


def is_hard_reject(scored: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    if scored is not None and bool(scored.get("fatal_flag") or scored.get("sstk_hard_reject")):
        return True
    if bool(candidate.get("is_hard_negative") or candidate.get("is_unsafe_negative") or candidate.get("is_overflow_candidate")):
        return True
    reject_tags = {str(tag) for tag in (candidate.get("reject_tags") or [])}
    hard_tags = {
        "face_cut",
        "head_top_cut",
        "headroom_violation",
        "joint_cutoff",
        "lookroom_violation",
        "person_cut",
        "subject_cutoff",
        "text_cut",
        "sstk_hard_negative",
        "sstk_unsafe_negative",
    }
    return bool(reject_tags.intersection(hard_tags))


def select_positive(
    batch_row: dict[str, Any],
    candidates: list[dict[str, Any]],
    scored_by_key: dict[tuple[str, str, str], dict[str, Any]],
    adjusted: dict[tuple[str, str, str], float],
    *,
    fallback_policy: str,
) -> tuple[set[str], str]:
    ranked: list[tuple[float, float, str, dict[str, Any], dict[str, Any] | None]] = []
    for candidate in candidates:
        key = candidate_key(batch_row, candidate)
        scored = scored_by_key.get(key)
        adj = adjusted.get(key, 0.0)
        raw = score_value(scored or {}, "ranker_score", default=candidate_score(candidate, "crop_utility_prob", 0.0))
        ranked.append((adj, raw, str(candidate.get("candidate_id", "")), candidate, scored))
    safe = [item for item in ranked if not is_hard_reject(item[4], item[3])]
    if safe:
        best = max(safe, key=lambda item: (item[0], item[1], item[2]))
        return {best[2]}, "external_best_safe"
    if fallback_policy == "sstk_positive":
        positives = [
            str(candidate.get("candidate_id", ""))
            for candidate in candidates
            if candidate.get("is_positive_candidate") or candidate in (batch_row.get("matching_targets") or [])
        ]
        if positives:
            return {positives[0]}, "fallback_sstk_positive_all_external_unsafe"
    if ranked:
        best = max(ranked, key=lambda item: (item[0], item[1], item[2]))
        return {best[2]}, "fallback_external_all_unsafe"
    return set(), "empty_group"


def merge_candidate(
    *,
    batch_row: dict[str, Any],
    candidate: dict[str, Any],
    scored: dict[str, Any] | None,
    adjusted_score: float,
    is_positive: bool,
    positive_reason: str,
    score_source_name: str,
    unsafe_score_cap: float,
) -> dict[str, Any]:
    out = json.loads(json.dumps(candidate, ensure_ascii=False))
    target_ar = str(batch_row.get("target_ar", "FREE"))
    hard_reject = is_hard_reject(scored, candidate)
    raw_rank_pct = score_value(scored or {}, "score_rank_pct_by_group", "score_rank_pct_by_image", default=adjusted_score)
    raw_ranker_score = score_value(scored or {}, "ranker_score", default=candidate_score(candidate, "crop_utility_prob", 0.0))
    external_score = clamp(adjusted_score)
    if hard_reject and not is_positive:
        external_score = min(external_score, float(unsafe_score_cap))
    score_targets = dict(out.get("score_targets") if isinstance(out.get("score_targets"), dict) else {})
    score_targets.update(
        {
            "sstk_score_prob": candidate_score(candidate, "score_prob", candidate_score(candidate, "crop_utility_prob", 0.0)),
            "sstk_crop_utility_prob": candidate_score(candidate, "crop_utility_prob", candidate_score(candidate, "score_prob", 0.0)),
            "sstk_score_rank_pct": candidate_score(candidate, "score_rank_pct", candidate_score(candidate, "rank_pct", 0.0)),
            "sstk_crop_utility_rank_pct": candidate_score(candidate, "crop_utility_rank_pct", candidate_score(candidate, "rank_pct", 0.0)),
            "raw_external_score_rank_pct": clamp(raw_rank_pct),
            "raw_external_ranker_score": raw_ranker_score,
            "external_score_rank_pct": external_score,
            "score_rank_pct": external_score,
            "crop_utility_rank_pct": external_score,
            "rank_pct": external_score,
            "score_prob": external_score,
            "crop_utility_prob": external_score,
            "pseudo_mos_1to5": 1.0 + 4.0 * external_score,
            "target_ar_log_error": box_ar_error(out.get("bbox_norm_xyxy"), target_ar),
        }
    )
    out["score_targets"] = score_targets
    out["score_prob"] = external_score
    out["crop_utility_prob"] = external_score
    out["score_rank_pct"] = external_score
    out["crop_utility_rank_pct"] = external_score
    out["is_positive_candidate"] = bool(is_positive)
    out["is_soft_positive"] = bool((not is_positive) and external_score >= 0.85 and not hard_reject)
    out["is_hard_negative"] = bool((not is_positive) and (hard_reject or out.get("is_hard_negative", False)))
    out["is_unsafe_negative"] = bool((not is_positive) and (hard_reject or out.get("is_unsafe_negative", False)))
    tier = str((scored or {}).get("label_quality_tier") or ("main_positive" if is_positive else "candidate"))
    if hard_reject and not is_positive:
        tier = "external_high_sstk_hard_reject_demoted" if raw_rank_pct >= 0.85 else "sstk_hard_reject"
    out["training_bucket"] = tier
    base_weight = clamp(score_value(scored or {}, "training_weight", default=out.get("candidate_weight", out.get("training_weight", 1.0))), 0.05, 1.0)
    if is_positive:
        out["candidate_weight"] = max(base_weight, 0.75)
    elif hard_reject:
        out["candidate_weight"] = max(base_weight, 0.85)
    else:
        out["candidate_weight"] = base_weight
    out.setdefault("reject_tags", [])
    if hard_reject and "external_score_hard_reject_demoted" not in out["reject_tags"]:
        out["reject_tags"] = list(out["reject_tags"]) + ["external_score_hard_reject_demoted"]
    out["external_score_teacher"] = {
        "score_source": score_source_name,
        "ranker_method": str((scored or {}).get("ranker_method", "")),
        "ranker_score": raw_ranker_score,
        "raw_score_rank_pct": clamp(raw_rank_pct),
        "score_rank_pct": external_score,
        "selected_by_ranker": bool((scored or {}).get("selected_by_ranker", False)),
        "selected_by_converter": bool(is_positive),
        "positive_reason": positive_reason,
        "label_quality_tier": str((scored or {}).get("label_quality_tier", "")),
        "fatal_flag": bool((scored or {}).get("fatal_flag", False)),
        "contradiction_flag": bool((scored or {}).get("contradiction_flag", False)),
        "explanation_score": score_value(scored or {}, "explanation_score", default=0.0),
        "explanation_confidence": score_value(scored or {}, "explanation_confidence", default=0.0),
        "warnings": list((scored or {}).get("warnings") or []),
        "sstk_hard_reject": bool((scored or {}).get("sstk_hard_reject", hard_reject)),
    }
    return out


def iter_merged_rows(
    input_jsonl: Path,
    scored_by_key: dict[tuple[str, str, str], dict[str, Any]],
    adjusted: dict[tuple[str, str, str], float],
    *,
    score_source_name: str,
    fallback_policy: str,
    unsafe_score_cap: float,
    max_rows: int,
) -> Iterable[dict[str, Any]]:
    for batch_row in iter_jsonl(input_jsonl, max_rows=max_rows):
        out = json.loads(json.dumps(batch_row, ensure_ascii=False))
        candidates = list(batch_row.get("matching_targets") or []) + list(batch_row.get("candidate_pool") or [])
        positive_ids, positive_reason = select_positive(
            batch_row,
            candidates,
            scored_by_key,
            adjusted,
            fallback_policy=fallback_policy,
        )
        merged: list[dict[str, Any]] = []
        for candidate in candidates:
            key = candidate_key(batch_row, candidate)
            cid = str(candidate.get("candidate_id", ""))
            merged.append(
                merge_candidate(
                    batch_row=batch_row,
                    candidate=candidate,
                    scored=scored_by_key.get(key),
                    adjusted_score=adjusted.get(key, 0.0),
                    is_positive=cid in positive_ids,
                    positive_reason=positive_reason,
                    score_source_name=score_source_name,
                    unsafe_score_cap=float(unsafe_score_cap),
                )
            )
        merged.sort(
            key=lambda c: (
                1 if c.get("is_positive_candidate") else 0,
                safe_float(c.get("score_targets", {}).get("crop_utility_rank_pct"), 0.0),
                str(c.get("candidate_id", "")),
            ),
            reverse=True,
        )
        matching = [c for c in merged if c.get("is_positive_candidate")]
        pool = [c for c in merged if not c.get("is_positive_candidate")]
        out["matching_targets"] = matching
        out["candidate_pool"] = pool
        source_label_generation = dict(out.get("label_generation") if isinstance(out.get("label_generation"), dict) else {})
        out["label_generation"] = {
            **source_label_generation,
            "source": score_source_name,
            "source_parent": source_label_generation.get("source", ""),
            "score_replacement": True,
            "positive_selection_policy": "best_safe_external_with_sstk_fallback",
            "fallback_policy": fallback_policy,
            "unsafe_score_cap": float(unsafe_score_cap),
            "positive_reason": positive_reason,
        }
        top = matching[0] if matching else (merged[0] if merged else None)
        if top is not None:
            decision = dict(out.get("decision_target") if isinstance(out.get("decision_target"), dict) else {})
            decision.update(
                {
                    "winner_post_gate_candidate_id": str(top.get("candidate_id", "")),
                    "winner_post_gate_bbox": top.get("bbox_norm_xyxy"),
                    "chosen_score_rank": safe_float(top.get("score_targets", {}).get("crop_utility_rank_pct"), 0.0),
                    "chosen_crop_utility_raw": safe_float(top.get("score_targets", {}).get("raw_external_ranker_score"), 0.0),
                    "external_score_source": score_source_name,
                }
            )
            if decision.get("decision_type") is None:
                decision["decision_type"] = "crop"
            if decision.get("decision_id") is None:
                decision["decision_id"] = 2
            out["decision_target"] = decision
        teacher_meta = dict(out.get("teacher_meta") if isinstance(out.get("teacher_meta"), dict) else {})
        teacher_meta.update(
            {
                "score_source": score_source_name,
                "score_replacement": True,
                "positive_reason": positive_reason,
                "candidate_count": len(merged),
                "selectable_candidate_count": sum(1 for c in merged if not c.get("is_unsafe_negative")),
            }
        )
        out["teacher_meta"] = teacher_meta
        score_semantics = dict(out.get("score_semantics") if isinstance(out.get("score_semantics"), dict) else {})
        score_semantics.update(
            {
                "official_score_name": score_source_name,
                "official_score_field": "score_rank_pct",
                "official_prob_name": score_source_name,
                "official_prob_field": "crop_utility_prob",
                "decision_semantics": "product_ar_external_score_replacement_preserve_policy",
            }
        )
        out["score_semantics"] = score_semantics
        yield out


def softmax(values: Sequence[float], temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(1e-4, float(temperature))
    scaled = [float(v) / temp for v in values]
    m = max(scaled)
    exp = [math.exp(v - m) for v in scaled]
    denom = sum(exp) or 1.0
    return [v / denom for v in exp]


def candidates_for_rank(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("matching_targets") or []) + list(row.get("candidate_pool") or [])


def build_listwise_rows(rows: Sequence[dict[str, Any]], *, source: str, temperature: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        candidates = candidates_for_rank(row)
        scores = [safe_float(c.get("score_targets", {}).get("crop_utility_rank_pct", c.get("crop_utility_rank_pct")), 0.0) for c in candidates]
        probs = softmax(scores, temperature)
        out.append(
            {
                "image_id": str(row.get("image_id", "")),
                "target_ar": str(row.get("target_ar", "FREE")),
                "source": source,
                "candidates": [
                    {
                        "candidate_id": str(c.get("candidate_id", "")),
                        "score_rank_pct": float(score),
                        "crop_utility_rank_pct": float(score),
                        "rank_pct": float(score),
                        "score_softmax_local": float(prob),
                        "crop_utility_softmax_local": float(prob),
                        "teacher_softmax_local": float(prob),
                        "is_positive_candidate": bool(c.get("is_positive_candidate", False)),
                        "is_hard_negative": bool(c.get("is_hard_negative", False)),
                        "is_unsafe_negative": bool(c.get("is_unsafe_negative", False)),
                    }
                    for c, score, prob in zip(candidates, scores, probs)
                ],
            }
        )
    return out


def build_pairwise_rows(rows: Sequence[dict[str, Any]], *, source: str, max_pairs_per_image: int, score_margin: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        candidates = sorted(
            candidates_for_rank(row),
            key=lambda c: (
                safe_float(c.get("score_targets", {}).get("crop_utility_rank_pct", c.get("crop_utility_rank_pct")), 0.0),
                1.0 if c.get("is_positive_candidate") else 0.0,
                str(c.get("candidate_id", "")),
            ),
            reverse=True,
        )
        pair_count = 0
        for i, better in enumerate(candidates):
            if pair_count >= int(max_pairs_per_image):
                break
            better_score = safe_float(better.get("score_targets", {}).get("crop_utility_rank_pct", better.get("crop_utility_rank_pct")), 0.0)
            for worse in candidates[i + 1 :]:
                worse_score = safe_float(worse.get("score_targets", {}).get("crop_utility_rank_pct", worse.get("crop_utility_rank_pct")), 0.0)
                margin = better_score - worse_score
                if margin < float(score_margin):
                    continue
                out.append(
                    {
                        "image_id": str(row.get("image_id", "")),
                        "target_ar": str(row.get("target_ar", "FREE")),
                        "candidate_id_a": str(better.get("candidate_id", "")),
                        "candidate_id_b": str(worse.get("candidate_id", "")),
                        "label": 1,
                        "score_margin": float(margin),
                        "crop_utility_margin": float(margin),
                        "pair_type": "top1_vs_negative" if worse.get("is_hard_negative") or worse.get("is_unsafe_negative") else "rank_margin",
                        "source": source,
                    }
                )
                pair_count += 1
                if pair_count >= int(max_pairs_per_image):
                    break
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    for row in rows:
        target_ar = str(row.get("target_ar", "FREE"))
        counters[f"target_ar::{target_ar}"] += 1
        candidates = candidates_for_rank(row)
        counters["candidate_count"] += len(candidates)
        if row.get("label_generation", {}).get("positive_reason"):
            counters[f"positive_reason::{row['label_generation']['positive_reason']}"] += 1
        for candidate in candidates:
            bucket = str(candidate.get("training_bucket", ""))
            counters[f"bucket::{bucket}"] += 1
            if candidate.get("is_positive_candidate"):
                counters[f"positive_bucket::{bucket}"] += 1
            if candidate.get("is_unsafe_negative"):
                counters["unsafe_negative"] += 1
    return {
        "row_count": len(rows),
        "counter": dict(sorted(counters.items())),
        "candidate_count_mean": float(counters["candidate_count"] / len(rows)) if rows else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Product AR/full MobileCropNet v4 labels by replacing SSTK score targets with external T6/public scores.")
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--scored_candidates_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--output_pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--output_listwise_jsonl", type=Path, default=None)
    parser.add_argument("--score_source_name", default="external_product_ar_score")
    parser.add_argument("--fallback_policy", choices=["sstk_positive", "external_any"], default="sstk_positive")
    parser.add_argument("--hard_reject_policy", choices=["cap_and_demote", "exclude_positive"], default="cap_and_demote")
    parser.add_argument("--unsafe_score_cap", type=float, default=0.05)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_pairs_per_image", type=int, default=48)
    parser.add_argument("--pairwise_score_margin", type=float, default=0.03)
    parser.add_argument("--listwise_temperature", type=float, default=0.12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored_rows = read_jsonl(args.scored_candidates_jsonl)
    scored_by_key = {score_row_key(row): row for row in scored_rows}
    adjusted = score_adjustment(
        scored_rows,
        unsafe_score_cap=float(args.unsafe_score_cap),
        hard_reject_policy=str(args.hard_reject_policy),
    )
    rows = list(
        iter_merged_rows(
            args.input_jsonl,
            scored_by_key,
            adjusted,
            score_source_name=str(args.score_source_name),
            fallback_policy=str(args.fallback_policy),
            unsafe_score_cap=float(args.unsafe_score_cap),
            max_rows=int(args.max_rows),
        )
    )
    written = write_jsonl(args.output_jsonl, rows)
    pairwise_count = 0
    listwise_count = 0
    if args.output_pairwise_jsonl is not None:
        pairwise_count = write_jsonl(
            args.output_pairwise_jsonl,
            build_pairwise_rows(
                rows,
                source=str(args.score_source_name),
                max_pairs_per_image=int(args.max_pairs_per_image),
                score_margin=float(args.pairwise_score_margin),
            ),
        )
    if args.output_listwise_jsonl is not None:
        listwise_count = write_jsonl(
            args.output_listwise_jsonl,
            build_listwise_rows(rows, source=str(args.score_source_name), temperature=float(args.listwise_temperature)),
        )
    summary = {
        "status": "ok",
        "input_jsonl": str(args.input_jsonl),
        "scored_candidates_jsonl": str(args.scored_candidates_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_pairwise_jsonl": str(args.output_pairwise_jsonl) if args.output_pairwise_jsonl is not None else None,
        "output_listwise_jsonl": str(args.output_listwise_jsonl) if args.output_listwise_jsonl is not None else None,
        "written_rows": int(written),
        "pairwise_rows": int(pairwise_count),
        "listwise_rows": int(listwise_count),
        "score_source_name": str(args.score_source_name),
        "fallback_policy": str(args.fallback_policy),
        "hard_reject_policy": str(args.hard_reject_policy),
        "unsafe_score_cap": float(args.unsafe_score_cap),
        **summarize(rows),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
