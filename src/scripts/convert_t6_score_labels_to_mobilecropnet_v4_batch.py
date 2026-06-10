#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


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


def xyxy_to_cxcywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = [safe_float(v) for v in box[:4]]
    return [0.5 * (x1 + x2), 0.5 * (y1 + y2), max(0.0, x2 - x1), max(0.0, y2 - y1)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
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


def resolve_image_path(image_id: str, image_roots: Sequence[Path]) -> Path:
    names = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png"]
    for root in image_roots:
        root = Path(root)
        for name in names:
            direct = root / name
            if direct.exists():
                return direct
        for sub in ("train", "val", "test", "images"):
            for name in names:
                nested = root / sub / name
                if nested.exists():
                    return nested
    return Path(image_roots[0]) / f"{image_id}.jpg" if image_roots else Path(f"{image_id}.jpg")


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def is_unsafe_t6_row(
    row: dict[str, Any],
    *,
    unsafe_warning_tags: set[str] | None = None,
    include_contradiction: bool = False,
) -> bool:
    tier = str(row.get("label_quality_tier", ""))
    warnings = {str(v) for v in (row.get("warnings") or [])}
    return bool(
        row.get("fatal_flag", False)
        or tier == "hard_negative"
        or (include_contradiction and row.get("contradiction_flag", False))
        or (unsafe_warning_tags is not None and bool(warnings.intersection(unsafe_warning_tags)))
    )


def adjusted_rank_scores(
    rows: Sequence[dict[str, Any]],
    *,
    policy: str,
    unsafe_score_cap: float,
    unsafe_warning_tags: set[str] | None,
    unsafe_include_contradiction: bool = False,
) -> dict[str, float]:
    by_id: dict[str, float] = {}
    if policy == "keep":
        for row in rows:
            by_id[str(row.get("candidate_id", ""))] = clamp(safe_float(row.get("score_rank_pct_by_image"), 0.0))
        return by_id
    if policy == "unsafe_cap":
        for row in rows:
            raw = clamp(safe_float(row.get("score_rank_pct_by_image"), 0.0))
            if is_unsafe_t6_row(row, unsafe_warning_tags=unsafe_warning_tags, include_contradiction=unsafe_include_contradiction):
                raw = min(raw, float(unsafe_score_cap))
            by_id[str(row.get("candidate_id", ""))] = clamp(raw)
        return by_id
    if policy != "safe_rescale":
        raise ValueError(f"unsupported score_adjustment_policy: {policy}")

    safe_rows = [row for row in rows if not is_unsafe_t6_row(row, unsafe_warning_tags=unsafe_warning_tags, include_contradiction=unsafe_include_contradiction)]
    safe_sorted = sorted(
        safe_rows,
        key=lambda row: (
            safe_float(row.get("score_rank_pct_by_image"), 0.0),
            safe_float(row.get("ranker_score"), 0.0),
            str(row.get("candidate_id", "")),
        ),
    )
    denom = float(max(1, len(safe_sorted) - 1))
    for rank, row in enumerate(safe_sorted):
        if len(safe_sorted) == 1:
            score = 1.0
        else:
            score = float(rank) / denom
        by_id[str(row.get("candidate_id", ""))] = clamp(score)
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if cid in by_id:
            continue
        raw = clamp(safe_float(row.get("score_rank_pct_by_image"), 0.0))
        by_id[cid] = clamp(min(raw, float(unsafe_score_cap)))
    return by_id


def raw_score_targets(
    rows: Sequence[dict[str, Any]],
    *,
    adjusted_scores: dict[str, float],
    policy: str,
    raw_score_temperature: float,
    raw_score_blend_weight: float,
    unsafe_score_cap: float,
    unsafe_warning_tags: set[str] | None,
    unsafe_include_contradiction: bool = False,
) -> dict[str, dict[str, float]]:
    raw_values = [safe_float(row.get("ranker_score"), 0.0) for row in rows]
    if not raw_values:
        return {}
    mean = float(sum(raw_values) / len(raw_values))
    var = float(sum((v - mean) ** 2 for v in raw_values) / max(1, len(raw_values)))
    std = math.sqrt(max(var, 1e-12))
    lo = min(raw_values)
    hi = max(raw_values)
    span = max(1e-6, hi - lo)
    temp = max(1e-4, float(raw_score_temperature))
    blend = clamp(float(raw_score_blend_weight), 0.0, 1.0)
    out: dict[str, dict[str, float]] = {}
    for row, raw in zip(rows, raw_values):
        cid = str(row.get("candidate_id", ""))
        rank_pct = clamp(adjusted_scores.get(cid, safe_float(row.get("score_rank_pct_by_image"), 0.0)))
        z = (raw - mean) / std
        z_prob = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, -z / temp))))
        minmax_prob = clamp((raw - lo) / span)
        if policy in {"rank_pct", "safe_rank_pct"}:
            target = rank_pct
        elif policy == "raw_zscore":
            target = z_prob
        elif policy == "blend_rank_zscore":
            target = (1.0 - blend) * rank_pct + blend * z_prob
        elif policy == "raw_minmax":
            target = minmax_prob
        elif policy == "blend_rank_minmax":
            target = (1.0 - blend) * rank_pct + blend * minmax_prob
        else:
            raise ValueError(f"unsupported score_target_policy: {policy}")
        if is_unsafe_t6_row(row, unsafe_warning_tags=unsafe_warning_tags, include_contradiction=unsafe_include_contradiction):
            target = min(target, float(unsafe_score_cap))
            z_prob = min(z_prob, float(unsafe_score_cap))
            minmax_prob = min(minmax_prob, float(unsafe_score_cap))
        out[cid] = {
            "score_target": clamp(target),
            "raw_ranker_zscore": float(z),
            "raw_ranker_zscore_prob": clamp(z_prob),
            "raw_ranker_minmax_prob": clamp(minmax_prob),
        }
    return out


def t6_candidate(
    row: dict[str, Any],
    *,
    top_score: float,
    top_ranker_score: float,
    is_positive: bool,
    adjusted_rank_pct: float | None = None,
    unsafe_warning_tags: set[str] | None = None,
    unsafe_include_contradiction: bool = False,
    unsafe_candidate_weight: float = 0.85,
    candidate_source: str = "t6_score_only_official",
    raw_target_payload: dict[str, float] | None = None,
) -> dict[str, Any]:
    raw_rank_pct = clamp(safe_float(row.get("score_rank_pct_by_image"), 0.0))
    rank_pct = clamp(raw_rank_pct if adjusted_rank_pct is None else float(adjusted_rank_pct))
    raw_payload = raw_target_payload or {}
    score_target = clamp(safe_float(raw_payload.get("score_target"), rank_pct))
    ranker_score = safe_float(row.get("ranker_score"), 0.0)
    margin = max(0.0, float(top_score) - score_target)
    ranker_margin = abs(float(top_ranker_score) - ranker_score)
    fatal = bool(row.get("fatal_flag", False))
    contradiction = bool(row.get("contradiction_flag", False))
    hard_negative = str(row.get("label_quality_tier", "")) == "hard_negative"
    unsafe = is_unsafe_t6_row(row, unsafe_warning_tags=unsafe_warning_tags, include_contradiction=unsafe_include_contradiction)
    weight = clamp(safe_float(row.get("training_weight"), 1.0), 0.05, 1.0)
    if unsafe and not is_positive:
        weight = max(weight, clamp(float(unsafe_candidate_weight), 0.05, 1.0))
    box = [clamp(safe_float(v), 0.0, 1.0) for v in (row.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])[:4]]
    score_targets = {
        "score_prob": score_target,
        "crop_utility_prob": score_target,
        "score_rank_pct": rank_pct,
        "crop_utility_rank_pct": rank_pct,
        "rank_pct": rank_pct,
        "raw_score_rank_pct": raw_rank_pct,
        "pseudo_mos_1to5": 1.0 + 4.0 * rank_pct,
        "t6_ranker_score": ranker_score,
        "raw_ranker_score": ranker_score,
        "raw_ranker_zscore": safe_float(raw_payload.get("raw_ranker_zscore"), 0.0),
        "raw_ranker_zscore_prob": clamp(safe_float(raw_payload.get("raw_ranker_zscore_prob"), score_target)),
        "raw_ranker_minmax_prob": clamp(safe_float(raw_payload.get("raw_ranker_minmax_prob"), score_target)),
        "t6_training_weight": weight,
        "score_margin_to_top1": margin,
        "crop_utility_margin_to_top1": margin,
        "t6_ranker_margin_to_top1": ranker_margin,
    }
    return {
        "candidate_id": str(row.get("candidate_id", "")),
        "bbox_norm_xyxy": box,
        "bbox_cxcywh": xyxy_to_cxcywh(box),
        "source": str(candidate_source),
        "label_type": "positive" if is_positive else ("hard_negative" if hard_negative else "candidate"),
        "score_rank_pct": rank_pct,
        "crop_utility_rank_pct": rank_pct,
        "score_prob": rank_pct,
        "crop_utility_prob": rank_pct,
        "score_targets": score_targets,
        "is_positive_candidate": bool(is_positive),
        "is_soft_positive": bool((not is_positive) and rank_pct >= 0.90 and not unsafe),
        "is_hard_negative": bool(hard_negative),
        "is_unsafe_negative": bool(fatal or unsafe),
        "is_ignore_candidate": False,
        "is_overflow_candidate": False,
        "candidate_weight": weight,
        "training_bucket": str(row.get("label_quality_tier", "candidate")),
        "reject_tags": [str(v) for v in (row.get("warnings") or [])],
        "t6_teacher": {
            "score_source": str(row.get("score_source", "deep_crop_ranker")),
            "ranker_method": str(row.get("ranker_method", "deep_crop_ranker_score_only")),
            "protocol": str(row.get("protocol", "Gc")),
            "ranker_score": ranker_score,
            "raw_score_rank_pct_by_image": raw_rank_pct,
            "score_rank_pct_by_image": rank_pct,
            "selected_by_ranker": bool(row.get("selected_by_ranker", False)),
            "selected_by_converter": bool(is_positive),
            "mos": safe_float(row.get("mos"), 0.0),
            "explanation_score": safe_float(row.get("explanation_score"), 0.0),
            "explanation_confidence": safe_float(row.get("explanation_confidence"), 0.0),
            "fatal_flag": fatal,
            "contradiction_flag": contradiction,
            "unsafe_after_converter": bool(unsafe),
            "label_quality_tier": str(row.get("label_quality_tier", "")),
            "training_weight": weight,
            "warnings": list(row.get("warnings") or []),
            "checklist_labels": row.get("checklist_labels", {}),
        },
    }


def build_batch_rows(
    labels: Sequence[dict[str, Any]],
    *,
    project_root: Path,
    image_roots: Sequence[Path],
    split: str,
    max_candidates_per_image: int,
    positive_top_k: int,
    positive_selection_policy: str = "best_safe",
    unsafe_group_policy: str = "skip",
    score_adjustment_policy: str = "safe_rescale",
    unsafe_score_cap: float = 0.05,
    unsafe_candidate_weight: float = 0.85,
    unsafe_warning_tags: set[str] | None = None,
    unsafe_include_contradiction: bool = False,
    score_target_policy: str = "safe_rank_pct",
    raw_score_temperature: float = 1.0,
    raw_score_blend_weight: float = 0.35,
    label_source_name: str = "t6_deep_crop_ranker_score_only",
    candidate_source: str = "t6_score_only_official",
    score_semantics_name: str = "t6_score_rank_pct",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        grouped[str(row.get("image_id", ""))].append(row)

    out: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for image_id, rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda r: (
                1 if r.get("selected_by_ranker") else 0,
                safe_float(r.get("score_rank_pct_by_image"), 0.0),
                safe_float(r.get("ranker_score"), 0.0),
            ),
            reverse=True,
        )
        if max_candidates_per_image > 0:
            rows = rows[: int(max_candidates_per_image)]
        adjusted_scores = adjusted_rank_scores(
            rows,
            policy=score_adjustment_policy,
            unsafe_score_cap=float(unsafe_score_cap),
            unsafe_warning_tags=unsafe_warning_tags,
            unsafe_include_contradiction=bool(unsafe_include_contradiction),
        )
        raw_targets = raw_score_targets(
            rows,
            adjusted_scores=adjusted_scores,
            policy=str(score_target_policy),
            raw_score_temperature=float(raw_score_temperature),
            raw_score_blend_weight=float(raw_score_blend_weight),
            unsafe_score_cap=float(unsafe_score_cap),
            unsafe_warning_tags=unsafe_warning_tags,
            unsafe_include_contradiction=bool(unsafe_include_contradiction),
        )
        selectable_rows = [row for row in rows if not is_unsafe_t6_row(row, unsafe_warning_tags=unsafe_warning_tags, include_contradiction=bool(unsafe_include_contradiction))]
        if positive_selection_policy == "ranker_top1":
            positive_source_rows = rows[: max(1, int(positive_top_k))]
        elif positive_selection_policy == "best_safe":
            positive_source_rows = sorted(
                selectable_rows,
                key=lambda r: (
                    adjusted_scores.get(str(r.get("candidate_id", "")), 0.0),
                    safe_float(r.get("score_rank_pct_by_image"), 0.0),
                    safe_float(r.get("ranker_score"), 0.0),
                ),
                reverse=True,
            )[: max(1, int(positive_top_k))]
        else:
            raise ValueError(f"unsupported positive_selection_policy: {positive_selection_policy}")
        if not positive_source_rows and unsafe_group_policy == "skip":
            counters["skipped_no_selectable_positive"] += 1
            continue
        if not positive_source_rows and unsafe_group_policy not in {"keep_negative"}:
            raise ValueError(f"unsupported unsafe_group_policy: {unsafe_group_policy}")
        top_row = positive_source_rows[0] if positive_source_rows else rows[0]
        top_score = adjusted_scores.get(str(top_row.get("candidate_id", "")), safe_float(top_row.get("score_rank_pct_by_image"), 1.0))
        top_ranker_score = safe_float(top_row.get("ranker_score"), 0.0)
        positive_ids = {str(r.get("candidate_id", "")) for r in positive_source_rows}
        image_path = resolve_image_path(image_id, image_roots)
        with Image.open(image_path) as img:
            width, height = img.size
        matching_targets = []
        candidate_pool = []
        for row in rows:
            is_pos = str(row.get("candidate_id", "")) in positive_ids
            candidate = t6_candidate(
                row,
                top_score=top_score,
                top_ranker_score=top_ranker_score,
                is_positive=is_pos,
                adjusted_rank_pct=adjusted_scores.get(str(row.get("candidate_id", ""))),
                unsafe_warning_tags=unsafe_warning_tags,
                unsafe_include_contradiction=bool(unsafe_include_contradiction),
                unsafe_candidate_weight=float(unsafe_candidate_weight),
                candidate_source=str(candidate_source),
                raw_target_payload=raw_targets.get(str(row.get("candidate_id", ""))),
            )
            if is_pos:
                matching_targets.append(candidate)
            else:
                candidate_pool.append(candidate)
            counters[str(candidate["training_bucket"])] += 1
            if candidate["t6_teacher"].get("unsafe_after_converter"):
                counters["converter_unsafe_candidates"] += 1
            if is_pos and candidate["t6_teacher"].get("unsafe_after_converter"):
                counters["converter_unsafe_positive"] += 1
        row = {
            "schema_version": "mobilecropnet_v4_t6_score_only_batch_v2",
            "image_id": image_id,
            "image_path": relative_to_project(image_path, project_root),
            "width": width,
            "height": height,
            "target_ar": "FREE",
            "label_generation": {
                "source": str(label_source_name),
                "split": split,
                "candidate_count": len(rows),
                "positive_top_k": int(positive_top_k),
                "max_candidates_per_image": int(max_candidates_per_image),
                "positive_selection_policy": str(positive_selection_policy),
                "unsafe_group_policy": str(unsafe_group_policy),
                "score_adjustment_policy": str(score_adjustment_policy),
                "unsafe_score_cap": float(unsafe_score_cap),
                "unsafe_candidate_weight": float(unsafe_candidate_weight),
                "unsafe_warning_tags": sorted(unsafe_warning_tags or []),
                "unsafe_include_contradiction": bool(unsafe_include_contradiction),
                "score_target_policy": str(score_target_policy),
                "raw_score_temperature": float(raw_score_temperature),
                "raw_score_blend_weight": float(raw_score_blend_weight),
            },
            "routing": {
                "subject_mode": "scene_general",
                "subject_mode_id": 6,
                "subject_mode_conf": 0.0,
                "route_conf": 0.0,
                "policy_id": "t6_score_only_global_quality",
                "flags": {"t6_score_only_label": True},
            },
            "baseline": {
                "candidate_id": "full_image_baseline",
                "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0],
                "bbox_cxcywh": [0.5, 0.5, 1.0, 1.0],
                "score_rank_pct": 0.0,
                "crop_utility_rank_pct": 0.0,
                "score_prob": 0.0,
                "crop_utility_prob": 0.0,
                "score_targets": {
                    "score_prob": 0.0,
                    "crop_utility_prob": 0.0,
                    "score_rank_pct": 0.0,
                    "crop_utility_rank_pct": 0.0,
                    "rank_pct": 0.0,
                    "pseudo_mos_1to5": 1.0,
                },
            },
            "decision_target": {
                "decision_type": "crop",
                "decision_id": 2,
                "delta_vs_base": 1.0,
                "base_candidate_id": "full_image_baseline",
                "winner_post_gate_candidate_id": str(top_row.get("candidate_id", "")),
                "winner_post_gate_bbox": top_row.get("bbox_norm_xyxy"),
                "chosen_score_rank": top_score,
                "chosen_crop_utility_raw": top_ranker_score,
            },
            "matching_targets": matching_targets,
            "candidate_pool": candidate_pool,
            "ignored_candidates": [],
            "overflow_candidates": [],
            "teacher_meta": {
                "teacher_confidence": 1.0,
                "target_reliability": 1.0,
                "candidate_count": len(rows),
                "selectable_candidate_count": len(selectable_rows),
                "score_source": str(label_source_name),
            },
            "score_semantics": {
                "official_score_name": str(score_semantics_name),
                "official_score_field": "score_rank_pct",
                "official_prob_name": str(score_semantics_name),
                "official_prob_field": "score_rank_pct",
                "decision_semantics": "gaic_mos_ranker",
            },
        }
        out.append(row)
        candidate_counts.append(len(rows))
    summary = {
        "schema_version": "mobilecropnet_v4_t6_score_only_batch_v2",
        "split": split,
        "image_count": len(out),
        "candidate_count": int(sum(candidate_counts)),
        "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
        "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
        "candidate_count_mean": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
        "bucket_counts": dict(sorted(counters.items())),
        "max_candidates_per_image": int(max_candidates_per_image),
        "positive_top_k": int(positive_top_k),
        "positive_selection_policy": str(positive_selection_policy),
        "unsafe_group_policy": str(unsafe_group_policy),
        "score_adjustment_policy": str(score_adjustment_policy),
        "unsafe_score_cap": float(unsafe_score_cap),
        "unsafe_candidate_weight": float(unsafe_candidate_weight),
        "unsafe_warning_tags": sorted(unsafe_warning_tags or []),
        "unsafe_include_contradiction": bool(unsafe_include_contradiction),
        "score_target_policy": str(score_target_policy),
        "raw_score_temperature": float(raw_score_temperature),
        "raw_score_blend_weight": float(raw_score_blend_weight),
        "label_source_name": str(label_source_name),
        "candidate_source": str(candidate_source),
        "score_semantics_name": str(score_semantics_name),
    }
    return out, summary


def softmax_scores(values: Sequence[float], *, temperature: float = 0.12) -> list[float]:
    if not values:
        return []
    temp = max(1e-4, float(temperature))
    arr = [float(v) / temp for v in values]
    m = max(arr)
    exp = [math.exp(v - m) for v in arr]
    denom = sum(exp) or 1.0
    return [float(v / denom) for v in exp]


def candidate_iter_for_rank_rows(batch_row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(batch_row.get("matching_targets") or []) + list(batch_row.get("candidate_pool") or [])


def build_listwise_rows(batch_rows: Sequence[dict[str, Any]], *, temperature: float = 0.12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in batch_rows:
        candidates = candidate_iter_for_rank_rows(row)
        scores = [safe_float(candidate.get("score_targets", {}).get("crop_utility_prob", candidate.get("crop_utility_rank_pct")), 0.0) for candidate in candidates]
        probs = softmax_scores(scores, temperature=temperature)
        out.append(
            {
                "image_id": str(row.get("image_id", "")),
                "target_ar": str(row.get("target_ar", "FREE")),
                "source": "t6_score_only_converter_v2",
                "candidates": [
                    {
                        "candidate_id": str(candidate.get("candidate_id", "")),
                        "score_rank_pct": float(score),
                        "crop_utility_rank_pct": float(score),
                        "rank_pct": float(score),
                        "score_softmax_local": float(prob),
                        "crop_utility_softmax_local": float(prob),
                        "teacher_softmax_local": float(prob),
                        "is_positive_candidate": bool(candidate.get("is_positive_candidate", False)),
                        "is_hard_negative": bool(candidate.get("is_hard_negative", False)),
                        "is_unsafe_negative": bool(candidate.get("is_unsafe_negative", False)),
                    }
                    for candidate, score, prob in zip(candidates, scores, probs)
                ],
            }
        )
    return out


def build_pairwise_rows(
    batch_rows: Sequence[dict[str, Any]],
    *,
    max_pairs_per_image: int = 96,
    score_margin: float = 0.03,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in batch_rows:
        candidates = candidate_iter_for_rank_rows(row)
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                safe_float(candidate.get("score_targets", {}).get("crop_utility_rank_pct", candidate.get("crop_utility_rank_pct")), 0.0),
                1.0 if candidate.get("is_positive_candidate") else 0.0,
                str(candidate.get("candidate_id", "")),
            ),
            reverse=True,
        )
        pairs: list[dict[str, Any]] = []
        for i, better in enumerate(ranked):
            if len(pairs) >= int(max_pairs_per_image):
                break
            better_score = safe_float(better.get("score_targets", {}).get("crop_utility_rank_pct", better.get("crop_utility_rank_pct")), 0.0)
            for worse in ranked[i + 1 :]:
                worse_score = safe_float(worse.get("score_targets", {}).get("crop_utility_rank_pct", worse.get("crop_utility_rank_pct")), 0.0)
                margin = better_score - worse_score
                if margin < float(score_margin):
                    continue
                pair_type = "top1_vs_negative" if bool(worse.get("is_hard_negative") or worse.get("is_unsafe_negative")) else "rank_margin"
                pairs.append(
                    {
                        "image_id": str(row.get("image_id", "")),
                        "target_ar": str(row.get("target_ar", "FREE")),
                        "candidate_id_a": str(better.get("candidate_id", "")),
                        "candidate_id_b": str(worse.get("candidate_id", "")),
                        "label": 1,
                        "score_margin": float(margin),
                        "crop_utility_margin": float(margin),
                        "pair_type": pair_type,
                        "source": "t6_score_only_converter_v2",
                    }
                )
                if len(pairs) >= int(max_pairs_per_image):
                    break
        out.extend(pairs)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert T6 score-only labels to MobileCropNet v4 batch JSONL labels.")
    parser.add_argument("--t6_labels_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--image_root", type=Path, action="append", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max_candidates_per_image", type=int, default=0)
    parser.add_argument("--positive_top_k", type=int, default=1)
    parser.add_argument("--positive_selection_policy", choices=["best_safe", "ranker_top1"], default="best_safe")
    parser.add_argument("--unsafe_group_policy", choices=["skip", "keep_negative"], default="skip")
    parser.add_argument("--score_adjustment_policy", choices=["safe_rescale", "unsafe_cap", "keep"], default="safe_rescale")
    parser.add_argument("--unsafe_score_cap", type=float, default=0.05)
    parser.add_argument("--unsafe_candidate_weight", type=float, default=0.85)
    parser.add_argument("--unsafe_warning_tags", default="", help="Comma-separated warning tags to additionally treat as unsafe.")
    parser.add_argument("--unsafe_include_contradiction", action="store_true", help="Treat contradiction_flag as hard unsafe. Default keeps it as low-weight metadata.")
    parser.add_argument("--score_target_policy", choices=["safe_rank_pct", "rank_pct", "raw_zscore", "blend_rank_zscore", "raw_minmax", "blend_rank_minmax"], default="safe_rank_pct")
    parser.add_argument("--raw_score_temperature", type=float, default=1.0)
    parser.add_argument("--raw_score_blend_weight", type=float, default=0.35)
    parser.add_argument("--label_source_name", default="t6_deep_crop_ranker_score_only")
    parser.add_argument("--candidate_source", default="t6_score_only_official")
    parser.add_argument("--score_semantics_name", default="t6_score_rank_pct")
    parser.add_argument("--output_pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--output_listwise_jsonl", type=Path, default=None)
    parser.add_argument("--max_pairs_per_image", type=int, default=96)
    parser.add_argument("--pairwise_score_margin", type=float, default=0.03)
    parser.add_argument("--listwise_temperature", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_jsonl(args.t6_labels_jsonl)
    rows, summary = build_batch_rows(
        labels,
        project_root=Path(args.project_root),
        image_roots=[Path(p) for p in args.image_root],
        split=str(args.split),
        max_candidates_per_image=int(args.max_candidates_per_image),
        positive_top_k=int(args.positive_top_k),
        positive_selection_policy=str(args.positive_selection_policy),
        unsafe_group_policy=str(args.unsafe_group_policy),
        score_adjustment_policy=str(args.score_adjustment_policy),
        unsafe_score_cap=float(args.unsafe_score_cap),
        unsafe_candidate_weight=float(args.unsafe_candidate_weight),
        unsafe_warning_tags={part.strip() for part in str(args.unsafe_warning_tags).split(",") if part.strip()} or None,
        unsafe_include_contradiction=bool(args.unsafe_include_contradiction),
        score_target_policy=str(args.score_target_policy),
        raw_score_temperature=float(args.raw_score_temperature),
        raw_score_blend_weight=float(args.raw_score_blend_weight),
        label_source_name=str(args.label_source_name),
        candidate_source=str(args.candidate_source),
        score_semantics_name=str(args.score_semantics_name),
    )
    written = write_jsonl(args.output_jsonl, rows)
    pairwise_count = 0
    listwise_count = 0
    if args.output_pairwise_jsonl is not None:
        pairwise_count = write_jsonl(
            args.output_pairwise_jsonl,
            build_pairwise_rows(rows, max_pairs_per_image=int(args.max_pairs_per_image), score_margin=float(args.pairwise_score_margin)),
        )
    if args.output_listwise_jsonl is not None:
        listwise_count = write_jsonl(args.output_listwise_jsonl, build_listwise_rows(rows, temperature=float(args.listwise_temperature)))
    summary.update(
        {
            "t6_labels_jsonl": str(args.t6_labels_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "written_rows": int(written),
            "image_roots": [str(p) for p in args.image_root],
            "output_pairwise_jsonl": str(args.output_pairwise_jsonl) if args.output_pairwise_jsonl is not None else None,
            "output_listwise_jsonl": str(args.output_listwise_jsonl) if args.output_listwise_jsonl is not None else None,
            "pairwise_rows": int(pairwise_count),
            "listwise_rows": int(listwise_count),
        }
    )
    write_json(args.summary_json, summary)
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
