from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from public_benchmark.metrics import load_predictions
from public_benchmark.staging import write_jsonl

PredictionKey = tuple[str, str, str | None]


def candidate_signature(candidate: dict[str, Any]) -> tuple[str, Any]:
    candidate_id = str(candidate.get("candidate_id", "")).strip()
    if candidate_id:
        return ("candidate_id", candidate_id)
    annotation_id = candidate.get("annotation_id")
    if annotation_id is not None:
        return ("annotation_id", str(annotation_id))
    candidate_index = candidate.get("candidate_index")
    if candidate_index is not None:
        return ("candidate_index", str(candidate_index))
    bbox = candidate.get("bbox_xyxy_norm", candidate.get("bbox_norm_xyxy"))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return ("bbox", tuple(round(float(v), 6) for v in bbox))
        except (TypeError, ValueError):
            pass
    return ("fallback", id(candidate))


def _finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _normalized_scores(values: dict[tuple[str, Any], float], mode: str) -> dict[tuple[str, Any], float]:
    if not values:
        return {}
    mode_key = str(mode).strip().lower()
    items = list(values.items())
    raw_scores = [float(score) for _, score in items]
    if mode_key in {"raw", ""}:
        return {key: float(score) for key, score in items}
    if mode_key == "zscore":
        mean = float(sum(raw_scores) / len(raw_scores))
        var = float(sum((score - mean) ** 2 for score in raw_scores) / len(raw_scores))
        std = math.sqrt(max(0.0, var))
        if std <= 1e-12:
            return {key: 0.0 for key, _ in items}
        return {key: float((score - mean) / std) for key, score in items}
    if mode_key == "rank_pct":
        ordered = sorted(items, key=lambda item: (float(item[1]), str(item[0])), reverse=True)
        if len(ordered) == 1:
            return {ordered[0][0]: 1.0}
        out: dict[tuple[str, Any], float] = {}
        pos = 0
        denom = float(max(1, len(ordered) - 1))
        while pos < len(ordered):
            end = pos + 1
            while end < len(ordered) and float(ordered[end][1]) == float(ordered[pos][1]):
                end += 1
            avg_rank = 0.5 * (float(pos + 1) + float(end))
            score = float(1.0 - (avg_rank - 1.0) / denom)
            for key, _ in ordered[pos:end]:
                out[key] = score
            pos = end
        return out
    raise ValueError(f"Unsupported normalization mode: {mode}")


def normalized_candidate_scores(
    candidates: Iterable[dict[str, Any]],
    *,
    normalization: str,
    score_field: str = "score",
) -> dict[tuple[str, Any], float]:
    finite_scores: dict[tuple[str, Any], float] = {}
    field = str(score_field).strip() or "score"
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        score = _finite_score(candidate.get(field))
        if score is None and field != "score":
            score = _finite_score(candidate.get("score"))
        if score is None:
            continue
        finite_scores[candidate_signature(candidate)] = float(score)
    return _normalized_scores(finite_scores, normalization)


def fuse_candidate_groups(
    candidate_groups_by_method: dict[str, list[dict[str, Any]]],
    *,
    method_weights: dict[str, float],
    normalization: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_signatures: list[tuple[str, Any]] = []
    canonical: dict[tuple[str, Any], dict[str, Any]] = {}
    normalized_by_method: dict[str, dict[tuple[str, Any], float]] = {}
    method_candidate_counts: dict[str, int] = {}
    for method_name, candidates in candidate_groups_by_method.items():
        finite_scores: dict[tuple[str, Any], float] = {}
        method_candidate_counts[method_name] = len(candidates)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            signature = candidate_signature(candidate)
            if signature not in canonical:
                canonical[signature] = copy.deepcopy(candidate)
                ordered_signatures.append(signature)
            score = _finite_score(candidate.get("score"))
            if score is not None:
                finite_scores[signature] = score
        normalized_by_method[method_name] = _normalized_scores(finite_scores, normalization)

    fused_candidates: list[dict[str, Any]] = []
    missing_scores = 0
    for signature in ordered_signatures:
        base = copy.deepcopy(canonical[signature])
        fused_sum = 0.0
        fused_weight = 0.0
        method_scores: dict[str, float | None] = {}
        for method_name, normalized_scores in normalized_by_method.items():
            value = normalized_scores.get(signature)
            if value is None:
                method_scores[method_name] = None
                continue
            weight = float(method_weights.get(method_name, 1.0))
            fused_sum += weight * float(value)
            fused_weight += weight
            method_scores[method_name] = float(value)
        if fused_weight <= 0.0:
            base["score"] = None
            missing_scores += 1
        else:
            base["score"] = float(fused_sum / fused_weight)
        scores = base.get("scores", {}) if isinstance(base.get("scores"), dict) else {}
        scores.update({f"{name}_normalized": value for name, value in method_scores.items() if value is not None})
        scores["fused_score"] = base.get("score")
        base["scores"] = scores
        fused_candidates.append(base)

    stats = {
        "candidate_count": len(fused_candidates),
        "missing_fused_score_count": missing_scores,
        "method_candidate_counts": method_candidate_counts,
        "normalization": str(normalization),
        "method_weights": {name: float(weight) for name, weight in method_weights.items()},
    }
    return fused_candidates, stats


def fuse_prediction_groups(
    prediction_groups_by_method: dict[str, dict[PredictionKey, list[dict[str, Any]]]],
    *,
    method_weights: dict[str, float],
    normalization: str,
) -> tuple[dict[PredictionKey, list[dict[str, Any]]], dict[str, Any]]:
    all_keys: list[PredictionKey] = []
    seen_keys: set[PredictionKey] = set()
    for groups in prediction_groups_by_method.values():
        for key in groups.keys():
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_keys.append(key)

    fused: dict[PredictionKey, list[dict[str, Any]]] = {}
    per_key_stats: dict[str, Any] = {}
    missing_group_count = 0
    for key in all_keys:
        candidates_by_method = {
            method_name: list(groups.get(key, []))
            for method_name, groups in prediction_groups_by_method.items()
        }
        if any(not rows for rows in candidates_by_method.values()):
            missing_group_count += 1
        fused_candidates, stats = fuse_candidate_groups(
            candidates_by_method,
            method_weights=method_weights,
            normalization=normalization,
        )
        fused[key] = fused_candidates
        per_key_stats[str(key)] = stats
    summary = {
        "group_count": len(all_keys),
        "missing_group_count": missing_group_count,
        "normalization": str(normalization),
        "method_weights": {name: float(weight) for name, weight in method_weights.items()},
        "per_key_stats": per_key_stats,
    }
    return fused, summary


def write_grouped_predictions(
    output_jsonl: Path,
    groups: dict[PredictionKey, list[dict[str, Any]]],
    *,
    method_name: str,
    source: str,
) -> int:
    rows: list[dict[str, Any]] = []
    for (dataset, image_id, target_ar), candidates in groups.items():
        rows.append(
            {
                "dataset": str(dataset),
                "image_id": str(image_id),
                "target_ar": target_ar,
                "method": str(method_name),
                "source": str(source),
                "candidates": copy.deepcopy(candidates),
            }
        )
    return write_jsonl(output_jsonl, rows)


def load_prediction_groups(path: Path) -> dict[PredictionKey, list[dict[str, Any]]]:
    return load_predictions(Path(path))


def keyed_candidate_scores(groups: dict[PredictionKey, list[dict[str, Any]]]) -> dict[PredictionKey, dict[tuple[str, Any], float]]:
    out: dict[PredictionKey, dict[tuple[str, Any], float]] = defaultdict(dict)
    for key, candidates in groups.items():
        for candidate in candidates:
            signature = candidate_signature(candidate)
            score = _finite_score(candidate.get("score"))
            if score is None:
                continue
            out[key][signature] = score
    return dict(out)
