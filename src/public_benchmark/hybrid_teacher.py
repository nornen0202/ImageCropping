from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

from public_benchmark.ensemble import (
    PredictionKey,
    candidate_signature,
    fuse_candidate_groups,
    normalized_candidate_scores,
)


def _finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _uctr_uncertainty_by_signature(candidates: list[dict[str, Any]]) -> dict[tuple[str, Any], float]:
    out: dict[tuple[str, Any], float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        scores = candidate.get("scores", {}) if isinstance(candidate.get("scores"), dict) else {}
        uncertainty = _finite_score(scores.get("uctr_uncertainty"))
        if uncertainty is None:
            continue
        out[candidate_signature(candidate)] = min(1.0, max(0.0, float(uncertainty)))
    return out


@dataclass
class HybridTeacherConfig:
    public_normalization: str = "rank_pct"
    public_gaic_weight: float = 0.65
    hybrid_normalization: str = "rank_pct"
    uctr_weight: float = 0.65
    uctr_uncertainty_gamma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fuse_hybrid_candidate_groups(
    uctr_candidates: list[dict[str, Any]],
    public_candidates_by_method: dict[str, list[dict[str, Any]]],
    *,
    config: HybridTeacherConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_names = list(public_candidates_by_method.keys())
    if len(method_names) != 2:
        raise ValueError("Hybrid public ensemble currently expects exactly two public methods")
    public_method_weights = {
        method_names[0]: float(config.public_gaic_weight),
        method_names[1]: float(1.0 - float(config.public_gaic_weight)),
    }
    public_fused_candidates, public_fuse_stats = fuse_candidate_groups(
        public_candidates_by_method,
        method_weights=public_method_weights,
        normalization=str(config.public_normalization),
    )
    canonical: dict[tuple[str, Any], dict[str, Any]] = {}
    ordered_signatures: list[tuple[str, Any]] = []
    for candidate in list(uctr_candidates) + list(public_fused_candidates):
        if not isinstance(candidate, dict):
            continue
        signature = candidate_signature(candidate)
        if signature in canonical:
            continue
        canonical[signature] = copy.deepcopy(candidate)
        ordered_signatures.append(signature)

    uctr_norm = normalized_candidate_scores(uctr_candidates, normalization=str(config.hybrid_normalization))
    public_norm = normalized_candidate_scores(public_fused_candidates, normalization=str(config.hybrid_normalization))
    uctr_uncertainty = _uctr_uncertainty_by_signature(uctr_candidates)
    fused_candidates: list[dict[str, Any]] = []
    missing_score_count = 0
    for signature in ordered_signatures:
        base = copy.deepcopy(canonical[signature])
        uctr_value = uctr_norm.get(signature)
        public_value = public_norm.get(signature)
        uctr_raw_weight = float(config.uctr_weight)
        uncertainty = float(uctr_uncertainty.get(signature, 0.0))
        if uctr_value is not None and float(config.uctr_uncertainty_gamma) > 0.0:
            uctr_raw_weight *= float(max(0.0, 1.0 - uncertainty) ** float(config.uctr_uncertainty_gamma))
        public_raw_weight = float(1.0 - float(config.uctr_weight))
        fused_sum = 0.0
        fused_weight = 0.0
        if uctr_value is not None and uctr_raw_weight > 0.0:
            fused_sum += float(uctr_value) * float(uctr_raw_weight)
            fused_weight += float(uctr_raw_weight)
        if public_value is not None and public_raw_weight > 0.0:
            fused_sum += float(public_value) * float(public_raw_weight)
            fused_weight += float(public_raw_weight)
        if fused_weight <= 0.0:
            score = None
            missing_score_count += 1
        else:
            score = float(fused_sum / fused_weight)
        base["score"] = score
        scores = base.get("scores", {}) if isinstance(base.get("scores"), dict) else {}
        if uctr_value is not None:
            scores["uctr_normalized"] = float(uctr_value)
        if public_value is not None:
            scores["public_ensemble_normalized"] = float(public_value)
        scores["hybrid_score"] = score
        scores["hybrid_uctr_weight_effective"] = float(uctr_raw_weight)
        scores["hybrid_public_weight_effective"] = float(public_raw_weight)
        scores["hybrid_uctr_uncertainty"] = float(uncertainty)
        base["scores"] = scores
        fused_candidates.append(base)

    order = sorted(
        range(len(fused_candidates)),
        key=lambda idx: (
            -1e12 if fused_candidates[idx].get("score") is None else float(fused_candidates[idx]["score"]),
            str(fused_candidates[idx].get("candidate_id", "")),
        ),
        reverse=True,
    )
    for rank, idx in enumerate(order, 1):
        fused_candidates[idx]["model_rank"] = int(rank)

    stats = {
        "candidate_count": len(fused_candidates),
        "missing_fused_score_count": int(missing_score_count),
        "public_fuse": public_fuse_stats,
        "config": config.to_dict(),
    }
    return fused_candidates, stats


def fuse_hybrid_prediction_groups(
    uctr_groups: dict[PredictionKey, list[dict[str, Any]]],
    public_groups_by_method: dict[str, dict[PredictionKey, list[dict[str, Any]]]],
    *,
    config: HybridTeacherConfig,
) -> tuple[dict[PredictionKey, list[dict[str, Any]]], dict[str, Any]]:
    all_keys: list[PredictionKey] = []
    seen_keys: set[PredictionKey] = set()
    for key in uctr_groups.keys():
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_keys.append(key)
    for groups in public_groups_by_method.values():
        for key in groups.keys():
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_keys.append(key)

    fused: dict[PredictionKey, list[dict[str, Any]]] = {}
    per_key_stats: dict[str, Any] = {}
    missing_public_group_count = 0
    missing_uctr_group_count = 0
    for key in all_keys:
        uctr_candidates = list(uctr_groups.get(key, []))
        public_candidates = {method: list(groups.get(key, [])) for method, groups in public_groups_by_method.items()}
        if not uctr_candidates:
            missing_uctr_group_count += 1
        if any(not rows for rows in public_candidates.values()):
            missing_public_group_count += 1
        fused_candidates, stats = fuse_hybrid_candidate_groups(
            uctr_candidates,
            public_candidates_by_method=public_candidates,
            config=config,
        )
        fused[key] = fused_candidates
        per_key_stats[str(key)] = stats
    summary = {
        "group_count": len(all_keys),
        "missing_public_group_count": int(missing_public_group_count),
        "missing_uctr_group_count": int(missing_uctr_group_count),
        "config": config.to_dict(),
        "per_key_stats": per_key_stats,
    }
    return fused, summary
