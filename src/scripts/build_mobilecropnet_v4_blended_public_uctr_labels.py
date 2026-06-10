#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator


SCORE_KEYS = ("crop_utility_prob", "score_prob")
RANK_KEYS = ("crop_utility_rank_pct", "score_rank_pct")
RISK_KEYS = ("is_hard_negative", "is_unsafe_negative", "is_ignore_candidate", "is_overflow_candidate")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _row_key(row: dict[str, Any], fallback_index: int) -> tuple[str, str, str]:
    return (
        str(row.get("image_id") or row.get("image_path") or fallback_index),
        str(row.get("target_ar") or row.get("target_ar_id") or "FREE"),
        str(row.get("schema_version") or ""),
    )


def _candidate_box_key(candidate: dict[str, Any]) -> str:
    box = candidate.get("bbox_norm_xyxy") or candidate.get("box") or candidate.get("bbox_cxcywh") or []
    if isinstance(box, list):
        return ",".join(f"{_safe_float(v, 0.0) or 0.0:.5f}" for v in box[:4])
    return ""


def _candidate_key(candidate: dict[str, Any], idx: int) -> str:
    cid = str(candidate.get("candidate_id") or "").strip()
    if cid:
        return cid
    return f"anon:{idx}:{_candidate_box_key(candidate)}"


def _score(candidate: dict[str, Any]) -> float:
    for key in (*SCORE_KEYS, *RANK_KEYS):
        value = _safe_float(candidate.get(key))
        if value is not None:
            return _clamp01(value)
    targets = candidate.get("score_targets")
    if isinstance(targets, dict):
        for key in (*SCORE_KEYS, *RANK_KEYS):
            value = _safe_float(targets.get(key))
            if value is not None:
                return _clamp01(value)
    return 0.0


def _weighted_average(
    public_candidate: dict[str, Any] | None,
    uctr_candidate: dict[str, Any] | None,
    key: str,
    *,
    public_weight: float,
    uctr_weight: float,
) -> float | None:
    values: list[tuple[float, float]] = []
    if public_candidate is not None:
        value = _safe_float(public_candidate.get(key))
        if value is not None:
            values.append((value, public_weight))
    if uctr_candidate is not None:
        value = _safe_float(uctr_candidate.get(key))
        if value is not None:
            values.append((value, uctr_weight))
    if not values:
        return None
    denom = sum(weight for _value, weight in values)
    return _clamp01(sum(value * weight for value, weight in values) / max(1e-8, denom))


def _merge_score_targets(
    out: dict[str, Any],
    public_candidate: dict[str, Any] | None,
    uctr_candidate: dict[str, Any] | None,
    *,
    public_weight: float,
    uctr_weight: float,
) -> None:
    public_targets = public_candidate.get("score_targets") if isinstance(public_candidate, dict) else None
    uctr_targets = uctr_candidate.get("score_targets") if isinstance(uctr_candidate, dict) else None
    if not isinstance(public_targets, dict) and not isinstance(uctr_targets, dict):
        return
    merged = dict(uctr_targets or public_targets or {})
    keys = set()
    if isinstance(public_targets, dict):
        keys.update(public_targets.keys())
    if isinstance(uctr_targets, dict):
        keys.update(uctr_targets.keys())
    for key in sorted(keys):
        values: list[tuple[float, float]] = []
        if isinstance(public_targets, dict):
            value = _safe_float(public_targets.get(key))
            if value is not None:
                values.append((value, public_weight))
        if isinstance(uctr_targets, dict):
            value = _safe_float(uctr_targets.get(key))
            if value is not None:
                values.append((value, uctr_weight))
        if values:
            denom = sum(weight for _value, weight in values)
            merged[key] = _clamp01(sum(value * weight for value, weight in values) / max(1e-8, denom))
    out["score_targets"] = merged


def _merge_candidate(
    key: str,
    public_candidate: dict[str, Any] | None,
    uctr_candidate: dict[str, Any] | None,
    *,
    public_weight: float,
    uctr_weight: float,
) -> dict[str, Any]:
    base = deepcopy(uctr_candidate if uctr_candidate is not None else public_candidate)
    if base is None:
        raise ValueError(f"candidate {key!r} has no source")
    base["candidate_id"] = base.get("candidate_id") or key
    public_score = _score(public_candidate) if public_candidate is not None else None
    uctr_score = _score(uctr_candidate) if uctr_candidate is not None else None
    blended_values: dict[str, float] = {}
    for score_key in SCORE_KEYS:
        value = _weighted_average(public_candidate, uctr_candidate, score_key, public_weight=public_weight, uctr_weight=uctr_weight)
        if value is not None:
            blended_values[score_key] = value
            base[score_key] = value
    _merge_score_targets(base, public_candidate, uctr_candidate, public_weight=public_weight, uctr_weight=uctr_weight)

    source_count = int(public_candidate is not None) + int(uctr_candidate is not None)
    disagreement = abs(float(public_score or 0.0) - float(uctr_score or 0.0)) if source_count == 2 else 0.25
    explicit_weight = max(
        _safe_float(public_candidate.get("candidate_weight"), 0.0) if public_candidate is not None else 0.0,
        _safe_float(uctr_candidate.get("candidate_weight"), 0.0) if uctr_candidate is not None else 0.0,
    )
    confidence = max(0.08, min(1.0, (0.35 + 0.65 * (1.0 - disagreement)) * max(0.35, explicit_weight)))
    if source_count == 1:
        confidence *= 0.75
    base["candidate_weight"] = confidence

    for risk_key in RISK_KEYS:
        base[risk_key] = bool(
            (public_candidate is not None and public_candidate.get(risk_key))
            or (uctr_candidate is not None and uctr_candidate.get(risk_key))
        )
    base["is_positive_candidate"] = bool(
        (public_candidate is not None and public_candidate.get("is_positive_candidate"))
        or (uctr_candidate is not None and uctr_candidate.get("is_positive_candidate"))
    )
    base["is_soft_positive"] = bool(
        (public_candidate is not None and public_candidate.get("is_soft_positive"))
        or (uctr_candidate is not None and uctr_candidate.get("is_soft_positive"))
    )
    base["label_source_blend"] = {
        "mode": "public_uctr_candidate_score_blend",
        "candidate_id": key,
        "source_count": source_count,
        "public_score": public_score,
        "uctr_score": uctr_score,
        "blended_score": _score(base),
        "public_weight": public_weight,
        "uctr_weight": uctr_weight,
        "score_disagreement_abs": disagreement,
    }
    return base


def _candidate_map(row: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(row.get(field) or []):
        if isinstance(candidate, dict):
            out[_candidate_key(candidate, idx)] = candidate
    return out


def _top_candidate_id(candidates: dict[str, dict[str, Any]]) -> tuple[str, float]:
    pool = list(candidates.items())
    if not pool:
        return "", 0.0
    safe_pool = [(key, candidate) for key, candidate in pool if not any(bool(candidate.get(k)) for k in RISK_KEYS)]
    ranking_pool = safe_pool or pool
    key, candidate = max(
        ranking_pool,
        key=lambda item: (
            _score(item[1]),
            -sum(float(v) for v in (item[1].get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])[:4]),
        ),
    )
    return str(key), _score(candidate)


def _rerank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates.sort(
        key=lambda c: (
            not any(bool(c.get(k)) for k in RISK_KEYS),
            _score(c),
            -sum(float(v) for v in (c.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])[:4]),
        ),
        reverse=True,
    )
    n = max(1, len(candidates))
    for idx, candidate in enumerate(candidates):
        rank_pct = float(n - idx - 1) / float(n)
        for key in RANK_KEYS:
            candidate[key] = rank_pct
        targets = candidate.get("score_targets")
        if isinstance(targets, dict):
            for key in RANK_KEYS:
                targets[key] = rank_pct
    return candidates


def _blend_row(
    public_row: dict[str, Any],
    uctr_row: dict[str, Any],
    *,
    public_weight: float,
    uctr_weight: float,
    matching_top_k: int,
    soft_positive_margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(uctr_row)
    out["schema_version"] = public_row.get("schema_version", out.get("schema_version"))
    out["label_generation"] = deepcopy(out.get("label_generation") or {})
    out["label_generation"]["public_uctr_blend"] = {
        "public_weight": public_weight,
        "uctr_weight": uctr_weight,
        "matching_top_k": matching_top_k,
        "soft_positive_margin": soft_positive_margin,
    }
    out["label_source_blend"] = {
        "row_mode": "public_uctr_candidate_score_blend",
        "base_row": "uctr_stage3",
        "score_public_weight": public_weight,
        "score_uctr_weight": uctr_weight,
    }

    public_pool = _candidate_map(public_row, "candidate_pool")
    uctr_pool = _candidate_map(uctr_row, "candidate_pool")
    public_matching = _candidate_map(public_row, "matching_targets")
    uctr_matching = _candidate_map(uctr_row, "matching_targets")
    public_top_id, public_top_score = _top_candidate_id(public_pool or public_matching)
    uctr_top_id, uctr_top_score = _top_candidate_id(uctr_pool or uctr_matching)
    keys = list(dict.fromkeys([*public_matching.keys(), *uctr_matching.keys(), *public_pool.keys(), *uctr_pool.keys()]))
    merged = [
        _merge_candidate(
            key,
            public_pool.get(key) or public_matching.get(key),
            uctr_pool.get(key) or uctr_matching.get(key),
            public_weight=public_weight,
            uctr_weight=uctr_weight,
        )
        for key in keys
    ]
    merged = _rerank_candidates(merged)
    safe_merged = [c for c in merged if not any(bool(c.get(k)) for k in RISK_KEYS)]
    positive_source = safe_merged or merged
    max_score = _score(positive_source[0]) if positive_source else 0.0
    positives: list[dict[str, Any]] = []
    for candidate in positive_source:
        if len(positives) >= max(1, matching_top_k):
            break
        if _score(candidate) >= max_score - max(0.0, soft_positive_margin):
            pos = deepcopy(candidate)
            pos["is_positive_candidate"] = len(positives) == 0
            pos["is_soft_positive"] = len(positives) > 0
            positives.append(pos)
    positive_ids = {str(c.get("candidate_id")) for c in positives}
    for candidate in merged:
        cid = str(candidate.get("candidate_id"))
        if cid in positive_ids:
            candidate["is_positive_candidate"] = cid == str(positives[0].get("candidate_id"))
            candidate["is_soft_positive"] = cid != str(positives[0].get("candidate_id"))
    out["candidate_pool"] = merged
    out["matching_targets"] = positives
    stats = {
        "candidate_count": len(merged),
        "positive_count": len(positives),
        "candidate_id_overlap": len(set(public_pool) & set(uctr_pool)),
        "public_candidate_count": len(public_pool),
        "uctr_candidate_count": len(uctr_pool),
        "top_public_score": _score(public_pool.get(str(merged[0].get("candidate_id")), {})) if merged else 0.0,
        "top_uctr_score": _score(uctr_pool.get(str(merged[0].get("candidate_id")), {})) if merged else 0.0,
        "top_blended_score": _score(merged[0]) if merged else 0.0,
        "public_top_id": public_top_id,
        "uctr_top_id": uctr_top_id,
        "source_top_id_match": bool(public_top_id and public_top_id == uctr_top_id),
        "source_top_score_gap": abs(float(public_top_score) - float(uctr_top_score)),
    }
    return out, stats


def _build_split(
    *,
    public_path: Path,
    uctr_path: Path,
    output_path: Path,
    public_weight: float,
    uctr_weight: float,
    matching_top_k: int,
    soft_positive_margin: float,
    require_top_id_agreement: bool,
    max_source_top_score_gap: float,
    max_blended_top_score_gap: float,
    max_rows: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap = 0
    key_mismatch = 0
    public_rows = 0
    uctr_rows = 0
    output_rows = 0
    skipped_top_id_mismatch = 0
    skipped_source_top_score_gap = 0
    skipped_blended_top_score_gap = 0
    candidate_counts: list[int] = []
    positive_counts: list[int] = []
    candidate_id_overlaps: list[int] = []
    top_disagreements: list[float] = []
    source_top_id_matches: list[float] = []
    source_top_score_gaps: list[float] = []
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, (public_row, uctr_row) in enumerate(zip(_iter_jsonl(public_path), _iter_jsonl(uctr_path))):
            if max_rows > 0 and idx >= max_rows:
                break
            public_rows += 1
            uctr_rows += 1
            public_key = _row_key(public_row, idx)
            uctr_key = _row_key(uctr_row, idx)
            if public_key != uctr_key:
                key_mismatch += 1
                continue
            overlap += 1
            blended, stats = _blend_row(
                public_row,
                uctr_row,
                public_weight=public_weight,
                uctr_weight=uctr_weight,
                matching_top_k=matching_top_k,
                soft_positive_margin=soft_positive_margin,
            )
            blended_top_gap = abs(float(stats["top_public_score"]) - float(stats["top_uctr_score"]))
            if require_top_id_agreement and not bool(stats["source_top_id_match"]):
                skipped_top_id_mismatch += 1
                continue
            if max_source_top_score_gap >= 0.0 and float(stats["source_top_score_gap"]) > max_source_top_score_gap:
                skipped_source_top_score_gap += 1
                continue
            if max_blended_top_score_gap >= 0.0 and blended_top_gap > max_blended_top_score_gap:
                skipped_blended_top_score_gap += 1
                continue
            out_f.write(json.dumps(blended, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_rows += 1
            candidate_counts.append(int(stats["candidate_count"]))
            positive_counts.append(int(stats["positive_count"]))
            candidate_id_overlaps.append(int(stats["candidate_id_overlap"]))
            top_disagreements.append(blended_top_gap)
            source_top_id_matches.append(1.0 if bool(stats["source_top_id_match"]) else 0.0)
            source_top_score_gaps.append(float(stats["source_top_score_gap"]))
    return {
        "public_path": str(public_path),
        "uctr_path": str(uctr_path),
        "output_path": str(output_path),
        "public_rows_read": public_rows,
        "uctr_rows_read": uctr_rows,
        "paired_rows": overlap,
        "key_mismatch_rows": key_mismatch,
        "output_rows": output_rows,
        "skipped_top_id_mismatch_rows": skipped_top_id_mismatch,
        "skipped_source_top_score_gap_rows": skipped_source_top_score_gap,
        "skipped_blended_top_score_gap_rows": skipped_blended_top_score_gap,
        "candidate_count_mean": sum(candidate_counts) / max(1, len(candidate_counts)),
        "positive_count_mean": sum(positive_counts) / max(1, len(positive_counts)),
        "candidate_id_overlap_mean": sum(candidate_id_overlaps) / max(1, len(candidate_id_overlaps)),
        "top_source_disagreement_abs_mean": sum(top_disagreements) / max(1, len(top_disagreements)),
        "source_top_id_match_rate": sum(source_top_id_matches) / max(1, len(source_top_id_matches)),
        "source_top_score_gap_mean": sum(source_top_score_gaps) / max(1, len(source_top_score_gaps)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-id blended public/UCTR labels for MobileCropNet v4 repair runs.")
    parser.add_argument("--public_train_jsonl", required=True, type=Path)
    parser.add_argument("--uctr_train_jsonl", required=True, type=Path)
    parser.add_argument("--public_val_jsonl", required=True, type=Path)
    parser.add_argument("--uctr_val_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--public_weight", type=float, default=0.4)
    parser.add_argument("--uctr_weight", type=float, default=0.6)
    parser.add_argument("--matching_top_k", type=int, default=2)
    parser.add_argument("--soft_positive_margin", type=float, default=0.08)
    parser.add_argument("--require_top_id_agreement", action="store_true")
    parser.add_argument("--max_source_top_score_gap", type=float, default=-1.0)
    parser.add_argument("--max_blended_top_score_gap", type=float, default=-1.0)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_weight = max(0.0, float(args.public_weight))
    uctr_weight = max(0.0, float(args.uctr_weight))
    if public_weight <= 0.0 and uctr_weight <= 0.0:
        raise ValueError("at least one of public_weight or uctr_weight must be positive")
    total = public_weight + uctr_weight
    public_weight /= total
    uctr_weight /= total
    train_summary = _build_split(
        public_path=args.public_train_jsonl,
        uctr_path=args.uctr_train_jsonl,
        output_path=args.output_dir / "train_blended_public_uctr.jsonl",
        public_weight=public_weight,
        uctr_weight=uctr_weight,
        matching_top_k=max(1, int(args.matching_top_k)),
        soft_positive_margin=float(args.soft_positive_margin),
        require_top_id_agreement=bool(args.require_top_id_agreement),
        max_source_top_score_gap=float(args.max_source_top_score_gap),
        max_blended_top_score_gap=float(args.max_blended_top_score_gap),
        max_rows=max(0, int(args.max_train_rows)),
    )
    val_summary = _build_split(
        public_path=args.public_val_jsonl,
        uctr_path=args.uctr_val_jsonl,
        output_path=args.output_dir / "val_blended_public_uctr.jsonl",
        public_weight=public_weight,
        uctr_weight=uctr_weight,
        matching_top_k=max(1, int(args.matching_top_k)),
        soft_positive_margin=float(args.soft_positive_margin),
        require_top_id_agreement=bool(args.require_top_id_agreement),
        max_source_top_score_gap=float(args.max_source_top_score_gap),
        max_blended_top_score_gap=float(args.max_blended_top_score_gap),
        max_rows=max(0, int(args.max_val_rows)),
    )
    summary = {
        "schema_version": "mobilecropnet_v4_blended_public_uctr_labels_v1",
        "public_weight": public_weight,
        "uctr_weight": uctr_weight,
        "matching_top_k": max(1, int(args.matching_top_k)),
        "soft_positive_margin": float(args.soft_positive_margin),
        "require_top_id_agreement": bool(args.require_top_id_agreement),
        "max_source_top_score_gap": float(args.max_source_top_score_gap),
        "max_blended_top_score_gap": float(args.max_blended_top_score_gap),
        "train": train_summary,
        "val": val_summary,
        "artifacts": {
            "train_jsonl": str(args.output_dir / "train_blended_public_uctr.jsonl"),
            "val_jsonl": str(args.output_dir / "val_blended_public_uctr.jsonl"),
            "summary_json": str(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
