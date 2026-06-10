from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from public_benchmark.staging import write_jsonl


def _nested_get(row: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_keymap(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(path):
        staged_id = str(row.get("staged_image_id", ""))
        if staged_id:
            out[staged_id].append(row)
    return dict(out)


def _bbox_from_candidate(candidate: dict[str, Any]) -> Optional[list[float]]:
    box = candidate.get("bbox_xyxy_norm", candidate.get("bbox_norm_xyxy"))
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None


def _score_from_candidate(candidate: dict[str, Any], score_field: str) -> Optional[float]:
    value = _nested_get(candidate, score_field) if "." in score_field else candidate.get(score_field)
    if value is None and score_field != "score":
        value = candidate.get("score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score


def _prediction_row(
    *,
    mapping: dict[str, Any],
    staged_image_id: str,
    candidate: dict[str, Any],
    score_field: str,
) -> Optional[dict[str, Any]]:
    bbox = _bbox_from_candidate(candidate)
    if bbox is None:
        return None
    score = _score_from_candidate(candidate, score_field)
    row = {
        "dataset": str(mapping.get("dataset", "")).lower(),
        "image_id": str(mapping.get("image_id", "")),
        "target_ar": mapping.get("target_ar"),
        "candidate_id": str(candidate.get("candidate_id", "")),
        "bbox_xyxy_norm": bbox,
        "score": score,
        "source": str(candidate.get("source", "")),
        "source_staged_image_id": staged_image_id,
        "source_sstk_target_ar": mapping.get("sstk_target_ar"),
    }
    if isinstance(candidate.get("scores"), dict):
        row["scores"] = copy.deepcopy(candidate["scores"])
    if isinstance(candidate.get("meta"), dict):
        row["meta"] = copy.deepcopy(candidate["meta"])
    return row


def _teacher_candidates_for_group(ar_result: dict[str, Any], group: str) -> list[dict[str, Any]]:
    if group == "best_only":
        best = ar_result.get("best_candidate")
        return [best] if isinstance(best, dict) else []
    value = ar_result.get(group)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def export_predictions_from_teacher_scores(
    *,
    teacher_scores_jsonl: Path,
    keymap_jsonl: Path,
    output_jsonl: Path,
    score_field: str = "scores.crop_utility_raw",
    candidate_group: str = "utility_pool",
) -> dict[str, Any]:
    keymap = load_keymap(keymap_jsonl)
    rows: list[dict[str, Any]] = []
    missing_ar = 0
    for rec in _load_jsonl(teacher_scores_jsonl):
        staged_id = str(rec.get("image_id", ""))
        mappings = keymap.get(staged_id, [])
        if not mappings:
            continue
        scorer = rec.get("teacher_scorer", {}) if isinstance(rec.get("teacher_scorer"), dict) else {}
        by_ar = scorer.get("results_by_ar", {}) if isinstance(scorer.get("results_by_ar"), dict) else {}
        for mapping in mappings:
            ar_key = str(mapping.get("sstk_target_ar") or ("FREE" if mapping.get("target_ar") is None else mapping.get("target_ar")))
            ar_result = by_ar.get(ar_key)
            if not isinstance(ar_result, dict):
                missing_ar += 1
                continue
            for candidate in _teacher_candidates_for_group(ar_result, candidate_group):
                row = _prediction_row(mapping=mapping, staged_image_id=staged_id, candidate=candidate, score_field=score_field)
                if row is not None:
                    rows.append(row)
    count = write_jsonl(output_jsonl, rows)
    return {
        "source": str(teacher_scores_jsonl.resolve()),
        "keymap_jsonl": str(keymap_jsonl.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "score_field": score_field,
        "candidate_group": candidate_group,
        "prediction_rows": count,
        "missing_ar_results": missing_ar,
    }


def export_predictions_from_candidates(
    *,
    candidates_jsonl: Path,
    keymap_jsonl: Path,
    output_jsonl: Path,
    score_field: str = "score",
) -> dict[str, Any]:
    keymap = load_keymap(keymap_jsonl)
    rows: list[dict[str, Any]] = []
    missing_ar = 0
    for rec in _load_jsonl(candidates_jsonl):
        staged_id = str(rec.get("image_id", ""))
        mappings = keymap.get(staged_id, [])
        if not mappings:
            continue
        by_ar = rec.get("candidates_by_ar", {}) if isinstance(rec.get("candidates_by_ar"), dict) else {}
        for mapping in mappings:
            ar_key = str(mapping.get("sstk_target_ar") or ("FREE" if mapping.get("target_ar") is None else mapping.get("target_ar")))
            candidates = by_ar.get(ar_key)
            if not isinstance(candidates, list):
                missing_ar += 1
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                row = _prediction_row(mapping=mapping, staged_image_id=staged_id, candidate=candidate, score_field=score_field)
                if row is not None:
                    rows.append(row)
    count = write_jsonl(output_jsonl, rows)
    return {
        "source": str(candidates_jsonl.resolve()),
        "keymap_jsonl": str(keymap_jsonl.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "score_field": score_field,
        "prediction_rows": count,
        "missing_ar_results": missing_ar,
    }
