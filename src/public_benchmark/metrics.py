from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional

from public_benchmark.adapters import box_area, parse_ar, valid_box
from public_benchmark.schema import BenchmarkBox, BenchmarkTask


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(a) + box_area(b) - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def p50(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(median(vals)) if vals else 0.0


def p10(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    idx = int(math.floor(0.10 * (len(vals) - 1)))
    return float(vals[idx])


def _as_candidate_dict(box: BenchmarkBox) -> dict[str, Any]:
    return {
        "candidate_id": box.candidate_id,
        "bbox_xyxy_norm": [float(v) for v in box.bbox_xyxy_norm],
        "source": box.source,
        "label": box.label,
        "score": box.score,
        "meta": dict(box.meta),
    }


def _bbox_from_candidate(row: dict[str, Any]) -> list[float]:
    box = row.get("bbox_xyxy_norm", row.get("bbox_norm_xyxy"))
    if not isinstance(box, list) or len(box) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    out = [float(v) for v in box]
    if not valid_box(out):
        return [0.0, 0.0, 1.0, 1.0]
    return out


def _candidate_ar(box: list[float], *, image_aspect: float = 1.0) -> float:
    normalized_ar = max(1e-6, (float(box[2]) - float(box[0])) / max(1e-6, float(box[3]) - float(box[1])))
    return float(normalized_ar * max(1e-6, float(image_aspect)))


def _ar_rel_error(row: dict[str, Any], target_ar: Optional[str], *, image_aspect: float = 1.0) -> Optional[float]:
    if not target_ar:
        return None
    target = parse_ar(target_ar)
    ar = _candidate_ar(_bbox_from_candidate(row), image_aspect=image_aspect)
    return abs(ar / max(1e-6, target) - 1.0)


def _is_ar_compliant(row: dict[str, Any], target_ar: Optional[str], tolerance: float, *, image_aspect: float = 1.0) -> bool:
    err = _ar_rel_error(row, target_ar, image_aspect=image_aspect)
    return True if err is None else err <= float(tolerance)


def _filter_candidates_by_ar(
    candidates: list[dict[str, Any]],
    *,
    target_ar: Optional[str],
    ar_tolerance: float,
    mode: str,
    image_aspect: float = 1.0,
    keep_injected_gt: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    if str(mode).lower() in {"", "none", "off", "0"} or not target_ar:
        return candidates, 0
    if str(mode).lower() != "hard":
        raise ValueError(f"Unsupported target_ar_filter_mode: {mode}")
    out: list[dict[str, Any]] = []
    dropped = 0
    for cand in candidates:
        if keep_injected_gt and bool(cand.get("is_injected_gt", False)):
            out.append(cand)
            continue
        if _is_ar_compliant(cand, target_ar, ar_tolerance, image_aspect=image_aspect):
            out.append(cand)
        else:
            dropped += 1
    return out, dropped


def _nested_get(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _candidate_score(
    row: dict[str, Any],
    *,
    score_field: str,
    target_ar: Optional[str],
    scoring_fallback: str,
    image_aspect: float = 1.0,
) -> tuple[float, bool]:
    value = _nested_get(row, score_field) if "." in score_field else row.get(score_field)
    if value is None and score_field != "score":
        value = row.get("score")
    try:
        score = float(value)
        if math.isfinite(score):
            return score, False
    except (TypeError, ValueError):
        pass
    if scoring_fallback == "none":
        return float("nan"), True
    return center_area_prior_score(_bbox_from_candidate(row), target_ar=target_ar, image_aspect=image_aspect), True


def center_area_prior_score(box: list[float], *, target_ar: Optional[str], image_aspect: float = 1.0) -> float:
    area = box_area(box)
    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    center_dist = min(1.0, math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5))
    area_pref = 1.0 - abs(area - 0.62) / 0.62
    score = 0.55 * max(0.0, area_pref) + 0.35 * (1.0 - center_dist)
    if target_ar:
        ar = _candidate_ar(box, image_aspect=image_aspect)
        target = parse_ar(target_ar)
        ar_err = abs(math.log(max(1e-6, ar / target)))
        score += 0.10 * max(0.0, 1.0 - ar_err / math.log(2.0))
    return float(score)


def _task_image_aspect(task: BenchmarkTask) -> float:
    try:
        width = float(task.meta.get("image_width", 1.0))
        height = float(task.meta.get("image_height", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(1e-6, width / max(1e-6, height))


def load_predictions(path: Optional[Path]) -> dict[tuple[str, str, Optional[str]], list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    out: dict[tuple[str, str, Optional[str]], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row.get("dataset", "")).lower()
            image_id = str(row.get("image_id", ""))
            target_ar = row.get("target_ar")
            if target_ar == "":
                target_ar = None
            key = (dataset, image_id, target_ar)
            candidates = row.get("candidates")
            if isinstance(candidates, list):
                for cand in candidates:
                    if isinstance(cand, dict):
                        out[key].append(copy.deepcopy(cand))
            elif "bbox_xyxy_norm" in row or "bbox_norm_xyxy" in row:
                out[key].append(copy.deepcopy(row))
    return dict(out)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_boxes: set[tuple[float, float, float, float]] = set()
    for cand in candidates:
        cid = str(cand.get("candidate_id", ""))
        if cid and cid in seen_ids:
            continue
        box = _bbox_from_candidate(cand)
        key = tuple(round(float(v), 5) for v in box)
        if key in seen_boxes:
            continue
        if cid:
            seen_ids.add(cid)
        seen_boxes.add(key)
        cand = copy.deepcopy(cand)
        cand["bbox_xyxy_norm"] = box
        out.append(cand)
    return out


def validate_task(task: BenchmarkTask) -> list[str]:
    issues: list[str] = []
    if not task.image_id:
        issues.append("missing_image_id")
    if not task.image_path:
        issues.append("missing_image_path")
    for idx, gt in enumerate(task.gt_boxes):
        if not valid_box(gt.bbox_xyxy_norm):
            issues.append(f"invalid_gt_box_{idx}")
    for idx, cand in enumerate(task.candidate_windows):
        if not valid_box(cand.bbox_xyxy_norm):
            issues.append(f"invalid_candidate_box_{idx}")
    for idx, pair in enumerate(task.pairwise):
        if pair.preferred not in {"a", "b"}:
            issues.append(f"invalid_pairwise_preference_{idx}")
    return issues


def _compact_pairwise_candidate(cand: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(cand.get("candidate_id", "")),
        "bbox_xyxy_norm": _bbox_from_candidate(cand),
        "source": str(cand.get("source", "")),
        "score": cand.get("score"),
        "eval_score": float(cand.get("_eval_score", 0.0)),
        "scores": copy.deepcopy(cand.get("scores", {})) if isinstance(cand.get("scores"), dict) else {},
        "macro_scores": copy.deepcopy(cand.get("macro_scores", {})) if isinstance(cand.get("macro_scores"), dict) else {},
    }


def _pairwise_failure_detail(
    *,
    task: BenchmarkTask,
    pair_index: int,
    pair: Any,
    cand_a: dict[str, Any],
    cand_b: dict[str, Any],
    score_a: float,
    score_b: float,
    hit: float,
) -> dict[str, Any]:
    predicted = "tie"
    if score_a > score_b:
        predicted = "a"
    elif score_b > score_a:
        predicted = "b"
    preferred_score = score_a if pair.preferred == "a" else score_b
    other_score = score_b if pair.preferred == "a" else score_a
    return {
        "dataset": task.dataset,
        "split": task.split,
        "image_id": task.image_id,
        "target_ar": task.target_ar,
        "pair_index": int(pair_index),
        "candidate_id_a": pair.candidate_id_a,
        "candidate_id_b": pair.candidate_id_b,
        "preferred": pair.preferred,
        "predicted": predicted,
        "hit": float(hit),
        "weight": float(pair.weight),
        "score_a": float(score_a),
        "score_b": float(score_b),
        "preferred_score_minus_other": float(preferred_score - other_score),
        "votes": dict(pair.votes),
        "meta": dict(pair.meta),
        "candidate_a": _compact_pairwise_candidate(cand_a),
        "candidate_b": _compact_pairwise_candidate(cand_b),
    }


def evaluate_task(
    task: BenchmarkTask,
    *,
    predictions: list[dict[str, Any]],
    mode: str = "S",
    inject_gt: bool = True,
    score_field: str = "score",
    scoring_fallback: str = "center_area_prior",
    coverage_thresholds: tuple[float, ...] = (0.5, 0.7, 0.9),
    ar_tolerance: float = 0.03,
    target_ar_filter_mode: str = "none",
    collect_pairwise_details: bool = False,
    max_pairwise_details: int = 20,
) -> dict[str, Any]:
    image_aspect = _task_image_aspect(task)
    benchmark_candidates = [_as_candidate_dict(box) for box in task.candidate_windows]
    pred_candidates = [copy.deepcopy(row) for row in predictions]
    if mode.upper() == "SC":
        preinject_unfiltered = _dedupe_candidates(benchmark_candidates + pred_candidates)
    elif pred_candidates:
        preinject_unfiltered = _dedupe_candidates(pred_candidates)
    else:
        preinject_unfiltered = _dedupe_candidates(benchmark_candidates)
    preinject, ar_filter_drop_count_preinject = _filter_candidates_by_ar(
        preinject_unfiltered,
        target_ar=task.target_ar,
        ar_tolerance=ar_tolerance,
        mode=target_ar_filter_mode,
        image_aspect=image_aspect,
        keep_injected_gt=False,
    )

    pool = list(preinject)
    if inject_gt:
        for idx, gt in enumerate(task.gt_boxes):
            pool.append(
                {
                    "candidate_id": gt.candidate_id or f"gt_{idx}",
                    "bbox_xyxy_norm": [float(v) for v in gt.bbox_xyxy_norm],
                    "source": gt.source or "benchmark_gt",
                    "label": gt.label or "gt",
                    "score": gt.score,
                    "meta": dict(gt.meta),
                    "is_injected_gt": True,
                }
            )
    pool_unfiltered = _dedupe_candidates(pool)
    pool, ar_filter_drop_count_pool = _filter_candidates_by_ar(
        pool_unfiltered,
        target_ar=task.target_ar,
        ar_tolerance=ar_tolerance,
        mode=target_ar_filter_mode,
        image_aspect=image_aspect,
        keep_injected_gt=True,
    )

    missing_scores = 0
    nan_scores = 0
    for cand in pool:
        score, was_missing = _candidate_score(
            cand,
            score_field=score_field,
            target_ar=task.target_ar,
            scoring_fallback=scoring_fallback,
            image_aspect=image_aspect,
        )
        if was_missing:
            missing_scores += 1
        if not math.isfinite(score):
            nan_scores += 1
            score = -1e18
        cand["_eval_score"] = float(score)

    sorted_pool = sorted(pool, key=lambda row: float(row.get("_eval_score", -1e18)), reverse=True)
    sorted_preinject = sorted(
        preinject,
        key=lambda row: _candidate_score(
            row,
            score_field=score_field,
            target_ar=task.target_ar,
            scoring_fallback=scoring_fallback,
            image_aspect=image_aspect,
        )[0],
        reverse=True,
    )
    top1 = sorted_pool[0] if sorted_pool else None
    top1_preinject = sorted_preinject[0] if sorted_preinject else None

    gt_ranks: list[int] = []
    gt_margins: list[float] = []
    gt_iou_top1: list[float] = []
    gt_iou_top1_preinject: list[float] = []
    coverage: dict[str, list[float]] = {f"coverage_at_{str(thr).replace('.', '')}": [] for thr in coverage_thresholds}
    coverage_pre: dict[str, list[float]] = {f"coverage_at_{str(thr).replace('.', '')}_preinject": [] for thr in coverage_thresholds}
    for gt in task.gt_boxes:
        gt_box = gt.bbox_xyxy_norm
        best_gt_idx = -1
        best_gt_iou = -1.0
        for idx, cand in enumerate(sorted_pool):
            cur_iou = iou_xyxy(_bbox_from_candidate(cand), gt_box)
            if cur_iou > best_gt_iou:
                best_gt_iou = cur_iou
                best_gt_idx = idx
        if best_gt_idx >= 0:
            gt_ranks.append(best_gt_idx + 1)
            gt_score = float(sorted_pool[best_gt_idx].get("_eval_score", -1e18))
            non_gt_scores = [
                float(cand.get("_eval_score", -1e18))
                for cand in sorted_pool
                if iou_xyxy(_bbox_from_candidate(cand), gt_box) < 0.999
            ]
            gt_margins.append(gt_score - (max(non_gt_scores) if non_gt_scores else gt_score))
        if top1 is not None:
            gt_iou_top1.append(iou_xyxy(_bbox_from_candidate(top1), gt_box))
        if top1_preinject is not None:
            gt_iou_top1_preinject.append(iou_xyxy(_bbox_from_candidate(top1_preinject), gt_box))
        max_iou = max([iou_xyxy(_bbox_from_candidate(cand), gt_box) for cand in pool] or [0.0])
        max_iou_pre = max([iou_xyxy(_bbox_from_candidate(cand), gt_box) for cand in preinject] or [0.0])
        for thr in coverage_thresholds:
            coverage[f"coverage_at_{str(thr).replace('.', '')}"].append(1.0 if max_iou >= thr else 0.0)
            coverage_pre[f"coverage_at_{str(thr).replace('.', '')}_preinject"].append(1.0 if max_iou_pre >= thr else 0.0)

    cand_by_id = {str(cand.get("candidate_id", "")): cand for cand in pool if str(cand.get("candidate_id", ""))}
    pair_hits = 0.0
    pair_weighted_hits = 0.0
    pair_total = 0
    pair_weight_total = 0.0
    hard_hits = 0.0
    hard_total = 0
    pairwise_failures: list[dict[str, Any]] = []
    for pair_idx, pair in enumerate(task.pairwise):
        cand_a = cand_by_id.get(pair.candidate_id_a)
        cand_b = cand_by_id.get(pair.candidate_id_b)
        if cand_a is None or cand_b is None:
            continue
        score_a = float(cand_a.get("_eval_score", -1e18))
        score_b = float(cand_b.get("_eval_score", -1e18))
        if score_a == score_b:
            hit = 0.5
        elif (score_a > score_b and pair.preferred == "a") or (score_b > score_a and pair.preferred == "b"):
            hit = 1.0
        else:
            hit = 0.0
        weight = max(1e-6, float(pair.weight))
        pair_hits += hit
        pair_weighted_hits += hit * weight
        pair_total += 1
        pair_weight_total += weight
        if weight >= 3.0:
            hard_hits += hit
            hard_total += 1
        if collect_pairwise_details and hit < 1.0:
            pairwise_failures.append(
                _pairwise_failure_detail(
                    task=task,
                    pair_index=pair_idx,
                    pair=pair,
                    cand_a=cand_a,
                    cand_b=cand_b,
                    score_a=score_a,
                    score_b=score_b,
                    hit=hit,
                )
            )
    pairwise_failures.sort(key=lambda item: (float(item.get("weight", 0.0)) * (1.0 - float(item.get("hit", 0.0))), float(item.get("weight", 0.0))), reverse=True)
    if max_pairwise_details >= 0:
        pairwise_failures = pairwise_failures[: int(max_pairwise_details)]

    ar_violation = 0.0
    if top1 is not None and task.target_ar:
        box = _bbox_from_candidate(top1)
        ar = _candidate_ar(box, image_aspect=image_aspect)
        target = parse_ar(task.target_ar)
        ar_violation = 1.0 if abs(ar / target - 1.0) > float(ar_tolerance) else 0.0

    keep_full = 0.0
    if top1 is not None:
        keep_full = 1.0 if iou_xyxy(_bbox_from_candidate(top1), [0.0, 0.0, 1.0, 1.0]) >= 0.995 else 0.0

    schema_issues = validate_task(task)
    top1_pre_score = None
    if top1_preinject is not None:
        top1_pre_score = _candidate_score(
            top1_preinject,
            score_field=score_field,
            target_ar=task.target_ar,
            scoring_fallback=scoring_fallback,
            image_aspect=image_aspect,
        )[0]
    row: dict[str, Any] = {
        "dataset": task.dataset,
        "split": task.split,
        "image_id": task.image_id,
        "image_path": task.image_path,
        "target_ar": task.target_ar,
        "image_aspect": image_aspect,
        "task_type": task.task_type,
        "target_ar_filter_mode": str(target_ar_filter_mode),
        "candidate_count_preinject_unfiltered": len(preinject_unfiltered),
        "ar_filter_drop_count_preinject": ar_filter_drop_count_preinject,
        "ar_filter_drop_count": ar_filter_drop_count_pool,
        "candidate_count_preinject": len(preinject),
        "candidate_count": len(pool),
        "gt_count": len(task.gt_boxes),
        "pairwise_count": pair_total,
        "pairwise_acc": pair_hits / pair_total if pair_total else None,
        "weighted_pairwise_acc": pair_weighted_hits / pair_weight_total if pair_weight_total else None,
        "hard_case_pairwise_acc": hard_hits / hard_total if hard_total else None,
        "gt_rank_best": min(gt_ranks) if gt_ranks else None,
        "gt_rank_at_1": 1.0 if gt_ranks and min(gt_ranks) <= 1 else 0.0,
        "gt_rank_at_5": 1.0 if gt_ranks and min(gt_ranks) <= 5 else 0.0,
        "gt_margin_best": max(gt_margins) if gt_margins else None,
        "iou_top1": mean(gt_iou_top1),
        "iou_top1_preinject": mean(gt_iou_top1_preinject),
        "ar_constraint_violation": ar_violation,
        "keep_full": keep_full,
        "missing_score_count": missing_scores,
        "nan_score_count": nan_scores,
        "schema_issue_count": len(schema_issues),
        "schema_issues": schema_issues,
        "gt_boxes": [[float(v) for v in gt.bbox_xyxy_norm] for gt in task.gt_boxes],
        "top1_candidate_id": str(top1.get("candidate_id", "")) if top1 else "",
        "top1_source": str(top1.get("source", "")) if top1 else "",
        "top1_score": float(top1.get("_eval_score", 0.0)) if top1 else 0.0,
        "top1_bbox_xyxy_norm": _bbox_from_candidate(top1) if top1 else None,
        "top1_preinject_candidate_id": str(top1_preinject.get("candidate_id", "")) if top1_preinject else "",
        "top1_preinject_source": str(top1_preinject.get("source", "")) if top1_preinject else "",
        "top1_preinject_score": float(top1_pre_score) if top1_pre_score is not None and math.isfinite(float(top1_pre_score)) else None,
        "top1_preinject_bbox_xyxy_norm": _bbox_from_candidate(top1_preinject) if top1_preinject else None,
    }
    if collect_pairwise_details:
        row["pairwise_failures"] = pairwise_failures
        row["pairwise_failure_count_recorded"] = len(pairwise_failures)
    for key, values in coverage.items():
        row[key] = mean(values)
    for key, values in coverage_pre.items():
        row[key] = mean(values)
    return row


def aggregate_rows(rows: list[dict[str, Any]], *, include_by_ar: bool = True) -> dict[str, Any]:
    if not rows:
        return {}

    def numeric(name: str) -> list[float]:
        vals: list[float] = []
        for row in rows:
            value = row.get(name)
            if value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fval):
                vals.append(fval)
        return vals

    pair_weights = numeric("pairwise_count")
    pair_accs = numeric("weighted_pairwise_acc")
    ar_filter_drops = numeric("ar_filter_drop_count_preinject")
    ar_filter_candidates = numeric("candidate_count_preinject_unfiltered")
    summary: dict[str, Any] = {
        "n_tasks": len(rows),
        "n_images": len({str(row.get("image_id", "")) for row in rows}),
        "n_pairwise": int(sum(pair_weights)),
        "ar_filter_drop_count_preinject": int(sum(ar_filter_drops)),
        "ar_filter_drop_rate_preinject": sum(ar_filter_drops) / max(1.0, sum(ar_filter_candidates)),
        "schema_fail_rate": mean(1.0 if float(row.get("schema_issue_count", 0)) > 0 else 0.0 for row in rows),
        "nan_score_rate": sum(float(row.get("nan_score_count", 0)) for row in rows) / max(1.0, sum(float(row.get("candidate_count", 0)) for row in rows)),
        "missing_score_rate": sum(float(row.get("missing_score_count", 0)) for row in rows) / max(1.0, sum(float(row.get("candidate_count", 0)) for row in rows)),
        "gt_rank_at_1": mean(numeric("gt_rank_at_1")),
        "gt_rank_at_5": mean(numeric("gt_rank_at_5")),
        "gt_margin_p50": p50(numeric("gt_margin_best")),
        "gt_margin_p10": p10(numeric("gt_margin_best")),
        "iou_top1": mean(numeric("iou_top1")),
        "iou_top1_preinject": mean(numeric("iou_top1_preinject")),
        "coverage_at_05": mean(numeric("coverage_at_05")),
        "coverage_at_07": mean(numeric("coverage_at_07")),
        "coverage_at_09": mean(numeric("coverage_at_09")),
        "coverage_at_05_preinject": mean(numeric("coverage_at_05_preinject")),
        "coverage_at_07_preinject": mean(numeric("coverage_at_07_preinject")),
        "coverage_at_09_preinject": mean(numeric("coverage_at_09_preinject")),
        "ar_constraint_violation_rate": mean(numeric("ar_constraint_violation")),
        "keep_full_rate": mean(numeric("keep_full")),
    }
    if pair_accs:
        summary["weighted_pairwise_acc"] = mean(pair_accs)
        summary["pairwise_acc"] = mean(numeric("pairwise_acc"))
        summary["hard_case_pairwise_acc"] = mean(numeric("hard_case_pairwise_acc"))
    else:
        summary["weighted_pairwise_acc"] = None
        summary["pairwise_acc"] = None
        summary["hard_case_pairwise_acc"] = None
    by_ar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_ar = row.get("target_ar")
        if target_ar:
            by_ar[str(target_ar)].append(row)
    if include_by_ar and by_ar:
        summary["by_target_ar"] = {key: aggregate_rows(value, include_by_ar=False) for key, value in sorted(by_ar.items())}
    return summary
