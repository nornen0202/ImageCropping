from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from mobilecropnet_v4.data import box_iou_xyxy, safe_float
from mobilecropnet_v4.eval_utils import mean, pearson_corr, spearman_corr


DEFAULT_RETURN_K = (1, 2, 3, 4)
DEFAULT_TOP_N = (5, 10)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def coco_bbox_to_norm_xyxy(bbox: Sequence[float], *, width: int, height: int) -> list[float]:
    x, y, w, h = [safe_float(v) for v in bbox[:4]]
    width_f = float(max(1, int(width)))
    height_f = float(max(1, int(height)))
    x1 = _clamp01(x / width_f)
    y1 = _clamp01(y / height_f)
    x2 = _clamp01((x + max(0.0, w)) / width_f)
    y2 = _clamp01((y + max(0.0, h)) / height_f)
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-6)
    return [x1, y1, x2, y2]


def resolve_image_path(file_name: str, image_roots: Sequence[Path]) -> Path | None:
    name = Path(str(file_name)).name
    for root in image_roots:
        candidate = Path(root) / name
        if candidate.exists():
            return candidate
    return None


def load_gaic_annotation_records(
    annotation_json: Path,
    *,
    image_roots: Sequence[Path] = (),
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    images = {int(row["id"]): dict(row) for row in payload.get("images", [])}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        grouped.setdefault(image_id, []).append(dict(ann))

    records: list[dict[str, Any]] = []
    for image_id in sorted(images):
        image = images[image_id]
        width = int(image["width"])
        height = int(image["height"])
        anns = sorted(grouped.get(image_id, []), key=lambda row: int(row.get("id", 0)))
        candidates = []
        for ann in anns:
            candidates.append(
                {
                    "annotation_id": int(ann.get("id", 0)),
                    "candidate_id": f"gaic_ann_{int(ann.get('id', 0)):06d}",
                    "bbox_norm_xyxy": coco_bbox_to_norm_xyxy(ann.get("bbox", [0, 0, width, height]), width=width, height=height),
                    "bbox_xywh": [safe_float(v) for v in ann.get("bbox", [])[:4]],
                    "mos": safe_float(ann.get("score"), 0.0),
                    "gt_flag": int(ann.get("gt_flag", 0)),
                    "area": safe_float(ann.get("area"), 0.0),
                }
            )
        image_path = resolve_image_path(str(image.get("file_name", "")), image_roots)
        records.append(
            {
                "image_id": str(image_id),
                "file_name": str(image.get("file_name", "")),
                "width": width,
                "height": height,
                "image_path": str(image_path) if image_path is not None else "",
                "candidates": candidates,
            }
        )
        if max_images is not None and len(records) >= int(max_images):
            break
    return records


def ranks_desc(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate([float(v) for v in values]), key=lambda item: item[1], reverse=True)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = 0.5 * (float(pos + 1) + float(end))
        for idx, _ in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def order_desc(values: Sequence[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: (float(values[idx]), -idx), reverse=True)


def ordinal_ranks_desc(values: Sequence[float]) -> list[int]:
    ranks = [0] * len(values)
    for rank, idx in enumerate(order_desc(values), 1):
        ranks[idx] = rank
    return ranks


def acc_k_of_top_n(mos: Sequence[float], scores: Sequence[float], *, k: int, n: int) -> float:
    if not mos or len(mos) != len(scores):
        return 0.0
    top_n = set(order_desc(mos)[: min(int(n), len(mos))])
    top_k = order_desc(scores)[: min(int(k), len(scores))]
    if not top_k:
        return 0.0
    return float(sum(1 for idx in top_k if idx in top_n) / float(int(k)))


def rank_weighted_acc_k_of_top_n(
    mos: Sequence[float],
    scores: Sequence[float],
    *,
    k: int,
    n: int,
    beta: float = 1.0,
) -> float:
    if not mos or len(mos) != len(scores):
        return 0.0
    k = int(k)
    n = int(n)
    top_k = order_desc(scores)[: min(k, len(scores))]
    if not top_k:
        return 0.0
    mos_ranks = ordinal_ranks_desc(mos)
    returned_sorted_by_mos = sorted(top_k, key=lambda idx: float(mos[idx]), reverse=True)
    total = 0.0
    for j, idx in enumerate(returned_sorted_by_mos, 1):
        rank = float(mos_ranks[idx])
        if rank <= float(n):
            total += math.exp(-float(beta) * (rank - float(j)) / float(n))
    return float(total / float(k))


def per_image_gaic_metrics(
    mos: Sequence[float],
    scores: Sequence[float],
    *,
    return_k_values: Sequence[int] = DEFAULT_RETURN_K,
    top_n_values: Sequence[int] = DEFAULT_TOP_N,
    beta: float = 1.0,
) -> dict[str, float]:
    if not mos or len(mos) != len(scores):
        return {}
    mos_vals = [float(v) for v in mos]
    score_vals = [float(v) for v in scores]
    top_idx = order_desc(score_vals)[0]
    mos_order = order_desc(mos_vals)
    best_mos = float(mos_vals[mos_order[0]])
    rank_by_idx = {idx: rank for rank, idx in enumerate(mos_order, 1)}
    top_rank = int(rank_by_idx.get(top_idx, len(mos_vals)))
    denom = max(1, len(mos_vals) - 1)
    out = {
        "pcc": pearson_corr(score_vals, mos_vals),
        "srcc": spearman_corr(score_vals, mos_vals),
        "top1_mos": float(mos_vals[top_idx]),
        "best_mos": best_mos,
        "top1_mos_regret": max(0.0, best_mos - float(mos_vals[top_idx])),
        "top1_rank": float(top_rank),
        "top1_rank_percentile": float(1.0 - (top_rank - 1) / denom),
        "candidate_count": float(len(mos_vals)),
    }
    for n in top_n_values:
        out[f"top1_in_top{int(n)}"] = float(top_rank <= int(n))
    for k in return_k_values:
        for n in top_n_values:
            out[f"acc{int(k)}_of_top{int(n)}"] = acc_k_of_top_n(mos_vals, score_vals, k=int(k), n=int(n))
            out[f"accw{int(k)}_of_top{int(n)}"] = rank_weighted_acc_k_of_top_n(
                mos_vals,
                score_vals,
                k=int(k),
                n=int(n),
                beta=float(beta),
            )
    return out


def summarize_metric_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean([row[key] for row in rows if key in row]) for key in keys}


def evaluate_scored_records(
    records: Sequence[dict[str, Any]],
    scores_by_image_id: dict[str, Sequence[float]],
    *,
    method_name: str,
    coverage_by_image_id: dict[str, dict[str, float]] | None = None,
    return_k_values: Sequence[int] = DEFAULT_RETURN_K,
    top_n_values: Sequence[int] = DEFAULT_TOP_N,
    beta: float = 1.0,
) -> dict[str, Any]:
    per_image: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    missing: list[str] = []
    for record in records:
        image_id = str(record["image_id"])
        candidates = list(record.get("candidates") or [])
        scores = scores_by_image_id.get(image_id)
        if scores is None or len(scores) != len(candidates):
            missing.append(image_id)
            continue
        mos = [safe_float(c.get("mos"), 0.0) for c in candidates]
        metrics = per_image_gaic_metrics(mos, scores, return_k_values=return_k_values, top_n_values=top_n_values, beta=beta)
        if not metrics:
            missing.append(image_id)
            continue
        coverage = (coverage_by_image_id or {}).get(image_id, {})
        for key, value in coverage.items():
            metrics[f"coverage_{key}"] = float(value)
        metric_rows.append(metrics)
        pred_order = order_desc(scores)
        mos_order = order_desc(mos)
        per_image.append(
            {
                "image_id": image_id,
                "file_name": record.get("file_name", ""),
                "method": method_name,
                "candidate_count": len(candidates),
                "metrics": metrics,
                "top_predictions": [
                    {
                        "rank": rank,
                        "candidate_index": int(idx),
                        "annotation_id": candidates[idx].get("annotation_id"),
                        "mos": mos[idx],
                        "score": float(scores[idx]),
                        "bbox_norm_xyxy": candidates[idx].get("bbox_norm_xyxy"),
                    }
                    for rank, idx in enumerate(pred_order[:10], 1)
                ],
                "top_mos": [
                    {
                        "rank": rank,
                        "candidate_index": int(idx),
                        "annotation_id": candidates[idx].get("annotation_id"),
                        "mos": mos[idx],
                        "score": float(scores[idx]),
                        "bbox_norm_xyxy": candidates[idx].get("bbox_norm_xyxy"),
                    }
                    for rank, idx in enumerate(mos_order[:10], 1)
                ],
            }
        )
    summary = summarize_metric_rows(metric_rows)
    return {
        "method": method_name,
        "image_count": len(metric_rows),
        "missing_image_count": len(missing),
        "missing_image_ids": missing[:50],
        "metrics": summary,
        "per_image": per_image,
    }


def _candidate_score(candidate: dict[str, Any], score_field: str) -> float | None:
    scores = candidate.get("scores")
    if isinstance(scores, dict) and scores.get(score_field) is not None:
        return safe_float(scores.get(score_field))
    if candidate.get(score_field) is not None:
        return safe_float(candidate.get(score_field))
    for key in ("crop_utility_raw", "score_policy", "policy_safe", "score_rank", "rank", "final"):
        if isinstance(scores, dict) and scores.get(key) is not None:
            return safe_float(scores.get(key))
        if candidate.get(key) is not None:
            return safe_float(candidate.get(key))
    return None


def extract_teacher_candidates(
    teacher_row: dict[str, Any],
    *,
    target_ar: str = "FREE",
    score_field: str = "crop_utility_raw",
) -> list[dict[str, Any]]:
    result = teacher_row.get("teacher_scorer", {}).get("results_by_ar", {}).get(str(target_ar), {})
    if not isinstance(result, dict):
        return []
    pools: list[Any] = []
    for key in ("rank_pool", "utility_pool", "selected_topk", "cheap_top_m", "also_considered_rejected"):
        value = result.get(key)
        if isinstance(value, list):
            pools.extend(value)
    for key in ("best_candidate", "baseline_candidate"):
        value = result.get(key)
        if isinstance(value, dict):
            pools.append(value)

    out: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for candidate in pools:
        if not isinstance(candidate, dict):
            continue
        box = candidate.get("bbox_norm_xyxy")
        if not isinstance(box, list) or len(box) < 4:
            continue
        score = _candidate_score(candidate, score_field)
        if score is None:
            continue
        norm_box = [_clamp01(safe_float(v)) for v in box[:4]]
        key = tuple(round(v, 6) for v in norm_box)
        prev = out.get(key)
        if prev is None or float(score) > float(prev["score"]):
            out[key] = {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "bbox_norm_xyxy": norm_box,
                "score": float(score),
                "source": str(candidate.get("source", "")),
            }
    return list(out.values())


def load_teacher_rows(
    teacher_jsonl: Path,
    *,
    image_ids: Iterable[str],
    target_ar: str = "FREE",
    score_field: str = "crop_utility_raw",
) -> dict[str, list[dict[str, Any]]]:
    needed = {str(v) for v in image_ids}
    out: dict[str, list[dict[str, Any]]] = {}
    with Path(teacher_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", ""))
            if image_id not in needed:
                continue
            out[image_id] = extract_teacher_candidates(row, target_ar=target_ar, score_field=score_field)
            if len(out) >= len(needed):
                break
    return out


def load_teacher_candidate_eval_scores(
    candidate_eval_jsonl: Path,
    records: Sequence[dict[str, Any]],
    *,
    protocol: str = "Gc",
    score_field: str = "crop_utility_raw",
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
    image_id_to_annotation_index: dict[str, dict[int, int]] = {}
    image_id_to_box_index: dict[str, dict[tuple[float, float, float, float], int]] = {}
    for record in records:
        image_id = str(record["image_id"])
        image_id_to_annotation_index[image_id] = {
            int(candidate.get("annotation_id")): idx
            for idx, candidate in enumerate(record.get("candidates") or [])
            if candidate.get("annotation_id") is not None
        }
        image_id_to_box_index[image_id] = {
            tuple(round(float(v), 6) for v in candidate.get("bbox_norm_xyxy", [])[:4]): idx
            for idx, candidate in enumerate(record.get("candidates") or [])
            if isinstance(candidate.get("bbox_norm_xyxy"), list) and len(candidate.get("bbox_norm_xyxy")) >= 4
        }

    raw_scores: dict[str, list[float | None]] = {
        image_id: [None] * len(index_by_ann) for image_id, index_by_ann in image_id_to_annotation_index.items()
    }
    protocol = str(protocol)
    with Path(candidate_eval_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("protocol", "")) != protocol:
                continue
            image_id = str(row.get("image_id", ""))
            ann_index = image_id_to_annotation_index.get(image_id)
            if ann_index is None:
                continue
            ann_id_value = row.get("gt_annotation_id")
            if ann_id_value is None:
                candidate_id = str(row.get("candidate_id", ""))
                if candidate_id.startswith("gaic_gt_"):
                    ann_id_value = candidate_id.rsplit("_", 1)[-1]
            if ann_id_value is None:
                continue
            ann_id = int(ann_id_value)
            idx = ann_index.get(ann_id)
            if idx is None:
                box = row.get("bbox_norm_xyxy")
                if isinstance(box, list) and len(box) >= 4:
                    idx = image_id_to_box_index.get(image_id, {}).get(tuple(round(float(v), 6) for v in box[:4]))
            if idx is None:
                continue
            score = row.get(score_field)
            if score is None:
                continue
            raw_scores[image_id][idx] = safe_float(score)

    scores_by_image: dict[str, list[float]] = {}
    coverage_by_image: dict[str, dict[str, float]] = {}
    for image_id, values in raw_scores.items():
        total = len(values)
        present = [value for value in values if value is not None]
        if total <= 0 or not present:
            continue
        coverage_by_image[image_id] = {
            "annotation_score_rate": float(len(present) / max(1, total)),
            "teacher_candidate_count": float(len(present)),
        }
        if len(present) != total:
            continue
        scores_by_image[image_id] = [float(value) for value in values if value is not None]
    return scores_by_image, coverage_by_image


def project_teacher_scores_to_annotations(
    records: Sequence[dict[str, Any]],
    teacher_candidates_by_image_id: dict[str, list[dict[str, Any]]],
    *,
    match_iou_threshold: float = 0.5,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
    scores_by_image: dict[str, list[float]] = {}
    coverage_by_image: dict[str, dict[str, float]] = {}
    for record in records:
        image_id = str(record["image_id"])
        anns = list(record.get("candidates") or [])
        teacher_candidates = teacher_candidates_by_image_id.get(image_id, [])
        if not anns or not teacher_candidates:
            continue
        teacher_scores = [float(c["score"]) for c in teacher_candidates]
        low_score = min(teacher_scores) - max(1.0, abs(min(teacher_scores)) * 0.1)
        projected = [float(low_score)] * len(anns)
        matched = [0] * len(anns)
        matched_ious: list[float] = []
        for teacher_candidate in sorted(teacher_candidates, key=lambda c: float(c["score"]), reverse=True):
            teacher_box = teacher_candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
            best_idx = -1
            best_iou = -1.0
            for idx, ann in enumerate(anns):
                iou = box_iou_xyxy(teacher_box, ann.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= float(match_iou_threshold):
                projected[best_idx] = max(projected[best_idx], float(teacher_candidate["score"]))
                matched[best_idx] = 1
                matched_ious.append(float(best_iou))
        scores_by_image[image_id] = projected
        coverage_by_image[image_id] = {
            "annotation_match_rate": float(sum(matched) / max(1, len(matched))),
            "teacher_candidate_count": float(len(teacher_candidates)),
            "matched_teacher_candidate_count": float(len(matched_ious)),
            "matched_iou_mean": mean(matched_ious),
        }
    return scores_by_image, coverage_by_image
