from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from universal_crop_teacher.data import (
    box_iou,
    clamp_box,
    normalize_score,
    remap_project_path,
    safe_float,
    split_by_hash,
)


def read_jsonl(path: Path, *, max_rows: int = 0) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        count = 0
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if int(max_rows) > 0 and count >= int(max_rows):
                return


def _task_id(dataset: str, image_id: str, target_ar: Any) -> str:
    ar = "" if target_ar is None else str(target_ar)
    return f"{str(dataset).lower()}|{image_id}|{ar}"


def _image_dims_from_task(task: dict[str, Any]) -> tuple[float, float]:
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    width = safe_float(meta.get("image_width"), 0.0)
    height = safe_float(meta.get("image_height"), 0.0)
    return width, height


def _candidate_score_values(candidates: Sequence[dict[str, Any]]) -> list[float | None]:
    values: list[float | None] = []
    for cand in candidates:
        value: float | None = None
        if cand.get("score") is not None:
            value = safe_float(cand.get("score"), float("nan"))
        if value is None or not math.isfinite(value):
            meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
            if meta.get("score_norm") is not None:
                value = safe_float(meta.get("score_norm"), float("nan"))
        values.append(value if value is not None and math.isfinite(value) else None)
    return values


def public_task_to_warehouse_row(
    task: dict[str, Any],
    *,
    split_seed: int = 20260420,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    force_hash_split: bool = False,
) -> dict[str, Any]:
    dataset = str(task.get("dataset", "unknown")).lower()
    image_id = str(task.get("image_id", ""))
    target_ar = task.get("target_ar")
    task_id = _task_id(dataset, image_id, target_ar)
    split = str(task.get("split", "") or "").lower()
    if force_hash_split or split not in {"train", "val", "test"}:
        split = split_by_hash(task_id, val_fraction=val_fraction, test_fraction=test_fraction, seed=split_seed)

    raw_candidates = [dict(cand) for cand in task.get("candidate_windows", []) if isinstance(cand, dict)]
    raw_scores = _candidate_score_values(raw_candidates)
    finite_scores = [float(v) for v in raw_scores if v is not None and math.isfinite(float(v))]
    min_score = min(finite_scores) if finite_scores else 0.0
    max_score = max(finite_scores) if finite_scores else 1.0
    gt_boxes = [clamp_box(gt.get("bbox_xyxy_norm", gt.get("bbox_norm_xyxy"))) for gt in task.get("gt_boxes", []) if isinstance(gt, dict)]

    candidates: list[dict[str, Any]] = []
    for idx, cand in enumerate(raw_candidates):
        box = clamp_box(cand.get("bbox_xyxy_norm", cand.get("bbox_norm_xyxy")))
        gt_iou = max([box_iou(box, gt_box) for gt_box in gt_boxes] or [0.0])
        source_score = raw_scores[idx] if idx < len(raw_scores) else None
        if dataset == "cpc" and source_score is not None:
            target_score = normalize_score(float(source_score), min_value=min_score, max_value=max_score)
            confidence = 1.0
        elif gt_boxes:
            target_score = gt_iou
            confidence = 1.0 if gt_iou >= 0.5 else 0.75
        else:
            target_score = 0.5
            confidence = 0.2
        candidates.append(
            {
                "candidate_id": str(cand.get("candidate_id") or f"candidate_{idx:04d}"),
                "bbox_xyxy_norm": box,
                "source": str(cand.get("source", "")),
                "label": str(cand.get("label", "")),
                "target_score": float(max(0.0, min(1.0, target_score))),
                "label_confidence": float(max(0.0, min(1.0, confidence))),
                "gt_iou": float(gt_iou),
                "raw_annotation_score": float(source_score) if source_score is not None else None,
                "candidate_weight": safe_float(cand.get("weight"), 1.0),
                "meta": cand.get("meta", {}) if isinstance(cand.get("meta"), dict) else {},
            }
        )

    pairwise: list[dict[str, Any]] = []
    candidate_ids = {str(cand["candidate_id"]) for cand in candidates}
    for pair in task.get("pairwise", []) or []:
        if not isinstance(pair, dict):
            continue
        a = str(pair.get("candidate_id_a", ""))
        b = str(pair.get("candidate_id_b", ""))
        if a not in candidate_ids or b not in candidate_ids:
            continue
        preferred = str(pair.get("preferred", "")).lower()
        if preferred not in {"a", "b"}:
            continue
        pairwise.append(
            {
                "candidate_id_a": a,
                "candidate_id_b": b,
                "preferred": preferred,
                "weight": max(1e-6, safe_float(pair.get("weight"), 1.0)),
                "votes": pair.get("votes", {}) if isinstance(pair.get("votes"), dict) else {},
                "meta": pair.get("meta", {}) if isinstance(pair.get("meta"), dict) else {},
            }
        )

    width, height = _image_dims_from_task(task)
    image_path = remap_project_path(task.get("image_path", ""))
    return {
        "format": "universal_crop_teacher_warehouse_v1",
        "task_id": task_id,
        "dataset": dataset,
        "split": split,
        "source_split": str(task.get("split", "")),
        "image_id": image_id,
        "image_path": image_path,
        "task_type": str(task.get("task_type", "crop_ranking")),
        "target_ar": target_ar,
        "gt_boxes": task.get("gt_boxes", []),
        "pairwise": pairwise,
        "candidates": candidates,
        "meta": {
            **(task.get("meta", {}) if isinstance(task.get("meta"), dict) else {}),
            "image_width": width,
            "image_height": height,
            "source_task_manifest": True,
        },
    }


def gaic_record_to_warehouse_row(
    record: dict[str, Any],
    *,
    split_seed: int = 20260420,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    min_mos_gap: float = 0.20,
    max_pairs: int = 256,
    preserve_official_split: bool = True,
) -> dict[str, Any]:
    image_id = str(record.get("image_id", ""))
    task_id = _task_id("gaic", image_id, "FREE")
    official_split = str(record.get("official_split", "")).lower()
    if bool(preserve_official_split) and official_split in {"train", "val", "test"}:
        split = official_split
    else:
        split = split_by_hash(task_id, val_fraction=val_fraction, test_fraction=test_fraction, seed=split_seed)
    raw_candidates = [dict(cand) for cand in record.get("candidates", []) if isinstance(cand, dict)]
    mos_values = [safe_float(cand.get("mos"), 0.0) for cand in raw_candidates]
    min_mos = min(mos_values) if mos_values else 0.0
    max_mos = max(mos_values) if mos_values else 1.0

    candidates: list[dict[str, Any]] = []
    for idx, cand in enumerate(raw_candidates):
        mos = safe_float(cand.get("mos"), 0.0)
        box = clamp_box(cand.get("bbox_norm_xyxy", cand.get("bbox_xyxy_norm")))
        candidates.append(
            {
                "candidate_id": str(cand.get("candidate_id") or f"gaic_ann_{idx:06d}"),
                "bbox_xyxy_norm": box,
                "source": "gaic_annotation_candidate",
                "label": "gaic_mos_candidate",
                "target_score": normalize_score(mos, min_value=min_mos, max_value=max_mos),
                "label_confidence": 1.0,
                "gt_iou": 0.0,
                "mos": float(mos),
                "annotation_id": cand.get("annotation_id"),
                "gt_flag": int(safe_float(cand.get("gt_flag"), 0.0)),
                "meta": {},
            }
        )

    pair_candidates: list[tuple[float, int, int]] = []
    for i, mos_i in enumerate(mos_values):
        for j, mos_j in enumerate(mos_values):
            if i >= j:
                continue
            gap = abs(float(mos_i) - float(mos_j))
            if gap < float(min_mos_gap):
                continue
            pair_candidates.append((gap, i, j))
    pair_candidates.sort(reverse=True)
    pairwise: list[dict[str, Any]] = []
    for gap, i, j in pair_candidates[: max(0, int(max_pairs))]:
        preferred = "a" if mos_values[i] >= mos_values[j] else "b"
        pairwise.append(
            {
                "candidate_id_a": candidates[i]["candidate_id"],
                "candidate_id_b": candidates[j]["candidate_id"],
                "preferred": preferred,
                "weight": float(gap),
                "meta": {"mos_a": mos_values[i], "mos_b": mos_values[j]},
            }
        )

    return {
        "format": "universal_crop_teacher_warehouse_v1",
        "task_id": task_id,
        "dataset": "gaic",
        "split": split,
        "source_split": str(record.get("official_split", "annotation_json")),
        "image_id": image_id,
        "image_path": remap_project_path(record.get("image_path", "")),
        "task_type": "gaic_mos_ranking",
        "target_ar": "FREE",
        "gt_boxes": [],
        "pairwise": pairwise,
        "candidates": candidates,
        "meta": {
            "image_width": safe_float(record.get("width"), 0.0),
            "image_height": safe_float(record.get("height"), 0.0),
            "file_name": str(record.get("file_name", "")),
            "official_split": official_split,
            "annotation_json": str(record.get("annotation_json", "")),
            "source_gaic_annotations": True,
        },
    }


def summarize_warehouse(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(row.get("dataset", "unknown")) for row in rows)
    by_split = Counter(str(row.get("split", "")) for row in rows)
    split_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_counts: list[int] = []
    pair_counts: list[int] = []
    missing_images = 0
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        split = str(row.get("split", ""))
        split_by_dataset[dataset][split] += 1
        candidate_counts.append(len(row.get("candidates", []) or []))
        pair_counts.append(len(row.get("pairwise", []) or []))
        image_path = str(row.get("image_path", ""))
        if not image_path or not Path(image_path).exists():
            missing_images += 1
    return {
        "format": "universal_crop_teacher_warehouse_summary_v1",
        "task_count": len(rows),
        "candidate_count": int(sum(candidate_counts)),
        "pairwise_count": int(sum(pair_counts)),
        "missing_image_count": int(missing_images),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_split": dict(sorted(by_split.items())),
        "split_by_dataset": {key: dict(sorted(value.items())) for key, value in sorted(split_by_dataset.items())},
        "candidate_count_mean": float(sum(candidate_counts) / max(1, len(candidate_counts))),
        "candidate_count_max": int(max(candidate_counts) if candidate_counts else 0),
        "pairwise_count_mean": float(sum(pair_counts) / max(1, len(pair_counts))),
        "pairwise_count_max": int(max(pair_counts) if pair_counts else 0),
    }
