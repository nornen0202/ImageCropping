#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from public_benchmark.metrics import center_area_prior_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export explicit static-baseline prediction JSONL for staged public benchmark tasks. "
            "The exporter intentionally does not copy any original candidate annotation score."
        )
    )
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", default="", type=Path)
    parser.add_argument("--method_name", default="center_area_prior")
    parser.add_argument("--max_tasks", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def clamp_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [float(x1), float(y1), float(x2), float(y2)]


def candidate_box(candidate: dict[str, Any]) -> list[float] | None:
    box = candidate.get("bbox_xyxy_norm", candidate.get("bbox_norm_xyxy"))
    if not valid_box(box):
        return None
    return clamp_box([float(v) for v in box])


def task_image_aspect(task: dict[str, Any]) -> float:
    meta = task.get("meta", {}) if isinstance(task.get("meta"), dict) else {}
    try:
        width = float(meta.get("image_width", 1.0))
        height = float(meta.get("image_height", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(1e-6, width / max(1e-6, height))


def score_task(task: dict[str, Any], *, method_name: str) -> tuple[dict[str, Any], int, int]:
    image_aspect = task_image_aspect(task)
    candidates: list[dict[str, Any]] = []
    original_non_null_scores = 0
    invalid_candidates = 0
    for idx, candidate in enumerate(task.get("candidate_windows") or []):
        if not isinstance(candidate, dict):
            invalid_candidates += 1
            continue
        box = candidate_box(candidate)
        if box is None:
            invalid_candidates += 1
            continue
        if candidate.get("score") is not None:
            original_non_null_scores += 1
        score = center_area_prior_score(box, target_ar=task.get("target_ar"), image_aspect=image_aspect)
        candidates.append(
            {
                "candidate_id": str(candidate.get("candidate_id") or f"candidate_{idx:04d}"),
                "bbox_xyxy_norm": box,
                "score": float(score),
                "source": str(candidate.get("source", "")),
                "label": str(candidate.get("label", "")),
                "scores": {"center_area_prior": float(score)},
            }
        )

    order = sorted(range(len(candidates)), key=lambda cidx: float(candidates[cidx]["score"]), reverse=True)
    for rank, cidx in enumerate(order, start=1):
        candidates[cidx]["model_rank"] = int(rank)

    row = {
        "dataset": str(task.get("dataset", "")).lower(),
        "image_id": str(task.get("image_id", "")),
        "target_ar": task.get("target_ar"),
        "method": str(method_name),
        "source": "static_public_benchmark_baseline",
        "candidates": candidates,
    }
    return row, original_non_null_scores, invalid_candidates


def main() -> None:
    args = parse_args()
    start = time.time()
    rows: list[dict[str, Any]] = []
    task_count = 0
    candidate_count = 0
    invalid_candidate_count = 0
    original_non_null_score_count = 0
    for task in read_jsonl(args.task_manifest_jsonl.resolve()):
        if int(args.max_tasks) > 0 and task_count >= int(args.max_tasks):
            break
        row, original_non_null_scores, invalid_candidates = score_task(task, method_name=str(args.method_name))
        rows.append(row)
        task_count += 1
        candidate_count += len(row["candidates"])
        invalid_candidate_count += invalid_candidates
        original_non_null_score_count += original_non_null_scores

    written = write_jsonl(args.output_jsonl.resolve(), rows)
    summary = {
        "method_name": str(args.method_name),
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "prediction_rows": int(written),
        "task_count": int(task_count),
        "candidate_count": int(candidate_count),
        "invalid_candidate_count": int(invalid_candidate_count),
        "original_candidate_non_null_score_count": int(original_non_null_score_count),
        "original_candidate_scores_copied": False,
        "duration_sec": round(time.time() - start, 3),
    }
    if str(args.summary_json).strip():
        args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
