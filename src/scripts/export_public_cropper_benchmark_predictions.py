#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score staged FCDB/CPC/GNMC public benchmark candidate windows with public cropping teacher models."
    )
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=["cacnet", "cgs", "gaic"], choices=["cacnet", "cgs", "gaic", "s2cnet"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument(
        "--group_by_image",
        type=int,
        default=1,
        help="Score each source image once and fan scores back out to AR/task records. This is safe because public croppers are not AR-conditioned.",
    )
    parser.add_argument("--strict", action="store_true")
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


def load_public_cropper_eval_module() -> Any:
    path = PROJECT_ROOT / "src/scripts/evaluate_public_croppers_gaic_v2.py"
    spec = importlib.util.spec_from_file_location("evaluate_public_croppers_gaic_v2_for_public_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import public cropper evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_box(candidate: dict[str, Any]) -> list[float] | None:
    box = candidate.get("bbox_xyxy_norm", candidate.get("bbox_norm_xyxy"))
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)), max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))]


def task_key(task: dict[str, Any]) -> str:
    target_ar = task.get("target_ar")
    ar_part = "FREE" if target_ar is None or str(target_ar).strip() == "" else str(target_ar)
    return f"{str(task.get('dataset')).lower()}::{task.get('image_id')}::{ar_part}"


def resolve_manifest_image_path(path_value: Any) -> Path:
    image_path = Path(str(path_value or ""))
    if image_path.exists():
        return image_path
    marker = "/ImageCropping/"
    path_text = str(image_path)
    if marker in path_text:
        suffix = path_text.split(marker, 1)[1]
        remapped = PROJECT_ROOT / suffix
        if remapped.exists():
            return remapped
    return image_path


def load_records(task_manifest: Path, *, max_tasks: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    meta_by_record_id: dict[str, dict[str, Any]] = {}
    for task in read_jsonl(task_manifest):
        if max_tasks > 0 and len(records) >= max_tasks:
            break
        image_path = resolve_manifest_image_path(task.get("image_path", ""))
        if not image_path.exists():
            continue
        candidates: list[dict[str, Any]] = []
        for idx, candidate in enumerate(task.get("candidate_windows") or []):
            if not isinstance(candidate, dict):
                continue
            box = candidate_box(candidate)
            if box is None:
                continue
            candidates.append(
                {
                    "candidate_id": str(candidate.get("candidate_id") or f"candidate_{idx:04d}"),
                    "bbox_norm_xyxy": box,
                    "bbox_xyxy_norm": box,
                    "source": str(candidate.get("source", "")),
                    "label": str(candidate.get("label", "")),
                    "meta": dict(candidate.get("meta", {})) if isinstance(candidate.get("meta"), dict) else {},
                }
            )
        if not candidates:
            continue
        rid = task_key(task)
        records.append({"image_id": rid, "image_path": str(image_path), "candidates": candidates})
        meta_by_record_id[rid] = {
            "dataset": str(task.get("dataset", "")).lower(),
            "image_id": str(task.get("image_id", "")),
            "target_ar": task.get("target_ar"),
            "task_type": str(task.get("task_type", "")),
            "candidate_count": len(candidates),
        }
    return records, meta_by_record_id


def candidate_signature(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    box = candidate.get("bbox_norm_xyxy", candidate.get("bbox_xyxy_norm", [0.0, 0.0, 1.0, 1.0]))
    return tuple(round(float(v), 6) for v in box[:4])  # type: ignore[index]


def group_records_by_image(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []
    record_map: dict[str, dict[str, Any]] = {}
    for record in records:
        image_path = str(record.get("image_path", ""))
        if image_path not in groups:
            group_id = f"group_{len(groups):06d}"
            groups[image_path] = {
                "image_id": group_id,
                "image_path": image_path,
                "candidates": [],
                "_index_by_box": {},
            }
            group_order.append(image_path)
        group = groups[image_path]
        index_by_box = group["_index_by_box"]
        indices: list[int] = []
        for candidate in record.get("candidates") or []:
            sig = candidate_signature(candidate)
            if sig not in index_by_box:
                index_by_box[sig] = len(group["candidates"])
                group["candidates"].append(dict(candidate))
            indices.append(int(index_by_box[sig]))
        record_map[str(record["image_id"])] = {"group_id": str(group["image_id"]), "candidate_indices": indices}

    grouped_records: list[dict[str, Any]] = []
    for image_path in group_order:
        group = dict(groups[image_path])
        group.pop("_index_by_box", None)
        grouped_records.append(group)
    return grouped_records, record_map


def expand_group_scores_to_records(
    *,
    records: list[dict[str, Any]],
    group_scores_by_record: dict[str, list[float]],
    group_coverage_by_record: dict[str, dict[str, float]],
    record_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], int]:
    scores_by_record: dict[str, list[float]] = {}
    coverage_by_record: dict[str, dict[str, float]] = {}
    missing_record_count = 0
    for record in records:
        rid = str(record["image_id"])
        mapping = record_map.get(rid, {})
        group_id = str(mapping.get("group_id", ""))
        group_scores = group_scores_by_record.get(group_id)
        indices = mapping.get("candidate_indices", [])
        if not isinstance(indices, list) or group_scores is None:
            missing_record_count += 1
            continue
        scores: list[float] = []
        for idx in indices:
            try:
                scores.append(float(group_scores[int(idx)]))
            except (IndexError, TypeError, ValueError):
                pass
        if len(scores) != len(record.get("candidates") or []):
            missing_record_count += 1
            continue
        scores_by_record[rid] = scores
        group_coverage = dict(group_coverage_by_record.get(group_id, {}))
        group_coverage["candidate_count"] = float(len(scores))
        group_coverage["scored_candidate_rate"] = 1.0
        if scores:
            group_coverage["record_score_max"] = float(max(scores))
            group_coverage["record_score_mean"] = float(sum(scores) / len(scores))
        coverage_by_record[rid] = group_coverage
    return scores_by_record, coverage_by_record, missing_record_count


def prediction_rows_for_method(
    *,
    method: str,
    records: list[dict[str, Any]],
    meta_by_record_id: dict[str, dict[str, Any]],
    scores_by_record: dict[str, list[float]],
    coverage_by_record: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rid = str(record["image_id"])
        scores = scores_by_record.get(rid)
        if scores is None:
            continue
        task_meta = meta_by_record_id[rid]
        candidates: list[dict[str, Any]] = []
        raw_candidates = record.get("candidates") or []
        order = sorted(range(min(len(raw_candidates), len(scores))), key=lambda idx: float(scores[idx]), reverse=True)
        rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(order)}
        for idx, candidate in enumerate(raw_candidates[: len(scores)]):
            candidates.append(
                {
                    "candidate_id": str(candidate.get("candidate_id", f"candidate_{idx:04d}")),
                    "bbox_xyxy_norm": [float(v) for v in candidate["bbox_norm_xyxy"]],
                    "score": float(scores[idx]),
                    "source": str(candidate.get("source", "")),
                    "label": str(candidate.get("label", "")),
                    "model_rank": int(rank_by_idx.get(idx, idx + 1)),
                    "scores": {f"{method}_score": float(scores[idx])},
                }
            )
        rows.append(
            {
                "dataset": task_meta["dataset"],
                "image_id": task_meta["image_id"],
                "target_ar": task_meta["target_ar"],
                "method": f"public_cropper_{method}",
                "source": "public_cropper_candidate_ranker" if method in {"cgs", "gaic", "s2cnet"} else "public_cropper_single_crop_projection",
                "score_semantics": "native_candidate_score" if method in {"cgs", "gaic"} else ("native_candidate_score_with_heuristic_graph_nodes" if method == "s2cnet" else "iou_to_single_predicted_crop_projection"),
                "model_metadata": metadata,
                "coverage": coverage_by_record.get(rid, {}),
                "candidates": candidates,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, meta_by_record_id = load_records(args.task_manifest_jsonl.resolve(), max_tasks=int(args.max_tasks))
    if int(args.group_by_image) > 0:
        scoring_records, record_map = group_records_by_image(records)
    else:
        scoring_records = records
        record_map = {str(record["image_id"]): {"group_id": str(record["image_id"]), "candidate_indices": list(range(len(record.get("candidates") or [])))} for record in records}
    module = load_public_cropper_eval_module()
    device = torch.device(str(args.device))
    summary: dict[str, Any] = {
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "requested_methods": list(args.methods),
        "group_by_image": bool(int(args.group_by_image) > 0),
        "record_count": len(records),
        "scoring_record_count": len(scoring_records),
        "candidate_count": int(sum(len(record.get("candidates") or []) for record in records)),
        "scoring_candidate_count": int(sum(len(record.get("candidates") or []) for record in scoring_records)),
        "methods": {},
    }
    scorers = {"cacnet": module._score_cacnet, "cgs": module._score_cgs, "gaic": module._score_gaic, "s2cnet": module._score_s2cnet}
    for method in args.methods:
        method_start = time.time()
        try:
            raw_scores_by_record, raw_coverage_by_record, metadata = scorers[method](scoring_records, device=device)
            if int(args.group_by_image) > 0:
                scores_by_record, coverage_by_record, missing_record_count = expand_group_scores_to_records(
                    records=records,
                    group_scores_by_record=raw_scores_by_record,
                    group_coverage_by_record=raw_coverage_by_record,
                    record_map=record_map,
                )
            else:
                scores_by_record = raw_scores_by_record
                coverage_by_record = raw_coverage_by_record
                missing_record_count = 0
            rows = prediction_rows_for_method(
                method=method,
                records=records,
                meta_by_record_id=meta_by_record_id,
                scores_by_record=scores_by_record,
                coverage_by_record=coverage_by_record,
                metadata=metadata,
            )
            prediction_jsonl = args.output_dir / f"public_cropper_{method}_predictions.jsonl"
            written = write_jsonl(prediction_jsonl, rows)
            summary["methods"][method] = {
                "status": "ok",
                "prediction_jsonl": str(prediction_jsonl.resolve()),
                "prediction_rows": int(written),
                "scored_record_count": len(scores_by_record),
                "raw_scored_record_count": len(raw_scores_by_record),
                "missing_record_count": int(missing_record_count),
                "coverage_record_count": len(coverage_by_record),
                "metadata": metadata,
                "duration_sec": round(time.time() - method_start, 3),
            }
        except Exception as exc:
            if args.strict:
                raise
            summary["methods"][method] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_sec": round(time.time() - method_start, 3),
            }
    summary["duration_sec"] = round(time.time() - start, 3)
    summary_path = args.output_dir / "public_cropper_benchmark_predictions_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
