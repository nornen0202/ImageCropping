#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from public_benchmark.metrics import aggregate_rows, evaluate_task, load_predictions  # noqa: E402
from public_benchmark.schema import BenchmarkBox, BenchmarkTask, PairwisePreference  # noqa: E402
from universal_crop_teacher.warehouse import public_task_to_warehouse_row, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions against an explicit public benchmark task manifest.")
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--allowed_warehouse_jsonl", default="", type=Path)
    parser.add_argument("--allowed_split", default="")
    parser.add_argument("--method_name", default="")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--split_filter", default="")
    parser.add_argument("--split_seed", type=int, default=20260420)
    parser.add_argument("--force_public_hash_split", type=int, default=0)
    parser.add_argument("--mode", choices=["S", "SC"], default="S")
    parser.add_argument("--inject_gt", type=int, default=0)
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--scoring_fallback", choices=["center_area_prior", "none"], default="none")
    parser.add_argument("--target_ar_filter_mode", choices=["none", "hard"], default="hard")
    parser.add_argument("--ar_tolerance", type=float, default=0.03)
    parser.add_argument("--max_pairwise_details", type=int, default=0)
    parser.add_argument("--prediction_load_mode", choices=["auto", "preload", "stream", "sqlite"], default="auto")
    parser.add_argument("--stream_predictions", action="store_true", help="Alias for --prediction_load_mode stream.")
    parser.add_argument("--stream_auto_threshold_mb", type=float, default=128.0)
    parser.add_argument("--progress_json", default="", type=Path)
    parser.add_argument("--progress_every", type=int, default=10000)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    raw = str(path).strip()
    if raw in ("", "."):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _box(row: dict[str, Any]) -> BenchmarkBox:
    return BenchmarkBox(
        bbox_xyxy_norm=[float(v) for v in row.get("bbox_xyxy_norm", row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))[:4]],
        label=str(row.get("label", "")),
        weight=float(row.get("weight", 1.0) or 1.0),
        candidate_id=str(row.get("candidate_id", "")),
        source=str(row.get("source", "")),
        score=row.get("score"),
        meta=row.get("meta", {}) if isinstance(row.get("meta"), dict) else {},
    )


def _pair(row: dict[str, Any]) -> PairwisePreference:
    return PairwisePreference(
        bbox_a_xyxy_norm=[float(v) for v in row.get("bbox_a_xyxy_norm", [0.0, 0.0, 1.0, 1.0])[:4]],
        bbox_b_xyxy_norm=[float(v) for v in row.get("bbox_b_xyxy_norm", [0.0, 0.0, 1.0, 1.0])[:4]],
        preferred=str(row.get("preferred", "a")),
        candidate_id_a=str(row.get("candidate_id_a", "")),
        candidate_id_b=str(row.get("candidate_id_b", "")),
        weight=float(row.get("weight", 1.0) or 1.0),
        votes=row.get("votes", {}) if isinstance(row.get("votes"), dict) else {},
        meta=row.get("meta", {}) if isinstance(row.get("meta"), dict) else {},
    )


def task_from_dict(row: dict[str, Any]) -> BenchmarkTask:
    return BenchmarkTask(
        dataset=str(row.get("dataset", "")).lower(),
        split=str(row.get("split", "")),
        image_id=str(row.get("image_id", "")),
        image_path=str(row.get("image_path", "")),
        task_type=str(row.get("task_type", "")),
        target_ar=row.get("target_ar"),
        gt_boxes=[_box(gt) for gt in row.get("gt_boxes", []) if isinstance(gt, dict)],
        pairwise=[_pair(pair) for pair in row.get("pairwise", []) if isinstance(pair, dict)],
        candidate_windows=[_box(cand) for cand in row.get("candidate_windows", []) if isinstance(cand, dict)],
        meta=row.get("meta", {}) if isinstance(row.get("meta"), dict) else {},
    )


def prediction_key(row: dict[str, Any]) -> tuple[str, str, Any]:
    target_ar = row.get("target_ar")
    if target_ar == "":
        target_ar = None
    return (str(row.get("dataset", "")).lower(), str(row.get("image_id", "")), target_ar)


def prediction_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        return [cand for cand in candidates if isinstance(cand, dict)]
    if "bbox_xyxy_norm" in row or "bbox_norm_xyxy" in row:
        return [row]
    return []


def prediction_key_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    dataset, image_id, target_ar = prediction_key(row)
    return (str(dataset), str(image_id), "" if target_ar is None else str(target_ar))


def load_allowed_task_ids(path: Path, *, split: str = "") -> set[str]:
    allowed: set[str] = set()
    raw = str(path).strip()
    if raw in ("", "."):
        return allowed
    path = Path(path)
    if not path.exists() or path.is_dir():
        return allowed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(split).strip() and str(row.get("split", "")) != str(split).strip():
                continue
            if str(row.get("dataset", "")).lower() == "gaic":
                continue
            task_id = str(row.get("task_id", ""))
            if task_id:
                allowed.add(task_id)
    return allowed


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Public Task Manifest 예측 평가 보고서",
        "",
        f"- 방법: `{summary.get('method_name', '')}`",
        f"- 예측 파일: `{summary.get('predictions_jsonl', '')}`",
        f"- 평가 task 수: `{summary.get('overall', {}).get('n_tasks', 0)}`",
        f"- split filter: `{summary.get('run', {}).get('split_filter', '')}`",
        "",
        "| 데이터셋 | task | IoU@top1 | rank@1 | rank@5 | pairwise | weighted pairwise | cov@0.9 | AR 위반 | NaN score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in sorted(summary.get("datasets", {}).items()):
        lines.append(
            "| {dataset} | {tasks} | {iou:.4f} | {r1:.4f} | {r5:.4f} | {pair} | {wpair} | {cov:.4f} | {ar:.4f} | {nan:.6f} |".format(
                dataset=dataset,
                tasks=int(row.get("n_tasks", 0)),
                iou=float(row.get("iou_top1", 0.0) or 0.0),
                r1=float(row.get("gt_rank_at_1", 0.0) or 0.0),
                r5=float(row.get("gt_rank_at_5", 0.0) or 0.0),
                pair="-" if row.get("pairwise_acc") is None else f"{float(row.get('pairwise_acc')):.4f}",
                wpair="-" if row.get("weighted_pairwise_acc") is None else f"{float(row.get('weighted_pairwise_acc')):.4f}",
                cov=float(row.get("coverage_at_09", 0.0) or 0.0),
                ar=float(row.get("ar_constraint_violation_rate", 0.0) or 0.0),
                nan=float(row.get("nan_score_rate", 0.0) or 0.0),
            )
        )
    overall = summary.get("overall", {})
    lines.extend(
        [
            "",
            "## 전체 요약",
            "",
            f"- IoU@top1: `{float(overall.get('iou_top1', 0.0) or 0.0):.6f}`",
            f"- rank@1: `{float(overall.get('gt_rank_at_1', 0.0) or 0.0):.6f}`",
            f"- rank@5: `{float(overall.get('gt_rank_at_5', 0.0) or 0.0):.6f}`",
            f"- CPC weighted pairwise: `{overall.get('weighted_pairwise_acc')}`",
            f"- schema_fail_rate: `{float(overall.get('schema_fail_rate', 0.0) or 0.0):.6f}`",
            f"- nan_score_rate: `{float(overall.get('nan_score_rate', 0.0) or 0.0):.6f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def should_stream_predictions(args: argparse.Namespace) -> bool:
    if bool(args.stream_predictions):
        return True
    if str(args.prediction_load_mode) == "stream":
        return True
    if str(args.prediction_load_mode) == "preload":
        return False
    try:
        size_mb = float(args.predictions_jsonl.stat().st_size) / (1024.0 * 1024.0)
    except OSError:
        size_mb = 0.0
    return size_mb >= float(args.stream_auto_threshold_mb)


def iter_filtered_tasks(args: argparse.Namespace, allowed_task_ids: set[str]) -> Iterable[BenchmarkTask]:
    task_count = 0
    for raw_task in read_jsonl(args.task_manifest_jsonl.resolve()):
        if int(args.max_tasks) > 0 and task_count >= int(args.max_tasks):
            break
        warehouse_row = public_task_to_warehouse_row(
            raw_task,
            split_seed=int(args.split_seed),
            force_hash_split=bool(int(args.force_public_hash_split) > 0),
        )
        task_id = str(warehouse_row.get("task_id", ""))
        if allowed_task_ids and task_id not in allowed_task_ids:
            continue
        split = warehouse_row.get("split", "")
        if str(args.split_filter).strip() and str(split) != str(args.split_filter).strip():
            continue
        task_count += 1
        yield task_from_dict(raw_task)


def evaluate_with_preloaded_predictions(
    args: argparse.Namespace,
    allowed_task_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    predictions = load_predictions(args.predictions_jsonl.resolve())
    all_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    progress_every = max(1, int(args.progress_every))
    started_at = time.time()
    for idx, task in enumerate(iter_filtered_tasks(args, allowed_task_ids), start=1):
        row = evaluate_task(
            task,
            predictions=predictions.get(task.key(), []),
            mode=str(args.mode),
            inject_gt=bool(int(args.inject_gt) > 0),
            score_field=str(args.score_field),
            scoring_fallback=str(args.scoring_fallback),
            ar_tolerance=float(args.ar_tolerance),
            target_ar_filter_mode=str(args.target_ar_filter_mode),
            collect_pairwise_details=int(args.max_pairwise_details) > 0,
            max_pairwise_details=int(args.max_pairwise_details),
        )
        all_rows.append(row)
        by_dataset[task.dataset].append(row)
        if idx % progress_every == 0:
            write_progress(
                args.progress_json,
                {
                    "state": "running",
                    "phase": "public_eval_preload",
                    "processed_tasks": idx,
                    "prediction_task_count": len(predictions),
                    "elapsed_sec": round(time.time() - started_at, 3),
                },
            )
    stats = {
        "prediction_load_mode_effective": "preload",
        "preloaded_prediction_task_count": len(predictions),
    }
    return all_rows, by_dataset, stats


def evaluate_with_streamed_predictions(
    args: argparse.Namespace,
    allowed_task_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    tasks_by_key: dict[tuple[str, str, Any], list[BenchmarkTask]] = defaultdict(list)
    for task in iter_filtered_tasks(args, allowed_task_ids):
        tasks_by_key[task.key()].append(task)

    all_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, str, Any]] = set()
    prediction_rows_seen = 0
    prediction_rows_matched = 0
    duplicate_prediction_rows = 0
    progress_every = max(1, int(args.progress_every))
    started_at = time.time()

    with args.predictions_jsonl.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prediction_rows_seen += 1
            pred_row = json.loads(line)
            key = prediction_key(pred_row)
            tasks = tasks_by_key.get(key, [])
            if prediction_rows_seen % progress_every == 0:
                write_progress(
                    args.progress_json,
                    {
                        "state": "running",
                        "phase": "public_eval_stream",
                        "manifest_task_key_count": len(tasks_by_key),
                        "prediction_rows_seen": prediction_rows_seen,
                        "prediction_rows_matched": prediction_rows_matched,
                        "duplicate_prediction_rows_ignored": duplicate_prediction_rows,
                        "evaluated_rows": len(all_rows),
                        "elapsed_sec": round(time.time() - started_at, 3),
                    },
                )
            if not tasks:
                continue
            if key in seen_keys:
                duplicate_prediction_rows += 1
                continue
            seen_keys.add(key)
            prediction_rows_matched += 1
            candidates = prediction_candidates(pred_row)
            for task in tasks:
                row = evaluate_task(
                    task,
                    predictions=candidates,
                    mode=str(args.mode),
                    inject_gt=bool(int(args.inject_gt) > 0),
                    score_field=str(args.score_field),
                    scoring_fallback=str(args.scoring_fallback),
                    ar_tolerance=float(args.ar_tolerance),
                    target_ar_filter_mode=str(args.target_ar_filter_mode),
                    collect_pairwise_details=int(args.max_pairwise_details) > 0,
                    max_pairwise_details=int(args.max_pairwise_details),
                )
                all_rows.append(row)
                by_dataset[task.dataset].append(row)

    missing_prediction_task_count = 0
    for key, tasks in sorted(tasks_by_key.items()):
        if key in seen_keys:
            continue
        missing_prediction_task_count += len(tasks)
        for task in tasks:
            row = evaluate_task(
                task,
                predictions=[],
                mode=str(args.mode),
                inject_gt=bool(int(args.inject_gt) > 0),
                score_field=str(args.score_field),
                scoring_fallback=str(args.scoring_fallback),
                ar_tolerance=float(args.ar_tolerance),
                target_ar_filter_mode=str(args.target_ar_filter_mode),
                collect_pairwise_details=int(args.max_pairwise_details) > 0,
                max_pairwise_details=int(args.max_pairwise_details),
            )
            all_rows.append(row)
            by_dataset[task.dataset].append(row)
        if missing_prediction_task_count and missing_prediction_task_count % progress_every == 0:
            write_progress(
                args.progress_json,
                {
                    "state": "running",
                    "phase": "public_eval_missing_predictions",
                    "manifest_task_key_count": len(tasks_by_key),
                    "prediction_rows_seen": prediction_rows_seen,
                    "prediction_rows_matched": prediction_rows_matched,
                    "missing_prediction_task_count": missing_prediction_task_count,
                    "evaluated_rows": len(all_rows),
                    "elapsed_sec": round(time.time() - started_at, 3),
                },
            )

    stats = {
        "prediction_load_mode_effective": "stream",
        "manifest_task_key_count": len(tasks_by_key),
        "prediction_rows_seen": prediction_rows_seen,
        "prediction_rows_matched": prediction_rows_matched,
        "duplicate_prediction_rows_ignored": duplicate_prediction_rows,
        "missing_prediction_task_count": missing_prediction_task_count,
    }
    return all_rows, by_dataset, stats


def build_sqlite_prediction_index(
    *,
    predictions_jsonl: Path,
    sqlite_path: Path,
    progress_json: Path | None,
    progress_every: int,
) -> dict[str, Any]:
    if sqlite_path.exists():
        sqlite_path.unlink()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    rows_seen = 0
    rows_indexed = 0
    duplicate_rows = 0
    progress_every = max(1, int(progress_every))
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            """
            CREATE TABLE predictions (
                dataset TEXT NOT NULL,
                image_id TEXT NOT NULL,
                target_ar TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                PRIMARY KEY (dataset, image_id, target_ar)
            )
            """
        )
        with predictions_jsonl.resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_seen += 1
                pred_row = json.loads(line)
                dataset, image_id, target_ar = prediction_key_parts(pred_row)
                candidates = prediction_candidates(pred_row)
                cur = conn.execute(
                    "INSERT OR IGNORE INTO predictions(dataset, image_id, target_ar, candidates_json) VALUES (?, ?, ?, ?)",
                    (dataset, image_id, target_ar, json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))),
                )
                if int(cur.rowcount or 0) > 0:
                    rows_indexed += 1
                else:
                    duplicate_rows += 1
                if rows_seen % progress_every == 0:
                    conn.commit()
                    write_progress(
                        progress_json,
                        {
                            "state": "running",
                            "phase": "public_eval_sqlite_index",
                            "prediction_rows_seen": rows_seen,
                            "prediction_rows_indexed": rows_indexed,
                            "duplicate_prediction_rows_ignored": duplicate_rows,
                            "sqlite_path": str(sqlite_path),
                            "elapsed_sec": round(time.time() - started_at, 3),
                        },
                    )
        conn.commit()
    finally:
        conn.close()
    return {
        "prediction_rows_seen": rows_seen,
        "prediction_rows_indexed": rows_indexed,
        "duplicate_prediction_rows_ignored": duplicate_rows,
        "sqlite_path": str(sqlite_path),
    }


def evaluate_with_sqlite_predictions(
    args: argparse.Namespace,
    allowed_task_ids: set[str],
    *,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    sqlite_path = output_dir / "prediction_index.sqlite"
    index_stats = build_sqlite_prediction_index(
        predictions_jsonl=args.predictions_jsonl,
        sqlite_path=sqlite_path,
        progress_json=args.progress_json,
        progress_every=int(args.progress_every),
    )
    all_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_rows_matched = 0
    missing_prediction_task_count = 0
    progress_every = max(1, int(args.progress_every))
    started_at = time.time()
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for idx, task in enumerate(iter_filtered_tasks(args, allowed_task_ids), start=1):
            dataset, image_id, target_ar = prediction_key_parts(
                {"dataset": task.dataset, "image_id": task.image_id, "target_ar": task.target_ar}
            )
            cur = conn.execute(
                "SELECT candidates_json FROM predictions WHERE dataset = ? AND image_id = ? AND target_ar = ?",
                (dataset, image_id, target_ar),
            )
            fetched = cur.fetchone()
            if fetched is None:
                candidates: list[dict[str, Any]] = []
                missing_prediction_task_count += 1
            else:
                prediction_rows_matched += 1
                loaded = json.loads(str(fetched[0]))
                candidates = loaded if isinstance(loaded, list) else []
            row = evaluate_task(
                task,
                predictions=candidates,
                mode=str(args.mode),
                inject_gt=bool(int(args.inject_gt) > 0),
                score_field=str(args.score_field),
                scoring_fallback=str(args.scoring_fallback),
                ar_tolerance=float(args.ar_tolerance),
                target_ar_filter_mode=str(args.target_ar_filter_mode),
                collect_pairwise_details=int(args.max_pairwise_details) > 0,
                max_pairwise_details=int(args.max_pairwise_details),
            )
            all_rows.append(row)
            by_dataset[task.dataset].append(row)
            if idx % progress_every == 0:
                write_progress(
                    args.progress_json,
                    {
                        "state": "running",
                        "phase": "public_eval_sqlite",
                        "processed_tasks": idx,
                        "prediction_rows_matched": prediction_rows_matched,
                        "missing_prediction_task_count": missing_prediction_task_count,
                        "elapsed_sec": round(time.time() - started_at, 3),
                    },
                )
    finally:
        conn.close()
    stats = {
        "prediction_load_mode_effective": "sqlite",
        "prediction_rows_matched": prediction_rows_matched,
        "missing_prediction_task_count": missing_prediction_task_count,
        **index_stats,
    }
    return all_rows, by_dataset, stats


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_json = args.progress_json if str(args.progress_json).strip() not in ("", ".") else None
    args.progress_json = progress_json
    write_progress(
        progress_json,
        {
            "state": "running",
            "phase": "public_eval_start",
            "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
            "predictions_jsonl": str(args.predictions_jsonl.resolve()),
            "output_dir": str(output_dir),
            "prediction_load_mode_requested": "stream" if bool(args.stream_predictions) else str(args.prediction_load_mode),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        },
    )
    allowed_task_ids = load_allowed_task_ids(args.allowed_warehouse_jsonl, split=str(args.allowed_split))
    if str(args.prediction_load_mode) == "sqlite":
        all_rows, by_dataset, prediction_stats = evaluate_with_sqlite_predictions(args, allowed_task_ids, output_dir=output_dir)
    elif should_stream_predictions(args):
        all_rows, by_dataset, prediction_stats = evaluate_with_streamed_predictions(args, allowed_task_ids)
    else:
        all_rows, by_dataset, prediction_stats = evaluate_with_preloaded_predictions(args, allowed_task_ids)

    summary = {
        "format": "public_task_manifest_prediction_eval_v1",
        "method_name": str(args.method_name),
        "predictions_jsonl": str(args.predictions_jsonl.resolve()),
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "run": {
            "mode": str(args.mode),
            "inject_gt": bool(int(args.inject_gt) > 0),
            "score_field": str(args.score_field),
            "scoring_fallback": str(args.scoring_fallback),
            "target_ar_filter_mode": str(args.target_ar_filter_mode),
            "split_filter": str(args.split_filter),
            "allowed_warehouse_jsonl": str(args.allowed_warehouse_jsonl.resolve())
            if str(args.allowed_warehouse_jsonl).strip() not in ("", ".") and args.allowed_warehouse_jsonl.exists() and not args.allowed_warehouse_jsonl.is_dir()
            else "",
            "allowed_split": str(args.allowed_split),
            "allowed_task_count": len(allowed_task_ids),
            "duration_sec": round(time.time() - start, 3),
            "prediction_load_mode_requested": "stream" if bool(args.stream_predictions) else str(args.prediction_load_mode),
            "stream_auto_threshold_mb": float(args.stream_auto_threshold_mb),
            **prediction_stats,
        },
        "overall": aggregate_rows(all_rows),
        "datasets": {dataset: aggregate_rows(rows) for dataset, rows in sorted(by_dataset.items())},
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "per_task_jsonl": str(output_dir / "per_task_metrics.jsonl"),
            "report_md": str(output_dir / "REPORT.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_task_metrics.jsonl", all_rows)
    (output_dir / "REPORT.md").write_text(report_markdown(summary), encoding="utf-8")
    write_progress(
        progress_json,
        {
            "state": "completed",
            "phase": "public_eval_complete",
            "summary_json": str(output_dir / "summary.json"),
            "per_task_jsonl": str(output_dir / "per_task_metrics.jsonl"),
            "report_md": str(output_dir / "REPORT.md"),
            "duration_sec": summary["run"]["duration_sec"],
            **prediction_stats,
        },
    )
    print(json.dumps(summary["artifacts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
