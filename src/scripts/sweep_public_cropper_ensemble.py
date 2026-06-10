#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = SRC_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_public_task_manifest_predictions import load_allowed_task_ids, task_from_dict  # noqa: E402
from mobilecropnet_v4.gaic_benchmark import evaluate_scored_records, load_gaic_annotation_records  # noqa: E402
from public_benchmark.ensemble import (  # noqa: E402
    candidate_signature,
    fuse_prediction_groups,
    load_prediction_groups,
    write_grouped_predictions,
)
from public_benchmark.metrics import aggregate_rows, evaluate_task  # noqa: E402
from universal_crop_teacher.warehouse import public_task_to_warehouse_row, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep weighted public cropper ensembles on public holdout + GAIC official benchmark.")
    parser.add_argument("--method_names", nargs="+", required=True)
    parser.add_argument("--public_prediction_jsonls", nargs="+", required=True, type=Path)
    parser.add_argument("--gaic_prediction_jsonls", nargs="+", required=True, type=Path)
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--allowed_warehouse_jsonl", required=True, type=Path)
    parser.add_argument("--allowed_split", default="test")
    parser.add_argument("--split_seed", type=int, default=20260420)
    parser.add_argument("--force_public_hash_split", type=int, default=1)
    parser.add_argument("--normalizations", nargs="+", default=["rank_pct", "zscore"])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.2, 0.35, 0.5, 0.65, 0.8])
    parser.add_argument(
        "--gaic_annotations_json",
        type=Path,
        default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json/instances_test.json",
    )
    parser.add_argument(
        "--gaic_image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
        ],
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _public_summary(
    predictions: dict[tuple[str, str, str | None], list[dict[str, Any]]],
    *,
    task_manifest_jsonl: Path,
    allowed_warehouse_jsonl: Path,
    allowed_split: str,
    split_seed: int,
    force_public_hash_split: bool,
) -> dict[str, Any]:
    allowed_task_ids = load_allowed_task_ids(allowed_warehouse_jsonl.resolve(), split=str(allowed_split))
    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_task in read_jsonl(task_manifest_jsonl.resolve()):
        warehouse_row = public_task_to_warehouse_row(
            raw_task,
            split_seed=int(split_seed),
            force_hash_split=bool(force_public_hash_split),
        )
        task_id = str(warehouse_row.get("task_id", ""))
        if allowed_task_ids and task_id not in allowed_task_ids:
            continue
        task = task_from_dict(raw_task)
        task_predictions = predictions.get(task.key(), [])
        row = evaluate_task(
            task,
            predictions=task_predictions,
            mode="S",
            inject_gt=False,
            score_field="score",
            scoring_fallback="none",
            ar_tolerance=0.03,
            target_ar_filter_mode="hard",
        )
        rows.append(row)
        by_dataset[task.dataset].append(row)
    return {
        "overall": aggregate_rows(rows),
        "datasets": {dataset: aggregate_rows(dataset_rows) for dataset, dataset_rows in sorted(by_dataset.items())},
        "task_count": len(rows),
    }


def _gaic_summary(
    predictions: dict[tuple[str, str, str | None], list[dict[str, Any]]],
    *,
    records: list[dict[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    scores_by_image: dict[str, list[float]] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        key = ("gaic", image_id, "FREE")
        candidates = predictions.get(key, [])
        score_by_signature = {
            candidate_signature(candidate): float(candidate.get("score"))
            for candidate in candidates
            if candidate.get("score") is not None
        }
        scores_by_image[image_id] = [
            float(score_by_signature.get(candidate_signature(candidate), -1e9))
            for candidate in list(record.get("candidates") or [])
        ]
    result = evaluate_scored_records(records, scores_by_image, method_name=method_name)
    return {
        "image_count": int(result.get("image_count", 0)),
        "missing_image_count": int(result.get("missing_image_count", 0)),
        "metrics": result.get("metrics", {}),
    }


def _public_macro(summary: dict[str, Any]) -> float:
    datasets = summary.get("datasets", {}) if isinstance(summary.get("datasets"), dict) else {}
    values: list[float] = []
    fcdb = datasets.get("fcdb", {}) if isinstance(datasets.get("fcdb"), dict) else {}
    cpc = datasets.get("cpc", {}) if isinstance(datasets.get("cpc"), dict) else {}
    gnmc = datasets.get("gnmc", {}) if isinstance(datasets.get("gnmc"), dict) else {}
    if fcdb:
        values.append(float(fcdb.get("iou_top1", 0.0) or 0.0))
    if cpc:
        values.append(float(cpc.get("weighted_pairwise_acc", 0.0) or 0.0))
    if gnmc:
        values.append(float(gnmc.get("iou_top1", 0.0) or 0.0))
    return float(sum(values) / max(1, len(values)))


def _gaic_component(summary: dict[str, Any]) -> float:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    return float(
        0.50 * (float(metrics.get("top1_mos", 0.0) or 0.0) / 5.0)
        + 0.35 * float(metrics.get("srcc", 0.0) or 0.0)
        + 0.15 * float(metrics.get("accw4_of_top10", 0.0) or 0.0)
    )


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Public Cropper Ensemble Sweep",
        "",
        f"- methods: `{', '.join(summary.get('method_names', []))}`",
        f"- configs: `{len(summary.get('leaderboard', []))}`",
        f"- best_config: `{summary.get('best_config', {}).get('config_name', '')}`",
        "",
        "| config | public_macro | GAIC component | combined | FCDB IoU | CPC weighted | GNMC IoU | GAIC SRCC | Acc1@10 | Accw4@10 | top1 MOS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("leaderboard", []):
        public_row = row.get("public_summary", {}) if isinstance(row.get("public_summary"), dict) else {}
        gaic_row = row.get("gaic_summary", {}) if isinstance(row.get("gaic_summary"), dict) else {}
        public_datasets = public_row.get("datasets", {}) if isinstance(public_row.get("datasets"), dict) else {}
        gaic_metrics = gaic_row.get("metrics", {}) if isinstance(gaic_row.get("metrics"), dict) else {}
        lines.append(
            "| {config} | {public_macro:.4f} | {gaic_component:.4f} | {combined:.4f} | {fcdb:.4f} | {cpc:.4f} | {gnmc:.4f} | {srcc:.4f} | {acc1:.3f} | {accw4:.4f} | {top1:.4f} |".format(
                config=row.get("config_name", ""),
                public_macro=float(row.get("public_macro", 0.0) or 0.0),
                gaic_component=float(row.get("gaic_component", 0.0) or 0.0),
                combined=float(row.get("combined", 0.0) or 0.0),
                fcdb=float(((public_datasets.get("fcdb", {}) if isinstance(public_datasets.get("fcdb"), dict) else {}).get("iou_top1", 0.0) or 0.0)),
                cpc=float(((public_datasets.get("cpc", {}) if isinstance(public_datasets.get("cpc"), dict) else {}).get("weighted_pairwise_acc", 0.0) or 0.0)),
                gnmc=float(((public_datasets.get("gnmc", {}) if isinstance(public_datasets.get("gnmc"), dict) else {}).get("iou_top1", 0.0) or 0.0)),
                srcc=float(gaic_metrics.get("srcc", 0.0) or 0.0),
                acc1=float(gaic_metrics.get("acc1_of_top10", 0.0) or 0.0),
                accw4=float(gaic_metrics.get("accw4_of_top10", 0.0) or 0.0),
                top1=float(gaic_metrics.get("top1_mos", 0.0) or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if len(args.method_names) != len(args.public_prediction_jsonls) or len(args.method_names) != len(args.gaic_prediction_jsonls):
        raise ValueError("method_names, public_prediction_jsonls, gaic_prediction_jsonls lengths must match")
    start = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    public_groups_by_method = {
        str(name): load_prediction_groups(path.resolve())
        for name, path in zip(args.method_names, args.public_prediction_jsonls)
    }
    gaic_groups_by_method = {
        str(name): load_prediction_groups(path.resolve())
        for name, path in zip(args.method_names, args.gaic_prediction_jsonls)
    }
    gaic_records = load_gaic_annotation_records(
        args.gaic_annotations_json.resolve(),
        image_roots=[Path(path).resolve() for path in args.gaic_image_roots],
    )

    leaderboard: list[dict[str, Any]] = []
    best_public_groups: dict[tuple[str, str, str | None], list[dict[str, Any]]] | None = None
    best_gaic_groups: dict[tuple[str, str, str | None], list[dict[str, Any]]] | None = None
    best_combined = float("-inf")
    for normalization in args.normalizations:
        for first_weight in args.weights:
            weight_a = float(first_weight)
            weight_b = float(1.0 - weight_a)
            weights = {
                str(args.method_names[0]): weight_a,
                str(args.method_names[1]): weight_b,
            }
            config_name = f"{normalization}_{args.method_names[0]}{int(round(weight_a * 100)):02d}_{args.method_names[1]}{int(round(weight_b * 100)):02d}"
            fused_public, public_fuse_summary = fuse_prediction_groups(
                public_groups_by_method,
                method_weights=weights,
                normalization=str(normalization),
            )
            fused_gaic, gaic_fuse_summary = fuse_prediction_groups(
                gaic_groups_by_method,
                method_weights=weights,
                normalization=str(normalization),
            )
            public_summary = _public_summary(
                fused_public,
                task_manifest_jsonl=args.task_manifest_jsonl,
                allowed_warehouse_jsonl=args.allowed_warehouse_jsonl,
                allowed_split=str(args.allowed_split),
                split_seed=int(args.split_seed),
                force_public_hash_split=bool(int(args.force_public_hash_split) > 0),
            )
            gaic_summary = _gaic_summary(fused_gaic, records=gaic_records, method_name=config_name)
            public_macro = _public_macro(public_summary)
            gaic_component = _gaic_component(gaic_summary)
            combined = float(public_macro + gaic_component)
            if combined > best_combined:
                best_combined = combined
                best_public_groups = fused_public
                best_gaic_groups = fused_gaic
            leaderboard.append(
                {
                    "config_name": config_name,
                    "normalization": str(normalization),
                    "method_weights": weights,
                    "public_macro": public_macro,
                    "gaic_component": gaic_component,
                    "combined": combined,
                    "public_summary": public_summary,
                    "gaic_summary": gaic_summary,
                    "public_fuse_summary": public_fuse_summary,
                    "gaic_fuse_summary": gaic_fuse_summary,
                }
            )

    leaderboard.sort(key=lambda row: float(row.get("combined", 0.0) or 0.0), reverse=True)
    best = leaderboard[0]
    best_public_path = output_dir / "best_public_predictions.jsonl"
    best_gaic_path = output_dir / "best_gaic_predictions.jsonl"
    write_grouped_predictions(
        best_public_path,
        best_public_groups or {},
        method_name=str(best.get("config_name", "")),
        source="public_cropper_ensemble",
    )
    write_grouped_predictions(
        best_gaic_path,
        best_gaic_groups or {},
        method_name=str(best.get("config_name", "")),
        source="public_cropper_ensemble",
    )
    summary = {
        "format": "public_cropper_ensemble_sweep_v1",
        "method_names": [str(name) for name in args.method_names],
        "public_prediction_jsonls": [str(path.resolve()) for path in args.public_prediction_jsonls],
        "gaic_prediction_jsonls": [str(path.resolve()) for path in args.gaic_prediction_jsonls],
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "allowed_warehouse_jsonl": str(args.allowed_warehouse_jsonl.resolve()),
        "allowed_split": str(args.allowed_split),
        "gaic_annotations_json": str(args.gaic_annotations_json.resolve()),
        "gaic_image_roots": [str(Path(path).resolve()) for path in args.gaic_image_roots],
        "leaderboard": leaderboard,
        "best_config": leaderboard[0],
        "artifacts": {
            "best_public_predictions_jsonl": str(best_public_path),
            "best_gaic_predictions_jsonl": str(best_gaic_path),
            "summary_json": str(output_dir / "summary.json"),
            "report_md": str(output_dir / "REPORT.md"),
        },
        "duration_sec": round(time.time() - start, 3),
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["artifacts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
