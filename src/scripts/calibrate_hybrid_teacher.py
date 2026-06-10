#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = SRC_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_public_task_manifest_predictions import task_from_dict  # noqa: E402
from mobilecropnet_v4.gaic_benchmark import evaluate_scored_records, load_gaic_annotation_records  # noqa: E402
from public_benchmark.ensemble import candidate_signature, load_prediction_groups, write_grouped_predictions  # noqa: E402
from public_benchmark.hybrid_teacher import HybridTeacherConfig, fuse_hybrid_prediction_groups  # noqa: E402
from public_benchmark.metrics import aggregate_rows, evaluate_task  # noqa: E402
from universal_crop_teacher.warehouse import public_task_to_warehouse_row, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a production hybrid teacher from UCTR stage3 + public cropper ensemble.")
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--warehouse_jsonl", required=True, type=Path)
    parser.add_argument("--uctr_public_val_jsonl", required=True, type=Path)
    parser.add_argument("--uctr_public_test_jsonl", required=True, type=Path)
    parser.add_argument("--public_method_names", nargs="+", required=True)
    parser.add_argument("--public_prediction_jsonls", nargs="+", required=True, type=Path)
    parser.add_argument("--uctr_gaic_val_jsonl", required=True, type=Path)
    parser.add_argument("--uctr_gaic_test_jsonl", required=True, type=Path)
    parser.add_argument("--gaic_method_names", nargs="+", required=True)
    parser.add_argument("--gaic_val_prediction_jsonls", nargs="+", required=True, type=Path)
    parser.add_argument("--gaic_test_prediction_jsonls", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--gaic_val_annotations_json",
        type=Path,
        default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json/instances_val.json",
    )
    parser.add_argument(
        "--gaic_test_annotations_json",
        type=Path,
        default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json/instances_test.json",
    )
    parser.add_argument(
        "--gaic_val_image_roots",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "data/Publics/GAIC_v2/images/val"],
    )
    parser.add_argument(
        "--gaic_test_image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
        ],
    )
    parser.add_argument("--split_seed", type=int, default=20260420)
    parser.add_argument("--force_public_hash_split", type=int, default=1)
    parser.add_argument("--public_normalizations", nargs="+", default=["rank_pct", "zscore"])
    parser.add_argument("--public_weights", nargs="+", type=float, default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    parser.add_argument("--hybrid_normalizations", nargs="+", default=["rank_pct", "zscore"])
    parser.add_argument("--uctr_weights", nargs="+", type=float, default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    parser.add_argument("--uncertainty_gammas", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _public_keys_by_split(
    task_manifest_jsonl: Path,
    *,
    split_seed: int,
    force_public_hash_split: bool,
) -> dict[str, set[tuple[str, str, str | None]]]:
    out: dict[str, set[tuple[str, str, str | None]]] = defaultdict(set)
    for raw_task in read_jsonl(task_manifest_jsonl.resolve()):
        warehouse_row = public_task_to_warehouse_row(
            raw_task,
            split_seed=int(split_seed),
            force_hash_split=bool(force_public_hash_split),
        )
        if str(warehouse_row.get("dataset", "")).lower() == "gaic":
            continue
        split = str(warehouse_row.get("split", "")).strip()
        task = task_from_dict(raw_task)
        out[split].add(task.key())
    return {key: set(value) for key, value in out.items()}


def _filter_groups(
    groups: dict[tuple[str, str, str | None], list[dict[str, Any]]],
    allowed_keys: set[tuple[str, str, str | None]],
) -> dict[tuple[str, str, str | None], list[dict[str, Any]]]:
    return {key: value for key, value in groups.items() if key in allowed_keys}


def _load_public_tasks_by_split(
    task_manifest_jsonl: Path,
    *,
    split_seed: int,
    force_public_hash_split: bool,
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for raw_task in read_jsonl(task_manifest_jsonl.resolve()):
        warehouse_row = public_task_to_warehouse_row(
            raw_task,
            split_seed=int(split_seed),
            force_hash_split=bool(force_public_hash_split),
        )
        if str(warehouse_row.get("dataset", "")).lower() == "gaic":
            continue
        split = str(warehouse_row.get("split", "")).strip()
        out[split].append(task_from_dict(raw_task))
    return {key: list(value) for key, value in out.items()}


def _public_summary(
    predictions: dict[tuple[str, str, str | None], list[dict[str, Any]]],
    *,
    tasks_by_split: dict[str, list[Any]],
    split: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks_by_split.get(str(split), []):
        row = evaluate_task(
            task,
            predictions=predictions.get(task.key(), []),
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


def _iter_configs(args: argparse.Namespace) -> Iterable[HybridTeacherConfig]:
    for public_normalization in args.public_normalizations:
        for public_gaic_weight in args.public_weights:
            for hybrid_normalization in args.hybrid_normalizations:
                for uctr_weight in args.uctr_weights:
                    for uncertainty_gamma in args.uncertainty_gammas:
                        yield HybridTeacherConfig(
                            public_normalization=str(public_normalization),
                            public_gaic_weight=float(public_gaic_weight),
                            hybrid_normalization=str(hybrid_normalization),
                            uctr_weight=float(uctr_weight),
                            uctr_uncertainty_gamma=float(uncertainty_gamma),
                        )


def _config_name(config: HybridTeacherConfig) -> str:
    return (
        f"pub{config.public_normalization}_pg{int(round(config.public_gaic_weight * 100)):03d}_"
        f"hy{config.hybrid_normalization}_uw{int(round(config.uctr_weight * 100)):03d}_"
        f"ug{str(config.uctr_uncertainty_gamma).replace('.', 'p')}"
    )


def _leaderboard_row_public(
    config: HybridTeacherConfig,
    summary: dict[str, Any],
    *,
    fuse_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": config.to_dict(),
        "config_name": _config_name(config),
        "public_macro": _public_macro(summary),
        "public_summary": summary,
        "fuse_summary": _compact_fuse_summary(fuse_summary),
    }


def _leaderboard_row_gaic(
    config: HybridTeacherConfig,
    summary: dict[str, Any],
    *,
    fuse_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": config.to_dict(),
        "config_name": _config_name(config),
        "gaic_component": _gaic_component(summary),
        "gaic_summary": summary,
        "fuse_summary": _compact_fuse_summary(fuse_summary),
    }


def _compact_fuse_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "group_count": int(summary.get("group_count", 0) or 0),
        "missing_public_group_count": int(summary.get("missing_public_group_count", 0) or 0),
        "missing_uctr_group_count": int(summary.get("missing_uctr_group_count", 0) or 0),
        "config": dict(summary.get("config", {})) if isinstance(summary.get("config"), dict) else {},
    }


def _short_public_table(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[: max(1, int(limit))]:
        datasets = row.get("public_summary", {}).get("datasets", {})
        out.append(
            {
                "config_name": row.get("config_name", ""),
                "public_macro": float(row.get("public_macro", 0.0) or 0.0),
                "fcdb_iou_top1": float(((datasets.get("fcdb", {}) if isinstance(datasets.get("fcdb"), dict) else {}).get("iou_top1", 0.0) or 0.0)),
                "cpc_weighted_pairwise": float(((datasets.get("cpc", {}) if isinstance(datasets.get("cpc"), dict) else {}).get("weighted_pairwise_acc", 0.0) or 0.0)),
                "gnmc_iou_top1": float(((datasets.get("gnmc", {}) if isinstance(datasets.get("gnmc"), dict) else {}).get("iou_top1", 0.0) or 0.0)),
            }
        )
    return out


def _short_gaic_table(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[: max(1, int(limit))]:
        metrics = row.get("gaic_summary", {}).get("metrics", {})
        out.append(
            {
                "config_name": row.get("config_name", ""),
                "gaic_component": float(row.get("gaic_component", 0.0) or 0.0),
                "top1_mos": float(metrics.get("top1_mos", 0.0) or 0.0),
                "srcc": float(metrics.get("srcc", 0.0) or 0.0),
                "acc1_of_top10": float(metrics.get("acc1_of_top10", 0.0) or 0.0),
                "accw4_of_top10": float(metrics.get("accw4_of_top10", 0.0) or 0.0),
            }
        )
    return out


def _select_config_row(rows: list[dict[str, Any]], config: HybridTeacherConfig) -> dict[str, Any]:
    target = _config_name(config)
    for row in rows:
        if str(row.get("config_name", "")) == target:
            return row
    raise KeyError(f"Config row not found for {target}")


def _report_markdown(summary: dict[str, Any]) -> str:
    public_best = summary.get("public_domain", {}).get("best", {})
    gaic_best = summary.get("gaic_domain", {}).get("best", {})
    public_test = summary.get("artifacts", {}).get("public_test_summary", {})
    gaic_test = summary.get("artifacts", {}).get("gaic_test_summary", {})
    public_test_datasets = public_test.get("datasets", {}) if isinstance(public_test.get("datasets"), dict) else {}
    gaic_metrics = gaic_test.get("metrics", {}) if isinstance(gaic_test.get("metrics"), dict) else {}
    lines = [
        "# Hybrid Teacher Calibration",
        "",
        f"- public best config: `{public_best.get('config_name', '')}`",
        f"- gaic best config: `{gaic_best.get('config_name', '')}`",
        "",
        "## Test Metrics",
        "",
        "| domain | selected config | primary | FCDB IoU | CPC weighted | GNMC IoU | top1 MOS | SRCC | Acc1@10 | Accw4@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| public | {cfg} | {primary:.4f} | {fcdb:.4f} | {cpc:.4f} | {gnmc:.4f} | - | - | - | - |".format(
            cfg=public_best.get("config_name", ""),
            primary=float(public_test.get("public_macro", 0.0) or 0.0),
            fcdb=float(((public_test_datasets.get("fcdb", {}) if isinstance(public_test_datasets.get("fcdb"), dict) else {}).get("iou_top1", 0.0) or 0.0)),
            cpc=float(((public_test_datasets.get("cpc", {}) if isinstance(public_test_datasets.get("cpc"), dict) else {}).get("weighted_pairwise_acc", 0.0) or 0.0)),
            gnmc=float(((public_test_datasets.get("gnmc", {}) if isinstance(public_test_datasets.get("gnmc"), dict) else {}).get("iou_top1", 0.0) or 0.0)),
        ),
        "| gaic | {cfg} | {primary:.4f} | - | - | - | {top1:.4f} | {srcc:.4f} | {acc1:.3f} | {accw4:.4f} |".format(
            cfg=gaic_best.get("config_name", ""),
            primary=float(gaic_test.get("gaic_component", 0.0) or 0.0),
            top1=float(gaic_metrics.get("top1_mos", 0.0) or 0.0),
            srcc=float(gaic_metrics.get("srcc", 0.0) or 0.0),
            acc1=float(gaic_metrics.get("acc1_of_top10", 0.0) or 0.0),
            accw4=float(gaic_metrics.get("accw4_of_top10", 0.0) or 0.0),
        ),
        "",
        "## Public Val Top Configs",
        "",
        "| config | public macro | FCDB IoU | CPC weighted | GNMC IoU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("public_domain", {}).get("leaderboard_top", []):
        lines.append(
            "| {config_name} | {public_macro:.4f} | {fcdb_iou_top1:.4f} | {cpc_weighted_pairwise:.4f} | {gnmc_iou_top1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## GAIC Val Top Configs",
            "",
            "| config | gaic component | top1 MOS | SRCC | Acc1@10 | Accw4@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("gaic_domain", {}).get("leaderboard_top", []):
        lines.append(
            "| {config_name} | {gaic_component:.4f} | {top1_mos:.4f} | {srcc:.4f} | {acc1_of_top10:.3f} | {accw4_of_top10:.4f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if len(args.public_method_names) != len(args.public_prediction_jsonls):
        raise ValueError("public_method_names and public_prediction_jsonls lengths must match")
    if len(args.gaic_method_names) != len(args.gaic_val_prediction_jsonls) or len(args.gaic_method_names) != len(args.gaic_test_prediction_jsonls):
        raise ValueError("gaic method names and prediction jsonls lengths must match")
    if len(args.public_method_names) != 2 or len(args.gaic_method_names) != 2:
        raise ValueError("Current hybrid implementation expects exactly two public cropper methods")

    start = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_keys = _public_keys_by_split(
        args.task_manifest_jsonl,
        split_seed=int(args.split_seed),
        force_public_hash_split=bool(int(args.force_public_hash_split) > 0),
    )
    public_tasks_by_split = _load_public_tasks_by_split(
        args.task_manifest_jsonl,
        split_seed=int(args.split_seed),
        force_public_hash_split=bool(int(args.force_public_hash_split) > 0),
    )
    public_val_keys = split_keys.get("val", set())
    public_test_keys = split_keys.get("test", set())

    uctr_public_val_groups = _filter_groups(load_prediction_groups(args.uctr_public_val_jsonl.resolve()), public_val_keys)
    uctr_public_test_groups = _filter_groups(load_prediction_groups(args.uctr_public_test_jsonl.resolve()), public_test_keys)
    public_groups = {
        str(name): load_prediction_groups(path.resolve())
        for name, path in zip(args.public_method_names, args.public_prediction_jsonls)
    }
    public_val_groups = {name: _filter_groups(groups, public_val_keys) for name, groups in public_groups.items()}
    public_test_groups = {name: _filter_groups(groups, public_test_keys) for name, groups in public_groups.items()}

    uctr_gaic_val_groups = {
        key: value
        for key, value in load_prediction_groups(args.uctr_gaic_val_jsonl.resolve()).items()
        if key[0] == "gaic"
    }
    uctr_gaic_test_groups = {
        key: value
        for key, value in load_prediction_groups(args.uctr_gaic_test_jsonl.resolve()).items()
        if key[0] == "gaic"
    }
    gaic_val_groups = {
        str(name): load_prediction_groups(path.resolve())
        for name, path in zip(args.gaic_method_names, args.gaic_val_prediction_jsonls)
    }
    gaic_test_groups = {
        str(name): load_prediction_groups(path.resolve())
        for name, path in zip(args.gaic_method_names, args.gaic_test_prediction_jsonls)
    }
    gaic_val_records = load_gaic_annotation_records(
        args.gaic_val_annotations_json.resolve(),
        image_roots=[Path(path).resolve() for path in args.gaic_val_image_roots],
    )
    gaic_test_records = load_gaic_annotation_records(
        args.gaic_test_annotations_json.resolve(),
        image_roots=[Path(path).resolve() for path in args.gaic_test_image_roots],
    )

    public_leaderboard: list[dict[str, Any]] = []
    best_public_score = float("-inf")
    best_public_config: HybridTeacherConfig | None = None
    for config in _iter_configs(args):
        fused_public_val, fuse_summary = fuse_hybrid_prediction_groups(
            uctr_public_val_groups,
            public_val_groups,
            config=config,
        )
        summary = _public_summary(
            fused_public_val,
            tasks_by_split=public_tasks_by_split,
            split="val",
        )
        row = _leaderboard_row_public(config, summary, fuse_summary=fuse_summary)
        public_leaderboard.append(row)
        if float(row["public_macro"]) > best_public_score:
            best_public_score = float(row["public_macro"])
            best_public_config = config
    if best_public_config is None:
        raise RuntimeError("Failed to calibrate public hybrid config")
    public_leaderboard.sort(key=lambda row: (float(row["public_macro"]), row["config_name"]), reverse=True)

    gaic_leaderboard: list[dict[str, Any]] = []
    best_gaic_score = float("-inf")
    best_gaic_config: HybridTeacherConfig | None = None
    for config in _iter_configs(args):
        fused_gaic_val, fuse_summary = fuse_hybrid_prediction_groups(
            uctr_gaic_val_groups,
            gaic_val_groups,
            config=config,
        )
        summary = _gaic_summary(fused_gaic_val, records=gaic_val_records, method_name=_config_name(config))
        row = _leaderboard_row_gaic(config, summary, fuse_summary=fuse_summary)
        gaic_leaderboard.append(row)
        if float(row["gaic_component"]) > best_gaic_score:
            best_gaic_score = float(row["gaic_component"])
            best_gaic_config = config
    if best_gaic_config is None:
        raise RuntimeError("Failed to calibrate GAIC hybrid config")
    gaic_leaderboard.sort(key=lambda row: (float(row["gaic_component"]), row["config_name"]), reverse=True)

    fused_public_val, public_val_fuse_summary = fuse_hybrid_prediction_groups(
        uctr_public_val_groups,
        public_val_groups,
        config=best_public_config,
    )
    fused_public_test, public_test_fuse_summary = fuse_hybrid_prediction_groups(
        uctr_public_test_groups,
        public_test_groups,
        config=best_public_config,
    )
    public_val_summary = _public_summary(
        fused_public_val,
        tasks_by_split=public_tasks_by_split,
        split="val",
    )
    public_test_summary = _public_summary(
        fused_public_test,
        tasks_by_split=public_tasks_by_split,
        split="test",
    )

    fused_gaic_val, gaic_val_fuse_summary = fuse_hybrid_prediction_groups(
        uctr_gaic_val_groups,
        gaic_val_groups,
        config=best_gaic_config,
    )
    fused_gaic_test, gaic_test_fuse_summary = fuse_hybrid_prediction_groups(
        uctr_gaic_test_groups,
        gaic_test_groups,
        config=best_gaic_config,
    )
    gaic_val_summary = _gaic_summary(fused_gaic_val, records=gaic_val_records, method_name="hybrid_teacher_gaic_val")
    gaic_test_summary = _gaic_summary(fused_gaic_test, records=gaic_test_records, method_name="hybrid_teacher_gaic_test")

    public_val_jsonl = output_dir / "public_val_hybrid_predictions.jsonl"
    public_test_jsonl = output_dir / "public_test_hybrid_predictions.jsonl"
    gaic_val_jsonl = output_dir / "gaic_val_hybrid_predictions.jsonl"
    gaic_test_jsonl = output_dir / "gaic_test_hybrid_predictions.jsonl"
    write_grouped_predictions(public_val_jsonl, fused_public_val, method_name="hybrid_teacher_public", source="hybrid_teacher")
    write_grouped_predictions(public_test_jsonl, fused_public_test, method_name="hybrid_teacher_public", source="hybrid_teacher")
    write_grouped_predictions(gaic_val_jsonl, fused_gaic_val, method_name="hybrid_teacher_gaic", source="hybrid_teacher")
    write_grouped_predictions(gaic_test_jsonl, fused_gaic_test, method_name="hybrid_teacher_gaic", source="hybrid_teacher")

    selected_public_row = _select_config_row(public_leaderboard, best_public_config)
    selected_gaic_row = _select_config_row(gaic_leaderboard, best_gaic_config)

    summary = {
        "format": "hybrid_teacher_calibration_v1",
        "input": {
            "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
            "warehouse_jsonl": str(args.warehouse_jsonl.resolve()),
            "uctr_public_val_jsonl": str(args.uctr_public_val_jsonl.resolve()),
            "uctr_public_test_jsonl": str(args.uctr_public_test_jsonl.resolve()),
            "public_method_names": [str(name) for name in args.public_method_names],
            "public_prediction_jsonls": [str(path.resolve()) for path in args.public_prediction_jsonls],
            "uctr_gaic_val_jsonl": str(args.uctr_gaic_val_jsonl.resolve()),
            "uctr_gaic_test_jsonl": str(args.uctr_gaic_test_jsonl.resolve()),
            "gaic_method_names": [str(name) for name in args.gaic_method_names],
            "gaic_val_prediction_jsonls": [str(path.resolve()) for path in args.gaic_val_prediction_jsonls],
            "gaic_test_prediction_jsonls": [str(path.resolve()) for path in args.gaic_test_prediction_jsonls],
        },
        "public_domain": {
            "best": selected_public_row,
            "leaderboard_top": _short_public_table(public_leaderboard, limit=12),
            "leaderboard_size": len(public_leaderboard),
            "selected_val_summary": {
                "public_macro": _public_macro(public_val_summary),
                **public_val_summary,
            },
            "selected_val_fuse_summary": _compact_fuse_summary(public_val_fuse_summary),
        },
        "gaic_domain": {
            "best": selected_gaic_row,
            "leaderboard_top": _short_gaic_table(gaic_leaderboard, limit=12),
            "leaderboard_size": len(gaic_leaderboard),
            "selected_val_summary": {
                "gaic_component": _gaic_component(gaic_val_summary),
                **gaic_val_summary,
            },
            "selected_val_fuse_summary": _compact_fuse_summary(gaic_val_fuse_summary),
        },
        "artifacts": {
            "public_val_predictions": str(public_val_jsonl.resolve()),
            "public_test_predictions": str(public_test_jsonl.resolve()),
            "gaic_val_predictions": str(gaic_val_jsonl.resolve()),
            "gaic_test_predictions": str(gaic_test_jsonl.resolve()),
            "public_val_summary": {
                "public_macro": _public_macro(public_val_summary),
                **public_val_summary,
            },
            "public_test_summary": {
                "public_macro": _public_macro(public_test_summary),
                **public_test_summary,
            },
            "gaic_val_summary": {
                "gaic_component": _gaic_component(gaic_val_summary),
                **gaic_val_summary,
            },
            "gaic_test_summary": {
                "gaic_component": _gaic_component(gaic_test_summary),
                **gaic_test_summary,
            },
            "public_test_fuse_summary": _compact_fuse_summary(public_test_fuse_summary),
            "gaic_test_fuse_summary": _compact_fuse_summary(gaic_test_fuse_summary),
        },
        "duration_sec": round(time.time() - start, 3),
    }
    summary_json = output_dir / "summary.json"
    report_md = output_dir / "REPORT.md"
    _write_json(summary_json, summary)
    report_md.write_text(_report_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
