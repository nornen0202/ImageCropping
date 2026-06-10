#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METHOD_INFO: dict[str, dict[str, Any]] = {
    "cacnet": {
        "display": "Public cropper `CACNet` projection",
        "role": "single-crop regressor projected to benchmark candidates by IoU",
        "native_output": "single crop regression",
        "params_m": 19.525,
        "gaic_metric_source": "projection_metrics",
        "note": "projection-only; no native GAIC candidate ranking score",
    },
    "s2cnet": {
        "display": "Public cropper `S2CNet`",
        "role": "MobileNetV2 + graph-attention relation crop ranker",
        "native_output": "candidate scorer with relation graph",
        "params_m": 5.796,
        "gaic_metric_source": "metrics",
        "note": "public-weight diagnostic; uses deterministic candidate-derived graph nodes instead of original Faster-RCNN object boxes",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _zscore(value: float, values: list[float]) -> float:
    sigma = float(pstdev(values))
    if sigma <= 1e-12:
        return 0.0
    return (float(value) - float(mean(values))) / sigma


def _gaic_primary(*, top1_mos: float | None, srcc: float | None, accw4_top10: float | None) -> float | None:
    if top1_mos is None or srcc is None or accw4_top10 is None:
        return None
    return float(0.50 * (top1_mos / 5.0) + 0.35 * srcc + 0.15 * accw4_top10)


def _reference_metric_values(reference_leaderboard: dict[str, Any]) -> dict[str, list[float]]:
    keys = {
        "fcdb_iou_top1": "fcdb_iou_top1",
        "cpc_weighted_pairwise": "cpc_weighted_pairwise",
        "gnmc_iou_top1": "gnmc_iou_top1",
        "gaic_primary": "gaic_primary",
    }
    rows = reference_leaderboard.get("rows") or []
    values: dict[str, list[float]] = {}
    for out_key, row_key in keys.items():
        metric_values: list[float] = []
        for row in rows:
            number = _float(row.get(row_key))
            if number is not None:
                metric_values.append(number)
        values[out_key] = metric_values
    return values


def _public_metrics(public_summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        "fcdb_iou_top1": _float(_get(public_summary, ("datasets", "fcdb", "iou_top1"))),
        "cpc_weighted_pairwise": _float(_get(public_summary, ("datasets", "cpc", "weighted_pairwise_acc"))),
        "gnmc_iou_top1": _float(_get(public_summary, ("datasets", "gnmc", "iou_top1"))),
    }


def _method_timing(prediction_summary: dict[str, Any], method: str) -> dict[str, Any]:
    row = (prediction_summary.get("methods") or {}).get(method) or {}
    metadata = row.get("metadata") or {}
    raw_count = _float(row.get("raw_scored_record_count"))
    scoring_candidates = _float(prediction_summary.get("scoring_candidate_count"))
    duration_sec = _float(row.get("duration_sec"))
    elapsed_sec = _float(metadata.get("elapsed_sec"))
    return {
        "status": row.get("status"),
        "duration_sec": duration_sec,
        "core_elapsed_sec": elapsed_sec,
        "raw_scored_record_count": raw_count,
        "scoring_candidate_count": scoring_candidates,
        "ms_per_scored_image_group": (duration_sec * 1000.0 / raw_count) if duration_sec is not None and raw_count else None,
        "core_ms_per_scored_image_group": (elapsed_sec * 1000.0 / raw_count) if elapsed_sec is not None and raw_count else None,
        "candidate_count_mean": (scoring_candidates / raw_count) if scoring_candidates is not None and raw_count else None,
        "metadata": metadata,
    }


def build_rows(*, run_dir: Path, public_eval_root: Path, reference_leaderboard: dict[str, Any]) -> list[dict[str, Any]]:
    gpu_summary = _read_json(run_dir / "gpu_benchmark_summary.json")
    prediction_summary = _read_json(Path(gpu_summary["prediction_summary_json"]))
    gaic_summary = _read_json(Path(gpu_summary["gaic_v2_metrics_json"]))
    reference_values = _reference_metric_values(reference_leaderboard)

    rows: list[dict[str, Any]] = []
    for method, info in METHOD_INFO.items():
        eval_summary = _read_json(public_eval_root / f"public_cropper_{method}" / "public_benchmark_summary.json")
        public = _public_metrics(eval_summary)
        gaic_row = (gaic_summary.get("methods") or {}).get(method) or {}
        gaic_source_key = str(info["gaic_metric_source"])
        gaic_metrics = gaic_row.get(gaic_source_key) or {}
        top1_mos = _float(gaic_metrics.get("top1_mos"))
        srcc = _float(gaic_metrics.get("srcc"))
        accw4 = _float(gaic_metrics.get("accw4_of_top10"))
        gaic_primary = _gaic_primary(top1_mos=top1_mos, srcc=srcc, accw4_top10=accw4)

        metric_values = {
            "fcdb_iou_top1": public["fcdb_iou_top1"],
            "cpc_weighted_pairwise": public["cpc_weighted_pairwise"],
            "gnmc_iou_top1": public["gnmc_iou_top1"],
            "gaic_primary": gaic_primary,
        }
        z_values: dict[str, float | None] = {}
        for key, value in metric_values.items():
            z_values[f"z_{key}"] = _zscore(value, reference_values[key]) if value is not None else None
        complete_values = [value for value in metric_values.values() if value is not None]
        equal4_raw = float(sum(complete_values) / len(complete_values)) if len(complete_values) == 4 else None
        complete_z = [value for value in z_values.values() if value is not None]
        equal4_z = float(sum(complete_z) / len(complete_z)) if len(complete_z) == 4 else None
        worst_z = float(min(complete_z)) if len(complete_z) == 4 else None

        rows.append(
            {
                "method": f"public_cropper_{method}",
                "display": info["display"],
                "role": info["role"],
                "native_output": info["native_output"],
                "status": "ok" if eval_summary and gaic_row.get("status") in {"ok", "reference"} else gaic_row.get("status"),
                "fcdb_iou_top1": public["fcdb_iou_top1"],
                "cpc_weighted_pairwise": public["cpc_weighted_pairwise"],
                "gnmc_iou_top1": public["gnmc_iou_top1"],
                "top1_mos": top1_mos,
                "srcc": srcc,
                "accw4_of_top10": accw4,
                "gaic_primary": gaic_primary,
                "gaic_metric_source": gaic_source_key,
                "equal4_raw_mean": equal4_raw,
                "equal4_zscore_ref20260423": equal4_z,
                "worst_dataset_z_ref20260423": worst_z,
                **z_values,
                "params_m": info["params_m"],
                "timing": _method_timing(prediction_summary, method),
                "note": info["note"],
            }
        )
    rows.sort(key=lambda row: (_float(row.get("equal4_raw_mean")) or -1.0), reverse=True)
    return rows


def build_markdown(rows: list[dict[str, Any]], *, run_dir: Path, public_eval_root: Path) -> str:
    lines = [
        "# Public Cropper Extension Benchmark Tables",
        "",
        f"- GPU run dir: `{run_dir}`",
        f"- public eval root: `{public_eval_root}`",
        "- `equal4_zscore_ref20260423` uses the original 2026-04-23 teacher/public leaderboard distribution as the fixed z-score reference.",
        "- CACNet rows are projection diagnostics because CACNet emits one crop, not native candidate scores.",
        "",
        "## Teacher/Public Ceiling Extension",
        "",
        "| Method | status | native output | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary | equal4 raw | equal4 z ref | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {native} | {fcdb} | {cpc} | {gnmc} | {mos} | {srcc} | {accw4} | {gaicp} | {raw} | {z} | {note} |".format(
                method=row["display"],
                status=row.get("status", "-"),
                native=row.get("native_output", "-"),
                fcdb=_fmt(row.get("fcdb_iou_top1")),
                cpc=_fmt(row.get("cpc_weighted_pairwise")),
                gnmc=_fmt(row.get("gnmc_iou_top1")),
                mos=_fmt(row.get("top1_mos")),
                srcc=_fmt(row.get("srcc")),
                accw4=_fmt(row.get("accw4_of_top10")),
                gaicp=_fmt(row.get("gaic_primary")),
                raw=_fmt(row.get("equal4_raw_mean"), 6),
                z=_fmt(row.get("equal4_zscore_ref20260423"), 6),
                note=row.get("note", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## Public Cropper Scoring Time",
            "",
            "| Method | device/run | params M | scored image groups | candidate avg | duration sec | ms/group wall | ms/group core | note |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        timing = row.get("timing") or {}
        lines.append(
            "| {method} | A100 interactive full export | {params} | {groups} | {cand_avg} | {dur} | {wall} | {core} | {note} |".format(
                method=row["display"],
                params=_fmt(row.get("params_m"), 3),
                groups=_fmt(timing.get("raw_scored_record_count"), 0),
                cand_avg=_fmt(timing.get("candidate_count_mean"), 2),
                dur=_fmt(timing.get("duration_sec"), 3),
                wall=_fmt(timing.get("ms_per_scored_image_group"), 3),
                core=_fmt(timing.get("core_ms_per_scored_image_group"), 3),
                note=row.get("note", "-"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report tables for CACNet/S2CNet public cropper extension.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--public_eval_root", type=Path, required=True)
    parser.add_argument(
        "--reference_equal4_json",
        type=Path,
        default=Path("artifacts/unified_public_benchmark_20260423_equal4_teacher/equal4_teacher_leaderboard.json"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = _read_json(args.reference_equal4_json)
    rows = build_rows(run_dir=args.run_dir, public_eval_root=args.public_eval_root, reference_leaderboard=reference)
    payload = {
        "run_dir": str(args.run_dir),
        "public_eval_root": str(args.public_eval_root),
        "reference_equal4_json": str(args.reference_equal4_json),
        "rows": rows,
    }
    (args.output_dir / "public_cropper_extension_tables.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "PUBLIC_CROPPER_EXTENSION_TABLES.md").write_text(
        build_markdown(rows, run_dir=args.run_dir, public_eval_root=args.public_eval_root),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
