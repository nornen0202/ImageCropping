from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from public_benchmark.adapters import iter_tasks
from public_benchmark.gallery import build_failure_gallery
from public_benchmark.metrics import aggregate_rows, evaluate_task, load_predictions
from public_benchmark.pairwise_analysis import (
    build_pairwise_failure_report,
    extract_pairwise_failures,
    summarize_pairwise_failures,
    write_jsonl as write_pairwise_jsonl,
)
from public_benchmark.regression import build_regression_report, compare_summaries, load_gate_config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["dataset", "image_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _get_path(payload: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _copy_metric_keys(payload: Optional[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {key: None for key in keys}
    return {key: payload.get(key) for key in keys}


def _first_dict(*items: Any) -> dict[str, Any]:
    for item in items:
        if isinstance(item, dict):
            return item
    return {}


def _summarize_gaic_protocol(payload: dict[str, Any], protocol: str) -> dict[str, Any]:
    score_field = _get_path(payload, "config.score_profile.semantics.official_score_field", "crop_utility_raw")
    trend_by_field = _get_path(payload, f"{protocol}.trend_by_field", {})
    trend = _first_dict(
        trend_by_field.get(score_field) if isinstance(trend_by_field, dict) else None,
        trend_by_field.get("crop_utility_raw") if isinstance(trend_by_field, dict) else None,
        trend_by_field.get("crop_utility") if isinstance(trend_by_field, dict) else None,
        trend_by_field.get("score_policy") if isinstance(trend_by_field, dict) else None,
        trend_by_field.get("score_rank") if isinstance(trend_by_field, dict) else None,
    )
    metric_keys = [
        "image_count",
        "mean_spearman",
        "mean_kendall_tau_b",
        "mean_weighted_pair_acc",
        "pooled_weighted_pair_acc",
        "mean_hit_at_1",
        "mean_topq_jaccard",
        "mean_ndcg_at_topq",
    ]
    out = _copy_metric_keys(trend, metric_keys)
    out["score_field"] = score_field
    out["confidence_intervals"] = trend.get("confidence_intervals", {}) if isinstance(trend, dict) else {}
    return out


def summarize_gaic_benchmark(summary_json: Path, report_md: Optional[Path] = None) -> dict[str, Any]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    prod_selection = _first_dict(
        _get_path(payload, "Ge.prod_selection_crop_utility"),
        _get_path(payload, "Ge.prod_selection"),
        _get_path(payload, "Ge.prod_selection_policy"),
        _get_path(payload, "Ge.prod_selection_rank"),
    )
    dataset_keys = [
        "run_image_count",
        "feature_image_count",
        "gt_total_image_count",
        "overlap_image_count",
        "overlap_train_count",
        "overlap_test_count",
        "raw_free_candidate_count_mean",
        "training_label_free_candidate_count_mean",
    ]
    coverage_keys = [
        "top1_oracle_iou_mean",
        "top1_coverage_at_05_mean",
        "top1_coverage_at_07_mean",
        "topq_any_coverage_at_05_mean",
    ]
    prod_keys = [
        "image_count",
        "non_gt_winner_rate",
        "exact_gt_winner_rate",
        "gt_best_iou_mean",
        "gt_best_hit_at_05",
        "gt_best_hit_at_07",
        "matched_gt_iou_mean",
        "matched_gt_mos_percentile_mean",
        "matched_gt_topq_rate",
        "gt_vs_prod_disagreement_rate",
    ]
    return {
        "included": True,
        "source_summary_json": str(summary_json),
        "source_report_md": str(report_md) if report_md else "",
        "run": {
            "generated_at": _get_path(payload, "run.generated_at"),
            "duration_sec": _get_path(payload, "run.duration_sec"),
        },
        "score_profile": {
            "profile_name": _get_path(payload, "config.score_profile.profile_name"),
            "label": _get_path(payload, "config.score_profile.label"),
            "official_score_field": _get_path(payload, "config.score_profile.semantics.official_score_field"),
            "decision_semantics": _get_path(payload, "config.score_profile.semantics.decision_semantics"),
        },
        "dataset": _copy_metric_keys(_get_path(payload, "dataset", {}), dataset_keys),
        "coverage": _copy_metric_keys(_get_path(payload, "coverage", {}), coverage_keys),
        "protocols": {
            "Gc": _summarize_gaic_protocol(payload, "Gc"),
            "Ge": _summarize_gaic_protocol(payload, "Ge"),
        },
        "prod_selection": _copy_metric_keys(prod_selection, prod_keys),
        "winner_source_family_counts": prod_selection.get("winner_source_family_counts", {}) if prod_selection else {},
        "winner_source_family_topq_rate": prod_selection.get("winner_source_family_topq_rate", {}) if prod_selection else {},
    }


def build_combined_summary(summary: dict[str, Any]) -> dict[str, Any]:
    gaic = summary.get("gaic", {})
    public_overall = summary.get("overall", {})
    gaic_images = _get_path(gaic, "dataset.overlap_image_count", 0) if gaic else 0
    return {
        "gaic_included": bool(gaic),
        "public_dataset_count": len(summary.get("datasets", {})),
        "public_task_count": public_overall.get("n_tasks", 0),
        "public_image_count": public_overall.get("n_images", 0),
        "gaic_overlap_image_count": gaic_images or 0,
        "eval_unit_count": (public_overall.get("n_tasks", 0) or 0) + (gaic_images or 0),
        "primary_axes": [
            "GAIC Gc/Ge ranking and production-selection agreement",
            "FCDB/CPC free-form GT/ranking sanity",
            "GNMC AR-conditioned editor-crop sanity",
        ],
    }


def build_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    has_gaic = bool(summary.get("gaic"))
    lines.append("# SSTK Combined Benchmark Eval Report" if has_gaic else "# SSTK Public Benchmark Eval Report")
    lines.append("")
    if has_gaic:
        lines.append("이 리포트는 GAIC Gc/Ge 평가와 FCDB/CPC/GNMC 공개 벤치마크 하네스 결과를 한 산출물로 묶어 후보 생성기와 scorer 변경의 회귀를 함께 보기 위한 결과입니다.")
    else:
        lines.append("이 리포트는 GAIC 전용 Gc/Ge 평가를 대체하지 않고, FCDB/CPC/GNMC를 같은 회귀 검증 축에 추가하기 위한 공개 벤치마크 하네스 결과입니다.")
    lines.append("")
    lines.append("## Run")
    run = summary.get("run", {})
    lines.append("")
    lines.append(f"- mode: `{run.get('mode', '')}`")
    lines.append(f"- score_field: `{run.get('score_field', '')}`")
    lines.append(f"- scoring source: `{run.get('scoring_source', '')}`")
    lines.append(f"- target_ar_filter_mode: `{run.get('target_ar_filter_mode', 'none')}`")
    lines.append(f"- pairwise_failure_max_per_task: `{run.get('pairwise_failure_max_per_task', 0)}`")
    lines.append(f"- duration_sec: `{run.get('duration_sec', 0.0)}`")
    if has_gaic:
        gaic = summary["gaic"]
        lines.append(f"- gaic summary: `{gaic.get('source_summary_json', '')}`")
        lines.append(f"- gaic profile: `{_get_path(gaic, 'score_profile.profile_name', '')}`")
    lines.append("")
    lines.append("## Combined Summary")
    lines.append("")
    combined = summary.get("combined", {})
    lines.append(f"- gaic_included: `{combined.get('gaic_included', False)}`")
    lines.append(f"- public_task_count: `{combined.get('public_task_count', 0)}`")
    lines.append(f"- public_image_count: `{combined.get('public_image_count', 0)}`")
    lines.append(f"- gaic_overlap_image_count: `{combined.get('gaic_overlap_image_count', 0)}`")
    lines.append("")
    lines.append("| suite | dataset/protocol | eval_units | rank_metric | coverage_or_selection |")
    lines.append("|---|---|---|---|---|")
    if has_gaic:
        gaic = summary["gaic"]
        for protocol in ["Gc", "Ge"]:
            row = _get_path(gaic, f"protocols.{protocol}", {})
            lines.append(
                "| GAIC | "
                + protocol
                + " | "
                + _format_value(row.get("image_count"))
                + " | spearman="
                + _format_value(row.get("mean_spearman"))
                + ", pair="
                + _format_value(row.get("mean_weighted_pair_acc"))
                + " | oracle_iou="
                + _format_value(_get_path(gaic, "coverage.top1_oracle_iou_mean"))
                + ", cov@0.7="
                + _format_value(_get_path(gaic, "coverage.top1_coverage_at_07_mean"))
                + " |"
            )
        prod = gaic.get("prod_selection", {})
        lines.append(
            "| GAIC | Ge prod_selection | "
            + _format_value(prod.get("image_count"))
            + " | matched_gt_pct="
            + _format_value(prod.get("matched_gt_mos_percentile_mean"))
            + " | matched_iou="
            + _format_value(prod.get("matched_gt_iou_mean"))
            + ", gt_hit@0.7="
            + _format_value(prod.get("gt_best_hit_at_07"))
            + " |"
        )
    for dataset, row in sorted(summary.get("datasets", {}).items()):
        lines.append(
            "| Public | "
            + dataset
            + " | "
            + _format_value(row.get("n_tasks"))
            + " | pairwise="
            + _format_value(row.get("weighted_pairwise_acc"))
            + ", gt_rank@5="
            + _format_value(row.get("gt_rank_at_5"))
            + " | iou_top1="
            + _format_value(row.get("iou_top1"))
            + ", cov@0.7="
            + _format_value(row.get("coverage_at_07_preinject"))
            + " |"
        )
    lines.append("")
    lines.append("## Public Dataset Summary")
    lines.append("")
    headers = [
        "dataset",
        "tasks",
        "images",
        "pairwise_acc",
        "gt_rank@5",
        "cov@0.7 pre",
        "iou_top1 pre",
        "iou_top1",
        "ar_viol",
        "schema_fail",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for dataset, row in sorted(summary.get("datasets", {}).items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    _format_value(row.get("n_tasks")),
                    _format_value(row.get("n_images")),
                    _format_value(row.get("weighted_pairwise_acc")),
                    _format_value(row.get("gt_rank_at_5")),
                    _format_value(row.get("coverage_at_07_preinject")),
                    _format_value(row.get("iou_top1_preinject")),
                    _format_value(row.get("iou_top1")),
                    _format_value(row.get("ar_constraint_violation_rate")),
                    _format_value(row.get("schema_fail_rate")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- FCDB 로컬 사본은 현재 단일 GT crop만 포함하므로 pairwise 지표는 산출하지 않습니다.")
    lines.append("- CPC는 annotator score에서 pairwise preference를 생성합니다. 실제 모델 평가 시에는 `--predictions_jsonl`에 SSTK 후보/스코어를 넣어야 합니다.")
    lines.append("- GNMC는 AR별 task로 풀어 `iou_top1`, `coverage@0.7`, AR constraint violation을 산출합니다.")
    if has_gaic:
        lines.append("- GAIC 값은 지정한 `--gaic_summary_json`을 정규화해 포함한 것이며, GAIC raw 재평가는 별도 `run_gaic_benchmark_eval.py` 산출물에서 수행합니다.")
    gate = summary.get("regression_gate")
    if isinstance(gate, dict):
        lines.append("")
        lines.append("## Regression Gate")
        lines.append("")
        lines.append(f"- status: `{gate.get('status', 'unknown')}`")
        lines.append(f"- failures: `{gate.get('failure_count', 0)}`")
        for failure in gate.get("failures", [])[:10]:
            lines.append(
                "- fail: "
                + f"`{failure.get('suite')}/{failure.get('item')}/{failure.get('metric')}` "
                + f"baseline=`{_format_value(failure.get('baseline'))}` "
                + f"current=`{_format_value(failure.get('current'))}`"
            )
    gallery = summary.get("failure_gallery")
    if isinstance(gallery, dict):
        lines.append("")
        lines.append("## Failure Gallery")
        lines.append("")
        lines.append(f"- rendered_cases: `{gallery.get('rendered_cases', 0)}`")
        lines.append(f"- index: `{gallery.get('index_md', '')}`")
    pairwise = summary.get("pairwise_failure_analysis")
    if isinstance(pairwise, dict):
        lines.append("")
        lines.append("## Pairwise Failure Analysis")
        lines.append("")
        lines.append(f"- failure_count_recorded: `{pairwise.get('failure_count', 0)}`")
        lines.append(f"- report: `{pairwise.get('report_md', '')}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCDB/CPC/GNMC public benchmark validation with optional GAIC summary aggregation.")
    parser.add_argument("--data_root", default="data/Publics")
    parser.add_argument("--datasets", nargs="+", default=["fcdb", "cpc", "gnmc"])
    parser.add_argument("--split", default="all")
    parser.add_argument("--mode", choices=["S", "SC"], default="S")
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tasks_per_dataset", type=int, default=0)
    parser.add_argument("--max_pairwise_per_task", type=int, default=200)
    parser.add_argument("--min_pairwise_gap", type=float, default=0.0)
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--scoring_fallback", choices=["center_area_prior", "none"], default="center_area_prior")
    parser.add_argument("--inject_gt", type=int, default=1)
    parser.add_argument("--ar_tolerance", type=float, default=0.03)
    parser.add_argument("--target_ar_filter_mode", choices=["none", "hard"], default="none")
    parser.add_argument("--pairwise_failure_max_per_task", type=int, default=0)
    parser.add_argument("--gaic_summary_json", default="")
    parser.add_argument("--gaic_report_md", default="")
    parser.add_argument("--baseline_summary_json", default="")
    parser.add_argument("--gate_config_json", default="")
    parser.add_argument("--fail_on_regression", type=int, default=0)
    parser.add_argument("--failure_gallery_max", type=int, default=24)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions_jsonl).resolve() if str(args.predictions_jsonl).strip() else None
    predictions = load_predictions(predictions_path)

    all_rows: list[dict[str, Any]] = []
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_samples: list[dict[str, Any]] = []
    for dataset in [str(item).lower() for item in args.datasets]:
        task_count = 0
        for task in iter_tasks(
            dataset,
            data_root,
            split=str(args.split),
            max_pairwise_per_task=int(args.max_pairwise_per_task),
            min_pairwise_gap=float(args.min_pairwise_gap),
        ):
            if int(args.max_tasks_per_dataset) > 0 and task_count >= int(args.max_tasks_per_dataset):
                break
            task_count += 1
            task_predictions = predictions.get(task.key(), [])
            row = evaluate_task(
                task,
                predictions=task_predictions,
                mode=str(args.mode),
                inject_gt=bool(int(args.inject_gt) > 0),
                score_field=str(args.score_field),
                scoring_fallback=str(args.scoring_fallback),
                ar_tolerance=float(args.ar_tolerance),
                target_ar_filter_mode=str(args.target_ar_filter_mode),
                collect_pairwise_details=int(args.pairwise_failure_max_per_task) > 0,
                max_pairwise_details=int(args.pairwise_failure_max_per_task),
            )
            all_rows.append(row)
            rows_by_dataset[dataset].append(row)
            if len(task_samples) < 50:
                task_samples.append(task.to_dict())

    datasets_summary = {dataset: aggregate_rows(rows) for dataset, rows in sorted(rows_by_dataset.items())}
    summary = {
        "run": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
            "duration_sec": round(time.time() - start, 3),
            "mode": str(args.mode),
            "inject_gt": bool(int(args.inject_gt) > 0),
            "score_field": str(args.score_field),
            "scoring_fallback": str(args.scoring_fallback),
            "ar_tolerance": float(args.ar_tolerance),
            "target_ar_filter_mode": str(args.target_ar_filter_mode),
            "pairwise_failure_max_per_task": int(args.pairwise_failure_max_per_task),
            "scoring_source": "predictions_jsonl" if predictions_path else f"dataset_score_or_{args.scoring_fallback}",
            "predictions_jsonl": str(predictions_path) if predictions_path else "",
            "data_root": str(data_root),
            "max_tasks_per_dataset": int(args.max_tasks_per_dataset),
            "max_pairwise_per_task": int(args.max_pairwise_per_task),
        },
        "overall": aggregate_rows(all_rows),
        "datasets": datasets_summary,
        "artifacts": {
            "summary_json": str(output_dir / "public_benchmark_summary.json"),
            "combined_summary_json": str(output_dir / "combined_benchmark_summary.json"),
            "per_task_jsonl": str(output_dir / "per_task_metrics.jsonl"),
            "per_task_csv": str(output_dir / "per_task_metrics.csv"),
            "task_samples_jsonl": str(output_dir / "task_samples.jsonl"),
            "report_md": str(output_dir / "PUBLIC_BENCHMARK_EVAL_REPORT_KO.md"),
            "combined_report_md": str(output_dir / "SSTK_BENCHMARK_EVAL_REPORT_KO.md"),
            "regression_gate_json": str(output_dir / "regression_gate.json"),
            "regression_gate_report_md": str(output_dir / "REGRESSION_GATE_REPORT_KO.md"),
            "failure_gallery_dir": str(output_dir / "failure_gallery"),
            "pairwise_failures_jsonl": str(output_dir / "pairwise_failures.jsonl"),
            "pairwise_failure_report_md": str(output_dir / "PAIRWISE_FAILURE_REPORT_KO.md"),
        },
    }
    gaic_summary_path = Path(args.gaic_summary_json).resolve() if str(args.gaic_summary_json).strip() else None
    if gaic_summary_path:
        gaic_report_path = Path(args.gaic_report_md).resolve() if str(args.gaic_report_md).strip() else None
        summary["gaic"] = summarize_gaic_benchmark(gaic_summary_path, gaic_report_path)
    summary["combined"] = build_combined_summary(summary)
    if int(args.failure_gallery_max) > 0:
        summary["failure_gallery"] = build_failure_gallery(
            all_rows,
            output_dir / "failure_gallery",
            max_items=int(args.failure_gallery_max),
        )
    baseline_path = Path(args.baseline_summary_json).resolve() if str(args.baseline_summary_json).strip() else None
    if baseline_path:
        gate_config_path = Path(args.gate_config_json).resolve() if str(args.gate_config_json).strip() else None
        baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
        gate_config = load_gate_config(gate_config_path)
        summary["regression_gate"] = compare_summaries(
            current=summary,
            baseline=baseline_summary,
            gate_config=gate_config,
        )
    if int(args.pairwise_failure_max_per_task) > 0:
        pairwise_failures = extract_pairwise_failures(all_rows)
        pairwise_summary = summarize_pairwise_failures(pairwise_failures)
        pairwise_summary["jsonl"] = str(output_dir / "pairwise_failures.jsonl")
        pairwise_summary["report_md"] = str(output_dir / "PAIRWISE_FAILURE_REPORT_KO.md")
        summary["pairwise_failure_analysis"] = pairwise_summary
        write_pairwise_jsonl(output_dir / "pairwise_failures.jsonl", pairwise_failures)
        (output_dir / "PAIRWISE_FAILURE_REPORT_KO.md").write_text(
            build_pairwise_failure_report(pairwise_summary, pairwise_failures),
            encoding="utf-8",
        )
    _write_json(output_dir / "public_benchmark_summary.json", summary)
    _write_json(output_dir / "combined_benchmark_summary.json", summary)
    _write_jsonl(output_dir / "per_task_metrics.jsonl", all_rows)
    _write_csv(output_dir / "per_task_metrics.csv", all_rows)
    _write_jsonl(output_dir / "task_samples.jsonl", task_samples)
    if "regression_gate" in summary:
        _write_json(output_dir / "regression_gate.json", summary["regression_gate"])
        (output_dir / "REGRESSION_GATE_REPORT_KO.md").write_text(
            build_regression_report(summary["regression_gate"]),
            encoding="utf-8",
        )
    report = build_report(summary)
    (output_dir / "PUBLIC_BENCHMARK_EVAL_REPORT_KO.md").write_text(report, encoding="utf-8")
    (output_dir / "SSTK_BENCHMARK_EVAL_REPORT_KO.md").write_text(report, encoding="utf-8")
    if summary.get("regression_gate", {}).get("status") == "fail" and int(args.fail_on_regression) > 0:
        raise SystemExit(2)
    return summary


def main() -> None:
    summary = run_from_args(parse_args())
    print(json.dumps(summary["artifacts"], ensure_ascii=False, indent=2, sort_keys=True))
