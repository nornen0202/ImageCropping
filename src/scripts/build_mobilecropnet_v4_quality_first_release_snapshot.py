#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="기존 MobileCropNet v4 최종 리더보드에 quality-first no-prior 후보를 합산한 스냅샷을 만든다."
    )
    parser.add_argument("--base_leaderboard_json", required=True, type=Path)
    parser.add_argument("--quality_bundle_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--run_name", default="mcn-q-cnv2b448-actionroute-hardrank-r2-swinroute-embedded-20260428")
    parser.add_argument("--profile", default="cnv2b_448_swinroute_vthr030")
    parser.add_argument("--track", default="Quality-first no-prior")
    parser.add_argument("--variant", default="actionroute_hardrank_r2_swinroute_embedded")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float | None]) -> float | None:
    items = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(items) / len(items) if items else None


def _zscore(value: float | None, population: list[float]) -> float | None:
    if value is None or not population:
        return None
    mean = sum(population) / len(population)
    var = sum((item - mean) ** 2 for item in population) / max(1, len(population))
    std = math.sqrt(max(var, 1.0e-12))
    return (value - mean) / std


def _load_gaic_primary(metrics: dict[str, Any]) -> dict[str, float | None]:
    methods = metrics.get("methods", {}) if isinstance(metrics.get("methods"), dict) else {}
    chosen: dict[str, Any] = {}
    for name, row in methods.items():
        if "oracle" not in str(name).lower():
            chosen = row
            break
    raw = chosen.get("metrics", {}) if isinstance(chosen, dict) else {}
    top1 = _f(raw.get("top1_mos"))
    srcc = _f(raw.get("srcc"))
    accw4 = _f(raw.get("accw4_of_top10"))
    primary = None
    if top1 is not None and srcc is not None and accw4 is not None:
        primary = 0.50 * (top1 / 5.0) + 0.35 * srcc + 0.15 * accw4
    return {
        "official_top1_mos": top1,
        "official_srcc": srcc,
        "official_accw4_top10": accw4,
        "official_pcc": _f(raw.get("pcc")),
        "gaic_primary": primary,
    }


def _load_latency(path: Path) -> dict[str, float | None]:
    payload = _read_json(path)
    rows = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    if not rows:
        return {}
    benches = rows[0].get("benchmarks", []) if isinstance(rows[0], dict) else []
    selected = {}
    for bench in benches:
        if int(bench.get("candidate_count") or 0) == 64:
            selected = bench
            break
    if not selected and benches:
        selected = benches[0]
    return {
        "latency_mean_ms": _f(selected.get("mean_ms")),
        "latency_p90_ms": _f(selected.get("p90_ms")),
        "latency_peak_memory_mb": _f(selected.get("peak_memory_mb")),
        "latency_images_per_sec": _f(selected.get("images_per_sec")),
    }


def _append_quality_row(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = args.quality_bundle_dir
    public = _read_json(bundle / "public_benchmark" / "eval_multi_cpu" / "summary.json")
    gaic = _read_json(bundle / "gaic_official_test" / "metrics.json")
    head = _read_json(bundle / "test_gate_vthr020b" / "head_audit_test" / "release_head_audit_summary.json")
    direct = _read_json(bundle / "test_gate_vthr030" / "direct_test_proposal_topk_rerank" / "metrics.json")
    qual = _read_json(bundle / "test_gate_vthr030" / "qualitative_test" / "qualitative_review_pack.json")
    latency = _load_latency(bundle / "test_gate_vthr020b" / "latency" / "latency.json")

    datasets = public.get("datasets", {}) if isinstance(public.get("datasets"), dict) else {}
    metrics = direct.get("metrics", {}) if isinstance(direct.get("metrics"), dict) else {}
    route = head.get("route", {}) if isinstance(head.get("route"), dict) else {}
    subject = head.get("subject_box", {}) if isinstance(head.get("subject_box"), dict) else {}
    failure = head.get("failure_taxonomy", {}) if isinstance(head.get("failure_taxonomy"), dict) else {}
    gaic_values = _load_gaic_primary(gaic)
    row: dict[str, Any] = {
        "track": args.track,
        "variant": args.variant,
        "profile": args.profile,
        "run_name": args.run_name,
        "run_dir": str(args.quality_bundle_dir),
        "checkpoint": str(args.checkpoint),
        "eval_output_dir": str(args.quality_bundle_dir),
        "public_summary_exists": bool(public),
        "head_summary_exists": bool(head),
        "direct_summary_exists": bool(direct),
        "fcdb_primary": _f((datasets.get("fcdb") or {}).get("iou_top1")),
        "cpc_primary": _f((datasets.get("cpc") or {}).get("weighted_pairwise_acc")),
        "gnmc_primary": _f((datasets.get("gnmc") or {}).get("iou_top1")),
        "route_accuracy": _f(route.get("accuracy")),
        "route_balanced_accuracy": _f(route.get("balanced_accuracy")),
        "direct_test_route_acc": _f(metrics.get("route_acc")),
        "direct_test_final_positive_hit_iou_0_5": _f(metrics.get("final_positive_hit_iou_0_5")),
        "direct_test_final_best_iou_to_positive": _f(metrics.get("final_best_iou_to_positive")),
        "direct_test_final_iou_to_best_positive": _f(metrics.get("final_iou_to_best_positive")),
        "direct_test_final_risk_hit_iou_0_5": _f(metrics.get("final_risk_hit_iou_0_5")),
        "direct_test_final_target_ar_compatible": _f(metrics.get("final_target_ar_compatible")),
        "direct_test_proposal_positive_recall_at_5_iou_0_5": _f(metrics.get("proposal_positive_recall_at_5_iou_0_5")),
        "direct_test_proposal_target_ar_recall_at_5_iou_0_5": _f(metrics.get("proposal_target_ar_recall_at_5_iou_0_5")),
        "direct_test_subject_box_iou_to_teacher": _f(metrics.get("subject_box_iou_to_teacher")),
        "direct_test_subject_box_pred_conf": _f(metrics.get("subject_box_pred_conf")),
        "direct_test_subject_box_valid_acc": _f(metrics.get("subject_box_valid_acc")),
        "direct_test_subject_box_valid_negative_acc": _f(metrics.get("subject_box_valid_negative_acc")),
        "direct_test_subject_box_valid_positive_acc": _f(metrics.get("subject_box_valid_positive_acc")),
        "subject_valid_threshold": _f(metrics.get("subject_box_valid_threshold")),
        "head_top1_not_positive_rate": _f(((failure.get("top1_not_positive") or {}).get("rate"))),
        "head_route_mismatch_rate": _f(((failure.get("route_mismatch") or {}).get("rate"))),
        "qual_subject_valid_mismatch_rate": _f(((qual.get("failure_summary") or {}).get("subject_valid_mismatch") or {}).get("rate")),
        "qual_subject_iou_low_rate": _f(((qual.get("failure_summary") or {}).get("subject_iou_low") or {}).get("rate")),
        "qual_final_not_positive_rate": _f(((qual.get("failure_summary") or {}).get("final_not_positive") or {}).get("rate")),
        "qual_risk_hit_rate": _f(((qual.get("failure_summary") or {}).get("risk_hit") or {}).get("rate")),
        "qual_proposal_top5_miss_rate": _f(((qual.get("failure_summary") or {}).get("proposal_top5_miss") or {}).get("rate")),
        "input_size": 448,
        "candidate_k": 64,
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k_384 + embedded_swinv2b_route_expert",
        "selection_metric": "quality_first_release_snapshot",
        "subject_box_valid_release_gate_pass": bool(subject.get("recommended_valid_calibration", {}).get("release_gate_pass")),
        **gaic_values,
        **latency,
    }
    row["gaic_primary"] = gaic_values.get("gaic_primary")
    row["equal4_raw_mean"] = _mean([row["fcdb_primary"], row["cpc_primary"], row["gnmc_primary"], row["gaic_primary"]])
    row["direct_alignment_score"] = _mean(
        [
            row["direct_test_final_positive_hit_iou_0_5"],
            row["direct_test_final_best_iou_to_positive"],
            1.0 - row["direct_test_final_risk_hit_iou_0_5"]
            if row["direct_test_final_risk_hit_iou_0_5"] is not None
            else None,
            row["direct_test_final_target_ar_compatible"],
            row["direct_test_proposal_positive_recall_at_5_iou_0_5"],
            row["direct_test_proposal_target_ar_recall_at_5_iou_0_5"],
        ]
    )
    row["head_sanity_score"] = _mean(
        [
            row["route_balanced_accuracy"],
            1.0 - row["head_route_mismatch_rate"] if row["head_route_mismatch_rate"] is not None else None,
            1.0 - row["head_top1_not_positive_rate"] if row["head_top1_not_positive_rate"] is not None else None,
            row["direct_test_subject_box_valid_acc"],
        ]
    )
    rows.append(row)
    return row


def _decorate(rows: list[dict[str, Any]]) -> None:
    axes = ("fcdb_primary", "cpc_primary", "gnmc_primary", "gaic_primary")
    populations = {key: [float(row[key]) for row in rows if row.get(key) is not None] for key in axes}
    best_by_axis = {key: max(values) if values else None for key, values in populations.items()}
    best_crop = None
    for row in rows:
        z_values = []
        gaps = []
        for key in axes:
            z = _zscore(_f(row.get(key)), populations[key])
            row[f"{key}_z"] = z
            if z is not None:
                z_values.append(z)
            best = best_by_axis[key]
            value = _f(row.get(key))
            if best is not None and value is not None:
                gaps.append(float(best) - float(value))
        row["equal4_zscore"] = _mean(z_values)
        row["worst_dataset_z"] = min(z_values) if z_values else None
        row["worst_dataset_gap"] = max(gaps) if gaps else None
        if row.get("public_summary_exists") and row.get("equal4_zscore") is not None:
            best_crop = max(float(row["equal4_zscore"]), best_crop if best_crop is not None else float("-inf"))
    for row in rows:
        row["crop_top_set"] = bool(
            best_crop is not None
            and row.get("public_summary_exists")
            and row.get("equal4_zscore") is not None
            and float(best_crop) - float(row["equal4_zscore"]) <= 0.01
        )
        row["worst_dataset_gate"] = bool(row.get("worst_dataset_gap") is not None and float(row["worst_dataset_gap"]) <= 0.03)
        row["hard_gate_target_ar"] = bool(
            row.get("direct_test_final_target_ar_compatible") is not None
            and float(row["direct_test_final_target_ar_compatible"]) >= 0.999
        )
        row["hard_gate_risk"] = bool(
            row.get("direct_test_final_risk_hit_iou_0_5") is not None
            and float(row["direct_test_final_risk_hit_iou_0_5"]) <= 0.05
        )
        row["hard_gate_prop_target"] = bool(
            row.get("direct_test_proposal_target_ar_recall_at_5_iou_0_5") is not None
            and float(row["direct_test_proposal_target_ar_recall_at_5_iou_0_5"]) >= 0.90
        )
        row["hard_gate_route_balance"] = bool(
            row.get("route_balanced_accuracy") is not None and float(row["route_balanced_accuracy"]) >= 0.50
        )
        row["hard_gate_subject_box"] = bool(
            row.get("direct_test_subject_box_iou_to_teacher") is not None
            and float(row["direct_test_subject_box_iou_to_teacher"]) >= 0.50
            and row.get("direct_test_subject_box_valid_acc") is not None
            and float(row["direct_test_subject_box_valid_acc"]) >= 0.55
            and row.get("direct_test_subject_box_valid_negative_acc") is not None
            and float(row["direct_test_subject_box_valid_negative_acc"]) >= 0.60
            and row.get("direct_test_subject_box_valid_positive_acc") is not None
            and float(row["direct_test_subject_box_valid_positive_acc"]) >= 0.50
        )
        row["qualitative_final_crop_gate"] = bool(
            row.get("qual_final_not_positive_rate") is not None
            and float(row["qual_final_not_positive_rate"]) <= 0.05
            and row.get("qual_risk_hit_rate") is not None
            and float(row["qual_risk_hit_rate"]) <= 0.05
        )
        row["final_gate_pass"] = bool(
            row.get("crop_top_set")
            and row.get("worst_dataset_gate")
            and row.get("hard_gate_target_ar")
            and row.get("hard_gate_risk")
            and row.get("hard_gate_prop_target")
            and row.get("hard_gate_route_balance")
            and row.get("hard_gate_subject_box")
            and row.get("qualitative_final_crop_gate")
        )


def _fmt(value: Any, digits: int = 6) -> str:
    number = _f(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _write_markdown(path: Path, payload: dict[str, Any], quality_row: dict[str, Any]) -> None:
    rows = sorted(
        payload["rows"],
        key=lambda row: (
            1 if row.get("final_gate_pass") else 0,
            _f(row.get("equal4_zscore")) or -1.0e9,
            _f(row.get("direct_alignment_score")) or -1.0e9,
        ),
        reverse=True,
    )
    lines = [
        "# MobileCropNet v4 품질 우선 릴리스 스냅샷",
        "",
        f"- 생성 시각: `{payload['generated_at']}`",
        f"- 전체 row: `{len(rows)}`",
        f"- 신규 후보: `{quality_row['run_name']}`",
        f"- 배포 판정: `{'통과' if quality_row.get('final_gate_pass') else '보류'}`",
        "",
        "## 신규 후보 요약",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| FCDB IoU | {_fmt(quality_row.get('fcdb_primary'))} |",
        f"| CPC weighted | {_fmt(quality_row.get('cpc_primary'))} |",
        f"| GNMC IoU | {_fmt(quality_row.get('gnmc_primary'))} |",
        f"| GAIC top1 MOS | {_fmt(quality_row.get('official_top1_mos'))} |",
        f"| GAIC SRCC | {_fmt(quality_row.get('official_srcc'))} |",
        f"| GAIC Accw4@10 | {_fmt(quality_row.get('official_accw4_top10'))} |",
        f"| GAIC primary | {_fmt(quality_row.get('gaic_primary'))} |",
        f"| equal4 raw | {_fmt(quality_row.get('equal4_raw_mean'))} |",
        f"| equal4 z | {_fmt(quality_row.get('equal4_zscore'))} |",
        f"| route balanced | {_fmt(quality_row.get('route_balanced_accuracy'))} |",
        f"| subject IoU | {_fmt(quality_row.get('direct_test_subject_box_iou_to_teacher'))} |",
        f"| subject valid negative | {_fmt(quality_row.get('direct_test_subject_box_valid_negative_acc'))} |",
        f"| final hit@0.5 | {_fmt(quality_row.get('direct_test_final_positive_hit_iou_0_5'))} |",
        f"| final risk hit | {_fmt(quality_row.get('direct_test_final_risk_hit_iou_0_5'))} |",
        f"| latency mean ms | {_fmt(quality_row.get('latency_mean_ms'), 3)} |",
        "",
        "## 판정",
        "",
        "이 후보는 `image + target_ar` no-prior 계약, subject bbox/valid calibration, target-AR compatibility, final crop action direct gate를 가장 많이 닫았다. "
        "하지만 public/equal-4 crop quality가 기존 top-set보다 낮고 raw top1/ranker failure가 높아 최종 배포 확정은 보류한다. "
        "따라서 public utility/ranker 보정 후속 run의 terminal 결과와 함께 다시 gate를 판정한다.",
        "",
        "## 통합 리더보드",
        "",
        "| 순위 | 트랙 | 변형 | 프로파일 | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary | equal4 raw | equal4 z | route bal | subj IoU | subj neg | final hit | risk hit | latency ms | gate |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row.get('track')} | {row.get('variant')} | {row.get('profile')} | "
            f"{_fmt(row.get('fcdb_primary'))} | {_fmt(row.get('cpc_primary'))} | {_fmt(row.get('gnmc_primary'))} | "
            f"{_fmt(row.get('official_top1_mos'))} | {_fmt(row.get('official_srcc'))} | {_fmt(row.get('official_accw4_top10'))} | "
            f"{_fmt(row.get('gaic_primary'))} | {_fmt(row.get('equal4_raw_mean'))} | {_fmt(row.get('equal4_zscore'))} | "
            f"{_fmt(row.get('route_balanced_accuracy'))} | {_fmt(row.get('direct_test_subject_box_iou_to_teacher'))} | "
            f"{_fmt(row.get('direct_test_subject_box_valid_negative_acc'))} | {_fmt(row.get('direct_test_final_positive_hit_iou_0_5'))} | "
            f"{_fmt(row.get('direct_test_final_risk_hit_iou_0_5'))} | {_fmt(row.get('latency_mean_ms'), 3)} | "
            f"{'통과' if row.get('final_gate_pass') else '미통과'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    base = _read_json(args.base_leaderboard_json)
    rows = [dict(row) for row in base.get("rows", [])]
    quality_row = _append_quality_row(args, rows)
    _decorate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "source_base_leaderboard_json": str(args.base_leaderboard_json),
        "quality_bundle_dir": str(args.quality_bundle_dir),
        "entry_count": len(rows),
        "completed_entry_count": sum(
            1
            for row in rows
            if row.get("public_summary_exists") and row.get("head_summary_exists") and row.get("direct_summary_exists")
        ),
        "quality_first_candidate": quality_row,
        "rows": rows,
    }
    (args.output_dir / "final_deployment_leaderboard_quality_first_snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "final_deployment_leaderboard_quality_first_snapshot.md", payload, quality_row)
    print(json.dumps({"output_dir": str(args.output_dir), "quality_final_gate_pass": quality_row.get("final_gate_pass")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
