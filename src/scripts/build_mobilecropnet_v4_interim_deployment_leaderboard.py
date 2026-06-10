#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an interim deployment leaderboard for completed MobileCropNet v4 product runs.")
    parser.add_argument("--interim_track_leaderboard_json", type=Path, required=True)
    parser.add_argument("--matrix_status_json", type=Path, required=True)
    parser.add_argument("--sstk_public_summary_json", type=Path, default=None)
    parser.add_argument("--sstk_public_collection_json", type=Path, default=None)
    parser.add_argument("--sstk_t1_summary_json", type=Path, default=None)
    parser.add_argument("--sstk_t1_collection_json", type=Path, default=None)
    parser.add_argument("--latency_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _track_rows_from_matrix(track_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list((payload.get("tracks", {}).get(track_name, {}) or {}).get("profiles", []))


def _track_rows_from_summary(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary_payload.get("runs", []))


def _latency_index(payload: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, float]]:
    out: dict[tuple[int, int, str], dict[str, float]] = {}
    groups: dict[tuple[int, int, str], list[dict[str, float]]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        input_size = int(row.get("input_size") or 0)
        candidate_k = int(row.get("candidate_k") or 0)
        backbone = str(row.get("backbone") or "")
        if not input_size or not candidate_k or not backbone:
            continue
        key = (input_size, candidate_k, backbone)
        for bench in row.get("benchmarks", []):
            if int(bench.get("candidate_count") or 0) != candidate_k:
                continue
            groups.setdefault(key, []).append(
                {
                    "mean_ms": float(bench.get("mean_ms") or 0.0),
                    "p90_ms": float(bench.get("p90_ms") or 0.0),
                    "peak_memory_mb": float(bench.get("peak_memory_mb") or 0.0),
                    "images_per_sec": float(bench.get("images_per_sec") or 0.0),
                }
            )
    for key, rows in groups.items():
        out[key] = {
            "mean_ms": median([row["mean_ms"] for row in rows]),
            "p90_ms": median([row["p90_ms"] for row in rows]),
            "peak_memory_mb": median([row["peak_memory_mb"] for row in rows]),
            "images_per_sec": median([row["images_per_sec"] for row in rows]),
        }
    return out


def _collection_remaining(collection_payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in collection_payload.get("selected_runs", []):
        if row.get("artifact_status") != "complete":
            out.add(str(row.get("profile") or ""))
    return out


def _load_track_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    matrix_payload = _read_json(args.matrix_status_json)
    tracks = {
        "GAIC T1": _track_rows_from_matrix("GAIC T1", matrix_payload),
        "GAIC UCTR": _track_rows_from_matrix("GAIC UCTR", matrix_payload),
        "GAIC public": _track_rows_from_matrix("GAIC public", matrix_payload),
        "GAIC UCTR subjectprior": _track_rows_from_matrix("GAIC UCTR subjectprior", matrix_payload),
        "SSTK UCTR subjectprior": _track_rows_from_matrix("SSTK UCTR subjectprior", matrix_payload),
    }
    sstk_public_summary = _read_json(args.sstk_public_summary_json)
    if sstk_public_summary:
        rows = _track_rows_from_summary(sstk_public_summary)
        for row in rows:
            row.setdefault("track", "SSTK public")
            row.setdefault("variant", "subjectprior_route_proposal_v1")
        tracks["SSTK public"] = rows
    sstk_t1_summary = _read_json(args.sstk_t1_summary_json)
    if sstk_t1_summary:
        rows = _track_rows_from_summary(sstk_t1_summary)
        for row in rows:
            row.setdefault("track", "SSTK T1")
            row.setdefault("variant", "baseline_current")
        tracks["SSTK T1"] = rows
    return tracks


def _enrich_row(row: dict[str, Any], latency_by_signature: dict[tuple[int, int, str], dict[str, float]]) -> dict[str, Any]:
    run_dir = Path(row.get("run_dir") or row.get("output_dir") or "")
    config = _read_json(run_dir / "config.json")
    eval_test = _read_json(run_dir / "eval_test" / "metrics.json")
    compare = _read_json(run_dir / "compare_test" / "comparison_metrics.json")
    head_summary = _read_json(run_dir / "head_analysis" / "head_analysis_summary.json")
    latency = latency_by_signature.get(
        (
            int(config.get("input_size") or row.get("input_size") or 0),
            int(config.get("candidate_k") or row.get("candidate_k") or 0),
            str(config.get("backbone_name") or row.get("backbone") or ""),
        )
    )
    eval_metrics = eval_test.get("metrics", {}) if isinstance(eval_test.get("metrics"), dict) else {}
    compare_v4 = (compare.get("methods", {}) or {}).get("v4_model", {}) if isinstance(compare.get("methods"), dict) else {}
    return {
        "track": row.get("track"),
        "variant": row.get("variant"),
        "profile": row.get("profile"),
        "run_name": row.get("run_name"),
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt"),
        "best_selection_score": _f(row.get("best_selection_score")),
        "official_top1_mos": _f(row.get("official_top1_mos")),
        "official_regret": _f(row.get("official_regret")),
        "official_pcc": _f(row.get("official_pcc")),
        "official_srcc": _f(row.get("official_srcc")),
        "official_acc1_top5": _f(row.get("official_acc1_top5")),
        "official_acc1_top10": _f(row.get("official_acc1_top10")),
        "official_accw4_top10": _f(row.get("official_accw4_top10")),
        "eval_candidate_top1_hit": _f(eval_metrics.get("candidate_top1_hit")),
        "eval_candidate_top1_exact_best": _f(eval_metrics.get("candidate_top1_exact_best")),
        "eval_ndcg_at_5": _f(eval_metrics.get("ndcg_at_5")),
        "eval_pcc": _f(eval_metrics.get("pcc")),
        "eval_srcc": _f(eval_metrics.get("srcc")),
        "proposal_recall_at_5_iou_0_5": _f(eval_metrics.get("proposal_recall_at_5_iou_0_5")),
        "proposal_target_ar_recall_at_5_iou_0_5": _f(eval_metrics.get("proposal_target_ar_recall_at_5_iou_0_5")),
        "top1_iou_to_best_positive": _f(eval_metrics.get("top1_iou_to_best_positive")),
        "explain_applicability_precision": _f(eval_metrics.get("explain_applicability_precision")),
        "explain_applicability_recall": _f(eval_metrics.get("explain_applicability_recall")),
        "explain_label_agreement_when_applicable": _f(eval_metrics.get("explain_label_agreement_when_applicable")),
        "explain_not_applicable_false_positive_rate": _f(eval_metrics.get("explain_not_applicable_false_positive_rate")),
        "explain_mode_consistency_violation_rate": _f(eval_metrics.get("explain_mode_consistency_violation_rate")),
        "compare_chosen_label_score": _f(compare_v4.get("chosen_label_score")),
        "compare_utility_regret": _f(compare_v4.get("utility_regret")),
        "compare_candidate_top1_hit": _f(compare_v4.get("candidate_top1_hit")),
        "compare_candidate_top1_exact_best": _f(compare_v4.get("candidate_top1_exact_best")),
        "latency_mean_ms": _f((latency or {}).get("mean_ms")),
        "latency_p90_ms": _f((latency or {}).get("p90_ms")),
        "latency_peak_memory_mb": _f((latency or {}).get("peak_memory_mb")),
        "latency_images_per_sec": _f((latency or {}).get("images_per_sec")),
        "has_equal4_crop_benchmark": False,
        "has_head_analysis": bool(head_summary),
        "has_direct_eval_full": False,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    def _fmt(value: Any, digits: int = 6) -> str:
        number = _f(value)
        if number is None:
            return "-"
        return f"{number:.{digits}f}"

    lines = [
        "# MobileCropNet v4 Interim Deployment Leaderboard",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- note: `equal-4 public benchmark`, full `head_analysis`, full `product direct eval`가 없는 run은 결측으로 유지했다.",
        "",
        "## Coverage",
        "",
        "| track | completed | shortlist | missing_equal4 | missing_head | missing_direct_eval |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("coverage", []):
        lines.append(
            f"| {row['track']} | {row['completed_profile_count']} | {row['shortlist_count']} | "
            f"{row['shortlist_missing_equal4']} | {row['shortlist_missing_head_analysis']} | {row['shortlist_missing_direct_eval']} |"
        )
    lines.extend(
        [
            "",
            "## Shortlist",
            "",
            "| track | profile | sel | top1 MOS | PCC | SRCC | eval top1 hit | prop@5 | explain agree | chosen label | latency mean | gaps |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("shortlist_rows", []):
        gaps: list[str] = []
        if not row.get("has_equal4_crop_benchmark"):
            gaps.append("equal4")
        if not row.get("has_head_analysis"):
            gaps.append("head")
        if not row.get("has_direct_eval_full"):
            gaps.append("direct")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("track") or ""),
                    str(row.get("profile") or ""),
                    _fmt(row.get("best_selection_score")),
                    _fmt(row.get("official_top1_mos")),
                    _fmt(row.get("official_pcc")),
                    _fmt(row.get("official_srcc")),
                    _fmt(row.get("eval_candidate_top1_hit")),
                    _fmt(row.get("proposal_recall_at_5_iou_0_5")),
                    _fmt(row.get("explain_label_agreement_when_applicable")),
                    _fmt(row.get("compare_chosen_label_score")),
                    _fmt(row.get("latency_mean_ms"), digits=3),
                    ", ".join(gaps),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interim_payload = _read_json(args.interim_track_leaderboard_json)
    track_rows = _load_track_rows(args)
    latency_by_signature = _latency_index(_read_json(args.latency_json))

    shortlist_map = {row.get("track"): list(row.get("shortlist_profiles", [])) for row in interim_payload.get("tracks", [])}
    all_rows: list[dict[str, Any]] = []
    shortlist_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for track_name, rows in track_rows.items():
        enriched = [_enrich_row(row, latency_by_signature) for row in rows]
        all_rows.extend(enriched)
        shortlist_profiles = shortlist_map.get(track_name, [])
        selected = [row for row in enriched if row.get("profile") in shortlist_profiles]
        shortlist_rows.extend(selected)
        coverage_rows.append(
            {
                "track": track_name,
                "completed_profile_count": len(enriched),
                "shortlist_count": len(selected),
                "shortlist_missing_equal4": sum(0 if row.get("has_equal4_crop_benchmark") else 1 for row in selected),
                "shortlist_missing_head_analysis": sum(0 if row.get("has_head_analysis") else 1 for row in selected),
                "shortlist_missing_direct_eval": sum(0 if row.get("has_direct_eval_full") else 1 for row in selected),
            }
        )

    shortlist_rows.sort(key=lambda row: (str(row.get("track")), -(row.get("official_top1_mos") or 0.0), -(row.get("best_selection_score") or 0.0)))
    all_rows.sort(key=lambda row: (str(row.get("track")), str(row.get("profile"))))

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "coverage": coverage_rows,
        "shortlist_rows": shortlist_rows,
        "all_rows": all_rows,
    }
    (args.output_dir / "interim_deployment_leaderboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "interim_deployment_leaderboard.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
