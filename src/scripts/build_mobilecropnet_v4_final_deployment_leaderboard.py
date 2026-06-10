#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="shortlist 평가 산출물에서 최종 배포 리더보드를 생성한다.")
    parser.add_argument("--shortlist_manifest_json", type=Path, required=True)
    parser.add_argument("--latency_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _zscore(value: float | None, population: list[float]) -> float | None:
    if value is None or not population:
        return None
    mean = sum(population) / len(population)
    var = sum((item - mean) ** 2 for item in population) / max(1, len(population))
    std = math.sqrt(max(var, 1e-12))
    return (value - mean) / std


def _load_latency_index(payload: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, float]]:
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
    out: dict[tuple[int, int, str], dict[str, float]] = {}
    for key, rows in groups.items():
        out[key] = {
            "mean_ms": median([row["mean_ms"] for row in rows]),
            "p90_ms": median([row["p90_ms"] for row in rows]),
            "peak_memory_mb": median([row["peak_memory_mb"] for row in rows]),
            "images_per_sec": median([row["images_per_sec"] for row in rows]),
        }
    return out


def _normalise_profile_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"balanced_288", "balanced288"}:
        return "balanced_288"
    if text in {"rank_320", "rank320"}:
        return "rank_320"
    if text in {"plus_384", "plus384"}:
        return "plus_384"
    if text in {"q24_288_w075", "q24", "turbo_288"}:
        return "q24_288_w075"
    if text in {"hybrid_384", "hybrid384"}:
        return "hybrid_384"
    return text


def _profile_from_latency_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if "balanced288" in text or "balanced_288" in text:
        return "balanced_288"
    if "rank320" in text or "rank_320" in text:
        return "rank_320"
    if "plus384" in text or "plus_384" in text:
        return "plus_384"
    if "q24" in text:
        return "q24_288_w075"
    if "hybrid384" in text or "hybrid_384" in text:
        return "hybrid_384"
    return ""


def _load_latency_profile_index(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        profile = _profile_from_latency_name(row.get("name")) or _normalise_profile_key(row.get("profile"))
        if not profile:
            continue
        candidate_k = int(row.get("candidate_k") or 0)
        benches = [
            bench
            for bench in row.get("benchmarks", [])
            if isinstance(bench, dict) and (not candidate_k or int(bench.get("candidate_count") or 0) == candidate_k)
        ]
        if not benches:
            continue
        out[profile] = {
            "mean_ms": median([float(bench.get("mean_ms") or 0.0) for bench in benches]),
            "p90_ms": median([float(bench.get("p90_ms") or 0.0) for bench in benches]),
            "peak_memory_mb": median([float(bench.get("peak_memory_mb") or 0.0) for bench in benches]),
            "images_per_sec": median([float(bench.get("images_per_sec") or 0.0) for bench in benches]),
        }
    return out


def _select_gaic_method(methods: dict[str, Any]) -> dict[str, Any]:
    chosen = {}
    for name, row in methods.items():
        if str(name).startswith("mcn_v4"):
            chosen = row
            break
    if not chosen:
        for name, row in methods.items():
            text = str(name).lower()
            if "teacher" in text or "oracle" in text:
                continue
            chosen = row
            break
    return chosen


def _load_gaic_metrics(run_dir: Path, eval_root: Path) -> tuple[dict[str, Any], str | None]:
    summary = _read_json(run_dir / "run_summary.json")
    if summary:
        chosen = _select_gaic_method((summary.get("gaic_official_metrics") or {}).get("methods", {}))
        if chosen:
            return chosen, str(run_dir / "run_summary.json")

    for path in (run_dir / "gaic_official_test" / "metrics.json", eval_root / "gaic_official_test" / "metrics.json"):
        payload = _read_json(path)
        chosen = _select_gaic_method(payload.get("methods", {}) if isinstance(payload.get("methods"), dict) else {})
        if chosen:
            return chosen, str(path)
    return {}, None


def _extract_model_metrics(run_dir: Path, eval_root: Path) -> dict[str, Any]:
    chosen, source = _load_gaic_metrics(run_dir, eval_root)
    metrics = chosen.get("metrics", {}) if isinstance(chosen, dict) else {}
    summary = _read_json(run_dir / "run_summary.json")
    train_config = (summary.get("train_config") if summary else None) or _read_json(run_dir / "config.json")
    return {
        "official_top1_mos": _f(metrics.get("top1_mos")),
        "official_regret": _f(metrics.get("top1_mos_regret")),
        "official_pcc": _f(metrics.get("pcc")),
        "official_srcc": _f(metrics.get("srcc")),
        "official_acc1_top5": _f(metrics.get("acc1_of_top5")),
        "official_acc1_top10": _f(metrics.get("acc1_of_top10")),
        "official_acc4_top10": _f(metrics.get("acc4_of_top10")),
        "official_accw4_top10": _f(metrics.get("accw4_of_top10")),
        "input_size": int(train_config.get("input_size") or 0),
        "candidate_k": int(train_config.get("candidate_k") or 0),
        "backbone": str(train_config.get("backbone_name") or ""),
        "selection_metric": str(train_config.get("selection_metric") or ""),
        "gaic_official_metrics_source": source,
    }


def _load_row(
    entry: dict[str, Any],
    latency_index: dict[tuple[int, int, str], dict[str, float]],
    latency_profile_index: dict[str, dict[str, float]],
) -> dict[str, Any]:
    run_dir = Path(entry["run_dir"])
    eval_root = Path(entry["output_dir"])
    metrics = _extract_model_metrics(run_dir, eval_root)
    public_summary = _read_json(eval_root / "public_benchmark" / "eval" / "summary.json")
    head_summary = _read_json(eval_root / "head_analysis" / "head_analysis_summary.json")
    direct_val = _read_json(eval_root / "direct_val_proposal_topk_rerank" / "metrics.json")
    direct_test = _read_json(eval_root / "direct_test_proposal_topk_rerank" / "metrics.json")
    direct_test_top1 = _read_json(eval_root / "direct_test_utility_top1" / "metrics.json")
    config = _read_json(run_dir / "config.json")
    signature = (
        int(metrics.get("input_size") or config.get("input_size") or 0),
        int(metrics.get("candidate_k") or config.get("candidate_k") or 0),
        str(metrics.get("backbone") or config.get("backbone_name") or ""),
    )
    latency = latency_index.get(signature, {})
    if not latency:
        latency = latency_profile_index.get(_normalise_profile_key(entry.get("profile")), {})
    row = {
        "track": entry["track"],
        "variant": entry.get("variant"),
        "profile": entry["profile"],
        "run_name": entry["run_name"],
        "run_dir": str(run_dir),
        "checkpoint": entry["checkpoint"],
        "eval_output_dir": entry["output_dir"],
        "public_summary_exists": bool(public_summary),
        "head_summary_exists": bool(head_summary),
        "direct_summary_exists": bool(direct_test),
        **metrics,
        "latency_mean_ms": _f(latency.get("mean_ms")),
        "latency_p90_ms": _f(latency.get("p90_ms")),
        "latency_peak_memory_mb": _f(latency.get("peak_memory_mb")),
        "latency_images_per_sec": _f(latency.get("images_per_sec")),
    }
    for key in (
        "official_top1_mos",
        "official_regret",
        "official_pcc",
        "official_srcc",
        "official_acc1_top5",
        "official_acc1_top10",
        "official_acc4_top10",
        "official_accw4_top10",
        "input_size",
        "candidate_k",
        "backbone",
        "selection_metric",
        "gaic_official_metrics_source",
    ):
        row.setdefault(key, None)
    datasets = public_summary.get("datasets", {}) if isinstance(public_summary.get("datasets"), dict) else {}
    row["fcdb_primary"] = _f((datasets.get("fcdb") or {}).get("iou_top1"))
    row["cpc_primary"] = _f((datasets.get("cpc") or {}).get("weighted_pairwise_acc"))
    row["gnmc_primary"] = _f((datasets.get("gnmc") or {}).get("iou_top1"))
    if row.get("official_top1_mos") is not None and row.get("official_srcc") is not None and row.get("official_accw4_top10") is not None:
        row["gaic_primary"] = 0.50 * (row["official_top1_mos"] / 5.0) + 0.35 * row["official_srcc"] + 0.15 * row["official_accw4_top10"]
    else:
        row["gaic_primary"] = None
    if all(row.get(key) is not None for key in ("fcdb_primary", "cpc_primary", "gnmc_primary", "gaic_primary")):
        row["equal4_raw_mean"] = _mean([row["fcdb_primary"], row["cpc_primary"], row["gnmc_primary"], row["gaic_primary"]])
    else:
        row["equal4_raw_mean"] = None

    route = head_summary.get("route", {}) if isinstance(head_summary.get("route"), dict) else {}
    decision = head_summary.get("decision", {}) if isinstance(head_summary.get("decision"), dict) else {}
    by_subject = head_summary.get("by_subject_mode", {}) if isinstance(head_summary.get("by_subject_mode"), dict) else {}
    mode_route_values = [
        _f((payload or {}).get("route_acc"))
        for payload in by_subject.values()
        if isinstance(payload, dict) and _f(payload.get("route_acc")) is not None
    ]
    portrait_single_route = _f((by_subject.get("portrait_single") or {}).get("route_acc"))
    portrait_group_route = _f((by_subject.get("portrait_group") or {}).get("route_acc"))
    route_accuracy = _f(route.get("accuracy"))
    route_balanced = _mean(mode_route_values)
    route_collapse = False
    for value in (portrait_single_route, portrait_group_route):
        if value is None or route_accuracy is None:
            continue
        if value < max(0.35, route_accuracy - 0.15):
            route_collapse = True
    checklist = head_summary.get("checklist", {}) if isinstance(head_summary.get("checklist"), dict) else {}
    safety_recalls = [
        _f(((checklist.get(name) or {}).get("applicability") or {}).get("recall"))
        for name in ("headroom", "lookroom", "face_cut", "joint_cut")
    ]
    macro = head_summary.get("macro", {}) if isinstance(head_summary.get("macro"), dict) else {}
    macro_pearsons = [_f((payload or {}).get("pearson")) for payload in macro.values() if isinstance(payload, dict)]
    why_micro = _f(((head_summary.get("why_tags") or {}).get("micro") or {}).get("f1"))
    proposal = head_summary.get("proposal", {}) if isinstance(head_summary.get("proposal"), dict) else {}
    row.update(
        {
            "route_accuracy": route_accuracy,
            "route_balanced_accuracy": route_balanced,
            "decision_accuracy": _f(decision.get("accuracy")),
            "why_tag_micro_f1": why_micro,
            "macro_score_mean_pearson": _mean(macro_pearsons),
            "checklist_safety_recall_mean": _mean(safety_recalls),
            "proposal_subject_hit_iou_0_5": _f(proposal.get("proposal_subject_hit_iou_0_5")),
            "proposal_to_subject_iou_mean": _f(proposal.get("proposal_to_subject_iou_mean")),
            "selected_to_subject_iou_mean": _f(proposal.get("selected_to_subject_iou_mean")),
            "route_portrait_single_acc": portrait_single_route,
            "route_portrait_group_acc": portrait_group_route,
            "route_collapse_flag": route_collapse,
        }
    )
    macro_pearson_norm = None
    if row["macro_score_mean_pearson"] is not None:
        macro_pearson_norm = _clamp01((row["macro_score_mean_pearson"] + 1.0) / 2.0)
    row["head_sanity_score"] = _mean(
        [
            row["route_balanced_accuracy"],
            row["decision_accuracy"],
            row["why_tag_micro_f1"],
            macro_pearson_norm,
            row["checklist_safety_recall_mean"],
        ]
    )

    direct_test_metrics = direct_test.get("metrics", {}) if isinstance(direct_test.get("metrics"), dict) else {}
    direct_val_metrics = direct_val.get("metrics", {}) if isinstance(direct_val.get("metrics"), dict) else {}
    direct_top1_metrics = direct_test_top1.get("metrics", {}) if isinstance(direct_test_top1.get("metrics"), dict) else {}
    row.update(
        {
            "direct_val_final_positive_hit_iou_0_5": _f(direct_val_metrics.get("final_positive_hit_iou_0_5")),
            "direct_test_final_positive_hit_iou_0_5": _f(direct_test_metrics.get("final_positive_hit_iou_0_5")),
            "direct_test_final_best_iou_to_positive": _f(direct_test_metrics.get("final_best_iou_to_positive")),
            "direct_test_final_risk_hit_iou_0_5": _f(direct_test_metrics.get("final_risk_hit_iou_0_5")),
            "direct_test_final_target_ar_compatible": _f(direct_test_metrics.get("final_target_ar_compatible")),
            "direct_test_proposal_positive_recall_at_5_iou_0_5": _f(direct_test_metrics.get("proposal_positive_recall_at_5_iou_0_5")),
            "direct_test_proposal_target_ar_recall_at_5_iou_0_5": _f(direct_test_metrics.get("proposal_target_ar_recall_at_5_iou_0_5")),
            "direct_test_route_acc": _f(direct_test_metrics.get("route_acc")),
            "direct_test_decision_acc": _f(direct_test_metrics.get("decision_acc")),
            "direct_test_utility_top1_hit_iou_0_5": _f(direct_top1_metrics.get("final_positive_hit_iou_0_5")),
            "direct_test_subject_box_iou_to_teacher": _f(direct_test_metrics.get("subject_box_iou_to_teacher")),
            "direct_test_subject_box_pred_conf": _f(direct_test_metrics.get("subject_box_pred_conf")),
            "direct_test_subject_box_target_valid": _f(direct_test_metrics.get("subject_box_target_valid")),
            "direct_test_subject_box_valid_acc": _f(direct_test_metrics.get("subject_box_valid_acc")),
            "direct_test_subject_box_valid_negative_acc": _f(direct_test_metrics.get("subject_box_valid_negative_acc")),
            "direct_test_subject_box_valid_positive_acc": _f(direct_test_metrics.get("subject_box_valid_positive_acc")),
        }
    )
    row["direct_alignment_score"] = _mean(
        [
            row["direct_test_final_positive_hit_iou_0_5"],
            row["direct_test_final_best_iou_to_positive"],
            _clamp01(1.0 - (row["direct_test_final_risk_hit_iou_0_5"] or 0.0)) if row["direct_test_final_risk_hit_iou_0_5"] is not None else None,
            row["direct_test_final_target_ar_compatible"],
            row["direct_test_proposal_positive_recall_at_5_iou_0_5"],
            row["direct_test_proposal_target_ar_recall_at_5_iou_0_5"],
            row["proposal_subject_hit_iou_0_5"],
            row["proposal_to_subject_iou_mean"],
            row["selected_to_subject_iou_mean"],
        ]
    )
    return row


def _decorate(rows: list[dict[str, Any]]) -> None:
    primaries = {}
    for key in ("fcdb_primary", "cpc_primary", "gnmc_primary", "gaic_primary"):
        primaries[key] = [float(row[key]) for row in rows if row.get(key) is not None]
    best_by_axis = {key: max(values) if values else None for key, values in primaries.items()}
    for row in rows:
        zs = {}
        for key, values in primaries.items():
            zs[key] = _zscore(row.get(key), values)
            row[f"{key}_z"] = zs[key]
        z_values = [value for value in zs.values() if value is not None]
        row["equal4_zscore"] = _mean(z_values)
        row["worst_dataset_z"] = min(z_values) if z_values else None
        gaps = []
        for key, best in best_by_axis.items():
            value = row.get(key)
            if best is None or value is None:
                continue
            gaps.append(float(best) - float(value))
        row["worst_dataset_gap"] = max(gaps) if gaps else None

    completed_crop_rows = [
        row
        for row in rows
        if row.get("equal4_raw_mean") is not None
        and row.get("equal4_zscore") is not None
        and row.get("public_summary_exists")
    ]
    best_crop = max((float(row["equal4_zscore"]) for row in completed_crop_rows), default=None)
    for row in rows:
        row["crop_top_set"] = bool(
            best_crop is not None
            and row.get("equal4_raw_mean") is not None
            and row.get("equal4_zscore") is not None
            and row.get("public_summary_exists")
            and float(best_crop) - float(row["equal4_zscore"]) <= 0.01
        )
        row["worst_dataset_gate"] = bool(row.get("worst_dataset_gap") is not None and float(row["worst_dataset_gap"]) <= 0.03)

    crop_top_completed = [row for row in rows if row.get("crop_top_set") and row.get("head_sanity_score") is not None and row.get("direct_alignment_score") is not None]
    best_head = max((float(row["head_sanity_score"]) for row in crop_top_completed), default=None)
    best_direct = max((float(row["direct_alignment_score"]) for row in crop_top_completed), default=None)
    best_prop_target = max((float(row["direct_test_proposal_target_ar_recall_at_5_iou_0_5"]) for row in crop_top_completed if row.get("direct_test_proposal_target_ar_recall_at_5_iou_0_5") is not None), default=None)
    best_risk = min((float(row["direct_test_final_risk_hit_iou_0_5"]) for row in crop_top_completed if row.get("direct_test_final_risk_hit_iou_0_5") is not None), default=None)

    for row in rows:
        row["head_gate"] = bool(best_head is not None and row.get("head_sanity_score") is not None and float(row["head_sanity_score"]) >= float(best_head) - 0.03)
        row["direct_gate"] = bool(best_direct is not None and row.get("direct_alignment_score") is not None and float(row["direct_alignment_score"]) >= float(best_direct) - 0.03)
        risk_ok = row.get("direct_test_final_risk_hit_iou_0_5") is not None and best_risk is not None and float(row["direct_test_final_risk_hit_iou_0_5"]) <= float(best_risk) + 0.03
        prop_target_ok = row.get("direct_test_proposal_target_ar_recall_at_5_iou_0_5") is not None and best_prop_target is not None and float(row["direct_test_proposal_target_ar_recall_at_5_iou_0_5"]) >= float(best_prop_target) - 0.03
        target_ar_ok = row.get("direct_test_final_target_ar_compatible") is not None and float(row["direct_test_final_target_ar_compatible"]) >= 0.999
        subject_box_ok = (
            row.get("direct_test_subject_box_iou_to_teacher") is not None
            and float(row["direct_test_subject_box_iou_to_teacher"]) >= 0.50
            and row.get("direct_test_subject_box_pred_conf") is not None
            and float(row["direct_test_subject_box_pred_conf"]) >= 0.05
            and row.get("direct_test_subject_box_valid_acc") is not None
            and float(row["direct_test_subject_box_valid_acc"]) >= 0.55
            and row.get("direct_test_subject_box_valid_negative_acc") is not None
            and float(row["direct_test_subject_box_valid_negative_acc"]) >= 0.60
            and row.get("direct_test_subject_box_valid_positive_acc") is not None
            and float(row["direct_test_subject_box_valid_positive_acc"]) >= 0.50
        )
        route_balance_ok = row.get("route_balanced_accuracy") is not None and float(row["route_balanced_accuracy"]) >= 0.50
        row["hard_gate_target_ar"] = bool(target_ar_ok)
        row["hard_gate_risk"] = bool(risk_ok)
        row["hard_gate_prop_target"] = bool(prop_target_ok)
        row["hard_gate_route_collapse"] = not bool(row.get("route_collapse_flag"))
        row["hard_gate_route_balance"] = bool(route_balance_ok)
        row["hard_gate_subject_box"] = bool(subject_box_ok)
        row["final_gate_pass"] = bool(
            row.get("crop_top_set")
            and row.get("worst_dataset_gate")
            and row.get("head_gate")
            and row.get("direct_gate")
            and row.get("hard_gate_target_ar")
            and row.get("hard_gate_risk")
            and row.get("hard_gate_prop_target")
            and row.get("hard_gate_route_collapse")
            and row.get("hard_gate_route_balance")
            and row.get("hard_gate_subject_box")
        )

    final_candidates = [row for row in rows if row.get("final_gate_pass")]
    quality_ranked = sorted(
        final_candidates,
        key=lambda row: (
            _f(row.get("equal4_zscore")) or -1e9,
            _f(row.get("head_sanity_score")) or -1e9,
            _f(row.get("direct_alignment_score")) or -1e9,
            -(_f(row.get("latency_mean_ms")) or 1e9),
        ),
        reverse=True,
    )
    latency_rows = [row for row in final_candidates if row.get("latency_mean_ms") is not None]
    latency_threshold = median([float(row["latency_mean_ms"]) for row in latency_rows]) if latency_rows else None
    default_pool = [row for row in quality_ranked if latency_threshold is not None and row.get("latency_mean_ms") is not None and float(row["latency_mean_ms"]) <= float(latency_threshold)]
    default_winner = default_pool[0] if default_pool else (quality_ranked[0] if quality_ranked else None)
    quality_winner = quality_ranked[0] if quality_ranked else None
    latency_winner = min(latency_rows, key=lambda row: float(row["latency_mean_ms"])) if latency_rows else None
    blocked_fallback_rows = sorted(
        [row for row in rows if row.get("crop_top_set")],
        key=lambda row: (
            _f(row.get("equal4_zscore")) or -1e9,
            _f(row.get("head_sanity_score")) or -1e9,
            _f(row.get("direct_alignment_score")) or -1e9,
        ),
        reverse=True,
    )
    return {
        "best_crop_zscore": best_crop,
        "best_head_sanity_score": best_head,
        "best_direct_alignment_score": best_direct,
        "best_prop_target_recall": best_prop_target,
        "best_risk_hit_rate": best_risk,
        "latency_threshold_median_ms": latency_threshold,
        "release_gate_thresholds": {
            "route_balanced_accuracy_min": 0.50,
            "subject_box_iou_to_teacher_min": 0.50,
            "subject_box_pred_conf_min": 0.05,
            "subject_box_valid_acc_min": 0.55,
            "subject_box_valid_negative_acc_min": 0.60,
            "subject_box_valid_positive_acc_min": 0.50,
            "target_ar_compatible_min": 0.999,
        },
        "default_deploy_winner": default_winner,
        "quality_first_winner": quality_winner,
        "latency_first_fallback": latency_winner,
        "blocked_fallback_top": blocked_fallback_rows[0] if blocked_fallback_rows else None,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 6) -> str:
        number = _f(value)
        if number is None:
            return "-"
        return f"{number:.{digits}f}"

    winners = payload.get("winners", {})
    lines = [
        "# MobileCropNet v4 최종 배포 리더보드",
        "",
        f"- 생성 시각: `{payload.get('generated_at')}`",
        f"- 완료 항목: `{payload.get('completed_entry_count')}` / `{payload.get('entry_count')}`",
        "",
        "## 배포 슬롯",
        "",
        "| 슬롯 | 트랙 | 프로파일 | equal4 z | head | direct | latency ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("기본 배포", "default_deploy_winner"),
        ("품질 우선", "quality_first_winner"),
        ("지연시간 우선 보조", "latency_first_fallback"),
    ):
        row = winners.get(key) or {}
        lines.append(
            f"| {label} | {row.get('track', '-')} | {row.get('profile', '-')} | {fmt(row.get('equal4_zscore'))} | "
            f"{fmt(row.get('head_sanity_score'))} | {fmt(row.get('direct_alignment_score'))} | {fmt(row.get('latency_mean_ms'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## 통합 리더보드",
            "",
            "| 순위 | 트랙 | 변형 | 프로파일 | FCDB IoU | CPC weighted | GNMC IoU | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 | GAIC primary | equal4 raw | equal4 z | worst gap | head | direct | route bal | subj IoU | subj neg | final hit | risk hit | latency ms | 통과 gate |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    rows = list(payload.get("rows", []))
    rows.sort(
        key=lambda row: (
            1 if row.get("final_gate_pass") else 0,
            _f(row.get("equal4_zscore")) or -1e9,
            _f(row.get("head_sanity_score")) or -1e9,
            _f(row.get("direct_alignment_score")) or -1e9,
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        gates = []
        for key in ("crop_top_set", "worst_dataset_gate", "head_gate", "direct_gate", "hard_gate_target_ar", "hard_gate_risk", "hard_gate_prop_target", "hard_gate_route_collapse", "hard_gate_route_balance", "hard_gate_subject_box"):
            if row.get(key):
                gates.append(key.replace("_gate_", ":").replace("hard_", "hard:"))
        lines.append(
            f"| {idx} | {row.get('track')} | {row.get('variant')} | {row.get('profile')} | "
            f"{fmt(row.get('fcdb_primary'))} | {fmt(row.get('cpc_primary'))} | {fmt(row.get('gnmc_primary'))} | "
            f"{fmt(row.get('official_top1_mos'))} | {fmt(row.get('official_srcc'))} | {fmt(row.get('official_accw4_top10'))} | "
            f"{fmt(row.get('gaic_primary'))} | {fmt(row.get('equal4_raw_mean'))} | {fmt(row.get('equal4_zscore'))} | "
            f"{fmt(row.get('worst_dataset_gap'))} | {fmt(row.get('head_sanity_score'))} | {fmt(row.get('direct_alignment_score'))} | "
            f"{fmt(row.get('route_balanced_accuracy'))} | {fmt(row.get('direct_test_subject_box_iou_to_teacher'))} | "
            f"{fmt(row.get('direct_test_subject_box_valid_negative_acc'))} | {fmt(row.get('direct_test_final_positive_hit_iou_0_5'))} | "
            f"{fmt(row.get('direct_test_final_risk_hit_iou_0_5'))} | {fmt(row.get('latency_mean_ms'), 3)} | {', '.join(gates)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(args.shortlist_manifest_json)
    latency_payload = _read_json(args.latency_json)
    latency_index = _load_latency_index(latency_payload)
    latency_profile_index = _load_latency_profile_index(latency_payload)
    rows = [_load_row(entry, latency_index, latency_profile_index) for entry in manifest.get("entries", [])]
    winners = _decorate(rows)
    completed_entries = sum(
        1
        for row in rows
        if row.get("public_summary_exists") and row.get("head_summary_exists") and row.get("direct_summary_exists")
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "entry_count": len(rows),
        "completed_entry_count": completed_entries,
        "winners": winners,
        "rows": rows,
    }
    _write_json(args.output_dir / "final_deployment_leaderboard.json", payload)
    _write_markdown(args.output_dir / "final_deployment_leaderboard.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
