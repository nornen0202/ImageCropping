#!/usr/bin/env python3
"""
Teacher scorer QA report generator.

Reads teacher_scores jsonl and builds global/per-AR QA statistics for routing and
top1 crop quality checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import pandas as pd


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    alpha = pos - lo
    return (1.0 - alpha) * arr[lo] + alpha * arr[hi]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QA report from teacher_scores jsonl")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_by_ar_csv", default="")
    p.add_argument("--negative_score_thr", type=float, default=0.0)
    p.add_argument("--subject_coverage_fail_thr", type=float, default=0.9)
    p.add_argument("--copyspace_keep_thr", type=float, default=0.25)
    return p.parse_args()


def make_bucket() -> Dict[str, Any]:
    return {
        "count": 0,
        "decision": Counter(),
        "final_scores": [],
        "delta_improve": [],
        "neg_final_count": 0,
        "face_cut_count": 0,
        "joint_cut_count": 0,
        "subject_cov_fail_count": 0,
        "subject_cut_risk_count": 0,
        "copyspace_subset_count": 0,
        "copyspace_preserve_count": 0,
        "portrait_route_count": 0,
        "portrait_no_human_count": 0,
        "group_no_human_count": 0,
        "expensive_source": Counter(),
        "fallback_activated_count": 0,
        "fallback_mode": Counter(),
        "norm_size_source": Counter(),
    }


def update_bucket(
    bucket: Dict[str, Any],
    *,
    ar_res: Dict[str, Any],
    route_global: Dict[str, Any],
    negative_score_thr: float,
    subject_coverage_fail_thr: float,
    copyspace_keep_thr: float,
) -> None:
    bucket["count"] += 1

    decision = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}
    decision_type = str(decision.get("decision_type", "unknown"))
    delta = safe_float(decision.get("delta_improve", 0.0))
    bucket["decision"][decision_type] += 1
    bucket["delta_improve"].append(delta)

    topk = ar_res.get("selected_topk", [])
    if not isinstance(topk, list) or not topk:
        return
    top1 = topk[0] if isinstance(topk[0], dict) else {}

    score_final = safe_float(top1.get("scores", {}).get("final", 0.0))
    bucket["final_scores"].append(score_final)
    if score_final < float(negative_score_thr):
        bucket["neg_final_count"] += 1

    flags = top1.get("flags", {}) if isinstance(top1.get("flags"), dict) else {}
    face_cut = bool(flags.get("face_cut", False))
    joint_cut = safe_float(flags.get("joint_cutoff_score", 0.0)) > 0.35
    subj_cov = safe_float(flags.get("subject_coverage", 1.0))
    subj_cov_fail = subj_cov < float(subject_coverage_fail_thr)

    if face_cut:
        bucket["face_cut_count"] += 1
    if joint_cut:
        bucket["joint_cut_count"] += 1
    if subj_cov_fail:
        bucket["subject_cov_fail_count"] += 1
    if face_cut or joint_cut or subj_cov_fail:
        bucket["subject_cut_risk_count"] += 1

    comps = top1.get("scores", {}).get("components", {})
    if isinstance(comps, dict):
        src = str(comps.get("expensive_source", "unknown"))
        bucket["expensive_source"][src] += 1

    routing = ar_res.get("routing", {}) if isinstance(ar_res.get("routing"), dict) else {}
    flags_route = routing.get("flags", {}) if isinstance(routing.get("flags"), dict) else {}
    shot_type = str(routing.get("shot_type", "unknown"))
    norm_size_source = str(
        routing.get(
            "norm_size_source",
            route_global.get("norm_size_source", "unknown"),
        )
    )
    bucket["norm_size_source"][norm_size_source] += 1

    fallback = ar_res.get("fallback", {}) if isinstance(ar_res.get("fallback"), dict) else {}
    if bool(fallback.get("activated", False)):
        bucket["fallback_activated_count"] += 1
        bucket["fallback_mode"][str(fallback.get("mode", "unknown"))] += 1

    has_copyspace = bool(flags_route.get("has_copy_space", False))
    has_human_evidence = bool(route_global.get("has_human_evidence", False))
    if not has_human_evidence:
        has_human_evidence = (
            int(safe_float(route_global.get("num_people", 0), 0.0)) > 0
            or int(safe_float(route_global.get("c2_person_count", 0), 0.0)) > 0
        )

    if has_copyspace:
        bucket["copyspace_subset_count"] += 1
        context = top1.get("composition_checks", {}).get("context", {})
        if isinstance(context, dict):
            subj_area = safe_float(context.get("subject_area", 1.0))
            if (1.0 - subj_area) >= float(copyspace_keep_thr):
                bucket["copyspace_preserve_count"] += 1

    if shot_type in {"headshot", "half", "full", "group"}:
        bucket["portrait_route_count"] += 1
        if not has_human_evidence:
            bucket["portrait_no_human_count"] += 1
    if shot_type == "group" and not has_human_evidence:
        bucket["group_no_human_count"] += 1


def summarize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, int(bucket["count"]))
    finals = [float(v) for v in bucket["final_scores"]]
    deltas = [float(v) for v in bucket["delta_improve"]]

    return {
        "count": int(bucket["count"]),
        "decision_counts": dict(bucket["decision"]),
        "decision_rates": {k: float(v) / n for k, v in bucket["decision"].items()},
        "final_score": {
            "mean": mean(finals) if finals else 0.0,
            "min": min(finals) if finals else 0.0,
            "max": max(finals) if finals else 0.0,
            "p50": percentile(finals, 0.50),
            "p90": percentile(finals, 0.90),
            "neg_rate": float(bucket["neg_final_count"]) / n,
        },
        "delta_improve": {
            "mean": mean(deltas) if deltas else 0.0,
            "p50": percentile(deltas, 0.50),
            "p90": percentile(deltas, 0.90),
        },
        "risk_rates": {
            "face_cut_rate": float(bucket["face_cut_count"]) / n,
            "joint_cut_rate": float(bucket["joint_cut_count"]) / n,
            "subject_coverage_fail_rate": float(bucket["subject_cov_fail_count"]) / n,
            "subject_cut_risk_rate": float(bucket["subject_cut_risk_count"]) / n,
        },
        "routing_consistency": {
            "portrait_route_rate": float(bucket["portrait_route_count"]) / n,
            "portrait_route_without_human_rate": float(bucket["portrait_no_human_count"]) / max(
                1, int(bucket["portrait_route_count"])
            ),
            "group_route_without_human_rate": float(bucket["group_no_human_count"]) / n,
        },
        "copyspace": {
            "subset_count": int(bucket["copyspace_subset_count"]),
            "preserve_rate": float(bucket["copyspace_preserve_count"]) / max(
                1, int(bucket["copyspace_subset_count"])
            ),
        },
        "expensive_source_counts": dict(bucket["expensive_source"]),
        "fallback": {
            "activated_rate": float(bucket["fallback_activated_count"]) / n,
            "mode_counts": dict(bucket["fallback_mode"]),
        },
        "norm_size_source_counts": dict(bucket["norm_size_source"]),
    }


def flatten_for_csv(ar: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "target_ar": ar,
        "count": summary.get("count", 0),
        "keep_full_rate": summary.get("decision_rates", {}).get("keep_full", 0.0),
        "minimal_crop_rate": summary.get("decision_rates", {}).get("minimal_crop", 0.0),
        "crop_rate": summary.get("decision_rates", {}).get("crop", 0.0),
        "final_mean": summary.get("final_score", {}).get("mean", 0.0),
        "final_p50": summary.get("final_score", {}).get("p50", 0.0),
        "final_p90": summary.get("final_score", {}).get("p90", 0.0),
        "final_neg_rate": summary.get("final_score", {}).get("neg_rate", 0.0),
        "delta_mean": summary.get("delta_improve", {}).get("mean", 0.0),
        "face_cut_rate": summary.get("risk_rates", {}).get("face_cut_rate", 0.0),
        "joint_cut_rate": summary.get("risk_rates", {}).get("joint_cut_rate", 0.0),
        "subject_cov_fail_rate": summary.get("risk_rates", {}).get("subject_coverage_fail_rate", 0.0),
        "subject_cut_risk_rate": summary.get("risk_rates", {}).get("subject_cut_risk_rate", 0.0),
        "portrait_route_rate": summary.get("routing_consistency", {}).get("portrait_route_rate", 0.0),
        "portrait_no_human_rate": summary.get("routing_consistency", {}).get(
            "portrait_route_without_human_rate", 0.0
        ),
        "group_no_human_rate": summary.get("routing_consistency", {}).get(
            "group_route_without_human_rate", 0.0
        ),
        "copyspace_subset_count": summary.get("copyspace", {}).get("subset_count", 0),
        "copyspace_preserve_rate": summary.get("copyspace", {}).get("preserve_rate", 0.0),
        "fallback_activated_rate": summary.get("fallback", {}).get("activated_rate", 0.0),
    }
    return out


def main() -> None:
    args = parse_args()

    in_path = Path(args.teacher_scores_jsonl)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_by_ar_csv) if args.output_by_ar_csv else None

    if not in_path.exists():
        raise FileNotFoundError(f"teacher_scores_jsonl not found: {in_path}")

    global_bucket = make_bucket()
    per_ar: Dict[str, Dict[str, Any]] = {}
    image_ids = set()
    total_ar_results = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if image_id:
                image_ids.add(image_id)

            route_global = rec.get("route_global", {}) if isinstance(rec.get("route_global"), dict) else {}
            by_ar = (
                rec.get("teacher_scorer", {}).get("results_by_ar", {})
                if isinstance(rec.get("teacher_scorer"), dict)
                else {}
            )
            if not isinstance(by_ar, dict):
                continue

            for ar_text, ar_res in by_ar.items():
                if not isinstance(ar_res, dict):
                    continue
                total_ar_results += 1
                if ar_text not in per_ar:
                    per_ar[ar_text] = make_bucket()

                update_bucket(
                    global_bucket,
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                )
                update_bucket(
                    per_ar[ar_text],
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                )

    global_summary = summarize_bucket(global_bucket)
    by_ar_summary = {k: summarize_bucket(v) for k, v in sorted(per_ar.items())}

    report = {
        "input_jsonl": str(in_path),
        "num_images": len(image_ids),
        "num_ar_results": int(total_ar_results),
        "config": {
            "negative_score_thr": float(args.negative_score_thr),
            "subject_coverage_fail_thr": float(args.subject_coverage_fail_thr),
            "copyspace_keep_thr": float(args.copyspace_keep_thr),
        },
        "global": global_summary,
        "by_ar": by_ar_summary,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [flatten_for_csv(ar, sm) for ar, sm in sorted(by_ar_summary.items())]
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[done] report_json={out_json}")
    if out_csv is not None:
        print(f"[done] report_by_ar_csv={out_csv}")


if __name__ == "__main__":
    main()
