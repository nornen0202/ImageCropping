#!/usr/bin/env python3
"""
Rebuild teacher overview json/csv from merged teacher_scores jsonl.

This is used by multi-shard teacher scoring where each shard writes partial
outputs and we merge jsonl first, then rebuild global/per-AR summary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild teacher overview from teacher_scores jsonl")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_by_ar_csv", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))

    from score_teacher import (  # pylint: disable=import-outside-toplevel
        ARStats,
        TeacherScorerConfig,
        box_area,
        parse_ar,
        safe_float,
        summarize_stats,
        update_stats,
    )

    in_path = Path(args.teacher_scores_jsonl)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_by_ar_csv) if str(args.output_by_ar_csv).strip() else None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg = TeacherScorerConfig()
    cfg_loaded = False

    stats_by_ar: Dict[str, ARStats] = {}
    stats_all = ARStats()
    images_written = 0
    expensive_real_applied = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = rec.get("teacher_scorer", {})
            if not isinstance(ts, dict):
                continue

            if not cfg_loaded:
                cfg_in = ts.get("config", {})
                if isinstance(cfg_in, dict):
                    for k, v in cfg_in.items():
                        if hasattr(cfg, k):
                            try:
                                setattr(cfg, k, v)
                            except Exception:
                                pass
                cfg_loaded = True

            if bool(ts.get("expensive_real_applied", False)):
                expensive_real_applied += 1

            route_global = rec.get("route_global", {})
            if not isinstance(route_global, dict):
                route_global = {}
            image_ar = safe_float(rec.get("image_ar", 1.0))

            results_by_ar = ts.get("results_by_ar", {})
            if not isinstance(results_by_ar, dict):
                continue

            for ar_text, ar_res in results_by_ar.items():
                if not isinstance(ar_res, dict):
                    continue
                decision = ar_res.get("decision", {})
                if not isinstance(decision, dict):
                    decision = {}
                decision_type = str(decision.get("decision_type", "crop"))
                delta_improve = safe_float(decision.get("delta_improve", 0.0))

                selected_topk = ar_res.get("selected_topk", [])
                if not isinstance(selected_topk, list):
                    selected_topk = []

                baseline = ar_res.get("baseline_candidate", {})
                if not isinstance(baseline, dict):
                    baseline = {}
                ref_center_area = box_area(baseline.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))

                routing = ar_res.get("routing", {})
                if not isinstance(routing, dict):
                    routing = {}
                route = {
                    "shot_type": str(routing.get("shot_type", route_global.get("shot_type", "unknown"))),
                    "num_people": int(safe_float(route_global.get("num_people", routing.get("num_people", 0)), 0.0)),
                    "flags": routing.get("flags", route_global.get("flags", {})),
                }
                if not isinstance(route["flags"], dict):
                    route["flags"] = {}

                fallback = ar_res.get("fallback", {})
                if not isinstance(fallback, dict):
                    fallback = {}

                if ar_text not in stats_by_ar:
                    stats_by_ar[ar_text] = ARStats()
                s = stats_by_ar[ar_text]

                s.candidates_in_mean += float(safe_float(ar_res.get("num_input_candidates", 0), 0.0))
                s.cheap_kept_mean += float(safe_float(ar_res.get("num_cheap_kept", 0), 0.0))
                update_stats(
                    stats=s,
                    image_ar=image_ar,
                    target_ar=parse_ar(ar_text),
                    decision_type=decision_type,
                    delta_improve=float(delta_improve),
                    topk=selected_topk,
                    ref_center_area=float(ref_center_area),
                    route=route,
                    fallback=fallback,
                    cfg=cfg,
                )

                stats_all.candidates_in_mean += float(safe_float(ar_res.get("num_input_candidates", 0), 0.0))
                stats_all.cheap_kept_mean += float(safe_float(ar_res.get("num_cheap_kept", 0), 0.0))
                update_stats(
                    stats=stats_all,
                    image_ar=image_ar,
                    target_ar=parse_ar(ar_text),
                    decision_type=decision_type,
                    delta_improve=float(delta_improve),
                    topk=selected_topk,
                    ref_center_area=float(ref_center_area),
                    route=route,
                    fallback=fallback,
                    cfg=cfg,
                )

            images_written += 1

    for s in list(stats_by_ar.values()) + [stats_all]:
        denom = max(1, s.images)
        s.candidates_in_mean /= denom
        s.cheap_kept_mean /= denom

    overview = {
        "schema_version": "teacher_scorer_v2_rebuilt_overview",
        "config": asdict(cfg),
        "inputs": {"teacher_scores_jsonl": str(in_path)},
        "outputs": {"overview_json": str(out_json), "overview_by_ar_csv": str(out_csv) if out_csv else None},
        "expensive_stage": {"real_applied_images": int(expensive_real_applied)},
        "summary": {
            "images_written": int(images_written),
            "global": summarize_stats(stats_all),
            "by_ar": {ar: summarize_stats(st) for ar, st in sorted(stats_by_ar.items())},
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    if out_csv is not None:
        rows: List[Dict[str, Any]] = []
        for ar, st in sorted(stats_by_ar.items()):
            rec = summarize_stats(st)
            flat: Dict[str, Any] = {"target_ar": ar}
            for k, v in rec.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, dict):
                            continue
                        flat[f"{k}.{kk}"] = vv
                else:
                    flat[k] = v
            rows.append(flat)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[done] rebuilt overview: {out_json}")
    if out_csv is not None:
        print(f"[done] rebuilt overview csv: {out_csv}")


if __name__ == "__main__":
    main()

