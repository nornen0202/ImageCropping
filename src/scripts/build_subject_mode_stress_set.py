#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def tags_to_lower(tags: Any) -> List[str]:
    if not isinstance(tags, list):
        return []
    out: List[str] = []
    for t in tags:
        s = str(t).strip().lower()
        if s:
            out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stress set buckets from teacher_scores jsonl")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_bucket", type=int, default=120)
    return p.parse_args()


def bucket_of(rec: Dict[str, Any]) -> str:
    route = rec.get("route_global", {}) if isinstance(rec.get("route_global"), dict) else {}
    mode = str(route.get("subject_mode", "other_ambiguous"))
    flags = route.get("flags", {}) if isinstance(route.get("flags"), dict) else {}
    subject_set = route.get("subject_set", {}) if isinstance(route.get("subject_set"), dict) else {}
    tags = tags_to_lower(rec.get("tags"))
    has_human = bool(route.get("has_human_evidence", False))
    num_people = int(safe_float(route.get("num_people", 0), 0.0))
    conflict = bool(route.get("subject_mode_conflict", False))
    multi = bool(subject_set.get("multi_subject", False))

    if conflict:
        return "E_mode_conflict"
    if mode in {"scene_general", "scene_landscape"}:
        if any("night" in t or "dark" in t for t in tags):
            return "A2_scene_night"
        if any("interior" in t or "indoor" in t or "architecture" in t for t in tags):
            return "A3_scene_interior"
        return "A1_scene_general"
    if mode == "background_texture_copyspace" or bool(flags.get("has_copy_space", False)):
        return "B_copyspace_background"
    if mode == "text_document":
        return "C_text_document"
    if mode == "object_multi" or (multi and num_people <= 0):
        return "D_object_multi_nonhuman"
    if mode.startswith("portrait") and (not has_human):
        return "F_portrait_without_human"
    return "Z_other"


def pick_preview(rec: Dict[str, Any]) -> Dict[str, Any]:
    by_ar = rec.get("teacher_scorer", {}).get("results_by_ar", {})
    if not isinstance(by_ar, dict):
        by_ar = {}
    ar = "1:1" if "1:1" in by_ar else (next(iter(by_ar.keys())) if by_ar else "")
    ar_res = by_ar.get(ar, {}) if isinstance(by_ar.get(ar), dict) else {}
    top1 = None
    topk = ar_res.get("selected_topk", [])
    if isinstance(topk, list) and topk:
        top1 = topk[0] if isinstance(topk[0], dict) else None
    decision = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}
    return {
        "target_ar": ar,
        "decision_type": str(decision.get("decision_type", "")),
        "delta_improve": safe_float(decision.get("delta_improve", 0.0), 0.0),
        "top1_candidate_id": str((top1 or {}).get("candidate_id", "")),
        "top1_source": str((top1 or {}).get("source", "")),
        "top1_final": safe_float(((top1 or {}).get("scores", {}) or {}).get("final", 0.0), 0.0),
    }


def main() -> None:
    args = parse_args()
    in_path = Path(args.teacher_scores_jsonl)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv) if args.output_csv else None
    if not in_path.exists():
        raise FileNotFoundError(f"teacher_scores_jsonl not found: {in_path}")

    random.seed(int(args.seed))
    max_per_bucket = max(1, int(args.max_per_bucket))
    buckets: Dict[str, List[Dict[str, Any]]] = {}

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if not image_id:
                continue
            b = bucket_of(rec)
            row = {
                "bucket": b,
                "image_id": image_id,
                "subject_mode": str((rec.get("route_global", {}) or {}).get("subject_mode", "")),
                "policy_id": str((rec.get("route_global", {}) or {}).get("policy_id", "")),
                "subject_mode_conflict": bool((rec.get("route_global", {}) or {}).get("subject_mode_conflict", False)),
                "tags_top8": tags_to_lower(rec.get("tags"))[:8],
                **pick_preview(rec),
            }
            buckets.setdefault(b, []).append(row)

    sampled: Dict[str, List[Dict[str, Any]]] = {}
    flat_rows: List[Dict[str, Any]] = []
    for b, rows in sorted(buckets.items()):
        rows = list(rows)
        random.shuffle(rows)
        picked = rows[:max_per_bucket]
        sampled[b] = picked
        flat_rows.extend(picked)

    payload = {
        "schema_version": "subject_mode_stress_set_v1",
        "input_teacher_scores_jsonl": str(in_path),
        "max_per_bucket": max_per_bucket,
        "num_buckets": len(sampled),
        "num_rows": len(flat_rows),
        "bucket_counts": {k: len(v) for k, v in sampled.items()},
        "rows": flat_rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print(f"[done] stress_set_json={out_json}")
    if out_csv is not None:
        print(f"[done] stress_set_csv={out_csv}")
    print(f"[done] bucket_counts={payload['bucket_counts']}")


if __name__ == "__main__":
    main()
