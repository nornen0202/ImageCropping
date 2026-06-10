#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from progress_utils import ProgressTracker, progress_log
except ModuleNotFoundError:
    from scripts.progress_utils import ProgressTracker, progress_log


CANDIDATE_KEEP_KEYS = {
    "candidate_id",
    "source",
    "bbox_norm_xyxy",
    "area_ratio",
    "ar",
    "scores",
    "macro_scores",
    "macro_components",
    "macro_masks",
    "checklist",
    "checklist_labels",
    "composition_checks",
    "flags",
    "hard_reject",
    "hard_reject_strict",
    "must_keep",
    "reject_tags",
    "why_tags",
    "why_tags_numeric",
    "why_text_template",
    "policy",
    "teacher_derived",
}

AR_RESULT_KEEP_KEYS = {
    "target_ar",
    "target_ar_value",
    "is_freeform",
    "decision",
    "keep_policy",
    "routing",
    "proposal_injection",
    "num_input_candidates",
    "num_valid_candidates",
    "num_effective_candidates",
    "num_cheap_kept",
    "num_expensive_eval",
    "fallback",
}

TOP_LEVEL_KEEP_KEYS = {
    "image_id",
    "width",
    "height",
    "image_ar",
    "tags",
    "meta_norm",
    "route_global",
    "subject_prior",
    "candidate_meta",
    "proposal_injected",
    "pipeline_debug",
}

BUCKET_KEYS = (
    "cheap_top_m",
    "selected_topk",
    "hard_negatives",
    "also_considered_rejected",
)
SINGLE_CANDIDATE_KEYS = (
    "best_candidate",
    "baseline_candidate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact teacher JSONL for CPU-heavy downstream label/report generation."
    )
    parser.add_argument("--teacher_scores_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument(
        "--max_cheap_top_m",
        type=int,
        default=0,
        help="Optional cap per AR for cheap_top_m. 0 preserves all rows after field compaction.",
    )
    parser.add_argument(
        "--max_also_considered",
        type=int,
        default=0,
        help="Optional cap per AR for also_considered_rejected. 0 preserves all rows after field compaction.",
    )
    parser.add_argument("--progress", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=250)
    parser.add_argument("--progress_min_seconds", type=float, default=10.0)
    return parser.parse_args()


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def compact_dict(value: Dict[str, Any], keep_keys: Iterable[str]) -> Dict[str, Any]:
    keep = set(keep_keys)
    return {key: value[key] for key in keep if key in value}


def compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return compact_dict(safe_dict(candidate), CANDIDATE_KEEP_KEYS)


def limit_bucket(rows: List[Any], limit: int) -> List[Any]:
    if limit > 0:
        return rows[:limit]
    return rows


def compact_ar_result(
    ar_result: Dict[str, Any],
    *,
    max_cheap_top_m: int,
    max_also_considered: int,
    stats: Counter[str],
) -> Dict[str, Any]:
    out = compact_dict(safe_dict(ar_result), AR_RESULT_KEEP_KEYS)
    for key in BUCKET_KEYS:
        raw_rows = [safe_dict(row) for row in safe_list(ar_result.get(key)) if isinstance(row, dict)]
        if key == "cheap_top_m":
            raw_rows = limit_bucket(raw_rows, max_cheap_top_m)
        elif key == "also_considered_rejected":
            raw_rows = limit_bucket(raw_rows, max_also_considered)
        out[key] = [compact_candidate(row) for row in raw_rows]
        stats[f"candidate_rows.{key}"] += len(out[key])
    for key in SINGLE_CANDIDATE_KEYS:
        cand = safe_dict(ar_result.get(key))
        if cand:
            out[key] = compact_candidate(cand)
            stats[f"candidate_rows.{key}"] += 1
    return out


def compact_teacher_record(
    record: Dict[str, Any],
    *,
    max_cheap_top_m: int,
    max_also_considered: int,
    stats: Counter[str],
) -> Dict[str, Any]:
    out = compact_dict(record, TOP_LEVEL_KEEP_KEYS)
    teacher_scorer = safe_dict(record.get("teacher_scorer"))
    compact_scorer: Dict[str, Any] = {}
    for key in ("config", "expensive_real_applied"):
        if key in teacher_scorer:
            compact_scorer[key] = teacher_scorer[key]
    results_by_ar = safe_dict(teacher_scorer.get("results_by_ar"))
    compact_results: Dict[str, Any] = {}
    for target_ar, ar_result in results_by_ar.items():
        compact_results[str(target_ar)] = compact_ar_result(
            safe_dict(ar_result),
            max_cheap_top_m=max_cheap_top_m,
            max_also_considered=max_also_considered,
            stats=stats,
        )
        stats["ar_tasks"] += 1
    compact_scorer["results_by_ar"] = compact_results
    out["teacher_scorer"] = compact_scorer
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.teacher_scores_jsonl)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if str(args.summary_json).strip() else output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    progress_enabled = bool(int(args.progress))
    tracker = ProgressTracker(
        f"build_teacher_downstream_compact:{input_path.name}",
        unit="rows",
        every=max(1, int(args.progress_every)),
        min_seconds=max(0.0, float(args.progress_min_seconds)),
        enabled=progress_enabled,
    )
    progress_log(
        f"build_teacher_downstream_compact: start | input={input_path} | output={output_path}",
        enabled=progress_enabled,
    )

    stats: Counter[str] = Counter()
    output_tmp = output_path.with_name(output_path.name + ".tmp")
    with input_path.open("r", encoding="utf-8") as src, output_tmp.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            compact = compact_teacher_record(
                record,
                max_cheap_top_m=max(0, int(args.max_cheap_top_m)),
                max_also_considered=max(0, int(args.max_also_considered)),
                stats=stats,
            )
            dst.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["rows"] += 1
            tracker.update(stats["rows"], extra=f"line={line_idx}")
    output_tmp.replace(output_path)
    tracker.finish(stats["rows"], extra=f"ar_tasks={stats['ar_tasks']}")

    input_size = input_path.stat().st_size if input_path.exists() else 0
    output_size = output_path.stat().st_size if output_path.exists() else 0
    summary = {
        "status": "ok",
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "row_count": int(stats["rows"]),
        "ar_task_count": int(stats["ar_tasks"]),
        "candidate_rows": {
            key.split(".", 1)[1]: int(value)
            for key, value in stats.items()
            if key.startswith("candidate_rows.")
        },
        "size_bytes": {
            "input": int(input_size),
            "output": int(output_size),
            "ratio": round(float(output_size) / float(input_size), 6) if input_size else None,
        },
        "policy": {
            "max_cheap_top_m": max(0, int(args.max_cheap_top_m)),
            "max_also_considered": max(0, int(args.max_also_considered)),
            "candidate_keep_keys": sorted(CANDIDATE_KEEP_KEYS),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
