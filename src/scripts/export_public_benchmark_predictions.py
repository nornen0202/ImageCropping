#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from public_benchmark.prediction_export import export_predictions_from_candidates, export_predictions_from_teacher_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SSTK public benchmark predictions keyed by FCDB/CPC/GNMC task ids.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher_scores_jsonl", default="")
    source.add_argument("--candidates_jsonl", default="")
    parser.add_argument("--keymap_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--score_field", default="")
    parser.add_argument(
        "--candidate_group",
        choices=["selected_topk", "utility_pool", "rank_pool", "cheap_top_m", "best_only"],
        default="utility_pool",
        help="Teacher-score candidate list to export.",
    )
    parser.add_argument("--summary_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.teacher_scores_jsonl:
        summary = export_predictions_from_teacher_scores(
            teacher_scores_jsonl=Path(args.teacher_scores_jsonl).resolve(),
            keymap_jsonl=Path(args.keymap_jsonl).resolve(),
            output_jsonl=Path(args.output_jsonl).resolve(),
            score_field=str(args.score_field or "scores.crop_utility_raw"),
            candidate_group=str(args.candidate_group),
        )
    else:
        summary = export_predictions_from_candidates(
            candidates_jsonl=Path(args.candidates_jsonl).resolve(),
            keymap_jsonl=Path(args.keymap_jsonl).resolve(),
            output_jsonl=Path(args.output_jsonl).resolve(),
            score_field=str(args.score_field or "score"),
        )
    if str(args.summary_json).strip():
        path = Path(args.summary_json).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
