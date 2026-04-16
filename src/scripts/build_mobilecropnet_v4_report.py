#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a paper-ready MobileCropNet v4 experiment report.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--viz_dir", type=Path, default=None)
    parser.add_argument("--comparison_dir", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--title", default="MobileCropNet v4.0 GAIC Experiment Report")
    return parser


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _metric_table(metrics: dict[str, Any]) -> list[str]:
    lines = ["| metric | value |", "| --- | ---: |"]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)):
            lines.append(f"| {key} | {float(value):.6f} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(args.run_dir / "config.json", {})
    dataset = _read_json(args.run_dir / "dataset_summary.json", {})
    history = _read_json(args.run_dir / "metrics.json", [])
    eval_summary = _read_json(args.eval_dir / "metrics.json", {})
    viz_manifest = _read_json(args.viz_dir / "visualization_manifest.json", {}) if args.viz_dir else {}
    comparison = _read_json(args.comparison_dir / "comparison_metrics.json", {}) if args.comparison_dir else {}
    best_epoch = max(history, key=lambda row: float(row.get("val", {}).get("top1_hit", 0.0)), default={})
    metrics = eval_summary.get("metrics", {})

    lines = [
        f"# {args.title}",
        "",
        "## Executive Summary",
        "",
        "MobileCropNet v4.0 was implemented as a candidate-first, Teacher-topology-agnostic mobile cropper with a learned proposal generator, operator-safe box pooling, RelationLite candidate ranking, and baseline-aware policy prediction. This report records the current GAIC replay-label training/evaluation run and is intended to be extended with public-model comparison rows as those workloads complete.",
        "",
        "## Implementation Scope",
        "",
        "- Data adapter: current `train_conditional_detr_batch.jsonl` schema with `baseline`, `decision_target`, `matching_targets`, and `candidate_pool`.",
        "- Model: shared mobile encoder, AR-conditioned learned proposals, inside/boundary/context mask pooling, RelationLite set ranking, route/decision/delta heads.",
        "- Training objective: utility BCE, listwise KL-style ranking, pairwise logistic ranking, positive/risk heads, macro auxiliary head, policy heads, and dense proposal auxiliary loss.",
        "- Evaluation: candidate top-1 hit, recall@K, SRCC/PCC, NDCG@K, IoU to best positive, and proposal recall@K.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Dataset Summary",
        "",
        "```json",
        json.dumps(dataset, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Best Validation Epoch",
        "",
        "```json",
        json.dumps(best_epoch, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evaluation Metrics",
        "",
        *_metric_table(metrics),
        "",
        "## Qualitative Artifacts",
        "",
    ]
    if viz_manifest:
        lines.extend(
            [
                f"- contact_sheet: `{viz_manifest.get('contact_sheet')}`",
                f"- overlay_dir: `{viz_manifest.get('overlay_dir')}`",
                f"- sample_count: {viz_manifest.get('sample_count')}",
            ]
        )
    else:
        lines.append("- Visualization artifacts were not provided for this report build.")
    if comparison:
        lines.extend(["", "## Method Comparison", "", "| method | top1_hit | exact_best | iou_to_best | utility_regret |", "| --- | ---: | ---: | ---: | ---: |"])
        for method, row in comparison.get("methods", {}).items():
            lines.append(
                "| {method} | {hit:.6f} | {exact:.6f} | {iou:.6f} | {regret:.6f} |".format(
                    method=method,
                    hit=float(row.get("candidate_top1_hit", 0.0)),
                    exact=float(row.get("candidate_top1_exact_best", 0.0)),
                    iou=float(row.get("top1_iou_to_best_positive", 0.0)),
                    regret=float(row.get("utility_regret", 0.0)),
                )
            )
    lines.extend(
        [
            "",
            "## Analysis Notes",
            "",
            "- Candidate replay metrics measure whether v4 learns the current Teacher label topology independent of the Teacher implementation internals.",
            "- Proposal recall measures whether the learned generator can recover positive crop geometry without relying on a static micro-bank.",
            "- Product-level readiness requires adding public cropper and Teacher comparator rows under the same GAIC split, plus latency/export validation on the target device.",
            "",
            "## Next Experiments",
            "",
            "- Increase proposal queries from Q16 to Q24/Q32 and compare proposal recall vs latency.",
            "- Sweep input resolution 256/288/320 and token width 96/128 under a fixed GPU budget.",
            "- Add static-bank and Teacher replay baselines to the same `predictions.jsonl` metric schema for direct comparison.",
            "- Run held-out GAIC test JSONL evaluation after train/val hyperparameter selection.",
        ]
    )
    report_path = args.output_dir / "MobileCropNet_v4_GAIC_Report_KO.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
