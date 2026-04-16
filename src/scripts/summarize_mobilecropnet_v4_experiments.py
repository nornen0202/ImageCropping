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
    parser = argparse.ArgumentParser(description="Summarize MobileCropNet v4 experiment directories.")
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _last_epoch_metrics(run_dir: Path) -> dict[str, Any]:
    history = _read_json(run_dir / "metrics.json", [])
    if not isinstance(history, list) or not history:
        return {}
    best = max(
        history,
        key=lambda row: float(row.get("val", {}).get("top1_hit", 0.0))
        + 0.25 * float(row.get("val", {}).get("proposal_recall_0_5", 0.0))
        + 0.1 * float(row.get("val", {}).get("decision_acc", 0.0)),
    )
    return {"epoch_count": len(history), "best_epoch": best}


def _run_summary(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json", {})
    train = _last_epoch_metrics(run_dir)
    eval_test = _read_json(run_dir / "eval_test" / "metrics.json", {})
    compare = _read_json(run_dir / "compare_test" / "comparison_metrics.json", {})
    metrics = eval_test.get("metrics", {}) if isinstance(eval_test, dict) else {}
    methods = compare.get("methods", {}) if isinstance(compare, dict) else {}
    best_epoch = train.get("best_epoch", {})
    best_val = best_epoch.get("val", {}) if isinstance(best_epoch, dict) else {}
    best_train = best_epoch.get("train", {}) if isinstance(best_epoch, dict) else {}
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "input_size": config.get("input_size"),
        "candidate_k": config.get("candidate_k"),
        "proposal_q": config.get("proposal_q"),
        "width_mult": config.get("width_mult"),
        "token_dim": config.get("token_dim"),
        "epoch_count": train.get("epoch_count", 0),
        "best_epoch": best_epoch.get("epoch") if isinstance(best_epoch, dict) else None,
        "val_top1_hit": best_val.get("top1_hit"),
        "val_proposal_recall_0_5": best_val.get("proposal_recall_0_5"),
        "test_candidate_top1_hit": metrics.get("candidate_top1_hit"),
        "test_ndcg_at_5": metrics.get("ndcg_at_5"),
        "test_srcc": metrics.get("srcc"),
        "test_top1_iou_to_best_positive": metrics.get("top1_iou_to_best_positive"),
        "test_proposal_recall_at_5_iou_0_5": metrics.get("proposal_recall_at_5_iou_0_5"),
        "compare_v4_utility_regret": methods.get("v4_model", {}).get("utility_regret") if isinstance(methods, dict) else None,
        "compare_baseline_utility_regret": methods.get("baseline_candidate", {}).get("utility_regret") if isinstance(methods, dict) else None,
        "cuda_peak_allocated_mb": best_train.get("cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": best_train.get("cuda_peak_reserved_mb"),
        "nvidia_gpu_util_pct_mean": best_train.get("nvidia_gpu_util_pct_mean"),
        "nvidia_gpu_util_pct_max": best_train.get("nvidia_gpu_util_pct_max"),
        "nvidia_mem_used_mb_mean": best_train.get("nvidia_mem_used_mb_mean"),
        "nvidia_mem_used_mb_max": best_train.get("nvidia_mem_used_mb_max"),
        "nvidia_power_w_mean": best_train.get("nvidia_power_w_mean"),
        "nvidia_power_w_max": best_train.get("nvidia_power_w_max"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_run_summary(path) for path in args.run_dirs]
    rows.sort(
        key=lambda row: (
            float(row.get("test_candidate_top1_hit") or -1.0),
            float(row.get("test_ndcg_at_5") or -1.0),
            float(row.get("test_proposal_recall_at_5_iou_0_5") or -1.0),
        ),
        reverse=True,
    )
    summary = {"run_count": len(rows), "runs": rows}
    (args.output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    headers = [
        "run_name",
        "input_size",
        "proposal_q",
        "width_mult",
        "best_epoch",
        "test_candidate_top1_hit",
        "test_ndcg_at_5",
        "test_srcc",
        "test_proposal_recall_at_5_iou_0_5",
        "cuda_peak_allocated_mb",
        "nvidia_gpu_util_pct_mean",
        "nvidia_gpu_util_pct_max",
        "nvidia_mem_used_mb_max",
    ]
    lines = ["# MobileCropNet v4 Experiment Summary", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header)) for header in headers) + " |")
    (args.output_dir / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
