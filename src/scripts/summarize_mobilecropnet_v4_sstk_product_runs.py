#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MobileCropNet v4 SSTK-only product experiment runs.")
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--baseline_json", type=Path, default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _method(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    methods = summary.get("gaic_official_metrics", {}).get("methods", {})
    for name, row in methods.items():
        if str(name).startswith(prefix):
            return row
    return {}


def _teacher_method(summary: dict[str, Any]) -> dict[str, Any]:
    methods = summary.get("gaic_official_metrics", {}).get("methods", {})
    for name, row in methods.items():
        if str(name).startswith("sstk_teacher"):
            return row
    for name, row in methods.items():
        method_name = str(name)
        if "teacher" in method_name and "__teacher_overlap" not in method_name:
            return row
    return {}


def _last_metric(history: list[dict[str, Any]], section: str, key: str) -> float:
    rows = [row for row in history if isinstance(row.get(section), dict)]
    if not rows:
        return 0.0
    return _finite(rows[-1][section].get(key))


def _best_metric(history: list[dict[str, Any]], section: str, key: str) -> float:
    rows = [row for row in history if isinstance(row.get(section), dict)]
    if not rows:
        return 0.0
    return max(_finite(row[section].get(key)) for row in rows)


def _extract_run(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "run_summary.json")
    if not summary:
        summary = {
            "run_name": run_dir.name,
            "output_dir": str(run_dir),
            "train_config": _read_json(run_dir / "config.json"),
            "dataset_summary": _read_json(run_dir / "dataset_summary.json"),
            "train_history": _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else [],
            "replay_test_metrics": _read_json(run_dir / "eval_test" / "metrics.json"),
            "comparison_metrics": _read_json(run_dir / "compare_test" / "comparison_metrics.json"),
            "gaic_official_metrics": _read_json(run_dir / "gaic_official_test" / "metrics.json"),
        }
    train_config = summary.get("train_config") or {}
    history = summary.get("train_history") or []
    best_row = max(history, key=lambda row: _finite(row.get("selection_score"), -1e9), default={})
    replay = (summary.get("replay_test_metrics") or {}).get("metrics", {})
    compare = (summary.get("comparison_metrics") or {}).get("methods", {}).get("v4_model", {})
    model_method = _method(summary, "mcn_v4")
    model_metrics = model_method.get("metrics", {}) if isinstance(model_method, dict) else {}
    teacher = _teacher_method(summary)
    teacher_metrics = teacher.get("metrics", {}) if isinstance(teacher, dict) else {}
    return {
        "run_name": summary.get("run_name") or run_dir.name,
        "profile": summary.get("profile") or train_config.get("model_profile_effective") or train_config.get("model_profile"),
        "output_dir": str(run_dir),
        "status": "complete" if (run_dir / "run_summary.json").exists() else "partial",
        "best_epoch": best_row.get("epoch", summary.get("best_epoch")),
        "best_selection_score": _finite(best_row.get("selection_score", summary.get("best_selection_score"))),
        "batch_size": train_config.get("batch_size"),
        "input_size": train_config.get("input_size"),
        "candidate_k": train_config.get("candidate_k"),
        "proposal_q": train_config.get("proposal_q"),
        "backbone": train_config.get("backbone_name"),
        "top_return_weight": train_config.get("top_return_weight"),
        "selection_metric": train_config.get("selection_metric"),
        "val_top_return_hit_best": _best_metric(history, "val", "top_return_hit"),
        "val_top1_hit_best": _best_metric(history, "val", "top1_hit"),
        "gpu_util_train_last": _last_metric(history, "train", "nvidia_gpu_util_pct_mean"),
        "gpu_util_train_max_last": _last_metric(history, "train", "nvidia_gpu_util_pct_max"),
        "gpu_mem_train_max_last": _last_metric(history, "train", "nvidia_mem_used_mb_max"),
        "replay_top1_hit": _finite(replay.get("candidate_top1_hit")),
        "replay_exact_best": _finite(replay.get("candidate_top1_exact_best")),
        "replay_ndcg5": _finite(replay.get("ndcg_at_5")),
        "replay_srcc": _finite(replay.get("srcc")),
        "replay_pcc": _finite(replay.get("pcc")),
        "compare_chosen_label_score": _finite(compare.get("chosen_label_score")),
        "compare_utility_regret": _finite(compare.get("utility_regret")),
        "official_pcc": _finite(model_metrics.get("pcc")),
        "official_srcc": _finite(model_metrics.get("srcc")),
        "official_acc1_top5": _finite(model_metrics.get("acc1_of_top5")),
        "official_acc1_top10": _finite(model_metrics.get("acc1_of_top10")),
        "official_acc4_top10": _finite(model_metrics.get("acc4_of_top10")),
        "official_accw4_top10": _finite(model_metrics.get("accw4_of_top10")),
        "official_top1_mos": _finite(model_metrics.get("top1_mos")),
        "official_regret": _finite(model_metrics.get("top1_mos_regret")),
        "teacher_pcc": _finite(teacher_metrics.get("pcc")),
        "teacher_srcc": _finite(teacher_metrics.get("srcc")),
        "teacher_acc1_top5": _finite(teacher_metrics.get("acc1_of_top5")),
        "teacher_acc1_top10": _finite(teacher_metrics.get("acc1_of_top10")),
        "teacher_top1_mos": _finite(teacher_metrics.get("top1_mos")),
        "teacher_regret": _finite(teacher_metrics.get("top1_mos_regret")),
    }


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        ("run", "run_name"),
        ("profile", "profile"),
        ("best_ep", "best_epoch"),
        ("sel", "best_selection_score"),
        ("val_topret", "val_top_return_hit_best"),
        ("gpu%", "gpu_util_train_last"),
        ("PCC", "official_pcc"),
        ("SRCC", "official_srcc"),
        ("Acc1/5", "official_acc1_top5"),
        ("Acc1/10", "official_acc1_top10"),
        ("Acc4/10", "official_acc4_top10"),
        ("Accw4/10", "official_accw4_top10"),
        ("top1 MOS", "official_top1_mos"),
        ("regret", "official_regret"),
        ("replay_top1", "replay_top1_hit"),
        ("utility_regret", "compare_utility_regret"),
    ]
    lines = [
        "# MobileCropNet v4 SSTK-Only Product Experiment Summary",
        "",
        "| " + " | ".join(title for title, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        values = []
        for _, key in cols:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value if value is not None else ""))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_extract_run(path) for path in args.run_dirs]
    rows.sort(key=lambda row: (_finite(row.get("official_top1_mos")), -_finite(row.get("official_regret"))), reverse=True)
    payload = {"run_count": len(rows), "runs": rows}
    (args.output_dir / "sstk_product_topreturn_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(args.output_dir / "sstk_product_topreturn_summary.md", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
