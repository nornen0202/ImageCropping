#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


VARIANT_SPECS: dict[str, list[str]] = {
    "baseline_current": [],
    "subjectprior_route_proposal_v1": [
        "--use_subject_prior",
        "--route_use_subject_prior",
        "--proposal_use_subject_prior",
        "--route_balanced_ce",
        "--proposal_subject_weight",
        "0.25",
    ],
    "subjectprior_route_recover_v1": [
        "--use_subject_prior",
        "--route_use_subject_prior",
        "--route_use_candidate_context",
        "--route_use_subject_box_features",
        "--proposal_use_subject_prior",
        "--route_balanced_ce",
        "--route_weight",
        "0.14",
        "--route_object_single_margin_weight",
        "0.08",
        "--route_object_single_margin",
        "0.18",
        "--proposal_subject_weight",
        "0.25",
    ],
    "subjectprior_route_recover_v2": [
        "--use_subject_prior",
        "--route_use_subject_prior",
        "--route_use_candidate_context",
        "--proposal_use_subject_prior",
        "--route_balanced_ce",
        "--train_route_balanced_sampler",
        "--route_weight",
        "0.16",
        "--route_object_single_margin_weight",
        "0.08",
        "--route_object_single_margin",
        "0.18",
        "--proposal_subject_weight",
        "0.25",
    ],
    "policy_score_v1": [
        "--policy_score_head",
        "--policy_score_weight",
        "0.25",
        "--decision_weight",
        "0.10",
        "--detail_score_weight",
        "0.08",
        "--checklist_class_weight",
        "0.04",
    ],
    "hybrid_subject_policy_v1": [
        "--use_subject_prior",
        "--route_use_subject_prior",
        "--policy_use_subject_prior",
        "--proposal_use_subject_prior",
        "--route_balanced_ce",
        "--proposal_subject_weight",
        "0.25",
        "--policy_score_head",
        "--policy_score_weight",
        "0.25",
        "--decision_weight",
        "0.08",
        "--detail_score_weight",
        "0.08",
        "--checklist_class_weight",
        "0.04",
        "--why_tag_weight",
        "0.02",
    ],
    "deploy_no_prior_executor_v1": [
        "--selection_metric",
        "deploy_align_topreturn",
        "--route_balanced_ce",
        "--policy_score_head",
        "--policy_score_weight",
        "0.15",
        "--decision_weight",
        "0.18",
        "--action_consistency_weight",
        "0.25",
        "--action_consistency_margin",
        "0.15",
        "--generated_proposal_align_weight",
        "0.15",
        "--generated_proposal_score_weight",
        "1.0",
        "--generated_proposal_listwise_weight",
        "0.5",
        "--generated_proposal_positive_weight",
        "0.25",
        "--generated_proposal_risk_weight",
        "0.15",
        "--detail_score_weight",
        "0.08",
        "--checklist_class_weight",
        "0.04",
        "--why_tag_weight",
        "0.02",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MobileCropNet v4 head-improvement smoke variants and compare metrics.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--listwise_jsonl", type=Path, default=None)
    parser.add_argument("--val_pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--val_listwise_jsonl", type=Path, default=None)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model_profile", default="balanced_288")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline_current",
            "subjectprior_route_proposal_v1",
            "subjectprior_route_recover_v1",
            "subjectprior_route_recover_v2",
            "policy_score_v1",
            "hybrid_subject_policy_v1",
        ],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_rows", type=int, default=1024)
    parser.add_argument("--max_val_rows", type=int, default=256)
    parser.add_argument("--limit_train_steps", type=int, default=48)
    parser.add_argument("--limit_val_steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--extra_train_args", nargs="*", default=[])
    return parser


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _variant_score(row: dict[str, Any]) -> float:
    eval_metrics = row.get("eval_metrics", {})
    head = row.get("head_metrics", {})
    checklist = head.get("checklist", {})
    headroom_recall = _safe_float(((checklist.get("headroom") or {}).get("applicability") or {}).get("recall"))
    lookroom_recall = _safe_float(((checklist.get("lookroom") or {}).get("applicability") or {}).get("recall"))
    return (
        0.26 * _safe_float(eval_metrics.get("candidate_top1_hit"))
        + 0.16 * _safe_float(eval_metrics.get("proposal_recall_at_5_iou_0_5"))
        + 0.16 * _safe_float((head.get("route") or {}).get("accuracy"))
        + 0.12 * _safe_float((head.get("proposal") or {}).get("proposal_subject_hit_iou_0_5"))
        + 0.10 * _safe_float((head.get("proposal") or {}).get("proposal_to_subject_iou_mean"))
        + 0.08 * _safe_float(((head.get("why_tags") or {}).get("micro") or {}).get("f1"))
        + 0.06 * headroom_recall
        + 0.06 * lookroom_recall
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python_bin = sys.executable
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for variant in args.variants:
        if variant not in VARIANT_SPECS:
            raise ValueError(f"unsupported variant: {variant}")
        run_dir = args.output_root / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        train_dir = run_dir / "train"
        eval_dir = run_dir / "eval"
        head_dir = run_dir / "head_analysis"
        train_cmd = [
            python_bin,
            "src/scripts/train_mobilecropnet_v4.py",
            "--train_jsonl",
            str(args.train_jsonl),
            "--val_jsonl",
            str(args.val_jsonl),
            "--project_root",
            str(args.project_root),
            "--output_dir",
            str(train_dir),
            "--model_profile",
            str(args.model_profile),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--max_train_rows",
            str(args.max_train_rows),
            "--max_val_rows",
            str(args.max_val_rows),
            "--limit_train_steps",
            str(args.limit_train_steps),
            "--limit_val_steps",
            str(args.limit_val_steps),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--precompute_sample_tensors",
            "--selection_metric",
            "deploy_align_topreturn",
            "--top_return_weight",
            "0.6",
            "--listwise_weight",
            "0.7",
            "--pairwise_weight",
            "0.5",
            "--explicit_pairwise_weight",
            "0.25",
            "--top1_risk_weight",
            "0.15",
            "--teacher_distill_weight",
            "0.2",
            "--topk_coverage_weight",
            "0.1",
            "--proposal_weight",
            "0.7",
        ]
        if args.pairwise_jsonl is not None:
            train_cmd += ["--pairwise_jsonl", str(args.pairwise_jsonl)]
        if args.listwise_jsonl is not None:
            train_cmd += ["--listwise_jsonl", str(args.listwise_jsonl)]
        if args.val_pairwise_jsonl is not None:
            train_cmd += ["--val_pairwise_jsonl", str(args.val_pairwise_jsonl)]
        if args.val_listwise_jsonl is not None:
            train_cmd += ["--val_listwise_jsonl", str(args.val_listwise_jsonl)]
        train_cmd += VARIANT_SPECS[variant] + list(args.extra_train_args)
        _run(train_cmd, cwd=args.project_root, log_path=run_dir / "train.log")

        eval_cmd = [
            python_bin,
            "src/scripts/evaluate_mobilecropnet_v4.py",
            "--checkpoint",
            str(train_dir / "best.pt"),
            "--eval_jsonl",
            str(args.eval_jsonl),
            "--project_root",
            str(args.project_root),
            "--output_dir",
            str(eval_dir),
            "--batch_size",
            str(max(1, args.batch_size * 2)),
            "--num_workers",
            str(args.num_workers),
            "--device",
            str(args.device),
        ]
        _run(eval_cmd, cwd=args.project_root, log_path=run_dir / "eval.log")

        analyze_cmd = [
            python_bin,
            "src/scripts/analyze_mobilecropnet_v4_head_predictions.py",
            "--predictions_jsonl",
            str(eval_dir / "predictions.jsonl"),
            "--eval_jsonl",
            str(args.eval_jsonl),
            "--output_dir",
            str(head_dir),
        ]
        _run(analyze_cmd, cwd=args.project_root, log_path=run_dir / "head_analysis.log")

        train_history = _read_json(train_dir / "metrics.json")
        best_row = max(train_history, key=lambda row: float(row.get("selection_score", float("-inf")))) if isinstance(train_history, list) and train_history else {}
        eval_metrics = (_read_json(eval_dir / "metrics.json") or {}).get("metrics", {})
        head_metrics = _read_json(head_dir / "head_analysis_summary.json")
        row = {
            "variant": variant,
            "run_dir": str(run_dir),
            "best_epoch": best_row.get("epoch"),
            "best_selection_score": best_row.get("selection_score"),
            "eval_metrics": eval_metrics,
            "head_metrics": head_metrics,
        }
        row["variant_score"] = _variant_score(row)
        rows.append(row)

    rows.sort(key=lambda item: float(item.get("variant_score", 0.0)), reverse=True)
    summary = {
        "output_root": str(args.output_root),
        "model_profile": str(args.model_profile),
        "variant_count": len(rows),
        "best_variant": rows[0]["variant"] if rows else None,
        "rows": rows,
    }
    (args.output_root / "variant_smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
