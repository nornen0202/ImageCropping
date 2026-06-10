#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a subject-box-head report for MobileCropNet v4 runs.")
    parser.add_argument("--manifest_tsv", nargs="*", type=Path, default=[])
    parser.add_argument("--run_dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    parsed = _f(value)
    return "-" if parsed is None else f"{parsed:.{digits}f}"


def _load_manifest_run_dirs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                run_dir = Path(row.get("out_base", "")) / str(row.get("run_name", ""))
                if run_dir in seen:
                    continue
                seen.add(run_dir)
                rows.append(
                    {
                        "run_dir": run_dir,
                        "run_name": row.get("run_name"),
                        "profile": row.get("profile"),
                        "variant": row.get("variant"),
                        "track_name": row.get("track_name"),
                        "run_id": row.get("run_id"),
                    }
                )
    return rows


def _last_history_row(history: Any) -> dict[str, Any]:
    if isinstance(history, list) and history:
        last = history[-1]
        return last if isinstance(last, dict) else {}
    return {}


def _best_history_row(history: Any) -> dict[str, Any]:
    if not isinstance(history, list):
        return {}
    rows = [row for row in history if isinstance(row, dict)]
    if not rows:
        return {}
    return max(rows, key=lambda row: _f(row.get("selection_score")) if _f(row.get("selection_score")) is not None else float("-inf"))


def _metrics(path: Path, key: str = "metrics") -> dict[str, Any]:
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get(key), dict):
        return dict(payload[key])
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _direct_metrics(run_dir: Path, name: str) -> dict[str, Any]:
    return _metrics(run_dir / name / "metrics.json")


def _calibration(run_dir: Path, name: str) -> dict[str, Any]:
    payload = _read_json(run_dir / name / "subject_box_calibration" / "subject_box_confidence_calibration.json")
    return payload if isinstance(payload, dict) else {}


def _infer_track_name(run_name: str, run_dir: Path) -> str | None:
    text = f"{run_name} {run_dir}".lower()
    if "gaic" in text and "uctr" in text and ("v2" in text or "balanced-valid" in text or "balanced_valid" in text):
        return "GAIC UCTR subjectprior v2 balanced-valid"
    if "sstk" in text and "uctr" in text and ("v2" in text or "balanced-valid" in text or "balanced_valid" in text):
        return "SSTK UCTR subjectprior v2 balanced-valid"
    if "gaic" in text and "uctr" in text:
        return "GAIC UCTR subjectprior"
    if "sstk" in text and "uctr" in text:
        return "SSTK UCTR subjectprior"
    return None


def _infer_variant(run_name: str, run_dir: Path) -> str | None:
    text = f"{run_name} {run_dir}".lower()
    if "v2" in text or "balanced-valid" in text or "balanced_valid" in text:
        return "subject_box_prior_v2_balanced_valid"
    if "subjbox" in text or "subject_box" in text:
        return "subject_box_prior_v1"
    return None


def _infer_profile(run_name: str, config: dict[str, Any]) -> str | None:
    profile = config.get("model_profile_effective") or config.get("model_profile")
    if profile:
        return str(profile)
    if "plus-384" in run_name or "plus_384" in run_name:
        return "plus_384"
    if "rank-320" in run_name or "rank_320" in run_name:
        return "rank_320"
    return None


def _row_from_run(meta: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(meta["run_dir"])
    history = _read_json(run_dir / "metrics.json")
    run_summary = _read_json(run_dir / "run_summary.json")
    if not isinstance(history, list) and isinstance(run_summary, dict):
        history = run_summary.get("train_history")
    config = _read_json(run_dir / "config.json") or {}
    dataset_summary = _read_json(run_dir / "dataset_summary.json") or {}
    last_row = _last_history_row(history)
    best_row = _best_history_row(history)
    best_val = best_row.get("val", {}) if isinstance(best_row.get("val"), dict) else {}
    last_val = last_row.get("val", {}) if isinstance(last_row.get("val"), dict) else {}
    eval_test = _metrics(run_dir / "eval_test" / "metrics.json")
    direct_test = _direct_metrics(run_dir, "direct_test_proposal_topk_rerank")
    direct_val = _direct_metrics(run_dir, "direct_val_proposal_topk_rerank")
    direct_utility = _direct_metrics(run_dir, "direct_test_utility_top1")
    direct_test_subject_best = _direct_metrics(run_dir, "direct_test_proposal_topk_rerank_subject_box_best")
    post_eval_summary = _read_json(run_dir / "subject_box_post_eval_summary.json")
    direct_test_cal = _calibration(run_dir, "direct_test_proposal_topk_rerank")
    direct_test_cal_best = direct_test_cal.get("selected") if isinstance(direct_test_cal.get("selected"), dict) else {}
    direct_test_cal_target_neg = (
        direct_test_cal.get("selected_target_negative_acc")
        if isinstance(direct_test_cal.get("selected_target_negative_acc"), dict)
        else (
            direct_test_cal.get("target_negative_acc")
            if isinstance(direct_test_cal.get("target_negative_acc"), dict)
            else {}
        )
    )
    direct_test_subject_best_cal = _calibration(run_dir, "direct_test_proposal_topk_rerank_subject_box_best")
    direct_test_subject_best_cal_best = (
        direct_test_subject_best_cal.get("selected") if isinstance(direct_test_subject_best_cal.get("selected"), dict) else {}
    )
    viz_manifest = _read_json(run_dir / "viz_test" / "visualization_manifest.json")
    direct_viz_manifest = _read_json(run_dir / "direct_viz_test_proposal_topk_rerank" / "visualization_manifest.json")
    run_name = str(meta.get("run_name") or run_dir.name)
    if run_summary:
        artifact_status = "complete"
    elif direct_test or post_eval_summary:
        artifact_status = "partial_post_eval"
    elif isinstance(history, list):
        artifact_status = "training"
    elif run_dir.exists():
        artifact_status = "partial"
    else:
        artifact_status = "missing"
    viz_contact_sheet = None
    if isinstance(direct_viz_manifest, dict):
        viz_contact_sheet = str(run_dir / "direct_viz_test_proposal_topk_rerank" / "contact_sheet.png")
    elif isinstance(viz_manifest, dict):
        viz_contact_sheet = str(run_dir / "viz_test" / "contact_sheet.png")
    return {
        "track_name": meta.get("track_name") or _infer_track_name(run_name, run_dir),
        "variant": meta.get("variant") or config.get("variant") or _infer_variant(run_name, run_dir),
        "profile": meta.get("profile") or _infer_profile(run_name, config),
        "run_id": meta.get("run_id"),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "artifact_status": artifact_status,
        "epoch_count": len(history) if isinstance(history, list) else 0,
        "best_epoch": (run_summary or {}).get("best_epoch", best_row.get("epoch")) if isinstance(run_summary, dict) else best_row.get("epoch"),
        "best_selection_score": (run_summary or {}).get("best_selection_score", best_row.get("selection_score")) if isinstance(run_summary, dict) else best_row.get("selection_score"),
        "input_size": config.get("input_size"),
        "candidate_k": config.get("candidate_k"),
        "subject_box_head": config.get("subject_box_head"),
        "subject_box_coord_space": config.get("subject_box_coord_space"),
        "subject_box_refine_with_proposals": config.get("subject_box_refine_with_proposals"),
        "use_pred_subject_box_as_prior": config.get("use_pred_subject_box_as_prior"),
        "subject_box_best_checkpoint_exists": (run_dir / "subject_box_best.pt").exists(),
        "train_rows": ((dataset_summary.get("train") or {}).get("row_count") if isinstance(dataset_summary.get("train"), dict) else None),
        "val_rows": ((dataset_summary.get("val") or {}).get("row_count") if isinstance(dataset_summary.get("val"), dict) else None),
        "best_val_subject_box_iou": best_val.get("subject_box_iou"),
        "best_val_subject_box_valid_acc": best_val.get("subject_box_valid_acc"),
        "best_val_subject_box_valid_precision": best_val.get("subject_box_valid_precision"),
        "best_val_subject_box_valid_recall": best_val.get("subject_box_valid_recall"),
        "last_val_subject_box_iou": last_val.get("subject_box_iou"),
        "last_val_subject_box_valid_acc": last_val.get("subject_box_valid_acc"),
        "replay_subject_box_iou_to_teacher": eval_test.get("subject_box_iou_to_teacher"),
        "replay_subject_box_valid_acc": eval_test.get("subject_box_valid_acc"),
        "replay_subject_box_valid_positive_acc": eval_test.get("subject_box_valid_positive_acc"),
        "replay_subject_box_valid_negative_acc": eval_test.get("subject_box_valid_negative_acc"),
        "replay_subject_box_pred_conf": eval_test.get("subject_box_pred_conf"),
        "replay_candidate_top1_hit": eval_test.get("candidate_top1_hit"),
        "replay_proposal_recall_at_5_iou_0_5": eval_test.get("proposal_recall_at_5_iou_0_5"),
        "direct_val_subject_box_iou_to_teacher": direct_val.get("subject_box_iou_to_teacher"),
        "direct_test_subject_box_iou_to_teacher": direct_test.get("subject_box_iou_to_teacher"),
        "direct_test_subject_box_valid_acc": direct_test.get("subject_box_valid_acc"),
        "direct_test_subject_box_valid_positive_acc": direct_test.get("subject_box_valid_positive_acc"),
        "direct_test_subject_box_valid_negative_acc": direct_test.get("subject_box_valid_negative_acc"),
        "direct_test_final_positive_hit_iou_0_5": direct_test.get("final_positive_hit_iou_0_5"),
        "direct_test_final_best_iou_to_positive": direct_test.get("final_best_iou_to_positive"),
        "direct_test_proposal_recall_at_5_iou_0_5": direct_test.get("proposal_positive_recall_at_5_iou_0_5"),
        "direct_test_subject_box_cal_best_threshold": direct_test_cal_best.get("threshold"),
        "direct_test_subject_box_cal_best_balanced_acc": direct_test_cal_best.get("balanced_acc"),
        "direct_test_subject_box_cal_best_negative_acc": direct_test_cal_best.get("negative_acc"),
        "direct_test_subject_box_cal_best_positive_recall": direct_test_cal_best.get("positive_recall"),
        "direct_test_subject_box_cal_target_neg_threshold": direct_test_cal_target_neg.get("threshold"),
        "direct_test_subject_box_cal_target_neg_balanced_acc": direct_test_cal_target_neg.get("balanced_acc"),
        "direct_test_subject_box_cal_target_neg_negative_acc": direct_test_cal_target_neg.get("negative_acc"),
        "direct_test_subject_box_cal_target_neg_positive_recall": direct_test_cal_target_neg.get("positive_recall"),
        "subject_best_direct_test_subject_box_iou_to_teacher": direct_test_subject_best.get("subject_box_iou_to_teacher"),
        "subject_best_direct_test_subject_box_valid_negative_acc": direct_test_subject_best.get("subject_box_valid_negative_acc"),
        "subject_best_direct_test_final_positive_hit_iou_0_5": direct_test_subject_best.get("final_positive_hit_iou_0_5"),
        "subject_best_direct_test_cal_best_threshold": direct_test_subject_best_cal_best.get("threshold"),
        "subject_best_direct_test_cal_best_balanced_acc": direct_test_subject_best_cal_best.get("balanced_acc"),
        "direct_utility_final_positive_hit_iou_0_5": direct_utility.get("final_positive_hit_iou_0_5"),
        "viz_contact_sheet": viz_contact_sheet,
        "has_eval_test": bool(eval_test),
        "has_direct_test": bool(direct_test),
        "has_viz": isinstance(viz_manifest, dict) or isinstance(direct_viz_manifest, dict),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# MobileCropNet v4 Subject Box Head Report",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- run_count: `{len(payload.get('runs', []))}`",
        "",
        "## Summary",
        "",
        "| track | profile | status | epochs | best sel | val subj IoU | val valid acc | replay subj IoU | replay neg acc | direct subj IoU | direct neg acc | direct hit@0.5 | viz |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("runs", []):
        viz = row.get("viz_contact_sheet") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("track_name") or ""),
                    str(row.get("profile") or ""),
                    str(row.get("artifact_status") or ""),
                    str(row.get("epoch_count") or 0),
                    _fmt(row.get("best_selection_score")),
                    _fmt(row.get("best_val_subject_box_iou")),
                    _fmt(row.get("best_val_subject_box_valid_acc")),
                    _fmt(row.get("replay_subject_box_iou_to_teacher")),
                    _fmt(row.get("replay_subject_box_valid_negative_acc")),
                    _fmt(row.get("direct_test_subject_box_iou_to_teacher")),
                    _fmt(row.get("direct_test_subject_box_valid_negative_acc")),
                    _fmt(row.get("direct_test_final_positive_hit_iou_0_5")),
                    f"`{viz}`" if viz else "-",
                ]
            )
            + " |"
        )
    subject_best_rows = [row for row in payload.get("runs", []) if _f(row.get("subject_best_direct_test_subject_box_iou_to_teacher")) is not None]
    if subject_best_rows:
        lines.extend(
            [
                "",
                "## Crop-Best vs Subject-Best",
                "",
                "| track | profile | crop-best subj IoU | subject-best subj IoU | crop-best neg acc | subject-best neg acc | crop-best hit@0.5 | subject-best hit@0.5 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subject_best_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("track_name") or ""),
                        str(row.get("profile") or ""),
                        _fmt(row.get("direct_test_subject_box_iou_to_teacher")),
                        _fmt(row.get("subject_best_direct_test_subject_box_iou_to_teacher")),
                        _fmt(row.get("direct_test_subject_box_valid_negative_acc")),
                        _fmt(row.get("subject_best_direct_test_subject_box_valid_negative_acc")),
                        _fmt(row.get("direct_test_final_positive_hit_iou_0_5")),
                        _fmt(row.get("subject_best_direct_test_final_positive_hit_iou_0_5")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Confidence Calibration",
            "",
            "| track | profile | best thr | best bal acc | best neg acc | best pos recall | target-neg thr | target-neg bal acc |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("runs", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("track_name") or ""),
                    str(row.get("profile") or ""),
                    _fmt(row.get("direct_test_subject_box_cal_best_threshold"), 3),
                    _fmt(row.get("direct_test_subject_box_cal_best_balanced_acc")),
                    _fmt(row.get("direct_test_subject_box_cal_best_negative_acc")),
                    _fmt(row.get("direct_test_subject_box_cal_best_positive_recall")),
                    _fmt(row.get("direct_test_subject_box_cal_target_neg_threshold"), 3),
                    _fmt(row.get("direct_test_subject_box_cal_target_neg_balanced_acc")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `val subj IoU`는 training validation에서 teacher subject box valid sample에 대한 bbox IoU 평균이다.",
            "- `valid acc`는 subject가 없는 scene/copyspace/background-like sample에서 confidence를 낮추는지까지 포함한 binary validity 정확도다.",
            "- `replay/direct neg acc`는 target subject가 없는 sample만 대상으로 predicted valid confidence가 0.5 미만인지 평가한다.",
            "- `replay subj IoU`와 `direct subj IoU`는 inference output의 `subject_box.predicted`를 teacher box와 비교한 값이다.",
            "- `Confidence Calibration`은 `subject_box.predicted.confidence`를 threshold sweep하여 teacher의 subject-valid label과 비교한 post-processing 가능성 평가다.",
            "- `viz` contact sheet는 selected crop, teacher crop, model-predicted subject bbox, teacher subject bbox를 함께 확인하는 정성 리뷰 entry다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metas = _load_manifest_run_dirs(args.manifest_tsv)
    seen = {Path(row["run_dir"]) for row in metas}
    for run_dir in args.run_dirs:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        metas.append({"run_dir": run_dir})
    rows = [_row_from_run(meta) for meta in metas]
    rows.sort(key=lambda row: (str(row.get("track_name") or ""), str(row.get("profile") or ""), str(row.get("run_name") or "")))
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_tsv": [str(path) for path in args.manifest_tsv],
        "runs": rows,
    }
    (args.output_dir / "subject_box_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output_dir / "subject_box_report.md", payload)
    print(json.dumps({"run_count": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
