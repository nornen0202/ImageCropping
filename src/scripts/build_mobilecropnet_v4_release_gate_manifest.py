#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


HARD_GATE_KEYS = (
    "crop_top_set",
    "worst_dataset_gate",
    "head_gate",
    "direct_gate",
    "hard_gate_target_ar",
    "hard_gate_risk",
    "hard_gate_prop_target",
    "hard_gate_route_collapse",
    "hard_gate_route_balance",
    "hard_gate_subject_box",
)

CATASTROPHIC_BUCKETS = {
    "route_mismatch",
    "decision_mismatch",
    "low_subject_iou",
    "low_top1_iou",
    "low_proposal_recall",
    "checklist_disagreement",
    "model_rationale_warning",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 release gate manifest를 생성한다.")
    parser.add_argument("--leaderboard_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--qualitative_pack_json", type=Path)
    parser.add_argument(
        "--route_smoke_root",
        type=Path,
        default=Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424/route_smokes"),
        help="release gate 차단 요인 근거로 사용할 선택적 route-only/internal-prior smoke artifact.",
    )
    parser.add_argument(
        "--head_audit_root",
        type=Path,
        default=Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424/head_audits"),
        help="route/subject/policy/proposal head audit artifact root.",
    )
    parser.add_argument(
        "--internal_composite_root",
        type=Path,
        default=Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424/internal_composite_smokes"),
        help="internal subject-box -> crop composite no-prior smoke artifact root.",
    )
    parser.add_argument("--inference_script", default="src/scripts/infer_mobilecropnet_v4.py")
    return parser


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _f(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _slug(text: Any) -> str:
    out = []
    for ch in str(text or "").lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            1 if row.get("final_gate_pass") else 0,
            _f(row.get("equal4_zscore")) or -1e9,
            _f(row.get("head_sanity_score")) or -1e9,
            _f(row.get("direct_alignment_score")) or -1e9,
            -(_f(row.get("latency_mean_ms")) or 1e9),
        ),
        reverse=True,
    )


def _candidate_id(row: dict[str, Any]) -> str:
    return f"{_slug(row.get('track'))}__{_slug(row.get('profile'))}"


def _failed_gates(row: dict[str, Any]) -> list[str]:
    return [key for key in HARD_GATE_KEYS if not bool(row.get(key))]


def _qualitative_summary(path: Path | None, candidate: dict[str, Any] | None) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"path": str(path) if path else None, "exists": False, "gate_pass": None}
    counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    candidate_key = _candidate_id(candidate) if candidate else None
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        buckets = [str(bucket) for bucket in sample.get("failure_buckets") or []]
        for bucket in buckets:
            if bucket in CATASTROPHIC_BUCKETS:
                counts[bucket] = counts.get(bucket, 0) + 1
                if candidate_key and sample.get("candidate_id") == candidate_key:
                    candidate_counts[bucket] = candidate_counts.get(bucket, 0) + 1
    candidate_catastrophic_total = sum(candidate_counts.values())
    return {
        "path": str(path),
        "exists": True,
        "sample_count": payload.get("sample_count"),
        "candidate_count": payload.get("candidate_count"),
        "catastrophic_bucket_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "candidate_id": candidate_key,
        "candidate_catastrophic_bucket_counts": dict(sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))),
        "candidate_catastrophic_total": candidate_catastrophic_total,
        "gate_pass": bool(candidate is not None and candidate_catastrophic_total < 3),
    }


def _gate_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": row.get("track"),
        "variant": row.get("variant"),
        "profile": row.get("profile"),
        "run_name": row.get("run_name"),
        "run_dir": row.get("run_dir"),
        "checkpoint": row.get("checkpoint"),
        "config": str(Path(str(row.get("run_dir") or "")) / "config.json") if row.get("run_dir") else None,
        "eval_output_dir": row.get("eval_output_dir"),
        "final_gate_pass": bool(row.get("final_gate_pass")),
        "failed_gates": _failed_gates(row),
        "metrics": {
            "equal4_zscore": row.get("equal4_zscore"),
            "equal4_raw_mean": row.get("equal4_raw_mean"),
            "worst_dataset_gap": row.get("worst_dataset_gap"),
            "official_top1_mos": row.get("official_top1_mos"),
            "official_srcc": row.get("official_srcc"),
            "official_accw4_top10": row.get("official_accw4_top10"),
            "head_sanity_score": row.get("head_sanity_score"),
            "route_balanced_accuracy": row.get("route_balanced_accuracy"),
            "route_collapse_flag": row.get("route_collapse_flag"),
            "decision_accuracy": row.get("decision_accuracy"),
            "why_tag_micro_f1": row.get("why_tag_micro_f1"),
            "checklist_safety_recall_mean": row.get("checklist_safety_recall_mean"),
            "direct_alignment_score": row.get("direct_alignment_score"),
            "direct_test_final_positive_hit_iou_0_5": row.get("direct_test_final_positive_hit_iou_0_5"),
            "direct_test_final_best_iou_to_positive": row.get("direct_test_final_best_iou_to_positive"),
            "direct_test_final_risk_hit_iou_0_5": row.get("direct_test_final_risk_hit_iou_0_5"),
            "direct_test_final_target_ar_compatible": row.get("direct_test_final_target_ar_compatible"),
            "direct_test_subject_box_iou_to_teacher": row.get("direct_test_subject_box_iou_to_teacher"),
            "direct_test_subject_box_pred_conf": row.get("direct_test_subject_box_pred_conf"),
            "direct_test_subject_box_valid_acc": row.get("direct_test_subject_box_valid_acc"),
            "direct_test_subject_box_valid_negative_acc": row.get("direct_test_subject_box_valid_negative_acc"),
            "direct_test_subject_box_valid_positive_acc": row.get("direct_test_subject_box_valid_positive_acc"),
            "latency_mean_ms": row.get("latency_mean_ms"),
            "latency_p90_ms": row.get("latency_p90_ms"),
        },
    }


def _collect_route_smokes(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return {"root": str(root), "exists": False, "rows": rows, "best": None}
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        if not isinstance(metrics, list):
            continue
        best_row = None
        best_route = None
        for row in metrics:
            if not isinstance(row, dict):
                continue
            val = row.get("val") if isinstance(row.get("val"), dict) else {}
            route = _f(val.get("route_balanced_acc"))
            if route is None:
                route = _f(val.get("route_acc"))
            if route is None:
                continue
            if best_route is None or route > best_route:
                best_route = route
                best_row = row
        status = _read_json(run_dir / "train_status.json") or _read_json(run_dir / "summary.json") or {}
        config = _read_json(run_dir / "config.json") or {}
        if best_row is None:
            continue
        best_val = best_row.get("val") if isinstance(best_row.get("val"), dict) else {}
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "state": status.get("state"),
                "profile": config.get("model_profile_effective") or config.get("model_profile"),
                "selection_metric": best_row.get("selection_metric") or status.get("selection_metric"),
                "best_epoch": best_row.get("epoch"),
                "best_route_acc": best_route,
                "best_route_raw_acc": best_val.get("route_acc"),
                "best_route_balanced_acc": best_val.get("route_balanced_acc"),
                "best_subject_iou": best_val.get("subject_box_iou"),
                "best_subject_valid_balanced_acc": best_val.get("subject_box_valid_balanced_acc"),
                "best_top1_hit": best_val.get("top1_hit"),
                "best_generated_align_top1_hit": best_val.get("generated_align_top1_hit"),
            }
        )
    rows.sort(key=lambda row: _f(row.get("best_route_acc")) or -1e9, reverse=True)
    return {"root": str(root), "exists": True, "rows": rows, "best": rows[0] if rows else None}


def _collect_head_audits(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return {"root": str(root), "exists": False, "rows": rows, "best": None}
    for summary_path in sorted(root.glob("*/release_head_audit_summary.json")):
        run_dir = summary_path.parent
        payload = _read_json(summary_path)
        if not isinstance(payload, dict):
            continue
        route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
        subject = payload.get("subject_box") if isinstance(payload.get("subject_box"), dict) else {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        gate = payload.get("release_gate") if isinstance(payload.get("release_gate"), dict) else {}
        failures = payload.get("failure_taxonomy") if isinstance(payload.get("failure_taxonomy"), dict) else {}
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "image_count": payload.get("image_count"),
                "checkpoint": payload.get("checkpoint"),
                "route_accuracy": route.get("accuracy"),
                "route_balanced_accuracy": route.get("balanced_accuracy"),
                "subject_valid_iou_mean": subject.get("valid_iou_mean"),
                "subject_valid_accuracy": subject.get("valid_accuracy"),
                "top1_hit": metrics.get("candidate_top1_hit"),
                "top1_iou_to_best_positive": metrics.get("top1_iou_to_best_positive"),
                "proposal_target_ar_recall_at_5_iou_0_5": metrics.get("proposal_target_ar_recall_at_5_iou_0_5"),
                "proposal_top1_target_ar_compatible": metrics.get("proposal_top1_target_ar_compatible"),
                "checklist_agreement": metrics.get("explain_label_agreement_when_applicable"),
                "deploy_candidate": bool(gate.get("deploy_candidate")),
                "failure_taxonomy": failures,
            }
        )
    rows.sort(
        key=lambda row: (
            1 if row.get("deploy_candidate") else 0,
            _f(row.get("route_balanced_accuracy")) or -1e9,
            _f(row.get("subject_valid_iou_mean")) or -1e9,
        ),
        reverse=True,
    )
    return {"root": str(root), "exists": True, "rows": rows, "best": rows[0] if rows else None}


def _collect_internal_composite_smokes(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return {"root": str(root), "exists": False, "rows": rows, "best": None}
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        payload = _read_json(metrics_path)
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "state": payload.get("state"),
                "image_count": payload.get("image_count"),
                "subject_checkpoint": payload.get("subject_checkpoint"),
                "crop_checkpoint": payload.get("crop_checkpoint"),
                "route_balanced_accuracy": metrics.get("route_balanced_acc"),
                "route_accuracy": metrics.get("route_correct"),
                "subject_box_iou": metrics.get("subject_box_iou"),
                "subject_box_valid_acc": metrics.get("subject_box_valid_acc"),
                "top1_hit": metrics.get("top1_hit"),
                "top1_iou_to_positive": metrics.get("top1_iou_to_positive"),
            }
        )
    rows.sort(key=lambda row: _f(row.get("route_balanced_accuracy")) or -1e9, reverse=True)
    return {"root": str(root), "exists": True, "rows": rows, "best": rows[0] if rows else None}


def _recommended_recovery_plan(
    route_smokes: dict[str, Any],
    head_audits: dict[str, Any],
    deploy_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if deploy_row is not None:
        return [
            {
                "phase": "final_gate_recheck",
                "action": "선정된 checkpoint/config를 동결하고 태깅 전에 새 qualitative pack으로 full release gate를 재평가한다.",
                "exit_criteria": "deploy 후보가 final_gate_pass=true이고 qualitative catastrophic total < 3을 만족한다.",
            }
        ]
    best_smoke = route_smokes.get("best") if isinstance(route_smokes.get("best"), dict) else {}
    best_audit = head_audits.get("best") if isinstance(head_audits.get("best"), dict) else {}
    best_route = _f(best_audit.get("route_balanced_accuracy"))
    if best_route is None:
        best_route = _f(best_smoke.get("best_route_acc"))
    return [
        {
            "phase": "route_label_and_taxonomy_audit",
            "action": (
                "2026-04-27 병렬 route smoke가 gate를 통과하지 못했으므로, 다음 학습 전에 route class별 confusion, "
                "target-AR별 route 분포, subject-valid/IoU bucket별 route 오류, teacher taxonomy noise를 먼저 분리한다."
            ),
            "exit_criteria": "route balanced accuracy 저하가 label taxonomy 문제인지 모델 capacity/calibration 문제인지 class별로 분리된다.",
            "current_best_route_acc": best_route,
        },
        {
            "phase": "subject_valid_calibration_patch",
            "action": "subject confidence bin별 target-positive rate와 IoU를 기준으로 valid threshold/loss weight를 재보정한다.",
            "exit_criteria": "subject valid accuracy와 positive recall이 동시에 개선되고 valid subject IoU >= 0.50을 안정적으로 넘는다.",
        },
        {
            "phase": "bounded_retrain_after_audit",
            "action": "위 audit에서 확인된 label/calibration 수정만 반영해 1-GPU smoke를 병렬 재실행한다.",
            "exit_criteria": "held-out validation에서 route balanced accuracy >= 0.52 및 subject IoU >= 0.50을 동시에 만족한다.",
        },
        {
            "phase": "full_rerun_gate",
            "action": "bounded retrain이 통과한 경우에만 full direct/product, equal-4, GAIC official, latency, qualitative gate 평가를 실행한다.",
            "exit_criteria": "단일 checkpoint/config가 crop, route, subject-box, action/policy, target-AR, latency, qualitative gate를 모두 통과한다.",
        },
    ]


def _inference_command(row: dict[str, Any], inference_script: str) -> list[str]:
    return [
        "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python",
        inference_script,
        "--checkpoint",
        str(row.get("checkpoint")),
        "--image",
        "<image_path>",
        "--target_ar",
        "<FREE|1_1|4_3|3_4|16_9|9_16>",
        "--selection_policy",
        "proposal_topk_rerank",
        "--proposal_top_m",
        "8",
        "--exact_target_ar_postprocess",
        "--output_json",
        "artifacts/mobilecropnet_v4/release_inference/sample_output.json",
        "--output_png",
        "artifacts/mobilecropnet_v4/release_inference/sample_output.png",
    ]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    candidate = payload.get("deploy_candidate") or {}
    lines = [
        "# MobileCropNet v4 릴리스 게이트 판정",
        "",
        f"- 생성 시각: `{payload.get('generated_at')}`",
        f"- 리더보드 JSON: `{payload.get('leaderboard_json')}`",
        f"- 상태: `{decision.get('state')}`",
        f"- 사유: {decision.get('reason')}",
        "",
        "## 배포 후보",
        "",
    ]
    if candidate:
        lines.extend(
            [
                f"- 트랙/프로파일: `{candidate.get('track')}` / `{candidate.get('profile')}`",
                f"- checkpoint: `{candidate.get('checkpoint')}`",
                f"- config: `{candidate.get('config')}`",
                f"- 평가 산출물: `{candidate.get('eval_output_dir')}`",
                f"- 추론 명령: `{' '.join(payload.get('inference_command') or [])}`",
            ]
        )
    else:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 상위 Gate 행",
            "",
            "| 순위 | 트랙 | 프로파일 | 최종 | 실패 gate | equal4_z | route_bal | subj_iou | subj_neg | hit | risk |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(payload.get("top_gate_rows", []), start=1):
        metrics = row.get("metrics") or {}
        lines.append(
            f"| {idx} | {row.get('track')} | {row.get('profile')} | {row.get('final_gate_pass')} | "
            f"{', '.join(row.get('failed_gates') or []) or '-'} | {_fmt(metrics.get('equal4_zscore'))} | "
            f"{_fmt(metrics.get('route_balanced_accuracy'))} | {_fmt(metrics.get('direct_test_subject_box_iou_to_teacher'))} | "
            f"{_fmt(metrics.get('direct_test_subject_box_valid_negative_acc'))} | "
            f"{_fmt(metrics.get('direct_test_final_positive_hit_iou_0_5'))} | {_fmt(metrics.get('direct_test_final_risk_hit_iou_0_5'))} |"
        )
    qual = payload.get("qualitative") or {}
    route_smokes = payload.get("route_smokes") or {}
    lines.extend(
        [
            "",
            "## 정성 Gate",
            "",
            f"- pack: `{qual.get('path')}`",
            f"- gate 통과: `{qual.get('gate_pass')}`",
            f"- catastrophic bucket count: `{qual.get('catastrophic_bucket_counts')}`",
            "",
            "## 차단 요인",
            "",
        ]
    )
    for blocker in decision.get("blockers") or []:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## 최신 Route Smoke 근거",
            "",
            f"- root: `{route_smokes.get('root')}`",
            "",
            "| run | 상태 | best epoch | route bal | subj IoU | valid bal | top1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (route_smokes.get("rows") or [])[:8]:
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('state')} | {row.get('best_epoch')} | "
            f"{_fmt(row.get('best_route_acc'))} | {_fmt(row.get('best_subject_iou'))} | "
            f"{_fmt(row.get('best_subject_valid_balanced_acc'))} | {_fmt(row.get('best_top1_hit'))} |"
        )
    head_audits = payload.get("head_audits") or {}
    lines.extend(
        [
            "",
            "## 릴리스 헤드 감사 근거",
            "",
            f"- root: `{head_audits.get('root')}`",
            "",
            "| audit | images | route raw | route bal | subj IoU | valid acc | top1 | top1 IoU | 배포 후보 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in head_audits.get("rows") or []:
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('image_count')} | {_fmt(row.get('route_accuracy'))} | "
            f"{_fmt(row.get('route_balanced_accuracy'))} | {_fmt(row.get('subject_valid_iou_mean'))} | "
            f"{_fmt(row.get('subject_valid_accuracy'))} | {_fmt(row.get('top1_hit'))} | "
            f"{_fmt(row.get('top1_iou_to_best_positive'))} | {row.get('deploy_candidate')} |"
        )
    composites = payload.get("internal_composites") or {}
    lines.extend(
        [
            "",
            "## 내부 Composite No-Prior 근거",
            "",
            f"- root: `{composites.get('root')}`",
            "",
            "| run | state | images | route bal | route raw | subj IoU | valid acc | top1 | top1 IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in composites.get("rows") or []:
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('state')} | {row.get('image_count')} | "
            f"{_fmt(row.get('route_balanced_accuracy'))} | {_fmt(row.get('route_accuracy'))} | "
            f"{_fmt(row.get('subject_box_iou'))} | {_fmt(row.get('subject_box_valid_acc'))} | "
            f"{_fmt(row.get('top1_hit'))} | {_fmt(row.get('top1_iou_to_positive'))} |"
        )
    lines.extend(["", "## 권장 회복 계획", ""])
    for step in payload.get("recommended_recovery_plan") or []:
        lines.extend(
            [
                f"### {step.get('phase')}",
                "",
                f"- 실행: {step.get('action')}",
                f"- 통과 기준: {step.get('exit_criteria')}",
            ]
        )
        if step.get("current_best_route_acc") is not None:
            lines.append(f"- 현재 best route acc: `{_fmt(step.get('current_best_route_acc'))}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = _read_json(args.leaderboard_json)
    if not isinstance(leaderboard, dict):
        raise SystemExit(f"leaderboard not found or invalid: {args.leaderboard_json}")
    rows = [row for row in leaderboard.get("rows", []) if isinstance(row, dict)]
    ranked = _rank_rows(rows)
    passing = [row for row in ranked if row.get("final_gate_pass")]
    deploy_row = passing[0] if passing else None
    qualitative = _qualitative_summary(args.qualitative_pack_json, deploy_row)
    route_smokes = _collect_route_smokes(args.route_smoke_root)
    head_audits = _collect_head_audits(args.head_audit_root)
    internal_composites = _collect_internal_composite_smokes(args.internal_composite_root)
    if deploy_row is not None and qualitative.get("gate_pass") is False:
        deploy_row = None
        passing = []

    blockers: list[str] = []
    if deploy_row is None:
        blockers.append("현재 strict product release gate를 통과한 row가 없다.")
        for row in ranked[:5]:
            failed = _failed_gates(row)
            if failed:
                blockers.append(f"{row.get('track')} / {row.get('profile')}: 실패 gate = {', '.join(failed)}")
        best_smoke = route_smokes.get("best") if isinstance(route_smokes.get("best"), dict) else None
        best_smoke_route = _f((best_smoke or {}).get("best_route_acc"))
        if best_smoke_route is not None and best_smoke_route < 0.50:
            blockers.append(
                "최신 route smoke 증거가 0.50 balanced accuracy 미만이다 "
                f"({(best_smoke or {}).get('run_name')}: {_fmt(best_smoke_route)})."
            )
        best_audit = head_audits.get("best") if isinstance(head_audits.get("best"), dict) else None
        best_audit_route = _f((best_audit or {}).get("route_balanced_accuracy"))
        best_audit_subject = _f((best_audit or {}).get("subject_valid_iou_mean"))
        if best_audit_route is not None and best_audit_route < 0.52:
            blockers.append(
                "release-head audit의 실제 route balanced accuracy가 0.52 gate 미만이다 "
                f"({(best_audit or {}).get('run_name')}: {_fmt(best_audit_route)})."
            )
        if best_audit_subject is not None and best_audit_subject < 0.50:
            blockers.append(
                "release-head audit의 subject valid IoU가 0.50 gate 미만이다 "
                f"({(best_audit or {}).get('run_name')}: {_fmt(best_audit_subject)})."
            )
        best_composite = internal_composites.get("best") if isinstance(internal_composites.get("best"), dict) else None
        best_comp_route = _f((best_composite or {}).get("route_balanced_accuracy"))
        if best_comp_route is not None and best_comp_route < 0.52:
            blockers.append(
                "internal subject-bbox composite no-prior smoke가 route gate를 크게 미달해 배포 후보에서 제외됐다 "
                f"({(best_composite or {}).get('run_name')}: route_bal={_fmt(best_comp_route)})."
            )
        if qualitative.get("exists") and qualitative.get("catastrophic_bucket_counts"):
            blockers.append("qualitative review pack에 catastrophic bucket이 남아 있다.")
    elif qualitative.get("gate_pass") is False:
        blockers.append("정량 상위 후보가 반복적인 catastrophic qualitative failure로 차단됐다.")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "leaderboard_json": str(args.leaderboard_json),
        "entry_count": leaderboard.get("entry_count"),
        "completed_entry_count": leaderboard.get("completed_entry_count"),
        "final_gate_pass_count": len(passing),
        "decision": {
            "state": "PASS" if deploy_row is not None else "BLOCKED",
            "reason": "strict product release gate 통과" if deploy_row is not None else "strict product release gate 미충족",
            "blockers": blockers,
        },
        "deploy_candidate": _gate_snapshot(deploy_row) if deploy_row is not None else None,
        "inference_command": _inference_command(deploy_row, args.inference_script) if deploy_row is not None else None,
        "top_gate_rows": [_gate_snapshot(row) for row in ranked[:8]],
        "qualitative": qualitative,
        "route_smokes": route_smokes,
        "head_audits": head_audits,
        "internal_composites": internal_composites,
        "recommended_recovery_plan": _recommended_recovery_plan(route_smokes, head_audits, deploy_row),
        "release_gate_thresholds": (leaderboard.get("winners") or {}).get("release_gate_thresholds"),
    }
    _write_json(args.output_dir / "release_gate_manifest.json", payload)
    _write_markdown(args.output_dir / "RELEASE_GATE_DECISION.md", payload)
    print(json.dumps({"state": payload["decision"]["state"], "final_gate_pass_count": payload["final_gate_pass_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
