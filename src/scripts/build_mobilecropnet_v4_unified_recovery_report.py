#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="durable artifact에서 MobileCropNet v4 통합 회복 보고서를 생성한다.")
    parser.add_argument("--base_leaderboard_json", type=Path, required=True)
    parser.add_argument("--extended_leaderboard_json", type=Path)
    parser.add_argument("--shortlist_manifest_json", type=Path)
    parser.add_argument("--route_recovery_root", type=Path, default=Path("artifacts/mobilecropnet_v4/route_gate_recover_20260424"))
    parser.add_argument("--runtime_no_prior_root", type=Path, default=Path("artifacts/mobilecropnet_v4/runtime_no_prior_rerun_20260424"))
    parser.add_argument("--subject_box_report_json", type=Path, default=Path("artifacts/mobilecropnet_v4/subject_box_head_uctr_20260423/subject_box_report_latest/subject_box_report.json"))
    parser.add_argument("--subject_box_patch_root", type=Path, default=Path("artifacts/mobilecropnet_v4/subject_box_iou_patch_20260424_full"))
    parser.add_argument("--release_gate_smoke_root", type=Path, default=Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424/route_smokes"))
    parser.add_argument("--head_audit_root", type=Path, default=Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424/head_audits"))
    parser.add_argument("--internal_composite_root", type=Path, default=Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424/internal_composite_smokes"))
    parser.add_argument("--qualitative_pack_json", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--doc_path", type=Path)
    return parser


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _configure_plot_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Noto Sans KR", "Malgun Gothic"):
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _f(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _f(value)
    return "-" if number is None else f"{100.0 * number:.2f}%"


def _slug(text: Any) -> str:
    out = []
    for ch in str(text or "").lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _metrics(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return dict(payload["metrics"])
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _leaderboard_summary(path: Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"path": str(path) if path else None, "exists": False}
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    ranked = sorted(
        rows,
        key=lambda row: (
            1 if row.get("final_gate_pass") else 0,
            _f(row.get("equal4_zscore")) or -1e9,
            _f(row.get("head_sanity_score")) or -1e9,
            _f(row.get("direct_alignment_score")) or -1e9,
        ),
        reverse=True,
    )
    return {
        "path": str(path),
        "exists": True,
        "entry_count": payload.get("entry_count"),
        "completed_entry_count": payload.get("completed_entry_count"),
        "route_collapse_count": sum(1 for row in rows if row.get("route_collapse_flag")),
        "final_gate_pass_count": sum(1 for row in rows if row.get("final_gate_pass")),
        "winners": payload.get("winners", {}),
        "top_rows": ranked[:8],
    }


def _route_collapse_from_head(head_summary: dict[str, Any]) -> bool | None:
    route = head_summary.get("route", {}) if isinstance(head_summary.get("route"), dict) else {}
    by_subject = head_summary.get("by_subject_mode", {}) if isinstance(head_summary.get("by_subject_mode"), dict) else {}
    route_accuracy = _f(route.get("accuracy"))
    if route_accuracy is None:
        return None
    flags = []
    for mode in ("portrait_single", "portrait_group"):
        value = _f((by_subject.get(mode) or {}).get("route_acc")) if isinstance(by_subject.get(mode), dict) else None
        if value is not None:
            flags.append(value < max(0.35, route_accuracy - 0.15))
    return any(flags) if flags else None


def _route_balanced_from_head(head_summary: dict[str, Any]) -> float | None:
    by_subject = head_summary.get("by_subject_mode", {}) if isinstance(head_summary.get("by_subject_mode"), dict) else {}
    vals = [
        _f((payload or {}).get("route_acc"))
        for payload in by_subject.values()
        if isinstance(payload, dict) and _f(payload.get("route_acc")) is not None
    ]
    return sum(vals) / len(vals) if vals else None


def _collect_route_recovery(root: Path) -> dict[str, Any]:
    full_runs = []
    for run_summary_path in sorted((root / "full_runs").glob("*/*/run_summary.json")):
        run_dir = run_summary_path.parent
        run_summary = _read_json(run_summary_path) or {}
        status = _read_json(run_dir / "status.json") or {}
        replay = _metrics((run_summary.get("replay_test_metrics") or {}))
        gaic_methods = (run_summary.get("gaic_official_metrics") or {}).get("methods") or {}
        gaic_row = {}
        for name, row in gaic_methods.items():
            if str(name).startswith("mcn_v4"):
                gaic_row = row.get("metrics") or {}
                break
        full_runs.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "status": status.get("state") or status.get("phase"),
                "profile": run_summary.get("profile"),
                "best_epoch": run_summary.get("best_epoch"),
                "best_selection_score": run_summary.get("best_selection_score"),
                "replay_candidate_top1_hit": replay.get("candidate_top1_hit"),
                "replay_proposal_recall_at_5_iou_0_5": replay.get("proposal_recall_at_5_iou_0_5"),
                "gaic_top1_mos": gaic_row.get("top1_mos"),
                "gaic_srcc": gaic_row.get("srcc"),
                "gaic_accw4_top10": gaic_row.get("accw4_of_top10"),
            }
        )

    final_bundles = []
    for summary_path in sorted((root / "final_bundle").glob("*/final_eval_bundle_summary.json")):
        bundle_dir = summary_path.parent
        summary = _read_json(summary_path) or {}
        status = _read_json(bundle_dir / "status.json") or {}
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        head = artifacts.get("head_analysis") if isinstance(artifacts.get("head_analysis"), dict) else {}
        direct = _metrics(artifacts.get("direct_test"))
        public_eval = artifacts.get("public_eval_summary") if isinstance(artifacts.get("public_eval_summary"), dict) else {}
        final_bundles.append(
            {
                "bundle_dir": str(bundle_dir),
                "method_id": summary.get("method_id"),
                "status": status.get("state") or status.get("phase"),
                "route_accuracy": ((head.get("route") or {}).get("accuracy") if isinstance(head.get("route"), dict) else None),
                "route_balanced_accuracy": _route_balanced_from_head(head),
                "route_collapse_flag": _route_collapse_from_head(head),
                "direct_hit_iou_0_5": direct.get("final_positive_hit_iou_0_5"),
                "direct_best_iou": direct.get("final_best_iou_to_positive"),
                "direct_risk_hit_iou_0_5": direct.get("final_risk_hit_iou_0_5"),
                "target_ar_compatible": direct.get("final_target_ar_compatible"),
                "public_macro_score": public_eval.get("macro_avg_score"),
                "public_dataset_count": len(public_eval.get("datasets") or {}) if isinstance(public_eval.get("datasets"), dict) else None,
            }
        )
    return {"root": str(root), "full_runs": full_runs, "final_bundles": final_bundles}


def _collect_route_smokes(root: Path) -> dict[str, Any]:
    rows = []
    if not root.exists():
        return {"root": str(root), "exists": False, "rows": rows, "best": None}
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        if not isinstance(metrics, list):
            continue
        best_row = None
        best_route = None
        for item in metrics:
            if not isinstance(item, dict):
                continue
            val = item.get("val") if isinstance(item.get("val"), dict) else {}
            route_acc = _f(val.get("route_balanced_acc"))
            if route_acc is None:
                route_acc = _f(val.get("route_acc"))
            if route_acc is None:
                continue
            if best_route is None or route_acc > best_route:
                best_route = route_acc
                best_row = item
        if best_row is None:
            continue
        status = _read_json(run_dir / "train_status.json") or _read_json(run_dir / "summary.json") or {}
        config = _read_json(run_dir / "config.json") or {}
        val = best_row.get("val") if isinstance(best_row.get("val"), dict) else {}
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "state": status.get("state"),
                "profile": config.get("model_profile_effective") or config.get("model_profile"),
                "selection_metric": best_row.get("selection_metric") or status.get("selection_metric"),
                "best_epoch": best_row.get("epoch"),
                "best_route_acc": best_route,
                "best_route_raw_acc": val.get("route_acc"),
                "best_route_balanced_acc": val.get("route_balanced_acc"),
                "best_subject_iou": val.get("subject_box_iou"),
                "best_subject_valid_balanced_acc": val.get("subject_box_valid_balanced_acc"),
                "best_top1_hit": val.get("top1_hit"),
                "best_generated_align_top1_hit": val.get("generated_align_top1_hit"),
            }
        )
    rows.sort(key=lambda row: _f(row.get("best_route_acc")) or -1e9, reverse=True)
    return {"root": str(root), "exists": True, "rows": rows, "best": rows[0] if rows else None}


def _collect_head_audits(root: Path) -> dict[str, Any]:
    rows = []
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
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "image_count": payload.get("image_count"),
                "checkpoint": payload.get("checkpoint"),
                "route_accuracy": route.get("accuracy"),
                "route_balanced_accuracy": route.get("balanced_accuracy"),
                "subject_valid_iou_mean": subject.get("valid_iou_mean"),
                "subject_valid_accuracy": subject.get("valid_accuracy"),
                "top1_hit": metrics.get("candidate_top1_hit"),
                "top1_iou_to_best_positive": metrics.get("top1_iou_to_best_positive"),
                "proposal_target_ar_recall_at_5_iou_0_5": metrics.get("proposal_target_ar_recall_at_5_iou_0_5"),
                "checklist_agreement": metrics.get("explain_label_agreement_when_applicable"),
                "deploy_candidate": bool(gate.get("deploy_candidate")),
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


def _collect_internal_composites(root: Path) -> dict[str, Any]:
    rows = []
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
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "state": payload.get("state"),
                "image_count": payload.get("image_count"),
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


def _collect_runtime_no_prior(root: Path) -> dict[str, Any]:
    rows = []
    for summary_path in sorted(root.glob("*/run_summary.json")):
        run_dir = summary_path.parent
        summary = _read_json(summary_path) or {}
        status = _read_json(run_dir / "status.json") or {}
        direct = _metrics(summary.get("direct_test_metrics"))
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": summary.get("run_name") or run_dir.name,
                "profile": summary.get("profile"),
                "variant": run_dir.name.split("_")[-1] if "_" in run_dir.name else "",
                "status": status.get("state") or status.get("phase"),
                "best_epoch": summary.get("best_epoch"),
                "best_selection_score": summary.get("best_selection_score"),
                "direct_hit_iou_0_5": direct.get("final_positive_hit_iou_0_5"),
                "direct_best_iou": direct.get("final_best_iou_to_positive"),
                "direct_risk_hit_iou_0_5": direct.get("final_risk_hit_iou_0_5"),
                "target_ar_compatible": direct.get("final_target_ar_compatible"),
                "route_acc": direct.get("route_acc"),
                "decision_acc": direct.get("decision_acc"),
            }
        )
    matrices = []
    for matrix_path in sorted(root.glob("*/matrix_status.json")):
        payload = _read_json(matrix_path) or {}
        matrices.append(
            {
                "matrix_dir": str(matrix_path.parent),
                "matrix_name": payload.get("matrix_name"),
                "profile": payload.get("profile"),
                "phase": payload.get("phase"),
                "state": payload.get("state"),
                "current_variant": payload.get("current_variant"),
                "current_index": payload.get("current_index"),
                "variant_count": payload.get("variant_count"),
            }
        )
    return {"root": str(root), "runs": rows, "matrices": matrices}


def _subject_row_from_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path) or {}
    run_dir = path.parent
    status = _read_json(run_dir / "status.json") or {}
    post = payload.get("post_eval_best") if isinstance(payload.get("post_eval_best"), dict) else {}
    post_best = payload.get("post_eval_subject_box_best") if isinstance(payload.get("post_eval_subject_box_best"), dict) else {}
    run_summary = payload.get("run_summary") if isinstance(payload.get("run_summary"), dict) else {}
    direct = _metrics(post.get("direct_test"))
    direct_best = _metrics(post_best.get("direct_test"))
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "profile": ((run_summary.get("train_config") or {}).get("model_profile_effective") if isinstance(run_summary.get("train_config"), dict) else None) or status.get("profile"),
        "status": status.get("state") or status.get("phase"),
        "best_epoch": run_summary.get("best_epoch"),
        "best_selection_score": run_summary.get("best_selection_score"),
        "direct_subject_iou": direct.get("subject_box_iou_to_teacher"),
        "direct_negative_acc": direct.get("subject_box_valid_negative_acc"),
        "direct_hit_iou_0_5": direct.get("final_positive_hit_iou_0_5"),
        "subject_best_direct_subject_iou": direct_best.get("subject_box_iou_to_teacher"),
        "subject_best_negative_acc": direct_best.get("subject_box_valid_negative_acc"),
        "subject_best_hit_iou_0_5": direct_best.get("final_positive_hit_iou_0_5"),
    }


def _collect_subject_box(report_json: Path, patch_root: Path) -> dict[str, Any]:
    report = _read_json(report_json)
    report_rows = report.get("runs", []) if isinstance(report, dict) and isinstance(report.get("runs"), list) else []
    patch_rows = [_subject_row_from_summary(path) for path in sorted(patch_root.glob("*/*/subject_box_iou_retrain_summary.json"))]
    running_rows = []
    for status_path in sorted(patch_root.glob("*/*/status.json")):
        run_dir = status_path.parent
        if (run_dir / "subject_box_iou_retrain_summary.json").exists():
            continue
        status = _read_json(status_path) or {}
        running_rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "profile": status.get("profile"),
                "status": status.get("state") or status.get("phase"),
                "phase": status.get("phase"),
            }
        )
    return {
        "report_json": str(report_json),
        "legacy_v2_runs": report_rows,
        "patch_root": str(patch_root),
        "patch_runs": patch_rows,
        "patch_running": running_rows,
    }


def _collect_qualitative(path: Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"path": str(path) if path else None, "exists": False}
    buckets: dict[str, int] = {}
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        for bucket in sample.get("failure_buckets") or []:
            buckets[str(bucket)] = buckets.get(str(bucket), 0) + 1
    return {
        "path": str(path),
        "exists": True,
        "candidate_count": payload.get("candidate_count"),
        "sample_count": payload.get("sample_count"),
        "bucket_counts": dict(sorted(buckets.items(), key=lambda item: (-item[1], item[0]))),
    }


def _build_figure_prompt_set(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": payload["generated_at"],
        "prompts": [
            {
                "id": "runtime_no_prior_architecture",
                "target": "논문 그림",
                "prompt": (
                    "MobileCropNet runtime inference용 기술 구조도를 깔끔하게 그린다. 입력은 image와 target aspect ratio만 표시하고, "
                    "mobile encoder, AR-conditioned proposal generator, runtime baseline lane, top-M reranker, route/policy/risk/explanation heads, "
                    "decision-aware executor, exact target-AR repair, final crop 순서로 연결한다. runtime에는 teacher model과 외부 subject prior가 없음을 명시한다."
                ),
            },
            {
                "id": "route_collapse_recovery_gate",
                "target": "보고서 그림",
                "prompt": (
                    "route collapse 회복 gate를 시각화한다. baseline final leaderboard에서 모든 row가 hard_gate_route_collapse를 실패하는 상태를 먼저 보이고, "
                    "routefix 재학습, head 분석, direct no-prior 평가, public equal-4 평가, 최종 배포 gate 판정 흐름을 표시한다."
                ),
            },
            {
                "id": "subject_box_integration",
                "target": "보고서 그림",
                "prompt": (
                    "teacher subject box, predicted subject box, proposal-conditioned refinement, confidence calibration, direct subject IoU, "
                    "negative accuracy, crop hit@0.5를 비교하는 subject-box 통합 그림을 만든다. rank_320과 plus_384를 별도 lane으로 표시한다."
                ),
            },
            {
                "id": "final_leaderboard_dashboard",
                "target": "논문 부록",
                "prompt": (
                    "최종 MobileCropNet 후보에 대해 equal-4 score, head sanity, direct alignment, route-collapse flag, target-AR compatibility, "
                    "risk hit rate, latency, final gate pass를 한 화면에 담는 compact dashboard 그림을 만든다."
                ),
            },
        ],
    }


def _plot_scoreboard(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#f7f8fb")
    axes = axes.flatten()

    base = payload.get("base_leaderboard", {})
    extended = payload.get("extended_leaderboard", {})
    labels = ["base route collapse", "base final pass", "extended route collapse", "extended final pass"]
    values = [
        base.get("route_collapse_count") or 0,
        base.get("final_gate_pass_count") or 0,
        extended.get("route_collapse_count") or 0,
        extended.get("final_gate_pass_count") or 0,
    ]
    axes[0].bar(labels, values, color=["#d95f02", "#1b9e77", "#7570b3", "#66a61e"])
    axes[0].set_title("배포 Gate 개수")
    axes[0].tick_params(axis="x", rotation=20)

    route_rows = payload.get("route_recovery", {}).get("final_bundles", [])
    axes[1].bar(
        [str(row.get("method_id") or Path(row.get("bundle_dir", "")).name)[:22] for row in route_rows],
        [_f(row.get("route_balanced_accuracy")) or 0.0 for row in route_rows],
        color="#4c78a8",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Routefix Bundle Route Balance")
    axes[1].tick_params(axis="x", rotation=20)

    no_prior_rows = payload.get("runtime_no_prior", {}).get("runs", [])
    axes[2].bar(
        [str(row.get("run_name"))[-28:] for row in no_prior_rows],
        [_f(row.get("direct_hit_iou_0_5")) or 0.0 for row in no_prior_rows],
        color="#59a14f",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Runtime No-Prior Direct Hit@0.5")
    axes[2].tick_params(axis="x", rotation=25)

    subject_rows = payload.get("subject_box", {}).get("patch_runs", [])
    axes[3].bar(
        [str(row.get("run_name"))[-28:] for row in subject_rows],
        [_f(row.get("direct_subject_iou")) or 0.0 for row in subject_rows],
        color="#e15759",
    )
    axes[3].set_ylim(0, 0.7)
    axes[3].set_title("Subject-Box Patch Direct IoU")
    axes[3].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(axis="y", color="#d8dde8", alpha=0.8)
        ax.set_facecolor("#f7f8fb")
    fig.suptitle("MobileCropNet v4 통합 회복 스냅샷", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_route_report(path: Path, payload: dict[str, Any]) -> None:
    route = payload["route_recovery"]
    route_smokes = payload.get("route_smokes", {})
    lines = [
        "# MobileCropNet v4 경로 붕괴 회복 보고서",
        "",
        f"- 생성 시각: `{payload['generated_at']}`",
        f"- 기준 경로: `{route.get('root')}`",
        "",
        "## 전체 학습 실행",
        "",
        "| 실행 | 상태 | best sel | replay top1 | proposal r@5 | GAIC top1 | GAIC SRCC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in route.get("full_runs", []):
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('status')} | {_fmt(row.get('best_selection_score'))} | "
            f"{_fmt(row.get('replay_candidate_top1_hit'))} | {_fmt(row.get('replay_proposal_recall_at_5_iou_0_5'))} | "
            f"{_fmt(row.get('gaic_top1_mos'))} | {_fmt(row.get('gaic_srcc'))} |"
        )
    lines.extend(
        [
            "",
            "## 최종 번들 점검",
            "",
            "| 방법 | 상태 | route acc | route bal | 붕괴 | direct hit | risk hit | target AR | public dataset |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in route.get("final_bundles", []):
        lines.append(
            f"| `{row.get('method_id') or Path(row.get('bundle_dir', '')).name}` | {row.get('status')} | "
            f"{_fmt(row.get('route_accuracy'))} | {_fmt(row.get('route_balanced_accuracy'))} | {row.get('route_collapse_flag')} | "
            f"{_fmt(row.get('direct_hit_iou_0_5'))} | {_fmt(row.get('direct_risk_hit_iou_0_5'))} | "
            f"{_fmt(row.get('target_ar_compatible'))} | {row.get('public_dataset_count') if row.get('public_dataset_count') is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Release-Gate v2 경로 Smoke 근거",
            "",
            f"- 기준 경로: `{route_smokes.get('root')}`",
            "",
            "| 실행 | 상태 | best epoch | route bal | subj IoU | valid bal | top1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in route_smokes.get("rows", []):
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('state')} | {row.get('best_epoch')} | {_fmt(row.get('best_route_acc'))} | "
            f"{_fmt(row.get('best_subject_iou'))} | {_fmt(row.get('best_subject_valid_balanced_acc'))} | {_fmt(row.get('best_top1_hit'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_no_prior_report(path: Path, payload: dict[str, Any]) -> None:
    no_prior = payload["runtime_no_prior"]
    lines = [
        "# MobileCropNet v4 런타임 무사전입력 재실행 보고서",
        "",
        f"- 생성 시각: `{payload['generated_at']}`",
        f"- 기준 경로: `{no_prior.get('root')}`",
        "",
        "## 실행 행렬 상태",
        "",
        "| 행렬 | 프로파일 | 상태 | 단계 | 현재 변형 | index |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in no_prior.get("matrices", []):
        lines.append(
            f"| `{row.get('matrix_name')}` | {row.get('profile')} | {row.get('state')} | {row.get('phase')} | "
            f"{row.get('current_variant') or '-'} | {row.get('current_index')} / {row.get('variant_count')} |"
        )
    lines.extend(
        [
            "",
            "## 재실행 지표",
            "",
            "| 실행 | 프로파일 | 상태 | best sel | direct hit | best IoU | risk hit | target AR | route | decision |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in no_prior.get("runs", []):
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('profile')} | {row.get('status')} | {_fmt(row.get('best_selection_score'))} | "
            f"{_fmt(row.get('direct_hit_iou_0_5'))} | {_fmt(row.get('direct_best_iou'))} | {_fmt(row.get('direct_risk_hit_iou_0_5'))} | "
            f"{_fmt(row.get('target_ar_compatible'))} | {_fmt(row.get('route_acc'))} | {_fmt(row.get('decision_acc'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subject_box_report(path: Path, payload: dict[str, Any]) -> None:
    subject = payload["subject_box"]
    legacy = subject.get("legacy_v2_runs", [])
    patch = subject.get("patch_runs", [])
    running = subject.get("patch_running", [])
    lines = [
        "# MobileCropNet v4 주체 박스 통합 보고서",
        "",
        f"- 생성 시각: `{payload['generated_at']}`",
        f"- v2 보고서 JSON: `{subject.get('report_json')}`",
        f"- 패치 기준 경로: `{subject.get('patch_root')}`",
        "",
        "## 2026-04-23 V2 Balanced-Valid 실행",
        "",
        "| 트랙 | 프로파일 | 상태 | val subj IoU | direct subj IoU | direct neg acc | direct hit |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in legacy:
        track = row.get("track_name")
        if "SSTK UCTR subjectprior v2" not in str(track):
            continue
        lines.append(
            f"| {track} | {row.get('profile')} | {row.get('artifact_status')} | {_fmt(row.get('best_val_subject_box_iou'))} | "
            f"{_fmt(row.get('direct_test_subject_box_iou_to_teacher'))} | {_fmt(row.get('direct_test_subject_box_valid_negative_acc'))} | "
            f"{_fmt(row.get('direct_test_final_positive_hit_iou_0_5'))} |"
        )
    lines.extend(
        [
            "",
            "## 2026-04-24 패치 전체 실행",
            "",
            "| 실행 | 프로파일 | 상태 | best sel | direct subj IoU | neg acc | direct hit | subject-best IoU | subject-best hit |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in patch:
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('profile')} | {row.get('status')} | {_fmt(row.get('best_selection_score'))} | "
            f"{_fmt(row.get('direct_subject_iou'))} | {_fmt(row.get('direct_negative_acc'))} | {_fmt(row.get('direct_hit_iou_0_5'))} | "
            f"{_fmt(row.get('subject_best_direct_subject_iou'))} | {_fmt(row.get('subject_best_hit_iou_0_5'))} |"
        )
    if running:
        lines.extend(["", "## 아직 실행 중인 항목", "", "| 실행 | 프로파일 | 단계 | 상태 |", "| --- | --- | --- | --- |"])
        for row in running:
            lines.append(f"| `{row.get('run_name')}` | {row.get('profile')} | {row.get('phase')} | {row.get('status')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_prompt_set(path: Path, prompt_set: dict[str, Any]) -> None:
    lines = ["# MobileCropNet v4 그림 프롬프트 세트", "", f"- 생성 시각: `{prompt_set.get('generated_at')}`", ""]
    for item in prompt_set.get("prompts", []):
        lines.extend([f"## {item.get('id')}", "", f"- 대상: `{item.get('target')}`", "", item.get("prompt", ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    base = payload.get("base_leaderboard", {})
    ext = payload.get("extended_leaderboard", {})
    route = payload.get("route_recovery", {})
    route_smokes = payload.get("route_smokes", {})
    head_audits = payload.get("head_audits", {})
    composites = payload.get("internal_composites", {})
    no_prior = payload.get("runtime_no_prior", {})
    subject = payload.get("subject_box", {})
    qual = payload.get("qualitative", {})
    top = (ext.get("top_rows") or base.get("top_rows") or [{}])[0]
    lines = [
        "# MobileCropNet v4.0 통합 회복 보고서",
        "",
        f"- 생성 시각: `{payload['generated_at']}`",
        f"- 산출물 경로: `{payload['output_dir']}`",
        f"- base 리더보드: `{base.get('path')}`",
        f"- 확장 리더보드: `{ext.get('path') if ext.get('exists') else '없음'}`",
        "",
        "## 1. 현재 Gate 상태",
        "",
        f"- base 완료 항목: `{base.get('completed_entry_count')}` / `{base.get('entry_count')}`",
        f"- base route collapse row: `{base.get('route_collapse_count')}`",
        f"- base final gate 통과 row: `{base.get('final_gate_pass_count')}`",
        f"- 현재 최상위 후보: `{top.get('track', '-')}` / `{top.get('profile', '-')}` / equal4_z `{_fmt(top.get('equal4_zscore'))}`",
        "",
        "## 2. Route-Collapse 회복",
        "",
            f"- full route recovery run: `{len(route.get('full_runs', []))}`",
            f"- final bundle 점검: `{len(route.get('final_bundles', []))}`",
            f"- release-gate v2 route smoke run: `{len(route_smokes.get('rows', []))}`",
            f"- release-head audit run: `{len(head_audits.get('rows', []))}`",
            "",
        "| method/run | route bal | collapse | direct hit | risk |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in route.get("final_bundles", []):
        lines.append(
            f"| `{row.get('method_id') or Path(row.get('bundle_dir', '')).name}` | {_fmt(row.get('route_balanced_accuracy'))} | "
            f"{row.get('route_collapse_flag')} | {_fmt(row.get('direct_hit_iou_0_5'))} | {_fmt(row.get('direct_risk_hit_iou_0_5'))} |"
        )
    if not route.get("final_bundles"):
        lines.append("| - | - | - | - | - |")
    lines.extend(["", "| route smoke | best route bal | subj IoU | valid bal | top1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in route_smokes.get("rows", [])[:8]:
        lines.append(
            f"| `{row.get('run_name')}` | {_fmt(row.get('best_route_acc'))} | {_fmt(row.get('best_subject_iou'))} | "
            f"{_fmt(row.get('best_subject_valid_balanced_acc'))} | {_fmt(row.get('best_top1_hit'))} |"
        )
    lines.extend(
        [
            "",
            "### 릴리스 헤드 감사",
            "",
            "| audit | images | route raw | route bal | subj IoU | valid acc | top1 | top1 IoU | 배포 후보 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in head_audits.get("rows", []):
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('image_count')} | {_fmt(row.get('route_accuracy'))} | "
            f"{_fmt(row.get('route_balanced_accuracy'))} | {_fmt(row.get('subject_valid_iou_mean'))} | "
            f"{_fmt(row.get('subject_valid_accuracy'))} | {_fmt(row.get('top1_hit'))} | "
            f"{_fmt(row.get('top1_iou_to_best_positive'))} | {row.get('deploy_candidate')} |"
        )
    lines.extend(
        [
            "",
            "### 내부 Composite No-Prior",
            "",
            "| run | state | images | route bal | route raw | subj IoU | valid acc | top1 | top1 IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in composites.get("rows", []):
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('state')} | {row.get('image_count')} | "
            f"{_fmt(row.get('route_balanced_accuracy'))} | {_fmt(row.get('route_accuracy'))} | "
            f"{_fmt(row.get('subject_box_iou'))} | {_fmt(row.get('subject_box_valid_acc'))} | "
            f"{_fmt(row.get('top1_hit'))} | {_fmt(row.get('top1_iou_to_positive'))} |"
        )
    lines.extend(
        [
            "",
            "## 3. Runtime No-Prior 재실행",
            "",
            "| matrix | state | current |",
            "| --- | --- | --- |",
        ]
    )
    for row in no_prior.get("matrices", []):
        lines.append(f"| `{row.get('matrix_name')}` | {row.get('state')} | {row.get('current_variant') or '-'} |")
    lines.extend(["", "| run | hit@0.5 | risk | target AR |", "| --- | ---: | ---: | ---: |"])
    for row in no_prior.get("runs", []):
        lines.append(
            f"| `{row.get('run_name')}` | {_fmt(row.get('direct_hit_iou_0_5'))} | "
            f"{_fmt(row.get('direct_risk_hit_iou_0_5'))} | {_fmt(row.get('target_ar_compatible'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. Subject-Box 통합",
            "",
            f"- 2026-04-23 v2 source report row: `{len(subject.get('legacy_v2_runs', []))}`",
            f"- 2026-04-24 patch 완료 row: `{len(subject.get('patch_runs', []))}`",
            f"- 2026-04-24 patch 실행 중 row: `{len(subject.get('patch_running', []))}`",
            "",
            "| patch run | profile | subj IoU | neg acc | hit@0.5 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in subject.get("patch_runs", []):
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('profile')} | {_fmt(row.get('direct_subject_iou'))} | "
            f"{_fmt(row.get('direct_negative_acc'))} | {_fmt(row.get('direct_hit_iou_0_5'))} |"
        )
    lines.extend(
        [
            "",
            "## 5. 정성 Pack 및 Figure",
            "",
            f"- 정성 pack 존재: `{qual.get('exists')}`",
            f"- 정성 sample: `{qual.get('sample_count') if qual.get('exists') else '-'}`",
            f"- 그림 prompt set: `{payload.get('figure_prompt_set_md')}`",
            f"- 생성 scorecard: `{payload.get('scoreboard_png')}`",
            "",
            "## 6. 해석",
            "",
            "- extended leaderboard row가 `hard_gate_route_collapse`를 통과하기 전까지 기존 final leaderboard는 route-collapse hard gate에서 막혀 있다.",
            "- Release-gate v2 route smoke와 2026-04-27 release-head audit 기준 모두 route/subject gate 미달이므로 route head와 subject bbox calibration은 여전히 배포 차단 요인이다.",
            "- internal subject-bbox composite no-prior smoke는 route/top1/subject 지표가 모두 크게 낮아 배포 후보에서 제외한다.",
            "- Runtime no-prior rerun은 배포 계약을 별도로 검증한다. 제품 inference는 image와 target AR만 사용해야 하며 mandatory external subject prior를 요구하면 안 된다.",
            "- Subject-box patch는 subject localization 축을 개선했지만, 현재 주 후보를 대체하려면 crop quality와 route gate를 동일 final bundle에서 함께 통과해야 한다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "output_dir": str(args.output_dir),
        "shortlist_manifest_json": str(args.shortlist_manifest_json) if args.shortlist_manifest_json else None,
        "base_leaderboard": _leaderboard_summary(args.base_leaderboard_json),
        "extended_leaderboard": _leaderboard_summary(args.extended_leaderboard_json),
        "route_recovery": _collect_route_recovery(args.route_recovery_root),
        "route_smokes": _collect_route_smokes(args.release_gate_smoke_root),
        "head_audits": _collect_head_audits(args.head_audit_root),
        "internal_composites": _collect_internal_composites(args.internal_composite_root),
        "runtime_no_prior": _collect_runtime_no_prior(args.runtime_no_prior_root),
        "subject_box": _collect_subject_box(args.subject_box_report_json, args.subject_box_patch_root),
        "qualitative": _collect_qualitative(args.qualitative_pack_json),
    }
    prompt_set = _build_figure_prompt_set(payload)
    _write_json(args.output_dir / "unified_recovery_manifest.json", payload)
    _write_json(args.output_dir / "route_collapse_recovery_report.json", payload["route_recovery"])
    _write_json(args.output_dir / "runtime_no_prior_rerun_report.json", payload["runtime_no_prior"])
    _write_json(args.output_dir / "subject_box_integration_report.json", payload["subject_box"])
    _write_json(args.output_dir / "figure_prompt_set.json", prompt_set)

    _write_route_report(args.output_dir / "ROUTE_COLLAPSE_RECOVERY_REPORT.md", payload)
    _write_no_prior_report(args.output_dir / "RUNTIME_NO_PRIOR_RERUN_REPORT.md", payload)
    _write_subject_box_report(args.output_dir / "SUBJECT_BOX_INTEGRATION_REPORT.md", payload)
    _write_prompt_set(args.output_dir / "FIGURE_PROMPT_SET.md", prompt_set)
    scoreboard_png = args.output_dir / "fig_unified_recovery_scoreboard.png"
    _plot_scoreboard(scoreboard_png, payload)
    payload["scoreboard_png"] = str(scoreboard_png)
    payload["figure_prompt_set_md"] = str(args.output_dir / "FIGURE_PROMPT_SET.md")
    _write_json(args.output_dir / "unified_recovery_manifest.json", payload)

    if args.doc_path:
        _write_doc(args.doc_path, payload)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "doc_path": str(args.doc_path) if args.doc_path else None,
                "route_final_bundles": len(payload["route_recovery"].get("final_bundles", [])),
                "route_smokes": len(payload["route_smokes"].get("rows", [])),
                "head_audits": len(payload["head_audits"].get("rows", [])),
                "internal_composites": len(payload["internal_composites"].get("rows", [])),
                "runtime_no_prior_runs": len(payload["runtime_no_prior"].get("runs", [])),
                "subject_box_patch_runs": len(payload["subject_box"].get("patch_runs", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
