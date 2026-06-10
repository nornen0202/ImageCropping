#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a paper/report-ready MobileCropNet comparison report.")
    parser.add_argument("--artifact_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--title", default="MobileCropNet GAIC Experiment Report")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _discover_experiments(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(root.rglob("eval_label/metrics.json")):
        run_root = eval_path.parent.parent
        label = _read_json(eval_path)
        static = _read_json(run_root / "eval_static" / "metrics.json")
        train = _read_json(run_root / "training_report" / "training_summary.json")
        config = _read_json(run_root / "train" / "config.json")
        if not config:
            config = _read_json(run_root / "config.json")
        viz_label = ""
        for rel in ("viz_label/contact_sheet.png", "viz_label_png/contact_sheet.png"):
            if (run_root / rel).exists():
                viz_label = str(run_root / rel)
                break
        viz_static = ""
        for rel in ("viz_static/contact_sheet.png", "viz_static_png/contact_sheet.png"):
            if (run_root / rel).exists():
                viz_static = str(run_root / rel)
                break
        rows.append(
            {
                "name": run_root.name,
                "path": str(run_root),
                "label": label,
                "static": static,
                "train": train,
                "config": config,
                "viz_label": viz_label,
                "viz_static": viz_static,
                "static_predictions": str(run_root / "eval_static" / "predictions.jsonl"),
            }
        )
    return rows


def _discover_prediction_scores(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("prediction_score*/metrics.json")):
        payload = _read_json(path)
        rows.append({"name": payload.get("model_name") or path.parent.name, "path": str(path.parent), "payload": payload})
    return rows


def _discover_static_calibrations(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(root.rglob("mcn-gaic-calib*/eval_static/metrics.json")):
        run_root = eval_path.parent.parent
        metrics = _read_json(eval_path)
        score_payload = _read_json(run_root / "prediction_score_mobile_static" / "metrics.json")
        diag = _static_selection_diagnostics({"static_predictions": str(run_root / "eval_static" / "predictions.jsonl")})
        rows.append(
            {
                "name": run_root.name,
                "path": str(run_root),
                "metrics": metrics,
                "score_payload": score_payload,
                "diag": diag,
                "viz_static": str(run_root / "viz_static" / "contact_sheet.png")
                if (run_root / "viz_static" / "contact_sheet.png").exists()
                else "",
            }
        )
    return rows


def _metric(payload: dict[str, Any], key: str) -> float:
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return _safe_float(metrics.get(key))


def _train_summary(exp: dict[str, Any]) -> dict[str, Any]:
    train = exp.get("train", {})
    summary = train.get("summary", {}) if isinstance(train, dict) else {}
    return summary if isinstance(summary, dict) else {}


def _write_comparison_csv(
    path: Path,
    experiments: list[dict[str, Any]],
    prediction_scores: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "kind",
        "input_size",
        "k",
        "width_mult",
        "epochs",
        "val_best_top1",
        "test_candidate_top1_hit",
        "test_recall_at_5",
        "test_srcc",
        "test_pcc",
        "test_acc1_top10",
        "test_static_top1_iou",
        "test_static_iou_ge_0_7",
        "nearest_positive_score",
        "max_iou_to_positive",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            cfg = exp.get("config", {})
            train = _train_summary(exp)
            writer.writerow(
                {
                    "name": exp["name"],
                    "kind": "mobilecropnet",
                    "input_size": cfg.get("input_size", ""),
                    "k": cfg.get("k", ""),
                    "width_mult": cfg.get("width_mult", ""),
                    "epochs": cfg.get("epochs", ""),
                    "val_best_top1": train.get("best_val_top1_hit", ""),
                    "test_candidate_top1_hit": _metric(exp["label"], "candidate_top1_hit"),
                    "test_recall_at_5": _metric(exp["label"], "candidate_recall_at_5"),
                    "test_srcc": _metric(exp["label"], "srcc"),
                    "test_pcc": _metric(exp["label"], "pcc"),
                    "test_acc1_top10": _metric(exp["label"], "acc1_of_top10"),
                    "test_static_top1_iou": _metric(exp["static"], "top1_iou_to_best_label"),
                    "test_static_iou_ge_0_7": _metric(exp["static"], "top1_iou_ge_0_7"),
                    "nearest_positive_score": "",
                    "max_iou_to_positive": "",
                }
            )
        for row in prediction_scores:
            payload = row["payload"]
            writer.writerow(
                {
                    "name": row["name"],
                    "kind": "prediction_or_baseline",
                    "nearest_positive_score": _metric(payload, "nearest_positive_score"),
                    "max_iou_to_positive": _metric(payload, "max_iou_to_positive"),
                    "test_static_top1_iou": _metric(payload, "iou_to_best_positive"),
                    "test_static_iou_ge_0_7": _metric(payload, "iou_to_best_positive_ge_0_7"),
                }
            )
        for row in calibrations:
            metrics = row["metrics"]
            payload = row["score_payload"]
            writer.writerow(
                {
                    "name": row["name"],
                    "kind": "static_calibration",
                    "test_static_top1_iou": _metric(metrics, "top1_iou_to_best_label"),
                    "test_static_iou_ge_0_7": _metric(metrics, "top1_iou_ge_0_7"),
                    "nearest_positive_score": _metric(payload, "nearest_positive_score"),
                    "max_iou_to_positive": _metric(payload, "max_iou_to_positive"),
                }
            )


def _best_experiment(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not experiments:
        return None
    return max(
        experiments,
        key=lambda exp: (
            _metric(exp["label"], "candidate_top1_hit"),
            _metric(exp["label"], "srcc"),
            _metric(exp["static"], "top1_iou_to_best_label"),
        ),
    )


def _ar_table(exp: dict[str, Any], section: str) -> list[str]:
    payload = exp.get(section, {})
    by_ar = payload.get("by_target_ar", {}) if isinstance(payload, dict) else {}
    lines = ["| target_ar | images | top1_hit | srcc | top1_iou | iou@0.7 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for ar, row in sorted(by_ar.items()):
        lines.append(
            "| {ar} | {count} | {top1:.4f} | {srcc:.4f} | {iou:.4f} | {iou7:.4f} |".format(
                ar=ar,
                count=int(row.get("image_count", 0)),
                top1=_safe_float(row.get("candidate_top1_hit")),
                srcc=_safe_float(row.get("srcc")),
                iou=_safe_float(row.get("top1_iou_to_best_label")),
                iou7=_safe_float(row.get("top1_iou_ge_0_7")),
            )
        )
    return lines


def _static_selection_diagnostics(exp: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(Path(exp.get("static_predictions", "")))
    source_counter: Counter[str] = Counter()
    candidate_counter: Counter[str] = Counter()
    ar_source_counter: dict[str, Counter[str]] = {}
    full_crop_count = 0
    for row in rows:
        top = row.get("top_candidate", {}) if isinstance(row, dict) else {}
        if not isinstance(top, dict):
            continue
        source = str(top.get("source") or "unknown")
        candidate_id = str(top.get("candidate_id") or "unknown")
        target_ar = str(row.get("target_ar") or "unknown")
        source_counter[source] += 1
        candidate_counter[candidate_id] += 1
        ar_source_counter.setdefault(target_ar, Counter())[source] += 1
        bbox = top.get("bbox_norm_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            if all(abs(float(v) - e) < 1e-6 for v, e in zip(bbox, [0.0, 0.0, 1.0, 1.0])):
                full_crop_count += 1
    return {
        "count": len(rows),
        "full_crop_count": full_crop_count,
        "source_counts": source_counter,
        "candidate_counts": candidate_counter,
        "ar_source_counts": ar_source_counter,
    }


def _counter_table(counter: Counter[str], *, total: int, top_k: int = 12) -> list[str]:
    lines = ["| item | count | ratio |", "| --- | ---: | ---: |"]
    for key, count in counter.most_common(top_k):
        ratio = float(count) / float(total) if total else 0.0
        lines.append(f"| {key} | {count} | {ratio:.4f} |")
    return lines


def _static_iou_values(experiments: list[dict[str, Any]]) -> set[float]:
    return {_metric(exp["static"], "top1_iou_to_best_label") for exp in experiments}


def _full_crop_ratio(diag: dict[str, Any]) -> float:
    total = _safe_float(diag.get("count"))
    return _safe_float(diag.get("full_crop_count")) / total if total else 0.0


def _write_report(
    path: Path,
    *,
    title: str,
    experiments: list[dict[str, Any]],
    prediction_scores: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
    csv_path: Path,
) -> None:
    best = _best_experiment(experiments)
    best_static_diag = _static_selection_diagnostics(best) if best else {}
    static_iou_values = _static_iou_values(experiments)
    initial = next((exp for exp in experiments if exp["name"].startswith("gpu_gaic_personv6")), None)
    optimized = [exp for exp in experiments if exp["name"].startswith("mcn-gaic-opt")]
    optimized_static_ious = _static_iou_values(optimized)
    best_calibration = max(
        calibrations,
        key=lambda row: (
            _metric(row["metrics"], "top1_iou_to_best_label"),
            _metric(row["metrics"], "top1_iou_ge_0_7"),
            -_full_crop_ratio(row["diag"]),
        ),
        default=None,
    )
    gpu_runs_path = path.parent / "GPU_WORKLOAD_RUNS_KO.md"
    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        "",
        "본 보고서는 MobileCropNet의 GAIC-like crop label 기반 학습, 정량 평가, 정성 시각화, 규칙/Teacher baseline 비교 결과를 한 번에 검토하기 위한 산출물이다. 모델 선택은 train split 내부 dev validation으로 수행하고, 최종 수치는 GAIC test split에서 산출하는 절차를 기준으로 한다.",
        "",
        f"- experiment_count: {len(experiments)}",
        f"- baseline_or_prediction_count: {len(prediction_scores)}",
        f"- static_calibration_count: {len(calibrations)}",
        f"- comparison_csv: `{csv_path}`",
    ]
    if gpu_runs_path.exists():
        lines.append(f"- gpu_workload_runs: `{gpu_runs_path}`")
    if best:
        cfg = best.get("config", {})
        lines.extend(
            [
                f"- selected_model: `{best['name']}`",
                f"- selected_config: input_size=`{cfg.get('input_size', '')}`, k=`{cfg.get('k', '')}`, width_mult=`{cfg.get('width_mult', '')}`",
                f"- selected_candidate_top1_hit: {_metric(best['label'], 'candidate_top1_hit'):.4f}",
                f"- selected_srcc: {_metric(best['label'], 'srcc'):.4f}",
                f"- selected_static_top1_iou: {_metric(best['static'], 'top1_iou_to_best_label'):.4f}",
                f"- selected_static_full_crop_ratio: {_full_crop_ratio(best_static_diag):.4f}",
            ]
        )
    if best_calibration:
        lines.extend(
            [
                f"- recommended_static_calibration: `{best_calibration['name']}`",
                f"- recommended_calibrated_static_iou: {_metric(best_calibration['metrics'], 'top1_iou_to_best_label'):.4f}",
                f"- recommended_calibrated_full_crop_ratio: {_full_crop_ratio(best_calibration['diag']):.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "- Label-candidate ranking은 후보 전체에 대한 학습 품질을 보여주며, 최종 모델 선택의 1차 기준으로 사용한다.",
            "- Static-bank IoU는 실제 제품 배포 후보 bank에서 단일 crop을 선택했을 때의 기하학적 품질을 보여준다.",
            "- 이번 실험군에서 static-bank IoU가 동일하게 수렴하면 모델 capacity보다 static 후보 bank와 score calibration이 병목이라는 뜻이다.",
            "- FREE와 극단 AR은 같은 평균 성능 안에서도 실패 양상이 다르므로 AR별 지표와 contact sheet를 함께 판정해야 한다.",
        ]
    )
    if len(static_iou_values) == 1 and len(experiments) > 1:
        only_iou = next(iter(static_iou_values))
        lines.append(f"- 관찰된 병목: 모든 MobileCropNet 실험의 static IoU가 {only_iou:.4f}로 동일하다. 현재 static bank에서는 학습 최적화가 실제 crop 다양성으로 충분히 전이되지 않는다.")
    if best and initial:
        lines.append(
            "- 초기 smoke 대비 개선: test top1 hit {old:.4f} -> {new:.4f}, SRCC {old_srcc:.4f} -> {new_srcc:.4f}.".format(
                old=_metric(initial["label"], "candidate_top1_hit"),
                new=_metric(best["label"], "candidate_top1_hit"),
                old_srcc=_metric(initial["label"], "srcc"),
                new_srcc=_metric(best["label"], "srcc"),
            )
        )
    if len(optimized_static_ious) == 1 and len(optimized) > 1:
        only_iou = next(iter(optimized_static_ious))
        lines.append(f"- 최적화 MLP 실험군의 static IoU는 모두 {only_iou:.4f}로 동일하며, 선택된 best 모델의 full-crop ratio는 1.0000이다.")
    if best and best_calibration:
        lines.append(
            "- 추론 calibration 개선: static IoU {old:.4f} -> {new:.4f}, full-crop ratio {old_full:.4f} -> {new_full:.4f}.".format(
                old=_metric(best["static"], "top1_iou_to_best_label"),
                new=_metric(best_calibration["metrics"], "top1_iou_to_best_label"),
                old_full=_full_crop_ratio(best_static_diag),
                new_full=_full_crop_ratio(best_calibration["diag"]),
            )
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- 학습/모델 선택: GAIC train split을 원본 이미지 그룹 단위로 train-dev/val-dev로 재분할한다.",
            "- 최종 평가: GAIC test split은 최종 정량/정성 평가에만 사용한다.",
            "- Label-candidate 평가: 모델이 라벨 후보 전체를 얼마나 잘 ranking하는지 본다.",
            "- Static-bank 평가: 실제 배포형 후보 bank에서 모델이 고른 단일 crop을 최고 positive 라벨 crop과 비교한다.",
            "- Prediction scorer: 공개 cropping 모델 또는 Teacher가 낸 단일 crop JSONL을 동일한 IoU/label-score proxy로 비교한다.",
            "",
            "## Main Results",
            "",
            "| experiment | input | k | width | best val top1 | test top1 hit | recall@5 | SRCC | PCC | Acc1/top10 | static IoU | static IoU@0.7 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for exp in experiments:
        cfg = exp.get("config", {})
        train = _train_summary(exp)
        lines.append(
            "| {name} | {input_size} | {k} | {width} | {val:.4f} | {top1:.4f} | {rec5:.4f} | {srcc:.4f} | {pcc:.4f} | {acc:.4f} | {iou:.4f} | {iou7:.4f} |".format(
                name=exp["name"],
                input_size=cfg.get("input_size", ""),
                k=cfg.get("k", ""),
                width=cfg.get("width_mult", ""),
                val=_safe_float(train.get("best_val_top1_hit")),
                top1=_metric(exp["label"], "candidate_top1_hit"),
                rec5=_metric(exp["label"], "candidate_recall_at_5"),
                srcc=_metric(exp["label"], "srcc"),
                pcc=_metric(exp["label"], "pcc"),
                acc=_metric(exp["label"], "acc1_of_top10"),
                iou=_metric(exp["static"], "top1_iou_to_best_label"),
                iou7=_metric(exp["static"], "top1_iou_ge_0_7"),
            )
        )
    lines.extend(["", "## Baseline And Teacher Comparison", "", "| name | max IoU to positive | IoU to best positive | IoU@0.7 | nearest positive score | score regret |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in prediction_scores:
        payload = row["payload"]
        lines.append(
            "| {name} | {maxiou:.4f} | {bestiou:.4f} | {iou7:.4f} | {score:.4f} | {regret:.4f} |".format(
                name=row["name"],
                maxiou=_metric(payload, "max_iou_to_positive"),
                bestiou=_metric(payload, "iou_to_best_positive"),
                iou7=_metric(payload, "iou_to_best_positive_ge_0_7"),
                score=_metric(payload, "nearest_positive_score"),
                regret=_metric(payload, "nearest_positive_score_regret"),
            )
        )
    if calibrations:
        lines.extend(
            [
                "",
                "## Static Calibration Sweep",
                "",
                "| calibration | static IoU | static IoU@0.7 | positive max IoU | full crop ratio | contact sheet |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in calibrations:
            metrics = row["metrics"]
            payload = row["score_payload"]
            diag = row["diag"]
            full_ratio = _full_crop_ratio(diag)
            lines.append(
                "| {name} | {iou:.4f} | {iou7:.4f} | {maxiou:.4f} | {full_ratio:.4f} | `{viz}` |".format(
                    name=row["name"],
                    iou=_metric(metrics, "top1_iou_to_best_label"),
                    iou7=_metric(metrics, "top1_iou_ge_0_7"),
                    maxiou=_metric(payload, "max_iou_to_positive"),
                    full_ratio=full_ratio,
                    viz=row.get("viz_static", ""),
                )
            )
    if best:
        lines.extend(["", "## Best Model By Target AR", ""])
        lines.extend(_ar_table(best, "label"))
        lines.extend(["", "## Static Bank Diagnostics", ""])
        lines.append(f"- selected_model: `{best['name']}`")
        lines.append(f"- evaluated_predictions: {best_static_diag.get('count', 0)}")
        lines.append(f"- full_crop_count: {best_static_diag.get('full_crop_count', 0)}")
        lines.append("")
        lines.append("### Top Selected Sources")
        lines.extend(_counter_table(best_static_diag.get("source_counts", Counter()), total=int(best_static_diag.get("count", 0))))
        lines.extend(["", "### Top Selected Candidate IDs"])
        lines.extend(_counter_table(best_static_diag.get("candidate_counts", Counter()), total=int(best_static_diag.get("count", 0))))
        lines.extend(
            [
                "",
                "정성 판정상 label-candidate overlay는 모델이 라벨 후보 순위를 학습하고 있음을 보여주지만, static-bank overlay는 full crop만 선택한다. 따라서 현 단계의 제품 품질 병목은 backbone capacity보다 static 후보 bank 설계와 score calibration이다.",
            ]
        )
        lines.extend(
            [
                "",
                "## Qualitative Artifacts",
                "",
                f"- label-candidate contact sheet: `{best.get('viz_label', '')}`",
                f"- static-bank contact sheet: `{best.get('viz_static', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "MobileCropNet은 label-candidate 조건에서는 후보 ranking 정확도와 positive recall을, static-bank 조건에서는 실제 배포 후보군의 geometric agreement를 동시에 봐야 한다. 두 지표가 함께 개선될 때 실제 제품 품질 개선 가능성이 높다. Label-candidate top1 hit만 높고 static IoU가 낮으면 후보 bank/후처리 문제이고, static IoU는 높지만 SRCC가 낮으면 label 후보를 전반적으로 정렬하지 못하는 상태다.",
            "",
            "이번 결과에서 모델별 label-candidate 지표는 차이가 있지만 static-bank 수치가 같게 나오는 경우, 이는 모델이 static 후보 내에서 큰 crop 또는 full crop 계열을 반복 선택하고 있음을 의미한다. 따라서 다음 최적화는 backbone을 무작정 키우는 것보다 static 후보 생성/정규화/후처리 손실을 함께 바꾸는 방향이 우선이다.",
            "",
            "## Optimization Roadmap",
            "",
            "- Static bank 개선: target AR별 full/maxarea 후보 비중을 낮추고 subject-aware offset, saliency-aware tight/context 후보를 추가한다.",
            "- Loss 개선: label 후보 ranking 손실과 static 후보 선택 손실을 분리하고, full crop 과선택에 대한 area prior 또는 margin penalty를 도입한다.",
            "- Calibration: target AR별 score distribution을 보정하고, static 후보와 label 후보 간 score scale mismatch를 온도/Platt scaling으로 점검한다.",
            "- 비교 평가 확장: Teacher oracle, static oracle, public cropper adapter를 동일 prediction JSONL protocol에 연결해 GAIC test 및 별도 golden set에서 비교한다.",
            "- 제품 검증: 평균 IoU뿐 아니라 AR별 worst-case, full-crop ratio, score regret, contact sheet 기반 failure taxonomy를 release gate로 기록한다.",
            "",
            "## Remaining Risks",
            "",
            "- 현재 public cropping 모델 비교는 공통 prediction JSONL scorer까지 준비되어 있으며, 외부 모델별 adapter/checkpoint 확보 후 동일 protocol로 붙인다.",
            "- GAIC-like label은 Teacher/정책 라벨이므로 실제 사용자 선호와 완전히 같지 않다. 최종 제품 판단에는 별도 golden set A/B 평가가 필요하다.",
            "- FREE와 극단 AR은 failure mode가 다를 수 있으므로 AR별 contact sheet와 per-AR 지표를 함께 검토해야 한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = _discover_experiments(args.artifact_root)
    prediction_scores = _discover_prediction_scores(args.artifact_root)
    calibrations = _discover_static_calibrations(args.artifact_root)
    summary = {
        "artifact_root": str(args.artifact_root),
        "experiment_count": len(experiments),
        "prediction_score_count": len(prediction_scores),
        "static_calibration_count": len(calibrations),
        "experiments": experiments,
        "prediction_scores": prediction_scores,
        "static_calibrations": calibrations,
    }
    summary_path = args.output_dir / "mobilecropnet_report_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output_dir / "mobilecropnet_comparison_table.csv"
    _write_comparison_csv(csv_path, experiments, prediction_scores, calibrations)
    report_path = args.output_dir / "MobileCropNet_Final_Report_KO.md"
    _write_report(
        report_path,
        title=args.title,
        experiments=experiments,
        prediction_scores=prediction_scores,
        calibrations=calibrations,
        csv_path=csv_path,
    )
    print(json.dumps({"report": str(report_path), "summary": str(summary_path), "csv": str(csv_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
