#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mobilecropnet_v4.eval_utils import infer_image_norm, load_mobilecropnet_v4_checkpoint  # noqa: E402
from mobilecropnet_v4.data import box_iou_xyxy  # noqa: E402
from mobilecropnet_v4.gaic_benchmark import load_gaic_annotation_records, order_desc  # noqa: E402
from scripts.evaluate_public_croppers_gaic_v2 import _score_cgs, _score_gaic  # noqa: E402
from scripts.visualize_mobilecropnet_v4_public_ar_comparison import (  # noqa: E402
    AR_COLORS,
    center_crop_for_ar,
    crop_ar,
    load_font,
    load_label_targets,
    norm_to_px,
    run_mobilecropnet_for_record,
    safe_float,
)


TARGET_ARS = ("FREE", "1:1", "3:4", "4:3", "16:9", "9:16")
BENCHMARK_ORDER = ("fcdb", "cpc", "gnmc", "gaic")
HEADGOOD_PANEL_ORDER = ("headgood_01", "headgood_02", "headgood_03", "headgood_04")
TEACHER_MODE_ORDER = (
    "portrait_single",
    "object_single",
    "object_multi",
    "scene_general",
    "background_texture_copyspace",
)
PANEL_TITLES = {
    "headgood_01": "Head-good Release Gate Slice 01",
    "headgood_02": "Head-good Release Gate Slice 02",
    "headgood_03": "Head-good Release Gate Slice 03",
    "headgood_04": "Head-good Release Gate Slice 04",
}
PUBLIC_GAIC_COLOR = (245, 217, 70)
PUBLIC_CGS_COLOR = (158, 93, 23)
GT_COLOR = (238, 238, 238)
MCN_BENCH_COLOR = (240, 80, 160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MCN-vs-public-cropper qualitative comparison panels for the MobileCropNet v4 report."
    )
    parser.add_argument(
        "--mcn_checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_public_subjectprior_spot/"
        / "mcn-sstk-public-subjasync-r5bal-balanced-288-20260423-034145/best.pt",
    )
    parser.add_argument(
        "--mcn_public_metrics_jsonl",
        type=Path,
        default=PROJECT_ROOT / "artifacts/unified_public_benchmark_20260420/eval_public/mcn_pubscore_r320k96/per_task_metrics.jsonl",
    )
    parser.add_argument(
        "--selection_source",
        choices=["release_gate_head_good", "public_benchmark"],
        default="release_gate_head_good",
        help="release_gate_head_good uses teacher-labeled direct eval rows so MCN head output can be filtered by route/decision match.",
    )
    parser.add_argument(
        "--release_predictions_jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/mobilecropnet_v4/shortlist_final_eval_20260423/sstk-public/balanced-288/"
        / "direct_test_proposal_topk_rerank/predictions.jsonl",
    )
    parser.add_argument(
        "--release_label_jsonl",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT
            / "data/SSTK/Full_10000/artifacts/"
            / "training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422/"
            / "test/train_conditional_detr_batch.jsonl"
        ],
    )
    parser.add_argument(
        "--public_gaic_metrics_jsonl",
        type=Path,
        default=PROJECT_ROOT / "artifacts/unified_public_benchmark_20260420/eval_public/public_cropper_gaic/per_task_metrics.jsonl",
    )
    parser.add_argument(
        "--public_cgs_metrics_jsonl",
        type=Path,
        default=PROJECT_ROOT / "artifacts/unified_public_benchmark_20260420/eval_public/public_cropper_cgs/per_task_metrics.jsonl",
    )
    parser.add_argument(
        "--gaic_mcn_per_image_jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/mobilecropnet_v4/gaic_v2_official_labels/mcn-gaicv2-balanced288-20260417-101528/"
        / "gaic_official_test/mcn_v4_balanced_288_gaicv2_labels_per_image.jsonl",
    )
    parser.add_argument(
        "--gaic_annotations_json",
        type=Path,
        default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json/instances_test.json",
    )
    parser.add_argument(
        "--gaic_image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
        ],
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT
        / "Implement_Docs/assets_mobilecropnet_v4_master_20260427/public_cropper_qualitative_comparison_20260508",
    )
    parser.add_argument("--assets_root", type=Path, default=PROJECT_ROOT / "Implement_Docs/assets_mobilecropnet_v4_master_20260427")
    parser.add_argument("--pool_per_benchmark", type=int, default=36)
    parser.add_argument("--samples_per_benchmark", type=int, default=5)
    parser.add_argument("--samples_per_subject_mode", type=int, default=4)
    parser.add_argument("--release_route_match_min", type=float, default=1.0)
    parser.add_argument("--release_decision_match_min", type=float, default=1.0)
    parser.add_argument("--release_hit_min", type=float, default=1.0)
    parser.add_argument("--release_mean_iou_min", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selection_policy", choices=["utility_top1", "proposal_top1", "proposal_topk_rerank"], default="proposal_topk_rerank")
    parser.add_argument("--proposal_top_m", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")).lower(),
        str(row.get("image_id", "")),
        "" if row.get("target_ar") is None else str(row.get("target_ar")),
    )


def metric_score(row: dict[str, Any]) -> float:
    dataset = str(row.get("dataset", "")).lower()
    if dataset == "cpc":
        pair = row.get("weighted_pairwise_acc")
        if pair is not None:
            return safe_float(pair) + 0.05 * safe_float(row.get("coverage_at_09"))
        pair = row.get("pairwise_acc")
        if pair is not None:
            return safe_float(pair)
    base = safe_float(row.get("iou_top1"))
    base += 0.15 * safe_float(row.get("gt_rank_at_1"))
    base += 0.06 * safe_float(row.get("coverage_at_09"))
    base -= 0.05 * safe_float(row.get("ar_constraint_violation"))
    return base


def load_public_metric_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[key_for(row)] = row
    return out


def load_public_candidates(path: Path, *, pool_per_benchmark: int) -> list[dict[str, Any]]:
    by_image: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        dataset = str(row.get("dataset", "")).lower()
        if dataset not in {"fcdb", "cpc", "gnmc"}:
            continue
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.exists():
            continue
        key = (dataset, str(row.get("image_id", "")))
        old = by_image.get(key)
        if old is None or metric_score(row) > metric_score(old):
            by_image[key] = row
    out: list[dict[str, Any]] = []
    for dataset in ("fcdb", "cpc", "gnmc"):
        rows = [row for (ds, _), row in by_image.items() if ds == dataset]
        rows.sort(key=metric_score, reverse=True)
        for row in rows[:pool_per_benchmark]:
            out.append(
                {
                    "benchmark": dataset,
                    "dataset": dataset,
                    "image_id": str(row.get("image_id", "")),
                    "image_path": str(row.get("image_path", "")),
                    "width": 0,
                    "height": 0,
                    "selection_metric": row,
                    "selection_score": metric_score(row),
                    "task_key": key_for(row),
                    "gaic_record": None,
                }
            )
    return out


def load_gaic_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = load_gaic_annotation_records(args.gaic_annotations_json, image_roots=args.gaic_image_roots, max_images=None)
    records_by_id = {str(row.get("image_id", "")): row for row in records}
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.gaic_mcn_per_image_jsonl):
        image_id = str(row.get("image_id", ""))
        record = records_by_id.get(image_id)
        if record is None or not Path(str(record.get("image_path", ""))).exists():
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        regret = safe_float(metrics.get("top1_mos_regret"), 99.0)
        top_mos = safe_float(metrics.get("top1_mos"))
        best_mos = safe_float(metrics.get("best_mos"))
        score = (1.0 - min(1.0, regret / 1.5)) + 0.12 * top_mos + 0.08 * safe_float(metrics.get("top1_in_top10"))
        rows.append(
            {
                "benchmark": "gaic",
                "dataset": "gaic",
                "image_id": image_id,
                "image_path": str(record.get("image_path", "")),
                "width": int(record.get("width", 0) or 0),
                "height": int(record.get("height", 0) or 0),
                "selection_metric": {
                    "dataset": "gaic",
                    "image_id": image_id,
                    "image_path": str(record.get("image_path", "")),
                    "target_ar": None,
                    "top1_mos": top_mos,
                    "best_mos": best_mos,
                    "top1_mos_regret": regret,
                    "top1_rank": metrics.get("top1_rank"),
                    "top1_bbox_xyxy_norm": (row.get("top_predictions") or [{}])[0].get("bbox_norm_xyxy"),
                    "top1_score": (row.get("top_predictions") or [{}])[0].get("score"),
                    "gt_boxes": [((row.get("top_mos") or [{}])[0].get("bbox_norm_xyxy") if row.get("top_mos") else None)],
                },
                "selection_score": score,
                "task_key": ("gaic", image_id, ""),
                "gaic_record": record,
            }
        )
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    return rows[: int(args.pool_per_benchmark)], records_by_id


def label_majority(labels: Sequence[str]) -> str:
    counts = Counter(str(item) for item in labels if str(item))
    return counts.most_common(1)[0][0] if counts else "unknown"


def load_release_head_good_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    label_targets = load_label_targets(args.release_label_jsonl)
    by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(args.release_predictions_jsonl):
        image_id = str(row.get("image_id", "")).strip()
        target_ar = str(row.get("target_ar", "")).strip()
        if not image_id or target_ar not in TARGET_ARS:
            continue
        by_image[image_id][target_ar] = row

    out: list[dict[str, Any]] = []
    for image_id, rows_by_ar in by_image.items():
        if any(ar not in rows_by_ar for ar in TARGET_ARS):
            continue
        image_path = Path(str(rows_by_ar["FREE"].get("image_path", "")))
        if not image_path.exists():
            continue
        labels_by_ar = label_targets.get(image_id, {})
        teacher_modes: list[str] = []
        teacher_decisions: list[str] = []
        release_metrics_by_ar: dict[str, dict[str, Any]] = {}
        for ar in TARGET_ARS:
            row = rows_by_ar[ar]
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            label = labels_by_ar.get(ar, {})
            teacher_modes.append(str(label.get("subject_mode", "unknown")))
            teacher_decisions.append(str(label.get("decision_type", "unknown")))
            selected = row.get("selected") if isinstance(row.get("selected"), dict) else {}
            release_metrics_by_ar[ar] = {
                "route_acc": safe_float(metrics.get("route_acc")),
                "decision_acc": safe_float(metrics.get("decision_acc")),
                "final_positive_hit_iou_0_5": safe_float(metrics.get("final_positive_hit_iou_0_5")),
                "final_iou_to_best_positive": safe_float(metrics.get("final_iou_to_best_positive")),
                "final_target_ar_compatible": safe_float(metrics.get("final_target_ar_compatible")),
                "subject_box_valid_acc": safe_float(metrics.get("subject_box_valid_acc")),
                "subject_box_pred_conf": safe_float(metrics.get("subject_box_pred_conf")),
                "selected_bbox_norm_xyxy": selected.get("bbox_norm_xyxy"),
                "selected_score": selected.get("score"),
                "teacher_subject_mode": label.get("subject_mode", "unknown"),
                "teacher_decision_type": label.get("decision_type", "unknown"),
            }
        values = list(release_metrics_by_ar.values())
        route_match = sum(safe_float(v.get("route_acc")) for v in values) / max(1, len(values))
        decision_match = sum(safe_float(v.get("decision_acc")) for v in values) / max(1, len(values))
        hit_rate = sum(safe_float(v.get("final_positive_hit_iou_0_5")) for v in values) / max(1, len(values))
        ar_compat = sum(safe_float(v.get("final_target_ar_compatible")) for v in values) / max(1, len(values))
        mean_iou = sum(safe_float(v.get("final_iou_to_best_positive")) for v in values) / max(1, len(values))
        if route_match + 1e-9 < float(args.release_route_match_min):
            continue
        if decision_match + 1e-9 < float(args.release_decision_match_min):
            continue
        if hit_rate + 1e-9 < float(args.release_hit_min):
            continue
        if ar_compat + 1e-9 < 1.0:
            continue
        if mean_iou + 1e-9 < float(args.release_mean_iou_min):
            continue
        teacher_mode = label_majority(teacher_modes)
        teacher_decision = label_majority(teacher_decisions)
        score = mean_iou + 0.08 * route_match + 0.05 * decision_match + 0.05 * hit_rate
        out.append(
            {
                "benchmark": "release_head_good",
                "dataset": "sstk_release_gate",
                "image_id": image_id,
                "image_path": str(image_path),
                "width": 0,
                "height": 0,
                "selection_metric": {
                    "dataset": "sstk_release_gate",
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "mean_iou_to_best_positive": mean_iou,
                    "route_match_rate": route_match,
                    "decision_match_rate": decision_match,
                    "hit_rate_iou_0_5": hit_rate,
                    "ar_compatible_rate": ar_compat,
                    "target_ar": None,
                    "gt_boxes": [],
                },
                "selection_score": score,
                "task_key": ("sstk_release_gate", image_id, ""),
                "teacher_subject_mode": teacher_mode,
                "teacher_decision": teacher_decision,
                "teacher_labels_by_ar": labels_by_ar,
                "release_metrics_by_ar": release_metrics_by_ar,
                "gaic_record": None,
            }
        )
    out.sort(key=lambda item: item["selection_score"], reverse=True)
    return out


def probe_image_size(sample: dict[str, Any]) -> None:
    if int(sample.get("width") or 0) > 0 and int(sample.get("height") or 0) > 0:
        return
    with Image.open(sample["image_path"]) as image:
        sample["width"], sample["height"] = image.size


def summarize_mcn_heads(mcn: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mode_counter: Counter[str] = Counter()
    decision_counter: Counter[str] = Counter()
    scores: list[float] = []
    positives: list[float] = []
    risks: list[float] = []
    for ar in TARGET_ARS:
        payload = mcn.get(ar, {})
        mode = str(payload.get("subject_mode_label", ""))
        decision = str(payload.get("decision_label", ""))
        if mode:
            mode_counter[mode] += 1
        if decision:
            decision_counter[decision] += 1
        if payload.get("score") is not None:
            scores.append(safe_float(payload.get("score")))
        if payload.get("positive_score") is not None:
            positives.append(safe_float(payload.get("positive_score")))
        if payload.get("risk_score") is not None:
            risks.append(safe_float(payload.get("risk_score")))
    mode = mode_counter.most_common(1)[0][0] if mode_counter else "unknown"
    decision = decision_counter.most_common(1)[0][0] if decision_counter else "unknown"
    return {
        "subject_mode": mode,
        "decision": decision,
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "mean_positive": (sum(positives) / len(positives)) if positives else None,
        "mean_risk": (sum(risks) / len(risks)) if risks else None,
        "route_counts": dict(mode_counter),
        "decision_counts": dict(decision_counter),
    }


def valid_norm_box(box: Any) -> list[float] | None:
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        vals = [clamp01(safe_float(v)) for v in box[:4]]
        if vals[2] > vals[0] and vals[3] > vals[1]:
            return vals
    return None


def best_iou_to_refs(box: Any, refs: Sequence[Any]) -> float | None:
    box_vals = valid_norm_box(box)
    if box_vals is None:
        return None
    ref_vals = [valid_norm_box(ref) for ref in refs]
    ref_vals = [ref for ref in ref_vals if ref is not None]
    if not ref_vals:
        return None
    return max(box_iou_xyxy(box_vals, ref) for ref in ref_vals)


def attach_mcn_iou_fields(sample: dict[str, Any]) -> None:
    release_metrics = sample.get("release_metrics_by_ar") if isinstance(sample.get("release_metrics_by_ar"), dict) else {}
    refs = sample.get("selection_metric", {}).get("gt_boxes") or []
    for ar, payload in (sample.get("mobilecropnet") or {}).items():
        if not isinstance(payload, dict):
            continue
        if isinstance(release_metrics.get(ar), dict):
            payload["benchmark_iou"] = safe_float(release_metrics[ar].get("final_iou_to_best_positive"))
            payload["benchmark_iou_source"] = "release_gate_final_iou_to_best_positive"
            continue
        iou = best_iou_to_refs(payload.get("bbox_norm_xyxy"), refs)
        if iou is not None:
            payload["benchmark_iou"] = float(iou)
            payload["benchmark_iou_source"] = "gt_box_iou"


def attach_mcn_outputs(args: argparse.Namespace, samples: list[dict[str, Any]]) -> None:
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.mcn_checkpoint, device=device)
    input_size = int(ckpt.get("train_config", {}).get("input_size", 320))
    image_mean, image_std = infer_image_norm(ckpt)
    for idx, sample in enumerate(samples, 1):
        probe_image_size(sample)
        record = {
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "width": int(sample["width"]),
            "height": int(sample["height"]),
        }
        sample["mobilecropnet"] = run_mobilecropnet_for_record(
            model=model,
            checkpoint_path=args.mcn_checkpoint,
            record=record,
            target_ars=TARGET_ARS,
            input_size=input_size,
            image_mean=image_mean,
            image_std=image_std,
            device=device,
            selection_policy=args.selection_policy,
            proposal_top_m=int(args.proposal_top_m),
        )
        sample["head_summary"] = summarize_mcn_heads(sample["mobilecropnet"])
        attach_mcn_iou_fields(sample)
        if idx % 20 == 0 or idx == len(samples):
            print(f"[mcn] decoded {idx}/{len(samples)} samples", flush=True)


def attach_release_gate_mcn_outputs(samples: list[dict[str, Any]]) -> None:
    for idx, sample in enumerate(samples, 1):
        probe_image_size(sample)
        mcn: dict[str, dict[str, Any]] = {}
        release_metrics = sample.get("release_metrics_by_ar") if isinstance(sample.get("release_metrics_by_ar"), dict) else {}
        for ar in TARGET_ARS:
            metrics = release_metrics.get(ar) if isinstance(release_metrics.get(ar), dict) else {}
            teacher_mode = str(metrics.get("teacher_subject_mode", sample.get("teacher_subject_mode", "unknown")))
            teacher_decision = str(metrics.get("teacher_decision_type", sample.get("teacher_decision", "unknown")))
            route_label = teacher_mode if safe_float(metrics.get("route_acc")) >= 0.5 else "route_mismatch"
            decision_label = teacher_decision if safe_float(metrics.get("decision_acc")) >= 0.5 else "decision_mismatch"
            mcn[ar] = {
                "model": "MobileCropNet v4 release-gate direct eval",
                "target_ar": ar,
                "selection_source": "release_gate_direct_prediction",
                "bbox_norm_xyxy": metrics.get("selected_bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]),
                "score": metrics.get("selected_score"),
                "positive_score": None,
                "risk_score": None,
                "proposal_score": None,
                "decision_label": decision_label,
                "subject_mode_label": route_label,
                "benchmark_iou": safe_float(metrics.get("final_iou_to_best_positive")),
                "benchmark_iou_source": "release_gate_final_iou_to_best_positive",
                "detail_summary": [
                    f"route_match={safe_float(metrics.get('route_acc')):.0f}",
                    f"decision_match={safe_float(metrics.get('decision_acc')):.0f}",
                    f"hit@0.5={safe_float(metrics.get('final_positive_hit_iou_0_5')):.0f}",
                    f"ar_ok={safe_float(metrics.get('final_target_ar_compatible')):.0f}",
                ],
                "detailed_checklist": {},
                "why_tags": [],
            }
        sample["mobilecropnet"] = mcn
        sample["head_summary"] = summarize_mcn_heads(mcn)
        if idx % 20 == 0 or idx == len(samples):
            print(f"[release] attached direct predictions {idx}/{len(samples)} samples", flush=True)


def select_balanced_samples(candidates: list[dict[str, Any]], *, samples_per_benchmark: int) -> list[dict[str, Any]]:
    target_modes = {
        "portrait_single": 4,
        "portrait_group": 3,
        "object_single": 4,
        "object_multi": 3,
        "scene_general": 4,
        "background_texture_copyspace": 2,
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[str, str]] = set()
    dataset_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    def adjusted_score(sample: dict[str, Any]) -> float:
        mode = str(sample.get("head_summary", {}).get("subject_mode", "unknown"))
        quota = max(1, target_modes.get(mode, 2))
        balance_bonus = max(0.0, 1.0 - float(mode_counts[mode]) / float(quota))
        portrait_bonus = 0.35 if mode.startswith("portrait") and sum(mode_counts[m] for m in mode_counts if m.startswith("portrait")) < 5 else 0.0
        return float(sample.get("selection_score", 0.0)) + 0.22 * balance_bonus + portrait_bonus

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in candidates:
        by_dataset[str(sample.get("benchmark"))].append(sample)
    for rows in by_dataset.values():
        rows.sort(key=lambda row: (row.get("selection_score", 0.0), row.get("head_summary", {}).get("mean_score", 0.0)), reverse=True)

    while any(dataset_counts[ds] < samples_per_benchmark for ds in BENCHMARK_ORDER):
        progressed = False
        for dataset in BENCHMARK_ORDER:
            if dataset_counts[dataset] >= samples_per_benchmark:
                continue
            available = [
                row
                for row in by_dataset.get(dataset, [])
                if (str(row.get("benchmark")), str(row.get("image_id"))) not in selected_ids
            ]
            if not available:
                continue
            best = max(available, key=adjusted_score)
            selected.append(best)
            selected_ids.add((str(best.get("benchmark")), str(best.get("image_id"))))
            dataset_counts[dataset] += 1
            mode_counts[str(best.get("head_summary", {}).get("subject_mode", "unknown"))] += 1
            progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: (BENCHMARK_ORDER.index(str(row.get("benchmark"))), -float(row.get("selection_score", 0.0))))
    for idx, row in enumerate(selected, 1):
        row["sample_index"] = idx
    return selected


def select_release_head_good_samples(candidates: list[dict[str, Any]], *, samples_per_subject_mode: int) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_mode[str(row.get("teacher_subject_mode", "unknown"))].append(row)
    for rows in by_mode.values():
        rows.sort(
            key=lambda row: (
                safe_float(row.get("selection_metric", {}).get("mean_iou_to_best_positive")),
                safe_float(row.get("selection_metric", {}).get("route_match_rate")),
                safe_float(row.get("selection_score")),
            ),
            reverse=True,
        )

    chosen_by_mode: dict[str, list[dict[str, Any]]] = {}
    selected_ids: set[str] = set()
    for mode in TEACHER_MODE_ORDER:
        rows: list[dict[str, Any]] = []
        for row in by_mode.get(mode, []):
            image_id = str(row.get("image_id", ""))
            if image_id in selected_ids:
                continue
            rows.append(row)
            selected_ids.add(image_id)
            if len(rows) >= int(samples_per_subject_mode):
                break
        chosen_by_mode[mode] = rows

    target_count = len(TEACHER_MODE_ORDER) * int(samples_per_subject_mode)
    if sum(len(rows) for rows in chosen_by_mode.values()) < target_count:
        remaining = [row for row in candidates if str(row.get("image_id", "")) not in selected_ids]
        remaining.sort(key=lambda row: safe_float(row.get("selection_score")), reverse=True)
        for row in remaining:
            mode = str(row.get("teacher_subject_mode", "unknown"))
            chosen_by_mode.setdefault(mode, []).append(row)
            selected_ids.add(str(row.get("image_id", "")))
            if sum(len(rows) for rows in chosen_by_mode.values()) >= target_count:
                break

    ordered: list[dict[str, Any]] = []
    for slot in range(int(samples_per_subject_mode)):
        for mode in TEACHER_MODE_ORDER:
            rows = chosen_by_mode.get(mode, [])
            if slot < len(rows):
                sample = rows[slot]
                panel = HEADGOOD_PANEL_ORDER[min(slot, len(HEADGOOD_PANEL_ORDER) - 1)]
                sample["benchmark"] = panel
                sample["panel"] = panel
                ordered.append(sample)
    for idx, row in enumerate(ordered, 1):
        row["sample_index"] = idx
    return ordered


def attach_public_boxes(
    selected: list[dict[str, Any]],
    *,
    gaic_public_index: dict[tuple[str, str, str], dict[str, Any]],
    cgs_public_index: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    for sample in selected:
        key = tuple(sample.get("task_key", ("", "", "")))
        if sample.get("benchmark") == "gaic":
            continue
        gaic = gaic_public_index.get(key)
        cgs = cgs_public_index.get(key)
        if gaic is not None:
            sample["public_gaic"] = public_metric_payload(gaic, "GAIC")
        if cgs is not None:
            sample["public_cgs"] = public_metric_payload(cgs, "CGS")


def public_metric_payload(row: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "method": method,
        "bbox_norm_xyxy": row.get("top1_bbox_xyxy_norm", [0.0, 0.0, 1.0, 1.0]),
        "score": row.get("top1_score"),
        "candidate_id": row.get("top1_candidate_id"),
        "iou_top1": row.get("iou_top1"),
        "weighted_pairwise_acc": row.get("weighted_pairwise_acc"),
        "target_ar": row.get("target_ar"),
    }


def attach_gaic_public_scores(selected: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, Any]:
    gaic_samples = [sample for sample in selected if sample.get("benchmark") == "gaic"]
    records = [records_by_id[str(sample["image_id"])] for sample in gaic_samples if str(sample["image_id"]) in records_by_id]
    if not records:
        return {"skipped": True, "reason": "no_gaic_samples"}
    metadata: dict[str, Any] = {}
    for method, scorer, output_key in (("GAIC", _score_gaic, "public_gaic"), ("CGS", _score_cgs, "public_cgs")):
        scores_by_image, coverage_by_image, meta = scorer(records, device=device)
        metadata[method] = meta
        for sample in gaic_samples:
            image_id = str(sample["image_id"])
            record = records_by_id.get(image_id)
            if record is None:
                continue
            candidates = list(record.get("candidates") or [])
            scores = list(scores_by_image.get(image_id) or [])
            if not candidates or len(scores) != len(candidates):
                continue
            ranked = order_desc(scores)
            top_idx = ranked[0]
            top = candidates[top_idx]
            sample[output_key] = {
                "method": method,
                "bbox_norm_xyxy": top.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]),
                "score": float(scores[top_idx]),
                "candidate_id": top.get("candidate_id", top.get("annotation_id")),
                "mos": top.get("mos"),
                "coverage": coverage_by_image.get(image_id, {}),
            }
    return metadata


def append_unique_candidate(bank: list[dict[str, Any]], seen: set[tuple[float, float, float, float]], candidate: dict[str, Any]) -> None:
    box = valid_norm_box(candidate.get("bbox_norm_xyxy"))
    if box is None:
        return
    key = tuple(round(v, 5) for v in box)
    if key in seen:
        return
    seen.add(key)
    payload = dict(candidate)
    payload["bbox_norm_xyxy"] = box
    bank.append(payload)


def attach_teacher_candidate_bank(selected: list[dict[str, Any]], label_jsonls: Sequence[Path]) -> None:
    selected_ids = {str(sample.get("image_id", "")) for sample in selected}
    banks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    iou_refs: dict[str, list[list[float]]] = defaultdict(list)
    seen_by_image: dict[str, set[tuple[float, float, float, float]]] = defaultdict(set)
    seen_refs_by_image: dict[str, set[tuple[float, float, float, float]]] = defaultdict(set)
    for path in label_jsonls:
        if path is None or not Path(path).exists():
            continue
        for row in read_jsonl(Path(path)):
            image_id = str(row.get("image_id", ""))
            if image_id not in selected_ids:
                continue
            target_ar = str(row.get("target_ar", ""))
            baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
            append_unique_candidate(
                banks[image_id],
                seen_by_image[image_id],
                {
                    "candidate_id": baseline.get("candidate_id", f"teacher_baseline_{target_ar}"),
                    "bbox_norm_xyxy": baseline.get("bbox_norm_xyxy"),
                    "source": "teacher_baseline",
                    "target_ar": target_ar,
                },
            )
            for section in ("matching_targets", "candidate_pool"):
                for idx, candidate in enumerate(row.get(section) or []):
                    if not isinstance(candidate, dict):
                        continue
                    box = valid_norm_box(candidate.get("bbox_norm_xyxy"))
                    is_iou_ref = bool(
                        section == "matching_targets"
                        or candidate.get("is_positive_candidate") is True
                        or str(candidate.get("training_bucket", "")) == "main_positive"
                    )
                    if box is not None and is_iou_ref:
                        ref_key = tuple(round(v, 5) for v in box)
                        if ref_key not in seen_refs_by_image[image_id]:
                            seen_refs_by_image[image_id].add(ref_key)
                            iou_refs[image_id].append(box)
                    append_unique_candidate(
                        banks[image_id],
                        seen_by_image[image_id],
                        {
                            "candidate_id": candidate.get("candidate_id", f"{section}_{target_ar}_{idx:03d}"),
                            "bbox_norm_xyxy": candidate.get("bbox_norm_xyxy"),
                            "source": section,
                            "target_ar": target_ar,
                            "score_rank_pct": candidate.get("score_rank_pct"),
                            "is_positive_candidate": candidate.get("is_positive_candidate"),
                        },
                    )
    for sample in selected:
        sample["teacher_candidate_bank"] = banks.get(str(sample.get("image_id", "")), [])
        sample["teacher_candidate_bank_count"] = len(sample["teacher_candidate_bank"])
        sample["public_iou_reference_boxes"] = iou_refs.get(str(sample.get("image_id", "")), [])
        sample["public_iou_reference_count"] = len(sample["public_iou_reference_boxes"])


def build_public_candidate_record(sample: dict[str, Any]) -> dict[str, Any]:
    probe_image_size(sample)
    bank: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for ar in TARGET_ARS:
        payload = sample.get("mobilecropnet", {}).get(ar, {})
        append_unique_candidate(
            bank,
            seen,
            {
                "candidate_id": f"mcn_{ar}",
                "bbox_norm_xyxy": payload.get("bbox_norm_xyxy"),
                "source": "mcn_ar_crop",
                "target_ar": ar,
            },
        )
    for ar in TARGET_ARS:
        append_unique_candidate(
            bank,
            seen,
            {
                "candidate_id": f"center_{ar}",
                "bbox_norm_xyxy": center_crop_for_ar(ar, int(sample["width"]), int(sample["height"])),
                "source": "center_ar_crop",
                "target_ar": ar,
            },
        )
    for candidate in sample.get("teacher_candidate_bank") or []:
        append_unique_candidate(bank, seen, candidate)
    sample["public_candidate_bank_count"] = len(bank)
    return {
        "image_id": str(sample["image_id"]),
        "image_path": str(sample["image_path"]),
        "width": int(sample["width"]),
        "height": int(sample["height"]),
        "candidates": bank,
    }


def attach_public_scores_from_candidate_bank(selected: list[dict[str, Any]], *, device: torch.device) -> dict[str, Any]:
    records = [build_public_candidate_record(sample) for sample in selected]
    records_by_id = {str(record["image_id"]): record for record in records}
    metadata: dict[str, Any] = {}
    for method, scorer, output_key in (("GAIC", _score_gaic, "public_gaic"), ("CGS", _score_cgs, "public_cgs")):
        try:
            scores_by_image, coverage_by_image, meta = scorer(records, device=device)
        except Exception as exc:
            metadata[method] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        metadata[method] = {"available": True, **meta}
        for sample in selected:
            image_id = str(sample["image_id"])
            record = records_by_id.get(image_id)
            if record is None:
                continue
            candidates = list(record.get("candidates") or [])
            scores = list(scores_by_image.get(image_id) or [])
            if not candidates or len(scores) != len(candidates):
                continue
            top_idx = order_desc(scores)[0]
            top = candidates[top_idx]
            sample[output_key] = {
                "method": method,
                "bbox_norm_xyxy": top.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]),
                "score": float(scores[top_idx]),
                "candidate_id": top.get("candidate_id"),
                "candidate_source": top.get("source"),
                "target_ar": top.get("target_ar"),
                "candidate_bank_count": len(candidates),
                "coverage": coverage_by_image.get(image_id, {}),
            }
    return metadata


def attach_public_iou_fields(selected: list[dict[str, Any]]) -> None:
    for sample in selected:
        refs = sample.get("public_iou_reference_boxes") or sample.get("selection_metric", {}).get("gt_boxes") or []
        for key in ("public_gaic", "public_cgs"):
            payload = sample.get(key)
            if not isinstance(payload, dict):
                continue
            iou = best_iou_to_refs(payload.get("bbox_norm_xyxy"), refs)
            if iou is not None:
                payload["benchmark_iou"] = float(iou)
                payload["benchmark_iou_source"] = "teacher_positive_or_matching_candidate_iou"


def open_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_to_box(image: Image.Image, size: tuple[int, int], *, bg: tuple[int, int, int] = (20, 20, 20)) -> Image.Image:
    out = Image.new("RGB", size, bg)
    work = image.copy()
    work.thumbnail(size, Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
    out.paste(work, ((size[0] - work.width) // 2, (size[1] - work.height) // 2))
    return out


def crop_norm(image: Image.Image, box: Sequence[float]) -> Image.Image:
    x1, y1, x2, y2 = norm_to_px(box, image.width, image.height)
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width, x2))
    y2 = max(y1 + 1, min(image.height, y2))
    return image.crop((x1, y1, x2, y2))


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    *,
    line_gap: int = 3,
) -> int:
    x, y = xy
    words = str(text).replace("\n", " \n ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if word == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = int(getattr(font, "size", 14) * 1.25) + line_gap
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y


def render_original_overlay(sample: dict[str, Any], image: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = fit_to_box(image, size, bg=(15, 16, 18))
    scale = min(size[0] / image.width, size[1] / image.height)
    offset = ((size[0] - int(round(image.width * scale))) // 2, (size[1] - int(round(image.height * scale))) // 2)
    draw = ImageDraw.Draw(tile)
    font = load_font(13)

    def map_box(box: Sequence[float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [clamp01(safe_float(v)) for v in box[:4]]
        return (
            offset[0] + int(round(x1 * image.width * scale)),
            offset[1] + int(round(y1 * image.height * scale)),
            offset[0] + int(round(x2 * image.width * scale)),
            offset[1] + int(round(y2 * image.height * scale)),
        )

    for gt in (sample.get("selection_metric", {}).get("gt_boxes") or [])[:2]:
        if isinstance(gt, list) and len(gt) == 4:
            draw.rectangle(map_box(gt), outline=GT_COLOR, width=2)
    bench_box = sample.get("selection_metric", {}).get("top1_bbox_xyxy_norm")
    if isinstance(bench_box, list):
        draw.rectangle(map_box(bench_box), outline=MCN_BENCH_COLOR, width=2)
    for ar, payload in sample.get("mobilecropnet", {}).items():
        box = payload.get("bbox_norm_xyxy")
        if isinstance(box, list):
            draw.rectangle(map_box(box), outline=AR_COLORS.get(ar, (210, 210, 210)), width=2)
    for key, color in (("public_gaic", PUBLIC_GAIC_COLOR), ("public_cgs", PUBLIC_CGS_COLOR)):
        payload = sample.get(key)
        if isinstance(payload, dict):
            draw.rectangle(map_box(payload.get("bbox_norm_xyxy", [0, 0, 1, 1])), outline=color, width=3)
    if sample.get("dataset") == "sstk_release_gate":
        legend = "MCN AR boxes | GAIC | CGS"
    else:
        legend = "MCN AR boxes | GAIC | CGS | GT | MCN metric"
    tw = int(draw.textlength(legend, font=font))
    draw.rectangle([8, size[1] - 28, min(size[0] - 8, 18 + tw), size[1] - 6], fill=(0, 0, 0))
    draw.text((13, size[1] - 25), legend, fill=(245, 245, 245), font=font)
    return tile


def make_labeled_crop(
    image: Image.Image,
    box: Sequence[float],
    label: str,
    size: tuple[int, int],
    *,
    color: tuple[int, int, int],
    score_text: str = "",
) -> Image.Image:
    label_h = 28
    out = Image.new("RGB", (size[0], size[1] + label_h), (18, 18, 18))
    crop = fit_to_box(crop_norm(image, box), size, bg=(8, 8, 8))
    out.paste(crop, (0, label_h))
    draw = ImageDraw.Draw(out)
    font = load_font(12)
    draw.rectangle([0, 0, size[0], label_h], fill=color)
    text = label if not score_text else f"{label} {score_text}"
    draw.text((6, 7), text[:42], fill=(255, 255, 255), font=font)
    return out


def render_mcn_crop_grid(sample: dict[str, Any], image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (18, 19, 21))
    tile_w = size[0] // 3
    tile_h = size[1] // 2
    for idx, ar in enumerate(TARGET_ARS):
        payload = sample.get("mobilecropnet", {}).get(ar, {})
        box = payload.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
        parts: list[str] = []
        if payload.get("score") is not None:
            parts.append(f"s={safe_float(payload.get('score')):.2f}")
        if payload.get("benchmark_iou") is not None:
            parts.append(f"iou={safe_float(payload.get('benchmark_iou')):.2f}")
        tile = make_labeled_crop(
            image,
            box,
            ar,
            (tile_w - 8, tile_h - 34),
            color=AR_COLORS.get(ar, (100, 100, 100)),
            score_text=" ".join(parts) if parts else "iou=n/a",
        )
        x = (idx % 3) * tile_w + 4
        y = (idx // 3) * tile_h + 4
        out.paste(tile, (x, y))
    return out


def render_public_crop_stack(sample: dict[str, Any], image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (18, 19, 21))
    entries = [
        ("public_gaic", "Public GAIC", PUBLIC_GAIC_COLOR),
        ("public_cgs", "Public CGS", PUBLIC_CGS_COLOR),
    ]
    tile_h = size[1] // 2
    for idx, (key, label, color) in enumerate(entries):
        payload = sample.get(key)
        if not isinstance(payload, dict):
            tile = Image.new("RGB", (size[0] - 8, tile_h - 8), (32, 32, 32))
            draw = ImageDraw.Draw(tile)
            draw.text((12, 12), f"{label}: n/a", fill=(220, 220, 220), font=load_font(14))
        else:
            parts: list[str] = []
            if payload.get("mos") is not None:
                parts.append(f"mos={safe_float(payload.get('mos')):.2f}")
            if payload.get("weighted_pairwise_acc") is not None:
                parts.append(f"pw={safe_float(payload.get('weighted_pairwise_acc')):.2f}")
            if payload.get("score") is not None:
                parts.append(f"s={safe_float(payload.get('score')):.2f}")
            if payload.get("benchmark_iou") is not None:
                parts.append(f"iou={safe_float(payload.get('benchmark_iou')):.2f}")
            elif payload.get("iou_top1") is not None:
                parts.append(f"iou={safe_float(payload.get('iou_top1')):.2f}")
            tile = make_labeled_crop(
                image,
                payload.get("bbox_norm_xyxy", [0, 0, 1, 1]),
                label,
                (size[0] - 8, tile_h - 36),
                color=color,
                score_text=" ".join(parts),
            )
        out.paste(tile, (4, idx * tile_h + 4))
    return out


def render_info_tile(sample: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (245, 246, 248))
    draw = ImageDraw.Draw(out)
    title_font = load_font(17)
    font = load_font(13)
    small = load_font(11)
    x = 16
    y = 14
    benchmark = str(sample.get("benchmark", "")).upper()
    image_id = str(sample.get("image_id", ""))
    y = draw_wrapped(draw, (x, y), f"{benchmark} | {image_id}", title_font, (18, 24, 32), size[0] - 2 * x)
    metric = sample.get("selection_metric", {})
    score_bits = []
    if metric.get("iou_top1") is not None:
        score_bits.append(f"IoU {safe_float(metric.get('iou_top1')):.3f}")
    if metric.get("weighted_pairwise_acc") is not None:
        score_bits.append(f"wPair {safe_float(metric.get('weighted_pairwise_acc')):.3f}")
    if metric.get("top1_mos") is not None:
        score_bits.append(f"MOS {safe_float(metric.get('top1_mos')):.2f}/{safe_float(metric.get('best_mos')):.2f}")
    if metric.get("mean_iou_to_best_positive") is not None:
        score_bits.append(f"mean IoU {safe_float(metric.get('mean_iou_to_best_positive')):.3f}")
    if metric.get("route_match_rate") is not None:
        score_bits.append(f"route {safe_float(metric.get('route_match_rate')):.2f}")
    if metric.get("decision_match_rate") is not None:
        score_bits.append(f"decision {safe_float(metric.get('decision_match_rate')):.2f}")
    if metric.get("hit_rate_iou_0_5") is not None:
        score_bits.append(f"hit@0.5 {safe_float(metric.get('hit_rate_iou_0_5')):.2f}")
    if metric.get("target_ar") not in (None, ""):
        score_bits.append(f"task AR {metric.get('target_ar')}")
    y = draw_wrapped(draw, (x, y + 4), "selection: " + ", ".join(score_bits), font, (54, 65, 78), size[0] - 2 * x)
    teacher_mode = sample.get("teacher_subject_mode")
    teacher_decision = sample.get("teacher_decision")
    if teacher_mode:
        y = draw_wrapped(
            draw,
            (x, y + 4),
            f"teacher head: {teacher_mode} | decision: {teacher_decision or 'unknown'}",
            font,
            (72, 56, 118),
            size[0] - 2 * x,
        )
    head = sample.get("head_summary", {})
    y = draw_wrapped(
        draw,
        (x, y + 8),
        f"MCN head vote: {head.get('subject_mode', 'unknown')} | decision: {head.get('decision', 'unknown')}",
        font,
        (30, 85, 55),
        size[0] - 2 * x,
    )
    mean_bits = []
    if head.get("mean_score") is not None:
        mean_bits.append(f"utility {safe_float(head.get('mean_score')):.2f}")
    if head.get("mean_positive") is not None:
        mean_bits.append(f"positive {safe_float(head.get('mean_positive')):.2f}")
    if head.get("mean_risk") is not None:
        mean_bits.append(f"risk {safe_float(head.get('mean_risk')):.2f}")
    if mean_bits:
        y = draw_wrapped(draw, (x, y + 4), "mean head: " + ", ".join(mean_bits), font, (64, 64, 64), size[0] - 2 * x)
    route_counts = ", ".join(f"{k}:{v}" for k, v in sorted((head.get("route_counts") or {}).items()))
    y = draw_wrapped(draw, (x, y + 4), "AR route votes: " + route_counts, small, (75, 78, 84), size[0] - 2 * x)
    free = sample.get("mobilecropnet", {}).get("FREE", {})
    why = free.get("why_tags") or []
    why_text = ", ".join(str(item.get("tag")) for item in why[:4] if isinstance(item, dict))
    if why_text:
        y = draw_wrapped(draw, (x, y + 8), "FREE why: " + why_text, small, (76, 91, 125), size[0] - 2 * x)
    detail = free.get("detail_summary") or []
    for item in detail[:4]:
        y = draw_wrapped(draw, (x, y + 2), "- " + str(item), small, (76, 91, 125), size[0] - 2 * x)
    return out


def render_benchmark_panel(benchmark: str, samples: list[dict[str, Any]], output_path: Path) -> None:
    row_h = 330
    header_h = 74
    info_w = 360
    original_w = 360
    mcn_w = 690
    public_w = 360
    width = info_w + original_w + mcn_w + public_w
    height = header_h + row_h * len(samples)
    canvas = Image.new("RGB", (width, height), (236, 238, 241))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    head_font = load_font(15)
    draw.rectangle([0, 0, width, header_h], fill=(28, 33, 40))
    title = f"MCN vs Public Cropper Qualitative Comparison | {PANEL_TITLES.get(benchmark, benchmark.upper())}"
    draw.text((22, 16), title, fill=(255, 255, 255), font=title_font)
    columns = [(0, info_w, "sample / MCN heads"), (info_w, original_w, "original + crop boxes"), (info_w + original_w, mcn_w, "MCN AR-specific crops"), (info_w + original_w + mcn_w, public_w, "public cropper free-form")]
    for x0, w, text in columns:
        draw.text((x0 + 14, header_h - 24), text, fill=(214, 220, 226), font=head_font)
    for row_idx, sample in enumerate(samples):
        y0 = header_h + row_idx * row_h
        image = open_image(sample["image_path"])
        tiles = [
            render_info_tile(sample, (info_w, row_h)),
            render_original_overlay(sample, image, (original_w, row_h)),
            render_mcn_crop_grid(sample, image, (mcn_w, row_h)),
            render_public_crop_stack(sample, image, (public_w, row_h)),
        ]
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, y0))
            x += tile.width
        draw.line([0, y0, width, y0], fill=(210, 214, 219), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def root_figure_name(benchmark: str) -> str:
    return f"fig16_mcn_public_cropper_qualitative_{benchmark}.png"


def render_all_panels(args: argparse.Namespace, selected: list[dict[str, Any]]) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    panel_order = HEADGOOD_PANEL_ORDER if args.selection_source == "release_gate_head_good" else BENCHMARK_ORDER
    for benchmark in panel_order:
        rows = [sample for sample in selected if sample.get("benchmark") == benchmark]
        out = args.assets_root / root_figure_name(benchmark)
        render_benchmark_panel(benchmark, rows, out)
        panels[benchmark] = {
            "file": root_figure_name(benchmark),
            "path": str(out.relative_to(PROJECT_ROOT)),
            "bytes": out.stat().st_size,
            "sample_count": len(rows),
        }
    return panels


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    gaic_public_meta: dict[str, Any]
    if args.selection_source == "release_gate_head_good":
        candidates = load_release_head_good_candidates(args)
        print(f"[select] release-gate head-good candidate pool={len(candidates)}", flush=True)
        selected = select_release_head_good_samples(candidates, samples_per_subject_mode=int(args.samples_per_subject_mode))
        attach_release_gate_mcn_outputs(selected)
        attach_teacher_candidate_bank(selected, args.release_label_jsonl)
        gaic_public_meta = attach_public_scores_from_candidate_bank(selected, device=torch.device(args.device))
        attach_public_iou_fields(selected)
    else:
        gaic_public_index = load_public_metric_index(args.public_gaic_metrics_jsonl)
        cgs_public_index = load_public_metric_index(args.public_cgs_metrics_jsonl)
        public_candidates = load_public_candidates(args.mcn_public_metrics_jsonl, pool_per_benchmark=int(args.pool_per_benchmark))
        gaic_candidates, gaic_records_by_id = load_gaic_candidates(args)
        candidates = public_candidates + gaic_candidates
        print(f"[select] public benchmark candidate pool={len(candidates)}", flush=True)
        attach_mcn_outputs(args, candidates)
        selected = select_balanced_samples(candidates, samples_per_benchmark=int(args.samples_per_benchmark))
        attach_public_boxes(selected, gaic_public_index=gaic_public_index, cgs_public_index=cgs_public_index)
        gaic_public_meta = attach_gaic_public_scores(selected, gaic_records_by_id, device=torch.device(args.device))
        attach_public_iou_fields(selected)
    panels = render_all_panels(args, selected)
    panel_order = HEADGOOD_PANEL_ORDER if args.selection_source == "release_gate_head_good" else BENCHMARK_ORDER
    if args.selection_source == "release_gate_head_good":
        selection_note = (
            "Samples are selected from release-gate direct evaluation rows with teacher labels. "
            "The filter requires route/decision match, crop hit@0.5, AR compatibility, and mean IoU thresholds before balancing by teacher subject_mode. "
            "MCN crop boxes and head labels are taken from the same release-gate direct prediction rows, and public GAIC/CGS are then run as free-form rankers over a candidate bank built from MCN AR crops, center crops, and teacher label candidates. "
            "Displayed s=* values are model/ranker scores; displayed iou=* values are IoU to the release-gate best-positive metric for MCN and best IoU to teacher matching/positive candidates for public croppers."
        )
    else:
        selection_note = (
            "Public FCDB/CPC/GNMC candidates are selected from high MCN public-benchmark metric rows; GAIC candidates are selected from high MCN GAIC per-image MOS rows. "
            "Final samples are greedily balanced by MCN-predicted subject_mode and benchmark quota."
        )
    manifest = {
        "generated_at_kst": time.strftime("%Y-%m-%d %H:%M:%S KST", time.localtime()),
        "purpose": "MobileCropNet v4 vs public cropper qualitative comparison for report Section 16.",
        "selection_source": args.selection_source,
        "mcn_checkpoint": str(args.mcn_checkpoint.relative_to(PROJECT_ROOT) if args.mcn_checkpoint.is_absolute() else args.mcn_checkpoint),
        "mcn_public_selection_metrics": str(args.mcn_public_metrics_jsonl.relative_to(PROJECT_ROOT) if args.mcn_public_metrics_jsonl.is_absolute() else args.mcn_public_metrics_jsonl),
        "release_predictions": str(args.release_predictions_jsonl.relative_to(PROJECT_ROOT) if args.release_predictions_jsonl.is_absolute() else args.release_predictions_jsonl),
        "release_label_jsonl": [
            str(path.relative_to(PROJECT_ROOT) if path.is_absolute() else path) for path in args.release_label_jsonl
        ],
        "gaic_selection_metrics": str(args.gaic_mcn_per_image_jsonl.relative_to(PROJECT_ROOT) if args.gaic_mcn_per_image_jsonl.is_absolute() else args.gaic_mcn_per_image_jsonl),
        "public_gaic_metrics": str(args.public_gaic_metrics_jsonl.relative_to(PROJECT_ROOT) if args.public_gaic_metrics_jsonl.is_absolute() else args.public_gaic_metrics_jsonl),
        "public_cgs_metrics": str(args.public_cgs_metrics_jsonl.relative_to(PROJECT_ROOT) if args.public_cgs_metrics_jsonl.is_absolute() else args.public_cgs_metrics_jsonl),
        "target_ars": list(TARGET_ARS),
        "benchmark_order": list(panel_order),
        "sample_count": len(selected),
        "subject_mode_counts": dict(Counter(str(s.get("head_summary", {}).get("subject_mode", "unknown")) for s in selected)),
        "teacher_subject_mode_counts": dict(Counter(str(s.get("teacher_subject_mode", "unknown")) for s in selected)),
        "benchmark_counts": dict(Counter(str(s.get("benchmark", "unknown")) for s in selected)),
        "panels": panels,
        "gaic_public_cropper_metadata": gaic_public_meta,
        "selection_note": selection_note,
        "samples": selected,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(args.output_dir / "mcn_public_cropper_qualitative_comparison_manifest.json", manifest)
    print(json.dumps({"status": "ok", "sample_count": len(selected), "panels": panels}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
