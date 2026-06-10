#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from gaic_support_metrics import compute_subject_box_diagnostics
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GAIC saliency/effective-subject-region A/B pipeline.")
    p.add_argument("--filtered_parquet", default="data/GAIC/All/filtered_gaic_all.parquet")
    p.add_argument(
        "--input_features_jsonl",
        default="",
        help="base feature jsonl without new saliency/rerouting. auto-resolves when empty",
    )
    p.add_argument("--image_dir", default="data/GAIC/All/images")
    p.add_argument("--actual_size_cache_json", default="data/GAIC/All/cache/actual_image_size_map.json")
    p.add_argument("--gaic_train_json", default="data/Publics/GAIC/annotations_json/instances_train.json")
    p.add_argument("--gaic_test_json", default="data/Publics/GAIC/annotations_json/instances_test.json")
    p.add_argument("--baseline_candidates_jsonl", default="data/GAIC/All/artifacts/candidates/candidates_ar_gaic_260320_r0.jsonl")
    p.add_argument("--baseline_benchmark_summary", default="data/GAIC/All/artifacts/reports/gaic_benchmark_eval_gaic_260320_r0/benchmark_summary.json")
    p.add_argument(
        "--teacher_proposals_jsonl",
        default="",
        help="optional teacher proposal jsonl(s), comma-separated, injected into variant candidate generation",
    )
    p.add_argument("--saliency_priority", choices=["quality_first", "high_efficiency"], default="quality_first")
    p.add_argument("--saliency_jsonl_override", default="", help="reuse an existing saliency-augmented jsonl instead of rerunning Phase 1")
    p.add_argument("--weights_dir", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--run_tag", default="gaic_260320_r0_saliency_v1")
    p.add_argument("--sample_count", type=int, default=8)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--skip_existing", type=int, default=1)
    p.add_argument("--gpu_ids", default="")
    p.add_argument("--gpu_workers", type=int, default=0)
    return p.parse_args()


def parse_gpu_ids(gpu_ids: str) -> List[str]:
    return [part.strip() for part in str(gpu_ids or "").split(",") if part.strip()]


def resolve_gpu_workers(gpu_ids: Sequence[str], requested_workers: int) -> int:
    req = int(requested_workers)
    if req > 0:
        return max(1, req)
    if gpu_ids:
        return max(1, len(gpu_ids))
    return 1


def run_cmd(args: Sequence[str], *, cwd: Path) -> None:
    print(f"[GAIC-AB] RUN {' '.join(args)}")
    subprocess.run(list(args), cwd=str(cwd), check=True)


def maybe_run(args: Sequence[str], *, cwd: Path, outputs: Sequence[Path], skip_existing: bool) -> None:
    if skip_existing and outputs and all(path.exists() for path in outputs):
        print(f"[GAIC-AB] SKIP existing: {outputs[0]}")
        return
    run_cmd(args, cwd=cwd)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip_box01(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [_safe_float(v, 0.0) for v in box]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [_safe_float(v, 0.0) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [_safe_float(v, 0.0) for v in a]
    bx1, by1, bx2, by2 = [_safe_float(v, 0.0) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return 0.0 if union <= 1e-8 else float(inter / union)


def _norm_bbox_xywh(bbox_xywh: Sequence[float], width: int, height: int) -> List[float]:
    x, y, w, h = [_safe_float(v, 0.0) for v in bbox_xywh]
    iw = max(1.0, float(width))
    ih = max(1.0, float(height))
    return _clip_box01([x / iw, y / ih, (x + w) / iw, (y + h) / ih])


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_input_features_jsonl(path_arg: str) -> Path:
    if str(path_arg).strip():
        return REPO_ROOT / str(path_arg)
    preferred = REPO_ROOT / "data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl"
    fallback = REPO_ROOT / "data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not resolve base feature jsonl.")


def load_gt_catalog(train_json: Path, test_json: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for split, path in (("train", train_json), ("test", test_json)):
        data = _load_json(path)
        image_meta = {
            str(img.get("id")): {
                "width": int(img.get("width", 0)),
                "height": int(img.get("height", 0)),
            }
            for img in data.get("images", [])
        }
        anns_by_image: Dict[str, List[Dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            image_id = str(ann.get("image_id"))
            meta = image_meta.get(image_id)
            if meta is None:
                continue
            anns_by_image.setdefault(image_id, []).append(ann)
        for image_id, anns in anns_by_image.items():
            meta = image_meta.get(image_id)
            if meta is None or not anns:
                continue
            ordered = sorted(anns, key=lambda ann: _safe_float(ann.get("score", 0.0), 0.0), reverse=True)
            best = ordered[0]
            out[image_id] = {
                "annotations": ordered,
                "bbox_norm_xyxy": _norm_bbox_xywh(best.get("bbox", [0, 0, 0, 0]), meta["width"], meta["height"]),
                "mos": _safe_float(best.get("score", 0.0), 0.0),
                "split": split,
                "width": meta["width"],
                "height": meta["height"],
            }
    return out


def load_subject_region_rows(candidates_jsonl: Path, gt_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with candidates_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            gt = gt_catalog.get(image_id)
            if gt is None:
                continue
            routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
            router_signals = routing.get("router_signals", {}) if isinstance(routing.get("router_signals"), dict) else {}
            subject_prior = rec.get("subject_prior", {}) if isinstance(rec.get("subject_prior"), dict) else {}
            baseline_box = subject_prior.get("anchor_bbox_norm_xyxy") or subject_prior.get("bbox_norm_xyxy")
            anchor_box = subject_prior.get("candidate_anchor_bbox_norm_xyxy") or subject_prior.get("anchor_bbox_norm_xyxy") or baseline_box
            effective_box = subject_prior.get("effective_bbox_norm_xyxy") or baseline_box
            diag = compute_subject_box_diagnostics(
                raw_anchor=baseline_box,
                candidate_anchor=anchor_box,
                support_region=effective_box,
                gt_best_box=gt["bbox_norm_xyxy"],
                anns=gt.get("annotations", []),
                width=int(gt.get("width", 0)),
                height=int(gt.get("height", 0)),
            )
            rows.append(
                {
                    "image_id": image_id,
                    "split": str(gt.get("split", "unknown")),
                    "subject_mode": str(routing.get("subject_mode", "")),
                    "c2_primary_area_ratio": _safe_float(
                        router_signals.get("c2_primary_area_ratio", routing.get("subject_set", {}).get("c2_primary_area_ratio", 0.0)),
                        0.0,
                    ),
                    "baseline_iou": _safe_float(diag.get("raw_anchor", {}).get("iou_to_gt_best", 0.0), 0.0),
                    "anchor_iou": _safe_float(diag.get("candidate_anchor", {}).get("iou_to_gt_best", 0.0), 0.0),
                    "effective_iou": _safe_float(diag.get("support_region", {}).get("iou_to_gt_best", 0.0), 0.0),
                    "baseline_mass_recall": _safe_float(diag.get("raw_anchor", {}).get("mass_recall", 0.0), 0.0),
                    "anchor_mass_recall": _safe_float(diag.get("candidate_anchor", {}).get("mass_recall", 0.0), 0.0),
                    "effective_mass_recall": _safe_float(diag.get("support_region", {}).get("mass_recall", 0.0), 0.0),
                    "baseline_centroid_inside": _safe_float(diag.get("raw_anchor", {}).get("centroid_inside", 0.0), 0.0),
                    "anchor_centroid_inside": _safe_float(diag.get("candidate_anchor", {}).get("centroid_inside", 0.0), 0.0),
                    "effective_centroid_inside": _safe_float(diag.get("support_region", {}).get("centroid_inside", 0.0), 0.0),
                    "baseline_centroid_distance": _safe_float(diag.get("raw_anchor", {}).get("centroid_distance", 0.0), 0.0),
                    "anchor_centroid_distance": _safe_float(diag.get("candidate_anchor", {}).get("centroid_distance", 0.0), 0.0),
                    "effective_centroid_distance": _safe_float(diag.get("support_region", {}).get("centroid_distance", 0.0), 0.0),
                    "baseline_latent_bbox_iou": _safe_float(diag.get("raw_anchor", {}).get("latent_bbox_iou", 0.0), 0.0),
                    "anchor_latent_bbox_iou": _safe_float(diag.get("candidate_anchor", {}).get("latent_bbox_iou", 0.0), 0.0),
                    "effective_latent_bbox_iou": _safe_float(diag.get("support_region", {}).get("latent_bbox_iou", 0.0), 0.0),
                    "score_mode": str(
                        subject_prior.get("effective_subject_region", {}).get("score_mode", "")
                        if isinstance(subject_prior.get("effective_subject_region"), dict)
                        else ""
                    ),
                }
            )
    return rows


def summarize_subject_rows(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    vals = [_safe_float(row.get(field, 0.0), 0.0) for row in rows]
    if not vals:
        return {
            "count": 0,
            "mean_iou": 0.0,
            "low_iou_rate": 0.0,
            "mean_mass_recall": 0.0,
            "low_mass_rate": 0.0,
            "centroid_inside_rate": 0.0,
            "centroid_distance_mean": 0.0,
            "latent_bbox_iou_mean": 0.0,
        }

    prefix = field.replace("_iou", "")

    def _bucket(bucket_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        bucket_rows = list(bucket_rows)
        bucket_vals = [_safe_float(row.get(field, 0.0), 0.0) for row in bucket_rows]
        if not bucket_vals:
            return {
                "count": 0,
                "mean_iou": 0.0,
                "low_iou_rate": 0.0,
                "mean_mass_recall": 0.0,
                "low_mass_rate": 0.0,
                "centroid_inside_rate": 0.0,
                "centroid_distance_mean": 0.0,
                "latent_bbox_iou_mean": 0.0,
            }
        mass_vals = [_safe_float(row.get(f"{prefix}_mass_recall", 0.0), 0.0) for row in bucket_rows]
        centroid_inside_vals = [_safe_float(row.get(f"{prefix}_centroid_inside", 0.0), 0.0) for row in bucket_rows]
        centroid_distance_vals = [_safe_float(row.get(f"{prefix}_centroid_distance", 0.0), 0.0) for row in bucket_rows]
        latent_bbox_iou_vals = [_safe_float(row.get(f"{prefix}_latent_bbox_iou", 0.0), 0.0) for row in bucket_rows]
        return {
            "count": int(len(bucket_vals)),
            "mean_iou": round(sum(bucket_vals) / len(bucket_vals), 6),
            "low_iou_rate": round(sum(1 for v in bucket_vals if v < 0.1) / len(bucket_vals), 6),
            "mean_mass_recall": round(sum(mass_vals) / len(mass_vals), 6),
            "low_mass_rate": round(sum(1 for v in mass_vals if v < 0.5) / len(mass_vals), 6),
            "centroid_inside_rate": round(sum(centroid_inside_vals) / len(centroid_inside_vals), 6),
            "centroid_distance_mean": round(sum(centroid_distance_vals) / len(centroid_distance_vals), 6),
            "latent_bbox_iou_mean": round(sum(latent_bbox_iou_vals) / len(latent_bbox_iou_vals), 6),
        }

    scene_rows = [row for row in rows if str(row.get("subject_mode", "")) == "scene_general"]
    ambiguous_tiny_rows = [
        row
        for row in rows
        if str(row.get("subject_mode", "")) == "other_ambiguous" and _safe_float(row.get("c2_primary_area_ratio", 0.0), 0.0) < 0.04
    ]
    neutralized_rows = [row for row in rows if str(row.get("score_mode", "")) == "neutralized"]
    return {
        "overall": _bucket(rows),
        "scene_general": _bucket(scene_rows),
        "other_ambiguous_tiny": _bucket(ambiguous_tiny_rows),
        "neutralized": _bucket(neutralized_rows),
    }


def _metric(summary: Dict[str, Any], *path: str) -> Optional[float]:
    cur: Any = summary
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return None if cur is None else _safe_float(cur, 0.0)


def build_metric_rows(baseline_summary: Dict[str, Any], variant_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    metric_defs = [
        ("Gc", "split_breakdown", "test", "mean_spearman", "higher"),
        ("Gc", "split_breakdown", "test", "mean_kendall_tau_b", "higher"),
        ("Gc", "split_breakdown", "test", "mean_weighted_pair_acc", "higher"),
        ("Gc", "split_breakdown", "test", "mean_hit_at_1", "higher"),
        ("Gc", "split_breakdown", "test", "mean_topq_jaccard", "higher"),
        ("Gc", "local_probability", "test", "mean_local_prob_mae", "lower"),
        ("Gc", "local_probability", "test", "brier_topq", "lower"),
        ("Gc", "local_probability", "test", "ece_topq", "lower"),
        ("Gc", "global_calibration", "test", "mae", "lower"),
        ("Gc", "global_calibration", "test", "qwk", "higher"),
        ("Ge", "split_breakdown", "test", "mean_spearman", "higher"),
        ("Ge", "split_breakdown", "test", "mean_kendall_tau_b", "higher"),
        ("Ge", "split_breakdown", "test", "mean_weighted_pair_acc", "higher"),
        ("Ge", "split_breakdown", "test", "mean_hit_at_1", "higher"),
        ("Ge", "split_breakdown", "test", "mean_topq_jaccard", "higher"),
        ("Ge", "local_probability", "test", "mean_local_prob_mae", "lower"),
        ("Ge", "local_probability", "test", "brier_topq", "lower"),
        ("Ge", "local_probability", "test", "ece_topq", "lower"),
        ("Ge", "global_calibration", "test", "mae", "lower"),
        ("Ge", "global_calibration", "test", "qwk", "higher"),
    ]
    rows: List[Dict[str, Any]] = []
    for proto, block, split, metric_name, better in metric_defs:
        base = _metric(baseline_summary, proto, block, split, metric_name)
        variant = _metric(variant_summary, proto, block, split, metric_name)
        if base is None or variant is None:
            continue
        rows.append(
            {
                "protocol": proto,
                "metric": metric_name,
                "better": better,
                "baseline": round(base, 6),
                "variant": round(variant, 6),
                "delta": round(variant - base, 6),
            }
        )
    return rows


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def write_report(
    *,
    report_path: Path,
    run_tag: str,
    saliency_priority: str,
    metric_rows: List[Dict[str, Any]],
    baseline_subject_diag: Dict[str, Any],
    variant_anchor_diag: Dict[str, Any],
    variant_effective_diag: Dict[str, Any],
    paths: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append(f"# GAIC Saliency / Effective Subject Region A/B Report ({run_tag})")
    lines.append("")
    lines.append("## 1. 실행 개요")
    lines.append("")
    lines.append(f"- saliency priority: `{saliency_priority}`")
    lines.append("- Phase 1: saliency feature export -> Phase 2: rerouting/effective_subject_region -> Phase 3: benchmark A/B")
    lines.append("- variant는 saliency augmentation, rerouting, saliency-aware candidate generation, neutralized subject scoring을 포함한다.")
    lines.append("")
    lines.append("## 2. 경로")
    lines.append("")
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## 3. Benchmark Delta")
    lines.append("")
    lines.append("| protocol | metric | better | baseline | variant | delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for row in metric_rows:
        lines.append(
            f"| {row['protocol']} | {row['metric']} | {row['better']} | "
            f"{_fmt(row['baseline'])} | {_fmt(row['variant'])} | {_fmt(row['delta'])} |"
        )
    lines.append("")
    lines.append("## 4. Subject Region Diagnostics")
    lines.append("")
    lines.append("### 4.1 bbox IoU")
    lines.append("")
    lines.append("| bucket | baseline prior mean IoU | baseline low-IoU(<0.1) | variant anchor mean IoU | variant anchor low-IoU | variant effective mean IoU | variant effective low-IoU |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for bucket in ["overall", "scene_general", "other_ambiguous_tiny", "neutralized"]:
        base = baseline_subject_diag.get(bucket, {})
        anchor = variant_anchor_diag.get(bucket, {})
        eff = variant_effective_diag.get(bucket, {})
        lines.append(
            f"| {bucket} | "
            f"{_fmt(base.get('mean_iou'))} | {_fmt(base.get('low_iou_rate'))} | "
            f"{_fmt(anchor.get('mean_iou'))} | {_fmt(anchor.get('low_iou_rate'))} | "
            f"{_fmt(eff.get('mean_iou'))} | {_fmt(eff.get('low_iou_rate'))} |"
        )
    lines.append("")
    lines.append("### 4.2 latent support mass")
    lines.append("")
    lines.append("| bucket | baseline prior mass | baseline low-mass(<0.5) | variant anchor mass | variant anchor low-mass | variant effective mass | variant effective low-mass | centroid-in(effect) | latent bbox IoU(effect) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for bucket in ["overall", "scene_general", "other_ambiguous_tiny", "neutralized"]:
        base = baseline_subject_diag.get(bucket, {})
        anchor = variant_anchor_diag.get(bucket, {})
        eff = variant_effective_diag.get(bucket, {})
        lines.append(
            f"| {bucket} | "
            f"{_fmt(base.get('mean_mass_recall'))} | {_fmt(base.get('low_mass_rate'))} | "
            f"{_fmt(anchor.get('mean_mass_recall'))} | {_fmt(anchor.get('low_mass_rate'))} | "
            f"{_fmt(eff.get('mean_mass_recall'))} | {_fmt(eff.get('low_mass_rate'))} | "
            f"{_fmt(eff.get('centroid_inside_rate'))} | {_fmt(eff.get('latent_bbox_iou_mean'))} |"
        )
    lines.append("")
    lines.append("## 5. 해석 메모")
    lines.append("")
    lines.append("- `baseline prior`는 기존 candidate jsonl의 `subject_prior.bbox_norm_xyxy` 기준이다.")
    lines.append("- `variant anchor`는 후보 생성 seed로 사용된 `candidate_anchor_bbox_norm_xyxy` 기준이다.")
    lines.append("- `variant effective`는 scorer가 subject-preservation에 실제로 사용하는 `effective_bbox_norm_xyxy` 기준이다.")
    lines.append("- `low-IoU(<0.1)` 비율이 낮아질수록 GT MOS-best와 거의 무관한 tiny prior 실패가 줄었다는 뜻이다.")
    lines.append("- `mean_mass_recall`은 GT 상위 crop latent support를 해당 box가 얼마나 많이 담는지다. distributed scene에서는 bbox IoU보다 이 값이 더 직접적인 진단이다.")
    lines.append("- benchmark 해석에서는 `scene_general`의 distributed foreground 때문에 `variant effective mean IoU`보다 `variant effective mass`를 우선 보는 편이 맞다.")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    py = sys.executable
    filtered_parquet = REPO_ROOT / args.filtered_parquet
    input_features_jsonl = _resolve_input_features_jsonl(args.input_features_jsonl)
    image_dir = REPO_ROOT / args.image_dir
    actual_size_cache_json = REPO_ROOT / args.actual_size_cache_json
    gaic_train_json = REPO_ROOT / args.gaic_train_json
    gaic_test_json = REPO_ROOT / args.gaic_test_json
    baseline_candidates_jsonl = REPO_ROOT / args.baseline_candidates_jsonl
    baseline_benchmark_summary = REPO_ROOT / args.baseline_benchmark_summary
    teacher_proposals_paths = [str((REPO_ROOT / token.strip()).resolve()) for token in str(args.teacher_proposals_jsonl).split(",") if token.strip()]

    precompute_dir = REPO_ROOT / "data/GAIC/All/artifacts/precompute"
    candidate_dir = REPO_ROOT / "data/GAIC/All/artifacts/candidates"
    teacher_dir = REPO_ROOT / "data/GAIC/All/artifacts/teacher/scores"
    training_dir = REPO_ROOT / "data/GAIC/All/artifacts/training_labels" / args.run_tag
    report_dir = REPO_ROOT / f"data/GAIC/All/artifacts/reports/gaic_benchmark_eval_{args.run_tag}"
    ab_report_dir = REPO_ROOT / f"data/GAIC/All/artifacts/reports/gaic_subject_region_ab_{args.run_tag}"
    ab_report_dir.mkdir(parents=True, exist_ok=True)

    saliency_jsonl = precompute_dir / f"{input_features_jsonl.stem}_{args.run_tag}_saliency.jsonl"
    saliency_summary_json = ab_report_dir / "saliency_augment_summary.json"
    if str(args.saliency_jsonl_override).strip():
        saliency_jsonl = (REPO_ROOT / str(args.saliency_jsonl_override)).resolve()
    routed_jsonl = precompute_dir / f"{input_features_jsonl.stem}_{args.run_tag}_routed.jsonl"
    candidates_jsonl = candidate_dir / f"candidates_ar_{args.run_tag}.jsonl"
    candidate_overview_json = candidate_dir / f"candidates_ar_{args.run_tag}_overview.json"
    candidate_overview_by_ar_csv = candidate_dir / f"candidates_ar_{args.run_tag}_overview_by_ar.csv"
    teacher_jsonl = teacher_dir / f"teacher_scores_{args.run_tag}.jsonl"
    teacher_overview_json = teacher_dir / f"teacher_scores_{args.run_tag}_overview.json"
    teacher_overview_by_ar_csv = teacher_dir / f"teacher_scores_{args.run_tag}_overview_by_ar.csv"

    if str(args.saliency_jsonl_override).strip():
        if not saliency_jsonl.exists():
            raise FileNotFoundError(f"saliency_jsonl_override not found: {saliency_jsonl}")
        saliency_summary_json.write_text(
            json.dumps(
                {
                    "status": "reused",
                    "saliency_jsonl": str(saliency_jsonl),
                    "requested_priority": args.saliency_priority,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        saliency_cmd = [
            py,
            "src/scripts/augment_saliency_subject_features.py",
            "--input_jsonl",
            str(input_features_jsonl),
            "--output_jsonl",
            str(saliency_jsonl),
            "--image_dir",
            str(image_dir),
            "--priority",
            str(args.saliency_priority),
            "--weights_dir",
            str(args.weights_dir),
            "--device",
            str(args.device),
            "--summary_json",
            str(saliency_summary_json),
            "--overwrite",
            "1",
            "--max_images",
            str(args.max_images),
        ]
        gpu_ids = parse_gpu_ids(str(args.gpu_ids))
        if len(gpu_ids) > 1 and str(args.device) != "cpu":
            saliency_cmd.extend(
                [
                    "--multi_gpu",
                    "1",
                    "--gpu_ids",
                    ",".join(gpu_ids),
                    "--num_workers",
                    str(resolve_gpu_workers(gpu_ids, int(args.gpu_workers))),
                ]
            )
        maybe_run(
            saliency_cmd,
            cwd=REPO_ROOT,
            outputs=[saliency_jsonl, saliency_summary_json],
            skip_existing=bool(int(args.skip_existing)),
        )

    maybe_run(
        [
            py,
            "src/scripts/enrich_subject_mode_jsonl.py",
            "--input_feats_jsonl",
            str(saliency_jsonl),
            "--input_filtered_parquet",
            str(filtered_parquet),
            "--output_jsonl",
            str(routed_jsonl),
        ],
        cwd=REPO_ROOT,
        outputs=[routed_jsonl],
        skip_existing=bool(int(args.skip_existing)),
    )

    candidate_cmd = [
        py,
        "src/generate_candidates.py",
        "--input_parquet",
        str(filtered_parquet),
        "--feats_c2_jsonl",
        str(routed_jsonl),
        "--feats_c3_jsonl",
        str(routed_jsonl),
        "--output_jsonl",
        str(candidates_jsonl),
        "--actual_size_cache_json",
        str(actual_size_cache_json),
        "--max_images",
        str(args.max_images),
        "--num_workers",
        "0",
        "--mp_chunksize",
        "64",
        "--mp_start_method",
        "auto",
        "--ar_list",
        "FREE",
        "1:1",
        "9:16",
        "16:9",
        "3:4",
        "4:3",
        "--image_dir",
        str(image_dir),
        "--overview_json",
        str(candidate_overview_json),
        "--overview_ar_csv",
        str(candidate_overview_by_ar_csv),
    ]
    if teacher_proposals_paths:
        candidate_cmd.append("--teacher_proposals_jsonl")
        candidate_cmd.extend(teacher_proposals_paths)

    maybe_run(
        candidate_cmd,
        cwd=REPO_ROOT,
        outputs=[candidates_jsonl, candidate_overview_json, candidate_overview_by_ar_csv],
        skip_existing=bool(int(args.skip_existing)),
    )

    maybe_run(
        [
            py,
            "src/score_teacher.py",
            "--candidates_jsonl",
            str(candidates_jsonl),
            "--features_jsonl",
            str(routed_jsonl),
            "--parquet",
            str(filtered_parquet),
            "--image_dir",
            str(image_dir),
            "--output_jsonl",
            str(teacher_jsonl),
            "--output_overview_json",
            str(teacher_overview_json),
            "--output_overview_by_ar_csv",
            str(teacher_overview_by_ar_csv),
            "--use_real_expensive",
            "0",
            "--max_images",
            str(args.max_images),
        ],
        cwd=REPO_ROOT,
        outputs=[teacher_jsonl, teacher_overview_json, teacher_overview_by_ar_csv],
        skip_existing=bool(int(args.skip_existing)),
    )

    maybe_run(
        [
            py,
            "src/scripts/build_finalscore_training_data.py",
            "--teacher_scores_jsonl",
            str(teacher_jsonl),
            "--out_dir",
            str(training_dir),
            "--image_root",
            str(image_dir),
        ],
        cwd=REPO_ROOT,
        outputs=[training_dir / "qa_summary.json"],
        skip_existing=bool(int(args.skip_existing)),
    )

    if int(args.max_images) > 0 and int(args.max_images) < 64:
        smoke_summary = {
            "status": "partial_ok",
            "reason": "benchmark skipped for tiny subset because train/val calibration split becomes empty",
            "run_tag": args.run_tag,
            "max_images": int(args.max_images),
            "paths": {
                "saliency_jsonl": str(saliency_jsonl),
                "routed_jsonl": str(routed_jsonl),
                "candidates_jsonl": str(candidates_jsonl),
                "teacher_jsonl": str(teacher_jsonl),
                "training_dir": str(training_dir),
            },
        }
        smoke_summary_path = ab_report_dir / "smoke_summary.json"
        smoke_summary_path.write_text(json.dumps(smoke_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "partial_ok", "smoke_summary": str(smoke_summary_path)}, ensure_ascii=False, indent=2))
        return

    maybe_run(
        [
            py,
            "src/scripts/run_gaic_benchmark_eval.py",
            "--candidates_jsonl",
            str(candidates_jsonl),
            "--features_jsonl",
            str(routed_jsonl),
            "--teacher_jsonl",
            str(teacher_jsonl),
            "--training_label_dir",
            str(training_dir),
            "--gaic_train_json",
            str(gaic_train_json),
            "--gaic_test_json",
            str(gaic_test_json),
            "--image_dir",
            str(image_dir),
            "--output_dir",
            str(report_dir),
            "--sample_count",
            str(args.sample_count),
            "--max_images",
            str(args.max_images),
        ],
        cwd=REPO_ROOT,
        outputs=[report_dir / "benchmark_summary.json"],
        skip_existing=bool(int(args.skip_existing)),
    )

    baseline_summary = _load_json(baseline_benchmark_summary)
    variant_summary = _load_json(report_dir / "benchmark_summary.json")
    metric_rows = build_metric_rows(baseline_summary=baseline_summary, variant_summary=variant_summary)

    gt_catalog = load_gt_catalog(gaic_train_json, gaic_test_json)
    baseline_subject_rows = load_subject_region_rows(baseline_candidates_jsonl, gt_catalog)
    variant_subject_rows = load_subject_region_rows(candidates_jsonl, gt_catalog)
    baseline_subject_diag = summarize_subject_rows(baseline_subject_rows, "baseline_iou")
    variant_anchor_diag = summarize_subject_rows(variant_subject_rows, "anchor_iou")
    variant_effective_diag = summarize_subject_rows(variant_subject_rows, "effective_iou")

    ab_summary = {
        "run_tag": args.run_tag,
        "saliency_priority": args.saliency_priority,
        "paths": {
            "input_features_jsonl": str(input_features_jsonl),
            "saliency_jsonl": str(saliency_jsonl),
            "routed_jsonl": str(routed_jsonl),
            "candidates_jsonl": str(candidates_jsonl),
            "teacher_jsonl": str(teacher_jsonl),
            "training_dir": str(training_dir),
            "benchmark_report_dir": str(report_dir),
            "baseline_benchmark_summary": str(baseline_benchmark_summary),
        },
        "metric_rows": metric_rows,
        "subject_region_diagnostics": {
            "baseline_prior": baseline_subject_diag,
            "variant_anchor": variant_anchor_diag,
            "variant_effective": variant_effective_diag,
        },
    }
    summary_json = ab_report_dir / "ab_summary.json"
    summary_json.write_text(json.dumps(ab_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = ab_report_dir / "AB_COMPARISON_KO.md"
    write_report(
        report_path=report_md,
        run_tag=args.run_tag,
        saliency_priority=args.saliency_priority,
        metric_rows=metric_rows,
        baseline_subject_diag=baseline_subject_diag,
        variant_anchor_diag=variant_anchor_diag,
        variant_effective_diag=variant_effective_diag,
        paths=ab_summary["paths"],
    )

    print(json.dumps({"status": "ok", "summary_json": str(summary_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
