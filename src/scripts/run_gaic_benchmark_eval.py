#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from extract_features.c7_saliency import SaliencyFeatureExtractor
from gaic_support_metrics import build_latent_support, compute_box_mass_metrics as compute_latent_box_mass_metrics, compute_subject_box_diagnostics
from score_teacher import (
    TeacherScorerConfig,
    _has_training_severe_reject,
    _is_face_safe,
    _is_structural_valid,
    _relax_joint_only_hard_reject,
    _resolve_subject_routing_hint,
    apply_subject_policy_overrides,
    area_weight_for_route,
    box_area,
    box_center,
    build_subject_bbox,
    build_teacher_consensus_context,
    collect_c3_info,
    collect_c4_info,
    collect_c5_info,
    compute_candidate_scores,
    context_keep_target_range,
    context_target_range,
    count_c2_person_instances,
    decide_keep_vs_crop,
    effective_lambdas,
    headroom_prior,
    infer_flags,
    infer_portrait_category,
    infer_shot_type,
    iou_xyxy,
    lookroom_prior,
    normalize_final_scores,
    pick_baseline_candidates,
    resolve_num_people_for_route,
    subject_scale_target_range,
    tags_to_tokens,
    tau_improve_for_route,
)
from scripts.build_finalscore_training_data import rank_pct_desc, robust_z_scores, sigmoid, softmax_local
from scripts.build_single_image_crop_report import (
    build_panel as build_detailed_panel,
    denorm_box as report_denorm_box,
    extract_score_details as extract_detailed_score_rows,
    summarize_provenance,
)


PRIMARY_SCORE_FIELDS = (
    "score_rank",
    "score_policy",
    "score_rank_pct",
    "score_z_local",
    "score_z_local_std",
    "score_sigmoid_z_local",
    "score_softmax_local",
    "pseudo_mos_1to5",
    "score_policy_z_local",
    "score_policy_z_local_std",
    "score_policy_sigmoid_z_local",
    "policy_pseudo_mos_1to5",
)
PRIMARY_PROTOCOL_FIELD = "score_rank"
LOCAL_Z_FIELD = "score_z_local"
DEFAULT_ALPHA_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_TOP_QUANTILE = 0.8
DEFAULT_GE_FULL_TOPK = 10
PROBABILITY_VARIANTS = (
    ("rank_robust", "score_z_local", "rank robust-z"),
    ("rank_std", "score_z_local_std", "rank standard-z"),
    ("policy_robust", "score_policy_z_local", "policy robust-z"),
    ("policy_std", "score_policy_z_local_std", "policy standard-z"),
)
SAMPLE_PRIORITY_IDS = ("253929", "302853", "303099", "210333", "343339", "441946", "349220")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIC benchmark evaluation in Gc/Ge modes.")
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--features_jsonl", required=True)
    parser.add_argument("--teacher_jsonl", required=True)
    parser.add_argument("--training_label_dir", required=True)
    parser.add_argument("--gaic_train_json", required=True)
    parser.add_argument("--gaic_test_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_quantile", type=float, default=DEFAULT_TOP_QUANTILE)
    parser.add_argument("--alpha_sweep", nargs="*", type=float, default=list(DEFAULT_ALPHA_SWEEP))
    parser.add_argument("--softmax_tau", type=float, default=0.2)
    parser.add_argument("--sample_count", type=int, default=6)
    parser.add_argument("--ge_full_topk", type=int, default=DEFAULT_GE_FULL_TOPK)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    pos = 0
    while pos < len(indexed):
        end = pos
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[pos][1]:
            end += 1
        avg_rank = 0.5 * (pos + end) + 1.0
        for idx in range(pos, end + 1):
            ranks[indexed[idx][0]] = avg_rank
        pos = end + 1
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return float(cov / math.sqrt(vx * vy))


def spearman_rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return pearson_corr(average_ranks(x), average_ranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = 0
            if x[i] > x[j]:
                dx = 1
            elif x[i] < x[j]:
                dx = -1
            dy = 0
            if y[i] > y[j]:
                dy = 1
            elif y[i] < y[j]:
                dy = -1
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(float((concordant + discordant + ties_x) * (concordant + discordant + ties_y)))
    if denom <= 1e-12:
        return 0.0
    return float((concordant - discordant) / denom)


def weighted_pairwise_concordance(
    pred: Sequence[float],
    gt: Sequence[float],
    *,
    quadratic_cap: Optional[float] = None,
) -> Tuple[float, float, float]:
    numerator = 0.0
    denominator = 0.0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            gt_delta = float(gt[i] - gt[j])
            weight = abs(gt_delta)
            if weight <= 1e-12:
                continue
            if quadratic_cap is not None:
                weight = min(weight * weight, float(quadratic_cap))
            denominator += weight
            pred_delta = float(pred[i] - pred[j])
            if pred_delta * gt_delta > 0.0:
                numerator += weight
    if denominator <= 1e-12:
        return 0.0, 0.0, 0.0
    return float(numerator / denominator), float(numerator), float(denominator)


def top_k_from_quantile(n_items: int, q: float) -> int:
    if n_items <= 0:
        return 0
    return max(1, int(math.ceil((1.0 - float(q)) * n_items)))


def top_indices(values: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), -idx), reverse=True)
    return order[: min(k, len(order))]


def hit_at_1(pred: Sequence[float], gt: Sequence[float]) -> float:
    if not pred or not gt:
        return 0.0
    pred_max = max(float(v) for v in pred)
    gt_max = max(float(v) for v in gt)
    pred_set = {idx for idx, value in enumerate(pred) if float(value) == pred_max}
    gt_set = {idx for idx, value in enumerate(gt) if float(value) == gt_max}
    return 1.0 if pred_set.intersection(gt_set) else 0.0


def top_quantile_jaccard(pred: Sequence[float], gt: Sequence[float], q: float) -> float:
    if not pred or not gt:
        return 0.0
    k = top_k_from_quantile(len(pred), q)
    pred_set = set(top_indices(pred, k))
    gt_set = set(top_indices(gt, k))
    union = pred_set.union(gt_set)
    if not union:
        return 0.0
    return float(len(pred_set.intersection(gt_set)) / float(len(union)))


def ndcg_at_k(pred: Sequence[float], gt: Sequence[float], k: int) -> float:
    if not pred or not gt or k <= 0:
        return 0.0
    k = min(k, len(pred))
    pred_order = top_indices(pred, k)
    ideal_order = top_indices(gt, k)

    def dcg(order: Sequence[int]) -> float:
        total = 0.0
        for rank, idx in enumerate(order, start=1):
            gain = max(float(gt[idx]), 0.0)
            total += (2.0**gain - 1.0) / math.log2(rank + 1.0)
        return total

    actual = dcg(pred_order)
    ideal = dcg(ideal_order)
    if ideal <= 1e-12:
        return 0.0
    return float(actual / ideal)


def binary_top_quantile_labels(values: Sequence[float], q: float) -> List[int]:
    labels = [0] * len(values)
    for idx in top_indices(values, top_k_from_quantile(len(values), q)):
        labels[idx] = 1
    return labels


def log_loss_binary(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    eps = 1e-6
    terms = []
    for target, prob in zip(y_true, y_prob):
        p = clamp(float(prob), eps, 1.0 - eps)
        if int(target) == 1:
            terms.append(-math.log(p))
        else:
            terms.append(-math.log(1.0 - p))
    return mean(terms)


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    num_bins: int = 10,
) -> float:
    if not y_true:
        return 0.0
    bins: List[List[Tuple[int, float]]] = [[] for _ in range(max(1, num_bins))]
    for target, prob in zip(y_true, y_prob):
        p = clamp(float(prob), 0.0, 1.0)
        index = min(len(bins) - 1, int(p * len(bins)))
        bins[index].append((int(target), p))
    total = float(len(y_true))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        acc = mean([float(item[0]) for item in bucket])
        conf = mean([float(item[1]) for item in bucket])
        ece += (len(bucket) / total) * abs(acc - conf)
    return float(ece)


def calibration_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    num_bins: int = 10,
) -> List[Dict[str, float]]:
    if not y_true:
        return []
    bins: List[List[Tuple[int, float]]] = [[] for _ in range(max(1, num_bins))]
    for target, prob in zip(y_true, y_prob):
        p = clamp(float(prob), 0.0, 1.0)
        index = min(len(bins) - 1, int(p * len(bins)))
        bins[index].append((int(target), p))
    rows: List[Dict[str, float]] = []
    for idx, bucket in enumerate(bins):
        lo = idx / float(len(bins))
        hi = (idx + 1) / float(len(bins))
        if not bucket:
            rows.append({"bin_lo": lo, "bin_hi": hi, "confidence": 0.0, "accuracy": 0.0, "count": 0.0})
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "confidence": mean([float(item[1]) for item in bucket]),
                "accuracy": mean([float(item[0]) for item in bucket]),
                "count": float(len(bucket)),
            }
        )
    return rows


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    return float(math.sqrt(mean([(float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)])))


def concordance_correlation_coefficient(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0
    mx = mean(y_true)
    my = mean(y_pred)
    vx = mean([(float(v) - mx) ** 2 for v in y_true])
    vy = mean([(float(v) - my) ** 2 for v in y_pred])
    cov = mean([(float(a) - mx) * (float(b) - my) for a, b in zip(y_true, y_pred)])
    denom = vx + vy + (mx - my) ** 2
    if denom <= 1e-12:
        return 0.0
    return float((2.0 * cov) / denom)


def quadratic_weighted_kappa(y_true: Sequence[float], y_pred: Sequence[float], *, min_rating: int = 1, max_rating: int = 5) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        return 0.0
    n_cat = int(max_rating - min_rating + 1)
    observed = np.zeros((n_cat, n_cat), dtype=np.float64)
    hist_true = np.zeros(n_cat, dtype=np.float64)
    hist_pred = np.zeros(n_cat, dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        ai = min(n_cat - 1, max(0, int(round(float(a))) - min_rating))
        bi = min(n_cat - 1, max(0, int(round(float(b))) - min_rating))
        observed[ai, bi] += 1.0
        hist_true[ai] += 1.0
        hist_pred[bi] += 1.0
    expected = np.outer(hist_true, hist_pred) / max(1.0, float(len(y_true)))
    weights = np.zeros((n_cat, n_cat), dtype=np.float64)
    for i in range(n_cat):
        for j in range(n_cat):
            weights[i, j] = ((i - j) ** 2) / float((n_cat - 1) ** 2)
    observed_weight = float((weights * observed).sum())
    expected_weight = float((weights * expected).sum())
    if expected_weight <= 1e-12:
        return 0.0
    return float(1.0 - (observed_weight / expected_weight))


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    seed: int,
    num_bootstrap: int = 1000,
    alpha: float = 0.95,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    samples = []
    for _ in range(max(100, num_bootstrap)):
        indices = rng.integers(0, len(arr), size=len(arr))
        samples.append(float(arr[indices].mean()))
    lo_q = (1.0 - alpha) / 2.0
    hi_q = 1.0 - lo_q
    return float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q))


def stable_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def assign_calibration_split(official_split: str, image_id: str) -> str:
    if str(official_split) == "test":
        return "test"
    return "train" if stable_fraction(str(image_id)) < 0.8 else "val"


def mode_bucket(subject_mode: str) -> str:
    mode = str(subject_mode or "").strip().lower()
    if mode.startswith("portrait"):
        return "portrait"
    if mode.startswith("scene"):
        return "scene"
    if "copyspace" in mode:
        return "copyspace"
    if mode.startswith("text") or "document" in mode:
        return "text"
    return "other"


def local_z_valid(values: Sequence[float]) -> bool:
    if len(values) < 5:
        return False
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
    abs_dev = sorted(abs(v - median) for v in ordered)
    mad = abs_dev[mid] if len(abs_dev) % 2 == 1 else 0.5 * (abs_dev[mid - 1] + abs_dev[mid])
    return mad >= 1e-6


def standard_z_scores(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    if len(vals) == 1:
        return [0.0]
    mu = mean(vals)
    variance = mean([(float(v) - mu) ** 2 for v in vals])
    std = math.sqrt(max(variance, 0.0))
    if std < 1e-8:
        return [0.0 for _ in vals]
    return [float((float(v) - mu) / std) for v in vals]


def score_rank_value(candidate: Dict[str, Any]) -> float:
    return safe_float(candidate.get("score_rank", safe_float(candidate.get("scores", {}).get("rank", 0.0), 0.0)), 0.0)


def score_policy_value(candidate: Dict[str, Any]) -> float:
    return safe_float(candidate.get("score_policy", safe_float(candidate.get("scores", {}).get("policy", 0.0), 0.0)), 0.0)


def canonical_source_family(source: Any) -> str:
    text = str(source or "").strip()
    if not text:
        return "unknown"
    prefixes = (
        "gaic_gt",
        "baseline_full",
        "baseline_maxarea_subject",
        "baseline_maxarea_center",
        "baseline_maxarea_slide",
        "copyspace_align",
        "object_template",
        "saliency_jitter",
        "phi_thirds",
        "grid",
        "jitter",
        "teacher:",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return "teacher" if prefix == "teacher:" else prefix
    return text.split(":", 1)[0].split("_", 1)[0]


def optional_norm_box(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return [round(float(v), 6) for v in box]


def same_box(a: Optional[Sequence[float]], b: Optional[Sequence[float]], tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return False
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def gt_match_summary(
    *,
    candidate: Dict[str, Any],
    anns: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    top_quantile: float,
) -> Dict[str, Any]:
    bbox = [float(v) for v in candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])]
    if not anns:
        return {
            "best_iou": 0.0,
            "matched_gt_annotation_id": None,
            "matched_gt_mos": None,
            "matched_gt_rank": None,
            "matched_gt_mos_percentile": None,
            "matched_gt_is_topq": False,
            "exact_gt_annotation_id": None,
        }
    gt_boxes = [gt_bbox_norm_xyxy(ann, width, height) for ann in anns]
    gt_scores = [safe_float(ann.get("score", 0.0), 0.0) for ann in anns]
    ranked_ids = [safe_int(anns[idx].get("id", 0)) for idx in top_indices(gt_scores, len(gt_scores))]
    rank_by_ann = {ann_id: idx + 1 for idx, ann_id in enumerate(ranked_ids)}
    topq_ann_ids = {
        safe_int(anns[idx].get("id", 0))
        for idx in top_indices(gt_scores, top_k_from_quantile(len(gt_scores), top_quantile))
    }
    best_idx = max(range(len(anns)), key=lambda idx: iou_xyxy(bbox, gt_boxes[idx]))
    best_ann = anns[best_idx]
    best_iou = iou_xyxy(bbox, gt_boxes[best_idx])
    best_ann_id = safe_int(best_ann.get("id", 0))
    matched_rank = rank_by_ann.get(best_ann_id)
    if matched_rank is None:
        matched_percentile = None
    elif len(anns) <= 1:
        matched_percentile = 1.0
    else:
        matched_percentile = 1.0 - float(matched_rank - 1) / float(len(anns) - 1)
    exact_gt_annotation_id = None
    candidate_id = str(candidate.get("candidate_id", ""))
    if candidate_id.startswith("gaic_gt_"):
        exact_gt_annotation_id = safe_int(candidate_id.split("_")[-1], 0)
    return {
        "best_iou": round(float(best_iou), 6),
        "matched_gt_annotation_id": best_ann_id,
        "matched_gt_mos": round(safe_float(best_ann.get("score", 0.0), 0.0), 6),
        "matched_gt_rank": matched_rank,
        "matched_gt_mos_percentile": (None if matched_percentile is None else round(float(matched_percentile), 6)),
        "matched_gt_is_topq": bool(best_ann_id in topq_ann_ids),
        "exact_gt_annotation_id": exact_gt_annotation_id,
    }


def build_prod_selection_payload(
    *,
    candidate: Dict[str, Any],
    anns: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    top_quantile: float,
    gt_best_row: Dict[str, Any],
    decision_type: str,
) -> Dict[str, Any]:
    prod_match = gt_match_summary(
        candidate=candidate,
        anns=anns,
        width=width,
        height=height,
        top_quantile=top_quantile,
    )
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "bbox_norm_xyxy": [round(float(v), 6) for v in candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])],
        "score_rank": round(score_rank_value(candidate), 6),
        "score_policy": round(score_policy_value(candidate), 6),
        "score_sigmoid_z_local": round(
            safe_float(
                candidate.get("score_sigmoid_z_local", candidate.get("pseudo_prob_rank_local", 0.0)),
                0.0,
            ),
            6,
        ),
        "score_z_local": round(safe_float(candidate.get("score_z_local", 0.0), 0.0), 6),
        "score_policy_z_local": round(safe_float(candidate.get("score_policy_z_local", 0.0), 0.0), 6),
        "pseudo_prob_rank_local": round(safe_float(candidate.get("pseudo_prob_rank_local", 0.0), 0.0), 6),
        "pseudo_prob_policy_local": round(safe_float(candidate.get("pseudo_prob_policy_local", 0.0), 0.0), 6),
        "decision_type": decision_type,
        "source": str(candidate.get("source", "")),
        "winner_source_family": canonical_source_family(candidate.get("source", "")),
        "is_gt_candidate": str(candidate.get("candidate_id", "")).startswith("gaic_gt_"),
        "gt_best_candidate_id": gt_best_row["candidate_id"],
        "iou_to_gt_best": round(iou_xyxy(candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), gt_best_row["bbox_norm_xyxy"]), 6),
        "matched_gt_iou": prod_match["best_iou"],
        "matched_gt_annotation_id": prod_match["matched_gt_annotation_id"],
        "matched_gt_mos": prod_match["matched_gt_mos"],
        "matched_gt_rank": prod_match["matched_gt_rank"],
        "matched_gt_mos_percentile": prod_match["matched_gt_mos_percentile"],
        "matched_gt_is_topq": prod_match["matched_gt_is_topq"],
        "exact_gt_annotation_id": prod_match["exact_gt_annotation_id"],
    }


def make_output_row(
    *,
    image_id: str,
    protocol: str,
    split: str,
    calib_split: str,
    subject_mode: str,
    row: Dict[str, Any],
    ann: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "protocol": protocol,
        "official_split": split,
        "calibration_split": calib_split,
        "subject_mode": subject_mode,
        "mode_bucket": mode_bucket(subject_mode),
        "candidate_id": str(row.get("candidate_id", "")),
        "gt_annotation_id": safe_int(ann.get("id", 0)),
        "gt_flag": safe_int(ann.get("gt_flag", 0)),
        "bbox_norm_xyxy": [round(float(v), 6) for v in row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])],
        "mos": round(safe_float(ann.get("score", 0.0), 0.0), 6),
        "score_rank": round(score_rank_value(row), 6),
        "score_policy": round(score_policy_value(row), 6),
        "score_rank_pct": round(safe_float(row.get("score_rank_pct", 0.0), 0.0), 6),
        "score_z_local": round(safe_float(row.get("score_z_local", 0.0), 0.0), 6),
        "score_z_local_std": round(safe_float(row.get("score_z_local_std", 0.0), 0.0), 6),
        "score_sigmoid_z_local": round(sigmoid(safe_float(row.get("score_z_local", 0.0), 0.0)), 6),
        "pseudo_prob_rank_local": round(safe_float(row.get("pseudo_prob_rank_local", row.get("score_sigmoid_z_local", 0.0)), 0.0), 6),
        "score_softmax_local": round(safe_float(row.get("score_softmax_local", 0.0), 0.0), 6),
        "pseudo_mos_1to5": round(safe_float(row.get("pseudo_mos_1to5", 0.0), 0.0), 6),
        "score_policy_rank_pct": round(safe_float(row.get("score_policy_rank_pct", 0.0), 0.0), 6),
        "score_policy_z_local": round(safe_float(row.get("score_policy_z_local", 0.0), 0.0), 6),
        "score_policy_z_local_std": round(safe_float(row.get("score_policy_z_local_std", 0.0), 0.0), 6),
        "score_policy_sigmoid_z_local": round(safe_float(row.get("score_policy_sigmoid_z_local", 0.0), 0.0), 6),
        "pseudo_prob_policy_local": round(safe_float(row.get("pseudo_prob_policy_local", 0.0), 0.0), 6),
        "policy_pseudo_mos_1to5": round(safe_float(row.get("policy_pseudo_mos_1to5", 0.0), 0.0), 6),
        "raw_score_rank": round(safe_float(row.get("scores", {}).get("rank", 0.0), 0.0), 6),
        "raw_score_policy": round(safe_float(row.get("scores", {}).get("policy", 0.0), 0.0), 6),
    }


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0.0, "min": 0.0, "p50": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": float(len(vals)),
        "min": round(min(vals), 6),
        "p50": round(percentile(vals, 0.5), 6),
        "mean": round(mean(vals), 6),
        "p90": round(percentile(vals, 0.9), 6),
        "max": round(max(vals), 6),
    }


def load_scorer_config(teacher_jsonl: Path) -> TeacherScorerConfig:
    cfg_fields = set(TeacherScorerConfig.__dataclass_fields__.keys())
    with teacher_jsonl.open("r", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    raw_cfg = first.get("teacher_scorer", {}).get("config", {})
    filtered = {key: raw_cfg[key] for key in raw_cfg if key in cfg_fields}
    return TeacherScorerConfig(**filtered)


def load_map_by_image_id(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        image_id = str(row.get("image_id", ""))
        if image_id:
            out[image_id] = row
    return out


def load_training_regression_groups(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if str(row.get("target_ar", "")) != "FREE":
            continue
        image_id = str(row.get("image_id", ""))
        if image_id:
            grouped[image_id].append(row)
    return grouped


def load_gaic_annotations(train_json: Path, test_json: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    images: Dict[str, Dict[str, Any]] = {}
    anns_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for split, path in (("train", train_json), ("test", test_json)):
        payload = load_json(path)
        for image in payload.get("images", []):
            image_id = str(image.get("id"))
            images[image_id] = {
                "image_id": image_id,
                "file_name": str(image.get("file_name", "")),
                "width": safe_int(image.get("width", 0)),
                "height": safe_int(image.get("height", 0)),
                "official_split": split,
            }
        for ann in payload.get("annotations", []):
            image_id = str(ann.get("image_id"))
            if image_id:
                anns_by_image[image_id].append(copy.deepcopy(ann))
    for ann_list in anns_by_image.values():
        ann_list.sort(key=lambda ann: (safe_float(ann.get("score", 0.0), 0.0), -safe_int(ann.get("id", 0))), reverse=True)
    return images, anns_by_image


def gt_bbox_norm_xyxy(ann: Dict[str, Any], width: int, height: int) -> List[float]:
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    width = max(1, int(width))
    height = max(1, int(height))
    x1 = clamp(float(x) / float(width), 0.0, 1.0)
    y1 = clamp(float(y) / float(height), 0.0, 1.0)
    x2 = clamp(float(x + w) / float(width), 0.0, 1.0)
    y2 = clamp(float(y + h) / float(height), 0.0, 1.0)
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def build_gt_candidates(anns: Sequence[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    out = []
    for ann in anns:
        bbox = gt_bbox_norm_xyxy(ann, width, height)
        out.append(
            {
                "candidate_id": f"gaic_gt_{safe_int(ann.get('id', 0))}",
                "bbox_norm_xyxy": bbox,
                "area_ratio": round(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 6),
                "ar": round((bbox[2] - bbox[0]) / max(1e-6, (bbox[3] - bbox[1])), 6),
                "source": "gaic_gt",
                "source_types": ["gaic_gt"],
                "source_lineage": ["gaic_gt"],
                "must_keep": False,
                "priority": 0.0,
                "teacher_derived": False,
                "teacher_ids": [],
                "teacher_stages": [],
                "teacher_provenance": [],
            }
        )
    return out


def build_scoring_context(cand_rec: Dict[str, Any], feat_rec: Dict[str, Any], cfg: TeacherScorerConfig) -> Dict[str, Any]:
    meta_norm = cand_rec.get("meta_norm", {})
    if not isinstance(meta_norm, dict):
        meta_norm = {}

    width = int(safe_float(meta_norm.get("width_norm_ref", meta_norm.get("width", cand_rec.get("width", 1))), 1.0))
    height = int(safe_float(meta_norm.get("height_norm_ref", meta_norm.get("height", cand_rec.get("height", 1))), 1.0))
    width = max(1, width)
    height = max(1, height)
    image_ar = safe_float(meta_norm.get("image_ar_norm_ref", cand_rec.get("image_ar", float(width) / float(height))), 1.0)
    norm_size_source = str(meta_norm.get("size_source", cand_rec.get("size_source", "unknown")))

    tags = tags_to_tokens(cand_rec.get("tags", []))
    routing_hint = _resolve_subject_routing_hint(cand_rec=cand_rec, feat_rec=feat_rec)

    c3_info = collect_c3_info(feat_rec=feat_rec, width=width, height=height, cfg=cfg)
    c5_info = collect_c5_info(feat_rec=feat_rec)
    c4_info = collect_c4_info(feat_rec=feat_rec, width=width, height=height)

    c2_person_count = count_c2_person_instances(feat_rec)
    person_stats = resolve_num_people_for_route(
        routing_hint=routing_hint,
        c3_info=c3_info,
        c2_person_count=c2_person_count,
    )
    num_people = int(person_stats["num_people"])
    has_human_evidence = num_people > 0

    shot_type = infer_shot_type(tags, has_human_evidence=has_human_evidence)
    portrait_category = infer_portrait_category(tags)
    flags = infer_flags(tags, has_human_evidence=has_human_evidence)
    has_people = num_people > 0

    hint_shot_type = str(routing_hint.get("shot_type", "")).strip().lower()
    if hint_shot_type in {"headshot", "half", "full", "group"}:
        shot_type = hint_shot_type

    hint_subject_mode = str(routing_hint.get("subject_mode", "")).strip()
    hint_policy_id = str(routing_hint.get("policy_id", "")).strip()
    hint_scene_subtype = str(routing_hint.get("scene_subtype", "")).strip()

    subject_scale_range, subject_scale_target_region = subject_scale_target_range(
        subject_mode=hint_subject_mode,
        shot_type=shot_type,
        scene_subtype=hint_scene_subtype,
        flags=flags,
        has_human_evidence=bool(has_human_evidence),
    )
    context_keep_range, context_target_region = context_keep_target_range(
        subject_mode=hint_subject_mode,
        scene_subtype=hint_scene_subtype,
        flags=flags,
    )

    route = {
        "shot_type": shot_type,
        "portrait_category": portrait_category,
        "flags": flags,
        "num_people": num_people,
        "c2_person_count": c2_person_count,
        "face_count": int(person_stats["face_count"]),
        "has_human_evidence": bool(has_human_evidence),
        "norm_size_source": norm_size_source,
        "headroom_range": headroom_prior(shot_type, portrait_category, flags),
        "lookroom_range": lookroom_prior(shot_type, flags),
        "context_range": context_keep_range,
        "legacy_context_range": context_target_range(
            shot_type=shot_type,
            flags=flags,
            portrait_category=portrait_category,
            has_human_evidence=bool(has_human_evidence),
        ),
        "subject_scale_range": subject_scale_range,
        "subject_scale_target_region": subject_scale_target_region,
        "context_target_region": context_target_region,
        "tau_improve": tau_improve_for_route(shot_type, flags, num_people),
        "w_area": area_weight_for_route(flags),
        "lambdas": effective_lambdas(cfg, shot_type, flags, has_people),
    }
    route = apply_subject_policy_overrides(
        route=route,
        subject_mode=hint_subject_mode,
        policy_id=hint_policy_id,
        routing_hint=routing_hint,
    )
    subject_box = build_subject_bbox(cand_rec, feat_rec, width=width, height=height)
    subject_centroid = box_center(subject_box)
    teacher_ctx = build_teacher_consensus_context(cand_rec=cand_rec, ar_text="FREE", cfg=cfg)

    return {
        "width": width,
        "height": height,
        "image_ar": image_ar,
        "subject_box": subject_box,
        "subject_centroid": subject_centroid,
        "route": route,
        "teacher_ctx": teacher_ctx,
        "c3_info": c3_info,
        "c4_info": c4_info,
        "c5_info": c5_info,
    }


def attach_local_normalization(candidates: Sequence[Dict[str, Any]], *, softmax_tau: float) -> List[Dict[str, Any]]:
    scored = [copy.deepcopy(candidate) for candidate in candidates]
    rank_scores = [safe_float(candidate.get("scores", {}).get("rank", candidate.get("scores", {}).get("final", 0.0)), 0.0) for candidate in scored]
    policy_scores = [safe_float(candidate.get("scores", {}).get("policy", candidate.get("scores", {}).get("final", 0.0)), 0.0) for candidate in scored]
    rank_pct = rank_pct_desc(rank_scores)
    rank_z_local = robust_z_scores(rank_scores)
    rank_z_local_std = standard_z_scores(rank_scores)
    rank_softmax = softmax_local(rank_scores, tau=softmax_tau)
    policy_pct = rank_pct_desc(policy_scores)
    policy_z_local = robust_z_scores(policy_scores)
    policy_z_local_std = standard_z_scores(policy_scores)
    policy_softmax = softmax_local(policy_scores, tau=softmax_tau)
    for candidate, pct, z_score, z_std, softmax_value, policy_pct_v, policy_z, policy_z_std, policy_softmax_v in zip(
        scored,
        rank_pct,
        rank_z_local,
        rank_z_local_std,
        rank_softmax,
        policy_pct,
        policy_z_local,
        policy_z_local_std,
        policy_softmax,
    ):
        candidate["score_rank"] = safe_float(candidate.get("scores", {}).get("rank", candidate.get("scores", {}).get("final", 0.0)), 0.0)
        candidate["score_policy"] = safe_float(candidate.get("scores", {}).get("policy", candidate.get("scores", {}).get("final", 0.0)), 0.0)
        candidate["score_rank_pct"] = float(pct)
        candidate["score_z_local"] = float(z_score)
        candidate["score_z_local_std"] = float(z_std)
        candidate["score_sigmoid_z_local"] = float(sigmoid(z_score))
        candidate["score_sigmoid_z_local_std"] = float(sigmoid(z_std))
        candidate["pseudo_prob_rank_local"] = float(sigmoid(z_score))
        candidate["pseudo_prob_rank_local_std"] = float(sigmoid(z_std))
        candidate["score_softmax_local"] = float(softmax_value)
        candidate["pseudo_mos_1to5"] = float(1.0 + 4.0 * sigmoid(z_score))
        candidate["score_policy_rank_pct"] = float(policy_pct_v)
        candidate["score_policy_z_local"] = float(policy_z)
        candidate["score_policy_z_local_std"] = float(policy_z_std)
        candidate["score_policy_sigmoid_z_local"] = float(sigmoid(policy_z))
        candidate["score_policy_sigmoid_z_local_std"] = float(sigmoid(policy_z_std))
        candidate["pseudo_prob_policy_local"] = float(sigmoid(policy_z))
        candidate["pseudo_prob_policy_local_std"] = float(sigmoid(policy_z_std))
        candidate["score_policy_softmax_local"] = float(policy_softmax_v)
        candidate["policy_pseudo_mos_1to5"] = float(1.0 + 4.0 * sigmoid(policy_z))
    return scored


def score_candidates_freeform(
    *,
    cand_rec: Dict[str, Any],
    feat_rec: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    cfg: TeacherScorerConfig,
    softmax_tau: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    context = build_scoring_context(cand_rec=cand_rec, feat_rec=feat_rec, cfg=cfg)
    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        scored.append(
            compute_candidate_scores(
                candidate=copy.deepcopy(candidate),
                target_ar=None,
                image_ar=context["image_ar"],
                subject_box=context["subject_box"],
                subject_centroid=context["subject_centroid"],
                c3_info=context["c3_info"],
                c5_info=context["c5_info"],
                c4_info=context["c4_info"],
                route=context["route"],
                teacher_ctx=context["teacher_ctx"],
                cfg=cfg,
            )
        )
    normalized = attach_local_normalization(scored, softmax_tau=softmax_tau)
    return normalized, context["route"], context["teacher_ctx"]


def derive_decision_from_full_pool(
    *,
    scored: Sequence[Dict[str, Any]],
    route: Dict[str, Any],
    teacher_ctx: Dict[str, Any],
    cfg: TeacherScorerConfig,
) -> Dict[str, Any]:
    if not scored:
        return {}
    baseline, _ = pick_baseline_candidates(scored)
    strict_valid = [candidate for candidate in scored if not _has_training_severe_reject(candidate)]
    valid_effective: List[Dict[str, Any]] = list(strict_valid)
    if not valid_effective:
        relaxed_joint: List[Dict[str, Any]] = []
        for candidate in scored:
            if (not _is_structural_valid(candidate)) or (not _is_face_safe(candidate)):
                continue
            repaired = _relax_joint_only_hard_reject(candidate)
            if repaired is not None and not bool(repaired.get("hard_reject", False)):
                relaxed_joint.append(repaired)
        if relaxed_joint:
            valid_effective = relaxed_joint
        else:
            face_safe_non_struct = [candidate for candidate in scored if _is_structural_valid(candidate) and _is_face_safe(candidate)]
            if face_safe_non_struct:
                valid_effective = face_safe_non_struct
            else:
                non_struct = [candidate for candidate in scored if _is_structural_valid(candidate)]
                valid_effective = non_struct if non_struct else list(scored)

    valid_effective = sorted(
        valid_effective,
        key=lambda candidate: safe_float(candidate.get("scores", {}).get("cheap", -1e9), -1e9),
        reverse=True,
    )
    cheap_top_m = list(valid_effective[: max(1, int(cfg.cheap_top_m))])

    if baseline is not None:
        baseline_hard = _has_training_severe_reject(baseline)
        if baseline_hard:
            baseline = None
        elif all(str(candidate.get("candidate_id", "")) != str(baseline.get("candidate_id", "")) for candidate in cheap_top_m):
            cheap_top_m.append(baseline)

    normalize_final_scores(cheap_top_m)
    exp_sorted = sorted(
        cheap_top_m,
        key=lambda candidate: safe_float(candidate.get("scores", {}).get("final", -1e9), -1e9),
        reverse=True,
    )
    non_hard_sorted = [candidate for candidate in exp_sorted if not _has_training_severe_reject(candidate)]
    rank_pool = non_hard_sorted if non_hard_sorted else exp_sorted
    best = rank_pool[0] if rank_pool else baseline or scored[0]

    baseline_effective = baseline
    if baseline_effective is None or bool(baseline_effective.get("hard_reject", False)):
        baseline_effective = non_hard_sorted[0] if non_hard_sorted else best
    if baseline_effective is None:
        baseline_effective = best

    teacher_boxes = teacher_ctx.get("boxes", []) if isinstance(teacher_ctx.get("boxes"), list) else []
    baseline_iou_to_teacher = 0.0
    if baseline_effective is not None and teacher_boxes:
        baseline_iou_to_teacher = max(
            iou_xyxy(baseline_effective.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), box) for box in teacher_boxes
        )
    tau_base = float(route.get("tau_improve", cfg.tau_improve_default))
    tau_effective = tau_base
    if bool(teacher_ctx.get("consensus", False)) and baseline_iou_to_teacher >= float(cfg.teacher_tau_boost_baseline_iou):
        tau_effective = tau_base + float(cfg.teacher_tau_boost_delta)

    decision = decide_keep_vs_crop(best=best, baseline=baseline_effective, tau_improve=tau_effective)
    return {
        "best_candidate": best,
        "baseline_candidate": baseline_effective,
        "decision": decision,
        "cheap_top_m": cheap_top_m,
        "rank_pool": rank_pool,
        "baseline_iou_to_teacher": baseline_iou_to_teacher,
    }


def compute_protocol_trend_metrics(rows: Sequence[Dict[str, Any]], field: str, top_quantile: float) -> Dict[str, float]:
    pred = [safe_float(row.get(field, 0.0), 0.0) for row in rows]
    mos = [safe_float(row.get("mos", 0.0), 0.0) for row in rows]
    pair_acc, pair_num, pair_den = weighted_pairwise_concordance(pred, mos)
    k = top_k_from_quantile(len(rows), top_quantile)
    return {
        "spearman": spearman_rank_corr(pred, mos),
        "kendall_tau_b": kendall_tau_b(pred, mos),
        "weighted_pair_acc": pair_acc,
        "weighted_pair_num": pair_num,
        "weighted_pair_den": pair_den,
        "hit_at_1": hit_at_1(pred, mos),
        "topq_jaccard": top_quantile_jaccard(pred, mos, top_quantile),
        "ndcg_at_topq": ndcg_at_k(pred, mos, k),
    }


def aggregate_trend_metrics(
    metric_rows: Sequence[Dict[str, float]],
    *,
    seed: int,
) -> Dict[str, Any]:
    if not metric_rows:
        return {
            "image_count": 0,
            "mean_spearman": 0.0,
            "mean_kendall_tau_b": 0.0,
            "mean_weighted_pair_acc": 0.0,
            "pooled_weighted_pair_acc": 0.0,
            "mean_hit_at_1": 0.0,
            "mean_topq_jaccard": 0.0,
            "mean_ndcg_at_topq": 0.0,
            "confidence_intervals": {},
        }
    spearman_values = [float(row["spearman"]) for row in metric_rows]
    kendall_values = [float(row["kendall_tau_b"]) for row in metric_rows]
    pair_values = [float(row["weighted_pair_acc"]) for row in metric_rows]
    hit_values = [float(row["hit_at_1"]) for row in metric_rows]
    jaccard_values = [float(row["topq_jaccard"]) for row in metric_rows]
    ndcg_values = [float(row["ndcg_at_topq"]) for row in metric_rows]
    pair_num = sum(float(row["weighted_pair_num"]) for row in metric_rows)
    pair_den = sum(float(row["weighted_pair_den"]) for row in metric_rows)
    return {
        "image_count": len(metric_rows),
        "mean_spearman": mean(spearman_values),
        "mean_kendall_tau_b": mean(kendall_values),
        "mean_weighted_pair_acc": mean(pair_values),
        "pooled_weighted_pair_acc": (pair_num / pair_den) if pair_den > 1e-12 else 0.0,
        "mean_hit_at_1": mean(hit_values),
        "mean_topq_jaccard": mean(jaccard_values),
        "mean_ndcg_at_topq": mean(ndcg_values),
        "confidence_intervals": {
            "mean_spearman": bootstrap_confidence_interval(spearman_values, seed=seed),
            "mean_kendall_tau_b": bootstrap_confidence_interval(kendall_values, seed=seed + 1),
            "mean_weighted_pair_acc": bootstrap_confidence_interval(pair_values, seed=seed + 2),
            "mean_hit_at_1": bootstrap_confidence_interval(hit_values, seed=seed + 3),
            "mean_topq_jaccard": bootstrap_confidence_interval(jaccard_values, seed=seed + 4),
        },
    }


def compute_local_probability_summary(
    image_rows: Sequence[Sequence[Dict[str, Any]]],
    *,
    alpha: float,
    top_quantile: float,
    z_field: str = "score_z_local",
) -> Dict[str, Any]:
    per_image_mae: List[float] = []
    z_valid_pred = 0
    z_valid_mos = 0
    pooled_topq_targets: List[int] = []
    pooled_topq_probs: List[float] = []
    for rows in image_rows:
        pred_z = [safe_float(row.get(z_field, 0.0), 0.0) for row in rows]
        mos = [safe_float(row.get("mos", 0.0), 0.0) for row in rows]
        mos_z = robust_z_scores(mos)
        p_pred = [sigmoid(alpha * value) for value in pred_z]
        p_mos = [sigmoid(value) for value in mos_z]
        per_image_mae.append(mean([abs(float(a) - float(b)) for a, b in zip(p_pred, p_mos)]))
        if local_z_valid(pred_z):
            z_valid_pred += 1
        if local_z_valid(mos):
            z_valid_mos += 1
        pooled_topq_targets.extend(binary_top_quantile_labels(mos, top_quantile))
        pooled_topq_probs.extend(p_pred)
    if pooled_topq_targets:
        brier = mean([(float(p) - float(y)) ** 2 for y, p in zip(pooled_topq_targets, pooled_topq_probs)])
        nll = log_loss_binary(pooled_topq_targets, pooled_topq_probs)
        ece = expected_calibration_error(pooled_topq_targets, pooled_topq_probs)
        positive_rate = mean([float(value) for value in pooled_topq_targets])
        const_prob = positive_rate
        const_brier = mean([(const_prob - float(y)) ** 2 for y in pooled_topq_targets])
        const_nll = log_loss_binary(pooled_topq_targets, [const_prob for _ in pooled_topq_targets])
    else:
        brier = 0.0
        nll = 0.0
        ece = 0.0
        positive_rate = 0.0
        const_prob = 0.0
        const_brier = 0.0
        const_nll = 0.0
    auroc = 0.0
    auprc = 0.0
    if pooled_topq_targets and len(set(pooled_topq_targets)) > 1:
        auroc = float(roc_auc_score(pooled_topq_targets, pooled_topq_probs))
        auprc = float(average_precision_score(pooled_topq_targets, pooled_topq_probs))
    return {
        "z_field": z_field,
        "alpha": float(alpha),
        "image_count": len(image_rows),
        "mean_local_prob_mae": mean(per_image_mae),
        "brier_topq": float(brier),
        "nll_topq": float(nll),
        "ece_topq": float(ece),
        "auroc_topq": float(auroc),
        "auprc_topq": float(auprc),
        "topq_positive_rate": float(positive_rate),
        "constant_prior_prob": float(const_prob),
        "constant_prior_brier": float(const_brier),
        "constant_prior_nll": float(const_nll),
        "z_local_valid_rate_pred": float(z_valid_pred / max(1, len(image_rows))),
        "z_local_valid_rate_mos": float(z_valid_mos / max(1, len(image_rows))),
        "z_local_fallback_rate_pred": float(1.0 - (z_valid_pred / max(1, len(image_rows)))),
        "topq_curve": calibration_curve(pooled_topq_targets, pooled_topq_probs),
        "pooled_topq_target_count": len(pooled_topq_targets),
    }


def tune_alpha(
    image_rows: Sequence[Sequence[Dict[str, Any]]],
    *,
    alpha_candidates: Sequence[float],
    top_quantile: float,
    z_field: str = "score_z_local",
) -> Tuple[float, List[Dict[str, Any]]]:
    summaries = [
        compute_local_probability_summary(
            image_rows=image_rows,
            alpha=float(alpha),
            top_quantile=top_quantile,
            z_field=z_field,
        )
        for alpha in alpha_candidates
    ]
    if not summaries:
        return 1.0, []
    best = min(
        summaries,
        key=lambda item: (
            float(item["brier_topq"]),
            float(item["nll_topq"]),
            float(item["mean_local_prob_mae"]),
        ),
    )
    return float(best["alpha"]), summaries


def fit_isotonic_calibrator(
    rows: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    z_field: str = "score_z_local",
) -> IsotonicRegression:
    model = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip")
    features = [sigmoid(alpha * safe_float(row.get(z_field, 0.0), 0.0)) for row in rows]
    targets = [safe_float(row.get("mos", 0.0), 0.0) for row in rows]
    model.fit(features, targets)
    return model


def evaluate_isotonic(
    model: IsotonicRegression,
    rows: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    z_field: str = "score_z_local",
) -> Dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "pearson_r": 0.0,
            "ccc": 0.0,
            "qwk": 0.0,
        }
    features = np.asarray([sigmoid(alpha * safe_float(row.get(z_field, 0.0), 0.0)) for row in rows], dtype=np.float64)
    targets = [safe_float(row.get("mos", 0.0), 0.0) for row in rows]
    pred = model.predict(features).tolist()
    return {
        "sample_count": len(rows),
        "mae": mean([abs(float(a) - float(b)) for a, b in zip(targets, pred)]),
        "rmse": root_mean_squared_error(targets, pred),
        "pearson_r": pearson_corr(targets, pred),
        "ccc": concordance_correlation_coefficient(targets, pred),
        "qwk": quadratic_weighted_kappa(targets, pred),
        "predictions": pred,
    }


def compute_candidate_coverage(
    production_candidates: Sequence[Dict[str, Any]],
    anns: Sequence[Dict[str, Any]],
    *,
    width: int,
    height: int,
    top_quantile: float,
) -> Dict[str, Any]:
    if not production_candidates or not anns:
        return {
            "top1_oracle_iou": 0.0,
            "top1_coverage_at_05": 0.0,
            "top1_coverage_at_07": 0.0,
            "topq_any_coverage_at_05": 0.0,
        }
    prod_boxes = [candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]) for candidate in production_candidates]
    gt_boxes = [gt_bbox_norm_xyxy(ann, width, height) for ann in anns]
    gt_scores = [safe_float(ann.get("score", 0.0), 0.0) for ann in anns]
    top1_boxes = [gt_boxes[idx] for idx in top_indices(gt_scores, 1)]
    topq_boxes = [gt_boxes[idx] for idx in top_indices(gt_scores, top_k_from_quantile(len(gt_boxes), top_quantile))]
    top1_oracle_iou = max(iou_xyxy(prod_box, gt_box) for prod_box in prod_boxes for gt_box in top1_boxes)
    topq_oracle_iou = max(iou_xyxy(prod_box, gt_box) for prod_box in prod_boxes for gt_box in topq_boxes)
    return {
        "top1_oracle_iou": float(top1_oracle_iou),
        "top1_coverage_at_05": 1.0 if top1_oracle_iou >= 0.5 else 0.0,
        "top1_coverage_at_07": 1.0 if top1_oracle_iou >= 0.7 else 0.0,
        "topq_any_coverage_at_05": 1.0 if topq_oracle_iou >= 0.5 else 0.0,
    }


def summarize_prod_selection(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "image_count": 0,
            "non_gt_winner_rate": 0.0,
            "exact_gt_winner_rate": 0.0,
            "gt_best_iou_mean": 0.0,
            "gt_best_hit_at_05": 0.0,
            "gt_best_hit_at_07": 0.0,
            "matched_gt_iou_mean": 0.0,
            "matched_gt_mos_percentile_mean": 0.0,
            "matched_gt_topq_rate": 0.0,
            "gt_vs_prod_disagreement_rate": 0.0,
            "winner_source_family_counts": {},
            "winner_source_family_topq_rate": {},
        }
    source_counts = Counter(str(row.get("winner_source_family", "unknown")) for row in rows)
    source_hits = Counter(str(row.get("winner_source_family", "unknown")) for row in rows if bool(row.get("matched_gt_is_topq", False)))
    family_topq_rate = {
        key: round(float(source_hits.get(key, 0)) / float(max(1, count)), 6)
        for key, count in source_counts.items()
    }
    return {
        "image_count": len(rows),
        "non_gt_winner_rate": round(mean([0.0 if bool(row.get("is_gt_candidate", False)) else 1.0 for row in rows]), 6),
        "exact_gt_winner_rate": round(mean([1.0 if bool(row.get("is_gt_candidate", False)) else 0.0 for row in rows]), 6),
        "gt_best_iou_mean": round(mean([safe_float(row.get("iou_to_gt_best", 0.0), 0.0) for row in rows]), 6),
        "gt_best_hit_at_05": round(mean([1.0 if safe_float(row.get("iou_to_gt_best", 0.0), 0.0) >= 0.5 else 0.0 for row in rows]), 6),
        "gt_best_hit_at_07": round(mean([1.0 if safe_float(row.get("iou_to_gt_best", 0.0), 0.0) >= 0.7 else 0.0 for row in rows]), 6),
        "matched_gt_iou_mean": round(mean([safe_float(row.get("matched_gt_iou", 0.0), 0.0) for row in rows]), 6),
        "matched_gt_mos_percentile_mean": round(mean([safe_float(row.get("matched_gt_mos_percentile", 0.0), 0.0) for row in rows]), 6),
        "matched_gt_topq_rate": round(mean([1.0 if bool(row.get("matched_gt_is_topq", False)) else 0.0 for row in rows]), 6),
        "gt_vs_prod_disagreement_rate": round(mean([1.0 if bool(row.get("gt_best_candidate_id") != row.get("candidate_id")) else 0.0 for row in rows]), 6),
        "winner_source_family_counts": dict(sorted(source_counts.items())),
        "winner_source_family_topq_rate": family_topq_rate,
    }


def summarize_prod_selection_pairwise(
    rank_rows: Sequence[Dict[str, Any]],
    policy_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    paired = list(zip(rank_rows, policy_rows))
    if not paired:
        return {
            "image_count": 0,
            "policy_vs_rank_disagreement_rate": 0.0,
            "rank_better_iou_rate": 0.0,
            "policy_better_iou_rate": 0.0,
            "rank_better_percentile_rate": 0.0,
            "policy_better_percentile_rate": 0.0,
        }
    return {
        "image_count": len(paired),
        "policy_vs_rank_disagreement_rate": round(
            mean([1.0 if str(rank_row.get("candidate_id", "")) != str(policy_row.get("candidate_id", "")) else 0.0 for rank_row, policy_row in paired]),
            6,
        ),
        "rank_better_iou_rate": round(
            mean([1.0 if safe_float(rank_row.get("iou_to_gt_best", 0.0), 0.0) > safe_float(policy_row.get("iou_to_gt_best", 0.0), 0.0) else 0.0 for rank_row, policy_row in paired]),
            6,
        ),
        "policy_better_iou_rate": round(
            mean([1.0 if safe_float(policy_row.get("iou_to_gt_best", 0.0), 0.0) > safe_float(rank_row.get("iou_to_gt_best", 0.0), 0.0) else 0.0 for rank_row, policy_row in paired]),
            6,
        ),
        "rank_better_percentile_rate": round(
            mean([1.0 if safe_float(rank_row.get("matched_gt_mos_percentile", 0.0), 0.0) > safe_float(policy_row.get("matched_gt_mos_percentile", 0.0), 0.0) else 0.0 for rank_row, policy_row in paired]),
            6,
        ),
        "policy_better_percentile_rate": round(
            mean([1.0 if safe_float(policy_row.get("matched_gt_mos_percentile", 0.0), 0.0) > safe_float(rank_row.get("matched_gt_mos_percentile", 0.0), 0.0) else 0.0 for rank_row, policy_row in paired]),
            6,
        ),
    }


def summarize_score_modes(
    *,
    image_records: Sequence[Dict[str, Any]],
    protocol: str,
    alpha: float,
    top_quantile: float,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total = max(1, len(image_records))
    for record in image_records:
        grouped[str(record.get("score_mode", "") or "missing")].append(record)
    out: Dict[str, Any] = {}
    for mode_name, rows in sorted(grouped.items()):
        trend_rows = [record[protocol]["metrics_by_field"]["score_rank"] for record in rows]
        local_prob = compute_local_probability_summary(
            [record[protocol]["rows"] for record in rows],
            alpha=alpha,
            top_quantile=top_quantile,
        )
        prod_rows = [record["Ge"]["production_best"] for record in rows if isinstance(record.get("Ge", {}).get("production_best"), dict)]
        out[mode_name] = {
            "image_count": len(rows),
            "share": round(float(len(rows)) / float(total), 6),
            "mean_spearman": round(mean([safe_float(row.get("spearman", 0.0), 0.0) for row in trend_rows]), 6),
            "mean_weighted_pair_acc": round(mean([safe_float(row.get("weighted_pair_acc", 0.0), 0.0) for row in trend_rows]), 6),
            "mean_hit_at_1": round(mean([safe_float(row.get("hit_at_1", 0.0), 0.0) for row in trend_rows]), 6),
            "brier_topq": round(safe_float(local_prob.get("brier_topq", 0.0), 0.0), 6),
            "ece_topq": round(safe_float(local_prob.get("ece_topq", 0.0), 0.0), 6),
            "prod_gt_best_iou_mean": round(mean([safe_float(row.get("iou_to_gt_best", 0.0), 0.0) for row in prod_rows]), 6) if prod_rows else 0.0,
            "prod_matched_gt_percentile_mean": round(mean([safe_float(row.get("matched_gt_mos_percentile", 0.0), 0.0) for row in prod_rows]), 6) if prod_rows else 0.0,
        }
    return out


def summarize_failure_cohorts(
    *,
    image_records: Sequence[Dict[str, Any]],
    protocol: str,
    alpha: float,
    top_quantile: float,
) -> Dict[str, Any]:
    cohort_defs = (
        ("subject_mode_conf_lt_0_5", lambda rec: safe_float(rec.get("subject_mode_conf", 0.0), 0.0) < 0.5),
        ("raw_anchor_area_lt_0_03", lambda rec: safe_float(rec.get("raw_anchor_area", 0.0), 0.0) < 0.03),
        ("subject_reliability_lt_0_4", lambda rec: safe_float(rec.get("subject_reliability", 0.0), 0.0) < 0.4),
        ("subject_repr_soft_or_guard", lambda rec: str(rec.get("subject_repr_type", "")) in {"soft_scene_region", "guard_proxy"}),
        ("neutralized", lambda rec: bool(rec.get("subject_terms_neutralized", False))),
        (
            "severe_disagreement",
            lambda rec: bool(rec.get("subject_disagreement_severe", False))
            or (
                safe_float(rec.get("subject_agreement_iou", 1.0), 1.0) < 0.10
                and safe_float(rec.get("subject_center_distance", 0.0), 0.0) > 0.20
            ),
        ),
        ("support_fallback", lambda rec: bool(str(rec.get("support_fallback_reason", "")).strip())),
        ("scene_general", lambda rec: str(rec.get("subject_mode", "")) == "scene_general"),
        ("other_ambiguous", lambda rec: str(rec.get("subject_mode", "")) == "other_ambiguous"),
        ("blank_ratio_gt_0_95", lambda rec: safe_float(rec.get("blank_ratio_est", 0.0), 0.0) > 0.95),
        ("winner_baseline_full", lambda rec: str(rec.get("ge_winner_source_family", "")) == "baseline_full"),
        ("winner_baseline_maxarea_subject", lambda rec: str(rec.get("ge_winner_source_family", "")) == "baseline_maxarea_subject"),
        ("winner_object_template", lambda rec: str(rec.get("ge_winner_source_family", "")) == "object_template"),
        ("winner_saliency_jitter", lambda rec: str(rec.get("ge_winner_source_family", "")) == "saliency_jitter"),
        ("winner_teacher", lambda rec: str(rec.get("ge_winner_source_family", "")) == "teacher"),
        ("policy_vs_rank_disagree", lambda rec: bool(rec.get("ge_policy_rank_disagree", False))),
    )
    out: Dict[str, Any] = {}
    for name, predicate in cohort_defs:
        rows = [record for record in image_records if predicate(record)]
        if not rows:
            out[name] = {"image_count": 0}
            continue
        trend_rows = [record[protocol]["metrics_by_field"]["score_rank"] for record in rows]
        local_prob = compute_local_probability_summary(
            [record[protocol]["rows"] for record in rows],
            alpha=alpha,
            top_quantile=top_quantile,
        )
        prod_rows = [record["Ge"]["production_best"] for record in rows if isinstance(record.get("Ge", {}).get("production_best"), dict)]
        out[name] = {
            "image_count": len(rows),
            "mean_spearman": round(mean([safe_float(row.get("spearman", 0.0), 0.0) for row in trend_rows]), 6),
            "mean_weighted_pair_acc": round(mean([safe_float(row.get("weighted_pair_acc", 0.0), 0.0) for row in trend_rows]), 6),
            "mean_hit_at_1": round(mean([safe_float(row.get("hit_at_1", 0.0), 0.0) for row in trend_rows]), 6),
            "brier_topq": round(safe_float(local_prob.get("brier_topq", 0.0), 0.0), 6),
            "ece_topq": round(safe_float(local_prob.get("ece_topq", 0.0), 0.0), 6),
            "prod_gt_best_iou_mean": round(mean([safe_float(row.get("iou_to_gt_best", 0.0), 0.0) for row in prod_rows]), 6) if prod_rows else 0.0,
            "prod_matched_gt_percentile_mean": round(mean([safe_float(row.get("matched_gt_mos_percentile", 0.0), 0.0) for row in prod_rows]), 6) if prod_rows else 0.0,
        }
    return out


def summarize_subject_support_diagnostics(image_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def _bucket(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"image_count": 0}
        return {
            "image_count": int(len(rows)),
            "gt_best_mass_recall_mean": round(mean([safe_float(row["subject_support_metrics"]["gt_best"]["mass_recall"], 0.0) for row in rows]), 6),
            "raw_iou_mean": round(mean([safe_float(row["subject_support_metrics"]["raw_anchor"]["iou_to_gt_best"], 0.0) for row in rows]), 6),
            "candidate_iou_mean": round(mean([safe_float(row["subject_support_metrics"]["candidate_anchor"]["iou_to_gt_best"], 0.0) for row in rows]), 6),
            "support_iou_mean": round(mean([safe_float(row["subject_support_metrics"]["support_region"]["iou_to_gt_best"], 0.0) for row in rows]), 6),
            "raw_mass_recall_mean": round(mean([safe_float(row["subject_support_metrics"]["raw_anchor"]["mass_recall"], 0.0) for row in rows]), 6),
            "candidate_mass_recall_mean": round(mean([safe_float(row["subject_support_metrics"]["candidate_anchor"]["mass_recall"], 0.0) for row in rows]), 6),
            "support_mass_recall_mean": round(mean([safe_float(row["subject_support_metrics"]["support_region"]["mass_recall"], 0.0) for row in rows]), 6),
            "support_low_mass_rate": round(mean([1.0 if safe_float(row["subject_support_metrics"]["support_region"]["mass_recall"], 0.0) < 0.5 else 0.0 for row in rows]), 6),
            "support_centroid_inside_rate": round(mean([safe_float(row["subject_support_metrics"]["support_region"]["centroid_inside"], 0.0) for row in rows]), 6),
            "support_centroid_distance_mean": round(mean([safe_float(row["subject_support_metrics"]["support_region"]["centroid_distance"], 0.0) for row in rows]), 6),
            "support_latent_bbox_iou_mean": round(mean([safe_float(row["subject_support_metrics"]["support_region"]["latent_bbox_iou"], 0.0) for row in rows]), 6),
        }

    scene_rows = [row for row in image_records if str(row.get("subject_mode", "")) == "scene_general"]
    ambiguous_tiny_rows = [
        row
        for row in image_records
        if str(row.get("subject_mode", "")) == "other_ambiguous" and safe_float(row.get("c2_primary_area_ratio", 0.0), 0.0) < 0.04
    ]
    neutralized_rows = [row for row in image_records if bool(row.get("subject_terms_neutralized", False))]
    return {
        "overall": _bucket(image_records),
        "scene_general": _bucket(scene_rows),
        "other_ambiguous_tiny": _bucket(ambiguous_tiny_rows),
        "neutralized": _bucket(neutralized_rows),
    }


def image_path_for(image_dir: Path, file_name: str, image_id: str) -> Optional[Path]:
    direct = image_dir / file_name
    if direct.exists():
        return direct
    stem = Path(file_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    fallback = image_dir / file_name
    return fallback if fallback.exists() else None


def collect_subject_visuals(
    *,
    cand_rec: Dict[str, Any],
    feat_rec: Dict[str, Any],
    route: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    subject_prior = cand_rec.get("subject_prior", {}) if isinstance(cand_rec.get("subject_prior"), dict) else {}
    route = route if isinstance(route, dict) else {}
    effective_subject_region = (
        route.get("effective_subject_region", {})
        if isinstance(route.get("effective_subject_region"), dict)
        else (
            subject_prior.get("effective_subject_region", {})
            if isinstance(subject_prior.get("effective_subject_region"), dict)
            else {}
        )
    )
    saliency = feat_rec.get("c7_saliency", {}) if isinstance(feat_rec.get("c7_saliency"), dict) else {}
    if not saliency and isinstance(effective_subject_region.get("saliency_summary"), dict):
        saliency = effective_subject_region.get("saliency_summary", {})
    raw_anchor = optional_norm_box(
        subject_prior.get("anchor_bbox_norm_xyxy")
        or effective_subject_region.get("raw_anchor_bbox_norm_xyxy")
    )
    candidate_anchor = optional_norm_box(
        subject_prior.get("candidate_anchor_bbox_norm_xyxy")
        or effective_subject_region.get("candidate_anchor_bbox_norm_xyxy")
        or raw_anchor
    )
    scoring_region = optional_norm_box(
        effective_subject_region.get("scoring_bbox_norm_xyxy")
        or effective_subject_region.get("effective_bbox_norm_xyxy")
        or subject_prior.get("effective_bbox_norm_xyxy")
        or subject_prior.get("bbox_norm_xyxy")
    )
    support_region = optional_norm_box(
        effective_subject_region.get("support_bbox_norm_xyxy")
        or scoring_region
    )
    return {
        "raw_anchor_norm_xyxy": raw_anchor,
        "candidate_anchor_norm_xyxy": candidate_anchor,
        "scoring_region_norm_xyxy": scoring_region,
        "support_region_norm_xyxy": support_region,
        "subject_repr_type": str(effective_subject_region.get("subject_repr_type", subject_prior.get("subject_repr_type", "")) or ""),
        "subject_reliability": round(safe_float(effective_subject_region.get("subject_reliability", subject_prior.get("subject_reliability", 0.0)), 0.0), 6),
        "subject_placeholder_flag": bool(effective_subject_region.get("placeholder_flag", subject_prior.get("placeholder_flag", False))),
        "score_mode": str(effective_subject_region.get("score_mode", route.get("flags", {}).get("subject_score_mode", "")) or ""),
        "state": str(effective_subject_region.get("state", route.get("flags", {}).get("subject_effective_state", "")) or ""),
        "source": str(effective_subject_region.get("source", subject_prior.get("source", "")) or ""),
        "subject_source": str(effective_subject_region.get("subject_source", "") or ""),
        "attention_layout": str(effective_subject_region.get("attention_layout", "") or ""),
        "support_kind": str(effective_subject_region.get("support_kind", "") or ""),
        "support_map_enabled": bool(effective_subject_region.get("support_map_enabled", False)),
        "subject_agreement_iou": round(safe_float(effective_subject_region.get("subject_agreement_iou", 0.0), 0.0), 6),
        "subject_center_distance": round(safe_float(effective_subject_region.get("subject_center_distance", 0.0), 0.0), 6),
        "subject_disagreement_severe": bool(effective_subject_region.get("subject_disagreement_severe", False)),
        "raw_anchor_support_mass": round(safe_float(effective_subject_region.get("raw_anchor_support_mass", 0.0), 0.0), 6),
        "candidate_anchor_support_mass": round(safe_float(effective_subject_region.get("candidate_anchor_support_mass", 0.0), 0.0), 6),
        "effective_support_mass": round(safe_float(effective_subject_region.get("effective_support_mass", 0.0), 0.0), 6),
        "support_fallback_reason": str(effective_subject_region.get("support_fallback_reason", "") or ""),
        "saliency_backend": str(saliency.get("backend", "none") or "none"),
        "saliency_support_norm_xyxy": optional_norm_box(saliency.get("support_bbox_norm_xyxy")),
        "saliency_top1_norm_xyxy": optional_norm_box(saliency.get("top_component_bbox_norm_xyxy")),
        "saliency_top2_norm_xyxy": optional_norm_box(saliency.get("top2_union_bbox_norm_xyxy")),
        "saliency_foreground_norm_xyxy": optional_norm_box(saliency.get("foreground_bbox_norm_xyxy")),
        "saliency_weighted_centroid_xy_norm": (
            [round(float(v), 6) for v in saliency.get("weighted_centroid_xy_norm", [])]
            if isinstance(saliency.get("weighted_centroid_xy_norm"), (list, tuple)) and len(saliency.get("weighted_centroid_xy_norm")) == 2
            else None
        ),
        "saliency_foreground_area_ratio": round(safe_float(saliency.get("foreground_area_ratio", 0.0), 0.0), 6),
        "saliency_blank_ratio_est": round(safe_float(saliency.get("blank_ratio_est", 0.0), 0.0), 6),
        "saliency_dominance_score": round(safe_float(saliency.get("dominance_score", 0.0), 0.0), 6),
        "saliency_component_count": safe_int(saliency.get("component_count", 0)),
    }


def draw_box(draw: ImageDraw.ImageDraw, bbox: Sequence[float], width: int, height: int, color: Tuple[int, int, int], label: str, y_offset: int = 0) -> None:
    x1 = int(round(float(bbox[0]) * width))
    y1 = int(round(float(bbox[1]) * height))
    x2 = int(round(float(bbox[2]) * width))
    y2 = int(round(float(bbox[3]) * height))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    label_lines = [line for line in str(label).split("\n") if line] or [""]
    label_w = max(110, 8 * max(len(line) for line in label_lines))
    label_h = 20 * len(label_lines) + 4
    label_top = max(0, y1 - label_h + y_offset)
    label_bottom = max(0, y1 + y_offset)
    draw.rectangle([x1, label_top, min(width, x1 + label_w), label_bottom], fill=color)
    draw.multiline_text((x1 + 4, label_top + 2), "\n".join(label_lines), fill=(255, 255, 255), spacing=2)


def select_sample_image_ids(image_records: Sequence[Dict[str, Any]], sample_count: int) -> List[str]:
    if not image_records:
        return []
    selected: List[str] = []
    lookup = {str(row["image_id"]): row for row in image_records}
    for image_id in SAMPLE_PRIORITY_IDS:
        if image_id in lookup and image_id not in selected:
            selected.append(image_id)
        if len(selected) >= max(1, sample_count):
            return selected[:sample_count]
    ranked = sorted(image_records, key=lambda row: float(row["Ge"]["metrics_by_field"]["score_rank"]["spearman"]))
    candidates = [
        ranked[0],
        ranked[len(ranked) // 2],
        ranked[-1],
        max(image_records, key=lambda row: abs(float(row["Gc"]["metrics_by_field"]["score_rank"]["spearman"]) - float(row["Ge"]["metrics_by_field"]["score_rank"]["spearman"]))),
        max(image_records, key=lambda row: float(row["coverage"]["top1_oracle_iou"])),
        min(image_records, key=lambda row: float(row["coverage"]["top1_oracle_iou"])),
    ]
    for row in candidates:
        image_id = str(row["image_id"])
        if image_id not in selected:
            selected.append(image_id)
        if len(selected) >= max(1, sample_count):
            return selected[:sample_count]
    for row in ranked:
        image_id = str(row["image_id"])
        if image_id not in selected:
            selected.append(image_id)
        if len(selected) >= max(1, sample_count):
            break
    return selected[:sample_count]


def render_sample_overlay(
    image_record: Dict[str, Any],
    image_path: Path,
    out_path: Path,
    *,
    saliency_mask: Optional[np.ndarray] = None,
) -> None:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    canvas = overlay_mask(canvas, saliency_mask, (241, 196, 15), alpha=0.22)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    gc_best = image_record["Gc"]["best_gt_row"]
    ge_best = image_record["Ge"]["best_gt_row"]
    ge_prod = image_record["Ge"]["production_best"]
    gt_best = image_record["gt_best_row"]
    subject_visuals = image_record.get("subject_visuals", {}) if isinstance(image_record.get("subject_visuals"), dict) else {}

    raw_anchor = subject_visuals.get("raw_anchor_norm_xyxy")
    candidate_anchor = subject_visuals.get("candidate_anchor_norm_xyxy")
    scoring_region = subject_visuals.get("scoring_region_norm_xyxy")
    if raw_anchor is not None:
        draw_box(draw, raw_anchor, width, height, (0, 188, 212), "raw anchor", 0)
    if candidate_anchor is not None and not same_box(candidate_anchor, raw_anchor):
        draw_box(draw, candidate_anchor, width, height, (214, 48, 128), "candidate anchor", 24)
    if scoring_region is not None and not same_box(scoring_region, raw_anchor) and not same_box(scoring_region, candidate_anchor):
        draw_box(draw, scoring_region, width, height, (102, 52, 156), "scoring region", 48)

    draw_box(draw, gt_best["bbox_norm_xyxy"], width, height, (46, 204, 113), crop_overlay_label("GT MOS best", gt_best), 72)
    draw_box(draw, gc_best["bbox_norm_xyxy"], width, height, (52, 152, 219), crop_overlay_label("Gc best", gc_best), 126)
    draw_box(draw, ge_best["bbox_norm_xyxy"], width, height, (243, 156, 18), crop_overlay_label("Ge GT best", ge_best), 180)
    draw_box(draw, ge_prod["bbox_norm_xyxy"], width, height, (231, 76, 60), crop_overlay_label("Ge prod best", ge_prod), 234)

    panel_height = 148
    panel = Image.new("RGB", (width, panel_height), (18, 23, 29))
    panel_draw = ImageDraw.Draw(panel)
    lines = [
        f"image_id={image_record['image_id']} split={image_record['official_split']} mode={image_record['subject_mode']}",
        f"Gc spearman={image_record['Gc']['metrics_by_field']['score_rank']['spearman']:.3f}  Ge spearman={image_record['Ge']['metrics_by_field']['score_rank']['spearman']:.3f}",
        f"coverage top1_iou={image_record['coverage']['top1_oracle_iou']:.3f}  cov@0.5={image_record['coverage']['top1_coverage_at_05']:.0f}  cov@0.7={image_record['coverage']['top1_coverage_at_07']:.0f}",
        f"GT best MOS={gt_best['mos']:.2f}  Gc best MOS={gc_best['mos']:.2f}  Ge GT best MOS={ge_best['mos']:.2f}",
        (
            f"repr={subject_visuals.get('subject_repr_type', 'na')}  reliability={safe_float(subject_visuals.get('subject_reliability', 0.0), 0.0):.2f}  "
            f"score_mode={subject_visuals.get('score_mode', 'na')}  support_map={bool(subject_visuals.get('support_map_enabled', False))}  saliency_mask={subject_visuals.get('saliency_backend', 'none')}"
        ),
    ]
    for idx, line in enumerate(lines):
        panel_draw.text((12, 12 + idx * 24), line, fill=(245, 247, 250))
    stacked = Image.new("RGB", (width, height + panel_height), (0, 0, 0))
    stacked.paste(panel, (0, 0))
    stacked.paste(canvas, (0, panel_height))
    ensure_dir(out_path.parent)
    stacked.save(out_path)


def first_face_center_norm(c3_info: Dict[str, Any]) -> Optional[List[float]]:
    face_boxes = c3_info.get("face_boxes", []) if isinstance(c3_info, dict) else []
    if not isinstance(face_boxes, list) or not face_boxes:
        return None
    face_box = face_boxes[0]
    if not isinstance(face_box, (list, tuple)) or len(face_box) != 4:
        return None
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in face_box]
    return [round(0.5 * (x1 + x2), 6), round(0.5 * (y1 + y2), 6)]


def color_box_with_label(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    width: int,
    height: int,
    color: Tuple[int, int, int],
    label: str,
    *,
    y_offset: int = 0,
    box_width: int = 4,
) -> None:
    x1 = int(round(float(bbox[0]) * width))
    y1 = int(round(float(bbox[1]) * height))
    x2 = int(round(float(bbox[2]) * width))
    y2 = int(round(float(bbox[3]) * height))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
    label_lines = [line for line in str(label).split("\n") if line] or [""]
    label_w = max(110, 8 * max(len(line) for line in label_lines))
    label_h = 20 * len(label_lines) + 4
    label_top = max(0, y1 - label_h + y_offset)
    label_bottom = max(0, y1 + y_offset)
    draw.rectangle([x1, label_top, min(width, x1 + label_w), label_bottom], fill=color)
    draw.multiline_text((x1 + 4, label_top + 2), "\n".join(label_lines), fill=(255, 255, 255), spacing=2)


def saliency_priority_for_visual(subject_visuals: Dict[str, Any]) -> str:
    backend = str(subject_visuals.get("saliency_backend", "none") or "none").lower()
    if backend == "birefnet":
        return "quality_first"
    return "high_efficiency"


class SampleSaliencyMaskResolver:
    def __init__(self) -> None:
        self._extractors: Dict[str, SaliencyFeatureExtractor] = {}
        self._cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    def close(self) -> None:
        for extractor in self._extractors.values():
            extractor.close()
        self._extractors.clear()
        self._cache.clear()

    def resolve(self, image_path: Path, subject_visuals: Dict[str, Any]) -> Optional[np.ndarray]:
        backend = str(subject_visuals.get("saliency_backend", "none") or "none").lower()
        if backend in {"", "none"}:
            return None
        priority = saliency_priority_for_visual(subject_visuals)
        cache_key = (str(image_path), priority)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            extractor = self._extractors.get(priority)
            if extractor is None:
                extractor = SaliencyFeatureExtractor(priority=priority)
                self._extractors[priority] = extractor
            with Image.open(image_path) as image:
                _, mask = extractor.process_image_with_mask(image.convert("RGB"))
            self._cache[cache_key] = mask.astype(bool)
        except BaseException:
            self._cache[cache_key] = None
        return self._cache[cache_key]


def overlay_mask(canvas: Image.Image, mask: Optional[np.ndarray], color: Tuple[int, int, int], alpha: float = 0.24) -> Image.Image:
    if mask is None:
        return canvas
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        return canvas
    height, width = mask_bool.shape
    rgba = canvas.convert("RGBA")
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask_bool] = [int(color[0]), int(color[1]), int(color[2]), int(round(255.0 * alpha))]
    blended = Image.alpha_composite(rgba, Image.fromarray(overlay, mode="RGBA"))
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_draw = ImageDraw.Draw(blended)
    for contour in contours:
        pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(pts) >= 2:
            contour_draw.line(pts + [pts[0]], fill=color + (255,), width=2)
    return blended.convert("RGB")


def crop_overlay_label(prefix: str, row: Dict[str, Any]) -> str:
    is_gt = row.get("mos") is not None
    sigz = safe_float(
        row.get("score_sigmoid_z_local", row.get("pseudo_prob_rank_local", 0.0)),
        0.0,
    )
    rank = safe_float(row.get("score_rank", 0.0), 0.0)
    if is_gt:
        mos = safe_float(row.get("mos", 0.0), 0.0)
        return f"{prefix}\nMOS={mos:.2f} sigz={sigz:.3f} rank={rank:.3f}"
    return f"{prefix}\nsigz={sigz:.3f} rank={rank:.3f}"


def render_sample_diagnostic_overlay(
    *,
    image_path: Path,
    out_path: Path,
    image_id: str,
    official_split: str,
    subject_mode: str,
    router_rule_id: str,
    subject_visuals: Dict[str, Any],
    face_center_norm: Optional[Sequence[float]],
    gt_mos_best: Dict[str, Any],
    ge_gt_best: Dict[str, Any],
    ge_prod_best: Dict[str, Any],
    rank_gap: float,
    policy_gap: float,
    saliency_mask: Optional[np.ndarray] = None,
) -> None:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    canvas = overlay_mask(canvas, saliency_mask, (255, 193, 7), alpha=0.22)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)

    raw_anchor = subject_visuals.get("raw_anchor_norm_xyxy")
    candidate_anchor = subject_visuals.get("candidate_anchor_norm_xyxy")
    scoring_region = subject_visuals.get("scoring_region_norm_xyxy")
    if raw_anchor is not None:
        color_box_with_label(
            draw,
            raw_anchor,
            width,
            height,
            (0, 188, 212),
            "raw anchor",
            y_offset=0,
            box_width=2,
        )
    if candidate_anchor is not None and not same_box(candidate_anchor, raw_anchor):
        color_box_with_label(
            draw,
            candidate_anchor,
            width,
            height,
            (214, 48, 128),
            "candidate anchor",
            y_offset=18,
            box_width=2,
        )
    if scoring_region is not None and not same_box(scoring_region, candidate_anchor) and not same_box(scoring_region, raw_anchor):
        color_box_with_label(
            draw,
            scoring_region,
            width,
            height,
            (102, 52, 156),
            "scoring region",
            y_offset=36,
            box_width=2,
        )
    if face_center_norm is not None:
        fx = int(round(float(face_center_norm[0]) * width))
        fy = int(round(float(face_center_norm[1]) * height))
        draw.ellipse((fx - 5, fy - 5, fx + 5, fy + 5), fill=(236, 64, 122), outline=(255, 255, 255), width=1)

    color_box_with_label(
        draw,
        gt_mos_best["bbox_norm_xyxy"],
        width,
        height,
        (46, 204, 113),
        crop_overlay_label("GT MOS best", gt_mos_best),
        y_offset=54,
    )
    color_box_with_label(
        draw,
        ge_gt_best["bbox_norm_xyxy"],
        width,
        height,
        (243, 156, 18),
        crop_overlay_label("Ge GT best", ge_gt_best),
        y_offset=126,
    )
    color_box_with_label(
        draw,
        ge_prod_best["bbox_norm_xyxy"],
        width,
        height,
        (231, 76, 60),
        crop_overlay_label("Ge prod best", ge_prod_best),
        y_offset=198,
    )

    panel_height = 220
    panel = Image.new("RGB", (width, panel_height), (18, 23, 29))
    panel_draw = ImageDraw.Draw(panel)
    lines = [
        f"image_id={image_id} split={official_split} mode={subject_mode} route={router_rule_id}",
        (
            f"GT MOS best={safe_float(gt_mos_best.get('mos', 0.0), 0.0):.2f}  "
            f"Ge GT rank={safe_float(ge_gt_best.get('score_rank', 0.0), 0.0):.6f}  "
            f"Ge prod rank={safe_float(ge_prod_best.get('score_rank', 0.0), 0.0):.6f}"
        ),
        (
            f"delta rank(prod-gt)={rank_gap:+.6f}  "
            f"delta policy(prod-gt)={policy_gap:+.6f}"
        ),
        (
            f"repr={subject_visuals.get('subject_repr_type', 'na')}  src={subject_visuals.get('subject_source', 'na')}  "
            f"layout={subject_visuals.get('attention_layout', 'na')}  reliability={safe_float(subject_visuals.get('subject_reliability', 0.0), 0.0):.2f}"
        ),
        (
            f"state={subject_visuals.get('state', 'na')}  support={subject_visuals.get('support_kind', 'na')}  "
            f"score_mode={subject_visuals.get('score_mode', 'na')}  raw-sal-ioU={safe_float(subject_visuals.get('subject_agreement_iou', 0.0), 0.0):.3f}"
        ),
        (
            f"saliency backend={subject_visuals.get('saliency_backend', 'none')}  fg={safe_float(subject_visuals.get('saliency_foreground_area_ratio', 0.0), 0.0):.3f}  "
            f"blank={safe_float(subject_visuals.get('saliency_blank_ratio_est', 0.0), 0.0):.3f}  dom={safe_float(subject_visuals.get('saliency_dominance_score', 0.0), 0.0):.3f}"
        ),
        "cyan=raw anchor  pink=candidate anchor  purple=scoring region  yellow mask=c7 saliency  green/blue/orange/red=GT/Gc/Ge 결과",
    ]
    for idx, line in enumerate(lines):
        panel_draw.text((12, 12 + idx * 28), line, fill=(245, 247, 250))

    stacked = Image.new("RGB", (width, height + panel_height), (0, 0, 0))
    stacked.paste(panel, (0, 0))
    stacked.paste(canvas, (0, panel_height))
    ensure_dir(out_path.parent)
    stacked.save(out_path)


def compact_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "na"
    return f"{float(value):.{digits}f}"


def summarize_sample_candidate(
    *,
    row: Dict[str, Any],
    role: str,
    mos: Optional[float],
    scoring_region_norm: Optional[Sequence[float]],
) -> Dict[str, Any]:
    components = row.get("scores", {}).get("components", {}) if isinstance(row.get("scores", {}), dict) else {}
    macro_scores = row.get("macro_scores", {}) if isinstance(row.get("macro_scores"), dict) else {}
    comp_checks = row.get("composition_checks", {}) if isinstance(row.get("composition_checks"), dict) else {}
    subject_iou = None
    if scoring_region_norm is not None:
        subject_iou = iou_xyxy(row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), scoring_region_norm)
    return {
        "role": role,
        "candidate_id": str(row.get("candidate_id", "")),
        "source": str(row.get("source", "")),
        "bbox_norm_xyxy": [round(float(v), 6) for v in row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])],
        "mos": (None if mos is None else round(float(mos), 6)),
        "score_rank": round(safe_float(row.get("scores", {}).get("rank", row.get("scores", {}).get("final", 0.0)), 0.0), 6),
        "score_policy": round(safe_float(row.get("scores", {}).get("policy", row.get("scores", {}).get("final", 0.0)), 0.0), 6),
        "score_sigmoid_z_local": round(
            safe_float(
                row.get("score_sigmoid_z_local", row.get("pseudo_prob_rank_local", sigmoid(safe_float(row.get("score_z_local", 0.0), 0.0)))),
                0.0,
            ),
            6,
        ),
        "score_rank_pct": round(safe_float(row.get("score_rank_pct", 0.0), 0.0), 6),
        "score_z_local": round(safe_float(row.get("score_z_local", 0.0), 0.0), 6),
        "score_softmax_local": round(safe_float(row.get("score_softmax_local", 0.0), 0.0), 6),
        "area_log_prior": round(safe_float(row.get("scores", {}).get("area_log_prior", 0.0), 0.0), 6),
        "macro_scores": {
            "A_macro": macro_scores.get("A_macro"),
            "S_macro": macro_scores.get("S_macro"),
            "C_macro": macro_scores.get("C_macro"),
            "T_macro": macro_scores.get("T_macro"),
        },
        "components": {
            "cov": components.get("cov"),
            "subject_cov_raw": components.get("subject_cov_raw"),
            "subject_area_raw": components.get("subject_area_raw"),
            "subject_terms_neutralized": components.get("subject_terms_neutralized"),
            "p_cut": components.get("p_cut"),
            "p_text": components.get("p_text"),
            "r_comp": components.get("r_comp"),
            "r_context": components.get("r_context"),
            "context_value": components.get("context_value"),
            "r_sym": components.get("r_sym"),
            "r_headroom": components.get("r_headroom"),
            "r_lookroom": components.get("r_lookroom"),
            "r_teach": components.get("r_teach"),
            "teacher_rho": components.get("teacher_rho"),
            "p_ar_free": components.get("p_ar_free"),
            "copyspace_blank_ratio_keep": components.get("copyspace_blank_ratio_keep"),
            "copyspace_side_consistency": components.get("copyspace_side_consistency"),
        },
        "checklist_labels": row.get("checklist_labels", {}),
        "why_tags": row.get("why_tags", []),
        "hard_reject": bool(row.get("hard_reject", False)),
        "scoring_region_iou": (None if subject_iou is None else round(float(subject_iou), 6)),
        "context_target_range": (
            comp_checks.get("context", {}).get("target_range")
            if isinstance(comp_checks.get("context"), dict)
            else None
        ),
    }


def sample_analysis_bullets(
    *,
    route: Dict[str, Any],
    cand_rec: Dict[str, Any],
    subject_visuals: Dict[str, Any],
    subject_support_metrics: Dict[str, Any],
    crop_support_metrics: Dict[str, Any],
    gt_mos_best: Dict[str, Any],
    ge_gt_best: Dict[str, Any],
    ge_prod_best: Dict[str, Any],
) -> List[str]:
    bullets: List[str] = []
    route_mode = str(route.get("subject_mode", "unknown"))
    route_conf = safe_float(route.get("subject_mode_conf", 0.0), 0.0)
    router_rule_id = str(route.get("router_rule_id", "unknown"))
    subject_set = route.get("subject_set", {}) if isinstance(route.get("subject_set"), dict) else {}
    router_signals = route.get("router_signals", {}) if isinstance(route.get("router_signals"), dict) else {}
    tags = cand_rec.get("tags", []) if isinstance(cand_rec.get("tags"), list) else []
    raw_anchor = subject_visuals.get("raw_anchor_norm_xyxy")
    scoring_region = subject_visuals.get("scoring_region_norm_xyxy")
    raw_anchor_area = box_area(raw_anchor) if raw_anchor is not None else None
    scoring_region_area = box_area(scoring_region) if scoring_region is not None else None

    bullets.append(
        f"router는 이 이미지를 `{route_mode}` / `{router_rule_id}`로 분기했습니다. subject_mode_conf={route_conf:.3f} 이고, 탐지된 주피사체는 {int(subject_set.get('num_effective_subjects', 0))}개입니다."
    )
    if raw_anchor_area is not None or scoring_region_area is not None:
        bullets.append(
            f"raw anchor 면적은 `{compact_float(raw_anchor_area, 4)}`, scorer가 실제로 쓰는 scoring region 면적은 `{compact_float(scoring_region_area, 4)}` 입니다."
        )
    bullets.append(
        f"subject 표현 타입은 `{subject_visuals.get('subject_repr_type', 'unknown')}`, source=`{subject_visuals.get('subject_source', 'unknown')}`, attention_layout=`{subject_visuals.get('attention_layout', 'unknown')}`, reliability=`{safe_float(subject_visuals.get('subject_reliability', 0.0), 0.0):.2f}`, score_mode=`{subject_visuals.get('score_mode', 'unknown')}` 입니다."
    )
    support_kind = str(subject_visuals.get("support_kind", "") or "")
    support_region = subject_visuals.get("support_region_norm_xyxy")
    if support_kind:
        bullets.append(
            f"scoring support는 `{support_kind}` 기준이며, support bbox=`{format_bbox_text(support_region) if support_region is not None else 'na'}` 입니다."
        )
    agreement_iou = safe_float(subject_visuals.get("subject_agreement_iou", 0.0), 0.0)
    center_dist = safe_float(subject_visuals.get("subject_center_distance", 0.0), 0.0)
    if agreement_iou > 0.0 or center_dist > 0.0:
        bullets.append(
            f"raw anchor와 saliency 주성분의 정합도는 IoU=`{agreement_iou:.3f}`, centroid distance=`{center_dist:.3f}` 입니다."
        )
    raw_diag = subject_support_metrics.get("raw_anchor", {}) if isinstance(subject_support_metrics.get("raw_anchor"), dict) else {}
    candidate_diag = subject_support_metrics.get("candidate_anchor", {}) if isinstance(subject_support_metrics.get("candidate_anchor"), dict) else {}
    support_diag = subject_support_metrics.get("support_region", {}) if isinstance(subject_support_metrics.get("support_region"), dict) else {}
    bullets.append(
        "latent GT support 기준 mass recall은 "
        + f"raw=`{safe_float(raw_diag.get('mass_recall', 0.0), 0.0):.3f}`, "
        + f"candidate=`{safe_float(candidate_diag.get('mass_recall', 0.0), 0.0):.3f}`, "
        + f"support=`{safe_float(support_diag.get('mass_recall', 0.0), 0.0):.3f}` 입니다."
    )
    if safe_float(support_diag.get("mass_recall", 0.0), 0.0) > safe_float(candidate_diag.get("mass_recall", 0.0), 0.0) + 0.1:
        bullets.append("support region이 candidate anchor보다 latent GT support를 더 많이 포함합니다. scene-like 이미지에서는 bbox IoU보다 이 값이 더 직접적인 품질 신호입니다.")
    if safe_float(support_diag.get("iou_to_gt_best", 0.0), 0.0) < 0.2 and safe_float(support_diag.get("mass_recall", 0.0), 0.0) >= 0.6:
        bullets.append("support bbox IoU는 낮지만 latent support mass recall은 높습니다. 분산 foreground를 넓게 담는 장면에서는 '작은 정답 box와의 IoU'만으로는 quality를 설명하기 어렵습니다.")
    if str(subject_visuals.get("subject_source", "")) == "fusion":
        bullets.append("tiny detector anchor와 saliency dominant region이 서로 다른 위치를 가리켜, 두 영역을 union grouping해 multi-subject support로 재해석했습니다.")
    elif str(subject_visuals.get("attention_layout", "")) == "distributed":
        bullets.append("saliency가 여러 foreground component로 분산되어 있어, top1/top2가 아니라 foreground support 전체를 scoring region 근거로 사용했습니다.")
    if not tags:
        bullets.append("caption/tag 메타데이터는 비어 있습니다. 이번 설명에서는 텍스트 캡션 대신 routing signal과 점수 내부 항을 근거로 해석해야 합니다.")
    else:
        bullets.append(f"caption/tag 메타데이터: `{', '.join(str(tag) for tag in tags[:8])}`")
    if ge_prod_best["macro_scores"].get("A_macro") is None and ge_prod_best["macro_scores"].get("T_macro") is None:
        bullets.append("이번 run은 proxy-expensive 설정이라 `A_macro`와 `T_macro`가 비활성입니다. 따라서 `score_rank`는 사실상 활성 macro인 `S_macro`와 `C_macro`의 평균으로 결정됩니다.")

    rank_gap = safe_float(ge_prod_best.get("score_rank", 0.0), 0.0) - safe_float(ge_gt_best.get("score_rank", 0.0), 0.0)
    policy_gap = safe_float(ge_prod_best.get("score_policy", 0.0), 0.0) - safe_float(ge_gt_best.get("score_policy", 0.0), 0.0)
    if rank_gap > 0.0:
        bullets.append(
            f"`Ge prod best`가 `Ge GT best`보다 `score_rank`에서 `{rank_gap:+.6f}`, `score_policy`에서 `{policy_gap:+.6f}` 앞섰습니다."
        )
    elif rank_gap < 0.0:
        bullets.append(
            f"`Ge prod best`가 전체 후보 선택 결과이지만 `Ge GT best`보다 `score_rank`에서는 `{rank_gap:+.6f}` 낮습니다. 이 경우 keep-vs-crop 또는 baseline 정책 쪽 확인이 필요합니다."
        )
    else:
        bullets.append("`Ge prod best`와 `Ge GT best`의 `score_rank`는 동일합니다.")

    deltas = {
        "S_macro": safe_float(ge_prod_best["macro_scores"].get("S_macro", 0.0), 0.0) - safe_float(ge_gt_best["macro_scores"].get("S_macro", 0.0), 0.0),
        "C_macro": safe_float(ge_prod_best["macro_scores"].get("C_macro", 0.0), 0.0) - safe_float(ge_gt_best["macro_scores"].get("C_macro", 0.0), 0.0),
        "C_comp": safe_float(ge_prod_best["components"].get("r_comp", 0.0), 0.0) - safe_float(ge_gt_best["components"].get("r_comp", 0.0), 0.0),
        "S_scale": safe_float(ge_prod_best["macro_scores"].get("S_macro", 0.0), 0.0) - safe_float(ge_gt_best["macro_scores"].get("S_macro", 0.0), 0.0),
        "context_value": safe_float(ge_prod_best["components"].get("context_value", 0.0), 0.0) - safe_float(ge_gt_best["components"].get("context_value", 0.0), 0.0),
        "r_sym": safe_float(ge_prod_best["components"].get("r_sym", 0.0), 0.0) - safe_float(ge_gt_best["components"].get("r_sym", 0.0), 0.0),
        "p_ar_free": safe_float(ge_prod_best["components"].get("p_ar_free", 0.0), 0.0) - safe_float(ge_gt_best["components"].get("p_ar_free", 0.0), 0.0),
    }
    sorted_deltas = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)
    bullets.append(
        "주요 차이 항: "
        + ", ".join(f"`{name}` {value:+.4f}" for name, value in sorted_deltas[:4])
        + "."
    )
    if gt_mos_best["candidate_id"] != ge_gt_best["candidate_id"]:
        bullets.append(
            f"사람 MOS 최고 GT(`{gt_mos_best['candidate_id']}`)와 scorer가 고른 GT 최고(`{ge_gt_best['candidate_id']}`)도 다릅니다. 즉 이 샘플은 사람 선호와 scorer 선호가 이미 GT 내부에서도 어긋나는 케이스입니다."
        )
    gt_crop_diag = crop_support_metrics.get("gt_mos_best", {}) if isinstance(crop_support_metrics.get("gt_mos_best"), dict) else {}
    ge_gt_crop_diag = crop_support_metrics.get("ge_gt_best", {}) if isinstance(crop_support_metrics.get("ge_gt_best"), dict) else {}
    ge_prod_crop_diag = crop_support_metrics.get("ge_prod_best", {}) if isinstance(crop_support_metrics.get("ge_prod_best"), dict) else {}
    bullets.append(
        "GT latent support mass 기준 crop recall은 "
        + f"GT MOS best=`{safe_float(gt_crop_diag.get('mass_recall', 0.0), 0.0):.3f}`, "
        + f"Ge GT best=`{safe_float(ge_gt_crop_diag.get('mass_recall', 0.0), 0.0):.3f}`, "
        + f"Ge prod best=`{safe_float(ge_prod_crop_diag.get('mass_recall', 0.0), 0.0):.3f}` 입니다."
    )
    if safe_float(ge_gt_crop_diag.get("mass_recall", 0.0), 0.0) > safe_float(ge_prod_crop_diag.get("mass_recall", 0.0), 0.0) + 0.05:
        bullets.append("Ge GT best가 Ge prod best보다 latent GT support를 더 많이 담습니다. 즉 production winner가 사람 관심 영역을 덜 보존하면서 composition/context 쪽에서 이겼을 가능성이 큽니다.")
    elif safe_float(ge_prod_crop_diag.get("mass_recall", 0.0), 0.0) > safe_float(ge_gt_crop_diag.get("mass_recall", 0.0), 0.0) + 0.05:
        bullets.append("Ge prod best가 Ge GT best보다 latent GT support를 더 많이 담습니다. 이 경우 GT 내부 최고보다 production 후보가 더 넓은 관심 영역을 보존했을 가능성이 있습니다.")

    blank_ratio = safe_float(router_signals.get("blank_ratio_est", 0.0), 0.0)
    symmetry_score = safe_float(router_signals.get("symmetry_score", 0.0), 0.0)
    saliency_backend = str(subject_visuals.get("saliency_backend", "none"))
    saliency_fg = safe_float(subject_visuals.get("saliency_foreground_area_ratio", 0.0), 0.0)
    saliency_dom = safe_float(subject_visuals.get("saliency_dominance_score", 0.0), 0.0)
    bullets.append(
        f"routing signal 기준 blank_ratio_est={blank_ratio:.3f}, symmetry_score={symmetry_score:.3f} 이고 c7 saliency backend=`{saliency_backend}`, foreground_area_ratio={saliency_fg:.3f}, dominance_score={saliency_dom:.3f} 입니다."
    )
    return bullets


def build_sample_analysis(
    *,
    image_id: str,
    cand_rec: Dict[str, Any],
    feat_rec: Dict[str, Any],
    image_meta: Dict[str, Any],
    anns: Sequence[Dict[str, Any]],
    cfg: TeacherScorerConfig,
    image_path: Path,
    output_dir: Path,
    softmax_tau: float,
    saliency_mask_resolver: Optional[SampleSaliencyMaskResolver] = None,
) -> Dict[str, Any]:
    context = build_scoring_context(cand_rec=cand_rec, feat_rec=feat_rec, cfg=cfg)
    width = int(image_meta["width"])
    height = int(image_meta["height"])
    gt_candidates = build_gt_candidates(anns, width=width, height=height)
    gt_lookup = {f"gaic_gt_{safe_int(ann.get('id', 0))}": ann for ann in anns}
    raw_free_pool = copy.deepcopy((cand_rec.get("candidates_by_ar") or {}).get("FREE") or [])
    ge_scored, ge_route, ge_teacher_ctx = score_candidates_freeform(
        cand_rec=cand_rec,
        feat_rec=feat_rec,
        candidates=raw_free_pool + gt_candidates,
        cfg=cfg,
        softmax_tau=softmax_tau,
    )
    ge_decision = derive_decision_from_full_pool(scored=ge_scored, route=ge_route, teacher_ctx=ge_teacher_ctx, cfg=cfg)
    ge_rows_by_id = {str(row.get("candidate_id", "")): row for row in ge_scored}
    ge_gt_rows = [row for row in ge_scored if str(row.get("candidate_id", "")).startswith("gaic_gt_")]
    ge_gt_rows.sort(key=score_rank_value, reverse=True)

    gt_mos_ann = anns[0]
    gt_mos_candidate_id = f"gaic_gt_{safe_int(gt_mos_ann.get('id', 0))}"
    gt_mos_row = ge_rows_by_id.get(gt_mos_candidate_id, ge_gt_rows[0])
    ge_gt_best_row = ge_gt_rows[0]
    ge_prod_best_row = ge_decision["best_candidate"]

    subject_visuals = collect_subject_visuals(cand_rec=cand_rec, feat_rec=feat_rec, route=ge_route)
    saliency_mask = (
        None
        if saliency_mask_resolver is None
        else saliency_mask_resolver.resolve(image_path, subject_visuals)
    )
    scoring_region_norm = (
        subject_visuals.get("scoring_region_norm_xyxy")
        or [round(float(v), 6) for v in context["subject_box"]]
    )
    face_center_norm = first_face_center_norm(context.get("c3_info", {}))
    latent_support = build_latent_support(anns, width=width, height=height)
    subject_support_metrics = compute_subject_box_diagnostics(
        raw_anchor=subject_visuals.get("raw_anchor_norm_xyxy"),
        candidate_anchor=subject_visuals.get("candidate_anchor_norm_xyxy"),
        support_region=subject_visuals.get("support_region_norm_xyxy") or scoring_region_norm,
        gt_best_box=gt_mos_row["bbox_norm_xyxy"],
        anns=anns,
        width=width,
        height=height,
    )

    gt_mos_summary = summarize_sample_candidate(
        row=gt_mos_row,
        role="GT MOS best",
        mos=safe_float(gt_mos_ann.get("score", 0.0), 0.0),
        scoring_region_norm=scoring_region_norm,
    )
    ge_gt_summary = summarize_sample_candidate(
        row=ge_gt_best_row,
        role="Ge GT best",
        mos=safe_float(gt_lookup[str(ge_gt_best_row["candidate_id"])].get("score", 0.0), 0.0),
        scoring_region_norm=scoring_region_norm,
    )
    ge_prod_ann = gt_lookup.get(str(ge_prod_best_row.get("candidate_id", "")))
    ge_prod_summary = summarize_sample_candidate(
        row=ge_prod_best_row,
        role="Ge prod best",
        mos=(None if ge_prod_ann is None else safe_float(ge_prod_ann.get("score", 0.0), 0.0)),
        scoring_region_norm=scoring_region_norm,
    )
    crop_support_metrics = {
        "gt_mos_best": {
            "bbox_norm_xyxy": gt_mos_summary["bbox_norm_xyxy"],
            "iou_to_gt_best": round(iou_xyxy(gt_mos_summary["bbox_norm_xyxy"], gt_mos_summary["bbox_norm_xyxy"]), 6),
            **compute_latent_box_mass_metrics(gt_mos_summary["bbox_norm_xyxy"], latent_support),
        },
        "ge_gt_best": {
            "bbox_norm_xyxy": ge_gt_summary["bbox_norm_xyxy"],
            "iou_to_gt_best": round(iou_xyxy(ge_gt_summary["bbox_norm_xyxy"], gt_mos_summary["bbox_norm_xyxy"]), 6),
            **compute_latent_box_mass_metrics(ge_gt_summary["bbox_norm_xyxy"], latent_support),
        },
        "ge_prod_best": {
            "bbox_norm_xyxy": ge_prod_summary["bbox_norm_xyxy"],
            "iou_to_gt_best": round(iou_xyxy(ge_prod_summary["bbox_norm_xyxy"], gt_mos_summary["bbox_norm_xyxy"]), 6),
            **compute_latent_box_mass_metrics(ge_prod_summary["bbox_norm_xyxy"], latent_support),
        },
    }

    rank_gap = ge_prod_summary["score_rank"] - ge_gt_summary["score_rank"]
    policy_gap = ge_prod_summary["score_policy"] - ge_gt_summary["score_policy"]
    overlay_path = output_dir / "samples" / "details" / f"{image_id}_diagnostic_overlay.png"
    panel_path = output_dir / "samples" / "details" / f"{image_id}_diagnostic_panel.jpg"
    analysis_json_path = output_dir / "samples" / "details" / f"{image_id}_analysis.json"

    render_sample_diagnostic_overlay(
        image_path=image_path,
        out_path=overlay_path,
        image_id=image_id,
        official_split=str(image_meta["official_split"]),
        subject_mode=str(ge_route.get("subject_mode", "unknown")),
        router_rule_id=str(ge_route.get("router_rule_id", "unknown")),
        subject_visuals=subject_visuals,
        face_center_norm=face_center_norm,
        gt_mos_best=gt_mos_summary,
        ge_gt_best=ge_gt_summary,
        ge_prod_best=ge_prod_summary,
        rank_gap=rank_gap,
        policy_gap=policy_gap,
        saliency_mask=saliency_mask,
    )

    with Image.open(image_path) as image:
        panel_items: List[Dict[str, Any]] = []
        for role_name, row_obj, mos_value in (
            ("GT MOS best", gt_mos_row, gt_mos_summary["mos"]),
            ("Ge GT best", ge_gt_best_row, ge_gt_summary["mos"]),
            ("Ge prod best", ge_prod_best_row, ge_prod_summary["mos"]),
        ):
            bbox_px = report_denorm_box(row_obj["bbox_norm_xyxy"], width, height)
            subtitle_parts = [
                f"rank={safe_float(row_obj.get('scores', {}).get('rank', 0.0), 0.0):.4f}",
                f"policy={safe_float(row_obj.get('scores', {}).get('policy', 0.0), 0.0):.4f}",
            ]
            if mos_value is not None:
                subtitle_parts.append(f"MOS={safe_float(mos_value, 0.0):.2f}")
            panel_items.append(
                {
                    "bbox_px": bbox_px,
                    "bbox_norm": row_obj["bbox_norm_xyxy"],
                    "title": role_name,
                    "subtitle": " | ".join(subtitle_parts),
                    "labels": row_obj.get("checklist_labels", {}),
                    "scores": row_obj.get("scores", {}),
                    "components": row_obj.get("scores", {}).get("components", {}),
                    "macro_scores": row_obj.get("macro_scores", {}),
                    "score_details": extract_detailed_score_rows(row_obj, ge_route, asdict(cfg)),
                    "teacher_agreement": [],
                    "subject_box_norm": scoring_region_norm,
                    "face_center_norm": face_center_norm,
                    "provenance_text": summarize_provenance(row_obj),
                }
            )
        build_detailed_panel(
            image=image.convert("RGB"),
            items=panel_items,
            out_path=panel_path,
            title=f"{image_id} | Ge GT best vs Ge prod best 진단 패널",
        )

    analysis = {
        "image_id": image_id,
        "official_split": str(image_meta["official_split"]),
        "file_name": str(image_meta["file_name"]),
        "subject_mode": str(ge_route.get("subject_mode", "unknown")),
        "policy_id": str(ge_route.get("policy_id", "unknown")),
        "subject_mode_conf": round(safe_float(ge_route.get("subject_mode_conf", 0.0), 0.0), 6),
        "subject_mode_reasons": ge_route.get("subject_mode_reasons", []),
        "router_rule_id": str(ge_route.get("router_rule_id", "unknown")),
        "route_flags": ge_route.get("flags", {}),
        "route_lambdas": ge_route.get("lambdas", {}),
        "router_signals": ge_route.get("router_signals", {}),
        "subject_set": ge_route.get("subject_set", {}),
        "subject_visuals": subject_visuals,
        "raw_anchor_norm_xyxy": subject_visuals.get("raw_anchor_norm_xyxy"),
        "raw_anchor_area": round(box_area(subject_visuals["raw_anchor_norm_xyxy"]), 6) if subject_visuals.get("raw_anchor_norm_xyxy") else None,
        "candidate_anchor_norm_xyxy": subject_visuals.get("candidate_anchor_norm_xyxy"),
        "candidate_anchor_area": round(box_area(subject_visuals["candidate_anchor_norm_xyxy"]), 6) if subject_visuals.get("candidate_anchor_norm_xyxy") else None,
        "scoring_region_norm_xyxy": scoring_region_norm,
        "scoring_region_area": round(box_area(scoring_region_norm), 6),
        "saliency_top1_norm_xyxy": subject_visuals.get("saliency_top1_norm_xyxy"),
        "saliency_top2_norm_xyxy": subject_visuals.get("saliency_top2_norm_xyxy"),
        "saliency_foreground_norm_xyxy": subject_visuals.get("saliency_foreground_norm_xyxy"),
        "face_center_norm_xy": face_center_norm,
        "tags": cand_rec.get("tags", []) if isinstance(cand_rec.get("tags"), list) else [],
        "gt_mos_best": gt_mos_summary,
        "ge_gt_best": ge_gt_summary,
        "ge_prod_best": ge_prod_summary,
        "comparisons": {
            "prod_vs_gt_rank_gap": round(rank_gap, 6),
            "prod_vs_gt_policy_gap": round(policy_gap, 6),
            "prod_vs_gt_iou": round(iou_xyxy(ge_prod_best_row["bbox_norm_xyxy"], ge_gt_best_row["bbox_norm_xyxy"]), 6),
            "gt_mos_vs_ge_gt_iou": round(iou_xyxy(gt_mos_row["bbox_norm_xyxy"], ge_gt_best_row["bbox_norm_xyxy"]), 6),
            "gt_mos_vs_ge_prod_iou": round(iou_xyxy(gt_mos_row["bbox_norm_xyxy"], ge_prod_best_row["bbox_norm_xyxy"]), 6),
        },
        "latent_support": {
            "topk_count": safe_int(subject_support_metrics.get("latent_support_topk_count", 0)),
            "bbox_norm_xyxy": subject_support_metrics.get("latent_support_bbox_norm_xyxy"),
            "centroid_xy_norm": subject_support_metrics.get("latent_support_centroid_xy_norm"),
        },
        "subject_support_metrics": subject_support_metrics,
        "crop_support_metrics": crop_support_metrics,
        "analysis_bullets": sample_analysis_bullets(
            route=ge_route,
            cand_rec=cand_rec,
            subject_visuals=subject_visuals,
            subject_support_metrics=subject_support_metrics,
            crop_support_metrics=crop_support_metrics,
            gt_mos_best=gt_mos_summary,
            ge_gt_best=ge_gt_summary,
            ge_prod_best=ge_prod_summary,
        ),
        "assets": {
            "overlay": str(overlay_path),
            "panel": str(panel_path),
            "analysis_json": str(analysis_json_path),
        },
    }
    write_json(analysis_json_path, analysis)
    return analysis


def plot_histogram(values: Sequence[float], title: str, xlabel: str, out_path: Path, *, bins: int = 30, color: str = "#1f77b4") -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.hist(list(values), bins=bins, color=color, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter(x: Sequence[float], y: Sequence[float], title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 6))
    plt.scatter(list(x), list(y), s=8, alpha=0.25, color="#1f77b4")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_reliability(curve_rows: Sequence[Dict[str, float]], title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    xs = [float(row["confidence"]) for row in curve_rows if float(row["count"]) > 0.0]
    ys = [float(row["accuracy"]) for row in curve_rows if float(row["count"]) > 0.0]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.2)
    plt.plot(xs, ys, marker="o", color="#d62728", linewidth=2.0)
    plt.title(title)
    plt.xlabel("predicted probability")
    plt.ylabel("empirical frequency")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_protocol_comparison(summary: Dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    metrics = ["mean_spearman", "mean_kendall_tau_b", "mean_weighted_pair_acc", "mean_hit_at_1", "mean_topq_jaccard"]
    labels = ["Spearman", "Kendall", "PairAcc", "Hit@1", "Top-q Jaccard"]
    gc_vals = [safe_float(summary["Gc"]["trend_by_field"]["score_rank"][metric], 0.0) for metric in metrics]
    ge_vals = [safe_float(summary["Ge"]["trend_by_field"]["score_rank"][metric], 0.0) for metric in metrics]
    x = np.arange(len(metrics))
    width = 0.38
    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2.0, gc_vals, width=width, color="#4c78a8", label="Gc")
    plt.bar(x + width / 2.0, ge_vals, width=width, color="#f58518", label="Ge")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title("Primary ranking metrics by protocol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def format_ci(ci: Tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def format_bbox_text(bbox: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in bbox) + "]"


def format_point_text(point_xy: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in point_xy) + ")"


def append_sample_analysis_markdown(
    lines: List[str],
    *,
    output_dir: Path,
    sample_analysis: Dict[str, Any],
) -> None:
    gt_mos = sample_analysis["gt_mos_best"]
    ge_gt = sample_analysis["ge_gt_best"]
    ge_prod = sample_analysis["ge_prod_best"]
    assets = sample_analysis["assets"]
    comparisons = sample_analysis["comparisons"]
    router_signals = sample_analysis.get("router_signals", {}) if isinstance(sample_analysis.get("router_signals"), dict) else {}
    route_lambdas = sample_analysis.get("route_lambdas", {}) if isinstance(sample_analysis.get("route_lambdas"), dict) else {}
    subject_set = sample_analysis.get("subject_set", {}) if isinstance(sample_analysis.get("subject_set"), dict) else {}
    subject_visuals = sample_analysis.get("subject_visuals", {}) if isinstance(sample_analysis.get("subject_visuals"), dict) else {}
    latent_support = sample_analysis.get("latent_support", {}) if isinstance(sample_analysis.get("latent_support"), dict) else {}
    subject_support_metrics = sample_analysis.get("subject_support_metrics", {}) if isinstance(sample_analysis.get("subject_support_metrics"), dict) else {}
    crop_support_metrics = sample_analysis.get("crop_support_metrics", {}) if isinstance(sample_analysis.get("crop_support_metrics"), dict) else {}

    lines.append(f"### image_id={sample_analysis['image_id']}")
    lines.append("")
    lines.append(
        f"- split=`{sample_analysis['official_split']}`, subject_mode=`{sample_analysis['subject_mode']}`, policy_id=`{sample_analysis['policy_id']}`, router_rule_id=`{sample_analysis['router_rule_id']}`"
    )
    lines.append(
        f"- subject_mode_conf=`{sample_analysis['subject_mode_conf']:.3f}`, raw anchor=`{format_bbox_text(sample_analysis['raw_anchor_norm_xyxy']) if sample_analysis.get('raw_anchor_norm_xyxy') else 'na'}`, scoring region=`{format_bbox_text(sample_analysis['scoring_region_norm_xyxy'])}`, scoring_region_area=`{sample_analysis['scoring_region_area']:.4f}`"
    )
    lines.append(
        f"- subject_mode_reasons=`{', '.join(sample_analysis.get('subject_mode_reasons', [])) or '없음'}`, tags=`{', '.join(sample_analysis.get('tags', [])) or '없음'}`"
    )
    lines.append(
        f"- subject repr=`{subject_visuals.get('subject_repr_type', 'na')}`, source=`{subject_visuals.get('subject_source', 'na')}`, attention_layout=`{subject_visuals.get('attention_layout', 'na')}`, support_kind=`{subject_visuals.get('support_kind', 'na')}`, reliability=`{safe_float(subject_visuals.get('subject_reliability', 0.0), 0.0):.2f}`, placeholder=`{bool(subject_visuals.get('subject_placeholder_flag', False))}`, score_mode=`{subject_visuals.get('score_mode', 'na')}`, support_map=`{bool(subject_visuals.get('support_map_enabled', False))}`"
    )
    lines.append(
        f"- router signals: num_effective_subjects=`{subject_set.get('num_effective_subjects', 0)}`, blank_ratio_est=`{safe_float(router_signals.get('blank_ratio_est', 0.0), 0.0):.3f}`, symmetry_score=`{safe_float(router_signals.get('symmetry_score', 0.0), 0.0):.3f}`, scene_score=`{safe_float(router_signals.get('scene_score', 0.0), 0.0):.3f}`, horizon_conf=`{safe_float(router_signals.get('horizon_conf', 0.0), 0.0):.3f}`"
    )
    lines.append(
        f"- c7 saliency: backend=`{subject_visuals.get('saliency_backend', 'none')}`, fg_area=`{safe_float(subject_visuals.get('saliency_foreground_area_ratio', 0.0), 0.0):.3f}`, blank_ratio=`{safe_float(subject_visuals.get('saliency_blank_ratio_est', 0.0), 0.0):.3f}`, dominance=`{safe_float(subject_visuals.get('saliency_dominance_score', 0.0), 0.0):.3f}`, components=`{safe_int(subject_visuals.get('saliency_component_count', 0))}`"
    )
    lines.append(
        f"- c7 mask reference bbox: top1=`{format_bbox_text(subject_visuals['saliency_top1_norm_xyxy']) if subject_visuals.get('saliency_top1_norm_xyxy') else 'na'}`, top2=`{format_bbox_text(subject_visuals['saliency_top2_norm_xyxy']) if subject_visuals.get('saliency_top2_norm_xyxy') else 'na'}`, foreground=`{format_bbox_text(subject_visuals['saliency_foreground_norm_xyxy']) if subject_visuals.get('saliency_foreground_norm_xyxy') else 'na'}`, support=`{format_bbox_text(subject_visuals['saliency_support_norm_xyxy']) if subject_visuals.get('saliency_support_norm_xyxy') else 'na'}`"
    )
    lines.append(
        f"- raw-vs-saliency agreement: IoU=`{safe_float(subject_visuals.get('subject_agreement_iou', 0.0), 0.0):.3f}`, centroid distance=`{safe_float(subject_visuals.get('subject_center_distance', 0.0), 0.0):.3f}`"
    )
    lines.append(
        f"- latent GT support: topk=`{safe_int(latent_support.get('topk_count', 0))}`, centroid=`{format_point_text(latent_support['centroid_xy_norm']) if isinstance(latent_support.get('centroid_xy_norm'), list) and len(latent_support.get('centroid_xy_norm')) == 2 else 'na'}`, bbox=`{format_bbox_text(latent_support['bbox_norm_xyxy']) if latent_support.get('bbox_norm_xyxy') else 'na'}`"
    )
    lines.append(
        f"- route lambdas: cov=`{safe_float(route_lambdas.get('cov', 0.0), 0.0):.2f}`, cut=`{safe_float(route_lambdas.get('cut', 0.0), 0.0):.2f}`, comp=`{safe_float(route_lambdas.get('comp', 0.0), 0.0):.2f}`, ctx=`{safe_float(route_lambdas.get('ctx', 0.0), 0.0):.2f}`, sym=`{safe_float(route_lambdas.get('sym', 0.0), 0.0):.2f}`, ar_free=`{safe_float(route_lambdas.get('ar_free', 0.0), 0.0):.2f}`"
    )
    lines.append("")
    lines.append(f"![{sample_analysis['image_id']} diagnostic overlay]({Path(assets['overlay']).relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![{sample_analysis['image_id']} diagnostic panel]({Path(assets['panel']).relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append("후보 비교:")
    lines.append("")
    lines.extend(
        markdown_table(
            ["역할", "candidate_id", "source", "bbox", "MOS", "sigmoid(z)", "rank", "policy", "rank_pct", "robust z", "area_log_prior", "scoring region IoU"],
            [
                [
                    row["role"],
                    row["candidate_id"],
                    row["source"],
                    format_bbox_text(row["bbox_norm_xyxy"]),
                    compact_float(row.get("mos"), 3),
                    compact_float(row.get("score_sigmoid_z_local"), 6),
                    compact_float(row.get("score_rank"), 6),
                    compact_float(row.get("score_policy"), 6),
                    compact_float(row.get("score_rank_pct"), 6),
                    compact_float(row.get("score_z_local"), 6),
                    compact_float(row.get("area_log_prior"), 6),
                    compact_float(row.get("scoring_region_iou"), 3),
                ]
                for row in (gt_mos, ge_gt, ge_prod)
            ],
        )
    )
    lines.append("")
    lines.append("subject support 진단:")
    lines.append("")
    lines.append("- `latent mass recall`은 GT 상위 crop inclusion heatmap이 해당 box 안에 얼마나 들어가는지입니다. `scene_general`처럼 foreground가 넓게 퍼진 이미지는 bbox IoU보다 이 값을 우선 해석하는 편이 맞습니다.")
    lines.append("")
    lines.extend(
        markdown_table(
            ["영역", "bbox", "IoU->GTbest", "latent mass recall", "centroid inside", "centroid dist", "latent bbox IoU"],
            [
                [
                    "raw anchor",
                    format_bbox_text(sample_analysis["raw_anchor_norm_xyxy"]) if sample_analysis.get("raw_anchor_norm_xyxy") else "na",
                    compact_float(subject_support_metrics.get("raw_anchor", {}).get("iou_to_gt_best"), 3),
                    compact_float(subject_support_metrics.get("raw_anchor", {}).get("mass_recall"), 3),
                    compact_float(subject_support_metrics.get("raw_anchor", {}).get("centroid_inside"), 3),
                    compact_float(subject_support_metrics.get("raw_anchor", {}).get("centroid_distance"), 3),
                    compact_float(subject_support_metrics.get("raw_anchor", {}).get("latent_bbox_iou"), 3),
                ],
                [
                    "candidate anchor",
                    format_bbox_text(sample_analysis["candidate_anchor_norm_xyxy"]) if sample_analysis.get("candidate_anchor_norm_xyxy") else "na",
                    compact_float(subject_support_metrics.get("candidate_anchor", {}).get("iou_to_gt_best"), 3),
                    compact_float(subject_support_metrics.get("candidate_anchor", {}).get("mass_recall"), 3),
                    compact_float(subject_support_metrics.get("candidate_anchor", {}).get("centroid_inside"), 3),
                    compact_float(subject_support_metrics.get("candidate_anchor", {}).get("centroid_distance"), 3),
                    compact_float(subject_support_metrics.get("candidate_anchor", {}).get("latent_bbox_iou"), 3),
                ],
                [
                    "scoring region",
                    format_bbox_text(sample_analysis["scoring_region_norm_xyxy"]) if sample_analysis.get("scoring_region_norm_xyxy") else "na",
                    compact_float(subject_support_metrics.get("support_region", {}).get("iou_to_gt_best"), 3),
                    compact_float(subject_support_metrics.get("support_region", {}).get("mass_recall"), 3),
                    compact_float(subject_support_metrics.get("support_region", {}).get("centroid_inside"), 3),
                    compact_float(subject_support_metrics.get("support_region", {}).get("centroid_distance"), 3),
                    compact_float(subject_support_metrics.get("support_region", {}).get("latent_bbox_iou"), 3),
                ],
            ],
        )
    )
    lines.append("")
    lines.append("crop vs latent support:")
    lines.append("")
    lines.extend(
        markdown_table(
            ["역할", "bbox", "IoU->GTbest", "latent mass recall", "centroid inside", "centroid dist", "latent bbox IoU"],
            [
                [
                    "GT MOS best",
                    format_bbox_text(gt_mos["bbox_norm_xyxy"]),
                    compact_float(crop_support_metrics.get("gt_mos_best", {}).get("iou_to_gt_best"), 3),
                    compact_float(crop_support_metrics.get("gt_mos_best", {}).get("mass_recall"), 3),
                    compact_float(crop_support_metrics.get("gt_mos_best", {}).get("centroid_inside"), 3),
                    compact_float(crop_support_metrics.get("gt_mos_best", {}).get("centroid_distance"), 3),
                    compact_float(crop_support_metrics.get("gt_mos_best", {}).get("latent_bbox_iou"), 3),
                ],
                [
                    "Ge GT best",
                    format_bbox_text(ge_gt["bbox_norm_xyxy"]),
                    compact_float(crop_support_metrics.get("ge_gt_best", {}).get("iou_to_gt_best"), 3),
                    compact_float(crop_support_metrics.get("ge_gt_best", {}).get("mass_recall"), 3),
                    compact_float(crop_support_metrics.get("ge_gt_best", {}).get("centroid_inside"), 3),
                    compact_float(crop_support_metrics.get("ge_gt_best", {}).get("centroid_distance"), 3),
                    compact_float(crop_support_metrics.get("ge_gt_best", {}).get("latent_bbox_iou"), 3),
                ],
                [
                    "Ge prod best",
                    format_bbox_text(ge_prod["bbox_norm_xyxy"]),
                    compact_float(crop_support_metrics.get("ge_prod_best", {}).get("iou_to_gt_best"), 3),
                    compact_float(crop_support_metrics.get("ge_prod_best", {}).get("mass_recall"), 3),
                    compact_float(crop_support_metrics.get("ge_prod_best", {}).get("centroid_inside"), 3),
                    compact_float(crop_support_metrics.get("ge_prod_best", {}).get("centroid_distance"), 3),
                    compact_float(crop_support_metrics.get("ge_prod_best", {}).get("latent_bbox_iou"), 3),
                ],
            ],
        )
    )
    lines.append("")
    lines.append("내부 항 비교:")
    lines.append("")
    lines.extend(
        markdown_table(
            ["역할", "S_macro", "C_macro", "cov", "cov_raw", "neutralized", "r_comp", "context_value", "r_context", "r_sym", "p_ar_free"],
            [
                [
                    row["role"],
                    compact_float(row["macro_scores"].get("S_macro"), 4),
                    compact_float(row["macro_scores"].get("C_macro"), 4),
                    compact_float(row["components"].get("cov"), 4),
                    compact_float(row["components"].get("subject_cov_raw"), 4),
                    str(bool(row["components"].get("subject_terms_neutralized", False))),
                    compact_float(row["components"].get("r_comp"), 4),
                    compact_float(row["components"].get("context_value"), 4),
                    compact_float(row["components"].get("r_context"), 4),
                    compact_float(row["components"].get("r_sym"), 4),
                    compact_float(row["components"].get("p_ar_free"), 4),
                ]
                for row in (gt_mos, ge_gt, ge_prod)
            ],
        )
    )
    lines.append("")
    lines.append("비교 해석:")
    lines.append("")
    lines.append(
        f"- `Ge prod best` vs `Ge GT best`: rank gap=`{comparisons['prod_vs_gt_rank_gap']:+.6f}`, policy gap=`{comparisons['prod_vs_gt_policy_gap']:+.6f}`, IoU=`{comparisons['prod_vs_gt_iou']:.3f}`"
    )
    lines.append(
        f"- `GT MOS best` vs `Ge GT best` IoU=`{comparisons['gt_mos_vs_ge_gt_iou']:.3f}`, `GT MOS best` vs `Ge prod best` IoU=`{comparisons['gt_mos_vs_ge_prod_iou']:.3f}`"
    )
    for bullet in sample_analysis.get("analysis_bullets", []):
        lines.append(f"- {bullet}")
    lines.append("")
    for row in (gt_mos, ge_gt, ge_prod):
        lines.append(
            f"- `{row['role']}` labels: `{', '.join(str(v) for v in row.get('checklist_labels', {}).values() if v)}`"
        )
        lines.append(
            f"- `{row['role']}` why_tags: `{', '.join(str(v) for v in row.get('why_tags', [])) or '없음'}`"
        )
    lines.append("")


def build_markdown_report(
    *,
    output_dir: Path,
    summary: Dict[str, Any],
    image_records: Sequence[Dict[str, Any]],
    qa_summary: Dict[str, Any],
    validation_summary: Dict[str, Any],
    sample_files: Sequence[Path],
    sample_analyses: Sequence[Dict[str, Any]],
    plot_paths: Dict[str, Path],
) -> str:
    overlap = summary["dataset"]["overlap_image_count"]
    gt_total = summary["dataset"]["gt_total_image_count"]
    lines: List[str] = []
    lines.append("# GAIC 벤치마크 평가 리포트")
    lines.append("")
    lines.append("## 1. 평가 대상")
    lines.append("")
    lines.append(f"- 생성 시각: `{summary['run']['generated_at']}`")
    lines.append(f"- 실제 평가된 겹침 이미지 수: `{overlap}` / 로컬 실행 이미지 수 `{summary['dataset']['run_image_count']}` / 전체 GT 이미지 수 `{gt_total}`")
    lines.append(f"- official split 기준 GT overlap: train `{summary['dataset']['overlap_train_count']}`, test `{summary['dataset']['overlap_test_count']}`")
    lines.append(f"- 사용한 production artifact: `{summary['sources']['training_label_dir']}`")
    lines.append(f"- `Ge` 점수화에 사용한 raw candidate source: `{summary['sources']['candidates_jsonl']}`")
    lines.append(f"- GT 재점수화에 사용한 feature source: `{summary['sources']['features_jsonl']}`")
    lines.append(f"- benchmark 범위 메모: {summary['dataset'].get('benchmark_scope_note', '')}")
    lines.append("")
    lines.append("## 2. 프로토콜 정의")
    lines.append("")
    lines.append("- `Gc`: GAIC GT crop 집합만 사용합니다. local normalization은 GT annotation 집합 내부에서만 계산합니다.")
    lines.append("- `Ge`: GAIC GT crop을 production FREE-form candidate pool에 주입합니다. local normalization은 전체 production pool과 주입된 GT를 함께 놓고 계산합니다.")
    lines.append("- 5장과 6장은 `GT-only ranking` 관점입니다. 실제 production end-to-end 선택은 7장에서 따로 봅니다.")
    lines.append("- `score_z_local`과 `score_policy_z_local`은 mean/std가 아니라 median/MAD 기반의 robust z-score 입니다.")
    lines.append("- `sigmoid(alpha * robust_z_local)` 계열 값은 calibrated probability가 아니라 `pseudo-probability` 또는 `top-q propensity`로 해석해야 합니다.")
    lines.append("")
    lines.append("## 3. 지표 설명")
    lines.append("")
    lines.extend(
        markdown_table(
            ["지표", "범위", "좋은 방향", "해석"],
            [
                ["Spearman", "[-1, 1]", "높을수록 좋음", "이미지 내부 crop 순서가 인간 MOS 순서와 얼마나 비슷한지 봅니다."],
                ["Kendall tau-b", "[-1, 1]", "높을수록 좋음", "동점이 있는 경우까지 포함해 순서 일치도를 봅니다."],
                ["Weighted pairwise acc", "[0, 1]", "높을수록 좋음", "MOS 차이가 큰 pair를 더 중요하게 보는 쌍대 비교 정확도입니다."],
                ["Hit@1", "[0, 1]", "높을수록 좋음", "예측 최고 crop이 GT 최고 crop과 맞는지 봅니다."],
                ["Top-q Jaccard", "[0, 1]", "높을수록 좋음", "예측 상위 집합과 GT 상위 집합의 겹침 정도입니다."],
                ["Brier / NLL / ECE", "[0, inf)", "낮을수록 좋음", "`pseudo-probability = sigmoid(alpha * robust_z_local)`를 확률처럼 해석했을 때의 품질입니다."],
                ["MAE / RMSE", "[0, inf)", "낮을수록 좋음", "isotonic mapping 이후 전역 MOS 보정 오차입니다."],
                ["QWK / CCC", "[-1, 1]", "높을수록 좋음", "보정된 MOS 예측이 GT와 얼마나 잘 맞는지 봅니다."],
                ["Prod winner GT percentile", "[0, 1]", "높을수록 좋음", "production 최종 winner가 GT MOS 분포에서 얼마나 높은 구간에 대응되는지 봅니다."],
            ],
        )
    )
    lines.append("")
    lines.append("## 4. 핵심 결과 요약")
    lines.append("")
    key_rows = []
    for protocol in ("Gc", "Ge"):
        trend = summary[protocol]["trend_by_field"]["score_rank"]
        local_prob = summary[protocol]["local_probability"]["test"]
        calib = summary[protocol]["global_calibration"]["test"]
        key_rows.append(
            [
                protocol,
                f"{trend['mean_spearman']:.3f} {format_ci(tuple(trend['confidence_intervals']['mean_spearman']))}",
                f"{trend['mean_kendall_tau_b']:.3f} {format_ci(tuple(trend['confidence_intervals']['mean_kendall_tau_b']))}",
                f"{trend['mean_weighted_pair_acc']:.3f} {format_ci(tuple(trend['confidence_intervals']['mean_weighted_pair_acc']))}",
                f"{trend['mean_hit_at_1']:.3f} {format_ci(tuple(trend['confidence_intervals']['mean_hit_at_1']))}",
                f"{trend['mean_topq_jaccard']:.3f} {format_ci(tuple(trend['confidence_intervals']['mean_topq_jaccard']))}",
                f"{local_prob['brier_topq']:.3f}",
                f"{local_prob['ece_topq']:.3f}",
                f"{calib['mae']:.3f}",
                f"{calib['qwk']:.3f}",
            ]
        )
    lines.extend(markdown_table(["프로토콜", "Spearman", "Kendall", "PairAcc", "Hit@1", "Top-q", "Brier", "ECE", "MAE", "QWK"], key_rows))
    lines.append("")
    lines.append("해석:")
    lines.append("")
    lines.append("- `Gc`는 높고 `Ge`가 떨어지면, scorer 자체는 GT crop을 잘 정렬하지만 production candidate space 또는 policy layer에서 drift가 생긴다는 뜻입니다.")
    lines.append("- `score_policy`가 `score_rank`보다 MOS와 더 잘 맞는 구간이 실제로 존재하므로 calibration도 `rank`와 `policy`를 같이 보는 편이 안전합니다.")
    lines.append("- raw `score_rank`, raw `score_policy`는 candidate-local score이고, end-to-end 차이는 local-normalized 계열과 실제 production winner에서 드러납니다.")
    lines.append("")
    lines.append("## 5. 공식 Test Subset (`GT-only ranking`)")
    lines.append("")
    test_rows = []
    for protocol in ("Gc", "Ge"):
        test_summary = summary[protocol]["split_breakdown"]["test"]
        test_rows.append(
            [
                protocol,
                str(test_summary["image_count"]),
                f"{test_summary['mean_spearman']:.3f}",
                f"{test_summary['mean_kendall_tau_b']:.3f}",
                f"{test_summary['mean_weighted_pair_acc']:.3f}",
                f"{test_summary['mean_hit_at_1']:.3f}",
                f"{test_summary['mean_topq_jaccard']:.3f}",
            ]
        )
    lines.extend(markdown_table(["프로토콜", "이미지 수", "Spearman", "Kendall", "PairAcc", "Hit@1", "Top-q"], test_rows))
    lines.append("")
    lines.append("## 6. Score 필드 비교")
    lines.append("")
    for protocol in ("Gc", "Ge"):
        rows = []
        for field in PRIMARY_SCORE_FIELDS:
            trend = summary[protocol]["trend_by_field"][field]
            rows.append(
                [
                    field,
                    f"{trend['mean_spearman']:.3f}",
                    f"{trend['mean_kendall_tau_b']:.3f}",
                    f"{trend['mean_weighted_pair_acc']:.3f}",
                    f"{trend['mean_hit_at_1']:.3f}",
                    f"{trend['mean_topq_jaccard']:.3f}",
                ]
            )
        lines.append(f"### {protocol}")
        lines.append("")
        lines.extend(markdown_table(["필드", "Spearman", "Kendall", "PairAcc", "Hit@1", "Top-q"], rows))
        lines.append("")
    lines.append("## 7. Ge Production-Inclusive Selection")
    lines.append("")
    ge_prod = summary["Ge"].get("prod_selection", {})
    ge_prod_policy = summary["Ge"].get("prod_selection_policy", {})
    ge_prod_pairwise = summary["Ge"].get("prod_selection_rank_vs_policy", {})
    lines.extend(
        markdown_table(
            ["지표", "값", "해석"],
            [
                ["이미지 수", str(ge_prod.get("image_count", 0)), "production winner를 집계한 이미지 수입니다."],
                ["rank winner non-GT rate", f"{safe_float(ge_prod.get('non_gt_winner_rate', 0.0), 0.0):.3f}", "`score_rank` 기준 최종 winner가 GT annotation 그 자체가 아닌 production 후보인 비율입니다."],
                ["policy winner non-GT rate", f"{safe_float(ge_prod_policy.get('non_gt_winner_rate', 0.0), 0.0):.3f}", "`score_policy` 기준 최종 winner가 GT annotation 그 자체가 아닌 production 후보인 비율입니다."],
                ["rank winner IoU to GT MOS best", f"{safe_float(ge_prod.get('gt_best_iou_mean', 0.0), 0.0):.3f}", "`score_rank` winner가 사람 MOS 최고 GT와 평균적으로 얼마나 겹치는지입니다."],
                ["policy winner IoU to GT MOS best", f"{safe_float(ge_prod_policy.get('gt_best_iou_mean', 0.0), 0.0):.3f}", "`score_policy` winner가 사람 MOS 최고 GT와 평균적으로 얼마나 겹치는지입니다."],
                ["rank winner GT percentile", f"{safe_float(ge_prod.get('matched_gt_mos_percentile_mean', 0.0), 0.0):.3f}", "`score_rank` winner와 가장 많이 겹치는 GT의 MOS percentile 평균입니다."],
                ["policy winner GT percentile", f"{safe_float(ge_prod_policy.get('matched_gt_mos_percentile_mean', 0.0), 0.0):.3f}", "`score_policy` winner와 가장 많이 겹치는 GT의 MOS percentile 평균입니다."],
                ["rank-policy disagreement", f"{safe_float(ge_prod_pairwise.get('policy_vs_rank_disagreement_rate', 0.0), 0.0):.3f}", "두 production winner가 서로 다른 비율입니다."],
                ["policy better GT percentile", f"{safe_float(ge_prod_pairwise.get('policy_better_percentile_rate', 0.0), 0.0):.3f}", "`score_policy` winner가 `score_rank` winner보다 GT percentile이 높은 비율입니다."],
                ["policy better IoU", f"{safe_float(ge_prod_pairwise.get('policy_better_iou_rate', 0.0), 0.0):.3f}", "`score_policy` winner가 `score_rank` winner보다 GT best IoU가 높은 비율입니다."],
                ["GT-vs-rank disagreement", f"{safe_float(ge_prod.get('gt_vs_prod_disagreement_rate', 0.0), 0.0):.3f}", "Ge GT best와 `score_rank` production winner가 다른 비율입니다."],
            ],
        )
    )
    lines.append("")
    source_rows = [
        [name, str(count), compact_float(ge_prod.get("winner_source_family_topq_rate", {}).get(name), 3)]
        for name, count in sorted(ge_prod.get("winner_source_family_counts", {}).items(), key=lambda item: (-item[1], item[0]))
    ]
    if source_rows:
        lines.append("rank winner source family 분포:")
        lines.append("")
        lines.extend(markdown_table(["source family", "count", "matched GT top-q rate"], source_rows))
        lines.append("")
    policy_source_rows = [
        [name, str(count), compact_float(ge_prod_policy.get("winner_source_family_topq_rate", {}).get(name), 3)]
        for name, count in sorted(ge_prod_policy.get("winner_source_family_counts", {}).items(), key=lambda item: (-item[1], item[0]))
    ]
    if policy_source_rows:
        lines.append("policy winner source family 분포:")
        lines.append("")
        lines.extend(markdown_table(["source family", "count", "matched GT top-q rate"], policy_source_rows))
        lines.append("")
    score_mode_rows = [
        [
            mode_name,
            str(safe_int(mode_metrics.get("image_count", 0))),
            compact_float(mode_metrics.get("share"), 3),
            compact_float(mode_metrics.get("mean_spearman"), 3),
            compact_float(mode_metrics.get("mean_weighted_pair_acc"), 3),
            compact_float(mode_metrics.get("brier_topq"), 3),
            compact_float(mode_metrics.get("ece_topq"), 3),
            compact_float(mode_metrics.get("prod_gt_best_iou_mean"), 3),
            compact_float(mode_metrics.get("prod_matched_gt_percentile_mean"), 3),
        ]
        for mode_name, mode_metrics in sorted(summary["Ge"].get("score_mode_distribution", {}).items())
    ]
    if score_mode_rows:
        lines.append("score_mode 분포:")
        lines.append("")
        lines.extend(markdown_table(["score_mode", "images", "share", "Spearman", "PairAcc", "Brier", "ECE", "prod IoU->GTbest", "prod GT percentile"], score_mode_rows))
        lines.append("")
    lines.append("## 8. 로컬 pseudo-probability 및 전역 보정")
    lines.append("")
    lines.append("- alpha sweep의 validation objective는 image-level validation split에서의 `Brier(top-q)`입니다.")
    lines.append(f"- Top-q 임계치는 `{summary['config']['top_quantile']}` 입니다.")
    lines.append("- 아래 표의 `const Brier/NLL`은 같은 positive prevalence만 쓰는 trivial constant prior baseline 입니다. pseudo-probability가 이 값보다 나쁘면 순위는 있어도 확률 해석은 약하다는 뜻입니다.")
    lines.append("")
    prob_rows = []
    for protocol in ("Gc", "Ge"):
        variants = summary[protocol].get("local_probability_variants", {})
        calib_variants = summary[protocol].get("global_calibration_variants", {})
        for variant_key, _z_field, label in PROBABILITY_VARIANTS:
            lp = variants.get(variant_key, {})
            gcali = calib_variants.get(variant_key, {})
            prob_rows.append(
                [
                    protocol,
                    label,
                    f"{safe_float(lp.get('best_alpha', 0.0), 0.0):.2f}",
                    f"{safe_float(lp.get('test', {}).get('brier_topq', 0.0), 0.0):.3f}",
                    f"{safe_float(lp.get('test', {}).get('constant_prior_brier', 0.0), 0.0):.3f}",
                    f"{safe_float(lp.get('test', {}).get('nll_topq', 0.0), 0.0):.3f}",
                    f"{safe_float(lp.get('test', {}).get('constant_prior_nll', 0.0), 0.0):.3f}",
                    f"{safe_float(lp.get('test', {}).get('ece_topq', 0.0), 0.0):.3f}",
                    f"{safe_float(gcali.get('test', {}).get('mae', 0.0), 0.0):.3f}",
                    f"{safe_float(gcali.get('test', {}).get('qwk', 0.0), 0.0):.3f}",
                ]
            )
    lines.extend(markdown_table(["프로토콜", "variant", "alpha", "Brier", "const Brier", "NLL", "const NLL", "ECE", "MAE", "QWK"], prob_rows))
    lines.append("")
    lines.append("## 9. 후보 공간 메모")
    lines.append("")
    lines.append(f"- raw FREE-form candidate 수 평균: `{summary['dataset']['raw_free_candidate_count_mean']:.2f}`")
    lines.append(f"- distilled training-label FREE candidate 수 평균: `{summary['dataset']['training_label_free_candidate_count_mean']:.2f}`")
    lines.append("- `train_regression.jsonl`은 full production pool이 아니라 distilled teacher-label view입니다. 그래서 `Ge` 평가는 benchmark 명세대로 full-pool local normalization을 보존하기 위해 raw candidate JSONL을 다시 사용합니다.")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Coverage 지표", "값"],
            [
                ["Top-1 oracle IoU 평균", f"{summary['coverage']['top1_oracle_iou_mean']:.3f}"],
                ["Top-1 coverage @ 0.5", f"{summary['coverage']['top1_coverage_at_05_mean']:.3f}"],
                ["Top-1 coverage @ 0.7", f"{summary['coverage']['top1_coverage_at_07_mean']:.3f}"],
                ["Top-q any coverage @ 0.5", f"{summary['coverage']['topq_any_coverage_at_05_mean']:.3f}"],
            ],
        )
    )
    lines.append("")
    lines.append("## 10. Failure Cohorts (`Ge score_rank`)")
    lines.append("")
    cohort_rows = []
    for cohort_name, cohort in summary["Ge"].get("failure_cohorts", {}).items():
        if int(cohort.get("image_count", 0)) <= 0:
            continue
        cohort_rows.append(
            [
                cohort_name,
                str(cohort["image_count"]),
                compact_float(cohort.get("mean_spearman"), 3),
                compact_float(cohort.get("mean_weighted_pair_acc"), 3),
                compact_float(cohort.get("mean_hit_at_1"), 3),
                compact_float(cohort.get("brier_topq"), 3),
                compact_float(cohort.get("ece_topq"), 3),
                compact_float(cohort.get("prod_gt_best_iou_mean"), 3),
                compact_float(cohort.get("prod_matched_gt_percentile_mean"), 3),
            ]
        )
    lines.extend(markdown_table(["cohort", "images", "Spearman", "PairAcc", "Hit@1", "Brier", "ECE", "prod IoU->GTbest", "prod GT percentile"], cohort_rows))
    lines.append("")
    lines.append("### Subject Support Diagnostics (`bbox IoU` vs `latent support mass`)")
    lines.append("")
    lines.append("- `latent GT support`는 이미지별 상위 MOS GT crop들을 64x64 grid 위에 누적한 inclusion heatmap 입니다.")
    lines.append("- `support mass recall`은 raw anchor / candidate anchor / scoring region이 그 soft support를 얼마나 많이 담는지 봅니다.")
    lines.append("- distributed foreground 장면에서는 `support_iou_mean`이 낮아도 `support_mass_recall_mean`이 높을 수 있습니다. 이 경우 bbox IoU보다 mass 계열 지표가 더 직접적인 진단 값입니다.")
    lines.append("")
    support_diag = summary.get("subject_support_diagnostics", {})
    support_iou_rows = []
    support_mass_rows = []
    for bucket in ("overall", "scene_general", "other_ambiguous_tiny", "neutralized"):
        bucket_diag = support_diag.get(bucket, {}) if isinstance(support_diag.get(bucket), dict) else {}
        support_iou_rows.append(
            [
                bucket,
                str(safe_int(bucket_diag.get("image_count", 0))),
                compact_float(bucket_diag.get("raw_iou_mean"), 3),
                compact_float(bucket_diag.get("candidate_iou_mean"), 3),
                compact_float(bucket_diag.get("support_iou_mean"), 3),
                compact_float(bucket_diag.get("support_latent_bbox_iou_mean"), 3),
            ]
        )
        support_mass_rows.append(
            [
                bucket,
                compact_float(bucket_diag.get("gt_best_mass_recall_mean"), 3),
                compact_float(bucket_diag.get("raw_mass_recall_mean"), 3),
                compact_float(bucket_diag.get("candidate_mass_recall_mean"), 3),
                compact_float(bucket_diag.get("support_mass_recall_mean"), 3),
                compact_float(bucket_diag.get("support_low_mass_rate"), 3),
                compact_float(bucket_diag.get("support_centroid_inside_rate"), 3),
                compact_float(bucket_diag.get("support_centroid_distance_mean"), 3),
            ]
        )
    lines.extend(markdown_table(["bucket", "images", "raw IoU", "candidate IoU", "support IoU", "support latent bbox IoU"], support_iou_rows))
    lines.append("")
    lines.extend(markdown_table(["bucket", "GTbest mass", "raw mass", "candidate mass", "support mass", "support<0.5", "centroid-in", "centroid dist"], support_mass_rows))
    lines.append("")
    lines.append("## 11. 모드별 분해 (`score_rank`)")
    lines.append("")
    for protocol in ("Gc", "Ge"):
        rows = []
        for mode_name, mode_metrics in sorted(summary[protocol]["mode_breakdown"].items()):
            rows.append(
                [
                    mode_name,
                    str(mode_metrics["image_count"]),
                    f"{mode_metrics['mean_spearman']:.3f}",
                    f"{mode_metrics['mean_weighted_pair_acc']:.3f}",
                    f"{mode_metrics['mean_hit_at_1']:.3f}",
                ]
            )
        lines.append(f"### {protocol}")
        lines.append("")
        lines.extend(markdown_table(["모드", "이미지 수", "Spearman", "PairAcc", "Hit@1"], rows))
        lines.append("")
    lines.append("## 12. 기존 Training Label QA")
    lines.append("")
    lines.append(f"- 기존 QA summary 파일: `{summary['sources']['qa_summary_json']}`")
    lines.append(f"- 기존 validation summary 파일: `{summary['sources']['validation_summary_json']}`")
    if qa_summary:
        counts = qa_summary.get("counts", {})
        lines.append(f"- Pairwise row 수: `{counts.get('pairwise', 0)}`")
        lines.append(f"- Regression row 수: `{counts.get('regression', 0)}`")
    if validation_summary:
        lines.append(f"- Validation 상태: `{validation_summary.get('status', 'unknown')}`")
        lines.append(f"- Validation error 수: `{validation_summary.get('error_count', 0)}`")
    lines.append("")
    lines.append("## 13. 그림")
    lines.append("")
    lines.append(f"![주요 지표 비교]({plot_paths['protocol_compare'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![Gc Spearman 히스토그램]({plot_paths['gc_spearman_hist'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![Ge Spearman 히스토그램]({plot_paths['ge_spearman_hist'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![Gc 산점도]({plot_paths['gc_scatter'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![Ge 산점도]({plot_paths['ge_scatter'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![Ge rank 신뢰도 곡선]({plot_paths['ge_reliability'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append(f"![Ge policy 신뢰도 곡선]({plot_paths['ge_policy_reliability'].relative_to(output_dir).as_posix()})")
    lines.append("")
    lines.append("## 14. 샘플 갤러리")
    lines.append("")
    lines.append("샘플 overlay 읽는 법:")
    lines.append("")
    lines.append("- 하늘색 bbox `raw anchor`: detector/union 기반의 원시 subject anchor 입니다. 후보 생성의 출발점이며, 잘못 잡히면 subject-driven family가 오염될 수 있습니다.")
    lines.append("- 분홍 bbox `candidate anchor`: 실제 후보 생성 seed 중심입니다. raw anchor와 다르면 2차 튜닝 로직이 보수적으로 보정한 상태입니다.")
    lines.append("- 보라색 bbox `scoring region`: scorer가 실제 `subject preservation` 항에서 참고하는 region 입니다. distributed scene이면 saliency foreground support, raw-saliency disagreement면 union grouping 결과가 들어갈 수 있습니다.")
    lines.append("- `score_mode=support_map` 이면 실제 `cov` 계산은 bbox가 아니라 c7 saliency에서 내려온 low-res soft support map으로 수행됩니다. 이 경우 보라색 bbox는 시각화용 support envelope일 뿐, 점수는 map 기준으로 계산됩니다.")
    lines.append("- 노랑 반투명 mask `c7 saliency`: BiRefNet/IS-Net/OpenCV fallback으로 샘플 이미지에 대해 다시 추론한 saliency foreground 영역입니다. bbox가 아니라 mask 자체를 오버레이합니다.")
    lines.append("- 각 bbox 라벨에는 `sigz`와 `rank`를 같이 적습니다. 여기서 `sigz`는 기본적으로 `score_policy_sigmoid_z_local`인 policy pseudo probability 입니다. 해당 값이 없을 때만 `score_sigmoid_z_local = sigmoid(score_z_local)` 계열 fallback을 사용합니다. training label의 `score_prob`도 같은 기본값을 사용하며, calibrated probability가 아니라 pseudo-probability로 해석해야 합니다.")
    lines.append("- GT 후보인 경우 라벨은 `MOS`, `sigz`, `rank`를 모두 적습니다. 즉 사람 평점과 teacher의 local-normalized 점수를 동시에 볼 수 있습니다.")
    lines.append("- 비GT 후보인 경우 라벨은 `sigz`, `rank`만 적습니다. MOS가 없기 때문입니다.")
    lines.append("- 초록 bbox `GT MOS best`: 해당 이미지의 GAIC GT 중 사람 MOS가 가장 높은 정답 크롭입니다.")
    lines.append("- 파랑 bbox `Gc best`: `Gc` 프로토콜에서 GT 크롭들만 놓고 점수화했을 때 `score_rank`가 가장 높은 GT 크롭입니다.")
    lines.append("- 주황 bbox `Ge GT best`: `Ge` 프로토콜에서 GT 크롭을 프로덕션 후보 풀에 주입한 뒤, GT 크롭들만 다시 비교했을 때 `score_rank`가 가장 높은 GT 크롭입니다.")
    lines.append("- 빨강 bbox `Ge prod best`: `Ge` 프로토콜에서 프로덕션 자유형 후보와 주입된 GT를 모두 합친 전체 후보 중 최종 winner입니다.")
    lines.append("- `score_rank`와 `sigz`는 MOS나 IoU가 아니라 모델의 후보 내부 순위화 계열 점수입니다. 절대값보다 같은 이미지 안에서의 상대 순서를 봐야 합니다.")
    lines.append("- overlay 상단 패널의 `Gc/Ge spearman`은 해당 이미지에서 예측 순위와 GT MOS 순위의 일치도이고, `coverage top1_iou`는 `Ge prod best` 박스가 GT 최고 MOS 박스와 얼마나 겹치는지의 IoU입니다.")
    lines.append("- 샘플별 상세 표의 `latent mass recall`은 GT 상위 crop latent support 기준입니다. distributed scene에서는 이 값이 bbox IoU보다 더 중요한 진단 근거가 됩니다.")
    lines.append("")
    for sample_file in sample_files:
        lines.append(f"![{sample_file.stem}]({sample_file.relative_to(output_dir).as_posix()})")
        lines.append("")
    lines.append("### 샘플별 상세 진단")
    lines.append("")
    lines.append("아래 섹션은 각 샘플에 대해 `GT MOS best`, `Ge GT best`, `Ge prod best`를 다시 점수화해, 왜 전체 production 후보 1위가 GT 최고 후보와 달라졌는지 추적할 수 있도록 만든 진단 결과입니다. raw anchor, candidate anchor, scoring region, c7 saliency를 함께 표시합니다.")
    lines.append("")
    for sample_analysis in sample_analyses:
        append_sample_analysis_markdown(lines, output_dir=output_dir, sample_analysis=sample_analysis)
    lines.append("## 15. 산출물 파일")
    lines.append("")
    lines.append("- `benchmark_summary.json`: 전체 기계 판독용 요약입니다.")
    lines.append("- `candidate_eval_rows.jsonl`: `Gc`와 `Ge`의 GT-only candidate 행과 예측 점수를 담습니다.")
    lines.append("- `ge_full_candidates_topk.jsonl`: `Ge` full production+GT decision pool의 상위 후보를 이미지별로 저장한 prod-inclusive artifact 입니다.")
    lines.append("- `per_image_metrics.csv`: 이미지/프로토콜 단위의 1차 지표 요약입니다.")
    lines.append("- `benchmark_summary.json`의 `split_breakdown`: 공식 train/test 부분집합 기준 `score_rank` 핵심 지표입니다.")
    lines.append("- `benchmark_summary.json`의 `prod_selection`: production 최종 winner의 GT percentile / IoU / source family 분해입니다.")
    lines.append("- `benchmark_summary.json`의 `prod_selection_policy`: `score_policy` 기준 production winner 분해입니다.")
    lines.append("- `benchmark_summary.json`의 `prod_selection_rank_vs_policy`: rank winner와 policy winner 비교 표입니다.")
    lines.append("- `benchmark_summary.json`의 `failure_cohorts`: low-confidence subject, tiny raw anchor, neutralized, source-family cohort를 따로 집계한 결과입니다.")
    lines.append("- `benchmark_summary.json`의 `score_mode_distribution`: `subject_preservation / support_map / neutralized`별 품질 분해입니다.")
    lines.append("- `benchmark_summary.json`의 `subject_support_diagnostics`: bbox IoU와 latent support mass를 함께 보는 subject-region 진단 요약입니다.")
    lines.append("- `plots/`: 본 리포트에서 사용한 진단용 시각화입니다.")
    lines.append("- `samples/`: 오류 사례 분석용 sample overlay 이미지입니다.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    start_ts = time.time()

    candidates_jsonl = Path(args.candidates_jsonl).resolve()
    features_jsonl = Path(args.features_jsonl).resolve()
    teacher_jsonl = Path(args.teacher_jsonl).resolve()
    training_label_dir = Path(args.training_label_dir).resolve()
    gaic_train_json = Path(args.gaic_train_json).resolve()
    gaic_test_json = Path(args.gaic_test_json).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    cfg = load_scorer_config(teacher_jsonl)

    candidate_map = load_map_by_image_id(candidates_jsonl)
    feature_map = load_map_by_image_id(features_jsonl)
    training_regression_path = training_label_dir / "train_regression.jsonl"
    training_groups = load_training_regression_groups(training_regression_path)
    qa_summary_path = training_label_dir / "qa_summary.json"
    validation_summary_path = training_label_dir / "validation_summary.json"
    qa_summary = load_json(qa_summary_path) if qa_summary_path.exists() else {}
    validation_summary = load_json(validation_summary_path) if validation_summary_path.exists() else {}

    gt_images, gt_anns = load_gaic_annotations(gaic_train_json, gaic_test_json)

    overlap_ids = sorted(set(candidate_map.keys()).intersection(feature_map.keys()).intersection(gt_images.keys()).intersection(gt_anns.keys()))
    if args.max_images > 0:
        overlap_ids = overlap_ids[: int(args.max_images)]

    candidate_eval_rows: List[Dict[str, Any]] = []
    ge_full_topk_rows: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []
    per_image_csv_rows: List[Dict[str, Any]] = []

    raw_free_counts: List[int] = []
    training_free_counts: List[int] = []
    coverage_rows: List[Dict[str, Any]] = []

    for image_id in overlap_ids:
        cand_rec = candidate_map[image_id]
        feat_rec = feature_map[image_id]
        image_meta = gt_images[image_id]
        anns = gt_anns[image_id]
        width = int(image_meta["width"])
        height = int(image_meta["height"])
        gt_candidates = build_gt_candidates(anns, width=width, height=height)
        gt_lookup = {f"gaic_gt_{safe_int(ann.get('id', 0))}": ann for ann in anns}

        raw_free_pool = copy.deepcopy((cand_rec.get("candidates_by_ar") or {}).get("FREE") or [])
        raw_free_counts.append(len(raw_free_pool))
        training_free_counts.append(len(training_groups.get(image_id, [])))

        gc_scored, gc_route, _ = score_candidates_freeform(
            cand_rec=cand_rec,
            feat_rec=feat_rec,
            candidates=gt_candidates,
            cfg=cfg,
            softmax_tau=float(args.softmax_tau),
        )
        ge_scored, ge_route, ge_teacher_ctx = score_candidates_freeform(
            cand_rec=cand_rec,
            feat_rec=feat_rec,
            candidates=raw_free_pool + gt_candidates,
            cfg=cfg,
            softmax_tau=float(args.softmax_tau),
        )
        ge_decision = derive_decision_from_full_pool(
            scored=ge_scored,
            route=ge_route,
            teacher_ctx=ge_teacher_ctx,
            cfg=cfg,
        )

        gc_gt_rows = [row for row in gc_scored if str(row.get("candidate_id", "")).startswith("gaic_gt_")]
        ge_gt_rows = [row for row in ge_scored if str(row.get("candidate_id", "")).startswith("gaic_gt_")]
        gc_gt_rows.sort(key=score_rank_value, reverse=True)
        ge_gt_rows.sort(key=score_rank_value, reverse=True)

        official_split = str(image_meta["official_split"])
        calib_split = assign_calibration_split(official_split, image_id)
        subject_mode = str(ge_route.get("subject_mode", gc_route.get("subject_mode", "other_ambiguous")))
        subject_visuals = collect_subject_visuals(cand_rec=cand_rec, feat_rec=feat_rec, route=ge_route)

        gt_best_ann = anns[0]
        gt_best_candidate_id = f"gaic_gt_{safe_int(gt_best_ann.get('id', 0))}"
        gc_gt_row_by_id = {str(row.get("candidate_id", "")): row for row in gc_gt_rows}
        gt_best_scored_row = gc_gt_row_by_id.get(gt_best_candidate_id, gc_gt_rows[0])
        gt_best_row = make_output_row(
            image_id=image_id,
            protocol="GT",
            split=official_split,
            calib_split=calib_split,
            subject_mode=subject_mode,
            row=gt_best_scored_row,
            ann=gt_best_ann,
        )
        gc_best_row = make_output_row(
            image_id=image_id,
            protocol="Gc",
            split=official_split,
            calib_split=calib_split,
            subject_mode=subject_mode,
            row=gc_gt_rows[0],
            ann=gt_lookup[str(gc_gt_rows[0]["candidate_id"])],
        )
        ge_best_row = make_output_row(
            image_id=image_id,
            protocol="Ge",
            split=official_split,
            calib_split=calib_split,
            subject_mode=subject_mode,
            row=ge_gt_rows[0],
            ann=gt_lookup[str(ge_gt_rows[0]["candidate_id"])],
        )
        subject_support_metrics = compute_subject_box_diagnostics(
            raw_anchor=subject_visuals.get("raw_anchor_norm_xyxy"),
            candidate_anchor=subject_visuals.get("candidate_anchor_norm_xyxy"),
            support_region=subject_visuals.get("support_region_norm_xyxy") or subject_visuals.get("scoring_region_norm_xyxy"),
            gt_best_box=gt_best_row["bbox_norm_xyxy"],
            anns=anns,
            width=width,
            height=height,
        )
        decision_pool = sorted(
            ge_decision.get("rank_pool", []),
            key=lambda row: safe_float(row.get("scores", {}).get("final", -1e9), -1e9),
            reverse=True,
        )
        policy_pool = sorted(decision_pool, key=score_policy_value, reverse=True)
        rank_best_candidate = ge_decision["best_candidate"]
        policy_best_candidate = policy_pool[0] if policy_pool else rank_best_candidate
        prod_best_payload = build_prod_selection_payload(
            candidate=rank_best_candidate,
            anns=anns,
            width=width,
            height=height,
            top_quantile=float(args.top_quantile),
            gt_best_row=gt_best_row,
            decision_type=str(ge_decision["decision"]["decision_type"]),
        )
        policy_best_payload = build_prod_selection_payload(
            candidate=policy_best_candidate,
            anns=anns,
            width=width,
            height=height,
            top_quantile=float(args.top_quantile),
            gt_best_row=gt_best_row,
            decision_type="policy_rank_pool",
        )
        prod_source_family = str(prod_best_payload["winner_source_family"])
        policy_source_family = str(policy_best_payload["winner_source_family"])
        policy_rank_disagree = str(prod_best_payload["candidate_id"]) != str(policy_best_payload["candidate_id"])
        for rank_idx, row in enumerate(decision_pool[: max(1, int(args.ge_full_topk))], start=1):
            topk_match = gt_match_summary(
                candidate=row,
                anns=anns,
                width=width,
                height=height,
                top_quantile=float(args.top_quantile),
            )
            ge_full_topk_rows.append(
                {
                    "image_id": image_id,
                    "official_split": official_split,
                    "calibration_split": calib_split,
                    "subject_mode": subject_mode,
                    "rank_in_decision_pool": rank_idx,
                    "candidate_id": str(row.get("candidate_id", "")),
                    "source": str(row.get("source", "")),
                    "winner_source_family": canonical_source_family(row.get("source", "")),
                    "is_gt_candidate": str(row.get("candidate_id", "")).startswith("gaic_gt_"),
                    "bbox_norm_xyxy": [round(float(v), 6) for v in row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])],
                    "score_final": round(safe_float(row.get("scores", {}).get("final", 0.0), 0.0), 6),
                    "score_rank": round(score_rank_value(row), 6),
                    "score_policy": round(score_policy_value(row), 6),
                    "score_z_local": round(safe_float(row.get("score_z_local", 0.0), 0.0), 6),
                    "score_policy_z_local": round(safe_float(row.get("score_policy_z_local", 0.0), 0.0), 6),
                    "matched_gt_iou": topk_match["best_iou"],
                    "matched_gt_annotation_id": topk_match["matched_gt_annotation_id"],
                    "matched_gt_mos": topk_match["matched_gt_mos"],
                    "matched_gt_rank": topk_match["matched_gt_rank"],
                    "matched_gt_mos_percentile": topk_match["matched_gt_mos_percentile"],
                    "matched_gt_is_topq": topk_match["matched_gt_is_topq"],
                    "iou_to_gt_best": round(iou_xyxy(row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), gt_best_row["bbox_norm_xyxy"]), 6),
                }
            )

        protocol_payloads: Dict[str, Dict[str, Any]] = {}
        for protocol_name, rows in (("Gc", gc_gt_rows), ("Ge", ge_gt_rows)):
            rendered_rows = [
                make_output_row(
                    image_id=image_id,
                    protocol=protocol_name,
                    split=official_split,
                    calib_split=calib_split,
                    subject_mode=subject_mode,
                    row=row,
                    ann=gt_lookup[str(row["candidate_id"])],
                )
                for row in rows
            ]
            candidate_eval_rows.extend(rendered_rows)
            metrics_by_field = {
                field: compute_protocol_trend_metrics(rendered_rows, field, float(args.top_quantile))
                for field in PRIMARY_SCORE_FIELDS
            }
            protocol_payloads[protocol_name] = {
                "rows": rendered_rows,
                "metrics_by_field": metrics_by_field,
                "best_gt_row": rendered_rows[0],
            }
            per_image_csv_rows.append(
                {
                    "image_id": image_id,
                    "protocol": protocol_name,
                    "official_split": official_split,
                    "calibration_split": calib_split,
                    "subject_mode": subject_mode,
                    "mode_bucket": mode_bucket(subject_mode),
                    "num_gt": len(rendered_rows),
                    "spearman": metrics_by_field["score_rank"]["spearman"],
                    "kendall_tau_b": metrics_by_field["score_rank"]["kendall_tau_b"],
                    "weighted_pair_acc": metrics_by_field["score_rank"]["weighted_pair_acc"],
                    "hit_at_1": metrics_by_field["score_rank"]["hit_at_1"],
                    "topq_jaccard": metrics_by_field["score_rank"]["topq_jaccard"],
                    "subject_mode_conf": round(safe_float(ge_route.get("subject_mode_conf", 0.0), 0.0), 6),
                    "raw_anchor_area": round(box_area(subject_visuals["raw_anchor_norm_xyxy"]), 6) if subject_visuals.get("raw_anchor_norm_xyxy") else 0.0,
                    "subject_reliability": round(safe_float(subject_visuals.get("subject_reliability", 0.0), 0.0), 6),
                    "subject_repr_type": str(subject_visuals.get("subject_repr_type", "")),
                    "score_mode": str(subject_visuals.get("score_mode", "")),
                    "subject_terms_neutralized": bool(subject_visuals.get("score_mode", "") == "neutralized"),
                    "subject_agreement_iou": round(safe_float(subject_visuals.get("subject_agreement_iou", 0.0), 0.0), 6),
                    "subject_center_distance": round(safe_float(subject_visuals.get("subject_center_distance", 0.0), 0.0), 6),
                    "subject_disagreement_severe": bool(subject_visuals.get("subject_disagreement_severe", False)),
                    "raw_anchor_support_mass": round(safe_float(subject_visuals.get("raw_anchor_support_mass", 0.0), 0.0), 6),
                    "candidate_anchor_mass_recall": round(safe_float(subject_support_metrics["candidate_anchor"]["mass_recall"], 0.0), 6),
                    "candidate_anchor_support_mass": round(safe_float(subject_visuals.get("candidate_anchor_support_mass", 0.0), 0.0), 6),
                    "support_mass_recall": round(safe_float(subject_support_metrics["support_region"]["mass_recall"], 0.0), 6),
                    "effective_support_mass": round(safe_float(subject_visuals.get("effective_support_mass", 0.0), 0.0), 6),
                    "support_fallback_reason": str(subject_visuals.get("support_fallback_reason", "")),
                    "support_centroid_inside": round(safe_float(subject_support_metrics["support_region"]["centroid_inside"], 0.0), 6),
                    "support_centroid_distance": round(safe_float(subject_support_metrics["support_region"]["centroid_distance"], 0.0), 6),
                    "blank_ratio_est": round(safe_float(subject_visuals.get("saliency_blank_ratio_est", ge_route.get("router_signals", {}).get("blank_ratio_est", 0.0)), 0.0), 6),
                    "ge_winner_source_family": prod_source_family,
                    "ge_policy_winner_source_family": policy_source_family,
                    "ge_policy_rank_disagree": bool(policy_rank_disagree),
                    "ge_prod_gt_best_iou": prod_best_payload["iou_to_gt_best"],
                    "ge_prod_matched_gt_percentile": prod_best_payload["matched_gt_mos_percentile"],
                    "ge_policy_gt_best_iou": policy_best_payload["iou_to_gt_best"],
                    "ge_policy_matched_gt_percentile": policy_best_payload["matched_gt_mos_percentile"],
                }
            )

        coverage = compute_candidate_coverage(
            production_candidates=raw_free_pool,
            anns=anns,
            width=width,
            height=height,
            top_quantile=float(args.top_quantile),
        )
        coverage_rows.append(coverage)
        image_records.append(
            {
                "image_id": image_id,
                "official_split": official_split,
                "calibration_split": calib_split,
                "subject_mode": subject_mode,
                "mode_bucket": mode_bucket(subject_mode),
                "num_gt": len(anns),
                "file_name": str(image_meta["file_name"]),
                "coverage": coverage,
                "subject_mode_conf": round(safe_float(ge_route.get("subject_mode_conf", 0.0), 0.0), 6),
                "raw_anchor_area": round(box_area(subject_visuals["raw_anchor_norm_xyxy"]), 6) if subject_visuals.get("raw_anchor_norm_xyxy") else 0.0,
                "c2_primary_area_ratio": round(safe_float(ge_route.get("subject_set", {}).get("c2_primary_area_ratio", 0.0), 0.0), 6),
                "subject_reliability": round(safe_float(subject_visuals.get("subject_reliability", 0.0), 0.0), 6),
                "subject_repr_type": str(subject_visuals.get("subject_repr_type", "")),
                "score_mode": str(subject_visuals.get("score_mode", "")),
                "subject_terms_neutralized": bool(subject_visuals.get("score_mode", "") == "neutralized"),
                "subject_agreement_iou": round(safe_float(subject_visuals.get("subject_agreement_iou", 0.0), 0.0), 6),
                "subject_center_distance": round(safe_float(subject_visuals.get("subject_center_distance", 0.0), 0.0), 6),
                "subject_disagreement_severe": bool(subject_visuals.get("subject_disagreement_severe", False)),
                "raw_anchor_support_mass": round(safe_float(subject_visuals.get("raw_anchor_support_mass", 0.0), 0.0), 6),
                "candidate_anchor_support_mass": round(safe_float(subject_visuals.get("candidate_anchor_support_mass", 0.0), 0.0), 6),
                "effective_support_mass": round(safe_float(subject_visuals.get("effective_support_mass", 0.0), 0.0), 6),
                "support_fallback_reason": str(subject_visuals.get("support_fallback_reason", "")),
                "blank_ratio_est": round(safe_float(subject_visuals.get("saliency_blank_ratio_est", ge_route.get("router_signals", {}).get("blank_ratio_est", 0.0)), 0.0), 6),
                "ge_winner_source_family": prod_source_family,
                "ge_policy_winner_source_family": policy_source_family,
                "ge_policy_rank_disagree": bool(policy_rank_disagree),
                "subject_support_metrics": subject_support_metrics,
                "subject_visuals": subject_visuals,
                "gt_best_row": gt_best_row,
                "Gc": protocol_payloads["Gc"],
                "Ge": {
                    **protocol_payloads["Ge"],
                    "production_best": prod_best_payload,
                    "production_best_policy": policy_best_payload,
                },
            }
        )

    candidate_eval_path = output_dir / "candidate_eval_rows.jsonl"
    write_jsonl(candidate_eval_path, candidate_eval_rows)
    ge_full_topk_path = output_dir / "ge_full_candidates_topk.jsonl"
    write_jsonl(ge_full_topk_path, ge_full_topk_rows)

    per_image_csv_path = output_dir / "per_image_metrics.csv"
    with per_image_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_csv_rows[0].keys()) if per_image_csv_rows else ["image_id"])
        writer.writeheader()
        if per_image_csv_rows:
            writer.writerows(per_image_csv_rows)

    summary: Dict[str, Any] = {
        "run": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
            "duration_sec": round(time.time() - start_ts, 3),
        },
        "config": {
            "top_quantile": float(args.top_quantile),
            "alpha_sweep": [float(value) for value in args.alpha_sweep],
            "softmax_tau": float(args.softmax_tau),
            "sample_count": int(args.sample_count),
            "scorer_config": asdict(cfg),
        },
        "sources": {
            "candidates_jsonl": str(candidates_jsonl),
            "features_jsonl": str(features_jsonl),
            "teacher_jsonl": str(teacher_jsonl),
            "training_label_dir": str(training_label_dir),
            "qa_summary_json": str(qa_summary_path),
            "validation_summary_json": str(validation_summary_path),
            "gaic_train_json": str(gaic_train_json),
            "gaic_test_json": str(gaic_test_json),
            "image_dir": str(image_dir),
        },
        "dataset": {
            "run_image_count": len(candidate_map),
            "feature_image_count": len(feature_map),
            "gt_total_image_count": len(gt_images),
            "overlap_image_count": len(overlap_ids),
            "overlap_train_count": sum(1 for row in image_records if row["official_split"] == "train"),
            "overlap_test_count": sum(1 for row in image_records if row["official_split"] == "test"),
            "raw_free_candidate_count_mean": mean(raw_free_counts),
            "training_label_free_candidate_count_mean": mean(training_free_counts),
            "benchmark_scope_note": "run_image_count는 local GAIC/All 실행 이미지 수이고, gt_total_image_count는 data/Publics/GAIC/annotations_json 의 merged train/test catalog 수입니다. 따라서 본 리포트는 official leaderboard 재현이 아니라 local overlap benchmark입니다.",
        },
        "coverage": {
            "top1_oracle_iou_mean": mean([row["top1_oracle_iou"] for row in coverage_rows]),
            "top1_coverage_at_05_mean": mean([row["top1_coverage_at_05"] for row in coverage_rows]),
            "top1_coverage_at_07_mean": mean([row["top1_coverage_at_07"] for row in coverage_rows]),
            "topq_any_coverage_at_05_mean": mean([row["topq_any_coverage_at_05"] for row in coverage_rows]),
        },
    }

    for protocol in ("Gc", "Ge"):
        protocol_image_rows = [record[protocol]["rows"] for record in image_records]
        summary[protocol] = {
            "trend_by_field": {},
            "mode_breakdown": {},
            "split_breakdown": {},
        }
        for field in PRIMARY_SCORE_FIELDS:
            per_image_metrics = [record[protocol]["metrics_by_field"][field] for record in image_records]
            summary[protocol]["trend_by_field"][field] = aggregate_trend_metrics(
                per_image_metrics,
                seed=int(args.seed) + (100 if protocol == "Ge" else 0) + PRIMARY_SCORE_FIELDS.index(field),
            )

        by_mode: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for record in image_records:
            by_mode[record["mode_bucket"]].append(record[protocol]["metrics_by_field"][PRIMARY_PROTOCOL_FIELD])
        for mode_name, rows in by_mode.items():
            summary[protocol]["mode_breakdown"][mode_name] = {
                "image_count": len(rows),
                "mean_spearman": mean([row["spearman"] for row in rows]),
                "mean_weighted_pair_acc": mean([row["weighted_pair_acc"] for row in rows]),
                "mean_hit_at_1": mean([row["hit_at_1"] for row in rows]),
            }

        for split_name in ("train", "test"):
            split_rows = [record[protocol]["metrics_by_field"][PRIMARY_PROTOCOL_FIELD] for record in image_records if record["official_split"] == split_name]
            summary[protocol]["split_breakdown"][split_name] = {
                "image_count": len(split_rows),
                "mean_spearman": mean([row["spearman"] for row in split_rows]),
                "mean_kendall_tau_b": mean([row["kendall_tau_b"] for row in split_rows]),
                "mean_weighted_pair_acc": mean([row["weighted_pair_acc"] for row in split_rows]),
                "mean_hit_at_1": mean([row["hit_at_1"] for row in split_rows]),
                "mean_topq_jaccard": mean([row["topq_jaccard"] for row in split_rows]),
            }

        val_image_rows = [rows for rows in protocol_image_rows if str(rows[0]["calibration_split"]) == "val"]
        test_image_rows = [rows for rows in protocol_image_rows if str(rows[0]["calibration_split"]) == "test"]
        best_alpha, alpha_summaries = tune_alpha(
            val_image_rows,
            alpha_candidates=[float(value) for value in args.alpha_sweep],
            top_quantile=float(args.top_quantile),
            z_field="score_z_local",
        )
        train_plus_val_rows = [
            row
            for rows in protocol_image_rows
            if str(rows[0]["calibration_split"]) in {"train", "val"}
            for row in rows
        ]
        calibrator_fit_rows = train_plus_val_rows or [row for rows in protocol_image_rows for row in rows]
        calibrator = fit_isotonic_calibrator(calibrator_fit_rows, alpha=best_alpha, z_field="score_z_local")
        summary[protocol]["local_probability"] = {
            "best_alpha": best_alpha,
            "alpha_sweep": alpha_summaries,
            "train": compute_local_probability_summary(
                [rows for rows in protocol_image_rows if str(rows[0]["calibration_split"]) == "train"],
                alpha=best_alpha,
                top_quantile=float(args.top_quantile),
                z_field="score_z_local",
            ),
            "val": compute_local_probability_summary(
                val_image_rows,
                alpha=best_alpha,
                top_quantile=float(args.top_quantile),
                z_field="score_z_local",
            ),
            "test": compute_local_probability_summary(
                test_image_rows,
                alpha=best_alpha,
                top_quantile=float(args.top_quantile),
                z_field="score_z_local",
            ),
        }
        summary[protocol]["global_calibration"] = {
            "best_alpha": best_alpha,
            "train_plus_val_fit_count": len(train_plus_val_rows),
            "test": evaluate_isotonic(
                calibrator,
                [row for rows in test_image_rows for row in rows],
                alpha=best_alpha,
                z_field="score_z_local",
            ),
        }
        summary[protocol]["local_probability_variants"] = {}
        summary[protocol]["global_calibration_variants"] = {}
        for variant_key, z_field, label in PROBABILITY_VARIANTS:
            variant_best_alpha, variant_alpha_summaries = tune_alpha(
                val_image_rows,
                alpha_candidates=[float(value) for value in args.alpha_sweep],
                top_quantile=float(args.top_quantile),
                z_field=z_field,
            )
            variant_calibrator = fit_isotonic_calibrator(calibrator_fit_rows, alpha=variant_best_alpha, z_field=z_field)
            summary[protocol]["local_probability_variants"][variant_key] = {
                "label": label,
                "z_field": z_field,
                "best_alpha": variant_best_alpha,
                "alpha_sweep": variant_alpha_summaries,
                "train": compute_local_probability_summary(
                    [rows for rows in protocol_image_rows if str(rows[0]["calibration_split"]) == "train"],
                    alpha=variant_best_alpha,
                    top_quantile=float(args.top_quantile),
                    z_field=z_field,
                ),
                "val": compute_local_probability_summary(
                    val_image_rows,
                    alpha=variant_best_alpha,
                    top_quantile=float(args.top_quantile),
                    z_field=z_field,
                ),
                "test": compute_local_probability_summary(
                    test_image_rows,
                    alpha=variant_best_alpha,
                    top_quantile=float(args.top_quantile),
                    z_field=z_field,
                ),
            }
            summary[protocol]["global_calibration_variants"][variant_key] = {
                "label": label,
                "z_field": z_field,
                "best_alpha": variant_best_alpha,
                "train_plus_val_fit_count": len(train_plus_val_rows),
                "test": evaluate_isotonic(
                    variant_calibrator,
                    [row for rows in test_image_rows for row in rows],
                    alpha=variant_best_alpha,
                    z_field=z_field,
                ),
            }

    summary["Ge"]["prod_selection"] = summarize_prod_selection([record["Ge"]["production_best"] for record in image_records])
    summary["Ge"]["prod_selection_policy"] = summarize_prod_selection(
        [record["Ge"]["production_best_policy"] for record in image_records if isinstance(record.get("Ge", {}).get("production_best_policy"), dict)]
    )
    summary["Ge"]["prod_selection_rank_vs_policy"] = summarize_prod_selection_pairwise(
        [record["Ge"]["production_best"] for record in image_records],
        [record["Ge"]["production_best_policy"] for record in image_records if isinstance(record.get("Ge", {}).get("production_best_policy"), dict)],
    )
    summary["Ge"]["score_mode_distribution"] = summarize_score_modes(
        image_records=image_records,
        protocol="Ge",
        alpha=float(summary["Ge"]["local_probability"]["best_alpha"]),
        top_quantile=float(args.top_quantile),
    )
    summary["Ge"]["failure_cohorts"] = summarize_failure_cohorts(
        image_records=image_records,
        protocol="Ge",
        alpha=float(summary["Ge"]["local_probability"]["best_alpha"]),
        top_quantile=float(args.top_quantile),
    )
    summary["subject_support_diagnostics"] = summarize_subject_support_diagnostics(image_records)

    plot_dir = output_dir / "plots"
    ensure_dir(plot_dir)
    gc_spearman_values = [record["Gc"]["metrics_by_field"]["score_rank"]["spearman"] for record in image_records]
    ge_spearman_values = [record["Ge"]["metrics_by_field"]["score_rank"]["spearman"] for record in image_records]
    plot_paths = {
        "protocol_compare": plot_dir / "protocol_compare_primary.png",
        "gc_spearman_hist": plot_dir / "gc_spearman_hist.png",
        "ge_spearman_hist": plot_dir / "ge_spearman_hist.png",
        "gc_scatter": plot_dir / "gc_score_rank_vs_mos.png",
        "ge_scatter": plot_dir / "ge_score_rank_vs_mos.png",
        "ge_reliability": plot_dir / "ge_reliability_topq.png",
        "ge_policy_reliability": plot_dir / "ge_policy_reliability_topq.png",
    }
    plot_protocol_comparison(summary, plot_paths["protocol_compare"])
    plot_histogram(gc_spearman_values, "Gc per-image Spearman", "Spearman", plot_paths["gc_spearman_hist"], color="#4c78a8")
    plot_histogram(ge_spearman_values, "Ge per-image Spearman", "Spearman", plot_paths["ge_spearman_hist"], color="#f58518")
    plot_scatter(
        [row["score_rank"] for row in candidate_eval_rows if row["protocol"] == "Gc"],
        [row["mos"] for row in candidate_eval_rows if row["protocol"] == "Gc"],
        "Gc score_rank vs MOS",
        "score_rank",
        "MOS",
        plot_paths["gc_scatter"],
    )
    plot_scatter(
        [row["score_rank"] for row in candidate_eval_rows if row["protocol"] == "Ge"],
        [row["mos"] for row in candidate_eval_rows if row["protocol"] == "Ge"],
        "Ge score_rank vs MOS",
        "score_rank",
        "MOS",
        plot_paths["ge_scatter"],
    )
    plot_reliability(
        summary["Ge"]["local_probability"]["test"]["topq_curve"],
        "Ge top-q reliability",
        plot_paths["ge_reliability"],
    )
    plot_reliability(
        summary["Ge"]["local_probability_variants"]["policy_robust"]["test"]["topq_curve"],
        "Ge policy top-q reliability",
        plot_paths["ge_policy_reliability"],
    )

    sample_ids = select_sample_image_ids(image_records, int(args.sample_count))
    sample_files: List[Path] = []
    sample_analyses: List[Dict[str, Any]] = []
    saliency_mask_resolver = SampleSaliencyMaskResolver()
    try:
        for image_id in sample_ids:
            record = next(row for row in image_records if str(row["image_id"]) == str(image_id))
            img_path = image_path_for(image_dir, record["file_name"], image_id)
            if img_path is None:
                continue
            saliency_mask = saliency_mask_resolver.resolve(img_path, record.get("subject_visuals", {}))
            out_path = output_dir / "samples" / f"{image_id}_overlay.png"
            render_sample_overlay(record, img_path, out_path, saliency_mask=saliency_mask)
            sample_files.append(out_path)
            sample_analyses.append(
                build_sample_analysis(
                    image_id=image_id,
                    cand_rec=candidate_map[image_id],
                    feat_rec=feature_map[image_id],
                    image_meta=gt_images[image_id],
                    anns=gt_anns[image_id],
                    cfg=cfg,
                    image_path=img_path,
                    output_dir=output_dir,
                    softmax_tau=float(args.softmax_tau),
                    saliency_mask_resolver=saliency_mask_resolver,
                )
            )
    finally:
        saliency_mask_resolver.close()

    summary_path = output_dir / "benchmark_summary.json"
    write_json(summary_path, summary)

    report_text = build_markdown_report(
        output_dir=output_dir,
        summary=summary,
        image_records=image_records,
        qa_summary=qa_summary,
        validation_summary=validation_summary,
        sample_files=sample_files,
        sample_analyses=sample_analyses,
        plot_paths=plot_paths,
    )
    report_path = output_dir / "GAIC_BENCHMARK_EVAL_REPORT_KO.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "summary_json": str(summary_path),
                "report_md": str(report_path),
                "candidate_eval_jsonl": str(candidate_eval_path),
                "ge_full_topk_jsonl": str(ge_full_topk_path),
                "per_image_csv": str(per_image_csv_path),
                "sample_count": len(sample_files),
                "overlap_image_count": len(overlap_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
