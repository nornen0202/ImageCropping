#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
except Exception:  # pragma: no cover - optional dependency on some local/dev hosts
    ExtraTreesRegressor = None  # type: ignore[assignment]
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from mobilecropnet_v4.gaic_benchmark import per_image_gaic_metrics, summarize_metric_rows


DEFAULT_SCORE_FIELDS = (
    "crop_utility_raw",
    "score_rank",
    "score_policy",
    "score_rank_pct",
    "score_z_local",
    "score_z_local_std",
    "score_sigmoid_z_local",
    "score_softmax_local",
    "pseudo_mos_1to5",
    "score_policy_rank_pct",
    "score_policy_z_local",
    "score_policy_z_local_std",
    "score_policy_sigmoid_z_local",
    "pseudo_prob_policy_local",
    "crop_utility_prob",
    "policy_pseudo_mos_1to5",
    "raw_score_rank",
    "raw_score_policy",
)

DEFAULT_AUDIT_FIELDS = (
    "crop_utility_raw",
    "crop_utility_prob",
    "score_policy",
    "score_policy_sigmoid_z_local",
    "score_policy_z_local",
    "score_rank",
    "score_rank_pct",
)

DEFAULT_EXTRA_FEATURE_FIELDS = (
    "profile_A_macro",
    "profile_S_macro",
    "profile_C_macro",
    "profile_T_macro",
    "profile_A_active",
    "profile_S_active",
    "profile_C_active",
    "profile_T_active",
    "profile_area_log_prior",
    "profile_safety_penalty_total",
    "profile_safety_penalty_soft",
    "profile_safety_penalty_hard",
    "profile_component_aesthetic_norm",
    "profile_component_cosine_img_text",
    "profile_component_cov",
    "profile_component_p_cut",
    "profile_component_p_text",
    "profile_component_p_ar_free",
    "profile_component_r_comp",
    "profile_component_r_headroom",
    "profile_component_r_lookroom",
    "profile_component_r_horizon",
    "profile_component_r_sym",
    "profile_component_r_context",
    "profile_component_r_teach",
)

DEFAULT_MODEL_SPECS = (
    "ridge_mos_image_equal",
    "ridge_rankpct_image_equal",
    "ridge_zscore_image_equal",
    "hgb_mos",
    "hgb_rankpct",
    "hgb_zscore",
    "extra_trees_mos_light",
    "extra_trees_rankpct_light",
    "extra_trees_zscore_light",
)


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_candidate_eval_rows(path: Optional[Path], *, protocol: str) -> List[Dict[str, Any]]:
    if path is None:
        return []
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("protocol", "")) != str(protocol):
                continue
            rows.append(row)
    return rows


def group_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("image_id", ""))].append(dict(row))
    for image_rows in grouped.values():
        image_rows.sort(key=lambda row: (safe_int(row.get("gt_annotation_id"), 0), str(row.get("candidate_id", ""))))
    return dict(sorted(grouped.items()))


def bbox_features(row: Dict[str, Any]) -> Dict[str, float]:
    box = row.get("bbox_norm_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
    else:
        x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box[:4]]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    area = w * h
    ar = w / h
    third_x = min(abs(cx - 1.0 / 3.0), abs(cx - 2.0 / 3.0))
    third_y = min(abs(cy - 1.0 / 3.0), abs(cy - 2.0 / 3.0))
    return {
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": area,
        "bbox_ar": ar,
        "bbox_log_area": math.log(max(area, 1e-6)),
        "bbox_log_ar": math.log(max(ar, 1e-6)),
        "bbox_cx": cx,
        "bbox_cy": cy,
        "bbox_margin_l": x1,
        "bbox_margin_t": y1,
        "bbox_margin_r": 1.0 - x2,
        "bbox_margin_b": 1.0 - y2,
        "bbox_center_dist": math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2),
        "bbox_third_dist": min(third_x, third_y),
    }


def derived_features(row: Dict[str, Any]) -> Dict[str, float]:
    box = bbox_features(row)
    log_area = safe_float(box.get("bbox_log_area"), math.log(0.65))
    log_ar = safe_float(box.get("bbox_log_ar"), 0.0)
    area = safe_float(box.get("bbox_area"), 1.0)
    center_dist = safe_float(box.get("bbox_center_dist"), 0.0)
    third_dist = safe_float(box.get("bbox_third_dist"), 0.0)
    a_macro = safe_float(row.get("profile_A_macro"), 0.0)
    s_macro = safe_float(row.get("profile_S_macro"), 0.0)
    c_macro = safe_float(row.get("profile_C_macro"), 0.0)
    safety = safe_float(row.get("profile_safety_penalty_total"), 0.0)
    p_cut = safe_float(row.get("profile_component_p_cut"), 0.0)
    p_text = safe_float(row.get("profile_component_p_text"), 0.0)
    p_ar = safe_float(row.get("profile_component_p_ar_free"), 0.0)
    return {
        "derived_neg_safety_total": -safety,
        "derived_neg_p_cut": -p_cut,
        "derived_neg_p_text": -p_text,
        "derived_neg_p_ar_free": -p_ar,
        "derived_abs_bbox_log_ar": abs(log_ar),
        "derived_bbox_log_area_sq": log_area * log_area,
        "derived_bbox_area_sq": area * area,
        "derived_mid_area_060": math.exp(-0.5 * ((log_area - math.log(0.60)) / 0.42) ** 2),
        "derived_mid_area_075": math.exp(-0.5 * ((log_area - math.log(0.75)) / 0.42) ** 2),
        "derived_center_score": max(0.0, 1.0 - center_dist / 0.70710678),
        "derived_third_score": max(0.0, 1.0 - third_dist / 0.33333333),
        "derived_macro_asc_mean": (a_macro + s_macro + c_macro) / 3.0,
        "derived_macro_ac_mean": (a_macro + c_macro) / 2.0,
        "derived_macro_sc_mean": (s_macro + c_macro) / 2.0,
        "derived_penalty_sum": safety + p_cut + p_text + p_ar,
    }


def collect_categories(rows: Sequence[Dict[str, Any]], key: str, max_categories: int = 32) -> List[str]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(key, "") or "").strip()
        if value:
            counts[value] += 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_categories]]


def build_feature_spec(train_rows: Sequence[Dict[str, Any]], score_fields: Sequence[str]) -> Dict[str, Any]:
    bbox_names = list(bbox_features({}).keys())
    derived_names = list(derived_features({}).keys())
    subject_modes = collect_categories(train_rows, "subject_mode")
    mode_buckets = collect_categories(train_rows, "mode_bucket")
    feature_names = list(score_fields) + bbox_names
    feature_names += derived_names
    feature_names += [f"subject_mode={value}" for value in subject_modes]
    feature_names += [f"mode_bucket={value}" for value in mode_buckets]
    return {
        "score_fields": list(score_fields),
        "bbox_feature_names": bbox_names,
        "derived_feature_names": derived_names,
        "subject_modes": subject_modes,
        "mode_buckets": mode_buckets,
        "feature_names": feature_names,
    }


def row_to_features(row: Dict[str, Any], spec: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for field in spec["score_fields"]:
        values.append(safe_float(row.get(field), 0.0))
    box_values = bbox_features(row)
    for field in spec["bbox_feature_names"]:
        values.append(safe_float(box_values.get(field), 0.0))
    derived_values = derived_features(row)
    for field in spec["derived_feature_names"]:
        values.append(safe_float(derived_values.get(field), 0.0))
    subject_mode = str(row.get("subject_mode", "") or "").strip()
    mode_bucket = str(row.get("mode_bucket", "") or "").strip()
    values.extend(1.0 if subject_mode == value else 0.0 for value in spec["subject_modes"])
    values.extend(1.0 if mode_bucket == value else 0.0 for value in spec["mode_buckets"])
    return values


def rows_to_matrix(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row_to_features(row, spec) for row in rows], dtype=np.float64)
    y = np.asarray([safe_float(row.get("mos"), 0.0) for row in rows], dtype=np.float64)
    return x, y


def _rank_pct(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n <= 1:
        return [0.5 for _ in values]
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: (item[1], item[0]))
    ranks = [0.0 for _ in values]
    denom = float(n - 1)
    start = 0
    while start < n:
        end = start + 1
        while end < n and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (float(start) + float(end - 1)) * 0.5 / denom
        for pos in range(start, end):
            ranks[indexed[pos][0]] = avg_rank
        start = end
    return ranks


def target_values(rows: Sequence[Dict[str, Any]], *, target: str) -> np.ndarray:
    target = str(target)
    mos = np.asarray([safe_float(row.get("mos"), 0.0) for row in rows], dtype=np.float64)
    if target == "mos":
        return mos
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row.get("image_id", ""))].append(idx)
    out = np.zeros_like(mos)
    for indices in grouped.values():
        vals = [float(mos[idx]) for idx in indices]
        if target == "rankpct":
            ranks = _rank_pct(vals)
            for idx, rank in zip(indices, ranks):
                out[idx] = rank
        elif target == "zscore":
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            if std < 1e-8:
                std = 1.0
            for idx, value in zip(indices, vals):
                out[idx] = (value - mean) / std
        else:
            raise ValueError(f"unsupported target: {target}")
    return out


def sample_weights(rows: Sequence[Dict[str, Any]], *, mode: str) -> Optional[np.ndarray]:
    if mode == "row" or not rows:
        return None
    if mode != "image_equal":
        raise ValueError(f"unsupported fit_weighting: {mode}")
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("image_id", ""))] += 1
    weights = np.asarray([1.0 / max(1, counts[str(row.get("image_id", ""))]) for row in rows], dtype=np.float64)
    weights *= float(len(weights)) / max(float(weights.sum()), 1e-12)
    return weights


def weighted_mean_std(x: np.ndarray, weights: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if weights is None:
        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        if float(w.sum()) <= 0.0:
            mu = x.mean(axis=0)
            sigma = x.std(axis=0)
        else:
            w_norm = w / w.sum()
            mu = np.sum(x * w_norm[:, None], axis=0)
            var = np.sum(((x - mu) ** 2) * w_norm[:, None], axis=0)
            sigma = np.sqrt(np.maximum(var, 0.0))
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    mu, sigma = weighted_mean_std(x, weights)
    x_norm = (x - mu) / sigma
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm], axis=1)
    y_fit = y
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        if float(w.sum()) > 0.0:
            w = w / max(float(w.mean()), 1e-12)
            scale = np.sqrt(w)[:, None]
            x_aug = x_aug * scale
            y_fit = y * scale[:, 0]
    if float(alpha) <= 0.0:
        beta = np.linalg.lstsq(x_aug, y_fit, rcond=None)[0]
    else:
        reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_fit)
    return {
        "alpha": float(alpha),
        "feature_mean": mu.tolist(),
        "feature_std": sigma.tolist(),
        "coef_intercept_first": beta.tolist(),
    }


def predict_rows(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any], model: Dict[str, Any]) -> List[float]:
    if not rows:
        return []
    x, _ = rows_to_matrix(rows, spec)
    return predict_ridge_matrix(x, model)


def predict_ridge_matrix(x: np.ndarray, model: Dict[str, Any]) -> List[float]:
    if x.size == 0:
        return []
    mu = np.asarray(model["feature_mean"], dtype=np.float64)
    sigma = np.asarray(model["feature_std"], dtype=np.float64)
    beta = np.asarray(model["coef_intercept_first"], dtype=np.float64)
    x_norm = (x - mu) / sigma
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm], axis=1)
    return [float(v) for v in (x_aug @ beta)]


def parse_model_specs(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw or raw == "default":
        return list(DEFAULT_MODEL_SPECS)
    return dedupe_keep_order(raw.split(","))


def model_spec_info(name: str, default_weighting: str) -> Dict[str, Any]:
    name = str(name).strip()
    if name.startswith("ridge_"):
        target = "mos"
        if "rankpct" in name:
            target = "rankpct"
        elif "zscore" in name:
            target = "zscore"
        weighting = "row" if name.endswith("_row") else default_weighting
        if name.endswith("_image_equal"):
            weighting = "image_equal"
        return {"name": name, "family": "ridge", "target": target, "fit_weighting": weighting}
    if name.startswith("hgb_"):
        target = "mos"
        if "rankpct" in name:
            target = "rankpct"
        elif "zscore" in name:
            target = "zscore"
        family = "hist_gradient_boosting_abs" if name.startswith("hgb_abs_") else "hist_gradient_boosting"
        return {"name": name, "family": family, "target": target, "fit_weighting": default_weighting}
    if name.startswith("extra_trees_"):
        target = "mos"
        if "rankpct" in name:
            target = "rankpct"
        elif "zscore" in name:
            target = "zscore"
        return {"name": name, "family": "extra_trees", "target": target, "fit_weighting": default_weighting}
    if name.startswith("rf_"):
        target = "mos"
        if "rankpct" in name:
            target = "rankpct"
        elif "zscore" in name:
            target = "zscore"
        return {"name": name, "family": "random_forest", "target": target, "fit_weighting": default_weighting}
    raise ValueError(f"unsupported model spec: {name}")


def subset_indices(count: int, max_count: int, *, seed: int) -> np.ndarray:
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.arange(count, dtype=np.int64), size=int(max_count), replace=False)
    return np.sort(chosen)


def predict_sklearn_rows(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any], model: Any) -> List[float]:
    if not rows:
        return []
    x, _ = rows_to_matrix(rows, spec)
    return [float(v) for v in model.predict(x)]


def result_with_meta(result: Dict[str, Any], *, family: str, target: str, fit_weighting: str, train_rows_used: int) -> Dict[str, Any]:
    result["family"] = family
    result["target"] = target
    result["fit_weighting"] = fit_weighting
    result["train_rows_used"] = int(train_rows_used)
    result["objective"] = round(objective(result.get("metrics", {})), 9)
    return result


def strip_runtime(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in candidate.items() if not k.startswith("_")}


def compact_method_name(name: str, limit: int = 34) -> str:
    cleaned = str(name).replace("ridge_", "r_").replace("extra_trees_", "et_").replace("hist_gradient_boosting", "hgb")
    cleaned = cleaned.replace("_image_equal", "_ie").replace("_light", "")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit]


def blend_scores(a: Sequence[float], b: Sequence[float], weight_a: float) -> List[float]:
    wa = float(weight_a)
    wb = 1.0 - wa
    return [wa * float(x) + wb * float(y) for x, y in zip(a, b)]


def evaluate_rows(rows: Sequence[Dict[str, Any]], scores: Sequence[float], *, method: str) -> Dict[str, Any]:
    paired_grouped: Dict[str, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        paired_grouped[str(row.get("image_id", ""))].append((dict(row), float(score)))
    metric_rows: List[Dict[str, float]] = []
    for image_id in sorted(paired_grouped):
        pairs = sorted(
            paired_grouped[image_id],
            key=lambda item: (safe_int(item[0].get("gt_annotation_id"), 0), str(item[0].get("candidate_id", ""))),
        )
        image_rows = [row for row, _ in pairs]
        image_scores = [score for _, score in pairs]
        mos = [safe_float(row.get("mos"), 0.0) for row in image_rows]
        metrics = per_image_gaic_metrics(mos, image_scores)
        if metrics:
            metric_rows.append(metrics)
    return {
        "method": method,
        "image_count": len(metric_rows),
        "candidate_count": len(rows),
        "metrics": summarize_metric_rows(metric_rows),
    }


def evaluate_field(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    result = evaluate_rows(rows, [safe_float(row.get(field), 0.0) for row in rows], method=field)
    result["family"] = "score_field"
    result["target"] = "none"
    result["fit_weighting"] = "none"
    result["train_rows_used"] = 0
    result["objective"] = round(objective(result.get("metrics", {})), 9)
    return result


def objective(metrics: Dict[str, float]) -> float:
    return (
        safe_float(metrics.get("srcc"), 0.0)
        + safe_float(metrics.get("pcc"), 0.0)
        + safe_float(metrics.get("acc1_of_top10"), 0.0)
        + safe_float(metrics.get("acc4_of_top10"), 0.0)
        + 0.10 * safe_float(metrics.get("top1_mos"), 0.0)
        - 0.10 * safe_float(metrics.get("top1_mos_regret"), 0.0)
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def metric_row_for_report(split_name: str, result: Dict[str, Any]) -> List[str]:
    metrics = result.get("metrics", {})
    return [
        split_name,
        str(result.get("method", "")),
        str(result.get("image_count", 0)),
        f"{safe_float(metrics.get('pcc'), 0.0):.6f}",
        f"{safe_float(metrics.get('srcc'), 0.0):.6f}",
        f"{safe_float(metrics.get('acc1_of_top5'), 0.0):.6f}",
        f"{safe_float(metrics.get('acc1_of_top10'), 0.0):.6f}",
        f"{safe_float(metrics.get('acc4_of_top10'), 0.0):.6f}",
        f"{safe_float(metrics.get('top1_mos'), 0.0):.6f}",
        f"{safe_float(metrics.get('top1_mos_regret'), 0.0):.6f}",
    ]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(summary: Dict[str, Any]) -> str:
    rows: List[List[str]] = []
    for split_name in ("train", "val", "test"):
        for result in summary.get("evaluations", {}).get(split_name, []):
            rows.append(metric_row_for_report(split_name, result))
    comparison_rows: List[List[str]] = []
    for candidate in summary.get("model_candidates", []):
        val_result = candidate.get("evaluations", {}).get("val", {})
        test_result = candidate.get("evaluations", {}).get("test", {})
        val_metrics = val_result.get("metrics", {})
        test_metrics = test_result.get("metrics", {})
        comparison_rows.append(
            [
                str(candidate.get("method", "")),
                str(candidate.get("family", "")),
                str(candidate.get("target", "")),
                str(candidate.get("fit_weighting", "")),
                str(candidate.get("train_rows_used", "")),
                f"{safe_float(val_result.get('objective'), objective(val_metrics)):.6f}",
                f"{safe_float(val_metrics.get('pcc'), 0.0):.6f}",
                f"{safe_float(val_metrics.get('srcc'), 0.0):.6f}",
                f"{safe_float(val_metrics.get('acc1_of_top10'), 0.0):.6f}",
                f"{safe_float(test_metrics.get('pcc'), 0.0):.6f}",
                f"{safe_float(test_metrics.get('srcc'), 0.0):.6f}",
                f"{safe_float(test_metrics.get('acc1_of_top10'), 0.0):.6f}",
                f"{safe_float(test_metrics.get('top1_mos'), 0.0):.6f}",
                f"{safe_float(test_metrics.get('top1_mos_regret'), 0.0):.6f}",
            ]
        )
    lines = [
        "# GAIC Teacher MOS Calibration Report",
        "",
        f"- protocol: `{summary.get('protocol')}`",
        f"- selected_method_by_val: `{summary.get('selected_method_by_val')}`",
        f"- selected_alpha: `{summary.get('selected_alpha')}`",
        f"- selected_by: `{summary.get('selected_by')}`",
        "",
        "## Model Comparison",
        "",
        markdown_table(
            [
                "method",
                "family",
                "target",
                "weighting",
                "train rows",
                "val objective",
                "val PCC",
                "val SRCC",
                "val Acc1/10",
                "test PCC",
                "test SRCC",
                "test Acc1/10",
                "test top1 MOS",
                "test regret",
            ],
            comparison_rows,
        ),
        "",
        "## Split Metrics",
        "",
        markdown_table(
            ["split", "method", "images", "PCC", "SRCC", "Acc1/5", "Acc1/10", "Acc4/10", "top1 MOS", "MOS regret"],
            rows,
        ),
        "",
        "## Notes",
        "",
        "- This script fits only on the rows passed as train/val. Do not tune on GAIC v2 test rows.",
        "- The calibrated score is a benchmark-calibration layer over existing teacher candidate-eval features, not a replacement for SSTK production policy unless it also passes non-GAIC public benchmarks.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def parse_alpha_values(text: str) -> List[float]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values or [1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit/evaluate a GAIC MOS calibration layer for SSTK teacher candidate-eval rows.")
    parser.add_argument("--train_candidate_eval_jsonl", type=Path, default=None)
    parser.add_argument("--val_candidate_eval_jsonl", type=Path, default=None)
    parser.add_argument("--test_candidate_eval_jsonl", type=Path, default=None)
    parser.add_argument("--protocol", default="Gc")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--score_fields", default=",".join(DEFAULT_SCORE_FIELDS))
    parser.add_argument("--audit_fields", default=",".join(DEFAULT_AUDIT_FIELDS))
    parser.add_argument("--extra_feature_fields", default=",".join(DEFAULT_EXTRA_FEATURE_FIELDS))
    parser.add_argument("--alpha_values", default="0,1e-4,1e-3,1e-2,1e-1,1,10,100")
    parser.add_argument("--fit_weighting", default="image_equal", choices=["row", "image_equal"])
    parser.add_argument("--model_specs", default="default")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hgb_max_iter", type=int, default=260)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.045)
    parser.add_argument("--hgb_max_leaf_nodes", type=int, default=31)
    parser.add_argument("--hgb_l2_regularization", type=float, default=0.01)
    parser.add_argument("--extra_trees_n_estimators", type=int, default=220)
    parser.add_argument("--extra_trees_max_depth", type=int, default=20)
    parser.add_argument("--extra_trees_min_samples_leaf", type=int, default=2)
    parser.add_argument("--extra_trees_n_jobs", type=int, default=-1)
    parser.add_argument("--extra_trees_max_train_rows", type=int, default=180000)
    parser.add_argument("--rf_n_estimators", type=int, default=220)
    parser.add_argument("--rf_max_depth", type=int, default=22)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=2)
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    parser.add_argument("--rf_max_train_rows", type=int, default=160000)
    parser.add_argument("--enable_ensembles", type=int, default=1)
    parser.add_argument("--ensemble_source_top_k", type=int, default=8)
    parser.add_argument("--ensemble_keep_top_k", type=int, default=20)
    parser.add_argument("--ensemble_weight_steps", type=int, default=9)
    parser.add_argument("--write_predictions", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    score_fields = dedupe_keep_order(str(args.score_fields).split(","))
    audit_fields = dedupe_keep_order(str(args.audit_fields).split(","))
    extra_feature_fields = dedupe_keep_order(str(args.extra_feature_fields).split(","))
    feature_fields = dedupe_keep_order(list(score_fields) + list(extra_feature_fields))

    split_rows = {
        "train": load_candidate_eval_rows(args.train_candidate_eval_jsonl, protocol=args.protocol),
        "val": load_candidate_eval_rows(args.val_candidate_eval_jsonl, protocol=args.protocol),
        "test": load_candidate_eval_rows(args.test_candidate_eval_jsonl, protocol=args.protocol),
    }
    available_rows = [row for rows in split_rows.values() for row in rows]
    if not available_rows:
        raise RuntimeError("no candidate-eval rows loaded")

    train_rows = split_rows["train"] or split_rows["val"] or split_rows["test"]
    spec = build_feature_spec(train_rows, feature_fields)

    evaluations: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, rows in split_rows.items():
        if not rows:
            evaluations[split_name] = []
            continue
        field_results = [evaluate_field(rows, field) for field in audit_fields if any(field in row for row in rows)]
        evaluations[split_name] = field_results

    split_x: Dict[str, np.ndarray] = {}
    for split_name, rows in split_rows.items():
        if rows:
            split_x[split_name], _ = rows_to_matrix(rows, spec)

    selected_alpha: Optional[float] = None
    selected_by = "baseline_only"
    selected_method_by_val = ""
    alpha_results: List[Dict[str, Any]] = []
    runtime_candidates: List[Dict[str, Any]] = []
    model_specs = parse_model_specs(str(args.model_specs))

    if split_rows["train"]:
        x_train = split_x["train"]
        selection_rows = split_rows["val"] or split_rows["train"]
        selection_split = "val" if split_rows["val"] else "train"
        x_selection = split_x[selection_split]
        field_runtime_candidates: List[Dict[str, Any]] = []
        for field in audit_fields:
            scores_by_split: Dict[str, List[float]] = {}
            candidate_evals: Dict[str, Dict[str, Any]] = {}
            for split_name, rows in split_rows.items():
                if not rows or not any(field in row for row in rows):
                    continue
                scores = [safe_float(row.get(field), 0.0) for row in rows]
                scores_by_split[split_name] = scores
                candidate_evals[split_name] = result_with_meta(
                    evaluate_rows(rows, scores, method=field),
                    family="score_field",
                    target="none",
                    fit_weighting="none",
                    train_rows_used=0,
                )
            if selection_split in scores_by_split:
                field_runtime_candidates.append(
                    {
                        "method": field,
                        "base_model_spec": field,
                        "family": "score_field",
                        "target": "none",
                        "fit_weighting": "none",
                        "train_rows_used": 0,
                        "selected_alpha": None,
                        "selected_by": selection_split,
                        "selection_objective": float(objective(candidate_evals.get(selection_split, {}).get("metrics", {}))),
                        "evaluations": candidate_evals,
                        "_predict_kind": "precomputed",
                        "_scores_by_split": scores_by_split,
                    }
                )

        for model_name in model_specs:
            info = model_spec_info(model_name, str(args.fit_weighting))
            target = str(info["target"])
            family = str(info["family"])
            fit_weighting = str(info["fit_weighting"])
            y_target = target_values(split_rows["train"], target=target)
            train_weights_all = sample_weights(split_rows["train"], mode=fit_weighting)

            if family == "ridge":
                best_model: Optional[Dict[str, Any]] = None
                best_alpha: Optional[float] = None
                best_sel_result: Optional[Dict[str, Any]] = None
                best_value = -1e18
                for alpha in parse_alpha_values(args.alpha_values):
                    model = fit_ridge(x_train, y_target, alpha=alpha, weights=train_weights_all)
                    sel_scores = predict_ridge_matrix(x_selection, model)
                    method = f"{model_name}_alpha_{alpha:g}"
                    sel_result = result_with_meta(
                        evaluate_rows(selection_rows, sel_scores, method=method),
                        family=family,
                        target=target,
                        fit_weighting=fit_weighting,
                        train_rows_used=len(split_rows["train"]),
                    )
                    value = objective(sel_result.get("metrics", {}))
                    alpha_results.append(
                        {
                            "method": method,
                            "base_model_spec": model_name,
                            "alpha": float(alpha),
                            "target": target,
                            "fit_weighting": fit_weighting,
                            "selection_split": selection_split,
                            "selection_objective": float(value),
                            "selection_metrics": sel_result,
                        }
                    )
                    if value > best_value:
                        best_value = value
                        best_model = model
                        best_alpha = float(alpha)
                        best_sel_result = sel_result
                if best_model is None or best_sel_result is None:
                    continue
                method = f"{model_name}_alpha_{best_alpha:g}"
                candidate_evals: Dict[str, Dict[str, Any]] = {}
                scores_by_split: Dict[str, List[float]] = {}
                for split_name, rows in split_rows.items():
                    if not rows:
                        continue
                    scores = predict_ridge_matrix(split_x[split_name], best_model)
                    scores_by_split[split_name] = scores
                    result = result_with_meta(
                        evaluate_rows(rows, scores, method=method),
                        family=family,
                        target=target,
                        fit_weighting=fit_weighting,
                        train_rows_used=len(split_rows["train"]),
                    )
                    candidate_evals[split_name] = result
                    evaluations.setdefault(split_name, []).append(result)
                runtime_candidates.append(
                    {
                        "method": method,
                        "base_model_spec": model_name,
                        "family": family,
                        "target": target,
                        "fit_weighting": fit_weighting,
                        "train_rows_used": len(split_rows["train"]),
                        "selected_alpha": best_alpha,
                        "selected_by": selection_split,
                        "selection_objective": float(best_value),
                        "evaluations": candidate_evals,
                        "_predict_kind": "ridge",
                        "_model": best_model,
                        "_scores_by_split": scores_by_split,
                    }
                )
                continue

            if family in ("hist_gradient_boosting", "hist_gradient_boosting_abs"):
                if HistGradientBoostingRegressor is None:
                    raise RuntimeError("sklearn HistGradientBoostingRegressor is not available")
                train_indices = subset_indices(len(split_rows["train"]), 0, seed=int(args.random_seed))
                model = HistGradientBoostingRegressor(
                    loss="absolute_error" if family == "hist_gradient_boosting_abs" else "squared_error",
                    max_iter=int(args.hgb_max_iter),
                    learning_rate=float(args.hgb_learning_rate),
                    max_leaf_nodes=int(args.hgb_max_leaf_nodes),
                    l2_regularization=float(args.hgb_l2_regularization),
                    random_state=int(args.random_seed),
                )
            elif family == "extra_trees":
                if ExtraTreesRegressor is None:
                    raise RuntimeError("sklearn ExtraTreesRegressor is not available")
                train_indices = subset_indices(
                    len(split_rows["train"]),
                    int(args.extra_trees_max_train_rows) if "light" in model_name else 0,
                    seed=int(args.random_seed),
                )
                model = ExtraTreesRegressor(
                    n_estimators=int(args.extra_trees_n_estimators),
                    max_depth=int(args.extra_trees_max_depth),
                    min_samples_leaf=int(args.extra_trees_min_samples_leaf),
                    n_jobs=int(args.extra_trees_n_jobs),
                    random_state=int(args.random_seed),
                )
            elif family == "random_forest":
                if RandomForestRegressor is None:
                    raise RuntimeError("sklearn RandomForestRegressor is not available")
                train_indices = subset_indices(
                    len(split_rows["train"]),
                    int(args.rf_max_train_rows) if "light" in model_name else 0,
                    seed=int(args.random_seed),
                )
                model = RandomForestRegressor(
                    n_estimators=int(args.rf_n_estimators),
                    max_depth=int(args.rf_max_depth),
                    min_samples_leaf=int(args.rf_min_samples_leaf),
                    n_jobs=int(args.rf_n_jobs),
                    random_state=int(args.random_seed),
                )
            else:
                raise ValueError(f"unsupported model family: {family}")

            x_fit = x_train[train_indices]
            y_fit = y_target[train_indices]
            weights_fit = None if train_weights_all is None else train_weights_all[train_indices]
            model.fit(x_fit, y_fit, sample_weight=weights_fit)
            candidate_evals = {}
            scores_by_split = {}
            for split_name, rows in split_rows.items():
                if not rows:
                    continue
                scores = [float(v) for v in model.predict(split_x[split_name])]
                scores_by_split[split_name] = scores
                result = result_with_meta(
                    evaluate_rows(rows, scores, method=model_name),
                    family=family,
                    target=target,
                    fit_weighting=fit_weighting,
                    train_rows_used=len(train_indices),
                )
                candidate_evals[split_name] = result
                evaluations.setdefault(split_name, []).append(result)
            runtime_candidates.append(
                {
                    "method": model_name,
                    "base_model_spec": model_name,
                    "family": family,
                    "target": target,
                    "fit_weighting": fit_weighting,
                    "train_rows_used": len(train_indices),
                    "selected_alpha": None,
                    "selected_by": selection_split,
                    "selection_objective": float(objective(candidate_evals.get(selection_split, {}).get("metrics", {}))),
                    "evaluations": candidate_evals,
                    "_predict_kind": "sklearn",
                    "_model": model,
                    "_scores_by_split": scores_by_split,
                }
            )

        if int(args.enable_ensembles):
            source_pool = field_runtime_candidates + runtime_candidates
            source_pool = [
                item for item in source_pool if selection_split in item.get("_scores_by_split", {}) and item.get("evaluations", {}).get(selection_split)
            ]
            source_pool.sort(key=lambda item: objective(item["evaluations"][selection_split].get("metrics", {})), reverse=True)
            source_pool = source_pool[: max(2, int(args.ensemble_source_top_k))]
            pair_best: List[Dict[str, Any]] = []
            steps = max(1, int(args.ensemble_weight_steps))
            weights = [float(i) / float(steps + 1) for i in range(1, steps + 1)]
            for i in range(len(source_pool)):
                for j in range(i + 1, len(source_pool)):
                    left = source_pool[i]
                    right = source_pool[j]
                    left_scores = left["_scores_by_split"][selection_split]
                    right_scores = right["_scores_by_split"][selection_split]
                    best_pair: Optional[Dict[str, Any]] = None
                    best_pair_value = -1e18
                    for weight in weights:
                        scores = blend_scores(left_scores, right_scores, weight)
                        method = "ens_{left}_{right}_w{w:02d}".format(
                            left=compact_method_name(left["method"], 28),
                            right=compact_method_name(right["method"], 28),
                            w=int(round(weight * 100)),
                        )
                        result = result_with_meta(
                            evaluate_rows(selection_rows, scores, method=method),
                            family="linear_blend_ensemble",
                            target=f"{left.get('target')}+{right.get('target')}",
                            fit_weighting="val_grid",
                            train_rows_used=int(left.get("train_rows_used", 0)) + int(right.get("train_rows_used", 0)),
                        )
                        value = objective(result.get("metrics", {}))
                        if value > best_pair_value:
                            best_pair_value = value
                            best_pair = {
                                "method": method,
                                "left_method": left["method"],
                                "right_method": right["method"],
                                "left_weight": float(weight),
                                "right_weight": float(1.0 - weight),
                                "selection_objective": float(value),
                                "selection_result": result,
                                "left": left,
                                "right": right,
                            }
                    if best_pair is not None:
                        pair_best.append(best_pair)
            pair_best.sort(key=lambda item: item["selection_objective"], reverse=True)
            for item in pair_best[: max(0, int(args.ensemble_keep_top_k))]:
                left = item["left"]
                right = item["right"]
                weight = float(item["left_weight"])
                candidate_evals = {}
                scores_by_split = {}
                for split_name, rows in split_rows.items():
                    if not rows:
                        continue
                    if split_name not in left.get("_scores_by_split", {}) or split_name not in right.get("_scores_by_split", {}):
                        continue
                    scores = blend_scores(left["_scores_by_split"][split_name], right["_scores_by_split"][split_name], weight)
                    scores_by_split[split_name] = scores
                    result = result_with_meta(
                        evaluate_rows(rows, scores, method=item["method"]),
                        family="linear_blend_ensemble",
                        target=f"{left.get('target')}+{right.get('target')}",
                        fit_weighting="val_grid",
                        train_rows_used=int(left.get("train_rows_used", 0)) + int(right.get("train_rows_used", 0)),
                    )
                    candidate_evals[split_name] = result
                    evaluations.setdefault(split_name, []).append(result)
                runtime_candidates.append(
                    {
                        "method": item["method"],
                        "base_model_spec": "linear_blend_ensemble",
                        "family": "linear_blend_ensemble",
                        "target": f"{left.get('target')}+{right.get('target')}",
                        "fit_weighting": "val_grid",
                        "train_rows_used": int(left.get("train_rows_used", 0)) + int(right.get("train_rows_used", 0)),
                        "selected_alpha": None,
                        "selected_by": selection_split,
                        "selection_objective": float(item["selection_objective"]),
                        "left_method": item["left_method"],
                        "right_method": item["right_method"],
                        "left_weight": float(item["left_weight"]),
                        "right_weight": float(item["right_weight"]),
                        "evaluations": candidate_evals,
                        "_predict_kind": "precomputed",
                        "_scores_by_split": scores_by_split,
                    }
                )

    if runtime_candidates:
        selected_runtime = max(
            runtime_candidates,
            key=lambda item: objective(item.get("evaluations", {}).get("val", item.get("evaluations", {}).get("train", {})).get("metrics", {})),
        )
        selected_alpha = selected_runtime.get("selected_alpha")
        selected_by = str(selected_runtime.get("selected_by", "val"))
        selected_method_by_val = str(selected_runtime.get("method", ""))
    else:
        selected_runtime = {}

    prediction_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    if selected_runtime and int(args.write_predictions):
        model = selected_runtime.get("_model")
        kind = str(selected_runtime.get("_predict_kind", ""))
        for split_name, rows in split_rows.items():
            if not rows:
                continue
            if kind == "ridge":
                scores = predict_ridge_matrix(split_x[split_name], model)
            elif kind == "precomputed":
                scores = selected_runtime.get("_scores_by_split", {}).get(split_name, [])
            else:
                scores = [float(v) for v in model.predict(split_x[split_name])]
            pred_rows = []
            for row, score in zip(rows, scores):
                pred_rows.append(
                    {
                        "image_id": str(row.get("image_id", "")),
                        "candidate_id": str(row.get("candidate_id", "")),
                        "gt_annotation_id": row.get("gt_annotation_id"),
                        "protocol": row.get("protocol"),
                        "mos": safe_float(row.get("mos"), 0.0),
                        "score": float(score),
                        "method": selected_method_by_val,
                    }
                )
            prediction_rows_by_split[split_name] = pred_rows
            write_jsonl(output_dir / f"predictions_{split_name}.jsonl", pred_rows)

    model_candidates = [strip_runtime(candidate) for candidate in runtime_candidates]

    summary = {
        "protocol": str(args.protocol),
        "inputs": {
            "train_candidate_eval_jsonl": str(args.train_candidate_eval_jsonl or ""),
            "val_candidate_eval_jsonl": str(args.val_candidate_eval_jsonl or ""),
            "test_candidate_eval_jsonl": str(args.test_candidate_eval_jsonl or ""),
        },
        "score_fields": score_fields,
        "audit_fields": audit_fields,
        "extra_feature_fields": extra_feature_fields,
        "feature_fields": feature_fields,
        "fit_weighting": str(args.fit_weighting),
        "model_specs": model_specs,
        "feature_spec": spec,
        "selected_method_by_val": selected_method_by_val,
        "selected_alpha": selected_alpha,
        "selected_by": selected_by,
        "alpha_results": alpha_results,
        "model_candidates": model_candidates,
        "evaluations": evaluations,
        "prediction_counts": {split: len(rows) for split, rows in prediction_rows_by_split.items()},
    }
    write_json(output_dir / "calibration_summary.json", summary)
    (output_dir / "CALIBRATION_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected_method_by_val": selected_method_by_val,
                "selected_alpha": selected_alpha,
                "selected_by": selected_by,
                "model_candidate_count": len(model_candidates),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
