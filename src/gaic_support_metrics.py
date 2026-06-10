from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clip_box01(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def box_area(box: Optional[Sequence[float]]) -> float:
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return 0.0
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not (isinstance(a, (list, tuple)) and len(a) == 4 and isinstance(b, (list, tuple)) and len(b) == 4):
        return 0.0
    ax1, ay1, ax2, ay2 = [safe_float(v, 0.0) for v in a]
    bx1, by1, bx2, by2 = [safe_float(v, 0.0) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 1e-8 else float(inter / union)


def _norm_bbox_xywh(bbox_xywh: Sequence[float], width: int, height: int) -> List[float]:
    x, y, w, h = [safe_float(v, 0.0) for v in bbox_xywh]
    iw = max(1.0, float(width))
    ih = max(1.0, float(height))
    return clip_box01([x / iw, y / ih, (x + w) / iw, (y + h) / ih])


def _select_top_annotations(
    anns: Sequence[Dict[str, Any]],
    *,
    top_fraction: float,
    min_topk: int,
    max_topk: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(anns, key=lambda ann: safe_float(ann.get("score", 0.0), 0.0), reverse=True)
    if not ordered:
        return []
    n = len(ordered)
    topk = int(math.ceil(float(n) * float(top_fraction)))
    topk = max(int(min_topk), min(int(max_topk), max(1, topk)))
    cutoff_score = safe_float(ordered[topk - 1].get("score", 0.0), 0.0)
    selected = [ann for ann in ordered if safe_float(ann.get("score", 0.0), 0.0) >= cutoff_score]
    return selected[: max(topk, len(selected))]


def build_latent_support(
    anns: Sequence[Dict[str, Any]],
    *,
    width: int,
    height: int,
    grid_size: int = 64,
    top_fraction: float = 0.20,
    min_topk: int = 3,
    max_topk: int = 12,
) -> Dict[str, Any]:
    selected = _select_top_annotations(anns, top_fraction=top_fraction, min_topk=min_topk, max_topk=max_topk)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if not selected:
        return {
            "grid": grid,
            "grid_size": int(grid_size),
            "mass_total": 0.0,
            "topk_count": 0,
            "latent_bbox_norm_xyxy": None,
            "centroid_xy_norm": [0.5, 0.5],
        }

    for ann in selected:
        box = _norm_bbox_xywh(ann.get("bbox", [0.0, 0.0, 0.0, 0.0]), width=width, height=height)
        x1 = max(0, min(grid_size, int(math.floor(float(box[0]) * grid_size))))
        y1 = max(0, min(grid_size, int(math.floor(float(box[1]) * grid_size))))
        x2 = max(x1 + 1, min(grid_size, int(math.ceil(float(box[2]) * grid_size))))
        y2 = max(y1 + 1, min(grid_size, int(math.ceil(float(box[3]) * grid_size))))
        grid[y1:y2, x1:x2] += 1.0

    mass_total = float(grid.sum())
    if mass_total > 0.0:
        grid /= mass_total

    ys, xs = np.mgrid[0:grid_size, 0:grid_size]
    total = float(grid.sum())
    if total > 0.0:
        centroid_x = float((grid * (xs + 0.5)).sum() / total) / float(grid_size)
        centroid_y = float((grid * (ys + 0.5)).sum() / total) / float(grid_size)
    else:
        centroid_x = 0.5
        centroid_y = 0.5

    max_val = float(grid.max()) if grid.size > 0 else 0.0
    latent_bbox = None
    if max_val > 0.0:
        mask = grid >= (0.50 * max_val)
        yy, xx = np.where(mask)
        if len(xx) > 0:
            latent_bbox = clip_box01(
                [
                    float(xx.min()) / float(grid_size),
                    float(yy.min()) / float(grid_size),
                    float(xx.max() + 1) / float(grid_size),
                    float(yy.max() + 1) / float(grid_size),
                ]
            )

    return {
        "grid": grid,
        "grid_size": int(grid_size),
        "mass_total": float(total),
        "topk_count": int(len(selected)),
        "latent_bbox_norm_xyxy": latent_bbox,
        "centroid_xy_norm": [round(float(centroid_x), 6), round(float(centroid_y), 6)],
    }


def compute_box_mass_metrics(box: Optional[Sequence[float]], latent_support: Dict[str, Any]) -> Dict[str, Any]:
    grid = latent_support.get("grid")
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or float(grid.sum()) <= 0.0:
        return {
            "mass_recall": 0.0,
            "centroid_inside": 0.0,
            "centroid_distance": 0.0,
            "latent_bbox_iou": 0.0,
            "box_area_ratio": box_area(box),
        }
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return {
            "mass_recall": 0.0,
            "centroid_inside": 0.0,
            "centroid_distance": 0.0,
            "latent_bbox_iou": 0.0,
            "box_area_ratio": 0.0,
        }

    box = clip_box01(box)
    grid_size = int(latent_support.get("grid_size", grid.shape[0]))
    x1 = max(0, min(grid_size, int(math.floor(float(box[0]) * grid_size))))
    y1 = max(0, min(grid_size, int(math.floor(float(box[1]) * grid_size))))
    x2 = max(x1 + 1, min(grid_size, int(math.ceil(float(box[2]) * grid_size))))
    y2 = max(y1 + 1, min(grid_size, int(math.ceil(float(box[3]) * grid_size))))
    mass_recall = float(grid[y1:y2, x1:x2].sum())

    cx, cy = [safe_float(v, 0.5) for v in latent_support.get("centroid_xy_norm", [0.5, 0.5])]
    centroid_inside = 1.0 if (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]) else 0.0
    box_cx = 0.5 * (box[0] + box[2])
    box_cy = 0.5 * (box[1] + box[3])
    centroid_distance = math.sqrt(((box_cx - cx) ** 2) + ((box_cy - cy) ** 2))

    return {
        "mass_recall": round(float(clamp(mass_recall, 0.0, 1.0)), 6),
        "centroid_inside": round(float(centroid_inside), 6),
        "centroid_distance": round(float(centroid_distance), 6),
        "latent_bbox_iou": round(float(iou_xyxy(box, latent_support.get("latent_bbox_norm_xyxy"))), 6),
        "box_area_ratio": round(float(box_area(box)), 6),
    }


def compute_subject_box_diagnostics(
    *,
    raw_anchor: Optional[Sequence[float]],
    candidate_anchor: Optional[Sequence[float]],
    support_region: Optional[Sequence[float]],
    gt_best_box: Optional[Sequence[float]],
    anns: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, Any]:
    latent_support = build_latent_support(anns, width=width, height=height)
    gt_best_metrics = compute_box_mass_metrics(gt_best_box, latent_support)
    raw_metrics = compute_box_mass_metrics(raw_anchor, latent_support)
    candidate_metrics = compute_box_mass_metrics(candidate_anchor, latent_support)
    support_metrics = compute_box_mass_metrics(support_region, latent_support)
    return {
        "latent_support_topk_count": int(latent_support.get("topk_count", 0)),
        "latent_support_centroid_xy_norm": latent_support.get("centroid_xy_norm", [0.5, 0.5]),
        "latent_support_bbox_norm_xyxy": latent_support.get("latent_bbox_norm_xyxy"),
        "gt_best": {
            "iou_to_gt_best": round(float(iou_xyxy(gt_best_box, gt_best_box)), 6) if gt_best_box is not None else 0.0,
            **gt_best_metrics,
        },
        "raw_anchor": {
            "iou_to_gt_best": round(float(iou_xyxy(raw_anchor, gt_best_box)), 6),
            **raw_metrics,
        },
        "candidate_anchor": {
            "iou_to_gt_best": round(float(iou_xyxy(candidate_anchor, gt_best_box)), 6),
            **candidate_metrics,
        },
        "support_region": {
            "iou_to_gt_best": round(float(iou_xyxy(support_region, gt_best_box)), 6),
            **support_metrics,
        },
    }
