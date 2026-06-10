from __future__ import annotations

import base64
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


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [safe_float(v, 0.0) for v in a]
    bx1, by1, bx2, by2 = [safe_float(v, 0.0) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _center_area_box(cx: float, cy: float, area_ratio: float, image_ar: float, target_ar: float) -> List[float]:
    image_ar = max(1e-6, float(image_ar))
    target_ar = max(1e-6, float(target_ar))
    area_ratio = clamp(float(area_ratio), 1e-4, 0.98)
    w_norm = math.sqrt(area_ratio * target_ar / image_ar)
    h_norm = math.sqrt(area_ratio * image_ar / target_ar)
    return clip_box01([cx - 0.5 * w_norm, cy - 0.5 * h_norm, cx + 0.5 * w_norm, cy + 0.5 * h_norm])


def decode_support_grid(crop_guidance_spec: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(crop_guidance_spec, dict):
        return None
    support_spec = crop_guidance_spec.get("support_spec", {})
    if not isinstance(support_spec, dict):
        return None
    grid_size = int(safe_float(support_spec.get("grid_size", support_spec.get("support_grid_size", 0)), 0.0))
    mass_b64 = support_spec.get("mass_grid_f16_b64", support_spec.get("support_mass_grid_f16_b64"))
    if grid_size <= 0 or not isinstance(mass_b64, str) or not mass_b64:
        return None
    try:
        mass_grid = np.frombuffer(base64.b64decode(mass_b64.encode("ascii")), dtype=np.float16).astype(np.float32)
    except Exception:
        return None
    if mass_grid.size != grid_size * grid_size:
        return None
    mass_grid = mass_grid.reshape((grid_size, grid_size))
    mass_sum = float(mass_grid.sum())
    if mass_sum <= 1e-8:
        return None
    return mass_grid / mass_sum


def _crop_grid_slices(grid_size: int, crop: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in crop]
    gx1 = max(0, min(grid_size, int(math.floor(x1 * grid_size))))
    gy1 = max(0, min(grid_size, int(math.floor(y1 * grid_size))))
    gx2 = max(gx1 + 1, min(grid_size, int(math.ceil(x2 * grid_size))))
    gy2 = max(gy1 + 1, min(grid_size, int(math.ceil(y2 * grid_size))))
    return gx1, gy1, gx2, gy2


def _support_objective(
    crop_guidance_spec: Dict[str, Any],
    box: Sequence[float],
    support_grid: Optional[np.ndarray],
    support_envelope: Sequence[float],
) -> float:
    if isinstance(support_grid, np.ndarray) and support_grid.ndim == 2:
        grid_size = int(support_grid.shape[0])
        gx1, gy1, gx2, gy2 = _crop_grid_slices(grid_size, box)
        inside_mass = float(support_grid[gy1:gy2, gx1:gx2].sum())
        crop_w = max(1, gx2 - gx1)
        crop_h = max(1, gy2 - gy1)
        margin = max(1, int(math.ceil(0.08 * min(crop_w, crop_h))))
        edge_mask = np.zeros((crop_h, crop_w), dtype=bool)
        edge_mask[:margin, :] = True
        edge_mask[-margin:, :] = True
        edge_mask[:, :margin] = True
        edge_mask[:, -margin:] = True
        edge_mass = float(support_grid[gy1:gy2, gx1:gx2][edge_mask].sum())
    else:
        inside_mass = inter_area(support_envelope, box) / max(1e-8, box_area(support_envelope))
        edge_mass = 0.0
    envelope_area = max(1e-6, box_area(support_envelope))
    area_ratio = box_area(box)
    context_reward = clamp((area_ratio - envelope_area) / max(0.05, envelope_area), 0.0, 1.0)
    return float(inside_mass - 0.30 * edge_mass + 0.06 * context_reward)


def _pick_top_family_boxes(
    family: str,
    candidates: Sequence[Sequence[float]],
    crop_guidance_spec: Dict[str, Any],
    support_grid: Optional[np.ndarray],
    support_envelope: Sequence[float],
    max_per_family: int,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    seen = set()
    for box in candidates:
        clipped = clip_box01(box)
        if box_area(clipped) <= 1e-6:
            continue
        key = tuple(round(v, 5) for v in clipped)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            {
                "family": family,
                "bbox_norm_xyxy": clipped,
                "score": round(_support_objective(crop_guidance_spec, clipped, support_grid, support_envelope), 6),
            }
        )
    ranked.sort(key=lambda row: (safe_float(row.get("score", 0.0), 0.0), box_area(row.get("bbox_norm_xyxy", [0, 0, 0, 0]))), reverse=True)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ranked[: max(1, int(max_per_family))], start=1):
        item = dict(row)
        item["family_rank"] = idx
        out.append(item)
    return out


def seed_family_windows(
    crop_guidance_spec: Optional[Dict[str, Any]],
    *,
    target_ar: float,
    image_ar: float,
    max_per_family: int = 1,
) -> List[Dict[str, Any]]:
    if not isinstance(crop_guidance_spec, dict):
        return []
    support_spec = crop_guidance_spec.get("support_spec", {})
    layout_spec = crop_guidance_spec.get("layout_spec", {})
    if not isinstance(support_spec, dict):
        return []
    support_envelope = support_spec.get("envelope_norm_xyxy")
    if not (isinstance(support_envelope, (list, tuple)) and len(support_envelope) == 4):
        return []
    support_envelope = clip_box01(support_envelope)
    support_centroid_xy = support_spec.get("centroid_xy_norm")
    if isinstance(support_centroid_xy, (list, tuple)) and len(support_centroid_xy) == 2:
        support_centroid = (
            clamp(safe_float(support_centroid_xy[0], 0.5), 0.0, 1.0),
            clamp(safe_float(support_centroid_xy[1], 0.5), 0.0, 1.0),
        )
    else:
        support_centroid = box_center(support_envelope)
    envelope_center = box_center(support_envelope)
    envelope_area = max(0.04, min(0.85, box_area(support_envelope)))
    foreground_area = clamp(safe_float(support_spec.get("foreground_area_ratio", envelope_area), envelope_area), 0.0, 1.0)
    base_area = clamp(max(envelope_area, foreground_area, 0.04), 0.04, 0.85)
    attention_layout = str(support_spec.get("attention_layout", "none") or "none")
    support_grid = decode_support_grid(crop_guidance_spec)
    layout_modes = (
        list(layout_spec.get("composition_mode_candidates", []))
        if isinstance(layout_spec, dict) and isinstance(layout_spec.get("composition_mode_candidates"), list)
        else []
    )
    copyspace_pref = str(layout_spec.get("copyspace_preference", "none") or "none").strip().lower() if isinstance(layout_spec, dict) else "none"
    horizon_y = None
    if isinstance(layout_spec, dict) and layout_spec.get("horizon_y_norm") is not None:
        try:
            horizon_y = clamp(float(layout_spec.get("horizon_y_norm")), 0.0, 1.0)
        except Exception:
            horizon_y = None

    families: List[Tuple[str, List[List[float]]]] = []
    cover_areas = [clamp(max(base_area * 1.00, 0.06), 0.04, 0.95), clamp(max(base_area * 1.30, 0.10), 0.04, 0.95)]
    context_areas = [clamp(max(base_area * 1.70, 0.18), 0.04, 0.95), clamp(max(base_area * 2.20, 0.24), 0.04, 0.95)]
    wide_areas = [clamp(max(base_area * 2.60, 0.28), 0.08, 0.98), clamp(max(base_area * 3.20, 0.36), 0.08, 0.98)]

    families.append(
        (
            "support_cover_seed",
            [_center_area_box(cx, cy, area_t, image_ar, target_ar) for cx, cy in (support_centroid, envelope_center) for area_t in cover_areas],
        )
    )
    families.append(
        (
            "support_context_seed",
            [_center_area_box(cx, cy, area_t, image_ar, target_ar) for cx, cy in (envelope_center, support_centroid) for area_t in context_areas],
        )
    )
    if attention_layout in {"distributed", "none"}:
        families.append(
            (
                "scene_wide_seed",
                [_center_area_box(cx, cy, area_t, image_ar, target_ar) for cx, cy in (envelope_center, (0.5, 0.5)) for area_t in wide_areas],
            )
        )
    if attention_layout == "multi_subject":
        multi_area = clamp(max(base_area * 2.35, 0.22), 0.08, 0.98)
        families.append(
            (
                "component_union_seed",
                [
                    _center_area_box(envelope_center[0], envelope_center[1], multi_area, image_ar, target_ar),
                    _center_area_box(support_centroid[0], support_centroid[1], clamp(multi_area * 1.15, 0.08, 0.98), image_ar, target_ar),
                ],
            )
        )
    if "center" in layout_modes:
        families.append(("center_seed", [_center_area_box(0.5, 0.5, max(0.22, base_area * 1.8), image_ar, target_ar)]))
    if "thirds" in layout_modes and attention_layout == "dominant":
        cy = support_centroid[1]
        third_area = clamp(max(0.14, base_area * 1.45), 0.06, 0.95)
        families.append(("thirds_left_seed", [_center_area_box(1.0 / 3.0, cy, third_area, image_ar, target_ar)]))
        families.append(("thirds_right_seed", [_center_area_box(2.0 / 3.0, cy, third_area, image_ar, target_ar)]))
    if horizon_y is not None:
        horizon_area = clamp(max(0.28, base_area * 2.7), 0.08, 0.98)
        families.append(("horizon_band_wide", [_center_area_box(0.5, horizon_y, horizon_area, image_ar, target_ar)]))
    if copyspace_pref in {"left", "right"}:
        cx = 0.62 if copyspace_pref == "left" else 0.38
        cy = envelope_center[1]
        copy_area = clamp(max(0.24, base_area * 2.0), 0.08, 0.98)
        families.append((f"copyspace_{copyspace_pref}_seed", [_center_area_box(cx, cy, copy_area, image_ar, target_ar)]))
    if copyspace_pref in {"top", "bottom"}:
        cx = envelope_center[0]
        cy = 0.62 if copyspace_pref == "top" else 0.38
        copy_area = clamp(max(0.24, base_area * 2.0), 0.08, 0.98)
        families.append((f"copyspace_{copyspace_pref}_seed", [_center_area_box(cx, cy, copy_area, image_ar, target_ar)]))

    rows: List[Dict[str, Any]] = []
    for family, candidates in families:
        rows.extend(_pick_top_family_boxes(family, candidates, crop_guidance_spec, support_grid, support_envelope, max_per_family=max_per_family))
    rows.sort(key=lambda row: (safe_float(row.get("score", 0.0), 0.0), box_area(row.get("bbox_norm_xyxy", [0, 0, 0, 0]))), reverse=True)
    return rows


def support_mass_in_box(
    crop_guidance_spec: Optional[Dict[str, Any]],
    box: Optional[Sequence[float]],
) -> Optional[float]:
    if box is None:
        return None
    support_grid = decode_support_grid(crop_guidance_spec)
    clipped = clip_box01(box)
    if isinstance(support_grid, np.ndarray) and support_grid.ndim == 2:
        gx1, gy1, gx2, gy2 = _crop_grid_slices(int(support_grid.shape[0]), clipped)
        return float(clamp(float(support_grid[gy1:gy2, gx1:gx2].sum()), 0.0, 1.0))
    if not isinstance(crop_guidance_spec, dict):
        return None
    support_spec = crop_guidance_spec.get("support_spec", {})
    if not isinstance(support_spec, dict):
        return None
    envelope = support_spec.get("envelope_norm_xyxy")
    if not isinstance(envelope, (list, tuple)) or len(envelope) != 4:
        return None
    env = clip_box01(envelope)
    env_area = box_area(env)
    if env_area <= 1e-8:
        return None
    return float(clamp(inter_area(clipped, env) / env_area, 0.0, 1.0))
