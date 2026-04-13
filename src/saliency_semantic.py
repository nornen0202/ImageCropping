from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from subject_region import box_area, center_distance, clip_box01, iou_xyxy, norm_box_xyxy, summarize_saliency_signal, union_boxes


PERSON_CLASS_IDS = {0}
ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77}
VEHICLE_CLASS_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
FOOD_CLASS_IDS = {39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60}
PRODUCT_CLASS_IDS = {
    24,
    25,
    26,
    27,
    28,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    78,
    79,
}

CAPTION_FOOD_HINTS = ("food", "meal", "dish", "steak", "plate", "bowl", "dessert", "drink", "fruit", "table")
CAPTION_BOOK_HINTS = ("book", "notebook", "magazine", "newspaper", "menu", "page", "paper")
CAPTION_BUILDING_HINTS = ("building", "architecture", "interior", "room", "house", "bridge", "tower", "city")
CAPTION_LANDSCAPE_HINTS = ("landscape", "mountain", "forest", "beach", "river", "lake", "ocean", "sky", "nature")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _intersection_ratio_to_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    denom = min(max(1e-8, box_area(a)), max(1e-8, box_area(b)))
    return float(inter / denom)


def _box_width_to_height(box: Sequence[float]) -> float:
    w = max(1e-8, float(box[2]) - float(box[0]))
    h = max(1e-8, float(box[3]) - float(box[1]))
    return float(w / h)


def _normalize_free_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _text_has_any(text: str, hints: Sequence[str]) -> bool:
    return any(h in text for h in hints)


def _decode_support_mass_grid(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    grid_size = int(_safe_float(payload.get("support_grid_size", 0), 0.0))
    mass_b64 = payload.get("support_mass_grid_f16_b64")
    if grid_size <= 0 or not isinstance(mass_b64, str) or not mass_b64:
        return None
    try:
        raw = base64.b64decode(mass_b64)
        grid = np.frombuffer(raw, dtype=np.float16)
        if grid.size != grid_size * grid_size:
            return None
        out = grid.astype(np.float32).reshape((grid_size, grid_size))
        total = float(out.sum())
        if total <= 1e-8:
            return None
        return out / total
    except Exception:
        return None


def _crop_grid_slices(grid_size: int, crop: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in crop]
    gx1 = max(0, min(grid_size, int(math.floor(x1 * grid_size))))
    gy1 = max(0, min(grid_size, int(math.floor(y1 * grid_size))))
    gx2 = max(gx1 + 1, min(grid_size, int(math.ceil(x2 * grid_size))))
    gy2 = max(gy1 + 1, min(grid_size, int(math.ceil(y2 * grid_size))))
    return gx1, gy1, gx2, gy2


def _support_mass_for_box(box_norm_xyxy: Sequence[float], saliency_summary: Dict[str, Any], mass_grid: Optional[np.ndarray]) -> float:
    box = clip_box01(box_norm_xyxy)
    if mass_grid is not None:
        gx1, gy1, gx2, gy2 = _crop_grid_slices(int(mass_grid.shape[0]), box)
        return _clamp(float(mass_grid[gy1:gy2, gx1:gx2].sum()), 0.0, 1.0)
    support_box = saliency_summary.get("support_bbox_norm_xyxy")
    if isinstance(support_box, (list, tuple)) and len(support_box) == 4:
        denom = max(1e-8, box_area(support_box))
        inter = union_boxes([support_box, box])
        if inter is None:
            return 0.0
        ix1 = max(float(support_box[0]), float(box[0]))
        iy1 = max(float(support_box[1]), float(box[1]))
        ix2 = min(float(support_box[2]), float(box[2]))
        iy2 = min(float(support_box[3]), float(box[3]))
        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return _clamp(inter_area / denom, 0.0, 1.0)
    return 0.0


def _coarse_family_and_subtype(class_id: int, caption_text: str) -> Tuple[str, str]:
    if class_id in PERSON_CLASS_IDS:
        return "human", ""
    if class_id in ANIMAL_CLASS_IDS:
        return "animal", "animal"
    if class_id in VEHICLE_CLASS_IDS:
        return "object", "vehicle"
    if class_id in FOOD_CLASS_IDS:
        return "object", "food"
    if class_id == 73 or _text_has_any(caption_text, CAPTION_BOOK_HINTS):
        return "object", "book"
    if class_id in PRODUCT_CLASS_IDS:
        return "object", "product"
    return "object", ""


def _cluster_merge_allowed(cluster: Dict[str, Any], atom: Dict[str, Any]) -> bool:
    if str(cluster.get("family", "")) != str(atom.get("family", "")):
        return False
    cluster_box = cluster.get("bbox_norm_xyxy")
    atom_box = atom.get("bbox_norm_xyxy")
    if not isinstance(cluster_box, list) or len(cluster_box) != 4:
        return False
    if not isinstance(atom_box, list) or len(atom_box) != 4:
        return False
    family = str(cluster.get("family", ""))
    subtype = str(cluster.get("subtype", "") or "")
    atom_subtype = str(atom.get("subtype", "") or "")
    iou = iou_xyxy(cluster_box, atom_box)
    containment = _intersection_ratio_to_smaller(cluster_box, atom_box)
    dist = center_distance(cluster_box, atom_box)
    if family == "human":
        return bool(iou >= 0.25 or containment >= 0.55 or dist <= 0.11)
    subtype_match = (not subtype) or (not atom_subtype) or subtype == atom_subtype
    return bool(subtype_match and (iou >= 0.35 or containment >= 0.60 or dist <= 0.10))


def _trust_tier_from_signals(*, saliency_summary: Dict[str, Any], primary_mass: float, attributed_mass: float) -> str:
    support_map_available = bool(saliency_summary.get("support_map_available", False))
    dominance = _safe_float(saliency_summary.get("dominance_score", 0.0), 0.0)
    if support_map_available and primary_mass >= 0.25 and attributed_mass >= 0.40 and dominance >= 0.45:
        return "high"
    if primary_mass >= 0.12 or attributed_mass >= 0.25:
        return "medium"
    return "low"


def build_saliency_semantic_summary(
    *,
    width: int,
    height: int,
    c2_instances: Sequence[Dict[str, Any]],
    c3_pose: Any,
    c7_saliency: Optional[Dict[str, Any]],
    ocr_text_boxes_count: int = 0,
    caption_text: Optional[str] = None,
) -> Dict[str, Any]:
    saliency_summary = summarize_saliency_signal(c7_saliency, width=width, height=height)
    mass_grid = _decode_support_mass_grid(saliency_summary)
    caption_norm = _normalize_free_text(caption_text)
    atoms: List[Dict[str, Any]] = []
    human_pose_boxes: List[List[float]] = []

    for pose in c3_pose if isinstance(c3_pose, list) else []:
        if not isinstance(pose, dict):
            continue
        pose_box = norm_box_xyxy(pose.get("bbox"), width, height)
        if box_area(pose_box) <= 0.0:
            continue
        face = pose.get("face", {}) if isinstance(pose.get("face"), dict) else {}
        pose_score = _safe_float(pose.get("score", 0.0), 0.0)
        face_score = _safe_float(face.get("score", 0.0), 0.0)
        aspect_wh = _box_width_to_height(pose_box)
        if face_score >= 0.20 or (pose_score >= 0.55 and aspect_wh <= 1.10):
            human_pose_boxes.append(pose_box)

    for idx, inst in enumerate(c2_instances):
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        box = norm_box_xyxy(inst.get("box"), width, height)
        if box_area(box) <= 0.0:
            continue
        class_id = int(_safe_float(inst.get("class_id", -1), -1))
        family, subtype = _coarse_family_and_subtype(class_id, caption_norm)
        area_ratio = _clamp(_safe_float(inst.get("area_ratio", box_area(box)), box_area(box)), 0.0, 1.0)
        importance = _clamp(_safe_float(inst.get("importance_score", 0.0), 0.0), 0.0, 1.0)
        det_score = _clamp(_safe_float(inst.get("score", 0.0), 0.0), 0.0, 1.0)
        support_mass = _support_mass_for_box(box, saliency_summary, mass_grid)
        if family == "human":
            pose_overlap = max((_intersection_ratio_to_smaller(box, pbox) for pbox in human_pose_boxes), default=0.0)
            aspect_wh = _box_width_to_height(box)
            wide_weak_person = bool(aspect_wh > 1.35 and pose_overlap < 0.10 and support_mass < 0.05 and det_score < 0.75)
            trust = _clamp(0.18 + 0.25 * det_score + 0.32 * pose_overlap + 0.25 * support_mass, 0.0, 1.0)
            if wide_weak_person:
                trust *= 0.40
        else:
            shape_source = "mask" if bool(inst.get("mask_rle")) else "box"
            shape_bonus = 0.10 if shape_source == "mask" else 0.0
            trust = _clamp(0.25 + 0.35 * support_mass + 0.20 * importance + 0.10 * area_ratio + shape_bonus, 0.0, 1.0)
        if family == "human" and trust < 0.22 and support_mass < 0.03:
            continue
        if support_mass < 0.01 and importance < 0.20 and area_ratio < 0.01 and family != "human":
            continue
        atoms.append(
            {
                "atom_id": f"atom_{idx}",
                "family": family,
                "subtype": subtype,
                "bbox_norm_xyxy": box,
                "area_ratio": round(float(area_ratio), 6),
                "support_mass": round(float(support_mass), 6),
                "importance": round(float(importance), 6),
                "trust": round(float(trust), 6),
                "shape_source": "mask" if bool(inst.get("mask_rle")) else "box",
            }
        )

    if not any(str(a.get("family")) == "human" for a in atoms):
        for idx, pbox in enumerate(human_pose_boxes):
            support_mass = _support_mass_for_box(pbox, saliency_summary, mass_grid)
            trust = _clamp(0.30 + 0.35 * support_mass + 0.20, 0.0, 1.0)
            atoms.append(
                {
                    "atom_id": f"pose_{idx}",
                    "family": "human",
                    "subtype": "",
                    "bbox_norm_xyxy": pbox,
                    "area_ratio": round(float(box_area(pbox)), 6),
                    "support_mass": round(float(support_mass), 6),
                    "importance": round(float(0.35 + 0.35 * support_mass), 6),
                    "trust": round(float(trust), 6),
                    "shape_source": "pose",
                }
            )

    atoms.sort(
        key=lambda row: (
            float(row.get("support_mass", 0.0)),
            float(row.get("trust", 0.0)),
            float(row.get("importance", 0.0)),
            float(row.get("area_ratio", 0.0)),
        ),
        reverse=True,
    )

    clusters: List[Dict[str, Any]] = []
    for atom in atoms:
        matched = False
        for cluster in clusters:
            if _cluster_merge_allowed(cluster, atom):
                members = cluster.setdefault("members", [])
                members.append(atom)
                merged_box = union_boxes([cluster["bbox_norm_xyxy"], atom["bbox_norm_xyxy"]])
                if merged_box is not None:
                    cluster["bbox_norm_xyxy"] = merged_box
                matched = True
                break
        if not matched:
            clusters.append(
                {
                    "cluster_id": f"cluster_{len(clusters)}",
                    "family": atom["family"],
                    "subtype": atom["subtype"],
                    "bbox_norm_xyxy": list(atom["bbox_norm_xyxy"]),
                    "members": [atom],
                }
            )

    cluster_rows: List[Dict[str, Any]] = []
    for cluster in clusters:
        bbox = clip_box01(cluster["bbox_norm_xyxy"])
        members = cluster.get("members", [])
        support_mass = _support_mass_for_box(bbox, saliency_summary, mass_grid)
        density = support_mass / max(1e-8, box_area(bbox))
        avg_trust = (
            sum(float(member.get("trust", 0.0)) for member in members) / float(max(1, len(members)))
            if members
            else 0.0
        )
        subtype = str(cluster.get("subtype", "") or "")
        if not subtype:
            family = str(cluster.get("family", ""))
            if family == "scene_region":
                if _text_has_any(caption_norm, CAPTION_BUILDING_HINTS):
                    subtype = "building_hint"
                elif _text_has_any(caption_norm, CAPTION_LANDSCAPE_HINTS):
                    subtype = "landscape_hint"
            elif family == "object":
                if _text_has_any(caption_norm, CAPTION_FOOD_HINTS):
                    subtype = "food"
                elif _text_has_any(caption_norm, CAPTION_BOOK_HINTS):
                    subtype = "book"
        cluster_rows.append(
            {
                "cluster_id": str(cluster.get("cluster_id", "")),
                "family": str(cluster.get("family", "unknown") or "unknown"),
                "subtype": subtype,
                "member_count": int(len(members)),
                "support_mass": round(float(_clamp(support_mass, 0.0, 1.0)), 6),
                "support_density": round(float(_clamp(density, 0.0, 1.0)), 6),
                "bbox_norm_xyxy": bbox,
                "shape_source": "mask" if any(str(member.get("shape_source", "")) == "mask" for member in members) else "box",
                "trust": round(float(_clamp(avg_trust, 0.0, 1.0)), 6),
            }
        )

    cluster_rows.sort(
        key=lambda row: (
            float(row.get("support_mass", 0.0)),
            float(row.get("trust", 0.0)),
            float(box_area(row.get("bbox_norm_xyxy", [0.0, 0.0, 0.0, 0.0]))),
        ),
        reverse=True,
    )

    primary = cluster_rows[0] if cluster_rows else None
    second = cluster_rows[1] if len(cluster_rows) > 1 else None
    primary_family = str(primary.get("family", "")) if isinstance(primary, dict) else ""
    primary_subtype = str(primary.get("subtype", "")) if isinstance(primary, dict) else ""
    primary_mass = float(primary.get("support_mass", 0.0)) if isinstance(primary, dict) else 0.0
    attributed_mass = _clamp(sum(float(row.get("support_mass", 0.0)) for row in cluster_rows), 0.0, 1.0)
    layout_structure = "single"
    if bool(saliency_summary.get("distributed_attention_likely", False)) and primary_mass < 0.35:
        layout_structure = "distributed"
    elif second is not None:
        second_mass = float(second.get("support_mass", 0.0))
        primary_box = primary.get("bbox_norm_xyxy", [0.0, 0.0, 0.0, 0.0]) if isinstance(primary, dict) else [0.0, 0.0, 0.0, 0.0]
        second_box = second.get("bbox_norm_xyxy", [0.0, 0.0, 0.0, 0.0]) if isinstance(second, dict) else [0.0, 0.0, 0.0, 0.0]
        if second_mass >= 0.18 and second_mass >= max(0.12, 0.55 * primary_mass) and center_distance(primary_box, second_box) >= 0.12:
            layout_structure = "multi"

    if not primary_family:
        if int(ocr_text_boxes_count) > 0:
            primary_family = "text"
        elif bool(saliency_summary.get("scene_like_signal", False)) or layout_structure == "distributed":
            primary_family = "scene_region"
        else:
            primary_family = "unknown"

    if primary_family == "unknown" and bool(saliency_summary.get("scene_like_signal", False)):
        primary_family = "scene_region"
    if primary_family == "scene_region" and not primary_subtype:
        if _text_has_any(caption_norm, CAPTION_BUILDING_HINTS):
            primary_subtype = "building_hint"
        elif _text_has_any(caption_norm, CAPTION_LANDSCAPE_HINTS):
            primary_subtype = "landscape_hint"

    primary_box = primary.get("bbox_norm_xyxy") if isinstance(primary, dict) else None
    union_box = union_boxes([row.get("bbox_norm_xyxy", []) for row in cluster_rows[:2] if isinstance(row, dict)])
    human_cluster_count = sum(1 for row in cluster_rows if str(row.get("family", "")) == "human")
    human_union_box = union_boxes([row.get("bbox_norm_xyxy", []) for row in cluster_rows if str(row.get("family", "")) == "human"])
    tier = _trust_tier_from_signals(
        saliency_summary=saliency_summary,
        primary_mass=primary_mass,
        attributed_mass=attributed_mass,
    )

    semantic_agreement = 0.0
    if primary is not None:
        dominant_support = saliency_summary.get("top_component_bbox_norm_xyxy")
        if isinstance(dominant_support, (list, tuple)) and len(dominant_support) == 4:
            semantic_agreement = iou_xyxy(primary["bbox_norm_xyxy"], dominant_support)
        else:
            semantic_agreement = primary_mass

    return {
        "primary_family": primary_family,
        "primary_subtype": primary_subtype,
        "layout_structure": layout_structure,
        "copyspace_intent": False,
        "support_trust_tier": tier,
        "latent_support_bbox_norm_xyxy": saliency_summary.get("latent_support_bbox_norm_xyxy"),
        "latent_support_core_bbox_norm_xyxy": saliency_summary.get("latent_support_core_bbox_norm_xyxy"),
        "primary_cluster_bbox_norm_xyxy": primary_box,
        "union_bbox_norm_xyxy": union_box,
        "human_union_bbox_norm_xyxy": human_union_box,
        "human_cluster_count": int(human_cluster_count),
        "primary_cluster_count": 0 if primary is None else 1,
        "primary_support_mass": round(float(primary_mass), 6),
        "attributed_support_mass": round(float(attributed_mass), 6),
        "residual_scene_mass": round(float(_clamp(1.0 - attributed_mass, 0.0, 1.0)), 6),
        "semantic_agreement": round(float(_clamp(semantic_agreement, 0.0, 1.0)), 6),
        "text_box_count": int(max(0, int(ocr_text_boxes_count))),
        "cluster_rows": cluster_rows,
    }
