"""
generate_candidates.py
AR-conditioned candidate generator (Phase B1).

Input:
- filtered parquet (image_id, width, height, tags, ...)
- feats_c2.jsonl (image_id -> c2_seg)
- feats_c3.jsonl (image_id -> c3_pose)

Output:
- candidates_ar.jsonl
  {
    image_id, width, height, image_ar, tags,
    subject_prior, candidate_gen_hash,
    candidates_by_ar, stats
  }
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_AR_LIST = ("1:1", "9:16", "16:9", "3:4", "4:3")
DEFAULT_SCALE_SET = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_JITTER_FRACS = (0.0, 0.03, 0.06)
DEFAULT_JITTER_SCALES = (0.45, 0.55, 0.65)
DEFAULT_PHI_SCALES = (0.35, 0.55, 0.75)
DEFAULT_OBJ_TEMPLATE_SCALES = (1.1, 1.25, 1.45)

COPYSPACE_HINT_WORDS = (
    "copy space",
    "copy-space",
    "copyspace",
    "negative space",
    "blank space",
    "text space",
    "banner",
    "template",
)


@dataclass
class CandidateGenConfig:
    ar_list: Tuple[str, ...] = DEFAULT_AR_LIST
    grid_m: int = 12
    grid_n: int = 12
    scale_set: Tuple[float, ...] = DEFAULT_SCALE_SET
    a_min: float = 0.15
    a_max: float = 0.95
    ar_tol: float = 0.02
    max_candidates_per_ar: int = 240
    nms_iou: float = 0.90
    eps_ar_full: float = 0.01
    maxarea_slide_offsets: Tuple[float, ...] = (-0.25, 0.0, 0.25)
    jitter_fracs: Tuple[float, ...] = DEFAULT_JITTER_FRACS
    jitter_scales: Tuple[float, ...] = DEFAULT_JITTER_SCALES
    use_phi_thirds: bool = True
    phi_scales: Tuple[float, ...] = DEFAULT_PHI_SCALES
    object_template_scales: Tuple[float, ...] = DEFAULT_OBJ_TEMPLATE_SCALES
    min_subject_size: float = 0.08


def parse_ar(ar_text: str) -> float:
    t = ar_text.strip()
    if ":" in t:
        a, b = t.split(":", 1)
        return float(a) / float(b)
    return float(t)


def ar_token(ar_text: str) -> str:
    return ar_text.strip().replace(":", "x")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def norm_box_xyxy(box: Sequence[float], width: float, height: float) -> List[float]:
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 1.0, 1.0]

    x1, y1, x2, y2 = [float(x) for x in box]
    x1n = clamp(x1 / width, 0.0, 1.0)
    y1n = clamp(y1 / height, 0.0, 1.0)
    x2n = clamp(x2 / width, 0.0, 1.0)
    y2n = clamp(y2 / height, 0.0, 1.0)

    if x2n < x1n:
        x1n, x2n = x2n, x1n
    if y2n < y1n:
        y1n, y2n = y2n, y1n
    return [x1n, y1n, x2n, y2n]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_wh(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    x1 = min(float(b[0]) for b in boxes)
    y1 = min(float(b[1]) for b in boxes)
    x2 = max(float(b[2]) for b in boxes)
    y2 = max(float(b[3]) for b in boxes)
    return [
        clamp(x1, 0.0, 1.0),
        clamp(y1, 0.0, 1.0),
        clamp(x2, 0.0, 1.0),
        clamp(y2, 0.0, 1.0),
    ]


def actual_ar(box: Sequence[float], image_ar: float) -> float:
    w, h = box_wh(box)
    if h <= 0:
        return 0.0
    return (w * image_ar) / h


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = box_area(a) + box_area(b) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def dedupe_by_rounded_box(
    cands: Sequence[Dict[str, Any]],
    ndigits: int = 6,
) -> List[Dict[str, Any]]:
    kept: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}
    for c in cands:
        box = c["bbox_norm_xyxy"]
        key = tuple(round(float(v), ndigits) for v in box)
        if key not in kept:
            kept[key] = dict(c)
            continue

        prev = kept[key]
        prev["source_types"] = sorted(
            set(prev.get("source_types", [])) | set(c.get("source_types", []))
        )
        prev["must_keep"] = bool(
            prev.get("must_keep", False) or c.get("must_keep", False)
        )
    return list(kept.values())


def nms_candidates(
    candidates: Sequence[Dict[str, Any]],
    iou_thr: float,
    existing: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    kept: List[Dict[str, Any]] = []
    if existing:
        kept.extend(existing)

    sorted_idx = sorted(
        range(len(candidates)),
        key=lambda i: (
            1 if candidates[i].get("must_keep", False) else 0,
            candidates[i].get("priority", 0.0),
            candidates[i].get("area_ratio", 0.0),
        ),
        reverse=True,
    )

    accepted: List[Dict[str, Any]] = []
    for idx in sorted_idx:
        cand = candidates[idx]
        box = cand["bbox_norm_xyxy"]
        if any(iou_xyxy(box, k["bbox_norm_xyxy"]) > iou_thr for k in kept):
            continue
        kept.append(cand)
        accepted.append(cand)
    return accepted


def diversity_sample(candidates: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    if len(candidates) <= k:
        return list(candidates)

    feats = []
    for c in candidates:
        x1, y1, x2, y2 = c["bbox_norm_xyxy"]
        cx, cy = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        w, h = (x2 - x1, y2 - y1)
        feats.append([cx, cy, math.log(max(w, 1e-8)), math.log(max(h, 1e-8))])
    arr = np.asarray(feats, dtype=np.float32)

    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            candidates[i].get("priority", 0.0),
            candidates[i].get("area_ratio", 0.0),
        ),
        reverse=True,
    )

    picked = [order[0]]
    dists = np.linalg.norm(arr - arr[picked[0]][None, :], axis=1)

    while len(picked) < k:
        for p in picked:
            dists[p] = -1.0
        nxt = int(np.argmax(dists))
        if dists[nxt] < 0:
            break
        picked.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(arr - arr[nxt][None, :], axis=1))

    picked = picked[:k]
    return [candidates[i] for i in picked]


def ensure_jsonable_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        if "|" in raw_tags:
            return [x.strip().lower() for x in raw_tags.split("|") if x.strip()]
        return [raw_tags.strip().lower()] if raw_tags.strip() else []
    if isinstance(raw_tags, (list, tuple)):
        return [str(x).strip().lower() for x in raw_tags if str(x).strip()]
    if isinstance(raw_tags, np.ndarray):
        return [str(x).strip().lower() for x in raw_tags.tolist() if str(x).strip()]
    return [str(raw_tags).strip().lower()] if str(raw_tags).strip() else []


def has_copy_space_hint(tags: Sequence[str]) -> bool:
    text = " ".join(tags)
    return any(w in text for w in COPYSPACE_HINT_WORDS)


def resolve_subject_prior(
    width: int,
    height: int,
    c2_seg: Optional[List[Dict[str, Any]]],
    c3_pose: Optional[List[Dict[str, Any]]],
    cfg: CandidateGenConfig,
) -> Dict[str, Any]:
    boxes: List[List[float]] = []
    source_parts: List[str] = []

    if c2_seg:
        def c2_rank(seg: Dict[str, Any]) -> float:
            area = float(seg.get("area", 0.0))
            score = float(seg.get("score", 0.0))
            return score + 0.001 * math.sqrt(max(area, 0.0))

        best = max(c2_seg, key=c2_rank)
        if "box" in best:
            boxes.append(norm_box_xyxy(best["box"], float(width), float(height)))
            source_parts.append("c2")

    num_people = 0
    if c3_pose:
        pose_boxes = []
        for p in c3_pose:
            b = p.get("bbox")
            if not b:
                continue
            pose_boxes.append(norm_box_xyxy(b, float(width), float(height)))
        num_people = len(pose_boxes)
        if pose_boxes:
            pb = union_boxes(pose_boxes)
            if pb is not None:
                boxes.append(pb)
                source_parts.append("c3")

    subj_box = union_boxes(boxes)
    if subj_box is None or box_area(subj_box) <= 0:
        subj_box = [0.25, 0.25, 0.75, 0.75]
        source = "fallback_center"
    else:
        source = "_".join(source_parts) if source_parts else "unknown"

    cx, cy = box_center(subj_box)
    ws, hs = box_wh(subj_box)
    ws = max(ws, cfg.min_subject_size)
    hs = max(hs, cfg.min_subject_size)

    return {
        "source": source,
        "bbox_norm_xyxy": [round(v, 6) for v in subj_box],
        "centroid": [round(cx, 6), round(cy, 6)],
        "size": [round(ws, 6), round(hs, 6)],
        "has_people": num_people > 0,
        "num_people": int(num_people),
    }


def add_candidate(
    out: List[Dict[str, Any]],
    box: Sequence[float],
    image_ar: float,
    target_ar: float,
    ar_tol: float,
    a_min: float,
    a_max: float,
    source: str,
    scale_idx: str,
    must_keep: bool = False,
    priority: float = 0.0,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        return
    if x2 <= x1 or y2 <= y1:
        return
    b = [x1, y1, x2, y2]
    area = box_area(b)
    if area < a_min or area > a_max:
        return
    ar_val = actual_ar(b, image_ar)
    if abs(ar_val - target_ar) > ar_tol:
        return
    out.append(
        {
            "bbox_norm_xyxy": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
            "area_ratio": round(area, 6),
            "source_types": [source],
            "source": source,
            "scale_idx": str(scale_idx),
            "must_keep": bool(must_keep),
            "priority": float(priority),
        }
    )


def add_center_area_box(
    out: List[Dict[str, Any]],
    cx: float,
    cy: float,
    area_t: float,
    image_ar: float,
    target_ar: float,
    cfg: CandidateGenConfig,
    source: str,
    scale_idx: str,
    must_keep: bool = False,
    priority: float = 0.0,
) -> None:
    w_norm = math.sqrt(max(area_t, 1e-12) * target_ar / image_ar)
    h_norm = math.sqrt(max(area_t, 1e-12) * image_ar / target_ar)
    x1 = cx - w_norm * 0.5
    y1 = cy - h_norm * 0.5
    x2 = cx + w_norm * 0.5
    y2 = cy + h_norm * 0.5

    add_candidate(
        out=out,
        box=[x1, y1, x2, y2],
        image_ar=image_ar,
        target_ar=target_ar,
        ar_tol=cfg.ar_tol,
        a_min=cfg.a_min,
        a_max=cfg.a_max,
        source=source,
        scale_idx=scale_idx,
        must_keep=must_keep,
        priority=priority,
    )


def generate_baseline_candidates(
    target_ar: float,
    image_ar: float,
    subject_centroid: Tuple[float, float],
    has_copy_hint: bool,
    cfg: CandidateGenConfig,
) -> List[Dict[str, Any]]:
    # Baseline candidates are "must-include" anchors to avoid systematic over-cropping:
    # 1) full-frame when AR already matches
    # 2) max-area center crop for target AR
    # 3) max-area slide variants aligned to subject/copy-space intent
    cands: List[Dict[str, Any]] = []

    if abs(image_ar - target_ar) <= cfg.eps_ar_full:
        add_candidate(
            out=cands,
            box=[0.0, 0.0, 1.0, 1.0],
            image_ar=image_ar,
            target_ar=target_ar,
            ar_tol=cfg.ar_tol,
            a_min=cfg.a_min,
            a_max=cfg.a_max,
            source="baseline_full",
            scale_idx="full",
            must_keep=True,
            priority=10.0,
        )

    cx_subj, cy_subj = subject_centroid
    offsets = list(cfg.maxarea_slide_offsets)
    if has_copy_hint:
        offsets = [-0.45, -0.25, 0.0, 0.25, 0.45]

    if image_ar >= target_ar:
        bw = target_ar / image_ar
        slack = max(0.0, 1.0 - bw)
        x1 = 0.5 * slack
        add_candidate(
            out=cands,
            box=[x1, 0.0, x1 + bw, 1.0],
            image_ar=image_ar,
            target_ar=target_ar,
            ar_tol=cfg.ar_tol,
            a_min=cfg.a_min,
            a_max=cfg.a_max,
            source="baseline_maxarea_center",
            scale_idx="maxc",
            must_keep=True,
            priority=9.0,
        )

        for i, frac in enumerate(offsets):
            x1s = clamp(cx_subj - bw * 0.5 + frac * slack, 0.0, slack)
            add_candidate(
                out=cands,
                box=[x1s, 0.0, x1s + bw, 1.0],
                image_ar=image_ar,
                target_ar=target_ar,
                ar_tol=cfg.ar_tol,
                a_min=cfg.a_min,
                a_max=cfg.a_max,
                source="baseline_maxarea_slide",
                scale_idx=f"mx{i}",
                must_keep=True,
                priority=8.5,
            )

        if has_copy_hint:
            desired_in_crop = 0.35 if cx_subj < 0.5 else 0.65
            x1_copy = clamp(cx_subj - desired_in_crop * bw, 0.0, slack)
            add_candidate(
                out=cands,
                box=[x1_copy, 0.0, x1_copy + bw, 1.0],
                image_ar=image_ar,
                target_ar=target_ar,
                ar_tol=cfg.ar_tol,
                a_min=cfg.a_min,
                a_max=cfg.a_max,
                source="copyspace_align",
                scale_idx="copyx",
                must_keep=False,
                priority=8.0,
            )
        return cands

    bh = image_ar / target_ar
    slack = max(0.0, 1.0 - bh)
    y1 = 0.5 * slack
    add_candidate(
        out=cands,
        box=[0.0, y1, 1.0, y1 + bh],
        image_ar=image_ar,
        target_ar=target_ar,
        ar_tol=cfg.ar_tol,
        a_min=cfg.a_min,
        a_max=cfg.a_max,
        source="baseline_maxarea_center",
        scale_idx="maxc",
        must_keep=True,
        priority=9.0,
    )

    for i, frac in enumerate(offsets):
        y1s = clamp(cy_subj - bh * 0.5 + frac * slack, 0.0, slack)
        add_candidate(
            out=cands,
            box=[0.0, y1s, 1.0, y1s + bh],
            image_ar=image_ar,
            target_ar=target_ar,
            ar_tol=cfg.ar_tol,
            a_min=cfg.a_min,
            a_max=cfg.a_max,
            source="baseline_maxarea_slide",
            scale_idx=f"my{i}",
            must_keep=True,
            priority=8.5,
        )

    if has_copy_hint:
        desired_in_crop = 0.35 if cy_subj < 0.5 else 0.65
        y1_copy = clamp(cy_subj - desired_in_crop * bh, 0.0, slack)
        add_candidate(
            out=cands,
            box=[0.0, y1_copy, 1.0, y1_copy + bh],
            image_ar=image_ar,
            target_ar=target_ar,
            ar_tol=cfg.ar_tol,
            a_min=cfg.a_min,
            a_max=cfg.a_max,
            source="copyspace_align",
            scale_idx="copyy",
            must_keep=False,
            priority=8.0,
        )
    return cands


def generate_candidates_for_ar(
    target_ar: float,
    target_ar_text: str,
    image_ar: float,
    subject_centroid: Tuple[float, float],
    subject_size: Tuple[float, float],
    has_copy_hint: bool,
    cfg: CandidateGenConfig,
) -> List[Dict[str, Any]]:
    # Core generation pipeline (design doc section 8):
    # baseline -> global grid -> subject templates -> saliency jitter -> phi/thirds
    # -> dedupe/NMS -> must-keep preservation -> diversity sampling -> stable candidate IDs.
    cx_subj, cy_subj = subject_centroid
    ws, hs = subject_size
    all_cands: List[Dict[str, Any]] = []

    all_cands.extend(
        generate_baseline_candidates(
            target_ar=target_ar,
            image_ar=image_ar,
            subject_centroid=subject_centroid,
            has_copy_hint=has_copy_hint,
            cfg=cfg,
        )
    )

    xs = [i / cfg.grid_m for i in range(cfg.grid_m + 1)]
    ys = [j / cfg.grid_n for j in range(cfg.grid_n + 1)]
    # (A) Global AR-conditional coverage via grid anchors + multi-scale areas.
    for s_idx, area_t in enumerate(cfg.scale_set):
        if not (cfg.a_min <= area_t <= cfg.a_max):
            continue
        for cx in xs:
            for cy in ys:
                add_center_area_box(
                    out=all_cands,
                    cx=cx,
                    cy=cy,
                    area_t=float(area_t),
                    image_ar=image_ar,
                    target_ar=target_ar,
                    cfg=cfg,
                    source="grid",
                    scale_idx=str(s_idx),
                    must_keep=False,
                    priority=3.0,
                )

    # (B) Object-centered template crops around the estimated main subject.
    subj_area = max(cfg.a_min, min(cfg.a_max, ws * hs))
    for i, m in enumerate(cfg.object_template_scales):
        area_t = max(cfg.a_min, min(cfg.a_max, subj_area * (m ** 2)))
        add_center_area_box(
            out=all_cands,
            cx=cx_subj,
            cy=cy_subj,
            area_t=area_t,
            image_ar=image_ar,
            target_ar=target_ar,
            cfg=cfg,
            source="object_template",
            scale_idx=f"obj{i}",
            must_keep=False,
            priority=6.0,
        )

    # (C) Saliency-guided jitter around subject centroid to fill grid gaps.
    dxs = [0.0]
    dys = [0.0]
    for frac in cfg.jitter_fracs:
        if frac <= 0:
            continue
        dxs.extend([-frac * ws, frac * ws])
        dys.extend([-frac * hs, frac * hs])
    dxs = sorted(set(round(v, 6) for v in dxs))
    dys = sorted(set(round(v, 6) for v in dys))

    for j_idx, area_t in enumerate(cfg.jitter_scales):
        if not (cfg.a_min <= area_t <= cfg.a_max):
            continue
        for dx in dxs:
            for dy in dys:
                add_center_area_box(
                    out=all_cands,
                    cx=cx_subj + dx,
                    cy=cy_subj + dy,
                    area_t=float(area_t),
                    image_ar=image_ar,
                    target_ar=target_ar,
                    cfg=cfg,
                    source="jitter",
                    scale_idx=f"jit{j_idx}",
                    must_keep=False,
                    priority=5.0,
                )

    # (D) Optional thirds/phi priors as sparse composition anchors.
    if cfg.use_phi_thirds:
        third = [1 / 3, 2 / 3]
        phi = [0.382, 0.618]
        centers = [(x, y) for x in third + phi for y in third + phi]
        for p_idx, area_t in enumerate(cfg.phi_scales):
            for c_idx, (cx, cy) in enumerate(centers):
                add_center_area_box(
                    out=all_cands,
                    cx=cx,
                    cy=cy,
                    area_t=float(area_t),
                    image_ar=image_ar,
                    target_ar=target_ar,
                    cfg=cfg,
                    source="phi_thirds",
                    scale_idx=f"phi{p_idx}_{c_idx}",
                    must_keep=False,
                    priority=4.0,
                )

    # (E) Deduplicate and apply two-stage NMS:
    # keep critical baselines first, then suppress non-keep candidates against them.
    all_cands = dedupe_by_rounded_box(all_cands, ndigits=6)
    must_keep = [c for c in all_cands if c.get("must_keep", False)]
    non_keep = [c for c in all_cands if not c.get("must_keep", False)]

    must_keep = nms_candidates(must_keep, iou_thr=0.98)
    non_keep = nms_candidates(non_keep, iou_thr=cfg.nms_iou, existing=must_keep)

    # (F) Respect candidate cap while preserving baselines:
    # must-keep are never dropped by diversity sampling.
    if len(must_keep) >= cfg.max_candidates_per_ar:
        final_cands = sorted(
            must_keep,
            key=lambda c: (c.get("priority", 0.0), c.get("area_ratio", 0.0)),
            reverse=True,
        )[: cfg.max_candidates_per_ar]
    else:
        rem = cfg.max_candidates_per_ar - len(must_keep)
        sampled_non_keep = diversity_sample(non_keep, k=rem)
        final_cands = must_keep + sampled_non_keep

    # (G) Final deterministic ordering + AR-specific candidate ID assignment.
    final_cands = sorted(
        final_cands,
        key=lambda c: (
            1 if c.get("must_keep", False) else 0,
            c.get("priority", 0.0),
            c.get("area_ratio", 0.0),
        ),
        reverse=True,
    )

    token = ar_token(target_ar_text)
    for i, c in enumerate(final_cands, start=1):
        c["candidate_id"] = f"{token}_g{cfg.grid_m}x{cfg.grid_n}_s{c['scale_idx']}_i{i:04d}"
    return final_cands


def load_jsonl_map(path: Path, value_key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img_id = str(d.get("image_id", ""))
            if not img_id:
                continue
            out[img_id] = d.get(value_key, [])
    return out


def config_hash(cfg: CandidateGenConfig) -> str:
    payload = asdict(cfg)
    payload["impl_version"] = "candidate_gen_v1_8"
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def build_output_record(
    row: pd.Series,
    c2_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    cfg: CandidateGenConfig,
    cfg_hash: str,
) -> Dict[str, Any]:
    img_id = str(row["image_id"])
    width = int(row["width"])
    height = int(row["height"])
    image_ar = float(width) / float(max(1, height))
    tags = ensure_jsonable_tags(row.get("tags"))

    c2_seg = c2_map.get(img_id, [])
    c3_pose = c3_map.get(img_id, [])
    subj = resolve_subject_prior(
        width=width,
        height=height,
        c2_seg=c2_seg,
        c3_pose=c3_pose,
        cfg=cfg,
    )

    cx, cy = float(subj["centroid"][0]), float(subj["centroid"][1])
    ws, hs = float(subj["size"][0]), float(subj["size"][1])
    copy_hint = has_copy_space_hint(tags)

    candidates_by_ar: Dict[str, List[Dict[str, Any]]] = {}
    per_ar_count: Dict[str, int] = {}
    total_candidates = 0
    for ar_text in cfg.ar_list:
        ar_t = parse_ar(ar_text)
        cands = generate_candidates_for_ar(
            target_ar=ar_t,
            target_ar_text=ar_text,
            image_ar=image_ar,
            subject_centroid=(cx, cy),
            subject_size=(ws, hs),
            has_copy_hint=copy_hint,
            cfg=cfg,
        )
        candidates_by_ar[ar_text] = cands
        per_ar_count[ar_text] = len(cands)
        total_candidates += len(cands)

    return {
        "image_id": img_id,
        "width": width,
        "height": height,
        "image_ar": round(image_ar, 6),
        "tags": tags,
        "subject_prior": subj,
        "candidate_gen_hash": cfg_hash,
        "candidates_by_ar": candidates_by_ar,
        "stats": {
            "num_candidates_total": int(total_candidates),
            "num_candidates_by_ar": per_ar_count,
        },
    }


def _safe_percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_candidates_jsonl(
    output_jsonl: Path,
    viz_sample_images: int = 16,
) -> Dict[str, Any]:
    rows = 0
    total_candidates_list: List[int] = []
    image_ar_list: List[float] = []
    centroid_x_list: List[float] = []
    centroid_y_list: List[float] = []

    hash_counter: Counter[str] = Counter()
    subject_source_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    has_people_count = 0

    per_ar_counts: Dict[str, List[int]] = defaultdict(list)
    source_by_ar: Dict[str, Counter[str]] = defaultdict(Counter)
    sample_records: List[Dict[str, Any]] = []

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows += 1

            if len(sample_records) < viz_sample_images:
                sample_records.append(rec)

            hash_counter[str(rec.get("candidate_gen_hash", ""))] += 1
            image_ar_list.append(float(rec.get("image_ar", 0.0)))

            stats = rec.get("stats", {})
            total_candidates_list.append(int(stats.get("num_candidates_total", 0)))

            subj = rec.get("subject_prior", {})
            subject_source_counter[str(subj.get("source", "unknown"))] += 1
            if bool(subj.get("has_people", False)):
                has_people_count += 1
            centroid = subj.get("centroid", [0.5, 0.5])
            if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
                centroid_x_list.append(float(centroid[0]))
                centroid_y_list.append(float(centroid[1]))

            cands_by_ar = rec.get("candidates_by_ar", {})
            for ar_text, cands in cands_by_ar.items():
                cands = cands or []
                per_ar_counts[ar_text].append(len(cands))
                for c in cands:
                    src = str(c.get("source", "unknown"))
                    source_counter[src] += 1
                    source_by_ar[ar_text][src] += 1

    per_ar_summary: Dict[str, Dict[str, float]] = {}
    for ar_text in sorted(per_ar_counts.keys()):
        vals = per_ar_counts[ar_text]
        per_ar_summary[ar_text] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "min": float(min(vals)) if vals else 0.0,
            "max": float(max(vals)) if vals else 0.0,
            "p50": _safe_percentile(vals, 50),
            "p90": _safe_percentile(vals, 90),
            "total": float(sum(vals)),
        }

    overview = {
        "num_images": int(rows),
        "candidate_gen_hash_counts": dict(hash_counter),
        "avg_total_candidates": float(np.mean(total_candidates_list)) if total_candidates_list else 0.0,
        "min_total_candidates": float(min(total_candidates_list)) if total_candidates_list else 0.0,
        "max_total_candidates": float(max(total_candidates_list)) if total_candidates_list else 0.0,
        "p50_total_candidates": _safe_percentile(total_candidates_list, 50),
        "p90_total_candidates": _safe_percentile(total_candidates_list, 90),
        "avg_image_ar": float(np.mean(image_ar_list)) if image_ar_list else 0.0,
        "subject_source_counts": dict(subject_source_counter),
        "has_people_rate": float(has_people_count / rows) if rows > 0 else 0.0,
        "source_counts_overall": dict(source_counter),
        "per_ar_summary": per_ar_summary,
    }

    df_ar = pd.DataFrame(
        [
            {
                "ar": ar_text,
                "mean": per_ar_summary[ar_text]["mean"],
                "min": per_ar_summary[ar_text]["min"],
                "max": per_ar_summary[ar_text]["max"],
                "p50": per_ar_summary[ar_text]["p50"],
                "p90": per_ar_summary[ar_text]["p90"],
                "total": per_ar_summary[ar_text]["total"],
            }
            for ar_text in sorted(per_ar_summary.keys())
        ]
    )

    df_source = pd.DataFrame(
        [{"source": k, "count": int(v)} for k, v in source_counter.items()]
    ).sort_values("count", ascending=False)

    df_subject_source = pd.DataFrame(
        [{"subject_source": k, "count": int(v)} for k, v in subject_source_counter.items()]
    ).sort_values("count", ascending=False)

    return {
        "overview": overview,
        "df_ar": df_ar,
        "df_source": df_source,
        "df_subject_source": df_subject_source,
        "source_by_ar": {k: dict(v) for k, v in source_by_ar.items()},
        "sample_records": sample_records,
        "centroid_x": centroid_x_list,
        "centroid_y": centroid_y_list,
        "total_candidates": total_candidates_list,
    }


def save_overview_files(
    summary: Dict[str, Any],
    overview_json: Path,
    overview_ar_csv: Path,
    overview_source_csv: Path,
    overview_subject_source_csv: Path,
) -> None:
    overview_json.parent.mkdir(parents=True, exist_ok=True)
    overview_ar_csv.parent.mkdir(parents=True, exist_ok=True)
    overview_source_csv.parent.mkdir(parents=True, exist_ok=True)
    overview_subject_source_csv.parent.mkdir(parents=True, exist_ok=True)

    with overview_json.open("w", encoding="utf-8") as f:
        json.dump(summary["overview"], f, ensure_ascii=False, indent=2)

    summary["df_ar"].to_csv(overview_ar_csv, index=False)
    summary["df_source"].to_csv(overview_source_csv, index=False)
    summary["df_subject_source"].to_csv(overview_subject_source_csv, index=False)


def save_candidate_visualizations(
    summary: Dict[str, Any],
    viz_dir: Path,
    viz_ar: str = "1:1",
    viz_max_images: int = 16,
    viz_max_boxes: int = 80,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        print(f"[CandidateGen] Visualization skipped (matplotlib unavailable): {e}")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: histogram of total candidates per image.
    totals = summary["total_candidates"]
    if totals:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(totals, bins=30, color="#3b82f6", edgecolor="black", alpha=0.8)
        ax.set_title("Total Candidates per Image")
        ax.set_xlabel("num_candidates_total")
        ax.set_ylabel("num_images")
        fig.tight_layout()
        fig.savefig(viz_dir / "hist_total_candidates.png", dpi=150)
        plt.close(fig)

    # Plot 2: average candidates by AR.
    df_ar = summary["df_ar"]
    if not df_ar.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_ar["ar"], df_ar["mean"], color="#10b981", edgecolor="black")
        ax.set_title("Average Candidates by AR")
        ax.set_xlabel("AR")
        ax.set_ylabel("mean candidates")
        fig.tight_layout()
        fig.savefig(viz_dir / "bar_avg_candidates_by_ar.png", dpi=150)
        plt.close(fig)

    # Plot 3: source composition by AR (top-6 + other).
    source_by_ar = summary["source_by_ar"]
    all_sources = Counter()
    for src_map in source_by_ar.values():
        all_sources.update(src_map)
    top_sources = [s for s, _ in all_sources.most_common(6)]

    ar_keys = sorted(source_by_ar.keys())
    if ar_keys and top_sources:
        stack_data = {s: [] for s in top_sources}
        stack_data["other"] = []
        for ar in ar_keys:
            src_map = source_by_ar[ar]
            total = float(sum(src_map.values()))
            used = 0.0
            for s in top_sources:
                v = float(src_map.get(s, 0))
                used += v
                stack_data[s].append(v / total if total > 0 else 0.0)
            stack_data["other"].append((total - used) / total if total > 0 else 0.0)

        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(ar_keys), dtype=np.float64)
        for s, vals in stack_data.items():
            vals_np = np.asarray(vals, dtype=np.float64)
            ax.bar(ar_keys, vals_np, bottom=bottom, label=s)
            bottom += vals_np
        ax.set_title("Candidate Source Mix by AR")
        ax.set_xlabel("AR")
        ax.set_ylabel("ratio")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(viz_dir / "stacked_source_mix_by_ar.png", dpi=150)
        plt.close(fig)

    # Plot 4: subject centroid scatter.
    cx = summary["centroid_x"]
    cy = summary["centroid_y"]
    if cx and cy:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(cx, cy, s=8, alpha=0.4, color="#ef4444")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Subject Centroid Distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("y (top->bottom)")
        fig.tight_layout()
        fig.savefig(viz_dir / "scatter_subject_centroids.png", dpi=150)
        plt.close(fig)

    # Plot 5: sample candidate boxes (normalized canvas, no image decoding).
    '''
    •red: must_keep=True 후보 박스 (baseline 계열 강제 포함 후보)
    •blue: 일반 후보 박스 (must_keep=False)
    •green: subject_prior.bbox_norm_xyxy (C2/C3 기반 주 피사체 prior 박스)
    '''
    samples = summary["sample_records"][:viz_max_images]
    if samples:
        cols = 4
        rows = int(math.ceil(len(samples) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes_arr = np.atleast_1d(axes).reshape(rows, cols)
        for idx in range(rows * cols):
            ax = axes_arr[idx // cols, idx % cols]
            if idx >= len(samples):
                ax.axis("off")
                continue

            rec = samples[idx]
            cands_by_ar = rec.get("candidates_by_ar", {})
            cands = cands_by_ar.get(viz_ar, [])
            if not cands and cands_by_ar:
                first_ar = sorted(cands_by_ar.keys())[0]
                cands = cands_by_ar[first_ar]

            # Draw a normalized image frame.
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.2))

            # Draw candidates: must_keep highlighted.
            for c in cands[:viz_max_boxes]:
                x1, y1, x2, y2 = c["bbox_norm_xyxy"]
                w = x2 - x1
                h = y2 - y1
                if c.get("must_keep", False):
                    color = "#ef4444"
                    alpha = 0.45
                    lw = 1.2
                else:
                    color = "#2563eb"
                    alpha = 0.08
                    lw = 0.7
                ax.add_patch(Rectangle((x1, y1), w, h, fill=False, edgecolor=color, alpha=alpha, linewidth=lw))

            # Draw subject prior bbox.
            subj = rec.get("subject_prior", {})
            sb = subj.get("bbox_norm_xyxy")
            if isinstance(sb, list) and len(sb) == 4:
                x1, y1, x2, y2 = sb
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="#16a34a",
                        linewidth=1.5,
                    )
                )

            image_id = str(rec.get("image_id", "unknown"))
            ax.set_title(f"{image_id[:24]}... ({len(cands)} boxes)", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(viz_dir / "sample_candidate_boxes.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR-conditioned candidate generator (Phase B1)")
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--feats_c2_jsonl", required=True)
    p.add_argument("--feats_c3_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--ar_list", nargs="+", default=list(DEFAULT_AR_LIST))
    p.add_argument("--grid_m", type=int, default=12)
    p.add_argument("--grid_n", type=int, default=12)
    p.add_argument("--scale_set", nargs="+", type=float, default=list(DEFAULT_SCALE_SET))
    p.add_argument("--a_min", type=float, default=0.15)
    p.add_argument("--a_max", type=float, default=0.95)
    p.add_argument("--ar_tol", type=float, default=0.02)
    p.add_argument("--max_candidates_per_ar", type=int, default=240)
    p.add_argument("--nms_iou", type=float, default=0.90)
    p.add_argument("--eps_ar_full", type=float, default=0.01)
    p.add_argument("--jitter_fracs", nargs="+", type=float, default=list(DEFAULT_JITTER_FRACS))
    p.add_argument("--jitter_scales", nargs="+", type=float, default=list(DEFAULT_JITTER_SCALES))
    p.add_argument("--use_phi_thirds", type=int, default=1, help="1=use, 0=disable")
    p.add_argument("--phi_scales", nargs="+", type=float, default=list(DEFAULT_PHI_SCALES))
    p.add_argument("--max_images", type=int, default=0, help="0 means all")

    # Post-processing outputs: overview tables + visualization images.
    p.add_argument("--save_overview", type=int, default=1, help="1=save overview files")
    p.add_argument("--save_visualization", type=int, default=1, help="1=save visualization images")
    p.add_argument("--overview_json", default="", help="overview json path")
    p.add_argument("--overview_ar_csv", default="", help="per-AR overview csv path")
    p.add_argument("--overview_source_csv", default="", help="source-count overview csv path")
    p.add_argument("--overview_subject_source_csv", default="", help="subject-source overview csv path")
    p.add_argument("--viz_dir", default="", help="visualization output directory")
    p.add_argument("--viz_ar", default="1:1", help="AR to draw in sample box visualization")
    p.add_argument("--viz_max_images", type=int, default=16, help="num sample images for box visualization")
    p.add_argument("--viz_max_boxes", type=int, default=80, help="max boxes per sample image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CandidateGenConfig(
        ar_list=tuple(args.ar_list),
        grid_m=int(args.grid_m),
        grid_n=int(args.grid_n),
        scale_set=tuple(float(x) for x in args.scale_set),
        a_min=float(args.a_min),
        a_max=float(args.a_max),
        ar_tol=float(args.ar_tol),
        max_candidates_per_ar=int(args.max_candidates_per_ar),
        nms_iou=float(args.nms_iou),
        eps_ar_full=float(args.eps_ar_full),
        jitter_fracs=tuple(float(x) for x in args.jitter_fracs),
        jitter_scales=tuple(float(x) for x in args.jitter_scales),
        use_phi_thirds=bool(args.use_phi_thirds),
        phi_scales=tuple(float(x) for x in args.phi_scales),
    )
    cfg_hash = config_hash(cfg)

    input_parquet = Path(args.input_parquet)
    c2_path = Path(args.feats_c2_jsonl)
    c3_path = Path(args.feats_c3_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[CandidateGen] Loading inputs...")
    df = pd.read_parquet(input_parquet)
    needed = {"image_id", "width", "height"}
    if not needed.issubset(set(df.columns)):
        missing = sorted(needed - set(df.columns))
        raise ValueError(f"Input parquet missing required columns: {missing}")

    c2_map = load_jsonl_map(c2_path, value_key="c2_seg")
    c3_map = load_jsonl_map(c3_path, value_key="c3_pose")
    print(
        f"[CandidateGen] rows={len(df)} c2={len(c2_map)} c3={len(c3_map)} cfg_hash={cfg_hash}"
    )

    if args.max_images > 0:
        df = df.head(int(args.max_images))
        print(
            f"[CandidateGen] max_images={args.max_images} -> processing rows={len(df)}"
        )

    num_written = 0
    avg_total = 0.0
    with output_path.open("w", encoding="utf-8") as out_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CandidateGen"):
            rec = build_output_record(
                row=row,
                c2_map=c2_map,
                c3_map=c3_map,
                cfg=cfg,
                cfg_hash=cfg_hash,
            )
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1
            avg_total += rec["stats"]["num_candidates_total"]

    avg_total = avg_total / max(1, num_written)
    print(
        f"[CandidateGen] Done. written={num_written} avg_total_candidates={avg_total:.2f}"
    )
    print(f"[CandidateGen] output={output_path}")

    save_overview = bool(args.save_overview)
    save_visualization = bool(args.save_visualization)
    if save_overview or save_visualization:
        print("[CandidateGen] Building overview from output jsonl...")
        summary = summarize_candidates_jsonl(
            output_jsonl=output_path,
            viz_sample_images=max(1, int(args.viz_max_images)),
        )

        if save_overview:
            overview_json = (
                Path(args.overview_json)
                if args.overview_json
                else output_path.with_name(f"{output_path.stem}_overview.json")
            )
            overview_ar_csv = (
                Path(args.overview_ar_csv)
                if args.overview_ar_csv
                else output_path.with_name(f"{output_path.stem}_overview_by_ar.csv")
            )
            overview_source_csv = (
                Path(args.overview_source_csv)
                if args.overview_source_csv
                else output_path.with_name(f"{output_path.stem}_overview_by_source.csv")
            )
            overview_subject_source_csv = (
                Path(args.overview_subject_source_csv)
                if args.overview_subject_source_csv
                else output_path.with_name(f"{output_path.stem}_overview_by_subject_source.csv")
            )
            save_overview_files(
                summary=summary,
                overview_json=overview_json,
                overview_ar_csv=overview_ar_csv,
                overview_source_csv=overview_source_csv,
                overview_subject_source_csv=overview_subject_source_csv,
            )
            print(
                "[CandidateGen] Overview saved:"
                f" {overview_json}, {overview_ar_csv}, {overview_source_csv}, {overview_subject_source_csv}"
            )

        if save_visualization:
            viz_dir = (
                Path(args.viz_dir)
                if args.viz_dir
                else output_path.parent / "visualizations" / f"{output_path.stem}_viz"
            )
            save_candidate_visualizations(
                summary=summary,
                viz_dir=viz_dir,
                viz_ar=str(args.viz_ar),
                viz_max_images=max(1, int(args.viz_max_images)),
                viz_max_boxes=max(1, int(args.viz_max_boxes)),
            )
            print(f"[CandidateGen] Visualization saved under: {viz_dir}")


if __name__ == "__main__":
    main()
