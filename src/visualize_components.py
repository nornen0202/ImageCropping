"""
Component visualization utility for C2/C3/C4/C5/C6 features.

Examples
--------
Local (separate jsonl):
  python3 src/visualize_components.py \
      --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
      --c2_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c2.jsonl \
      --c3_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c3_v2_strict_enriched.jsonl \
      --c4_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c4.jsonl \
      --c5_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c5.jsonl \
      --c6_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c6.jsonl \
      --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
      --out_dir data/SSTK/10K_local/artifacts/precompute/visualizations/components_v2_local \
      --num_samples 100 \
      --draw_combined 1

Local (merged jsonl):
  python3 src/visualize_components.py \
      --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
      --merged_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl \
      --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
      --out_dir data/SSTK/10K_local/artifacts/precompute/visualizations/components_v2_local \
      --num_samples 100 \
      --draw_combined 1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

COCO_SKELETON_EDGES: List[Tuple[int, int]] = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]
IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def as_bool_flag(v: int) -> bool:
    return int(v) != 0


def decode_rle(rle_string: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode 1-indexed run-length encoding to binary mask."""
    h, w = shape
    if not rle_string:
        return np.zeros((h, w), dtype=bool)

    try:
        runs = np.asarray([int(x) for x in str(rle_string).split()], dtype=np.int64)
    except Exception:
        return np.zeros((h, w), dtype=bool)

    if runs.size == 0 or (runs.size % 2) != 0:
        return np.zeros((h, w), dtype=bool)

    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths

    flat = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        lo_i = int(max(0, min(lo, flat.size)))
        hi_i = int(max(0, min(hi, flat.size)))
        if hi_i > lo_i:
            flat[lo_i:hi_i] = 1
    return flat.reshape((h, w)).astype(bool)


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> Optional[str]:
    if pd.isna(bucket):
        bucket = ""
    candidates = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, str(bucket), tar_name) if bucket else None,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def find_image_in_dir(image_dir: str, image_id: str) -> Optional[Path]:
    if not image_dir:
        return None
    root = Path(image_dir)
    for ext in IMAGE_EXTS:
        p = root / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def parse_image_ids_arg(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        if raw is None:
            continue
        toks = [tok.strip() for tok in str(raw).split(",")]
        for tok in toks:
            if tok:
                out.append(tok)
    return out


def parse_image_ids_file(path: str) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image_ids_file not found: {path}")
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.extend([tok.strip() for tok in s.split(",") if tok.strip()])
    return out


def unique_keep_order(ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def color_for_index(i: int) -> Tuple[int, int, int]:
    """Deterministic pseudo-random BGR color by index."""
    rng = np.random.default_rng(1729 + int(i) * 37)
    c = rng.integers(40, 255, size=3, dtype=np.uint8).tolist()
    return int(c[0]), int(c[1]), int(c[2])


def to_int_pt(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def clip_pt(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xx = int(round(max(0.0, min(float(w - 1), float(x)))))
    yy = int(round(max(0.0, min(float(h - 1), float(y)))))
    return xx, yy


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def draw_c2_layer(
    image_bgr: np.ndarray,
    c2_list: Sequence[Dict[str, Any]],
    max_masks: int = 8,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    overlay = np.zeros_like(out)

    for idx, obj in enumerate(c2_list[: max(1, int(max_masks))]):
        box = obj.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        score = float(obj.get("score", 0.0))
        class_id = obj.get("class_id", -1)
        mask_rle = obj.get("mask_rle", "")
        mask = decode_rle(mask_rle, (h, w)) if isinstance(mask_rle, str) else np.zeros((h, w), dtype=bool)

        color = color_for_index(idx)
        overlay[mask] = color

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, y1 = clip_pt(x1, y1, w, h)
        x2, y2 = clip_pt(x2, y2, w, h)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"c{class_id} s={score:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    mask_any = np.any(overlay != 0, axis=-1)
    if np.any(mask_any):
        blended = cv2.addWeighted(out, 0.55, overlay, 0.45, 0.0)
        out[mask_any] = blended[mask_any]
    return out


def draw_c3_layer(
    image_bgr: np.ndarray,
    c3_list: Sequence[Dict[str, Any]],
    kp_thr: float = 0.05,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]

    for pid, obj in enumerate(c3_list):
        box = obj.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        score = float(obj.get("score", 0.0))

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, y1 = clip_pt(x1, y1, w, h)
        x2, y2 = clip_pt(x2, y2, w, h)

        bbox_color = (0, 220, 0)
        kp_color = (0, 0, 255)
        face_color = (255, 255, 0)
        gaze_color = (255, 180, 30)

        cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, 2)
        cv2.putText(
            out,
            f"p{pid} s={score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            bbox_color,
            1,
            cv2.LINE_AA,
        )

        keypoints = obj.get("keypoints", [])
        valid: Dict[int, Tuple[float, float]] = {}
        for kidx, kp in enumerate(keypoints):
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                continue
            xk, yk, sk = float(kp[0]), float(kp[1]), float(kp[2])
            if sk < float(kp_thr):
                continue
            px, py = clip_pt(xk, yk, w, h)
            valid[kidx] = (float(px), float(py))
            cv2.circle(out, (px, py), 2, kp_color, -1)

        for a, b in COCO_SKELETON_EDGES:
            if a in valid and b in valid:
                pa = to_int_pt(*valid[a])
                pb = to_int_pt(*valid[b])
                cv2.line(out, pa, pb, (80, 120, 255), 1, cv2.LINE_AA)

        face = obj.get("face")
        if isinstance(face, dict):
            fbox = face.get("bbox")
            if isinstance(fbox, (list, tuple)) and len(fbox) == 4:
                fx1, fy1, fx2, fy2 = [int(round(float(v))) for v in fbox]
                fx1, fy1 = clip_pt(fx1, fy1, w, h)
                fx2, fy2 = clip_pt(fx2, fy2, w, h)
                cv2.rectangle(out, (fx1, fy1), (fx2, fy2), face_color, 2)

            lms = face.get("landmarks5", [])
            if isinstance(lms, list):
                for lm in lms:
                    if not isinstance(lm, (list, tuple)) or len(lm) < 2:
                        continue
                    lx, ly = clip_pt(float(lm[0]), float(lm[1]), w, h)
                    cv2.circle(out, (lx, ly), 2, face_color, -1)

        hp = obj.get("headpose_gaze")
        if isinstance(hp, dict):
            yaw = float(hp.get("yaw_proxy", 0.0))
            pitch = float(hp.get("pitch_proxy", 0.0))
            roll = float(hp.get("roll_deg", 0.0))
            gdir = str(hp.get("gaze_dir", "unknown"))

            txt = f"g={gdir} yaw={yaw:+.2f} p={pitch:+.2f} r={roll:+.1f}"
            cv2.putText(
                out,
                txt,
                (x1, min(h - 5, y2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                gaze_color,
                1,
                cv2.LINE_AA,
            )

            # Draw a simple gaze direction arrow from nose (or bbox center fallback).
            origin: Tuple[int, int]
            if isinstance(face, dict):
                lms = face.get("landmarks5", [])
                if isinstance(lms, list) and len(lms) > 0 and isinstance(lms[0], (list, tuple)) and len(lms[0]) >= 2:
                    origin = clip_pt(float(lms[0][0]), float(lms[0][1]), w, h)
                else:
                    origin = to_int_pt((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            else:
                origin = to_int_pt((x1 + x2) * 0.5, (y1 + y2) * 0.5)

            length = int(max(20, min(80, 0.6 * max(1, x2 - x1))))
            if gdir == "right":
                target = clip_pt(origin[0] + length, origin[1], w, h)
            elif gdir == "left":
                target = clip_pt(origin[0] - length, origin[1], w, h)
            else:
                target = clip_pt(origin[0] + int(round(yaw * length)), origin[1], w, h)
            cv2.arrowedLine(out, origin, target, gaze_color, 2, cv2.LINE_AA, tipLength=0.25)

    return out


def _box_from_quad(quad: Any) -> Optional[List[float]]:
    if not isinstance(quad, (list, tuple)) or len(quad) < 4:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for p in quad:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _to_abs_box_xyxy(box_any: Any, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(box_any, (list, tuple)) or len(box_any) != 4:
        return None
    vals = [float(v) for v in box_any]
    if max(abs(v) for v in vals) <= 1.5:
        x1, y1, x2, y2 = vals
        x1 *= float(w)
        y1 *= float(h)
        x2 *= float(w)
        y2 *= float(h)
    else:
        x1, y1, x2, y2 = vals
    p1 = clip_pt(x1, y1, w, h)
    p2 = clip_pt(x2, y2, w, h)
    return p1[0], p1[1], p2[0], p2[1]


def to_c4_boxes(c4_value: Any) -> List[Dict[str, Any]]:
    if isinstance(c4_value, dict):
        boxes = c4_value.get("boxes", [])
        return boxes if isinstance(boxes, list) else []
    if isinstance(c4_value, list):
        return c4_value
    return []


def draw_c4_layer(image_bgr: np.ndarray, c4_value: Any) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    boxes = to_c4_boxes(c4_value)
    color = (0, 255, 255)
    kept = 0
    for idx, item in enumerate(boxes):
        if not isinstance(item, dict):
            continue
        box_abs: Optional[Tuple[int, int, int, int]] = None
        for key in ("box_xyxy", "box_norm_xyxy", "box"):
            if key not in item:
                continue
            src = item.get(key)
            if key == "box":
                src = _box_from_quad(src)
            box_abs = _to_abs_box_xyxy(src, w=w, h=h)
            if box_abs is not None:
                break
        if box_abs is None:
            continue
        x1, y1, x2, y2 = box_abs
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        score = float(item.get("score", 0.0))
        cv2.putText(
            out,
            f"t{idx} s={score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        kept += 1
    cv2.putText(
        out,
        f"ocr_boxes={kept}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def to_c6_people(c6_value: Any) -> List[Dict[str, Any]]:
    if isinstance(c6_value, dict):
        people = c6_value.get("people", [])
        return people if isinstance(people, list) else []
    if isinstance(c6_value, list):
        return c6_value
    return []


def draw_c6_layer(image_bgr: np.ndarray, c6_value: Any) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    people = to_c6_people(c6_value)
    head_color = (0, 165, 255)
    gaze_color = (255, 80, 80)
    for idx, p in enumerate(people):
        if not isinstance(p, dict):
            continue
        head_box = p.get("head_bbox_norm_xyxy")
        if isinstance(head_box, (list, tuple)) and len(head_box) == 4:
            box_abs = _to_abs_box_xyxy([clamp01(v) for v in head_box], w=w, h=h)
            if box_abs is not None:
                x1, y1, x2, y2 = box_abs
                cv2.rectangle(out, (x1, y1), (x2, y2), head_color, 2)
        else:
            x1 = y1 = x2 = y2 = None

        gaze_xy = p.get("gaze_target_norm_xy")
        origin: Optional[Tuple[int, int]] = None
        if all(v is not None for v in (x1, y1, x2, y2)):
            origin = to_int_pt((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        if isinstance(gaze_xy, (list, tuple)) and len(gaze_xy) == 2 and origin is not None:
            gx = clamp01(float(gaze_xy[0])) * float(w)
            gy = clamp01(float(gaze_xy[1])) * float(h)
            target = clip_pt(gx, gy, w, h)
            cv2.arrowedLine(out, origin, target, gaze_color, 2, cv2.LINE_AA, tipLength=0.22)

        gdir = str(p.get("gaze_dir", "unknown"))
        conf = float(p.get("conf", 0.0))
        pid = int(p.get("person_index", idx))
        txt = f"p{pid} g={gdir} c={conf:.2f}"
        text_y = 18 + idx * 16
        cv2.putText(out, txt, (8, min(h - 6, text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, head_color, 1, cv2.LINE_AA)
    return out


def _denorm_line(line_norm_xyxy: Sequence[float], w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1n, yn1, x2n, yn2 = [float(v) for v in line_norm_xyxy]
    p1 = clip_pt(x1n * w, yn1 * h, w, h)
    p2 = clip_pt(x2n * w, yn2 * h, w, h)
    return p1, p2


def to_c5_dict(c5_value: Any) -> Dict[str, Any]:
    if isinstance(c5_value, dict):
        return c5_value
    if isinstance(c5_value, list) and c5_value and isinstance(c5_value[0], dict):
        return c5_value[0]
    return {}


def draw_c5_layer(image_bgr: np.ndarray, c5_value: Any) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]

    c5 = to_c5_dict(c5_value)
    if not c5:
        return out

    hr = c5.get("horizon_roll", {}) if isinstance(c5.get("horizon_roll"), dict) else {}
    sym = c5.get("symmetry", {}) if isinstance(c5.get("symmetry"), dict) else {}

    horizon_color = (255, 0, 255)
    sym_color = (50, 200, 255)

    line_norm = hr.get("line_norm_xyxy")
    if isinstance(line_norm, (list, tuple)) and len(line_norm) == 4:
        p1, p2 = _denorm_line(line_norm, w, h)
        cv2.line(out, p1, p2, horizon_color, 2, cv2.LINE_AA)
    else:
        y_norm = hr.get("horizon_y_norm")
        if y_norm is not None:
            yy = int(round(float(y_norm) * h))
            yy = max(0, min(h - 1, yy))
            cv2.line(out, (0, yy), (w - 1, yy), horizon_color, 2, cv2.LINE_AA)

    conf = hr.get("conf")
    roll = hr.get("roll_deg")
    theta = hr.get("theta_deg")
    info = f"roll={roll} theta={theta} conf={conf}"
    cv2.putText(out, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, horizon_color, 1, cv2.LINE_AA)

    sym_score = sym.get("score")
    if sym_score is not None:
        cv2.putText(
            out,
            f"sym={float(sym_score):.3f}",
            (8, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            sym_color,
            1,
            cv2.LINE_AA,
        )

    # Reference center line for symmetry reading.
    cx = w // 2
    cv2.line(out, (cx, 0), (cx, h - 1), sym_color, 1, cv2.LINE_AA)
    return out


def load_component_map(path: str, key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img_id = str(d.get("image_id", ""))
            if not img_id:
                continue
            out[img_id] = d.get(key, [] if key in ("c2_seg", "c3_pose") else {})
    return out


def load_feature_maps(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    c2_map: Dict[str, Any] = {}
    c3_map: Dict[str, Any] = {}
    c4_map: Dict[str, Any] = {}
    c5_map: Dict[str, Any] = {}
    c6_map: Dict[str, Any] = {}

    # 1) merged map (if provided)
    if args.merged_jsonl and Path(args.merged_jsonl).exists():
        with Path(args.merged_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                img_id = str(d.get("image_id", ""))
                if not img_id:
                    continue
                if "c2_seg" in d:
                    c2_map[img_id] = d.get("c2_seg", [])
                if "c3_pose" in d:
                    c3_map[img_id] = d.get("c3_pose", [])
                if "c4_ocr" in d:
                    c4_map[img_id] = d.get("c4_ocr", [])
                if "c5_geom" in d:
                    c5_map[img_id] = d.get("c5_geom", {})
                if "c6_gaze" in d:
                    c6_map[img_id] = d.get("c6_gaze", {})

    # 2) dedicated component maps override merged for explicit control.
    if args.c2_jsonl:
        c2_map.update(load_component_map(args.c2_jsonl, "c2_seg"))
    if args.c3_jsonl:
        c3_map.update(load_component_map(args.c3_jsonl, "c3_pose"))
    if args.c4_jsonl:
        c4_map.update(load_component_map(args.c4_jsonl, "c4_ocr"))
    if args.c5_jsonl:
        c5_map.update(load_component_map(args.c5_jsonl, "c5_geom"))
    if args.c6_jsonl:
        c6_map.update(load_component_map(args.c6_jsonl, "c6_gaze"))

    return c2_map, c3_map, c4_map, c5_map, c6_map


def should_use_image(
    image_id: str,
    c2_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    c4_map: Dict[str, Any],
    c5_map: Dict[str, Any],
    c6_map: Dict[str, Any],
    draw_c2: bool,
    draw_c3: bool,
    draw_c4: bool,
    draw_c5: bool,
    draw_c6: bool,
) -> bool:
    if draw_c2 and len(c2_map.get(image_id, [])) > 0:
        return True
    if draw_c3 and len(c3_map.get(image_id, [])) > 0:
        return True
    if draw_c4 and len(to_c4_boxes(c4_map.get(image_id, []))) > 0:
        return True
    if draw_c5:
        c5 = to_c5_dict(c5_map.get(image_id, {}))
        if bool(c5):
            return True
    if draw_c6 and len(to_c6_people(c6_map.get(image_id, {}))) > 0:
        return True
    return False


def save_visualizations_for_image(
    image_id: str,
    raw_bgr: np.ndarray,
    c2_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    c4_map: Dict[str, Any],
    c5_map: Dict[str, Any],
    c6_map: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    c2 = c2_map.get(image_id, [])
    c3 = c3_map.get(image_id, [])
    c4 = c4_map.get(image_id, [])
    c5 = c5_map.get(image_id, {})
    c6 = c6_map.get(image_id, {})

    if as_bool_flag(args.draw_c2):
        vis = draw_c2_layer(raw_bgr, c2, max_masks=args.max_masks_per_image)
        cv2.imwrite(str(Path(args.out_dir) / "c2_seg" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_c3):
        vis = draw_c3_layer(raw_bgr, c3, kp_thr=args.kp_score_thr)
        cv2.imwrite(str(Path(args.out_dir) / "c3_pose" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_c4):
        vis = draw_c4_layer(raw_bgr, c4)
        cv2.imwrite(str(Path(args.out_dir) / "c4_ocr" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_c5):
        vis = draw_c5_layer(raw_bgr, c5)
        cv2.imwrite(str(Path(args.out_dir) / "c5_geom" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_c6):
        vis = draw_c6_layer(raw_bgr, c6)
        cv2.imwrite(str(Path(args.out_dir) / "c6_gaze" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_combined):
        vis = raw_bgr.copy()
        if as_bool_flag(args.draw_c2):
            vis = draw_c2_layer(vis, c2, max_masks=args.max_masks_per_image)
        if as_bool_flag(args.draw_c3):
            vis = draw_c3_layer(vis, c3, kp_thr=args.kp_score_thr)
        if as_bool_flag(args.draw_c4):
            vis = draw_c4_layer(vis, c4)
        if as_bool_flag(args.draw_c5):
            vis = draw_c5_layer(vis, c5)
        if as_bool_flag(args.draw_c6):
            vis = draw_c6_layer(vis, c6)
        cv2.imwrite(str(Path(args.out_dir) / "combined_all" / f"{image_id}.jpg"), vis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize C2/C3/C4/C5/C6 features")
    parser.add_argument("--parquet", type=str, default="data/SSTK/10K_local/filtered_sstk_100.parquet")
    parser.add_argument("--merged_jsonl", type=str, default="")
    parser.add_argument("--c2_jsonl", type=str, default="")
    parser.add_argument("--c3_jsonl", type=str, default="")
    parser.add_argument("--c4_jsonl", type=str, default="")
    parser.add_argument("--c5_jsonl", type=str, default="")
    parser.add_argument("--c6_jsonl", type=str, default="")
    parser.add_argument("--tar_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="", help="optional local curated image dir (<image_id>.<ext>)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/SSTK/10K_local/artifacts/precompute/visualizations/components_v2_local",
    )
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument(
        "--image_ids",
        nargs="*",
        default=[],
        help="explicit image ids (space/comma separated). if set, this list is prioritized over auto sampling",
    )
    parser.add_argument("--image_ids_file", type=str, default="", help="text file with image_id entries (1 per line)")

    parser.add_argument("--draw_c2", type=int, default=1, help="1=save c2 layer, 0=skip")
    parser.add_argument("--draw_c3", type=int, default=1, help="1=save c3 layer, 0=skip")
    parser.add_argument("--draw_c4", type=int, default=1, help="1=save c4 layer, 0=skip")
    parser.add_argument("--draw_c5", type=int, default=1, help="1=save c5 layer, 0=skip")
    parser.add_argument("--draw_c6", type=int, default=1, help="1=save c6 layer, 0=skip")
    parser.add_argument("--draw_combined", type=int, default=0, help="1=save combined C2+C3+C4+C5+C6 overlay")

    parser.add_argument("--max_masks_per_image", type=int, default=8)
    parser.add_argument("--kp_score_thr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "original").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_c2):
        (out_dir / "c2_seg").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_c3):
        (out_dir / "c3_pose").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_c4):
        (out_dir / "c4_ocr").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_c5):
        (out_dir / "c5_geom").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_c6):
        (out_dir / "c6_gaze").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_combined):
        (out_dir / "combined_all").mkdir(parents=True, exist_ok=True)

    print("Loading parquet...")
    df = pd.read_parquet(args.parquet)
    if "bucket" in df.columns:
        mapping = df.set_index("image_id")[["tar_name", "bucket"]].to_dict("index")
    else:
        mapping = df.set_index("image_id")[["tar_name"]].to_dict("index")

    print("Loading features...")
    c2_map, c3_map, c4_map, c5_map, c6_map = load_feature_maps(args)

    if not c2_map and not c3_map and not c4_map and not c5_map and not c6_map:
        raise RuntimeError(
            "No features loaded. Provide --merged_jsonl or at least one of --c2_jsonl/--c3_jsonl/--c4_jsonl/--c5_jsonl/--c6_jsonl"
        )

    print("Selecting samples...")
    ordered_ids = [str(x) for x in df["image_id"].tolist()]
    ordered_id_set = set(ordered_ids)

    explicit_ids = unique_keep_order(parse_image_ids_arg(args.image_ids) + parse_image_ids_file(args.image_ids_file))
    selected_ids: List[str] = []
    missing_explicit: List[str] = []
    if explicit_ids:
        for image_id in explicit_ids:
            if image_id in ordered_id_set:
                selected_ids.append(image_id)
            else:
                missing_explicit.append(image_id)
        if missing_explicit:
            print(f"[warn] {len(missing_explicit)} explicit image_ids not found in parquet")
        if args.num_samples > 0:
            selected_ids = selected_ids[: int(args.num_samples)]
    else:
        for image_id in ordered_ids:
            if should_use_image(
                image_id=image_id,
                c2_map=c2_map,
                c3_map=c3_map,
                c4_map=c4_map,
                c5_map=c5_map,
                c6_map=c6_map,
                draw_c2=as_bool_flag(args.draw_c2),
                draw_c3=as_bool_flag(args.draw_c3),
                draw_c4=as_bool_flag(args.draw_c4),
                draw_c5=as_bool_flag(args.draw_c5),
                draw_c6=as_bool_flag(args.draw_c6),
            ):
                selected_ids.append(image_id)
                if args.num_samples > 0 and len(selected_ids) >= args.num_samples:
                    break

    print(f"Processing {len(selected_ids)} sample(s)...")

    tar_to_ids: Dict[str, List[str]] = defaultdict(list)
    rendered_ids: List[str] = []
    missing_ids: List[str] = []

    for image_id in selected_ids:
        orig_path = out_dir / "original" / f"{image_id}.jpg"
        if orig_path.exists():
            raw = cv2.imread(str(orig_path))
            if raw is not None:
                save_visualizations_for_image(image_id, raw, c2_map, c3_map, c4_map, c5_map, c6_map, args)
                rendered_ids.append(image_id)
                continue

        local_img_path = find_image_in_dir(args.image_dir, image_id)
        if local_img_path is not None:
            raw = cv2.imread(str(local_img_path))
            if raw is not None:
                cv2.imwrite(str(orig_path), raw)
                save_visualizations_for_image(image_id, raw, c2_map, c3_map, c4_map, c5_map, c6_map, args)
                rendered_ids.append(image_id)
                continue

        if image_id not in mapping:
            missing_ids.append(image_id)
            continue
        info = mapping[image_id]
        tar_name = info.get("tar_name", "")
        bucket = info.get("bucket", "")
        tar_path = resolve_tar_path(args.tar_dir, bucket, tar_name)
        if tar_path:
            tar_to_ids[tar_path].append(image_id)

    for tar_path, ids in tar_to_ids.items():
        print(f"Opening {tar_path} for {len(ids)} image(s)...")
        id_set = set(ids)

        try:
            with tarfile.open(tar_path, "r|") as tf:
                for member in tf:
                    name = os.path.basename(member.name)
                    base, ext = os.path.splitext(name)
                    if ext.lower() not in IMAGE_EXTS:
                        continue
                    if base not in id_set:
                        continue

                    f = tf.extractfile(member)
                    if f is None:
                        id_set.remove(base)
                        if not id_set:
                            break
                        continue

                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    raw_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_dir / "original" / f"{base}.jpg"), raw_bgr)

                    save_visualizations_for_image(base, raw_bgr, c2_map, c3_map, c4_map, c5_map, c6_map, args)
                    rendered_ids.append(base)

                    id_set.remove(base)
                    if not id_set:
                        break
            if id_set:
                missing_ids.extend(sorted(id_set))
        except Exception as e:
            print(f"Error reading tar {tar_path}: {e}")
            missing_ids.extend(sorted(id_set))

    summary = {
        "inputs": {
            "parquet": str(args.parquet),
            "merged_jsonl": str(args.merged_jsonl),
            "c2_jsonl": str(args.c2_jsonl),
            "c3_jsonl": str(args.c3_jsonl),
            "c4_jsonl": str(args.c4_jsonl),
            "c5_jsonl": str(args.c5_jsonl),
            "c6_jsonl": str(args.c6_jsonl),
            "tar_dir": str(args.tar_dir),
            "image_dir": str(args.image_dir),
        },
        "selected_count": len(selected_ids),
        "rendered_count": len(rendered_ids),
        "missing_count": len(missing_ids),
        "selected_ids": selected_ids,
        "missing_ids": missing_ids,
    }
    with (out_dir / "viz_overview.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Visualization complete.")
    print(f"[done] rendered={len(rendered_ids)} missing={len(missing_ids)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
