"""
Component visualization utility for C2/C3/C5 features.

Examples
--------
Local (separate jsonl):
  python3 src/visualize_components.py \
      --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
      --c2_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c2.jsonl \
      --c3_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c3_v2_strict_enriched.jsonl \
      --c5_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c5.jsonl \
      --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
      --out_dir data/SSTK/10K_local/artifacts/visualizations/components_v2_local \
      --num_samples 100 \
      --draw_combined 1

Local (merged jsonl):
  python3 src/visualize_components.py \
      --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
      --merged_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl \
      --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
      --out_dir data/SSTK/10K_local/artifacts/visualizations/components_v2_local \
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


def load_feature_maps(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    c2_map: Dict[str, Any] = {}
    c3_map: Dict[str, Any] = {}
    c5_map: Dict[str, Any] = {}

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
                if "c5_geom" in d:
                    c5_map[img_id] = d.get("c5_geom", {})

    # 2) dedicated component maps override merged for explicit control.
    if args.c2_jsonl:
        c2_map.update(load_component_map(args.c2_jsonl, "c2_seg"))
    if args.c3_jsonl:
        c3_map.update(load_component_map(args.c3_jsonl, "c3_pose"))
    if args.c5_jsonl:
        c5_map.update(load_component_map(args.c5_jsonl, "c5_geom"))

    return c2_map, c3_map, c5_map


def should_use_image(
    image_id: str,
    c2_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    c5_map: Dict[str, Any],
    draw_c2: bool,
    draw_c3: bool,
    draw_c5: bool,
) -> bool:
    if draw_c2 and len(c2_map.get(image_id, [])) > 0:
        return True
    if draw_c3 and len(c3_map.get(image_id, [])) > 0:
        return True
    if draw_c5:
        c5 = to_c5_dict(c5_map.get(image_id, {}))
        if bool(c5):
            return True
    return False


def save_visualizations_for_image(
    image_id: str,
    raw_bgr: np.ndarray,
    c2_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    c5_map: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    c2 = c2_map.get(image_id, [])
    c3 = c3_map.get(image_id, [])
    c5 = c5_map.get(image_id, {})

    if as_bool_flag(args.draw_c2):
        vis = draw_c2_layer(raw_bgr, c2, max_masks=args.max_masks_per_image)
        cv2.imwrite(str(Path(args.out_dir) / "c2_seg" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_c3):
        vis = draw_c3_layer(raw_bgr, c3, kp_thr=args.kp_score_thr)
        cv2.imwrite(str(Path(args.out_dir) / "c3_pose" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_c5):
        vis = draw_c5_layer(raw_bgr, c5)
        cv2.imwrite(str(Path(args.out_dir) / "c5_geom" / f"{image_id}.jpg"), vis)

    if as_bool_flag(args.draw_combined):
        vis = raw_bgr.copy()
        if as_bool_flag(args.draw_c2):
            vis = draw_c2_layer(vis, c2, max_masks=args.max_masks_per_image)
        if as_bool_flag(args.draw_c3):
            vis = draw_c3_layer(vis, c3, kp_thr=args.kp_score_thr)
        if as_bool_flag(args.draw_c5):
            vis = draw_c5_layer(vis, c5)
        cv2.imwrite(str(Path(args.out_dir) / "combined_all" / f"{image_id}.jpg"), vis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize C2/C3/C5 features")
    parser.add_argument("--parquet", type=str, default="data/SSTK/10K_local/filtered_sstk_100.parquet")
    parser.add_argument("--merged_jsonl", type=str, default="")
    parser.add_argument("--c2_jsonl", type=str, default="")
    parser.add_argument("--c3_jsonl", type=str, default="")
    parser.add_argument("--c5_jsonl", type=str, default="")
    parser.add_argument("--tar_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/SSTK/10K_local/artifacts/visualizations/components_v2_local")
    parser.add_argument("--num_samples", type=int, default=50)

    parser.add_argument("--draw_c2", type=int, default=1, help="1=save c2 layer, 0=skip")
    parser.add_argument("--draw_c3", type=int, default=1, help="1=save c3 layer, 0=skip")
    parser.add_argument("--draw_c5", type=int, default=1, help="1=save c5 layer, 0=skip")
    parser.add_argument("--draw_combined", type=int, default=0, help="1=save combined C2+C3+C5 overlay")

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
    if as_bool_flag(args.draw_c5):
        (out_dir / "c5_geom").mkdir(parents=True, exist_ok=True)
    if as_bool_flag(args.draw_combined):
        (out_dir / "combined_all").mkdir(parents=True, exist_ok=True)

    print("Loading parquet...")
    df = pd.read_parquet(args.parquet)
    if "bucket" in df.columns:
        mapping = df.set_index("image_id")[["tar_name", "bucket"]].to_dict("index")
    else:
        mapping = df.set_index("image_id")[["tar_name"]].to_dict("index")

    print("Loading features...")
    c2_map, c3_map, c5_map = load_feature_maps(args)

    if not c2_map and not c3_map and not c5_map:
        raise RuntimeError(
            "No features loaded. Provide --merged_jsonl or at least one of --c2_jsonl/--c3_jsonl/--c5_jsonl"
        )

    print("Selecting samples...")
    ordered_ids = [str(x) for x in df["image_id"].tolist()]
    selected_ids: List[str] = []
    for image_id in ordered_ids:
        if should_use_image(
            image_id=image_id,
            c2_map=c2_map,
            c3_map=c3_map,
            c5_map=c5_map,
            draw_c2=as_bool_flag(args.draw_c2),
            draw_c3=as_bool_flag(args.draw_c3),
            draw_c5=as_bool_flag(args.draw_c5),
        ):
            selected_ids.append(image_id)
            if args.num_samples > 0 and len(selected_ids) >= args.num_samples:
                break

    print(f"Processing {len(selected_ids)} sample(s)...")

    tar_to_ids: Dict[str, List[str]] = defaultdict(list)

    for image_id in selected_ids:
        orig_path = out_dir / "original" / f"{image_id}.jpg"
        if orig_path.exists():
            raw = cv2.imread(str(orig_path))
            if raw is not None:
                save_visualizations_for_image(image_id, raw, c2_map, c3_map, c5_map, args)
            continue

        if image_id not in mapping:
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
                    if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
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

                    save_visualizations_for_image(base, raw_bgr, c2_map, c3_map, c5_map, args)

                    id_set.remove(base)
                    if not id_set:
                        break
        except Exception as e:
            print(f"Error reading tar {tar_path}: {e}")

    print("Visualization complete.")


if __name__ == "__main__":
    main()
