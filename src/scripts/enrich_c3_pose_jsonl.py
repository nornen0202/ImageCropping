#!/usr/bin/env python3
"""
Enrich existing C3 pose jsonl with face/gaze proxy fields derived from keypoints.

Input records are expected to include:
  {"image_id": ..., "c3_pose": [{"bbox": ..., "keypoints": ...}, ...]}

Output keeps all original fields and appends for each pose item:
  - face
  - headpose_gaze
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
import math
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> Optional[str]:
    if not tar_name:
        return None
    candidates = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, str(bucket), tar_name) if bucket else None,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def build_local_image_index(image_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not image_dir or not os.path.isdir(image_dir):
        return out
    try:
        for name in os.listdir(image_dir):
            p = os.path.join(image_dir, name)
            if not os.path.isfile(p):
                continue
            stem, ext = os.path.splitext(name)
            if not stem:
                continue
            if ext.lower() not in DEFAULT_IMAGE_EXTS:
                continue
            if stem not in out:
                out[stem] = p
    except Exception:
        return {}
    return out


def build_actual_size_map(
    df: pd.DataFrame,
    tar_dir: str,
    image_dir: str = "",
    cache_json: Optional[Path] = None,
    use_cache_if_available: bool = True,
) -> Dict[str, Tuple[int, int]]:
    if cache_json is not None and use_cache_if_available and cache_json.exists():
        out: Dict[str, Tuple[int, int]] = {}
        with cache_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for k, v in payload.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[str(k)] = (int(v[0]), int(v[1]))
        if out:
            return out

    out: Dict[str, Tuple[int, int]] = {}
    remaining = {str(x) for x in df["image_id"].astype(str).tolist()}

    local_index = build_local_image_index(image_dir)
    if local_index:
        for img_id in list(remaining):
            p = local_index.get(img_id)
            if p is None:
                continue
            try:
                with Image.open(p) as im:
                    out[img_id] = (int(im.width), int(im.height))
                remaining.remove(img_id)
            except Exception:
                continue

    if not remaining:
        if cache_json is not None:
            cache_json.parent.mkdir(parents=True, exist_ok=True)
            with cache_json.open("w", encoding="utf-8") as f:
                json.dump({k: [int(v[0]), int(v[1])] for k, v in out.items()}, f, ensure_ascii=False)
        return out

    if not str(tar_dir).strip():
        return out
    if "tar_name" not in df.columns:
        return out

    by_tar: Dict[str, list[Tuple[str, str]]] = defaultdict(list)
    has_bucket = "bucket" in df.columns
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        if img_id not in remaining:
            continue
        tar_name = str(row["tar_name"])
        bucket = str(row["bucket"]) if has_bucket else ""
        by_tar[tar_name].append((img_id, bucket))

    for tar_name, items in tqdm(by_tar.items(), total=len(by_tar), desc="size-map"):
        bucket = items[0][1] if items else ""
        tar_path = resolve_tar_path(tar_dir=tar_dir, bucket=bucket, tar_name=tar_name)
        if tar_path is None:
            continue

        wanted = {f"{img_id}.jpg": img_id for img_id, _ in items}
        wanted.update({f"{img_id}.jpeg": img_id for img_id, _ in items})
        wanted.update({f"{img_id}.png": img_id for img_id, _ in items})
        wanted.update({f"{img_id}.webp": img_id for img_id, _ in items})

        try:
            with tarfile.open(tar_path, "r") as tf:
                members = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}
                for fname, img_id in wanted.items():
                    m = members.get(fname)
                    if m is None:
                        continue
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    try:
                        im = Image.open(io.BytesIO(data))
                        out[img_id] = (int(im.width), int(im.height))
                    except Exception:
                        continue
        except Exception:
            continue

    if cache_json is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        with cache_json.open("w", encoding="utf-8") as f:
            json.dump({k: [int(v[0]), int(v[1])] for k, v in out.items()}, f, ensure_ascii=False)

    return out


def _load_hw_map(
    parquet_path: Path,
    *,
    use_actual_image_size: bool,
    tar_dir: str,
    image_dir: str,
    actual_size_cache_json: Optional[Path],
) -> Dict[str, Tuple[int, int]]:
    cols = ["image_id", "width", "height"]
    probe = pd.read_parquet(parquet_path, columns=None)
    if "tar_name" in probe.columns:
        cols.append("tar_name")
    if "bucket" in probe.columns:
        cols.append("bucket")
    df = pd.read_parquet(parquet_path, columns=cols)

    out: Dict[str, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        out[str(row["image_id"])] = (int(row["width"]), int(row["height"]))

    if not use_actual_image_size:
        return out
    if not tar_dir.strip() and not image_dir.strip():
        raise ValueError("use_actual_image_size=1 requires --tar_dir or --image_dir")
    if not image_dir.strip() and "tar_name" not in df.columns:
        raise ValueError("use_actual_image_size=1 with tar mode requires parquet columns: tar_name")

    actual_map = build_actual_size_map(
        df=df,
        tar_dir=tar_dir,
        image_dir=image_dir,
        cache_json=actual_size_cache_json,
        use_cache_if_available=True,
    )
    for img_id, wh in actual_map.items():
        if wh[0] > 0 and wh[1] > 0:
            out[img_id] = (int(wh[0]), int(wh[1]))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich C3 pose jsonl with face/gaze proxy fields")
    p.add_argument("--input_c3_jsonl", required=True)
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--use_actual_image_size", type=int, default=1, help="1=use TAR actual image size")
    p.add_argument("--tar_dir", default="", help="required when use_actual_image_size=1")
    p.add_argument("--image_dir", default="", help="optional local curated image dir (<image_id>.<ext>)")
    p.add_argument("--actual_size_cache_json", default="", help="optional image_id->(w,h) cache json")
    return p.parse_args()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _clamp_box_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = _clamp(x1, 0.0, float(max(0, w - 1)))
    y1 = _clamp(y1, 0.0, float(max(0, h - 1)))
    x2 = _clamp(x2, 0.0, float(w))
    y2 = _clamp(y2, 0.0, float(h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _get_kp(kps: list[list[float]], idx: int, min_conf: float = 0.05) -> Optional[Tuple[float, float, float]]:
    if idx < 0 or idx >= len(kps):
        return None
    if not isinstance(kps[idx], (list, tuple)) or len(kps[idx]) < 3:
        return None
    x, y, c = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(c):
        return None
    if c < min_conf:
        return None
    return (x, y, c)


def _default_face_landmarks(bbox: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return [
        [x1 + 0.50 * w, y1 + 0.55 * h, 0.0],
        [x1 + 0.35 * w, y1 + 0.38 * h, 0.0],
        [x1 + 0.65 * w, y1 + 0.38 * h, 0.0],
        [x1 + 0.20 * w, y1 + 0.48 * h, 0.0],
        [x1 + 0.80 * w, y1 + 0.48 * h, 0.0],
    ]


def derive_face_from_keypoints(keypoints: list[list[float]], image_w: int, image_h: int) -> Optional[Dict[str, object]]:
    kp_nose = 0
    kp_left_eye = 1
    kp_right_eye = 2
    kp_left_ear = 3
    kp_right_ear = 4

    face_points: list[Tuple[float, float, float]] = []
    for idx in (kp_nose, kp_left_eye, kp_right_eye, kp_left_ear, kp_right_ear):
        kp = _get_kp(keypoints, idx)
        if kp is not None:
            face_points.append(kp)

    if len(face_points) < 2:
        return None

    xs = [p[0] for p in face_points]
    ys = [p[1] for p in face_points]
    cs = [p[2] for p in face_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    base = max(4.0, max(max_x - min_x, max_y - min_y))
    pad_x = 0.60 * base
    pad_y_top = 0.75 * base
    pad_y_bot = 0.65 * base
    bbox = _clamp_box_xyxy(
        (min_x - pad_x, min_y - pad_y_top, max_x + pad_x, max_y + pad_y_bot),
        image_w,
        image_h,
    )

    landmarks = _default_face_landmarks(bbox)
    idx_to_lm = {kp_nose: 0, kp_left_eye: 1, kp_right_eye: 2, kp_left_ear: 3, kp_right_ear: 4}
    for kp_idx, lm_idx in idx_to_lm.items():
        kp = _get_kp(keypoints, kp_idx, min_conf=0.0)
        if kp is not None:
            landmarks[lm_idx] = [float(kp[0]), float(kp[1]), float(kp[2])]

    conf = float(np.clip(np.mean(cs), 0.0, 1.0)) if cs else 0.0
    x1, y1, x2, y2 = bbox
    bbox_norm = [
        round(x1 / max(1, image_w), 6),
        round(y1 / max(1, image_h), 6),
        round(x2 / max(1, image_w), 6),
        round(y2 / max(1, image_h), 6),
    ]
    return {
        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "bbox_norm": bbox_norm,
        "landmarks5": [[round(v, 3) for v in lm] for lm in landmarks],
        "score": round(conf, 4),
        "source": "pose_kp_proxy",
    }


def estimate_headpose_gaze_proxy(face: Optional[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "yaw_proxy": 0.0,
        "pitch_proxy": 0.0,
        "roll_deg": 0.0,
        "gaze_dir": "unknown",
        "conf": 0.0,
        "source": "pose_kp_proxy",
    }
    if face is None:
        return out
    lm = face.get("landmarks5", [])
    if not isinstance(lm, list) or len(lm) < 3:
        return out

    nose = lm[0]
    left_eye = lm[1]
    right_eye = lm[2]
    nx, ny, nc = float(nose[0]), float(nose[1]), float(nose[2])
    lx, ly, lc = float(left_eye[0]), float(left_eye[1]), float(left_eye[2])
    rx, ry, rc = float(right_eye[0]), float(right_eye[1]), float(right_eye[2])
    if lc <= 0.0 or rc <= 0.0:
        return out

    eye_mid_x = 0.5 * (lx + rx)
    eye_mid_y = 0.5 * (ly + ry)
    inter_eye = max(1.0, math.hypot(rx - lx, ry - ly))
    yaw_proxy = (nx - eye_mid_x) / max(1e-6, 0.5 * inter_eye)
    pitch_proxy = (ny - eye_mid_y) / max(1e-6, inter_eye)
    roll_deg = math.degrees(math.atan2((ry - ly), (rx - lx + 1e-6)))
    yaw_proxy = float(np.clip(yaw_proxy, -1.0, 1.0))
    pitch_proxy = float(np.clip(pitch_proxy, -1.0, 1.0))

    if yaw_proxy > 0.12:
        gaze_dir = "right"
    elif yaw_proxy < -0.12:
        gaze_dir = "left"
    else:
        gaze_dir = "center"
    conf = float(np.clip((nc + lc + rc) / 3.0, 0.0, 1.0))
    out.update(
        {
            "yaw_proxy": round(yaw_proxy, 4),
            "pitch_proxy": round(pitch_proxy, 4),
            "roll_deg": round(float(roll_deg), 3),
            "gaze_dir": gaze_dir,
            "conf": round(conf, 4),
        }
    )
    return out


def _is_gazelle_headpose_gaze(headpose_gaze: object) -> bool:
    if not isinstance(headpose_gaze, dict):
        return False
    source = str(headpose_gaze.get("source", "") or "").strip().lower()
    return source in {"c6_gaze", "gaze_lle_gazelle", "gazelle", "gazelle_gaze_lle"}


def _normalize_c6_headpose_gaze(person: Dict[str, Any]) -> Dict[str, object]:
    return {
        "yaw_proxy": round(float(person.get("yaw_proxy", 0.0) or 0.0), 4),
        "pitch_proxy": round(float(person.get("pitch_proxy", 0.0) or 0.0), 4),
        "roll_deg": round(float(person.get("roll_deg", 0.0) or 0.0), 3),
        "gaze_dir": str(person.get("gaze_dir", "unknown")),
        "conf": round(float(person.get("conf", 0.0) or 0.0), 4),
        "source": str(person.get("source", "c6_gaze") or "c6_gaze"),
    }


def _build_c6_person_map(rec: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    c6 = rec.get("c6_gaze")
    if not isinstance(c6, dict):
        return {}
    people = c6.get("people")
    if not isinstance(people, list):
        return {}
    out: Dict[int, Dict[str, object]] = {}
    for p in people:
        if not isinstance(p, dict):
            continue
        try:
            person_index = int(p.get("person_index", -1))
        except Exception:
            person_index = -1
        if person_index < 0:
            continue
        out[person_index] = _normalize_c6_headpose_gaze(p)
    return out


def enrich_pose_item_from_keypoints(
    pose_item: Dict[str, object],
    image_w: int,
    image_h: int,
    c6_headpose_gaze: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    out = dict(pose_item)
    keypoints = pose_item.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) == 0:
        out.setdefault("face", None)
        if c6_headpose_gaze is not None:
            out["headpose_gaze"] = dict(c6_headpose_gaze)
        else:
            out.setdefault("headpose_gaze", estimate_headpose_gaze_proxy(None))
        return out
    face = derive_face_from_keypoints(keypoints=keypoints, image_w=image_w, image_h=image_h)
    out["face"] = face
    if c6_headpose_gaze is not None:
        out["headpose_gaze"] = dict(c6_headpose_gaze)
    else:
        existing_headpose = out.get("headpose_gaze")
        if _is_gazelle_headpose_gaze(existing_headpose):
            out["headpose_gaze"] = dict(existing_headpose)
        else:
            out["headpose_gaze"] = estimate_headpose_gaze_proxy(face)
    return out


def main() -> None:
    args = parse_args()
    print("[enrich_c3] running CPU-only JSONL post-process (no GPU inference).")

    c3_path = Path(args.input_c3_jsonl)
    pq_path = Path(args.input_parquet)
    out_path = Path(args.output_jsonl)

    hw_map = _load_hw_map(
        parquet_path=pq_path,
        use_actual_image_size=bool(int(args.use_actual_image_size)),
        tar_dir=str(args.tar_dir),
        image_dir=str(args.image_dir),
        actual_size_cache_json=Path(args.actual_size_cache_json) if args.actual_size_cache_json else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    enriched = 0
    missing_wh = 0

    with c3_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="enrich_c3"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            img_id = str(rec.get("image_id", ""))
            pose_list = rec.get("c3_pose", [])

            wh = hw_map.get(img_id)
            if wh is None:
                missing_wh += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            w, h = wh
            if isinstance(pose_list, list) and pose_list:
                c6_person_map = _build_c6_person_map(rec)
                new_pose = []
                for idx, item in enumerate(pose_list):
                    if isinstance(item, dict):
                        new_pose.append(
                            enrich_pose_item_from_keypoints(
                                item,
                                image_w=w,
                                image_h=h,
                                c6_headpose_gaze=c6_person_map.get(idx),
                            )
                        )
                    else:
                        new_pose.append(item)
                rec["c3_pose"] = new_pose
                enriched += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        "[done] total="
        f"{total} enriched_records={enriched} missing_hw={missing_wh} "
        f"use_actual_image_size={int(bool(int(args.use_actual_image_size)))} "
        f"output={out_path}"
    )


if __name__ == "__main__":
    main()
