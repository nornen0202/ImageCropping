"""
generate_candidates.py
AR-conditioned + FREE-form candidate generator (Phase B1).

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
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DEFAULT_AR_LIST = ("FREE", "1:1", "9:16", "16:9", "3:4", "4:3")
DEFAULT_SCALE_SET = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_JITTER_FRACS = (0.0, 0.03, 0.06)
DEFAULT_JITTER_SCALES = (0.45, 0.55, 0.65)
DEFAULT_PHI_SCALES = (0.35, 0.55, 0.75)
DEFAULT_OBJ_TEMPLATE_SCALES = (1.1, 1.25, 1.45)
DEFAULT_TEACHER_JITTER_SHIFT_FRACS = (0.03,)
DEFAULT_TEACHER_JITTER_SCALES = (0.92, 1.00, 1.08)
DEFAULT_FREE_AR_VALUES = (0.67, 0.75, 1.0, 1.33, 1.5)
DEFAULT_FREE_AR_LOG_JITTER = (0.0, -0.12, 0.12)
DEFAULT_FREE_JITTER_FRACS = (0.0, 0.04)
DEFAULT_FREE_JITTER_SCALES = (0.45, 0.65)
DEFAULT_FREE_SCALE_SET = (0.45, 0.65, 0.85)
DEFAULT_FREE_PHI_SCALES = (0.55,)
DEFAULT_FREE_TEACHER_AR_JITTER = (0.85, 1.0, 1.15)

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
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


_WORKER_C2_SEG_MAP: Optional[Dict[str, Any]] = None
_WORKER_C2_DET_MAP: Optional[Dict[str, Any]] = None
_WORKER_C3_MAP: Optional[Dict[str, Any]] = None
_WORKER_ROUTING_MAP: Optional[Dict[str, Any]] = None
_WORKER_C2_PRIMARY_IDX_MAP: Optional[Dict[str, Any]] = None
_WORKER_C2_UNION_BOX_MAP: Optional[Dict[str, Any]] = None
_WORKER_CFG: Optional["CandidateGenConfig"] = None
_WORKER_CFG_HASH: str = ""
_WORKER_ACTUAL_SIZE_MAP: Optional[Dict[str, Tuple[int, int]]] = None
_WORKER_STRICT_ACTUAL_SIZE: bool = True
_WORKER_TEACHER_PROPOSALS_MAP: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None


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
    teacher_nms_iou: float = 0.95
    teacher_max_seeds_per_teacher: int = 1
    teacher_prefer_expand: bool = True
    teacher_jitter_shift_fracs: Tuple[float, ...] = DEFAULT_TEACHER_JITTER_SHIFT_FRACS
    teacher_jitter_scales: Tuple[float, ...] = DEFAULT_TEACHER_JITTER_SCALES
    teacher_seed_priority: float = 6.5
    teacher_jitter_priority: float = 6.2
    # FREE-form generation (AR-unconditioned)
    free_ar_values: Tuple[float, ...] = DEFAULT_FREE_AR_VALUES
    free_ar_log_jitter: Tuple[float, ...] = DEFAULT_FREE_AR_LOG_JITTER
    free_grid_m: int = 4
    free_grid_n: int = 4
    free_scale_set: Tuple[float, ...] = DEFAULT_FREE_SCALE_SET
    free_jitter_fracs: Tuple[float, ...] = DEFAULT_FREE_JITTER_FRACS
    free_jitter_scales: Tuple[float, ...] = DEFAULT_FREE_JITTER_SCALES
    free_phi_scales: Tuple[float, ...] = DEFAULT_FREE_PHI_SCALES
    free_nms_iou: float = 0.85
    free_bucket_portrait_max: float = 0.90
    free_bucket_square_max: float = 1.10
    free_teacher_ar_jitter: Tuple[float, ...] = DEFAULT_FREE_TEACHER_AR_JITTER


def _init_build_worker(
    c2_seg_map: Dict[str, Any],
    c2_det_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    routing_map: Dict[str, Any],
    c2_primary_idx_map: Dict[str, Any],
    c2_union_box_map: Dict[str, Any],
    cfg: CandidateGenConfig,
    cfg_hash: str,
    actual_size_map: Optional[Dict[str, Tuple[int, int]]],
    strict_actual_size: bool,
    teacher_proposals_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> None:
    global _WORKER_C2_SEG_MAP
    global _WORKER_C2_DET_MAP
    global _WORKER_C3_MAP
    global _WORKER_ROUTING_MAP
    global _WORKER_C2_PRIMARY_IDX_MAP
    global _WORKER_C2_UNION_BOX_MAP
    global _WORKER_CFG
    global _WORKER_CFG_HASH
    global _WORKER_ACTUAL_SIZE_MAP
    global _WORKER_STRICT_ACTUAL_SIZE
    global _WORKER_TEACHER_PROPOSALS_MAP

    _WORKER_C2_SEG_MAP = c2_seg_map
    _WORKER_C2_DET_MAP = c2_det_map
    _WORKER_C3_MAP = c3_map
    _WORKER_ROUTING_MAP = routing_map
    _WORKER_C2_PRIMARY_IDX_MAP = c2_primary_idx_map
    _WORKER_C2_UNION_BOX_MAP = c2_union_box_map
    _WORKER_CFG = cfg
    _WORKER_CFG_HASH = cfg_hash
    _WORKER_ACTUAL_SIZE_MAP = actual_size_map
    _WORKER_STRICT_ACTUAL_SIZE = bool(strict_actual_size)
    _WORKER_TEACHER_PROPOSALS_MAP = teacher_proposals_map


def _build_output_record_worker(row: Dict[str, Any]) -> Dict[str, Any]:
    if (
        _WORKER_C2_SEG_MAP is None
        or _WORKER_C2_DET_MAP is None
        or _WORKER_C3_MAP is None
        or _WORKER_ROUTING_MAP is None
        or _WORKER_C2_PRIMARY_IDX_MAP is None
        or _WORKER_C2_UNION_BOX_MAP is None
        or _WORKER_CFG is None
    ):
        raise RuntimeError("candidate worker state is not initialized")
    return build_output_record(
        row=row,
        c2_seg_map=_WORKER_C2_SEG_MAP,
        c2_det_map=_WORKER_C2_DET_MAP,
        c3_map=_WORKER_C3_MAP,
        routing_map=_WORKER_ROUTING_MAP,
        c2_primary_idx_map=_WORKER_C2_PRIMARY_IDX_MAP,
        c2_union_box_map=_WORKER_C2_UNION_BOX_MAP,
        cfg=_WORKER_CFG,
        cfg_hash=_WORKER_CFG_HASH,
        actual_size_map=_WORKER_ACTUAL_SIZE_MAP,
        strict_actual_size=_WORKER_STRICT_ACTUAL_SIZE,
        teacher_proposals_map=_WORKER_TEACHER_PROPOSALS_MAP,
    )


def resolve_num_workers(requested_workers: int, num_items: int) -> int:
    if num_items <= 1:
        return 1
    req = int(requested_workers)
    if req == 1:
        return 1
    if req <= 0:
        cpu = int(os.cpu_count() or 1)
        # Keep one core for IO/OS and avoid excessive process fan-out by default.
        req = max(1, min(16, cpu - 1))
        if num_items < 256:
            req = 1
    return max(1, min(req, num_items))


def resolve_mp_context(start_method: str) -> Optional[Any]:
    method = str(start_method).strip().lower()
    if method in {"", "auto"}:
        for cand in ("fork", "forkserver", "spawn"):
            try:
                return mp.get_context(cand)
            except Exception:
                continue
        return None
    try:
        return mp.get_context(method)
    except Exception as e:
        raise ValueError(f"invalid mp start method: {start_method}") from e


def iter_build_output_records(
    *,
    rows: Sequence[Dict[str, Any]],
    num_workers: int,
    mp_chunksize: int,
    mp_start_method: str,
    c2_seg_map: Dict[str, Any],
    c2_det_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    routing_map: Dict[str, Any],
    c2_primary_idx_map: Dict[str, Any],
    c2_union_box_map: Dict[str, Any],
    cfg: CandidateGenConfig,
    cfg_hash: str,
    actual_size_map: Optional[Dict[str, Tuple[int, int]]],
    strict_actual_size: bool,
    teacher_proposals_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> Iterable[Dict[str, Any]]:
    if int(num_workers) <= 1:
        for row in rows:
            yield build_output_record(
                row=row,
                c2_seg_map=c2_seg_map,
                c2_det_map=c2_det_map,
                c3_map=c3_map,
                routing_map=routing_map,
                c2_primary_idx_map=c2_primary_idx_map,
                c2_union_box_map=c2_union_box_map,
                cfg=cfg,
                cfg_hash=cfg_hash,
                actual_size_map=actual_size_map,
                strict_actual_size=bool(strict_actual_size),
                teacher_proposals_map=teacher_proposals_map,
            )
        return

    chunk = max(1, int(mp_chunksize))
    mp_ctx = resolve_mp_context(mp_start_method)
    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        mp_context=mp_ctx,
        initializer=_init_build_worker,
        initargs=(
            c2_seg_map,
            c2_det_map,
            c3_map,
            routing_map,
            c2_primary_idx_map,
            c2_union_box_map,
            cfg,
            cfg_hash,
            actual_size_map,
            bool(strict_actual_size),
            teacher_proposals_map,
        ),
    ) as executor:
        for rec in executor.map(_build_output_record_worker, rows, chunksize=chunk):
            yield rec


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> Optional[str]:
    if not tar_name:
        return None
    cand = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, str(bucket), tar_name) if bucket else None,
    ]
    for p in cand:
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
    """
    Build image_id -> (actual_width, actual_height) by probing images in TARs.

    The filtered parquet width/height are original source sizes and often differ
    from TAR-resized images; C2/C3 pixel boxes are on TAR image scale, so
    candidate normalization must use these actual TAR sizes.
    """
    target_ids = {str(x) for x in df["image_id"].astype(str).tolist()}

    # Cache is used as a seed only. Partial cache must NOT short-circuit.
    cache_all: Dict[str, Tuple[int, int]] = {}
    if cache_json is not None and cache_json.exists():
        try:
            with cache_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    if not isinstance(v, (list, tuple)) or len(v) != 2:
                        continue
                    try:
                        w = int(v[0])
                        h = int(v[1])
                    except Exception:
                        continue
                    if w > 0 and h > 0:
                        cache_all[str(k)] = (w, h)
        except Exception:
            cache_all = {}

    out: Dict[str, Tuple[int, int]] = {}
    if use_cache_if_available and cache_all:
        for img_id in target_ids:
            wh = cache_all.get(img_id)
            if wh is not None and wh[0] > 0 and wh[1] > 0:
                out[img_id] = (int(wh[0]), int(wh[1]))

    remaining = set(target_ids - set(out.keys()))

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
            merged = dict(cache_all)
            merged.update(out)
            with cache_json.open("w", encoding="utf-8") as f:
                json.dump({k: [int(v[0]), int(v[1])] for k, v in merged.items()}, f, ensure_ascii=False)
        return out

    if not str(tar_dir).strip():
        return out
    if "tar_name" not in df.columns:
        raise ValueError("use_actual_image_size=1 with tar mode requires parquet column 'tar_name'")

    by_tar: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    has_bucket = "bucket" in df.columns
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        if img_id not in remaining:
            continue
        tar_name = str(row["tar_name"])
        bucket = str(row["bucket"]) if has_bucket else ""
        by_tar[tar_name].append((img_id, bucket))

    for tar_name, items in tqdm(by_tar.items(), total=len(by_tar), desc="size-map"):
        wanted_ids = {img_id for img_id, _ in items if img_id in remaining}
        if not wanted_ids:
            continue
        bucket = items[0][1] if items else ""
        tar_path = resolve_tar_path(tar_dir=tar_dir, bucket=bucket, tar_name=tar_name)
        if tar_path is None:
            continue
        try:
            with tarfile.open(tar_path, "r") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    base = os.path.basename(m.name)
                    stem, ext = os.path.splitext(base)
                    if stem not in wanted_ids:
                        continue
                    if ext.lower() not in DEFAULT_IMAGE_EXTS:
                        continue
                    if stem in out:
                        continue
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    try:
                        im = Image.open(io.BytesIO(data))
                        out[stem] = (int(im.width), int(im.height))
                    except Exception:
                        continue
        except Exception:
            continue
        remaining = set(target_ids - set(out.keys()))
        if not remaining:
            break

    if cache_json is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        merged = dict(cache_all)
        merged.update(out)
        with cache_json.open("w", encoding="utf-8") as f:
            json.dump({k: [int(v[0]), int(v[1])] for k, v in merged.items()}, f, ensure_ascii=False)

    return out


def parse_ar(ar_text: str) -> float:
    t = ar_text.strip()
    if ":" in t:
        a, b = t.split(":", 1)
        return float(a) / float(b)
    return float(t)


def is_free_ar_text(ar_text: str) -> bool:
    return str(ar_text).strip().upper() in {"FREE", "AR_FREE", "FREEFORM", "FREE_FORM"}


def parse_target_ar(ar_text: str) -> Optional[float]:
    if is_free_ar_text(ar_text):
        return None
    return parse_ar(ar_text)


def parse_ar_loose(ar_text: str) -> float:
    t = str(ar_text).strip().lower().replace("x", ":")
    return parse_ar(t)


def ar_token(ar_text: str) -> str:
    return ar_text.strip().replace(":", "x")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


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


def clip_box01(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_teacher_box(item: Any) -> Optional[List[float]]:
    box = None
    if isinstance(item, (list, tuple)) and len(item) == 4:
        box = [safe_float(v) for v in item]
    elif isinstance(item, dict):
        for k in (
            "bbox_norm_xyxy",
            "bbox_norm",
            "bbox",
            "box_norm_xyxy",
            "box_norm",
            "box",
        ):
            v = item.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                box = [safe_float(x) for x in v]
                break
    if box is None:
        return None
    box = clip_box01(box)
    if box_area(box) <= 1e-8:
        return None
    return [round(v, 6) for v in box]


def extract_teacher_score(item: Any) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    for k in ("teacher_score", "score", "conf", "confidence"):
        if k in item:
            return safe_float(item.get(k), 0.0)
    return None


def normalize_teacher_id(v: Any) -> str:
    t = str(v).strip().lower()
    if not t:
        return "unknown"
    return t.replace(" ", "_")


def parse_teacher_proposal_list(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, (list, tuple)):
        return out
    for item in payload:
        box = normalize_teacher_box(item)
        if box is None:
            continue
        out.append(
            {
                "bbox_norm_xyxy": box,
                "teacher_score": extract_teacher_score(item),
            }
        )
    return out


def parse_teacher_payload(payload: Any) -> Dict[str, Any]:
    out = {
        "free_form": [],
        "by_ar": {},
    }

    if isinstance(payload, (list, tuple)):
        out["free_form"] = parse_teacher_proposal_list(payload)
        return out

    if not isinstance(payload, dict):
        return out

    free_form = []
    for k in ("free_form", "freeform", "seed", "seeds", "proposals"):
        if k in payload:
            free_form.extend(parse_teacher_proposal_list(payload.get(k)))
    out["free_form"] = free_form

    by_ar_raw = (
        payload.get("by_ar")
        or payload.get("proposals_by_ar")
        or payload.get("target_ar")
        or payload.get("ar_specific")
        or {}
    )
    by_ar: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(by_ar_raw, dict):
        for ar_key, arr in by_ar_raw.items():
            items = parse_teacher_proposal_list(arr)
            if items:
                by_ar[str(ar_key)] = items
    out["by_ar"] = by_ar
    return out


def load_teacher_proposals_map(paths: Sequence[Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Parse teacher proposal jsonl(s).

    Canonical output:
    {
      image_id: {
        teacher_id: {
          "free_form": [{"bbox_norm_xyxy":[...], "teacher_score":...}, ...],
          "by_ar": {"1:1": [...], ...}
        }
      }
    }
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def _merge_teacher_payload(
        image_id: str,
        teacher_id: str,
        parsed: Dict[str, Any],
    ) -> None:
        if not parsed["free_form"] and not parsed["by_ar"]:
            return
        t_map = out[image_id].setdefault(teacher_id, {"free_form": [], "by_ar": {}})
        t_map["free_form"].extend(parsed["free_form"])
        for ar_text, arr in parsed["by_ar"].items():
            t_map["by_ar"].setdefault(ar_text, [])
            t_map["by_ar"][ar_text].extend(arr)

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                image_id = str(d.get("image_id", "")).strip()
                if not image_id:
                    continue

                # Format A: {"teacher_proposals": {"gaic": {...}, ...}}
                block = d.get("teacher_proposals") or d.get("proposals_by_teacher")
                if isinstance(block, dict):
                    for teacher_id, payload in block.items():
                        _merge_teacher_payload(
                            image_id=image_id,
                            teacher_id=normalize_teacher_id(teacher_id),
                            parsed=parse_teacher_payload(payload),
                        )
                    continue

                # Format B: {"teachers":[{"teacher_id":"gaic", ...}, ...]}
                teachers = d.get("teachers")
                if isinstance(teachers, list):
                    for t in teachers:
                        if not isinstance(t, dict):
                            continue
                        teacher_id = normalize_teacher_id(t.get("teacher_id", "unknown"))
                        _merge_teacher_payload(
                            image_id=image_id,
                            teacher_id=teacher_id,
                            parsed=parse_teacher_payload(t),
                        )
                    continue

                # Format C: single-teacher line.
                teacher_id = d.get("teacher_id")
                if teacher_id is not None:
                    _merge_teacher_payload(
                        image_id=image_id,
                        teacher_id=normalize_teacher_id(teacher_id),
                        parsed=parse_teacher_payload(d),
                    )
    return {k: v for k, v in out.items()}


def project_box_to_ar(
    box: Sequence[float],
    target_ar: float,
    prefer_expand: bool = True,
) -> List[float]:
    """
    Project a normalized box to target AR while keeping center as stable as possible.
    """
    x1, y1, x2, y2 = clip_box01(box)
    cx, cy = box_center([x1, y1, x2, y2])
    w, h = box_wh([x1, y1, x2, y2])
    w = max(w, 1e-6)
    h = max(h, 1e-6)
    ar = w / h
    ar_t = max(target_ar, 1e-6)

    if abs(ar - ar_t) <= 1e-6:
        return [x1, y1, x2, y2]

    if ar < ar_t:
        # box is too tall.
        if prefer_expand:
            w2 = min(h * ar_t, 1.0)
            h2 = w2 / ar_t
        else:
            h2 = min(w / ar_t, 1.0)
            w2 = h2 * ar_t
    else:
        # box is too wide.
        if prefer_expand:
            h2 = min(w / ar_t, 1.0)
            w2 = h2 * ar_t
        else:
            w2 = min(h * ar_t, 1.0)
            h2 = w2 / ar_t

    # Ensure inside [0,1] bounds while preserving AR.
    if w2 > 1.0:
        w2 = 1.0
        h2 = w2 / ar_t
    if h2 > 1.0:
        h2 = 1.0
        w2 = h2 * ar_t

    x1n = cx - 0.5 * w2
    x2n = cx + 0.5 * w2
    y1n = cy - 0.5 * h2
    y2n = cy + 0.5 * h2

    dx = 0.0
    if x1n < 0.0:
        dx = -x1n
    elif x2n > 1.0:
        dx = 1.0 - x2n
    x1n += dx
    x2n += dx

    dy = 0.0
    if y1n < 0.0:
        dy = -y1n
    elif y2n > 1.0:
        dy = 1.0 - y2n
    y1n += dy
    y2n += dy

    return clip_box01([x1n, y1n, x2n, y2n])


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


def free_ar_sample_values(cfg: CandidateGenConfig) -> List[float]:
    out: List[float] = []
    for base in cfg.free_ar_values:
        b = max(0.2, min(5.0, float(base)))
        for lj in cfg.free_ar_log_jitter:
            ar = b * math.exp(float(lj))
            ar = max(0.2, min(5.0, ar))
            out.append(round(ar, 6))
    uniq = sorted(set(out))
    return uniq if uniq else [1.0]


def _free_bucket_of_ar(ar: float, cfg: CandidateGenConfig) -> str:
    if ar < float(cfg.free_bucket_portrait_max):
        return "portrait"
    if ar <= float(cfg.free_bucket_square_max):
        return "square"
    return "landscape"


def diversity_sample_free(
    candidates: Sequence[Dict[str, Any]],
    k: int,
    cfg: CandidateGenConfig,
) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    if len(candidates) <= k:
        return list(candidates)

    buckets: Dict[str, List[Dict[str, Any]]] = {"portrait": [], "square": [], "landscape": []}
    for c in candidates:
        ar = safe_float(c.get("ar", 1.0), 1.0)
        buckets[_free_bucket_of_ar(ar, cfg)].append(c)

    # Keep stronger candidates first inside each bucket.
    for key in list(buckets.keys()):
        buckets[key] = sorted(
            buckets[key],
            key=lambda x: (
                safe_float(x.get("priority", 0.0), 0.0),
                safe_float(x.get("area_ratio", 0.0), 0.0),
            ),
            reverse=True,
        )

    picked: List[Dict[str, Any]] = []
    picked_keys: set[Tuple[float, float, float, float]] = set()

    def _append_unique(arr: Sequence[Dict[str, Any]]) -> None:
        for c in arr:
            if len(picked) >= k:
                return
            box = c.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
            key = tuple(round(float(v), 6) for v in box)
            if key in picked_keys:
                continue
            picked.append(c)
            picked_keys.add(key)

    n_buckets = 3
    quota = max(1, int(math.ceil(float(k) / float(n_buckets))))
    for bname in ("portrait", "square", "landscape"):
        if len(picked) >= k:
            break
        part = diversity_sample(buckets[bname], min(quota, len(buckets[bname])))
        _append_unique(part)

    if len(picked) < k:
        remain = []
        for c in candidates:
            box = c.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
            key = tuple(round(float(v), 6) for v in box)
            if key in picked_keys:
                continue
            remain.append(c)
        fill = diversity_sample(remain, min(k - len(picked), len(remain)))
        _append_unique(fill)

    return picked[:k]


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
    c2_det: Optional[List[Dict[str, Any]]],
    c3_pose: Optional[List[Dict[str, Any]]],
    cfg: CandidateGenConfig,
    routing: Optional[Dict[str, Any]] = None,
    c2_primary_idx: Optional[int] = None,
    c2_union_box_xyxy: Optional[List[float]] = None,
) -> Dict[str, Any]:
    boxes: List[List[float]] = []
    source_parts: List[str] = []
    mode = ""
    policy_id = ""
    subject_set: Dict[str, Any] = {}
    if isinstance(routing, dict):
        mode = str(routing.get("subject_mode", ""))
        policy_id = str(routing.get("policy_id", ""))
        ss = routing.get("subject_set", {})
        if isinstance(ss, dict):
            subject_set = ss

    # Prefer explicit person detections when available.
    det_person_boxes: List[List[float]] = []
    if c2_det:
        for d in c2_det:
            if not isinstance(d, dict):
                continue
            try:
                cid = int(d.get("class_id", -1))
            except Exception:
                cid = -1
            if cid != 0:
                continue
            if float(d.get("score", 0.0)) < 0.15:
                continue
            b = d.get("box")
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            det_person_boxes.append(norm_box_xyxy(b, float(width), float(height)))
    if det_person_boxes:
        pb = union_boxes(det_person_boxes)
        if pb is not None:
            boxes.append(pb)
            source_parts.append("c2det_person")

    if c2_seg:
        def c2_rank(seg: Dict[str, Any]) -> float:
            imp = float(seg.get("importance_score", float("-inf")))
            if math.isfinite(imp):
                return imp
            area = float(seg.get("area", 0.0))
            score = float(seg.get("score", 0.0))
            return score + 0.001 * math.sqrt(max(area, 0.0))

        best = None
        if c2_primary_idx is not None:
            try:
                idx = int(c2_primary_idx)
            except Exception:
                idx = -1
            if 0 <= idx < len(c2_seg):
                best = c2_seg[idx]
        if best is None:
            best = max(c2_seg, key=c2_rank)
        if "box" in best:
            boxes.append(norm_box_xyxy(best["box"], float(width), float(height)))
            source_parts.append("c2")

    num_people = len(det_person_boxes)
    if c3_pose:
        pose_boxes = []
        for p in c3_pose:
            b = p.get("bbox")
            if not b:
                continue
            pose_boxes.append(norm_box_xyxy(b, float(width), float(height)))
        num_people = max(num_people, len(pose_boxes))
        if pose_boxes:
            pb = union_boxes(pose_boxes)
            if pb is not None:
                boxes.append(pb)
                source_parts.append("c3")

    union_pref = None
    if isinstance(c2_union_box_xyxy, (list, tuple)) and len(c2_union_box_xyxy) == 4:
        union_pref = norm_box_xyxy(c2_union_box_xyxy, float(width), float(height))
    union_in_routing = subject_set.get("union_box_xyxy")
    if isinstance(union_in_routing, (list, tuple)) and len(union_in_routing) == 4:
        union_pref = norm_box_xyxy(union_in_routing, float(width), float(height))

    union_used = False
    if mode in {"portrait_group", "object_multi"} and union_pref is not None:
        subj_box = union_pref
        source_parts.append("routing_union")
        union_used = True
    elif mode in {"scene_landscape", "background_texture_copyspace", "text_document"}:
        if union_pref is not None:
            subj_box = union_pref
            source_parts.append("routing_union")
            union_used = True
        else:
            subj_box = [0.2, 0.2, 0.8, 0.8]
            source_parts.append("routing_none")
    else:
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
        "subject_mode": mode,
        "policy_id": policy_id,
        "union_used": bool(union_used),
        "multi_subject": bool(subject_set.get("multi_subject", False)),
    }


def add_candidate(
    out: List[Dict[str, Any]],
    box: Sequence[float],
    image_ar: float,
    target_ar: Optional[float],
    ar_tol: float,
    a_min: float,
    a_max: float,
    source: str,
    scale_idx: str,
    must_keep: bool = False,
    priority: float = 0.0,
    source_types: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    b = [x1, y1, x2, y2]
    area = box_area(b)
    if area < a_min or area > a_max:
        allow_full_baseline = bool(must_keep) and str(source).startswith("baseline_full") and area <= 1.000001
        if not allow_full_baseline:
            return None
    ar_val = actual_ar(b, image_ar)
    if target_ar is not None and abs(ar_val - target_ar) > ar_tol:
        return None
    cand = {
        "bbox_norm_xyxy": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
        "area_ratio": round(area, 6),
        "ar": round(ar_val, 6),
        "source_types": list(source_types) if source_types else [source],
        "source": source,
        "scale_idx": str(scale_idx),
        "must_keep": bool(must_keep),
        "priority": float(priority),
    }
    if extra:
        cand.update(extra)
    out.append(cand)
    return cand


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
) -> Optional[Dict[str, Any]]:
    w_norm = math.sqrt(max(area_t, 1e-12) * target_ar / image_ar)
    h_norm = math.sqrt(max(area_t, 1e-12) * image_ar / target_ar)
    x1 = cx - w_norm * 0.5
    y1 = cy - h_norm * 0.5
    x2 = cx + w_norm * 0.5
    y2 = cy + h_norm * 0.5

    return add_candidate(
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
        x1_subj = clamp(cx_subj - bw * 0.5, 0.0, slack)
        add_candidate(
            out=cands,
            box=[x1_subj, 0.0, x1_subj + bw, 1.0],
            image_ar=image_ar,
            target_ar=target_ar,
            ar_tol=cfg.ar_tol,
            a_min=cfg.a_min,
            a_max=cfg.a_max,
            source="baseline_maxarea_subject",
            scale_idx="maxs",
            must_keep=True,
            priority=9.1,
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
    y1_subj = clamp(cy_subj - bh * 0.5, 0.0, slack)
    add_candidate(
        out=cands,
        box=[0.0, y1_subj, 1.0, y1_subj + bh],
        image_ar=image_ar,
        target_ar=target_ar,
        ar_tol=cfg.ar_tol,
        a_min=cfg.a_min,
        a_max=cfg.a_max,
        source="baseline_maxarea_subject",
        scale_idx="maxs",
        must_keep=True,
        priority=9.1,
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


def _teacher_ar_match(
    by_ar: Dict[str, List[Dict[str, Any]]],
    target_ar: float,
    ar_tol: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ar_text, arr in by_ar.items():
        try:
            ar_val = parse_ar_loose(ar_text)
        except Exception:
            continue
        if abs(ar_val - target_ar) <= max(1e-4, ar_tol):
            out.extend(arr)
    return out


def _teacher_jitter_boxes(
    seed_box: Sequence[float],
    target_norm_ar: float,
    cfg: CandidateGenConfig,
) -> List[List[float]]:
    cx, cy = box_center(seed_box)
    sw, sh = box_wh(seed_box)
    sw = max(sw, 1e-6)
    sh = max(sh, 1e-6)

    shift_vals = [0.0]
    for frac in cfg.teacher_jitter_shift_fracs:
        f = abs(float(frac))
        if f <= 0:
            continue
        shift_vals.extend([-f, f])
    shift_vals = sorted(set(round(v, 6) for v in shift_vals))

    out: List[List[float]] = []
    for s in cfg.teacher_jitter_scales:
        s = float(s)
        if s <= 0:
            continue
        w2 = sw * s
        h2 = sh * s
        for dx in shift_vals:
            for dy in shift_vals:
                if abs(dx) <= 1e-12 and abs(dy) <= 1e-12 and abs(s - 1.0) <= 1e-12:
                    continue
                cx2 = cx + dx * sw
                cy2 = cy + dy * sh
                box0 = clip_box01(
                    [
                        cx2 - 0.5 * w2,
                        cy2 - 0.5 * h2,
                        cx2 + 0.5 * w2,
                        cy2 + 0.5 * h2,
                    ]
                )
                box1 = project_box_to_ar(
                    box0,
                    target_ar=target_norm_ar,
                    prefer_expand=False,
                )
                if box_area(box1) <= 1e-8:
                    continue
                out.append([round(v, 6) for v in box1])
    return out


def inject_teacher_proposals(
    out: List[Dict[str, Any]],
    *,
    target_ar: float,
    target_ar_text: str,
    image_ar: float,
    teacher_proposals: Optional[Dict[str, Dict[str, Any]]],
    cfg: CandidateGenConfig,
    teacher_meta_out: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Inject proposal-teacher seeds/projections/jitters into candidate pool.

    Candidate-level source tags follow v1.9 design:
    - "teacher:{teacher_id}" for seed/proj candidates
    - "teacher:jitter" for local-neighborhood candidates
    """
    if not teacher_proposals:
        return
    target_norm_ar = max(float(target_ar) / max(float(image_ar), 1e-6), 1e-6)

    for teacher_id in sorted(teacher_proposals.keys()):
        payload = teacher_proposals.get(teacher_id, {})
        if not isinstance(payload, dict):
            continue
        free_form = payload.get("free_form", [])
        by_ar = payload.get("by_ar", {})
        if not isinstance(by_ar, dict):
            by_ar = {}

        seed_budget = max(0, int(cfg.teacher_max_seeds_per_teacher))
        if seed_budget <= 0:
            continue

        ar_specific = _teacher_ar_match(
            by_ar=by_ar,
            target_ar=target_ar,
            ar_tol=max(cfg.ar_tol, 0.02),
        )

        used = 0

        def _append_meta(
            *,
            stage: str,
            box: Sequence[float],
            score: Optional[float],
        ) -> None:
            if teacher_meta_out is None:
                return
            teacher_meta_out.append(
                {
                    "teacher_id": teacher_id,
                    "target_ar": target_ar_text,
                    "stage": stage,
                    "bbox_norm_xyxy": [round(float(v), 6) for v in box],
                    "bbox_norm": [round(float(v), 6) for v in box],
                    "teacher_score": score,
                }
            )

        def _add_seed_and_jitter(
            *,
            seed_box: Sequence[float],
            seed_stage: str,
            teacher_score: Optional[float],
            seed_idx: int,
        ) -> None:
            scale_idx_seed = f"teach_{teacher_id}_{seed_stage}{seed_idx}"
            seed_cand = add_candidate(
                out=out,
                box=seed_box,
                image_ar=image_ar,
                target_ar=target_ar,
                ar_tol=cfg.ar_tol,
                a_min=cfg.a_min,
                a_max=cfg.a_max,
                source=f"teacher:{teacher_id}",
                source_types=[f"teacher:{teacher_id}"],
                scale_idx=scale_idx_seed,
                must_keep=False,
                priority=cfg.teacher_seed_priority,
                extra={
                    "teacher_id": teacher_id,
                    "teacher_score": teacher_score,
                    "teacher_stage": seed_stage,
                },
            )
            if seed_cand is not None:
                _append_meta(stage=seed_stage, box=seed_cand["bbox_norm_xyxy"], score=teacher_score)

            jitter_boxes = _teacher_jitter_boxes(
                seed_box=seed_box,
                target_norm_ar=target_norm_ar,
                cfg=cfg,
            )
            for j_idx, jb in enumerate(jitter_boxes):
                jitter_cand = add_candidate(
                    out=out,
                    box=jb,
                    image_ar=image_ar,
                    target_ar=target_ar,
                    ar_tol=cfg.ar_tol,
                    a_min=cfg.a_min,
                    a_max=cfg.a_max,
                    source="teacher:jitter",
                    source_types=["teacher:jitter", f"teacher:{teacher_id}"],
                    scale_idx=f"teach_{teacher_id}_jit{seed_idx}_{j_idx}",
                    must_keep=False,
                    priority=cfg.teacher_jitter_priority,
                    extra={
                        "teacher_id": teacher_id,
                        "teacher_score": teacher_score,
                        "teacher_stage": "jitter",
                    },
                )
                if jitter_cand is not None:
                    _append_meta(
                        stage="jitter",
                        box=jitter_cand["bbox_norm_xyxy"],
                        score=teacher_score,
                    )

        for item in ar_specific:
            if used >= seed_budget:
                break
            box = normalize_teacher_box(item)
            if box is None:
                continue
            score = extract_teacher_score(item)
            _add_seed_and_jitter(
                seed_box=box,
                seed_stage="seed",
                teacher_score=score,
                seed_idx=used,
            )
            used += 1

        for item in free_form:
            if used >= seed_budget:
                break
            seed_box = normalize_teacher_box(item)
            if seed_box is None:
                continue
            score = extract_teacher_score(item)
            _append_meta(stage="seed", box=seed_box, score=score)

            proj_box = project_box_to_ar(
                seed_box,
                target_ar=target_norm_ar,
                prefer_expand=cfg.teacher_prefer_expand,
            )
            _add_seed_and_jitter(
                seed_box=proj_box,
                seed_stage="proj",
                teacher_score=score,
                seed_idx=used,
            )
            used += 1


def _teacher_jitter_boxes_free(
    seed_box: Sequence[float],
    cfg: CandidateGenConfig,
) -> List[List[float]]:
    cx, cy = box_center(seed_box)
    sw, sh = box_wh(seed_box)
    sw = max(sw, 1e-6)
    sh = max(sh, 1e-6)

    shift_vals = [0.0]
    for frac in cfg.teacher_jitter_shift_fracs:
        f = abs(float(frac))
        if f <= 0:
            continue
        shift_vals.extend([-f, f])
    shift_vals = sorted(set(round(v, 6) for v in shift_vals))

    ar_jitter = [float(x) for x in cfg.free_teacher_ar_jitter if float(x) > 0]
    if not ar_jitter:
        ar_jitter = [1.0]

    out: List[List[float]] = []
    for s in cfg.teacher_jitter_scales:
        s = float(s)
        if s <= 0:
            continue
        for ar_mul in ar_jitter:
            ar_mul = max(0.2, min(5.0, float(ar_mul)))
            ar_sqrt = math.sqrt(ar_mul)
            w2 = sw * s * ar_sqrt
            h2 = sh * s / ar_sqrt
            for dx in shift_vals:
                for dy in shift_vals:
                    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12 and abs(s - 1.0) <= 1e-12 and abs(ar_mul - 1.0) <= 1e-12:
                        continue
                    cx2 = cx + dx * sw
                    cy2 = cy + dy * sh
                    box0 = clip_box01(
                        [
                            cx2 - 0.5 * w2,
                            cy2 - 0.5 * h2,
                            cx2 + 0.5 * w2,
                            cy2 + 0.5 * h2,
                        ]
                    )
                    if box_area(box0) <= 1e-8:
                        continue
                    out.append([round(v, 6) for v in box0])
    return out


def inject_teacher_proposals_freeform(
    out: List[Dict[str, Any]],
    *,
    target_ar_text: str,
    image_ar: float,
    teacher_proposals: Optional[Dict[str, Dict[str, Any]]],
    cfg: CandidateGenConfig,
    teacher_meta_out: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    FREE-form proposal injection:
    - keep teacher boxes in native AR (no target-AR projection)
    - explore local shift/scale/ar-jitter neighborhoods
    """
    if not teacher_proposals:
        return

    for teacher_id in sorted(teacher_proposals.keys()):
        payload = teacher_proposals.get(teacher_id, {})
        if not isinstance(payload, dict):
            continue

        free_form = payload.get("free_form", [])
        by_ar = payload.get("by_ar", {})
        if not isinstance(by_ar, dict):
            by_ar = {}

        seed_budget = max(0, int(cfg.teacher_max_seeds_per_teacher))
        if seed_budget <= 0:
            continue

        raw_seed_items: List[Tuple[int, Any]] = []
        if isinstance(free_form, list):
            for idx, item in enumerate(free_form):
                raw_seed_items.append((idx, item))

        if by_ar:
            offset = len(raw_seed_items)
            # Deterministic order for reproducibility.
            for ar_text in sorted(by_ar.keys()):
                arr = by_ar.get(ar_text, [])
                if not isinstance(arr, list):
                    continue
                for j, item in enumerate(arr):
                    raw_seed_items.append((offset + j, item))
                offset += len(arr)

        if not raw_seed_items:
            continue

        uniq: Dict[Tuple[float, float, float, float], Tuple[float, int, List[float]]] = {}
        for order_idx, item in raw_seed_items:
            box = normalize_teacher_box(item)
            if box is None:
                continue
            score = extract_teacher_score(item)
            score_v = safe_float(score, -1e9)
            key = tuple(round(float(v), 6) for v in box)
            prev = uniq.get(key)
            if prev is None or score_v > prev[0]:
                uniq[key] = (score_v, int(order_idx), [round(float(v), 6) for v in box])

        seed_list = sorted(
            uniq.values(),
            key=lambda x: (
                x[0],          # higher teacher score first
                -x[1],         # stable: earlier raw item first when same score
            ),
            reverse=True,
        )

        for seed_idx, (score_v, _ord, seed_box) in enumerate(seed_list[:seed_budget]):
            teacher_score = None if score_v <= -1e8 else float(score_v)
            seed_cand = add_candidate(
                out=out,
                box=seed_box,
                image_ar=image_ar,
                target_ar=None,
                ar_tol=cfg.ar_tol,
                a_min=cfg.a_min,
                a_max=cfg.a_max,
                source=f"teacher:{teacher_id}",
                source_types=[f"teacher:{teacher_id}"],
                scale_idx=f"teach_{teacher_id}_seed{seed_idx}",
                must_keep=False,
                priority=cfg.teacher_seed_priority,
                extra={
                    "teacher_id": teacher_id,
                    "teacher_score": teacher_score,
                    "teacher_stage": "seed",
                },
            )
            if seed_cand is not None and teacher_meta_out is not None:
                teacher_meta_out.append(
                    {
                        "teacher_id": teacher_id,
                        "target_ar": target_ar_text,
                        "stage": "seed",
                        "bbox_norm_xyxy": seed_cand["bbox_norm_xyxy"],
                        "bbox_norm": seed_cand["bbox_norm_xyxy"],
                        "teacher_score": teacher_score,
                    }
                )

            jitter_boxes = _teacher_jitter_boxes_free(seed_box=seed_box, cfg=cfg)
            for j_idx, jb in enumerate(jitter_boxes):
                jit_cand = add_candidate(
                    out=out,
                    box=jb,
                    image_ar=image_ar,
                    target_ar=None,
                    ar_tol=cfg.ar_tol,
                    a_min=cfg.a_min,
                    a_max=cfg.a_max,
                    source="teacher:jitter",
                    source_types=["teacher:jitter", f"teacher:{teacher_id}"],
                    scale_idx=f"teach_{teacher_id}_jit{seed_idx}_{j_idx}",
                    must_keep=False,
                    priority=cfg.teacher_jitter_priority,
                    extra={
                        "teacher_id": teacher_id,
                        "teacher_score": teacher_score,
                        "teacher_stage": "jitter",
                    },
                )
                if jit_cand is not None and teacher_meta_out is not None:
                    teacher_meta_out.append(
                        {
                            "teacher_id": teacher_id,
                            "target_ar": target_ar_text,
                            "stage": "jitter",
                            "bbox_norm_xyxy": jit_cand["bbox_norm_xyxy"],
                            "bbox_norm": jit_cand["bbox_norm_xyxy"],
                            "teacher_score": teacher_score,
                        }
                    )


def generate_freeform_candidates(
    target_ar_text: str,
    image_ar: float,
    subject_centroid: Tuple[float, float],
    subject_size: Tuple[float, float],
    has_copy_hint: bool,
    cfg: CandidateGenConfig,
    teacher_proposals: Optional[Dict[str, Dict[str, Any]]] = None,
    teacher_meta_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    cx_subj, cy_subj = subject_centroid
    ws, hs = subject_size
    all_cands: List[Dict[str, Any]] = []

    # (A) FREE baselines: full-frame keep anchor + max-area family over sampled ARs.
    add_candidate(
        out=all_cands,
        box=[0.0, 0.0, 1.0, 1.0],
        image_ar=image_ar,
        target_ar=None,
        ar_tol=cfg.ar_tol,
        a_min=cfg.a_min,
        a_max=cfg.a_max,
        source="baseline_full",
        scale_idx="free_full",
        must_keep=True,
        priority=10.0,
    )

    ar_samples = free_ar_sample_values(cfg)
    for i, ar_t in enumerate(ar_samples):
        bs = generate_baseline_candidates(
            target_ar=float(ar_t),
            image_ar=image_ar,
            subject_centroid=subject_centroid,
            has_copy_hint=has_copy_hint,
            cfg=cfg,
        )
        for c in bs:
            c["scale_idx"] = f"freeb{i}_{c.get('scale_idx', 'x')}"
            all_cands.append(c)

    # (B) Grid + multi-scale over sampled ARs.
    xs = [i / cfg.free_grid_m for i in range(cfg.free_grid_m + 1)]
    ys = [j / cfg.free_grid_n for j in range(cfg.free_grid_n + 1)]
    for ar_idx, ar_t in enumerate(ar_samples):
        for s_idx, area_t in enumerate(cfg.free_scale_set):
            if not (cfg.a_min <= float(area_t) <= cfg.a_max):
                continue
            for cx in xs:
                for cy in ys:
                    add_center_area_box(
                        out=all_cands,
                        cx=cx,
                        cy=cy,
                        area_t=float(area_t),
                        image_ar=image_ar,
                        target_ar=float(ar_t),
                        cfg=cfg,
                        source="grid",
                        scale_idx=f"fgrid{ar_idx}_{s_idx}",
                        must_keep=False,
                        priority=3.0,
                    )

    # (C) Subject templates + local jitter with AR sampling.
    subj_area = max(cfg.a_min, min(cfg.a_max, ws * hs))
    for ar_idx, ar_t in enumerate(ar_samples):
        for i, m in enumerate(cfg.object_template_scales):
            area_t = max(cfg.a_min, min(cfg.a_max, subj_area * (float(m) ** 2)))
            add_center_area_box(
                out=all_cands,
                cx=cx_subj,
                cy=cy_subj,
                area_t=area_t,
                image_ar=image_ar,
                target_ar=float(ar_t),
                cfg=cfg,
                source="object_template",
                scale_idx=f"fobj{ar_idx}_{i}",
                must_keep=False,
                priority=6.0,
            )

    dxs = [0.0]
    dys = [0.0]
    for frac in cfg.free_jitter_fracs:
        if float(frac) <= 0:
            continue
        dxs.extend([-float(frac) * ws, float(frac) * ws])
        dys.extend([-float(frac) * hs, float(frac) * hs])
    dxs = sorted(set(round(v, 6) for v in dxs))
    dys = sorted(set(round(v, 6) for v in dys))

    for ar_idx, ar_t in enumerate(ar_samples):
        for j_idx, area_t in enumerate(cfg.free_jitter_scales):
            if not (cfg.a_min <= float(area_t) <= cfg.a_max):
                continue
            for dx in dxs:
                for dy in dys:
                    add_center_area_box(
                        out=all_cands,
                        cx=cx_subj + dx,
                        cy=cy_subj + dy,
                        area_t=float(area_t),
                        image_ar=image_ar,
                        target_ar=float(ar_t),
                        cfg=cfg,
                        source="jitter",
                        scale_idx=f"fjit{ar_idx}_{j_idx}",
                        must_keep=False,
                        priority=5.0,
                    )

    # (D) Composition anchors (thirds/phi).
    if cfg.use_phi_thirds:
        third = [1 / 3, 2 / 3]
        phi = [0.382, 0.618]
        centers = [(x, y) for x in third + phi for y in third + phi]
        for ar_idx, ar_t in enumerate(ar_samples):
            for p_idx, area_t in enumerate(cfg.free_phi_scales):
                for c_idx, (cx, cy) in enumerate(centers):
                    add_center_area_box(
                        out=all_cands,
                        cx=cx,
                        cy=cy,
                        area_t=float(area_t),
                        image_ar=image_ar,
                        target_ar=float(ar_t),
                        cfg=cfg,
                        source="phi_thirds",
                        scale_idx=f"fphi{ar_idx}_{p_idx}_{c_idx}",
                        must_keep=False,
                        priority=4.0,
                    )

    # (E) FREE teacher seed injection (native AR + local/ar jitter).
    inject_teacher_proposals_freeform(
        out=all_cands,
        target_ar_text=target_ar_text,
        image_ar=image_ar,
        teacher_proposals=teacher_proposals,
        cfg=cfg,
        teacher_meta_out=teacher_meta_out,
    )

    # (F) Dedupe/NMS and FREE-specific diversity (position + AR buckets).
    all_cands = dedupe_by_rounded_box(all_cands, ndigits=6)
    must_keep = [c for c in all_cands if c.get("must_keep", False)]
    non_keep = [c for c in all_cands if not c.get("must_keep", False)]
    teacher_non_keep = [
        c
        for c in non_keep
        if str(c.get("source", "")).startswith("teacher:")
        or "teacher:jitter" in set(c.get("source_types", []))
    ]
    other_non_keep = [
        c
        for c in non_keep
        if not (
            str(c.get("source", "")).startswith("teacher:")
            or "teacher:jitter" in set(c.get("source_types", []))
        )
    ]

    must_keep = nms_candidates(must_keep, iou_thr=0.98)
    teacher_non_keep = nms_candidates(
        teacher_non_keep,
        iou_thr=cfg.teacher_nms_iou,
        existing=must_keep,
    )
    other_non_keep = nms_candidates(other_non_keep, iou_thr=cfg.free_nms_iou, existing=must_keep)
    non_keep = teacher_non_keep + other_non_keep

    if len(must_keep) >= cfg.max_candidates_per_ar:
        final_cands = sorted(
            must_keep,
            key=lambda c: (safe_float(c.get("priority", 0.0), 0.0), safe_float(c.get("area_ratio", 0.0), 0.0)),
            reverse=True,
        )[: cfg.max_candidates_per_ar]
    else:
        rem = cfg.max_candidates_per_ar - len(must_keep)
        sampled_non_keep = diversity_sample_free(non_keep, k=rem, cfg=cfg)
        final_cands = must_keep + sampled_non_keep

    final_cands = sorted(
        final_cands,
        key=lambda c: (
            1 if c.get("must_keep", False) else 0,
            safe_float(c.get("priority", 0.0), 0.0),
            safe_float(c.get("area_ratio", 0.0), 0.0),
        ),
        reverse=True,
    )

    token = ar_token(target_ar_text)
    for i, c in enumerate(final_cands, start=1):
        c["candidate_id"] = f"{token}_g{cfg.free_grid_m}x{cfg.free_grid_n}_s{c['scale_idx']}_i{i:04d}"
    return final_cands


def generate_candidates_for_ar(
    target_ar: float,
    target_ar_text: str,
    image_ar: float,
    subject_centroid: Tuple[float, float],
    subject_size: Tuple[float, float],
    has_copy_hint: bool,
    cfg: CandidateGenConfig,
    teacher_proposals: Optional[Dict[str, Dict[str, Any]]] = None,
    teacher_meta_out: Optional[List[Dict[str, Any]]] = None,
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

    # (E) v1.9: optional teacher proposal injection (free-form seed -> AR projection
    # + local jitter neighborhood). This complements grid/rule blind spots.
    inject_teacher_proposals(
        out=all_cands,
        target_ar=target_ar,
        target_ar_text=target_ar_text,
        image_ar=image_ar,
        teacher_proposals=teacher_proposals,
        cfg=cfg,
        teacher_meta_out=teacher_meta_out,
    )

    # (F) Deduplicate and apply staged NMS:
    # keep critical baselines first, then suppress non-keep candidates against them.
    all_cands = dedupe_by_rounded_box(all_cands, ndigits=6)
    must_keep = [c for c in all_cands if c.get("must_keep", False)]
    non_keep = [c for c in all_cands if not c.get("must_keep", False)]
    teacher_non_keep = [
        c
        for c in non_keep
        if str(c.get("source", "")).startswith("teacher:")
        or "teacher:jitter" in set(c.get("source_types", []))
    ]
    other_non_keep = [
        c
        for c in non_keep
        if not (
            str(c.get("source", "")).startswith("teacher:")
            or "teacher:jitter" in set(c.get("source_types", []))
        )
    ]

    must_keep = nms_candidates(must_keep, iou_thr=0.98)
    teacher_non_keep = nms_candidates(
        teacher_non_keep,
        iou_thr=cfg.teacher_nms_iou,
        existing=must_keep,
    )
    other_non_keep = nms_candidates(other_non_keep, iou_thr=cfg.nms_iou, existing=must_keep)
    non_keep = teacher_non_keep + other_non_keep

    # (G) Respect candidate cap while preserving baselines:
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

    # (H) Final deterministic ordering + AR-specific candidate ID assignment.
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
    payload["impl_version"] = "candidate_gen_v1_10_free"
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def build_output_record(
    row: pd.Series,
    c2_seg_map: Dict[str, Any],
    c2_det_map: Dict[str, Any],
    c3_map: Dict[str, Any],
    routing_map: Dict[str, Any],
    c2_primary_idx_map: Dict[str, Any],
    c2_union_box_map: Dict[str, Any],
    cfg: CandidateGenConfig,
    cfg_hash: str,
    actual_size_map: Optional[Dict[str, Tuple[int, int]]] = None,
    strict_actual_size: bool = True,
    teacher_proposals_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    img_id = str(row["image_id"])
    width_pq = int(row["width"])
    height_pq = int(row["height"])
    width = width_pq
    height = height_pq
    size_source = "parquet"
    if actual_size_map is not None:
        wh = actual_size_map.get(img_id)
        if wh is not None and wh[0] > 0 and wh[1] > 0:
            width, height = int(wh[0]), int(wh[1])
            size_source = "tar_actual"
        elif strict_actual_size:
            raise ValueError(f"missing actual image size for image_id={img_id}")
    image_ar = float(width) / float(max(1, height))
    tags = ensure_jsonable_tags(row.get("tags"))

    c2_seg = c2_seg_map.get(img_id, [])
    c2_det = c2_det_map.get(img_id, [])
    c3_pose = c3_map.get(img_id, [])
    routing = routing_map.get(img_id, {}) if isinstance(routing_map.get(img_id, {}), dict) else {}
    c2_primary_idx = c2_primary_idx_map.get(img_id, None)
    c2_union_box_xyxy = c2_union_box_map.get(img_id, None)
    subj = resolve_subject_prior(
        width=width,
        height=height,
        c2_seg=c2_seg,
        c2_det=c2_det,
        c3_pose=c3_pose,
        routing=routing,
        c2_primary_idx=c2_primary_idx,
        c2_union_box_xyxy=c2_union_box_xyxy,
        cfg=cfg,
    )

    cx, cy = float(subj["centroid"][0]), float(subj["centroid"][1])
    ws, hs = float(subj["size"][0]), float(subj["size"][1])
    copy_hint = has_copy_space_hint(tags)
    teacher_payload = teacher_proposals_map.get(img_id, {}) if teacher_proposals_map else {}

    candidates_by_ar: Dict[str, List[Dict[str, Any]]] = {}
    per_ar_count: Dict[str, int] = {}
    per_ar_teacher_count: Dict[str, int] = {}
    iou_to_teacher_top1_by_ar: Dict[str, float] = {}
    teacher_candidates: List[Dict[str, Any]] = []
    total_candidates = 0
    for ar_text in cfg.ar_list:
        ar_text_norm = str(ar_text).strip()
        teacher_meta_ar: List[Dict[str, Any]] = []
        if is_free_ar_text(ar_text_norm):
            cands = generate_freeform_candidates(
                target_ar_text="FREE",
                image_ar=image_ar,
                subject_centroid=(cx, cy),
                subject_size=(ws, hs),
                has_copy_hint=copy_hint,
                cfg=cfg,
                teacher_proposals=teacher_payload,
                teacher_meta_out=teacher_meta_ar,
            )
            ar_key = "FREE"
        else:
            ar_t = parse_ar(ar_text_norm)
            cands = generate_candidates_for_ar(
                target_ar=ar_t,
                target_ar_text=ar_text_norm,
                image_ar=image_ar,
                subject_centroid=(cx, cy),
                subject_size=(ws, hs),
                has_copy_hint=copy_hint,
                cfg=cfg,
                teacher_proposals=teacher_payload,
                teacher_meta_out=teacher_meta_ar,
            )
            ar_key = ar_text_norm
        candidates_by_ar[ar_key] = cands
        per_ar_count[ar_key] = len(cands)
        total_candidates += len(cands)
        if teacher_meta_ar:
            teacher_candidates.extend(teacher_meta_ar)
            per_ar_teacher_count[ar_key] = len(teacher_meta_ar)
            seed_boxes = [
                m["bbox_norm_xyxy"]
                for m in teacher_meta_ar
                if str(m.get("stage", "")) in {"seed", "proj"}
            ]
            if seed_boxes and cands:
                top1_box = cands[0]["bbox_norm_xyxy"]
                iou_to_teacher_top1_by_ar[ar_key] = round(
                    max(iou_xyxy(top1_box, b) for b in seed_boxes),
                    6,
                )
            else:
                iou_to_teacher_top1_by_ar[ar_key] = 0.0

    return {
        "image_id": img_id,
        "width": width,
        "height": height,
        "width_parquet": width_pq,
        "height_parquet": height_pq,
        "size_source": size_source,
        "image_ar": round(image_ar, 6),
        "meta_norm": {
            "width_norm_ref": int(width),
            "height_norm_ref": int(height),
            "image_ar_norm_ref": round(image_ar, 6),
            "size_source": size_source,
            "width_parquet": int(width_pq),
            "height_parquet": int(height_pq),
        },
        "tags": tags,
        "routing": {
            "subject_mode": str(routing.get("subject_mode", "")),
            "policy_id": str(routing.get("policy_id", "")),
            "subject_mode_conf": safe_float(routing.get("subject_mode_conf", 0.0), 0.0),
            "subject_set": routing.get("subject_set", {}) if isinstance(routing.get("subject_set"), dict) else {},
            "subject_mode_reasons": routing.get("subject_mode_reasons", [])
            if isinstance(routing.get("subject_mode_reasons"), list)
            else [],
            "subject_mode_flags": routing.get("subject_mode_flags", {})
            if isinstance(routing.get("subject_mode_flags"), dict)
            else {},
            "subject_mode_conflict": bool(routing.get("subject_mode_conflict", False)),
            "router_rule_id": str(routing.get("router_rule_id", "")),
            "router_signals": routing.get("router_signals", {})
            if isinstance(routing.get("router_signals"), dict)
            else {},
            "shot_type": str(routing.get("shot_type", "unknown") or "unknown"),
            "primary_subject_type": str(routing.get("primary_subject_type", "other") or "other"),
            "primary_subject_source": str(routing.get("primary_subject_source", "none") or "none"),
        },
        "subject": {
            "primary_box": [round(v, 6) for v in subj.get("bbox_norm_xyxy", [0.25, 0.25, 0.75, 0.75])],
            "union_box": routing.get("subject_set", {}).get("union_box_xyxy")
            if isinstance(routing.get("subject_set"), dict)
            else None,
        },
        "subject_prior": subj,
        "candidate_gen_hash": cfg_hash,
        "proposal_injected": bool(teacher_candidates),
        "teacher_candidates": teacher_candidates,
        "iou_to_teacher_top1_by_ar": iou_to_teacher_top1_by_ar,
        "candidates_by_ar": candidates_by_ar,
        "stats": {
            "num_candidates_total": int(total_candidates),
            "num_candidates_by_ar": per_ar_count,
            "num_teacher_candidates_total": int(len(teacher_candidates)),
            "num_teacher_candidates_by_ar": per_ar_teacher_count,
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
    subject_mode_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    has_people_count = 0
    proposal_injected_count = 0
    teacher_stage_counter: Counter[str] = Counter()
    teacher_id_counter: Counter[str] = Counter()

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
            routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
            subject_mode_counter[str(routing.get("subject_mode", "unknown"))] += 1
            policy_counter[str(routing.get("policy_id", "unknown"))] += 1
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

            if bool(rec.get("proposal_injected", False)):
                proposal_injected_count += 1
            for tc in rec.get("teacher_candidates", []) or []:
                if not isinstance(tc, dict):
                    continue
                teacher_stage_counter[str(tc.get("stage", "unknown"))] += 1
                teacher_id_counter[str(tc.get("teacher_id", "unknown"))] += 1

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
        "subject_mode_counts": dict(subject_mode_counter),
        "policy_id_counts": dict(policy_counter),
        "has_people_rate": float(has_people_count / rows) if rows > 0 else 0.0,
        "proposal_injected_rate": float(proposal_injected_count / rows) if rows > 0 else 0.0,
        "teacher_stage_counts": dict(teacher_stage_counter),
        "teacher_id_counts": dict(teacher_id_counter),
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
    p = argparse.ArgumentParser(description="AR-conditioned + FREE-form candidate generator (Phase B1)")
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
    p.add_argument("--use_actual_image_size", type=int, default=1, help="1=use tar actual size for normalization")
    p.add_argument("--strict_actual_size", type=int, default=1, help="1=error when any image lacks actual TAR size")
    p.add_argument("--tar_dir", default="", help="required when use_actual_image_size=1")
    p.add_argument("--image_dir", default="", help="optional local curated image dir (<image_id>.<ext>)")
    p.add_argument("--actual_size_cache_json", default="", help="optional cache for image_id->(w,h)")
    p.add_argument(
        "--teacher_proposals_jsonl",
        nargs="+",
        default=[],
        help="optional proposal-teacher jsonl path(s) for v1.9 seed injection",
    )
    p.add_argument("--teacher_nms_iou", type=float, default=0.95)
    p.add_argument("--teacher_max_seeds_per_teacher", type=int, default=1)
    p.add_argument("--teacher_prefer_expand", type=int, default=1, help="1=expand-first AR projection")
    p.add_argument(
        "--teacher_jitter_shift_fracs",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEACHER_JITTER_SHIFT_FRACS),
    )
    p.add_argument(
        "--teacher_jitter_scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEACHER_JITTER_SCALES),
    )
    p.add_argument("--free_ar_values", nargs="+", type=float, default=list(DEFAULT_FREE_AR_VALUES))
    p.add_argument("--free_ar_log_jitter", nargs="+", type=float, default=list(DEFAULT_FREE_AR_LOG_JITTER))
    p.add_argument("--free_grid_m", type=int, default=4)
    p.add_argument("--free_grid_n", type=int, default=4)
    p.add_argument("--free_scale_set", nargs="+", type=float, default=list(DEFAULT_FREE_SCALE_SET))
    p.add_argument("--free_jitter_fracs", nargs="+", type=float, default=list(DEFAULT_FREE_JITTER_FRACS))
    p.add_argument("--free_jitter_scales", nargs="+", type=float, default=list(DEFAULT_FREE_JITTER_SCALES))
    p.add_argument("--free_phi_scales", nargs="+", type=float, default=list(DEFAULT_FREE_PHI_SCALES))
    p.add_argument("--free_nms_iou", type=float, default=0.85)
    p.add_argument(
        "--free_teacher_ar_jitter",
        nargs="+",
        type=float,
        default=list(DEFAULT_FREE_TEACHER_AR_JITTER),
        help="FREE teacher jitter AR multipliers",
    )

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
    p.add_argument("--num_workers", type=int, default=0, help="candidate build workers (0=auto, 1=single)")
    p.add_argument("--mp_chunksize", type=int, default=64, help="chunksize for multiprocessing map")
    p.add_argument(
        "--mp_start_method",
        type=str,
        default="auto",
        choices=("auto", "fork", "forkserver", "spawn"),
        help="multiprocessing start method",
    )
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
        teacher_nms_iou=float(args.teacher_nms_iou),
        teacher_max_seeds_per_teacher=int(args.teacher_max_seeds_per_teacher),
        teacher_prefer_expand=bool(int(args.teacher_prefer_expand)),
        teacher_jitter_shift_fracs=tuple(float(x) for x in args.teacher_jitter_shift_fracs),
        teacher_jitter_scales=tuple(float(x) for x in args.teacher_jitter_scales),
        free_ar_values=tuple(float(x) for x in args.free_ar_values),
        free_ar_log_jitter=tuple(float(x) for x in args.free_ar_log_jitter),
        free_grid_m=max(1, int(args.free_grid_m)),
        free_grid_n=max(1, int(args.free_grid_n)),
        free_scale_set=tuple(float(x) for x in args.free_scale_set),
        free_jitter_fracs=tuple(float(x) for x in args.free_jitter_fracs),
        free_jitter_scales=tuple(float(x) for x in args.free_jitter_scales),
        free_phi_scales=tuple(float(x) for x in args.free_phi_scales),
        free_nms_iou=float(args.free_nms_iou),
        free_teacher_ar_jitter=tuple(float(x) for x in args.free_teacher_ar_jitter),
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

    c2_seg_map = load_jsonl_map(c2_path, value_key="c2_seg")
    c2_det_map = load_jsonl_map(c2_path, value_key="c2_det")
    c3_map = load_jsonl_map(c3_path, value_key="c3_pose")
    routing_map = load_jsonl_map(c2_path, value_key="routing")
    c2_primary_idx_map = load_jsonl_map(c2_path, value_key="c2_primary_idx")
    c2_union_box_map = load_jsonl_map(c2_path, value_key="c2_union_box_xyxy")
    teacher_proposal_paths = [Path(x) for x in args.teacher_proposals_jsonl if str(x).strip()]
    teacher_proposals_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if teacher_proposal_paths:
        teacher_proposals_map = load_teacher_proposals_map(teacher_proposal_paths)

    c2_det_nonempty = sum(1 for v in c2_det_map.values() if isinstance(v, list) and len(v) > 0)
    c3_nonempty = sum(1 for v in c3_map.values() if isinstance(v, list) and len(v) > 0)
    print(
        f"[CandidateGen] rows={len(df)} c2_seg={len(c2_seg_map)} "
        f"c2_det={len(c2_det_map)}(nonempty={c2_det_nonempty}) "
        f"c3={len(c3_map)}(nonempty={c3_nonempty}) "
        f"routing={len(routing_map)} "
        f"teacher_proposals={len(teacher_proposals_map)} cfg_hash={cfg_hash}"
    )

    if args.max_images > 0:
        df = df.head(int(args.max_images))
        print(
            f"[CandidateGen] max_images={args.max_images} -> processing rows={len(df)}"
        )

    row_df = df[["image_id", "width", "height"]].copy()
    if "tags" in df.columns:
        row_df["tags"] = df["tags"]
    else:
        row_df["tags"] = None
    rows: List[Dict[str, Any]] = row_df.to_dict(orient="records")

    num_workers = resolve_num_workers(
        requested_workers=int(args.num_workers),
        num_items=len(rows),
    )
    if num_workers <= 1:
        print("[CandidateGen] worker mode: single-process")
    else:
        print(
            f"[CandidateGen] worker mode: multiprocess workers={num_workers} "
            f"chunksize={max(1, int(args.mp_chunksize))} start={args.mp_start_method}"
        )

    actual_size_map: Optional[Dict[str, Tuple[int, int]]] = None
    if bool(int(args.use_actual_image_size)):
        if not str(args.tar_dir).strip() and not str(args.image_dir).strip():
            raise ValueError("use_actual_image_size=1 requires --tar_dir or --image_dir")
        cache_path = Path(args.actual_size_cache_json) if args.actual_size_cache_json else None
        print("[CandidateGen] Building/loading actual TAR image-size map...")
        actual_size_map = build_actual_size_map(
            df=df,
            tar_dir=str(args.tar_dir),
            image_dir=str(args.image_dir),
            cache_json=cache_path,
            use_cache_if_available=True,
        )
        if not actual_size_map:
            raise RuntimeError("Failed to build actual_size_map (empty result)")
        print(f"[CandidateGen] actual_size_map loaded: {len(actual_size_map)} images")
        strict_actual = bool(int(args.strict_actual_size))
        missing_actual = [str(r.get("image_id", "")) for r in rows if str(r.get("image_id", "")) not in actual_size_map]
        if missing_actual:
            head = ", ".join(missing_actual[:5])
            msg = (
                f"actual_size_map incomplete: missing={len(missing_actual)}/{len(rows)} "
                f"(e.g., {head})"
            )
            if strict_actual:
                raise ValueError(msg)
            print(f"[CandidateGen] WARN: {msg} -> fallback to parquet size for missing ids")

    num_written = 0
    avg_total = 0.0
    size_source_counter = Counter()
    t0 = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as out_f:
        desc = "CandidateGen-MP" if num_workers > 1 else "CandidateGen"
        iter_rec = iter_build_output_records(
            rows=rows,
            num_workers=num_workers,
            mp_chunksize=max(1, int(args.mp_chunksize)),
            mp_start_method=str(args.mp_start_method),
            c2_seg_map=c2_seg_map,
            c2_det_map=c2_det_map,
            c3_map=c3_map,
            routing_map=routing_map,
            c2_primary_idx_map=c2_primary_idx_map,
            c2_union_box_map=c2_union_box_map,
            cfg=cfg,
            cfg_hash=cfg_hash,
            actual_size_map=actual_size_map,
            strict_actual_size=bool(int(args.strict_actual_size)),
            teacher_proposals_map=teacher_proposals_map,
        )
        for rec in tqdm(iter_rec, total=len(rows), desc=desc):
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1
            avg_total += rec["stats"]["num_candidates_total"]
            size_source_counter[str(rec.get("size_source", "unknown"))] += 1

    elapsed = max(1e-8, time.perf_counter() - t0)
    avg_total = avg_total / max(1, num_written)
    print(
        f"[CandidateGen] Done. written={num_written} avg_total_candidates={avg_total:.2f} "
        f"elapsed={elapsed:.2f}s throughput={num_written/elapsed:.2f} img/s"
    )
    if size_source_counter:
        print(f"[CandidateGen] size_source_counts={dict(size_source_counter)}")
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
