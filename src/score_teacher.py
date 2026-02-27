"""
Teacher Scorer (Section 9): cheap -> expensive -> Top-K diversity.

Implements a deterministic scorer pipeline for AR-conditioned candidates using
precomputed features (C2/C3/C5) and candidate boxes.

Notes
-----
- OCR term is intentionally disabled in this implementation.
- Expensive stage uses lightweight deterministic proxies when full crop-level
  aesthetic/text-alignment models are unavailable in precompute inputs.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import tarfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


SHOT_TYPE_GROUP_WORDS = {
    "group",
    "family",
    "team",
    "friends",
    "couple",
    "crowd",
    "people",
}
SHOT_TYPE_FULL_WORDS = {
    "full body",
    "full length",
    "full-length",
    "whole body",
    "standing",
    "head to toe",
    "feet",
    "legs",
}
SHOT_TYPE_HALF_WORDS = {
    "waist up",
    "upper body",
    "half body",
    "half length",
    "mid shot",
    "medium shot",
    "torso",
}
SHOT_TYPE_HEADSHOT_WORDS = {
    "headshot",
    "close up",
    "close-up",
    "closeup",
    "portrait",
    "face",
    "beauty",
    "selfie",
    "profile picture",
}

FORMAL_ID_WORDS = {"passport", "id photo", "mugshot", "profile", "linkedin", "resume"}
CORPORATE_WORDS = {"business", "corporate", "professional", "office", "executive"}
BEAUTY_FASHION_WORDS = {"beauty", "makeup", "fashion", "model", "glamour", "hairstyle"}
LIFESTYLE_WORDS = {"lifestyle", "outdoor", "travel", "candid"}

PROFILE_VIEW_WORDS = {"profile", "side view", "looking left", "looking right"}
COPYSPACE_WORDS = {
    "copy space",
    "copyspace",
    "negative space",
    "blank space",
    "text space",
    "banner",
    "template",
}
ISOLATED_WORDS = {"isolated", "cut out", "on white", "white background", "packshot"}
LANDSCAPE_HINTS = {
    "landscape",
    "seascape",
    "horizon",
    "skyline",
    "mountain",
    "panoramic",
    "wide",
    "architecture",
}
PRODUCT_HINTS = {"product", "packshot", "merchandise", "mockup"}

AESTHETIC_DEFAULT_URL = (
    "https://raw.githubusercontent.com/christophschuhmann/"
    "improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth"
)

DEFAULT_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")


# COCO-17 keypoint indices.
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16
CRITICAL_JOINTS = ("neck", "wrist_l", "wrist_r", "knee_l", "knee_r", "ankle_l", "ankle_r")


@dataclass
class TeacherScorerConfig:
    cheap_top_m: int = 30
    top_k: int = 5
    tau_div: float = 0.75
    use_real_expensive: bool = True

    # Structural hard checks
    area_min: float = 0.15
    area_max: float = 0.95
    ar_hard_eps: float = 0.01
    hard_face_rule: bool = True
    severe_kp_margin_alpha: float = 0.03
    hard_joint_reject_count: int = 2

    # Cheap scoring weights (base)
    lambda_cov: float = 1.20
    lambda_cut: float = 1.80
    lambda_text: float = 0.0  # OCR-disabled
    lambda_comp: float = 1.00
    lambda_hr: float = 0.65
    lambda_lr: float = 0.55
    lambda_sym: float = 0.25
    lambda_ctx: float = 0.45
    lambda_cs: float = 0.35

    # Cut penalty composition
    alpha_face_cut: float = 2.0
    alpha_joint_cut: float = 1.0
    alpha_subject_border: float = 0.7

    # Composition sub-weights
    w_third: float = 0.45
    w_phi: float = 0.15
    w_center: float = 0.25
    w_horizon: float = 0.15
    horizon_conf_thr: float = 0.25

    # Headroom/lookroom formula constants
    sigma_h: float = 0.05
    gamma_h: float = 2.0
    sigma_l: float = 0.25
    gamma_l: float = 1.0

    # Expensive score weights
    w_a: float = 1.0
    w_ca: float = 0.3
    w_cov: float = 0.5
    w_cut: float = 2.0
    w_text: float = 0.0  # OCR-disabled
    w_edge: float = 0.5
    aesthetic_score_min: float = 1.0
    aesthetic_score_max: float = 10.0
    expensive_eval_top_m: int = 0  # 0 means evaluate all cheap_top_m in expensive stage
    exp_preprocess_workers: int = 0  # 0 means auto (bounded by CPU count)
    exp_pin_memory: bool = True

    # Keep-vs-crop
    w_area_default: float = 0.10
    tau_improve_default: float = 0.035

    # QA thresholds
    horizon_third_tau: float = 0.08
    roll_violation_deg: float = 3.0
    subject_coverage_fail_thr: float = 0.9
    overcrop_factor: float = 0.75


@dataclass
class ARStats:
    images: int = 0
    candidates_in_mean: float = 0.0
    cheap_kept_mean: float = 0.0
    keep_full_count: int = 0
    minimal_crop_count: int = 0
    crop_count: int = 0
    overcrop_count: int = 0

    top1_face_cut_count: int = 0
    top1_joint_cut_count: int = 0
    top1_subject_cov_fail_count: int = 0

    portrait_subset_count: int = 0
    headroom_violation_count: int = 0
    lookroom_subset_count: int = 0
    lookroom_violation_count: int = 0

    horizon_subset_count: int = 0
    horizon_violation_count: int = 0
    roll_subset_count: int = 0
    roll_violation_count: int = 0

    copyspace_subset_count: int = 0
    copyspace_preserve_count: int = 0

    fallback_activated_count: int = 0
    fallback_mode_counts: Dict[str, int] = None

    delta_improve: List[float] = None
    top1_top2_iou: List[float] = None
    horizon_third_dist: List[float] = None

    def __post_init__(self) -> None:
        if self.delta_improve is None:
            self.delta_improve = []
        if self.top1_top2_iou is None:
            self.top1_top2_iou = []
        if self.horizon_third_dist is None:
            self.horizon_third_dist = []
        if self.fallback_mode_counts is None:
            self.fallback_mode_counts = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher scorer: cheap -> expensive -> Top-K diversity")
    p.add_argument("--candidates_jsonl", required=True)
    p.add_argument("--features_jsonl", required=True)
    p.add_argument("--c1_jsonl", default="")
    p.add_argument("--parquet", default="", help="filtered parquet with tar_name/bucket for crop loading")
    p.add_argument("--tar_dir", default="", help="SSTK tar root directory")
    p.add_argument("--image_dir", default="", help="optional local curated image dir (<image_id>.<ext>)")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_overview_json", required=True)
    p.add_argument("--output_overview_by_ar_csv", default="")

    p.add_argument("--cheap_top_m", type=int, default=30)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--tau_div", type=float, default=0.75)
    p.add_argument("--use_real_expensive", type=int, default=1, help="1=real A(I_b)/cos(E_I,E_T), 0=proxy fallback")
    p.add_argument("--align_model_name", type=str, default="", help="OpenCLIP model for E_I(I_b)")
    p.add_argument("--align_pretrained", type=str, default="", help="OpenCLIP pretrained tag for E_I(I_b)")
    p.add_argument("--align_device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--aesthetic_device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--exp_batch_size", type=int, default=24)
    p.add_argument("--expensive_eval_top_m", type=int, default=0, help="0=all cheap_top_m, >0=capped expensive eval")
    p.add_argument("--exp_preprocess_workers", type=int, default=0, help="0=auto, >0=thread workers for clip preprocess")
    p.add_argument("--exp_pin_memory", type=int, default=1, help="1=pin CPU batch tensor before H2D copy")
    p.add_argument(
        "--aesthetic_mlp_path",
        type=str,
        default="weights/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
    )
    p.add_argument("--aesthetic_mlp_url", type=str, default=AESTHETIC_DEFAULT_URL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0, help="0 means all")
    p.add_argument("--num_shards", type=int, default=1, help="Shard count for multi-GPU/process execution")
    p.add_argument("--shard_index", type=int, default=0, help="Current shard index [0, num_shards)")
    p.add_argument("--progress", type=int, default=1)
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def parse_ar(ar_text: str) -> float:
    t = str(ar_text).strip()
    if ":" in t:
        a, b = t.split(":", 1)
        return float(a) / float(b)
    return float(t)


def box_area(b: Sequence[float]) -> float:
    x1, y1, x2, y2 = [safe_float(v) for v in b]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_wh(b: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v) for v in b]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def box_center(b: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v) for v in b]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def box_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    ix1, iy1, ix2, iy2 = [safe_float(v) for v in inner]
    ox1, oy1, ox2, oy2 = [safe_float(v) for v in outer]
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def clip_box01(b: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [safe_float(v) for v in b]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [safe_float(v) for v in a]
    bx1, by1, bx2, by2 = [safe_float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    inter = inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def norm_box_xyxy(box_px: Sequence[float], width: int, height: int) -> List[float]:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    x1, y1, x2, y2 = [safe_float(v) for v in box_px]
    return clip_box01([x1 / w, y1 / h, x2 / w, y2 / h])


def local_coords(crop: Sequence[float], x: float, y: float) -> Tuple[float, float]:
    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    w = max(1e-8, x2 - x1)
    h = max(1e-8, y2 - y1)
    return (x - x1) / w, (y - y1) / h


def distance_to_thirds(local_xy: Tuple[float, float]) -> float:
    cx, cy = local_xy
    points = ((1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0))
    return min(math.hypot(cx - px, cy - py) for px, py in points)


def distance_to_phi(local_xy: Tuple[float, float]) -> float:
    cx, cy = local_xy
    points = ((0.382, 0.382), (0.618, 0.382), (0.382, 0.618), (0.618, 0.618))
    return min(math.hypot(cx - px, cy - py) for px, py in points)


def distance_to_center(local_xy: Tuple[float, float]) -> float:
    cx, cy = local_xy
    return math.hypot(cx - 0.5, cy - 0.5)


def reward_dist(d: float, scale: float) -> float:
    s = max(1e-6, float(scale))
    return math.exp(-max(0.0, float(d)) / s)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = clamp(q, 0.0, 1.0) * (len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    alpha = pos - lo
    return (1.0 - alpha) * arr[lo] + alpha * arr[hi]


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


def normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32, copy=False)


def pick_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s in {"", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if s.startswith("cuda"):
        try:
            import torch

            return s if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


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
            if ext.lower().lstrip(".") not in DEFAULT_IMAGE_EXTS:
                continue
            if stem not in out:
                out[stem] = p
    except Exception:
        return {}
    return out


class LocalImageLoader:
    """
    Local image loader for curated image cache directory.
    """

    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self._index = build_local_image_index(image_dir)

    def close(self) -> None:
        return

    def load(self, image_id: str) -> Optional["Image.Image"]:
        try:
            from PIL import Image
        except Exception:
            return None
        p = self._index.get(str(image_id))
        if p is None:
            return None
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None


class TarImageLoader:
    """
    Lightweight TAR image loader with one-open-tar cache.
    """

    def __init__(self, mapping: Dict[str, Dict[str, Any]], tar_dir: str):
        self.mapping = mapping
        self.tar_dir = tar_dir
        self._current_tar_path: Optional[str] = None
        self._current_tar: Optional[tarfile.TarFile] = None
        self._members_by_name: Dict[str, tarfile.TarInfo] = {}

    def close(self) -> None:
        if self._current_tar is not None:
            try:
                self._current_tar.close()
            except Exception:
                pass
        self._current_tar = None
        self._current_tar_path = None
        self._members_by_name = {}

    def _open_tar(self, tar_path: str) -> None:
        if self._current_tar_path == tar_path and self._current_tar is not None:
            return
        self.close()
        tf = tarfile.open(tar_path, "r")
        members = {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            if not base:
                continue
            members[base] = m
        self._current_tar = tf
        self._members_by_name = members
        self._current_tar_path = tar_path

    def load(self, image_id: str) -> Optional["Image.Image"]:
        try:
            from PIL import Image
        except Exception:
            return None

        info = self.mapping.get(str(image_id))
        if not info:
            return None
        tar_name = str(info.get("tar_name", ""))
        bucket = str(info.get("bucket", "")) if "bucket" in info else ""
        tar_path = resolve_tar_path(self.tar_dir, bucket, tar_name)
        if tar_path is None:
            return None

        self._open_tar(tar_path)
        if self._current_tar is None:
            return None

        for ext in DEFAULT_IMAGE_EXTS:
            name = f"{image_id}.{ext}"
            member = self._members_by_name.get(name)
            if member is None:
                continue
            fobj = self._current_tar.extractfile(member)
            if fobj is None:
                continue
            data = fobj.read()
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None
        return None


class HybridImageLoader:
    """
    Prefer local curated images, fallback to TAR when local cache is missing/corrupt.
    """

    def __init__(
        self,
        *,
        local_loader: Optional[LocalImageLoader],
        tar_loader: Optional[TarImageLoader],
    ):
        self.local_loader = local_loader
        self.tar_loader = tar_loader
        self.local_hits = 0
        self.tar_hits = 0
        self.misses = 0

    def close(self) -> None:
        if self.local_loader is not None:
            self.local_loader.close()
        if self.tar_loader is not None:
            self.tar_loader.close()

    def load(self, image_id: str) -> Optional["Image.Image"]:
        img = None
        if self.local_loader is not None:
            img = self.local_loader.load(image_id)
            if img is not None:
                self.local_hits += 1
                return img
        if self.tar_loader is not None:
            img = self.tar_loader.load(image_id)
            if img is not None:
                self.tar_hits += 1
                return img
        self.misses += 1
        return None


def load_tar_mapping(parquet_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    if "image_id" not in df.columns or "tar_name" not in df.columns:
        raise ValueError(f"parquet missing required columns image_id/tar_name: {parquet_path}")
    if "bucket" in df.columns:
        return df.set_index("image_id")[["tar_name", "bucket"]].to_dict("index")
    return df.set_index("image_id")[["tar_name"]].to_dict("index")


def load_c1_map(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            image_id = str(d.get("image_id", ""))
            if not image_id:
                continue

            img_emb = d.get("c1_img_embed", [])
            txt_emb = d.get("c1_txt_embed", [])
            if not isinstance(img_emb, list) or not isinstance(txt_emb, list):
                continue
            if not img_emb or not txt_emb:
                continue

            try:
                img_arr = normalize_vec(np.asarray(img_emb, dtype=np.float32))
                txt_arr = normalize_vec(np.asarray(txt_emb, dtype=np.float32))
            except Exception:
                continue
            out[image_id] = {"c1_img_embed": img_arr, "c1_txt_embed": txt_arr}
    return out


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_download_file(url: str, dst: Path) -> None:
    if dst.exists():
        return
    ensure_parent_dir(dst)
    print(f"[download] {url} -> {dst}")
    urllib.request.urlretrieve(url, str(dst))


class AestheticMLP:
    """
    MLP architecture from improved-aesthetic-predictor (linearMSE variant).
    """

    def __init__(self, torch_mod: Any, input_size: int = 768):
        nn = torch_mod.nn
        self._torch = torch_mod
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


class ExpensiveModels:
    """
    Real expensive scorer bundle:
    - A(I_b): improved-aesthetic-predictor
    - cos(E_I(I_b), E_T): crop image embedding vs C1 text embedding
    """

    def __init__(
        self,
        *,
        text_dim: int,
        align_model_name: str,
        align_pretrained: str,
        align_device: str,
        aesthetic_device: str,
        batch_size: int,
        preprocess_workers: int,
        pin_memory: bool,
        aesthetic_mlp_path: Path,
        aesthetic_mlp_url: str,
    ):
        import torch
        import open_clip

        self.torch = torch
        self.open_clip = open_clip
        self.batch_size = max(1, int(batch_size))
        req_workers = int(preprocess_workers)
        if req_workers <= 0:
            req_workers = min(8, max(1, int(os.cpu_count() or 1)))
        self.preprocess_workers = max(1, req_workers)
        self.pin_memory = bool(pin_memory)
        self._preprocess_pool: Optional[ThreadPoolExecutor] = None
        if self.preprocess_workers > 1:
            self._preprocess_pool = ThreadPoolExecutor(max_workers=self.preprocess_workers)

        self.align_device = pick_device(align_device)
        self.aes_device = pick_device(aesthetic_device)

        # Align model for cos(E_I(I_b), E_T): match C1 text embedding dimension.
        if not align_model_name or not align_pretrained:
            if int(text_dim) == 1024:
                align_model_name = "ViT-H-14"
                align_pretrained = "laion2b_s32b_b79k"
            elif int(text_dim) == 768:
                align_model_name = "ViT-L-14"
                align_pretrained = "openai"
            else:
                raise ValueError(
                    f"Unsupported C1 embedding dim={text_dim}. "
                    "Provide --align_model_name/--align_pretrained explicitly."
                )

        print(
            f"[expensive] align model: {align_model_name}/{align_pretrained} on {self.align_device} "
            f"(text_dim={text_dim})"
        )
        self.align_model, self.align_preprocess, self.align_device = self._create_clip_with_fallback(
            model_name=align_model_name,
            pretrained=align_pretrained,
            preferred_device=self.align_device,
        )
        self.align_model.eval()
        if self.align_device != "cpu":
            self.align_model = self.align_model.half()

        # Aesthetic predictor (repo: improved-aesthetic-predictor)
        maybe_download_file(aesthetic_mlp_url, aesthetic_mlp_path)
        self.aes_clip, self.aes_preprocess, self.aes_device = self._create_clip_with_fallback(
            model_name="ViT-L-14",
            pretrained="openai",
            preferred_device=self.aes_device,
        )
        self.aes_clip.eval()
        if self.aes_device != "cpu":
            self.aes_clip = self.aes_clip.half()

        mlp = AestheticMLP(torch, input_size=768)
        try:
            state = torch.load(str(aesthetic_mlp_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(aesthetic_mlp_path), map_location="cpu")
        if isinstance(state, dict) and state and all(str(k).startswith("layers.") for k in state.keys()):
            state = {str(k)[7:]: v for k, v in state.items()}
        mlp.model.load_state_dict(state)
        mlp.model.to(self.aes_device)
        mlp.model.eval()
        self.aes_mlp = mlp.model

    def close(self) -> None:
        if self._preprocess_pool is not None:
            self._preprocess_pool.shutdown(wait=True, cancel_futures=False)
            self._preprocess_pool = None

    def _preprocess_batch(
        self,
        *,
        batch_imgs: Sequence["Image.Image"],
        preprocess_fn: Callable[[Any], Any],
        device: str,
    ) -> Any:
        if self._preprocess_pool is not None and len(batch_imgs) > 1:
            tensors = list(self._preprocess_pool.map(preprocess_fn, batch_imgs))
        else:
            tensors = [preprocess_fn(img) for img in batch_imgs]

        batch = self.torch.stack(tensors, dim=0)
        if device != "cpu":
            if self.pin_memory:
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=True)
            batch = batch.half()
        else:
            batch = batch.to(device)
        return batch

    def _create_clip_with_fallback(
        self,
        *,
        model_name: str,
        pretrained: str,
        preferred_device: str,
    ) -> Tuple[Any, Any, str]:
        try:
            model, _, preprocess = self.open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=preferred_device,
            )
            return model, preprocess, preferred_device
        except Exception as e:
            msg = str(e).lower()
            if preferred_device != "cpu" and ("out of memory" in msg or "cuda" in msg):
                print(
                    f"[expensive][warn] failed to load {model_name}/{pretrained} on {preferred_device}: {e}\n"
                    " -> retry on CPU"
                )
                model, _, preprocess = self.open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    device="cpu",
                )
                return model, preprocess, "cpu"
            raise

    def _batched_encode_align(self, crops: Sequence["Image.Image"]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 1), dtype=np.float32)

        all_out: List[np.ndarray] = []
        try:
            with self.torch.inference_mode():
                for i in range(0, len(crops), self.batch_size):
                    batch_imgs = crops[i : i + self.batch_size]
                    batch = self._preprocess_batch(
                        batch_imgs=batch_imgs,
                        preprocess_fn=self.align_preprocess,
                        device=self.align_device,
                    )
                    emb = self.align_model.encode_image(batch)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    all_out.append(emb.detach().cpu().float().numpy())
        except Exception as e:
            if self.align_device != "cpu" and "out of memory" in str(e).lower():
                print("[expensive][warn] align batch OOM on GPU -> switching align model to CPU")
                try:
                    self.torch.cuda.empty_cache()
                except Exception:
                    pass
                self.align_model.to("cpu")
                self.align_device = "cpu"
                return self._batched_encode_align(crops)
            raise
        return np.concatenate(all_out, axis=0).astype(np.float32, copy=False)

    def _batched_predict_aesthetic(self, crops: Sequence["Image.Image"]) -> np.ndarray:
        if not crops:
            return np.zeros((0,), dtype=np.float32)

        all_out: List[np.ndarray] = []
        try:
            with self.torch.inference_mode():
                for i in range(0, len(crops), self.batch_size):
                    batch_imgs = crops[i : i + self.batch_size]
                    batch = self._preprocess_batch(
                        batch_imgs=batch_imgs,
                        preprocess_fn=self.aes_preprocess,
                        device=self.aes_device,
                    )
                    emb = self.aes_clip.encode_image(batch)
                    emb = emb.float()
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    pred = self.aes_mlp(emb).reshape(-1)
                    all_out.append(pred.detach().cpu().float().numpy())
        except Exception as e:
            if self.aes_device != "cpu" and "out of memory" in str(e).lower():
                print("[expensive][warn] aesthetic batch OOM on GPU -> switching aesthetic model to CPU")
                try:
                    self.torch.cuda.empty_cache()
                except Exception:
                    pass
                self.aes_clip.to("cpu")
                self.aes_mlp.to("cpu")
                self.aes_device = "cpu"
                return self._batched_predict_aesthetic(crops)
            raise
        return np.concatenate(all_out, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _crop_from_norm_box(image: "Image.Image", box_norm: Sequence[float]) -> Optional["Image.Image"]:
        w, h = image.size
        x1n, y1n, x2n, y2n = [float(v) for v in box_norm]
        x1 = int(round(clamp(x1n, 0.0, 1.0) * w))
        y1 = int(round(clamp(y1n, 0.0, 1.0) * h))
        x2 = int(round(clamp(x2n, 0.0, 1.0) * w))
        y2 = int(round(clamp(y2n, 0.0, 1.0) * h))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(1, min(w, x2))
        y2 = max(1, min(h, y2))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))

    def predict_for_candidates(
        self,
        image: "Image.Image",
        candidates: Sequence[Dict[str, Any]],
        text_embed: Optional[np.ndarray],
        cfg: TeacherScorerConfig,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns candidate_id -> expensive signals.
        """
        if not candidates:
            return {}

        if getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")

        text_vec = normalize_vec(np.asarray(text_embed, dtype=np.float32)) if text_embed is not None else None

        cid_to_box: Dict[str, List[float]] = {}
        for c in candidates:
            cid = str(c.get("candidate_id", ""))
            b = c.get("bbox_norm_xyxy")
            if not cid or not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            cid_to_box[cid] = [float(v) for v in b]

        # Dedupe by rounded box because 같은 박스가 여러 source/candidate로 반복될 수 있다.
        boxkey_to_crop: Dict[Tuple[float, float, float, float], Any] = {}
        cid_to_boxkey: Dict[str, Tuple[float, float, float, float]] = {}
        for cid, b in cid_to_box.items():
            k = tuple(round(float(v), 6) for v in b)
            cid_to_boxkey[cid] = k
            if k in boxkey_to_crop:
                continue
            crop = self._crop_from_norm_box(image, b)
            if crop is not None:
                boxkey_to_crop[k] = crop

        if not boxkey_to_crop:
            return {}

        keys = list(boxkey_to_crop.keys())
        crops = [boxkey_to_crop[k] for k in keys]

        align_emb = self._batched_encode_align(crops)
        aes_raw = self._batched_predict_aesthetic(crops)

        # Match paper/repo scale to [0,1] for stable weighting.
        aes_norm = (aes_raw - cfg.aesthetic_score_min) / max(1e-8, cfg.aesthetic_score_max - cfg.aesthetic_score_min)
        aes_norm = np.clip(aes_norm, 0.0, 1.0)

        boxkey_to_signal: Dict[Tuple[float, float, float, float], Dict[str, float]] = {}
        for i, k in enumerate(keys):
            if text_vec is None or align_emb.shape[1] != text_vec.shape[0]:
                cos_val = 0.0
            else:
                cos_val = float(np.dot(normalize_vec(align_emb[i]), text_vec))
            boxkey_to_signal[k] = {
                "aesthetic_raw": float(aes_raw[i]),
                "aesthetic_norm": float(aes_norm[i]),
                "cosine_img_text": float(cos_val),
            }

        out: Dict[str, Dict[str, float]] = {}
        for cid, k in cid_to_boxkey.items():
            if k in boxkey_to_signal:
                out[cid] = boxkey_to_signal[k]
        return out


def tags_to_tokens(tags: Any) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = [x.strip().lower() for x in tags.split("|")] if "|" in tags else [tags.strip().lower()]
    elif isinstance(tags, (list, tuple)):
        raw = [str(x).strip().lower() for x in tags]
    else:
        raw = [str(tags).strip().lower()]
    return [x for x in raw if x]


def infer_shot_type(tags: Sequence[str], has_human_evidence: bool = True) -> str:
    if not bool(has_human_evidence):
        return "unknown"
    text = " ".join(tags)
    if any(w in tags for w in SHOT_TYPE_GROUP_WORDS):
        return "group"
    if any(w in text for w in SHOT_TYPE_FULL_WORDS):
        return "full"
    if any(w in text for w in SHOT_TYPE_HALF_WORDS):
        return "half"
    if any(w in text for w in SHOT_TYPE_HEADSHOT_WORDS):
        return "headshot"
    return "unknown"


def infer_portrait_category(tags: Sequence[str]) -> str:
    text = " ".join(tags)
    if any(w in text for w in FORMAL_ID_WORDS):
        return "formal_id"
    if any(w in text for w in CORPORATE_WORDS):
        return "corporate"
    if any(w in text for w in BEAUTY_FASHION_WORDS):
        return "beauty_fashion"
    if any(w in text for w in LIFESTYLE_WORDS):
        return "lifestyle"
    return "generic"


def infer_flags(tags: Sequence[str], has_human_evidence: bool = False) -> Dict[str, bool]:
    text = " ".join(tags)
    explicit_product = any(w in text for w in PRODUCT_HINTS)
    has_packshot = "packshot" in text
    has_white_bg = ("on white" in text) or ("white background" in text)
    has_isolated = ("isolated" in text) or ("cut out" in text)

    # Product/packshot route is disabled when human evidence exists.
    is_product = bool(explicit_product) and (not bool(has_human_evidence))
    is_isolated_packshot = (not bool(has_human_evidence)) and (
        has_packshot or has_white_bg or (has_isolated and explicit_product)
    )

    return {
        "is_profile_view": any(w in text for w in PROFILE_VIEW_WORDS),
        "has_copy_space": any(w in text for w in COPYSPACE_WORDS),
        "is_isolated_packshot": bool(is_isolated_packshot),
        "is_landscape_scene": any(w in text for w in LANDSCAPE_HINTS),
        "is_product": bool(is_product),
    }


def headroom_prior(shot_type: str, portrait_category: str, flags: Dict[str, bool]) -> Tuple[float, float, float]:
    table = {
        "headshot": (0.03, 0.07, 0.12),
        "half": (0.04, 0.08, 0.14),
        "full": (0.02, 0.06, 0.10),
        "group": (0.03, 0.08, 0.14),
        "unknown": (0.03, 0.08, 0.14),
    }
    hmin, htar, hmax = table.get(shot_type, table["unknown"])

    if portrait_category in {"formal_id", "corporate"}:
        hmax -= 0.03
    if portrait_category == "beauty_fashion":
        hmax += 0.05
    if flags.get("has_copy_space", False):
        hmax = min(0.35, hmax + 0.15)

    hmin = clamp(hmin, 0.0, 0.5)
    htar = clamp(htar, hmin + 1e-6, 0.6)
    hmax = clamp(hmax, htar + 1e-6, 0.8)
    return hmin, htar, hmax


def lookroom_prior(shot_type: str, flags: Dict[str, bool]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    table = {
        "headshot": (1.20, 1.50, 2.50),
        "half": (1.15, 1.30, 2.00),
        "full": (1.05, 1.15, 1.60),
        "group": (None, None, None),
        "unknown": (1.10, 1.25, 1.80),
    }
    lo = table.get(shot_type, table["unknown"])
    if lo[0] is None:
        return lo

    lmin, ltar, lmax = lo
    if flags.get("is_profile_view", False):
        ltar += 0.2
        lmax += 0.5
    if flags.get("has_copy_space", False):
        lmax = min(4.0, lmax + 1.0)
    return float(lmin), float(ltar), float(lmax)


def context_target_range(
    shot_type: str,
    flags: Dict[str, bool],
    portrait_category: str,
    has_human_evidence: bool,
) -> Tuple[float, float]:
    if flags.get("is_landscape_scene", False):
        return (0.05, 0.25)
    if flags.get("is_isolated_packshot", False) or flags.get("is_product", False):
        return (0.50, 0.85)

    if shot_type == "headshot":
        lo, hi = (0.55, 0.75)
    elif shot_type == "half":
        lo, hi = (0.40, 0.65)
    elif shot_type == "full":
        lo, hi = (0.30, 0.55)
    elif shot_type == "group":
        lo, hi = (0.25, 0.55)
    else:
        if has_human_evidence:
            lo, hi = (0.20, 0.50)
        else:
            # Non-human unknown scenes should avoid overly small subject-area target.
            lo, hi = (0.30, 0.70)

    if portrait_category == "lifestyle":
        lo = max(0.10, lo - 0.15)
        hi = max(lo + 0.05, hi - 0.15)
    return (lo, hi)


def tau_improve_for_route(shot_type: str, flags: Dict[str, bool], num_people: int) -> float:
    if flags.get("has_copy_space", False) or flags.get("is_landscape_scene", False):
        return 0.055
    if shot_type == "group" or num_people >= 2:
        return 0.045
    if shot_type in {"headshot", "half", "full"}:
        return 0.03
    if flags.get("is_product", False) or flags.get("is_isolated_packshot", False):
        return 0.015
    return 0.035


def area_weight_for_route(flags: Dict[str, bool]) -> float:
    if flags.get("has_copy_space", False) or flags.get("is_landscape_scene", False):
        return 0.12
    if flags.get("is_product", False):
        return 0.06
    return 0.10


def is_portrait_route(shot_type: str, num_people: int) -> bool:
    return shot_type in {"headshot", "half", "full", "group"} or num_people > 0


def safe_first(items: Sequence[Any]) -> Any:
    return items[0] if items else None


def count_c2_person_instances(feat_rec: Dict[str, Any], min_score: float = 0.15) -> int:
    c2 = feat_rec.get("c2_det", []) or feat_rec.get("c2_seg", []) or []
    cnt = 0
    for d in c2:
        if not isinstance(d, dict):
            continue
        cid = d.get("class_id")
        try:
            cid_i = int(cid)
        except Exception:
            continue
        if cid_i != 0:
            continue
        if safe_float(d.get("score", 0.0)) < float(min_score):
            continue
        cnt += 1
    return cnt


def build_subject_bbox(
    candidate_rec: Dict[str, Any],
    feat_rec: Dict[str, Any],
    width: int,
    height: int,
) -> List[float]:
    sp = candidate_rec.get("subject_prior", {})
    if isinstance(sp, dict):
        b = sp.get("bbox_norm_xyxy")
        if isinstance(b, (list, tuple)) and len(b) == 4 and box_area(b) > 0:
            return clip_box01(b)

    boxes: List[List[float]] = []
    # Prefer explicit C2 detections when available (e.g., person detector output).
    for d in feat_rec.get("c2_det", []) or []:
        if not isinstance(d, dict):
            continue
        b = d.get("box")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            boxes.append(norm_box_xyxy(b, width, height))
    for d in feat_rec.get("c2_seg", []) or []:
        b = d.get("box")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            boxes.append(norm_box_xyxy(b, width, height))
    for d in feat_rec.get("c3_pose", []) or []:
        b = d.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            boxes.append(norm_box_xyxy(b, width, height))

    if not boxes:
        return [0.25, 0.25, 0.75, 0.75]

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return clip_box01([x1, y1, x2, y2])


def collect_c3_info(feat_rec: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    c3 = feat_rec.get("c3_pose", []) or []
    face_boxes: List[List[float]] = []
    head_y_norm: List[float] = []
    gaze_entries: List[Dict[str, Any]] = []
    keypoints_norm: List[List[List[float]]] = []

    for human in c3:
        human_face_box: Optional[List[float]] = None

        # Face boxes
        face = human.get("face") if isinstance(human.get("face"), dict) else None
        if face is not None:
            fb: Optional[List[float]] = None
            # Prefer pixel bbox -> normalize by current image size (candidate scale).
            fb_px = face.get("bbox")
            if isinstance(fb_px, (list, tuple)) and len(fb_px) == 4:
                fb = norm_box_xyxy(fb_px, width, height)
            if fb is None:
                fb_n = face.get("bbox_norm")
                if isinstance(fb_n, (list, tuple)) and len(fb_n) == 4:
                    fb = clip_box01(fb_n)
            if isinstance(fb, (list, tuple)) and len(fb) == 4 and box_area(fb) > 0:
                fb = clip_box01(fb)
                human_face_box = [float(v) for v in fb]
                face_boxes.append(human_face_box)
                fy1, fy2 = float(fb[1]), float(fb[3])
                fh = max(1e-6, fy2 - fy1)
                head_y = clamp(fy1 - 0.15 * fh, 0.0, 1.0)
                head_y_norm.append(head_y)

        # Keypoints
        kps = human.get("keypoints")
        if isinstance(kps, list) and kps:
            kn: List[List[float]] = []
            for kp in kps:
                if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                    continue
                x = safe_float(kp[0]) / max(1.0, float(width))
                y = safe_float(kp[1]) / max(1.0, float(height))
                c = safe_float(kp[2])
                kn.append([clamp(x, 0.0, 1.0), clamp(y, 0.0, 1.0), c])
            if kn:
                keypoints_norm.append(kn)
                # Fallback head y from top facial keypoints if face not available.
                top_candidates: List[float] = []
                for idx in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE):
                    if idx < len(kn) and kn[idx][2] >= 0.05:
                        top_candidates.append(kn[idx][1])
                if top_candidates:
                    head_y_norm.append(clamp(min(top_candidates) - 0.04, 0.0, 1.0))

        # Gaze/headpose proxy
        hp = human.get("headpose_gaze") if isinstance(human.get("headpose_gaze"), dict) else {}
        gaze_dir = str(hp.get("gaze_dir", "unknown"))
        conf = safe_float(hp.get("conf", 0.0))

        # Anchor x for lookroom: face center preferred, else human bbox center.
        anchor_x: Optional[float] = None
        anchor_y: Optional[float] = None
        if human_face_box is not None:
            fx1, fy1, fx2, fy2 = human_face_box
            anchor_x = 0.5 * (fx1 + fx2)
            anchor_y = 0.5 * (fy1 + fy2)
        else:
            hb = human.get("bbox")
            if isinstance(hb, (list, tuple)) and len(hb) == 4:
                hb_n = norm_box_xyxy(hb, width, height)
                anchor_x, anchor_y = box_center(hb_n)

        if anchor_x is not None and anchor_y is not None:
            gaze_entries.append(
                {
                    "gaze_dir": gaze_dir,
                    "conf": conf,
                    "anchor_x": clamp(anchor_x, 0.0, 1.0),
                    "anchor_y": clamp(anchor_y, 0.0, 1.0),
                    "yaw_proxy": safe_float(hp.get("yaw_proxy", 0.0)),
                }
            )

    return {
        "face_boxes": face_boxes,
        "head_y_norm": head_y_norm,
        "gaze_entries": gaze_entries,
        "keypoints_norm": keypoints_norm,
        "num_people": len(c3),
    }


def collect_c5_info(feat_rec: Dict[str, Any]) -> Dict[str, Any]:
    c5 = feat_rec.get("c5_geom")
    if isinstance(c5, list):
        c5 = safe_first(c5)
    if not isinstance(c5, dict):
        return {"horizon_y_norm": None, "horizon_conf": 0.0, "roll_deg": None, "symmetry": 0.0}

    hr = c5.get("horizon_roll") if isinstance(c5.get("horizon_roll"), dict) else {}
    sym = c5.get("symmetry") if isinstance(c5.get("symmetry"), dict) else {}
    return {
        "horizon_y_norm": hr.get("horizon_y_norm"),
        "horizon_conf": safe_float(hr.get("conf", 0.0)),
        "roll_deg": hr.get("roll_deg"),
        "symmetry": clamp(safe_float(sym.get("score", 0.0)), 0.0, 1.0),
    }


def joint_points_from_keypoints(kps: Sequence[Sequence[float]]) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}

    def pick(idx: int, name: str) -> None:
        if idx < len(kps):
            x, y, c = safe_float(kps[idx][0]), safe_float(kps[idx][1]), safe_float(kps[idx][2])
            out[name] = (x, y, c)

    pick(KP_LEFT_WRIST, "wrist_l")
    pick(KP_RIGHT_WRIST, "wrist_r")
    pick(KP_LEFT_KNEE, "knee_l")
    pick(KP_RIGHT_KNEE, "knee_r")
    pick(KP_LEFT_ANKLE, "ankle_l")
    pick(KP_RIGHT_ANKLE, "ankle_r")

    # neck proxy: midpoint of shoulders
    if KP_LEFT_SHOULDER < len(kps) and KP_RIGHT_SHOULDER < len(kps):
        lx, ly, lc = kps[KP_LEFT_SHOULDER]
        rx, ry, rc = kps[KP_RIGHT_SHOULDER]
        c = min(safe_float(lc), safe_float(rc))
        out["neck"] = (0.5 * (safe_float(lx) + safe_float(rx)), 0.5 * (safe_float(ly) + safe_float(ry)), c)

    return out


def point_border_distance(pt: Tuple[float, float], crop: Sequence[float]) -> float:
    x, y = pt
    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    return min(x - x1, y - y1, x2 - x, y2 - y)


def eval_joint_cut(
    keypoints_norm: Sequence[Sequence[Sequence[float]]],
    crop: Sequence[float],
    margin_alpha: float,
) -> Dict[str, Any]:
    w, h = box_wh(crop)
    margin = float(margin_alpha) * min(w, h)

    near_count = 0
    severe_count = 0
    total_count = 0
    per_joint: Dict[str, bool] = {k: False for k in CRITICAL_JOINTS}

    for kps in keypoints_norm:
        joints = joint_points_from_keypoints(kps)
        for name, (x, y, conf) in joints.items():
            if name not in per_joint:
                continue
            if conf < 0.05:
                continue
            total_count += 1
            d = point_border_distance((x, y), crop)
            if d < margin:
                near_count += 1
                per_joint[name] = True
            if d < 0.0:
                severe_count += 1
                per_joint[name] = True

    if total_count <= 0:
        return {
            "joint_cut_score": 0.0,
            "near_count": 0,
            "severe_count": 0,
            "total_count": 0,
            "violated": [],
        }

    score = (near_count + 1.5 * severe_count) / total_count
    violated = [k for k, v in per_joint.items() if v]
    return {
        "joint_cut_score": float(score),
        "near_count": int(near_count),
        "severe_count": int(severe_count),
        "total_count": int(total_count),
        "violated": violated,
    }


def face_cut_flags(face_boxes: Sequence[Sequence[float]], crop: Sequence[float]) -> Tuple[bool, int, int]:
    if not face_boxes:
        return False, 0, 0
    kept = 0
    for fb in face_boxes:
        if box_inside(fb, crop):
            kept += 1
    total = len(face_boxes)
    return kept < total, kept, total


def subject_border_touch(subject_box: Sequence[float], crop: Sequence[float], margin_alpha: float = 0.03) -> bool:
    if inter_area(subject_box, crop) <= 0.0:
        return True
    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    sw, sh = box_wh(crop)
    m = margin_alpha * min(sw, sh)

    sx1, sy1, sx2, sy2 = [safe_float(v) for v in subject_box]
    d = min(sx1 - x1, sy1 - y1, x2 - sx2, y2 - sy2)
    return d < m


def subject_coverage(subject_box: Sequence[float], crop: Sequence[float]) -> Tuple[float, float]:
    isect = inter_area(subject_box, crop)
    s_area = box_area(subject_box)
    c_area = max(1e-8, box_area(crop))
    cov = isect / max(1e-8, s_area)
    subj_area_ratio = isect / c_area
    return clamp(cov, 0.0, 1.0), clamp(subj_area_ratio, 0.0, 1.0)


def compute_headroom_term(
    crop: Sequence[float],
    head_y_norm_values: Sequence[float],
    headroom_range: Tuple[float, float, float],
    sigma_h: float,
    gamma_h: float,
) -> Dict[str, Any]:
    hmin, htar, hmax = headroom_range
    if not head_y_norm_values:
        return {
            "value": None,
            "score": 0.0,
            "pass": True,
            "target_range": [round(hmin, 4), round(hmax, 4)],
            "target": round(htar, 4),
        }

    y_head = min(float(v) for v in head_y_norm_values)
    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    hh = max(1e-8, y2 - y1)
    r = (y_head - y1) / hh
    overflow = max(0.0, hmin - r, r - hmax)
    score = -abs(r - htar) / max(1e-8, sigma_h) - gamma_h * overflow
    ok = (hmin <= r <= hmax)
    return {
        "value": round(float(r), 6),
        "score": float(score),
        "pass": bool(ok),
        "target_range": [round(hmin, 4), round(hmax, 4)],
        "target": round(htar, 4),
    }


def compute_lookroom_term(
    crop: Sequence[float],
    gaze_entries: Sequence[Dict[str, Any]],
    lookroom_range: Tuple[Optional[float], Optional[float], Optional[float]],
    sigma_l: float,
    gamma_l: float,
) -> Dict[str, Any]:
    lmin, ltar, lmax = lookroom_range
    if lmin is None or ltar is None or lmax is None:
        return {
            "value": None,
            "score": 0.0,
            "pass": True,
            "target_range": None,
            "target": None,
            "gaze_dir": "n/a",
            "gaze_conf": 0.0,
        }

    if not gaze_entries:
        return {
            "value": None,
            "score": 0.0,
            "pass": True,
            "target_range": [round(lmin, 4), round(lmax, 4)],
            "target": round(ltar, 4),
            "gaze_dir": "unknown",
            "gaze_conf": 0.0,
        }

    best = max(gaze_entries, key=lambda x: safe_float(x.get("conf", 0.0)))
    gaze_dir = str(best.get("gaze_dir", "unknown"))
    conf = safe_float(best.get("conf", 0.0))

    # If direction is center/unknown, lookroom is not activated.
    if gaze_dir not in {"left", "right"}:
        return {
            "value": None,
            "score": 0.0,
            "pass": True,
            "target_range": [round(lmin, 4), round(lmax, 4)],
            "target": round(ltar, 4),
            "gaze_dir": gaze_dir,
            "gaze_conf": round(conf, 4),
        }

    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    ax = clamp(safe_float(best.get("anchor_x", 0.5)), x1, x2)
    m_l = max(1e-8, ax - x1)
    m_r = max(1e-8, x2 - ax)

    if gaze_dir == "right":
        fwd = m_r
        back = m_l
    else:
        fwd = m_l
        back = m_r

    r = (fwd + 1e-6) / (back + 1e-6)
    overflow = max(0.0, lmin - r, r - lmax)
    score = -abs(r - ltar) / max(1e-8, sigma_l) - gamma_l * overflow
    ok = (lmin <= r <= lmax)

    return {
        "value": round(float(r), 6),
        "score": float(score),
        "pass": bool(ok),
        "target_range": [round(lmin, 4), round(lmax, 4)],
        "target": round(ltar, 4),
        "gaze_dir": gaze_dir,
        "gaze_conf": round(conf, 4),
    }


def compute_horizon_reward(
    crop: Sequence[float],
    horizon_y_norm: Optional[float],
    horizon_conf: float,
    conf_thr: float,
) -> Dict[str, Any]:
    if horizon_y_norm is None or not math.isfinite(float(horizon_y_norm)):
        return {
            "dist": None,
            "reward": 0.0,
            "active": False,
            "value": None,
            "pass": True,
        }

    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    hh = max(1e-8, y2 - y1)
    y_local = (float(horizon_y_norm) - y1) / hh
    d = min(abs(y_local - (1.0 / 3.0)), abs(y_local - (2.0 / 3.0)))
    active = horizon_conf >= conf_thr
    reward = reward_dist(d, 0.08) if active else 0.0
    return {
        "dist": float(d),
        "reward": float(reward),
        "active": bool(active),
        "value": float(y_local),
        "pass": bool(d <= 0.08) if active else True,
    }


def edge_term(subject_box: Sequence[float], crop: Sequence[float], margin_alpha: float = 0.03) -> float:
    if inter_area(subject_box, crop) <= 0.0:
        return -1.0

    sx1, sy1, sx2, sy2 = [safe_float(v) for v in subject_box]
    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    cw, ch = box_wh(crop)
    m = margin_alpha * min(cw, ch)

    d_left = sx1 - x1
    d_top = sy1 - y1
    d_right = x2 - sx2
    d_bottom = y2 - sy2
    d = min(d_left, d_top, d_right, d_bottom)

    # Negative value means penalty (as in spec).
    return -max(0.0, m - d) / max(1e-8, m)


def effective_comp_weights(
    cfg: TeacherScorerConfig,
    shot_type: str,
    portrait_category: str,
    flags: Dict[str, bool],
    symmetry_score: float,
    horizon_active: bool,
) -> Dict[str, float]:
    w_third = cfg.w_third
    w_phi = cfg.w_phi
    w_center = cfg.w_center
    w_hor = cfg.w_horizon if horizon_active else 0.0

    if portrait_category in {"formal_id", "corporate"}:
        w_center += 0.20
        w_third -= 0.10
        w_phi -= 0.05
    if portrait_category == "lifestyle":
        w_third += 0.10
        w_phi += 0.05
        w_center -= 0.05

    # Symmetry-dependent center bias (v1.6 guideline).
    w_center += 0.30 * clamp(symmetry_score, 0.0, 1.0)

    if flags.get("is_landscape_scene", False):
        w_hor += 0.20
    if shot_type == "group":
        w_center += 0.05

    vals = {
        "third": max(0.0, w_third),
        "phi": max(0.0, w_phi),
        "center": max(0.0, w_center),
        "horizon": max(0.0, w_hor),
    }
    s = sum(vals.values())
    if s <= 1e-8:
        return {"third": 0.5, "phi": 0.0, "center": 0.5, "horizon": 0.0}
    return {k: v / s for k, v in vals.items()}


def effective_lambdas(
    cfg: TeacherScorerConfig,
    shot_type: str,
    flags: Dict[str, bool],
    has_people: bool,
) -> Dict[str, float]:
    lam = {
        "cov": cfg.lambda_cov,
        "cut": cfg.lambda_cut,
        "text": cfg.lambda_text,
        "comp": cfg.lambda_comp,
        "hr": cfg.lambda_hr,
        "lr": cfg.lambda_lr,
        "sym": cfg.lambda_sym,
        "ctx": cfg.lambda_ctx,
        "cs": cfg.lambda_cs,
    }

    if not has_people:
        lam["hr"] = 0.0
        lam["lr"] = 0.0

    if flags.get("has_copy_space", False):
        lam["cs"] += 0.20
        lam["ctx"] += 0.10

    if flags.get("is_landscape_scene", False):
        lam["comp"] += 0.15
        lam["sym"] += 0.05
        lam["hr"] = 0.0
        lam["lr"] = 0.0

    if shot_type == "group":
        lam["lr"] = min(lam["lr"], 0.10)

    if flags.get("is_product", False) or flags.get("is_isolated_packshot", False):
        lam["ctx"] -= 0.10
        lam["comp"] -= 0.10
        lam["cut"] += 0.15

    for k in lam:
        lam[k] = max(0.0, float(lam[k]))
    return lam


def apply_expensive_score(
    candidate: Dict[str, Any],
    cfg: TeacherScorerConfig,
    w_area: float,
    expensive_signal: Optional[Dict[str, float]],
) -> None:
    comps = candidate.get("scores", {}).get("components", {})
    cov = safe_float(comps.get("cov", 0.0))
    p_cut = safe_float(comps.get("p_cut", 0.0))
    p_text = safe_float(comps.get("p_text", 0.0))
    r_edge = safe_float(comps.get("r_edge", 0.0))
    area = safe_float(candidate.get("area_ratio", 0.0))

    if expensive_signal is not None:
        a_raw = safe_float(expensive_signal.get("aesthetic_raw", 0.0))
        a_norm = clamp(safe_float(expensive_signal.get("aesthetic_norm", 0.0)), 0.0, 1.0)
        ca_val = clamp(safe_float(expensive_signal.get("cosine_img_text", 0.0)), -1.0, 1.0)
        source = "real"
    else:
        a_raw = None
        a_norm = clamp(safe_float(comps.get("aesthetic_proxy", 0.0)), 0.0, 1.0)
        ca_val = clamp(safe_float(comps.get("ca_proxy", 0.0)), -1.0, 1.0)
        source = "proxy"

    exp_score = (
        cfg.w_a * a_norm
        + cfg.w_ca * ca_val
        + cfg.w_cov * cov
        - cfg.w_cut * p_cut
        - cfg.w_text * p_text
        + cfg.w_edge * r_edge
    )
    final_score = exp_score + float(w_area) * math.log(max(1e-8, area))

    candidate["scores"]["expensive"] = float(exp_score)
    candidate["scores"]["final"] = float(final_score)
    comps["aesthetic_raw"] = None if a_raw is None else float(a_raw)
    comps["aesthetic_norm"] = float(a_norm)
    comps["cosine_img_text"] = float(ca_val)
    comps["expensive_source"] = source


def _has_hard_tag(candidate: Dict[str, Any], tag: str) -> bool:
    return str(tag) in set(str(x) for x in (candidate.get("hard_reject_tags") or []))


def _is_structural_valid(candidate: Dict[str, Any]) -> bool:
    return (not _has_hard_tag(candidate, "area_violation")) and (not _has_hard_tag(candidate, "ar_violation"))


def _is_face_safe(candidate: Dict[str, Any]) -> bool:
    return not _has_hard_tag(candidate, "face_cut")


def _relax_joint_only_hard_reject(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Relax strict hard reject only for candidates blocked solely by `joint_cutoff`.

    Structural violations (area/ar) and face cut remain hard.
    """
    if not bool(candidate.get("hard_reject", False)):
        return candidate

    hard_tags = set(str(x) for x in (candidate.get("hard_reject_tags") or []))
    if any(t in hard_tags for t in ("area_violation", "ar_violation", "face_cut")):
        return None
    if "joint_cutoff" not in hard_tags:
        return None

    c = copy.deepcopy(candidate)
    c["hard_reject"] = False
    c["fallback"] = {
        "joint_relaxed": True,
        "hard_reject_tags_strict": sorted(list(hard_tags)),
    }
    return c


def compute_candidate_scores(
    candidate: Dict[str, Any],
    target_ar: float,
    image_ar: float,
    subject_box: Sequence[float],
    subject_centroid: Tuple[float, float],
    c3_info: Dict[str, Any],
    c5_info: Dict[str, Any],
    route: Dict[str, Any],
    cfg: TeacherScorerConfig,
) -> Dict[str, Any]:
    crop = clip_box01(candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))
    area = box_area(crop)
    c_w, c_h = box_wh(crop)
    ar = (c_w * image_ar / max(1e-8, c_h)) if c_h > 0 else 0.0

    hard_reject_tags: List[str] = []

    # Structural hard checks.
    if not (cfg.area_min <= area <= cfg.area_max):
        hard_reject_tags.append("area_violation")
    if abs(ar - target_ar) > cfg.ar_hard_eps:
        hard_reject_tags.append("ar_violation")

    f_cut, f_kept, f_total = face_cut_flags(c3_info["face_boxes"], crop)
    if cfg.hard_face_rule and f_total > 0 and f_cut:
        hard_reject_tags.append("face_cut")

    joint = eval_joint_cut(
        keypoints_norm=c3_info["keypoints_norm"],
        crop=crop,
        margin_alpha=cfg.severe_kp_margin_alpha,
    )
    if joint["severe_count"] >= cfg.hard_joint_reject_count:
        hard_reject_tags.append("joint_cutoff")

    # Cheap terms
    cov, subj_area_ratio = subject_coverage(subject_box, crop)
    subj_touch = subject_border_touch(subject_box, crop)

    p_cut = (
        cfg.alpha_face_cut * (1.0 if f_cut else 0.0)
        + cfg.alpha_joint_cut * joint["joint_cut_score"]
        + cfg.alpha_subject_border * (1.0 if subj_touch else 0.0)
    )

    # OCR excluded.
    p_text = 0.0

    c_local = local_coords(crop, subject_centroid[0], subject_centroid[1])
    d_third = distance_to_thirds(c_local)
    d_phi = distance_to_phi(c_local)
    d_center = distance_to_center(c_local)

    r_third = reward_dist(d_third, 0.18)
    r_phi = reward_dist(d_phi, 0.16)
    r_center = reward_dist(d_center, 0.15)

    horizon = compute_horizon_reward(
        crop=crop,
        horizon_y_norm=c5_info["horizon_y_norm"],
        horizon_conf=c5_info["horizon_conf"],
        conf_thr=cfg.horizon_conf_thr,
    )

    comp_w = effective_comp_weights(
        cfg=cfg,
        shot_type=route["shot_type"],
        portrait_category=route["portrait_category"],
        flags=route["flags"],
        symmetry_score=c5_info["symmetry"],
        horizon_active=bool(horizon["active"]),
    )

    r_comp = (
        comp_w["third"] * r_third
        + comp_w["phi"] * r_phi
        + comp_w["center"] * r_center
        + comp_w["horizon"] * horizon["reward"]
    )

    hr = compute_headroom_term(
        crop=crop,
        head_y_norm_values=c3_info["head_y_norm"],
        headroom_range=route["headroom_range"],
        sigma_h=cfg.sigma_h,
        gamma_h=cfg.gamma_h,
    )
    lr = compute_lookroom_term(
        crop=crop,
        gaze_entries=c3_info["gaze_entries"],
        lookroom_range=route["lookroom_range"],
        sigma_l=cfg.sigma_l,
        gamma_l=cfg.gamma_l,
    )

    sym = c5_info["symmetry"]

    ctx_lo, ctx_hi = route["context_range"]
    ctx_target = 0.5 * (ctx_lo + ctx_hi)
    sigma_a = max(0.05, 0.5 * (ctx_hi - ctx_lo))
    r_context = -abs(subj_area_ratio - ctx_target) / sigma_a
    ctx_pass = ctx_lo <= subj_area_ratio <= ctx_hi

    copyspace_ratio = clamp(1.0 - subj_area_ratio, 0.0, 1.0)
    r_copyspace = copyspace_ratio if route["flags"].get("has_copy_space", False) else 0.0

    lam = route["lambdas"]

    cheap = (
        lam["cov"] * cov
        - lam["cut"] * p_cut
        - lam["text"] * p_text
        + lam["comp"] * r_comp
        + lam["hr"] * hr["score"]
        + lam["lr"] * lr["score"]
        + lam["sym"] * sym
        + lam["ctx"] * r_context
        + lam["cs"] * r_copyspace
    )

    # Expensive priors (used as fallback when real expensive model is unavailable).
    context_fit = reward_dist(abs(subj_area_ratio - ctx_target), max(0.05, sigma_a))
    aesthetic_proxy = (
        0.45 * r_comp
        + 0.20 * sym
        + 0.20 * cov
        + 0.15 * context_fit
        - 0.30 * min(1.0, p_cut)
    )
    ca_proxy = 0.65 * cov + 0.35 * context_fit
    r_edge = edge_term(subject_box=subject_box, crop=crop)

    # Deterministic tags/checks for rationale.
    why_tags: List[str] = []
    if d_third <= 0.18:
        why_tags.append("rule_of_thirds")
    if d_phi <= 0.15:
        why_tags.append("phi_grid")
    if d_center <= 0.15:
        why_tags.append("centered_subject")
    if sym >= 0.75:
        why_tags.append("symmetry")

    if hr["value"] is not None:
        why_tags.append("headroom_ok" if hr["pass"] else "headroom_violation")
    if lr["gaze_dir"] in {"left", "right"}:
        why_tags.append("lookroom_ok" if lr["pass"] else "lookroom_violation")
    if horizon["active"]:
        why_tags.append("horizon_on_third" if horizon["pass"] else "horizon_off_third")

    why_tags.append("context_preserved" if ctx_pass else "context_loss")
    if not f_cut:
        why_tags.append("avoid_face_cut")
    if joint["joint_cut_score"] <= 0.10:
        why_tags.append("avoid_person_cut")
    if route["flags"].get("has_copy_space", False):
        why_tags.append("copy_space_kept" if copyspace_ratio >= 0.25 else "copy_space_lost")
    why_tags.append("ar_fits_well" if abs(ar - target_ar) <= cfg.ar_hard_eps else "ar_mismatch")
    why_tags.append("tight_crop" if area < 0.35 else ("wide_crop" if area > 0.75 else "balanced_crop"))

    reject_tags = list(hard_reject_tags)
    if f_cut and "face_cut" not in reject_tags:
        reject_tags.append("face_cut")
    if joint["joint_cut_score"] > 0.35 and "joint_cutoff" not in reject_tags:
        reject_tags.append("joint_cutoff")
    if lr["gaze_dir"] in {"left", "right"} and not lr["pass"]:
        reject_tags.append("lookroom_violation")

    out = {
        "candidate_id": candidate.get("candidate_id"),
        "bbox_norm_xyxy": crop,
        "area_ratio": round(area, 6),
        "source": candidate.get("source"),
        "source_types": candidate.get("source_types", []),
        "must_keep": bool(candidate.get("must_keep", False)),
        "priority": safe_float(candidate.get("priority", 0.0)),
        "hard_reject": len(hard_reject_tags) > 0,
        "hard_reject_strict": len(hard_reject_tags) > 0,
        "hard_reject_tags": hard_reject_tags,
        "scores": {
            "cheap": float(cheap),
            "expensive": 0.0,
            "final": 0.0,
            "components": {
                "cov": float(cov),
                "p_cut": float(p_cut),
                "p_text": float(p_text),
                "r_comp": float(r_comp),
                "r_headroom": float(hr["score"]),
                "r_lookroom": float(lr["score"]),
                "r_sym": float(sym),
                "r_context": float(r_context),
                "r_copyspace": float(r_copyspace),
                "r_edge": float(r_edge),
                "aesthetic_proxy": float(aesthetic_proxy),
                "ca_proxy": float(ca_proxy),
                "aesthetic_raw": None,
                "aesthetic_norm": None,
                "cosine_img_text": None,
                "expensive_source": "pending",
            },
        },
        "flags": {
            "face_cut": bool(f_cut),
            "joint_cutoff_score": float(joint["joint_cut_score"]),
            "joint_severe_count": int(joint["severe_count"]),
            "subject_touch_border": bool(subj_touch),
            "subject_coverage": float(cov),
            "subject_area": float(subj_area_ratio),
            "ar_error": float(abs(ar - target_ar)),
        },
        "composition_checks": {
            "headroom": {
                "value": hr["value"],
                "target_range": hr["target_range"],
                "pass": bool(hr["pass"]),
            },
            "lookroom": {
                "value": lr["value"],
                "target_range": lr["target_range"],
                "pass": bool(lr["pass"]),
                "gaze_dir": lr["gaze_dir"],
            },
            "horizon": {
                "y": horizon["value"],
                "roll_deg": c5_info["roll_deg"],
                "conf": c5_info["horizon_conf"],
                "third_dist": horizon["dist"],
                "pass": bool(horizon["pass"]),
            },
            "symmetry": {
                "value": float(sym),
                "pass": bool(sym >= 0.50),
            },
            "context": {
                "subject_area": float(subj_area_ratio),
                "target_range": [round(ctx_lo, 4), round(ctx_hi, 4)],
                "pass": bool(ctx_pass),
            },
            "cutoff": {
                "joint_cutoff_score": float(joint["joint_cut_score"]),
                "joint_severe_count": int(joint["severe_count"]),
                "joint_violated": joint["violated"],
                "face_cut": bool(f_cut),
                "pass": bool((not f_cut) and (joint["joint_cut_score"] <= 0.35)),
            },
            "text": {
                "text_keep_ratio": None,
                "pass": True,
                "skipped": "ocr_disabled",
            },
        },
        "why_tags": sorted(set(why_tags)),
        "reject_tags": sorted(set(reject_tags)),
    }
    apply_expensive_score(
        candidate=out,
        cfg=cfg,
        w_area=float(route["w_area"]),
        expensive_signal=None,
    )
    return out


def pick_baseline_candidates(cands: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not cands:
        return None, None

    def key_priority(c: Dict[str, Any]) -> Tuple[float, float]:
        return (safe_float(c.get("priority", 0.0)), safe_float(c.get("area_ratio", 0.0)))

    baseline_full = [c for c in cands if str(c.get("source", "")).startswith("baseline_full")]
    baseline_center = [c for c in cands if str(c.get("source", "")) == "baseline_maxarea_center"]
    baseline_subject = [c for c in cands if str(c.get("source", "")) == "baseline_maxarea_subject"]
    baseline_slide = [c for c in cands if str(c.get("source", "")).startswith("baseline_maxarea_slide")]
    must_keep = [c for c in cands if bool(c.get("must_keep", False))]

    baseline = None
    for pool in (baseline_full, baseline_center, baseline_subject, baseline_slide, must_keep):
        if pool:
            baseline = max(pool, key=key_priority)
            break
    if baseline is None:
        baseline = max(cands, key=key_priority)

    # Overcrop reference prefers center max-area if available.
    if baseline_center:
        ref_center = max(baseline_center, key=key_priority)
    elif baseline:
        ref_center = baseline
    else:
        ref_center = max(cands, key=lambda x: safe_float(x.get("area_ratio", 0.0)))

    return baseline, ref_center


def normalize_final_scores(cands: Sequence[Dict[str, Any]]) -> None:
    vals = [safe_float(c["scores"].get("final", 0.0)) for c in cands]
    if not vals:
        return
    mn = min(vals)
    mx = max(vals)
    if mx - mn < 1e-9:
        for c in cands:
            c["scores"]["final_norm_0_100"] = 50.0
        return
    for c in cands:
        v = safe_float(c["scores"].get("final", 0.0))
        c["scores"]["final_norm_0_100"] = 100.0 * (v - mn) / (mx - mn)


def decide_keep_vs_crop(best: Dict[str, Any], baseline: Dict[str, Any], tau_improve: float) -> Dict[str, Any]:
    b_best = safe_float(best["scores"].get("final", -1e9))
    b_base = safe_float(baseline["scores"].get("final", -1e9))
    delta = b_best - b_base

    if delta < tau_improve:
        src = str(baseline.get("source", ""))
        if src.startswith("baseline_full"):
            decision_type = "keep_full"
        else:
            decision_type = "minimal_crop"
        chosen = baseline
    else:
        decision_type = "crop"
        chosen = best

    return {
        "decision_type": decision_type,
        "delta_improve": float(delta),
        "tau_improve": float(tau_improve),
        "chosen_candidate_id": chosen.get("candidate_id"),
    }


def select_topk_diverse(
    sorted_cands: Sequence[Dict[str, Any]],
    k: int,
    tau_div: float,
    force_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if k <= 0:
        return []

    by_id = {str(c.get("candidate_id")): c for c in sorted_cands}
    selected: List[Dict[str, Any]] = []

    if force_ids:
        for cid in force_ids:
            c = by_id.get(str(cid))
            if c is None:
                continue
            if any(str(x.get("candidate_id")) == str(cid) for x in selected):
                continue
            selected.append(c)
            if len(selected) >= k:
                return selected[:k]

    for c in sorted_cands:
        cid = str(c.get("candidate_id"))
        if any(str(x.get("candidate_id")) == cid for x in selected):
            continue
        if selected:
            if max(iou_xyxy(c["bbox_norm_xyxy"], s["bbox_norm_xyxy"]) for s in selected) >= tau_div:
                continue
        selected.append(c)
        if len(selected) >= k:
            return selected[:k]

    # Fill remainder without diversity gating.
    if len(selected) < k:
        for c in sorted_cands:
            cid = str(c.get("candidate_id"))
            if any(str(x.get("candidate_id")) == cid for x in selected):
                continue
            selected.append(c)
            if len(selected) >= k:
                break

    return selected[:k]


def select_hard_negatives(cands: Sequence[Dict[str, Any]], max_n: int = 5) -> List[Dict[str, Any]]:
    mids = [c for c in cands if 40.0 <= safe_float(c["scores"].get("final_norm_0_100", -1)) <= 60.0]
    mids.sort(key=lambda x: safe_float(x["scores"].get("final", 0.0)), reverse=True)
    return mids[:max_n]


def candidate_brief(c: Dict[str, Any], rank: Optional[int] = None) -> Dict[str, Any]:
    out = {
        "candidate_id": c.get("candidate_id"),
        "bbox_norm_xyxy": [round(float(v), 6) for v in c.get("bbox_norm_xyxy", [0, 0, 1, 1])],
        "area_ratio": round(safe_float(c.get("area_ratio", 0.0)), 6),
        "source": c.get("source"),
        "must_keep": bool(c.get("must_keep", False)),
        "hard_reject": bool(c.get("hard_reject", False)),
        "hard_reject_strict": bool(c.get("hard_reject_strict", c.get("hard_reject", False))),
        "scores": {
            "cheap": round(safe_float(c["scores"].get("cheap", 0.0)), 6),
            "expensive": round(safe_float(c["scores"].get("expensive", 0.0)), 6),
            "final": round(safe_float(c["scores"].get("final", 0.0)), 6),
            "final_norm_0_100": round(safe_float(c["scores"].get("final_norm_0_100", 0.0)), 3),
            "components": c["scores"].get("components", {}),
        },
        "flags": c.get("flags", {}),
        "why_tags": c.get("why_tags", []),
        "reject_tags": c.get("reject_tags", []),
        "composition_checks": c.get("composition_checks", {}),
    }
    if rank is not None:
        out["rank"] = int(rank)
    return out


def update_stats(
    stats: ARStats,
    image_ar: float,
    target_ar: float,
    decision_type: str,
    delta_improve: float,
    topk: Sequence[Dict[str, Any]],
    ref_center_area: float,
    route: Dict[str, Any],
    fallback: Optional[Dict[str, Any]],
    cfg: TeacherScorerConfig,
) -> None:
    stats.images += 1
    stats.delta_improve.append(float(delta_improve))

    if isinstance(fallback, dict) and bool(fallback.get("activated", False)):
        stats.fallback_activated_count += 1
        mode = str(fallback.get("mode", "unknown"))
        stats.fallback_mode_counts[mode] = int(stats.fallback_mode_counts.get(mode, 0)) + 1

    if decision_type == "keep_full":
        stats.keep_full_count += 1
    elif decision_type == "minimal_crop":
        stats.minimal_crop_count += 1
    else:
        stats.crop_count += 1

    if not topk:
        return

    t1 = topk[0]
    area = safe_float(t1.get("area_ratio", 0.0))
    if ref_center_area > 1e-8 and area < cfg.overcrop_factor * ref_center_area:
        stats.overcrop_count += 1

    flags = t1.get("flags", {})
    checks = t1.get("composition_checks", {})
    if bool(flags.get("face_cut", False)):
        stats.top1_face_cut_count += 1
    if safe_float(flags.get("joint_cutoff_score", 0.0)) > 0.35:
        stats.top1_joint_cut_count += 1
    if safe_float(flags.get("subject_coverage", 0.0)) < cfg.subject_coverage_fail_thr:
        stats.top1_subject_cov_fail_count += 1

    if len(topk) >= 2:
        iou12 = iou_xyxy(topk[0].get("bbox_norm_xyxy", [0, 0, 1, 1]), topk[1].get("bbox_norm_xyxy", [0, 0, 1, 1]))
        stats.top1_top2_iou.append(float(iou12))

    # Portrait checks
    if is_portrait_route(route["shot_type"], route["num_people"]):
        stats.portrait_subset_count += 1
        headroom = checks.get("headroom", {}) if isinstance(checks.get("headroom"), dict) else {}
        if headroom.get("value") is not None and not bool(headroom.get("pass", True)):
            stats.headroom_violation_count += 1

        lookroom = checks.get("lookroom", {}) if isinstance(checks.get("lookroom"), dict) else {}
        gaze_dir = str(lookroom.get("gaze_dir", "unknown"))
        if gaze_dir in {"left", "right"}:
            stats.lookroom_subset_count += 1
            if not bool(lookroom.get("pass", True)):
                stats.lookroom_violation_count += 1

    # Horizon / roll checks
    horizon = checks.get("horizon", {}) if isinstance(checks.get("horizon"), dict) else {}
    h_conf = safe_float(horizon.get("conf", 0.0))
    h_dist = horizon.get("third_dist")
    if h_dist is not None and h_conf >= cfg.horizon_conf_thr:
        stats.horizon_subset_count += 1
        stats.horizon_third_dist.append(float(h_dist))
        if not bool(horizon.get("pass", True)):
            stats.horizon_violation_count += 1

    if route["flags"].get("is_landscape_scene", False):
        roll = horizon.get("roll_deg")
        if roll is not None:
            stats.roll_subset_count += 1
            if abs(safe_float(roll, 0.0)) > cfg.roll_violation_deg:
                stats.roll_violation_count += 1

    # Copy-space preservation
    if route["flags"].get("has_copy_space", False):
        stats.copyspace_subset_count += 1
        context = checks.get("context", {}) if isinstance(checks.get("context"), dict) else {}
        subj_area = safe_float(context.get("subject_area", 0.0))
        if (1.0 - subj_area) >= 0.25:
            stats.copyspace_preserve_count += 1


def summarize_stats(stats: ARStats) -> Dict[str, Any]:
    n = max(1, stats.images)

    out = {
        "images": stats.images,
        "candidates_in_mean": stats.candidates_in_mean,
        "cheap_kept_mean": stats.cheap_kept_mean,
        "keep_full_rate": stats.keep_full_count / n,
        "minimal_crop_rate": stats.minimal_crop_count / n,
        "crop_rate": stats.crop_count / n,
        "overcrop_rate": stats.overcrop_count / n,
        "delta_improve": {
            "mean": mean(stats.delta_improve) if stats.delta_improve else 0.0,
            "p50": percentile(stats.delta_improve, 0.50),
            "p90": percentile(stats.delta_improve, 0.90),
        },
        "face_cut_rate": stats.top1_face_cut_count / n,
        "joint_cut_rate": stats.top1_joint_cut_count / n,
        "subject_coverage_fail_rate": stats.top1_subject_cov_fail_count / n,
        "headroom_violation_rate": (
            stats.headroom_violation_count / max(1, stats.portrait_subset_count)
        ),
        "lookroom_violation_rate": (
            stats.lookroom_violation_count / max(1, stats.lookroom_subset_count)
        ),
        "horizon_third_dist_p90": percentile(stats.horizon_third_dist, 0.90),
        "roll_violation_rate": stats.roll_violation_count / max(1, stats.roll_subset_count),
        "copyspace_preserve_rate": stats.copyspace_preserve_count / max(1, stats.copyspace_subset_count),
        "fallback_rate": stats.fallback_activated_count / n,
        "fallback_mode_counts": dict(stats.fallback_mode_counts),
        "topk_mean_iou_top1_top2": mean(stats.top1_top2_iou) if stats.top1_top2_iou else 0.0,
        "denominators": {
            "portrait_subset": stats.portrait_subset_count,
            "lookroom_subset": stats.lookroom_subset_count,
            "horizon_subset": stats.horizon_subset_count,
            "roll_subset": stats.roll_subset_count,
            "copyspace_subset": stats.copyspace_subset_count,
        },
    }
    return out


def load_feature_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            image_id = str(d.get("image_id", ""))
            if not image_id:
                continue
            out[image_id] = d
    return out


def process_one_image(
    cand_rec: Dict[str, Any],
    feat_rec: Dict[str, Any],
    cfg: TeacherScorerConfig,
    expensive_models: Optional[ExpensiveModels] = None,
    image_pil: Optional["Image.Image"] = None,
    c1_text_embed: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    # Prefer explicit normalization metadata to keep all geometry on the same
    # reference image size (actual TAR image size when available).
    meta_norm = cand_rec.get("meta_norm", {})
    if not isinstance(meta_norm, dict):
        meta_norm = {}

    width = int(
        safe_float(
            meta_norm.get("width_norm_ref", meta_norm.get("width", cand_rec.get("width", 1))),
            1.0,
        )
    )
    height = int(
        safe_float(
            meta_norm.get("height_norm_ref", meta_norm.get("height", cand_rec.get("height", 1))),
            1.0,
        )
    )
    width = max(1, width)
    height = max(1, height)
    image_ar = safe_float(
        meta_norm.get(
            "image_ar_norm_ref",
            cand_rec.get("image_ar", float(width) / max(1.0, float(height))),
        )
    )
    norm_size_source = str(
        meta_norm.get(
            "size_source",
            cand_rec.get("size_source", "unknown"),
        )
    )

    tags = tags_to_tokens(cand_rec.get("tags", []))

    c3_info = collect_c3_info(feat_rec=feat_rec, width=width, height=height)
    c5_info = collect_c5_info(feat_rec=feat_rec)

    c2_person_count = count_c2_person_instances(feat_rec)
    num_people = c3_info["num_people"]
    has_human_evidence = (num_people > 0) or (c2_person_count > 0)

    shot_type = infer_shot_type(tags, has_human_evidence=has_human_evidence)
    portrait_category = infer_portrait_category(tags)
    flags = infer_flags(tags, has_human_evidence=has_human_evidence)

    has_people = num_people > 0

    route = {
        "shot_type": shot_type,
        "portrait_category": portrait_category,
        "flags": flags,
        "num_people": num_people,
        "c2_person_count": c2_person_count,
        "has_human_evidence": bool(has_human_evidence),
        "norm_size_source": norm_size_source,
        "headroom_range": headroom_prior(shot_type, portrait_category, flags),
        "lookroom_range": lookroom_prior(shot_type, flags),
        "context_range": context_target_range(
            shot_type=shot_type,
            flags=flags,
            portrait_category=portrait_category,
            has_human_evidence=bool(has_human_evidence),
        ),
        "tau_improve": tau_improve_for_route(shot_type, flags, num_people),
        "w_area": area_weight_for_route(flags),
        "lambdas": effective_lambdas(cfg, shot_type, flags, has_people),
    }

    subject_box = build_subject_bbox(cand_rec, feat_rec, width=width, height=height)
    subject_centroid = box_center(subject_box)

    results_by_ar: Dict[str, Any] = {}
    stats_for_ar: Dict[str, Dict[str, float]] = {}
    per_ar_tmp: Dict[str, Dict[str, Any]] = {}

    for ar_text, cands in (cand_rec.get("candidates_by_ar", {}) or {}).items():
        target_ar = parse_ar(ar_text)
        if not isinstance(cands, list) or not cands:
            continue

        scored: List[Dict[str, Any]] = []
        for c in cands:
            scored.append(
                compute_candidate_scores(
                    candidate=c,
                    target_ar=target_ar,
                    image_ar=image_ar,
                    subject_box=subject_box,
                    subject_centroid=subject_centroid,
                    c3_info=c3_info,
                    c5_info=c5_info,
                    route=route,
                    cfg=cfg,
                )
            )

        baseline, ref_center = pick_baseline_candidates(scored)

        strict_valid = [c for c in scored if not bool(c.get("hard_reject", False))]
        fallback_info: Dict[str, Any] = {
            "activated": False,
            "mode": "none",
            "strict_valid_count": int(len(strict_valid)),
            "effective_valid_count": int(len(strict_valid)),
            "joint_relaxed_count": 0,
        }

        valid_effective: List[Dict[str, Any]] = list(strict_valid)
        if not valid_effective:
            relaxed_joint: List[Dict[str, Any]] = []
            for c in scored:
                if (not _is_structural_valid(c)) or (not _is_face_safe(c)):
                    continue
                cc = _relax_joint_only_hard_reject(c)
                if cc is not None and not bool(cc.get("hard_reject", False)):
                    relaxed_joint.append(cc)

            if relaxed_joint:
                valid_effective = relaxed_joint
                fallback_info.update(
                    {
                        "activated": True,
                        "mode": "relaxed_joint_only",
                        "effective_valid_count": int(len(relaxed_joint)),
                        "joint_relaxed_count": int(len(relaxed_joint)),
                    }
                )
            else:
                face_safe_non_struct = [
                    c for c in scored if _is_structural_valid(c) and _is_face_safe(c)
                ]
                if face_safe_non_struct:
                    valid_effective = face_safe_non_struct
                    fallback_info.update(
                        {
                            "activated": True,
                            "mode": "non_structural_face_safe",
                            "effective_valid_count": int(len(face_safe_non_struct)),
                        }
                    )
                else:
                    non_struct = [c for c in scored if _is_structural_valid(c)]
                    if non_struct:
                        valid_effective = non_struct
                        fallback_info.update(
                            {
                                "activated": True,
                                "mode": "non_structural_only",
                                "effective_valid_count": int(len(non_struct)),
                            }
                        )
                    else:
                        valid_effective = list(scored)
                        fallback_info.update(
                            {
                                "activated": True,
                                "mode": "all_structural_invalid",
                                "effective_valid_count": int(len(scored)),
                            }
                        )

        valid_effective.sort(key=lambda x: safe_float(x["scores"].get("cheap", -1e9)), reverse=True)
        cheap_top_m = valid_effective[: max(1, int(cfg.cheap_top_m))]

        # Baseline is included only when it is not hard-invalid,
        # except for fully-failed fallback modes where no clean option exists.
        if baseline is not None:
            baseline_hard = bool(baseline.get("hard_reject", False))
            allow_hard_baseline = fallback_info["mode"] in {"all_structural_invalid", "non_structural_only"}
            if baseline_hard and not allow_hard_baseline:
                baseline = None

        if baseline is not None:
            b_id = str(baseline.get("candidate_id"))
            if all(str(c.get("candidate_id")) != b_id for c in cheap_top_m):
                cheap_top_m.append(baseline)

        expensive_eval_pool = cheap_top_m
        exp_cap = int(cfg.expensive_eval_top_m)
        if exp_cap > 0 and len(expensive_eval_pool) > exp_cap:
            narrowed = list(expensive_eval_pool[:exp_cap])
            if baseline is not None:
                b_id = str(baseline.get("candidate_id", ""))
                if b_id and all(str(c.get("candidate_id", "")) != b_id for c in narrowed):
                    narrowed.append(baseline)
            expensive_eval_pool = narrowed

        per_ar_tmp[ar_text] = {
            "target_ar": target_ar,
            "scored": scored,
            "baseline": baseline,
            "ref_center": ref_center,
            "cheap_top_m": cheap_top_m,
            "expensive_eval_pool": expensive_eval_pool,
            "fallback_info": fallback_info,
        }

    # Real expensive stage on union(M candidates across AR), if resources are ready.
    expensive_real_applied = False
    if expensive_models is not None and image_pil is not None and c1_text_embed is not None:
        union: Dict[str, Dict[str, Any]] = {}
        for tmp in per_ar_tmp.values():
            for c in tmp.get("expensive_eval_pool", tmp["cheap_top_m"]):
                cid = str(c.get("candidate_id", ""))
                if cid:
                    union[cid] = c
        signals = expensive_models.predict_for_candidates(
            image=image_pil,
            candidates=list(union.values()),
            text_embed=c1_text_embed,
            cfg=cfg,
        )
        for cid, cand in union.items():
            apply_expensive_score(
                candidate=cand,
                cfg=cfg,
                w_area=float(route["w_area"]),
                expensive_signal=signals.get(cid),
            )
        expensive_real_applied = True

    # Finalize per-AR ranking/selection.
    for ar_text, tmp in per_ar_tmp.items():
        scored = tmp["scored"]
        baseline = tmp["baseline"]
        ref_center = tmp["ref_center"]
        cheap_top_m = tmp["cheap_top_m"]
        expensive_eval_pool = tmp.get("expensive_eval_pool", cheap_top_m)
        fallback_info = tmp["fallback_info"]
        target_ar = float(tmp["target_ar"])

        normalize_final_scores(cheap_top_m)
        exp_sorted = sorted(cheap_top_m, key=lambda x: safe_float(x["scores"].get("final", -1e9)), reverse=True)
        non_hard_sorted = [c for c in exp_sorted if not bool(c.get("hard_reject", False))]
        rank_pool = non_hard_sorted if non_hard_sorted else exp_sorted

        if rank_pool:
            best = rank_pool[0]
        else:
            best = baseline if baseline is not None else scored[0]
            exp_sorted = [best]
            rank_pool = exp_sorted
            normalize_final_scores(exp_sorted)

        baseline_effective = baseline
        if baseline_effective is None or bool(baseline_effective.get("hard_reject", False)):
            baseline_effective = non_hard_sorted[0] if non_hard_sorted else best
        if baseline_effective is None:
            baseline_effective = best

        decision = decide_keep_vs_crop(
            best=best,
            baseline=baseline_effective,
            tau_improve=route["tau_improve"],
        )

        force_ids = [decision["chosen_candidate_id"], baseline_effective.get("candidate_id")]
        selected_topk = select_topk_diverse(
            sorted_cands=rank_pool,
            k=cfg.top_k,
            tau_div=cfg.tau_div,
            force_ids=force_ids,
        )
        if len(selected_topk) < cfg.top_k and rank_pool is not exp_sorted:
            already = {str(x.get("candidate_id", "")) for x in selected_topk}
            tail = [c for c in exp_sorted if str(c.get("candidate_id", "")) not in already]
            if tail:
                fill = select_topk_diverse(
                    sorted_cands=tail,
                    k=cfg.top_k - len(selected_topk),
                    tau_div=cfg.tau_div,
                    force_ids=None,
                )
                selected_topk.extend(fill)
        selected_topk = selected_topk[: cfg.top_k]

        if selected_topk:
            normalize_final_scores(selected_topk)
        hard_negs = select_hard_negatives(exp_sorted, max_n=5)

        rejected_sorted = sorted(
            [c for c in scored if c.get("hard_reject", False)],
            key=lambda x: safe_float(x["scores"].get("cheap", -1e9)),
            reverse=True,
        )

        ar_result = {
            "target_ar": ar_text,
            "target_ar_value": target_ar,
            "num_input_candidates": len(scored),
            "num_valid_candidates": int(fallback_info.get("strict_valid_count", 0)),
            "num_effective_candidates": int(fallback_info.get("effective_valid_count", 0)),
            "num_cheap_kept": len(cheap_top_m),
            "num_expensive_eval": len(expensive_eval_pool),
            "baseline_candidate": candidate_brief(baseline_effective),
            "best_candidate": candidate_brief(best),
            "keep_policy": {
                "tau_improve": route["tau_improve"],
                "w_area": route["w_area"],
                "force_include_baseline_in_topm": True,
            },
            "fallback": fallback_info,
            "decision": decision,
            "cheap_top_m": [candidate_brief(c) for c in cheap_top_m],
            "selected_topk": [candidate_brief(c, rank=i + 1) for i, c in enumerate(selected_topk)],
            "hard_negatives": [candidate_brief(c) for c in hard_negs],
            "also_considered_rejected": [candidate_brief(c) for c in rejected_sorted[:5]],
            "routing": {
                "shot_type": shot_type,
                "portrait_category": portrait_category,
                "flags": flags,
                "num_people": int(num_people),
                "c2_person_count": int(c2_person_count),
                "has_human_evidence": bool(has_human_evidence),
                "norm_size_source": norm_size_source,
                "headroom_range": route["headroom_range"],
                "lookroom_range": route["lookroom_range"],
                "context_range": route["context_range"],
                "lambdas": route["lambdas"],
            },
            "expensive": {
                "real_applied": bool(expensive_real_applied),
                "uses_aesthetic_predictor": bool(expensive_real_applied),
                "uses_clip_text_alignment": bool(expensive_real_applied),
            },
        }
        results_by_ar[ar_text] = ar_result

        stats_for_ar[ar_text] = {
            "candidates_in": float(len(scored)),
            "cheap_kept": float(len(cheap_top_m)),
            "decision_type": decision["decision_type"],
            "delta_improve": float(decision["delta_improve"]),
            "ref_center_area": safe_float(ref_center.get("area_ratio", 0.0)) if ref_center is not None else 0.0,
        }

    return {
        "image_id": cand_rec.get("image_id"),
        "width": width,
        "height": height,
        "image_ar": image_ar,
        "meta_norm": meta_norm,
        "tags": tags,
        "subject_prior": cand_rec.get("subject_prior", {}),
        "route_global": {
            "shot_type": shot_type,
            "portrait_category": portrait_category,
            "flags": flags,
            "num_people": num_people,
            "c2_person_count": int(c2_person_count),
            "has_human_evidence": bool(has_human_evidence),
            "norm_size_source": norm_size_source,
        },
        "teacher_scorer": {
            "config": asdict(cfg),
            "expensive_real_applied": bool(expensive_real_applied),
            "results_by_ar": results_by_ar,
        },
        "_stats_for_ar": stats_for_ar,
    }


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    if int(args.num_shards) < 1:
        raise ValueError(f"--num_shards must be >= 1 (got {args.num_shards})")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(
            f"--shard_index must satisfy 0 <= shard_index < num_shards "
            f"(got shard_index={args.shard_index}, num_shards={args.num_shards})"
        )

    cfg = TeacherScorerConfig(
        cheap_top_m=max(1, int(args.cheap_top_m)),
        top_k=max(1, int(args.top_k)),
        tau_div=float(args.tau_div),
        use_real_expensive=bool(int(args.use_real_expensive)),
        expensive_eval_top_m=max(0, int(args.expensive_eval_top_m)),
        exp_preprocess_workers=max(0, int(args.exp_preprocess_workers)),
        exp_pin_memory=bool(int(args.exp_pin_memory)),
    )

    cand_path = Path(args.candidates_jsonl)
    feat_path = Path(args.features_jsonl)
    c1_path = Path(args.c1_jsonl) if str(args.c1_jsonl).strip() else None
    parquet_path = Path(args.parquet) if str(args.parquet).strip() else None
    out_jsonl = Path(args.output_jsonl)
    out_overview_json = Path(args.output_overview_json)
    out_overview_csv = Path(args.output_overview_by_ar_csv) if args.output_overview_by_ar_csv else None

    if not cand_path.exists():
        raise FileNotFoundError(f"candidates_jsonl not found: {cand_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"features_jsonl not found: {feat_path}")
    if cfg.use_real_expensive:
        if c1_path is None or not c1_path.exists():
            raise FileNotFoundError("use_real_expensive=1 requires valid --c1_jsonl")
        if str(args.image_dir).strip():
            if not os.path.isdir(str(args.image_dir)):
                raise FileNotFoundError(f"use_real_expensive=1 image_dir not found: {args.image_dir}")
        else:
            if parquet_path is None or not parquet_path.exists():
                raise FileNotFoundError("use_real_expensive=1 requires valid --parquet (tar mode)")
            if not str(args.tar_dir).strip():
                raise ValueError("use_real_expensive=1 requires --tar_dir or --image_dir")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_overview_json.parent.mkdir(parents=True, exist_ok=True)
    if out_overview_csv is not None:
        out_overview_csv.parent.mkdir(parents=True, exist_ok=True)

    feat_map = load_feature_map(feat_path)
    c1_map: Dict[str, Dict[str, np.ndarray]] = {}
    tar_mapping: Dict[str, Dict[str, Any]] = {}
    image_loader: Optional[Any] = None
    expensive_models: Optional[ExpensiveModels] = None
    expensive_ready = False

    if cfg.use_real_expensive:
        c1_map = load_c1_map(c1_path)
        if not c1_map:
            raise RuntimeError(f"Failed to load c1 embeddings from {c1_path}")

        first_text = next(iter(c1_map.values()))["c1_txt_embed"]
        text_dim = int(first_text.shape[0])
        local_loader: Optional[LocalImageLoader] = None
        tar_loader: Optional[TarImageLoader] = None
        if str(args.image_dir).strip():
            local_loader = LocalImageLoader(image_dir=str(args.image_dir))
        if str(args.tar_dir).strip() and parquet_path is not None and parquet_path.exists():
            tar_mapping = load_tar_mapping(parquet_path)
            tar_loader = TarImageLoader(mapping=tar_mapping, tar_dir=str(args.tar_dir))

        if local_loader is not None and tar_loader is not None:
            image_loader = HybridImageLoader(local_loader=local_loader, tar_loader=tar_loader)
        elif local_loader is not None:
            image_loader = local_loader
        elif tar_loader is not None:
            image_loader = tar_loader
        else:
            raise ValueError("use_real_expensive=1 requires --image_dir and/or (--tar_dir + --parquet)")
        expensive_models = ExpensiveModels(
            text_dim=text_dim,
            align_model_name=str(args.align_model_name).strip(),
            align_pretrained=str(args.align_pretrained).strip(),
            align_device=str(args.align_device),
            aesthetic_device=str(args.aesthetic_device),
            batch_size=int(args.exp_batch_size),
            preprocess_workers=int(args.exp_preprocess_workers),
            pin_memory=bool(int(args.exp_pin_memory)),
            aesthetic_mlp_path=Path(args.aesthetic_mlp_path),
            aesthetic_mlp_url=str(args.aesthetic_mlp_url),
        )
        expensive_ready = True

    stats_by_ar: Dict[str, ARStats] = {}
    stats_all = ARStats()
    expensive_applied_images = 0
    missing_image_count = 0
    missing_c1_count = 0

    try:
        written = 0
        with cand_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
            iterable: Iterable[str] = fin
            if int(args.progress) != 0:
                iterable = tqdm(iterable, desc="teacher-score")

            line_idx = -1
            for line in iterable:
                line = line.strip()
                if not line:
                    continue
                line_idx += 1
                if args.max_images > 0 and line_idx >= args.max_images:
                    break
                if int(args.num_shards) > 1:
                    if (line_idx % int(args.num_shards)) != int(args.shard_index):
                        continue
                cand_rec = json.loads(line)
                image_id = str(cand_rec.get("image_id", ""))
                if not image_id:
                    continue

                feat_rec = feat_map.get(image_id, {"image_id": image_id, "c2_seg": [], "c3_pose": [], "c5_geom": {}})

                image_pil = None
                c1_text_embed = None
                if expensive_ready:
                    c1e = c1_map.get(image_id)
                    if c1e is None:
                        missing_c1_count += 1
                    else:
                        c1_text_embed = c1e.get("c1_txt_embed")
                    if image_loader is not None:
                        image_pil = image_loader.load(image_id)
                        if image_pil is None:
                            missing_image_count += 1

                one = process_one_image(
                    cand_rec=cand_rec,
                    feat_rec=feat_rec,
                    cfg=cfg,
                    expensive_models=expensive_models if (image_pil is not None and c1_text_embed is not None) else None,
                    image_pil=image_pil,
                    c1_text_embed=c1_text_embed,
                )
                if bool(one.get("teacher_scorer", {}).get("expensive_real_applied", False)):
                    expensive_applied_images += 1

                per_ar = one.pop("_stats_for_ar")
                for ar_text, st in per_ar.items():
                    if ar_text not in stats_by_ar:
                        stats_by_ar[ar_text] = ARStats()
                    s = stats_by_ar[ar_text]

                    # Means are accumulated then normalized at end.
                    s.candidates_in_mean += st["candidates_in"]
                    s.cheap_kept_mean += st["cheap_kept"]

                    selected_topk = one["teacher_scorer"]["results_by_ar"][ar_text].get("selected_topk", [])
                    route_info = one["teacher_scorer"]["results_by_ar"][ar_text].get("routing", {})
                    route_for_stats = {
                        "shot_type": route_info.get("shot_type", "unknown"),
                        "num_people": one.get("route_global", {}).get("num_people", 0),
                        "flags": route_info.get("flags", {}),
                    }

                    update_stats(
                        stats=s,
                        image_ar=safe_float(one.get("image_ar", 1.0)),
                        target_ar=parse_ar(ar_text),
                        decision_type=str(st["decision_type"]),
                        delta_improve=float(st["delta_improve"]),
                        topk=selected_topk,
                        ref_center_area=float(st["ref_center_area"]),
                        route=route_for_stats,
                        fallback=one["teacher_scorer"]["results_by_ar"][ar_text].get("fallback", {}),
                        cfg=cfg,
                    )

                    # Also aggregate global stats.
                    stats_all.candidates_in_mean += st["candidates_in"]
                    stats_all.cheap_kept_mean += st["cheap_kept"]
                    update_stats(
                        stats=stats_all,
                        image_ar=safe_float(one.get("image_ar", 1.0)),
                        target_ar=parse_ar(ar_text),
                        decision_type=str(st["decision_type"]),
                        delta_improve=float(st["delta_improve"]),
                        topk=selected_topk,
                        ref_center_area=float(st["ref_center_area"]),
                        route=route_for_stats,
                        fallback=one["teacher_scorer"]["results_by_ar"][ar_text].get("fallback", {}),
                        cfg=cfg,
                    )

                fout.write(json.dumps(one, ensure_ascii=False) + "\n")
                written += 1
    finally:
        if image_loader is not None:
            image_loader.close()
        if expensive_models is not None:
            expensive_models.close()

    # Finalize means.
    for s in list(stats_by_ar.values()) + [stats_all]:
        denom = max(1, s.images)
        s.candidates_in_mean /= denom
        s.cheap_kept_mean /= denom

    overview = {
        "schema_version": "teacher_scorer_v2_real_expensive_no_ocr",
        "config": asdict(cfg),
        "inputs": {
            "candidates_jsonl": str(cand_path),
            "features_jsonl": str(feat_path),
            "c1_jsonl": str(c1_path) if c1_path is not None else None,
            "parquet": str(parquet_path) if parquet_path is not None else None,
            "tar_dir": str(args.tar_dir) if str(args.tar_dir).strip() else None,
            "image_dir": str(args.image_dir) if str(args.image_dir).strip() else None,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "outputs": {
            "teacher_scores_jsonl": str(out_jsonl),
        },
        "expensive_stage": {
            "enabled": bool(cfg.use_real_expensive),
            "ready": bool(expensive_ready),
            "real_applied_images": int(expensive_applied_images),
            "missing_image_count": int(missing_image_count),
            "missing_c1_count": int(missing_c1_count),
            "align_model_name": (
                str(args.align_model_name) if str(args.align_model_name).strip() else None
            ),
            "align_pretrained": (
                str(args.align_pretrained) if str(args.align_pretrained).strip() else None
            ),
            "align_device_requested": str(args.align_device),
            "aesthetic_device_requested": str(args.aesthetic_device),
            "align_device_actual": (
                str(expensive_models.align_device)
                if expensive_models is not None
                else None
            ),
            "aesthetic_device_actual": (
                str(expensive_models.aes_device)
                if expensive_models is not None
                else None
            ),
            "aesthetic_mlp_path": str(args.aesthetic_mlp_path),
            "image_loader_type": (type(image_loader).__name__ if image_loader is not None else None),
            "local_hits": int(getattr(image_loader, "local_hits", 0)) if image_loader is not None else 0,
            "tar_hits": int(getattr(image_loader, "tar_hits", 0)) if image_loader is not None else 0,
            "loader_misses": int(getattr(image_loader, "misses", 0)) if image_loader is not None else 0,
            "exp_batch_size": int(args.exp_batch_size),
            "exp_preprocess_workers": int(cfg.exp_preprocess_workers),
            "exp_pin_memory": bool(cfg.exp_pin_memory),
            "expensive_eval_top_m": int(cfg.expensive_eval_top_m),
        },
        "summary": {
            "images_written": written,
            "global": summarize_stats(stats_all),
            "by_ar": {ar: summarize_stats(st) for ar, st in sorted(stats_by_ar.items())},
        },
    }

    with out_overview_json.open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    if out_overview_csv is not None:
        rows: List[Dict[str, Any]] = []
        for ar, st in sorted(stats_by_ar.items()):
            rec = summarize_stats(st)
            flat = {"target_ar": ar}
            for k, v in rec.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, dict):
                            continue
                        flat[f"{k}.{kk}"] = vv
                else:
                    flat[k] = v
            rows.append(flat)
        df = pd.DataFrame(rows)
        df.to_csv(out_overview_csv, index=False)

    print(f"[done] images={written} out_jsonl={out_jsonl}")
    print(f"[done] overview_json={out_overview_json}")
    if out_overview_csv is not None:
        print(f"[done] overview_by_ar_csv={out_overview_csv}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
