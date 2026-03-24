"""
Teacher Scorer (Section 9): cheap -> expensive -> Top-K diversity.

Implements a deterministic scorer pipeline for AR-conditioned + FREE-form
candidates using precomputed features (C2/C3/C5/C6) and candidate boxes.

Notes
-----
- Expensive stage uses lightweight deterministic proxies when full crop-level
  aesthetic/text-alignment models are unavailable in precompute inputs.
"""

from __future__ import annotations

import argparse
import base64
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
NIMA_DEFAULT_PATH = "weights/nima/NIMA_VGG16_ava-dc4e8265.pth"
NIMA_DEFAULT_URL = ""

DEFAULT_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
SCENE_MODES = {"scene_general", "scene_landscape"}
SCENE_SUBTYPE_UNKNOWN = "scene_general_unknown"
ALIGN_GPU_OOM_FALLBACKS: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {
    ("vit-h-14", "laion2b_s32b_b79k"): (("ViT-B-32", "openai"),),
}
SCENE_HORIZON_TARGETS: Dict[str, Tuple[float, ...]] = {
    "scene_landscape_nature": (1.0 / 3.0, 2.0 / 3.0),
    "scene_reflection_symmetry": (0.5,),
    "scene_city_architecture": (1.0 / 3.0, 0.5, 2.0 / 3.0),
    "scene_interior_architecture": (),
    "scene_structural_pattern": (),
    "scene_contextual_object": (),
    SCENE_SUBTYPE_UNKNOWN: (),
}


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
    hard_head_top_rule: bool = True
    hard_portrait_lookroom_rule: bool = True
    lookroom_hard_reject_min_subject_area_ratio: float = 0.04
    lookroom_hard_reject_min_route_conf: float = 0.88
    severe_kp_margin_alpha: float = 0.03
    hard_joint_reject_count: int = 2
    head_top_face_expand_alpha: float = 0.35
    head_top_kp_expand: float = 0.06
    head_top_min_margin: float = 0.008
    head_top_face_margin_alpha: float = 0.20

    # Cheap scoring weights (base)
    lambda_cov: float = 1.20
    lambda_cut: float = 1.80
    lambda_text: float = 0.20
    lambda_comp: float = 1.00
    lambda_hr: float = 0.65
    lambda_lr: float = 0.55
    lambda_sym: float = 0.25
    lambda_ctx: float = 0.45
    lambda_cs: float = 0.35
    lambda_teach: float = 0.10
    lambda_ar_free: float = 0.12

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
    horizon_visibility_thr: float = 0.70

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
    w_text: float = 0.12
    w_edge: float = 0.5
    w_teach: float = 0.10
    w_ar_free: float = 0.20
    rank_weight_a: float = 1.0
    rank_weight_s: float = 1.0
    rank_weight_c: float = 1.0
    rank_weight_t: float = 0.10
    a_macro_aesthetic_weight: float = 0.75
    a_macro_align_weight: float = 0.25
    aesthetic_score_min: float = 1.0
    aesthetic_score_max: float = 10.0
    expensive_eval_top_m: int = 0  # 0 means evaluate all cheap_top_m in expensive stage
    save_public_teacher_ref_eval: bool = True
    exp_preprocess_workers: int = 0  # 0 means auto (bounded by CPU count)
    exp_pin_memory: bool = True

    # OCR text-preservation checks (C4)
    text_keep_min_default: float = 0.90
    text_keep_min_overlay: float = 0.95
    text_keep_min_text_document: float = 0.98
    text_box_full_keep_thr: float = 0.98
    text_box_severe_keep_thr: float = 0.60
    text_severe_ratio_max_default: float = 0.20
    text_severe_ratio_max_text_document: float = 0.05
    text_penalty_keep_weight: float = 1.00
    text_penalty_cut_weight: float = 0.35
    text_penalty_severe_weight: float = 0.65

    # FREE-form AR prior (weak regularizer, not hard constraint)
    free_ar_log_tau: float = 0.90
    free_ar_portrait_max: float = 1.20
    free_ar_scene_min: float = 0.80
    free_ar_bias_scale: float = 0.20
    free_topk_ar_log_gap: float = 0.18

    # Teacher-consensus prior (v1.8)
    enable_r_teach: bool = True
    teach_rho_tau: float = 0.75
    teach_rho_beta: float = 0.05
    teach_consensus_pair_iou_thr: float = 0.85
    teach_require_consensus: bool = True

    # Keep-vs-crop
    w_area_default: float = 0.10
    tau_improve_default: float = 0.035
    teacher_tau_boost_delta: float = 0.02
    teacher_tau_boost_baseline_iou: float = 0.90

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
    top1_head_top_cut_count: int = 0
    top1_subject_cov_fail_count: int = 0

    portrait_subset_count: int = 0
    headroom_violation_count: int = 0
    head_top_subset_count: int = 0
    head_top_violation_count: int = 0
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
    roll_abs: List[float] = None

    def __post_init__(self) -> None:
        if self.delta_improve is None:
            self.delta_improve = []
        if self.top1_top2_iou is None:
            self.top1_top2_iou = []
        if self.horizon_third_dist is None:
            self.horizon_third_dist = []
        if self.roll_abs is None:
            self.roll_abs = []
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
    p.add_argument("--enable_r_teach", type=int, default=1, help="1=enable teacher-consensus prior R_teach")
    p.add_argument("--teach_rho_tau", type=float, default=0.75)
    p.add_argument("--teach_rho_beta", type=float, default=0.05)
    p.add_argument("--teach_consensus_pair_iou_thr", type=float, default=0.85)
    p.add_argument("--teach_require_consensus", type=int, default=1, help="1=gate R_teach by consensus")
    p.add_argument("--lambda_teach", type=float, default=0.10, help="cheap-stage R_teach weight")
    p.add_argument("--w_teach", type=float, default=0.10, help="expensive-stage R_teach weight")
    p.add_argument("--rank_weight_a", type=float, default=1.0, help="macro rank fusion weight for A_macro")
    p.add_argument("--rank_weight_s", type=float, default=1.0, help="macro rank fusion weight for S_macro")
    p.add_argument("--rank_weight_c", type=float, default=1.0, help="macro rank fusion weight for C_macro")
    p.add_argument("--rank_weight_t", type=float, default=0.10, help="macro rank fusion weight for T_macro")
    p.add_argument(
        "--a_macro_aesthetic_weight",
        type=float,
        default=0.75,
        help="internal A_macro weight for crop aesthetic score",
    )
    p.add_argument(
        "--a_macro_align_weight",
        type=float,
        default=0.25,
        help="internal A_macro weight for image-text alignment quality",
    )
    p.add_argument("--teacher_tau_boost_delta", type=float, default=0.02, help="tau_improve boost when baseline aligns with teacher consensus")
    p.add_argument("--teacher_tau_boost_baseline_iou", type=float, default=0.90, help="baseline IoU threshold for tau boost")
    p.add_argument("--hard_head_top_rule", type=int, default=1, help="1=portrait head-top(hair) cut hard reject")
    p.add_argument("--hard_portrait_lookroom_rule", type=int, default=1, help="1=portrait insufficient lookroom hard reject")
    p.add_argument(
        "--lookroom_hard_reject_min_subject_area_ratio",
        type=float,
        default=0.04,
        help="disable portrait lookroom hard reject when routed subject is smaller than this image-area ratio",
    )
    p.add_argument(
        "--lookroom_hard_reject_min_route_conf",
        type=float,
        default=0.88,
        help="disable portrait lookroom hard reject when subject-mode confidence is below this threshold",
    )
    p.add_argument("--head_top_face_expand_alpha", type=float, default=0.35, help="head-top estimate: face_y1 - alpha*face_h")
    p.add_argument("--head_top_kp_expand", type=float, default=0.06, help="head-top estimate from keypoints: top_kp_y - value")
    p.add_argument("--head_top_min_margin", type=float, default=0.008, help="minimum safety margin for head-top inclusion")
    p.add_argument("--head_top_face_margin_alpha", type=float, default=0.20, help="face-height based head-top safety margin")
    p.add_argument("--align_model_name", type=str, default="", help="OpenCLIP model for E_I(I_b)")
    p.add_argument("--align_pretrained", type=str, default="", help="OpenCLIP pretrained tag for E_I(I_b)")
    p.add_argument("--align_device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--aesthetic_device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument(
        "--aesthetic_backend",
        type=str,
        default="hybrid",
        help="laion|nima|hybrid (hybrid: NIMA primary + LAION prior)",
    )
    p.add_argument(
        "--aesthetic_prior_laion_weight",
        type=float,
        default=0.15,
        help="LAION prior mixing weight for hybrid backend",
    )
    p.add_argument("--exp_batch_size", type=int, default=24)
    p.add_argument("--expensive_eval_top_m", type=int, default=0, help="0=all cheap_top_m, >0=capped expensive eval")
    p.add_argument(
        "--save_public_teacher_ref_eval",
        type=int,
        default=1,
        help="1=save exact scored public teacher seed/proj refs per AR regardless of cheap_top_m/top_k",
    )
    p.add_argument("--exp_preprocess_workers", type=int, default=0, help="0=auto, >0=thread workers for clip preprocess")
    p.add_argument("--exp_pin_memory", type=int, default=1, help="1=pin CPU batch tensor before H2D copy")
    p.add_argument(
        "--aesthetic_mlp_path",
        type=str,
        default="weights/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
    )
    p.add_argument("--aesthetic_mlp_url", type=str, default=AESTHETIC_DEFAULT_URL)
    p.add_argument("--nima_model_path", type=str, default=NIMA_DEFAULT_PATH)
    p.add_argument("--nima_model_url", type=str, default=NIMA_DEFAULT_URL)
    p.add_argument(
        "--nima_use_imagenet_backbone",
        type=int,
        default=1,
        help="1=initialize VGG16 backbone from ImageNet weights when NIMA ckpt is absent",
    )
    p.add_argument(
        "--nima_require_ckpt",
        type=int,
        default=1,
        help="1=require NIMA checkpoint to activate NIMA backend",
    )
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


def safe_optional_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def parse_ar(ar_text: str) -> float:
    t = str(ar_text).strip()
    if t.upper() in {"FREE", "AR_FREE", "FREEFORM", "FREE_FORM"}:
        # Backward-compatible numeric fallback for generic analytics codepaths.
        return 1.0
    if ":" in t:
        a, b = t.split(":", 1)
        return float(a) / float(b)
    return float(t)


def parse_target_ar(ar_text: str) -> Optional[float]:
    t = str(ar_text).strip()
    if t.upper() in {"FREE", "AR_FREE", "FREEFORM", "FREE_FORM"}:
        return None
    return parse_ar(t)


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


def normalize_ar_key(ar_text: Any) -> str:
    t = str(ar_text).strip()
    if parse_target_ar(t) is None:
        return "FREE"
    return t


def sigmoid(x: float) -> float:
    z = float(x)
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def build_teacher_consensus_context(
    *,
    cand_rec: Dict[str, Any],
    ar_text: str,
    cfg: TeacherScorerConfig,
) -> Dict[str, Any]:
    raw = cand_rec.get("teacher_candidates", [])
    if not isinstance(raw, list) or not raw:
        return {
            "enabled": False,
            "available": False,
            "num_teachers": 0,
            "num_boxes": 0,
            "consensus": False,
            "max_pair_iou": 0.0,
            "boxes": [],
            "teachers": [],
        }

    target_key = normalize_ar_key(ar_text)
    per_teacher: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip().lower()
        if stage not in {"seed", "proj"}:
            continue
        if normalize_ar_key(item.get("target_ar", "")) != target_key:
            continue
        b = item.get("bbox_norm_xyxy")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        bb = clip_box01([safe_float(v) for v in b])
        if box_area(bb) <= 0.0:
            continue
        tid = str(item.get("teacher_id", "unknown")).strip() or "unknown"
        score = safe_float(item.get("teacher_score", -1e9), -1e9)
        prev = per_teacher.get(tid)
        if prev is None or score > safe_float(prev.get("teacher_score", -1e9), -1e9):
            per_teacher[tid] = {
                "teacher_id": tid,
                "teacher_score": float(score),
                "bbox_norm_xyxy": bb,
            }

    teachers = list(per_teacher.values())
    boxes = [t["bbox_norm_xyxy"] for t in teachers]
    num_teachers = len(teachers)
    enabled = bool(num_teachers > 0)
    max_pair_iou = 0.0
    for i in range(num_teachers):
        bi = boxes[i]
        for j in range(i + 1, num_teachers):
            bj = boxes[j]
            max_pair_iou = max(max_pair_iou, iou_xyxy(bi, bj))

    if bool(cfg.teach_require_consensus):
        consensus = (num_teachers >= 2) and (max_pair_iou >= float(cfg.teach_consensus_pair_iou_thr))
    else:
        consensus = enabled

    return {
        "enabled": bool(enabled),
        "available": bool(num_teachers >= 2),
        "num_teachers": int(num_teachers),
        "num_boxes": int(len(boxes)),
        "consensus": bool(consensus),
        "max_pair_iou": float(max_pair_iou),
        "boxes": boxes,
        "teachers": teachers,
    }


def select_public_teacher_refs(
    *,
    cand_rec: Dict[str, Any],
    ar_text: str,
) -> List[Dict[str, Any]]:
    raw = cand_rec.get("teacher_candidates", [])
    if not isinstance(raw, list) or not raw:
        return []

    target_key = normalize_ar_key(ar_text)
    by_teacher_stage: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip().lower()
        if stage not in {"seed", "proj"}:
            continue
        if normalize_ar_key(item.get("target_ar", "")) != target_key:
            continue
        bbox = item.get("bbox_norm_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        bbox01 = clip_box01([safe_float(v) for v in bbox])
        if box_area(bbox01) <= 0.0:
            continue
        teacher_id = str(item.get("teacher_id", "unknown")).strip() or "unknown"
        score = safe_float(item.get("teacher_score", -1e9), -1e9)
        stage_bucket = by_teacher_stage.setdefault(teacher_id, {})
        current = stage_bucket.get(stage)
        if current is None or score > safe_float(current.get("teacher_score", -1e9), -1e9):
            stage_bucket[stage] = {
                "teacher_id": teacher_id,
                "target_ar": str(item.get("target_ar", ar_text)),
                "stage": stage,
                "bbox_norm_xyxy": bbox01,
                "teacher_score": float(score),
            }
    out: List[Dict[str, Any]] = []
    prefer_seed = normalize_ar_key(ar_text) == "FREE"
    for teacher_id, stage_map in sorted(by_teacher_stage.items()):
        chosen = None
        if prefer_seed:
            chosen = stage_map.get("seed") or stage_map.get("proj")
        else:
            chosen = stage_map.get("proj") or stage_map.get("seed")
        if chosen is not None:
            out.append(chosen)
    return out


def compute_teacher_consensus_reward(
    *,
    crop: Sequence[float],
    teacher_ctx: Optional[Dict[str, Any]],
    cfg: TeacherScorerConfig,
) -> Dict[str, Any]:
    if (not bool(cfg.enable_r_teach)) or (not isinstance(teacher_ctx, dict)):
        return {
            "enabled": False,
            "available": False,
            "consensus": False,
            "rho": 0.0,
            "r_teach": 0.0,
        }

    boxes = teacher_ctx.get("boxes", []) if isinstance(teacher_ctx.get("boxes"), list) else []
    if not boxes:
        return {
            "enabled": bool(teacher_ctx.get("enabled", False)),
            "available": bool(teacher_ctx.get("available", False)),
            "consensus": bool(teacher_ctx.get("consensus", False)),
            "rho": 0.0,
            "r_teach": 0.0,
        }

    rho = max(iou_xyxy(crop, b) for b in boxes)
    consensus_gate = bool(teacher_ctx.get("consensus", False))
    beta = max(1e-6, float(cfg.teach_rho_beta))
    r = sigmoid((float(rho) - float(cfg.teach_rho_tau)) / beta)
    if bool(cfg.teach_require_consensus) and (not consensus_gate):
        r = 0.0
    return {
        "enabled": bool(teacher_ctx.get("enabled", False)),
        "available": bool(teacher_ctx.get("available", False)),
        "consensus": bool(consensus_gate),
        "rho": float(rho),
        "r_teach": float(r),
    }


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


class NIMAVGG16Model:
    """
    NIMA reference head (AVA style):
    - VGG16 feature backbone
    - Dropout(0.75) + Linear(25088->10) + Softmax
    """

    def __init__(self, torch_mod: Any, base_features: Any):
        class _NIMAImpl(torch_mod.nn.Module):
            def __init__(self, features: Any):
                super().__init__()
                nn = torch_mod.nn
                self.features = features
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.75),
                    nn.Linear(in_features=25088, out_features=10),
                    nn.Softmax(dim=1),
                )

            def forward(self, x: Any) -> Any:
                out = self.features(x)
                out = out.view(out.size(0), -1)
                return self.classifier(out)

        self.model = _NIMAImpl(base_features)


class NIMAVGG16GAPModel:
    """
    NIMA VGG16 variant used by `NIMA_VGG16_ava-dc4e8265.pth`:
    - VGG16 features
    - AdaptiveAvgPool2d(1) -> Flatten -> Linear(512->10) -> Softmax
    """

    def __init__(self, torch_mod: Any, base_features: Any):
        class _NIMAImpl(torch_mod.nn.Module):
            def __init__(self, features: Any):
                super().__init__()
                nn = torch_mod.nn
                self.features = features
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(in_features=512, out_features=10),
                    nn.Softmax(dim=1),
                )

            def forward(self, x: Any) -> Any:
                out = self.features(x)
                return self.classifier(out)

        self.model = _NIMAImpl(base_features)


class ExpensiveModels:
    """
    Real expensive scorer bundle:
    - A(I_b):
      * NIMA (primary A_gen)
      * LAION aesthetic predictor (demoted to optional prior)
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
        aesthetic_backend: str,
        aesthetic_prior_laion_weight: float,
        aesthetic_mlp_path: Path,
        aesthetic_mlp_url: str,
        nima_model_path: Path,
        nima_model_url: str,
        nima_use_imagenet_backbone: bool,
        nima_require_ckpt: bool,
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
        self.nima_device = self.aes_device
        self.aesthetic_backend_requested = self._normalize_aesthetic_backend(aesthetic_backend)
        self.aesthetic_backend_effective = "none"
        self.laion_prior_weight = clamp(float(aesthetic_prior_laion_weight), 0.0, 1.0)
        self.laion_prior_weight_effective = 0.0
        self.nima_ckpt_loaded = False

        self.aes_clip = None
        self.aes_preprocess = None
        self.aes_mlp = None
        self.nima_model = None
        self.nima_preprocess = None
        self._laion_ready = False
        self._nima_ready = False
        self.align_model_name = ""
        self.align_pretrained = ""

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
            purpose="align",
        )
        self.align_model.eval()
        if self.align_device != "cpu":
            self.align_model = self.align_model.half()

        use_laion_requested = self.aesthetic_backend_requested in {"laion", "hybrid"}
        use_nima_requested = self.aesthetic_backend_requested in {"nima", "hybrid"}

        if use_laion_requested:
            try:
                maybe_download_file(aesthetic_mlp_url, aesthetic_mlp_path)
                self.aes_clip, self.aes_preprocess, self.aes_device = self._create_clip_with_fallback(
                    model_name="ViT-L-14",
                    pretrained="openai",
                    preferred_device=self.aes_device,
                    purpose="aesthetic",
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
                self._laion_ready = True
            except Exception as e:
                print(f"[expensive][warn] failed to initialize LAION aesthetic predictor: {e}")
                self._laion_ready = False
                if self.aesthetic_backend_requested == "laion":
                    raise

        if use_nima_requested:
            try:
                (
                    self.nima_model,
                    self.nima_preprocess,
                    self.nima_device,
                    self.nima_ckpt_loaded,
                ) = self._create_nima_with_fallback(
                    preferred_device=self.nima_device,
                    nima_model_path=nima_model_path,
                    nima_model_url=nima_model_url,
                    use_imagenet_backbone=bool(nima_use_imagenet_backbone),
                )
                self._nima_ready = self.nima_model is not None
                if bool(nima_require_ckpt) and (not bool(self.nima_ckpt_loaded)):
                    print(
                        "[expensive][warn] NIMA checkpoint required but unavailable; "
                        "NIMA backend deactivated."
                    )
                    self._nima_ready = False
                    self.nima_model = None
                    self.nima_preprocess = None
            except Exception as e:
                print(f"[expensive][warn] failed to initialize NIMA aesthetic predictor: {e}")
                self._nima_ready = False
                if self.aesthetic_backend_requested == "nima":
                    raise

        if self.aesthetic_backend_requested == "hybrid":
            if self._nima_ready and self._laion_ready:
                self.aesthetic_backend_effective = "hybrid"
                self.laion_prior_weight_effective = float(self.laion_prior_weight)
            elif self._nima_ready:
                self.aesthetic_backend_effective = "nima"
                self.laion_prior_weight_effective = 0.0
            elif self._laion_ready:
                self.aesthetic_backend_effective = "laion"
                self.laion_prior_weight_effective = 1.0
            else:
                raise RuntimeError("Neither NIMA nor LAION aesthetic backend initialized successfully")
        elif self.aesthetic_backend_requested == "nima":
            if not self._nima_ready:
                raise RuntimeError("NIMA aesthetic backend requested but initialization failed")
            self.aesthetic_backend_effective = "nima"
            self.laion_prior_weight_effective = 0.0
        else:
            if not self._laion_ready:
                raise RuntimeError("LAION aesthetic backend requested but initialization failed")
            self.aesthetic_backend_effective = "laion"
            self.laion_prior_weight_effective = 1.0

        print(
            "[expensive] aesthetic backend: "
            f"requested={self.aesthetic_backend_requested} "
            f"effective={self.aesthetic_backend_effective} "
            f"nima_ready={self._nima_ready}(ckpt_loaded={self.nima_ckpt_loaded}) "
            f"laion_ready={self._laion_ready} "
            f"laion_prior_w={self.laion_prior_weight_effective:.3f}"
        )

    @staticmethod
    def _normalize_aesthetic_backend(v: str) -> str:
        s = str(v).strip().lower()
        if s not in {"laion", "nima", "hybrid"}:
            raise ValueError(f"Unsupported --aesthetic_backend={v}. Use one of: laion|nima|hybrid")
        return s

    def _build_nima_model(self, use_imagenet_backbone: bool, variant: str) -> Any:
        from torchvision import models as tv_models

        base = None
        if bool(use_imagenet_backbone):
            try:
                weights = tv_models.VGG16_Weights.IMAGENET1K_V1
                base = tv_models.vgg16(weights=weights)
            except Exception:
                try:
                    base = tv_models.vgg16(pretrained=True)
                except Exception as e:
                    print(f"[expensive][warn] failed to load ImageNet VGG16 backbone for NIMA: {e}")
                    base = None
        if base is None:
            try:
                base = tv_models.vgg16(weights=None)
            except TypeError:
                base = tv_models.vgg16(pretrained=False)

        if str(variant).lower() == "gap":
            return NIMAVGG16GAPModel(self.torch, base.features).model
        return NIMAVGG16Model(self.torch, base.features).model

    def _detect_nima_variant(self, state_dict: Dict[str, Any]) -> str:
        if not isinstance(state_dict, dict):
            return "flatten"
        if any(str(k).startswith("base_model.features_") for k in state_dict.keys()):
            return "gap"
        gap_key = state_dict.get("classifier.2.weight")
        if hasattr(gap_key, "shape"):
            try:
                if len(gap_key.shape) == 2 and int(gap_key.shape[1]) == 512:
                    return "gap"
            except Exception:
                pass
        return "flatten"

    def _load_torch_state_dict(self, path: Path) -> Tuple[Dict[str, Any], str]:
        try:
            state = self.torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            state = self.torch.load(str(path), map_location="cpu")

        if isinstance(state, dict):
            for k in ("state_dict", "model_state_dict", "model", "net", "params"):
                v = state.get(k)
                if isinstance(v, dict):
                    state = v
                    break
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint is not a state-dict: {path}")

        variant = self._detect_nima_variant(state)

        cleaned: Dict[str, Any] = {}
        for k, v in state.items():
            kk = str(k)
            if kk.startswith("module."):
                kk = kk[7:]
            if kk.startswith("model."):
                kk = kk[6:]
            if kk.startswith("base_model."):
                kk = kk[11:]
            if kk.startswith("features_"):
                rem = kk[9:]
                if "." in rem:
                    idx, tail = rem.split(".", 1)
                    if idx.isdigit():
                        kk = f"features.{idx}.{tail}"
            if variant == "flatten" and kk.startswith("classifier.2."):
                kk = "classifier.1." + kk[len("classifier.2.") :]
            cleaned[kk] = v
        return cleaned, variant

    def _create_nima_with_fallback(
        self,
        *,
        preferred_device: str,
        nima_model_path: Path,
        nima_model_url: str,
        use_imagenet_backbone: bool,
    ) -> Tuple[Any, Any, str, bool]:
        from torchvision import transforms as tv_transforms

        nima_variant = "flatten"
        state: Optional[Dict[str, Any]] = None
        ckpt_loaded = False

        if (not nima_model_path.exists()) and str(nima_model_url).strip():
            maybe_download_file(str(nima_model_url).strip(), nima_model_path)
        if nima_model_path.exists():
            state, nima_variant = self._load_torch_state_dict(nima_model_path)
        model = self._build_nima_model(
            use_imagenet_backbone=bool(use_imagenet_backbone),
            variant=nima_variant,
        )
        if state is not None:
            missing, unexpected = model.load_state_dict(state, strict=False)
            ckpt_loaded = True
            print(f"[expensive] NIMA checkpoint variant={nima_variant}")
            if missing:
                print(f"[expensive][warn] NIMA missing keys: {len(missing)}")
            if unexpected:
                print(f"[expensive][warn] NIMA unexpected keys: {len(unexpected)}")
        else:
            print(
                "[expensive][warn] NIMA checkpoint not found; "
                "using ImageNet-backed VGG16 + randomly initialized NIMA head."
            )

        try:
            interp = tv_transforms.InterpolationMode.BILINEAR
            resize = tv_transforms.Resize(256, interpolation=interp)
        except Exception:
            resize = tv_transforms.Resize(256)

        preprocess = tv_transforms.Compose(
            [
                resize,
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        actual_device = preferred_device
        try:
            model.to(actual_device)
        except Exception as e:
            msg = str(e).lower()
            if actual_device != "cpu" and ("out of memory" in msg or "cuda" in msg):
                print(f"[expensive][warn] failed to place NIMA on {actual_device}: {e} -> retry on CPU")
                model.to("cpu")
                actual_device = "cpu"
            else:
                raise
        model.eval()
        return model, preprocess, actual_device, ckpt_loaded

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
        use_half: bool = True,
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
            batch = batch.half() if bool(use_half) else batch.float()
        else:
            batch = batch.to(device)
            if not bool(use_half):
                batch = batch.float()
        return batch

    def _create_clip_with_fallback(
        self,
        *,
        model_name: str,
        pretrained: str,
        preferred_device: str,
        purpose: str = "clip",
    ) -> Tuple[Any, Any, str]:
        try:
            model, _, preprocess = self.open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=preferred_device,
            )
            if purpose == "align":
                self.align_model_name = str(model_name)
                self.align_pretrained = str(pretrained)
            return model, preprocess, preferred_device
        except Exception as e:
            msg = str(e).lower()
            if preferred_device != "cpu" and ("out of memory" in msg or "cuda" in msg):
                fallback_key = (str(model_name).strip().lower(), str(pretrained).strip().lower())
                for fb_model_name, fb_pretrained in ALIGN_GPU_OOM_FALLBACKS.get(fallback_key, ()):
                    try:
                        print(
                            f"[expensive][warn] failed to load {model_name}/{pretrained} on {preferred_device}: {e}\n"
                            f" -> retry on {preferred_device} with downgraded {fb_model_name}/{fb_pretrained}"
                        )
                        model, _, preprocess = self.open_clip.create_model_and_transforms(
                            fb_model_name,
                            pretrained=fb_pretrained,
                            device=preferred_device,
                        )
                        if purpose == "align":
                            self.align_model_name = str(fb_model_name)
                            self.align_pretrained = str(fb_pretrained)
                        return model, preprocess, preferred_device
                    except Exception as fb_e:
                        print(
                            f"[expensive][warn] downgraded {fb_model_name}/{fb_pretrained} on {preferred_device} also failed: {fb_e}"
                        )
                print(
                    f"[expensive][warn] failed to load {model_name}/{pretrained} on {preferred_device}: {e}\n"
                    " -> retry on CPU"
                )
                model, _, preprocess = self.open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    device="cpu",
                )
                if purpose == "align":
                    self.align_model_name = str(model_name)
                    self.align_pretrained = str(pretrained)
                return model, preprocess, "cpu"
            raise

    def _downgrade_align_model_after_oom(self) -> bool:
        if self.align_device == "cpu":
            return False
        key = (str(self.align_model_name).strip().lower(), str(self.align_pretrained).strip().lower())
        fallbacks = ALIGN_GPU_OOM_FALLBACKS.get(key, ())
        for fb_model_name, fb_pretrained in fallbacks:
            try:
                print(
                    f"[expensive][warn] align batch OOM on GPU -> switching align model to {fb_model_name}/{fb_pretrained} on {self.align_device}"
                )
                model, preprocess, device = self._create_clip_with_fallback(
                    model_name=fb_model_name,
                    pretrained=fb_pretrained,
                    preferred_device=self.align_device,
                    purpose="align",
                )
                model.eval()
                if device != "cpu":
                    model = model.half()
                self.align_model = model
                self.align_preprocess = preprocess
                self.align_device = device
                self.align_model_name = str(fb_model_name)
                self.align_pretrained = str(fb_pretrained)
                return True
            except Exception as e:
                print(f"[expensive][warn] failed to downgrade align model after OOM: {e}")
        return False

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
                        use_half=True,
                    )
                    emb = self.align_model.encode_image(batch)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    all_out.append(emb.detach().cpu().float().numpy())
        except Exception as e:
            if self.align_device != "cpu" and "out of memory" in str(e).lower():
                if self._downgrade_align_model_after_oom():
                    try:
                        self.torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return self._batched_encode_align(crops)
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

    def _batched_predict_laion_aesthetic(self, crops: Sequence["Image.Image"]) -> np.ndarray:
        if (not crops) or (not self._laion_ready) or self.aes_clip is None or self.aes_mlp is None:
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
                        use_half=True,
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
                return self._batched_predict_laion_aesthetic(crops)
            raise
        return np.concatenate(all_out, axis=0).astype(np.float32, copy=False)

    def _batched_predict_nima(self, crops: Sequence["Image.Image"]) -> Tuple[np.ndarray, np.ndarray]:
        if (not crops) or (not self._nima_ready) or self.nima_model is None or self.nima_preprocess is None:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        try:
            with self.torch.inference_mode():
                for i in range(0, len(crops), self.batch_size):
                    batch_imgs = crops[i : i + self.batch_size]
                    batch = self._preprocess_batch(
                        batch_imgs=batch_imgs,
                        preprocess_fn=self.nima_preprocess,
                        device=self.nima_device,
                        use_half=False,
                    )
                    dist = self.nima_model(batch).float()
                    dist = self.torch.clamp(dist, min=1e-8)
                    dist = dist / dist.sum(dim=1, keepdim=True)

                    bins = self.torch.arange(
                        1, 11, dtype=dist.dtype, device=dist.device
                    ).view(1, -1)
                    mean_val = (dist * bins).sum(dim=1)
                    var_val = (dist * (bins - mean_val.unsqueeze(1)) ** 2).sum(dim=1)
                    std_val = self.torch.sqrt(self.torch.clamp(var_val, min=0.0))
                    means.append(mean_val.detach().cpu().numpy().astype(np.float32, copy=False))
                    stds.append(std_val.detach().cpu().numpy().astype(np.float32, copy=False))
        except Exception as e:
            if self.nima_device != "cpu" and "out of memory" in str(e).lower():
                print("[expensive][warn] NIMA batch OOM on GPU -> switching NIMA to CPU")
                try:
                    self.torch.cuda.empty_cache()
                except Exception:
                    pass
                self.nima_model.to("cpu")
                self.nima_device = "cpu"
                return self._batched_predict_nima(crops)
            raise
        if not means:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return (
            np.concatenate(means, axis=0).astype(np.float32, copy=False),
            np.concatenate(stds, axis=0).astype(np.float32, copy=False),
        )

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
    ) -> Dict[str, Dict[str, Any]]:
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
        a_den = max(1e-8, cfg.aesthetic_score_max - cfg.aesthetic_score_min)

        if self._laion_ready:
            laion_raw = self._batched_predict_laion_aesthetic(crops)
            laion_norm = np.clip((laion_raw - cfg.aesthetic_score_min) / a_den, 0.0, 1.0)
        else:
            laion_raw = np.zeros((len(crops),), dtype=np.float32)
            laion_norm = np.zeros((len(crops),), dtype=np.float32)

        if self._nima_ready:
            nima_mean, nima_std = self._batched_predict_nima(crops)
            nima_norm = np.clip((nima_mean - cfg.aesthetic_score_min) / a_den, 0.0, 1.0)
        else:
            nima_mean = np.zeros((len(crops),), dtype=np.float32)
            nima_std = np.zeros((len(crops),), dtype=np.float32)
            nima_norm = np.zeros((len(crops),), dtype=np.float32)

        if self.aesthetic_backend_effective == "nima":
            aes_raw = nima_mean
            aes_norm = nima_norm
        elif self.aesthetic_backend_effective == "laion":
            aes_raw = laion_raw
            aes_norm = laion_norm
        else:
            w_laion = float(self.laion_prior_weight_effective)
            aes_raw = (1.0 - w_laion) * nima_mean + w_laion * laion_raw
            aes_norm = (1.0 - w_laion) * nima_norm + w_laion * laion_norm

        boxkey_to_signal: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}
        for i, k in enumerate(keys):
            if text_vec is None or align_emb.shape[1] != text_vec.shape[0]:
                cos_val = 0.0
            else:
                cos_val = float(np.dot(normalize_vec(align_emb[i]), text_vec))
            boxkey_to_signal[k] = {
                "aesthetic_raw": float(aes_raw[i]),
                "aesthetic_norm": float(aes_norm[i]),
                "aesthetic_backend": str(self.aesthetic_backend_effective),
                "aesthetic_prior_laion_weight": float(self.laion_prior_weight_effective),
                "aesthetic_raw_laion": (float(laion_raw[i]) if self._laion_ready else None),
                "aesthetic_norm_laion": (float(laion_norm[i]) if self._laion_ready else None),
                "aesthetic_mean_nima": (float(nima_mean[i]) if self._nima_ready else None),
                "aesthetic_std_nima": (float(nima_std[i]) if self._nima_ready else None),
                "aesthetic_norm_nima": (float(nima_norm[i]) if self._nima_ready else None),
                "cosine_img_text": float(cos_val),
            }

        out: Dict[str, Dict[str, Any]] = {}
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


def subject_scale_target_range(
    *,
    subject_mode: str,
    shot_type: str,
    scene_subtype: str,
    flags: Dict[str, bool],
    has_human_evidence: bool,
) -> Tuple[Tuple[float, float], str]:
    mode = str(subject_mode or "").strip().lower()
    subtype = normalize_scene_subtype(scene_subtype)
    if mode == "portrait_single":
        table = {
            "headshot": (0.55, 0.82),
            "half": (0.38, 0.62),
            "full": (0.22, 0.42),
            "unknown": (0.28, 0.60),
        }
        return table.get(shot_type, table["unknown"]), "primary_subject"
    if mode == "portrait_group":
        return (0.22, 0.52), "subject_union"
    if mode == "object_single":
        return (0.35, 0.72), "primary_subject"
    if mode == "object_multi":
        return (0.25, 0.58), "subject_union"
    if mode == "background_texture_copyspace":
        return (0.08, 0.30), "primary_subject"
    if mode == "text_document":
        return (0.35, 0.90), "primary_subject"
    if mode in SCENE_MODES or mode.startswith("scene"):
        if subtype == "scene_contextual_object":
            return (0.12, 0.32), "subject_union"
        return (0.05, 0.22), "primary_subject"
    if flags.get("is_product", False):
        return (0.40, 0.80), "primary_subject"
    if has_human_evidence:
        return (0.22, 0.55), "primary_subject"
    return (0.12, 0.45), "primary_subject"


def context_keep_target_range(
    *,
    subject_mode: str,
    scene_subtype: str,
    flags: Dict[str, bool],
) -> Tuple[Tuple[float, float], str]:
    mode = str(subject_mode or "").strip().lower()
    subtype = normalize_scene_subtype(scene_subtype)
    if mode == "background_texture_copyspace":
        return (0.55, 1.00), "negative_space"
    if mode == "text_document":
        return (0.70, 1.00), "secondary_context"
    if mode in {"object_single", "portrait_single"}:
        return (0.18, 0.55), "secondary_context"
    if mode in {"object_multi", "portrait_group"}:
        return (0.28, 0.65), "secondary_context"
    if mode in SCENE_MODES or mode.startswith("scene"):
        if subtype == "scene_contextual_object":
            return (0.30, 0.78), "secondary_context"
        if subtype in {"scene_reflection_symmetry", "scene_city_architecture", "scene_interior_architecture"}:
            return (0.45, 0.95), "scene_structure"
        return (0.40, 0.95), "scene_structure"
    if flags.get("is_product", False):
        return (0.10, 0.35), "secondary_context"
    return (0.20, 0.60), "secondary_context"


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
    routing = candidate_rec.get("routing", {})
    if not isinstance(routing, dict):
        routing = feat_rec.get("routing", {}) if isinstance(feat_rec.get("routing"), dict) else {}
    elif isinstance(feat_rec.get("routing"), dict):
        routing = {**routing, **feat_rec.get("routing", {})}
    subject_set = routing.get("subject_set", {}) if isinstance(routing.get("subject_set"), dict) else {}
    mode = str(routing.get("subject_mode", "")).strip().lower()

    effective_subject_region = (
        routing.get("effective_subject_region", {})
        if isinstance(routing.get("effective_subject_region"), dict)
        else {}
    )
    effective_box = effective_subject_region.get("effective_bbox_norm_xyxy")
    if isinstance(effective_box, (list, tuple)) and len(effective_box) == 4:
        eb = clip_box01(effective_box)
        if box_area(eb) > 0:
            return eb

    sp = candidate_rec.get("subject_prior", {})
    if isinstance(sp, dict):
        nested_effective = (
            sp.get("effective_subject_region", {})
            if isinstance(sp.get("effective_subject_region"), dict)
            else {}
        )
        for key in ("effective_bbox_norm_xyxy",):
            box = nested_effective.get(key)
            if isinstance(box, (list, tuple)) and len(box) == 4:
                eb = clip_box01(box)
                if box_area(eb) > 0:
                    return eb
        b = sp.get("effective_bbox_norm_xyxy")
        if isinstance(b, (list, tuple)) and len(b) == 4 and box_area(b) > 0:
            return clip_box01(b)

    union_box = subject_set.get("union_box_xyxy")
    if isinstance(union_box, (list, tuple)) and len(union_box) == 4:
        ub = clip_box01(norm_box_xyxy(union_box, width, height))
        if box_area(ub) > 0:
            if mode in {"portrait_group", "object_multi", "scene_general", "scene_landscape", "background_texture_copyspace"}:
                return ub

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


def collect_c3_info(feat_rec: Dict[str, Any], width: int, height: int, cfg: TeacherScorerConfig) -> Dict[str, Any]:
    c3 = feat_rec.get("c3_pose", []) or []
    c6 = feat_rec.get("c6_gaze") if isinstance(feat_rec.get("c6_gaze"), dict) else {}
    c6_people = c6.get("people", []) if isinstance(c6.get("people"), list) else []
    c6_by_index: Dict[int, Dict[str, Any]] = {}
    for p in c6_people:
        if not isinstance(p, dict):
            continue
        idx = int(safe_float(p.get("person_index", -1), -1.0))
        if idx >= 0:
            c6_by_index[idx] = p

    face_boxes: List[List[float]] = []
    head_y_norm: List[float] = []
    head_guard_y_norm: List[float] = []
    head_guard_margin_norm: List[float] = []
    gaze_entries: List[Dict[str, Any]] = []
    keypoints_norm: List[List[List[float]]] = []

    for idx, human in enumerate(c3):
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
                head_y = clamp(fy1 - float(cfg.head_top_face_expand_alpha) * fh, 0.0, 1.0)
                guard_margin = clamp(
                    max(float(cfg.head_top_min_margin), float(cfg.head_top_face_margin_alpha) * fh),
                    float(cfg.head_top_min_margin),
                    0.08,
                )
                head_y_norm.append(head_y)
                head_guard_y_norm.append(head_y)
                head_guard_margin_norm.append(guard_margin)

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
                    head_y_kp = clamp(min(top_candidates) - float(cfg.head_top_kp_expand), 0.0, 1.0)
                    head_y_norm.append(head_y_kp)
                    head_guard_y_norm.append(head_y_kp)
                    head_guard_margin_norm.append(float(cfg.head_top_min_margin))

        # Gaze/headpose proxy + C6 override/fallback.
        hp = human.get("headpose_gaze") if isinstance(human.get("headpose_gaze"), dict) else {}
        c6_person = c6_by_index.get(idx, {})
        if isinstance(c6_person, dict) and c6_person:
            hp_conf = safe_float(hp.get("conf", 0.0))
            c6_conf = safe_float(c6_person.get("conf", 0.0))
            hp_source = str(hp.get("source", "")).strip().lower()
            prefer_c6 = (not hp) or (c6_conf >= hp_conf) or ("proxy" in hp_source)
            if prefer_c6:
                hp = {
                    "yaw_proxy": safe_float(c6_person.get("yaw_proxy", 0.0)),
                    "pitch_proxy": safe_float(c6_person.get("pitch_proxy", 0.0)),
                    "roll_deg": safe_float(c6_person.get("roll_deg", 0.0)),
                    "gaze_dir": str(c6_person.get("gaze_dir", "unknown")),
                    "conf": c6_conf,
                    "source": str(c6_person.get("source", "c6_gaze")),
                }

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
            else:
                c6_head = c6_person.get("head_bbox_norm_xyxy") if isinstance(c6_person, dict) else None
                if isinstance(c6_head, (list, tuple)) and len(c6_head) == 4:
                    ch = clip_box01(c6_head)
                    if box_area(ch) > 0.0:
                        anchor_x, anchor_y = box_center(ch)

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
        "head_guard_y_norm": head_guard_y_norm,
        "head_guard_margin_norm": head_guard_margin_norm,
        "gaze_entries": gaze_entries,
        "keypoints_norm": keypoints_norm,
        "num_people": max(len(c3), len(c6_by_index)),
    }


def collect_c5_info(feat_rec: Dict[str, Any]) -> Dict[str, Any]:
    c5 = feat_rec.get("c5_geom")
    if isinstance(c5, list):
        c5 = safe_first(c5)
    if not isinstance(c5, dict):
        return {
            "horizon_y_norm": None,
            "horizon_conf": 0.0,
            "horizon_exists_prob": 0.0,
            "roll_deg": None,
            "line_norm_xyxy": None,
            "symmetry": 0.0,
        }

    hr = c5.get("horizon_roll") if isinstance(c5.get("horizon_roll"), dict) else {}
    sym = c5.get("symmetry") if isinstance(c5.get("symmetry"), dict) else {}
    horizon_conf = safe_float(hr.get("conf", c5.get("horizon_conf", 0.0)))
    return {
        "horizon_y_norm": hr.get("horizon_y_norm"),
        "horizon_conf": horizon_conf,
        "horizon_exists_prob": horizon_conf,
        "roll_deg": hr.get("roll_deg"),
        "line_norm_xyxy": hr.get("line_norm_xyxy"),
        "symmetry": clamp(safe_float(sym.get("score", 0.0)), 0.0, 1.0),
    }


def _norm_box_maybe_xyxy(box: Sequence[Any], width: int, height: int) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    vals = [safe_float(v) for v in box]
    if max(abs(v) for v in vals) <= 1.5:
        b = clip_box01(vals)
    else:
        b = norm_box_xyxy(vals, width, height)
    if box_area(b) <= 0.0:
        return None
    return b


def collect_c4_info(feat_rec: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    raw_c4 = feat_rec.get("c4_ocr")
    c4_meta = feat_rec.get("c4_ocr_meta") if isinstance(feat_rec.get("c4_ocr_meta"), dict) else {}

    raw_boxes: List[Any] = []
    method = str(c4_meta.get("method", "unknown"))
    coverage_ratio = clamp(safe_float(c4_meta.get("coverage_ratio", 0.0)), 0.0, 1.0)
    text_overlay_likely = bool(
        feat_rec.get("text_overlay_likely")
        or feat_rec.get("has_text_overlay")
        or feat_rec.get("ocr_text_overlay_likely")
    )

    if isinstance(raw_c4, dict):
        if isinstance(raw_c4.get("boxes"), list):
            raw_boxes = list(raw_c4.get("boxes"))
        method = str(raw_c4.get("method", method))
        coverage_ratio = clamp(safe_float(raw_c4.get("coverage_ratio", coverage_ratio)), 0.0, 1.0)
        text_overlay_likely = bool(raw_c4.get("text_overlay_likely", text_overlay_likely))
    elif isinstance(raw_c4, list):
        raw_boxes = list(raw_c4)

    boxes_norm: List[List[float]] = []
    for it in raw_boxes:
        b: Optional[List[float]] = None
        if isinstance(it, dict):
            b = _norm_box_maybe_xyxy(it.get("box_norm_xyxy", []), width=width, height=height)
            if b is None:
                b = _norm_box_maybe_xyxy(it.get("box_xyxy", []), width=width, height=height)
            if b is None:
                quad = it.get("box")
                if isinstance(quad, (list, tuple)) and len(quad) >= 4:
                    xs = [safe_float(p[0]) for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2]
                    ys = [safe_float(p[1]) for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2]
                    if xs and ys:
                        b = _norm_box_maybe_xyxy([min(xs), min(ys), max(xs), max(ys)], width=width, height=height)
        else:
            b = _norm_box_maybe_xyxy(it, width=width, height=height)
        if b is not None:
            boxes_norm.append(b)

    count_alias = int(
        max(
            safe_float(feat_rec.get("ocr_text_boxes", 0), 0.0),
            safe_float(feat_rec.get("ocr_num_boxes", 0), 0.0),
            safe_float(feat_rec.get("ocr_box_count", 0), 0.0),
            safe_float(feat_rec.get("c4_text_boxes", 0), 0.0),
            safe_float(feat_rec.get("c4_ocr_boxes", 0), 0.0),
        )
    )
    num_boxes = max(len(boxes_norm), max(0, count_alias))
    if len(boxes_norm) > 0:
        coverage_ratio = clamp(sum(box_area(b) for b in boxes_norm), 0.0, 1.0)
    if not text_overlay_likely:
        text_overlay_likely = bool(num_boxes >= 3 or coverage_ratio >= 0.015)

    return {
        "boxes_norm": boxes_norm,
        "num_boxes": int(num_boxes),
        "coverage_ratio": float(coverage_ratio),
        "text_overlay_likely": bool(text_overlay_likely),
        "method": method,
    }


def compute_text_penalty(
    crop: Sequence[float],
    c4_info: Dict[str, Any],
    route: Dict[str, Any],
    cfg: TeacherScorerConfig,
) -> Dict[str, Any]:
    boxes = [clip_box01(b) for b in (c4_info.get("boxes_norm", []) or []) if isinstance(b, (list, tuple)) and len(b) == 4]
    num_boxes = int(c4_info.get("num_boxes", len(boxes)) or len(boxes))
    text_overlay_likely = bool(c4_info.get("text_overlay_likely", False))
    method = str(c4_info.get("method", "unknown"))
    coverage_ratio = clamp(safe_float(c4_info.get("coverage_ratio", 0.0)), 0.0, 1.0)
    mode = str(route.get("subject_mode", "")).strip().lower()
    policy = str(route.get("policy_id", "")).strip().lower()
    text_mode = (mode == "text_document") or (policy == "text_v1")

    target_keep_ratio = float(cfg.text_keep_min_default)
    if text_overlay_likely:
        target_keep_ratio = max(target_keep_ratio, float(cfg.text_keep_min_overlay))
    if text_mode:
        target_keep_ratio = max(target_keep_ratio, float(cfg.text_keep_min_text_document))

    severe_max = float(cfg.text_severe_ratio_max_text_document if text_mode else cfg.text_severe_ratio_max_default)

    if not boxes:
        return {
            "p_text": 0.0,
            "pass": True,
            "available": False,
            "text_keep_ratio": 1.0,
            "cut_box_ratio": 0.0,
            "severe_cut_ratio": 0.0,
            "num_boxes": int(num_boxes),
            "target_keep_ratio": float(target_keep_ratio),
            "severe_ratio_max": float(severe_max),
            "text_overlay_likely": bool(text_overlay_likely),
            "coverage_ratio": float(coverage_ratio),
            "method": method,
        }

    keep_ratios: List[float] = []
    total_area = 0.0
    kept_area = 0.0
    for b in boxes:
        a = box_area(b)
        if a <= 0.0:
            continue
        k = inter_area(b, crop) / max(1e-8, a)
        keep_ratios.append(clamp(k, 0.0, 1.0))
        total_area += a
        kept_area += inter_area(b, crop)

    if total_area <= 0.0 or not keep_ratios:
        return {
            "p_text": 0.0,
            "pass": True,
            "available": False,
            "text_keep_ratio": 1.0,
            "cut_box_ratio": 0.0,
            "severe_cut_ratio": 0.0,
            "num_boxes": int(num_boxes),
            "target_keep_ratio": float(target_keep_ratio),
            "severe_ratio_max": float(severe_max),
            "text_overlay_likely": bool(text_overlay_likely),
            "coverage_ratio": float(coverage_ratio),
            "method": method,
        }

    keep_ratio = clamp(kept_area / max(1e-8, total_area), 0.0, 1.0)
    full_keep_thr = float(cfg.text_box_full_keep_thr)
    severe_thr = float(cfg.text_box_severe_keep_thr)

    full_keep_count = sum(1 for r in keep_ratios if r >= full_keep_thr)
    severe_cut_count = sum(1 for r in keep_ratios if r < severe_thr)
    n = max(1, len(keep_ratios))
    cut_box_ratio = clamp(1.0 - (full_keep_count / float(n)), 0.0, 1.0)
    severe_cut_ratio = clamp(severe_cut_count / float(n), 0.0, 1.0)

    p_text = (
        float(cfg.text_penalty_keep_weight) * (1.0 - keep_ratio)
        + float(cfg.text_penalty_cut_weight) * cut_box_ratio
        + float(cfg.text_penalty_severe_weight) * severe_cut_ratio
    )
    p_text = clamp(p_text, 0.0, 3.0)

    passed = bool((keep_ratio >= target_keep_ratio) and (severe_cut_ratio <= severe_max))
    return {
        "p_text": float(p_text),
        "pass": passed,
        "available": True,
        "text_keep_ratio": float(keep_ratio),
        "cut_box_ratio": float(cut_box_ratio),
        "severe_cut_ratio": float(severe_cut_ratio),
        "num_boxes": int(num_boxes),
        "target_keep_ratio": float(target_keep_ratio),
        "severe_ratio_max": float(severe_max),
        "text_overlay_likely": bool(text_overlay_likely),
        "coverage_ratio": float(coverage_ratio),
        "method": method,
    }


def summarize_feature_stage_debug(
    *,
    feat_rec: Dict[str, Any],
    c3_info: Dict[str, Any],
    c4_info: Dict[str, Any],
    c5_info: Dict[str, Any],
    c2_person_count: int,
    c1_text_embed: Optional[np.ndarray],
) -> Dict[str, Any]:
    c2_seg = feat_rec.get("c2_seg", [])
    c2_det = feat_rec.get("c2_det", [])
    c3_pose = feat_rec.get("c3_pose", [])

    c5_raw = feat_rec.get("c5_geom")
    if isinstance(c5_raw, list):
        c5_raw = safe_first(c5_raw)
    if not isinstance(c5_raw, dict):
        c5_raw = {}
    c5_hr = c5_raw.get("horizon_roll") if isinstance(c5_raw.get("horizon_roll"), dict) else {}

    c6_raw = feat_rec.get("c6_gaze")
    if not isinstance(c6_raw, dict):
        c6_raw = {}

    c1_dim = 0
    if isinstance(c1_text_embed, np.ndarray) and c1_text_embed.ndim == 1:
        c1_dim = int(c1_text_embed.shape[0])

    return {
        "c1": {
            "text_embed_available": bool(c1_dim > 0),
            "text_embed_dim": int(c1_dim),
        },
        "c2": {
            "seg_count": int(len(c2_seg)) if isinstance(c2_seg, list) else 0,
            "det_count": int(len(c2_det)) if isinstance(c2_det, list) else 0,
            "person_count_proxy": int(max(0, c2_person_count)),
        },
        "c3": {
            "pose_count": int(len(c3_pose)) if isinstance(c3_pose, list) else 0,
            "num_people_effective": int(max(0, c3_info.get("num_people", 0))),
            "face_box_count": int(len(c3_info.get("face_boxes", []))),
            "keypoint_set_count": int(len(c3_info.get("keypoints_norm", []))),
            "gaze_entry_count": int(len(c3_info.get("gaze_entries", []))),
        },
        "c4": {
            "available": bool(c4_info.get("num_boxes", 0) > 0),
            "num_boxes": int(max(0, safe_float(c4_info.get("num_boxes", 0), 0.0))),
            "coverage_ratio": float(clamp(safe_float(c4_info.get("coverage_ratio", 0.0), 0.0), 0.0, 1.0)),
            "text_overlay_likely": bool(c4_info.get("text_overlay_likely", False)),
            "method": str(c4_info.get("method", "unknown")),
        },
        "c5": {
            "backend_runtime": str(c5_raw.get("backend_runtime", "unknown")),
            "horizon_method": str(c5_hr.get("method", "unknown")),
            "horizon_conf": float(c5_info.get("horizon_conf", 0.0)),
            "roll_deg": c5_info.get("roll_deg"),
            "symmetry_score": float(c5_info.get("symmetry", 0.0)),
        },
        "c6": {
            "backend_runtime": str(c6_raw.get("backend_runtime", "unknown")),
            "method": str(c6_raw.get("method", "unknown")),
            "people_count": int(len(c6_raw.get("people", []))) if isinstance(c6_raw.get("people"), list) else 0,
            "conf": float(safe_float(c6_raw.get("conf", 0.0), 0.0)),
        },
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


def _decode_subject_support_map(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    grid_size = int(safe_float(payload.get("support_grid_size", 0), 0.0))
    mass_b64 = payload.get("support_mass_grid_f16_b64")
    occ_b64 = payload.get("support_occ_grid_u8_b64")
    if grid_size <= 0 or not isinstance(mass_b64, str) or not mass_b64 or not isinstance(occ_b64, str) or not occ_b64:
        return None
    try:
        mass_grid = np.frombuffer(base64.b64decode(mass_b64.encode("ascii")), dtype=np.float16).astype(np.float32)
        occ_grid = np.frombuffer(base64.b64decode(occ_b64.encode("ascii")), dtype=np.uint8).astype(np.float32)
    except Exception:
        return None
    if mass_grid.size != grid_size * grid_size or occ_grid.size != grid_size * grid_size:
        return None
    mass_grid = mass_grid.reshape((grid_size, grid_size))
    occ_grid = occ_grid.reshape((grid_size, grid_size))
    mass_sum = float(mass_grid.sum())
    if mass_sum <= 1e-8:
        return None
    mass_grid /= mass_sum
    occ_grid = (occ_grid > 0.0).astype(np.float32)
    centroid = payload.get("weighted_centroid_xy_norm")
    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        cx = clamp(safe_float(centroid[0], 0.5), 0.0, 1.0)
        cy = clamp(safe_float(centroid[1], 0.5), 0.0, 1.0)
    else:
        ys, xs = np.mgrid[0:grid_size, 0:grid_size]
        cx = float((mass_grid * (xs + 0.5)).sum() / max(1e-8, float(mass_grid.sum()))) / float(grid_size)
        cy = float((mass_grid * (ys + 0.5)).sum() / max(1e-8, float(mass_grid.sum()))) / float(grid_size)
    return {
        "mode": "support_map",
        "grid_size": int(grid_size),
        "mass_grid": mass_grid,
        "occ_grid": occ_grid,
        "centroid_xy_norm": (float(cx), float(cy)),
        "foreground_area_ratio": clamp(safe_float(payload.get("foreground_area_ratio", 0.0), 0.0), 0.0, 1.0),
    }


def build_subject_support_context(
    subject_box: Sequence[float],
    feat_rec: Dict[str, Any],
    route: Dict[str, Any],
) -> Dict[str, Any]:
    effective_subject_region = (
        route.get("effective_subject_region", {})
        if isinstance(route.get("effective_subject_region"), dict)
        else {}
    )
    score_mode = str(effective_subject_region.get("score_mode", route.get("flags", {}).get("subject_score_mode", "")) or "")
    support_map_enabled = bool(effective_subject_region.get("support_map_enabled", False)) or score_mode == "support_map"
    payloads: List[Dict[str, Any]] = []
    saliency_summary = effective_subject_region.get("saliency_summary")
    if isinstance(saliency_summary, dict):
        payloads.append(saliency_summary)
    c7_saliency = feat_rec.get("c7_saliency")
    if isinstance(c7_saliency, dict):
        payloads.append(c7_saliency)
    if support_map_enabled:
        for payload in payloads:
            decoded = _decode_subject_support_map(payload)
            if decoded is not None:
                decoded["bbox_norm_xyxy"] = clip_box01(subject_box)
                decoded["support_map_enabled"] = True
                return decoded
    return {
        "mode": "bbox",
        "bbox_norm_xyxy": clip_box01(subject_box),
        "centroid_xy_norm": box_center(subject_box),
        "support_map_enabled": False,
    }


def _crop_grid_slices(grid_size: int, crop: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in crop]
    gx1 = max(0, min(grid_size, int(math.floor(x1 * grid_size))))
    gy1 = max(0, min(grid_size, int(math.floor(y1 * grid_size))))
    gx2 = max(gx1 + 1, min(grid_size, int(math.ceil(x2 * grid_size))))
    gy2 = max(gy1 + 1, min(grid_size, int(math.ceil(y2 * grid_size))))
    return gx1, gy1, gx2, gy2


def subject_coverage_soft(subject_support: Dict[str, Any], crop: Sequence[float]) -> Tuple[float, float]:
    mass_grid = subject_support.get("mass_grid")
    occ_grid = subject_support.get("occ_grid")
    if not isinstance(mass_grid, np.ndarray) or mass_grid.ndim != 2:
        bbox = subject_support.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
        return subject_coverage(bbox, crop)
    grid_size = int(mass_grid.shape[0])
    gx1, gy1, gx2, gy2 = _crop_grid_slices(grid_size, crop)
    mass_recall = float(mass_grid[gy1:gy2, gx1:gx2].sum())
    crop_area = max(1e-8, box_area(crop))
    if isinstance(occ_grid, np.ndarray) and occ_grid.shape == mass_grid.shape:
        occ_area_ratio = float(occ_grid[gy1:gy2, gx1:gx2].sum()) / float(occ_grid.size)
        subj_area_ratio = occ_area_ratio / crop_area
    else:
        subj_area_ratio = mass_recall / crop_area
    return clamp(mass_recall, 0.0, 1.0), clamp(subj_area_ratio, 0.0, 1.0)


def subject_border_touch_soft(subject_support: Dict[str, Any], crop: Sequence[float], margin_alpha: float = 0.03) -> bool:
    mass_grid = subject_support.get("mass_grid")
    occ_grid = subject_support.get("occ_grid")
    if not isinstance(mass_grid, np.ndarray) or mass_grid.ndim != 2:
        bbox = subject_support.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
        return subject_border_touch(bbox, crop, margin_alpha=margin_alpha)
    grid_size = int(mass_grid.shape[0])
    gx1, gy1, gx2, gy2 = _crop_grid_slices(grid_size, crop)
    inside_mass = float(mass_grid[gy1:gy2, gx1:gx2].sum())
    if inside_mass <= 1e-6:
        return False
    crop_w = max(1, gx2 - gx1)
    crop_h = max(1, gy2 - gy1)
    margin = max(1, int(math.ceil(margin_alpha * min(crop_w, crop_h))))
    edge_mask = np.zeros((crop_h, crop_w), dtype=bool)
    edge_mask[:margin, :] = True
    edge_mask[-margin:, :] = True
    edge_mask[:, :margin] = True
    edge_mask[:, -margin:] = True
    edge_mass_grid = mass_grid[gy1:gy2, gx1:gx2]
    edge_mass = float(edge_mass_grid[edge_mask].sum())
    if isinstance(occ_grid, np.ndarray) and occ_grid.shape == mass_grid.shape:
        edge_occ = float(occ_grid[gy1:gy2, gx1:gx2][edge_mask].sum()) / float(max(1, occ_grid.size))
    else:
        edge_occ = 0.0
    return bool(edge_mass >= 0.05 or edge_occ >= 0.01)


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


def compute_head_top_guard(
    crop: Sequence[float],
    head_y_norm_values: Sequence[float],
    head_margin_values: Sequence[float],
    min_margin: float,
) -> Dict[str, Any]:
    if not head_y_norm_values:
        return {
            "activated": False,
            "pass": True,
            "min_required_y1": None,
            "crop_y1": round(safe_float(crop[1], 0.0), 6),
            "violated_count": 0,
            "head_count": 0,
        }

    y1 = safe_float(crop[1], 0.0)
    limits: List[float] = []
    violated = 0
    for idx, y_head in enumerate(head_y_norm_values):
        margin = float(min_margin)
        if idx < len(head_margin_values):
            margin = max(float(min_margin), safe_float(head_margin_values[idx], float(min_margin)))
        lim = clamp(safe_float(y_head, 0.0) - margin, 0.0, 1.0)
        limits.append(lim)
        if y1 > lim:
            violated += 1

    min_required_y1 = min(limits) if limits else None
    ok = (min_required_y1 is None) or (y1 <= min_required_y1)
    return {
        "activated": True,
        "pass": bool(ok),
        "min_required_y1": round(float(min_required_y1), 6) if min_required_y1 is not None else None,
        "crop_y1": round(float(y1), 6),
        "violated_count": int(violated),
        "head_count": int(len(limits)),
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


def is_scene_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in SCENE_MODES or str(mode or "").strip().lower().startswith("scene")


def normalize_scene_subtype(scene_subtype: Any) -> str:
    subtype = str(scene_subtype or "").strip().lower()
    return subtype if subtype else SCENE_SUBTYPE_UNKNOWN


def horizon_thresholds_for_subtype(scene_subtype: str) -> Tuple[float, float]:
    subtype = normalize_scene_subtype(scene_subtype)
    if subtype == "scene_landscape_nature":
        return 0.20, 0.35
    if subtype == "scene_reflection_symmetry":
        return 0.20, 0.30
    if subtype == "scene_city_architecture":
        return 0.20, 0.50
    if subtype in {"scene_interior_architecture", "scene_structural_pattern"}:
        return 0.25, 0.45
    return 0.25, 0.45


def _clip_line_segment_to_rect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rect: Sequence[float],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    rx1, ry1, rx2, ry2 = [float(v) for v in rect]
    dx = x2 - x1
    dy = y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - rx1, rx2 - x1, y1 - ry1, ry2 - y1)
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) <= 1e-12:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            u1 = max(u1, t)
        else:
            if t < u1:
                return None
            u2 = min(u2, t)
    if u2 < u1:
        return None
    return ((x1 + u1 * dx, y1 + u1 * dy), (x1 + u2 * dx, y1 + u2 * dy))


def compute_horizon_reward(
    crop: Sequence[float],
    horizon_y_norm: Optional[float],
    horizon_conf: float,
    horizon_exists_prob: float,
    scene_subtype: str,
    line_norm_xyxy: Optional[Sequence[Any]],
    conf_thr: float,
    visibility_thr: float,
) -> Dict[str, Any]:
    if horizon_y_norm is None or not math.isfinite(float(horizon_y_norm)):
        return {
            "dist": None,
            "reward": 0.0,
            "active": False,
            "value": None,
            "pass": True,
            "state": "none",
            "exists_prob": float(clamp(horizon_exists_prob, 0.0, 1.0)),
            "visible_ratio": 0.0,
            "target_set": [],
        }

    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    hh = max(1e-8, y2 - y1)
    y_local = (float(horizon_y_norm) - y1) / hh
    subtype = normalize_scene_subtype(scene_subtype)
    target_set = list(SCENE_HORIZON_TARGETS.get(subtype, ()))
    if not target_set:
        target_set = [1.0 / 3.0, 2.0 / 3.0]
    exists_thr, subtype_conf_thr = horizon_thresholds_for_subtype(subtype)
    effective_conf_thr = min(float(conf_thr), float(subtype_conf_thr))

    exists_prob = clamp(horizon_exists_prob, 0.0, 1.0)
    conf = clamp(horizon_conf, 0.0, 1.0)
    if exists_prob < exists_thr:
        state = "none"
    elif conf < effective_conf_thr:
        state = "weak"
    else:
        state = "strong"

    d = min(abs(y_local - t) for t in target_set)
    visible_ratio = 0.0
    if isinstance(line_norm_xyxy, (list, tuple)) and len(line_norm_xyxy) == 4:
        clipped = _clip_line_segment_to_rect(
            (safe_float(line_norm_xyxy[0]), safe_float(line_norm_xyxy[1])),
            (safe_float(line_norm_xyxy[2]), safe_float(line_norm_xyxy[3])),
            crop,
        )
        if clipped is not None:
            seg_len = math.hypot(clipped[1][0] - clipped[0][0], clipped[1][1] - clipped[0][1])
            visible_ratio = clamp(seg_len / max(1e-8, x2 - x1), 0.0, 1.0)
    if visible_ratio <= 0.0:
        visible_ratio = 1.0 if (0.0 <= y_local <= 1.0) else 0.0

    active = state != "none"
    pos_reward = reward_dist(d, 0.08) if active else 0.0
    vis_reward = clamp(visible_ratio / max(1e-8, visibility_thr), 0.0, 1.0) if active else 0.0
    state_weight = 1.0 if state == "strong" else 0.45
    reward = state_weight * (0.70 * pos_reward + 0.30 * vis_reward)
    pass_flag = active and d <= 0.08 and visible_ratio >= 0.35
    return {
        "dist": float(d),
        "reward": float(reward),
        "active": bool(active),
        "value": float(y_local),
        "pass": bool(pass_flag) if active else True,
        "state": state,
        "exists_prob": float(exists_prob),
        "visible_ratio": float(visible_ratio),
        "target_set": [round(float(v), 4) for v in target_set],
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
        "teach": cfg.lambda_teach,
        "ar_free": cfg.lambda_ar_free,
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


def free_ar_prior_penalty(
    *,
    ar: float,
    route: Dict[str, Any],
    cfg: TeacherScorerConfig,
) -> float:
    if not math.isfinite(ar) or ar <= 1e-8:
        return 0.0

    # Base weak regularizer: penalize only very extreme AR in log-space.
    p = max(0.0, abs(math.log(ar)) - float(cfg.free_ar_log_tau))
    mode = str(route.get("subject_mode", "")).strip().lower()

    # Route-aware asymmetry.
    if mode.startswith("portrait") and ar > float(cfg.free_ar_portrait_max):
        p += float(cfg.free_ar_bias_scale) * (ar - float(cfg.free_ar_portrait_max))
    if mode.startswith("scene") and ar < float(cfg.free_ar_scene_min):
        p += float(cfg.free_ar_bias_scale) * (float(cfg.free_ar_scene_min) - ar)
    return float(max(0.0, p))


def classify_subject_coverage_label(cov: float) -> str:
    c = clamp(safe_float(cov, 0.0), 0.0, 1.0)
    if c >= 0.95:
        return "excellent"
    if c >= 0.85:
        return "good"
    if c >= 0.70:
        return "marginal"
    return "poor"


def classify_subject_scale_label(subj_area_ratio: float, target_range: Tuple[float, float]) -> str:
    lo, hi = target_range
    v = clamp(safe_float(subj_area_ratio, 0.0), 0.0, 1.0)
    if v < float(lo):
        return "too_loose"
    if v > float(hi):
        return "too_tight"
    return "ideal_scale"


def classify_joint_cut_label(joint_cut_score: float, severe_count: int) -> str:
    score = max(0.0, safe_float(joint_cut_score, 0.0))
    sev = int(max(0, severe_count))
    if sev >= 2 or score > 0.35:
        return "joint_cut_severe"
    if sev > 0 or score > 0.10:
        return "joint_cut_mild"
    return "no_joint_cut"


def classify_headroom_label(value: Optional[float], target_range: Sequence[float]) -> str:
    if value is None:
        return "headroom_na"
    if not isinstance(target_range, (list, tuple)) or len(target_range) < 2:
        return "headroom_na"
    hmin = safe_float(target_range[0], 0.0)
    hmax = safe_float(target_range[1], 1.0)
    v = safe_float(value, 0.0)
    if v < hmin:
        return "headroom_tight"
    if v > hmax:
        return "headroom_loose"
    return "headroom_ok"


def classify_lookroom_label(value: Optional[float], target_range: Sequence[float], gaze_dir: str) -> str:
    g = str(gaze_dir).strip().lower()
    if g not in {"left", "right"} or value is None:
        return "lookroom_na"
    if not isinstance(target_range, (list, tuple)) or len(target_range) < 2:
        return "lookroom_na"
    lmin = safe_float(target_range[0], 0.0)
    lmax = safe_float(target_range[1], 999.0)
    v = safe_float(value, 0.0)
    if v < lmin:
        return "lookroom_insufficient"
    if v > lmax:
        return "lookroom_excessive"
    return "lookroom_adequate"


def classify_text_keep_label(text_eval: Dict[str, Any]) -> str:
    if not bool(text_eval.get("available", False)):
        return "text_na"
    keep_ratio = clamp(safe_float(text_eval.get("text_keep_ratio", 1.0), 1.0), 0.0, 1.0)
    target_keep = clamp(safe_float(text_eval.get("target_keep_ratio", 0.9), 0.9), 0.0, 1.0)
    severe = clamp(safe_float(text_eval.get("severe_cut_ratio", 0.0), 0.0), 0.0, 1.0)
    severe_max = clamp(safe_float(text_eval.get("severe_ratio_max", 0.2), 0.2), 0.0, 1.0)
    if keep_ratio >= target_keep and severe <= severe_max:
        return "text_preserved"
    if keep_ratio >= 0.60:
        return "text_partial"
    return "text_lost"


def classify_rule_strength_label(
    dist: Optional[float],
    *,
    strong_thr: float,
    strong_label: str,
    weak_label: str,
    na_label: str,
) -> str:
    d = safe_optional_float(dist)
    if d is None:
        return na_label
    if d <= float(strong_thr):
        return strong_label
    return weak_label


def classify_horizon_label(horizon: Dict[str, Any], third_tau: float) -> str:
    if not isinstance(horizon, dict):
        return "horizon_na"
    if str(horizon.get("state", "")).strip() == "weak":
        return "horizon_weak"
    if not bool(horizon.get("active", False)) or horizon.get("value") is None:
        return "horizon_na"
    if bool(horizon.get("pass", False)):
        return "horizon_on_target"
    y = safe_optional_float(horizon.get("value"))
    if y is not None and abs(y - 0.5) <= max(0.10, 1.5 * float(third_tau)):
        return "horizon_middle"
    return "horizon_off"


def classify_context_label(context_value: float, target_range: Tuple[float, float]) -> str:
    lo, hi = target_range
    v = clamp(safe_float(context_value, 0.0), 0.0, 1.0)
    if v < max(0.0, float(lo) * 0.60):
        return "context_poor"
    if v < float(lo):
        return "context_partial"
    if v > float(hi):
        return "context_excessive"
    return "context_preserved"


def classify_roll_label(roll_deg: Optional[float], roll_thr: float) -> str:
    if roll_deg is None:
        return "roll_na"
    return "needs_leveling" if abs(safe_float(roll_deg, 0.0)) > float(roll_thr) else "level_ok"


def compute_context_preserved(
    *,
    crop_area: float,
    subject_area_in_crop_ratio: float,
    subject_box_area_full: float,
    scene_bonus: float,
    text_bonus: float,
    negative_space_bonus: float,
) -> float:
    subject_area_in_crop = clamp(subject_area_in_crop_ratio, 0.0, 1.0) * clamp(crop_area, 0.0, 1.0)
    non_subject_crop = max(0.0, clamp(crop_area, 0.0, 1.0) - subject_area_in_crop)
    denom = max(1e-6, 1.0 - clamp(subject_box_area_full, 0.0, 0.98))
    base = clamp(non_subject_crop / denom, 0.0, 1.0)
    return clamp(0.70 * base + 0.15 * scene_bonus + 0.10 * text_bonus + 0.05 * negative_space_bonus, 0.0, 1.0)


def compute_copyspace_reward(
    *,
    crop: Sequence[float],
    subject_box: Sequence[float],
    route: Dict[str, Any],
) -> Dict[str, Any]:
    side = str(route.get("copyspace_side", "unknown")).strip().lower()
    if side not in {"left", "right", "top", "bottom"}:
        return {"reward": 0.0, "blank_ratio_keep": 0.0, "side_consistency": 0.0, "intrusion": 0.0}

    x1, y1, x2, y2 = [safe_float(v) for v in crop]
    sx1, sy1, sx2, sy2 = [safe_float(v) for v in subject_box]
    crop_w = max(1e-8, x2 - x1)
    crop_h = max(1e-8, y2 - y1)
    if side == "left":
        blank_ratio_keep = clamp((sx1 - x1) / crop_w, 0.0, 1.0)
        intrusion = clamp((sx2 - x1) / crop_w, 0.0, 1.0)
    elif side == "right":
        blank_ratio_keep = clamp((x2 - sx2) / crop_w, 0.0, 1.0)
        intrusion = clamp((x2 - sx1) / crop_w, 0.0, 1.0)
    elif side == "top":
        blank_ratio_keep = clamp((sy1 - y1) / crop_h, 0.0, 1.0)
        intrusion = clamp((sy2 - y1) / crop_h, 0.0, 1.0)
    else:
        blank_ratio_keep = clamp((y2 - sy2) / crop_h, 0.0, 1.0)
        intrusion = clamp((y2 - sy1) / crop_h, 0.0, 1.0)
    side_consistency = 1.0 if blank_ratio_keep >= 0.18 else clamp(blank_ratio_keep / 0.18, 0.0, 1.0)
    intrusion_pen = 1.0 - clamp(blank_ratio_keep * 1.5, 0.0, 1.0)
    reward = clamp(0.55 * blank_ratio_keep + 0.30 * side_consistency - 0.25 * intrusion_pen, -1.0, 1.0)
    return {
        "reward": float(reward),
        "blank_ratio_keep": float(blank_ratio_keep),
        "side_consistency": float(side_consistency),
        "intrusion": float(intrusion_pen),
    }


def classify_crop_tightness_label(area_ratio: float) -> str:
    a = clamp(safe_float(area_ratio, 0.0), 0.0, 1.0)
    if a < 0.35:
        return "tight_crop"
    if a > 0.75:
        return "wide_crop"
    return "balanced_crop"


def build_why_text_template(
    *,
    checklist_labels: Dict[str, str],
    why_tags: Sequence[str],
    is_freeform: bool,
) -> str:
    parts: List[str] = []
    cov = str(checklist_labels.get("subject_coverage", ""))
    if cov in {"excellent", "good"}:
        parts.append("subject is well preserved")
    elif cov == "marginal":
        parts.append("subject coverage is marginal")
    elif cov == "poor":
        parts.append("subject coverage is poor")

    scale = str(checklist_labels.get("subject_scale", ""))
    if scale == "ideal_scale":
        parts.append("subject scale is in the intended range")
    elif scale == "too_tight":
        parts.append("crop is tight around the subject")
    elif scale == "too_loose":
        parts.append("crop is relatively loose around the subject")

    if "rule_of_thirds" in why_tags:
        parts.append("placement is close to rule-of-thirds")
    elif "centered_subject" in why_tags:
        parts.append("subject is centered with composition stability")

    head = str(checklist_labels.get("headroom", ""))
    if head == "headroom_ok":
        parts.append("headroom is acceptable")
    elif head == "headroom_tight":
        parts.append("headroom is tight")

    look = str(checklist_labels.get("lookroom", ""))
    if look == "lookroom_adequate":
        parts.append("lookroom is adequate")
    elif look == "lookroom_insufficient":
        parts.append("lookroom is insufficient")

    text = str(checklist_labels.get("text_keep_ratio", ""))
    if text == "text_preserved":
        parts.append("text region is preserved")
    elif text == "text_lost":
        parts.append("text region is heavily cropped")

    if "teacher_consensus" in why_tags:
        parts.append("selection is aligned with public teacher consensus")
    if is_freeform:
        parts.append("free-form AR prior is applied")

    if not parts:
        return "Selection follows deterministic safety and composition checks."
    # Keep a concise deterministic sentence.
    uniq_parts: List[str] = []
    seen = set()
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        uniq_parts.append(p)
    return "; ".join(uniq_parts[:6]) + "."


def _resolve_subject_routing_hint(cand_rec: Dict[str, Any], feat_rec: Dict[str, Any]) -> Dict[str, Any]:
    # Feature-level routing is authoritative on reruns because candidate jsonl may
    # carry stale routing hints baked at generation time. Candidate-level fields are
    # only used as backfill when the refreshed feature record lacks them.
    cand_routing = cand_rec.get("routing", {})
    feat_routing = feat_rec.get("routing", {})
    out: Dict[str, Any] = {}
    if isinstance(cand_routing, dict):
        out.update(cand_routing)
    if isinstance(feat_routing, dict):
        out.update(feat_routing)
    return out


def _fallback_subject_mode_when_no_person(
    *,
    mode: str,
    flags: Dict[str, Any],
    routing_hint: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    reasons: List[str] = ["scorer_guard_no_person_for_portrait"]
    if not mode.startswith("portrait"):
        return mode, "generic_v1", reasons

    router_signals = routing_hint.get("router_signals", {}) if isinstance(routing_hint.get("router_signals"), dict) else {}
    has_text_signal = bool(router_signals.get("text_signal", False)) or bool(flags.get("subject_mode_has_text_heavy", False))
    copyspace_allowed = bool(router_signals.get("copyspace_allowed", False))
    has_copyspace_tag = bool(flags.get("subject_mode_has_copyspace_tag", False))

    if has_text_signal:
        reasons.append("fallback_text_signal")
        return "text_document", "text_v1", reasons
    if copyspace_allowed or has_copyspace_tag:
        reasons.append("fallback_copyspace_signal")
        return "background_texture_copyspace", "copyspace_v1", reasons
    if bool(flags.get("is_landscape_scene", False)):
        reasons.append("fallback_scene_signal")
        return "scene_general", "scene_v1", reasons
    if bool(flags.get("is_product", False)) or bool(flags.get("is_isolated_packshot", False)):
        reasons.append("fallback_object_signal")
        return "object_single", "object_v1", reasons

    reasons.append("fallback_ambiguous")
    return "other_ambiguous", "generic_v1", reasons


def apply_subject_policy_overrides(
    route: Dict[str, Any],
    subject_mode: str,
    policy_id: str,
    routing_hint: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(route)
    lambdas = dict(route.get("lambdas", {}))
    flags = dict(route.get("flags", {}))

    # Keep compatibility with existing flags while exposing richer routing info.
    if isinstance(routing_hint.get("subject_mode_flags"), dict):
        sm_flags = routing_hint.get("subject_mode_flags", {})
        flags["subject_mode_has_person"] = bool(sm_flags.get("has_person", False))
        flags["subject_mode_has_text_heavy"] = bool(sm_flags.get("has_text_heavy", False))
        flags["subject_mode_has_copyspace_tag"] = bool(sm_flags.get("has_copyspace_tag", False))
        flags["subject_mode_is_background_like"] = bool(sm_flags.get("is_background_like", False))
        flags["subject_mode_contextual_tiny_human"] = bool(sm_flags.get("contextual_tiny_human", False))
    if isinstance(routing_hint.get("copyspace"), dict):
        flags["has_copy_space"] = bool(routing_hint.get("copyspace", {}).get("gate_passed", flags.get("has_copy_space", False)))

    mode = str(subject_mode or "").strip().lower()
    policy = str(policy_id or "").strip().lower()
    scene_subtype = normalize_scene_subtype(routing_hint.get("scene_subtype"))
    copyspace_meta = routing_hint.get("copyspace", {}) if isinstance(routing_hint.get("copyspace"), dict) else {}
    effective_subject_region = (
        routing_hint.get("effective_subject_region", {})
        if isinstance(routing_hint.get("effective_subject_region"), dict)
        else {}
    )
    effective_subject_state = str(effective_subject_region.get("state", "") or "")
    effective_score_mode = str(effective_subject_region.get("score_mode", "") or "")
    effective_repr_type = str(effective_subject_region.get("subject_repr_type", "") or "")
    subject_reliability = clamp(safe_float(effective_subject_region.get("subject_reliability", 1.0), 1.0), 0.0, 1.0)
    subject_placeholder_flag = bool(effective_subject_region.get("placeholder_flag", False))
    subject_terms_neutralized = effective_score_mode == "neutralized"
    if not mode:
        mode = "other_ambiguous"
    if not policy:
        policy = "generic_v1"

    # Extra scorer-side sanity guard for stale/misaligned routed features.
    # If portrait mode has no person signal, force a safer non-portrait fallback.
    router_signals = routing_hint.get("router_signals", {}) if isinstance(routing_hint.get("router_signals"), dict) else {}
    signal_num_person = int(
        safe_float(
            router_signals.get("num_person", route.get("num_people", 0)),
            0.0,
        )
    )
    if mode.startswith("portrait") and signal_num_person <= 0:
        fb_mode, fb_policy, fb_reasons = _fallback_subject_mode_when_no_person(
            mode=mode,
            flags=flags,
            routing_hint=routing_hint,
        )
        mode = fb_mode
        policy = fb_policy
        out["shot_type"] = "unknown"
        if route.get("num_people", 0) <= 0:
            out["num_people"] = 0
        existing = (
            routing_hint.get("subject_mode_reasons", [])
            if isinstance(routing_hint.get("subject_mode_reasons"), list)
            else []
        )
        out["subject_mode_reasons"] = list(existing) + fb_reasons
        out["subject_mode_conflict"] = True

    # Route-level schedules tuned by subject mode.
    if mode in {"portrait_single", "portrait_group"} or policy in {"portrait_single_v1", "portrait_group_v1"}:
        if mode == "portrait_group":
            out["shot_type"] = "group"
            out["tau_improve"] = max(float(out.get("tau_improve", 0.03)), 0.045)
            if lambdas.get("lr", 0.0) > 0.12:
                lambdas["lr"] = 0.12
            lambdas["cut"] = max(lambdas.get("cut", 0.0), 1.10)
        else:
            out["tau_improve"] = max(float(out.get("tau_improve", 0.03)), 0.03)
            lambdas["hr"] = max(lambdas.get("hr", 0.0), 0.25)
            lambdas["lr"] = max(lambdas.get("lr", 0.0), 0.20)
            lambdas["cut"] = max(lambdas.get("cut", 0.0), 1.05)
        out["w_area"] = max(float(out.get("w_area", 0.10)), 0.10)

    elif mode in {"object_single", "object_multi"} or policy in {"object_v1", "object_multi_v1"}:
        out["tau_improve"] = max(float(out.get("tau_improve", 0.03)), 0.02)
        out["w_area"] = max(float(out.get("w_area", 0.08)), 0.08)
        lambdas["cut"] = max(lambdas.get("cut", 0.0), 1.05)
        lambdas["ctx"] = max(lambdas.get("ctx", 0.0), 0.18)
        if mode == "object_multi":
            lambdas["cov"] = max(lambdas.get("cov", 0.0), 0.55)

    elif mode in SCENE_MODES or policy == "scene_v1":
        out["tau_improve"] = max(float(out.get("tau_improve", 0.035)), 0.055)
        out["w_area"] = min(float(out.get("w_area", 0.10)), 0.08)
        lambdas["cov"] = min(lambdas.get("cov", 0.0), 0.30)
        lambdas["comp"] = max(lambdas.get("comp", 0.0), 0.55)
        lambdas["ctx"] = max(lambdas.get("ctx", 0.0), 0.35)
        lambdas["hr"] = 0.0
        lambdas["lr"] = 0.0

    elif mode == "background_texture_copyspace" or policy == "copyspace_v1":
        out["tau_improve"] = max(float(out.get("tau_improve", 0.035)), 0.05)
        out["w_area"] = min(float(out.get("w_area", 0.10)), 0.08)
        flags["has_copy_space"] = True
        lambdas["cs"] = max(lambdas.get("cs", 0.0), 0.35)
        lambdas["ctx"] = max(lambdas.get("ctx", 0.0), 0.30)
        lambdas["hr"] = 0.0
        lambdas["lr"] = 0.0

    elif mode == "text_document" or policy == "text_v1":
        out["tau_improve"] = max(float(out.get("tau_improve", 0.035)), 0.05)
        flags["is_product"] = False
        lambdas["text"] = max(lambdas.get("text", 0.0), 0.40)
        lambdas["cut"] = max(lambdas.get("cut", 0.0), 1.00)
        lambdas["hr"] = 0.0
        lambdas["lr"] = 0.0

    if not mode.startswith("portrait"):
        out["shot_type"] = "unknown"

    scale_range, scale_region = subject_scale_target_range(
        subject_mode=mode,
        shot_type=str(out.get("shot_type", "unknown")),
        scene_subtype=scene_subtype,
        flags=flags,
        has_human_evidence=bool(out.get("has_human_evidence", False)),
    )
    ctx_range, ctx_region = context_keep_target_range(
        subject_mode=mode,
        scene_subtype=scene_subtype,
        flags=flags,
    )
    out["subject_scale_range"] = scale_range
    out["subject_scale_target_region"] = scale_region
    out["context_range"] = ctx_range
    out["context_target_region"] = ctx_region

    for k in list(lambdas.keys()):
        lambdas[k] = max(0.0, float(lambdas[k]))
    if subject_terms_neutralized:
        lambdas["cov"] = 0.0
    else:
        lambdas["cov"] = float(lambdas.get("cov", 0.0)) * subject_reliability
        lambdas["cut"] = float(lambdas.get("cut", 0.0)) * max(subject_reliability, 0.35)
        if subject_placeholder_flag and subject_reliability <= 0.25:
            lambdas["cov"] = 0.0

    out["flags"] = flags
    out["lambdas"] = lambdas
    out["subject_mode"] = mode
    out["policy_id"] = policy
    out["scene_subtype"] = scene_subtype
    out["scene_conf"] = safe_float(routing_hint.get("scene_conf", 0.0))
    out["copyspace"] = copyspace_meta
    out["copyspace_side"] = str(copyspace_meta.get("side", "unknown"))
    out["copyspace_quality"] = str(copyspace_meta.get("quality", "none"))
    out["subject_mode_conf"] = safe_float(routing_hint.get("subject_mode_conf", 0.0))
    if "subject_mode_reasons" not in out:
        out["subject_mode_reasons"] = (
            routing_hint.get("subject_mode_reasons", [])
            if isinstance(routing_hint.get("subject_mode_reasons"), list)
            else []
        )
    out["subject_set"] = routing_hint.get("subject_set", {}) if isinstance(routing_hint.get("subject_set"), dict) else {}
    if "subject_mode_conflict" not in out:
        out["subject_mode_conflict"] = bool(routing_hint.get("subject_mode_conflict", False))
    out["router_rule_id"] = str(routing_hint.get("router_rule_id", ""))
    out["router_signals"] = (
        routing_hint.get("router_signals", {})
        if isinstance(routing_hint.get("router_signals"), dict)
        else {}
    )
    out["effective_subject_region"] = effective_subject_region
    out["flags"]["subject_effective_state"] = effective_subject_state
    out["flags"]["subject_score_mode"] = effective_score_mode
    out["flags"]["subject_repr_type"] = effective_repr_type
    out["flags"]["subject_reliability"] = float(subject_reliability)
    out["flags"]["subject_placeholder_flag"] = bool(subject_placeholder_flag)
    out["flags"]["subject_terms_neutralized"] = bool(subject_terms_neutralized)
    out["flags"]["subject_support_map_enabled"] = bool(effective_subject_region.get("support_map_enabled", False))
    return out


def resolve_num_people_for_route(
    routing_hint: Dict[str, Any],
    c3_info: Dict[str, Any],
    c2_person_count: int,
) -> Dict[str, int]:
    routing_subject_set = routing_hint.get("subject_set", {}) if isinstance(routing_hint.get("subject_set"), dict) else {}
    routing_signals = routing_hint.get("router_signals", {}) if isinstance(routing_hint.get("router_signals"), dict) else {}
    face_count = len(c3_info.get("face_boxes", []) or [])
    routed_num_people = int(
        safe_float(
            routing_signals.get(
                "num_person",
                routing_subject_set.get("num_person", 0),
            ),
            0.0,
        )
    )
    detector_num_people = max(int(c2_person_count), int(face_count), int(routed_num_people))
    pose_num_people = int(safe_float(c3_info.get("num_people", 0), 0.0))
    num_people = detector_num_people if detector_num_people > 0 else pose_num_people
    return {
        "num_people": int(num_people),
        "c2_person_count": int(c2_person_count),
        "face_count": int(face_count),
        "routed_num_people": int(routed_num_people),
        "detector_num_people": int(detector_num_people),
        "pose_num_people": int(pose_num_people),
    }


def apply_expensive_score(
    candidate: Dict[str, Any],
    cfg: TeacherScorerConfig,
    w_area: float,
    expensive_signal: Optional[Dict[str, Any]],
) -> None:
    comps = candidate.get("scores", {}).get("components", {})
    cov = safe_float(comps.get("cov", 0.0))
    p_cut = safe_float(comps.get("p_cut", 0.0))
    p_text = safe_float(comps.get("p_text", 0.0))
    p_ar_free = safe_float(comps.get("p_ar_free", 0.0))
    r_edge = safe_float(comps.get("r_edge", 0.0))
    r_teach = safe_float(comps.get("r_teach", 0.0))
    area = safe_float(candidate.get("area_ratio", 0.0))

    a_backend = "disabled"
    a_prior_laion_w = None
    a_raw_laion = None
    a_norm_laion = None
    a_mean_nima = None
    a_std_nima = None
    a_norm_nima = None
    expensive_active = False
    if expensive_signal is not None:
        a_raw = safe_float(expensive_signal.get("aesthetic_raw", 0.0))
        a_norm = clamp(safe_float(expensive_signal.get("aesthetic_norm", 0.0)), 0.0, 1.0)
        ca_val = clamp(safe_float(expensive_signal.get("cosine_img_text", 0.0)), -1.0, 1.0)
        a_backend = str(expensive_signal.get("aesthetic_backend", "real")).strip().lower() or "real"
        a_prior_laion_w = safe_optional_float(expensive_signal.get("aesthetic_prior_laion_weight"))
        a_raw_laion = safe_optional_float(expensive_signal.get("aesthetic_raw_laion"))
        laion_norm_val = safe_optional_float(expensive_signal.get("aesthetic_norm_laion"))
        a_norm_laion = None if laion_norm_val is None else clamp(laion_norm_val, 0.0, 1.0)
        a_mean_nima = safe_optional_float(expensive_signal.get("aesthetic_mean_nima"))
        a_std_nima = safe_optional_float(expensive_signal.get("aesthetic_std_nima"))
        nima_norm_val = safe_optional_float(expensive_signal.get("aesthetic_norm_nima"))
        a_norm_nima = None if nima_norm_val is None else clamp(nima_norm_val, 0.0, 1.0)
        source = "real"
        expensive_active = True
    else:
        a_raw = None
        a_norm = 0.0
        ca_val = 0.0
        source = "missing" if bool(cfg.use_real_expensive) else "disabled"

    if expensive_active:
        exp_score = (
            cfg.w_a * a_norm
            + cfg.w_ca * ca_val
            + cfg.w_cov * cov
            - cfg.w_cut * p_cut
            - cfg.w_text * p_text
            - cfg.w_ar_free * p_ar_free
            + cfg.w_edge * r_edge
            + cfg.w_teach * r_teach
        )
        legacy_final_score = exp_score + float(w_area) * math.log(max(1e-8, area))
    else:
        exp_score = 0.0
        legacy_final_score = 0.0

    candidate["scores"]["expensive"] = float(exp_score)
    candidate["scores"]["final_legacy"] = float(legacy_final_score)
    comps["aesthetic_raw"] = None if a_raw is None else float(a_raw)
    comps["aesthetic_norm"] = float(a_norm)
    comps["cosine_img_text"] = float(ca_val)
    comps["expensive_source"] = source
    comps["expensive_active"] = bool(expensive_active)
    comps["aesthetic_backend"] = str(a_backend)
    comps["aesthetic_prior_laion_weight"] = (
        None if a_prior_laion_w is None else float(clamp(a_prior_laion_w, 0.0, 1.0))
    )
    comps["aesthetic_raw_laion"] = None if a_raw_laion is None else float(a_raw_laion)
    comps["aesthetic_norm_laion"] = None if a_norm_laion is None else float(a_norm_laion)
    comps["aesthetic_mean_nima"] = None if a_mean_nima is None else float(a_mean_nima)
    comps["aesthetic_std_nima"] = None if a_std_nima is None else float(max(0.0, a_std_nima))
    comps["aesthetic_norm_nima"] = None if a_norm_nima is None else float(a_norm_nima)
    macro_scores, macro_components, macro_masks = compute_macro_score_bundle(
        candidate=candidate,
        cfg=cfg,
    )
    score_rank = fuse_macro_scores(macro_scores=macro_scores, macro_masks=macro_masks, cfg=cfg)
    area_log_prior = float(w_area) * math.log(max(1e-8, area))
    score_policy = score_rank + area_log_prior
    candidate["scores"]["rank"] = float(score_rank)
    candidate["scores"]["policy"] = float(score_policy)
    candidate["scores"]["final"] = float(score_rank)
    candidate["scores"]["area_log_prior"] = float(area_log_prior)
    candidate["macro_scores"] = macro_scores
    candidate["macro_components"] = macro_components
    candidate["macro_masks"] = macro_masks
    candidate["policy"] = {
        "area_ratio": float(area),
        "area_log_prior": float(area_log_prior),
    }


def _score_from_centered_band(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    if value is None:
        return None
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        return 1.0 if float(value) >= lo_f else 0.0
    v = float(value)
    if lo_f <= v <= hi_f:
        return 1.0
    span = max(1e-6, hi_f - lo_f)
    dist = (lo_f - v) if v < lo_f else (v - hi_f)
    return clamp(1.0 - dist / span, 0.0, 1.0)


def _score_from_signed_term(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return clamp(0.5 + 0.5 * float(value), 0.0, 1.0)


def _score_from_penalty(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return clamp(1.0 - float(value), 0.0, 1.0)


def _weighted_subset_average(values: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for key, weight in weights.items():
        val = values.get(key)
        w = max(0.0, float(weight))
        if val is None or w <= 0.0:
            continue
        num += w * float(val)
        den += w
    if den <= 0.0:
        return None
    return num / den


def compute_macro_score_bundle(
    candidate: Dict[str, Any],
    cfg: TeacherScorerConfig,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, int]]:
    comps = candidate.get("scores", {}).get("components", {})
    checklist = candidate.get("checklist", {}) if isinstance(candidate.get("checklist"), dict) else {}
    flags = candidate.get("flags", {}) if isinstance(candidate.get("flags"), dict) else {}
    subject_mode = str(candidate.get("subject_mode", "")).strip().lower()
    expensive_source = str(comps.get("expensive_source", "")).strip().lower()
    expensive_active = expensive_source == "real"
    subject_terms_neutralized = bool(flags.get("subject_terms_neutralized", False))

    align_value = clamp(0.5 * (safe_float(comps.get("cosine_img_text", 0.0), 0.0) + 1.0), 0.0, 1.0) if expensive_active else 0.0
    a_components = {
        "A_aesthetic": clamp(safe_float(comps.get("aesthetic_norm", 0.0), 0.0), 0.0, 1.0) if expensive_active else 0.0,
        "A_align": align_value,
    }
    A_macro = None
    if expensive_active:
        A_macro = _weighted_subset_average(
            a_components,
            {
                "A_aesthetic": float(cfg.a_macro_aesthetic_weight),
                "A_align": float(cfg.a_macro_align_weight),
            },
        )

    subject_scale = checklist.get("subject_scale", {}) if isinstance(checklist.get("subject_scale"), dict) else {}
    scale_value = safe_optional_float(subject_scale.get("value"))
    scale_target_range = subject_scale.get("target_range", [])
    if isinstance(scale_target_range, (list, tuple)) and len(scale_target_range) >= 2:
        scale_score = _score_from_centered_band(scale_value, scale_target_range[0], scale_target_range[1])
    else:
        scale_score = None
    s_components = {
        "S_cov": clamp(safe_float(comps.get("cov", 0.0), 0.0), 0.0, 1.0),
        "S_scale": scale_score,
        "S_border": 0.0 if bool(flags.get("subject_touch_border", False)) else 1.0,
        "S_softcut_quality": _score_from_penalty(safe_optional_float(comps.get("p_cut"))),
    }
    if subject_terms_neutralized:
        s_components["S_cov"] = None
        s_components["S_scale"] = None
        s_components["S_border"] = None
    if subject_mode.startswith("scene_"):
        s_weights = {
            "S_cov": 0.45,
            "S_scale": 0.20,
            "S_border": 0.15,
            "S_softcut_quality": 0.20,
        }
    elif subject_mode.startswith("portrait"):
        s_weights = {
            "S_cov": 0.40,
            "S_scale": 0.25,
            "S_border": 0.20,
            "S_softcut_quality": 0.15,
        }
    else:
        s_weights = {
            "S_cov": 0.35,
            "S_scale": 0.25,
            "S_border": 0.20,
            "S_softcut_quality": 0.20,
        }
    S_macro = _weighted_subset_average(s_components, s_weights)

    headroom_meta = checklist.get("headroom", {}) if isinstance(checklist.get("headroom"), dict) else {}
    lookroom_meta = checklist.get("lookroom", {}) if isinstance(checklist.get("lookroom"), dict) else {}
    horizon_meta = checklist.get("horizon", {}) if isinstance(checklist.get("horizon"), dict) else {}
    context_meta = checklist.get("context", {}) if isinstance(checklist.get("context"), dict) else {}
    horizon_dist = safe_optional_float(horizon_meta.get("third_dist"))
    horizon_score = None
    if horizon_dist is not None:
        horizon_score = clamp(1.0 - float(horizon_dist) / max(1e-6, float(cfg.horizon_third_tau)), 0.0, 1.0)
    context_value = safe_optional_float(context_meta.get("value"))
    context_target_range = context_meta.get("target_range", [])
    if isinstance(context_target_range, (list, tuple)) and len(context_target_range) >= 2:
        context_score = _score_from_centered_band(context_value, context_target_range[0], context_target_range[1])
    else:
        context_score = None
    copyspace_meta = checklist.get("copyspace", {}) if isinstance(checklist.get("copyspace"), dict) else {}
    copyspace_score = clamp(safe_float(comps.get("r_copyspace", 0.0), 0.0), 0.0, 1.0)
    c_components = {
        "C_comp": clamp(safe_float(comps.get("r_comp", 0.0), 0.0), 0.0, 1.0),
        "C_headroom": _score_from_signed_term(safe_optional_float(comps.get("r_headroom"))),
        "C_lookroom": _score_from_signed_term(safe_optional_float(comps.get("r_lookroom"))),
        "C_horizon_y": horizon_score,
        "C_sym": clamp(safe_float(comps.get("r_sym", 0.0), 0.0), 0.0, 1.0),
        "C_context": context_score,
        "C_copyspace": copyspace_score if subject_mode == "background_copyspace" else None,
    }
    if subject_mode.startswith("portrait"):
        c_weights = {
            "C_comp": 0.20,
            "C_headroom": 0.30,
            "C_lookroom": 0.30,
            "C_sym": 0.10,
            "C_context": 0.10,
        }
    elif subject_mode in SCENE_MODES:
        c_weights = {
            "C_comp": 0.20,
            "C_horizon_y": 0.30,
            "C_sym": 0.10,
            "C_context": 0.25,
            "C_copyspace": 0.0,
        }
    elif subject_mode == "background_copyspace":
        c_weights = {
            "C_comp": 0.15,
            "C_context": 0.25,
            "C_copyspace": 0.45,
            "C_sym": 0.15,
        }
    else:
        c_weights = {
            "C_comp": 0.35,
            "C_sym": 0.20,
            "C_context": 0.25,
            "C_headroom": 0.10,
            "C_lookroom": 0.10,
        }
    if subject_mode != "background_copyspace" and str(copyspace_meta.get("label", "")).strip() == "copyspace_preserved":
        c_components["C_copyspace"] = None
    C_macro = _weighted_subset_average(c_components, c_weights)

    teacher_meta = checklist.get("teacher_consensus", {}) if isinstance(checklist.get("teacher_consensus"), dict) else {}
    T_macro = None
    if bool(teacher_meta.get("enabled", False)):
        T_macro = clamp(safe_float(teacher_meta.get("value", 0.0), 0.0), 0.0, 1.0)

    macro_scores = {
        "A_macro": (None if A_macro is None else float(A_macro)),
        "S_macro": (None if S_macro is None else float(S_macro)),
        "C_macro": (None if C_macro is None else float(C_macro)),
        "T_macro": (None if T_macro is None else float(T_macro)),
    }
    macro_components: Dict[str, Optional[float]] = {}
    macro_components.update(a_components)
    macro_components.update(s_components)
    macro_components.update(c_components)
    macro_components["T_teacher"] = T_macro
    macro_masks = {
        "A_active": int(A_macro is not None),
        "S_active": int(S_macro is not None),
        "C_active": int(C_macro is not None),
        "T_active": int(T_macro is not None),
    }
    return macro_scores, macro_components, macro_masks


def fuse_macro_scores(
    macro_scores: Dict[str, Optional[float]],
    macro_masks: Dict[str, int],
    cfg: TeacherScorerConfig,
) -> float:
    score_map = {
        "A_macro": max(0.0, float(cfg.rank_weight_a)),
        "S_macro": max(0.0, float(cfg.rank_weight_s)),
        "C_macro": max(0.0, float(cfg.rank_weight_c)),
        "T_macro": max(0.0, float(cfg.rank_weight_t)),
    }
    mask_map = {
        "A_macro": "A_active",
        "S_macro": "S_active",
        "C_macro": "C_active",
        "T_macro": "T_active",
    }
    num = 0.0
    den = 0.0
    for key, weight in score_map.items():
        value = macro_scores.get(key)
        active = int(macro_masks.get(mask_map[key], 1))
        if value is None or weight <= 0.0 or active <= 0:
            continue
        num += weight * float(value)
        den += weight
    if den <= 0.0:
        return 0.0
    return num / den


def _has_hard_tag(candidate: Dict[str, Any], tag: str) -> bool:
    return str(tag) in set(str(x) for x in (candidate.get("hard_reject_tags") or []))


def _is_structural_valid(candidate: Dict[str, Any]) -> bool:
    return (not _has_hard_tag(candidate, "area_violation")) and (not _has_hard_tag(candidate, "ar_violation"))


def _is_face_safe(candidate: Dict[str, Any]) -> bool:
    return not any(
        _has_hard_tag(candidate, tag)
        for tag in ("face_cut", "head_top_cut", "lookroom_cut")
    )


TRAINING_SEVERE_REJECT_TAGS = {
    "face_cut",
    "head_top_cut",
    "joint_cutoff",
    "text_cutoff",
    "lookroom_cut",
}


def _has_training_severe_reject(candidate: Dict[str, Any]) -> bool:
    if bool(candidate.get("hard_reject", False)):
        return True
    reject_tags = {str(tag) for tag in (candidate.get("reject_tags") or [])}
    return any(tag in TRAINING_SEVERE_REJECT_TAGS for tag in reject_tags)


def _relax_joint_only_hard_reject(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Relax strict hard reject only for candidates blocked solely by `joint_cutoff`.

    Structural violations (area/ar) and face cut remain hard.
    """
    if not bool(candidate.get("hard_reject", False)):
        return candidate

    hard_tags = set(str(x) for x in (candidate.get("hard_reject_tags") or []))
    if any(t in hard_tags for t in ("area_violation", "ar_violation", "face_cut", "head_top_cut", "lookroom_cut")):
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
    target_ar: Optional[float],
    image_ar: float,
    subject_box: Sequence[float],
    subject_centroid: Tuple[float, float],
    c3_info: Dict[str, Any],
    c5_info: Dict[str, Any],
    c4_info: Dict[str, Any],
    route: Dict[str, Any],
    teacher_ctx: Optional[Dict[str, Any]],
    cfg: TeacherScorerConfig,
    subject_support: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    crop = clip_box01(candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))
    area = box_area(crop)
    c_w, c_h = box_wh(crop)
    ar = (c_w * image_ar / max(1e-8, c_h)) if c_h > 0 else 0.0

    hard_reject_tags: List[str] = []
    is_baseline_full = bool(candidate.get("must_keep", False)) and str(candidate.get("source", "")).startswith("baseline_full")

    # Structural hard checks.
    allow_full_baseline = is_baseline_full and area <= 1.000001
    if (not (cfg.area_min <= area <= cfg.area_max)) and (not allow_full_baseline):
        hard_reject_tags.append("area_violation")
    is_freeform = target_ar is None
    if (target_ar is not None) and (abs(ar - target_ar) > cfg.ar_hard_eps):
        hard_reject_tags.append("ar_violation")

    is_portrait_mode = str(route.get("subject_mode", "")).startswith("portrait")
    f_cut, f_kept, f_total = face_cut_flags(c3_info["face_boxes"], crop)
    if cfg.hard_face_rule and f_total > 0 and f_cut and ((not is_baseline_full) or is_portrait_mode):
        hard_reject_tags.append("face_cut")

    head_top = compute_head_top_guard(
        crop=crop,
        head_y_norm_values=c3_info.get("head_guard_y_norm", []),
        head_margin_values=c3_info.get("head_guard_margin_norm", []),
        min_margin=float(cfg.head_top_min_margin),
    )
    if (
        bool(cfg.hard_head_top_rule)
        and is_portrait_mode
        and ((not is_baseline_full) or is_portrait_mode)
        and bool(head_top.get("activated", False))
        and (not bool(head_top.get("pass", True)))
    ):
        hard_reject_tags.append("head_top_cut")

    joint = eval_joint_cut(
        keypoints_norm=c3_info["keypoints_norm"],
        crop=crop,
        margin_alpha=cfg.severe_kp_margin_alpha,
    )
    if joint["severe_count"] >= cfg.hard_joint_reject_count and ((not is_baseline_full) or is_portrait_mode):
        hard_reject_tags.append("joint_cutoff")

    # Cheap terms
    subject_terms_neutralized = bool(route.get("flags", {}).get("subject_terms_neutralized", False))
    subject_support = subject_support if isinstance(subject_support, dict) else {"mode": "bbox", "bbox_norm_xyxy": clip_box01(subject_box)}
    subject_cov_mode = str(subject_support.get("mode", "bbox"))
    if subject_cov_mode == "support_map":
        cov, subj_area_ratio = subject_coverage_soft(subject_support, crop)
        subj_touch_raw = subject_border_touch_soft(subject_support, crop)
    else:
        cov, subj_area_ratio = subject_coverage(subject_box, crop)
        subj_touch_raw = subject_border_touch(subject_box, crop)
    subj_touch = False if subject_terms_neutralized else subj_touch_raw

    p_cut = (
        cfg.alpha_face_cut * (1.0 if f_cut else 0.0)
        + cfg.alpha_joint_cut * joint["joint_cut_score"]
        + cfg.alpha_subject_border * (1.0 if subj_touch else 0.0)
    )

    text_eval = compute_text_penalty(
        crop=crop,
        c4_info=c4_info,
        route=route,
        cfg=cfg,
    )
    p_text = safe_float(text_eval.get("p_text", 0.0), 0.0)

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
        horizon_exists_prob=safe_float(c5_info.get("horizon_exists_prob", c5_info["horizon_conf"]), 0.0),
        scene_subtype=str(route.get("scene_subtype", "")),
        line_norm_xyxy=c5_info.get("line_norm_xyxy"),
        conf_thr=cfg.horizon_conf_thr,
        visibility_thr=cfg.horizon_visibility_thr,
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
    subject_box_area_full = box_area(subject_box)
    route_subject_set = route.get("subject_set", {}) if isinstance(route.get("subject_set"), dict) else {}
    route_primary_area_ratio = safe_float(route_subject_set.get("c2_primary_area_ratio", subject_box_area_full))
    route_conf = safe_float(route.get("subject_mode_conf", 0.0))
    relax_lookroom_hard_reject = bool(
        subject_box_area_full < cfg.lookroom_hard_reject_min_subject_area_ratio
        or route_primary_area_ratio < cfg.lookroom_hard_reject_min_subject_area_ratio
        or route_conf < cfg.lookroom_hard_reject_min_route_conf
        or str(route.get("shot_type", "unknown")).strip().lower() == "unknown"
        or bool(route.get("flags", {}).get("subject_mode_contextual_tiny_human", False))
    )
    lookroom_cut = (
        bool(cfg.hard_portrait_lookroom_rule)
        and is_portrait_mode
        and lr["gaze_dir"] in {"left", "right"}
        and (not bool(lr["pass"]))
        and (not relax_lookroom_hard_reject)
    )
    if lookroom_cut:
        hard_reject_tags.append("lookroom_cut")

    sym = c5_info["symmetry"]

    scale_lo, scale_hi = route.get("subject_scale_range", route.get("legacy_context_range", route["context_range"]))
    ctx_lo, ctx_hi = route["context_range"]
    text_keep_ratio = (
        clamp(safe_float(text_eval.get("text_keep_ratio", 1.0), 1.0), 0.0, 1.0)
        if bool(text_eval.get("available", False))
        else 0.0
    )
    copyspace_ratio = clamp(1.0 - subj_area_ratio, 0.0, 1.0)
    context_value = compute_context_preserved(
        crop_area=area,
        subject_area_in_crop_ratio=subj_area_ratio,
        subject_box_area_full=subject_box_area_full,
        scene_bonus=0.5 * clamp(safe_float(horizon.get("visible_ratio", 0.0), 0.0), 0.0, 1.0) + 0.5 * sym,
        text_bonus=text_keep_ratio,
        negative_space_bonus=copyspace_ratio,
    )
    ctx_target = 0.5 * (ctx_lo + ctx_hi)
    sigma_a = max(0.05, 0.5 * (ctx_hi - ctx_lo))
    r_context = -abs(context_value - ctx_target) / sigma_a
    ctx_pass = ctx_lo <= context_value <= ctx_hi

    copyspace_eval = compute_copyspace_reward(crop=crop, subject_box=subject_box, route=route)
    r_copyspace = copyspace_eval["reward"] if route["flags"].get("has_copy_space", False) else 0.0

    p_ar_free = free_ar_prior_penalty(ar=ar, route=route, cfg=cfg) if is_freeform else 0.0
    teach_eval = compute_teacher_consensus_reward(crop=crop, teacher_ctx=teacher_ctx, cfg=cfg)
    r_teach = safe_float(teach_eval.get("r_teach", 0.0), 0.0)
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
        + lam.get("teach", 0.0) * r_teach
        - lam.get("ar_free", 0.0) * p_ar_free
    )

    # Expensive priors (used as fallback when real expensive model is unavailable).
    context_fit = reward_dist(abs(context_value - ctx_target), max(0.05, sigma_a))
    aesthetic_proxy = (
        0.45 * r_comp
        + 0.20 * sym
        + 0.20 * cov
        + 0.15 * context_fit
        - 0.30 * min(1.0, p_cut)
    )
    ca_proxy = 0.65 * cov + 0.35 * context_fit
    r_edge = edge_term(subject_box=subject_box, crop=crop)

    # Explainability labels (continuous value + categorical label).
    ar_error = (None if is_freeform else float(abs(ar - float(target_ar))))
    if subject_terms_neutralized:
        subject_cov_label = "subject_coverage_na"
        subject_scale_label = "subject_scale_na"
    else:
        subject_cov_label = classify_subject_coverage_label(cov)
        subject_scale_label = classify_subject_scale_label(subj_area_ratio, (scale_lo, scale_hi))
    face_cut_label = "face_cut" if f_cut else "no_face_cut"
    joint_cut_label = classify_joint_cut_label(joint["joint_cut_score"], joint["severe_count"])
    headroom_label = classify_headroom_label(hr.get("value"), hr.get("target_range", []))
    lookroom_label = classify_lookroom_label(lr.get("value"), lr.get("target_range", []), lr.get("gaze_dir", "unknown"))
    third_label = classify_rule_strength_label(
        d_third,
        strong_thr=0.10,
        strong_label="rule_of_thirds_strong",
        weak_label="rule_of_thirds_weak",
        na_label="rule_of_thirds_na",
    )
    phi_label = classify_rule_strength_label(
        d_phi,
        strong_thr=0.10,
        strong_label="phi_grid_strong",
        weak_label="phi_grid_weak",
        na_label="phi_grid_na",
    )
    center_label = classify_rule_strength_label(
        d_center,
        strong_thr=0.12,
        strong_label="center_comp_strong",
        weak_label="center_comp_weak",
        na_label="center_comp_na",
    )
    horizon_label = classify_horizon_label(horizon, cfg.horizon_third_tau)
    context_label = classify_context_label(context_value, (ctx_lo, ctx_hi))
    roll_label = classify_roll_label(c5_info.get("roll_deg"), cfg.roll_violation_deg)
    text_label = classify_text_keep_label(text_eval)
    teacher_label = "teacher_consensus_na"
    if bool(teach_eval.get("enabled", False)):
        if bool(teach_eval.get("consensus", False)):
            teacher_label = (
                "teacher_aligned"
                if safe_float(teach_eval.get("rho", 0.0), 0.0) >= float(cfg.teach_rho_tau)
                else "teacher_mismatch"
            )
        else:
            teacher_label = "teacher_no_consensus"
    ar_label = "ar_choice_freeform" if is_freeform else (
        "ar_fits_well" if (ar_error is not None and ar_error <= float(cfg.ar_hard_eps)) else "ar_mismatch"
    )
    crop_tightness_label = classify_crop_tightness_label(area)

    checklist = {
        "subject_coverage": {
            "value": (None if subject_terms_neutralized else float(cov)),
            "label": subject_cov_label,
            "thresholds": {"excellent": 0.95, "good": 0.85, "marginal": 0.70},
        },
        "subject_scale": {
            "value": (None if subject_terms_neutralized else float(subj_area_ratio)),
            "label": subject_scale_label,
            "target_range": [round(float(scale_lo), 4), round(float(scale_hi), 4)],
            "target_region": str(route.get("subject_scale_target_region", "primary_subject")),
        },
        "face_cut": {"value": int(bool(f_cut)), "label": face_cut_label},
        "joint_cut": {
            "value": float(joint["joint_cut_score"]),
            "label": joint_cut_label,
            "severe_count": int(joint["severe_count"]),
            "near_count": int(joint["near_count"]),
        },
        "text_keep_ratio": {
            "value": (
                None
                if not bool(text_eval.get("available", False))
                else float(text_eval.get("text_keep_ratio", 1.0))
            ),
            "label": text_label,
            "available": bool(text_eval.get("available", False)),
            "target_keep_ratio": float(text_eval.get("target_keep_ratio", cfg.text_keep_min_default)),
        },
        "third_dist": {"value": float(d_third), "label": third_label},
        "phi_dist": {"value": float(d_phi), "label": phi_label},
        "center_dist": {"value": float(d_center), "label": center_label},
        "headroom": {
            "value": hr.get("value"),
            "label": headroom_label,
            "target_range": hr.get("target_range"),
        },
        "lookroom": {
            "value": lr.get("value"),
            "label": lookroom_label,
            "target_range": lr.get("target_range"),
            "gaze_dir": lr.get("gaze_dir", "unknown"),
        },
        "horizon": {
            "value": horizon.get("value"),
            "label": horizon_label,
            "third_dist": horizon.get("dist"),
            "conf": float(c5_info["horizon_conf"]),
            "state": str(horizon.get("state", "none")),
            "exists_prob": float(safe_float(horizon.get("exists_prob", 0.0), 0.0)),
            "visible_ratio": float(safe_float(horizon.get("visible_ratio", 0.0), 0.0)),
            "target_set": horizon.get("target_set", []),
            "roll_used_in_score": False,
        },
        "context": {
            "value": float(context_value),
            "label": context_label,
            "target_range": [round(float(ctx_lo), 4), round(float(ctx_hi), 4)],
            "target_region": str(route.get("context_target_region", "secondary_context")),
        },
        "copyspace": {
            "value": float(r_copyspace),
            "label": (
                "copyspace_preserved"
                if copyspace_eval["blank_ratio_keep"] >= 0.18 and copyspace_eval["side_consistency"] >= 0.5
                else "copyspace_partial"
            ),
            "side": str(route.get("copyspace_side", "unknown")),
            "quality": str(route.get("copyspace_quality", "none")),
        },
        "roll": {
            "value": (None if c5_info.get("roll_deg") is None else float(safe_float(c5_info.get("roll_deg"), 0.0))),
            "label": roll_label,
            "needs_leveling": bool(roll_label == "needs_leveling"),
            "used_in_score": False,
        },
        "teacher_consensus": {
            "value": float(safe_float(teach_eval.get("rho", 0.0), 0.0)),
            "label": teacher_label,
            "consensus": bool(teach_eval.get("consensus", False)),
            "enabled": bool(teach_eval.get("enabled", False)),
        },
        "ar": {
            "value": (None if is_freeform else float(ar_error)),
            "label": ar_label,
            "target_ar": (None if is_freeform else float(target_ar)),
        },
        "crop_tightness": {"value": float(area), "label": crop_tightness_label},
    }
    checklist_labels = {
        k: str(v.get("label", ""))
        for k, v in checklist.items()
        if isinstance(v, dict)
    }

    # Deterministic tags/checks for rationale.
    why_tags: List[str] = []
    if subject_terms_neutralized:
        why_tags.append("subject_terms_neutralized")
    elif subject_cov_label in {"excellent", "good"}:
        why_tags.append("subject_preserved")
    elif subject_cov_label == "marginal":
        why_tags.append("subject_marginal")
    else:
        why_tags.append("subject_poor")
    if not subject_terms_neutralized:
        if subject_scale_label == "ideal_scale":
            why_tags.append("subject_scale_ideal")
        elif subject_scale_label == "too_tight":
            why_tags.append("subject_scale_tight")
        else:
            why_tags.append("subject_scale_loose")
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
    if bool(head_top.get("activated", False)):
        why_tags.append("head_top_safe" if bool(head_top.get("pass", True)) else "head_top_cut_risk")
    if lr["gaze_dir"] in {"left", "right"}:
        why_tags.append("lookroom_ok" if lr["pass"] else "lookroom_violation")
    if horizon["active"]:
        why_tags.append("horizon_on_target" if horizon["pass"] else "horizon_off_target")

    why_tags.append("context_preserved" if ctx_pass else "context_loss")
    if not f_cut:
        why_tags.append("avoid_face_cut")
    if joint["joint_cut_score"] <= 0.10:
        why_tags.append("avoid_person_cut")
    if route["flags"].get("has_copy_space", False):
        why_tags.append("copy_space_kept" if copyspace_eval["blank_ratio_keep"] >= 0.18 else "copy_space_lost")
    if roll_label == "needs_leveling":
        why_tags.append("needs_leveling")
    if bool(text_eval.get("available", False)):
        why_tags.append("text_preserved" if bool(text_eval.get("pass", True)) else "text_cut_risk")
    if bool(teach_eval.get("consensus", False)):
        why_tags.append("teacher_consensus")
        if safe_float(teach_eval.get("rho", 0.0), 0.0) >= float(cfg.teach_rho_tau):
            why_tags.append("teacher_aligned")
        else:
            why_tags.append("teacher_mismatch")
    if is_freeform:
        why_tags.append("ar_choice_freeform")
        if p_ar_free > 1e-6:
            why_tags.append("ar_extreme_penalty")
    else:
        why_tags.append("ar_fits_well" if abs(ar - float(target_ar)) <= cfg.ar_hard_eps else "ar_mismatch")
    why_tags.append(crop_tightness_label)

    why_tags_numeric = [
        {
            "why_tag": "subject_preserved",
            "metric": "subject_coverage",
            "value": (None if subject_terms_neutralized else float(cov)),
            "label": subject_cov_label,
            "pass": (None if subject_terms_neutralized else bool(cov >= 0.85)),
        },
        {
            "why_tag": "subject_scale_ideal",
            "metric": "subject_scale",
            "value": (None if subject_terms_neutralized else float(subj_area_ratio)),
            "label": subject_scale_label,
            "pass": (None if subject_terms_neutralized else bool(subject_scale_label == "ideal_scale")),
        },
        {
            "why_tag": "avoid_face_cut",
            "metric": "face_cut",
            "value": int(bool(f_cut)),
            "label": face_cut_label,
            "pass": bool(not f_cut),
        },
        {
            "why_tag": "avoid_person_cut",
            "metric": "joint_cut",
            "value": float(joint["joint_cut_score"]),
            "label": joint_cut_label,
            "pass": bool(joint["joint_cut_score"] <= 0.10),
        },
        {
            "why_tag": "rule_of_thirds",
            "metric": "third_dist",
            "value": float(d_third),
            "label": third_label,
            "pass": bool(d_third <= 0.18),
        },
        {
            "why_tag": "headroom_ok",
            "metric": "headroom",
            "value": hr.get("value"),
            "label": headroom_label,
            "pass": (None if hr.get("value") is None else bool(hr["pass"])),
        },
        {
            "why_tag": "lookroom_ok",
            "metric": "lookroom",
            "value": lr.get("value"),
            "label": lookroom_label,
            "pass": (
                None
                if lr.get("gaze_dir") not in {"left", "right"} or lr.get("value") is None
                else bool(lr["pass"])
            ),
        },
        {
            "why_tag": "horizon_on_target",
            "metric": "horizon_dist",
            "value": horizon.get("dist"),
            "label": horizon_label,
            "pass": (None if not bool(horizon.get("active", False)) else bool(horizon["pass"])),
        },
        {
            "why_tag": "context_preserved",
            "metric": "context_preserved",
            "value": float(context_value),
            "label": context_label,
            "pass": bool(ctx_pass),
        },
        {
            "why_tag": "text_preserved",
            "metric": "text_keep_ratio",
            "value": (
                None
                if not bool(text_eval.get("available", False))
                else float(text_eval.get("text_keep_ratio", 1.0))
            ),
            "label": text_label,
            "pass": (
                None
                if not bool(text_eval.get("available", False))
                else bool(text_eval.get("pass", True))
            ),
        },
        {
            "why_tag": "teacher_aligned",
            "metric": "teacher_rho",
            "value": float(safe_float(teach_eval.get("rho", 0.0), 0.0)),
            "label": teacher_label,
            "pass": bool(teacher_label == "teacher_aligned"),
        },
        {
            "why_tag": "ar_fits_well",
            "metric": "ar_error",
            "value": (None if ar_error is None else float(ar_error)),
            "label": ar_label,
            "pass": (None if is_freeform else bool(ar_error is not None and ar_error <= float(cfg.ar_hard_eps))),
        },
    ]
    why_tags_final = sorted(set(why_tags))
    why_text_template = build_why_text_template(
        checklist_labels=checklist_labels,
        why_tags=why_tags_final,
        is_freeform=is_freeform,
    )

    reject_tags = list(hard_reject_tags)
    if f_cut and "face_cut" not in reject_tags:
        reject_tags.append("face_cut")
    if bool(head_top.get("activated", False)) and (not bool(head_top.get("pass", True))) and "head_top_cut" not in reject_tags:
        reject_tags.append("head_top_cut")
    if joint["joint_cut_score"] > 0.35 and "joint_cutoff" not in reject_tags:
        reject_tags.append("joint_cutoff")
    if bool(text_eval.get("available", False)) and (not bool(text_eval.get("pass", True))) and "text_cutoff" not in reject_tags:
        reject_tags.append("text_cutoff")
    if is_portrait_mode and lr["gaze_dir"] in {"left", "right"} and not lr["pass"]:
        reject_tags.append("lookroom_violation")
    if lookroom_cut and "lookroom_cut" not in reject_tags:
        reject_tags.append("lookroom_cut")

    out = {
        "candidate_id": candidate.get("candidate_id"),
        "bbox_norm_xyxy": crop,
        "ar": round(ar, 6),
        "area_ratio": round(area, 6),
        "source": candidate.get("source"),
        "subject_mode": str(route.get("subject_mode", "other_ambiguous")),
        "policy_id": str(route.get("policy_id", "generic_v1")),
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
            "rank": 0.0,
            "policy": 0.0,
            "final_legacy": 0.0,
            "area_log_prior": 0.0,
            "components": {
                "cov": float(cov),
                "p_cut": float(p_cut),
                "p_text": float(p_text),
                "subject_terms_neutralized": bool(subject_terms_neutralized),
                "subject_cov_mode": subject_cov_mode,
                "subject_cov_raw": float(cov),
                "subject_area_raw": float(subj_area_ratio),
                "r_comp": float(r_comp),
                "r_horizon": float(horizon["reward"]),
                "r_headroom": float(hr["score"]),
                "r_lookroom": float(lr["score"]),
                "r_sym": float(sym),
                "r_context": float(r_context),
                "context_value": float(context_value),
                "r_copyspace": float(r_copyspace),
                "copyspace_blank_ratio_keep": float(copyspace_eval["blank_ratio_keep"]),
                "copyspace_side_consistency": float(copyspace_eval["side_consistency"]),
                "r_teach": float(r_teach),
                "teacher_rho": float(safe_float(teach_eval.get("rho", 0.0), 0.0)),
                "p_ar_free": float(p_ar_free),
                "r_edge": float(r_edge),
                "aesthetic_proxy": float(aesthetic_proxy),
                "ca_proxy": float(ca_proxy),
                "aesthetic_raw": None,
                "aesthetic_norm": None,
                "aesthetic_backend": "pending",
                "aesthetic_prior_laion_weight": None,
                "aesthetic_raw_laion": None,
                "aesthetic_norm_laion": None,
                "aesthetic_mean_nima": None,
                "aesthetic_std_nima": None,
                "aesthetic_norm_nima": None,
                "cosine_img_text": None,
                "expensive_source": "pending",
            },
        },
        "flags": {
            "face_cut": bool(f_cut),
            "head_top_cut": bool(head_top.get("activated", False) and (not bool(head_top.get("pass", True)))),
            "lookroom_cut": bool(lookroom_cut),
            "joint_cutoff_score": float(joint["joint_cut_score"]),
            "joint_severe_count": int(joint["severe_count"]),
            "subject_touch_border": bool(subj_touch),
            "subject_touch_border_raw": bool(subj_touch_raw),
            "subject_terms_neutralized": bool(subject_terms_neutralized),
            "subject_effective_state": str(route.get("flags", {}).get("subject_effective_state", "")),
            "subject_coverage": float(cov),
            "subject_area": float(subj_area_ratio),
            "ar_error": ar_error,
            "ar_free_prior_penalty": float(p_ar_free),
        },
        "composition_checks": {
            "headroom": {
                "value": hr["value"],
                "target_range": hr["target_range"],
                "pass": bool(hr["pass"]),
            },
            "head_top": {
                "activated": bool(head_top.get("activated", False)),
                "pass": bool(head_top.get("pass", True)),
                "min_required_y1": head_top.get("min_required_y1"),
                "crop_y1": head_top.get("crop_y1"),
                "violated_count": int(head_top.get("violated_count", 0)),
                "head_count": int(head_top.get("head_count", 0)),
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
                "state": str(horizon.get("state", "none")),
                "exists_prob": float(safe_float(horizon.get("exists_prob", 0.0), 0.0)),
                "visible_ratio": float(safe_float(horizon.get("visible_ratio", 0.0), 0.0)),
                "target_set": horizon.get("target_set", []),
                "roll_used_in_score": False,
            },
            "roll": {
                "value": (None if c5_info.get("roll_deg") is None else float(safe_float(c5_info.get("roll_deg"), 0.0))),
                "label": roll_label,
                "needs_leveling": bool(roll_label == "needs_leveling"),
                "rotation_hint_deg": (
                    None if c5_info.get("roll_deg") is None else float(-safe_float(c5_info.get("roll_deg"), 0.0))
                ),
                "used_in_score": False,
            },
            "symmetry": {
                "value": float(sym),
                "pass": bool(sym >= 0.50),
            },
            "context": {
                "subject_area": float(subj_area_ratio),
                "context_preserved": float(context_value),
                "target_range": [round(ctx_lo, 4), round(ctx_hi, 4)],
                "pass": bool(ctx_pass),
                "target_region": str(route.get("context_target_region", "secondary_context")),
            },
            "copyspace": {
                "side": str(route.get("copyspace_side", "unknown")),
                "quality": str(route.get("copyspace_quality", "none")),
                "blank_ratio_keep": float(copyspace_eval["blank_ratio_keep"]),
                "side_consistency": float(copyspace_eval["side_consistency"]),
                "intrusion": float(copyspace_eval["intrusion"]),
                "pass": bool(copyspace_eval["blank_ratio_keep"] >= 0.18),
            },
            "cutoff": {
                "joint_cutoff_score": float(joint["joint_cut_score"]),
                "joint_severe_count": int(joint["severe_count"]),
                "joint_violated": joint["violated"],
                "face_cut": bool(f_cut),
                "pass": bool((not f_cut) and (joint["joint_cut_score"] <= 0.35)),
            },
            "text": {
                "available": bool(text_eval.get("available", False)),
                "method": str(text_eval.get("method", "unknown")),
                "num_boxes": int(safe_float(text_eval.get("num_boxes", 0), 0.0)),
                "coverage_ratio": float(text_eval.get("coverage_ratio", 0.0)),
                "text_overlay_likely": bool(text_eval.get("text_overlay_likely", False)),
                "text_keep_ratio": float(text_eval.get("text_keep_ratio", 1.0)),
                "cut_box_ratio": float(text_eval.get("cut_box_ratio", 0.0)),
                "severe_cut_ratio": float(text_eval.get("severe_cut_ratio", 0.0)),
                "target_keep_ratio": float(text_eval.get("target_keep_ratio", cfg.text_keep_min_default)),
                "severe_ratio_max": float(text_eval.get("severe_ratio_max", cfg.text_severe_ratio_max_default)),
                "pass": bool(text_eval.get("pass", True)),
            },
            "teacher_consensus": {
                "enabled": bool(teach_eval.get("enabled", False)),
                "available": bool(teach_eval.get("available", False)),
                "consensus": bool(teach_eval.get("consensus", False)),
                "rho": float(safe_float(teach_eval.get("rho", 0.0), 0.0)),
                "pass": bool(safe_float(teach_eval.get("rho", 0.0), 0.0) >= float(cfg.teach_rho_tau)),
            },
        },
        "checklist": checklist,
        "checklist_labels": checklist_labels,
        "why_tags": why_tags_final,
        "why_tags_numeric": why_tags_numeric,
        "why_text_template": why_text_template,
        "reject_tags": sorted(set(reject_tags)),
    }
    if "teacher_id" in candidate:
        out["teacher_id"] = candidate.get("teacher_id")
    if "teacher_stage" in candidate:
        out["teacher_stage"] = candidate.get("teacher_stage")
    if "teacher_target_ar" in candidate:
        out["teacher_target_ar"] = candidate.get("teacher_target_ar")
    if "teacher_raw_score" in candidate:
        out["teacher_raw_score"] = safe_float(candidate.get("teacher_raw_score", 0.0), 0.0)
    if "teacher_ref_kind" in candidate:
        out["teacher_ref_kind"] = candidate.get("teacher_ref_kind")
    if "source_lineage" in candidate:
        out["source_lineage"] = candidate.get("source_lineage", [])
    if "teacher_derived" in candidate:
        out["teacher_derived"] = bool(candidate.get("teacher_derived", False))
    if "teacher_ids" in candidate:
        out["teacher_ids"] = candidate.get("teacher_ids", [])
    if "teacher_stages" in candidate:
        out["teacher_stages"] = candidate.get("teacher_stages", [])
    if "teacher_provenance" in candidate:
        out["teacher_provenance"] = candidate.get("teacher_provenance", [])
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

    def key_priority(c: Dict[str, Any]) -> Tuple[float, float, float, float]:
        policy_score = safe_float(c.get("scores", {}).get("policy", -1e9), -1e9)
        rank_score = safe_float(c.get("scores", {}).get("rank", c.get("scores", {}).get("final", -1e9)), -1e9)
        has_policy = 1.0 if policy_score > -1e8 else 0.0
        return (
            has_policy,
            policy_score,
            rank_score,
            safe_float(c.get("priority", 0.0)),
        )

    baseline_full = [c for c in cands if str(c.get("source", "")).startswith("baseline_full")]
    baseline_center = [c for c in cands if str(c.get("source", "")) == "baseline_maxarea_center"]
    baseline_subject = [c for c in cands if str(c.get("source", "")) == "baseline_maxarea_subject"]
    baseline_slide = [c for c in cands if str(c.get("source", "")).startswith("baseline_maxarea_slide")]
    must_keep = [c for c in cands if bool(c.get("must_keep", False))]

    baseline = None
    if baseline_full:
        baseline = max(baseline_full, key=key_priority)
    else:
        maxarea_pool = baseline_center + baseline_subject
        if maxarea_pool:
            baseline = max(maxarea_pool, key=key_priority)
        elif baseline_slide:
            baseline = max(baseline_slide, key=key_priority)
        elif must_keep:
            baseline = max(must_keep, key=key_priority)
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
    vals = [safe_float(c["scores"].get("rank", c["scores"].get("final", 0.0))) for c in cands]
    if not vals:
        return
    mn = min(vals)
    mx = max(vals)
    if mx - mn < 1e-9:
        for c in cands:
            c["scores"]["final_norm_0_100"] = 50.0
            c["scores"]["rank_norm_0_100"] = 50.0
            c["scores"]["policy_norm_0_100"] = 50.0
        return
    policy_vals = [safe_float(c["scores"].get("policy", 0.0)) for c in cands]
    policy_mn = min(policy_vals)
    policy_mx = max(policy_vals)
    for c in cands:
        v = safe_float(c["scores"].get("rank", c["scores"].get("final", 0.0)))
        rank_norm = 100.0 * (v - mn) / (mx - mn)
        c["scores"]["final_norm_0_100"] = rank_norm
        c["scores"]["rank_norm_0_100"] = rank_norm
        if policy_mx - policy_mn < 1e-9:
            c["scores"]["policy_norm_0_100"] = 50.0
        else:
            p = safe_float(c["scores"].get("policy", 0.0))
            c["scores"]["policy_norm_0_100"] = 100.0 * (p - policy_mn) / (policy_mx - policy_mn)


def decide_keep_vs_crop(best: Dict[str, Any], baseline: Dict[str, Any], tau_improve: float) -> Dict[str, Any]:
    b_best = safe_float(best["scores"].get("policy", best["scores"].get("final", -1e9)))
    b_base = safe_float(baseline["scores"].get("policy", baseline["scores"].get("final", -1e9)))
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
        "chosen_score_rank": float(safe_float(chosen["scores"].get("rank", chosen["scores"].get("final", 0.0)))),
        "chosen_score_policy": float(safe_float(chosen["scores"].get("policy", chosen["scores"].get("final", 0.0)))),
    }


def select_topk_diverse(
    sorted_cands: Sequence[Dict[str, Any]],
    k: int,
    tau_div: float,
    force_ids: Optional[Sequence[str]] = None,
    ar_log_gap_thr: Optional[float] = None,
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
            suppress = False
            for s in selected:
                iou = iou_xyxy(c["bbox_norm_xyxy"], s["bbox_norm_xyxy"])
                if iou < tau_div:
                    continue
                if ar_log_gap_thr is None:
                    suppress = True
                    break

                ar_c = safe_float(c.get("ar", 0.0), 0.0)
                ar_s = safe_float(s.get("ar", 0.0), 0.0)
                if ar_c <= 1e-8 or ar_s <= 1e-8:
                    suppress = True
                    break
                ar_gap = abs(math.log(max(ar_c, 1e-8) / max(ar_s, 1e-8)))
                if ar_gap < float(ar_log_gap_thr):
                    suppress = True
                    break
            if suppress:
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
    mids.sort(key=lambda x: safe_float(x["scores"].get("rank", x["scores"].get("final", 0.0))), reverse=True)
    return mids[:max_n]


def candidate_brief(c: Dict[str, Any], rank: Optional[int] = None) -> Dict[str, Any]:
    out = {
        "candidate_id": c.get("candidate_id"),
        "bbox_norm_xyxy": [round(float(v), 6) for v in c.get("bbox_norm_xyxy", [0, 0, 1, 1])],
        "ar": round(safe_float(c.get("ar", 0.0), 0.0), 6),
        "area_ratio": round(safe_float(c.get("area_ratio", 0.0)), 6),
        "source": c.get("source"),
        "must_keep": bool(c.get("must_keep", False)),
        "hard_reject": bool(c.get("hard_reject", False)),
        "hard_reject_strict": bool(c.get("hard_reject_strict", c.get("hard_reject", False))),
        "scores": {
            "cheap": round(safe_float(c["scores"].get("cheap", 0.0)), 6),
            "expensive": round(safe_float(c["scores"].get("expensive", 0.0)), 6),
            "rank": round(safe_float(c["scores"].get("rank", c["scores"].get("final", 0.0)), 0.0), 6),
            "policy": round(safe_float(c["scores"].get("policy", c["scores"].get("final", 0.0)), 0.0), 6),
            "final": round(safe_float(c["scores"].get("final", 0.0)), 6),
            "final_legacy": round(safe_float(c["scores"].get("final_legacy", 0.0)), 6),
            "final_norm_0_100": round(safe_float(c["scores"].get("final_norm_0_100", 0.0)), 3),
            "rank_norm_0_100": round(safe_float(c["scores"].get("rank_norm_0_100", 0.0)), 3),
            "policy_norm_0_100": round(safe_float(c["scores"].get("policy_norm_0_100", 0.0)), 3),
            "area_log_prior": round(safe_float(c["scores"].get("area_log_prior", 0.0), 0.0), 6),
            "components": c["scores"].get("components", {}),
        },
        "macro_scores": c.get("macro_scores", {}),
        "macro_components": c.get("macro_components", {}),
        "macro_masks": c.get("macro_masks", {}),
        "policy": c.get("policy", {}),
        "flags": c.get("flags", {}),
        "checklist": c.get("checklist", {}),
        "checklist_labels": c.get("checklist_labels", {}),
        "why_tags": c.get("why_tags", []),
        "why_tags_numeric": c.get("why_tags_numeric", []),
        "why_text_template": c.get("why_text_template", ""),
        "reject_tags": c.get("reject_tags", []),
        "composition_checks": c.get("composition_checks", {}),
    }
    if "teacher_id" in c:
        out["teacher_id"] = c.get("teacher_id")
    if "teacher_stage" in c:
        out["teacher_stage"] = c.get("teacher_stage")
    if "teacher_target_ar" in c:
        out["teacher_target_ar"] = c.get("teacher_target_ar")
    if "teacher_raw_score" in c:
        out["teacher_raw_score"] = round(safe_float(c.get("teacher_raw_score", 0.0)), 6)
    if "teacher_ref_kind" in c:
        out["teacher_ref_kind"] = c.get("teacher_ref_kind")
    if "source_lineage" in c:
        out["source_lineage"] = c.get("source_lineage", [])
    if "teacher_derived" in c:
        out["teacher_derived"] = bool(c.get("teacher_derived", False))
    if "teacher_ids" in c:
        out["teacher_ids"] = c.get("teacher_ids", [])
    if "teacher_stages" in c:
        out["teacher_stages"] = c.get("teacher_stages", [])
    if "teacher_provenance" in c:
        out["teacher_provenance"] = c.get("teacher_provenance", [])
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
    if bool(flags.get("head_top_cut", False)):
        stats.top1_head_top_cut_count += 1
    if safe_float(flags.get("joint_cutoff_score", 0.0)) > 0.35:
        stats.top1_joint_cut_count += 1
    if (not bool(flags.get("subject_terms_neutralized", False))) and safe_float(flags.get("subject_coverage", 0.0)) < cfg.subject_coverage_fail_thr:
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
        head_top = checks.get("head_top", {}) if isinstance(checks.get("head_top"), dict) else {}
        if bool(head_top.get("activated", False)):
            stats.head_top_subset_count += 1
            if not bool(head_top.get("pass", True)):
                stats.head_top_violation_count += 1

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
    if str(horizon.get("state", "")).strip() != "none" and h_dist is not None:
        stats.horizon_subset_count += 1
        stats.horizon_third_dist.append(float(h_dist))
        if not bool(horizon.get("pass", True)):
            stats.horizon_violation_count += 1

    if route["flags"].get("is_landscape_scene", False) or is_scene_mode(route.get("subject_mode", "")):
        roll_check = checks.get("roll", {}) if isinstance(checks.get("roll"), dict) else {}
        roll = roll_check.get("value", horizon.get("roll_deg"))
        if roll is not None:
            stats.roll_abs.append(abs(safe_float(roll, 0.0)))
            stats.roll_subset_count += 1
            if abs(safe_float(roll, 0.0)) > cfg.roll_violation_deg:
                stats.roll_violation_count += 1

    # Copy-space preservation
    if route["flags"].get("has_copy_space", False):
        stats.copyspace_subset_count += 1
        copyspace = checks.get("copyspace", {}) if isinstance(checks.get("copyspace"), dict) else {}
        if bool(copyspace.get("pass", False)):
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
        "head_top_cut_rate": stats.top1_head_top_cut_count / n,
        "joint_cut_rate": stats.top1_joint_cut_count / n,
        "subject_coverage_fail_rate": stats.top1_subject_cov_fail_count / n,
        "headroom_violation_rate": (
            stats.headroom_violation_count / max(1, stats.portrait_subset_count)
        ),
        "head_top_violation_rate": (
            stats.head_top_violation_count / max(1, stats.head_top_subset_count)
        ),
        "lookroom_violation_rate": (
            stats.lookroom_violation_count / max(1, stats.lookroom_subset_count)
        ),
        "horizon_third_dist_p90": percentile(stats.horizon_third_dist, 0.90),
        "roll_violation_rate": stats.roll_violation_count / max(1, stats.roll_subset_count),
        "roll_abs_p50": percentile(stats.roll_abs, 0.50),
        "roll_abs_p90": percentile(stats.roll_abs, 0.90),
        "needs_leveling_rate": stats.roll_violation_count / max(1, stats.roll_subset_count),
        "copyspace_preserve_rate": stats.copyspace_preserve_count / max(1, stats.copyspace_subset_count),
        "fallback_rate": stats.fallback_activated_count / n,
        "fallback_mode_counts": dict(stats.fallback_mode_counts),
        "topk_mean_iou_top1_top2": mean(stats.top1_top2_iou) if stats.top1_top2_iou else 0.0,
        "denominators": {
            "portrait_subset": stats.portrait_subset_count,
            "head_top_subset": stats.head_top_subset_count,
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
    routing_hint = _resolve_subject_routing_hint(cand_rec=cand_rec, feat_rec=feat_rec)

    c3_info = collect_c3_info(feat_rec=feat_rec, width=width, height=height, cfg=cfg)
    c5_info = collect_c5_info(feat_rec=feat_rec)
    c4_info = collect_c4_info(feat_rec=feat_rec, width=width, height=height)

    c2_person_count = count_c2_person_instances(feat_rec)
    person_stats = resolve_num_people_for_route(
        routing_hint=routing_hint,
        c3_info=c3_info,
        c2_person_count=c2_person_count,
    )
    num_people = int(person_stats["num_people"])
    face_count = int(person_stats["face_count"])
    has_human_evidence = num_people > 0

    shot_type = infer_shot_type(tags, has_human_evidence=has_human_evidence)
    portrait_category = infer_portrait_category(tags)
    flags = infer_flags(tags, has_human_evidence=has_human_evidence)

    has_people = num_people > 0

    hint_shot_type = str(routing_hint.get("shot_type", "")).strip().lower()
    if hint_shot_type in {"headshot", "half", "full", "group"}:
        shot_type = hint_shot_type
    hint_subject_mode = str(routing_hint.get("subject_mode", "")).strip()
    hint_policy_id = str(routing_hint.get("policy_id", "")).strip()
    hint_scene_subtype = str(routing_hint.get("scene_subtype", "")).strip()
    subject_scale_range, subject_scale_target_region = subject_scale_target_range(
        subject_mode=hint_subject_mode,
        shot_type=shot_type,
        scene_subtype=hint_scene_subtype,
        flags=flags,
        has_human_evidence=bool(has_human_evidence),
    )
    context_keep_range, context_target_region = context_keep_target_range(
        subject_mode=hint_subject_mode,
        scene_subtype=hint_scene_subtype,
        flags=flags,
    )

    route = {
        "shot_type": shot_type,
        "portrait_category": portrait_category,
        "flags": flags,
        "num_people": num_people,
        "c2_person_count": c2_person_count,
        "face_count": face_count,
        "has_human_evidence": bool(has_human_evidence),
        "norm_size_source": norm_size_source,
        "headroom_range": headroom_prior(shot_type, portrait_category, flags),
        "lookroom_range": lookroom_prior(shot_type, flags),
        "context_range": context_keep_range,
        "legacy_context_range": context_target_range(
            shot_type=shot_type,
            flags=flags,
            portrait_category=portrait_category,
            has_human_evidence=bool(has_human_evidence),
        ),
        "subject_scale_range": subject_scale_range,
        "subject_scale_target_region": subject_scale_target_region,
        "context_target_region": context_target_region,
        "tau_improve": tau_improve_for_route(shot_type, flags, num_people),
        "w_area": area_weight_for_route(flags),
        "lambdas": effective_lambdas(cfg, shot_type, flags, has_people),
    }
    route = apply_subject_policy_overrides(
        route=route,
        subject_mode=hint_subject_mode,
        policy_id=hint_policy_id,
        routing_hint=routing_hint,
    )
    feature_stage_debug = summarize_feature_stage_debug(
        feat_rec=feat_rec,
        c3_info=c3_info,
        c4_info=c4_info,
        c5_info=c5_info,
        c2_person_count=c2_person_count,
        c1_text_embed=c1_text_embed,
    )
    routing_hint_debug = {
        "subject_mode": str(hint_subject_mode or ""),
        "scene_subtype": str(routing_hint.get("scene_subtype", "") or ""),
        "policy_id": str(hint_policy_id or ""),
        "shot_type_hint": str(hint_shot_type or ""),
        "subject_mode_conf": safe_float(routing_hint.get("subject_mode_conf", 0.0), 0.0),
        "subject_mode_reasons": (
            routing_hint.get("subject_mode_reasons", [])
            if isinstance(routing_hint.get("subject_mode_reasons"), list)
            else []
        ),
        "subject_mode_conflict": bool(routing_hint.get("subject_mode_conflict", False)),
        "subject_set": (
            routing_hint.get("subject_set", {})
            if isinstance(routing_hint.get("subject_set"), dict)
            else {}
        ),
        "router_rule_id": str(routing_hint.get("router_rule_id", "")),
        "router_signals": (
            routing_hint.get("router_signals", {})
            if isinstance(routing_hint.get("router_signals"), dict)
            else {}
        ),
        "copyspace": (
            routing_hint.get("copyspace", {})
            if isinstance(routing_hint.get("copyspace"), dict)
            else {}
        ),
    }

    subject_box = build_subject_bbox(cand_rec, feat_rec, width=width, height=height)
    subject_support = build_subject_support_context(subject_box, feat_rec, route)
    subject_centroid = tuple(float(v) for v in subject_support.get("centroid_xy_norm", box_center(subject_box)))

    cand_stats = cand_rec.get("stats", {}) if isinstance(cand_rec.get("stats"), dict) else {}
    proposal_injected = bool(cand_rec.get("proposal_injected", False))
    teacher_cands_total = int(safe_float(cand_stats.get("num_teacher_candidates_total", 0), 0.0))
    teacher_cands_by_ar = (
        cand_stats.get("num_teacher_candidates_by_ar", {})
        if isinstance(cand_stats.get("num_teacher_candidates_by_ar"), dict)
        else {}
    )
    iou_to_teacher_top1_by_ar = (
        cand_rec.get("iou_to_teacher_top1_by_ar", {})
        if isinstance(cand_rec.get("iou_to_teacher_top1_by_ar"), dict)
        else {}
    )

    results_by_ar: Dict[str, Any] = {}
    stats_for_ar: Dict[str, Dict[str, float]] = {}
    per_ar_tmp: Dict[str, Dict[str, Any]] = {}

    for ar_text, cands in (cand_rec.get("candidates_by_ar", {}) or {}).items():
        target_ar = parse_target_ar(ar_text)
        is_freeform = target_ar is None
        if not isinstance(cands, list) or not cands:
            continue
        teacher_ctx = build_teacher_consensus_context(cand_rec=cand_rec, ar_text=ar_text, cfg=cfg)
        public_teacher_refs_raw = (
            select_public_teacher_refs(cand_rec=cand_rec, ar_text=ar_text)
            if bool(cfg.save_public_teacher_ref_eval)
            else []
        )

        scored: List[Dict[str, Any]] = []
        for c in cands:
            scored.append(
                compute_candidate_scores(
                    candidate=c,
                    target_ar=target_ar,
                    image_ar=image_ar,
                    subject_box=subject_box,
                    subject_support=subject_support,
                    subject_centroid=subject_centroid,
                    c3_info=c3_info,
                    c5_info=c5_info,
                    c4_info=c4_info,
                    route=route,
                    teacher_ctx=teacher_ctx,
                    cfg=cfg,
                )
            )

        public_teacher_ref_eval: List[Dict[str, Any]] = []
        for ref in public_teacher_refs_raw:
            ref_bbox = ref.get("bbox_norm_xyxy")
            if not isinstance(ref_bbox, list) or len(ref_bbox) != 4:
                continue
            ref_candidate = {
                "candidate_id": f"public_teacher_ref::{ar_text}::{ref['teacher_id']}::{ref['stage']}",
                "bbox_norm_xyxy": [safe_float(v) for v in ref_bbox],
                "area_ratio": float(box_area(ref_bbox)),
                "ar": float(parse_target_ar(ar_text) or image_ar),
                "source": f"public_teacher_ref:{ref['teacher_id']}",
                "source_types": ["public_teacher_ref"],
                "must_keep": False,
                "priority": 0.0,
                "teacher_id": str(ref["teacher_id"]),
                "teacher_stage": str(ref["stage"]),
                "teacher_target_ar": str(ref.get("target_ar", ar_text)),
                "teacher_raw_score": float(ref.get("teacher_score", 0.0)),
                "teacher_ref_kind": "raw_public_teacher_ref",
            }
            public_teacher_ref_eval.append(
                compute_candidate_scores(
                    candidate=ref_candidate,
                    target_ar=target_ar,
                    image_ar=image_ar,
                    subject_box=subject_box,
                    subject_support=subject_support,
                    subject_centroid=subject_centroid,
                    c3_info=c3_info,
                    c5_info=c5_info,
                    c4_info=c4_info,
                    route=route,
                    teacher_ctx=teacher_ctx,
                    cfg=cfg,
                )
            )

        baseline, ref_center = pick_baseline_candidates(scored)

        strict_valid = [c for c in scored if not _has_training_severe_reject(c)]
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
            baseline_hard = _has_training_severe_reject(baseline)
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
            "is_freeform": bool(is_freeform),
            "scored": scored,
            "baseline": baseline,
            "ref_center": ref_center,
            "cheap_top_m": cheap_top_m,
            "expensive_eval_pool": expensive_eval_pool,
            "fallback_info": fallback_info,
            "teacher_ctx": teacher_ctx,
            "public_teacher_ref_eval": public_teacher_ref_eval,
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
            for c in tmp.get("public_teacher_ref_eval", []):
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
        teacher_ctx = tmp.get("teacher_ctx", {}) if isinstance(tmp.get("teacher_ctx"), dict) else {}
        public_teacher_ref_eval = tmp.get("public_teacher_ref_eval", [])
        target_ar = tmp.get("target_ar")
        is_freeform = bool(tmp.get("is_freeform", False))

        normalize_final_scores(cheap_top_m)
        exp_sorted = sorted(cheap_top_m, key=lambda x: safe_float(x["scores"].get("final", -1e9)), reverse=True)
        non_hard_sorted = [c for c in exp_sorted if not _has_training_severe_reject(c)]
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

        teacher_boxes = teacher_ctx.get("boxes", []) if isinstance(teacher_ctx.get("boxes"), list) else []
        baseline_iou_to_teacher = 0.0
        if baseline_effective is not None and teacher_boxes:
            baseline_iou_to_teacher = max(
                iou_xyxy(baseline_effective.get("bbox_norm_xyxy", [0, 0, 1, 1]), tb) for tb in teacher_boxes
            )
        tau_base = float(route["tau_improve"])
        tau_effective = tau_base
        tau_boost_applied = False
        if (
            bool(teacher_ctx.get("consensus", False))
            and baseline_iou_to_teacher >= float(cfg.teacher_tau_boost_baseline_iou)
        ):
            tau_effective = tau_base + float(cfg.teacher_tau_boost_delta)
            tau_boost_applied = True

        decision = decide_keep_vs_crop(
            best=best,
            baseline=baseline_effective,
            tau_improve=tau_effective,
        )

        force_ids = [decision["chosen_candidate_id"], baseline_effective.get("candidate_id")]
        selected_topk = select_topk_diverse(
            sorted_cands=rank_pool,
            k=cfg.top_k,
            tau_div=cfg.tau_div,
            force_ids=force_ids,
            ar_log_gap_thr=(cfg.free_topk_ar_log_gap if is_freeform else None),
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
                    ar_log_gap_thr=(cfg.free_topk_ar_log_gap if is_freeform else None),
                )
                selected_topk.extend(fill)
        selected_topk = selected_topk[: cfg.top_k]

        if selected_topk:
            normalize_final_scores(selected_topk)
        hard_negs = select_hard_negatives(exp_sorted, max_n=5)
        top1_selected = selected_topk[0] if selected_topk else {}
        if not isinstance(top1_selected, dict):
            top1_selected = {}

        rejected_sorted = sorted(
            [c for c in scored if _has_training_severe_reject(c)],
            key=lambda x: safe_float(x["scores"].get("cheap", -1e9)),
            reverse=True,
        )

        ar_result = {
            "target_ar": ar_text,
            "target_ar_value": (None if is_freeform else float(target_ar)),
            "is_freeform": bool(is_freeform),
            "num_input_candidates": len(scored),
            "num_valid_candidates": int(fallback_info.get("strict_valid_count", 0)),
            "num_effective_candidates": int(fallback_info.get("effective_valid_count", 0)),
            "num_cheap_kept": len(cheap_top_m),
            "num_expensive_eval": len(expensive_eval_pool),
            "baseline_candidate": candidate_brief(baseline_effective),
            "best_candidate": candidate_brief(best),
            "keep_policy": {
                "tau_improve": tau_effective,
                "tau_improve_base": tau_base,
                "w_area": route["w_area"],
                "ranking_metric": "score_rank",
                "decision_metric": "score_policy",
                "force_include_baseline_in_topm": True,
                "is_freeform": bool(is_freeform),
                "teacher_tau_boost": {
                    "applied": bool(tau_boost_applied),
                    "delta": float(cfg.teacher_tau_boost_delta),
                    "baseline_iou": float(baseline_iou_to_teacher),
                    "baseline_iou_thr": float(cfg.teacher_tau_boost_baseline_iou),
                    "consensus_required": bool(cfg.teach_require_consensus),
                },
            },
            "fallback": fallback_info,
            "decision": decision,
            "cheap_top_m": [candidate_brief(c) for c in cheap_top_m],
            "selected_topk": [candidate_brief(c, rank=i + 1) for i, c in enumerate(selected_topk)],
            "hard_negatives": [candidate_brief(c) for c in hard_negs],
            "also_considered_rejected": [candidate_brief(c) for c in rejected_sorted[:5]],
            "routing": {
                "shot_type": route.get("shot_type", shot_type),
                "portrait_category": portrait_category,
                "flags": route.get("flags", flags),
                "num_people": int(num_people),
                "c2_person_count": int(c2_person_count),
                "has_human_evidence": bool(has_human_evidence),
                "norm_size_source": norm_size_source,
                "subject_mode": route.get("subject_mode", "other_ambiguous"),
                "policy_id": route.get("policy_id", "generic_v1"),
                "subject_mode_conf": safe_float(route.get("subject_mode_conf", 0.0)),
                "subject_mode_reasons": route.get("subject_mode_reasons", []),
                "subject_mode_conflict": bool(route.get("subject_mode_conflict", False)),
                "subject_set": route.get("subject_set", {}),
                "router_rule_id": route.get("router_rule_id", ""),
                "router_signals": route.get("router_signals", {}),
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
            "proposal_injection": {
                "enabled": bool(proposal_injected),
                "num_teacher_candidates_ar": int(safe_float(teacher_cands_by_ar.get(ar_text, 0), 0.0)),
                "candidate_top1_iou_to_teacher_seed": safe_float(iou_to_teacher_top1_by_ar.get(ar_text, 0.0), 0.0),
                "public_teacher_ref_eval_saved": bool(cfg.save_public_teacher_ref_eval),
                "public_teacher_ref_eval": [candidate_brief(c) for c in public_teacher_ref_eval],
                "teacher_consensus": {
                    "available": bool(teacher_ctx.get("available", False)),
                    "num_teachers": int(safe_float(teacher_ctx.get("num_teachers", 0), 0.0)),
                    "num_boxes": int(safe_float(teacher_ctx.get("num_boxes", 0), 0.0)),
                    "consensus": bool(teacher_ctx.get("consensus", False)),
                    "max_pair_iou": float(safe_float(teacher_ctx.get("max_pair_iou", 0.0), 0.0)),
                    "baseline_iou": float(baseline_iou_to_teacher),
                    "tau_boost_applied": bool(tau_boost_applied),
                },
            },
            "stage_debug": {
                "candidate_generation": {
                    "num_input_candidates": int(len(scored)),
                    "num_strict_valid_candidates": int(fallback_info.get("strict_valid_count", 0)),
                    "num_effective_candidates": int(fallback_info.get("effective_valid_count", 0)),
                    "proposal_injected": bool(proposal_injected),
                    "teacher_candidates_ar": int(safe_float(teacher_cands_by_ar.get(ar_text, 0), 0.0)),
                },
                "cheap_stage": {
                    "num_cheap_kept": int(len(cheap_top_m)),
                    "cheap_top_m_ids": [str(c.get("candidate_id", "")) for c in cheap_top_m[:10]],
                },
                "expensive_stage": {
                    "num_expensive_eval": int(len(expensive_eval_pool)),
                    "real_applied": bool(expensive_real_applied),
                    "uses_clip_text_alignment": bool(expensive_real_applied),
                },
                "decision_stage": {
                    "decision_type": str(decision.get("decision_type", "")),
                    "chosen_candidate_id": str(decision.get("chosen_candidate_id", "")),
                    "baseline_candidate_id": str(baseline_effective.get("candidate_id", "")),
                    "best_candidate_id": str(best.get("candidate_id", "")),
                    "delta_improve": float(decision.get("delta_improve", 0.0)),
                    "tau_improve": float(decision.get("tau_improve", tau_effective)),
                    "best_score_rank": float(safe_float(best.get("scores", {}).get("rank", 0.0), 0.0)),
                    "best_score_policy": float(safe_float(best.get("scores", {}).get("policy", 0.0), 0.0)),
                    "baseline_score_rank": float(safe_float(baseline_effective.get("scores", {}).get("rank", 0.0), 0.0)),
                    "baseline_score_policy": float(safe_float(baseline_effective.get("scores", {}).get("policy", 0.0), 0.0)),
                },
                "top1_explainability": {
                    "candidate_id": str(top1_selected.get("candidate_id", "")),
                    "checklist_labels": (
                        top1_selected.get("checklist_labels", {})
                        if isinstance(top1_selected.get("checklist_labels"), dict)
                        else {}
                    ),
                    "why_tags": (
                        top1_selected.get("why_tags", [])
                        if isinstance(top1_selected.get("why_tags"), list)
                        else []
                    ),
                    "why_text_template": str(top1_selected.get("why_text_template", "")),
                    "macro_scores": (
                        top1_selected.get("macro_scores", {})
                        if isinstance(top1_selected.get("macro_scores"), dict)
                        else {}
                    ),
                },
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
        "proposal_injected": bool(proposal_injected),
        "candidate_meta": {
            "proposal_injected": bool(proposal_injected),
            "num_teacher_candidates_total": int(teacher_cands_total),
            "num_teacher_candidates_by_ar": teacher_cands_by_ar,
            "iou_to_teacher_top1_by_ar": iou_to_teacher_top1_by_ar,
        },
        "route_global": {
            "shot_type": route.get("shot_type", shot_type),
            "portrait_category": portrait_category,
            "flags": route.get("flags", flags),
            "num_people": num_people,
            "c2_person_count": int(c2_person_count),
            "has_human_evidence": bool(has_human_evidence),
            "norm_size_source": norm_size_source,
            "subject_mode": route.get("subject_mode", "other_ambiguous"),
            "scene_subtype": route.get("scene_subtype", SCENE_SUBTYPE_UNKNOWN),
            "scene_conf": safe_float(route.get("scene_conf", 0.0)),
            "policy_id": route.get("policy_id", "generic_v1"),
            "subject_mode_conf": safe_float(route.get("subject_mode_conf", 0.0)),
            "subject_mode_reasons": route.get("subject_mode_reasons", []),
            "subject_mode_conflict": bool(route.get("subject_mode_conflict", False)),
            "subject_set": route.get("subject_set", {}),
            "router_rule_id": route.get("router_rule_id", ""),
            "router_signals": route.get("router_signals", {}),
            "copyspace": route.get("copyspace", {}),
        },
        "pipeline_debug": {
            "feature_stage": feature_stage_debug,
            "routing_hint": routing_hint_debug,
            "route_applied": {
                "subject_mode": route.get("subject_mode", "other_ambiguous"),
                "scene_subtype": route.get("scene_subtype", SCENE_SUBTYPE_UNKNOWN),
                "policy_id": route.get("policy_id", "generic_v1"),
                "subject_mode_conf": safe_float(route.get("subject_mode_conf", 0.0)),
                "subject_mode_reasons": route.get("subject_mode_reasons", []),
                "subject_mode_conflict": bool(route.get("subject_mode_conflict", False)),
                "shot_type": route.get("shot_type", shot_type),
                "portrait_category": route.get("portrait_category", portrait_category),
                "headroom_range": route.get("headroom_range", []),
                "lookroom_range": route.get("lookroom_range", []),
                "context_range": route.get("context_range", []),
                "subject_scale_range": route.get("subject_scale_range", []),
                "subject_scale_target_region": route.get("subject_scale_target_region", ""),
                "context_target_region": route.get("context_target_region", ""),
                "lambdas": route.get("lambdas", {}),
            },
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
        enable_r_teach=bool(int(args.enable_r_teach)),
        teach_rho_tau=float(args.teach_rho_tau),
        teach_rho_beta=float(args.teach_rho_beta),
        teach_consensus_pair_iou_thr=float(args.teach_consensus_pair_iou_thr),
        teach_require_consensus=bool(int(args.teach_require_consensus)),
        lambda_teach=float(args.lambda_teach),
        w_teach=float(args.w_teach),
        rank_weight_a=float(args.rank_weight_a),
        rank_weight_s=float(args.rank_weight_s),
        rank_weight_c=float(args.rank_weight_c),
        rank_weight_t=float(args.rank_weight_t),
        a_macro_aesthetic_weight=float(args.a_macro_aesthetic_weight),
        a_macro_align_weight=float(args.a_macro_align_weight),
        teacher_tau_boost_delta=float(args.teacher_tau_boost_delta),
        teacher_tau_boost_baseline_iou=float(args.teacher_tau_boost_baseline_iou),
        hard_head_top_rule=bool(int(args.hard_head_top_rule)),
        hard_portrait_lookroom_rule=bool(int(args.hard_portrait_lookroom_rule)),
        lookroom_hard_reject_min_subject_area_ratio=float(args.lookroom_hard_reject_min_subject_area_ratio),
        lookroom_hard_reject_min_route_conf=float(args.lookroom_hard_reject_min_route_conf),
        head_top_face_expand_alpha=float(args.head_top_face_expand_alpha),
        head_top_kp_expand=float(args.head_top_kp_expand),
        head_top_min_margin=float(args.head_top_min_margin),
        head_top_face_margin_alpha=float(args.head_top_face_margin_alpha),
        expensive_eval_top_m=max(0, int(args.expensive_eval_top_m)),
        save_public_teacher_ref_eval=bool(int(args.save_public_teacher_ref_eval)),
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
            aesthetic_backend=str(args.aesthetic_backend),
            aesthetic_prior_laion_weight=float(args.aesthetic_prior_laion_weight),
            aesthetic_mlp_path=Path(args.aesthetic_mlp_path),
            aesthetic_mlp_url=str(args.aesthetic_mlp_url),
            nima_model_path=Path(args.nima_model_path),
            nima_model_url=str(args.nima_model_url),
            nima_use_imagenet_backbone=bool(int(args.nima_use_imagenet_backbone)),
            nima_require_ckpt=bool(int(args.nima_require_ckpt)),
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
                        "subject_mode": route_info.get("subject_mode", "other_ambiguous"),
                        "policy_id": route_info.get("policy_id", "generic_v1"),
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
        "schema_version": "teacher_scorer_v7_explainability_checklist_debug",
        "config": asdict(cfg),
        "explainability_contract": {
            "enabled": True,
            "fields": ["checklist", "checklist_labels", "why_tags", "why_tags_numeric", "why_text_template"],
            "excluded_unimplemented_items": [
                "comp9_class",
                "comp24_class",
                "picd_compenc",
                "composition_classifier_logits",
                "ocr_layout_class",
            ],
        },
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
            "aesthetic_backend_requested": str(args.aesthetic_backend),
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
            "aesthetic_backend_effective": (
                str(expensive_models.aesthetic_backend_effective)
                if expensive_models is not None
                else None
            ),
            "aesthetic_prior_laion_weight_requested": float(args.aesthetic_prior_laion_weight),
            "aesthetic_prior_laion_weight_effective": (
                float(expensive_models.laion_prior_weight_effective)
                if expensive_models is not None
                else None
            ),
            "aesthetic_mlp_path": str(args.aesthetic_mlp_path),
            "nima_model_path": str(args.nima_model_path),
            "nima_require_ckpt": bool(int(args.nima_require_ckpt)),
            "nima_ckpt_loaded": (
                bool(expensive_models.nima_ckpt_loaded)
                if expensive_models is not None
                else False
            ),
            "nima_device_actual": (
                str(expensive_models.nima_device)
                if expensive_models is not None
                else None
            ),
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
