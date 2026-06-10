from __future__ import annotations

import argparse
import base64
import copy
import csv
import json
import os
import random
import shutil
import sys
import textwrap
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from debug_viz_detail_utils import build_teacher_detail_index
from progress_utils import ProgressTracker, progress_log
from score_profile_utils import active_score_components


POSITIVE_COLOR = "#2E8B57"
NEGATIVE_COLOR = "#C0392B"
GT_COLOR = "#1F77B4"
TEXT_BG = (0, 0, 0, 245)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TOP_PANEL_BG = (12, 12, 14, 255)
CHIP_GOOD = ("#D9FBE7", "#173D2A", "#2FA66A")
CHIP_WARN = ("#FFF4C2", "#4E3C00", "#D0A020")
CHIP_BAD = ("#FFD9D6", "#4D1614", "#D94C43")
CHIP_NA = ("#D7D9E0", "#2F3440", "#7C8496")
DEFAULT_VIZ_FORMAT = "jpg"
DEFAULT_JPG_QUALITY = 50
SUBJECT_COLOR = "#10D4D8"
SUBJECT_CORE_COLOR = "#F59E0B"
SUBJECT_SECONDARY_COLOR = "#34D399"
SUBJECT_ENVELOPE_COLOR = "#7DD3FC"
SCORE_GROUP_COLORS = {
    "utility": "#4AA9FF",
    "penalty": "#E86B5A",
    "A": "#D8A038",
    "S": "#36B37E",
    "C": "#4AA9FF",
    "T": "#51C4B8",
    "other": "#A6B0C3",
}
FOCUS_MACRO_ORDER = ("A_macro", "S_macro", "C_macro", "T_macro")
FOCUS_COMPONENT_ORDER = (
    "A_aesthetic",
    "A_align",
    "S_cov",
    "S_scale",
    "S_support_structure",
    "S_border",
    "S_softcut_quality",
    "C_place",
    "C_comp",
    "C_headroom",
    "C_lookroom",
    "C_horizon_y",
    "C_sym",
    "C_context",
    "C_copyspace",
    "T_teacher",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GAIC training-label debug visualizations.")
    p.add_argument(
        "--label_json",
        default="",
    )
    p.add_argument(
        "--batch_jsonl",
        default="data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl",
        help="Optional JSONL input used only for ignored-candidate counts in the summary.",
    )
    p.add_argument("--gaic_gt_train_json", default="data/Publics/GAIC/annotations_json/instances_train.json")
    p.add_argument("--gaic_gt_test_json", default="data/Publics/GAIC/annotations_json/instances_test.json")
    p.add_argument(
        "--gt_score_cache_jsonl",
        default="",
        help="Optional GAIC GT FREE-context training score cache JSONL produced by build_gaic_gt_score_cache.py.",
    )
    p.add_argument(
        "--teacher_jsonl",
        default="",
        help="Optional teacher JSONL used to render checklist/macro/policy side panels.",
    )
    p.add_argument(
        "--candidates_jsonl",
        default="",
        help="Optional candidates JSONL used to render subject-region overlay (support-map preferred, bbox fallback).",
    )
    p.add_argument("--image_root", default="data/GAIC/All/images")
    p.add_argument(
        "--subject_mode_vocab",
        default="data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/subject_mode_vocab.json",
    )
    p.add_argument("--sample_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--image_ids_csv",
        default="",
        help="Optional CSV with an image_id column. When set, sampling is bypassed and these image_ids are rendered in the given order.",
    )
    p.add_argument("--top_k_gt", type=int, default=3)
    p.add_argument("--top_k_positive", type=int, default=10)
    p.add_argument("--top_k_negative", type=int, default=10)
    p.add_argument("--show_why_text", type=int, default=0, help="Set to 1 to render why_text in checklist panels.")
    p.add_argument("--skip_by_ar", type=int, default=0, help="Set to 1 to skip by-target-ar visualization outputs.")
    p.add_argument("--format", choices=("auto", "png", "jpg"), default=DEFAULT_VIZ_FORMAT)
    p.add_argument("--jpg_quality", type=int, default=DEFAULT_JPG_QUALITY)
    p.add_argument(
        "--score_profile_json",
        default="",
        help="Optional score_profile.json emitted by build_finalscore_training_data.py. When set, checklist panels only render active score components.",
    )
    p.add_argument(
        "--out_dir",
        default="data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/debug_visualizations_balanced50_bottomneg",
    )
    p.add_argument("--num_workers", type=int, default=0, help="CPU render workers for per-image debug visualizations (0=all cores, 1=single).")
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--progress_min_seconds", type=float, default=10.0)
    return p.parse_args()


def resolve_num_workers(requested_workers: int, num_items: int) -> int:
    if num_items <= 1:
        return 1
    req = int(requested_workers)
    if req == 1:
        return 1
    if req <= 0:
        req = max(1, int(os.cpu_count() or 1))
    return max(1, min(req, num_items))


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_image_ids_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        result: List[str] = []
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                result.append(image_id)
        return result


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def merge_dicts_prefer_override(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(safe_dict(base))
    for key, value in safe_dict(override).items():
        if value in (None, "", {}, []):
            continue
        out[key] = copy.deepcopy(value)
    return out


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_viz_format(fmt: str) -> str:
    normalized = str(fmt or "").strip().lower()
    if normalized in {"", "auto"}:
        return "jpg"
    if normalized in {"jpg", "jpeg"}:
        return "jpg"
    return "png"


def viz_suffix(fmt: str) -> str:
    return ".jpg" if resolve_viz_format(fmt) == "jpg" else ".png"


def build_viz_path(directory: Path, stem: str, fmt: str) -> Path:
    return directory / f"{stem}{viz_suffix(fmt)}"


def save_pil_visualization(image: Image.Image, out_path: Path, *, jpg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            background.paste(image.convert("RGB"), mask=alpha)
            image = background
        else:
            image = image.convert("RGB")
        image.save(
            out_path,
            format="JPEG",
            quality=max(1, min(100, int(jpg_quality))),
            subsampling=0,
            optimize=True,
        )
        return
    image.save(out_path)


def save_matplotlib_figure(fig: plt.Figure, out_path: Path, *, jpg_quality: int, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        fig.savefig(
            out_path,
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pil_kwargs={
                "quality": max(1, min(100, int(jpg_quality))),
                "subsampling": 0,
                "optimize": True,
            },
        )
        return
    fig.savefig(out_path, dpi=dpi)


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def clip_box01(box: Sequence[Any]) -> List[float]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1 = clamp(safe_float(box[0]), 0.0, 1.0)
    y1 = clamp(safe_float(box[1]), 0.0, 1.0)
    x2 = clamp(safe_float(box[2]), 0.0, 1.0)
    y2 = clamp(safe_float(box[3]), 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _decode_support_map(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    grid_size = safe_int(first_nonempty(payload.get("support_grid_size"), payload.get("grid_size")), 0)
    mass_b64 = first_nonempty(payload.get("support_mass_grid_f16_b64"), payload.get("mass_grid_f16_b64"))
    occ_b64 = first_nonempty(payload.get("support_occ_grid_u8_b64"), payload.get("occ_grid_u8_b64"))
    if grid_size <= 0 or not isinstance(mass_b64, str) or not isinstance(occ_b64, str) or not mass_b64 or not occ_b64:
        return None
    try:
        mass_grid = np.frombuffer(base64.b64decode(mass_b64.encode("ascii")), dtype=np.float16).astype(np.float32)
        occ_grid = np.frombuffer(base64.b64decode(occ_b64.encode("ascii")), dtype=np.uint8).astype(np.float32)
    except Exception:
        return None
    if mass_grid.size != grid_size * grid_size or occ_grid.size != grid_size * grid_size:
        return None
    mass_grid = mass_grid.reshape((grid_size, grid_size))
    occ_grid = (occ_grid.reshape((grid_size, grid_size)) > 0.0).astype(np.float32)
    max_mass = float(np.max(mass_grid))
    mass_sum = float(np.sum(mass_grid))
    if max_mass <= 1e-8 or mass_sum <= 1e-8:
        return None
    centroid = payload.get("centroid_xy_norm")
    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        centroid_xy = (clamp(safe_float(centroid[0], 0.5), 0.0, 1.0), clamp(safe_float(centroid[1], 0.5), 0.0, 1.0))
    else:
        ys, xs = np.mgrid[0:grid_size, 0:grid_size]
        centroid_xy = (
            float((mass_grid * (xs + 0.5)).sum() / mass_sum) / float(grid_size),
            float((mass_grid * (ys + 0.5)).sum() / mass_sum) / float(grid_size),
        )
    return {
        "grid_size": int(grid_size),
        "mass_grid": mass_grid,
        "occ_grid": occ_grid,
        "centroid_xy_norm": centroid_xy,
        "foreground_area_ratio": clamp(safe_float(payload.get("foreground_area_ratio", 0.0), 0.0), 0.0, 1.0),
    }


def load_subject_overlay_index(
    *,
    candidates_jsonl: Path,
    batch_jsonl: Path,
    image_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    wanted = {str(image_id).strip() for image_id in image_ids if str(image_id).strip()}
    if not wanted:
        return {}
    by_image: Dict[str, Dict[str, Any]] = {}

    # Prefer candidate-side support maps when available; batch labels may be an
    # empty compact sidecar in smoke tests or may omit the full support grid.
    if candidates_jsonl.exists():
        with candidates_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(by_image) >= len(wanted):
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = str(row.get("image_id", "")).strip()
                if image_id not in wanted or image_id in by_image:
                    continue
                routing = safe_dict(row.get("routing"))
                subject_prior = safe_dict(row.get("subject_prior"))
                effective_subject_region = safe_dict(
                    first_nonempty(
                        routing.get("effective_subject_region"),
                        subject_prior.get("effective_subject_region"),
                    )
                )
                crop_guidance_spec = safe_dict(effective_subject_region.get("crop_guidance_spec"))
                support_spec = safe_dict(crop_guidance_spec.get("support_spec"))
                semantic_spec = safe_dict(crop_guidance_spec.get("semantic_spec"))
                guidance_spec = safe_dict(crop_guidance_spec.get("guidance_spec"))
                overlay_mode = (
                    "support_map"
                    if (
                        safe_bool(effective_subject_region.get("support_map_enabled"), False)
                        or str(first_nonempty(effective_subject_region.get("score_mode"), "") or "") == "support_map"
                    )
                    else "bbox"
                )
                support_map_enabled = safe_bool(
                    first_nonempty(effective_subject_region.get("support_map_enabled"), support_spec.get("mode") == "support_map"),
                    False,
                )
                support_hybrid_enabled = safe_bool(
                    first_nonempty(effective_subject_region.get("support_hybrid_enabled"), support_spec.get("hybrid_enabled")),
                    False,
                )
                use_support_map = bool(overlay_mode == "support_map" or support_map_enabled or support_hybrid_enabled)
                decoded = _decode_support_map(support_spec) if use_support_map else None
                bbox = clip_box01(
                    first_nonempty(
                        effective_subject_region.get("effective_bbox_norm_xyxy"),
                        effective_subject_region.get("support_bbox_norm_xyxy"),
                        effective_subject_region.get("scoring_bbox_norm_xyxy"),
                        subject_prior.get("bbox_norm_xyxy"),
                    )
                    or [0.0, 0.0, 1.0, 1.0]
                )
                by_image[image_id] = {
                    "mode": overlay_mode,
                    "bbox_norm_xyxy": bbox,
                    "guidance_core_bbox_norm_xyxy": clip_box01(guidance_spec.get("core_bbox_norm_xyxy") or bbox),
                    "guidance_envelope_bbox_norm_xyxy": clip_box01(guidance_spec.get("envelope_bbox_norm_xyxy") or bbox),
                    "guidance_secondary_core_bbox_norm_xyxy": (
                        clip_box01(guidance_spec.get("secondary_core_bbox_norm_xyxy"))
                        if isinstance(guidance_spec.get("secondary_core_bbox_norm_xyxy"), (list, tuple))
                        and len(guidance_spec.get("secondary_core_bbox_norm_xyxy")) == 4
                        else None
                    ),
                    "decoded_support_map": decoded,
                    "subject_mode": str(routing.get("subject_mode", "")),
                    "subject_family": str(first_nonempty(routing.get("subject_family"), semantic_spec.get("subject_family")) or ""),
                    "subject_subtype": str(first_nonempty(routing.get("subject_subtype"), semantic_spec.get("subject_subtype")) or ""),
                    "layout_structure": str(first_nonempty(routing.get("layout_structure"), semantic_spec.get("layout_structure")) or ""),
                    "support_trust_tier": str(
                        first_nonempty(
                            routing.get("support_trust_tier"),
                            semantic_spec.get("support_trust_tier"),
                            support_spec.get("support_trust_tier"),
                        )
                        or ""
                    ),
                    "subject_repr_type": str(first_nonempty(effective_subject_region.get("subject_repr_type"), subject_prior.get("subject_repr_type")) or ""),
                    "score_mode": str(first_nonempty(effective_subject_region.get("score_mode"), safe_dict(routing.get("flags")).get("subject_score_mode")) or ""),
                    "state": str(first_nonempty(effective_subject_region.get("state"), safe_dict(routing.get("flags")).get("subject_effective_state")) or ""),
                    "support_map_enabled": bool(support_map_enabled and decoded is not None),
                    "support_hybrid_enabled": bool(support_hybrid_enabled),
                    "subject_terms_neutralized": safe_bool(safe_dict(routing.get("flags")).get("subject_terms_neutralized"), False),
                    "source": "candidates_jsonl",
                }

    remaining = wanted - set(by_image)
    if remaining and batch_jsonl.exists():
        with batch_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not remaining:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = str(row.get("image_id", "")).strip()
                if image_id not in remaining:
                    continue
                routing = safe_dict(row.get("routing"))
                effective_subject_region = safe_dict(routing.get("effective_subject_region"))
                crop_guidance_spec = safe_dict(effective_subject_region.get("crop_guidance_spec"))
                support_spec = safe_dict(crop_guidance_spec.get("support_spec"))
                semantic_spec = safe_dict(crop_guidance_spec.get("semantic_spec"))
                guidance_spec = safe_dict(crop_guidance_spec.get("guidance_spec"))
                overlay_stub = safe_dict(routing.get("subject_support_overlay"))
                overlay_mode = str(
                    first_nonempty(
                        effective_subject_region.get("score_mode") == "support_map" and "support_map" or None,
                        overlay_stub.get("mode"),
                        support_spec.get("mode") == "support_map" and "support_map" or None,
                        "bbox",
                    )
                    or "bbox"
                ).strip().lower()
                support_map_enabled = safe_bool(
                    first_nonempty(
                        effective_subject_region.get("support_map_enabled"),
                        overlay_stub.get("support_map_enabled"),
                        safe_dict(routing.get("flags")).get("subject_support_map_enabled"),
                    ),
                    False,
                )
                support_hybrid_enabled = safe_bool(
                    first_nonempty(effective_subject_region.get("support_hybrid_enabled"), support_spec.get("hybrid_enabled")),
                    False,
                )
                use_support_map = bool(overlay_mode == "support_map" or support_map_enabled or support_hybrid_enabled)
                decoded = _decode_support_map(support_spec if support_spec else safe_dict(overlay_stub.get("support_spec"))) if use_support_map else None
                bbox = clip_box01(
                    first_nonempty(
                        effective_subject_region.get("effective_bbox_norm_xyxy"),
                        effective_subject_region.get("support_bbox_norm_xyxy"),
                        overlay_stub.get("bbox_norm_xyxy"),
                        routing.get("subject_prior_bbox_norm_xyxy"),
                    )
                    or [0.0, 0.0, 1.0, 1.0]
                )
                by_image[image_id] = {
                    "mode": "support_map" if use_support_map and decoded is not None else "bbox",
                    "bbox_norm_xyxy": bbox,
                    "guidance_core_bbox_norm_xyxy": clip_box01(
                        first_nonempty(guidance_spec.get("core_bbox_norm_xyxy"), overlay_stub.get("core_bbox_norm_xyxy"), bbox)
                    ),
                    "guidance_envelope_bbox_norm_xyxy": clip_box01(
                        first_nonempty(guidance_spec.get("envelope_bbox_norm_xyxy"), overlay_stub.get("envelope_bbox_norm_xyxy"), bbox)
                    ),
                    "guidance_secondary_core_bbox_norm_xyxy": (
                        clip_box01(first_nonempty(guidance_spec.get("secondary_core_bbox_norm_xyxy"), overlay_stub.get("secondary_core_bbox_norm_xyxy")))
                        if isinstance(first_nonempty(guidance_spec.get("secondary_core_bbox_norm_xyxy"), overlay_stub.get("secondary_core_bbox_norm_xyxy")), (list, tuple))
                        and len(first_nonempty(guidance_spec.get("secondary_core_bbox_norm_xyxy"), overlay_stub.get("secondary_core_bbox_norm_xyxy"))) == 4
                        else None
                    ),
                    "decoded_support_map": decoded,
                    "subject_mode": str(routing.get("subject_mode", "")),
                    "subject_family": str(first_nonempty(routing.get("subject_family"), semantic_spec.get("subject_family")) or ""),
                    "subject_subtype": str(first_nonempty(routing.get("subject_subtype"), semantic_spec.get("subject_subtype")) or ""),
                    "layout_structure": str(first_nonempty(routing.get("layout_structure"), semantic_spec.get("layout_structure")) or ""),
                    "support_trust_tier": str(
                        first_nonempty(
                            routing.get("support_trust_tier"),
                            semantic_spec.get("support_trust_tier"),
                            support_spec.get("support_trust_tier"),
                        )
                        or ""
                    ),
                    "subject_repr_type": str(first_nonempty(effective_subject_region.get("subject_repr_type"), overlay_stub.get("subject_repr_type")) or ""),
                    "score_mode": str(first_nonempty(effective_subject_region.get("score_mode"), overlay_stub.get("score_mode"), safe_dict(routing.get("flags")).get("subject_score_mode")) or ""),
                    "state": str(first_nonempty(effective_subject_region.get("state"), overlay_stub.get("state"), safe_dict(routing.get("flags")).get("subject_effective_state")) or ""),
                    "support_map_enabled": bool(support_map_enabled and decoded is not None),
                    "support_hybrid_enabled": bool(support_hybrid_enabled),
                    "subject_terms_neutralized": safe_bool(safe_dict(routing.get("flags")).get("subject_terms_neutralized"), False),
                    "source": "batch_jsonl",
                }
                remaining.remove(image_id)

    return by_image


def load_score_profile_display(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"macros": {}, "components": {}, "semantics": {}}
    payload = load_json(path)
    active = safe_dict(payload.get("active_components"))
    semantics = safe_dict(payload.get("semantics"))
    if active:
        return {
            "macros": safe_dict(active.get("macros")),
            "components": safe_dict(active.get("components")),
            "semantics": semantics,
        }
    scorer_cfg = safe_dict(payload.get("scorer_cfg"))
    if scorer_cfg:
        class CfgObj:
            pass

        cfg_obj = CfgObj()
        for key, value in scorer_cfg.items():
            setattr(cfg_obj, key, value)
        out = active_score_components(cfg_obj)
        out["semantics"] = semantics
        return out
    return {"macros": {}, "components": {}, "semantics": semantics}


def reset_output_subdirs(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def invert_vocab(path: Path) -> Dict[int, str]:
    payload = load_json(path)
    return {safe_int(v, -1): str(k) for k, v in payload.items()}


def choose_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=int(size))
    except OSError:
        return ImageFont.load_default()


def find_image_path(image_root: Path, image_id: str, hinted_path: str) -> Path:
    hinted = Path(hinted_path)
    if hinted.exists():
        return hinted
    rel_hint = (Path.cwd() / hinted) if not hinted.is_absolute() else hinted
    if rel_hint.exists():
        return rel_hint
    for ext in IMAGE_EXTS:
        cand = image_root / f"{image_id}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"image not found for image_id={image_id}")


def norm_to_xyxy(box_norm: Sequence[Any], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(round(clamp(safe_float(box_norm[0]), 0.0, 1.0) * width))
    y1 = int(round(clamp(safe_float(box_norm[1]), 0.0, 1.0) * height))
    x2 = int(round(clamp(safe_float(box_norm[2]), 0.0, 1.0) * width))
    y2 = int(round(clamp(safe_float(box_norm[3]), 0.0, 1.0) * height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def xywh_to_xyxy(box_xywh: Sequence[Any]) -> Tuple[int, int, int, int]:
    x = int(round(safe_float(box_xywh[0])))
    y = int(round(safe_float(box_xywh[1])))
    w = int(round(safe_float(box_xywh[2])))
    h = int(round(safe_float(box_xywh[3])))
    return x, y, x + max(0, w), y + max(0, h)


def label_box(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, x: int, y: int, text: str, color: str) -> None:
    left = max(0, x)
    top = max(0, y)
    bbox = draw.textbbox((left, top), text, font=font)
    pad_x = 4
    pad_y = 3
    rect = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
    draw.rectangle(rect, fill=TEXT_BG, outline=color, width=1)
    draw.text((left, top), text, fill=color, font=font, stroke_width=1, stroke_fill="#000000")


def status_chip_style(label: Any) -> Tuple[str, str, str]:
    text = str(label or "").strip().lower()
    if not text or text in {"na", "none", "n/a"} or text.endswith("_na"):
        return CHIP_NA

    good_exact = {
        "excellent",
        "good",
        "ideal_scale",
        "headroom_ok",
        "lookroom_adequate",
        "no_face_cut",
        "no_joint_cut",
        "text_preserved",
        "copyspace_preserved",
        "context_preserved",
        "rule_of_thirds_strong",
        "phi_grid_strong",
        "center_comp_strong",
        "teacher_aligned",
        "roll_ok",
        "ar_ok",
        "horizon_aligned",
    }
    warn_tokens = ("partial", "marginal", "moderate", "weak")
    bad_tokens = (
        "cut",
        "tight",
        "loose",
        "poor",
        "lost",
        "missing",
        "violation",
        "off_target",
        "reject",
        "unsafe",
        "too_loose",
        "too_tight",
        "failed",
    )
    if text in good_exact or text.endswith("_ok") or text.endswith("_adequate") or text.endswith("_preserved"):
        return CHIP_GOOD
    if any(token in text for token in bad_tokens):
        return CHIP_BAD
    if any(token in text for token in warn_tokens):
        return CHIP_WARN
    return CHIP_WARN


def draw_status_chip(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    cell_width: int,
    key_text: str,
    value_text: str,
    font: ImageFont.ImageFont,
    key_fill: str = "#D6D8DE",
    row_height: int = 56,
) -> None:
    draw.text((x, y + 2), clamp_text(key_text, 14), fill=key_fill, font=font)
    chip_x = x
    chip_y = y + 22
    max_chip_text = max(12, min(26, int(max(1, cell_width) // 10)))
    chip_text = clamp_text(value_text, max_chip_text)
    text_color, fill_color, outline_color = status_chip_style(value_text)
    text_bbox = draw.textbbox((chip_x + 10, chip_y + 5), chip_text, font=font)
    preferred_w = (text_bbox[2] - text_bbox[0]) + 20
    chip_w = max(88, min(int(cell_width), preferred_w))
    chip_h = row_height - 26
    draw.rounded_rectangle(
        (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h),
        radius=8,
        fill=fill_color,
        outline=outline_color,
        width=2,
    )
    draw.text((chip_x + 10, chip_y + 5), chip_text, fill=text_color, font=font)


def draw_label_chip_grid(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    pairs: Sequence[Tuple[str, str]],
    font: ImageFont.ImageFont,
    columns: int = 2,
) -> int:
    if not pairs:
        return y
    columns = max(1, int(columns))
    col_gap = 16
    col_width = max(180, (width - (columns - 1) * col_gap) // columns)
    row_height = 58
    cur_y = y
    for idx, (key_text, value_text) in enumerate(pairs):
        col_idx = idx % columns
        row_idx = idx // columns
        cell_x = x + col_idx * (col_width + col_gap)
        cell_y = y + row_idx * row_height
        draw_status_chip(
            draw,
            x=cell_x,
            y=cell_y,
            cell_width=col_width,
            key_text=key_text,
            value_text=value_text,
            font=font,
            row_height=row_height,
        )
        cur_y = cell_y + row_height
    return cur_y


def candidate_score(item: Dict[str, Any]) -> float:
    if "score_prob" in item:
        return safe_float(item.get("score_prob"), 0.0)
    return safe_float(item.get("score"), 0.0)


def select_positive_overlay_candidates(candidates: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(candidates, key=candidate_score, reverse=True)
    return ordered[: max(0, int(limit))]


def select_negative_overlay_candidates(candidates: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            candidate_score(item),
            str(item.get("target_ar", "")),
            str(item.get("candidate_id", "")),
        ),
    )
    return ordered[: max(0, int(limit))]


def resolve_overlay_label(group_name: str, item: Dict[str, Any], rank: int) -> str:
    if group_name == "gt":
        label = f"GT{rank} MOS={safe_float(item.get('score'), 0.0):.2f}"
        gt_score = item.get("training_score_prob")
        if gt_score is not None:
            label += f" Score={safe_float(gt_score, 0.0):.3f}"
        return label
    label = f"{group_name.upper()[0]}{rank} Score={candidate_score(item):.3f}"
    label_type = str(item.get("label_type", "")).strip()
    if group_name == "negative" and label_type:
        label += f" {label_type}"
    return label


def box_overlay_label(group_name: str, item: Dict[str, Any], rank: int) -> str:
    if group_name == "gt":
        label = f"GT{rank} MOS={safe_float(item.get('score'), 0.0):.2f}"
        gt_score = item.get("training_score_prob")
        if gt_score is not None:
            label += f" Score={safe_float(gt_score, 0.0):.3f}"
        return label
    return f"{group_name.upper()[0]}{rank} Score={candidate_score(item):.3f}"


def subject_overlay_legend_label(subject_overlay: Optional[Dict[str, Any]]) -> Optional[str]:
    if not subject_overlay:
        return None
    mode = str(subject_overlay.get("mode", "bbox"))
    subject_mode = str(subject_overlay.get("subject_mode", "")).strip()
    state = str(subject_overlay.get("state", "")).strip()
    family = str(subject_overlay.get("subject_family", "")).strip()
    subtype = str(subject_overlay.get("subject_subtype", "")).strip()
    layout_structure = str(subject_overlay.get("layout_structure", "")).strip()
    support_trust_tier = str(subject_overlay.get("support_trust_tier", "")).strip()
    support_hybrid_enabled = safe_bool(subject_overlay.get("support_hybrid_enabled"), False)
    bits = ["SUBJ"]
    if support_hybrid_enabled and mode != "support_map":
        bits.append("hybrid")
    else:
        bits.append("support-map" if mode == "support_map" else "bbox")
    if family:
        bits.append(family if not subtype else f"{family}/{subtype}")
    if layout_structure and layout_structure not in {"", "single"}:
        bits.append(layout_structure)
    if support_trust_tier and support_trust_tier not in {"", "low"}:
        bits.append(support_trust_tier)
    if subject_mode:
        bits.append(subject_mode)
    if state:
        bits.append(state)
    return " | ".join(bits)


def render_subject_overlay(
    canvas: Image.Image,
    *,
    subject_overlay: Optional[Dict[str, Any]],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image_x: int,
    image_y: int,
    display_width: int,
    display_height: int,
    original_width: int,
    original_height: int,
) -> None:
    if not subject_overlay:
        return
    bbox_norm_xyxy = clip_box01(subject_overlay.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])
    envelope_norm_xyxy = clip_box01(subject_overlay.get("guidance_envelope_bbox_norm_xyxy") or bbox_norm_xyxy)
    core_norm_xyxy = clip_box01(subject_overlay.get("guidance_core_bbox_norm_xyxy") or bbox_norm_xyxy)
    secondary_core_norm_xyxy = (
        clip_box01(subject_overlay.get("guidance_secondary_core_bbox_norm_xyxy"))
        if isinstance(subject_overlay.get("guidance_secondary_core_bbox_norm_xyxy"), (list, tuple))
        and len(subject_overlay.get("guidance_secondary_core_bbox_norm_xyxy")) == 4
        else None
    )
    decoded = subject_overlay.get("decoded_support_map")
    if isinstance(decoded, dict) and isinstance(decoded.get("mass_grid"), np.ndarray):
        mass_grid = np.asarray(decoded.get("mass_grid"), dtype=np.float32)
        occ_grid = np.asarray(decoded.get("occ_grid"), dtype=np.float32) if isinstance(decoded.get("occ_grid"), np.ndarray) else None
        max_mass = float(np.max(mass_grid)) if mass_grid.size else 0.0
        if max_mass > 1e-8:
            alpha = np.sqrt(np.clip(mass_grid / max_mass, 0.0, 1.0))
            alpha *= 150.0
            if occ_grid is not None and occ_grid.shape == mass_grid.shape:
                alpha = np.maximum(alpha, occ_grid * 34.0)
            alpha_uint8 = np.asarray(np.clip(alpha, 0.0, 255.0), dtype=np.uint8)
            alpha_img = Image.fromarray(alpha_uint8, mode="L").resize(
                (int(display_width), int(display_height)),
                Image.Resampling.BILINEAR,
            )
            color_img = Image.new("RGBA", (int(display_width), int(display_height)), (16, 212, 216, 0))
            color_img.putalpha(alpha_img)
            canvas.alpha_composite(color_img, (int(image_x), int(image_y)))
    ex1, ey1, ex2, ey2 = norm_to_xyxy(envelope_norm_xyxy, original_width, original_height)
    cx1, cy1, cx2, cy2 = norm_to_xyxy(core_norm_xyxy, original_width, original_height)
    x1, y1, x2, y2 = norm_to_xyxy(bbox_norm_xyxy, original_width, original_height)
    sx1 = int(image_x + round(x1 * (display_width / float(max(1, original_width)))))
    sy1 = int(image_y + round(y1 * (display_height / float(max(1, original_height)))))
    sx2 = int(image_x + round(x2 * (display_width / float(max(1, original_width)))))
    sy2 = int(image_y + round(y2 * (display_height / float(max(1, original_height)))))
    sex1 = int(image_x + round(ex1 * (display_width / float(max(1, original_width)))))
    sey1 = int(image_y + round(ey1 * (display_height / float(max(1, original_height)))))
    sex2 = int(image_x + round(ex2 * (display_width / float(max(1, original_width)))))
    sey2 = int(image_y + round(ey2 * (display_height / float(max(1, original_height)))))
    scx1 = int(image_x + round(cx1 * (display_width / float(max(1, original_width)))))
    scy1 = int(image_y + round(cy1 * (display_height / float(max(1, original_height)))))
    scx2 = int(image_x + round(cx2 * (display_width / float(max(1, original_width)))))
    scy2 = int(image_y + round(cy2 * (display_height / float(max(1, original_height)))))
    fill_rect = (sx1, sy1, sx2, sy2)
    if decoded is None:
        overlay_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_layer, "RGBA")
        overlay_draw.rectangle(fill_rect, fill=(16, 212, 216, 40))
        canvas.alpha_composite(overlay_layer)
    if (sex1, sey1, sex2, sey2) != fill_rect:
        draw.rectangle((sex1, sey1, sex2, sey2), outline=SUBJECT_ENVELOPE_COLOR, width=1)
    draw.rectangle(fill_rect, outline=SUBJECT_COLOR, width=2)
    if (scx1, scy1, scx2, scy2) != fill_rect:
        draw.rectangle((scx1, scy1, scx2, scy2), outline=SUBJECT_CORE_COLOR, width=2)
    if secondary_core_norm_xyxy is not None:
        qx1, qy1, qx2, qy2 = norm_to_xyxy(secondary_core_norm_xyxy, original_width, original_height)
        sqx1 = int(image_x + round(qx1 * (display_width / float(max(1, original_width)))))
        sqy1 = int(image_y + round(qy1 * (display_height / float(max(1, original_height)))))
        sqx2 = int(image_x + round(qx2 * (display_width / float(max(1, original_width)))))
        sqy2 = int(image_y + round(qy2 * (display_height / float(max(1, original_height)))))
        draw.rectangle((sqx1, sqy1, sqx2, sqy2), outline=SUBJECT_SECONDARY_COLOR, width=2)
    label = subject_overlay_legend_label(subject_overlay)
    if label:
        label_box(draw, font, sx1, max(image_y, sy1 - 20), label, SUBJECT_COLOR)


def target_ar_dir_name(target_ar: str) -> str:
    cleaned = str(target_ar or "unknown").strip()
    if not cleaned:
        return "unknown"
    if cleaned.upper() == "FREE":
        return "FREE"
    return cleaned.replace(":", "x").replace("/", "_")


def derive_default_teacher_jsonl(batch_jsonl: Path) -> Path:
    training_dir = batch_jsonl.parent
    artifacts_dir = training_dir.parent.parent
    name = training_dir.name
    suffix_map = [
        "_leftover_ignore_monotonic",
        "_leftover_keepneg_monotonic",
        "_leftover_softpos_monotonic",
        "_leftover_ignore",
        "_leftover_keepneg",
        "_leftover_softpos",
    ]
    base_run_tag = name
    is_monotonic = False
    for suffix in suffix_map:
        if name.endswith(suffix):
            base_run_tag = name[: -len(suffix)]
            is_monotonic = suffix.endswith("_monotonic")
            break
    candidate_paths = []
    if is_monotonic:
        candidate_paths.append(artifacts_dir / "teacher" / "scores" / f"teacher_scores_ar_{base_run_tag}_monotonic.jsonl")
        candidate_paths.append(artifacts_dir / "teacher" / "scores" / f"teacher_scores_ar_{base_run_tag}.jsonl")
    else:
        candidate_paths.append(artifacts_dir / "teacher" / "scores" / f"teacher_scores_ar_{base_run_tag}.jsonl")
        candidate_paths.append(artifacts_dir / "teacher" / "scores" / f"teacher_scores_ar_{base_run_tag}_monotonic.jsonl")
    for path in candidate_paths:
        if path.exists():
            return path
    return candidate_paths[0]


def round_opt(value: Any, digits: int = 6) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def attach_batch_fields(detail: Dict[str, Any], candidate_stub: Dict[str, Any], *, target_ar: str) -> Dict[str, Any]:
    out = copy.deepcopy(detail)
    if safe_dict(detail):
        out["teacher_snapshot_score_detail"] = copy.deepcopy(safe_dict(detail.get("score_detail")))
    score_detail = safe_dict(out.get("score_detail"))
    stub_score_targets = safe_dict(candidate_stub.get("score_targets"))
    stub_safety_penalty = safe_dict(candidate_stub.get("safety_penalty"))
    score_detail["score_prob"] = round_opt(
        first_nonempty(
            stub_score_targets.get("score_prob"),
            stub_score_targets.get("crop_utility_prob"),
            candidate_score(candidate_stub),
            score_detail.get("score_prob"),
        ),
        9,
    )
    score_detail["crop_utility_prob"] = round_opt(
        first_nonempty(
            stub_score_targets.get("crop_utility_prob"),
            stub_score_targets.get("score_prob"),
            score_detail.get("crop_utility_prob"),
            score_detail.get("score_prob"),
        ),
        9,
    )
    score_detail["score_rank_pct"] = round_opt(
        first_nonempty(
            stub_score_targets.get("rank_pct"),
            candidate_stub.get("score_rank_pct"),
            score_detail.get("score_rank_pct"),
        ),
        9,
    )
    score_detail["crop_utility_rank_pct"] = round_opt(
        first_nonempty(
            stub_score_targets.get("crop_utility_rank_pct"),
            candidate_stub.get("crop_utility_rank_pct"),
            candidate_stub.get("score_policy_rank_pct"),
            score_detail.get("crop_utility_rank_pct"),
            score_detail.get("score_policy_rank_pct"),
            score_detail.get("score_rank_pct"),
        ),
        9,
    )
    for src_key, dst_key in (
        ("score_raw_rank", "score_raw_rank"),
        ("score_raw_policy", "score_raw_policy"),
        ("score_raw_policy_base", "score_raw_policy_base"),
        ("score_raw_policy_safe", "score_raw_policy_safe"),
        ("crop_utility_raw", "crop_utility_raw"),
        ("score_raw_rank_macro", "score_raw_rank_macro"),
        ("score_raw_safety_penalty_total", "safety_penalty_total"),
        ("z_local", "z_local"),
        ("softmax_local", "softmax_local"),
    ):
        if stub_score_targets.get(src_key) is not None:
            score_detail[dst_key] = round_opt(stub_score_targets.get(src_key), 9)
    if stub_safety_penalty:
        score_detail["safety_penalty_total"] = round_opt(
            first_nonempty(
                stub_safety_penalty.get("total"),
                score_detail.get("safety_penalty_total"),
            ),
            9,
        )
        score_detail["safety_penalty_soft"] = round_opt(
            first_nonempty(
                stub_safety_penalty.get("soft_total"),
                score_detail.get("safety_penalty_soft"),
            ),
            9,
        )
        score_detail["safety_penalty_hard"] = round_opt(
            first_nonempty(
                stub_safety_penalty.get("hard_total"),
                score_detail.get("safety_penalty_hard"),
            ),
            9,
        )
    if score_detail.get("score_raw_policy_base") is not None and score_detail.get("score_raw_rank_macro") is not None:
        score_detail["area_log_prior"] = round_opt(
            safe_float(score_detail.get("score_raw_policy_base"), 0.0)
            - safe_float(score_detail.get("score_raw_rank_macro"), 0.0),
            9,
        )
    out["score_detail"] = score_detail
    if safe_dict(candidate_stub.get("macro_targets")):
        out["macro_scores"] = merge_dicts_prefer_override(
            safe_dict(out.get("macro_scores")),
            safe_dict(candidate_stub.get("macro_targets")),
        )
    if safe_dict(candidate_stub.get("macro_components")):
        out["macro_components"] = merge_dicts_prefer_override(
            safe_dict(out.get("macro_components")),
            safe_dict(candidate_stub.get("macro_components")),
        )
    if safe_dict(candidate_stub.get("composition_focus")):
        out["composition_focus"] = merge_dicts_prefer_override(
            safe_dict(out.get("composition_focus")),
            safe_dict(candidate_stub.get("composition_focus")),
        )
    if stub_safety_penalty:
        out["safety_penalty"] = merge_dicts_prefer_override(
            safe_dict(out.get("safety_penalty")),
            stub_safety_penalty,
        )
    if safe_dict(candidate_stub.get("checklist_labels")):
        out["checklist_labels"] = merge_dicts_prefer_override(
            safe_dict(out.get("checklist_labels")),
            safe_dict(candidate_stub.get("checklist_labels")),
        )
    if safe_dict(candidate_stub.get("checklist_scores")):
        out["checklist_scores"] = merge_dicts_prefer_override(
            safe_dict(out.get("checklist_scores")),
            safe_dict(candidate_stub.get("checklist_scores")),
        )
    out["candidate_id"] = str(candidate_stub.get("candidate_id", out.get("candidate_id", "")))
    out["bbox_norm_xyxy"] = list(candidate_stub.get("bbox_norm_xyxy", out.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])))
    out["label_type"] = str(candidate_stub.get("label_type", out.get("label_type", "")))
    out["target_ar"] = str(target_ar)
    out["training_bucket"] = str(candidate_stub.get("training_bucket", out.get("training_bucket", "")))
    out["positive_anchor_reason"] = str(candidate_stub.get("positive_anchor_reason", out.get("positive_anchor_reason", "")))
    out["monotonic_label_state"] = str(candidate_stub.get("monotonic_label_state", out.get("monotonic_label_state", "")))
    out["source"] = str(candidate_stub.get("source", out.get("source", "")))
    if str(candidate_stub.get("why_text_template", "")).strip():
        out["why_text_template"] = str(candidate_stub.get("why_text_template", "")).strip()
    if safe_list(candidate_stub.get("why_tags")):
        out["why_tags"] = [str(tag) for tag in safe_list(candidate_stub.get("why_tags"))]
    if safe_list(candidate_stub.get("reject_tags")):
        out["reject_tags"] = [str(tag) for tag in safe_list(candidate_stub.get("reject_tags"))]
    return out


def build_candidate_focus_detail(
    *,
    candidate_stub: Optional[Dict[str, Any]],
    image_id: str,
    target_ar: str,
    teacher_detail_index: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    if not candidate_stub:
        return None
    candidate_id = str(candidate_stub.get("candidate_id", "")).strip()
    raw_detail = safe_dict(safe_dict(safe_dict(teacher_detail_index.get(image_id, {})).get(str(target_ar), {})).get(candidate_id))
    detail = attach_batch_fields(raw_detail, candidate_stub, target_ar=str(target_ar))
    return detail


def build_gt_focus_detail(gt_ann: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not gt_ann:
        return None
    score_detail = safe_dict(gt_ann.get("score_detail"))
    score_detail["score_prob"] = round_opt(gt_ann.get("training_score_prob"), 9)
    score_detail["score_rank_pct"] = round_opt(gt_ann.get("training_score_rank_pct"), 9)
    score_detail["crop_utility_rank_pct"] = round_opt(
        first_nonempty(
            gt_ann.get("training_crop_utility_rank_pct"),
            gt_ann.get("training_score_rank_pct"),
        ),
        9,
    )
    return {
        "candidate_id": f"gaic_gt_{safe_int(gt_ann.get('id'), 0)}",
        "bbox": list(gt_ann.get("bbox", [0.0, 0.0, 1.0, 1.0])),
        "mos": round_opt(gt_ann.get("score"), 6),
        "score_detail": score_detail,
        "macro_scores": copy.deepcopy(safe_dict(gt_ann.get("macro_scores"))),
        "macro_components": copy.deepcopy(safe_dict(gt_ann.get("macro_components"))),
        "checklist_labels": copy.deepcopy(safe_dict(gt_ann.get("checklist_labels"))),
        "checklist_scores": copy.deepcopy(safe_dict(gt_ann.get("checklist_scores"))),
        "composition_focus": copy.deepcopy(safe_dict(gt_ann.get("composition_focus"))),
        "safety_penalty": copy.deepcopy(safe_dict(gt_ann.get("safety_penalty"))),
        "why_tags": [str(tag) for tag in safe_list(gt_ann.get("why_tags"))],
        "reject_tags": [str(tag) for tag in safe_list(gt_ann.get("reject_tags"))],
        "why_text_template": str(gt_ann.get("why_text_template", "")).strip(),
        "source": "gaic_gt",
        "target_ar": "FREE",
    }


def attach_gt_detail_fields(target_ann: Dict[str, Any], cache_row: Dict[str, Any]) -> None:
    for key in (
        "score_detail",
        "macro_scores",
        "macro_components",
        "checklist_labels",
        "checklist_scores",
        "composition_focus",
        "safety_penalty",
        "why_tags",
        "reject_tags",
        "why_text_template",
    ):
        value = cache_row.get(key)
        if value is not None:
            target_ann[key] = copy.deepcopy(value)


def outline_passes_for_rank(base_width: int, rank: int) -> List[Tuple[int, int]]:
    width = max(1, int(base_width))
    if int(rank) == 1:
        return [(3, width + 2), (0, width)]
    return [(0, width)]


def draw_overlay_groups(
    *,
    image_path: Path,
    title: str,
    groups: Sequence[Dict[str, Any]],
    subject_overlay: Optional[Dict[str, Any]] = None,
    out_path: Path,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
) -> None:
    with Image.open(image_path) as image_obj:
        base = image_obj.convert("RGBA")
        width = int(image_obj.width)
        height = int(image_obj.height)
    legend_entries: List[Tuple[str, str]] = []
    subject_label = subject_overlay_legend_label(subject_overlay)
    if subject_label:
        legend_entries.append((subject_label, SUBJECT_COLOR))
    for group in groups:
        group_name = str(group.get("name", "candidate"))
        color = str(group.get("color", "#111111"))
        for idx, item in enumerate(list(group.get("items", [])), start=1):
            legend_entries.append((resolve_overlay_label(group_name, item, idx), color))

    legend_columns = 3 if len(legend_entries) <= 18 else 4
    legend_rows = max(1, int(np.ceil(len(legend_entries) / float(legend_columns))))
    top_padding = 34 + legend_rows * 18 + 8

    canvas = Image.new("RGBA", (base.width, base.height + top_padding), TOP_PANEL_BG)
    canvas.paste(base, (0, top_padding))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = choose_font()
    title_text = f"{title} | {image_path.stem} | size={width}x{height}"
    label_box(draw, font, 10, 8, title_text, "#F5F5F5")
    render_subject_overlay(
        canvas,
        subject_overlay=subject_overlay,
        draw=draw,
        font=font,
        image_x=0,
        image_y=top_padding,
        display_width=base.width,
        display_height=base.height,
        original_width=width,
        original_height=height,
    )

    col_width = max(1, canvas.width // legend_columns)
    legend_y0 = 30
    for idx, (text, color) in enumerate(legend_entries):
        col_idx = idx % legend_columns
        row_idx = idx // legend_columns
        label_box(draw, font, 10 + col_idx * col_width, legend_y0 + row_idx * 18, text, color)

    line_width = max(2, int(round(min(width, height) / 250.0)))
    image_y_offset = top_padding
    group_label_offsets = {"gt": -18, "positive": 2, "negative": 18}
    for group in groups:
        group_name = str(group.get("name", "candidate"))
        color = str(group.get("color", "#111111"))
        box_key = str(group.get("box_key", "bbox"))
        items = list(group.get("items", []))
        for idx, item in enumerate(items, start=1):
            if box_key == "bbox_norm_xyxy":
                x1, y1, x2, y2 = norm_to_xyxy(item["bbox_norm_xyxy"], width, height)
            else:
                x1, y1, x2, y2 = xywh_to_xyxy(item["bbox"])
            for expand, draw_width in outline_passes_for_rank(line_width, idx):
                left = max(0, x1 - expand)
                top = max(image_y_offset, y1 + image_y_offset - expand)
                right = min(canvas.width - 1, x2 + expand)
                bottom = min(canvas.height - 1, y2 + image_y_offset + expand)
                draw.rectangle((left, top, right, bottom), outline=color, width=draw_width)
            label_y = min(
                canvas.height - 20,
                max(
                    image_y_offset,
                    y1 + image_y_offset + group_label_offsets.get(group_name, 0) + ((idx - 1) % 3) * 16,
                ),
            )
            label_box(draw, font, x1, label_y, box_overlay_label(group_name, item, idx), color)

    save_pil_visualization(canvas, out_path, jpg_quality=jpg_quality)


def format_metric(value: Any, digits: int = 3, default: str = "NA") -> str:
    parsed = round_opt(value, digits)
    if parsed is None:
        return default
    return f"{parsed:.{digits}f}"


def wrapped_lines(text: str, width: int) -> List[str]:
    content = str(text or "").strip()
    if not content:
        return []
    return textwrap.wrap(content, width=max(12, int(width)), break_long_words=False, break_on_hyphens=False)


def draw_progress_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: Optional[float],
    color: str,
    font: ImageFont.ImageFont,
) -> int:
    value_text = "NA" if value is None else f"{float(value):.3f}"
    draw.text((x, y), f"{label} {value_text}", fill="#F3F4F6", font=font)
    bar_x = x + 188
    bar_y = y + 4
    bar_w = max(120, width - 208)
    bar_h = 14
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=4, fill="#23252C")
    if value is not None:
        fill_w = int(round(clamp(float(value), 0.0, 1.0) * bar_w))
        draw.rounded_rectangle((bar_x, bar_y, bar_x + max(0, fill_w), bar_y + bar_h), radius=4, fill=color)
    return y + 30


def draw_section_text_rows(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    rows: Sequence[str],
    font: ImageFont.ImageFont,
    fill: str = "#D6D8DE",
) -> int:
    cur_y = y
    for row in rows:
        for line in wrapped_lines(row, max(18, width // 9)):
            draw.text((x, cur_y), line, fill=fill, font=font)
            cur_y += 20
    return cur_y


def build_focus_title(group_name: str, detail: Dict[str, Any], display_spec: Optional[Dict[str, Any]] = None) -> str:
    semantics = safe_dict(safe_dict(display_spec).get("semantics"))
    score_prob_label = "UtilityProb" if bool(semantics.get("single_scorer_semantics", False)) else "ScoreProb"
    raw_label = "UtilityRaw" if bool(semantics.get("single_scorer_semantics", False)) else "PolicySafe"
    score_detail = safe_dict(detail.get("score_detail"))
    score_prob = score_detail.get("crop_utility_prob", score_detail.get("score_prob"))
    raw_value = score_detail.get("crop_utility_raw", score_detail.get("score_raw_policy_safe"))
    if group_name == "gt":
        title = f"GT1 | MOS={format_metric(detail.get('mos'), 2)} | {score_prob_label}={format_metric(score_prob)}"
        if raw_value is not None:
            title += f" | {raw_label}={format_metric(raw_value)}"
        return title
    label_type = str(detail.get("label_type", "")).strip()
    suffix = f" | {label_type}" if label_type else ""
    title = f"{group_name.upper()}1 | {score_prob_label}={format_metric(score_prob)}"
    if raw_value is not None:
        title += f" | {raw_label}={format_metric(raw_value)}"
    return title + suffix


def clamp_text(text: str, max_chars: int) -> str:
    content = str(text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[: max(0, max_chars - 1)] + "..."


def nonzero_penalty_rows(detail: Dict[str, Any]) -> List[str]:
    penalty = safe_dict(detail.get("safety_penalty"))
    items: List[Tuple[str, float]] = []
    for prefix, bucket in (("soft", safe_dict(penalty.get("soft_components"))), ("hard", safe_dict(penalty.get("hard_components")))):
        for key, value in bucket.items():
            parsed = round_opt(value, 3)
            if parsed is None or parsed <= 0.0:
                continue
            items.append((f"{prefix}:{key}", float(parsed)))
    items.sort(key=lambda item: (-item[1], item[0]))
    if not items:
        return ["PenaltyDrivers=none"]
    rows = []
    for name, value in items[:4]:
        prefix, key = name.split(":", 1)
        if prefix == "soft":
            rows.append(f"PenaltySoft:{key}={value:.3f}")
        else:
            rows.append(f"PenaltyHard:{key}={value:.3f}")
    return rows


def normalize_anchor_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"thirds", "third", "rule_of_thirds"}:
        return "third"
    if text in {"phi", "phi_grid"}:
        return "phi"
    if text in {"center", "centered", "center_comp"}:
        return "center"
    return "na"


def anchor_strength_from_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    if not text or text in {"na", "none", "n/a"} or text.endswith("_na"):
        return "na"
    if text.endswith("_strong"):
        return "strong"
    if text.endswith("_weak"):
        return "weak"
    return text


def derive_place_display(
    *,
    composition_focus: Dict[str, Any],
    checklist_labels: Dict[str, Any],
) -> Tuple[str, str]:
    bundle_best = normalize_anchor_name(composition_focus.get("placement_family_best"))
    fallback_best = normalize_anchor_name(composition_focus.get("dominant_anchor"))
    if bundle_best != "na":
        return bundle_best, "bundle"
    if fallback_best != "na":
        return fallback_best, "distance"
    for anchor_name, key in (("third", "third_dist"), ("phi", "phi_dist"), ("center", "center_dist")):
        if anchor_strength_from_label(checklist_labels.get(key)) != "na":
            return anchor_name, "label"
    return "na", "none"


def penalty_component_bar_rows(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    penalty = safe_dict(detail.get("safety_penalty"))
    rows: List[Dict[str, Any]] = []
    for prefix, bucket in (("PenaltySoft", safe_dict(penalty.get("soft_components"))), ("PenaltyHard", safe_dict(penalty.get("hard_components")))):
        for key, value in bucket.items():
            parsed = round_opt(value, 6)
            if parsed is None or parsed <= 0.0:
                continue
            rows.append(
                {
                    "key": f"{prefix}:{key}",
                    "label": clamp_text(f"{prefix}:{key}", 24),
                    "value": float(parsed),
                    "kind": "positive",
                    "cap": 4.0,
                    "color": SCORE_GROUP_COLORS["penalty"],
                }
            )
    rows.sort(key=lambda row: (-safe_float(row.get("value"), 0.0), str(row.get("key", ""))))
    return rows[:6]


def metric_bar_color(key: str, accent: str) -> str:
    if key.startswith("A_"):
        return SCORE_GROUP_COLORS["A"]
    if key.startswith("S_"):
        return SCORE_GROUP_COLORS["S"]
    if key.startswith("C_"):
        return SCORE_GROUP_COLORS["C"]
    if key.startswith("T_"):
        return SCORE_GROUP_COLORS["T"]
    if "penalty" in key:
        return SCORE_GROUP_COLORS["penalty"]
    if key in {"crop_utility_prob", "crop_utility_raw", "score_raw_policy_base", "score_raw_rank", "score_raw_rank_macro", "area_log_prior"}:
        return accent
    return SCORE_GROUP_COLORS["other"]


def metric_bar_spec(key: str) -> Tuple[str, float]:
    if key in {
        "crop_utility_prob",
        "score_raw_rank",
        "score_raw_rank_macro",
        "A_macro",
        "S_macro",
        "C_macro",
        "T_macro",
        "A_aesthetic",
        "A_align",
        "S_cov",
        "S_scale",
        "S_support_structure",
        "S_border",
        "S_softcut_quality",
        "C_place",
        "C_comp",
        "C_headroom",
        "C_lookroom",
        "C_horizon_y",
        "C_sym",
        "C_context",
        "C_copyspace",
        "T_teacher",
    }:
        return ("unit", 1.0)
    if key == "area_log_prior":
        return ("signed", 0.75)
    if key in {"safety_penalty_total", "safety_penalty_soft", "safety_penalty_hard"}:
        return ("positive", 4.0)
    return ("signed", 4.0)


def metric_display_label(key: str, *, single_scorer: bool) -> str:
    if key == "crop_utility_prob":
        return "UtilityProb" if single_scorer else "ScoreProb"
    if key == "crop_utility_raw":
        return "UtilityRaw" if single_scorer else "PolicySafe"
    labels = {
        "score_raw_policy_base": "PolicyBase",
        "score_raw_rank": "RankRaw",
        "score_raw_rank_macro": "RankMacro",
        "area_log_prior": "AreaPrior",
        "safety_penalty_total": "PenaltyTot",
        "safety_penalty_soft": "PenaltySoft",
        "safety_penalty_hard": "PenaltyHard",
        "A_aesthetic": "A_aes",
        "A_align": "A_align",
        "S_cov": "S_cov",
        "S_scale": "S_scale",
        "S_support_structure": "S_struct",
        "S_border": "S_border",
        "S_softcut_quality": "S_softcut",
        "C_place": "C_place",
        "C_comp": "C_comp",
        "C_headroom": "C_head",
        "C_lookroom": "C_look",
        "C_horizon_y": "C_horizon",
        "C_sym": "C_sym",
        "C_context": "C_ctx",
        "C_copyspace": "C_copy",
        "T_teacher": "T_teacher",
    }
    return labels.get(key, key)


def maybe_append_metric_row(
    rows: List[Dict[str, Any]],
    *,
    key: str,
    value: Any,
    single_scorer: bool,
    accent: str,
) -> None:
    rounded = round_opt(value, 6)
    if rounded is None:
        return
    kind, cap = metric_bar_spec(key)
    rows.append(
        {
            "key": key,
            "label": metric_display_label(key, single_scorer=single_scorer),
            "value": float(rounded),
            "kind": kind,
            "cap": float(cap),
            "color": metric_bar_color(key, accent),
        }
    )


def build_focus_section_rows(detail: Dict[str, Any], display_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    score_detail = safe_dict(detail.get("score_detail"))
    macro_scores = safe_dict(detail.get("macro_scores"))
    macro_components = safe_dict(detail.get("macro_components"))
    checklist_labels = safe_dict(detail.get("checklist_labels"))
    checklist_scores = safe_dict(detail.get("checklist_scores"))
    composition_focus = safe_dict(detail.get("composition_focus"))
    why_tags = [str(tag) for tag in safe_list(detail.get("why_tags"))]
    reject_tags = [str(tag) for tag in safe_list(detail.get("reject_tags"))]
    active_macros = safe_dict(safe_dict(display_spec).get("macros"))
    active_components = safe_dict(safe_dict(display_spec).get("components"))
    semantics = safe_dict(safe_dict(display_spec).get("semantics"))
    single_scorer = bool(semantics.get("single_scorer_semantics", False))

    def macro_on(key: str) -> bool:
        return bool(active_macros.get(key, True))

    def comp_on(key: str) -> bool:
        return bool(active_components.get(key, True))
    pct_label = "utility_pct" if single_scorer else "rank_pct"
    pct_value = first_nonempty(
        score_detail.get("crop_utility_rank_pct") if single_scorer else score_detail.get("score_rank_pct"),
        detail.get("crop_utility_rank_pct") if single_scorer else detail.get("score_rank_pct"),
        score_detail.get("score_policy_rank_pct") if single_scorer else None,
        detail.get("score_policy_rank_pct") if single_scorer else None,
        score_detail.get("score_rank_pct"),
        detail.get("score_rank_pct"),
    )
    stats_rows = [
        f"candidate={detail.get('candidate_id', 'NA')} target_ar={detail.get('target_ar', 'NA')}",
        (
            f"{pct_label}={format_metric(pct_value)} "
            f"bucket={detail.get('training_bucket', 'NA') or 'NA'}"
        ),
    ]
    if "admissible" in detail:
        stats_rows.append(f"admissible={'yes' if bool(detail.get('admissible')) else 'no'}")
    if str(detail.get("source", "")).strip():
        stats_rows.append(f"source={detail['source']}")
    score_bar_rows: List[Dict[str, Any]] = []
    maybe_append_metric_row(
        score_bar_rows,
        key="crop_utility_prob",
        value=score_detail.get("crop_utility_prob", score_detail.get("score_prob")),
        single_scorer=single_scorer,
        accent=SCORE_GROUP_COLORS["utility"],
    )
    maybe_append_metric_row(
        score_bar_rows,
        key="crop_utility_raw",
        value=score_detail.get("crop_utility_raw", score_detail.get("score_raw_policy_safe")),
        single_scorer=single_scorer,
        accent=SCORE_GROUP_COLORS["utility"],
    )
    policy_base_value = maybe_float(score_detail.get("score_raw_policy_base"))
    utility_raw_value = maybe_float(score_detail.get("crop_utility_raw", score_detail.get("score_raw_policy_safe")))
    if (
        policy_base_value is not None
        and utility_raw_value is not None
        and abs(policy_base_value - utility_raw_value) > 1e-6
    ):
        maybe_append_metric_row(
            score_bar_rows,
            key="score_raw_policy_base",
            value=policy_base_value,
            single_scorer=single_scorer,
            accent=SCORE_GROUP_COLORS["utility"],
        )
    rank_raw_value = maybe_float(score_detail.get("score_raw_rank"))
    rank_macro_value = maybe_float(score_detail.get("score_raw_rank_macro"))
    if not single_scorer:
        if rank_raw_value is None:
            rank_raw_value = rank_macro_value
        maybe_append_metric_row(
            score_bar_rows,
            key="score_raw_rank",
            value=rank_raw_value,
            single_scorer=single_scorer,
            accent=GT_COLOR,
        )
        if (
            rank_macro_value is not None
            and (rank_raw_value is None or abs(rank_macro_value - rank_raw_value) > 1e-6)
        ):
            maybe_append_metric_row(
                score_bar_rows,
                key="score_raw_rank_macro",
                value=rank_macro_value,
                single_scorer=single_scorer,
                accent=GT_COLOR,
            )
    if comp_on("area_log_prior"):
        maybe_append_metric_row(
            score_bar_rows,
            key="area_log_prior",
            value=score_detail.get("area_log_prior"),
            single_scorer=single_scorer,
            accent="#C3A13B",
        )
    for key in FOCUS_MACRO_ORDER:
        if macro_on(key):
            maybe_append_metric_row(
                score_bar_rows,
                key=key,
                value=macro_scores.get(key),
                single_scorer=single_scorer,
                accent=metric_bar_color(key, SCORE_GROUP_COLORS["other"]),
            )
    for key in FOCUS_COMPONENT_ORDER:
        if comp_on(key):
            maybe_append_metric_row(
                score_bar_rows,
                key=key,
                value=macro_components.get(key),
                single_scorer=single_scorer,
                accent=metric_bar_color(key, SCORE_GROUP_COLORS["other"]),
            )
    if comp_on("safety_bundle"):
        maybe_append_metric_row(
            score_bar_rows,
            key="safety_penalty_total",
            value=score_detail.get("safety_penalty_total"),
            single_scorer=single_scorer,
            accent=SCORE_GROUP_COLORS["penalty"],
        )
        maybe_append_metric_row(
            score_bar_rows,
            key="safety_penalty_soft",
            value=score_detail.get("safety_penalty_soft"),
            single_scorer=single_scorer,
            accent=SCORE_GROUP_COLORS["penalty"],
        )
        maybe_append_metric_row(
            score_bar_rows,
            key="safety_penalty_hard",
            value=score_detail.get("safety_penalty_hard"),
            single_scorer=single_scorer,
            accent=SCORE_GROUP_COLORS["penalty"],
        )
        score_bar_rows.extend(penalty_component_bar_rows(detail))

    composition_rows: List[str] = []
    if comp_on("C_place") or comp_on("C_comp"):
        place_best, place_source = derive_place_display(
            composition_focus=composition_focus,
            checklist_labels=checklist_labels,
        )
        source_suffix = "" if place_source == "bundle" else (" (dist)" if place_source == "distance" else (" (label)" if place_source == "label" else ""))
        composition_rows.append(
            f"best_anchor={place_best}{source_suffix} "
            f"margin={format_metric(first_nonempty(composition_focus.get('placement_family_margin'), macro_components.get('C_place_margin')))} "
            f"smax={format_metric(first_nonempty(composition_focus.get('placement_score'), macro_components.get('C_place')))}"
        )
        composition_rows.append(
            f"dist_third={format_metric(checklist_scores.get('third_dist'))} "
            f"dist_phi={format_metric(checklist_scores.get('phi_dist'))} "
            f"dist_center={format_metric(checklist_scores.get('center_dist'))} "
            f"sym={format_metric(checklist_scores.get('symmetry_score'))}"
        )
        composition_rows.append(
            f"label_third={anchor_strength_from_label(checklist_labels.get('third_dist'))} "
            f"label_phi={anchor_strength_from_label(checklist_labels.get('phi_dist'))} "
            f"label_center={anchor_strength_from_label(checklist_labels.get('center_dist'))}"
        )
        if any(
            composition_focus.get(key) is not None
            for key in ("placement_reward_third", "placement_reward_phi", "placement_reward_center")
        ):
            composition_rows.append(
                f"reward_third={format_metric(composition_focus.get('placement_reward_third'))} "
                f"reward_phi={format_metric(composition_focus.get('placement_reward_phi'))} "
                f"reward_center={format_metric(composition_focus.get('placement_reward_center'))}"
            )
    label_pairs: List[Tuple[str, str]] = []
    if comp_on("S_cov") or comp_on("S_support_structure"):
        label_pairs.append(("subject", checklist_labels.get("subject_coverage", "na")))
    if comp_on("S_scale"):
        label_pairs.append(("scale", checklist_labels.get("subject_scale", "na")))
    if comp_on("C_context"):
        label_pairs.append(("context", checklist_labels.get("context", "na")))
    if comp_on("C_copyspace"):
        label_pairs.append(("copyspace", checklist_labels.get("copyspace", "na")))
    if comp_on("C_headroom"):
        label_pairs.append(("head", checklist_labels.get("headroom", "na")))
    if comp_on("C_lookroom"):
        label_pairs.append(("look", checklist_labels.get("lookroom", "na")))
    label_pairs.append(("face", checklist_labels.get("face_cut", "na")))
    label_pairs.append(("joint", checklist_labels.get("joint_cut", "na")))
    if comp_on("C_place") or comp_on("C_comp"):
        label_pairs.extend(
            [
                ("third", checklist_labels.get("third_dist", "na")),
                ("phi", checklist_labels.get("phi_dist", "na")),
                ("center", checklist_labels.get("center_dist", "na")),
            ]
        )
    if comp_on("T_teacher"):
        label_pairs.append(("teacher", checklist_labels.get("teacher_consensus", "na")))
    label_pairs.append(("ar", checklist_labels.get("ar", "na")))
    if comp_on("C_horizon_y"):
        label_pairs.append(("horizon", first_nonempty(checklist_labels.get("horizon"), checklist_labels.get("horizon_state"), "na")))
    tag_row = ""
    if why_tags:
        tag_row = "why=" + ", ".join(why_tags[:5])
    if reject_tags:
        tag_row = (tag_row + " | " if tag_row else "") + "reject=" + ",".join(reject_tags[:4])
    if not tag_row:
        tag_row = "why=none"
    penalty_summary_rows = [
        (
            f"PenaltyTotal={format_metric(score_detail.get('safety_penalty_total'))} "
            f"PenaltySoft={format_metric(score_detail.get('safety_penalty_soft'))} "
            f"PenaltyHard={format_metric(score_detail.get('safety_penalty_hard'))}"
        )
    ]
    return {
        "stats_rows": stats_rows,
        "score_bar_rows": score_bar_rows,
        "composition_rows": composition_rows,
        "label_pairs": label_pairs,
        "penalty_summary_rows": penalty_summary_rows,
        "penalty_rows": nonzero_penalty_rows(detail),
        "tag_row": tag_row,
        "why_text": str(detail.get("why_text_template", "")).strip(),
    }


def draw_metric_block(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
    rows: Sequence[str],
    fill: str = "#D6D8DE",
    max_lines: Optional[int] = None,
) -> int:
    cur_y = y
    line_budget = max_lines if max_lines is not None else 10**9
    used = 0
    for row in rows:
        for line in wrapped_lines(clamp_text(row, 180), max(18, width // 9)):
            if used >= line_budget:
                draw.text((x, cur_y), "...", fill=fill, font=font)
                return cur_y + 18
            draw.text((x, cur_y), line, fill=fill, font=font)
            cur_y += 20
            used += 1
    return cur_y


def estimate_wrapped_line_count(rows: Sequence[str], width: int) -> int:
    count = 0
    for row in rows:
        count += max(1, len(wrapped_lines(clamp_text(row, 180), max(18, width // 9))))
    return count


def estimate_focus_section_height(
    content: Dict[str, Any],
    *,
    width: int,
    show_why_text: bool,
) -> int:
    left_w = max(260, int(width * 0.56) - 18)
    right_w = max(220, width - left_w - 30)
    stats_height = estimate_wrapped_line_count(content["stats_rows"], left_w) * 20
    score_rows_height = len(content["score_bar_rows"]) * 28
    left_height = 36 + stats_height + 12 + score_rows_height + 20
    right_height = 36
    right_height += estimate_wrapped_line_count(content["composition_rows"], right_w) * 20
    right_height += 10
    chip_rows = max(1, int(np.ceil(max(1, len(content["label_pairs"])) / 2.0)))
    right_height += chip_rows * 58
    right_height += 12
    right_height += estimate_wrapped_line_count(list(content["penalty_rows"]) + [content["tag_row"]], right_w) * 20
    if show_why_text and content["why_text"]:
        right_height += 12 + estimate_wrapped_line_count([f"why_text={content['why_text']}"], right_w) * 20
    return max(360, max(left_height, right_height) + 18)


def draw_metric_bar_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: Optional[float],
    kind: str,
    cap: float,
    color: str,
    font: ImageFont.ImageFont,
) -> int:
    label_text = clamp_text(label, 18)
    value_text = "NA" if value is None else f"{float(value):.3f}"
    value_bbox = draw.textbbox((0, 0), value_text, font=font)
    value_w = value_bbox[2] - value_bbox[0]
    draw.text((x, y), label_text, fill="#F3F4F6", font=font)
    draw.text((x + width - value_w, y), value_text, fill="#D6D8DE", font=font)
    bar_y = y + 22
    bar_h = 10
    bar_x = x
    bar_w = width
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=4, fill="#23252C")
    if value is not None:
        if kind == "signed":
            center_x = bar_x + bar_w // 2
            draw.line((center_x, bar_y - 1, center_x, bar_y + bar_h + 1), fill="#5A6270", width=1)
            norm_value = clamp(float(value) / max(1e-6, float(cap)), -1.0, 1.0)
            if norm_value >= 0.0:
                fill_w = int(round((bar_w / 2.0) * norm_value))
                if fill_w > 0:
                    draw.rounded_rectangle((center_x, bar_y, center_x + fill_w, bar_y + bar_h), radius=4, fill=color)
            else:
                fill_w = int(round((bar_w / 2.0) * abs(norm_value)))
                if fill_w > 0:
                    draw.rounded_rectangle((center_x - fill_w, bar_y, center_x, bar_y + bar_h), radius=4, fill=color)
        elif kind == "positive":
            fill_w = int(round(clamp(float(value) / max(1e-6, float(cap)), 0.0, 1.0) * bar_w))
            if fill_w > 0:
                draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=4, fill=color)
        else:
            fill_w = int(round(clamp(float(value), 0.0, 1.0) * bar_w))
            if fill_w > 0:
                draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=4, fill=color)
    return y + 28


def render_focus_section(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    title: str,
    accent: str,
    detail: Optional[Dict[str, Any]],
    font: ImageFont.ImageFont,
    show_why_text: bool,
    display_spec: Optional[Dict[str, Any]] = None,
) -> int:
    content = build_focus_section_rows(detail or {}, display_spec=display_spec) if detail else None
    section_height = estimate_focus_section_height(content, width=width, show_why_text=show_why_text) if content else 360
    draw.rounded_rectangle((x, y, x + width, y + section_height), radius=12, fill="#111318", outline=accent, width=2)
    if not detail:
        label_box(draw, font, x + 10, y + 10, title + " | unavailable", accent)
        return y + section_height + 12
    label_box(draw, font, x + 10, y + 10, title, accent)
    assert content is not None
    left_x = x + 12
    left_w = max(280, int(width * 0.56) - 18)
    right_x = left_x + left_w + 18
    right_w = width - (right_x - x) - 12
    cur_y = y + 36
    cur_y = draw_metric_block(draw, x=left_x, y=cur_y, width=left_w, rows=content["stats_rows"], font=font, max_lines=5)
    cur_y += 6
    for row in content["score_bar_rows"]:
        cur_y = draw_metric_bar_row(
            draw,
            x=left_x,
            y=cur_y,
            width=left_w,
            label=str(row.get("label", "")),
            value=maybe_float(row.get("value")),
            kind=str(row.get("kind", "unit")),
            cap=safe_float(row.get("cap"), 1.0),
            color=str(row.get("color", accent)),
            font=font,
        )

    right_y = y + 36
    right_y = draw_metric_block(
        draw,
        x=right_x,
        y=right_y,
        width=right_w,
        rows=content["composition_rows"],
        font=font,
        max_lines=6,
    )
    right_y += 10
    right_y = draw_label_chip_grid(
        draw,
        x=right_x,
        y=right_y,
        width=right_w,
        pairs=content["label_pairs"],
        font=font,
        columns=2,
    )
    right_y += 12
    right_y = draw_metric_block(
        draw,
        x=right_x,
        y=right_y,
        width=right_w,
        rows=list(content["penalty_summary_rows"]) + list(content["penalty_rows"]) + [content["tag_row"]],
        font=font,
        max_lines=10,
    )
    why_text = content["why_text"]
    if show_why_text and why_text:
        right_y = draw_metric_block(
            draw,
            x=right_x,
            y=min(right_y + 8, y + section_height - 96),
            width=right_w,
            rows=[f"why_text={why_text}"],
            font=font,
            fill="#B9BDC6",
            max_lines=4,
        )
    return y + section_height + 12


def draw_checklist_debug_panel(
    *,
    image_path: Path,
    title: str,
    groups: Sequence[Dict[str, Any]],
    subject_overlay: Optional[Dict[str, Any]] = None,
    out_path: Path,
    gt_detail: Optional[Dict[str, Any]],
    positive_detail: Optional[Dict[str, Any]],
    negative_detail: Optional[Dict[str, Any]],
    show_why_text: bool,
    display_spec: Optional[Dict[str, Any]] = None,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
) -> None:
    with Image.open(image_path) as image_obj:
        base = image_obj.convert("RGBA")
        width = int(image_obj.width)
        height = int(image_obj.height)
    legend_entries: List[Tuple[str, str]] = []
    subject_label = subject_overlay_legend_label(subject_overlay)
    if subject_label:
        legend_entries.append((subject_label, SUBJECT_COLOR))
    for group in groups:
        group_name = str(group.get("name", "candidate"))
        color = str(group.get("color", "#111111"))
        for idx, item in enumerate(list(group.get("items", [])), start=1):
            legend_entries.append((resolve_overlay_label(group_name, item, idx), color))

    canvas_width = 2520
    top_bar_height = 186
    outer_margin = 20
    left_panel_width = 1360
    sidebar_width = canvas_width - left_panel_width - outer_margin * 3
    panel_x = outer_margin + left_panel_width + outer_margin
    section_width = sidebar_width - 18
    focus_sections: List[Tuple[str, str, Optional[Dict[str, Any]]]] = [
        (build_focus_title("gt", gt_detail or {}, display_spec=display_spec), GT_COLOR, gt_detail),
        (build_focus_title("positive", positive_detail or {}, display_spec=display_spec), POSITIVE_COLOR, positive_detail),
    ]
    if negative_detail is not None:
        focus_sections.append(
            (build_focus_title("negative", negative_detail or {}, display_spec=display_spec), NEGATIVE_COLOR, negative_detail)
        )
    section_heights = [
        estimate_focus_section_height(
            build_focus_section_rows(detail or {}, display_spec=display_spec) if detail else {"stats_rows": [], "score_bar_rows": [], "composition_rows": [], "label_pairs": [], "penalty_rows": [], "tag_row": "why=none", "why_text": ""},
            width=section_width,
            show_why_text=show_why_text,
        )
        if detail
        else 360
        for _, _, detail in focus_sections
    ]
    sidebar_content_height = 14 + sum(height + 12 for height in section_heights)
    min_canvas_height = 1660 if show_why_text else 1540
    canvas_height = max(min_canvas_height, top_bar_height + sidebar_content_height + outer_margin)
    left_panel_height = canvas_height - top_bar_height - outer_margin * 2
    image_box_w = left_panel_width - 16
    image_box_h = left_panel_height - 16
    scale = min(image_box_w / float(max(1, base.width)), image_box_h / float(max(1, base.height)))
    disp_w = max(1, int(round(base.width * scale)))
    disp_h = max(1, int(round(base.height * scale)))
    disp_x = outer_margin + (left_panel_width - disp_w) // 2
    disp_y = top_bar_height + outer_margin + (left_panel_height - disp_h) // 2
    scaled_base = base.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (canvas_width, canvas_height), TOP_PANEL_BG)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle(
        (outer_margin, top_bar_height, outer_margin + left_panel_width, top_bar_height + left_panel_height),
        radius=14,
        fill="#0A0C10",
        outline="#242934",
        width=2,
    )
    canvas.paste(scaled_base, (disp_x, disp_y))
    font = choose_font(20)
    title_text = f"{title} | {image_path.stem} | size={width}x{height}"
    label_box(draw, font, 10, 8, title_text, "#F5F5F5")
    render_subject_overlay(
        canvas,
        subject_overlay=subject_overlay,
        draw=draw,
        font=font,
        image_x=disp_x,
        image_y=disp_y,
        display_width=disp_w,
        display_height=disp_h,
        original_width=width,
        original_height=height,
    )

    legend_columns = 4 if len(legend_entries) <= 16 else 5
    col_width = max(1, (canvas_width - 24) // legend_columns)
    legend_y0 = 36
    for idx, (text, color) in enumerate(legend_entries):
        col_idx = idx % legend_columns
        row_idx = idx // legend_columns
        label_box(draw, font, 12 + col_idx * col_width, legend_y0 + row_idx * 22, text, color)

    line_width = max(2, int(round(min(disp_w, disp_h) / 250.0)))
    group_label_offsets = {"gt": -18, "positive": 2, "negative": 18}
    for group in groups:
        group_name = str(group.get("name", "candidate"))
        color = str(group.get("color", "#111111"))
        box_key = str(group.get("box_key", "bbox"))
        items = list(group.get("items", []))
        for idx, item in enumerate(items, start=1):
            if box_key == "bbox_norm_xyxy":
                x1, y1, x2, y2 = norm_to_xyxy(item["bbox_norm_xyxy"], width, height)
            else:
                x1, y1, x2, y2 = xywh_to_xyxy(item["bbox"])
            x1 = disp_x + int(round(x1 * scale))
            y1 = disp_y + int(round(y1 * scale))
            x2 = disp_x + int(round(x2 * scale))
            y2 = disp_y + int(round(y2 * scale))
            for expand, draw_width in outline_passes_for_rank(line_width, idx):
                left = max(0, x1 - expand)
                top = max(disp_y, y1 - expand)
                right = min(outer_margin + left_panel_width - 1, x2 + expand)
                bottom = min(top_bar_height + left_panel_height - 1, y2 + expand)
                draw.rectangle((left, top, right, bottom), outline=color, width=draw_width)
            label_y = min(
                top_bar_height + left_panel_height - 20,
                max(
                    disp_y,
                    y1 + group_label_offsets.get(group_name, 0) + ((idx - 1) % 3) * 16,
                ),
            )
            label = box_overlay_label(group_name, item, idx)
            label_box(draw, font, x1, label_y, label, color)

    draw.rounded_rectangle((panel_x - 6, top_bar_height, canvas.width - outer_margin, canvas.height - outer_margin), radius=14, fill="#0D0F13", outline="#2A2E39", width=2)
    section_y = top_bar_height + 14
    for section_title, accent, detail in focus_sections:
        section_y = render_focus_section(
            draw,
            x=panel_x,
            y=section_y,
            width=section_width,
            title=section_title,
            accent=accent,
            detail=detail,
            font=font,
            show_why_text=show_why_text,
            display_spec=display_spec,
        )
    save_pil_visualization(canvas, out_path, jpg_quality=jpg_quality)


def save_distribution_plot(
    *,
    image_id: str,
    mode_name: str,
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    out_path: Path,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    bins = 20
    score_sets = [
        ("positive", positive_scores, POSITIVE_COLOR),
        ("negative", negative_scores, NEGATIVE_COLOR),
    ]
    for ax, (label, scores, color) in zip(axes, score_sets):
        arr = np.asarray(scores, dtype=float)
        if arr.size > 0:
            ax.hist(arr, bins=bins, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(float(arr.mean()), color="black", linestyle="--", linewidth=1.0, label=f"mean={arr.mean():.3f}")
            ax.axvline(float(np.median(arr)), color="#34495E", linestyle=":", linewidth=1.5, label=f"median={np.median(arr):.3f}")
            ax.legend(loc="upper left", fontsize=9)
        ax.set_title(f"{label} score distribution (n={arr.size})")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
    xmax = max(
        1.0,
        max((max(positive_scores) if positive_scores else 0.0), (max(negative_scores) if negative_scores else 0.0)) * 1.05,
    )
    for ax in axes:
        ax.set_xlim(0.0, xmax)
    fig.suptitle(f"{image_id} | {mode_name} | candidate score distributions", fontsize=14)
    save_matplotlib_figure(fig, out_path, jpg_quality=jpg_quality, dpi=150)
    plt.close(fig)


def save_subject_mode_summary_plot(
    *,
    positive_by_mode: Dict[int, List[float]],
    negative_by_mode: Dict[int, List[float]],
    mode_names: Dict[int, str],
    out_path: Path,
    title_suffix: str,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
) -> None:
    mode_ids = sorted(set(positive_by_mode) | set(negative_by_mode))
    labels = [f"{mode_id}:{mode_names.get(mode_id, 'unknown')}" for mode_id in mode_ids]
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    pos_data = [positive_by_mode.get(mode_id, []) or [0.0] for mode_id in mode_ids]
    neg_data = [negative_by_mode.get(mode_id, []) or [0.0] for mode_id in mode_ids]
    base_positions = np.arange(1, len(mode_ids) + 1, dtype=float)
    pos_positions = base_positions - 0.17
    neg_positions = base_positions + 0.17

    pos_parts = ax.violinplot(pos_data, positions=pos_positions, showmeans=True, showmedians=True, widths=0.28)
    neg_parts = ax.violinplot(neg_data, positions=neg_positions, showmeans=True, showmedians=True, widths=0.28)

    for parts, color in ((pos_parts, POSITIVE_COLOR), (neg_parts, NEGATIVE_COLOR)):
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            part = parts.get(key)
            if part is not None:
                part.set_edgecolor("#2C3E50")
                part.set_linewidth(1.0)

    ax.set_xticks(base_positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_title(f"Positive/negative score distributions by subject_mode_id ({title_suffix})")
    ax.grid(axis="y", alpha=0.2)
    ax.scatter([], [], color=POSITIVE_COLOR, label="positive")
    ax.scatter([], [], color=NEGATIVE_COLOR, label="negative")
    ax.legend(loc="upper right")

    for idx, mode_id in enumerate(mode_ids):
        pos_scores = positive_by_mode.get(mode_id, [])
        neg_scores = negative_by_mode.get(mode_id, [])
        ymax = max(pos_scores + neg_scores) if (pos_scores or neg_scores) else 0.0
        ax.text(
            base_positions[idx],
            ymax + 0.03,
            f"P n={len(pos_scores)} / N n={len(neg_scores)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.suptitle(f"Subject-mode score distributions ({title_suffix})", fontsize=14)
    save_matplotlib_figure(fig, out_path, jpg_quality=jpg_quality, dpi=150)
    plt.close(fig)


def summarize_scores(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 6),
        "p50": round(float(np.median(arr)), 6),
        "mean": round(float(arr.mean()), 6),
        "max": round(float(arr.max()), 6),
    }


def collect_mode_score_distributions(
    image_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    positive_by_mode: Dict[int, List[float]] = defaultdict(list)
    negative_by_mode: Dict[int, List[float]] = defaultdict(list)
    for row in image_rows:
        mode_id = safe_int(row.get("subject_mode_id"), -1)
        positive_by_mode[mode_id].extend(candidate_score(cand) for cand in row.get("positive_candidates", []))
        negative_by_mode[mode_id].extend(candidate_score(cand) for cand in row.get("negative_candidates", []))
    return positive_by_mode, negative_by_mode


def sort_target_ars(target_ars: Iterable[str]) -> List[str]:
    order = {"FREE": 0, "1:1": 1, "9:16": 2, "16:9": 3, "3:4": 4, "4:3": 5}
    unique = {str(target_ar) for target_ar in target_ars if str(target_ar).strip()}
    return sorted(unique, key=lambda value: (order.get(value, 999), value))


def dedupe_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        candidate_id = str(cand.get("candidate_id", "")).strip()
        if candidate_id and candidate_id in unique:
            continue
        key = candidate_id or json.dumps(cand.get("bbox", []), sort_keys=True)
        unique[key] = cand
    return list(unique.values())


def select_balanced_images(
    image_rows: Sequence[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_mode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        by_mode[safe_int(row.get("subject_mode_id"), -1)].append(row)
    for rows in by_mode.values():
        rows.sort(key=lambda row: str(row.get("image_id", "")))
        rng.shuffle(rows)

    mode_ids = sorted(by_mode)
    if not mode_ids:
        return []
    total_available = sum(len(rows) for rows in by_mode.values())
    requested = int(sample_size)
    if requested <= 0:
        target = total_available
    else:
        target = min(requested, total_available)
    allocation = {mode_id: 0 for mode_id in mode_ids}
    allocated = 0
    while allocated < target:
        progressed = False
        for mode_id in mode_ids:
            if allocation[mode_id] >= len(by_mode[mode_id]):
                continue
            allocation[mode_id] += 1
            allocated += 1
            progressed = True
            if allocated >= target:
                break
        if not progressed:
            break

    selected: List[Dict[str, Any]] = []
    for mode_id in mode_ids:
        selected.extend(by_mode[mode_id][: allocation[mode_id]])
    selected.sort(key=lambda row: (safe_int(row.get("subject_mode_id"), -1), str(row.get("image_id", ""))))
    return selected


def build_image_index_from_label_json(
    label_json: Path,
    image_root: Path,
    mode_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    payload = load_json(label_json)
    images_by_id = {safe_int(img.get("id"), -1): img for img in safe_list(payload.get("images"))}
    image_map: Dict[str, Dict[str, Any]] = {}
    for ann in safe_list(payload.get("annotations")):
        image_row = safe_dict(images_by_id.get(safe_int(ann.get("image_id"), -1)))
        if not image_row:
            continue
        original_image_id = str(image_row.get("sstk_image_id", image_row.get("id", "")))
        mode_id = safe_int(ann.get("subject_mode_id"), -1)
        mode_name = mode_names.get(mode_id, "unknown")
        file_name = str(image_row.get("file_name", f"{original_image_id}.jpg"))
        entry = image_map.setdefault(
            original_image_id,
            {
                "image_id": original_image_id,
                "image_path": str(find_image_path(image_root, original_image_id, file_name)),
                "image_size": {
                    "width": safe_int(image_row.get("width"), 0),
                    "height": safe_int(image_row.get("height"), 0),
                },
                "subject_mode_id": mode_id,
                "subject_mode": mode_name,
                "subject_mode_ids_seen": Counter(),
                "target_ars": set(),
                "positive_candidates": [],
                "negative_candidates": [],
                "positive_candidates_by_ar": defaultdict(list),
                "negative_candidates_by_ar": defaultdict(list),
                "ignored_candidates": [],
            },
        )
        entry["subject_mode_ids_seen"][mode_id] += 1
        target_ar = str(ann.get("target_ar", image_row.get("target_ar", "")))
        entry["target_ars"].add(target_ar)
        candidate = {
            "candidate_id": str(ann.get("candidate_id", "")),
            "target_ar": target_ar,
            "bbox": list(ann.get("bbox", [0.0, 0.0, 1.0, 1.0])),
            "score_prob": safe_float(ann.get("score"), 0.0),
            "label_type": str(ann.get("label_type", "")),
            "is_hard_negative": bool(ann.get("is_hard_negative", False)),
            "is_unsafe_negative": bool(ann.get("is_unsafe_negative", False)),
        }
        if safe_int(ann.get("gt_flag"), 0) == 1:
            entry["positive_candidates"].append(candidate)
            entry["positive_candidates_by_ar"][target_ar].append(candidate)
        else:
            entry["negative_candidates"].append(candidate)
            entry["negative_candidates_by_ar"][target_ar].append(candidate)

    image_rows: List[Dict[str, Any]] = []
    inconsistencies: Dict[str, Dict[str, int]] = {}
    for image_id, row in image_map.items():
        seen = row.pop("subject_mode_ids_seen")
        if len(seen) > 1:
            inconsistencies[image_id] = dict(seen)
        winner_mode_id = sorted(seen.items(), key=lambda item: (-item[1], item[0]))[0][0]
        row["subject_mode_id"] = winner_mode_id
        row["subject_mode"] = mode_names.get(winner_mode_id, row.get("subject_mode", "unknown"))
        row["positive_candidates"] = dedupe_candidates(row["positive_candidates"])
        row["negative_candidates"] = dedupe_candidates(row["negative_candidates"])
        row["positive_candidates_by_ar"] = {
            str(target_ar): dedupe_candidates(cands)
            for target_ar, cands in sorted(row["positive_candidates_by_ar"].items())
        }
        row["negative_candidates_by_ar"] = {
            str(target_ar): dedupe_candidates(cands)
            for target_ar, cands in sorted(row["negative_candidates_by_ar"].items())
        }
        row["target_ars"] = sort_target_ars(row["target_ars"])
        row["target_record_count"] = len(row["target_ars"])
        image_rows.append(row)
    return image_rows, inconsistencies


def build_batch_candidate_detail_index(
    batch_jsonl: Path,
    *,
    image_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    if not batch_jsonl.exists():
        return {}
    wanted = {str(image_id).strip() for image_id in safe_list(image_ids) if str(image_id).strip()}
    by_image: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    with batch_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            target_ar = str(row.get("target_ar", "")).strip()
            if not image_id or not target_ar:
                continue
            if wanted and image_id not in wanted:
                continue
            target_map = by_image.setdefault(image_id, {}).setdefault(target_ar, {})
            for bucket_key, default_bucket in (
                ("matching_targets", "matching_target"),
                ("candidate_pool", "candidate_pool"),
                ("ignored_candidates", "ignored_candidate"),
                ("overflow_candidates", "overflow_candidate"),
            ):
                for cand in safe_list(row.get(bucket_key)):
                    cand_dict = safe_dict(cand)
                    candidate_id = str(cand_dict.get("candidate_id", "")).strip()
                    if not candidate_id:
                        continue
                    stub = copy.deepcopy(cand_dict)
                    stub.setdefault("target_ar", target_ar)
                    stub.setdefault("training_bucket", default_bucket)
                    target_map[candidate_id] = stub
    return by_image


def enrich_image_rows_with_batch_details(
    image_rows: Sequence[Dict[str, Any]],
    batch_detail_index: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> None:
    if not batch_detail_index:
        return

    def enrich_candidate(image_id: str, candidate: Dict[str, Any]) -> None:
        target_ar = str(candidate.get("target_ar", "")).strip()
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not image_id or not target_ar or not candidate_id:
            return
        batch_stub = safe_dict(safe_dict(safe_dict(batch_detail_index.get(image_id, {})).get(target_ar, {})).get(candidate_id))
        if not batch_stub:
            return
        for key in (
            "bbox_norm_xyxy",
            "score_targets",
            "macro_targets",
            "macro_components",
            "composition_focus",
            "safety_penalty",
            "checklist_labels",
            "checklist_scores",
            "why_tags",
            "reject_tags",
            "why_text_template",
            "training_bucket",
            "positive_anchor_reason",
            "monotonic_label_state",
            "source",
        ):
            if candidate.get(key) not in (None, "", {}, []):
                continue
            value = batch_stub.get(key)
            if value in (None, "", {}, []):
                continue
            candidate[key] = copy.deepcopy(value)

    for row in image_rows:
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        for candidate in safe_list(row.get("positive_candidates")):
            enrich_candidate(image_id, candidate)
        for candidate in safe_list(row.get("negative_candidates")):
            enrich_candidate(image_id, candidate)
        for bucket_name in ("positive_candidates_by_ar", "negative_candidates_by_ar"):
            by_ar = row.get(bucket_name)
            if not isinstance(by_ar, dict):
                continue
            for candidates in by_ar.values():
                for candidate in safe_list(candidates):
                    enrich_candidate(image_id, candidate)


def load_gaic_gt_index(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    image_index: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = load_json(path)
        images = {str(safe_int(img.get("id"), -1)): img for img in safe_list(payload.get("images"))}
        by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ann in safe_list(payload.get("annotations")):
            by_image[str(safe_int(ann.get("image_id"), -1))].append(ann)
        for image_id, anns in by_image.items():
            image_row = safe_dict(images.get(image_id))
            if not image_row:
                continue
            ordered = sorted(anns, key=lambda ann: safe_float(ann.get("score"), 0.0), reverse=True)
            image_index[image_id] = {
                "file_name": str(image_row.get("file_name", "")),
                "width": safe_int(image_row.get("width"), 0),
                "height": safe_int(image_row.get("height"), 0),
                "top3": ordered[:3],
                "annotation_count": len(anns),
            }
    return image_index


def load_gt_score_cache(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return {}
    by_image: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            gt_scores = safe_list(row.get("gt_scores"))
            if not image_id or not gt_scores:
                continue
            image_map = by_image.setdefault(image_id, {})
            for gt_row in gt_scores:
                ann_id = str(safe_int(safe_dict(gt_row).get("ann_id", 0), 0))
                if ann_id and ann_id != "0":
                    image_map[ann_id] = safe_dict(gt_row)
    return by_image


def attach_gt_training_scores(
    gt_index: Dict[str, Dict[str, Any]],
    gt_score_cache: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, int]:
    stats = {"images_with_cache": 0, "gt_boxes_with_score": 0, "gt_boxes_without_score": 0}
    for image_id, gt_entry in gt_index.items():
        cache_rows = gt_score_cache.get(str(image_id), {})
        if cache_rows:
            stats["images_with_cache"] += 1
        for ann in safe_list(gt_entry.get("top3")):
            ann_id = str(safe_int(ann.get("id", 0), 0))
            cache_row = safe_dict(cache_rows.get(ann_id))
            if cache_row:
                ann["training_score_prob"] = safe_float(
                    cache_row.get(
                        "training_score_prob",
                        cache_row.get("score_policy_sigmoid_z_local", cache_row.get("score_sigmoid_z_local", 0.0)),
                    ),
                    0.0,
                )
                ann["training_score_rank_pct"] = safe_float(cache_row.get("score_rank_pct", 0.0), 0.0)
                ann["training_crop_utility_rank_pct"] = safe_float(
                    cache_row.get(
                        "crop_utility_rank_pct",
                        cache_row.get("score_policy_rank_pct", cache_row.get("score_rank_pct", 0.0)),
                    ),
                    0.0,
                )
                ann["training_score_policy"] = safe_float(cache_row.get("score_policy", 0.0), 0.0)
                attach_gt_detail_fields(ann, cache_row)
                stats["gt_boxes_with_score"] += 1
            else:
                stats["gt_boxes_without_score"] += 1
    return stats


def merge_ignored_candidate_counts(
    batch_jsonl: Path,
    image_rows: Sequence[Dict[str, Any]],
    *,
    image_ids: Optional[Sequence[str]] = None,
) -> None:
    if not batch_jsonl.exists():
        return
    wanted = {str(image_id).strip() for image_id in safe_list(image_ids) if str(image_id).strip()}
    ignored_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with batch_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", ""))
            if wanted and image_id not in wanted:
                continue
            for cand in safe_list(row.get("ignored_candidates")):
                ignored_by_image[image_id].append(
                    {
                        "candidate_id": str(cand.get("candidate_id", "")),
                        "target_ar": str(row.get("target_ar", "")),
                        "bbox_norm_xyxy": list(cand.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
                        "score_prob": safe_float(cand.get("score_prob"), 0.0),
                        "label_type": str(cand.get("label_type", "")),
                    }
                )
    for row in image_rows:
        row["ignored_candidates"] = ignored_by_image.get(str(row.get("image_id", "")), [])


def write_manifest_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "image_id",
        "subject_mode_id",
        "subject_mode",
        "image_path",
        "available_target_ars",
        "subject_overlay_mode",
        "positive_count",
        "negative_count",
        "ignored_count",
        "gaic_gt_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))
    resolved_format = resolve_viz_format(str(args.format))
    image_suffix = viz_suffix(resolved_format)
    jpg_quality = max(1, min(100, int(args.jpg_quality)))
    label_json_arg = args.label_json
    if not str(label_json_arg).strip():
        label_json_arg = "data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/label_json/gaic_like_labels_full.json"
    label_json = Path(label_json_arg)
    batch_jsonl = Path(args.batch_jsonl)
    gaic_gt_paths = [Path(args.gaic_gt_train_json), Path(args.gaic_gt_test_json)]
    gt_score_cache_path = Path(args.gt_score_cache_jsonl) if str(args.gt_score_cache_jsonl).strip() else Path(
        label_json.parent.parent / "gaic_gt_score_cache_free.jsonl"
    )
    teacher_jsonl = Path(args.teacher_jsonl) if str(args.teacher_jsonl).strip() else derive_default_teacher_jsonl(batch_jsonl)
    candidates_jsonl = Path(args.candidates_jsonl) if str(args.candidates_jsonl).strip() else None
    score_profile_json = (
        Path(args.score_profile_json)
        if str(args.score_profile_json).strip()
        else (batch_jsonl.parent / "score_profile.json")
    )
    score_profile_display = load_score_profile_display(score_profile_json) if score_profile_json.exists() else {"macros": {}, "components": {}}
    image_root = Path(args.image_root)
    mode_names = invert_vocab(Path(args.subject_mode_vocab))
    out_dir = Path(args.out_dir)
    overlay_pos_dir = out_dir / "per_image" / "positive_vs_gt"
    overlay_neg_dir = out_dir / "per_image" / "negative_vs_gt"
    overlay_all_dir = out_dir / "per_image" / "gt_pos_neg"
    overlay_checklist_dir = out_dir / "per_image" / "gt_pos_neg_checklist"
    distribution_dir = out_dir / "per_image" / "score_distributions"
    overlay_pos_by_ar_dir = out_dir / "per_image" / "positive_vs_gt_by_ar"
    overlay_neg_by_ar_dir = out_dir / "per_image" / "negative_vs_gt_by_ar"
    overlay_all_by_ar_dir = out_dir / "per_image" / "gt_pos_neg_by_ar"
    overlay_checklist_by_ar_dir = out_dir / "per_image" / "gt_pos_neg_checklist_by_ar"
    distribution_by_ar_dir = out_dir / "per_image" / "score_distributions_by_ar"
    summary_dir = out_dir / "summary"
    reset_output_subdirs(
        (
            overlay_pos_dir,
            overlay_neg_dir,
            overlay_all_dir,
            overlay_checklist_dir,
            distribution_dir,
            overlay_pos_by_ar_dir,
            overlay_neg_by_ar_dir,
            overlay_all_by_ar_dir,
            overlay_checklist_by_ar_dir,
            distribution_by_ar_dir,
            summary_dir,
        )
    )
    progress_log(
        f"build_gaic_training_label_debug_viz: start | label_json={label_json} | batch_jsonl={batch_jsonl}",
        enabled=progress_enabled,
    )

    image_rows, mode_inconsistencies = build_image_index_from_label_json(label_json, image_root, mode_names)
    gt_index = load_gaic_gt_index(gaic_gt_paths)
    gt_score_cache_stats = {"images_with_cache": 0, "gt_boxes_with_score": 0, "gt_boxes_without_score": 0}
    if gt_index and gt_score_cache_path.exists():
        gt_score_cache_stats = attach_gt_training_scores(gt_index, load_gt_score_cache(gt_score_cache_path))
    eligible_rows = [row for row in image_rows if str(row.get("image_id", "")) in gt_index]
    if not image_rows:
        write_manifest_csv(summary_dir / "selected_images.csv", [])
        write_json(
            summary_dir / "summary.json",
            {
                "status": "skipped",
                "reason": "no_training_images_in_label_json",
                "input": {
                    "label_json": str(label_json),
                    "batch_jsonl": str(batch_jsonl),
                        "gaic_gt_jsons": [str(path) for path in gaic_gt_paths],
                        "gt_score_cache_jsonl": str(gt_score_cache_path) if gt_score_cache_path.exists() else "",
                        "teacher_jsonl": str(teacher_jsonl) if teacher_jsonl.exists() else "",
                        "image_root": str(image_root),
                    },
                "pool_stats": {
                    "training_images_in_coco": len(image_rows),
                    "eligible_images_with_gt": 0,
                },
                "mode_inconsistencies": mode_inconsistencies,
                "gt_score_cache_stats": gt_score_cache_stats,
                "outputs": {
                    "selected_images_csv": str(summary_dir / "selected_images.csv"),
                },
            },
        )
        print(json.dumps({"status": "skipped", "reason": "no_training_images_in_coco", "out_dir": str(out_dir)}, ensure_ascii=False))
        return

    has_gt_overlap = bool(eligible_rows)
    selection_rows = eligible_rows if has_gt_overlap else image_rows
    selection_title_prefix = (
        "subject_mode_id round-robin balanced sampling across eligible GT images"
        if has_gt_overlap
        else "subject_mode_id round-robin balanced sampling across training-label images without GAIC GT overlap"
    )
    aggregate_title_suffix = "all eligible images" if has_gt_overlap else "all training-label images (no GT overlap)"
    selected_title_suffix = "selected images" if has_gt_overlap else "selected images (no GT overlap)"

    explicit_image_ids = load_image_ids_csv(Path(args.image_ids_csv)) if str(args.image_ids_csv).strip() else []
    if explicit_image_ids:
        selection_lookup = {str(row.get("image_id", "")): row for row in selection_rows}
        selected_images = [selection_lookup[image_id] for image_id in explicit_image_ids if image_id in selection_lookup]
        selection_title_prefix = (
            f"explicit image_id selection from {Path(args.image_ids_csv).name}"
            + (" (eligible GT overlap)" if has_gt_overlap else " (training-label images without GAIC GT overlap)")
        )
        selected_title_suffix = (
            f"explicit image_id selection ({Path(args.image_ids_csv).name})"
            + ("" if has_gt_overlap else " (no GT overlap)")
        )
    else:
        selected_images = select_balanced_images(selection_rows, sample_size=args.sample_size, seed=args.seed)
    selected_image_ids = [str(row.get("image_id", "")) for row in selected_images]
    batch_detail_index = build_batch_candidate_detail_index(batch_jsonl, image_ids=selected_image_ids)
    enrich_image_rows_with_batch_details(selected_images, batch_detail_index)
    merge_ignored_candidate_counts(batch_jsonl, selected_images, image_ids=selected_image_ids)
    eligible_positive_by_mode, eligible_negative_by_mode = collect_mode_score_distributions(selection_rows)
    selected_positive_by_mode, selected_negative_by_mode = collect_mode_score_distributions(selected_images)
    teacher_detail_index = (
        build_teacher_detail_index(teacher_jsonl, image_ids=selected_image_ids)
        if teacher_jsonl.exists()
        else {}
    )
    subject_overlay_index = load_subject_overlay_index(
        candidates_jsonl=candidates_jsonl if candidates_jsonl is not None else Path("__missing_candidates_jsonl__"),
        batch_jsonl=batch_jsonl,
        image_ids=selected_image_ids,
    )
    render_workers = resolve_num_workers(int(args.num_workers), len(selected_images))
    distribution_plot_lock = threading.Lock()
    progress_log(
        f"build_gaic_training_label_debug_viz: selected_images={len(selected_images)} | render_workers={render_workers}",
        enabled=progress_enabled,
    )

    manifest_rows: List[Dict[str, Any]] = []
    per_image_summary: List[Dict[str, Any]] = []
    sampled_mode_counts = Counter()

    def render_selected_image(row_idx: int, row: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
        image_id = str(row["image_id"])
        mode_id = safe_int(row["subject_mode_id"], -1)
        mode_name = str(row["subject_mode"])
        image_path = Path(str(row["image_path"]))
        gt_entry = gt_index.get(image_id, {"top3": [], "annotation_count": 0})
        subject_overlay = subject_overlay_index.get(image_id)
        gt_topk = safe_list(gt_entry.get("top3"))[: args.top_k_gt]
        positives = select_positive_overlay_candidates(row["positive_candidates"], len(row["positive_candidates"]))
        negatives = select_negative_overlay_candidates(row["negative_candidates"], len(row["negative_candidates"]))
        positive_overlay_items = positives[: args.top_k_positive]
        negative_overlay_items = negatives[: args.top_k_negative]
        positive_scores = [candidate_score(cand) for cand in positives]
        negative_scores = [candidate_score(cand) for cand in negatives]

        positive_groups = []
        negative_groups = []
        combined_groups = []
        if has_gt_overlap:
            positive_groups.append({"name": "gt", "color": GT_COLOR, "box_key": "bbox", "items": gt_topk})
            negative_groups.append({"name": "gt", "color": GT_COLOR, "box_key": "bbox", "items": gt_topk})
            combined_groups.append({"name": "gt", "color": GT_COLOR, "box_key": "bbox", "items": gt_topk})
            positive_title = f"gaic_gt_top{args.top_k_gt}_vs_positive_top{args.top_k_positive} | mode={mode_id}:{mode_name}"
            negative_title = f"gaic_gt_top{args.top_k_gt}_vs_negative_bottom{args.top_k_negative} | mode={mode_id}:{mode_name}"
            combined_title = (
                f"gaic_gt_top{args.top_k_gt}_plus_positive_top{args.top_k_positive}_plus_negative_bottom{args.top_k_negative}"
                f" | mode={mode_id}:{mode_name}"
            )
        else:
            positive_title = f"positive_top{args.top_k_positive}_only | mode={mode_id}:{mode_name} | no_gt_overlap"
            negative_title = f"negative_bottom{args.top_k_negative}_only | mode={mode_id}:{mode_name} | no_gt_overlap"
            combined_title = (
                f"positive_top{args.top_k_positive}_plus_negative_bottom{args.top_k_negative}"
                f" | mode={mode_id}:{mode_name} | no_gt_overlap"
            )
        positive_groups.append(
            {
                "name": "positive",
                "color": POSITIVE_COLOR,
                "box_key": "bbox",
                "items": positive_overlay_items,
            }
        )
        negative_groups.append(
            {
                "name": "negative",
                "color": NEGATIVE_COLOR,
                "box_key": "bbox",
                "items": negative_overlay_items,
            }
        )
        combined_groups.extend(
            [
                {
                    "name": "positive",
                    "color": POSITIVE_COLOR,
                    "box_key": "bbox",
                    "items": positive_overlay_items,
                },
                {
                    "name": "negative",
                    "color": NEGATIVE_COLOR,
                    "box_key": "bbox",
                    "items": negative_overlay_items,
                },
            ]
        )
        gt_focus_detail = build_gt_focus_detail(gt_topk[0] if gt_topk else None)
        positive_focus_stub = positives[0] if positives else None
        positive_focus_detail = build_candidate_focus_detail(
            candidate_stub=positive_focus_stub,
            image_id=image_id,
            target_ar=str(positive_focus_stub.get("target_ar", "")) if positive_focus_stub else "",
            teacher_detail_index=teacher_detail_index,
        )
        negative_focus_stub = negatives[0] if negatives else None
        negative_focus_detail = build_candidate_focus_detail(
            candidate_stub=negative_focus_stub,
            image_id=image_id,
            target_ar=str(negative_focus_stub.get("target_ar", "")) if negative_focus_stub else "",
            teacher_detail_index=teacher_detail_index,
        )
        checklist_groups = copy.deepcopy(combined_groups)

        draw_overlay_groups(
            image_path=image_path,
            title=positive_title,
            groups=positive_groups,
            subject_overlay=subject_overlay,
            out_path=build_viz_path(overlay_pos_dir, image_id, resolved_format),
            jpg_quality=jpg_quality,
        )
        draw_overlay_groups(
            image_path=image_path,
            title=negative_title,
            groups=negative_groups,
            subject_overlay=subject_overlay,
            out_path=build_viz_path(overlay_neg_dir, image_id, resolved_format),
            jpg_quality=jpg_quality,
        )
        draw_overlay_groups(
            image_path=image_path,
            title=combined_title,
            groups=combined_groups,
            subject_overlay=subject_overlay,
            out_path=build_viz_path(overlay_all_dir, image_id, resolved_format),
            jpg_quality=jpg_quality,
        )
        draw_checklist_debug_panel(
            image_path=image_path,
            title=combined_title + " | checklist_debug",
            groups=checklist_groups,
            subject_overlay=subject_overlay,
            out_path=build_viz_path(overlay_checklist_dir, image_id, resolved_format),
            gt_detail=gt_focus_detail,
            positive_detail=positive_focus_detail,
            negative_detail=negative_focus_detail,
            show_why_text=bool(args.show_why_text),
            display_spec=score_profile_display,
            jpg_quality=jpg_quality,
        )
        with distribution_plot_lock:
            save_distribution_plot(
                image_id=image_id,
                mode_name=mode_name,
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                out_path=build_viz_path(distribution_dir, image_id, resolved_format),
                jpg_quality=jpg_quality,
            )

        if int(args.skip_by_ar) == 0:
            for target_ar in sort_target_ars(set(row.get("target_ars", []))):
                target_ar_slug = target_ar_dir_name(target_ar)
                ar_positive_dir = overlay_pos_by_ar_dir / target_ar_slug
                ar_negative_dir = overlay_neg_by_ar_dir / target_ar_slug
                ar_combined_dir = overlay_all_by_ar_dir / target_ar_slug
                ar_distribution_dir = distribution_by_ar_dir / target_ar_slug
                for path in (ar_positive_dir, ar_negative_dir, ar_combined_dir, ar_distribution_dir):
                    path.mkdir(parents=True, exist_ok=True)

                raw_positives_by_ar = list(row.get("positive_candidates_by_ar", {}).get(target_ar, []))
                raw_negatives_by_ar = list(row.get("negative_candidates_by_ar", {}).get(target_ar, []))
                positives_by_ar = select_positive_overlay_candidates(
                    raw_positives_by_ar,
                    len(raw_positives_by_ar),
                )
                negatives_by_ar = select_negative_overlay_candidates(
                    raw_negatives_by_ar,
                    len(raw_negatives_by_ar),
                )
                positive_overlay_items_by_ar = positives_by_ar[: args.top_k_positive]
                negative_overlay_items_by_ar = negatives_by_ar[: args.top_k_negative]
                positive_scores_by_ar = [candidate_score(cand) for cand in positives_by_ar]
                negative_scores_by_ar = [candidate_score(cand) for cand in negatives_by_ar]

                ar_positive_groups = []
                ar_negative_groups = []
                ar_combined_groups = []
                if has_gt_overlap:
                    ar_positive_groups.append({"name": "gt", "color": GT_COLOR, "box_key": "bbox", "items": gt_topk})
                    ar_negative_groups.append({"name": "gt", "color": GT_COLOR, "box_key": "bbox", "items": gt_topk})
                    ar_combined_groups.append({"name": "gt", "color": GT_COLOR, "box_key": "bbox", "items": gt_topk})
                    ar_positive_title = (
                        f"gaic_gt_top{args.top_k_gt}_vs_positive_top{args.top_k_positive}"
                        f" | mode={mode_id}:{mode_name} | target_ar={target_ar}"
                    )
                    ar_negative_title = (
                        f"gaic_gt_top{args.top_k_gt}_vs_negative_bottom{args.top_k_negative}"
                        f" | mode={mode_id}:{mode_name} | target_ar={target_ar}"
                    )
                    ar_combined_title = (
                        f"gaic_gt_top{args.top_k_gt}_plus_positive_top{args.top_k_positive}_plus_negative_bottom{args.top_k_negative}"
                        f" | mode={mode_id}:{mode_name} | target_ar={target_ar}"
                    )
                else:
                    ar_positive_title = (
                        f"positive_top{args.top_k_positive}_only"
                        f" | mode={mode_id}:{mode_name} | target_ar={target_ar} | no_gt_overlap"
                    )
                    ar_negative_title = (
                        f"negative_bottom{args.top_k_negative}_only"
                        f" | mode={mode_id}:{mode_name} | target_ar={target_ar} | no_gt_overlap"
                    )
                    ar_combined_title = (
                        f"positive_top{args.top_k_positive}_plus_negative_bottom{args.top_k_negative}"
                        f" | mode={mode_id}:{mode_name} | target_ar={target_ar} | no_gt_overlap"
                    )
                ar_positive_groups.append(
                    {
                        "name": "positive",
                        "color": POSITIVE_COLOR,
                        "box_key": "bbox",
                        "items": positive_overlay_items_by_ar,
                    }
                )
                ar_negative_groups.append(
                    {
                        "name": "negative",
                        "color": NEGATIVE_COLOR,
                        "box_key": "bbox",
                        "items": negative_overlay_items_by_ar,
                    }
                )
                ar_combined_groups.extend(
                    [
                        {
                            "name": "positive",
                            "color": POSITIVE_COLOR,
                            "box_key": "bbox",
                            "items": positive_overlay_items_by_ar,
                        },
                        {
                            "name": "negative",
                            "color": NEGATIVE_COLOR,
                            "box_key": "bbox",
                            "items": negative_overlay_items_by_ar,
                        },
                    ]
                )
                gt_focus_detail_by_ar = build_gt_focus_detail(gt_topk[0] if gt_topk else None)
                positive_focus_stub_by_ar = positives_by_ar[0] if positives_by_ar else None
                positive_focus_detail_by_ar = build_candidate_focus_detail(
                    candidate_stub=positive_focus_stub_by_ar,
                    image_id=image_id,
                    target_ar=str(target_ar),
                    teacher_detail_index=teacher_detail_index,
                )
                negative_focus_stub_by_ar = negatives_by_ar[0] if negatives_by_ar else None
                negative_focus_detail_by_ar = build_candidate_focus_detail(
                    candidate_stub=negative_focus_stub_by_ar,
                    image_id=image_id,
                    target_ar=str(target_ar),
                    teacher_detail_index=teacher_detail_index,
                )
                ar_checklist_groups = copy.deepcopy(ar_combined_groups)

                draw_overlay_groups(
                    image_path=image_path,
                    title=ar_positive_title,
                    groups=ar_positive_groups,
                    subject_overlay=subject_overlay,
                    out_path=build_viz_path(ar_positive_dir, image_id, resolved_format),
                    jpg_quality=jpg_quality,
                )
                draw_overlay_groups(
                    image_path=image_path,
                    title=ar_negative_title,
                    groups=ar_negative_groups,
                    subject_overlay=subject_overlay,
                    out_path=build_viz_path(ar_negative_dir, image_id, resolved_format),
                    jpg_quality=jpg_quality,
                )
                draw_overlay_groups(
                    image_path=image_path,
                    title=ar_combined_title,
                    groups=ar_combined_groups,
                    subject_overlay=subject_overlay,
                    out_path=build_viz_path(ar_combined_dir, image_id, resolved_format),
                    jpg_quality=jpg_quality,
                )
                checklist_dir_by_ar = overlay_checklist_by_ar_dir / target_ar_slug
                checklist_dir_by_ar.mkdir(parents=True, exist_ok=True)
                draw_checklist_debug_panel(
                    image_path=image_path,
                    title=ar_combined_title + " | checklist_debug",
                    groups=ar_checklist_groups,
                    subject_overlay=subject_overlay,
                    out_path=build_viz_path(checklist_dir_by_ar, image_id, resolved_format),
                    gt_detail=gt_focus_detail_by_ar,
                    positive_detail=positive_focus_detail_by_ar,
                    negative_detail=negative_focus_detail_by_ar,
                    show_why_text=bool(args.show_why_text),
                    display_spec=score_profile_display,
                    jpg_quality=jpg_quality,
                )
                with distribution_plot_lock:
                    save_distribution_plot(
                        image_id=image_id,
                        mode_name=f"{mode_name} | {target_ar}",
                        positive_scores=positive_scores_by_ar,
                        negative_scores=negative_scores_by_ar,
                        out_path=build_viz_path(ar_distribution_dir, image_id, resolved_format),
                        jpg_quality=jpg_quality,
                    )

        manifest_row = {
            "image_id": image_id,
            "subject_mode_id": mode_id,
            "subject_mode": mode_name,
            "image_path": str(image_path),
            "available_target_ars": list(row.get("target_ars", [])),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "ignored_count": len(row["ignored_candidates"]),
            "gaic_gt_count": safe_int(gt_entry.get("annotation_count"), 0),
            "subject_overlay_mode": str(safe_dict(subject_overlay).get("mode", "")),
        }
        per_image_row = {
            "image_id": image_id,
            "subject_mode_id": mode_id,
            "subject_mode": mode_name,
            "target_record_count": safe_int(row.get("target_record_count"), 0),
            "available_target_ars": list(row.get("target_ars", [])),
            "positive_summary": summarize_scores(positive_scores),
            "negative_summary": summarize_scores(negative_scores),
            "gaic_gt_topk": [
                {
                    "bbox": ann.get("bbox"),
                    "mos": safe_float(ann.get("score"), 0.0),
                    "training_score_prob": (
                        round(safe_float(ann.get("training_score_prob"), 0.0), 9)
                        if ann.get("training_score_prob") is not None
                        else None
                    ),
                }
                for ann in gt_topk
            ],
            "positive_topk": [
                {
                    "candidate_id": cand.get("candidate_id"),
                    "target_ar": cand.get("target_ar"),
                    "score": candidate_score(cand),
                }
                for cand in positive_overlay_items
            ],
            "negative_bottomk": [
                {
                    "candidate_id": cand.get("candidate_id"),
                    "target_ar": cand.get("target_ar"),
                    "score": candidate_score(cand),
                    "label_type": cand.get("label_type"),
                }
                for cand in negative_overlay_items
            ],
            "subject_overlay": {
                "mode": str(safe_dict(subject_overlay).get("mode", "")),
                "subject_mode": str(safe_dict(subject_overlay).get("subject_mode", "")),
                "subject_family": str(safe_dict(subject_overlay).get("subject_family", "")),
                "subject_subtype": str(safe_dict(subject_overlay).get("subject_subtype", "")),
                "layout_structure": str(safe_dict(subject_overlay).get("layout_structure", "")),
                "support_trust_tier": str(safe_dict(subject_overlay).get("support_trust_tier", "")),
                "score_mode": str(safe_dict(subject_overlay).get("score_mode", "")),
                "state": str(safe_dict(subject_overlay).get("state", "")),
                "support_map_enabled": bool(safe_dict(subject_overlay).get("support_map_enabled", False)),
                "support_hybrid_enabled": bool(safe_dict(subject_overlay).get("support_hybrid_enabled", False)),
                "subject_terms_neutralized": bool(safe_dict(subject_overlay).get("subject_terms_neutralized", False)),
            },
        }
        return row_idx, mode_id, manifest_row, per_image_row

    render_tracker = ProgressTracker(
        "build_gaic_training_label_debug_viz:render_selected_images",
        total=len(selected_images),
        unit="images",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress_enabled,
    )
    if render_workers <= 1:
        for row_idx, row in enumerate(selected_images, start=1):
            _, mode_id, manifest_row, per_image_row = render_selected_image(row_idx - 1, row)
            sampled_mode_counts[mode_id] += 1
            manifest_rows.append(manifest_row)
            per_image_summary.append(per_image_row)
            render_tracker.update(row_idx, extra=f"image_id={row.get('image_id', '')}")
    else:
        ordered_results: List[Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]]] = [None] * len(selected_images)
        with ThreadPoolExecutor(max_workers=render_workers) as executor:
            future_map = {
                executor.submit(render_selected_image, row_idx, row): row_idx
                for row_idx, row in enumerate(selected_images)
            }
            completed = 0
            for future in as_completed(future_map):
                row_idx, mode_id, manifest_row, per_image_row = future.result()
                ordered_results[row_idx] = (mode_id, manifest_row, per_image_row)
                completed += 1
                render_tracker.update(completed, extra=f"image_id={selected_images[row_idx].get('image_id', '')}")
        for row_idx, ordered in enumerate(ordered_results):
            if ordered is None:
                raise RuntimeError(f"Missing render result for selected_images[{row_idx}]")
            mode_id, manifest_row, per_image_row = ordered
            sampled_mode_counts[mode_id] += 1
            manifest_rows.append(manifest_row)
            per_image_summary.append(per_image_row)
    render_tracker.finish(len(selected_images))

    save_subject_mode_summary_plot(
        positive_by_mode=eligible_positive_by_mode,
        negative_by_mode=eligible_negative_by_mode,
        mode_names=mode_names,
        out_path=build_viz_path(summary_dir, "subject_mode_score_distributions_all_eligible", resolved_format),
        title_suffix=aggregate_title_suffix,
        jpg_quality=jpg_quality,
    )
    save_subject_mode_summary_plot(
        positive_by_mode=selected_positive_by_mode,
        negative_by_mode=selected_negative_by_mode,
        mode_names=mode_names,
        out_path=build_viz_path(summary_dir, "subject_mode_score_distributions_selected", resolved_format),
        title_suffix=selected_title_suffix,
        jpg_quality=jpg_quality,
    )
    write_manifest_csv(summary_dir / "selected_images.csv", manifest_rows)
    write_json(
        summary_dir / "summary.json",
        {
            "input": {
                "label_json": str(label_json),
                "batch_jsonl": str(batch_jsonl),
                "gaic_gt_jsons": [str(path) for path in gaic_gt_paths],
                "gt_score_cache_jsonl": str(gt_score_cache_path) if gt_score_cache_path.exists() else "",
                "teacher_jsonl": str(teacher_jsonl) if teacher_jsonl.exists() else "",
                "candidates_jsonl": str(candidates_jsonl) if candidates_jsonl is not None and candidates_jsonl.exists() else "",
                "score_profile_json": str(score_profile_json) if score_profile_json.exists() else "",
                "image_root": str(image_root),
            },
            "score_profile_display": score_profile_display,
            "sample_size_requested": int(args.sample_size),
            "sample_size_selected": len(selected_images),
            "image_ids_csv": str(args.image_ids_csv),
            "seed": int(args.seed),
            "render": {
                "format_requested": str(args.format),
                "format_resolved": resolved_format,
                "image_suffix": image_suffix,
                "jpg_quality": jpg_quality,
                "skip_by_ar": bool(int(args.skip_by_ar) > 0),
                "num_workers": render_workers,
            },
            "selection_policy": selection_title_prefix,
            "gt_overlay_enabled": has_gt_overlap,
            "top_k": {
                "gt": int(args.top_k_gt),
                "positive": int(args.top_k_positive),
                "negative": int(args.top_k_negative),
            },
            "overlay_selection": {
                "positive": "top score descending",
                "negative": "bottom score ascending",
            },
            "pool_stats": {
                "training_images_in_coco": len(image_rows),
                "eligible_images_with_gt": len(eligible_rows),
            },
            "gt_score_cache_stats": gt_score_cache_stats,
            "subject_overlay_stats": {
                "selected_images_with_overlay": sum(1 for row in selected_images if str(row.get("image_id", "")) in subject_overlay_index),
                "selected_support_map_overlays": sum(
                    1 for row in selected_images if str(safe_dict(subject_overlay_index.get(str(row.get("image_id", "")), {})).get("mode", "")) == "support_map"
                ),
                "selected_hybrid_overlays": sum(
                    1 for row in selected_images if bool(safe_dict(subject_overlay_index.get(str(row.get("image_id", "")), {})).get("support_hybrid_enabled", False))
                ),
                "selected_bbox_overlays": sum(
                    1 for row in selected_images if str(safe_dict(subject_overlay_index.get(str(row.get("image_id", "")), {})).get("mode", "")) == "bbox"
                ),
            },
            "selection_source": "eligible_gt_images" if has_gt_overlap else "all_training_images",
            "sampled_subject_mode_counts": dict(sorted(sampled_mode_counts.items())),
            "mode_inconsistencies": mode_inconsistencies,
            "selected_images": per_image_summary,
            "aggregate": {
                "eligible_positive_by_mode": {
                    str(mode_id): {
                        "mode_name": mode_names.get(mode_id, "unknown"),
                        **summarize_scores(scores),
                    }
                    for mode_id, scores in sorted(eligible_positive_by_mode.items())
                },
                "eligible_negative_by_mode": {
                    str(mode_id): {
                        "mode_name": mode_names.get(mode_id, "unknown"),
                        **summarize_scores(scores),
                    }
                    for mode_id, scores in sorted(eligible_negative_by_mode.items())
                },
                "selected_positive_by_mode": {
                    str(mode_id): {
                        "mode_name": mode_names.get(mode_id, "unknown"),
                        **summarize_scores(scores),
                    }
                    for mode_id, scores in sorted(selected_positive_by_mode.items())
                },
                "selected_negative_by_mode": {
                    str(mode_id): {
                        "mode_name": mode_names.get(mode_id, "unknown"),
                        **summarize_scores(scores),
                    }
                    for mode_id, scores in sorted(selected_negative_by_mode.items())
                },
            },
            "outputs": {
                "positive_overlay_dir": str(overlay_pos_dir),
                "negative_overlay_dir": str(overlay_neg_dir),
                "combined_overlay_dir": str(overlay_all_dir),
                "per_image_checklist_dir": str(overlay_checklist_dir),
                "per_image_distribution_dir": str(distribution_dir),
                "positive_overlay_by_ar_dir": str(overlay_pos_by_ar_dir),
                "negative_overlay_by_ar_dir": str(overlay_neg_by_ar_dir),
                "combined_overlay_by_ar_dir": str(overlay_all_by_ar_dir),
                "per_image_checklist_by_ar_dir": str(overlay_checklist_by_ar_dir),
                "per_image_distribution_by_ar_dir": str(distribution_by_ar_dir),
                "subject_mode_summary_plot_all_eligible": str(
                    build_viz_path(summary_dir, "subject_mode_score_distributions_all_eligible", resolved_format)
                ),
                "subject_mode_summary_plot_selected": str(
                    build_viz_path(summary_dir, "subject_mode_score_distributions_selected", resolved_format)
                ),
                "selected_images_csv": str(summary_dir / "selected_images.csv"),
            },
        },
    )
    progress_log(
        f"build_gaic_training_label_debug_viz: finished | selected_images={len(selected_images)} | out_dir={out_dir}",
        enabled=progress_enabled,
    )
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "selected_images": len(selected_images)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
