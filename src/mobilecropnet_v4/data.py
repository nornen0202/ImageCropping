from __future__ import annotations

import base64
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_AR_VOCAB = ["FREE", "1:1", "9:16", "16:9", "3:4", "4:3"]
TARGET_AR_VALUES = {
    "FREE": None,
    "1:1": 1.0,
    "9:16": 9.0 / 16.0,
    "16:9": 16.0 / 9.0,
    "3:4": 3.0 / 4.0,
    "4:3": 4.0 / 3.0,
}
DECISION_VOCAB = ["keep_full", "minimal_crop", "crop"]
SUBJECT_MODE_VOCAB = [
    "background_texture_copyspace",
    "object_multi",
    "object_single",
    "portrait_group",
    "portrait_single",
    "scene_general",
    "other_ambiguous",
]
SUBJECT_MODE_COUNT = 7
DEFAULT_DESTRUCTIVE_REJECT_TAGS = {
    "face_cut",
    "head_top_cut",
    "headroom_violation",
    "joint_cutoff",
    "lookroom_violation",
    "person_cut",
    "subject_cutoff",
    "text_cut",
}
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
CHECKLIST_CLASS_VOCABS: dict[str, tuple[str, ...]] = {
    "subject_coverage": ("na", "poor", "marginal", "good", "excellent"),
    "subject_scale": ("na", "too_loose", "ideal_scale", "too_tight"),
    "headroom": ("na", "headroom_tight", "headroom_ok", "headroom_loose"),
    "lookroom": ("na", "lookroom_insufficient", "lookroom_adequate", "lookroom_excessive"),
    "face_cut": ("na", "no_face_cut", "face_cut", "face_cut_mild"),
    "joint_cut": ("na", "no_joint_cut", "joint_cut_mild", "joint_cut"),
    "copyspace": ("na", "copyspace_missing", "copyspace_partial", "copyspace_preserved"),
    "context": ("na", "context_poor", "context_partial", "context_preserved", "context_excessive", "context_loss", "context_lost"),
    "third_dist": ("na", "rule_of_thirds_weak", "rule_of_thirds_strong"),
    "phi_dist": ("na", "phi_grid_weak", "phi_grid_strong"),
    "center_dist": ("na", "center_comp_weak", "center_comp_strong"),
    "ar": ("na", "ar_fits_well", "ar_choice_freeform"),
    "horizon_state": ("na", "weak", "ok", "strong", "horizon_on_target", "horizon_off_target"),
}
CHECKLIST_CLASS_KEYS = tuple(CHECKLIST_CLASS_VOCABS.keys())
CHECKLIST_CLASS_SIZES = tuple(len(CHECKLIST_CLASS_VOCABS[key]) for key in CHECKLIST_CLASS_KEYS)
CHECKLIST_CLASS_TOTAL = int(sum(CHECKLIST_CLASS_SIZES))
CHECKLIST_SCORE_KEYS = (
    "subject_coverage_ratio",
    "subject_scale_ratio",
    "headroom_ratio_norm",
    "lookroom_ratio_norm",
    "third_strength",
    "phi_strength",
    "center_strength",
    "horizon_y_norm",
    "horizon_visible_ratio",
    "context_value",
    "symmetry_score",
    "copyspace_blank_ratio_keep",
    "placement_score",
    "placement_family_margin",
    "placement_reward_third",
    "placement_reward_phi",
    "placement_reward_center",
    "C_headroom",
    "C_lookroom",
    "C_context",
    "safety_penalty_soft",
    "safety_penalty_hard",
)
WHY_TAG_VOCAB = (
    "avoid_face_cut",
    "avoid_person_cut",
    "symmetry",
    "ar_fits_well",
    "ar_choice_freeform",
    "balanced_crop",
    "wide_crop",
    "tight_crop",
    "context_preserved",
    "context_loss",
    "rule_of_thirds",
    "phi_grid",
    "centered_subject",
    "subject_preserved",
    "subject_poor",
    "subject_marginal",
    "subject_scale_ideal",
    "subject_scale_loose",
    "subject_scale_tight",
    "subject_terms_neutralized",
    "head_top_safe",
    "headroom_ok",
    "headroom_violation",
    "lookroom_ok",
    "lookroom_violation",
    "copy_space_kept",
    "copy_space_lost",
    "needs_leveling",
    "horizon_on_target",
    "horizon_off_target",
    "ar_extreme_penalty",
)
CHECKLIST_SCORE_COUNT = len(CHECKLIST_SCORE_KEYS)
WHY_TAG_COUNT = len(WHY_TAG_VOCAB)
CHECKLIST_CLASS_COUNT = len(CHECKLIST_CLASS_KEYS)
_COMPOSITION_CHECKLIST_KEYS = {"third_dist", "phi_dist", "center_dist"}
_UNIVERSAL_CHECKLIST_KEYS = {*_COMPOSITION_CHECKLIST_KEYS, "copyspace", "context", "ar"}
_SUBJECT_CHECKLIST_KEYS = {"subject_coverage", "subject_scale"}
_PERSON_CHECKLIST_KEYS = {"headroom", "lookroom", "face_cut", "joint_cut"}
_SCENE_CHECKLIST_KEYS = {"horizon_state"}
EXPLANATION_APPLICABILITY_BY_MODE: dict[str, frozenset[str]] = {
    "portrait_single": frozenset(_UNIVERSAL_CHECKLIST_KEYS | _SUBJECT_CHECKLIST_KEYS | _PERSON_CHECKLIST_KEYS | _SCENE_CHECKLIST_KEYS),
    "portrait_group": frozenset(_UNIVERSAL_CHECKLIST_KEYS | _SUBJECT_CHECKLIST_KEYS | _PERSON_CHECKLIST_KEYS | _SCENE_CHECKLIST_KEYS),
    "object_single": frozenset(_UNIVERSAL_CHECKLIST_KEYS | _SUBJECT_CHECKLIST_KEYS),
    "object_multi": frozenset(_UNIVERSAL_CHECKLIST_KEYS | _SUBJECT_CHECKLIST_KEYS),
    "scene_general": frozenset(_UNIVERSAL_CHECKLIST_KEYS | _SCENE_CHECKLIST_KEYS),
    "background_texture_copyspace": frozenset({"copyspace", "context", "ar", "third_dist", "phi_dist", "center_dist"}),
    "other_ambiguous": frozenset({"context", "ar", "third_dist", "phi_dist", "center_dist"}),
}
CHECKLIST_SCORE_APPLICABILITY_KEY: dict[str, str | None] = {
    "subject_coverage_ratio": "subject_coverage",
    "subject_scale_ratio": "subject_scale",
    "headroom_ratio_norm": "headroom",
    "lookroom_ratio_norm": "lookroom",
    "third_strength": "third_dist",
    "phi_strength": "phi_dist",
    "center_strength": "center_dist",
    "horizon_y_norm": "horizon_state",
    "horizon_visible_ratio": "horizon_state",
    "context_value": "context",
    "symmetry_score": "center_dist",
    "copyspace_blank_ratio_keep": "copyspace",
    "placement_score": None,
    "placement_family_margin": None,
    "placement_reward_third": "third_dist",
    "placement_reward_phi": "phi_dist",
    "placement_reward_center": "center_dist",
    "C_headroom": "headroom",
    "C_lookroom": "lookroom",
    "C_context": "context",
    "safety_penalty_soft": None,
    "safety_penalty_hard": None,
}
PERSON_ONLY_WHY_TAGS = frozenset(
    {
        "avoid_face_cut",
        "avoid_person_cut",
        "head_top_safe",
        "headroom_ok",
        "headroom_violation",
        "lookroom_ok",
        "lookroom_violation",
    }
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def target_ar_id(value: str) -> int:
    try:
        return TARGET_AR_VOCAB.index(str(value))
    except ValueError:
        return 0


def target_ar_value(value: str | int | None) -> float | None:
    if isinstance(value, int):
        if 0 <= value < len(TARGET_AR_VOCAB):
            return TARGET_AR_VALUES.get(TARGET_AR_VOCAB[value])
        return None
    return TARGET_AR_VALUES.get(str(value or "FREE"))


def subject_mode_label_from_id(value: int) -> str:
    idx = max(0, min(len(SUBJECT_MODE_VOCAB) - 1, int(value)))
    return SUBJECT_MODE_VOCAB[idx]


def subject_mode_label_from_row(row: dict[str, Any]) -> str:
    route = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    if isinstance(route.get("subject_mode"), str) and route.get("subject_mode") in SUBJECT_MODE_VOCAB:
        return str(route.get("subject_mode"))
    return subject_mode_label_from_id(int(max(0, min(SUBJECT_MODE_COUNT - 1, int(safe_float(route.get("subject_mode_id"), 0))))))


def checklist_applicability_by_mode(subject_mode: str | None) -> list[float]:
    applicable = EXPLANATION_APPLICABILITY_BY_MODE.get(str(subject_mode or "other_ambiguous"), EXPLANATION_APPLICABILITY_BY_MODE["other_ambiguous"])
    return [float(key in applicable) for key in CHECKLIST_CLASS_KEYS]


def checklist_score_applicability_by_mode(subject_mode: str | None) -> list[float]:
    class_app = dict(zip(CHECKLIST_CLASS_KEYS, checklist_applicability_by_mode(subject_mode)))
    out: list[float] = []
    for score_key in CHECKLIST_SCORE_KEYS:
        class_key = CHECKLIST_SCORE_APPLICABILITY_KEY.get(score_key)
        out.append(1.0 if class_key is None else float(class_app.get(class_key, 0.0) > 0.0))
    return out


def why_tag_applicability_by_mode(subject_mode: str | None) -> list[float]:
    mode = str(subject_mode or "other_ambiguous")
    person_mode = mode in {"portrait_single", "portrait_group"}
    return [float(person_mode or tag not in PERSON_ONLY_WHY_TAGS) for tag in WHY_TAG_VOCAB]


def decision_id(value: Any, fallback: int = 2) -> int:
    if isinstance(value, int):
        return max(0, min(len(DECISION_VOCAB) - 1, value))
    try:
        return DECISION_VOCAB.index(str(value))
    except ValueError:
        return max(0, min(len(DECISION_VOCAB) - 1, int(fallback)))


def decision_label_from_row(row: dict[str, Any]) -> str:
    payload = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    idx = decision_id(payload.get("decision_id", payload.get("decision_type", 2)))
    return DECISION_VOCAB[idx]


def decision_score_target(decision: dict[str, Any] | None) -> float:
    payload = decision if isinstance(decision, dict) else {}
    did = decision_id(payload.get("decision_id", payload.get("decision_type", 2)))
    base = 0.0 if did == 0 else 0.5 if did == 1 else 1.0
    delta = safe_float(payload.get("delta_vs_base", payload.get("decision_delta_vs_baseline")), 0.0)
    delta_norm = max(-1.0, min(1.0, delta / 3.0))
    return max(0.0, min(1.0, base + 0.15 * delta_norm))


def _clamp_box(box: Sequence[float]) -> list[float]:
    if len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [max(0.0, min(1.0, safe_float(v))) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [x1, y1, x2, y2]


def xyxy_to_cxcywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = _clamp_box(box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1 + 0.5 * w, y1 + 0.5 * h, w, h]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = _clamp_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _clamp_box(a)
    bx1, by1, bx2, by2 = _clamp_box(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


SUBJECT_BOX_TARGET_SOURCES = (
    "legacy",
    "raw",
    "overlay",
    "support_latent",
    "support_core",
    "support_envelope",
    "support_hybrid",
)

SUBJECT_BOX_VALID_TARGET_MODES = (
    "route_gated",
    "route_reliable",
    "box_exists",
    "box_reliability",
    "box_primary_or_reliable",
)


def _subject_support_spec_from_row(row: dict[str, Any]) -> dict[str, Any]:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    overlay = routing.get("subject_support_overlay") if isinstance(routing.get("subject_support_overlay"), dict) else {}
    spec = overlay.get("support_spec") if isinstance(overlay.get("support_spec"), dict) else {}
    return spec


def _first_valid_box(*values: Any) -> list[float] | None:
    for value in values:
        if isinstance(value, list):
            box = _clamp_box(value)
            if box[2] > box[0] and box[3] > box[1]:
                return box
    return None


def _subject_prior_box_from_row(row: dict[str, Any], *, source: str = "legacy") -> list[float] | None:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    overlay = routing.get("subject_support_overlay") if isinstance(routing.get("subject_support_overlay"), dict) else {}
    spec = overlay.get("support_spec") if isinstance(overlay.get("support_spec"), dict) else {}
    source = str(source or "legacy").strip().lower()
    raw = routing.get("subject_prior_bbox_norm_xyxy")
    overlay_box = overlay.get("bbox_norm_xyxy")
    latent = spec.get("latent_support_bbox_norm_xyxy")
    core = spec.get("core_bbox_norm_xyxy", spec.get("latent_support_core_bbox_norm_xyxy"))
    latent_core = spec.get("latent_support_core_bbox_norm_xyxy")
    envelope = spec.get("envelope_norm_xyxy")

    if source in {"legacy", "raw", "subject_prior"}:
        return _first_valid_box(raw, overlay_box)
    if source in {"overlay", "overlay_bbox"}:
        return _first_valid_box(overlay_box, raw)
    if source in {"support_latent", "latent", "latent_support"}:
        return _first_valid_box(latent, envelope, overlay_box, raw)
    if source in {"support_core", "core", "latent_core"}:
        return _first_valid_box(latent_core, core, latent, overlay_box, raw)
    if source in {"support_envelope", "envelope"}:
        return _first_valid_box(envelope, latent, overlay_box, raw)
    if source in {"support_hybrid", "hybrid"}:
        mode = subject_mode_label_from_row(row)
        if mode in {"scene_general", "background_texture_copyspace"}:
            return _first_valid_box(envelope, latent, overlay_box, raw)
        return _first_valid_box(latent, latent_core, core, overlay_box, raw)
    if isinstance(routing.get("subject_prior_bbox_norm_xyxy"), list):
        return _clamp_box(routing.get("subject_prior_bbox_norm_xyxy"))
    if isinstance(overlay.get("bbox_norm_xyxy"), list):
        return _clamp_box(overlay.get("bbox_norm_xyxy"))
    return None


def _subject_prior_reliability_from_row(row: dict[str, Any]) -> float:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    flags = routing.get("flags") if isinstance(routing.get("flags"), dict) else {}
    overlay = routing.get("subject_support_overlay") if isinstance(routing.get("subject_support_overlay"), dict) else {}
    value = safe_float(flags.get("subject_reliability"), safe_float(overlay.get("subject_reliability"), 0.0))
    return max(0.0, min(1.0, value))


def _subject_box_supervision_valid_from_row(
    row: dict[str, Any],
    *,
    has_box: bool,
    mode: str = "route_gated",
    reliability_min: float = 0.25,
) -> float:
    if not has_box:
        return 0.0
    target_mode = str(mode or "route_gated").strip().lower()
    reliability_min = max(0.0, min(1.0, float(reliability_min)))
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    flags = routing.get("flags") if isinstance(routing.get("flags"), dict) else {}
    subject_set = routing.get("subject_set") if isinstance(routing.get("subject_set"), dict) else {}
    route_mode = str(routing.get("subject_mode", subject_mode_label_from_id(int(safe_float(routing.get("subject_mode_id"), 3)))))
    effective_state = str(flags.get("subject_effective_state", "") or "").strip().lower()
    reliability = _subject_prior_reliability_from_row(row)

    if bool(flags.get("subject_placeholder_flag", False)) or bool(flags.get("subject_terms_neutralized", False)):
        return 0.0
    if target_mode == "box_exists":
        return 1.0
    if target_mode == "box_reliability":
        return 1.0 if reliability >= reliability_min else 0.0
    has_primary = bool(subject_set.get("primary_subject_exists", False))
    if target_mode == "box_primary_or_reliable":
        return 1.0 if has_primary or reliability >= reliability_min else 0.0
    if bool(flags.get("subject_mode_is_background_like", False)):
        return 0.0
    if target_mode == "route_reliable":
        if route_mode in {"portrait_single", "portrait_group", "object_single", "object_multi"}:
            return 1.0 if reliability >= reliability_min else 0.0
        dominant_states = {"dominant_subject", "primary_subject", "single_subject"}
        return 1.0 if effective_state in dominant_states and has_primary and reliability >= reliability_min else 0.0
    if route_mode in {"portrait_single", "portrait_group", "object_single", "object_multi"}:
        return 1.0

    # Scene/copyspace/ambiguous modes often have no single foreground subject.
    # Keep supervision only when the teacher explicitly found a dominant subject.
    dominant_states = {"dominant_subject", "primary_subject", "single_subject"}
    if effective_state in dominant_states and has_primary and reliability >= 0.25:
        return 1.0
    return 0.0


def _decode_subject_support_grid(row: dict[str, Any], *, grid_size: int = 24) -> np.ndarray | None:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    overlay = routing.get("subject_support_overlay") if isinstance(routing.get("subject_support_overlay"), dict) else {}
    spec = overlay.get("support_spec") if isinstance(overlay.get("support_spec"), dict) else {}
    payload = spec.get("support_mass_grid_f16_b64") or spec.get("mass_grid_f16_b64")
    if not isinstance(payload, str) or not payload:
        return None
    try:
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    except Exception:
        return None
    spec_grid = int(safe_float(spec.get("support_grid_size", spec.get("grid_size", grid_size)), grid_size))
    if spec_grid <= 0:
        spec_grid = int(round(math.sqrt(max(1, int(arr.size)))))
    if arr.size != spec_grid * spec_grid:
        side = int(round(math.sqrt(max(1, int(arr.size)))))
        if side * side != arr.size:
            return None
        spec_grid = side
    grid = arr.reshape(spec_grid, spec_grid)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.maximum(grid, 0.0)
    if spec_grid == grid_size:
        return grid
    # Small, dependency-free nearest resize is sufficient for a supervision mask.
    ys = np.linspace(0, spec_grid - 1, grid_size).round().astype(np.int64)
    xs = np.linspace(0, spec_grid - 1, grid_size).round().astype(np.int64)
    return grid[ys[:, None], xs[None, :]]


def _subject_support_mask_from_row(
    row: dict[str, Any],
    transform: "LetterboxTransform",
    *,
    grid_size: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = _decode_subject_support_grid(row, grid_size=grid_size)
    if grid is None or float(grid.max(initial=0.0)) <= 0.0:
        return torch.zeros((grid_size, grid_size), dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    grid = grid / max(float(grid.max()), 1e-6)
    content = letterbox_content_box(transform)
    x1, y1, x2, y2 = content
    out = np.zeros((grid_size, grid_size), dtype=np.float32)
    for yy in range(grid_size):
        ly = (float(yy) + 0.5) / float(grid_size)
        if ly < y1 or ly > y2:
            continue
        src_y = min(grid_size - 1, max(0, int(((ly - y1) / max(1e-6, y2 - y1)) * grid_size)))
        for xx in range(grid_size):
            lx = (float(xx) + 0.5) / float(grid_size)
            if lx < x1 or lx > x2:
                continue
            src_x = min(grid_size - 1, max(0, int(((lx - x1) / max(1e-6, x2 - x1)) * grid_size)))
            out[yy, xx] = float(grid[src_y, src_x])
    if float(out.max(initial=0.0)) <= 0.0:
        return torch.zeros((grid_size, grid_size), dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    out = out / max(float(out.max()), 1e-6)
    return torch.tensor(out, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    input_size: int
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int

    @property
    def image_ar_log(self) -> float:
        return math.log(max(1e-6, float(self.original_width) / float(max(1, self.original_height))))


def compute_letterbox_transform(width: int, height: int, input_size: int) -> LetterboxTransform:
    width = max(1, int(width))
    height = max(1, int(height))
    size = int(input_size)
    scale = min(float(size) / float(width), float(size) / float(height))
    resized_width = max(1, int(round(float(width) * scale)))
    resized_height = max(1, int(round(float(height) * scale)))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    return LetterboxTransform(
        original_width=width,
        original_height=height,
        input_size=size,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def original_to_letterbox_box(box: Sequence[float], transform: LetterboxTransform) -> list[float]:
    x1, y1, x2, y2 = _clamp_box(box)
    sx = float(transform.resized_width) / float(transform.input_size)
    sy = float(transform.resized_height) / float(transform.input_size)
    ox = float(transform.pad_x) / float(transform.input_size)
    oy = float(transform.pad_y) / float(transform.input_size)
    return _clamp_box([ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy])


def letterbox_to_original_box(box: Sequence[float], transform: LetterboxTransform) -> list[float]:
    x1, y1, x2, y2 = _clamp_box(box)
    sx = float(transform.resized_width) / float(transform.input_size)
    sy = float(transform.resized_height) / float(transform.input_size)
    ox = float(transform.pad_x) / float(transform.input_size)
    oy = float(transform.pad_y) / float(transform.input_size)
    if sx <= 0 or sy <= 0:
        return [0.0, 0.0, 1.0, 1.0]
    return _clamp_box([(x1 - ox) / sx, (y1 - oy) / sy, (x2 - ox) / sx, (y2 - oy) / sy])


def letterbox_content_box(transform: LetterboxTransform) -> list[float]:
    size = float(max(1, int(transform.input_size)))
    x1 = float(transform.pad_x) / size
    y1 = float(transform.pad_y) / size
    x2 = float(transform.pad_x + transform.resized_width) / size
    y2 = float(transform.pad_y + transform.resized_height) / size
    return _clamp_box([x1, y1, x2, y2])


def _norm_triplet(values: Sequence[float] | None, default: Sequence[float]) -> tuple[float, float, float]:
    if values is None:
        return (float(default[0]), float(default[1]), float(default[2]))
    items = [float(v) for v in values]
    if len(items) != 3:
        raise ValueError(f"expected 3 normalization values, got {len(items)}")
    return (items[0], items[1], items[2])


def load_image_tensor_letterbox(
    path: Path,
    input_size: int,
    *,
    image_mean: Sequence[float] | None = None,
    image_std: Sequence[float] | None = None,
) -> tuple[torch.Tensor, LetterboxTransform]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        transform = compute_letterbox_transform(width, height, input_size)
        resized = rgb.resize((transform.resized_width, transform.resized_height), Image.BILINEAR)
        canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        canvas.paste(resized, (transform.pad_x, transform.pad_y))
    mean = np.asarray(_norm_triplet(image_mean, DEFAULT_IMAGE_MEAN), dtype=np.float32)
    std = np.asarray(_norm_triplet(image_std, DEFAULT_IMAGE_STD), dtype=np.float32)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous(), transform


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _score_from_candidate(candidate: dict[str, Any]) -> float:
    for key in ("crop_utility_prob", "score_prob", "score_policy", "score_rank"):
        if candidate.get(key) is not None:
            return max(0.0, min(1.0, safe_float(candidate.get(key))))
    score_targets = candidate.get("score_targets")
    if isinstance(score_targets, dict):
        for key in ("crop_utility_prob", "score_prob"):
            if score_targets.get(key) is not None:
                return max(0.0, min(1.0, safe_float(score_targets.get(key))))
    raw = safe_float(candidate.get("crop_utility_raw"), 0.0)
    if 0.0 <= raw <= 1.0:
        return raw
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-raw))))


def _candidate_value(candidate: dict[str, Any], key: str) -> float | None:
    if candidate.get(key) is not None:
        return safe_float(candidate.get(key))
    score_targets = candidate.get("score_targets")
    if isinstance(score_targets, dict) and score_targets.get(key) is not None:
        return safe_float(score_targets.get(key))
    return None


def _score_from_candidate_mode(candidate: dict[str, Any], mode: str) -> float:
    mode = str(mode or "default")
    if mode in {"default", "crop_utility_prob"}:
        value = _candidate_value(candidate, "crop_utility_prob")
        if value is not None:
            return max(0.0, min(1.0, value))
        if mode == "crop_utility_prob":
            return _score_from_candidate(candidate)
        return _score_from_candidate(candidate)
    if mode == "score_prob":
        value = _candidate_value(candidate, "score_prob")
        return max(0.0, min(1.0, value)) if value is not None else _score_from_candidate(candidate)
    if mode in {"score_rank_pct", "crop_utility_rank_pct", "rank_pct"}:
        keys = ("score_rank_pct", "crop_utility_rank_pct", "rank_pct") if mode == "rank_pct" else (mode,)
        for key in keys:
            value = _candidate_value(candidate, key)
            if value is not None:
                return max(0.0, min(1.0, value))
        return _score_from_candidate(candidate)
    if mode in {"score_softmax_local", "crop_utility_softmax_local", "softmax_local"}:
        keys = ("score_softmax_local", "crop_utility_softmax_local", "softmax_local") if mode == "softmax_local" else (mode,)
        for key in keys:
            value = _candidate_value(candidate, key)
            if value is not None:
                return max(0.0, min(1.0, value))
        return _score_from_candidate(candidate)
    if mode in {"raw_ranker_zscore_prob", "raw_ranker_minmax_prob"}:
        value = _candidate_value(candidate, mode)
        return max(0.0, min(1.0, value)) if value is not None else _score_from_candidate(candidate)
    if mode == "pseudo_mos_1to5":
        value = _candidate_value(candidate, "pseudo_mos_1to5")
        return max(0.0, min(1.0, (value - 1.0) / 4.0)) if value is not None else _score_from_candidate(candidate)
    if mode == "hybrid_rank":
        values = []
        for key in ("crop_utility_prob", "score_prob", "crop_utility_rank_pct", "score_rank_pct"):
            value = _candidate_value(candidate, key)
            if value is not None:
                values.append(max(0.0, min(1.0, value)))
        return float(sum(values) / len(values)) if values else _score_from_candidate(candidate)
    return _score_from_candidate(candidate)


def _candidate_confidence(candidate: dict[str, Any], *, source: str) -> float:
    explicit = candidate.get("candidate_weight", candidate.get("training_weight"))
    if explicit is not None:
        return max(0.05, min(1.0, safe_float(explicit, 1.0)))
    margins = [
        abs(safe_float(_candidate_value(candidate, "score_margin_to_top1"), 0.0)),
        abs(safe_float(_candidate_value(candidate, "crop_utility_margin_to_top1"), 0.0)),
    ]
    margin_conf = min(1.0, max(margins) * 8.0)
    conf = 0.35 + 0.65 * margin_conf
    if source == "matching_target" or bool(candidate.get("is_positive_candidate", False)):
        conf = max(conf, 0.85)
    if bool(candidate.get("is_hard_negative", False)) or bool(candidate.get("is_unsafe_negative", False)):
        conf = max(conf, 0.75)
    if bool(candidate.get("is_ignore_candidate", False)):
        conf *= 0.5
    if bool(candidate.get("is_overflow_candidate", False)):
        conf = max(conf, 0.8)
    return max(0.05, min(1.0, conf))


def _merge_candidate_label(candidate: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if not extra:
        return candidate
    merged = dict(candidate)
    for key, value in extra.items():
        if key not in merged or merged.get(key) is None:
            merged[key] = value
    return merged


def _load_listwise_map(path: str | Path | None) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))
            bucket = out.setdefault(key, {})
            for candidate in row.get("candidates") or []:
                cid = str(candidate.get("candidate_id", ""))
                if cid:
                    bucket[cid] = candidate
    return out


def _load_pairwise_map(path: str | Path | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))
            out.setdefault(key, []).append(row)
    return out


def _macro_targets(candidate: dict[str, Any]) -> list[float]:
    macro = candidate.get("macro_targets")
    if not isinstance(macro, dict):
        macro = candidate.get("macro_components")
    if not isinstance(macro, dict):
        return [0.0, 0.0, 0.0, 0.0]
    keys = ("aesthetic_prob", "subject_prob", "composition_prob", "technical_prob")
    if any(k in macro for k in keys):
        return [max(0.0, min(1.0, safe_float(macro.get(k)))) for k in keys]
    macro_keys = ("A_macro", "S_macro", "C_macro", "T_macro")
    if any(k in macro for k in macro_keys):
        return [max(0.0, min(1.0, safe_float(macro.get(k)))) for k in macro_keys]
    return [
        max(0.0, min(1.0, safe_float(macro.get("A", macro.get("aesthetic", 0.0))))),
        max(0.0, min(1.0, safe_float(macro.get("S", macro.get("subject", 0.0))))),
        max(0.0, min(1.0, safe_float(macro.get("C", macro.get("composition", 0.0))))),
        max(0.0, min(1.0, safe_float(macro.get("T", macro.get("technical", 0.0))))),
    ]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _clip01_or_none(value: Any) -> float | None:
    parsed = _maybe_float(value)
    if parsed is None:
        return None
    return max(0.0, min(1.0, parsed))


def _is_meaningful_checklist_label(value: Any, vocab: Sequence[str]) -> bool:
    if value is None:
        return False
    raw = str(value).strip()
    if raw.lower() in {"", "na", "n/a", "none", "null", "nan"}:
        return False
    return raw in vocab


def _strength_from_distance(value: Any, *, scale: float = 0.3) -> float | None:
    parsed = _maybe_float(value)
    if parsed is None:
        return None
    return max(0.0, min(1.0, 1.0 - parsed / max(1e-6, float(scale))))


def _checklist_detail_targets(candidate: dict[str, Any], *, subject_mode_label: str | None) -> dict[str, Any]:
    labels = _safe_dict(candidate.get("checklist_labels"))
    scores = _safe_dict(candidate.get("checklist_scores"))
    macro_components = _safe_dict(candidate.get("macro_components"))
    composition_focus = _safe_dict(candidate.get("composition_focus"))
    safety_penalty = _safe_dict(candidate.get("safety_penalty"))
    class_applicable = checklist_applicability_by_mode(subject_mode_label)
    score_applicable = checklist_score_applicability_by_mode(subject_mode_label)
    why_applicable = why_tag_applicability_by_mode(subject_mode_label)

    class_target: list[int] = []
    class_valid: list[float] = []
    for key, applicable in zip(CHECKLIST_CLASS_KEYS, class_applicable):
        vocab = CHECKLIST_CLASS_VOCABS[key]
        raw = str(labels.get(key, "na") or "na")
        class_target.append(vocab.index(raw) if raw in vocab else 0)
        class_valid.append(float(applicable > 0.0 and key in labels and _is_meaningful_checklist_label(raw, vocab)))

    score_values: dict[str, float | None] = {
        "subject_coverage_ratio": _clip01_or_none(scores.get("subject_coverage_ratio")),
        "subject_scale_ratio": _clip01_or_none(scores.get("subject_scale_ratio")),
        "headroom_ratio_norm": _clip01_or_none(((_maybe_float(scores.get("headroom_ratio")) or 0.0) / 0.6) if scores.get("headroom_ratio") is not None else None),
        "lookroom_ratio_norm": _clip01_or_none(((_maybe_float(scores.get("lookroom_ratio")) or 0.0) / 3.0) if scores.get("lookroom_ratio") is not None else None),
        "third_strength": _strength_from_distance(scores.get("third_dist")),
        "phi_strength": _strength_from_distance(scores.get("phi_dist")),
        "center_strength": _strength_from_distance(scores.get("center_dist")),
        "horizon_y_norm": _clip01_or_none(scores.get("horizon_y")),
        "horizon_visible_ratio": _clip01_or_none(scores.get("horizon_visible_ratio")),
        "context_value": _clip01_or_none(scores.get("context_value")),
        "symmetry_score": _clip01_or_none(scores.get("symmetry_score")),
        "copyspace_blank_ratio_keep": _clip01_or_none(scores.get("copyspace_blank_ratio_keep")),
        "placement_score": _clip01_or_none(composition_focus.get("placement_score")),
        "placement_family_margin": _clip01_or_none(composition_focus.get("placement_family_margin")),
        "placement_reward_third": _clip01_or_none(composition_focus.get("placement_reward_third")),
        "placement_reward_phi": _clip01_or_none(composition_focus.get("placement_reward_phi")),
        "placement_reward_center": _clip01_or_none(composition_focus.get("placement_reward_center")),
        "C_headroom": _clip01_or_none(macro_components.get("C_headroom")),
        "C_lookroom": _clip01_or_none(macro_components.get("C_lookroom")),
        "C_context": _clip01_or_none(macro_components.get("C_context")),
        "safety_penalty_soft": _clip01_or_none(safety_penalty.get("soft_total")),
        "safety_penalty_hard": _clip01_or_none(safety_penalty.get("hard_total")),
    }
    score_target = [float(score_values[key]) if score_values[key] is not None else 0.0 for key in CHECKLIST_SCORE_KEYS]
    score_valid = [float(score_values[key] is not None and score_applicable[idx] > 0.0) for idx, key in enumerate(CHECKLIST_SCORE_KEYS)]

    tag_set = {str(tag) for tag in (candidate.get("why_tags") or [])}
    reject_tag_set = {str(tag) for tag in (candidate.get("reject_tags") or [])}
    tag_set.update(reject_tag_set)
    why_target = [float(tag in tag_set and why_applicable[idx] > 0.0) for idx, tag in enumerate(WHY_TAG_VOCAB)]
    why_valid = float(bool(tag_set))
    return {
        "checklist_class_target": class_target,
        "checklist_class_valid": class_valid,
        "checklist_applicability_target": class_applicable,
        "checklist_applicability_valid": [1.0] * len(class_applicable),
        "detail_score_target": score_target,
        "detail_score_valid": score_valid,
        "why_tag_target": why_target,
        "why_tag_valid": why_valid,
        "why_tag_applicable": why_applicable,
        "teacher_checklist_labels": dict(labels),
        "teacher_checklist_scores": dict(scores),
        "teacher_composition_focus": dict(composition_focus),
        "teacher_why_tags": sorted(tag_set),
    }


def _candidate_id(candidate: dict[str, Any], fallback: str) -> str:
    return str(candidate.get("candidate_id") or candidate.get("target_id") or fallback)


def _candidate_box(candidate: dict[str, Any]) -> list[float]:
    if "bbox_norm_xyxy" in candidate:
        return _clamp_box(candidate["bbox_norm_xyxy"])
    if "bbox_cxcywh" in candidate:
        cx, cy, w, h = [safe_float(v) for v in candidate["bbox_cxcywh"][:4]]
        return _clamp_box([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h])
    return [0.0, 0.0, 1.0, 1.0]


def _box_meta(box: Sequence[float], *, is_base: float) -> list[float]:
    cxcywh = xyxy_to_cxcywh(box)
    area = box_area(box)
    aspect = cxcywh[2] / max(1e-6, cxcywh[3])
    return [*_clamp_box(box), *cxcywh, area, aspect, float(is_base)]


def _resolve_image_path(row: dict[str, Any], *, project_root: Path, image_root: Path | None) -> Path:
    raw = Path(str(row.get("image_path", "")))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(project_root / raw)
        if image_root is not None:
            candidates.append(image_root / raw)
            candidates.append(image_root / raw.name)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else raw


class MobileCropNetV4BatchDataset(Dataset):
    """Dataset adapter for current SSTK conditional-DETR batch JSONL labels."""

    def __init__(
        self,
        *,
        jsonl_path: str | Path,
        project_root: str | Path,
        image_root: str | Path | None = None,
        input_size: int = 256,
        candidate_k: int = 24,
        max_positive_boxes: int = 8,
        max_rows: int | None = None,
        include_ignored_candidates: bool = False,
        include_overflow_candidates: bool = False,
        min_soft_positive_score: float = 0.6,
        precompute_sample_tensors: bool = False,
        image_tensor_cache_size: int = 0,
        image_mean: Sequence[float] | None = None,
        image_std: Sequence[float] | None = None,
        pairwise_jsonl: str | Path | None = None,
        listwise_jsonl: str | Path | None = None,
        max_pairwise_pairs: int = 32,
        score_target_mode: str = "default",
        subject_box_target_source: str = "legacy",
        subject_box_valid_target_mode: str = "route_gated",
        subject_box_valid_reliability_min: float = 0.25,
        high_score_safe_positive_threshold: float = 0.72,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.project_root = Path(project_root)
        self.image_root = Path(image_root) if image_root is not None else None
        self.input_size = int(input_size)
        self.candidate_k = int(candidate_k)
        self.max_positive_boxes = int(max_positive_boxes)
        self.include_ignored_candidates = bool(include_ignored_candidates)
        self.include_overflow_candidates = bool(include_overflow_candidates)
        self.min_soft_positive_score = float(min_soft_positive_score)
        self.precompute_sample_tensors = bool(precompute_sample_tensors)
        self.image_tensor_cache_size = max(0, int(image_tensor_cache_size))
        self.image_mean = _norm_triplet(image_mean, DEFAULT_IMAGE_MEAN)
        self.image_std = _norm_triplet(image_std, DEFAULT_IMAGE_STD)
        self.max_pairwise_pairs = max(0, int(max_pairwise_pairs))
        self.score_target_mode = str(score_target_mode or "default")
        self.subject_box_target_source = str(subject_box_target_source or "legacy").strip().lower()
        if self.subject_box_target_source not in SUBJECT_BOX_TARGET_SOURCES:
            raise ValueError(
                "unsupported subject_box_target_source: "
                f"{self.subject_box_target_source!r}; choices={SUBJECT_BOX_TARGET_SOURCES}"
            )
        self.subject_box_valid_target_mode = str(subject_box_valid_target_mode or "route_gated").strip().lower()
        if self.subject_box_valid_target_mode not in SUBJECT_BOX_VALID_TARGET_MODES:
            raise ValueError(
                "unsupported subject_box_valid_target_mode: "
                f"{self.subject_box_valid_target_mode!r}; choices={SUBJECT_BOX_VALID_TARGET_MODES}"
            )
        self.subject_box_valid_reliability_min = max(0.0, min(1.0, float(subject_box_valid_reliability_min)))
        self.high_score_safe_positive_threshold = float(high_score_safe_positive_threshold)
        self.pairwise_map = _load_pairwise_map(pairwise_jsonl)
        self.listwise_map = _load_listwise_map(listwise_jsonl)
        self._image_tensor_cache: OrderedDict[str, tuple[torch.Tensor, LetterboxTransform]] = OrderedDict()

        records: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if max_rows is not None and len(records) >= int(max_rows):
                    break
        if not records:
            raise ValueError(f"no records found in {self.jsonl_path}")
        self.records = records
        self._precomputed_samples: list[dict[str, Any]] | None = None
        if self.precompute_sample_tensors:
            self._precomputed_samples = [self._precompute_sample(row) for row in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> dict[str, Any]:
        ar_counts: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        decision_counts: dict[str, int] = {}
        matching_count = 0
        pool_count = 0
        subject_prior_box_count = 0
        for row in self.records:
            ar = str(row.get("target_ar", "FREE"))
            ar_counts[ar] = ar_counts.get(ar, 0) + 1
            mode = subject_mode_label_from_row(row)
            route_counts[mode] = route_counts.get(mode, 0) + 1
            decision = decision_label_from_row(row)
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            matching_count += len(row.get("matching_targets") or [])
            pool_count += len(row.get("candidate_pool") or [])
            subject_prior_box_count += int(_subject_prior_box_from_row(row, source=self.subject_box_target_source) is not None)
        image_count = len(self.records)
        return {
            "jsonl_path": str(self.jsonl_path),
            "image_count": image_count,
            "target_ar_counts": ar_counts,
            "subject_mode_counts": route_counts,
            "decision_type_counts": decision_counts,
            "matching_target_count": matching_count,
            "candidate_pool_count": pool_count,
            "matching_targets_per_row_mean": float(matching_count / max(1, image_count)),
            "candidate_pool_per_row_mean": float(pool_count / max(1, image_count)),
            "subject_prior_box_count": subject_prior_box_count,
            "subject_prior_box_rate": float(subject_prior_box_count / max(1, image_count)),
            "crop_decision_rate": float(decision_counts.get("crop", 0) / max(1, image_count)),
            "baseline_decision_rate": float(
                (decision_counts.get("keep_full", 0) + decision_counts.get("minimal_crop", 0)) / max(1, image_count)
            ),
            "input_size": self.input_size,
            "candidate_k": self.candidate_k,
            "max_positive_boxes": self.max_positive_boxes,
            "precompute_sample_tensors": self.precompute_sample_tensors,
            "image_tensor_cache_size": self.image_tensor_cache_size,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "score_target_mode": self.score_target_mode,
            "subject_box_target_source": self.subject_box_target_source,
            "subject_box_valid_target_mode": self.subject_box_valid_target_mode,
            "subject_box_valid_reliability_min": self.subject_box_valid_reliability_min,
            "max_pairwise_pairs": self.max_pairwise_pairs,
            "pairwise_label_group_count": len(self.pairwise_map),
            "listwise_label_group_count": len(self.listwise_map),
            "high_score_safe_positive_threshold": self.high_score_safe_positive_threshold,
            "checklist_class_keys": list(CHECKLIST_CLASS_KEYS),
            "checklist_score_keys": list(CHECKLIST_SCORE_KEYS),
            "why_tag_vocab": list(WHY_TAG_VOCAB),
        }

    def _candidate_records(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[float, float, float, float]]] = set()
        listwise = self.listwise_map.get((str(row.get("image_id", "")), str(row.get("target_ar", "FREE"))), {})
        subject_mode = subject_mode_label_from_row(row)

        def add(candidate: dict[str, Any], *, source: str, is_base: bool = False, positive_override: bool | None = None) -> None:
            cid = _candidate_id(candidate, f"{source}_{len(selected)}")
            candidate = _merge_candidate_label(candidate, listwise.get(cid))
            box = _candidate_box(candidate)
            key = (cid, tuple(round(v, 5) for v in box))
            if key in seen:
                return
            seen.add(key)
            score = _score_from_candidate_mode(candidate, self.score_target_mode)
            is_positive = bool(positive_override) if positive_override is not None else bool(candidate.get("is_positive_candidate", False))
            is_soft_positive = bool(candidate.get("is_soft_positive", False))
            is_hard_negative = bool(candidate.get("is_hard_negative", False))
            is_unsafe_negative = bool(candidate.get("is_unsafe_negative", False))
            is_ignore = bool(candidate.get("is_ignore_candidate", False))
            is_overflow = bool(candidate.get("is_overflow_candidate", False))
            reject_tags = [str(tag) for tag in (candidate.get("reject_tags") or [])]
            is_destructive = any(tag in DEFAULT_DESTRUCTIVE_REJECT_TAGS for tag in reject_tags)
            detail_targets = _checklist_detail_targets(candidate, subject_mode_label=subject_mode)
            record = {
                "candidate_id": cid,
                "bbox_norm_xyxy": box,
                "score_target": score,
                "teacher_soft_target": max(
                    0.0,
                    min(
                        1.0,
                        safe_float(
                            _candidate_value(candidate, "teacher_softmax_local")
                            if _candidate_value(candidate, "teacher_softmax_local") is not None
                            else _candidate_value(candidate, "crop_utility_softmax_local"),
                            0.0,
                        ),
                    ),
                ),
                "positive_target": float(is_positive or is_soft_positive or positive_override is True),
                "risk_target": float(is_hard_negative or is_unsafe_negative or is_overflow or is_destructive),
                "candidate_weight": _candidate_confidence(candidate, source=source),
                "is_base": float(is_base),
                "is_ignore": is_ignore,
                "is_overflow": is_overflow,
                "is_hard_negative": is_hard_negative,
                "is_unsafe_negative": is_unsafe_negative,
                "reject_tags": reject_tags,
                "label_type": str(candidate.get("label_type", source)),
                "training_bucket": str(candidate.get("training_bucket", "")),
                "source": source,
                "subject_mode_label": subject_mode,
                "macro_target": _macro_targets(candidate),
            }
            record.update(detail_targets)
            selected.append(record)

        baseline = row.get("baseline")
        if isinstance(baseline, dict):
            add(baseline, source="baseline", is_base=True, positive_override=False)

        matching_targets = list(row.get("matching_targets") or [])
        matching_targets.sort(key=_score_from_candidate, reverse=True)
        for idx, candidate in enumerate(matching_targets):
            add(candidate, source="matching_target", positive_override=True)

        pool = list(row.get("candidate_pool") or [])
        if self.include_ignored_candidates:
            pool.extend(row.get("ignored_candidates") or [])
        if self.include_overflow_candidates:
            pool.extend(row.get("overflow_candidates") or [])

        def pool_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
            score = _score_from_candidate(candidate)
            return (
                float(bool(candidate.get("is_positive_candidate", False) or bool(candidate.get("is_soft_positive", False)))),
                float(bool(candidate.get("is_hard_negative", False) or bool(candidate.get("is_unsafe_negative", False)))),
                score,
                -box_area(_candidate_box(candidate)),
            )

        for candidate in sorted(pool, key=pool_key, reverse=True):
            if bool(candidate.get("is_ignore_candidate", False)) and not self.include_ignored_candidates:
                continue
            if bool(candidate.get("is_overflow_candidate", False)) and not self.include_overflow_candidates:
                continue
            add(candidate, source="candidate_pool")
            if len(selected) >= self.candidate_k:
                break
        return selected[: self.candidate_k]

    def _positive_boxes(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        positives: list[dict[str, Any]] = []
        seen: set[tuple[float, float, float, float]] = set()

        def add(candidate: dict[str, Any]) -> None:
            box = _candidate_box(candidate)
            key = tuple(round(v, 5) for v in box)
            if key in seen:
                return
            seen.add(key)
            positives.append({"bbox_norm_xyxy": box, "score": _score_from_candidate_mode(candidate, self.score_target_mode)})

        for candidate in sorted(row.get("matching_targets") or [], key=_score_from_candidate, reverse=True):
            add(candidate)
        for candidate in sorted(row.get("candidate_pool") or [], key=_score_from_candidate, reverse=True):
            score = _score_from_candidate_mode(candidate, self.score_target_mode)
            safe = not (
                bool(candidate.get("is_hard_negative", False))
                or bool(candidate.get("is_unsafe_negative", False))
                or bool(candidate.get("is_ignore_candidate", False))
                or bool(candidate.get("is_overflow_candidate", False))
            )
            if bool(candidate.get("is_positive_candidate", False)) or (
                bool(candidate.get("is_soft_positive", False)) and score >= self.min_soft_positive_score
            ) or (
                safe and score >= self.high_score_safe_positive_threshold
            ):
                add(candidate)
        return positives[: self.max_positive_boxes]

    def _explicit_pairwise_tensors(self, row: dict[str, Any], slot_by_candidate_id: dict[str, int]) -> dict[str, torch.Tensor]:
        p = self.max_pairwise_pairs
        pair_i = torch.zeros((p,), dtype=torch.long)
        pair_j = torch.zeros((p,), dtype=torch.long)
        pair_label = torch.zeros((p,), dtype=torch.float32)
        pair_weight = torch.zeros((p,), dtype=torch.float32)
        pair_valid = torch.zeros((p,), dtype=torch.float32)
        if p <= 0 or not self.pairwise_map:
            return {
                "pair_i": pair_i,
                "pair_j": pair_j,
                "pair_label": pair_label,
                "pair_weight": pair_weight,
                "pair_valid": pair_valid,
            }
        key = (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))
        rows = []
        for pair in self.pairwise_map.get(key, []):
            a = str(pair.get("candidate_id_a", ""))
            b = str(pair.get("candidate_id_b", ""))
            if a not in slot_by_candidate_id or b not in slot_by_candidate_id:
                continue
            margin = max(abs(safe_float(pair.get("score_margin"), 0.0)), abs(safe_float(pair.get("crop_utility_margin"), 0.0)))
            weight = max(0.05, min(1.0, margin * 8.0))
            if str(pair.get("pair_type", "")).startswith("top1"):
                weight = max(weight, 0.75)
            label = safe_float(pair.get("label"), 1.0)
            sign = 1.0 if label >= 0 else -1.0
            rows.append((weight, slot_by_candidate_id[a], slot_by_candidate_id[b], sign))
        rows.sort(key=lambda item: item[0], reverse=True)
        for idx, (weight, i, j, sign) in enumerate(rows[:p]):
            pair_i[idx] = int(i)
            pair_j[idx] = int(j)
            pair_label[idx] = float(sign)
            pair_weight[idx] = float(weight)
            pair_valid[idx] = 1.0
        return {
            "pair_i": pair_i,
            "pair_j": pair_j,
            "pair_label": pair_label,
            "pair_weight": pair_weight,
            "pair_valid": pair_valid,
        }

    def _build_sample_tensors(self, row: dict[str, Any], *, image_path: Path, transform: LetterboxTransform) -> dict[str, Any]:
        candidate_records = self._candidate_records(row)
        positive_records = self._positive_boxes(row)

        k = self.candidate_k
        p = self.max_positive_boxes
        boxes = torch.zeros((k, 4), dtype=torch.float32)
        boxes_orig = torch.zeros((k, 4), dtype=torch.float32)
        box_meta = torch.zeros((k, 11), dtype=torch.float32)
        score_target = torch.zeros((k,), dtype=torch.float32)
        positive_target = torch.zeros((k,), dtype=torch.float32)
        risk_target = torch.zeros((k,), dtype=torch.float32)
        teacher_soft_target = torch.zeros((k,), dtype=torch.float32)
        candidate_is_base = torch.zeros((k,), dtype=torch.float32)
        candidate_weight = torch.zeros((k,), dtype=torch.float32)
        valid = torch.zeros((k,), dtype=torch.float32)
        macro_target = torch.zeros((k, 4), dtype=torch.float32)
        macro_valid = torch.zeros((k,), dtype=torch.float32)
        checklist_class_target = torch.zeros((k, len(CHECKLIST_CLASS_KEYS)), dtype=torch.long)
        checklist_class_valid = torch.zeros((k, len(CHECKLIST_CLASS_KEYS)), dtype=torch.float32)
        checklist_applicability_target = torch.zeros((k, len(CHECKLIST_CLASS_KEYS)), dtype=torch.float32)
        checklist_applicability_valid = torch.zeros((k, len(CHECKLIST_CLASS_KEYS)), dtype=torch.float32)
        detail_score_target = torch.zeros((k, CHECKLIST_SCORE_COUNT), dtype=torch.float32)
        detail_score_valid = torch.zeros((k, CHECKLIST_SCORE_COUNT), dtype=torch.float32)
        why_tag_target = torch.zeros((k, WHY_TAG_COUNT), dtype=torch.float32)
        why_tag_valid = torch.zeros((k,), dtype=torch.float32)
        why_tag_applicable = torch.zeros((k, WHY_TAG_COUNT), dtype=torch.float32)
        slot_by_candidate_id: dict[str, int] = {}

        for slot, candidate in enumerate(candidate_records[:k]):
            orig_box = _clamp_box(candidate["bbox_norm_xyxy"])
            padded_box = original_to_letterbox_box(orig_box, transform)
            is_base = float(candidate.get("is_base", 0.0))
            boxes[slot] = torch.tensor(padded_box, dtype=torch.float32)
            boxes_orig[slot] = torch.tensor(orig_box, dtype=torch.float32)
            box_meta[slot] = torch.tensor(_box_meta(padded_box, is_base=is_base), dtype=torch.float32)
            score_target[slot] = float(candidate.get("score_target", 0.0))
            teacher_soft_target[slot] = float(candidate.get("teacher_soft_target", 0.0))
            positive_target[slot] = float(candidate.get("positive_target", 0.0))
            risk_target[slot] = float(candidate.get("risk_target", 0.0))
            candidate_is_base[slot] = is_base
            candidate_weight[slot] = float(candidate.get("candidate_weight", 1.0))
            valid[slot] = 1.0
            slot_by_candidate_id[str(candidate.get("candidate_id", slot))] = slot
            mt = candidate.get("macro_target")
            if isinstance(mt, list) and len(mt) == 4 and any(abs(float(v)) > 0 for v in mt):
                macro_target[slot] = torch.tensor(mt, dtype=torch.float32)
                macro_valid[slot] = 1.0
            ct = candidate.get("checklist_class_target")
            cv = candidate.get("checklist_class_valid")
            if isinstance(ct, list) and len(ct) == len(CHECKLIST_CLASS_KEYS):
                checklist_class_target[slot] = torch.tensor([int(v) for v in ct], dtype=torch.long)
            if isinstance(cv, list) and len(cv) == len(CHECKLIST_CLASS_KEYS):
                checklist_class_valid[slot] = torch.tensor([float(v) for v in cv], dtype=torch.float32)
            at = candidate.get("checklist_applicability_target")
            av = candidate.get("checklist_applicability_valid")
            if isinstance(at, list) and len(at) == len(CHECKLIST_CLASS_KEYS):
                checklist_applicability_target[slot] = torch.tensor([float(v) for v in at], dtype=torch.float32)
            if isinstance(av, list) and len(av) == len(CHECKLIST_CLASS_KEYS):
                checklist_applicability_valid[slot] = torch.tensor([float(v) for v in av], dtype=torch.float32)
            ds = candidate.get("detail_score_target")
            dv = candidate.get("detail_score_valid")
            if isinstance(ds, list) and len(ds) == CHECKLIST_SCORE_COUNT:
                detail_score_target[slot] = torch.tensor([float(v) for v in ds], dtype=torch.float32)
            if isinstance(dv, list) and len(dv) == CHECKLIST_SCORE_COUNT:
                detail_score_valid[slot] = torch.tensor([float(v) for v in dv], dtype=torch.float32)
            wt = candidate.get("why_tag_target")
            if isinstance(wt, list) and len(wt) == WHY_TAG_COUNT:
                why_tag_target[slot] = torch.tensor([float(v) for v in wt], dtype=torch.float32)
                why_tag_valid[slot] = float(candidate.get("why_tag_valid", 0.0))
            wa = candidate.get("why_tag_applicable")
            if isinstance(wa, list) and len(wa) == WHY_TAG_COUNT:
                why_tag_applicable[slot] = torch.tensor([float(v) for v in wa], dtype=torch.float32)

        positive_boxes = torch.zeros((p, 4), dtype=torch.float32)
        positive_boxes_orig = torch.zeros((p, 4), dtype=torch.float32)
        positive_valid = torch.zeros((p,), dtype=torch.float32)
        positive_scores = torch.zeros((p,), dtype=torch.float32)
        for slot, positive in enumerate(positive_records[:p]):
            orig_box = _clamp_box(positive["bbox_norm_xyxy"])
            positive_boxes[slot] = torch.tensor(original_to_letterbox_box(orig_box, transform), dtype=torch.float32)
            positive_boxes_orig[slot] = torch.tensor(orig_box, dtype=torch.float32)
            positive_scores[slot] = float(positive.get("score", 0.0))
            positive_valid[slot] = 1.0

        decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
        route = row.get("routing") if isinstance(row.get("routing"), dict) else {}
        decision_target = decision_id(decision.get("decision_id", decision.get("decision_type", 2)))
        route_target = int(max(0, min(SUBJECT_MODE_COUNT - 1, int(safe_float(route.get("subject_mode_id"), 0)))))
        delta_target = safe_float(
            decision.get("delta_vs_base", decision.get("decision_delta_vs_baseline")),
            0.0,
        )
        subject_prior_box_orig = _subject_prior_box_from_row(row, source=self.subject_box_target_source)
        subject_prior_box = torch.zeros((4,), dtype=torch.float32)
        subject_prior_valid = torch.tensor(0.0, dtype=torch.float32)
        if subject_prior_box_orig is not None:
            subject_prior_box = torch.tensor(original_to_letterbox_box(subject_prior_box_orig, transform), dtype=torch.float32)
            subject_prior_valid = torch.tensor(1.0, dtype=torch.float32)
        subject_prior_reliability = torch.tensor(_subject_prior_reliability_from_row(row), dtype=torch.float32)
        subject_box_valid = torch.tensor(
            _subject_box_supervision_valid_from_row(
                row,
                has_box=subject_prior_box_orig is not None,
                mode=self.subject_box_valid_target_mode,
                reliability_min=self.subject_box_valid_reliability_min,
            ),
            dtype=torch.float32,
        )
        subject_box_weight = torch.tensor(
            max(0.25, float(subject_prior_reliability.item())) if float(subject_box_valid.item()) > 0.0 else 1.0,
            dtype=torch.float32,
        )
        subject_support_mask, subject_support_mask_valid = _subject_support_mask_from_row(row, transform)

        sample = {
            "boxes": boxes,
            "boxes_orig": boxes_orig,
            "box_meta": box_meta,
            "valid": valid,
            "score_target": score_target,
            "teacher_soft_target": teacher_soft_target,
            "positive_target": positive_target,
            "risk_target": risk_target,
            "candidate_is_base": candidate_is_base,
            "candidate_weight": candidate_weight,
            "macro_target": macro_target,
            "macro_valid": macro_valid,
            "checklist_class_target": checklist_class_target,
            "checklist_class_valid": checklist_class_valid,
            "checklist_applicability_target": checklist_applicability_target,
            "checklist_applicability_valid": checklist_applicability_valid,
            "detail_score_target": detail_score_target,
            "detail_score_valid": detail_score_valid,
            "why_tag_target": why_tag_target,
            "why_tag_valid": why_tag_valid,
            "why_tag_applicable": why_tag_applicable,
            "positive_boxes": positive_boxes,
            "positive_boxes_orig": positive_boxes_orig,
            "positive_valid": positive_valid,
            "positive_scores": positive_scores,
            "target_ar_id": torch.tensor(target_ar_id(str(row.get("target_ar", "FREE"))), dtype=torch.long),
            "image_ar_log": torch.tensor(transform.image_ar_log, dtype=torch.float32),
            "letterbox_content_box": torch.tensor(letterbox_content_box(transform), dtype=torch.float32),
            "decision_target": torch.tensor(decision_target, dtype=torch.long),
            "decision_score_target": torch.tensor(decision_score_target(decision), dtype=torch.float32),
            "route_target": torch.tensor(route_target, dtype=torch.long),
            "delta_target": torch.tensor(delta_target, dtype=torch.float32),
            "subject_prior_box": subject_prior_box,
            "subject_prior_valid": subject_prior_valid,
            "subject_prior_reliability": subject_prior_reliability,
            "subject_box_target": subject_prior_box,
            "subject_box_valid": subject_box_valid,
            "subject_box_weight": subject_box_weight,
            "subject_support_mask": subject_support_mask,
            "subject_support_mask_valid": subject_support_mask_valid,
            "image_id": str(row.get("image_id", "")),
            "image_path": str(image_path),
            "target_ar": str(row.get("target_ar", "FREE")),
            "width": transform.original_width,
            "height": transform.original_height,
            "transform": transform,
            "candidate_records": candidate_records,
            "positive_records": positive_records,
        }
        sample.update(self._explicit_pairwise_tensors(row, slot_by_candidate_id))
        return sample

    def _precompute_sample(self, row: dict[str, Any]) -> dict[str, Any]:
        image_path = _resolve_image_path(row, project_root=self.project_root, image_root=self.image_root)
        width, height = read_image_size(image_path)
        transform = compute_letterbox_transform(width, height, self.input_size)
        return self._build_sample_tensors(row, image_path=image_path, transform=transform)

    @staticmethod
    def _clone_precomputed_sample(sample: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in sample.items():
            out[key] = value.clone() if torch.is_tensor(value) else value
        return out

    def _load_image(self, image_path: Path) -> tuple[torch.Tensor, LetterboxTransform]:
        if self.image_tensor_cache_size <= 0:
            return load_image_tensor_letterbox(image_path, self.input_size, image_mean=self.image_mean, image_std=self.image_std)
        cache_key = f"{image_path.resolve()}::{self.input_size}::{self.image_mean}::{self.image_std}"
        cached = self._image_tensor_cache.get(cache_key)
        if cached is not None:
            self._image_tensor_cache.move_to_end(cache_key)
            image, transform = cached
            return image.clone(), transform
        image, transform = load_image_tensor_letterbox(image_path, self.input_size, image_mean=self.image_mean, image_std=self.image_std)
        self._image_tensor_cache[cache_key] = (image, transform)
        while len(self._image_tensor_cache) > self.image_tensor_cache_size:
            self._image_tensor_cache.popitem(last=False)
        return image.clone(), transform

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        precomputed = self._precomputed_samples[idx] if self._precomputed_samples is not None else None
        image_path = (
            Path(str(precomputed["image_path"]))
            if precomputed is not None
            else _resolve_image_path(row, project_root=self.project_root, image_root=self.image_root)
        )
        image, transform = self._load_image(image_path)
        sample = self._clone_precomputed_sample(precomputed) if precomputed is not None else self._build_sample_tensors(row, image_path=image_path, transform=transform)
        sample["image"] = image
        return sample


def mobilecropnet_v4_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "image",
        "boxes",
        "boxes_orig",
        "box_meta",
        "valid",
        "score_target",
        "teacher_soft_target",
        "positive_target",
        "risk_target",
        "candidate_is_base",
        "candidate_weight",
        "macro_target",
        "macro_valid",
        "checklist_class_target",
        "checklist_class_valid",
        "checklist_applicability_target",
        "checklist_applicability_valid",
        "detail_score_target",
        "detail_score_valid",
        "why_tag_target",
        "why_tag_valid",
        "why_tag_applicable",
        "positive_boxes",
        "positive_boxes_orig",
        "positive_valid",
        "positive_scores",
        "target_ar_id",
        "image_ar_log",
        "letterbox_content_box",
        "decision_target",
        "decision_score_target",
        "route_target",
        "delta_target",
        "subject_prior_box",
        "subject_prior_valid",
        "subject_prior_reliability",
        "subject_box_target",
        "subject_box_valid",
        "subject_box_weight",
        "subject_support_mask",
        "subject_support_mask_valid",
        "pair_i",
        "pair_j",
        "pair_label",
        "pair_weight",
        "pair_valid",
    ]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in (
        "image_id",
        "image_path",
        "target_ar",
        "width",
        "height",
        "transform",
        "candidate_records",
        "positive_records",
    ):
        out[key] = [item[key] for item in batch]
    return out
