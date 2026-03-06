from __future__ import annotations

import importlib
import math
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


def _fallback_apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    if chunk_size <= 0:
        return forward_fn(*input_tensors)
    if len(input_tensors) == 0:
        return forward_fn()
    dim_size = input_tensors[0].shape[chunk_dim]
    if dim_size % chunk_size != 0:
        raise ValueError(
            f"chunk_size={chunk_size} must divide input size {dim_size} at dim={chunk_dim}"
        )
    num_chunks = dim_size // chunk_size
    import torch

    input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
    output_chunks = tuple(forward_fn(*chunk_args) for chunk_args in zip(*input_chunks))
    return torch.cat(output_chunks, dim=chunk_dim)


def _fallback_find_pruneable_heads_and_indices(
    heads, n_heads: int, head_size: int, already_pruned_heads
):
    import torch

    heads = set(int(h) for h in heads) - set(int(h) for h in already_pruned_heads)
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        if 0 <= head < n_heads:
            mask[head] = False
    mask = mask.view(-1)
    index = torch.arange(mask.numel(), dtype=torch.long)[mask]
    return heads, index


def _fallback_prune_linear_layer(layer, index, dim: int = 0):
    import torch
    import torch.nn as nn

    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            bias = layer.bias.clone().detach()
        else:
            bias = layer.bias.index_select(0, index).clone().detach()
    else:
        bias = None

    new_size = list(layer.weight.size())
    new_size[dim] = int(index.numel())
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True

    if bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(bias.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def _patch_transformers_generation_compat() -> None:
    """
    mmpretrain(ofa/blip) imports several symbols from transformers.modeling_utils.
    Newer transformers versions may move or rename some of them, causing
    mmpretrain to set PreTrainedModel=None and fail during class definition.
    Provide backward-compatible aliases when possible.
    """
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        # Best-effort compatibility shim; if unavailable, downstream path handles it.
        return

    # Generation symbols (moved across transformers versions).
    try:
        gen_mod = importlib.import_module("transformers.generation")
    except Exception:
        gen_mod = None
    generation_mixin = getattr(gen_mod, "GenerationMixin", None) if gen_mod is not None else None
    generation_config = getattr(gen_mod, "GenerationConfig", None) if gen_mod is not None else None
    if getattr(modeling_utils, "GenerationMixin", None) is None and generation_mixin is not None:
        modeling_utils.GenerationMixin = generation_mixin
    if getattr(modeling_utils, "GenerationConfig", None) is None and generation_config is not None:
        modeling_utils.GenerationConfig = generation_config

    # Some mmpretrain paths import these helpers from modeling_utils, but in
    # newer transformers they may live in pytorch_utils.
    try:
        pytorch_utils = importlib.import_module("transformers.pytorch_utils")
    except Exception:
        pytorch_utils = None

    helper_fallbacks = {
        "apply_chunking_to_forward": _fallback_apply_chunking_to_forward,
        "find_pruneable_heads_and_indices": _fallback_find_pruneable_heads_and_indices,
        "prune_linear_layer": _fallback_prune_linear_layer,
    }
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if getattr(modeling_utils, name, None) is not None:
            continue
        sym = getattr(pytorch_utils, name, None) if pytorch_utils is not None else None
        if sym is None:
            sym = helper_fallbacks[name]
        setattr(modeling_utils, name, sym)

    # Backfill modeling_outputs aliases used by mmpretrain BLIP in some
    # transformers version ranges.
    try:
        modeling_outputs = importlib.import_module("transformers.modeling_outputs")
    except Exception:
        modeling_outputs = None
    if modeling_outputs is not None:
        alias_map = {
            "BaseModelOutputWithPastAndCrossAttentions": "BaseModelOutputWithPast",
            "BaseModelOutputWithPoolingAndCrossAttentions": "BaseModelOutputWithPooling",
            "CausalLMOutputWithCrossAttentions": "CausalLMOutputWithPast",
        }
        for dst, src in alias_map.items():
            if getattr(modeling_outputs, dst, None) is not None:
                continue
            src_obj = getattr(modeling_outputs, src, None)
            if src_obj is not None:
                setattr(modeling_outputs, dst, src_obj)

    # Keep ACT2FN importable even in minimal/legacy builds.
    try:
        activations_mod = importlib.import_module("transformers.activations")
    except Exception:
        activations_mod = None
    if activations_mod is not None and getattr(activations_mod, "ACT2FN", None) is None:
        activations_mod.ACT2FN = {}

    # Ensure PreTrainedModel exists and is a class on modeling_utils namespace.
    # In some broken envs, symbol exists but is None. Reload first, then backfill.
    if getattr(modeling_utils, "PreTrainedModel", None) is None:
        try:
            modeling_utils = importlib.reload(modeling_utils)
        except Exception:
            pass
    if getattr(modeling_utils, "PreTrainedModel", None) is None:
        try:
            from transformers import PreTrainedModel  # type: ignore

            if PreTrainedModel is not None:
                modeling_utils.PreTrainedModel = PreTrainedModel
        except Exception:
            pass

    # Backfill transformers.models.bert.configuration_bert.BertConfig when missing.
    try:
        importlib.import_module("transformers.models.bert.configuration_bert")
    except Exception:
        try:
            from transformers import BertConfig  # type: ignore

            mod = types.ModuleType("transformers.models.bert.configuration_bert")
            mod.BertConfig = BertConfig
            sys.modules["transformers.models.bert.configuration_bert"] = mod
        except Exception:
            pass

    # Verify the exact import tuple mmpretrain.blip.language_model expects.
    # If this check fails, C3 should still degrade gracefully (model init fallback),
    # but we print an explicit reason to avoid opaque "NoneType takes no arguments".
    try:
        from transformers.activations import ACT2FN  # noqa: F401
        from transformers.modeling_outputs import (  # noqa: F401
            BaseModelOutputWithPastAndCrossAttentions,
            BaseModelOutputWithPoolingAndCrossAttentions,
            CausalLMOutputWithCrossAttentions,
        )
        from transformers.modeling_utils import (  # noqa: F401
            PreTrainedModel,
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )
        from transformers.models.bert.configuration_bert import BertConfig  # noqa: F401
    except Exception as e:
        print(f"[C3 Pose] transformers compatibility shim warning: {e}")


_patch_transformers_generation_compat()

try:
    from mmpose.apis import inference_topdown, init_model as init_pose_model
    from mmdet.apis import inference_detector, init_detector

    _MMPOSE_AVAILABLE = True
except ImportError as e:
    _MMPOSE_AVAILABLE = False
    print(f"Warning: mmpose or mmdet is not installed properly ({e}). C3 Pose will not work.")

try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError as e:
    _YOLO_AVAILABLE = False
    YOLO = None
    print(f"Warning: ultralytics is not installed properly ({e}). C3 person verifier will be disabled.")


# COCO-17 keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4

# Tags that strongly imply people are present.
POSITIVE_PERSON_WORDS = {
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "face",
    "portrait",
    "selfie",
    "couple",
    "family",
    "crowd",
    "team",
    "group",
    "friends",
    "meeting",
    "audience",
    "twins",
    "men",
    "women",
    "girls",
    "boys",
    "kids",
    "children",
    "parents",
    "headshot",
    "child",
    "baby",
    "gentleman",
    "lady",
    "guy",
    "bride",
    "groom",
    "model",
}

POSITIVE_PERSON_PHRASES = {
    "one person",
    "two people",
    "three people",
    "full body",
    "upper body",
}

# If explicit non-person terms appear and no positive people cue exists, skip C3.
NON_PERSON_HINT_WORDS = {
    "animal",
    "bird",
    "parrot",
    "cat",
    "dog",
    "fish",
    "insect",
    "wildlife",
    "pet",
    "mammal",
    "reptile",
    "flower",
    "plant",
    "food",
    "furniture",
    "interior",
    "room",
    "bedroom",
    "architecture",
    "landscape",
    "nature",
    "object",
}


def normalize_tags(tags: Optional[Sequence[Any]]) -> List[str]:
    if tags is None:
        return []
    out: List[str] = []
    for t in tags:
        s = str(t).strip().lower()
        if s:
            out.append(s)
    return out


def should_run_pose_by_tags(tags: Optional[Sequence[Any]]) -> bool:
    """
    Conservative tag gate for expensive pose inference.
    - Run if clear people cue exists.
    - Skip if only non-person cues are present.
    - Default skip when cues are absent (quality/stability first).
    """
    tt = normalize_tags(tags)
    if not tt:
        return False

    text = " ".join(tt)
    has_positive = any(t in POSITIVE_PERSON_WORDS for t in tt) or any(p in text for p in POSITIVE_PERSON_PHRASES)
    if has_positive:
        return True

    has_non_person = any(t in NON_PERSON_HINT_WORDS for t in tt)
    if has_non_person:
        return False

    return False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _clamp_box_xyxy(box: Sequence[float], w: int, h: int) -> List[float]:
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


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    uni = a_area + b_area - inter
    if uni <= 0.0:
        return 0.0
    return inter / uni


def _point_in_box(x: float, y: float, box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def candidate_matches_person_prior(
    det_box: Sequence[float],
    person_boxes: Sequence[Sequence[float]],
    min_iou: float = 0.10,
) -> bool:
    """
    Match C3 detector bbox against YOLO person priors.
    Uses IoU + center-in-box checks to keep recall on nested boxes.
    """
    if not person_boxes:
        return False

    dx1, dy1, dx2, dy2 = [float(v) for v in det_box]
    dcx, dcy = 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)

    for pb in person_boxes:
        if _bbox_iou_xyxy(det_box, pb) >= float(min_iou):
            return True
        px1, py1, px2, py2 = [float(v) for v in pb]
        pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
        if _point_in_box(dcx, dcy, pb) or _point_in_box(pcx, pcy, det_box):
            return True
    return False


def filter_candidate_boxes_with_person_prior(
    det_boxes: np.ndarray,
    person_boxes: Sequence[Sequence[float]],
    min_iou: float = 0.10,
) -> np.ndarray:
    if det_boxes is None or len(det_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if not person_boxes:
        return np.zeros((0, 4), dtype=np.float32)

    keep: List[np.ndarray] = []
    for box in det_boxes:
        if candidate_matches_person_prior(box.tolist(), person_boxes, min_iou=min_iou):
            keep.append(box)

    if not keep:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(keep, axis=0).astype(np.float32)


def _get_kp(kps: List[List[float]], idx: int, min_conf: float = 0.05) -> Optional[Tuple[float, float, float]]:
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


def _default_face_landmarks(bbox: List[float]) -> List[List[float]]:
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


def derive_face_from_keypoints(
    keypoints: List[List[float]],
    image_w: int,
    image_h: int,
) -> Optional[Dict[str, Any]]:
    face_points: List[Tuple[float, float, float]] = []
    for idx in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR):
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
        [min_x - pad_x, min_y - pad_y_top, max_x + pad_x, max_y + pad_y_bot],
        image_w,
        image_h,
    )

    landmarks = _default_face_landmarks(bbox)
    idx_to_lm = {
        KP_NOSE: 0,
        KP_LEFT_EYE: 1,
        KP_RIGHT_EYE: 2,
        KP_LEFT_EAR: 3,
        KP_RIGHT_EAR: 4,
    }
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


def estimate_headpose_gaze_proxy(face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
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
    if len(lm) < 3:
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


def enrich_pose_item_from_keypoints(
    pose_item: Dict[str, Any],
    image_w: int,
    image_h: int,
) -> Dict[str, Any]:
    out = dict(pose_item)
    keypoints = pose_item.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) == 0:
        out.setdefault("face", None)
        out.setdefault("headpose_gaze", estimate_headpose_gaze_proxy(None))
        return out

    face = derive_face_from_keypoints(keypoints=keypoints, image_w=image_w, image_h=image_h)
    out["face"] = face
    out["headpose_gaze"] = estimate_headpose_gaze_proxy(face)
    return out


class PoseFeatureExtractor:
    """
    ViTPose top-down C3 extractor with robust non-human suppression.

    Key robustness step:
    - mmdet person bbox candidates are verified against YOLO(person) priors
      (same detector family used in C2 pipeline), reducing non-human false positives.
    """

    def __init__(
        self,
        det_config,
        det_ckpt,
        pose_config,
        pose_ckpt,
        device=None,
        priority="high_efficiency",
        person_verify_model: str = "yolov8n.pt",
        person_verify_conf: float = 0.20,
        person_verify_iou: float = 0.10,
        person_verify_strict: bool = True,
    ):
        self.device = device if device else "cuda:0"
        self.priority = priority
        self.person_verify_conf = float(person_verify_conf)
        self.person_verify_iou = float(person_verify_iou)
        self.person_verify_strict = bool(person_verify_strict)

        print(f"[C3 Pose] Initializing Det({det_ckpt}) & ViTPose({pose_ckpt}) on {self.device} (Mode: {self.priority})...")

        if not _MMPOSE_AVAILABLE:
            print("mmpose/mmdet not available. C3 Pose disabled.")
            self.detector = None
            self.pose_estimator = None
        else:
            try:
                self.detector = init_detector(det_config, det_ckpt, device=self.device)
                try:
                    from mmpose.utils import adapt_mmdet_pipeline

                    self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
                except ImportError:
                    pass
                self.pose_estimator = init_pose_model(pose_config, pose_ckpt, device=self.device)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Failed to load Pose models: {e}")
                self.detector = None
                self.pose_estimator = None

        self.person_verifier = None
        if _YOLO_AVAILABLE:
            try:
                self.person_verifier = YOLO(person_verify_model)
                self.person_verifier.to(self.device)
                print(f"[C3 Pose] Person verifier enabled: {person_verify_model}")
            except Exception as e:
                self.person_verifier = None
                print(f"[C3 Pose] Person verifier init failed: {e}")
        else:
            print("[C3 Pose] Person verifier disabled (ultralytics not available).")

    def _run_person_verifier(self, img_bgr: np.ndarray) -> List[List[float]]:
        if self.person_verifier is None:
            return []
        try:
            res = self.person_verifier(img_bgr, conf=self.person_verify_conf, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            classes = res.boxes.cls.cpu().numpy().astype(np.int32)
            person_boxes = boxes[classes == 0]
            return person_boxes.tolist() if len(person_boxes) else []
        except Exception:
            return []

    def process_image(
        self,
        image: Image.Image,
        run_anyway: bool = False,
        tags: Optional[Sequence[Any]] = None,
        person_prior_boxes: Optional[Sequence[Sequence[float]]] = None,
    ):
        """
        Returns list of pose dicts. Empty list means C3 skipped/no reliable person.
        """
        results_out = []
        if self.detector is None or self.pose_estimator is None:
            return results_out

        has_person_prior_hint = bool(person_prior_boxes)
        if not run_anyway and (not has_person_prior_hint) and (not should_run_pose_by_tags(tags)):
            return results_out

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        det_result = inference_detector(self.detector, img_bgr)
        pred_instances = det_result.pred_instances.cpu().numpy()

        scores = pred_instances.scores
        labels = pred_instances.labels if "labels" in pred_instances else np.zeros_like(scores)

        person_indices = scores > 0.2
        if "labels" in pred_instances:
            person_indices &= labels == 0

        bboxes = pred_instances.bboxes[person_indices]
        if len(bboxes) == 0:
            return results_out

        # Additional person prior verification to suppress non-human detections.
        prior_boxes: List[List[float]] = []
        if person_prior_boxes is not None:
            for pb in person_prior_boxes:
                if not isinstance(pb, (list, tuple)) or len(pb) != 4:
                    continue
                prior_boxes.append(_clamp_box_xyxy([float(v) for v in pb], w, h))
        else:
            prior_boxes = self._run_person_verifier(img_bgr)

        verify_enabled = (person_prior_boxes is not None) or (self.person_verifier is not None)
        if verify_enabled:
            if not prior_boxes and self.person_verify_strict:
                return results_out
            if prior_boxes:
                bboxes = filter_candidate_boxes_with_person_prior(
                    det_boxes=np.asarray(bboxes, dtype=np.float32),
                    person_boxes=prior_boxes,
                    min_iou=self.person_verify_iou,
                )
                if len(bboxes) == 0 and self.person_verify_strict:
                    return results_out

        if len(bboxes) == 0:
            return results_out

        pose_results = inference_topdown(self.pose_estimator, img_bgr, bboxes, bbox_format="xyxy")

        for p_res in pose_results:
            p_inst = p_res.pred_instances
            if "keypoints" not in p_inst or len(p_inst.keypoints) == 0:
                continue

            kb = p_inst.keypoints[0]
            ks = p_inst.keypoint_scores[0]
            bb = p_inst.bboxes[0]
            bs = float(p_inst.bbox_scores[0])

            kps_out: List[List[float]] = []
            for i in range(len(kb)):
                kps_out.append([float(kb[i][0]), float(kb[i][1]), float(ks[i])])

            item = {
                "bbox": [float(x) for x in bb],
                "score": bs,
                "keypoints": kps_out,
            }
            item = enrich_pose_item_from_keypoints(item, image_w=w, image_h=h)
            results_out.append(item)

        return results_out
