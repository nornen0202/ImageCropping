"""
C2 segmentation feature extractor.

Modes:
- high_efficiency: YOLOv8 detections + EfficientViT-SAM mask prompting.
- quality_first: semantic-prior selection with YOLO + SAM2 predictor,
  then AMG fallback for difficult cases.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# EfficientViT-SAM (high efficiency mode)
try:
    try:
        from efficientvit.sam_model_zoo import create_sam_model
    except ImportError:
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model as create_sam_model
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
except ImportError as e:
    create_sam_model = None
    EfficientViTSamPredictor = None
    print(f"Warning: efficientvit import failed ({e}). C2 high_efficiency is disabled.")

# SAM2 (quality-first mode)
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None
    SAM2ImagePredictor = None
    print(f"Warning: sam2 import failed ({e}). C2 quality_first is disabled.")


def encode_rle(mask: np.ndarray) -> str:
    """Encode HxW binary mask into simple run-length encoding."""
    pixels = mask.astype(np.uint8).flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    x, y, w, h = bbox_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def _bbox_area_xyxy(b: List[float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _clamp_xyxy(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = b
    x1 = float(max(0.0, min(x1, w - 1)))
    y1 = float(max(0.0, min(y1, h - 1)))
    x2 = float(max(0.0, min(x2, w)))
    y2 = float(max(0.0, min(y2, h)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _bbox_center_xyxy(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _thirds_points(w: int, h: int) -> List[Tuple[float, float]]:
    return [
        (w / 3.0, h / 3.0),
        (2.0 * w / 3.0, h / 3.0),
        (w / 3.0, 2.0 * h / 3.0),
        (2.0 * w / 3.0, 2.0 * h / 3.0),
    ]


class SegFeatureExtractor:
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        sam_model: str = "l2",
        device: Optional[str] = None,
        priority: str = "high_efficiency",
        weights_dir: str = "",
        qf_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.priority = priority
        self.yolo = None
        self.sam = None
        self.predictor = None
        self.sam2_model = None
        self.mask_generator = None
        self.sam2_predictor = None

        self.qf_cfg: Dict[str, Any] = {
            # semantic prior
            "use_yolo_prior": True,
            "yolo_conf": 0.25,
            "preferred_class_ids": [0],  # COCO person=0
            "min_det_area_ratio": 0.01,
            "min_det_cover": 0.30,
            "min_any_det_cover": 0.20,
            "merge_multi_det": True,
            # SAM mask filtering
            "min_mask_area_ratio": 0.003,
            "max_mask_area_ratio": 0.92,
            "min_pred_iou": 0.0,
            "min_stability": 0.0,
            # fallback scoring
            "w_area": 0.50,
            "w_center": 0.10,
            "w_thirds": 0.20,
            "w_saliency": 0.35,
            "w_semantic": 0.70,
            "w_part_penalty": 0.35,
            "w_border_penalty": 0.20,
            # containment
            "containment_thr": 0.92,
            "containment_area_ratio": 1.8,
            # AMG memory controls
            "amg_points_per_side": 16,
            "amg_points_per_batch": 32,
            "amg_crop_n_layers": 0,
            # debug
            "debug": False,
        }
        if qf_cfg:
            self.qf_cfg.update(qf_cfg)

        if not weights_dir:
            here = os.path.dirname(os.path.abspath(__file__))
            for cand in [
                os.path.normpath(os.path.join(here, "../../weights")),
                os.path.normpath(os.path.join(here, "../scripts/weights")),
            ]:
                if os.path.isdir(cand):
                    weights_dir = cand
                    break
            else:
                weights_dir = os.path.normpath(os.path.join(here, "../../weights"))
        self._weights_dir = weights_dir

        print(f"[C2 Seg] Initializing on {self.device} (mode={self.priority})")

        if self.priority == "high_efficiency":
            self._init_yolo(yolo_model)
            self._init_efficientvit(sam_model)
        elif self.priority == "quality_first":
            if self.qf_cfg.get("use_yolo_prior", True):
                self._init_yolo(yolo_model)
            self._init_sam2()

    def _init_yolo(self, yolo_model: str) -> None:
        try:
            self.yolo = YOLO(yolo_model)
            self.yolo.to(self.device)
        except Exception as e:
            self.yolo = None
            print(f"[C2 Seg] YOLO init failed: {e}")

    def _init_efficientvit(self, sam_model: str) -> None:
        if create_sam_model is None:
            print("[C2 Seg] EfficientViT-SAM unavailable.")
            return
        try:
            self.sam = create_sam_model(sam_model, True).to(self.device).eval()
            self.predictor = EfficientViTSamPredictor(self.sam)
        except Exception as e:
            print(f"[C2 Seg] EfficientViT-SAM init failed: {e}")
            self.sam = None
            self.predictor = None

    def _init_sam2(self) -> None:
        if build_sam2 is None:
            print("[C2 Seg] SAM2 unavailable.")
            return
        try:
            cfg_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
            ckpt_path = os.path.join(self._weights_dir, "sam2.1_hiera_large.pt")
            if not os.path.exists(ckpt_path):
                print(f"[C2 Seg] SAM2 weights missing: {ckpt_path}")
                self._download_sam2_weights(ckpt_path)

            self.sam2_model = build_sam2(cfg_name, ckpt_path, device=self.device, apply_postprocessing=False)

            amg_kwargs = {
                "points_per_side": int(self.qf_cfg.get("amg_points_per_side", 16)),
                "points_per_batch": int(self.qf_cfg.get("amg_points_per_batch", 32)),
                "crop_n_layers": int(self.qf_cfg.get("amg_crop_n_layers", 0)),
            }
            self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model, **amg_kwargs)

            if SAM2ImagePredictor is not None:
                self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        except Exception as e:
            print(f"[C2 Seg] SAM2 init failed: {e}")
            self.sam2_model = None
            self.mask_generator = None
            self.sam2_predictor = None

    @staticmethod
    def _download_sam2_weights(dest_path: str) -> None:
        import urllib.request

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt"
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception as e:
            print(f"[C2 Seg] SAM2 weight download failed: {e}")

    def _run_yolo(self, img_bgr: np.ndarray, conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.yolo is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )
        try:
            yolo_res = self.yolo(img_bgr, conf=conf, verbose=False)[0]
            return (
                yolo_res.boxes.xyxy.cpu().numpy().astype(np.float32),
                yolo_res.boxes.cls.cpu().numpy().astype(np.int32),
                yolo_res.boxes.conf.cpu().numpy().astype(np.float32),
            )
        except Exception:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )

    @staticmethod
    def _reset_sam2_predictor(predictor: Any) -> None:
        if predictor is None:
            return
        if hasattr(predictor, "reset_predictor"):
            predictor.reset_predictor()
        elif hasattr(predictor, "reset_image"):
            predictor.reset_image()

    @staticmethod
    def _mask_bbox(mask_bool: np.ndarray, w: int, h: int) -> Optional[List[float]]:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None
        return _clamp_xyxy([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], w, h)

    @staticmethod
    def _mask_box_overlap_scores(mask_bool: np.ndarray, det_box: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = det_box
        x1i = int(max(0, math.floor(x1)))
        y1i = int(max(0, math.floor(y1)))
        x2i = int(max(x1i + 1, math.ceil(x2)))
        y2i = int(max(y1i + 1, math.ceil(y2)))

        det_area = float(max(1, (x2i - x1i) * (y2i - y1i)))
        inside = float(mask_bool[y1i:y2i, x1i:x2i].sum())
        mask_area = float(max(1, int(mask_bool.sum())))

        det_cover = inside / det_area
        inside_ratio = inside / mask_area
        return det_cover, inside_ratio

    def _pick_best_pred_mask(
        self,
        pred_masks: np.ndarray,
        det_box: List[float],
        img_area: int,
    ) -> Tuple[Optional[np.ndarray], float, float]:
        pred_masks = np.asarray(pred_masks)
        if pred_masks.ndim == 2:
            pred_masks = pred_masks[None, ...]
        if pred_masks.size == 0:
            return None, 0.0, -float("inf")

        min_ar = float(self.qf_cfg.get("min_mask_area_ratio", 0.003))
        max_ar = float(self.qf_cfg.get("max_mask_area_ratio", 0.92))

        best_mask = None
        best_cover = 0.0
        best_score = -float("inf")

        for m in pred_masks:
            mask_bool = np.asarray(m).astype(bool)
            area_ratio = float(mask_bool.sum()) / float(max(1, img_area))
            if area_ratio < min_ar or area_ratio > max_ar:
                continue
            det_cover, inside_ratio = self._mask_box_overlap_scores(mask_bool, det_box)
            if det_cover <= 0.0:
                continue

            score = 0.70 * det_cover + 0.20 * inside_ratio + 0.10 * math.sqrt(max(area_ratio, 1e-9))
            if score > best_score:
                best_score = score
                best_cover = det_cover
                best_mask = mask_bool

        return best_mask, best_cover, best_score

    def _segment_from_detection_boxes(
        self,
        img_rgb: np.ndarray,
        w: int,
        h: int,
        img_area: int,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        min_cover: float,
        merge_multi_det: bool,
    ) -> Tuple[Optional[np.ndarray], float, List[int]]:
        if self.sam2_predictor is None or det_boxes.shape[0] == 0:
            return None, 0.0, []

        selected_masks: List[np.ndarray] = []
        selected_scores: List[float] = []
        selected_indices: List[int] = []

        self.sam2_predictor.set_image(img_rgb)
        try:
            for i in range(det_boxes.shape[0]):
                det_box = _clamp_xyxy([float(x) for x in det_boxes[i].tolist()], w, h)
                try:
                    pred_masks, _, _ = self.sam2_predictor.predict(
                        box=np.array(det_box, dtype=np.float32)[None, :],
                        multimask_output=True,
                    )
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
                best_mask, best_cover, best_mask_score = self._pick_best_pred_mask(pred_masks, det_box, img_area)
                if best_mask is None or best_cover < min_cover:
                    continue

                det_score = float(det_scores[i]) if len(det_scores) > i else 0.0
                selected_masks.append(best_mask)
                selected_scores.append(0.5 * det_score + 0.5 * best_mask_score)
                selected_indices.append(i)

                if not merge_multi_det:
                    break
        finally:
            self._reset_sam2_predictor(self.sam2_predictor)

        if not selected_masks:
            return None, 0.0, []

        union_mask = np.zeros((h, w), dtype=bool)
        for m in selected_masks:
            union_mask |= m

        if int(union_mask.sum()) == 0:
            return None, 0.0, []

        return union_mask, float(max(selected_scores)), selected_indices

    def _compute_containment_flags(self, masks: List[Dict[str, Any]], w: int, h: int) -> List[bool]:
        if len(masks) == 0:
            return []

        boxes = [_clamp_xyxy(_xywh_to_xyxy(m["bbox"]), w, h) for m in masks]
        areas = np.array([_bbox_area_xyxy(b) for b in boxes], dtype=np.float32)
        order = np.argsort(-areas)  # big -> small

        is_part = [False] * len(masks)
        thr = float(self.qf_cfg.get("containment_thr", 0.92))
        area_ratio_thr = float(self.qf_cfg.get("containment_area_ratio", 1.8))

        for ii in range(len(order) - 1, -1, -1):
            i = int(order[ii])
            ai = float(areas[i])
            if ai <= 1.0:
                continue
            for jj in range(0, ii):
                j = int(order[jj])
                aj = float(areas[j])
                if aj < ai * area_ratio_thr:
                    break
                inter = _bbox_intersection_area_xyxy(boxes[i], boxes[j])
                contain = inter / (ai + 1e-6)
                if contain >= thr:
                    is_part[i] = True
                    break
        return is_part

    @staticmethod
    def _safe_generate_masks(mask_generator: Any, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
        if mask_generator is None:
            return []
        try:
            return mask_generator.generate(img_rgb)
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return []
            raise

    @staticmethod
    def _build_saliency_foreground(img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        saliency_fg = np.zeros((h, w), dtype=bool)
        if not hasattr(cv2, "saliency"):
            return saliency_fg
        try:
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, saliency_map = saliency.computeSaliency(img_bgr)
            if not ok:
                return saliency_fg
            saliency_map = (saliency_map * 255.0).astype(np.uint8)
            _, fg = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            saliency_fg = fg > 127
        except Exception:
            pass
        return saliency_fg

    def _saliency_mask_fallback(self, img_bgr: np.ndarray, img_area: int) -> Optional[np.ndarray]:
        saliency_fg = self._build_saliency_foreground(img_bgr)
        if not saliency_fg.any():
            return None

        min_ar = float(self.qf_cfg.get("min_mask_area_ratio", 0.003))
        max_ar = float(self.qf_cfg.get("max_mask_area_ratio", 0.92))
        min_ar_loose = min_ar * 0.5

        u8 = saliency_fg.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)

        best_label = -1
        best_area = 0
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            area_ratio = area / float(max(1, img_area))
            if area_ratio < min_ar_loose or area_ratio > max_ar:
                continue
            if area > best_area:
                best_area = area
                best_label = i

        if best_label > 0:
            return labels == best_label

        full_area_ratio = float(u8.sum()) / float(max(1, img_area))
        if min_ar_loose <= full_area_ratio <= max_ar:
            return saliency_fg
        return None

    def _build_output_from_mask(
        self,
        mask_bool: np.ndarray,
        w: int,
        h: int,
        class_id: int,
        score: float,
        dbg: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        box_xyxy = self._mask_bbox(mask_bool, w, h)
        if box_xyxy is None:
            return None
        out: Dict[str, Any] = {
            "box": [float(x) for x in box_xyxy],
            "class_id": int(class_id),
            "score": float(score),
            "mask_rle": encode_rle(mask_bool),
            "area": int(mask_bool.sum()),
        }
        if self.qf_cfg.get("debug", False) and dbg is not None:
            out["_dbg"] = dbg
        return out

    def _filter_amg_masks(self, masks: List[Dict[str, Any]], img_area: int) -> List[Dict[str, Any]]:
        min_ar = float(self.qf_cfg.get("min_mask_area_ratio", 0.003))
        max_ar = float(self.qf_cfg.get("max_mask_area_ratio", 0.92))
        min_pred_iou = float(self.qf_cfg.get("min_pred_iou", 0.0))
        min_stability = float(self.qf_cfg.get("min_stability", 0.0))

        out: List[Dict[str, Any]] = []
        for m in masks:
            area_ratio = float(m.get("area", 0.0)) / float(max(1, img_area))
            if area_ratio < min_ar or area_ratio > max_ar:
                continue
            if float(m.get("predicted_iou", 1.0)) < min_pred_iou:
                continue
            if float(m.get("stability_score", 1.0)) < min_stability:
                continue
            out.append(m)
        return out

    def _semantic_cover_from_dets(self, mask_bbox: List[float], det_boxes: np.ndarray, w: int, h: int) -> float:
        if det_boxes.shape[0] == 0:
            return 0.0
        covers: List[float] = []
        for i in range(det_boxes.shape[0]):
            db = _clamp_xyxy([float(x) for x in det_boxes[i].tolist()], w, h)
            inter = _bbox_intersection_area_xyxy(mask_bbox, db)
            covers.append(inter / (_bbox_area_xyxy(db) + 1e-6))
        return float(max(covers)) if covers else 0.0

    def process_image(self, image: Image.Image, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        results_out: List[Dict[str, Any]] = []

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_area = h * w

        if self.priority == "high_efficiency":
            if self.yolo is None or self.predictor is None:
                return results_out

            yolo_res = self.yolo(img_bgr, conf=conf_threshold, verbose=False)[0]
            boxes = yolo_res.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = yolo_res.boxes.conf.cpu().numpy().astype(np.float32)
            class_ids = yolo_res.boxes.cls.cpu().numpy().astype(np.int32)

            if len(boxes) == 0:
                return results_out

            self.predictor.set_image(img_rgb)
            for i, box in enumerate(boxes):
                mask, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,
                )
                mask_2d = mask[0].astype(bool)
                results_out.append(
                    {
                        "box": [float(x) for x in box],
                        "class_id": int(class_ids[i]),
                        "score": float(scores[i]),
                        "mask_rle": encode_rle(mask_2d),
                        "area": int(mask_2d.sum()),
                    }
                )
            self.predictor.reset_image()
            return results_out

        if self.priority != "quality_first":
            return results_out

        # 1) YOLO detections (for semantic prior and semantic fallback)
        yolo_conf = float(self.qf_cfg.get("yolo_conf", conf_threshold))
        det_boxes, det_cls, det_scores = self._run_yolo(img_bgr, conf=yolo_conf)
        min_det_area_ratio = float(self.qf_cfg.get("min_det_area_ratio", 0.01))
        valid_det_idx = []
        for i in range(det_boxes.shape[0]):
            db = _clamp_xyxy([float(x) for x in det_boxes[i].tolist()], w, h)
            if _bbox_area_xyxy(db) / float(max(1, img_area)) >= min_det_area_ratio:
                valid_det_idx.append(i)

        det_boxes_v = det_boxes[valid_det_idx] if valid_det_idx else np.zeros((0, 4), dtype=np.float32)
        det_cls_v = det_cls[valid_det_idx] if valid_det_idx else np.zeros((0,), dtype=np.int32)
        det_scores_v = det_scores[valid_det_idx] if valid_det_idx else np.zeros((0,), dtype=np.float32)

        # 2) Preferred-class semantic prior (person-first by default)
        preferred_ids = set(int(x) for x in self.qf_cfg.get("preferred_class_ids", [0]))
        preferred_idx = [i for i in range(det_boxes_v.shape[0]) if int(det_cls_v[i]) in preferred_ids]
        merge_multi = bool(self.qf_cfg.get("merge_multi_det", True))
        min_det_cover = float(self.qf_cfg.get("min_det_cover", 0.30))

        if preferred_idx and self.sam2_predictor is not None:
            p_boxes = det_boxes_v[preferred_idx]
            p_scores = det_scores_v[preferred_idx]
            union_mask, score, selected = self._segment_from_detection_boxes(
                img_rgb=img_rgb,
                w=w,
                h=h,
                img_area=img_area,
                det_boxes=p_boxes,
                det_scores=p_scores,
                min_cover=min_det_cover,
                merge_multi_det=merge_multi,
            )
            if union_mask is not None:
                out = self._build_output_from_mask(
                    mask_bool=union_mask,
                    w=w,
                    h=h,
                    class_id=int(self.qf_cfg.get("preferred_class_ids", [0])[0]),
                    score=float(score),
                    dbg={"via": "preferred_det_prompt", "det_count": len(preferred_idx), "selected_det_count": len(selected)},
                )
                if out is not None:
                    results_out.append(out)
                    return results_out

        # 3) Any-class semantic fallback to reduce background picks
        if det_boxes_v.shape[0] > 0 and self.sam2_predictor is not None:
            quality = []
            for i in range(det_boxes_v.shape[0]):
                db = _clamp_xyxy([float(x) for x in det_boxes_v[i].tolist()], w, h)
                area_ratio = _bbox_area_xyxy(db) / float(max(1, img_area))
                quality.append(float(det_scores_v[i]) * math.sqrt(max(area_ratio, 1e-9)))
            top_i = int(np.argmax(np.asarray(quality))) if quality else 0

            min_any_det_cover = float(self.qf_cfg.get("min_any_det_cover", 0.20))
            union_mask, score, selected = self._segment_from_detection_boxes(
                img_rgb=img_rgb,
                w=w,
                h=h,
                img_area=img_area,
                det_boxes=det_boxes_v[top_i : top_i + 1],
                det_scores=det_scores_v[top_i : top_i + 1],
                min_cover=min_any_det_cover,
                merge_multi_det=False,
            )
            if union_mask is not None:
                out = self._build_output_from_mask(
                    mask_bool=union_mask,
                    w=w,
                    h=h,
                    class_id=int(det_cls_v[top_i]),
                    score=float(score),
                    dbg={"via": "any_det_prompt", "det_class_id": int(det_cls_v[top_i]), "selected_det_count": len(selected)},
                )
                if out is not None:
                    results_out.append(out)
                    return results_out

        # 4) AMG fallback
        masks = self._safe_generate_masks(self.mask_generator, img_rgb)
        if not masks:
            saliency_mask = self._saliency_mask_fallback(img_bgr, img_area)
            if saliency_mask is not None:
                out = self._build_output_from_mask(
                    mask_bool=saliency_mask,
                    w=w,
                    h=h,
                    class_id=-1,
                    score=0.0,
                    dbg={"via": "saliency_fallback"},
                )
                if out is not None:
                    results_out.append(out)
            return results_out

        masks_f = self._filter_amg_masks(masks, img_area)
        if not masks_f:
            saliency_mask = self._saliency_mask_fallback(img_bgr, img_area)
            if saliency_mask is not None:
                out = self._build_output_from_mask(
                    mask_bool=saliency_mask,
                    w=w,
                    h=h,
                    class_id=-1,
                    score=0.0,
                    dbg={"via": "saliency_fallback"},
                )
                if out is not None:
                    results_out.append(out)
            return results_out

        part_flags = self._compute_containment_flags(masks_f, w, h)
        saliency_fg = self._build_saliency_foreground(img_bgr)

        best_mask_info = None
        best_score = -float("inf")

        img_center = (w / 2.0, h / 2.0)
        max_dist = math.sqrt(img_center[0] ** 2 + img_center[1] ** 2) + 1e-6
        thirds_pts = _thirds_points(w, h)

        w_area = float(self.qf_cfg.get("w_area", 0.50))
        w_center = float(self.qf_cfg.get("w_center", 0.10))
        w_thirds = float(self.qf_cfg.get("w_thirds", 0.20))
        w_saliency = float(self.qf_cfg.get("w_saliency", 0.35))
        w_semantic = float(self.qf_cfg.get("w_semantic", 0.70))
        w_part = float(self.qf_cfg.get("w_part_penalty", 0.35))
        w_border = float(self.qf_cfg.get("w_border_penalty", 0.20))

        for idx, m in enumerate(masks_f):
            mask_bool = m["segmentation"].astype(bool)
            mask_bbox = _clamp_xyxy(_xywh_to_xyxy(m["bbox"]), w, h)
            area_ratio = float(mask_bool.sum()) / float(max(1, img_area))
            cx, cy = _bbox_center_xyxy(mask_bbox)

            center_score = 1.0 - (_dist((cx, cy), img_center) / max_dist)
            thirds_score = 1.0 - (min(_dist((cx, cy), t) for t in thirds_pts) / max_dist)
            area_score = math.sqrt(max(area_ratio, 1e-9))

            part_penalty = 1.0 if part_flags[idx] else 0.0
            x1, y1, x2, y2 = mask_bbox
            border_touch = (
                (1.0 if x1 <= 1.0 else 0.0)
                + (1.0 if y1 <= 1.0 else 0.0)
                + (1.0 if x2 >= float(w - 1) else 0.0)
                + (1.0 if y2 >= float(h - 1) else 0.0)
            ) / 4.0

            if saliency_fg.any():
                inter = float(np.logical_and(mask_bool, saliency_fg).sum())
                union = float(np.logical_or(mask_bool, saliency_fg).sum()) + 1e-6
                saliency_score = inter / union
            else:
                saliency_score = 0.0

            semantic_score = self._semantic_cover_from_dets(mask_bbox, det_boxes_v, w, h)

            score = (
                w_area * area_score
                + w_center * center_score
                + w_thirds * thirds_score
                + w_saliency * saliency_score
                + w_semantic * semantic_score
                - w_part * part_penalty
                - w_border * border_touch
            )

            if score > best_score:
                best_score = score
                best_mask_info = m

        if best_mask_info is None:
            saliency_mask = self._saliency_mask_fallback(img_bgr, img_area)
            if saliency_mask is not None:
                out = self._build_output_from_mask(
                    mask_bool=saliency_mask,
                    w=w,
                    h=h,
                    class_id=-1,
                    score=0.0,
                    dbg={"via": "saliency_fallback"},
                )
                if out is not None:
                    results_out.append(out)
            return results_out

        mask_2d = best_mask_info["segmentation"].astype(bool)
        out = self._build_output_from_mask(
            mask_bool=mask_2d,
            w=w,
            h=h,
            class_id=-1,
            score=float(best_score),
            dbg={"via": "amg_fallback", "num_masks": len(masks_f)},
        )
        if out is not None:
            results_out.append(out)

        return results_out
