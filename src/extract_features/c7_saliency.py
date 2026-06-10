from __future__ import annotations

import base64
import importlib.util
import math
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize as tv_normalize
from transformers import AutoModelForImageSegmentation


torch.set_float32_matmul_precision("high")

BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet"
ISNET_REPO_ID = "NimaBoscarino/IS-Net_DIS-general-use"
ISNET_WEIGHTS_FILE = "isnet-general-use.pth"
ISNET_MODEL_URL = "https://raw.githubusercontent.com/xuebinqin/DIS/main/IS-Net/models/isnet.py"
ISNET_MODEL_FILENAME = "isnet_model.py"
SUPPORT_GRID_SIZE = 24


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> List[float]:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    x1 = _clamp(float(x1), 0.0, w)
    y1 = _clamp(float(y1), 0.0, h)
    x2 = _clamp(float(x2), 0.0, w)
    y2 = _clamp(float(y2), 0.0, h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _norm_xyxy(box: List[float], width: int, height: int) -> List[float]:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    return [
        round(_clamp(box[0] / w, 0.0, 1.0), 6),
        round(_clamp(box[1] / h, 0.0, 1.0), 6),
        round(_clamp(box[2] / w, 0.0, 1.0), 6),
        round(_clamp(box[3] / h, 0.0, 1.0), 6),
    ]


def _resolve_device(device: Optional[str] = None) -> str:
    if device and str(device).strip().lower() not in {"", "auto"}:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _ensure_isnet_module(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_file = cache_dir / ISNET_MODEL_FILENAME
    if not model_file.exists():
        urllib.request.urlretrieve(ISNET_MODEL_URL, str(model_file))
    module_name = "_codex_isnet_model"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build module spec for IS-Net model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prob_to_mask(prob_map: np.ndarray) -> Tuple[np.ndarray, float]:
    prob_map = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    u8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.2, sigmaY=1.2)
    _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = float(np.max([1.0, fg[fg > 0].min() if np.any(fg > 0) else 0.0]) / 255.0) if np.any(fg > 0) else 0.0
    mask = fg > 127
    fg_ratio = float(mask.mean()) if mask.size > 0 else 0.0
    if fg_ratio > 0.92:
        thr = max(0.45, float(np.quantile(prob_map, 0.75)))
        mask = prob_map >= thr
        threshold = thr
    elif fg_ratio < 0.003:
        thr = max(0.20, float(np.quantile(prob_map, 0.90)))
        mask = prob_map >= thr
        threshold = thr
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_u8 > 127, float(threshold)


def _bbox_from_mask(mask: np.ndarray, width: int, height: int) -> Optional[List[float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return _clip_xyxy(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, width, height)


def _weighted_centroid(prob_map: np.ndarray) -> Tuple[float, float]:
    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    total = float(prob.sum())
    if total <= 1e-8:
        return 0.5, 0.5
    h, w = prob.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((prob * xs).sum() / total) / max(1.0, float(w))
    cy = float((prob * ys).sum() / total) / max(1.0, float(h))
    return _clamp(cx, 0.0, 1.0), _clamp(cy, 0.0, 1.0)


def _dispersion_score(prob_map: np.ndarray) -> float:
    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    total = float(prob.sum())
    if total <= 1e-8:
        return 1.0
    h, w = prob.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    mx = float((prob * xs).sum() / total)
    my = float((prob * ys).sum() / total)
    var_x = float((prob * ((xs - mx) ** 2)).sum() / total) / max(1.0, float(w) ** 2)
    var_y = float((prob * ((ys - my) ** 2)).sum() / total) / max(1.0, float(h) ** 2)
    std_norm = math.sqrt(max(0.0, var_x + var_y))
    return _clamp(std_norm / math.sqrt(0.5 ** 2 + 0.5 ** 2), 0.0, 1.0)


def _entropy_norm(prob_map: np.ndarray, bins: int = 32) -> float:
    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    small = cv2.resize(prob, (bins, bins), interpolation=cv2.INTER_AREA)
    mass = small / max(1e-8, float(small.sum()))
    flat = mass.reshape(-1)
    nz = flat[flat > 1e-12]
    if len(nz) == 0:
        return 1.0
    entropy = float(-(nz * np.log(nz)).sum())
    return _clamp(entropy / math.log(float(len(flat))), 0.0, 1.0)


def _summarize_components(mask: np.ndarray, width: int, height: int) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    total_fg = int(u8.sum())
    comps: List[Dict[str, Any]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label_idx]]
        if area <= 0:
            continue
        box = _clip_xyxy(x, y, x + w, y + h, width, height)
        comps.append(
            {
                "bbox_xyxy": box,
                "bbox_norm_xyxy": _norm_xyxy(box, width, height),
                "area_ratio": round(float(area) / float(max(1, width * height)), 6),
                "mass_ratio": round(float(area) / float(max(1, total_fg)), 6),
            }
        )
    comps.sort(key=lambda item: (item["area_ratio"], item["mass_ratio"]), reverse=True)
    top2_union = None
    if comps:
        use = comps[:2]
        x1 = min(item["bbox_xyxy"][0] for item in use)
        y1 = min(item["bbox_xyxy"][1] for item in use)
        x2 = max(item["bbox_xyxy"][2] for item in use)
        y2 = max(item["bbox_xyxy"][3] for item in use)
        top2_union = _clip_xyxy(x1, y1, x2, y2, width, height)
    return comps, top2_union


def _encode_support_map(prob_map: np.ndarray, mask: np.ndarray, grid_size: int = SUPPORT_GRID_SIZE) -> Dict[str, Any]:
    support = np.clip(prob_map.astype(np.float32), 0.0, 1.0) * mask.astype(np.float32)
    if float(support.sum()) <= 1e-8:
        support = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    support_small = cv2.resize(support, (grid_size, grid_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    support_small = np.clip(support_small, 0.0, None)
    support_sum = float(support_small.sum())
    if support_sum > 1e-8:
        support_small /= support_sum
    occ_small = cv2.resize(mask.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    occ_small = (occ_small >= 0.15).astype(np.uint8)
    support_f16 = support_small.astype(np.float16)
    return {
        "support_grid_size": int(grid_size),
        "support_grid_encoding": "f16_base64",
        "support_mass_grid_f16_b64": base64.b64encode(support_f16.tobytes()).decode("ascii"),
        "support_occ_grid_u8_b64": base64.b64encode(occ_small.tobytes()).decode("ascii"),
    }


class SaliencyFeatureExtractor:
    def __init__(
        self,
        *,
        priority: str = "high_efficiency",
        weights_dir: str = "",
        device: Optional[str] = None,
    ) -> None:
        self.priority = str(priority or "high_efficiency").strip().lower()
        self.device = _resolve_device(device)
        self.weights_dir = Path(weights_dir).resolve() if str(weights_dir).strip() else Path.cwd() / "weights"
        self.cache_dir = self.weights_dir / "saliency_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.priority in {"opencv", "opencv_only"}:
            self.requested_backend = "opencv"
        elif self.priority == "quality_first":
            self.requested_backend = "birefnet"
        else:
            self.requested_backend = "isnet"
        self.backend = "none"
        self.model: Optional[torch.nn.Module] = None
        self.birefnet_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._load_requested_backend()

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_requested_backend(self) -> None:
        if self.requested_backend == "opencv":
            order = ["opencv"]
        elif self.requested_backend == "birefnet":
            order = ["birefnet", "isnet", "opencv"]
        else:
            order = ["isnet", "opencv"]
        last_exc: Optional[BaseException] = None
        for backend in order:
            try:
                self._load_backend(backend)
                return
            except BaseException as exc:  # pragma: no cover - exercised in integration, not unit
                last_exc = exc
                self.close()
        if last_exc is not None:
            raise RuntimeError(f"Failed to initialize any saliency backend. last_error={last_exc!r}") from last_exc

    def _load_backend(self, backend: str) -> None:
        backend = backend.strip().lower()
        if backend == "opencv":
            self.model = None
            self.backend = "opencv_saliency"
            return
        if backend == "birefnet":
            model = AutoModelForImageSegmentation.from_pretrained(BIREFNET_REPO_ID, trust_remote_code=True)
            model.eval()
            if self.device.startswith("cuda"):
                model = model.to(self.device).half()
            else:
                model = model.to(self.device)
            self.model = model
            self.backend = "birefnet"
            return
        if backend == "isnet":
            module = _ensure_isnet_module(self.cache_dir / "isnet")
            model_cls = getattr(module, "ISNetDIS")
            model = model_cls()
            weights_path = hf_hub_download(ISNET_REPO_ID, ISNET_WEIGHTS_FILE)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model = model.to(self.device)
            self.model = model
            self.backend = "isnet"
            return
        raise ValueError(f"Unknown saliency backend: {backend}")

    def _fallback_backend(self, reason: str) -> str:
        if self.backend == "birefnet":
            self.close()
            try:
                self._load_backend("isnet")
                return f"{reason}->isnet"
            except BaseException:
                self._load_backend("opencv")
                return f"{reason}->opencv"
        if self.backend == "isnet":
            self.close()
            self._load_backend("opencv")
            return f"{reason}->opencv"
        return reason

    def _infer_prob_map_birefnet(self, image: Image.Image) -> np.ndarray:
        assert self.model is not None
        x = self.birefnet_transform(image.convert("RGB")).unsqueeze(0)
        if self.device.startswith("cuda"):
            x = x.to(self.device).half()
        else:
            x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)[-1].sigmoid()
        prob = logits[0, 0].detach().float().cpu().numpy()
        return prob

    def _infer_prob_map_isnet(self, image: Image.Image) -> np.ndarray:
        assert self.model is not None
        img = np.array(image.convert("RGB"))
        im_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(im_tensor.unsqueeze(0), [1024, 1024], mode="bilinear").type(torch.uint8)
        inp = torch.divide(im_tensor, 255.0)
        inp = tv_normalize(inp, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(self.device)
        with torch.no_grad():
            result = self.model(inp)
        prob = result[0][0][0, 0].detach().float().cpu().numpy()
        return np.clip(prob, 0.0, 1.0)

    def _infer_prob_map_opencv(self, image: Image.Image) -> np.ndarray:
        bgr = _pil_to_bgr(image)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, saliency_map = saliency.computeSaliency(bgr)
        if not ok:
            h, w = bgr.shape[:2]
            return np.zeros((h, w), dtype=np.float32)
        return np.clip(saliency_map.astype(np.float32), 0.0, 1.0)

    def _infer_prob_map(self, image: Image.Image) -> np.ndarray:
        if self.backend == "birefnet":
            return self._infer_prob_map_birefnet(image)
        if self.backend == "isnet":
            return self._infer_prob_map_isnet(image)
        return self._infer_prob_map_opencv(image)

    def _summarize_prob_map(self, image: Image.Image, prob_map: np.ndarray, fallback_reason: str) -> Dict[str, Any]:
        width, height = image.size
        prob_map = cv2.resize(prob_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        prob_map = np.clip(prob_map, 0.0, 1.0)
        mask, threshold = _prob_to_mask(prob_map)
        fg_bbox = _bbox_from_mask(mask, width, height)
        comps, top2_union = _summarize_components(mask, width, height)
        centroid_x, centroid_y = _weighted_centroid(prob_map)
        total_fg_ratio = round(float(mask.mean()) if mask.size > 0 else 0.0, 6)
        dominant_area_ratio = float(comps[0]["area_ratio"]) if comps else 0.0
        dominance_score = float(comps[0]["mass_ratio"]) if comps else 0.0
        top2_mass_ratio = float(sum(item["mass_ratio"] for item in comps[:2])) if comps else 0.0
        summary = {
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "fallback_reason": str(fallback_reason or ""),
            "available": True,
            "image_size": [int(width), int(height)],
            "mask_threshold": round(float(threshold), 6),
            "foreground_area_ratio": total_fg_ratio,
            "blank_ratio_est": round(1.0 - total_fg_ratio, 6),
            "dominant_component_area_ratio": round(dominant_area_ratio, 6),
            "dominance_score": round(dominance_score, 6),
            "top2_mass_ratio": round(top2_mass_ratio, 6),
            "component_count": int(len(comps)),
            "dispersion_score": round(float(_dispersion_score(prob_map)), 6),
            "entropy_norm": round(float(_entropy_norm(prob_map)), 6),
            "weighted_centroid_xy_norm": [round(float(centroid_x), 6), round(float(centroid_y), 6)],
            "foreground_bbox_xyxy": fg_bbox,
            "foreground_bbox_norm_xyxy": (_norm_xyxy(fg_bbox, width, height) if fg_bbox else None),
            "top_component_bbox_xyxy": (comps[0]["bbox_xyxy"] if comps else None),
            "top_component_bbox_norm_xyxy": (comps[0]["bbox_norm_xyxy"] if comps else None),
            "top2_union_bbox_xyxy": top2_union,
            "top2_union_bbox_norm_xyxy": (_norm_xyxy(top2_union, width, height) if top2_union else None),
            "components_topk": comps[:3],
        }
        summary.update(_encode_support_map(prob_map=prob_map, mask=mask, grid_size=SUPPORT_GRID_SIZE))
        return summary

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        fallback_reason = ""
        try:
            prob_map = self._infer_prob_map(image)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                fallback_reason = self._fallback_backend("cuda_oom")
                prob_map = self._infer_prob_map(image)
            else:
                raise
        except BaseException:
            fallback_reason = self._fallback_backend("backend_error")
            prob_map = self._infer_prob_map(image)
        return self._summarize_prob_map(image=image, prob_map=prob_map, fallback_reason=fallback_reason)

    def process_image_with_mask(self, image: Image.Image) -> Tuple[Dict[str, Any], np.ndarray]:
        fallback_reason = ""
        try:
            prob_map = self._infer_prob_map(image)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                fallback_reason = self._fallback_backend("cuda_oom")
                prob_map = self._infer_prob_map(image)
            else:
                raise
        except BaseException:
            fallback_reason = self._fallback_backend("backend_error")
            prob_map = self._infer_prob_map(image)

        width, height = image.size
        prob_map = cv2.resize(prob_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        prob_map = np.clip(prob_map, 0.0, 1.0)
        mask, threshold = _prob_to_mask(prob_map)
        fg_bbox = _bbox_from_mask(mask, width, height)
        comps, top2_union = _summarize_components(mask, width, height)
        centroid_x, centroid_y = _weighted_centroid(prob_map)
        total_fg_ratio = round(float(mask.mean()) if mask.size > 0 else 0.0, 6)
        dominant_area_ratio = float(comps[0]["area_ratio"]) if comps else 0.0
        dominance_score = float(comps[0]["mass_ratio"]) if comps else 0.0
        top2_mass_ratio = float(sum(item["mass_ratio"] for item in comps[:2])) if comps else 0.0
        summary = {
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "fallback_reason": str(fallback_reason or ""),
            "available": True,
            "image_size": [int(width), int(height)],
            "mask_threshold": round(float(threshold), 6),
            "foreground_area_ratio": total_fg_ratio,
            "blank_ratio_est": round(1.0 - total_fg_ratio, 6),
            "dominant_component_area_ratio": round(dominant_area_ratio, 6),
            "dominance_score": round(dominance_score, 6),
            "top2_mass_ratio": round(top2_mass_ratio, 6),
            "component_count": int(len(comps)),
            "dispersion_score": round(float(_dispersion_score(prob_map)), 6),
            "entropy_norm": round(float(_entropy_norm(prob_map)), 6),
            "weighted_centroid_xy_norm": [round(float(centroid_x), 6), round(float(centroid_y), 6)],
            "foreground_bbox_xyxy": fg_bbox,
            "foreground_bbox_norm_xyxy": (_norm_xyxy(fg_bbox, width, height) if fg_bbox else None),
            "top_component_bbox_xyxy": (comps[0]["bbox_xyxy"] if comps else None),
            "top_component_bbox_norm_xyxy": (comps[0]["bbox_norm_xyxy"] if comps else None),
            "top2_union_bbox_xyxy": top2_union,
            "top2_union_bbox_norm_xyxy": (_norm_xyxy(top2_union, width, height) if top2_union else None),
            "components_topk": comps[:3],
        }
        summary.update(_encode_support_map(prob_map=prob_map, mask=mask, grid_size=SUPPORT_GRID_SIZE))
        return summary, mask.astype(bool)
