from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


def _to_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        fv = float(v)
        if not np.isfinite(fv):
            return float(default)
        return float(fv)
    except Exception:
        return float(default)


def _norm_tags(tags: Optional[Sequence[Any]]) -> List[str]:
    if tags is None:
        return []
    out: List[str] = []
    for t in tags:
        s = str(t).strip().lower()
        if s:
            out.append(s)
    return out


TEXT_HINT_WORDS = {
    "text",
    "document",
    "paper",
    "invoice",
    "receipt",
    "form",
    "contract",
    "typography",
    "handwriting",
    "subtitle",
    "caption",
    "title",
    "label",
    "poster",
    "banner",
    "advertisement",
    "sign",
    "street sign",
    "menu",
    "logo",
    "brand",
    "word",
    "letter",
    "number",
}


def _has_text_hint(tags: Optional[Sequence[Any]]) -> bool:
    nt = _norm_tags(tags)
    if not nt:
        return False
    joined = " ".join(nt)
    return any(w in joined for w in TEXT_HINT_WORDS)


def _poly_to_xy_array(poly: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(poly, dtype=np.float32)
    except Exception:
        return None
    if arr.size < 6:
        return None
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    elif arr.ndim >= 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    if arr.shape[0] < 3:
        return None
    return arr


def _aabb_from_poly(poly_xy: np.ndarray, image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    x1 = float(np.clip(np.min(poly_xy[:, 0]), 0.0, max(0.0, image_w - 1.0)))
    y1 = float(np.clip(np.min(poly_xy[:, 1]), 0.0, max(0.0, image_h - 1.0)))
    x2 = float(np.clip(np.max(poly_xy[:, 0]), 0.0, max(0.0, image_w - 1.0)))
    y2 = float(np.clip(np.max(poly_xy[:, 1]), 0.0, max(0.0, image_h - 1.0)))
    return x1, y1, x2, y2


class _PPocrV5DetBackend:
    """PaddleOCR v3 `TextDetection` backend."""

    def __init__(
        self,
        model_name: str,
        model_dir: str = "",
        device: str = "gpu:0",
        limit_side_len: int = 2560,
        limit_type: str = "max",
        thresh: float = 0.30,
        box_thresh: float = 0.60,
        unclip_ratio: float = 1.5,
    ) -> None:
        self.available = False
        self.init_error: Optional[str] = None
        self.model_name = str(model_name)
        self.model_dir = str(model_dir or "")
        self.device = str(device)
        self.limit_side_len = max(320, int(limit_side_len))
        self.limit_type = str(limit_type or "max")
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.unclip_ratio = float(unclip_ratio)
        self._det = None

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import TextDetection

            kwargs: Dict[str, Any] = {
                "model_name": self.model_name,
                "device": self.device,
                "limit_side_len": self.limit_side_len,
                "limit_type": self.limit_type,
                "thresh": self.thresh,
                "box_thresh": self.box_thresh,
                "unclip_ratio": self.unclip_ratio,
            }
            if self.model_dir:
                kwargs["model_dir"] = self.model_dir
            self._det = TextDetection(**kwargs)
            self.available = True
        except Exception as e:
            # GPU init failed -> retry on CPU
            if self.device.startswith("gpu"):
                try:
                    from paddleocr import TextDetection

                    kwargs = {
                        "model_name": self.model_name,
                        "device": "cpu",
                        "limit_side_len": self.limit_side_len,
                        "limit_type": self.limit_type,
                        "thresh": self.thresh,
                        "box_thresh": self.box_thresh,
                        "unclip_ratio": self.unclip_ratio,
                    }
                    if self.model_dir:
                        kwargs["model_dir"] = self.model_dir
                    self._det = TextDetection(**kwargs)
                    self.device = "cpu"
                    self.available = True
                    self.init_error = f"gpu_init_failed_then_cpu_fallback: {e}"
                except Exception as e2:
                    self.init_error = f"gpu_init_failed: {e}; cpu_fallback_failed: {e2}"
                    self.available = False
            else:
                self.init_error = str(e)
                self.available = False

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available or self._det is None:
            return []
        try:
            pred = self._det.predict(input=image_bgr)
        except Exception:
            return []
        if pred is None:
            return []
        if not isinstance(pred, list):
            pred = list(pred)
        if len(pred) <= 0:
            return []

        first = pred[0]
        rec = first if isinstance(first, dict) else {}
        dt_polys = rec.get("dt_polys", [])
        dt_scores = rec.get("dt_scores", [])
        if dt_polys is None:
            return []

        out: List[Dict[str, Any]] = []
        for i, poly in enumerate(list(dt_polys)):
            score = _safe_float(dt_scores[i], 1.0) if i < len(dt_scores) else 1.0
            out.append({"poly": poly, "score": float(score)})
        return out


class _LegacyPaddleOCRDetBackend:
    """Compatibility backend for legacy PaddleOCR API."""

    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "en",
        limit_side_len: int = 960,
    ) -> None:
        self.available = False
        self.init_error: Optional[str] = None
        self.model_name = "legacy_paddleocr_det"
        self.device = "gpu:0" if bool(use_gpu) else "cpu"
        self._ocr = None

        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=lang,
                use_gpu=bool(use_gpu),
                det=True,
                rec=False,
                cls=False,
                det_limit_side_len=max(320, int(limit_side_len)),
                show_log=False,
            )
            self.available = True
        except Exception as e:
            self.init_error = str(e)
            self.available = False

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available or self._ocr is None:
            return []
        try:
            raw = self._ocr.ocr(image_bgr, det=True, rec=False, cls=False)
        except Exception:
            return []
        if not raw:
            return []

        page = raw[0] if isinstance(raw, list) and raw else raw
        if page is None:
            return []
        out: List[Dict[str, Any]] = []
        for item in page:
            poly = None
            score = 1.0
            if isinstance(item, (list, tuple)):
                if len(item) >= 2 and isinstance(item[1], (int, float, np.floating)):
                    poly = item[0]
                    score = _safe_float(item[1], 1.0)
                elif len(item) >= 1:
                    poly = item[0] if isinstance(item[0], (list, tuple, np.ndarray)) else item
            elif isinstance(item, dict):
                poly = item.get("poly") or item.get("box") or item.get("points")
                score = _safe_float(item.get("score"), 1.0)
            if poly is None:
                continue
            out.append({"poly": poly, "score": float(score)})
        return out


class OcrFeatureExtractor:
    """
    C4 OCR text-box detector.

    Backend policy:
    - quality_first: PP-OCRv5 server detector (PP-OCRv5_server_det)
    - high_efficiency: PP-OCRv5 mobile detector (PP-OCRv5_mobile_det)
    - fallback: legacy PaddleOCR det-only path
    """

    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "en",
        priority: str = "high_efficiency",
        c4_backend: str = "auto",
        init_backend: bool = True,
    ) -> None:
        self.use_gpu = bool(use_gpu)
        self.lang = str(lang)
        self.priority = str(priority)

        req_backend = str(c4_backend or "auto").strip().lower()
        env_backend = os.environ.get("C4_BACKEND", "").strip().lower()
        if env_backend in {"auto", "ppocrv5_server", "ppocrv5_mobile", "legacy_paddleocr", "disabled"}:
            req_backend = env_backend
        if req_backend not in {"auto", "ppocrv5_server", "ppocrv5_mobile", "legacy_paddleocr", "disabled"}:
            req_backend = "auto"
        if req_backend == "auto":
            req_backend = "ppocrv5_server" if self.priority == "quality_first" else "ppocrv5_mobile"

        self.requested_backend = req_backend
        self.backend_runtime = "unavailable"
        self.model_name = ""
        self._backend: Optional[Any] = None
        self.init_error: Optional[str] = None

        self.default_device = os.environ.get("C4_PPOCR_DEVICE", "").strip() or ("gpu:0" if self.use_gpu else "cpu")
        self.score_thr = float(
            os.environ.get(
                "C4_SCORE_THR",
                "0.15" if self.priority == "quality_first" else "0.30",
            )
        )
        self.min_box_area_ratio = float(
            os.environ.get(
                "C4_MIN_BOX_AREA_RATIO",
                "0.00001" if self.priority == "quality_first" else "0.00005",
            )
        )
        self.enable_text_hint_gate = _to_bool_env(
            "C4_ENABLE_TEXT_HINT_GATE",
            default=(self.priority != "quality_first"),
        )

        if not init_backend or self.requested_backend == "disabled":
            self.backend_runtime = "disabled"
            return

        self._init_backend()

    def _init_backend(self) -> None:
        server_model = os.environ.get("C4_PPOCR_SERVER_MODEL", "PP-OCRv5_server_det").strip()
        mobile_model = os.environ.get("C4_PPOCR_MOBILE_MODEL", "PP-OCRv5_mobile_det").strip()
        server_model_dir = os.environ.get("C4_PPOCR_SERVER_MODEL_DIR", "").strip()
        mobile_model_dir = os.environ.get("C4_PPOCR_MOBILE_MODEL_DIR", "").strip()
        server_limit = int(os.environ.get("C4_PPOCR_SERVER_LIMIT_SIDE_LEN", "2560"))
        mobile_limit = int(os.environ.get("C4_PPOCR_MOBILE_LIMIT_SIDE_LEN", "960"))
        legacy_limit = int(os.environ.get("C4_LEGACY_LIMIT_SIDE_LEN", "960"))

        init_errors: List[str] = []

        def _try_ppocrv5(model_name: str, model_dir: str, limit_side_len: int) -> Optional[_PPocrV5DetBackend]:
            be = _PPocrV5DetBackend(
                model_name=model_name,
                model_dir=model_dir,
                device=self.default_device,
                limit_side_len=limit_side_len,
            )
            if be.available:
                return be
            init_errors.append(f"{model_name}: {be.init_error}")
            return None

        def _try_legacy() -> Optional[_LegacyPaddleOCRDetBackend]:
            be = _LegacyPaddleOCRDetBackend(use_gpu=self.use_gpu, lang=self.lang, limit_side_len=legacy_limit)
            if be.available:
                return be
            init_errors.append(f"legacy_paddleocr: {be.init_error}")
            return None

        if self.requested_backend == "ppocrv5_server":
            backend = _try_ppocrv5(server_model, server_model_dir, server_limit)
            if backend is None:
                backend = _try_ppocrv5(mobile_model, mobile_model_dir, mobile_limit)
                if backend is not None:
                    self.backend_runtime = "ppocrv5_mobile_fallback"
            if backend is None:
                backend = _try_legacy()
                if backend is not None:
                    self.backend_runtime = "legacy_fallback"
        elif self.requested_backend == "ppocrv5_mobile":
            backend = _try_ppocrv5(mobile_model, mobile_model_dir, mobile_limit)
            if backend is None:
                backend = _try_legacy()
                if backend is not None:
                    self.backend_runtime = "legacy_fallback"
        elif self.requested_backend == "legacy_paddleocr":
            backend = _try_legacy()
        else:
            backend = None

        if backend is None:
            self._backend = None
            self.backend_runtime = "unavailable"
            self.init_error = " | ".join(init_errors) if init_errors else "backend_init_failed"
            print(f"[C4 OCR] backend unavailable. requested={self.requested_backend} reason={self.init_error}")
            return

        self._backend = backend
        self.model_name = getattr(backend, "model_name", "")
        if self.backend_runtime in {"unavailable", "disabled", ""}:
            if isinstance(backend, _PPocrV5DetBackend):
                self.backend_runtime = "ppocrv5_server" if "server" in self.model_name.lower() else "ppocrv5_mobile"
            else:
                self.backend_runtime = "legacy_paddleocr"
        self.init_error = getattr(backend, "init_error", None)
        print(
            "[C4 OCR] init done. "
            f"priority={self.priority} requested={self.requested_backend} runtime={self.backend_runtime} "
            f"model={self.model_name or '<none>'} device={getattr(backend, 'device', self.default_device)} "
            f"gate_by_tags={int(self.enable_text_hint_gate)} score_thr={self.score_thr:.2f}"
        )

    def _should_run(self, tags: Optional[Sequence[Any]], run_anyway: bool) -> bool:
        if run_anyway:
            return True
        if not self.enable_text_hint_gate:
            return True
        return _has_text_hint(tags)

    def _empty_payload(self, method: str) -> Dict[str, Any]:
        return {
            "boxes": [],
            "num_boxes": 0,
            "coverage_ratio": 0.0,
            "text_overlay_likely": False,
            "method": method,
            "requested_backend": self.requested_backend,
            "backend_runtime": self.backend_runtime,
            "model_name": self.model_name or None,
        }

    def process_image(
        self,
        image: Image.Image,
        run_anyway: bool = False,
        tags: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        if not self._should_run(tags=tags, run_anyway=run_anyway):
            return self._empty_payload(method="skipped")
        if self._backend is None:
            return self._empty_payload(method="unavailable")

        image_np = np.asarray(image, dtype=np.uint8)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            return self._empty_payload(method="invalid_image")
        h, w = int(image_np.shape[0]), int(image_np.shape[1])
        if h <= 0 or w <= 0:
            return self._empty_payload(method="invalid_image")

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        raw_items = self._backend.detect(image_bgr)
        if not raw_items:
            method = (
                "ppocrv5_server_det"
                if self.backend_runtime.startswith("ppocrv5_server")
                else "ppocrv5_mobile_det"
                if self.backend_runtime.startswith("ppocrv5_mobile")
                else "paddleocr_legacy_det"
                if self.backend_runtime.startswith("legacy")
                else "unavailable"
            )
            return self._empty_payload(method=method)

        boxes: List[Dict[str, Any]] = []
        area_sum = 0.0
        max_width_ratio = 0.0

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            score = _safe_float(item.get("score"), 1.0)
            if score < self.score_thr:
                continue
            poly_xy = _poly_to_xy_array(item.get("poly"))
            if poly_xy is None:
                continue

            x1, y1, x2, y2 = _aabb_from_poly(poly_xy=poly_xy, image_w=w, image_h=h)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0.0 or bh <= 0.0:
                continue
            area_ratio = (bw * bh) / float(max(1.0, float(w * h)))
            if area_ratio < self.min_box_area_ratio:
                continue

            quad = [
                [round(x1, 2), round(y1, 2)],
                [round(x2, 2), round(y1, 2)],
                [round(x2, 2), round(y2, 2)],
                [round(x1, 2), round(y2, 2)],
            ]
            poly = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in poly_xy.tolist()]
            box_xyxy = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            box_norm_xyxy = [
                round(x1 / float(w), 6),
                round(y1 / float(h), 6),
                round(x2 / float(w), 6),
                round(y2 / float(h), 6),
            ]

            boxes.append(
                {
                    "box": quad,
                    "poly": poly,
                    "box_xyxy": box_xyxy,
                    "box_norm_xyxy": box_norm_xyxy,
                    "score": round(float(score), 6),
                    "area_ratio": round(float(area_ratio), 8),
                }
            )
            area_sum += float(area_ratio)
            max_width_ratio = max(max_width_ratio, bw / float(max(1, w)))

        coverage_ratio = _clamp01(area_sum)
        text_overlay_likely = bool(
            len(boxes) > 0
            and (
                len(boxes) >= 3
                or coverage_ratio >= 0.015
                or max_width_ratio >= 0.35
            )
        )

        if self.backend_runtime.startswith("ppocrv5_server"):
            method = "ppocrv5_server_det"
        elif self.backend_runtime.startswith("ppocrv5_mobile"):
            method = "ppocrv5_mobile_det"
        elif self.backend_runtime.startswith("legacy"):
            method = "paddleocr_legacy_det"
        else:
            method = "unavailable"

        return {
            "boxes": boxes,
            "num_boxes": len(boxes),
            "coverage_ratio": round(float(coverage_ratio), 6),
            "text_overlay_likely": text_overlay_likely,
            "method": method,
            "requested_backend": self.requested_backend,
            "backend_runtime": self.backend_runtime,
            "model_name": self.model_name or None,
        }
