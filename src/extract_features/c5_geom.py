from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _safe_float(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


class _ScaleLSDLineBackend:
    """
    Optional ScaleLSD wrapper.
    - If ScaleLSD import/weights/model init fails, `available=False`.
    - Caller should fallback to lightweight Hough path.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        ckpt_path: str = "",
        input_width: int = 512,
        input_height: int = 512,
        use_lsd: bool = True,
        use_nms: bool = True,
    ) -> None:
        self.available = False
        self.device = str(device or os.environ.get("C5_SCALELSD_DEVICE", "cuda"))
        self.input_width = max(64, int(input_width))
        self.input_height = max(64, int(input_height))
        self.use_lsd = bool(use_lsd)
        self.use_nms = bool(use_nms)
        self.init_error: Optional[str] = None
        self._torch = None
        self._model = None

        if not ckpt_path:
            ckpt_path = os.environ.get("C5_SCALELSD_CKPT", "").strip()
        default_ckpt = os.path.join("weights", "scalelsd", "scalelsd-vitbase-v1-train-sa1b.pt")
        if not ckpt_path:
            ckpt_candidates = [
                default_ckpt,
                os.path.join("third_party", "scalelsd", "models", "scalelsd-vitbase-v1-train-sa1b.pt"),
                os.path.join("third_party", "scalelsd_probe_1", "models", "scalelsd-vitbase-v1-train-sa1b.pt"),
            ]
            for cand in ckpt_candidates:
                if os.path.isfile(cand):
                    ckpt_path = cand
                    break
        if not ckpt_path:
            ckpt_path = default_ckpt
        self.ckpt_path = ckpt_path

        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            if self._maybe_download_default_ckpt(self.ckpt_path):
                pass
            else:
                self.init_error = f"scalelsd checkpoint not found: {self.ckpt_path or '<empty>'}"
                return

        try:
            import torch
            from scalelsd.ssl.misc.train_utils import load_scalelsd_model
            from scalelsd.base.csrc import _C as scalelsd_csrc
            from scalelsd.ssl.models.detector import ScaleLSD

            dev = self.device
            if dev.startswith("cuda") and (not torch.cuda.is_available()):
                dev = "cpu"

            # If CUDA extension is unavailable (common when nvcc is absent),
            # disable LSD-rectifier path and use native forward branch.
            if self.use_lsd and scalelsd_csrc is None:
                self.use_lsd = False
                print("[C5 Geom] ScaleLSD C++/CUDA extension unavailable. use_lsd=False fallback enabled.")

            # Required by ScaleLSD forward_test: class attrs are initialized via configure().
            num_junctions = int(os.environ.get("C5_SCALELSD_NUM_JUNCTIONS", "512"))
            junction_hm = float(os.environ.get("C5_SCALELSD_JUNCTION_HM", "0.008"))
            ScaleLSD.configure(SimpleNamespace(num_junctions=num_junctions, junction_hm=junction_hm))

            self._torch = torch
            self._model = load_scalelsd_model(self.ckpt_path, device=dev)
            self.device = dev
            self.available = True
        except Exception as e:
            self.init_error = str(e)
            self.available = False

    @staticmethod
    def _maybe_download_default_ckpt(dest_path: str) -> bool:
        auto_download = os.environ.get("C5_SCALELSD_AUTO_DOWNLOAD", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if not auto_download:
            return False
        if not dest_path:
            return False
        if os.path.isfile(dest_path):
            return True

        url = os.environ.get(
            "C5_SCALELSD_CKPT_URL",
            "https://huggingface.co/cherubicxn/scalelsd/resolve/main/scalelsd-vitbase-v1-train-sa1b.pt",
        ).strip()
        if not url:
            return False

        try:
            import urllib.request

            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            print(f"[C5 Geom] Downloading ScaleLSD ckpt -> {dest_path}")
            urllib.request.urlretrieve(url, dest_path)
            return os.path.isfile(dest_path)
        except Exception as e:
            print(f"[C5 Geom] ScaleLSD ckpt auto-download failed: {e}")
            return False

    def detect_lines(self, gray: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if not self.available or self._model is None or self._torch is None:
            return None

        try:
            resized = cv2.resize(gray, (self.input_width, self.input_height))
            inp = self._torch.from_numpy(resized).float().div(255.0).unsqueeze(0).unsqueeze(0).to(self.device)
            meta = {
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "filename": "",
                "use_lsd": bool(self.use_lsd),
                "use_nms": bool(self.use_nms),
            }
            with self._torch.no_grad():
                outputs, _ = self._model(inp, meta)
            if not outputs:
                return None

            out0 = outputs[0]
            lines = out0.get("lines_pred")
            scores = out0.get("lines_score")
            if lines is None:
                return None

            if hasattr(lines, "detach"):
                lines_np = lines.detach().cpu().numpy().astype(np.float32)
            else:
                lines_np = np.asarray(lines, dtype=np.float32)
            if lines_np.ndim != 2 or lines_np.shape[1] < 4:
                return None
            lines_np = lines_np[:, :4]

            if scores is None:
                score_np = np.ones((lines_np.shape[0],), dtype=np.float32)
            elif hasattr(scores, "detach"):
                score_np = scores.detach().cpu().numpy().astype(np.float32).reshape(-1)
            else:
                score_np = np.asarray(scores, dtype=np.float32).reshape(-1)
            if score_np.shape[0] != lines_np.shape[0]:
                score_np = np.ones((lines_np.shape[0],), dtype=np.float32)

            return {"lines": lines_np, "scores": score_np}
        except Exception:
            return None


class GeoFeatureExtractor:
    """
    Geometry extractor for scorer-critical cues.

    Backends:
    - high_efficiency: OpenCV Canny + HoughLinesP
    - quality_first: ScaleLSD(+RANSAC) if available, else Hough fallback

    Outputs:
    - horizon_roll: horizon y/angle/roll + confidence
    - symmetry: horizontal mirror symmetry score [0, 1]
    """

    def __init__(
        self,
        canny_low: int = 80,
        canny_high: int = 180,
        hough_threshold: int = 60,
        min_line_len_ratio: float = 0.25,
        max_line_gap_ratio: float = 0.03,
        horizon_max_abs_theta: float = 30.0,
        priority: str = "high_efficiency",
        c5_backend: str = "auto",
        scalelsd_ckpt: str = "",
        scalelsd_input_width: int = 512,
        scalelsd_input_height: int = 512,
        scalelsd_device: Optional[str] = None,
        scalelsd_use_lsd: bool = True,
        scalelsd_use_nms: bool = True,
        scalelsd_min_line_len_ratio: float = 0.03,
        ransac_theta_tol_deg: float = 2.0,
        ransac_y_tol_ratio: float = 0.04,
    ) -> None:
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.hough_threshold = int(hough_threshold)
        self.min_line_len_ratio = float(min_line_len_ratio)
        self.max_line_gap_ratio = float(max_line_gap_ratio)
        self.horizon_max_abs_theta = float(horizon_max_abs_theta)
        self.priority = str(priority)
        self.scalelsd_min_line_len_ratio = max(0.005, float(scalelsd_min_line_len_ratio))
        self.ransac_theta_tol_deg = max(0.5, float(ransac_theta_tol_deg))
        self.ransac_y_tol_ratio = max(0.005, float(ransac_y_tol_ratio))

        env_backend = os.environ.get("C5_BACKEND", "").strip().lower()
        req_backend = str(c5_backend or "auto").strip().lower()
        if env_backend in {"auto", "houghp", "scalelsd"}:
            req_backend = env_backend
        if req_backend not in {"auto", "houghp", "scalelsd"}:
            req_backend = "auto"

        if req_backend == "auto":
            req_backend = "scalelsd" if self.priority == "quality_first" else "houghp"
        self.requested_backend = req_backend
        self.backend_runtime = "houghp"

        self.scalelsd: Optional[_ScaleLSDLineBackend] = None
        if self.requested_backend == "scalelsd":
            self.scalelsd = _ScaleLSDLineBackend(
                device=scalelsd_device,
                ckpt_path=scalelsd_ckpt,
                input_width=scalelsd_input_width,
                input_height=scalelsd_input_height,
                use_lsd=scalelsd_use_lsd,
                use_nms=scalelsd_use_nms,
            )
            if self.scalelsd.available:
                self.backend_runtime = "scalelsd"
            else:
                self.backend_runtime = "houghp_fallback"
                print(f"[C5 Geom] ScaleLSD unavailable -> fallback to Hough. reason={self.scalelsd.init_error}")

    @staticmethod
    def _normalize_theta_deg(theta_deg: float) -> float:
        t = float(theta_deg)
        while t > 90.0:
            t -= 180.0
        while t < -90.0:
            t += 180.0
        return t

    def _empty_horizon(self, method: str) -> Dict[str, Any]:
        return {
            "horizon_y_norm": None,
            "theta_deg": None,
            "roll_deg": None,
            "conf": 0.0,
            "line_norm_xyxy": None,
            "num_lines": 0,
            "num_inliers": 0,
            "method": method,
        }

    def _line_candidates_from_xyxy(
        self,
        lines_xyxy: np.ndarray,
        scores: np.ndarray,
        w: int,
        h: int,
        min_line_len: float,
    ) -> List[Dict[str, float]]:
        candidates: List[Dict[str, float]] = []
        n = int(min(len(lines_xyxy), len(scores)))
        for i in range(n):
            x1, y1, x2, y2 = [float(v) for v in lines_xyxy[i, :4]]
            dx = x2 - x1
            dy = y2 - y1
            line_len = math.hypot(dx, dy)
            if line_len < float(min_line_len):
                continue

            theta = self._normalize_theta_deg(math.degrees(math.atan2(dy, dx + 1e-6)))
            abs_theta = abs(theta)
            if abs_theta > self.horizon_max_abs_theta:
                continue

            if abs(dx) < 1e-6:
                y_at_center = 0.5 * (y1 + y2)
            else:
                y_at_center = y1 + ((0.5 * float(w) - x1) / dx) * dy
            y_at_center = max(0.0, min(float(h - 1), float(y_at_center)))

            len_term = line_len / max(1.0, float(w))
            angle_term = math.exp(-abs_theta / 12.0)
            center_term = 1.0 - min(1.0, abs(y_at_center - 0.5 * h) / (0.5 * h + 1e-6))
            model_score = float(max(0.0, float(scores[i])))
            weight = len_term * (0.65 * angle_term + 0.25 * center_term + 0.10 * min(1.0, model_score))
            candidates.append(
                {
                    "weight": float(max(1e-6, weight)),
                    "theta": float(theta),
                    "y_at_center": float(y_at_center),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        return candidates

    def _estimate_horizon_roll_hough(self, gray: np.ndarray, method: str = "houghp") -> Dict[str, Any]:
        h, w = gray.shape[:2]
        if w < 32 or h < 32:
            return self._empty_horizon(method=method)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high, L2gradient=True)
        min_line_len = max(24, int(self.min_line_len_ratio * w))
        max_line_gap = max(4, int(self.max_line_gap_ratio * w))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )

        if lines is None:
            return self._empty_horizon(method=method)
        lines_xyxy = np.asarray(lines[:, 0, :], dtype=np.float32)
        score = np.ones((lines_xyxy.shape[0],), dtype=np.float32)
        candidates = self._line_candidates_from_xyxy(lines_xyxy, score, w=w, h=h, min_line_len=float(min_line_len))
        if not candidates:
            return self._empty_horizon(method=method)

        best = max(candidates, key=lambda x: x["weight"])
        conf = _clamp01(best["weight"])
        line_norm = [
            round(best["x1"] / max(1.0, float(w)), 6),
            round(best["y1"] / max(1.0, float(h)), 6),
            round(best["x2"] / max(1.0, float(w)), 6),
            round(best["y2"] / max(1.0, float(h)), 6),
        ]
        horizon_y_norm = best["y_at_center"] / max(1.0, float(h))
        theta = _safe_float(best.get("theta"))
        roll_deg = theta
        return {
            "horizon_y_norm": None if horizon_y_norm is None else round(float(horizon_y_norm), 6),
            "theta_deg": None if theta is None else round(theta, 3),
            "roll_deg": None if roll_deg is None else round(float(roll_deg), 3),
            "conf": round(conf, 4),
            "line_norm_xyxy": line_norm,
            "num_lines": len(candidates),
            "num_inliers": len(candidates),
            "method": method,
        }

    def _estimate_horizon_roll_scalelsd(self, gray: np.ndarray) -> Dict[str, Any]:
        h, w = gray.shape[:2]
        if self.scalelsd is None or not self.scalelsd.available:
            return self._empty_horizon(method="scalelsd_unavailable")
        if w < 32 or h < 32:
            return self._empty_horizon(method="scalelsd_ransac")

        out = self.scalelsd.detect_lines(gray)
        if out is None:
            return self._empty_horizon(method="scalelsd_no_output")

        lines_xyxy = out.get("lines")
        scores = out.get("scores")
        if lines_xyxy is None or scores is None:
            return self._empty_horizon(method="scalelsd_missing_fields")

        min_line_len = max(6, int(self.scalelsd_min_line_len_ratio * w))
        candidates = self._line_candidates_from_xyxy(
            np.asarray(lines_xyxy, dtype=np.float32),
            np.asarray(scores, dtype=np.float32).reshape(-1),
            w=w,
            h=h,
            min_line_len=float(min_line_len),
        )
        if not candidates:
            return self._empty_horizon(method="scalelsd_filtered_empty")

        y_tol = max(6.0, float(h) * self.ransac_y_tol_ratio)
        best_model: Optional[Dict[str, Any]] = None
        total_weight = max(1e-6, sum(float(c["weight"]) for c in candidates))
        for anchor in candidates:
            theta0 = float(anchor["theta"])
            y0 = float(anchor["y_at_center"])
            inliers: List[Dict[str, Any]] = []
            score = 0.0
            for cand in candidates:
                dtheta = abs(float(cand["theta"]) - theta0)
                dy = abs(float(cand["y_at_center"]) - y0)
                if dtheta > self.ransac_theta_tol_deg or dy > y_tol:
                    continue
                w_local = float(cand["weight"]) * math.exp(-dtheta / self.ransac_theta_tol_deg) * math.exp(-dy / y_tol)
                score += w_local
                inliers.append({"cand": cand, "w": w_local})
            if best_model is None or score > float(best_model["score"]):
                best_model = {"score": float(score), "inliers": inliers}

        if best_model is None or not best_model["inliers"]:
            return self._empty_horizon(method="scalelsd_ransac_failed")

        wsum = max(1e-6, sum(float(x["w"]) for x in best_model["inliers"]))
        theta_hat = sum(float(x["cand"]["theta"]) * float(x["w"]) for x in best_model["inliers"]) / wsum
        y_hat = sum(float(x["cand"]["y_at_center"]) * float(x["w"]) for x in best_model["inliers"]) / wsum
        y_hat = max(0.0, min(float(h - 1), float(y_hat)))

        rep = min(
            [x["cand"] for x in best_model["inliers"]],
            key=lambda c: abs(float(c["theta"]) - theta_hat) + 0.25 * abs(float(c["y_at_center"]) - y_hat) / y_tol,
        )
        line_norm = [
            round(float(rep["x1"]) / max(1.0, float(w)), 6),
            round(float(rep["y1"]) / max(1.0, float(h)), 6),
            round(float(rep["x2"]) / max(1.0, float(w)), 6),
            round(float(rep["y2"]) / max(1.0, float(h)), 6),
        ]
        conf = _clamp01(float(best_model["score"]) / total_weight)
        return {
            "horizon_y_norm": round(float(y_hat) / max(1.0, float(h)), 6),
            "theta_deg": round(float(theta_hat), 3),
            "roll_deg": round(float(theta_hat), 3),
            "conf": round(conf, 4),
            "line_norm_xyxy": line_norm,
            "num_lines": len(candidates),
            "num_inliers": len(best_model["inliers"]),
            "method": "scalelsd_ransac",
        }

    def _estimate_horizon_roll(self, gray: np.ndarray) -> Dict[str, Any]:
        if self.backend_runtime == "scalelsd":
            out = self._estimate_horizon_roll_scalelsd(gray)
            if float(out.get("conf", 0.0) or 0.0) > 0.0:
                return out
            return self._estimate_horizon_roll_hough(gray, method="houghp_fallback")
        if self.requested_backend == "scalelsd":
            return self._estimate_horizon_roll_hough(gray, method="houghp_fallback")
        return self._estimate_horizon_roll_hough(gray, method="houghp")

    @staticmethod
    def _estimate_symmetry(gray: np.ndarray) -> Dict[str, Any]:
        h, w = gray.shape[:2]
        if w < 4 or h < 4:
            return {"score": 0.0, "method": "mirror_l1"}

        arr = gray.astype(np.float32) / 255.0
        half = w // 2
        if half <= 1:
            return {"score": 0.0, "method": "mirror_l1"}

        left = arr[:, :half]
        right = arr[:, w - half :]
        right = np.fliplr(right)

        diff = float(np.mean(np.abs(left - right)))
        score = _clamp01(1.0 - diff)
        return {"score": round(score, 4), "method": "mirror_l1"}

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        horizon_roll = self._estimate_horizon_roll(gray)
        symmetry = self._estimate_symmetry(gray)
        horizon_conf = float(horizon_roll.get("conf", 0.0) or 0.0)
        symmetry_score = float(symmetry.get("score", 0.0) or 0.0)

        return {
            "horizon_roll": horizon_roll,
            "symmetry": symmetry,
            "horizon_conf": round(horizon_conf, 4),
            "symmetry_score": round(symmetry_score, 4),
            "backend_runtime": self.backend_runtime,
            "requested_backend": self.requested_backend,
        }
