from __future__ import annotations

import math
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


class GeoFeatureExtractor:
    """
    Lightweight geometry extractor for scorer-critical cues.

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
    ):
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.hough_threshold = int(hough_threshold)
        self.min_line_len_ratio = float(min_line_len_ratio)
        self.max_line_gap_ratio = float(max_line_gap_ratio)
        self.horizon_max_abs_theta = float(horizon_max_abs_theta)

    @staticmethod
    def _normalize_theta_deg(theta_deg: float) -> float:
        t = float(theta_deg)
        while t > 90.0:
            t -= 180.0
        while t < -90.0:
            t += 180.0
        return t

    def _estimate_horizon_roll(self, gray: np.ndarray) -> Dict[str, Any]:
        h, w = gray.shape[:2]
        if w < 32 or h < 32:
            return {
                "horizon_y_norm": None,
                "theta_deg": None,
                "roll_deg": None,
                "conf": 0.0,
                "line_norm_xyxy": None,
                "num_lines": 0,
                "method": "houghp",
            }

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

        candidates: List[Dict[str, float]] = []
        if lines is not None:
            for row in lines:
                x1, y1, x2, y2 = [float(v) for v in row[0]]
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
                    y_at_center = y1 + ((0.5 * w - x1) / dx) * dy
                y_at_center = max(0.0, min(float(h - 1), float(y_at_center)))

                len_term = line_len / max(1.0, float(w))
                angle_term = math.exp(-abs_theta / 12.0)
                center_term = 1.0 - min(1.0, abs(y_at_center - 0.5 * h) / (0.5 * h + 1e-6))

                score = len_term * (0.75 * angle_term + 0.25 * center_term)
                candidates.append(
                    {
                        "score": float(score),
                        "theta": float(theta),
                        "y_at_center": float(y_at_center),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        if not candidates:
            return {
                "horizon_y_norm": None,
                "theta_deg": None,
                "roll_deg": None,
                "conf": 0.0,
                "line_norm_xyxy": None,
                "num_lines": 0,
                "method": "houghp",
            }

        best = max(candidates, key=lambda x: x["score"])
        conf = _clamp01(best["score"])

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
            "method": "houghp",
        }

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

        return {
            "horizon_roll": horizon_roll,
            "symmetry": symmetry,
        }
