from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_AR_VOCAB = ["FREE", "1:1", "9:16", "16:9", "3:4", "4:3"]
DECISION_VOCAB = ["keep_full", "minimal_crop", "crop"]
SUBJECT_MODE_COUNT = 7


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


def decision_id(value: Any, fallback: int = 2) -> int:
    if isinstance(value, int):
        return max(0, min(len(DECISION_VOCAB) - 1, value))
    try:
        return DECISION_VOCAB.index(str(value))
    except ValueError:
        return max(0, min(len(DECISION_VOCAB) - 1, int(fallback)))


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


def load_image_tensor_letterbox(path: Path, input_size: int) -> tuple[torch.Tensor, LetterboxTransform]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        transform = compute_letterbox_transform(width, height, input_size)
        resized = rgb.resize((transform.resized_width, transform.resized_height), Image.BILINEAR)
        canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        canvas.paste(resized, (transform.pad_x, transform.pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous(), transform


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

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> dict[str, Any]:
        ar_counts: dict[str, int] = {}
        matching_count = 0
        pool_count = 0
        for row in self.records:
            ar = str(row.get("target_ar", "FREE"))
            ar_counts[ar] = ar_counts.get(ar, 0) + 1
            matching_count += len(row.get("matching_targets") or [])
            pool_count += len(row.get("candidate_pool") or [])
        return {
            "jsonl_path": str(self.jsonl_path),
            "image_count": len(self.records),
            "target_ar_counts": ar_counts,
            "matching_target_count": matching_count,
            "candidate_pool_count": pool_count,
            "input_size": self.input_size,
            "candidate_k": self.candidate_k,
            "max_positive_boxes": self.max_positive_boxes,
        }

    def _candidate_records(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[float, float, float, float]]] = set()

        def add(candidate: dict[str, Any], *, source: str, is_base: bool = False, positive_override: bool | None = None) -> None:
            box = _candidate_box(candidate)
            cid = _candidate_id(candidate, f"{source}_{len(selected)}")
            key = (cid, tuple(round(v, 5) for v in box))
            if key in seen:
                return
            seen.add(key)
            score = _score_from_candidate(candidate)
            is_positive = bool(positive_override) if positive_override is not None else bool(candidate.get("is_positive_candidate", False))
            is_soft_positive = bool(candidate.get("is_soft_positive", False))
            is_hard_negative = bool(candidate.get("is_hard_negative", False))
            is_unsafe_negative = bool(candidate.get("is_unsafe_negative", False))
            is_ignore = bool(candidate.get("is_ignore_candidate", False))
            is_overflow = bool(candidate.get("is_overflow_candidate", False))
            selected.append(
                {
                    "candidate_id": cid,
                    "bbox_norm_xyxy": box,
                    "score_target": score,
                    "positive_target": float(is_positive or is_soft_positive or positive_override is True),
                    "risk_target": float(is_hard_negative or is_unsafe_negative or is_overflow),
                    "is_base": float(is_base),
                    "is_ignore": is_ignore,
                    "is_overflow": is_overflow,
                    "label_type": str(candidate.get("label_type", source)),
                    "training_bucket": str(candidate.get("training_bucket", "")),
                    "source": source,
                    "macro_target": _macro_targets(candidate),
                }
            )

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
            positives.append({"bbox_norm_xyxy": box, "score": _score_from_candidate(candidate)})

        for candidate in sorted(row.get("matching_targets") or [], key=_score_from_candidate, reverse=True):
            add(candidate)
        for candidate in sorted(row.get("candidate_pool") or [], key=_score_from_candidate, reverse=True):
            score = _score_from_candidate(candidate)
            if bool(candidate.get("is_positive_candidate", False)) or (
                bool(candidate.get("is_soft_positive", False)) and score >= self.min_soft_positive_score
            ):
                add(candidate)
        return positives[: self.max_positive_boxes]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        image_path = _resolve_image_path(row, project_root=self.project_root, image_root=self.image_root)
        image, transform = load_image_tensor_letterbox(image_path, self.input_size)
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
        candidate_is_base = torch.zeros((k,), dtype=torch.float32)
        valid = torch.zeros((k,), dtype=torch.float32)
        macro_target = torch.zeros((k, 4), dtype=torch.float32)
        macro_valid = torch.zeros((k,), dtype=torch.float32)

        for slot, candidate in enumerate(candidate_records[:k]):
            orig_box = _clamp_box(candidate["bbox_norm_xyxy"])
            padded_box = original_to_letterbox_box(orig_box, transform)
            is_base = float(candidate.get("is_base", 0.0))
            boxes[slot] = torch.tensor(padded_box, dtype=torch.float32)
            boxes_orig[slot] = torch.tensor(orig_box, dtype=torch.float32)
            box_meta[slot] = torch.tensor(_box_meta(padded_box, is_base=is_base), dtype=torch.float32)
            score_target[slot] = float(candidate.get("score_target", 0.0))
            positive_target[slot] = float(candidate.get("positive_target", 0.0))
            risk_target[slot] = float(candidate.get("risk_target", 0.0))
            candidate_is_base[slot] = is_base
            valid[slot] = 1.0
            mt = candidate.get("macro_target")
            if isinstance(mt, list) and len(mt) == 4 and any(abs(float(v)) > 0 for v in mt):
                macro_target[slot] = torch.tensor(mt, dtype=torch.float32)
                macro_valid[slot] = 1.0

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

        return {
            "image": image,
            "boxes": boxes,
            "boxes_orig": boxes_orig,
            "box_meta": box_meta,
            "valid": valid,
            "score_target": score_target,
            "positive_target": positive_target,
            "risk_target": risk_target,
            "candidate_is_base": candidate_is_base,
            "macro_target": macro_target,
            "macro_valid": macro_valid,
            "positive_boxes": positive_boxes,
            "positive_boxes_orig": positive_boxes_orig,
            "positive_valid": positive_valid,
            "positive_scores": positive_scores,
            "target_ar_id": torch.tensor(target_ar_id(str(row.get("target_ar", "FREE"))), dtype=torch.long),
            "image_ar_log": torch.tensor(transform.image_ar_log, dtype=torch.float32),
            "decision_target": torch.tensor(decision_target, dtype=torch.long),
            "route_target": torch.tensor(route_target, dtype=torch.long),
            "delta_target": torch.tensor(delta_target, dtype=torch.float32),
            "image_id": str(row.get("image_id", "")),
            "image_path": str(image_path),
            "target_ar": str(row.get("target_ar", "FREE")),
            "width": transform.original_width,
            "height": transform.original_height,
            "transform": transform,
            "candidate_records": candidate_records,
            "positive_records": positive_records,
        }


def mobilecropnet_v4_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "image",
        "boxes",
        "boxes_orig",
        "box_meta",
        "valid",
        "score_target",
        "positive_target",
        "risk_target",
        "candidate_is_base",
        "macro_target",
        "macro_valid",
        "positive_boxes",
        "positive_boxes_orig",
        "positive_valid",
        "positive_scores",
        "target_ar_id",
        "image_ar_log",
        "decision_target",
        "route_target",
        "delta_target",
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
