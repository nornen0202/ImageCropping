from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkBox:
    bbox_xyxy_norm: list[float]
    label: str = ""
    weight: float = 1.0
    candidate_id: str = ""
    source: str = ""
    score: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy_norm": [float(v) for v in self.bbox_xyxy_norm],
            "label": self.label,
            "weight": float(self.weight),
            "candidate_id": self.candidate_id,
            "source": self.source,
            "score": self.score,
            "meta": dict(self.meta),
        }


@dataclass
class PairwisePreference:
    bbox_a_xyxy_norm: list[float]
    bbox_b_xyxy_norm: list[float]
    preferred: str
    candidate_id_a: str = ""
    candidate_id_b: str = ""
    weight: float = 1.0
    votes: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_a_xyxy_norm": [float(v) for v in self.bbox_a_xyxy_norm],
            "bbox_b_xyxy_norm": [float(v) for v in self.bbox_b_xyxy_norm],
            "preferred": self.preferred,
            "candidate_id_a": self.candidate_id_a,
            "candidate_id_b": self.candidate_id_b,
            "weight": float(self.weight),
            "votes": dict(self.votes),
            "meta": dict(self.meta),
        }


@dataclass
class BenchmarkTask:
    dataset: str
    split: str
    image_id: str
    image_path: str
    task_type: str
    target_ar: Optional[str] = None
    gt_boxes: list[BenchmarkBox] = field(default_factory=list)
    pairwise: list[PairwisePreference] = field(default_factory=list)
    candidate_windows: list[BenchmarkBox] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, Optional[str]]:
        return (self.dataset, self.image_id, self.target_ar)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "task_type": self.task_type,
            "target_ar": self.target_ar,
            "gt_boxes": [box.to_dict() for box in self.gt_boxes],
            "pairwise": [pair.to_dict() for pair in self.pairwise],
            "candidate_windows": [box.to_dict() for box in self.candidate_windows],
            "meta": dict(self.meta),
        }
