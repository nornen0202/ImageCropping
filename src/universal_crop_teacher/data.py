from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_MARKER = "/Sources/ImageCropping/"

DATASET_VOCAB = ("unknown", "fcdb", "cpc", "gnmc", "gaic", "sstk", "test_images", "internal")
TARGET_AR_VOCAB = ("FREE", "1:1", "9:16", "16:9", "3:4", "4:3", "2:1")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
UNKNOWN_SOURCE_BUCKETS = 8
CANDIDATE_FEATURE_DIM = 25


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def stable_hash_int(text: Any, modulo: int = 10000) -> int:
    raw = str(text).encode("utf-8", errors="ignore")
    digest = hashlib.sha1(raw).hexdigest()
    return int(digest[:12], 16) % int(modulo)


def remap_project_path(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    candidate = Path(text)
    if candidate.exists():
        return str(candidate)
    if not candidate.is_absolute():
        rel_candidate = (PROJECT_ROOT / candidate).resolve()
        if rel_candidate.exists():
            return str(rel_candidate)
    normalized = text.replace("\\", "/")
    if PROJECT_MARKER in normalized:
        rel_text = normalized.split(PROJECT_MARKER, 1)[1]
        mapped = (PROJECT_ROOT / rel_text).resolve()
        if mapped.exists():
            return str(mapped)
    for anchor in ("/data/", "/artifacts/", "/Implement_Docs/"):
        if anchor in normalized:
            rel_text = normalized.split(anchor, 1)[1]
            mapped = (PROJECT_ROOT / anchor.strip("/") / rel_text).resolve()
            if mapped.exists():
                return str(mapped)
    return text


def split_by_hash(key: Any, *, val_fraction: float = 0.10, test_fraction: float = 0.10, seed: int = 20260420) -> str:
    value = stable_hash_int(f"{seed}:{key}", 1000000) / 1000000.0
    if value < float(test_fraction):
        return "test"
    if value < float(test_fraction) + float(val_fraction):
        return "val"
    return "train"


def dataset_id(value: Any) -> int:
    text = str(value or "unknown").lower()
    try:
        return DATASET_VOCAB.index(text)
    except ValueError:
        return 0


def target_ar_id(value: Any) -> int:
    text = "FREE" if value is None or str(value).strip() == "" else str(value).strip()
    try:
        return TARGET_AR_VOCAB.index(text)
    except ValueError:
        return 0


def parse_ar(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        if ":" in text:
            left, right = text.split(":", 1)
            return float(left) / max(1e-6, float(right))
        return float(text)
    except (TypeError, ValueError):
        return None


def clamp_box(box: Sequence[Any] | None) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
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
    return [float(x1), float(y1), float(x2), float(y2)]


def box_area(box: Sequence[Any]) -> float:
    x1, y1, x2, y2 = clamp_box(box)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def box_iou(a: Sequence[Any], b: Sequence[Any]) -> float:
    ax1, ay1, ax2, ay2 = clamp_box(a)
    bx1, by1, bx2, by2 = clamp_box(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area([ax1, ay1, ax2, ay2]) + box_area([bx1, by1, bx2, by2]) - inter
    return float(inter / max(1e-6, union))


def normalize_score(value: float, *, min_value: float, max_value: float) -> float:
    denom = float(max_value) - float(min_value)
    if abs(denom) < 1e-8:
        return 0.5
    return float(max(0.0, min(1.0, (float(value) - float(min_value)) / denom)))


def load_warehouse_rows(path: Path, *, split: str | None = None, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split is not None and str(row.get("split", "")) != str(split):
                continue
            rows.append(row)
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def candidate_source_flags(candidate: dict[str, Any]) -> list[float]:
    text = " ".join(str(candidate.get(key, "")) for key in ("candidate_id", "source", "label")).lower()
    return [
        1.0 if "full" in text else 0.0,
        1.0 if "synthetic" in text else 0.0,
        1.0 if "dataset_candidate" in text or "cpc_view" in text else 0.0,
        1.0 if "gt" in text or "expert" in text or "editor" in text else 0.0,
    ]


def candidate_feature_vector(
    candidate: dict[str, Any],
    *,
    target_ar: Any = None,
    image_aspect: float = 1.0,
) -> list[float]:
    box = clamp_box(candidate.get("bbox_xyxy_norm", candidate.get("bbox_norm_xyxy")))
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    area = w * h
    aspect = w / h
    rendered_aspect = aspect * max(1e-6, float(image_aspect))
    target = parse_ar(target_ar)
    ar_log_error = 0.0 if target is None else abs(math.log(max(1e-6, rendered_aspect / max(1e-6, target))))
    center_dist = min(1.0, math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5))
    source_flags = candidate_source_flags(candidate)
    source_bucket = stable_hash_int(candidate.get("source", ""), UNKNOWN_SOURCE_BUCKETS) / max(1.0, float(UNKNOWN_SOURCE_BUCKETS - 1))
    feature = [
        x1,
        y1,
        x2,
        y2,
        cx,
        cy,
        w,
        h,
        area,
        math.log(max(1e-6, aspect)),
        math.log(max(1e-6, rendered_aspect)),
        center_dist,
        min(x1, 1.0 - x2),
        min(y1, 1.0 - y2),
        1.0 if x1 <= 1e-5 else 0.0,
        1.0 if y1 <= 1e-5 else 0.0,
        1.0 if x2 >= 1.0 - 1e-5 else 0.0,
        1.0 if y2 >= 1.0 - 1e-5 else 0.0,
        min(4.0, ar_log_error) / 4.0,
        box_iou(box, [0.0, 0.0, 1.0, 1.0]),
        *source_flags,
        source_bucket,
    ]
    if len(feature) != CANDIDATE_FEATURE_DIM:
        raise AssertionError(f"candidate feature dim mismatch: {len(feature)}")
    return [float(v) for v in feature]


def _open_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def _pil_to_tensor(image: Image.Image, *, size: int) -> torch.Tensor:
    resized = image.resize((int(size), int(size)), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _crop_tensor(image: Image.Image, box: Sequence[Any], *, size: int) -> torch.Tensor:
    width, height = image.size
    x1, y1, x2, y2 = clamp_box(box)
    left = int(round(x1 * width))
    upper = int(round(y1 * height))
    right = int(round(x2 * width))
    lower = int(round(y2 * height))
    right = max(left + 1, min(width, right))
    lower = max(upper + 1, min(height, lower))
    crop = image.crop((left, upper, right, lower))
    return _pil_to_tensor(crop, size=size)


def image_aspect_from_row(row: dict[str, Any], image: Image.Image | None = None) -> float:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    width = safe_float(meta.get("image_width"), 0.0)
    height = safe_float(meta.get("image_height"), 0.0)
    if width <= 0.0 or height <= 0.0:
        width = safe_float(row.get("width"), 0.0)
        height = safe_float(row.get("height"), 0.0)
    if (width <= 0.0 or height <= 0.0) and image is not None:
        width, height = image.size
    return float(max(1e-6, width / max(1e-6, height)))


class UniversalCropTeacherDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        max_candidates: int = 128,
        image_size: int = 224,
        crop_size: int = 160,
        include_images: bool = True,
        training: bool = False,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.max_candidates = int(max_candidates)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.include_images = bool(include_images)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.rows)

    def _candidate_indices(self, row: dict[str, Any], candidates: list[dict[str, Any]]) -> list[int]:
        if self.max_candidates <= 0 or len(candidates) <= self.max_candidates:
            return list(range(len(candidates)))
        if self.training and any("target_score" in cand for cand in candidates):
            dataset = str(row.get("dataset", "unknown")).lower()
            scores = [safe_float(cand.get("target_score"), 0.0) for cand in candidates]
            order = sorted(range(len(candidates)), key=lambda idx: (scores[idx], -idx), reverse=True)

            def push_unique(selected: list[int], seen: set[int], values: Sequence[int]) -> None:
                for value in values:
                    if len(selected) >= self.max_candidates:
                        break
                    idx = int(value)
                    if idx in seen:
                        continue
                    selected.append(idx)
                    seen.add(idx)

            if dataset == "gaic":
                keep_bottom = min(max(1, self.max_candidates // 8), max(0, len(order) - 1))
                keep_top = min(len(order), max(1, self.max_candidates // 2))
                if keep_top + keep_bottom >= self.max_candidates:
                    keep_top = max(1, self.max_candidates - keep_bottom - 1)
                selected: list[int] = []
                seen: set[int] = set()
                push_unique(selected, seen, order[:keep_top])
                if keep_bottom > 0:
                    push_unique(selected, seen, order[-keep_bottom:])
                remaining_budget = max(0, self.max_candidates - len(selected))
                if remaining_budget > 0:
                    remaining = [idx for idx in order if idx not in seen]
                    num_bins = max(4, min(12, remaining_budget))
                    for bin_idx in range(num_bins):
                        if not remaining:
                            break
                        left = int(round(len(remaining) * bin_idx / num_bins))
                        right = int(round(len(remaining) * (bin_idx + 1) / num_bins))
                        if left >= right:
                            continue
                        chunk = remaining[left:right]
                        pick = chunk[len(chunk) // 2]
                        push_unique(selected, seen, [pick])
                        if len(selected) >= self.max_candidates:
                            break
                    if len(selected) < self.max_candidates:
                        mid_pref = sorted(
                            remaining,
                            key=lambda idx: (
                                abs(scores[idx] - 0.5),
                                -scores[idx],
                                idx,
                            ),
                        )
                        for idx in mid_pref:
                            push_unique(selected, seen, [idx])
                            if len(selected) >= self.max_candidates:
                                break
                return sorted(selected)

            keep_top = max(1, self.max_candidates // 3)
            keep_bottom = max(1, self.max_candidates // 10)
            selected = []
            seen: set[int] = set()
            push_unique(selected, seen, order[:keep_top])
            push_unique(selected, seen, order[-keep_bottom:])
            remaining_budget = max(0, self.max_candidates - len(selected))
            if remaining_budget > 0:
                remaining = [idx for idx in order if idx not in seen]
                num_bins = max(4, min(10, remaining_budget))
                for bin_idx in range(num_bins):
                    left = int(round(len(remaining) * bin_idx / num_bins))
                    right = int(round(len(remaining) * (bin_idx + 1) / num_bins))
                    if left >= right:
                        continue
                    chunk = remaining[left:right]
                    pick = chunk[len(chunk) // 2]
                    push_unique(selected, seen, [pick])
                    if len(selected) >= self.max_candidates:
                        break
                if len(selected) < self.max_candidates:
                    stride = max(1, len(remaining) // max(1, self.max_candidates - len(selected)))
                    for idx in remaining[::stride]:
                        push_unique(selected, seen, [idx])
                        if len(selected) >= self.max_candidates:
                            break
            return sorted(selected)
        return list(range(self.max_candidates))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        candidates_all = [dict(cand) for cand in row.get("candidates", []) if isinstance(cand, dict)]
        if not candidates_all:
            candidates_all = [
                {
                    "candidate_id": "synthetic_full",
                    "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                    "target_score": 0.5,
                    "label_confidence": 0.1,
                    "source": "empty_fallback",
                }
            ]
        selected_indices = self._candidate_indices(row, candidates_all)
        candidates = [candidates_all[i] for i in selected_indices]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        cid_to_new = {str(c.get("candidate_id", "")): idx2 for idx2, c in enumerate(candidates)}

        image_path = remap_project_path(row.get("image_path", ""))
        image: Image.Image | None = None
        if self.include_images and image_path and Path(image_path).exists():
            try:
                image = _open_rgb(image_path)
            except Exception:
                image = None
        image_aspect = image_aspect_from_row(row, image)
        if image is None:
            global_image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            crop_images = torch.zeros((len(candidates), 3, self.crop_size, self.crop_size), dtype=torch.float32)
        else:
            global_image = _pil_to_tensor(image, size=self.image_size)
            crop_images = torch.stack(
                [
                    _crop_tensor(
                        image,
                        cand.get("bbox_xyxy_norm", cand.get("bbox_norm_xyxy")),
                        size=self.crop_size,
                    )
                    for cand in candidates
                ],
                dim=0,
            )

        features = torch.tensor(
            [
                candidate_feature_vector(cand, target_ar=row.get("target_ar"), image_aspect=image_aspect)
                for cand in candidates
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([safe_float(cand.get("target_score"), 0.0) for cand in candidates], dtype=torch.float32)
        confidence = torch.tensor([safe_float(cand.get("label_confidence"), 1.0) for cand in candidates], dtype=torch.float32)
        gt_iou = torch.tensor([safe_float(cand.get("gt_iou"), safe_float(cand.get("target_score"), 0.0)) for cand in candidates], dtype=torch.float32)

        pair_rows: list[list[float]] = []
        for pair in row.get("pairwise", []) or []:
            if not isinstance(pair, dict):
                continue
            a_idx = cid_to_new.get(str(pair.get("candidate_id_a", "")))
            b_idx = cid_to_new.get(str(pair.get("candidate_id_b", "")))
            if a_idx is None or b_idx is None:
                continue
            preferred = str(pair.get("preferred", "a")).lower()
            sign = 1.0 if preferred == "a" else -1.0
            pair_rows.append([float(a_idx), float(b_idx), sign, max(1e-6, safe_float(pair.get("weight"), 1.0))])
        if not pair_rows:
            # Fall back to target-derived pairs for non-CPC datasets.
            values = targets.tolist()
            for a_idx, a_val in enumerate(values):
                for b_idx, b_val in enumerate(values):
                    if a_idx >= b_idx or abs(a_val - b_val) < 0.15:
                        continue
                    sign = 1.0 if a_val > b_val else -1.0
                    pair_rows.append([float(a_idx), float(b_idx), sign, abs(float(a_val) - float(b_val))])

        return {
            "task_id": str(row.get("task_id", "")),
            "dataset": str(row.get("dataset", "unknown")).lower(),
            "split": str(row.get("split", "")),
            "image_id": str(row.get("image_id", "")),
            "image_path": image_path,
            "target_ar": row.get("target_ar"),
            "dataset_id": dataset_id(row.get("dataset")),
            "target_ar_id": target_ar_id(row.get("target_ar")),
            "global_image": global_image,
            "crop_images": crop_images,
            "candidate_features": features,
            "target_scores": targets.clamp(0.0, 1.0),
            "label_confidence": confidence.clamp(0.0, 1.0),
            "gt_iou": gt_iou.clamp(0.0, 1.0),
            "pairwise": torch.tensor(pair_rows, dtype=torch.float32) if pair_rows else torch.zeros((0, 4), dtype=torch.float32),
            "candidate_ids": [str(c.get("candidate_id", f"candidate_{i:04d}")) for i, c in enumerate(candidates)],
            "candidate_boxes": torch.tensor(
                [clamp_box(c.get("bbox_xyxy_norm", c.get("bbox_norm_xyxy"))) for c in candidates],
                dtype=torch.float32,
            ),
            "source_row": row,
            "selected_original_indices": selected_indices,
            "old_to_new": old_to_new,
        }


def collate_uctr_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty UCTR batch")
    batch_size = len(items)
    max_k = max(int(item["candidate_features"].shape[0]) for item in items)
    crop_shape = items[0]["crop_images"].shape[1:]
    feat_dim = int(items[0]["candidate_features"].shape[-1])

    global_images = torch.stack([item["global_image"] for item in items], dim=0)
    crop_images = torch.zeros((batch_size, max_k, *crop_shape), dtype=torch.float32)
    candidate_features = torch.zeros((batch_size, max_k, feat_dim), dtype=torch.float32)
    target_scores = torch.zeros((batch_size, max_k), dtype=torch.float32)
    label_confidence = torch.zeros((batch_size, max_k), dtype=torch.float32)
    gt_iou = torch.zeros((batch_size, max_k), dtype=torch.float32)
    candidate_boxes = torch.zeros((batch_size, max_k, 4), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)

    pairwise: list[torch.Tensor] = []
    candidate_ids: list[list[str]] = []
    source_rows: list[dict[str, Any]] = []
    for bidx, item in enumerate(items):
        k = int(item["candidate_features"].shape[0])
        crop_images[bidx, :k] = item["crop_images"]
        candidate_features[bidx, :k] = item["candidate_features"]
        target_scores[bidx, :k] = item["target_scores"]
        label_confidence[bidx, :k] = item["label_confidence"]
        gt_iou[bidx, :k] = item["gt_iou"]
        candidate_boxes[bidx, :k] = item["candidate_boxes"]
        valid_mask[bidx, :k] = True
        pairwise.append(item["pairwise"])
        candidate_ids.append(item["candidate_ids"])
        source_rows.append(item["source_row"])

    return {
        "task_ids": [item["task_id"] for item in items],
        "datasets": [item["dataset"] for item in items],
        "splits": [item["split"] for item in items],
        "image_ids": [item["image_id"] for item in items],
        "target_ars": [item["target_ar"] for item in items],
        "dataset_ids": torch.tensor([int(item["dataset_id"]) for item in items], dtype=torch.long),
        "target_ar_ids": torch.tensor([int(item["target_ar_id"]) for item in items], dtype=torch.long),
        "global_images": global_images,
        "crop_images": crop_images,
        "candidate_features": candidate_features,
        "target_scores": target_scores,
        "label_confidence": label_confidence,
        "gt_iou": gt_iou,
        "candidate_boxes": candidate_boxes,
        "valid_mask": valid_mask,
        "pairwise": pairwise,
        "candidate_ids": candidate_ids,
        "source_rows": source_rows,
    }
