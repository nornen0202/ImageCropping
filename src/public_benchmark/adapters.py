from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from PIL import Image

from public_benchmark.schema import BenchmarkBox, BenchmarkTask, PairwisePreference


GNMC_AR_ORDER = ("1:1", "4:3", "3:4", "16:9", "2:1")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def valid_box(box: list[float]) -> bool:
    return len(box) == 4 and all(math.isfinite(float(v)) for v in box) and box[2] > box[0] and box[3] > box[1]


def normalize_box(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    out = [clamp(x1), clamp(y1), clamp(x2), clamp(y2)]
    if not valid_box(out):
        return [0.0, 0.0, 1.0, 1.0]
    return [round(v, 6) for v in out]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return max(1, int(width)), max(1, int(height))


def _first_existing_dataset_root(candidates: Iterable[Path], required_paths: Iterable[str], dataset_name: str) -> Path:
    for root in candidates:
        if all((root / rel).exists() for rel in required_paths):
            return root
    checked = ", ".join(str(path) for path in candidates)
    required = ", ".join(required_paths)
    raise FileNotFoundError(f"{dataset_name} root not found. required=[{required}] checked=[{checked}]")


def resolve_fcdb_root(data_root: Path) -> Path:
    return _first_existing_dataset_root(
        [data_root / "FCDB", data_root],
        ["FCDB-all.json", "images"],
        "FCDB",
    )


def resolve_cpc_root(data_root: Path) -> Path:
    return _first_existing_dataset_root(
        [
            data_root / "CPCDataset",
            data_root / "cpc" / "extracted" / "CPCDataset",
            data_root / "downloads" / "cpc" / "extracted" / "CPCDataset",
            data_root,
        ],
        ["CollectedAnnotationsRaw", "images"],
        "CPCDataset",
    )


def resolve_gnmc_root(data_root: Path) -> Path:
    return _first_existing_dataset_root(
        [
            data_root / "GNMC",
            data_root / "GNMC" / "GNMC",
            data_root,
        ],
        ["json", "train", "validation", "test"],
        "GNMC",
    )


def xywh_pixels_to_norm(box: Iterable[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return normalize_box([x / width, y / height, (x + w) / width, (y + h) / height])


def xyxy_pixels_to_norm(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return normalize_box([x1 / width, y1 / height, x2 / width, y2 / height])


def box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_aspect(box: list[float]) -> float:
    return max(1e-6, float(box[2]) - float(box[0])) / max(1e-6, float(box[3]) - float(box[1]))


def parse_ar(ar_text: Optional[str], fallback: float = 1.0) -> float:
    if not ar_text:
        return float(fallback)
    text = str(ar_text).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        den = max(1e-6, float(right))
        return float(left) / den
    return float(text)


def _box_from_center_ar(cx: float, cy: float, aspect: float, area: float) -> list[float]:
    area = clamp(area, 1e-6, 1.0)
    aspect = max(1e-6, float(aspect))
    width = math.sqrt(area * aspect)
    height = math.sqrt(area / aspect)
    scale = min(1.0, 1.0 / max(width, 1e-6), 1.0 / max(height, 1e-6))
    width *= scale
    height *= scale
    x1 = clamp(cx - width / 2.0)
    y1 = clamp(cy - height / 2.0)
    x2 = clamp(x1 + width)
    y2 = clamp(y1 + height)
    x1 = clamp(x2 - width)
    y1 = clamp(y2 - height)
    return normalize_box([x1, y1, x2, y2])


def synthetic_windows(*, target_ar: float, gt_box: Optional[list[float]] = None, include_full: bool = True) -> list[BenchmarkBox]:
    windows: list[BenchmarkBox] = []
    if include_full:
        windows.append(
            BenchmarkBox(
                candidate_id="synthetic_full",
                bbox_xyxy_norm=[0.0, 0.0, 1.0, 1.0],
                label="synthetic_full",
                source="synthetic_eval_window",
            )
        )
    areas = (0.35, 0.55, 0.75, 0.90)
    centers = [(0.5, 0.5), (0.38, 0.5), (0.62, 0.5), (0.5, 0.38), (0.5, 0.62)]
    idx = 0
    for area in areas:
        for cx, cy in centers:
            idx += 1
            windows.append(
                BenchmarkBox(
                    candidate_id=f"synthetic_{idx:03d}",
                    bbox_xyxy_norm=_box_from_center_ar(cx, cy, target_ar, area),
                    label="synthetic_window",
                    source="synthetic_eval_window",
                )
            )
    if gt_box is not None:
        gt_ar = box_aspect(gt_box)
        for area_scale in (0.70, 1.25):
            idx += 1
            cx = 0.5 * (gt_box[0] + gt_box[2])
            cy = 0.5 * (gt_box[1] + gt_box[3])
            windows.append(
                BenchmarkBox(
                    candidate_id=f"synthetic_gt_ar_{idx:03d}",
                    bbox_xyxy_norm=_box_from_center_ar(cx, cy, gt_ar, clamp(box_area(gt_box) * area_scale, 0.05, 0.95)),
                    label="synthetic_gt_ar_window",
                    source="synthetic_eval_window",
                )
            )
    return _dedupe_boxes(windows)


def _dedupe_boxes(boxes: list[BenchmarkBox], tol_digits: int = 4) -> list[BenchmarkBox]:
    out: list[BenchmarkBox] = []
    seen: set[tuple[float, float, float, float]] = set()
    for box in boxes:
        key = tuple(round(float(v), tol_digits) for v in box.bbox_xyxy_norm)
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out


def _fcdb_image_path(root: Path, url: str) -> Optional[Path]:
    name = Path(str(url)).name
    direct = root / "images" / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    matches = sorted((root / "images").glob(f"{stem}.*"))
    return matches[0] if matches else None


def iter_fcdb_tasks(data_root: Path, *, split: str = "all") -> Iterator[BenchmarkTask]:
    root = resolve_fcdb_root(data_root)
    ann_path = root / "FCDB-all.json"
    rows = json.loads(ann_path.read_text(encoding="utf-8"))
    for row in rows:
        image_path = _fcdb_image_path(root, str(row.get("url", "")))
        if image_path is None:
            continue
        width, height = image_size(image_path)
        image_id = str(row.get("flickr_photo_id") or Path(str(row.get("url", ""))).stem)
        gt_box = xywh_pixels_to_norm(row.get("crop", [0, 0, width, height]), width, height)
        target_ar = box_aspect(gt_box)
        gt = BenchmarkBox(
            candidate_id="fcdb_gt_0",
            bbox_xyxy_norm=gt_box,
            label="expert_crop",
            source="fcdb_gt",
            weight=1.0,
        )
        yield BenchmarkTask(
            dataset="fcdb",
            split=split,
            image_id=image_id,
            image_path=str(image_path.resolve()),
            task_type="freeform",
            target_ar=None,
            gt_boxes=[gt],
            pairwise=[],
            candidate_windows=synthetic_windows(target_ar=target_ar, gt_box=gt_box),
            meta={
                "annotation_path": str(ann_path),
                "image_width": width,
                "image_height": height,
                "note": "Local FCDB copy contains GT crops only; pairwise annotations are not present.",
            },
        )


def _load_cpc_annotation(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    return json.loads(json.loads(raw))


def _build_cpc_pairwise(
    *,
    candidates: list[BenchmarkBox],
    annotator_scores: list[list[float]],
    max_pairs: int,
    min_gap: float,
) -> list[PairwisePreference]:
    pairs: list[tuple[float, PairwisePreference]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            score_i = float(candidates[i].score if candidates[i].score is not None else 0.0)
            score_j = float(candidates[j].score if candidates[j].score is not None else 0.0)
            gap = abs(score_i - score_j)
            if gap < float(min_gap):
                continue
            votes_i = 0
            votes_j = 0
            ties = 0
            for scores in annotator_scores:
                if i >= len(scores) or j >= len(scores):
                    continue
                if float(scores[i]) > float(scores[j]):
                    votes_i += 1
                elif float(scores[j]) > float(scores[i]):
                    votes_j += 1
                else:
                    ties += 1
            if score_i == score_j and votes_i == votes_j:
                continue
            preferred = "a" if (score_i, votes_i) >= (score_j, votes_j) else "b"
            pair = PairwisePreference(
                candidate_id_a=candidates[i].candidate_id,
                candidate_id_b=candidates[j].candidate_id,
                bbox_a_xyxy_norm=candidates[i].bbox_xyxy_norm,
                bbox_b_xyxy_norm=candidates[j].bbox_xyxy_norm,
                preferred=preferred,
                weight=max(gap, abs(votes_i - votes_j), 1.0),
                votes={"a": float(votes_i), "b": float(votes_j), "ties": float(ties)},
                meta={"score_a": score_i, "score_b": score_j},
            )
            pairs.append((gap, pair))
    pairs.sort(key=lambda item: (item[0], item[1].weight), reverse=True)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return [pair for _, pair in pairs]


def iter_cpc_tasks(
    data_root: Path,
    *,
    split: str = "train",
    max_pairwise_per_task: int = 200,
    min_pairwise_gap: float = 0.0,
) -> Iterator[BenchmarkTask]:
    base = resolve_cpc_root(data_root)
    ann_dir = base / "CollectedAnnotationsRaw"
    image_dir = base / "images"
    for ann_path in sorted(ann_dir.glob("*.txt")):
        image_name = ann_path.name[:-4]
        image_path = image_dir / image_name
        if not image_path.exists():
            continue
        width, height = image_size(image_path)
        payload = _load_cpc_annotation(ann_path)
        bboxes = payload.get("bboxes", [])
        scores = payload.get("scores", [])
        candidates: list[BenchmarkBox] = []
        for idx, bbox in enumerate(bboxes):
            annotator_values = [float(row[idx]) for row in scores if idx < len(row)]
            if not annotator_values:
                continue
            mean_score = sum(annotator_values) / float(len(annotator_values))
            candidates.append(
                BenchmarkBox(
                    candidate_id=f"cpc_view_{idx:02d}",
                    bbox_xyxy_norm=xyxy_pixels_to_norm(bbox, width, height),
                    label="candidate_view",
                    source="dataset_candidate",
                    score=float(mean_score),
                    meta={"score_norm": float(mean_score) / 5.0},
                )
            )
        if not candidates:
            continue
        best_score = max(float(c.score or 0.0) for c in candidates)
        gt_boxes = [
            BenchmarkBox(
                candidate_id=f"cpc_gt_{cand.candidate_id}",
                bbox_xyxy_norm=cand.bbox_xyxy_norm,
                label="annotator_top_view",
                source="cpc_top_view",
                score=cand.score,
            )
            for cand in candidates
            if float(cand.score or 0.0) == best_score
        ]
        yield BenchmarkTask(
            dataset="cpc",
            split=split,
            image_id=Path(image_name).stem,
            image_path=str(image_path.resolve()),
            task_type="freeform",
            target_ar=None,
            gt_boxes=gt_boxes,
            pairwise=_build_cpc_pairwise(
                candidates=candidates,
                annotator_scores=[[float(v) for v in row] for row in scores],
                max_pairs=max_pairwise_per_task,
                min_gap=min_pairwise_gap,
            ),
            candidate_windows=candidates,
            meta={
                "annotation_path": str(ann_path),
                "image_width": width,
                "image_height": height,
                "annotator_count": len(scores),
            },
        )


def _gnmc_json_paths(root: Path, split: str) -> list[tuple[str, Path]]:
    json_dir = root / "json"
    if split == "all":
        return [(name, json_dir / f"{name}.json") for name in ("train", "validation", "test")]
    return [(split, json_dir / f"{split}.json")]


def iter_gnmc_tasks(data_root: Path, *, split: str = "all") -> Iterator[BenchmarkTask]:
    root = resolve_gnmc_root(data_root)
    for split_name, json_path in _gnmc_json_paths(root, split):
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        image_dir = root / split_name
        for row in rows:
            filename = str(row.get("filename", ""))
            image_path = image_dir / filename
            if not image_path.exists():
                continue
            width, height = image_size(image_path)
            for ar_text in GNMC_AR_ORDER:
                raw_box = row.get("crop_bboxes", {}).get(ar_text)
                if raw_box is None:
                    continue
                gt_box = normalize_box(raw_box)
                image_aspect = width / max(1e-6, float(height))
                normalized_target_ar = parse_ar(ar_text) / max(1e-6, image_aspect)
                gt = BenchmarkBox(
                    candidate_id=f"gnmc_gt_{ar_text}",
                    bbox_xyxy_norm=gt_box,
                    label="editor_crop",
                    source="gnmc_gt",
                    weight=1.0,
                )
                yield BenchmarkTask(
                    dataset="gnmc",
                    split=split_name,
                    image_id=Path(filename).stem,
                    image_path=str(image_path.resolve()),
                    task_type="ar_conditioned",
                    target_ar=ar_text,
                    gt_boxes=[gt],
                    pairwise=[],
                    candidate_windows=synthetic_windows(target_ar=normalized_target_ar, gt_box=gt_box),
                    meta={
                        "annotation_path": str(json_path),
                        "image_width": width,
                        "image_height": height,
                    },
                )


def iter_tasks(
    dataset: str,
    data_root: Path,
    *,
    split: str = "all",
    max_pairwise_per_task: int = 200,
    min_pairwise_gap: float = 0.0,
) -> Iterator[BenchmarkTask]:
    key = dataset.lower()
    if key == "fcdb":
        yield from iter_fcdb_tasks(data_root, split=split)
    elif key == "cpc":
        cpc_split = "train" if split == "all" else split
        yield from iter_cpc_tasks(
            data_root,
            split=cpc_split,
            max_pairwise_per_task=max_pairwise_per_task,
            min_pairwise_gap=min_pairwise_gap,
        )
    elif key == "gnmc":
        yield from iter_gnmc_tasks(data_root, split=split)
    else:
        raise ValueError(f"Unsupported public benchmark dataset: {dataset}")
