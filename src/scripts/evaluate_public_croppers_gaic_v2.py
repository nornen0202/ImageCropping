#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import box_iou_xyxy
from mobilecropnet_v4.eval_utils import mean, write_jsonl
from mobilecropnet_v4.gaic_benchmark import (
    DEFAULT_RETURN_K,
    DEFAULT_TOP_N,
    evaluate_scored_records,
    load_gaic_annotation_records,
)

PUBLIC_TEACHER_ROOT = PROJECT_ROOT / "third_party/public_cropping_teachers"
PUBLIC_WEIGHT_ROOT = PROJECT_ROOT / "weights/public_cropping_teachers"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate public cropper baselines on GAIC v2 official annotations.")
    parser.add_argument(
        "--annotations_json",
        type=Path,
        default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json/instances_test.json",
    )
    parser.add_argument(
        "--image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
        ],
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="*", default=["cacnet", "cgs", "gaic"], choices=["cacnet", "cgs", "gaic", "s2cnet"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--save_per_image", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Raise on method failure instead of recording unavailable status.")
    return parser


def _safe_method_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _load_state_dict(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def _resample_lanczos() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.LANCZOS)
    return int(Image.LANCZOS)


def _norm_box_to_pixel_xyxy(box: Sequence[float], *, width: int, height: int) -> list[float]:
    x1 = max(0.0, min(float(width), float(box[0]) * float(width)))
    y1 = max(0.0, min(float(height), float(box[1]) * float(height)))
    x2 = max(0.0, min(float(width), float(box[2]) * float(width)))
    y2 = max(0.0, min(float(height), float(box[3]) * float(height)))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2, y2]


def _pixel_box_to_norm_xyxy(box: Sequence[float], *, width: int, height: int) -> list[float]:
    width_f = float(max(1, int(width)))
    height_f = float(max(1, int(height)))
    return [
        max(0.0, min(1.0, float(box[0]) / width_f)),
        max(0.0, min(1.0, float(box[1]) / height_f)),
        max(0.0, min(1.0, float(box[2]) / width_f)),
        max(0.0, min(1.0, float(box[3]) / height_f)),
    ]


def _resize_keep_min_side(width: int, height: int, min_side: float) -> tuple[int, int]:
    scale = float(min_side) / float(max(1, min(width, height)))
    out_h = int(round(float(height) * scale / 32.0) * 32)
    out_w = int(round(float(width) * scale / 32.0) * 32)
    return max(32, out_w), max(32, out_h)


def _append_paths(paths: Sequence[Path]) -> None:
    for path in reversed([str(Path(p)) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_module_from_file(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_cacnet(records: Sequence[dict[str, Any]], *, device: torch.device) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], dict[str, Any]]:
    model_dir = PUBLIC_TEACHER_ROOT / "cacnet"
    weight_path = PUBLIC_WEIGHT_ROOT / "cacnet/best-FLMS_iou.pth"
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)
    _append_paths([model_dir])
    from CACNet import CACNet  # type: ignore
    import cv2

    model = CACNet(loadweights=False)
    model.load_state_dict(_load_state_dict(weight_path, map_location="cpu"))
    model.to(device).eval()
    scores_by_image: dict[str, list[float]] = {}
    coverage_by_image: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, str]] = []
    elapsed_start = time.time()

    def flush_batch(batch: list[tuple[str, int, int, list[dict[str, Any]], torch.Tensor]]) -> None:
        if not batch:
            return
        tensors = torch.cat([item[4] for item in batch], dim=0).to(device)
        _, _, crops = model(tensors, only_classify=False)
        crops_list = crops.detach().cpu().float().tolist()
        for (image_id, width, height, candidates, _tensor), pred in zip(batch, crops_list):
            pred = list(pred)
            pred[0] = max(0.0, min(224.0, float(pred[0]))) / 224.0 * float(width)
            pred[1] = max(0.0, min(224.0, float(pred[1]))) / 224.0 * float(height)
            pred[2] = max(0.0, min(224.0, float(pred[2]))) / 224.0 * float(width)
            pred[3] = max(0.0, min(224.0, float(pred[3]))) / 224.0 * float(height)
            if pred[2] <= pred[0]:
                pred[2] = min(float(width), pred[0] + 1.0)
            if pred[3] <= pred[1]:
                pred[3] = min(float(height), pred[1] + 1.0)
            pred_norm = _pixel_box_to_norm_xyxy(pred, width=width, height=height)
            ious = [box_iou_xyxy(pred_norm, c.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])) for c in candidates]
            scores_by_image[image_id] = [float(v) for v in ious]
            coverage_by_image[image_id] = {
                "predicted_crop_candidate_iou_max": float(max(ious) if ious else 0.0),
                "predicted_crop_candidate_iou_mean": float(mean(ious)),
                "candidate_count": float(len(candidates)),
            }

    batch_size = 64 if device.type == "cuda" else 1
    batch: list[tuple[str, int, int, list[dict[str, Any]], torch.Tensor]] = []
    with torch.no_grad():
        for record in records:
            image_id = str(record["image_id"])
            image_path = Path(str(record.get("image_path", "")))
            candidates = list(record.get("candidates") or [])
            if not image_path.exists() or not candidates:
                skipped.append({"image_id": image_id, "reason": "missing_image_or_candidates"})
                continue
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                skipped.append({"image_id": image_id, "reason": "cv2_read_failed"})
                continue
            height, width = int(bgr.shape[0]), int(bgr.shape[1])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            resized = resized.astype(np.float32) / 255.0
            resized -= np.array(IMAGENET_MEAN, dtype=np.float32)
            resized /= np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(resized.transpose((2, 0, 1))).unsqueeze(0)
            batch.append((image_id, width, height, candidates, tensor))
            if len(batch) >= batch_size:
                flush_batch(batch)
                batch = []
        flush_batch(batch)
    return scores_by_image, coverage_by_image, {
        "model_kind": "single_crop_regressor_projected_by_iou",
        "code_dir": str(model_dir),
        "weight_path": str(weight_path),
        "device": str(device),
        "batch_size": batch_size,
        "preprocess": "cv2.imread + cv2.INTER_AREA resize to 224x224 + ImageNet normalization",
        "elapsed_sec": round(time.time() - elapsed_start, 3),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "metric_note": "CACNet emits one crop, so candidate scores are IoU to the emitted crop for GAIC annotation-space projection.",
    }


def _score_cgs(records: Sequence[dict[str, Any]], *, device: torch.device) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], dict[str, Any]]:
    if device.type != "cuda":
        raise RuntimeError("CGS evaluation requires CUDA because the bundled RoI/RoD Align extensions are CUDA ops.")
    model_dir = PUBLIC_TEACHER_ROOT / "cgs"
    extractor_weight = PUBLIC_WEIGHT_ROOT / "cgs/pretrained_model/pretrained_model/extractor-best-srcc.pth"
    gnn_weight = PUBLIC_WEIGHT_ROOT / "cgs/pretrained_model/pretrained_model/gnn-best-srcc.pth"
    for path in (model_dir, extractor_weight, gnn_weight):
        if not path.exists():
            raise FileNotFoundError(path)
    _append_paths([model_dir, model_dir / "roi_align", model_dir / "rod_align"])
    from croppingModel import CroppingGraph, RegionFeatureExtractor  # type: ignore

    extractor = RegionFeatureExtractor(loadweight=False)
    extractor.load_state_dict(_load_state_dict(extractor_weight, map_location="cpu"))
    gnn = CroppingGraph()
    gnn.load_state_dict(_load_state_dict(gnn_weight, map_location="cpu"))
    extractor.to(device).eval()
    gnn.to(device).eval()
    image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    scores_by_image: dict[str, list[float]] = {}
    coverage_by_image: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, str]] = []
    elapsed_start = time.time()
    with torch.no_grad():
        for record in records:
            image_id = str(record["image_id"])
            image_path = Path(str(record.get("image_path", "")))
            candidates = list(record.get("candidates") or [])
            if not image_path.exists() or not candidates:
                skipped.append({"image_id": image_id, "reason": "missing_image_or_candidates"})
                continue
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            resized_w, resized_h = _resize_keep_min_side(width, height, 256.0)
            resized = image.resize((resized_w, resized_h), _resample_lanczos())
            tensor = image_transform(resized).unsqueeze(0).to(device)
            ratio_w = float(resized_w) / float(width)
            ratio_h = float(resized_h) / float(height)
            rois = []
            for candidate in candidates:
                x1, y1, x2, y2 = _norm_box_to_pixel_xyxy(candidate["bbox_norm_xyxy"], width=width, height=height)
                rois.append([math.floor(x1 * ratio_w), math.floor(y1 * ratio_h), math.ceil(x2 * ratio_w), math.ceil(y2 * ratio_h)])
            rois_tensor = torch.tensor([rois], dtype=torch.float32, device=device)
            region_feat = extractor(tensor, rois_tensor)
            _, scores = gnn(region_feat)
            scores_by_image[image_id] = [float(v) for v in scores.reshape(-1).detach().cpu().tolist()]
            coverage_by_image[image_id] = {"candidate_count": float(len(candidates)), "scored_candidate_rate": 1.0}
    return scores_by_image, coverage_by_image, {
        "model_kind": "candidate_ranker",
        "code_dir": str(model_dir),
        "extractor_weight": str(extractor_weight),
        "gnn_weight": str(gnn_weight),
        "device": str(device),
        "elapsed_sec": round(time.time() - elapsed_start, 3),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }


def _score_gaic(records: Sequence[dict[str, Any]], *, device: torch.device) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], dict[str, Any]]:
    if device.type != "cuda":
        raise RuntimeError("GAIC evaluation requires CUDA because the replacement RoI/RoD Align extensions are CUDA ops.")
    model_dir = PUBLIC_TEACHER_ROOT / "gaic"
    cgs_dir = PUBLIC_TEACHER_ROOT / "cgs"
    weight_path = PUBLIC_WEIGHT_ROOT / "gaic/mobilenet_0.682_0.643_0.613_0.585_0.844_0.827_0.807_0.787_0.849_0.874.pth"
    for path in (model_dir, cgs_dir, weight_path):
        if not path.exists():
            raise FileNotFoundError(path)
    # The GAIC checkout contains stale Python 2/3.5 pyc extension packages.
    # Reuse the locally built Py3.10 CGS RoI/RoD Align modules, which expose the same package names used here.
    _append_paths([cgs_dir, cgs_dir / "roi_align", cgs_dir / "rod_align", model_dir])
    gaic_model_module = _load_module_from_file("gaic_public_cropping_model", model_dir / "croppingModel.py")
    model = gaic_model_module.build_crop_model(scale="multi", alignsize=9, reddim=8, loadweight=False, model="mobilenetv2", downsample=4)
    model.load_state_dict(_load_state_dict(weight_path, map_location="cpu"))
    model.to(device).eval()
    scores_by_image: dict[str, list[float]] = {}
    coverage_by_image: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, str]] = []
    elapsed_start = time.time()
    with torch.no_grad():
        for record in records:
            image_id = str(record["image_id"])
            image_path = Path(str(record.get("image_path", "")))
            candidates = list(record.get("candidates") or [])
            if not image_path.exists() or not candidates:
                skipped.append({"image_id": image_id, "reason": "missing_image_or_candidates"})
                continue
            import cv2

            bgr = cv2.imread(str(image_path))
            if bgr is None:
                skipped.append({"image_id": image_id, "reason": "cv2_read_failed"})
                continue
            rgb = bgr[:, :, (2, 1, 0)]
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            resized_w, resized_h = _resize_keep_min_side(width, height, 256.0)
            resized = cv2.resize(rgb, (resized_w, resized_h)) / 256.0
            resized = resized.astype(np.float32)
            resized -= np.array(IMAGENET_MEAN, dtype=np.float32)
            resized /= np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(resized.transpose((2, 0, 1))).unsqueeze(0).float().to(device)
            ratio_w = float(resized_w) / float(width)
            ratio_h = float(resized_h) / float(height)
            rois = []
            for candidate in candidates:
                x1, y1, x2, y2 = _norm_box_to_pixel_xyxy(candidate["bbox_norm_xyxy"], width=width, height=height)
                rois.append([0.0, math.floor(x1 * ratio_w), math.floor(y1 * ratio_h), math.ceil(x2 * ratio_w), math.ceil(y2 * ratio_h)])
            roi_tensor = torch.tensor(rois, dtype=torch.float32, device=device)
            scores = model(tensor, roi_tensor)
            scores_by_image[image_id] = [float(v) for v in scores.reshape(-1).detach().cpu().tolist()]
            coverage_by_image[image_id] = {"candidate_count": float(len(candidates)), "scored_candidate_rate": 1.0}
    return scores_by_image, coverage_by_image, {
        "model_kind": "candidate_ranker",
        "code_dir": str(model_dir),
        "weight_path": str(weight_path),
        "replacement_extension_source": str(cgs_dir),
        "device": str(device),
        "elapsed_sec": round(time.time() - elapsed_start, 3),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "compatibility_note": "GAIC's bundled Python2/3.5 roi_align/rod_align artifacts are incompatible with Py3.10; the evaluation uses the Py3.10-built CGS extensions with the same API.",
    }


def _s2cnet_graph_node_boxes(candidates: Sequence[dict[str, Any]]) -> list[list[float]]:
    def area(box: Sequence[float]) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    def iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        denom = area(a) + area(b) - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    boxes: list[list[float]] = []
    raw_boxes: list[list[float]] = []
    for candidate in candidates:
        box = candidate.get("bbox_norm_xyxy", candidate.get("bbox_xyxy_norm"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        raw_boxes.append([max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)), max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))])

    for box in sorted(raw_boxes, key=area, reverse=True):
        if len(boxes) >= 5:
            break
        if area(box) > 0.985:
            continue
        if all(iou(box, prev) < 0.85 for prev in boxes):
            boxes.append(box)

    fallback = [
        [0.0, 0.0, 1.0, 1.0],
        [0.2, 0.2, 0.8, 0.8],
        [0.0, 0.15, 0.62, 0.85],
        [0.38, 0.15, 1.0, 0.85],
        [0.15, 0.0, 0.85, 0.62],
    ]
    for box in fallback:
        if len(boxes) >= 5:
            break
        if all(iou(box, prev) < 0.85 for prev in boxes):
            boxes.append(box)
    while len(boxes) < 5:
        boxes.append(fallback[len(boxes) % len(fallback)])
    return boxes[:5]


def _score_s2cnet(records: Sequence[dict[str, Any]], *, device: torch.device) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], dict[str, Any]]:
    if device.type != "cuda":
        raise RuntimeError("S2CNet evaluation requires CUDA because the RoI/RoD Align extensions are CUDA ops.")
    import argparse
    import cv2

    model_dir = PUBLIC_TEACHER_ROOT / "s2cnet"
    cgs_dir = PUBLIC_TEACHER_ROOT / "cgs"
    weight_path = PUBLIC_WEIGHT_ROOT / "s2cnet/s2cnet_gaicv2.pth"
    for path in (model_dir, cgs_dir, weight_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # S2CNet uses the same roi_align/rod_align package names. Reuse the already built CGS Py3.10 extensions.
    _append_paths([cgs_dir, cgs_dir / "roi_align", cgs_dir / "rod_align", model_dir])
    from model.ssc import SSC  # type: ignore

    cfg = argparse.Namespace(
        base_model="mobilenetv2",
        loadweight=False,
        downsample=4,
        align_size=15,
        reduced_dim=8,
        num_feature_node=256,
        num_features_relation=256,
        num_depth=2,
        num_heads=4,
        mlp_dim=1024,
        only_crop=False,
        bbox_num=5,
    )
    model = SSC(cfg)
    model.load_state_dict(_load_state_dict(weight_path, map_location="cpu"))
    model.to(device).eval()

    scores_by_image: dict[str, list[float]] = {}
    coverage_by_image: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, str]] = []
    elapsed_start = time.time()
    with torch.no_grad():
        for record in records:
            image_id = str(record["image_id"])
            image_path = Path(str(record.get("image_path", "")))
            candidates = list(record.get("candidates") or [])
            if not image_path.exists() or not candidates:
                skipped.append({"image_id": image_id, "reason": "missing_image_or_candidates"})
                continue
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                skipped.append({"image_id": image_id, "reason": "cv2_read_failed"})
                continue
            rgb = bgr[:, :, (2, 1, 0)]
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            resized_w, resized_h = _resize_keep_min_side(width, height, 256.0)
            resized = cv2.resize(rgb, (resized_w, resized_h)) / 256.0
            resized = resized.astype(np.float32)
            resized -= np.array(IMAGENET_MEAN, dtype=np.float32)
            resized /= np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(resized.transpose((2, 0, 1))).unsqueeze(0).float().to(device)
            ratio_w = float(resized_w) / float(width)
            ratio_h = float(resized_h) / float(height)

            rois = []
            for candidate in candidates:
                x1, y1, x2, y2 = _norm_box_to_pixel_xyxy(candidate["bbox_norm_xyxy"], width=width, height=height)
                rois.append([0.0, math.floor(x1 * ratio_w), math.floor(y1 * ratio_h), math.ceil(x2 * ratio_w), math.ceil(y2 * ratio_h)])
            graph_nodes = []
            for box in _s2cnet_graph_node_boxes(candidates):
                x1, y1, x2, y2 = _norm_box_to_pixel_xyxy(box, width=width, height=height)
                graph_nodes.append([0.0, math.floor(x1 * ratio_w), math.floor(y1 * ratio_h), math.ceil(x2 * ratio_w), math.ceil(y2 * ratio_h)])
            roi_tensor = torch.tensor(rois, dtype=torch.float32, device=device)
            graph_tensor = torch.tensor(graph_nodes, dtype=torch.float32, device=device)
            scores = model(tensor, roi_tensor, graph_tensor)
            scores_by_image[image_id] = [float(v) for v in scores.reshape(-1).detach().cpu().tolist()]
            coverage_by_image[image_id] = {
                "candidate_count": float(len(candidates)),
                "scored_candidate_rate": 1.0,
                "s2cnet_graph_node_count": float(len(graph_nodes)),
            }
    return scores_by_image, coverage_by_image, {
        "model_kind": "candidate_ranker_with_heuristic_graph_nodes",
        "code_dir": str(model_dir),
        "weight_path": str(weight_path),
        "replacement_extension_source": str(cgs_dir),
        "device": str(device),
        "elapsed_sec": round(time.time() - elapsed_start, 3),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "compatibility_note": "S2CNet's original evaluation uses 5 Faster-RCNN object boxes per image. This project benchmark manifest does not include those object boxes, so scoring uses 5 deterministic candidate-derived graph nodes. Treat the row as a public-weight diagnostic, not an exact reproduction of the paper protocol.",
    }


def _oracle_scores(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    return {str(record["image_id"]): [float(c.get("mos", 0.0)) for c in record.get("candidates", [])] for record in records}


def _evaluate_method(
    method: str,
    records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    scorer = {
        "cacnet": _score_cacnet,
        "cgs": _score_cgs,
        "gaic": _score_gaic,
        "s2cnet": _score_s2cnet,
    }[method]
    scores, coverage, metadata = scorer(records, device=device)
    evaluated = evaluate_scored_records(
        records,
        scores,
        method_name=method,
        coverage_by_image_id=coverage,
        return_k_values=DEFAULT_RETURN_K,
        top_n_values=DEFAULT_TOP_N,
    )
    evaluated["metadata"] = metadata
    evaluated["status"] = "evaluated"
    return evaluated


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


PRIMARY_GAIC_METRIC_KEYS = (
    "pcc",
    "srcc",
    "acc1_of_top5",
    "acc1_of_top10",
    "acc2_of_top5",
    "acc2_of_top10",
    "acc3_of_top5",
    "acc3_of_top10",
    "acc4_of_top5",
    "acc4_of_top10",
    "accw1_of_top5",
    "accw1_of_top10",
    "accw2_of_top5",
    "accw2_of_top10",
    "accw3_of_top5",
    "accw3_of_top10",
    "accw4_of_top5",
    "accw4_of_top10",
    "top1_in_top5",
    "top1_in_top10",
    "top1_mos",
    "top1_mos_regret",
    "top1_rank",
    "top1_rank_percentile",
)


def _mark_cacnet_metrics_as_projection_only(row: dict[str, Any]) -> None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return
    row["projection_metrics"] = {key: metrics.get(key) for key in PRIMARY_GAIC_METRIC_KEYS if key in metrics}
    row["metric_applicability"] = {
        "primary_gaic_candidate_ranking_metrics": "not_applicable",
        "reason": "CACNet emits one crop and does not provide native scores for the official GAIC candidate set.",
        "projection_note": "projection_metrics were computed by ranking official candidates by IoU to the emitted crop and are diagnostic only.",
    }
    for key in PRIMARY_GAIC_METRIC_KEYS:
        if key in metrics:
            metrics[key] = None


def _metric_cell(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "-"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Public Cropper GAIC v2 벤치마크 평가 보고서",
        "",
        f"- annotations_json: `{summary['annotations_json']}`",
        f"- 이미지 수: {summary['image_count']}",
        f"- 후보 수: {summary['candidate_count']}",
        f"- 실행 장치: `{summary['device']}`",
        "",
        "## 사용 가능성",
        "",
        "| 방법 | 상태 | 사유 / 메모 |",
        "| --- | --- | --- |",
    ]
    for method in summary["method_order"]:
        row = summary["methods"][method]
        status = row.get("status", "")
        note = row.get("error") or row.get("metadata", {}).get("compatibility_note") or row.get("metadata", {}).get("metric_note") or ""
        lines.append(f"| `{method}` | {status} | {str(note).replace('|', '/')} |")
    lines.extend(
        [
            "",
            "## 지표 정의",
            "",
            "- `PCC`, `SRCC`: official GAIC MOS and predicted candidate scores의 이미지별 상관계수를 평균한 값이며 높을수록 좋다.",
            "- `AccK/N`: 모델이 반환한 상위 K개 후보 중 official MOS 상위 N개에 포함되는 비율이며 높을수록 좋다.",
            "- `AccwK/N`: `AccK/N`에 official rank와 반환 순서의 차이를 반영한 rank-weighted 지표이며 높을수록 좋다.",
            "- `top1 MOS`: 모델 top-1 crop의 official MOS이며 높을수록 좋다.",
            "- `MOS regret`: 이미지 내 최고 official MOS와 top-1 MOS의 차이며 낮을수록 좋다.",
            "- `top1 rank pct`: top-1 crop의 official rank percentile이며 1.0에 가까울수록 좋다.",
            "- `-`: 모델 출력 형식상 해당 GAIC candidate-ranking 지표를 원 방식으로 계산할 수 없음을 뜻한다.",
            "",
            "## 방법별 비교",
            "",
            "| method | images | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Accw4/5 | top1 MOS | top1 rank pct | MOS regret |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in summary["method_order"]:
        row = summary["methods"][method]
        metrics = row.get("metrics", {})
        lines.append(
            "| {method} | {images} | {pcc} | {srcc} | {acc15} | {acc110} | {acc45} | {accw45} | {mos} | {pct} | {regret} |".format(
                method=f"`{method}`",
                images=int(row.get("image_count", 0)),
                pcc=_metric_cell(metrics, "pcc"),
                srcc=_metric_cell(metrics, "srcc"),
                acc15=_metric_cell(metrics, "acc1_of_top5"),
                acc110=_metric_cell(metrics, "acc1_of_top10"),
                acc45=_metric_cell(metrics, "acc4_of_top5"),
                accw45=_metric_cell(metrics, "accw4_of_top5"),
                mos=_metric_cell(metrics, "top1_mos"),
                pct=_metric_cell(metrics, "top1_rank_percentile"),
                regret=_metric_cell(metrics, "top1_mos_regret"),
            )
        )
    cacnet = summary["methods"].get("cacnet")
    if isinstance(cacnet, dict) and isinstance(cacnet.get("projection_metrics"), dict):
        projection = cacnet["projection_metrics"]
        metrics = cacnet.get("metrics", {})
        lines.extend(
            [
                "",
                "## CACNet 투영 진단",
                "",
                "CACNet은 단일 crop만 출력하므로 위 GAIC candidate-ranking 표에서는 해당 지표를 `-`로 표시했다. 아래 값은 예측 crop과 official candidate 간 IoU 투영으로 계산한 참고 진단값이며, 원 GAIC 평가 지표로 해석하면 안 된다.",
                "",
                "| images | projected top1 MOS | projected MOS regret | projected top1 rank pct | candidate IoU max | candidate IoU mean |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {images} | {mos} | {regret} | {pct} | {iou_max:.6f} | {iou_mean:.6f} |".format(
                    images=int(cacnet.get("image_count", 0)),
                    mos=_metric_cell(projection, "top1_mos"),
                    regret=_metric_cell(projection, "top1_mos_regret"),
                    pct=_metric_cell(projection, "top1_rank_percentile"),
                    iou_max=_metric(metrics, "coverage_predicted_crop_candidate_iou_max"),
                    iou_mean=_metric(metrics, "coverage_predicted_crop_candidate_iou_mean"),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 해석 메모",
            "",
            "- `gaic_mos_oracle`은 공식 MOS 자체로 후보를 정렬한 상한선이며 배포 가능한 모델이 아니다.",
            "- CACNet은 단일 crop regressor이므로 기본 비교표에서 native GAIC candidate-ranking 지표는 적용 불가로 표시한다.",
            "- CGS와 GAIC는 candidate ranker이므로 공식 GAIC annotation 후보에서 직접 평가한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_gaic_annotation_records(args.annotations_json, image_roots=args.image_roots, max_images=args.max_images)
    records = [record for record in records if str(record.get("image_path", "")).strip()]
    device = torch.device(args.device)
    candidate_count = sum(len(record.get("candidates") or []) for record in records)

    methods: dict[str, Any] = {}
    method_order: list[str] = []
    for method in args.methods:
        method_order.append(method)
        try:
            methods[method] = _evaluate_method(method, records, device=device)
            if method == "cacnet":
                _mark_cacnet_metrics_as_projection_only(methods[method])
        except Exception as exc:
            if args.strict:
                raise
            methods[method] = {
                "method": method,
                "status": "unavailable",
                "image_count": 0,
                "missing_image_count": len(records),
                "metrics": {},
                "metadata": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    method_order.append("gaic_mos_oracle")
    methods["gaic_mos_oracle"] = evaluate_scored_records(
        records,
        _oracle_scores(records),
        method_name="gaic_mos_oracle",
        return_k_values=DEFAULT_RETURN_K,
        top_n_values=DEFAULT_TOP_N,
    )
    methods["gaic_mos_oracle"]["status"] = "reference"

    summary = {
        "annotations_json": str(args.annotations_json),
        "image_roots": [str(p) for p in args.image_roots],
        "image_count": len(records),
        "candidate_count": candidate_count,
        "candidate_count_mean": float(candidate_count / max(1, len(records))),
        "device": str(device),
        "method_order": method_order,
        "methods": {name: {k: v for k, v in row.items() if k != "per_image"} for name, row in methods.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_per_image:
        for name, row in methods.items():
            if "per_image" in row:
                write_jsonl(args.output_dir / f"{_safe_method_name(name)}_per_image.jsonl", row["per_image"])
    _write_report(args.output_dir / "PUBLIC_CROPPER_GAIC_V2_EVAL_REPORT.md", summary)
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
