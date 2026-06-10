#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageStat

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional on minimal hosts
    PCA = None  # type: ignore[assignment]
    ExtraTreesRegressor = None  # type: ignore[assignment]
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    make_pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.fit_gaic_teacher_mos_calibration import (  # noqa: E402
    DEFAULT_AUDIT_FIELDS,
    DEFAULT_EXTRA_FEATURE_FIELDS,
    DEFAULT_SCORE_FIELDS,
    build_feature_spec,
    evaluate_rows,
    objective,
    row_to_features,
    safe_float,
    safe_int,
    sample_weights,
    target_values,
)


HANDCRAFTED_FEATURE_NAMES = (
    "vis_crop_luma_mean",
    "vis_crop_luma_std",
    "vis_crop_luma_p10",
    "vis_crop_luma_p90",
    "vis_crop_contrast",
    "vis_crop_saturation_mean",
    "vis_crop_saturation_std",
    "vis_crop_colorfulness",
    "vis_crop_dark_clip",
    "vis_crop_bright_clip",
    "vis_crop_entropy",
    "vis_crop_laplacian_var",
    "vis_crop_edge_density",
    "vis_crop_rule_third_x",
    "vis_crop_rule_third_y",
    "vis_full_luma_mean",
    "vis_full_luma_std",
    "vis_full_saturation_mean",
    "vis_full_colorfulness",
    "vis_crop_minus_full_luma_mean",
    "vis_crop_minus_full_luma_std",
    "vis_crop_minus_full_saturation",
    "vis_crop_minus_full_colorfulness",
)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_rows(path: Path | None, *, protocol: str, max_rows: int = 0) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("protocol", "")) != str(protocol):
                continue
            rows.append(row)
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def split_dir_name(split_name: str, row: dict[str, Any] | None = None) -> str:
    raw = str((row or {}).get("official_split", "") or "").strip().lower()
    if raw in {"train", "val", "test"}:
        return raw
    split = str(split_name).strip().lower()
    if split.startswith("train"):
        return "train"
    if split.startswith("val"):
        return "val"
    if split.startswith("test"):
        return "test"
    return split


def resolve_gaic_image_path(row: dict[str, Any], *, split_name: str, image_root: Path) -> Path:
    raw = str(row.get("image_path", "") or "").strip()
    candidates: list[Path] = []
    if raw:
        raw_path = Path(raw)
        candidates.append(raw_path if raw_path.is_absolute() else image_root / raw_path)
        candidates.append(image_root / raw_path.name)
    image_id = str(row.get("image_id", "") or "").strip()
    if image_id:
        split_dir = split_dir_name(split_name, row)
        candidates.append(image_root / split_dir / f"{image_id}.jpg")
        candidates.append(image_root / split_dir / f"{image_id}.jpeg")
        candidates.append(image_root / split_dir / f"{image_id}.png")
        candidates.append(image_root / f"{image_id}.jpg")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    return image_root / "missing.jpg"


def clamp_box(box: Any) -> tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 0.0, 0.0, 1.0, 1.0
    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box[:4]]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-6)
    return x1, y1, x2, y2


def crop_image(image: Image.Image, box: Any) -> Image.Image:
    x1, y1, x2, y2 = clamp_box(box)
    width, height = image.size
    left = int(round(x1 * max(1, width)))
    top = int(round(y1 * max(1, height)))
    right = int(round(x2 * max(1, width)))
    bottom = int(round(y2 * max(1, height)))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    return image.crop((left, top, right, bottom))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _saturation(rgb: np.ndarray) -> np.ndarray:
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    return (maxc - minc) / np.maximum(maxc, 1e-6)


def _colorfulness(rgb: np.ndarray) -> float:
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))
    return float(math.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * math.sqrt(mean_rg * mean_rg + mean_yb * mean_yb))


def _entropy(luma_u8: np.ndarray) -> float:
    hist = np.bincount(luma_u8.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / max(1.0, float(hist.sum()))
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum() / 8.0)


def _edge_and_sharpness(luma_u8: np.ndarray) -> tuple[float, float]:
    try:
        import cv2  # type: ignore

        lap = cv2.Laplacian(luma_u8, cv2.CV_64F)
        sharp = float(np.var(lap) / (255.0 * 255.0))
        edges = cv2.Canny(luma_u8, 80, 180)
        edge_density = float(np.mean(edges > 0))
        return sharp, edge_density
    except Exception:
        gy, gx = np.gradient(luma_u8.astype(np.float32) / 255.0)
        grad = np.sqrt(gx * gx + gy * gy)
        return float(np.var(grad)), float(np.mean(grad > 0.12))


def _basic_visual_stats(image: Image.Image, *, max_side: int = 256) -> dict[str, float]:
    img = image.convert("RGB")
    width, height = img.size
    scale = float(max(width, height)) / float(max(1, max_side))
    if scale > 1.0:
        img = img.resize((max(1, int(round(width / scale))), max(1, int(round(height / scale)))), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
    luma = _luma(rgb)
    sat = _saturation(rgb)
    luma_u8 = np.clip(np.round(luma * 255.0), 0, 255).astype(np.uint8)
    sharp, edge_density = _edge_and_sharpness(luma_u8)
    return {
        "luma_mean": float(np.mean(luma)),
        "luma_std": float(np.std(luma)),
        "luma_p10": float(np.percentile(luma, 10)),
        "luma_p90": float(np.percentile(luma, 90)),
        "contrast": float(np.percentile(luma, 90) - np.percentile(luma, 10)),
        "saturation_mean": float(np.mean(sat)),
        "saturation_std": float(np.std(sat)),
        "colorfulness": _colorfulness(rgb),
        "dark_clip": float(np.mean(luma < 0.04)),
        "bright_clip": float(np.mean(luma > 0.96)),
        "entropy": _entropy(luma_u8),
        "laplacian_var": sharp,
        "edge_density": edge_density,
    }


def handcrafted_visual_features(row: dict[str, Any], image: Image.Image) -> list[float]:
    crop = crop_image(image, row.get("bbox_norm_xyxy"))
    crop_stats = _basic_visual_stats(crop)
    full_stats = _basic_visual_stats(image)
    x1, y1, x2, y2 = clamp_box(row.get("bbox_norm_xyxy"))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    third_x = max(0.0, 1.0 - min(abs(cx - 1.0 / 3.0), abs(cx - 2.0 / 3.0)) / (1.0 / 3.0))
    third_y = max(0.0, 1.0 - min(abs(cy - 1.0 / 3.0), abs(cy - 2.0 / 3.0)) / (1.0 / 3.0))
    values = [
        crop_stats["luma_mean"],
        crop_stats["luma_std"],
        crop_stats["luma_p10"],
        crop_stats["luma_p90"],
        crop_stats["contrast"],
        crop_stats["saturation_mean"],
        crop_stats["saturation_std"],
        crop_stats["colorfulness"],
        crop_stats["dark_clip"],
        crop_stats["bright_clip"],
        crop_stats["entropy"],
        crop_stats["laplacian_var"],
        crop_stats["edge_density"],
        third_x,
        third_y,
        full_stats["luma_mean"],
        full_stats["luma_std"],
        full_stats["saturation_mean"],
        full_stats["colorfulness"],
        crop_stats["luma_mean"] - full_stats["luma_mean"],
        crop_stats["luma_std"] - full_stats["luma_std"],
        crop_stats["saturation_mean"] - full_stats["saturation_mean"],
        crop_stats["colorfulness"] - full_stats["colorfulness"],
    ]
    return [float(v) if math.isfinite(float(v)) else 0.0 for v in values]


def extract_handcrafted_matrix(rows: Sequence[dict[str, Any]], *, split_name: str, image_root: Path, progress_every: int = 5000) -> np.ndarray:
    out = np.zeros((len(rows), len(HANDCRAFTED_FEATURE_NAMES)), dtype=np.float32)
    grouped: dict[Path, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[resolve_gaic_image_path(row, split_name=split_name, image_root=image_root)].append(idx)
    done = 0
    for image_path, indices in grouped.items():
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            for idx in indices:
                out[idx, :] = np.asarray(handcrafted_visual_features(rows[idx], image), dtype=np.float32)
                done += 1
                if progress_every > 0 and done % int(progress_every) == 0:
                    print(json.dumps({"progress": "handcrafted", "split": split_name, "rows_done": done, "rows_total": len(rows)}), flush=True)
    return out


def row_cache_key(row: dict[str, Any]) -> str:
    box = row.get("bbox_norm_xyxy")
    if isinstance(box, (list, tuple)):
        box_text = ",".join(f"{safe_float(v, 0.0):.6f}" for v in box[:4])
    else:
        box_text = ""
    return "|".join(
        [
            str(row.get("image_id", "")),
            str(row.get("candidate_id", "")),
            str(row.get("gt_annotation_id", "")),
            box_text,
        ]
    )


def load_feature_cache(cache_dir: Path | None, *, backend: str, cache_key: str, split_name: str, rows: Sequence[dict[str, Any]]) -> np.ndarray | None:
    if cache_dir is None:
        return None
    prefix = Path(cache_dir) / f"{cache_key}_{split_name}_{backend}"
    feature_path = prefix.with_suffix(".npy")
    keys_path = prefix.with_suffix(".keys.jsonl")
    if not feature_path.exists() or not keys_path.exists():
        return None
    features = np.load(feature_path)
    cached_keys = [line.strip() for line in keys_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    requested_keys = [row_cache_key(row) for row in rows]
    if len(cached_keys) != int(features.shape[0]):
        return None
    if cached_keys == requested_keys:
        print(json.dumps({"status": "feature_cache_hit", "backend": backend, "split": split_name, "mode": "same_order", "path": str(feature_path)}), flush=True)
        return features.astype(np.float32, copy=False)
    index = {key: idx for idx, key in enumerate(cached_keys)}
    if len(index) == len(cached_keys) and all(key in index for key in requested_keys):
        order = np.asarray([index[key] for key in requested_keys], dtype=np.int64)
        print(json.dumps({"status": "feature_cache_hit", "backend": backend, "split": split_name, "mode": "reordered", "path": str(feature_path)}), flush=True)
        return features[order].astype(np.float32, copy=False)
    return None


def save_feature_cache(cache_dir: Path | None, *, backend: str, cache_key: str, split_name: str, rows: Sequence[dict[str, Any]], features: np.ndarray) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = cache_dir / f"{cache_key}_{split_name}_{backend}"
    np.save(prefix.with_suffix(".npy"), features.astype(np.float32, copy=False))
    with prefix.with_suffix(".keys.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row_cache_key(row) + "\n")
    print(json.dumps({"status": "feature_cache_saved", "backend": backend, "split": split_name, "path": str(prefix.with_suffix('.npy'))}), flush=True)


def load_vgg16_feature_model(weights_path: Path | None, *, device: str):
    import torch
    import torchvision.models as models

    model = models.vgg16(weights=None)
    if weights_path is not None and Path(weights_path).exists():
        state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state)
    feature_model = torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((1, 1))).to(device)
    feature_model.eval()
    return feature_model


def _crop_tensor(image: Image.Image, box: Any, *, input_size: int) -> "Any":
    import torch

    crop = crop_image(image, box).resize((int(input_size), int(input_size)), Image.BILINEAR)
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _image_tensor(image: Image.Image, *, input_size: int) -> "Any":
    import torch

    img = image.convert("RGB").resize((int(input_size), int(input_size)), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def extract_vgg16_embeddings(
    rows: Sequence[dict[str, Any]],
    *,
    split_name: str,
    image_root: Path,
    weights_path: Path | None,
    device: str,
    input_size: int,
    batch_size: int,
    progress_every: int = 10000,
) -> np.ndarray:
    import torch

    model = load_vgg16_feature_model(weights_path, device=device)
    out = np.zeros((len(rows), 512), dtype=np.float32)
    batch_tensors: list[Any] = []
    batch_indices: list[int] = []
    done = 0

    def flush() -> None:
        nonlocal batch_tensors, batch_indices, done
        if not batch_tensors:
            return
        tensor = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                emb = model(tensor).flatten(1)
        out[np.asarray(batch_indices, dtype=np.int64), :] = emb.detach().float().cpu().numpy()
        done += len(batch_indices)
        if progress_every > 0 and done % int(progress_every) < len(batch_indices):
            print(json.dumps({"progress": "vgg16", "split": split_name, "rows_done": done, "rows_total": len(rows)}), flush=True)
        batch_tensors = []
        batch_indices = []

    grouped: dict[Path, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[resolve_gaic_image_path(row, split_name=split_name, image_root=image_root)].append(idx)
    for image_path, indices in grouped.items():
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            for idx in indices:
                batch_tensors.append(_crop_tensor(image, rows[idx].get("bbox_norm_xyxy"), input_size=input_size))
                batch_indices.append(idx)
                if len(batch_tensors) >= int(batch_size):
                    flush()
    flush()
    return out


def load_timm_feature_model(model_name: str, weights_path: Path | None, *, device: str):
    import torch
    import timm

    model = timm.create_model(str(model_name), pretrained=False, num_classes=0, global_pool="avg")
    if weights_path is not None and Path(weights_path).exists():
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(str(weights_path), device="cpu")
        else:
            state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def extract_timm_embeddings(
    rows: Sequence[dict[str, Any]],
    *,
    split_name: str,
    image_root: Path,
    model_name: str,
    weights_path: Path | None,
    device: str,
    input_size: int,
    batch_size: int,
    progress_every: int = 10000,
) -> np.ndarray:
    import torch

    model = load_timm_feature_model(model_name, weights_path, device=device)
    out_chunks: list[np.ndarray] = []
    out_indices: list[np.ndarray] = []
    batch_tensors: list[Any] = []
    batch_indices: list[int] = []
    done = 0

    def flush() -> None:
        nonlocal batch_tensors, batch_indices, done
        if not batch_tensors:
            return
        tensor = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                emb = model(tensor).flatten(1)
        out_chunks.append(emb.detach().float().cpu().numpy().astype(np.float32))
        out_indices.append(np.asarray(batch_indices, dtype=np.int64))
        done += len(batch_indices)
        if progress_every > 0 and done % int(progress_every) < len(batch_indices):
            print(json.dumps({"progress": "timm", "split": split_name, "rows_done": done, "rows_total": len(rows)}), flush=True)
        batch_tensors = []
        batch_indices = []

    grouped: dict[Path, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[resolve_gaic_image_path(row, split_name=split_name, image_root=image_root)].append(idx)
    for image_path, indices in grouped.items():
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            for idx in indices:
                batch_tensors.append(_crop_tensor(image, rows[idx].get("bbox_norm_xyxy"), input_size=input_size))
                batch_indices.append(idx)
                if len(batch_tensors) >= int(batch_size):
                    flush()
    flush()
    if not out_chunks:
        return np.zeros((len(rows), 0), dtype=np.float32)
    dim = int(out_chunks[0].shape[1])
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for indices, chunk in zip(out_indices, out_chunks):
        out[indices, :] = chunk
    return out


def load_timm_feature_map_model(model_name: str, weights_path: Path | None, *, device: str):
    import torch
    import timm

    model = timm.create_model(str(model_name), pretrained=False, features_only=True, out_indices=(-1,))
    if weights_path is not None and Path(weights_path).exists():
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(str(weights_path), device="cpu")
        else:
            state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def extract_timm_roi_embeddings(
    rows: Sequence[dict[str, Any]],
    *,
    split_name: str,
    image_root: Path,
    model_name: str,
    weights_path: Path | None,
    device: str,
    input_size: int,
    progress_every: int = 100,
) -> np.ndarray:
    import torch

    model = load_timm_feature_map_model(model_name, weights_path, device=device)
    grouped: dict[Path, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[resolve_gaic_image_path(row, split_name=split_name, image_root=image_root)].append(idx)

    out: np.ndarray | None = None
    done_images = 0
    done_rows = 0
    for image_path, indices in grouped.items():
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        tensor = _image_tensor(image, input_size=input_size).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                features = model(tensor)
        feat = features[-1].detach().float().cpu().squeeze(0).numpy()
        channels, feat_h, feat_w = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])
        global_feat = feat.mean(axis=(1, 2))
        if out is None:
            out = np.zeros((len(rows), channels * 2), dtype=np.float32)
        for idx in indices:
            x1, y1, x2, y2 = clamp_box(rows[idx].get("bbox_norm_xyxy"))
            ix1 = max(0, min(feat_w - 1, int(math.floor(x1 * feat_w))))
            iy1 = max(0, min(feat_h - 1, int(math.floor(y1 * feat_h))))
            ix2 = max(ix1 + 1, min(feat_w, int(math.ceil(x2 * feat_w))))
            iy2 = max(iy1 + 1, min(feat_h, int(math.ceil(y2 * feat_h))))
            pooled = feat[:, iy1:iy2, ix1:ix2].mean(axis=(1, 2))
            out[idx, :] = np.concatenate([pooled, pooled - global_feat], axis=0).astype(np.float32)
        done_images += 1
        done_rows += len(indices)
        if progress_every > 0 and done_images % int(progress_every) == 0:
            print(
                json.dumps(
                    {
                        "progress": "timm_roi",
                        "split": split_name,
                        "images_done": done_images,
                        "images_total": len(grouped),
                        "rows_done": done_rows,
                        "rows_total": len(rows),
                    }
                ),
                flush=True,
            )
    if out is None:
        return np.zeros((len(rows), 0), dtype=np.float32)
    return out


def build_compact_matrix(rows: Sequence[dict[str, Any]], spec: dict[str, Any]) -> np.ndarray:
    return np.asarray([row_to_features(row, spec) for row in rows], dtype=np.float32)


def fit_visual_pca(train_emb: np.ndarray, split_emb: dict[str, np.ndarray], *, components: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if PCA is None:
        raise RuntimeError("sklearn PCA is not available")
    n_components = min(int(components), int(train_emb.shape[1]), max(1, int(train_emb.shape[0]) - 1))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    train_transformed = pca.fit_transform(train_emb)
    out = {"train": train_transformed.astype(np.float32)}
    for split_name, emb in split_emb.items():
        if split_name == "train":
            continue
        out[split_name] = pca.transform(emb).astype(np.float32)
    meta = {
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return out, meta


def _rank_pct(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [0.5 for _ in values]
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: (item[1], item[0]))
    ranks = [0.0 for _ in values]
    denom = float(n - 1)
    start = 0
    while start < n:
        end = start + 1
        while end < n and indexed[end][1] == indexed[start][1]:
            end += 1
        avg = 0.5 * float(start + end - 1) / denom
        for pos in range(start, end):
            ranks[indexed[pos][0]] = avg
        start = end
    return ranks


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray | None,
    *,
    spec_name: str,
    seed: int,
    hgb_max_iter: int,
    hgb_learning_rate: float,
    extra_trees_n_estimators: int,
    extra_trees_max_depth: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    n_jobs: int,
) -> Any:
    name = str(spec_name)
    if name.startswith("ridge"):
        if Ridge is None or make_pipeline is None or StandardScaler is None:
            raise RuntimeError("sklearn Ridge pipeline is not available")
        alpha = 10.0 if "strong" in name else 1.0
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=int(seed)))
        model.fit(x_train, y_train, ridge__sample_weight=weights)
        return model
    if name.startswith("hgb"):
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("sklearn HistGradientBoostingRegressor is not available")
        loss = "absolute_error" if "abs" in name else "squared_error"
        model = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=int(hgb_max_iter),
            learning_rate=float(hgb_learning_rate),
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=int(seed),
        )
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    if name.startswith("extra_trees"):
        if ExtraTreesRegressor is None:
            raise RuntimeError("sklearn ExtraTreesRegressor is not available")
        model = ExtraTreesRegressor(
            n_estimators=int(extra_trees_n_estimators),
            max_depth=int(extra_trees_max_depth),
            min_samples_leaf=2,
            n_jobs=int(n_jobs),
            random_state=int(seed),
        )
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    if name.startswith("rf"):
        if RandomForestRegressor is None:
            raise RuntimeError("sklearn RandomForestRegressor is not available")
        model = RandomForestRegressor(
            n_estimators=int(rf_n_estimators),
            max_depth=int(rf_max_depth),
            min_samples_leaf=2,
            n_jobs=int(n_jobs),
            random_state=int(seed),
        )
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    raise ValueError(f"unsupported model spec: {spec_name}")


def target_for_spec(rows: Sequence[dict[str, Any]], spec_name: str) -> np.ndarray:
    name = str(spec_name)
    if "rankpct" in name:
        return target_values(rows, target="rankpct").astype(np.float32)
    if "zscore" in name:
        return target_values(rows, target="zscore").astype(np.float32)
    return target_values(rows, target="mos").astype(np.float32)


def evaluate_prediction(rows: Sequence[dict[str, Any]], scores: Sequence[float], *, method: str, family: str, target: str, train_rows_used: int) -> dict[str, Any]:
    result = evaluate_rows(rows, scores, method=method)
    result["family"] = family
    result["target"] = target
    result["train_rows_used"] = int(train_rows_used)
    result["objective"] = round(objective(result.get("metrics", {})), 9)
    return result


def blend_scores(left: Sequence[float], right: Sequence[float], weight: float) -> list[float]:
    w = float(weight)
    return [w * float(a) + (1.0 - w) * float(b) for a, b in zip(left, right)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prediction_rows(rows: Sequence[dict[str, Any]], scores: Sequence[float], *, method: str) -> Iterable[dict[str, Any]]:
    for row, score in zip(rows, scores):
        yield {
            "image_id": str(row.get("image_id", "")),
            "candidate_id": str(row.get("candidate_id", "")),
            "gt_annotation_id": row.get("gt_annotation_id"),
            "protocol": row.get("protocol"),
            "mos": safe_float(row.get("mos"), 0.0),
            "score": float(score),
            "method": method,
            "bbox_norm_xyxy": row.get("bbox_norm_xyxy"),
        }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def metric_cells(result: dict[str, Any]) -> list[str]:
    m = result.get("metrics", {})
    return [
        str(result.get("method", "")),
        str(result.get("family", "")),
        str(result.get("target", "")),
        f"{safe_float(result.get('objective'), 0.0):.6f}",
        f"{safe_float(m.get('pcc'), 0.0):.6f}",
        f"{safe_float(m.get('srcc'), 0.0):.6f}",
        f"{safe_float(m.get('acc1_of_top10'), 0.0):.6f}",
        f"{safe_float(m.get('top1_mos'), 0.0):.6f}",
        f"{safe_float(m.get('top1_mos_regret'), 0.0):.6f}",
    ]


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# GAIC Visual Crop Ranker Report",
        "",
        f"- protocol: `{summary.get('protocol')}`",
        f"- selected_method_by_val: `{summary.get('selected_method_by_val')}`",
        f"- visual_backends: `{','.join(summary.get('visual_backends', []))}`",
        f"- row_counts: `{summary.get('row_counts')}`",
        "",
        "## Model Comparison",
        "",
    ]
    rows = []
    for candidate in summary.get("model_candidates", []):
        val = candidate.get("evaluations", {}).get("val", {})
        test = candidate.get("evaluations", {}).get("test", {})
        val_m = val.get("metrics", {})
        test_m = test.get("metrics", {})
        rows.append(
            [
                str(candidate.get("method", "")),
                str(candidate.get("family", "")),
                str(candidate.get("target", "")),
                f"{safe_float(val.get('objective'), 0.0):.6f}",
                f"{safe_float(val_m.get('pcc'), 0.0):.6f}",
                f"{safe_float(val_m.get('srcc'), 0.0):.6f}",
                f"{safe_float(val_m.get('acc1_of_top10'), 0.0):.6f}",
                f"{safe_float(test_m.get('pcc'), 0.0):.6f}",
                f"{safe_float(test_m.get('srcc'), 0.0):.6f}",
                f"{safe_float(test_m.get('acc1_of_top10'), 0.0):.6f}",
                f"{safe_float(test_m.get('top1_mos'), 0.0):.6f}",
                f"{safe_float(test_m.get('top1_mos_regret'), 0.0):.6f}",
            ]
        )
    lines.append(
        markdown_table(
            ["method", "family", "target", "val obj", "val PCC", "val SRCC", "val Acc1/10", "test PCC", "test SRCC", "test Acc1/10", "test top1 MOS", "test regret"],
            rows,
        )
    )
    lines.extend(["", "## Selected Split Metrics", ""])
    selected_rows = []
    for split_name in ("train", "val", "test"):
        selected = summary.get("selected_evaluations", {}).get(split_name, {})
        if selected:
            selected_rows.append([split_name] + metric_cells(selected))
    lines.append(markdown_table(["split", "method", "family", "target", "objective", "PCC", "SRCC", "Acc1/10", "top1 MOS", "regret"], selected_rows))
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAIC MOS rankers with SSTK compact features plus crop RGB visual descriptors.")
    parser.add_argument("--train_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--val_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--test_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--protocol", choices=["Gc", "Ge"], default="Gc")
    parser.add_argument("--image_root", type=Path, default=Path("data/Publics/GAIC_v2/images"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--visual_backends", default="handcrafted", help="Comma-separated: handcrafted,vgg16,timm,timm_roi.")
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--feature_cache_key", default="default")
    parser.add_argument("--score_fields", default=",".join(DEFAULT_SCORE_FIELDS))
    parser.add_argument("--extra_feature_fields", default=",".join(DEFAULT_EXTRA_FEATURE_FIELDS))
    parser.add_argument("--audit_fields", default=",".join(DEFAULT_AUDIT_FIELDS))
    parser.add_argument("--model_specs", default="ridge_rankpct,hgb_rankpct,hgb_mos,extra_trees_rankpct")
    parser.add_argument("--enable_ensembles", type=int, default=1)
    parser.add_argument("--ensemble_source_top_k", type=int, default=6)
    parser.add_argument("--ensemble_keep_top_k", type=int, default=8)
    parser.add_argument("--ensemble_weight_steps", type=int, default=9)
    parser.add_argument("--vgg16_weights", type=Path, default=Path("/home/user/.cache/torch/hub/checkpoints/vgg16-397923af.pth"))
    parser.add_argument("--vgg_input_size", type=int, default=224)
    parser.add_argument("--vgg_batch_size", type=int, default=256)
    parser.add_argument("--vgg_pca_components", type=int, default=64)
    parser.add_argument("--timm_model_name", default="mobilenetv4_conv_small.e3600_r256_in1k")
    parser.add_argument("--timm_weights", type=Path, default=Path("weights/hf/timm/mobilenetv4_conv_small.e3600_r256_in1k/model.safetensors"))
    parser.add_argument("--timm_input_size", type=int, default=224)
    parser.add_argument("--timm_batch_size", type=int, default=384)
    parser.add_argument("--timm_pca_components", type=int, default=64)
    parser.add_argument("--timm_roi_input_size", type=int, default=384)
    parser.add_argument("--timm_roi_pca_components", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hgb_max_iter", type=int, default=220)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.045)
    parser.add_argument("--extra_trees_n_estimators", type=int, default=220)
    parser.add_argument("--extra_trees_max_depth", type=int, default=20)
    parser.add_argument("--rf_n_estimators", type=int, default=180)
    parser.add_argument("--rf_max_depth", type=int, default=22)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--max_test_rows", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=20260417)
    parser.add_argument("--write_predictions", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)
    visual_backends = parse_csv(args.visual_backends)
    score_fields = parse_csv(args.score_fields)
    extra_feature_fields = parse_csv(args.extra_feature_fields)
    audit_fields = parse_csv(args.audit_fields)
    feature_fields = list(dict.fromkeys(score_fields + extra_feature_fields))
    model_specs = parse_csv(args.model_specs)

    rows_by_split = {
        "train": read_rows(args.train_candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_train_rows)),
        "val": read_rows(args.val_candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_val_rows)),
        "test": read_rows(args.test_candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_test_rows)),
    }
    if not rows_by_split["train"] or not rows_by_split["val"]:
        raise RuntimeError("train and val rows are required")

    print(json.dumps({"status": "loaded_rows", "protocol": args.protocol, "row_counts": {k: len(v) for k, v in rows_by_split.items()}}), flush=True)
    compact_spec = build_feature_spec(rows_by_split["train"], feature_fields)
    matrices: dict[str, list[np.ndarray]] = {}
    feature_meta: dict[str, Any] = {"compact_feature_count": len(compact_spec.get("feature_names", []))}
    for split_name, rows in rows_by_split.items():
        matrices[split_name] = [build_compact_matrix(rows, compact_spec)]

    if "handcrafted" in visual_backends:
        for split_name, rows in rows_by_split.items():
            cached = load_feature_cache(
                args.feature_cache_dir,
                backend="handcrafted",
                cache_key=str(args.feature_cache_key),
                split_name=split_name,
                rows=rows,
            )
            if cached is None:
                cached = extract_handcrafted_matrix(rows, split_name=split_name, image_root=image_root, progress_every=int(args.progress_every))
                save_feature_cache(
                    args.feature_cache_dir,
                    backend="handcrafted",
                    cache_key=str(args.feature_cache_key),
                    split_name=split_name,
                    rows=rows,
                    features=cached,
                )
            matrices[split_name].append(cached)
        feature_meta["handcrafted_feature_names"] = list(HANDCRAFTED_FEATURE_NAMES)

    if "vgg16" in visual_backends:
        raw_emb: dict[str, np.ndarray] = {}
        for split_name, rows in rows_by_split.items():
            cached = load_feature_cache(
                args.feature_cache_dir,
                backend="vgg16_raw",
                cache_key=str(args.feature_cache_key),
                split_name=split_name,
                rows=rows,
            )
            if cached is None:
                cached = extract_vgg16_embeddings(
                    rows,
                    split_name=split_name,
                    image_root=image_root,
                    weights_path=args.vgg16_weights,
                    device=str(args.device),
                    input_size=int(args.vgg_input_size),
                    batch_size=int(args.vgg_batch_size),
                    progress_every=int(args.progress_every),
                )
                save_feature_cache(
                    args.feature_cache_dir,
                    backend="vgg16_raw",
                    cache_key=str(args.feature_cache_key),
                    split_name=split_name,
                    rows=rows,
                    features=cached,
                )
            raw_emb[split_name] = cached
        pca_emb, pca_meta = fit_visual_pca(raw_emb["train"], raw_emb, components=int(args.vgg_pca_components), seed=int(args.random_seed))
        feature_meta["vgg16"] = {"raw_dim": 512, "pca": pca_meta, "weights": str(args.vgg16_weights)}
        for split_name in rows_by_split:
            matrices[split_name].append(pca_emb[split_name])

    if "timm" in visual_backends:
        raw_emb = {}
        backend_name = "timm_" + str(args.timm_model_name).replace("/", "_").replace(".", "_")
        for split_name, rows in rows_by_split.items():
            cached = load_feature_cache(
                args.feature_cache_dir,
                backend=backend_name + "_raw",
                cache_key=str(args.feature_cache_key),
                split_name=split_name,
                rows=rows,
            )
            if cached is None:
                cached = extract_timm_embeddings(
                    rows,
                    split_name=split_name,
                    image_root=image_root,
                    model_name=str(args.timm_model_name),
                    weights_path=args.timm_weights,
                    device=str(args.device),
                    input_size=int(args.timm_input_size),
                    batch_size=int(args.timm_batch_size),
                    progress_every=int(args.progress_every),
                )
                save_feature_cache(
                    args.feature_cache_dir,
                    backend=backend_name + "_raw",
                    cache_key=str(args.feature_cache_key),
                    split_name=split_name,
                    rows=rows,
                    features=cached,
                )
            raw_emb[split_name] = cached
        pca_emb, pca_meta = fit_visual_pca(raw_emb["train"], raw_emb, components=int(args.timm_pca_components), seed=int(args.random_seed))
        feature_meta["timm"] = {
            "model_name": str(args.timm_model_name),
            "raw_dim": int(raw_emb["train"].shape[1]),
            "pca": pca_meta,
            "weights": str(args.timm_weights),
        }
        for split_name in rows_by_split:
            matrices[split_name].append(pca_emb[split_name])

    if "timm_roi" in visual_backends:
        raw_emb = {}
        backend_name = "timm_roi_" + str(args.timm_model_name).replace("/", "_").replace(".", "_")
        for split_name, rows in rows_by_split.items():
            cached = load_feature_cache(
                args.feature_cache_dir,
                backend=backend_name + "_raw",
                cache_key=str(args.feature_cache_key),
                split_name=split_name,
                rows=rows,
            )
            if cached is None:
                cached = extract_timm_roi_embeddings(
                    rows,
                    split_name=split_name,
                    image_root=image_root,
                    model_name=str(args.timm_model_name),
                    weights_path=args.timm_weights,
                    device=str(args.device),
                    input_size=int(args.timm_roi_input_size),
                    progress_every=max(1, int(args.progress_every)),
                )
                save_feature_cache(
                    args.feature_cache_dir,
                    backend=backend_name + "_raw",
                    cache_key=str(args.feature_cache_key),
                    split_name=split_name,
                    rows=rows,
                    features=cached,
                )
            raw_emb[split_name] = cached
        pca_emb, pca_meta = fit_visual_pca(raw_emb["train"], raw_emb, components=int(args.timm_roi_pca_components), seed=int(args.random_seed))
        feature_meta["timm_roi"] = {
            "model_name": str(args.timm_model_name),
            "raw_dim": int(raw_emb["train"].shape[1]),
            "pca": pca_meta,
            "weights": str(args.timm_weights),
            "input_size": int(args.timm_roi_input_size),
        }
        for split_name in rows_by_split:
            matrices[split_name].append(pca_emb[split_name])

    x_by_split = {split: np.concatenate(parts, axis=1).astype(np.float32) for split, parts in matrices.items()}
    feature_meta["total_feature_count"] = int(x_by_split["train"].shape[1])
    print(json.dumps({"status": "features_ready", "feature_meta": feature_meta}), flush=True)

    candidate_pool: list[dict[str, Any]] = []
    evaluations: dict[str, list[dict[str, Any]]] = {split: [] for split in rows_by_split}
    for field in audit_fields:
        scores_by_split: dict[str, list[float]] = {}
        evals: dict[str, dict[str, Any]] = {}
        for split_name, rows in rows_by_split.items():
            scores = [safe_float(row.get(field), 0.0) for row in rows]
            scores_by_split[split_name] = scores
            result = evaluate_prediction(rows, scores, method=field, family="score_field", target="none", train_rows_used=0)
            evals[split_name] = result
            evaluations[split_name].append(result)
        candidate_pool.append(
            {
                "method": field,
                "family": "score_field",
                "target": "none",
                "evaluations": evals,
                "_scores_by_split": scores_by_split,
            }
        )

    train_weights = sample_weights(rows_by_split["train"], mode="image_equal")
    for spec_name in model_specs:
        y_train = target_for_spec(rows_by_split["train"], spec_name)
        model = train_model(
            x_by_split["train"],
            y_train,
            train_weights,
            spec_name=spec_name,
            seed=int(args.random_seed),
            hgb_max_iter=int(args.hgb_max_iter),
            hgb_learning_rate=float(args.hgb_learning_rate),
            extra_trees_n_estimators=int(args.extra_trees_n_estimators),
            extra_trees_max_depth=int(args.extra_trees_max_depth),
            rf_n_estimators=int(args.rf_n_estimators),
            rf_max_depth=int(args.rf_max_depth),
            n_jobs=int(args.n_jobs),
        )
        target = "rankpct" if "rankpct" in spec_name else ("zscore" if "zscore" in spec_name else "mos")
        method = f"visual_{spec_name}"
        evals = {}
        scores_by_split = {}
        for split_name, rows in rows_by_split.items():
            scores = [float(v) for v in model.predict(x_by_split[split_name])]
            scores_by_split[split_name] = scores
            result = evaluate_prediction(rows, scores, method=method, family="visual_ranker", target=target, train_rows_used=len(rows_by_split["train"]))
            evals[split_name] = result
            evaluations[split_name].append(result)
        candidate_pool.append(
            {
                "method": method,
                "family": "visual_ranker",
                "target": target,
                "evaluations": evals,
                "_scores_by_split": scores_by_split,
            }
        )
        print(json.dumps({"status": "model_done", "method": method, "val_objective": evals["val"]["objective"]}), flush=True)

    if int(args.enable_ensembles):
        source_pool = sorted(candidate_pool, key=lambda item: objective(item["evaluations"]["val"].get("metrics", {})), reverse=True)
        source_pool = source_pool[: max(2, int(args.ensemble_source_top_k))]
        pair_candidates: list[dict[str, Any]] = []
        steps = max(1, int(args.ensemble_weight_steps))
        for i in range(len(source_pool)):
            for j in range(i + 1, len(source_pool)):
                left = source_pool[i]
                right = source_pool[j]
                best: dict[str, Any] | None = None
                for step in range(1, steps + 1):
                    weight = float(step) / float(steps + 1)
                    val_scores = blend_scores(left["_scores_by_split"]["val"], right["_scores_by_split"]["val"], weight)
                    method = f"visual_ens_{i}_{j}_w{int(round(weight * 100)):02d}"
                    val_result = evaluate_prediction(rows_by_split["val"], val_scores, method=method, family="visual_linear_blend", target=f"{left['target']}+{right['target']}", train_rows_used=len(rows_by_split["train"]))
                    value = objective(val_result.get("metrics", {}))
                    if best is None or value > best["selection_objective"]:
                        best = {
                            "method": method,
                            "left": left,
                            "right": right,
                            "left_method": left["method"],
                            "right_method": right["method"],
                            "left_weight": weight,
                            "right_weight": 1.0 - weight,
                            "selection_objective": float(value),
                        }
                if best is not None:
                    pair_candidates.append(best)
        pair_candidates.sort(key=lambda item: item["selection_objective"], reverse=True)
        for item in pair_candidates[: max(0, int(args.ensemble_keep_top_k))]:
            evals = {}
            scores_by_split = {}
            left = item["left"]
            right = item["right"]
            weight = float(item["left_weight"])
            for split_name, rows in rows_by_split.items():
                scores = blend_scores(left["_scores_by_split"][split_name], right["_scores_by_split"][split_name], weight)
                scores_by_split[split_name] = scores
                result = evaluate_prediction(rows, scores, method=item["method"], family="visual_linear_blend", target=f"{left['target']}+{right['target']}", train_rows_used=len(rows_by_split["train"]))
                evals[split_name] = result
                evaluations[split_name].append(result)
            candidate_pool.append(
                {
                    "method": item["method"],
                    "family": "visual_linear_blend",
                    "target": f"{left['target']}+{right['target']}",
                    "left_method": item["left_method"],
                    "right_method": item["right_method"],
                    "left_weight": item["left_weight"],
                    "right_weight": item["right_weight"],
                    "evaluations": evals,
                    "_scores_by_split": scores_by_split,
                }
            )

    selected = max(candidate_pool, key=lambda item: objective(item["evaluations"]["val"].get("metrics", {})))
    selected_method = str(selected["method"])
    if int(args.write_predictions):
        for split_name, rows in rows_by_split.items():
            write_jsonl(output_dir / f"predictions_{split_name}.jsonl", prediction_rows(rows, selected["_scores_by_split"][split_name], method=selected_method))

    model_candidates = []
    for candidate in candidate_pool:
        clean = {key: value for key, value in candidate.items() if not key.startswith("_")}
        model_candidates.append(clean)
    model_candidates.sort(key=lambda item: objective(item["evaluations"]["val"].get("metrics", {})), reverse=True)
    summary = {
        "protocol": str(args.protocol),
        "inputs": {
            "train_candidate_eval_jsonl": str(args.train_candidate_eval_jsonl),
            "val_candidate_eval_jsonl": str(args.val_candidate_eval_jsonl),
            "test_candidate_eval_jsonl": str(args.test_candidate_eval_jsonl),
            "image_root": str(args.image_root),
        },
        "row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "visual_backends": visual_backends,
        "model_specs": model_specs,
        "feature_meta": feature_meta,
        "selected_method_by_val": selected_method,
        "selected_evaluations": selected["evaluations"],
        "model_candidates": model_candidates,
        "prediction_counts": {split: len(rows) for split, rows in rows_by_split.items()} if int(args.write_predictions) else {},
    }
    write_json(output_dir / "visual_ranker_summary.json", summary)
    (output_dir / "VISUAL_RANKER_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "protocol": str(args.protocol),
                "selected_method_by_val": selected_method,
                "row_counts": summary["row_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
