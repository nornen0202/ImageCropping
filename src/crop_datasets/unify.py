"""
Unified Annotation Converter for Image Cropping Datasets
=========================================================
Converts CPC, XPView, FLMS, GAIC, GAIC_v2, and Unsplash Lite datasets
into a single JSONL file with a normalised schema suitable for cropping
model training.

Output schema (one JSON object per line):
{
  "image_id":     "<dataset>_<stem>",
  "dataset":      "cpc",
  "split":        "train",
  "image_path":   "relative/path/to/image.jpg",
  "image_width":  W,
  "image_height": H,
  "crops": [
    {"crop_x1": .., "crop_y1": .., "crop_x2": .., "crop_y2": ..,
     "score": 0.0-1.0, "source": "annotator_mean|expert|mos|gt|full_image"}
  ]
}

Usage:
  python -m crop_datasets.unify --data_root ./data --output ./data/unified_crops.jsonl
  python -m crop_datasets.unify --data_root ./data --output ./data/unified_crops.jsonl --limit 10
  python -m crop_datasets.unify --data_root ./data --output ./data/unified_crops.jsonl --datasets cpc gaic_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Helper: get image dimensions
# --------------------------------------------------------------------------
_dim_cache: dict[str, tuple[int, int]] = {}


def _get_dims(img_path: Path) -> tuple[int, int]:
    """Return (width, height) for an image file. Uses PIL if available, else (0,0)."""
    key = str(img_path)
    if key in _dim_cache:
        return _dim_cache[key]

    if HAS_PIL and img_path.exists():
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            _dim_cache[key] = (w, h)
            return w, h
        except Exception:
            pass
    _dim_cache[key] = (0, 0)
    return 0, 0


# --------------------------------------------------------------------------
# Per-dataset parsers
# --------------------------------------------------------------------------

def _parse_cpc(data_root: Path, limit: int = 0) -> Iterator[dict[str, Any]]:
    """
    CPC: CollectedAnnotationsRaw/*.txt (double-JSON encoded)
         {bboxes: [[x1,y1,x2,y2], ...], scores: [[s,...], ...]}
    """
    base = data_root / "cpc" / "extracted" / "CPCDataset"
    ann_dir = base / "CollectedAnnotationsRaw"
    img_dir = base / "images"

    if not ann_dir.is_dir():
        log.warning("CPC annotations directory not found: %s", ann_dir)
        return

    files = sorted(ann_dir.iterdir())
    if limit > 0:
        files = files[:limit]

    for fpath in files:
        if not fpath.name.endswith(".txt"):
            continue
        # filename pattern: <image_name>.txt  (e.g. ava_obj_0_10711.jpg.txt)
        img_name = fpath.name[:-4]  # remove .txt → ava_obj_0_10711.jpg
        img_path = img_dir / img_name
        stem = Path(img_name).stem

        try:
            raw = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            data = json.loads(json.loads(raw))
        except (json.JSONDecodeError, TypeError) as exc:
            log.debug("CPC: skip %s (%s)", fpath.name, exc)
            continue

        bboxes = data.get("bboxes", [])
        scores = data.get("scores", [])  # [annotator][crop_idx]
        num_crops = len(bboxes)

        crops = []
        for ci in range(num_crops):
            # Mean score across annotators
            annotator_scores = [s[ci] for s in scores if ci < len(s)]
            if not annotator_scores:
                continue
            mean_s = sum(annotator_scores) / len(annotator_scores)
            normalised = mean_s / 5.0  # 0-5 → 0-1

            x1, y1, x2, y2 = bboxes[ci]
            crops.append({
                "crop_x1": int(x1), "crop_y1": int(y1),
                "crop_x2": int(x2), "crop_y2": int(y2),
                "score": round(normalised, 4),
                "source": "annotator_mean",
            })

        if not crops:
            continue

        w, h = _get_dims(img_path)
        yield {
            "image_id": f"cpc_{stem}",
            "dataset": "cpc",
            "split": "train",
            "image_path": str(img_path.relative_to(data_root)),
            "image_width": w,
            "image_height": h,
            "crops": crops,
        }


def _parse_xpview(data_root: Path, limit: int = 0) -> Iterator[dict[str, Any]]:
    """
    XPView: annotations_strict/*.txt — CSV h_min,w_min,h_max,w_max,score
    """
    base = data_root / "xpview" / "extracted" / "XPDataset"
    ann_dir = base / "annotations_strict"
    img_dir = base / "images"

    if not ann_dir.is_dir():
        log.warning("XPView annotations_strict not found: %s", ann_dir)
        return

    files = sorted(ann_dir.iterdir())
    if limit > 0:
        files = files[:limit]

    for fpath in files:
        if not fpath.name.endswith(".txt"):
            continue
        img_name = fpath.name[:-4]  # remove .txt → image_name.jpg
        img_path = img_dir / img_name
        stem = Path(img_name).stem

        try:
            lines = fpath.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except Exception:
            continue

        crops = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            h_min, w_min, h_max, w_max = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            raw_score = float(parts[4])
            # Score is vote count (0–3 experts); normalise to 0–1
            score = raw_score / 3.0
            crops.append({
                "crop_x1": w_min, "crop_y1": h_min,
                "crop_x2": w_max, "crop_y2": h_max,
                "score": round(score, 4),
                "source": "expert",
            })

        if not crops:
            continue

        w, h = _get_dims(img_path)
        yield {
            "image_id": f"xpview_{stem}",
            "dataset": "xpview",
            "split": "train",
            "image_path": str(img_path.relative_to(data_root)),
            "image_width": w,
            "image_height": h,
            "crops": crops,
        }


def _parse_flms(data_root: Path, limit: int = 0) -> Iterator[dict[str, Any]]:
    """
    FLMS: 500_image_dataset.mat — MATLAB struct (img_gt with img + bbox fields)
    bbox = [h_min, w_min, h_max, w_max] (MATLAB coords)
    """
    base = data_root / "flms" / "extracted"
    mat_path = base / "500_image_dataset.mat"
    img_dir = base / "image"

    if not mat_path.exists():
        log.warning("FLMS .mat file not found: %s", mat_path)
        return

    try:
        import scipy.io as sio  # type: ignore
        mat = sio.loadmat(str(mat_path), squeeze_me=True)
    except ImportError:
        log.warning("scipy required for FLMS .mat parsing — pip install scipy")
        return
    except Exception as exc:
        log.warning("Failed to load FLMS .mat: %s", exc)
        return

    img_gt = mat.get("img_gt")
    if img_gt is None:
        log.warning("FLMS: 'img_gt' key not found in .mat file")
        return

    count = 0
    for entry in img_gt:
        if limit > 0 and count >= limit:
            break
        try:
            img_name = str(entry["img"])
            bboxes = entry["bbox"]
        except (KeyError, TypeError, IndexError):
            continue

        img_path = img_dir / img_name
        stem = Path(img_name).stem

        # bboxes can be 1D (single crop) or 2D (multiple)
        import numpy as np  # type: ignore
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)

        crops = []
        for row in bboxes:
            if len(row) < 4:
                continue
            h_min, w_min, h_max, w_max = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            # skip invalid (negative coords from turker errors)
            if h_min < 0 or w_min < 0 or h_max < 0 or w_max < 0:
                continue
            crops.append({
                "crop_x1": w_min, "crop_y1": h_min,
                "crop_x2": w_max, "crop_y2": h_max,
                "score": 1.0,
                "source": "gt",
            })

        if not crops:
            continue

        w, h = _get_dims(img_path)
        yield {
            "image_id": f"flms_{stem}",
            "dataset": "flms",
            "split": "train",
            "image_path": str(img_path.relative_to(data_root)),
            "image_width": w,
            "image_height": h,
            "crops": crops,
        }
        count += 1


def _parse_gaic(
    data_root: Path,
    dataset_name: str,  # "gaic" or "gaic_v2"
    limit: int = 0,
) -> Iterator[dict[str, Any]]:
    """
    GAIC / GAIC_v2: annotations/*.txt — space-delimited y1 x1 y2 x2 MOS
    GAIC:    images/{train,test}/  annotations/ (flat)
    GAIC_v2: images/{train,val,test}/  annotations/{train,val,test}/
    """
    if dataset_name == "gaic":
        base = data_root / "GAIC"
    else:
        base = data_root / "GAIC_v2"

    if not base.is_dir():
        log.warning("%s directory not found: %s", dataset_name.upper(), base)
        return

    ann_base = base / "annotations"
    img_base = base / "images"

    # Determine split layout
    is_v2 = dataset_name == "gaic_v2"

    if is_v2:
        # v2: annotations/{train,val,test}/, images/{train,val,test}/
        splits = ["train", "val", "test"]
    else:
        # v1: annotations/ (flat for all), images/{train,test}/
        splits = ["_flat"]  # special marker

    count = 0
    for split_name in splits:
        if is_v2:
            ann_dir = ann_base / split_name
            img_dir = img_base / split_name
        else:
            ann_dir = ann_base
            img_dir = None  # determine per-file

        if not ann_dir.is_dir():
            continue

        files = sorted(ann_dir.iterdir())
        for fpath in files:
            if limit > 0 and count >= limit:
                return
            if not fpath.name.endswith(".txt"):
                continue

            stem = fpath.stem  # e.g. "210333"
            img_name = stem + ".jpg"

            # Resolve image path
            if is_v2:
                img_path = img_dir / img_name
            else:
                # v1: try train/ first, then test/
                img_path = img_base / "train" / img_name
                actual_split = "train"
                if not img_path.exists():
                    img_path = img_base / "test" / img_name
                    actual_split = "test"

            if is_v2:
                actual_split = split_name

            try:
                lines = fpath.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            except Exception:
                continue

            crops = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                y1 = int(parts[0])
                x1 = int(parts[1])
                y2 = int(parts[2])
                x2 = int(parts[3])
                mos = float(parts[4])
                score = (mos - 1.0) / 4.0  # 1-5 → 0-1
                score = max(0.0, min(1.0, score))
                crops.append({
                    "crop_x1": x1, "crop_y1": y1,
                    "crop_x2": x2, "crop_y2": y2,
                    "score": round(score, 4),
                    "source": "mos",
                })

            if not crops:
                continue

            w, h = _get_dims(img_path)
            yield {
                "image_id": f"{dataset_name}_{stem}",
                "dataset": dataset_name,
                "split": actual_split,
                "image_path": str(img_path.relative_to(data_root)),
                "image_width": w,
                "image_height": h,
                "crops": crops,
            }
            count += 1


def _parse_unsplash(data_root: Path, limit: int = 0) -> Iterator[dict[str, Any]]:
    """
    Unsplash Lite: images/<photo_id>.jpg + annotations/<photo_id>.json
    No crop annotations — treat full image as the best crop.
    """
    base = data_root / "unsplash"
    img_dir = base / "images"
    ann_dir = base / "annotations"

    if not img_dir.is_dir():
        log.warning("Unsplash images directory not found: %s", img_dir)
        return

    files = sorted(img_dir.glob("*.jpg"))
    if limit > 0:
        files = files[:limit]

    for img_path in files:
        photo_id = img_path.stem
        w, h = _get_dims(img_path)

        # Unsplash has no crop box — full image = best crop
        crop = {
            "crop_x1": 0, "crop_y1": 0,
            "crop_x2": w if w > 0 else 0,
            "crop_y2": h if h > 0 else 0,
            "score": 1.0,
            "source": "full_image",
        }

        yield {
            "image_id": f"unsplash_{photo_id}",
            "dataset": "unsplash",
            "split": "train",
            "image_path": str(img_path.relative_to(data_root)),
            "image_width": w,
            "image_height": h,
            "crops": [crop],
        }


# --------------------------------------------------------------------------
# Registry of dataset parsers
# --------------------------------------------------------------------------
PARSERS = {
    "cpc":     _parse_cpc,
    "xpview":  _parse_xpview,
    "flms":    _parse_flms,
    "gaic":    lambda root, limit: _parse_gaic(root, "gaic", limit),
    "gaic_v2": lambda root, limit: _parse_gaic(root, "gaic_v2", limit),
    "unsplash": _parse_unsplash,
}

ALL_DATASETS = list(PARSERS.keys())


# --------------------------------------------------------------------------
# Main unify pipeline
# --------------------------------------------------------------------------
def unify(
    data_root: Path,
    output: Path,
    datasets: list[str] | None = None,
    limit: int = 0,
) -> dict[str, dict[str, int]]:
    """
    Run the unified conversion.
    Returns stats: {dataset_name: {"images": N, "crops": M}}.
    """
    if datasets is None:
        datasets = ALL_DATASETS

    output.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}

    with output.open("w", encoding="utf-8") as f:
        for ds_name in datasets:
            parser = PARSERS.get(ds_name)
            if parser is None:
                log.warning("Unknown dataset: %s (available: %s)", ds_name, ALL_DATASETS)
                continue

            log.info("Processing %s …", ds_name)
            img_count = 0
            crop_count = 0

            for record in parser(data_root, limit):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                img_count += 1
                crop_count += len(record["crops"])

            stats[ds_name] = {"images": img_count, "crops": crop_count}
            log.info("  %s: %d images, %d crops", ds_name, img_count, crop_count)

    return stats


# --------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="crop-datasets-unify",
        description="Convert all image-cropping datasets into one unified JSONL file.",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("./data"),
        help="Root directory containing dataset subdirectories (default: ./data)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./data/unified_crops.jsonl"),
        help="Output JSONL path (default: ./data/unified_crops.jsonl)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=None,
        help=f"Datasets to include (default: all). Choices: {ALL_DATASETS}",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N images per dataset for testing (0 = all, default: 0)",
    )
    p.add_argument(
        "--no-dims",
        action="store_true",
        help="Skip reading image dimensions (faster, but width/height will be 0).",
    )

    args = p.parse_args()

    if args.no_dims:
        global HAS_PIL
        HAS_PIL = False

    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    stats = unify(data_root, output, args.datasets, args.limit)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Unified annotation file: {output}")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'Images':>8} {'Crops':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    total_img = 0
    total_crop = 0
    for ds_name, s in stats.items():
        print(f"  {ds_name:<12} {s['images']:>8,} {s['crops']:>10,}")
        total_img += s["images"]
        total_crop += s["crops"]
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    print(f"  {'TOTAL':<12} {total_img:>8,} {total_crop:>10,}")
    print()


if __name__ == "__main__":
    main()
