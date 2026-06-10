"""
Unsplash Lite Dataset Preparation
==================================
Downloads images and produces refined annotation files from the Unsplash Lite
dataset TSV files (photos / keywords / collections / colors).

Usage (CLI):
    python -m crop_datasets.datasets.unsplash_prepare \\
        --dataset_root ./data/unsplash-research-dataset-lite-latest \\
        --output_root  ./data/unsplash \\
        --img_size 640 \\
        --workers 8 \\
        --limit 0          # 0 = all

Output layout:
    <output_root>/
        images/
            <photo_id>.jpg        # downloaded images
        annotations/
            <photo_id>.json       # per-image annotation sidecar
        index.jsonl               # one JSON-line per photo (all merged fields)
        photos_refined.tsv        # filtered/normalised photos table
        keywords_pivot.tsv        # keywords aggregated per photo_id
        colors_pivot.tsv          # dominant colour per photo_id
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(0)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Unsplash dynamic image resize API: ?w=<width>&fit=max&auto=format&q=80
_IMG_URL_TEMPLATE = "{base}?w={size}&fit=max&auto=format&q=80"

# TSV files in the dataset directory (*.csv000 with tab separator)
_TSV_FILES = {
    "photos":      "photos.csv000",
    "keywords":    "keywords.csv000",
    "collections": "collections.csv000",
    "colors":      "colors.csv000",
    "conversions": "conversions.csv000",
}


# ---------------------------------------------------------------------------
# TSV loading
# ---------------------------------------------------------------------------
def _load_tsv(path: Path, usecols: list[str] | None = None) -> "pd.DataFrame":
    """Load a tab-separated file robustly."""
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        low_memory=False,
        on_bad_lines="skip",
    )


def load_tables(dataset_root: Path) -> dict[str, "pd.DataFrame"]:
    """Load all TSV tables present in dataset_root."""
    tables: dict[str, "pd.DataFrame"] = {}
    for key, fname in _TSV_FILES.items():
        fpath = dataset_root / fname
        if fpath.exists():
            log.info("Loading %s …", fpath.name)
            tables[key] = _load_tsv(fpath)
        else:
            log.warning("File not found, skipping: %s", fpath)
    return tables


# ---------------------------------------------------------------------------
# Annotation refinement
# ---------------------------------------------------------------------------
def build_photos_refined(photos: "pd.DataFrame") -> "pd.DataFrame":
    """
    Normalise and filter the photos table:
    - Drop rows without photo_id or photo_image_url
    - Parse numeric fields
    - Add derived fields (aspect_ratio float, landscape/portrait flag)
    """
    df = photos.copy()
    df = df.dropna(subset=["photo_id", "photo_image_url"])
    df["photo_id"] = df["photo_id"].astype(str).str.strip()

    # Coerce numeric columns
    for col in ("photo_width", "photo_height", "stats_views", "stats_downloads",
                "exif_iso", "exif_aperture_value", "exif_focal_length",
                "photo_location_latitude", "photo_location_longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recompute aspect ratio (width / height)
    if "photo_width" in df.columns and "photo_height" in df.columns:
        df["aspect_ratio"] = (df["photo_width"] / df["photo_height"]).round(4)
    elif "photo_aspect_ratio" in df.columns:
        df["aspect_ratio"] = pd.to_numeric(df["photo_aspect_ratio"], errors="coerce")

    # Orientation flag
    if "aspect_ratio" in df.columns:
        df["orientation"] = df["aspect_ratio"].apply(
            lambda r: "landscape" if (isinstance(r, float) and r >= 1.0) else "portrait"
        )

    # photo_featured → bool
    if "photo_featured" in df.columns:
        df["photo_featured"] = df["photo_featured"].map({"t": True, "f": False})

    return df.reset_index(drop=True)


def build_keywords_pivot(keywords: "pd.DataFrame") -> "pd.DataFrame":
    """
    Aggregate keywords per photo_id into a comma-separated list.
    Returns a DataFrame with columns: photo_id, keywords, keyword_count.
    """
    kw = keywords[["photo_id", "keyword"]].dropna()
    kw["photo_id"] = kw["photo_id"].astype(str).str.strip()
    grouped = (
        kw.groupby("photo_id")["keyword"]
        .apply(lambda s: ",".join(sorted(set(s.str.strip().tolist()))))
        .reset_index()
    )
    grouped.columns = ["photo_id", "keywords"]
    grouped["keyword_count"] = grouped["keywords"].str.split(",").str.len()
    return grouped


def build_colors_pivot(colors: "pd.DataFrame") -> "pd.DataFrame":
    """
    Pick dominant colour per photo (highest coverage).
    Returns: photo_id, dominant_hex, dominant_color_name, dominant_coverage.
    """
    c = colors.copy()
    c["photo_id"] = c["photo_id"].astype(str).str.strip()
    c["coverage"] = pd.to_numeric(c["coverage"], errors="coerce")
    dominant = (
        c.sort_values("coverage", ascending=False)
        .groupby("photo_id")
        .first()
        .reset_index()[["photo_id", "hex", "keyword", "coverage"]]
    )
    dominant.columns = ["photo_id", "dominant_hex", "dominant_color_name", "dominant_coverage"]
    return dominant


def merge_annotations(
    photos: "pd.DataFrame",
    keywords_pivot: "pd.DataFrame | None",
    colors_pivot: "pd.DataFrame | None",
) -> "pd.DataFrame":
    """Left-join all annotation tables on photo_id."""
    df = photos.copy()
    if keywords_pivot is not None:
        df = df.merge(keywords_pivot, on="photo_id", how="left")
    if colors_pivot is not None:
        df = df.merge(colors_pivot, on="photo_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------
def _image_url(base_url: str, size: int) -> str:
    """Build a resized Unsplash image URL."""
    # Strip existing query params so we don't double-append
    base = base_url.split("?")[0]
    return _IMG_URL_TEMPLATE.format(base=base, size=size)


def _download_one(
    photo_id: str,
    image_url: str,
    dest: Path,
    size: int,
    retries: int = 3,
    chunk_size: int = 256 * 1024,
) -> tuple[str, bool, str]:
    """
    Download a single image. Returns (photo_id, success, message).
    Supports resume via Range header.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return photo_id, True, "already-exists"

    url = _image_url(image_url, size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume_from = dest.stat().st_size if dest.exists() else 0

    for attempt in range(retries + 1):
        try:
            headers: dict[str, str] = {}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                code = getattr(resp, "status", 200)
                mode = "ab" if code == 206 else "wb"
                with dest.open(mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            return photo_id, True, "ok"
        except HTTPError as exc:
            if exc.code == 416:  # Range Not Satisfiable → already complete
                return photo_id, True, "already-complete"
            if attempt >= retries:
                return photo_id, False, f"HTTP {exc.code}"
        except (URLError, OSError) as exc:
            if attempt >= retries:
                return photo_id, False, str(exc)
        time.sleep(2 ** attempt)

    return photo_id, False, "exhausted-retries"


def download_images(
    df: "pd.DataFrame",
    images_dir: Path,
    size: int = 640,
    workers: int = 8,
) -> "pd.DataFrame":
    """
    Download images for all rows in df (must have photo_id, photo_image_url).
    Adds column 'image_path' (relative to images_dir parent) and 'download_ok'.
    Returns updated DataFrame.
    """
    results: dict[str, tuple[bool, str]] = {}
    rows = list(df[["photo_id", "photo_image_url"]].dropna().itertuples(index=False))

    log.info("Downloading %d images with %d parallel workers (size=%dpx)…", len(rows), workers, size)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                row.photo_id,
                row.photo_image_url,
                images_dir / f"{row.photo_id}.jpg",
                size,
            ): row.photo_id
            for row in rows
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            pid, ok, msg = fut.result()
            results[pid] = (ok, msg)
            if not ok:
                log.warning("[WARN] %s: %s", pid, msg)

    df = df.copy()
    df["download_ok"] = df["photo_id"].map(lambda pid: results.get(pid, (False, ""))[0])
    df["image_path"] = df["photo_id"].apply(lambda pid: f"images/{pid}.jpg")
    return df


# ---------------------------------------------------------------------------
# Annotation sidecar writing
# ---------------------------------------------------------------------------
def write_sidecar_annotations(df: "pd.DataFrame", annotations_dir: Path) -> None:
    """Write one JSON sidecar file per photo in annotations_dir."""
    annotations_dir.mkdir(parents=True, exist_ok=True)
    sidecar_cols = [c for c in df.columns if c != "photo_image_url"]  # skip raw URL (in sidecar it's derived)

    log.info("Writing %d annotation sidecar files…", len(df))
    for row in df[sidecar_cols].itertuples(index=False):
        rec: dict[str, Any] = {k: v for k, v in row._asdict().items()
                               if not (isinstance(v, float) and str(v) == "nan")}
        # Re-attach a clean image URL note
        fpath = annotations_dir / f"{rec['photo_id']}.json"
        fpath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# index.jsonl
# ---------------------------------------------------------------------------
def write_index(df: "pd.DataFrame", out_path: Path) -> None:
    """Write a JSONL index (one line per photo) for fast streaming access."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing index → %s", out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            rec = {k: v for k, v in row._asdict().items()
                   if not (isinstance(v, float) and str(v) == "nan")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def prepare(
    dataset_root: Path,
    output_root: Path,
    img_size: int = 640,
    workers: int = 8,
    limit: int = 0,
    skip_images: bool = False,
) -> None:
    if not HAS_PANDAS:
        sys.exit("[ERROR] pandas is required: pip install pandas")

    output_root.mkdir(parents=True, exist_ok=True)
    images_dir = output_root / "images"
    annotations_dir = output_root / "annotations"

    # --- 1. Load tables ---
    tables = load_tables(dataset_root)
    if "photos" not in tables:
        sys.exit(f"[ERROR] photos.csv000 not found in {dataset_root}")

    # --- 2. Refine photos ---
    photos = build_photos_refined(tables["photos"])
    if limit > 0:
        photos = photos.head(limit)
    log.info("Photos loaded: %d rows", len(photos))

    # --- 3. Build pivot tables ---
    kw_pivot = build_keywords_pivot(tables["keywords"]) if "keywords" in tables else None
    col_pivot = build_colors_pivot(tables["colors"]) if "colors" in tables else None

    # --- 4. Merge annotations ---
    df = merge_annotations(photos, kw_pivot, col_pivot)

    # --- 5. Save refined TSVs ---
    photos_out = output_root / "photos_refined.tsv"
    photos.to_csv(photos_out, sep="\t", index=False)
    log.info("Saved → %s", photos_out)

    if kw_pivot is not None:
        kw_out = output_root / "keywords_pivot.tsv"
        kw_pivot.to_csv(kw_out, sep="\t", index=False)
        log.info("Saved → %s", kw_out)

    if col_pivot is not None:
        col_out = output_root / "colors_pivot.tsv"
        col_pivot.to_csv(col_out, sep="\t", index=False)
        log.info("Saved → %s", col_out)

    # --- 6. Download images ---
    if not skip_images:
        df = download_images(df, images_dir, size=img_size, workers=workers)
        n_ok = int(df["download_ok"].sum())
        log.info("Downloaded: %d / %d", n_ok, len(df))
    else:
        df["download_ok"] = False
        df["image_path"] = df["photo_id"].apply(lambda pid: f"images/{pid}.jpg")
        log.info("--skip-images set: skipping image downloads.")

    # --- 7. Write annotation sidecars ---
    write_sidecar_annotations(df, annotations_dir)

    # --- 8. Write JSONL index ---
    write_index(df, output_root / "index.jsonl")

    log.info("Done. Output root: %s", output_root)
    print(f"\n✅ Unsplash Lite dataset prepared.")
    print(f"   Photos    : {len(df)}")
    print(f"   Images dir: {images_dir}")
    print(f"   Annot dir : {annotations_dir}")
    print(f"   Index     : {output_root / 'index.jsonl'}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="unsplash-prepare",
        description="Download images and refine annotations for the Unsplash Lite dataset.",
    )
    p.add_argument(
        "--dataset_root",
        required=True,
        type=Path,
        help="Path to the extracted Unsplash Lite directory (containing photos.csv000, etc.)",
    )
    p.add_argument(
        "--output_root",
        default="./data/unsplash",
        type=Path,
        help="Output directory (default: ./data/unsplash)",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Pixel width to request from the Unsplash CDN (default: 640). "
             "Use 0 to download full-resolution (not recommended).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download threads (default: 8)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N photos for testing (0 = all, default: 0)",
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image downloads; only build annotation files and index.",
    )

    args = p.parse_args()
    prepare(
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        img_size=args.img_size,
        workers=args.workers,
        limit=args.limit,
        skip_images=args.skip_images,
    )


if __name__ == "__main__":
    main()
