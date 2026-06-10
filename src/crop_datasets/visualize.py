"""
Dataset Visualization for Unified Crop Annotations
====================================================
Reads the unified JSONL file produced by `crop_datasets.unify` and generates
per-dataset visualisation PNG panels:

  1. Sample images with crop bounding-box overlays (opacity/color = score)
  2. Score distribution histogram
  3. Crops-per-image distribution box-plot / bar chart
  4. Global summary figure (all datasets side-by-side)

Usage:
    python -m crop_datasets.visualize \\
        --jsonl   ./data/unified_crops.jsonl \\
        --data_root ./data \\
        --output_dir ./visualizations \\
        --n_samples 6 \\
        --datasets cpc xpview gaic gaic_v2 unsplash

Output layout:
    visualizations/
        summary.png            # all-datasets statistics panel
        cpc/
            overview.png       # sample images + crop overlays
            score_dist.png     # histogram of crop scores
        xpview/
            overview.png
            score_dist.png
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import numpy as np

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

log = logging.getLogger(__name__)

# ── palette ──────────────────────────────────────────────────────────────────
DATASET_COLORS: dict[str, str] = {
    "cpc":     "#4E79A7",
    "xpview":  "#F28E2B",
    "flms":    "#E15759",
    "gaic":    "#76B7B2",
    "gaic_v2": "#59A14F",
    "unsplash": "#B07AA1",
}
CMAP_CROP = plt.cm.RdYlGn   # score: red (bad) → green (good)

# ── helpers ──────────────────────────────────────────────────────────────────

def _rgba(score: float, alpha: float = 0.35) -> tuple:
    norm = max(0.0, min(1.0, score))
    r, g, b, _ = CMAP_CROP(norm)
    return r, g, b, alpha


def _load_image(img_path: Path) -> "Image.Image | None":
    if not HAS_PIL:
        return None
    if not img_path.exists():
        return None
    try:
        return Image.open(img_path).convert("RGB")
    except Exception:
        return None


def _placeholder(w: int = 400, h: int = 300, label: str = "") -> "Image.Image":
    """Grey placeholder if image file is missing."""
    from PIL import ImageDraw
    img = Image.new("RGB", (w, h), color=(200, 200, 200))
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((10, h // 2 - 10), label, fill=(80, 80, 80))
    return img


def _best_worst_avg_crops(crops: list[dict]) -> tuple:
    """Return (best_crop, worst_crop, mean_score)."""
    if not crops:
        return None, None, 0.0
    sorted_c = sorted(crops, key=lambda c: c["score"])
    return sorted_c[-1], sorted_c[0], sum(c["score"] for c in crops) / len(crops)


# ── per-dataset loader ────────────────────────────────────────────────────────

def load_jsonl(jsonl_path: Path, datasets: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Load unified JSONL into {dataset_name: [record, ...]}.
    """
    data: dict[str, list[dict]] = defaultdict(list)
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ds = rec["dataset"]
            if datasets and ds not in datasets:
                continue
            data[ds].append(rec)
    return dict(data)


# ── per-image overlay plot ────────────────────────────────────────────────────

def _draw_crop_overlays(ax, img: "Image.Image", crops: list[dict],
                        max_crops: int = 30, top_k: int = 5) -> None:
    """Draw crop rectangles on ax, colored by score."""
    ax.imshow(img)
    w, h = img.size

    # Show at most max_crops crops, keeping top_k best + random sample of rest
    sorted_c = sorted(crops, key=lambda c: c["score"], reverse=True)
    shown = sorted_c[:top_k]
    remaining = sorted_c[top_k:]
    if remaining:
        n_extra = min(max_crops - top_k, len(remaining))
        shown += random.sample(remaining, n_extra)

    norm = Normalize(vmin=0, vmax=1)
    for crop in shown:
        x1 = max(0, crop["crop_x1"])
        y1 = max(0, crop["crop_y1"])
        x2 = min(w, crop["crop_x2"])
        y2 = min(h, crop["crop_y2"])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        color = CMAP_CROP(norm(crop["score"]))
        lw = 2.5 if crop["score"] >= 0.7 else 1.0
        rect = mpatches.FancyBboxPatch(
            (x1, y1), bw, bh,
            boxstyle="square,pad=0",
            linewidth=lw,
            edgecolor=color,
            facecolor=(*color[:3], 0.08),
        )
        ax.add_patch(rect)

    # Colorbar legend
    sm = ScalarMappable(cmap=CMAP_CROP, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="score")
    ax.axis("off")


# ── overview panel ────────────────────────────────────────────────────────────

def make_overview(
    records: list[dict],
    data_root: Path,
    dataset_name: str,
    n_samples: int = 6,
    seed: int = 42,
) -> plt.Figure:
    """
    Grid of n_samples images, each showing the top-5 (green) and
    bottom-5 (red) crop boxes, plus the image's mean score.

    Layout: 2 rows × (n_samples/2) cols for images,
            1 row below for score-distribution histogram.
    """
    random.seed(seed)
    color = DATASET_COLORS.get(dataset_name, "#888888")

    # Sample records that have a real image
    available = [r for r in records if
                 (data_root / r["image_path"]).exists() and r["crops"]]
    if not available:
        available = records
    samples = random.sample(available, min(n_samples, len(available)))

    ncols = min(n_samples, 3)
    nrows = math.ceil(len(samples) / ncols)

    # Figure layout: image grid (70%) + score hist (30%)
    fig = plt.figure(figsize=(ncols * 5, nrows * 4 + 3.5), facecolor="#111111")
    fig.suptitle(
        f"Dataset: {dataset_name.upper()}   "
        f"({len(records):,} images · {sum(len(r['crops']) for r in records):,} crops)",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )

    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[nrows * 4, 3.0], hspace=0.35)
    img_grid = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=outer[0], hspace=0.05, wspace=0.05
    )

    for idx, rec in enumerate(samples):
        ax = fig.add_subplot(img_grid[idx // ncols, idx % ncols])
        img_path = data_root / rec["image_path"]
        img = _load_image(img_path) or _placeholder(label=rec["image_id"])
        _, _, mean_score = _best_worst_avg_crops(rec["crops"])
        _draw_crop_overlays(ax, img, rec["crops"])
        ax.set_title(
            f"{rec['image_id']}\n"
            f"crops={len(rec['crops'])}  "
            f"mean_score={mean_score:.3f}",
            fontsize=7, color="white", pad=2,
        )

    # Score histogram (all crops in selected samples)
    ax_hist = fig.add_subplot(outer[1])
    all_scores = [c["score"] for r in records for c in r["crops"]]
    ax_hist.hist(all_scores, bins=50, color=color, edgecolor="white",
                 linewidth=0.3, alpha=0.85)
    ax_hist.set_facecolor("#1e1e1e")
    ax_hist.set_xlabel("Normalised Score (0 = worst, 1 = best)",
                       color="white", fontsize=9)
    ax_hist.set_ylabel("# Crops", color="white", fontsize=9)
    ax_hist.set_title(f"Score Distribution — {dataset_name}",
                      color="white", fontsize=10, fontweight="bold")
    ax_hist.tick_params(colors="white")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#555")
    ax_hist.set_xlim(0, 1)
    ax_hist.axvline(float(np.mean(all_scores)), color="yellow",
                    linestyle="--", linewidth=1.5,
                    label=f"mean = {np.mean(all_scores):.3f}")
    ax_hist.legend(fontsize=8, labelcolor="white",
                   facecolor="#333", edgecolor="#555")

    fig.patch.set_facecolor("#111111")
    return fig


# ── score distribution (standalone) ──────────────────────────────────────────

def make_score_dist(records: list[dict], dataset_name: str) -> plt.Figure:
    """Detailed score-distribution panel: histogram + box per split."""
    color = DATASET_COLORS.get(dataset_name, "#888888")
    splits = defaultdict(list)
    for rec in records:
        for c in rec["crops"]:
            splits[rec.get("split", "train")].append(c["score"])

    all_scores = [c["score"] for r in records for c in r["crops"]]
    ncols = 1 + len(splits)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5),
                             facecolor="#111111")
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f"{dataset_name.upper()} — Score Analysis",
                 color="white", fontweight="bold", fontsize=13)

    # Global histogram
    ax0 = axes[0]
    ax0.set_facecolor("#1e1e1e")
    ax0.hist(all_scores, bins=60, color=color, edgecolor="none", alpha=0.85)
    ax0.axvline(np.mean(all_scores), color="yellow", ls="--", lw=1.5,
                label=f"μ={np.mean(all_scores):.3f}")
    ax0.axvline(np.median(all_scores), color="cyan", ls=":", lw=1.5,
                label=f"med={np.median(all_scores):.3f}")
    ax0.set_xlabel("Score", color="white")
    ax0.set_ylabel("Count", color="white")
    ax0.set_title("All Crops", color="white")
    ax0.legend(fontsize=8, labelcolor="white", facecolor="#333", edgecolor="#555")
    ax0.tick_params(colors="white")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#555")
    ax0.set_xlim(0, 1)

    # Per-split boxplots
    split_names = sorted(splits.keys())
    for i, split in enumerate(split_names, start=1):
        ax = axes[i]
        ax.set_facecolor("#1e1e1e")
        s_scores = splits[split]
        bp = ax.boxplot(s_scores, vert=True, patch_artist=True,
                        medianprops=dict(color="yellow", linewidth=2),
                        boxprops=dict(facecolor=color, alpha=0.6),
                        whiskerprops=dict(color="white"),
                        capprops=dict(color="white"),
                        flierprops=dict(markerfacecolor=color,
                                        marker=".", markersize=2, alpha=0.3))
        ax.set_title(f"split={split}\nn={len(s_scores):,}", color="white")
        ax.set_ylabel("Score", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.set_ylim(0, 1)
        ax.text(1.25, np.mean(s_scores), f"μ={np.mean(s_scores):.3f}",
                color="yellow", fontsize=8, va="center")

    for ax in axes:
        ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#111111")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── crops-per-image distribution ─────────────────────────────────────────────

def make_crops_per_image(records: list[dict], dataset_name: str) -> plt.Figure:
    """Bar / histogram of how many crops each image has."""
    color = DATASET_COLORS.get(dataset_name, "#888888")
    counts = [len(r["crops"]) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   facecolor="#111111")
    fig.suptitle(f"{dataset_name.upper()} — Crops per Image",
                 color="white", fontweight="bold", fontsize=13)

    # Histogram
    ax1.set_facecolor("#1e1e1e")
    ax1.hist(counts, bins=min(40, max(counts) - min(counts) + 1),
             color=color, edgecolor="none", alpha=0.85)
    ax1.axvline(np.mean(counts), color="yellow", ls="--", lw=1.5,
                label=f"μ={np.mean(counts):.1f}")
    ax1.set_xlabel("# Crops", color="white")
    ax1.set_ylabel("# Images", color="white")
    ax1.set_title("Distribution", color="white")
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#333", edgecolor="#555")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#555")

    # Cumulative CDF
    ax2.set_facecolor("#1e1e1e")
    sorted_c = np.sort(counts)
    cdf = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
    ax2.plot(sorted_c, cdf, color=color, linewidth=2)
    ax2.axvline(np.percentile(sorted_c, 50), color="yellow", ls="--", lw=1,
                label=f"p50={np.percentile(sorted_c,50):.0f}")
    ax2.axvline(np.percentile(sorted_c, 90), color="cyan", ls=":", lw=1,
                label=f"p90={np.percentile(sorted_c,90):.0f}")
    ax2.set_xlabel("# Crops", color="white")
    ax2.set_ylabel("CDF", color="white")
    ax2.set_title("CDF", color="white")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#333", edgecolor="#555")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#555")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.15, color="white")

    fig.patch.set_facecolor("#111111")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── global summary ────────────────────────────────────────────────────────────

def make_summary(all_data: dict[str, list[dict]]) -> plt.Figure:
    """
    Multi-panel summary comparing all datasets:
    - Bar chart: image count per dataset
    - Bar chart: crop count per dataset
    - Violin/box: score distribution per dataset
    - Bar chart: avg crops per image
    """
    datasets = sorted(all_data.keys())
    colors = [DATASET_COLORS.get(ds, "#888888") for ds in datasets]

    img_counts  = [len(all_data[ds]) for ds in datasets]
    crop_counts = [sum(len(r["crops"]) for r in all_data[ds]) for ds in datasets]
    avg_crops   = [c / max(i, 1) for c, i in zip(crop_counts, img_counts)]
    all_scores  = {ds: [c["score"] for r in all_data[ds] for c in r["crops"]]
                   for ds in datasets}
    mean_scores = [np.mean(s) if s else 0 for s in all_scores.values()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#111111")
    fig.suptitle("Unified Dataset Summary", color="white",
                 fontsize=16, fontweight="bold", y=0.98)

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)

    # ①  Image count
    ax = axes[0, 0]
    bars = ax.bar(datasets, img_counts, color=colors, edgecolor="none", alpha=0.85)
    for bar, v in zip(bars, img_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:,}", ha="center", va="bottom", fontsize=8, color="white")
    _style(ax, "Images per Dataset", "Dataset", "Count")
    ax.tick_params(axis="x", rotation=20)

    # ②  Crop count
    ax = axes[0, 1]
    bars = ax.bar(datasets, crop_counts, color=colors, edgecolor="none", alpha=0.85)
    for bar, v in zip(bars, crop_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:,}", ha="center", va="bottom", fontsize=8, color="white")
    _style(ax, "Crops per Dataset", "Dataset", "Count")
    ax.tick_params(axis="x", rotation=20)

    # ③  Avg crops per image
    ax = axes[0, 2]
    bars = ax.bar(datasets, avg_crops, color=colors, edgecolor="none", alpha=0.85)
    for bar, v in zip(bars, avg_crops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="white")
    _style(ax, "Avg Crops / Image", "Dataset", "Count")
    ax.tick_params(axis="x", rotation=20)

    # ④  Score violin
    ax = axes[1, 0]
    score_lists = [all_scores[ds] for ds in datasets]
    valid = [(i, s) for i, s in enumerate(score_lists) if len(s) > 0]
    if valid:
        vp = ax.violinplot([s for _, s in valid],
                           positions=[i + 1 for i, _ in valid],
                           showmedians=True, showextrema=True)
        for body, (i, _) in zip(vp["bodies"], valid):
            body.set_facecolor(colors[i])
            body.set_alpha(0.6)
        vp["cmedians"].set_color("yellow")
        vp["cbars"].set_color("white")
        vp["cmaxes"].set_color("white")
        vp["cmins"].set_color("white")
        ax.set_xticks(range(1, len(datasets) + 1))
        ax.set_xticklabels(datasets, rotation=20, ha="right")
    _style(ax, "Score Distribution (violin)", "Dataset", "Score (0–1)")
    ax.set_ylim(0, 1)

    # ⑤  Mean score bar
    ax = axes[1, 1]
    bars = ax.bar(datasets, mean_scores, color=colors, edgecolor="none", alpha=0.85)
    for bar, v in zip(bars, mean_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="white")
    _style(ax, "Mean Score per Dataset", "Dataset", "Mean Score (0–1)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(0.5, color="#aaa", ls="--", lw=0.8)

    # ⑥  Score stacked histogram (all datasets overlaid)
    ax = axes[1, 2]
    ax.set_facecolor("#1e1e1e")
    for ds in datasets:
        s = all_scores[ds]
        if not s:
            continue
        weights = np.ones(len(s)) / len(s)
        ax.hist(s, bins=40, weights=weights,
                color=DATASET_COLORS.get(ds, "#888"),
                alpha=0.55, label=ds, edgecolor="none")
    ax.set_xlabel("Score", color="white", fontsize=9)
    ax.set_ylabel("Normalised Freq.", color="white", fontsize=9)
    ax.set_title("Score Distribution Overlay", color="white",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#333",
              edgecolor="#555", loc="upper left")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.set_xlim(0, 1)

    fig.patch.set_facecolor("#111111")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── main pipeline ─────────────────────────────────────────────────────────────

def visualize(
    jsonl_path: Path,
    data_root: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    n_samples: int = 6,
    seed: int = 42,
) -> None:
    if not HAS_PIL:
        log.warning("Pillow not installed — image overlays disabled. pip install Pillow")

    log.info("Loading %s …", jsonl_path)
    all_data = load_jsonl(jsonl_path, datasets)

    if not all_data:
        log.error("No records loaded. Check --jsonl path and --datasets filter.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset figures
    for ds_name, records in sorted(all_data.items()):
        ds_dir = output_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        log.info("[%s] %d images, %d crops",
                 ds_name, len(records), sum(len(r["crops"]) for r in records))

        # 1. Overview (sample images + overlays)
        log.info("  → overview.png")
        fig = make_overview(records, data_root, ds_name,
                            n_samples=n_samples, seed=seed)
        fig.savefig(ds_dir / "overview.png", dpi=130,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # 2. Score distribution
        log.info("  → score_dist.png")
        fig = make_score_dist(records, ds_name)
        fig.savefig(ds_dir / "score_dist.png", dpi=130,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # 3. Crops-per-image distribution
        log.info("  → crops_per_image.png")
        fig = make_crops_per_image(records, ds_name)
        fig.savefig(ds_dir / "crops_per_image.png", dpi=130,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    # Global summary
    log.info("→ summary.png")
    fig = make_summary(all_data)
    fig.savefig(output_dir / "summary.png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Report
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(f"   summary.png")
    for ds in sorted(all_data.keys()):
        print(f"   {ds}/overview.png")
        print(f"   {ds}/score_dist.png")
        print(f"   {ds}/crops_per_image.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="crop-datasets-visualize",
        description="Visualise unified crop annotations per dataset.",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=Path("./data/unified_crops.jsonl"),
        help="Path to unified JSONL file (default: ./data/unified_crops.jsonl)",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("./data"),
        help="Dataset root directory for loading images (default: ./data)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./visualizations"),
        help="Directory to save visualisation PNGs (default: ./visualizations)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to visualise (default: all found in JSONL)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=6,
        help="Number of sample images to show per dataset overview (default: 6)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )

    args = p.parse_args()

    visualize(
        jsonl_path=args.jsonl.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        datasets=args.datasets,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
