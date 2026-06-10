#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "Implement_Docs/assets_mobilecropnet_v4_master_20260427"
SOURCE_MANIFEST = ASSET_ROOT / "fig17_latency_profile_public_cropper_comparison_manifest.json"
OUTPUT_FIGURE = ASSET_ROOT / "fig17_latency_subset_mcn_public_cropper.png"
OUTPUT_MANIFEST = ASSET_ROOT / "fig17_latency_subset_mcn_public_cropper_manifest.json"

KEEP_METHODS = (
    "MCN balanced_288",
    "MCN rank_320",
    "MCN plus_384",
    "Public GAIC",
    "Public CGS",
)

COLORS = {
    "MobileCropNet": "#2E77BB",
    "Public cropper": "#C87818",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build representative latency subset figure for MobileCropNet v4 report.")
    parser.add_argument("--source_manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output_figure", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument("--output_manifest", type=Path, default=OUTPUT_MANIFEST)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_method = {str(row.get("method")): row for row in payload.get("rows", [])}
    rows = [by_method[name] for name in KEEP_METHODS if name in by_method]
    missing = [name for name in KEEP_METHODS if name not in by_method]
    if missing:
        raise KeyError(f"missing latency rows: {missing}")
    return rows


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f} ms"


def build_figure(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [str(row["method"]).replace("MCN ", "MCN\n").replace("Public ", "Public\n") for row in rows]
    means = np.array([float(row["mean_ms_output"]) for row in rows], dtype=float)
    p95 = np.array([float(row["p95_ms"]) if row.get("p95_ms") is not None else np.nan for row in rows], dtype=float)
    groups = [str(row.get("group", "")) for row in rows]
    colors = [COLORS.get(group, "#777777") for group in groups]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=180)
    x = np.arange(len(rows))
    for ax, values, title, y_limit in (
        (axes[0], means, "Mean Forward Latency", max(means) * 1.22),
        (axes[1], p95, "p95 Forward Latency", float(np.nanmax(p95)) * 1.14),
    ):
        bars = ax.bar(x, values, width=0.62, color=colors, edgecolor="#26313D", linewidth=0.9)
        for idx, (bar, value) in enumerate(zip(bars, values)):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + y_limit * 0.025,
                fmt_ms(float(value)),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#111111",
            )
            if ax is axes[0]:
                params = rows[idx].get("params_m")
                peak = rows[idx].get("peak_mb")
                detail = []
                if params is not None:
                    detail.append(f"{float(params):.1f}M")
                if peak is not None:
                    detail.append(f"{float(peak):.0f}MB")
                if detail and value > 8.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        max(0.6, value * 0.08),
                        " / ".join(detail),
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="white",
                    )
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, y_limit)
        ax.grid(axis="y", color="#C7CDD4", linewidth=0.8, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Latency per output / image group (ms)")
    axes[1].set_ylabel("Latency per output / image group (ms)")
    fig.suptitle("Representative Latency Subset: MCN Profiles vs Public Croppers", fontsize=15.5, fontweight="bold", y=0.985)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["MobileCropNet"], ec="#26313D", label="MobileCropNet"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["Public cropper"], ec="#26313D", label="Public cropper"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=False)

    footer = (
        "RTX 3050 OEM. MCN rows: synthetic model forward-only. "
        "Public GAIC/CGS rows: candidate-ranker forward-only on precomputed tensor/RoI inputs."
    )
    fig.text(0.02, 0.025, footer, fontsize=8.5, color="#48505A")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_manifest(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "figure": str(args.output_figure.relative_to(PROJECT_ROOT)),
        "source_manifest": str(args.source_manifest.relative_to(PROJECT_ROOT)),
        "selection": list(KEEP_METHODS),
        "metric_note": "Subset of Figure 17-2 manifest. Mean and p95 latency are plotted; labels also show params M and peak MB where available.",
        "rows": rows,
        "bytes": args.output_figure.stat().st_size,
    }
    args.output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.source_manifest)
    build_figure(rows, args.output_figure)
    write_manifest(rows, args)
    print(json.dumps({"figure": str(args.output_figure), "bytes": args.output_figure.stat().st_size}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
