#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Implement_Docs" / "assets_mobilecropnet_training_data_master_20260424"

COLOR_BG = "#F6F7FB"
COLOR_TEXT = "#162033"
COLOR_GRID = "#D7DCE8"
COLOR_ACCENT = "#1F4E79"
COLOR_ACCENT_2 = "#FF7A59"
COLOR_ACCENT_3 = "#4CAF50"
COLOR_ACCENT_4 = "#8E5EA2"
COLOR_WARN = "#D62728"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "text.color": COLOR_TEXT,
            "axes.titleweight": "bold",
            "axes.titlepad": 10.0,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
            "font.size": 10.0,
        }
    )


def add_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.08, title, fontsize=12, fontweight="bold", va="top", ha="left")
    ax.text(x + 0.02, y + h - 0.15, body, fontsize=9.5, va="top", ha="left", linespacing=1.4)


def add_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = COLOR_TEXT) -> None:
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=color)
    ax.add_patch(arrow)


def plot_pipeline_overview(out_path: Path) -> None:
    figure_style()
    fig, ax = plt.subplots(figsize=(16, 6.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.03,
        0.56,
        0.18,
        0.28,
        "1. Source Corpus",
        "SSTK / GAIC / public sets\nExisting docs + feature stores\nCandidate traces + benchmark artifacts",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.25,
        0.56,
        0.18,
        0.28,
        "2. Perception C1-C7",
        "Caption, OCR, pose, face,\nsegmentation, geometry,\nsaliency-support attribution",
        COLOR_ACCENT_2,
    )
    add_box(
        ax,
        0.47,
        0.56,
        0.20,
        0.28,
        "3. Routing + Guidance",
        "subject_mode / route_family\nentity atoms\nquery-local envelopes and anchors",
        COLOR_ACCENT_3,
    )
    add_box(
        ax,
        0.71,
        0.56,
        0.24,
        0.28,
        "4. Candidate Bank",
        "AR-conditioned bank\nsynthetic seed expansion\nproposal injection\nFREE + target-AR candidates",
        COLOR_ACCENT_4,
    )
    add_box(
        ax,
        0.08,
        0.13,
        0.23,
        0.24,
        "5. Teacher Scoring",
        "FinalScore macros/policy\nmultimode query scorer\nT6 score-only ranker\nUCTR-H / public ensemble",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.39,
        0.13,
        0.24,
        0.24,
        "6. Label Builders",
        "pairwise / listwise / decision\nlocal-normalized regression\nCOCO multimode annotations\nMobileCropNet batch JSONL",
        COLOR_ACCENT_2,
    )
    add_box(
        ax,
        0.70,
        0.13,
        0.22,
        0.24,
        "7. Evaluation Loop",
        "GAIC official\nFCDB / CPC / GNMC\nEqual-4 leaderboards\ndeployment shortlist diagnostics",
        COLOR_ACCENT_3,
    )

    add_arrow(ax, 0.21, 0.70, 0.25, 0.70)
    add_arrow(ax, 0.43, 0.70, 0.47, 0.70)
    add_arrow(ax, 0.67, 0.70, 0.71, 0.70)
    add_arrow(ax, 0.83, 0.56, 0.83, 0.37)
    add_arrow(ax, 0.59, 0.49, 0.49, 0.37)
    add_arrow(ax, 0.19, 0.49, 0.19, 0.37)
    add_arrow(ax, 0.31, 0.25, 0.39, 0.25)
    add_arrow(ax, 0.63, 0.25, 0.70, 0.25)
    add_arrow(ax, 0.81, 0.13, 0.81, 0.06, color=COLOR_WARN)
    ax.text(
        0.81,
        0.03,
        "Current blocker: route-collapse hard gate",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLOR_WARN,
    )
    ax.set_title("MobileCropNet Training-Data Generation: End-to-End Factory", fontsize=16, pad=16)
    save_fig(fig, out_path)


def plot_training_contract(out_path: Path) -> None:
    figure_style()
    fig, ax = plt.subplots(figsize=(16.8, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.03,
        0.56,
        0.24,
        0.30,
        "1. Batch JSONL Contract",
        "image_path, target_ar\nbaseline / decision_target\nmatching_targets / candidate_pool\nignored / overflow / teacher_meta",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.03,
        0.14,
        0.24,
        0.24,
        "2. Sidecars",
        "pairwise.jsonl\nlistwise.jsonl\ncandidate_id keyed merge\nteacher_softmax_local",
        COLOR_ACCENT_2,
    )
    add_box(
        ax,
        0.33,
        0.56,
        0.25,
        0.30,
        "3. Candidate Selection",
        "baseline always included\nmatching_targets first\nsafe high-score positives\noptional ignored / overflow reuse\ntruncate to candidate_k",
        COLOR_ACCENT_3,
    )
    add_box(
        ax,
        0.33,
        0.14,
        0.25,
        0.24,
        "4. Detail Expansion",
        "score_target / risk_target\nmacro_target\nchecklist class + detail score\nwhy_tags / applicability\nexplicit pairwise tensors",
        COLOR_ACCENT_4,
    )
    add_box(
        ax,
        0.64,
        0.56,
        0.30,
        0.30,
        "5. Geometry + Priors",
        "letterbox image tensor\nboxes / boxes_orig / box_meta\npositive_boxes\nsubject_prior_box\nsubject_box_valid / weight",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.64,
        0.14,
        0.30,
        0.24,
        "6. Multi-Head Supervision",
        "rank / teacher_distill\npositive / risk\nroute / decision / delta\nproposal + subject-box heads\nchecklist / why-tag heads",
        COLOR_ACCENT_2,
    )

    add_arrow(ax, 0.27, 0.71, 0.33, 0.71)
    add_arrow(ax, 0.27, 0.26, 0.33, 0.26)
    add_arrow(ax, 0.455, 0.56, 0.455, 0.38)
    add_arrow(ax, 0.58, 0.71, 0.64, 0.71)
    add_arrow(ax, 0.58, 0.26, 0.64, 0.26)
    ax.text(
        0.49,
        0.04,
        "Source of truth: mobilecropnet_v4.data merges batch JSONL and pairwise/listwise sidecars into fixed-size tensor supervision",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=COLOR_TEXT,
    )
    ax.set_title("Actual Student Input Contract for MobileCropNet v4", fontsize=16, pad=14)
    save_fig(fig, out_path)


def plot_label_lineages(out_path: Path) -> None:
    figure_style()
    fig, ax = plt.subplots(figsize=(17.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Core label lineages for MobileCropNet training", fontsize=17, fontweight="bold")
    ax.text(
        0.03,
        0.915,
        "Three code-grounded lineages converge to the same student contract: SSTK main factory, GAIC teacher distillation, and Product-AR teacher fusion.",
        fontsize=10.5,
    )

    y_positions = [0.68, 0.40, 0.12]
    row_titles = [
        ("A. SSTK curation + main factory", COLOR_ACCENT),
        ("B. GAIC teacher distillation", COLOR_ACCENT_2),
        ("C. Product-AR teacher factory", COLOR_ACCENT_3),
    ]
    for (title, color), y in zip(row_titles, y_positions):
        ax.text(0.03, y + 0.16, title, fontsize=12.5, fontweight="bold", color=color)
        ax.plot([0.03, 0.97], [y + 0.13, y + 0.13], color=COLOR_GRID, linewidth=1.0)

    add_box(
        ax,
        0.03,
        0.66,
        0.18,
        0.17,
        "Curate",
        "filter_sstk_dataset.py\nPhoto only, dedup reject,\naesthetic percentile cut,\nlong-tail weighted sampling",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.27,
        0.66,
        0.18,
        0.17,
        "PhaseA E2E",
        "run_phaseA_to_teacher_e2e.sh\nC1-C7 -> routing -> candidates\nteacher -> QA/report",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.51,
        0.66,
        0.18,
        0.17,
        "FinalScore Builder",
        "build_finalscore_training_data.py\nrepair + safe_leftover_policy\nmonotonic split",
        COLOR_ACCENT,
    )
    add_box(
        ax,
        0.75,
        0.66,
        0.21,
        0.17,
        "Output Contract",
        "train_conditional_detr_batch.jsonl\npairwise/listwise/decision\nmature mainline",
        COLOR_ACCENT,
    )
    add_arrow(ax, 0.21, 0.745, 0.27, 0.745, color=COLOR_ACCENT)
    add_arrow(ax, 0.45, 0.745, 0.51, 0.745, color=COLOR_ACCENT)
    add_arrow(ax, 0.69, 0.745, 0.75, 0.745, color=COLOR_ACCENT)
    ax.text(0.855, 0.635, "SSTK batch rows: 48,766", ha="center", fontsize=9.5, color=COLOR_ACCENT)

    add_box(
        ax,
        0.03,
        0.38,
        0.18,
        0.17,
        "Prepare",
        "prepare_gaic_curated_dataset.py\nflat image dir + pseudo parquet\nall images treated as curated",
        COLOR_ACCENT_2,
    )
    add_box(
        ax,
        0.27,
        0.38,
        0.18,
        0.17,
        "GAIC E2E",
        "run_gaic_to_teacher_e2e.sh\nGAIC-safe defaults\nreal-expensive off by default",
        COLOR_ACCENT_2,
    )
    add_box(
        ax,
        0.51,
        0.38,
        0.18,
        0.17,
        "T6 / Public Export",
        "export_gaic_ranker_training_labels.py\nconvert_t6_score_labels_to_mobilecropnet_v4_batch.py\npublic explain-safe/distill",
        COLOR_ACCENT_2,
    )
    add_box(
        ax,
        0.75,
        0.38,
        0.21,
        0.17,
        "Output Lineages",
        "corrected_v2b: 86.4 cand/task\nproduct_ar_v1: 6.0 cand/task\npublic distill: dense pairwise",
        COLOR_ACCENT_2,
    )
    add_arrow(ax, 0.21, 0.465, 0.27, 0.465, color=COLOR_ACCENT_2)
    add_arrow(ax, 0.45, 0.465, 0.51, 0.465, color=COLOR_ACCENT_2)
    add_arrow(ax, 0.69, 0.465, 0.75, 0.465, color=COLOR_ACCENT_2)

    add_box(
        ax,
        0.03,
        0.10,
        0.18,
        0.17,
        "Candidate Groups",
        "export_mobilecropnet_v4_product_ar_candidates.py\nexisting batch -> Product-AR groups\nfallback rows allowed",
        COLOR_ACCENT_3,
    )
    add_box(
        ax,
        0.27,
        0.10,
        0.18,
        0.17,
        "Teacher Branches",
        "T1 direct\nUCTR stage3 scoring\npublic ensemble fusion",
        COLOR_ACCENT_3,
    )
    add_box(
        ax,
        0.51,
        0.10,
        0.18,
        0.17,
        "Batch Builder",
        "build_mobilecropnet_v4_product_ar_t1_labels.py\nbuild_mobilecropnet_v4_product_ar_score_labels.py\nfallback=sstk_positive",
        COLOR_ACCENT_3,
    )
    add_box(
        ax,
        0.75,
        0.10,
        0.21,
        0.17,
        "Current Status",
        "Code path complete\nmanifest + smoke verified\nfull-corpus materialization still partial",
        COLOR_ACCENT_3,
    )
    add_arrow(ax, 0.21, 0.185, 0.27, 0.185, color=COLOR_ACCENT_3)
    add_arrow(ax, 0.45, 0.185, 0.51, 0.185, color=COLOR_ACCENT_3)
    add_arrow(ax, 0.69, 0.185, 0.75, 0.185, color=COLOR_ACCENT_3)
    ax.text(
        0.86,
        0.045,
        "Multimode is a strong auxiliary corpus,\nnot the only default narrative for Product-AR student training.",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color=COLOR_TEXT,
    )
    save_fig(fig, out_path)


def annotate_barh(ax: plt.Axes, values: list[float], ys: Iterable[float], fmt: str) -> None:
    for y, value in zip(ys, values):
        ax.text(value, y, " " + format(value, fmt), va="center", ha="left", fontsize=9)


def plot_multimode_summary(multimode_summary: dict[str, Any], out_path: Path) -> None:
    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.suptitle("SSTK Multimode Label Build Summary", fontsize=16, fontweight="bold")

    mode_summary = multimode_summary["mode_summary"]
    modes = list(mode_summary.keys())
    values = [int(mode_summary[k]["positive_query_count"]) for k in modes]
    order = sorted(range(len(modes)), key=lambda idx: values[idx])
    modes = [modes[idx] for idx in order]
    values = [values[idx] for idx in order]
    axes[0].barh(modes, values, color=COLOR_ACCENT)
    annotate_barh(axes[0], values, range(len(modes)), ",d")
    axes[0].set_title("Positive queries by mode")
    axes[0].set_xlabel("queries")

    rate_names = modes
    rate_values = [100.0 * safe_float(mode_summary[k]["positive_query_rate"], 0.0) for k in modes]
    axes[1].bar(rate_names, rate_values, color=COLOR_ACCENT_2)
    for x, value in enumerate(rate_values):
        axes[1].text(x, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Positive-query rate by mode")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylabel("rate (%)")

    target_ar = multimode_summary["target_ar"]
    dec_names = list(target_ar.keys())
    dec_values = [int(target_ar[k]["positive_query_count"]) for k in dec_names]
    axes[2].bar(dec_names, dec_values, color=[COLOR_ACCENT_3, COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_4, COLOR_ACCENT, COLOR_ACCENT_2])
    for x, value in enumerate(dec_values):
        axes[2].text(x, value, format(value, ",d"), ha="center", va="bottom", fontsize=9)
    axes[2].set_title("Positive queries by target AR")
    axes[2].set_ylabel("queries")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save_fig(fig, out_path)


def collect_gaic_conversion_stats(project_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    splits = {
        "Train": project_root / "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636",
        "Val": project_root / "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200",
        "Test": project_root / "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500",
    }
    label_dirs = {
        "T6 corrected_v2b": "training_labels_t6_score_only_corrected_v2b/conversion_summary.json",
        "T6 product_ar_v1": "training_labels_t6_score_product_ar_v1/conversion_summary.json",
    }
    stats: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for split_name, split_root in splits.items():
        for label_name, rel in label_dirs.items():
            payload = read_json(split_root / "artifacts" / rel)
            stats[label_name][split_name] = {
                "written_rows": safe_float(payload.get("written_rows"), 0.0),
                "pairwise_rows": safe_float(payload.get("pairwise_rows"), 0.0),
                "listwise_rows": safe_float(payload.get("listwise_rows"), 0.0),
                "candidate_count_mean": safe_float(payload.get("candidate_count_mean"), 0.0),
            }
    return stats


def plot_gaic_conversion_comparison(stats: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.suptitle("GAIC Label-Lineage Comparison", fontsize=16, fontweight="bold")
    splits = ["Train", "Val", "Test"]
    series = list(stats.keys())
    colors = [COLOR_ACCENT, COLOR_ACCENT_2]
    metrics = [
        ("written_rows", "Written tasks"),
        ("pairwise_rows", "Pairwise rows"),
        ("candidate_count_mean", "Mean candidates / task"),
    ]
    width = 0.36
    xs = list(range(len(splits)))
    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for idx, name in enumerate(series):
            vals = [stats[name][split][metric_key] for split in splits]
            bars = ax.bar([x + (idx - 0.5) * width for x in xs], vals, width=width, label=name, color=colors[idx])
            for bar, value in zip(bars, vals):
                label = f"{value:.2f}" if metric_key == "candidate_count_mean" else format(int(round(value)), ",d")
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(splits)
        ax.set_title(metric_title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    save_fig(fig, out_path)


def shorten_method(name: str) -> str:
    mapping = {
        "production_final_hybrid": "Hybrid final",
        "universal_crop_teacher_h_stage3_gate128": "UCTR-H stage3",
        "universal_crop_teacher_h_stage2_fullhonest": "UCTR-H stage2",
        "public_cropper_ensemble_best": "Public ensemble",
        "public_cropper_gaic": "Public GAIC",
        "public_cropper_cgs": "Public CGS",
        "teacher_proxy_compact_utility_pool": "Compact proxy",
        "sstk_teacher_t1_historical_gaic_plus_public_proxy": "Historical T1",
    }
    return mapping.get(name, name)


def plot_teacher_equal4(leaderboard: dict[str, Any], out_path: Path) -> None:
    figure_style()
    rows = sorted(leaderboard["rows"], key=lambda row: safe_float(row.get("equal4_zscore"), -1e9), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2))
    fig.suptitle("Teacher Benchmark Landscape", fontsize=16, fontweight="bold")

    names = [shorten_method(str(row["method"])) for row in rows]
    scores = [safe_float(row.get("equal4_zscore"), 0.0) for row in rows]
    colors = [COLOR_ACCENT if idx == 0 else COLOR_ACCENT_2 if idx < 3 else "#9BA7BF" for idx in range(len(rows))]
    axes[0].barh(names[::-1], scores[::-1], color=colors[::-1])
    axes[0].set_title("Equal-4 z-score")
    axes[0].set_xlabel("z-score")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    top_rows = rows[:4]
    metric_keys = [
        ("fcdb_iou_top1", "FCDB"),
        ("cpc_weighted_pairwise", "CPC"),
        ("gnmc_iou_top1", "GNMC"),
        ("gaic_primary", "GAIC"),
    ]
    base_x = list(range(len(top_rows)))
    bar_w = 0.18
    metric_colors = [COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_3, COLOR_ACCENT_4]
    for metric_idx, ((metric_key, metric_label), color) in enumerate(zip(metric_keys, metric_colors)):
        vals = [safe_float(row.get(metric_key), 0.0) for row in top_rows]
        bars = axes[1].bar([x + (metric_idx - 1.5) * bar_w for x in base_x], vals, width=bar_w, label=metric_label, color=color)
        for bar, value in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(base_x)
    axes[1].set_xticklabels([shorten_method(str(row["method"])) for row in top_rows], rotation=10)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Per-dataset primaries (top 4)")
    axes[1].legend(frameon=False, ncol=2, loc="lower right")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    save_fig(fig, out_path)


def family_from_track(track: str) -> str:
    track_l = str(track).lower()
    if "sstk public" in track_l:
        return "SSTK public"
    if "sstk uctr" in track_l:
        return "SSTK UCTR"
    if "sstk t1" in track_l:
        return "SSTK T1"
    if "gaic public" in track_l:
        return "GAIC public"
    if "gaic uctr subjectprior" in track_l:
        return "GAIC UCTR subj"
    if "gaic uctr" in track_l:
        return "GAIC UCTR"
    if "gaic t1" in track_l:
        return "GAIC T1"
    return str(track)


def short_student_label(row: dict[str, Any]) -> str:
    track = str(row.get("track", ""))
    profile = str(row.get("profile", ""))
    mapping = {
        "SSTK public": "SSTK public",
        "SSTK UCTR subjectprior": "SSTK UCTR",
        "SSTK T1": "SSTK T1",
        "GAIC public": "GAIC public",
        "GAIC UCTR subjectprior": "GAIC UCTR+subj",
        "GAIC UCTR": "GAIC UCTR",
        "GAIC T1": "GAIC T1",
    }
    track_short = mapping.get(track, track)
    return f"{track_short} / {profile}"


def plot_student_diagnostics(leaderboard: dict[str, Any], out_path: Path) -> None:
    figure_style()
    rows = [row for row in leaderboard["rows"] if row.get("public_summary_exists")]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4))
    fig.suptitle("MobileCropNet Shortlist Diagnostics", fontsize=16, fontweight="bold")
    family_colors = {
        "SSTK public": COLOR_ACCENT,
        "SSTK UCTR": COLOR_ACCENT_3,
        "SSTK T1": COLOR_ACCENT_4,
        "GAIC public": COLOR_ACCENT_2,
        "GAIC UCTR subj": "#17A2B8",
        "GAIC UCTR": "#6C5CE7",
        "GAIC T1": "#E67E22",
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[family_from_track(str(row.get("track", "")))].append(row)

    for family, family_rows in grouped.items():
        xs = [safe_float(row.get("equal4_raw_mean"), 0.0) for row in family_rows]
        ys = [safe_float(row.get("official_top1_mos"), 0.0) for row in family_rows]
        axes[0].scatter(xs, ys, s=90, label=family, color=family_colors.get(family, "#555555"), alpha=0.9)
    key_labels = []
    for row in rows:
        track = str(row.get("track", ""))
        profile = str(row.get("profile", ""))
        if track == "SSTK public" and profile in {"balanced_288", "rank_320"}:
            key_labels.append(row)
        if track == "GAIC UCTR subjectprior" and profile == "plus_384":
            key_labels.append(row)
    for row in key_labels:
        axes[0].annotate(
            short_student_label(row),
            (safe_float(row.get("equal4_raw_mean"), 0.0), safe_float(row.get("official_top1_mos"), 0.0)),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_xlabel("Equal-4 raw mean")
    axes[0].set_ylabel("GAIC official top1 MOS")
    axes[0].set_title("Benchmark quality trade-off")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    for family, family_rows in grouped.items():
        xs = [safe_float(row.get("route_portrait_single_acc"), 0.0) for row in family_rows]
        ys = [safe_float(row.get("route_portrait_group_acc"), 0.0) for row in family_rows]
        sizes = [260.0 * max(0.2, safe_float(row.get("equal4_raw_mean"), 0.0)) for row in family_rows]
        axes[1].scatter(xs, ys, s=sizes, label=family, color=family_colors.get(family, "#555555"), alpha=0.78)
    axes[1].axvline(0.20, color=COLOR_WARN, linestyle="--", linewidth=1.1)
    axes[1].axhline(0.20, color=COLOR_WARN, linestyle="--", linewidth=1.1)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("portrait_single accuracy")
    axes[1].set_ylabel("portrait_group accuracy")
    axes[1].set_title("Route-collapse diagnostic space")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    save_fig(fig, out_path)


def font_for_pil(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def label_image(canvas: Image.Image, text: str, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 42], fill=(255, 255, 255))
    draw.text((18, 11), text, fill=(22, 32, 51), font=font)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.pad(image, size, color=(245, 246, 250), method=Image.Resampling.LANCZOS)


def build_qualitative_panel(out_path: Path) -> dict[str, str]:
    winner = PROJECT_ROOT / "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_public_subjectprior_spot/mcn-sstk-public-subjasync-r5bal-balanced-288-20260423-034145/viz_test/overlays/0001_sstk_image_1009275442_FREE.png"
    fallback = PROJECT_ROOT / "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_public_subjectprior_spot/mcn-sstk-public-subjasync-r1-rank-320-20260423-025404/viz_test/overlays/0001_sstk_image_1009275442_FREE.png"
    plus = PROJECT_ROOT / "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_public_subjectprior_spot/mcn-sstk-public-subjasync-r5plus-plus-384-20260423-034246/viz_test/overlays/0001_sstk_image_1009275442_FREE.png"
    contact_sheet = PROJECT_ROOT / "Implement_Docs/assets_post_20260317_subject_c7/contact_sheet_subject_c7_before_after.png"
    sources = [winner, fallback, plus, contact_sheet]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)

    canvas = Image.new("RGB", (1800, 1180), color=(246, 247, 251))
    font = font_for_pil(28)
    title_font = font_for_pil(38)
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 22), "Qualitative snapshots: routing-aware winner and C7 subject support", fill=(22, 32, 51), font=title_font)

    cell_w = 540
    cell_h = 390
    margin_x = 36
    top_y = 90
    labels = [
        "Winner: balanced_288",
        "Fallback: rank_320",
        "Alternative: plus_384",
        "C7 subject-support contact sheet",
    ]
    positions = [(margin_x, top_y), (margin_x + 590, top_y), (margin_x + 1180, top_y), (margin_x + 295, top_y + 480)]
    sizes = [(cell_w, cell_h), (cell_w, cell_h), (cell_w, cell_h), (1210, 520)]

    for source, label, (x, y), size in zip(sources, labels, positions, sizes):
        panel = fit_image(source, size)
        framed = Image.new("RGB", (size[0], size[1] + 42), color=(255, 255, 255))
        framed.paste(panel, (0, 42))
        label_image(framed, label, font)
        canvas.paste(framed, (x, y))

    ensure_dir(out_path.parent)
    canvas.save(out_path)
    return {
        "winner": str(winner.relative_to(PROJECT_ROOT)),
        "fallback": str(fallback.relative_to(PROJECT_ROOT)),
        "plus": str(plus.relative_to(PROJECT_ROOT)),
        "contact_sheet": str(contact_sheet.relative_to(PROJECT_ROOT)),
    }


def collect_manifest(output_dir: Path) -> dict[str, Any]:
    qa_summary = read_json(
        PROJECT_ROOT / "data/SSTK/Full_10000/artifacts/training_labels/260413_v1_multimode_leftover_ignore_monotonic/qa_summary.json"
    )
    multimode_summary = read_json(
        PROJECT_ROOT / "data/SSTK/Full_10000/artifacts/training_labels_multimode/260413_v1_multimode_multimode_v1/summary.json"
    )
    teacher_equal4 = read_json(PROJECT_ROOT / "artifacts/unified_public_benchmark_20260423_equal4_teacher/equal4_teacher_leaderboard.json")
    student_shortlist = read_json(
        PROJECT_ROOT / "artifacts/mobilecropnet_v4/shortlist_final_eval_20260423/final_deployment_leaderboard_latest/final_deployment_leaderboard.json"
    )
    warehouse = read_json(PROJECT_ROOT / "artifacts/universal_crop_teacher_h_20260421/full_honest/warehouse_summary.json")
    hybrid_summary = read_json(
        PROJECT_ROOT
        / "artifacts/universal_crop_teacher_h_20260421/hybrid_teacher_20260421/calibrated_stage3_publicensemble/summary.json"
    )
    gaic_conversion = collect_gaic_conversion_stats(PROJECT_ROOT)
    route_collapse_count = sum(1 for row in student_shortlist["rows"] if bool(row.get("route_collapse_flag")))
    return {
        "figures_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "training_contract_sources": [
            "src/mobilecropnet_v4/data.py",
            "src/scripts/train_mobilecropnet_v4.py",
        ],
        "sstk_main_batch": {
            "counts": qa_summary["counts"],
            "decision_counts": qa_summary["decision"]["decision_counts"],
            "pair_count_by_mode": qa_summary["pairwise"]["pair_count_by_mode"],
        },
        "sstk_multimode": {
            "image_count": multimode_summary["image_count"],
            "image_task_count": multimode_summary["image_task_count"],
            "annotation_count": multimode_summary["annotation_count"],
            "positive_annotation_count": multimode_summary["positive_annotation_count"],
            "query_count": multimode_summary["query_count"],
            "positive_query_count": multimode_summary["positive_query_count"],
            "no_positive_query_count": multimode_summary["no_positive_query_count"],
            "mode_summary": multimode_summary["mode_summary"],
            "target_ar": multimode_summary["target_ar"],
        },
        "gaic_conversion": gaic_conversion,
        "uctr_warehouse": {
            "task_count": warehouse.get("task_count"),
            "pairwise_count": warehouse.get("pairwise_count"),
            "candidate_count": warehouse.get("candidate_count"),
            "candidate_count_mean": warehouse.get("candidate_count_mean"),
        },
        "teacher_equal4_top3": teacher_equal4["rows"][:3],
        "hybrid_teacher_gaic_test": hybrid_summary["artifacts"]["gaic_test_summary"],
        "student_shortlist": {
            "entry_count": student_shortlist.get("entry_count"),
            "completed_entry_count": student_shortlist.get("completed_entry_count"),
            "route_collapse_count": route_collapse_count,
            "default_winner": student_shortlist["winners"]["default_deploy_winner"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figures for the MobileCropNet training-data generation master report.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    ensure_dir(output_dir)

    multimode_summary = read_json(
        PROJECT_ROOT / "data/SSTK/Full_10000/artifacts/training_labels_multimode/260413_v1_multimode_multimode_v1/summary.json"
    )
    teacher_equal4 = read_json(PROJECT_ROOT / "artifacts/unified_public_benchmark_20260423_equal4_teacher/equal4_teacher_leaderboard.json")
    student_shortlist = read_json(
        PROJECT_ROOT / "artifacts/mobilecropnet_v4/shortlist_final_eval_20260423/final_deployment_leaderboard_latest/final_deployment_leaderboard.json"
    )
    gaic_conversion = collect_gaic_conversion_stats(PROJECT_ROOT)

    plot_pipeline_overview(output_dir / "fig_pipeline_overview.png")
    plot_training_contract(output_dir / "fig_training_contract.png")
    plot_label_lineages(output_dir / "fig_label_lineages.png")
    plot_multimode_summary(multimode_summary, output_dir / "fig_multimode_sstk_summary.png")
    plot_gaic_conversion_comparison(gaic_conversion, output_dir / "fig_gaic_label_conversion_comparison.png")
    plot_teacher_equal4(teacher_equal4, output_dir / "fig_teacher_equal4_leaderboard.png")
    plot_student_diagnostics(student_shortlist, output_dir / "fig_student_shortlist_diagnostics.png")
    qualitative_sources = build_qualitative_panel(output_dir / "fig_qualitative_panel.png")

    manifest = collect_manifest(output_dir)
    manifest["qualitative_sources"] = qualitative_sources
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "figure_count": 8}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
