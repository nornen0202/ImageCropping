from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("Implement_Docs/assets_post_20260317_summary")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_source_counts(path: str | Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = str(row.get("source", "")).strip()
            if source:
                counts[source] = int(float(row.get("count", 0)))
    return counts


def draw_flow_diagram(
    path: Path,
    *,
    title: str,
    rows: Sequence[Sequence[str]],
    notes: Sequence[str] = (),
    width: int = 1500,
    box_w: int = 250,
    box_h: int = 82,
) -> None:
    font_title = load_font(30)
    font = load_font(18)
    small = load_font(15)
    margin_x = 60
    margin_y = 72
    row_gap = 92
    col_gap = 42
    height = margin_y + len(rows) * (box_h + row_gap) + 110 + max(0, len(notes) - 1) * 24
    img = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.text((margin_x, 28), title, font=font_title, fill=(27, 33, 42))
    palette = [
        (226, 244, 241),
        (232, 243, 255),
        (255, 244, 225),
        (239, 239, 255),
        (235, 247, 229),
    ]
    outline = (78, 94, 112)
    for r_idx, row in enumerate(rows):
        n = len(row)
        total_w = n * box_w + (n - 1) * col_gap
        start_x = max(margin_x, (width - total_w) // 2)
        y = margin_y + r_idx * (box_h + row_gap)
        boxes = []
        for c_idx, text in enumerate(row):
            x = start_x + c_idx * (box_w + col_gap)
            fill = palette[(r_idx + c_idx) % len(palette)]
            draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=8, fill=fill, outline=outline, width=2)
            lines = wrap_text(text, font, box_w - 28)
            text_h = len(lines) * 22
            for i, line in enumerate(lines):
                draw.text((x + 14, y + (box_h - text_h) // 2 + i * 22), line, font=font, fill=(31, 39, 50))
            boxes.append((x, y, x + box_w, y + box_h))
        for a, b in zip(boxes, boxes[1:]):
            ax2 = a[2]
            ay = (a[1] + a[3]) // 2
            bx1 = b[0]
            draw.line((ax2 + 4, ay, bx1 - 14, ay), fill=(75, 91, 109), width=3)
            draw.polygon([(bx1 - 14, ay - 7), (bx1 - 14, ay + 7), (bx1 - 2, ay)], fill=(75, 91, 109))
        if r_idx < len(rows) - 1:
            src = boxes[-1]
            dst_n = len(rows[r_idx + 1])
            dst_total_w = dst_n * box_w + (dst_n - 1) * col_gap
            dst_start_x = max(margin_x, (width - dst_total_w) // 2)
            dst = (dst_start_x, y + box_h + row_gap, dst_start_x + box_w, y + box_h + row_gap + box_h)
            sx = (src[0] + src[2]) // 2
            sy = src[3] + 4
            dx = (dst[0] + dst[2]) // 2
            dy = dst[1] - 14
            mid_y = (sy + dy) // 2
            draw.line((sx, sy, sx, mid_y, dx, mid_y, dx, dy), fill=(75, 91, 109), width=3)
            draw.polygon([(dx - 7, dy), (dx + 7, dy), (dx, dy + 12)], fill=(75, 91, 109))
    note_y = height - 72 - max(0, len(notes) - 1) * 24
    for note in notes:
        draw.text((margin_x, note_y), note, font=small, fill=(84, 94, 108))
        note_y += 24
    img.save(path)


def wrap_text(text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        if font.getbbox(candidate)[2] <= max_w:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text]


def plot_candidate_source_mix() -> Path:
    before = read_source_counts(
        "data/GAIC_v2/Test500_SSTK_LocalQF_Smoke10_B1/artifacts/candidates/"
        "candidates_ar_gaic_v2_test500_localqf_smoke10_b1_overview_by_source.csv"
    )
    after = read_source_counts(
        "data/GAIC_v2/Test500_SSTK_LocalQF_C7_Smoke10_B1/artifacts/candidates/"
        "candidates_ar_gaic_v2_test500_localqf_c7_smoke10_b1_overview_by_source.csv"
    )
    all_sources = sorted(set(before) | set(after), key=lambda s: after.get(s, 0) + before.get(s, 0), reverse=True)[:14]
    x = np.arange(len(all_sources))
    plt.figure(figsize=(12, 5.2))
    plt.bar(x - 0.18, [before.get(s, 0) for s in all_sources], width=0.36, label="Before no C7", color="#8A7C72")
    plt.bar(x + 0.18, [after.get(s, 0) for s in all_sources], width=0.36, label="After C7", color="#238B76")
    plt.xticks(x, all_sources, rotation=35, ha="right", fontsize=8)
    plt.ylabel("candidate count")
    plt.title("Candidate Source Mix: no-C7 vs C7 paired GAIC smoke10")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    path = OUT_DIR / "candidate_source_mix_c7_before_after.png"
    savefig(path)
    return path


def plot_data_pipeline_throughput() -> Path:
    items = [
        ("baseline_w4p", "artifacts/mobilecropnet_v4/data_pipeline_bench/baseline_w4p_rows512.json"),
        ("precompute_w4p", "artifacts/mobilecropnet_v4/data_pipeline_bench/precompute_w4p_rows512.json"),
        ("precompute+cache", "artifacts/mobilecropnet_v4/data_pipeline_bench/precompute_cache_w4p_rows512.json"),
        ("remote precompute+cache", "artifacts/mobilecropnet_v4/data_pipeline_bench/remote_precompute_cache_w4p_rows512.json"),
    ]
    labels, values = [], []
    for label, path in items:
        obj = read_json(path)
        labels.append(label)
        values.append(float(obj["timing"]["samples_per_sec"]))
    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, values, color=["#7B8794", "#2F80ED", "#219653", "#56CCF2"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.015, f"{value:.0f}", ha="center", fontsize=9)
    plt.ylabel("samples/sec")
    plt.title("MobileCropNet v4 Dataset Loader Throughput")
    plt.grid(axis="y", alpha=0.25)
    path = OUT_DIR / "data_pipeline_throughput.png"
    savefig(path)
    return path


def find_run(summary: Dict[str, Any], name_part: str) -> Dict[str, Any]:
    for run in summary.get("runs", []):
        if name_part in str(run.get("run_name", "")):
            return run
    raise KeyError(name_part)


def plot_mobilecropnet_metrics() -> Path:
    next_summary = read_json("artifacts/mobilecropnet_v4/sstk_product_next_20260417-191315/summary/sstk_product_topreturn_summary.json")
    t6_summary = read_json("artifacts/mobilecropnet_v4/t6_corrected_v2b_product_20260417-191315/summary/sstk_product_topreturn_summary.json")
    r320 = find_run(next_summary, "r320-tr06-s19")
    hybrid = find_run(next_summary, "hybrid384-tr06-s17")
    r070 = find_run(next_summary, "r320-tr070-s17")
    t6 = find_run(t6_summary, "t6safe-r320-tr06-s18")
    teacher = {
        "official_pcc": r320["teacher_pcc"],
        "official_srcc": r320["teacher_srcc"],
        "official_acc1_top10": r320["teacher_acc1_top10"],
        "official_top1_mos": r320["teacher_top1_mos"],
        "official_regret": r320["teacher_regret"],
    }
    q24 = {
        "official_pcc": 0.399840,
        "official_srcc": 0.368395,
        "official_acc1_top10": 0.630,
        "official_top1_mos": 3.804520,
        "official_regret": 0.427380,
    }
    rows = [
        ("Teacher fullopt", teacher),
        ("q24 initial", q24),
        ("r320-tr06-s19", r320),
        ("hybrid384-tr06", hybrid),
        ("r320-tr070", r070),
        ("T6safe diagnostic", t6),
    ]
    names = [r[0] for r in rows]
    metrics = [
        ("PCC", "official_pcc"),
        ("SRCC", "official_srcc"),
        ("Acc1/10", "official_acc1_top10"),
        ("Top1 MOS", "official_top1_mos"),
        ("Regret", "official_regret"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(names))
    w = 0.23
    for idx, (label, key) in enumerate(metrics[:3]):
        axes[0].bar(x + (idx - 1) * w, [float(r[1][key]) for r in rows], width=w, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylim(0, 0.75)
    axes[0].set_title("Official GAIC ranking / hit metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - 0.15, [float(r[1]["official_top1_mos"]) for r in rows], width=0.3, label="Top1 MOS", color="#219653")
    axes[1].bar(x + 0.15, [float(r[1]["official_regret"]) for r in rows], width=0.3, label="Regret", color="#D64545")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_title("Top-return quality")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    path = OUT_DIR / "mobilecropnet_v4_official_metrics.png"
    savefig(path)
    return path


def plot_latency() -> Path:
    obj = read_json("artifacts/mobilecropnet_v4/inference_latency/a100_mcn_v4_primary_20260420_114321.json")
    keep = ["q24bal-tr06-s17", "turbobal-tr06-s17", "r320-tr06-s19", "r320-tr070-s17", "hybrid384-tr06-s17", "t6safe-r320-tr06-s18"]
    rows = []
    for result in obj["results"]:
        if result.get("name") not in keep or result.get("status") != "ok":
            continue
        bench = result["benchmarks"][0]
        rows.append((result["name"], result["parameter_count"] / 1_000_000, float(bench["mean_ms"])))
    rows.sort(key=lambda item: item[2])
    names = [r[0] for r in rows]
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(rows))
    ax1.bar(x - 0.18, [r[2] for r in rows], width=0.36, label="mean ms", color="#2F80ED")
    ax1.set_ylabel("mean ms / image")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, [r[1] for r in rows], marker="o", label="params M", color="#EB5757")
    ax2.set_ylabel("parameters (M)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax1.set_title("A100 forward latency and model size")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    path = OUT_DIR / "mobilecropnet_v4_latency_a100.png"
    savefig(path)
    return path


def plot_public_benchmark() -> Tuple[Path, Path]:
    obj = read_json("artifacts/public_benchmark_eval/full_public_sanity_baseline_20260415/mode_SC/public_benchmark_summary.json")
    datasets = obj["datasets"]
    names = ["fcdb", "cpc", "gnmc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, [datasets[n]["n_images"] for n in names], width=0.36, label="images", color="#2D9CDB")
    axes[0].bar(x + 0.18, [datasets[n]["n_tasks"] for n in names], width=0.36, label="tasks", color="#27AE60")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.upper() for n in names])
    axes[0].set_yscale("log")
    axes[0].set_title("Public Benchmark Coverage")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [datasets[n].get("n_pairwise", 0) for n in names], color="#7B8794")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.upper() for n in names])
    axes[1].set_yscale("log")
    axes[1].set_title("Pairwise units")
    axes[1].grid(axis="y", alpha=0.25)
    coverage_path = OUT_DIR / "public_benchmark_coverage.png"
    savefig(coverage_path)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    metrics = ["iou_top1", "gt_rank_at_1", "gt_rank_at_5", "coverage_at_09"]
    width = 0.2
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [float(datasets[n].get(metric) or 0.0) for n in names], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_title("Public Benchmark Quality Metrics")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    quality_path = OUT_DIR / "public_benchmark_quality.png"
    savefig(quality_path)
    return coverage_path, quality_path


def plot_t6_conversion() -> Path:
    paths = [
        "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_t6_score_only_corrected_v2b/conversion_summary.json",
        "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_t6_score_only_corrected_v2b/conversion_summary.json",
        "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_t6_score_only_corrected_v2b/conversion_summary.json",
    ]
    summaries = [read_json(p) for p in paths]
    splits = [s["split"] for s in summaries]
    keys = ["candidate", "converter_unsafe_candidates", "low_weight_high_score", "low_weight_positive", "main_positive", "hard_negative"]
    colors = ["#B9C0C9", "#D64545", "#F2C94C", "#56CCF2", "#27AE60", "#8E6C8A"]
    x = np.arange(len(splits))
    bottom = np.zeros(len(splits))
    plt.figure(figsize=(10.5, 5))
    for key, color in zip(keys, colors):
        vals = np.array([s["bucket_counts"].get(key, 0) for s in summaries])
        plt.bar(x, vals, bottom=bottom, label=key, color=color)
        bottom += vals
    plt.xticks(x, [s.capitalize() for s in splits])
    plt.yscale("log")
    plt.ylabel("candidate/bucket count (log)")
    plt.title("T6 corrected v2b MobileCropNet label conversion buckets")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.25)
    path = OUT_DIR / "t6_label_conversion_buckets.png"
    savefig(path)
    return path


def write_summary_json(paths: Dict[str, str]) -> Path:
    c7_manifest = read_json("Implement_Docs/assets_post_20260317_subject_c7/subject_c7_before_after_manifest.json")
    public = read_json("artifacts/public_benchmark_eval/full_public_sanity_baseline_20260415/mode_SC/public_benchmark_summary.json")
    throughput = {
        name: read_json(path)["timing"]["samples_per_sec"]
        for name, path in {
            "baseline_w4p": "artifacts/mobilecropnet_v4/data_pipeline_bench/baseline_w4p_rows512.json",
            "precompute_cache_w4p": "artifacts/mobilecropnet_v4/data_pipeline_bench/precompute_cache_w4p_rows512.json",
            "remote_precompute_cache_w4p": "artifacts/mobilecropnet_v4/data_pipeline_bench/remote_precompute_cache_w4p_rows512.json",
        }.items()
    }
    summary = {
        "assets": paths,
        "c7_visual_audit": {
            "num_samples": c7_manifest["num_samples"],
            "all_resolved_by_decision_chosen_candidate": all(
                r["before_selected_resolution"] == "decision.chosen_candidate_id"
                and r["after_selected_resolution"] == "decision.chosen_candidate_id"
                for r in c7_manifest["records"]
            ),
            "before_best_differs_from_chosen": sum(
                r["before_best_candidate_id"] != r["before_chosen_candidate_id"] for r in c7_manifest["records"]
            ),
            "after_best_differs_from_chosen": sum(
                r["after_best_candidate_id"] != r["after_chosen_candidate_id"] for r in c7_manifest["records"]
            ),
        },
        "public_benchmark_combined": public["combined"],
        "data_pipeline_samples_per_sec": throughput,
    }
    out = OUT_DIR / "post20260317_report_asset_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    draw_flow_diagram(
        OUT_DIR / "data_factory_pipeline_overview.png",
        title="SSTK Data Factory: current post-2026-03-17 implementation",
        rows=[
            ["Filter / Prepare", "C1-C6 Precompute", "C7 Support Map", "Subject Routing"],
            ["Saliency-Semantic Attribution", "Effective Subject Region", "Crop Guidance Spec", "Candidate Bank"],
            ["Teacher Scoring", "FinalScore Labels", "Public / GAIC Eval", "MobileCropNet v4"],
        ],
        notes=[
            "C7/crop guidance changes flow downstream into candidates, scorer, labels, and model evaluation.",
            "Teacher output is supervision and benchmark evidence, not the runtime topology of MobileCropNet v4.",
        ],
    )
    paths["data_factory_pipeline_overview"] = str(OUT_DIR / "data_factory_pipeline_overview.png")

    draw_flow_diagram(
        OUT_DIR / "mobilecropnet_v4_architecture.png",
        title="MobileCropNet v4: learned cropper architecture",
        rows=[
            ["Image Backbone", "Global Token + AR Embedding", "Learned Proposal Generator"],
            ["Candidate Box Pooling", "Relation-Lite / Set Ranker", "Utility + Positive + Risk"],
            ["Macro + Checklist + Why Tags", "Route + Decision + Delta", "Selected Crop / Explanations"],
        ],
        notes=[
            "Runtime does not mirror the teacher tree; teacher/data-factory artifacts are used as supervision.",
            "Fixed-AR proposals preserve target AR, while FREE proposals remain 4D boxes.",
        ],
    )
    paths["mobilecropnet_v4_architecture"] = str(OUT_DIR / "mobilecropnet_v4_architecture.png")

    draw_flow_diagram(
        OUT_DIR / "mobilecropnet_v4_batch_schema.png",
        title="MobileCropNet v4 dataset adapter: batch tensor groups",
        rows=[
            ["Image + Letterbox", "Candidate Boxes + Geometry", "Score / Positive / Risk Targets"],
            ["Pairwise + Listwise", "Macro + Checklist + Why Tags", "Decision + Route + Delta"],
            ["Positive Box Bag", "Proposal Targets", "Metadata / IDs"],
        ],
        notes=[
            "Adapter reads existing FinalScore / conditional-DETR JSONL; no separate v4 physical JSON is required first.",
            "Hard/unsafe/overflow candidates are separated from the main positive supervision pool.",
        ],
    )
    paths["mobilecropnet_v4_batch_schema"] = str(OUT_DIR / "mobilecropnet_v4_batch_schema.png")

    generated = {
        "candidate_source_mix": plot_candidate_source_mix(),
        "data_pipeline_throughput": plot_data_pipeline_throughput(),
        "mobilecropnet_v4_official_metrics": plot_mobilecropnet_metrics(),
        "mobilecropnet_v4_latency_a100": plot_latency(),
        "t6_label_conversion_buckets": plot_t6_conversion(),
    }
    public_cov, public_quality = plot_public_benchmark()
    generated["public_benchmark_coverage"] = public_cov
    generated["public_benchmark_quality"] = public_quality
    for key, path in generated.items():
        paths[key] = str(path)

    summary = write_summary_json(paths)
    print(json.dumps({"status": "ok", "asset_dir": str(OUT_DIR), "summary": str(summary), "asset_count": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
