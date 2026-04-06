#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_gaic_training_label_debug_viz import build_image_index_from_coco, invert_vocab, load_gaic_gt_index, select_balanced_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged crop-score priority experiments with benchmark/debug-viz comparison.")
    parser.add_argument(
        "--teacher_jsonl",
        default="data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_gaic_260402_reroute_v1.jsonl",
    )
    parser.add_argument(
        "--candidates_jsonl",
        default="data/GAIC/All/artifacts/candidates/candidates_ar_gaic_260402_reroute_v1.jsonl",
    )
    parser.add_argument(
        "--features_jsonl",
        default="data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl",
    )
    parser.add_argument("--gaic_train_json", default="data/Publics/GAIC/annotations_json/instances_train.json")
    parser.add_argument("--gaic_test_json", default="data/Publics/GAIC/annotations_json/instances_test.json")
    parser.add_argument("--image_root", default="data/GAIC/All/images")
    parser.add_argument("--c1_jsonl", default="data/GAIC/All/artifacts/precompute/feats_c1.jsonl")
    parser.add_argument(
        "--teacher_max_images",
        type=int,
        default=0,
        help="Optional smoke-mode cap on unique image_id rows kept from teacher_jsonl before building training labels/benchmark.",
    )
    parser.add_argument(
        "--out_root",
        default="data/GAIC/All/artifacts/reports/crop_score_priority_experiments_gaic_260402_reroute_v1",
    )
    parser.add_argument("--safe_leftover_policy", default="ignore")
    parser.add_argument("--softmax_tau", type=float, default=0.2)
    parser.add_argument("--listwise_top_pos", type=int, default=5)
    parser.add_argument("--listwise_neg", type=int, default=15)
    parser.add_argument("--pair_margin_min", type=float, default=0.01)
    parser.add_argument("--hard_negative_rank_pct_max", type=float, default=0.2)
    parser.add_argument("--near_margin_max", type=float, default=0.05)
    parser.add_argument("--max_hard_pairs", type=int, default=4)
    parser.add_argument("--max_near_pairs", type=int, default=4)
    parser.add_argument("--benchmark_top_quantile", type=float, default=0.8)
    parser.add_argument("--benchmark_softmax_tau", type=float, default=0.2)
    parser.add_argument("--benchmark_ge_full_topk", type=int, default=10)
    parser.add_argument("--benchmark_sample_count", type=int, default=0)
    parser.add_argument("--benchmark_max_images", type=int, default=24)
    parser.add_argument("--winner_benchmark_max_images", type=int, default=48)
    parser.add_argument("--run_full_expensive", type=int, default=1)
    parser.add_argument("--run_winner_refresh", type=int, default=1)
    parser.add_argument("--debug_sample_size", type=int, default=50)
    parser.add_argument(
        "--debug_sample_size_all",
        type=int,
        default=0,
        help="Optional reduced sample size for non-winner experiments when --build_debug_viz_for_all 1.",
    )
    parser.add_argument("--debug_seed", type=int, default=42)
    parser.add_argument("--show_why_text", type=int, default=0)
    parser.add_argument("--debug_viz_format", choices=("auto", "png", "jpg"), default="jpg")
    parser.add_argument("--debug_viz_jpg_quality", type=int, default=50)
    parser.add_argument("--debug_skip_by_ar", type=int, default=0)
    parser.add_argument("--build_debug_viz_for_all", type=int, default=0)
    parser.add_argument("--skip_debug_viz", type=int, default=0)
    parser.add_argument("--skip_gt_cache", type=int, default=0)
    parser.add_argument(
        "--experiment_ids",
        nargs="*",
        default=[],
        help="Optional subset of experiment_id values to run. Useful for smoke validation.",
    )
    parser.add_argument("--parallel_experiment_jobs", type=int, default=1)
    parser.add_argument(
        "--worker_cpu_threads",
        type=int,
        default=-1,
        help="Per-subprocess BLAS/OpenMP thread cap. -1 or omitted = auto-resolve from available CPU cores and concurrent workers.",
    )
    parser.add_argument("--benchmark_gpu_ids", default="")
    parser.add_argument("--benchmark_num_workers", type=int, default=0)
    parser.add_argument("--gt_cache_gpu_ids", default="")
    parser.add_argument("--gt_cache_num_workers", type=int, default=0)
    parser.add_argument("--force", type=int, default=0)
    return parser.parse_args()


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    stage: int
    label: str
    description: str
    profile_name: str
    overrides: Dict[str, Any]
    stage_family: str


EXPERIMENT_SPECS: List[ExperimentSpec] = [
    ExperimentSpec(
        experiment_id="baseline_current_refined",
        stage=0,
        label="Baseline Current Refined",
        description="기존 refined monotonic scorer baseline.",
        profile_name="current_refined",
        overrides={},
        stage_family="baseline",
    ),
    ExperimentSpec(
        experiment_id="stage1_balanced",
        stage=1,
        label="Stage1 Balanced",
        description="1순위 항목만 사용한 기본형.",
        profile_name="priority_stage1",
        overrides={},
        stage_family="stage1",
    ),
    ExperimentSpec(
        experiment_id="stage1_subject_heavy",
        stage=1,
        label="Stage1 Subject Heavy",
        description="S축과 border/softcut을 더 강하게 보는 변형.",
        profile_name="priority_stage1",
        overrides={
            "rank_weight_s": 1.35,
            "rank_weight_c": 0.80,
            "policy_area_weight_scale": 0.85,
            "s_macro_border_weight_scale": 1.15,
            "s_macro_softcut_weight_scale": 1.15,
            "s_macro_cov_weight_scale": 1.10,
        },
        stage_family="stage1",
    ),
    ExperimentSpec(
        experiment_id="stage1_framing_guard",
        stage=1,
        label="Stage1 Framing Guard",
        description="C_place/headroom/lookroom/context를 약간 더 강하게 보는 변형.",
        profile_name="priority_stage1",
        overrides={
            "rank_weight_s": 1.10,
            "rank_weight_c": 1.05,
            "policy_area_weight_scale": 0.85,
            "c_macro_place_weight_scale": 1.75,
            "c_macro_headroom_weight_scale": 1.10,
            "c_macro_lookroom_weight_scale": 1.10,
            "c_macro_context_weight_scale": 1.20,
        },
        stage_family="stage1",
    ),
    ExperimentSpec(
        experiment_id="stage2_balanced",
        stage=2,
        label="Stage2 Balanced",
        description="Stage1 + A/T 기본형.",
        profile_name="priority_stage2",
        overrides={},
        stage_family="stage2",
    ),
    ExperimentSpec(
        experiment_id="stage2_aesthetic_heavy",
        stage=2,
        label="Stage2 Aesthetic Heavy",
        description="2순위 확장에서 aesthetic 비중을 더 높인 변형.",
        profile_name="priority_stage2",
        overrides={
            "rank_weight_a": 0.95,
            "rank_weight_t": 0.08,
            "a_macro_aesthetic_weight": 0.82,
            "a_macro_align_weight": 0.18,
        },
        stage_family="stage2",
    ),
    ExperimentSpec(
        experiment_id="stage2_teacher_tiebreak",
        stage=2,
        label="Stage2 Teacher Tiebreak",
        description="teacher prior를 조금 더 강하게 쓰는 변형.",
        profile_name="priority_stage2",
        overrides={
            "rank_weight_a": 0.75,
            "rank_weight_t": 0.18,
            "a_macro_aesthetic_weight": 0.78,
            "a_macro_align_weight": 0.22,
        },
        stage_family="stage2",
    ),
    ExperimentSpec(
        experiment_id="stage3_balanced",
        stage=3,
        label="Stage3 Balanced",
        description="Stage2 + 3순위 composition 확장 기본형.",
        profile_name="priority_stage3",
        overrides={},
        stage_family="stage3",
    ),
    ExperimentSpec(
        experiment_id="stage3_comp_light",
        stage=3,
        label="Stage3 Comp Light",
        description="3순위 composition 항목의 개입을 줄인 변형.",
        profile_name="priority_stage3",
        overrides={
            "c_macro_comp_weight_scale": 0.40,
            "c_macro_horizon_weight_scale": 0.45,
            "c_macro_sym_weight_scale": 0.35,
        },
        stage_family="stage3",
    ),
    ExperimentSpec(
        experiment_id="stage3_comp_heavy",
        stage=3,
        label="Stage3 Comp Heavy",
        description="3순위 composition 항목의 개입을 늘린 변형.",
        profile_name="priority_stage3",
        overrides={
            "rank_weight_c": 1.05,
            "c_macro_comp_weight_scale": 0.75,
            "c_macro_horizon_weight_scale": 0.75,
            "c_macro_sym_weight_scale": 0.65,
            "horizon_conf_thr": 0.20,
            "horizon_visibility_thr": 0.60,
        },
        stage_family="stage3",
    ),
    ExperimentSpec(
        experiment_id="stage4_balanced",
        stage=4,
        label="Stage4 Balanced",
        description="Stage3 + copyspace 기본형.",
        profile_name="priority_stage4",
        overrides={},
        stage_family="stage4",
    ),
    ExperimentSpec(
        experiment_id="stage4_copyspace_light",
        stage=4,
        label="Stage4 Copyspace Light",
        description="copyspace 가중치를 낮춘 변형.",
        profile_name="priority_stage4",
        overrides={
            "c_macro_copyspace_weight_scale": 0.20,
        },
        stage_family="stage4",
    ),
    ExperimentSpec(
        experiment_id="stage4_copyspace_heavy",
        stage=4,
        label="Stage4 Copyspace Heavy",
        description="copyspace 가중치를 조금 더 높인 변형.",
        profile_name="priority_stage4",
        overrides={
            "c_macro_copyspace_weight_scale": 0.50,
        },
        stage_family="stage4",
    ),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
) -> None:
    ensure_dir(log_path.parent)
    proc_env = os.environ.copy()
    if env:
        proc_env.update({str(key): str(value) for key, value in env.items()})
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=proc_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)} | log={log_path}")


def round9(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 9)
    except (TypeError, ValueError):
        return None


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def stable_shard_index(text: str, num_shards: int) -> int:
    n = max(1, int(num_shards))
    if n == 1:
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % n


def parse_gpu_ids_csv(csv_text: str) -> List[str]:
    text = str(csv_text or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def detect_all_gpu_ids() -> List[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return [str(idx) for idx in range(len(lines))]


def resolve_gpu_ids(csv_text: str) -> List[str]:
    parsed = parse_gpu_ids_csv(csv_text)
    if parsed:
        return parsed
    visible = parse_gpu_ids_csv(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if visible:
        return visible
    return detect_all_gpu_ids()


def resolve_worker_count(gpu_ids: Sequence[str], requested_workers: int) -> int:
    if int(requested_workers) > 0:
        return int(requested_workers)
    if gpu_ids:
        return len(gpu_ids)
    return 1


def capped_thread_env(cpu_threads: int, *, gpu_id: Optional[str] = None) -> Dict[str, str]:
    threads = max(1, int(cpu_threads))
    env = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }
    if gpu_id is not None and str(gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id).strip()
    return env


def available_cpu_cores() -> int:
    return max(1, int(os.cpu_count() or 1))


def resolve_worker_cpu_threads(requested_threads: int, *, concurrent_workers: int) -> int:
    requested = int(requested_threads)
    if requested > 0:
        return requested
    workers = max(1, int(concurrent_workers))
    return max(1, available_cpu_cores() // workers)


def merge_jsonl_unique(input_paths: Sequence[Path], output_path: Path, *, key_field: str) -> int:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in input_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get(key_field, "")).strip()
                if key:
                    merged[key] = row
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for key in sorted(merged.keys()):
            handle.write(json.dumps(merged[key], ensure_ascii=False) + "\n")
    return len(merged)


def benchmark_objective(summary: Dict[str, Any], qa_summary: Dict[str, Any], validation_summary: Dict[str, Any]) -> float:
    ge_policy = safe_dict(safe_dict(summary.get("Ge", {})).get("trend_by_field", {})).get("score_policy", {})
    prod_policy = safe_dict(safe_dict(summary.get("Ge", {})).get("prod_selection_policy", {}))
    consistency = safe_dict(qa_summary.get("consistency_audit"))
    validation_ok = 1.0 if str(validation_summary.get("status", "")).lower() == "ok" else 0.0

    objective = 0.0
    objective += 0.35 * safe_float(safe_dict(ge_policy).get("mean_spearman"), 0.0)
    objective += 0.15 * safe_float(safe_dict(ge_policy).get("mean_topq_jaccard"), 0.0)
    objective += 0.20 * safe_float(prod_policy.get("gt_best_iou_mean"), 0.0)
    objective += 0.20 * safe_float(prod_policy.get("matched_gt_mos_percentile_mean"), 0.0)
    objective += 0.10 * safe_float(prod_policy.get("matched_gt_topq_rate"), 0.0)
    objective += 0.05 * validation_ok

    if int(consistency.get("rows_with_main_pool_monotonicity_violation", 0)) > 0:
        objective -= 0.50
    if int(consistency.get("rows_with_higher_scored_safe_pool_candidates", 0)) > 0:
        objective -= 0.35
    objective -= min(0.15, 0.0002 * int(consistency.get("rows_with_higher_scored_unsafe_pool_candidates", 0)))
    objective -= min(0.10, 0.00005 * int(consistency.get("overflow_candidate_total", 0)))
    return round(objective, 9)


def collect_experiment_metrics(training_dir: Path, benchmark_dir: Path) -> Dict[str, Any]:
    qa_summary = load_json(training_dir / "qa_summary.json")
    validation_summary = load_json(training_dir / "validation_summary.json")
    benchmark_summary = load_json(benchmark_dir / "benchmark_summary.json")
    consistency = safe_dict(qa_summary.get("consistency_audit"))
    ge_policy = safe_dict(safe_dict(benchmark_summary.get("Ge", {})).get("trend_by_field", {})).get("score_policy", {})
    ge_prod_policy = safe_dict(safe_dict(benchmark_summary.get("Ge", {})).get("prod_selection_policy", {}))
    return {
        "objective": benchmark_objective(benchmark_summary, qa_summary, validation_summary),
        "validation_status": str(validation_summary.get("status", "")),
        "benchmark_overlap_image_count": int(safe_dict(benchmark_summary.get("dataset", {})).get("overlap_image_count", 0)),
        "mean_spearman": round9(safe_dict(ge_policy).get("mean_spearman")),
        "mean_topq_jaccard": round9(safe_dict(ge_policy).get("mean_topq_jaccard")),
        "prod_gt_best_iou_mean": round9(ge_prod_policy.get("gt_best_iou_mean")),
        "prod_gt_mos_percentile_mean": round9(ge_prod_policy.get("matched_gt_mos_percentile_mean")),
        "prod_gt_topq_rate": round9(ge_prod_policy.get("matched_gt_topq_rate")),
        "rows_with_main_pool_monotonicity_violation": int(consistency.get("rows_with_main_pool_monotonicity_violation", 0)),
        "rows_with_higher_scored_safe_pool_candidates": int(consistency.get("rows_with_higher_scored_safe_pool_candidates", 0)),
        "rows_with_higher_scored_unsafe_pool_candidates": int(consistency.get("rows_with_higher_scored_unsafe_pool_candidates", 0)),
        "rows_with_higher_scored_overflow_candidates": int(consistency.get("rows_with_higher_scored_overflow_candidates", 0)),
        "overflow_candidate_total": int(consistency.get("overflow_candidate_total", 0)),
        "weak_positive_pruned_total": int(consistency.get("weak_positive_pruned_total", 0)),
    }


def write_selected_images_csv(path: Path, image_ids: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id"])
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow({"image_id": image_id})


def build_teacher_subset_jsonl(src_path: Path, out_path: Path, max_images: int) -> Path:
    if max_images <= 0:
        return src_path
    ensure_dir(out_path.parent)
    kept_ids: set[str] = set()
    kept_rows: List[str] = []
    with src_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue
            if image_id not in kept_ids and len(kept_ids) >= max_images:
                continue
            kept_ids.add(image_id)
            kept_rows.append(json.dumps(row, ensure_ascii=False))
    out_path.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")
    return out_path


def select_debug_image_ids(coco_json: Path, image_root: Path, subject_mode_vocab: Path, gt_paths: Sequence[Path], sample_size: int, seed: int) -> List[str]:
    mode_names = invert_vocab(subject_mode_vocab)
    image_rows, _ = build_image_index_from_coco(coco_json, image_root, mode_names)
    gt_index = load_gaic_gt_index(gt_paths)
    eligible_rows = [row for row in image_rows if str(row.get("image_id", "")) in gt_index]
    selection_rows = eligible_rows if eligible_rows else image_rows
    selected_rows = select_balanced_images(selection_rows, sample_size=sample_size, seed=seed)
    return [str(row.get("image_id", "")) for row in selected_rows]


def experiment_report_markdown(
    *,
    args: argparse.Namespace,
    experiment_rows: Sequence[Dict[str, Any]],
    stage_winners: Dict[str, Dict[str, Any]],
) -> str:
    benchmark_gpu_ids = resolve_gpu_ids(str(args.benchmark_gpu_ids))
    benchmark_workers = resolve_worker_count(benchmark_gpu_ids, int(args.benchmark_num_workers))
    gt_cache_gpu_ids = resolve_gpu_ids(str(args.gt_cache_gpu_ids))
    gt_cache_workers = resolve_worker_count(gt_cache_gpu_ids, int(args.gt_cache_num_workers))
    resolved_threads_experiment = resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=max(1, int(args.parallel_experiment_jobs)))
    resolved_threads_benchmark = resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=benchmark_workers)
    resolved_threads_gt = resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=gt_cache_workers)
    lines: List[str] = []
    lines.append("# Crop Score Priority Experiment Report")
    lines.append("")
    lines.append(f"- 생성 시각: `{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())}`")
    lines.append(f"- teacher_jsonl: `{args.teacher_jsonl}`")
    lines.append(f"- candidates_jsonl: `{args.candidates_jsonl}`")
    lines.append(f"- features_jsonl: `{args.features_jsonl}`")
    lines.append(f"- benchmark max_images: `{args.benchmark_max_images}`")
    lines.append(f"- winner refresh max_images: `{args.winner_benchmark_max_images}`")
    lines.append(f"- full expensive: `{bool(int(args.run_full_expensive) > 0)}`")
    lines.append(f"- parallel experiment jobs: `{int(args.parallel_experiment_jobs)}`")
    lines.append(f"- worker cpu threads requested: `{int(args.worker_cpu_threads)}`")
    lines.append(f"- worker cpu threads resolved: experiment=`{resolved_threads_experiment}` benchmark=`{resolved_threads_benchmark}` gt_cache=`{resolved_threads_gt}`")
    lines.append(f"- debug viz render: format=`{args.debug_viz_format}` jpg_quality=`{int(args.debug_viz_jpg_quality)}`")
    lines.append(f"- debug viz sample size: winners/baseline=`{int(args.debug_sample_size)}` all-others=`{int(args.debug_sample_size_all) or int(args.debug_sample_size)}` skip_by_ar=`{bool(int(args.debug_skip_by_ar) > 0)}`")
    lines.append(f"- benchmark shard workers: `{benchmark_workers}` gpu_ids=`{args.benchmark_gpu_ids or 'auto'}`")
    lines.append(f"- GT cache shard workers: `{gt_cache_workers}` gpu_ids=`{args.gt_cache_gpu_ids or 'auto'}`")
    lines.append("")
    lines.append("## Stage Winners")
    lines.append("")
    for stage_key in ("stage1", "stage2", "stage3", "stage4"):
        winner = stage_winners.get(stage_key)
        if not winner:
            continue
        lines.append(
            f"- `{stage_key}` winner: `{winner['experiment_id']}` "
            f"(objective=`{winner['metrics']['objective']:.4f}`, "
            f"spearman=`{winner['metrics']['mean_spearman']}`, "
            f"IoU=`{winner['metrics']['prod_gt_best_iou_mean']}`, "
            f"MOS pct=`{winner['metrics']['prod_gt_mos_percentile_mean']}`)"
        )
    lines.append("")
    lines.append("## All Experiments")
    lines.append("")
    headers = [
        "experiment_id",
        "stage_family",
        "objective",
        "spearman",
        "topq_jaccard",
        "gt_best_iou",
        "gt_mos_pct",
        "topq_rate",
        "main_pool_violation",
        "higher_safe",
        "overflow_total",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in experiment_rows:
        metrics = safe_dict(row.get("metrics"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment_id", "")),
                    str(row.get("stage_family", "")),
                    f"{safe_float(metrics.get('objective'), 0.0):.4f}",
                    str(metrics.get("mean_spearman")),
                    str(metrics.get("mean_topq_jaccard")),
                    str(metrics.get("prod_gt_best_iou_mean")),
                    str(metrics.get("prod_gt_mos_percentile_mean")),
                    str(metrics.get("prod_gt_topq_rate")),
                    str(metrics.get("rows_with_main_pool_monotonicity_violation")),
                    str(metrics.get("rows_with_higher_scored_safe_pool_candidates")),
                    str(metrics.get("overflow_candidate_total")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `objective`는 `Ge.score_policy` trend + production-policy GT matching 지표를 합산하고, monotonic/safe-pool 위반에 penalty를 준 값입니다.")
    lines.append("- `baseline_current_refined`는 비교 기준이며 stage winner 선발에는 stage1~4만 사용합니다.")
    lines.append("- debug viz는 기본적으로 baseline + 각 stage winner에 대해 생성됩니다. `--build_debug_viz_for_all 1`이면 모든 실험에 대해 생성합니다.")
    return "\n".join(lines).rstrip() + "\n"


def build_python_cmd(script_rel: str, *args: str) -> List[str]:
    return [sys.executable, str(PROJECT_ROOT / script_rel), *args]


def resolve_input_path(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / raw).resolve()


def benchmark_runtime_args() -> List[str]:
    return []


def benchmark_expensive_runtime_args(args: argparse.Namespace) -> List[str]:
    return [
        "--align_device",
        "auto",
        "--aesthetic_device",
        "auto",
        "--exp_batch_size",
        "24",
        "--exp_preprocess_workers",
        "0",
        "--exp_pin_memory",
        "1",
    ]


def precompute_shared_full_expensive_cache(
    *,
    args: argparse.Namespace,
    teacher_jsonl_path: Path,
    gt_paths: Sequence[Path],
    out_root: Path,
    shared_full_expensive_cache: Path,
    logs_dir: Path,
) -> None:
    if int(args.run_full_expensive) <= 0:
        return
    gpu_ids = resolve_gpu_ids(str(args.benchmark_gpu_ids))
    workers = resolve_worker_count(gpu_ids, int(args.benchmark_num_workers))
    worker_cpu_threads = resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=workers)
    precompute_max_images = max(int(args.benchmark_max_images), int(args.winner_benchmark_max_images) if int(args.run_winner_refresh) > 0 else 0)
    if workers <= 1:
        return
    if shared_full_expensive_cache.exists() and not bool(int(args.force) > 0):
        return

    shard_dir = out_root / "shared_full_expensive_cache_shards"
    ensure_dir(shard_dir)
    shard_cache_paths: List[Path] = []
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for shard_index in range(workers):
            shard_cache = shard_dir / f"full_expensive_cache_shard_{shard_index}.jsonl"
            shard_cache_paths.append(shard_cache)
            gpu_id = gpu_ids[shard_index % len(gpu_ids)] if gpu_ids else None
            env = capped_thread_env(worker_cpu_threads, gpu_id=gpu_id)
            cmd = build_python_cmd(
                "src/scripts/run_gaic_benchmark_eval.py",
                "--candidates_jsonl",
                str(resolve_input_path(args.candidates_jsonl)),
                "--features_jsonl",
                str(resolve_input_path(args.features_jsonl)),
                "--teacher_jsonl",
                str(teacher_jsonl_path),
                "--training_label_dir",
                str(out_root),
                "--gaic_train_json",
                str(gt_paths[0]),
                "--gaic_test_json",
                str(gt_paths[1]),
                "--image_dir",
                str(resolve_input_path(args.image_root)),
                "--output_dir",
                str(out_root / f"_tmp_benchmark_cache_precompute_{shard_index}"),
                "--c1_jsonl",
                str(resolve_input_path(args.c1_jsonl)),
                "--softmax_tau",
                str(args.benchmark_softmax_tau),
                "--max_images",
                str(precompute_max_images),
                "--run_full_expensive",
                "1",
                "--precompute_full_expensive_cache_only",
                "1",
                "--full_expensive_cache_jsonl",
                str(shard_cache),
                "--num_shards",
                str(workers),
                "--shard_index",
                str(shard_index),
                *benchmark_expensive_runtime_args(args),
            )
            futures.append(
                executor.submit(
                    run_command,
                    cmd,
                    cwd=PROJECT_ROOT,
                    log_path=logs_dir / f"shared_full_expensive_cache_shard_{shard_index}.log",
                    env=env,
                )
            )
        for future in as_completed(futures):
            future.result()
    merge_jsonl_unique(shard_cache_paths, shared_full_expensive_cache, key_field="cache_key")


def run_experiment_pipeline(
    *,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    teacher_jsonl_path: Path,
    gt_paths: Sequence[Path],
    out_root: Path,
    logs_dir: Path,
    shared_full_expensive_cache: Path,
) -> Dict[str, Any]:
    benchmark_gpu_ids = resolve_gpu_ids(str(args.benchmark_gpu_ids))
    experiment_cpu_threads = resolve_worker_cpu_threads(
        int(args.worker_cpu_threads),
        concurrent_workers=max(1, int(args.parallel_experiment_jobs)),
    )
    benchmark_env = capped_thread_env(
        resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=1),
        gpu_id=benchmark_gpu_ids[0] if benchmark_gpu_ids else None,
    )
    experiment_root = out_root / spec.experiment_id
    training_dir = experiment_root / "training_labels"
    benchmark_dir = experiment_root / "benchmark"
    coco_dir = training_dir / "coco"
    override_json = experiment_root / "score_profile_overrides.json"
    write_json(override_json, spec.overrides)

    benchmark_max_images = int(args.benchmark_max_images)
    if spec.stage == 0:
        benchmark_max_images = max(benchmark_max_images, int(args.benchmark_max_images))

    need_run = bool(int(args.force) > 0) or not (benchmark_dir / "benchmark_summary.json").exists()
    if need_run:
        ensure_dir(experiment_root)
        run_command(
            build_python_cmd(
                "src/scripts/build_finalscore_training_data.py",
                "--teacher_scores_jsonl",
                str(teacher_jsonl_path),
                "--out_dir",
                str(training_dir),
                "--image_root",
                str(resolve_input_path(args.image_root)),
                "--softmax_tau",
                str(args.softmax_tau),
                "--listwise_top_pos",
                str(args.listwise_top_pos),
                "--listwise_neg",
                str(args.listwise_neg),
                "--pair_margin_min",
                str(args.pair_margin_min),
                "--hard_negative_rank_pct_max",
                str(args.hard_negative_rank_pct_max),
                "--near_margin_max",
                str(args.near_margin_max),
                "--max_hard_pairs",
                str(args.max_hard_pairs),
                "--max_near_pairs",
                str(args.max_near_pairs),
                "--safe_leftover_policy",
                str(args.safe_leftover_policy),
                "--report_examples",
                "0",
                "--score_profile",
                spec.profile_name,
                "--score_profile_overrides_json",
                str(override_json),
            ),
            cwd=PROJECT_ROOT,
            log_path=logs_dir / f"{spec.experiment_id}_build_training_labels.log",
            env=capped_thread_env(experiment_cpu_threads),
        )
        run_command(
            build_python_cmd(
                "src/scripts/convert_sstk_detr_batch_to_gaic_like.py",
                "--batch_jsonl",
                str(training_dir / "train_conditional_detr_batch.jsonl"),
                "--gaic_reference_json",
                str(gt_paths[0]),
                "--gaic_train_reference_json",
                str(gt_paths[0]),
                "--gaic_test_reference_json",
                str(gt_paths[1]),
                "--out_json",
                str(coco_dir / "instances_conditional_detr_batch_gaic_like.json"),
                "--out_summary_json",
                str(coco_dir / "gaic_like_conversion_summary.json"),
                "--out_guide_md",
                str(coco_dir / "GAIC_LIKE_FORMAT_GUIDE.md"),
                "--image_root",
                str(resolve_input_path(args.image_root)),
            ),
            cwd=PROJECT_ROOT,
            log_path=logs_dir / f"{spec.experiment_id}_convert_gaic_like.log",
            env=capped_thread_env(experiment_cpu_threads),
        )
        run_command(
            build_python_cmd(
                "src/scripts/run_gaic_benchmark_eval.py",
                "--candidates_jsonl",
                str(resolve_input_path(args.candidates_jsonl)),
                "--features_jsonl",
                str(resolve_input_path(args.features_jsonl)),
                "--teacher_jsonl",
                str(teacher_jsonl_path),
                "--training_label_dir",
                str(training_dir),
                "--gaic_train_json",
                str(gt_paths[0]),
                "--gaic_test_json",
                str(gt_paths[1]),
                "--image_dir",
                str(resolve_input_path(args.image_root)),
                "--output_dir",
                str(benchmark_dir),
                "--c1_jsonl",
                str(resolve_input_path(args.c1_jsonl)),
                "--top_quantile",
                str(args.benchmark_top_quantile),
                "--softmax_tau",
                str(args.benchmark_softmax_tau),
                "--sample_count",
                str(args.benchmark_sample_count),
                "--ge_full_topk",
                str(args.benchmark_ge_full_topk),
                "--max_images",
                str(benchmark_max_images),
                "--score_profile",
                spec.profile_name,
                "--score_profile_overrides_json",
                str(override_json),
                "--run_full_expensive",
                str(int(args.run_full_expensive)),
                "--full_expensive_cache_jsonl",
                str(shared_full_expensive_cache),
                *benchmark_expensive_runtime_args(args),
            ),
            cwd=PROJECT_ROOT,
            log_path=logs_dir / f"{spec.experiment_id}_benchmark.log",
            env=benchmark_env,
        )

    metrics = collect_experiment_metrics(training_dir, benchmark_dir)
    return {
        "experiment_id": spec.experiment_id,
        "stage": spec.stage,
        "stage_family": spec.stage_family,
        "label": spec.label,
        "description": spec.description,
        "profile_name": spec.profile_name,
        "override_json": str(override_json),
        "overrides": spec.overrides,
        "training_dir": str(training_dir),
        "benchmark_dir": str(benchmark_dir),
        "metrics": metrics,
    }


def build_gt_cache_for_experiment(
    *,
    row: Dict[str, Any],
    args: argparse.Namespace,
    teacher_jsonl_path: Path,
    gt_paths: Sequence[Path],
    out_root: Path,
    logs_dir: Path,
    shared_gt_expensive_cache: Path,
    selected_ids_csv: Path,
    gt_cache_jsonl: Path,
) -> None:
    gpu_ids = resolve_gpu_ids(str(args.gt_cache_gpu_ids))
    workers = resolve_worker_count(gpu_ids, int(args.gt_cache_num_workers))
    worker_cpu_threads = resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=workers)
    if workers <= 1:
        run_command(
            build_python_cmd(
                "src/scripts/build_gaic_gt_score_cache.py",
                "--candidates_jsonl",
                str(resolve_input_path(args.candidates_jsonl)),
                "--features_jsonl",
                str(resolve_input_path(args.features_jsonl)),
                "--teacher_jsonl",
                str(teacher_jsonl_path),
                "--gaic_train_json",
                str(gt_paths[0]),
                "--gaic_test_json",
                str(gt_paths[1]),
                "--out_jsonl",
                str(gt_cache_jsonl),
                "--image_ids_csv",
                str(selected_ids_csv),
                "--c1_jsonl",
                str(resolve_input_path(args.c1_jsonl)),
                "--image_root",
                str(resolve_input_path(args.image_root)),
                "--sample_expensive_cache_jsonl",
                str(shared_gt_expensive_cache),
                "--score_profile",
                str(row["profile_name"]),
                "--score_profile_overrides_json",
                str(row["override_json"]),
            ),
            cwd=PROJECT_ROOT,
            log_path=logs_dir / f"{row['experiment_id']}_gt_cache.log",
            env=capped_thread_env(
                worker_cpu_threads,
                gpu_id=gpu_ids[0] if gpu_ids else None,
            ),
        )
        return

    shard_dir = out_root / row["experiment_id"] / "gt_cache_shards"
    ensure_dir(shard_dir)
    shard_gt_paths: List[Path] = []
    shard_expensive_paths: List[Path] = []
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for shard_index in range(workers):
            shard_gt = shard_dir / f"gaic_gt_score_cache_free_shard_{shard_index}.jsonl"
            shard_exp = shard_dir / f"gaic_gt_expensive_cache_free_shard_{shard_index}.jsonl"
            shard_gt_paths.append(shard_gt)
            shard_expensive_paths.append(shard_exp)
            gpu_id = gpu_ids[shard_index % len(gpu_ids)] if gpu_ids else None
            env = capped_thread_env(worker_cpu_threads, gpu_id=gpu_id)
            cmd = build_python_cmd(
                "src/scripts/build_gaic_gt_score_cache.py",
                "--candidates_jsonl",
                str(resolve_input_path(args.candidates_jsonl)),
                "--features_jsonl",
                str(resolve_input_path(args.features_jsonl)),
                "--teacher_jsonl",
                str(teacher_jsonl_path),
                "--gaic_train_json",
                str(gt_paths[0]),
                "--gaic_test_json",
                str(gt_paths[1]),
                "--out_jsonl",
                str(shard_gt),
                "--image_ids_csv",
                str(selected_ids_csv),
                "--c1_jsonl",
                str(resolve_input_path(args.c1_jsonl)),
                "--image_root",
                str(resolve_input_path(args.image_root)),
                "--sample_expensive_cache_jsonl",
                str(shard_exp),
                "--score_profile",
                str(row["profile_name"]),
                "--score_profile_overrides_json",
                str(row["override_json"]),
                "--num_shards",
                str(workers),
                "--shard_index",
                str(shard_index),
            )
            futures.append(
                executor.submit(
                    run_command,
                    cmd,
                    cwd=PROJECT_ROOT,
                    log_path=logs_dir / f"{row['experiment_id']}_gt_cache_shard_{shard_index}.log",
                    env=env,
                )
            )
        for future in as_completed(futures):
            future.result()
    merge_jsonl_unique(shard_gt_paths, gt_cache_jsonl, key_field="image_id")
    merge_jsonl_unique(shard_expensive_paths, shared_gt_expensive_cache, key_field="cache_key")


def main() -> None:
    args = parse_args()
    out_root = (PROJECT_ROOT / args.out_root).resolve()
    ensure_dir(out_root)
    logs_dir = out_root / "logs"
    ensure_dir(logs_dir)
    gt_paths = [resolve_input_path(args.gaic_train_json), resolve_input_path(args.gaic_test_json)]
    shared_full_expensive_cache = out_root / "shared_full_expensive_cache.jsonl"
    shared_gt_expensive_cache = out_root / "shared_gt_expensive_cache_free.jsonl"
    teacher_jsonl_path = build_teacher_subset_jsonl(
        resolve_input_path(args.teacher_jsonl),
        out_root / "teacher_subset.jsonl",
        int(args.teacher_max_images),
    )

    requested_experiment_ids = {str(value).strip() for value in args.experiment_ids if str(value).strip()}
    selected_specs = [spec for spec in EXPERIMENT_SPECS if not requested_experiment_ids or spec.experiment_id in requested_experiment_ids]
    if not selected_specs:
        raise SystemExit("no experiment specs selected")
    precompute_shared_full_expensive_cache(
        args=args,
        teacher_jsonl_path=teacher_jsonl_path,
        gt_paths=gt_paths,
        out_root=out_root,
        shared_full_expensive_cache=shared_full_expensive_cache,
        logs_dir=logs_dir,
    )

    experiment_rows_by_id: Dict[str, Dict[str, Any]] = {}
    max_jobs = max(1, int(args.parallel_experiment_jobs))
    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        future_map = {
            executor.submit(
                run_experiment_pipeline,
                spec=spec,
                args=args,
                teacher_jsonl_path=teacher_jsonl_path,
                gt_paths=gt_paths,
                out_root=out_root,
                logs_dir=logs_dir,
                shared_full_expensive_cache=shared_full_expensive_cache,
            ): spec
            for spec in selected_specs
        }
        for future in as_completed(future_map):
            result = future.result()
            experiment_rows_by_id[str(result["experiment_id"])] = result
    experiment_rows = [experiment_rows_by_id[spec.experiment_id] for spec in selected_specs]

    stage_winners: Dict[str, Dict[str, Any]] = {}
    for stage_family in ("stage1", "stage2", "stage3", "stage4"):
        candidates = [row for row in experiment_rows if row["stage_family"] == stage_family]
        if not candidates:
            continue
        candidates.sort(key=lambda row: safe_float(safe_dict(row.get("metrics")).get("objective"), -1e9), reverse=True)
        stage_winners[stage_family] = candidates[0]

    if int(args.run_winner_refresh) > 0:
        benchmark_gpu_ids = resolve_gpu_ids(str(args.benchmark_gpu_ids))
        benchmark_env = capped_thread_env(
            resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=1),
            gpu_id=benchmark_gpu_ids[0] if benchmark_gpu_ids else None,
        )
        for stage_family, winner in stage_winners.items():
            spec = next(spec for spec in selected_specs if spec.experiment_id == winner["experiment_id"])
            experiment_root = out_root / spec.experiment_id
            benchmark_dir = experiment_root / "benchmark_winner_refresh"
            override_json = experiment_root / "score_profile_overrides.json"
            if bool(int(args.force) > 0) or not (benchmark_dir / "benchmark_summary.json").exists():
                run_command(
                    build_python_cmd(
                        "src/scripts/run_gaic_benchmark_eval.py",
                        "--candidates_jsonl",
                        str(resolve_input_path(args.candidates_jsonl)),
                        "--features_jsonl",
                        str(resolve_input_path(args.features_jsonl)),
                        "--teacher_jsonl",
                        str(teacher_jsonl_path),
                        "--training_label_dir",
                        str(experiment_root / "training_labels"),
                        "--gaic_train_json",
                        str(gt_paths[0]),
                        "--gaic_test_json",
                        str(gt_paths[1]),
                        "--image_dir",
                        str(resolve_input_path(args.image_root)),
                        "--output_dir",
                        str(benchmark_dir),
                        "--c1_jsonl",
                        str(resolve_input_path(args.c1_jsonl)),
                        "--top_quantile",
                        str(args.benchmark_top_quantile),
                        "--softmax_tau",
                        str(args.benchmark_softmax_tau),
                        "--sample_count",
                        str(args.benchmark_sample_count),
                        "--ge_full_topk",
                        str(args.benchmark_ge_full_topk),
                        "--max_images",
                        str(args.winner_benchmark_max_images),
                        "--score_profile",
                        spec.profile_name,
                        "--score_profile_overrides_json",
                        str(override_json),
                        "--run_full_expensive",
                        str(int(args.run_full_expensive)),
                        "--full_expensive_cache_jsonl",
                        str(shared_full_expensive_cache),
                        *benchmark_expensive_runtime_args(args),
                    ),
                    cwd=PROJECT_ROOT,
                    log_path=logs_dir / f"{spec.experiment_id}_benchmark_winner_refresh.log",
                    env=benchmark_env,
                )
            winner["winner_refresh_benchmark_dir"] = str(benchmark_dir)
            winner["winner_refresh_metrics"] = collect_experiment_metrics(experiment_root / "training_labels", benchmark_dir)

    if int(args.skip_debug_viz) == 0:
        core_debug_targets = set()
        core_debug_targets.add("baseline_current_refined")
        core_debug_targets.update(winner["experiment_id"] for winner in stage_winners.values())
        debug_targets = set(core_debug_targets)
        if int(args.build_debug_viz_for_all) > 0:
            debug_targets.update(row["experiment_id"] for row in experiment_rows)

        for row in experiment_rows:
            if row["experiment_id"] not in debug_targets:
                continue
            experiment_root = out_root / row["experiment_id"]
            training_dir = experiment_root / "training_labels"
            coco_json = training_dir / "coco" / "instances_conditional_detr_batch_gaic_like.json"
            subject_mode_vocab = training_dir / "subject_mode_vocab.json"
            debug_sample_size = int(args.debug_sample_size)
            if (
                int(args.build_debug_viz_for_all) > 0
                and row["experiment_id"] not in core_debug_targets
                and int(args.debug_sample_size_all) > 0
            ):
                debug_sample_size = int(args.debug_sample_size_all)
            selected_ids = select_debug_image_ids(
                coco_json=coco_json,
                image_root=resolve_input_path(args.image_root),
                subject_mode_vocab=subject_mode_vocab,
                gt_paths=gt_paths,
                sample_size=debug_sample_size,
                seed=int(args.debug_seed),
            )
            selected_ids_csv = experiment_root / "debug_selected_images.csv"
            write_selected_images_csv(selected_ids_csv, selected_ids)

            gt_cache_jsonl = training_dir / "gaic_gt_score_cache_free.jsonl"
            if int(args.skip_gt_cache) == 0 and (bool(int(args.force) > 0) or not gt_cache_jsonl.exists()):
                build_gt_cache_for_experiment(
                    row=row,
                    args=args,
                    teacher_jsonl_path=teacher_jsonl_path,
                    gt_paths=gt_paths,
                    out_root=out_root,
                    logs_dir=logs_dir,
                    shared_gt_expensive_cache=shared_gt_expensive_cache,
                    selected_ids_csv=selected_ids_csv,
                    gt_cache_jsonl=gt_cache_jsonl,
                )

            debug_out_dir = training_dir / "debug_visualizations_balanced50_bottomneg"
            if bool(int(args.force) > 0) or not (debug_out_dir / "summary" / "summary.json").exists():
                run_command(
                    build_python_cmd(
                        "src/scripts/build_gaic_training_label_debug_viz.py",
                        "--coco_json",
                        str(coco_json),
                        "--batch_jsonl",
                        str(training_dir / "train_conditional_detr_batch.jsonl"),
                        "--gaic_gt_train_json",
                        str(gt_paths[0]),
                        "--gaic_gt_test_json",
                        str(gt_paths[1]),
                        "--gt_score_cache_jsonl",
                        str(gt_cache_jsonl),
                        "--teacher_jsonl",
                        str(teacher_jsonl_path),
                        "--image_root",
                        str(resolve_input_path(args.image_root)),
                        "--subject_mode_vocab",
                        str(subject_mode_vocab),
                        "--sample_size",
                        str(debug_sample_size),
                        "--seed",
                        str(args.debug_seed),
                        "--show_why_text",
                        str(int(args.show_why_text)),
                        "--skip_by_ar",
                        str(int(args.debug_skip_by_ar)),
                        "--format",
                        str(args.debug_viz_format),
                        "--jpg_quality",
                        str(int(args.debug_viz_jpg_quality)),
                        "--score_profile_json",
                        str(training_dir / "score_profile.json"),
                        "--out_dir",
                        str(debug_out_dir),
                    ),
                    cwd=PROJECT_ROOT,
                    log_path=logs_dir / f"{row['experiment_id']}_debug_viz.log",
                    env=capped_thread_env(resolve_worker_cpu_threads(int(args.worker_cpu_threads), concurrent_workers=1)),
                )
            row["debug_viz_dir"] = str(debug_out_dir)

    summary_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "args": vars(args),
        "teacher_jsonl_resolved": str(teacher_jsonl_path),
        "experiments": experiment_rows,
        "stage_winners": stage_winners,
    }
    write_json(out_root / "experiment_summary.json", summary_payload)
    report_text = experiment_report_markdown(args=args, experiment_rows=experiment_rows, stage_winners=stage_winners)
    (out_root / "EXPERIMENT_REPORT_KO.md").write_text(report_text, encoding="utf-8")
    print(json.dumps({"status": "ok", "out_root": str(out_root), "experiment_count": len(experiment_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
