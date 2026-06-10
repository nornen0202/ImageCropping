from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from public_benchmark.adapters import box_area, box_aspect, image_size, iter_tasks
from public_benchmark.schema import BenchmarkBox, BenchmarkTask


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sanitize_token(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:180] or "empty"


def staged_image_id(dataset: str, image_id: str) -> str:
    return f"pb_{sanitize_token(dataset.lower())}_{sanitize_token(image_id)}"


def sstk_target_ar(task: BenchmarkTask) -> str:
    return str(task.target_ar) if task.target_ar else "FREE"


def task_key(dataset: str, image_id: str, target_ar: Optional[str]) -> str:
    return "|".join([str(dataset).lower(), str(image_id), "" if target_ar is None else str(target_ar)])


def _candidate_for_teacher(box: BenchmarkBox, *, fallback_idx: int) -> dict[str, Any]:
    bbox = [float(v) for v in box.bbox_xyxy_norm]
    score = box.score
    out = {
        "candidate_id": box.candidate_id or f"benchmark_candidate_{fallback_idx:04d}",
        "bbox_norm_xyxy": bbox,
        "bbox_xyxy_norm": bbox,
        "source": box.source or "benchmark_candidate",
        "label": box.label,
        "score": score,
        "area_ratio": float(box_area(bbox)),
        "ar": float(box_aspect(bbox)),
        "must_keep": bool(str(box.source).endswith("_gt")),
        "priority": float(score) if score is not None else float(box.weight),
        "scale_idx": int(fallback_idx),
        "meta": dict(box.meta),
    }
    return out


def _link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def collect_public_benchmark_tasks(
    *,
    data_root: Path,
    datasets: list[str],
    split: str = "all",
    max_tasks_per_dataset: int = 0,
    max_pairwise_per_task: int = 200,
    min_pairwise_gap: float = 0.0,
) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for dataset in [str(item).lower() for item in datasets]:
        count = 0
        for task in iter_tasks(
            dataset,
            data_root,
            split=split,
            max_pairwise_per_task=max_pairwise_per_task,
            min_pairwise_gap=min_pairwise_gap,
        ):
            if max_tasks_per_dataset > 0 and count >= max_tasks_per_dataset:
                break
            tasks.append(task)
            count += 1
    return tasks


def stage_public_benchmark_inputs(
    *,
    data_root: Path,
    output_dir: Path,
    datasets: list[str],
    split: str = "all",
    max_tasks_per_dataset: int = 0,
    max_pairwise_per_task: int = 200,
    min_pairwise_gap: float = 0.0,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stage_dir = output_dir / "images"
    tasks = collect_public_benchmark_tasks(
        data_root=data_root,
        datasets=datasets,
        split=split,
        max_tasks_per_dataset=max_tasks_per_dataset,
        max_pairwise_per_task=max_pairwise_per_task,
        min_pairwise_gap=min_pairwise_gap,
    )

    image_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    target_ars_by_image: dict[tuple[str, str], set[str]] = defaultdict(set)
    task_rows: list[dict[str, Any]] = []
    keymap_rows: list[dict[str, Any]] = []
    candidates_by_image_ar: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    size_by_staged_id: dict[str, tuple[int, int]] = {}

    for task in tasks:
        src = Path(task.image_path)
        width = int(task.meta.get("image_width", 0) or 0)
        height = int(task.meta.get("image_height", 0) or 0)
        if width <= 0 or height <= 0:
            width, height = image_size(src)
        sid = staged_image_id(task.dataset, task.image_id)
        ext = src.suffix.lower() or ".jpg"
        staged_path = image_stage_dir / f"{sid}{ext}"
        _link_or_copy_image(src, staged_path, link_mode)

        image_key = (task.dataset, task.image_id)
        image_rows_by_key[image_key] = {
            "dataset": task.dataset,
            "image_id": task.image_id,
            "staged_image_id": sid,
            "source_image_path": str(src.resolve()),
            "staged_image_path": str(staged_path.resolve()),
            "width": width,
            "height": height,
        }
        size_by_staged_id[sid] = (width, height)
        ar_key = sstk_target_ar(task)
        target_ars_by_image[image_key].add(ar_key)

        row = task.to_dict()
        row["staging"] = {
            "task_key": task_key(task.dataset, task.image_id, task.target_ar),
            "staged_image_id": sid,
            "staged_image_path": str(staged_path.resolve()),
            "sstk_target_ar": ar_key,
        }
        task_rows.append(row)
        keymap_rows.append(
            {
                "dataset": task.dataset,
                "image_id": task.image_id,
                "target_ar": task.target_ar,
                "task_key": task_key(task.dataset, task.image_id, task.target_ar),
                "staged_image_id": sid,
                "sstk_target_ar": ar_key,
                "source_image_path": str(src.resolve()),
                "staged_image_path": str(staged_path.resolve()),
            }
        )

        for idx, box in enumerate(task.candidate_windows, start=1):
            candidates_by_image_ar[sid][ar_key].append(_candidate_for_teacher(box, fallback_idx=idx))

    image_rows: list[dict[str, Any]] = []
    parquet_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for image_key, image_row in sorted(image_rows_by_key.items()):
        sid = image_row["staged_image_id"]
        width, height = size_by_staged_id[sid]
        target_ars = sorted(target_ars_by_image[image_key])
        image_row = dict(image_row)
        image_row["sstk_target_ars"] = target_ars
        image_rows.append(image_row)
        parquet_rows.append(
            {
                "image_id": sid,
                "width": int(width),
                "height": int(height),
                "tags": [],
                "source_dataset": image_row["dataset"],
                "source_image_id": image_row["image_id"],
                "source_image_path": image_row["source_image_path"],
            }
        )
        routing = {
            "subject_mode": "scene_general",
            "policy_id": "generic_v1",
            "subject_mode_conf": 0.0,
            "subject_mode_reasons": ["public_benchmark_staging_stub"],
            "subject_set": {},
            "shot_type": "unknown",
            "primary_subject_type": "other",
            "primary_subject_source": "none",
        }
        feature_rows.append(
            {
                "image_id": sid,
                "width": int(width),
                "height": int(height),
                "c2_seg": [],
                "c2_det": [],
                "c3_pose": [],
                "c5_geom": {},
                "c7_saliency": {},
                "routing": routing,
            }
        )
        ar_candidates = dict(candidates_by_image_ar.get(sid, {}))
        candidate_rows.append(
            {
                "image_id": sid,
                "width": int(width),
                "height": int(height),
                "image_ar": float(width) / float(max(1, height)),
                "meta_norm": {
                    "width_norm_ref": int(width),
                    "height_norm_ref": int(height),
                    "image_ar_norm_ref": float(width) / float(max(1, height)),
                    "size_source": "public_benchmark_staged_image",
                },
                "tags": [],
                "routing": routing,
                "subject_prior": {
                    "bbox_norm_xyxy": [0.25, 0.25, 0.75, 0.75],
                    "centroid": [0.5, 0.5],
                    "size": [0.5, 0.5],
                    "subject_mode": "scene_general",
                    "subject_reliability": 0.0,
                },
                "candidate_gen_hash": "public_benchmark_candidate_windows_v1",
                "proposal_injected": False,
                "teacher_candidates": [],
                "iou_to_teacher_top1_by_ar": {},
                "candidates_by_ar": ar_candidates,
                "stats": {
                    "num_candidates_total": int(sum(len(v) for v in ar_candidates.values())),
                    "num_candidates_by_ar": {k: len(v) for k, v in ar_candidates.items()},
                    "num_teacher_candidates_total": 0,
                    "num_teacher_candidates_by_ar": {},
                },
            }
        )

    task_manifest = output_dir / "benchmark_task_manifest.jsonl"
    image_manifest = output_dir / "benchmark_image_manifest.jsonl"
    keymap_jsonl = output_dir / "benchmark_keymap.jsonl"
    features_jsonl = output_dir / "features_stub.jsonl"
    candidates_jsonl = output_dir / "benchmark_candidates_for_teacher.jsonl"
    parquet_path = output_dir / "sstk_public_images.parquet"

    write_jsonl(task_manifest, task_rows)
    write_jsonl(image_manifest, image_rows)
    write_jsonl(keymap_jsonl, keymap_rows)
    write_jsonl(features_jsonl, feature_rows)
    write_jsonl(candidates_jsonl, candidate_rows)

    try:
        import pandas as pd

        pd.DataFrame(parquet_rows).to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        parquet_written = False
        (output_dir / "sstk_public_images.parquet.error.txt").write_text(str(exc), encoding="utf-8")

    commands_path = output_dir / "PIPELINE_COMMANDS.md"
    commands_path.write_text(
        "\n".join(
            [
                "# Public Benchmark SSTK Pipeline Commands",
                "",
                "Candidate generator smoke using staged public images:",
                "",
                "```bash",
                "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/generate_candidates.py \\",
                f"  --input_parquet {parquet_path} \\",
                f"  --feats_c2_jsonl {features_jsonl} \\",
                f"  --feats_c3_jsonl {features_jsonl} \\",
                f"  --image_dir {image_stage_dir} \\",
                "  --ar_list FREE 1:1 4:3 3:4 16:9 2:1 \\",
                "  --output_jsonl <output-candidates.jsonl> \\",
                "  --use_actual_image_size 1 --num_workers 1",
                "```",
                "",
                "Teacher scorer over benchmark candidate windows:",
                "",
                "```bash",
                "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/score_teacher.py \\",
                f"  --candidates_jsonl {candidates_jsonl} \\",
                f"  --features_jsonl {features_jsonl} \\",
                f"  --image_dir {image_stage_dir} \\",
                "  --output_jsonl <teacher-scores.jsonl> \\",
                "  --output_overview_json <teacher-overview.json> \\",
                "  --use_real_expensive 0 --max_images 0",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "data_root": str(data_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "datasets": [str(item).lower() for item in datasets],
        "split": split,
        "link_mode": link_mode,
        "n_tasks": len(task_rows),
        "n_images": len(image_rows),
        "n_candidate_windows": int(sum(len(v) for row in candidate_rows for v in row["candidates_by_ar"].values())),
        "parquet_written": parquet_written,
        "artifacts": {
            "task_manifest_jsonl": str(task_manifest.resolve()),
            "image_manifest_jsonl": str(image_manifest.resolve()),
            "keymap_jsonl": str(keymap_jsonl.resolve()),
            "features_stub_jsonl": str(features_jsonl.resolve()),
            "benchmark_candidates_for_teacher_jsonl": str(candidates_jsonl.resolve()),
            "sstk_public_images_parquet": str(parquet_path.resolve()),
            "image_stage_dir": str(image_stage_dir.resolve()),
            "pipeline_commands_md": str(commands_path.resolve()),
        },
    }
    summary_path = output_dir / "staging_summary.json"
    summary["artifacts"]["summary_json"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary
