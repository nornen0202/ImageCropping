#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (  # noqa: E402
    TARGET_AR_VALUES,
    box_area,
    letterbox_content_box,
    load_image_tensor_letterbox,
    original_to_letterbox_box,
    target_ar_id,
    xyxy_to_cxcywh,
)
from mobilecropnet_v4.eval_utils import (  # noqa: E402
    SCORE_SURFACES,
    infer_image_norm,
    infer_input_size,
    load_mobilecropnet_v4_checkpoint,
    parse_norm_triplet,
    select_score_logits,
)


SUPPORTED_MODEL_ARS = tuple(key for key, value in TARGET_AR_VALUES.items() if value is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score staged FCDB/CPC/GNMC public benchmark candidate windows with a MobileCropNet v4 checkpoint."
    )
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", default="", type=Path)
    parser.add_argument("--method_name", default="mobilecropnet_v4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project_root", default=PROJECT_ROOT, type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast for checkpoint scoring.")
    parser.add_argument(
        "--resume_existing_output",
        action="store_true",
        help="Append to an existing output JSONL and skip task rows already present in that file.",
    )
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--image_mean", default=None)
    parser.add_argument("--image_std", default=None)
    parser.add_argument("--image_cache_size", type=int, default=256)
    parser.add_argument(
        "--score_surface",
        choices=SCORE_SURFACES,
        default="deployment",
        help="Score surface for public benchmark ranking. deployment matches runtime inference/direct evaluation.",
    )
    parser.add_argument("--task_shard_index", type=int, default=0)
    parser.add_argument("--task_shard_count", type=int, default=1)
    parser.add_argument(
        "--progress_json",
        default="",
        type=Path,
        help="Optional durable progress JSON. Defaults to <output_jsonl parent>/export_progress.json.",
    )
    parser.add_argument(
        "--unsupported_target_ar_policy",
        choices=["nearest", "free", "skip"],
        default="nearest",
        help="MobileCropNet v4 has no native 2:1 AR id. nearest maps it to the closest trained AR class.",
    )
    return parser.parse_args()


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("image_id", "")),
        str(row.get("target_ar", "")),
        str(row.get("model_target_ar", "")),
    )


def load_existing_prediction_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                keys.add(prediction_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def parse_ar(text: Any) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    raw = str(text).strip()
    try:
        if ":" in raw:
            left, right = raw.split(":", 1)
            return float(left) / max(1e-6, float(right))
        return float(raw)
    except (TypeError, ValueError):
        return None


def model_target_ar(public_target_ar: Any, *, policy: str) -> tuple[str, bool]:
    if public_target_ar is None or str(public_target_ar).strip() == "":
        return "FREE", False
    text = str(public_target_ar).strip()
    if text in TARGET_AR_VALUES:
        return text, False
    if policy == "free":
        return "FREE", True
    if policy == "skip":
        return "", True
    value = parse_ar(text)
    if value is None:
        return "FREE", True
    nearest = min(
        SUPPORTED_MODEL_ARS,
        key=lambda key: abs(math.log(max(1e-6, float(TARGET_AR_VALUES[key] or 1.0)) / max(1e-6, value))),
    )
    return str(nearest), True


def valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def clamp_box(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [float(x1), float(y1), float(x2), float(y2)]


def candidate_box(candidate: dict[str, Any]) -> list[float] | None:
    box = candidate.get("bbox_xyxy_norm", candidate.get("bbox_norm_xyxy"))
    if not valid_box(box):
        return None
    return clamp_box([float(v) for v in box])


def is_base_candidate(candidate: dict[str, Any]) -> float:
    text = " ".join(
        str(candidate.get(key, ""))
        for key in ("candidate_id", "source", "label")
    ).lower()
    return 1.0 if ("baseline" in text or "synthetic_full" in text or "full" == text.strip()) else 0.0


def box_meta(box: Sequence[float], *, is_base: float) -> list[float]:
    cxcywh = xyxy_to_cxcywh(box)
    area = box_area(box)
    aspect = cxcywh[2] / max(1e-6, cxcywh[3])
    return [*box, *cxcywh, float(area), float(aspect), float(is_base)]


class ImageTensorCache:
    def __init__(self, *, input_size: int, image_mean: Sequence[float] | None, image_std: Sequence[float] | None, max_items: int) -> None:
        self.input_size = int(input_size)
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_items = max(0, int(max_items))
        self._cache: OrderedDict[str, tuple[torch.Tensor, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, path: Path) -> tuple[torch.Tensor, Any]:
        key = str(path.resolve())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached
        image, transform = load_image_tensor_letterbox(
            path,
            self.input_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        self.misses += 1
        if self.max_items > 0:
            self._cache[key] = (image, transform)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return image, transform


def task_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, candidate in enumerate(task.get("candidate_windows") or []):
        if not isinstance(candidate, dict):
            continue
        box = candidate_box(candidate)
        if box is None:
            continue
        row = dict(candidate)
        row["candidate_id"] = str(row.get("candidate_id") or f"candidate_{idx:04d}")
        row["bbox_xyxy_norm"] = box
        out.append(row)
    return out


def remap_path_to_project_root(path: Path, project_root: Path) -> Path:
    if path.exists():
        return path
    parts = list(path.parts)
    for anchor_parts in (("Sources", "ImageCropping"), ("artifacts",), ("data",)):
        width = len(anchor_parts)
        for idx in range(0, len(parts) - width + 1):
            if tuple(parts[idx : idx + width]) != anchor_parts:
                continue
            rel_parts = parts[idx + width :] if width == 2 else parts[idx:]
            if not rel_parts:
                continue
            candidate = project_root / Path(*rel_parts)
            if candidate.exists():
                return candidate
    return path


def resolve_task_image_path(task: dict[str, Any], *, project_root: Path) -> Path | None:
    raw_candidates: list[str] = []
    staging = task.get("staging")
    if isinstance(staging, dict):
        staged_image_path = staging.get("staged_image_path")
        if staged_image_path:
            raw_candidates.append(str(staged_image_path))
    raw_image_path = task.get("image_path")
    if raw_image_path:
        raw_candidates.append(str(raw_image_path))
    meta = task.get("meta")
    if isinstance(meta, dict):
        for key in ("staged_image_path", "source_image_path", "image_path"):
            value = meta.get(key)
            if value:
                raw_candidates.append(str(value))
    seen: set[str] = set()
    for raw in raw_candidates:
        if raw in seen:
            continue
        seen.add(raw)
        resolved = remap_path_to_project_root(Path(raw), project_root)
        if resolved.exists():
            return resolved
    return None


def score_batch(
    *,
    model: torch.nn.Module,
    batch: list[dict[str, Any]],
    cache: ImageTensorCache,
    device: torch.device,
    amp: bool,
    score_surface: str,
) -> list[dict[str, Any]]:
    images: list[torch.Tensor] = []
    model_ar_ids: list[int] = []
    image_ar_logs: list[float] = []
    content_boxes: list[list[float]] = []
    padded_boxes: list[list[list[float]]] = []
    padded_meta: list[list[list[float]]] = []
    padded_base: list[list[float]] = []
    valid: list[list[float]] = []
    max_k = max(len(item["candidates"]) for item in batch)

    for item in batch:
        image, transform = cache.get(Path(str(item["image_path"])))
        images.append(image)
        model_ar_ids.append(target_ar_id(str(item["model_target_ar"])))
        image_ar_logs.append(float(transform.image_ar_log))
        content_boxes.append(letterbox_content_box(transform))
        cur_boxes: list[list[float]] = []
        cur_meta: list[list[float]] = []
        cur_base: list[float] = []
        for candidate in item["candidates"]:
            orig_box = candidate["bbox_xyxy_norm"]
            padded = original_to_letterbox_box(orig_box, transform)
            base = is_base_candidate(candidate)
            cur_boxes.append([float(v) for v in padded])
            cur_meta.append(box_meta(padded, is_base=base))
            cur_base.append(float(base))
        while len(cur_boxes) < max_k:
            cur_boxes.append([0.0, 0.0, 1e-4, 1e-4])
            cur_meta.append([0.0] * 11)
            cur_base.append(0.0)
        padded_boxes.append(cur_boxes)
        padded_meta.append(cur_meta)
        padded_base.append(cur_base)
        valid.append([1.0] * len(item["candidates"]) + [0.0] * (max_k - len(item["candidates"])))

    autocast_enabled = bool(amp and device.type == "cuda")
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        outputs = model(
            torch.stack(images, dim=0).to(device),
            torch.tensor(padded_boxes, dtype=torch.float32, device=device),
            torch.tensor(valid, dtype=torch.float32, device=device),
            torch.tensor(model_ar_ids, dtype=torch.long, device=device),
            torch.tensor(image_ar_logs, dtype=torch.float32, device=device),
            torch.tensor(padded_base, dtype=torch.float32, device=device),
            torch.tensor(padded_meta, dtype=torch.float32, device=device),
            torch.tensor(content_boxes, dtype=torch.float32, device=device),
        )
        score_logits = select_score_logits(outputs, score_surface=score_surface)
        scores = torch.sigmoid(score_logits).detach().cpu()
        positives = torch.sigmoid(outputs.get("positive_logits", outputs["utility_logits"])).detach().cpu()
        risks = torch.sigmoid(outputs.get("risk_logits", outputs["utility_logits"])).detach().cpu()

    rows: list[dict[str, Any]] = []
    for bidx, item in enumerate(batch):
        candidates: list[dict[str, Any]] = []
        raw_scores = scores[bidx, : len(item["candidates"])].tolist()
        order = sorted(range(len(raw_scores)), key=lambda idx: float(raw_scores[idx]), reverse=True)
        rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(order)}
        for cidx, candidate in enumerate(item["candidates"]):
            out = {
                "candidate_id": str(candidate.get("candidate_id", f"candidate_{cidx:04d}")),
                "bbox_xyxy_norm": [float(v) for v in candidate["bbox_xyxy_norm"]],
                "score": float(raw_scores[cidx]),
                "source": str(candidate.get("source", "")),
                "label": str(candidate.get("label", "")),
                "model_rank": int(rank_by_idx[cidx]),
                "scores": {
                    "mobilecropnet_utility": float(raw_scores[cidx]),
                    "mobilecropnet_positive": float(positives[bidx, cidx].item()),
                    "mobilecropnet_risk": float(risks[bidx, cidx].item()),
                },
            }
            if isinstance(candidate.get("meta"), dict):
                out["meta"] = dict(candidate["meta"])
            candidates.append(out)
        rows.append(
            {
                "dataset": str(item["dataset"]).lower(),
                "image_id": str(item["image_id"]),
                "target_ar": item.get("target_ar"),
                "method": str(item["method_name"]),
                "source": "mobilecropnet_v4_checkpoint",
                "checkpoint": str(item["checkpoint"]),
                "model_target_ar": str(item["model_target_ar"]),
                "target_ar_was_mapped": bool(item["target_ar_was_mapped"]),
                "score_surface": str(item["score_surface"]),
                "candidates": candidates,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    start = time.time()
    checkpoint = args.checkpoint.resolve()
    device = torch.device(str(args.device))
    project_root = args.project_root.resolve()
    model, ckpt = load_mobilecropnet_v4_checkpoint(checkpoint, device=device)
    input_size = infer_input_size(ckpt, args.input_size)
    image_mean = parse_norm_triplet(args.image_mean, default=None) if args.image_mean else None
    image_std = parse_norm_triplet(args.image_std, default=None) if args.image_std else None
    norm_mean, norm_std = infer_image_norm(ckpt, explicit_mean=image_mean, explicit_std=image_std)
    cache = ImageTensorCache(
        input_size=input_size,
        image_mean=norm_mean,
        image_std=norm_std,
        max_items=args.image_cache_size,
    )
    progress_arg = str(args.progress_json).strip()
    progress_json = args.progress_json if progress_arg and progress_arg != "." else args.output_jsonl.parent / "export_progress.json"
    existing_prediction_keys = (
        load_existing_prediction_keys(args.output_jsonl.resolve()) if bool(args.resume_existing_output) else set()
    )
    existing_prediction_rows = len(existing_prediction_keys)
    progress_base = {
        "method_name": str(args.method_name),
        "checkpoint": str(checkpoint),
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "summary_json": str(args.summary_json.resolve()) if str(args.summary_json).strip() else "",
        "project_root": str(project_root),
        "input_size": int(input_size),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "amp": bool(args.amp),
        "score_surface": str(args.score_surface),
        "resume_existing_output": bool(args.resume_existing_output),
        "existing_prediction_rows": int(existing_prediction_rows),
        "max_tasks": int(args.max_tasks),
        "task_shard_index": int(args.task_shard_index),
        "task_shard_count": int(args.task_shard_count),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
    }
    write_progress(
        progress_json.resolve(),
        {
            **progress_base,
            "phase": "public_export",
            "state": "running",
            "task_count": 0,
            "candidate_count": 0,
            "prediction_rows": int(existing_prediction_rows),
            "resumed_existing_count": 0,
            "skipped_count": 0,
            "duration_sec": 0.0,
        },
    )

    args.output_jsonl.resolve().parent.mkdir(parents=True, exist_ok=True)
    prediction_rows = int(existing_prediction_rows)
    resumed_existing_count = 0
    skipped: list[dict[str, Any]] = []
    mapped_target_ar_counts: dict[str, int] = {}
    batch: list[dict[str, Any]] = []
    task_count = 0
    manifest_rows_seen = 0
    shard_skipped_count = 0
    candidate_count = 0
    task_shard_count = max(1, int(args.task_shard_count))
    task_shard_index = int(args.task_shard_index)
    if task_shard_index < 0 or task_shard_index >= task_shard_count:
        raise SystemExit(f"--task_shard_index must be in [0, {task_shard_count - 1}], got {task_shard_index}")
    output_mode = "a" if bool(args.resume_existing_output) else "w"
    with args.output_jsonl.resolve().open(output_mode, encoding="utf-8") as output_handle:
        def flush_batch() -> None:
            nonlocal batch, prediction_rows
            if not batch:
                return
            for row in score_batch(
                model=model,
                batch=batch,
                cache=cache,
                device=device,
                amp=bool(args.amp),
                score_surface=str(args.score_surface),
            ):
                output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                prediction_rows += 1
            output_handle.flush()
            batch = []
            write_progress(
                progress_json.resolve(),
                {
                    **progress_base,
                    "phase": "public_export",
                    "state": "running",
                    "task_count": int(task_count),
                    "manifest_rows_seen": int(manifest_rows_seen),
                    "shard_skipped_count": int(shard_skipped_count),
                    "candidate_count": int(candidate_count),
                    "prediction_rows": int(prediction_rows),
                    "resumed_existing_count": int(resumed_existing_count),
                    "skipped_count": int(len(skipped)),
                    "image_cache": {"max_items": int(args.image_cache_size), "hits": int(cache.hits), "misses": int(cache.misses)},
                    "duration_sec": round(time.time() - start, 3),
                },
            )

        for manifest_index, task in enumerate(read_jsonl(args.task_manifest_jsonl.resolve())):
            manifest_rows_seen += 1
            if task_shard_count > 1 and (manifest_index % task_shard_count) != task_shard_index:
                shard_skipped_count += 1
                continue
            if args.max_tasks > 0 and task_count >= args.max_tasks:
                break
            image_path = resolve_task_image_path(task, project_root=project_root)
            if image_path is None:
                skipped.append({"dataset": task.get("dataset"), "image_id": task.get("image_id"), "reason": "missing_image"})
                continue
            candidates = task_candidates(task)
            if not candidates:
                skipped.append({"dataset": task.get("dataset"), "image_id": task.get("image_id"), "reason": "no_candidates"})
                continue
            ar, mapped = model_target_ar(task.get("target_ar"), policy=str(args.unsupported_target_ar_policy))
            if ar == "":
                skipped.append(
                    {
                        "dataset": task.get("dataset"),
                        "image_id": task.get("image_id"),
                        "target_ar": task.get("target_ar"),
                        "reason": "unsupported_target_ar",
                    }
                )
                continue
            if mapped:
                key = f"{task.get('target_ar')}->{ar}"
                mapped_target_ar_counts[key] = mapped_target_ar_counts.get(key, 0) + 1
            task_key = (str(task.get("dataset", "")), str(task.get("image_id", "")), str(task.get("target_ar", "")), str(ar))
            if task_key in existing_prediction_keys:
                task_count += 1
                resumed_existing_count += 1
                continue
            batch.append(
                {
                    "dataset": task.get("dataset"),
                    "image_id": task.get("image_id"),
                    "target_ar": task.get("target_ar"),
                    "image_path": str(image_path),
                    "candidates": candidates,
                    "method_name": str(args.method_name),
                    "checkpoint": str(checkpoint),
                    "model_target_ar": ar,
                    "target_ar_was_mapped": mapped,
                    "score_surface": str(args.score_surface),
                }
            )
            task_count += 1
            candidate_count += len(candidates)
            if len(batch) >= max(1, int(args.batch_size)):
                flush_batch()
        flush_batch()

    written = prediction_rows
    summary = {
        "method_name": str(args.method_name),
        "checkpoint": str(checkpoint),
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "prediction_rows": int(written),
        "task_count": int(task_count),
        "manifest_rows_seen": int(manifest_rows_seen),
        "shard_skipped_count": int(shard_skipped_count),
        "task_shard_index": int(task_shard_index),
        "task_shard_count": int(task_shard_count),
        "candidate_count": int(candidate_count),
        "resume_existing_output": bool(args.resume_existing_output),
        "existing_prediction_rows": int(existing_prediction_rows),
        "resumed_existing_count": int(resumed_existing_count),
        "project_root": str(project_root),
        "skipped_count": len(skipped),
        "skipped": skipped[:100],
        "mapped_target_ar_counts": mapped_target_ar_counts,
        "unsupported_target_ar_policy": str(args.unsupported_target_ar_policy),
        "score_surface": str(args.score_surface),
        "input_size": int(input_size),
        "image_mean": norm_mean,
        "image_std": norm_std,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "amp": bool(args.amp),
        "image_cache": {"max_items": int(args.image_cache_size), "hits": int(cache.hits), "misses": int(cache.misses)},
        "duration_sec": round(time.time() - start, 3),
    }
    if str(args.summary_json).strip():
        args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_progress(progress_json.resolve(), {**progress_base, "phase": "public_export", "state": "completed", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
