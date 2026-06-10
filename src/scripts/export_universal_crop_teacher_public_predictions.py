#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train_universal_crop_teacher import move_batch_to_device, prediction_rows_for_batch  # noqa: E402
from universal_crop_teacher.data import UniversalCropTeacherDataset, collate_uctr_batch  # noqa: E402
from universal_crop_teacher.model import UniversalCropTeacherH, UniversalCropTeacherHConfig  # noqa: E402
from universal_crop_teacher.warehouse import public_task_to_warehouse_row, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public benchmark predictions from a UniversalCropTeacher-H checkpoint.")
    parser.add_argument("--task_manifest_jsonl", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", default="", type=Path)
    parser.add_argument("--method_name", default="universal_crop_teacher_h")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--split_filter", default="", help="Optional warehouse-style split filter after deterministic split assignment.")
    parser.add_argument("--split_seed", type=int, default=20260420)
    parser.add_argument("--force_public_hash_split", type=int, default=0)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    start = time.time()
    ckpt = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = UniversalCropTeacherHConfig.from_dict(ckpt.get("config", {}))
    model = UniversalCropTeacherH(config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(str(args.device))
    model.to(device).eval()

    rows: list[dict[str, Any]] = []
    for task in read_jsonl(args.task_manifest_jsonl.resolve(), max_rows=int(args.max_tasks)):
        row = public_task_to_warehouse_row(
            task,
            split_seed=int(args.split_seed),
            force_hash_split=bool(int(args.force_public_hash_split) > 0),
        )
        if str(args.split_filter).strip() and str(row.get("split", "")) != str(args.split_filter).strip():
            continue
        rows.append(row)

    dataset = UniversalCropTeacherDataset(
        rows,
        max_candidates=int(args.max_candidates),
        image_size=int(args.image_size),
        crop_size=int(args.crop_size),
        include_images=True,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_uctr_batch,
    )
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch_to_device(batch, device)
            outputs = model(
                batch_dev["global_images"],
                batch_dev["crop_images"],
                batch_dev["candidate_features"],
                batch_dev["valid_mask"],
                batch_dev["dataset_ids"],
                batch_dev["target_ar_ids"],
            )
            predictions.extend(prediction_rows_for_batch(outputs, batch, method=str(args.method_name)))
    written = write_jsonl(args.output_jsonl.resolve(), predictions)
    candidate_count = sum(len(row.get("candidates", []) or []) for row in predictions)
    summary = {
        "method_name": str(args.method_name),
        "checkpoint": str(args.checkpoint.resolve()),
        "task_manifest_jsonl": str(args.task_manifest_jsonl.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "prediction_rows": int(written),
        "candidate_count": int(candidate_count),
        "split_filter": str(args.split_filter),
        "duration_sec": round(time.time() - start, 3),
        "device": str(device),
    }
    if str(args.summary_json).strip():
        args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
