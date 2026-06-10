#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import MobileCropNetV4BatchDataset, mobilecropnet_v4_collate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MobileCropNet v4 data loading throughput.")
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--input_size", type=int, default=288)
    parser.add_argument("--candidate_k", type=int, default=24)
    parser.add_argument("--image_mean", default=None, help="Comma-separated RGB mean.")
    parser.add_argument("--image_std", default=None, help="Comma-separated RGB std.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--pin_memory_device", default="")
    parser.add_argument("--precompute_sample_tensors", action="store_true")
    parser.add_argument("--image_tensor_cache_size", type=int, default=0)
    return parser


def _parse_norm_triplet(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    items = [float(part.strip()) for part in str(value).split(",")]
    if len(items) != 3:
        raise ValueError(f"expected 3 normalization values, got {len(items)}")
    return items


def _loader_kwargs(args: argparse.Namespace, dataset: MobileCropNetV4BatchDataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "collate_fn": mobilecropnet_v4_collate,
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(args.prefetch_factor)
    if bool(args.pin_memory) and str(args.pin_memory_device).strip():
        kwargs["pin_memory_device"] = str(args.pin_memory_device)
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    init_start = time()
    dataset = MobileCropNetV4BatchDataset(
        jsonl_path=args.jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=args.input_size,
        candidate_k=args.candidate_k,
        image_mean=_parse_norm_triplet(args.image_mean),
        image_std=_parse_norm_triplet(args.image_std),
        max_rows=args.max_rows,
        precompute_sample_tensors=args.precompute_sample_tensors,
        image_tensor_cache_size=args.image_tensor_cache_size,
    )
    init_sec = time() - init_start
    loader = DataLoader(**_loader_kwargs(args, dataset))

    warmup_steps = max(0, int(args.warmup_steps))
    measured_steps = max(1, int(args.steps))
    total_steps = warmup_steps + measured_steps
    batch_shapes: dict[str, list[int]] = {}
    sample_count = 0
    measured_start: float | None = None
    measured_sec = 0.0

    iterator = iter(loader)
    for step in range(total_steps):
        if step == warmup_steps:
            measured_start = time()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if step >= warmup_steps:
            if torch.is_tensor(batch.get("image")):
                sample_count += int(batch["image"].shape[0])
            if not batch_shapes:
                for key in ("image", "boxes", "box_meta", "positive_boxes"):
                    value = batch.get(key)
                    if torch.is_tensor(value):
                        batch_shapes[key] = list(value.shape)
    if measured_start is not None:
        measured_sec = time() - measured_start

    result = {
        "jsonl": str(args.jsonl),
        "dataset_summary": dataset.summary(),
        "config": {
            "input_size": int(args.input_size),
            "candidate_k": int(args.candidate_k),
            "image_mean": dataset.summary().get("image_mean"),
            "image_std": dataset.summary().get("image_std"),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": args.prefetch_factor,
            "pin_memory": bool(args.pin_memory),
            "pin_memory_device": str(args.pin_memory_device),
            "precompute_sample_tensors": bool(args.precompute_sample_tensors),
            "image_tensor_cache_size": int(args.image_tensor_cache_size),
            "max_rows": args.max_rows,
            "warmup_steps": warmup_steps,
            "requested_steps": measured_steps,
        },
        "timing": {
            "dataset_init_sec": float(init_sec),
            "measured_sec": float(measured_sec),
            "measured_samples": int(sample_count),
            "samples_per_sec": float(sample_count / max(1e-9, measured_sec)),
            "batches_per_sec": float((sample_count / max(1, int(args.batch_size))) / max(1e-9, measured_sec)),
        },
        "batch_shapes": batch_shapes,
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
