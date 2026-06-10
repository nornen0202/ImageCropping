#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.score_mobilecropnet_v4_product_ar_candidates_with_public_cropper import (  # noqa: E402
    iter_label_rows,
    read_jsonl,
    summarize,
    write_json,
    write_jsonl,
)
from scripts.train_universal_crop_teacher import move_batch_to_device, prediction_rows_for_batch  # noqa: E402
from universal_crop_teacher.data import UniversalCropTeacherDataset, collate_uctr_batch  # noqa: E402
from universal_crop_teacher.model import UniversalCropTeacherH, UniversalCropTeacherHConfig  # noqa: E402


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("image_id", "")), str(row.get("target_ar", "FREE")))


def infer_dataset_name(rows: Sequence[dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return str(explicit).lower()
    if not rows:
        return "sstk"
    text = " ".join(str(row.get("image_path", "")).lower() for row in rows[:8])
    if "/gaic" in text:
        return "gaic"
    if "/testimages" in text or "/test_images" in text:
        return "test_images"
    return "sstk"


def candidate_rows_to_warehouse_rows(rows: Sequence[dict[str, Any]], *, dataset_name: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(dict(row))
    out: list[dict[str, Any]] = []
    for (image_id, target_ar), group in sorted(grouped.items()):
        first = group[0]
        candidates = []
        for row in group:
            candidates.append(
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "bbox_xyxy_norm": list(row.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0]),
                    "target_score": clamp(float(row.get("sstk_crop_utility_prob", row.get("crop_utility_prob", 0.5)) or 0.5)),
                    "label_confidence": clamp(float(row.get("sstk_candidate_weight", 1.0) or 1.0), 0.05, 1.0),
                    "gt_iou": clamp(float(row.get("sstk_crop_utility_prob", row.get("crop_utility_prob", 0.5)) or 0.5)),
                    "source": str(row.get("candidate_role", "")),
                    "label": str(row.get("sstk_training_bucket", "")),
                }
            )
        out.append(
            {
                "task_id": str(first.get("score_group_id") or f"{image_id}::{target_ar}"),
                "dataset": str(dataset_name).lower(),
                "split": str(first.get("official_split", "")).lower(),
                "image_id": image_id,
                "image_path": str(first.get("image_path", "")),
                "target_ar": target_ar,
                "candidates": candidates,
                "pairwise": [],
            }
        )
    return out


def score_candidate_rows_with_uctr(
    rows: Sequence[dict[str, Any]],
    *,
    checkpoint: Path,
    dataset_name: str,
    method_name: str,
    device: torch.device,
    max_candidates: int,
    image_size: int,
    crop_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[list[float], dict[str, Any]]:
    ckpt = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = UniversalCropTeacherHConfig.from_dict(ckpt.get("config", {}))
    model = UniversalCropTeacherH(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    warehouse_rows = candidate_rows_to_warehouse_rows(rows, dataset_name=dataset_name)
    dataset = UniversalCropTeacherDataset(
        warehouse_rows,
        max_candidates=int(max_candidates),
        image_size=int(image_size),
        crop_size=int(crop_size),
        include_images=True,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_uctr_batch,
    )
    score_map: dict[tuple[str, str, str], float] = {}
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
            for pred_row in prediction_rows_for_batch(outputs, batch, method=str(method_name)):
                image_id = str(pred_row.get("image_id", ""))
                target_ar = str(pred_row.get("target_ar", "FREE"))
                for candidate in pred_row.get("candidates", []) or []:
                    score_map[(image_id, target_ar, str(candidate.get("candidate_id", "")))] = float(candidate.get("score", 0.0))

    scores = [float(score_map.get((str(row.get("image_id", "")), str(row.get("target_ar", "FREE")), str(row.get("candidate_id", ""))), 0.0)) for row in rows]
    metadata = {
        "checkpoint": str(checkpoint),
        "dataset_name": str(dataset_name),
        "device": str(device),
        "image_size": int(image_size),
        "crop_size": int(crop_size),
        "max_candidates": int(max_candidates),
        "row_count": len(rows),
        "task_count": len(warehouse_rows),
        "score_map_count": len(score_map),
    }
    return scores, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Product AR candidate rows with a UniversalCropTeacher-H checkpoint.")
    parser.add_argument("--candidate_rows_jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--method_name", default="universal_crop_teacher_h_stage3_gate128")
    parser.add_argument("--score_source", default="uctr_stage3_gate128_product_ar")
    parser.add_argument("--high_score_threshold", type=float, default=0.85)
    parser.add_argument("--low_score_threshold", type=float, default=0.20)
    parser.add_argument("--contradiction_threshold", type=float, default=0.35)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.candidate_rows_jsonl, max_rows=int(args.max_rows))
    dataset_name = infer_dataset_name(rows, args.dataset_name)
    scores, scorer_metadata = score_candidate_rows_with_uctr(
        rows,
        checkpoint=args.checkpoint,
        dataset_name=dataset_name,
        method_name=str(args.method_name),
        device=torch.device(str(args.device)),
        max_candidates=int(args.max_candidates),
        image_size=int(args.image_size),
        crop_size=int(args.crop_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    written = write_jsonl(
        args.output_jsonl,
        iter_label_rows(
            rows,
            scores,
            method=str(args.method_name),
            score_source=str(args.score_source),
            high_threshold=float(args.high_score_threshold),
            low_threshold=float(args.low_score_threshold),
            contradiction_threshold=float(args.contradiction_threshold),
        ),
    )
    summary = {
        "status": "ok",
        "candidate_rows_jsonl": str(args.candidate_rows_jsonl),
        "checkpoint": str(args.checkpoint),
        "output_jsonl": str(args.output_jsonl),
        "method_name": str(args.method_name),
        "score_source": str(args.score_source),
        "written_rows": int(written),
        "high_score_threshold": float(args.high_score_threshold),
        "low_score_threshold": float(args.low_score_threshold),
        "contradiction_threshold": float(args.contradiction_threshold),
        "scorer_metadata": scorer_metadata,
        **summarize(args.output_jsonl),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
