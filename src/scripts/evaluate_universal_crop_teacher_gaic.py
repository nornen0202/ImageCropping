#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.gaic_benchmark import evaluate_scored_records, load_gaic_annotation_records  # noqa: E402
from scripts.train_universal_crop_teacher import move_batch_to_device, prediction_rows_for_batch  # noqa: E402
from universal_crop_teacher.data import UniversalCropTeacherDataset, collate_uctr_batch  # noqa: E402
from universal_crop_teacher.model import UniversalCropTeacherH, UniversalCropTeacherHConfig  # noqa: E402
from universal_crop_teacher.warehouse import gaic_record_to_warehouse_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UniversalCropTeacher-H on GAIC official candidate annotations.")
    parser.add_argument("--annotations_json", required=True, type=Path)
    parser.add_argument("--image_roots", nargs="*", type=Path, default=[])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--method_name", default="universal_crop_teacher_h")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> int:
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = UniversalCropTeacherHConfig.from_dict(ckpt.get("config", {}))
    model = UniversalCropTeacherH(config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(str(args.device))
    model.to(device).eval()

    max_images = int(args.max_images) if int(args.max_images) > 0 else None
    records = load_gaic_annotation_records(args.annotations_json.resolve(), image_roots=args.image_roots, max_images=max_images)
    rows = [
        gaic_record_to_warehouse_row(
            record,
            preserve_official_split=True,
        )
        for record in records
    ]
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

    prediction_rows: list[dict] = []
    scores_by_image_id: dict[str, list[float]] = {}
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
            pred_rows = prediction_rows_for_batch(outputs, batch, method=str(args.method_name))
            prediction_rows.extend(pred_rows)
            for source_row, pred_row in zip(batch["source_rows"], pred_rows):
                score_map = {str(cand.get("candidate_id", "")): float(cand.get("score", 0.0)) for cand in pred_row.get("candidates", [])}
                ordered_scores = [float(score_map.get(str(cand.get("candidate_id", "")), -1e9)) for cand in source_row.get("candidates", [])]
                scores_by_image_id[str(source_row.get("image_id", ""))] = ordered_scores

    metrics = evaluate_scored_records(records, scores_by_image_id, method_name=str(args.method_name))
    prediction_jsonl = output_dir / f"{args.method_name}_gaic_predictions.jsonl"
    summary_json = output_dir / "summary.json"
    write_jsonl(prediction_jsonl, prediction_rows)
    summary = {
        "method_name": str(args.method_name),
        "annotations_json": str(args.annotations_json.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "prediction_jsonl": str(prediction_jsonl),
        "summary_json": str(summary_json),
        "metrics": metrics,
        "record_count": len(records),
        "duration_sec": round(time.time() - start, 3),
        "device": str(device),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
