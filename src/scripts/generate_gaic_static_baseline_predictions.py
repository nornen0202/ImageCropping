#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet.bank import build_static_micro_bank
from mobilecropnet.eval_utils import annotation_candidates_for_record, box_iou_xyxy, load_label_crop_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate static GAIC crop baseline predictions.")
    parser.add_argument("--label_json", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument(
        "--policy",
        choices=["maxarea", "center_080", "thirds_or_center", "static_oracle_iou", "teacher_oracle"],
        default="maxarea",
    )
    parser.add_argument("--static_k", type=int, default=16)
    return parser


def _score_label(ann_like: dict[str, Any]) -> float:
    value = ann_like.get("label_score", ann_like.get("score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _choose_static(record: Any, *, policy: str, static_k: int) -> dict[str, Any]:
    image = record.image
    width = int(image["width"])
    height = int(image["height"])
    target_ar = str(image.get("target_ar", "FREE"))
    bank = build_static_micro_bank(image_width=width, image_height=height, target_ar=target_ar, k=static_k)
    if policy == "maxarea":
        return {"candidate_id": bank[0].candidate_id, "bbox_norm_xyxy": bank[0].bbox_norm_xyxy, "source": bank[0].source}
    if policy == "center_080":
        idx = min(2, len(bank) - 1)
        return {"candidate_id": bank[idx].candidate_id, "bbox_norm_xyxy": bank[idx].bbox_norm_xyxy, "source": bank[idx].source}
    if policy == "thirds_or_center":
        pool = [cand for cand in bank if cand.source in {"thirds", "center_crop"}]
        cand = pool[0] if pool else bank[0]
        return {"candidate_id": cand.candidate_id, "bbox_norm_xyxy": cand.bbox_norm_xyxy, "source": cand.source}

    label_candidates = annotation_candidates_for_record(record.annotations, width=width, height=height)
    positives = [cand for cand in label_candidates if int(cand.get("gt_flag", 0)) == 1]
    if policy == "teacher_oracle":
        best = max(positives, key=_score_label, default=max(label_candidates, key=_score_label))
        return {"candidate_id": best["candidate_id"], "bbox_norm_xyxy": best["bbox_norm_xyxy"], "source": "teacher_oracle"}

    positive_boxes = [cand["bbox_norm_xyxy"] for cand in positives]
    best_bank = max(
        bank,
        key=lambda cand: max((box_iou_xyxy(cand.bbox_norm_xyxy, box) for box in positive_boxes), default=0.0),
    )
    return {
        "candidate_id": best_bank.candidate_id,
        "bbox_norm_xyxy": best_bank.bbox_norm_xyxy,
        "source": "static_oracle_iou",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label_json = args.label_json
    records = load_label_crop_records(label_json)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            image = record.image
            chosen = _choose_static(record, policy=args.policy, static_k=args.static_k)
            row = {
                "model_name": f"static_{args.policy}",
                "image_id": str(image.get("sstk_image_id", image.get("id"))),
                "file_name": str(image.get("file_name")),
                "target_ar": str(image.get("target_ar", "FREE")),
                "bbox_norm_xyxy": chosen["bbox_norm_xyxy"],
                "score": 1.0,
                "candidate_id": chosen["candidate_id"],
                "source": chosen["source"],
                "policy": args.policy,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "policy": args.policy, "count": len(records)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
