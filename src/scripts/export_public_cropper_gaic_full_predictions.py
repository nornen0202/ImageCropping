#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = SRC_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import evaluate_public_croppers_gaic_v2 as public_cropper_gaic_eval  # noqa: E402

from mobilecropnet_v4.gaic_benchmark import evaluate_scored_records, load_gaic_annotation_records  # noqa: E402
from public_benchmark.ensemble import write_grouped_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full official GAIC candidate predictions for public croppers.")
    parser.add_argument("--method", required=True, choices=["gaic", "cgs", "cacnet"])
    parser.add_argument(
        "--annotations_json",
        type=Path,
        default=PROJECT_ROOT / "data/Publics/GAIC_v2/annotations_json/instances_test.json",
    )
    parser.add_argument(
        "--image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
        ],
    )
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", required=True, type=Path)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method_name", default="")
    return parser.parse_args()


def _scorer(name: str):
    if name == "gaic":
        return public_cropper_gaic_eval._score_gaic
    if name == "cgs":
        return public_cropper_gaic_eval._score_cgs
    if name == "cacnet":
        return public_cropper_gaic_eval._score_cacnet
    raise ValueError(f"Unsupported method: {name}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    device = torch.device(str(args.device))
    records = load_gaic_annotation_records(
        args.annotations_json.resolve(),
        image_roots=[Path(path).resolve() for path in args.image_roots],
        max_images=int(args.max_images) if int(args.max_images) > 0 else None,
    )
    scores_by_image, coverage_by_image, metadata = _scorer(str(args.method))(records, device=device)
    method_name = str(args.method_name).strip() or f"public_cropper_{args.method}"
    grouped_rows: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        scores = scores_by_image.get(image_id)
        candidates = list(record.get("candidates") or [])
        if scores is None or len(scores) != len(candidates):
            continue
        row_candidates: list[dict[str, Any]] = []
        for candidate_index, (candidate, score) in enumerate(zip(candidates, scores)):
            row_candidates.append(
                {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "annotation_id": candidate.get("annotation_id"),
                    "candidate_index": int(candidate_index),
                    "bbox_xyxy_norm": [float(v) for v in candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])[:4]],
                    "label": "gaic_mos_candidate",
                    "score": float(score),
                    "scores": {f"{args.method}_score": float(score)},
                    "source": "gaic_annotation_candidate",
                }
            )
        grouped_rows[("gaic", image_id, "FREE")] = row_candidates
    prediction_rows = write_grouped_predictions(
        args.output_jsonl.resolve(),
        grouped_rows,
        method_name=method_name,
        source=f"public_cropper_{args.method}",
    )
    result = evaluate_scored_records(records, scores_by_image, method_name=method_name, coverage_by_image_id=coverage_by_image)
    summary = {
        "format": "public_cropper_gaic_full_predictions_v1",
        "method": str(args.method),
        "method_name": method_name,
        "annotations_json": str(args.annotations_json.resolve()),
        "image_roots": [str(Path(path).resolve()) for path in args.image_roots],
        "output_jsonl": str(args.output_jsonl.resolve()),
        "prediction_rows": int(prediction_rows),
        "result": result,
        "metadata": metadata,
        "duration_sec": round(time.time() - start, 3),
    }
    _write_json(args.summary_json.resolve(), summary)
    print(json.dumps({"output_jsonl": summary["output_jsonl"], "summary_json": str(args.summary_json.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
