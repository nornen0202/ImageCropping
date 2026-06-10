#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.train_gaic_visual_crop_ranker import extract_timm_roi_embeddings, save_feature_cache  # noqa: E402


def read_rows(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache timm ROI features for arbitrary GAIC candidate rows.")
    parser.add_argument("--candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=Path("data/Publics/GAIC_v2/images"))
    parser.add_argument("--feature_cache_dir", type=Path, required=True)
    parser.add_argument("--feature_cache_key", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--timm_model_name", default="mobilenetv4_conv_small.e3600_r256_in1k")
    parser.add_argument("--timm_weights", type=Path, default=Path("weights/hf/timm/mobilenetv4_conv_small.e3600_r256_in1k/model.safetensors"))
    parser.add_argument("--timm_roi_input_size", type=int, default=384)
    parser.add_argument("--progress_every", type=int, default=10000)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.candidate_eval_jsonl, max_rows=int(args.max_rows))
    backend = "timm_roi_" + str(args.timm_model_name).replace("/", "_").replace(".", "_") + "_raw"
    features = extract_timm_roi_embeddings(
        rows,
        split_name=str(args.split_name),
        image_root=Path(args.image_root),
        model_name=str(args.timm_model_name),
        weights_path=Path(args.timm_weights),
        device=str(args.device),
        input_size=int(args.timm_roi_input_size),
        progress_every=max(1, int(args.progress_every)),
    )
    save_feature_cache(
        Path(args.feature_cache_dir),
        backend=backend,
        cache_key=str(args.feature_cache_key),
        split_name=str(args.split_name),
        rows=rows,
        features=features,
    )
    summary = {
        "status": "ok",
        "candidate_eval_jsonl": str(args.candidate_eval_jsonl),
        "feature_cache_dir": str(args.feature_cache_dir),
        "feature_cache_key": str(args.feature_cache_key),
        "split_name": str(args.split_name),
        "backend": backend,
        "row_count": int(len(rows)),
        "feature_shape": list(features.shape),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
