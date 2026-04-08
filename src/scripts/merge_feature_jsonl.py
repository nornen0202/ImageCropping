#!/usr/bin/env python3
"""
Merge per-component feature jsonl files by image_id.

Each input jsonl is expected to contain records with `image_id` and one or more
component keys (e.g., c2_seg, c3_pose, c5_geom).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from progress_utils import ProgressTracker, count_nonempty_lines, progress_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge feature jsonl files by image_id")
    p.add_argument("--input_parquet", required=True, help="Provides image_id ordering")
    p.add_argument("--inputs", nargs="+", required=True, help="Input feature jsonl paths")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=1000)
    p.add_argument("--progress_min_seconds", type=float, default=10.0)
    return p.parse_args()


def load_jsonl(
    path: Path,
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    total = count_nonempty_lines(path) if progress else None
    tracker = ProgressTracker(
        f"merge_feature_jsonl:load_jsonl:{path.name}",
        total=total,
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img_id = str(d.get("image_id", ""))
            if not img_id:
                tracker.update(idx)
                continue
            out[img_id] = d
            tracker.update(idx, extra=f"mapped={len(out)}")
    tracker.finish(len(out), extra=f"path={path.name}")
    return out


def main() -> None:
    args = parse_args()
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))

    progress_log(
        f"merge_feature_jsonl: start | input_parquet={args.input_parquet} | output_jsonl={args.output_jsonl}",
        enabled=progress_enabled,
    )
    ids = [str(x) for x in pd.read_parquet(args.input_parquet, columns=["image_id"])["image_id"].tolist()]

    maps: List[Dict[str, dict]] = [
        load_jsonl(
            Path(p),
            progress=progress_enabled,
            progress_every=progress_every,
            progress_min_seconds=progress_min_seconds,
        )
        for p in args.inputs
    ]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    tracker = ProgressTracker(
        "merge_feature_jsonl:write_output",
        total=len(ids),
        unit="images",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress_enabled,
    )
    with out_path.open("w", encoding="utf-8") as f:
        for img_id in ids:
            merged = {"image_id": img_id}
            for m in maps:
                rec = m.get(img_id)
                if not rec:
                    continue
                for k, v in rec.items():
                    if k == "image_id":
                        continue
                    merged[k] = v
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1
            tracker.update(written, extra=f"image_id={img_id}")
    tracker.finish(written, extra=f"output={out_path.name}")

    print(f"[done] merged={written} output={out_path}")


if __name__ == "__main__":
    main()
