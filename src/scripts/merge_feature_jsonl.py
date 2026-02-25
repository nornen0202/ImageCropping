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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge feature jsonl files by image_id")
    p.add_argument("--input_parquet", required=True, help="Provides image_id ordering")
    p.add_argument("--inputs", nargs="+", required=True, help="Input feature jsonl paths")
    p.add_argument("--output_jsonl", required=True)
    return p.parse_args()


def load_jsonl(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img_id = str(d.get("image_id", ""))
            if not img_id:
                continue
            out[img_id] = d
    return out


def main() -> None:
    args = parse_args()

    ids = [str(x) for x in pd.read_parquet(args.input_parquet, columns=["image_id"])["image_id"].tolist()]

    maps: List[Dict[str, dict]] = [load_jsonl(Path(p)) for p in args.inputs]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
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

    print(f"[done] merged={written} output={out_path}")


if __name__ == "__main__":
    main()
