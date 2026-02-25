#!/usr/bin/env python3
"""
Enrich existing C3 pose jsonl with face/gaze proxy fields derived from keypoints.

Input records are expected to include:
  {"image_id": ..., "c3_pose": [{"bbox": ..., "keypoints": ...}, ...]}

Output keeps all original fields and appends for each pose item:
  - face
  - headpose_gaze
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm


def _load_hw_map(parquet_path: Path) -> Dict[str, Tuple[int, int]]:
    df = pd.read_parquet(parquet_path, columns=["image_id", "width", "height"])
    out: Dict[str, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        out[str(row["image_id"])] = (int(row["width"]), int(row["height"]))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich C3 pose jsonl with face/gaze proxy fields")
    p.add_argument("--input_c3_jsonl", required=True)
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--output_jsonl", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "extract_features"))
    from c3_pose import enrich_pose_item_from_keypoints

    c3_path = Path(args.input_c3_jsonl)
    pq_path = Path(args.input_parquet)
    out_path = Path(args.output_jsonl)

    hw_map = _load_hw_map(pq_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    enriched = 0
    missing_wh = 0

    with c3_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="enrich_c3"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            img_id = str(rec.get("image_id", ""))
            pose_list = rec.get("c3_pose", [])

            wh = hw_map.get(img_id)
            if wh is None:
                missing_wh += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            w, h = wh
            if isinstance(pose_list, list) and pose_list:
                new_pose = []
                for item in pose_list:
                    if isinstance(item, dict):
                        new_pose.append(enrich_pose_item_from_keypoints(item, image_w=w, image_h=h))
                    else:
                        new_pose.append(item)
                rec["c3_pose"] = new_pose
                enriched += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] total={total} enriched_records={enriched} missing_hw={missing_wh} output={out_path}")


if __name__ == "__main__":
    main()
