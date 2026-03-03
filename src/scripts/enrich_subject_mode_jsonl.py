#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from routing.subject_mode_router import enrich_c2_topn, normalize_tags, route_subject_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich precompute jsonl with C2 Top-N + subject-mode routing.")
    p.add_argument("--input_feats_jsonl", required=True)
    p.add_argument("--input_filtered_parquet", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--c2_top_n", type=int, default=5)
    p.add_argument("--c2_union_top_m", type=int, default=3)
    p.add_argument("--allow_det_proxy", type=int, default=1)
    p.add_argument("--progress", type=int, default=1)
    return p.parse_args()


def _load_meta_map(parquet_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    needed = {"image_id", "width", "height"}
    if not needed.issubset(set(df.columns)):
        miss = sorted(needed - set(df.columns))
        raise ValueError(f"input parquet missing columns: {miss}")

    out: Dict[str, Dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        image_id = str(getattr(row, "image_id"))
        out[image_id] = {
            "width": int(getattr(row, "width")),
            "height": int(getattr(row, "height")),
            "tags": getattr(row, "tags", None),
            "super_cat": str(getattr(row, "super_cat", "")),
        }
    return out


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_feats_jsonl)
    input_parquet = Path(args.input_filtered_parquet)
    output_jsonl = Path(args.output_jsonl)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_feats_jsonl not found: {input_jsonl}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"input_filtered_parquet not found: {input_parquet}")

    meta_map = _load_meta_map(input_parquet)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    routed = 0
    missing_meta = 0
    mode_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()

    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        iterable = fin
        if int(args.progress) != 0:
            iterable = tqdm(fin, desc="subject-mode-enrich")
        for line in iterable:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if not image_id:
                continue

            meta = meta_map.get(image_id)
            if meta is None:
                missing_meta += 1
                width = int(rec.get("width", 1) or 1)
                height = int(rec.get("height", 1) or 1)
                tags = rec.get("tags", [])
                super_cat = ""
            else:
                width = int(meta["width"])
                height = int(meta["height"])
                tags = meta.get("tags", [])
                super_cat = str(meta.get("super_cat", ""))

            c2_payload = enrich_c2_topn(
                c2_seg=rec.get("c2_seg", []),
                c2_det=rec.get("c2_det", []),
                c3_pose=rec.get("c3_pose", []),
                width=width,
                height=height,
                top_n=max(1, int(args.c2_top_n)),
                union_top_m=max(1, int(args.c2_union_top_m)),
                allow_det_proxy=bool(int(args.allow_det_proxy)),
            )
            rec["c2_seg"] = c2_payload["instances"]
            rec["c2_primary_idx"] = c2_payload["c2_primary_idx"]
            rec["c2_union_box_xyxy"] = c2_payload["c2_union_box_xyxy"]
            rec["c2_topn"] = c2_payload["c2_topn"]
            rec["c2_stats"] = c2_payload["c2_stats"]

            tags_norm = normalize_tags(tags)
            routing = route_subject_mode(
                tags_norm=tags_norm,
                super_cat=super_cat,
                c3_pose=rec.get("c3_pose", []),
                c2_instances=rec.get("c2_seg", []),
                c2_union_box_xyxy=rec.get("c2_union_box_xyxy"),
                c2_primary_idx=int(rec.get("c2_primary_idx", -1)),
                width=width,
                height=height,
            )
            rec["routing"] = routing

            mode = str(routing.get("subject_mode", "unknown"))
            policy = str(routing.get("policy_id", "unknown"))
            mode_counter[mode] += 1
            policy_counter[policy] += 1
            routed += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[done] total={total} routed={routed} missing_meta={missing_meta} "
        f"output={output_jsonl}"
    )
    print(f"[done] subject_mode_counts={dict(mode_counter)}")
    print(f"[done] policy_counts={dict(policy_counter)}")


if __name__ == "__main__":
    main()

